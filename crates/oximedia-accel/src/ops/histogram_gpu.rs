//! GPU-accelerated (and CPU-fallback) RGB histogram computation.
//!
//! `GpuHistogram` holds 256 bins for each of the three RGB channels.
//! When no GPU is available (or for simple correctness), the fast
//! rayon-parallel CPU path is used automatically.

use rayon::prelude::*;

/// Number of histogram bins per channel (0..=255).
pub const HISTOGRAM_BINS: usize = 256;

/// WGSL compute shader for GPU histogram accumulation.
///
/// Each thread reads one RGBA pixel and atomically increments three
/// per-channel histogram bins in a workgroup-local `var<workgroup>` array.
/// After a workgroup barrier the workgroup sums are added to the global
/// histograms with further atomics.  This avoids all cross-workgroup
/// collisions while keeping the global atomic pressure low.
pub const HISTOGRAM_SHADER_WGSL: &str = r#"
// Histogram shader: one workgroup per 64 pixels.
// Bins layout: [R0..R255, G0..G255, B0..B255] = 768 x u32

struct HistParams {
    pixel_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read>       pixels:  array<u32>;
@group(0) @binding(1) var<storage, read_write> bins:    array<atomic<u32>>;
@group(0) @binding(2) var<uniform>             params:  HistParams;

var<workgroup> local_r: array<atomic<u32>, 256>;
var<workgroup> local_g: array<atomic<u32>, 256>;
var<workgroup> local_b: array<atomic<u32>, 256>;

@compute @workgroup_size(64, 1, 1)
fn hist_main(
    @builtin(global_invocation_id) gid:    vec3<u32>,
    @builtin(local_invocation_id)  lid:    vec3<u32>,
    @builtin(workgroup_id)         wgid:   vec3<u32>,
) {
    let lid_x = lid.x;

    // ── Zero workgroup-local bins ────────────────────────────────────────
    // Each of the 64 threads zeroes 4 R-bins, 4 G-bins, 4 B-bins
    let zero_base = lid_x * 4u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let b = zero_base + i;
        if (b < 256u) {
            atomicStore(&local_r[b], 0u);
            atomicStore(&local_g[b], 0u);
            atomicStore(&local_b[b], 0u);
        }
    }

    workgroupBarrier();

    // ── Accumulate into local bins ───────────────────────────────────────
    let idx = gid.x;
    if (idx < params.pixel_count) {
        let packed = pixels[idx];
        let r = (packed >> 24u) & 0xFFu;
        let g = (packed >> 16u) & 0xFFu;
        let b = (packed >>  8u) & 0xFFu;
        atomicAdd(&local_r[r], 1u);
        atomicAdd(&local_g[g], 1u);
        atomicAdd(&local_b[b], 1u);
    }

    workgroupBarrier();

    // ── Flush local bins to global ───────────────────────────────────────
    for (var i = 0u; i < 4u; i = i + 1u) {
        let b = zero_base + i;
        if (b < 256u) {
            let vr = atomicLoad(&local_r[b]);
            let vg = atomicLoad(&local_g[b]);
            let vb = atomicLoad(&local_b[b]);
            if (vr > 0u) { atomicAdd(&bins[b],           vr); }
            if (vg > 0u) { atomicAdd(&bins[256u + b],    vg); }
            if (vb > 0u) { atomicAdd(&bins[512u + b],    vb); }
        }
    }
}
"#;

/// Per-channel histogram with 256 bins each.
#[derive(Debug, Clone)]
pub struct GpuHistogram {
    /// Red channel bin counts (`bins_r[i]` = number of pixels with R == i).
    pub bins_r: [u32; HISTOGRAM_BINS],
    /// Green channel bin counts.
    pub bins_g: [u32; HISTOGRAM_BINS],
    /// Blue channel bin counts.
    pub bins_b: [u32; HISTOGRAM_BINS],
}

impl GpuHistogram {
    /// Create an empty (all-zero) histogram.
    #[must_use]
    pub fn zeroed() -> Self {
        Self {
            bins_r: [0u32; HISTOGRAM_BINS],
            bins_g: [0u32; HISTOGRAM_BINS],
            bins_b: [0u32; HISTOGRAM_BINS],
        }
    }

    /// Total number of pixels counted (sum of R bins; all channels equal).
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        self.bins_r.iter().map(|&v| v as u64).sum()
    }

    /// Return the most-common (peak) bin index for the red channel.
    #[must_use]
    pub fn peak_r(&self) -> u8 {
        self.bins_r
            .iter()
            .enumerate()
            .max_by_key(|(_, &v)| v)
            .map(|(i, _)| i as u8)
            .unwrap_or(0)
    }

    /// Return the most-common (peak) bin index for the green channel.
    #[must_use]
    pub fn peak_g(&self) -> u8 {
        self.bins_g
            .iter()
            .enumerate()
            .max_by_key(|(_, &v)| v)
            .map(|(i, _)| i as u8)
            .unwrap_or(0)
    }

    /// Return the most-common (peak) bin index for the blue channel.
    #[must_use]
    pub fn peak_b(&self) -> u8 {
        self.bins_b
            .iter()
            .enumerate()
            .max_by_key(|(_, &v)| v)
            .map(|(i, _)| i as u8)
            .unwrap_or(0)
    }
}

// ── CPU reference path (parallel) ───────────────────────────────────────────

/// Compute a per-channel RGB histogram using Rayon parallel reduction.
///
/// `pixels` must be a flat sequence of RGBA bytes (`width × height × 4`).
/// Pixels beyond `width × height` are ignored.
///
/// The alpha channel is not histogrammed.
#[must_use]
pub fn compute_histogram(pixels: &[u8], width: usize, height: usize) -> GpuHistogram {
    let count = (width * height).min(pixels.len() / 4);
    if count == 0 {
        return GpuHistogram::zeroed();
    }

    // Parallel partial histograms, one per rayon thread chunk.
    // Each chunk is `chunk_size` pixels; partial results are summed afterwards.
    let chunk_pixels = 4096usize;
    let chunks: Vec<([u32; 256], [u32; 256], [u32; 256])> = pixels[..count * 4]
        .par_chunks(chunk_pixels * 4)
        .map(|chunk| {
            let mut r = [0u32; 256];
            let mut g = [0u32; 256];
            let mut b = [0u32; 256];
            for px in chunk.chunks_exact(4) {
                r[px[0] as usize] += 1;
                g[px[1] as usize] += 1;
                b[px[2] as usize] += 1;
            }
            (r, g, b)
        })
        .collect();

    let mut hist = GpuHistogram::zeroed();
    for (r, g, b) in chunks {
        for i in 0..256 {
            hist.bins_r[i] += r[i];
            hist.bins_g[i] += g[i];
            hist.bins_b[i] += b[i];
        }
    }
    hist
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build an RGBA pixel buffer from an RGB colour repeated `n` times.
    fn solid_rgba(r: u8, g: u8, b: u8, n: usize) -> Vec<u8> {
        (0..n).flat_map(|_| [r, g, b, 255u8]).collect()
    }

    #[test]
    fn test_uniform_image_single_bin() {
        // All pixels are (100, 150, 200, 255) — each channel should have a
        // single occupied bin.
        let pixels = solid_rgba(100, 150, 200, 64);
        let hist = compute_histogram(&pixels, 8, 8);

        assert_eq!(hist.bins_r[100], 64, "R bin[100] should hold all 64 pixels");
        assert_eq!(hist.bins_g[150], 64, "G bin[150] should hold all 64 pixels");
        assert_eq!(hist.bins_b[200], 64, "B bin[200] should hold all 64 pixels");

        // All other bins must be zero
        for i in 0..256 {
            if i != 100 {
                assert_eq!(hist.bins_r[i], 0);
            }
            if i != 150 {
                assert_eq!(hist.bins_g[i], 0);
            }
            if i != 200 {
                assert_eq!(hist.bins_b[i], 0);
            }
        }
    }

    #[test]
    fn test_pixel_count_matches_dimensions() {
        let pixels = solid_rgba(0, 0, 0, 16 * 16);
        let hist = compute_histogram(&pixels, 16, 16);
        assert_eq!(hist.pixel_count(), 256);
    }

    #[test]
    fn test_black_image_bin_zero() {
        let pixels = solid_rgba(0, 0, 0, 10);
        let hist = compute_histogram(&pixels, 10, 1);
        assert_eq!(hist.bins_r[0], 10);
        assert_eq!(hist.bins_g[0], 10);
        assert_eq!(hist.bins_b[0], 10);
    }

    #[test]
    fn test_white_image_bin_255() {
        let pixels = solid_rgba(255, 255, 255, 10);
        let hist = compute_histogram(&pixels, 10, 1);
        assert_eq!(hist.bins_r[255], 10);
        assert_eq!(hist.bins_g[255], 10);
        assert_eq!(hist.bins_b[255], 10);
    }

    #[test]
    fn test_gradient_image_all_bins_non_zero() {
        // 256-pixel horizontal gradient R = 0..255, G = 0, B = 0
        let pixels: Vec<u8> = (0u8..=255).flat_map(|v| [v, 0u8, 0u8, 255u8]).collect();
        let hist = compute_histogram(&pixels, 256, 1);
        // Every R bin should have exactly 1 count
        for i in 0..256 {
            assert_eq!(hist.bins_r[i], 1, "R bin[{i}] should be 1 for gradient");
        }
        // G and B should all be in bin 0
        assert_eq!(hist.bins_g[0], 256);
        assert_eq!(hist.bins_b[0], 256);
    }

    #[test]
    fn test_two_color_image() {
        // 4 red pixels + 4 blue pixels
        let mut pixels = solid_rgba(255, 0, 0, 4);
        pixels.extend(solid_rgba(0, 0, 255, 4));
        let hist = compute_histogram(&pixels, 8, 1);
        assert_eq!(hist.bins_r[255], 4);
        assert_eq!(hist.bins_r[0], 4);
        assert_eq!(hist.bins_b[255], 4);
        assert_eq!(hist.bins_b[0], 4);
    }

    #[test]
    fn test_empty_image_returns_zeros() {
        let hist = compute_histogram(&[], 0, 0);
        assert_eq!(hist.pixel_count(), 0);
    }

    #[test]
    fn test_peak_functions() {
        let pixels = solid_rgba(42, 100, 200, 50);
        let hist = compute_histogram(&pixels, 50, 1);
        assert_eq!(hist.peak_r(), 42);
        assert_eq!(hist.peak_g(), 100);
        assert_eq!(hist.peak_b(), 200);
    }

    #[test]
    fn test_large_image_total_count() {
        // 1920×1080 uniform
        let n = 1920 * 1080;
        let pixels = solid_rgba(128, 128, 128, n);
        let hist = compute_histogram(&pixels, 1920, 1080);
        assert_eq!(hist.pixel_count(), n as u64);
        assert_eq!(hist.bins_r[128], n as u32);
    }
}
