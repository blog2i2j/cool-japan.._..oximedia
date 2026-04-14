//! Histogram equalization for image contrast enhancement.
//!
//! Provides four distinct equalization strategies:
//!
//! - **Global equalization** — classic single-pass CDF remapping over the whole image.
//! - **CLAHE** (Contrast Limited Adaptive Histogram Equalization) — divides the image
//!   into tiles, limits each tile's histogram with a clip limit to prevent over-amplification,
//!   and bilinearly interpolates results from four neighbouring tile CLTs.
//! - **Per-channel equalization** — applies global equalization independently to each
//!   channel of an interleaved RGB image, preserving hue loosely.
//! - **Luminance-only equalization** — converts RGB → YCbCr, equalises Y, and converts back;
//!   hue and saturation are preserved exactly.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]

use crate::error::{ImageError, ImageResult};

// ─── Constants ────────────────────────────────────────────────────────────────

/// Histogram bin count for standard 8-bit data.
const BINS: usize = 256;

// ─── CDF / mapping helpers ─────────────────────────────────────────────────────

/// Build a cumulative distribution function from a histogram.
fn build_cdf(hist: &[u64; BINS]) -> [u64; BINS] {
    let mut cdf = [0u64; BINS];
    cdf[0] = hist[0];
    for i in 1..BINS {
        cdf[i] = cdf[i - 1] + hist[i];
    }
    cdf
}

/// Build an 8-bit lookup table that maps input intensities to equalised values.
///
/// `total_pixels` is the number of pixels used to fill `hist`.
fn build_lut(hist: &[u64; BINS], total_pixels: u64) -> [u8; BINS] {
    let cdf = build_cdf(hist);
    let cdf_min = *cdf.iter().find(|&&v| v > 0).unwrap_or(&0);
    let denom = total_pixels.saturating_sub(cdf_min);
    let mut lut = [0u8; BINS];
    for i in 0..BINS {
        if denom == 0 {
            lut[i] = i as u8;
        } else {
            let numerator = cdf[i].saturating_sub(cdf_min);
            let scaled = (numerator as f64 / denom as f64) * 255.0;
            lut[i] = scaled.round().clamp(0.0, 255.0) as u8;
        }
    }
    lut
}

// ─── Global equalization ──────────────────────────────────────────────────────

/// Apply global histogram equalization to a single-channel (grayscale) image.
///
/// Modifies `pixels` in place.  Returns an error only if `pixels` is empty.
///
/// # Algorithm
///
/// 1. Build the intensity histogram.
/// 2. Compute the CDF.
/// 3. Remap each pixel via the normalised CDF as a lookup table.
pub fn equalize_global(pixels: &mut [u8]) -> ImageResult<()> {
    if pixels.is_empty() {
        return Err(ImageError::invalid_format(
            "equalize_global: empty pixel buffer",
        ));
    }

    let mut hist = [0u64; BINS];
    for &p in pixels.iter() {
        hist[p as usize] += 1;
    }

    let lut = build_lut(&hist, pixels.len() as u64);

    for p in pixels.iter_mut() {
        *p = lut[*p as usize];
    }
    Ok(())
}

// ─── CLAHE ────────────────────────────────────────────────────────────────────

/// Configuration for CLAHE.
#[derive(Debug, Clone)]
pub struct ClaheConfig {
    /// Number of tiles in the horizontal direction (must be ≥ 1).
    pub tiles_x: usize,
    /// Number of tiles in the vertical direction (must be ≥ 1).
    pub tiles_y: usize,
    /// Clip limit as a fraction of the uniform histogram height.
    ///
    /// A value of 1.0 means no redistribution (full clip to uniform distribution).
    /// Typical range: 2.0 – 4.0.  Use `f64::INFINITY` to disable clipping.
    pub clip_limit: f64,
}

impl Default for ClaheConfig {
    fn default() -> Self {
        Self {
            tiles_x: 8,
            tiles_y: 8,
            clip_limit: 2.0,
        }
    }
}

impl ClaheConfig {
    /// Create a new CLAHE configuration.
    #[must_use]
    pub fn new(tiles_x: usize, tiles_y: usize, clip_limit: f64) -> Self {
        Self {
            tiles_x,
            tiles_y,
            clip_limit,
        }
    }
}

/// Clip a tile histogram so that no bin exceeds `clip_count`.
///
/// Excess counts are redistributed uniformly across all bins.
fn clip_histogram(hist: &mut [u64; BINS], clip_count: u64) {
    let mut excess: u64 = 0;
    for bin in hist.iter_mut() {
        if *bin > clip_count {
            excess += *bin - clip_count;
            *bin = clip_count;
        }
    }
    // Redistribute excess uniformly
    let add_per_bin = excess / BINS as u64;
    let remainder = (excess % BINS as u64) as usize;
    for bin in hist.iter_mut() {
        *bin += add_per_bin;
    }
    // Distribute remaining counts one per bin from the start
    for bin in hist.iter_mut().take(remainder) {
        *bin += 1;
    }
}

/// Apply CLAHE to a single-channel (grayscale) image.
///
/// # Arguments
///
/// * `pixels` — mutable grayscale pixel buffer, modified in place.
/// * `width` — image width in pixels.
/// * `height` — image height in pixels.
/// * `cfg` — CLAHE parameters.
///
/// # Errors
///
/// Returns an error if dimensions don't match the buffer or if tile counts are zero.
pub fn equalize_clahe(
    pixels: &mut [u8],
    width: usize,
    height: usize,
    cfg: &ClaheConfig,
) -> ImageResult<()> {
    if pixels.len() != width * height {
        return Err(ImageError::invalid_format(format!(
            "CLAHE: buffer {} != {}×{}",
            pixels.len(),
            width,
            height
        )));
    }
    if cfg.tiles_x == 0 || cfg.tiles_y == 0 {
        return Err(ImageError::invalid_format(
            "CLAHE: tiles_x and tiles_y must be >= 1",
        ));
    }

    let tx = cfg.tiles_x;
    let ty = cfg.tiles_y;

    // Tile dimensions (may not divide evenly; last tile absorbs remainder)
    let tile_w = (width + tx - 1) / tx;
    let tile_h = (height + ty - 1) / ty;

    // Build per-tile LUTs ─────────────────────────────────────────────────────
    //
    // luts[ry][rx] is the 256-entry lookup table for tile (rx, ry).
    let mut luts: Vec<Vec<[u8; BINS]>> = vec![vec![[0u8; BINS]; tx]; ty];

    for ry in 0..ty {
        for rx in 0..tx {
            let x0 = rx * tile_w;
            let y0 = ry * tile_h;
            let x1 = (x0 + tile_w).min(width);
            let y1 = (y0 + tile_h).min(height);

            // Accumulate histogram for this tile
            let mut hist = [0u64; BINS];
            for y in y0..y1 {
                for x in x0..x1 {
                    hist[pixels[y * width + x] as usize] += 1;
                }
            }

            let tile_pixels = ((x1 - x0) * (y1 - y0)) as u64;

            // Clip histogram
            if cfg.clip_limit.is_finite() && cfg.clip_limit > 0.0 {
                // clip_limit is relative to the uniform distribution height
                let uniform_height = tile_pixels / BINS as u64;
                let clip_count = ((cfg.clip_limit * uniform_height as f64).round() as u64).max(1);
                clip_histogram(&mut hist, clip_count);
            }

            luts[ry][rx] = build_lut(&hist, tile_pixels);
        }
    }

    // Bilinear interpolation of tile LUTs ─────────────────────────────────────
    //
    // For each pixel we find the four surrounding tile centres and interpolate.
    let tile_cx: Vec<f64> = (0..tx)
        .map(|rx| {
            let x0 = rx * tile_w;
            let x1 = (x0 + tile_w).min(width);
            (x0 + x1) as f64 / 2.0
        })
        .collect();
    let tile_cy: Vec<f64> = (0..ty)
        .map(|ry| {
            let y0 = ry * tile_h;
            let y1 = (y0 + tile_h).min(height);
            (y0 + y1) as f64 / 2.0
        })
        .collect();

    let out: Vec<u8> = (0..height)
        .flat_map(|y| {
            let py = y as f64;
            let luts_ref = &luts;
            let tile_cx_ref = &tile_cx;
            let tile_cy_ref = &tile_cy;
            let pixels_ref = &*pixels;
            (0..width).map(move |x| {
                let px = x as f64;
                let val = pixels_ref[y * width + x];

                // Find the two tile-centre indices bracketing py and px
                let (rx0, rx1, fx) = bracket(tile_cx_ref, px);
                let (ry0, ry1, fy) = bracket(tile_cy_ref, py);

                // Retrieve the four tile LUT values
                let q00 = luts_ref[ry0][rx0][val as usize] as f64;
                let q10 = luts_ref[ry0][rx1][val as usize] as f64;
                let q01 = luts_ref[ry1][rx0][val as usize] as f64;
                let q11 = luts_ref[ry1][rx1][val as usize] as f64;

                // Bilinear interpolation
                let top = q00 * (1.0 - fx) + q10 * fx;
                let bot = q01 * (1.0 - fx) + q11 * fx;
                let interp = top * (1.0 - fy) + bot * fy;
                interp.round().clamp(0.0, 255.0) as u8
            })
        })
        .collect();

    pixels.copy_from_slice(&out);
    Ok(())
}

/// Return `(i0, i1, fraction)` where `centres[i0]` and `centres[i1]` bracket `pos`.
///
/// `fraction` is the interpolation weight toward `i1` (`0.0` → fully at `i0`).
fn bracket(centres: &[f64], pos: f64) -> (usize, usize, f64) {
    let n = centres.len();
    if n == 1 {
        return (0, 0, 0.0);
    }
    // Find the index of the last centre <= pos
    let i = centres.iter().rposition(|&c| c <= pos).unwrap_or(0);
    let i0 = i;
    let i1 = (i + 1).min(n - 1);
    if i0 == i1 {
        return (i0, i1, 0.0);
    }
    let span = centres[i1] - centres[i0];
    let frac = if span > f64::EPSILON {
        (pos - centres[i0]) / span
    } else {
        0.0
    };
    (i0, i1, frac.clamp(0.0, 1.0))
}

// ─── Per-channel equalization ─────────────────────────────────────────────────

/// Apply global histogram equalization independently to each RGB channel.
///
/// `pixels` is an interleaved RGB buffer (R, G, B, R, G, B, …).
///
/// # Errors
///
/// Returns an error if `pixels.len()` is not divisible by 3.
pub fn equalize_per_channel_rgb(pixels: &mut [u8]) -> ImageResult<()> {
    if pixels.len() % 3 != 0 {
        return Err(ImageError::invalid_format(
            "equalize_per_channel_rgb: buffer length must be divisible by 3",
        ));
    }
    if pixels.is_empty() {
        return Ok(());
    }

    let n = pixels.len() / 3;

    // Build histograms per channel
    let mut hist_r = [0u64; BINS];
    let mut hist_g = [0u64; BINS];
    let mut hist_b = [0u64; BINS];
    for chunk in pixels.chunks_exact(3) {
        hist_r[chunk[0] as usize] += 1;
        hist_g[chunk[1] as usize] += 1;
        hist_b[chunk[2] as usize] += 1;
    }

    let lut_r = build_lut(&hist_r, n as u64);
    let lut_g = build_lut(&hist_g, n as u64);
    let lut_b = build_lut(&hist_b, n as u64);

    for chunk in pixels.chunks_exact_mut(3) {
        chunk[0] = lut_r[chunk[0] as usize];
        chunk[1] = lut_g[chunk[1] as usize];
        chunk[2] = lut_b[chunk[2] as usize];
    }
    Ok(())
}

// ─── Luminance-only equalization ──────────────────────────────────────────────

/// RGB ↔ YCbCr conversion constants (BT.601, full-swing).
const KR: f32 = 0.299;
const KG: f32 = 0.587;
const KB: f32 = 0.114;

/// Convert a single `(r, g, b)` triple in `[0, 255]` to `(Y, Cb, Cr)`.
#[inline]
fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let r = r as f32;
    let g = g as f32;
    let b = b as f32;
    let y = KR * r + KG * g + KB * b;
    let cb = 128.0 + (b - y) / (2.0 * (1.0 - KB));
    let cr = 128.0 + (r - y) / (2.0 * (1.0 - KR));
    (y, cb, cr)
}

/// Convert `(Y, Cb, Cr)` back to `(R, G, B)` (clamped to `[0, 255]`).
#[inline]
fn ycbcr_to_rgb(y: f32, cb: f32, cr: f32) -> (u8, u8, u8) {
    let cb = cb - 128.0;
    let cr = cr - 128.0;
    let r = y + 2.0 * (1.0 - KR) * cr;
    let g = y - (KB / KG) * 2.0 * (1.0 - KB) * cb - (KR / KG) * 2.0 * (1.0 - KR) * cr;
    let b = y + 2.0 * (1.0 - KB) * cb;
    let clamp = |v: f32| v.round().clamp(0.0, 255.0) as u8;
    (clamp(r), clamp(g), clamp(b))
}

/// Apply histogram equalization only to the luminance (Y) channel of an RGB image.
///
/// Hue and saturation are preserved by converting through BT.601 YCbCr,
/// equalising Y, and converting back.
///
/// `pixels` is an interleaved RGB buffer (R, G, B, R, G, B, …).
///
/// # Errors
///
/// Returns an error if `pixels.len()` is not divisible by 3.
pub fn equalize_luminance(pixels: &mut [u8]) -> ImageResult<()> {
    if pixels.len() % 3 != 0 {
        return Err(ImageError::invalid_format(
            "equalize_luminance: buffer length must be divisible by 3",
        ));
    }
    if pixels.is_empty() {
        return Ok(());
    }

    let n = pixels.len() / 3;

    // Extract Y values and build histogram
    let mut y_values: Vec<u8> = Vec::with_capacity(n);
    let mut hist = [0u64; BINS];
    for chunk in pixels.chunks_exact(3) {
        let (y, _cb, _cr) = rgb_to_ycbcr(chunk[0], chunk[1], chunk[2]);
        let y_u8 = y.round().clamp(0.0, 255.0) as u8;
        y_values.push(y_u8);
        hist[y_u8 as usize] += 1;
    }

    let lut = build_lut(&hist, n as u64);

    for (chunk, &y_orig) in pixels.chunks_exact_mut(3).zip(y_values.iter()) {
        let (_, cb, cr) = rgb_to_ycbcr(chunk[0], chunk[1], chunk[2]);
        let y_eq = lut[y_orig as usize] as f32;
        let (r2, g2, b2) = ycbcr_to_rgb(y_eq, cb, cr);
        chunk[0] = r2;
        chunk[1] = g2;
        chunk[2] = b2;
    }
    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a gradient image: pixels uniformly spaced 0..255.
    fn gradient_gray(n: usize) -> Vec<u8> {
        (0..n).map(|i| ((i * 255) / (n - 1)) as u8).collect()
    }

    #[test]
    fn test_equalize_global_empty_error() {
        let mut buf: Vec<u8> = vec![];
        assert!(equalize_global(&mut buf).is_err());
    }

    #[test]
    fn test_equalize_global_uniform_image() {
        // A uniform image (all same value) should map to itself or 255.
        let mut buf = vec![128u8; 100];
        equalize_global(&mut buf).unwrap();
        // All pixels are the same value; the output should be uniform too.
        let first = buf[0];
        assert!(buf.iter().all(|&v| v == first));
    }

    #[test]
    fn test_equalize_global_increases_contrast() {
        // A dark image concentrated in [0, 64] should spread to [0, 255].
        let mut buf: Vec<u8> = (0..128).map(|i| (i % 64) as u8).collect();
        equalize_global(&mut buf).unwrap();
        let max = *buf.iter().max().unwrap();
        let min = *buf.iter().min().unwrap();
        assert!(
            max > 200,
            "After equalization max should be close to 255, got {max}"
        );
        assert!(
            min < 50,
            "After equalization min should be close to 0, got {min}"
        );
    }

    #[test]
    fn test_equalize_global_full_gradient() {
        let mut buf = gradient_gray(256);
        equalize_global(&mut buf).unwrap();
        assert_eq!(buf.len(), 256);
        // The output should still be monotonically non-decreasing.
        let is_nondecreasing = buf.windows(2).all(|w| w[0] <= w[1]);
        assert!(
            is_nondecreasing,
            "Output should remain monotonically non-decreasing"
        );
    }

    #[test]
    fn test_clahe_basic() {
        let width = 64usize;
        let height = 64usize;
        let mut pixels: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();
        let cfg = ClaheConfig::default();
        equalize_clahe(&mut pixels, width, height, &cfg).unwrap();
        assert_eq!(pixels.len(), width * height);
        // u8 values are always in [0, 255] by type
    }

    #[test]
    fn test_clahe_size_mismatch_error() {
        let mut buf = vec![0u8; 10];
        let cfg = ClaheConfig::default();
        assert!(equalize_clahe(&mut buf, 4, 4, &cfg).is_err());
    }

    #[test]
    fn test_clahe_zero_tiles_error() {
        let mut buf = vec![0u8; 16];
        let cfg = ClaheConfig::new(0, 4, 2.0);
        assert!(equalize_clahe(&mut buf, 4, 4, &cfg).is_err());
    }

    #[test]
    fn test_per_channel_rgb_wrong_length_error() {
        let mut buf = vec![0u8; 10]; // not divisible by 3
        assert!(equalize_per_channel_rgb(&mut buf).is_err());
    }

    #[test]
    fn test_per_channel_rgb_preserves_length() {
        let mut buf: Vec<u8> = (0..300).map(|i| (i % 256) as u8).collect();
        let original_len = buf.len();
        equalize_per_channel_rgb(&mut buf).unwrap();
        assert_eq!(buf.len(), original_len);
    }

    #[test]
    fn test_luminance_only_hue_stability() {
        // A red image: equalising luminance should not change the dominant hue.
        // All pixels are the same pure red, so equalization should leave them red.
        let mut buf: Vec<u8> = vec![200, 0, 0].repeat(100);
        equalize_luminance(&mut buf).unwrap();
        // R should remain higher than G and B for all pixels.
        for chunk in buf.chunks_exact(3) {
            assert!(
                chunk[0] >= chunk[1] && chunk[0] >= chunk[2],
                "Red channel should dominate: {:?}",
                chunk
            );
        }
    }

    #[test]
    fn test_luminance_wrong_length_error() {
        let mut buf = vec![0u8; 7]; // not divisible by 3
        assert!(equalize_luminance(&mut buf).is_err());
    }

    #[test]
    fn test_clahe_single_tile_matches_global() {
        // With a single tile and no clipping CLAHE should approximate global equalization.
        let mut global_buf: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let mut clahe_buf = global_buf.clone();

        equalize_global(&mut global_buf).unwrap();
        let cfg = ClaheConfig::new(1, 1, f64::INFINITY); // no clipping
        equalize_clahe(&mut clahe_buf, 16, 16, &cfg).unwrap();

        // Results should be close (within rounding from bilinear interpolation).
        let max_diff = global_buf
            .iter()
            .zip(clahe_buf.iter())
            .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs())
            .max()
            .unwrap_or(0);
        // Allow up to 2 levels of rounding difference.
        assert!(max_diff <= 2, "CLAHE 1×1 vs global max diff = {max_diff}");
    }

    #[test]
    fn test_build_lut_monotone() {
        // The lookup table should be monotonically non-decreasing.
        let mut hist = [0u64; BINS];
        for i in 0..BINS {
            hist[i] = (i + 1) as u64;
        }
        let total: u64 = hist.iter().sum();
        let lut = build_lut(&hist, total);
        assert!(
            lut.windows(2).all(|w| w[0] <= w[1]),
            "LUT should be monotone"
        );
    }
}
