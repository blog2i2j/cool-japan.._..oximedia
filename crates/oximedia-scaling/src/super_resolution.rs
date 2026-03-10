//! Super-resolution upscaling algorithms.
//!
//! Provides edge-preserving and frequency-domain super-resolution methods.

use std::f32::consts::PI;

/// Super-resolution algorithm selection.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SrAlgorithm {
    /// Simple bicubic upscaling.
    Bicubic,
    /// Lanczos-based upscaling.
    Lanczos,
    /// Edge-preserving upscaling using Sobel detection.
    EdgePreserving,
    /// Frequency-domain upscaling (zero-padding DFT).
    Frequency,
    /// Neural-network stub (placeholder).
    NeuralStub,
}

/// Configuration for super-resolution.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SrConfig {
    /// Algorithm to use.
    pub algorithm: SrAlgorithm,
    /// Integer scale factor (e.g. 2 = double resolution).
    pub scale: u32,
    /// Sharpening strength applied post-upscale (0.0–1.0).
    pub sharpening_strength: f32,
}

impl Default for SrConfig {
    fn default() -> Self {
        Self {
            algorithm: SrAlgorithm::EdgePreserving,
            scale: 2,
            sharpening_strength: 0.3,
        }
    }
}

/// Edge-preserving upscaler using edge-directed interpolation (EDI) with
/// Sobel guidance and unsharp masking.
pub struct EdgePreservingUpscaler;

impl EdgePreservingUpscaler {
    /// Upscale a single-channel image by an integer `scale` factor using a
    /// classical edge-guided super-resolution pipeline:
    ///
    /// 1. **Bicubic upscale** to target size.
    /// 2. **Edge-directed interpolation (EDI)**: at each pixel compute the
    ///    horizontal gradient magnitude |Gx| and vertical magnitude |Gy| from
    ///    Sobel kernels on the upscaled image.  Where |Gx| > |Gy| (strong
    ///    horizontal gradient → vertical edge) prefer horizontal-neighbour
    ///    averaging to preserve the edge.  At low-gradient pixels the bicubic
    ///    value is kept unchanged.
    /// 3. **Unsharp mask** (radius 1, strength 0.5) to recover edge sharpness
    ///    lost during upscaling.
    ///
    /// The `NeuralStub` variant in `SrAlgorithm` falls through to this
    /// implementation since no inference engine is available at compile time.
    #[must_use]
    #[allow(dead_code)]
    pub fn upscale(src: &[f32], src_w: u32, src_h: u32, scale: u32) -> Vec<f32> {
        if src_w == 0 || src_h == 0 || scale == 0 {
            return Vec::new();
        }
        let dst_w = src_w * scale;
        let dst_h = src_h * scale;
        let sw = src_w as usize;
        let sh = src_h as usize;
        let dw = dst_w as usize;
        let dh = dst_h as usize;

        // ------------------------------------------------------------------ //
        // Pass 1: bicubic upscale as base.
        // ------------------------------------------------------------------ //
        let bicubic = bicubic_upscale(src, sw, sh, dw, dh);

        // ------------------------------------------------------------------ //
        // Pass 2: edge-directed interpolation on the bicubic output.
        // ------------------------------------------------------------------ //
        let edi = edge_directed_interpolation(&bicubic, dw, dh);

        // ------------------------------------------------------------------ //
        // Pass 3: unsharp mask to recover sharpness.
        // ------------------------------------------------------------------ //
        unsharp_mask(&edi, dw, dh, 0.5)
    }
}

/// Edge-directed interpolation (EDI) on an already-upscaled image.
///
/// For each pixel computes the horizontal (|Gx|) and vertical (|Gy|) Sobel
/// gradient magnitudes.
///
/// * **Strong vertical edge** (|Gx| > |Gy|): replace the pixel with the
///   average of its left and right horizontal neighbours.  This smooths along
///   the edge direction while preserving the edge itself.
/// * **Otherwise**: keep the bicubic value as-is.
///
/// Interior pixels only — border pixels are copied unchanged.
#[allow(dead_code)]
fn edge_directed_interpolation(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = src.to_vec();

    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            // 3×3 neighbourhood.
            let tl = src[(y - 1) * w + (x - 1)];
            let tc = src[(y - 1) * w + x];
            let tr = src[(y - 1) * w + (x + 1)];
            let ml = src[y * w + (x - 1)];
            let mr = src[y * w + (x + 1)];
            let bl = src[(y + 1) * w + (x - 1)];
            let bc = src[(y + 1) * w + x];
            let br = src[(y + 1) * w + (x + 1)];

            // Sobel Gx and Gy magnitudes.
            let gx = (-tl - 2.0 * ml - bl + tr + 2.0 * mr + br).abs();
            let gy = (-tl - 2.0 * tc - tr + bl + 2.0 * bc + br).abs();

            if gx > gy {
                // Vertical edge: smooth horizontally (preserve the edge).
                out[y * w + x] = (ml + mr) * 0.5;
            }
            // else: keep bicubic value
        }
    }

    out
}

/// Apply a simple unsharp mask with a 3×3 box blur and the given `strength`.
///
/// `output = clamp(src + strength × (src − blur))` where blur is a 3×3 box
/// average.  `strength` = 0.5 recovers moderate sharpness without ringing.
#[allow(dead_code)]
fn unsharp_mask(src: &[f32], w: usize, h: usize, strength: f32) -> Vec<f32> {
    // 3×3 box blur.
    let mut blur = src.to_vec();
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let sum = src[(y - 1) * w + (x - 1)]
                + src[(y - 1) * w + x]
                + src[(y - 1) * w + (x + 1)]
                + src[y * w + (x - 1)]
                + src[y * w + x]
                + src[y * w + (x + 1)]
                + src[(y + 1) * w + (x - 1)]
                + src[(y + 1) * w + x]
                + src[(y + 1) * w + (x + 1)];
            blur[y * w + x] = sum / 9.0;
        }
    }

    // Unsharp mask: original + strength × (original − blurred).
    src.iter()
        .zip(blur.iter())
        .map(|(&s, &b)| (s + strength * (s - b)).clamp(0.0, 1.0))
        .collect()
}

/// Bicubic upscale using cubic Hermite spline.
#[allow(dead_code)]
fn bicubic_upscale(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; dw * dh];
    let scale_x = sw as f32 / dw as f32;
    let scale_y = sh as f32 / dh as f32;

    for dy in 0..dh {
        for dx in 0..dw {
            let fx = (dx as f32 + 0.5) * scale_x - 0.5;
            let fy = (dy as f32 + 0.5) * scale_y - 0.5;
            dst[dy * dw + dx] = bicubic_sample(src, sw, sh, fx, fy);
        }
    }
    dst
}

/// Sample with bicubic (Catmull-Rom) interpolation.
#[allow(dead_code)]
fn bicubic_sample(src: &[f32], sw: usize, sh: usize, fx: f32, fy: f32) -> f32 {
    let ix = fx.floor() as i32;
    let iy = fy.floor() as i32;
    let tx = fx - ix as f32;
    let ty = fy - iy as f32;

    let wx = catmull_rom_weights(tx);
    let wy = catmull_rom_weights(ty);

    let mut result = 0.0f32;
    for (j, &wy_j) in wy.iter().enumerate() {
        for (i, &wx_i) in wx.iter().enumerate() {
            let px = (ix + i as i32 - 1).clamp(0, sw as i32 - 1) as usize;
            let py = (iy + j as i32 - 1).clamp(0, sh as i32 - 1) as usize;
            result += src[py * sw + px] * wx_i * wy_j;
        }
    }
    result.clamp(0.0, 1.0)
}

/// Catmull-Rom spline weights for fractional position `t` in [0,1].
#[allow(dead_code)]
fn catmull_rom_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ]
}

/// Compute normalized Sobel edge magnitude (0.0–1.0).
#[allow(dead_code)]
fn compute_sobel_edges(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut edges = vec![0.0f32; w * h];
    let mut max_val = 0.0f32;

    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let tl = src[(y - 1) * w + (x - 1)];
            let tc = src[(y - 1) * w + x];
            let tr = src[(y - 1) * w + (x + 1)];
            let ml = src[y * w + (x - 1)];
            let mr = src[y * w + (x + 1)];
            let bl = src[(y + 1) * w + (x - 1)];
            let bc = src[(y + 1) * w + x];
            let br = src[(y + 1) * w + (x + 1)];

            let gx = -tl - 2.0 * ml - bl + tr + 2.0 * mr + br;
            let gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;
            let mag = (gx * gx + gy * gy).sqrt();
            edges[y * w + x] = mag;
            if mag > max_val {
                max_val = mag;
            }
        }
    }

    // Normalize
    if max_val > 1e-8 {
        for e in &mut edges {
            *e /= max_val;
        }
    }
    edges
}

/// Frequency-domain upscaler using DFT zero-padding.
pub struct FrequencyUpscaler;

impl FrequencyUpscaler {
    /// Upscale a single-channel image by zero-padding in the frequency domain.
    ///
    /// Implements a simplified real 2D DFT: computes the DFT of the source,
    /// zero-pads the frequency coefficients, and applies the inverse DFT.
    #[must_use]
    #[allow(dead_code)]
    pub fn upscale(src: &[f32], src_w: u32, src_h: u32, scale: u32) -> Vec<f32> {
        if src_w == 0 || src_h == 0 || scale == 0 {
            return Vec::new();
        }
        let sw = src_w as usize;
        let sh = src_h as usize;
        let dw = sw * scale as usize;
        let dh = sh * scale as usize;

        // Forward DFT (real-to-complex using naive DFT for correctness)
        let freqs = dft_2d(src, sw, sh);

        // Zero-pad in frequency domain: center-pad the spectrum
        let mut padded = vec![(0.0f32, 0.0f32); dw * dh];
        let half_sw = sw / 2;
        let half_sh = sh / 2;

        for fy in 0..sh {
            for fx in 0..sw {
                let dst_fx = if fx < half_sw { fx } else { dw - (sw - fx) };
                let dst_fy = if fy < half_sh { fy } else { dh - (sh - fy) };
                if dst_fx < dw && dst_fy < dh {
                    padded[dst_fy * dw + dst_fx] = freqs[fy * sw + fx];
                }
            }
        }

        // Inverse DFT
        let scale_factor = (scale * scale) as f32;
        let spatial = idft_2d(&padded, dw, dh);
        spatial
            .iter()
            .map(|&v| (v / scale_factor).clamp(0.0, 1.0))
            .collect()
    }
}

/// Naive 2D DFT (O(N^4) — suitable for small test images only).
#[allow(dead_code)]
fn dft_2d(src: &[f32], w: usize, h: usize) -> Vec<(f32, f32)> {
    let mut out = vec![(0.0f32, 0.0f32); w * h];
    let wf = w as f32;
    let hf = h as f32;

    for vy in 0..h {
        for vx in 0..w {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for sy in 0..h {
                for sx in 0..w {
                    let angle =
                        -2.0 * PI * (vx as f32 * sx as f32 / wf + vy as f32 * sy as f32 / hf);
                    let val = src[sy * w + sx];
                    re += val * angle.cos();
                    im += val * angle.sin();
                }
            }
            out[vy * w + vx] = (re, im);
        }
    }
    out
}

/// Naive 2D inverse DFT.
#[allow(dead_code)]
fn idft_2d(freq: &[(f32, f32)], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    let wf = w as f32;
    let hf = h as f32;
    let norm = wf * hf;

    for sy in 0..h {
        for sx in 0..w {
            let mut val = 0.0f32;
            for vy in 0..h {
                for vx in 0..w {
                    let angle =
                        2.0 * PI * (vx as f32 * sx as f32 / wf + vy as f32 * sy as f32 / hf);
                    let (re, im) = freq[vy * w + vx];
                    val += re * angle.cos() - im * angle.sin();
                }
            }
            out[sy * w + sx] = val / norm;
        }
    }
    out
}

/// Quality estimate for super-resolution output.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SrQualityEstimate {
    /// Estimated PSNR improvement in dB.
    pub psnr_db_estimate: f32,
    /// Edge sharpness score (0.0–1.0, higher is sharper).
    pub edge_sharpness: f32,
    /// Aliasing score (0.0–1.0, lower is better).
    pub alias_score: f32,
}

impl SrQualityEstimate {
    /// Compute quality metrics by comparing original and upscaled images.
    #[must_use]
    #[allow(dead_code)]
    pub fn compute(original: &[f32], upscaled: &[f32], scale: u32) -> Self {
        // Downsample upscaled back to original size and compute MSE
        let s = scale as usize;
        let dst_len = upscaled.len() / (s * s);
        let orig_len = original.len();
        let compare_len = dst_len.min(orig_len);

        // Simple PSNR estimate: compare downsampled-upscaled vs original
        let mut mse = 0.0f32;
        for i in 0..compare_len {
            let up_y = (i / (compare_len / s.max(1)).max(1)) * s;
            let up_x = (i % (compare_len / s.max(1)).max(1)) * s;
            let w_up = ((upscaled.len() as f32).sqrt() as usize).max(1);
            let idx = (up_y * w_up + up_x).min(upscaled.len() - 1);
            let diff = original[i] - upscaled[idx];
            mse += diff * diff;
        }
        mse /= compare_len.max(1) as f32;

        let psnr_db_estimate = if mse < 1e-10 {
            100.0
        } else {
            10.0 * (1.0 / mse).log10()
        };

        // Edge sharpness: measure gradient magnitude in upscaled
        let w = ((upscaled.len() as f32).sqrt() as usize).max(1);
        let h = (upscaled.len() / w).max(1);
        let edges = compute_sobel_edges(upscaled, w, h);
        let edge_sharpness = edges.iter().copied().sum::<f32>() / edges.len() as f32;

        // Alias score: high-frequency energy ratio
        let total_energy: f32 = upscaled.iter().map(|&v| v * v).sum();
        let hf_energy: f32 = edges.iter().map(|&v| v * v).sum();
        let alias_score = if total_energy > 1e-8 {
            (hf_energy / total_energy).min(1.0)
        } else {
            0.0
        };

        Self {
            psnr_db_estimate,
            edge_sharpness,
            alias_score,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sr_config_default() {
        let config = SrConfig::default();
        assert_eq!(config.scale, 2);
        assert_eq!(config.algorithm, SrAlgorithm::EdgePreserving);
    }

    #[test]
    fn test_edge_preserving_upscale_output_size() {
        let src = vec![0.5f32; 16]; // 4x4
        let dst = EdgePreservingUpscaler::upscale(&src, 4, 4, 2);
        assert_eq!(dst.len(), 64); // 8x8
    }

    #[test]
    fn test_edge_preserving_upscale_uniform() {
        // Uniform source should produce uniform output
        let src = vec![0.5f32; 16];
        let dst = EdgePreservingUpscaler::upscale(&src, 4, 4, 2);
        for &v in &dst {
            assert!((v - 0.5).abs() < 0.01, "Expected ~0.5, got {v}");
        }
    }

    #[test]
    fn test_edge_preserving_upscale_empty() {
        let dst = EdgePreservingUpscaler::upscale(&[], 0, 0, 2);
        assert!(dst.is_empty());
    }

    #[test]
    fn test_sobel_edges_uniform_image() {
        let src = vec![0.5f32; 16];
        let edges = compute_sobel_edges(&src, 4, 4);
        // Uniform image has zero gradients
        for &e in &edges {
            assert!(e.abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_sobel_edges_step_edge() {
        let mut src = vec![0.0f32; 16];
        // Left half = 0, right half = 1
        for y in 0..4 {
            for x in 2..4 {
                src[y * 4 + x] = 1.0;
            }
        }
        let edges = compute_sobel_edges(&src, 4, 4);
        // Edge should be detected in the middle column
        assert!(edges[1 * 4 + 2] > 0.0 || edges[2 * 4 + 2] > 0.0);
    }

    #[test]
    fn test_frequency_upscaler_output_size() {
        // Use very small image to keep DFT tractable
        let src = vec![0.5f32; 4]; // 2x2
        let dst = FrequencyUpscaler::upscale(&src, 2, 2, 2);
        assert_eq!(dst.len(), 16); // 4x4
    }

    #[test]
    fn test_frequency_upscaler_empty() {
        let dst = FrequencyUpscaler::upscale(&[], 0, 0, 2);
        assert!(dst.is_empty());
    }

    #[test]
    fn test_sr_quality_estimate_perfect() {
        let img = vec![0.5f32; 16];
        let upscaled = vec![0.5f32; 64];
        let q = SrQualityEstimate::compute(&img, &upscaled, 2);
        assert!(q.psnr_db_estimate > 30.0);
    }

    #[test]
    fn test_bicubic_upscale_size() {
        let src = vec![0.5f32; 16];
        let dst = bicubic_upscale(&src, 4, 4, 8, 8);
        assert_eq!(dst.len(), 64);
    }

    #[test]
    fn test_catmull_rom_at_zero() {
        let w = catmull_rom_weights(0.0);
        // At t=0 the weight for the second control point should be 1.0
        assert!((w[1] - 1.0).abs() < 1e-5);
    }
}
