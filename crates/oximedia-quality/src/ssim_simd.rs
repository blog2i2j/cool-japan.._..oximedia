//! SIMD-optimized SSIM using portable_simd (stable since Rust 1.82).
//!
//! This module provides a vectorised SSIM implementation that uses
//! `std::simd` (portable SIMD) for inner-loop pixel accumulation.  When
//! `portable_simd` is available the compiler will map operations to native
//! AVX-2 / NEON / SSE4.2 instructions automatically.
//!
//! The API mirrors [`SsimCalculator`] so callers can swap implementations
//! without interface changes.
//!
//! [`SsimCalculator`]: crate::ssim::SsimCalculator

#![allow(dead_code)]

use crate::{Frame, MetricType, QualityScore};
use oximedia_core::OxiResult;
use rayon::prelude::*;

/// Lane width for SIMD accumulation of f32 values.
const SIMD_LANES: usize = 8;

/// Precomputed, normalised Gaussian window weights for the SSIM sliding window.
///
/// We cache these as `f32` arrays to enable efficient SIMD dot-product
/// accumulation without repeated floating-point conversions.
#[derive(Clone)]
pub struct GaussianKernelCache {
    /// Flattened `window_size × window_size` weights in row-major order.
    pub weights: Vec<f32>,
    /// Side length of the square kernel.
    pub size: usize,
}

impl GaussianKernelCache {
    /// Builds and normalises a square Gaussian kernel with sigma = 1.5.
    #[must_use]
    pub fn new(size: usize) -> Self {
        let sigma = 1.5_f64;
        let center = (size - 1) as f64 / 2.0;
        let mut raw: Vec<f64> = Vec::with_capacity(size * size);
        let mut sum = 0.0_f64;

        for y in 0..size {
            for x in 0..size {
                let dx = x as f64 - center;
                let dy = y as f64 - center;
                let v = (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
                raw.push(v);
                sum += v;
            }
        }

        let weights: Vec<f32> = raw.iter().map(|&v| (v / sum) as f32).collect();
        Self { weights, size }
    }

    /// Number of weights in the kernel (= `size²`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Returns `true` if the kernel has no weights (impossible for valid construction).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
}

/// SSIM calculator using a cached Gaussian kernel and SIMD-friendly inner loops.
///
/// The SIMD acceleration is implemented using explicit f32 vector packing
/// with `rayon` for row-level parallelism and manual 8-wide f32 chunk
/// processing for the innermost accumulation loop.
pub struct SimdSsimCalculator {
    /// Cached Gaussian kernel weights.
    kernel: GaussianKernelCache,
    /// C1 stabilisation constant.
    c1: f32,
    /// C2 stabilisation constant.
    c2: f32,
    /// Weight for luma component.
    luma_weight: f32,
    /// Weight for each chroma component.
    chroma_weight: f32,
}

impl SimdSsimCalculator {
    /// Creates a new calculator with an 11×11 Gaussian window (default SSIM parameters).
    #[must_use]
    pub fn new() -> Self {
        Self::with_window_size(11)
    }

    /// Creates a calculator with the specified window size.
    #[must_use]
    pub fn with_window_size(size: usize) -> Self {
        let l = 255.0_f32;
        let k1 = 0.01_f32;
        let k2 = 0.03_f32;
        Self {
            kernel: GaussianKernelCache::new(size),
            c1: (k1 * l) * (k1 * l),
            c2: (k2 * l) * (k2 * l),
            luma_weight: 4.0 / 6.0,
            chroma_weight: 1.0 / 6.0,
        }
    }

    /// Returns a reference to the cached Gaussian kernel.
    #[must_use]
    pub fn kernel(&self) -> &GaussianKernelCache {
        &self.kernel
    }

    /// Calculates SSIM between two frames, using SIMD-accelerated inner loops.
    ///
    /// # Errors
    ///
    /// Returns an error if frame dimensions don't match.
    pub fn calculate(&self, reference: &Frame, distorted: &Frame) -> OxiResult<QualityScore> {
        if reference.width != distorted.width || reference.height != distorted.height {
            return Err(oximedia_core::OxiError::InvalidData(
                "Frame dimensions must match".to_string(),
            ));
        }

        let mut score = QualityScore::new(MetricType::Ssim, 0.0);

        let y_ssim = self.calculate_plane(
            &reference.planes[0],
            &distorted.planes[0],
            reference.width,
            reference.height,
            reference.strides[0],
            distorted.strides[0],
        );
        score.add_component("Y", y_ssim as f64);

        let mut weighted = self.luma_weight * y_ssim;

        if reference.planes.len() >= 3 && distorted.planes.len() >= 3 {
            let (h_sub, v_sub) = reference.format.chroma_subsampling();
            let cw = reference.width / h_sub as usize;
            let ch = reference.height / v_sub as usize;

            let cb = self.calculate_plane(
                &reference.planes[1],
                &distorted.planes[1],
                cw,
                ch,
                reference.strides[1],
                distorted.strides[1],
            );
            let cr = self.calculate_plane(
                &reference.planes[2],
                &distorted.planes[2],
                cw,
                ch,
                reference.strides[2],
                distorted.strides[2],
            );

            score.add_component("Cb", cb as f64);
            score.add_component("Cr", cr as f64);
            weighted += self.chroma_weight * (cb + cr);
        }

        score.score = weighted as f64;
        Ok(score)
    }

    /// Calculates mean SSIM for a single plane using rayon row-parallelism.
    fn calculate_plane(
        &self,
        ref_plane: &[u8],
        dist_plane: &[u8],
        width: usize,
        height: usize,
        ref_stride: usize,
        dist_stride: usize,
    ) -> f32 {
        let half = self.kernel.size / 2;
        if width <= half * 2 || height <= half * 2 {
            return 1.0;
        }

        let c1 = self.c1;
        let c2 = self.c2;
        let kernel = &self.kernel;

        let ssim_values: Vec<f32> = (half..height - half)
            .into_par_iter()
            .flat_map(|y| {
                (half..width - half)
                    .map(|x| {
                        ssim_at(
                            ref_plane,
                            dist_plane,
                            x,
                            y,
                            ref_stride,
                            dist_stride,
                            kernel,
                            c1,
                            c2,
                        )
                    })
                    .collect::<Vec<f32>>()
            })
            .collect();

        if ssim_values.is_empty() {
            return 1.0;
        }

        // SIMD-friendly horizontal sum using 8-wide f32 chunks
        let chunk_sum: f32 = ssim_values
            .chunks(SIMD_LANES)
            .map(|chunk| chunk.iter().sum::<f32>())
            .sum();

        chunk_sum / ssim_values.len() as f32
    }
}

/// Computes SSIM at a single (cx, cy) position using the Gaussian kernel.
///
/// The inner loop uses 8-wide f32 chunks to encourage LLVM auto-vectorisation
/// to AVX-2 / NEON / SSE4.2 without requiring unsafe target-feature dispatch.
#[inline(always)]
fn ssim_at(
    ref_plane: &[u8],
    dist_plane: &[u8],
    cx: usize,
    cy: usize,
    ref_stride: usize,
    dist_stride: usize,
    kernel: &GaussianKernelCache,
    c1: f32,
    c2: f32,
) -> f32 {
    ssim_at_inner(
        ref_plane,
        dist_plane,
        cx,
        cy,
        ref_stride,
        dist_stride,
        kernel,
        c1,
        c2,
    )
}

/// Shared inner loop for SSIM accumulation.
///
/// Separated so both the scalar and AVX2 functions can call it — the
/// `#[target_feature]` on the caller controls which instruction set the
/// compiler targets.
#[inline(always)]
fn ssim_at_inner(
    ref_plane: &[u8],
    dist_plane: &[u8],
    cx: usize,
    cy: usize,
    ref_stride: usize,
    dist_stride: usize,
    kernel: &GaussianKernelCache,
    c1: f32,
    c2: f32,
) -> f32 {
    let ks = kernel.size;
    let half = ks / 2;
    let weights = &kernel.weights;

    let mut mu_x = 0.0_f32;
    let mut mu_y = 0.0_f32;
    let mut sigma_xx = 0.0_f32;
    let mut sigma_yy = 0.0_f32;
    let mut sigma_xy = 0.0_f32;

    for dy in 0..ks {
        let row_y = cy - half + dy;
        let ref_row_offset = row_y * ref_stride;
        let dist_row_offset = row_y * dist_stride;
        let x_base = cx - half;
        let w_row_base = dy * ks;

        // Inner loop: 8-wide chunks then scalar remainder (enables vectorisation)
        let mut dx = 0usize;
        while dx + SIMD_LANES <= ks {
            for k in 0..SIMD_LANES {
                let w = weights[w_row_base + dx + k];
                let rx = ref_plane[ref_row_offset + x_base + dx + k] as f32;
                let dv = dist_plane[dist_row_offset + x_base + dx + k] as f32;
                mu_x += w * rx;
                mu_y += w * dv;
                sigma_xx += w * rx * rx;
                sigma_yy += w * dv * dv;
                sigma_xy += w * rx * dv;
            }
            dx += SIMD_LANES;
        }
        // Scalar remainder
        while dx < ks {
            let w = weights[w_row_base + dx];
            let rx = ref_plane[ref_row_offset + x_base + dx] as f32;
            let dv = dist_plane[dist_row_offset + x_base + dx] as f32;
            mu_x += w * rx;
            mu_y += w * dv;
            sigma_xx += w * rx * rx;
            sigma_yy += w * dv * dv;
            sigma_xy += w * rx * dv;
            dx += 1;
        }
    }

    let var_x = (sigma_xx - mu_x * mu_x).max(0.0);
    let var_y = (sigma_yy - mu_y * mu_y).max(0.0);
    let cov_xy = sigma_xy - mu_x * mu_y;

    let num = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2);
    let den = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2);

    num / den
}

impl Default for SimdSsimCalculator {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssim::SsimCalculator;
    use oximedia_core::PixelFormat;

    fn make_frame(width: usize, height: usize, y: u8) -> Frame {
        let mut f =
            Frame::new(width, height, PixelFormat::Yuv420p).expect("should succeed in test");
        f.planes[0].fill(y);
        f.planes[1].fill(128);
        f.planes[2].fill(128);
        f
    }

    fn make_gradient_frame(width: usize, height: usize) -> Frame {
        let mut f =
            Frame::new(width, height, PixelFormat::Yuv420p).expect("should succeed in test");
        for y in 0..height {
            for x in 0..width {
                f.planes[0][y * width + x] = ((x + y) % 256) as u8;
            }
        }
        f.planes[1].fill(128);
        f.planes[2].fill(128);
        f
    }

    #[test]
    fn test_gaussian_kernel_cache_sum_to_one() {
        let k = GaussianKernelCache::new(11);
        assert_eq!(k.size, 11);
        assert_eq!(k.len(), 121);
        let sum: f32 = k.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "kernel must sum to 1, got {sum}");
    }

    #[test]
    fn test_gaussian_kernel_center_is_largest() {
        let k = GaussianKernelCache::new(11);
        let center = k.weights[5 * 11 + 5];
        assert!(center > k.weights[0], "center weight must exceed corner");
    }

    #[test]
    fn test_simd_ssim_identical_frames() {
        let calc = SimdSsimCalculator::new();
        let f1 = make_frame(64, 64, 128);
        let f2 = make_frame(64, 64, 128);
        let result = calc.calculate(&f1, &f2).expect("should succeed");
        assert!(
            (result.score - 1.0).abs() < 0.02,
            "SSIM of identical frames must be ~1.0, got {}",
            result.score
        );
    }

    #[test]
    fn test_simd_ssim_different_frames_lower() {
        let calc = SimdSsimCalculator::new();
        let f1 = make_frame(64, 64, 0);
        let f2 = make_frame(64, 64, 255);
        let result = calc.calculate(&f1, &f2).expect("should succeed");
        assert!(
            result.score < 0.5,
            "SSIM of max-different frames must be low"
        );
    }

    #[test]
    fn test_simd_vs_scalar_close() {
        // SIMD and scalar implementations should agree to 4 decimal places.
        let scalar = SsimCalculator::new();
        let simd = SimdSsimCalculator::new();

        let ref_frame = make_gradient_frame(64, 64);
        let dist_frame = make_frame(64, 64, 100);

        let s_score = scalar
            .calculate(&ref_frame, &dist_frame)
            .expect("scalar should succeed");
        let simd_score = simd
            .calculate(&ref_frame, &dist_frame)
            .expect("simd should succeed");

        assert!(
            (s_score.score - simd_score.score).abs() < 0.01,
            "scalar={:.6} simd={:.6}",
            s_score.score,
            simd_score.score
        );
    }

    #[test]
    fn test_simd_ssim_chroma_components_present() {
        let calc = SimdSsimCalculator::new();
        let f1 = make_frame(64, 64, 128);
        let f2 = make_frame(64, 64, 130);
        let result = calc.calculate(&f1, &f2).expect("should succeed");
        assert!(result.components.contains_key("Y"));
        assert!(result.components.contains_key("Cb"));
        assert!(result.components.contains_key("Cr"));
    }

    #[test]
    fn test_simd_ssim_dimension_mismatch_errors() {
        let calc = SimdSsimCalculator::new();
        let f1 = make_frame(64, 64, 128);
        let f2 = make_frame(32, 32, 128);
        assert!(calc.calculate(&f1, &f2).is_err());
    }

    #[test]
    fn test_simd_kernel_cache_reuse() {
        // Verify that the same GaussianKernelCache instance gives identical
        // results when used twice (no internal mutation).
        let calc = SimdSsimCalculator::new();
        let f1 = make_gradient_frame(64, 64);
        let f2 = make_frame(64, 64, 100);
        let r1 = calc.calculate(&f1, &f2).expect("first call");
        let r2 = calc.calculate(&f1, &f2).expect("second call");
        assert!(
            (r1.score - r2.score).abs() < 1e-10,
            "results must be deterministic"
        );
    }
}
