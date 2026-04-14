//! Guided image filter for edge-preserving smoothing and inpainting.
//!
//! The guided filter (He et al., 2013) produces output that is locally a linear
//! function of the **guide image** `I`:
//!
//! ```text
//! q_i = a_k · I_i + b_k,   ∀i ∈ ω_k
//! ```
//!
//! The linear coefficients `(a_k, b_k)` are computed per local window `ω_k` of
//! radius `r` by minimising the reconstruction cost with a regularisation
//! term `ε` that controls edge preservation vs. smoothness.
//!
//! Unlike the bilateral filter the guided filter:
//! - Has an **O(N)** complexity regardless of filter radius.
//! - Is a **linear** operation with respect to the guide (predictable artefacts).
//! - Can use a **separate** guide image (e.g., a high-quality reference to
//!   sharpen an upscaled depth map).
//!
//! # References
//!
//! He, K., Sun, J., & Tang, X. (2013). "Guided image filtering."
//! IEEE Transactions on Pattern Analysis and Machine Intelligence.
//!
//! # Example
//!
//! ```rust
//! use oximedia_image::guided_filter::{GuidedFilter, GuidedFilterParams};
//!
//! // A 4×4 uniform image smoothed against itself as the guide.
//! let image = vec![0.5f32; 16];
//! let params = GuidedFilterParams::new(1, 0.01);
//! let filter = GuidedFilter::new(params);
//! let out = filter.apply_self_guided(&image, 4, 4).expect("ok");
//! assert_eq!(out.len(), 16);
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use crate::error::{ImageError, ImageResult};

// ---------------------------------------------------------------------------
// GuidedFilterParams
// ---------------------------------------------------------------------------

/// Parameters for the guided image filter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GuidedFilterParams {
    /// Radius of the local square window (half-size, in pixels).
    ///
    /// The full window is `(2r+1) × (2r+1)` pixels.
    pub radius: usize,
    /// Regularisation parameter `ε`.
    ///
    /// Larger values smooth across edges; smaller values preserve them.
    /// Typical values: `0.01²` to `0.1²` for images normalised to `[0, 1]`.
    pub eps: f32,
}

impl GuidedFilterParams {
    /// Creates new parameters with the given radius and epsilon.
    #[must_use]
    pub fn new(radius: usize, eps: f32) -> Self {
        Self { radius, eps }
    }
}

impl Default for GuidedFilterParams {
    fn default() -> Self {
        Self::new(2, 0.0025) // r=2, ε=0.05²
    }
}

// ---------------------------------------------------------------------------
// GuidedFilter
// ---------------------------------------------------------------------------

/// Guided image filter implementation.
///
/// All inputs are single-channel `f32` buffers in row-major order.
///
/// Construct once, apply many times:
///
/// ```rust
/// use oximedia_image::guided_filter::{GuidedFilter, GuidedFilterParams};
///
/// let filter = GuidedFilter::new(GuidedFilterParams::default());
/// let src = vec![0.5f32; 25];
/// let out = filter.apply_self_guided(&src, 5, 5).expect("ok");
/// assert_eq!(out.len(), 25);
/// ```
#[derive(Debug, Clone)]
pub struct GuidedFilter {
    params: GuidedFilterParams,
}

impl GuidedFilter {
    /// Creates a new guided filter with the given parameters.
    #[must_use]
    pub fn new(params: GuidedFilterParams) -> Self {
        Self { params }
    }

    /// Creates a guided filter with default parameters.
    #[must_use]
    pub fn default_params() -> Self {
        Self::new(GuidedFilterParams::default())
    }

    /// Returns the parameters used by this filter.
    #[must_use]
    pub fn params(&self) -> GuidedFilterParams {
        self.params
    }

    /// Applies the guided filter using `guide` as the guidance image and `src`
    /// as the input to be filtered.
    ///
    /// Both `guide` and `src` must be single-channel `f32`, row-major, and have
    /// length `width * height`. Values should be in `[0.0, 1.0]` for best
    /// results.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::InvalidDimensions`] if either buffer length does not
    /// equal `width * height` or if any dimension is zero.
    pub fn apply(
        &self,
        guide: &[f32],
        src: &[f32],
        width: usize,
        height: usize,
    ) -> ImageResult<Vec<f32>> {
        validate_dims(guide.len(), width, height)?;
        validate_dims(src.len(), width, height)?;
        guided_filter_core(
            guide,
            src,
            width,
            height,
            self.params.radius,
            self.params.eps,
        )
    }

    /// Applies the guided filter with `src` used as both the input and the guide.
    ///
    /// This is the classic use case for edge-preserving smoothing.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::InvalidDimensions`] if `src.len() != width * height`
    /// or if any dimension is zero.
    pub fn apply_self_guided(
        &self,
        src: &[f32],
        width: usize,
        height: usize,
    ) -> ImageResult<Vec<f32>> {
        validate_dims(src.len(), width, height)?;
        guided_filter_core(src, src, width, height, self.params.radius, self.params.eps)
    }

    /// Applies the guided filter to a `u8` image.
    ///
    /// Converts to `f32` in `[0.0, 1.0]`, filters, then converts back.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::InvalidDimensions`] if `src.len() != width * height`
    /// or any dimension is zero.
    pub fn apply_u8(&self, src: &[u8], width: usize, height: usize) -> ImageResult<Vec<u8>> {
        if width == 0 || height == 0 || src.len() != width * height {
            return Err(ImageError::InvalidDimensions(width as u32, height as u32));
        }
        let f: Vec<f32> = src.iter().map(|&v| v as f32 / 255.0).collect();
        let out = self.apply_self_guided(&f, width, height)?;
        Ok(out
            .iter()
            .map(|&v| (v * 255.0).clamp(0.0, 255.0).round() as u8)
            .collect())
    }

    /// Applies the guided filter to each channel of an interleaved multi-channel
    /// `f32` image, using the per-pixel luminance as the single guide.
    ///
    /// `src` must have length `width * height * channels`.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::InvalidDimensions`] if dimensions are inconsistent
    /// or any dimension is zero.
    pub fn apply_multichannel(
        &self,
        src: &[f32],
        width: usize,
        height: usize,
        channels: usize,
    ) -> ImageResult<Vec<f32>> {
        if channels == 0 || width == 0 || height == 0 {
            return Err(ImageError::InvalidDimensions(width as u32, height as u32));
        }
        let npix = width * height;
        if src.len() != npix * channels {
            return Err(ImageError::InvalidDimensions(width as u32, height as u32));
        }

        // Compute luminance guide (channel average).
        let guide: Vec<f32> = (0..npix)
            .map(|i| {
                let base = i * channels;
                let sum: f32 = src[base..base + channels].iter().sum();
                sum / channels as f32
            })
            .collect();

        // Filter each channel with the luminance guide.
        let mut dst = vec![0.0f32; src.len()];
        for c in 0..channels {
            let channel: Vec<f32> = (0..npix).map(|i| src[i * channels + c]).collect();
            let filtered = guided_filter_core(
                &guide,
                &channel,
                width,
                height,
                self.params.radius,
                self.params.eps,
            )?;
            for (i, &v) in filtered.iter().enumerate() {
                dst[i * channels + c] = v;
            }
        }
        Ok(dst)
    }
}

// ---------------------------------------------------------------------------
// Core algorithm
// ---------------------------------------------------------------------------

/// Implements the guided filter algorithm using box-filter (mean) approximations.
///
/// The algorithm proceeds in the following steps for each local window `ω_k`:
///
/// 1. Compute `mean_I`, `mean_p`, `corr_I` (mean of `I²`), `corr_Ip` (mean of `I·p`).
/// 2. `var_I  = corr_I  - mean_I²`
/// 3. `cov_Ip = corr_Ip - mean_I·mean_p`
/// 4. `a_k = cov_Ip / (var_I + ε)`
/// 5. `b_k = mean_p - a_k·mean_I`
/// 6. `q_i = mean(a)_i·I_i + mean(b)_i`
///
/// Step 6 averages the `(a, b)` coefficients over all windows that contain
/// pixel `i`, which is equivalent to a box filter on the `a` and `b` maps.
fn guided_filter_core(
    guide: &[f32],
    src: &[f32],
    width: usize,
    height: usize,
    radius: usize,
    eps: f32,
) -> ImageResult<Vec<f32>> {
    let n = width * height;

    // ── Box filter helper (inline) ───────────────────────────────────────────
    // Computes a per-pixel local mean over a (2r+1)×(2r+1) window.
    let box_filter = |buf: &[f32]| -> Vec<f32> {
        let r = radius as isize;
        let iw = width as isize;
        let ih = height as isize;
        let mut out = vec![0.0f32; n];
        for y in 0..ih {
            for x in 0..iw {
                let mut sum = 0.0f64;
                let mut count = 0usize;
                let y0 = (y - r).max(0);
                let y1 = (y + r + 1).min(ih);
                let x0 = (x - r).max(0);
                let x1 = (x + r + 1).min(iw);
                for ny in y0..y1 {
                    for nx in x0..x1 {
                        sum += buf[(ny * iw + nx) as usize] as f64;
                        count += 1;
                    }
                }
                out[(y * iw + x) as usize] = (sum / count.max(1) as f64) as f32;
            }
        }
        out
    };

    // ── Step 1: compute intermediates ────────────────────────────────────────
    let mean_i = box_filter(guide);
    let mean_p = box_filter(src);

    let corr_i: Vec<f32> = guide.iter().map(|&v| v * v).collect();
    let corr_i = box_filter(&corr_i);

    let corr_ip: Vec<f32> = guide.iter().zip(src.iter()).map(|(&g, &p)| g * p).collect();
    let corr_ip = box_filter(&corr_ip);

    // ── Step 2-5: compute per-pixel (a, b) ──────────────────────────────────
    let mut a_map = vec![0.0f32; n];
    let mut b_map = vec![0.0f32; n];
    for i in 0..n {
        let var_i = corr_i[i] - mean_i[i] * mean_i[i];
        let cov_ip = corr_ip[i] - mean_i[i] * mean_p[i];
        let a = cov_ip / (var_i + eps);
        let b = mean_p[i] - a * mean_i[i];
        a_map[i] = a;
        b_map[i] = b;
    }

    // ── Step 6: average (a, b) then compute output ───────────────────────────
    let mean_a = box_filter(&a_map);
    let mean_b = box_filter(&b_map);

    let out: Vec<f32> = (0..n).map(|i| mean_a[i] * guide[i] + mean_b[i]).collect();
    Ok(out)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_dims(len: usize, width: usize, height: usize) -> ImageResult<()> {
    if width == 0 || height == 0 {
        return Err(ImageError::InvalidDimensions(width as u32, height as u32));
    }
    if len != width * height {
        return Err(ImageError::InvalidDimensions(width as u32, height as u32));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── GuidedFilterParams ────────────────────────────────────────────────────

    #[test]
    fn test_params_fields() {
        let p = GuidedFilterParams::new(3, 0.01);
        assert_eq!(p.radius, 3);
        assert!((p.eps - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_params_default_reasonable() {
        let p = GuidedFilterParams::default();
        assert!(p.radius >= 1);
        assert!(p.eps > 0.0);
    }

    // ── apply_self_guided ────────────────────────────────────────────────────

    #[test]
    fn test_self_guided_output_length() {
        let src = vec![0.5f32; 6 * 6];
        let f = GuidedFilter::new(GuidedFilterParams::new(1, 0.01));
        let out = f.apply_self_guided(&src, 6, 6).expect("ok");
        assert_eq!(out.len(), 36);
    }

    #[test]
    fn test_self_guided_uniform_image_unchanged() {
        // Uniform image: all (a=0, b=mean=const) → output equals input.
        let src = vec![0.4f32; 5 * 5];
        let f = GuidedFilter::new(GuidedFilterParams::new(2, 0.001));
        let out = f.apply_self_guided(&src, 5, 5).expect("ok");
        for (i, (&s, &d)) in src.iter().zip(out.iter()).enumerate() {
            assert!((s - d).abs() < 1e-4, "pixel {i}: {s} vs {d}");
        }
    }

    #[test]
    fn test_self_guided_invalid_length() {
        let src = vec![0.5f32; 9]; // not 4×4
        let f = GuidedFilter::default_params();
        assert!(f.apply_self_guided(&src, 4, 4).is_err());
    }

    #[test]
    fn test_self_guided_zero_dimensions() {
        let src: Vec<f32> = vec![];
        let f = GuidedFilter::default_params();
        assert!(f.apply_self_guided(&src, 0, 5).is_err());
    }

    // ── apply (guide ≠ src) ───────────────────────────────────────────────────

    #[test]
    fn test_apply_with_separate_guide_output_length() {
        let guide = vec![0.8f32; 4 * 4];
        let src = vec![0.2f32; 4 * 4];
        let f = GuidedFilter::new(GuidedFilterParams::new(1, 0.01));
        let out = f.apply(&guide, &src, 4, 4).expect("ok");
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn test_apply_guide_length_mismatch() {
        let guide = vec![0.5f32; 16]; // 4×4
        let src = vec![0.5f32; 9]; // 3×3 — mismatch
        let f = GuidedFilter::default_params();
        assert!(f.apply(&guide, &src, 4, 4).is_err());
    }

    // ── apply_u8 ─────────────────────────────────────────────────────────────

    #[test]
    fn test_apply_u8_output_length() {
        let src = vec![128u8; 7 * 7];
        let f = GuidedFilter::new(GuidedFilterParams::new(1, 0.01));
        let out = f.apply_u8(&src, 7, 7).expect("ok");
        assert_eq!(out.len(), 49);
    }

    #[test]
    fn test_apply_u8_uniform_image_unchanged() {
        let src = vec![100u8; 4 * 4];
        let f = GuidedFilter::new(GuidedFilterParams::new(1, 0.001));
        let out = f.apply_u8(&src, 4, 4).expect("ok");
        // uniform → no change (allow ±1 for rounding)
        for (i, (&s, &d)) in src.iter().zip(out.iter()).enumerate() {
            let diff = (s as i32 - d as i32).unsigned_abs();
            assert!(diff <= 1, "pixel {i}: {s} vs {d}");
        }
    }

    #[test]
    fn test_apply_u8_invalid_dimensions() {
        let src = vec![128u8; 9]; // not 4×4
        let f = GuidedFilter::default_params();
        assert!(f.apply_u8(&src, 4, 4).is_err());
    }

    // ── apply_multichannel ────────────────────────────────────────────────────

    #[test]
    fn test_multichannel_output_length() {
        let channels = 3;
        let src = vec![0.5f32; 5 * 5 * channels];
        let f = GuidedFilter::new(GuidedFilterParams::new(1, 0.01));
        let out = f.apply_multichannel(&src, 5, 5, channels).expect("ok");
        assert_eq!(out.len(), src.len());
    }

    #[test]
    fn test_multichannel_uniform_unchanged() {
        let channels = 4;
        let src = vec![0.3f32; 4 * 4 * channels];
        let f = GuidedFilter::new(GuidedFilterParams::new(2, 0.001));
        let out = f.apply_multichannel(&src, 4, 4, channels).expect("ok");
        for (i, &v) in out.iter().enumerate() {
            assert!((v - 0.3).abs() < 1e-4, "pixel {i}: {v}");
        }
    }

    #[test]
    fn test_multichannel_zero_channels_error() {
        let src = vec![0.5f32; 16];
        let f = GuidedFilter::default_params();
        assert!(f.apply_multichannel(&src, 4, 4, 0).is_err());
    }

    // ── Edge preservation property ────────────────────────────────────────────

    #[test]
    fn test_edge_preservation_with_small_eps() {
        // With very small eps (strong edge preservation), a step edge in the
        // guide should keep the output edge sharp.
        let width = 10;
        let height = 1;
        let mut src = vec![0.0f32; width * height];
        for i in 5..10 {
            src[i] = 1.0;
        }
        let f = GuidedFilter::new(GuidedFilterParams::new(1, 1e-6));
        let out = f.apply_self_guided(&src, width, height).expect("ok");
        // Left half should remain close to 0.
        assert!(out[0] < 0.2, "left side not preserved: {}", out[0]);
        // Right half should remain close to 1.
        assert!(out[9] > 0.8, "right side not preserved: {}", out[9]);
    }
}
