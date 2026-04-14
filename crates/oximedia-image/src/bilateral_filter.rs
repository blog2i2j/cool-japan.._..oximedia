//! Bilateral filter for edge-preserving image denoising.
//!
//! The bilateral filter smooths images while preserving edges by weighting
//! neighbouring pixels both by spatial distance and by photometric (intensity)
//! similarity. Unlike a simple Gaussian blur it therefore does not blur across
//! strong intensity edges.
//!
//! # Algorithm
//!
//! For each output pixel `(x, y)` and every neighbour `(xi, yi)` within the
//! filter radius, the weight is:
//!
//! ```text
//! w(x,xi) = exp(-(dx²+dy²) / (2·σ_s²)) * exp(-(I(x)-I(xi))² / (2·σ_r²))
//! ```
//!
//! where:
//! - `σ_s` (sigma_spatial) controls how wide the spatial Gaussian is.
//! - `σ_r` (sigma_range) controls how tolerant to intensity differences the
//!   filter is — large values allow blurring across edges.
//!
//! The output is the weighted average of the neighbours.
//!
//! # Complexity
//!
//! Naïve: O(W·H·(2·r+1)²). Use `sigma_spatial` to control the radius
//! (`radius = ceil(3·sigma_spatial)`). For large images consider
//! [`BilateralFilter::apply_multichannel`] which processes channels in one pass.
//!
//! # Example
//!
//! ```rust
//! use oximedia_image::bilateral_filter::{BilateralFilter, BilateralParams};
//!
//! // 4×4 uniform grey image — no smoothing artefacts expected.
//! let src = vec![128u8; 16];
//! let params = BilateralParams::new(1.5, 25.0);
//! let filter = BilateralFilter::new(params);
//! let dst = filter.apply_u8(&src, 4, 4).expect("filter failed");
//! assert_eq!(dst.len(), 16);
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use crate::error::{ImageError, ImageResult};

// ---------------------------------------------------------------------------
// BilateralParams
// ---------------------------------------------------------------------------

/// Parameters controlling the bilateral filter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BilateralParams {
    /// Spatial standard deviation (controls the neighbourhood size).
    ///
    /// The filter radius is `ceil(3 * sigma_spatial)`.
    pub sigma_spatial: f32,

    /// Range (intensity) standard deviation.
    ///
    /// Larger values allow blurring across stronger edges.
    pub sigma_range: f32,
}

impl BilateralParams {
    /// Creates new bilateral filter parameters.
    ///
    /// # Arguments
    ///
    /// * `sigma_spatial` – spatial Gaussian standard deviation (must be > 0).
    /// * `sigma_range`   – range/intensity Gaussian standard deviation (must be > 0).
    #[must_use]
    pub fn new(sigma_spatial: f32, sigma_range: f32) -> Self {
        Self {
            sigma_spatial,
            sigma_range,
        }
    }

    /// Returns the integer filter radius derived from `sigma_spatial`.
    ///
    /// Pixels further than `radius` are assigned negligible weight and are
    /// ignored, bounding the computation while keeping accuracy.
    #[must_use]
    pub fn radius(&self) -> usize {
        ((3.0 * self.sigma_spatial).ceil() as usize).max(1)
    }
}

impl Default for BilateralParams {
    fn default() -> Self {
        Self::new(2.0, 30.0)
    }
}

// ---------------------------------------------------------------------------
// Pre-computed lookup tables
// ---------------------------------------------------------------------------

/// Pre-computed Gaussian lookup tables shared between calls.
///
/// Building the tables once and reusing them across all output pixels reduces
/// repeated `exp` evaluations, which dominate the cost on large images.
struct GaussianTables {
    /// `spatial_lut[d²] = exp(-d² / (2·σ_s²))`, indexed by squared integer offset.
    spatial_lut: Vec<f32>,
    /// `range_lut[Δ] = exp(-Δ² / (2·σ_r²))` for Δ in 0..256 (u8 differences).
    range_lut: [f32; 256],
}

impl GaussianTables {
    fn build(params: &BilateralParams) -> Self {
        let radius = params.radius();
        let max_sq = 2 * radius * radius + 1;
        let two_ss2 = 2.0 * params.sigma_spatial * params.sigma_spatial;
        let two_sr2 = 2.0 * params.sigma_range * params.sigma_range;

        let spatial_lut: Vec<f32> = (0..=max_sq)
            .map(|sq| (-(sq as f32) / two_ss2).exp())
            .collect();

        let mut range_lut = [0.0f32; 256];
        for (delta, val) in range_lut.iter_mut().enumerate() {
            let d = delta as f32;
            *val = (-(d * d) / two_sr2).exp();
        }

        Self {
            spatial_lut,
            range_lut,
        }
    }

    /// Returns the spatial weight for squared distance `sq_dist`.
    #[inline]
    fn spatial(&self, sq_dist: usize) -> f32 {
        // Index may exceed pre-computed range for a pixel at the corner of the
        // radius × radius window; in that case the contribution is negligible.
        self.spatial_lut.get(sq_dist).copied().unwrap_or(0.0)
    }

    /// Returns the range weight for intensity difference `delta` (u8 absolute diff).
    #[inline]
    fn range(&self, delta: u8) -> f32 {
        self.range_lut[delta as usize]
    }
}

// ---------------------------------------------------------------------------
// BilateralFilter
// ---------------------------------------------------------------------------

/// Bilateral filter implementation.
///
/// Construct once and call [`BilateralFilter::apply_u8`] or
/// [`BilateralFilter::apply_f32`] on each image.
#[derive(Debug, Clone)]
pub struct BilateralFilter {
    params: BilateralParams,
}

impl BilateralFilter {
    /// Creates a new bilateral filter with the given parameters.
    #[must_use]
    pub fn new(params: BilateralParams) -> Self {
        Self { params }
    }

    /// Creates a bilateral filter with the default parameters (`σ_s=2`, `σ_r=30`).
    #[must_use]
    pub fn default_params() -> Self {
        Self::new(BilateralParams::default())
    }

    /// Returns the parameters used by this filter.
    #[must_use]
    pub fn params(&self) -> BilateralParams {
        self.params
    }

    /// Applies the bilateral filter to a single-channel `u8` image.
    ///
    /// `src` must have length `width * height`. Returns a new buffer of the same
    /// length with the filtered result.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::InvalidDimensions`] if `src.len() != width * height`
    /// or if either dimension is zero.
    pub fn apply_u8(&self, src: &[u8], width: usize, height: usize) -> ImageResult<Vec<u8>> {
        validate_dimensions(src.len(), width, height)?;
        let tables = GaussianTables::build(&self.params);
        let radius = self.params.radius();
        let mut dst = vec![0u8; src.len()];

        for y in 0..height {
            for x in 0..width {
                let center = src[y * width + x];
                let mut weight_sum = 0.0f32;
                let mut value_sum = 0.0f32;

                let y_start = y.saturating_sub(radius);
                let y_end = (y + radius + 1).min(height);
                let x_start = x.saturating_sub(radius);
                let x_end = (x + radius + 1).min(width);

                for ny in y_start..y_end {
                    for nx in x_start..x_end {
                        let neighbour = src[ny * width + nx];
                        let dy = (ny as isize) - (y as isize);
                        let dx = (nx as isize) - (x as isize);
                        let sq = (dx * dx + dy * dy) as usize;
                        let delta = center.abs_diff(neighbour);
                        let w = tables.spatial(sq) * tables.range(delta);
                        weight_sum += w;
                        value_sum += w * (neighbour as f32);
                    }
                }

                let out = if weight_sum > 0.0 {
                    (value_sum / weight_sum).clamp(0.0, 255.0) as u8
                } else {
                    center
                };
                dst[y * width + x] = out;
            }
        }
        Ok(dst)
    }

    /// Applies the bilateral filter to a single-channel `f32` image.
    ///
    /// Pixel values are expected in `[0.0, 1.0]` but the filter also works for
    /// wider ranges as long as `sigma_range` is set appropriately.
    ///
    /// `src` must have length `width * height`. Returns a new buffer.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::InvalidDimensions`] if `src.len() != width * height`
    /// or if either dimension is zero.
    pub fn apply_f32(&self, src: &[f32], width: usize, height: usize) -> ImageResult<Vec<f32>> {
        validate_dimensions(src.len(), width, height)?;
        let radius = self.params.radius();
        let two_ss2 = 2.0 * self.params.sigma_spatial * self.params.sigma_spatial;
        let two_sr2 = 2.0 * self.params.sigma_range * self.params.sigma_range;
        let mut dst = vec![0.0f32; src.len()];

        for y in 0..height {
            for x in 0..width {
                let center = src[y * width + x];
                let mut weight_sum = 0.0f32;
                let mut value_sum = 0.0f32;

                let y_start = y.saturating_sub(radius);
                let y_end = (y + radius + 1).min(height);
                let x_start = x.saturating_sub(radius);
                let x_end = (x + radius + 1).min(width);

                for ny in y_start..y_end {
                    for nx in x_start..x_end {
                        let neighbour = src[ny * width + nx];
                        let dy = (ny as f32) - (y as f32);
                        let dx = (nx as f32) - (x as f32);
                        let sq_spatial = dx * dx + dy * dy;
                        let diff_range = neighbor_diff_f32(center, neighbour);
                        let w =
                            (-(sq_spatial / two_ss2) - (diff_range * diff_range / two_sr2)).exp();
                        weight_sum += w;
                        value_sum += w * neighbour;
                    }
                }

                dst[y * width + x] = if weight_sum > 0.0 {
                    value_sum / weight_sum
                } else {
                    center
                };
            }
        }
        Ok(dst)
    }

    /// Applies the bilateral filter to an interleaved multi-channel `u8` image.
    ///
    /// `src` must have length `width * height * channels`. Each channel is
    /// filtered independently using the *same* per-channel intensity difference
    /// (i.e., the range weight is computed from the *luminance* of each pixel to
    /// preserve colour consistency).
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::InvalidDimensions`] if `src.len() != width * height * channels`
    /// or if any dimension is zero.
    pub fn apply_multichannel(
        &self,
        src: &[u8],
        width: usize,
        height: usize,
        channels: usize,
    ) -> ImageResult<Vec<u8>> {
        if channels == 0 {
            return Err(ImageError::InvalidDimensions(0, 0));
        }
        validate_dimensions(src.len(), width * channels, height)?;

        let tables = GaussianTables::build(&self.params);
        let radius = self.params.radius();
        let npix = width * height;
        let mut dst = vec![0u8; src.len()];

        // Pre-compute per-pixel luminance (average of all channels) for range weight.
        let luma: Vec<f32> = (0..npix)
            .map(|idx| {
                let base = idx * channels;
                let sum: u32 = src[base..base + channels].iter().map(|&v| v as u32).sum();
                sum as f32 / channels as f32
            })
            .collect();

        for y in 0..height {
            for x in 0..width {
                let center_luma = luma[y * width + x];
                let mut weight_sum = 0.0f32;
                let mut value_sums = vec![0.0f32; channels];

                let y_start = y.saturating_sub(radius);
                let y_end = (y + radius + 1).min(height);
                let x_start = x.saturating_sub(radius);
                let x_end = (x + radius + 1).min(width);

                for ny in y_start..y_end {
                    for nx in x_start..x_end {
                        let nidx = ny * width + nx;
                        let neighbour_luma = luma[nidx];
                        let dy = (ny as isize) - (y as isize);
                        let dx = (nx as isize) - (x as isize);
                        let sq = (dx * dx + dy * dy) as usize;
                        let luma_diff_abs =
                            (center_luma - neighbour_luma).abs().clamp(0.0, 255.0) as u8;
                        let w = tables.spatial(sq) * tables.range(luma_diff_abs);
                        weight_sum += w;
                        let nbase = nidx * channels;
                        for c in 0..channels {
                            value_sums[c] += w * (src[nbase + c] as f32);
                        }
                    }
                }

                let base = (y * width + x) * channels;
                for c in 0..channels {
                    dst[base + c] = if weight_sum > 0.0 {
                        (value_sums[c] / weight_sum).clamp(0.0, 255.0) as u8
                    } else {
                        src[base + c]
                    };
                }
            }
        }
        Ok(dst)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_dimensions(len: usize, width: usize, height: usize) -> ImageResult<()> {
    if width == 0 || height == 0 {
        return Err(ImageError::InvalidDimensions(width as u32, height as u32));
    }
    if len != width * height {
        return Err(ImageError::InvalidDimensions(width as u32, height as u32));
    }
    Ok(())
}

#[inline]
fn neighbor_diff_f32(a: f32, b: f32) -> f32 {
    a - b
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params_default_radius() {
        let p = BilateralParams::new(2.0, 30.0);
        // radius = ceil(3 * 2.0) = 6
        assert_eq!(p.radius(), 6);
    }

    #[test]
    fn test_params_small_sigma() {
        // sigma_spatial = 0.1 → radius = ceil(0.3) = 1
        let p = BilateralParams::new(0.1, 10.0);
        assert_eq!(p.radius(), 1);
    }

    #[test]
    fn test_apply_u8_uniform_image_unchanged() {
        // Uniform image: bilateral filter must leave every pixel intact (allow ±1 for float rounding).
        let src = vec![100u8; 5 * 5];
        let filter = BilateralFilter::new(BilateralParams::new(1.5, 25.0));
        let dst = filter.apply_u8(&src, 5, 5).expect("should succeed");
        for (i, (&s, &d)) in src.iter().zip(dst.iter()).enumerate() {
            let diff = (s as i32 - d as i32).unsigned_abs();
            assert!(diff <= 1, "pixel {i} changed too much: {s} vs {d}");
        }
    }

    #[test]
    fn test_apply_u8_output_length() {
        let src = vec![128u8; 8 * 8];
        let filter = BilateralFilter::default_params();
        let dst = filter.apply_u8(&src, 8, 8).expect("ok");
        assert_eq!(dst.len(), 64);
    }

    #[test]
    fn test_apply_u8_invalid_dimensions_returns_error() {
        let src = vec![0u8; 10]; // not 4×4 = 16
        let filter = BilateralFilter::default_params();
        assert!(filter.apply_u8(&src, 4, 4).is_err());
    }

    #[test]
    fn test_apply_u8_zero_width_returns_error() {
        let src: Vec<u8> = vec![];
        let filter = BilateralFilter::default_params();
        assert!(filter.apply_u8(&src, 0, 4).is_err());
    }

    #[test]
    fn test_apply_u8_preserves_step_edge() {
        // A step edge (half black, half white): bilateral should keep the
        // center pixels of each half close to their original values.
        let width = 8;
        let height = 1;
        let mut src = vec![0u8; width * height];
        for i in 4..8 {
            src[i] = 255;
        }
        let filter = BilateralFilter::new(BilateralParams::new(2.0, 20.0));
        let dst = filter.apply_u8(&src, width, height).expect("ok");
        // Left half should stay closer to 0 than 128
        assert!(dst[0] < 128, "left edge pixel leaked: {}", dst[0]);
        // Right half should stay closer to 255 than 128
        assert!(dst[7] > 128, "right edge pixel leaked: {}", dst[7]);
    }

    #[test]
    fn test_apply_f32_uniform_unchanged() {
        let src = vec![0.5f32; 6 * 6];
        let filter = BilateralFilter::new(BilateralParams::new(1.5, 0.5));
        let dst = filter.apply_f32(&src, 6, 6).expect("ok");
        for (i, (&s, &d)) in src.iter().zip(dst.iter()).enumerate() {
            assert!((s - d).abs() < 1e-5, "pixel {i}: {s} vs {d}");
        }
    }

    #[test]
    fn test_apply_f32_output_length() {
        let src = vec![0.3f32; 4 * 5];
        let filter = BilateralFilter::default_params();
        let dst = filter.apply_f32(&src, 4, 5).expect("ok");
        assert_eq!(dst.len(), 20);
    }

    #[test]
    fn test_apply_f32_invalid_length_returns_error() {
        let src = vec![0.5f32; 9]; // not 4×4
        let filter = BilateralFilter::default_params();
        assert!(filter.apply_f32(&src, 4, 4).is_err());
    }

    #[test]
    fn test_apply_multichannel_uniform_unchanged() {
        let channels = 3;
        let src = vec![120u8; 4 * 4 * channels];
        let filter = BilateralFilter::new(BilateralParams::new(1.5, 30.0));
        let dst = filter.apply_multichannel(&src, 4, 4, channels).expect("ok");
        for (i, (&s, &d)) in src.iter().zip(dst.iter()).enumerate() {
            let diff = (s as i32 - d as i32).unsigned_abs();
            assert!(diff <= 1, "channel pixel {i} changed too much: {s} vs {d}");
        }
    }

    #[test]
    fn test_apply_multichannel_output_length() {
        let channels = 4;
        let src = vec![200u8; 3 * 3 * channels];
        let filter = BilateralFilter::default_params();
        let dst = filter.apply_multichannel(&src, 3, 3, channels).expect("ok");
        assert_eq!(dst.len(), src.len());
    }

    #[test]
    fn test_apply_multichannel_zero_channels_returns_error() {
        let src = vec![0u8; 16];
        let filter = BilateralFilter::default_params();
        assert!(filter.apply_multichannel(&src, 4, 4, 0).is_err());
    }

    #[test]
    fn test_gaussian_tables_range_zero_delta_is_one() {
        let params = BilateralParams::new(2.0, 30.0);
        let tables = GaussianTables::build(&params);
        // range weight for delta=0 must be exp(0) = 1.0
        assert!((tables.range(0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gaussian_tables_range_decreases() {
        let params = BilateralParams::new(2.0, 30.0);
        let tables = GaussianTables::build(&params);
        // range weight should decrease as delta grows
        assert!(tables.range(10) > tables.range(50));
        assert!(tables.range(50) > tables.range(200));
    }
}
