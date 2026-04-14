//! Downsampled analysis for real-time preview at reduced accuracy.
//!
//! Real-time scope monitoring at full resolution (e.g. 4K 60fps) can be
//! prohibitively expensive.  This module provides configurable spatial
//! downsampling so that scopes can be updated every frame at the cost of
//! reduced spatial accuracy while preserving statistical fidelity (mean,
//! peak, histogram shape).
//!
//! # Strategy
//!
//! The frame is divided into an evenly-spaced grid of sample points.  Only
//! those sample points are processed by the scope pipeline.  The sampling
//! density is controlled by [`DownsampleFactor`]:
//!
//! | Factor | Pixels processed | Relative speed |
//! |--------|-----------------|----------------|
//! | `Full` | 100 % | 1× |
//! | `Half` |  25 % (every 2nd in x, 2nd in y) | ~4× |
//! | `Quarter` | ~6.25 % | ~16× |
//! | `Eighth` | ~1.56 % | ~64× |
//! | `Custom(n)` | 1/n² | n²× |
//!
//! # Example
//!
//! ```
//! use oximedia_scopes::preview_downsample::{DownsampleFactor, downsample_frame};
//!
//! let frame = vec![100u8, 150, 200, 50, 75, 100]; // 2 pixels (RGB)
//! let result = downsample_frame(&frame, 2, 1, DownsampleFactor::Full).unwrap();
//! assert_eq!(result.pixels.len(), 2 * 3); // 2 pixels × 3 bytes
//! ```

use oximedia_core::{OxiError, OxiResult};

/// Spatial downsampling factor for real-time preview mode.
///
/// `Custom(n)` processes every n-th pixel in both X and Y,
/// giving a 1/n² sample density.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownsampleFactor {
    /// No downsampling — process every pixel.
    Full,
    /// Process every 2nd pixel (in both axes) → 25 % sample density.
    Half,
    /// Process every 4th pixel → ~6.25 % sample density.
    Quarter,
    /// Process every 8th pixel → ~1.56 % sample density.
    Eighth,
    /// User-specified step. Values ≥ 1 are valid; 1 == `Full`.
    Custom(u32),
}

impl DownsampleFactor {
    /// Returns the pixel step in each spatial dimension.
    #[must_use]
    pub fn step(self) -> u32 {
        match self {
            Self::Full => 1,
            Self::Half => 2,
            Self::Quarter => 4,
            Self::Eighth => 8,
            Self::Custom(n) => n.max(1),
        }
    }

    /// Approximate fraction of the original pixels that will be sampled.
    #[must_use]
    pub fn sample_fraction(self) -> f64 {
        let s = f64::from(self.step());
        1.0 / (s * s)
    }
}

impl Default for DownsampleFactor {
    fn default() -> Self {
        Self::Half
    }
}

/// The result of downsampling a frame.
#[derive(Debug, Clone)]
pub struct DownsampledFrame {
    /// Downsampled pixel data (RGB-interleaved, row-major).
    pub pixels: Vec<u8>,
    /// Width of the downsampled frame in pixels.
    pub width: u32,
    /// Height of the downsampled frame in pixels.
    pub height: u32,
    /// Step used in the X direction.
    pub step_x: u32,
    /// Step used in the Y direction.
    pub step_y: u32,
    /// Original frame width.
    pub orig_width: u32,
    /// Original frame height.
    pub orig_height: u32,
}

impl DownsampledFrame {
    /// Total number of sampled pixels.
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        (self.width * self.height) as usize
    }

    /// Actual sampling fraction relative to the original frame.
    #[must_use]
    pub fn actual_fraction(&self) -> f64 {
        let orig = (self.orig_width as f64) * (self.orig_height as f64);
        if orig < 1.0 {
            return 0.0;
        }
        self.pixel_count() as f64 / orig
    }
}

/// Downsample an RGB-24 frame by uniformly skipping pixels.
///
/// Each retained pixel uses the nearest-neighbour value; no blending is
/// performed so the function is O(output_pixels) and cache-friendly.
///
/// # Arguments
///
/// * `frame`  – RGB-interleaved source buffer (`width * height * 3` bytes).
/// * `width`  – Source frame width in pixels.
/// * `height` – Source frame height in pixels.
/// * `factor` – Spatial downsampling factor.
///
/// # Errors
///
/// Returns [`OxiError::InvalidData`] if `frame` is shorter than
/// `width * height * 3` bytes or if either dimension is zero.
pub fn downsample_frame(
    frame: &[u8],
    width: u32,
    height: u32,
    factor: DownsampleFactor,
) -> OxiResult<DownsampledFrame> {
    if width == 0 || height == 0 {
        return Err(OxiError::InvalidData("Frame dimensions must be non-zero".into()));
    }
    let expected = (width as usize) * (height as usize) * 3;
    if frame.len() < expected {
        return Err(OxiError::InvalidData(format!(
            "Frame too small: need {expected} bytes, got {}",
            frame.len()
        )));
    }

    let step = factor.step();
    let out_w = (width + step - 1) / step;
    let out_h = (height + step - 1) / step;

    let mut pixels = Vec::with_capacity((out_w * out_h * 3) as usize);

    let mut sy = 0u32;
    while sy < height {
        let row_off = (sy * width) as usize * 3;
        let mut sx = 0u32;
        while sx < width {
            let off = row_off + sx as usize * 3;
            pixels.push(frame[off]);
            pixels.push(frame[off + 1]);
            pixels.push(frame[off + 2]);
            sx += step;
        }
        sy += step;
    }

    Ok(DownsampledFrame {
        pixels,
        width: out_w,
        height: out_h,
        step_x: step,
        step_y: step,
        orig_width: width,
        orig_height: height,
    })
}

/// Adaptive downsampling: choose the coarsest factor that keeps the
/// number of output pixels at or below `max_pixels`.
///
/// Returns `DownsampleFactor::Full` when the full frame is already within
/// budget. The candidates are evaluated in order from finest to coarsest
/// so that the finest acceptable factor is returned.
///
/// # Examples
///
/// ```
/// use oximedia_scopes::preview_downsample::{DownsampleFactor, adaptive_factor};
/// // 1920×1080 = 2 073 600 pixels → needs downsampling for a 500k budget
/// let f = adaptive_factor(1920, 1080, 500_000);
/// assert!(f.step() >= 2);
/// ```
#[must_use]
pub fn adaptive_factor(width: u32, height: u32, max_pixels: u32) -> DownsampleFactor {
    let candidates = [
        DownsampleFactor::Full,
        DownsampleFactor::Half,
        DownsampleFactor::Quarter,
        DownsampleFactor::Eighth,
    ];

    for &factor in &candidates {
        let s = factor.step();
        let out_w = (width + s - 1) / s;
        let out_h = (height + s - 1) / s;
        if out_w * out_h <= max_pixels {
            return factor;
        }
    }

    // Fall back to Custom with the minimum step that fits.
    let total = (width as u64) * (height as u64);
    let step = ((total as f64 / max_pixels as f64).sqrt().ceil() as u32).max(1);
    DownsampleFactor::Custom(step)
}

/// Configuration for preview-mode analysis.
#[derive(Debug, Clone)]
pub struct PreviewConfig {
    /// Maximum pixels to process per frame (budget).
    pub max_pixels: u32,
    /// Fixed downsampling factor, or `None` to use adaptive selection.
    pub fixed_factor: Option<DownsampleFactor>,
}

impl Default for PreviewConfig {
    fn default() -> Self {
        // Default budget: ≤ 256 × 256 = 65 536 pixels → smooth real-time.
        Self {
            max_pixels: 65_536,
            fixed_factor: None,
        }
    }
}

impl PreviewConfig {
    /// Construct a new `PreviewConfig` with a specific fixed factor.
    #[must_use]
    pub fn with_factor(factor: DownsampleFactor) -> Self {
        Self {
            fixed_factor: Some(factor),
            ..Self::default()
        }
    }

    /// Construct a `PreviewConfig` with adaptive selection and a custom budget.
    #[must_use]
    pub fn with_budget(max_pixels: u32) -> Self {
        Self {
            max_pixels,
            fixed_factor: None,
        }
    }

    /// Resolve the actual downsampling factor for a given frame size.
    #[must_use]
    pub fn resolve_factor(&self, width: u32, height: u32) -> DownsampleFactor {
        if let Some(f) = self.fixed_factor {
            f
        } else {
            adaptive_factor(width, height, self.max_pixels)
        }
    }
}

/// Downsample a frame using the settings from a [`PreviewConfig`].
///
/// # Errors
///
/// Propagates errors from [`downsample_frame`].
pub fn downsample_for_preview(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &PreviewConfig,
) -> OxiResult<DownsampledFrame> {
    let factor = config.resolve_factor(width, height);
    downsample_frame(frame, width, height, factor)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_frame(r: u8, g: u8, b: u8, width: u32, height: u32) -> Vec<u8> {
        let n = (width * height) as usize;
        let mut v = Vec::with_capacity(n * 3);
        for _ in 0..n {
            v.extend_from_slice(&[r, g, b]);
        }
        v
    }

    #[test]
    fn test_full_factor_preserves_all_pixels() {
        let frame = solid_frame(100, 150, 200, 4, 4);
        let ds = downsample_frame(&frame, 4, 4, DownsampleFactor::Full).expect("ok");
        assert_eq!(ds.width, 4);
        assert_eq!(ds.height, 4);
        assert_eq!(ds.pixel_count(), 16);
        assert_eq!(ds.pixels, frame);
    }

    #[test]
    fn test_half_factor_reduces_to_quarter() {
        let frame = solid_frame(10, 20, 30, 8, 8);
        let ds = downsample_frame(&frame, 8, 8, DownsampleFactor::Half).expect("ok");
        assert_eq!(ds.width, 4);
        assert_eq!(ds.height, 4);
        assert_eq!(ds.pixel_count(), 16);
        // Every sampled pixel should still be (10, 20, 30).
        for chunk in ds.pixels.chunks_exact(3) {
            assert_eq!(chunk, [10, 20, 30]);
        }
    }

    #[test]
    fn test_quarter_factor() {
        let frame = solid_frame(255, 0, 128, 8, 8);
        let ds = downsample_frame(&frame, 8, 8, DownsampleFactor::Quarter).expect("ok");
        assert_eq!(ds.width, 2);
        assert_eq!(ds.height, 2);
        assert_eq!(ds.pixel_count(), 4);
    }

    #[test]
    fn test_custom_factor_step() {
        let f = DownsampleFactor::Custom(3);
        assert_eq!(f.step(), 3);
    }

    #[test]
    fn test_custom_factor_zero_clamped_to_one() {
        let f = DownsampleFactor::Custom(0);
        assert_eq!(f.step(), 1); // clamped to 1 == Full
    }

    #[test]
    fn test_invalid_zero_dimensions() {
        let result = downsample_frame(&[], 0, 10, DownsampleFactor::Full);
        assert!(result.is_err());
    }

    #[test]
    fn test_insufficient_frame_data() {
        let frame = vec![0u8; 5]; // not enough for 4×4×3
        let result = downsample_frame(&frame, 4, 4, DownsampleFactor::Full);
        assert!(result.is_err());
    }

    #[test]
    fn test_adaptive_factor_full_resolution_within_budget() {
        // 4×4 = 16 pixels, budget = 100 → Full should be chosen.
        let f = adaptive_factor(4, 4, 100);
        assert_eq!(f.step(), 1);
    }

    #[test]
    fn test_adaptive_factor_hd_exceeds_budget() {
        // 1920×1080 >> 65 536 → step should be ≥ 2.
        let f = adaptive_factor(1920, 1080, 65_536);
        assert!(f.step() >= 2, "expected step>=2, got {}", f.step());
    }

    #[test]
    fn test_preview_config_with_factor() {
        let cfg = PreviewConfig::with_factor(DownsampleFactor::Quarter);
        let resolved = cfg.resolve_factor(1920, 1080);
        assert_eq!(resolved, DownsampleFactor::Quarter);
    }

    #[test]
    fn test_preview_config_adaptive_budget() {
        let cfg = PreviewConfig::with_budget(16);
        let factor = cfg.resolve_factor(16, 16);
        // 16×16 = 256 pixels, budget 16 → needs step ≥ 4.
        assert!(factor.step() >= 4, "step={}", factor.step());
    }

    #[test]
    fn test_downsample_for_preview_matches_direct() {
        let frame = solid_frame(77, 88, 99, 4, 4);
        let cfg = PreviewConfig::with_factor(DownsampleFactor::Half);
        let via_config =
            downsample_for_preview(&frame, 4, 4, &cfg).expect("ok");
        let direct =
            downsample_frame(&frame, 4, 4, DownsampleFactor::Half).expect("ok");
        assert_eq!(via_config.pixels, direct.pixels);
    }

    #[test]
    fn test_actual_fraction_full() {
        let frame = solid_frame(0, 0, 0, 8, 8);
        let ds = downsample_frame(&frame, 8, 8, DownsampleFactor::Full).expect("ok");
        let frac = ds.actual_fraction();
        assert!((frac - 1.0).abs() < 1e-9, "frac={frac}");
    }

    #[test]
    fn test_sample_fraction_half() {
        let frac = DownsampleFactor::Half.sample_fraction();
        assert!((frac - 0.25).abs() < 1e-9);
    }
}
