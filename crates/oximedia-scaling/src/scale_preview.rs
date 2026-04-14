//! Fast low-quality preview scaling before committing to a full-quality scale.
//!
//! When processing large images it can be beneficial to generate a quick
//! preview at a reduced scale so the user can verify composition and colour
//! before the time-consuming full-quality resampling runs.
//!
//! `ScalePreview` generates previews using nearest-neighbour interpolation
//! which is O(n) in the output size and requires no pre-computation.
//!
//! # Example
//!
//! ```
//! use oximedia_scaling::scale_preview::ScalePreview;
//!
//! let preview = ScalePreview::new(0.25); // 1/4 scale preview
//! let src = vec![128u8; 1920 * 1080 * 4];
//! let (pw, ph, pixels) = preview.generate(&src, 1920, 1080);
//! assert_eq!(pw, 480);
//! assert_eq!(ph, 270);
//! assert_eq!(pixels.len(), 480 * 270 * 4);
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

const CHANNELS: usize = 4;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Generates fast nearest-neighbour preview images at a fractional scale.
#[derive(Debug, Clone)]
pub struct ScalePreview {
    /// Scale factor applied to both dimensions (e.g., `0.25` for a quarter-size preview).
    /// Clamped to `(0.0, 1.0]`.
    pub factor: f64,
}

impl ScalePreview {
    /// Create a new preview generator.
    ///
    /// `factor` is the scale factor for the preview relative to the source:
    /// `0.25` produces a quarter-size preview.  Factors ≤ 0 are clamped to a
    /// minimum of `1/65535` to avoid division by zero.
    pub fn new(factor: f64) -> Self {
        Self {
            factor: factor.clamp(1.0 / 65535.0, 1.0),
        }
    }

    /// Generate a preview image.
    ///
    /// # Parameters
    ///
    /// * `src`   — RGBA source pixel buffer (`src_w × src_h × 4` bytes).
    /// * `src_w` — source width.
    /// * `src_h` — source height.
    ///
    /// # Returns
    ///
    /// `(preview_width, preview_height, pixels)` where `pixels` has length
    /// `preview_width × preview_height × 4`.
    pub fn generate(&self, src: &[u8], src_w: u32, src_h: u32) -> (u32, u32, Vec<u8>) {
        if src_w == 0 || src_h == 0 {
            return (0, 0, Vec::new());
        }

        let pw = ((src_w as f64 * self.factor).round() as u32).max(1);
        let ph = ((src_h as f64 * self.factor).round() as u32).max(1);

        let sw = src_w as usize;
        let sh = src_h as usize;
        let pw_u = pw as usize;
        let ph_u = ph as usize;

        let mut out = vec![0u8; pw_u * ph_u * CHANNELS];

        for dy in 0..ph_u {
            let sy = (dy * sh / ph_u).min(sh - 1);
            for dx in 0..pw_u {
                let sx = (dx * sw / pw_u).min(sw - 1);
                let src_off = (sy * sw + sx) * CHANNELS;
                let dst_off = (dy * pw_u + dx) * CHANNELS;
                if src_off + CHANNELS <= src.len() {
                    out[dst_off..dst_off + CHANNELS]
                        .copy_from_slice(&src[src_off..src_off + CHANNELS]);
                }
            }
        }

        (pw, ph, out)
    }

    /// Return the preview dimensions for a given source size without generating pixels.
    pub fn preview_size(&self, src_w: u32, src_h: u32) -> (u32, u32) {
        if src_w == 0 || src_h == 0 {
            return (0, 0);
        }
        let pw = ((src_w as f64 * self.factor).round() as u32).max(1);
        let ph = ((src_h as f64 * self.factor).round() as u32).max(1);
        (pw, ph)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quarter_scale_dimensions() {
        let p = ScalePreview::new(0.25);
        let src = vec![0u8; 1920 * 1080 * 4];
        let (pw, ph, pixels) = p.generate(&src, 1920, 1080);
        assert_eq!(pw, 480);
        assert_eq!(ph, 270);
        assert_eq!(pixels.len(), 480 * 270 * 4);
    }

    #[test]
    fn test_full_scale_identity() {
        let p = ScalePreview::new(1.0);
        let src: Vec<u8> = (0u8..=255).cycle().take(8 * 8 * 4).collect();
        let (pw, ph, pixels) = p.generate(&src, 8, 8);
        assert_eq!(pw, 8);
        assert_eq!(ph, 8);
        assert_eq!(pixels.len(), src.len());
    }

    #[test]
    fn test_zero_src_returns_empty() {
        let p = ScalePreview::new(0.5);
        let (pw, ph, pixels) = p.generate(&[], 0, 0);
        assert_eq!(pw, 0);
        assert_eq!(ph, 0);
        assert!(pixels.is_empty());
    }

    #[test]
    fn test_preview_size_method() {
        let p = ScalePreview::new(0.5);
        let (pw, ph) = p.preview_size(100, 80);
        assert_eq!(pw, 50);
        assert_eq!(ph, 40);
    }

    #[test]
    fn test_factor_clamped_above_one() {
        // factor > 1.0 should be clamped to 1.0
        let p = ScalePreview::new(2.0);
        assert!((p.factor - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_factor_clamped_at_zero() {
        let p = ScalePreview::new(0.0);
        assert!(p.factor > 0.0, "factor must be positive");
    }

    #[test]
    fn test_pixel_content_preserved_uniform() {
        // A uniform-color image should produce uniform output at any preview scale.
        let fill = 200u8;
        let src = vec![fill; 16 * 16 * 4];
        let p = ScalePreview::new(0.5);
        let (pw, ph, pixels) = p.generate(&src, 16, 16);
        assert_eq!(pw, 8);
        assert_eq!(ph, 8);
        for &byte in &pixels {
            assert_eq!(
                byte, fill,
                "pixel value should be preserved for uniform image"
            );
        }
    }

    #[test]
    fn test_preview_size_zero_src() {
        let p = ScalePreview::new(0.5);
        let (pw, ph) = p.preview_size(0, 100);
        assert_eq!((pw, ph), (0, 0));
        let (pw2, ph2) = p.preview_size(100, 0);
        assert_eq!((pw2, ph2), (0, 0));
    }

    #[test]
    fn test_small_factor_produces_minimum_one_pixel() {
        // Even with a tiny factor, output must be at least 1×1.
        let p = ScalePreview::new(0.001);
        let (pw, ph, pixels) = p.generate(&[128u8; 10 * 10 * 4], 10, 10);
        assert!(pw >= 1, "width must be at least 1");
        assert!(ph >= 1, "height must be at least 1");
        assert!(!pixels.is_empty());
    }
}
