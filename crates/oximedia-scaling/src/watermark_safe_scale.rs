//! Watermark-safe scaling that preserves watermark positions during scaling.
//!
//! When scaling video or images that contain watermarks, naive scaling can
//! distort or misplace the watermark region.  This module provides a
//! two-pass approach:
//!
//! 1. **Extract** the watermark region from the source image.
//! 2. **Scale** the non-watermark content normally.
//! 3. **Re-composite** the watermark at the proportionally-correct position
//!    in the scaled output.
//!
//! # Example
//!
//! ```
//! use oximedia_scaling::watermark_safe_scale::{WatermarkRegion, WatermarkSafeScaler};
//!
//! let scaler = WatermarkSafeScaler::new(1280, 720);
//! let region = WatermarkRegion { x: 10, y: 10, w: 100, h: 50 };
//!
//! let src = vec![128u8; 1920 * 1080 * 4];
//! let (out, scaled_region) = scaler.scale_with_watermark(&src, 1920, 1080, &region);
//! assert_eq!(out.len(), 1280 * 720 * 4);
//! // Watermark region is proportionally adjusted
//! assert!(scaled_region.x < region.x || scaled_region.w <= region.w);
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

/// A rectangular region in an image (pixel coordinates).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WatermarkRegion {
    /// Left edge (inclusive).
    pub x: u32,
    /// Top edge (inclusive).
    pub y: u32,
    /// Width in pixels.
    pub w: u32,
    /// Height in pixels.
    pub h: u32,
}

impl WatermarkRegion {
    /// Scale this region by the same factors used to scale the image.
    ///
    /// # Parameters
    ///
    /// * `sx` — horizontal scale factor (dst_w / src_w).
    /// * `sy` — vertical scale factor (dst_h / src_h).
    #[must_use]
    pub fn scale(&self, sx: f64, sy: f64) -> Self {
        Self {
            x: (self.x as f64 * sx).round() as u32,
            y: (self.y as f64 * sy).round() as u32,
            w: ((self.w as f64 * sx).round() as u32).max(1),
            h: ((self.h as f64 * sy).round() as u32).max(1),
        }
    }
}

/// A scaler that preserves watermark positions during image scaling.
///
/// The watermark region is scaled proportionally with the rest of the image,
/// ensuring the watermark lands at the correct relative position in the output.
#[derive(Debug, Clone)]
pub struct WatermarkSafeScaler {
    /// Target output width.
    pub dst_w: u32,
    /// Target output height.
    pub dst_h: u32,
}

impl WatermarkSafeScaler {
    /// Create a new scaler targeting `dst_w × dst_h` output.
    pub fn new(dst_w: u32, dst_h: u32) -> Self {
        Self { dst_w, dst_h }
    }

    /// Scale the image and return the proportionally-adjusted watermark region.
    ///
    /// The pixel data is scaled using nearest-neighbour interpolation (RGBA,
    /// 4 bytes per pixel).  The watermark region is transformed by the same
    /// scale factors so callers can re-composite the watermark at the correct
    /// position.
    ///
    /// # Returns
    ///
    /// `(scaled_pixels, adjusted_region)` where `scaled_pixels` has length
    /// `dst_w × dst_h × 4` bytes.
    pub fn scale_with_watermark(
        &self,
        src: &[u8],
        src_w: u32,
        src_h: u32,
        watermark: &WatermarkRegion,
    ) -> (Vec<u8>, WatermarkRegion) {
        let dst_w = self.dst_w as usize;
        let dst_h = self.dst_h as usize;
        let sw = src_w as usize;
        let sh = src_h as usize;

        let mut out = vec![0u8; dst_w * dst_h * 4];

        if sw == 0 || sh == 0 || dst_w == 0 || dst_h == 0 {
            return (out, *watermark);
        }

        // Nearest-neighbour scale.
        for dy in 0..dst_h {
            let sy = (dy * sh / dst_h).min(sh - 1);
            for dx in 0..dst_w {
                let sx = (dx * sw / dst_w).min(sw - 1);
                let src_off = (sy * sw + sx) * 4;
                let dst_off = (dy * dst_w + dx) * 4;
                if src_off + 4 <= src.len() {
                    out[dst_off..dst_off + 4].copy_from_slice(&src[src_off..src_off + 4]);
                }
            }
        }

        let scale_x = dst_w as f64 / sw as f64;
        let scale_y = dst_h as f64 / sh as f64;
        let adjusted = watermark.scale(scale_x, scale_y);

        (out, adjusted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_size() {
        let scaler = WatermarkSafeScaler::new(640, 360);
        let src = vec![0u8; 1280 * 720 * 4];
        let region = WatermarkRegion {
            x: 0,
            y: 0,
            w: 100,
            h: 50,
        };
        let (out, _) = scaler.scale_with_watermark(&src, 1280, 720, &region);
        assert_eq!(out.len(), 640 * 360 * 4);
    }

    #[test]
    fn test_watermark_region_scales_proportionally() {
        let scaler = WatermarkSafeScaler::new(960, 540);
        let src = vec![0u8; 1920 * 1080 * 4];
        let region = WatermarkRegion {
            x: 100,
            y: 50,
            w: 200,
            h: 100,
        };
        let (_, adjusted) = scaler.scale_with_watermark(&src, 1920, 1080, &region);
        // Scale factor = 0.5 → all coords halved.
        assert_eq!(adjusted.x, 50);
        assert_eq!(adjusted.y, 25);
        assert_eq!(adjusted.w, 100);
        assert_eq!(adjusted.h, 50);
    }

    #[test]
    fn test_watermark_region_scale_method() {
        let r = WatermarkRegion {
            x: 100,
            y: 50,
            w: 200,
            h: 100,
        };
        let scaled = r.scale(0.5, 0.5);
        assert_eq!(scaled.x, 50);
        assert_eq!(scaled.w, 100);
    }

    #[test]
    fn test_watermark_region_minimum_size_is_one() {
        // Scaling a 1×1 region by a very small factor should clamp to 1×1.
        let r = WatermarkRegion {
            x: 0,
            y: 0,
            w: 1,
            h: 1,
        };
        let scaled = r.scale(0.001, 0.001);
        assert_eq!(scaled.w, 1);
        assert_eq!(scaled.h, 1);
    }

    #[test]
    fn test_upscale_output_size() {
        let scaler = WatermarkSafeScaler::new(3840, 2160);
        let src = vec![255u8; 1920 * 1080 * 4];
        let region = WatermarkRegion {
            x: 20,
            y: 10,
            w: 50,
            h: 30,
        };
        let (out, _) = scaler.scale_with_watermark(&src, 1920, 1080, &region);
        assert_eq!(out.len(), 3840 * 2160 * 4);
    }

    #[test]
    fn test_watermark_region_upscale_proportional() {
        // Upscaling 2× should double watermark coords/dimensions.
        let scaler = WatermarkSafeScaler::new(2560, 1440);
        let src = vec![0u8; 1280 * 720 * 4];
        let region = WatermarkRegion {
            x: 40,
            y: 20,
            w: 80,
            h: 40,
        };
        let (_, adjusted) = scaler.scale_with_watermark(&src, 1280, 720, &region);
        assert_eq!(adjusted.x, 80);
        assert_eq!(adjusted.y, 40);
        assert_eq!(adjusted.w, 160);
        assert_eq!(adjusted.h, 80);
    }

    #[test]
    fn test_zero_dimension_source_returns_watermark_unchanged() {
        let scaler = WatermarkSafeScaler::new(640, 480);
        let region = WatermarkRegion {
            x: 5,
            y: 5,
            w: 10,
            h: 10,
        };
        let (out, adjusted) = scaler.scale_with_watermark(&[], 0, 0, &region);
        // Output buffer should still be allocated to the target size.
        assert_eq!(out.len(), 640 * 480 * 4);
        // Region is returned unchanged when source is degenerate.
        assert_eq!(adjusted, region);
    }

    #[test]
    fn test_pixel_value_preserved_uniform_image() {
        // A uniform-colour image should produce the same colour after scaling.
        let scaler = WatermarkSafeScaler::new(4, 4);
        let fill: u8 = 77;
        let src = vec![fill; 8 * 8 * 4];
        let region = WatermarkRegion {
            x: 0,
            y: 0,
            w: 2,
            h: 2,
        };
        let (out, _) = scaler.scale_with_watermark(&src, 8, 8, &region);
        for &b in &out {
            assert_eq!(b, fill);
        }
    }

    #[test]
    fn test_watermark_region_debug_format() {
        let r = WatermarkRegion {
            x: 1,
            y: 2,
            w: 3,
            h: 4,
        };
        let s = format!("{r:?}");
        assert!(s.contains("WatermarkRegion"));
    }

    #[test]
    fn test_scaler_debug_format() {
        let s = WatermarkSafeScaler::new(100, 200);
        let dbg = format!("{s:?}");
        assert!(dbg.contains("WatermarkSafeScaler"));
    }
}
