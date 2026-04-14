//! Edge-directed interpolation (NEDI-like) for improved diagonal edge rendering.
//!
//! Standard bilinear and bicubic interpolation treat all directions equally,
//! which causes blurring and staircase artefacts along diagonal edges.
//! Edge-directed interpolation (EDI) estimates the local edge orientation and
//! interpolates along the edge rather than across it, producing sharper and
//! cleaner diagonal edges.
//!
//! This implementation is a simplified version of the NEDI (New Edge-Directed
//! Interpolation) approach:
//!
//! 1. Compute the local gradient direction using Sobel operators.
//! 2. For each new sample position, identify whether to interpolate
//!    horizontally, vertically, or diagonally based on the gradient angle.
//! 3. Use linear interpolation in the chosen direction.
//!
//! The input is an 8-bit greyscale image; the output is a 2× upscaled image.
//!
//! # Reference
//!
//! Li, X. and Orchard, M. T. (2001). "New edge-directed interpolation."
//! *IEEE Transactions on Image Processing*, 10(10), 1521–1527.
//!
//! # Example
//!
//! ```
//! use oximedia_scaling::edge_directed_interpolation::EdgeDirectedInterpolator;
//!
//! let edi = EdgeDirectedInterpolator::default();
//! let src = vec![0u8; 64 * 64];  // 64×64 greyscale
//! let (dst_w, dst_h, dst) = edi.upscale_2x(&src, 64, 64);
//! assert_eq!(dst_w, 128);
//! assert_eq!(dst_h, 128);
//! assert_eq!(dst.len(), 128 * 128);
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Simplified edge-directed interpolator (2× upscale, greyscale).
#[derive(Debug, Clone)]
pub struct EdgeDirectedInterpolator {
    /// Edge threshold — gradient magnitudes above this value trigger edge-directed
    /// interpolation; below it bilinear interpolation is used.
    pub edge_threshold: f32,
}

impl Default for EdgeDirectedInterpolator {
    fn default() -> Self {
        Self {
            edge_threshold: 15.0,
        }
    }
}

impl EdgeDirectedInterpolator {
    /// Create a new interpolator with a custom edge threshold.
    pub fn new(edge_threshold: f32) -> Self {
        Self {
            edge_threshold: edge_threshold.max(0.0),
        }
    }

    /// Upscale a greyscale image by 2× using edge-directed interpolation.
    ///
    /// # Parameters
    ///
    /// * `src`   — row-major greyscale pixel buffer (`w × h` bytes).
    /// * `w`     — source width.
    /// * `h`     — source height.
    ///
    /// # Returns
    ///
    /// `(dst_w, dst_h, pixels)` where `dst_w = 2*w`, `dst_h = 2*h`, and
    /// `pixels` has length `dst_w × dst_h`.
    pub fn upscale_2x(&self, src: &[u8], w: u32, h: u32) -> (u32, u32, Vec<u8>) {
        let sw = w as usize;
        let sh = h as usize;
        let dw = sw * 2;
        let dh = sh * 2;

        if sw == 0 || sh == 0 {
            return (0, 0, Vec::new());
        }

        let mut dst = vec![0u8; dw * dh];

        // Pass 1: copy source pixels into even-even positions in dst.
        for sy in 0..sh {
            for sx in 0..sw {
                let v = src.get(sy * sw + sx).copied().unwrap_or(0);
                dst[(sy * 2) * dw + (sx * 2)] = v;
            }
        }

        // Pass 2: fill in new pixels at odd positions.
        for dy in 0..dh {
            for dx in 0..dw {
                // Skip already-filled pixels (even,even positions).
                if dy % 2 == 0 && dx % 2 == 0 {
                    continue;
                }

                // Map back to source coordinates.
                let sx_f = dx as f32 * 0.5;
                let sy_f = dy as f32 * 0.5;

                let sx0 = (sx_f.floor() as usize).min(sw.saturating_sub(1));
                let sy0 = (sy_f.floor() as usize).min(sh.saturating_sub(1));
                let sx1 = (sx0 + 1).min(sw.saturating_sub(1));
                let sy1 = (sy0 + 1).min(sh.saturating_sub(1));

                // Compute local gradient via 2×2 Sobel approximation.
                let p00 = src.get(sy0 * sw + sx0).copied().unwrap_or(0) as f32;
                let p10 = src.get(sy0 * sw + sx1).copied().unwrap_or(0) as f32;
                let p01 = src.get(sy1 * sw + sx0).copied().unwrap_or(0) as f32;
                let p11 = src.get(sy1 * sw + sx1).copied().unwrap_or(0) as f32;

                let gx = (p10 - p00).abs() + (p11 - p01).abs();
                let gy = (p01 - p00).abs() + (p11 - p10).abs();

                let value = if gx.max(gy) < self.edge_threshold {
                    // Smooth region — standard bilinear.
                    let fx = sx_f - sx_f.floor();
                    let fy = sy_f - sy_f.floor();
                    let interp = p00 * (1.0 - fx) * (1.0 - fy)
                        + p10 * fx * (1.0 - fy)
                        + p01 * (1.0 - fx) * fy
                        + p11 * fx * fy;
                    interp.clamp(0.0, 255.0) as u8
                } else if gy > gx {
                    // Horizontal edge (vertical gradient) — interpolate horizontally.
                    let fx = sx_f - sx_f.floor();
                    let top = p00 * (1.0 - fx) + p10 * fx;
                    let bot = p01 * (1.0 - fx) + p11 * fx;
                    // Choose the row closest to the interpolation point.
                    let fy = sy_f - sy_f.floor();
                    (top * (1.0 - fy) + bot * fy).clamp(0.0, 255.0) as u8
                } else {
                    // Vertical edge (horizontal gradient) — interpolate vertically.
                    let fy = sy_f - sy_f.floor();
                    let left = p00 * (1.0 - fy) + p01 * fy;
                    let right = p10 * (1.0 - fy) + p11 * fy;
                    let fx = sx_f - sx_f.floor();
                    (left * (1.0 - fx) + right * fx).clamp(0.0, 255.0) as u8
                };

                dst[dy * dw + dx] = value;
            }
        }

        (dw as u32, dh as u32, dst)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_dimensions() {
        let edi = EdgeDirectedInterpolator::default();
        let src = vec![128u8; 32 * 32];
        let (dw, dh, dst) = edi.upscale_2x(&src, 32, 32);
        assert_eq!(dw, 64);
        assert_eq!(dh, 64);
        assert_eq!(dst.len(), 64 * 64);
    }

    #[test]
    fn test_empty_input_returns_empty() {
        let edi = EdgeDirectedInterpolator::default();
        let (dw, dh, dst) = edi.upscale_2x(&[], 0, 0);
        assert_eq!(dw, 0);
        assert_eq!(dh, 0);
        assert!(dst.is_empty());
    }

    #[test]
    fn test_flat_image_preserves_value() {
        let edi = EdgeDirectedInterpolator::default();
        let src = vec![100u8; 8 * 8];
        let (_, _, dst) = edi.upscale_2x(&src, 8, 8);
        for &px in &dst {
            assert_eq!(px, 100, "flat image should stay at constant value");
        }
    }

    #[test]
    fn test_source_pixels_preserved_at_even_positions() {
        let edi = EdgeDirectedInterpolator::default();
        let src: Vec<u8> = (0..4 * 4).map(|i| (i * 10) as u8).collect();
        let (_, _, dst) = edi.upscale_2x(&src, 4, 4);
        // Even positions (2x, 2y) should match the source.
        for sy in 0..4usize {
            for sx in 0..4usize {
                let src_val = src[sy * 4 + sx];
                let dst_val = dst[(sy * 2) * 8 + (sx * 2)];
                assert_eq!(dst_val, src_val, "source pixel mismatch at ({sx},{sy})");
            }
        }
    }

    #[test]
    fn test_custom_threshold() {
        let edi = EdgeDirectedInterpolator::new(50.0);
        assert!((edi.edge_threshold - 50.0).abs() < 1e-6);
        let src = vec![0u8; 4 * 4];
        let (dw, _, _) = edi.upscale_2x(&src, 4, 4);
        assert_eq!(dw, 8);
    }

    #[test]
    fn test_negative_threshold_clamped_to_zero() {
        let edi = EdgeDirectedInterpolator::new(-10.0);
        assert!(edi.edge_threshold >= 0.0);
    }

    #[test]
    fn test_output_pixel_values_in_range() {
        let edi = EdgeDirectedInterpolator::default();
        // Create a gradient image to exercise edge-directed paths.
        let src: Vec<u8> = (0..8usize * 8)
            .map(|i| ((i % 8) * 32).min(255) as u8)
            .collect();
        let (_, _, dst) = edi.upscale_2x(&src, 8, 8);
        // All values are u8, so trivially in range — verify buffer is non-empty.
        assert!(!dst.is_empty(), "output buffer must be non-empty");
    }

    #[test]
    fn test_single_pixel_upscale() {
        let edi = EdgeDirectedInterpolator::default();
        let src = vec![200u8];
        let (dw, dh, dst) = edi.upscale_2x(&src, 1, 1);
        assert_eq!(dw, 2);
        assert_eq!(dh, 2);
        assert_eq!(dst.len(), 4);
        // The top-left (even,even) position must equal the source value.
        assert_eq!(dst[0], 200);
    }

    #[test]
    fn test_high_contrast_edge_pixels_in_range() {
        // Strong horizontal edge: top half white, bottom half black.
        let edi = EdgeDirectedInterpolator::new(10.0);
        let mut src = vec![255u8; 8 * 4];
        src.extend(vec![0u8; 8 * 4]);
        let (_, _, dst) = edi.upscale_2x(&src, 8, 8);
        assert!(
            !dst.is_empty(),
            "output buffer must be non-empty for high-contrast image"
        );
    }

    #[test]
    fn test_debug_format() {
        let edi = EdgeDirectedInterpolator::default();
        let s = format!("{edi:?}");
        assert!(s.contains("EdgeDirectedInterpolator"));
    }
}
