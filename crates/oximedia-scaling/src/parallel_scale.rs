//! Rayon-parallelised row processing for multi-core image scaling.
//!
//! The `ParallelScaler` distributes horizontal and vertical resampling
//! across available CPU cores using Rayon.  Each output row is computed
//! independently, making horizontal resampling embarrassingly parallel.
//! For vertical resampling a column-stripe approach is used so that each
//! thread processes a contiguous range of columns.
//!
//! This module operates on single-channel greyscale `f32` images stored
//! row-major.  Multi-channel images can be handled by processing each
//! channel independently.
//!
//! # Example
//!
//! ```
//! use oximedia_scaling::parallel_scale::ParallelScaler;
//!
//! let scaler = ParallelScaler::new(2, 2, 4, 4);
//! let src = vec![1.0f32; 2 * 2];
//! let dst = scaler.scale_bilinear(&src).expect("scale ok");
//! assert_eq!(dst.len(), 4 * 4);
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use rayon::prelude::*;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during parallel scaling.
#[derive(Debug, Error, PartialEq)]
pub enum ParallelScaleError {
    /// Source buffer has fewer elements than `src_w × src_h`.
    #[error("source buffer too small: expected {expected}, got {actual}")]
    BufferTooSmall {
        /// Expected minimum length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },
    /// One or more dimensions are zero.
    #[error("zero dimension in {dim}")]
    ZeroDimension {
        /// Which dimension is zero.
        dim: &'static str,
    },
}

// ---------------------------------------------------------------------------
// Interpolation kernels
// ---------------------------------------------------------------------------

/// Bilinear weight for a fractional position `t` in `[0, 1)`.
#[inline]
fn bilinear_weight(t: f32) -> (f32, f32) {
    (1.0 - t, t)
}

/// Cubic (Catmull-Rom, a = -0.5) 4-tap weights.
fn cubic_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    let a = -0.5f32;
    [
        a * t3 - 2.0 * a * t2 + a * t,
        (2.0 + a) * t3 - (3.0 + a) * t2 + 1.0,
        -(2.0 + a) * t3 + (3.0 + 2.0 * a) * t2 - a * t,
        -a * t3 + a * t2,
    ]
}

// ---------------------------------------------------------------------------
// ParallelScaler
// ---------------------------------------------------------------------------

/// A parallel image scaler using Rayon for row-level concurrency.
#[derive(Debug, Clone)]
pub struct ParallelScaler {
    /// Source width.
    pub src_w: usize,
    /// Source height.
    pub src_h: usize,
    /// Destination width.
    pub dst_w: usize,
    /// Destination height.
    pub dst_h: usize,
}

impl ParallelScaler {
    /// Create a new parallel scaler.
    pub fn new(src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Self {
        Self {
            src_w,
            src_h,
            dst_w,
            dst_h,
        }
    }

    /// Validate inputs and return the source pixel count.
    fn validate(&self, src: &[f32]) -> Result<(), ParallelScaleError> {
        if self.src_w == 0 || self.src_h == 0 {
            return Err(ParallelScaleError::ZeroDimension { dim: "source" });
        }
        if self.dst_w == 0 || self.dst_h == 0 {
            return Err(ParallelScaleError::ZeroDimension { dim: "destination" });
        }
        let expected = self.src_w * self.src_h;
        if src.len() < expected {
            return Err(ParallelScaleError::BufferTooSmall {
                expected,
                actual: src.len(),
            });
        }
        Ok(())
    }

    /// Scale using bilinear interpolation with parallel row processing.
    ///
    /// Each output row is computed independently on a Rayon thread.
    pub fn scale_bilinear(&self, src: &[f32]) -> Result<Vec<f32>, ParallelScaleError> {
        self.validate(src)?;
        let sw = self.src_w;
        let sh = self.src_h;
        let dw = self.dst_w;
        let dh = self.dst_h;

        let x_scale = sw as f32 / dw as f32;
        let y_scale = sh as f32 / dh as f32;

        let mut output = vec![0.0f32; dw * dh];
        output.par_chunks_mut(dw).enumerate().for_each(|(dy, row)| {
            let src_y = dy as f32 * y_scale;
            let y0 = (src_y.floor() as usize).min(sh - 1);
            let y1 = (y0 + 1).min(sh - 1);
            let (wy0, wy1) = bilinear_weight(src_y - src_y.floor());

            for (dx, pixel) in row.iter_mut().enumerate() {
                let src_x = dx as f32 * x_scale;
                let x0 = (src_x.floor() as usize).min(sw - 1);
                let x1 = (x0 + 1).min(sw - 1);
                let (wx0, wx1) = bilinear_weight(src_x - src_x.floor());

                let v00 = src[y0 * sw + x0];
                let v10 = src[y0 * sw + x1];
                let v01 = src[y1 * sw + x0];
                let v11 = src[y1 * sw + x1];

                *pixel = wy0 * (wx0 * v00 + wx1 * v10) + wy1 * (wx0 * v01 + wx1 * v11);
            }
        });

        Ok(output)
    }

    /// Scale using bicubic (Catmull-Rom) interpolation with parallel row processing.
    pub fn scale_bicubic(&self, src: &[f32]) -> Result<Vec<f32>, ParallelScaleError> {
        self.validate(src)?;
        let sw = self.src_w;
        let sh = self.src_h;
        let dw = self.dst_w;
        let dh = self.dst_h;

        let x_scale = sw as f32 / dw as f32;
        let y_scale = sh as f32 / dh as f32;

        let mut output = vec![0.0f32; dw * dh];
        output.par_chunks_mut(dw).enumerate().for_each(|(dy, row)| {
            let src_y = dy as f32 * y_scale;
            let iy = src_y.floor() as isize;
            let fy = src_y - src_y.floor();
            let wy = cubic_weights(fy);

            for (dx, pixel) in row.iter_mut().enumerate() {
                let src_x = dx as f32 * x_scale;
                let ix = src_x.floor() as isize;
                let fx = src_x - src_x.floor();
                let wx = cubic_weights(fx);

                let mut val = 0.0f32;
                for j in 0..4i32 {
                    let sy = (iy + j as isize - 1).max(0).min(sh as isize - 1) as usize;
                    for i in 0..4i32 {
                        let sx = (ix + i as isize - 1).max(0).min(sw as isize - 1) as usize;
                        val += wy[j as usize] * wx[i as usize] * src[sy * sw + sx];
                    }
                }
                *pixel = val;
            }
        });

        Ok(output)
    }

    /// Scale using nearest-neighbor with parallel row processing.
    pub fn scale_nearest(&self, src: &[f32]) -> Result<Vec<f32>, ParallelScaleError> {
        self.validate(src)?;
        let sw = self.src_w;
        let sh = self.src_h;
        let dw = self.dst_w;
        let dh = self.dst_h;

        let x_scale = sw as f32 / dw as f32;
        let y_scale = sh as f32 / dh as f32;

        let mut output = vec![0.0f32; dw * dh];
        output.par_chunks_mut(dw).enumerate().for_each(|(dy, row)| {
            let sy = ((dy as f32 * y_scale) as usize).min(sh - 1);
            for (dx, pixel) in row.iter_mut().enumerate() {
                let sx = ((dx as f32 * x_scale) as usize).min(sw - 1);
                *pixel = src[sy * sw + sx];
            }
        });

        Ok(output)
    }

    /// Compute the scale factors (x, y).
    pub fn scale_factors(&self) -> (f32, f32) {
        let x = self.src_w as f32 / self.dst_w.max(1) as f32;
        let y = self.src_h as f32 / self.dst_h.max(1) as f32;
        (x, y)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bilinear_upscale_output_size() {
        let scaler = ParallelScaler::new(4, 4, 8, 8);
        let src = vec![1.0f32; 16];
        let dst = scaler.scale_bilinear(&src).expect("ok");
        assert_eq!(dst.len(), 64);
    }

    #[test]
    fn test_bilinear_uniform_preserves_value() {
        let scaler = ParallelScaler::new(8, 8, 4, 4);
        let src = vec![0.5f32; 64];
        let dst = scaler.scale_bilinear(&src).expect("ok");
        for &v in &dst {
            assert!(
                (v - 0.5).abs() < 1e-4,
                "uniform value should be preserved, got {v}"
            );
        }
    }

    #[test]
    fn test_bicubic_output_size() {
        let scaler = ParallelScaler::new(6, 6, 12, 12);
        let src = vec![0.3f32; 36];
        let dst = scaler.scale_bicubic(&src).expect("ok");
        assert_eq!(dst.len(), 144);
    }

    #[test]
    fn test_bicubic_uniform_preserves_value() {
        let scaler = ParallelScaler::new(8, 8, 4, 4);
        let src = vec![0.7f32; 64];
        let dst = scaler.scale_bicubic(&src).expect("ok");
        for &v in &dst {
            assert!(
                (v - 0.7).abs() < 0.05,
                "uniform bicubic should stay near 0.7, got {v}"
            );
        }
    }

    #[test]
    fn test_nearest_output_size() {
        let scaler = ParallelScaler::new(3, 3, 6, 6);
        let src = vec![1.0f32; 9];
        let dst = scaler.scale_nearest(&src).expect("ok");
        assert_eq!(dst.len(), 36);
    }

    #[test]
    fn test_nearest_preserves_exact_values() {
        let scaler = ParallelScaler::new(2, 2, 4, 4);
        let src = vec![10.0, 20.0, 30.0, 40.0];
        let dst = scaler.scale_nearest(&src).expect("ok");
        // Top-left quadrant should be 10.0
        assert_eq!(dst[0], 10.0);
        assert_eq!(dst[1], 10.0);
    }

    #[test]
    fn test_zero_source_dimension_error() {
        let scaler = ParallelScaler::new(0, 4, 4, 4);
        let result = scaler.scale_bilinear(&[]);
        assert!(matches!(
            result,
            Err(ParallelScaleError::ZeroDimension { dim: "source" })
        ));
    }

    #[test]
    fn test_zero_dest_dimension_error() {
        let scaler = ParallelScaler::new(4, 4, 0, 4);
        let src = vec![1.0f32; 16];
        let result = scaler.scale_bilinear(&src);
        assert!(matches!(
            result,
            Err(ParallelScaleError::ZeroDimension { dim: "destination" })
        ));
    }

    #[test]
    fn test_buffer_too_small_error() {
        let scaler = ParallelScaler::new(4, 4, 2, 2);
        let src = vec![1.0f32; 8]; // need 16
        let result = scaler.scale_bilinear(&src);
        assert!(matches!(
            result,
            Err(ParallelScaleError::BufferTooSmall { .. })
        ));
    }

    #[test]
    fn test_scale_factors() {
        let scaler = ParallelScaler::new(100, 200, 50, 100);
        let (sx, sy) = scaler.scale_factors();
        assert!((sx - 2.0).abs() < 1e-6);
        assert!((sy - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_identity_scale_bilinear() {
        let scaler = ParallelScaler::new(4, 4, 4, 4);
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let dst = scaler.scale_bilinear(&src).expect("ok");
        for (i, (&s, &d)) in src.iter().zip(dst.iter()).enumerate() {
            assert!(
                (s - d).abs() < 1e-3,
                "identity bilinear mismatch at {i}: {s} vs {d}"
            );
        }
    }

    #[test]
    fn test_identity_scale_nearest() {
        let scaler = ParallelScaler::new(4, 4, 4, 4);
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let dst = scaler.scale_nearest(&src).expect("ok");
        for (i, (&s, &d)) in src.iter().zip(dst.iter()).enumerate() {
            assert!(
                (s - d).abs() < 1e-6,
                "identity nearest mismatch at {i}: {s} vs {d}"
            );
        }
    }

    #[test]
    fn test_debug_format() {
        let scaler = ParallelScaler::new(10, 10, 20, 20);
        let s = format!("{scaler:?}");
        assert!(s.contains("ParallelScaler"));
    }

    #[test]
    fn test_downscale_bilinear_output_values_in_range() {
        let scaler = ParallelScaler::new(8, 8, 2, 2);
        // Gradient image 0..63
        let src: Vec<f32> = (0..64).map(|i| i as f32 / 63.0).collect();
        let dst = scaler.scale_bilinear(&src).expect("ok");
        for &v in &dst {
            assert!(v >= 0.0 && v <= 1.0, "value {v} out of [0,1] range");
        }
    }

    #[test]
    fn test_error_display() {
        let e = ParallelScaleError::BufferTooSmall {
            expected: 16,
            actual: 8,
        };
        let s = e.to_string();
        assert!(s.contains("16"));
        assert!(s.contains("8"));
    }
}
