//! GPU-accelerated perspective transform and lens distortion correction.
//!
//! This module provides two closely related geometric operations:
//!
//! 1. **Perspective (homography) transform**: maps a quadrilateral region of
//!    the source image to a rectangle in the output (or vice versa).  The
//!    transform is specified as a 3×3 homography matrix.
//!
//! 2. **Lens distortion correction**: removes barrel or pincushion distortion
//!    using the Brown-Conrady radial/tangential distortion model.
//!
//! Both operations use backward-mapping with bilinear interpolation: for each
//! output pixel the inverse transform is applied to find the corresponding
//! source location, which is then sampled bilinearly from the input.
//!
//! All heavy work is parallelised over rows using rayon.
//!
//! # Example
//!
//! ```no_run
//! use oximedia_gpu::perspective_transform::{
//!     HomographyMatrix, PerspectiveTransform, LensDistortionParams, LensDistortionCorrector,
//! };
//!
//! let src = vec![0u8; 640 * 480 * 4];
//! let mut dst = vec![0u8; 640 * 480 * 4];
//!
//! // Identity transform
//! let h = HomographyMatrix::identity();
//! PerspectiveTransform::new(h)
//!     .warp_rgba(&src, 640, 480, &mut dst, 640, 480)
//!     .unwrap();
//! ```

use crate::{GpuError, Result};
use rayon::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// Homography matrix
// ─────────────────────────────────────────────────────────────────────────────

/// A 3×3 homography (perspective transform) matrix stored in row-major order.
///
/// The matrix maps homogeneous source coordinates `[x, y, 1]` to destination
/// coordinates:
/// ```text
/// [x', y', w'] = H * [x, y, 1]
/// dst = (x'/w', y'/w')
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HomographyMatrix {
    /// Row-major 3×3 matrix coefficients.
    pub m: [[f64; 3]; 3],
}

impl HomographyMatrix {
    /// Create a homography from a row-major 3×3 array.
    #[must_use]
    pub fn new(m: [[f64; 3]; 3]) -> Self {
        Self { m }
    }

    /// Identity homography (no transform).
    #[must_use]
    pub fn identity() -> Self {
        Self {
            m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Create a 2D translation matrix.
    #[must_use]
    pub fn translation(tx: f64, ty: f64) -> Self {
        Self {
            m: [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]],
        }
    }

    /// Create a uniform scale matrix.
    #[must_use]
    pub fn scale(sx: f64, sy: f64) -> Self {
        Self {
            m: [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Create a counter-clockwise rotation matrix around the origin.
    #[must_use]
    pub fn rotation(angle_rad: f64) -> Self {
        let c = angle_rad.cos();
        let s = angle_rad.sin();
        Self {
            m: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Apply the homography to a point `(x, y)`.
    ///
    /// Returns `None` if the projective `w` coordinate is near zero (degenerate).
    #[must_use]
    pub fn apply(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let m = &self.m;
        let xp = m[0][0] * x + m[0][1] * y + m[0][2];
        let yp = m[1][0] * x + m[1][1] * y + m[1][2];
        let wp = m[2][0] * x + m[2][1] * y + m[2][2];
        if wp.abs() < 1e-10 {
            return None;
        }
        Some((xp / wp, yp / wp))
    }

    /// Compute the inverse of this homography using Cramer's rule.
    ///
    /// Returns `None` if the matrix is singular.
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        let m = &self.m;

        // Cofactors
        let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
        let c01 = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]);
        let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
        let c10 = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]);
        let c11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
        let c12 = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]);
        let c20 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
        let c21 = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]);
        let c22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];

        let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
        if det.abs() < 1e-12 {
            return None;
        }

        let inv_det = 1.0 / det;

        // Adjugate (transpose of cofactor matrix)
        Some(Self {
            m: [
                [c00 * inv_det, c10 * inv_det, c20 * inv_det],
                [c01 * inv_det, c11 * inv_det, c21 * inv_det],
                [c02 * inv_det, c12 * inv_det, c22 * inv_det],
            ],
        })
    }

    /// Multiply two homographies.
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        let a = &self.m;
        let b = &other.m;
        let mut result = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        Self { m: result }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Perspective transform
// ─────────────────────────────────────────────────────────────────────────────

/// Perspective (homography) transform applied to RGBA images.
#[derive(Debug, Clone)]
pub struct PerspectiveTransform {
    /// The forward homography (src → dst mapping).
    pub homography: HomographyMatrix,
}

impl PerspectiveTransform {
    /// Create a new perspective transform with the given homography.
    #[must_use]
    pub fn new(homography: HomographyMatrix) -> Self {
        Self { homography }
    }

    /// Apply a perspective warp to an RGBA source image.
    ///
    /// Uses backward-mapping: for each destination pixel, the inverse
    /// homography maps it to a source coordinate, which is sampled bilinearly.
    ///
    /// Out-of-bounds source coordinates are filled with black (`[0, 0, 0, 0]`).
    ///
    /// # Errors
    ///
    /// Returns an error if buffer sizes are inconsistent or if the homography
    /// is not invertible.
    pub fn warp_rgba(
        &self,
        src: &[u8],
        src_w: u32,
        src_h: u32,
        dst: &mut [u8],
        dst_w: u32,
        dst_h: u32,
    ) -> Result<()> {
        let src_expected = (src_w as usize) * (src_h as usize) * 4;
        let dst_expected = (dst_w as usize) * (dst_h as usize) * 4;

        if src.len() != src_expected {
            return Err(GpuError::InvalidBufferSize {
                expected: src_expected,
                actual: src.len(),
            });
        }
        if dst.len() != dst_expected {
            return Err(GpuError::InvalidBufferSize {
                expected: dst_expected,
                actual: dst.len(),
            });
        }

        let inv = self
            .homography
            .inverse()
            .ok_or_else(|| GpuError::Internal("Homography is not invertible".to_string()))?;

        let sw = src_w as usize;
        let sh = src_h as usize;
        let dw = dst_w as usize;

        // Process rows in parallel
        dst.par_chunks_exact_mut(dw * 4)
            .enumerate()
            .for_each(|(dy, row)| {
                for dx in 0..dw {
                    let (sx, sy) = match inv.apply(dx as f64, dy as f64) {
                        Some(p) => p,
                        None => {
                            let off = dx * 4;
                            row[off..off + 4].copy_from_slice(&[0u8; 4]);
                            continue;
                        }
                    };

                    let pixel = bilinear_sample_rgba(src, sx, sy, sw, sh);
                    let off = dx * 4;
                    row[off..off + 4].copy_from_slice(&pixel);
                }
            });

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Lens distortion correction
// ─────────────────────────────────────────────────────────────────────────────

/// Brown-Conrady radial and tangential lens distortion parameters.
///
/// These describe the distortion of a physical lens. Negative radial
/// coefficients model barrel distortion; positive values model pincushion.
#[derive(Debug, Clone, Copy)]
pub struct LensDistortionParams {
    /// Radial distortion coefficient k1.
    pub k1: f64,
    /// Radial distortion coefficient k2.
    pub k2: f64,
    /// Radial distortion coefficient k3.
    pub k3: f64,
    /// Tangential distortion coefficient p1.
    pub p1: f64,
    /// Tangential distortion coefficient p2.
    pub p2: f64,
    /// Principal point x (normalised, typically 0.5).
    pub cx: f64,
    /// Principal point y (normalised, typically 0.5).
    pub cy: f64,
    /// Focal length x (normalised, typically ~1.0 for a 90° FOV).
    pub fx: f64,
    /// Focal length y (normalised).
    pub fy: f64,
}

impl Default for LensDistortionParams {
    fn default() -> Self {
        Self {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            cx: 0.5,
            cy: 0.5,
            fx: 1.0,
            fy: 1.0,
        }
    }
}

impl LensDistortionParams {
    /// Create parameters modelling mild barrel distortion (typical wide-angle lens).
    #[must_use]
    pub fn barrel() -> Self {
        Self {
            k1: -0.3,
            k2: 0.1,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            ..Default::default()
        }
    }

    /// Create parameters modelling mild pincushion distortion (typical telephoto).
    #[must_use]
    pub fn pincushion() -> Self {
        Self {
            k1: 0.3,
            k2: -0.05,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            ..Default::default()
        }
    }

    /// Map a distorted normalised pixel coordinate to an undistorted one.
    ///
    /// The coordinate is normalised so that (0, 0) = top-left and
    /// (1, 1) = bottom-right.
    #[must_use]
    pub fn undistort_point(&self, x_nd: f64, y_nd: f64) -> (f64, f64) {
        // Convert to camera coordinates centred on principal point
        let x = (x_nd - self.cx) / self.fx;
        let y = (y_nd - self.cy) / self.fy;

        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let r6 = r2 * r4;

        let radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;
        let x_tan = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
        let y_tan = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;

        let xu = x * radial + x_tan;
        let yu = y * radial + y_tan;

        // Back to normalised image coordinates
        (xu * self.fx + self.cx, yu * self.fy + self.cy)
    }
}

/// Lens distortion corrector.
///
/// Removes lens distortion from RGBA frames using the Brown-Conrady model.
#[derive(Debug, Clone)]
pub struct LensDistortionCorrector {
    params: LensDistortionParams,
}

impl LensDistortionCorrector {
    /// Create a new corrector with the given lens parameters.
    #[must_use]
    pub fn new(params: LensDistortionParams) -> Self {
        Self { params }
    }

    /// Remove lens distortion from an RGBA frame.
    ///
    /// For each output pixel the undistortion mapping is applied to find the
    /// source location, which is sampled bilinearly.
    ///
    /// # Errors
    ///
    /// Returns an error if `src` and `dst` buffer sizes don't match the
    /// declared dimensions.
    pub fn undistort_rgba(&self, src: &[u8], src_w: u32, src_h: u32, dst: &mut [u8]) -> Result<()> {
        let expected = (src_w as usize) * (src_h as usize) * 4;
        if src.len() != expected {
            return Err(GpuError::InvalidBufferSize {
                expected,
                actual: src.len(),
            });
        }
        if dst.len() != expected {
            return Err(GpuError::InvalidBufferSize {
                expected,
                actual: dst.len(),
            });
        }

        let sw = src_w as usize;
        let sh = src_h as usize;

        dst.par_chunks_exact_mut(sw * 4)
            .enumerate()
            .for_each(|(dy, row)| {
                for dx in 0..sw {
                    // Normalised coordinates [0, 1)
                    let x_nd = (dx as f64 + 0.5) / sw as f64;
                    let y_nd = (dy as f64 + 0.5) / sh as f64;

                    let (sx, sy) = self.params.undistort_point(x_nd, y_nd);

                    // Convert back to pixel coordinates
                    let sx_px = sx * sw as f64 - 0.5;
                    let sy_px = sy * sh as f64 - 0.5;

                    let pixel = bilinear_sample_rgba(src, sx_px, sy_px, sw, sh);
                    let off = dx * 4;
                    row[off..off + 4].copy_from_slice(&pixel);
                }
            });

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared bilinear sampling helper
// ─────────────────────────────────────────────────────────────────────────────

/// Bilinear sample from an RGBA image at fractional pixel coordinate `(x, y)`.
///
/// Out-of-bounds coordinates produce `[0, 0, 0, 0]`.
fn bilinear_sample_rgba(src: &[u8], x: f64, y: f64, w: usize, h: usize) -> [u8; 4] {
    let x0 = x.floor() as isize;
    let y0 = y.floor() as isize;
    let tx = (x - x0 as f64) as f32;
    let ty = (y - y0 as f64) as f32;

    let get_pixel = |xi: isize, yi: isize| -> [f32; 4] {
        if xi < 0 || yi < 0 || xi >= w as isize || yi >= h as isize {
            return [0.0; 4];
        }
        let off = (yi as usize * w + xi as usize) * 4;
        [
            src[off] as f32,
            src[off + 1] as f32,
            src[off + 2] as f32,
            src[off + 3] as f32,
        ]
    };

    let c00 = get_pixel(x0, y0);
    let c10 = get_pixel(x0 + 1, y0);
    let c01 = get_pixel(x0, y0 + 1);
    let c11 = get_pixel(x0 + 1, y0 + 1);

    let mut result = [0u8; 4];
    for i in 0..4 {
        let v = c00[i] * (1.0 - tx) * (1.0 - ty)
            + c10[i] * tx * (1.0 - ty)
            + c01[i] * (1.0 - tx) * ty
            + c11[i] * tx * ty;
        result[i] = v.clamp(0.0, 255.0) as u8;
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rgba_frame(w: usize, h: usize) -> Vec<u8> {
        (0..w * h * 4).map(|i| (i % 256) as u8).collect()
    }

    // ── HomographyMatrix ──────────────────────────────────────────────────────

    #[test]
    fn test_identity_apply() {
        let h = HomographyMatrix::identity();
        let (x, y) = h.apply(3.0, 7.0).expect("should not be degenerate");
        assert!((x - 3.0).abs() < 1e-10);
        assert!((y - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_of_identity() {
        let h = HomographyMatrix::identity();
        let inv = h.inverse().expect("identity is invertible");
        let (x, y) = inv.apply(5.0, 9.0).expect("ok");
        assert!((x - 5.0).abs() < 1e-9);
        assert!((y - 9.0).abs() < 1e-9);
    }

    #[test]
    fn test_translation_roundtrip() {
        let h = HomographyMatrix::translation(10.0, -5.0);
        let inv = h.inverse().expect("translation is invertible");
        let (x, y) = h.apply(3.0, 3.0).expect("ok");
        let (x2, y2) = inv.apply(x, y).expect("ok");
        assert!((x2 - 3.0).abs() < 1e-9);
        assert!((y2 - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_scale_apply() {
        let h = HomographyMatrix::scale(2.0, 3.0);
        let (x, y) = h.apply(4.0, 5.0).expect("ok");
        assert!((x - 8.0).abs() < 1e-9);
        assert!((y - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_rotation_preserves_magnitude() {
        use std::f64::consts::PI;
        let h = HomographyMatrix::rotation(PI / 4.0);
        let (x, y) = h.apply(1.0, 0.0).expect("ok");
        let mag = (x * x + y * y).sqrt();
        assert!((mag - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compose() {
        let t1 = HomographyMatrix::translation(1.0, 0.0);
        let t2 = HomographyMatrix::translation(2.0, 0.0);
        let composed = t1.compose(&t2);
        let (x, _) = composed.apply(0.0, 0.0).expect("ok");
        assert!((x - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_singular_inverse_returns_none() {
        let h = HomographyMatrix::new([[0.0; 3]; 3]);
        assert!(h.inverse().is_none());
    }

    // ── PerspectiveTransform ──────────────────────────────────────────────────

    #[test]
    fn test_identity_warp_preserves_size() {
        let w = 8u32;
        let h = 8u32;
        let src = rgba_frame(w as usize, h as usize);
        let mut dst = vec![0u8; src.len()];
        let pt = PerspectiveTransform::new(HomographyMatrix::identity());
        pt.warp_rgba(&src, w, h, &mut dst, w, h)
            .expect("warp should succeed");
        assert_eq!(dst.len(), src.len());
    }

    #[test]
    fn test_warp_wrong_size_rejected() {
        let src = vec![0u8; 8 * 8 * 4];
        let mut dst = vec![0u8; 4 * 4 * 4]; // wrong size for declared dimensions
        let pt = PerspectiveTransform::new(HomographyMatrix::identity());
        // dst declared as 8x8 but actually 4x4 → error
        let res = pt.warp_rgba(&src, 8, 8, &mut dst, 8, 8);
        assert!(res.is_err());
    }

    #[test]
    fn test_singular_homography_rejected() {
        let src = vec![0u8; 4 * 4 * 4];
        let mut dst = vec![0u8; 4 * 4 * 4];
        let pt = PerspectiveTransform::new(HomographyMatrix::new([[0.0; 3]; 3]));
        let res = pt.warp_rgba(&src, 4, 4, &mut dst, 4, 4);
        assert!(res.is_err());
    }

    // ── LensDistortionCorrector ───────────────────────────────────────────────

    #[test]
    fn test_no_distortion_identity() {
        let params = LensDistortionParams::default();
        let (xu, yu) = params.undistort_point(0.5, 0.5);
        assert!((xu - 0.5).abs() < 1e-10);
        assert!((yu - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_undistort_rgba_correct_size() {
        let w = 8u32;
        let h = 8u32;
        let src = rgba_frame(w as usize, h as usize);
        let mut dst = vec![0u8; src.len()];
        let corrector = LensDistortionCorrector::new(LensDistortionParams::default());
        corrector
            .undistort_rgba(&src, w, h, &mut dst)
            .expect("should succeed");
        assert_eq!(dst.len(), src.len());
    }

    #[test]
    fn test_undistort_rgba_wrong_size_rejected() {
        let src = vec![0u8; 8 * 8 * 4];
        let mut dst = vec![0u8; 4]; // too small
        let corrector = LensDistortionCorrector::new(LensDistortionParams::default());
        let res = corrector.undistort_rgba(&src, 8, 8, &mut dst);
        assert!(res.is_err());
    }

    #[test]
    fn test_barrel_pincushion_differ() {
        let barrel = LensDistortionParams::barrel();
        let pin = LensDistortionParams::pincushion();
        assert!(barrel.k1 < 0.0);
        assert!(pin.k1 > 0.0);
    }
}
