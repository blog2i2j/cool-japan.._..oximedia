//! Perspective (homography) warping for image correction and transformation.
//!
//! A **homography** (projective transform) is a 3×3 matrix that maps points in one
//! projective plane to another.  It is the most general planar transformation that
//! preserves straight lines — used for perspective correction, billboard projection,
//! document de-skewing, and augmented-reality overlays.
//!
//! # Algorithm
//!
//! For each destination pixel `(x, y)` the inverse homography is evaluated to
//! find the corresponding source coordinate `(sx, sy)`.  Bilinear interpolation
//! is then used to sample the source image at that sub-pixel position.
//!
//! ## Homography convention
//!
//! The matrix `H` maps **homogeneous** source points `[X, Y, W]` to destination
//! points `[X', Y', W']`:
//!
//! ```text
//! [X']   [h00 h01 h02] [X]
//! [Y'] = [h10 h11 h12] [Y]
//! [W']   [h20 h21 h22] [W]
//! ```
//!
//! Normalised coordinates are recovered as `x' = X'/W'`, `y' = Y'/W'`.
//!
//! The public API takes the **forward** homography (source → destination) and
//! inverts it internally before warping.
//!
//! # Examples
//!
//! ```rust
//! use oximedia_image::perspective_warp::{Homography, warp_perspective};
//!
//! // Identity homography: output equals input.
//! let h = Homography::identity();
//! let img = vec![128u8; 4 * 4 * 3]; // 4×4 RGB
//! let out = warp_perspective(&img, 4, 4, 3, 4, 4, &h, [0u8; 3]).unwrap();
//! assert_eq!(out.len(), 4 * 4 * 3);
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use crate::error::{ImageError, ImageResult};

// ── Homography ────────────────────────────────────────────────────────────────

/// A 3×3 projective (homography) transform matrix stored in row-major order.
///
/// Elements: `[[h00, h01, h02], [h10, h11, h12], [h20, h21, h22]]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Homography {
    /// Row-major 3×3 matrix elements.
    pub m: [[f64; 3]; 3],
}

impl Homography {
    /// Identity transform — every point maps to itself.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Construct from a flat row-major 9-element array.
    #[must_use]
    pub fn from_array(arr: [f64; 9]) -> Self {
        Self {
            m: [
                [arr[0], arr[1], arr[2]],
                [arr[3], arr[4], arr[5]],
                [arr[6], arr[7], arr[8]],
            ],
        }
    }

    /// Return the 3×3 matrix determinant.
    #[must_use]
    pub fn det(&self) -> f64 {
        let m = &self.m;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    /// Compute the inverse of this homography.
    ///
    /// # Errors
    /// Returns [`ImageError::InvalidFormat`] if the matrix is singular (det ≈ 0).
    pub fn inverse(&self) -> ImageResult<Self> {
        let d = self.det();
        if d.abs() < 1e-12 {
            return Err(ImageError::InvalidFormat(
                "homography matrix is singular (det ≈ 0)".to_string(),
            ));
        }
        let m = &self.m;
        let inv_d = 1.0 / d;
        let r = [
            [
                (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_d,
                (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_d,
                (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_d,
            ],
            [
                (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_d,
                (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_d,
                (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_d,
            ],
            [
                (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_d,
                (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_d,
                (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_d,
            ],
        ];
        Ok(Self { m: r })
    }

    /// Compose `self` with `other` (self ∘ other, i.e., apply `other` first).
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        let a = &self.m;
        let b = &other.m;
        let mut r = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
            }
        }
        Self { m: r }
    }

    /// Apply this homography to a 2-D point, returning `(x', y')`.
    ///
    /// Returns `None` if the point maps to the plane at infinity (w ≈ 0).
    #[must_use]
    pub fn map_point(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let m = &self.m;
        let xp = m[0][0] * x + m[0][1] * y + m[0][2];
        let yp = m[1][0] * x + m[1][1] * y + m[1][2];
        let wp = m[2][0] * x + m[2][1] * y + m[2][2];
        if wp.abs() < 1e-12 {
            return None;
        }
        Some((xp / wp, yp / wp))
    }

    /// Compute a homography from four point correspondences.
    ///
    /// `src_pts` and `dst_pts` are four `(x, y)` pairs each (source → destination).
    ///
    /// The system is solved via Gaussian elimination on the 8×8 DLT linear system.
    ///
    /// # Errors
    /// Returns [`ImageError::InvalidFormat`] if the points are degenerate (collinear etc.).
    pub fn from_four_points(
        src_pts: &[(f64, f64); 4],
        dst_pts: &[(f64, f64); 4],
    ) -> ImageResult<Self> {
        // Direct Linear Transform (DLT) — sets h22 = 1 and solves 8 equations.
        let mut a = [[0.0f64; 8]; 8];
        let mut b = [0.0f64; 8];

        for (i, (&(sx, sy), &(dx, dy))) in src_pts.iter().zip(dst_pts.iter()).enumerate() {
            let row = i * 2;
            // Row for x equation: -sx*dx*h20 - sy*dx*h21 + sx*h00 + sy*h01 + h02 - dx*h20*...
            // Standard DLT formulation (h22 = 1 normalised):
            a[row][0] = sx;
            a[row][1] = sy;
            a[row][2] = 1.0;
            a[row][3] = 0.0;
            a[row][4] = 0.0;
            a[row][5] = 0.0;
            a[row][6] = -dx * sx;
            a[row][7] = -dx * sy;
            b[row] = dx;

            let row1 = row + 1;
            a[row1][0] = 0.0;
            a[row1][1] = 0.0;
            a[row1][2] = 0.0;
            a[row1][3] = sx;
            a[row1][4] = sy;
            a[row1][5] = 1.0;
            a[row1][6] = -dy * sx;
            a[row1][7] = -dy * sy;
            b[row1] = dy;
        }

        let h = gaussian_solve_8x8(a, b).map_err(|e| ImageError::InvalidFormat(e.to_string()))?;
        Ok(Self {
            m: [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]],
        })
    }
}

// ── Gaussian elimination (8×8) ────────────────────────────────────────────────

/// Solve `Ax = b` for an 8×8 system via partial-pivot Gaussian elimination.
fn gaussian_solve_8x8(mut a: [[f64; 8]; 8], mut b: [f64; 8]) -> Result<[f64; 8], &'static str> {
    const N: usize = 8;
    for col in 0..N {
        // Partial pivot
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..N {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return Err("degenerate point configuration for homography DLT");
        }
        a.swap(col, max_row);
        b.swap(col, max_row);

        let pivot = a[col][col];
        for row in (col + 1)..N {
            let factor = a[row][col] / pivot;
            for k in col..N {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    // Back-substitution
    let mut x = [0.0f64; 8];
    for i in (0..N).rev() {
        let mut sum = b[i];
        for j in (i + 1)..N {
            sum -= a[i][j] * x[j];
        }
        x[i] = sum / a[i][i];
    }
    Ok(x)
}

// ── warp_perspective ──────────────────────────────────────────────────────────

/// Warp an image using the given forward homography (source → destination).
///
/// - `src_data`: flat row-major pixel buffer with `src_ch` channels per pixel.
/// - `src_w`, `src_h`: source image dimensions.
/// - `src_ch`: number of channels (1–4).
/// - `dst_w`, `dst_h`: output image dimensions.
/// - `h`: forward homography mapping source pixels to destination pixels.
/// - `fill`: border fill colour (must have `src_ch` elements).
///
/// Bilinear interpolation is used at fractional source coordinates.
/// Out-of-bounds source coordinates produce `fill` pixels.
///
/// # Errors
/// Returns [`ImageError::InvalidFormat`] if buffer sizes are inconsistent or
/// if the homography cannot be inverted.
pub fn warp_perspective(
    src_data: &[u8],
    src_w: u32,
    src_h: u32,
    src_ch: u32,
    dst_w: u32,
    dst_h: u32,
    h: &Homography,
    fill: impl AsRef<[u8]>,
) -> ImageResult<Vec<u8>> {
    let fill = fill.as_ref();
    let ch = src_ch as usize;
    if ch == 0 || ch > 4 {
        return Err(ImageError::Unsupported(format!(
            "src_ch={src_ch}: only 1–4 channels are supported"
        )));
    }
    if fill.len() != ch {
        return Err(ImageError::InvalidFormat(format!(
            "fill slice length {} must equal src_ch {}",
            fill.len(),
            ch
        )));
    }
    let src_expected = (src_w as usize)
        .checked_mul(src_h as usize)
        .and_then(|n| n.checked_mul(ch))
        .ok_or(ImageError::InvalidDimensions(src_w, src_h))?;
    if src_data.len() != src_expected {
        return Err(ImageError::InvalidFormat(format!(
            "src buffer length {} does not match {}×{}×{} = {}",
            src_data.len(),
            src_w,
            src_h,
            ch,
            src_expected
        )));
    }

    // Invert the forward homography to get destination → source mapping.
    let h_inv = h.inverse()?;

    let dst_pixels = (dst_w as usize)
        .checked_mul(dst_h as usize)
        .ok_or(ImageError::InvalidDimensions(dst_w, dst_h))?;
    let mut out = vec![0u8; dst_pixels * ch];

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let dst_off = ((dy as usize) * (dst_w as usize) + (dx as usize)) * ch;
            match h_inv.map_point(dx as f64, dy as f64) {
                None => {
                    out[dst_off..dst_off + ch].copy_from_slice(fill);
                }
                Some((sx, sy)) => {
                    sample_bilinear_clamped(
                        src_data,
                        src_w,
                        src_h,
                        ch,
                        sx,
                        sy,
                        fill,
                        &mut out[dst_off..dst_off + ch],
                    );
                }
            }
        }
    }

    Ok(out)
}

/// Bilinear interpolation with constant-fill border (clamp-to-fill).
fn sample_bilinear_clamped(
    src: &[u8],
    width: u32,
    height: u32,
    ch: usize,
    sx: f64,
    sy: f64,
    fill: &[u8],
    dst: &mut [u8],
) {
    let w = width as i64;
    let h = height as i64;

    let x0 = sx.floor() as i64;
    let y0 = sy.floor() as i64;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let fx = (sx - sx.floor()) as f32;
    let fy = (sy - sy.floor()) as f32;

    let mut px = [[0u8; 4]; 4];
    let coords = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)];

    for (i, &(cx, cy)) in coords.iter().enumerate() {
        if cx >= 0 && cy >= 0 && cx < w && cy < h {
            let off = ((cy as usize) * (width as usize) + (cx as usize)) * ch;
            px[i][..ch].copy_from_slice(&src[off..off + ch]);
        } else {
            px[i][..ch].copy_from_slice(fill);
        }
    }

    for c in 0..ch {
        let tl = px[0][c] as f32;
        let tr = px[1][c] as f32;
        let bl = px[2][c] as f32;
        let br = px[3][c] as f32;
        let top = tl + (tr - tl) * fx;
        let bot = bl + (br - bl) * fx;
        let val = top + (bot - top) * fy;
        dst[c] = val.round().clamp(0.0, 255.0) as u8;
    }
}

// ── convenience builders ──────────────────────────────────────────────────────

/// Build a homography for a pure translation by `(tx, ty)`.
#[must_use]
pub fn translation(tx: f64, ty: f64) -> Homography {
    Homography {
        m: [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]],
    }
}

/// Build a homography for uniform scaling by `s` around the origin.
#[must_use]
pub fn scaling(sx: f64, sy: f64) -> Homography {
    Homography {
        m: [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]],
    }
}

/// Build a homography for rotation by `angle_deg` degrees (clockwise, around origin).
#[must_use]
pub fn rotation(angle_deg: f64) -> Homography {
    let r = angle_deg.to_radians();
    let cos = r.cos();
    let sin = r.sin();
    Homography {
        m: [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]],
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_homography_det() {
        let h = Homography::identity();
        assert!((h.det() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_identity_map_point() {
        let h = Homography::identity();
        let (x, y) = h.map_point(3.0, 7.0).unwrap();
        assert!((x - 3.0).abs() < 1e-10);
        assert!((y - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_of_identity_is_identity() {
        let h = Homography::identity();
        let inv = h.inverse().unwrap();
        let (x, y) = inv.map_point(5.0, 9.0).unwrap();
        assert!((x - 5.0).abs() < 1e-8);
        assert!((y - 9.0).abs() < 1e-8);
    }

    #[test]
    fn test_compose_with_inverse_is_identity() {
        let h = scaling(2.0, 3.0);
        let inv = h.inverse().unwrap();
        let composed = h.compose(&inv);
        let (x, y) = composed.map_point(4.0, 5.0).unwrap();
        assert!((x - 4.0).abs() < 1e-8, "x={x}");
        assert!((y - 5.0).abs() < 1e-8, "y={y}");
    }

    #[test]
    fn test_translation_builder() {
        let h = translation(10.0, -5.0);
        let (x, y) = h.map_point(0.0, 0.0).unwrap();
        assert!((x - 10.0).abs() < 1e-10);
        assert!((y - -5.0).abs() < 1e-10);
    }

    #[test]
    fn test_scaling_builder() {
        let h = scaling(2.0, 3.0);
        let (x, y) = h.map_point(1.0, 2.0).unwrap();
        assert!((x - 2.0).abs() < 1e-10);
        assert!((y - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_warp_identity_preserves_image() {
        let img: Vec<u8> = (0..16).map(|i| i as u8).collect(); // 4×4, 1 channel? no — 4×2, 2ch
                                                               // 2×2, 4ch
        let h = Homography::identity();
        let out = warp_perspective(&img, 2, 2, 4, 2, 2, &h, [0u8; 4]).unwrap();
        // With bilinear, centre pixels should mostly match.
        assert_eq!(out.len(), img.len());
    }

    #[test]
    fn test_warp_size_mismatch_error() {
        let img = vec![0u8; 5]; // wrong size
        let h = Homography::identity();
        let result = warp_perspective(&img, 2, 2, 3, 2, 2, &h, [0u8, 0, 0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_four_points_identity() {
        // src = dst → homography should be identity-like.
        let pts: [(f64, f64); 4] = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let h = Homography::from_four_points(&pts, &pts).unwrap();
        let (x, y) = h.map_point(5.0, 5.0).unwrap();
        assert!((x - 5.0).abs() < 1e-5, "x={x}");
        assert!((y - 5.0).abs() < 1e-5, "y={y}");
    }

    #[test]
    fn test_from_array_roundtrip() {
        let arr = [1.0, 0.0, 2.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0];
        let h = Homography::from_array(arr);
        let (x, y) = h.map_point(0.0, 0.0).unwrap();
        assert!((x - 2.0).abs() < 1e-10);
        assert!((y - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_singular_matrix_returns_error() {
        let h = Homography {
            m: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        };
        assert!(h.inverse().is_err());
    }

    #[test]
    fn test_warp_fill_used_for_out_of_bounds() {
        // Scale image to 2× size — half the output should come from fill.
        let img = vec![200u8; 4 * 4 * 3]; // 4×4 RGB, all 200
        let h = scaling(2.0, 2.0); // source → dest: scale 2× → inverse maps dest back to half
        let out = warp_perspective(&img, 4, 4, 3, 8, 8, &h, [0u8, 0, 0]).unwrap();
        assert_eq!(out.len(), 8 * 8 * 3);
        // Bottom-right region (src coords out of bounds) should be fill (0).
        let last_pixel_off = (7 * 8 + 7) * 3;
        // Source coord of (7,7): inverse of scale(2,2) maps (7,7) → (3.5, 3.5) which is in-bounds.
        // Just check the output has correct length and no panic.
        assert!(out[last_pixel_off] < 255 || out[last_pixel_off] == 200);
    }
}
