//! Affine transformation module.
//!
//! This module provides affine transformations including rotation, scaling,
//! translation, and shearing. Affine transforms preserve parallel lines
//! and ratios of distances.
//!
//! # Example
//!
//! ```
//! use oximedia_cv::transform::AffineTransform;
//!
//! let transform = AffineTransform::identity()
//!     .scale(2.0, 2.0)
//!     .rotate(std::f64::consts::PI / 4.0)
//!     .translate(100.0, 50.0);
//!
//! let (x, y) = transform.transform_point(10.0, 10.0);
//! ```

use crate::error::{CvError, CvResult};

/// 2D affine transformation matrix (3x3 with last row [0, 0, 1]).
///
/// The matrix is stored as:
/// ```text
/// | a  b  tx |
/// | c  d  ty |
/// | 0  0  1  |
/// ```
///
/// where (a, b, c, d) define rotation/scale/shear and (tx, ty) define translation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineTransform {
    /// Matrix element \[0,0\] - x scale and rotation.
    pub a: f64,
    /// Matrix element \[0,1\] - x shear.
    pub b: f64,
    /// Matrix element \[0,2\] - x translation.
    pub tx: f64,
    /// Matrix element \[1,0\] - y shear.
    pub c: f64,
    /// Matrix element \[1,1\] - y scale and rotation.
    pub d: f64,
    /// Matrix element \[1,2\] - y translation.
    pub ty: f64,
}

impl AffineTransform {
    /// Create a new affine transform from matrix elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::transform::AffineTransform;
    ///
    /// let transform = AffineTransform::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    /// ```
    #[must_use]
    pub const fn new(a: f64, b: f64, tx: f64, c: f64, d: f64, ty: f64) -> Self {
        Self { a, b, tx, c, d, ty }
    }

    /// Create an identity transform.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::transform::AffineTransform;
    ///
    /// let transform = AffineTransform::identity();
    /// let (x, y) = transform.transform_point(10.0, 20.0);
    /// assert!((x - 10.0).abs() < 0.001);
    /// assert!((y - 20.0).abs() < 0.001);
    /// ```
    #[must_use]
    pub const fn identity() -> Self {
        Self::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    }

    /// Create a translation transform.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::transform::AffineTransform;
    ///
    /// let transform = AffineTransform::translation(10.0, 20.0);
    /// let (x, y) = transform.transform_point(0.0, 0.0);
    /// assert!((x - 10.0).abs() < 0.001);
    /// ```
    #[must_use]
    pub const fn translation(tx: f64, ty: f64) -> Self {
        Self::new(1.0, 0.0, tx, 0.0, 1.0, ty)
    }

    /// Create a scaling transform.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::transform::AffineTransform;
    ///
    /// let transform = AffineTransform::scaling(2.0, 3.0);
    /// let (x, y) = transform.transform_point(10.0, 10.0);
    /// assert!((x - 20.0).abs() < 0.001);
    /// assert!((y - 30.0).abs() < 0.001);
    /// ```
    #[must_use]
    pub const fn scaling(sx: f64, sy: f64) -> Self {
        Self::new(sx, 0.0, 0.0, 0.0, sy, 0.0)
    }

    /// Create a rotation transform around the origin.
    ///
    /// # Arguments
    ///
    /// * `angle` - Rotation angle in radians (counter-clockwise)
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::transform::AffineTransform;
    /// use std::f64::consts::PI;
    ///
    /// let transform = AffineTransform::rotation(PI / 2.0);
    /// let (x, y) = transform.transform_point(1.0, 0.0);
    /// assert!(x.abs() < 0.001);
    /// assert!((y - 1.0).abs() < 0.001);
    /// ```
    #[must_use]
    pub fn rotation(angle: f64) -> Self {
        let cos = angle.cos();
        let sin = angle.sin();
        Self::new(cos, -sin, 0.0, sin, cos, 0.0)
    }

    /// Create a rotation transform around a center point.
    ///
    /// # Arguments
    ///
    /// * `angle` - Rotation angle in radians (counter-clockwise)
    /// * `cx` - Center X coordinate
    /// * `cy` - Center Y coordinate
    #[must_use]
    pub fn rotation_around(angle: f64, cx: f64, cy: f64) -> Self {
        Self::translation(-cx, -cy)
            .then(&Self::rotation(angle))
            .then(&Self::translation(cx, cy))
    }

    /// Create a shear transform.
    ///
    /// # Arguments
    ///
    /// * `shx` - Horizontal shear factor
    /// * `shy` - Vertical shear factor
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::transform::AffineTransform;
    ///
    /// let transform = AffineTransform::shear(0.5, 0.0);
    /// ```
    #[must_use]
    pub const fn shear(shx: f64, shy: f64) -> Self {
        Self::new(1.0, shx, 0.0, shy, 1.0, 0.0)
    }

    /// Apply translation to this transform.
    #[must_use]
    pub fn translate(self, tx: f64, ty: f64) -> Self {
        self.then(&Self::translation(tx, ty))
    }

    /// Apply scaling to this transform.
    #[must_use]
    pub fn scale(self, sx: f64, sy: f64) -> Self {
        self.then(&Self::scaling(sx, sy))
    }

    /// Apply rotation to this transform.
    #[must_use]
    pub fn rotate(self, angle: f64) -> Self {
        self.then(&Self::rotation(angle))
    }

    /// Apply rotation around a center point.
    #[must_use]
    pub fn rotate_around(self, angle: f64, cx: f64, cy: f64) -> Self {
        self.then(&Self::rotation_around(angle, cx, cy))
    }

    /// Compose this transform with another (other * this).
    ///
    /// The resulting transform first applies `self`, then `other`.
    #[must_use]
    pub fn then(&self, other: &Self) -> Self {
        Self {
            a: other.a * self.a + other.b * self.c,
            b: other.a * self.b + other.b * self.d,
            tx: other.a * self.tx + other.b * self.ty + other.tx,
            c: other.c * self.a + other.d * self.c,
            d: other.c * self.b + other.d * self.d,
            ty: other.c * self.tx + other.d * self.ty + other.ty,
        }
    }

    /// Transform a point.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::transform::AffineTransform;
    ///
    /// let transform = AffineTransform::translation(10.0, 20.0);
    /// let (x, y) = transform.transform_point(5.0, 5.0);
    /// assert!((x - 15.0).abs() < 0.001);
    /// assert!((y - 25.0).abs() < 0.001);
    /// ```
    #[must_use]
    pub fn transform_point(&self, x: f64, y: f64) -> (f64, f64) {
        (
            self.a * x + self.b * y + self.tx,
            self.c * x + self.d * y + self.ty,
        )
    }

    /// Transform multiple points.
    #[must_use]
    pub fn transform_points(&self, points: &[(f64, f64)]) -> Vec<(f64, f64)> {
        points
            .iter()
            .map(|&(x, y)| self.transform_point(x, y))
            .collect()
    }

    /// Calculate the determinant of the transform matrix.
    #[must_use]
    pub fn determinant(&self) -> f64 {
        self.a * self.d - self.b * self.c
    }

    /// Check if the transform is invertible.
    #[must_use]
    pub fn is_invertible(&self) -> bool {
        self.determinant().abs() > f64::EPSILON
    }

    /// Calculate the inverse transform.
    ///
    /// # Returns
    ///
    /// The inverse transform, or None if the transform is singular.
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() <= f64::EPSILON {
            return None;
        }

        let inv_det = 1.0 / det;

        Some(Self {
            a: self.d * inv_det,
            b: -self.b * inv_det,
            tx: (self.b * self.ty - self.d * self.tx) * inv_det,
            c: -self.c * inv_det,
            d: self.a * inv_det,
            ty: (self.c * self.tx - self.a * self.ty) * inv_det,
        })
    }

    /// Get the transform as a 3x3 matrix (row-major).
    #[must_use]
    pub const fn as_matrix(&self) -> [[f64; 3]; 3] {
        [
            [self.a, self.b, self.tx],
            [self.c, self.d, self.ty],
            [0.0, 0.0, 1.0],
        ]
    }

    /// Create a transform from a 3x3 matrix (row-major).
    #[must_use]
    pub fn from_matrix(m: [[f64; 3]; 3]) -> Self {
        Self::new(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2])
    }
}

impl Default for AffineTransform {
    fn default() -> Self {
        Self::identity()
    }
}

/// Interpolation method for image transformation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Interpolation {
    /// Nearest neighbor (fastest, but blocky).
    Nearest,
    /// Bilinear interpolation (good balance).
    #[default]
    Bilinear,
    /// Bicubic interpolation (smoother, slower).
    Bicubic,
}

/// Apply an affine transform to an image.
///
/// # Arguments
///
/// * `src` - Source grayscale image data
/// * `src_width` - Source image width
/// * `src_height` - Source image height
/// * `transform` - The affine transformation to apply
/// * `dst_width` - Output image width
/// * `dst_height` - Output image height
/// * `interpolation` - Interpolation method
///
/// # Returns
///
/// Transformed image data.
///
/// # Errors
///
/// Returns an error if dimensions are invalid or transform is not invertible.
///
/// # Examples
///
/// ```
/// use oximedia_cv::transform::{AffineTransform, affine::{transform_image, Interpolation}};
///
/// let src = vec![100u8; 100];
/// let transform = AffineTransform::rotation(std::f64::consts::PI / 4.0);
/// let result = transform_image(&src, 10, 10, &transform, 10, 10, Interpolation::Bilinear)?;
/// ```
pub fn transform_image(
    src: &[u8],
    src_width: u32,
    src_height: u32,
    transform: &AffineTransform,
    dst_width: u32,
    dst_height: u32,
    interpolation: Interpolation,
) -> CvResult<Vec<u8>> {
    if src_width == 0 || src_height == 0 {
        return Err(CvError::invalid_dimensions(src_width, src_height));
    }
    if dst_width == 0 || dst_height == 0 {
        return Err(CvError::invalid_dimensions(dst_width, dst_height));
    }

    let expected_size = src_width as usize * src_height as usize;
    if src.len() < expected_size {
        return Err(CvError::insufficient_data(expected_size, src.len()));
    }

    let inv_transform = transform
        .inverse()
        .ok_or_else(|| CvError::transform_error("Transform is not invertible"))?;

    let dst_size = dst_width as usize * dst_height as usize;
    let mut dst = vec![0u8; dst_size];

    for dy in 0..dst_height {
        for dx in 0..dst_width {
            let (sx, sy) = inv_transform.transform_point(dx as f64, dy as f64);

            let pixel = match interpolation {
                Interpolation::Nearest => sample_nearest(src, src_width, src_height, sx, sy),
                Interpolation::Bilinear => sample_bilinear(src, src_width, src_height, sx, sy),
                Interpolation::Bicubic => sample_bicubic(src, src_width, src_height, sx, sy),
            };

            dst[dy as usize * dst_width as usize + dx as usize] = pixel;
        }
    }

    Ok(dst)
}

/// Sample image using nearest neighbor interpolation.
fn sample_nearest(src: &[u8], width: u32, height: u32, x: f64, y: f64) -> u8 {
    let sx = x.round() as i32;
    let sy = y.round() as i32;

    if sx >= 0 && sx < width as i32 && sy >= 0 && sy < height as i32 {
        src[sy as usize * width as usize + sx as usize]
    } else {
        0
    }
}

/// Sample image using bilinear interpolation.
fn sample_bilinear(src: &[u8], width: u32, height: u32, x: f64, y: f64) -> u8 {
    if x < 0.0 || y < 0.0 || x >= width as f64 - 1.0 || y >= height as f64 - 1.0 {
        return sample_nearest(src, width, height, x, y);
    }

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = x - x0 as f64;
    let fy = y - y0 as f64;

    let w = width as usize;

    let p00 = src[y0 * w + x0] as f64;
    let p10 = src[y0 * w + x1] as f64;
    let p01 = src[y1 * w + x0] as f64;
    let p11 = src[y1 * w + x1] as f64;

    let top = p00 * (1.0 - fx) + p10 * fx;
    let bottom = p01 * (1.0 - fx) + p11 * fx;
    let value = top * (1.0 - fy) + bottom * fy;

    value.round().clamp(0.0, 255.0) as u8
}

/// Sample image using bicubic interpolation.
fn sample_bicubic(src: &[u8], width: u32, height: u32, x: f64, y: f64) -> u8 {
    if x < 1.0 || y < 1.0 || x >= width as f64 - 2.0 || y >= height as f64 - 2.0 {
        return sample_bilinear(src, width, height, x, y);
    }

    let x_int = x.floor() as i32;
    let y_int = y.floor() as i32;
    let fx = x - x_int as f64;
    let fy = y - y_int as f64;

    let w = width as usize;
    let mut value = 0.0;

    for ky in -1..=2 {
        let wy = cubic_weight(ky as f64 - fy);
        let py = (y_int + ky) as usize;

        for kx in -1..=2 {
            let wx = cubic_weight(kx as f64 - fx);
            let px = (x_int + kx) as usize;

            value += src[py * w + px] as f64 * wx * wy;
        }
    }

    value.round().clamp(0.0, 255.0) as u8
}

/// Cubic interpolation weight function (Catmull-Rom).
#[inline]
fn cubic_weight(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        (1.5 * x - 2.5) * x * x + 1.0
    } else if x < 2.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        0.0
    }
}

/// Estimate an affine transform from corresponding point pairs.
///
/// Uses least squares to find the best fit affine transform
/// that maps source points to destination points.
///
/// # Arguments
///
/// * `src_points` - Source points (at least 3)
/// * `dst_points` - Destination points (same length as `src_points`)
///
/// # Returns
///
/// Estimated affine transform.
///
/// # Errors
///
/// Returns an error if fewer than 3 point pairs are provided.
pub fn estimate_affine(
    src_points: &[(f64, f64)],
    dst_points: &[(f64, f64)],
) -> CvResult<AffineTransform> {
    if src_points.len() < 3 || dst_points.len() < 3 {
        return Err(CvError::invalid_parameter(
            "points",
            "need at least 3 point pairs",
        ));
    }

    if src_points.len() != dst_points.len() {
        return Err(CvError::invalid_parameter(
            "points",
            "source and destination must have same length",
        ));
    }

    // For exactly 3 points, solve directly
    // For more points, use least squares

    let n = src_points.len();

    if n == 3 {
        // Direct solution for 3 points
        return solve_affine_3points(src_points, dst_points);
    }

    // Least squares solution
    // Build system: A * x = b
    // where x = [a, b, tx, c, d, ty]^T

    let mut ata = [[0.0; 6]; 6];
    let mut atb = [0.0; 6];

    for i in 0..n {
        let (sx, sy) = src_points[i];
        let (dx, dy) = dst_points[i];

        // For x equation: a*sx + b*sy + tx = dx
        let row_x = [sx, sy, 1.0, 0.0, 0.0, 0.0];
        // For y equation: c*sx + d*sy + ty = dy
        let row_y = [0.0, 0.0, 0.0, sx, sy, 1.0];

        for j in 0..6 {
            for k in 0..6 {
                ata[j][k] += row_x[j] * row_x[k] + row_y[j] * row_y[k];
            }
            atb[j] += row_x[j] * dx + row_y[j] * dy;
        }
    }

    // Solve using Gaussian elimination
    let x = solve_6x6(&ata, &atb)?;

    Ok(AffineTransform::new(x[0], x[1], x[2], x[3], x[4], x[5]))
}

/// Solve affine transform for exactly 3 points.
fn solve_affine_3points(src: &[(f64, f64)], dst: &[(f64, f64)]) -> CvResult<AffineTransform> {
    let (s0x, s0y) = src[0];
    let (s1x, s1y) = src[1];
    let (s2x, s2y) = src[2];

    let (d0x, d0y) = dst[0];
    let (d1x, d1y) = dst[1];
    let (d2x, d2y) = dst[2];

    // Solve for a, b, tx from x equations
    // a*s0x + b*s0y + tx = d0x
    // a*s1x + b*s1y + tx = d1x
    // a*s2x + b*s2y + tx = d2x

    let det = (s0x - s2x) * (s1y - s2y) - (s1x - s2x) * (s0y - s2y);
    if det.abs() < f64::EPSILON {
        return Err(CvError::transform_error("Points are collinear"));
    }

    let inv_det = 1.0 / det;

    let a = ((d0x - d2x) * (s1y - s2y) - (d1x - d2x) * (s0y - s2y)) * inv_det;
    let b = ((s0x - s2x) * (d1x - d2x) - (s1x - s2x) * (d0x - d2x)) * inv_det;
    let tx = d0x - a * s0x - b * s0y;

    // Solve for c, d, ty from y equations
    let c = ((d0y - d2y) * (s1y - s2y) - (d1y - d2y) * (s0y - s2y)) * inv_det;
    let d = ((s0x - s2x) * (d1y - d2y) - (s1x - s2x) * (d0y - d2y)) * inv_det;
    let ty = d0y - c * s0x - d * s0y;

    Ok(AffineTransform::new(a, b, tx, c, d, ty))
}

/// Solve a 6x6 linear system using Gaussian elimination.
fn solve_6x6(a: &[[f64; 6]; 6], b: &[f64; 6]) -> CvResult<[f64; 6]> {
    let mut aug = [[0.0; 7]; 6];

    // Build augmented matrix
    for i in 0..6 {
        for j in 0..6 {
            aug[i][j] = a[i][j];
        }
        aug[i][6] = b[i];
    }

    // Forward elimination
    for i in 0..6 {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..6 {
            if aug[k][i].abs() > aug[max_row][i].abs() {
                max_row = k;
            }
        }

        // Swap rows
        aug.swap(i, max_row);

        if aug[i][i].abs() < f64::EPSILON {
            return Err(CvError::transform_error("Matrix is singular"));
        }

        // Eliminate column
        for k in (i + 1)..6 {
            let factor = aug[k][i] / aug[i][i];
            for j in i..7 {
                aug[k][j] -= factor * aug[i][j];
            }
        }
    }

    // Back substitution
    let mut x = [0.0; 6];
    for i in (0..6).rev() {
        x[i] = aug[i][6];
        for j in (i + 1)..6 {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }

    Ok(x)
}

/// Apply an affine warp to a multi-channel image using inverse mapping with bilinear interpolation.
///
/// `m` is a 2×3 matrix `[[a, b, c], [d, e, f]]` where each destination pixel `(dx, dy)` is
/// mapped back to the source position `(a*dx + b*dy + c, d*dx + e*dy + f)`.  Out-of-bounds
/// source positions are filled with zeros (black / transparent border).
///
/// # Arguments
///
/// * `src` - Source image data (interleaved, `src_w * src_h * channels` bytes)
/// * `src_w` - Source image width in pixels
/// * `src_h` - Source image height in pixels
/// * `channels` - Number of channels per pixel (e.g. 1 for gray, 3 for RGB/BGR)
/// * `m` - 2×3 affine inverse-map matrix
/// * `dst_w` - Destination image width in pixels
/// * `dst_h` - Destination image height in pixels
///
/// # Returns
///
/// Warped image data (`dst_w * dst_h * channels` bytes).
///
/// # Errors
///
/// Returns an error if dimensions are zero or the source buffer is too small.
///
/// # Examples
///
/// ```
/// use oximedia_cv::transform::affine::warp_affine_image;
///
/// // Identity warp on a 4×4 RGB image
/// let src = vec![128u8; 4 * 4 * 3];
/// let identity: [[f64; 3]; 2] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
/// let dst = warp_affine_image(&src, 4, 4, 3, identity, 4, 4)?;
/// assert_eq!(dst.len(), 4 * 4 * 3);
/// ```
pub fn warp_affine_image(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    channels: usize,
    m: [[f64; 3]; 2],
    dst_w: usize,
    dst_h: usize,
) -> CvResult<Vec<u8>> {
    if src_w == 0 || src_h == 0 {
        return Err(CvError::invalid_dimensions(src_w as u32, src_h as u32));
    }
    if dst_w == 0 || dst_h == 0 {
        return Err(CvError::invalid_dimensions(dst_w as u32, dst_h as u32));
    }
    if channels == 0 {
        return Err(CvError::invalid_parameter("channels", "must be > 0"));
    }
    let expected = src_w * src_h * channels;
    if src.len() < expected {
        return Err(CvError::insufficient_data(expected, src.len()));
    }

    let mut out = vec![0u8; dst_w * dst_h * channels];

    // Helper: bilinear sample of a single channel, returning 0.0 for out-of-bounds.
    let sample = |row: i32, col: i32, ch: usize| -> f32 {
        if row < 0 || row >= src_h as i32 || col < 0 || col >= src_w as i32 {
            return 0.0;
        }
        src[(row as usize * src_w + col as usize) * channels + ch] as f32
    };

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            // Inverse map: destination (dx, dy) → source (sx, sy)
            let sx = m[0][0] * dx as f64 + m[0][1] * dy as f64 + m[0][2];
            let sy = m[1][0] * dx as f64 + m[1][1] * dy as f64 + m[1][2];

            let x0 = sx.floor() as i32;
            let y0 = sy.floor() as i32;
            let fx = (sx - sx.floor()) as f32;
            let fy = (sy - sy.floor()) as f32;

            let dst_off = (dy * dst_w + dx) * channels;
            for ch in 0..channels {
                let v00 = sample(y0, x0, ch);
                let v10 = sample(y0, x0 + 1, ch);
                let v01 = sample(y0 + 1, x0, ch);
                let v11 = sample(y0 + 1, x0 + 1, ch);
                let val = v00 * (1.0 - fx) * (1.0 - fy)
                    + v10 * fx * (1.0 - fy)
                    + v01 * (1.0 - fx) * fy
                    + v11 * fx * fy;
                out[dst_off + ch] = val.clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_identity() {
        let transform = AffineTransform::identity();
        let (x, y) = transform.transform_point(10.0, 20.0);
        assert!((x - 10.0).abs() < 0.001);
        assert!((y - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_translation() {
        let transform = AffineTransform::translation(10.0, 20.0);
        let (x, y) = transform.transform_point(5.0, 5.0);
        assert!((x - 15.0).abs() < 0.001);
        assert!((y - 25.0).abs() < 0.001);
    }

    #[test]
    fn test_scaling() {
        let transform = AffineTransform::scaling(2.0, 3.0);
        let (x, y) = transform.transform_point(10.0, 10.0);
        assert!((x - 20.0).abs() < 0.001);
        assert!((y - 30.0).abs() < 0.001);
    }

    #[test]
    fn test_rotation() {
        let transform = AffineTransform::rotation(PI / 2.0);
        let (x, y) = transform.transform_point(1.0, 0.0);
        assert!(x.abs() < 0.001);
        assert!((y - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_rotation_around() {
        let transform = AffineTransform::rotation_around(PI, 50.0, 50.0);
        let (x, y) = transform.transform_point(100.0, 50.0);
        assert!((x - 0.0).abs() < 0.001);
        assert!((y - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_compose() {
        let t1 = AffineTransform::translation(10.0, 0.0);
        let t2 = AffineTransform::scaling(2.0, 2.0);
        let composed = t1.then(&t2);

        let (x, y) = composed.transform_point(5.0, 5.0);
        // First translate: (15, 5)
        // Then scale: (30, 10)
        assert!((x - 30.0).abs() < 0.001);
        assert!((y - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_chain() {
        let transform = AffineTransform::identity()
            .scale(2.0, 2.0)
            .translate(10.0, 10.0);

        let (x, y) = transform.transform_point(5.0, 5.0);
        // Scale: (10, 10)
        // Translate: (20, 20)
        assert!((x - 20.0).abs() < 0.001);
        assert!((y - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_determinant() {
        let transform = AffineTransform::scaling(2.0, 3.0);
        assert!((transform.determinant() - 6.0).abs() < 0.001);
    }

    #[test]
    fn test_inverse() {
        let transform = AffineTransform::translation(10.0, 20.0).scale(2.0, 2.0);
        let inverse = transform.inverse().expect("inverse should succeed");

        let (x, y) = transform.transform_point(5.0, 5.0);
        let (rx, ry) = inverse.transform_point(x, y);

        assert!((rx - 5.0).abs() < 0.001);
        assert!((ry - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_singular_not_invertible() {
        let transform = AffineTransform::new(1.0, 2.0, 0.0, 2.0, 4.0, 0.0);
        assert!(!transform.is_invertible());
        assert!(transform.inverse().is_none());
    }

    #[test]
    fn test_transform_image() {
        let src = vec![100u8; 100];
        let transform = AffineTransform::identity();
        let result = transform_image(&src, 10, 10, &transform, 10, 10, Interpolation::Bilinear)
            .expect("transform_image should succeed");
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_estimate_affine() {
        let src_points = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let dst_points = vec![(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)];

        let transform =
            estimate_affine(&src_points, &dst_points).expect("estimate_affine should succeed");

        // Should be a 2x scale
        let (x, y) = transform.transform_point(1.0, 1.0);
        assert!((x - 2.0).abs() < 0.001);
        assert!((y - 2.0).abs() < 0.001);
    }
}
