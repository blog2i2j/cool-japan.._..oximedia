//! Perspective transformation module.
//!
//! This module provides perspective (projective) transformations
//! and homography computation for applications like image warping
//! and perspective correction.
//!
//! # Example
//!
//! ```
//! use oximedia_cv::transform::PerspectiveTransform;
//!
//! let transform = PerspectiveTransform::identity();
//! let (x, y) = transform.transform_point(10.0, 20.0);
//! ```

use crate::error::{CvError, CvResult};

/// 2D perspective (projective) transformation matrix (3x3).
///
/// The matrix is:
/// ```text
/// | h00  h01  h02 |
/// | h10  h11  h12 |
/// | h20  h21  h22 |
/// ```
///
/// A point (x, y) is transformed to (x', y') by:
/// ```text
/// w = h20*x + h21*y + h22
/// x' = (h00*x + h01*y + h02) / w
/// y' = (h10*x + h11*y + h12) / w
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerspectiveTransform {
    /// Transformation matrix elements (row-major).
    pub matrix: [[f64; 3]; 3],
}

impl PerspectiveTransform {
    /// Create a new perspective transform from matrix elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::transform::PerspectiveTransform;
    ///
    /// let transform = PerspectiveTransform::new([
    ///     [1.0, 0.0, 0.0],
    ///     [0.0, 1.0, 0.0],
    ///     [0.0, 0.0, 1.0],
    /// ]);
    /// ```
    #[must_use]
    pub const fn new(matrix: [[f64; 3]; 3]) -> Self {
        Self { matrix }
    }

    /// Create an identity transform.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::transform::PerspectiveTransform;
    ///
    /// let transform = PerspectiveTransform::identity();
    /// let (x, y) = transform.transform_point(10.0, 20.0);
    /// assert!((x - 10.0).abs() < 0.001);
    /// ```
    #[must_use]
    pub const fn identity() -> Self {
        Self::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    }

    /// Transform a point.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::transform::PerspectiveTransform;
    ///
    /// let transform = PerspectiveTransform::identity();
    /// let (x, y) = transform.transform_point(10.0, 20.0);
    /// assert!((x - 10.0).abs() < 0.001);
    /// ```
    #[must_use]
    pub fn transform_point(&self, x: f64, y: f64) -> (f64, f64) {
        let m = &self.matrix;

        let w = m[2][0] * x + m[2][1] * y + m[2][2];
        if w.abs() < f64::EPSILON {
            return (f64::NAN, f64::NAN);
        }

        let x_new = (m[0][0] * x + m[0][1] * y + m[0][2]) / w;
        let y_new = (m[1][0] * x + m[1][1] * y + m[1][2]) / w;

        (x_new, y_new)
    }

    /// Transform multiple points.
    #[must_use]
    pub fn transform_points(&self, points: &[(f64, f64)]) -> Vec<(f64, f64)> {
        points
            .iter()
            .map(|&(x, y)| self.transform_point(x, y))
            .collect()
    }

    /// Calculate the determinant of the matrix.
    #[must_use]
    pub fn determinant(&self) -> f64 {
        let m = &self.matrix;

        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
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

        let m = &self.matrix;
        let inv_det = 1.0 / det;

        // Compute adjugate matrix and divide by determinant
        let inv = [
            [
                (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
                (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
                (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
            ],
            [
                (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
                (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
                (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
            ],
            [
                (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
                (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
                (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
            ],
        ];

        Some(Self::new(inv))
    }

    /// Compose this transform with another (other * this).
    ///
    /// The resulting transform first applies `self`, then `other`.
    #[must_use]
    pub fn then(&self, other: &Self) -> Self {
        let a = &other.matrix;
        let b = &self.matrix;

        let mut result = [[0.0; 3]; 3];

        for i in 0..3 {
            for j in 0..3 {
                result[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
            }
        }

        Self::new(result)
    }

    /// Normalize the matrix so that h22 = 1 (if non-zero).
    #[must_use]
    pub fn normalized(&self) -> Self {
        let h22 = self.matrix[2][2];
        if h22.abs() < f64::EPSILON {
            return *self;
        }

        let mut result = self.matrix;
        for row in &mut result {
            for elem in row {
                *elem /= h22;
            }
        }

        Self::new(result)
    }
}

impl Default for PerspectiveTransform {
    fn default() -> Self {
        Self::identity()
    }
}

/// Find homography matrix that maps source points to destination points.
///
/// Uses the Direct Linear Transform (DLT) algorithm to compute
/// the 3x3 homography matrix.
///
/// # Arguments
///
/// * `src_points` - Source points (at least 4)
/// * `dst_points` - Destination points (same length as `src_points`)
///
/// # Returns
///
/// Estimated perspective transform.
///
/// # Errors
///
/// Returns an error if fewer than 4 point pairs are provided.
///
/// # Examples
///
/// ```
/// use oximedia_cv::transform::perspective::find_homography;
///
/// let src = vec![(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)];
/// let dst = vec![(10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)];
///
/// let transform = find_homography(&src, &dst)?;
/// ```
pub fn find_homography(
    src_points: &[(f64, f64)],
    dst_points: &[(f64, f64)],
) -> CvResult<PerspectiveTransform> {
    if src_points.len() < 4 || dst_points.len() < 4 {
        return Err(CvError::invalid_parameter(
            "points",
            "need at least 4 point pairs",
        ));
    }

    if src_points.len() != dst_points.len() {
        return Err(CvError::invalid_parameter(
            "points",
            "source and destination must have same length",
        ));
    }

    let n = src_points.len();

    // Build the 2n x 9 matrix A for DLT
    // Each point pair contributes 2 rows:
    // [-x, -y, -1, 0, 0, 0, x*x', y*x', x']
    // [0, 0, 0, -x, -y, -1, x*y', y*y', y']

    // For simplicity, we solve the 8x8 system (fixing h22 = 1)
    // This works well for most cases

    let mut a = [[0.0; 8]; 8];
    let mut b = [0.0; 8];

    // Use first 4 points (exactly determined system)
    for i in 0..4.min(n) {
        let (sx, sy) = src_points[i];
        let (dx, dy) = dst_points[i];

        let row_x = 2 * i;
        let row_y = 2 * i + 1;

        // x' = (h00*x + h01*y + h02) / (h20*x + h21*y + 1)
        // x'*(h20*x + h21*y + 1) = h00*x + h01*y + h02
        // h00*x + h01*y + h02 - x'*h20*x - x'*h21*y = x'

        a[row_x][0] = sx;
        a[row_x][1] = sy;
        a[row_x][2] = 1.0;
        a[row_x][3] = 0.0;
        a[row_x][4] = 0.0;
        a[row_x][5] = 0.0;
        a[row_x][6] = -dx * sx;
        a[row_x][7] = -dx * sy;
        b[row_x] = dx;

        // y' = (h10*x + h11*y + h12) / (h20*x + h21*y + 1)
        a[row_y][0] = 0.0;
        a[row_y][1] = 0.0;
        a[row_y][2] = 0.0;
        a[row_y][3] = sx;
        a[row_y][4] = sy;
        a[row_y][5] = 1.0;
        a[row_y][6] = -dy * sx;
        a[row_y][7] = -dy * sy;
        b[row_y] = dy;
    }

    // Solve using Gaussian elimination
    let h = solve_8x8(&a, &b)?;

    let matrix = [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]];

    Ok(PerspectiveTransform::new(matrix))
}

/// Solve an 8x8 linear system using Gaussian elimination.
fn solve_8x8(a: &[[f64; 8]; 8], b: &[f64; 8]) -> CvResult<[f64; 8]> {
    let mut aug = [[0.0; 9]; 8];

    // Build augmented matrix
    for i in 0..8 {
        for j in 0..8 {
            aug[i][j] = a[i][j];
        }
        aug[i][8] = b[i];
    }

    // Forward elimination with partial pivoting
    for i in 0..8 {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..8 {
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
        for k in (i + 1)..8 {
            let factor = aug[k][i] / aug[i][i];
            for j in i..9 {
                aug[k][j] -= factor * aug[i][j];
            }
        }
    }

    // Back substitution
    let mut x = [0.0; 8];
    for i in (0..8).rev() {
        x[i] = aug[i][8];
        for j in (i + 1)..8 {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }

    Ok(x)
}

/// Apply perspective warp to an image.
///
/// # Arguments
///
/// * `src` - Source grayscale image data
/// * `src_width` - Source image width
/// * `src_height` - Source image height
/// * `transform` - The perspective transformation
/// * `dst_width` - Output image width
/// * `dst_height` - Output image height
///
/// # Returns
///
/// Warped image data.
///
/// # Errors
///
/// Returns an error if dimensions are invalid or transform is not invertible.
///
/// # Examples
///
/// ```
/// use oximedia_cv::transform::{PerspectiveTransform, perspective::warp_perspective};
///
/// let src = vec![100u8; 100];
/// let transform = PerspectiveTransform::identity();
/// let result = warp_perspective(&src, 10, 10, &transform, 10, 10)?;
/// ```
pub fn warp_perspective(
    src: &[u8],
    src_width: u32,
    src_height: u32,
    transform: &PerspectiveTransform,
    dst_width: u32,
    dst_height: u32,
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

            if sx.is_nan() || sy.is_nan() {
                continue;
            }

            // Bilinear interpolation
            let pixel = sample_bilinear(src, src_width, src_height, sx, sy);
            dst[dy as usize * dst_width as usize + dx as usize] = pixel;
        }
    }

    Ok(dst)
}

/// Sample image using bilinear interpolation.
fn sample_bilinear(src: &[u8], width: u32, height: u32, x: f64, y: f64) -> u8 {
    if x < 0.0 || y < 0.0 || x >= width as f64 - 1.0 || y >= height as f64 - 1.0 {
        // Out of bounds - use nearest neighbor for edge handling
        let sx = x.round().clamp(0.0, (width - 1) as f64) as usize;
        let sy = y.round().clamp(0.0, (height - 1) as f64) as usize;

        if sx < width as usize && sy < height as usize {
            return src[sy * width as usize + sx];
        }
        return 0;
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

/// Get the perspective transform for a quadrilateral to rectangle mapping.
///
/// Useful for perspective correction (e.g., document scanning).
///
/// # Arguments
///
/// * `quad` - Four corners of the source quadrilateral (top-left, top-right, bottom-right, bottom-left)
/// * `rect_width` - Target rectangle width
/// * `rect_height` - Target rectangle height
///
/// # Returns
///
/// Perspective transform that maps the quadrilateral to a rectangle.
///
/// # Errors
///
/// Returns an error if the homography cannot be computed.
pub fn quad_to_rect(
    quad: &[(f64, f64); 4],
    rect_width: f64,
    rect_height: f64,
) -> CvResult<PerspectiveTransform> {
    let src = quad.to_vec();
    let dst = vec![
        (0.0, 0.0),
        (rect_width, 0.0),
        (rect_width, rect_height),
        (0.0, rect_height),
    ];

    find_homography(&src, &dst)
}

/// Get the perspective transform for a rectangle to quadrilateral mapping.
///
/// # Arguments
///
/// * `rect_width` - Source rectangle width
/// * `rect_height` - Source rectangle height
/// * `quad` - Four corners of the target quadrilateral
///
/// # Returns
///
/// Perspective transform that maps a rectangle to the quadrilateral.
///
/// # Errors
///
/// Returns an error if the homography cannot be computed.
pub fn rect_to_quad(
    rect_width: f64,
    rect_height: f64,
    quad: &[(f64, f64); 4],
) -> CvResult<PerspectiveTransform> {
    let src = vec![
        (0.0, 0.0),
        (rect_width, 0.0),
        (rect_width, rect_height),
        (0.0, rect_height),
    ];
    let dst = quad.to_vec();

    find_homography(&src, &dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let transform = PerspectiveTransform::identity();
        let (x, y) = transform.transform_point(10.0, 20.0);
        assert!((x - 10.0).abs() < 0.001);
        assert!((y - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_determinant() {
        let transform = PerspectiveTransform::identity();
        assert!((transform.determinant() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_inverse() {
        let transform =
            PerspectiveTransform::new([[2.0, 0.0, 10.0], [0.0, 2.0, 20.0], [0.0, 0.0, 1.0]]);

        let inverse = transform.inverse().expect("inverse should succeed");

        let (x, y) = transform.transform_point(5.0, 5.0);
        let (rx, ry) = inverse.transform_point(x, y);

        assert!((rx - 5.0).abs() < 0.001);
        assert!((ry - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_compose() {
        let t1 = PerspectiveTransform::new([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let t2 = PerspectiveTransform::new([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]);

        let composed = t1.then(&t2);
        let (x, y) = composed.transform_point(5.0, 5.0);

        // First translate: (15, 5)
        // Then scale: (30, 10)
        assert!((x - 30.0).abs() < 0.001);
        assert!((y - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_find_homography() {
        let src = vec![(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)];
        let dst = vec![(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)];

        let transform = find_homography(&src, &dst).expect("find_homography should succeed");

        // Should be close to identity
        let (x, y) = transform.transform_point(50.0, 50.0);
        assert!((x - 50.0).abs() < 0.01);
        assert!((y - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_find_homography_scale() {
        let src = vec![(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)];
        let dst = vec![(0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (0.0, 200.0)];

        let transform = find_homography(&src, &dst).expect("find_homography should succeed");

        // Should scale by 2
        let (x, y) = transform.transform_point(50.0, 50.0);
        assert!((x - 100.0).abs() < 0.01);
        assert!((y - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_warp_perspective() {
        let src = vec![100u8; 100];
        let transform = PerspectiveTransform::identity();
        let result = warp_perspective(&src, 10, 10, &transform, 10, 10)
            .expect("warp_perspective should succeed");
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_quad_to_rect() {
        let quad = [(10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)];

        let transform = quad_to_rect(&quad, 100.0, 100.0).expect("quad_to_rect should succeed");

        // Top-left corner should map to (0, 0)
        let (x, y) = transform.transform_point(10.0, 10.0);
        assert!(x.abs() < 0.1);
        assert!(y.abs() < 0.1);

        // Bottom-right should map to (100, 100)
        let (x, y) = transform.transform_point(90.0, 90.0);
        assert!((x - 100.0).abs() < 0.1);
        assert!((y - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_normalized() {
        let transform =
            PerspectiveTransform::new([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]);

        let normalized = transform.normalized();
        assert!((normalized.matrix[2][2] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_insufficient_points() {
        let src = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)];
        let dst = vec![(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)];

        assert!(find_homography(&src, &dst).is_err());
    }
}
