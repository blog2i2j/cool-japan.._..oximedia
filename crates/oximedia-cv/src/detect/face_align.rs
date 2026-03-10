//! Face alignment module for normalizing face images based on landmarks.
//!
//! This module provides functionality to align faces to a canonical pose
//! using facial landmarks (typically 5-point or 68-point landmarks).
//! The alignment process involves computing a similarity transform
//! (rotation, scale, translation) to map detected landmarks to reference
//! landmark positions.
//!
//! # Examples
//!
//! ```
//! use oximedia_cv::detect::face_align::{FaceAligner, Point2f, ReferenceTemplate};
//!
//! // Create face aligner with 112x112 output
//! let aligner = FaceAligner::new(112, 112);
//!
//! // Define detected landmarks (5-point)
//! let landmarks = vec![
//!     Point2f::new(30.0, 40.0),  // Left eye
//!     Point2f::new(70.0, 40.0),  // Right eye
//!     Point2f::new(50.0, 60.0),  // Nose
//!     Point2f::new(35.0, 80.0),  // Left mouth
//!     Point2f::new(65.0, 80.0),  // Right mouth
//! ];
//!
//! // Align face (dummy image for example)
//! let image = vec![0u8; 100 * 100 * 3];
//! let aligned = aligner.align(&image, 100, 100, &landmarks)?;
//! ```

use crate::error::{CvError, CvResult};

/// 2D point with floating-point coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2f {
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
}

impl Point2f {
    /// Create a new 2D point.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::face_align::Point2f;
    ///
    /// let point = Point2f::new(10.5, 20.3);
    /// assert_eq!(point.x, 10.5);
    /// assert_eq!(point.y, 20.3);
    /// ```
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Calculate Euclidean distance to another point.
    #[must_use]
    pub fn distance(&self, other: &Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Calculate dot product with another point (as vector).
    #[must_use]
    pub fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y
    }
}

/// 2x3 affine transformation matrix.
///
/// Represents an affine transformation: [x', y'] = [a, b, c; d, e, f] * [x, y, 1]
#[derive(Debug, Clone, Copy)]
pub struct AffineMatrix {
    /// Matrix elements: [[a, b, c], [d, e, f]]
    pub data: [[f32; 3]; 2],
}

impl AffineMatrix {
    /// Create a new affine matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::face_align::AffineMatrix;
    ///
    /// // Identity matrix
    /// let mat = AffineMatrix::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0,
    /// );
    /// ```
    #[must_use]
    pub const fn new(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32) -> Self {
        Self {
            data: [[a, b, c], [d, e, f]],
        }
    }

    /// Create an identity affine matrix.
    #[must_use]
    pub const fn identity() -> Self {
        Self::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    }

    /// Transform a point using this affine matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::face_align::{AffineMatrix, Point2f};
    ///
    /// let mat = AffineMatrix::identity();
    /// let point = Point2f::new(10.0, 20.0);
    /// let transformed = mat.transform_point(&point);
    ///
    /// assert!((transformed.x - 10.0).abs() < 0.001);
    /// assert!((transformed.y - 20.0).abs() < 0.001);
    /// ```
    #[must_use]
    pub fn transform_point(&self, p: &Point2f) -> Point2f {
        Point2f {
            x: self.data[0][0] * p.x + self.data[0][1] * p.y + self.data[0][2],
            y: self.data[1][0] * p.x + self.data[1][1] * p.y + self.data[1][2],
        }
    }

    /// Get the inverse of this affine matrix.
    ///
    /// Returns `None` if the matrix is singular (not invertible).
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        let a = self.data[0][0];
        let b = self.data[0][1];
        let c = self.data[0][2];
        let d = self.data[1][0];
        let e = self.data[1][1];
        let f = self.data[1][2];

        let det = a * e - b * d;

        if det.abs() < 1e-6 {
            return None; // Singular matrix
        }

        let inv_det = 1.0 / det;

        Some(Self::new(
            e * inv_det,
            -b * inv_det,
            (b * f - c * e) * inv_det,
            -d * inv_det,
            a * inv_det,
            (c * d - a * f) * inv_det,
        ))
    }
}

/// Reference face landmark template.
///
/// Defines canonical positions for facial landmarks in a normalized face image.
#[derive(Debug, Clone)]
pub struct ReferenceTemplate {
    /// Reference landmark positions.
    pub landmarks: Vec<Point2f>,
    /// Output image width.
    pub width: u32,
    /// Output image height.
    pub height: u32,
}

impl ReferenceTemplate {
    /// Create a custom reference template.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::face_align::{ReferenceTemplate, Point2f};
    ///
    /// let landmarks = vec![
    ///     Point2f::new(38.0, 45.0),
    ///     Point2f::new(74.0, 45.0),
    ///     Point2f::new(56.0, 70.0),
    ///     Point2f::new(42.0, 88.0),
    ///     Point2f::new(70.0, 88.0),
    /// ];
    /// let template = ReferenceTemplate::new(landmarks, 112, 112);
    /// ```
    #[must_use]
    pub fn new(landmarks: Vec<Point2f>, width: u32, height: u32) -> Self {
        Self {
            landmarks,
            width,
            height,
        }
    }

    /// Create a standard 5-point reference template for face alignment.
    ///
    /// This template is commonly used for face recognition tasks.
    /// Landmark order: left_eye, right_eye, nose, left_mouth, right_mouth
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::face_align::ReferenceTemplate;
    ///
    /// let template = ReferenceTemplate::standard_5_point(112, 112);
    /// assert_eq!(template.landmarks.len(), 5);
    /// ```
    #[must_use]
    pub fn standard_5_point(width: u32, height: u32) -> Self {
        // Standard face template based on MTCNN/ArcFace models
        // Eyes are horizontally aligned, with typical proportions
        let w = width as f32;
        let h = height as f32;

        let landmarks = vec![
            Point2f::new(w * 0.34, h * 0.40), // Left eye
            Point2f::new(w * 0.66, h * 0.40), // Right eye
            Point2f::new(w * 0.50, h * 0.62), // Nose
            Point2f::new(w * 0.38, h * 0.79), // Left mouth
            Point2f::new(w * 0.62, h * 0.79), // Right mouth
        ];

        Self::new(landmarks, width, height)
    }

    /// Create a wider 5-point reference template (more face context).
    #[must_use]
    pub fn wide_5_point(width: u32, height: u32) -> Self {
        let w = width as f32;
        let h = height as f32;

        let landmarks = vec![
            Point2f::new(w * 0.30, h * 0.35),
            Point2f::new(w * 0.70, h * 0.35),
            Point2f::new(w * 0.50, h * 0.55),
            Point2f::new(w * 0.35, h * 0.75),
            Point2f::new(w * 0.65, h * 0.75),
        ];

        Self::new(landmarks, width, height)
    }
}

/// Face alignment engine.
///
/// Performs face normalization by computing and applying similarity transforms
/// to warp detected faces into a canonical pose based on landmark positions.
///
/// # Examples
///
/// ```
/// use oximedia_cv::detect::face_align::{FaceAligner, Point2f};
///
/// let aligner = FaceAligner::new(112, 112);
///
/// let landmarks = vec![
///     Point2f::new(30.0, 40.0),
///     Point2f::new(70.0, 40.0),
///     Point2f::new(50.0, 60.0),
///     Point2f::new(35.0, 80.0),
///     Point2f::new(65.0, 80.0),
/// ];
///
/// let image = vec![128u8; 100 * 100 * 3];
/// let aligned = aligner.align(&image, 100, 100, &landmarks)?;
/// assert_eq!(aligned.len(), 112 * 112 * 3);
/// ```
pub struct FaceAligner {
    /// Reference landmark template.
    reference: ReferenceTemplate,
    /// Interpolation method.
    interpolation: InterpolationMethod,
}

/// Interpolation method for image warping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    /// Nearest neighbor interpolation (fastest).
    Nearest,
    /// Bilinear interpolation (balanced).
    Bilinear,
    /// Bicubic interpolation (highest quality).
    Bicubic,
}

impl FaceAligner {
    /// Create a new face aligner with default settings.
    ///
    /// Uses standard 5-point reference template and bilinear interpolation.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::face_align::FaceAligner;
    ///
    /// let aligner = FaceAligner::new(112, 112);
    /// ```
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            reference: ReferenceTemplate::standard_5_point(width, height),
            interpolation: InterpolationMethod::Bilinear,
        }
    }

    /// Create a face aligner with custom reference template.
    #[must_use]
    pub fn with_template(template: ReferenceTemplate) -> Self {
        Self {
            reference: template,
            interpolation: InterpolationMethod::Bilinear,
        }
    }

    /// Set interpolation method.
    #[must_use]
    pub const fn with_interpolation(mut self, method: InterpolationMethod) -> Self {
        self.interpolation = method;
        self
    }

    /// Align a face based on detected landmarks.
    ///
    /// # Arguments
    ///
    /// * `image` - Source RGB image data (interleaved format)
    /// * `width` - Source image width
    /// * `height` - Source image height
    /// * `landmarks` - Detected facial landmarks
    ///
    /// # Returns
    ///
    /// Aligned face image with size matching the reference template.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Image dimensions are invalid
    /// - Landmark count doesn't match reference
    /// - Transform computation fails
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::detect::face_align::{FaceAligner, Point2f};
    ///
    /// let aligner = FaceAligner::new(112, 112);
    ///
    /// let landmarks = vec![
    ///     Point2f::new(30.0, 40.0),
    ///     Point2f::new(70.0, 40.0),
    ///     Point2f::new(50.0, 60.0),
    ///     Point2f::new(35.0, 80.0),
    ///     Point2f::new(65.0, 80.0),
    /// ];
    ///
    /// let image = vec![128u8; 100 * 100 * 3];
    /// let aligned = aligner.align(&image, 100, 100, &landmarks)?;
    /// assert_eq!(aligned.len(), 112 * 112 * 3);
    /// ```
    pub fn align(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        landmarks: &[Point2f],
    ) -> CvResult<Vec<u8>> {
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let expected_size = (width * height * 3) as usize;
        if image.len() < expected_size {
            return Err(CvError::insufficient_data(expected_size, image.len()));
        }

        if landmarks.len() != self.reference.landmarks.len() {
            return Err(CvError::transform_error(format!(
                "Landmark count mismatch: got {}, expected {}",
                landmarks.len(),
                self.reference.landmarks.len()
            )));
        }

        // Compute similarity transform
        let transform = self.estimate_similarity_transform(landmarks, &self.reference.landmarks)?;

        // Apply transform to warp image
        let aligned = self.warp_affine(
            image,
            width,
            height,
            &transform,
            self.reference.width,
            self.reference.height,
        )?;

        Ok(aligned)
    }

    /// Estimate similarity transform between two sets of landmarks.
    ///
    /// Uses least-squares fitting to find the best similarity transform
    /// (rotation + scale + translation) that maps source landmarks to
    /// destination landmarks.
    fn estimate_similarity_transform(
        &self,
        src: &[Point2f],
        dst: &[Point2f],
    ) -> CvResult<AffineMatrix> {
        if src.is_empty() || src.len() != dst.len() {
            return Err(CvError::transform_error(
                "Source and destination landmark counts must match and be non-empty",
            ));
        }

        // Compute centroids
        let n = src.len() as f32;
        let mut src_centroid = Point2f::new(0.0, 0.0);
        let mut dst_centroid = Point2f::new(0.0, 0.0);

        for (s, d) in src.iter().zip(dst.iter()) {
            src_centroid.x += s.x;
            src_centroid.y += s.y;
            dst_centroid.x += d.x;
            dst_centroid.y += d.y;
        }

        src_centroid.x /= n;
        src_centroid.y /= n;
        dst_centroid.x /= n;
        dst_centroid.y /= n;

        // Compute centered coordinates
        let mut src_centered = Vec::with_capacity(src.len());
        let mut dst_centered = Vec::with_capacity(dst.len());

        for (s, d) in src.iter().zip(dst.iter()) {
            src_centered.push(Point2f::new(s.x - src_centroid.x, s.y - src_centroid.y));
            dst_centered.push(Point2f::new(d.x - dst_centroid.x, d.y - dst_centroid.y));
        }

        // Compute covariance matrix elements
        let mut a = 0.0;
        let mut b = 0.0;
        let mut src_norm_sq = 0.0;

        for (sc, dc) in src_centered.iter().zip(dst_centered.iter()) {
            a += sc.x * dc.x + sc.y * dc.y;
            b += sc.x * dc.y - sc.y * dc.x;
            src_norm_sq += sc.x * sc.x + sc.y * sc.y;
        }

        if src_norm_sq < 1e-6 {
            return Err(CvError::transform_error(
                "Source landmarks are degenerate (all at same point)",
            ));
        }

        // Compute scale and rotation
        let scale = (a * a + b * b).sqrt() / src_norm_sq;
        let cos_theta = a / (a * a + b * b).sqrt();
        let sin_theta = b / (a * a + b * b).sqrt();

        // Build affine matrix: R * S * src + T
        let a_mat = scale * cos_theta;
        let b_mat = -scale * sin_theta;
        let c_mat = dst_centroid.x - (a_mat * src_centroid.x + b_mat * src_centroid.y);
        let d_mat = scale * sin_theta;
        let e_mat = scale * cos_theta;
        let f_mat = dst_centroid.y - (d_mat * src_centroid.x + e_mat * src_centroid.y);

        Ok(AffineMatrix::new(a_mat, b_mat, c_mat, d_mat, e_mat, f_mat))
    }

    /// Apply affine warp to an image.
    ///
    /// # Arguments
    ///
    /// * `image` - Source RGB image
    /// * `src_width` - Source image width
    /// * `src_height` - Source image height
    /// * `transform` - Affine transformation matrix
    /// * `dst_width` - Output image width
    /// * `dst_height` - Output image height
    #[allow(clippy::too_many_arguments)]
    fn warp_affine(
        &self,
        image: &[u8],
        src_width: u32,
        src_height: u32,
        transform: &AffineMatrix,
        dst_width: u32,
        dst_height: u32,
    ) -> CvResult<Vec<u8>> {
        let mut output = vec![0u8; (dst_width * dst_height * 3) as usize];

        // Get inverse transform to map from destination to source
        let inv_transform = transform
            .inverse()
            .ok_or_else(|| CvError::transform_error("Transform is singular"))?;

        match self.interpolation {
            InterpolationMethod::Nearest => {
                self.warp_nearest(
                    image,
                    src_width,
                    src_height,
                    &mut output,
                    dst_width,
                    dst_height,
                    &inv_transform,
                );
            }
            InterpolationMethod::Bilinear => {
                self.warp_bilinear(
                    image,
                    src_width,
                    src_height,
                    &mut output,
                    dst_width,
                    dst_height,
                    &inv_transform,
                );
            }
            InterpolationMethod::Bicubic => {
                self.warp_bicubic(
                    image,
                    src_width,
                    src_height,
                    &mut output,
                    dst_width,
                    dst_height,
                    &inv_transform,
                );
            }
        }

        Ok(output)
    }

    /// Warp using nearest neighbor interpolation.
    #[allow(clippy::too_many_arguments)]
    fn warp_nearest(
        &self,
        src: &[u8],
        src_w: u32,
        src_h: u32,
        dst: &mut [u8],
        dst_w: u32,
        dst_h: u32,
        inv_transform: &AffineMatrix,
    ) {
        for dy in 0..dst_h {
            for dx in 0..dst_w {
                let dst_point = Point2f::new(dx as f32, dy as f32);
                let src_point = inv_transform.transform_point(&dst_point);

                let sx = src_point.x.round() as i32;
                let sy = src_point.y.round() as i32;

                if sx >= 0 && sx < src_w as i32 && sy >= 0 && sy < src_h as i32 {
                    let src_idx = (sy as u32 * src_w + sx as u32) as usize * 3;
                    let dst_idx = (dy * dst_w + dx) as usize * 3;

                    dst[dst_idx..dst_idx + 3].copy_from_slice(&src[src_idx..src_idx + 3]);
                }
            }
        }
    }

    /// Warp using bilinear interpolation.
    #[allow(clippy::too_many_arguments)]
    fn warp_bilinear(
        &self,
        src: &[u8],
        src_w: u32,
        src_h: u32,
        dst: &mut [u8],
        dst_w: u32,
        dst_h: u32,
        inv_transform: &AffineMatrix,
    ) {
        for dy in 0..dst_h {
            for dx in 0..dst_w {
                let dst_point = Point2f::new(dx as f32 + 0.5, dy as f32 + 0.5);
                let src_point = inv_transform.transform_point(&dst_point);

                let sx = src_point.x - 0.5;
                let sy = src_point.y - 0.5;

                let x0 = sx.floor() as i32;
                let y0 = sy.floor() as i32;
                let x1 = x0 + 1;
                let y1 = y0 + 1;

                if x0 >= 0 && x1 < src_w as i32 && y0 >= 0 && y1 < src_h as i32 {
                    let fx = sx - x0 as f32;
                    let fy = sy - y0 as f32;

                    for c in 0..3 {
                        let idx00 = (y0 as u32 * src_w + x0 as u32) as usize * 3 + c;
                        let idx01 = (y0 as u32 * src_w + x1 as u32) as usize * 3 + c;
                        let idx10 = (y1 as u32 * src_w + x0 as u32) as usize * 3 + c;
                        let idx11 = (y1 as u32 * src_w + x1 as u32) as usize * 3 + c;

                        let v00 = f32::from(src[idx00]);
                        let v01 = f32::from(src[idx01]);
                        let v10 = f32::from(src[idx10]);
                        let v11 = f32::from(src[idx11]);

                        let v0 = v00 * (1.0 - fx) + v01 * fx;
                        let v1 = v10 * (1.0 - fx) + v11 * fx;
                        let v = v0 * (1.0 - fy) + v1 * fy;

                        let dst_idx = (dy * dst_w + dx) as usize * 3 + c;
                        dst[dst_idx] = v.round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
    }

    /// Warp using bicubic interpolation.
    #[allow(clippy::too_many_arguments)]
    fn warp_bicubic(
        &self,
        src: &[u8],
        src_w: u32,
        src_h: u32,
        dst: &mut [u8],
        dst_w: u32,
        dst_h: u32,
        inv_transform: &AffineMatrix,
    ) {
        for dy in 0..dst_h {
            for dx in 0..dst_w {
                let dst_point = Point2f::new(dx as f32 + 0.5, dy as f32 + 0.5);
                let src_point = inv_transform.transform_point(&dst_point);

                let sx = src_point.x - 0.5;
                let sy = src_point.y - 0.5;

                let x0 = sx.floor() as i32;
                let y0 = sy.floor() as i32;

                // Check bounds for 4x4 neighborhood
                if x0 >= 1 && x0 + 2 < src_w as i32 && y0 >= 1 && y0 + 2 < src_h as i32 {
                    let fx = sx - x0 as f32;
                    let fy = sy - y0 as f32;

                    for c in 0..3 {
                        let mut value = 0.0;

                        // Bicubic kernel
                        for j in -1..=2 {
                            for i in -1..=2 {
                                let px = x0 + i;
                                let py = y0 + j;
                                let idx = (py as u32 * src_w + px as u32) as usize * 3 + c;
                                let pixel = f32::from(src[idx]);

                                let wx = cubic_weight(fx - i as f32);
                                let wy = cubic_weight(fy - j as f32);

                                value += pixel * wx * wy;
                            }
                        }

                        let dst_idx = (dy * dst_w + dx) as usize * 3 + c;
                        dst[dst_idx] = value.round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
    }
}

/// Cubic interpolation weight function.
///
/// Uses the bicubic kernel: f(x) = { 1 - 2|x|^2 + |x|^3 for |x| < 1,
///                                   4 - 8|x| + 5|x|^2 - |x|^3 for 1 <= |x| < 2,
///                                   0 otherwise }
fn cubic_weight(x: f32) -> f32 {
    let x = x.abs();
    if x < 1.0 {
        1.0 - 2.0 * x * x + x * x * x
    } else if x < 2.0 {
        4.0 - 8.0 * x + 5.0 * x * x - x * x * x
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point2f_new() {
        let p = Point2f::new(10.5, 20.3);
        assert_eq!(p.x, 10.5);
        assert_eq!(p.y, 20.3);
    }

    #[test]
    fn test_point2f_distance() {
        let p1 = Point2f::new(0.0, 0.0);
        let p2 = Point2f::new(3.0, 4.0);
        assert!((p1.distance(&p2) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_point2f_dot() {
        let p1 = Point2f::new(1.0, 2.0);
        let p2 = Point2f::new(3.0, 4.0);
        assert_eq!(p1.dot(&p2), 11.0);
    }

    #[test]
    fn test_affine_matrix_identity() {
        let mat = AffineMatrix::identity();
        let p = Point2f::new(10.0, 20.0);
        let transformed = mat.transform_point(&p);

        assert!((transformed.x - 10.0).abs() < 0.001);
        assert!((transformed.y - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_affine_matrix_translation() {
        let mat = AffineMatrix::new(1.0, 0.0, 5.0, 0.0, 1.0, 10.0);
        let p = Point2f::new(0.0, 0.0);
        let transformed = mat.transform_point(&p);

        assert_eq!(transformed.x, 5.0);
        assert_eq!(transformed.y, 10.0);
    }

    #[test]
    fn test_affine_matrix_scale() {
        let mat = AffineMatrix::new(2.0, 0.0, 0.0, 0.0, 2.0, 0.0);
        let p = Point2f::new(10.0, 20.0);
        let transformed = mat.transform_point(&p);

        assert_eq!(transformed.x, 20.0);
        assert_eq!(transformed.y, 40.0);
    }

    #[test]
    fn test_affine_matrix_inverse() {
        let mat = AffineMatrix::new(2.0, 0.0, 5.0, 0.0, 2.0, 10.0);
        let inv = mat.inverse().expect("inverse should succeed");

        let p = Point2f::new(10.0, 20.0);
        let transformed = mat.transform_point(&p);
        let back = inv.transform_point(&transformed);

        assert!((back.x - 10.0).abs() < 0.001);
        assert!((back.y - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_reference_template_standard() {
        let template = ReferenceTemplate::standard_5_point(112, 112);
        assert_eq!(template.landmarks.len(), 5);
        assert_eq!(template.width, 112);
        assert_eq!(template.height, 112);
    }

    #[test]
    fn test_reference_template_wide() {
        let template = ReferenceTemplate::wide_5_point(112, 112);
        assert_eq!(template.landmarks.len(), 5);
    }

    #[test]
    fn test_face_aligner_new() {
        let aligner = FaceAligner::new(112, 112);
        assert_eq!(aligner.reference.width, 112);
        assert_eq!(aligner.reference.height, 112);
        assert_eq!(aligner.interpolation, InterpolationMethod::Bilinear);
    }

    #[test]
    fn test_face_aligner_with_interpolation() {
        let aligner = FaceAligner::new(112, 112).with_interpolation(InterpolationMethod::Bicubic);
        assert_eq!(aligner.interpolation, InterpolationMethod::Bicubic);
    }

    #[test]
    fn test_estimate_similarity_transform_identity() {
        let aligner = FaceAligner::new(112, 112);

        let landmarks = vec![
            Point2f::new(38.0, 45.0),
            Point2f::new(74.0, 45.0),
            Point2f::new(56.0, 70.0),
            Point2f::new(42.0, 88.0),
            Point2f::new(70.0, 88.0),
        ];

        // Transform landmarks to themselves should give identity
        let transform = aligner
            .estimate_similarity_transform(&landmarks, &landmarks)
            .expect("operation should succeed");

        let p = Point2f::new(50.0, 50.0);
        let transformed = transform.transform_point(&p);

        assert!((transformed.x - 50.0).abs() < 0.1);
        assert!((transformed.y - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_face_aligner_align() {
        let aligner = FaceAligner::new(112, 112);

        // Create a simple gradient image
        let src_w = 100;
        let src_h = 100;
        let mut image = vec![0u8; src_w * src_h * 3];
        for y in 0..src_h {
            for x in 0..src_w {
                let idx = (y * src_w + x) * 3;
                image[idx] = ((x + y) % 256) as u8;
                image[idx + 1] = ((x + y) % 256) as u8;
                image[idx + 2] = ((x + y) % 256) as u8;
            }
        }

        let landmarks = vec![
            Point2f::new(30.0, 40.0),
            Point2f::new(70.0, 40.0),
            Point2f::new(50.0, 60.0),
            Point2f::new(35.0, 80.0),
            Point2f::new(65.0, 80.0),
        ];

        let aligned = aligner
            .align(&image, src_w as u32, src_h as u32, &landmarks)
            .expect("operation should succeed");

        assert_eq!(aligned.len(), 112 * 112 * 3);
    }

    #[test]
    fn test_face_aligner_align_invalid_dimensions() {
        let aligner = FaceAligner::new(112, 112);
        let image = vec![0u8; 100];
        let landmarks = vec![Point2f::new(0.0, 0.0)];

        let result = aligner.align(&image, 0, 0, &landmarks);
        assert!(result.is_err());
    }

    #[test]
    fn test_face_aligner_align_insufficient_data() {
        let aligner = FaceAligner::new(112, 112);
        let image = vec![0u8; 10];
        let landmarks = vec![Point2f::new(0.0, 0.0)];

        let result = aligner.align(&image, 100, 100, &landmarks);
        assert!(result.is_err());
    }

    #[test]
    fn test_face_aligner_landmark_mismatch() {
        let aligner = FaceAligner::new(112, 112);
        let image = vec![0u8; 100 * 100 * 3];

        // Only 3 landmarks instead of 5
        let landmarks = vec![
            Point2f::new(30.0, 40.0),
            Point2f::new(70.0, 40.0),
            Point2f::new(50.0, 60.0),
        ];

        let result = aligner.align(&image, 100, 100, &landmarks);
        assert!(result.is_err());
    }

    #[test]
    fn test_cubic_weight() {
        assert_eq!(cubic_weight(0.0), 1.0);
        assert_eq!(cubic_weight(2.0), 0.0);
        assert!(cubic_weight(0.5) > 0.0);
        assert!(cubic_weight(0.5) < 1.0);
    }

    #[test]
    fn test_interpolation_methods() {
        let aligner_nearest =
            FaceAligner::new(64, 64).with_interpolation(InterpolationMethod::Nearest);
        let aligner_bilinear =
            FaceAligner::new(64, 64).with_interpolation(InterpolationMethod::Bilinear);
        let aligner_bicubic =
            FaceAligner::new(64, 64).with_interpolation(InterpolationMethod::Bicubic);

        let image = vec![128u8; 100 * 100 * 3];
        let landmarks = vec![
            Point2f::new(30.0, 40.0),
            Point2f::new(70.0, 40.0),
            Point2f::new(50.0, 60.0),
            Point2f::new(35.0, 80.0),
            Point2f::new(65.0, 80.0),
        ];

        let aligned_nearest = aligner_nearest
            .align(&image, 100, 100, &landmarks)
            .expect("align should succeed");
        let aligned_bilinear = aligner_bilinear
            .align(&image, 100, 100, &landmarks)
            .expect("operation should succeed");
        let aligned_bicubic = aligner_bicubic
            .align(&image, 100, 100, &landmarks)
            .expect("align should succeed");

        assert_eq!(aligned_nearest.len(), 64 * 64 * 3);
        assert_eq!(aligned_bilinear.len(), 64 * 64 * 3);
        assert_eq!(aligned_bicubic.len(), 64 * 64 * 3);
    }
}
