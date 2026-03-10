//! Video alignment and registration tools.
//!
//! This module provides comprehensive video and image registration capabilities:
//!
//! - **Feature-based registration**: SIFT, SURF, ORB feature detection and matching
//! - **Phase correlation**: FFT-based sub-pixel alignment
//! - **Optical flow**: Dense and sparse flow-based registration
//! - **ECC alignment**: Enhanced Correlation Coefficient optimization
//! - **Image warping**: Multiple interpolation methods
//! - **Video stabilization**: Motion smoothing and rolling shutter correction
//! - **Camera calibration**: Lens distortion correction
//!
//! # Example
//!
//! ```
//! use oximedia_cv::registration::{VideoRegistration, RegistrationMethod};
//!
//! let registration = VideoRegistration::new(RegistrationMethod::PhaseCorrelation);
//! ```

pub mod calibration;
pub mod ecc;
pub mod feature_based;
pub mod optical_flow;
pub mod phase_correlation;
pub mod stabilization;
pub mod warp;

use crate::error::{CvError, CvResult};

/// Video and image registration methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegistrationMethod {
    /// Feature-based registration using keypoint detection and matching.
    FeatureBased,
    /// Phase correlation using FFT for sub-pixel translation estimation.
    PhaseCorrelation,
    /// Optical flow-based registration.
    OpticalFlow,
    /// Enhanced Correlation Coefficient (ECC) iterative optimization.
    Ecc,
    /// Hybrid approach combining multiple methods.
    Hybrid,
}

/// Transformation type for image alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformationType {
    /// Translation only (2 DOF).
    Translation,
    /// Euclidean (translation + rotation, 3 DOF).
    Euclidean,
    /// Affine (6 DOF).
    Affine,
    /// Homography/Perspective (8 DOF).
    Homography,
}

/// Registration quality metrics.
#[derive(Debug, Clone)]
pub struct RegistrationQuality {
    /// Registration success flag.
    pub success: bool,
    /// Root mean square error.
    pub rmse: f64,
    /// Number of inliers (for feature-based methods).
    pub inliers: usize,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f64,
    /// Iterations taken (for iterative methods).
    pub iterations: usize,
}

impl RegistrationQuality {
    /// Create a new quality metric.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            success: false,
            rmse: 0.0,
            inliers: 0,
            confidence: 0.0,
            iterations: 0,
        }
    }

    /// Check if registration is of good quality.
    #[must_use]
    pub fn is_good(&self) -> bool {
        self.success && self.confidence > 0.7 && self.rmse < 1.0
    }
}

impl Default for RegistrationQuality {
    fn default() -> Self {
        Self::new()
    }
}

/// Transformation matrix representation.
///
/// For 2D transformations, this represents a 3x3 matrix:
/// ```text
/// [a b c]   [x]
/// [d e f] * [y]
/// [g h i]   [1]
/// ```
#[derive(Debug, Clone)]
pub struct TransformMatrix {
    /// Matrix elements in row-major order.
    pub data: [f64; 9],
    /// Transformation type.
    pub transform_type: TransformationType,
}

impl TransformMatrix {
    /// Create an identity transformation.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            transform_type: TransformationType::Affine,
        }
    }

    /// Create a translation matrix.
    #[must_use]
    pub fn translation(tx: f64, ty: f64) -> Self {
        Self {
            data: [1.0, 0.0, tx, 0.0, 1.0, ty, 0.0, 0.0, 1.0],
            transform_type: TransformationType::Translation,
        }
    }

    /// Create a rotation matrix (angle in radians).
    #[must_use]
    pub fn rotation(angle: f64) -> Self {
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        Self {
            data: [cos_a, -sin_a, 0.0, sin_a, cos_a, 0.0, 0.0, 0.0, 1.0],
            transform_type: TransformationType::Euclidean,
        }
    }

    /// Create a scale matrix.
    #[must_use]
    pub fn scale(sx: f64, sy: f64) -> Self {
        Self {
            data: [sx, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 1.0],
            transform_type: TransformationType::Affine,
        }
    }

    /// Transform a point.
    #[must_use]
    pub fn transform_point(&self, x: f64, y: f64) -> (f64, f64) {
        let w = self.data[6] * x + self.data[7] * y + self.data[8];
        if w.abs() < f64::EPSILON {
            return (x, y);
        }
        let x_new = (self.data[0] * x + self.data[1] * y + self.data[2]) / w;
        let y_new = (self.data[3] * x + self.data[4] * y + self.data[5]) / w;
        (x_new, y_new)
    }

    /// Compose two transformations (self * other).
    #[must_use]
    pub fn compose(&self, other: &TransformMatrix) -> Self {
        let mut result = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    result[i * 3 + j] += self.data[i * 3 + k] * other.data[k * 3 + j];
                }
            }
        }
        Self {
            data: result,
            transform_type: TransformationType::Homography,
        }
    }

    /// Compute inverse transformation.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is singular.
    pub fn inverse(&self) -> CvResult<Self> {
        let det = self.determinant();
        if det.abs() < f64::EPSILON {
            return Err(CvError::computation("singular matrix, cannot invert"));
        }

        let inv_det = 1.0 / det;
        let a = self.data[0];
        let b = self.data[1];
        let c = self.data[2];
        let d = self.data[3];
        let e = self.data[4];
        let f = self.data[5];
        let g = self.data[6];
        let h = self.data[7];
        let i = self.data[8];

        let mut inv = [0.0; 9];
        inv[0] = (e * i - f * h) * inv_det;
        inv[1] = (c * h - b * i) * inv_det;
        inv[2] = (b * f - c * e) * inv_det;
        inv[3] = (f * g - d * i) * inv_det;
        inv[4] = (a * i - c * g) * inv_det;
        inv[5] = (c * d - a * f) * inv_det;
        inv[6] = (d * h - e * g) * inv_det;
        inv[7] = (b * g - a * h) * inv_det;
        inv[8] = (a * e - b * d) * inv_det;

        Ok(Self {
            data: inv,
            transform_type: self.transform_type,
        })
    }

    /// Compute matrix determinant.
    #[must_use]
    pub fn determinant(&self) -> f64 {
        let a = self.data[0];
        let b = self.data[1];
        let c = self.data[2];
        let d = self.data[3];
        let e = self.data[4];
        let f = self.data[5];
        let g = self.data[6];
        let h = self.data[7];
        let i = self.data[8];

        a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    }

    /// Extract translation component.
    #[must_use]
    pub fn get_translation(&self) -> (f64, f64) {
        (self.data[2], self.data[5])
    }

    /// Extract rotation angle (for Euclidean transforms).
    #[must_use]
    pub fn get_rotation(&self) -> f64 {
        self.data[3].atan2(self.data[0])
    }

    /// Extract scale factors.
    #[must_use]
    pub fn get_scale(&self) -> (f64, f64) {
        let sx = (self.data[0] * self.data[0] + self.data[3] * self.data[3]).sqrt();
        let sy = (self.data[1] * self.data[1] + self.data[4] * self.data[4]).sqrt();
        (sx, sy)
    }
}

/// Main video registration interface.
///
/// Provides a unified API for various registration methods.
///
/// # Example
///
/// ```
/// use oximedia_cv::registration::{VideoRegistration, RegistrationMethod};
///
/// let registration = VideoRegistration::new(RegistrationMethod::PhaseCorrelation);
/// ```
#[derive(Debug, Clone)]
pub struct VideoRegistration {
    /// Registration method.
    method: RegistrationMethod,
    /// Maximum iterations for iterative methods.
    max_iterations: usize,
    /// Convergence threshold.
    convergence_threshold: f64,
    /// Use multi-resolution pyramid.
    use_pyramid: bool,
    /// Number of pyramid levels.
    pyramid_levels: usize,
}

impl VideoRegistration {
    /// Create a new video registration instance.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_cv::registration::{VideoRegistration, RegistrationMethod};
    ///
    /// let reg = VideoRegistration::new(RegistrationMethod::Ecc);
    /// ```
    #[must_use]
    pub const fn new(method: RegistrationMethod) -> Self {
        Self {
            method,
            max_iterations: 100,
            convergence_threshold: 1e-6,
            use_pyramid: true,
            pyramid_levels: 3,
        }
    }

    /// Set maximum iterations.
    #[must_use]
    pub const fn with_max_iterations(mut self, iterations: usize) -> Self {
        self.max_iterations = iterations;
        self
    }

    /// Set convergence threshold.
    #[must_use]
    pub const fn with_convergence_threshold(mut self, threshold: f64) -> Self {
        self.convergence_threshold = threshold;
        self
    }

    /// Enable/disable multi-resolution pyramid.
    #[must_use]
    pub const fn with_pyramid(mut self, enabled: bool, levels: usize) -> Self {
        self.use_pyramid = enabled;
        self.pyramid_levels = levels;
        self
    }

    /// Register two images.
    ///
    /// # Arguments
    ///
    /// * `reference` - Reference (template) image (grayscale)
    /// * `target` - Target image to align (grayscale)
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `transform_type` - Type of transformation to estimate
    ///
    /// # Returns
    ///
    /// Transformation matrix and quality metrics.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid or registration fails.
    #[allow(clippy::too_many_arguments)]
    pub fn register(
        &self,
        reference: &[u8],
        target: &[u8],
        width: u32,
        height: u32,
        transform_type: TransformationType,
    ) -> CvResult<(TransformMatrix, RegistrationQuality)> {
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let size = width as usize * height as usize;
        if reference.len() < size || target.len() < size {
            return Err(CvError::insufficient_data(size, reference.len().min(target.len())));
        }

        match self.method {
            RegistrationMethod::FeatureBased => {
                feature_based::register_feature_based(reference, target, width, height, transform_type)
            }
            RegistrationMethod::PhaseCorrelation => {
                phase_correlation::register_phase_correlation(reference, target, width, height)
            }
            RegistrationMethod::OpticalFlow => {
                optical_flow::register_optical_flow(reference, target, width, height, transform_type)
            }
            RegistrationMethod::Ecc => {
                ecc::register_ecc(
                    reference,
                    target,
                    width,
                    height,
                    transform_type,
                    self.max_iterations,
                    self.convergence_threshold,
                )
            }
            RegistrationMethod::Hybrid => {
                self.register_hybrid(reference, target, width, height, transform_type)
            }
        }
    }

    /// Hybrid registration combining multiple methods.
    fn register_hybrid(
        &self,
        reference: &[u8],
        target: &[u8],
        width: u32,
        height: u32,
        transform_type: TransformationType,
    ) -> CvResult<(TransformMatrix, RegistrationQuality)> {
        // Try phase correlation first for initial alignment
        let (init_transform, phase_quality) =
            phase_correlation::register_phase_correlation(reference, target, width, height)?;

        // If phase correlation gives good results, refine with ECC
        if phase_quality.confidence > 0.5 {
            let (final_transform, ecc_quality) = ecc::register_ecc(
                reference,
                target,
                width,
                height,
                transform_type,
                self.max_iterations,
                self.convergence_threshold,
            )?;

            // Compose transformations
            let composed = init_transform.compose(&final_transform);
            Ok((composed, ecc_quality))
        } else {
            // Fall back to feature-based if phase correlation fails
            feature_based::register_feature_based(reference, target, width, height, transform_type)
        }
    }
}

impl Default for VideoRegistration {
    fn default() -> Self {
        Self::new(RegistrationMethod::Ecc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_matrix_identity() {
        let t = TransformMatrix::identity();
        let (x, y) = t.transform_point(10.0, 20.0);
        assert!((x - 10.0).abs() < 1e-6);
        assert!((y - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_transform_matrix_translation() {
        let t = TransformMatrix::translation(5.0, 10.0);
        let (x, y) = t.transform_point(0.0, 0.0);
        assert!((x - 5.0).abs() < 1e-6);
        assert!((y - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_transform_matrix_scale() {
        let t = TransformMatrix::scale(2.0, 3.0);
        let (x, y) = t.transform_point(10.0, 20.0);
        assert!((x - 20.0).abs() < 1e-6);
        assert!((y - 60.0).abs() < 1e-6);
    }

    #[test]
    fn test_transform_matrix_compose() {
        let t1 = TransformMatrix::translation(10.0, 0.0);
        let t2 = TransformMatrix::translation(0.0, 20.0);
        let composed = t1.compose(&t2);
        let (x, y) = composed.transform_point(0.0, 0.0);
        assert!((x - 10.0).abs() < 1e-6);
        assert!((y - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_transform_matrix_inverse() {
        let t = TransformMatrix::translation(10.0, 20.0);
        let inv = t.inverse().expect("inverse should succeed");
        let identity = t.compose(&inv);
        let (x, y) = identity.transform_point(5.0, 15.0);
        assert!((x - 5.0).abs() < 1e-6);
        assert!((y - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_registration_quality() {
        let q = RegistrationQuality::new();
        assert!(!q.is_good());
        assert!(!q.success);
    }

    #[test]
    fn test_video_registration_new() {
        let reg = VideoRegistration::new(RegistrationMethod::PhaseCorrelation);
        assert_eq!(reg.method, RegistrationMethod::PhaseCorrelation);
        assert_eq!(reg.max_iterations, 100);
    }

    #[test]
    fn test_video_registration_with_params() {
        let reg = VideoRegistration::new(RegistrationMethod::Ecc)
            .with_max_iterations(50)
            .with_convergence_threshold(1e-5);

        assert_eq!(reg.max_iterations, 50);
        assert!((reg.convergence_threshold - 1e-5).abs() < 1e-10);
    }
}
