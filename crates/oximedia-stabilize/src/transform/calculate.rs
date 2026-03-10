//! Transform calculation from trajectories.
//!
//! Computes the stabilization transforms needed to correct camera motion
//! based on the difference between original and smoothed trajectories.

use crate::error::{StabilizeError, StabilizeResult};
use crate::motion::model::Matrix3x3;
use crate::motion::trajectory::Trajectory;

/// Transform calculator that computes stabilization transforms.
#[derive(Debug)]
pub struct TransformCalculator {
    /// Minimum scale to avoid excessive zooming
    min_scale: f64,
    /// Maximum scale to avoid excessive cropping
    max_scale: f64,
}

impl TransformCalculator {
    /// Create a new transform calculator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_scale: 0.8,
            max_scale: 1.5,
        }
    }

    /// Set scale limits.
    pub fn set_scale_limits(&mut self, min: f64, max: f64) {
        self.min_scale = min.max(0.1);
        self.max_scale = max.max(min);
    }

    /// Calculate stabilization transforms.
    ///
    /// Computes the transforms needed to correct camera motion by comparing
    /// original and smoothed trajectories.
    ///
    /// # Errors
    ///
    /// Returns an error if trajectories have different lengths or are empty.
    pub fn calculate(
        &self,
        original: &Trajectory,
        smoothed: &Trajectory,
    ) -> StabilizeResult<Vec<StabilizationTransform>> {
        if original.is_empty() || smoothed.is_empty() {
            return Err(StabilizeError::EmptyFrameSequence);
        }

        if original.len() != smoothed.len() {
            return Err(StabilizeError::dimension_mismatch(
                format!("{}", original.len()),
                format!("{}", smoothed.len()),
            ));
        }

        let mut transforms = Vec::with_capacity(original.len());

        for i in 0..original.len() {
            let transform = self.calculate_frame_transform(original, smoothed, i)?;
            transforms.push(transform);
        }

        Ok(transforms)
    }

    /// Calculate transform for a single frame.
    fn calculate_frame_transform(
        &self,
        original: &Trajectory,
        smoothed: &Trajectory,
        frame: usize,
    ) -> StabilizeResult<StabilizationTransform> {
        let orig_point = original
            .at(frame)
            .ok_or_else(|| StabilizeError::invalid_parameter("frame", frame.to_string()))?;
        let smooth_point = smoothed
            .at(frame)
            .ok_or_else(|| StabilizeError::invalid_parameter("frame", frame.to_string()))?;

        // Compute difference transformation
        let dx = smooth_point.x - orig_point.x;
        let dy = smooth_point.y - orig_point.y;
        let da = smooth_point.angle - orig_point.angle;
        let ds = smooth_point.scale / orig_point.scale;

        // Clamp scale
        let ds = ds.clamp(self.min_scale, self.max_scale);

        Ok(StabilizationTransform {
            dx,
            dy,
            angle: da,
            scale: ds,
            frame_index: frame,
            confidence: 1.0,
        })
    }

    /// Calculate cumulative transforms (accumulate over time).
    ///
    /// # Errors
    ///
    /// Returns an error if the transforms vector is empty.
    pub fn cumulative_transforms(
        &self,
        transforms: &[StabilizationTransform],
    ) -> StabilizeResult<Vec<StabilizationTransform>> {
        if transforms.is_empty() {
            return Err(StabilizeError::EmptyFrameSequence);
        }

        let mut cumulative = Vec::with_capacity(transforms.len());
        let mut cum_dx = 0.0;
        let mut cum_dy = 0.0;
        let mut cum_angle = 0.0;
        let mut cum_scale = 1.0;

        for transform in transforms {
            cum_dx += transform.dx;
            cum_dy += transform.dy;
            cum_angle += transform.angle;
            cum_scale *= transform.scale;

            cumulative.push(StabilizationTransform {
                dx: cum_dx,
                dy: cum_dy,
                angle: cum_angle,
                scale: cum_scale,
                frame_index: transform.frame_index,
                confidence: transform.confidence,
            });
        }

        Ok(cumulative)
    }

    /// Convert to affine matrices.
    #[must_use]
    pub fn to_affine_matrices(&self, transforms: &[StabilizationTransform]) -> Vec<Matrix3x3> {
        transforms
            .iter()
            .map(|t| {
                let cos_a = t.angle.cos();
                let sin_a = t.angle.sin();
                let s = t.scale;

                Matrix3x3::new(
                    s * cos_a,
                    -s * sin_a,
                    t.dx,
                    s * sin_a,
                    s * cos_a,
                    t.dy,
                    0.0,
                    0.0,
                    1.0,
                )
            })
            .collect()
    }

    /// Invert transforms.
    #[must_use]
    pub fn invert_transforms(
        &self,
        transforms: &[StabilizationTransform],
    ) -> Vec<StabilizationTransform> {
        transforms
            .iter()
            .map(|t| {
                // For small angles, use approximation
                // For general case, compute full inverse
                let matrix = self.transform_to_matrix(t);
                if let Some(inv) = matrix.try_inverse() {
                    self.matrix_to_transform(&inv, t.frame_index)
                } else {
                    // Fallback: simple negation
                    StabilizationTransform {
                        dx: -t.dx,
                        dy: -t.dy,
                        angle: -t.angle,
                        scale: 1.0 / t.scale,
                        frame_index: t.frame_index,
                        confidence: t.confidence,
                    }
                }
            })
            .collect()
    }

    /// Convert transform to matrix.
    fn transform_to_matrix(&self, transform: &StabilizationTransform) -> Matrix3x3 {
        let cos_a = transform.angle.cos();
        let sin_a = transform.angle.sin();
        let s = transform.scale;

        Matrix3x3::new(
            s * cos_a,
            -s * sin_a,
            transform.dx,
            s * sin_a,
            s * cos_a,
            transform.dy,
            0.0,
            0.0,
            1.0,
        )
    }

    /// Convert matrix to transform.
    fn matrix_to_transform(
        &self,
        matrix: &Matrix3x3,
        frame_index: usize,
    ) -> StabilizationTransform {
        let dx = matrix.get(0, 2);
        let dy = matrix.get(1, 2);

        let a = matrix.get(0, 0);
        let b = matrix.get(0, 1);

        let scale = (a * a + b * b).sqrt();
        let angle = b.atan2(a);

        StabilizationTransform {
            dx,
            dy,
            angle,
            scale,
            frame_index,
            confidence: 1.0,
        }
    }
}

impl Default for TransformCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// A stabilization transform for a single frame.
#[derive(Debug, Clone, Copy)]
pub struct StabilizationTransform {
    /// Translation in X
    pub dx: f64,
    /// Translation in Y
    pub dy: f64,
    /// Rotation angle in radians
    pub angle: f64,
    /// Scale factor
    pub scale: f64,
    /// Frame index
    pub frame_index: usize,
    /// Confidence in this transform (0-1)
    pub confidence: f64,
}

impl StabilizationTransform {
    /// Create a new stabilization transform.
    #[must_use]
    pub const fn new(dx: f64, dy: f64, angle: f64, scale: f64, frame_index: usize) -> Self {
        Self {
            dx,
            dy,
            angle,
            scale,
            frame_index,
            confidence: 1.0,
        }
    }

    /// Create an identity transform.
    #[must_use]
    pub const fn identity(frame_index: usize) -> Self {
        Self {
            dx: 0.0,
            dy: 0.0,
            angle: 0.0,
            scale: 1.0,
            frame_index,
            confidence: 1.0,
        }
    }

    /// Get transform magnitude.
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        let trans = (self.dx * self.dx + self.dy * self.dy).sqrt();
        let rot = self.angle.abs();
        let scale_dev = (self.scale - 1.0).abs();

        trans + rot * 10.0 + scale_dev * 10.0
    }

    /// Transform a point.
    #[must_use]
    pub fn transform_point(&self, x: f64, y: f64) -> (f64, f64) {
        let cos_a = self.angle.cos();
        let sin_a = self.angle.sin();
        let s = self.scale;

        let tx = s * (cos_a * x - sin_a * y) + self.dx;
        let ty = s * (sin_a * x + cos_a * y) + self.dy;

        (tx, ty)
    }

    /// Compose with another transform.
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        // T2 * T1 = combined transform
        let _cos_a1 = self.angle.cos();
        let _sin_a1 = self.angle.sin();
        let cos_a2 = other.angle.cos();
        let sin_a2 = other.angle.sin();

        let s1 = self.scale;
        let s2 = other.scale;

        // Combined rotation
        let angle = self.angle + other.angle;

        // Combined scale
        let scale = s1 * s2;

        // Combined translation
        let dx = s2 * (cos_a2 * self.dx - sin_a2 * self.dy) + other.dx;
        let dy = s2 * (sin_a2 * self.dx + cos_a2 * self.dy) + other.dy;

        Self {
            dx,
            dy,
            angle,
            scale,
            frame_index: other.frame_index,
            confidence: self.confidence * other.confidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculator_creation() {
        let calc = TransformCalculator::new();
        assert!((calc.min_scale - 0.8).abs() < f64::EPSILON);
        assert!((calc.max_scale - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_transform_identity() {
        let transform = StabilizationTransform::identity(0);
        let (x, y) = transform.transform_point(10.0, 20.0);
        assert!((x - 10.0).abs() < 1e-10);
        assert!((y - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_transform_translation() {
        let transform = StabilizationTransform::new(5.0, 10.0, 0.0, 1.0, 0);
        let (x, y) = transform.transform_point(0.0, 0.0);
        assert!((x - 5.0).abs() < 1e-10);
        assert!((y - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_transform_magnitude() {
        let transform = StabilizationTransform::new(3.0, 4.0, 0.0, 1.0, 0);
        let mag = transform.magnitude();
        assert!((mag - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_transform_composition() {
        let t1 = StabilizationTransform::new(10.0, 0.0, 0.0, 1.0, 0);
        let t2 = StabilizationTransform::new(5.0, 0.0, 0.0, 1.0, 1);
        let composed = t1.compose(&t2);
        assert!((composed.dx - 15.0).abs() < 1e-10);
    }
}
