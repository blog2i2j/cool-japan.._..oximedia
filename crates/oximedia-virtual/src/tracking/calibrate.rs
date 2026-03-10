//! Camera calibration for tracking systems
//!
//! Provides calibration workflows for camera tracking systems including
//! coordinate system alignment and reference point mapping.

use super::CameraPose;
use crate::math::{Matrix3, Matrix4, Point3, UnitQuaternion};
use crate::{Result, VirtualProductionError};
use serde::{Deserialize, Serialize};

/// Calibration method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationMethod {
    /// Single point origin
    SinglePoint,
    /// Three point alignment (origin, X-axis, XY-plane)
    ThreePoint,
    /// Multi-point least squares
    MultiPoint,
}

/// Calibration point
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CalibrationPoint {
    /// Physical world position
    pub world_position: Point3<f64>,
    /// Measured tracker position
    pub tracker_position: Point3<f64>,
    /// Point identifier
    pub id: usize,
}

impl CalibrationPoint {
    /// Create new calibration point
    #[must_use]
    pub fn new(world_position: Point3<f64>, tracker_position: Point3<f64>, id: usize) -> Self {
        Self {
            world_position,
            tracker_position,
            id,
        }
    }
}

/// Calibration result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResult {
    /// Transformation matrix from tracker to world coordinates
    pub transform: Matrix4<f64>,
    /// Calibration error (RMS in meters)
    pub error_rms: f64,
    /// Number of points used
    pub num_points: usize,
    /// Calibration method used
    pub method: CalibrationMethod,
}

impl CalibrationResult {
    /// Apply calibration to a pose
    #[must_use]
    pub fn apply(&self, pose: &CameraPose) -> CameraPose {
        // Transform position
        let pos_homogeneous = self
            .transform
            .mul_homogeneous(&pose.position.to_homogeneous());
        let position = Point3::from_homogeneous(pos_homogeneous).unwrap_or(pose.position);

        // Extract rotation from transform matrix and compose with existing orientation
        let rotation_3x3 = self.transform.fixed_view_3x3(0, 0);
        let cal_rotation = UnitQuaternion::from_matrix(&rotation_3x3);
        let orientation = cal_rotation * pose.orientation;

        CameraPose {
            position,
            orientation,
            timestamp_ns: pose.timestamp_ns,
            confidence: pose.confidence,
        }
    }
}

/// Camera tracking calibrator
pub struct TrackingCalibrator {
    points: Vec<CalibrationPoint>,
    method: CalibrationMethod,
}

impl TrackingCalibrator {
    /// Create new calibrator
    #[must_use]
    pub fn new(method: CalibrationMethod) -> Self {
        Self {
            points: Vec::new(),
            method,
        }
    }

    /// Add calibration point
    pub fn add_point(&mut self, point: CalibrationPoint) {
        self.points.push(point);
    }

    /// Clear all calibration points
    pub fn clear(&mut self) {
        self.points.clear();
    }

    /// Compute calibration
    pub fn calibrate(&self) -> Result<CalibrationResult> {
        match self.method {
            CalibrationMethod::SinglePoint => self.calibrate_single_point(),
            CalibrationMethod::ThreePoint => self.calibrate_three_point(),
            CalibrationMethod::MultiPoint => self.calibrate_multi_point(),
        }
    }

    /// Single point calibration (translation only)
    fn calibrate_single_point(&self) -> Result<CalibrationResult> {
        if self.points.is_empty() {
            return Err(VirtualProductionError::Calibration(
                "No calibration points provided".to_string(),
            ));
        }

        let point = &self.points[0];
        let translation = point.world_position - point.tracker_position;

        let mut transform = Matrix4::identity();
        transform.set_block_3x1(0, 3, &translation);

        let error = self.compute_error(&transform);

        Ok(CalibrationResult {
            transform,
            error_rms: error,
            num_points: self.points.len(),
            method: CalibrationMethod::SinglePoint,
        })
    }

    /// Three point calibration (full rigid transform)
    fn calibrate_three_point(&self) -> Result<CalibrationResult> {
        if self.points.len() < 3 {
            return Err(VirtualProductionError::Calibration(format!(
                "Three point calibration requires 3 points, got {}",
                self.points.len()
            )));
        }

        // Use first three points to establish coordinate system
        let p0 = &self.points[0];
        let p1 = &self.points[1];
        let p2 = &self.points[2];

        // Compute world coordinate axes
        let world_x = (p1.world_position - p0.world_position).normalize();
        let world_xy = p2.world_position - p0.world_position;
        let world_z = world_x.cross(&world_xy).normalize();
        let world_y = world_z.cross(&world_x);

        // Compute tracker coordinate axes
        let tracker_x = (p1.tracker_position - p0.tracker_position).normalize();
        let tracker_xy = p2.tracker_position - p0.tracker_position;
        let tracker_z = tracker_x.cross(&tracker_xy).normalize();
        let tracker_y = tracker_z.cross(&tracker_x);

        // Build rotation matrix
        let mut world_mat = Matrix4::identity();
        world_mat.set_block_3x1(0, 0, &world_x);
        world_mat.set_block_3x1(0, 1, &world_y);
        world_mat.set_block_3x1(0, 2, &world_z);

        let mut tracker_mat = Matrix4::identity();
        tracker_mat.set_block_3x1(0, 0, &tracker_x);
        tracker_mat.set_block_3x1(0, 1, &tracker_y);
        tracker_mat.set_block_3x1(0, 2, &tracker_z);

        // Compute transform
        let rotation = world_mat * tracker_mat.try_inverse().unwrap_or(Matrix4::identity());
        let mut transform = rotation;

        // Add translation
        let rot_3x3 = rotation.fixed_view_3x3(0, 0);
        let translation =
            p0.world_position.coords() - rot_3x3.mul_vec(&p0.tracker_position.coords());
        transform.set_block_3x1(0, 3, &translation);

        let error = self.compute_error(&transform);

        Ok(CalibrationResult {
            transform,
            error_rms: error,
            num_points: self.points.len(),
            method: CalibrationMethod::ThreePoint,
        })
    }

    /// Multi-point calibration using least squares
    fn calibrate_multi_point(&self) -> Result<CalibrationResult> {
        if self.points.len() < 4 {
            return Err(VirtualProductionError::Calibration(format!(
                "Multi-point calibration requires at least 4 points, got {}",
                self.points.len()
            )));
        }

        // Compute centroids
        let mut world_centroid = Point3::origin();
        let mut tracker_centroid = Point3::origin();

        for point in &self.points {
            world_centroid += point.world_position.coords();
            tracker_centroid += point.tracker_position.coords();
        }

        let n = self.points.len() as f64;
        world_centroid = Point3::from(world_centroid.coords() / n);
        tracker_centroid = Point3::from(tracker_centroid.coords() / n);

        // Build correlation matrix
        let mut h = Matrix3::zeros();

        for point in &self.points {
            let world_centered = point.world_position - world_centroid;
            let tracker_centered = point.tracker_position - tracker_centroid;
            h += tracker_centered * world_centered.transpose();
        }

        // SVD to find rotation
        let svd = h.svd(true, true);
        let u = svd.u.unwrap_or(Matrix3::identity());
        let v_t = svd.v_t.unwrap_or(Matrix3::identity());
        let mut rotation_3x3 = v_t.transpose() * u.transpose();

        // Ensure proper rotation (det = 1)
        if rotation_3x3.determinant() < 0.0 {
            let mut v = v_t.transpose();
            // Negate third column of V
            let mut col2 = v.column(2);
            col2.scale_mut(-1.0);
            v.data[0][2] = col2.x;
            v.data[1][2] = col2.y;
            v.data[2][2] = col2.z;
            rotation_3x3 = v * u.transpose();
        }

        // Build 4x4 transform
        let mut transform = Matrix4::identity();
        transform.set_block_3x3(0, 0, &rotation_3x3);

        // Compute translation
        let translation =
            world_centroid.coords() - rotation_3x3.mul_vec(&tracker_centroid.coords());
        transform.set_block_3x1(0, 3, &translation);

        let error = self.compute_error(&transform);

        Ok(CalibrationResult {
            transform,
            error_rms: error,
            num_points: self.points.len(),
            method: CalibrationMethod::MultiPoint,
        })
    }

    /// Compute RMS error for a transform
    fn compute_error(&self, transform: &Matrix4<f64>) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }

        let mut sum_squared_error = 0.0;

        for point in &self.points {
            let transformed = transform.mul_homogeneous(&point.tracker_position.to_homogeneous());
            let transformed_point =
                Point3::from_homogeneous(transformed).unwrap_or(point.tracker_position);
            let error = (transformed_point - point.world_position).norm();
            sum_squared_error += error * error;
        }

        (sum_squared_error / self.points.len() as f64).sqrt()
    }

    /// Get number of calibration points
    #[must_use]
    pub fn num_points(&self) -> usize {
        self.points.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_point() {
        let point =
            CalibrationPoint::new(Point3::new(1.0, 2.0, 3.0), Point3::new(0.0, 0.0, 0.0), 0);
        assert_eq!(point.id, 0);
    }

    #[test]
    fn test_calibrator_creation() {
        let cal = TrackingCalibrator::new(CalibrationMethod::SinglePoint);
        assert_eq!(cal.num_points(), 0);
    }

    #[test]
    fn test_calibrator_add_point() {
        let mut cal = TrackingCalibrator::new(CalibrationMethod::SinglePoint);
        cal.add_point(CalibrationPoint::new(Point3::origin(), Point3::origin(), 0));
        assert_eq!(cal.num_points(), 1);
    }

    #[test]
    fn test_single_point_calibration() {
        let mut cal = TrackingCalibrator::new(CalibrationMethod::SinglePoint);
        cal.add_point(CalibrationPoint::new(
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 0.0),
            0,
        ));

        let result = cal.calibrate();
        assert!(result.is_ok());
        let result = result.expect("should succeed in test");
        assert_eq!(result.num_points, 1);
        assert!(result.error_rms < 0.1);
    }

    #[test]
    fn test_calibration_error() {
        let mut cal = TrackingCalibrator::new(CalibrationMethod::MultiPoint);

        // Add points with perfect alignment
        for i in 0..5 {
            let pos = Point3::new(i as f64, i as f64, i as f64);
            cal.add_point(CalibrationPoint::new(pos, pos, i));
        }

        let transform = Matrix4::identity();
        let error = cal.compute_error(&transform);
        assert!(error < 1e-10);
    }
}
