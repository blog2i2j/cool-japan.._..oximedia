//! Camera tracking subsystem for virtual production
//!
//! Provides real-time camera position and orientation tracking with
//! sub-millimeter accuracy and sub-degree rotational precision.

pub mod calibrate;
pub mod camera;
pub mod filter;
pub mod imu;
pub mod markers;

use crate::math::{Point3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};

/// Camera pose (position and orientation)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CameraPose {
    /// Position in world space (meters)
    pub position: Point3<f64>,
    /// Orientation as unit quaternion
    pub orientation: UnitQuaternion<f64>,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Tracking confidence (0.0 - 1.0)
    pub confidence: f32,
}

impl CameraPose {
    /// Create new camera pose
    #[must_use]
    pub fn new(position: Point3<f64>, orientation: UnitQuaternion<f64>, timestamp_ns: u64) -> Self {
        Self {
            position,
            orientation,
            timestamp_ns,
            confidence: 1.0,
        }
    }

    /// Get forward vector
    #[must_use]
    pub fn forward(&self) -> Vector3<f64> {
        self.orientation * Vector3::new(0.0, 0.0, -1.0)
    }

    /// Get up vector
    #[must_use]
    pub fn up(&self) -> Vector3<f64> {
        self.orientation * Vector3::new(0.0, 1.0, 0.0)
    }

    /// Get right vector
    #[must_use]
    pub fn right(&self) -> Vector3<f64> {
        self.orientation * Vector3::new(1.0, 0.0, 0.0)
    }

    /// Interpolate between two poses
    #[must_use]
    pub fn interpolate(&self, other: &Self, t: f64) -> Self {
        let position = self.position + (other.position - self.position) * t;
        let orientation = self.orientation.slerp(&other.orientation, t);
        let timestamp_ns =
            self.timestamp_ns + ((other.timestamp_ns - self.timestamp_ns) as f64 * t) as u64;
        let confidence = self.confidence + (other.confidence - self.confidence) * t as f32;

        Self {
            position,
            orientation,
            timestamp_ns,
            confidence,
        }
    }
}

impl Default for CameraPose {
    fn default() -> Self {
        Self {
            position: Point3::origin(),
            orientation: UnitQuaternion::identity(),
            timestamp_ns: 0,
            confidence: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_pose_default() {
        let pose = CameraPose::default();
        assert_eq!(pose.position, Point3::origin());
        assert_eq!(pose.confidence, 1.0);
    }

    #[test]
    fn test_camera_pose_vectors() {
        let pose = CameraPose::default();
        let forward = pose.forward();
        assert!((forward.z + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pose_interpolation() {
        let pose1 = CameraPose::new(Point3::new(0.0, 0.0, 0.0), UnitQuaternion::identity(), 0);
        let pose2 = CameraPose::new(Point3::new(1.0, 1.0, 1.0), UnitQuaternion::identity(), 1000);

        let interp = pose1.interpolate(&pose2, 0.5);
        assert!((interp.position.x - 0.5).abs() < 1e-6);
        assert!((interp.position.y - 0.5).abs() < 1e-6);
        assert!((interp.position.z - 0.5).abs() < 1e-6);
    }
}
