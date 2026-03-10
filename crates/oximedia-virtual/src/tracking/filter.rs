//! Tracking filters and smoothing
//!
//! Provides Kalman filtering and smoothing for camera tracking data.

use super::CameraPose;
use crate::math::{Matrix3, Matrix3x6, Matrix6, Point3, UnitQuaternion, Vector3, Vector6};
use serde::{Deserialize, Serialize};

/// Kalman filter for tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalmanFilter {
    /// State vector (position, velocity)
    state: Vector6<f64>,
    /// Covariance matrix
    covariance: Matrix6<f64>,
    /// Process noise
    process_noise: Matrix6<f64>,
    /// Measurement noise
    measurement_noise: Matrix3<f64>,
}

impl KalmanFilter {
    /// Create new Kalman filter
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Vector6::zeros(),
            covariance: Matrix6::identity(),
            process_noise: Matrix6::identity() * 0.01,
            measurement_noise: Matrix3::identity() * 0.1,
        }
    }

    /// Predict next state
    pub fn predict(&mut self, dt: f64) {
        // State transition matrix
        let mut f = Matrix6::identity();
        f[(0, 3)] = dt;
        f[(1, 4)] = dt;
        f[(2, 5)] = dt;

        // Predict state
        self.state = f * self.state;

        // Predict covariance
        self.covariance = f * self.covariance * f.transpose() + self.process_noise;
    }

    /// Update with measurement
    pub fn update(&mut self, measurement: &Point3<f64>) {
        // Measurement matrix (only observing position)
        let _h = Matrix3::from_iterator([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

        // Innovation
        let h_6x3 = Matrix3x6::from_iterator([
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0,
        ]);

        let predicted_measurement = h_6x3 * self.state;
        let innovation = measurement.coords() - predicted_measurement;

        // Innovation covariance
        let s = h_6x3 * self.covariance * h_6x3.transpose() + self.measurement_noise;

        // Kalman gain
        if let Some(s_inv) = s.try_inverse() {
            let k = self.covariance * h_6x3.transpose() * s_inv;

            // Update state
            self.state += k.mul_vec3(&innovation);

            // Update covariance
            let i_kh = Matrix6::identity() - k.mul_3x6(&h_6x3);
            self.covariance = i_kh * self.covariance;
        }
    }

    /// Get filtered position
    #[must_use]
    pub fn position(&self) -> Point3<f64> {
        Point3::new(self.state[0], self.state[1], self.state[2])
    }

    /// Get filtered velocity
    #[must_use]
    pub fn velocity(&self) -> Vector3<f64> {
        Vector3::new(self.state[3], self.state[4], self.state[5])
    }
}

impl Default for KalmanFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Exponential smoothing filter
#[derive(Debug, Clone)]
pub struct ExponentialSmoothingFilter {
    alpha: f64,
    filtered_position: Option<Point3<f64>>,
    filtered_orientation: Option<UnitQuaternion<f64>>,
}

impl ExponentialSmoothingFilter {
    /// Create new exponential smoothing filter
    #[must_use]
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0),
            filtered_position: None,
            filtered_orientation: None,
        }
    }

    /// Filter pose
    pub fn filter(&mut self, pose: &CameraPose) -> CameraPose {
        // Filter position
        let filtered_pos = if let Some(prev_pos) = self.filtered_position {
            Point3::from(
                prev_pos.coords() * (1.0 - self.alpha) + pose.position.coords() * self.alpha,
            )
        } else {
            pose.position
        };
        self.filtered_position = Some(filtered_pos);

        // Filter orientation using SLERP
        let filtered_orient = if let Some(prev_orient) = self.filtered_orientation {
            prev_orient.slerp(&pose.orientation, self.alpha)
        } else {
            pose.orientation
        };
        self.filtered_orientation = Some(filtered_orient);

        CameraPose {
            position: filtered_pos,
            orientation: filtered_orient,
            timestamp_ns: pose.timestamp_ns,
            confidence: pose.confidence,
        }
    }

    /// Reset filter
    pub fn reset(&mut self) {
        self.filtered_position = None;
        self.filtered_orientation = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kalman_filter() {
        let mut filter = KalmanFilter::new();

        // Predict
        filter.predict(0.016); // 60 FPS

        // Update with measurement
        let measurement = Point3::new(1.0, 2.0, 3.0);
        filter.update(&measurement);

        let position = filter.position();
        assert!(position.x > 0.0);
    }

    #[test]
    fn test_exponential_smoothing() {
        let mut filter = ExponentialSmoothingFilter::new(0.3);

        let pose1 = CameraPose::new(Point3::new(0.0, 0.0, 0.0), UnitQuaternion::identity(), 0);

        let filtered1 = filter.filter(&pose1);
        assert_eq!(filtered1.position, pose1.position);

        let pose2 = CameraPose::new(
            Point3::new(10.0, 10.0, 10.0),
            UnitQuaternion::identity(),
            1000,
        );

        let filtered2 = filter.filter(&pose2);
        assert!(filtered2.position.x > 0.0);
        assert!(filtered2.position.x < 10.0); // Should be smoothed
    }
}
