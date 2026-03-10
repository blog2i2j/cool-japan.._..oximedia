//! Temporal coherence optimization for smooth video output.
//!
//! Ensures that smoothed trajectories maintain temporal consistency and
//! avoid artifacts like jitter or wobble.

use crate::error::{StabilizeError, StabilizeResult};
use crate::motion::trajectory::Trajectory;
use scirs2_core::ndarray::Array1;

/// Temporal coherence optimizer.
pub struct TemporalCoherence {
    /// Maximum allowed velocity change per frame
    max_velocity_change: f64,
    /// Maximum allowed acceleration
    max_acceleration: f64,
    /// Smoothing iterations
    iterations: usize,
}

impl TemporalCoherence {
    /// Create a new temporal coherence optimizer.
    #[must_use]
    pub fn new(max_velocity_change: f64, max_acceleration: f64) -> Self {
        Self {
            max_velocity_change,
            max_acceleration,
            iterations: 5,
        }
    }

    /// Set maximum velocity change.
    pub fn set_max_velocity_change(&mut self, value: f64) {
        self.max_velocity_change = value.max(0.0);
    }

    /// Set maximum acceleration.
    pub fn set_max_acceleration(&mut self, value: f64) {
        self.max_acceleration = value.max(0.0);
    }

    /// Optimize trajectory for temporal coherence.
    ///
    /// # Errors
    ///
    /// Returns an error if the trajectory is empty.
    pub fn optimize(&self, trajectory: &Trajectory) -> StabilizeResult<Trajectory> {
        if trajectory.is_empty() {
            return Err(StabilizeError::EmptyFrameSequence);
        }

        let mut optimized = trajectory.clone();

        for _ in 0..self.iterations {
            optimized = self.enforce_velocity_constraints(&optimized)?;
            optimized = self.enforce_acceleration_constraints(&optimized)?;
        }

        Ok(optimized)
    }

    /// Enforce velocity constraints.
    fn enforce_velocity_constraints(&self, trajectory: &Trajectory) -> StabilizeResult<Trajectory> {
        Ok(Trajectory {
            x: self.limit_velocity(&trajectory.x),
            y: self.limit_velocity(&trajectory.y),
            angle: self.limit_velocity(&trajectory.angle),
            scale: trajectory.scale.clone(), // Scale is multiplicative, handle separately
            frame_count: trajectory.frame_count,
        })
    }

    /// Enforce acceleration constraints.
    fn enforce_acceleration_constraints(
        &self,
        trajectory: &Trajectory,
    ) -> StabilizeResult<Trajectory> {
        Ok(Trajectory {
            x: self.limit_acceleration(&trajectory.x),
            y: self.limit_acceleration(&trajectory.y),
            angle: self.limit_acceleration(&trajectory.angle),
            scale: trajectory.scale.clone(),
            frame_count: trajectory.frame_count,
        })
    }

    /// Limit velocity (first derivative).
    fn limit_velocity(&self, data: &Array1<f64>) -> Array1<f64> {
        let n = data.len();
        if n < 2 {
            return data.clone();
        }

        let mut result = data.clone();

        for i in 1..n {
            let velocity = result[i] - result[i - 1];

            if velocity.abs() > self.max_velocity_change {
                let limited_velocity = velocity.signum() * self.max_velocity_change;
                result[i] = result[i - 1] + limited_velocity;
            }
        }

        result
    }

    /// Limit acceleration (second derivative).
    fn limit_acceleration(&self, data: &Array1<f64>) -> Array1<f64> {
        let n = data.len();
        if n < 3 {
            return data.clone();
        }

        let mut result = data.clone();

        for i in 2..n {
            let v1 = result[i - 1] - result[i - 2];
            let v2 = result[i] - result[i - 1];
            let acceleration = v2 - v1;

            if acceleration.abs() > self.max_acceleration {
                let limited_accel = acceleration.signum() * self.max_acceleration;
                let new_velocity = v1 + limited_accel;
                result[i] = result[i - 1] + new_velocity;
            }
        }

        result
    }

    /// Compute velocity (first derivative).
    #[must_use]
    pub fn compute_velocity(&self, data: &Array1<f64>) -> Array1<f64> {
        let n = data.len();
        let mut velocity = Array1::zeros(n);

        for i in 1..n {
            velocity[i] = data[i] - data[i - 1];
        }

        velocity
    }

    /// Compute acceleration (second derivative).
    #[must_use]
    pub fn compute_acceleration(&self, data: &Array1<f64>) -> Array1<f64> {
        let velocity = self.compute_velocity(data);
        self.compute_velocity(&velocity)
    }

    /// Measure temporal smoothness.
    #[must_use]
    pub fn measure_smoothness(&self, trajectory: &Trajectory) -> SmoothnessMetrics {
        let vel_x = self.compute_velocity(&trajectory.x);
        let vel_y = self.compute_velocity(&trajectory.y);
        let acc_x = self.compute_acceleration(&trajectory.x);
        let acc_y = self.compute_acceleration(&trajectory.y);

        let avg_velocity = (vel_x.mapv(|v| v.abs()).sum() + vel_y.mapv(|v| v.abs()).sum())
            / (2.0 * trajectory.frame_count as f64);
        let avg_acceleration = (acc_x.mapv(|a| a.abs()).sum() + acc_y.mapv(|a| a.abs()).sum())
            / (2.0 * trajectory.frame_count as f64);

        let max_velocity = vel_x
            .iter()
            .chain(vel_y.iter())
            .map(|v| v.abs())
            .fold(0.0, f64::max);
        let max_acceleration = acc_x
            .iter()
            .chain(acc_y.iter())
            .map(|a| a.abs())
            .fold(0.0, f64::max);

        SmoothnessMetrics {
            avg_velocity,
            avg_acceleration,
            max_velocity,
            max_acceleration,
            velocity_variance: self.compute_variance(&vel_x) + self.compute_variance(&vel_y),
            acceleration_variance: self.compute_variance(&acc_x) + self.compute_variance(&acc_y),
        }
    }

    /// Compute variance of a signal.
    fn compute_variance(&self, data: &Array1<f64>) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let mean = data.mean().unwrap_or(0.0);
        let sum_sq_diff: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum();
        sum_sq_diff / data.len() as f64
    }
}

/// Metrics for measuring temporal smoothness.
#[derive(Debug, Clone, Copy)]
pub struct SmoothnessMetrics {
    /// Average velocity magnitude
    pub avg_velocity: f64,
    /// Average acceleration magnitude
    pub avg_acceleration: f64,
    /// Maximum velocity magnitude
    pub max_velocity: f64,
    /// Maximum acceleration magnitude
    pub max_acceleration: f64,
    /// Velocity variance (jitter measure)
    pub velocity_variance: f64,
    /// Acceleration variance (wobble measure)
    pub acceleration_variance: f64,
}

impl SmoothnessMetrics {
    /// Get overall smoothness score (0-1, higher is better).
    #[must_use]
    pub fn smoothness_score(&self) -> f64 {
        // Lower variance = smoother
        let velocity_score = 1.0 / (1.0 + self.velocity_variance);
        let acceleration_score = 1.0 / (1.0 + self.acceleration_variance);

        (velocity_score + acceleration_score) / 2.0
    }

    /// Check if motion is smooth enough.
    #[must_use]
    pub fn is_smooth(&self, threshold: f64) -> bool {
        self.smoothness_score() > threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_coherence_creation() {
        let tc = TemporalCoherence::new(5.0, 2.0);
        assert!((tc.max_velocity_change - 5.0).abs() < f64::EPSILON);
        assert!((tc.max_acceleration - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_velocity_computation() {
        let tc = TemporalCoherence::new(5.0, 2.0);
        let data = Array1::from_vec(vec![0.0, 1.0, 3.0, 6.0]);
        let velocity = tc.compute_velocity(&data);

        assert!((velocity[1] - 1.0).abs() < f64::EPSILON);
        assert!((velocity[2] - 2.0).abs() < f64::EPSILON);
        assert!((velocity[3] - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_acceleration_computation() {
        let tc = TemporalCoherence::new(5.0, 2.0);
        let data = Array1::from_vec(vec![0.0, 1.0, 3.0, 6.0]);
        let acceleration = tc.compute_acceleration(&data);

        assert!((acceleration[2] - 1.0).abs() < f64::EPSILON);
        assert!((acceleration[3] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_velocity_limiting() {
        let tc = TemporalCoherence::new(2.0, 1.0);
        let data = Array1::from_vec(vec![0.0, 5.0, 10.0]);
        let limited = tc.limit_velocity(&data);

        // Velocity should be limited to 2.0 per frame
        assert!((limited[1] - 2.0).abs() < f64::EPSILON);
        assert!((limited[2] - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_smoothness_metrics() {
        let tc = TemporalCoherence::new(5.0, 2.0);
        let data = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let variance = tc.compute_variance(&data);
        assert!(variance >= 0.0);
    }
}
