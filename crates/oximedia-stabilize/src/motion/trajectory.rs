//! Motion trajectory representation and manipulation.
//!
//! Trajectories represent the accumulated camera motion over time, built from
//! inter-frame motion models.

use crate::error::{StabilizeError, StabilizeResult};
use crate::motion::model::MotionModel;
use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};

/// A trajectory represents accumulated camera motion over a sequence of frames.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    /// Translation in X over time
    pub x: Array1<f64>,
    /// Translation in Y over time
    pub y: Array1<f64>,
    /// Rotation angle over time (radians)
    pub angle: Array1<f64>,
    /// Scale factor over time
    pub scale: Array1<f64>,
    /// Frame count
    pub frame_count: usize,
}

impl Trajectory {
    /// Create a new trajectory with the given frame count.
    #[must_use]
    pub fn new(frame_count: usize) -> Self {
        Self {
            x: Array1::zeros(frame_count),
            y: Array1::zeros(frame_count),
            angle: Array1::zeros(frame_count),
            scale: Array1::ones(frame_count),
            frame_count,
        }
    }

    /// Build a trajectory from a sequence of motion models.
    ///
    /// The trajectory represents cumulative transformations from frame 0.
    ///
    /// # Errors
    ///
    /// Returns an error if the models vector is empty.
    pub fn from_models(models: &[Box<dyn MotionModel>]) -> StabilizeResult<Self> {
        if models.is_empty() {
            return Err(StabilizeError::EmptyFrameSequence);
        }

        let frame_count = models.len();
        let mut trajectory = Self::new(frame_count);

        // First frame is at origin
        trajectory.x[0] = 0.0;
        trajectory.y[0] = 0.0;
        trajectory.angle[0] = 0.0;
        trajectory.scale[0] = 1.0;

        // Accumulate transformations
        for i in 1..frame_count {
            let params = models[i].parameters();

            // Extract motion parameters (works for translation, affine, etc.)
            let dx = if params.is_empty() { 0.0 } else { params[0] };
            let dy = if params.len() < 2 { 0.0 } else { params[1] };
            let da = if params.len() < 3 { 0.0 } else { params[2] };
            let ds = if params.len() < 4 { 1.0 } else { params[3] };

            // Cumulative sum
            trajectory.x[i] = trajectory.x[i - 1] + dx;
            trajectory.y[i] = trajectory.y[i - 1] + dy;
            trajectory.angle[i] = trajectory.angle[i - 1] + da;
            trajectory.scale[i] = trajectory.scale[i - 1] * ds;
        }

        Ok(trajectory)
    }

    /// Get trajectory length (number of frames).
    #[must_use]
    pub const fn len(&self) -> usize {
        self.frame_count
    }

    /// Check if trajectory is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.frame_count == 0
    }

    /// Get motion parameters at a specific frame.
    #[must_use]
    pub fn at(&self, frame: usize) -> Option<TrajectoryPoint> {
        if frame >= self.frame_count {
            return None;
        }

        Some(TrajectoryPoint {
            x: self.x[frame],
            y: self.y[frame],
            angle: self.angle[frame],
            scale: self.scale[frame],
        })
    }

    /// Calculate motion magnitude at each frame.
    #[must_use]
    pub fn motion_magnitudes(&self) -> Array1<f64> {
        let mut magnitudes = Array1::zeros(self.frame_count);

        for i in 1..self.frame_count {
            let dx = self.x[i] - self.x[i - 1];
            let dy = self.y[i] - self.y[i - 1];
            let da = self.angle[i] - self.angle[i - 1];
            let ds = (self.scale[i] - self.scale[i - 1]).abs();

            magnitudes[i] = (dx * dx + dy * dy).sqrt() + da.abs() * 10.0 + ds * 10.0;
        }

        magnitudes
    }

    /// Calculate average motion magnitude.
    #[must_use]
    pub fn avg_motion_magnitude(&self) -> f64 {
        if self.frame_count <= 1 {
            return 0.0;
        }

        let magnitudes = self.motion_magnitudes();
        magnitudes.sum() / (self.frame_count - 1) as f64
    }

    /// Calculate maximum motion magnitude.
    #[must_use]
    pub fn max_motion_magnitude(&self) -> f64 {
        let magnitudes = self.motion_magnitudes();
        magnitudes.iter().fold(0.0, |a, &b| a.max(b))
    }

    /// Subtract another trajectory (for computing stabilization difference).
    ///
    /// # Errors
    ///
    /// Returns an error if trajectories have different lengths.
    pub fn subtract(&self, other: &Self) -> StabilizeResult<Self> {
        if self.frame_count != other.frame_count {
            return Err(StabilizeError::dimension_mismatch(
                format!("{}", self.frame_count),
                format!("{}", other.frame_count),
            ));
        }

        Ok(Self {
            x: &self.x - &other.x,
            y: &self.y - &other.y,
            angle: &self.angle - &other.angle,
            scale: &self.scale / &other.scale,
            frame_count: self.frame_count,
        })
    }

    /// Add another trajectory.
    ///
    /// # Errors
    ///
    /// Returns an error if trajectories have different lengths.
    pub fn add(&self, other: &Self) -> StabilizeResult<Self> {
        if self.frame_count != other.frame_count {
            return Err(StabilizeError::dimension_mismatch(
                format!("{}", self.frame_count),
                format!("{}", other.frame_count),
            ));
        }

        Ok(Self {
            x: &self.x + &other.x,
            y: &self.y + &other.y,
            angle: &self.angle + &other.angle,
            scale: &self.scale * &other.scale,
            frame_count: self.frame_count,
        })
    }

    /// Apply a scale factor to the trajectory.
    #[must_use]
    pub fn scale_by(&self, factor: f64) -> Self {
        Self {
            x: &self.x * factor,
            y: &self.y * factor,
            angle: &self.angle * factor,
            scale: self.scale.mapv(|s| 1.0 + (s - 1.0) * factor),
            frame_count: self.frame_count,
        }
    }

    /// Get statistics about the trajectory.
    #[must_use]
    pub fn statistics(&self) -> TrajectoryStatistics {
        TrajectoryStatistics {
            avg_motion: self.avg_motion_magnitude(),
            max_motion: self.max_motion_magnitude(),
            total_displacement_x: self.x[self.frame_count - 1] - self.x[0],
            total_displacement_y: self.y[self.frame_count - 1] - self.y[0],
            total_rotation: self.angle[self.frame_count - 1] - self.angle[0],
            avg_scale: self.scale.mean().unwrap_or(1.0),
            frame_count: self.frame_count,
        }
    }
}

/// A single point on a trajectory.
#[derive(Debug, Clone, Copy)]
pub struct TrajectoryPoint {
    /// X position
    pub x: f64,
    /// Y position
    pub y: f64,
    /// Rotation angle
    pub angle: f64,
    /// Scale factor
    pub scale: f64,
}

impl TrajectoryPoint {
    /// Get magnitude of this trajectory point.
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt() + self.angle.abs() * 10.0
    }
}

/// Statistics about a trajectory.
#[derive(Debug, Clone, Copy)]
pub struct TrajectoryStatistics {
    /// Average motion magnitude per frame
    pub avg_motion: f64,
    /// Maximum motion magnitude
    pub max_motion: f64,
    /// Total displacement in X
    pub total_displacement_x: f64,
    /// Total displacement in Y
    pub total_displacement_y: f64,
    /// Total rotation angle
    pub total_rotation: f64,
    /// Average scale factor
    pub avg_scale: f64,
    /// Number of frames
    pub frame_count: usize,
}

impl TrajectoryStatistics {
    /// Get total displacement magnitude.
    #[must_use]
    pub fn total_displacement(&self) -> f64 {
        (self.total_displacement_x.powi(2) + self.total_displacement_y.powi(2)).sqrt()
    }

    /// Check if motion is significant.
    #[must_use]
    pub fn has_significant_motion(&self, threshold: f64) -> bool {
        self.avg_motion > threshold
    }
}

/// Trajectory builder for incremental construction.
pub struct TrajectoryBuilder {
    x: Vec<f64>,
    y: Vec<f64>,
    angle: Vec<f64>,
    scale: Vec<f64>,
}

impl TrajectoryBuilder {
    /// Create a new trajectory builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            x: vec![0.0],
            y: vec![0.0],
            angle: vec![0.0],
            scale: vec![1.0],
        }
    }

    /// Add a relative motion step.
    pub fn add_step(&mut self, dx: f64, dy: f64, da: f64, ds: f64) {
        let last_idx = self.x.len() - 1;
        self.x.push(self.x[last_idx] + dx);
        self.y.push(self.y[last_idx] + dy);
        self.angle.push(self.angle[last_idx] + da);
        self.scale.push(self.scale[last_idx] * ds);
    }

    /// Build the trajectory.
    #[must_use]
    pub fn build(self) -> Trajectory {
        let frame_count = self.x.len();
        Trajectory {
            x: Array1::from_vec(self.x),
            y: Array1::from_vec(self.y),
            angle: Array1::from_vec(self.angle),
            scale: Array1::from_vec(self.scale),
            frame_count,
        }
    }
}

impl Default for TrajectoryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motion::model::TranslationModel;

    #[test]
    fn test_trajectory_creation() {
        let traj = Trajectory::new(10);
        assert_eq!(traj.len(), 10);
        assert!(!traj.is_empty());
    }

    #[test]
    fn test_trajectory_from_models() {
        let models: Vec<Box<dyn MotionModel>> = vec![
            Box::new(TranslationModel::identity()),
            Box::new(TranslationModel::new(10.0, 0.0)),
            Box::new(TranslationModel::new(5.0, 5.0)),
        ];

        let traj = Trajectory::from_models(&models).expect("should succeed in test");
        assert_eq!(traj.len(), 3);
        assert!((traj.x[0] - 0.0).abs() < f64::EPSILON);
        assert!((traj.x[1] - 10.0).abs() < f64::EPSILON);
        assert!((traj.x[2] - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trajectory_at() {
        let traj = Trajectory::new(5);
        let point = traj.at(0).expect("should succeed in test");
        assert!((point.x - 0.0).abs() < f64::EPSILON);
        assert!(traj.at(10).is_none());
    }

    #[test]
    fn test_trajectory_subtract() {
        let traj1 = Trajectory::new(5);
        let traj2 = Trajectory::new(5);
        let diff = traj1.subtract(&traj2).expect("should succeed in test");
        assert_eq!(diff.len(), 5);
    }

    #[test]
    fn test_trajectory_builder() {
        let mut builder = TrajectoryBuilder::new();
        builder.add_step(10.0, 20.0, 0.1, 1.0);
        builder.add_step(5.0, 10.0, 0.05, 1.0);

        let traj = builder.build();
        assert_eq!(traj.len(), 3);
        assert!((traj.x[1] - 10.0).abs() < f64::EPSILON);
        assert!((traj.x[2] - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trajectory_statistics() {
        let traj = Trajectory::new(10);
        let stats = traj.statistics();
        assert_eq!(stats.frame_count, 10);
    }
}
