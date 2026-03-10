//! Adaptive smoothing based on motion characteristics.
//!
//! Adjusts smoothing strength based on the type and magnitude of camera motion.

use crate::error::{StabilizeError, StabilizeResult};
use crate::motion::trajectory::Trajectory;
use crate::smooth::filter::{GaussianFilter, KalmanFilter};
use scirs2_core::ndarray::Array1;

/// Adaptive smoother that adjusts filtering based on motion characteristics.
pub struct AdaptiveSmoother {
    window_size: usize,
    max_motion: f64,
    gaussian_filter: GaussianFilter,
    kalman_filter: KalmanFilter,
    motion_threshold_low: f64,
    motion_threshold_high: f64,
}

impl AdaptiveSmoother {
    /// Create a new adaptive smoother.
    #[must_use]
    pub fn new(window_size: usize, max_motion: f64) -> Self {
        Self {
            window_size,
            max_motion,
            gaussian_filter: GaussianFilter::new(window_size),
            kalman_filter: KalmanFilter::new(),
            motion_threshold_low: max_motion * 0.1,
            motion_threshold_high: max_motion * 0.5,
        }
    }

    /// Set motion thresholds.
    pub fn set_thresholds(&mut self, low: f64, high: f64) {
        self.motion_threshold_low = low;
        self.motion_threshold_high = high;
    }

    /// Smooth a trajectory with adaptive strength.
    ///
    /// # Errors
    ///
    /// Returns an error if the trajectory is empty.
    pub fn smooth(&mut self, trajectory: &Trajectory) -> StabilizeResult<Trajectory> {
        if trajectory.is_empty() {
            return Err(StabilizeError::EmptyFrameSequence);
        }

        // Analyze motion characteristics
        let motion_magnitudes = trajectory.motion_magnitudes();
        let motion_classification = self.classify_motion(&motion_magnitudes);

        // Apply adaptive smoothing
        let smoothed_x =
            self.adaptive_smooth_signal(&trajectory.x, &motion_magnitudes, &motion_classification);
        let smoothed_y =
            self.adaptive_smooth_signal(&trajectory.y, &motion_magnitudes, &motion_classification);
        let smoothed_angle = self.adaptive_smooth_signal(
            &trajectory.angle,
            &motion_magnitudes,
            &motion_classification,
        );
        let smoothed_scale = self.adaptive_smooth_signal(
            &trajectory.scale,
            &motion_magnitudes,
            &motion_classification,
        );

        Ok(Trajectory {
            x: smoothed_x,
            y: smoothed_y,
            angle: smoothed_angle,
            scale: smoothed_scale,
            frame_count: trajectory.frame_count,
        })
    }

    /// Classify motion at each frame.
    fn classify_motion(&self, magnitudes: &Array1<f64>) -> Vec<MotionType> {
        magnitudes
            .iter()
            .map(|&mag| {
                if mag < self.motion_threshold_low {
                    MotionType::Stable
                } else if mag < self.motion_threshold_high {
                    MotionType::Moderate
                } else {
                    MotionType::High
                }
            })
            .collect()
    }

    /// Apply adaptive smoothing to a signal.
    fn adaptive_smooth_signal(
        &self,
        data: &Array1<f64>,
        magnitudes: &Array1<f64>,
        classification: &[MotionType],
    ) -> Array1<f64> {
        let n = data.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            let motion_type = classification[i];
            let strength = self.compute_smoothing_strength(motion_type, magnitudes[i]);

            // Compute weighted average in local window
            let window = self.compute_adaptive_window(motion_type);
            let half = window / 2;
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(n);

            let mut sum = 0.0;
            let mut weight_sum = 0.0;

            for j in start..end {
                let distance = (j as i32 - i as i32).unsigned_abs() as f64;
                let weight = self.compute_weight(distance, strength);
                sum += data[j] * weight;
                weight_sum += weight;
            }

            result[i] = if weight_sum > 0.0 {
                sum / weight_sum
            } else {
                data[i]
            };
        }

        result
    }

    /// Compute smoothing strength based on motion type and magnitude.
    fn compute_smoothing_strength(&self, motion_type: MotionType, magnitude: f64) -> f64 {
        match motion_type {
            MotionType::Stable => {
                // High smoothing for stable motion
                0.9
            }
            MotionType::Moderate => {
                // Medium smoothing, scaled by magnitude
                let ratio = magnitude / self.motion_threshold_high;
                0.7 * (1.0 - ratio * 0.5)
            }
            MotionType::High => {
                // Low smoothing for high motion to preserve intentional camera moves
                0.3
            }
        }
    }

    /// Compute adaptive window size based on motion type.
    fn compute_adaptive_window(&self, motion_type: MotionType) -> usize {
        match motion_type {
            MotionType::Stable => self.window_size,
            MotionType::Moderate => self.window_size / 2,
            MotionType::High => self.window_size / 4,
        }
    }

    /// Compute weight based on distance and smoothing strength.
    fn compute_weight(&self, distance: f64, strength: f64) -> f64 {
        let sigma = self.window_size as f64 * strength / 3.0;
        (-0.5 * (distance / sigma).powi(2)).exp()
    }

    /// Detect sudden motion changes (shocks).
    fn detect_shocks(&self, magnitudes: &Array1<f64>) -> Vec<bool> {
        let n = magnitudes.len();
        let mut shocks = vec![false; n];

        if n < 2 {
            return shocks;
        }

        // Detect large changes in magnitude
        for i in 1..n {
            let delta = (magnitudes[i] - magnitudes[i - 1]).abs();
            if delta > self.max_motion * 0.3 {
                shocks[i] = true;
            }
        }

        shocks
    }

    /// Apply shock-aware smoothing.
    pub fn smooth_with_shock_detection(
        &mut self,
        trajectory: &Trajectory,
    ) -> StabilizeResult<Trajectory> {
        if trajectory.is_empty() {
            return Err(StabilizeError::EmptyFrameSequence);
        }

        let motion_magnitudes = trajectory.motion_magnitudes();
        let shocks = self.detect_shocks(&motion_magnitudes);

        // Split trajectory at shock points
        let segments = self.split_at_shocks(trajectory, &shocks);

        // Smooth each segment independently
        let mut smoothed_segments = Vec::new();
        for segment in segments {
            let smoothed = self.smooth(&segment)?;
            smoothed_segments.push(smoothed);
        }

        // Merge segments
        self.merge_segments(&smoothed_segments)
    }

    /// Split trajectory at shock points.
    fn split_at_shocks(&self, trajectory: &Trajectory, shocks: &[bool]) -> Vec<Trajectory> {
        let mut segments = Vec::new();
        let mut start = 0;

        for (i, &is_shock) in shocks.iter().enumerate() {
            if is_shock && i > start {
                // Create segment from start to i
                let segment = self.extract_segment(trajectory, start, i);
                segments.push(segment);
                start = i;
            }
        }

        // Add final segment
        if start < trajectory.len() {
            let segment = self.extract_segment(trajectory, start, trajectory.len());
            segments.push(segment);
        }

        if segments.is_empty() {
            segments.push(trajectory.clone());
        }

        segments
    }

    /// Extract a trajectory segment.
    fn extract_segment(&self, trajectory: &Trajectory, start: usize, end: usize) -> Trajectory {
        let len = end - start;
        Trajectory {
            x: trajectory
                .x
                .slice(scirs2_core::ndarray::s![start..end])
                .to_owned(),
            y: trajectory
                .y
                .slice(scirs2_core::ndarray::s![start..end])
                .to_owned(),
            angle: trajectory
                .angle
                .slice(scirs2_core::ndarray::s![start..end])
                .to_owned(),
            scale: trajectory
                .scale
                .slice(scirs2_core::ndarray::s![start..end])
                .to_owned(),
            frame_count: len,
        }
    }

    /// Merge trajectory segments.
    fn merge_segments(&self, segments: &[Trajectory]) -> StabilizeResult<Trajectory> {
        if segments.is_empty() {
            return Err(StabilizeError::EmptyFrameSequence);
        }

        if segments.len() == 1 {
            return Ok(segments[0].clone());
        }

        let total_frames: usize = segments.iter().map(|s| s.frame_count).sum();
        let mut x = Vec::with_capacity(total_frames);
        let mut y = Vec::with_capacity(total_frames);
        let mut angle = Vec::with_capacity(total_frames);
        let mut scale = Vec::with_capacity(total_frames);

        for segment in segments {
            x.extend(segment.x.iter());
            y.extend(segment.y.iter());
            angle.extend(segment.angle.iter());
            scale.extend(segment.scale.iter());
        }

        Ok(Trajectory {
            x: Array1::from_vec(x),
            y: Array1::from_vec(y),
            angle: Array1::from_vec(angle),
            scale: Array1::from_vec(scale),
            frame_count: total_frames,
        })
    }
}

/// Motion classification for adaptive smoothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MotionType {
    /// Stable motion (small magnitude)
    Stable,
    /// Moderate motion
    Moderate,
    /// High motion (large magnitude)
    High,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_smoother_creation() {
        let smoother = AdaptiveSmoother::new(30, 100.0);
        assert_eq!(smoother.window_size, 30);
        assert!((smoother.max_motion - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_motion_classification() {
        let smoother = AdaptiveSmoother::new(30, 100.0);
        let magnitudes = Array1::from_vec(vec![5.0, 25.0, 75.0]);
        let classification = smoother.classify_motion(&magnitudes);

        assert_eq!(classification[0], MotionType::Stable);
        assert_eq!(classification[1], MotionType::Moderate);
        assert_eq!(classification[2], MotionType::High);
    }

    #[test]
    fn test_smoothing_strength() {
        let smoother = AdaptiveSmoother::new(30, 100.0);

        let stable_strength = smoother.compute_smoothing_strength(MotionType::Stable, 5.0);
        let high_strength = smoother.compute_smoothing_strength(MotionType::High, 75.0);

        assert!(stable_strength > high_strength);
    }

    #[test]
    fn test_shock_detection() {
        let smoother = AdaptiveSmoother::new(30, 100.0);
        let magnitudes = Array1::from_vec(vec![5.0, 5.0, 50.0, 5.0]);
        let shocks = smoother.detect_shocks(&magnitudes);

        assert!(!shocks[0]);
        assert!(!shocks[1]);
        assert!(shocks[2]);
    }
}
