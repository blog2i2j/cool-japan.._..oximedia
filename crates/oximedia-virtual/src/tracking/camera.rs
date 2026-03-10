//! Camera position and rotation tracking
//!
//! Provides real-time 6DOF (six degrees of freedom) camera tracking
//! using optical markers, IMU sensors, and sensor fusion.

use super::{imu::ImuSensor, markers::MarkerDetector, CameraPose};
use crate::math::{Point3, UnitQuaternion, Vector3};
use crate::{Result, VirtualProductionError};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Instant;

/// Camera tracker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraTrackerConfig {
    /// Update rate in Hz
    pub update_rate: f64,
    /// Enable optical tracking
    pub optical_tracking: bool,
    /// Enable IMU tracking
    pub imu_tracking: bool,
    /// Sensor fusion weight (0.0 = IMU only, 1.0 = optical only)
    pub fusion_weight: f32,
    /// Position smoothing window
    pub smoothing_window: usize,
    /// Maximum tracking latency in milliseconds
    pub max_latency_ms: f64,
}

impl Default for CameraTrackerConfig {
    fn default() -> Self {
        Self {
            update_rate: 120.0,
            optical_tracking: true,
            imu_tracking: true,
            fusion_weight: 0.7,
            smoothing_window: 5,
            max_latency_ms: 60_000.0,
        }
    }
}

/// Camera tracking state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackingState {
    /// No tracking data available
    Lost,
    /// Tracking with reduced confidence
    Limited,
    /// Full tracking available
    Tracking,
}

/// Camera tracker
pub struct CameraTracker {
    config: CameraTrackerConfig,
    marker_detector: Option<MarkerDetector>,
    imu_sensor: Option<ImuSensor>,
    current_pose: CameraPose,
    pose_history: VecDeque<CameraPose>,
    tracking_state: TrackingState,
    last_update: Option<Instant>,
}

impl CameraTracker {
    /// Create new camera tracker
    pub fn new(config: CameraTrackerConfig) -> Result<Self> {
        let marker_detector = if config.optical_tracking {
            Some(MarkerDetector::new(Default::default())?)
        } else {
            None
        };

        let imu_sensor = if config.imu_tracking {
            Some(ImuSensor::new(Default::default())?)
        } else {
            None
        };

        let pose_history = VecDeque::with_capacity(config.smoothing_window);

        Ok(Self {
            config,
            marker_detector,
            imu_sensor,
            current_pose: CameraPose::default(),
            pose_history,
            tracking_state: TrackingState::Lost,
            last_update: None,
        })
    }

    /// Update camera tracking
    pub fn update(&mut self, timestamp_ns: u64) -> Result<CameraPose> {
        let now = Instant::now();

        // Check latency
        if let Some(last) = self.last_update {
            let latency = now.duration_since(last);
            if latency.as_secs_f64() * 1000.0 > self.config.max_latency_ms {
                self.tracking_state = TrackingState::Lost;
                return Err(VirtualProductionError::CameraTracking(format!(
                    "Tracking latency exceeded: {latency:?}"
                )));
            }
        }

        // Get optical tracking data
        let optical_pose = if let Some(detector) = &mut self.marker_detector {
            detector.detect_pose(timestamp_ns)?
        } else {
            None
        };

        // Get IMU tracking data
        let imu_pose = if let Some(imu) = &mut self.imu_sensor {
            Some(imu.get_pose(timestamp_ns)?)
        } else {
            None
        };

        // Sensor fusion
        let fused_pose = self.fuse_sensors(optical_pose, imu_pose, timestamp_ns)?;

        // Update tracking state
        self.tracking_state = self.determine_tracking_state(&fused_pose);

        // Smooth pose
        self.pose_history.push_back(fused_pose);
        if self.pose_history.len() > self.config.smoothing_window {
            self.pose_history.pop_front();
        }

        self.current_pose = self.smooth_pose();
        self.last_update = Some(now);

        Ok(self.current_pose)
    }

    /// Fuse optical and IMU sensor data
    fn fuse_sensors(
        &self,
        optical: Option<CameraPose>,
        imu: Option<CameraPose>,
        timestamp_ns: u64,
    ) -> Result<CameraPose> {
        match (optical, imu) {
            (Some(opt), Some(imu_pose)) => {
                // Weighted fusion
                let weight = f64::from(self.config.fusion_weight);
                let position = opt.position * weight + imu_pose.position.coords() * (1.0 - weight);
                let orientation = opt.orientation.slerp(&imu_pose.orientation, 1.0 - weight);
                let confidence = opt.confidence * self.config.fusion_weight
                    + imu_pose.confidence * (1.0 - self.config.fusion_weight);

                Ok(CameraPose {
                    position: Point3::from(position),
                    orientation,
                    timestamp_ns,
                    confidence,
                })
            }
            (Some(opt), None) => Ok(opt),
            (None, Some(imu_pose)) => Ok(imu_pose),
            (None, None) => {
                // Use last known pose with reduced confidence
                let mut pose = self.current_pose;
                pose.confidence *= 0.5;
                pose.timestamp_ns = timestamp_ns;
                Ok(pose)
            }
        }
    }

    /// Smooth pose using history
    fn smooth_pose(&self) -> CameraPose {
        if self.pose_history.is_empty() {
            return self.current_pose;
        }

        let n = self.pose_history.len() as f64;
        let mut avg_position = Vector3::zeros();
        let mut avg_orientation = UnitQuaternion::identity();
        let mut avg_confidence = 0.0;

        for pose in &self.pose_history {
            avg_position += pose.position.coords() / n;
            avg_confidence += pose.confidence / n as f32;
        }

        // Use most recent orientation (SLERP doesn't average well)
        if let Some(last) = self.pose_history.back() {
            avg_orientation = last.orientation;
        }

        CameraPose {
            position: Point3::from(avg_position),
            orientation: avg_orientation,
            timestamp_ns: self.current_pose.timestamp_ns,
            confidence: avg_confidence,
        }
    }

    /// Determine tracking state based on pose quality
    fn determine_tracking_state(&self, pose: &CameraPose) -> TrackingState {
        if pose.confidence > 0.8 {
            TrackingState::Tracking
        } else if pose.confidence > 0.3 {
            TrackingState::Limited
        } else {
            TrackingState::Lost
        }
    }

    /// Get current camera pose
    #[must_use]
    pub fn current_pose(&self) -> &CameraPose {
        &self.current_pose
    }

    /// Get tracking state
    #[must_use]
    pub fn tracking_state(&self) -> TrackingState {
        self.tracking_state
    }

    /// Reset tracking
    pub fn reset(&mut self) {
        self.current_pose = CameraPose::default();
        self.pose_history.clear();
        self.tracking_state = TrackingState::Lost;
        self.last_update = None;
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &CameraTrackerConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_tracker_creation() {
        let config = CameraTrackerConfig::default();
        let tracker = CameraTracker::new(config);
        assert!(tracker.is_ok());
    }

    #[test]
    fn test_camera_tracker_state() {
        let config = CameraTrackerConfig::default();
        let tracker = CameraTracker::new(config).expect("should succeed in test");
        assert_eq!(tracker.tracking_state(), TrackingState::Lost);
    }

    #[test]
    fn test_camera_tracker_reset() {
        let config = CameraTrackerConfig::default();
        let mut tracker = CameraTracker::new(config).expect("should succeed in test");
        tracker.reset();
        assert_eq!(tracker.tracking_state(), TrackingState::Lost);
    }

    #[test]
    fn test_tracking_state_determination() {
        let config = CameraTrackerConfig::default();
        let tracker = CameraTracker::new(config).expect("should succeed in test");

        let mut pose = CameraPose::default();
        pose.confidence = 0.9;
        assert_eq!(
            tracker.determine_tracking_state(&pose),
            TrackingState::Tracking
        );

        pose.confidence = 0.5;
        assert_eq!(
            tracker.determine_tracking_state(&pose),
            TrackingState::Limited
        );

        pose.confidence = 0.1;
        assert_eq!(tracker.determine_tracking_state(&pose), TrackingState::Lost);
    }
}
