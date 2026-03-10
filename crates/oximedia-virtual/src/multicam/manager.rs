//! Multi-camera coordination manager

use super::{CameraId, MultiCameraState};
use crate::{tracking::CameraPose, Result, VirtualProductionError};
use serde::{Deserialize, Serialize};

/// Multi-camera configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiCameraConfig {
    /// Number of cameras
    pub num_cameras: usize,
    /// Enable auto-switching
    pub auto_switch: bool,
}

impl Default for MultiCameraConfig {
    fn default() -> Self {
        Self {
            num_cameras: 1,
            auto_switch: false,
        }
    }
}

/// Multi-camera manager
pub struct MultiCameraManager {
    config: MultiCameraConfig,
    state: MultiCameraState,
}

impl MultiCameraManager {
    /// Create new multi-camera manager
    pub fn new(config: MultiCameraConfig) -> Result<Self> {
        if config.num_cameras == 0 {
            return Err(VirtualProductionError::MultiCamera(
                "Number of cameras must be > 0".to_string(),
            ));
        }

        Ok(Self {
            config,
            state: MultiCameraState::new(),
        })
    }

    /// Update camera pose
    pub fn update_camera(&mut self, camera_id: CameraId, pose: CameraPose) {
        if let Some(entry) = self.state.poses.iter_mut().find(|(id, _)| *id == camera_id) {
            entry.1 = pose;
        } else {
            self.state.poses.push((camera_id, pose));
        }
    }

    /// Set active camera
    pub fn set_active_camera(&mut self, camera_id: CameraId) {
        self.state.active_camera = camera_id;
    }

    /// Get active camera
    #[must_use]
    pub fn active_camera(&self) -> CameraId {
        self.state.active_camera
    }

    /// Get active camera pose
    #[must_use]
    pub fn active_pose(&self) -> Option<&CameraPose> {
        self.state.active_pose()
    }

    /// Get all camera poses
    #[must_use]
    pub fn all_poses(&self) -> &[(CameraId, CameraPose)] {
        &self.state.poses
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &MultiCameraConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Point3, UnitQuaternion};

    #[test]
    fn test_multicam_manager() {
        let config = MultiCameraConfig {
            num_cameras: 4,
            auto_switch: false,
        };
        let manager = MultiCameraManager::new(config);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_multicam_update() {
        let config = MultiCameraConfig::default();
        let mut manager = MultiCameraManager::new(config).expect("should succeed in test");

        let pose = CameraPose::new(Point3::origin(), UnitQuaternion::identity(), 0);

        manager.update_camera(CameraId(0), pose);
        assert!(manager.active_pose().is_some());
    }

    #[test]
    fn test_multicam_switch() {
        let config = MultiCameraConfig {
            num_cameras: 2,
            auto_switch: false,
        };
        let mut manager = MultiCameraManager::new(config).expect("should succeed in test");

        manager.set_active_camera(CameraId(1));
        assert_eq!(manager.active_camera(), CameraId(1));
    }
}
