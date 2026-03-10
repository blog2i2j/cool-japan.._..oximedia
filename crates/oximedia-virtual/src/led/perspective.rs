//! Perspective correction for LED walls
//!
//! Computes perspective-correct rendering based on camera position
//! to create the illusion of depth on flat LED panels.

use super::LedPanel;
use crate::math::{Matrix4, Point3, Vector3};
use crate::{tracking::CameraPose, Result};
use serde::{Deserialize, Serialize};

/// Perspective correction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerspectiveCorrectionConfig {
    /// Field of view in degrees
    pub fov: f64,
    /// Near clipping plane
    pub near_plane: f64,
    /// Far clipping plane
    pub far_plane: f64,
    /// Enable sub-pixel accuracy
    pub subpixel_accuracy: bool,
}

impl Default for PerspectiveCorrectionConfig {
    fn default() -> Self {
        Self {
            fov: 90.0,
            near_plane: 0.1,
            far_plane: 100.0,
            subpixel_accuracy: true,
        }
    }
}

/// Perspective correction for LED walls
pub struct PerspectiveCorrection {
    config: PerspectiveCorrectionConfig,
}

impl PerspectiveCorrection {
    /// Create new perspective correction
    pub fn new() -> Result<Self> {
        Ok(Self {
            config: PerspectiveCorrectionConfig::default(),
        })
    }

    /// Create with configuration
    pub fn with_config(config: PerspectiveCorrectionConfig) -> Result<Self> {
        Ok(Self { config })
    }

    /// Compute perspective transform for panel
    pub fn compute_transform(
        &self,
        camera_pose: &CameraPose,
        panel: &LedPanel,
    ) -> Result<Matrix4<f64>> {
        // Build view matrix from camera pose
        let view = self.build_view_matrix(camera_pose);

        // Build projection matrix
        let aspect = panel.aspect_ratio();
        let projection = self.build_projection_matrix(aspect);

        // Combine view and projection
        Ok(projection * view)
    }

    /// Build view matrix from camera pose
    fn build_view_matrix(&self, camera_pose: &CameraPose) -> Matrix4<f64> {
        let eye = camera_pose.position;
        let forward = camera_pose.forward();
        let up = camera_pose.up();
        let target = eye + forward;

        Self::look_at(&eye, &target, &up)
    }

    /// Build projection matrix
    fn build_projection_matrix(&self, aspect: f64) -> Matrix4<f64> {
        let fov_rad = self.config.fov.to_radians();
        let f = 1.0 / (fov_rad / 2.0).tan();

        let mut proj = Matrix4::zeros();
        proj.data[0][0] = f / aspect;
        proj.data[1][1] = f;
        proj.data[2][2] = (self.config.far_plane + self.config.near_plane)
            / (self.config.near_plane - self.config.far_plane);
        proj.data[2][3] = (2.0 * self.config.far_plane * self.config.near_plane)
            / (self.config.near_plane - self.config.far_plane);
        proj.data[3][2] = -1.0;

        proj
    }

    /// Look-at matrix construction
    fn look_at(eye: &Point3<f64>, target: &Point3<f64>, up: &Vector3<f64>) -> Matrix4<f64> {
        let f = (target - eye).normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(&f);

        let mut view = Matrix4::identity();
        view.data[0][0] = s.x;
        view.data[0][1] = s.y;
        view.data[0][2] = s.z;
        view.data[1][0] = u.x;
        view.data[1][1] = u.y;
        view.data[1][2] = u.z;
        view.data[2][0] = -f.x;
        view.data[2][1] = -f.y;
        view.data[2][2] = -f.z;
        view.data[0][3] = -s.dot(&eye.coords());
        view.data[1][3] = -u.dot(&eye.coords());
        view.data[2][3] = f.dot(&eye.coords());

        view
    }

    /// Correct pixel coordinates for perspective
    pub fn correct_pixel(
        &self,
        pixel_x: usize,
        pixel_y: usize,
        panel: &LedPanel,
        camera_pose: &CameraPose,
    ) -> Result<(f64, f64)> {
        // Compute world position of pixel
        let x = (pixel_x as f64 / panel.resolution.0 as f64) * panel.width;
        let y = (pixel_y as f64 / panel.resolution.1 as f64) * panel.height;
        let world_pos = panel.position + Vector3::new(x, y, 0.0);

        // Transform to camera space
        let transform = self.compute_transform(camera_pose, panel)?;
        let transformed = transform * world_pos.to_homogeneous();

        // Project to screen space
        let w = if transformed[3].abs() > 1e-15 {
            transformed[3]
        } else {
            1.0
        };
        let screen_x = transformed[0] / w;
        let screen_y = transformed[1] / w;

        Ok((screen_x, screen_y))
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &PerspectiveCorrectionConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::UnitQuaternion;

    #[test]
    fn test_perspective_correction_creation() {
        let correction = PerspectiveCorrection::new();
        assert!(correction.is_ok());
    }

    #[test]
    fn test_perspective_with_config() {
        let config = PerspectiveCorrectionConfig::default();
        let correction = PerspectiveCorrection::with_config(config);
        assert!(correction.is_ok());
    }

    #[test]
    fn test_build_projection_matrix() {
        let correction = PerspectiveCorrection::new().expect("should succeed in test");
        let proj = correction.build_projection_matrix(16.0 / 9.0);

        // Check that matrix is non-zero
        assert!(proj[(0, 0)] != 0.0);
        assert!(proj[(1, 1)] != 0.0);
    }

    #[test]
    fn test_look_at_matrix() {
        let eye = Point3::new(0.0, 0.0, 5.0);
        let target = Point3::new(0.0, 0.0, 0.0);
        let up = Vector3::new(0.0, 1.0, 0.0);

        let view = PerspectiveCorrection::look_at(&eye, &target, &up);

        // Check that we get a valid transformation
        assert!(view[(3, 3)] != 0.0);
    }

    #[test]
    fn test_compute_transform() {
        let correction = PerspectiveCorrection::new().expect("should succeed in test");
        let pose = CameraPose::new(Point3::new(0.0, 0.0, 5.0), UnitQuaternion::identity(), 0);
        let panel = LedPanel::new(Point3::origin(), 5.0, 3.0, (1920, 1080), 2.5);

        let transform = correction.compute_transform(&pose, &panel);
        assert!(transform.is_ok());
    }
}
