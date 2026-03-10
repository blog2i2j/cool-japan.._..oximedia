//! Viewing frustum calculation for LED walls
//!
//! Computes the viewing frustum based on camera position and LED wall
//! geometry for accurate perspective rendering.

use super::LedPanel;
use crate::math::{Point3, Vector3};
use crate::{tracking::CameraPose, Result};
use serde::{Deserialize, Serialize};

/// Frustum plane
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FrustumPlane {
    /// Plane normal
    pub normal: Vector3<f64>,
    /// Distance from origin
    pub distance: f64,
}

impl FrustumPlane {
    /// Create new frustum plane
    #[must_use]
    pub fn new(normal: Vector3<f64>, distance: f64) -> Self {
        Self { normal, distance }
    }

    /// Create from point and normal
    #[must_use]
    pub fn from_point_normal(point: &Point3<f64>, normal: Vector3<f64>) -> Self {
        let normalized = normal.normalize();
        let distance = -normalized.dot(&point.coords());
        Self {
            normal: normalized,
            distance,
        }
    }

    /// Test if point is inside (positive side of) plane
    #[must_use]
    pub fn is_inside(&self, point: &Point3<f64>) -> bool {
        self.normal.dot(&point.coords()) + self.distance >= 0.0
    }

    /// Distance from point to plane
    #[must_use]
    pub fn distance_to(&self, point: &Point3<f64>) -> f64 {
        self.normal.dot(&point.coords()) + self.distance
    }
}

/// Viewing frustum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewingFrustum {
    /// Near plane
    pub near: FrustumPlane,
    /// Far plane
    pub far: FrustumPlane,
    /// Left plane
    pub left: FrustumPlane,
    /// Right plane
    pub right: FrustumPlane,
    /// Top plane
    pub top: FrustumPlane,
    /// Bottom plane
    pub bottom: FrustumPlane,
}

impl ViewingFrustum {
    /// Create frustum from camera and panel
    pub fn from_camera_and_panel(camera_pose: &CameraPose, panel: &LedPanel) -> Result<Self> {
        let camera_pos = camera_pose.position;
        let forward = camera_pose.forward();
        let _up = camera_pose.up();
        let _right = camera_pose.right();

        // Compute panel corners
        let panel_min = panel.position;
        let panel_max = panel.position + Vector3::new(panel.width, panel.height, 0.0);

        let corners = [
            panel_min,
            Point3::new(panel_max.x, panel_min.y, panel_min.z),
            Point3::new(panel_max.x, panel_max.y, panel_min.z),
            Point3::new(panel_min.x, panel_max.y, panel_min.z),
        ];

        // Near plane (camera forward)
        let near_distance = 0.1;
        let near =
            FrustumPlane::from_point_normal(&(camera_pos + forward * near_distance), forward);

        // Far plane (behind LED panel)
        let far_distance = (panel.position - camera_pos).norm() + 1.0;
        let far = FrustumPlane::from_point_normal(&(camera_pos + forward * far_distance), -forward);

        // Compute side planes from camera to panel edges
        let left_normal = Self::compute_plane_normal(&camera_pos, &corners[0], &corners[3]);
        let left = FrustumPlane::from_point_normal(&camera_pos, left_normal);

        let right_normal = Self::compute_plane_normal(&camera_pos, &corners[2], &corners[1]);
        let right = FrustumPlane::from_point_normal(&camera_pos, right_normal);

        let top_normal = Self::compute_plane_normal(&camera_pos, &corners[3], &corners[2]);
        let top = FrustumPlane::from_point_normal(&camera_pos, top_normal);

        let bottom_normal = Self::compute_plane_normal(&camera_pos, &corners[1], &corners[0]);
        let bottom = FrustumPlane::from_point_normal(&camera_pos, bottom_normal);

        Ok(Self {
            near,
            far,
            left,
            right,
            top,
            bottom,
        })
    }

    /// Compute plane normal from three points
    fn compute_plane_normal(p0: &Point3<f64>, p1: &Point3<f64>, p2: &Point3<f64>) -> Vector3<f64> {
        let v1 = p1 - p0;
        let v2 = p2 - p0;
        v1.cross(&v2).normalize()
    }

    /// Test if point is inside frustum
    #[must_use]
    pub fn contains_point(&self, point: &Point3<f64>) -> bool {
        self.near.is_inside(point)
            && self.far.is_inside(point)
            && self.left.is_inside(point)
            && self.right.is_inside(point)
            && self.top.is_inside(point)
            && self.bottom.is_inside(point)
    }

    /// Test if sphere is inside or intersecting frustum
    #[must_use]
    pub fn contains_sphere(&self, center: &Point3<f64>, radius: f64) -> bool {
        let planes = [
            &self.near,
            &self.far,
            &self.left,
            &self.right,
            &self.top,
            &self.bottom,
        ];

        for plane in &planes {
            let distance = plane.distance_to(center);
            if distance < -radius {
                return false;
            }
        }

        true
    }

    /// Get frustum corner points
    #[must_use]
    pub fn get_corners(&self) -> [Point3<f64>; 8] {
        // Simplified corner computation
        // In a real implementation, this would compute actual intersection points
        let origin = Point3::origin();
        [
            origin, origin, origin, origin, origin, origin, origin, origin,
        ]
    }

    /// Get all planes
    #[must_use]
    pub fn planes(&self) -> [&FrustumPlane; 6] {
        [
            &self.near,
            &self.far,
            &self.left,
            &self.right,
            &self.top,
            &self.bottom,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::UnitQuaternion;

    #[test]
    fn test_frustum_plane() {
        let plane = FrustumPlane::new(Vector3::new(0.0, 0.0, 1.0), 0.0);
        assert_eq!(plane.normal, Vector3::new(0.0, 0.0, 1.0));
        assert_eq!(plane.distance, 0.0);
    }

    #[test]
    fn test_frustum_plane_from_point_normal() {
        let point = Point3::new(0.0, 0.0, 5.0);
        let normal = Vector3::new(0.0, 0.0, 1.0);
        let plane = FrustumPlane::from_point_normal(&point, normal);

        assert!((plane.distance + 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_frustum_plane_is_inside() {
        let plane = FrustumPlane::new(Vector3::new(0.0, 0.0, 1.0), 0.0);

        assert!(plane.is_inside(&Point3::new(0.0, 0.0, 1.0)));
        assert!(!plane.is_inside(&Point3::new(0.0, 0.0, -1.0)));
    }

    #[test]
    fn test_viewing_frustum_creation() {
        let pose = CameraPose::new(Point3::new(0.0, 0.0, 5.0), UnitQuaternion::identity(), 0);

        let panel = LedPanel::new(Point3::origin(), 5.0, 3.0, (1920, 1080), 2.5);

        let frustum = ViewingFrustum::from_camera_and_panel(&pose, &panel);
        assert!(frustum.is_ok());
    }

    #[test]
    fn test_frustum_contains_point() {
        let pose = CameraPose::new(Point3::new(0.0, 0.0, 5.0), UnitQuaternion::identity(), 0);

        let panel = LedPanel::new(Point3::origin(), 5.0, 3.0, (1920, 1080), 2.5);

        let frustum =
            ViewingFrustum::from_camera_and_panel(&pose, &panel).expect("should succeed in test");

        // Point in front of camera should be inside
        assert!(frustum.contains_point(&Point3::new(0.0, 0.0, 2.0)));
    }

    #[test]
    fn test_frustum_planes() {
        let pose = CameraPose::new(Point3::new(0.0, 0.0, 5.0), UnitQuaternion::identity(), 0);

        let panel = LedPanel::new(Point3::origin(), 5.0, 3.0, (1920, 1080), 2.5);

        let frustum =
            ViewingFrustum::from_camera_and_panel(&pose, &panel).expect("should succeed in test");
        let planes = frustum.planes();
        assert_eq!(planes.len(), 6);
    }
}
