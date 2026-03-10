//! Skeleton tracking for motion capture

use crate::math::Point3;

/// Skeleton joint
#[derive(Debug, Clone)]
pub struct Joint {
    /// Joint name
    pub name: String,
    /// Position
    pub position: Point3<f64>,
}

/// Skeleton tracker
pub struct SkeletonTracker {
    #[allow(dead_code)]
    joints: Vec<Joint>,
}

impl SkeletonTracker {
    /// Create new skeleton tracker
    #[must_use]
    pub fn new() -> Self {
        Self { joints: Vec::new() }
    }
}

impl Default for SkeletonTracker {
    fn default() -> Self {
        Self::new()
    }
}
