//! Motion capture integration
//!
//! Provides motion capture integration for actor and prop tracking.

pub mod integrate;
pub mod prop;
pub mod skeleton;

use crate::math::Point3;
use serde::{Deserialize, Serialize};

/// Motion capture marker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MocapMarker {
    /// Marker ID
    pub id: usize,
    /// Position in 3D space
    pub position: Point3<f64>,
    /// Tracking confidence
    pub confidence: f32,
}

impl MocapMarker {
    /// Create new mocap marker
    #[must_use]
    pub fn new(id: usize, position: Point3<f64>) -> Self {
        Self {
            id,
            position,
            confidence: 1.0,
        }
    }
}
