//! Prop tracking for motion capture

use crate::math::Point3;

/// Tracked prop
#[derive(Debug, Clone)]
pub struct Prop {
    /// Prop name
    pub name: String,
    /// Position
    pub position: Point3<f64>,
}

/// Prop tracker
pub struct PropTracker {
    #[allow(dead_code)]
    props: Vec<Prop>,
}

impl PropTracker {
    /// Create new prop tracker
    #[must_use]
    pub fn new() -> Self {
        Self { props: Vec::new() }
    }
}

impl Default for PropTracker {
    fn default() -> Self {
        Self::new()
    }
}
