//! Virtual production stage element layout management.
//!
//! Provides a registry of typed stage elements (LED volumes, cameras, tracking
//! markers, light fixtures, etc.) with spatial queries, AABB collision
//! detection, and bounding-box computation.
//!
//! # Design
//! Each element is described by a [`StageElement`] with a unique string `id`,
//! an [`ElementType`], a world-space `position` (metres), a Euler rotation
//! (degrees), and axis-aligned `dimensions` (metres). The [`StageLayout`]
//! struct holds the element registry and exposes high-level query helpers.
//!
//! # Example
//! ```rust
//! use oximedia_virtual::stage_element_layout::{StageLayout, StageElement, ElementType};
//!
//! let mut layout = StageLayout::new();
//! layout.add_element(StageElement {
//!     id: "led_main".to_string(),
//!     element_type: ElementType::LedVolume,
//!     position: [0.0, 2.0, 0.0],
//!     rotation_deg: [0.0; 3],
//!     dimensions: [10.0, 4.0, 0.1],
//! }).expect("unique id");
//! ```

use thiserror::Error;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Functional type of a stage element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ElementType {
    /// LED volume (curved or flat LED screen array).
    LedVolume,
    /// Physical camera on the stage.
    Camera,
    /// Optical or infra-red tracking marker / target.
    TrackingMarker,
    /// Hard or soft light fixture.
    LightFixture,
    /// Green screen surface.
    GreenScreen,
    /// Physical prop item.
    PropItem,
    /// Safety boundary marker / exclusion zone.
    SafetyBoundary,
}

impl std::fmt::Display for ElementType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::LedVolume => "LedVolume",
            Self::Camera => "Camera",
            Self::TrackingMarker => "TrackingMarker",
            Self::LightFixture => "LightFixture",
            Self::GreenScreen => "GreenScreen",
            Self::PropItem => "PropItem",
            Self::SafetyBoundary => "SafetyBoundary",
        };
        write!(f, "{name}")
    }
}

/// A single element within the virtual production stage.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StageElement {
    /// Unique string identifier.
    pub id: String,
    /// Functional type of this element.
    pub element_type: ElementType,
    /// World-space position (X, Y, Z) of the element's origin in metres.
    pub position: [f32; 3],
    /// Euler rotation angles in degrees (roll, pitch, yaw).
    ///
    /// Currently stored for reference; collision detection uses
    /// axis-aligned bounding boxes only.
    pub rotation_deg: [f32; 3],
    /// Axis-aligned bounding-box half-extents (width, height, depth) in metres.
    pub dimensions: [f32; 3],
}

impl StageElement {
    /// Compute the axis-aligned bounding box (AABB) min/max corners.
    ///
    /// Returns `(min, max)` where each is `[x, y, z]` in metres.
    /// The AABB is centred at `self.position` with half-extents
    /// `dimensions / 2` along each axis.
    #[must_use]
    pub fn aabb(&self) -> ([f32; 3], [f32; 3]) {
        let half = [
            self.dimensions[0] * 0.5,
            self.dimensions[1] * 0.5,
            self.dimensions[2] * 0.5,
        ];
        let min = [
            self.position[0] - half[0],
            self.position[1] - half[1],
            self.position[2] - half[2],
        ];
        let max = [
            self.position[0] + half[0],
            self.position[1] + half[1],
            self.position[2] + half[2],
        ];
        (min, max)
    }
}

/// Errors returned by [`StageLayout`] operations.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum StageError {
    /// An element with the same `id` already exists.
    #[error("duplicate element id: {0}")]
    DuplicateId(String),

    /// No element with the requested `id` exists.
    #[error("element not found: {0}")]
    NotFound(String),

    /// One or more dimension values are invalid (e.g. negative or NaN).
    #[error("invalid dimensions for element")]
    InvalidDimensions,
}

/// Registry of all stage elements with spatial query capabilities.
#[derive(Debug, Clone, Default)]
pub struct StageLayout {
    elements: Vec<StageElement>,
}

impl StageLayout {
    /// Create an empty stage layout.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an element to the layout.
    ///
    /// # Errors
    /// - [`StageError::DuplicateId`] if an element with the same `id` already exists.
    /// - [`StageError::InvalidDimensions`] if any dimension value is negative, zero, or non-finite.
    pub fn add_element(&mut self, element: StageElement) -> Result<(), StageError> {
        // Validate dimensions.
        for &d in &element.dimensions {
            if !d.is_finite() || d <= 0.0 {
                return Err(StageError::InvalidDimensions);
            }
        }

        // Check for duplicate id.
        if self.elements.iter().any(|e| e.id == element.id) {
            return Err(StageError::DuplicateId(element.id));
        }

        self.elements.push(element);
        Ok(())
    }

    /// Remove an element by id.
    ///
    /// # Errors
    /// - [`StageError::NotFound`] if no element with that `id` exists.
    pub fn remove_element(&mut self, id: &str) -> Result<(), StageError> {
        let pos = self
            .elements
            .iter()
            .position(|e| e.id == id)
            .ok_or_else(|| StageError::NotFound(id.to_owned()))?;
        self.elements.swap_remove(pos);
        Ok(())
    }

    /// Return references to all elements of the given type.
    #[must_use]
    pub fn find_by_type(&self, element_type: ElementType) -> Vec<&StageElement> {
        self.elements
            .iter()
            .filter(|e| e.element_type == element_type)
            .collect()
    }

    /// Compute the axis-aligned bounding box that encloses **all** elements.
    ///
    /// Returns `None` if the layout is empty, otherwise `Some((min, max))`.
    #[must_use]
    pub fn bounding_box(&self) -> Option<([f32; 3], [f32; 3])> {
        let mut iter = self.elements.iter();
        let first = iter.next()?;
        let (f_min, f_max) = first.aabb();
        let mut global_min = f_min;
        let mut global_max = f_max;

        for elem in iter {
            let (e_min, e_max) = elem.aabb();
            for i in 0..3 {
                global_min[i] = global_min[i].min(e_min[i]);
                global_max[i] = global_max[i].max(e_max[i]);
            }
        }

        Some((global_min, global_max))
    }

    /// Detect pairs of elements whose axis-aligned bounding boxes overlap.
    ///
    /// Returns a `Vec` of `(id_a, id_b)` pairs (lexicographically ordered by
    /// position in the internal array) for every colliding combination.
    /// The detection is O(n²) and intended for small stage layouts (<1000 elements).
    #[must_use]
    pub fn check_collisions(&self) -> Vec<(String, String)> {
        let mut collisions = Vec::new();
        let n = self.elements.len();

        for i in 0..n {
            for j in (i + 1)..n {
                let (a_min, a_max) = self.elements[i].aabb();
                let (b_min, b_max) = self.elements[j].aabb();

                if aabb_overlap(&a_min, &a_max, &b_min, &b_max) {
                    collisions.push((
                        self.elements[i].id.clone(),
                        self.elements[j].id.clone(),
                    ));
                }
            }
        }

        collisions
    }

    /// Return a reference to all elements.
    #[must_use]
    pub fn elements(&self) -> &[StageElement] {
        &self.elements
    }

    /// Return the number of elements currently registered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Return `true` if no elements are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Returns `true` if two AABBs overlap (inclusive boundary test).
#[inline]
fn aabb_overlap(a_min: &[f32; 3], a_max: &[f32; 3], b_min: &[f32; 3], b_max: &[f32; 3]) -> bool {
    a_min[0] <= b_max[0]
        && a_max[0] >= b_min[0]
        && a_min[1] <= b_max[1]
        && a_max[1] >= b_min[1]
        && a_min[2] <= b_max[2]
        && a_max[2] >= b_min[2]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_element(id: &str, etype: ElementType, pos: [f32; 3], dims: [f32; 3]) -> StageElement {
        StageElement {
            id: id.to_owned(),
            element_type: etype,
            position: pos,
            rotation_deg: [0.0; 3],
            dimensions: dims,
        }
    }

    // -----------------------------------------------------------------------
    // Add / remove
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_element_succeeds() {
        let mut layout = StageLayout::new();
        let elem = make_element("led_1", ElementType::LedVolume, [0.0, 2.0, 0.0], [10.0, 4.0, 0.1]);
        layout.add_element(elem).expect("should succeed");
        assert_eq!(layout.len(), 1);
    }

    #[test]
    fn test_add_duplicate_id_rejected() {
        let mut layout = StageLayout::new();
        layout
            .add_element(make_element("cam", ElementType::Camera, [0.0; 3], [0.3, 0.2, 0.2]))
            .expect("first insert ok");
        let err = layout
            .add_element(make_element("cam", ElementType::Camera, [1.0, 0.0, 0.0], [0.3, 0.2, 0.2]))
            .expect_err("duplicate should fail");
        assert_eq!(err, StageError::DuplicateId("cam".to_owned()));
    }

    #[test]
    fn test_add_invalid_dimensions_rejected() {
        let mut layout = StageLayout::new();
        let bad = make_element("bad", ElementType::PropItem, [0.0; 3], [1.0, -0.5, 1.0]);
        assert_eq!(
            layout.add_element(bad).unwrap_err(),
            StageError::InvalidDimensions
        );
    }

    #[test]
    fn test_remove_element_succeeds() {
        let mut layout = StageLayout::new();
        layout
            .add_element(make_element("m1", ElementType::TrackingMarker, [0.0; 3], [0.05, 0.05, 0.05]))
            .expect("add ok");
        layout.remove_element("m1").expect("remove ok");
        assert!(layout.is_empty());
    }

    #[test]
    fn test_remove_nonexistent_returns_error() {
        let mut layout = StageLayout::new();
        let err = layout.remove_element("ghost").expect_err("should fail");
        assert_eq!(err, StageError::NotFound("ghost".to_owned()));
    }

    // -----------------------------------------------------------------------
    // Type filtering
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_by_type_single_type() {
        let mut layout = StageLayout::new();
        layout
            .add_element(make_element("c1", ElementType::Camera, [0.0; 3], [0.3, 0.2, 0.2]))
            .expect("ok");
        layout
            .add_element(make_element("l1", ElementType::LightFixture, [1.0, 0.0, 0.0], [0.5, 0.5, 0.5]))
            .expect("ok");
        layout
            .add_element(make_element("c2", ElementType::Camera, [2.0, 0.0, 0.0], [0.3, 0.2, 0.2]))
            .expect("ok");

        let cameras = layout.find_by_type(ElementType::Camera);
        assert_eq!(cameras.len(), 2, "should find 2 cameras");
        let lights = layout.find_by_type(ElementType::LightFixture);
        assert_eq!(lights.len(), 1, "should find 1 light");
        let gs = layout.find_by_type(ElementType::GreenScreen);
        assert!(gs.is_empty(), "no green screens");
    }

    // -----------------------------------------------------------------------
    // Bounding box
    // -----------------------------------------------------------------------

    #[test]
    fn test_bounding_box_empty() {
        let layout = StageLayout::new();
        assert!(layout.bounding_box().is_none());
    }

    #[test]
    fn test_bounding_box_single_element() {
        let mut layout = StageLayout::new();
        // Element centred at (0,2,0) with dims (10,4,0.1)
        // AABB: min=(-5, 0, -0.05), max=(5, 4, 0.05)
        layout
            .add_element(make_element("led", ElementType::LedVolume, [0.0, 2.0, 0.0], [10.0, 4.0, 0.1]))
            .expect("ok");
        let (min, max) = layout.bounding_box().expect("should have bb");
        assert!((min[0] - (-5.0)).abs() < 1e-5);
        assert!((min[1] - 0.0).abs() < 1e-5);
        assert!((max[0] - 5.0).abs() < 1e-5);
        assert!((max[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_bounding_box_multiple_elements() {
        let mut layout = StageLayout::new();
        // Element A: centred at (0,0,0), dims (2,2,2) → AABB [-1,-1,-1]..[1,1,1]
        layout
            .add_element(make_element("a", ElementType::PropItem, [0.0; 3], [2.0, 2.0, 2.0]))
            .expect("ok");
        // Element B: centred at (4,0,0), dims (2,2,2) → AABB [3,-1,-1]..[5,1,1]
        layout
            .add_element(make_element("b", ElementType::PropItem, [4.0, 0.0, 0.0], [2.0, 2.0, 2.0]))
            .expect("ok");

        let (min, max) = layout.bounding_box().expect("some");
        assert!((min[0] - (-1.0)).abs() < 1e-5, "min x: {}", min[0]);
        assert!((max[0] - 5.0).abs() < 1e-5, "max x: {}", max[0]);
    }

    // -----------------------------------------------------------------------
    // Collision detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_no_collisions_when_separated() {
        let mut layout = StageLayout::new();
        // Two non-overlapping cameras far apart.
        layout
            .add_element(make_element("c1", ElementType::Camera, [0.0; 3], [0.3, 0.2, 0.2]))
            .expect("ok");
        layout
            .add_element(make_element("c2", ElementType::Camera, [10.0, 0.0, 0.0], [0.3, 0.2, 0.2]))
            .expect("ok");
        let collisions = layout.check_collisions();
        assert!(collisions.is_empty(), "no collisions expected");
    }

    #[test]
    fn test_collision_detected_when_overlapping() {
        let mut layout = StageLayout::new();
        // Overlap: both elements at same position.
        layout
            .add_element(make_element("e1", ElementType::PropItem, [0.0; 3], [1.0, 1.0, 1.0]))
            .expect("ok");
        layout
            .add_element(make_element("e2", ElementType::SafetyBoundary, [0.0; 3], [1.0, 1.0, 1.0]))
            .expect("ok");
        let collisions = layout.check_collisions();
        assert_eq!(collisions.len(), 1, "one collision expected");
        assert!(collisions[0].0 == "e1" || collisions[0].1 == "e1");
        assert!(collisions[0].0 == "e2" || collisions[0].1 == "e2");
    }

    #[test]
    fn test_partial_overlap_detected() {
        let mut layout = StageLayout::new();
        // Elements overlapping in X by 0.5 m.
        // A: centre (0,0,0) dims (2,1,1) → AABB [-1,-0.5,-0.5]..[1,0.5,0.5]
        // B: centre (1.5,0,0) dims (2,1,1) → AABB [0.5,-0.5,-0.5]..[2.5,0.5,0.5]
        // X overlap: 0.5..1.0 → overlapping.
        layout
            .add_element(make_element("a", ElementType::LedVolume, [0.0; 3], [2.0, 1.0, 1.0]))
            .expect("ok");
        layout
            .add_element(make_element("b", ElementType::GreenScreen, [1.5, 0.0, 0.0], [2.0, 1.0, 1.0]))
            .expect("ok");
        let collisions = layout.check_collisions();
        assert_eq!(collisions.len(), 1, "partial overlap should be detected");
    }

    #[test]
    fn test_multiple_collisions() {
        let mut layout = StageLayout::new();
        // Three elements all at origin – 3 pairs.
        for (i, etype) in [
            ElementType::Camera,
            ElementType::LightFixture,
            ElementType::TrackingMarker,
        ]
        .iter()
        .enumerate()
        {
            layout
                .add_element(make_element(
                    &format!("e{i}"),
                    *etype,
                    [0.0; 3],
                    [1.0, 1.0, 1.0],
                ))
                .expect("ok");
        }
        let collisions = layout.check_collisions();
        assert_eq!(collisions.len(), 3, "3 pair collisions expected");
    }
}
