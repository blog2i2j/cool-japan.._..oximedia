//! Collaborative annotation layer for frame-level review and markup.
//!
//! Provides geometric shapes, per-annotation metadata, and a layer container
//! for managing annotations within a collaborative review workflow.

#![allow(dead_code)]

/// Geometric shape of an annotation.
#[derive(Debug, Clone, PartialEq)]
pub enum AnnotationShape {
    /// A single point at `(x, y)` in normalised frame coordinates.
    Point(f32, f32),
    /// An axis-aligned rectangle `(x, y, width, height)`.
    Rect(f32, f32, f32, f32),
    /// An arrow from `(x1, y1)` to `(x2, y2)`.
    Arrow(f32, f32, f32, f32),
}

impl AnnotationShape {
    /// Return the geometric centre of the shape.
    #[must_use]
    pub fn center(&self) -> (f32, f32) {
        match *self {
            Self::Point(x, y) => (x, y),
            Self::Rect(x, y, w, h) => (x + w / 2.0, y + h / 2.0),
            Self::Arrow(x1, y1, x2, y2) => ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        }
    }
}

/// A single annotation placed by a collaborator on an asset frame.
#[derive(Debug, Clone)]
pub struct Annotation {
    /// Unique annotation identifier within the layer.
    pub id: u64,
    /// Display name of the annotation's author.
    pub author: String,
    /// Geometric shape of the annotation.
    pub shape: AnnotationShape,
    /// RGB display colour of the annotation.
    pub color: [u8; 3],
    /// Optional text comment attached to the annotation.
    pub text: String,
    /// Wall-clock creation time in milliseconds since the Unix epoch.
    pub timestamp_ms: u64,
    /// Whether this annotation has been marked as resolved.
    pub resolved: bool,
}

impl Annotation {
    /// Mark the annotation as resolved.
    pub fn resolve(&mut self) {
        self.resolved = true;
    }

    /// Return the age of this annotation relative to `now` (milliseconds).
    #[must_use]
    pub fn age_ms(&self, now: u64) -> u64 {
        now.saturating_sub(self.timestamp_ms)
    }
}

/// A collection of annotations attached to a specific asset and optional frame.
pub struct AnnotationLayer {
    /// Asset identifier this layer belongs to.
    pub asset_id: String,
    /// Optional frame number; `None` means the annotation is asset-wide.
    pub frame: Option<u64>,
    /// All stored annotations.
    pub annotations: Vec<Annotation>,
    /// Counter used to assign unique ids.
    pub next_id: u64,
}

impl AnnotationLayer {
    /// Create a new, empty `AnnotationLayer`.
    #[must_use]
    pub fn new(asset_id: impl Into<String>, frame: Option<u64>) -> Self {
        Self {
            asset_id: asset_id.into(),
            frame,
            annotations: Vec::new(),
            next_id: 1,
        }
    }

    /// Add a new annotation and return its assigned id.
    pub fn add(
        &mut self,
        author: impl Into<String>,
        shape: AnnotationShape,
        color: [u8; 3],
        text: impl Into<String>,
        now_ms: u64,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.annotations.push(Annotation {
            id,
            author: author.into(),
            shape,
            color,
            text: text.into(),
            timestamp_ms: now_ms,
            resolved: false,
        });
        id
    }

    /// Mark the annotation with the given `id` as resolved.
    ///
    /// Returns `true` if the annotation was found and resolved.
    pub fn resolve(&mut self, id: u64) -> bool {
        if let Some(ann) = self.annotations.iter_mut().find(|a| a.id == id) {
            ann.resolve();
            true
        } else {
            false
        }
    }

    /// Return references to all unresolved annotations.
    #[must_use]
    pub fn unresolved(&self) -> Vec<&Annotation> {
        self.annotations.iter().filter(|a| !a.resolved).collect()
    }

    /// Return references to all annotations created by `author`.
    #[must_use]
    pub fn by_author(&self, author: &str) -> Vec<&Annotation> {
        self.annotations
            .iter()
            .filter(|a| a.author == author)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_layer() -> AnnotationLayer {
        AnnotationLayer::new("asset-001", Some(42))
    }

    // ---- AnnotationShape ----

    #[test]
    fn test_point_center() {
        let s = AnnotationShape::Point(0.3, 0.7);
        let (cx, cy) = s.center();
        assert!((cx - 0.3).abs() < 1e-6);
        assert!((cy - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_rect_center() {
        // rect at (0,0) with w=4, h=2 → center (2, 1)
        let s = AnnotationShape::Rect(0.0, 0.0, 4.0, 2.0);
        let (cx, cy) = s.center();
        assert!((cx - 2.0).abs() < 1e-6);
        assert!((cy - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_arrow_center() {
        let s = AnnotationShape::Arrow(0.0, 0.0, 1.0, 1.0);
        let (cx, cy) = s.center();
        assert!((cx - 0.5).abs() < 1e-6);
        assert!((cy - 0.5).abs() < 1e-6);
    }

    // ---- Annotation ----

    #[test]
    fn test_annotation_resolve() {
        let mut layer = make_layer();
        let id = layer.add(
            "alice",
            AnnotationShape::Point(0.5, 0.5),
            [255, 0, 0],
            "note",
            1000,
        );
        layer.resolve(id);
        assert!(layer.annotations[0].resolved);
    }

    #[test]
    fn test_annotation_age_ms() {
        let mut layer = make_layer();
        let id = layer.add(
            "alice",
            AnnotationShape::Point(0.0, 0.0),
            [0, 0, 0],
            "",
            1000,
        );
        let ann = layer
            .annotations
            .iter()
            .find(|a| a.id == id)
            .expect("collab test operation should succeed");
        assert_eq!(ann.age_ms(3000), 2000);
    }

    #[test]
    fn test_annotation_age_ms_before_creation() {
        let mut layer = make_layer();
        let id = layer.add(
            "alice",
            AnnotationShape::Point(0.0, 0.0),
            [0, 0, 0],
            "",
            5000,
        );
        let ann = layer
            .annotations
            .iter()
            .find(|a| a.id == id)
            .expect("collab test operation should succeed");
        // now < timestamp → saturating_sub → 0
        assert_eq!(ann.age_ms(1000), 0);
    }

    // ---- AnnotationLayer ----

    #[test]
    fn test_layer_add_returns_incrementing_ids() {
        let mut layer = make_layer();
        let id1 = layer.add("a", AnnotationShape::Point(0.0, 0.0), [0, 0, 0], "", 0);
        let id2 = layer.add("b", AnnotationShape::Point(0.0, 0.0), [0, 0, 0], "", 0);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }

    #[test]
    fn test_layer_new_is_empty() {
        let layer = make_layer();
        assert!(layer.annotations.is_empty());
        assert_eq!(layer.asset_id, "asset-001");
        assert_eq!(layer.frame, Some(42));
    }

    #[test]
    fn test_layer_resolve_returns_true_when_found() {
        let mut layer = make_layer();
        let id = layer.add("x", AnnotationShape::Point(0.0, 0.0), [0, 0, 0], "", 0);
        assert!(layer.resolve(id));
    }

    #[test]
    fn test_layer_resolve_returns_false_when_not_found() {
        let mut layer = make_layer();
        assert!(!layer.resolve(999));
    }

    #[test]
    fn test_layer_unresolved_filters_resolved() {
        let mut layer = make_layer();
        let id1 = layer.add("a", AnnotationShape::Point(0.0, 0.0), [0, 0, 0], "", 0);
        layer.add("b", AnnotationShape::Point(0.0, 0.0), [0, 0, 0], "", 0);
        layer.resolve(id1);
        let unresolved = layer.unresolved();
        assert_eq!(unresolved.len(), 1);
        assert_eq!(unresolved[0].author, "b");
    }

    #[test]
    fn test_layer_unresolved_all_resolved() {
        let mut layer = make_layer();
        let id = layer.add("a", AnnotationShape::Point(0.0, 0.0), [0, 0, 0], "", 0);
        layer.resolve(id);
        assert!(layer.unresolved().is_empty());
    }

    #[test]
    fn test_layer_by_author() {
        let mut layer = make_layer();
        layer.add("alice", AnnotationShape::Point(0.0, 0.0), [0, 0, 0], "", 0);
        layer.add("bob", AnnotationShape::Point(0.0, 0.0), [0, 0, 0], "", 0);
        layer.add(
            "alice",
            AnnotationShape::Rect(0.0, 0.0, 1.0, 1.0),
            [0, 0, 0],
            "",
            0,
        );
        let alice = layer.by_author("alice");
        assert_eq!(alice.len(), 2);
        assert!(alice.iter().all(|a| a.author == "alice"));
    }

    #[test]
    fn test_layer_by_author_missing() {
        let layer = make_layer();
        assert!(layer.by_author("nobody").is_empty());
    }

    #[test]
    fn test_annotation_color_stored() {
        let mut layer = make_layer();
        layer.add("a", AnnotationShape::Point(0.0, 0.0), [10, 20, 30], "", 0);
        assert_eq!(layer.annotations[0].color, [10, 20, 30]);
    }

    #[test]
    fn test_annotation_text_stored() {
        let mut layer = make_layer();
        layer.add("a", AnnotationShape::Point(0.0, 0.0), [0, 0, 0], "hello", 0);
        assert_eq!(layer.annotations[0].text, "hello");
    }

    #[test]
    fn test_layer_frame_none() {
        let layer = AnnotationLayer::new("asset-002", None);
        assert_eq!(layer.frame, None);
    }
}
