//! Gaming stream overlay system.
//!
//! Provides a scene-graph–style overlay compositor for placing graphical
//! elements (text, images, etc.) on top of game capture streams.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// OverlayElement
// ---------------------------------------------------------------------------

/// A rectangular element placed in the overlay scene.
#[derive(Debug, Clone)]
pub struct OverlayElement {
    /// Unique element identifier.
    pub id: u32,
    /// Human-readable name.
    pub name: String,
    /// Horizontal position (pixels from the left edge).
    pub x: f32,
    /// Vertical position (pixels from the top edge).
    pub y: f32,
    /// Element width in pixels.
    pub width: f32,
    /// Element height in pixels.
    pub height: f32,
    /// Whether the element is currently rendered.
    pub visible: bool,
    /// Compositing order: higher values are drawn on top.
    pub z_order: i32,
}

impl OverlayElement {
    /// Create a new visible overlay element.
    #[must_use]
    pub fn new(id: u32, name: impl Into<String>, x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            id,
            name: name.into(),
            x,
            y,
            width,
            height,
            visible: true,
            z_order: 0,
        }
    }

    /// Returns `true` when the point (`px`, `py`) lies within the element's
    /// axis-aligned bounding box.
    #[must_use]
    pub fn contains_point(&self, px: f32, py: f32) -> bool {
        px >= self.x && px < self.x + self.width && py >= self.y && py < self.y + self.height
    }

    /// Returns `true` when this element's bounding box overlaps `other`'s.
    #[must_use]
    pub fn overlaps(&self, other: &OverlayElement) -> bool {
        self.x < other.x + other.width
            && self.x + self.width > other.x
            && self.y < other.y + other.height
            && self.y + self.height > other.y
    }
}

// ---------------------------------------------------------------------------
// TextElement
// ---------------------------------------------------------------------------

/// An overlay element that renders a text string.
#[derive(Debug, Clone)]
pub struct TextElement {
    /// Base overlay geometry and visibility.
    pub base: OverlayElement,
    /// Text content to display.
    pub text: String,
    /// Font size in points.
    pub font_size: u32,
    /// RGBA color (red, green, blue, alpha).
    pub color: [u8; 4],
}

impl TextElement {
    /// Number of Unicode scalar values (characters) in `text`.
    #[must_use]
    pub fn char_count(&self) -> usize {
        self.text.chars().count()
    }
}

// ---------------------------------------------------------------------------
// ImageElement
// ---------------------------------------------------------------------------

/// An overlay element that renders an image from a file path.
#[derive(Debug, Clone)]
pub struct ImageElement {
    /// Base overlay geometry and visibility.
    pub base: OverlayElement,
    /// Filesystem path to the source image.
    pub src_path: String,
    /// Opacity (0.0 = fully transparent, 1.0 = fully opaque).
    pub opacity: f32,
}

impl ImageElement {
    /// Returns `true` when the element is fully transparent (`opacity == 0`).
    #[must_use]
    pub fn is_transparent(&self) -> bool {
        self.opacity <= 0.0
    }
}

// ---------------------------------------------------------------------------
// OverlayScene
// ---------------------------------------------------------------------------

/// A composited overlay scene containing multiple `OverlayElement`s.
#[derive(Debug, Default)]
pub struct OverlayScene {
    /// All elements registered in the scene.
    pub elements: Vec<OverlayElement>,
    /// Scene canvas width in pixels.
    pub width: u32,
    /// Scene canvas height in pixels.
    pub height: u32,
}

impl OverlayScene {
    /// Create an empty scene with the given dimensions.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            elements: Vec::new(),
            width,
            height,
        }
    }

    /// Add an element to the scene.
    pub fn add_element(&mut self, elem: OverlayElement) {
        self.elements.push(elem);
    }

    /// Remove the element with the given `id`.
    ///
    /// Returns `true` if an element was removed.
    pub fn remove_element(&mut self, id: u32) -> bool {
        if let Some(pos) = self.elements.iter().position(|e| e.id == id) {
            self.elements.remove(pos);
            true
        } else {
            false
        }
    }

    /// References to all currently visible elements.
    #[must_use]
    pub fn visible_elements(&self) -> Vec<&OverlayElement> {
        self.elements.iter().filter(|e| e.visible).collect()
    }

    /// Reference to the topmost visible element at canvas position (`x`, `y`),
    /// i.e. the visible element with the highest `z_order` that contains the
    /// point.  Returns `None` if no element contains the point.
    #[must_use]
    pub fn element_at(&self, x: f32, y: f32) -> Option<&OverlayElement> {
        self.elements
            .iter()
            .filter(|e| e.visible && e.contains_point(x, y))
            .max_by_key(|e| e.z_order)
    }

    /// All elements ordered by ascending `z_order` (bottom to top).
    #[must_use]
    pub fn sorted_by_z(&self) -> Vec<&OverlayElement> {
        let mut sorted: Vec<&OverlayElement> = self.elements.iter().collect();
        sorted.sort_by_key(|e| e.z_order);
        sorted
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_elem(id: u32, x: f32, y: f32, w: f32, h: f32) -> OverlayElement {
        OverlayElement::new(id, format!("elem-{id}"), x, y, w, h)
    }

    // OverlayElement

    #[test]
    fn test_contains_point_inside() {
        let e = make_elem(1, 10.0, 10.0, 100.0, 50.0);
        assert!(e.contains_point(50.0, 30.0));
    }

    #[test]
    fn test_contains_point_outside() {
        let e = make_elem(1, 10.0, 10.0, 100.0, 50.0);
        assert!(!e.contains_point(5.0, 30.0));
    }

    #[test]
    fn test_contains_point_on_left_edge() {
        let e = make_elem(1, 10.0, 10.0, 100.0, 50.0);
        assert!(e.contains_point(10.0, 20.0));
    }

    #[test]
    fn test_contains_point_on_right_edge_exclusive() {
        let e = make_elem(1, 10.0, 10.0, 100.0, 50.0);
        // x == x + width is excluded
        assert!(!e.contains_point(110.0, 20.0));
    }

    #[test]
    fn test_overlaps_true() {
        let a = make_elem(1, 0.0, 0.0, 100.0, 100.0);
        let b = make_elem(2, 50.0, 50.0, 100.0, 100.0);
        assert!(a.overlaps(&b));
    }

    #[test]
    fn test_overlaps_false_adjacent() {
        let a = make_elem(1, 0.0, 0.0, 100.0, 100.0);
        let b = make_elem(2, 100.0, 0.0, 100.0, 100.0);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_overlaps_false_far() {
        let a = make_elem(1, 0.0, 0.0, 50.0, 50.0);
        let b = make_elem(2, 200.0, 200.0, 50.0, 50.0);
        assert!(!a.overlaps(&b));
    }

    // TextElement

    #[test]
    fn test_text_element_char_count_ascii() {
        let t = TextElement {
            base: make_elem(1, 0.0, 0.0, 200.0, 30.0),
            text: "Hello".to_string(),
            font_size: 24,
            color: [255, 255, 255, 255],
        };
        assert_eq!(t.char_count(), 5);
    }

    #[test]
    fn test_text_element_char_count_unicode() {
        let t = TextElement {
            base: make_elem(2, 0.0, 0.0, 200.0, 30.0),
            text: "こんにちは".to_string(),
            font_size: 24,
            color: [0, 0, 0, 255],
        };
        assert_eq!(t.char_count(), 5);
    }

    // ImageElement

    #[test]
    fn test_image_element_is_transparent_true() {
        let img = ImageElement {
            base: make_elem(3, 0.0, 0.0, 100.0, 100.0),
            src_path: std::env::temp_dir()
                .join("oximedia-gaming-overlay-logo.png")
                .to_string_lossy()
                .into_owned(),
            opacity: 0.0,
        };
        assert!(img.is_transparent());
    }

    #[test]
    fn test_image_element_is_transparent_false() {
        let img = ImageElement {
            base: make_elem(4, 0.0, 0.0, 100.0, 100.0),
            src_path: std::env::temp_dir()
                .join("oximedia-gaming-overlay-logo.png")
                .to_string_lossy()
                .into_owned(),
            opacity: 0.5,
        };
        assert!(!img.is_transparent());
    }

    // OverlayScene

    #[test]
    fn test_scene_add_and_count() {
        let mut scene = OverlayScene::new(1920, 1080);
        scene.add_element(make_elem(1, 0.0, 0.0, 100.0, 50.0));
        scene.add_element(make_elem(2, 200.0, 0.0, 100.0, 50.0));
        assert_eq!(scene.elements.len(), 2);
    }

    #[test]
    fn test_scene_remove_element_found() {
        let mut scene = OverlayScene::new(1920, 1080);
        scene.add_element(make_elem(1, 0.0, 0.0, 100.0, 50.0));
        assert!(scene.remove_element(1));
        assert!(scene.elements.is_empty());
    }

    #[test]
    fn test_scene_remove_element_not_found() {
        let mut scene = OverlayScene::new(1920, 1080);
        assert!(!scene.remove_element(99));
    }

    #[test]
    fn test_scene_visible_elements() {
        let mut scene = OverlayScene::new(1920, 1080);
        let mut hidden = make_elem(1, 0.0, 0.0, 100.0, 50.0);
        hidden.visible = false;
        scene.add_element(hidden);
        scene.add_element(make_elem(2, 200.0, 0.0, 100.0, 50.0));
        assert_eq!(scene.visible_elements().len(), 1);
    }

    #[test]
    fn test_scene_element_at_topmost_z() {
        let mut scene = OverlayScene::new(1920, 1080);
        let mut bottom = make_elem(1, 0.0, 0.0, 200.0, 200.0);
        bottom.z_order = 0;
        let mut top = make_elem(2, 50.0, 50.0, 100.0, 100.0);
        top.z_order = 10;
        scene.add_element(bottom);
        scene.add_element(top);
        let hit = scene.element_at(75.0, 75.0).expect("element should exist");
        assert_eq!(hit.id, 2);
    }

    #[test]
    fn test_scene_element_at_none() {
        let scene = OverlayScene::new(1920, 1080);
        assert!(scene.element_at(500.0, 500.0).is_none());
    }

    #[test]
    fn test_scene_sorted_by_z() {
        let mut scene = OverlayScene::new(1920, 1080);
        let mut e1 = make_elem(1, 0.0, 0.0, 50.0, 50.0);
        e1.z_order = 5;
        let mut e2 = make_elem(2, 60.0, 0.0, 50.0, 50.0);
        e2.z_order = 1;
        let mut e3 = make_elem(3, 120.0, 0.0, 50.0, 50.0);
        e3.z_order = 10;
        scene.add_element(e1);
        scene.add_element(e2);
        scene.add_element(e3);
        let sorted = scene.sorted_by_z();
        assert_eq!(sorted[0].z_order, 1);
        assert_eq!(sorted[1].z_order, 5);
        assert_eq!(sorted[2].z_order, 10);
    }
}
