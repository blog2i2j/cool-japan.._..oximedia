//! IMSC/TTML captions: timing model, style inheritance, and region layout.
//!
//! This module implements core components of the Internet Media Subtitles and Captions (IMSC)
//! standard, based on TTML (Timed Text Markup Language). It covers timing models,
//! style inheritance chains, and region/layout management.

#![allow(dead_code)]
#![allow(missing_docs)]

use std::collections::HashMap;
use std::fmt;

/// Time expression in IMSC/TTML (wall-clock time in milliseconds)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ImscTime(pub u64);

impl ImscTime {
    /// Create from hours, minutes, seconds, frames at the given frame rate
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn from_hmsf(hours: u32, minutes: u32, seconds: u32, frames: u32, fps: f64) -> Self {
        let frame_ms = (f64::from(frames) / fps * 1000.0) as u64;
        let total_ms = u64::from(hours) * 3_600_000
            + u64::from(minutes) * 60_000
            + u64::from(seconds) * 1_000
            + frame_ms;
        Self(total_ms)
    }

    /// Create from milliseconds
    #[must_use]
    pub const fn from_ms(ms: u64) -> Self {
        Self(ms)
    }

    /// Return value as milliseconds
    #[must_use]
    pub const fn as_ms(self) -> u64 {
        self.0
    }

    /// Duration between two time points (saturating subtraction)
    #[must_use]
    pub fn duration_to(self, end: Self) -> u64 {
        end.0.saturating_sub(self.0)
    }
}

impl fmt::Display for ImscTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ms = self.0 % 1000;
        let s = (self.0 / 1000) % 60;
        let m = (self.0 / 60_000) % 60;
        let h = self.0 / 3_600_000;
        write!(f, "{h:02}:{m:02}:{s:02}.{ms:03}")
    }
}

/// IMSC timing semantics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimingMode {
    /// Clock time anchored to media timeline
    MediaTime,
    /// Parallel time container (children share parent begin)
    Parallel,
    /// Sequential time container (children follow one another)
    Sequential,
}

/// A TTML/IMSC time interval
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeInterval {
    pub begin: ImscTime,
    pub end: ImscTime,
    pub mode: TimingMode,
}

impl TimeInterval {
    /// Create a new interval
    #[must_use]
    pub fn new(begin_ms: u64, end_ms: u64) -> Self {
        Self {
            begin: ImscTime::from_ms(begin_ms),
            end: ImscTime::from_ms(end_ms),
            mode: TimingMode::MediaTime,
        }
    }

    /// Test whether a given time falls within this interval
    #[must_use]
    pub fn contains(&self, t: ImscTime) -> bool {
        t >= self.begin && t < self.end
    }

    /// Duration in milliseconds
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.begin.duration_to(self.end)
    }

    /// Intersect with another interval, returning the overlapping span if any
    #[must_use]
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let begin = self.begin.max(other.begin);
        let end = self.end.min(other.end);
        if begin < end {
            Some(Self {
                begin,
                end,
                mode: self.mode,
            })
        } else {
            None
        }
    }
}

// ── Style model ─────────────────────────────────────────────────────────────

/// IMSC color (R, G, B, A each 0–255)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImscColor {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl ImscColor {
    pub const WHITE: Self = Self {
        r: 255,
        g: 255,
        b: 255,
        a: 255,
    };
    pub const BLACK: Self = Self {
        r: 0,
        g: 0,
        b: 0,
        a: 255,
    };
    pub const TRANSPARENT: Self = Self {
        r: 0,
        g: 0,
        b: 0,
        a: 0,
    };
    pub const YELLOW: Self = Self {
        r: 255,
        g: 255,
        b: 0,
        a: 255,
    };

    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }
}

/// Text alignment within a region
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextAlign {
    #[default]
    Left,
    Center,
    Right,
    Start,
    End,
}

/// Font style
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FontStyle {
    #[default]
    Normal,
    Italic,
    Oblique,
}

/// Font weight
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FontWeight {
    #[default]
    Normal,
    Bold,
}

/// Text decoration flags
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TextDecoration {
    pub underline: bool,
    pub line_through: bool,
    pub overline: bool,
}

/// A fully resolved IMSC style set
#[derive(Debug, Clone)]
pub struct ImscStyle {
    pub id: String,
    pub color: ImscColor,
    pub background_color: ImscColor,
    pub font_size_pct: f32,
    pub font_family: String,
    pub font_style: FontStyle,
    pub font_weight: FontWeight,
    pub text_align: TextAlign,
    pub text_decoration: TextDecoration,
    pub line_height_pct: f32,
}

impl Default for ImscStyle {
    fn default() -> Self {
        Self {
            id: String::new(),
            color: ImscColor::WHITE,
            background_color: ImscColor::TRANSPARENT,
            font_size_pct: 100.0,
            font_family: "monospace".to_string(),
            font_style: FontStyle::Normal,
            font_weight: FontWeight::Normal,
            text_align: TextAlign::Center,
            text_decoration: TextDecoration::default(),
            line_height_pct: 125.0,
        }
    }
}

/// Style registry with inheritance resolution
#[derive(Debug, Default)]
pub struct StyleRegistry {
    styles: HashMap<String, ImscStyle>,
    /// `parent_id` → child style ids
    inheritance: HashMap<String, Vec<String>>,
}

impl StyleRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a style
    pub fn register(&mut self, style: ImscStyle) {
        self.styles.insert(style.id.clone(), style);
    }

    /// Register an inheritance relationship (child inherits from parent)
    pub fn set_parent(&mut self, child_id: &str, parent_id: &str) {
        self.inheritance
            .entry(parent_id.to_string())
            .or_default()
            .push(child_id.to_string());
    }

    /// Resolve a style by id (returns default if not found)
    #[must_use]
    pub fn resolve(&self, id: &str) -> ImscStyle {
        self.styles.get(id).cloned().unwrap_or_default()
    }

    /// Merge child into parent (child properties override parent)
    #[must_use]
    pub fn merge(parent: &ImscStyle, child: &ImscStyle) -> ImscStyle {
        ImscStyle {
            id: child.id.clone(),
            color: child.color,
            background_color: child.background_color,
            font_size_pct: child.font_size_pct,
            font_family: if child.font_family.is_empty() {
                parent.font_family.clone()
            } else {
                child.font_family.clone()
            },
            font_style: child.font_style,
            font_weight: child.font_weight,
            text_align: child.text_align,
            text_decoration: child.text_decoration,
            line_height_pct: child.line_height_pct,
        }
    }

    /// Number of registered styles
    #[must_use]
    pub fn len(&self) -> usize {
        self.styles.len()
    }

    /// True if no styles are registered
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.styles.is_empty()
    }
}

// ── Region layout ────────────────────────────────────────────────────────────

/// Unit for region extents
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExtentUnit {
    /// Percentage of root container
    Percentage(f32),
    /// Pixel value
    Pixel(u32),
}

/// A positioned region in the IMSC layout
#[derive(Debug, Clone)]
pub struct ImscRegion {
    pub id: String,
    /// X origin as percentage of container width (0–100)
    pub origin_x_pct: f32,
    /// Y origin as percentage of container height (0–100)
    pub origin_y_pct: f32,
    /// Width as percentage of container width (0–100)
    pub extent_w_pct: f32,
    /// Height as percentage of container height (0–100)
    pub extent_h_pct: f32,
    pub display_align: DisplayAlign,
    pub overflow: RegionOverflow,
    pub style_id: Option<String>,
    pub writing_mode: WritingMode,
}

/// Vertical alignment of content within a region
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DisplayAlign {
    #[default]
    Before,
    Center,
    After,
}

/// How text overflowing a region is handled
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RegionOverflow {
    #[default]
    Hidden,
    Visible,
}

/// Writing direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WritingMode {
    #[default]
    LeftRightTopBottom,
    RightLeftTopBottom,
    TopBottomRightLeft,
}

impl ImscRegion {
    /// Create a standard bottom-subtitle region (80% wide, centred, bottom 15%)
    #[must_use]
    pub fn standard_bottom(id: &str) -> Self {
        Self {
            id: id.to_string(),
            origin_x_pct: 10.0,
            origin_y_pct: 80.0,
            extent_w_pct: 80.0,
            extent_h_pct: 15.0,
            display_align: DisplayAlign::After,
            overflow: RegionOverflow::Hidden,
            style_id: None,
            writing_mode: WritingMode::LeftRightTopBottom,
        }
    }

    /// Create a standard top-subtitle region
    #[must_use]
    pub fn standard_top(id: &str) -> Self {
        Self {
            id: id.to_string(),
            origin_x_pct: 10.0,
            origin_y_pct: 5.0,
            extent_w_pct: 80.0,
            extent_h_pct: 15.0,
            display_align: DisplayAlign::Before,
            overflow: RegionOverflow::Hidden,
            style_id: None,
            writing_mode: WritingMode::LeftRightTopBottom,
        }
    }

    /// Test whether a pixel coordinate lies within this region
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn contains_px(&self, x: u32, y: u32, container_w: u32, container_h: u32) -> bool {
        let ox = self.origin_x_pct / 100.0 * container_w as f32;
        let oy = self.origin_y_pct / 100.0 * container_h as f32;
        let ew = self.extent_w_pct / 100.0 * container_w as f32;
        let eh = self.extent_h_pct / 100.0 * container_h as f32;
        let fx = x as f32;
        let fy = y as f32;
        fx >= ox && fx < ox + ew && fy >= oy && fy < oy + eh
    }
}

/// Registry for all regions in a document
#[derive(Debug, Default)]
pub struct RegionRegistry {
    regions: HashMap<String, ImscRegion>,
}

impl RegionRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, region: ImscRegion) {
        self.regions.insert(region.id.clone(), region);
    }

    #[must_use]
    pub fn get(&self, id: &str) -> Option<&ImscRegion> {
        self.regions.get(id)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.regions.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &ImscRegion> {
        self.regions.values()
    }
}

// ── IMSC document ────────────────────────────────────────────────────────────

/// A single IMSC caption element
#[derive(Debug, Clone)]
pub struct ImscElement {
    pub id: String,
    pub text: String,
    pub timing: TimeInterval,
    pub region_id: Option<String>,
    pub style_id: Option<String>,
}

impl ImscElement {
    #[must_use]
    pub fn new(id: &str, text: &str, begin_ms: u64, end_ms: u64) -> Self {
        Self {
            id: id.to_string(),
            text: text.to_string(),
            timing: TimeInterval::new(begin_ms, end_ms),
            region_id: None,
            style_id: None,
        }
    }

    /// Assign this element to a region
    #[must_use]
    pub fn with_region(mut self, region_id: &str) -> Self {
        self.region_id = Some(region_id.to_string());
        self
    }

    /// Assign a style to this element
    #[must_use]
    pub fn with_style(mut self, style_id: &str) -> Self {
        self.style_id = Some(style_id.to_string());
        self
    }
}

/// A lightweight IMSC document
#[derive(Debug, Default)]
pub struct ImscDocument {
    pub body_timing: Option<TimeInterval>,
    pub styles: StyleRegistry,
    pub regions: RegionRegistry,
    pub elements: Vec<ImscElement>,
}

impl ImscDocument {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return all elements active at time `t`
    #[must_use]
    pub fn active_at(&self, t: ImscTime) -> Vec<&ImscElement> {
        self.elements
            .iter()
            .filter(|e| e.timing.contains(t))
            .collect()
    }

    /// Total number of caption elements
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.elements.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imsc_time_from_hmsf() {
        let t = ImscTime::from_hmsf(0, 1, 30, 0, 25.0);
        assert_eq!(t.as_ms(), 90_000);
    }

    #[test]
    fn test_imsc_time_display() {
        let t = ImscTime::from_ms(3_723_456);
        let s = t.to_string();
        assert!(s.contains("01:02:03"));
    }

    #[test]
    fn test_time_interval_contains() {
        let iv = TimeInterval::new(1000, 5000);
        assert!(iv.contains(ImscTime::from_ms(3000)));
        assert!(!iv.contains(ImscTime::from_ms(500)));
        assert!(!iv.contains(ImscTime::from_ms(5000)));
    }

    #[test]
    fn test_time_interval_duration() {
        let iv = TimeInterval::new(1000, 4000);
        assert_eq!(iv.duration_ms(), 3000);
    }

    #[test]
    fn test_time_interval_intersect() {
        let a = TimeInterval::new(0, 5000);
        let b = TimeInterval::new(3000, 8000);
        let overlap = a.intersect(&b).expect("intersection should succeed");
        assert_eq!(overlap.begin.as_ms(), 3000);
        assert_eq!(overlap.end.as_ms(), 5000);
    }

    #[test]
    fn test_time_interval_no_intersect() {
        let a = TimeInterval::new(0, 1000);
        let b = TimeInterval::new(2000, 3000);
        assert!(a.intersect(&b).is_none());
    }

    #[test]
    fn test_style_registry_register_and_resolve() {
        let mut reg = StyleRegistry::new();
        let mut style = ImscStyle::default();
        style.id = "s1".to_string();
        style.font_size_pct = 80.0;
        reg.register(style);
        let resolved = reg.resolve("s1");
        assert_eq!(resolved.font_size_pct, 80.0);
    }

    #[test]
    fn test_style_registry_missing_returns_default() {
        let reg = StyleRegistry::new();
        let s = reg.resolve("nonexistent");
        assert_eq!(s.font_size_pct, 100.0);
    }

    #[test]
    fn test_style_merge_child_overrides() {
        let parent = ImscStyle::default();
        let mut child = ImscStyle::default();
        child.id = "child".to_string();
        child.color = ImscColor::YELLOW;
        child.font_family = String::new(); // empty → inherit
        let merged = StyleRegistry::merge(&parent, &child);
        assert_eq!(merged.color, ImscColor::YELLOW);
        assert_eq!(merged.font_family, parent.font_family);
    }

    #[test]
    fn test_region_standard_bottom() {
        let r = ImscRegion::standard_bottom("r1");
        assert_eq!(r.id, "r1");
        assert_eq!(r.display_align, DisplayAlign::After);
        assert!(r.origin_y_pct > 50.0);
    }

    #[test]
    fn test_region_contains_px() {
        let r = ImscRegion::standard_bottom("r1");
        // origin 10% of 1920 = 192, extent 80% = 1536; y origin 80% of 1080 = 864
        assert!(r.contains_px(960, 900, 1920, 1080));
        assert!(!r.contains_px(50, 900, 1920, 1080)); // x outside
    }

    #[test]
    fn test_region_registry_operations() {
        let mut reg = RegionRegistry::new();
        reg.register(ImscRegion::standard_bottom("bottom"));
        reg.register(ImscRegion::standard_top("top"));
        assert_eq!(reg.len(), 2);
        assert!(reg.get("bottom").is_some());
        assert!(reg.get("missing").is_none());
    }

    #[test]
    fn test_imsc_element_with_region_and_style() {
        let el = ImscElement::new("e1", "Hello world", 0, 3000)
            .with_region("r1")
            .with_style("s1");
        assert_eq!(el.region_id.as_deref(), Some("r1"));
        assert_eq!(el.style_id.as_deref(), Some("s1"));
    }

    #[test]
    fn test_document_active_at() {
        let mut doc = ImscDocument::new();
        doc.elements.push(ImscElement::new("e1", "First", 0, 2000));
        doc.elements
            .push(ImscElement::new("e2", "Second", 3000, 6000));
        let active = doc.active_at(ImscTime::from_ms(1000));
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, "e1");
    }

    #[test]
    fn test_document_active_multiple() {
        let mut doc = ImscDocument::new();
        doc.elements.push(ImscElement::new("e1", "A", 0, 5000));
        doc.elements.push(ImscElement::new("e2", "B", 2000, 7000));
        doc.elements.push(ImscElement::new("e3", "C", 8000, 10000));
        let active = doc.active_at(ImscTime::from_ms(3000));
        assert_eq!(active.len(), 2);
    }

    #[test]
    fn test_writing_mode_default() {
        let r = ImscRegion::standard_bottom("r");
        assert_eq!(r.writing_mode, WritingMode::LeftRightTopBottom);
    }

    #[test]
    fn test_imsc_color_constants() {
        assert_eq!(ImscColor::WHITE.r, 255);
        assert_eq!(ImscColor::BLACK.r, 0);
        assert_eq!(ImscColor::TRANSPARENT.a, 0);
    }
}
