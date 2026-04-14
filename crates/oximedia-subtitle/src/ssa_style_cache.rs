//! Cached ASS/SSA style lookup to avoid re-parsing style definitions per dialogue line.
//!
//! When rendering a large ASS subtitle file, the same named styles (e.g. "Default",
//! "Title", "Notes") are referenced by hundreds or thousands of dialogue lines.
//! Naively re-parsing style definitions for each cue is wasteful; this module
//! provides [`AssStyleCache`] which pre-computes the resolved style for every
//! named style in the file and serves O(1) lookups.
//!
//! Additionally the cache implements a fallback chain: if a requested style is
//! not found, the configured fallback style (typically "Default") is returned
//! instead of an error.  Style overrides (`\an`, `\c`, `\fs` …) applied inline
//! can be merged on top of the cached base style via [`AssStyleCache::resolve`].
//!
//! # Example
//!
//! ```
//! use oximedia_subtitle::ssa_style_cache::{AssStyleCache, StyleResolution};
//!
//! let mut cache = AssStyleCache::new();
//! // Pre-register styles extracted from the [V4+ Styles] section.
//! cache.register("Default", oximedia_subtitle::SubtitleStyle::default());
//! cache.register("Bold", {
//!     let mut s = oximedia_subtitle::SubtitleStyle::default();
//!     s.font_size = 56.0;
//!     s
//! });
//!
//! // O(1) lookup:
//! let resolution = cache.resolve("Bold", None);
//! assert_eq!(resolution.style.font_size, 56.0);
//! assert!(!resolution.used_fallback);
//! ```

#![allow(dead_code)]

use crate::SubtitleStyle;
use std::collections::HashMap;

// ============================================================================
// StyleResolution
// ============================================================================

/// The result of resolving a named ASS style, including fallback information.
#[derive(Clone, Debug)]
pub struct StyleResolution {
    /// The resolved style (base + any applied overrides).
    pub style: SubtitleStyle,
    /// `true` if the named style was not found and the fallback was used.
    pub used_fallback: bool,
    /// The name of the style that was actually used (original or fallback).
    pub effective_name: String,
}

// ============================================================================
// StyleOverride
// ============================================================================

/// A set of per-cue inline overrides to merge on top of the base style.
///
/// Fields set to `Some(…)` replace the corresponding attribute in the base
/// style; `None` fields are left as-is.
#[derive(Clone, Debug, Default)]
pub struct StyleOverride {
    /// Override font size in pixels.
    pub font_size: Option<f32>,
    /// Override primary colour as RGBA.
    pub primary_color: Option<crate::style::Color>,
    /// Override bold.
    pub bold: Option<bool>,
    /// Override italic.
    pub italic: Option<bool>,
    /// Override alignment (1–9 numpad).
    pub alignment: Option<u8>,
    /// Override horizontal margin-left in pixels.
    pub margin_left: Option<u32>,
    /// Override horizontal margin-right in pixels.
    pub margin_right: Option<u32>,
    /// Override vertical margin in pixels.
    pub margin_v: Option<u32>,
}

// ============================================================================
// AssStyleCache
// ============================================================================

/// Cache for named ASS/SSA styles.
///
/// Stores one [`SubtitleStyle`] per named style and resolves lookups in O(1).
#[derive(Clone, Debug, Default)]
pub struct AssStyleCache {
    /// Map from style name → base style.
    styles: HashMap<String, SubtitleStyle>,
    /// Name of the fallback style to use when a lookup fails.
    fallback_name: String,
    /// Number of cache lookups served.
    lookup_count: u64,
    /// Number of lookups that fell back to the default style.
    fallback_count: u64,
}

impl AssStyleCache {
    /// Create a new empty cache.
    ///
    /// The fallback style name defaults to "Default".
    #[must_use]
    pub fn new() -> Self {
        Self {
            styles: HashMap::new(),
            fallback_name: "Default".to_string(),
            lookup_count: 0,
            fallback_count: 0,
        }
    }

    /// Create a cache pre-populated with a map of styles.
    #[must_use]
    pub fn from_map(styles: HashMap<String, SubtitleStyle>) -> Self {
        Self {
            styles,
            fallback_name: "Default".to_string(),
            lookup_count: 0,
            fallback_count: 0,
        }
    }

    /// Set the fallback style name (default: "Default").
    #[must_use]
    pub fn with_fallback(mut self, name: impl Into<String>) -> Self {
        self.fallback_name = name.into();
        self
    }

    /// Register or update a named style in the cache.
    pub fn register(&mut self, name: impl Into<String>, style: SubtitleStyle) {
        self.styles.insert(name.into(), style);
    }

    /// Register multiple styles at once from an iterator of `(name, style)` pairs.
    pub fn register_all(
        &mut self,
        iter: impl IntoIterator<Item = (impl Into<String>, SubtitleStyle)>,
    ) {
        for (name, style) in iter {
            self.styles.insert(name.into(), style);
        }
    }

    /// Resolve a named style with optional inline overrides.
    ///
    /// If the named style is not found, the fallback style is used.
    /// If the fallback is also missing, a plain `Default::default()` is returned
    /// with `used_fallback = true`.
    #[must_use]
    pub fn resolve(&mut self, name: &str, overrides: Option<&StyleOverride>) -> StyleResolution {
        self.lookup_count += 1;

        let (base, effective_name, used_fallback) = if let Some(s) = self.styles.get(name) {
            (s.clone(), name.to_string(), false)
        } else {
            self.fallback_count += 1;
            let fb_name = self.fallback_name.clone();
            let base = self
                .styles
                .get(&fb_name)
                .cloned()
                .unwrap_or_default();
            (base, fb_name, true)
        };

        let style = if let Some(ov) = overrides {
            apply_overrides(base, ov)
        } else {
            base
        };

        StyleResolution {
            style,
            used_fallback,
            effective_name,
        }
    }

    /// Look up a base style without applying overrides or recording metrics.
    ///
    /// Returns `None` if the name is not found (no fallback).
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&SubtitleStyle> {
        self.styles.get(name)
    }

    /// Check whether a named style exists in the cache.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.styles.contains_key(name)
    }

    /// Remove a named style from the cache.
    pub fn remove(&mut self, name: &str) -> Option<SubtitleStyle> {
        self.styles.remove(name)
    }

    /// Number of styles registered in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.styles.len()
    }

    /// `true` if no styles are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.styles.is_empty()
    }

    /// All registered style names.
    #[must_use]
    pub fn style_names(&self) -> Vec<&str> {
        self.styles.keys().map(String::as_str).collect()
    }

    // ------------------------------------------------------------------
    // Metrics
    // ------------------------------------------------------------------

    /// Total number of `resolve()` calls.
    #[must_use]
    pub fn lookup_count(&self) -> u64 {
        self.lookup_count
    }

    /// Number of `resolve()` calls that fell back to the default style.
    #[must_use]
    pub fn fallback_count(&self) -> u64 {
        self.fallback_count
    }

    /// Fraction of lookups that used the fallback style (0.0 if no lookups).
    #[must_use]
    pub fn fallback_ratio(&self) -> f64 {
        if self.lookup_count == 0 {
            return 0.0;
        }
        self.fallback_count as f64 / self.lookup_count as f64
    }

    /// Reset lookup metrics without clearing the style registry.
    pub fn reset_metrics(&mut self) {
        self.lookup_count = 0;
        self.fallback_count = 0;
    }

    /// Clear all registered styles and reset metrics.
    pub fn clear(&mut self) {
        self.styles.clear();
        self.lookup_count = 0;
        self.fallback_count = 0;
    }
}

// ============================================================================
// Override application
// ============================================================================

/// Merge `StyleOverride` fields onto a base style.
fn apply_overrides(mut base: SubtitleStyle, ov: &StyleOverride) -> SubtitleStyle {
    if let Some(size) = ov.font_size {
        base.font_size = size;
    }
    if let Some(color) = ov.primary_color {
        base.primary_color = color;
    }
    if let Some(bold) = ov.bold {
        base.font_weight = if bold {
            crate::style::FontWeight::Bold
        } else {
            crate::style::FontWeight::Normal
        };
    }
    if let Some(italic) = ov.italic {
        base.font_style = if italic {
            crate::style::FontStyle::Italic
        } else {
            crate::style::FontStyle::Normal
        };
    }
    if let Some(ml) = ov.margin_left {
        base.margin_left = ml;
    }
    if let Some(mr) = ov.margin_right {
        base.margin_right = mr;
    }
    if let Some(mv) = ov.margin_v {
        base.margin_bottom = mv;
    }
    base
}

// ============================================================================
// Builder helper
// ============================================================================

/// Builder for constructing an [`AssStyleCache`] from an ASS-like style
/// definition list (simple `"Name,FontName,Size,..."`-formatted strings).
///
/// This is intentionally simple — use [`AssStyleCache::from_map`] or
/// [`AssStyleCache::register`] for richer integration.
#[derive(Debug, Default)]
pub struct StyleCacheBuilder {
    styles: Vec<(String, SubtitleStyle)>,
    fallback: Option<String>,
}

impl StyleCacheBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a named style entry.
    #[must_use]
    pub fn add(mut self, name: impl Into<String>, style: SubtitleStyle) -> Self {
        self.styles.push((name.into(), style));
        self
    }

    /// Set the fallback style name.
    #[must_use]
    pub fn fallback(mut self, name: impl Into<String>) -> Self {
        self.fallback = Some(name.into());
        self
    }

    /// Build the cache.
    #[must_use]
    pub fn build(self) -> AssStyleCache {
        let mut cache = AssStyleCache::new();
        if let Some(fb) = self.fallback {
            cache.fallback_name = fb;
        }
        for (name, style) in self.styles {
            cache.register(name, style);
        }
        cache
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::style::Color;

    fn default_style() -> SubtitleStyle {
        SubtitleStyle::default()
    }

    fn bold_style() -> SubtitleStyle {
        let mut s = SubtitleStyle::default();
        s.font_size = 56.0;
        s
    }

    #[test]
    fn test_register_and_get() {
        let mut cache = AssStyleCache::new();
        cache.register("Default", default_style());
        assert!(cache.contains("Default"));
        assert!(!cache.contains("Missing"));
    }

    #[test]
    fn test_resolve_known_style() {
        let mut cache = AssStyleCache::new();
        cache.register("Default", default_style());
        let r = cache.resolve("Default", None);
        assert!(!r.used_fallback);
        assert_eq!(r.effective_name, "Default");
    }

    #[test]
    fn test_resolve_unknown_falls_back() {
        let mut cache = AssStyleCache::new();
        cache.register("Default", default_style());
        let r = cache.resolve("NonExistent", None);
        assert!(r.used_fallback);
        assert_eq!(r.effective_name, "Default");
    }

    #[test]
    fn test_resolve_missing_fallback_uses_default_trait() {
        let mut cache = AssStyleCache::new();
        // No styles registered at all.
        let r = cache.resolve("Anything", None);
        assert!(r.used_fallback);
    }

    #[test]
    fn test_style_override_font_size() {
        let mut cache = AssStyleCache::new();
        cache.register("Default", default_style());

        let ov = StyleOverride {
            font_size: Some(72.0),
            ..Default::default()
        };
        let r = cache.resolve("Default", Some(&ov));
        assert!((r.style.font_size - 72.0).abs() < 0.01);
    }

    #[test]
    fn test_style_override_color() {
        let mut cache = AssStyleCache::new();
        cache.register("Default", default_style());

        let red = Color::rgb(255, 0, 0);
        let ov = StyleOverride {
            primary_color: Some(red),
            ..Default::default()
        };
        let r = cache.resolve("Default", Some(&ov));
        assert_eq!(r.style.primary_color.r, 255);
        assert_eq!(r.style.primary_color.g, 0);
    }

    #[test]
    fn test_lookup_count_increments() {
        let mut cache = AssStyleCache::new();
        cache.register("Default", default_style());
        cache.resolve("Default", None);
        cache.resolve("Default", None);
        assert_eq!(cache.lookup_count(), 2);
    }

    #[test]
    fn test_fallback_count_increments() {
        let mut cache = AssStyleCache::new();
        cache.register("Default", default_style());
        cache.resolve("Missing", None);
        assert_eq!(cache.fallback_count(), 1);
    }

    #[test]
    fn test_fallback_ratio_calculation() {
        let mut cache = AssStyleCache::new();
        cache.register("Default", default_style());
        cache.resolve("Default", None); // hit
        cache.resolve("Default", None); // hit
        cache.resolve("Missing", None); // fallback
        let ratio = cache.fallback_ratio();
        assert!((ratio - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_register_all() {
        let mut cache = AssStyleCache::new();
        let styles = vec![
            ("Default".to_string(), default_style()),
            ("Title".to_string(), bold_style()),
        ];
        cache.register_all(styles);
        assert_eq!(cache.len(), 2);
        assert!(cache.contains("Title"));
    }

    #[test]
    fn test_reset_metrics() {
        let mut cache = AssStyleCache::new();
        cache.register("Default", default_style());
        cache.resolve("Default", None);
        cache.reset_metrics();
        assert_eq!(cache.lookup_count(), 0);
        assert_eq!(cache.fallback_count(), 0);
    }

    #[test]
    fn test_clear() {
        let mut cache = AssStyleCache::new();
        cache.register("Default", default_style());
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_builder_pattern() {
        let cache = StyleCacheBuilder::new()
            .add("Default", default_style())
            .add("Bold", bold_style())
            .fallback("Default")
            .build();

        assert_eq!(cache.len(), 2);
        assert!(cache.contains("Bold"));
    }

    #[test]
    fn test_remove_style() {
        let mut cache = AssStyleCache::new();
        cache.register("Default", default_style());
        let removed = cache.remove("Default");
        assert!(removed.is_some());
        assert!(!cache.contains("Default"));
    }

    #[test]
    fn test_style_names_list() {
        let mut cache = AssStyleCache::new();
        cache.register("Alpha", default_style());
        cache.register("Beta", bold_style());
        let mut names = cache.style_names();
        names.sort();
        assert_eq!(names, vec!["Alpha", "Beta"]);
    }

    #[test]
    fn test_bold_override() {
        let mut cache = AssStyleCache::new();
        cache.register("Default", default_style());

        let ov = StyleOverride {
            bold: Some(true),
            ..Default::default()
        };
        let r = cache.resolve("Default", Some(&ov));
        assert_eq!(r.style.font_weight, crate::style::FontWeight::Bold);
    }

    #[test]
    fn test_italic_override() {
        let mut cache = AssStyleCache::new();
        cache.register("Default", default_style());

        let ov = StyleOverride {
            italic: Some(true),
            ..Default::default()
        };
        let r = cache.resolve("Default", Some(&ov));
        assert_eq!(r.style.font_style, crate::style::FontStyle::Italic);
    }

    #[test]
    fn test_custom_fallback_name() {
        let mut cache = AssStyleCache::new().with_fallback("Fallback");
        cache.register("Fallback", bold_style());
        let r = cache.resolve("NonExistent", None);
        assert!(r.used_fallback);
        assert_eq!(r.effective_name, "Fallback");
        assert!((r.style.font_size - bold_style().font_size).abs() < 0.01);
    }

    #[test]
    fn test_fallback_ratio_zero_on_no_lookups() {
        let cache = AssStyleCache::new();
        assert!((cache.fallback_ratio() - 0.0).abs() < 1e-9);
    }
}
