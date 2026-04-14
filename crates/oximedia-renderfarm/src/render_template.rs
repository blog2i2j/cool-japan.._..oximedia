#![allow(dead_code)]
//! Reusable render configuration presets for the OxiMedia render farm.
//!
//! A [`RenderTemplate`] captures the common rendering parameters that tend to
//! be shared across many jobs (resolution, output codec, quality preset, and
//! an optional frame range).  A [`TemplateRegistry`] manages a named
//! collection of templates with full CRUD operations and JSON serialisation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── QualityPreset ────────────────────────────────────────────────────────────

/// Pre-defined quality levels that map to encoder settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityPreset {
    /// Fastest encode; suitable for internal previews.
    Draft,
    /// Good quality / reasonable file size; suitable for review.
    Preview,
    /// High-quality output suitable for broadcast / streaming.
    Production,
    /// Maximum quality; archival lossless or near-lossless.
    Archive,
}

impl QualityPreset {
    /// Returns a short, human-readable description of the preset.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::Draft => "Fast draft quality",
            Self::Preview => "Review-quality preview",
            Self::Production => "Broadcast-ready production output",
            Self::Archive => "Lossless or near-lossless archival",
        }
    }

    /// Suggested CRF / quality value (lower = better for most codecs).
    #[must_use]
    pub fn suggested_crf(&self) -> u8 {
        match self {
            Self::Draft => 35,
            Self::Preview => 23,
            Self::Production => 18,
            Self::Archive => 0,
        }
    }
}

impl std::fmt::Display for QualityPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Draft => write!(f, "Draft"),
            Self::Preview => write!(f, "Preview"),
            Self::Production => write!(f, "Production"),
            Self::Archive => write!(f, "Archive"),
        }
    }
}

// ── Resolution ───────────────────────────────────────────────────────────────

/// Output resolution (width × height in pixels).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Resolution {
    /// Pixel width.
    pub width: u32,
    /// Pixel height.
    pub height: u32,
}

impl Resolution {
    /// Create a new resolution.
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Returns `true` if both dimensions are non-zero.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.width > 0 && self.height > 0
    }

    /// Total pixel count.
    #[must_use]
    pub fn total_pixels(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Aspect ratio as width/height.
    #[must_use]
    pub fn aspect_ratio(&self) -> f64 {
        if self.height == 0 {
            return 0.0;
        }
        f64::from(self.width) / f64::from(self.height)
    }

    // Common presets
    /// 1920 × 1080 (Full HD).
    pub const HD_1080P: Self = Self::new(1920, 1080);
    /// 3840 × 2160 (4K UHD).
    pub const UHD_4K: Self = Self::new(3840, 2160);
    /// 7680 × 4320 (8K UHD).
    pub const UHD_8K: Self = Self::new(7680, 4320);
    /// 2048 × 1080 (DCI 2K).
    pub const DCI_2K: Self = Self::new(2048, 1080);
    /// 4096 × 2160 (DCI 4K).
    pub const DCI_4K: Self = Self::new(4096, 2160);
}

impl std::fmt::Display for Resolution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}×{}", self.width, self.height)
    }
}

// ── RenderTemplate ───────────────────────────────────────────────────────────

/// A reusable configuration preset for render jobs.
///
/// Templates capture the stable, per-project or per-department settings so
/// they can be applied to many jobs without repeating the same configuration
/// each time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderTemplate {
    /// Unique human-readable name (must be non-empty).
    pub name: String,
    /// Optional description shown in UIs and reports.
    pub description: Option<String>,
    /// Output frame resolution.
    pub resolution: Resolution,
    /// Output codec identifier (e.g. `"h264"`, `"av1"`, `"exr"`).
    pub codec: String,
    /// Quality trade-off preset.
    pub quality_preset: QualityPreset,
    /// Optional frame range `(start, end)` both inclusive.
    /// `None` means the job supplies its own frame range.
    pub frame_range: Option<(u64, u64)>,
    /// Additional key-value metadata (e.g. colour space, HDR flags).
    pub metadata: HashMap<String, String>,
}

impl RenderTemplate {
    /// Create a minimal template with required fields.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        resolution: Resolution,
        codec: impl Into<String>,
        quality_preset: QualityPreset,
    ) -> Self {
        Self {
            name: name.into(),
            description: None,
            resolution,
            codec: codec.into(),
            quality_preset,
            frame_range: None,
            metadata: HashMap::new(),
        }
    }

    /// Attach a description and return `self` (builder).
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set a specific frame range and return `self` (builder).
    #[must_use]
    pub fn with_frame_range(mut self, start: u64, end: u64) -> Self {
        self.frame_range = Some((start, end));
        self
    }

    /// Insert a metadata key-value pair and return `self` (builder).
    #[must_use]
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Returns `true` if the template's fields are internally consistent:
    /// - Non-empty name
    /// - Valid resolution
    /// - Non-empty codec
    /// - If a frame range is set, start ≤ end
    #[must_use]
    pub fn is_valid(&self) -> bool {
        if self.name.is_empty() || self.codec.is_empty() {
            return false;
        }
        if !self.resolution.is_valid() {
            return false;
        }
        if let Some((start, end)) = self.frame_range {
            if start > end {
                return false;
            }
        }
        true
    }

    /// Number of frames in the template's frame range, if set.
    #[must_use]
    pub fn frame_count(&self) -> Option<u64> {
        self.frame_range.map(|(s, e)| e.saturating_sub(s) + 1)
    }
}

// ── TemplateRegistryError ────────────────────────────────────────────────────

/// Errors returned by [`TemplateRegistry`] operations.
#[derive(Debug, thiserror::Error)]
pub enum TemplateError {
    /// A template with that name already exists.
    #[error("Template already exists: {0}")]
    AlreadyExists(String),

    /// No template found with that name.
    #[error("Template not found: {0}")]
    NotFound(String),

    /// The template failed validation.
    #[error("Invalid template '{name}': {reason}")]
    Invalid {
        /// Template name.
        name: String,
        /// Reason for invalidity.
        reason: String,
    },

    /// JSON serialisation / deserialisation error.
    #[error("Serialisation error: {0}")]
    Serialisation(#[from] serde_json::Error),
}

// ── TemplateRegistry ─────────────────────────────────────────────────────────

/// A named collection of [`RenderTemplate`] presets with CRUD operations.
///
/// Templates are stored in memory and can be exported/imported as JSON.
#[derive(Debug, Default)]
pub struct TemplateRegistry {
    templates: HashMap<String, RenderTemplate>,
}

impl TemplateRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new template.
    ///
    /// Fails if a template with the same name already exists or if the
    /// template fails validation.
    pub fn register(&mut self, template: RenderTemplate) -> Result<(), TemplateError> {
        if !template.is_valid() {
            return Err(TemplateError::Invalid {
                name: template.name.clone(),
                reason: "name/codec empty, invalid resolution, or bad frame range".to_string(),
            });
        }
        if self.templates.contains_key(&template.name) {
            return Err(TemplateError::AlreadyExists(template.name));
        }
        self.templates.insert(template.name.clone(), template);
        Ok(())
    }

    /// Look up a template by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&RenderTemplate> {
        self.templates.get(name)
    }

    /// Replace an existing template.
    ///
    /// Fails if no template with that name exists or the replacement is invalid.
    pub fn update(&mut self, template: RenderTemplate) -> Result<(), TemplateError> {
        if !template.is_valid() {
            return Err(TemplateError::Invalid {
                name: template.name.clone(),
                reason: "name/codec empty, invalid resolution, or bad frame range".to_string(),
            });
        }
        if !self.templates.contains_key(&template.name) {
            return Err(TemplateError::NotFound(template.name));
        }
        self.templates.insert(template.name.clone(), template);
        Ok(())
    }

    /// Remove a template by name.  Returns the removed template, or an error
    /// if no such template exists.
    pub fn remove(&mut self, name: &str) -> Result<RenderTemplate, TemplateError> {
        self.templates
            .remove(name)
            .ok_or_else(|| TemplateError::NotFound(name.to_string()))
    }

    /// Number of templates currently registered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.templates.len()
    }

    /// Returns `true` when no templates are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }

    /// Iterate over all registered templates.
    pub fn iter(&self) -> impl Iterator<Item = &RenderTemplate> {
        self.templates.values()
    }

    /// Serialise all templates to a JSON string.
    pub fn to_json(&self) -> Result<String, TemplateError> {
        let list: Vec<&RenderTemplate> = self.templates.values().collect();
        Ok(serde_json::to_string_pretty(&list)?)
    }

    /// Populate a registry from a JSON string produced by [`Self::to_json`].
    ///
    /// Existing entries are preserved; duplicates in the JSON are ignored.
    pub fn from_json(&mut self, json: &str) -> Result<usize, TemplateError> {
        let templates: Vec<RenderTemplate> = serde_json::from_str(json)?;
        let mut count = 0;
        for t in templates {
            if !self.templates.contains_key(&t.name) && t.is_valid() {
                self.templates.insert(t.name.clone(), t);
                count += 1;
            }
        }
        Ok(count)
    }

    /// Find all templates that match a given quality preset.
    #[must_use]
    pub fn by_quality(&self, preset: QualityPreset) -> Vec<&RenderTemplate> {
        self.templates
            .values()
            .filter(|t| t.quality_preset == preset)
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn preview_template() -> RenderTemplate {
        RenderTemplate::new(
            "1080p-preview",
            Resolution::HD_1080P,
            "h264",
            QualityPreset::Preview,
        )
    }

    // ── QualityPreset ────────────────────────────────────────────────────

    #[test]
    fn test_quality_preset_description_non_empty() {
        for preset in [
            QualityPreset::Draft,
            QualityPreset::Preview,
            QualityPreset::Production,
            QualityPreset::Archive,
        ] {
            assert!(!preset.description().is_empty());
        }
    }

    #[test]
    fn test_quality_preset_crf_ordering() {
        assert!(QualityPreset::Draft.suggested_crf() > QualityPreset::Production.suggested_crf());
        assert_eq!(QualityPreset::Archive.suggested_crf(), 0);
    }

    #[test]
    fn test_quality_preset_display() {
        assert_eq!(QualityPreset::Production.to_string(), "Production");
    }

    // ── Resolution ──────────────────────────────────────────────────────

    #[test]
    fn test_resolution_is_valid() {
        assert!(Resolution::HD_1080P.is_valid());
        assert!(!Resolution::new(0, 1080).is_valid());
        assert!(!Resolution::new(1920, 0).is_valid());
    }

    #[test]
    fn test_resolution_total_pixels() {
        let r = Resolution::HD_1080P;
        assert_eq!(r.total_pixels(), 1920 * 1080);
    }

    #[test]
    fn test_resolution_aspect_ratio() {
        let ar = Resolution::HD_1080P.aspect_ratio();
        assert!((ar - 16.0 / 9.0).abs() < 0.01);
    }

    #[test]
    fn test_resolution_display() {
        assert_eq!(Resolution::HD_1080P.to_string(), "1920×1080");
    }

    // ── RenderTemplate ───────────────────────────────────────────────────

    #[test]
    fn test_template_valid() {
        assert!(preview_template().is_valid());
    }

    #[test]
    fn test_template_invalid_empty_name() {
        let t = RenderTemplate::new("", Resolution::HD_1080P, "h264", QualityPreset::Draft);
        assert!(!t.is_valid());
    }

    #[test]
    fn test_template_invalid_empty_codec() {
        let t = RenderTemplate::new("name", Resolution::HD_1080P, "", QualityPreset::Draft);
        assert!(!t.is_valid());
    }

    #[test]
    fn test_template_invalid_frame_range() {
        let t = RenderTemplate::new("t", Resolution::HD_1080P, "av1", QualityPreset::Draft)
            .with_frame_range(100, 50); // start > end
        assert!(!t.is_valid());
    }

    #[test]
    fn test_template_frame_count() {
        let t = preview_template().with_frame_range(1, 100);
        assert_eq!(t.frame_count(), Some(100));
    }

    #[test]
    fn test_template_frame_count_none() {
        assert_eq!(preview_template().frame_count(), None);
    }

    #[test]
    fn test_template_with_meta() {
        let t = preview_template().with_meta("colorspace", "rec709");
        assert_eq!(
            t.metadata.get("colorspace").map(String::as_str),
            Some("rec709")
        );
    }

    // ── TemplateRegistry ─────────────────────────────────────────────────

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = TemplateRegistry::new();
        reg.register(preview_template()).expect("should register");
        assert!(reg.get("1080p-preview").is_some());
    }

    #[test]
    fn test_registry_register_duplicate_fails() {
        let mut reg = TemplateRegistry::new();
        reg.register(preview_template()).expect("first ok");
        let err = reg.register(preview_template());
        assert!(matches!(err, Err(TemplateError::AlreadyExists(_))));
    }

    #[test]
    fn test_registry_register_invalid_fails() {
        let mut reg = TemplateRegistry::new();
        let invalid = RenderTemplate::new("", Resolution::HD_1080P, "h264", QualityPreset::Draft);
        assert!(matches!(
            reg.register(invalid),
            Err(TemplateError::Invalid { .. })
        ));
    }

    #[test]
    fn test_registry_update() {
        let mut reg = TemplateRegistry::new();
        reg.register(preview_template()).expect("ok");
        let updated = RenderTemplate::new(
            "1080p-preview",
            Resolution::UHD_4K,
            "h265",
            QualityPreset::Production,
        );
        reg.update(updated).expect("update ok");
        let t = reg.get("1080p-preview").expect("should exist");
        assert_eq!(t.codec, "h265");
    }

    #[test]
    fn test_registry_update_nonexistent_fails() {
        let mut reg = TemplateRegistry::new();
        let err = reg.update(preview_template());
        assert!(matches!(err, Err(TemplateError::NotFound(_))));
    }

    #[test]
    fn test_registry_remove() {
        let mut reg = TemplateRegistry::new();
        reg.register(preview_template()).expect("ok");
        let removed = reg.remove("1080p-preview").expect("should remove");
        assert_eq!(removed.name, "1080p-preview");
        assert!(reg.is_empty());
    }

    #[test]
    fn test_registry_remove_nonexistent_fails() {
        let mut reg = TemplateRegistry::new();
        assert!(matches!(
            reg.remove("ghost"),
            Err(TemplateError::NotFound(_))
        ));
    }

    #[test]
    fn test_registry_len_and_is_empty() {
        let mut reg = TemplateRegistry::new();
        assert!(reg.is_empty());
        reg.register(preview_template()).expect("ok");
        assert_eq!(reg.len(), 1);
        assert!(!reg.is_empty());
    }

    #[test]
    fn test_registry_json_roundtrip() {
        let mut reg = TemplateRegistry::new();
        reg.register(preview_template()).expect("ok");
        let json = reg.to_json().expect("serialise");
        let mut reg2 = TemplateRegistry::new();
        let count = reg2.from_json(&json).expect("deserialise");
        assert_eq!(count, 1);
        assert!(reg2.get("1080p-preview").is_some());
    }

    #[test]
    fn test_registry_by_quality() {
        let mut reg = TemplateRegistry::new();
        reg.register(preview_template()).expect("ok");
        reg.register(RenderTemplate::new(
            "4k-prod",
            Resolution::UHD_4K,
            "av1",
            QualityPreset::Production,
        ))
        .expect("ok");
        let previews = reg.by_quality(QualityPreset::Preview);
        assert_eq!(previews.len(), 1);
        assert_eq!(previews[0].name, "1080p-preview");
    }
}
