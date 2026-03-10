//! Preset manager providing category-based organization and search.
//!
//! The `PresetManager` acts as the primary entry point for working with
//! encoding presets. It wraps the flat `PresetLibrary` and adds support for
//! category-based filtering, ranked search, and user-defined overrides.

#![allow(dead_code)]

use std::collections::HashMap;

/// Broad category used to organize presets in the manager.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ManagedPresetCategory {
    /// Official platform-specific presets (YouTube, Vimeo, etc.).
    Platform,
    /// Broadcast standard presets (ATSC, DVB, ISDB).
    Broadcast,
    /// Adaptive bitrate streaming ladders (HLS, DASH).
    Streaming,
    /// Lossless and mezzanine archive presets.
    Archive,
    /// Mobile device delivery presets.
    Mobile,
    /// Web / HTML5 delivery presets.
    Web,
    /// Social-media short-form presets.
    Social,
    /// User-defined custom presets.
    Custom,
}

impl std::fmt::Display for ManagedPresetCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Platform => "Platform",
            Self::Broadcast => "Broadcast",
            Self::Streaming => "Streaming",
            Self::Archive => "Archive",
            Self::Mobile => "Mobile",
            Self::Web => "Web",
            Self::Social => "Social",
            Self::Custom => "Custom",
        };
        write!(f, "{}", s)
    }
}

/// A managed preset entry.
#[derive(Debug, Clone)]
pub struct ManagedPreset {
    /// Unique identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Short description.
    pub description: String,
    /// Category this preset belongs to.
    pub category: ManagedPresetCategory,
    /// Arbitrary key-value properties (e.g. `"resolution" => "1920x1080"`).
    pub properties: HashMap<String, String>,
    /// Whether this is a built-in (official) preset.
    pub builtin: bool,
}

impl ManagedPreset {
    /// Create a new managed preset.
    #[must_use]
    pub fn new(id: &str, name: &str, category: ManagedPresetCategory) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            description: String::new(),
            category,
            properties: HashMap::new(),
            builtin: false,
        }
    }

    /// Builder: set the description.
    #[must_use]
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Builder: mark as builtin.
    #[must_use]
    pub fn as_builtin(mut self) -> Self {
        self.builtin = true;
        self
    }

    /// Builder: insert a property.
    #[must_use]
    pub fn with_property(mut self, key: &str, value: &str) -> Self {
        self.properties.insert(key.to_string(), value.to_string());
        self
    }

    /// Retrieve a property value.
    #[must_use]
    pub fn property(&self, key: &str) -> Option<&str> {
        self.properties.get(key).map(String::as_str)
    }
}

/// Manages a collection of `ManagedPreset` entries with search and filter support.
#[derive(Debug, Default)]
pub struct PresetManager {
    presets: HashMap<String, ManagedPreset>,
}

impl PresetManager {
    /// Create an empty preset manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or replace a preset.
    pub fn insert(&mut self, preset: ManagedPreset) {
        self.presets.insert(preset.id.clone(), preset);
    }

    /// Remove a preset by ID. Returns the removed preset, if it existed.
    pub fn remove(&mut self, id: &str) -> Option<ManagedPreset> {
        self.presets.remove(id)
    }

    /// Retrieve a preset by its unique ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&ManagedPreset> {
        self.presets.get(id)
    }

    /// Full-text search across `id`, `name`, and `description` (case-insensitive).
    ///
    /// Results are returned in ascending alphabetical order by name.
    #[must_use]
    pub fn search(&self, query: &str) -> Vec<&ManagedPreset> {
        let q = query.to_lowercase();
        let mut results: Vec<&ManagedPreset> = self
            .presets
            .values()
            .filter(|p| {
                p.id.to_lowercase().contains(&q)
                    || p.name.to_lowercase().contains(&q)
                    || p.description.to_lowercase().contains(&q)
            })
            .collect();
        results.sort_by(|a, b| a.name.cmp(&b.name));
        results
    }

    /// Filter presets by category.
    #[must_use]
    pub fn by_category(&self, category: &ManagedPresetCategory) -> Vec<&ManagedPreset> {
        self.presets
            .values()
            .filter(|p| &p.category == category)
            .collect()
    }

    /// Return only user-defined (non-builtin) presets.
    #[must_use]
    pub fn user_presets(&self) -> Vec<&ManagedPreset> {
        self.presets.values().filter(|p| !p.builtin).collect()
    }

    /// Return only builtin presets.
    #[must_use]
    pub fn builtin_presets(&self) -> Vec<&ManagedPreset> {
        self.presets.values().filter(|p| p.builtin).collect()
    }

    /// Number of presets in the manager.
    #[must_use]
    pub fn count(&self) -> usize {
        self.presets.len()
    }

    /// Returns `true` if a preset with the given ID exists.
    #[must_use]
    pub fn contains(&self, id: &str) -> bool {
        self.presets.contains_key(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager() -> PresetManager {
        let mut m = PresetManager::new();
        m.insert(
            ManagedPreset::new("yt-1080p", "YouTube 1080p", ManagedPresetCategory::Platform)
                .with_description("Full HD YouTube upload")
                .as_builtin()
                .with_property("resolution", "1920x1080"),
        );
        m.insert(
            ManagedPreset::new("yt-4k", "YouTube 4K", ManagedPresetCategory::Platform)
                .with_description("4K UHD YouTube upload")
                .as_builtin()
                .with_property("resolution", "3840x2160"),
        );
        m.insert(
            ManagedPreset::new("hls-720p", "HLS 720p", ManagedPresetCategory::Streaming)
                .with_description("HLS ABR 720p rung")
                .as_builtin(),
        );
        m.insert(
            ManagedPreset::new(
                "my-custom",
                "My Custom Preset",
                ManagedPresetCategory::Custom,
            )
            .with_description("User preset"),
        );
        m
    }

    #[test]
    fn test_empty_manager() {
        let m = PresetManager::new();
        assert_eq!(m.count(), 0);
    }

    #[test]
    fn test_insert_and_count() {
        let m = make_manager();
        assert_eq!(m.count(), 4);
    }

    #[test]
    fn test_get_by_id() {
        let m = make_manager();
        let p = m.get("yt-1080p").expect("p should be valid");
        assert_eq!(p.name, "YouTube 1080p");
    }

    #[test]
    fn test_contains() {
        let m = make_manager();
        assert!(m.contains("yt-1080p"));
        assert!(!m.contains("nonexistent"));
    }

    #[test]
    fn test_search_by_name() {
        let m = make_manager();
        let results = m.search("youtube");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_by_description() {
        let m = make_manager();
        let results = m.search("ABR");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "hls-720p");
    }

    #[test]
    fn test_search_empty_query_matches_all() {
        let m = make_manager();
        let results = m.search("");
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_search_no_match_returns_empty() {
        let m = make_manager();
        let results = m.search("zzznomatch");
        assert!(results.is_empty());
    }

    #[test]
    fn test_by_category() {
        let m = make_manager();
        let platform = m.by_category(&ManagedPresetCategory::Platform);
        assert_eq!(platform.len(), 2);
        let streaming = m.by_category(&ManagedPresetCategory::Streaming);
        assert_eq!(streaming.len(), 1);
    }

    #[test]
    fn test_builtin_presets() {
        let m = make_manager();
        let builtins = m.builtin_presets();
        assert_eq!(builtins.len(), 3);
    }

    #[test]
    fn test_user_presets() {
        let m = make_manager();
        let user = m.user_presets();
        assert_eq!(user.len(), 1);
        assert_eq!(user[0].id, "my-custom");
    }

    #[test]
    fn test_remove_preset() {
        let mut m = make_manager();
        let removed = m.remove("yt-1080p");
        assert!(removed.is_some());
        assert_eq!(m.count(), 3);
        assert!(!m.contains("yt-1080p"));
    }

    #[test]
    fn test_property_access() {
        let m = make_manager();
        let p = m.get("yt-1080p").expect("p should be valid");
        assert_eq!(p.property("resolution"), Some("1920x1080"));
        assert_eq!(p.property("nonexistent"), None);
    }

    #[test]
    fn test_category_display() {
        assert_eq!(ManagedPresetCategory::Platform.to_string(), "Platform");
        assert_eq!(ManagedPresetCategory::Streaming.to_string(), "Streaming");
        assert_eq!(ManagedPresetCategory::Custom.to_string(), "Custom");
    }
}
