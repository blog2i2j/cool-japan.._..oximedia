//! Preset library management: categories, search, versioning, and collection utilities.
//!
//! Provides tools for organizing presets into typed categories, performing
//! keyword search, and managing version history of preset configurations.

#![allow(dead_code)]

use std::collections::HashMap;

/// Semantic version for a preset.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PresetVersion {
    /// Major version number.
    pub major: u32,
    /// Minor version number.
    pub minor: u32,
    /// Patch version number.
    pub patch: u32,
}

impl PresetVersion {
    /// Create a new preset version.
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Parse a version string in the form "major.minor.patch".
    pub fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return None;
        }
        let major = parts[0].parse().ok()?;
        let minor = parts[1].parse().ok()?;
        let patch = parts[2].parse().ok()?;
        Some(Self {
            major,
            minor,
            patch,
        })
    }

    /// Render version as a dotted string.
    pub fn to_string_repr(&self) -> String {
        format!("{}.{}.{}", self.major, self.minor, self.patch)
    }

    /// Check if this version is compatible with another (same major version).
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.major == other.major
    }

    /// Check if this is a newer version than `other`.
    pub fn is_newer_than(&self, other: &Self) -> bool {
        self > other
    }
}

/// Category of a preset for organizational purposes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LibraryCategory {
    /// Social media platforms.
    Social,
    /// Broadcast standards.
    Broadcast,
    /// Streaming protocols.
    Streaming,
    /// Archive formats.
    Archive,
    /// User-defined categories.
    Custom(String),
}

impl LibraryCategory {
    /// Get a human-readable display name for the category.
    pub fn display_name(&self) -> String {
        match self {
            Self::Social => "Social Media".to_string(),
            Self::Broadcast => "Broadcast".to_string(),
            Self::Streaming => "Streaming".to_string(),
            Self::Archive => "Archive".to_string(),
            Self::Custom(name) => name.clone(),
        }
    }
}

/// Lightweight preset entry stored in the library catalogue.
#[derive(Debug, Clone)]
pub struct CatalogueEntry {
    /// Unique identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Short description.
    pub description: String,
    /// Category.
    pub category: LibraryCategory,
    /// Semantic version.
    pub version: PresetVersion,
    /// Searchable tags.
    pub tags: Vec<String>,
}

impl CatalogueEntry {
    /// Create a new catalogue entry.
    pub fn new(
        id: &str,
        name: &str,
        description: &str,
        category: LibraryCategory,
        version: PresetVersion,
    ) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            description: description.to_string(),
            category,
            version,
            tags: Vec::new(),
        }
    }

    /// Add a tag to the entry.
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_lowercase());
        self
    }

    /// Check if the entry matches a search query (case-insensitive).
    pub fn matches_query(&self, query: &str) -> bool {
        let q = query.to_lowercase();
        self.name.to_lowercase().contains(&q)
            || self.description.to_lowercase().contains(&q)
            || self.tags.iter().any(|t| t.contains(&q))
            || self.id.to_lowercase().contains(&q)
    }
}

/// In-memory preset catalogue for fast lookup and search.
#[derive(Debug, Default)]
pub struct PresetCatalogue {
    entries: HashMap<String, CatalogueEntry>,
}

impl PresetCatalogue {
    /// Create an empty catalogue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an entry to the catalogue.
    pub fn add(&mut self, entry: CatalogueEntry) {
        self.entries.insert(entry.id.clone(), entry);
    }

    /// Remove an entry by ID.
    pub fn remove(&mut self, id: &str) -> Option<CatalogueEntry> {
        self.entries.remove(id)
    }

    /// Get an entry by ID.
    pub fn get(&self, id: &str) -> Option<&CatalogueEntry> {
        self.entries.get(id)
    }

    /// Total number of entries in the catalogue.
    pub fn count(&self) -> usize {
        self.entries.len()
    }

    /// Search entries by a free-text query.
    pub fn search(&self, query: &str) -> Vec<&CatalogueEntry> {
        self.entries
            .values()
            .filter(|e| e.matches_query(query))
            .collect()
    }

    /// Find entries by category.
    pub fn by_category(&self, category: &LibraryCategory) -> Vec<&CatalogueEntry> {
        self.entries
            .values()
            .filter(|e| &e.category == category)
            .collect()
    }

    /// Find entries with a specific tag.
    pub fn by_tag(&self, tag: &str) -> Vec<&CatalogueEntry> {
        let tag_lower = tag.to_lowercase();
        self.entries
            .values()
            .filter(|e| e.tags.contains(&tag_lower))
            .collect()
    }

    /// Return all entries sorted by name.
    pub fn all_sorted_by_name(&self) -> Vec<&CatalogueEntry> {
        let mut entries: Vec<&CatalogueEntry> = self.entries.values().collect();
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        entries
    }

    /// Return the latest (highest) version among all entries.
    pub fn latest_version(&self) -> Option<&PresetVersion> {
        self.entries.values().map(|e| &e.version).max()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(id: &str, name: &str) -> CatalogueEntry {
        CatalogueEntry::new(
            id,
            name,
            "A test preset",
            LibraryCategory::Social,
            PresetVersion::new(1, 0, 0),
        )
    }

    #[test]
    fn test_version_parse_valid() {
        let v = PresetVersion::parse("2.3.1").expect("v should be valid");
        assert_eq!(v.major, 2);
        assert_eq!(v.minor, 3);
        assert_eq!(v.patch, 1);
    }

    #[test]
    fn test_version_parse_invalid() {
        assert!(PresetVersion::parse("1.0").is_none());
        assert!(PresetVersion::parse("abc.0.1").is_none());
        assert!(PresetVersion::parse("").is_none());
    }

    #[test]
    fn test_version_to_string() {
        let v = PresetVersion::new(1, 2, 3);
        assert_eq!(v.to_string_repr(), "1.2.3");
    }

    #[test]
    fn test_version_is_compatible() {
        let v1 = PresetVersion::new(1, 0, 0);
        let v2 = PresetVersion::new(1, 5, 3);
        let v3 = PresetVersion::new(2, 0, 0);
        assert!(v1.is_compatible_with(&v2));
        assert!(!v1.is_compatible_with(&v3));
    }

    #[test]
    fn test_version_ordering() {
        let v1 = PresetVersion::new(1, 0, 0);
        let v2 = PresetVersion::new(1, 1, 0);
        let v3 = PresetVersion::new(2, 0, 0);
        assert!(v2.is_newer_than(&v1));
        assert!(v3.is_newer_than(&v2));
        assert!(!v1.is_newer_than(&v2));
    }

    #[test]
    fn test_category_display_name() {
        assert_eq!(LibraryCategory::Social.display_name(), "Social Media");
        assert_eq!(LibraryCategory::Broadcast.display_name(), "Broadcast");
        let custom = LibraryCategory::Custom("My Cat".to_string());
        assert_eq!(custom.display_name(), "My Cat");
    }

    #[test]
    fn test_catalogue_add_and_get() {
        let mut cat = PresetCatalogue::new();
        cat.add(make_entry("id-1", "Alpha Preset"));
        assert!(cat.get("id-1").is_some());
        assert_eq!(cat.count(), 1);
    }

    #[test]
    fn test_catalogue_remove() {
        let mut cat = PresetCatalogue::new();
        cat.add(make_entry("id-1", "Alpha Preset"));
        let removed = cat.remove("id-1");
        assert!(removed.is_some());
        assert_eq!(cat.count(), 0);
    }

    #[test]
    fn test_catalogue_search_by_name() {
        let mut cat = PresetCatalogue::new();
        cat.add(make_entry("youtube-1080p", "YouTube 1080p"));
        cat.add(make_entry("vimeo-720p", "Vimeo 720p"));
        let results = cat.search("youtube");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "youtube-1080p");
    }

    #[test]
    fn test_catalogue_search_case_insensitive() {
        let mut cat = PresetCatalogue::new();
        cat.add(make_entry("yt-hd", "YouTube HD"));
        let results = cat.search("YOUTUBE");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_catalogue_by_category() {
        let mut cat = PresetCatalogue::new();
        cat.add(make_entry("s1", "Social One")); // LibraryCategory::Social
        let mut broadcast_entry = make_entry("b1", "Broadcast One");
        broadcast_entry.category = LibraryCategory::Broadcast;
        cat.add(broadcast_entry);
        let social = cat.by_category(&LibraryCategory::Social);
        assert_eq!(social.len(), 1);
    }

    #[test]
    fn test_catalogue_by_tag() {
        let mut cat = PresetCatalogue::new();
        let entry = make_entry("hls-1", "HLS 1080p")
            .with_tag("hls")
            .with_tag("streaming");
        cat.add(entry);
        let results = cat.by_tag("hls");
        assert_eq!(results.len(), 1);
        let none = cat.by_tag("rtmp");
        assert!(none.is_empty());
    }

    #[test]
    fn test_catalogue_sorted_by_name() {
        let mut cat = PresetCatalogue::new();
        cat.add(make_entry("z-id", "Zebra"));
        cat.add(make_entry("a-id", "Apple"));
        cat.add(make_entry("m-id", "Mango"));
        let sorted = cat.all_sorted_by_name();
        assert_eq!(sorted[0].name, "Apple");
        assert_eq!(sorted[1].name, "Mango");
        assert_eq!(sorted[2].name, "Zebra");
    }

    #[test]
    fn test_catalogue_latest_version() {
        let mut cat = PresetCatalogue::new();
        let mut e1 = make_entry("a", "Alpha");
        e1.version = PresetVersion::new(1, 0, 0);
        let mut e2 = make_entry("b", "Beta");
        e2.version = PresetVersion::new(2, 1, 0);
        cat.add(e1);
        cat.add(e2);
        let latest = cat.latest_version().expect("latest should be valid");
        assert_eq!(latest.major, 2);
    }

    #[test]
    fn test_catalogue_empty_latest_version() {
        let cat = PresetCatalogue::new();
        assert!(cat.latest_version().is_none());
    }
}
