//! Workflow marketplace for sharing and discovering reusable workflow templates.
//!
//! The marketplace provides a local registry of shareable workflow templates
//! with metadata for search and discovery. Templates can be published,
//! searched by tags or category, and downloaded for local use.
//!
//! # Example
//!
//! ```rust
//! use oximedia_workflow::workflow_marketplace::{
//!     WorkflowMarketplace, MarketplaceEntry, TemplateCategory,
//! };
//!
//! let mut marketplace = WorkflowMarketplace::new();
//!
//! let entry = MarketplaceEntry::new(
//!     "transcode-4k",
//!     "4K Transcoding Pipeline",
//!     "Transcodes source video to 4K AV1 with quality gate",
//!     TemplateCategory::Transcoding,
//! );
//!
//! marketplace.publish(entry.clone()).unwrap();
//! let results = marketplace.search("4K");
//! assert_eq!(results.len(), 1);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Error type for marketplace operations.
#[derive(Debug, Clone)]
pub enum MarketplaceError {
    /// An entry with this ID already exists.
    DuplicateId(String),
    /// No entry found with the given ID.
    NotFound(String),
    /// Validation failed for the entry.
    ValidationFailed(String),
}

impl std::fmt::Display for MarketplaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateId(id) => write!(f, "Entry with id '{id}' already exists"),
            Self::NotFound(id) => write!(f, "Entry '{id}' not found in marketplace"),
            Self::ValidationFailed(reason) => write!(f, "Validation failed: {reason}"),
        }
    }
}

impl std::error::Error for MarketplaceError {}

/// Category of a marketplace workflow template.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TemplateCategory {
    /// Video transcoding and encoding workflows.
    Transcoding,
    /// Quality control and validation workflows.
    QualityControl,
    /// Ingest and delivery pipelines.
    IngestDelivery,
    /// Archive and backup workflows.
    Archival,
    /// Audio processing workflows.
    AudioProcessing,
    /// Subtitle and caption workflows.
    Subtitling,
    /// Analytics and reporting workflows.
    Analytics,
    /// Multi-pass or complex encoding workflows.
    MultiPassEncoding,
    /// Custom or miscellaneous workflows.
    Custom,
}

impl TemplateCategory {
    /// Human-readable display name.
    #[must_use]
    pub const fn display_name(&self) -> &'static str {
        match self {
            Self::Transcoding => "Transcoding",
            Self::QualityControl => "Quality Control",
            Self::IngestDelivery => "Ingest & Delivery",
            Self::Archival => "Archival",
            Self::AudioProcessing => "Audio Processing",
            Self::Subtitling => "Subtitling",
            Self::Analytics => "Analytics",
            Self::MultiPassEncoding => "Multi-Pass Encoding",
            Self::Custom => "Custom",
        }
    }
}

impl std::fmt::Display for TemplateCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Serialisable workflow definition stored in the marketplace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowTemplatePayload {
    /// JSON or YAML workflow definition (format determined by `content_type`).
    pub content: String,
    /// Media type: `"application/json"` or `"application/x-yaml"`.
    pub content_type: String,
    /// Schema version of the template definition.
    pub schema_version: String,
}

impl Default for WorkflowTemplatePayload {
    fn default() -> Self {
        Self {
            content: String::new(),
            content_type: "application/json".to_string(),
            schema_version: "1.0".to_string(),
        }
    }
}

/// A marketplace entry representing a single reusable workflow template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplaceEntry {
    /// Unique identifier for this entry (slug-like, e.g. "transcode-4k").
    pub id: String,
    /// Human-readable display name.
    pub name: String,
    /// Detailed description of what the workflow does.
    pub description: String,
    /// Primary category.
    pub category: TemplateCategory,
    /// Version string for the entry (semver-like, e.g. "1.0.0").
    pub version: String,
    /// Author or publisher name.
    pub author: String,
    /// Searchable tags (lowercase).
    pub tags: Vec<String>,
    /// Download / usage count.
    pub download_count: u64,
    /// Star / rating count.
    pub star_count: u64,
    /// Optional workflow template payload.
    pub payload: Option<WorkflowTemplatePayload>,
    /// ISO 8601 creation timestamp (UTC).
    pub created_at: String,
    /// ISO 8601 last-updated timestamp (UTC).
    pub updated_at: String,
}

impl MarketplaceEntry {
    /// Create a new marketplace entry with required fields.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
        category: TemplateCategory,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: description.into(),
            category,
            version: "1.0.0".to_string(),
            author: String::new(),
            tags: Vec::new(),
            download_count: 0,
            star_count: 0,
            payload: None,
            created_at: "2026-01-01T00:00:00Z".to_string(),
            updated_at: "2026-01-01T00:00:00Z".to_string(),
        }
    }

    /// Set the version (builder pattern).
    #[must_use]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Set the author (builder pattern).
    #[must_use]
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = author.into();
        self
    }

    /// Add a tag (builder pattern).
    #[must_use]
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into().to_lowercase());
        self
    }

    /// Set multiple tags (builder pattern).
    #[must_use]
    pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags = tags.into_iter().map(|t| t.into().to_lowercase()).collect();
        self
    }

    /// Attach a workflow template payload (builder pattern).
    #[must_use]
    pub fn with_payload(mut self, payload: WorkflowTemplatePayload) -> Self {
        self.payload = Some(payload);
        self
    }

    /// Validate the entry before publishing.
    ///
    /// # Errors
    /// Returns `MarketplaceError::ValidationFailed` if any required field is
    /// empty or the ID contains whitespace.
    pub fn validate(&self) -> Result<(), MarketplaceError> {
        if self.id.trim().is_empty() {
            return Err(MarketplaceError::ValidationFailed(
                "id is empty".to_string(),
            ));
        }
        if self.id.contains(char::is_whitespace) {
            return Err(MarketplaceError::ValidationFailed(
                "id must not contain whitespace".to_string(),
            ));
        }
        if self.name.trim().is_empty() {
            return Err(MarketplaceError::ValidationFailed(
                "name is empty".to_string(),
            ));
        }
        if self.description.trim().is_empty() {
            return Err(MarketplaceError::ValidationFailed(
                "description is empty".to_string(),
            ));
        }
        Ok(())
    }

    /// Whether the entry matches a search query (case-insensitive substring match
    /// against name, description, tags, and author).
    #[must_use]
    pub fn matches_query(&self, query: &str) -> bool {
        if query.trim().is_empty() {
            return true;
        }
        let q = query.to_lowercase();
        self.name.to_lowercase().contains(&q)
            || self.description.to_lowercase().contains(&q)
            || self.author.to_lowercase().contains(&q)
            || self.tags.iter().any(|t| t.contains(&q))
            || self.id.to_lowercase().contains(&q)
    }
}

/// Sort order for marketplace search results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortOrder {
    /// Sort by download count descending (most popular first).
    MostDownloaded,
    /// Sort by star count descending.
    MostStarred,
    /// Sort alphabetically by name.
    Alphabetical,
    /// Sort by creation date descending (newest first).
    Newest,
}

impl Default for SortOrder {
    fn default() -> Self {
        Self::MostDownloaded
    }
}

/// Search parameters for querying the marketplace.
#[derive(Debug, Clone, Default)]
pub struct SearchParams {
    /// Full-text query string (empty = match all).
    pub query: String,
    /// Optional category filter.
    pub category: Option<TemplateCategory>,
    /// Tags that must all be present (AND semantics).
    pub required_tags: Vec<String>,
    /// Sort order for results.
    pub sort_order: SortOrder,
    /// Maximum number of results to return (0 = unlimited).
    pub limit: usize,
}

impl SearchParams {
    /// Create a simple text-only search.
    #[must_use]
    pub fn text(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            ..Default::default()
        }
    }

    /// Create a category-filtered search.
    #[must_use]
    pub fn category(category: TemplateCategory) -> Self {
        Self {
            category: Some(category),
            ..Default::default()
        }
    }
}

/// Statistics aggregated across the marketplace.
#[derive(Debug, Clone, Default)]
pub struct MarketplaceStats {
    /// Total number of published entries.
    pub total_entries: usize,
    /// Number of entries per category.
    pub entries_per_category: HashMap<String, usize>,
    /// Total downloads across all entries.
    pub total_downloads: u64,
    /// Total stars across all entries.
    pub total_stars: u64,
}

/// Local in-memory workflow marketplace registry.
///
/// This struct stores template entries by their ID and provides full-text
/// search, category filtering, and popularity-based sorting.
#[derive(Debug, Clone, Default)]
pub struct WorkflowMarketplace {
    entries: HashMap<String, MarketplaceEntry>,
}

impl WorkflowMarketplace {
    /// Create an empty marketplace.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Publish a new template entry.
    ///
    /// # Errors
    ///
    /// Returns `MarketplaceError::DuplicateId` if an entry with the same ID
    /// already exists, or `MarketplaceError::ValidationFailed` if the entry
    /// fails validation.
    pub fn publish(&mut self, entry: MarketplaceEntry) -> Result<(), MarketplaceError> {
        entry.validate()?;
        if self.entries.contains_key(&entry.id) {
            return Err(MarketplaceError::DuplicateId(entry.id.clone()));
        }
        self.entries.insert(entry.id.clone(), entry);
        Ok(())
    }

    /// Update an existing entry (replaces it by ID).
    ///
    /// # Errors
    ///
    /// Returns `MarketplaceError::NotFound` if no entry with that ID exists,
    /// or `MarketplaceError::ValidationFailed` if the new entry is invalid.
    pub fn update(&mut self, entry: MarketplaceEntry) -> Result<(), MarketplaceError> {
        entry.validate()?;
        if !self.entries.contains_key(&entry.id) {
            return Err(MarketplaceError::NotFound(entry.id.clone()));
        }
        self.entries.insert(entry.id.clone(), entry);
        Ok(())
    }

    /// Unpublish (remove) an entry by ID.
    ///
    /// # Errors
    ///
    /// Returns `MarketplaceError::NotFound` if no entry with that ID exists.
    pub fn unpublish(&mut self, id: &str) -> Result<MarketplaceEntry, MarketplaceError> {
        self.entries
            .remove(id)
            .ok_or_else(|| MarketplaceError::NotFound(id.to_string()))
    }

    /// Get an entry by ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&MarketplaceEntry> {
        self.entries.get(id)
    }

    /// Simple text search across all entries (matches name, description, tags, author).
    ///
    /// Returns entries sorted by download count descending.
    #[must_use]
    pub fn search(&self, query: &str) -> Vec<&MarketplaceEntry> {
        let params = SearchParams {
            query: query.to_string(),
            sort_order: SortOrder::MostDownloaded,
            ..Default::default()
        };
        self.search_with_params(&params)
    }

    /// Advanced search with full `SearchParams`.
    #[must_use]
    pub fn search_with_params(&self, params: &SearchParams) -> Vec<&MarketplaceEntry> {
        let mut results: Vec<&MarketplaceEntry> = self
            .entries
            .values()
            .filter(|e| {
                // Text query
                if !e.matches_query(&params.query) {
                    return false;
                }
                // Category filter
                if let Some(cat) = params.category {
                    if e.category != cat {
                        return false;
                    }
                }
                // Required tags (all must be present)
                for tag in &params.required_tags {
                    if !e.tags.contains(&tag.to_lowercase()) {
                        return false;
                    }
                }
                true
            })
            .collect();

        // Sort
        match params.sort_order {
            SortOrder::MostDownloaded => {
                results.sort_by(|a, b| b.download_count.cmp(&a.download_count));
            }
            SortOrder::MostStarred => {
                results.sort_by(|a, b| b.star_count.cmp(&a.star_count));
            }
            SortOrder::Alphabetical => {
                results.sort_by(|a, b| a.name.cmp(&b.name));
            }
            SortOrder::Newest => {
                results.sort_by(|a, b| b.created_at.cmp(&a.created_at));
            }
        }

        // Apply limit
        if params.limit > 0 && results.len() > params.limit {
            results.truncate(params.limit);
        }

        results
    }

    /// Increment the download counter for an entry.
    ///
    /// # Errors
    ///
    /// Returns `MarketplaceError::NotFound` if no entry with that ID exists.
    pub fn record_download(&mut self, id: &str) -> Result<u64, MarketplaceError> {
        let entry = self
            .entries
            .get_mut(id)
            .ok_or_else(|| MarketplaceError::NotFound(id.to_string()))?;
        entry.download_count += 1;
        Ok(entry.download_count)
    }

    /// Increment the star counter for an entry.
    ///
    /// # Errors
    ///
    /// Returns `MarketplaceError::NotFound` if no entry with that ID exists.
    pub fn star(&mut self, id: &str) -> Result<u64, MarketplaceError> {
        let entry = self
            .entries
            .get_mut(id)
            .ok_or_else(|| MarketplaceError::NotFound(id.to_string()))?;
        entry.star_count += 1;
        Ok(entry.star_count)
    }

    /// List all entries in the marketplace.
    #[must_use]
    pub fn list_all(&self) -> Vec<&MarketplaceEntry> {
        let mut all: Vec<&MarketplaceEntry> = self.entries.values().collect();
        all.sort_by(|a, b| b.download_count.cmp(&a.download_count));
        all
    }

    /// List all entries in a specific category.
    #[must_use]
    pub fn list_by_category(&self, category: TemplateCategory) -> Vec<&MarketplaceEntry> {
        let params = SearchParams::category(category);
        self.search_with_params(&params)
    }

    /// Aggregate marketplace statistics.
    #[must_use]
    pub fn stats(&self) -> MarketplaceStats {
        let mut entries_per_category: HashMap<String, usize> = HashMap::new();
        let mut total_downloads = 0u64;
        let mut total_stars = 0u64;

        for entry in self.entries.values() {
            *entries_per_category
                .entry(entry.category.display_name().to_string())
                .or_insert(0) += 1;
            total_downloads += entry.download_count;
            total_stars += entry.star_count;
        }

        MarketplaceStats {
            total_entries: self.entries.len(),
            entries_per_category,
            total_downloads,
            total_stars,
        }
    }

    /// Total number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the marketplace is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entry(id: &str) -> MarketplaceEntry {
        MarketplaceEntry::new(
            id,
            format!("Workflow {id}"),
            format!("Description for {id}"),
            TemplateCategory::Transcoding,
        )
    }

    #[test]
    fn test_publish_and_get() {
        let mut mp = WorkflowMarketplace::new();
        let entry = sample_entry("test-1");
        mp.publish(entry.clone()).expect("should publish");
        let got = mp.get("test-1").expect("should find");
        assert_eq!(got.id, "test-1");
        assert_eq!(mp.len(), 1);
    }

    #[test]
    fn test_duplicate_publish_fails() {
        let mut mp = WorkflowMarketplace::new();
        mp.publish(sample_entry("dup")).expect("first publish");
        let result = mp.publish(sample_entry("dup"));
        assert!(matches!(result, Err(MarketplaceError::DuplicateId(_))));
    }

    #[test]
    fn test_unpublish() {
        let mut mp = WorkflowMarketplace::new();
        mp.publish(sample_entry("rm-me")).expect("should publish");
        let removed = mp.unpublish("rm-me").expect("should remove");
        assert_eq!(removed.id, "rm-me");
        assert!(mp.get("rm-me").is_none());
    }

    #[test]
    fn test_unpublish_not_found() {
        let mut mp = WorkflowMarketplace::new();
        let result = mp.unpublish("ghost");
        assert!(matches!(result, Err(MarketplaceError::NotFound(_))));
    }

    #[test]
    fn test_search_text_match() {
        let mut mp = WorkflowMarketplace::new();
        mp.publish(sample_entry("4k-encode")).expect("publish 4k");
        mp.publish(MarketplaceEntry::new(
            "qc-pipe",
            "QC Pipeline",
            "Quality check workflow",
            TemplateCategory::QualityControl,
        ))
        .expect("publish qc");
        let results = mp.search("4k");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "4k-encode");
    }

    #[test]
    fn test_search_empty_query_returns_all() {
        let mut mp = WorkflowMarketplace::new();
        mp.publish(sample_entry("a")).expect("publish a");
        mp.publish(sample_entry("b")).expect("publish b");
        let results = mp.search("");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_by_category() {
        let mut mp = WorkflowMarketplace::new();
        mp.publish(sample_entry("transcode-1"))
            .expect("publish transcode");
        mp.publish(MarketplaceEntry::new(
            "archive-1",
            "Archive",
            "Archival workflow",
            TemplateCategory::Archival,
        ))
        .expect("publish archive");

        let tc = mp.list_by_category(TemplateCategory::Transcoding);
        assert_eq!(tc.len(), 1);
        assert_eq!(tc[0].id, "transcode-1");

        let ar = mp.list_by_category(TemplateCategory::Archival);
        assert_eq!(ar.len(), 1);
    }

    #[test]
    fn test_record_download_and_star() {
        let mut mp = WorkflowMarketplace::new();
        mp.publish(sample_entry("popular")).expect("publish");
        let dl = mp.record_download("popular").expect("download");
        assert_eq!(dl, 1);
        let dl2 = mp.record_download("popular").expect("download 2");
        assert_eq!(dl2, 2);
        let stars = mp.star("popular").expect("star");
        assert_eq!(stars, 1);
    }

    #[test]
    fn test_stats() {
        let mut mp = WorkflowMarketplace::new();
        mp.publish(sample_entry("s1")).expect("publish s1");
        mp.publish(MarketplaceEntry::new(
            "a1",
            "Archive",
            "A",
            TemplateCategory::Archival,
        ))
        .expect("publish a1");
        mp.record_download("s1").expect("dl");
        mp.star("a1").expect("star");

        let stats = mp.stats();
        assert_eq!(stats.total_entries, 2);
        assert_eq!(stats.total_downloads, 1);
        assert_eq!(stats.total_stars, 1);
        assert!(stats.entries_per_category.len() >= 2);
    }

    #[test]
    fn test_search_with_limit() {
        let mut mp = WorkflowMarketplace::new();
        for i in 0..5 {
            mp.publish(sample_entry(&format!("entry-{i}")))
                .expect("publish");
        }
        let params = SearchParams {
            limit: 3,
            ..Default::default()
        };
        let results = mp.search_with_params(&params);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_entry_validation_empty_id_fails() {
        let entry = MarketplaceEntry::new("", "Name", "Desc", TemplateCategory::Custom);
        assert!(entry.validate().is_err());
    }

    #[test]
    fn test_entry_validation_whitespace_id_fails() {
        let entry = MarketplaceEntry::new("my entry", "Name", "Desc", TemplateCategory::Custom);
        assert!(entry.validate().is_err());
    }

    #[test]
    fn test_entry_matches_query_tag() {
        let entry = MarketplaceEntry::new("t1", "Test", "Desc", TemplateCategory::Custom)
            .with_tag("av1")
            .with_tag("streaming");
        assert!(entry.matches_query("av1"));
        assert!(entry.matches_query("stream"));
        assert!(!entry.matches_query("hevc"));
    }

    #[test]
    fn test_update_entry() {
        let mut mp = WorkflowMarketplace::new();
        mp.publish(sample_entry("upd-1")).expect("publish");
        let mut updated = sample_entry("upd-1");
        updated.description = "Updated description".to_string();
        mp.update(updated).expect("update should succeed");
        let got = mp.get("upd-1").expect("should exist");
        assert_eq!(got.description, "Updated description");
    }

    #[test]
    fn test_update_not_found_fails() {
        let mut mp = WorkflowMarketplace::new();
        let result = mp.update(sample_entry("ghost"));
        assert!(matches!(result, Err(MarketplaceError::NotFound(_))));
    }

    #[test]
    fn test_sort_alphabetical() {
        let mut mp = WorkflowMarketplace::new();
        mp.publish(MarketplaceEntry::new(
            "z-workflow",
            "Zebra Workflow",
            "Z desc",
            TemplateCategory::Custom,
        ))
        .expect("publish z");
        mp.publish(MarketplaceEntry::new(
            "a-workflow",
            "Apple Workflow",
            "A desc",
            TemplateCategory::Custom,
        ))
        .expect("publish a");
        let params = SearchParams {
            sort_order: SortOrder::Alphabetical,
            ..Default::default()
        };
        let results = mp.search_with_params(&params);
        assert_eq!(results[0].id, "a-workflow");
        assert_eq!(results[1].id, "z-workflow");
    }

    #[test]
    fn test_category_display_name() {
        assert_eq!(TemplateCategory::Transcoding.display_name(), "Transcoding");
        assert_eq!(
            TemplateCategory::QualityControl.display_name(),
            "Quality Control"
        );
    }
}
