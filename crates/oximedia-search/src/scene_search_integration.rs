//! Scene search integration: bridging [`SceneSearchIndex`] with the
//! unified [`crate::SearchResults`] / [`crate::SearchResultItem`] API.
//!
//! The [`SceneSearchIndex`] in [`crate::scene_search`] is a standalone
//! in-memory structure that stores per-document object/scene annotations and
//! supports query-by-label searches.  This module provides the glue layer:
//!
//! - [`SceneSearchBridge`] wraps a [`SceneSearchIndex`] and translates
//!   [`SceneSearchResult`][crate::scene_search::SceneSearchResult]s into
//!   [`SearchResultItem`][crate::SearchResultItem]s that can be merged with
//!   results from other search indices.
//!
//! - [`SceneSearchPipeline`] combines a [`SceneSearchBridge`] with a metadata
//!   store so that the emitted `SearchResultItem` records carry title,
//!   `mime_type`, `duration_ms`, and `file_path` where available.
//!
//! # Example
//!
//! ```
//! use oximedia_search::scene_search_integration::{
//!     SceneSearchBridge, SceneMetadata,
//! };
//! use oximedia_search::scene_search::SceneSearchFilter;
//!
//! let mut bridge = SceneSearchBridge::new();
//! bridge.add_scene(
//!     "doc1".to_string(),
//!     vec!["car".to_string(), "outdoor".to_string()],
//! );
//!
//! let filter = SceneSearchFilter {
//!     detected_objects: vec!["car".to_string()],
//!     scene_types: vec![],
//!     min_confidence: 0.5,
//! };
//! let items = bridge.search(&filter);
//! assert_eq!(items.len(), 1);
//! assert_eq!(items[0].file_path, "doc1");
//! assert!(items[0].score >= 0.5);
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

use crate::scene_search::{SceneSearchFilter, SceneSearchIndex};
use crate::SearchResultItem;
use uuid::Uuid;

// ─────────────────────────────────────────────────────────────────────────────
// Metadata store
// ─────────────────────────────────────────────────────────────────────────────

/// Optional metadata associated with a document in the scene search index.
///
/// Fields are stored separately from the scene annotations so that the
/// annotation index stays lightweight and focused on matching.
#[derive(Debug, Clone, Default)]
pub struct SceneMetadata {
    /// Human-readable title.
    pub title: Option<String>,
    /// MIME type (e.g. `"video/mp4"`).
    pub mime_type: Option<String>,
    /// Duration in milliseconds.
    pub duration_ms: Option<i64>,
    /// Absolute or relative file path.
    pub file_path: Option<String>,
    /// Creation timestamp (Unix seconds).
    pub created_at: i64,
    /// Optional UUID for integration with other indices.
    pub asset_id: Option<Uuid>,
}

// ─────────────────────────────────────────────────────────────────────────────
// SceneSearchBridge
// ─────────────────────────────────────────────────────────────────────────────

/// Wraps a [`SceneSearchIndex`] and translates its results into
/// [`SearchResultItem`] records compatible with the unified search API.
///
/// The `confidence` score from [`SceneSearchIndex`] maps directly to the
/// `score` field of the result item, giving relevance-ordered results.
///
/// By default, the `file_path` of each result item is set to the document ID
/// used when calling [`add_scene`][SceneSearchBridge::add_scene].  Callers can
/// attach richer metadata using [`add_metadata`][SceneSearchBridge::add_metadata].
#[derive(Debug, Default)]
pub struct SceneSearchBridge {
    index: SceneSearchIndex,
    metadata: HashMap<String, SceneMetadata>,
}

impl SceneSearchBridge {
    /// Create an empty bridge.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register scene annotations for a document.
    ///
    /// `doc_id` is used as the `file_path` in the emitted result items unless
    /// overridden by [`add_metadata`][SceneSearchBridge::add_metadata].
    pub fn add_scene(&mut self, doc_id: String, objects: Vec<String>) {
        self.index.add_scene(doc_id, objects);
    }

    /// Remove a document from the index by ID.
    ///
    /// Returns `true` if the document was found and removed.
    pub fn remove_scene(&mut self, doc_id: &str) -> bool {
        self.metadata.remove(doc_id);
        self.index.remove_scene(doc_id)
    }

    /// Attach metadata to a document.
    ///
    /// This allows the bridge to emit richer [`SearchResultItem`] records.
    /// It is not required — documents without metadata still produce valid
    /// result items.
    pub fn add_metadata(&mut self, doc_id: &str, meta: SceneMetadata) {
        self.metadata.insert(doc_id.to_string(), meta);
    }

    /// Return the number of documents in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Return `true` if the index contains no documents.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Search the index and return matching results as [`SearchResultItem`]s.
    ///
    /// Results are ordered by confidence descending, mirroring the ordering
    /// produced by [`SceneSearchIndex::search`].
    #[must_use]
    pub fn search(&self, filter: &SceneSearchFilter) -> Vec<SearchResultItem> {
        self.index
            .search(filter)
            .into_iter()
            .map(|r| {
                let meta = self.metadata.get(&r.document_id);
                let file_path = meta
                    .and_then(|m| m.file_path.as_deref())
                    .unwrap_or(&r.document_id)
                    .to_string();

                SearchResultItem {
                    asset_id: meta
                        .and_then(|m| m.asset_id)
                        .unwrap_or_else(Uuid::new_v4),
                    score: r.confidence,
                    title: meta.and_then(|m| m.title.clone()),
                    description: Some(format!(
                        "Matched scene labels: {}",
                        r.matched_objects.join(", ")
                    )),
                    file_path,
                    mime_type: meta.and_then(|m| m.mime_type.clone()),
                    duration_ms: meta.and_then(|m| m.duration_ms),
                    created_at: meta.map(|m| m.created_at).unwrap_or(0),
                    modified_at: None,
                    file_size: None,
                    matched_fields: vec!["scene".to_string()],
                    thumbnail_url: None,
                }
            })
            .collect()
    }

    /// Get the raw object labels stored for a document.
    #[must_use]
    pub fn get_labels(&self, doc_id: &str) -> Option<&[String]> {
        self.index.get_labels(doc_id)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SceneSearchPipeline
// ─────────────────────────────────────────────────────────────────────────────

/// Pipeline stage that executes a scene search and scores results relative to
/// an overall candidate set.
///
/// Unlike [`SceneSearchBridge`] (which returns raw scene results), the
/// pipeline stage is designed to be integrated into a multi-signal ranking
/// pipeline.  It accepts a list of existing candidate `asset_id`s and boosts
/// (or demotes) them based on scene annotation confidence.
#[derive(Debug, Default)]
pub struct SceneSearchPipeline {
    bridge: SceneSearchBridge,
}

impl SceneSearchPipeline {
    /// Create a new pipeline stage.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Delegate to `bridge.add_scene`.
    pub fn add_scene(&mut self, doc_id: String, objects: Vec<String>) {
        self.bridge.add_scene(doc_id, objects);
    }

    /// Delegate to `bridge.add_metadata`.
    pub fn add_metadata(&mut self, doc_id: &str, meta: SceneMetadata) {
        self.bridge.add_metadata(doc_id, meta);
    }

    /// Execute the scene filter and return merged results.
    ///
    /// The returned items carry the scene confidence as their `score`.
    /// Callers are responsible for merging / re-ranking against other signals.
    #[must_use]
    pub fn execute(&self, filter: &SceneSearchFilter) -> Vec<SearchResultItem> {
        self.bridge.search(filter)
    }

    /// Number of indexed documents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bridge.len()
    }

    /// Whether the pipeline stage has no indexed documents.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bridge.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Score blending helper
// ─────────────────────────────────────────────────────────────────────────────

/// Blend a scene confidence score into an existing relevance score.
///
/// The blend is a weighted linear combination:
///
/// ```text
/// blended = base_score * (1.0 - scene_weight) + scene_confidence * scene_weight
/// ```
///
/// `scene_weight` is clamped to `[0.0, 1.0]`.
#[must_use]
pub fn blend_scene_score(base_score: f32, scene_confidence: f32, scene_weight: f32) -> f32 {
    let w = scene_weight.clamp(0.0, 1.0);
    base_score * (1.0 - w) + scene_confidence * w
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene_search::SceneSearchFilter;

    fn make_bridge() -> SceneSearchBridge {
        let mut bridge = SceneSearchBridge::new();
        bridge.add_scene(
            "doc1".to_string(),
            vec!["car".to_string(), "road".to_string(), "outdoor".to_string()],
        );
        bridge.add_scene(
            "doc2".to_string(),
            vec!["cat".to_string(), "sofa".to_string(), "indoor".to_string()],
        );
        bridge.add_scene(
            "doc3".to_string(),
            vec!["car".to_string(), "person".to_string(), "outdoor".to_string()],
        );
        bridge
    }

    // ── Basic search ──────────────────────────────────────────────────────

    #[test]
    fn test_bridge_search_returns_items() {
        let bridge = make_bridge();
        let filter = SceneSearchFilter {
            detected_objects: vec!["car".to_string()],
            scene_types: vec![],
            min_confidence: 0.0,
        };
        let items = bridge.search(&filter);
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_bridge_search_empty_filter() {
        let bridge = make_bridge();
        let filter = SceneSearchFilter::default();
        let items = bridge.search(&filter);
        assert!(items.is_empty());
    }

    #[test]
    fn test_bridge_score_equals_confidence() {
        let bridge = make_bridge();
        // One query term; matched doc should have confidence 1.0 (1/1)
        let filter = SceneSearchFilter {
            detected_objects: vec!["cat".to_string()],
            scene_types: vec![],
            min_confidence: 0.0,
        };
        let items = bridge.search(&filter);
        assert_eq!(items.len(), 1);
        assert!((items[0].score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_bridge_matched_fields() {
        let bridge = make_bridge();
        let filter = SceneSearchFilter {
            detected_objects: vec!["car".to_string()],
            scene_types: vec![],
            min_confidence: 0.0,
        };
        let items = bridge.search(&filter);
        for item in &items {
            assert!(item.matched_fields.contains(&"scene".to_string()));
        }
    }

    #[test]
    fn test_bridge_description_contains_label() {
        let bridge = make_bridge();
        let filter = SceneSearchFilter {
            detected_objects: vec!["sofa".to_string()],
            scene_types: vec![],
            min_confidence: 0.0,
        };
        let items = bridge.search(&filter);
        assert_eq!(items.len(), 1);
        let desc = items[0].description.as_deref().unwrap_or("");
        assert!(desc.contains("sofa"), "description should mention matched label");
    }

    // ── Metadata enrichment ───────────────────────────────────────────────

    #[test]
    fn test_bridge_metadata_enriches_results() {
        let mut bridge = make_bridge();
        let meta = SceneMetadata {
            title: Some("Outdoor Car Video".to_string()),
            mime_type: Some("video/mp4".to_string()),
            duration_ms: Some(60_000),
            file_path: Some("/media/car_outdoor.mp4".to_string()),
            created_at: 1_000_000,
            asset_id: None,
        };
        bridge.add_metadata("doc1", meta);

        let filter = SceneSearchFilter {
            detected_objects: vec!["car".to_string()],
            scene_types: vec![],
            min_confidence: 1.0, // doc1 has "car" as the only query term → conf = 1.0
        };
        let items = bridge.search(&filter);
        // With min_confidence = 1.0 and single-term query, only exact matches pass.
        let doc1_item = items
            .iter()
            .find(|i| i.file_path == "/media/car_outdoor.mp4");
        assert!(doc1_item.is_some(), "doc1 should appear with enriched path");
        let item = doc1_item.expect("item should exist");
        assert_eq!(item.title.as_deref(), Some("Outdoor Car Video"));
        assert_eq!(item.mime_type.as_deref(), Some("video/mp4"));
        assert_eq!(item.duration_ms, Some(60_000));
        assert_eq!(item.created_at, 1_000_000);
    }

    #[test]
    fn test_bridge_file_path_defaults_to_doc_id() {
        let bridge = make_bridge();
        let filter = SceneSearchFilter {
            detected_objects: vec!["cat".to_string()],
            scene_types: vec![],
            min_confidence: 0.0,
        };
        let items = bridge.search(&filter);
        assert_eq!(items.len(), 1);
        // No metadata attached → file_path should equal the doc_id.
        assert_eq!(items[0].file_path, "doc2");
    }

    // ── Remove ────────────────────────────────────────────────────────────

    #[test]
    fn test_bridge_remove_scene() {
        let mut bridge = make_bridge();
        assert_eq!(bridge.len(), 3);
        assert!(bridge.remove_scene("doc2"));
        assert_eq!(bridge.len(), 2);

        let filter = SceneSearchFilter {
            detected_objects: vec!["cat".to_string()],
            scene_types: vec![],
            min_confidence: 0.0,
        };
        let items = bridge.search(&filter);
        assert!(items.is_empty(), "doc2 should have been removed");
    }

    // ── SceneSearchPipeline ───────────────────────────────────────────────

    #[test]
    fn test_pipeline_execute() {
        let mut pipeline = SceneSearchPipeline::new();
        pipeline.add_scene("clip1".to_string(), vec!["beach".to_string()]);
        pipeline.add_scene("clip2".to_string(), vec!["mountain".to_string()]);

        let filter = SceneSearchFilter {
            detected_objects: vec!["beach".to_string()],
            scene_types: vec![],
            min_confidence: 0.0,
        };
        let results = pipeline.execute(&filter);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file_path, "clip1");
    }

    #[test]
    fn test_pipeline_is_empty() {
        let pipeline = SceneSearchPipeline::new();
        assert!(pipeline.is_empty());
        assert_eq!(pipeline.len(), 0);
    }

    // ── blend_scene_score ─────────────────────────────────────────────────

    #[test]
    fn test_blend_scene_score_zero_weight() {
        // weight = 0 → result equals base_score
        let blended = blend_scene_score(0.8, 1.0, 0.0);
        assert!((blended - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_blend_scene_score_full_weight() {
        // weight = 1 → result equals scene_confidence
        let blended = blend_scene_score(0.2, 0.9, 1.0);
        assert!((blended - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_blend_scene_score_half_weight() {
        let blended = blend_scene_score(0.6, 1.0, 0.5);
        let expected = 0.6 * 0.5 + 1.0 * 0.5;
        assert!((blended - expected).abs() < 1e-6);
    }

    #[test]
    fn test_blend_scene_score_clamps_weight() {
        // weight > 1.0 should clamp to 1.0
        let blended = blend_scene_score(0.0, 0.7, 2.0);
        assert!((blended - 0.7).abs() < 1e-6, "got {blended}");

        // weight < 0.0 should clamp to 0.0
        let blended2 = blend_scene_score(0.5, 0.0, -1.0);
        assert!((blended2 - 0.5).abs() < 1e-6, "got {blended2}");
    }
}
