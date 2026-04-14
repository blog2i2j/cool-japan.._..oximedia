//! Scene search integration: searching media by detected objects, scene types,
//! and visual activities.
//!
//! This module provides an in-memory [`SceneSearchIndex`] that stores per-document
//! object detection results and answers queries expressed as a
//! [`SceneSearchFilter`].  Confidence values are synthetic (derived from the
//! fraction of query objects matched) but provide a consistent ranking signal.
//!
//! # Example
//!
//! ```
//! use oximedia_search::scene_search::{SceneSearchFilter, SceneSearchIndex};
//!
//! let mut index = SceneSearchIndex::new();
//! index.add_scene("doc1".to_string(), vec!["car".to_string(), "road".to_string()]);
//! index.add_scene("doc2".to_string(), vec!["cat".to_string(), "sofa".to_string()]);
//!
//! let filter = SceneSearchFilter {
//!     detected_objects: vec!["car".to_string()],
//!     scene_types: vec![],
//!     min_confidence: 0.5,
//! };
//! let results = index.search(&filter);
//! assert_eq!(results.len(), 1);
//! assert_eq!(results[0].document_id, "doc1");
//! ```

#![allow(dead_code)]

// ─────────────────────────────────────────────────────────────────────────────
// Data types
// ─────────────────────────────────────────────────────────────────────────────

/// Filter criteria for scene-based search.
///
/// A document matches if **any** of the listed `detected_objects` or
/// `scene_types` appear in its annotation set, **and** the resulting
/// confidence score is ≥ `min_confidence`.
#[derive(Debug, Clone, Default)]
pub struct SceneSearchFilter {
    /// Object labels to match (e.g., `"car"`, `"person"`, `"tree"`).
    pub detected_objects: Vec<String>,
    /// Scene-type labels to match (e.g., `"outdoor"`, `"night"`, `"indoor"`).
    pub scene_types: Vec<String>,
    /// Minimum confidence threshold `[0.0, 1.0]`; results below this are
    /// excluded.  A value of `0.0` accepts everything.
    pub min_confidence: f32,
}

impl SceneSearchFilter {
    /// Return `true` when the filter has at least one non-empty query term.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.detected_objects.is_empty() && self.scene_types.is_empty()
    }

    /// Collect all query terms (objects + scene types) into a single Vec.
    fn all_terms(&self) -> Vec<&str> {
        self.detected_objects
            .iter()
            .chain(self.scene_types.iter())
            .map(String::as_str)
            .collect()
    }
}

/// One document entry in the result set.
#[derive(Debug, Clone)]
pub struct SceneSearchResult {
    /// Document identifier that was supplied to [`SceneSearchIndex::add_scene`].
    pub document_id: String,
    /// Subset of the filter's query terms that were found in this document.
    pub matched_objects: Vec<String>,
    /// Confidence score `[0.0, 1.0]` computed as
    /// `matched_count / total_query_terms`.
    pub confidence: f32,
}

impl SceneSearchResult {
    fn new(document_id: String, matched_objects: Vec<String>, confidence: f32) -> Self {
        Self {
            document_id,
            matched_objects,
            confidence,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Index
// ─────────────────────────────────────────────────────────────────────────────

/// In-memory index of scene annotations.
///
/// Each entry associates a document ID with a list of detected labels (objects
/// and/or scene types — the index makes no distinction).
///
/// Thread-safety: not thread-safe; wrap in a `Mutex` if shared across threads.
#[derive(Debug, Default)]
pub struct SceneSearchIndex {
    /// `(document_id, detected_labels)` pairs.
    scenes: Vec<(String, Vec<String>)>,
}

impl SceneSearchIndex {
    /// Create a new, empty index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or replace the scene annotations for `doc_id`.
    ///
    /// If `doc_id` already exists in the index its entry is **replaced** with
    /// the new `objects` list; otherwise a new entry is appended.
    pub fn add_scene(&mut self, doc_id: String, objects: Vec<String>) {
        // Check for an existing entry to update.
        for (id, labels) in &mut self.scenes {
            if id == &doc_id {
                *labels = objects;
                return;
            }
        }
        self.scenes.push((doc_id, objects));
    }

    /// Remove a document from the index by ID.
    ///
    /// Returns `true` if the document was found and removed.
    pub fn remove_scene(&mut self, doc_id: &str) -> bool {
        let before = self.scenes.len();
        self.scenes.retain(|(id, _)| id != doc_id);
        self.scenes.len() < before
    }

    /// Return the number of documents currently indexed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.scenes.len()
    }

    /// Return `true` when the index contains no documents.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.scenes.is_empty()
    }

    /// Search the index for documents matching `filter`.
    ///
    /// Confidence is defined as:
    ///
    /// ```text
    /// matched_count / total_query_terms
    /// ```
    ///
    /// where `total_query_terms` is `detected_objects.len() + scene_types.len()`.
    /// Results with `confidence < filter.min_confidence` are excluded.
    ///
    /// Results are returned sorted by confidence descending; ties are broken
    /// by document ID ascending for determinism.
    ///
    /// Returns an empty `Vec` when `filter.is_empty()`.
    #[must_use]
    pub fn search(&self, filter: &SceneSearchFilter) -> Vec<SceneSearchResult> {
        if filter.is_empty() {
            return Vec::new();
        }

        let all_terms = filter.all_terms();
        let total_terms = all_terms.len() as f32;

        let mut results: Vec<SceneSearchResult> = self
            .scenes
            .iter()
            .filter_map(|(doc_id, doc_labels)| {
                // Find which query terms appear in this document's labels.
                let matched: Vec<String> = all_terms
                    .iter()
                    .filter(|&&term| {
                        doc_labels
                            .iter()
                            .any(|label| label.eq_ignore_ascii_case(term))
                    })
                    .map(|s| s.to_string())
                    .collect();

                if matched.is_empty() {
                    return None;
                }

                let confidence = matched.len() as f32 / total_terms;
                if confidence < filter.min_confidence {
                    return None;
                }

                Some(SceneSearchResult::new(doc_id.clone(), matched, confidence))
            })
            .collect();

        // Sort by confidence descending, then document ID ascending.
        results.sort_by(|a, b| {
            b.confidence
                .total_cmp(&a.confidence)
                .then_with(|| a.document_id.cmp(&b.document_id))
        });

        results
    }

    /// Return the labels stored for a specific document, or `None` if not found.
    #[must_use]
    pub fn get_labels(&self, doc_id: &str) -> Option<&[String]> {
        self.scenes
            .iter()
            .find(|(id, _)| id == doc_id)
            .map(|(_, labels)| labels.as_slice())
    }

    /// Return all document IDs whose label set contains **all** of the supplied
    /// required labels (case-insensitive AND match).
    ///
    /// Useful for precise queries where every term must be present.
    #[must_use]
    pub fn search_all_of(&self, required: &[&str]) -> Vec<String> {
        if required.is_empty() {
            return Vec::new();
        }
        self.scenes
            .iter()
            .filter(|(_, labels)| {
                required.iter().all(|&req| {
                    labels
                        .iter()
                        .any(|label| label.eq_ignore_ascii_case(req))
                })
            })
            .map(|(id, _)| id.clone())
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_index() -> SceneSearchIndex {
        let mut idx = SceneSearchIndex::new();
        idx.add_scene("doc1".to_string(), vec!["car".to_string(), "road".to_string(), "outdoor".to_string()]);
        idx.add_scene("doc2".to_string(), vec!["cat".to_string(), "sofa".to_string(), "indoor".to_string()]);
        idx.add_scene("doc3".to_string(), vec!["car".to_string(), "person".to_string(), "outdoor".to_string()]);
        idx
    }

    // ── Basic search ──────────────────────────────────────────────────────

    #[test]
    fn test_search_single_object() {
        let idx = make_index();
        let filter = SceneSearchFilter {
            detected_objects: vec!["car".to_string()],
            scene_types: vec![],
            min_confidence: 0.0,
        };
        let res = idx.search(&filter);
        assert_eq!(res.len(), 2);
        // Both doc1 and doc3 contain "car"
        let ids: Vec<&str> = res.iter().map(|r| r.document_id.as_str()).collect();
        assert!(ids.contains(&"doc1"));
        assert!(ids.contains(&"doc3"));
    }

    #[test]
    fn test_search_no_match() {
        let idx = make_index();
        let filter = SceneSearchFilter {
            detected_objects: vec!["airplane".to_string()],
            scene_types: vec![],
            min_confidence: 0.0,
        };
        let res = idx.search(&filter);
        assert!(res.is_empty());
    }

    #[test]
    fn test_search_scene_type() {
        let idx = make_index();
        let filter = SceneSearchFilter {
            detected_objects: vec![],
            scene_types: vec!["indoor".to_string()],
            min_confidence: 0.0,
        };
        let res = idx.search(&filter);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].document_id, "doc2");
    }

    // ── Confidence ────────────────────────────────────────────────────────

    #[test]
    fn test_confidence_calculation() {
        let idx = make_index();
        // Query for 2 terms; doc1 matches only "car" (1/2 = 0.5)
        let filter = SceneSearchFilter {
            detected_objects: vec!["car".to_string(), "cat".to_string()],
            scene_types: vec![],
            min_confidence: 0.0,
        };
        let res = idx.search(&filter);
        // doc1 matches "car" (1/2), doc2 matches "cat" (1/2), doc3 matches "car" (1/2)
        for r in &res {
            assert!((r.confidence - 0.5).abs() < 1e-5, "id={} conf={}", r.document_id, r.confidence);
        }
    }

    #[test]
    fn test_min_confidence_filter() {
        let idx = make_index();
        // With 2 query terms, confidence = 0.5 for partial matches.
        // Setting min_confidence = 0.8 should exclude all partial matches.
        let filter = SceneSearchFilter {
            detected_objects: vec!["car".to_string(), "road".to_string()],
            scene_types: vec![],
            min_confidence: 0.8,
        };
        let res = idx.search(&filter);
        // Only doc1 has both "car" and "road" → confidence = 1.0
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].document_id, "doc1");
        assert!((res[0].confidence - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_empty_filter_returns_empty() {
        let idx = make_index();
        let filter = SceneSearchFilter::default();
        assert!(idx.search(&filter).is_empty());
    }

    // ── Matched objects field ─────────────────────────────────────────────

    #[test]
    fn test_matched_objects_populated() {
        let idx = make_index();
        let filter = SceneSearchFilter {
            detected_objects: vec!["car".to_string()],
            scene_types: vec![],
            min_confidence: 0.0,
        };
        let res = idx.search(&filter);
        for r in &res {
            assert!(!r.matched_objects.is_empty());
            assert!(r.matched_objects.iter().any(|m| m == "car"));
        }
    }

    // ── Add / remove ──────────────────────────────────────────────────────

    #[test]
    fn test_add_replaces_existing() {
        let mut idx = SceneSearchIndex::new();
        idx.add_scene("doc1".to_string(), vec!["car".to_string()]);
        idx.add_scene("doc1".to_string(), vec!["boat".to_string()]);
        // Only "boat" should remain
        let labels = idx.get_labels("doc1").expect("doc1 should exist");
        assert_eq!(labels, &["boat".to_string()]);
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_remove_scene() {
        let mut idx = make_index();
        assert!(idx.remove_scene("doc2"));
        assert!(!idx.remove_scene("doc2")); // already removed
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn test_is_empty() {
        let mut idx = SceneSearchIndex::new();
        assert!(idx.is_empty());
        idx.add_scene("x".to_string(), vec!["a".to_string()]);
        assert!(!idx.is_empty());
    }

    // ── Case-insensitive matching ─────────────────────────────────────────

    #[test]
    fn test_case_insensitive_match() {
        let mut idx = SceneSearchIndex::new();
        idx.add_scene("doc1".to_string(), vec!["Car".to_string()]);
        let filter = SceneSearchFilter {
            detected_objects: vec!["car".to_string()],
            scene_types: vec![],
            min_confidence: 0.0,
        };
        let res = idx.search(&filter);
        assert_eq!(res.len(), 1);
    }

    // ── search_all_of ─────────────────────────────────────────────────────

    #[test]
    fn test_search_all_of_match() {
        let idx = make_index();
        let ids = idx.search_all_of(&["car", "outdoor"]);
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"doc1".to_string()));
        assert!(ids.contains(&"doc3".to_string()));
    }

    #[test]
    fn test_search_all_of_no_match() {
        let idx = make_index();
        let ids = idx.search_all_of(&["car", "cat"]);
        assert!(ids.is_empty());
    }

    #[test]
    fn test_search_all_of_empty_required() {
        let idx = make_index();
        let ids = idx.search_all_of(&[]);
        assert!(ids.is_empty());
    }

    // ── get_labels ────────────────────────────────────────────────────────

    #[test]
    fn test_get_labels_existing() {
        let idx = make_index();
        let labels = idx.get_labels("doc1").expect("doc1 should exist");
        assert!(labels.contains(&"car".to_string()));
    }

    #[test]
    fn test_get_labels_nonexistent() {
        let idx = make_index();
        assert!(idx.get_labels("nonexistent").is_none());
    }

    // ── Sort order ────────────────────────────────────────────────────────

    #[test]
    fn test_results_sorted_by_confidence_desc() {
        let mut idx = SceneSearchIndex::new();
        // doc_a matches 2/2, doc_b matches 1/2
        idx.add_scene("doc_a".to_string(), vec!["cat".to_string(), "dog".to_string()]);
        idx.add_scene("doc_b".to_string(), vec!["cat".to_string(), "fish".to_string()]);
        let filter = SceneSearchFilter {
            detected_objects: vec!["cat".to_string(), "dog".to_string()],
            scene_types: vec![],
            min_confidence: 0.0,
        };
        let res = idx.search(&filter);
        assert!(res.len() >= 2);
        assert_eq!(res[0].document_id, "doc_a");
        assert!((res[0].confidence - 1.0).abs() < 1e-5);
        assert_eq!(res[1].document_id, "doc_b");
        assert!((res[1].confidence - 0.5).abs() < 1e-5);
    }
}
