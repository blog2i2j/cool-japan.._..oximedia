//! Multi-value faceted field aggregation.
//!
//! Standard facet aggregation assumes each document carries a **single** value
//! per facet dimension (e.g. one MIME type, one duration bucket).  Real-world
//! media assets often have *multiple* values on a single dimension:
//!
//! - A video asset can have **multiple tags** (e.g. `["sport", "outdoor", "4k"]`).
//! - An audio clip can target **multiple audiences** (e.g. `["music", "podcast"]`).
//! - A document can belong to **multiple categories** simultaneously.
//!
//! This module provides [`MultiValueFacetIndex`] for accumulating documents
//! that carry one or more values per facet field, and computing correct facet
//! counts where each document is counted **once per unique value** it carries
//! (not once per occurrence).
//!
//! # Algorithm
//!
//! For each field, we maintain a `HashMap<field_value, HashSet<doc_id>>`.
//! When counting, `|HashSet<doc_id>|` gives the correct distinct-document count
//! for that bucket, naturally handling duplicates.
//!
//! # Example
//!
//! ```rust
//! use oximedia_search::facet_multi_value::{MultiValueFacetIndex, FacetField};
//!
//! let mut index = MultiValueFacetIndex::new();
//! index.add_document("doc-1", FacetField::Tags, &["sport", "outdoor"]);
//! index.add_document("doc-2", FacetField::Tags, &["sport", "music"]);
//!
//! let counts = index.counts(FacetField::Tags);
//! // "sport" appears in both docs → count = 2.
//! let sport = counts.iter().find(|c| c.value == "sport").expect("found");
//! assert_eq!(sport.count, 2);
//! ```

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

// ─────────────────────────────────────────────────────────────────────────────
// FacetField enum
// ─────────────────────────────────────────────────────────────────────────────

/// Well-known facet dimensions that support multi-value assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FacetField {
    /// Free-form tags (e.g. `["sport", "outdoor", "documentary"]`).
    Tags,
    /// Codec names (a container may have both video AND audio codecs).
    Codecs,
    /// Target audiences / use cases.
    Audiences,
    /// Genre classifications.
    Genres,
    /// Languages (multilingual assets have multiple).
    Languages,
    /// Custom user-defined facet field.
    Custom(&'static str),
}

impl FacetField {
    /// Return a human-readable label for the field.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Tags => "tags",
            Self::Codecs => "codecs",
            Self::Audiences => "audiences",
            Self::Genres => "genres",
            Self::Languages => "languages",
            Self::Custom(name) => name,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FacetCount
// ─────────────────────────────────────────────────────────────────────────────

/// A single facet bucket with its distinct-document count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiFacetCount {
    /// Facet value (bucket label).
    pub value: String,
    /// Number of **distinct** documents carrying this value.
    pub count: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// MultiValueFacetIndex
// ─────────────────────────────────────────────────────────────────────────────

/// Index that accumulates multi-value facet data for a collection of documents.
///
/// Internally maps `FacetField → value → {doc_id}` so that counting is always
/// distinct-document based regardless of how many times a document carries the
/// same value.
#[derive(Debug, Default)]
pub struct MultiValueFacetIndex {
    /// field → value → set of document ids
    data: HashMap<String, HashMap<String, HashSet<String>>>,
    /// Total number of distinct documents indexed.
    doc_count: usize,
    /// Set of all known document ids (for computing complement / negation).
    all_docs: HashSet<String>,
}

impl MultiValueFacetIndex {
    /// Create a new, empty index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a document with one or more values for the given facet field.
    ///
    /// Calling this multiple times for the same `(doc_id, field, value)` triple
    /// is idempotent — the document will be counted only once per value.
    pub fn add_document(&mut self, doc_id: &str, field: FacetField, values: &[&str]) {
        let is_new = self.all_docs.insert(doc_id.to_string());
        if is_new {
            self.doc_count += 1;
        }
        let field_map = self.data.entry(field.label().to_string()).or_default();
        for &value in values {
            field_map
                .entry(value.to_string())
                .or_default()
                .insert(doc_id.to_string());
        }
    }

    /// Register a document with a single value for the given facet field.
    pub fn add_single(&mut self, doc_id: &str, field: FacetField, value: &str) {
        self.add_document(doc_id, field, &[value]);
    }

    /// Return sorted facet counts for the given field.
    ///
    /// Results are sorted by descending count, then ascending value (for
    /// stable ordering of ties).
    #[must_use]
    pub fn counts(&self, field: FacetField) -> Vec<MultiFacetCount> {
        let Some(field_map) = self.data.get(field.label()) else {
            return Vec::new();
        };
        let mut counts: Vec<MultiFacetCount> = field_map
            .iter()
            .map(|(value, docs)| MultiFacetCount {
                value: value.clone(),
                count: docs.len(),
            })
            .collect();
        counts.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.value.cmp(&b.value)));
        counts
    }

    /// Return the count for a single specific value within a field.
    #[must_use]
    pub fn count_for_value(&self, field: FacetField, value: &str) -> usize {
        self.data
            .get(field.label())
            .and_then(|m| m.get(value))
            .map(HashSet::len)
            .unwrap_or(0)
    }

    /// Return the total number of distinct documents in the index.
    #[must_use]
    pub fn doc_count(&self) -> usize {
        self.doc_count
    }

    /// Return the document IDs that carry `value` for `field`.
    #[must_use]
    pub fn docs_with_value(&self, field: FacetField, value: &str) -> Vec<String> {
        let mut ids: Vec<String> = self
            .data
            .get(field.label())
            .and_then(|m| m.get(value))
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default();
        ids.sort();
        ids
    }

    /// Return all unique values for the given field.
    #[must_use]
    pub fn values_for_field(&self, field: FacetField) -> Vec<String> {
        let Some(field_map) = self.data.get(field.label()) else {
            return Vec::new();
        };
        let mut values: Vec<String> = field_map.keys().cloned().collect();
        values.sort();
        values
    }

    /// Return the intersection: doc IDs that carry **all** of the given values
    /// for the given field (AND filter across values).
    ///
    /// This is the standard multi-select "AND" facet behaviour.
    #[must_use]
    pub fn docs_with_all_values(&self, field: FacetField, values: &[&str]) -> Vec<String> {
        if values.is_empty() {
            return self.all_docs.iter().cloned().collect();
        }
        let Some(field_map) = self.data.get(field.label()) else {
            return Vec::new();
        };

        // Start with the first value's document set and intersect.
        let first_set = match field_map.get(values[0]) {
            Some(s) => s.clone(),
            None => return Vec::new(),
        };

        let intersection: HashSet<String> = values[1..]
            .iter()
            .fold(first_set, |acc, &val| {
                if let Some(s) = field_map.get(val) {
                    acc.intersection(s).cloned().collect()
                } else {
                    HashSet::new()
                }
            });

        let mut ids: Vec<String> = intersection.into_iter().collect();
        ids.sort();
        ids
    }

    /// Return the union: doc IDs that carry **any** of the given values for the
    /// given field (OR filter across values).
    #[must_use]
    pub fn docs_with_any_value(&self, field: FacetField, values: &[&str]) -> Vec<String> {
        let Some(field_map) = self.data.get(field.label()) else {
            return Vec::new();
        };

        let union: HashSet<&String> = values
            .iter()
            .filter_map(|&val| field_map.get(val))
            .flat_map(HashSet::iter)
            .collect();

        let mut ids: Vec<String> = union.into_iter().cloned().collect();
        ids.sort();
        ids
    }

    /// Merge another `MultiValueFacetIndex` into this one.
    ///
    /// Document counts are unioned; value bucket counts are re-computed from
    /// the merged document sets.
    pub fn merge(&mut self, other: &MultiValueFacetIndex) {
        // Merge all_docs
        for doc_id in &other.all_docs {
            if self.all_docs.insert(doc_id.clone()) {
                self.doc_count += 1;
            }
        }
        // Merge field data
        for (field_label, other_field_map) in &other.data {
            let self_field_map = self.data.entry(field_label.clone()).or_default();
            for (value, other_doc_set) in other_field_map {
                self_field_map
                    .entry(value.clone())
                    .or_default()
                    .extend(other_doc_set.iter().cloned());
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Facet filter helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Filter a result set to only those document IDs that pass a multi-value facet
/// constraint.
///
/// - `mode = And`: doc must carry **all** of `required_values`.
/// - `mode = Or`: doc must carry **at least one** of `required_values`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FacetFilterMode {
    /// Logical AND — all values must be present.
    And,
    /// Logical OR — at least one value must be present.
    Or,
}

/// Apply a facet filter to a list of document IDs.
///
/// Returns only those document IDs that satisfy the constraint.
#[must_use]
pub fn apply_facet_filter(
    index: &MultiValueFacetIndex,
    field: FacetField,
    required_values: &[&str],
    mode: FacetFilterMode,
    candidates: &[String],
) -> Vec<String> {
    let matching: HashSet<String> = match mode {
        FacetFilterMode::And => index
            .docs_with_all_values(field, required_values)
            .into_iter()
            .collect(),
        FacetFilterMode::Or => index
            .docs_with_any_value(field, required_values)
            .into_iter()
            .collect(),
    };

    candidates
        .iter()
        .filter(|id| matching.contains(*id))
        .cloned()
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_index() -> MultiValueFacetIndex {
        let mut idx = MultiValueFacetIndex::new();
        idx.add_document("doc-1", FacetField::Tags, &["sport", "outdoor", "4k"]);
        idx.add_document("doc-2", FacetField::Tags, &["sport", "music"]);
        idx.add_document("doc-3", FacetField::Tags, &["music", "documentary"]);
        idx.add_document("doc-4", FacetField::Tags, &["outdoor"]);
        idx
    }

    #[test]
    fn test_count_for_shared_value() {
        let idx = build_index();
        // "sport" is in doc-1 and doc-2.
        assert_eq!(idx.count_for_value(FacetField::Tags, "sport"), 2);
    }

    #[test]
    fn test_count_for_unique_value() {
        let idx = build_index();
        assert_eq!(idx.count_for_value(FacetField::Tags, "documentary"), 1);
    }

    #[test]
    fn test_counts_sorted_descending() {
        let idx = build_index();
        let counts = idx.counts(FacetField::Tags);
        // "music" and "outdoor" and "sport" each appear in 2 docs;
        // "4k" and "documentary" appear in 1 each.
        // Top entries should all have count >= next entry.
        for pair in counts.windows(2) {
            assert!(
                pair[0].count >= pair[1].count,
                "counts should be non-increasing: {} vs {}",
                pair[0].count,
                pair[1].count
            );
        }
    }

    #[test]
    fn test_idempotent_add_same_doc_value() {
        let mut idx = MultiValueFacetIndex::new();
        idx.add_document("doc-1", FacetField::Tags, &["sport"]);
        idx.add_document("doc-1", FacetField::Tags, &["sport"]); // duplicate
        assert_eq!(idx.count_for_value(FacetField::Tags, "sport"), 1);
        assert_eq!(idx.doc_count(), 1);
    }

    #[test]
    fn test_docs_with_value() {
        let idx = build_index();
        let docs = idx.docs_with_value(FacetField::Tags, "music");
        assert_eq!(docs.len(), 2);
        assert!(docs.contains(&"doc-2".to_string()));
        assert!(docs.contains(&"doc-3".to_string()));
    }

    #[test]
    fn test_docs_with_all_values_intersection() {
        let idx = build_index();
        // Only doc-1 has both "sport" AND "outdoor".
        let docs = idx.docs_with_all_values(FacetField::Tags, &["sport", "outdoor"]);
        assert_eq!(docs, vec!["doc-1".to_string()]);
    }

    #[test]
    fn test_docs_with_any_value_union() {
        let idx = build_index();
        // "documentary" (doc-3) OR "4k" (doc-1).
        let docs = idx.docs_with_any_value(FacetField::Tags, &["documentary", "4k"]);
        assert_eq!(docs.len(), 2);
        assert!(docs.contains(&"doc-1".to_string()));
        assert!(docs.contains(&"doc-3".to_string()));
    }

    #[test]
    fn test_values_for_field() {
        let idx = build_index();
        let values = idx.values_for_field(FacetField::Tags);
        assert!(values.contains(&"sport".to_string()));
        assert!(values.contains(&"music".to_string()));
        assert!(values.contains(&"outdoor".to_string()));
        assert!(values.contains(&"documentary".to_string()));
        assert!(values.contains(&"4k".to_string()));
    }

    #[test]
    fn test_empty_field_returns_empty() {
        let idx = build_index();
        let counts = idx.counts(FacetField::Codecs); // nothing indexed for codecs
        assert!(counts.is_empty());
    }

    #[test]
    fn test_merge_two_indices() {
        let mut idx_a = MultiValueFacetIndex::new();
        idx_a.add_document("doc-a", FacetField::Genres, &["action", "drama"]);

        let mut idx_b = MultiValueFacetIndex::new();
        idx_b.add_document("doc-b", FacetField::Genres, &["drama", "comedy"]);

        idx_a.merge(&idx_b);

        assert_eq!(idx_a.doc_count(), 2);
        // "drama" should now appear in both doc-a and doc-b.
        assert_eq!(idx_a.count_for_value(FacetField::Genres, "drama"), 2);
        assert_eq!(idx_a.count_for_value(FacetField::Genres, "action"), 1);
        assert_eq!(idx_a.count_for_value(FacetField::Genres, "comedy"), 1);
    }

    #[test]
    fn test_apply_facet_filter_and_mode() {
        let idx = build_index();
        let candidates = vec![
            "doc-1".to_string(),
            "doc-2".to_string(),
            "doc-3".to_string(),
            "doc-4".to_string(),
        ];
        // AND: must have both "sport" AND "outdoor" → only doc-1.
        let result = apply_facet_filter(
            &idx,
            FacetField::Tags,
            &["sport", "outdoor"],
            FacetFilterMode::And,
            &candidates,
        );
        assert_eq!(result, vec!["doc-1".to_string()]);
    }

    #[test]
    fn test_apply_facet_filter_or_mode() {
        let idx = build_index();
        let candidates = vec![
            "doc-1".to_string(),
            "doc-2".to_string(),
            "doc-3".to_string(),
            "doc-4".to_string(),
        ];
        // OR: "documentary" OR "4k" → doc-1 and doc-3.
        let mut result = apply_facet_filter(
            &idx,
            FacetField::Tags,
            &["documentary", "4k"],
            FacetFilterMode::Or,
            &candidates,
        );
        result.sort();
        assert_eq!(result, vec!["doc-1".to_string(), "doc-3".to_string()]);
    }

    #[test]
    fn test_multiple_fields_independent() {
        let mut idx = MultiValueFacetIndex::new();
        idx.add_document("doc-1", FacetField::Tags, &["sport"]);
        idx.add_document("doc-1", FacetField::Languages, &["en", "fr"]);
        idx.add_document("doc-2", FacetField::Languages, &["en"]);

        assert_eq!(idx.count_for_value(FacetField::Tags, "sport"), 1);
        assert_eq!(idx.count_for_value(FacetField::Languages, "en"), 2);
        assert_eq!(idx.count_for_value(FacetField::Languages, "fr"), 1);
        // doc_count should be 2 (doc-1 counted once regardless of fields).
        assert_eq!(idx.doc_count(), 2);
    }

    #[test]
    fn test_custom_facet_field() {
        let mut idx = MultiValueFacetIndex::new();
        idx.add_document("doc-x", FacetField::Custom("ratings"), &["PG-13", "R"]);
        assert_eq!(idx.count_for_value(FacetField::Custom("ratings"), "PG-13"), 1);
        let values = idx.values_for_field(FacetField::Custom("ratings"));
        assert_eq!(values.len(), 2);
    }

    #[test]
    fn test_facet_field_label() {
        assert_eq!(FacetField::Tags.label(), "tags");
        assert_eq!(FacetField::Codecs.label(), "codecs");
        assert_eq!(FacetField::Custom("my_field").label(), "my_field");
    }

    #[test]
    fn test_docs_with_all_values_empty_requirement() {
        let idx = build_index();
        // Empty values list → returns all docs.
        let docs = idx.docs_with_all_values(FacetField::Tags, &[]);
        assert_eq!(docs.len(), idx.doc_count());
    }
}
