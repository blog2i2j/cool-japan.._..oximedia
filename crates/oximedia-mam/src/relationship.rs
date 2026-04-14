//! High-level asset relationship graph for the MAM system.
//!
//! Provides [`AssetRelationship`], a simple directed edge between two assets
//! identified by `u64` IDs with a free-form `rel_type` label, and
//! [`RelationshipGraph`] which stores and queries these edges.
//!
//! This complements the more detailed [`crate::asset_relations`] module
//! (which uses typed [`crate::asset_relations::RelationKind`] variants) with a
//! lighter string-keyed API suitable for dynamic or plugin-defined relationship
//! types.

#![allow(dead_code)]

use std::collections::HashMap;

// ── AssetRelationship ─────────────────────────────────────────────────────────

/// A directed relationship between two media assets.
///
/// `from` → `to` with a free-form `rel_type` label (e.g. `"derived_from"`,
/// `"part_of"`, `"next_version"`).
#[derive(Debug, Clone, PartialEq)]
pub struct AssetRelationship {
    /// Source asset identifier.
    pub from: u64,
    /// Target asset identifier.
    pub to: u64,
    /// Free-form relationship type label.
    pub rel_type: String,
}

impl AssetRelationship {
    /// Create a new `AssetRelationship`.
    ///
    /// # Arguments
    ///
    /// * `from`     — source asset ID.
    /// * `to`       — target asset ID.
    /// * `rel_type` — string describing the relationship kind.
    #[must_use]
    pub fn new(from: u64, to: u64, rel_type: &str) -> Self {
        Self {
            from,
            to,
            rel_type: rel_type.to_owned(),
        }
    }

    /// Returns `true` if this relationship has the given type label.
    #[must_use]
    pub fn is_type(&self, rel_type: &str) -> bool {
        self.rel_type == rel_type
    }
}

// ── RelationshipGraph ─────────────────────────────────────────────────────────

/// An in-memory directed graph of [`AssetRelationship`] edges.
///
/// Edges are indexed by source asset ID for O(1) neighbourhood lookup.
///
/// # Example
///
/// ```rust
/// use oximedia_mam::relationship::{AssetRelationship, RelationshipGraph};
///
/// let mut g = RelationshipGraph::new();
/// g.add(AssetRelationship::new(1, 2, "derived_from"));
/// g.add(AssetRelationship::new(1, 3, "part_of"));
///
/// let related = g.related_to(1);
/// assert_eq!(related.len(), 2);
/// ```
#[derive(Debug, Default)]
pub struct RelationshipGraph {
    /// Adjacency list: from_id → list of (to_id, rel_type).
    edges: HashMap<u64, Vec<(u64, String)>>,
    /// All relationships in insertion order.
    relations: Vec<AssetRelationship>,
}

impl RelationshipGraph {
    /// Create an empty `RelationshipGraph`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a relationship to the graph.
    ///
    /// Duplicate edges (same `from`, `to`, and `rel_type`) are silently ignored.
    pub fn add(&mut self, rel: AssetRelationship) {
        // Deduplication check
        if let Some(existing) = self.edges.get(&rel.from) {
            if existing
                .iter()
                .any(|(t, k)| *t == rel.to && k == &rel.rel_type)
            {
                return;
            }
        }
        self.edges
            .entry(rel.from)
            .or_default()
            .push((rel.to, rel.rel_type.clone()));
        self.relations.push(rel);
    }

    /// Return all assets related to `asset_id` as (target_id, rel_type) pairs.
    ///
    /// Only outgoing edges (where `asset_id` is the source) are returned.
    #[must_use]
    pub fn related_to(&self, asset_id: u64) -> Vec<(u64, &str)> {
        self.edges
            .get(&asset_id)
            .map(|v| v.iter().map(|(t, k)| (*t, k.as_str())).collect())
            .unwrap_or_default()
    }

    /// Return all relationships of the given type.
    #[must_use]
    pub fn by_type(&self, rel_type: &str) -> Vec<&AssetRelationship> {
        self.relations
            .iter()
            .filter(|r| r.rel_type == rel_type)
            .collect()
    }

    /// Return the total number of relationships in the graph.
    #[must_use]
    pub fn len(&self) -> usize {
        self.relations.len()
    }

    /// Return `true` if the graph has no relationships.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.relations.is_empty()
    }

    /// Remove all relationships involving `asset_id` (as source or target).
    pub fn remove_asset(&mut self, asset_id: u64) {
        self.relations
            .retain(|r| r.from != asset_id && r.to != asset_id);
        // Rebuild index
        self.edges.clear();
        for rel in &self.relations {
            self.edges
                .entry(rel.from)
                .or_default()
                .push((rel.to, rel.rel_type.clone()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_graph() -> RelationshipGraph {
        let mut g = RelationshipGraph::new();
        g.add(AssetRelationship::new(1, 2, "derived_from"));
        g.add(AssetRelationship::new(1, 3, "part_of"));
        g.add(AssetRelationship::new(4, 1, "next_version"));
        g
    }

    #[test]
    fn test_new_empty() {
        let g = RelationshipGraph::new();
        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
    }

    #[test]
    fn test_add_and_len() {
        let g = sample_graph();
        assert_eq!(g.len(), 3);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_related_to_returns_outgoing() {
        let g = sample_graph();
        let related = g.related_to(1);
        assert_eq!(related.len(), 2);
        let targets: Vec<u64> = related.iter().map(|(t, _)| *t).collect();
        assert!(targets.contains(&2));
        assert!(targets.contains(&3));
    }

    #[test]
    fn test_related_to_includes_rel_type() {
        let g = sample_graph();
        let related = g.related_to(1);
        let types: Vec<&str> = related.iter().map(|(_, k)| *k).collect();
        assert!(types.contains(&"derived_from"));
        assert!(types.contains(&"part_of"));
    }

    #[test]
    fn test_related_to_unknown_asset_empty() {
        let g = sample_graph();
        assert!(g.related_to(999).is_empty());
    }

    #[test]
    fn test_by_type() {
        let g = sample_graph();
        let derived = g.by_type("derived_from");
        assert_eq!(derived.len(), 1);
        assert_eq!(derived[0].from, 1);
        assert_eq!(derived[0].to, 2);
    }

    #[test]
    fn test_duplicate_ignored() {
        let mut g = sample_graph();
        g.add(AssetRelationship::new(1, 2, "derived_from")); // dup
        assert_eq!(g.len(), 3);
    }

    #[test]
    fn test_remove_asset_removes_both_incoming_and_outgoing() {
        let mut g = sample_graph();
        g.remove_asset(1);
        // (1→2), (1→3), (4→1) all removed
        assert_eq!(g.len(), 0);
        assert!(g.is_empty());
    }

    #[test]
    fn test_asset_relationship_is_type() {
        let rel = AssetRelationship::new(10, 20, "contains");
        assert!(rel.is_type("contains"));
        assert!(!rel.is_type("other"));
    }

    #[test]
    fn test_different_rel_type_same_endpoints_not_duplicate() {
        let mut g = RelationshipGraph::new();
        g.add(AssetRelationship::new(1, 2, "derived_from"));
        g.add(AssetRelationship::new(1, 2, "references")); // different type
        assert_eq!(g.len(), 2);
    }
}
