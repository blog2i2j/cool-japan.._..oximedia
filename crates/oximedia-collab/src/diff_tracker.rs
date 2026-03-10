#![allow(dead_code)]
//! Diff tracking for collaborative edit sessions.
//!
//! Records and compares document state differences so collaborators
//! can review what changed between any two versions of a project asset.

use std::collections::HashMap;

/// The category of change represented by a diff entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiffType {
    /// A new value was inserted.
    Insert,
    /// An existing value was deleted.
    Delete,
    /// An existing value was replaced.
    Replace,
    /// No change; content is identical.
    Equal,
}

impl std::fmt::Display for DiffType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiffType::Insert => write!(f, "Insert"),
            DiffType::Delete => write!(f, "Delete"),
            DiffType::Replace => write!(f, "Replace"),
            DiffType::Equal => write!(f, "Equal"),
        }
    }
}

/// A single file-level diff describing what changed between two versions.
#[derive(Debug, Clone)]
pub struct FileDiff {
    /// Path or identifier of the file/asset that changed.
    pub path: String,
    /// Kind of change.
    pub diff_type: DiffType,
    /// Old value, if applicable.
    pub old_value: Option<String>,
    /// New value, if applicable.
    pub new_value: Option<String>,
    /// Line or byte offset where the change starts.
    pub offset: usize,
}

impl FileDiff {
    /// Create an insert diff.
    pub fn insert(path: impl Into<String>, new_value: impl Into<String>, offset: usize) -> Self {
        Self {
            path: path.into(),
            diff_type: DiffType::Insert,
            old_value: None,
            new_value: Some(new_value.into()),
            offset,
        }
    }

    /// Create a delete diff.
    pub fn delete(path: impl Into<String>, old_value: impl Into<String>, offset: usize) -> Self {
        Self {
            path: path.into(),
            diff_type: DiffType::Delete,
            old_value: Some(old_value.into()),
            new_value: None,
            offset,
        }
    }

    /// Create a replace diff.
    pub fn replace(
        path: impl Into<String>,
        old_value: impl Into<String>,
        new_value: impl Into<String>,
        offset: usize,
    ) -> Self {
        Self {
            path: path.into(),
            diff_type: DiffType::Replace,
            old_value: Some(old_value.into()),
            new_value: Some(new_value.into()),
            offset,
        }
    }

    /// Returns true if this diff introduces new content.
    pub fn is_additive(&self) -> bool {
        matches!(self.diff_type, DiffType::Insert | DiffType::Replace)
    }

    /// Returns true if this diff removes content.
    pub fn is_destructive(&self) -> bool {
        matches!(self.diff_type, DiffType::Delete | DiffType::Replace)
    }
}

/// A snapshot of a named document represented as a map of key → value pairs.
#[derive(Debug, Clone, Default)]
pub struct DocumentSnapshot {
    /// Document identifier.
    pub doc_id: String,
    /// Version number of this snapshot.
    pub version: u64,
    /// Key-value content map.
    pub fields: HashMap<String, String>,
}

impl DocumentSnapshot {
    /// Create a new snapshot.
    pub fn new(doc_id: impl Into<String>, version: u64) -> Self {
        Self {
            doc_id: doc_id.into(),
            version,
            fields: HashMap::new(),
        }
    }

    /// Insert or update a field.
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.fields.insert(key.into(), value.into());
    }

    /// Remove a field, returning the previous value if any.
    pub fn remove(&mut self, key: &str) -> Option<String> {
        self.fields.remove(key)
    }
}

/// Tracks diffs between document snapshots across multiple versions.
#[derive(Debug, Default)]
pub struct DiffTracker {
    /// Stored diffs keyed by (doc_id, from_version, to_version).
    diffs: HashMap<(String, u64, u64), Vec<FileDiff>>,
}

impl DiffTracker {
    /// Create a new, empty diff tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute a diff between two document snapshots and store it.
    ///
    /// Returns the list of [`FileDiff`] entries representing the delta.
    pub fn compute_diff(
        &mut self,
        from: &DocumentSnapshot,
        to: &DocumentSnapshot,
    ) -> Vec<FileDiff> {
        assert_eq!(
            from.doc_id, to.doc_id,
            "Cannot diff snapshots of different documents"
        );

        let mut diffs = Vec::new();
        let path = &from.doc_id;

        // Keys present in `from`
        for (key, old_val) in &from.fields {
            match to.fields.get(key) {
                Some(new_val) if new_val != old_val => {
                    diffs.push(FileDiff::replace(
                        format!("{}/{}", path, key),
                        old_val.clone(),
                        new_val.clone(),
                        0,
                    ));
                }
                None => {
                    diffs.push(FileDiff::delete(
                        format!("{}/{}", path, key),
                        old_val.clone(),
                        0,
                    ));
                }
                _ => {} // Equal – no diff entry
            }
        }

        // Keys only in `to` (insertions)
        for (key, new_val) in &to.fields {
            if !from.fields.contains_key(key) {
                diffs.push(FileDiff::insert(
                    format!("{}/{}", path, key),
                    new_val.clone(),
                    0,
                ));
            }
        }

        let key = (from.doc_id.clone(), from.version, to.version);
        self.diffs.insert(key, diffs.clone());
        diffs
    }

    /// Retrieve previously computed diffs, if any.
    pub fn get_diff(
        &self,
        doc_id: &str,
        from_version: u64,
        to_version: u64,
    ) -> Option<&Vec<FileDiff>> {
        self.diffs
            .get(&(doc_id.to_string(), from_version, to_version))
    }

    /// Returns the total number of stored diff sets.
    pub fn diff_count(&self) -> usize {
        self.diffs.len()
    }

    /// Clear all stored diffs.
    pub fn clear(&mut self) {
        self.diffs.clear();
    }

    /// Returns the number of `Insert` entries across a specific diff.
    pub fn count_inserts(&self, doc_id: &str, from: u64, to: u64) -> usize {
        self.get_diff(doc_id, from, to)
            .map(|diffs| {
                diffs
                    .iter()
                    .filter(|d| d.diff_type == DiffType::Insert)
                    .count()
            })
            .unwrap_or(0)
    }

    /// Returns the number of `Delete` entries across a specific diff.
    pub fn count_deletes(&self, doc_id: &str, from: u64, to: u64) -> usize {
        self.get_diff(doc_id, from, to)
            .map(|diffs| {
                diffs
                    .iter()
                    .filter(|d| d.diff_type == DiffType::Delete)
                    .count()
            })
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snap(id: &str, ver: u64, fields: &[(&str, &str)]) -> DocumentSnapshot {
        let mut snap = DocumentSnapshot::new(id, ver);
        for (k, v) in fields {
            snap.set(*k, *v);
        }
        snap
    }

    #[test]
    fn test_diff_type_display() {
        assert_eq!(DiffType::Insert.to_string(), "Insert");
        assert_eq!(DiffType::Delete.to_string(), "Delete");
        assert_eq!(DiffType::Replace.to_string(), "Replace");
        assert_eq!(DiffType::Equal.to_string(), "Equal");
    }

    #[test]
    fn test_file_diff_insert_constructor() {
        let d = FileDiff::insert("doc/field", "newval", 0);
        assert_eq!(d.diff_type, DiffType::Insert);
        assert!(d.old_value.is_none());
        assert_eq!(d.new_value.as_deref(), Some("newval"));
    }

    #[test]
    fn test_file_diff_delete_constructor() {
        let d = FileDiff::delete("doc/field", "oldval", 5);
        assert_eq!(d.diff_type, DiffType::Delete);
        assert_eq!(d.old_value.as_deref(), Some("oldval"));
        assert!(d.new_value.is_none());
        assert_eq!(d.offset, 5);
    }

    #[test]
    fn test_file_diff_replace_constructor() {
        let d = FileDiff::replace("doc/field", "old", "new", 10);
        assert_eq!(d.diff_type, DiffType::Replace);
        assert!(d.is_additive());
        assert!(d.is_destructive());
    }

    #[test]
    fn test_file_diff_is_additive_destructive() {
        assert!(FileDiff::insert("p", "v", 0).is_additive());
        assert!(!FileDiff::delete("p", "v", 0).is_additive());
        assert!(FileDiff::delete("p", "v", 0).is_destructive());
        assert!(!FileDiff::insert("p", "v", 0).is_destructive());
    }

    #[test]
    fn test_document_snapshot_set_remove() {
        let mut snap = DocumentSnapshot::new("doc1", 1);
        snap.set("title", "Hello");
        assert_eq!(snap.fields.get("title").map(|s| s.as_str()), Some("Hello"));
        let removed = snap.remove("title");
        assert_eq!(removed.as_deref(), Some("Hello"));
        assert!(snap.fields.get("title").is_none());
    }

    #[test]
    fn test_compute_diff_insert() {
        let from = make_snap("doc", 1, &[]);
        let to = make_snap("doc", 2, &[("title", "New")]);
        let mut tracker = DiffTracker::new();
        let diffs = tracker.compute_diff(&from, &to);
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].diff_type, DiffType::Insert);
    }

    #[test]
    fn test_compute_diff_delete() {
        let from = make_snap("doc", 1, &[("title", "Old")]);
        let to = make_snap("doc", 2, &[]);
        let mut tracker = DiffTracker::new();
        let diffs = tracker.compute_diff(&from, &to);
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].diff_type, DiffType::Delete);
    }

    #[test]
    fn test_compute_diff_replace() {
        let from = make_snap("doc", 1, &[("title", "Old")]);
        let to = make_snap("doc", 2, &[("title", "New")]);
        let mut tracker = DiffTracker::new();
        let diffs = tracker.compute_diff(&from, &to);
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].diff_type, DiffType::Replace);
    }

    #[test]
    fn test_compute_diff_equal_produces_no_entry() {
        let from = make_snap("doc", 1, &[("title", "Same")]);
        let to = make_snap("doc", 2, &[("title", "Same")]);
        let mut tracker = DiffTracker::new();
        let diffs = tracker.compute_diff(&from, &to);
        assert!(diffs.is_empty());
    }

    #[test]
    fn test_get_diff_retrieval() {
        let from = make_snap("doc", 1, &[("a", "1")]);
        let to = make_snap("doc", 2, &[("a", "2")]);
        let mut tracker = DiffTracker::new();
        tracker.compute_diff(&from, &to);
        let stored = tracker.get_diff("doc", 1, 2);
        assert!(stored.is_some());
        assert_eq!(
            stored.expect("collab test operation should succeed").len(),
            1
        );
    }

    #[test]
    fn test_diff_count_and_clear() {
        let from = make_snap("docX", 1, &[("k", "v")]);
        let to = make_snap("docX", 2, &[("k", "v2")]);
        let mut tracker = DiffTracker::new();
        tracker.compute_diff(&from, &to);
        assert_eq!(tracker.diff_count(), 1);
        tracker.clear();
        assert_eq!(tracker.diff_count(), 0);
    }

    #[test]
    fn test_count_inserts_deletes() {
        let from = make_snap("d2", 1, &[("x", "1"), ("y", "2")]);
        let to = make_snap("d2", 2, &[("z", "3")]);
        let mut tracker = DiffTracker::new();
        tracker.compute_diff(&from, &to);
        assert_eq!(tracker.count_inserts("d2", 1, 2), 1);
        assert_eq!(tracker.count_deletes("d2", 1, 2), 2);
    }
}
