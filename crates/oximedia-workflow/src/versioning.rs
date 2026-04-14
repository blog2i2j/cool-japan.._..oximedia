//! Workflow version management and diffing.
//!
//! [`WorkflowVersion`] represents a named snapshot of a workflow's serialised
//! data string (e.g. a JSON or YAML definition). Two versions can be compared
//! with [`WorkflowVersion::diff`] which returns a line-level diff as a
//! `Vec<String>` in unified diff style (lines prefixed with `+` or `-`).
//!
//! # Design
//!
//! - Versions are identified by a numeric `id` (e.g. a monotonically increasing
//!   revision number or a timestamp).
//! - The `data` field holds the full serialised workflow as a `String`.
//! - [`WorkflowVersion::diff`] compares `self` against `other` line by line,
//!   returning lines that appear only in `self` (prefixed `+`) or only in
//!   `other` (prefixed `-`).
//!
//! # Example
//!
//! ```rust
//! use oximedia_workflow::versioning::WorkflowVersion;
//!
//! let v1 = WorkflowVersion::new(1, "step: ingest\nstep: transcode\n");
//! let v2 = WorkflowVersion::new(2, "step: ingest\nstep: upload\n");
//!
//! let diff = v1.diff(&v2);
//! assert!(diff.iter().any(|l| l.starts_with('+')));
//! assert!(diff.iter().any(|l| l.starts_with('-')));
//! ```

#![allow(dead_code)]

use std::collections::HashSet;

// ---------------------------------------------------------------------------
// WorkflowVersion
// ---------------------------------------------------------------------------

/// A versioned snapshot of a workflow's serialised definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkflowVersion {
    /// Monotonically increasing version identifier.
    pub id: u64,
    /// Full serialised workflow data (JSON, YAML, or custom DSL string).
    pub data: String,
}

impl WorkflowVersion {
    /// Create a new workflow version.
    #[must_use]
    pub fn new(id: u64, data: impl Into<String>) -> Self {
        Self {
            id,
            data: data.into(),
        }
    }

    // ── diffing ───────────────────────────────────────────────────────────────

    /// Compute a line-level diff between `self` and `other`.
    ///
    /// Returns a `Vec<String>` where:
    /// - Lines prefixed with `"+"` are present in `self` but not in `other`
    ///   (added relative to `other`).
    /// - Lines prefixed with `"-"` are present in `other` but not in `self`
    ///   (removed relative to `other`, i.e. present in the older version).
    ///
    /// Lines common to both versions are **not** included in the output.
    /// The order within each group (additions, removals) follows their order
    /// of first appearance in the respective version.
    ///
    /// For an identical pair the returned `Vec` is empty.
    #[must_use]
    pub fn diff(&self, other: &WorkflowVersion) -> Vec<String> {
        let self_lines: Vec<&str> = self.data.lines().collect();
        let other_lines: Vec<&str> = other.data.lines().collect();

        let self_set: HashSet<&str> = self_lines.iter().copied().collect();
        let other_set: HashSet<&str> = other_lines.iter().copied().collect();

        let mut result: Vec<String> = Vec::new();

        // Lines in self but not in other → added (+).
        for &line in &self_lines {
            if !other_set.contains(line) {
                result.push(format!("+{line}"));
            }
        }

        // Lines in other but not in self → removed (-).
        for &line in &other_lines {
            if !self_set.contains(line) {
                result.push(format!("-{line}"));
            }
        }

        result
    }

    /// Return `true` if `self` and `other` contain identical data strings.
    #[must_use]
    pub fn is_identical(&self, other: &WorkflowVersion) -> bool {
        self.data == other.data
    }

    /// Return a compact summary of this version's statistics.
    #[must_use]
    pub fn summary(&self) -> VersionSummary {
        let line_count = self.data.lines().count();
        let byte_count = self.data.len();
        VersionSummary {
            id: self.id,
            line_count,
            byte_count,
        }
    }
}

// ---------------------------------------------------------------------------
// VersionSummary
// ---------------------------------------------------------------------------

/// Compact statistics for a [`WorkflowVersion`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VersionSummary {
    /// Version identifier.
    pub id: u64,
    /// Number of lines in the data string.
    pub line_count: usize,
    /// Size in bytes of the data string.
    pub byte_count: usize,
}

// ---------------------------------------------------------------------------
// WorkflowVersionRegistry
// ---------------------------------------------------------------------------

/// An ordered registry of workflow versions, supporting diff and history queries.
#[derive(Debug, Default)]
pub struct WorkflowVersionRegistry {
    versions: Vec<WorkflowVersion>,
}

impl WorkflowVersionRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a new version.
    pub fn push(&mut self, version: WorkflowVersion) {
        self.versions.push(version);
    }

    /// Return the most recent version, if any.
    #[must_use]
    pub fn latest(&self) -> Option<&WorkflowVersion> {
        self.versions.last()
    }

    /// Find a version by its `id`.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&WorkflowVersion> {
        self.versions.iter().find(|v| v.id == id)
    }

    /// Compute the diff between two versions identified by their `id`s.
    ///
    /// Returns `None` if either id is not found.
    #[must_use]
    pub fn diff_ids(&self, from_id: u64, to_id: u64) -> Option<Vec<String>> {
        let from = self.get(from_id)?;
        let to = self.get(to_id)?;
        Some(from.diff(to))
    }

    /// Number of versions stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.versions.len()
    }

    /// Return `true` if no versions are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.versions.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_stores_id_and_data() {
        let v = WorkflowVersion::new(42, "hello world");
        assert_eq!(v.id, 42);
        assert_eq!(v.data, "hello world");
    }

    // ── diff ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_diff_identical_is_empty() {
        let v1 = WorkflowVersion::new(1, "a\nb\nc");
        let v2 = WorkflowVersion::new(2, "a\nb\nc");
        assert!(v1.diff(&v2).is_empty());
    }

    #[test]
    fn test_diff_added_lines() {
        let v1 = WorkflowVersion::new(1, "a\nb\nc");
        let v2 = WorkflowVersion::new(2, "a\nb");
        let diff = v1.diff(&v2);
        // "c" is in v1 but not v2 → added.
        assert!(
            diff.iter().any(|l| l == "+c"),
            "Expected +c in diff: {diff:?}"
        );
    }

    #[test]
    fn test_diff_removed_lines() {
        let v1 = WorkflowVersion::new(1, "a\nb");
        let v2 = WorkflowVersion::new(2, "a\nb\nc");
        let diff = v1.diff(&v2);
        // "c" is in v2 but not v1 → removed.
        assert!(
            diff.iter().any(|l| l == "-c"),
            "Expected -c in diff: {diff:?}"
        );
    }

    #[test]
    fn test_diff_both_adds_and_removes() {
        let v1 = WorkflowVersion::new(1, "step: ingest\nstep: transcode\n");
        let v2 = WorkflowVersion::new(2, "step: ingest\nstep: upload\n");
        let diff = v1.diff(&v2);
        assert!(diff.iter().any(|l| l.starts_with('+')));
        assert!(diff.iter().any(|l| l.starts_with('-')));
    }

    #[test]
    fn test_is_identical_same_data() {
        let v1 = WorkflowVersion::new(1, "data");
        let v2 = WorkflowVersion::new(2, "data");
        assert!(v1.is_identical(&v2));
    }

    #[test]
    fn test_is_identical_different_data() {
        let v1 = WorkflowVersion::new(1, "a");
        let v2 = WorkflowVersion::new(2, "b");
        assert!(!v1.is_identical(&v2));
    }

    // ── summary ───────────────────────────────────────────────────────────────

    #[test]
    fn test_summary_line_count() {
        let v = WorkflowVersion::new(1, "a\nb\nc");
        let s = v.summary();
        assert_eq!(s.id, 1);
        assert_eq!(s.line_count, 3);
        assert_eq!(s.byte_count, "a\nb\nc".len());
    }

    // ── registry ─────────────────────────────────────────────────────────────

    #[test]
    fn test_registry_push_and_get() {
        let mut reg = WorkflowVersionRegistry::new();
        reg.push(WorkflowVersion::new(1, "v1"));
        reg.push(WorkflowVersion::new(2, "v2"));
        assert_eq!(reg.len(), 2);
        assert_eq!(reg.get(1).map(|v| v.data.as_str()), Some("v1"));
        assert_eq!(reg.get(2).map(|v| v.data.as_str()), Some("v2"));
    }

    #[test]
    fn test_registry_latest() {
        let mut reg = WorkflowVersionRegistry::new();
        assert!(reg.latest().is_none());
        reg.push(WorkflowVersion::new(1, "v1"));
        reg.push(WorkflowVersion::new(2, "v2"));
        assert_eq!(reg.latest().map(|v| v.id), Some(2));
    }

    #[test]
    fn test_registry_diff_ids() {
        let mut reg = WorkflowVersionRegistry::new();
        reg.push(WorkflowVersion::new(1, "a\nb"));
        reg.push(WorkflowVersion::new(2, "a\nc"));
        let diff = reg.diff_ids(1, 2);
        assert!(diff.is_some());
        let diff = diff.unwrap();
        assert!(!diff.is_empty());
    }

    #[test]
    fn test_registry_diff_ids_missing_returns_none() {
        let reg = WorkflowVersionRegistry::new();
        assert!(reg.diff_ids(1, 2).is_none());
    }

    #[test]
    fn test_registry_is_empty() {
        let mut reg = WorkflowVersionRegistry::new();
        assert!(reg.is_empty());
        reg.push(WorkflowVersion::new(1, "x"));
        assert!(!reg.is_empty());
    }
}
