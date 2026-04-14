#![allow(dead_code)]
//! Content diff analysis between review iterations.
//!
//! Compares media metadata, timelines, and review state between different
//! versions of a review session to highlight what changed.

use std::collections::HashMap;
use std::fmt;

/// The kind of change detected in a diff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiffChangeKind {
    /// An element was added.
    Added,
    /// An element was removed.
    Removed,
    /// An element was modified.
    Modified,
    /// An element was moved to a different position.
    Moved,
    /// No change.
    Unchanged,
}

impl fmt::Display for DiffChangeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Added => "added",
            Self::Removed => "removed",
            Self::Modified => "modified",
            Self::Moved => "moved",
            Self::Unchanged => "unchanged",
        };
        write!(f, "{label}")
    }
}

/// Category of the entity that changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiffCategory {
    /// A timeline edit (cut, trim, splice).
    TimelineEdit,
    /// Audio content change.
    AudioChange,
    /// Video content change.
    VideoChange,
    /// A comment or annotation.
    CommentChange,
    /// An approval status change.
    ApprovalChange,
    /// Metadata change (title, tags, etc.).
    MetadataChange,
    /// Color grading change.
    ColorChange,
    /// Effects change.
    EffectsChange,
}

/// A single difference entry between two versions.
#[derive(Debug, Clone)]
pub struct DiffEntry {
    /// Unique identifier of the entry.
    pub id: String,
    /// Kind of change.
    pub change_kind: DiffChangeKind,
    /// Category of the changed entity.
    pub category: DiffCategory,
    /// Frame or time position (if applicable).
    pub frame_position: Option<u64>,
    /// Duration in frames (if applicable).
    pub frame_duration: Option<u64>,
    /// Human-readable description.
    pub description: String,
    /// Severity weight (0.0 = trivial, 1.0 = major).
    pub severity: f64,
}

/// Configuration for diff analysis.
#[derive(Debug, Clone)]
pub struct DiffConfig {
    /// Whether to include unchanged elements.
    pub include_unchanged: bool,
    /// Minimum severity threshold to include.
    pub min_severity: f64,
    /// Categories to include (empty means all).
    pub categories: Vec<DiffCategory>,
    /// Whether to group consecutive changes.
    pub group_consecutive: bool,
}

impl Default for DiffConfig {
    fn default() -> Self {
        Self {
            include_unchanged: false,
            min_severity: 0.0,
            categories: Vec::new(),
            group_consecutive: true,
        }
    }
}

/// Result of a diff analysis between two review versions.
#[derive(Debug, Clone)]
pub struct ReviewDiffResult {
    /// Version identifier of the source (older).
    pub source_version: String,
    /// Version identifier of the target (newer).
    pub target_version: String,
    /// All detected differences.
    pub entries: Vec<DiffEntry>,
}

impl ReviewDiffResult {
    /// Create a new empty diff result.
    #[must_use]
    pub fn new(source: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            source_version: source.into(),
            target_version: target.into(),
            entries: Vec::new(),
        }
    }

    /// Add a diff entry.
    pub fn add_entry(&mut self, entry: DiffEntry) {
        self.entries.push(entry);
    }

    /// Get the total number of changes (excluding unchanged).
    #[must_use]
    pub fn change_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.change_kind != DiffChangeKind::Unchanged)
            .count()
    }

    /// Get entries filtered by change kind.
    #[must_use]
    pub fn filter_by_kind(&self, kind: DiffChangeKind) -> Vec<&DiffEntry> {
        self.entries
            .iter()
            .filter(|e| e.change_kind == kind)
            .collect()
    }

    /// Get entries filtered by category.
    #[must_use]
    pub fn filter_by_category(&self, category: DiffCategory) -> Vec<&DiffEntry> {
        self.entries
            .iter()
            .filter(|e| e.category == category)
            .collect()
    }

    /// Get the average severity of all changes.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn average_severity(&self) -> f64 {
        let changes: Vec<&DiffEntry> = self
            .entries
            .iter()
            .filter(|e| e.change_kind != DiffChangeKind::Unchanged)
            .collect();
        if changes.is_empty() {
            return 0.0;
        }
        let total: f64 = changes.iter().map(|e| e.severity).sum();
        total / changes.len() as f64
    }

    /// Get a summary counting entries by change kind.
    #[must_use]
    pub fn summary_by_kind(&self) -> HashMap<DiffChangeKind, usize> {
        let mut counts = HashMap::new();
        for entry in &self.entries {
            *counts.entry(entry.change_kind).or_insert(0) += 1;
        }
        counts
    }

    /// Get a summary counting entries by category.
    #[must_use]
    pub fn summary_by_category(&self) -> HashMap<DiffCategory, usize> {
        let mut counts = HashMap::new();
        for entry in &self.entries {
            *counts.entry(entry.category).or_insert(0) += 1;
        }
        counts
    }

    /// Check whether there are any actual changes.
    #[must_use]
    pub fn has_changes(&self) -> bool {
        self.entries
            .iter()
            .any(|e| e.change_kind != DiffChangeKind::Unchanged)
    }

    /// Get the maximum severity among all entries.
    #[must_use]
    pub fn max_severity(&self) -> f64 {
        self.entries
            .iter()
            .map(|e| e.severity)
            .fold(0.0_f64, f64::max)
    }

    /// Filter entries above a minimum severity threshold.
    #[must_use]
    pub fn filter_above_severity(&self, min_severity: f64) -> Vec<&DiffEntry> {
        self.entries
            .iter()
            .filter(|e| e.severity >= min_severity)
            .collect()
    }
}

/// Analyzes differences between review metadata snapshots.
#[derive(Debug)]
pub struct ReviewDiffAnalyzer {
    /// Configuration for the analysis.
    config: DiffConfig,
}

impl Default for ReviewDiffAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl ReviewDiffAnalyzer {
    /// Create a new analyzer with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: DiffConfig::default(),
        }
    }

    /// Create an analyzer with custom configuration.
    #[must_use]
    pub fn with_config(config: DiffConfig) -> Self {
        Self { config }
    }

    /// Compare two metadata maps and produce a diff result.
    #[must_use]
    pub fn compare_metadata(
        &self,
        source: &HashMap<String, String>,
        target: &HashMap<String, String>,
        source_version: &str,
        target_version: &str,
    ) -> ReviewDiffResult {
        let mut result = ReviewDiffResult::new(source_version, target_version);
        let mut id_counter: u64 = 0;

        // Check for added and modified keys
        for (key, target_val) in target {
            id_counter += 1;
            if let Some(source_val) = source.get(key) {
                if source_val != target_val {
                    result.add_entry(DiffEntry {
                        id: format!("diff-{id_counter}"),
                        change_kind: DiffChangeKind::Modified,
                        category: DiffCategory::MetadataChange,
                        frame_position: None,
                        frame_duration: None,
                        description: format!(
                            "Key '{key}' changed from '{source_val}' to '{target_val}'"
                        ),
                        severity: 0.3,
                    });
                } else if self.config.include_unchanged {
                    result.add_entry(DiffEntry {
                        id: format!("diff-{id_counter}"),
                        change_kind: DiffChangeKind::Unchanged,
                        category: DiffCategory::MetadataChange,
                        frame_position: None,
                        frame_duration: None,
                        description: format!("Key '{key}' unchanged"),
                        severity: 0.0,
                    });
                }
            } else {
                result.add_entry(DiffEntry {
                    id: format!("diff-{id_counter}"),
                    change_kind: DiffChangeKind::Added,
                    category: DiffCategory::MetadataChange,
                    frame_position: None,
                    frame_duration: None,
                    description: format!("Key '{key}' added with value '{target_val}'"),
                    severity: 0.2,
                });
            }
        }

        // Check for removed keys
        for key in source.keys() {
            if !target.contains_key(key) {
                id_counter += 1;
                result.add_entry(DiffEntry {
                    id: format!("diff-{id_counter}"),
                    change_kind: DiffChangeKind::Removed,
                    category: DiffCategory::MetadataChange,
                    frame_position: None,
                    frame_duration: None,
                    description: format!("Key '{key}' removed"),
                    severity: 0.4,
                });
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entry(kind: DiffChangeKind, cat: DiffCategory, sev: f64) -> DiffEntry {
        DiffEntry {
            id: "test".to_string(),
            change_kind: kind,
            category: cat,
            frame_position: None,
            frame_duration: None,
            description: "test entry".to_string(),
            severity: sev,
        }
    }

    #[test]
    fn test_new_diff_result_empty() {
        let result = ReviewDiffResult::new("v1", "v2");
        assert_eq!(result.source_version, "v1");
        assert_eq!(result.target_version, "v2");
        assert!(result.entries.is_empty());
        assert!(!result.has_changes());
    }

    #[test]
    fn test_change_count_excludes_unchanged() {
        let mut result = ReviewDiffResult::new("v1", "v2");
        result.add_entry(sample_entry(
            DiffChangeKind::Added,
            DiffCategory::AudioChange,
            0.5,
        ));
        result.add_entry(sample_entry(
            DiffChangeKind::Unchanged,
            DiffCategory::VideoChange,
            0.0,
        ));
        result.add_entry(sample_entry(
            DiffChangeKind::Modified,
            DiffCategory::ColorChange,
            0.3,
        ));
        assert_eq!(result.change_count(), 2);
    }

    #[test]
    fn test_filter_by_kind() {
        let mut result = ReviewDiffResult::new("v1", "v2");
        result.add_entry(sample_entry(
            DiffChangeKind::Added,
            DiffCategory::AudioChange,
            0.5,
        ));
        result.add_entry(sample_entry(
            DiffChangeKind::Removed,
            DiffCategory::VideoChange,
            0.7,
        ));
        result.add_entry(sample_entry(
            DiffChangeKind::Added,
            DiffCategory::EffectsChange,
            0.4,
        ));
        let added = result.filter_by_kind(DiffChangeKind::Added);
        assert_eq!(added.len(), 2);
    }

    #[test]
    fn test_filter_by_category() {
        let mut result = ReviewDiffResult::new("v1", "v2");
        result.add_entry(sample_entry(
            DiffChangeKind::Added,
            DiffCategory::AudioChange,
            0.5,
        ));
        result.add_entry(sample_entry(
            DiffChangeKind::Modified,
            DiffCategory::AudioChange,
            0.3,
        ));
        result.add_entry(sample_entry(
            DiffChangeKind::Removed,
            DiffCategory::VideoChange,
            0.7,
        ));
        let audio = result.filter_by_category(DiffCategory::AudioChange);
        assert_eq!(audio.len(), 2);
    }

    #[test]
    fn test_average_severity() {
        let mut result = ReviewDiffResult::new("v1", "v2");
        result.add_entry(sample_entry(
            DiffChangeKind::Added,
            DiffCategory::AudioChange,
            0.4,
        ));
        result.add_entry(sample_entry(
            DiffChangeKind::Modified,
            DiffCategory::VideoChange,
            0.6,
        ));
        let avg = result.average_severity();
        assert!((avg - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_average_severity_empty() {
        let result = ReviewDiffResult::new("v1", "v2");
        assert!((result.average_severity() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_severity() {
        let mut result = ReviewDiffResult::new("v1", "v2");
        result.add_entry(sample_entry(
            DiffChangeKind::Added,
            DiffCategory::AudioChange,
            0.3,
        ));
        result.add_entry(sample_entry(
            DiffChangeKind::Modified,
            DiffCategory::VideoChange,
            0.9,
        ));
        result.add_entry(sample_entry(
            DiffChangeKind::Removed,
            DiffCategory::ColorChange,
            0.5,
        ));
        assert!((result.max_severity() - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_filter_above_severity() {
        let mut result = ReviewDiffResult::new("v1", "v2");
        result.add_entry(sample_entry(
            DiffChangeKind::Added,
            DiffCategory::AudioChange,
            0.2,
        ));
        result.add_entry(sample_entry(
            DiffChangeKind::Modified,
            DiffCategory::VideoChange,
            0.8,
        ));
        result.add_entry(sample_entry(
            DiffChangeKind::Removed,
            DiffCategory::ColorChange,
            0.5,
        ));
        let high = result.filter_above_severity(0.5);
        assert_eq!(high.len(), 2);
    }

    #[test]
    fn test_summary_by_kind() {
        let mut result = ReviewDiffResult::new("v1", "v2");
        result.add_entry(sample_entry(
            DiffChangeKind::Added,
            DiffCategory::AudioChange,
            0.5,
        ));
        result.add_entry(sample_entry(
            DiffChangeKind::Added,
            DiffCategory::VideoChange,
            0.5,
        ));
        result.add_entry(sample_entry(
            DiffChangeKind::Removed,
            DiffCategory::ColorChange,
            0.5,
        ));
        let summary = result.summary_by_kind();
        assert_eq!(summary.get(&DiffChangeKind::Added), Some(&2));
        assert_eq!(summary.get(&DiffChangeKind::Removed), Some(&1));
    }

    #[test]
    fn test_summary_by_category() {
        let mut result = ReviewDiffResult::new("v1", "v2");
        result.add_entry(sample_entry(
            DiffChangeKind::Added,
            DiffCategory::AudioChange,
            0.5,
        ));
        result.add_entry(sample_entry(
            DiffChangeKind::Modified,
            DiffCategory::AudioChange,
            0.3,
        ));
        result.add_entry(sample_entry(
            DiffChangeKind::Removed,
            DiffCategory::VideoChange,
            0.7,
        ));
        let summary = result.summary_by_category();
        assert_eq!(summary.get(&DiffCategory::AudioChange), Some(&2));
        assert_eq!(summary.get(&DiffCategory::VideoChange), Some(&1));
    }

    #[test]
    fn test_compare_metadata_added_key() {
        let analyzer = ReviewDiffAnalyzer::new();
        let source = HashMap::new();
        let mut target = HashMap::new();
        target.insert("title".to_string(), "New Title".to_string());
        let diff = analyzer.compare_metadata(&source, &target, "v1", "v2");
        assert_eq!(diff.change_count(), 1);
        assert_eq!(diff.entries[0].change_kind, DiffChangeKind::Added);
    }

    #[test]
    fn test_compare_metadata_removed_key() {
        let analyzer = ReviewDiffAnalyzer::new();
        let mut source = HashMap::new();
        source.insert("old_key".to_string(), "value".to_string());
        let target = HashMap::new();
        let diff = analyzer.compare_metadata(&source, &target, "v1", "v2");
        assert_eq!(diff.change_count(), 1);
        assert_eq!(diff.entries[0].change_kind, DiffChangeKind::Removed);
    }

    #[test]
    fn test_compare_metadata_modified_key() {
        let analyzer = ReviewDiffAnalyzer::new();
        let mut source = HashMap::new();
        source.insert("title".to_string(), "Old Title".to_string());
        let mut target = HashMap::new();
        target.insert("title".to_string(), "New Title".to_string());
        let diff = analyzer.compare_metadata(&source, &target, "v1", "v2");
        assert_eq!(diff.change_count(), 1);
        assert_eq!(diff.entries[0].change_kind, DiffChangeKind::Modified);
    }

    #[test]
    fn test_diff_change_kind_display() {
        assert_eq!(format!("{}", DiffChangeKind::Added), "added");
        assert_eq!(format!("{}", DiffChangeKind::Removed), "removed");
        assert_eq!(format!("{}", DiffChangeKind::Modified), "modified");
        assert_eq!(format!("{}", DiffChangeKind::Moved), "moved");
        assert_eq!(format!("{}", DiffChangeKind::Unchanged), "unchanged");
    }
}
