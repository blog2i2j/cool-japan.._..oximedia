//! Detailed deduplication reporting with disk-space savings, confidence
//! scores, and action recommendations.
//!
//! This module extends the base deduplication machinery with:
//!
//! - [`DetailedDuplicateGroup`]: Duplicate group with per-file metadata,
//!   size-savings estimates, confidence scores, and recommended action.
//! - [`RecommendedAction`]: Enum of actions the user or automation can take.
//! - [`SpaceSavingsEstimate`]: Byte-level space savings broken down by
//!   duplicate tier.
//! - [`DetailedReport`]: Aggregate report over all groups.
//! - [`DetailedReportBuilder`]: Fluent builder for assembling a report
//!   incrementally.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// RecommendedAction
// ---------------------------------------------------------------------------

/// Recommended action for a duplicate file group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecommendedAction {
    /// Delete all files except the best representative.
    DeleteDuplicates,
    /// Replace duplicate files with symbolic links to the representative.
    SymlinkDuplicates,
    /// Replace duplicate files with hard links to the representative.
    HardlinkDuplicates,
    /// Move duplicates to an archive directory for manual review.
    ArchiveDuplicates,
    /// Confidence is too low for automatic action; request manual review.
    ManualReview,
    /// Only one file in the group; no action needed.
    NoAction,
}

impl RecommendedAction {
    /// Return a short machine-readable label for this action.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::DeleteDuplicates => "delete",
            Self::SymlinkDuplicates => "symlink",
            Self::HardlinkDuplicates => "hardlink",
            Self::ArchiveDuplicates => "archive",
            Self::ManualReview => "manual_review",
            Self::NoAction => "no_action",
        }
    }

    /// Return a human-readable description.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::DeleteDuplicates => {
                "Delete all duplicate files, keeping only the representative."
            }
            Self::SymlinkDuplicates => {
                "Replace duplicate files with symbolic links to the representative."
            }
            Self::HardlinkDuplicates => {
                "Replace duplicate files with hard links to the representative."
            }
            Self::ArchiveDuplicates => {
                "Move duplicates to an archive directory for manual review."
            }
            Self::ManualReview => {
                "Similarity confidence is insufficient for automated action; review manually."
            }
            Self::NoAction => "Single-member group; no action required.",
        }
    }
}

// ---------------------------------------------------------------------------
// ConfidenceTier
// ---------------------------------------------------------------------------

/// Confidence tier based on the similarity score.
///
/// Variants are ordered from lowest to highest so that `Low < Medium < High < Exact`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConfidenceTier {
    /// Low confidence (score < 0.75); possible duplicates, manual review
    /// recommended.
    Low,
    /// Medium confidence (0.75 ≤ score < 0.90); probable duplicates.
    Medium,
    /// High confidence (0.90 ≤ score < 0.98); near-identical duplicates.
    High,
    /// Very high confidence (score ≥ 0.98); likely bitwise identical after
    /// normalization.
    Exact,
}

impl ConfidenceTier {
    /// Classify a similarity score into a confidence tier.
    #[must_use]
    pub fn from_score(score: f64) -> Self {
        if score >= 0.98 {
            Self::Exact
        } else if score >= 0.90 {
            Self::High
        } else if score >= 0.75 {
            Self::Medium
        } else {
            Self::Low
        }
    }

    /// Return a short label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::High => "high",
            Self::Medium => "medium",
            Self::Low => "low",
        }
    }
}

// ---------------------------------------------------------------------------
// FileEntry
// ---------------------------------------------------------------------------

/// Metadata for a single file within a duplicate group.
#[derive(Debug, Clone)]
pub struct FileEntry {
    /// Path or identifier for this file.
    pub path: String,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Whether this is the chosen representative of the group.
    pub is_representative: bool,
    /// Similarity score to the representative (1.0 for the representative
    /// itself).
    pub similarity_to_rep: f64,
}

impl FileEntry {
    /// Create a new file entry.
    #[must_use]
    pub fn new(path: impl Into<String>, size_bytes: u64, similarity_to_rep: f64) -> Self {
        Self {
            path: path.into(),
            size_bytes,
            is_representative: false,
            similarity_to_rep,
        }
    }
}

// ---------------------------------------------------------------------------
// SpaceSavingsEstimate
// ---------------------------------------------------------------------------

/// Space savings estimate for a single duplicate group or a whole report.
#[derive(Debug, Clone, Default)]
pub struct SpaceSavingsEstimate {
    /// Bytes that would be freed by deleting non-representative files.
    pub reclaimable_bytes: u64,
    /// Bytes occupied by the representative files (lower bound of total kept).
    pub retained_bytes: u64,
    /// Fraction of total group storage that is reclaimable (0.0 – 1.0).
    pub savings_ratio: f64,
}

impl SpaceSavingsEstimate {
    /// Compute estimates from a slice of [`FileEntry`]s.
    #[must_use]
    pub fn from_entries(entries: &[FileEntry]) -> Self {
        let total_bytes: u64 = entries.iter().map(|e| e.size_bytes).sum();
        let retained_bytes: u64 = entries
            .iter()
            .filter(|e| e.is_representative)
            .map(|e| e.size_bytes)
            .sum();
        let reclaimable_bytes = total_bytes.saturating_sub(retained_bytes);
        let savings_ratio = if total_bytes > 0 {
            reclaimable_bytes as f64 / total_bytes as f64
        } else {
            0.0
        };
        Self {
            reclaimable_bytes,
            retained_bytes,
            savings_ratio,
        }
    }

    /// Merge another estimate into this one (accumulate totals).
    pub fn merge(&mut self, other: &Self) {
        self.reclaimable_bytes += other.reclaimable_bytes;
        self.retained_bytes += other.retained_bytes;
        let total = self.reclaimable_bytes + self.retained_bytes;
        self.savings_ratio = if total > 0 {
            self.reclaimable_bytes as f64 / total as f64
        } else {
            0.0
        };
    }

    /// Return a human-readable string.
    #[must_use]
    pub fn description(&self) -> String {
        format!(
            "{} bytes reclaimable / {} bytes retained ({:.1}% savings)",
            self.reclaimable_bytes,
            self.retained_bytes,
            self.savings_ratio * 100.0,
        )
    }
}

// ---------------------------------------------------------------------------
// DetailedDuplicateGroup
// ---------------------------------------------------------------------------

/// A single group of duplicate or near-duplicate files, enriched with
/// detailed metadata.
#[derive(Debug, Clone)]
pub struct DetailedDuplicateGroup {
    /// Group identifier.
    pub id: usize,
    /// Detection method that found this group (e.g. "phash", "ssim").
    pub method: String,
    /// Member files.
    pub files: Vec<FileEntry>,
    /// Mean pairwise similarity within this group.
    pub mean_similarity: f64,
    /// Confidence tier derived from `mean_similarity`.
    pub confidence_tier: ConfidenceTier,
    /// Recommended action for this group.
    pub action: RecommendedAction,
    /// Space savings estimate for this group.
    pub space_savings: SpaceSavingsEstimate,
    /// Arbitrary key-value metadata (e.g. codec, container, resolution).
    pub metadata: HashMap<String, String>,
}

impl DetailedDuplicateGroup {
    /// Create a new group.
    #[must_use]
    pub fn new(id: usize, method: impl Into<String>, mean_similarity: f64) -> Self {
        let confidence_tier = ConfidenceTier::from_score(mean_similarity);
        Self {
            id,
            method: method.into(),
            files: Vec::new(),
            mean_similarity,
            confidence_tier,
            action: RecommendedAction::NoAction,
            space_savings: SpaceSavingsEstimate::default(),
            metadata: HashMap::new(),
        }
    }

    /// Add a file entry to this group.
    pub fn add_file(&mut self, entry: FileEntry) {
        self.files.push(entry);
    }

    /// Number of files.
    #[must_use]
    pub fn size(&self) -> usize {
        self.files.len()
    }

    /// True when the group has at least two members.
    #[must_use]
    pub fn is_duplicate(&self) -> bool {
        self.files.len() >= 2
    }

    /// Compute and cache the space savings estimate.
    pub fn compute_space_savings(&mut self) {
        self.space_savings = SpaceSavingsEstimate::from_entries(&self.files);
    }

    /// Assign a representative by picking the file with the largest size
    /// (on the assumption that higher-quality originals tend to be larger).
    pub fn select_largest_representative(&mut self) {
        if self.files.is_empty() {
            return;
        }
        // Clear existing representative flags.
        for f in &mut self.files {
            f.is_representative = false;
        }
        let best_idx = self
            .files
            .iter()
            .enumerate()
            .max_by_key(|(_, e)| e.size_bytes)
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.files[best_idx].is_representative = true;
    }

    /// Assign a representative by picking the file with the highest
    /// similarity_to_rep score (useful when the representative is
    /// externally determined).
    pub fn select_highest_similarity_representative(&mut self) {
        if self.files.is_empty() {
            return;
        }
        for f in &mut self.files {
            f.is_representative = false;
        }
        let best_idx = self
            .files
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.similarity_to_rep
                    .partial_cmp(&b.similarity_to_rep)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.files[best_idx].is_representative = true;
    }

    /// Assign the recommended action based on confidence tier and a policy.
    ///
    /// - `exact_action`: action when `confidence_tier == Exact`.
    /// - `high_action`: action when `confidence_tier == High`.
    /// - `fallback_action`: action for Medium/Low confidence.
    pub fn assign_action(
        &mut self,
        exact_action: RecommendedAction,
        high_action: RecommendedAction,
        fallback_action: RecommendedAction,
    ) {
        if !self.is_duplicate() {
            self.action = RecommendedAction::NoAction;
            return;
        }
        self.action = match self.confidence_tier {
            ConfidenceTier::Exact => exact_action,
            ConfidenceTier::High => high_action,
            _ => fallback_action,
        };
    }

    /// Insert a metadata key-value pair.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Return the path of the representative file, if one is marked.
    #[must_use]
    pub fn representative_path(&self) -> Option<&str> {
        self.files
            .iter()
            .find(|e| e.is_representative)
            .map(|e| e.path.as_str())
    }
}

// ---------------------------------------------------------------------------
// DetailedReport
// ---------------------------------------------------------------------------

/// Aggregate detailed report over all duplicate groups found.
#[derive(Debug, Clone)]
pub struct DetailedReport {
    /// All duplicate groups (size ≥ 2).
    pub groups: Vec<DetailedDuplicateGroup>,
    /// Aggregated space savings across all groups.
    pub total_space_savings: SpaceSavingsEstimate,
    /// Number of files examined in total.
    pub total_files_examined: usize,
    /// Breakdown of group counts by confidence tier.
    pub tier_counts: HashMap<String, usize>,
    /// Breakdown of group counts by recommended action.
    pub action_counts: HashMap<String, usize>,
}

impl DetailedReport {
    /// Number of duplicate groups.
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Total number of files that are in duplicate groups.
    #[must_use]
    pub fn duplicate_file_count(&self) -> usize {
        self.groups.iter().map(|g| g.size()).sum()
    }

    /// Return a multi-line human-readable summary.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "DetailedReport: {} groups | {} duplicate files | {} files examined\n\
             Space: {}\n\
             Tiers: {:?}\n\
             Actions: {:?}",
            self.group_count(),
            self.duplicate_file_count(),
            self.total_files_examined,
            self.total_space_savings.description(),
            self.tier_counts,
            self.action_counts,
        )
    }
}

// ---------------------------------------------------------------------------
// DetailedReportBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for assembling a [`DetailedReport`].
///
/// # Example
/// ```
/// use oximedia_dedup::dedup_report_detailed::{
///     DetailedReportBuilder, FileEntry, RecommendedAction,
/// };
///
/// let report = DetailedReportBuilder::new()
///     .total_files_examined(100)
///     .build();
///
/// assert_eq!(report.total_files_examined, 100);
/// assert!(report.groups.is_empty());
/// ```
#[derive(Debug, Default)]
pub struct DetailedReportBuilder {
    groups: Vec<DetailedDuplicateGroup>,
    total_files_examined: usize,
}

impl DetailedReportBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the total number of files examined.
    #[must_use]
    pub fn total_files_examined(mut self, n: usize) -> Self {
        self.total_files_examined = n;
        self
    }

    /// Add a pre-built [`DetailedDuplicateGroup`] to the report.
    #[must_use]
    pub fn add_group(mut self, group: DetailedDuplicateGroup) -> Self {
        self.groups.push(group);
        self
    }

    /// Convenience: add a simple group from paths and sizes.
    ///
    /// The group will use the largest-file-is-representative heuristic and
    /// the default action policy (Exact→delete, High→hardlink, else review).
    #[must_use]
    pub fn add_simple_group(
        mut self,
        id: usize,
        method: impl Into<String>,
        mean_similarity: f64,
        files: Vec<(String, u64)>,
    ) -> Self {
        let mut group = DetailedDuplicateGroup::new(id, method, mean_similarity);
        for (path, size) in files {
            group.add_file(FileEntry::new(path, size, mean_similarity));
        }
        group.select_largest_representative();
        group.assign_action(
            RecommendedAction::DeleteDuplicates,
            RecommendedAction::HardlinkDuplicates,
            RecommendedAction::ManualReview,
        );
        group.compute_space_savings();
        self.groups.push(group);
        self
    }

    /// Build the final [`DetailedReport`].
    #[must_use]
    pub fn build(self) -> DetailedReport {
        let mut total_space_savings = SpaceSavingsEstimate::default();
        let mut tier_counts: HashMap<String, usize> = HashMap::new();
        let mut action_counts: HashMap<String, usize> = HashMap::new();

        for group in &self.groups {
            total_space_savings.merge(&group.space_savings);
            *tier_counts
                .entry(group.confidence_tier.label().to_string())
                .or_insert(0) += 1;
            *action_counts
                .entry(group.action.label().to_string())
                .or_insert(0) += 1;
        }

        DetailedReport {
            groups: self.groups,
            total_space_savings,
            total_files_examined: self.total_files_examined,
            tier_counts,
            action_counts,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(path: &str, size: u64, sim: f64) -> FileEntry {
        FileEntry::new(path, size, sim)
    }

    #[test]
    fn test_confidence_tier_from_score() {
        assert_eq!(ConfidenceTier::from_score(1.0), ConfidenceTier::Exact);
        assert_eq!(ConfidenceTier::from_score(0.98), ConfidenceTier::Exact);
        assert_eq!(ConfidenceTier::from_score(0.95), ConfidenceTier::High);
        assert_eq!(ConfidenceTier::from_score(0.90), ConfidenceTier::High);
        assert_eq!(ConfidenceTier::from_score(0.80), ConfidenceTier::Medium);
        assert_eq!(ConfidenceTier::from_score(0.75), ConfidenceTier::Medium);
        assert_eq!(ConfidenceTier::from_score(0.50), ConfidenceTier::Low);
        assert_eq!(ConfidenceTier::from_score(0.0), ConfidenceTier::Low);
    }

    #[test]
    fn test_confidence_tier_ordering() {
        assert!(ConfidenceTier::Exact > ConfidenceTier::Low);
        assert!(ConfidenceTier::High > ConfidenceTier::Medium);
    }

    #[test]
    fn test_recommended_action_labels() {
        assert_eq!(RecommendedAction::DeleteDuplicates.label(), "delete");
        assert_eq!(RecommendedAction::SymlinkDuplicates.label(), "symlink");
        assert_eq!(RecommendedAction::HardlinkDuplicates.label(), "hardlink");
        assert_eq!(RecommendedAction::ArchiveDuplicates.label(), "archive");
        assert_eq!(RecommendedAction::ManualReview.label(), "manual_review");
        assert_eq!(RecommendedAction::NoAction.label(), "no_action");
    }

    #[test]
    fn test_space_savings_from_entries() {
        let mut files = vec![
            entry("a.mp4", 1000, 1.0),
            entry("b.mp4", 800, 0.95),
        ];
        files[0].is_representative = true;
        let est = SpaceSavingsEstimate::from_entries(&files);
        assert_eq!(est.retained_bytes, 1000);
        assert_eq!(est.reclaimable_bytes, 800);
        assert!((est.savings_ratio - 800.0 / 1800.0).abs() < 1e-9);
    }

    #[test]
    fn test_space_savings_empty() {
        let est = SpaceSavingsEstimate::from_entries(&[]);
        assert_eq!(est.reclaimable_bytes, 0);
        assert_eq!(est.savings_ratio, 0.0);
    }

    #[test]
    fn test_space_savings_merge() {
        let mut a = SpaceSavingsEstimate {
            reclaimable_bytes: 500,
            retained_bytes: 1000,
            savings_ratio: 0.333,
        };
        let b = SpaceSavingsEstimate {
            reclaimable_bytes: 300,
            retained_bytes: 700,
            savings_ratio: 0.3,
        };
        a.merge(&b);
        assert_eq!(a.reclaimable_bytes, 800);
        assert_eq!(a.retained_bytes, 1700);
        let expected_ratio = 800.0 / 2500.0;
        assert!((a.savings_ratio - expected_ratio).abs() < 1e-9);
    }

    #[test]
    fn test_group_select_largest_representative() {
        let mut group = DetailedDuplicateGroup::new(0, "phash", 0.95);
        group.add_file(entry("small.mp4", 100, 0.95));
        group.add_file(entry("large.mp4", 9000, 0.95));
        group.add_file(entry("medium.mp4", 500, 0.95));
        group.select_largest_representative();
        assert_eq!(group.representative_path(), Some("large.mp4"));
    }

    #[test]
    fn test_group_assign_action_exact() {
        let mut group = DetailedDuplicateGroup::new(0, "hash", 0.999);
        group.add_file(entry("a.mp4", 100, 1.0));
        group.add_file(entry("b.mp4", 100, 1.0));
        group.assign_action(
            RecommendedAction::DeleteDuplicates,
            RecommendedAction::HardlinkDuplicates,
            RecommendedAction::ManualReview,
        );
        assert_eq!(group.action, RecommendedAction::DeleteDuplicates);
    }

    #[test]
    fn test_group_assign_action_low_confidence() {
        let mut group = DetailedDuplicateGroup::new(0, "ssim", 0.65);
        group.add_file(entry("a.mp4", 100, 0.65));
        group.add_file(entry("b.mp4", 100, 0.65));
        group.assign_action(
            RecommendedAction::DeleteDuplicates,
            RecommendedAction::HardlinkDuplicates,
            RecommendedAction::ManualReview,
        );
        assert_eq!(group.action, RecommendedAction::ManualReview);
    }

    #[test]
    fn test_group_single_member_no_action() {
        let mut group = DetailedDuplicateGroup::new(0, "phash", 1.0);
        group.add_file(entry("only.mp4", 500, 1.0));
        group.assign_action(
            RecommendedAction::DeleteDuplicates,
            RecommendedAction::HardlinkDuplicates,
            RecommendedAction::ManualReview,
        );
        assert_eq!(group.action, RecommendedAction::NoAction);
    }

    #[test]
    fn test_group_metadata() {
        let mut group = DetailedDuplicateGroup::new(0, "phash", 0.95);
        group.set_metadata("codec", "h264");
        group.set_metadata("resolution", "1920x1080");
        assert_eq!(group.metadata.get("codec").map(String::as_str), Some("h264"));
        assert_eq!(group.metadata.len(), 2);
    }

    #[test]
    fn test_report_builder_empty() {
        let report = DetailedReportBuilder::new()
            .total_files_examined(50)
            .build();
        assert_eq!(report.total_files_examined, 50);
        assert!(report.groups.is_empty());
        assert_eq!(report.group_count(), 0);
        assert_eq!(report.duplicate_file_count(), 0);
    }

    #[test]
    fn test_report_builder_with_groups() {
        let report = DetailedReportBuilder::new()
            .total_files_examined(200)
            .add_simple_group(
                0,
                "phash",
                0.96,
                vec![
                    ("a.mp4".to_string(), 2000),
                    ("b.mp4".to_string(), 1500),
                ],
            )
            .add_simple_group(
                1,
                "ssim",
                0.82,
                vec![
                    ("c.mp4".to_string(), 1000),
                    ("d.mp4".to_string(), 900),
                ],
            )
            .build();

        assert_eq!(report.group_count(), 2);
        assert_eq!(report.duplicate_file_count(), 4);
        assert!(report.total_space_savings.reclaimable_bytes > 0);
        assert!(!report.summary().is_empty());
    }

    #[test]
    fn test_report_tier_and_action_counts() {
        let report = DetailedReportBuilder::new()
            .add_simple_group(
                0,
                "phash",
                0.999,
                vec![
                    ("a.mp4".to_string(), 1000),
                    ("b.mp4".to_string(), 800),
                ],
            )
            .add_simple_group(
                1,
                "ssim",
                0.60,
                vec![
                    ("c.mp4".to_string(), 500),
                    ("d.mp4".to_string(), 400),
                ],
            )
            .build();

        // Exact-confidence group → delete; Low-confidence → manual_review.
        assert_eq!(report.tier_counts.get("exact").copied().unwrap_or(0), 1);
        assert_eq!(report.tier_counts.get("low").copied().unwrap_or(0), 1);
        assert_eq!(report.action_counts.get("delete").copied().unwrap_or(0), 1);
        assert_eq!(
            report.action_counts.get("manual_review").copied().unwrap_or(0),
            1
        );
    }
}
