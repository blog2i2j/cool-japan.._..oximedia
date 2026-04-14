//! Disk space savings estimation for duplicate file groups.
//!
//! When duplicate files are found, the question arises: "how much storage can
//! I reclaim by eliminating redundant copies?"  This module answers that
//! question at various levels of granularity:
//!
//! - **Per-group savings**: the bytes that can be freed from a single
//!   duplicate group (size of all but the canonical copy).
//! - **Total savings**: sum of per-group savings across the entire library.
//! - **Savings breakdown**: bucketed by file extension for a quick view of
//!   *which formats* contribute most to storage waste.
//! - **Resolution strategy**: different strategies for choosing the canonical
//!   copy affect savings differently (e.g. keep the smallest vs largest).
//!
//! # Example
//!
//! ```
//! use oximedia_dedup::space_savings::{SpaceSavingsEstimator, RetentionStrategy};
//!
//! let mut estimator = SpaceSavingsEstimator::new(RetentionStrategy::KeepLargest);
//!
//! // Group 1: three copies of the same video
//! estimator.add_group(vec![
//!     ("video1.mp4".to_string(), 1_500_000_000),
//!     ("video1_copy.mp4".to_string(), 1_480_000_000),
//!     ("video1_backup.mp4".to_string(), 1_490_000_000),
//! ]);
//!
//! let report = estimator.estimate();
//! assert!(report.total_bytes_saved > 0);
//! ```

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// RetentionStrategy
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy for choosing which copy to retain when eliminating duplicates.
///
/// The strategy determines which file in a duplicate group is the "canonical"
/// copy that will be kept; all others count as recoverable savings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RetentionStrategy {
    /// Keep the largest file in each group (highest-quality encode).
    KeepLargest,
    /// Keep the smallest file in each group (most storage-efficient encode).
    KeepSmallest,
    /// Keep the first file encountered in each group (insertion order).
    KeepFirst,
    /// Keep the last file encountered in each group (insertion order).
    KeepLast,
    /// Calculate savings under the assumption that the canonical copy has
    /// the median size of the group.
    KeepMedianSized,
}

impl RetentionStrategy {
    /// Return a human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::KeepLargest => "keep-largest",
            Self::KeepSmallest => "keep-smallest",
            Self::KeepFirst => "keep-first",
            Self::KeepLast => "keep-last",
            Self::KeepMedianSized => "keep-median-sized",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GroupEntry / DuplicateGroupInfo
// ─────────────────────────────────────────────────────────────────────────────

/// A file entry within a duplicate group.
#[derive(Debug, Clone)]
pub struct GroupEntry {
    /// File path (or identifier).
    pub path: String,
    /// File size in bytes.
    pub size_bytes: u64,
}

impl GroupEntry {
    /// Create a new entry.
    #[must_use]
    pub fn new(path: String, size_bytes: u64) -> Self {
        Self { path, size_bytes }
    }

    /// Derive the file extension (lowercase, no dot) from the path.
    #[must_use]
    pub fn extension(&self) -> String {
        std::path::Path::new(&self.path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase()
    }
}

/// All information about a single duplicate group and the estimated savings.
#[derive(Debug, Clone)]
pub struct DuplicateGroupSavings {
    /// The files that would be removed.
    pub redundant_files: Vec<GroupEntry>,
    /// The file that would be kept (the canonical copy).
    pub canonical: GroupEntry,
    /// Total bytes that can be reclaimed by removing `redundant_files`.
    pub bytes_saved: u64,
    /// Percentage of the group's total storage that would be reclaimed.
    pub savings_percent: f64,
}

impl DuplicateGroupSavings {
    /// Human-readable summary line.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Keep '{}' ({} bytes); remove {} duplicate(s), saving {} bytes ({:.1}%)",
            self.canonical.path,
            self.canonical.size_bytes,
            self.redundant_files.len(),
            self.bytes_saved,
            self.savings_percent
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SpaceSavingsReport
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregated space savings report across the entire library.
#[derive(Debug, Clone)]
pub struct SpaceSavingsReport {
    /// Total bytes that can be reclaimed.
    pub total_bytes_saved: u64,
    /// Total files that would be removed.
    pub total_files_removed: usize,
    /// Number of duplicate groups analysed.
    pub total_groups: usize,
    /// Savings broken down by file extension.
    pub savings_by_extension: HashMap<String, u64>,
    /// Per-group details.
    pub group_details: Vec<DuplicateGroupSavings>,
    /// The retention strategy used for this estimate.
    pub strategy: RetentionStrategy,
}

impl SpaceSavingsReport {
    /// Format the total savings as a human-readable string (e.g. "1.23 GiB").
    #[must_use]
    pub fn total_saved_human(&self) -> String {
        format_bytes(self.total_bytes_saved)
    }

    /// Return the extension that accounts for the most wasted space.
    #[must_use]
    pub fn top_wasting_extension(&self) -> Option<(&str, u64)> {
        self.savings_by_extension
            .iter()
            .max_by_key(|(_, &v)| v)
            .map(|(k, &v)| (k.as_str(), v))
    }

    /// Overall savings as a percentage of all files' total size.
    ///
    /// Returns 0.0 if the total size tracked is zero.
    #[must_use]
    pub fn overall_savings_percent(&self, total_library_bytes: u64) -> f64 {
        if total_library_bytes == 0 {
            return 0.0;
        }
        self.total_bytes_saved as f64 / total_library_bytes as f64 * 100.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SpaceSavingsEstimator
// ─────────────────────────────────────────────────────────────────────────────

/// Accumulates duplicate groups and estimates storage savings.
///
/// # Example
///
/// ```
/// use oximedia_dedup::space_savings::{SpaceSavingsEstimator, RetentionStrategy};
///
/// let mut est = SpaceSavingsEstimator::new(RetentionStrategy::KeepLargest);
/// est.add_group(vec![
///     ("a.mp4".to_string(), 100_000),
///     ("b.mp4".to_string(), 120_000),
/// ]);
/// let report = est.estimate();
/// // The smaller file (a.mp4, 100 000 bytes) would be removed.
/// assert_eq!(report.total_bytes_saved, 100_000);
/// ```
#[derive(Debug)]
pub struct SpaceSavingsEstimator {
    strategy: RetentionStrategy,
    groups: Vec<Vec<GroupEntry>>,
}

impl SpaceSavingsEstimator {
    /// Create a new estimator with the specified retention strategy.
    #[must_use]
    pub fn new(strategy: RetentionStrategy) -> Self {
        Self {
            strategy,
            groups: Vec::new(),
        }
    }

    /// Add a duplicate group as `(path, size_bytes)` pairs.
    ///
    /// Groups with fewer than two entries are silently ignored.
    pub fn add_group(&mut self, entries: Vec<(String, u64)>) {
        if entries.len() < 2 {
            return;
        }
        let group: Vec<GroupEntry> = entries
            .into_iter()
            .map(|(p, s)| GroupEntry::new(p, s))
            .collect();
        self.groups.push(group);
    }

    /// Add a duplicate group as pre-built [`GroupEntry`] values.
    ///
    /// Groups with fewer than two entries are silently ignored.
    pub fn add_group_entries(&mut self, entries: Vec<GroupEntry>) {
        if entries.len() < 2 {
            return;
        }
        self.groups.push(entries);
    }

    /// Compute the savings estimate for all accumulated groups.
    #[must_use]
    pub fn estimate(&self) -> SpaceSavingsReport {
        let mut total_bytes_saved: u64 = 0;
        let mut total_files_removed: usize = 0;
        let mut savings_by_extension: HashMap<String, u64> = HashMap::new();
        let mut group_details: Vec<DuplicateGroupSavings> = Vec::new();

        for group in &self.groups {
            if group.is_empty() {
                continue;
            }
            let detail = compute_group_savings(group, self.strategy);
            total_bytes_saved = total_bytes_saved.saturating_add(detail.bytes_saved);
            total_files_removed += detail.redundant_files.len();

            // Accumulate by extension.
            for file in &detail.redundant_files {
                let ext = file.extension();
                *savings_by_extension.entry(ext).or_insert(0) += file.size_bytes;
            }

            group_details.push(detail);
        }

        SpaceSavingsReport {
            total_bytes_saved,
            total_files_removed,
            total_groups: self.groups.len(),
            savings_by_extension,
            group_details,
            strategy: self.strategy,
        }
    }

    /// Number of groups registered so far.
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Clear all accumulated groups.
    pub fn clear(&mut self) {
        self.groups.clear();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute savings for a single duplicate group under the given strategy.
fn compute_group_savings(entries: &[GroupEntry], strategy: RetentionStrategy) -> DuplicateGroupSavings {
    let canonical_idx = match strategy {
        RetentionStrategy::KeepLargest => entries
            .iter()
            .enumerate()
            .max_by_key(|(_, e)| e.size_bytes)
            .map(|(i, _)| i)
            .unwrap_or(0),
        RetentionStrategy::KeepSmallest => entries
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| e.size_bytes)
            .map(|(i, _)| i)
            .unwrap_or(0),
        RetentionStrategy::KeepFirst => 0,
        RetentionStrategy::KeepLast => entries.len() - 1,
        RetentionStrategy::KeepMedianSized => median_index(entries),
    };

    let canonical = entries[canonical_idx].clone();
    let total_size: u64 = entries.iter().map(|e| e.size_bytes).sum();
    let bytes_saved = total_size.saturating_sub(canonical.size_bytes);
    let savings_percent = if total_size > 0 {
        bytes_saved as f64 / total_size as f64 * 100.0
    } else {
        0.0
    };

    let redundant_files: Vec<GroupEntry> = entries
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != canonical_idx)
        .map(|(_, e)| e.clone())
        .collect();

    DuplicateGroupSavings {
        redundant_files,
        canonical,
        bytes_saved,
        savings_percent,
    }
}

/// Return the index of the entry with the median size.
///
/// For even-length slices the lower-median (index `n/2 - 1` after sorting) is
/// used to avoid introducing an average that may not correspond to a real file.
fn median_index(entries: &[GroupEntry]) -> usize {
    if entries.is_empty() {
        return 0;
    }
    let mut indexed: Vec<(usize, u64)> = entries
        .iter()
        .enumerate()
        .map(|(i, e)| (i, e.size_bytes))
        .collect();
    indexed.sort_by_key(|(_, s)| *s);
    let mid = if indexed.len() % 2 == 0 {
        indexed.len() / 2 - 1
    } else {
        indexed.len() / 2
    };
    indexed[mid].0
}

/// Format a byte count as a human-readable string with binary suffixes.
fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    let mut value = bytes as f64;
    let mut unit_idx = 0;
    while value >= 1024.0 && unit_idx + 1 < UNITS.len() {
        value /= 1024.0;
        unit_idx += 1;
    }
    if unit_idx == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.2} {}", UNITS[unit_idx])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_group(files: &[(&str, u64)]) -> Vec<(String, u64)> {
        files.iter().map(|(p, s)| (p.to_string(), *s)).collect()
    }

    #[test]
    fn test_keep_largest_removes_smaller() {
        let mut est = SpaceSavingsEstimator::new(RetentionStrategy::KeepLargest);
        est.add_group(make_group(&[("a.mp4", 100_000), ("b.mp4", 120_000)]));
        let report = est.estimate();
        assert_eq!(report.total_bytes_saved, 100_000);
        assert_eq!(report.total_files_removed, 1);
    }

    #[test]
    fn test_keep_smallest_removes_larger() {
        let mut est = SpaceSavingsEstimator::new(RetentionStrategy::KeepSmallest);
        est.add_group(make_group(&[("a.mp4", 200_000), ("b.mp4", 80_000)]));
        let report = est.estimate();
        assert_eq!(report.total_bytes_saved, 200_000);
        assert_eq!(report.total_files_removed, 1);
    }

    #[test]
    fn test_keep_first_strategy() {
        let mut est = SpaceSavingsEstimator::new(RetentionStrategy::KeepFirst);
        est.add_group(make_group(&[("a.mp4", 300), ("b.mp4", 100), ("c.mp4", 200)]));
        let report = est.estimate();
        // a.mp4 is kept; b.mp4 + c.mp4 = 300 bytes removed
        assert_eq!(report.total_bytes_saved, 300);
        assert_eq!(report.group_details[0].canonical.path, "a.mp4");
    }

    #[test]
    fn test_keep_last_strategy() {
        let mut est = SpaceSavingsEstimator::new(RetentionStrategy::KeepLast);
        est.add_group(make_group(&[("a.mp4", 300), ("b.mp4", 100), ("c.mp4", 200)]));
        let report = est.estimate();
        assert_eq!(report.group_details[0].canonical.path, "c.mp4");
        // a.mp4 + b.mp4 = 400 removed
        assert_eq!(report.total_bytes_saved, 400);
    }

    #[test]
    fn test_keep_median_three_files() {
        let mut est = SpaceSavingsEstimator::new(RetentionStrategy::KeepMedianSized);
        // Sizes: 100, 200, 300 → median = 200 (b.mp4)
        est.add_group(make_group(&[("a.mp4", 100), ("b.mp4", 200), ("c.mp4", 300)]));
        let report = est.estimate();
        assert_eq!(report.group_details[0].canonical.path, "b.mp4");
        assert_eq!(report.total_bytes_saved, 400); // 100 + 300
    }

    #[test]
    fn test_multiple_groups_sum() {
        let mut est = SpaceSavingsEstimator::new(RetentionStrategy::KeepLargest);
        est.add_group(make_group(&[("a.mp4", 100), ("b.mp4", 200)]));
        est.add_group(make_group(&[("c.avi", 50), ("d.avi", 80)]));
        let report = est.estimate();
        // Group 1: 100 saved; Group 2: 50 saved
        assert_eq!(report.total_bytes_saved, 150);
        assert_eq!(report.total_groups, 2);
        assert_eq!(report.total_files_removed, 2);
    }

    #[test]
    fn test_savings_by_extension() {
        let mut est = SpaceSavingsEstimator::new(RetentionStrategy::KeepLargest);
        est.add_group(make_group(&[("a.mp4", 100), ("b.mp4", 200)]));
        est.add_group(make_group(&[("c.avi", 50), ("d.avi", 80)]));
        let report = est.estimate();
        assert_eq!(report.savings_by_extension.get("mp4").copied(), Some(100));
        assert_eq!(report.savings_by_extension.get("avi").copied(), Some(50));
    }

    #[test]
    fn test_singleton_group_ignored() {
        let mut est = SpaceSavingsEstimator::new(RetentionStrategy::KeepLargest);
        est.add_group(make_group(&[("only.mp4", 500_000)]));
        let report = est.estimate();
        assert_eq!(report.total_groups, 0);
        assert_eq!(report.total_bytes_saved, 0);
    }

    #[test]
    fn test_empty_no_panic() {
        let est = SpaceSavingsEstimator::new(RetentionStrategy::KeepLargest);
        let report = est.estimate();
        assert_eq!(report.total_bytes_saved, 0);
        assert_eq!(report.total_files_removed, 0);
        assert_eq!(report.total_groups, 0);
    }

    #[test]
    fn test_savings_percent_calculation() {
        let entries = vec![
            GroupEntry::new("a.mp4".to_string(), 300),
            GroupEntry::new("b.mp4".to_string(), 100),
        ];
        let detail = compute_group_savings(&entries, RetentionStrategy::KeepLargest);
        // total = 400, saved = 100 → 25%
        assert!((detail.savings_percent - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_format_bytes_small() {
        assert_eq!(format_bytes(512), "512 B");
    }

    #[test]
    fn test_format_bytes_kib() {
        // 1024 bytes = 1 KiB
        assert_eq!(format_bytes(1024), "1.00 KiB");
    }

    #[test]
    fn test_format_bytes_gib() {
        // 1073741824 bytes = 1 GiB
        assert_eq!(format_bytes(1_073_741_824), "1.00 GiB");
    }

    #[test]
    fn test_top_wasting_extension() {
        let mut est = SpaceSavingsEstimator::new(RetentionStrategy::KeepLargest);
        // mp4 saves 1000, avi saves 50
        est.add_group(make_group(&[("a.mp4", 100), ("b.mp4", 1100)]));
        est.add_group(make_group(&[("c.avi", 10), ("d.avi", 60)]));
        let report = est.estimate();
        let (ext, _) = report.top_wasting_extension().expect("should have top");
        assert_eq!(ext, "mp4");
    }

    #[test]
    fn test_overall_savings_percent() {
        let mut est = SpaceSavingsEstimator::new(RetentionStrategy::KeepLargest);
        est.add_group(make_group(&[("a.mp4", 100), ("b.mp4", 300)]));
        let report = est.estimate();
        // Saved 100 bytes; library total 1000 bytes → 10%
        let pct = report.overall_savings_percent(1000);
        assert!((pct - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_group_summary_non_empty() {
        let entries = vec![
            GroupEntry::new("a.mp4".to_string(), 200),
            GroupEntry::new("b.mp4".to_string(), 100),
        ];
        let detail = compute_group_savings(&entries, RetentionStrategy::KeepLargest);
        let summary = detail.summary();
        assert!(summary.contains("a.mp4"));
        assert!(summary.contains("100"));
    }

    #[test]
    fn test_clear_resets_groups() {
        let mut est = SpaceSavingsEstimator::new(RetentionStrategy::KeepLargest);
        est.add_group(make_group(&[("a.mp4", 100), ("b.mp4", 200)]));
        assert_eq!(est.group_count(), 1);
        est.clear();
        assert_eq!(est.group_count(), 0);
        let report = est.estimate();
        assert_eq!(report.total_bytes_saved, 0);
    }

    #[test]
    fn test_retention_strategy_labels() {
        assert_eq!(RetentionStrategy::KeepLargest.label(), "keep-largest");
        assert_eq!(RetentionStrategy::KeepSmallest.label(), "keep-smallest");
        assert_eq!(RetentionStrategy::KeepFirst.label(), "keep-first");
        assert_eq!(RetentionStrategy::KeepLast.label(), "keep-last");
        assert_eq!(RetentionStrategy::KeepMedianSized.label(), "keep-median-sized");
    }
}
