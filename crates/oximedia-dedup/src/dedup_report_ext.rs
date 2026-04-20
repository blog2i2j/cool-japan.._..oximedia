//! Extended deduplication reporting and statistics.
//!
//! Augments the base [`report`](crate::report) module with:
//! - **`ReportBuilder`**: fluent builder for assembling detailed reports
//! - **`SizeDistribution`**: histogram of duplicate file sizes
//! - **`FormatBreakdown`**: per-format duplicate statistics
//! - **`ReportSummary`**: human-readable summary generation

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// DuplicateEntry
// ---------------------------------------------------------------------------

/// A single duplicate file entry for reporting.
#[derive(Debug, Clone)]
pub struct DuplicateEntry {
    /// Path to the file.
    pub path: PathBuf,
    /// File size in bytes.
    pub size: u64,
    /// Hash digest.
    pub digest: String,
    /// File extension (lowercase, no dot).
    pub extension: String,
}

impl DuplicateEntry {
    /// Create a new entry.
    pub fn new(path: PathBuf, size: u64, digest: &str) -> Self {
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        Self {
            path,
            size,
            digest: digest.to_string(),
            extension,
        }
    }
}

// ---------------------------------------------------------------------------
// ReportBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing an extended dedup report.
pub struct ReportBuilder {
    /// All duplicate entries.
    entries: Vec<DuplicateEntry>,
    /// Title of the report.
    title: String,
    /// Minimum group size to include.
    min_group_size: usize,
    /// Only include files larger than this threshold.
    min_file_size: u64,
}

impl ReportBuilder {
    /// Start building a new report.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            title: "Deduplication Report".to_string(),
            min_group_size: 2,
            min_file_size: 0,
        }
    }

    /// Set the report title.
    #[must_use]
    pub fn title(mut self, title: &str) -> Self {
        self.title = title.to_string();
        self
    }

    /// Set minimum duplicate group size to include.
    #[must_use]
    pub fn min_group_size(mut self, n: usize) -> Self {
        self.min_group_size = n;
        self
    }

    /// Set minimum file size filter.
    #[must_use]
    pub fn min_file_size(mut self, bytes: u64) -> Self {
        self.min_file_size = bytes;
        self
    }

    /// Add a duplicate entry.
    pub fn add_entry(&mut self, entry: DuplicateEntry) {
        self.entries.push(entry);
    }

    /// Add multiple entries.
    pub fn add_entries(&mut self, entries: impl IntoIterator<Item = DuplicateEntry>) {
        self.entries.extend(entries);
    }

    /// Build the final report.
    #[must_use]
    pub fn build(self) -> ExtendedReport {
        // Group entries by digest
        let mut groups: HashMap<String, Vec<DuplicateEntry>> = HashMap::new();
        for entry in self.entries {
            if entry.size >= self.min_file_size {
                groups.entry(entry.digest.clone()).or_default().push(entry);
            }
        }

        // Filter by min group size
        let dup_groups: Vec<DuplicateGroup> = groups
            .into_iter()
            .filter(|(_, v)| v.len() >= self.min_group_size)
            .map(|(digest, files)| {
                let total_size: u64 = files.iter().map(|f| f.size).sum();
                let recoverable = files.iter().skip(1).map(|f| f.size).sum();
                DuplicateGroup {
                    digest,
                    files,
                    total_size,
                    recoverable_bytes: recoverable,
                }
            })
            .collect();

        let total_files: usize = dup_groups.iter().map(|g| g.files.len()).sum();
        let total_recoverable: u64 = dup_groups.iter().map(|g| g.recoverable_bytes).sum();

        ExtendedReport {
            title: self.title,
            groups: dup_groups,
            total_duplicate_files: total_files,
            total_recoverable_bytes: total_recoverable,
        }
    }
}

impl Default for ReportBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// DuplicateGroup / ExtendedReport
// ---------------------------------------------------------------------------

/// A group of duplicate files sharing the same digest.
#[derive(Debug, Clone)]
pub struct DuplicateGroup {
    /// The shared digest.
    pub digest: String,
    /// Files in this group.
    pub files: Vec<DuplicateEntry>,
    /// Total size of all files in the group.
    pub total_size: u64,
    /// Bytes recoverable by keeping only one copy.
    pub recoverable_bytes: u64,
}

/// Full extended deduplication report.
#[derive(Debug, Clone)]
pub struct ExtendedReport {
    /// Report title.
    pub title: String,
    /// Duplicate groups.
    pub groups: Vec<DuplicateGroup>,
    /// Total number of duplicate files.
    pub total_duplicate_files: usize,
    /// Total bytes recoverable.
    pub total_recoverable_bytes: u64,
}

impl ExtendedReport {
    /// Return the number of duplicate groups.
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Return a per-extension breakdown of duplicates.
    #[must_use]
    pub fn format_breakdown(&self) -> FormatBreakdown {
        let mut by_ext: HashMap<String, ExtStats> = HashMap::new();

        for group in &self.groups {
            for file in &group.files {
                let ext = if file.extension.is_empty() {
                    "(none)".to_string()
                } else {
                    file.extension.clone()
                };
                let stats = by_ext.entry(ext).or_insert_with(ExtStats::default);
                stats.file_count += 1;
                stats.total_bytes += file.size;
            }
        }

        FormatBreakdown {
            by_extension: by_ext,
        }
    }

    /// Return a human-readable summary string.
    #[must_use]
    pub fn summary_text(&self) -> String {
        format!(
            "{}: {} duplicate groups, {} files, {:.2} MB recoverable",
            self.title,
            self.groups.len(),
            self.total_duplicate_files,
            self.total_recoverable_bytes as f64 / (1024.0 * 1024.0),
        )
    }

    /// Build a size distribution histogram with the given bucket boundaries.
    #[must_use]
    pub fn size_distribution(&self, bucket_boundaries: &[u64]) -> SizeDistribution {
        let mut buckets = vec![0u64; bucket_boundaries.len() + 1];

        for group in &self.groups {
            for file in &group.files {
                let idx = bucket_boundaries
                    .iter()
                    .position(|&b| file.size < b)
                    .unwrap_or(bucket_boundaries.len());
                buckets[idx] += 1;
            }
        }

        SizeDistribution {
            boundaries: bucket_boundaries.to_vec(),
            counts: buckets,
        }
    }

    /// Filter groups, keeping only those containing a file under `prefix`.
    #[must_use]
    pub fn filter_by_path(&self, prefix: &Path) -> Vec<&DuplicateGroup> {
        self.groups
            .iter()
            .filter(|g| g.files.iter().any(|f| f.path.starts_with(prefix)))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// FormatBreakdown
// ---------------------------------------------------------------------------

/// Per-extension statistics.
#[derive(Debug, Clone, Default)]
pub struct ExtStats {
    /// Number of files.
    pub file_count: usize,
    /// Total bytes.
    pub total_bytes: u64,
}

/// Duplicate statistics broken down by file extension.
#[derive(Debug, Clone)]
pub struct FormatBreakdown {
    /// Map from extension to stats.
    pub by_extension: HashMap<String, ExtStats>,
}

impl FormatBreakdown {
    /// Return the extension with the most duplicate files.
    #[must_use]
    pub fn most_common_ext(&self) -> Option<(&str, usize)> {
        self.by_extension
            .iter()
            .max_by_key(|(_, s)| s.file_count)
            .map(|(ext, s)| (ext.as_str(), s.file_count))
    }
}

// ---------------------------------------------------------------------------
// SizeDistribution
// ---------------------------------------------------------------------------

/// Histogram of file sizes.
#[derive(Debug, Clone)]
pub struct SizeDistribution {
    /// Bucket boundaries (upper-exclusive).
    pub boundaries: Vec<u64>,
    /// Count of files in each bucket.
    pub counts: Vec<u64>,
}

impl SizeDistribution {
    /// Return total file count across all buckets.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.counts.iter().sum()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entries() -> Vec<DuplicateEntry> {
        vec![
            DuplicateEntry::new(PathBuf::from("/a.mp4"), 1_000_000, "hash1"),
            DuplicateEntry::new(PathBuf::from("/b.mp4"), 1_000_000, "hash1"),
            DuplicateEntry::new(PathBuf::from("/c.mov"), 500_000, "hash2"),
            DuplicateEntry::new(PathBuf::from("/d.mov"), 500_000, "hash2"),
            DuplicateEntry::new(PathBuf::from("/e.wav"), 200_000, "hash3"),
        ]
    }

    #[test]
    fn test_report_builder_basic() {
        let mut builder = ReportBuilder::new();
        builder.add_entries(sample_entries());
        let report = builder.build();
        assert_eq!(report.group_count(), 2); // hash1 and hash2 have groups >= 2
    }

    #[test]
    fn test_report_builder_title() {
        let report = ReportBuilder::new().title("My Report").build();
        assert_eq!(report.title, "My Report");
    }

    #[test]
    fn test_report_builder_min_group_size() {
        let mut builder = ReportBuilder::new().min_group_size(3);
        builder.add_entries(sample_entries());
        let report = builder.build();
        assert_eq!(report.group_count(), 0); // no group has 3+ files
    }

    #[test]
    fn test_report_builder_min_file_size() {
        let mut builder = ReportBuilder::new().min_file_size(600_000);
        builder.add_entries(sample_entries());
        let report = builder.build();
        // Only hash1 group (1MB each) passes
        assert_eq!(report.group_count(), 1);
    }

    #[test]
    fn test_recoverable_bytes() {
        let mut builder = ReportBuilder::new();
        builder.add_entries(sample_entries());
        let report = builder.build();
        // hash1: 1M recoverable, hash2: 500k recoverable
        assert_eq!(report.total_recoverable_bytes, 1_500_000);
    }

    #[test]
    fn test_summary_text() {
        let mut builder = ReportBuilder::new().title("Test");
        builder.add_entries(sample_entries());
        let report = builder.build();
        let text = report.summary_text();
        assert!(text.contains("Test"));
        assert!(text.contains("duplicate groups"));
    }

    #[test]
    fn test_format_breakdown() {
        let mut builder = ReportBuilder::new();
        builder.add_entries(sample_entries());
        let report = builder.build();
        let breakdown = report.format_breakdown();
        assert!(breakdown.by_extension.contains_key("mp4"));
        assert!(breakdown.by_extension.contains_key("mov"));
    }

    #[test]
    fn test_most_common_ext() {
        let mut builder = ReportBuilder::new();
        builder.add_entries(sample_entries());
        let report = builder.build();
        let breakdown = report.format_breakdown();
        let (ext, count) = breakdown
            .most_common_ext()
            .expect("operation should succeed");
        // mp4 and mov both have 2 files; either is acceptable
        assert!(count >= 2);
        assert!(ext == "mp4" || ext == "mov");
    }

    #[test]
    fn test_size_distribution() {
        let mut builder = ReportBuilder::new();
        builder.add_entries(sample_entries());
        let report = builder.build();
        let dist = report.size_distribution(&[100_000, 750_000, 2_000_000]);
        assert_eq!(dist.total(), 4); // 4 files in 2 groups
    }

    #[test]
    fn test_filter_by_path() {
        let entries = vec![
            DuplicateEntry::new(PathBuf::from("/archive/a.mp4"), 100, "h1"),
            DuplicateEntry::new(PathBuf::from("/other/b.mp4"), 100, "h1"),
        ];
        let mut builder = ReportBuilder::new();
        builder.add_entries(entries);
        let report = builder.build();
        let filtered = report.filter_by_path(Path::new("/archive"));
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_empty_report() {
        let report = ReportBuilder::new().build();
        assert_eq!(report.group_count(), 0);
        assert_eq!(report.total_duplicate_files, 0);
        assert_eq!(report.total_recoverable_bytes, 0);
    }

    #[test]
    fn test_duplicate_entry_extension() {
        let e = DuplicateEntry::new(PathBuf::from("/foo.MP4"), 0, "x");
        assert_eq!(e.extension, "mp4");
    }
}
