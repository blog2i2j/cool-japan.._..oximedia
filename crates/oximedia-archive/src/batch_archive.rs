//! Batch archive operations for ingesting, verifying, and managing large
//! collections of media files.
//!
//! Supports job queuing, progress tracking, filtering by extension / size,
//! and summary reporting -- all without external runtime dependencies.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Status of a single batch item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemStatus {
    /// Waiting to be processed.
    Pending,
    /// Currently being processed.
    InProgress,
    /// Completed successfully.
    Success,
    /// Failed with an error.
    Failed,
    /// Skipped (e.g. filtered out).
    Skipped,
}

/// A single item in a batch job.
#[derive(Debug, Clone)]
pub struct BatchItem {
    /// Path to the source file.
    pub source: PathBuf,
    /// Size in bytes (0 if unknown).
    pub size_bytes: u64,
    /// Current status.
    pub status: ItemStatus,
    /// Error message if failed.
    pub error: Option<String>,
    /// Checksum computed during processing.
    pub checksum: Option<String>,
}

impl BatchItem {
    /// Create a new pending batch item.
    pub fn new(source: &Path, size_bytes: u64) -> Self {
        Self {
            source: source.to_path_buf(),
            size_bytes,
            status: ItemStatus::Pending,
            error: None,
            checksum: None,
        }
    }

    /// Mark as successfully processed.
    pub fn mark_success(&mut self, checksum: &str) {
        self.status = ItemStatus::Success;
        self.checksum = Some(checksum.to_string());
        self.error = None;
    }

    /// Mark as failed.
    pub fn mark_failed(&mut self, error: &str) {
        self.status = ItemStatus::Failed;
        self.error = Some(error.to_string());
    }

    /// Mark as skipped.
    pub fn mark_skipped(&mut self) {
        self.status = ItemStatus::Skipped;
    }
}

/// Filter criteria for selecting files in a batch.
#[derive(Debug, Clone)]
pub struct BatchFilter {
    /// Only include files with these extensions (empty = all).
    pub extensions: Vec<String>,
    /// Minimum file size in bytes (0 = no minimum).
    pub min_size: u64,
    /// Maximum file size in bytes (0 = no maximum).
    pub max_size: u64,
    /// Exclude files whose names contain any of these substrings.
    pub exclude_patterns: Vec<String>,
}

impl BatchFilter {
    /// Create a permissive filter that accepts everything.
    pub fn accept_all() -> Self {
        Self {
            extensions: Vec::new(),
            min_size: 0,
            max_size: 0,
            exclude_patterns: Vec::new(),
        }
    }

    /// Add an allowed extension (e.g. "mxf", "mov").
    pub fn with_extension(mut self, ext: &str) -> Self {
        self.extensions.push(ext.to_lowercase());
        self
    }

    /// Set minimum size in bytes.
    pub fn with_min_size(mut self, bytes: u64) -> Self {
        self.min_size = bytes;
        self
    }

    /// Set maximum size in bytes.
    pub fn with_max_size(mut self, bytes: u64) -> Self {
        self.max_size = bytes;
        self
    }

    /// Add an exclusion pattern.
    pub fn with_exclude(mut self, pattern: &str) -> Self {
        self.exclude_patterns.push(pattern.to_lowercase());
        self
    }

    /// Check if a file passes this filter.
    pub fn matches(&self, path: &Path, size_bytes: u64) -> bool {
        // Check extension
        if !self.extensions.is_empty() {
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .map(str::to_lowercase)
                .unwrap_or_default();
            if !self.extensions.contains(&ext) {
                return false;
            }
        }

        // Check size
        if self.min_size > 0 && size_bytes < self.min_size {
            return false;
        }
        if self.max_size > 0 && size_bytes > self.max_size {
            return false;
        }

        // Check exclusion patterns
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_lowercase();
        for pat in &self.exclude_patterns {
            if name.contains(pat) {
                return false;
            }
        }

        true
    }
}

impl Default for BatchFilter {
    fn default() -> Self {
        Self::accept_all()
    }
}

/// Summary statistics for a completed batch job.
#[derive(Debug, Clone)]
pub struct BatchSummary {
    /// Total number of items.
    pub total: usize,
    /// Number of successful items.
    pub success: usize,
    /// Number of failed items.
    pub failed: usize,
    /// Number of skipped items.
    pub skipped: usize,
    /// Number of pending items.
    pub pending: usize,
    /// Total bytes processed (success only).
    pub bytes_processed: u64,
    /// Per-extension counts.
    pub extension_counts: HashMap<String, usize>,
}

/// A batch archive job that tracks a collection of items.
#[derive(Debug)]
pub struct BatchJob {
    /// Unique job identifier.
    pub id: String,
    /// Human-readable label.
    pub label: String,
    /// Items in the batch.
    items: Vec<BatchItem>,
    /// Filter applied to new items.
    filter: BatchFilter,
}

impl BatchJob {
    /// Create a new batch job.
    pub fn new(id: &str, label: &str) -> Self {
        Self {
            id: id.to_string(),
            label: label.to_string(),
            items: Vec::new(),
            filter: BatchFilter::accept_all(),
        }
    }

    /// Set the filter for this job.
    pub fn with_filter(mut self, filter: BatchFilter) -> Self {
        self.filter = filter;
        self
    }

    /// Add a file to the batch. Returns true if accepted by the filter.
    pub fn add(&mut self, path: &Path, size_bytes: u64) -> bool {
        if !self.filter.matches(path, size_bytes) {
            return false;
        }
        self.items.push(BatchItem::new(path, size_bytes));
        true
    }

    /// Number of items in the batch.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns `true` if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Get a mutable reference to an item by index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut BatchItem> {
        self.items.get_mut(index)
    }

    /// Get a reference to an item by index.
    pub fn get(&self, index: usize) -> Option<&BatchItem> {
        self.items.get(index)
    }

    /// Iterate over all items.
    pub fn iter(&self) -> impl Iterator<Item = &BatchItem> {
        self.items.iter()
    }

    /// Process all pending items using the provided callback.
    /// The callback receives the source path and returns `Ok(checksum)` or `Err(message)`.
    pub fn process<F>(&mut self, mut processor: F)
    where
        F: FnMut(&Path) -> Result<String, String>,
    {
        for item in &mut self.items {
            if item.status != ItemStatus::Pending {
                continue;
            }
            item.status = ItemStatus::InProgress;
            match processor(&item.source) {
                Ok(checksum) => item.mark_success(&checksum),
                Err(msg) => item.mark_failed(&msg),
            }
        }
    }

    /// Compute a summary of the current batch state.
    #[allow(clippy::cast_precision_loss)]
    pub fn summary(&self) -> BatchSummary {
        let mut s = BatchSummary {
            total: self.items.len(),
            success: 0,
            failed: 0,
            skipped: 0,
            pending: 0,
            bytes_processed: 0,
            extension_counts: HashMap::new(),
        };

        for item in &self.items {
            match item.status {
                ItemStatus::Success => {
                    s.success += 1;
                    s.bytes_processed += item.size_bytes;
                }
                ItemStatus::Failed => s.failed += 1,
                ItemStatus::Skipped => s.skipped += 1,
                ItemStatus::Pending | ItemStatus::InProgress => s.pending += 1,
            }
            let ext = item
                .source
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("none")
                .to_lowercase();
            *s.extension_counts.entry(ext).or_insert(0) += 1;
        }

        s
    }

    /// Return indices of all failed items.
    pub fn failed_indices(&self) -> Vec<usize> {
        self.items
            .iter()
            .enumerate()
            .filter(|(_, it)| it.status == ItemStatus::Failed)
            .map(|(i, _)| i)
            .collect()
    }

    /// Retry all failed items (reset to Pending).
    pub fn retry_failed(&mut self) -> usize {
        let mut count = 0;
        for item in &mut self.items {
            if item.status == ItemStatus::Failed {
                item.status = ItemStatus::Pending;
                item.error = None;
                count += 1;
            }
        }
        count
    }

    /// Completion ratio as a float in [0.0, 1.0].
    #[allow(clippy::cast_precision_loss)]
    pub fn completion_ratio(&self) -> f64 {
        if self.items.is_empty() {
            return 1.0;
        }
        let done = self
            .items
            .iter()
            .filter(|i| {
                matches!(
                    i.status,
                    ItemStatus::Success | ItemStatus::Failed | ItemStatus::Skipped
                )
            })
            .count();
        done as f64 / self.items.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn mxf(name: &str) -> PathBuf {
        PathBuf::from(format!("/archive/{name}.mxf"))
    }

    #[test]
    fn test_batch_item_creation() {
        let item = BatchItem::new(Path::new("/a.mxf"), 1000);
        assert_eq!(item.status, ItemStatus::Pending);
        assert!(item.error.is_none());
    }

    #[test]
    fn test_batch_item_mark_success() {
        let mut item = BatchItem::new(Path::new("/a.mxf"), 1000);
        item.mark_success("abc123");
        assert_eq!(item.status, ItemStatus::Success);
        assert_eq!(item.checksum.as_deref(), Some("abc123"));
    }

    #[test]
    fn test_batch_item_mark_failed() {
        let mut item = BatchItem::new(Path::new("/a.mxf"), 1000);
        item.mark_failed("IO error");
        assert_eq!(item.status, ItemStatus::Failed);
        assert_eq!(item.error.as_deref(), Some("IO error"));
    }

    #[test]
    fn test_filter_accept_all() {
        let f = BatchFilter::accept_all();
        assert!(f.matches(Path::new("/a.mxf"), 100));
        assert!(f.matches(Path::new("/b.mov"), 0));
    }

    #[test]
    fn test_filter_extension() {
        let f = BatchFilter::accept_all().with_extension("mxf");
        assert!(f.matches(Path::new("/a.mxf"), 100));
        assert!(!f.matches(Path::new("/a.mov"), 100));
    }

    #[test]
    fn test_filter_size_range() {
        let f = BatchFilter::accept_all()
            .with_min_size(100)
            .with_max_size(500);
        assert!(f.matches(Path::new("/a.mxf"), 200));
        assert!(!f.matches(Path::new("/a.mxf"), 50));
        assert!(!f.matches(Path::new("/a.mxf"), 600));
    }

    #[test]
    fn test_filter_exclude_pattern() {
        let f = BatchFilter::accept_all().with_exclude("thumb");
        assert!(!f.matches(Path::new("/thumbnail.mxf"), 100));
        assert!(f.matches(Path::new("/master.mxf"), 100));
    }

    #[test]
    fn test_batch_job_add() {
        let mut job = BatchJob::new("j1", "Test batch");
        assert!(job.add(&mxf("clip1"), 1000));
        assert_eq!(job.len(), 1);
    }

    #[test]
    fn test_batch_job_filter_rejects() {
        let filter = BatchFilter::accept_all().with_extension("mxf");
        let mut job = BatchJob::new("j1", "Test").with_filter(filter);
        assert!(!job.add(Path::new("/a.mov"), 100));
        assert!(job.is_empty());
    }

    #[test]
    fn test_batch_process() {
        let mut job = BatchJob::new("j1", "Test");
        job.add(&mxf("a"), 100);
        job.add(&mxf("b"), 200);
        job.process(|_| Ok("checksum".to_string()));
        let s = job.summary();
        assert_eq!(s.success, 2);
        assert_eq!(s.failed, 0);
    }

    #[test]
    fn test_batch_process_failure() {
        let mut job = BatchJob::new("j1", "Test");
        job.add(&mxf("a"), 100);
        job.process(|_| Err("disk full".to_string()));
        let s = job.summary();
        assert_eq!(s.failed, 1);
        assert_eq!(s.success, 0);
    }

    #[test]
    fn test_summary_bytes_processed() {
        let mut job = BatchJob::new("j1", "Test");
        job.add(&mxf("a"), 100);
        job.add(&mxf("b"), 200);
        job.process(|_| Ok("cs".to_string()));
        assert_eq!(job.summary().bytes_processed, 300);
    }

    #[test]
    fn test_failed_indices() {
        let mut job = BatchJob::new("j1", "Test");
        job.add(&mxf("a"), 100);
        job.add(&mxf("b"), 200);
        // Fail the first, succeed the second
        let mut i = 0;
        job.process(|_| {
            i += 1;
            if i == 1 {
                Err("err".to_string())
            } else {
                Ok("ok".to_string())
            }
        });
        assert_eq!(job.failed_indices(), vec![0]);
    }

    #[test]
    fn test_retry_failed() {
        let mut job = BatchJob::new("j1", "Test");
        job.add(&mxf("a"), 100);
        job.process(|_| Err("err".to_string()));
        let retried = job.retry_failed();
        assert_eq!(retried, 1);
        assert_eq!(
            job.get(0).expect("get should succeed").status,
            ItemStatus::Pending
        );
    }

    #[test]
    fn test_completion_ratio_empty() {
        let job = BatchJob::new("j1", "Test");
        assert!((job.completion_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_completion_ratio_partial() {
        let mut job = BatchJob::new("j1", "Test");
        job.add(&mxf("a"), 100);
        job.add(&mxf("b"), 200);
        // Process only one (succeed first, skip second by only processing once)
        if let Some(item) = job.get_mut(0) {
            item.mark_success("cs");
        }
        assert!((job.completion_ratio() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_extension_counts() {
        let mut job = BatchJob::new("j1", "Test");
        job.add(Path::new("/a.mxf"), 100);
        job.add(Path::new("/b.mxf"), 200);
        job.add(Path::new("/c.mov"), 300);
        let s = job.summary();
        assert_eq!(s.extension_counts.get("mxf"), Some(&2));
        assert_eq!(s.extension_counts.get("mov"), Some(&1));
    }
}
