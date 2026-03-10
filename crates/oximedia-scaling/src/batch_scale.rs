#![allow(dead_code)]
//! Batch scaling operations with progress tracking.
//!
//! Provides infrastructure for scaling multiple images or video frames
//! in batch with configurable parallelism, progress reporting, and
//! error handling policies.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Error handling policy for batch operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorPolicy {
    /// Stop the entire batch on the first error.
    StopOnFirst,
    /// Skip failed items and continue processing.
    SkipAndContinue,
    /// Retry failed items up to the configured retry count.
    RetryThenSkip,
}

impl fmt::Display for ErrorPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StopOnFirst => write!(f, "StopOnFirst"),
            Self::SkipAndContinue => write!(f, "SkipAndContinue"),
            Self::RetryThenSkip => write!(f, "RetryThenSkip"),
        }
    }
}

/// Status of a single batch item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ItemStatus {
    /// Item is pending processing.
    Pending,
    /// Item is currently being processed.
    InProgress,
    /// Item completed successfully.
    Completed,
    /// Item failed with an error.
    Failed(String),
    /// Item was skipped.
    Skipped,
}

impl fmt::Display for ItemStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending => write!(f, "Pending"),
            Self::InProgress => write!(f, "InProgress"),
            Self::Completed => write!(f, "Completed"),
            Self::Failed(e) => write!(f, "Failed: {}", e),
            Self::Skipped => write!(f, "Skipped"),
        }
    }
}

/// A single item in a batch scaling job.
#[derive(Debug, Clone)]
pub struct BatchItem {
    /// Source file path.
    pub source: PathBuf,
    /// Destination file path.
    pub destination: PathBuf,
    /// Target width.
    pub target_width: u32,
    /// Target height.
    pub target_height: u32,
    /// Current status.
    pub status: ItemStatus,
    /// Number of retry attempts so far.
    pub retry_count: u32,
}

impl BatchItem {
    /// Creates a new batch item.
    pub fn new(
        source: PathBuf,
        destination: PathBuf,
        target_width: u32,
        target_height: u32,
    ) -> Self {
        Self {
            source,
            destination,
            target_width,
            target_height,
            status: ItemStatus::Pending,
            retry_count: 0,
        }
    }

    /// Returns true if this item is still pending.
    pub fn is_pending(&self) -> bool {
        self.status == ItemStatus::Pending
    }

    /// Returns true if this item completed successfully.
    pub fn is_completed(&self) -> bool {
        self.status == ItemStatus::Completed
    }

    /// Returns true if this item failed.
    pub fn is_failed(&self) -> bool {
        matches!(self.status, ItemStatus::Failed(_))
    }

    /// Marks the item as in progress.
    pub fn mark_in_progress(&mut self) {
        self.status = ItemStatus::InProgress;
    }

    /// Marks the item as completed.
    pub fn mark_completed(&mut self) {
        self.status = ItemStatus::Completed;
    }

    /// Marks the item as failed.
    pub fn mark_failed(&mut self, error: &str) {
        self.status = ItemStatus::Failed(error.to_string());
    }

    /// Marks the item as skipped.
    pub fn mark_skipped(&mut self) {
        self.status = ItemStatus::Skipped;
    }

    /// Increments the retry count.
    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
    }
}

/// Configuration for a batch scaling job.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of parallel workers.
    pub max_parallel: usize,
    /// Error handling policy.
    pub error_policy: ErrorPolicy,
    /// Maximum number of retries per item.
    pub max_retries: u32,
    /// Whether to overwrite existing destination files.
    pub overwrite_existing: bool,
    /// Whether to preserve source file metadata.
    pub preserve_metadata: bool,
    /// Optional output format override.
    pub output_format: Option<String>,
}

impl BatchConfig {
    /// Creates a new batch configuration.
    pub fn new() -> Self {
        Self {
            max_parallel: 4,
            error_policy: ErrorPolicy::SkipAndContinue,
            max_retries: 2,
            overwrite_existing: false,
            preserve_metadata: true,
            output_format: None,
        }
    }

    /// Sets the maximum parallel workers.
    pub fn with_max_parallel(mut self, n: usize) -> Self {
        self.max_parallel = n.max(1);
        self
    }

    /// Sets the error policy.
    pub fn with_error_policy(mut self, policy: ErrorPolicy) -> Self {
        self.error_policy = policy;
        self
    }

    /// Sets the maximum retries.
    pub fn with_max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    /// Sets whether to overwrite existing files.
    pub fn with_overwrite(mut self, overwrite: bool) -> Self {
        self.overwrite_existing = overwrite;
        self
    }

    /// Sets the output format.
    pub fn with_output_format(mut self, format: &str) -> Self {
        self.output_format = Some(format.to_string());
        self
    }
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Atomic progress counter for thread-safe tracking.
#[derive(Debug)]
pub struct ProgressTracker {
    /// Total number of items.
    total: u64,
    /// Number of completed items.
    completed: Arc<AtomicU64>,
    /// Number of failed items.
    failed: Arc<AtomicU64>,
    /// Number of skipped items.
    skipped: Arc<AtomicU64>,
}

impl ProgressTracker {
    /// Creates a new progress tracker.
    pub fn new(total: u64) -> Self {
        Self {
            total,
            completed: Arc::new(AtomicU64::new(0)),
            failed: Arc::new(AtomicU64::new(0)),
            skipped: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Records a completed item.
    pub fn record_completed(&self) {
        self.completed.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a failed item.
    pub fn record_failed(&self) {
        self.failed.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a skipped item.
    pub fn record_skipped(&self) {
        self.skipped.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns the total count.
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Returns the number completed.
    pub fn completed(&self) -> u64 {
        self.completed.load(Ordering::Relaxed)
    }

    /// Returns the number failed.
    pub fn failed(&self) -> u64 {
        self.failed.load(Ordering::Relaxed)
    }

    /// Returns the number skipped.
    pub fn skipped(&self) -> u64 {
        self.skipped.load(Ordering::Relaxed)
    }

    /// Returns the number of items still remaining.
    pub fn remaining(&self) -> u64 {
        let processed = self.completed() + self.failed() + self.skipped();
        self.total.saturating_sub(processed)
    }

    /// Returns the completion percentage (0.0 to 100.0).
    #[allow(clippy::cast_precision_loss)]
    pub fn progress_percent(&self) -> f64 {
        if self.total == 0 {
            return 100.0;
        }
        let processed = self.completed() + self.failed() + self.skipped();
        (processed as f64 / self.total as f64) * 100.0
    }

    /// Returns true if all items have been processed.
    pub fn is_finished(&self) -> bool {
        self.remaining() == 0
    }
}

/// Summary of a completed batch scaling job.
#[derive(Debug, Clone)]
pub struct BatchSummary {
    /// Total items in the batch.
    pub total: u64,
    /// Successfully completed items.
    pub completed: u64,
    /// Failed items.
    pub failed: u64,
    /// Skipped items.
    pub skipped: u64,
    /// Per-item error messages for failed items.
    pub errors: HashMap<String, String>,
}

impl BatchSummary {
    /// Creates a new summary.
    pub fn new(total: u64, completed: u64, failed: u64, skipped: u64) -> Self {
        Self {
            total,
            completed,
            failed,
            skipped,
            errors: HashMap::new(),
        }
    }

    /// Returns the success rate as a percentage.
    #[allow(clippy::cast_precision_loss)]
    pub fn success_rate(&self) -> f64 {
        if self.total == 0 {
            return 100.0;
        }
        (self.completed as f64 / self.total as f64) * 100.0
    }

    /// Returns true if all items were processed successfully.
    pub fn all_succeeded(&self) -> bool {
        self.failed == 0 && self.skipped == 0
    }

    /// Adds an error entry.
    pub fn add_error(&mut self, path: &str, error: &str) {
        self.errors.insert(path.to_string(), error.to_string());
    }
}

impl fmt::Display for BatchSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Batch: {}/{} completed, {} failed, {} skipped ({:.1}% success)",
            self.completed,
            self.total,
            self.failed,
            self.skipped,
            self.success_rate()
        )
    }
}

/// Batch scaling job that manages a collection of items.
#[derive(Debug)]
pub struct BatchScaleJob {
    /// Items in the batch.
    items: Vec<BatchItem>,
    /// Configuration.
    config: BatchConfig,
    /// Progress tracker.
    progress: ProgressTracker,
}

impl BatchScaleJob {
    /// Creates a new batch scaling job.
    pub fn new(items: Vec<BatchItem>, config: BatchConfig) -> Self {
        let total = items.len() as u64;
        Self {
            items,
            config,
            progress: ProgressTracker::new(total),
        }
    }

    /// Returns the number of items.
    pub fn item_count(&self) -> usize {
        self.items.len()
    }

    /// Returns the configuration.
    pub fn config(&self) -> &BatchConfig {
        &self.config
    }

    /// Returns the progress tracker.
    pub fn progress(&self) -> &ProgressTracker {
        &self.progress
    }

    /// Returns a reference to the items.
    pub fn items(&self) -> &[BatchItem] {
        &self.items
    }

    /// Returns a mutable reference to the items.
    pub fn items_mut(&mut self) -> &mut [BatchItem] {
        &mut self.items
    }

    /// Simulates processing all items (for testing/demonstration).
    pub fn process_all(&mut self) -> BatchSummary {
        let mut errors = HashMap::new();

        for item in &mut self.items {
            item.mark_in_progress();
            // Simulate: items with "fail" in the name fail
            let should_fail = item.source.to_string_lossy().contains("fail");

            if should_fail {
                match self.config.error_policy {
                    ErrorPolicy::StopOnFirst => {
                        item.mark_failed("simulated failure");
                        self.progress.record_failed();
                        errors.insert(
                            item.source.to_string_lossy().to_string(),
                            "simulated failure".to_string(),
                        );
                        break;
                    }
                    ErrorPolicy::SkipAndContinue => {
                        item.mark_failed("simulated failure");
                        self.progress.record_failed();
                        errors.insert(
                            item.source.to_string_lossy().to_string(),
                            "simulated failure".to_string(),
                        );
                    }
                    ErrorPolicy::RetryThenSkip => {
                        if item.retry_count < self.config.max_retries {
                            item.increment_retry();
                            item.mark_failed("simulated failure after retry");
                        } else {
                            item.mark_skipped();
                            self.progress.record_skipped();
                        }
                        self.progress.record_failed();
                        errors.insert(
                            item.source.to_string_lossy().to_string(),
                            "simulated failure".to_string(),
                        );
                    }
                }
            } else {
                item.mark_completed();
                self.progress.record_completed();
            }
        }

        let mut summary = BatchSummary::new(
            self.progress.total(),
            self.progress.completed(),
            self.progress.failed(),
            self.progress.skipped(),
        );
        summary.errors = errors;
        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_policy_display() {
        assert_eq!(ErrorPolicy::StopOnFirst.to_string(), "StopOnFirst");
        assert_eq!(ErrorPolicy::SkipAndContinue.to_string(), "SkipAndContinue");
        assert_eq!(ErrorPolicy::RetryThenSkip.to_string(), "RetryThenSkip");
    }

    #[test]
    fn test_item_status_display() {
        assert_eq!(ItemStatus::Pending.to_string(), "Pending");
        assert_eq!(ItemStatus::InProgress.to_string(), "InProgress");
        assert_eq!(ItemStatus::Completed.to_string(), "Completed");
        assert_eq!(ItemStatus::Skipped.to_string(), "Skipped");
        assert!(ItemStatus::Failed("oops".into())
            .to_string()
            .contains("oops"));
    }

    #[test]
    fn test_batch_item_lifecycle() {
        let mut item = BatchItem::new(
            PathBuf::from("/src/a.png"),
            PathBuf::from("/dst/a.png"),
            1920,
            1080,
        );
        assert!(item.is_pending());
        item.mark_in_progress();
        assert!(!item.is_pending());
        item.mark_completed();
        assert!(item.is_completed());
    }

    #[test]
    fn test_batch_item_failure() {
        let mut item = BatchItem::new(PathBuf::from("/a.png"), PathBuf::from("/b.png"), 100, 100);
        item.mark_failed("disk full");
        assert!(item.is_failed());
        assert!(!item.is_completed());
    }

    #[test]
    fn test_batch_config_builder() {
        let config = BatchConfig::new()
            .with_max_parallel(8)
            .with_error_policy(ErrorPolicy::StopOnFirst)
            .with_max_retries(5)
            .with_overwrite(true)
            .with_output_format("png");
        assert_eq!(config.max_parallel, 8);
        assert_eq!(config.error_policy, ErrorPolicy::StopOnFirst);
        assert_eq!(config.max_retries, 5);
        assert!(config.overwrite_existing);
        assert_eq!(config.output_format.as_deref(), Some("png"));
    }

    #[test]
    fn test_batch_config_min_parallel() {
        let config = BatchConfig::new().with_max_parallel(0);
        assert_eq!(config.max_parallel, 1);
    }

    #[test]
    fn test_progress_tracker() {
        let tracker = ProgressTracker::new(10);
        assert_eq!(tracker.total(), 10);
        assert_eq!(tracker.remaining(), 10);
        assert!(!tracker.is_finished());

        tracker.record_completed();
        tracker.record_completed();
        tracker.record_failed();
        assert_eq!(tracker.completed(), 2);
        assert_eq!(tracker.failed(), 1);
        assert_eq!(tracker.remaining(), 7);
    }

    #[test]
    fn test_progress_percent() {
        let tracker = ProgressTracker::new(4);
        tracker.record_completed();
        tracker.record_completed();
        assert!((tracker.progress_percent() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_progress_empty() {
        let tracker = ProgressTracker::new(0);
        assert!((tracker.progress_percent() - 100.0).abs() < f64::EPSILON);
        assert!(tracker.is_finished());
    }

    #[test]
    fn test_batch_summary() {
        let summary = BatchSummary::new(10, 8, 1, 1);
        assert!((summary.success_rate() - 80.0).abs() < f64::EPSILON);
        assert!(!summary.all_succeeded());
    }

    #[test]
    fn test_batch_summary_all_success() {
        let summary = BatchSummary::new(5, 5, 0, 0);
        assert!(summary.all_succeeded());
        assert!((summary.success_rate() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_batch_summary_display() {
        let summary = BatchSummary::new(10, 7, 2, 1);
        let display = format!("{}", summary);
        assert!(display.contains("7/10"));
        assert!(display.contains("2 failed"));
    }

    #[test]
    fn test_batch_job_process_all_success() {
        let items = vec![
            BatchItem::new(
                PathBuf::from("/a.png"),
                PathBuf::from("/out/a.png"),
                100,
                100,
            ),
            BatchItem::new(
                PathBuf::from("/b.png"),
                PathBuf::from("/out/b.png"),
                200,
                200,
            ),
        ];
        let config = BatchConfig::new();
        let mut job = BatchScaleJob::new(items, config);
        let summary = job.process_all();
        assert_eq!(summary.completed, 2);
        assert_eq!(summary.failed, 0);
    }

    #[test]
    fn test_batch_job_process_with_failure() {
        let items = vec![
            BatchItem::new(
                PathBuf::from("/ok.png"),
                PathBuf::from("/out/ok.png"),
                100,
                100,
            ),
            BatchItem::new(
                PathBuf::from("/fail_this.png"),
                PathBuf::from("/out/fail.png"),
                100,
                100,
            ),
            BatchItem::new(
                PathBuf::from("/ok2.png"),
                PathBuf::from("/out/ok2.png"),
                100,
                100,
            ),
        ];
        let config = BatchConfig::new().with_error_policy(ErrorPolicy::SkipAndContinue);
        let mut job = BatchScaleJob::new(items, config);
        let summary = job.process_all();
        assert_eq!(summary.completed, 2);
        assert_eq!(summary.failed, 1);
        assert!(!summary.errors.is_empty());
    }
}
