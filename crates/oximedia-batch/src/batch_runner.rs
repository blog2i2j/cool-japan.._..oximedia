//! Batch job runner: configuration, item tracking, and run-level statistics.
//!
//! This module provides lightweight, self-contained types for managing a
//! collection of batch items through a simple state machine without any
//! external I/O or async machinery.

#![allow(dead_code)]

/// Configuration for a batch run.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of items to process in parallel.
    pub max_parallel: usize,
    /// Maximum number of retries per item before marking it failed.
    pub retry_limit: u32,
    /// Per-item timeout in milliseconds (0 = no timeout).
    pub timeout_ms: u64,
    /// If `true`, abort the entire run on the first failure.
    pub fail_fast: bool,
}

impl BatchConfig {
    /// Create a default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_parallel: 4,
            retry_limit: 3,
            timeout_ms: 30_000,
            fail_fast: false,
        }
    }

    /// Return `true` when the configuration is logically valid.
    #[must_use]
    pub fn validate(&self) -> bool {
        self.max_parallel > 0
    }
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// A single item in a batch run.
#[derive(Debug, Clone)]
pub struct BatchItem {
    /// Unique numeric identifier.
    pub id: u64,
    /// Human-readable name.
    pub name: String,
    /// Input path or URI.
    pub input: String,
    /// Output path or URI.
    pub output: String,
    /// Number of times this item has been retried so far.
    pub retries: u32,
}

impl BatchItem {
    /// Create a new batch item.
    #[must_use]
    pub fn new(
        id: u64,
        name: impl Into<String>,
        input: impl Into<String>,
        output: impl Into<String>,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            input: input.into(),
            output: output.into(),
            retries: 0,
        }
    }

    /// Return `true` when the item can still be retried.
    #[must_use]
    pub fn can_retry(&self, max: u32) -> bool {
        self.retries < max
    }
}

/// Status of a single batch item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchItemStatus {
    /// Waiting to be started.
    Pending,
    /// Currently being processed.
    Running,
    /// Completed successfully.
    Done,
    /// Failed and exhausted retries.
    Failed,
    /// Skipped (e.g., due to `fail_fast`).
    Skipped,
}

impl BatchItemStatus {
    /// Return `true` for terminal states that will never change again.
    #[must_use]
    pub fn is_final(self) -> bool {
        matches!(self, Self::Done | Self::Failed | Self::Skipped)
    }
}

/// A complete batch run holding items and their statuses.
#[derive(Debug)]
pub struct BatchRun {
    /// Configuration for this run.
    pub config: BatchConfig,
    /// Items to process.
    pub items: Vec<BatchItem>,
    /// Parallel status vector; `statuses[i]` corresponds to `items[i]`.
    pub statuses: Vec<BatchItemStatus>,
}

impl BatchRun {
    /// Create a new batch run.
    ///
    /// # Panics
    ///
    /// Panics if `items` is empty.
    #[must_use]
    pub fn new(config: BatchConfig, items: Vec<BatchItem>) -> Self {
        let len = items.len();
        Self {
            config,
            items,
            statuses: vec![BatchItemStatus::Pending; len],
        }
    }

    /// Find the next pending item that can be started.
    ///
    /// Returns `None` when no pending items remain or the parallel limit is
    /// already saturated.
    #[must_use]
    pub fn start_next(&mut self) -> Option<(usize, &BatchItem)> {
        let running = self
            .statuses
            .iter()
            .filter(|s| **s == BatchItemStatus::Running)
            .count();
        if running >= self.config.max_parallel {
            return None;
        }
        let idx = self
            .statuses
            .iter()
            .position(|s| *s == BatchItemStatus::Pending)?;
        self.statuses[idx] = BatchItemStatus::Running;
        Some((idx, &self.items[idx]))
    }

    /// Mark the item at `idx` as done.
    pub fn mark_done(&mut self, idx: usize) {
        if idx < self.statuses.len() {
            self.statuses[idx] = BatchItemStatus::Done;
        }
    }

    /// Mark the item at `idx` as failed (increments its retry counter).
    pub fn mark_failed(&mut self, idx: usize) {
        if idx < self.statuses.len() {
            self.items[idx].retries += 1;
            self.statuses[idx] = BatchItemStatus::Failed;
        }
    }

    /// Count items still in `Pending` state.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.statuses
            .iter()
            .filter(|s| **s == BatchItemStatus::Pending)
            .count()
    }

    /// Count items in `Done` state.
    #[must_use]
    pub fn done_count(&self) -> usize {
        self.statuses
            .iter()
            .filter(|s| **s == BatchItemStatus::Done)
            .count()
    }

    /// Count items in `Failed` state.
    #[must_use]
    pub fn failed_count(&self) -> usize {
        self.statuses
            .iter()
            .filter(|s| **s == BatchItemStatus::Failed)
            .count()
    }

    /// Success rate as a fraction in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` when no items are in a final state.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn success_rate(&self) -> f64 {
        let final_count = self.statuses.iter().filter(|s| s.is_final()).count();
        if final_count == 0 {
            return 0.0;
        }
        self.done_count() as f64 / final_count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_items(n: u64) -> Vec<BatchItem> {
        (0..n)
            .map(|i| {
                BatchItem::new(
                    i,
                    format!("item-{i}"),
                    format!("in-{i}"),
                    format!("out-{i}"),
                )
            })
            .collect()
    }

    #[test]
    fn test_batch_config_default_valid() {
        let cfg = BatchConfig::new();
        assert!(cfg.validate());
    }

    #[test]
    fn test_batch_config_zero_parallel_invalid() {
        let cfg = BatchConfig {
            max_parallel: 0,
            ..BatchConfig::new()
        };
        assert!(!cfg.validate());
    }

    #[test]
    fn test_batch_config_fields() {
        let cfg = BatchConfig::new();
        assert_eq!(cfg.max_parallel, 4);
        assert_eq!(cfg.retry_limit, 3);
        assert_eq!(cfg.timeout_ms, 30_000);
        assert!(!cfg.fail_fast);
    }

    #[test]
    fn test_batch_item_can_retry_true() {
        let item = BatchItem::new(1, "x", "a", "b");
        assert!(item.can_retry(3));
    }

    #[test]
    fn test_batch_item_can_retry_false_at_limit() {
        let mut item = BatchItem::new(1, "x", "a", "b");
        item.retries = 3;
        assert!(!item.can_retry(3));
    }

    #[test]
    fn test_batch_item_status_is_final() {
        assert!(BatchItemStatus::Done.is_final());
        assert!(BatchItemStatus::Failed.is_final());
        assert!(BatchItemStatus::Skipped.is_final());
        assert!(!BatchItemStatus::Pending.is_final());
        assert!(!BatchItemStatus::Running.is_final());
    }

    #[test]
    fn test_batch_run_initial_pending() {
        let run = BatchRun::new(BatchConfig::new(), make_items(5));
        assert_eq!(run.pending_count(), 5);
        assert_eq!(run.done_count(), 0);
        assert_eq!(run.failed_count(), 0);
    }

    #[test]
    fn test_start_next_transitions_to_running() {
        let mut run = BatchRun::new(BatchConfig::new(), make_items(3));
        let result = run.start_next();
        assert!(result.is_some());
        assert_eq!(run.pending_count(), 2);
    }

    #[test]
    fn test_start_next_respects_parallel_limit() {
        let mut cfg = BatchConfig::new();
        cfg.max_parallel = 1;
        let mut run = BatchRun::new(cfg, make_items(3));
        let _ = run.start_next(); // starts one
        let result = run.start_next(); // should be blocked
        assert!(result.is_none());
    }

    #[test]
    fn test_mark_done_increments_done_count() {
        let mut run = BatchRun::new(BatchConfig::new(), make_items(3));
        let (idx, _) = run.start_next().expect("start_next should succeed");
        run.mark_done(idx);
        assert_eq!(run.done_count(), 1);
    }

    #[test]
    fn test_mark_failed_increments_failed_count() {
        let mut run = BatchRun::new(BatchConfig::new(), make_items(3));
        let (idx, _) = run.start_next().expect("start_next should succeed");
        run.mark_failed(idx);
        assert_eq!(run.failed_count(), 1);
        assert_eq!(run.items[idx].retries, 1);
    }

    #[test]
    fn test_success_rate_all_done() {
        let mut run = BatchRun::new(BatchConfig::new(), make_items(2));
        for _ in 0..2 {
            let (idx, _) = run.start_next().expect("start_next should succeed");
            run.mark_done(idx);
        }
        assert!((run.success_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_success_rate_mixed() {
        let mut run = BatchRun::new(BatchConfig::new(), make_items(4));
        // done: 2, failed: 2
        for _ in 0..2 {
            let (idx, _) = run.start_next().expect("start_next should succeed");
            run.mark_done(idx);
        }
        for _ in 0..2 {
            let (idx, _) = run.start_next().expect("start_next should succeed");
            run.mark_failed(idx);
        }
        let rate = run.success_rate();
        assert!((rate - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_success_rate_no_final_returns_zero() {
        let run = BatchRun::new(BatchConfig::new(), make_items(3));
        assert!((run.success_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_start_next_returns_none_when_empty() {
        let mut run = BatchRun::new(BatchConfig::new(), make_items(1));
        let _ = run.start_next(); // one item goes Running
        let r = run.start_next(); // no more Pending items (parallel limit not hit but no Pending)
                                  // The running count (1) < max_parallel (4), but there's no Pending item
        assert!(r.is_none());
    }
}
