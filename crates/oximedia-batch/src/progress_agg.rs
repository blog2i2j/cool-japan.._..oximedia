//! Batch-level progress aggregation.
//!
//! [`BatchProgressAggregator`] tracks completion across a set of N total jobs
//! and exposes a percentage-complete readout and a done flag.
//!
//! # Example
//!
//! ```
//! use oximedia_batch::progress_agg::BatchProgressAggregator;
//!
//! let mut agg = BatchProgressAggregator::new(10);
//! agg.complete(3);
//! assert!((agg.percent() - 30.0).abs() < 1e-4);
//! assert!(!agg.is_done());
//!
//! agg.complete(7);
//! assert!(agg.is_done());
//! ```

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// BatchProgressAggregator
// ---------------------------------------------------------------------------

/// Aggregates completion counts across a fixed-size batch of jobs.
#[derive(Debug, Clone)]
pub struct BatchProgressAggregator {
    /// Total number of jobs in the batch.
    total: usize,
    /// Number of completed jobs (clamped to `total`).
    completed: usize,
    /// Number of failed jobs (informational only; counted separately).
    failed: usize,
}

impl BatchProgressAggregator {
    /// Create a new aggregator for a batch of `total` jobs.
    ///
    /// # Panics
    ///
    /// (None — zero `total` is allowed; `percent()` will return `0.0` and
    /// `is_done()` will return `true`.)
    #[must_use]
    pub fn new(total: usize) -> Self {
        Self {
            total,
            completed: 0,
            failed: 0,
        }
    }

    /// Mark `n` additional jobs as successfully completed.
    ///
    /// The running total is clamped to `total` so over-reporting never causes
    /// a percentage above 100 %.
    pub fn complete(&mut self, n: usize) {
        self.completed = (self.completed.saturating_add(n)).min(self.total);
    }

    /// Mark `n` additional jobs as failed.
    ///
    /// Failed jobs are counted separately; they do **not** automatically
    /// count as completed.  Call `complete(n)` as well if you want to
    /// advance the progress.
    pub fn record_failure(&mut self, n: usize) {
        self.failed = self.failed.saturating_add(n);
    }

    /// Completion percentage in `[0.0, 100.0]`.
    ///
    /// Returns `100.0` when `total == 0` (vacuously done).
    #[must_use]
    pub fn percent(&self) -> f32 {
        if self.total == 0 {
            return 100.0;
        }
        (self.completed as f32 / self.total as f32) * 100.0
    }

    /// Returns `true` when all jobs have been completed (`completed >= total`).
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.completed >= self.total
    }

    /// Number of successfully completed jobs.
    #[must_use]
    pub fn completed(&self) -> usize {
        self.completed
    }

    /// Number of failed jobs.
    #[must_use]
    pub fn failed(&self) -> usize {
        self.failed
    }

    /// Total number of jobs in this batch.
    #[must_use]
    pub fn total(&self) -> usize {
        self.total
    }

    /// Number of jobs that have neither completed nor failed.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.total.saturating_sub(self.completed)
    }

    /// Reset counters (useful when reusing the aggregator for a new batch).
    pub fn reset(&mut self) {
        self.completed = 0;
        self.failed = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── new ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_new_starts_at_zero() {
        let agg = BatchProgressAggregator::new(10);
        assert_eq!(agg.completed(), 0);
        assert_eq!(agg.total(), 10);
        assert!(!agg.is_done());
    }

    #[test]
    fn test_new_zero_total_is_done() {
        let agg = BatchProgressAggregator::new(0);
        assert!(agg.is_done());
        assert!((agg.percent() - 100.0).abs() < 1e-4);
    }

    // ── complete ──────────────────────────────────────────────────────────────

    #[test]
    fn test_complete_increments_counter() {
        let mut agg = BatchProgressAggregator::new(10);
        agg.complete(3);
        assert_eq!(agg.completed(), 3);
    }

    #[test]
    fn test_complete_clamped_to_total() {
        let mut agg = BatchProgressAggregator::new(5);
        agg.complete(10); // more than total
        assert_eq!(agg.completed(), 5);
        assert!(agg.is_done());
    }

    #[test]
    fn test_complete_accumulates() {
        let mut agg = BatchProgressAggregator::new(10);
        agg.complete(3);
        agg.complete(4);
        assert_eq!(agg.completed(), 7);
    }

    // ── percent ───────────────────────────────────────────────────────────────

    #[test]
    fn test_percent_zero_completed() {
        let agg = BatchProgressAggregator::new(10);
        assert!((agg.percent() - 0.0).abs() < 1e-4);
    }

    #[test]
    fn test_percent_partial() {
        let mut agg = BatchProgressAggregator::new(10);
        agg.complete(3);
        assert!((agg.percent() - 30.0).abs() < 1e-3);
    }

    #[test]
    fn test_percent_full() {
        let mut agg = BatchProgressAggregator::new(4);
        agg.complete(4);
        assert!((agg.percent() - 100.0).abs() < 1e-4);
    }

    // ── is_done ───────────────────────────────────────────────────────────────

    #[test]
    fn test_is_done_false_initially() {
        let agg = BatchProgressAggregator::new(5);
        assert!(!agg.is_done());
    }

    #[test]
    fn test_is_done_true_when_all_complete() {
        let mut agg = BatchProgressAggregator::new(3);
        agg.complete(3);
        assert!(agg.is_done());
    }

    // ── record_failure ────────────────────────────────────────────────────────

    #[test]
    fn test_record_failure_increments_failed() {
        let mut agg = BatchProgressAggregator::new(10);
        agg.record_failure(2);
        assert_eq!(agg.failed(), 2);
    }

    #[test]
    fn test_record_failure_does_not_advance_completed() {
        let mut agg = BatchProgressAggregator::new(10);
        agg.record_failure(3);
        assert_eq!(agg.completed(), 0);
        assert!(!agg.is_done());
    }

    // ── remaining ────────────────────────────────────────────────────────────

    #[test]
    fn test_remaining_decreases_with_completion() {
        let mut agg = BatchProgressAggregator::new(10);
        agg.complete(4);
        assert_eq!(agg.remaining(), 6);
    }

    #[test]
    fn test_remaining_zero_when_done() {
        let mut agg = BatchProgressAggregator::new(3);
        agg.complete(3);
        assert_eq!(agg.remaining(), 0);
    }

    // ── reset ────────────────────────────────────────────────────────────────

    #[test]
    fn test_reset_clears_counters() {
        let mut agg = BatchProgressAggregator::new(10);
        agg.complete(5);
        agg.record_failure(2);
        agg.reset();
        assert_eq!(agg.completed(), 0);
        assert_eq!(agg.failed(), 0);
    }
}
