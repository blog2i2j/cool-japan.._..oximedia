//! Batch error recovery policy.
//!
//! [`BatchErrorRecovery`] records per-job failures and decides whether
//! the batch should continue processing remaining jobs or abort.
//!
//! Two modes are supported:
//! - **skip on error** (`skip_on_error = true`) — all failures are recorded
//!   and the batch continues; `should_continue()` always returns `true`.
//! - **abort on error** (`skip_on_error = false`) — the first failure causes
//!   `should_continue()` to return `false`, halting the batch.
//!
//! # Example
//!
//! ```
//! use oximedia_batch::error_recovery::BatchErrorRecovery;
//!
//! // Skip-on-error mode: continue despite failures
//! let mut rec = BatchErrorRecovery::new(true);
//! rec.record_failure(1, "file not found");
//! assert!(rec.should_continue());
//!
//! // Abort-on-error mode: stop after first failure
//! let mut rec = BatchErrorRecovery::new(false);
//! rec.record_failure(2, "codec error");
//! assert!(!rec.should_continue());
//! ```

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// FailureRecord
// ---------------------------------------------------------------------------

/// A single job failure event.
#[derive(Debug, Clone)]
pub struct FailureRecord {
    /// ID of the job that failed.
    pub job_id: u64,
    /// Human-readable error description.
    pub error: String,
}

// ---------------------------------------------------------------------------
// BatchErrorRecovery
// ---------------------------------------------------------------------------

/// Records batch job failures and enforces a continue/abort policy.
#[derive(Debug, Clone)]
pub struct BatchErrorRecovery {
    /// When `true`, failures are skipped and the batch continues.
    /// When `false`, the first failure causes `should_continue()` to return
    /// `false`.
    skip_on_error: bool,
    /// Ordered log of all recorded failures.
    failures: Vec<FailureRecord>,
}

impl BatchErrorRecovery {
    /// Create a new `BatchErrorRecovery` with the given policy.
    #[must_use]
    pub fn new(skip_on_error: bool) -> Self {
        Self {
            skip_on_error,
            failures: Vec::new(),
        }
    }

    /// Record a job failure.
    ///
    /// Appends a [`FailureRecord`] to the internal log regardless of the
    /// current policy.
    pub fn record_failure(&mut self, job_id: u64, err: &str) {
        self.failures.push(FailureRecord {
            job_id,
            error: err.to_string(),
        });
    }

    /// Returns `true` when batch processing should continue.
    ///
    /// - In **skip** mode: always `true` (failures are tolerated).
    /// - In **abort** mode: `true` only when no failures have been recorded.
    #[must_use]
    pub fn should_continue(&self) -> bool {
        if self.skip_on_error {
            true
        } else {
            self.failures.is_empty()
        }
    }

    /// Total number of recorded failures.
    #[must_use]
    pub fn failure_count(&self) -> usize {
        self.failures.len()
    }

    /// Return a slice of all recorded failures.
    #[must_use]
    pub fn failures(&self) -> &[FailureRecord] {
        &self.failures
    }

    /// Return failures for a specific job ID.
    #[must_use]
    pub fn failures_for(&self, job_id: u64) -> Vec<&FailureRecord> {
        self.failures.iter().filter(|r| r.job_id == job_id).collect()
    }

    /// Clear all recorded failures.
    pub fn reset(&mut self) {
        self.failures.clear();
    }

    /// Whether the recovery is in skip-on-error mode.
    #[must_use]
    pub fn skip_on_error(&self) -> bool {
        self.skip_on_error
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
    fn test_new_no_failures() {
        let rec = BatchErrorRecovery::new(true);
        assert_eq!(rec.failure_count(), 0);
        assert!(rec.failures().is_empty());
    }

    // ── should_continue — skip mode ───────────────────────────────────────────

    #[test]
    fn test_skip_mode_should_continue_with_no_failures() {
        let rec = BatchErrorRecovery::new(true);
        assert!(rec.should_continue());
    }

    #[test]
    fn test_skip_mode_should_continue_with_failures() {
        let mut rec = BatchErrorRecovery::new(true);
        rec.record_failure(1, "oops");
        assert!(rec.should_continue(), "Skip mode should always continue");
    }

    #[test]
    fn test_skip_mode_multiple_failures() {
        let mut rec = BatchErrorRecovery::new(true);
        rec.record_failure(1, "err1");
        rec.record_failure(2, "err2");
        rec.record_failure(3, "err3");
        assert!(rec.should_continue());
        assert_eq!(rec.failure_count(), 3);
    }

    // ── should_continue — abort mode ──────────────────────────────────────────

    #[test]
    fn test_abort_mode_should_continue_initially() {
        let rec = BatchErrorRecovery::new(false);
        assert!(rec.should_continue());
    }

    #[test]
    fn test_abort_mode_stops_on_first_failure() {
        let mut rec = BatchErrorRecovery::new(false);
        rec.record_failure(1, "codec error");
        assert!(!rec.should_continue());
    }

    // ── record_failure ────────────────────────────────────────────────────────

    #[test]
    fn test_record_failure_increments_count() {
        let mut rec = BatchErrorRecovery::new(true);
        rec.record_failure(42, "disk full");
        assert_eq!(rec.failure_count(), 1);
    }

    #[test]
    fn test_record_failure_stores_job_id() {
        let mut rec = BatchErrorRecovery::new(true);
        rec.record_failure(99, "timeout");
        let f = &rec.failures()[0];
        assert_eq!(f.job_id, 99);
    }

    #[test]
    fn test_record_failure_stores_error_string() {
        let mut rec = BatchErrorRecovery::new(true);
        rec.record_failure(1, "my error");
        assert_eq!(rec.failures()[0].error, "my error");
    }

    // ── failures_for ──────────────────────────────────────────────────────────

    #[test]
    fn test_failures_for_filters_by_job_id() {
        let mut rec = BatchErrorRecovery::new(true);
        rec.record_failure(1, "err1");
        rec.record_failure(2, "err2");
        rec.record_failure(1, "err3");
        let for_1 = rec.failures_for(1);
        assert_eq!(for_1.len(), 2);
        let for_2 = rec.failures_for(2);
        assert_eq!(for_2.len(), 1);
    }

    #[test]
    fn test_failures_for_empty_when_no_match() {
        let rec = BatchErrorRecovery::new(true);
        assert!(rec.failures_for(42).is_empty());
    }

    // ── reset ────────────────────────────────────────────────────────────────

    #[test]
    fn test_reset_clears_failures() {
        let mut rec = BatchErrorRecovery::new(false);
        rec.record_failure(1, "err");
        rec.reset();
        assert!(rec.failures().is_empty());
        assert!(rec.should_continue(), "After reset, abort mode should continue again");
    }

    // ── skip_on_error accessor ────────────────────────────────────────────────

    #[test]
    fn test_skip_on_error_true() {
        let rec = BatchErrorRecovery::new(true);
        assert!(rec.skip_on_error());
    }

    #[test]
    fn test_skip_on_error_false() {
        let rec = BatchErrorRecovery::new(false);
        assert!(!rec.skip_on_error());
    }
}
