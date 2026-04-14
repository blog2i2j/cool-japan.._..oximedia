// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Batch conversion progress tracking with ETA calculation.
//!
//! `BatchProgress` tracks how many items in a batch have been processed and
//! computes an estimated time to completion based on elapsed wall-clock time.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Atomic progress counter for batch operations.
///
/// Designed to be cheaply shared across threads via `Arc`.
///
/// # Example
///
/// ```
/// use oximedia_convert::batch_progress::BatchProgress;
///
/// let bp = BatchProgress::new(100);
/// bp.advance(10);
/// let eta = bp.eta_seconds(5.0); // 5 seconds elapsed
/// assert!(eta.is_some());
/// ```
#[derive(Debug)]
pub struct BatchProgress {
    total: u64,
    completed: Arc<AtomicU64>,
}

impl BatchProgress {
    /// Create a new `BatchProgress` for a batch of `total` items.
    ///
    /// `total` must be > 0; if 0 is given the value is clamped to 1 to avoid
    /// division by zero in ETA calculation.
    pub fn new(total: u64) -> Self {
        BatchProgress {
            total: total.max(1),
            completed: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Advance the completed item count by `n`.
    ///
    /// The internal counter is saturated at `total` and never wraps.
    pub fn advance(&self, n: u64) {
        let prev = self.completed.fetch_add(n, Ordering::Relaxed);
        // Saturate at total to avoid phantom ETA glitches
        let new_val = prev.saturating_add(n);
        if new_val > self.total {
            self.completed.store(self.total, Ordering::Relaxed);
        }
    }

    /// Return the number of completed items.
    pub fn completed(&self) -> u64 {
        self.completed.load(Ordering::Relaxed).min(self.total)
    }

    /// Return the total item count.
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Return the number of remaining items.
    pub fn remaining(&self) -> u64 {
        self.total.saturating_sub(self.completed())
    }

    /// Return a progress fraction in `[0.0, 1.0]`.
    pub fn fraction(&self) -> f64 {
        self.completed() as f64 / self.total as f64
    }

    /// Return `true` when all items have been processed.
    pub fn is_complete(&self) -> bool {
        self.completed() >= self.total
    }

    /// Estimate remaining seconds based on `elapsed_s` seconds elapsed so far.
    ///
    /// Returns `None` if no items have been completed yet (no throughput data),
    /// or if the batch is already complete.
    ///
    /// The estimate uses simple linear extrapolation:
    /// ```text
    /// rate = completed / elapsed_s
    /// eta  = remaining / rate
    ///      = remaining * elapsed_s / completed
    /// ```
    pub fn eta_seconds(&self, elapsed_s: f64) -> Option<f64> {
        let done = self.completed();
        let remaining = self.remaining();

        if done == 0 || elapsed_s <= 0.0 || remaining == 0 {
            return None;
        }

        let rate = done as f64 / elapsed_s;
        Some(remaining as f64 / rate)
    }

    /// Return a human-readable percentage string, e.g. `"42.5%"`.
    pub fn percentage_string(&self) -> String {
        format!("{:.1}%", self.fraction() * 100.0)
    }

    /// Create a clone of the internal `Arc` counter so that multiple threads
    /// can call [`advance`][Self::advance] concurrently on the same progress.
    pub fn shared_handle(&self) -> SharedBatchProgress {
        SharedBatchProgress {
            total: self.total,
            completed: Arc::clone(&self.completed),
        }
    }
}

/// A lightweight handle for advancing a [`BatchProgress`] from another thread.
///
/// Multiple `SharedBatchProgress` handles can be created from a single
/// `BatchProgress`; they all update the same atomic counter.
#[derive(Clone, Debug)]
pub struct SharedBatchProgress {
    total: u64,
    completed: Arc<AtomicU64>,
}

impl SharedBatchProgress {
    /// Advance the shared counter by `n`.
    pub fn advance(&self, n: u64) {
        let prev = self.completed.fetch_add(n, Ordering::Relaxed);
        let new_val = prev.saturating_add(n);
        if new_val > self.total {
            self.completed.store(self.total, Ordering::Relaxed);
        }
    }

    /// Return the current completed count.
    pub fn completed(&self) -> u64 {
        self.completed.load(Ordering::Relaxed).min(self.total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_starts_at_zero() {
        let bp = BatchProgress::new(100);
        assert_eq!(bp.completed(), 0);
        assert_eq!(bp.total(), 100);
        assert_eq!(bp.remaining(), 100);
    }

    #[test]
    fn advance_increments_counter() {
        let bp = BatchProgress::new(100);
        bp.advance(10);
        assert_eq!(bp.completed(), 10);
        assert_eq!(bp.remaining(), 90);
    }

    #[test]
    fn advance_saturates_at_total() {
        let bp = BatchProgress::new(10);
        bp.advance(100);
        assert_eq!(bp.completed(), 10);
        assert_eq!(bp.remaining(), 0);
        assert!(bp.is_complete());
    }

    #[test]
    fn fraction_is_correct() {
        let bp = BatchProgress::new(4);
        bp.advance(1);
        assert!((bp.fraction() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn eta_returns_none_before_any_progress() {
        let bp = BatchProgress::new(100);
        assert!(bp.eta_seconds(10.0).is_none());
    }

    #[test]
    fn eta_returns_none_when_complete() {
        let bp = BatchProgress::new(10);
        bp.advance(10);
        assert!(bp.eta_seconds(5.0).is_none());
    }

    #[test]
    fn eta_linear_extrapolation() {
        let bp = BatchProgress::new(100);
        bp.advance(50); // 50% done in 10 s → ETA ≈ 10 s
        let eta = bp.eta_seconds(10.0).expect("should have eta");
        assert!((eta - 10.0).abs() < 1e-6, "eta={eta}");
    }

    #[test]
    fn percentage_string_format() {
        let bp = BatchProgress::new(200);
        bp.advance(100);
        assert_eq!(bp.percentage_string(), "50.0%");
    }

    #[test]
    fn shared_handle_advances_same_counter() {
        let bp = BatchProgress::new(100);
        let handle = bp.shared_handle();
        handle.advance(30);
        assert_eq!(bp.completed(), 30);
    }

    #[test]
    fn total_zero_clamped_to_one() {
        let bp = BatchProgress::new(0);
        assert_eq!(bp.total(), 1);
    }
}
