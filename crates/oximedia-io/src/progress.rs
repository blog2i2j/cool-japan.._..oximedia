//! File transfer progress tracking with ETA estimation.
//!
//! [`FileProgress`] tracks how many bytes of a file transfer have been
//! processed and computes percentage completion and estimated time remaining.
//!
//! # Design
//!
//! - Byte counts accumulate monotonically via [`advance`](FileProgress::advance).
//! - [`percent`](FileProgress::percent) clamps to [0.0, 100.0].
//! - [`eta_secs`](FileProgress::eta_secs) requires a positive `elapsed` value
//!   and at least one byte processed; returns `None` when the estimate is not
//!   meaningful (division by zero, complete, or no data yet).
//!
//! # Example
//!
//! ```
//! use oximedia_io::progress::FileProgress;
//!
//! let mut p = FileProgress::new(1_000_000);
//! p.advance(250_000);
//! assert!((p.percent() - 25.0).abs() < 0.01);
//!
//! // After 1 second of elapsed time, ETA ≈ 3 seconds
//! let eta = p.eta_secs(1.0);
//! assert!(eta.is_some());
//! ```

#![allow(dead_code)]

/// Tracks byte-level progress for file I/O operations.
///
/// All values are stored as `u64` bytes. The percent and ETA methods use
/// `f64` arithmetic internally for precision.
#[derive(Debug, Clone)]
pub struct FileProgress {
    /// Total expected bytes.
    total_bytes: u64,
    /// Bytes processed so far.
    processed_bytes: u64,
}

impl FileProgress {
    /// Create a new progress tracker for a transfer of `total_bytes` bytes.
    ///
    /// `total_bytes` of `0` is allowed; [`percent`](Self::percent) will return
    /// `100.0` immediately and [`eta_secs`](Self::eta_secs) will return `None`.
    #[must_use]
    pub fn new(total_bytes: u64) -> Self {
        Self {
            total_bytes,
            processed_bytes: 0,
        }
    }

    /// Advance the progress counter by `bytes` bytes.
    ///
    /// The counter is clamped to `total_bytes` to prevent overshooting.
    pub fn advance(&mut self, bytes: u64) {
        self.processed_bytes = self
            .processed_bytes
            .saturating_add(bytes)
            .min(self.total_bytes);
    }

    /// Return the completion percentage in [0.0, 100.0].
    ///
    /// Returns `100.0` when `total_bytes` is `0` (nothing to transfer).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn percent(&self) -> f32 {
        if self.total_bytes == 0 {
            return 100.0;
        }
        let pct = (self.processed_bytes as f64 / self.total_bytes as f64) * 100.0;
        pct.clamp(0.0, 100.0) as f32
    }

    /// Estimate seconds remaining given the elapsed wall-clock time in seconds.
    ///
    /// Returns `None` when:
    /// - `elapsed` is ≤ 0 (invalid),
    /// - no bytes have been processed yet (rate = 0, no estimate possible),
    /// - the transfer is already complete (`processed_bytes == total_bytes`).
    ///
    /// Uses a linear rate: `eta = remaining_bytes / (processed_bytes / elapsed)`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn eta_secs(&self, elapsed: f64) -> Option<f64> {
        if elapsed <= 0.0 || self.processed_bytes == 0 {
            return None;
        }
        if self.processed_bytes >= self.total_bytes {
            return None; // already done
        }
        let remaining = (self.total_bytes - self.processed_bytes) as f64;
        let rate = self.processed_bytes as f64 / elapsed; // bytes per second
        if rate <= 0.0 {
            return None;
        }
        Some(remaining / rate)
    }

    /// Returns the number of bytes processed so far.
    #[must_use]
    pub fn processed_bytes(&self) -> u64 {
        self.processed_bytes
    }

    /// Returns the total expected bytes.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Returns the number of remaining bytes.
    #[must_use]
    pub fn remaining_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.processed_bytes)
    }

    /// Returns `true` if the transfer is complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.processed_bytes >= self.total_bytes
    }

    /// Reset the progress counter to zero without changing `total_bytes`.
    pub fn reset(&mut self) {
        self.processed_bytes = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_zero_total() {
        let p = FileProgress::new(0);
        assert_eq!(p.percent(), 100.0);
        assert!(p.is_complete());
    }

    #[test]
    fn test_percent_zero_progress() {
        let p = FileProgress::new(1000);
        assert_eq!(p.percent(), 0.0);
    }

    #[test]
    fn test_percent_half() {
        let mut p = FileProgress::new(1000);
        p.advance(500);
        assert!((p.percent() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_percent_complete() {
        let mut p = FileProgress::new(1000);
        p.advance(1000);
        assert_eq!(p.percent(), 100.0);
        assert!(p.is_complete());
    }

    #[test]
    fn test_advance_clamps_at_total() {
        let mut p = FileProgress::new(500);
        p.advance(600);
        assert_eq!(p.processed_bytes(), 500);
        assert_eq!(p.percent(), 100.0);
    }

    #[test]
    fn test_advance_accumulates() {
        let mut p = FileProgress::new(1000);
        p.advance(300);
        p.advance(300);
        assert_eq!(p.processed_bytes(), 600);
        assert!((p.percent() - 60.0).abs() < 0.01);
    }

    #[test]
    fn test_eta_no_progress() {
        let p = FileProgress::new(1000);
        assert!(p.eta_secs(1.0).is_none());
    }

    #[test]
    fn test_eta_zero_elapsed() {
        let mut p = FileProgress::new(1000);
        p.advance(100);
        assert!(p.eta_secs(0.0).is_none());
        assert!(p.eta_secs(-1.0).is_none());
    }

    #[test]
    fn test_eta_complete() {
        let mut p = FileProgress::new(1000);
        p.advance(1000);
        assert!(p.eta_secs(5.0).is_none());
    }

    #[test]
    fn test_eta_reasonable_value() {
        // 25% done in 1 second → ETA ≈ 3 seconds
        let mut p = FileProgress::new(1_000_000);
        p.advance(250_000);
        let eta = p.eta_secs(1.0).expect("should have ETA");
        assert!(
            (eta - 3.0).abs() < 0.01,
            "ETA should be ~3s, got {eta:.4}"
        );
    }

    #[test]
    fn test_remaining_bytes() {
        let mut p = FileProgress::new(1000);
        p.advance(400);
        assert_eq!(p.remaining_bytes(), 600);
    }

    #[test]
    fn test_reset() {
        let mut p = FileProgress::new(1000);
        p.advance(500);
        assert_eq!(p.processed_bytes(), 500);
        p.reset();
        assert_eq!(p.processed_bytes(), 0);
        assert_eq!(p.percent(), 0.0);
    }

    #[test]
    fn test_total_bytes_accessor() {
        let p = FileProgress::new(8192);
        assert_eq!(p.total_bytes(), 8192);
    }

    #[test]
    fn test_percent_quarter() {
        let mut p = FileProgress::new(400);
        p.advance(100);
        assert!((p.percent() - 25.0).abs() < 0.01);
    }
}
