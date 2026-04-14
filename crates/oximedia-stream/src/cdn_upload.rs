//! Parallel multi-CDN segment upload for low-latency distribution.
//!
//! [`CdnUploadManager`] fans out a single segment payload to multiple CDN
//! origins simultaneously, collects per-provider outcomes, and exposes
//! aggregate statistics.  Parallelism is simulated in a single-threaded model
//! via a work-queue abstraction; real async I/O is left to the caller.
//!
//! # Key types
//!
//! | Type | Purpose |
//! |---|---|
//! | [`UploadTarget`] | Destination CDN endpoint with metadata |
//! | [`UploadJob`] | One pending upload (segment key + payload reference) |
//! | [`UploadOutcome`] | Result of a single provider upload attempt |
//! | [`UploadBatch`] | Fan-out result across all configured providers |
//! | [`CdnUploadManager`] | Orchestrates fan-out, retries, and statistics |
//!
//! # Example
//!
//! ```
//! use oximedia_stream::cdn_upload::{CdnUploadManager, UploadTarget, UploadStatus};
//!
//! let targets = vec![
//!     UploadTarget::new("origin-a", "https://a.cdn.example/upload"),
//!     UploadTarget::new("origin-b", "https://b.cdn.example/upload"),
//! ];
//! let mut manager = CdnUploadManager::new(targets, 2);
//!
//! // Simulate a successful upload to all providers.
//! let batch = manager.simulate_upload("seg-0001", &[0u8; 512], |_target| {
//!     Ok(150) // 150 ms round-trip
//! });
//!
//! assert_eq!(batch.success_count(), 2);
//! assert!(batch.all_succeeded());
//! ```

use std::collections::HashMap;

use crate::StreamError;

// ─── Upload target ────────────────────────────────────────────────────────────

/// A single CDN endpoint that receives segment uploads.
#[derive(Debug, Clone)]
pub struct UploadTarget {
    /// Human-readable provider name, e.g. `"cloudfront-eu"`.
    pub name: String,
    /// Base upload URL, e.g. `"https://origin.cdn.example/segments"`.
    pub upload_url: String,
    /// Maximum number of retry attempts on transient failure (0 = no retries).
    pub max_retries: u32,
    /// Whether this provider is currently enabled for uploads.
    pub enabled: bool,
}

impl UploadTarget {
    /// Create a new target with default retry settings (2 retries, enabled).
    pub fn new(name: impl Into<String>, upload_url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            upload_url: upload_url.into(),
            max_retries: 2,
            enabled: true,
        }
    }

    /// Create a target with explicit retry count and enabled state.
    pub fn with_options(
        name: impl Into<String>,
        upload_url: impl Into<String>,
        max_retries: u32,
        enabled: bool,
    ) -> Self {
        Self {
            name: name.into(),
            upload_url: upload_url.into(),
            max_retries,
            enabled,
        }
    }
}

// ─── Upload job ───────────────────────────────────────────────────────────────

/// Describes a single pending upload task.
#[derive(Debug, Clone)]
pub struct UploadJob {
    /// Segment identifier / filename.
    pub segment_key: String,
    /// Target provider name.
    pub target_name: String,
    /// Size of the payload in bytes.
    pub payload_bytes: usize,
}

// ─── Upload status / outcome ──────────────────────────────────────────────────

/// The final status of a single provider upload attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UploadStatus {
    /// Upload succeeded.
    Success,
    /// Upload failed after exhausting all retries.
    Failed(String),
    /// Provider was disabled; upload was skipped.
    Skipped,
}

/// The result of uploading one segment to one provider.
#[derive(Debug, Clone)]
pub struct UploadOutcome {
    /// Provider name.
    pub target_name: String,
    /// Final status.
    pub status: UploadStatus,
    /// Round-trip time in milliseconds (only meaningful on success).
    pub rtt_ms: u64,
    /// Number of attempts made (including the initial try).
    pub attempts: u32,
}

impl UploadOutcome {
    /// Whether this outcome represents a successful upload.
    pub fn is_success(&self) -> bool {
        self.status == UploadStatus::Success
    }
}

// ─── Upload batch ─────────────────────────────────────────────────────────────

/// Aggregated results for one segment fan-out across all providers.
#[derive(Debug, Clone)]
pub struct UploadBatch {
    /// Segment identifier that was uploaded.
    pub segment_key: String,
    /// Per-provider outcomes, in registration order.
    pub outcomes: Vec<UploadOutcome>,
}

impl UploadBatch {
    /// Number of providers that succeeded.
    pub fn success_count(&self) -> usize {
        self.outcomes.iter().filter(|o| o.is_success()).count()
    }

    /// Number of providers that failed.
    pub fn failure_count(&self) -> usize {
        self.outcomes
            .iter()
            .filter(|o| matches!(o.status, UploadStatus::Failed(_)))
            .count()
    }

    /// Number of providers that were skipped (disabled).
    pub fn skipped_count(&self) -> usize {
        self.outcomes
            .iter()
            .filter(|o| o.status == UploadStatus::Skipped)
            .count()
    }

    /// `true` when every enabled provider succeeded.
    pub fn all_succeeded(&self) -> bool {
        self.failure_count() == 0
    }

    /// Average round-trip time across successful uploads (ms), or `None`.
    pub fn avg_rtt_ms(&self) -> Option<f64> {
        let successful: Vec<u64> = self
            .outcomes
            .iter()
            .filter(|o| o.is_success())
            .map(|o| o.rtt_ms)
            .collect();
        if successful.is_empty() {
            None
        } else {
            Some(successful.iter().sum::<u64>() as f64 / successful.len() as f64)
        }
    }

    /// The slowest successful RTT, or `None` if no successes.
    pub fn max_rtt_ms(&self) -> Option<u64> {
        self.outcomes
            .iter()
            .filter(|o| o.is_success())
            .map(|o| o.rtt_ms)
            .max()
    }
}

// ─── Upload statistics ────────────────────────────────────────────────────────

/// Cumulative statistics across all batches handled by a [`CdnUploadManager`].
#[derive(Debug, Clone, Default)]
pub struct UploadStats {
    /// Total segments uploaded (all attempts).
    pub total_segments: u64,
    /// Total successful provider uploads.
    pub total_successes: u64,
    /// Total failed provider uploads.
    pub total_failures: u64,
    /// Total skipped provider uploads.
    pub total_skipped: u64,
    /// Sum of all successful RTTs (for average calculation).
    rtt_sum_ms: u64,
    /// Count of successful RTT samples.
    rtt_sample_count: u64,
}

impl UploadStats {
    /// Average RTT across all successful uploads (ms), or `0.0` if none.
    pub fn avg_rtt_ms(&self) -> f64 {
        if self.rtt_sample_count == 0 {
            0.0
        } else {
            self.rtt_sum_ms as f64 / self.rtt_sample_count as f64
        }
    }

    /// Success ratio in `[0.0, 1.0]` across all provider upload attempts.
    pub fn success_ratio(&self) -> f64 {
        let total = self.total_successes + self.total_failures + self.total_skipped;
        if total == 0 {
            0.0
        } else {
            self.total_successes as f64 / total as f64
        }
    }
}

// ─── CdnUploadManager ─────────────────────────────────────────────────────────

/// Manages parallel fan-out segment uploads to multiple CDN providers.
///
/// In a real system each provider would use async I/O.  This implementation
/// exposes a `simulate_upload` method that accepts a closure modelling the
/// per-provider transfer, making the logic fully testable without networking.
#[derive(Debug)]
pub struct CdnUploadManager {
    /// Registered upload targets in insertion order.
    targets: Vec<UploadTarget>,
    /// Maximum upload concurrency (informational; parallelism is caller-driven).
    pub max_concurrency: usize,
    /// Cumulative statistics.
    stats: UploadStats,
    /// Per-provider consecutive failure counts.
    failure_counts: HashMap<String, u32>,
}

impl CdnUploadManager {
    /// Create a manager with the given targets and concurrency limit.
    pub fn new(targets: Vec<UploadTarget>, max_concurrency: usize) -> Self {
        let failure_counts = targets.iter().map(|t| (t.name.clone(), 0u32)).collect();
        Self {
            targets,
            max_concurrency: max_concurrency.max(1),
            stats: UploadStats::default(),
            failure_counts,
        }
    }

    /// Number of registered upload targets.
    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    /// Number of currently enabled targets.
    pub fn enabled_count(&self) -> usize {
        self.targets.iter().filter(|t| t.enabled).count()
    }

    /// Enable or disable a provider by name.  Returns `true` if found.
    pub fn set_enabled(&mut self, name: &str, enabled: bool) -> bool {
        if let Some(t) = self.targets.iter_mut().find(|t| t.name == name) {
            t.enabled = enabled;
            true
        } else {
            false
        }
    }

    /// Simulate a fan-out upload using a caller-provided transfer function.
    ///
    /// The closure receives a reference to the [`UploadTarget`] and returns
    /// either `Ok(rtt_ms)` for a successful upload or `Err(msg)` for a
    /// transient failure.  The manager retries up to `target.max_retries`
    /// times on each failure before recording the outcome as `Failed`.
    ///
    /// This method drives the statistics accumulators so that `stats()` always
    /// reflects the aggregate view.
    pub fn simulate_upload<F>(
        &mut self,
        segment_key: &str,
        payload: &[u8],
        mut transfer: F,
    ) -> UploadBatch
    where
        F: FnMut(&UploadTarget) -> Result<u64, String>,
    {
        let _ = payload; // payload size is for bookkeeping; content unused here
        let mut outcomes = Vec::with_capacity(self.targets.len());

        for target in &self.targets {
            if !target.enabled {
                outcomes.push(UploadOutcome {
                    target_name: target.name.clone(),
                    status: UploadStatus::Skipped,
                    rtt_ms: 0,
                    attempts: 0,
                });
                self.stats.total_skipped += 1;
                continue;
            }

            let mut last_err = String::new();
            let mut succeeded_rtt: Option<(u64, u32)> = None;

            for attempt in 0..=target.max_retries {
                match transfer(target) {
                    Ok(rtt) => {
                        succeeded_rtt = Some((rtt, attempt + 1));
                        break;
                    }
                    Err(e) => {
                        last_err = e;
                    }
                }
            }

            if let Some((rtt_ms, attempts)) = succeeded_rtt {
                outcomes.push(UploadOutcome {
                    target_name: target.name.clone(),
                    status: UploadStatus::Success,
                    rtt_ms,
                    attempts,
                });
                self.stats.total_successes += 1;
                self.stats.rtt_sum_ms += rtt_ms;
                self.stats.rtt_sample_count += 1;
                if let Some(count) = self.failure_counts.get_mut(&target.name) {
                    *count = 0;
                }
            } else {
                outcomes.push(UploadOutcome {
                    target_name: target.name.clone(),
                    status: UploadStatus::Failed(last_err),
                    rtt_ms: 0,
                    attempts: target.max_retries + 1,
                });
                self.stats.total_failures += 1;
                if let Some(count) = self.failure_counts.get_mut(&target.name) {
                    *count += 1;
                }
            }
        }

        self.stats.total_segments += 1;

        UploadBatch {
            segment_key: segment_key.to_string(),
            outcomes,
        }
    }

    /// Return a snapshot of cumulative upload statistics.
    pub fn stats(&self) -> &UploadStats {
        &self.stats
    }

    /// Reset cumulative statistics without changing targets or configuration.
    pub fn reset_stats(&mut self) {
        self.stats = UploadStats::default();
        for count in self.failure_counts.values_mut() {
            *count = 0;
        }
    }

    /// Return per-provider consecutive failure counts.
    pub fn failure_counts(&self) -> &HashMap<String, u32> {
        &self.failure_counts
    }

    /// Build a list of [`UploadJob`]s for a given segment across all enabled
    /// targets — useful for callers that implement their own async fan-out.
    pub fn build_jobs(&self, segment_key: &str, payload_bytes: usize) -> Vec<UploadJob> {
        self.targets
            .iter()
            .filter(|t| t.enabled)
            .map(|t| UploadJob {
                segment_key: segment_key.to_string(),
                target_name: t.name.clone(),
                payload_bytes,
            })
            .collect()
    }

    /// Record the result of an externally-executed upload (e.g. from async I/O).
    ///
    /// Returns `Err` if the named target is not registered.
    pub fn record_external_result(
        &mut self,
        target_name: &str,
        outcome: &UploadOutcome,
    ) -> Result<(), StreamError> {
        if !self.targets.iter().any(|t| t.name == target_name) {
            return Err(StreamError::Generic(format!(
                "unknown target: {target_name}"
            )));
        }
        match &outcome.status {
            UploadStatus::Success => {
                self.stats.total_successes += 1;
                self.stats.rtt_sum_ms += outcome.rtt_ms;
                self.stats.rtt_sample_count += 1;
                if let Some(count) = self.failure_counts.get_mut(target_name) {
                    *count = 0;
                }
            }
            UploadStatus::Failed(_) => {
                self.stats.total_failures += 1;
                if let Some(count) = self.failure_counts.get_mut(target_name) {
                    *count += 1;
                }
            }
            UploadStatus::Skipped => {
                self.stats.total_skipped += 1;
            }
        }
        Ok(())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn two_target_manager() -> CdnUploadManager {
        CdnUploadManager::new(
            vec![
                UploadTarget::new("cdn-a", "https://a.example/upload"),
                UploadTarget::new("cdn-b", "https://b.example/upload"),
            ],
            2,
        )
    }

    #[test]
    fn test_all_success_basic() {
        let mut mgr = two_target_manager();
        let batch = mgr.simulate_upload("seg-001", &[0u8; 256], |_t| Ok(100));
        assert_eq!(batch.success_count(), 2);
        assert!(batch.all_succeeded());
        assert_eq!(batch.failure_count(), 0);
    }

    #[test]
    fn test_all_failed_batch() {
        let mut mgr = CdnUploadManager::new(
            vec![UploadTarget::with_options("cdn-x", "https://x", 0, true)],
            1,
        );
        let batch = mgr.simulate_upload("seg-002", &[], |_| Err("timeout".to_string()));
        assert_eq!(batch.failure_count(), 1);
        assert!(!batch.all_succeeded());
    }

    #[test]
    fn test_skipped_when_disabled() {
        let targets = vec![
            UploadTarget::with_options("cdn-a", "https://a", 0, false),
            UploadTarget::with_options("cdn-b", "https://b", 0, true),
        ];
        let mut mgr = CdnUploadManager::new(targets, 2);
        let batch = mgr.simulate_upload("seg-003", &[], |_| Ok(50));
        assert_eq!(batch.skipped_count(), 1);
        assert_eq!(batch.success_count(), 1);
    }

    #[test]
    fn test_retry_succeeds_on_second_attempt() {
        let targets = vec![UploadTarget::with_options("cdn-r", "https://r", 2, true)];
        let mut mgr = CdnUploadManager::new(targets, 1);
        let mut attempt = 0u32;
        let batch = mgr.simulate_upload("seg-004", &[], |_| {
            attempt += 1;
            if attempt < 2 {
                Err("temporary".to_string())
            } else {
                Ok(200)
            }
        });
        assert_eq!(batch.success_count(), 1);
        let outcome = &batch.outcomes[0];
        assert!(outcome.attempts >= 2);
    }

    #[test]
    fn test_avg_rtt_calculation() {
        let mut mgr = two_target_manager();
        let batch = mgr.simulate_upload("seg-005", &[], |t| {
            if t.name == "cdn-a" {
                Ok(100)
            } else {
                Ok(200)
            }
        });
        let avg = batch.avg_rtt_ms().expect("should have avg");
        assert!((avg - 150.0).abs() < 1e-9, "expected 150ms avg, got {avg}");
    }

    #[test]
    fn test_max_rtt_returns_largest() {
        let mut mgr = two_target_manager();
        let batch = mgr.simulate_upload("seg-006", &[], |t| {
            if t.name == "cdn-a" {
                Ok(50)
            } else {
                Ok(300)
            }
        });
        assert_eq!(batch.max_rtt_ms(), Some(300));
    }

    #[test]
    fn test_cumulative_stats_accumulate() {
        let mut mgr = two_target_manager();
        mgr.simulate_upload("seg-001", &[], |_| Ok(100));
        mgr.simulate_upload("seg-002", &[], |_| Ok(200));
        let stats = mgr.stats();
        assert_eq!(stats.total_segments, 2);
        assert_eq!(stats.total_successes, 4); // 2 providers × 2 segments
    }

    #[test]
    fn test_failure_count_tracking() {
        let targets = vec![UploadTarget::with_options("cdn-f", "https://f", 0, true)];
        let mut mgr = CdnUploadManager::new(targets, 1);
        mgr.simulate_upload("seg-001", &[], |_| Err("err".to_string()));
        let counts = mgr.failure_counts();
        assert_eq!(counts.get("cdn-f").copied(), Some(1));
    }

    #[test]
    fn test_reset_stats_clears_counters() {
        let mut mgr = two_target_manager();
        mgr.simulate_upload("seg-001", &[], |_| Ok(100));
        mgr.reset_stats();
        let stats = mgr.stats();
        assert_eq!(stats.total_segments, 0);
        assert_eq!(stats.total_successes, 0);
    }

    #[test]
    fn test_build_jobs_only_enabled() {
        let targets = vec![
            UploadTarget::with_options("a", "https://a", 0, true),
            UploadTarget::with_options("b", "https://b", 0, false),
            UploadTarget::with_options("c", "https://c", 0, true),
        ];
        let mgr = CdnUploadManager::new(targets, 2);
        let jobs = mgr.build_jobs("seg-007", 512);
        assert_eq!(jobs.len(), 2);
        assert!(jobs.iter().all(|j| j.target_name != "b"));
    }

    #[test]
    fn test_set_enabled_changes_state() {
        let mut mgr = two_target_manager();
        assert_eq!(mgr.enabled_count(), 2);
        mgr.set_enabled("cdn-a", false);
        assert_eq!(mgr.enabled_count(), 1);
        mgr.set_enabled("cdn-a", true);
        assert_eq!(mgr.enabled_count(), 2);
    }

    #[test]
    fn test_record_external_result_success() {
        let mut mgr = two_target_manager();
        let outcome = UploadOutcome {
            target_name: "cdn-a".to_string(),
            status: UploadStatus::Success,
            rtt_ms: 80,
            attempts: 1,
        };
        mgr.record_external_result("cdn-a", &outcome)
            .expect("should succeed");
        assert_eq!(mgr.stats().total_successes, 1);
        assert!((mgr.stats().avg_rtt_ms() - 80.0).abs() < 1e-9);
    }

    #[test]
    fn test_record_external_result_unknown_target() {
        let mut mgr = two_target_manager();
        let outcome = UploadOutcome {
            target_name: "ghost".to_string(),
            status: UploadStatus::Success,
            rtt_ms: 50,
            attempts: 1,
        };
        let result = mgr.record_external_result("ghost", &outcome);
        assert!(result.is_err());
    }

    #[test]
    fn test_avg_rtt_zero_when_no_successes() {
        let stats = UploadStats::default();
        assert_eq!(stats.avg_rtt_ms(), 0.0);
    }

    #[test]
    fn test_success_ratio_empty() {
        let stats = UploadStats::default();
        assert_eq!(stats.success_ratio(), 0.0);
    }

    #[test]
    fn test_success_ratio_all_success() {
        let mut mgr = two_target_manager();
        mgr.simulate_upload("s", &[], |_| Ok(100));
        let ratio = mgr.stats().success_ratio();
        assert!((ratio - 1.0).abs() < 1e-9, "expected 1.0, got {ratio}");
    }
}
