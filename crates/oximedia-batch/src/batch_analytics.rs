//! Batch analytics — run-time trend analysis, failure rates, throughput
//! metrics, and SLA monitoring for batch processing workloads.
//!
//! [`BatchAnalytics`] ingests [`JobSample`] records as jobs complete and
//! maintains rolling time-series windows at configurable granularities.
//! Callers can query:
//!
//! * **Throughput** — jobs completed per minute/hour/day across a window.
//! * **Failure rate** — fraction of jobs that failed in a time window.
//! * **Latency percentiles** — P50/P95/P99 job wall-clock durations.
//! * **SLA compliance** — fraction of jobs completing within a target
//!   duration threshold.
//! * **Trend direction** — whether throughput is improving, degrading, or
//!   stable compared to the previous window of equal length.
//!
//! # Design
//!
//! All ingested samples are stored in a bounded ring-buffer ordered by
//! completion time.  Aggregations walk the buffer each time; no background
//! threads are required.  The module is `no_std`-compatible (aside from
//! `std::time`) and does not perform any I/O.

use std::collections::VecDeque;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::types::JobState;

// ---------------------------------------------------------------------------
// JobSample
// ---------------------------------------------------------------------------

/// A single completed-job record contributed to the analytics engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobSample {
    /// Human-readable job category (e.g. `"transcode:h264"`, `"ingest"`).
    pub category: String,
    /// Final state of the job.
    pub state: JobState,
    /// Wall-clock duration of the job in seconds (from queued to terminal).
    pub duration_secs: f64,
    /// Unix timestamp (seconds) when the job reached a terminal state.
    pub completed_at: u64,
    /// Optional priority level encoded as an integer (0 = low, 1 = normal,
    /// 2 = high).
    pub priority: u8,
    /// Bytes of output produced (0 if unknown / not applicable).
    pub output_bytes: u64,
}

impl JobSample {
    /// Construct a new sample with the current time as `completed_at`.
    #[must_use]
    pub fn now(category: impl Into<String>, state: JobState, duration_secs: f64) -> Self {
        let completed_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();
        Self {
            category: category.into(),
            state,
            duration_secs,
            completed_at,
            priority: 1,
            output_bytes: 0,
        }
    }

    /// Returns `true` if the job terminated successfully.
    #[must_use]
    pub fn is_success(&self) -> bool {
        matches!(self.state, JobState::Completed)
    }

    /// Returns `true` if the job failed.
    #[must_use]
    pub fn is_failure(&self) -> bool {
        matches!(self.state, JobState::Failed)
    }
}

// ---------------------------------------------------------------------------
// Time window helpers
// ---------------------------------------------------------------------------

/// A named time-aggregation window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Window {
    /// Last 60 seconds.
    LastMinute,
    /// Last 3 600 seconds.
    LastHour,
    /// Last 86 400 seconds.
    LastDay,
    /// Custom width in seconds.
    Custom(u64),
}

impl Window {
    /// Width of the window in seconds.
    #[must_use]
    pub fn secs(self) -> u64 {
        match self {
            Self::LastMinute => 60,
            Self::LastHour => 3_600,
            Self::LastDay => 86_400,
            Self::Custom(s) => s,
        }
    }
}

// ---------------------------------------------------------------------------
// Aggregated metrics
// ---------------------------------------------------------------------------

/// Aggregated metrics for a time window.
#[derive(Debug, Clone, Default)]
pub struct WindowMetrics {
    /// Number of samples in this window (all terminal states).
    pub total_jobs: usize,
    /// Number of successfully completed jobs.
    pub completed: usize,
    /// Number of failed jobs.
    pub failed: usize,
    /// Number of cancelled jobs.
    pub cancelled: usize,
    /// Mean duration in seconds (0.0 if `total_jobs` == 0).
    pub mean_duration_secs: f64,
    /// P50 duration in seconds (0.0 if insufficient data).
    pub p50_duration_secs: f64,
    /// P95 duration in seconds (0.0 if insufficient data).
    pub p95_duration_secs: f64,
    /// P99 duration in seconds (0.0 if insufficient data).
    pub p99_duration_secs: f64,
    /// Maximum duration in seconds.
    pub max_duration_secs: f64,
    /// Failure rate (0.0 ..= 1.0).
    pub failure_rate: f64,
    /// Total output bytes across all jobs in the window.
    pub total_output_bytes: u64,
    /// Effective throughput: completed jobs per second (averaged over window).
    pub throughput_per_sec: f64,
}

/// Trend direction comparing two equal-length windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Throughput is at least 5% higher in the current window.
    Improving,
    /// Throughput is within ±5% of the previous window.
    Stable,
    /// Throughput is at least 5% lower in the current window.
    Degrading,
    /// Insufficient data for comparison.
    Unknown,
}

/// SLA compliance result for a window.
#[derive(Debug, Clone)]
pub struct SlaReport {
    /// Jobs that completed within the SLA threshold.
    pub within_sla: usize,
    /// Jobs that exceeded the SLA threshold (including failures).
    pub breached_sla: usize,
    /// Compliance fraction (0.0 ..= 1.0).
    pub compliance_rate: f64,
    /// The threshold used.
    pub threshold_secs: f64,
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for [`BatchAnalytics`].
#[derive(Debug, Clone)]
pub struct AnalyticsConfig {
    /// Maximum number of samples retained in the ring-buffer.
    pub max_samples: usize,
    /// Optional SLA target in seconds; used by [`BatchAnalytics::sla_report`].
    pub sla_target_secs: Option<f64>,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            max_samples: 100_000,
            sla_target_secs: None,
        }
    }
}

// ---------------------------------------------------------------------------
// BatchAnalytics
// ---------------------------------------------------------------------------

/// Analytics engine for batch processing workloads.
///
/// # Thread safety
///
/// `BatchAnalytics` wraps its ring-buffer in a `std::sync::Mutex` internally
/// via [`parking_lot::Mutex`].  All public methods take `&self` and are safe
/// to call concurrently.
///
/// # Example
///
/// ```
/// use oximedia_batch::batch_analytics::{BatchAnalytics, AnalyticsConfig, JobSample, Window};
/// use oximedia_batch::types::JobState;
///
/// let analytics = BatchAnalytics::new(AnalyticsConfig::default());
/// analytics.ingest(JobSample::now("transcode", JobState::Completed, 12.5));
/// analytics.ingest(JobSample::now("transcode", JobState::Failed, 0.5));
///
/// let metrics = analytics.metrics(Window::LastHour);
/// assert_eq!(metrics.total_jobs, 2);
/// assert!((metrics.failure_rate - 0.5).abs() < 1e-4);
/// ```
pub struct BatchAnalytics {
    samples: parking_lot::Mutex<VecDeque<JobSample>>,
    config: AnalyticsConfig,
}

impl BatchAnalytics {
    /// Create a new analytics engine.
    #[must_use]
    pub fn new(config: AnalyticsConfig) -> Self {
        Self {
            samples: parking_lot::Mutex::new(VecDeque::new()),
            config,
        }
    }

    // -----------------------------------------------------------------------
    // Ingestion
    // -----------------------------------------------------------------------

    /// Ingest a completed job sample.
    pub fn ingest(&self, sample: JobSample) {
        let mut ring = self.samples.lock();
        ring.push_back(sample);
        while ring.len() > self.config.max_samples {
            ring.pop_front();
        }
    }

    /// Ingest multiple samples at once.
    pub fn ingest_batch(&self, samples: impl IntoIterator<Item = JobSample>) {
        let mut ring = self.samples.lock();
        for s in samples {
            ring.push_back(s);
        }
        while ring.len() > self.config.max_samples {
            ring.pop_front();
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs()
    }

    /// Extract all samples whose `completed_at` falls in `[since, now]`.
    fn samples_in_window(&self, window: Window) -> Vec<JobSample> {
        let now = Self::now_secs();
        let since = now.saturating_sub(window.secs());
        let ring = self.samples.lock();
        ring.iter()
            .filter(|s| s.completed_at >= since)
            .cloned()
            .collect()
    }

    /// Extract samples in the previous equal-length window (for trend calc).
    fn samples_in_prev_window(&self, window: Window) -> Vec<JobSample> {
        let now = Self::now_secs();
        let width = window.secs();
        let end = now.saturating_sub(width);
        let start = end.saturating_sub(width);
        let ring = self.samples.lock();
        ring.iter()
            .filter(|s| s.completed_at >= start && s.completed_at < end)
            .cloned()
            .collect()
    }

    // -----------------------------------------------------------------------
    // Aggregate metrics
    // -----------------------------------------------------------------------

    /// Compute aggregated metrics for `window`.
    #[must_use]
    pub fn metrics(&self, window: Window) -> WindowMetrics {
        let samples = self.samples_in_window(window);
        Self::compute_metrics(&samples, window.secs())
    }

    fn compute_metrics(samples: &[JobSample], window_secs: u64) -> WindowMetrics {
        if samples.is_empty() {
            return WindowMetrics::default();
        }

        let total_jobs = samples.len();
        let completed = samples.iter().filter(|s| s.is_success()).count();
        let failed = samples.iter().filter(|s| s.is_failure()).count();
        let cancelled = samples
            .iter()
            .filter(|s| matches!(s.state, JobState::Cancelled))
            .count();
        let total_output_bytes: u64 = samples.iter().map(|s| s.output_bytes).sum();

        // Duration stats.
        let mut durations: Vec<f64> = samples.iter().map(|s| s.duration_secs).collect();
        durations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean_duration_secs =
            durations.iter().sum::<f64>() / total_jobs as f64;
        let p50 = percentile(&durations, 0.50);
        let p95 = percentile(&durations, 0.95);
        let p99 = percentile(&durations, 0.99);
        let max_duration_secs = durations.last().copied().unwrap_or(0.0);

        let failure_rate = if total_jobs > 0 {
            failed as f64 / total_jobs as f64
        } else {
            0.0
        };

        let throughput_per_sec = if window_secs > 0 {
            completed as f64 / window_secs as f64
        } else {
            0.0
        };

        WindowMetrics {
            total_jobs,
            completed,
            failed,
            cancelled,
            mean_duration_secs,
            p50_duration_secs: p50,
            p95_duration_secs: p95,
            p99_duration_secs: p99,
            max_duration_secs,
            failure_rate,
            total_output_bytes,
            throughput_per_sec,
        }
    }

    // -----------------------------------------------------------------------
    // SLA
    // -----------------------------------------------------------------------

    /// Compute SLA compliance within `window` against `threshold_secs`.
    ///
    /// A job "breaches" SLA if its `duration_secs` exceeds `threshold_secs`
    /// OR if it failed or was cancelled.
    #[must_use]
    pub fn sla_report(&self, window: Window, threshold_secs: f64) -> SlaReport {
        let samples = self.samples_in_window(window);
        let total = samples.len();

        if total == 0 {
            return SlaReport {
                within_sla: 0,
                breached_sla: 0,
                compliance_rate: 1.0,
                threshold_secs,
            };
        }

        let within_sla = samples
            .iter()
            .filter(|s| s.is_success() && s.duration_secs <= threshold_secs)
            .count();
        let breached_sla = total - within_sla;
        let compliance_rate = within_sla as f64 / total as f64;

        SlaReport {
            within_sla,
            breached_sla,
            compliance_rate,
            threshold_secs,
        }
    }

    // -----------------------------------------------------------------------
    // Trend
    // -----------------------------------------------------------------------

    /// Compute the throughput trend direction by comparing the current `window`
    /// to the previous window of the same length.
    #[must_use]
    pub fn throughput_trend(&self, window: Window) -> TrendDirection {
        let current = self.metrics(window);
        let prev_samples = self.samples_in_prev_window(window);

        if prev_samples.is_empty() {
            return TrendDirection::Unknown;
        }

        let prev = Self::compute_metrics(&prev_samples, window.secs());

        let curr_tp = current.throughput_per_sec;
        let prev_tp = prev.throughput_per_sec;

        if prev_tp == 0.0 {
            if curr_tp > 0.0 {
                return TrendDirection::Improving;
            }
            return TrendDirection::Unknown;
        }

        let ratio = curr_tp / prev_tp;
        if ratio >= 1.05 {
            TrendDirection::Improving
        } else if ratio <= 0.95 {
            TrendDirection::Degrading
        } else {
            TrendDirection::Stable
        }
    }

    // -----------------------------------------------------------------------
    // Failure breakdown
    // -----------------------------------------------------------------------

    /// Return per-category failure rates within `window`.
    ///
    /// The map key is the category; the value is `(failed, total)`.
    #[must_use]
    pub fn failure_breakdown(
        &self,
        window: Window,
    ) -> std::collections::HashMap<String, (usize, usize)> {
        let samples = self.samples_in_window(window);
        let mut map: std::collections::HashMap<String, (usize, usize)> =
            std::collections::HashMap::new();

        for s in &samples {
            let entry = map.entry(s.category.clone()).or_insert((0, 0));
            entry.1 += 1; // total
            if s.is_failure() {
                entry.0 += 1; // failed
            }
        }
        map
    }

    // -----------------------------------------------------------------------
    // Top categories
    // -----------------------------------------------------------------------

    /// Return the `n` categories with the highest job count in `window`.
    ///
    /// Returns a `Vec<(category, count)>` sorted descending by count.
    #[must_use]
    pub fn top_categories(&self, window: Window, n: usize) -> Vec<(String, usize)> {
        let samples = self.samples_in_window(window);
        let mut map: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for s in &samples {
            *map.entry(s.category.clone()).or_insert(0) += 1;
        }
        let mut v: Vec<_> = map.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v.truncate(n);
        v
    }

    // -----------------------------------------------------------------------
    // Capacity
    // -----------------------------------------------------------------------

    /// Number of samples currently retained.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.lock().len()
    }

    /// Discard all retained samples.
    pub fn clear(&self) {
        self.samples.lock().clear();
    }
}

// ---------------------------------------------------------------------------
// Percentile helper
// ---------------------------------------------------------------------------

/// Compute the `p`-th percentile (0.0 ..= 1.0) of a *sorted* slice.
/// Returns 0.0 for empty slices.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = p * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::JobState;

    fn make_sample(state: JobState, duration: f64) -> JobSample {
        JobSample::now("test", state, duration)
    }

    // -----------------------------------------------------------------------
    // Percentile helper
    // -----------------------------------------------------------------------

    #[test]
    fn test_percentile_single_element() {
        assert!((percentile(&[5.0], 0.5) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile_sorted_range() {
        let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        // P50 of 1..100 should be near 50.
        let p50 = percentile(&data, 0.50);
        assert!(p50 >= 49.0 && p50 <= 51.0, "p50={p50}");
        // P99 should be near 99.
        let p99 = percentile(&data, 0.99);
        assert!(p99 >= 97.0 && p99 <= 100.0, "p99={p99}");
    }

    #[test]
    fn test_percentile_empty_returns_zero() {
        assert_eq!(percentile(&[], 0.5), 0.0);
    }

    // -----------------------------------------------------------------------
    // BatchAnalytics — ingestion
    // -----------------------------------------------------------------------

    #[test]
    fn test_ingest_and_sample_count() {
        let a = BatchAnalytics::new(AnalyticsConfig::default());
        a.ingest(make_sample(JobState::Completed, 1.0));
        a.ingest(make_sample(JobState::Failed, 0.5));
        assert_eq!(a.sample_count(), 2);
    }

    #[test]
    fn test_ingest_batch() {
        let a = BatchAnalytics::new(AnalyticsConfig::default());
        let samples: Vec<_> = (0..10).map(|_| make_sample(JobState::Completed, 2.0)).collect();
        a.ingest_batch(samples);
        assert_eq!(a.sample_count(), 10);
    }

    #[test]
    fn test_ring_buffer_max_capacity() {
        let cfg = AnalyticsConfig {
            max_samples: 3,
            sla_target_secs: None,
        };
        let a = BatchAnalytics::new(cfg);
        for _ in 0..10 {
            a.ingest(make_sample(JobState::Completed, 1.0));
        }
        assert_eq!(a.sample_count(), 3);
    }

    // -----------------------------------------------------------------------
    // Metrics
    // -----------------------------------------------------------------------

    #[test]
    fn test_metrics_all_successful() {
        let a = BatchAnalytics::new(AnalyticsConfig::default());
        a.ingest(make_sample(JobState::Completed, 10.0));
        a.ingest(make_sample(JobState::Completed, 20.0));
        let m = a.metrics(Window::LastHour);
        assert_eq!(m.total_jobs, 2);
        assert_eq!(m.completed, 2);
        assert_eq!(m.failed, 0);
        assert!((m.failure_rate).abs() < 1e-9);
    }

    #[test]
    fn test_metrics_failure_rate() {
        let a = BatchAnalytics::new(AnalyticsConfig::default());
        a.ingest(make_sample(JobState::Completed, 5.0));
        a.ingest(make_sample(JobState::Failed, 1.0));
        let m = a.metrics(Window::LastHour);
        assert!((m.failure_rate - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_metrics_empty_window() {
        let a = BatchAnalytics::new(AnalyticsConfig::default());
        let m = a.metrics(Window::LastMinute);
        assert_eq!(m.total_jobs, 0);
        assert!((m.failure_rate).abs() < 1e-9);
    }

    #[test]
    fn test_metrics_duration_percentiles() {
        let a = BatchAnalytics::new(AnalyticsConfig::default());
        for i in 1..=100 {
            a.ingest(make_sample(JobState::Completed, i as f64));
        }
        let m = a.metrics(Window::LastHour);
        assert!(m.p50_duration_secs >= 49.0 && m.p50_duration_secs <= 51.0);
        assert!(m.max_duration_secs >= 99.0 && m.max_duration_secs <= 101.0);
    }

    // -----------------------------------------------------------------------
    // SLA
    // -----------------------------------------------------------------------

    #[test]
    fn test_sla_all_within() {
        let a = BatchAnalytics::new(AnalyticsConfig::default());
        a.ingest(make_sample(JobState::Completed, 5.0));
        a.ingest(make_sample(JobState::Completed, 3.0));
        let report = a.sla_report(Window::LastHour, 10.0);
        assert_eq!(report.within_sla, 2);
        assert!((report.compliance_rate - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_sla_some_breached() {
        let a = BatchAnalytics::new(AnalyticsConfig::default());
        a.ingest(make_sample(JobState::Completed, 5.0));
        a.ingest(make_sample(JobState::Completed, 15.0)); // breaches 10s threshold
        a.ingest(make_sample(JobState::Failed, 1.0));     // failure always breaches
        let report = a.sla_report(Window::LastHour, 10.0);
        assert_eq!(report.within_sla, 1);
        assert_eq!(report.breached_sla, 2);
        assert!((report.compliance_rate - 1.0 / 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_sla_empty_window_full_compliance() {
        let a = BatchAnalytics::new(AnalyticsConfig::default());
        let report = a.sla_report(Window::LastHour, 30.0);
        assert!((report.compliance_rate - 1.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // Failure breakdown
    // -----------------------------------------------------------------------

    #[test]
    fn test_failure_breakdown_per_category() {
        let a = BatchAnalytics::new(AnalyticsConfig::default());

        let mut s1 = make_sample(JobState::Completed, 5.0);
        s1.category = "encode".into();
        let mut s2 = make_sample(JobState::Failed, 1.0);
        s2.category = "encode".into();
        let mut s3 = make_sample(JobState::Failed, 2.0);
        s3.category = "ingest".into();

        a.ingest(s1);
        a.ingest(s2);
        a.ingest(s3);

        let breakdown = a.failure_breakdown(Window::LastHour);
        let (enc_fail, enc_total) = breakdown["encode"];
        assert_eq!(enc_total, 2);
        assert_eq!(enc_fail, 1);
        let (ing_fail, ing_total) = breakdown["ingest"];
        assert_eq!(ing_total, 1);
        assert_eq!(ing_fail, 1);
    }

    // -----------------------------------------------------------------------
    // Top categories
    // -----------------------------------------------------------------------

    #[test]
    fn test_top_categories_sorted_descending() {
        let a = BatchAnalytics::new(AnalyticsConfig::default());
        for _ in 0..5 {
            let mut s = make_sample(JobState::Completed, 1.0);
            s.category = "encode".into();
            a.ingest(s);
        }
        for _ in 0..2 {
            let mut s = make_sample(JobState::Completed, 1.0);
            s.category = "ingest".into();
            a.ingest(s);
        }
        let top = a.top_categories(Window::LastHour, 2);
        assert_eq!(top[0].0, "encode");
        assert_eq!(top[0].1, 5);
        assert_eq!(top[1].0, "ingest");
    }

    // -----------------------------------------------------------------------
    // Trend
    // -----------------------------------------------------------------------

    #[test]
    fn test_trend_unknown_when_no_prior_data() {
        let a = BatchAnalytics::new(AnalyticsConfig::default());
        a.ingest(make_sample(JobState::Completed, 1.0));
        // No data in the previous window → Unknown.
        assert_eq!(a.throughput_trend(Window::LastMinute), TrendDirection::Unknown);
    }

    // -----------------------------------------------------------------------
    // Clear
    // -----------------------------------------------------------------------

    #[test]
    fn test_clear_empties_buffer() {
        let a = BatchAnalytics::new(AnalyticsConfig::default());
        a.ingest(make_sample(JobState::Completed, 1.0));
        a.clear();
        assert_eq!(a.sample_count(), 0);
        let m = a.metrics(Window::LastHour);
        assert_eq!(m.total_jobs, 0);
    }

    // -----------------------------------------------------------------------
    // JobSample helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_job_sample_is_success_failure() {
        let s_ok = make_sample(JobState::Completed, 1.0);
        let s_fail = make_sample(JobState::Failed, 0.5);
        let s_cancel = make_sample(JobState::Cancelled, 0.1);
        assert!(s_ok.is_success());
        assert!(!s_ok.is_failure());
        assert!(s_fail.is_failure());
        assert!(!s_cancel.is_success());
        assert!(!s_cancel.is_failure());
    }
}
