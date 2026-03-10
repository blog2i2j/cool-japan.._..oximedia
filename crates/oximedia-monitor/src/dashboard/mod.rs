//! Dashboard data aggregation and time-series retrieval.
//!
//! Provides a lightweight data layer for building operational dashboards.
//! A `Dashboard` holds a ring buffer of recent `DashboardSnapshot` values
//! and can answer queries for historical time-series data or the most
//! frequent recent errors.

use chrono::{DateTime, Duration, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// A single time-stamped data point for a named metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    /// Timestamp of the measurement.
    pub timestamp: DateTime<Utc>,
    /// Numeric value.
    pub value: f64,
}

impl DataPoint {
    /// Create a new data point with `Utc::now()` as the timestamp.
    #[must_use]
    pub fn now(value: f64) -> Self {
        Self {
            timestamp: Utc::now(),
            value,
        }
    }
}

/// Summary of a recurring error type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorSummary {
    /// Human-readable error description.
    pub message: String,
    /// Component or subsystem where the error occurred.
    pub component: String,
    /// Number of occurrences in the recent window.
    pub count: usize,
    /// Timestamp of the first occurrence in the window.
    pub first_seen: DateTime<Utc>,
    /// Timestamp of the most recent occurrence.
    pub last_seen: DateTime<Utc>,
}

/// A snapshot of all key operational metrics at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardSnapshot {
    /// Snapshot timestamp.
    pub timestamp: DateTime<Utc>,
    /// CPU utilisation 0–100 %.
    pub cpu_percent: f64,
    /// Memory utilisation 0–100 %.
    pub memory_percent: f64,
    /// Number of active media streams.
    pub active_streams: u64,
    /// Frames encoded per second.
    pub encoding_fps: f64,
    /// Current video bitrate in bits per second.
    pub video_bitrate_bps: u64,
    /// Error rate (errors per second over the last sample window).
    pub error_rate: f64,
    /// Number of queued transcoding jobs.
    pub queued_jobs: u64,
    /// Number of running transcoding jobs.
    pub running_jobs: u64,
    /// Number of completed transcoding jobs (all-time total).
    pub completed_jobs: u64,
    /// PSNR quality score (if available).
    pub psnr: Option<f64>,
    /// SSIM quality score (if available).
    pub ssim: Option<f64>,
    /// VMAF quality score (if available).
    pub vmaf: Option<f64>,
}

impl Default for DashboardSnapshot {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            cpu_percent: 0.0,
            memory_percent: 0.0,
            active_streams: 0,
            encoding_fps: 0.0,
            video_bitrate_bps: 0,
            error_rate: 0.0,
            queued_jobs: 0,
            running_jobs: 0,
            completed_jobs: 0,
            psnr: None,
            ssim: None,
            vmaf: None,
        }
    }
}

/// An error event recorded for dashboard display.
#[derive(Debug, Clone)]
struct ErrorEvent {
    timestamp: DateTime<Utc>,
    component: String,
    message: String,
}

/// Dashboard data aggregator.
///
/// Maintains a rolling window of [`DashboardSnapshot`] values and recent
/// error events.  Feed data in via [`Dashboard::push_snapshot`] and
/// [`Dashboard::record_error`], then query with [`Dashboard::current_snapshot`],
/// [`Dashboard::time_series`], and [`Dashboard::top_errors`].
pub struct Dashboard {
    /// Maximum number of snapshots to retain.
    capacity: usize,
    snapshots: Arc<RwLock<VecDeque<DashboardSnapshot>>>,
    errors: Arc<RwLock<VecDeque<ErrorEvent>>>,
    error_capacity: usize,
}

impl Dashboard {
    /// Create a new dashboard with the given snapshot buffer capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            snapshots: Arc::new(RwLock::new(VecDeque::with_capacity(capacity))),
            errors: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            error_capacity: 1000,
        }
    }

    /// Push a new snapshot into the ring buffer, evicting the oldest if full.
    pub fn push_snapshot(&self, snapshot: DashboardSnapshot) {
        let mut buf = self.snapshots.write();
        if buf.len() == self.capacity {
            buf.pop_front();
        }
        buf.push_back(snapshot);
    }

    /// Record an error event (component + message) for aggregation.
    pub fn record_error(&self, component: impl Into<String>, message: impl Into<String>) {
        let mut buf = self.errors.write();
        if buf.len() == self.error_capacity {
            buf.pop_front();
        }
        buf.push_back(ErrorEvent {
            timestamp: Utc::now(),
            component: component.into(),
            message: message.into(),
        });
    }

    /// Return the most recently pushed snapshot, or a default if none exists.
    #[must_use]
    pub fn current_snapshot(&self) -> DashboardSnapshot {
        self.snapshots.read().back().cloned().unwrap_or_default()
    }

    /// Return a time series of a named scalar metric between `start` and `end`,
    /// downsampled to approximately `resolution` points.
    ///
    /// Supported metric names (case-insensitive):
    /// - `cpu`
    /// - `memory`
    /// - `encoding_fps`
    /// - `video_bitrate`
    /// - `error_rate`
    /// - `active_streams`
    /// - `queued_jobs`
    /// - `running_jobs`
    ///
    /// Returns an empty `Vec` for unknown metric names.
    #[must_use]
    pub fn time_series(
        &self,
        metric: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        resolution: usize,
    ) -> Vec<DataPoint> {
        let snapshots = self.snapshots.read();

        // Collect all snapshots in the requested range.
        let filtered: Vec<&DashboardSnapshot> = snapshots
            .iter()
            .filter(|s| s.timestamp >= start && s.timestamp <= end)
            .collect();

        if filtered.is_empty() || resolution == 0 {
            return Vec::new();
        }

        let extractor: Box<dyn Fn(&DashboardSnapshot) -> f64> = match metric.to_lowercase().as_str()
        {
            "cpu" => Box::new(|s| s.cpu_percent),
            "memory" => Box::new(|s| s.memory_percent),
            "encoding_fps" => Box::new(|s| s.encoding_fps),
            "video_bitrate" => Box::new(|s| s.video_bitrate_bps as f64),
            "error_rate" => Box::new(|s| s.error_rate),
            "active_streams" => Box::new(|s| s.active_streams as f64),
            "queued_jobs" => Box::new(|s| s.queued_jobs as f64),
            "running_jobs" => Box::new(|s| s.running_jobs as f64),
            _ => return Vec::new(),
        };

        // Downsample: divide the filtered points into `resolution` buckets and
        // average each bucket.
        let bucket_size = ((filtered.len() as f64) / (resolution as f64)).ceil() as usize;
        let bucket_size = bucket_size.max(1);

        filtered
            .chunks(bucket_size)
            .map(|chunk| {
                let avg = chunk.iter().map(|s| extractor(s)).sum::<f64>() / chunk.len() as f64;
                let mid_ts = chunk[chunk.len() / 2].timestamp;
                DataPoint {
                    timestamp: mid_ts,
                    value: avg,
                }
            })
            .collect()
    }

    /// Return the top `n` most frequent errors seen within the last `window`.
    ///
    /// Results are sorted descending by occurrence count.
    #[must_use]
    pub fn top_errors(&self, n: usize) -> Vec<ErrorSummary> {
        let window_start = Utc::now() - Duration::hours(1);
        self.top_errors_since(n, window_start)
    }

    /// Return the top `n` most frequent errors since `since`.
    #[must_use]
    pub fn top_errors_since(&self, n: usize, since: DateTime<Utc>) -> Vec<ErrorSummary> {
        let errors = self.errors.read();

        // Group by (component, message).
        let mut groups: HashMap<(String, String), Vec<DateTime<Utc>>> = HashMap::new();
        for evt in errors.iter() {
            if evt.timestamp >= since {
                groups
                    .entry((evt.component.clone(), evt.message.clone()))
                    .or_default()
                    .push(evt.timestamp);
            }
        }

        let mut summaries: Vec<ErrorSummary> = groups
            .into_iter()
            .map(|((component, message), timestamps)| {
                let now = Utc::now();
                let first_seen = timestamps.iter().copied().min().unwrap_or(now);
                let last_seen = timestamps.iter().copied().max().unwrap_or(now);
                ErrorSummary {
                    message,
                    component,
                    count: timestamps.len(),
                    first_seen,
                    last_seen,
                }
            })
            .collect();

        summaries.sort_by(|a, b| b.count.cmp(&a.count));
        summaries.truncate(n);
        summaries
    }

    /// Return the number of stored snapshots.
    #[must_use]
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.read().len()
    }

    /// Return the number of stored error events.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.errors.read().len()
    }
}

impl Default for Dashboard {
    fn default() -> Self {
        // Default capacity covers 24 h at 1-second resolution.
        Self::new(86_400)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshot(cpu: f64, memory: f64, streams: u64) -> DashboardSnapshot {
        DashboardSnapshot {
            timestamp: Utc::now(),
            cpu_percent: cpu,
            memory_percent: memory,
            active_streams: streams,
            ..Default::default()
        }
    }

    #[test]
    fn test_push_and_current_snapshot() {
        let dashboard = Dashboard::new(10);
        dashboard.push_snapshot(make_snapshot(45.0, 60.0, 3));

        let snap = dashboard.current_snapshot();
        assert_eq!(snap.cpu_percent, 45.0);
        assert_eq!(snap.memory_percent, 60.0);
        assert_eq!(snap.active_streams, 3);
    }

    #[test]
    fn test_ring_buffer_eviction() {
        let dashboard = Dashboard::new(3);
        for i in 0..5u64 {
            dashboard.push_snapshot(make_snapshot(i as f64, 0.0, i));
        }
        assert_eq!(dashboard.snapshot_count(), 3);
        // Oldest three active_streams values should be 2, 3, 4.
        let snap = dashboard.current_snapshot();
        assert_eq!(snap.active_streams, 4);
    }

    #[test]
    fn test_current_snapshot_default_when_empty() {
        let dashboard = Dashboard::new(10);
        let snap = dashboard.current_snapshot();
        assert_eq!(snap.cpu_percent, 0.0);
    }

    #[test]
    fn test_time_series_cpu() {
        let dashboard = Dashboard::new(100);

        let base = Utc::now() - Duration::seconds(10);
        for i in 0..10i64 {
            let mut snap = DashboardSnapshot::default();
            snap.timestamp = base + Duration::seconds(i);
            snap.cpu_percent = i as f64 * 10.0;
            dashboard.push_snapshot(snap);
        }

        let start = base - Duration::seconds(1);
        let end = base + Duration::seconds(11);
        let series = dashboard.time_series("cpu", start, end, 10);
        assert!(!series.is_empty());
        // All values should be in 0–90 range.
        for dp in &series {
            assert!(dp.value >= 0.0 && dp.value <= 90.0);
        }
    }

    #[test]
    fn test_time_series_unknown_metric() {
        let dashboard = Dashboard::new(100);
        dashboard.push_snapshot(DashboardSnapshot::default());
        let series = dashboard.time_series(
            "nonexistent",
            Utc::now() - Duration::hours(1),
            Utc::now(),
            10,
        );
        assert!(series.is_empty());
    }

    #[test]
    fn test_time_series_resolution_downsampling() {
        let dashboard = Dashboard::new(1000);
        let base = Utc::now() - Duration::seconds(100);
        for i in 0..100i64 {
            let mut snap = DashboardSnapshot::default();
            snap.timestamp = base + Duration::seconds(i);
            snap.cpu_percent = 50.0;
            dashboard.push_snapshot(snap);
        }

        let series = dashboard.time_series("cpu", base, base + Duration::seconds(100), 10);
        // Should be approx 10 points.
        assert!(series.len() <= 10 + 1);
    }

    #[test]
    fn test_top_errors() {
        let dashboard = Dashboard::new(100);
        dashboard.record_error("encoder", "out of memory");
        dashboard.record_error("encoder", "out of memory");
        dashboard.record_error("encoder", "out of memory");
        dashboard.record_error("db", "connection timeout");
        dashboard.record_error("db", "connection timeout");
        dashboard.record_error("api", "bad request");

        let top = dashboard.top_errors(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].count, 3);
        assert_eq!(top[0].message, "out of memory");
        assert_eq!(top[1].count, 2);
    }

    #[test]
    fn test_top_errors_empty() {
        let dashboard = Dashboard::new(100);
        let top = dashboard.top_errors(5);
        assert!(top.is_empty());
    }

    #[test]
    fn test_top_errors_limit() {
        let dashboard = Dashboard::new(100);
        for i in 0..10 {
            dashboard.record_error("test", format!("error {i}"));
        }
        let top = dashboard.top_errors(3);
        assert_eq!(top.len(), 3);
    }

    #[test]
    fn test_error_ring_buffer_eviction() {
        let dashboard = Dashboard {
            capacity: 10,
            snapshots: Arc::new(RwLock::new(VecDeque::new())),
            errors: Arc::new(RwLock::new(VecDeque::with_capacity(5))),
            error_capacity: 5,
        };
        for i in 0..7 {
            dashboard.record_error("test", format!("msg {i}"));
        }
        assert_eq!(dashboard.error_count(), 5);
    }
}
