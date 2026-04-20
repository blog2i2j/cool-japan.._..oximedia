//! Analytics and metrics for live streaming.
//!
//! This module provides comprehensive analytics including:
//! - Stream metrics (bitrate, fps, quality)
//! - Viewer metrics (concurrent viewers, watch time)
//! - Performance metrics (latency, buffering)
//! - CDN metrics (cache hit rate, bandwidth)

use super::MediaPacket;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Duration;
use uuid::Uuid;

/// Stream metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamMetrics {
    /// Stream ID.
    pub stream_id: Uuid,

    /// Current video bitrate (bits per second).
    pub video_bitrate: u64,

    /// Current audio bitrate (bits per second).
    pub audio_bitrate: u64,

    /// Current framerate.
    pub framerate: f64,

    /// Dropped frames.
    pub dropped_frames: u64,

    /// Keyframe interval.
    pub keyframe_interval: u64,

    /// Total packets received.
    pub total_packets: u64,

    /// Total bytes received.
    pub total_bytes: u64,

    /// Average latency (milliseconds).
    pub avg_latency: u64,

    /// Packet loss rate (percentage).
    pub packet_loss_rate: f64,
}

impl StreamMetrics {
    /// Creates new stream metrics.
    #[must_use]
    pub fn new(stream_id: Uuid) -> Self {
        Self {
            stream_id,
            video_bitrate: 0,
            audio_bitrate: 0,
            framerate: 0.0,
            dropped_frames: 0,
            keyframe_interval: 0,
            total_packets: 0,
            total_bytes: 0,
            avg_latency: 0,
            packet_loss_rate: 0.0,
        }
    }
}

/// Viewer metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewerMetrics {
    /// Viewer ID.
    pub viewer_id: String,

    /// Stream ID being watched.
    pub stream_id: Uuid,

    /// Join time.
    pub join_time: DateTime<Utc>,

    /// Leave time (if left).
    pub leave_time: Option<DateTime<Utc>>,

    /// Total watch time.
    pub watch_time: Duration,

    /// Rebuffering events.
    pub rebuffer_count: u64,

    /// Total rebuffering duration.
    pub rebuffer_duration: Duration,

    /// Average bitrate consumed.
    pub avg_bitrate: u64,

    /// Quality switches.
    pub quality_switches: u64,

    /// Current quality variant.
    pub current_quality: Option<String>,

    /// Geographic location.
    pub location: Option<String>,

    /// User agent.
    pub user_agent: Option<String>,
}

impl ViewerMetrics {
    /// Creates new viewer metrics.
    #[must_use]
    pub fn new(viewer_id: impl Into<String>, stream_id: Uuid) -> Self {
        Self {
            viewer_id: viewer_id.into(),
            stream_id,
            join_time: Utc::now(),
            leave_time: None,
            watch_time: Duration::ZERO,
            rebuffer_count: 0,
            rebuffer_duration: Duration::ZERO,
            avg_bitrate: 0,
            quality_switches: 0,
            current_quality: None,
            location: None,
            user_agent: None,
        }
    }

    /// Marks viewer as left.
    pub fn leave(&mut self) {
        self.leave_time = Some(Utc::now());
        if let Ok(duration) = (Utc::now() - self.join_time).to_std() {
            self.watch_time = duration;
        }
    }

    /// Records a rebuffering event.
    pub fn record_rebuffer(&mut self, duration: Duration) {
        self.rebuffer_count += 1;
        self.rebuffer_duration += duration;
    }

    /// Records a quality switch.
    pub fn record_quality_switch(&mut self, new_quality: impl Into<String>) {
        self.quality_switches += 1;
        self.current_quality = Some(new_quality.into());
    }
}

/// Time-series data point.
#[derive(Debug, Clone)]
struct DataPoint {
    timestamp: DateTime<Utc>,
    value: f64,
}

/// Time-series buffer.
struct TimeSeries {
    data: VecDeque<DataPoint>,
    max_points: usize,
    window: Duration,
}

impl TimeSeries {
    fn new(max_points: usize, window: Duration) -> Self {
        Self {
            data: VecDeque::new(),
            max_points,
            window,
        }
    }

    fn add(&mut self, value: f64) {
        let now = Utc::now();
        self.data.push_back(DataPoint {
            timestamp: now,
            value,
        });

        // Remove old data points
        let cutoff = now - chrono::Duration::from_std(self.window).unwrap_or_default();
        while let Some(first) = self.data.front() {
            if first.timestamp < cutoff {
                self.data.pop_front();
            } else {
                break;
            }
        }

        // Limit size
        if self.data.len() > self.max_points {
            self.data.pop_front();
        }
    }

    fn average(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }

        let sum: f64 = self.data.iter().map(|p| p.value).sum();
        sum / self.data.len() as f64
    }

    fn max(&self) -> f64 {
        self.data
            .iter()
            .map(|p| p.value)
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(0.0)
    }

    fn min(&self) -> f64 {
        self.data
            .iter()
            .map(|p| p.value)
            .min_by(|a, b| a.total_cmp(b))
            .unwrap_or(0.0)
    }
}

/// Analytics tracker.
pub struct Analytics {
    /// Stream ID.
    stream_id: Uuid,

    /// Stream metrics.
    stream_metrics: RwLock<StreamMetrics>,

    /// Active viewers.
    viewers: RwLock<HashMap<String, ViewerMetrics>>,

    /// Bitrate time series.
    bitrate_series: RwLock<TimeSeries>,

    /// Framerate time series.
    framerate_series: RwLock<TimeSeries>,

    /// Viewer count time series.
    viewer_count_series: RwLock<TimeSeries>,

    /// Latency time series.
    latency_series: RwLock<TimeSeries>,

    /// Total watch time across all viewers.
    total_watch_time: RwLock<Duration>,
}

impl Analytics {
    /// Creates a new analytics tracker.
    #[must_use]
    pub fn new(stream_id: Uuid) -> Self {
        Self {
            stream_id,
            stream_metrics: RwLock::new(StreamMetrics::new(stream_id)),
            viewers: RwLock::new(HashMap::new()),
            bitrate_series: RwLock::new(TimeSeries::new(1000, Duration::from_secs(300))),
            framerate_series: RwLock::new(TimeSeries::new(1000, Duration::from_secs(300))),
            viewer_count_series: RwLock::new(TimeSeries::new(1000, Duration::from_secs(3600))),
            latency_series: RwLock::new(TimeSeries::new(1000, Duration::from_secs(300))),
            total_watch_time: RwLock::new(Duration::ZERO),
        }
    }

    /// Records a media packet.
    pub fn record_packet(&self, packet: &MediaPacket) {
        let mut metrics = self.stream_metrics.write();
        metrics.total_packets += 1;
        metrics.total_bytes += packet.data.len() as u64;

        // Update bitrate (simple moving average)
        let bitrate = (packet.data.len() * 8) as f64 / packet.duration as f64 * 1000.0;
        self.bitrate_series.write().add(bitrate);

        match packet.media_type {
            super::MediaType::Video => {
                if packet.keyframe {
                    // Update framerate
                    self.framerate_series.write().add(30.0); // Placeholder
                }
            }
            super::MediaType::Audio => {}
            super::MediaType::Metadata => {}
        }
    }

    /// Adds a viewer.
    pub fn add_viewer(&self, viewer_id: impl Into<String>) -> ViewerMetrics {
        let viewer_id = viewer_id.into();
        let metrics = ViewerMetrics::new(&viewer_id, self.stream_id);

        let mut viewers = self.viewers.write();
        viewers.insert(viewer_id.clone(), metrics.clone());

        // Update viewer count
        self.viewer_count_series.write().add(viewers.len() as f64);

        metrics
    }

    /// Removes a viewer.
    pub fn remove_viewer(&self, viewer_id: &str) {
        let mut viewers = self.viewers.write();

        if let Some(mut viewer) = viewers.remove(viewer_id) {
            viewer.leave();
            *self.total_watch_time.write() += viewer.watch_time;
        }

        // Update viewer count
        self.viewer_count_series.write().add(viewers.len() as f64);
    }

    /// Records a viewer rebuffering event.
    pub fn record_viewer_rebuffer(&self, viewer_id: &str, duration: Duration) {
        let mut viewers = self.viewers.write();
        if let Some(viewer) = viewers.get_mut(viewer_id) {
            viewer.record_rebuffer(duration);
        }
    }

    /// Records a viewer quality switch.
    pub fn record_viewer_quality_switch(&self, viewer_id: &str, quality: impl Into<String>) {
        let mut viewers = self.viewers.write();
        if let Some(viewer) = viewers.get_mut(viewer_id) {
            viewer.record_quality_switch(quality);
        }
    }

    /// Gets stream metrics.
    #[must_use]
    pub fn stream_metrics(&self) -> StreamMetrics {
        let mut metrics = self.stream_metrics.read().clone();

        // Update computed metrics
        metrics.video_bitrate = self.bitrate_series.read().average() as u64;
        metrics.framerate = self.framerate_series.read().average();
        metrics.avg_latency = self.latency_series.read().average() as u64;

        metrics
    }

    /// Gets viewer metrics.
    #[must_use]
    pub fn viewer_metrics(&self, viewer_id: &str) -> Option<ViewerMetrics> {
        let viewers = self.viewers.read();
        viewers.get(viewer_id).cloned()
    }

    /// Gets all viewer metrics.
    #[must_use]
    pub fn all_viewer_metrics(&self) -> Vec<ViewerMetrics> {
        let viewers = self.viewers.read();
        viewers.values().cloned().collect()
    }

    /// Gets current viewer count.
    #[must_use]
    pub fn viewer_count(&self) -> usize {
        let viewers = self.viewers.read();
        viewers.len()
    }

    /// Gets peak viewer count.
    #[must_use]
    pub fn peak_viewer_count(&self) -> usize {
        self.viewer_count_series.read().max() as usize
    }

    /// Gets average viewer count.
    #[must_use]
    pub fn avg_viewer_count(&self) -> f64 {
        self.viewer_count_series.read().average()
    }

    /// Gets total watch time.
    #[must_use]
    pub fn total_watch_time(&self) -> Duration {
        *self.total_watch_time.read()
    }

    /// Records latency measurement.
    pub fn record_latency(&self, latency_ms: u64) {
        self.latency_series.write().add(latency_ms as f64);
    }

    /// Gets analytics summary.
    #[must_use]
    pub fn summary(&self) -> AnalyticsSummary {
        AnalyticsSummary {
            stream_id: self.stream_id,
            stream_metrics: self.stream_metrics(),
            current_viewers: self.viewer_count(),
            peak_viewers: self.peak_viewer_count(),
            avg_viewers: self.avg_viewer_count(),
            total_watch_time: self.total_watch_time(),
            avg_bitrate: self.bitrate_series.read().average() as u64,
            avg_framerate: self.framerate_series.read().average(),
            avg_latency: self.latency_series.read().average() as u64,
        }
    }
}

/// Analytics summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsSummary {
    /// Stream ID.
    pub stream_id: Uuid,

    /// Stream metrics.
    pub stream_metrics: StreamMetrics,

    /// Current viewers.
    pub current_viewers: usize,

    /// Peak viewers.
    pub peak_viewers: usize,

    /// Average viewers.
    pub avg_viewers: f64,

    /// Total watch time.
    pub total_watch_time: Duration,

    /// Average bitrate.
    pub avg_bitrate: u64,

    /// Average framerate.
    pub avg_framerate: f64,

    /// Average latency.
    pub avg_latency: u64,
}
