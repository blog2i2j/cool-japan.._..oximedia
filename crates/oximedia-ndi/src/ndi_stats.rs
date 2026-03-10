#![allow(dead_code)]
//! Real-time statistics collection and aggregation for NDI streams.
//!
//! Tracks frame counts, bandwidth, latency, and dropped frames for
//! both sending and receiving NDI connections.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Rolling-window statistics tracker for a single numeric metric.
#[derive(Debug, Clone)]
pub struct MetricWindow {
    /// Window of recorded samples.
    samples: VecDeque<f64>,
    /// Maximum number of samples to retain.
    capacity: usize,
}

impl MetricWindow {
    /// Creates a new metric window with the given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(capacity),
            capacity: capacity.max(1),
        }
    }

    /// Records a new sample value.
    pub fn push(&mut self, value: f64) {
        if self.samples.len() >= self.capacity {
            self.samples.pop_front();
        }
        self.samples.push_back(value);
    }

    /// Returns the number of recorded samples.
    #[must_use]
    pub fn count(&self) -> usize {
        self.samples.len()
    }

    /// Returns the arithmetic mean of all samples, or `None` if empty.
    #[must_use]
    pub fn mean(&self) -> Option<f64> {
        if self.samples.is_empty() {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        let avg = self.samples.iter().sum::<f64>() / self.samples.len() as f64;
        Some(avg)
    }

    /// Returns the minimum sample value, or `None` if empty.
    #[must_use]
    pub fn min(&self) -> Option<f64> {
        self.samples.iter().copied().reduce(f64::min)
    }

    /// Returns the maximum sample value, or `None` if empty.
    #[must_use]
    pub fn max(&self) -> Option<f64> {
        self.samples.iter().copied().reduce(f64::max)
    }

    /// Returns the most recent sample, or `None` if empty.
    #[must_use]
    pub fn last(&self) -> Option<f64> {
        self.samples.back().copied()
    }

    /// Returns the standard deviation of the samples, or `None` if fewer than 2 samples.
    #[must_use]
    pub fn std_dev(&self) -> Option<f64> {
        if self.samples.len() < 2 {
            return None;
        }
        let mean = self.mean()?;
        #[allow(clippy::cast_precision_loss)]
        let variance = self.samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>()
            / (self.samples.len() - 1) as f64;
        Some(variance.sqrt())
    }

    /// Clears all recorded samples.
    pub fn clear(&mut self) {
        self.samples.clear();
    }
}

/// Aggregated statistics for an NDI video stream.
#[derive(Debug, Clone)]
pub struct VideoStreamStats {
    /// Total number of video frames sent or received.
    pub total_frames: u64,
    /// Number of dropped video frames.
    pub dropped_frames: u64,
    /// Rolling frame-to-frame latency in milliseconds.
    pub latency_ms: MetricWindow,
    /// Rolling frame size in bytes.
    pub frame_size_bytes: MetricWindow,
    /// Timestamp of the most recently processed frame.
    pub last_frame_time: Option<Instant>,
    /// Measured frames per second over the window period.
    pub measured_fps: f64,
    /// Frame timestamps for FPS calculation.
    frame_times: VecDeque<Instant>,
    /// Maximum number of timestamps to keep for FPS calculation.
    fps_window: usize,
}

impl VideoStreamStats {
    /// Creates a new video stream statistics tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_frames: 0,
            dropped_frames: 0,
            latency_ms: MetricWindow::new(120),
            frame_size_bytes: MetricWindow::new(120),
            last_frame_time: None,
            measured_fps: 0.0,
            frame_times: VecDeque::with_capacity(120),
            fps_window: 120,
        }
    }

    /// Records a new video frame arrival.
    #[allow(clippy::cast_precision_loss)]
    pub fn record_frame(&mut self, size_bytes: u64, now: Instant) {
        self.total_frames += 1;
        self.frame_size_bytes.push(size_bytes as f64);

        if let Some(prev) = self.last_frame_time {
            let elapsed = now.duration_since(prev);
            self.latency_ms.push(elapsed.as_secs_f64() * 1000.0);
        }
        self.last_frame_time = Some(now);

        if self.frame_times.len() >= self.fps_window {
            self.frame_times.pop_front();
        }
        self.frame_times.push_back(now);
        self.update_fps();
    }

    /// Records one or more dropped frames.
    pub fn record_dropped(&mut self, count: u64) {
        self.dropped_frames += count;
    }

    /// Returns the drop rate as a fraction (0.0 to 1.0).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn drop_rate(&self) -> f64 {
        let total = self.total_frames + self.dropped_frames;
        if total == 0 {
            return 0.0;
        }
        self.dropped_frames as f64 / total as f64
    }

    /// Recalculates the measured FPS from the frame timestamp window.
    #[allow(clippy::cast_precision_loss)]
    fn update_fps(&mut self) {
        if self.frame_times.len() < 2 {
            self.measured_fps = 0.0;
            return;
        }
        // Safety: len() >= 2 was verified above, so front() and back() are always Some.
        let first = match self.frame_times.front() {
            Some(t) => t,
            None => return,
        };
        let last = match self.frame_times.back() {
            Some(t) => t,
            None => return,
        };
        let elapsed = last.duration_since(*first).as_secs_f64();
        if elapsed > 0.0 {
            self.measured_fps = (self.frame_times.len() - 1) as f64 / elapsed;
        }
    }

    /// Resets all statistics.
    pub fn reset(&mut self) {
        self.total_frames = 0;
        self.dropped_frames = 0;
        self.latency_ms.clear();
        self.frame_size_bytes.clear();
        self.last_frame_time = None;
        self.measured_fps = 0.0;
        self.frame_times.clear();
    }
}

impl Default for VideoStreamStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregated statistics for an NDI audio stream.
#[derive(Debug, Clone)]
pub struct AudioStreamStats {
    /// Total number of audio frames processed.
    pub total_frames: u64,
    /// Total number of audio samples processed across all channels.
    pub total_samples: u64,
    /// Number of audio buffer underruns detected.
    pub underrun_count: u64,
    /// Number of audio buffer overruns detected.
    pub overrun_count: u64,
    /// Rolling audio buffer fill level (0.0 to 1.0).
    pub buffer_fill: MetricWindow,
    /// Peak audio level in dBFS over the window.
    pub peak_dbfs: MetricWindow,
}

impl AudioStreamStats {
    /// Creates a new audio stream statistics tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_frames: 0,
            total_samples: 0,
            underrun_count: 0,
            overrun_count: 0,
            buffer_fill: MetricWindow::new(300),
            peak_dbfs: MetricWindow::new(300),
        }
    }

    /// Records an audio frame.
    pub fn record_frame(&mut self, sample_count: u64, peak_level_dbfs: f64) {
        self.total_frames += 1;
        self.total_samples += sample_count;
        self.peak_dbfs.push(peak_level_dbfs);
    }

    /// Records the current audio buffer fill level (0.0 = empty, 1.0 = full).
    pub fn record_buffer_fill(&mut self, fill: f64) {
        self.buffer_fill.push(fill.clamp(0.0, 1.0));
    }

    /// Records a buffer underrun event.
    pub fn record_underrun(&mut self) {
        self.underrun_count += 1;
    }

    /// Records a buffer overrun event.
    pub fn record_overrun(&mut self) {
        self.overrun_count += 1;
    }

    /// Returns the total number of buffer error events.
    #[must_use]
    pub fn total_buffer_errors(&self) -> u64 {
        self.underrun_count + self.overrun_count
    }

    /// Resets all statistics.
    pub fn reset(&mut self) {
        self.total_frames = 0;
        self.total_samples = 0;
        self.underrun_count = 0;
        self.overrun_count = 0;
        self.buffer_fill.clear();
        self.peak_dbfs.clear();
    }
}

impl Default for AudioStreamStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Bandwidth tracking for an NDI connection.
#[derive(Debug, Clone)]
pub struct BandwidthStats {
    /// Total bytes transferred.
    pub total_bytes: u64,
    /// Rolling transfer rate in bytes/sec.
    pub rate_bps: MetricWindow,
    /// Timestamp of the first recorded byte.
    start_time: Option<Instant>,
    /// Timestamp of the last rate calculation.
    last_calc_time: Option<Instant>,
    /// Bytes accumulated since last rate calculation.
    bytes_since_last: u64,
}

impl BandwidthStats {
    /// Creates a new bandwidth statistics tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_bytes: 0,
            rate_bps: MetricWindow::new(60),
            start_time: None,
            last_calc_time: None,
            bytes_since_last: 0,
        }
    }

    /// Records a chunk of transferred bytes.
    #[allow(clippy::cast_precision_loss)]
    pub fn record_transfer(&mut self, bytes: u64, now: Instant) {
        if self.start_time.is_none() {
            self.start_time = Some(now);
            self.last_calc_time = Some(now);
        }
        self.total_bytes += bytes;
        self.bytes_since_last += bytes;

        if let Some(last) = self.last_calc_time {
            let elapsed = now.duration_since(last).as_secs_f64();
            if elapsed >= 0.5 {
                let rate = self.bytes_since_last as f64 / elapsed;
                self.rate_bps.push(rate);
                self.bytes_since_last = 0;
                self.last_calc_time = Some(now);
            }
        }
    }

    /// Returns the average transfer rate in bytes/sec, or `None` if unknown.
    #[must_use]
    pub fn average_rate(&self) -> Option<f64> {
        self.rate_bps.mean()
    }

    /// Returns the total uptime since the first transfer.
    #[must_use]
    pub fn uptime(&self) -> Option<Duration> {
        self.start_time.map(|s| s.elapsed())
    }

    /// Resets all statistics.
    pub fn reset(&mut self) {
        self.total_bytes = 0;
        self.rate_bps.clear();
        self.start_time = None;
        self.last_calc_time = None;
        self.bytes_since_last = 0;
    }
}

impl Default for BandwidthStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_window_empty() {
        let w = MetricWindow::new(10);
        assert_eq!(w.count(), 0);
        assert!(w.mean().is_none());
        assert!(w.min().is_none());
        assert!(w.max().is_none());
        assert!(w.last().is_none());
        assert!(w.std_dev().is_none());
    }

    #[test]
    fn test_metric_window_push_and_stats() {
        let mut w = MetricWindow::new(5);
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            w.push(v);
        }
        assert_eq!(w.count(), 5);
        assert!((w.mean().expect("expected mean to be available") - 3.0).abs() < 1e-9);
        assert!((w.min().expect("expected min to be available") - 1.0).abs() < 1e-9);
        assert!((w.max().expect("expected max to be available") - 5.0).abs() < 1e-9);
        assert!((w.last().expect("expected last to be available") - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_metric_window_capacity_eviction() {
        let mut w = MetricWindow::new(3);
        w.push(10.0);
        w.push(20.0);
        w.push(30.0);
        w.push(40.0); // should evict 10.0
        assert_eq!(w.count(), 3);
        assert!((w.min().expect("expected min to be available") - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_metric_window_std_dev() {
        let mut w = MetricWindow::new(4);
        w.push(2.0);
        w.push(4.0);
        w.push(4.0);
        w.push(4.0);
        // mean = 3.5, variance = [(2-3.5)^2 + 3*(4-3.5)^2] / 3 = [2.25+0.75]/3 = 1.0
        let sd = w.std_dev().expect("expected std_dev to be available");
        assert!((sd - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_metric_window_clear() {
        let mut w = MetricWindow::new(10);
        w.push(1.0);
        w.push(2.0);
        w.clear();
        assert_eq!(w.count(), 0);
    }

    #[test]
    fn test_video_stats_new() {
        let stats = VideoStreamStats::new();
        assert_eq!(stats.total_frames, 0);
        assert_eq!(stats.dropped_frames, 0);
        assert!((stats.measured_fps - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_video_stats_record_frame() {
        let mut stats = VideoStreamStats::new();
        let now = Instant::now();
        stats.record_frame(100_000, now);
        assert_eq!(stats.total_frames, 1);
        assert!(stats.last_frame_time.is_some());
    }

    #[test]
    fn test_video_stats_drop_rate() {
        let mut stats = VideoStreamStats::new();
        let now = Instant::now();
        for i in 0..10 {
            stats.record_frame(1000, now + Duration::from_millis(i * 33));
        }
        stats.record_dropped(2);
        // total_frames=10, dropped=2, rate = 2/12
        let rate = stats.drop_rate();
        assert!((rate - 2.0 / 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_video_stats_drop_rate_empty() {
        let stats = VideoStreamStats::new();
        assert!((stats.drop_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_video_stats_reset() {
        let mut stats = VideoStreamStats::new();
        let now = Instant::now();
        stats.record_frame(5000, now);
        stats.record_dropped(1);
        stats.reset();
        assert_eq!(stats.total_frames, 0);
        assert_eq!(stats.dropped_frames, 0);
    }

    #[test]
    fn test_audio_stats_new() {
        let stats = AudioStreamStats::new();
        assert_eq!(stats.total_frames, 0);
        assert_eq!(stats.total_samples, 0);
        assert_eq!(stats.underrun_count, 0);
    }

    #[test]
    fn test_audio_stats_record() {
        let mut stats = AudioStreamStats::new();
        stats.record_frame(1024, -12.0);
        stats.record_frame(1024, -6.0);
        assert_eq!(stats.total_frames, 2);
        assert_eq!(stats.total_samples, 2048);
        assert!(
            (stats.peak_dbfs.max().expect("expected max to be available") - (-6.0)).abs() < 1e-9
        );
    }

    #[test]
    fn test_audio_stats_buffer_errors() {
        let mut stats = AudioStreamStats::new();
        stats.record_underrun();
        stats.record_underrun();
        stats.record_overrun();
        assert_eq!(stats.total_buffer_errors(), 3);
    }

    #[test]
    fn test_audio_stats_buffer_fill_clamped() {
        let mut stats = AudioStreamStats::new();
        stats.record_buffer_fill(1.5); // should clamp to 1.0
        stats.record_buffer_fill(-0.5); // should clamp to 0.0
        assert!(
            (stats
                .buffer_fill
                .max()
                .expect("expected max to be available")
                - 1.0)
                .abs()
                < 1e-9
        );
        assert!(
            (stats
                .buffer_fill
                .min()
                .expect("expected min to be available")
                - 0.0)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn test_bandwidth_stats_new() {
        let stats = BandwidthStats::new();
        assert_eq!(stats.total_bytes, 0);
        assert!(stats.average_rate().is_none());
        assert!(stats.uptime().is_none());
    }

    #[test]
    fn test_bandwidth_stats_transfer() {
        let mut stats = BandwidthStats::new();
        let now = Instant::now();
        stats.record_transfer(1000, now);
        assert_eq!(stats.total_bytes, 1000);
        stats.record_transfer(2000, now + Duration::from_secs(1));
        assert_eq!(stats.total_bytes, 3000);
    }

    #[test]
    fn test_bandwidth_stats_reset() {
        let mut stats = BandwidthStats::new();
        let now = Instant::now();
        stats.record_transfer(5000, now);
        stats.reset();
        assert_eq!(stats.total_bytes, 0);
        assert!(stats.uptime().is_none());
    }
}
