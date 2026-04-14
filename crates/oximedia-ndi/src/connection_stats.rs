//! NDI connection statistics — per-source bitrate, dropped frames, jitter, and quality score.
//!
//! Provides detailed per-source link quality metrics that go beyond the simple counters in
//! [`crate::statistics`].  Key additions:
//!
//! * Rolling-window bitrate estimation (video and audio separately).
//! * Jitter estimation using the RFC 3550 (RTP) inter-arrival variance algorithm.
//! * A composite 0-100 connection quality score derived from drop rate, jitter, and reorder ratio.
//! * Per-source aggregation in [`ConnectionStatsRegistry`].

#![allow(dead_code)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::time::Duration;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Number of inter-arrival samples kept for jitter estimation.
const JITTER_WINDOW: usize = 32;

/// Rolling window for bitrate estimation (samples).
const BITRATE_WINDOW: usize = 16;

// ─────────────────────────────────────────────────────────────────────────────
// BitrateEstimator
// ─────────────────────────────────────────────────────────────────────────────

/// A rolling-window bitrate estimator that tracks bytes-per-interval samples.
#[derive(Debug, Clone)]
pub struct BitrateEstimator {
    /// Circular buffer of (timestamp_ms, bytes) samples.
    samples: Vec<(u64, u64)>,
    /// Write head position.
    head: usize,
    /// How many valid entries are in the buffer.
    count: usize,
    /// Maximum samples stored.
    capacity: usize,
}

impl BitrateEstimator {
    /// Create a new estimator with the given rolling-window size.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(2);
        Self {
            samples: vec![(0, 0); capacity],
            head: 0,
            count: 0,
            capacity,
        }
    }

    /// Record a received chunk of `bytes` at `timestamp_ms`.
    pub fn record(&mut self, timestamp_ms: u64, bytes: u64) {
        self.samples[self.head] = (timestamp_ms, bytes);
        self.head = (self.head + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    /// Estimate average bitrate in kbps across the current window.
    ///
    /// Returns `0.0` when fewer than 2 samples are available.
    pub fn avg_kbps(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }

        // Find the oldest and newest valid samples.
        let total_bytes: u64 = self.valid_samples().map(|(_, b)| b).sum();
        let timestamps: Vec<u64> = self.valid_samples().map(|(t, _)| t).collect();

        let t_min = *timestamps.iter().min().unwrap_or(&0);
        let t_max = *timestamps.iter().max().unwrap_or(&0);

        let elapsed_ms = t_max.saturating_sub(t_min);
        if elapsed_ms == 0 {
            return 0.0;
        }

        // bits / ms = kbps
        (total_bytes as f64 * 8.0) / elapsed_ms as f64
    }

    /// Peak bitrate observed within a single inter-sample interval (kbps).
    pub fn peak_kbps(&self) -> f64 {
        let mut peak = 0.0_f64;
        let samples: Vec<(u64, u64)> = self.valid_samples().collect();
        if samples.len() < 2 {
            return peak;
        }
        for window in samples.windows(2) {
            let dt = window[1].0.saturating_sub(window[0].0);
            if dt == 0 {
                continue;
            }
            let kbps = (window[1].1 as f64 * 8.0) / dt as f64;
            if kbps > peak {
                peak = kbps;
            }
        }
        peak
    }

    /// Iterator over valid (timestamp_ms, bytes) samples in insertion order.
    fn valid_samples(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        let start = if self.count == self.capacity {
            self.head
        } else {
            0
        };
        (0..self.count).map(move |i| self.samples[(start + i) % self.capacity])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// JitterEstimator
// ─────────────────────────────────────────────────────────────────────────────

/// RFC 3550 §A.8 inter-arrival jitter estimator.
///
/// Tracks the running mean absolute deviation of packet inter-arrival gaps,
/// expressed in milliseconds.
#[derive(Debug, Clone)]
pub struct JitterEstimator {
    /// RFC 3550 jitter accumulator (scaled × 16).
    jitter_scaled: f64,
    /// Timestamp of the previous packet arrival (ms).
    prev_arrival_ms: Option<u64>,
    /// Timestamp carried in the previous packet (ms).
    prev_send_ms: Option<u64>,
    /// Total packets processed.
    packet_count: u64,
}

impl JitterEstimator {
    /// Create a zeroed estimator.
    pub fn new() -> Self {
        Self {
            jitter_scaled: 0.0,
            prev_arrival_ms: None,
            prev_send_ms: None,
            packet_count: 0,
        }
    }

    /// Feed a new packet arrival.
    ///
    /// * `send_ms` — sender-side timestamp embedded in the NDI frame header.
    /// * `arrival_ms` — local wall-clock arrival time.
    pub fn record_packet(&mut self, send_ms: u64, arrival_ms: u64) {
        self.packet_count += 1;

        if let (Some(prev_arrival), Some(prev_send)) =
            (self.prev_arrival_ms, self.prev_send_ms)
        {
            // d = (arrival_i − arrival_{i−1}) − (send_i − send_{i−1})
            let recv_diff = arrival_ms as i64 - prev_arrival as i64;
            let send_diff = send_ms as i64 - prev_send as i64;
            let d = (recv_diff - send_diff).unsigned_abs() as f64;

            // J_n = J_{n-1} + (|d| − J_{n-1}) / 16
            self.jitter_scaled += (d - self.jitter_scaled) / 16.0;
        }

        self.prev_arrival_ms = Some(arrival_ms);
        self.prev_send_ms = Some(send_ms);
    }

    /// Current jitter estimate in milliseconds.
    pub fn jitter_ms(&self) -> f64 {
        self.jitter_scaled
    }

    /// Total packets processed.
    pub fn packet_count(&self) -> u64 {
        self.packet_count
    }

    /// Reset the estimator to its initial state.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for JitterEstimator {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ConnectionQuality
// ─────────────────────────────────────────────────────────────────────────────

/// Composite connection quality descriptor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConnectionQuality {
    /// 0–100 score (100 = perfect, 0 = unusable).
    pub score: f64,
    /// Drop rate [0.0, 1.0].
    pub drop_rate: f64,
    /// Estimated jitter in milliseconds.
    pub jitter_ms: f64,
    /// Reordering ratio [0.0, 1.0].
    pub reorder_rate: f64,
}

impl ConnectionQuality {
    /// Compute quality from constituent metrics.
    ///
    /// The score is a weighted linear combination:
    /// * Drop penalty: −50 × drop_rate
    /// * Jitter penalty: −30 × clamp(jitter_ms / 100, 0, 1)
    /// * Reorder penalty: −20 × reorder_rate
    pub fn compute(drop_rate: f64, jitter_ms: f64, reorder_rate: f64) -> Self {
        let drop_pen = 50.0 * drop_rate.clamp(0.0, 1.0);
        let jitter_pen = 30.0 * (jitter_ms / 100.0).clamp(0.0, 1.0);
        let reorder_pen = 20.0 * reorder_rate.clamp(0.0, 1.0);

        let score = (100.0 - drop_pen - jitter_pen - reorder_pen).max(0.0);

        Self {
            score,
            drop_rate,
            jitter_ms,
            reorder_rate,
        }
    }

    /// Returns `true` if the quality is good enough for broadcast use (score ≥ 80).
    pub fn is_broadcast_quality(&self) -> bool {
        self.score >= 80.0
    }

    /// Returns `true` if the connection is marginal (score between 50 and 80).
    pub fn is_marginal(&self) -> bool {
        self.score >= 50.0 && self.score < 80.0
    }

    /// Returns `true` if the connection is degraded (score < 50).
    pub fn is_degraded(&self) -> bool {
        self.score < 50.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SourceConnectionStats
// ─────────────────────────────────────────────────────────────────────────────

/// Complete per-source statistics record.
#[derive(Debug, Clone)]
pub struct SourceConnectionStats {
    /// Name of the NDI source.
    pub source_name: String,
    /// Total video frames received.
    pub video_frames: u64,
    /// Total audio frames received.
    pub audio_frames: u64,
    /// Total metadata frames received.
    pub metadata_frames: u64,
    /// Video frames that were dropped.
    pub video_dropped: u64,
    /// Audio frames that were dropped.
    pub audio_dropped: u64,
    /// Frames received out of sequence order.
    pub reordered: u64,
    /// Total bytes received across all frame types.
    pub total_bytes: u64,
    /// Video bitrate estimator.
    pub video_bitrate: BitrateEstimator,
    /// Audio bitrate estimator.
    pub audio_bitrate: BitrateEstimator,
    /// Jitter estimator for video frames.
    pub jitter: JitterEstimator,
    /// Time of first frame received (ms since epoch).
    pub first_frame_ms: Option<u64>,
    /// Time of most recent frame received (ms since epoch).
    pub last_frame_ms: Option<u64>,
}

impl SourceConnectionStats {
    /// Create a new stats record for `source_name`.
    pub fn new(source_name: impl Into<String>) -> Self {
        Self {
            source_name: source_name.into(),
            video_frames: 0,
            audio_frames: 0,
            metadata_frames: 0,
            video_dropped: 0,
            audio_dropped: 0,
            reordered: 0,
            total_bytes: 0,
            video_bitrate: BitrateEstimator::new(BITRATE_WINDOW),
            audio_bitrate: BitrateEstimator::new(BITRATE_WINDOW),
            jitter: JitterEstimator::new(),
            first_frame_ms: None,
            last_frame_ms: None,
        }
    }

    /// Record a received video frame.
    ///
    /// * `bytes` — encoded byte size of the frame.
    /// * `timestamp_ms` — sender-embedded timestamp.
    /// * `arrival_ms` — local wall-clock time of arrival.
    pub fn record_video_frame(&mut self, bytes: u64, timestamp_ms: u64, arrival_ms: u64) {
        self.video_frames += 1;
        self.total_bytes += bytes;
        self.video_bitrate.record(arrival_ms, bytes);
        self.jitter.record_packet(timestamp_ms, arrival_ms);
        self.update_timestamps(arrival_ms);
    }

    /// Record a received audio frame.
    pub fn record_audio_frame(&mut self, bytes: u64, arrival_ms: u64) {
        self.audio_frames += 1;
        self.total_bytes += bytes;
        self.audio_bitrate.record(arrival_ms, bytes);
        self.update_timestamps(arrival_ms);
    }

    /// Record a received metadata frame.
    pub fn record_metadata_frame(&mut self, arrival_ms: u64) {
        self.metadata_frames += 1;
        self.update_timestamps(arrival_ms);
    }

    /// Record a dropped video frame.
    pub fn record_video_drop(&mut self) {
        self.video_dropped += 1;
    }

    /// Record a dropped audio frame.
    pub fn record_audio_drop(&mut self) {
        self.audio_dropped += 1;
    }

    /// Record an out-of-order (reordered) frame.
    pub fn record_reorder(&mut self) {
        self.reordered += 1;
    }

    /// Compute the current connection quality.
    pub fn quality(&self) -> ConnectionQuality {
        let total_video = self.video_frames + self.video_dropped;
        let drop_rate = if total_video > 0 {
            self.video_dropped as f64 / total_video as f64
        } else {
            0.0
        };

        let total_frames = self.video_frames + self.audio_frames;
        let reorder_rate = if total_frames > 0 {
            self.reordered as f64 / total_frames as f64
        } else {
            0.0
        };

        ConnectionQuality::compute(drop_rate, self.jitter.jitter_ms(), reorder_rate)
    }

    /// Active duration derived from first/last frame timestamps.
    ///
    /// Returns `None` if fewer than 2 frames have been received.
    pub fn active_duration(&self) -> Option<Duration> {
        let first = self.first_frame_ms?;
        let last = self.last_frame_ms?;
        if last >= first {
            Some(Duration::from_millis(last - first))
        } else {
            None
        }
    }

    fn update_timestamps(&mut self, arrival_ms: u64) {
        if self.first_frame_ms.is_none() {
            self.first_frame_ms = Some(arrival_ms);
        }
        self.last_frame_ms = Some(arrival_ms);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ConnectionStatsRegistry
// ─────────────────────────────────────────────────────────────────────────────

/// A registry that maps source names to their [`SourceConnectionStats`].
#[derive(Debug, Default)]
pub struct ConnectionStatsRegistry {
    entries: HashMap<String, SourceConnectionStats>,
}

impl ConnectionStatsRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create a stats record for `source_name`.
    pub fn entry(&mut self, source_name: &str) -> &mut SourceConnectionStats {
        self.entries
            .entry(source_name.to_string())
            .or_insert_with(|| SourceConnectionStats::new(source_name))
    }

    /// Immutable reference to stats for `source_name`, or `None`.
    pub fn get(&self, source_name: &str) -> Option<&SourceConnectionStats> {
        self.entries.get(source_name)
    }

    /// Remove statistics for a source that has gone offline.
    pub fn remove(&mut self, source_name: &str) -> Option<SourceConnectionStats> {
        self.entries.remove(source_name)
    }

    /// Number of sources tracked.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no sources are tracked.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterator over all (source_name, stats) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &SourceConnectionStats)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Return all sources whose quality score falls below `threshold`.
    pub fn degraded_sources(&self, threshold: f64) -> Vec<&str> {
        self.entries
            .iter()
            .filter(|(_, s)| s.quality().score < threshold)
            .map(|(k, _)| k.as_str())
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── BitrateEstimator ──────────────────────────────────────────────────────

    #[test]
    fn test_bitrate_estimator_empty() {
        let e = BitrateEstimator::new(8);
        assert_eq!(e.avg_kbps(), 0.0);
        assert_eq!(e.peak_kbps(), 0.0);
    }

    #[test]
    fn test_bitrate_estimator_single_sample() {
        let mut e = BitrateEstimator::new(8);
        e.record(0, 1000);
        // Need at least 2 samples to estimate rate
        assert_eq!(e.avg_kbps(), 0.0);
    }

    #[test]
    fn test_bitrate_estimator_two_samples() {
        let mut e = BitrateEstimator::new(8);
        // 8000 bytes over 1000 ms = 64 kbps
        e.record(0, 8000);
        e.record(1000, 8000);
        // avg includes both samples; total 16000 bytes over 1000 ms span
        let kbps = e.avg_kbps();
        assert!(kbps > 0.0, "expected positive kbps, got {kbps}");
    }

    #[test]
    fn test_bitrate_estimator_window_eviction() {
        let mut e = BitrateEstimator::new(4);
        for i in 0..6u64 {
            e.record(i * 100, 1000);
        }
        // Should still give a valid (non-panicking) result
        let _ = e.avg_kbps();
    }

    // ── JitterEstimator ───────────────────────────────────────────────────────

    #[test]
    fn test_jitter_estimator_initially_zero() {
        let j = JitterEstimator::new();
        assert_eq!(j.jitter_ms(), 0.0);
        assert_eq!(j.packet_count(), 0);
    }

    #[test]
    fn test_jitter_estimator_perfect_packets() {
        let mut j = JitterEstimator::new();
        // Perfectly uniform: send every 33 ms, arrive every 33 ms
        for i in 0..20u64 {
            j.record_packet(i * 33, i * 33 + 5); // constant 5 ms one-way delay
        }
        // Jitter should be near zero
        assert!(
            j.jitter_ms() < 1.0,
            "expected near-zero jitter, got {}",
            j.jitter_ms()
        );
    }

    #[test]
    fn test_jitter_estimator_variable_delay() {
        let mut j = JitterEstimator::new();
        // Alternating 10 ms and 30 ms arrival delay
        let mut arrival = 0u64;
        for i in 0..20u64 {
            let delay = if i % 2 == 0 { 10 } else { 30 };
            arrival += 33 + delay;
            j.record_packet(i * 33, arrival);
        }
        assert!(
            j.jitter_ms() > 0.0,
            "expected non-zero jitter for variable delay"
        );
    }

    #[test]
    fn test_jitter_reset() {
        let mut j = JitterEstimator::new();
        j.record_packet(0, 5);
        j.record_packet(33, 50);
        j.reset();
        assert_eq!(j.packet_count(), 0);
        assert_eq!(j.jitter_ms(), 0.0);
    }

    // ── ConnectionQuality ─────────────────────────────────────────────────────

    #[test]
    fn test_quality_perfect() {
        let q = ConnectionQuality::compute(0.0, 0.0, 0.0);
        assert!((q.score - 100.0).abs() < 1e-9);
        assert!(q.is_broadcast_quality());
        assert!(!q.is_marginal());
        assert!(!q.is_degraded());
    }

    #[test]
    fn test_quality_total_drop() {
        let q = ConnectionQuality::compute(1.0, 0.0, 0.0);
        assert!((q.score - 50.0).abs() < 1e-9);
        assert!(q.is_marginal());
    }

    #[test]
    fn test_quality_high_jitter() {
        let q = ConnectionQuality::compute(0.0, 100.0, 0.0);
        // Full jitter penalty = 30
        assert!((q.score - 70.0).abs() < 1e-9);
        assert!(q.is_marginal());
    }

    #[test]
    fn test_quality_floor_zero() {
        let q = ConnectionQuality::compute(1.0, 200.0, 1.0);
        assert_eq!(q.score, 0.0);
        assert!(q.is_degraded());
    }

    // ── SourceConnectionStats ─────────────────────────────────────────────────

    #[test]
    fn test_source_stats_record_video() {
        let mut s = SourceConnectionStats::new("CAM1");
        s.record_video_frame(1500, 0, 5);
        assert_eq!(s.video_frames, 1);
        assert_eq!(s.total_bytes, 1500);
    }

    #[test]
    fn test_source_stats_active_duration() {
        let mut s = SourceConnectionStats::new("CAM1");
        s.record_video_frame(1500, 0, 0);
        s.record_video_frame(1500, 33, 33);
        s.record_video_frame(1500, 66, 1000);
        let dur = s.active_duration().expect("duration");
        assert_eq!(dur, Duration::from_millis(1000));
    }

    #[test]
    fn test_source_stats_quality_on_drops() {
        let mut s = SourceConnectionStats::new("CAM2");
        s.record_video_frame(1500, 0, 0);
        s.record_video_drop();
        // 1 received, 1 dropped → 50 % drop rate
        let q = s.quality();
        assert!(q.drop_rate > 0.0);
        assert!(q.score < 100.0);
    }

    // ── ConnectionStatsRegistry ───────────────────────────────────────────────

    #[test]
    fn test_registry_entry_created_on_demand() {
        let mut reg = ConnectionStatsRegistry::new();
        {
            let s = reg.entry("CAM1");
            s.record_video_frame(1000, 0, 0);
        }
        assert_eq!(reg.len(), 1);
        assert!(reg.get("CAM1").is_some());
    }

    #[test]
    fn test_registry_degraded_sources() {
        let mut reg = ConnectionStatsRegistry::new();
        {
            let s = reg.entry("BadSource");
            for _ in 0..10 {
                s.record_video_drop();
            }
        }
        let degraded = reg.degraded_sources(80.0);
        assert!(degraded.contains(&"BadSource"));
    }

    #[test]
    fn test_registry_remove() {
        let mut reg = ConnectionStatsRegistry::new();
        let _ = reg.entry("X");
        reg.remove("X");
        assert!(reg.is_empty());
    }
}
