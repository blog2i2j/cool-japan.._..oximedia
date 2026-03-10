#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]

//! Stream recording and playback for diagnostics.
//!
//! This module captures incoming video-over-IP stream data (packets, frames,
//! metadata) into an in-memory ring buffer for later analysis or diagnostic
//! replay. It is useful for debugging stream issues, auditing received data,
//! and performing post-hoc analysis of network conditions.

use std::collections::VecDeque;

/// Default maximum number of recorded entries.
const DEFAULT_MAX_ENTRIES: usize = 10_000;

/// Type of recorded event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordEventType {
    /// A video frame was received.
    VideoFrame,
    /// An audio frame was received.
    AudioFrame,
    /// A metadata packet was received.
    Metadata,
    /// A control message was received.
    Control,
    /// A packet loss was detected.
    PacketLoss,
    /// A jitter spike was detected.
    JitterSpike,
    /// A format change was detected.
    FormatChange,
}

impl RecordEventType {
    /// Returns a human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::VideoFrame => "video_frame",
            Self::AudioFrame => "audio_frame",
            Self::Metadata => "metadata",
            Self::Control => "control",
            Self::PacketLoss => "packet_loss",
            Self::JitterSpike => "jitter_spike",
            Self::FormatChange => "format_change",
        }
    }
}

/// A single recorded event.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordEntry {
    /// Monotonic timestamp in microseconds.
    pub timestamp_us: u64,
    /// Type of event.
    pub event_type: RecordEventType,
    /// Sequence number (if applicable).
    pub sequence: u64,
    /// Size in bytes (if applicable).
    pub size_bytes: u32,
    /// Optional description or details.
    pub detail: String,
}

impl RecordEntry {
    /// Create a new record entry.
    pub fn new(
        timestamp_us: u64,
        event_type: RecordEventType,
        sequence: u64,
        size_bytes: u32,
        detail: String,
    ) -> Self {
        Self {
            timestamp_us,
            event_type,
            sequence,
            size_bytes,
            detail,
        }
    }

    /// Create a simple video frame entry.
    pub fn video_frame(timestamp_us: u64, sequence: u64, size_bytes: u32) -> Self {
        Self::new(
            timestamp_us,
            RecordEventType::VideoFrame,
            sequence,
            size_bytes,
            String::new(),
        )
    }

    /// Create a simple audio frame entry.
    pub fn audio_frame(timestamp_us: u64, sequence: u64, size_bytes: u32) -> Self {
        Self::new(
            timestamp_us,
            RecordEventType::AudioFrame,
            sequence,
            size_bytes,
            String::new(),
        )
    }

    /// Create a packet loss event.
    pub fn packet_loss(timestamp_us: u64, sequence: u64, detail: &str) -> Self {
        Self::new(
            timestamp_us,
            RecordEventType::PacketLoss,
            sequence,
            0,
            detail.to_string(),
        )
    }
}

/// Recording statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordingStats {
    /// Total entries recorded (including evicted ones).
    pub total_recorded: u64,
    /// Current entries in the buffer.
    pub current_entries: usize,
    /// Maximum buffer size.
    pub max_entries: usize,
    /// Count of each event type.
    pub video_frame_count: u64,
    /// Count of audio frames recorded.
    pub audio_frame_count: u64,
    /// Count of packet loss events.
    pub packet_loss_count: u64,
    /// Total bytes recorded (sum of `size_bytes`).
    pub total_bytes: u64,
    /// Duration covered in microseconds (last - first timestamp).
    pub duration_us: u64,
}

/// Stream recorder with a ring buffer.
#[derive(Debug)]
pub struct StreamRecorder {
    /// Ring buffer of recorded events.
    entries: VecDeque<RecordEntry>,
    /// Maximum number of entries to keep.
    max_entries: usize,
    /// Whether recording is active.
    active: bool,
    /// Total entries ever recorded (including evicted).
    total_recorded: u64,
    /// Running counters.
    video_count: u64,
    /// Running audio frame count.
    audio_count: u64,
    /// Running packet loss count.
    loss_count: u64,
    /// Running total bytes.
    total_bytes: u64,
}

impl StreamRecorder {
    /// Create a new stream recorder with the given maximum entry count.
    pub fn new(max_entries: usize) -> Self {
        let max = if max_entries == 0 {
            DEFAULT_MAX_ENTRIES
        } else {
            max_entries
        };
        Self {
            entries: VecDeque::with_capacity(max.min(10_000)),
            max_entries: max,
            active: true,
            total_recorded: 0,
            video_count: 0,
            audio_count: 0,
            loss_count: 0,
            total_bytes: 0,
        }
    }

    /// Create with default capacity.
    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_MAX_ENTRIES)
    }

    /// Record an event.
    pub fn record(&mut self, entry: RecordEntry) {
        if !self.active {
            return;
        }

        // Update counters.
        self.total_recorded += 1;
        self.total_bytes += u64::from(entry.size_bytes);
        match entry.event_type {
            RecordEventType::VideoFrame => self.video_count += 1,
            RecordEventType::AudioFrame => self.audio_count += 1,
            RecordEventType::PacketLoss => self.loss_count += 1,
            _ => {}
        }

        // Evict oldest if at capacity.
        if self.entries.len() >= self.max_entries {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    /// Pause recording.
    pub fn pause(&mut self) {
        self.active = false;
    }

    /// Resume recording.
    pub fn resume(&mut self) {
        self.active = true;
    }

    /// Check if recording is active.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Get all current entries.
    pub fn entries(&self) -> &VecDeque<RecordEntry> {
        &self.entries
    }

    /// Get entries filtered by event type.
    pub fn entries_by_type(&self, event_type: RecordEventType) -> Vec<&RecordEntry> {
        self.entries
            .iter()
            .filter(|e| e.event_type == event_type)
            .collect()
    }

    /// Get entries in a time range (inclusive).
    pub fn entries_in_range(&self, start_us: u64, end_us: u64) -> Vec<&RecordEntry> {
        self.entries
            .iter()
            .filter(|e| e.timestamp_us >= start_us && e.timestamp_us <= end_us)
            .collect()
    }

    /// Compute recording statistics.
    pub fn stats(&self) -> RecordingStats {
        let duration = if self.entries.len() >= 2 {
            self.entries.back().map_or(0, |b| b.timestamp_us)
                - self.entries.front().map_or(0, |f| f.timestamp_us)
        } else {
            0
        };

        RecordingStats {
            total_recorded: self.total_recorded,
            current_entries: self.entries.len(),
            max_entries: self.max_entries,
            video_frame_count: self.video_count,
            audio_frame_count: self.audio_count,
            packet_loss_count: self.loss_count,
            total_bytes: self.total_bytes,
            duration_us: duration,
        }
    }

    /// Compute the average frame rate from video frame entries.
    pub fn estimated_frame_rate(&self) -> f64 {
        let video_entries: Vec<&RecordEntry> = self.entries_by_type(RecordEventType::VideoFrame);
        if video_entries.len() < 2 {
            return 0.0;
        }
        let first_ts = video_entries[0].timestamp_us;
        let last_ts = video_entries[video_entries.len() - 1].timestamp_us;
        let duration_s = (last_ts - first_ts) as f64 / 1_000_000.0;
        if duration_s <= 0.0 {
            return 0.0;
        }
        (video_entries.len() - 1) as f64 / duration_s
    }

    /// Clear all entries but keep counters.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Full reset — clear entries and counters.
    pub fn reset(&mut self) {
        self.entries.clear();
        self.total_recorded = 0;
        self.video_count = 0;
        self.audio_count = 0;
        self.loss_count = 0;
        self.total_bytes = 0;
    }

    /// Return the current number of entries in the buffer.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for StreamRecorder {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_type_label() {
        assert_eq!(RecordEventType::VideoFrame.label(), "video_frame");
        assert_eq!(RecordEventType::PacketLoss.label(), "packet_loss");
        assert_eq!(RecordEventType::FormatChange.label(), "format_change");
    }

    #[test]
    fn test_entry_creation() {
        let e = RecordEntry::video_frame(1000, 1, 4096);
        assert_eq!(e.timestamp_us, 1000);
        assert_eq!(e.event_type, RecordEventType::VideoFrame);
        assert_eq!(e.sequence, 1);
        assert_eq!(e.size_bytes, 4096);
    }

    #[test]
    fn test_empty_recorder() {
        let r = StreamRecorder::with_defaults();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
        assert!(r.is_active());
    }

    #[test]
    fn test_record_single_entry() {
        let mut r = StreamRecorder::with_defaults();
        r.record(RecordEntry::video_frame(1000, 0, 1024));
        assert_eq!(r.len(), 1);
        let stats = r.stats();
        assert_eq!(stats.video_frame_count, 1);
        assert_eq!(stats.total_bytes, 1024);
    }

    #[test]
    fn test_ring_buffer_eviction() {
        let mut r = StreamRecorder::new(5);
        for i in 0..10 {
            r.record(RecordEntry::video_frame(i * 1000, i, 100));
        }
        assert_eq!(r.len(), 5);
        // Oldest should be evicted: first entry should be seq 5
        assert_eq!(
            r.entries()
                .front()
                .expect("should succeed in test")
                .sequence,
            5
        );
    }

    #[test]
    fn test_pause_resume() {
        let mut r = StreamRecorder::with_defaults();
        r.pause();
        assert!(!r.is_active());
        r.record(RecordEntry::video_frame(1000, 0, 100));
        assert!(r.is_empty()); // Should not have recorded
        r.resume();
        r.record(RecordEntry::video_frame(2000, 1, 200));
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn test_entries_by_type() {
        let mut r = StreamRecorder::with_defaults();
        r.record(RecordEntry::video_frame(1000, 0, 100));
        r.record(RecordEntry::audio_frame(1100, 1, 50));
        r.record(RecordEntry::video_frame(2000, 2, 100));
        let videos = r.entries_by_type(RecordEventType::VideoFrame);
        assert_eq!(videos.len(), 2);
        let audios = r.entries_by_type(RecordEventType::AudioFrame);
        assert_eq!(audios.len(), 1);
    }

    #[test]
    fn test_entries_in_range() {
        let mut r = StreamRecorder::with_defaults();
        for i in 0..10 {
            r.record(RecordEntry::video_frame(i * 1000, i, 100));
        }
        let range = r.entries_in_range(3000, 6000);
        assert_eq!(range.len(), 4); // timestamps 3000,4000,5000,6000
    }

    #[test]
    fn test_estimated_frame_rate() {
        let mut r = StreamRecorder::with_defaults();
        // 30 frames over 1 second (33333us apart)
        for i in 0..30 {
            r.record(RecordEntry::video_frame(i * 33333, i, 100));
        }
        let fps = r.estimated_frame_rate();
        assert!(fps > 28.0 && fps < 32.0, "Expected ~30fps, got {fps}");
    }

    #[test]
    fn test_stats_duration() {
        let mut r = StreamRecorder::with_defaults();
        r.record(RecordEntry::video_frame(1000, 0, 100));
        r.record(RecordEntry::video_frame(5000, 1, 100));
        let stats = r.stats();
        assert_eq!(stats.duration_us, 4000);
    }

    #[test]
    fn test_packet_loss_counter() {
        let mut r = StreamRecorder::with_defaults();
        r.record(RecordEntry::packet_loss(1000, 5, "gap detected"));
        r.record(RecordEntry::packet_loss(2000, 10, "gap detected"));
        let stats = r.stats();
        assert_eq!(stats.packet_loss_count, 2);
    }

    #[test]
    fn test_clear_vs_reset() {
        let mut r = StreamRecorder::with_defaults();
        r.record(RecordEntry::video_frame(1000, 0, 100));
        r.clear();
        assert!(r.is_empty());
        // Counters should still be present
        let stats = r.stats();
        assert_eq!(stats.total_recorded, 1);

        r.reset();
        let stats2 = r.stats();
        assert_eq!(stats2.total_recorded, 0);
    }

    #[test]
    fn test_estimated_fps_insufficient_data() {
        let r = StreamRecorder::with_defaults();
        assert!((r.estimated_frame_rate()).abs() < f64::EPSILON);
    }
}
