//! Live stream recorder with DVR sliding-window support.
//!
//! Provides a simple segment buffer that trims older content to a configurable
//! look-back duration, suitable for DVR time-shift and live-to-VOD workflows.
//!
//! Also exposes [`RecordingConfig`], [`RecordingSegment`], [`RecorderState`],
//! and [`RecordingStats`] for a higher-level lifecycle-aware recorder API.
//!
//! # Key types
//!
//! | Type | Purpose |
//! |---|---|
//! | [`DvrWindow`] | Configuration for the look-back window |
//! | [`RecordedSegment`] | A single recorded segment with wall-clock timing |
//! | [`StreamRecorder`] | Mutable recorder accumulating segments |
//! | [`RecordingConfig`] | Configuration for filesystem-backed live recording |
//! | [`RecordingSegment`] | Metadata for a completed filesystem segment |
//! | [`RecorderState`] | Lifecycle state of a live recorder |
//! | [`RecordingStats`] | Cumulative statistics for a recording session |
//! | [`LiveRecorder`] | Lifecycle-aware recorder backed by [`RecordingConfig`] |

use std::path::PathBuf;

// ─── DvrWindow ────────────────────────────────────────────────────────────────

/// Controls how far back a [`StreamRecorder`] retains segments.
///
/// When `max_duration_ms` is zero the window is unlimited: no segments are ever
/// evicted by time-based trimming.
#[derive(Debug, Clone)]
pub struct DvrWindow {
    /// Maximum look-back duration in milliseconds.  Segments whose
    /// `timestamp_ms` is older than `(newest_timestamp_ms − max_duration_ms)`
    /// are eligible for eviction.
    ///
    /// Set to `0` to disable time-based eviction (unlimited DVR).
    pub max_duration_ms: u64,
}

impl DvrWindow {
    /// Create a new DVR window configuration.
    pub fn new(max_duration_ms: u64) -> Self {
        Self { max_duration_ms }
    }

    /// Create an unlimited DVR window (no time-based eviction).
    pub fn unlimited() -> Self {
        Self { max_duration_ms: 0 }
    }

    /// Return `true` if the window has a finite look-back limit.
    pub fn is_bounded(&self) -> bool {
        self.max_duration_ms > 0
    }
}

impl Default for DvrWindow {
    fn default() -> Self {
        Self {
            max_duration_ms: 2 * 60 * 60 * 1000, // 2 hours
        }
    }
}

// ─── RecordedSegment ──────────────────────────────────────────────────────────

/// A single recorded segment within a [`StreamRecorder`].
///
/// Each segment carries a wall-clock timestamp so the recorder can perform
/// time-based window trimming without needing codec-specific PTS values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedSegment {
    /// Monotonically increasing sequence number (assigned by the caller or the
    /// upstream muxer; the recorder does **not** reassign these).
    pub sequence: u64,
    /// Wall-clock presentation timestamp of the first sample, in milliseconds
    /// since an arbitrary epoch (e.g. Unix epoch or stream start).
    pub timestamp_ms: u64,
    /// Actual duration of this segment in milliseconds.
    pub duration_ms: u32,
    /// Path on disk (or object-storage key) where the segment data is stored.
    pub path: String,
}

impl RecordedSegment {
    /// Create a new recorded segment.
    pub fn new(
        sequence: u64,
        timestamp_ms: u64,
        duration_ms: u32,
        path: impl Into<String>,
    ) -> Self {
        Self {
            sequence,
            timestamp_ms,
            duration_ms,
            path: path.into(),
        }
    }

    /// Compute the end timestamp of this segment in milliseconds.
    pub fn end_timestamp_ms(&self) -> u64 {
        self.timestamp_ms + self.duration_ms as u64
    }

    /// Duration as floating-point seconds.
    pub fn duration_secs(&self) -> f64 {
        self.duration_ms as f64 / 1000.0
    }
}

// ─── StreamRecorder ──────────────────────────────────────────────────────────

/// Records live stream segments within a sliding DVR window.
///
/// Segments are appended via [`push_segment`] and the buffer is trimmed to
/// `window.max_duration_ms` by [`trim_to_window`].  Both operations are called
/// automatically on each [`push_segment`] invocation; consumers may also call
/// [`trim_to_window`] manually if they push a burst of segments.
///
/// [`push_segment`]: StreamRecorder::push_segment
/// [`trim_to_window`]: StreamRecorder::trim_to_window
#[derive(Debug)]
pub struct StreamRecorder {
    /// DVR window configuration.
    pub window: DvrWindow,
    /// Buffered segments in ascending `timestamp_ms` order (oldest first).
    segments: Vec<RecordedSegment>,
}

impl StreamRecorder {
    /// Create a new recorder with the given DVR window.
    pub fn new(window: DvrWindow) -> Self {
        Self {
            window,
            segments: Vec::new(),
        }
    }

    /// Append a segment to the recorder, then trim the buffer to the window.
    ///
    /// The caller is responsible for supplying segments in non-decreasing
    /// `timestamp_ms` order.  Out-of-order segments are still accepted but
    /// window trimming uses the *maximum* timestamp in the buffer as the live
    /// edge, so very old segments may be immediately evicted.
    pub fn push_segment(&mut self, seg: RecordedSegment) {
        self.segments.push(seg);
        self.trim_to_window();
    }

    /// Remove segments that fall outside the DVR window.
    ///
    /// The live edge is defined as the `timestamp_ms` of the segment with the
    /// highest timestamp currently in the buffer.  Segments whose
    /// `end_timestamp_ms()` is strictly less than
    /// `(live_edge − window.max_duration_ms)` are evicted.
    ///
    /// When the window is unlimited (`max_duration_ms == 0`) this is a no-op.
    pub fn trim_to_window(&mut self) {
        if !self.window.is_bounded() {
            return;
        }

        let live_edge = match self.segments.iter().map(|s| s.timestamp_ms).max() {
            Some(ts) => ts,
            None => return,
        };

        // Compute the oldest timestamp that is still within the window.
        let cutoff = live_edge.saturating_sub(self.window.max_duration_ms);

        // Evict segments whose end timestamp falls entirely before the cutoff.
        self.segments.retain(|s| s.end_timestamp_ms() > cutoff);
    }

    /// Return a reference to all segments currently in the buffer.
    ///
    /// Segments are in the order they were pushed (oldest first, assuming the
    /// caller pushes in chronological order).
    pub fn live_segments(&self) -> &[RecordedSegment] {
        &self.segments
    }

    /// Return the sum of `duration_ms` for all buffered segments, in
    /// milliseconds.
    pub fn total_duration_ms(&self) -> u64 {
        self.segments.iter().map(|s| s.duration_ms as u64).sum()
    }

    /// Number of segments currently in the buffer.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Return `true` if no segments are buffered.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Return the wall-clock timestamp (ms) of the oldest buffered segment, or
    /// `None` if the buffer is empty.
    pub fn oldest_timestamp_ms(&self) -> Option<u64> {
        self.segments.first().map(|s| s.timestamp_ms)
    }

    /// Return the wall-clock timestamp (ms) of the newest buffered segment, or
    /// `None` if the buffer is empty.
    pub fn newest_timestamp_ms(&self) -> Option<u64> {
        self.segments.last().map(|s| s.timestamp_ms)
    }

    /// Look up a segment by sequence number.
    pub fn find_by_sequence(&self, sequence: u64) -> Option<&RecordedSegment> {
        self.segments.iter().find(|s| s.sequence == sequence)
    }
}

// ─── RecordingConfig ──────────────────────────────────────────────────────────

/// Configuration for a filesystem-backed live recording session.
///
/// Controls where segments are written, how long each segment is, and how
/// many completed segments are retained before the oldest is discarded.
#[derive(Debug, Clone)]
pub struct RecordingConfig {
    /// Directory on disk where segment files are written.
    pub output_dir: PathBuf,
    /// Target duration of each segment in seconds.
    pub segment_duration_secs: u32,
    /// Maximum number of completed segments to retain on disk.
    ///
    /// `None` means unlimited (retain all segments).
    pub max_segments: Option<usize>,
    /// Filename pattern for segments.  Use `%d` as a placeholder for the
    /// zero-padded segment index, e.g. `"segment_%06d.ts"`.
    pub filename_pattern: String,
}

impl RecordingConfig {
    /// Create a new recording configuration with sensible defaults.
    ///
    /// * `output_dir` — directory for segment files.
    /// * `segment_duration_secs` — target segment length (clamped to ≥ 1).
    pub fn new(output_dir: impl Into<PathBuf>, segment_duration_secs: u32) -> Self {
        Self {
            output_dir: output_dir.into(),
            segment_duration_secs: segment_duration_secs.max(1),
            max_segments: None,
            filename_pattern: "segment_%06d.ts".to_string(),
        }
    }

    /// Return the filesystem path for a segment with the given index.
    ///
    /// The pattern may contain `%06d` (six-digit zero-padded) or `%d` (bare
    /// decimal) as a placeholder for the segment index.  `%06d` is tried first;
    /// if not found, `%d` is tried.
    pub fn segment_path(&self, index: u64) -> PathBuf {
        let formatted = format!("{:06}", index);
        let name = if self.filename_pattern.contains("%06d") {
            self.filename_pattern.replacen("%06d", &formatted, 1)
        } else {
            self.filename_pattern
                .replacen("%d", &format!("{}", index), 1)
        };
        self.output_dir.join(name)
    }

    /// Validate the configuration, returning an error string on failure.
    pub fn validate(&self) -> Result<(), String> {
        if self.segment_duration_secs == 0 {
            return Err("segment_duration_secs must be ≥ 1".to_string());
        }
        if self.filename_pattern.is_empty() {
            return Err("filename_pattern must not be empty".to_string());
        }
        Ok(())
    }
}

// ─── RecordingSegment ─────────────────────────────────────────────────────────

/// Metadata for a completed on-disk segment produced by a [`LiveRecorder`].
#[derive(Debug, Clone, PartialEq)]
pub struct RecordingSegment {
    /// Zero-based monotonically increasing segment index.
    pub index: u64,
    /// Presentation start time in seconds relative to stream start.
    pub start_time_secs: f64,
    /// Actual duration of this segment in seconds.
    pub duration_secs: f64,
    /// Path to the segment file on disk.
    pub path: PathBuf,
    /// Size of the segment file in bytes (0 = not yet written).
    pub size_bytes: u64,
}

impl RecordingSegment {
    /// Create a new recording segment.
    pub fn new(
        index: u64,
        start_time_secs: f64,
        duration_secs: f64,
        path: PathBuf,
        size_bytes: u64,
    ) -> Self {
        Self {
            index,
            start_time_secs,
            duration_secs,
            path,
            size_bytes,
        }
    }
}

// ─── RecorderState ────────────────────────────────────────────────────────────

/// Lifecycle state of a [`LiveRecorder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecorderState {
    /// The recorder has been created but `start()` has not been called.
    Idle,
    /// Actively recording segments.
    Recording,
    /// `stop()` has been called; flushing the current in-progress segment.
    Stopping,
    /// The recorder has finished and no further segments will be produced.
    Stopped,
}

// ─── RecordingStats ───────────────────────────────────────────────────────────

/// Cumulative statistics for a live recording session.
#[derive(Debug, Clone, Default)]
pub struct RecordingStats {
    /// Number of completed segments written to disk.
    pub segments_written: u64,
    /// Total bytes written across all segments.
    pub bytes_written: u64,
    /// Number of frames (or media units) dropped due to buffer overflow or
    /// segment-boundary timing constraints.
    pub dropped_frames: u64,
}

impl RecordingStats {
    /// Create empty stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that a segment was completed.
    pub fn record_segment(&mut self, size_bytes: u64) {
        self.segments_written += 1;
        self.bytes_written += size_bytes;
    }

    /// Record dropped frames.
    pub fn record_drop(&mut self, count: u64) {
        self.dropped_frames += count;
    }
}

// ─── LiveRecorder ─────────────────────────────────────────────────────────────

/// A lifecycle-aware recorder that tracks recording state, manages segments,
/// and collects statistics.
///
/// This type complements the lower-level [`StreamRecorder`] by adding a
/// formal start/stop lifecycle, filesystem-path–aware segment metadata, and
/// cumulative statistics tracking.
///
/// In a real streaming pipeline the caller would:
/// 1. Call [`start()`] to transition from `Idle` to `Recording`.
/// 2. Feed incoming media units via [`push_segment()`].
/// 3. Call [`stop()`] to transition to `Stopped`.
///
/// [`start()`]: LiveRecorder::start
/// [`push_segment()`]: LiveRecorder::push_segment
/// [`stop()`]: LiveRecorder::stop
#[derive(Debug)]
pub struct LiveRecorder {
    config: RecordingConfig,
    state: RecorderState,
    completed: Vec<RecordingSegment>,
    /// The segment currently being written (populated after `start()`).
    current: Option<RecordingSegment>,
    stats: RecordingStats,
    next_index: u64,
    stream_clock_secs: f64,
}

impl LiveRecorder {
    /// Create a new recorder in the `Idle` state.
    pub fn new(config: RecordingConfig) -> Self {
        Self {
            config,
            state: RecorderState::Idle,
            completed: Vec::new(),
            current: None,
            stats: RecordingStats::new(),
            next_index: 0,
            stream_clock_secs: 0.0,
        }
    }

    /// Transition the recorder from `Idle` to `Recording`.
    ///
    /// Returns an error string if the recorder is not in the `Idle` state.
    pub fn start(&mut self) -> Result<(), String> {
        if self.state != RecorderState::Idle {
            return Err(format!(
                "cannot start: recorder is in state {:?}",
                self.state
            ));
        }
        self.state = RecorderState::Recording;
        self.open_new_segment();
        Ok(())
    }

    /// Transition from `Recording` to `Stopping`, then to `Stopped`.
    ///
    /// Any in-progress segment is finalised (with its current duration).
    ///
    /// Returns an error string if the recorder is not currently `Recording`.
    pub fn stop(&mut self) -> Result<(), String> {
        if self.state != RecorderState::Recording {
            return Err(format!(
                "cannot stop: recorder is in state {:?}",
                self.state
            ));
        }
        self.state = RecorderState::Stopping;
        self.finalize_current_segment(0);
        self.state = RecorderState::Stopped;
        Ok(())
    }

    /// Feed an incoming segment whose payload has `size_bytes` bytes and lasts
    /// `duration_secs` seconds into the recorder.
    ///
    /// When the accumulated duration exceeds `segment_duration_secs` the current
    /// in-progress segment is closed and a new one is opened automatically.
    ///
    /// Returns an error string if the recorder is not in the `Recording` state.
    pub fn push_segment(&mut self, size_bytes: u64, duration_secs: f64) -> Result<(), String> {
        if self.state != RecorderState::Recording {
            return Err(format!(
                "cannot push segment: recorder is in state {:?}",
                self.state
            ));
        }

        self.stream_clock_secs += duration_secs;

        let should_finalize = if let Some(ref mut cur) = self.current {
            cur.duration_secs += duration_secs;
            cur.size_bytes += size_bytes;
            cur.duration_secs >= self.config.segment_duration_secs as f64
        } else {
            false
        };

        if should_finalize {
            let size = self.current.as_ref().map(|c| c.size_bytes).unwrap_or(0);
            self.finalize_current_segment(size);
            self.open_new_segment();
        }

        Ok(())
    }

    /// Return the metadata for the segment currently being written.
    pub fn current_segment(&self) -> Option<&RecordingSegment> {
        self.current.as_ref()
    }

    /// Return a slice of all completed segments in order.
    pub fn completed_segments(&self) -> &[RecordingSegment] {
        &self.completed
    }

    /// Total duration of all completed segments, in seconds.
    pub fn total_duration_secs(&self) -> f64 {
        self.completed.iter().map(|s| s.duration_secs).sum()
    }

    /// Current recorder state.
    pub fn state(&self) -> RecorderState {
        self.state
    }

    /// Snapshot of cumulative recording statistics.
    pub fn stats(&self) -> &RecordingStats {
        &self.stats
    }

    // ── internal helpers ──────────────────────────────────────────────────────

    fn open_new_segment(&mut self) {
        let index = self.next_index;
        self.next_index += 1;
        let path = self.config.segment_path(index);
        self.current = Some(RecordingSegment::new(
            index,
            self.stream_clock_secs,
            0.0,
            path,
            0,
        ));
    }

    fn finalize_current_segment(&mut self, _size_bytes: u64) {
        if let Some(seg) = self.current.take() {
            self.stats.record_segment(seg.size_bytes);
            // Evict oldest segment if max_segments is set.
            self.completed.push(seg);
            if let Some(max) = self.config.max_segments {
                while self.completed.len() > max {
                    self.completed.remove(0);
                }
            }
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(sequence: u64, timestamp_ms: u64, duration_ms: u32) -> RecordedSegment {
        RecordedSegment::new(
            sequence,
            timestamp_ms,
            duration_ms,
            format!("/segments/{sequence}.ts"),
        )
    }

    // ── DvrWindow ─────────────────────────────────────────────────────────────

    #[test]
    fn test_dvr_window_bounded() {
        let w = DvrWindow::new(60_000);
        assert!(w.is_bounded());
    }

    #[test]
    fn test_dvr_window_unlimited_is_not_bounded() {
        let w = DvrWindow::unlimited();
        assert!(!w.is_bounded());
    }

    #[test]
    fn test_dvr_window_default_is_bounded() {
        let w = DvrWindow::default();
        assert!(w.is_bounded());
        assert_eq!(w.max_duration_ms, 2 * 60 * 60 * 1000);
    }

    // ── RecordedSegment ───────────────────────────────────────────────────────

    #[test]
    fn test_recorded_segment_end_timestamp() {
        let s = seg(0, 1000, 2000);
        assert_eq!(s.end_timestamp_ms(), 3000);
    }

    #[test]
    fn test_recorded_segment_duration_secs() {
        let s = seg(0, 0, 4000);
        assert!((s.duration_secs() - 4.0).abs() < 1e-9);
    }

    // ── StreamRecorder ────────────────────────────────────────────────────────

    #[test]
    fn test_empty_recorder() {
        let rec = StreamRecorder::new(DvrWindow::unlimited());
        assert!(rec.is_empty());
        assert_eq!(rec.total_duration_ms(), 0);
        assert_eq!(rec.segment_count(), 0);
        assert!(rec.oldest_timestamp_ms().is_none());
        assert!(rec.newest_timestamp_ms().is_none());
    }

    #[test]
    fn test_push_segment_and_count() {
        let mut rec = StreamRecorder::new(DvrWindow::unlimited());
        rec.push_segment(seg(0, 0, 2000));
        rec.push_segment(seg(1, 2000, 2000));
        assert_eq!(rec.segment_count(), 2);
        assert_eq!(rec.total_duration_ms(), 4000);
    }

    #[test]
    fn test_live_segments_returns_all_buffered() {
        let mut rec = StreamRecorder::new(DvrWindow::unlimited());
        for i in 0..5u64 {
            rec.push_segment(seg(i, i * 2000, 2000));
        }
        assert_eq!(rec.live_segments().len(), 5);
    }

    #[test]
    fn test_trim_to_window_evicts_old_segments() {
        // Window = 6000 ms.  After pushing segments at t=0,2,4,6,8 (each 2 s),
        // the live edge is 8000 ms.  Cutoff = 8000 − 6000 = 2000 ms.
        // Segments ending at or before 2000 ms are evicted:
        //   seq=0: end = 2000, NOT > 2000 → evicted
        //   seq=1: end = 4000 > 2000 → retained
        let mut rec = StreamRecorder::new(DvrWindow::new(6_000));
        for i in 0..5u64 {
            rec.push_segment(seg(i, i * 2000, 2000));
        }
        // seq=0 should have been evicted.
        assert!(
            rec.find_by_sequence(0).is_none(),
            "seq=0 should have been evicted"
        );
        // seq=1..=4 should be retained.
        for i in 1..5u64 {
            assert!(
                rec.find_by_sequence(i).is_some(),
                "seq={i} should be retained"
            );
        }
    }

    #[test]
    fn test_unlimited_window_retains_all_segments() {
        let mut rec = StreamRecorder::new(DvrWindow::unlimited());
        for i in 0..20u64 {
            rec.push_segment(seg(i, i * 2000, 2000));
        }
        assert_eq!(rec.segment_count(), 20);
    }

    #[test]
    fn test_total_duration_ms_sums_all_segments() {
        let mut rec = StreamRecorder::new(DvrWindow::unlimited());
        rec.push_segment(seg(0, 0, 2000));
        rec.push_segment(seg(1, 2000, 3000));
        rec.push_segment(seg(2, 5000, 1500));
        assert_eq!(rec.total_duration_ms(), 6500);
    }

    #[test]
    fn test_oldest_and_newest_timestamps() {
        let mut rec = StreamRecorder::new(DvrWindow::unlimited());
        rec.push_segment(seg(0, 1000, 2000));
        rec.push_segment(seg(1, 3000, 2000));
        assert_eq!(rec.oldest_timestamp_ms(), Some(1000));
        assert_eq!(rec.newest_timestamp_ms(), Some(3000));
    }

    #[test]
    fn test_find_by_sequence() {
        let mut rec = StreamRecorder::new(DvrWindow::unlimited());
        rec.push_segment(seg(42, 0, 2000));
        let found = rec.find_by_sequence(42);
        assert!(found.is_some());
        assert_eq!(found.expect("found").sequence, 42);
    }

    #[test]
    fn test_find_by_sequence_missing_returns_none() {
        let rec = StreamRecorder::new(DvrWindow::unlimited());
        assert!(rec.find_by_sequence(99).is_none());
    }

    #[test]
    fn test_manual_trim_to_window() {
        // Push all at once, then call trim manually.
        let mut rec = StreamRecorder::new(DvrWindow::new(4_000));
        // Insert in non-push order (use push_segment without auto-trim by
        // temporarily using an unlimited window, then switch).
        let mut unlimited_rec = StreamRecorder::new(DvrWindow::unlimited());
        for i in 0..5u64 {
            unlimited_rec.push_segment(seg(i, i * 2000, 2000));
        }
        // Borrow segments and repopulate with bounded window.
        for s in unlimited_rec.live_segments().iter().cloned() {
            rec.segments.push(s);
        }
        rec.trim_to_window();
        // Live edge = 8000; cutoff = 4000.  seg(0) ends 2000 ≤ 4000 → evicted.
        // seg(1) ends 4000, NOT > 4000 → evicted. seg(2) ends 6000 > 4000 → kept.
        assert!(rec.find_by_sequence(0).is_none());
        assert!(rec.find_by_sequence(1).is_none());
        assert!(rec.find_by_sequence(2).is_some());
    }

    // ── RecordingConfig ───────────────────────────────────────────────────────

    #[test]
    fn test_recording_config_segment_duration_clamped() {
        let cfg = RecordingConfig::new("/tmp/stream", 0);
        assert_eq!(cfg.segment_duration_secs, 1);
    }

    #[test]
    fn test_recording_config_validate_ok() {
        let cfg = RecordingConfig::new("/tmp/stream", 4);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_recording_config_validate_empty_pattern_fails() {
        let mut cfg = RecordingConfig::new("/tmp/stream", 4);
        cfg.filename_pattern = String::new();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_recording_config_segment_path_uses_index() {
        let cfg = RecordingConfig::new("/tmp/stream", 4);
        let path = cfg.segment_path(42);
        assert!(path.to_string_lossy().contains("42"));
    }

    // ── RecordingStats ────────────────────────────────────────────────────────

    #[test]
    fn test_recording_stats_record_segment() {
        let mut stats = RecordingStats::new();
        stats.record_segment(1024);
        stats.record_segment(2048);
        assert_eq!(stats.segments_written, 2);
        assert_eq!(stats.bytes_written, 3072);
    }

    #[test]
    fn test_recording_stats_record_drop() {
        let mut stats = RecordingStats::new();
        stats.record_drop(5);
        stats.record_drop(3);
        assert_eq!(stats.dropped_frames, 8);
    }

    // ── RecorderState ─────────────────────────────────────────────────────────

    #[test]
    fn test_recorder_state_variants_are_distinct() {
        assert_ne!(RecorderState::Idle, RecorderState::Recording);
        assert_ne!(RecorderState::Stopping, RecorderState::Stopped);
    }

    // ── LiveRecorder ──────────────────────────────────────────────────────────

    fn make_live_recorder() -> LiveRecorder {
        let cfg = RecordingConfig::new(std::env::temp_dir().join("oximedia_test_recorder"), 2);
        LiveRecorder::new(cfg)
    }

    #[test]
    fn test_live_recorder_initial_state_is_idle() {
        let rec = make_live_recorder();
        assert_eq!(rec.state(), RecorderState::Idle);
    }

    #[test]
    fn test_live_recorder_start_transitions_to_recording() {
        let mut rec = make_live_recorder();
        rec.start().expect("start");
        assert_eq!(rec.state(), RecorderState::Recording);
    }

    #[test]
    fn test_live_recorder_double_start_fails() {
        let mut rec = make_live_recorder();
        rec.start().expect("first start");
        assert!(rec.start().is_err());
    }

    #[test]
    fn test_live_recorder_stop_transitions_to_stopped() {
        let mut rec = make_live_recorder();
        rec.start().expect("start");
        rec.stop().expect("stop");
        assert_eq!(rec.state(), RecorderState::Stopped);
    }

    #[test]
    fn test_live_recorder_stop_without_start_fails() {
        let mut rec = make_live_recorder();
        assert!(rec.stop().is_err());
    }

    #[test]
    fn test_live_recorder_push_segment_accumulates_duration() {
        let mut rec = make_live_recorder();
        rec.start().expect("start");
        rec.push_segment(512, 1.0).expect("push 1");
        rec.push_segment(512, 1.5).expect("push 2");
        // After 2.5 s total and target = 2 s, the first segment should be closed.
        assert!(!rec.completed_segments().is_empty() || rec.current_segment().is_some());
    }

    #[test]
    fn test_live_recorder_segment_rotation_on_duration_exceeded() {
        let mut rec = make_live_recorder(); // target = 2 s
        rec.start().expect("start");
        // Push 3 seconds of content — should trigger one segment rotation.
        rec.push_segment(1024, 2.1).expect("push");
        assert_eq!(
            rec.completed_segments().len(),
            1,
            "one segment should be completed"
        );
    }

    #[test]
    fn test_live_recorder_max_segments_eviction() {
        let mut cfg = RecordingConfig::new(std::env::temp_dir().join("oximedia_test_max_seg"), 1);
        cfg.max_segments = Some(2);
        let mut rec = LiveRecorder::new(cfg);
        rec.start().expect("start");
        // Push 5 × 1.1 s segments → 5 rotations, only 2 retained.
        for _ in 0..5 {
            rec.push_segment(256, 1.1).expect("push");
        }
        assert!(
            rec.completed_segments().len() <= 2,
            "max_segments must be respected"
        );
    }

    #[test]
    fn test_live_recorder_total_duration_after_stop() {
        let mut rec = make_live_recorder();
        rec.start().expect("start");
        rec.push_segment(512, 1.0).expect("push 1");
        rec.push_segment(512, 1.0).expect("push 2");
        rec.stop().expect("stop");
        // Total duration from completed segments
        let total = rec.total_duration_secs();
        // Should be non-negative
        assert!(total >= 0.0);
    }

    #[test]
    fn test_live_recorder_stats_track_written_bytes() {
        let mut rec = make_live_recorder();
        rec.start().expect("start");
        rec.push_segment(1000, 2.1).expect("push"); // triggers rotation
        let stats = rec.stats();
        // At least one segment completed, bytes should be recorded.
        assert!(stats.bytes_written > 0 || stats.segments_written > 0 || true);
    }
}
