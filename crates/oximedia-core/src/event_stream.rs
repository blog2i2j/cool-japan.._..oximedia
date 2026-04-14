//! Time-indexed media event stream.
//!
//! Provides an ordered sequence of [`StreamEvent`]s, each tagged with a
//! presentation timestamp. Events represent pipeline-significant occurrences
//! in a media stream such as keyframe boundaries, scene changes, silence
//! intervals, and chapter markers.
//!
//! # Design Goals
//!
//! - **Chronological order**: events are stored sorted by `pts_ticks` so that
//!   time-range queries can use binary search without a full scan.
//! - **Rich event taxonomy**: the [`StreamEventKind`] enum covers the most
//!   common pipeline events without requiring per-crate extension types.
//! - **Efficient range queries**: [`EventStream::in_range`] returns a slice of
//!   references for a `[start, end)` tick interval in O(log n + k) time.
//! - **No unsafe, no unwrap in library code**.
//!
//! # Example
//!
//! ```
//! use oximedia_core::event_stream::{EventStream, StreamEvent, StreamEventKind};
//! use oximedia_core::types::Rational;
//!
//! let tb = Rational::new(1, 90_000);
//! let mut stream = EventStream::new(tb);
//!
//! stream.insert(StreamEvent::new(0, StreamEventKind::Keyframe, tb));
//! stream.insert(StreamEvent::new(90_000, StreamEventKind::SceneChange, tb));
//! stream.insert(StreamEvent::new(180_000, StreamEventKind::Keyframe, tb));
//!
//! let events = stream.in_range(0, 180_000);
//! assert_eq!(events.len(), 2); // start inclusive, end exclusive
//!
//! assert_eq!(stream.first_of_kind(StreamEventKind::SceneChange).map(|e| e.pts_ticks), Some(90_000));
//! ```

use crate::types::Rational;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// StreamEventKind
// ---------------------------------------------------------------------------

/// The kind of event that occurred in the stream.
///
/// This enum covers the most common media pipeline events. Additional context
/// may be carried in the [`StreamEvent::payload`] field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamEventKind {
    /// A keyframe (Intra / IDR frame) is available at this PTS.
    Keyframe,
    /// A scene change was detected at this PTS.
    SceneChange,
    /// Audio silence began at this PTS.
    SilenceStart,
    /// Audio silence ended at this PTS.
    SilenceEnd,
    /// Chapter / chapter marker boundary.
    Chapter,
    /// A splice-in point (SCTE-35 / HLS `EXT-X-CUE-IN`).
    SpliceIn,
    /// A splice-out point (SCTE-35 / HLS `EXT-X-CUE-OUT`).
    SpliceOut,
    /// Audio peak loudness event (e.g. EBU R128 short-term threshold exceeded).
    LoudnessPeak,
    /// Flash or strobing hazard detected (Harding test).
    FlashHazard,
    /// Start of a black video segment (e.g. inter-programme gap).
    BlackStart,
    /// End of a black video segment.
    BlackEnd,
    /// End-of-stream signal.
    EndOfStream,
    /// Custom / application-defined event (use `payload` for details).
    Custom,
}

impl StreamEventKind {
    /// Returns `true` if this event kind marks a random-access point in the
    /// video stream.
    #[must_use]
    pub fn is_random_access(self) -> bool {
        matches!(self, Self::Keyframe)
    }

    /// Returns `true` if this event signals a content boundary (scene change,
    /// chapter, splice, or EOS).
    #[must_use]
    pub fn is_boundary(self) -> bool {
        matches!(
            self,
            Self::SceneChange
                | Self::Chapter
                | Self::SpliceIn
                | Self::SpliceOut
                | Self::EndOfStream
        )
    }

    /// Returns `true` if this event relates to audio characteristics (silence,
    /// loudness).
    #[must_use]
    pub fn is_audio_event(self) -> bool {
        matches!(
            self,
            Self::SilenceStart | Self::SilenceEnd | Self::LoudnessPeak
        )
    }
}

impl std::fmt::Display for StreamEventKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Keyframe => "keyframe",
            Self::SceneChange => "scene_change",
            Self::SilenceStart => "silence_start",
            Self::SilenceEnd => "silence_end",
            Self::Chapter => "chapter",
            Self::SpliceIn => "splice_in",
            Self::SpliceOut => "splice_out",
            Self::LoudnessPeak => "loudness_peak",
            Self::FlashHazard => "flash_hazard",
            Self::BlackStart => "black_start",
            Self::BlackEnd => "black_end",
            Self::EndOfStream => "end_of_stream",
            Self::Custom => "custom",
        };
        f.write_str(s)
    }
}

// ---------------------------------------------------------------------------
// StreamEvent
// ---------------------------------------------------------------------------

/// A single event in a [`EventStream`], tagged with a presentation timestamp.
///
/// Events are ordered by `pts_ticks` ascending; events with equal PTS are
/// ordered by insertion order (FIFO within the same tick).
#[derive(Debug, Clone, PartialEq)]
pub struct StreamEvent {
    /// Presentation timestamp in ticks (units of `time_base`).
    pub pts_ticks: i64,
    /// Time base used to interpret `pts_ticks`.
    pub time_base: Rational,
    /// The kind of event.
    pub kind: StreamEventKind,
    /// Optional free-form payload (chapter title, loudness value, …).
    pub payload: Option<String>,
    /// Optional confidence score in the range `[0.0, 1.0]` (for detector
    /// outputs such as scene-change probability).
    pub confidence: Option<f32>,
}

impl StreamEvent {
    /// Creates a new [`StreamEvent`] with no payload or confidence.
    #[must_use]
    pub fn new(pts_ticks: i64, kind: StreamEventKind, time_base: Rational) -> Self {
        Self {
            pts_ticks,
            time_base,
            kind,
            payload: None,
            confidence: None,
        }
    }

    /// Builder-style setter for the payload.
    #[must_use]
    pub fn with_payload(mut self, payload: impl Into<String>) -> Self {
        self.payload = Some(payload.into());
        self
    }

    /// Builder-style setter for the confidence score.
    ///
    /// Values outside `[0.0, 1.0]` are clamped to the valid range.
    #[must_use]
    pub fn with_confidence(mut self, score: f32) -> Self {
        self.confidence = Some(score.clamp(0.0, 1.0));
        self
    }

    /// Returns the wall-clock presentation time in seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn pts_secs(&self) -> f64 {
        self.pts_ticks as f64 * self.time_base.to_f64()
    }
}

impl std::fmt::Display for StreamEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:.6}s] {}", self.pts_secs(), self.kind)
    }
}

// ---------------------------------------------------------------------------
// EventStream
// ---------------------------------------------------------------------------

/// A chronologically ordered, time-indexed sequence of [`StreamEvent`]s.
///
/// Events are kept sorted by `pts_ticks` so that slice-based range queries
/// run in O(log n + k) time via binary search.  Insertion of out-of-order
/// events is supported but triggers a sort (O(n log n)); for best performance
/// insert events in ascending PTS order.
#[derive(Debug, Clone)]
pub struct EventStream {
    events: Vec<StreamEvent>,
    /// Default time base for newly inserted events that carry the same base.
    pub time_base: Rational,
    /// Whether the internal buffer is currently sorted.
    sorted: bool,
}

impl EventStream {
    /// Creates an empty `EventStream` with the given default time base.
    #[must_use]
    pub fn new(time_base: Rational) -> Self {
        Self {
            events: Vec::new(),
            time_base,
            sorted: true,
        }
    }

    /// Creates an `EventStream` pre-allocated for `capacity` events.
    #[must_use]
    pub fn with_capacity(time_base: Rational, capacity: usize) -> Self {
        Self {
            events: Vec::with_capacity(capacity),
            time_base,
            sorted: true,
        }
    }

    /// Inserts a [`StreamEvent`] into the stream.
    ///
    /// If the event has a PTS ≥ the current maximum PTS it is appended in
    /// O(1); otherwise the stream is marked unsorted and will be sorted lazily
    /// on the next query.
    pub fn insert(&mut self, event: StreamEvent) {
        if self.sorted {
            if let Some(last) = self.events.last() {
                if event.pts_ticks < last.pts_ticks {
                    self.sorted = false;
                }
            }
        }
        self.events.push(event);
    }

    /// Returns the number of events in the stream.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns `true` if the stream contains no events.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Returns all events whose PTS falls in `[start_ticks, end_ticks)`.
    ///
    /// Triggers a sort if the stream is not currently sorted.
    pub fn in_range(&mut self, start_ticks: i64, end_ticks: i64) -> Vec<&StreamEvent> {
        self.ensure_sorted();
        let lo = self.events.partition_point(|e| e.pts_ticks < start_ticks);
        let hi = self.events.partition_point(|e| e.pts_ticks < end_ticks);
        self.events[lo..hi].iter().collect()
    }

    /// Returns the event at exactly `pts_ticks`, or `None`.
    ///
    /// When multiple events share the same PTS, the first one in insertion
    /// order is returned.
    pub fn at(&mut self, pts_ticks: i64) -> Option<&StreamEvent> {
        self.ensure_sorted();
        let idx = self.events.partition_point(|e| e.pts_ticks < pts_ticks);
        self.events.get(idx).filter(|e| e.pts_ticks == pts_ticks)
    }

    /// Returns the first event of the given `kind`, in PTS order.
    pub fn first_of_kind(&mut self, kind: StreamEventKind) -> Option<&StreamEvent> {
        self.ensure_sorted();
        self.events.iter().find(|e| e.kind == kind)
    }

    /// Returns all events of the given `kind`, in PTS order.
    pub fn all_of_kind(&mut self, kind: StreamEventKind) -> Vec<&StreamEvent> {
        self.ensure_sorted();
        self.events.iter().filter(|e| e.kind == kind).collect()
    }

    /// Returns an iterator over all events in PTS order.
    ///
    /// If the stream is unsorted this triggers a sort.
    pub fn iter(&mut self) -> impl Iterator<Item = &StreamEvent> {
        self.ensure_sorted();
        self.events.iter()
    }

    /// Returns the event with the smallest PTS, or `None`.
    pub fn earliest(&mut self) -> Option<&StreamEvent> {
        self.ensure_sorted();
        self.events.first()
    }

    /// Returns the event with the largest PTS, or `None`.
    pub fn latest(&mut self) -> Option<&StreamEvent> {
        self.ensure_sorted();
        self.events.last()
    }

    /// Retains only events that satisfy the predicate `f`.
    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&StreamEvent) -> bool,
    {
        self.events.retain(f);
        // Relative order is preserved so sorted status is unchanged.
    }

    /// Removes all events from the stream.
    pub fn clear(&mut self) {
        self.events.clear();
        self.sorted = true;
    }

    /// Returns a histogram mapping each [`StreamEventKind`] to its count.
    pub fn kind_histogram(&mut self) -> HashMap<StreamEventKind, usize> {
        let mut map: HashMap<StreamEventKind, usize> = HashMap::new();
        for ev in &self.events {
            *map.entry(ev.kind).or_insert(0) += 1;
        }
        map
    }

    /// Returns the total span of this stream in seconds, i.e. the duration
    /// from the earliest to the latest event.
    ///
    /// Returns `0.0` if the stream contains fewer than two events.
    pub fn span_secs(&mut self) -> f64 {
        self.ensure_sorted();
        match (self.events.first(), self.events.last()) {
            (Some(first), Some(last)) if first.pts_ticks != last.pts_ticks => {
                last.pts_secs() - first.pts_secs()
            }
            _ => 0.0,
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn ensure_sorted(&mut self) {
        if !self.sorted {
            self.events.sort_by_key(|e| e.pts_ticks);
            self.sorted = true;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Rational;

    fn tb() -> Rational {
        Rational::new(1, 90_000)
    }

    fn ev(pts: i64, kind: StreamEventKind) -> StreamEvent {
        StreamEvent::new(pts, kind, tb())
    }

    // --- StreamEventKind ---

    #[test]
    fn test_event_kind_is_random_access() {
        assert!(StreamEventKind::Keyframe.is_random_access());
        assert!(!StreamEventKind::SceneChange.is_random_access());
    }

    #[test]
    fn test_event_kind_is_boundary() {
        assert!(StreamEventKind::SceneChange.is_boundary());
        assert!(StreamEventKind::Chapter.is_boundary());
        assert!(StreamEventKind::SpliceIn.is_boundary());
        assert!(StreamEventKind::EndOfStream.is_boundary());
        assert!(!StreamEventKind::Keyframe.is_boundary());
        assert!(!StreamEventKind::SilenceStart.is_boundary());
    }

    #[test]
    fn test_event_kind_is_audio_event() {
        assert!(StreamEventKind::SilenceStart.is_audio_event());
        assert!(StreamEventKind::SilenceEnd.is_audio_event());
        assert!(StreamEventKind::LoudnessPeak.is_audio_event());
        assert!(!StreamEventKind::Keyframe.is_audio_event());
    }

    #[test]
    fn test_event_kind_display() {
        assert_eq!(format!("{}", StreamEventKind::Keyframe), "keyframe");
        assert_eq!(format!("{}", StreamEventKind::SceneChange), "scene_change");
        assert_eq!(format!("{}", StreamEventKind::EndOfStream), "end_of_stream");
    }

    // --- StreamEvent ---

    #[test]
    fn test_stream_event_pts_secs() {
        let e = ev(90_000, StreamEventKind::Keyframe);
        assert!((e.pts_secs() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stream_event_with_payload() {
        let e = ev(0, StreamEventKind::Chapter).with_payload("Intro");
        assert_eq!(e.payload.as_deref(), Some("Intro"));
    }

    #[test]
    fn test_stream_event_with_confidence_clamped() {
        let e1 = ev(0, StreamEventKind::SceneChange).with_confidence(0.87);
        let e2 = ev(0, StreamEventKind::SceneChange).with_confidence(1.5); // should clamp
        let e3 = ev(0, StreamEventKind::SceneChange).with_confidence(-0.3); // should clamp
        assert!((e1.confidence.unwrap_or(0.0) - 0.87_f32).abs() < 1e-6);
        assert!((e2.confidence.unwrap_or(0.0) - 1.0_f32).abs() < 1e-6);
        assert!((e3.confidence.unwrap_or(1.0) - 0.0_f32).abs() < 1e-6);
    }

    #[test]
    fn test_stream_event_display() {
        let e = ev(90_000, StreamEventKind::Keyframe);
        let s = format!("{e}");
        assert!(s.contains("keyframe"));
        assert!(s.contains("1."));
    }

    // --- EventStream: basic ---

    #[test]
    fn test_event_stream_insert_and_len() {
        let mut stream = EventStream::new(tb());
        assert!(stream.is_empty());
        stream.insert(ev(0, StreamEventKind::Keyframe));
        stream.insert(ev(90_000, StreamEventKind::SceneChange));
        assert_eq!(stream.len(), 2);
    }

    #[test]
    fn test_event_stream_in_range_basic() {
        let mut stream = EventStream::new(tb());
        stream.insert(ev(0, StreamEventKind::Keyframe));
        stream.insert(ev(90_000, StreamEventKind::SceneChange));
        stream.insert(ev(180_000, StreamEventKind::Keyframe));
        stream.insert(ev(270_000, StreamEventKind::Chapter));

        // [0, 180_000) → 0 and 90_000 only
        let r = stream.in_range(0, 180_000);
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].pts_ticks, 0);
        assert_eq!(r[1].pts_ticks, 90_000);
    }

    #[test]
    fn test_event_stream_in_range_empty() {
        let mut stream = EventStream::new(tb());
        stream.insert(ev(90_000, StreamEventKind::Keyframe));
        // Query before any events
        let r = stream.in_range(0, 45_000);
        assert!(r.is_empty());
    }

    #[test]
    fn test_event_stream_at_exact() {
        let mut stream = EventStream::new(tb());
        stream.insert(ev(0, StreamEventKind::Keyframe));
        stream.insert(ev(90_000, StreamEventKind::SceneChange));

        let found = stream.at(90_000);
        assert!(found.is_some());
        assert_eq!(found.map(|e| e.kind), Some(StreamEventKind::SceneChange));

        assert!(stream.at(45_000).is_none());
    }

    #[test]
    fn test_event_stream_first_of_kind() {
        let mut stream = EventStream::new(tb());
        stream.insert(ev(90_000, StreamEventKind::SceneChange));
        stream.insert(ev(0, StreamEventKind::Keyframe));
        stream.insert(ev(180_000, StreamEventKind::SceneChange));

        let first_sc = stream
            .first_of_kind(StreamEventKind::SceneChange)
            .expect("should find scene change");
        assert_eq!(first_sc.pts_ticks, 90_000);
    }

    #[test]
    fn test_event_stream_all_of_kind() {
        let mut stream = EventStream::new(tb());
        stream.insert(ev(0, StreamEventKind::Keyframe));
        stream.insert(ev(90_000, StreamEventKind::Keyframe));
        stream.insert(ev(45_000, StreamEventKind::SceneChange));

        let kfs = stream.all_of_kind(StreamEventKind::Keyframe);
        assert_eq!(kfs.len(), 2);
        assert_eq!(kfs[0].pts_ticks, 0);
        assert_eq!(kfs[1].pts_ticks, 90_000);
    }

    #[test]
    fn test_event_stream_out_of_order_insertion() {
        let mut stream = EventStream::new(tb());
        stream.insert(ev(180_000, StreamEventKind::Chapter));
        stream.insert(ev(0, StreamEventKind::Keyframe));
        stream.insert(ev(90_000, StreamEventKind::SceneChange));

        // After any query the stream should be sorted
        let all: Vec<_> = stream.iter().collect();
        assert_eq!(all[0].pts_ticks, 0);
        assert_eq!(all[1].pts_ticks, 90_000);
        assert_eq!(all[2].pts_ticks, 180_000);
    }

    #[test]
    fn test_event_stream_earliest_and_latest() {
        let mut stream = EventStream::new(tb());
        stream.insert(ev(270_000, StreamEventKind::EndOfStream));
        stream.insert(ev(0, StreamEventKind::Keyframe));

        assert_eq!(stream.earliest().map(|e| e.pts_ticks), Some(0));
        assert_eq!(stream.latest().map(|e| e.pts_ticks), Some(270_000));
    }

    #[test]
    fn test_event_stream_retain() {
        let mut stream = EventStream::new(tb());
        stream.insert(ev(0, StreamEventKind::Keyframe));
        stream.insert(ev(90_000, StreamEventKind::SceneChange));
        stream.insert(ev(180_000, StreamEventKind::Keyframe));

        stream.retain(|e| e.kind == StreamEventKind::Keyframe);
        assert_eq!(stream.len(), 2);
    }

    #[test]
    fn test_event_stream_kind_histogram() {
        let mut stream = EventStream::new(tb());
        stream.insert(ev(0, StreamEventKind::Keyframe));
        stream.insert(ev(90_000, StreamEventKind::Keyframe));
        stream.insert(ev(45_000, StreamEventKind::SceneChange));

        let hist = stream.kind_histogram();
        assert_eq!(hist.get(&StreamEventKind::Keyframe).copied(), Some(2));
        assert_eq!(hist.get(&StreamEventKind::SceneChange).copied(), Some(1));
    }

    #[test]
    fn test_event_stream_span_secs() {
        let mut stream = EventStream::new(tb());
        stream.insert(ev(0, StreamEventKind::Keyframe));
        stream.insert(ev(270_000, StreamEventKind::EndOfStream));

        let span = stream.span_secs();
        assert!((span - 3.0).abs() < 1e-6, "expected 3.0s, got {span}");
    }

    #[test]
    fn test_event_stream_clear() {
        let mut stream = EventStream::new(tb());
        stream.insert(ev(0, StreamEventKind::Keyframe));
        stream.clear();
        assert!(stream.is_empty());
    }

    #[test]
    fn test_event_stream_with_capacity() {
        let stream = EventStream::with_capacity(tb(), 64);
        assert!(stream.is_empty());
        assert_eq!(stream.events.capacity(), 64);
    }
}
