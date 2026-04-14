//! NDI metadata channel — XML metadata frame parsing, embedded metadata types, and frame queuing.
//!
//! NDI sources can carry an asynchronous metadata channel alongside the video/audio essence.
//! This channel transports XML payloads that encode structured information such as tally state,
//! PTZ position, KVM events, or arbitrary user-defined key-value bags.  This module provides:
//!
//! * [`MetadataType`] — discriminated union of the recognised embedded payload kinds.
//! * [`MetadataChannelFrame`] — a fully parsed metadata frame ready for consumption.
//! * [`MetadataChannelParser`] — lightweight XML attribute extractor for NDI payloads.
//! * [`MetadataChannelQueue`] — bounded FIFO queue with overflow-drop semantics.
//! * [`MetadataChannelStats`] — counters for received, parsed, and overflow-dropped frames.

#![allow(dead_code)]
#![allow(clippy::module_name_repetitions)]

use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────────────────────────────────────
// MetadataType
// ─────────────────────────────────────────────────────────────────────────────

/// Tally state embedded inside an NDI metadata frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddedTally {
    /// Source is live on program output (red tally).
    pub program: bool,
    /// Source is in preview (green tally).
    pub preview: bool,
}

impl EmbeddedTally {
    /// Construct a new `EmbeddedTally`.
    pub fn new(program: bool, preview: bool) -> Self {
        Self { program, preview }
    }
}

/// PTZ absolute position payload embedded in an NDI metadata frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmbeddedPtz {
    /// Pan angle in degrees (−180 … 180).
    pub pan_deg: f32,
    /// Tilt angle in degrees (−90 … 90).
    pub tilt_deg: f32,
    /// Zoom level (1.0 = optical identity, higher = more zoom).
    pub zoom: f32,
}

impl EmbeddedPtz {
    /// Construct a new `EmbeddedPtz`.
    pub fn new(pan_deg: f32, tilt_deg: f32, zoom: f32) -> Self {
        Self {
            pan_deg,
            tilt_deg,
            zoom,
        }
    }
}

/// Discriminated union of all recognised NDI metadata payload types.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataType {
    /// Tally state update.
    Tally(EmbeddedTally),
    /// PTZ absolute position update.
    Ptz(EmbeddedPtz),
    /// Arbitrary key-value bag for user-defined metadata.
    Custom {
        /// Tag name of the XML root element (e.g. `"metadata"` or `"product"`).
        tag: String,
        /// Key-value attributes/fields extracted from the payload.
        fields: Vec<(String, String)>,
    },
    /// Raw XML that did not match any recognised schema.
    Unknown(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// MetadataChannelFrame
// ─────────────────────────────────────────────────────────────────────────────

/// A fully-parsed NDI metadata channel frame.
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataChannelFrame {
    /// Monotonic receive time (from `Instant::now()` at point of enqueue).
    pub received_at: Instant,
    /// Sender-supplied timestamp in milliseconds since Unix epoch (may be 0).
    pub timestamp_ms: u64,
    /// Original XML payload string.
    pub raw_xml: String,
    /// Parsed representation of the payload.
    pub kind: MetadataType,
}

impl MetadataChannelFrame {
    /// Create a new frame from raw XML; parsing is performed immediately.
    pub fn from_raw(raw_xml: String, timestamp_ms: u64) -> Self {
        let kind = MetadataChannelParser::classify(&raw_xml);
        Self {
            received_at: Instant::now(),
            timestamp_ms,
            raw_xml,
            kind,
        }
    }

    /// Returns `true` if this frame carries a tally update.
    pub fn is_tally(&self) -> bool {
        matches!(self.kind, MetadataType::Tally(_))
    }

    /// Returns `true` if this frame carries a PTZ position update.
    pub fn is_ptz(&self) -> bool {
        matches!(self.kind, MetadataType::Ptz(_))
    }

    /// Elapsed time since the frame was enqueued (useful for latency tracking).
    pub fn age(&self) -> Duration {
        self.received_at.elapsed()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MetadataChannelParser
// ─────────────────────────────────────────────────────────────────────────────

/// Lightweight XML attribute/field extractor for NDI metadata payloads.
///
/// NDI metadata XML is always small (< 4 KiB), so we use simple string scanning
/// rather than pulling in a full XML parser dependency.
pub struct MetadataChannelParser;

impl MetadataChannelParser {
    /// Classify an XML string into a [`MetadataType`].
    pub fn classify(xml: &str) -> MetadataType {
        let trimmed = xml.trim();

        if Self::starts_with_tag(trimmed, "tally") {
            return MetadataType::Tally(Self::parse_tally(trimmed));
        }

        if Self::starts_with_tag(trimmed, "ptz") {
            if let Some(ptz) = Self::parse_ptz(trimmed) {
                return MetadataType::Ptz(ptz);
            }
        }

        if let Some(tag) = Self::root_tag(trimmed) {
            let fields = Self::extract_fields(trimmed);
            return MetadataType::Custom { tag, fields };
        }

        MetadataType::Unknown(xml.to_string())
    }

    /// Parse a `<tally program="…" preview="…"/>` element.
    fn parse_tally(xml: &str) -> EmbeddedTally {
        let program = Self::attr_bool(xml, "program");
        let preview = Self::attr_bool(xml, "preview");
        EmbeddedTally::new(program, preview)
    }

    /// Parse a `<ptz pan="…" tilt="…" zoom="…"/>` element.
    fn parse_ptz(xml: &str) -> Option<EmbeddedPtz> {
        let pan = Self::attr_f32(xml, "pan")?;
        let tilt = Self::attr_f32(xml, "tilt")?;
        let zoom = Self::attr_f32(xml, "zoom")?;
        Some(EmbeddedPtz::new(pan, tilt, zoom))
    }

    /// Extract `<field key="k">v</field>` pairs from a metadata envelope.
    pub fn extract_fields(xml: &str) -> Vec<(String, String)> {
        let mut result = Vec::new();
        let mut pos = 0;
        while let Some(start) = xml[pos..].find("<field key=\"") {
            let abs = pos + start;
            let key_start = abs + "<field key=\"".len();
            if let Some(key_end_rel) = xml[key_start..].find('"') {
                let key_end = key_start + key_end_rel;
                let key = unescape(xml[key_start..key_end].trim());

                if let Some(tag_close_rel) = xml[key_end..].find('>') {
                    let val_start = key_end + tag_close_rel + 1;
                    if let Some(val_end_rel) = xml[val_start..].find("</field>") {
                        let val_end = val_start + val_end_rel;
                        let value = unescape(xml[val_start..val_end].trim());
                        result.push((key, value));
                        pos = val_end + "</field>".len();
                        continue;
                    }
                }
            }
            break;
        }
        result
    }

    /// Extract a named attribute value as a boolean from an XML tag.
    fn attr_bool(xml: &str, name: &str) -> bool {
        Self::attr_str(xml, name)
            .map(|v| v == "true" || v == "1" || v == "yes")
            .unwrap_or(false)
    }

    /// Extract a named attribute value as an `f32`.
    fn attr_f32(xml: &str, name: &str) -> Option<f32> {
        Self::attr_str(xml, name).and_then(|v| v.parse::<f32>().ok())
    }

    /// Extract a named attribute value as a `&str` slice from the XML string.
    fn attr_str<'a>(xml: &'a str, name: &str) -> Option<&'a str> {
        let needle = format!("{}=\"", name);
        let start = xml.find(needle.as_str())? + needle.len();
        let end = start + xml[start..].find('"')?;
        Some(&xml[start..end])
    }

    /// Check whether the XML string begins with a named opening tag.
    fn starts_with_tag(xml: &str, tag: &str) -> bool {
        let open = format!("<{}", tag);
        xml.starts_with(open.as_str())
    }

    /// Extract the root tag name from an XML string.
    fn root_tag(xml: &str) -> Option<String> {
        if !xml.starts_with('<') {
            return None;
        }
        let end = xml[1..].find([' ', '>', '/'])?;
        let tag = &xml[1..1 + end];
        if tag.is_empty() || tag.starts_with('/') {
            return None;
        }
        Some(tag.to_string())
    }
}

/// Unescape minimal XML character references.
fn unescape(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

// ─────────────────────────────────────────────────────────────────────────────
// MetadataChannelStats
// ─────────────────────────────────────────────────────────────────────────────

/// Counters for a [`MetadataChannelQueue`].
#[derive(Debug, Clone, Default)]
pub struct MetadataChannelStats {
    /// Total frames enqueued (including overflows that were subsequently dropped).
    pub total_received: u64,
    /// Total frames successfully dequeued by consumers.
    pub total_consumed: u64,
    /// Frames silently dropped because the queue was full.
    pub overflow_dropped: u64,
    /// Frames that could not be classified as tally or PTZ.
    pub unknown_frames: u64,
}

impl MetadataChannelStats {
    /// Create zeroed stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of frames currently pending in the queue.
    ///
    /// May undercount in the presence of concurrent access because these are
    /// plain `u64` fields (no atomics) — intended for single-threaded use.
    pub fn pending(&self) -> u64 {
        self.total_received
            .saturating_sub(self.total_consumed)
            .saturating_sub(self.overflow_dropped)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MetadataChannelQueue
// ─────────────────────────────────────────────────────────────────────────────

/// Bounded FIFO queue of [`MetadataChannelFrame`]s with overflow-drop semantics.
///
/// When the queue is full the *oldest* frame is evicted to make room for the
/// incoming one.  This matches the behaviour of NDI receivers which always prefer
/// the most recent metadata.
#[derive(Debug)]
pub struct MetadataChannelQueue {
    frames: VecDeque<MetadataChannelFrame>,
    capacity: usize,
    stats: MetadataChannelStats,
}

impl MetadataChannelQueue {
    /// Create a new queue with the given capacity (minimum 1).
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            frames: VecDeque::with_capacity(capacity),
            capacity,
            stats: MetadataChannelStats::new(),
        }
    }

    /// Push a raw XML string into the queue, parsing it automatically.
    ///
    /// If the queue is at capacity, the oldest frame is evicted first.
    pub fn push_raw(&mut self, xml: String, timestamp_ms: u64) {
        self.stats.total_received += 1;

        if self.frames.len() >= self.capacity {
            self.frames.pop_front();
            self.stats.overflow_dropped += 1;
        }

        let frame = MetadataChannelFrame::from_raw(xml, timestamp_ms);
        if matches!(frame.kind, MetadataType::Unknown(_)) {
            self.stats.unknown_frames += 1;
        }
        self.frames.push_back(frame);
    }

    /// Push a pre-built frame into the queue.
    ///
    /// If the queue is at capacity, the oldest frame is evicted first.
    pub fn push_frame(&mut self, frame: MetadataChannelFrame) {
        self.stats.total_received += 1;

        if self.frames.len() >= self.capacity {
            self.frames.pop_front();
            self.stats.overflow_dropped += 1;
        }

        if matches!(frame.kind, MetadataType::Unknown(_)) {
            self.stats.unknown_frames += 1;
        }
        self.frames.push_back(frame);
    }

    /// Pop the oldest frame from the queue, or `None` if empty.
    pub fn pop(&mut self) -> Option<MetadataChannelFrame> {
        let frame = self.frames.pop_front()?;
        self.stats.total_consumed += 1;
        Some(frame)
    }

    /// Peek at the oldest frame without removing it.
    pub fn peek(&self) -> Option<&MetadataChannelFrame> {
        self.frames.front()
    }

    /// Drain all frames that match a predicate, returning them.
    pub fn drain_matching(
        &mut self,
        pred: impl Fn(&MetadataChannelFrame) -> bool,
    ) -> Vec<MetadataChannelFrame> {
        let mut matched = Vec::new();
        let mut remaining = VecDeque::with_capacity(self.frames.len());
        for frame in self.frames.drain(..) {
            if pred(&frame) {
                self.stats.total_consumed += 1;
                matched.push(frame);
            } else {
                remaining.push_back(frame);
            }
        }
        self.frames = remaining;
        matched
    }

    /// Number of frames currently in the queue.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Returns `true` when the queue contains no frames.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Maximum number of frames the queue can hold without eviction.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Immutable reference to cumulative statistics.
    pub fn stats(&self) -> &MetadataChannelStats {
        &self.stats
    }

    /// Discard all queued frames.
    pub fn clear(&mut self) {
        self.frames.clear();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── parser ────────────────────────────────────────────────────────────────

    #[test]
    fn test_classify_tally_program() {
        let xml = r#"<tally program="true" preview="false"/>"#;
        match MetadataChannelParser::classify(xml) {
            MetadataType::Tally(t) => {
                assert!(t.program);
                assert!(!t.preview);
            }
            other => panic!("expected Tally, got {other:?}"),
        }
    }

    #[test]
    fn test_classify_tally_preview() {
        let xml = r#"<tally program="false" preview="true"/>"#;
        match MetadataChannelParser::classify(xml) {
            MetadataType::Tally(t) => {
                assert!(!t.program);
                assert!(t.preview);
            }
            other => panic!("expected Tally, got {other:?}"),
        }
    }

    #[test]
    fn test_classify_ptz() {
        let xml = r#"<ptz pan="45.5" tilt="-10.0" zoom="2.0"/>"#;
        match MetadataChannelParser::classify(xml) {
            MetadataType::Ptz(p) => {
                assert!((p.pan_deg - 45.5).abs() < 1e-4);
                assert!((p.tilt_deg - (-10.0)).abs() < 1e-4);
                assert!((p.zoom - 2.0).abs() < 1e-4);
            }
            other => panic!("expected Ptz, got {other:?}"),
        }
    }

    #[test]
    fn test_classify_custom_metadata() {
        let xml = r#"<metadata><field key="camera">CAM1</field></metadata>"#;
        match MetadataChannelParser::classify(xml) {
            MetadataType::Custom { tag, fields } => {
                assert_eq!(tag, "metadata");
                assert_eq!(fields.len(), 1);
                assert_eq!(fields[0].0, "camera");
                assert_eq!(fields[0].1, "CAM1");
            }
            other => panic!("expected Custom, got {other:?}"),
        }
    }

    #[test]
    fn test_classify_unknown() {
        let xml = "not xml at all";
        assert!(matches!(
            MetadataChannelParser::classify(xml),
            MetadataType::Unknown(_)
        ));
    }

    #[test]
    fn test_extract_fields_multiple() {
        let xml = r#"<metadata><field key="a">1</field><field key="b">2</field></metadata>"#;
        let fields = MetadataChannelParser::extract_fields(xml);
        assert_eq!(fields.len(), 2);
        assert!(fields.contains(&("a".to_string(), "1".to_string())));
        assert!(fields.contains(&("b".to_string(), "2".to_string())));
    }

    // ── frame ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_frame_from_raw_tally() {
        let xml = r#"<tally program="true" preview="false"/>"#;
        let frame = MetadataChannelFrame::from_raw(xml.to_string(), 9999);
        assert!(frame.is_tally());
        assert!(!frame.is_ptz());
        assert_eq!(frame.timestamp_ms, 9999);
    }

    #[test]
    fn test_frame_age_nonnegative() {
        let xml = r#"<ptz pan="0.0" tilt="0.0" zoom="1.0"/>"#;
        let frame = MetadataChannelFrame::from_raw(xml.to_string(), 0);
        assert!(frame.age() >= Duration::ZERO);
    }

    // ── queue ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_queue_basic_push_pop() {
        let mut q = MetadataChannelQueue::new(4);
        q.push_raw(r#"<tally program="true" preview="false"/>"#.to_string(), 1);
        assert_eq!(q.len(), 1);
        let frame = q.pop().expect("should have a frame");
        assert!(frame.is_tally());
        assert!(q.is_empty());
    }

    #[test]
    fn test_queue_overflow_evicts_oldest() {
        let mut q = MetadataChannelQueue::new(2);
        q.push_raw("<a/>".to_string(), 1);
        q.push_raw("<b/>".to_string(), 2);
        q.push_raw("<c/>".to_string(), 3); // evicts <a/>
        assert_eq!(q.len(), 2);
        assert_eq!(q.stats().overflow_dropped, 1);
        // oldest remaining should be <b/>
        let first = q.pop().expect("frame");
        assert!(first.raw_xml.contains("<b/>") || first.raw_xml.contains('b'));
    }

    #[test]
    fn test_queue_drain_matching() {
        let mut q = MetadataChannelQueue::new(10);
        q.push_raw(r#"<tally program="true" preview="false"/>"#.to_string(), 1);
        q.push_raw(r#"<ptz pan="0.0" tilt="0.0" zoom="1.0"/>"#.to_string(), 2);
        q.push_raw(r#"<tally program="false" preview="true"/>"#.to_string(), 3);

        let tally_frames = q.drain_matching(|f| f.is_tally());
        assert_eq!(tally_frames.len(), 2);
        assert_eq!(q.len(), 1); // only ptz remains
    }

    #[test]
    fn test_queue_stats_tracking() {
        let mut q = MetadataChannelQueue::new(2);
        q.push_raw("<a/>".to_string(), 0);
        q.push_raw("<b/>".to_string(), 0);
        q.push_raw("<c/>".to_string(), 0); // overflow
        q.pop();

        let s = q.stats();
        assert_eq!(s.total_received, 3);
        assert_eq!(s.total_consumed, 1);
        assert_eq!(s.overflow_dropped, 1);
    }

    #[test]
    fn test_queue_peek_does_not_consume() {
        let mut q = MetadataChannelQueue::new(4);
        q.push_raw("<a/>".to_string(), 0);
        let _ = q.peek().expect("peek");
        assert_eq!(q.len(), 1);
        assert_eq!(q.stats().total_consumed, 0);
    }

    #[test]
    fn test_queue_clear() {
        let mut q = MetadataChannelQueue::new(4);
        q.push_raw("<a/>".to_string(), 0);
        q.push_raw("<b/>".to_string(), 0);
        q.clear();
        assert!(q.is_empty());
    }
}
