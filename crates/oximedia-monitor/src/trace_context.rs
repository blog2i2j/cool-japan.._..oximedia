//! W3C Trace Context propagation using `traceparent` and `tracestate` headers.
//!
//! Implements <https://www.w3.org/TR/trace-context/> with:
//!
//! - [`TraceParent`] — the `traceparent` header value (version, trace-id, parent-id, flags)
//! - [`TraceState`] — the `tracestate` header value (vendor key=value pairs)
//! - [`TraceSpan`] — a single span within a distributed trace
//!
//! IDs are generated via an LCG seeded from a caller-supplied `u64` so that
//! tests are deterministic without requiring any PRNG dependency.

#![allow(dead_code)]

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// LCG helper
// ---------------------------------------------------------------------------

/// One step of the Knuth multiplicative LCG.
fn lcg_step(state: u64) -> u64 {
    state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

// ---------------------------------------------------------------------------
// Hex helpers
// ---------------------------------------------------------------------------

/// Decode a single ASCII hex nibble.
fn nibble(b: u8) -> Result<u8, String> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        other => Err(format!("invalid hex character: {}", other as char)),
    }
}

/// Decode a hex string of exactly `N` bytes into a byte array.
fn decode_hex<const N: usize>(s: &str) -> Result<[u8; N], String> {
    if s.len() != N * 2 {
        return Err(format!(
            "expected {} hex chars, got {}",
            N * 2,
            s.len()
        ));
    }
    let mut out = [0u8; N];
    for (i, chunk) in s.as_bytes().chunks(2).enumerate() {
        let hi = nibble(chunk[0])?;
        let lo = nibble(chunk[1])?;
        out[i] = (hi << 4) | lo;
    }
    Ok(out)
}

/// Encode a byte slice as a lowercase hex string.
fn encode_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------
// TraceParent
// ---------------------------------------------------------------------------

/// W3C `traceparent` header: `version-traceid-parentid-flags`.
///
/// Format: `"00-{32 hex trace_id}-{16 hex parent_id}-{2 hex flags}"`
#[derive(Debug, Clone, PartialEq)]
pub struct TraceParent {
    /// Always `0x00` per W3C spec.
    pub version: u8,
    /// 128-bit trace identifier.
    pub trace_id: [u8; 16],
    /// 64-bit span (parent) identifier.
    pub parent_id: [u8; 8],
    /// Bit 0 = sampled.
    pub flags: u8,
}

impl TraceParent {
    /// Create a new root trace, generating IDs from `seed` via LCG.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let s1 = lcg_step(seed);
        let s2 = lcg_step(s1);
        let s3 = lcg_step(s2);

        let mut trace_id = [0u8; 16];
        trace_id[..8].copy_from_slice(&s1.to_le_bytes());
        trace_id[8..].copy_from_slice(&s2.to_le_bytes());

        let parent_id = s3.to_le_bytes();

        Self {
            version: 0x00,
            trace_id,
            parent_id,
            flags: 0x01,
        }
    }

    /// Returns `true` if the sampled bit is set.
    #[must_use]
    pub fn sampled(&self) -> bool {
        self.flags & 0x01 != 0
    }

    /// Return the trace ID as a 32-character lowercase hex string.
    #[must_use]
    pub fn trace_id_hex(&self) -> String {
        encode_hex(&self.trace_id)
    }

    /// Return the parent ID as a 16-character lowercase hex string.
    #[must_use]
    pub fn parent_id_hex(&self) -> String {
        encode_hex(&self.parent_id)
    }

    /// Parse a `traceparent` header string.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the format is invalid.
    pub fn parse(s: &str) -> Result<Self, String> {
        let parts: Vec<&str> = s.splitn(4, '-').collect();
        if parts.len() != 4 {
            return Err(format!(
                "traceparent must have 4 '-'-separated parts, got {}",
                parts.len()
            ));
        }

        // Version must be "00".
        if parts[0] != "00" {
            return Err(format!("unsupported traceparent version: {}", parts[0]));
        }

        let trace_id = decode_hex::<16>(parts[1])?;
        let parent_id = decode_hex::<8>(parts[2])?;

        if parts[3].len() != 2 {
            return Err(format!(
                "flags must be 2 hex chars, got {}",
                parts[3].len()
            ));
        }
        let fh = nibble(parts[3].as_bytes()[0])?;
        let fl = nibble(parts[3].as_bytes()[1])?;
        let flags = (fh << 4) | fl;

        Ok(Self {
            version: 0x00,
            trace_id,
            parent_id,
            flags,
        })
    }

    /// Serialize to `traceparent` header string.
    #[must_use]
    pub fn to_header(&self) -> String {
        format!(
            "00-{}-{}-{:02x}",
            self.trace_id_hex(),
            self.parent_id_hex(),
            self.flags
        )
    }

    /// Create a child span: same `trace_id`, new `parent_id` from `seed`.
    #[must_use]
    pub fn child_span(&self, seed: u64) -> Self {
        let s1 = lcg_step(seed);
        Self {
            version: 0x00,
            trace_id: self.trace_id,
            parent_id: s1.to_le_bytes(),
            flags: self.flags,
        }
    }
}

// ---------------------------------------------------------------------------
// TraceState
// ---------------------------------------------------------------------------

/// W3C `tracestate` header: ordered vendor key=value pairs (max 32 entries).
#[derive(Debug, Clone, Default)]
pub struct TraceState {
    entries: Vec<(String, String)>,
}

impl TraceState {
    /// Maximum list-members per W3C spec recommendation.
    pub const MAX_ENTRIES: usize = 32;

    /// Create an empty `TraceState`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a key=value entry (builder pattern, truncates at 32).
    #[must_use]
    pub fn add(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        if self.entries.len() < Self::MAX_ENTRIES {
            self.entries.push((key.into(), value.into()));
        }
        self
    }

    /// Look up a value by key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Parse from a `tracestate` header string.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any entry is malformed.
    pub fn parse(s: &str) -> Result<Self, String> {
        let mut entries = Vec::new();
        for part in s.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            match part.find('=') {
                None => return Err(format!("tracestate entry missing '=': {part}")),
                Some(eq) => {
                    let key = part[..eq].trim().to_string();
                    let val = part[eq + 1..].trim().to_string();
                    if key.is_empty() {
                        return Err("tracestate entry has empty key".to_string());
                    }
                    entries.push((key, val));
                    if entries.len() == Self::MAX_ENTRIES {
                        break;
                    }
                }
            }
        }
        Ok(Self { entries })
    }

    /// Serialize to `tracestate` header string.
    #[must_use]
    pub fn to_header(&self) -> String {
        self.entries
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if there are no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// TraceSpan
// ---------------------------------------------------------------------------

/// A single span within a distributed trace.
#[derive(Debug, Clone)]
pub struct TraceSpan {
    /// W3C trace context for this span.
    pub context: TraceParent,
    /// Human-readable operation name.
    pub operation_name: String,
    /// Millisecond timestamp when the span started.
    pub start_time_ms: u64,
    /// Millisecond timestamp when the span finished (`None` while in-flight).
    pub end_time_ms: Option<u64>,
    /// Arbitrary key-value attributes.
    pub attributes: HashMap<String, String>,
}

impl TraceSpan {
    /// Start a new root span.
    #[must_use]
    pub fn start(operation: impl Into<String>, seed: u64, start_ms: u64) -> Self {
        Self {
            context: TraceParent::new(seed),
            operation_name: operation.into(),
            start_time_ms: start_ms,
            end_time_ms: None,
            attributes: HashMap::new(),
        }
    }

    /// Start a child span whose trace-id matches `parent`.
    #[must_use]
    pub fn start_child(
        parent: &TraceParent,
        operation: impl Into<String>,
        seed: u64,
        start_ms: u64,
    ) -> Self {
        Self {
            context: parent.child_span(seed),
            operation_name: operation.into(),
            start_time_ms: start_ms,
            end_time_ms: None,
            attributes: HashMap::new(),
        }
    }

    /// Add a key-value attribute (builder pattern).
    #[must_use]
    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// Mark the span as finished.
    pub fn finish(&mut self, end_ms: u64) {
        self.end_time_ms = Some(end_ms);
    }

    /// Duration in milliseconds, available only after [`finish`](Self::finish).
    #[must_use]
    pub fn duration_ms(&self) -> Option<u64> {
        self.end_time_ms
            .map(|end| end.saturating_sub(self.start_time_ms))
    }

    /// Serialize to a minimal JSON string for logging.
    #[must_use]
    pub fn to_json(&self) -> String {
        let attrs: Vec<String> = self
            .attributes
            .iter()
            .map(|(k, v)| format!(r#""{k}":"{v}""#))
            .collect();
        let attrs_json = attrs.join(",");

        let duration_field = match self.duration_ms() {
            Some(d) => format!(",\"duration_ms\":{d}"),
            None => String::new(),
        };

        format!(
            r#"{{"operation":"{op}","trace_id":"{tid}","parent_id":"{pid}","start_ms":{start}{dur},"attributes":{{{attrs}}}}}"#,
            op = self.operation_name,
            tid = self.context.trace_id_hex(),
            pid = self.context.parent_id_hex(),
            start = self.start_time_ms,
            dur = duration_field,
            attrs = attrs_json,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_parent_new_deterministic() {
        let tp1 = TraceParent::new(42);
        let tp2 = TraceParent::new(42);
        assert_eq!(tp1, tp2);
    }

    #[test]
    fn test_trace_parent_sampled_flag() {
        let tp = TraceParent::new(1);
        assert!(tp.sampled());
        assert_eq!(tp.flags, 0x01);
    }

    #[test]
    fn test_trace_id_hex_length() {
        let tp = TraceParent::new(999);
        assert_eq!(tp.trace_id_hex().len(), 32);
    }

    #[test]
    fn test_parent_id_hex_length() {
        let tp = TraceParent::new(999);
        assert_eq!(tp.parent_id_hex().len(), 16);
    }

    #[test]
    fn test_to_header_format() {
        let tp = TraceParent::new(7);
        let hdr = tp.to_header();
        // "00-{32}-{16}-{2}"
        let parts: Vec<&str> = hdr.split('-').collect();
        assert_eq!(parts.len(), 4);
        assert_eq!(parts[0], "00");
        assert_eq!(parts[1].len(), 32);
        assert_eq!(parts[2].len(), 16);
        assert_eq!(parts[3].len(), 2);
    }

    #[test]
    fn test_parse_valid_header() {
        let tp = TraceParent::new(123);
        let hdr = tp.to_header();
        let parsed = TraceParent::parse(&hdr).expect("should parse");
        assert_eq!(parsed, tp);
    }

    #[test]
    fn test_parse_invalid_returns_err() {
        assert!(TraceParent::parse("not-valid").is_err());
        assert!(TraceParent::parse("01-abc-def-01").is_err()); // non-"00" version
        assert!(TraceParent::parse("00-too-short-01").is_err());
    }

    #[test]
    fn test_child_span_same_trace_id() {
        let parent = TraceParent::new(55);
        let child = parent.child_span(77);
        assert_eq!(parent.trace_id, child.trace_id);
        // Parent ID should differ (different seed step)
        assert_ne!(parent.parent_id, child.parent_id);
    }

    #[test]
    fn test_child_span_inherits_flags() {
        let parent = TraceParent::new(55);
        let child = parent.child_span(77);
        assert_eq!(parent.flags, child.flags);
    }

    #[test]
    fn test_trace_state_parse_serialize() {
        let ts = TraceState::parse("vendor1=val1,vendor2=val2").expect("ok");
        assert_eq!(ts.len(), 2);
        assert_eq!(ts.get("vendor1"), Some("val1"));
        assert_eq!(ts.get("vendor2"), Some("val2"));
        let hdr = ts.to_header();
        assert!(hdr.contains("vendor1=val1"));
        assert!(hdr.contains("vendor2=val2"));
    }

    #[test]
    fn test_trace_state_add_builder() {
        let ts = TraceState::new().add("a", "1").add("b", "2");
        assert_eq!(ts.len(), 2);
        assert_eq!(ts.get("a"), Some("1"));
    }

    #[test]
    fn test_trace_state_max_entries() {
        let pairs: Vec<String> = (0..40).map(|i| format!("k{i}=v{i}")).collect();
        let ts = TraceState::parse(&pairs.join(",")).expect("ok");
        assert_eq!(ts.len(), TraceState::MAX_ENTRIES);
    }

    #[test]
    fn test_trace_state_parse_invalid_entry() {
        assert!(TraceState::parse("noequals").is_err());
    }

    #[test]
    fn test_trace_span_finish_duration() {
        let mut span = TraceSpan::start("encode_frame", 11, 1_000);
        span.finish(1_250);
        assert_eq!(span.duration_ms(), Some(250));
    }

    #[test]
    fn test_trace_span_duration_none_before_finish() {
        let span = TraceSpan::start("encode_frame", 11, 1_000);
        assert_eq!(span.duration_ms(), None);
    }

    #[test]
    fn test_trace_span_to_json_contains_operation() {
        let mut span = TraceSpan::start("transcode", 22, 500)
            .with_attribute("codec", "av1");
        span.finish(900);
        let json = span.to_json();
        assert!(json.contains("transcode"), "JSON should contain operation name");
        assert!(json.contains("av1"), "JSON should contain attribute value");
        assert!(json.contains("duration_ms"), "JSON should contain duration");
    }

    #[test]
    fn test_trace_span_child_same_trace_id() {
        let parent = TraceParent::new(33);
        let child_span = TraceSpan::start_child(&parent, "child_op", 44, 100);
        assert_eq!(parent.trace_id, child_span.context.trace_id);
    }
}
