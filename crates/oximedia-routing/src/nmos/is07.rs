//! NMOS IS-07 Event & Tally API.
//!
//! IS-07 defines a mechanism for broadcasting state-change events from NMOS
//! sources.  Each event source has a unique identifier and an event type
//! (e.g., `"boolean"`, `"number"`, `"string"`).  When a state change occurs
//! the source emits an event payload encoded as a JSON string.
//!
//! # Example
//!
//! ```
//! use oximedia_routing::nmos::is07::Is07EventSource;
//!
//! let mut src = Is07EventSource::new("src-001", "boolean");
//! let json = src.emit("true");
//! assert!(json.contains("\"id\":\"src-001\""));
//! assert!(json.contains("\"event_type\":\"boolean\""));
//! assert!(json.contains("\"value\":\"true\""));
//! ```

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// An IS-07 event source that emits JSON event payloads.
///
/// The source tracks a monotonically-increasing sequence number so that
/// consumers can detect missed events.
#[derive(Debug, Clone)]
pub struct Is07EventSource {
    /// The unique identifier of this event source (typically a UUID).
    pub id: String,
    /// The IS-07 event type (e.g., `"boolean"`, `"number"`, `"string"`).
    pub event_type: String,
    /// Monotonically-increasing sequence counter.
    sequence: u64,
}

impl Is07EventSource {
    /// Create a new IS-07 event source.
    ///
    /// # Parameters
    ///
    /// * `id`          — unique identifier for the source.
    /// * `event_type`  — IS-07 event type string (e.g., `"boolean"`).
    pub fn new(id: impl Into<String>, event_type: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            event_type: event_type.into(),
            sequence: 0,
        }
    }

    /// Emit an IS-07 event with the given `value`.
    ///
    /// The sequence number is incremented on every call so that consumers can
    /// detect dropped events.
    ///
    /// # Returns
    ///
    /// A JSON string conforming to the IS-07 event payload schema:
    ///
    /// ```json
    /// {
    ///   "id": "<source-id>",
    ///   "event_type": "<type>",
    ///   "sequence": <u64>,
    ///   "value": "<value>"
    /// }
    /// ```
    pub fn emit(&mut self, value: &str) -> String {
        self.sequence += 1;
        let seq = self.sequence;
        let id = escape_json_string(&self.id);
        let et = escape_json_string(&self.event_type);
        let val = escape_json_string(value);
        format!(
            r#"{{"id":"{id}","event_type":"{et}","sequence":{seq},"value":"{val}"}}"#,
        )
    }

    /// Return the current sequence number (number of events emitted so far).
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Minimal JSON string escaper for the handful of characters that MUST be
/// escaped per RFC 8259 §7.
fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str(r#"\""#),
            '\\' => out.push_str(r"\\"),
            '\n' => out.push_str(r"\n"),
            '\r' => out.push_str(r"\r"),
            '\t' => out.push_str(r"\t"),
            c if (c as u32) < 0x20 => {
                // Control characters: emit as \uXXXX
                let _ = std::fmt::Write::write_fmt(
                    &mut out,
                    format_args!(r"\u{:04x}", c as u32),
                );
            }
            c => out.push(c),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_source_sequence_starts_at_zero() {
        let src = Is07EventSource::new("id-1", "boolean");
        assert_eq!(src.sequence(), 0);
        assert_eq!(src.id, "id-1");
        assert_eq!(src.event_type, "boolean");
    }

    #[test]
    fn test_emit_increments_sequence() {
        let mut src = Is07EventSource::new("id-2", "number");
        src.emit("42");
        assert_eq!(src.sequence(), 1);
        src.emit("43");
        assert_eq!(src.sequence(), 2);
    }

    #[test]
    fn test_emit_contains_id() {
        let mut src = Is07EventSource::new("source-abc", "boolean");
        let json = src.emit("false");
        assert!(json.contains("\"id\":\"source-abc\""), "json: {json}");
    }

    #[test]
    fn test_emit_contains_event_type() {
        let mut src = Is07EventSource::new("x", "string");
        let json = src.emit("hello");
        assert!(json.contains("\"event_type\":\"string\""), "json: {json}");
    }

    #[test]
    fn test_emit_contains_value() {
        let mut src = Is07EventSource::new("x", "number");
        let json = src.emit("3.14");
        assert!(json.contains("\"value\":\"3.14\""), "json: {json}");
    }

    #[test]
    fn test_emit_contains_sequence() {
        let mut src = Is07EventSource::new("x", "boolean");
        let json = src.emit("true");
        assert!(json.contains("\"sequence\":1"), "json: {json}");
    }

    #[test]
    fn test_emit_sequence_increases_monotonically() {
        let mut src = Is07EventSource::new("s", "boolean");
        for expected in 1u64..=5 {
            let json = src.emit("true");
            let needle = format!("\"sequence\":{expected}");
            assert!(json.contains(&needle), "expected seq {expected} in: {json}");
        }
    }

    #[test]
    fn test_emit_escapes_special_chars_in_value() {
        let mut src = Is07EventSource::new("s", "string");
        let json = src.emit("say \"hi\"");
        // The value should have its quotes escaped
        assert!(json.contains(r#"\"hi\""#), "json: {json}");
    }

    #[test]
    fn test_emit_is_valid_json_like_format() {
        let mut src = Is07EventSource::new("dev-1", "boolean");
        let json = src.emit("true");
        // Must start and end with braces
        assert!(json.starts_with('{'), "should start with {{");
        assert!(json.ends_with('}'), "should end with }}");
    }
}
