//! CMX 3600 EDL format parser and serializer.
//!
//! The CMX 3600 format is the most widely used EDL format in professional
//! video post-production. This module provides a dedicated parser and
//! serializer for the CMX 3600 dialect.

#![allow(dead_code)]

/// A single event (edit) in a CMX 3600 EDL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CmxEvent {
    /// Event number (001–999).
    pub event_num: u32,
    /// Reel (tape) name, e.g. "A001" or "BL" for black.
    pub reel: String,
    /// Track designator, e.g. "V", "A", "A2", "AA", etc.
    pub track: String,
    /// Transition type: "C" (cut), "D" (dissolve), "W" (wipe), "K" (key).
    pub transition: String,
    /// Source timecode in point, formatted as HH:MM:SS:FF.
    pub source_in: String,
    /// Source timecode out point, formatted as HH:MM:SS:FF.
    pub source_out: String,
    /// Record timecode in point, formatted as HH:MM:SS:FF.
    pub record_in: String,
    /// Record timecode out point, formatted as HH:MM:SS:FF.
    pub record_out: String,
}

impl CmxEvent {
    /// Returns `true` if this event uses a dissolve transition.
    #[must_use]
    pub fn is_dissolve(&self) -> bool {
        self.transition.starts_with('D')
    }

    /// Returns `true` if this event is a straight cut.
    #[must_use]
    pub fn is_cut(&self) -> bool {
        self.transition == "C"
    }
}

/// A parsed CMX 3600 EDL document.
#[derive(Debug, Clone)]
pub struct CmxEdl {
    /// Optional title from the "TITLE:" header line.
    pub title: Option<String>,
    /// Frame rate indicated in the file (e.g. 25.0, 29.97).
    pub frame_rate: f32,
    /// All edit events in order.
    pub events: Vec<CmxEvent>,
}

impl CmxEdl {
    /// Create a new, empty CMX EDL.
    #[must_use]
    pub fn new(frame_rate: f32) -> Self {
        Self {
            title: None,
            frame_rate,
            events: Vec::new(),
        }
    }

    /// Total number of events in the EDL.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Returns a deduplicated, sorted list of reel names used by events.
    #[must_use]
    pub fn reels_used(&self) -> Vec<&str> {
        let mut reels: Vec<&str> = self.events.iter().map(|e| e.reel.as_str()).collect();
        reels.sort_unstable();
        reels.dedup();
        reels
    }
}

/// Parse a CMX 3600 timecode string `HH:MM:SS:FF` (or drop-frame `HH:MM:SS;FF`).
///
/// Returns `Some((hours, minutes, seconds, frames))` on success, `None` on failure.
#[must_use]
pub fn parse_cmx_timecode(s: &str) -> Option<(u8, u8, u8, u8)> {
    // Accept either ':' or ';' as the last separator (drop-frame)
    let s = s.trim();
    if s.len() < 11 {
        return None;
    }
    let parts: Vec<&str> = s.splitn(4, [':', ';']).collect();
    if parts.len() != 4 {
        return None;
    }
    let h: u8 = parts[0].parse().ok()?;
    let m: u8 = parts[1].parse().ok()?;
    let sec: u8 = parts[2].parse().ok()?;
    let f: u8 = parts[3].parse().ok()?;

    if m > 59 || sec > 59 {
        return None;
    }
    Some((h, m, sec, f))
}

/// Serialize a `CmxEdl` to a CMX 3600 text representation.
#[must_use]
pub fn serialize_cmx(edl: &CmxEdl) -> String {
    let mut out = String::new();
    if let Some(ref title) = edl.title {
        out.push_str(&format!("TITLE: {title}\n"));
    }
    out.push_str("FCM: NON-DROP FRAME\n\n");

    for ev in &edl.events {
        let line = format!(
            "{:03}  {:<8} {:<5} {:<8} {} {} {} {}\n",
            ev.event_num,
            ev.reel,
            ev.track,
            ev.transition,
            ev.source_in,
            ev.source_out,
            ev.record_in,
            ev.record_out,
        );
        out.push_str(&line);
    }
    out
}

/// Parse a CMX 3600 text EDL into a `CmxEdl`.
///
/// # Errors
///
/// Returns a descriptive `String` error if the input is malformed.
pub fn parse_cmx(input: &str) -> Result<CmxEdl, String> {
    let mut title: Option<String> = None;
    let mut frame_rate: f32 = 29.97;
    let mut events: Vec<CmxEvent> = Vec::new();

    for raw_line in input.lines() {
        let line = raw_line.trim();

        if line.is_empty() || line.starts_with('*') {
            continue;
        }

        if let Some(rest) = line.strip_prefix("TITLE:") {
            title = Some(rest.trim().to_string());
            continue;
        }

        if line.starts_with("FCM:") {
            let fcm = line.to_uppercase();
            frame_rate = if fcm.contains("DROP") { 29.97 } else { 29.97 };
            // Real implementations would distinguish 25/24/30 from comment or context.
            let _ = fcm; // consumed above
            continue;
        }

        // Try to parse an event line: NNN  REEL  TRACK  TRANS  SRC_IN SRC_OUT REC_IN REC_OUT
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() < 8 {
            continue;
        }

        let event_num: u32 = match cols[0].parse() {
            Ok(n) => n,
            Err(_) => continue,
        };

        // Validate that columns 4-7 look like timecodes
        let tc_cols = [cols[4], cols[5], cols[6], cols[7]];
        for tc in &tc_cols {
            if parse_cmx_timecode(tc).is_none() {
                return Err(format!("Invalid timecode '{}' in event {event_num}", tc));
            }
        }

        events.push(CmxEvent {
            event_num,
            reel: cols[1].to_string(),
            track: cols[2].to_string(),
            transition: cols[3].to_string(),
            source_in: cols[4].to_string(),
            source_out: cols[5].to_string(),
            record_in: cols[6].to_string(),
            record_out: cols[7].to_string(),
        });
    }

    Ok(CmxEdl {
        title,
        frame_rate,
        events,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- parse_cmx_timecode ---

    #[test]
    fn test_parse_timecode_basic() {
        let result = parse_cmx_timecode("01:02:03:04").expect("operation should succeed");
        assert_eq!(result, (1, 2, 3, 4));
    }

    #[test]
    fn test_parse_timecode_drop_frame_semicolon() {
        let result = parse_cmx_timecode("01:00:00;00").expect("operation should succeed");
        assert_eq!(result, (1, 0, 0, 0));
    }

    #[test]
    fn test_parse_timecode_zero() {
        let result = parse_cmx_timecode("00:00:00:00").expect("operation should succeed");
        assert_eq!(result, (0, 0, 0, 0));
    }

    #[test]
    fn test_parse_timecode_max_values() {
        let result = parse_cmx_timecode("23:59:59:29").expect("operation should succeed");
        assert_eq!(result, (23, 59, 59, 29));
    }

    #[test]
    fn test_parse_timecode_invalid_minutes() {
        assert!(parse_cmx_timecode("01:60:00:00").is_none());
    }

    #[test]
    fn test_parse_timecode_too_short() {
        assert!(parse_cmx_timecode("01:02").is_none());
    }

    // --- CmxEvent ---

    #[test]
    fn test_is_cut() {
        let ev = CmxEvent {
            event_num: 1,
            reel: "A001".to_string(),
            track: "V".to_string(),
            transition: "C".to_string(),
            source_in: "01:00:00:00".to_string(),
            source_out: "01:00:05:00".to_string(),
            record_in: "01:00:00:00".to_string(),
            record_out: "01:00:05:00".to_string(),
        };
        assert!(ev.is_cut());
        assert!(!ev.is_dissolve());
    }

    #[test]
    fn test_is_dissolve() {
        let ev = CmxEvent {
            event_num: 2,
            reel: "A002".to_string(),
            track: "V".to_string(),
            transition: "D 025".to_string(),
            source_in: "01:00:05:00".to_string(),
            source_out: "01:00:10:00".to_string(),
            record_in: "01:00:05:00".to_string(),
            record_out: "01:00:10:00".to_string(),
        };
        assert!(ev.is_dissolve());
        assert!(!ev.is_cut());
    }

    // --- CmxEdl ---

    #[test]
    fn test_event_count() {
        let edl = CmxEdl {
            title: None,
            frame_rate: 25.0,
            events: vec![
                CmxEvent {
                    event_num: 1,
                    reel: "A001".to_string(),
                    track: "V".to_string(),
                    transition: "C".to_string(),
                    source_in: "01:00:00:00".to_string(),
                    source_out: "01:00:05:00".to_string(),
                    record_in: "01:00:00:00".to_string(),
                    record_out: "01:00:05:00".to_string(),
                },
                CmxEvent {
                    event_num: 2,
                    reel: "A002".to_string(),
                    track: "V".to_string(),
                    transition: "C".to_string(),
                    source_in: "01:00:00:00".to_string(),
                    source_out: "01:00:05:00".to_string(),
                    record_in: "01:00:05:00".to_string(),
                    record_out: "01:00:10:00".to_string(),
                },
            ],
        };
        assert_eq!(edl.event_count(), 2);
    }

    #[test]
    fn test_reels_used_deduplication() {
        let make_event = |num: u32, reel: &str| CmxEvent {
            event_num: num,
            reel: reel.to_string(),
            track: "V".to_string(),
            transition: "C".to_string(),
            source_in: "01:00:00:00".to_string(),
            source_out: "01:00:05:00".to_string(),
            record_in: "01:00:00:00".to_string(),
            record_out: "01:00:05:00".to_string(),
        };
        let edl = CmxEdl {
            title: None,
            frame_rate: 25.0,
            events: vec![
                make_event(1, "A001"),
                make_event(2, "A002"),
                make_event(3, "A001"),
            ],
        };
        let reels = edl.reels_used();
        assert_eq!(reels.len(), 2);
        assert!(reels.contains(&"A001"));
        assert!(reels.contains(&"A002"));
    }

    // --- serialize / parse roundtrip ---

    #[test]
    fn test_serialize_contains_title() {
        let edl = CmxEdl {
            title: Some("My Cut".to_string()),
            frame_rate: 25.0,
            events: vec![],
        };
        let s = serialize_cmx(&edl);
        assert!(s.contains("TITLE: My Cut"));
    }

    #[test]
    fn test_serialize_event_line() {
        let edl = CmxEdl {
            title: None,
            frame_rate: 25.0,
            events: vec![CmxEvent {
                event_num: 1,
                reel: "A001".to_string(),
                track: "V".to_string(),
                transition: "C".to_string(),
                source_in: "01:00:00:00".to_string(),
                source_out: "01:00:05:00".to_string(),
                record_in: "01:00:00:00".to_string(),
                record_out: "01:00:05:00".to_string(),
            }],
        };
        let s = serialize_cmx(&edl);
        assert!(s.contains("001"));
        assert!(s.contains("A001"));
    }

    #[test]
    fn test_parse_cmx_roundtrip() {
        let input = "TITLE: RoundTrip\nFCM: NON-DROP FRAME\n\n\
            001  A001     V     C        01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n\
            002  A002     V     C        01:00:05:00 01:00:10:00 01:00:05:00 01:00:10:00\n";
        let edl = parse_cmx(input).expect("operation should succeed");
        assert_eq!(edl.title.as_deref(), Some("RoundTrip"));
        assert_eq!(edl.event_count(), 2);
        let back = serialize_cmx(&edl);
        let edl2 = parse_cmx(&back).expect("operation should succeed");
        assert_eq!(edl2.event_count(), 2);
    }

    #[test]
    fn test_parse_cmx_skips_comments() {
        let input =
            "001  AX       V     C        00:00:00:00 00:00:01:00 00:00:00:00 00:00:01:00\n\
            * FROM CLIP NAME: shot.mov\n";
        let edl = parse_cmx(input).expect("operation should succeed");
        assert_eq!(edl.event_count(), 1);
    }

    #[test]
    fn test_parse_cmx_invalid_timecode_returns_error() {
        let input = "001  AX       V     C        BADTC 00:00:01:00 00:00:00:00 00:00:01:00\n";
        assert!(parse_cmx(input).is_err());
    }

    #[test]
    fn test_new_edl_is_empty() {
        let edl = CmxEdl::new(25.0);
        assert_eq!(edl.event_count(), 0);
        assert!(edl.title.is_none());
        assert_eq!(edl.reels_used().len(), 0);
    }
}
