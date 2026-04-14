//! Subtitle stream muxing support for Matroska containers.
//!
//! Provides encoding of WebVTT and ASS/SSA subtitle tracks into the
//! Matroska BlockGroup format for embedding subtitles alongside video
//! and audio streams.
//!
//! # Supported Subtitle Formats
//!
//! - **WebVTT** (`S_TEXT/WEBVTT`) — W3C Web Video Text Tracks
//! - **SubStationAlpha** (`S_TEXT/ASS`) — Advanced SubStation Alpha
//! - **SubRip** (`S_TEXT/UTF8`) — simple timestamped text lines
//!
//! # Encoding
//!
//! Each subtitle cue is represented as a Matroska `SimpleBlock` (for fixed
//! duration) or `BlockGroup` (when the cue has an explicit duration).
//! The payload is the raw UTF-8 text of the cue (for WebVTT / SRT) or the
//! Dialogue line (for ASS).

#![forbid(unsafe_code)]

/// Matroska codec IDs for subtitle tracks.
pub mod codec_id {
    /// UTF-8 plain-text subtitles (SRT).
    pub const UTF8: &str = "S_TEXT/UTF8";
    /// WebVTT subtitles.
    pub const WEBVTT: &str = "S_TEXT/WEBVTT";
    /// Advanced SubStation Alpha.
    pub const ASS: &str = "S_TEXT/ASS";
    /// SubStation Alpha.
    pub const SSA: &str = "S_TEXT/SSA";
    /// HDMV PGS (Blu-ray bitmap subtitles, not supported for encode).
    pub const HDMV_PGS: &str = "S_HDMV/PGS";
    /// DVD bitmap subtitles, not supported for encode.
    pub const VOBSUB: &str = "S_VOBSUB";
}

/// A subtitle cue — the smallest unit of subtitle content.
#[derive(Debug, Clone, PartialEq)]
pub struct SubtitleCue {
    /// Start timestamp in milliseconds (relative to stream start).
    pub start_ms: i64,
    /// Duration in milliseconds (`None` = unknown, use until next cue).
    pub duration_ms: Option<u64>,
    /// UTF-8 text of the cue (or raw ASS Dialogue line).
    pub text: String,
    /// Optional cue identifier (WebVTT `id` or SRT sequence number as string).
    pub id: Option<String>,
}

impl SubtitleCue {
    /// Create a simple cue with no ID and known duration.
    #[must_use]
    pub fn new(start_ms: i64, duration_ms: u64, text: impl Into<String>) -> Self {
        Self {
            start_ms,
            duration_ms: Some(duration_ms),
            text: text.into(),
            id: None,
        }
    }

    /// Set the optional cue ID.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }
}

/// Parsed representation of a WebVTT file.
#[derive(Debug, Clone, Default)]
pub struct WebVttDocument {
    /// Header block (optional region/style definitions before first cue).
    pub header: Option<String>,
    /// Cues in order.
    pub cues: Vec<SubtitleCue>,
}

impl WebVttDocument {
    /// Parse a WebVTT document from a string.
    ///
    /// Returns a best-effort parse; malformed cues are silently skipped.
    #[must_use]
    pub fn parse(input: &str) -> Self {
        let mut doc = Self::default();

        // Must start with WEBVTT BOM/signature
        let trimmed = input.trim_start_matches('\u{FEFF}');
        if !trimmed.starts_with("WEBVTT") {
            return doc;
        }

        let mut lines = trimmed.lines().peekable();

        // Consume the WEBVTT header line
        lines.next();

        // Collect optional header block: the non-empty lines immediately after the
        // WEBVTT signature line, terminated by the first blank line.
        let mut header_lines: Vec<&str> = Vec::new();

        while let Some(line) = lines.peek().copied() {
            if line.trim().is_empty() {
                // End of header block; leave the blank line for `parse_vtt_cue` to skip.
                break;
            }
            if line.contains("-->") {
                // A timing line directly after the signature — no header block.
                break;
            }
            header_lines.push(line);
            lines.next();
        }

        let header_text = header_lines.join("\n").trim().to_owned();
        if !header_text.is_empty() {
            doc.header = Some(header_text);
        }

        // Parse cues
        while let Some(cue) = parse_vtt_cue(&mut lines) {
            doc.cues.push(cue);
        }

        doc
    }

    /// Serialize back to a WebVTT string.
    #[must_use]
    pub fn to_string(&self) -> String {
        let mut out = String::from("WEBVTT\n");
        if let Some(ref hdr) = self.header {
            out.push('\n');
            out.push_str(hdr);
            out.push('\n');
        }
        for cue in &self.cues {
            out.push('\n');
            if let Some(ref id) = cue.id {
                out.push_str(id);
                out.push('\n');
            }
            let dur = cue.duration_ms.unwrap_or(0);
            let end_ms = cue.start_ms as u64 + dur;
            out.push_str(&format_vtt_time(cue.start_ms as u64));
            out.push_str(" --> ");
            out.push_str(&format_vtt_time(end_ms));
            out.push('\n');
            out.push_str(&cue.text);
            out.push('\n');
        }
        out
    }
}

/// Parsed representation of an ASS (Advanced SubStation Alpha) document.
#[derive(Debug, Clone, Default)]
pub struct AssDocument {
    /// Script info section (key=value pairs).
    pub script_info: Vec<(String, String)>,
    /// Styles section.
    pub styles: Vec<AssStyle>,
    /// Events (subtitle lines).
    pub events: Vec<SubtitleCue>,
    /// Raw styles header for codec private data.
    pub styles_raw: String,
}

/// A single ASS style definition.
#[derive(Debug, Clone)]
pub struct AssStyle {
    /// Style name.
    pub name: String,
    /// Font name.
    pub fontname: String,
    /// Font size in points.
    pub fontsize: u32,
    /// Primary (fill) colour in ASS &HAABBGGRR format.
    pub primary_colour: String,
    /// Whether text is bold.
    pub bold: bool,
    /// Whether text is italic.
    pub italic: bool,
}

impl AssDocument {
    /// Parse a minimal ASS file from a string.
    #[must_use]
    pub fn parse(input: &str) -> Self {
        let mut doc = Self::default();
        let mut section = "";

        for line in input.lines() {
            let line = line.trim();
            if line.starts_with('[') && line.ends_with(']') {
                section = line;
                continue;
            }
            match section {
                "[Script Info]" => {
                    if let Some((k, v)) = line.split_once(':') {
                        doc.script_info
                            .push((k.trim().to_owned(), v.trim().to_owned()));
                    }
                }
                "[V4+ Styles]" | "[V4 Styles]" => {
                    doc.styles_raw.push_str(line);
                    doc.styles_raw.push('\n');
                    if let Some(stripped) = line.strip_prefix("Style: ") {
                        let parts: Vec<&str> = stripped.splitn(24, ',').collect();
                        if parts.len() >= 4 {
                            doc.styles.push(AssStyle {
                                name: parts[0].to_owned(),
                                fontname: parts[1].to_owned(),
                                fontsize: parts[2].parse().unwrap_or(20),
                                primary_colour: parts[3].to_owned(),
                                bold: parts.get(7).map_or(false, |&s| s == "-1"),
                                italic: parts.get(8).map_or(false, |&s| s == "-1"),
                            });
                        }
                    }
                }
                "[Events]" => {
                    if let Some(stripped) = line.strip_prefix("Dialogue: ") {
                        if let Some(cue) = parse_ass_dialogue(stripped) {
                            doc.events.push(cue);
                        }
                    }
                }
                _ => {}
            }
        }

        doc
    }

    /// Generate the codec private data for a Matroska ASS track header.
    ///
    /// The codec private data is the `[Script Info]` and `[V4+ Styles]` sections.
    #[must_use]
    pub fn codec_private(&self) -> String {
        let mut out = String::from("[Script Info]\n");
        for (k, v) in &self.script_info {
            out.push_str(k);
            out.push_str(": ");
            out.push_str(v);
            out.push('\n');
        }
        out.push_str("\n[V4+ Styles]\n");
        out.push_str(&self.styles_raw);
        out
    }
}

// ---------------------------------------------------------------------------
// Matroska subtitle track encoder
// ---------------------------------------------------------------------------

/// Subtitle track type for Matroska muxing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubtitleFormat {
    /// WebVTT (`S_TEXT/WEBVTT`).
    WebVtt,
    /// Advanced SubStation Alpha (`S_TEXT/ASS`).
    Ass,
    /// SubRip plain text (`S_TEXT/UTF8`).
    Utf8,
}

impl SubtitleFormat {
    /// Return the Matroska codec ID string.
    #[must_use]
    pub fn codec_id(self) -> &'static str {
        match self {
            Self::WebVtt => codec_id::WEBVTT,
            Self::Ass => codec_id::ASS,
            Self::Utf8 => codec_id::UTF8,
        }
    }
}

/// Encoded subtitle packet ready for writing into a Matroska cluster.
#[derive(Debug, Clone)]
pub struct SubtitlePacket {
    /// Timestamp in Matroska timecode units (milliseconds at default scale).
    pub timestamp_ms: i64,
    /// Duration in milliseconds.
    pub duration_ms: Option<u64>,
    /// Encoded payload bytes (UTF-8 text or ASS Dialogue line).
    pub payload: Vec<u8>,
    /// Whether this is a key frame (always true for subtitles).
    pub is_keyframe: bool,
}

/// Encoder that converts [`SubtitleCue`] slices into [`SubtitlePacket`] slices.
#[derive(Debug, Clone)]
pub struct SubtitleEncoder {
    format: SubtitleFormat,
}

impl SubtitleEncoder {
    /// Create a new subtitle encoder for the given format.
    #[must_use]
    pub fn new(format: SubtitleFormat) -> Self {
        Self { format }
    }

    /// Encode a single cue into a [`SubtitlePacket`].
    #[must_use]
    pub fn encode_cue(&self, cue: &SubtitleCue) -> SubtitlePacket {
        let payload = match self.format {
            SubtitleFormat::WebVtt => {
                // WebVTT: reconstruct the cue block (without the timing line)
                cue.text.as_bytes().to_vec()
            }
            SubtitleFormat::Ass => {
                // ASS: store as raw Dialogue line
                cue.text.as_bytes().to_vec()
            }
            SubtitleFormat::Utf8 => {
                // SRT: plain UTF-8 text
                cue.text.as_bytes().to_vec()
            }
        };

        SubtitlePacket {
            timestamp_ms: cue.start_ms,
            duration_ms: cue.duration_ms,
            payload,
            is_keyframe: true,
        }
    }

    /// Encode a slice of cues into a vector of packets.
    #[must_use]
    pub fn encode_all(&self, cues: &[SubtitleCue]) -> Vec<SubtitlePacket> {
        cues.iter().map(|c| self.encode_cue(c)).collect()
    }

    /// Return the Matroska codec ID for this encoder.
    #[must_use]
    pub fn codec_id(&self) -> &'static str {
        self.format.codec_id()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a single WebVTT cue from a line iterator.
fn parse_vtt_cue<'a>(
    lines: &mut std::iter::Peekable<impl Iterator<Item = &'a str>>,
) -> Option<SubtitleCue> {
    // Skip blank lines
    while let Some(&line) = lines.peek() {
        if line.trim().is_empty() {
            lines.next();
        } else {
            break;
        }
    }

    // Optional cue ID (line that does NOT contain `-->`)
    let mut cue_id: Option<String> = None;
    if let Some(&line) = lines.peek() {
        if !line.contains("-->") && !line.is_empty() {
            cue_id = Some(line.to_owned());
            lines.next();
        }
    }

    // Timing line
    let timing_line = lines.next()?;
    let (start_ms, duration_ms) = parse_vtt_timing(timing_line)?;

    // Text lines until blank
    let mut text_lines: Vec<&str> = Vec::new();
    while let Some(&line) = lines.peek() {
        if line.trim().is_empty() {
            break;
        }
        text_lines.push(line);
        lines.next();
    }

    if text_lines.is_empty() {
        return None;
    }

    Some(SubtitleCue {
        start_ms,
        duration_ms: Some(duration_ms),
        text: text_lines.join("\n"),
        id: cue_id,
    })
}

/// Parse a `HH:MM:SS.mmm --> HH:MM:SS.mmm` timing line, returning
/// `(start_ms, duration_ms)`.
fn parse_vtt_timing(line: &str) -> Option<(i64, u64)> {
    let parts: Vec<&str> = line.split("-->").collect();
    if parts.len() < 2 {
        return None;
    }
    let start = parse_vtt_timestamp(parts[0].trim())?;
    let end = parse_vtt_timestamp(parts[1].split_whitespace().next().unwrap_or(""))?;
    Some((start as i64, end.saturating_sub(start)))
}

/// Parse `HH:MM:SS.mmm` or `MM:SS.mmm` into milliseconds.
fn parse_vtt_timestamp(s: &str) -> Option<u64> {
    let parts: Vec<&str> = s.splitn(3, ':').collect();
    let (h, m, sec_ms) = match parts.len() {
        2 => (0u64, parts[0].parse::<u64>().ok()?, parts[1]),
        3 => (
            parts[0].parse::<u64>().ok()?,
            parts[1].parse::<u64>().ok()?,
            parts[2],
        ),
        _ => return None,
    };
    let (sec_str, ms_str) = sec_ms.split_once('.')?;
    let secs: u64 = sec_str.parse().ok()?;
    let ms: u64 = ms_str.parse().ok()?;
    Some(h * 3_600_000 + m * 60_000 + secs * 1000 + ms)
}

/// Format milliseconds as `HH:MM:SS.mmm`.
fn format_vtt_time(ms: u64) -> String {
    let h = ms / 3_600_000;
    let m = (ms % 3_600_000) / 60_000;
    let s = (ms % 60_000) / 1_000;
    let millis = ms % 1_000;
    format!("{h:02}:{m:02}:{s:02}.{millis:03}")
}

/// Parse an ASS Dialogue line (content after `Dialogue: `).
///
/// ASS Dialogue format (V4+):
/// `Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text`
fn parse_ass_dialogue(line: &str) -> Option<SubtitleCue> {
    let parts: Vec<&str> = line.splitn(10, ',').collect();
    if parts.len() < 10 {
        return None;
    }
    let start = parse_ass_time(parts[1].trim())?;
    let end = parse_ass_time(parts[2].trim())?;
    let text = parts[9].to_owned();
    Some(SubtitleCue {
        start_ms: start as i64,
        duration_ms: Some(end.saturating_sub(start)),
        text,
        id: None,
    })
}

/// Parse an ASS timestamp `H:MM:SS.cc` into milliseconds (cc = centiseconds).
fn parse_ass_time(s: &str) -> Option<u64> {
    let parts: Vec<&str> = s.splitn(3, ':').collect();
    if parts.len() != 3 {
        return None;
    }
    let h: u64 = parts[0].parse().ok()?;
    let m: u64 = parts[1].parse().ok()?;
    let (sec_str, cs_str) = parts[2].split_once('.')?;
    let secs: u64 = sec_str.parse().ok()?;
    let cs: u64 = cs_str.parse().ok()?;
    Some(h * 3_600_000 + m * 60_000 + secs * 1_000 + cs * 10)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webvtt_parse_simple() {
        let vtt = "WEBVTT\n\n1\n00:00:01.000 --> 00:00:03.000\nHello world\n";
        let doc = WebVttDocument::parse(vtt);
        assert_eq!(doc.cues.len(), 1);
        assert_eq!(doc.cues[0].start_ms, 1000);
        assert_eq!(doc.cues[0].duration_ms, Some(2000));
        assert_eq!(doc.cues[0].text, "Hello world");
        assert_eq!(doc.cues[0].id, Some("1".to_owned()));
    }

    #[test]
    fn test_webvtt_parse_no_id() {
        let vtt = "WEBVTT\n\n00:00:02.500 --> 00:00:05.000\nSubtitle line\n";
        let doc = WebVttDocument::parse(vtt);
        assert_eq!(doc.cues.len(), 1);
        assert_eq!(doc.cues[0].start_ms, 2500);
        assert!(doc.cues[0].id.is_none());
    }

    #[test]
    fn test_webvtt_roundtrip() {
        let original = "WEBVTT\n\n1\n00:00:01.000 --> 00:00:03.000\nHello\n";
        let doc = WebVttDocument::parse(original);
        let serialized = doc.to_string();
        let reparsed = WebVttDocument::parse(&serialized);
        assert_eq!(reparsed.cues.len(), 1);
        assert_eq!(reparsed.cues[0].text, "Hello");
    }

    #[test]
    fn test_ass_parse_dialogue() {
        let ass = "[Script Info]\nTitle: Test\n\n[V4+ Styles]\n\n[Events]\nDialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello ASS\n";
        let doc = AssDocument::parse(ass);
        assert_eq!(doc.events.len(), 1);
        assert_eq!(doc.events[0].start_ms, 1000);
        assert_eq!(doc.events[0].duration_ms, Some(2000));
        assert!(doc.events[0].text.contains("Hello ASS"));
    }

    #[test]
    fn test_subtitle_encoder_webvtt() {
        let encoder = SubtitleEncoder::new(SubtitleFormat::WebVtt);
        let cue = SubtitleCue::new(1000, 2000, "Test subtitle");
        let pkt = encoder.encode_cue(&cue);
        assert_eq!(pkt.timestamp_ms, 1000);
        assert_eq!(pkt.duration_ms, Some(2000));
        assert_eq!(pkt.payload, b"Test subtitle");
        assert!(pkt.is_keyframe);
    }

    #[test]
    fn test_subtitle_encoder_ass() {
        let encoder = SubtitleEncoder::new(SubtitleFormat::Ass);
        assert_eq!(encoder.codec_id(), "S_TEXT/ASS");
    }

    #[test]
    fn test_subtitle_encoder_utf8() {
        let encoder = SubtitleEncoder::new(SubtitleFormat::Utf8);
        assert_eq!(encoder.codec_id(), "S_TEXT/UTF8");
    }

    #[test]
    fn test_encode_all() {
        let encoder = SubtitleEncoder::new(SubtitleFormat::WebVtt);
        let cues = vec![
            SubtitleCue::new(0, 1000, "First"),
            SubtitleCue::new(2000, 1500, "Second"),
        ];
        let packets = encoder.encode_all(&cues);
        assert_eq!(packets.len(), 2);
        assert_eq!(packets[0].timestamp_ms, 0);
        assert_eq!(packets[1].timestamp_ms, 2000);
    }

    #[test]
    fn test_subtitle_cue_with_id() {
        let cue = SubtitleCue::new(500, 1000, "Line").with_id("42");
        assert_eq!(cue.id, Some("42".to_owned()));
    }

    #[test]
    fn test_format_vtt_time() {
        assert_eq!(format_vtt_time(61_500), "00:01:01.500");
        assert_eq!(format_vtt_time(3_661_001), "01:01:01.001");
    }

    #[test]
    fn test_parse_vtt_timestamp_mm_ss() {
        let ms = parse_vtt_timestamp("01:02.500");
        assert_eq!(ms, Some(62_500));
    }

    #[test]
    fn test_parse_ass_time() {
        let ms = parse_ass_time("0:01:30.00");
        assert_eq!(ms, Some(90_000));
    }

    #[test]
    fn test_webvtt_invalid() {
        let bad = "NOT WEBVTT\nsome garbage";
        let doc = WebVttDocument::parse(bad);
        assert!(doc.cues.is_empty());
    }

    #[test]
    fn test_subtitle_format_codec_ids() {
        assert_eq!(SubtitleFormat::WebVtt.codec_id(), "S_TEXT/WEBVTT");
        assert_eq!(SubtitleFormat::Ass.codec_id(), "S_TEXT/ASS");
        assert_eq!(SubtitleFormat::Utf8.codec_id(), "S_TEXT/UTF8");
    }
}
