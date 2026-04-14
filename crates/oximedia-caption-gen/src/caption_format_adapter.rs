//! Caption format adapter: serialize [`CaptionBlock`] tracks to SRT, WebVTT,
//! and TTML output strings.
//!
//! ## Supported formats
//!
//! | Format | Standard       | Extension |
//! |--------|----------------|-----------|
//! | SRT    | SubRip Text    | `.srt`    |
//! | VTT    | WebVTT (W3C)   | `.vtt`    |
//! | TTML   | Timed Text ML  | `.ttml`   |
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_caption_gen::caption_format_adapter::{CaptionFormatAdapter, OutputFormat};
//! use oximedia_caption_gen::{CaptionBlock, CaptionPosition};
//!
//! let block = CaptionBlock {
//!     id: 1,
//!     start_ms: 0,
//!     end_ms: 2000,
//!     lines: vec!["Hello world".to_string()],
//!     speaker_id: None,
//!     position: CaptionPosition::Bottom,
//! };
//! let srt = CaptionFormatAdapter::convert(&[block], OutputFormat::Srt).unwrap();
//! assert!(srt.contains("00:00:00,000 --> 00:00:02,000"));
//! ```

use crate::alignment::CaptionBlock;

/// Error type for format adaptation failures.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum FormatAdapterError {
    /// The block list is empty and the format requires at least one entry.
    #[error("caption track is empty")]
    EmptyTrack,

    /// A timestamp value is invalid (start_ms >= end_ms).
    #[error("block {block_id}: invalid timestamp (start_ms={start_ms} >= end_ms={end_ms})")]
    InvalidTimestamp {
        block_id: u32,
        start_ms: u64,
        end_ms: u64,
    },
}

/// Target output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// SubRip Text (`.srt`).
    Srt,
    /// Web Video Text Tracks (`.vtt`).
    Vtt,
    /// Timed Text Markup Language (`.ttml`).
    Ttml,
}

/// Converts a caption track (slice of [`CaptionBlock`]) to various text-based
/// subtitle formats.
pub struct CaptionFormatAdapter;

impl CaptionFormatAdapter {
    /// Convert `blocks` to the requested `format`.
    ///
    /// # Errors
    /// Returns [`FormatAdapterError`] if any block has an invalid timestamp.
    pub fn convert(
        blocks: &[CaptionBlock],
        format: OutputFormat,
    ) -> Result<String, FormatAdapterError> {
        // Validate timestamps first.
        for block in blocks {
            if block.start_ms >= block.end_ms && !(block.start_ms == 0 && block.end_ms == 0) {
                return Err(FormatAdapterError::InvalidTimestamp {
                    block_id: block.id,
                    start_ms: block.start_ms,
                    end_ms: block.end_ms,
                });
            }
        }

        match format {
            OutputFormat::Srt => Ok(Self::to_srt(blocks)),
            OutputFormat::Vtt => Ok(Self::to_vtt(blocks)),
            OutputFormat::Ttml => Ok(Self::to_ttml(blocks)),
        }
    }

    // ─── SRT ──────────────────────────────────────────────────────────────────

    fn to_srt(blocks: &[CaptionBlock]) -> String {
        let mut out = String::new();
        for block in blocks {
            out.push_str(&block.id.to_string());
            out.push('\n');
            out.push_str(&format_srt_timestamp(block.start_ms));
            out.push_str(" --> ");
            out.push_str(&format_srt_timestamp(block.end_ms));
            out.push('\n');
            for line in &block.lines {
                out.push_str(line);
                out.push('\n');
            }
            out.push('\n');
        }
        out
    }

    // ─── VTT ──────────────────────────────────────────────────────────────────

    fn to_vtt(blocks: &[CaptionBlock]) -> String {
        let mut out = String::from("WEBVTT\n\n");
        for block in blocks {
            // VTT cue identifier (optional but helpful).
            out.push_str(&format!("cue-{}\n", block.id));
            out.push_str(&format_vtt_timestamp(block.start_ms));
            out.push_str(" --> ");
            out.push_str(&format_vtt_timestamp(block.end_ms));
            // Append position cue settings if not bottom-default.
            let pos_setting = vtt_position_setting(&block.position);
            if !pos_setting.is_empty() {
                out.push(' ');
                out.push_str(&pos_setting);
            }
            out.push('\n');
            for line in &block.lines {
                out.push_str(line);
                out.push('\n');
            }
            out.push('\n');
        }
        out
    }

    // ─── TTML ─────────────────────────────────────────────────────────────────

    fn to_ttml(blocks: &[CaptionBlock]) -> String {
        let mut out = String::new();
        out.push_str(concat!(
            r#"<?xml version="1.0" encoding="UTF-8"?>"#,
            "\n",
            r#"<tt xmlns="http://www.w3.org/ns/ttml" xml:lang="en">"#,
            "\n",
            "  <body>\n",
            "    <div>\n"
        ));

        for block in blocks {
            let begin = format_ttml_timestamp(block.start_ms);
            let end = format_ttml_timestamp(block.end_ms);
            out.push_str(&format!(
                "      <p begin=\"{}\" end=\"{}\" xml:id=\"s{}\">\n",
                begin, end, block.id
            ));
            for (i, line) in block.lines.iter().enumerate() {
                // Escape XML special characters.
                let escaped = xml_escape(line);
                if i + 1 < block.lines.len() {
                    out.push_str(&format!("        {}<br/>\n", escaped));
                } else {
                    out.push_str(&format!("        {}\n", escaped));
                }
            }
            out.push_str("      </p>\n");
        }

        out.push_str("    </div>\n");
        out.push_str("  </body>\n");
        out.push_str("</tt>\n");
        out
    }
}

// ─── Timestamp formatting helpers ─────────────────────────────────────────────

/// Format milliseconds as `HH:MM:SS,mmm` (SRT format).
fn format_srt_timestamp(ms: u64) -> String {
    let hours = ms / 3_600_000;
    let minutes = (ms % 3_600_000) / 60_000;
    let seconds = (ms % 60_000) / 1_000;
    let millis = ms % 1_000;
    format!("{:02}:{:02}:{:02},{:03}", hours, minutes, seconds, millis)
}

/// Format milliseconds as `HH:MM:SS.mmm` (VTT format).
fn format_vtt_timestamp(ms: u64) -> String {
    let hours = ms / 3_600_000;
    let minutes = (ms % 3_600_000) / 60_000;
    let seconds = (ms % 60_000) / 1_000;
    let millis = ms % 1_000;
    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis)
}

/// Format milliseconds as `HH:MM:SS.mmm` (TTML uses the same dot separator as VTT).
fn format_ttml_timestamp(ms: u64) -> String {
    format_vtt_timestamp(ms)
}

/// Generate VTT cue position settings for non-default caption positions.
fn vtt_position_setting(position: &crate::alignment::CaptionPosition) -> String {
    use crate::alignment::CaptionPosition;
    match position {
        CaptionPosition::Bottom => String::new(),
        CaptionPosition::Top => "line:10%".to_string(),
        CaptionPosition::Custom(x, y) => {
            format!("position:{:.0}% line:{:.0}%", x, y)
        }
    }
}

/// Escape XML special characters in a text string.
fn xml_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    for ch in text.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            other => out.push(other),
        }
    }
    out
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alignment::CaptionPosition;

    fn make_block(id: u32, start_ms: u64, end_ms: u64, text: &str) -> CaptionBlock {
        CaptionBlock {
            id,
            start_ms,
            end_ms,
            lines: vec![text.to_string()],
            speaker_id: None,
            position: CaptionPosition::Bottom,
        }
    }

    fn make_two_line_block(id: u32, start_ms: u64, end_ms: u64) -> CaptionBlock {
        CaptionBlock {
            id,
            start_ms,
            end_ms,
            lines: vec!["Line one".to_string(), "Line two".to_string()],
            speaker_id: None,
            position: CaptionPosition::Bottom,
        }
    }

    // ─── timestamp helpers ────────────────────────────────────────────────────

    #[test]
    fn srt_timestamp_zero() {
        assert_eq!(format_srt_timestamp(0), "00:00:00,000");
    }

    #[test]
    fn srt_timestamp_one_hour() {
        assert_eq!(format_srt_timestamp(3_600_000), "01:00:00,000");
    }

    #[test]
    fn srt_timestamp_mixed() {
        // 1h 2m 3s 456ms
        let ms = 3_600_000 + 2 * 60_000 + 3_000 + 456;
        assert_eq!(format_srt_timestamp(ms), "01:02:03,456");
    }

    #[test]
    fn vtt_timestamp_uses_dot_separator() {
        assert_eq!(format_vtt_timestamp(0), "00:00:00.000");
    }

    #[test]
    fn vtt_timestamp_mixed() {
        let ms = 3_600_000 + 2 * 60_000 + 3_000 + 456;
        assert_eq!(format_vtt_timestamp(ms), "01:02:03.456");
    }

    // ─── SRT output ───────────────────────────────────────────────────────────

    #[test]
    fn srt_single_block() {
        let block = make_block(1, 0, 2000, "Hello world");
        let srt = CaptionFormatAdapter::convert(&[block], OutputFormat::Srt)
            .expect("convert should succeed");
        assert!(srt.contains("1\n"));
        assert!(srt.contains("00:00:00,000 --> 00:00:02,000"));
        assert!(srt.contains("Hello world"));
    }

    #[test]
    fn srt_multiple_blocks() {
        let blocks = vec![
            make_block(1, 0, 2000, "Block one"),
            make_block(2, 2500, 4000, "Block two"),
        ];
        let srt = CaptionFormatAdapter::convert(&blocks, OutputFormat::Srt)
            .expect("convert should succeed");
        assert!(srt.contains("Block one"));
        assert!(srt.contains("Block two"));
        assert!(srt.contains("2\n"));
    }

    #[test]
    fn srt_two_line_block_emits_two_lines() {
        let block = make_two_line_block(1, 0, 2000);
        let srt = CaptionFormatAdapter::convert(&[block], OutputFormat::Srt)
            .expect("convert should succeed");
        assert!(srt.contains("Line one\n"));
        assert!(srt.contains("Line two\n"));
    }

    #[test]
    fn srt_blocks_separated_by_blank_line() {
        let blocks = vec![make_block(1, 0, 1000, "A"), make_block(2, 1500, 2500, "B")];
        let srt = CaptionFormatAdapter::convert(&blocks, OutputFormat::Srt)
            .expect("convert should succeed");
        // Each block ends with double newline.
        assert!(srt.contains("A\n\n"));
        assert!(srt.contains("B\n\n"));
    }

    #[test]
    fn srt_empty_track_produces_empty_string() {
        let srt =
            CaptionFormatAdapter::convert(&[], OutputFormat::Srt).expect("convert should succeed");
        assert!(srt.is_empty());
    }

    // ─── VTT output ───────────────────────────────────────────────────────────

    #[test]
    fn vtt_starts_with_webvtt_header() {
        let block = make_block(1, 0, 2000, "Hello");
        let vtt = CaptionFormatAdapter::convert(&[block], OutputFormat::Vtt)
            .expect("convert should succeed");
        assert!(vtt.starts_with("WEBVTT\n"));
    }

    #[test]
    fn vtt_uses_dot_separator_in_timestamps() {
        let block = make_block(1, 0, 2000, "Hello");
        let vtt = CaptionFormatAdapter::convert(&[block], OutputFormat::Vtt)
            .expect("convert should succeed");
        assert!(vtt.contains("00:00:00.000 --> 00:00:02.000"));
    }

    #[test]
    fn vtt_cue_identifier_present() {
        let block = make_block(3, 0, 1000, "Test");
        let vtt = CaptionFormatAdapter::convert(&[block], OutputFormat::Vtt)
            .expect("convert should succeed");
        assert!(vtt.contains("cue-3"));
    }

    #[test]
    fn vtt_top_position_adds_line_cue_setting() {
        let block = CaptionBlock {
            id: 1,
            start_ms: 0,
            end_ms: 1000,
            lines: vec!["Top caption".to_string()],
            speaker_id: None,
            position: CaptionPosition::Top,
        };
        let vtt = CaptionFormatAdapter::convert(&[block], OutputFormat::Vtt)
            .expect("convert should succeed");
        assert!(vtt.contains("line:10%"));
    }

    #[test]
    fn vtt_bottom_position_no_cue_setting() {
        let block = make_block(1, 0, 1000, "Bottom");
        let vtt = CaptionFormatAdapter::convert(&[block], OutputFormat::Vtt)
            .expect("convert should succeed");
        // No extra position setting on the timestamp line.
        let ts_line = vtt.lines().find(|l| l.contains("-->")).unwrap_or_default();
        assert!(!ts_line.contains("line:"));
    }

    // ─── TTML output ──────────────────────────────────────────────────────────

    #[test]
    fn ttml_has_xml_declaration() {
        let block = make_block(1, 0, 1000, "Hello");
        let ttml = CaptionFormatAdapter::convert(&[block], OutputFormat::Ttml)
            .expect("convert should succeed");
        assert!(ttml.contains(r#"<?xml version="1.0""#));
    }

    #[test]
    fn ttml_has_tt_element() {
        let block = make_block(1, 0, 1000, "Hello");
        let ttml = CaptionFormatAdapter::convert(&[block], OutputFormat::Ttml)
            .expect("convert should succeed");
        assert!(ttml.contains("<tt "));
        assert!(ttml.contains("</tt>"));
    }

    #[test]
    fn ttml_has_p_element_with_timestamps() {
        let block = make_block(1, 0, 2000, "Hello world");
        let ttml = CaptionFormatAdapter::convert(&[block], OutputFormat::Ttml)
            .expect("convert should succeed");
        assert!(ttml.contains(r#"begin="00:00:00.000""#));
        assert!(ttml.contains(r#"end="00:00:02.000""#));
    }

    #[test]
    fn ttml_multi_line_uses_br_element() {
        let block = make_two_line_block(1, 0, 2000);
        let ttml = CaptionFormatAdapter::convert(&[block], OutputFormat::Ttml)
            .expect("convert should succeed");
        assert!(ttml.contains("<br/>"));
    }

    #[test]
    fn ttml_escapes_xml_special_chars() {
        let block = make_block(1, 0, 2000, "A & B <test>");
        let ttml = CaptionFormatAdapter::convert(&[block], OutputFormat::Ttml)
            .expect("convert should succeed");
        assert!(ttml.contains("A &amp; B &lt;test&gt;"));
    }

    // ─── Error handling ───────────────────────────────────────────────────────

    #[test]
    fn invalid_timestamp_returns_error() {
        let block = CaptionBlock {
            id: 1,
            start_ms: 5000,
            end_ms: 1000, // end < start → invalid
            lines: vec!["bad".to_string()],
            speaker_id: None,
            position: CaptionPosition::Bottom,
        };
        let result = CaptionFormatAdapter::convert(&[block], OutputFormat::Srt);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            FormatAdapterError::InvalidTimestamp { .. }
        ));
    }

    // ─── xml_escape ───────────────────────────────────────────────────────────

    #[test]
    fn xml_escape_all_specials() {
        let result = xml_escape("&<>\"'");
        assert_eq!(result, "&amp;&lt;&gt;&quot;&apos;");
    }

    #[test]
    fn xml_escape_plain_text_unchanged() {
        assert_eq!(xml_escape("Hello world"), "Hello world");
    }

    // ─── Additional tests ─────────────────────────────────────────────────────

    #[test]
    fn srt_timestamp_sub_second() {
        // 500ms should be "00:00:00,500"
        assert_eq!(format_srt_timestamp(500), "00:00:00,500");
    }

    #[test]
    fn vtt_timestamp_sub_second() {
        assert_eq!(format_vtt_timestamp(500), "00:00:00.500");
    }

    #[test]
    fn srt_timestamp_large_hours() {
        // 10h exactly = 36_000_000ms
        assert_eq!(format_srt_timestamp(36_000_000), "10:00:00,000");
    }

    #[test]
    fn srt_output_sequence_numbers_sequential() {
        let blocks = vec![
            make_block(1, 0, 1000, "A"),
            make_block(2, 1000, 2000, "B"),
            make_block(3, 2000, 3000, "C"),
        ];
        let srt = CaptionFormatAdapter::convert(&blocks, OutputFormat::Srt).expect("convert");
        assert!(srt.contains("1\n"));
        assert!(srt.contains("2\n"));
        assert!(srt.contains("3\n"));
    }

    #[test]
    fn vtt_empty_track_has_webvtt_header_only() {
        let vtt = CaptionFormatAdapter::convert(&[], OutputFormat::Vtt).expect("convert");
        assert!(vtt.starts_with("WEBVTT\n"));
        // No cue entries.
        assert!(!vtt.contains("-->"));
    }

    #[test]
    fn ttml_empty_track_produces_valid_xml_skeleton() {
        let ttml = CaptionFormatAdapter::convert(&[], OutputFormat::Ttml).expect("convert");
        assert!(ttml.contains("<tt "));
        assert!(ttml.contains("</tt>"));
        assert!(ttml.contains("<body>"));
        assert!(ttml.contains("</body>"));
        // No <p> elements for an empty track.
        assert!(!ttml.contains("<p "));
    }

    #[test]
    fn custom_position_block_vtt_has_position_setting() {
        let block = CaptionBlock {
            id: 1,
            start_ms: 0,
            end_ms: 1000,
            lines: vec!["Custom".to_string()],
            speaker_id: None,
            position: CaptionPosition::Custom(50.0, 75.0),
        };
        let vtt = CaptionFormatAdapter::convert(&[block], OutputFormat::Vtt).expect("convert");
        assert!(vtt.contains("position:"));
        assert!(vtt.contains("line:"));
    }

    #[test]
    fn srt_unicode_text_preserved() {
        let block = make_block(1, 0, 2000, "日本語テキスト");
        let srt = CaptionFormatAdapter::convert(&[block], OutputFormat::Srt).expect("convert");
        assert!(srt.contains("日本語テキスト"));
    }

    #[test]
    fn ttml_block_xml_id_uses_block_id() {
        let block = make_block(7, 0, 1000, "Test");
        let ttml = CaptionFormatAdapter::convert(&[block], OutputFormat::Ttml).expect("convert");
        assert!(ttml.contains(r#"xml:id="s7""#));
    }

    #[test]
    fn output_format_variants_are_eq() {
        assert_eq!(OutputFormat::Srt, OutputFormat::Srt);
        assert_ne!(OutputFormat::Srt, OutputFormat::Vtt);
        assert_ne!(OutputFormat::Vtt, OutputFormat::Ttml);
    }

    #[test]
    fn invalid_timestamp_zero_zero_is_allowed() {
        // start_ms == end_ms == 0 is treated as a degenerate "empty" block,
        // not flagged as an error (the guard is start >= end AND not (0,0)).
        let block = CaptionBlock {
            id: 1,
            start_ms: 0,
            end_ms: 0,
            lines: vec!["zero".to_string()],
            speaker_id: None,
            position: CaptionPosition::Bottom,
        };
        let result = CaptionFormatAdapter::convert(&[block], OutputFormat::Srt);
        assert!(result.is_ok());
    }
}
