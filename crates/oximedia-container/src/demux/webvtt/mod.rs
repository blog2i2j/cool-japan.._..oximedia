//! `WebVTT` subtitle demuxer.
//!
//! This module provides a demuxer for `WebVTT` (Web Video Text Tracks)
//! subtitle format. `WebVTT` is a text-based format used primarily for
//! web video subtitles.
//!
//! # Format
//!
//! `WebVTT` files start with "WEBVTT" signature and contain cues with
//! timestamps and text content:
//!
//! ```text
//! WEBVTT
//!
//! 00:00:00.000 --> 00:00:02.000
//! This is the first subtitle
//!
//! 00:00:02.500 --> 00:00:05.000
//! This is the second subtitle
//! ```

use async_trait::async_trait;
use bytes::Bytes;
use oximedia_core::{CodecId, OxiError, OxiResult, Rational, Timestamp};
use oximedia_io::MediaSource;

use crate::demux::Demuxer;
use crate::{ContainerFormat, Metadata, Packet, PacketFlags, ProbeResult, StreamInfo};

/// `WebVTT` demuxer.
///
/// Parses `WebVTT` subtitle files and extracts subtitle cues as packets.
pub struct WebVttDemuxer<R> {
    /// The underlying reader.
    source: R,
    /// Stream information.
    stream_info: Option<StreamInfo>,
    /// Buffer for reading data.
    buffer: Vec<u8>,
    /// Current position in file.
    #[allow(dead_code)]
    position: usize,
    /// Whether we've reached EOF.
    eof: bool,
    /// Whether header has been parsed.
    header_parsed: bool,
    /// Parsed cues ready to be returned.
    cues: Vec<WebVttCue>,
    /// Current cue index.
    cue_index: usize,
}

impl<R> WebVttDemuxer<R> {
    /// Creates a new `WebVTT` demuxer.
    #[must_use]
    pub fn new(source: R) -> Self {
        Self {
            source,
            stream_info: None,
            buffer: Vec::new(),
            position: 0,
            eof: false,
            header_parsed: false,
            cues: Vec::new(),
            cue_index: 0,
        }
    }
}

impl<R: MediaSource> WebVttDemuxer<R> {
    /// Reads all data from the source.
    async fn read_all(&mut self) -> OxiResult<()> {
        loop {
            let mut temp = vec![0u8; 8192];
            let n = self.source.read(&mut temp).await?;
            if n == 0 {
                self.eof = true;
                break;
            }
            self.buffer.extend_from_slice(&temp[..n]);
        }
        Ok(())
    }

    /// Parses the `WebVTT` header and cues.
    async fn parse_file(&mut self) -> OxiResult<()> {
        // Read entire file
        self.read_all().await?;

        // Convert to string
        let content = String::from_utf8(self.buffer.clone()).map_err(|_| OxiError::Parse {
            offset: 0,
            message: "Invalid UTF-8 in WebVTT file".to_string(),
        })?;

        // Check signature
        if !content.starts_with("WEBVTT") {
            return Err(OxiError::Parse {
                offset: 0,
                message: "Missing WEBVTT signature".to_string(),
            });
        }

        // Parse cues
        self.cues = parse_webvtt_cues(&content)?;

        // Create stream info
        let timebase = Rational::new(1, 1000); // WebVTT uses milliseconds
        let mut stream = StreamInfo::new(0, CodecId::WebVtt, timebase);
        stream.metadata = Metadata::new().with_entry("format", "webvtt");
        self.stream_info = Some(stream);

        self.header_parsed = true;
        Ok(())
    }
}

#[async_trait]
impl<R: MediaSource> Demuxer for WebVttDemuxer<R> {
    async fn probe(&mut self) -> OxiResult<ProbeResult> {
        if !self.header_parsed {
            self.parse_file().await?;
        }

        Ok(ProbeResult::new(ContainerFormat::WebVtt, 1.0))
    }

    async fn read_packet(&mut self) -> OxiResult<Packet> {
        if !self.header_parsed {
            self.parse_file().await?;
        }

        if self.cue_index >= self.cues.len() {
            return Err(OxiError::Eof);
        }

        let cue = &self.cues[self.cue_index];
        self.cue_index += 1;

        // Create packet with cue text
        let data = Bytes::from(cue.text.clone().into_bytes());
        let timestamp = Timestamp::new(cue.start_ms, Rational::new(1, 1000));

        Ok(Packet::new(0, data, timestamp, PacketFlags::KEYFRAME))
    }

    fn streams(&self) -> &[StreamInfo] {
        self.stream_info.as_slice()
    }
}

/// A `WebVTT` cue.
#[derive(Debug, Clone)]
struct WebVttCue {
    /// Cue identifier (optional).
    #[allow(dead_code)]
    id: Option<String>,
    /// Start time in milliseconds.
    start_ms: i64,
    /// End time in milliseconds.
    #[allow(dead_code)]
    end_ms: i64,
    /// Cue text content.
    text: String,
    /// Cue settings (positioning, etc.).
    #[allow(dead_code)]
    settings: Option<String>,
}

/// Parses `WebVTT` cues from content.
fn parse_webvtt_cues(content: &str) -> OxiResult<Vec<WebVttCue>> {
    let mut cues = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;

    // Skip header
    while i < lines.len() {
        let line = lines[i].trim();
        if line.is_empty() {
            i += 1;
            break;
        }
        if !line.starts_with("WEBVTT") && !line.starts_with("NOTE") {
            break;
        }
        i += 1;
    }

    // Parse cues
    while i < lines.len() {
        // Skip empty lines
        while i < lines.len() && lines[i].trim().is_empty() {
            i += 1;
        }

        if i >= lines.len() {
            break;
        }

        // Skip NOTE blocks
        if lines[i].trim().starts_with("NOTE") {
            i += 1;
            while i < lines.len() && !lines[i].trim().is_empty() {
                i += 1;
            }
            continue;
        }

        // Try to parse cue
        let cue = parse_cue(&lines, &mut i)?;
        cues.push(cue);
    }

    Ok(cues)
}

/// Parses a single `WebVTT` cue.
fn parse_cue(lines: &[&str], index: &mut usize) -> OxiResult<WebVttCue> {
    let i = *index;

    if i >= lines.len() {
        return Err(OxiError::Parse {
            offset: i as u64,
            message: "Unexpected end of file".to_string(),
        });
    }

    // Check if this line contains a timestamp (cue timing line)
    let mut cue_id = None;
    let mut timing_line_idx = i;

    if !lines[i].contains("-->") {
        // This is a cue identifier
        cue_id = Some(lines[i].trim().to_string());
        timing_line_idx = i + 1;

        if timing_line_idx >= lines.len() {
            return Err(OxiError::Parse {
                offset: timing_line_idx as u64,
                message: "Missing timing line after cue identifier".to_string(),
            });
        }
    }

    // Parse timing line
    let timing_line = lines[timing_line_idx];
    let (start_ms, end_ms, settings) = parse_timing_line(timing_line)?;

    // Parse cue text (everything until next empty line)
    let mut text_lines = Vec::new();
    let mut text_idx = timing_line_idx + 1;

    while text_idx < lines.len() && !lines[text_idx].trim().is_empty() {
        text_lines.push(lines[text_idx]);
        text_idx += 1;
    }

    let text = text_lines.join("\n");

    *index = text_idx;

    Ok(WebVttCue {
        id: cue_id,
        start_ms,
        end_ms,
        text,
        settings,
    })
}

/// Parses a `WebVTT` timing line.
///
/// Format: `00:00:00.000 --> 00:00:02.000 [settings]`
fn parse_timing_line(line: &str) -> OxiResult<(i64, i64, Option<String>)> {
    let parts: Vec<&str> = line.split("-->").collect();
    if parts.len() < 2 {
        return Err(OxiError::Parse {
            offset: 0,
            message: format!("Invalid timing line: {line}"),
        });
    }

    let start_str = parts[0].trim();
    let end_and_settings = parts[1].trim();

    // Split end time and settings
    let end_parts: Vec<&str> = end_and_settings.split_whitespace().collect();
    let end_str = end_parts[0];
    let settings = if end_parts.len() > 1 {
        Some(end_parts[1..].join(" "))
    } else {
        None
    };

    let start_ms = parse_timestamp(start_str)?;
    let end_ms = parse_timestamp(end_str)?;

    Ok((start_ms, end_ms, settings))
}

/// Parses a `WebVTT` timestamp.
///
/// Formats supported:
/// - `HH:MM:SS.mmm`
/// - `MM:SS.mmm`
fn parse_timestamp(s: &str) -> OxiResult<i64> {
    let parts: Vec<&str> = s.split(':').collect();

    let (hours, minutes, seconds) = match parts.len() {
        2 => {
            // MM:SS.mmm
            (0, parts[0], parts[1])
        }
        3 => {
            // HH:MM:SS.mmm
            (
                parts[0].parse::<i64>().map_err(|_| OxiError::Parse {
                    offset: 0,
                    message: format!("Invalid hour in timestamp: {s}"),
                })?,
                parts[1],
                parts[2],
            )
        }
        _ => {
            return Err(OxiError::Parse {
                offset: 0,
                message: format!("Invalid timestamp format: {s}"),
            });
        }
    };

    let minutes = minutes.parse::<i64>().map_err(|_| OxiError::Parse {
        offset: 0,
        message: format!("Invalid minutes in timestamp: {s}"),
    })?;

    // Parse seconds and milliseconds
    let sec_parts: Vec<&str> = seconds.split('.').collect();
    if sec_parts.len() != 2 {
        return Err(OxiError::Parse {
            offset: 0,
            message: format!("Invalid seconds format in timestamp: {s}"),
        });
    }

    let secs = sec_parts[0].parse::<i64>().map_err(|_| OxiError::Parse {
        offset: 0,
        message: format!("Invalid seconds in timestamp: {s}"),
    })?;

    let millis = sec_parts[1].parse::<i64>().map_err(|_| OxiError::Parse {
        offset: 0,
        message: format!("Invalid milliseconds in timestamp: {s}"),
    })?;

    Ok(hours * 3600 * 1000 + minutes * 60 * 1000 + secs * 1000 + millis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_io::MemorySource;

    #[test]
    fn test_parse_timestamp_mm_ss() {
        assert_eq!(
            parse_timestamp("00:01.500").expect("operation should succeed"),
            1500
        );
        assert_eq!(
            parse_timestamp("01:30.000").expect("operation should succeed"),
            90000
        );
    }

    #[test]
    fn test_parse_timestamp_hh_mm_ss() {
        assert_eq!(
            parse_timestamp("00:00:01.500").expect("operation should succeed"),
            1500
        );
        assert_eq!(
            parse_timestamp("01:30:00.000").expect("operation should succeed"),
            5400000
        );
    }

    #[test]
    fn test_parse_timing_line() {
        let line = "00:00:01.000 --> 00:00:03.500";
        let (start, end, settings) = parse_timing_line(line).expect("operation should succeed");
        assert_eq!(start, 1000);
        assert_eq!(end, 3500);
        assert!(settings.is_none());
    }

    #[test]
    fn test_parse_timing_line_with_settings() {
        let line = "00:00:01.000 --> 00:00:03.500 align:start position:10%";
        let (start, end, settings) = parse_timing_line(line).expect("operation should succeed");
        assert_eq!(start, 1000);
        assert_eq!(end, 3500);
        assert_eq!(settings, Some("align:start position:10%".to_string()));
    }

    #[tokio::test]
    async fn test_webvtt_demuxer_probe() {
        let content = "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nHello World\n";
        let source = MemorySource::new(Bytes::from(content));
        let mut demuxer = WebVttDemuxer::new(source);

        let result = demuxer.probe().await.expect("probe should succeed");
        assert_eq!(result.format, ContainerFormat::WebVtt);
        assert_eq!(result.confidence, 1.0);
    }

    #[tokio::test]
    async fn test_webvtt_demuxer_read_packet() {
        let content = "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nHello\n\n00:00:03.000 --> 00:00:05.000\nWorld\n";
        let source = MemorySource::new(Bytes::from(content));
        let mut demuxer = WebVttDemuxer::new(source);

        demuxer.probe().await.expect("probe should succeed");

        let packet1 = demuxer
            .read_packet()
            .await
            .expect("operation should succeed");
        assert_eq!(packet1.stream_index, 0);
        assert_eq!(
            String::from_utf8(packet1.data.to_vec()).expect("operation should succeed"),
            "Hello"
        );

        let packet2 = demuxer
            .read_packet()
            .await
            .expect("operation should succeed");
        assert_eq!(
            String::from_utf8(packet2.data.to_vec()).expect("operation should succeed"),
            "World"
        );

        let result = demuxer.read_packet().await;
        assert!(matches!(result, Err(OxiError::Eof)));
    }
}
