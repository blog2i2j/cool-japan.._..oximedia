//! `SubRip` (SRT) subtitle demuxer.
//!
//! This module provides a demuxer for `SubRip` (SRT) subtitle format.
//! SRT is a simple, widely-supported text-based subtitle format.
//!
//! # Format
//!
//! SRT files contain numbered subtitle entries with timestamps:
//!
//! ```text
//! 1
//! 00:00:00,000 --> 00:00:02,000
//! This is the first subtitle
//!
//! 2
//! 00:00:02,500 --> 00:00:05,000
//! This is the second subtitle
//! ```

use async_trait::async_trait;
use bytes::Bytes;
use oximedia_core::{CodecId, OxiError, OxiResult, Rational, Timestamp};
use oximedia_io::MediaSource;

use crate::demux::Demuxer;
use crate::{ContainerFormat, Metadata, Packet, PacketFlags, ProbeResult, StreamInfo};

/// `SubRip` (SRT) demuxer.
///
/// Parses SRT subtitle files and extracts subtitle entries as packets.
pub struct SrtDemuxer<R> {
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
    /// Parsed entries ready to be returned.
    entries: Vec<SrtEntry>,
    /// Current entry index.
    entry_index: usize,
}

impl<R> SrtDemuxer<R> {
    /// Creates a new SRT demuxer.
    #[must_use]
    pub fn new(source: R) -> Self {
        Self {
            source,
            stream_info: None,
            buffer: Vec::new(),
            position: 0,
            eof: false,
            header_parsed: false,
            entries: Vec::new(),
            entry_index: 0,
        }
    }
}

impl<R: MediaSource> SrtDemuxer<R> {
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

    /// Parses the SRT file and entries.
    async fn parse_file(&mut self) -> OxiResult<()> {
        // Read entire file
        self.read_all().await?;

        // Convert to string
        let content = String::from_utf8(self.buffer.clone()).map_err(|_| OxiError::Parse {
            offset: 0,
            message: "Invalid UTF-8 in SRT file".to_string(),
        })?;

        // Parse entries
        self.entries = parse_srt_entries(&content)?;

        // Create stream info
        let timebase = Rational::new(1, 1000); // SRT uses milliseconds
        let mut stream = StreamInfo::new(0, CodecId::Srt, timebase);
        stream.metadata = Metadata::new().with_entry("format", "srt");
        self.stream_info = Some(stream);

        self.header_parsed = true;
        Ok(())
    }
}

#[async_trait]
impl<R: MediaSource> Demuxer for SrtDemuxer<R> {
    async fn probe(&mut self) -> OxiResult<ProbeResult> {
        if !self.header_parsed {
            self.parse_file().await?;
        }

        Ok(ProbeResult::new(ContainerFormat::Srt, 1.0))
    }

    async fn read_packet(&mut self) -> OxiResult<Packet> {
        if !self.header_parsed {
            self.parse_file().await?;
        }

        if self.entry_index >= self.entries.len() {
            return Err(OxiError::Eof);
        }

        let entry = &self.entries[self.entry_index];
        self.entry_index += 1;

        // Create packet with entry text
        let data = Bytes::from(entry.text.clone().into_bytes());
        let timestamp = Timestamp::new(entry.start_ms, Rational::new(1, 1000));

        Ok(Packet::new(0, data, timestamp, PacketFlags::KEYFRAME))
    }

    fn streams(&self) -> &[StreamInfo] {
        self.stream_info.as_slice()
    }
}

/// An SRT subtitle entry.
#[derive(Debug, Clone)]
struct SrtEntry {
    /// Entry number (1-based).
    #[allow(dead_code)]
    number: u32,
    /// Start time in milliseconds.
    start_ms: i64,
    /// End time in milliseconds.
    #[allow(dead_code)]
    end_ms: i64,
    /// Entry text content.
    text: String,
}

/// Parses SRT entries from content.
fn parse_srt_entries(content: &str) -> OxiResult<Vec<SrtEntry>> {
    let mut entries = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        // Skip empty lines
        while i < lines.len() && lines[i].trim().is_empty() {
            i += 1;
        }

        if i >= lines.len() {
            break;
        }

        // Parse entry
        let entry = parse_entry(&lines, &mut i)?;
        entries.push(entry);
    }

    Ok(entries)
}

/// Parses a single SRT entry.
fn parse_entry(lines: &[&str], index: &mut usize) -> OxiResult<SrtEntry> {
    let i = *index;

    if i >= lines.len() {
        return Err(OxiError::Parse {
            offset: i as u64,
            message: "Unexpected end of file".to_string(),
        });
    }

    // Parse entry number
    let number_str = lines[i].trim();
    let number = number_str.parse::<u32>().map_err(|_| OxiError::Parse {
        offset: i as u64,
        message: format!("Invalid entry number: {number_str}"),
    })?;

    // Parse timing line
    if i + 1 >= lines.len() {
        return Err(OxiError::Parse {
            offset: (i + 1) as u64,
            message: "Missing timing line".to_string(),
        });
    }

    let timing_line = lines[i + 1];
    let (start_ms, end_ms) = parse_timing_line(timing_line)?;

    // Parse text (everything until next empty line or entry number)
    let mut text_lines = Vec::new();
    let mut text_idx = i + 2;

    while text_idx < lines.len() {
        let line = lines[text_idx].trim();

        // Stop at empty line
        if line.is_empty() {
            break;
        }

        // Stop if this looks like the next entry number
        if line.parse::<u32>().is_ok()
            && text_idx + 1 < lines.len()
            && lines[text_idx + 1].contains("-->")
        {
            break;
        }

        text_lines.push(lines[text_idx]);
        text_idx += 1;
    }

    let text = text_lines.join("\n");

    *index = text_idx;

    Ok(SrtEntry {
        number,
        start_ms,
        end_ms,
        text,
    })
}

/// Parses an SRT timing line.
///
/// Format: `00:00:00,000 --> 00:00:02,000`
fn parse_timing_line(line: &str) -> OxiResult<(i64, i64)> {
    let parts: Vec<&str> = line.split("-->").collect();
    if parts.len() != 2 {
        return Err(OxiError::Parse {
            offset: 0,
            message: format!("Invalid timing line: {line}"),
        });
    }

    let start_str = parts[0].trim();
    let end_str = parts[1].trim();

    let start_ms = parse_srt_timestamp(start_str)?;
    let end_ms = parse_srt_timestamp(end_str)?;

    Ok((start_ms, end_ms))
}

/// Parses an SRT timestamp.
///
/// Format: `HH:MM:SS,mmm`
fn parse_srt_timestamp(s: &str) -> OxiResult<i64> {
    let parts: Vec<&str> = s.split(':').collect();

    if parts.len() != 3 {
        return Err(OxiError::Parse {
            offset: 0,
            message: format!("Invalid timestamp format: {s}"),
        });
    }

    let hours = parts[0].parse::<i64>().map_err(|_| OxiError::Parse {
        offset: 0,
        message: format!("Invalid hours in timestamp: {s}"),
    })?;

    let minutes = parts[1].parse::<i64>().map_err(|_| OxiError::Parse {
        offset: 0,
        message: format!("Invalid minutes in timestamp: {s}"),
    })?;

    // Parse seconds and milliseconds (separated by comma)
    let sec_parts: Vec<&str> = parts[2].split(',').collect();
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
    fn test_parse_srt_timestamp() {
        assert_eq!(
            parse_srt_timestamp("00:00:01,500").expect("operation should succeed"),
            1500
        );
        assert_eq!(
            parse_srt_timestamp("01:30:00,000").expect("operation should succeed"),
            5400000
        );
        assert_eq!(
            parse_srt_timestamp("00:02:15,750").expect("operation should succeed"),
            135750
        );
    }

    #[test]
    fn test_parse_timing_line() {
        let line = "00:00:01,000 --> 00:00:03,500";
        let (start, end) = parse_timing_line(line).expect("operation should succeed");
        assert_eq!(start, 1000);
        assert_eq!(end, 3500);
    }

    #[tokio::test]
    async fn test_srt_demuxer_probe() {
        let content = "1\n00:00:00,000 --> 00:00:02,000\nHello World\n";
        let source = MemorySource::new(Bytes::from(content));
        let mut demuxer = SrtDemuxer::new(source);

        let result = demuxer.probe().await.expect("probe should succeed");
        assert_eq!(result.format, ContainerFormat::Srt);
        assert_eq!(result.confidence, 1.0);
    }

    #[tokio::test]
    async fn test_srt_demuxer_read_packet() {
        let content =
            "1\n00:00:00,000 --> 00:00:02,000\nHello\n\n2\n00:00:03,000 --> 00:00:05,000\nWorld\n";
        let source = MemorySource::new(Bytes::from(content));
        let mut demuxer = SrtDemuxer::new(source);

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

    #[tokio::test]
    async fn test_srt_multiline_text() {
        let content = "1\n00:00:00,000 --> 00:00:02,000\nLine 1\nLine 2\nLine 3\n";
        let source = MemorySource::new(Bytes::from(content));
        let mut demuxer = SrtDemuxer::new(source);

        demuxer.probe().await.expect("probe should succeed");

        let packet = demuxer
            .read_packet()
            .await
            .expect("operation should succeed");
        let text = String::from_utf8(packet.data.to_vec()).expect("operation should succeed");
        assert_eq!(text, "Line 1\nLine 2\nLine 3");
    }
}
