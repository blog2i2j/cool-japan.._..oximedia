//! STL (EBU-STL and Spruce STL) format parsers and writers

use crate::error::{CaptionError, Result};
use crate::formats::{FormatParser, FormatWriter};
use crate::types::{Caption, CaptionTrack, Language, Timestamp};

/// EBU-STL format parser (binary format)
pub struct EbuStlParser;

impl FormatParser for EbuStlParser {
    fn parse(&self, _data: &[u8]) -> Result<CaptionTrack> {
        // EBU-STL is a complex binary format
        // This is a simplified implementation
        Err(CaptionError::UnsupportedFormat(
            "EBU-STL parsing not yet fully implemented".to_string(),
        ))
    }
}

/// EBU-STL format writer
pub struct EbuStlWriter;

impl FormatWriter for EbuStlWriter {
    fn write(&self, _track: &CaptionTrack) -> Result<Vec<u8>> {
        Err(CaptionError::UnsupportedFormat(
            "EBU-STL writing not yet fully implemented".to_string(),
        ))
    }
}

/// Spruce STL format parser (text-based)
pub struct SpruceStlParser;

impl FormatParser for SpruceStlParser {
    fn parse(&self, data: &[u8]) -> Result<CaptionTrack> {
        let text = std::str::from_utf8(data).map_err(|e| CaptionError::Encoding(e.to_string()))?;

        parse_spruce_stl(text)
    }
}

/// Spruce STL format writer
pub struct SpruceStlWriter;

impl FormatWriter for SpruceStlWriter {
    fn write(&self, track: &CaptionTrack) -> Result<Vec<u8>> {
        let mut output = String::new();

        // Spruce STL header
        output.push_str("//Font select and font size\n");
        output.push_str("$FontName       = Arial\n");
        output.push_str("$FontSize       = 32\n\n");

        for (index, caption) in track.captions.iter().enumerate() {
            output.push_str(&format!("{}\n", index + 1));
            let start = format_spruce_timecode(caption.start);
            let end = format_spruce_timecode(caption.end);
            output.push_str(&format!("{start} , {end}\n"));
            output.push_str(&caption.text);
            output.push_str("\n\n");
        }

        Ok(output.into_bytes())
    }
}

fn parse_spruce_stl(text: &str) -> Result<CaptionTrack> {
    let mut track = CaptionTrack::new(Language::english());
    let blocks: Vec<&str> = text
        .split("\n\n")
        .filter(|s| !s.trim().is_empty())
        .collect();

    for block in blocks {
        if block.starts_with("//") || block.starts_with('$') {
            continue; // Skip comments and headers
        }

        if let Ok(caption) = parse_spruce_block(block) {
            track.add_caption(caption)?;
        }
    }

    Ok(track)
}

fn parse_spruce_block(block: &str) -> Result<Caption> {
    let lines: Vec<&str> = block.lines().collect();
    if lines.len() < 3 {
        return Err(CaptionError::Parse("Invalid Spruce STL block".to_string()));
    }

    // Line 0: subtitle number (skip)
    // Line 1: timecode
    let (start, end) = parse_spruce_timecode_line(lines[1])?;
    // Remaining: text
    let text = lines[2..].join("\n");

    Ok(Caption::new(start, end, text))
}

fn parse_spruce_timecode_line(line: &str) -> Result<(Timestamp, Timestamp)> {
    let parts: Vec<&str> = line.split(',').map(str::trim).collect();
    if parts.len() != 2 {
        return Err(CaptionError::Parse(
            "Invalid Spruce timecode line".to_string(),
        ));
    }

    let start = parse_spruce_timecode(parts[0])?;
    let end = parse_spruce_timecode(parts[1])?;

    Ok((start, end))
}

fn parse_spruce_timecode(s: &str) -> Result<Timestamp> {
    // Format: HH:MM:SS:FF
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 4 {
        return Err(CaptionError::Parse(format!("Invalid Spruce timecode: {s}")));
    }

    let hours = parts[0]
        .parse::<u32>()
        .map_err(|e| CaptionError::Parse(e.to_string()))?;
    let minutes = parts[1]
        .parse::<u32>()
        .map_err(|e| CaptionError::Parse(e.to_string()))?;
    let seconds = parts[2]
        .parse::<u32>()
        .map_err(|e| CaptionError::Parse(e.to_string()))?;
    let frames = parts[3]
        .parse::<u32>()
        .map_err(|e| CaptionError::Parse(e.to_string()))?;

    let frame_ms = frames * 1000 / 25; // Assume 25 fps (PAL)

    Ok(Timestamp::from_hmsm(hours, minutes, seconds, frame_ms))
}

fn format_spruce_timecode(ts: Timestamp) -> String {
    let (h, m, s, ms) = ts.as_hmsm();
    let frames = ms * 25 / 1000;
    format!("{h:02}:{m:02}:{s:02}:{frames:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spruce_timecode() {
        let ts = parse_spruce_timecode("01:30:45:12").expect("timestamp parsing should succeed");
        let (h, m, s, _) = ts.as_hmsm();
        assert_eq!((h, m, s), (1, 30, 45));
    }
}
