//! Scenarist Closed Caption (SCC) format parser and writer

use crate::error::{CaptionError, Result};
use crate::formats::{FormatParser, FormatWriter};
use crate::types::{Caption, CaptionTrack, Language, Timestamp};

/// SCC format parser
pub struct SccParser;

impl FormatParser for SccParser {
    fn parse(&self, data: &[u8]) -> Result<CaptionTrack> {
        let text = std::str::from_utf8(data).map_err(|e| CaptionError::Encoding(e.to_string()))?;

        parse_scc(text)
    }
}

/// SCC format writer
pub struct SccWriter;

impl FormatWriter for SccWriter {
    fn write(&self, track: &CaptionTrack) -> Result<Vec<u8>> {
        let mut output = String::new();

        // SCC header
        output.push_str("Scenarist_SCC V1.0\n\n");

        for caption in &track.captions {
            let timecode = format_scc_timecode(caption.start);
            // SCC hex codes (simplified - full implementation would encode text to CEA-608)
            output.push_str(&format!("{timecode}\t9420 9420\n\n"));
        }

        Ok(output.into_bytes())
    }
}

fn parse_scc(text: &str) -> Result<CaptionTrack> {
    let mut track = CaptionTrack::new(Language::english());

    // Verify header
    if !text.trim_start().starts_with("Scenarist_SCC") {
        return Err(CaptionError::InvalidFormat(
            "Missing SCC header".to_string(),
        ));
    }

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("Scenarist_SCC") {
            continue;
        }

        // Parse timecode and data
        if let Some((timecode, _data)) = trimmed.split_once('\t') {
            if let Ok(timestamp) = parse_scc_timecode(timecode) {
                // Simplified - would need full CEA-608 decoder
                let caption = Caption::new(timestamp, timestamp, String::new());
                track.add_caption(caption)?;
            }
        }
    }

    Ok(track)
}

fn parse_scc_timecode(s: &str) -> Result<Timestamp> {
    // Format: HH:MM:SS:FF (hours:minutes:seconds:frames)
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 4 {
        return Err(CaptionError::Parse(format!("Invalid SCC timecode: {s}")));
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

    // Assume 29.97 fps (NTSC)
    let frame_ms = (f64::from(frames) * 1000.0 / 29.97) as u32;

    Ok(Timestamp::from_hmsm(hours, minutes, seconds, frame_ms))
}

fn format_scc_timecode(ts: Timestamp) -> String {
    let (h, m, s, ms) = ts.as_hmsm();
    let frames = ((f64::from(ms) / 1000.0) * 29.97) as u32;
    format!("{h:02}:{m:02}:{s:02}:{frames:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_scc_timecode() {
        let ts = parse_scc_timecode("01:30:45:15").expect("timestamp parsing should succeed");
        let (h, m, s, _) = ts.as_hmsm();
        assert_eq!((h, m, s), (1, 30, 45));
    }
}
