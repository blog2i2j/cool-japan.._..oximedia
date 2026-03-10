//! `WebVTT` format parser and writer

use crate::error::{CaptionError, Result};
use crate::formats::{FormatParser, FormatWriter};
use crate::types::{Alignment, Caption, CaptionTrack, Language, Timestamp, VerticalPosition};

/// `WebVTT` format parser
pub struct WebVttParser;

impl FormatParser for WebVttParser {
    fn parse(&self, data: &[u8]) -> Result<CaptionTrack> {
        let text = std::str::from_utf8(data).map_err(|e| CaptionError::Encoding(e.to_string()))?;

        parse_webvtt(text)
    }
}

/// `WebVTT` format writer
pub struct WebVttWriter;

impl FormatWriter for WebVttWriter {
    fn write(&self, track: &CaptionTrack) -> Result<Vec<u8>> {
        let mut output = String::new();

        // Header
        output.push_str("WEBVTT\n\n");

        for (index, caption) in track.captions.iter().enumerate() {
            // Optional cue identifier
            if let Some(speaker) = &caption.speaker {
                output.push_str(&format!("Cue {index} - {speaker}\n"));
            }

            // Timestamp with optional settings
            let start = format_timestamp(caption.start);
            let end = format_timestamp(caption.end);
            output.push_str(&format!("{start} --> {end}"));

            // Add positioning settings
            let settings = format_settings(caption);
            if !settings.is_empty() {
                output.push(' ');
                output.push_str(&settings);
            }
            output.push('\n');

            // Text (may contain formatting)
            output.push_str(&caption.text);
            output.push_str("\n\n");
        }

        Ok(output.into_bytes())
    }
}

fn parse_webvtt(text: &str) -> Result<CaptionTrack> {
    // Verify WebVTT header
    if !text.trim_start().starts_with("WEBVTT") {
        return Err(CaptionError::InvalidFormat(
            "Missing WEBVTT header".to_string(),
        ));
    }

    let mut track = CaptionTrack::new(Language::english());

    // Split into blocks
    let blocks: Vec<&str> = text
        .split("\n\n")
        .skip(1)
        .filter(|s| !s.trim().is_empty())
        .collect();

    for block in blocks {
        // Skip NOTE blocks
        if block.trim_start().starts_with("NOTE") {
            continue;
        }

        // Skip STYLE blocks
        if block.trim_start().starts_with("STYLE") {
            continue;
        }

        if let Ok(caption) = parse_webvtt_cue(block) {
            track.add_caption(caption)?;
        }
    }

    Ok(track)
}

fn parse_webvtt_cue(block: &str) -> Result<Caption> {
    let lines: Vec<&str> = block.lines().collect();
    if lines.is_empty() {
        return Err(CaptionError::Parse("Empty cue block".to_string()));
    }

    let mut line_idx = 0;

    // Optional cue identifier (if line doesn't contain -->)
    if !lines[line_idx].contains("-->") {
        line_idx += 1;
    }

    if line_idx >= lines.len() {
        return Err(CaptionError::Parse("No timestamp line".to_string()));
    }

    // Parse timestamp line
    let (start, end, settings) = parse_timestamp_line(lines[line_idx])?;
    line_idx += 1;

    // Remaining lines are text
    let text = lines[line_idx..].join("\n");

    let mut caption = Caption::new(start, end, text);

    // Apply settings
    apply_settings(&mut caption, &settings);

    Ok(caption)
}

fn parse_timestamp_line(line: &str) -> Result<(Timestamp, Timestamp, Vec<String>)> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 3 {
        return Err(CaptionError::Parse("Invalid timestamp line".to_string()));
    }

    let start = parse_vtt_timestamp(parts[0])?;
    // parts[1] should be "-->"
    let end = parse_vtt_timestamp(parts[2])?;

    // Remaining parts are settings
    let settings: Vec<String> = parts[3..].iter().map(|s| (*s).to_string()).collect();

    Ok((start, end, settings))
}

fn parse_vtt_timestamp(s: &str) -> Result<Timestamp> {
    let parts: Vec<&str> = s.split(':').collect();

    if parts.len() == 2 {
        // MM:SS.mmm format
        let minutes = parts[0]
            .parse::<u32>()
            .map_err(|e| CaptionError::Parse(e.to_string()))?;
        let sec_parts: Vec<&str> = parts[1].split('.').collect();
        let seconds = sec_parts[0]
            .parse::<u32>()
            .map_err(|e| CaptionError::Parse(e.to_string()))?;
        let millis = sec_parts
            .get(1)
            .unwrap_or(&"0")
            .parse::<u32>()
            .map_err(|e| CaptionError::Parse(e.to_string()))?;

        Ok(Timestamp::from_hmsm(0, minutes, seconds, millis))
    } else if parts.len() == 3 {
        // HH:MM:SS.mmm format
        let hours = parts[0]
            .parse::<u32>()
            .map_err(|e| CaptionError::Parse(e.to_string()))?;
        let minutes = parts[1]
            .parse::<u32>()
            .map_err(|e| CaptionError::Parse(e.to_string()))?;
        let sec_parts: Vec<&str> = parts[2].split('.').collect();
        let seconds = sec_parts[0]
            .parse::<u32>()
            .map_err(|e| CaptionError::Parse(e.to_string()))?;
        let millis = sec_parts
            .get(1)
            .unwrap_or(&"0")
            .parse::<u32>()
            .map_err(|e| CaptionError::Parse(e.to_string()))?;

        Ok(Timestamp::from_hmsm(hours, minutes, seconds, millis))
    } else {
        Err(CaptionError::Parse("Invalid timestamp format".to_string()))
    }
}

fn apply_settings(caption: &mut Caption, settings: &[String]) {
    for setting in settings {
        if let Some((key, value)) = setting.split_once(':') {
            match key {
                "align" => {
                    caption.style.alignment = match value {
                        "left" => Alignment::Left,
                        "center" | "middle" => Alignment::Center,
                        "right" => Alignment::Right,
                        _ => Alignment::Center,
                    };
                }
                "line" => {
                    if let Ok(line) = value.trim_end_matches('%').parse::<u8>() {
                        caption.position.line = Some(line);
                    }
                }
                "position" => {
                    if let Ok(pos) = value.trim_end_matches('%').parse::<f32>() {
                        caption.position.horizontal = pos / 100.0;
                    }
                }
                "vertical" => {
                    caption.position.vertical = match value {
                        "rl" => VerticalPosition::Top,
                        "lr" => VerticalPosition::Bottom,
                        _ => VerticalPosition::Bottom,
                    };
                }
                _ => {}
            }
        }
    }
}

fn format_timestamp(ts: Timestamp) -> String {
    let (h, m, s, ms) = ts.as_hmsm();
    if h > 0 {
        format!("{h:02}:{m:02}:{s:02}.{ms:03}")
    } else {
        format!("{m:02}:{s:02}.{ms:03}")
    }
}

fn format_settings(caption: &Caption) -> String {
    let mut settings = Vec::new();

    // Alignment
    let align = match caption.style.alignment {
        Alignment::Left => "left",
        Alignment::Center => "center",
        Alignment::Right => "right",
        Alignment::Justified => "center",
    };
    settings.push(format!("align:{align}"));

    // Position
    if caption.position.horizontal != 0.5 {
        let pos = (caption.position.horizontal * 100.0) as u8;
        settings.push(format!("position:{pos}%"));
    }

    settings.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_webvtt() {
        let vtt = b"WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nFirst caption\n\n00:05.000 --> 00:07.500\nSecond caption\n\n";
        let parser = WebVttParser;
        let track = parser.parse(vtt).expect("parsing should succeed");

        assert_eq!(track.captions.len(), 2);
        assert_eq!(track.captions[0].text, "First caption");
        assert_eq!(track.captions[1].text, "Second caption");
    }

    #[test]
    fn test_write_webvtt() {
        let mut track = CaptionTrack::new(Language::english());
        track
            .add_caption(Caption::new(
                Timestamp::from_secs(1),
                Timestamp::from_secs(3),
                "Test caption".to_string(),
            ))
            .expect("operation should succeed in test");

        let writer = WebVttWriter;
        let output = writer.write(&track).expect("writing should succeed");
        let text = String::from_utf8(output).expect("output should be valid UTF-8");

        assert!(text.starts_with("WEBVTT\n"));
        assert!(text.contains("Test caption"));
        assert!(text.contains("-->"));
    }

    #[test]
    fn test_webvtt_with_settings() {
        let vtt =
            b"WEBVTT\n\n00:00:01.000 --> 00:00:03.000 align:left position:20%\nLeft aligned\n\n";
        let parser = WebVttParser;
        let track = parser.parse(vtt).expect("parsing should succeed");

        assert_eq!(track.captions.len(), 1);
        assert_eq!(track.captions[0].style.alignment, Alignment::Left);
    }
}
