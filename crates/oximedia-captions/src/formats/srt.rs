//! `SubRip` (SRT) format parser and writer

use crate::error::{CaptionError, Result};
use crate::formats::{FormatParser, FormatWriter};
use crate::types::{Caption, CaptionTrack, Language, Timestamp};
use nom::{
    bytes::complete::tag,
    character::complete::{char, digit1, space0},
    combinator::map_res,
    sequence::separated_pair,
    IResult, Parser,
};

/// SRT format parser
pub struct SrtParser;

impl FormatParser for SrtParser {
    fn parse(&self, data: &[u8]) -> Result<CaptionTrack> {
        let text = std::str::from_utf8(data).map_err(|e| CaptionError::Encoding(e.to_string()))?;

        parse_srt(text)
    }
}

/// SRT format writer
pub struct SrtWriter;

impl FormatWriter for SrtWriter {
    fn write(&self, track: &CaptionTrack) -> Result<Vec<u8>> {
        let mut output = String::new();

        for (index, caption) in track.captions.iter().enumerate() {
            // Caption number
            output.push_str(&format!("{}\n", index + 1));

            // Timestamp
            let start = format_timestamp(caption.start);
            let end = format_timestamp(caption.end);
            output.push_str(&format!("{start} --> {end}\n"));

            // Text
            output.push_str(&caption.text);
            output.push_str("\n\n");
        }

        Ok(output.into_bytes())
    }
}

fn parse_srt(text: &str) -> Result<CaptionTrack> {
    let mut track = CaptionTrack::new(Language::english());
    let blocks: Vec<&str> = text
        .split("\n\n")
        .filter(|s| !s.trim().is_empty())
        .collect();

    for block in blocks {
        if let Ok(caption) = parse_srt_block(block) {
            track.add_caption(caption)?;
        }
    }

    Ok(track)
}

fn parse_srt_block(block: &str) -> Result<Caption> {
    let lines: Vec<&str> = block.lines().collect();
    if lines.len() < 3 {
        return Err(CaptionError::Parse("Invalid SRT block".to_string()));
    }

    // Parse timestamp line
    let (start, end) = parse_timestamp_line(lines[1])?;

    // Combine remaining lines as text
    let text = lines[2..].join("\n");

    Ok(Caption::new(start, end, text))
}

fn parse_timestamp_line(line: &str) -> Result<(Timestamp, Timestamp)> {
    match srt_timestamp_line(line) {
        Ok((_, (start, end))) => Ok((start, end)),
        Err(e) => Err(CaptionError::Parse(format!("Invalid timestamp: {e}"))),
    }
}

fn srt_timestamp_line(input: &str) -> IResult<&str, (Timestamp, Timestamp)> {
    separated_pair(srt_timestamp, (space0, tag("-->"), space0), srt_timestamp).parse(input)
}

fn srt_timestamp(input: &str) -> IResult<&str, Timestamp> {
    let (input, hours) = map_res(digit1, |s: &str| s.parse::<u32>()).parse(input)?;
    let (input, _) = char(':').parse(input)?;
    let (input, minutes) = map_res(digit1, |s: &str| s.parse::<u32>()).parse(input)?;
    let (input, _) = char(':').parse(input)?;
    let (input, seconds) = map_res(digit1, |s: &str| s.parse::<u32>()).parse(input)?;
    let (input, _) = char(',').parse(input)?;
    let (input, millis) = map_res(digit1, |s: &str| s.parse::<u32>()).parse(input)?;

    Ok((input, Timestamp::from_hmsm(hours, minutes, seconds, millis)))
}

fn format_timestamp(ts: Timestamp) -> String {
    let (h, m, s, ms) = ts.as_hmsm();
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_srt() {
        let srt = b"1\n00:00:01,000 --> 00:00:03,000\nFirst caption\n\n2\n00:00:05,000 --> 00:00:07,500\nSecond caption\n\n";
        let parser = SrtParser;
        let track = parser.parse(srt).expect("parsing should succeed");

        assert_eq!(track.captions.len(), 2);
        assert_eq!(track.captions[0].text, "First caption");
        assert_eq!(track.captions[1].text, "Second caption");
    }

    #[test]
    fn test_write_srt() {
        let mut track = CaptionTrack::new(Language::english());
        track
            .add_caption(Caption::new(
                Timestamp::from_secs(1),
                Timestamp::from_secs(3),
                "Test caption".to_string(),
            ))
            .expect("operation should succeed in test");

        let writer = SrtWriter;
        let output = writer.write(&track).expect("writing should succeed");
        let text = String::from_utf8(output).expect("output should be valid UTF-8");

        assert!(text.contains("1\n"));
        assert!(text.contains("Test caption"));
        assert!(text.contains("-->"));
    }

    #[test]
    fn test_timestamp_parsing() {
        let (_, ts) = srt_timestamp("00:01:30,500").expect("timestamp parsing should succeed");
        assert_eq!(ts.as_hmsm(), (0, 1, 30, 500));
    }
}
