//! TTML (Timed Text Markup Language) format parser and writer

use crate::error::{CaptionError, Result};
use crate::formats::{FormatParser, FormatWriter};
use crate::types::{Caption, CaptionTrack, Language, Timestamp};
use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};
use quick_xml::{Reader, Writer};
use std::io::Cursor;

/// TTML format parser
pub struct TtmlParser;

impl FormatParser for TtmlParser {
    fn parse(&self, data: &[u8]) -> Result<CaptionTrack> {
        parse_ttml(data)
    }
}

/// TTML format writer
pub struct TtmlWriter;

impl FormatWriter for TtmlWriter {
    fn write(&self, track: &CaptionTrack) -> Result<Vec<u8>> {
        write_ttml(track)
    }
}

fn parse_ttml(data: &[u8]) -> Result<CaptionTrack> {
    let mut reader = Reader::from_reader(data);
    // trim_text is not available in quick-xml 0.36+

    let mut track = CaptionTrack::new(Language::english());
    let mut buf = Vec::new();
    let mut in_body = false;
    let mut current_text = String::new();
    let mut current_begin: Option<Timestamp> = None;
    let mut current_end: Option<Timestamp> = None;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                match e.name().as_ref() {
                    b"body" => in_body = true,
                    b"p" | b"div" => {
                        // Parse timing attributes
                        for attr in e.attributes().flatten() {
                            match attr.key.as_ref() {
                                b"begin" => {
                                    if let Ok(val) = std::str::from_utf8(&attr.value) {
                                        current_begin = parse_ttml_time(val).ok();
                                    }
                                }
                                b"end" => {
                                    if let Ok(val) = std::str::from_utf8(&attr.value) {
                                        current_end = parse_ttml_time(val).ok();
                                    }
                                }
                                _ => {}
                            }
                        }
                        current_text.clear();
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(e)) => {
                if in_body {
                    let text = String::from_utf8_lossy(e.as_ref());
                    current_text.push_str(&text);
                }
            }
            Ok(Event::End(ref e)) => {
                if matches!(e.name().as_ref(), b"p" | b"div") {
                    if let (Some(begin), Some(end)) = (current_begin, current_end) {
                        let caption = Caption::new(begin, end, current_text.clone());
                        track.add_caption(caption)?;
                        current_text.clear();
                        current_begin = None;
                        current_end = None;
                    }
                } else if e.name().as_ref() == b"body" {
                    in_body = false;
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(CaptionError::Xml(e.to_string())),
            _ => {}
        }
        buf.clear();
    }

    Ok(track)
}

fn write_ttml(track: &CaptionTrack) -> Result<Vec<u8>> {
    let mut writer = Writer::new(Cursor::new(Vec::new()));

    // XML declaration
    writer
        .write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))
        .map_err(|e| CaptionError::Xml(e.to_string()))?;

    // Root element
    let mut tt = BytesStart::new("tt");
    tt.push_attribute(("xmlns", "http://www.w3.org/ns/ttml"));
    tt.push_attribute(("xmlns:tts", "http://www.w3.org/ns/ttml#styling"));
    tt.push_attribute(("xml:lang", track.language.code.as_str()));
    writer
        .write_event(Event::Start(tt))
        .map_err(|e| CaptionError::Xml(e.to_string()))?;

    // Head (metadata)
    writer
        .write_event(Event::Start(BytesStart::new("head")))
        .map_err(|e| CaptionError::Xml(e.to_string()))?;

    if let Some(title) = &track.metadata.title {
        writer
            .write_event(Event::Start(BytesStart::new("metadata")))
            .map_err(|e| CaptionError::Xml(e.to_string()))?;
        writer
            .write_event(Event::Start(BytesStart::new("title")))
            .map_err(|e| CaptionError::Xml(e.to_string()))?;
        writer
            .write_event(Event::Text(BytesText::new(title)))
            .map_err(|e| CaptionError::Xml(e.to_string()))?;
        writer
            .write_event(Event::End(BytesEnd::new("title")))
            .map_err(|e| CaptionError::Xml(e.to_string()))?;
        writer
            .write_event(Event::End(BytesEnd::new("metadata")))
            .map_err(|e| CaptionError::Xml(e.to_string()))?;
    }

    writer
        .write_event(Event::End(BytesEnd::new("head")))
        .map_err(|e| CaptionError::Xml(e.to_string()))?;

    // Body
    writer
        .write_event(Event::Start(BytesStart::new("body")))
        .map_err(|e| CaptionError::Xml(e.to_string()))?;
    writer
        .write_event(Event::Start(BytesStart::new("div")))
        .map_err(|e| CaptionError::Xml(e.to_string()))?;

    // Captions
    for caption in &track.captions {
        let mut p = BytesStart::new("p");
        p.push_attribute(("begin", format_ttml_time(caption.start).as_str()));
        p.push_attribute(("end", format_ttml_time(caption.end).as_str()));

        writer
            .write_event(Event::Start(p))
            .map_err(|e| CaptionError::Xml(e.to_string()))?;
        writer
            .write_event(Event::Text(BytesText::new(&caption.text)))
            .map_err(|e| CaptionError::Xml(e.to_string()))?;
        writer
            .write_event(Event::End(BytesEnd::new("p")))
            .map_err(|e| CaptionError::Xml(e.to_string()))?;
    }

    writer
        .write_event(Event::End(BytesEnd::new("div")))
        .map_err(|e| CaptionError::Xml(e.to_string()))?;
    writer
        .write_event(Event::End(BytesEnd::new("body")))
        .map_err(|e| CaptionError::Xml(e.to_string()))?;
    writer
        .write_event(Event::End(BytesEnd::new("tt")))
        .map_err(|e| CaptionError::Xml(e.to_string()))?;

    Ok(writer.into_inner().into_inner())
}

fn parse_ttml_time(s: &str) -> Result<Timestamp> {
    // Support multiple formats:
    // HH:MM:SS.mmm
    // HH:MM:SS:ff (frames)
    // offset-time (e.g., "10s", "500ms")

    if s.ends_with('s') && !s.contains(':') {
        // Offset time
        let num_part = &s[..s.len() - 1];
        if s.ends_with("ms") {
            let ms = num_part[..num_part.len() - 1]
                .parse::<i64>()
                .map_err(|e| CaptionError::Parse(e.to_string()))?;
            return Ok(Timestamp::from_millis(ms));
        } else {
            let secs = num_part
                .parse::<f64>()
                .map_err(|e| CaptionError::Parse(e.to_string()))?;
            return Ok(Timestamp::from_micros((secs * 1_000_000.0) as i64));
        }
    }

    // Clock time format
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() < 3 {
        return Err(CaptionError::Parse(format!("Invalid TTML time: {s}")));
    }

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
    let millis = if sec_parts.len() > 1 {
        sec_parts[1]
            .parse::<u32>()
            .map_err(|e| CaptionError::Parse(e.to_string()))?
    } else {
        0
    };

    Ok(Timestamp::from_hmsm(hours, minutes, seconds, millis))
}

fn format_ttml_time(ts: Timestamp) -> String {
    let (h, m, s, ms) = ts.as_hmsm();
    format!("{h:02}:{m:02}:{s:02}.{ms:03}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ttml_time() {
        let ts = parse_ttml_time("01:30:45.500").expect("parsing should succeed");
        assert_eq!(ts.as_hmsm(), (1, 30, 45, 500));

        let ts = parse_ttml_time("10s").expect("parsing should succeed");
        assert_eq!(ts.as_secs(), 10);

        let ts = parse_ttml_time("500ms").expect("parsing should succeed");
        assert_eq!(ts.as_millis(), 500);
    }

    #[test]
    fn test_write_ttml() {
        let mut track = CaptionTrack::new(Language::english());
        track
            .add_caption(Caption::new(
                Timestamp::from_secs(1),
                Timestamp::from_secs(3),
                "Test".to_string(),
            ))
            .expect("operation should succeed in test");

        let writer = TtmlWriter;
        let output = writer.write(&track).expect("writing should succeed");
        let text = String::from_utf8(output).expect("output should be valid UTF-8");

        assert!(text.contains("<?xml"));
        assert!(text.contains("<tt"));
        assert!(text.contains("Test"));
    }
}
