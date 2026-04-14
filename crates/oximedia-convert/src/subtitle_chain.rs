// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Subtitle format conversion chain.
//!
//! Provides `SubtitleConverter` with conversion methods between common subtitle
//! formats: SRT → WebVTT and WebVTT → TTML (XML).
//!
//! Both conversions are pure-Rust text transformations; no external subtitle
//! parsing dependencies are required.

/// Subtitle format converter.
pub struct SubtitleConverter;

impl SubtitleConverter {
    /// Convert an SRT (SubRip) subtitle string to WebVTT format.
    ///
    /// # SRT structure
    /// ```text
    /// 1
    /// 00:00:01,000 --> 00:00:04,000
    /// Hello, world!
    ///
    /// 2
    /// 00:00:05,000 --> 00:00:08,500
    /// Second line.
    /// ```
    ///
    /// # WebVTT structure
    /// ```text
    /// WEBVTT
    ///
    /// 1
    /// 00:00:01.000 --> 00:00:04.000
    /// Hello, world!
    ///
    /// 2
    /// 00:00:05.000 --> 00:00:08.500
    /// Second line.
    /// ```
    ///
    /// The only structural difference is:
    /// - The leading `WEBVTT` header line.
    /// - Commas in timestamps are replaced by dots.
    pub fn srt_to_vtt(srt: &str) -> String {
        let mut output = String::with_capacity(srt.len() + 64);
        output.push_str("WEBVTT\n\n");

        for block in split_blocks(srt) {
            // Only replace commas with dots in timing lines, not in subtitle text
            let mut converted_lines = Vec::new();
            for line in block.lines() {
                if line.contains("-->") {
                    converted_lines.push(line.replace(',', "."));
                } else {
                    converted_lines.push(line.to_string());
                }
            }
            output.push_str(&converted_lines.join("\n"));
            output.push('\n');
        }

        output.trim_end().to_string() + "\n"
    }

    /// Convert a WebVTT subtitle string to TTML (Timed Text Markup Language).
    ///
    /// Produces a minimal TTML 1.0 document suitable for broadcast and DASH
    /// deployments.  The output follows the TTML W3C Recommendation structure:
    ///
    /// ```xml
    /// <?xml version="1.0" encoding="UTF-8"?>
    /// <tt xml:lang="en" xmlns="http://www.w3.org/ns/ttml" ...>
    ///   <body>
    ///     <div>
    ///       <p begin="00:00:01.000" end="00:00:04.000">Hello, world!</p>
    ///       ...
    ///     </div>
    ///   </body>
    /// </tt>
    /// ```
    pub fn vtt_to_ttml(vtt: &str) -> String {
        let cues = parse_vtt_cues(vtt);
        build_ttml(&cues)
    }
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// A single subtitle cue with timing and content.
#[derive(Debug, PartialEq)]
struct Cue {
    /// Optional cue identifier (numeric or named).
    id: Option<String>,
    /// Start timestamp string (dots as decimal separator).
    begin: String,
    /// End timestamp string (dots as decimal separator).
    end: String,
    /// One or more text lines.
    lines: Vec<String>,
}

/// Split a subtitle document into non-empty blocks separated by blank lines.
fn split_blocks(text: &str) -> Vec<String> {
    let mut blocks: Vec<String> = Vec::new();
    let mut current = String::new();

    for line in text.lines() {
        if line.trim().is_empty() {
            if !current.trim().is_empty() {
                blocks.push(current.trim_end().to_string());
            }
            current.clear();
        } else {
            current.push_str(line);
            current.push('\n');
        }
    }

    if !current.trim().is_empty() {
        blocks.push(current.trim_end().to_string());
    }

    blocks
}

/// Parse WebVTT cues from a VTT string.
///
/// Ignores the `WEBVTT` header and NOTE/STYLE/REGION blocks.
fn parse_vtt_cues(vtt: &str) -> Vec<Cue> {
    let mut cues = Vec::new();
    let blocks = split_blocks(vtt);

    for block in &blocks {
        let lines: Vec<&str> = block.lines().collect();
        if lines.is_empty() {
            continue;
        }

        // Skip the WEBVTT header block
        if lines[0].trim().starts_with("WEBVTT") {
            continue;
        }
        // Skip NOTE / STYLE / REGION blocks
        if lines[0].trim().starts_with("NOTE")
            || lines[0].trim().starts_with("STYLE")
            || lines[0].trim().starts_with("REGION")
        {
            continue;
        }

        // Determine if first line is a cue identifier (no "-->" in it)
        let (id_opt, timing_idx) = if !lines[0].contains("-->") {
            (Some(lines[0].trim().to_string()), 1)
        } else {
            (None, 0)
        };

        if timing_idx >= lines.len() {
            continue;
        }

        // Parse timing line: "HH:MM:SS.mmm --> HH:MM:SS.mmm [position ...]"
        let timing_line = lines[timing_idx];
        if let Some((begin, end)) = parse_timing_line(timing_line) {
            let text_lines: Vec<String> = lines[(timing_idx + 1)..]
                .iter()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect();

            if !text_lines.is_empty() {
                cues.push(Cue {
                    id: id_opt,
                    begin,
                    end,
                    lines: text_lines,
                });
            }
        }
    }

    cues
}

/// Parse a timing line and return (begin, end) timestamp strings.
///
/// Both SRT (`00:00:01,000`) and VTT (`00:00:01.000`) separators are accepted.
fn parse_timing_line(line: &str) -> Option<(String, String)> {
    let arrow_pos = line.find("-->")?;
    let begin_raw = line[..arrow_pos].trim().replace(',', ".");
    let rest = line[(arrow_pos + 3)..].trim();
    // VTT may have settings after end timestamp: "00:01:00.000 align:center"
    let end_raw = rest.split_whitespace().next()?.replace(',', ".");

    Some((begin_raw, end_raw))
}

/// Escape XML special characters in a text string.
fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
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

/// Build a TTML 1.0 document from a list of cues.
fn build_ttml(cues: &[Cue]) -> String {
    let mut out = String::with_capacity(512 + cues.len() * 120);

    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str("<tt xml:lang=\"en\"\n");
    out.push_str("    xmlns=\"http://www.w3.org/ns/ttml\"\n");
    out.push_str("    xmlns:tts=\"http://www.w3.org/ns/ttml#styling\"\n");
    out.push_str("    xmlns:ttp=\"http://www.w3.org/ns/ttml#parameter\"\n");
    out.push_str("    ttp:timeBase=\"media\">\n");
    out.push_str("  <body>\n");
    out.push_str("    <div>\n");

    for (i, cue) in cues.iter().enumerate() {
        let id_attr = if let Some(ref id) = cue.id {
            format!(" xml:id=\"{}\"", xml_escape(id))
        } else {
            format!(" xml:id=\"cue{}\"", i + 1)
        };

        let text = cue
            .lines
            .iter()
            .map(|l| xml_escape(l))
            .collect::<Vec<_>>()
            .join("<br/>");

        out.push_str(&format!(
            "      <p{} begin=\"{}\" end=\"{}\">{}</p>\n",
            id_attr, cue.begin, cue.end, text
        ));
    }

    out.push_str("    </div>\n");
    out.push_str("  </body>\n");
    out.push_str("</tt>\n");

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_SRT: &str = "\
1
00:00:01,000 --> 00:00:04,000
Hello, world!

2
00:00:05,000 --> 00:00:08,500
Second subtitle line.
";

    #[test]
    fn srt_to_vtt_starts_with_webvtt() {
        let vtt = SubtitleConverter::srt_to_vtt(SAMPLE_SRT);
        assert!(
            vtt.starts_with("WEBVTT\n"),
            "VTT must start with WEBVTT header"
        );
    }

    #[test]
    fn srt_to_vtt_replaces_comma_with_dot() {
        let vtt = SubtitleConverter::srt_to_vtt(SAMPLE_SRT);
        assert!(vtt.contains("00:00:01.000"), "comma should become dot");
        assert!(
            !vtt.contains("00:00:01,000"),
            "original comma format should be gone"
        );
    }

    #[test]
    fn srt_to_vtt_preserves_text() {
        let vtt = SubtitleConverter::srt_to_vtt(SAMPLE_SRT);
        assert!(vtt.contains("Hello, world!"));
        assert!(vtt.contains("Second subtitle line."));
    }

    #[test]
    fn vtt_to_ttml_produces_xml_header() {
        let vtt = SubtitleConverter::srt_to_vtt(SAMPLE_SRT);
        let ttml = SubtitleConverter::vtt_to_ttml(&vtt);
        assert!(ttml.starts_with("<?xml"));
    }

    #[test]
    fn vtt_to_ttml_contains_tt_element() {
        let vtt = SubtitleConverter::srt_to_vtt(SAMPLE_SRT);
        let ttml = SubtitleConverter::vtt_to_ttml(&vtt);
        assert!(ttml.contains("<tt "));
        assert!(ttml.contains("</tt>"));
    }

    #[test]
    fn vtt_to_ttml_contains_p_elements_with_timing() {
        let vtt = SubtitleConverter::srt_to_vtt(SAMPLE_SRT);
        let ttml = SubtitleConverter::vtt_to_ttml(&vtt);
        assert!(ttml.contains("begin=\"00:00:01.000\""));
        assert!(ttml.contains("end=\"00:00:04.000\""));
    }

    #[test]
    fn vtt_to_ttml_text_preserved() {
        let vtt = SubtitleConverter::srt_to_vtt(SAMPLE_SRT);
        let ttml = SubtitleConverter::vtt_to_ttml(&vtt);
        assert!(ttml.contains("Hello, world!"));
        assert!(ttml.contains("Second subtitle line."));
    }

    #[test]
    fn xml_escape_special_chars() {
        assert_eq!(xml_escape("<b>&\"'"), "&lt;b&gt;&amp;&quot;&apos;");
    }

    #[test]
    fn split_blocks_handles_trailing_newlines() {
        let text = "1\n00:00:01,000 --> 00:00:02,000\nText\n\n";
        let blocks = split_blocks(text);
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn srt_to_vtt_empty_input() {
        let vtt = SubtitleConverter::srt_to_vtt("");
        assert!(vtt.starts_with("WEBVTT"));
    }

    #[test]
    fn vtt_to_ttml_empty_input() {
        let ttml = SubtitleConverter::vtt_to_ttml("WEBVTT\n\n");
        assert!(ttml.contains("<tt "));
        assert!(!ttml.contains("<p ")); // no cues
    }
}
