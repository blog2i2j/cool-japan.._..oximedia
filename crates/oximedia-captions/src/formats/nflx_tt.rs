//! Netflix Timed Text (NFLX-TT) profile serializer and deserializer.
//!
//! Netflix Timed Text is a strict subset/profile of TTML 1.0 used by Netflix
//! for all its subtitle delivery. Key characteristics:
//!
//! - Mandatory `<tt>` root with `xml:lang`, `tts:*` namespace attributes
//! - Body uses named regions (`r1` = bottom, `r2` = top, `r3` = center)
//! - Inline styling via `<span tts:fontStyle="italic">` / `<span tts:fontWeight="bold">`
//! - Timing in `HH:MM:SS.mmm` format on `<p begin="…" end="…">` elements
//!
//! # References
//! - Netflix Partner Help Center: "Timed Text Style Guide"
//! - TTML 1.0 W3C Recommendation

// ── Data types ───────────────────────────────────────────────────────────────

/// A single Netflix Timed Text subtitle entry.
#[derive(Debug, Clone)]
pub struct NflxTtEntry {
    /// Start timestamp in `HH:MM:SS.mmm` format.
    pub begin: String,
    /// End timestamp in `HH:MM:SS.mmm` format.
    pub end: String,
    /// Subtitle display text.
    pub text: String,
    /// Screen region for positioning.
    pub region: NflxRegion,
    /// Optional inline styling.
    pub style: Option<NflxStyle>,
}

/// Screen region used by Netflix TTML.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NflxRegion {
    /// Default subtitle position (bottom of screen) — region `r1`.
    Bottom,
    /// Forced translation / top area — region `r2`.
    Top,
    /// Center of screen — region `r3`.
    Center,
}

impl NflxRegion {
    /// Return the TTML region identifier string.
    #[must_use]
    pub fn region_id(&self) -> &'static str {
        match self {
            Self::Bottom => "r1",
            Self::Top => "r2",
            Self::Center => "r3",
        }
    }

    /// Parse a region identifier string into an [`NflxRegion`].
    /// Unknown IDs default to [`NflxRegion::Bottom`].
    #[must_use]
    pub fn from_id(id: &str) -> Self {
        match id.trim() {
            "r2" => Self::Top,
            "r3" => Self::Center,
            _ => Self::Bottom,
        }
    }
}

impl Default for NflxRegion {
    fn default() -> Self {
        Self::Bottom
    }
}

/// Inline style attributes for a Netflix TTML entry.
#[derive(Debug, Clone, Default)]
pub struct NflxStyle {
    /// Whether the text is italic.
    pub italic: bool,
    /// Whether the text is bold.
    pub bold: bool,
    /// CSS color string, e.g. `"#FFFFFF"`.
    pub color: Option<String>,
    /// Font size relative to frame height in percent (e.g. `100` = default).
    pub font_size_percent: Option<u32>,
}

impl NflxStyle {
    /// Return `true` if any styling attribute is set.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        !self.italic && !self.bold && self.color.is_none() && self.font_size_percent.is_none()
    }
}

// ── Parser ───────────────────────────────────────────────────────────────────

/// Parse Netflix TTML-based subtitle XML into a list of [`NflxTtEntry`]s.
///
/// This is a simplified subset parser: it looks for `<p begin="…" end="…">`
/// elements and extracts text, timing, region, and inline span styling.
/// The parser is intentionally lenient about namespace prefixes.
#[must_use]
pub fn parse_nflx_tt(xml: &str) -> Vec<NflxTtEntry> {
    let mut entries = Vec::new();

    // Split around <p …> elements using a simple byte-scan approach.
    // We locate every opening <p tag and its matching </p>.
    let mut search_start = 0usize;
    while let Some(p_open) = xml[search_start..].find("<p ").or_else(|| xml[search_start..].find("<p\t")) {
        let abs_p_open = search_start + p_open;

        // Find the end of the opening tag
        let tag_end = match xml[abs_p_open..].find('>') {
            Some(pos) => abs_p_open + pos + 1,
            None => break,
        };
        let opening_tag = &xml[abs_p_open..tag_end];

        // Find the closing </p>
        let close_tag_offset = match xml[tag_end..].find("</p>") {
            Some(pos) => pos,
            None => {
                search_start = tag_end;
                continue;
            }
        };
        let inner_html = &xml[tag_end..tag_end + close_tag_offset];

        // Extract begin / end attributes
        let begin = extract_attr(opening_tag, "begin").unwrap_or_default();
        let end = extract_attr(opening_tag, "end").unwrap_or_default();

        // Extract region (defaults to r1 / Bottom)
        let region_id = extract_attr(opening_tag, "region").unwrap_or_else(|| "r1".to_string());
        let region = NflxRegion::from_id(&region_id);

        // Parse inline spans for style and collect text
        let (text, style) = extract_text_and_style(inner_html);

        entries.push(NflxTtEntry {
            begin,
            end,
            text,
            region,
            style: if style.is_empty() { None } else { Some(style) },
        });

        search_start = tag_end + close_tag_offset + 4; // past </p>
    }

    entries
}

/// Extract the value of an XML attribute from a tag string.
/// Handles both single and double quotes.
fn extract_attr(tag: &str, attr: &str) -> Option<String> {
    // Try to find `attr="…"` or `attr='…'`
    let search = format!("{attr}=");
    let idx = tag.find(&search)?;
    let after_eq = &tag[idx + search.len()..];
    let quote_char = after_eq.chars().next()?;
    if quote_char != '"' && quote_char != '\'' {
        return None;
    }
    let content = &after_eq[1..]; // skip the opening quote
    let close = content.find(quote_char)?;
    Some(content[..close].to_string())
}

/// Strip XML tags from inner HTML and detect italic/bold/color styling.
///
/// Detects:
/// - `<span tts:fontStyle="italic">` — sets `italic`
/// - `<span tts:fontWeight="bold">` — sets `bold`
/// - `<span tts:color="…">` — captures color
/// - `<br/>` / `<br />` — converted to space
///
/// All other tags are stripped. Text nodes are concatenated.
fn extract_text_and_style(inner: &str) -> (String, NflxStyle) {
    let mut style = NflxStyle::default();
    let mut text = String::with_capacity(inner.len());
    let mut remaining = inner;

    while !remaining.is_empty() {
        if let Some(tag_start) = remaining.find('<') {
            // Text before this tag
            text.push_str(&remaining[..tag_start]);
            remaining = &remaining[tag_start..];

            // Find end of tag
            let tag_end = match remaining.find('>') {
                Some(pos) => pos + 1,
                None => {
                    text.push_str(remaining);
                    break;
                }
            };
            let tag = &remaining[..tag_end];
            remaining = &remaining[tag_end..];

            // Detect <br>
            let tag_lower = tag.to_lowercase();
            if tag_lower.starts_with("<br") {
                text.push(' ');
                continue;
            }

            // Detect <span …> styling
            if tag_lower.starts_with("<span") {
                if tag.contains("fontStyle=\"italic\"") || tag.contains("fontStyle='italic'") {
                    style.italic = true;
                }
                if tag.contains("fontWeight=\"bold\"") || tag.contains("fontWeight='bold'") {
                    style.bold = true;
                }
                // Color extraction
                if style.color.is_none() {
                    if let Some(color_val) = extract_attr(tag, "tts:color")
                        .or_else(|| extract_attr(tag, "color"))
                    {
                        style.color = Some(color_val);
                    }
                }
                // Font size extraction
                if style.font_size_percent.is_none() {
                    if let Some(fs) = extract_attr(tag, "tts:fontSize")
                        .or_else(|| extract_attr(tag, "fontSize"))
                    {
                        // Parse "100%" → 100
                        let numeric: String = fs.chars().take_while(|c| c.is_ascii_digit()).collect();
                        if let Ok(val) = numeric.parse::<u32>() {
                            style.font_size_percent = Some(val);
                        }
                    }
                }
            }
            // All other tags (</span>, </p>, etc.) are skipped
        } else {
            text.push_str(remaining);
            break;
        }
    }

    // Collapse multiple whitespace / trim
    let text = text
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");

    (text, style)
}

// ── Serializer ───────────────────────────────────────────────────────────────

/// Serialize a list of [`NflxTtEntry`]s to Netflix TTML XML.
///
/// The output includes the required NFLX namespace declarations, a three-region
/// layout block, and the body `<div>` containing one `<p>` per entry.
#[must_use]
pub fn serialize_nflx_tt(entries: &[NflxTtEntry], title: &str, language: &str) -> String {
    let mut out = String::with_capacity(1024 + entries.len() * 128);

    // XML declaration + root element with NFLX namespaces
    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str("<tt xml:lang=\"");
    out.push_str(&xml_escape(language));
    out.push_str("\" xmlns=\"http://www.w3.org/ns/ttml\"");
    out.push_str(" xmlns:tts=\"http://www.w3.org/ns/ttml#styling\"");
    out.push_str(" xmlns:ttm=\"http://www.w3.org/ns/ttml#metadata\"");
    out.push_str(" xmlns:nflx=\"http://www.netflix.com/ns/ttml\">\n");

    // Head section
    out.push_str("  <head>\n");
    out.push_str("    <ttm:title>");
    out.push_str(&xml_escape(title));
    out.push_str("</ttm:title>\n");
    // Region definitions
    out.push_str("    <layout>\n");
    out.push_str("      <region xml:id=\"r1\" tts:displayAlign=\"after\" tts:textAlign=\"center\"/>\n");
    out.push_str("      <region xml:id=\"r2\" tts:displayAlign=\"before\" tts:textAlign=\"center\"/>\n");
    out.push_str("      <region xml:id=\"r3\" tts:displayAlign=\"center\" tts:textAlign=\"center\"/>\n");
    out.push_str("    </layout>\n");
    out.push_str("  </head>\n");

    // Body
    out.push_str("  <body>\n");
    out.push_str("    <div>\n");

    for entry in entries {
        out.push_str("      <p begin=\"");
        out.push_str(&entry.begin);
        out.push_str("\" end=\"");
        out.push_str(&entry.end);
        out.push_str("\" region=\"");
        out.push_str(entry.region.region_id());
        out.push('"');

        // Inline style attribute on <p> for color/fontSize
        if let Some(style) = &entry.style {
            if let Some(color) = &style.color {
                out.push_str(" tts:color=\"");
                out.push_str(&xml_escape(color));
                out.push('"');
            }
            if let Some(fs) = style.font_size_percent {
                out.push_str(" tts:fontSize=\"");
                out.push_str(&fs.to_string());
                out.push_str("%\"");
            }
        }
        out.push('>');

        // Text content with optional italic/bold wrapping
        let escaped_text = xml_escape(&entry.text);
        match &entry.style {
            Some(style) if style.italic && style.bold => {
                out.push_str("<span tts:fontStyle=\"italic\"><span tts:fontWeight=\"bold\">");
                out.push_str(&escaped_text);
                out.push_str("</span></span>");
            }
            Some(style) if style.italic => {
                out.push_str("<span tts:fontStyle=\"italic\">");
                out.push_str(&escaped_text);
                out.push_str("</span>");
            }
            Some(style) if style.bold => {
                out.push_str("<span tts:fontWeight=\"bold\">");
                out.push_str(&escaped_text);
                out.push_str("</span>");
            }
            _ => {
                out.push_str(&escaped_text);
            }
        }

        out.push_str("</p>\n");
    }

    out.push_str("    </div>\n");
    out.push_str("  </body>\n");
    out.push_str("</tt>\n");

    out
}

/// Escape XML special characters.
fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c => out.push(c),
        }
    }
    out
}

// ── FormatParser / FormatWriter adapters ─────────────────────────────────────

/// Parser adapter implementing [`crate::formats::FormatParser`] for NFLX-TT.
pub struct NflxTtParser;

impl crate::formats::FormatParser for NflxTtParser {
    fn parse(&self, data: &[u8]) -> crate::error::Result<crate::types::CaptionTrack> {
        let xml = std::str::from_utf8(data)
            .map_err(|e| crate::error::CaptionError::Parse(format!("UTF-8 decode error: {e}")))?;
        let entries = parse_nflx_tt(xml);
        let mut track = crate::types::CaptionTrack::new(crate::types::Language::english());
        for entry in entries {
            let start = nflx_time_to_timestamp(&entry.begin)
                .unwrap_or_else(|| crate::types::Timestamp::from_micros(0));
            let end = nflx_time_to_timestamp(&entry.end)
                .unwrap_or_else(|| crate::types::Timestamp::from_micros(0));
            let caption = crate::types::Caption::new(start, end, entry.text);
            track.add_caption(caption)?;
        }
        Ok(track)
    }
}

/// Writer adapter implementing [`crate::formats::FormatWriter`] for NFLX-TT.
pub struct NflxTtWriter;

impl crate::formats::FormatWriter for NflxTtWriter {
    fn write(&self, track: &crate::types::CaptionTrack) -> crate::error::Result<Vec<u8>> {
        let entries: Vec<NflxTtEntry> = track
            .captions
            .iter()
            .map(|c| NflxTtEntry {
                begin: timestamp_to_nflx_time(c.start),
                end: timestamp_to_nflx_time(c.end),
                text: c.text.clone(),
                region: NflxRegion::Bottom,
                style: None,
            })
            .collect();
        let lang = track.language.code.as_str();
        let title = track
            .metadata
            .title
            .as_deref()
            .unwrap_or("Untitled");
        Ok(serialize_nflx_tt(&entries, title, lang).into_bytes())
    }
}

/// Parse a NFLX-TT time string `HH:MM:SS.mmm` into a [`crate::types::Timestamp`].
fn nflx_time_to_timestamp(s: &str) -> Option<crate::types::Timestamp> {
    // Supports HH:MM:SS.mmm and HH:MM:SS,mmm
    let s = s.replace(',', ".");
    let parts: Vec<&str> = s.splitn(3, ':').collect();
    if parts.len() != 3 {
        return None;
    }
    let h: u32 = parts[0].parse().ok()?;
    let m: u32 = parts[1].parse().ok()?;
    // seconds may have fractional: "SS.mmm"
    let sec_parts: Vec<&str> = parts[2].splitn(2, '.').collect();
    let sec: u32 = sec_parts.first()?.parse().ok()?;
    let millis: u32 = if sec_parts.len() == 2 {
        let frac = sec_parts[1];
        // Pad or truncate to 3 digits
        let padded = format!("{:0<3}", &frac[..frac.len().min(3)]);
        padded.parse().ok()?
    } else {
        0
    };
    Some(crate::types::Timestamp::from_hmsm(h, m, sec, millis))
}

/// Convert a [`crate::types::Timestamp`] to NFLX-TT time string `HH:MM:SS.mmm`.
fn timestamp_to_nflx_time(ts: crate::types::Timestamp) -> String {
    let total_ms = ts.as_micros() / 1000;
    let h = total_ms / 3_600_000;
    let m = (total_ms % 3_600_000) / 60_000;
    let s = (total_ms % 60_000) / 1000;
    let ms = total_ms % 1000;
    format!("{h:02}:{m:02}:{s:02}.{ms:03}")
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_parse_single_entry() {
        let xml = r#"<?xml version="1.0"?>
<tt xml:lang="en" xmlns="http://www.w3.org/ns/ttml">
  <body><div>
    <p begin="00:00:01.000" end="00:00:03.000" region="r1">Hello world</p>
  </div></body>
</tt>"#;
        let entries = parse_nflx_tt(xml);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].begin, "00:00:01.000");
        assert_eq!(entries[0].end, "00:00:03.000");
        assert_eq!(entries[0].text, "Hello world");
    }

    #[test]
    fn test_parse_region_defaults_to_bottom() {
        let xml = r#"<tt><body><div>
<p begin="00:00:01.000" end="00:00:02.000">No region attr</p>
</div></body></tt>"#;
        let entries = parse_nflx_tt(xml);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].region, NflxRegion::Bottom);
    }

    #[test]
    fn test_parse_italic_span() {
        let xml = r#"<tt><body><div>
<p begin="00:00:01.000" end="00:00:02.000" region="r1"><span tts:fontStyle="italic">Emphasis</span></p>
</div></body></tt>"#;
        let entries = parse_nflx_tt(xml);
        assert_eq!(entries.len(), 1);
        let style = entries[0].style.as_ref().expect("style present");
        assert!(style.italic);
        assert!(!style.bold);
        assert_eq!(entries[0].text, "Emphasis");
    }

    #[test]
    fn test_parse_bold_span() {
        let xml = r#"<tt><body><div>
<p begin="00:00:01.000" end="00:00:02.000" region="r1"><span tts:fontWeight="bold">Bold text</span></p>
</div></body></tt>"#;
        let entries = parse_nflx_tt(xml);
        assert_eq!(entries.len(), 1);
        let style = entries[0].style.as_ref().expect("style present");
        assert!(style.bold);
        assert!(!style.italic);
    }

    #[test]
    fn test_parse_color_attribute_preserved() {
        let xml = r##"<tt><body><div>
<p begin="00:00:01.000" end="00:00:02.000" region="r1"><span tts:color="#FFFF00">Yellow text</span></p>
</div></body></tt>"##;
        let entries = parse_nflx_tt(xml);
        assert_eq!(entries.len(), 1);
        let style = entries[0].style.as_ref().expect("style present");
        assert_eq!(style.color.as_deref(), Some("#FFFF00"));
    }

    #[test]
    fn test_parse_empty_input_returns_empty() {
        let entries = parse_nflx_tt("");
        assert!(entries.is_empty());
    }

    #[test]
    fn test_parse_top_region() {
        let xml = r#"<tt><body><div>
<p begin="00:00:01.000" end="00:00:02.000" region="r2">Top text</p>
</div></body></tt>"#;
        let entries = parse_nflx_tt(xml);
        assert_eq!(entries[0].region, NflxRegion::Top);
    }

    // ── serialize tests ──────────────────────────────────────────────────────

    #[test]
    fn test_serialize_includes_language_tag() {
        let entry = NflxTtEntry {
            begin: "00:00:01.000".to_string(),
            end: "00:00:03.000".to_string(),
            text: "Hello".to_string(),
            region: NflxRegion::Bottom,
            style: None,
        };
        let output = serialize_nflx_tt(&[entry], "Test", "fr");
        assert!(output.contains("xml:lang=\"fr\""));
    }

    #[test]
    fn test_serialize_round_trip_preserves_text() {
        let entry = NflxTtEntry {
            begin: "00:00:01.000".to_string(),
            end: "00:00:03.000".to_string(),
            text: "Round-trip text".to_string(),
            region: NflxRegion::Bottom,
            style: None,
        };
        let xml = serialize_nflx_tt(&[entry], "Test", "en");
        let parsed = parse_nflx_tt(&xml);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].text, "Round-trip text");
        assert_eq!(parsed[0].begin, "00:00:01.000");
        assert_eq!(parsed[0].end, "00:00:03.000");
    }

    #[test]
    fn test_serialize_multi_line_entry_text_joined() {
        // Multi-word text should survive the round-trip intact (whitespace-collapsed)
        let entry = NflxTtEntry {
            begin: "00:00:05.000".to_string(),
            end: "00:00:07.000".to_string(),
            text: "First line second line".to_string(),
            region: NflxRegion::Center,
            style: None,
        };
        let xml = serialize_nflx_tt(&[entry], "T", "en");
        let parsed = parse_nflx_tt(&xml);
        assert_eq!(parsed.len(), 1);
        assert!(parsed[0].text.contains("First"));
        assert!(parsed[0].text.contains("second"));
    }
}
