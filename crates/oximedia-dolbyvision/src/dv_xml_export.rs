//! Dolby Vision XML export/import for post-production workflows.
//!
//! Implements a full round-trip XML serializer and parser for Dolby Vision
//! metadata using the industry-standard `DolbyVision_RPU.xml` sidecar format.
//! Supports schema versions 2.0.5, 5.1.0, and 6.0.6.
//!
//! # Examples
//!
//! ```rust
//! use oximedia_dolbyvision::dv_xml_export::{
//!     DvXmlDocument, DvXmlVersion, DvShotEntry, DvL2Entry,
//!     DvXmlExporter, DvXmlParser,
//! };
//!
//! let doc = DvXmlDocument {
//!     version: DvXmlVersion::V6_0_6,
//!     shots: vec![DvShotEntry {
//!         frame_start: 0,
//!         frame_end: 23,
//!         l1_min: 0.0,
//!         l1_mid: 0.1,
//!         l1_max: 0.58,
//!         l2_entries: vec![],
//!     }],
//!     frame_rate: (24, 1),
//!     total_frames: 24,
//! };
//!
//! let xml = DvXmlExporter::to_xml(&doc);
//! let parsed = DvXmlParser::from_xml(&xml).expect("round-trip should succeed");
//! assert_eq!(parsed.shots.len(), 1);
//! ```

use std::fmt;

/// Dolby Vision XML schema version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DvXmlVersion {
    /// Schema version 2.0.5 — legacy CM tools.
    V2_0_5,
    /// Schema version 5.1.0 — CM v2.9 / CM v4.0 workflows.
    V5_1_0,
    /// Schema version 6.0.6 — CM v5.x, current standard.
    V6_0_6,
}

impl DvXmlVersion {
    /// Return the version string used in the XML attribute.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::V2_0_5 => "2.0.5",
            Self::V5_1_0 => "5.1.0",
            Self::V6_0_6 => "6.0.6",
        }
    }

    /// Parse from a version string.
    ///
    /// # Errors
    ///
    /// Returns [`DvXmlError::InvalidVersion`] if the string is not recognised.
    pub fn from_str(s: &str) -> Result<Self, DvXmlError> {
        match s.trim() {
            "2.0.5" => Ok(Self::V2_0_5),
            "5.1.0" => Ok(Self::V5_1_0),
            "6.0.6" => Ok(Self::V6_0_6),
            other => Err(DvXmlError::InvalidVersion(other.to_string())),
        }
    }
}

impl fmt::Display for DvXmlVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── L2 trim entry ─────────────────────────────────────────────────────────────

/// A single Level-2 (trim-pass) entry targeting a specific display peak.
#[derive(Debug, Clone, PartialEq)]
pub struct DvL2Entry {
    /// Target display peak luminance encoded as PQ code value [0, 4095].
    pub target_max_pq: u16,
    /// Trim slope, typically in [0.0, 2.0].
    pub trim_slope: f32,
    /// Trim offset, typically in [-1.0, 1.0].
    pub trim_offset: f32,
    /// Trim power (gamma adjustment), typically in [0.0, 2.0].
    pub trim_power: f32,
}

impl DvL2Entry {
    /// Create a neutral (identity) L2 entry for the given target peak PQ.
    #[must_use]
    pub fn identity(target_max_pq: u16) -> Self {
        Self {
            target_max_pq,
            trim_slope: 1.0,
            trim_offset: 0.0,
            trim_power: 1.0,
        }
    }
}

// ── Shot entry ────────────────────────────────────────────────────────────────

/// Per-shot Dolby Vision metadata entry.
///
/// L1 values are normalised PQ in [0.0, 1.0] where 1.0 == 10 000 nits.
#[derive(Debug, Clone, PartialEq)]
pub struct DvShotEntry {
    /// First frame index (inclusive, 0-based).
    pub frame_start: u64,
    /// Last frame index (inclusive).
    pub frame_end: u64,
    /// L1 minimum luminance (normalised PQ).
    pub l1_min: f32,
    /// L1 mid (average) luminance (normalised PQ).
    pub l1_mid: f32,
    /// L1 maximum luminance (normalised PQ).
    pub l1_max: f32,
    /// Level-2 trim entries for one or more target displays.
    pub l2_entries: Vec<DvL2Entry>,
}

impl DvShotEntry {
    /// Duration of this shot in frames (end inclusive).
    #[must_use]
    pub fn duration(&self) -> u64 {
        self.frame_end.saturating_sub(self.frame_start) + 1
    }
}

// ── Document ──────────────────────────────────────────────────────────────────

/// Top-level Dolby Vision XML document.
#[derive(Debug, Clone, PartialEq)]
pub struct DvXmlDocument {
    /// XML schema version.
    pub version: DvXmlVersion,
    /// Shot entries in presentation order.
    pub shots: Vec<DvShotEntry>,
    /// Frame rate expressed as (numerator, denominator), e.g. `(24, 1)`.
    pub frame_rate: (u32, u32),
    /// Total number of frames in the programme.
    pub total_frames: u64,
}

impl DvXmlDocument {
    /// Create an empty document with the given version and frame rate.
    #[must_use]
    pub fn new(version: DvXmlVersion, frame_rate: (u32, u32)) -> Self {
        Self {
            version,
            shots: Vec::new(),
            frame_rate,
            total_frames: 0,
        }
    }
}

// ── Errors ────────────────────────────────────────────────────────────────────

/// Errors produced by XML export / import operations.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum DvXmlError {
    /// Generic XML parse error with a description.
    #[error("XML parse error: {0}")]
    ParseError(String),

    /// The version string is not a recognised Dolby Vision XML schema version.
    #[error("invalid Dolby Vision XML version: '{0}'")]
    InvalidVersion(String),

    /// A required XML element or attribute was absent.
    #[error("missing required field: '{0}'")]
    MissingField(String),
}

// ── Exporter ─────────────────────────────────────────────────────────────────

/// Serializes a [`DvXmlDocument`] to a Dolby Vision XML string.
pub struct DvXmlExporter;

impl DvXmlExporter {
    /// Serialize `doc` to a UTF-8 XML string.
    ///
    /// The output is a self-contained XML document that round-trips through
    /// [`DvXmlParser::from_xml`] without information loss (within f32 precision).
    #[must_use]
    pub fn to_xml(doc: &DvXmlDocument) -> String {
        let mut out = String::with_capacity(4096);

        out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        out.push_str("<DolbyVisionXML version=\"");
        out.push_str(doc.version.as_str());
        out.push_str("\">\n");

        // Frame rate
        push_tag(
            &mut out,
            "FrameRate",
            &format!("{}/{}", doc.frame_rate.0, doc.frame_rate.1),
            2,
        );

        // Total frames
        push_tag(
            &mut out,
            "TotalFrames",
            &doc.total_frames.to_string(),
            2,
        );

        // Shots
        out.push_str("  <Shots>\n");
        for shot in &doc.shots {
            out.push_str("    <Shot>\n");
            push_tag(&mut out, "FrameStart", &shot.frame_start.to_string(), 6);
            push_tag(&mut out, "FrameEnd", &shot.frame_end.to_string(), 6);
            out.push_str("      <Level1>\n");
            push_tag(&mut out, "L1Min", &format_f32(shot.l1_min), 8);
            push_tag(&mut out, "L1Mid", &format_f32(shot.l1_mid), 8);
            push_tag(&mut out, "L1Max", &format_f32(shot.l1_max), 8);
            out.push_str("      </Level1>\n");

            if !shot.l2_entries.is_empty() {
                out.push_str("      <Level2Entries>\n");
                for l2 in &shot.l2_entries {
                    out.push_str("        <L2Entry>\n");
                    push_tag(
                        &mut out,
                        "TargetMaxPQ",
                        &l2.target_max_pq.to_string(),
                        10,
                    );
                    push_tag(&mut out, "TrimSlope", &format_f32(l2.trim_slope), 10);
                    push_tag(&mut out, "TrimOffset", &format_f32(l2.trim_offset), 10);
                    push_tag(&mut out, "TrimPower", &format_f32(l2.trim_power), 10);
                    out.push_str("        </L2Entry>\n");
                }
                out.push_str("      </Level2Entries>\n");
            }

            out.push_str("    </Shot>\n");
        }
        out.push_str("  </Shots>\n");
        out.push_str("</DolbyVisionXML>\n");
        out
    }
}

// ── Parser ────────────────────────────────────────────────────────────────────

/// Parses a Dolby Vision XML string into a [`DvXmlDocument`].
pub struct DvXmlParser;

impl DvXmlParser {
    /// Parse `xml` into a [`DvXmlDocument`].
    ///
    /// # Errors
    ///
    /// Returns [`DvXmlError`] on malformed input, unrecognised version strings,
    /// or missing mandatory fields.
    pub fn from_xml(xml: &str) -> Result<DvXmlDocument, DvXmlError> {
        let version_str = extract_attr(xml, "DolbyVisionXML", "version")
            .ok_or_else(|| DvXmlError::MissingField("DolbyVisionXML@version".to_string()))?;
        let version = DvXmlVersion::from_str(&version_str)?;

        let frame_rate = parse_frame_rate(xml)?;
        let total_frames = parse_u64_tag(xml, "TotalFrames")
            .map_err(|e| DvXmlError::ParseError(e.to_string()))?;

        let shots = parse_shots(xml)?;

        Ok(DvXmlDocument {
            version,
            shots,
            frame_rate,
            total_frames,
        })
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

fn push_tag(out: &mut String, tag: &str, value: &str, indent: usize) {
    for _ in 0..indent {
        out.push(' ');
    }
    out.push('<');
    out.push_str(tag);
    out.push('>');
    out.push_str(&xml_escape(value));
    out.push_str("</");
    out.push_str(tag);
    out.push_str(">\n");
}

fn format_f32(v: f32) -> String {
    format!("{v:.8}")
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn xml_unescape(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
}

/// Extract the value of `attr` from the opening tag `<tag ...>`.
fn extract_attr(xml: &str, tag: &str, attr: &str) -> Option<String> {
    let tag_open = format!("<{}", tag);
    let pos = xml.find(&tag_open)?;
    let tag_end = xml[pos..].find('>')?;
    let tag_content = &xml[pos..pos + tag_end];

    let needle = format!("{}=\"", attr);
    let attr_pos = tag_content.find(&needle)?;
    let value_start = attr_pos + needle.len();
    let value_end = tag_content[value_start..].find('"')?;
    Some(xml_unescape(&tag_content[value_start..value_start + value_end]))
}

/// Extract text content of the first `<tag>…</tag>` within `xml`.
fn extract_tag_text(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{}>", tag);
    let close = format!("</{}>", tag);
    let start = xml.find(&open)? + open.len();
    let end = xml[start..].find(&close)?;
    Some(xml_unescape(xml[start..start + end].trim()))
}

fn parse_u64_tag(xml: &str, tag: &str) -> Result<u64, DvXmlError> {
    match extract_tag_text(xml, tag) {
        Some(s) => s.parse::<u64>().map_err(|e| {
            DvXmlError::ParseError(format!("invalid <{tag}> u64 value '{s}': {e}"))
        }),
        None => Ok(0),
    }
}

fn parse_f32_tag(xml: &str, tag: &str) -> Result<f32, DvXmlError> {
    match extract_tag_text(xml, tag) {
        Some(s) => s.parse::<f32>().map_err(|e| {
            DvXmlError::ParseError(format!("invalid <{tag}> f32 value '{s}': {e}"))
        }),
        None => Ok(0.0),
    }
}

/// Parse `<FrameRate>num/den</FrameRate>` into `(num, den)`.
fn parse_frame_rate(xml: &str) -> Result<(u32, u32), DvXmlError> {
    let raw = match extract_tag_text(xml, "FrameRate") {
        Some(s) => s,
        None => return Ok((24, 1)),
    };
    let mut parts = raw.splitn(2, '/');
    let num_str = parts.next().unwrap_or("24");
    let den_str = parts.next().unwrap_or("1");
    let num = num_str.parse::<u32>().map_err(|e| {
        DvXmlError::ParseError(format!("invalid FrameRate numerator '{num_str}': {e}"))
    })?;
    let den = den_str.parse::<u32>().map_err(|e| {
        DvXmlError::ParseError(format!("invalid FrameRate denominator '{den_str}': {e}"))
    })?;
    Ok((num, den))
}

fn parse_l2_entries(block: &str) -> Result<Vec<DvL2Entry>, DvXmlError> {
    let mut entries = Vec::new();
    let mut search_start = 0;

    while let Some(rel) = block[search_start..].find("<L2Entry>") {
        let abs = search_start + rel;
        let end_rel = block[abs..]
            .find("</L2Entry>")
            .ok_or_else(|| DvXmlError::ParseError("unclosed <L2Entry>".to_string()))?;
        let entry_block = &block[abs..abs + end_rel + "</L2Entry>".len()];

        let target_max_pq = match extract_tag_text(entry_block, "TargetMaxPQ") {
            Some(s) => s.parse::<u16>().map_err(|e| {
                DvXmlError::ParseError(format!("invalid TargetMaxPQ '{s}': {e}"))
            })?,
            None => {
                return Err(DvXmlError::MissingField("L2Entry/TargetMaxPQ".to_string()))
            }
        };

        entries.push(DvL2Entry {
            target_max_pq,
            trim_slope: parse_f32_tag(entry_block, "TrimSlope")?,
            trim_offset: parse_f32_tag(entry_block, "TrimOffset")?,
            trim_power: parse_f32_tag(entry_block, "TrimPower")?,
        });

        search_start = abs + end_rel + "</L2Entry>".len();
    }

    Ok(entries)
}

fn parse_shots(xml: &str) -> Result<Vec<DvShotEntry>, DvXmlError> {
    let mut shots = Vec::new();
    let mut search_start = 0;

    while let Some(rel) = xml[search_start..].find("<Shot>") {
        let abs = search_start + rel;
        let end_rel = xml[abs..]
            .find("</Shot>")
            .ok_or_else(|| DvXmlError::ParseError("unclosed <Shot>".to_string()))?;
        let block = &xml[abs..abs + end_rel + "</Shot>".len()];

        let frame_start = parse_u64_tag(block, "FrameStart")?;
        let frame_end = parse_u64_tag(block, "FrameEnd")?;

        // Parse Level1 block
        let (l1_min, l1_mid, l1_max) = if let Some(l1_rel) = block.find("<Level1>") {
            let l1_end = block[l1_rel..]
                .find("</Level1>")
                .ok_or_else(|| DvXmlError::ParseError("unclosed <Level1>".to_string()))?;
            let l1_block = &block[l1_rel..l1_rel + l1_end + "</Level1>".len()];
            (
                parse_f32_tag(l1_block, "L1Min")?,
                parse_f32_tag(l1_block, "L1Mid")?,
                parse_f32_tag(l1_block, "L1Max")?,
            )
        } else {
            (0.0, 0.0, 0.0)
        };

        // Parse Level2 entries
        let l2_entries = if block.contains("<Level2Entries>") {
            let l2_start = block.find("<Level2Entries>").unwrap_or(0);
            let l2_end = block[l2_start..]
                .find("</Level2Entries>")
                .ok_or_else(|| DvXmlError::ParseError("unclosed <Level2Entries>".to_string()))?;
            let l2_block = &block[l2_start..l2_start + l2_end + "</Level2Entries>".len()];
            parse_l2_entries(l2_block)?
        } else {
            Vec::new()
        };

        shots.push(DvShotEntry {
            frame_start,
            frame_end,
            l1_min,
            l1_mid,
            l1_max,
            l2_entries,
        });

        search_start = abs + end_rel + "</Shot>".len();
    }

    Ok(shots)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_shot(start: u64, end: u64) -> DvShotEntry {
        DvShotEntry {
            frame_start: start,
            frame_end: end,
            l1_min: 0.001,
            l1_mid: 0.12,
            l1_max: 0.58,
            l2_entries: vec![
                DvL2Entry {
                    target_max_pq: 2081,
                    trim_slope: 1.0,
                    trim_offset: 0.0,
                    trim_power: 1.0,
                },
                DvL2Entry {
                    target_max_pq: 3079,
                    trim_slope: 0.9,
                    trim_offset: 0.05,
                    trim_power: 1.1,
                },
            ],
        }
    }

    fn make_doc() -> DvXmlDocument {
        DvXmlDocument {
            version: DvXmlVersion::V6_0_6,
            shots: vec![make_shot(0, 23), make_shot(24, 71)],
            frame_rate: (24, 1),
            total_frames: 72,
        }
    }

    #[test]
    fn test_round_trip_version() {
        let doc = make_doc();
        let xml = DvXmlExporter::to_xml(&doc);
        let parsed = DvXmlParser::from_xml(&xml).expect("parse should succeed");
        assert_eq!(parsed.version, DvXmlVersion::V6_0_6);
    }

    #[test]
    fn test_round_trip_frame_rate() {
        let doc = make_doc();
        let xml = DvXmlExporter::to_xml(&doc);
        let parsed = DvXmlParser::from_xml(&xml).expect("parse should succeed");
        assert_eq!(parsed.frame_rate, (24, 1));
    }

    #[test]
    fn test_round_trip_total_frames() {
        let doc = make_doc();
        let xml = DvXmlExporter::to_xml(&doc);
        let parsed = DvXmlParser::from_xml(&xml).expect("parse should succeed");
        assert_eq!(parsed.total_frames, 72);
    }

    #[test]
    fn test_round_trip_shot_count() {
        let doc = make_doc();
        let xml = DvXmlExporter::to_xml(&doc);
        let parsed = DvXmlParser::from_xml(&xml).expect("parse should succeed");
        assert_eq!(parsed.shots.len(), 2);
    }

    #[test]
    fn test_round_trip_l1_values() {
        let doc = make_doc();
        let xml = DvXmlExporter::to_xml(&doc);
        let parsed = DvXmlParser::from_xml(&xml).expect("parse should succeed");
        let shot = &parsed.shots[0];
        assert!((shot.l1_max - 0.58).abs() < 1e-5, "l1_max={}", shot.l1_max);
        assert!((shot.l1_min - 0.001).abs() < 1e-5, "l1_min={}", shot.l1_min);
        assert!(
            (shot.l1_mid - 0.12).abs() < 1e-5,
            "l1_mid={}",
            shot.l1_mid
        );
    }

    #[test]
    fn test_round_trip_l2_entry_count() {
        let doc = make_doc();
        let xml = DvXmlExporter::to_xml(&doc);
        let parsed = DvXmlParser::from_xml(&xml).expect("parse should succeed");
        assert_eq!(parsed.shots[0].l2_entries.len(), 2);
    }

    #[test]
    fn test_round_trip_l2_values() {
        let doc = make_doc();
        let xml = DvXmlExporter::to_xml(&doc);
        let parsed = DvXmlParser::from_xml(&xml).expect("parse should succeed");
        let l2 = &parsed.shots[0].l2_entries[1];
        assert_eq!(l2.target_max_pq, 3079);
        assert!((l2.trim_slope - 0.9).abs() < 1e-5, "slope={}", l2.trim_slope);
        assert!(
            (l2.trim_offset - 0.05).abs() < 1e-5,
            "offset={}",
            l2.trim_offset
        );
    }

    #[test]
    fn test_version_string_v2_0_5() {
        assert_eq!(DvXmlVersion::V2_0_5.as_str(), "2.0.5");
    }

    #[test]
    fn test_version_string_v5_1_0() {
        assert_eq!(DvXmlVersion::V5_1_0.as_str(), "5.1.0");
    }

    #[test]
    fn test_version_string_v6_0_6() {
        assert_eq!(DvXmlVersion::V6_0_6.as_str(), "6.0.6");
    }

    #[test]
    fn test_version_parse_invalid() {
        let err = DvXmlVersion::from_str("99.0.0");
        assert!(matches!(err, Err(DvXmlError::InvalidVersion(_))));
    }

    #[test]
    fn test_xml_contains_xml_declaration() {
        let doc = make_doc();
        let xml = DvXmlExporter::to_xml(&doc);
        assert!(xml.starts_with("<?xml version"), "missing XML declaration");
    }

    #[test]
    fn test_shot_duration() {
        let shot = make_shot(10, 33);
        assert_eq!(shot.duration(), 24);
    }

    #[test]
    fn test_missing_version_attribute() {
        let xml = "<DolbyVisionXML><TotalFrames>10</TotalFrames><Shots></Shots></DolbyVisionXML>";
        let result = DvXmlParser::from_xml(xml);
        assert!(matches!(result, Err(DvXmlError::MissingField(_))));
    }

    #[test]
    fn test_unknown_version_returns_error() {
        let xml = "<DolbyVisionXML version=\"9.9.9\"><TotalFrames>0</TotalFrames><Shots></Shots></DolbyVisionXML>";
        let result = DvXmlParser::from_xml(xml);
        assert!(matches!(result, Err(DvXmlError::InvalidVersion(_))));
    }

    #[test]
    fn test_l2_identity_constructor() {
        let l2 = DvL2Entry::identity(2081);
        assert_eq!(l2.target_max_pq, 2081);
        assert!((l2.trim_slope - 1.0).abs() < f32::EPSILON);
        assert!((l2.trim_offset).abs() < f32::EPSILON);
        assert!((l2.trim_power - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_fractional_frame_rate() {
        let mut doc = make_doc();
        doc.frame_rate = (30000, 1001); // 29.97 fps
        let xml = DvXmlExporter::to_xml(&doc);
        let parsed = DvXmlParser::from_xml(&xml).expect("parse should succeed");
        assert_eq!(parsed.frame_rate, (30000, 1001));
    }
}
