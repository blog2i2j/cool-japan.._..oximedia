//! NDI metadata XML serialization and deserialization.
//!
//! This module provides structured serialization and deserialization of the NDI
//! metadata frames that travel alongside audio/video in the NDI protocol.
//!
//! ## Supported metadata kinds
//!
//! | Kind               | XML root element     | Payload type              |
//! |--------------------|----------------------|---------------------------|
//! | `PtzPosition`      | `<ndi_ptz …/>`       | [`PtzPayload`]            |
//! | `TallyState`       | `<ndi_tally …/>`     | [`TallyPayload`]          |
//! | `Connection`       | `<ndi_connection …/>`| [`MetadataPayload::Raw`]  |
//! | `Custom(tag)`      | `<{tag}>…</{tag}>`   | [`MetadataPayload::Raw`]  |
//!
//! ## Round-trip guarantees
//!
//! Every value produced by [`MetadataSerializer::to_xml`] can be decoded back
//! by [`MetadataSerializer::from_xml`] to an equivalent [`NdiMetadataFrame`].

#![allow(dead_code)]

use thiserror::Error;

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors that can occur during metadata serialization or deserialization.
#[derive(Debug, Error, PartialEq)]
pub enum MetaError {
    /// Malformed XML that could not be decoded.
    #[error("XML parse error: {0}")]
    ParseError(String),

    /// The root element tag is not recognized as any known NDI metadata kind.
    #[error("Unknown metadata kind: {0}")]
    UnknownKind(String),

    /// A required field was absent from the XML payload.
    #[error("Missing required field: {0}")]
    MissingField(String),
}

// ── Payload types ─────────────────────────────────────────────────────────────

/// PTZ (Pan-Tilt-Zoom) position payload.
#[derive(Debug, Clone, PartialEq)]
pub struct PtzPayload {
    /// Pan position in the range [-1.0, +1.0].
    pub pan: f32,
    /// Tilt position in the range [-1.0, +1.0].
    pub tilt: f32,
    /// Zoom level in the range [0.0, 1.0].
    pub zoom: f32,
    /// Focus position in [0.0, 1.0], or `None` if not specified.
    pub focus: Option<f32>,
}

/// Camera tally state payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TallyPayload {
    /// `true` when this source is on-air (program output).
    pub program: bool,
    /// `true` when this source is cued (preview/monitor output).
    pub preview: bool,
}

/// Discriminated union of all supported metadata payloads.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataPayload {
    /// PTZ position / speed payload.
    Ptz(PtzPayload),
    /// Tally light state payload.
    Tally(TallyPayload),
    /// Opaque raw XML string for unknown or custom payloads.
    Raw(String),
}

// ── Kind enum ─────────────────────────────────────────────────────────────────

/// NDI metadata frame kind, derived from the root XML element.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NdiMetadataKind {
    /// Camera PTZ position data (`<ndi_ptz>`).
    PtzPosition,
    /// Tally state update (`<ndi_tally>`).
    TallyState,
    /// Connection metadata (`<ndi_connection>`).
    Connection,
    /// User-defined / unknown root element.
    Custom(String),
}

impl NdiMetadataKind {
    /// Return the XML root element name for this kind.
    pub fn element_name(&self) -> &str {
        match self {
            Self::PtzPosition => "ndi_ptz",
            Self::TallyState => "ndi_tally",
            Self::Connection => "ndi_connection",
            Self::Custom(tag) => tag.as_str(),
        }
    }

    /// Parse a root element name into a [`NdiMetadataKind`].
    ///
    /// The parse is infallible: unrecognized tags become [`NdiMetadataKind::Custom`].
    pub fn from_element(element: &str) -> Self {
        match element {
            "ndi_ptz" => Self::PtzPosition,
            "ndi_tally" => Self::TallyState,
            "ndi_connection" => Self::Connection,
            other => Self::Custom(other.to_string()),
        }
    }
}

// ── Frame type ────────────────────────────────────────────────────────────────

/// A complete NDI metadata frame ready for wire transmission.
#[derive(Debug, Clone, PartialEq)]
pub struct NdiMetadataFrame {
    /// Identifies the type of metadata carried in this frame.
    pub kind: NdiMetadataKind,
    /// Structured or raw payload.
    pub payload: MetadataPayload,
    /// Name of the NDI source that generated this frame.
    pub source_name: String,
    /// Timecode in 100-nanosecond units (NDI convention, compatible with SMPTE).
    pub timecode: u64,
}

// ── Serializer ────────────────────────────────────────────────────────────────

/// Serialize and deserialize [`NdiMetadataFrame`] objects to/from XML strings.
pub struct MetadataSerializer;

impl MetadataSerializer {
    /// Serialize an [`NdiMetadataFrame`] to its XML wire representation.
    ///
    /// The output always includes a `source` attribute and a `timecode`
    /// attribute on the root element, followed by payload-specific attributes
    /// or inner text.
    pub fn to_xml(frame: &NdiMetadataFrame) -> String {
        let tag = frame.kind.element_name();
        let source = xml_escape(&frame.source_name);
        let tc = frame.timecode;

        match &frame.payload {
            MetadataPayload::Ptz(ptz) => {
                let pan = ptz.pan;
                let tilt = ptz.tilt;
                let zoom = ptz.zoom;
                if let Some(focus) = ptz.focus {
                    format!(
                        r#"<{tag} source="{source}" timecode="{tc}" pan="{pan:.6}" tilt="{tilt:.6}" zoom="{zoom:.6}" focus="{focus:.6}"/>"#
                    )
                } else {
                    format!(
                        r#"<{tag} source="{source}" timecode="{tc}" pan="{pan:.6}" tilt="{tilt:.6}" zoom="{zoom:.6}"/>"#
                    )
                }
            }
            MetadataPayload::Tally(tally) => {
                let prog = tally.program;
                let prev = tally.preview;
                format!(
                    r#"<{tag} source="{source}" timecode="{tc}" program="{prog}" preview="{prev}"/>"#
                )
            }
            MetadataPayload::Raw(inner) => {
                format!(r#"<{tag} source="{source}" timecode="{tc}">{inner}</{tag}>"#)
            }
        }
    }

    /// Deserialize an XML string into an [`NdiMetadataFrame`].
    ///
    /// # Errors
    ///
    /// Returns [`MetaError::ParseError`] if the string is not valid NDI XML,
    /// [`MetaError::MissingField`] if required attributes are absent, or
    /// [`MetaError::UnknownKind`] for a tag that requires a typed payload but
    /// has structurally invalid attributes.
    pub fn from_xml(xml: &str) -> Result<NdiMetadataFrame, MetaError> {
        let xml = xml.trim();

        // Extract the root element tag name.
        let tag = parse_root_tag(xml)?;
        let kind = NdiMetadataKind::from_element(tag);

        // Extract the mandatory `source` and `timecode` attributes.
        let source_name = parse_attr(xml, "source")
            .ok_or_else(|| MetaError::MissingField("source".to_string()))?;
        let timecode_str = parse_attr(xml, "timecode")
            .ok_or_else(|| MetaError::MissingField("timecode".to_string()))?;
        let timecode = timecode_str
            .parse::<u64>()
            .map_err(|e| MetaError::ParseError(format!("timecode: {e}")))?;

        let payload = match &kind {
            NdiMetadataKind::PtzPosition => {
                let pan = parse_f32_attr(xml, "pan")?;
                let tilt = parse_f32_attr(xml, "tilt")?;
                let zoom = parse_f32_attr(xml, "zoom")?;
                let focus = parse_attr(xml, "focus")
                    .map(|v| {
                        v.parse::<f32>()
                            .map_err(|e| MetaError::ParseError(format!("focus: {e}")))
                    })
                    .transpose()?;
                MetadataPayload::Ptz(PtzPayload {
                    pan,
                    tilt,
                    zoom,
                    focus,
                })
            }
            NdiMetadataKind::TallyState => {
                let program_str = parse_attr(xml, "program")
                    .ok_or_else(|| MetaError::MissingField("program".to_string()))?;
                let preview_str = parse_attr(xml, "preview")
                    .ok_or_else(|| MetaError::MissingField("preview".to_string()))?;
                MetadataPayload::Tally(TallyPayload {
                    program: program_str == "true",
                    preview: preview_str == "true",
                })
            }
            NdiMetadataKind::Connection | NdiMetadataKind::Custom(_) => {
                // Extract inner content between the root tags, if any.
                let inner = parse_inner_content(xml, tag);
                MetadataPayload::Raw(inner)
            }
        };

        Ok(NdiMetadataFrame {
            kind,
            payload,
            source_name,
            timecode,
        })
    }
}

// ── XML helpers ───────────────────────────────────────────────────────────────

/// Extract the root element tag name from an XML string.
fn parse_root_tag(xml: &str) -> Result<&str, MetaError> {
    let start = xml
        .find('<')
        .ok_or_else(|| MetaError::ParseError("no opening '<'".to_string()))?;
    let rest = &xml[start + 1..];
    // Tag ends at the first whitespace, '>' or '/'
    let end = rest
        .find(|c: char| c.is_ascii_whitespace() || c == '>' || c == '/')
        .ok_or_else(|| MetaError::ParseError("unclosed root element".to_string()))?;
    let tag = &rest[..end];
    if tag.is_empty() {
        return Err(MetaError::ParseError("empty root tag".to_string()));
    }
    Ok(tag)
}

/// Extract the string value of the named attribute from a simple XML element.
///
/// Handles both single-quoted and double-quoted attribute values.
fn parse_attr(xml: &str, attr: &str) -> Option<String> {
    // Look for `attr="value"` or `attr='value'`
    let needle_dq = format!(r#"{attr}=""#);
    let needle_sq = format!("{attr}='");

    if let Some(pos) = xml.find(&needle_dq) {
        let after = &xml[pos + needle_dq.len()..];
        let end = after.find('"')?;
        return Some(xml_unescape(&after[..end]));
    }
    if let Some(pos) = xml.find(&needle_sq) {
        let after = &xml[pos + needle_sq.len()..];
        let end = after.find('\'')?;
        return Some(xml_unescape(&after[..end]));
    }
    None
}

/// Parse a required f32 attribute, returning [`MetaError::MissingField`] or
/// [`MetaError::ParseError`] as appropriate.
fn parse_f32_attr(xml: &str, attr: &str) -> Result<f32, MetaError> {
    let raw = parse_attr(xml, attr)
        .ok_or_else(|| MetaError::MissingField(attr.to_string()))?;
    raw.parse::<f32>()
        .map_err(|e| MetaError::ParseError(format!("{attr}: {e}")))
}

/// Extract inner text content between `<tag …>…</tag>`.
///
/// Returns an empty string for self-closing elements.
fn parse_inner_content(xml: &str, tag: &str) -> String {
    let open_end = match xml.find('>') {
        Some(pos) => {
            // Self-closing: no inner content.
            if xml.as_bytes().get(pos.saturating_sub(1)) == Some(&b'/') {
                return String::new();
            }
            pos + 1
        }
        None => return String::new(),
    };

    let close_tag = format!("</{tag}>");
    let inner_end = match xml[open_end..].find(&close_tag) {
        Some(pos) => open_end + pos,
        None => return String::new(),
    };
    xml[open_end..inner_end].to_string()
}

/// Minimal XML attribute value escaping (only the characters that are illegal
/// inside double-quoted attribute values or text nodes).
fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
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

/// Reverse the minimal escape applied by [`xml_escape`].
fn xml_unescape(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── PTZ round-trip ────────────────────────────────────────────────────────

    #[test]
    fn ptz_roundtrip_without_focus() {
        let frame = NdiMetadataFrame {
            kind: NdiMetadataKind::PtzPosition,
            payload: MetadataPayload::Ptz(PtzPayload {
                pan: 0.5,
                tilt: -0.25,
                zoom: 0.75,
                focus: None,
            }),
            source_name: "Camera 1".to_string(),
            timecode: 9_000_000,
        };
        let xml = MetadataSerializer::to_xml(&frame);
        let decoded = MetadataSerializer::from_xml(&xml).expect("round-trip should succeed");
        assert_eq!(decoded.kind, frame.kind);
        assert_eq!(decoded.source_name, frame.source_name);
        assert_eq!(decoded.timecode, frame.timecode);
        if let MetadataPayload::Ptz(ref p) = decoded.payload {
            assert!((p.pan - 0.5).abs() < 1e-5);
            assert!((p.tilt - (-0.25)).abs() < 1e-5);
            assert!((p.zoom - 0.75).abs() < 1e-5);
            assert!(p.focus.is_none());
        } else {
            panic!("expected PTZ payload");
        }
    }

    #[test]
    fn ptz_roundtrip_with_focus() {
        let frame = NdiMetadataFrame {
            kind: NdiMetadataKind::PtzPosition,
            payload: MetadataPayload::Ptz(PtzPayload {
                pan: -1.0,
                tilt: 1.0,
                zoom: 0.0,
                focus: Some(0.333),
            }),
            source_name: "Camera 2".to_string(),
            timecode: 1_800_000,
        };
        let xml = MetadataSerializer::to_xml(&frame);
        let decoded = MetadataSerializer::from_xml(&xml).expect("round-trip should succeed");
        if let MetadataPayload::Ptz(ref p) = decoded.payload {
            assert!(p.focus.is_some());
            assert!((p.focus.expect("focus present") - 0.333).abs() < 1e-4);
        } else {
            panic!("expected PTZ payload");
        }
    }

    // ── Tally round-trip ──────────────────────────────────────────────────────

    #[test]
    fn tally_roundtrip_program_on() {
        let frame = NdiMetadataFrame {
            kind: NdiMetadataKind::TallyState,
            payload: MetadataPayload::Tally(TallyPayload {
                program: true,
                preview: false,
            }),
            source_name: "Switcher Output 1".to_string(),
            timecode: 0,
        };
        let xml = MetadataSerializer::to_xml(&frame);
        let decoded = MetadataSerializer::from_xml(&xml).expect("round-trip should succeed");
        if let MetadataPayload::Tally(ref t) = decoded.payload {
            assert!(t.program);
            assert!(!t.preview);
        } else {
            panic!("expected Tally payload");
        }
    }

    #[test]
    fn tally_roundtrip_preview_on() {
        let frame = NdiMetadataFrame {
            kind: NdiMetadataKind::TallyState,
            payload: MetadataPayload::Tally(TallyPayload {
                program: false,
                preview: true,
            }),
            source_name: "Preview Feed".to_string(),
            timecode: 42,
        };
        let xml = MetadataSerializer::to_xml(&frame);
        let decoded = MetadataSerializer::from_xml(&xml).expect("round-trip should succeed");
        if let MetadataPayload::Tally(ref t) = decoded.payload {
            assert!(!t.program);
            assert!(t.preview);
        } else {
            panic!("expected Tally payload");
        }
    }

    // ── Unknown kind → Custom ──────────────────────────────────────────────────

    #[test]
    fn unknown_element_becomes_custom() {
        let xml = r#"<my_custom_tag source="TestSrc" timecode="999">some content</my_custom_tag>"#;
        let decoded = MetadataSerializer::from_xml(xml).expect("should parse custom tag");
        assert_eq!(
            decoded.kind,
            NdiMetadataKind::Custom("my_custom_tag".to_string())
        );
        assert_eq!(decoded.source_name, "TestSrc");
        assert_eq!(decoded.timecode, 999);
        if let MetadataPayload::Raw(inner) = &decoded.payload {
            assert_eq!(inner, "some content");
        } else {
            panic!("expected Raw payload");
        }
    }

    #[test]
    fn connection_kind_parses_correctly() {
        let xml = r#"<ndi_connection source="Studio PC" timecode="12345"><info>test</info></ndi_connection>"#;
        let decoded = MetadataSerializer::from_xml(xml).expect("should parse connection");
        assert_eq!(decoded.kind, NdiMetadataKind::Connection);
        assert_eq!(decoded.source_name, "Studio PC");
    }

    // ── Missing field errors ──────────────────────────────────────────────────

    #[test]
    fn missing_pan_returns_missing_field_error() {
        // ndi_ptz without pan attribute
        let xml = r#"<ndi_ptz source="cam" timecode="0" tilt="0.0" zoom="0.0"/>"#;
        let result = MetadataSerializer::from_xml(xml);
        assert!(matches!(result, Err(MetaError::MissingField(ref f)) if f == "pan"));
    }

    #[test]
    fn missing_source_returns_missing_field_error() {
        let xml = r#"<ndi_tally timecode="0" program="true" preview="false"/>"#;
        let result = MetadataSerializer::from_xml(xml);
        assert!(matches!(result, Err(MetaError::MissingField(ref f)) if f == "source"));
    }

    #[test]
    fn missing_timecode_returns_missing_field_error() {
        let xml = r#"<ndi_tally source="cam" program="true" preview="false"/>"#;
        let result = MetadataSerializer::from_xml(xml);
        assert!(matches!(result, Err(MetaError::MissingField(ref f)) if f == "timecode"));
    }

    #[test]
    fn missing_program_returns_missing_field_error() {
        let xml = r#"<ndi_tally source="cam" timecode="0" preview="false"/>"#;
        let result = MetadataSerializer::from_xml(xml);
        assert!(matches!(result, Err(MetaError::MissingField(ref f)) if f == "program"));
    }

    // ── XML escaping round-trip ───────────────────────────────────────────────

    #[test]
    fn source_name_with_special_chars_round_trips() {
        let frame = NdiMetadataFrame {
            kind: NdiMetadataKind::TallyState,
            payload: MetadataPayload::Tally(TallyPayload {
                program: true,
                preview: false,
            }),
            source_name: r#"Studio "A" & <Live>"#.to_string(),
            timecode: 1,
        };
        let xml = MetadataSerializer::to_xml(&frame);
        let decoded = MetadataSerializer::from_xml(&xml).expect("should round-trip special chars");
        assert_eq!(decoded.source_name, frame.source_name);
    }

    // ── NdiMetadataKind helpers ───────────────────────────────────────────────

    #[test]
    fn element_name_round_trips() {
        for kind in [
            NdiMetadataKind::PtzPosition,
            NdiMetadataKind::TallyState,
            NdiMetadataKind::Connection,
            NdiMetadataKind::Custom("foo_bar".to_string()),
        ] {
            let name = kind.element_name();
            let parsed = NdiMetadataKind::from_element(name);
            assert_eq!(parsed, kind);
        }
    }
}
