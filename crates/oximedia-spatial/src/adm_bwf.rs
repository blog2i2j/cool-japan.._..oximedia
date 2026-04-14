//! ADM BWF (Broadcast Wave Format) metadata I/O for spatial audio.
//!
//! This module provides a simplified ADM (Audio Definition Model) metadata
//! representation suitable for embedding in BWF `axml` chunks, as defined
//! in ITU-R BS.2076.  The API is intentionally minimal — it covers the most
//! common use-case of describing a set of positioned audio objects and pack
//! formats, without pulling in an external XML crate.
//!
//! # Overview
//!
//! An [`AdmMetadata`] document contains:
//! - **[`AdmAudioObject`]**: a positioned, timed audio object (azimuth-free
//!   Cartesian x/y/z position, gain, start and duration in ms).
//! - **[`AdmAudioPack`]**: a pack format referencing a list of object IDs.
//!
//! Serialisation to ADM XML is provided by [`AdmMetadata::to_xml`]; a minimal
//! parser for the `<audioObject>` element is provided by
//! [`AdmMetadata::from_minimal_xml`].
//!
//! # Example
//!
//! ```rust
//! use oximedia_spatial::adm_bwf::{AdmAudioObject, AdmAudioPack, AdmMetadata};
//!
//! let obj = AdmAudioObject {
//!     id: "AO_1001".to_string(),
//!     name: "Dialogue".to_string(),
//!     start_ms: 0,
//!     duration_ms: 5000,
//!     gain: 1.0,
//!     position: (0.0, 1.0, 0.0),
//! };
//! let pack = AdmAudioPack {
//!     id: "AP_1001".to_string(),
//!     type_label: "Objects".to_string(),
//!     object_refs: vec!["AO_1001".to_string()],
//! };
//! let meta = AdmMetadata { objects: vec![obj], packs: vec![pack], num_channels: 1 };
//! let xml = meta.to_xml();
//! assert!(xml.contains("AO_1001"));
//! ```

// ─── Types ────────────────────────────────────────────────────────────────────

/// A single ADM `audioObject` element.
///
/// Positions follow the ADM Cartesian convention:
/// - x ∈ [−1, 1]: lateral (−1 = left, +1 = right)
/// - y ∈ [−1, 1]: depth  (−1 = back, +1 = front)
/// - z ∈ [−1, 1]: height (−1 = bottom, +1 = top)
#[derive(Debug, Clone, PartialEq)]
pub struct AdmAudioObject {
    /// Unique ADM `audioObjectID` (e.g. `"AO_1001"`).
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Absolute start time in milliseconds.
    pub start_ms: u64,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Linear gain applied to the object (1.0 = unity).
    pub gain: f32,
    /// Cartesian position `(x, y, z)`.
    pub position: (f32, f32, f32),
}

/// A single ADM `audioPackFormat` element referencing a list of object IDs.
#[derive(Debug, Clone, PartialEq)]
pub struct AdmAudioPack {
    /// Unique `audioPackFormatID` (e.g. `"AP_1001"`).
    pub id: String,
    /// `typeLabel` attribute (e.g. `"Objects"`, `"DirectSpeakers"`, `"HOA"`).
    pub type_label: String,
    /// `audioObjectIDRef` entries — the IDs of objects belonging to this pack.
    pub object_refs: Vec<String>,
}

/// Top-level ADM metadata container.
#[derive(Debug, Clone, PartialEq)]
pub struct AdmMetadata {
    /// All audio objects in the document.
    pub objects: Vec<AdmAudioObject>,
    /// All audio pack formats in the document.
    pub packs: Vec<AdmAudioPack>,
    /// Total number of audio channels described by this metadata.
    pub num_channels: u32,
}

// ─── AdmAudioObject ───────────────────────────────────────────────────────────

impl AdmAudioObject {
    /// Serialise this object to an ADM XML `<audioObject>` fragment.
    fn to_xml_fragment(&self) -> String {
        let (x, y, z) = self.position;
        format!(
            r#"    <audioObject audioObjectID="{id}" audioObjectName="{name}" start="{start}" duration="{dur}" gain="{gain:.6}">
      <audioBlockFormat cartesian="1">
        <position coordinate="X">{x:.6}</position>
        <position coordinate="Y">{y:.6}</position>
        <position coordinate="Z">{z:.6}</position>
      </audioBlockFormat>
    </audioObject>"#,
            id = escape_xml(&self.id),
            name = escape_xml(&self.name),
            start = self.start_ms,
            dur = self.duration_ms,
            gain = self.gain,
            x = x,
            y = y,
            z = z,
        )
    }
}

// ─── AdmAudioPack ─────────────────────────────────────────────────────────────

impl AdmAudioPack {
    /// Serialise this pack to an ADM XML `<audioPackFormat>` fragment.
    fn to_xml_fragment(&self) -> String {
        let refs: String = self
            .object_refs
            .iter()
            .map(|r| {
                format!(
                    "      <audioObjectIDRef>{}</audioObjectIDRef>\n",
                    escape_xml(r)
                )
            })
            .collect();
        format!(
            r#"    <audioPackFormat audioPackFormatID="{id}" typeLabel="{tl}">
{refs}    </audioPackFormat>"#,
            id = escape_xml(&self.id),
            tl = escape_xml(&self.type_label),
            refs = refs,
        )
    }
}

// ─── AdmMetadata ─────────────────────────────────────────────────────────────

impl AdmMetadata {
    /// Produce an ADM XML document containing `<audioObject>` and
    /// `<audioPackFormat>` elements for every object and pack in `self`.
    ///
    /// The output is valid ADM XML wrapped in an `<ebuCoreMain>` / `<coreMetadata>`
    /// / `<format>` / `<audioFormatExtended>` hierarchy as required by ITU-R
    /// BS.2076-2.
    pub fn to_xml(&self) -> String {
        let objects_xml: String = self
            .objects
            .iter()
            .map(|o| o.to_xml_fragment())
            .collect::<Vec<_>>()
            .join("\n");

        let packs_xml: String = self
            .packs
            .iter()
            .map(|p| p.to_xml_fragment())
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<ebuCoreMain xmlns="urn:ebu:metadata-schema:ebuCore_2015"
             xmlns:adm="urn:itu:bs:2076:2:adm">
  <coreMetadata>
    <format>
      <audioFormatExtended numChannels="{nc}">
{objs}
{packs}
      </audioFormatExtended>
    </format>
  </coreMetadata>
</ebuCoreMain>"#,
            nc = self.num_channels,
            objs = objects_xml,
            packs = packs_xml,
        )
    }

    /// Parse a minimal subset of ADM XML to reconstruct `AdmMetadata`.
    ///
    /// This parser handles only `<audioObject>` elements with the following
    /// attributes / child elements:
    /// - `audioObjectID` (attribute)
    /// - `audioObjectName` (attribute)
    /// - `start` (attribute, milliseconds, optional — defaults to 0)
    /// - `duration` (attribute, milliseconds, optional — defaults to 0)
    /// - `gain` (attribute, linear, optional — defaults to 1.0)
    /// - Cartesian `<position coordinate="X|Y|Z">value</position>` children
    ///
    /// `<audioPackFormat>` elements are **not** parsed — [`AdmMetadata::packs`]
    /// will be empty.  [`AdmMetadata::num_channels`] is set to the number of
    /// objects found.
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` if any required attribute is missing or cannot be
    /// parsed, or if the XML is structurally malformed.
    pub fn from_minimal_xml(xml: &str) -> Result<AdmMetadata, String> {
        let mut objects: Vec<AdmAudioObject> = Vec::new();

        // Find each <audioObject …> … </audioObject> block.
        let mut remaining = xml;
        while let Some(tag_start) = find_tag_start(remaining, "audioObject") {
            // Advance past the opening tag boundary.
            remaining = &remaining[tag_start..];

            // Find the end of the opening tag.
            let tag_end = remaining
                .find('>')
                .ok_or_else(|| "Malformed <audioObject>: missing closing '>'".to_string())?;
            let opening = &remaining[..tag_end + 1];

            // Check for self-closing tag; if so, skip.
            let self_closing = opening.trim_end_matches('>').ends_with('/');

            // Extract attributes from the opening tag.
            let id = extract_attr(opening, "audioObjectID")?;
            let name = extract_attr(opening, "audioObjectName").unwrap_or_default();
            let start_ms: u64 = extract_attr(opening, "start")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            let duration_ms: u64 = extract_attr(opening, "duration")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            let gain: f32 = extract_attr(opening, "gain")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1.0);

            // Find the inner content between the opening and closing tags.
            let (x, y, z) = if self_closing {
                (0.0_f32, 0.0_f32, 0.0_f32)
            } else {
                let close_tag = "</audioObject>";
                let inner_end = remaining[tag_end + 1..]
                    .find(close_tag)
                    .ok_or_else(|| format!("Missing </audioObject> for id={id}"))?;
                let inner = &remaining[tag_end + 1..tag_end + 1 + inner_end];
                let x = parse_position(inner, "X").unwrap_or(0.0);
                let y = parse_position(inner, "Y").unwrap_or(0.0);
                let z = parse_position(inner, "Z").unwrap_or(0.0);

                // Advance past this element.
                remaining = &remaining[tag_end + 1 + inner_end + close_tag.len()..];
                (x, y, z)
            };

            if self_closing {
                remaining = &remaining[tag_end + 1..];
            }

            objects.push(AdmAudioObject {
                id,
                name,
                start_ms,
                duration_ms,
                gain,
                position: (x, y, z),
            });
        }

        let num_channels = objects.len() as u32;
        Ok(AdmMetadata {
            objects,
            packs: Vec::new(),
            num_channels,
        })
    }
}

// ─── XML helpers ──────────────────────────────────────────────────────────────

/// Escape the five predefined XML entities in `s`.
fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Return the byte offset of the first occurrence of `<tag_name` (with optional
/// space/newline after the name) in `haystack`, or `None` if not found.
fn find_tag_start(haystack: &str, tag_name: &str) -> Option<usize> {
    let search = format!("<{}", tag_name);
    let idx = haystack.find(&search)?;
    // Make sure the next character after the tag name is whitespace, '/', or '>'.
    let after = haystack.get(idx + search.len()..)?;
    let first_char = after.chars().next()?;
    if first_char.is_whitespace() || first_char == '/' || first_char == '>' {
        Some(idx)
    } else {
        // Could be a tag like <audioObjectIDRef>; skip past and continue.
        let rest = &haystack[idx + 1..];
        find_tag_start(rest, tag_name).map(|n| idx + 1 + n)
    }
}

/// Extract the value of a quoted XML attribute `name="value"` or `name='value'`
/// from a tag string `tag`.
fn extract_attr(tag: &str, name: &str) -> Result<String, String> {
    let key_dq = format!("{}=\"", name);
    let key_sq = format!("{}='", name);

    if let Some(start) = tag.find(&key_dq) {
        let rest = &tag[start + key_dq.len()..];
        let end = rest
            .find('"')
            .ok_or_else(|| format!("Unterminated attribute '{name}'"))?;
        Ok(rest[..end].to_string())
    } else if let Some(start) = tag.find(&key_sq) {
        let rest = &tag[start + key_sq.len()..];
        let end = rest
            .find('\'')
            .ok_or_else(|| format!("Unterminated attribute '{name}'"))?;
        Ok(rest[..end].to_string())
    } else {
        Err(format!("Attribute '{name}' not found"))
    }
}

/// Parse a `<position coordinate="axis">value</position>` element from `xml`
/// for the given `axis` label (`"X"`, `"Y"`, or `"Z"`).
fn parse_position(xml: &str, axis: &str) -> Option<f32> {
    // Find coordinate="X" (or Y or Z).
    let key_dq = format!("coordinate=\"{}\"", axis);
    let key_sq = format!("coordinate='{}'", axis);

    let attr_pos = xml.find(&key_dq).or_else(|| xml.find(&key_sq))?;
    // Find the closing '>' of the <position ...> opening tag.
    let tag_close = xml[attr_pos..].find('>')?;
    let after_open = &xml[attr_pos + tag_close + 1..];
    // Extract text content up to </position>.
    let close_tag = after_open.find("</position>")?;
    let value_str = after_open[..close_tag].trim();
    value_str.parse().ok()
}

// ─── AdmBwfWriter ─────────────────────────────────────────────────────────────

/// Builder API for constructing ADM BWF headers incrementally.
///
/// Collects audio objects and then serialises them to an ADM XML byte payload
/// suitable for embedding in a BWF `axml` chunk.
///
/// # Example
///
/// ```rust
/// use oximedia_spatial::adm_bwf::AdmBwfWriter;
///
/// let mut writer = AdmBwfWriter::new();
/// writer.add_object(1, 0.0, 1.0, 0.0);
/// writer.add_object(2, -0.5, 0.8, 0.2);
/// let bytes = writer.write_header();
/// assert!(!bytes.is_empty());
/// assert!(std::str::from_utf8(&bytes).unwrap().contains("AO_0001"));
/// ```
#[derive(Debug, Clone, Default)]
pub struct AdmBwfWriter {
    objects: Vec<AdmAudioObject>,
}

impl AdmBwfWriter {
    /// Create an empty writer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a positioned audio object.
    ///
    /// The object ID is generated as `"AO_{id:04}"` (e.g. `"AO_0001"` for
    /// `id = 1`).  The name defaults to `"Object {id}"`.  Start time and
    /// duration are set to 0 / unbounded; gain is 1.0 (unity).
    ///
    /// # Arguments
    ///
    /// * `id` — Numeric object identifier (used in the generated ID string).
    /// * `x`  — Cartesian X position (−1 = left, +1 = right).
    /// * `y`  — Cartesian Y position (−1 = back, +1 = front).
    /// * `z`  — Cartesian Z position (−1 = bottom, +1 = top).
    pub fn add_object(&mut self, id: u32, x: f32, y: f32, z: f32) {
        self.objects.push(AdmAudioObject {
            id: format!("AO_{id:04}"),
            name: format!("Object {id}"),
            start_ms: 0,
            duration_ms: 0,
            gain: 1.0,
            position: (x, y, z),
        });
    }

    /// Serialise all objects to an ADM XML byte payload.
    ///
    /// Returns the UTF-8 encoded XML as a [`Vec<u8>`], ready for embedding in
    /// a BWF `axml` chunk.  Returns an empty vector if no objects have been
    /// added.
    #[must_use]
    pub fn write_header(&self) -> Vec<u8> {
        if self.objects.is_empty() {
            return Vec::new();
        }
        let meta = AdmMetadata {
            objects: self.objects.clone(),
            packs: Vec::new(),
            num_channels: self.objects.len() as u32,
        };
        meta.to_xml().into_bytes()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_object(id: &str) -> AdmAudioObject {
        AdmAudioObject {
            id: id.to_string(),
            name: "Test Object".to_string(),
            start_ms: 0,
            duration_ms: 5000,
            gain: 1.0,
            position: (0.5, 1.0, -0.5),
        }
    }

    fn sample_pack(id: &str, obj_id: &str) -> AdmAudioPack {
        AdmAudioPack {
            id: id.to_string(),
            type_label: "Objects".to_string(),
            object_refs: vec![obj_id.to_string()],
        }
    }

    // ── to_xml ───────────────────────────────────────────────────────────────

    #[test]
    fn test_to_xml_contains_object_id() {
        let meta = AdmMetadata {
            objects: vec![sample_object("AO_1001")],
            packs: vec![],
            num_channels: 1,
        };
        let xml = meta.to_xml();
        assert!(xml.contains("AO_1001"), "XML must contain object ID");
    }

    #[test]
    fn test_to_xml_contains_pack_id() {
        let meta = AdmMetadata {
            objects: vec![sample_object("AO_1001")],
            packs: vec![sample_pack("AP_1001", "AO_1001")],
            num_channels: 1,
        };
        let xml = meta.to_xml();
        assert!(xml.contains("AP_1001"), "XML must contain pack ID");
    }

    #[test]
    fn test_to_xml_contains_position_values() {
        let obj = AdmAudioObject {
            id: "AO_2001".to_string(),
            name: "Positioned".to_string(),
            start_ms: 100,
            duration_ms: 2000,
            gain: 0.8,
            position: (0.25, 0.75, 0.5),
        };
        let meta = AdmMetadata { objects: vec![obj], packs: vec![], num_channels: 1 };
        let xml = meta.to_xml();
        assert!(xml.contains("0.250000"), "XML should contain X=0.25");
        assert!(xml.contains("0.750000"), "XML should contain Y=0.75");
        assert!(xml.contains("0.500000"), "XML should contain Z=0.5");
    }

    #[test]
    fn test_to_xml_num_channels_attribute() {
        let meta = AdmMetadata {
            objects: vec![sample_object("AO_1")],
            packs: vec![],
            num_channels: 3,
        };
        let xml = meta.to_xml();
        assert!(xml.contains("numChannels=\"3\""), "XML must encode numChannels");
    }

    #[test]
    fn test_to_xml_multiple_objects() {
        let meta = AdmMetadata {
            objects: vec![sample_object("AO_1001"), sample_object("AO_1002")],
            packs: vec![],
            num_channels: 2,
        };
        let xml = meta.to_xml();
        assert!(xml.contains("AO_1001") && xml.contains("AO_1002"), "Both objects in XML");
    }

    // ── from_minimal_xml ─────────────────────────────────────────────────────

    #[test]
    fn test_roundtrip_minimal() {
        let meta = AdmMetadata {
            objects: vec![sample_object("AO_1001")],
            packs: vec![],
            num_channels: 1,
        };
        let xml = meta.to_xml();
        let parsed = AdmMetadata::from_minimal_xml(&xml).expect("parse ok");
        assert_eq!(parsed.objects.len(), 1);
        assert_eq!(parsed.objects[0].id, "AO_1001");
    }

    #[test]
    fn test_from_minimal_xml_position() {
        let meta = AdmMetadata {
            objects: vec![AdmAudioObject {
                id: "AO_3001".to_string(),
                name: "Choir".to_string(),
                start_ms: 0,
                duration_ms: 10000,
                gain: 1.0,
                position: (0.3, 0.6, 0.1),
            }],
            packs: vec![],
            num_channels: 1,
        };
        let xml = meta.to_xml();
        let parsed = AdmMetadata::from_minimal_xml(&xml).expect("parse ok");
        let (x, y, z) = parsed.objects[0].position;
        assert!((x - 0.3).abs() < 1e-4, "parsed X position close to 0.3, got {x}");
        assert!((y - 0.6).abs() < 1e-4, "parsed Y position close to 0.6, got {y}");
        assert!((z - 0.1).abs() < 1e-4, "parsed Z position close to 0.1, got {z}");
    }

    #[test]
    fn test_from_minimal_xml_multiple_objects() {
        let meta = AdmMetadata {
            objects: vec![sample_object("AO_1"), sample_object("AO_2"), sample_object("AO_3")],
            packs: vec![],
            num_channels: 3,
        };
        let xml = meta.to_xml();
        let parsed = AdmMetadata::from_minimal_xml(&xml).expect("parse ok");
        assert_eq!(parsed.objects.len(), 3, "Should parse 3 objects");
    }

    #[test]
    fn test_from_minimal_xml_missing_id_errors() {
        let bad_xml = r#"<?xml version="1.0"?>
<ebuCoreMain>
  <audioObject audioObjectName="Test">
    <audioBlockFormat cartesian="1">
      <position coordinate="X">0.5</position>
    </audioBlockFormat>
  </audioObject>
</ebuCoreMain>"#;
        let result = AdmMetadata::from_minimal_xml(bad_xml);
        assert!(result.is_err(), "Missing audioObjectID must produce an error");
    }

    #[test]
    fn test_from_minimal_xml_empty_document() {
        let result = AdmMetadata::from_minimal_xml("<ebuCoreMain></ebuCoreMain>");
        let meta = result.expect("empty document should be Ok");
        assert!(meta.objects.is_empty(), "No objects expected");
    }

    #[test]
    fn test_from_minimal_xml_gain_attribute() {
        let xml = r#"<?xml version="1.0"?>
<ebuCoreMain>
  <audioObject audioObjectID="AO_G" audioObjectName="Gain" start="0" duration="1000" gain="0.5">
    <audioBlockFormat cartesian="1">
      <position coordinate="X">0.0</position>
      <position coordinate="Y">1.0</position>
      <position coordinate="Z">0.0</position>
    </audioBlockFormat>
  </audioObject>
</ebuCoreMain>"#;
        let parsed = AdmMetadata::from_minimal_xml(xml).expect("parse ok");
        assert!((parsed.objects[0].gain - 0.5).abs() < 1e-5, "Gain attribute parsed correctly");
    }

    // ── XML escaping ─────────────────────────────────────────────────────────

    #[test]
    fn test_escape_xml_ampersand() {
        assert_eq!(escape_xml("a&b"), "a&amp;b");
    }

    #[test]
    fn test_escape_xml_quotes() {
        assert_eq!(escape_xml("say \"hello\""), "say &quot;hello&quot;");
    }

    #[test]
    fn test_escape_xml_clean_string_unchanged() {
        let s = "AudioObject_1001";
        assert_eq!(escape_xml(s), s);
    }

    // ── AdmBwfWriter ─────────────────────────────────────────────────────────

    #[test]
    fn test_writer_new_is_empty() {
        let writer = AdmBwfWriter::new();
        let bytes = writer.write_header();
        assert!(bytes.is_empty(), "Empty writer should produce empty header");
    }

    #[test]
    fn test_writer_add_object_and_write_header() {
        let mut writer = AdmBwfWriter::new();
        writer.add_object(1, 0.0, 1.0, 0.0);
        let bytes = writer.write_header();
        assert!(!bytes.is_empty());
        let xml = std::str::from_utf8(&bytes).expect("valid UTF-8");
        assert!(xml.contains("AO_0001"), "Header should contain generated object ID");
    }

    #[test]
    fn test_writer_multiple_objects() {
        let mut writer = AdmBwfWriter::new();
        writer.add_object(1, -0.5, 0.8, 0.0);
        writer.add_object(2, 0.5, 0.8, 0.0);
        writer.add_object(3, 0.0, 1.0, 0.5);
        let bytes = writer.write_header();
        let xml = std::str::from_utf8(&bytes).expect("valid UTF-8");
        assert!(xml.contains("AO_0001"));
        assert!(xml.contains("AO_0002"));
        assert!(xml.contains("AO_0003"));
        assert!(xml.contains("numChannels=\"3\""));
    }

    #[test]
    fn test_writer_position_values_in_xml() {
        let mut writer = AdmBwfWriter::new();
        writer.add_object(7, 0.25, 0.75, -0.5);
        let bytes = writer.write_header();
        let xml = std::str::from_utf8(&bytes).expect("valid UTF-8");
        assert!(xml.contains("0.250000"), "X position in header");
        assert!(xml.contains("0.750000"), "Y position in header");
        assert!(xml.contains("-0.500000"), "Z position in header");
    }
}
