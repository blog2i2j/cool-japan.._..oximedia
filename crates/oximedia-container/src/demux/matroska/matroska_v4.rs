//! Matroska v4 extension support.
//!
//! Implements parsing for Matroska v4-specific elements that extend the
//! baseline Matroska track description with Block Addition Mapping support.
//!
//! # Block Addition Mappings
//!
//! In Matroska v4 each track can carry zero or more `BlockAdditionMapping`
//! elements that describe auxiliary data channels attached to blocks.  Each
//! mapping has:
//!
//! | Field          | Element ID | Type   | Description |
//! |----------------|------------|--------|-------------|
//! | `id_name`      | 0x41A4     | String | Human-readable name |
//! | `id_type`      | 0x41E4     | UInt   | Numeric type identifier |
//! | `id_extra_data`| 0x41ED     | Binary | Codec-specific config |
//!
//! These are grouped under a `BlockAdditionMapping` container element (0x41CB).
//!
//! # Reference
//!
//! - [Matroska spec §BlockAdditionMapping](https://www.matroska.org/technical/elements.html#BlockAdditionMapping)

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]

use super::parser::MatroskaParser;
use oximedia_core::OxiResult;

// ─── Matroska v4 element IDs ─────────────────────────────────────────────────

/// Element ID constants for Matroska v4 BlockAdditionMapping elements.
pub mod v4_element_id {
    /// Container element for a single BlockAdditionMapping.
    pub const BLOCK_ADDITION_MAPPING: u32 = 0x41CB;
    /// Human-readable name for this block addition mapping.
    pub const BLOCK_ADD_ID_NAME: u32 = 0x41A4;
    /// Numeric type identifier for this block addition mapping.
    pub const BLOCK_ADD_ID_TYPE: u32 = 0x41E4;
    /// Codec-specific extra data for this block addition mapping.
    pub const BLOCK_ADD_ID_EXTRA_DATA: u32 = 0x41ED;
}

// ─── BlockAdditionMapping ────────────────────────────────────────────────────

/// A single `BlockAdditionMapping` element from a Matroska v4 track.
///
/// Block addition mappings describe the semantics of `BlockAdditional` data
/// attached to blocks in the track.  Each track may carry multiple mappings
/// (typically at most one or two in practice).
#[derive(Debug, Clone, Default)]
pub struct BlockAdditionMapping {
    /// Human-readable name identifying the mapping (e.g., `"alpha"`, `"depth"`).
    pub id_name: Option<String>,
    /// Numeric type identifier that specifies the semantics of the additional
    /// data (see the Matroska Block Addition Mapping Registry).
    pub id_type: Option<u64>,
    /// Codec-specific configuration payload for this mapping.  May be empty.
    pub id_extra_data: Vec<u8>,
}

impl BlockAdditionMapping {
    /// Creates a new empty `BlockAdditionMapping`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` if this mapping has a name.
    #[must_use]
    pub fn has_name(&self) -> bool {
        self.id_name.is_some()
    }

    /// Returns `true` if this mapping has a numeric type.
    #[must_use]
    pub fn has_type(&self) -> bool {
        self.id_type.is_some()
    }

    /// Returns `true` if this mapping carries extra codec-specific data.
    #[must_use]
    pub fn has_extra_data(&self) -> bool {
        !self.id_extra_data.is_empty()
    }
}

// ─── MatroskaTrackV4 ─────────────────────────────────────────────────────────

/// Matroska v4 extensions for a track entry.
///
/// Carries the zero or more [`BlockAdditionMapping`] elements attached to a
/// single Matroska track.  Attach this to the corresponding
/// [`super::types::TrackEntry`] after parsing.
///
/// # Example
///
/// ```ignore
/// use oximedia_container::demux::matroska::matroska_v4::{MatroskaTrackV4, BlockAdditionMapping};
///
/// let mut v4 = MatroskaTrackV4::new();
/// let mut mapping = BlockAdditionMapping::new();
/// mapping.id_name = Some("alpha".to_string());
/// mapping.id_type = Some(1);
/// v4.addition_mappings.push(mapping);
///
/// assert_eq!(v4.mapping_count(), 1);
/// assert_eq!(v4.find_by_name("alpha").unwrap().id_type, Some(1));
/// ```
#[derive(Debug, Clone, Default)]
pub struct MatroskaTrackV4 {
    /// All `BlockAdditionMapping` elements found in this track.
    pub addition_mappings: Vec<BlockAdditionMapping>,
}

impl MatroskaTrackV4 {
    /// Creates a new empty `MatroskaTrackV4`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of block addition mappings.
    #[must_use]
    pub fn mapping_count(&self) -> usize {
        self.addition_mappings.len()
    }

    /// Finds the first mapping with the given name, or `None`.
    ///
    /// Comparison is case-sensitive and exact.
    #[must_use]
    pub fn find_by_name(&self, name: &str) -> Option<&BlockAdditionMapping> {
        self.addition_mappings
            .iter()
            .find(|m| m.id_name.as_deref() == Some(name))
    }

    /// Finds the first mapping with the given numeric type, or `None`.
    #[must_use]
    pub fn find_by_type(&self, id_type: u64) -> Option<&BlockAdditionMapping> {
        self.addition_mappings
            .iter()
            .find(|m| m.id_type == Some(id_type))
    }
}

// ─── parse_block_addition_mapping ────────────────────────────────────────────

/// Parses a single `BlockAdditionMapping` EBML element from `data`.
///
/// `data` must be the raw content bytes of a `BlockAdditionMapping` (0x41CB)
/// element, **excluding** the element header (size + ID bytes).
///
/// # Errors
///
/// Returns an [`OxiError::Parse`] if a child element header cannot be decoded.
/// Unknown child elements are silently skipped (forward-compatible).
pub fn parse_block_addition_mapping(data: &[u8]) -> OxiResult<BlockAdditionMapping> {
    let mut parser = MatroskaParser::new(data);
    let end_pos = data.len();
    let mut mapping = BlockAdditionMapping::new();

    while parser.position() < end_pos && !parser.is_eof() {
        let element = parser.read_element()?;
        let elem_size = element.size as usize;

        match element.id {
            v4_element_id::BLOCK_ADD_ID_NAME => {
                mapping.id_name = Some(parser.read_string(elem_size)?);
            }
            v4_element_id::BLOCK_ADD_ID_TYPE => {
                mapping.id_type = Some(parser.read_uint(elem_size)?);
            }
            v4_element_id::BLOCK_ADD_ID_EXTRA_DATA => {
                mapping.id_extra_data = parser.read_binary(elem_size)?;
            }
            _ => {
                // Unknown element — skip (forward compatible)
                parser.skip(elem_size);
            }
        }
    }

    Ok(mapping)
}

/// Parses all `BlockAdditionMapping` children inside a containing EBML
/// element whose raw content (without the outer element's header) is `data`.
///
/// Iterates the content looking for 0x41CB child elements and delegates each
/// to [`parse_block_addition_mapping`].  Returns a [`MatroskaTrackV4`] that
/// holds all discovered mappings.
///
/// # Errors
///
/// Propagates parsing errors from child element header reads.
pub fn parse_all_block_addition_mappings(data: &[u8], size: u64) -> OxiResult<MatroskaTrackV4> {
    let mut parser = MatroskaParser::new(data);
    let end_pos = size as usize;
    let mut v4 = MatroskaTrackV4::new();

    while parser.position() < end_pos && !parser.is_eof() {
        let element = parser.read_element()?;
        let elem_size = element.size as usize;

        if element.id == v4_element_id::BLOCK_ADDITION_MAPPING {
            let mapping_data = parser.read_data(elem_size)?;
            match parse_block_addition_mapping(mapping_data) {
                Ok(m) => v4.addition_mappings.push(m),
                Err(_) => {
                    // On parse error skip this mapping (resilient parsing)
                }
            }
        } else {
            parser.skip(elem_size);
        }
    }

    Ok(v4)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: encode a minimal EBML element (ID + size + content).
    //
    // EBML element IDs already carry their length-marker bits embedded in the
    // value itself.  For example, 0x41A4 is a 2-byte ID because the high byte
    // is 0x41 (binary 01xxxxxx — 1 leading zero → 2-byte width).
    // We therefore just split the u32 into the minimum number of big-endian
    // bytes needed to represent it as a proper EBML ID.
    fn encode_element(id: u32, content: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        // Encode ID as raw big-endian bytes (width determined by leading-zero
        // count of the most-significant byte).
        if id <= 0xFF {
            out.push(id as u8);
        } else if id <= 0xFFFF {
            out.push((id >> 8) as u8);
            out.push(id as u8);
        } else if id <= 0xFF_FFFF {
            out.push((id >> 16) as u8);
            out.push((id >> 8) as u8);
            out.push(id as u8);
        } else {
            out.push((id >> 24) as u8);
            out.push((id >> 16) as u8);
            out.push((id >> 8) as u8);
            out.push(id as u8);
        }
        // Encode size as a VINT (marker bit included in the first byte).
        let len = content.len();
        if len < 0x7F {
            out.push((len as u8) | 0x80);
        } else if len < 0x3FFF {
            out.push(0x40 | (len >> 8) as u8);
            out.push(len as u8);
        } else {
            out.push(0x20 | (len >> 16) as u8);
            out.push((len >> 8) as u8);
            out.push(len as u8);
        }
        out.extend_from_slice(content);
        out
    }

    fn encode_string_element(id: u32, s: &str) -> Vec<u8> {
        encode_element(id, s.as_bytes())
    }

    fn encode_uint_element(id: u32, v: u64) -> Vec<u8> {
        // Use minimum number of bytes
        let bytes = if v == 0 {
            vec![0u8]
        } else {
            let mut b = v.to_be_bytes().to_vec();
            while b.len() > 1 && b[0] == 0 {
                b.remove(0);
            }
            b
        };
        encode_element(id, &bytes)
    }

    #[test]
    fn test_block_addition_mapping_default() {
        let m = BlockAdditionMapping::default();
        assert!(m.id_name.is_none());
        assert!(m.id_type.is_none());
        assert!(m.id_extra_data.is_empty());
        assert!(!m.has_name());
        assert!(!m.has_type());
        assert!(!m.has_extra_data());
    }

    #[test]
    fn test_parse_block_addition_mapping_empty() {
        let mapping = parse_block_addition_mapping(&[]).expect("should parse empty");
        assert!(mapping.id_name.is_none());
        assert!(mapping.id_type.is_none());
        assert!(mapping.id_extra_data.is_empty());
    }

    #[test]
    fn test_parse_block_addition_mapping_name_only() {
        let mut data = Vec::new();
        data.extend(encode_string_element(
            v4_element_id::BLOCK_ADD_ID_NAME,
            "alpha",
        ));
        let mapping = parse_block_addition_mapping(&data).expect("should parse");
        assert_eq!(mapping.id_name.as_deref(), Some("alpha"));
        assert!(mapping.id_type.is_none());
        assert!(mapping.id_extra_data.is_empty());
    }

    #[test]
    fn test_parse_block_addition_mapping_with_type() {
        let mut data = Vec::new();
        data.extend(encode_string_element(
            v4_element_id::BLOCK_ADD_ID_NAME,
            "depth",
        ));
        data.extend(encode_uint_element(v4_element_id::BLOCK_ADD_ID_TYPE, 2));
        let mapping = parse_block_addition_mapping(&data).expect("should parse");
        assert_eq!(mapping.id_name.as_deref(), Some("depth"));
        assert_eq!(mapping.id_type, Some(2));
    }

    #[test]
    fn test_parse_block_addition_mapping_full() {
        let extra = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let mut data = Vec::new();
        data.extend(encode_string_element(
            v4_element_id::BLOCK_ADD_ID_NAME,
            "test",
        ));
        data.extend(encode_uint_element(v4_element_id::BLOCK_ADD_ID_TYPE, 42));
        data.extend(encode_element(
            v4_element_id::BLOCK_ADD_ID_EXTRA_DATA,
            &extra,
        ));
        let mapping = parse_block_addition_mapping(&data).expect("should parse");
        assert_eq!(mapping.id_name.as_deref(), Some("test"));
        assert_eq!(mapping.id_type, Some(42));
        assert_eq!(mapping.id_extra_data, extra);
        assert!(mapping.has_name());
        assert!(mapping.has_type());
        assert!(mapping.has_extra_data());
    }

    #[test]
    fn test_matroska_track_v4_find_by_name() {
        let mut v4 = MatroskaTrackV4::new();
        let mut m1 = BlockAdditionMapping::new();
        m1.id_name = Some("alpha".to_string());
        m1.id_type = Some(1);
        let mut m2 = BlockAdditionMapping::new();
        m2.id_name = Some("depth".to_string());
        m2.id_type = Some(2);
        v4.addition_mappings.push(m1);
        v4.addition_mappings.push(m2);

        let found = v4.find_by_name("alpha").expect("should find alpha");
        assert_eq!(found.id_type, Some(1));

        let not_found = v4.find_by_name("nonexistent");
        assert!(not_found.is_none());
    }

    #[test]
    fn test_matroska_track_v4_find_by_type() {
        let mut v4 = MatroskaTrackV4::new();
        let mut m = BlockAdditionMapping::new();
        m.id_type = Some(7);
        m.id_name = Some("special".to_string());
        v4.addition_mappings.push(m);

        let found = v4.find_by_type(7).expect("should find by type");
        assert_eq!(found.id_name.as_deref(), Some("special"));

        assert!(v4.find_by_type(99).is_none());
    }

    #[test]
    fn test_matroska_track_v4_mapping_count() {
        let mut v4 = MatroskaTrackV4::new();
        assert_eq!(v4.mapping_count(), 0);
        v4.addition_mappings.push(BlockAdditionMapping::new());
        v4.addition_mappings.push(BlockAdditionMapping::new());
        assert_eq!(v4.mapping_count(), 2);
    }

    #[test]
    fn test_parse_all_block_addition_mappings() {
        // Build a fake containing-element body with two BlockAdditionMapping children.
        let m1_data = {
            let mut d = Vec::new();
            d.extend(encode_string_element(
                v4_element_id::BLOCK_ADD_ID_NAME,
                "first",
            ));
            d.extend(encode_uint_element(v4_element_id::BLOCK_ADD_ID_TYPE, 1));
            d
        };
        let m2_data = {
            let mut d = Vec::new();
            d.extend(encode_string_element(
                v4_element_id::BLOCK_ADD_ID_NAME,
                "second",
            ));
            d.extend(encode_uint_element(v4_element_id::BLOCK_ADD_ID_TYPE, 2));
            d
        };

        let mut body = Vec::new();
        body.extend(encode_element(
            v4_element_id::BLOCK_ADDITION_MAPPING,
            &m1_data,
        ));
        body.extend(encode_element(
            v4_element_id::BLOCK_ADDITION_MAPPING,
            &m2_data,
        ));

        let size = body.len() as u64;
        let v4 = parse_all_block_addition_mappings(&body, size).expect("should parse");
        assert_eq!(v4.mapping_count(), 2);
        assert_eq!(
            v4.addition_mappings[0].id_name.as_deref(),
            Some("first")
        );
        assert_eq!(
            v4.addition_mappings[1].id_name.as_deref(),
            Some("second")
        );
    }

    #[test]
    fn test_v4_element_id_constants() {
        assert_eq!(v4_element_id::BLOCK_ADDITION_MAPPING, 0x41CB);
        assert_eq!(v4_element_id::BLOCK_ADD_ID_NAME, 0x41A4);
        assert_eq!(v4_element_id::BLOCK_ADD_ID_TYPE, 0x41E4);
        assert_eq!(v4_element_id::BLOCK_ADD_ID_EXTRA_DATA, 0x41ED);
    }
}
