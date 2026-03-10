//! AV1 Open Bitstream Unit (OBU) parsing structures.
//!
//! This module implements parsing of AV1 OBU headers and LEB128
//! variable-length integer encoding/decoding as specified in the
//! AV1 bitstream specification.

/// AV1 OBU (Open Bitstream Unit) type identifiers.
///
/// Each OBU has a 4-bit type field in its header identifying the content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ObuType {
    /// Sequence header OBU – contains codec configuration.
    SequenceHeader,
    /// Temporal delimiter OBU – marks the start of a temporal unit.
    TemporalDelimiter,
    /// Frame header OBU – per-frame coding parameters.
    FrameHeader,
    /// Tile group OBU – contains compressed tile data.
    TileGroup,
    /// Metadata OBU – auxiliary metadata (HDR, display mapping, …).
    Metadata,
    /// Frame OBU – combined frame header + tile group.
    Frame,
    /// Redundant frame header OBU.
    RedundantFrameHeader,
    /// Padding OBU – ignored by decoders.
    Padding,
}

impl ObuType {
    /// Returns the 4-bit numeric value used in the bitstream.
    #[allow(dead_code)]
    pub fn value(self) -> u8 {
        match self {
            Self::SequenceHeader => 1,
            Self::TemporalDelimiter => 2,
            Self::FrameHeader => 3,
            Self::TileGroup => 4,
            Self::Metadata => 5,
            Self::Frame => 6,
            Self::RedundantFrameHeader => 7,
            Self::Padding => 15,
        }
    }

    /// Converts a 4-bit bitstream value to an `ObuType`, returning `None`
    /// for reserved or unknown values.
    #[allow(dead_code)]
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::SequenceHeader),
            2 => Some(Self::TemporalDelimiter),
            3 => Some(Self::FrameHeader),
            4 => Some(Self::TileGroup),
            5 => Some(Self::Metadata),
            6 => Some(Self::Frame),
            7 => Some(Self::RedundantFrameHeader),
            15 => Some(Self::Padding),
            _ => None,
        }
    }

    /// Returns `true` if this OBU type carries coded frame data (tile
    /// data or a combined frame OBU).
    #[allow(dead_code)]
    pub fn is_frame_data(self) -> bool {
        matches!(self, Self::TileGroup | Self::Frame)
    }
}

/// Parsed AV1 OBU header (1 or 2 bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub struct ObuHeader {
    /// Parsed OBU type.
    pub obu_type: ObuType,
    /// Whether a 1-byte extension header follows the main header byte.
    pub extension_flag: bool,
    /// Whether the payload size is encoded as a LEB128 field.
    pub has_size_field: bool,
    /// Temporal layer ID (from extension byte, or 0).
    pub temporal_id: u8,
    /// Spatial layer ID (from extension byte, or 0).
    pub spatial_id: u8,
}

impl ObuHeader {
    /// Parses a single header byte (the extension byte is not handled here;
    /// use `parse_with_extension` when `extension_flag` is set).
    ///
    /// Returns `None` when the `forbidden_bit` (bit 7) is set or the OBU
    /// type value is unrecognised.
    #[allow(dead_code)]
    pub fn parse(byte: u8) -> Option<Self> {
        // bit 7 must be 0 (forbidden_bit)
        if byte & 0x80 != 0 {
            return None;
        }
        let obu_type_val = (byte >> 3) & 0x0F;
        let obu_type = ObuType::from_u8(obu_type_val)?;
        let extension_flag = (byte >> 2) & 1 == 1;
        let has_size_field = (byte >> 1) & 1 == 1;
        Some(Self {
            obu_type,
            extension_flag,
            has_size_field,
            temporal_id: 0,
            spatial_id: 0,
        })
    }

    /// Parses the main header byte followed by an optional extension byte.
    ///
    /// Returns `(header, bytes_consumed)` where `bytes_consumed` is 1 or 2.
    #[allow(dead_code)]
    pub fn parse_with_extension(data: &[u8]) -> Option<(Self, usize)> {
        if data.is_empty() {
            return None;
        }
        let mut header = Self::parse(data[0])?;
        let mut consumed = 1;
        if header.extension_flag {
            if data.len() < 2 {
                return None;
            }
            let ext = data[1];
            header.temporal_id = (ext >> 5) & 0x07;
            header.spatial_id = (ext >> 3) & 0x03;
            consumed = 2;
        }
        Some((header, consumed))
    }
}

/// A fully located OBU within a bitstream buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub struct ObuUnit {
    /// Parsed OBU header.
    pub header: ObuHeader,
    /// Payload size in bytes (excluding the header and size-field bytes).
    pub size: u32,
    /// Byte offset from the start of the containing buffer at which the
    /// payload begins.
    pub payload_offset: u32,
}

impl ObuUnit {
    /// Total byte span of this OBU (header + size field + payload).
    #[allow(dead_code)]
    pub fn total_size(&self) -> u32 {
        self.payload_offset + self.size
    }

    /// Returns `true` when the OBU is a sequence header.
    #[allow(dead_code)]
    pub fn is_sequence_header(&self) -> bool {
        self.header.obu_type == ObuType::SequenceHeader
    }
}

/// LEB128 unsigned variable-length integer codec.
pub struct Leb128;

impl Leb128 {
    /// Decodes a LEB128-encoded unsigned integer from `data`.
    ///
    /// Returns `Some((value, bytes_consumed))` on success, or `None` when
    /// the input is malformed (e.g., more than 8 continuation bytes, or the
    /// buffer ends prematurely).
    #[allow(dead_code)]
    pub fn decode(data: &[u8]) -> Option<(u64, usize)> {
        let mut value: u64 = 0;
        let mut shift = 0u32;
        for (i, &byte) in data.iter().enumerate() {
            if shift >= 63 && byte > 1 {
                // Would overflow u64
                return None;
            }
            value |= (u64::from(byte & 0x7F)) << shift;
            shift += 7;
            if byte & 0x80 == 0 {
                return Some((value, i + 1));
            }
            if shift >= 70 {
                // Exceeded maximum LEB128 length for u64
                return None;
            }
        }
        // Ran off the end of the buffer without finding a terminating byte.
        None
    }

    /// Encodes `value` as a LEB128 unsigned integer, returning the bytes.
    #[allow(dead_code)]
    pub fn encode(mut value: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let mut byte = (value & 0x7F) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if value == 0 {
                break;
            }
        }
        out
    }
}

/// Scans `data` for AV1 OBU headers, building a list of located `ObuUnit`s.
///
/// Parsing stops at the first unrecognised or malformed OBU.
#[allow(dead_code)]
pub fn parse_obu_headers(data: &[u8]) -> Vec<ObuUnit> {
    let mut units = Vec::new();
    let mut pos = 0usize;

    while pos < data.len() {
        // Parse the OBU header (1 or 2 bytes).
        let (header, header_len) = match ObuHeader::parse_with_extension(&data[pos..]) {
            Some(v) => v,
            None => break,
        };

        let header_end = pos + header_len;

        // Parse the optional LEB128 size field.
        let (payload_size, size_field_len) = if header.has_size_field {
            match Leb128::decode(&data[header_end..]) {
                Some((v, n)) => {
                    if v > u64::from(u32::MAX) {
                        break;
                    }
                    (v as u32, n)
                }
                None => break,
            }
        } else {
            // No size field: payload runs to end of buffer.
            let remaining = (data.len() - header_end) as u32;
            (remaining, 0)
        };

        let payload_offset = (header_end + size_field_len) as u32;

        // Verify the payload fits inside `data`.
        let payload_end = payload_offset as usize + payload_size as usize;
        if payload_end > data.len() {
            break;
        }

        units.push(ObuUnit {
            header,
            size: payload_size,
            payload_offset,
        });

        pos = payload_end;
    }

    units
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- ObuType tests ---

    #[test]
    fn obu_type_value_roundtrip() {
        let types = [
            ObuType::SequenceHeader,
            ObuType::TemporalDelimiter,
            ObuType::FrameHeader,
            ObuType::TileGroup,
            ObuType::Metadata,
            ObuType::Frame,
            ObuType::RedundantFrameHeader,
            ObuType::Padding,
        ];
        for t in types {
            let v = t.value();
            assert_eq!(ObuType::from_u8(v), Some(t));
        }
    }

    #[test]
    fn obu_type_unknown_returns_none() {
        assert!(ObuType::from_u8(0).is_none());
        assert!(ObuType::from_u8(8).is_none());
        assert!(ObuType::from_u8(14).is_none());
        assert!(ObuType::from_u8(255).is_none());
    }

    #[test]
    fn obu_type_is_frame_data() {
        assert!(ObuType::TileGroup.is_frame_data());
        assert!(ObuType::Frame.is_frame_data());
        assert!(!ObuType::SequenceHeader.is_frame_data());
        assert!(!ObuType::Padding.is_frame_data());
    }

    // --- ObuHeader tests ---

    #[test]
    fn obu_header_parse_sequence_header() {
        // forbidden=0, type=SequenceHeader(1), ext=0, has_size=1, reserved=0
        // bits: 0_0001_0_1_0 = 0x0A
        let byte = 0b0000_1010u8;
        let h = ObuHeader::parse(byte).expect("should succeed");
        assert_eq!(h.obu_type, ObuType::SequenceHeader);
        assert!(!h.extension_flag);
        assert!(h.has_size_field);
    }

    #[test]
    fn obu_header_forbidden_bit_returns_none() {
        // Forbidden bit (bit 7) set.
        let byte = 0b1000_1010u8;
        assert!(ObuHeader::parse(byte).is_none());
    }

    #[test]
    fn obu_header_parse_with_extension_no_ext() {
        // type=TemporalDelimiter(2), ext=0, has_size=1
        // bits: 0_0010_0_1_0 = 0x12
        let data = [0b0001_0010u8];
        let (h, consumed) = ObuHeader::parse_with_extension(&data).expect("should succeed");
        assert_eq!(h.obu_type, ObuType::TemporalDelimiter);
        assert_eq!(consumed, 1);
        assert_eq!(h.temporal_id, 0);
    }

    #[test]
    fn obu_header_parse_with_extension_has_ext() {
        // type=FrameHeader(3), ext=1, has_size=1
        // main byte: 0_0011_1_1_0 = 0x1E
        // ext  byte: temporal_id=2 (bits 7:5=010), spatial_id=1 (bits 4:3=01), reserved=000 => 0b010_01_000 = 0x48
        let data = [0b0001_1110u8, 0b0100_1000u8];
        let (h, consumed) = ObuHeader::parse_with_extension(&data).expect("should succeed");
        assert_eq!(h.obu_type, ObuType::FrameHeader);
        assert_eq!(consumed, 2);
        assert_eq!(h.temporal_id, 2);
        assert_eq!(h.spatial_id, 1);
    }

    // --- Leb128 tests ---

    #[test]
    fn leb128_encode_decode_zero() {
        let enc = Leb128::encode(0);
        assert_eq!(enc, vec![0x00]);
        let (val, n) = Leb128::decode(&enc).expect("should succeed");
        assert_eq!(val, 0);
        assert_eq!(n, 1);
    }

    #[test]
    fn leb128_encode_decode_small() {
        for v in [1u64, 127, 128, 255, 300, 16383, 16384] {
            let enc = Leb128::encode(v);
            let (decoded, _) = Leb128::decode(&enc).expect("should succeed");
            assert_eq!(decoded, v, "roundtrip failed for {v}");
        }
    }

    #[test]
    fn leb128_encode_decode_large() {
        let v = u64::MAX >> 1; // 0x7FFF_FFFF_FFFF_FFFF
        let enc = Leb128::encode(v);
        let (decoded, _) = Leb128::decode(&enc).expect("should succeed");
        assert_eq!(decoded, v);
    }

    #[test]
    fn leb128_decode_incomplete_returns_none() {
        // A continuation byte with no terminator.
        let data = [0x80u8, 0x80];
        assert!(Leb128::decode(&data).is_none());
    }

    #[test]
    fn leb128_decode_empty_returns_none() {
        assert!(Leb128::decode(&[]).is_none());
    }

    // --- parse_obu_headers tests ---

    #[test]
    fn parse_obu_headers_sequence_header() {
        // Craft a minimal sequence-header OBU:
        //   header byte: type=1, ext=0, has_size=1 → 0b0000_1010 = 0x0A
        //   size LEB128(3):  0x03
        //   payload: 3 dummy bytes
        let data = [0x0Au8, 0x03, 0xAA, 0xBB, 0xCC];
        let units = parse_obu_headers(&data);
        assert_eq!(units.len(), 1);
        assert!(units[0].is_sequence_header());
        assert_eq!(units[0].size, 3);
        assert_eq!(units[0].payload_offset, 2);
        assert_eq!(units[0].total_size(), 5);
    }

    #[test]
    fn parse_obu_headers_multiple_obus() {
        // Two OBUs back-to-back.
        // OBU 1: type=TemporalDelimiter(2), no ext, has_size, payload=0
        //   0b0001_0010 = 0x12, LEB128(0)=0x00
        // OBU 2: type=TileGroup(4), no ext, has_size, payload=2
        //   0b0010_0010 = 0x22, LEB128(2)=0x02, 0xDE, 0xAD
        let data = [0x12u8, 0x00, 0x22, 0x02, 0xDE, 0xAD];
        let units = parse_obu_headers(&data);
        assert_eq!(units.len(), 2);
        assert_eq!(units[0].header.obu_type, ObuType::TemporalDelimiter);
        assert_eq!(units[0].size, 0);
        assert_eq!(units[1].header.obu_type, ObuType::TileGroup);
        assert_eq!(units[1].size, 2);
        assert!(units[1].header.obu_type.is_frame_data());
    }

    #[test]
    fn parse_obu_headers_empty_input() {
        assert!(parse_obu_headers(&[]).is_empty());
    }
}
