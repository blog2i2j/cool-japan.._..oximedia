//! AV1 OBU (Open Bitstream Unit) parsing.
//!
//! OBUs are the fundamental unit of AV1 bitstreams. Each OBU contains
//! a header followed by optional payload data.

use crate::error::{CodecError, CodecResult};
use oximedia_io::BitReader;

/// OBU types as defined in AV1 specification section 5.3.1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ObuType {
    /// Reserved (0).
    Reserved0,
    /// Sequence header OBU.
    SequenceHeader,
    /// Temporal delimiter OBU.
    TemporalDelimiter,
    /// Frame header OBU.
    FrameHeader,
    /// Tile group OBU.
    TileGroup,
    /// Metadata OBU.
    Metadata,
    /// Combined frame header and tile group.
    Frame,
    /// Redundant frame header OBU.
    RedundantFrameHeader,
    /// Tile list OBU.
    TileList,
    /// Reserved (9-14).
    Reserved(u8),
    /// Padding OBU.
    Padding,
}

impl From<u8> for ObuType {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Reserved0,
            1 => Self::SequenceHeader,
            2 => Self::TemporalDelimiter,
            3 => Self::FrameHeader,
            4 => Self::TileGroup,
            5 => Self::Metadata,
            6 => Self::Frame,
            7 => Self::RedundantFrameHeader,
            8 => Self::TileList,
            15 => Self::Padding,
            other => Self::Reserved(other),
        }
    }
}

impl From<ObuType> for u8 {
    fn from(obu_type: ObuType) -> Self {
        match obu_type {
            ObuType::Reserved0 => 0,
            ObuType::SequenceHeader => 1,
            ObuType::TemporalDelimiter => 2,
            ObuType::FrameHeader => 3,
            ObuType::TileGroup => 4,
            ObuType::Metadata => 5,
            ObuType::Frame => 6,
            ObuType::RedundantFrameHeader => 7,
            ObuType::TileList => 8,
            ObuType::Padding => 15,
            ObuType::Reserved(v) => v,
        }
    }
}

/// OBU header as defined in AV1 specification section 5.3.2.
#[derive(Clone, Debug)]
pub struct ObuHeader {
    /// OBU type.
    pub obu_type: ObuType,
    /// Extension flag present.
    pub has_extension: bool,
    /// Size field present.
    pub has_size: bool,
    /// Temporal ID (from extension header).
    pub temporal_id: u8,
    /// Spatial ID (from extension header).
    pub spatial_id: u8,
}

impl ObuHeader {
    /// Parse OBU header from bitstream.
    ///
    /// # Errors
    ///
    /// Returns error if header is malformed.
    #[allow(clippy::cast_possible_truncation)]
    pub fn parse(reader: &mut BitReader<'_>) -> CodecResult<Self> {
        let forbidden = reader.read_bit().map_err(CodecError::Core)?;
        if forbidden != 0 {
            return Err(CodecError::InvalidBitstream(
                "OBU forbidden bit is set".to_string(),
            ));
        }

        let obu_type_raw = reader.read_bits(4).map_err(CodecError::Core)? as u8;
        let obu_type = ObuType::from(obu_type_raw);
        let has_extension = reader.read_bit().map_err(CodecError::Core)? != 0;
        let has_size = reader.read_bit().map_err(CodecError::Core)? != 0;

        let reserved = reader.read_bit().map_err(CodecError::Core)?;
        if reserved != 0 {
            return Err(CodecError::InvalidBitstream(
                "OBU reserved bit is set".to_string(),
            ));
        }

        let (temporal_id, spatial_id) = if has_extension {
            let temporal_id = reader.read_bits(3).map_err(CodecError::Core)? as u8;
            let spatial_id = reader.read_bits(2).map_err(CodecError::Core)? as u8;
            let _reserved = reader.read_bits(3).map_err(CodecError::Core)?;
            (temporal_id, spatial_id)
        } else {
            (0, 0)
        };

        Ok(Self {
            obu_type,
            has_extension,
            has_size,
            temporal_id,
            spatial_id,
        })
    }

    /// Get the header size in bytes.
    #[must_use]
    pub const fn header_size(&self) -> usize {
        if self.has_extension {
            2
        } else {
            1
        }
    }

    /// Serialize OBU header to bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(2);
        let obu_type_val: u8 = self.obu_type.into();
        let first_byte = (obu_type_val << 3)
            | (u8::from(self.has_extension) << 2)
            | (u8::from(self.has_size) << 1);
        bytes.push(first_byte);

        if self.has_extension {
            let ext_byte = (self.temporal_id << 5) | (self.spatial_id << 3);
            bytes.push(ext_byte);
        }

        bytes
    }
}

/// Parse LEB128 encoded unsigned integer.
#[allow(clippy::cast_possible_truncation)]
pub fn parse_leb128(reader: &mut BitReader<'_>) -> CodecResult<u64> {
    let mut value: u64 = 0;
    let mut shift = 0;

    loop {
        let byte = reader.read_bits(8).map_err(CodecError::Core)? as u8;
        value |= u64::from(byte & 0x7F) << shift;

        if byte & 0x80 == 0 {
            break;
        }

        shift += 7;
        if shift >= 64 {
            return Err(CodecError::InvalidBitstream(
                "LEB128 value overflow".to_string(),
            ));
        }
    }

    Ok(value)
}

/// Encode a value as LEB128.
#[must_use]
pub fn encode_leb128(mut value: u64) -> Vec<u8> {
    let mut bytes = Vec::new();

    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;

        if value != 0 {
            byte |= 0x80;
        }
        bytes.push(byte);

        if value == 0 {
            break;
        }
    }

    bytes
}

/// Parse a complete OBU from data.
#[allow(clippy::cast_possible_truncation)]
pub fn parse_obu(data: &[u8]) -> CodecResult<(ObuHeader, &[u8], usize)> {
    let mut reader = BitReader::new(data);
    let header = ObuHeader::parse(&mut reader)?;

    let size = if header.has_size {
        parse_leb128(&mut reader)? as usize
    } else {
        let header_bytes = reader.bits_read().div_ceil(8);
        data.len().saturating_sub(header_bytes)
    };

    let header_bytes = reader.bits_read().div_ceil(8);
    let total_size = header_bytes + size;

    if total_size > data.len() {
        return Err(CodecError::InvalidBitstream(format!(
            "OBU size {} exceeds available data {}",
            total_size,
            data.len()
        )));
    }

    let payload = &data[header_bytes..header_bytes + size];
    Ok((header, payload, total_size))
}

/// Iterator over OBUs in a temporal unit.
pub struct ObuIterator<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> ObuIterator<'a> {
    /// Create a new OBU iterator.
    #[must_use]
    pub const fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }
}

impl<'a> Iterator for ObuIterator<'a> {
    type Item = CodecResult<(ObuHeader, &'a [u8])>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.data.len() {
            return None;
        }

        match parse_obu(&self.data[self.offset..]) {
            Ok((header, payload, total_size)) => {
                self.offset += total_size;
                Some(Ok((header, payload)))
            }
            Err(e) => Some(Err(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_obu_type_from_u8() {
        assert_eq!(ObuType::from(1), ObuType::SequenceHeader);
        assert_eq!(ObuType::from(6), ObuType::Frame);
    }

    #[test]
    fn test_leb128_single_byte() {
        let data = [0x7F];
        let mut reader = BitReader::new(&data);
        assert_eq!(parse_leb128(&mut reader).expect("should succeed"), 127);
    }

    #[test]
    fn test_leb128_multi_byte() {
        let data = [0x80, 0x01];
        let mut reader = BitReader::new(&data);
        assert_eq!(parse_leb128(&mut reader).expect("should succeed"), 128);
    }

    #[test]
    fn test_encode_leb128() {
        assert_eq!(encode_leb128(0), vec![0x00]);
        assert_eq!(encode_leb128(127), vec![0x7F]);
        assert_eq!(encode_leb128(128), vec![0x80, 0x01]);
    }
}
