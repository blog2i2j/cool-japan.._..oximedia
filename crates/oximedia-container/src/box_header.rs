//! ISO Base Media File Format (ISOBMFF / MP4) box header parsing.
//!
//! Provides [`BoxType`], [`BoxHeader`], and [`BoxParser`] for reading the
//! 8-byte (or 16-byte extended) headers that prefix every ISOBMFF box.

#![allow(dead_code)]

/// Well-known ISOBMFF box four-character codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoxType {
    /// `ftyp` — file type box.
    Ftyp,
    /// `moov` — movie container box.
    Moov,
    /// `mdat` — media data box.
    Mdat,
    /// `moof` — movie fragment box.
    Moof,
    /// `trak` — track box.
    Trak,
    /// `mdia` — media box.
    Mdia,
    /// `minf` — media information box.
    Minf,
    /// `stbl` — sample table box.
    Stbl,
    /// `udta` — user data box.
    Udta,
    /// `mvhd` — movie header box.
    Mvhd,
    /// `tkhd` — track header box.
    Tkhd,
    /// `mdhd` — media header box.
    Mdhd,
    /// Unknown or unrecognised four-character code.
    Unknown([u8; 4]),
}

impl BoxType {
    /// Parses a 4-byte `FourCC` into a [`BoxType`].
    #[must_use]
    pub fn from_fourcc(cc: [u8; 4]) -> Self {
        match &cc {
            b"ftyp" => Self::Ftyp,
            b"moov" => Self::Moov,
            b"mdat" => Self::Mdat,
            b"moof" => Self::Moof,
            b"trak" => Self::Trak,
            b"mdia" => Self::Mdia,
            b"minf" => Self::Minf,
            b"stbl" => Self::Stbl,
            b"udta" => Self::Udta,
            b"mvhd" => Self::Mvhd,
            b"tkhd" => Self::Tkhd,
            b"mdhd" => Self::Mdhd,
            _ => Self::Unknown(cc),
        }
    }

    /// Returns the raw `FourCC` bytes for this box type.
    #[must_use]
    pub fn to_fourcc(self) -> [u8; 4] {
        match self {
            Self::Ftyp => *b"ftyp",
            Self::Moov => *b"moov",
            Self::Mdat => *b"mdat",
            Self::Moof => *b"moof",
            Self::Trak => *b"trak",
            Self::Mdia => *b"mdia",
            Self::Minf => *b"minf",
            Self::Stbl => *b"stbl",
            Self::Udta => *b"udta",
            Self::Mvhd => *b"mvhd",
            Self::Tkhd => *b"tkhd",
            Self::Mdhd => *b"mdhd",
            Self::Unknown(cc) => cc,
        }
    }

    /// Returns `true` if this box type is a container (can hold child boxes).
    #[must_use]
    pub fn is_container(self) -> bool {
        matches!(
            self,
            Self::Moov
                | Self::Trak
                | Self::Mdia
                | Self::Minf
                | Self::Stbl
                | Self::Udta
                | Self::Moof
        )
    }
}

/// Parsed ISOBMFF box header.
#[derive(Debug, Clone)]
pub struct BoxHeader {
    /// Four-character code identifying the box type.
    pub box_type: BoxType,
    /// Total box size in bytes including the header.
    pub size: u64,
    /// `true` when the header uses the 16-byte extended-size form.
    pub extended_size: bool,
    /// Byte offset of the first byte of the header in the source stream.
    pub offset: u64,
}

impl BoxHeader {
    /// Returns the number of bytes of payload (data after the header).
    #[must_use]
    pub fn data_size(&self) -> u64 {
        let header_len: u64 = if self.extended_size { 16 } else { 8 };
        self.size.saturating_sub(header_len)
    }

    /// Returns `true` when the header uses the 16-byte extended-size form.
    #[must_use]
    pub fn is_extended_size(&self) -> bool {
        self.extended_size
    }

    /// Returns the byte offset of the first payload byte in the stream.
    #[must_use]
    pub fn payload_offset(&self) -> u64 {
        let header_len: u64 = if self.extended_size { 16 } else { 8 };
        self.offset + header_len
    }
}

/// Parses box headers from a byte slice representing a portion of an MP4 file.
#[derive(Debug, Default)]
pub struct BoxParser {
    /// Current read position within the underlying stream (logical).
    position: u64,
}

impl BoxParser {
    /// Creates a new parser starting at byte `position` in the stream.
    #[must_use]
    pub fn new(position: u64) -> Self {
        Self { position }
    }

    /// Parses the next box header from `data`, starting at `data[0]`.
    ///
    /// Returns `None` if `data` is shorter than 8 bytes.
    pub fn parse_header(&mut self, data: &[u8]) -> Option<BoxHeader> {
        if data.len() < 8 {
            return None;
        }
        let raw_size = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        let fourcc = [data[4], data[5], data[6], data[7]];
        let box_type = BoxType::from_fourcc(fourcc);

        let (size, extended_size) = if raw_size == 1 {
            // Extended 64-bit size follows the FourCC.
            if data.len() < 16 {
                return None;
            }
            let s = u64::from_be_bytes([
                data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
            ]);
            (s, true)
        } else if raw_size == 0 {
            // Size of 0 means "extends to end of file"; represent as u64::MAX.
            (u64::MAX, false)
        } else {
            (u64::from(raw_size), false)
        };

        let header = BoxHeader {
            box_type,
            size,
            extended_size,
            offset: self.position,
        };

        // Advance internal position past the full box.
        if size != u64::MAX {
            self.position += size;
        }

        Some(header)
    }

    /// Returns the byte offset of the start of the next box in the stream.
    #[must_use]
    pub fn next_box_start(&self) -> u64 {
        self.position
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_box_bytes(size: u32, fourcc: &[u8; 4]) -> [u8; 8] {
        let s = size.to_be_bytes();
        [
            s[0], s[1], s[2], s[3], fourcc[0], fourcc[1], fourcc[2], fourcc[3],
        ]
    }

    #[test]
    fn test_box_type_from_fourcc_moov() {
        assert_eq!(BoxType::from_fourcc(*b"moov"), BoxType::Moov);
    }

    #[test]
    fn test_box_type_from_fourcc_unknown() {
        let cc = *b"xxxx";
        assert_eq!(BoxType::from_fourcc(cc), BoxType::Unknown(cc));
    }

    #[test]
    fn test_box_type_to_fourcc_roundtrip() {
        let ty = BoxType::Ftyp;
        assert_eq!(BoxType::from_fourcc(ty.to_fourcc()), ty);
    }

    #[test]
    fn test_box_type_is_container_moov() {
        assert!(BoxType::Moov.is_container());
    }

    #[test]
    fn test_box_type_is_container_mdat_false() {
        assert!(!BoxType::Mdat.is_container());
    }

    #[test]
    fn test_box_type_is_container_stbl() {
        assert!(BoxType::Stbl.is_container());
    }

    #[test]
    fn test_box_header_data_size_normal() {
        let hdr = BoxHeader {
            box_type: BoxType::Moov,
            size: 100,
            extended_size: false,
            offset: 0,
        };
        assert_eq!(hdr.data_size(), 92);
    }

    #[test]
    fn test_box_header_data_size_extended() {
        let hdr = BoxHeader {
            box_type: BoxType::Mdat,
            size: 200,
            extended_size: true,
            offset: 0,
        };
        assert_eq!(hdr.data_size(), 184);
    }

    #[test]
    fn test_box_header_payload_offset() {
        let hdr = BoxHeader {
            box_type: BoxType::Ftyp,
            size: 24,
            extended_size: false,
            offset: 100,
        };
        assert_eq!(hdr.payload_offset(), 108);
    }

    #[test]
    fn test_parser_parse_header_normal() {
        let mut parser = BoxParser::new(0);
        let data = make_box_bytes(28, b"moov");
        let hdr = parser
            .parse_header(&data)
            .expect("operation should succeed");
        assert_eq!(hdr.box_type, BoxType::Moov);
        assert_eq!(hdr.size, 28);
        assert!(!hdr.is_extended_size());
        assert_eq!(hdr.data_size(), 20);
    }

    #[test]
    fn test_parser_parse_header_too_short_returns_none() {
        let mut parser = BoxParser::new(0);
        assert!(parser.parse_header(&[0, 0, 0]).is_none());
    }

    #[test]
    fn test_parser_next_box_start_advances() {
        let mut parser = BoxParser::new(0);
        let data = make_box_bytes(16, b"ftyp");
        parser
            .parse_header(&data)
            .expect("operation should succeed");
        assert_eq!(parser.next_box_start(), 16);
    }

    #[test]
    fn test_parser_parse_extended_size() {
        let mut parser = BoxParser::new(0);
        // raw_size = 1 signals extended size
        let mut data = [0u8; 16];
        data[0..4].copy_from_slice(&1u32.to_be_bytes());
        data[4..8].copy_from_slice(b"mdat");
        // 64-bit size = 1000
        data[8..16].copy_from_slice(&1000u64.to_be_bytes());
        let hdr = parser
            .parse_header(&data)
            .expect("operation should succeed");
        assert!(hdr.is_extended_size());
        assert_eq!(hdr.size, 1000);
        assert_eq!(hdr.data_size(), 984);
    }

    #[test]
    fn test_parser_initial_position() {
        let parser = BoxParser::new(512);
        assert_eq!(parser.next_box_start(), 512);
    }
}
