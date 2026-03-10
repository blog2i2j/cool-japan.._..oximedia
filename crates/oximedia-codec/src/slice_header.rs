//! H.264/H.265 slice header types and parsing utilities.
//!
//! Provides the [`SliceType`] enum, [`SliceHeader`] struct, and
//! [`SliceHeaderReader`] helper that extracts slice type and frame number
//! from a raw bitstream prefix.

#![allow(dead_code)]

/// H.264 slice types as defined in Table 7-6.
///
/// The spec defines values 0–9; values 5–9 are equivalent to 0–4 but
/// signal that all slices in the picture are the same type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SliceType {
    /// Predictive slice (inter-predicted from reference frames).
    P,
    /// Bi-predictive slice (two reference frames).
    B,
    /// Intra-only slice.
    I,
    /// Switching P slice.
    Sp,
    /// Switching I slice.
    Si,
}

impl SliceType {
    /// Returns `true` when this slice type uses only intra prediction.
    pub fn is_intra(self) -> bool {
        matches!(self, Self::I | Self::Si)
    }

    /// Returns `true` when this is a bi-predictive slice.
    pub fn is_bipredictive(self) -> bool {
        self == Self::B
    }

    /// Returns `true` when this is a switching slice (SP or SI).
    pub fn is_switching(self) -> bool {
        matches!(self, Self::Sp | Self::Si)
    }

    /// Decodes a raw ue(v) slice_type value (0–9) into a [`SliceType`].
    ///
    /// Values 5–9 are mapped to their equivalents 0–4.
    pub fn from_raw(raw: u8) -> Option<Self> {
        match raw % 5 {
            0 => Some(Self::P),
            1 => Some(Self::B),
            2 => Some(Self::I),
            3 => Some(Self::Sp),
            4 => Some(Self::Si),
            _ => None,
        }
    }
}

/// Parsed fields from a slice header.
#[derive(Debug, Clone)]
pub struct SliceHeader {
    /// Slice type.
    pub slice_type: SliceType,
    /// `frame_num` syntax element (wraps at `MaxFrameNum`).
    pub frame_num: u16,
    /// PPS id referenced by this slice.
    pub pic_parameter_set_id: u8,
    /// `field_pic_flag`: `true` when this slice belongs to a field (not a frame).
    pub field_pic_flag: bool,
    /// `bottom_field_flag`: relevant only when `field_pic_flag` is `true`.
    pub bottom_field_flag: bool,
    /// IDR picture id (only meaningful for IDR slices).
    pub idr_pic_id: Option<u16>,
    /// `nal_ref_idc` from the enclosing NAL unit (0 = non-reference).
    pub nal_ref_idc: u8,
}

impl SliceHeader {
    /// Returns `true` if this slice is a reference slice (nal_ref_idc > 0).
    pub fn is_reference(&self) -> bool {
        self.nal_ref_idc > 0
    }

    /// Returns `true` if this slice is an IDR slice.
    pub fn is_idr(&self) -> bool {
        self.idr_pic_id.is_some()
    }

    /// Returns `true` if this is a key-frame (IDR or I-slice that is reference).
    pub fn is_keyframe(&self) -> bool {
        self.is_idr() || (self.slice_type.is_intra() && self.is_reference())
    }
}

/// Reads slice header fields from a raw byte buffer.
///
/// This is a simplified reader that expects the buffer to begin immediately
/// after the NAL unit start code and header byte. Full Exp-Golomb parsing
/// is approximated with a fixed layout suitable for testing.
#[derive(Debug, Default)]
pub struct SliceHeaderReader {
    /// Log2 of `MaxFrameNum` (from SPS); used to mask `frame_num`.
    pub log2_max_frame_num: u8,
}

impl SliceHeaderReader {
    /// Creates a new reader with the given `log2_max_frame_num`.
    pub fn new(log2_max_frame_num: u8) -> Self {
        Self { log2_max_frame_num }
    }

    /// Reads the slice type from the first byte of `data`.
    ///
    /// Returns `None` if `data` is empty or the value is out of range.
    pub fn read_type(&self, data: &[u8]) -> Option<SliceType> {
        let raw = data.first().copied()?;
        SliceType::from_raw(raw % 10)
    }

    /// Reads the `frame_num` from bytes 1–2 of `data` (big-endian u16),
    /// masked to `(1 << log2_max_frame_num) - 1`.
    ///
    /// Returns `None` if `data` has fewer than 3 bytes.
    pub fn frame_num(&self, data: &[u8]) -> Option<u16> {
        if data.len() < 3 {
            return None;
        }
        let raw = u16::from_be_bytes([data[1], data[2]]);
        let mask = if self.log2_max_frame_num >= 16 {
            u16::MAX
        } else {
            ((1u32 << self.log2_max_frame_num) - 1) as u16
        };
        Some(raw & mask)
    }

    /// Full parse: reads slice type, frame_num, and other fixed fields.
    ///
    /// Returns `None` if `data` is too short (requires at least 5 bytes).
    pub fn parse(&self, nal_ref_idc: u8, data: &[u8]) -> Option<SliceHeader> {
        if data.len() < 5 {
            return None;
        }
        let slice_type = self.read_type(data)?;
        let frame_num = self.frame_num(data)?;
        let pic_parameter_set_id = data[3];
        let flags = data[4];
        let field_pic_flag = (flags & 0x80) != 0;
        let bottom_field_flag = (flags & 0x40) != 0;
        let is_idr = (flags & 0x20) != 0;
        let idr_pic_id = if is_idr {
            Some(u16::from_be_bytes([data[3], data[4]]) & 0x0FFF)
        } else {
            None
        };

        Some(SliceHeader {
            slice_type,
            frame_num,
            pic_parameter_set_id,
            field_pic_flag,
            bottom_field_flag,
            idr_pic_id,
            nal_ref_idc,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_type_is_intra_i() {
        assert!(SliceType::I.is_intra());
    }

    #[test]
    fn test_slice_type_is_intra_si() {
        assert!(SliceType::Si.is_intra());
    }

    #[test]
    fn test_slice_type_p_is_not_intra() {
        assert!(!SliceType::P.is_intra());
    }

    #[test]
    fn test_slice_type_b_is_bipredictive() {
        assert!(SliceType::B.is_bipredictive());
    }

    #[test]
    fn test_slice_type_p_is_not_bipredictive() {
        assert!(!SliceType::P.is_bipredictive());
    }

    #[test]
    fn test_slice_type_is_switching() {
        assert!(SliceType::Sp.is_switching());
        assert!(SliceType::Si.is_switching());
        assert!(!SliceType::I.is_switching());
    }

    #[test]
    fn test_slice_type_from_raw_0_to_4() {
        assert_eq!(SliceType::from_raw(0), Some(SliceType::P));
        assert_eq!(SliceType::from_raw(1), Some(SliceType::B));
        assert_eq!(SliceType::from_raw(2), Some(SliceType::I));
        assert_eq!(SliceType::from_raw(3), Some(SliceType::Sp));
        assert_eq!(SliceType::from_raw(4), Some(SliceType::Si));
    }

    #[test]
    fn test_slice_type_from_raw_5_to_9_maps_same() {
        assert_eq!(SliceType::from_raw(5), Some(SliceType::P));
        assert_eq!(SliceType::from_raw(7), Some(SliceType::I));
        assert_eq!(SliceType::from_raw(9), Some(SliceType::Si));
    }

    #[test]
    fn test_slice_header_is_reference() {
        let hdr = SliceHeader {
            slice_type: SliceType::I,
            frame_num: 0,
            pic_parameter_set_id: 0,
            field_pic_flag: false,
            bottom_field_flag: false,
            idr_pic_id: None,
            nal_ref_idc: 1,
        };
        assert!(hdr.is_reference());
    }

    #[test]
    fn test_slice_header_non_reference() {
        let hdr = SliceHeader {
            slice_type: SliceType::B,
            frame_num: 3,
            pic_parameter_set_id: 0,
            field_pic_flag: false,
            bottom_field_flag: false,
            idr_pic_id: None,
            nal_ref_idc: 0,
        };
        assert!(!hdr.is_reference());
    }

    #[test]
    fn test_slice_header_is_idr() {
        let hdr = SliceHeader {
            slice_type: SliceType::I,
            frame_num: 0,
            pic_parameter_set_id: 0,
            field_pic_flag: false,
            bottom_field_flag: false,
            idr_pic_id: Some(0),
            nal_ref_idc: 3,
        };
        assert!(hdr.is_idr());
        assert!(hdr.is_keyframe());
    }

    #[test]
    fn test_reader_read_type_i_slice() {
        let reader = SliceHeaderReader::new(4);
        // raw = 2 → I
        let ty = reader.read_type(&[2, 0, 0, 0, 0]);
        assert_eq!(ty, Some(SliceType::I));
    }

    #[test]
    fn test_reader_read_type_empty_returns_none() {
        let reader = SliceHeaderReader::new(4);
        assert!(reader.read_type(&[]).is_none());
    }

    #[test]
    fn test_reader_frame_num_masked() {
        let reader = SliceHeaderReader::new(4); // mask = 0x000F
                                                // raw frame_num bytes: 0x01, 0xFF → 0x01FF & 0x000F = 0x000F
        let fn_ = reader.frame_num(&[0, 0x01, 0xFF, 0, 0]);
        assert_eq!(fn_, Some(0x000F));
    }

    #[test]
    fn test_reader_frame_num_short_returns_none() {
        let reader = SliceHeaderReader::new(4);
        assert!(reader.frame_num(&[0, 1]).is_none());
    }

    #[test]
    fn test_reader_parse_full() {
        let reader = SliceHeaderReader::new(8);
        // slice_type=2 (I), frame_num=0x0005, pps_id=0, flags=0x00
        let data = [2u8, 0x00, 0x05, 0x00, 0x00];
        let hdr = reader.parse(1, &data).expect("parse should succeed");
        assert_eq!(hdr.slice_type, SliceType::I);
        assert_eq!(hdr.frame_num, 5);
        assert!(hdr.is_reference());
    }

    #[test]
    fn test_reader_parse_too_short_returns_none() {
        let reader = SliceHeaderReader::new(4);
        assert!(reader.parse(1, &[2, 0, 5]).is_none());
    }
}
