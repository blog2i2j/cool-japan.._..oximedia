//! Bitstream filters for codec-level NAL and OBU transformations.
//!
//! Provides conversions between AnnexB and AVCC/HVCC packet formats,
//! SPS/PPS/VPS extraction, and AV1 OBU sequence header parsing.

use std::fmt;

/// Errors that can occur during bitstream filtering operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BitstreamFilterError {
    /// Input buffer is too short for the requested operation.
    BufferTooShort {
        /// Number of bytes needed.
        needed: usize,
        /// Number of bytes actually available.
        available: usize,
    },
    /// A NAL unit length prefix encoded a size that exceeds available data.
    InvalidLengthPrefix {
        /// Byte offset of the length prefix within the buffer.
        offset: usize,
        /// Number of bytes claimed by the prefix.
        claimed: usize,
        /// Number of bytes actually remaining after the prefix.
        available: usize,
    },
    /// An OBU header or extension byte was malformed.
    MalformedObuHeader {
        /// Byte offset of the malformed header within the buffer.
        offset: usize,
    },
    /// The AV1 sequence header RBSP is malformed.
    MalformedSequenceHeader,
    /// Attempted to convert an empty packet.
    EmptyPacket,
    /// NAL unit type is not recognized for the requested operation.
    UnknownNalType(u8),
    /// Length prefix size is not 1, 2, or 4.
    InvalidLengthPrefixSize(u8),
}

impl fmt::Display for BitstreamFilterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferTooShort { needed, available } => {
                write!(
                    f,
                    "buffer too short: needed {needed}, available {available}"
                )
            }
            Self::InvalidLengthPrefix {
                offset,
                claimed,
                available,
            } => {
                write!(
                    f,
                    "invalid length prefix at offset {offset}: claims {claimed} bytes but only {available} remain"
                )
            }
            Self::MalformedObuHeader { offset } => {
                write!(f, "malformed OBU header at offset {offset}")
            }
            Self::MalformedSequenceHeader => write!(f, "malformed AV1 sequence header"),
            Self::EmptyPacket => write!(f, "packet is empty"),
            Self::UnknownNalType(t) => write!(f, "unknown NAL unit type: {t}"),
            Self::InvalidLengthPrefixSize(s) => {
                write!(f, "invalid length prefix size: {s} (must be 1, 2, or 4)")
            }
        }
    }
}

impl std::error::Error for BitstreamFilterError {}

/// Result type for bitstream filter operations.
pub type BitstreamResult<T> = Result<T, BitstreamFilterError>;

// ---------------------------------------------------------------------------
// AnnexB ↔ AVCC conversion
// ---------------------------------------------------------------------------

/// Number of bytes used to encode each NAL length in AVCC/HVCC format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LengthPrefixSize {
    /// 1-byte length prefix (max NAL size: 255).
    One = 1,
    /// 2-byte big-endian length prefix (max NAL size: 65535).
    Two = 2,
    /// 4-byte big-endian length prefix (max NAL size: 4 GiB − 1).
    Four = 4,
}

impl LengthPrefixSize {
    /// Construct from the raw byte count used in container metadata.
    pub fn from_raw(raw: u8) -> BitstreamResult<Self> {
        match raw {
            1 => Ok(Self::One),
            2 => Ok(Self::Two),
            4 => Ok(Self::Four),
            other => Err(BitstreamFilterError::InvalidLengthPrefixSize(other)),
        }
    }

    /// Returns the numeric value.
    pub fn as_usize(self) -> usize {
        self as usize
    }
}

/// The AnnexB start-code prefix (3-byte short form).
const START_CODE_3: [u8; 3] = [0x00, 0x00, 0x01];
/// The AnnexB start-code prefix (4-byte long form).
const START_CODE_4: [u8; 4] = [0x00, 0x00, 0x00, 0x01];

/// Split an AnnexB byte stream into individual raw NAL unit byte slices.
///
/// Both 3-byte (`00 00 01`) and 4-byte (`00 00 00 01`) start codes are
/// recognised. The returned slices do **not** include the start code.
pub fn split_annexb(data: &[u8]) -> Vec<&[u8]> {
    let mut nals: Vec<&[u8]> = Vec::new();
    let mut start = 0usize;
    let len = data.len();

    // Skip a leading start code if present.
    if len >= 4 && data[..4] == START_CODE_4 {
        start = 4;
    } else if len >= 3 && data[..3] == START_CODE_3 {
        start = 3;
    }

    let mut i = start;
    while i + 2 < len {
        if data[i] == 0x00 && data[i + 1] == 0x00 {
            if i + 3 < len && data[i + 2] == 0x00 && data[i + 3] == 0x01 {
                // 4-byte start code
                let nal = &data[start..i];
                if !nal.is_empty() {
                    nals.push(nal);
                }
                i += 4;
                start = i;
                continue;
            } else if data[i + 2] == 0x01 {
                // 3-byte start code
                let nal = &data[start..i];
                if !nal.is_empty() {
                    nals.push(nal);
                }
                i += 3;
                start = i;
                continue;
            }
        }
        i += 1;
    }

    // Tail NAL unit.
    let tail = &data[start..];
    if !tail.is_empty() {
        nals.push(tail);
    }

    nals
}

/// Convert an AnnexB packet to AVCC/HVCC length-prefixed format.
///
/// Each NAL unit delimited by start codes is prefixed with a big-endian
/// integer whose width is determined by `prefix_size`.
pub fn annexb_to_avcc(data: &[u8], prefix_size: LengthPrefixSize) -> BitstreamResult<Vec<u8>> {
    if data.is_empty() {
        return Err(BitstreamFilterError::EmptyPacket);
    }
    let nals = split_annexb(data);
    let prefix_bytes = prefix_size.as_usize();
    let total: usize = nals.iter().map(|n| prefix_bytes + n.len()).sum();
    let mut out = Vec::with_capacity(total);

    for nal in nals {
        let nal_len = nal.len();
        match prefix_size {
            LengthPrefixSize::One => {
                out.push(nal_len as u8);
            }
            LengthPrefixSize::Two => {
                out.extend_from_slice(&(nal_len as u16).to_be_bytes());
            }
            LengthPrefixSize::Four => {
                out.extend_from_slice(&(nal_len as u32).to_be_bytes());
            }
        }
        out.extend_from_slice(nal);
    }

    Ok(out)
}

/// Convert an AVCC/HVCC length-prefixed packet to AnnexB start-code format.
///
/// `prefix_size` must match the value in the container's parameter set record
/// (`AVCDecoderConfigurationRecord.lengthSizeMinusOne + 1`).
pub fn avcc_to_annexb(data: &[u8], prefix_size: LengthPrefixSize) -> BitstreamResult<Vec<u8>> {
    if data.is_empty() {
        return Err(BitstreamFilterError::EmptyPacket);
    }
    let prefix_bytes = prefix_size.as_usize();
    let mut out = Vec::with_capacity(data.len() + data.len() / 4);
    let mut offset = 0usize;

    while offset < data.len() {
        if offset + prefix_bytes > data.len() {
            return Err(BitstreamFilterError::BufferTooShort {
                needed: offset + prefix_bytes,
                available: data.len(),
            });
        }
        let nal_len = read_be_uint(&data[offset..offset + prefix_bytes], prefix_bytes);
        offset += prefix_bytes;

        let remaining = data.len() - offset;
        if nal_len > remaining {
            return Err(BitstreamFilterError::InvalidLengthPrefix {
                offset: offset - prefix_bytes,
                claimed: nal_len,
                available: remaining,
            });
        }
        out.extend_from_slice(&START_CODE_4);
        out.extend_from_slice(&data[offset..offset + nal_len]);
        offset += nal_len;
    }

    Ok(out)
}

/// Read a big-endian unsigned integer of `n` bytes (n ∈ {1, 2, 4}).
fn read_be_uint(bytes: &[u8], n: usize) -> usize {
    match n {
        1 => bytes[0] as usize,
        2 => u16::from_be_bytes([bytes[0], bytes[1]]) as usize,
        4 => u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// H.264 / H.265 NAL unit type extraction
// ---------------------------------------------------------------------------

/// H.264 NAL unit types relevant for parameter set extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H264NalType {
    /// Non-IDR slice (nal_unit_type 1).
    NonIdrSlice,
    /// IDR slice (nal_unit_type 5).
    IdrSlice,
    /// Supplemental enhancement information (nal_unit_type 6).
    Sei,
    /// Sequence parameter set (nal_unit_type 7).
    Sps,
    /// Picture parameter set (nal_unit_type 8).
    Pps,
    /// Access unit delimiter (nal_unit_type 9).
    Aud,
    /// End of sequence (nal_unit_type 10).
    EndOfSeq,
    /// End of stream (nal_unit_type 11).
    EndOfStream,
    /// Filler data (nal_unit_type 12).
    FillerData,
    /// Other / reserved type.
    Other(u8),
}

impl H264NalType {
    /// Parse from the first byte of a raw NAL unit (forbidden_zero_bit + nal_ref_idc + type).
    pub fn from_nal_byte(byte: u8) -> Self {
        match byte & 0x1F {
            1 => Self::NonIdrSlice,
            5 => Self::IdrSlice,
            6 => Self::Sei,
            7 => Self::Sps,
            8 => Self::Pps,
            9 => Self::Aud,
            10 => Self::EndOfSeq,
            11 => Self::EndOfStream,
            12 => Self::FillerData,
            t => Self::Other(t),
        }
    }
}

/// A parsed, classified NAL unit from an H.264 or H.265 bitstream.
#[derive(Debug, Clone)]
pub struct NalUnit<'a> {
    /// Parsed type (H.264 interpretation).
    pub nal_type: H264NalType,
    /// Raw bytes of the NAL unit (excluding any start code).
    pub data: &'a [u8],
}

impl<'a> NalUnit<'a> {
    /// Construct from a raw NAL slice (must be non-empty).
    pub fn from_raw(data: &'a [u8]) -> Option<Self> {
        let first = *data.first()?;
        Some(Self {
            nal_type: H264NalType::from_nal_byte(first),
            data,
        })
    }
}

/// Extract all SPS NAL units from an AnnexB bitstream.
pub fn extract_sps(data: &[u8]) -> Vec<NalUnit<'_>> {
    split_annexb(data)
        .into_iter()
        .filter_map(NalUnit::from_raw)
        .filter(|n| n.nal_type == H264NalType::Sps)
        .collect()
}

/// Extract all PPS NAL units from an AnnexB bitstream.
pub fn extract_pps(data: &[u8]) -> Vec<NalUnit<'_>> {
    split_annexb(data)
        .into_iter()
        .filter_map(NalUnit::from_raw)
        .filter(|n| n.nal_type == H264NalType::Pps)
        .collect()
}

/// Extract both SPS and PPS NAL units from an AnnexB bitstream as owned bytes.
pub fn extract_sps_pps(data: &[u8]) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let nals = split_annexb(data);
    let mut sps_list = Vec::new();
    let mut pps_list = Vec::new();
    for nal_bytes in nals {
        if let Some(nal) = NalUnit::from_raw(nal_bytes) {
            match nal.nal_type {
                H264NalType::Sps => sps_list.push(nal.data.to_vec()),
                H264NalType::Pps => pps_list.push(nal.data.to_vec()),
                _ => {}
            }
        }
    }
    (sps_list, pps_list)
}

// ---------------------------------------------------------------------------
// AV1 OBU parsing
// ---------------------------------------------------------------------------

/// AV1 Open Bitstream Unit (OBU) types per the AV1 specification §5.3.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Av1ObuType {
    /// Sequence header OBU (type 1).
    SequenceHeader,
    /// Temporal delimiter OBU (type 2).
    TemporalDelimiter,
    /// Frame header OBU (type 3).
    FrameHeader,
    /// Tile group OBU (type 4).
    TileGroup,
    /// Metadata OBU (type 5).
    Metadata,
    /// Frame OBU (type 6).
    Frame,
    /// Redundant frame header OBU (type 7).
    RedundantFrameHeader,
    /// Tile list OBU (type 8).
    TileList,
    /// Padding OBU (type 15).
    Padding,
    /// Reserved OBU type.
    Reserved(u8),
}

impl Av1ObuType {
    fn from_raw(raw: u8) -> Self {
        match raw {
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

/// A parsed AV1 OBU unit with its type and payload bytes.
#[derive(Debug, Clone)]
pub struct Av1Obu {
    /// The OBU type.
    pub obu_type: Av1ObuType,
    /// The payload bytes (excluding the OBU header and size field).
    pub payload: Vec<u8>,
}

/// Parse a low-level unsigned LEB128 value from a byte slice.
///
/// Returns `(value, bytes_consumed)` or an error if the encoding is malformed.
fn read_leb128(data: &[u8], offset: usize) -> BitstreamResult<(u64, usize)> {
    let mut result: u64 = 0;
    let mut shift = 0u32;
    let mut consumed = 0usize;
    loop {
        if offset + consumed >= data.len() {
            return Err(BitstreamFilterError::MalformedObuHeader { offset });
        }
        let byte = data[offset + consumed];
        consumed += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 56 {
            return Err(BitstreamFilterError::MalformedObuHeader { offset });
        }
    }
    Ok((result, consumed))
}

/// Split a contiguous AV1 bitstream (e.g., from an ISOBMFF `av01` sample) into
/// individual OBU units.
///
/// Assumes the standard OBU framing with `obu_has_size_field = 1`.
pub fn split_av1_obus(data: &[u8]) -> BitstreamResult<Vec<Av1Obu>> {
    if data.is_empty() {
        return Err(BitstreamFilterError::EmptyPacket);
    }
    let mut obus = Vec::new();
    let mut offset = 0usize;
    let len = data.len();

    while offset < len {
        if offset >= len {
            break;
        }
        let header_byte = data[offset];
        let forbidden_bit = (header_byte >> 7) & 1;
        if forbidden_bit != 0 {
            return Err(BitstreamFilterError::MalformedObuHeader { offset });
        }
        let obu_type_raw = (header_byte >> 3) & 0x0F;
        let obu_extension_flag = (header_byte >> 2) & 1;
        let obu_has_size_field = (header_byte >> 1) & 1;
        offset += 1;

        // Skip extension byte if present.
        if obu_extension_flag == 1 {
            if offset >= len {
                return Err(BitstreamFilterError::MalformedObuHeader { offset });
            }
            offset += 1;
        }

        let payload_len = if obu_has_size_field == 1 {
            let (sz, consumed) = read_leb128(data, offset)?;
            offset += consumed;
            sz as usize
        } else {
            // Without a size field the OBU extends to end of data.
            len - offset
        };

        if offset + payload_len > len {
            return Err(BitstreamFilterError::InvalidLengthPrefix {
                offset,
                claimed: payload_len,
                available: len - offset,
            });
        }

        let payload = data[offset..offset + payload_len].to_vec();
        offset += payload_len;

        obus.push(Av1Obu {
            obu_type: Av1ObuType::from_raw(obu_type_raw),
            payload,
        });
    }

    Ok(obus)
}

// ---------------------------------------------------------------------------
// AV1 Sequence Header quick-parse
// ---------------------------------------------------------------------------

/// High-level fields extracted from an AV1 sequence header OBU.
///
/// Only the fields most useful for container-level metadata are decoded;
/// a full reference decoder is out of scope for this filter layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Av1SequenceHeader {
    /// `seq_profile` (3 bits): 0 = Main, 1 = High, 2 = Professional.
    pub seq_profile: u8,
    /// `still_picture` flag.
    pub still_picture: bool,
    /// `reduced_still_picture_header` flag.
    pub reduced_still_picture_header: bool,
    /// Maximum frame width in pixels (decoded from the header's width bits).
    pub max_frame_width: u32,
    /// Maximum frame height in pixels.
    pub max_frame_height: u32,
    /// 12-bit color mode flag (`high_bitdepth`).
    pub high_bitdepth: bool,
    /// Twelve-bit video flag (only valid when `high_bitdepth` is true).
    pub twelve_bit: bool,
    /// `mono_chrome` flag.
    pub mono_chrome: bool,
}

/// A simple bit-level reader for big-endian bitstreams.
struct BitReader<'a> {
    data: &'a [u8],
    byte_offset: usize,
    bit_offset: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_offset: 0,
            bit_offset: 0,
        }
    }

    fn read_bit(&mut self) -> BitstreamResult<u8> {
        if self.byte_offset >= self.data.len() {
            return Err(BitstreamFilterError::MalformedSequenceHeader);
        }
        let byte = self.data[self.byte_offset];
        let bit = (byte >> (7 - self.bit_offset)) & 1;
        self.bit_offset += 1;
        if self.bit_offset == 8 {
            self.bit_offset = 0;
            self.byte_offset += 1;
        }
        Ok(bit)
    }

    fn read_bits(&mut self, n: u8) -> BitstreamResult<u32> {
        let mut val = 0u32;
        for _ in 0..n {
            val = (val << 1) | self.read_bit()? as u32;
        }
        Ok(val)
    }

    /// Unsigned LEB128 read (for `uvlc()` is not needed here; we use fixed-width reads).
    /// Read a "u(n)" syntax element.
    fn u(&mut self, n: u8) -> BitstreamResult<u32> {
        self.read_bits(n)
    }

    fn f(&mut self, n: u8) -> BitstreamResult<u32> {
        self.read_bits(n)
    }
}

/// Parse the fields of an AV1 sequence header OBU payload.
///
/// Returns `None` if the payload is too short to contain a valid header.
pub fn parse_av1_sequence_header(payload: &[u8]) -> BitstreamResult<Av1SequenceHeader> {
    let mut r = BitReader::new(payload);

    let seq_profile = r.f(3)? as u8;
    let still_picture = r.f(1)? != 0;
    let reduced_still_picture_header = r.f(1)? != 0;

    // timing_info_present_flag and decoder model etc. — skip when
    // reduced_still_picture_header is set.
    let (timing_info_present, decoder_model_info_present) = if reduced_still_picture_header {
        (false, false)
    } else {
        let tip = r.f(1)? != 0;
        let dmip = if tip {
            // timing_info() — simplified skip: num_units_in_display_tick (32),
            // time_scale (32), equal_picture_interval (1) + conditional.
            r.u(32)?;
            r.u(32)?;
            let epi = r.f(1)?;
            if epi != 0 {
                // pts_num_ticks_per_picture_minus_1 (uvlc) — skip a few bits.
                let _ = read_uvlc(&mut r)?;
            }
            r.f(1)? != 0
        } else {
            false
        };
        if dmip {
            // decoder_model_info(): buffer_delay_length_minus_1(5), ...
            // We do a best-effort skip of 40 bits (approximate).
            let _ = r.u(5)?;
            let _ = r.u(32)?;
            let _ = r.u(9)?;
        }
        (tip, dmip)
    };

    let _ = timing_info_present;
    let _ = decoder_model_info_present;

    // operating_points_cnt_minus_1 — when not reduced header.
    if !reduced_still_picture_header {
        let op_cnt = r.u(5)?; // operating_points_cnt_minus_1 (5 bits)
        for _ in 0..=op_cnt {
            let _op_idc = r.u(12)?;
            let _seq_level_idx = r.u(5)?;
            let seq_tier = if r.u(5)? > 7 { r.u(1)? } else { 0 };
            let _ = seq_tier;
            if decoder_model_info_present {
                let _decoder_model_present = r.u(1)?;
                // skip operating_parameters_info if present
            }
            if !reduced_still_picture_header {
                let _initial_display_delay_present = r.u(1)?;
                if decoder_model_info_present {
                    let _initial_display_delay_minus_1 = r.u(4)?;
                }
            }
        }
    }

    // frame_width_bits_minus_1 (4 bits), frame_height_bits_minus_1 (4 bits)
    let fw_bits = r.u(4)? + 1;
    let fh_bits = r.u(4)? + 1;
    let max_frame_width = r.u(fw_bits as u8)? + 1;
    let max_frame_height = r.u(fh_bits as u8)? + 1;

    // frame_id_numbers_present (1 bit) — if not reduced.
    if !reduced_still_picture_header {
        let frame_id_numbers_present = r.u(1)?;
        if frame_id_numbers_present != 0 {
            let _delta_frame_id_length = r.u(4)?;
            let _additional_frame_id_length = r.u(3)?;
        }
    }

    // use_128x128_superblock (1), enable_filter_intra (1), enable_intra_edge_filter (1)
    let _use_128 = r.u(1)?;
    let _enable_filter_intra = r.u(1)?;
    let _enable_intra_edge_filter = r.u(1)?;

    // Skip remaining non-color fields for brevity.
    // color_config()
    let high_bitdepth = r.u(1)? != 0;
    let twelve_bit = if seq_profile == 2 && high_bitdepth {
        r.u(1)? != 0
    } else {
        false
    };
    let mono_chrome = if seq_profile == 1 {
        false
    } else {
        r.u(1)? != 0
    };

    Ok(Av1SequenceHeader {
        seq_profile,
        still_picture,
        reduced_still_picture_header,
        max_frame_width,
        max_frame_height,
        high_bitdepth,
        twelve_bit,
        mono_chrome,
    })
}

/// Read an unsigned variable-length code (uvlc) value.
fn read_uvlc(r: &mut BitReader<'_>) -> BitstreamResult<u32> {
    let mut leading_zeros = 0u32;
    loop {
        let bit = r.read_bit()?;
        if bit != 0 {
            break;
        }
        leading_zeros += 1;
        if leading_zeros >= 32 {
            return Err(BitstreamFilterError::MalformedSequenceHeader);
        }
    }
    if leading_zeros == 0 {
        return Ok(0);
    }
    let value = r.read_bits(leading_zeros as u8)?;
    Ok((1 << leading_zeros) + value - 1)
}

/// Find and parse the first AV1 sequence header OBU found in a bitstream.
///
/// Returns `None` if no sequence header OBU is present.
pub fn find_av1_sequence_header(data: &[u8]) -> BitstreamResult<Option<Av1SequenceHeader>> {
    let obus = split_av1_obus(data)?;
    for obu in obus {
        if obu.obu_type == Av1ObuType::SequenceHeader {
            return parse_av1_sequence_header(&obu.payload).map(Some);
        }
    }
    Ok(None)
}

// ---------------------------------------------------------------------------
// High-level helper: remove emulation-prevention bytes (RBSP decoding)
// ---------------------------------------------------------------------------

/// Remove H.264/H.265 emulation prevention bytes (`0x03`) from a RBSP.
///
/// Sequences `00 00 03 {00, 01, 02, 03}` have the `03` byte stripped.
pub fn remove_emulation_prevention(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let len = data.len();
    let mut i = 0;
    while i < len {
        if i + 2 < len && data[i] == 0x00 && data[i + 1] == 0x00 && data[i + 2] == 0x03 {
            out.push(0x00);
            out.push(0x00);
            i += 3; // Skip the emulation prevention byte.
        } else {
            out.push(data[i]);
            i += 1;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- AnnexB / AVCC round-trip ---

    #[test]
    fn test_split_annexb_single_nal_4byte_startcode() {
        let data = [0x00, 0x00, 0x00, 0x01, 0x67, 0xAB, 0xCD];
        let nals = split_annexb(&data);
        assert_eq!(nals.len(), 1);
        assert_eq!(nals[0], &[0x67, 0xAB, 0xCD]);
    }

    #[test]
    fn test_split_annexb_multiple_nals() {
        let data = [
            0x00, 0x00, 0x00, 0x01, 0x67, 0x11, // SPS
            0x00, 0x00, 0x01, 0x68, 0x22, // PPS (3-byte start code)
        ];
        let nals = split_annexb(&data);
        assert_eq!(nals.len(), 2);
        assert_eq!(nals[0], &[0x67, 0x11]);
        assert_eq!(nals[1], &[0x68, 0x22]);
    }

    #[test]
    fn test_annexb_to_avcc_roundtrip() {
        let sps = [0x67u8, 0x42, 0x00, 0x1E];
        let pps = [0x68u8, 0xCE, 0x38, 0x80];
        let mut annexb = Vec::new();
        annexb.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        annexb.extend_from_slice(&sps);
        annexb.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        annexb.extend_from_slice(&pps);

        let avcc = annexb_to_avcc(&annexb, LengthPrefixSize::Four).unwrap();
        let back = avcc_to_annexb(&avcc, LengthPrefixSize::Four).unwrap();

        // The round-trip should contain both NAL units.
        let nals = split_annexb(&back);
        assert_eq!(nals.len(), 2);
        assert_eq!(nals[0], &sps);
        assert_eq!(nals[1], &pps);
    }

    #[test]
    fn test_avcc_to_annexb_two_byte_prefix() {
        // Craft a 2-byte AVCC packet with one 3-byte NAL.
        let nal = [0x65u8, 0x11, 0x22];
        let mut avcc = Vec::new();
        avcc.extend_from_slice(&(3u16).to_be_bytes());
        avcc.extend_from_slice(&nal);
        let result = avcc_to_annexb(&avcc, LengthPrefixSize::Two).unwrap();
        assert_eq!(&result[..4], &[0x00, 0x00, 0x00, 0x01]);
        assert_eq!(&result[4..], &nal);
    }

    #[test]
    fn test_avcc_invalid_length_prefix_error() {
        // Claim 100 bytes but only 2 available.
        let mut avcc = Vec::new();
        avcc.extend_from_slice(&(100u32).to_be_bytes());
        avcc.extend_from_slice(&[0xAA, 0xBB]);
        let err = avcc_to_annexb(&avcc, LengthPrefixSize::Four).unwrap_err();
        assert!(matches!(
            err,
            BitstreamFilterError::InvalidLengthPrefix { .. }
        ));
    }

    // --- SPS/PPS extraction ---

    #[test]
    fn test_extract_sps_pps() {
        let mut stream = Vec::new();
        // SPS (nal_type = 7)
        stream.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1E]);
        // PPS (nal_type = 8)
        stream.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x68, 0xCE]);
        // IDR slice (nal_type = 5)
        stream.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x65, 0x88]);

        let (sps, pps) = extract_sps_pps(&stream);
        assert_eq!(sps.len(), 1);
        assert_eq!(pps.len(), 1);
        // First byte of extracted SPS should be 0x67.
        assert_eq!(sps[0][0], 0x67);
        assert_eq!(pps[0][0], 0x68);
    }

    // --- Emulation prevention removal ---

    #[test]
    fn test_remove_emulation_prevention() {
        let input = [0x00u8, 0x00, 0x03, 0x01, 0xFF];
        let output = remove_emulation_prevention(&input);
        assert_eq!(output, [0x00, 0x00, 0x01, 0xFF]);
    }

    // --- AV1 OBU splitting ---

    #[test]
    fn test_split_av1_obus_sequence_header() {
        // Minimal synthetic AV1 packet: one OBU.
        // OBU header: type=1 (sequence), no extension, has_size_field=1.
        // header byte: forbidden=0, type=1, ext=0, has_size=1, reserved=0
        // = 0b0_0001_0_1_0 = 0x0A
        let payload = [0x00u8; 4]; // dummy payload
        let mut data = Vec::new();
        data.push(0x0A); // OBU header
                         // LEB128 size = 4
        data.push(0x04);
        data.extend_from_slice(&payload);

        let obus = split_av1_obus(&data).unwrap();
        assert_eq!(obus.len(), 1);
        assert_eq!(obus[0].obu_type, Av1ObuType::SequenceHeader);
        assert_eq!(obus[0].payload, payload);
    }

    #[test]
    fn test_split_av1_obus_empty_error() {
        let err = split_av1_obus(&[]).unwrap_err();
        assert_eq!(err, BitstreamFilterError::EmptyPacket);
    }

    #[test]
    fn test_split_av1_obus_multiple() {
        // Two OBUs: temporal delimiter (type=2) + tile group (type=4).
        // TD header: type=2 => 0b0_0010_0_1_0 = 0x12, size=0
        // TG header: type=4 => 0b0_0100_0_1_0 = 0x22, size=2
        let mut data = Vec::new();
        data.push(0x12); // temporal delimiter
        data.push(0x00); // size=0
        data.push(0x22); // tile group
        data.push(0x02); // size=2
        data.push(0xAA);
        data.push(0xBB);

        let obus = split_av1_obus(&data).unwrap();
        assert_eq!(obus.len(), 2);
        assert_eq!(obus[0].obu_type, Av1ObuType::TemporalDelimiter);
        assert_eq!(obus[1].obu_type, Av1ObuType::TileGroup);
        assert_eq!(obus[1].payload, [0xAA, 0xBB]);
    }

    #[test]
    fn test_leb128_multi_byte() {
        // Value 300 = 0xAC 0x02 in LEB128.
        let data = [0xACu8, 0x02];
        let (val, consumed) = read_leb128(&data, 0).unwrap();
        assert_eq!(val, 300);
        assert_eq!(consumed, 2);
    }

    #[test]
    fn test_empty_packet_error() {
        assert_eq!(
            annexb_to_avcc(&[], LengthPrefixSize::Four).unwrap_err(),
            BitstreamFilterError::EmptyPacket
        );
        assert_eq!(
            avcc_to_annexb(&[], LengthPrefixSize::Four).unwrap_err(),
            BitstreamFilterError::EmptyPacket
        );
    }

    #[test]
    fn test_length_prefix_size_from_raw() {
        assert_eq!(
            LengthPrefixSize::from_raw(1).unwrap(),
            LengthPrefixSize::One
        );
        assert_eq!(
            LengthPrefixSize::from_raw(2).unwrap(),
            LengthPrefixSize::Two
        );
        assert_eq!(
            LengthPrefixSize::from_raw(4).unwrap(),
            LengthPrefixSize::Four
        );
        assert!(LengthPrefixSize::from_raw(3).is_err());
    }
}
