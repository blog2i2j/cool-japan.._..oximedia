//! SEI / NAL-unit helpers for VP8/AV1 metadata attachment.
//!
//! VP8 does not have a formal SEI syntax, but its partition 0 allows embedding
//! application-level metadata as extension bytes.  AV1 has OBU metadata blocks
//! that serve a similar purpose (OBU type 5 — METADATA_OBU).
//!
//! This module provides:
//! - [`SeiPayloadType`] — typed catalogue of SEI payload kinds.
//! - [`UserDataUnregistered`] — UUID-tagged opaque metadata (SEI type 5).
//! - [`PictureTiming`] — clock tick / frame-rate SEI (type 1).
//! - [`SeiMessage`] — a complete SEI message (type + payload).
//! - [`SeiEncoder`] — serialises SEI messages into a byte buffer suitable for
//!   embedding into a VP8 partition or AV1 METADATA_OBU payload.
//! - [`SeiDecoder`] — parses that same byte buffer back into [`SeiMessage`]s.
//! - [`Av1MetadataObu`] — wraps a serialised SEI payload into a minimal
//!   AV1 METADATA OBU.
//! - [`Vp8MetadataBlock`] — wraps serialised metadata into a VP8-style
//!   user-data extension block.

use crate::error::{CodecError, CodecResult};

// ── Constants ─────────────────────────────────────────────────────────────────

/// OBU type value for AV1 METADATA_OBU.
pub const AV1_OBU_TYPE_METADATA: u8 = 5;

/// Marker byte that introduces a VP8 user-data extension block.
pub const VP8_USER_DATA_MARKER: u8 = 0xFE;

/// Length of a UUID as defined by RFC 4122 (16 bytes).
pub const UUID_LEN: usize = 16;

// ── SeiPayloadType ────────────────────────────────────────────────────────────

/// SEI payload type codes (ITU-T H.274 / ISO/IEC 23002-7 analogues for
/// royalty-free codecs).
///
/// The numeric values are deliberately kept aligned with the H.264/H.265 SEI
/// tables for tooling interoperability, but no H.264/H.265 patents apply here:
/// these structures are used solely inside VP8 extension blocks and AV1
/// METADATA_OBUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SeiPayloadType {
    /// Buffering period (type 0).
    BufferingPeriod = 0,
    /// Picture timing (type 1).
    PictureTiming = 1,
    /// Pan-scan rectangle (type 2).
    PanScanRect = 2,
    /// User data registered by ITU-T Rec. T.35 (type 4).
    UserDataRegistered = 4,
    /// User data unregistered — UUID + opaque bytes (type 5).
    UserDataUnregistered = 5,
    /// Recovery point (type 6).
    RecoveryPoint = 6,
    /// Frame packing arrangement (type 45).
    FramePacking = 45,
    /// Display orientation (type 47).
    DisplayOrientation = 47,
    /// Unknown / custom.
    Unknown = 255,
}

impl SeiPayloadType {
    /// Parse from a raw byte value.  Unrecognised values map to [`Self::Unknown`].
    pub fn from_byte(b: u8) -> Self {
        match b {
            0 => Self::BufferingPeriod,
            1 => Self::PictureTiming,
            2 => Self::PanScanRect,
            4 => Self::UserDataRegistered,
            5 => Self::UserDataUnregistered,
            6 => Self::RecoveryPoint,
            45 => Self::FramePacking,
            47 => Self::DisplayOrientation,
            _ => Self::Unknown,
        }
    }
}

// ── UserDataUnregistered ──────────────────────────────────────────────────────

/// SEI type 5: user-data unregistered.
///
/// Carries a 16-byte UUID (RFC 4122) followed by arbitrary application data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserDataUnregistered {
    /// 128-bit UUID identifying the data format.
    pub uuid: [u8; UUID_LEN],
    /// Arbitrary application payload.
    pub data: Vec<u8>,
}

impl UserDataUnregistered {
    /// Create with an explicit UUID and data.
    pub fn new(uuid: [u8; UUID_LEN], data: Vec<u8>) -> Self {
        Self { uuid, data }
    }

    /// Create with a nil UUID (all zeros) — useful for tests.
    pub fn with_nil_uuid(data: Vec<u8>) -> Self {
        Self::new([0u8; UUID_LEN], data)
    }

    /// Serialise into raw bytes: 16-byte UUID followed by payload.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(UUID_LEN + self.data.len());
        out.extend_from_slice(&self.uuid);
        out.extend_from_slice(&self.data);
        out
    }

    /// Parse from raw bytes.
    pub fn from_bytes(raw: &[u8]) -> CodecResult<Self> {
        if raw.len() < UUID_LEN {
            return Err(CodecError::InvalidBitstream(format!(
                "UserDataUnregistered: need {UUID_LEN} bytes for UUID, got {}",
                raw.len()
            )));
        }
        let mut uuid = [0u8; UUID_LEN];
        uuid.copy_from_slice(&raw[..UUID_LEN]);
        Ok(Self {
            uuid,
            data: raw[UUID_LEN..].to_vec(),
        })
    }
}

// ── PictureTiming ─────────────────────────────────────────────────────────────

/// SEI type 1: picture timing.
///
/// Carries HRD clock tick information and picture-structure metadata.
/// Fields correspond to the simplified timing syntax used in royalty-free
/// codec streams (VP8 extension / AV1 metadata).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PictureTiming {
    /// Clock timestamp flag — `true` if clock fields are valid.
    pub clock_timestamp_flag: bool,
    /// Clock units elapsed since the start of the sequence.
    pub clock_timestamp: u64,
    /// Presentation delay in ticks.
    pub presentation_delay: u32,
    /// Picture structure hint.
    pub pic_struct: PicStructure,
}

/// Picture structure for timing SEI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PicStructure {
    /// Progressive frame.
    Frame = 0,
    /// Top field only.
    TopField = 1,
    /// Bottom field only.
    BottomField = 2,
    /// Top field then bottom field (interleaved).
    TopBottomField = 3,
    /// Bottom field then top field (interleaved).
    BottomTopField = 4,
}

impl PicStructure {
    fn from_byte(b: u8) -> Self {
        match b {
            1 => Self::TopField,
            2 => Self::BottomField,
            3 => Self::TopBottomField,
            4 => Self::BottomTopField,
            _ => Self::Frame,
        }
    }
}

impl PictureTiming {
    /// Create a simple frame-level timing entry with a clock timestamp.
    pub fn frame(clock_timestamp: u64, presentation_delay: u32) -> Self {
        Self {
            clock_timestamp_flag: true,
            clock_timestamp,
            presentation_delay,
            pic_struct: PicStructure::Frame,
        }
    }

    /// Serialise to bytes (16 bytes: flags 1 + clock 8 + delay 4 + pic_struct 1 + pad 2).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(16);
        out.push(u8::from(self.clock_timestamp_flag));
        out.extend_from_slice(&self.clock_timestamp.to_be_bytes());
        out.extend_from_slice(&self.presentation_delay.to_be_bytes());
        out.push(self.pic_struct as u8);
        out.extend_from_slice(&[0u8; 2]); // reserved / padding
        out
    }

    /// Parse from bytes.
    pub fn from_bytes(raw: &[u8]) -> CodecResult<Self> {
        if raw.len() < 14 {
            return Err(CodecError::InvalidBitstream(format!(
                "PictureTiming: need 14 bytes, got {}",
                raw.len()
            )));
        }
        let clock_timestamp_flag = raw[0] != 0;
        let clock_timestamp =
            u64::from_be_bytes(raw[1..9].try_into().map_err(|_| {
                CodecError::InvalidBitstream("PictureTiming: bad clock bytes".into())
            })?);
        let presentation_delay =
            u32::from_be_bytes(raw[9..13].try_into().map_err(|_| {
                CodecError::InvalidBitstream("PictureTiming: bad delay bytes".into())
            })?);
        let pic_struct = PicStructure::from_byte(raw[13]);
        Ok(Self {
            clock_timestamp_flag,
            clock_timestamp,
            presentation_delay,
            pic_struct,
        })
    }
}

// ── SeiMessage ────────────────────────────────────────────────────────────────

/// A single SEI message: a payload type tag plus the encoded bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeiMessage {
    /// Payload type discriminant.
    pub payload_type: SeiPayloadType,
    /// Raw serialised payload bytes.
    pub payload: Vec<u8>,
}

impl SeiMessage {
    /// Create a raw SEI message with the given type and payload.
    pub fn new(payload_type: SeiPayloadType, payload: Vec<u8>) -> Self {
        Self {
            payload_type,
            payload,
        }
    }

    /// Build a `UserDataUnregistered` SEI message.
    pub fn user_data_unregistered(udu: &UserDataUnregistered) -> Self {
        Self::new(SeiPayloadType::UserDataUnregistered, udu.to_bytes())
    }

    /// Build a `PictureTiming` SEI message.
    pub fn picture_timing(pt: &PictureTiming) -> Self {
        Self::new(SeiPayloadType::PictureTiming, pt.to_bytes())
    }

    /// Parse the payload as [`UserDataUnregistered`].
    pub fn as_user_data_unregistered(&self) -> CodecResult<UserDataUnregistered> {
        if self.payload_type != SeiPayloadType::UserDataUnregistered {
            return Err(CodecError::InvalidData(
                "SEI: expected UserDataUnregistered payload type".into(),
            ));
        }
        UserDataUnregistered::from_bytes(&self.payload)
    }

    /// Parse the payload as [`PictureTiming`].
    pub fn as_picture_timing(&self) -> CodecResult<PictureTiming> {
        if self.payload_type != SeiPayloadType::PictureTiming {
            return Err(CodecError::InvalidData(
                "SEI: expected PictureTiming payload type".into(),
            ));
        }
        PictureTiming::from_bytes(&self.payload)
    }
}

// ── SeiEncoder / SeiDecoder ───────────────────────────────────────────────────

/// Wire format for a sequence of SEI messages.
///
/// Each message is framed as:
/// ```text
/// [type: u8][length: u32 big-endian][payload: <length> bytes]
/// ```
#[derive(Debug, Default)]
pub struct SeiEncoder {
    buf: Vec<u8>,
}

impl SeiEncoder {
    /// Create a new encoder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a single [`SeiMessage`].
    pub fn write_message(&mut self, msg: &SeiMessage) {
        self.buf.push(msg.payload_type as u8);
        let len = msg.payload.len() as u32;
        self.buf.extend_from_slice(&len.to_be_bytes());
        self.buf.extend_from_slice(&msg.payload);
    }

    /// Append multiple messages.
    pub fn write_messages(&mut self, msgs: &[SeiMessage]) {
        for msg in msgs {
            self.write_message(msg);
        }
    }

    /// Consume the encoder and return the serialised bytes.
    pub fn finish(self) -> Vec<u8> {
        self.buf
    }

    /// Return the current byte length of the accumulated buffer.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Returns `true` if nothing has been written yet.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

/// Parses a byte buffer produced by [`SeiEncoder`] into a list of [`SeiMessage`]s.
#[derive(Debug)]
pub struct SeiDecoder<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> SeiDecoder<'a> {
    /// Create a decoder over `data`.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Read the next message, or `None` if at end.
    pub fn next_message(&mut self) -> CodecResult<Option<SeiMessage>> {
        if self.pos >= self.data.len() {
            return Ok(None);
        }
        // Read type byte.
        let type_byte = self.data[self.pos];
        self.pos += 1;

        // Read 4-byte big-endian length.
        if self.pos + 4 > self.data.len() {
            return Err(CodecError::InvalidBitstream(
                "SEI: truncated length field".into(),
            ));
        }
        let length = u32::from_be_bytes(
            self.data[self.pos..self.pos + 4]
                .try_into()
                .map_err(|_| CodecError::InvalidBitstream("SEI: bad length bytes".into()))?,
        ) as usize;
        self.pos += 4;

        // Read payload.
        if self.pos + length > self.data.len() {
            return Err(CodecError::InvalidBitstream(format!(
                "SEI: payload truncated (need {length}, have {})",
                self.data.len() - self.pos
            )));
        }
        let payload = self.data[self.pos..self.pos + length].to_vec();
        self.pos += length;

        Ok(Some(SeiMessage {
            payload_type: SeiPayloadType::from_byte(type_byte),
            payload,
        }))
    }

    /// Collect all messages.
    pub fn collect_all(&mut self) -> CodecResult<Vec<SeiMessage>> {
        let mut result = Vec::new();
        while let Some(msg) = self.next_message()? {
            result.push(msg);
        }
        Ok(result)
    }
}

// ── Av1MetadataObu ────────────────────────────────────────────────────────────

/// A minimal AV1 METADATA_OBU wrapper.
///
/// AV1 metadata OBUs (type 5) carry a `metadata_type` varint followed by
/// the payload.  This struct uses a simple fixed-length encoding for the
/// metadata type (1 byte) to keep the implementation patent-free and simple.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Av1MetadataObu {
    /// AV1 metadata type.  Commonly 4 (ITUT T35) or 5 (custom).
    pub metadata_type: u8,
    /// OBU extension byte present flag.
    pub extension_flag: bool,
    /// Serialised SEI payload.
    pub payload: Vec<u8>,
}

impl Av1MetadataObu {
    /// Wrap a pre-serialised SEI payload into a METADATA_OBU.
    pub fn new(metadata_type: u8, payload: Vec<u8>) -> Self {
        Self {
            metadata_type,
            extension_flag: false,
            payload,
        }
    }

    /// Serialise to a byte slice ready for muxing.
    ///
    /// Wire format:
    /// ```text
    /// [obu_header: 1 byte][extension? 1 byte if extension_flag][metadata_type: 1 byte][payload...]
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        // OBU header: type=5 (0b0101 << 3), extension_flag bit, has_size_field=0
        let extension_bit = u8::from(self.extension_flag) << 2;
        let obu_header = (AV1_OBU_TYPE_METADATA << 3) | extension_bit;

        let mut out = Vec::with_capacity(2 + self.payload.len());
        out.push(obu_header);
        if self.extension_flag {
            out.push(0x00); // placeholder extension byte
        }
        out.push(self.metadata_type);
        out.extend_from_slice(&self.payload);
        out
    }

    /// Parse an AV1 METADATA_OBU from raw bytes.
    pub fn from_bytes(raw: &[u8]) -> CodecResult<Self> {
        if raw.is_empty() {
            return Err(CodecError::InvalidBitstream(
                "Av1MetadataObu: empty input".into(),
            ));
        }
        let obu_header = raw[0];
        let obu_type = (obu_header >> 3) & 0x0F;
        if obu_type != AV1_OBU_TYPE_METADATA {
            return Err(CodecError::InvalidBitstream(format!(
                "Av1MetadataObu: expected type {AV1_OBU_TYPE_METADATA}, got {obu_type}"
            )));
        }
        let extension_flag = (obu_header >> 2) & 1 != 0;
        let mut pos = 1usize;
        if extension_flag {
            pos += 1; // skip extension byte
        }
        if pos >= raw.len() {
            return Err(CodecError::InvalidBitstream(
                "Av1MetadataObu: missing metadata_type".into(),
            ));
        }
        let metadata_type = raw[pos];
        pos += 1;
        let payload = raw[pos..].to_vec();
        Ok(Self {
            metadata_type,
            extension_flag,
            payload,
        })
    }
}

// ── Vp8MetadataBlock ─────────────────────────────────────────────────────────

/// A VP8 user-data extension block.
///
/// VP8 does not define a formal SEI syntax, so this structure embeds metadata
/// using an application-level convention: a marker byte (`0xFE`) followed by a
/// 4-byte big-endian length, followed by the payload.  This block is placed
/// after the last partition data and before the end of the data partition.
///
/// **Important**: this is not part of the VP8 bitstream specification.  It is
/// OxiMedia's application-level convention for attaching metadata to VP8
/// packets when the container does not provide a side-data channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Vp8MetadataBlock {
    /// Serialised SEI payload (output of [`SeiEncoder::finish`]).
    pub payload: Vec<u8>,
}

impl Vp8MetadataBlock {
    /// Create a VP8 metadata block from a pre-serialised SEI payload.
    pub fn new(payload: Vec<u8>) -> Self {
        Self { payload }
    }

    /// Serialise to bytes.
    ///
    /// Wire format: `[0xFE][length: u32 BE][payload...]`
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(5 + self.payload.len());
        out.push(VP8_USER_DATA_MARKER);
        let len = self.payload.len() as u32;
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(&self.payload);
        out
    }

    /// Parse a VP8 metadata block from raw bytes.
    pub fn from_bytes(raw: &[u8]) -> CodecResult<Self> {
        if raw.is_empty() || raw[0] != VP8_USER_DATA_MARKER {
            return Err(CodecError::InvalidBitstream(
                "Vp8MetadataBlock: missing marker byte".into(),
            ));
        }
        if raw.len() < 5 {
            return Err(CodecError::InvalidBitstream(
                "Vp8MetadataBlock: truncated header".into(),
            ));
        }
        let length = u32::from_be_bytes(
            raw[1..5]
                .try_into()
                .map_err(|_| CodecError::InvalidBitstream("Vp8MetadataBlock: bad length".into()))?,
        ) as usize;
        if raw.len() < 5 + length {
            return Err(CodecError::InvalidBitstream(format!(
                "Vp8MetadataBlock: payload truncated (need {length}, have {})",
                raw.len() - 5
            )));
        }
        Ok(Self {
            payload: raw[5..5 + length].to_vec(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_data_unregistered_roundtrip() {
        let uuid = [0x01u8; UUID_LEN];
        let data = b"hello sei world".to_vec();
        let udu = UserDataUnregistered::new(uuid, data.clone());
        let raw = udu.to_bytes();
        let parsed = UserDataUnregistered::from_bytes(&raw).unwrap();
        assert_eq!(parsed.uuid, uuid);
        assert_eq!(parsed.data, data);
    }

    #[test]
    fn test_user_data_unregistered_too_short() {
        let result = UserDataUnregistered::from_bytes(&[0u8; 10]);
        assert!(result.is_err());
    }

    #[test]
    fn test_picture_timing_roundtrip() {
        let pt = PictureTiming::frame(123_456_789, 3000);
        let raw = pt.to_bytes();
        let parsed = PictureTiming::from_bytes(&raw).unwrap();
        assert_eq!(parsed.clock_timestamp, 123_456_789);
        assert_eq!(parsed.presentation_delay, 3000);
        assert_eq!(parsed.pic_struct, PicStructure::Frame);
        assert!(parsed.clock_timestamp_flag);
    }

    #[test]
    fn test_picture_timing_too_short() {
        let result = PictureTiming::from_bytes(&[0u8; 5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_sei_encoder_decoder_roundtrip() {
        let udu = UserDataUnregistered::with_nil_uuid(b"test payload".to_vec());
        let pt = PictureTiming::frame(999, 500);

        let mut enc = SeiEncoder::new();
        enc.write_message(&SeiMessage::user_data_unregistered(&udu));
        enc.write_message(&SeiMessage::picture_timing(&pt));
        let bytes = enc.finish();

        let mut dec = SeiDecoder::new(&bytes);
        let messages = dec.collect_all().unwrap();
        assert_eq!(messages.len(), 2);

        assert_eq!(
            messages[0].payload_type,
            SeiPayloadType::UserDataUnregistered
        );
        let recovered_udu = messages[0].as_user_data_unregistered().unwrap();
        assert_eq!(recovered_udu.data, b"test payload");

        assert_eq!(messages[1].payload_type, SeiPayloadType::PictureTiming);
        let recovered_pt = messages[1].as_picture_timing().unwrap();
        assert_eq!(recovered_pt.clock_timestamp, 999);
    }

    #[test]
    fn test_sei_decoder_truncated_length() {
        // Only type byte, no length field.
        let bad = &[SeiPayloadType::PictureTiming as u8];
        let mut dec = SeiDecoder::new(bad);
        assert!(dec.next_message().is_err());
    }

    #[test]
    fn test_sei_decoder_truncated_payload() {
        let mut enc = SeiEncoder::new();
        enc.write_message(&SeiMessage::new(
            SeiPayloadType::UserDataUnregistered,
            vec![0u8; 20],
        ));
        let mut bytes = enc.finish();
        // Truncate the payload
        bytes.truncate(bytes.len() - 5);
        let mut dec = SeiDecoder::new(&bytes);
        assert!(dec.next_message().is_err());
    }

    #[test]
    fn test_av1_metadata_obu_roundtrip() {
        let sei_payload = b"av1 sei data".to_vec();
        let obu = Av1MetadataObu::new(5, sei_payload.clone());
        let bytes = obu.to_bytes();
        let parsed = Av1MetadataObu::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.metadata_type, 5);
        assert_eq!(parsed.payload, sei_payload);
        assert!(!parsed.extension_flag);
    }

    #[test]
    fn test_av1_metadata_obu_wrong_type() {
        // Build a header with OBU type = 1 (sequence header) instead of 5
        let bad = &[0x08u8, 0x00, 0x00]; // type = 1 << 3 = 0x08
        let result = Av1MetadataObu::from_bytes(bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_vp8_metadata_block_roundtrip() {
        let payload = b"vp8 metadata payload".to_vec();
        let block = Vp8MetadataBlock::new(payload.clone());
        let bytes = block.to_bytes();
        assert_eq!(bytes[0], VP8_USER_DATA_MARKER);
        let parsed = Vp8MetadataBlock::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.payload, payload);
    }

    #[test]
    fn test_vp8_metadata_block_bad_marker() {
        let result = Vp8MetadataBlock::from_bytes(&[0x00, 0x00, 0x00, 0x00, 0x00]);
        assert!(result.is_err());
    }

    #[test]
    fn test_sei_payload_type_roundtrip() {
        for &(byte, expected) in &[
            (0u8, SeiPayloadType::BufferingPeriod),
            (1, SeiPayloadType::PictureTiming),
            (5, SeiPayloadType::UserDataUnregistered),
            (255, SeiPayloadType::Unknown),
        ] {
            assert_eq!(SeiPayloadType::from_byte(byte), expected);
        }
    }
}
