//! Core Audio Format (CAF) container support.
//!
//! Apple's CAF file format (documented in Apple Technical Note TN2267 and
//! the Core Audio Format Specification 1.0) wraps raw or compressed audio
//! in a simple chunk-based structure.  CAF supports arbitrarily large files
//! (no 4 GB limit) and any sample rate.
//!
//! # Format overview
//!
//! ```text
//! File header  (8 bytes)
//!   "caff" magic + version (0x0001) + flags (0x0000)
//!
//! Chunk sequence (one or more):
//!   chunk_type  (4 bytes, ASCII)
//!   chunk_size  (8 bytes, i64; -1 means "to end of file" for 'data')
//!   chunk_data  (chunk_size bytes)
//! ```
//!
//! This module implements:
//! - [`CafFileHeader`] — file-level header
//! - [`CafChunkHeader`] — per-chunk type + size
//! - [`CafAudioDescription`] — `desc` chunk (ASBD)
//! - [`CafChunkType`] — typed enum of well-known chunk types
//! - [`parse_caf_file_header`] / [`parse_caf_chunk_header`]
//! - [`serialize_caf_file_header`] / [`serialize_caf_chunk_header`]
//! - [`CafReader`] — sequential chunk reader over a byte slice
//! - [`CafWriter`] — in-memory CAF file writer
//!
//! # Example
//!
//! ```
//! use oximedia_container::caf::{CafWriter, CafAudioDescription, CafChunkType};
//!
//! let desc = CafAudioDescription {
//!     sample_rate: 48_000.0,
//!     format_id: *b"lpcm",
//!     format_flags: 0x0C, // float + little-endian
//!     bytes_per_packet: 4,
//!     frames_per_packet: 1,
//!     channels_per_frame: 1,
//!     bits_per_channel: 32,
//! };
//! let mut writer = CafWriter::new(desc);
//! writer.append_audio_data(&[0u8; 16]);
//! let bytes = writer.finish();
//! assert!(bytes.len() > 12);
//! ```

use std::fmt;
use thiserror::Error;

// ─── Errors ───────────────────────────────────────────────────────────────────

/// Errors that can occur during CAF parsing or writing.
#[derive(Debug, Error)]
pub enum CafError {
    /// Buffer is shorter than required.
    #[error("buffer too short: need {needed} bytes, got {got}")]
    BufferTooShort { needed: usize, got: usize },

    /// Magic bytes are not "caff".
    #[error("invalid CAF magic: expected b\"caff\", got {0:?}")]
    InvalidMagic([u8; 4]),

    /// Unsupported file version (must be 1).
    #[error("unsupported CAF version: {0}")]
    UnsupportedVersion(u16),

    /// A chunk extends past the buffer end.
    #[error("chunk extends past end of buffer")]
    ChunkOverflow,

    /// A required chunk was not found.
    #[error("required chunk '{0}' not found")]
    MissingChunk(String),

    /// The `desc` chunk has invalid size.
    #[error("invalid 'desc' chunk size: expected 32, got {0}")]
    InvalidDescSize(i64),

    /// Integer overflow or conversion error.
    #[error("integer overflow in chunk size")]
    Overflow,
}

// ─── Constants ────────────────────────────────────────────────────────────────

/// Magic bytes for a CAF file.
pub const CAF_MAGIC: &[u8; 4] = b"caff";
/// CAF file format version (big-endian u16 value 1).
pub const CAF_VERSION: u16 = 1;
/// Byte size of the CAF file header.
pub const CAF_FILE_HEADER_SIZE: usize = 8;
/// Byte size of a CAF chunk header.
pub const CAF_CHUNK_HEADER_SIZE: usize = 12;
/// Byte size of the Audio Stream Basic Description (ASBD / `desc` chunk).
pub const CAF_ASBD_SIZE: usize = 32;

// ─── Chunk type ───────────────────────────────────────────────────────────────

/// Well-known CAF chunk types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CafChunkType {
    /// `desc` — Audio Stream Basic Description (required, must be first).
    AudioDescription,
    /// `data` — Audio sample data.
    AudioData,
    /// `pakt` — Packet table (for variable bit-rate audio).
    PacketTable,
    /// `chan` — Channel layout.
    ChannelLayout,
    /// `info` — Metadata strings.
    Information,
    /// `kuki` — Magic cookie (codec-private data, e.g. Opus/Vorbis headers).
    MagicCookie,
    /// `mark` — Marker chunk (named time positions).
    Marker,
    /// `regn` — Region chunk.
    Region,
    /// `umid` — Unique material identifier.
    UniqueId,
    /// Unknown / user-defined chunk.
    Unknown([u8; 4]),
}

impl CafChunkType {
    /// Convert a 4-byte ASCII code to a [`CafChunkType`].
    #[must_use]
    pub fn from_bytes(b: [u8; 4]) -> Self {
        match &b {
            b"desc" => Self::AudioDescription,
            b"data" => Self::AudioData,
            b"pakt" => Self::PacketTable,
            b"chan" => Self::ChannelLayout,
            b"info" => Self::Information,
            b"kuki" => Self::MagicCookie,
            b"mark" => Self::Marker,
            b"regn" => Self::Region,
            b"umid" => Self::UniqueId,
            _ => Self::Unknown(b),
        }
    }

    /// Return the 4-byte ASCII representation.
    #[must_use]
    pub fn to_bytes(self) -> [u8; 4] {
        match self {
            Self::AudioDescription => *b"desc",
            Self::AudioData => *b"data",
            Self::PacketTable => *b"pakt",
            Self::ChannelLayout => *b"chan",
            Self::Information => *b"info",
            Self::MagicCookie => *b"kuki",
            Self::Marker => *b"mark",
            Self::Region => *b"regn",
            Self::UniqueId => *b"umid",
            Self::Unknown(b) => b,
        }
    }
}

impl fmt::Display for CafChunkType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let b = self.to_bytes();
        let s = std::str::from_utf8(&b).unwrap_or("????");
        write!(f, "'{s}'")
    }
}

// ─── File header ─────────────────────────────────────────────────────────────

/// The 8-byte CAF file header.
///
/// ```text
/// offset  size  field
///    0     4    file_type  ("caff")
///    4     2    file_version  (big-endian, must be 1)
///    6     2    file_flags  (must be 0)
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CafFileHeader {
    /// File format version (must be 1).
    pub version: u16,
    /// File flags (must be 0 per spec).
    pub flags: u16,
}

impl CafFileHeader {
    /// Create a default valid CAF file header.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            version: CAF_VERSION,
            flags: 0,
        }
    }
}

impl Default for CafFileHeader {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse a CAF file header from `buf`.
///
/// # Errors
///
/// Returns [`CafError`] if the buffer is too short, the magic is wrong, or the
/// version is not 1.
pub fn parse_caf_file_header(buf: &[u8]) -> Result<CafFileHeader, CafError> {
    if buf.len() < CAF_FILE_HEADER_SIZE {
        return Err(CafError::BufferTooShort {
            needed: CAF_FILE_HEADER_SIZE,
            got: buf.len(),
        });
    }
    let magic: [u8; 4] = buf[0..4].try_into().map_err(|_| CafError::BufferTooShort {
        needed: 4,
        got: buf.len(),
    })?;
    if &magic != CAF_MAGIC {
        return Err(CafError::InvalidMagic(magic));
    }
    let version =
        u16::from_be_bytes(buf[4..6].try_into().map_err(|_| CafError::BufferTooShort {
            needed: 6,
            got: buf.len(),
        })?);
    if version != CAF_VERSION {
        return Err(CafError::UnsupportedVersion(version));
    }
    let flags = u16::from_be_bytes(buf[6..8].try_into().map_err(|_| CafError::BufferTooShort {
        needed: 8,
        got: buf.len(),
    })?);
    Ok(CafFileHeader { version, flags })
}

/// Serialize a CAF file header to bytes (8 bytes).
#[must_use]
pub fn serialize_caf_file_header(hdr: &CafFileHeader) -> [u8; 8] {
    let mut out = [0u8; 8];
    out[0..4].copy_from_slice(CAF_MAGIC);
    out[4..6].copy_from_slice(&hdr.version.to_be_bytes());
    out[6..8].copy_from_slice(&hdr.flags.to_be_bytes());
    out
}

// ─── Chunk header ─────────────────────────────────────────────────────────────

/// The 12-byte header that precedes every CAF chunk body.
///
/// ```text
/// offset  size  field
///    0     4    chunk_type  (ASCII)
///    4     8    chunk_size  (big-endian i64; -1 = to end of file for 'data')
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CafChunkHeader {
    /// Chunk type.
    pub chunk_type: CafChunkType,
    /// Chunk body size in bytes, or `-1` for the open-ended `data` chunk.
    pub chunk_size: i64,
}

impl CafChunkHeader {
    /// Create a chunk header.
    #[must_use]
    pub const fn new(chunk_type: CafChunkType, chunk_size: i64) -> Self {
        Self {
            chunk_type,
            chunk_size,
        }
    }
}

/// Parse a CAF chunk header from `buf`.
///
/// # Errors
///
/// Returns [`CafError`] if the buffer is shorter than 12 bytes.
pub fn parse_caf_chunk_header(buf: &[u8]) -> Result<CafChunkHeader, CafError> {
    if buf.len() < CAF_CHUNK_HEADER_SIZE {
        return Err(CafError::BufferTooShort {
            needed: CAF_CHUNK_HEADER_SIZE,
            got: buf.len(),
        });
    }
    let type_bytes: [u8; 4] = buf[0..4].try_into().map_err(|_| CafError::BufferTooShort {
        needed: 4,
        got: buf.len(),
    })?;
    let chunk_type = CafChunkType::from_bytes(type_bytes);
    let chunk_size =
        i64::from_be_bytes(
            buf[4..12]
                .try_into()
                .map_err(|_| CafError::BufferTooShort {
                    needed: 12,
                    got: buf.len(),
                })?,
        );
    Ok(CafChunkHeader {
        chunk_type,
        chunk_size,
    })
}

/// Serialize a CAF chunk header to 12 bytes.
#[must_use]
pub fn serialize_caf_chunk_header(hdr: &CafChunkHeader) -> [u8; 12] {
    let mut out = [0u8; 12];
    out[0..4].copy_from_slice(&hdr.chunk_type.to_bytes());
    out[4..12].copy_from_slice(&hdr.chunk_size.to_be_bytes());
    out
}

// ─── AudioStreamBasicDescription ─────────────────────────────────────────────

/// Audio Stream Basic Description — the `desc` chunk body.
///
/// All fields are big-endian on the wire.
#[derive(Clone, Debug, PartialEq)]
pub struct CafAudioDescription {
    /// Sample rate in Hz (IEEE 754 big-endian f64).
    pub sample_rate: f64,
    /// Four-character-code identifying the audio format (e.g. `lpcm`, `opus`).
    pub format_id: [u8; 4],
    /// Format-specific flags.
    pub format_flags: u32,
    /// Bytes per packet (0 = variable bit-rate).
    pub bytes_per_packet: u32,
    /// Frames per packet (0 = variable).
    pub frames_per_packet: u32,
    /// Channels per audio frame.
    pub channels_per_frame: u32,
    /// Valid bits per channel (0 for compressed formats).
    pub bits_per_channel: u32,
}

impl CafAudioDescription {
    /// Create a description for raw 32-bit float mono audio at `sample_rate` Hz.
    #[must_use]
    pub fn pcm_f32_mono(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            format_id: *b"lpcm",
            format_flags: 0x0C, // kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked
            bytes_per_packet: 4,
            frames_per_packet: 1,
            channels_per_frame: 1,
            bits_per_channel: 32,
        }
    }

    /// Create a description for raw 16-bit integer stereo audio.
    #[must_use]
    pub fn pcm_i16_stereo(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            format_id: *b"lpcm",
            format_flags: 0x0E, // kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked | kAudioFormatFlagIsBigEndian
            bytes_per_packet: 4,
            frames_per_packet: 1,
            channels_per_frame: 2,
            bits_per_channel: 16,
        }
    }
}

/// Parse a `CafAudioDescription` from 32 bytes of chunk body.
///
/// # Errors
///
/// Returns [`CafError`] if `buf` is shorter than 32 bytes.
pub fn parse_audio_description(buf: &[u8]) -> Result<CafAudioDescription, CafError> {
    if buf.len() < CAF_ASBD_SIZE {
        return Err(CafError::BufferTooShort {
            needed: CAF_ASBD_SIZE,
            got: buf.len(),
        });
    }
    let rate_bytes: [u8; 8] = buf[0..8].try_into().map_err(|_| CafError::BufferTooShort {
        needed: 8,
        got: buf.len(),
    })?;
    let sample_rate = f64::from_be_bytes(rate_bytes);
    let format_id: [u8; 4] = buf[8..12]
        .try_into()
        .map_err(|_| CafError::BufferTooShort {
            needed: 12,
            got: buf.len(),
        })?;
    let format_flags =
        u32::from_be_bytes(
            buf[12..16]
                .try_into()
                .map_err(|_| CafError::BufferTooShort {
                    needed: 16,
                    got: buf.len(),
                })?,
        );
    let bytes_per_packet =
        u32::from_be_bytes(
            buf[16..20]
                .try_into()
                .map_err(|_| CafError::BufferTooShort {
                    needed: 20,
                    got: buf.len(),
                })?,
        );
    let frames_per_packet =
        u32::from_be_bytes(
            buf[20..24]
                .try_into()
                .map_err(|_| CafError::BufferTooShort {
                    needed: 24,
                    got: buf.len(),
                })?,
        );
    let channels_per_frame =
        u32::from_be_bytes(
            buf[24..28]
                .try_into()
                .map_err(|_| CafError::BufferTooShort {
                    needed: 28,
                    got: buf.len(),
                })?,
        );
    let bits_per_channel =
        u32::from_be_bytes(
            buf[28..32]
                .try_into()
                .map_err(|_| CafError::BufferTooShort {
                    needed: 32,
                    got: buf.len(),
                })?,
        );
    Ok(CafAudioDescription {
        sample_rate,
        format_id,
        format_flags,
        bytes_per_packet,
        frames_per_packet,
        channels_per_frame,
        bits_per_channel,
    })
}

/// Serialize a `CafAudioDescription` to 32 bytes.
#[must_use]
pub fn serialize_audio_description(desc: &CafAudioDescription) -> [u8; CAF_ASBD_SIZE] {
    let mut out = [0u8; CAF_ASBD_SIZE];
    out[0..8].copy_from_slice(&desc.sample_rate.to_be_bytes());
    out[8..12].copy_from_slice(&desc.format_id);
    out[12..16].copy_from_slice(&desc.format_flags.to_be_bytes());
    out[16..20].copy_from_slice(&desc.bytes_per_packet.to_be_bytes());
    out[20..24].copy_from_slice(&desc.frames_per_packet.to_be_bytes());
    out[24..28].copy_from_slice(&desc.channels_per_frame.to_be_bytes());
    out[28..32].copy_from_slice(&desc.bits_per_channel.to_be_bytes());
    out
}

// ─── CafChunk ─────────────────────────────────────────────────────────────────

/// A parsed CAF chunk with header and raw body bytes.
#[derive(Clone, Debug)]
pub struct CafChunk {
    /// Chunk header.
    pub header: CafChunkHeader,
    /// Raw body bytes (length equals `header.chunk_size` unless -1).
    pub body: Vec<u8>,
}

// ─── CafReader ────────────────────────────────────────────────────────────────

/// Sequential CAF chunk reader over a byte slice.
///
/// Iterates over chunks in file order.  Call [`CafReader::next_chunk`] repeatedly
/// until it returns `Ok(None)` (EOF) or an `Err`.
pub struct CafReader<'a> {
    buf: &'a [u8],
    pos: usize,
    file_header_parsed: bool,
}

impl<'a> CafReader<'a> {
    /// Create a new reader for `buf`.
    ///
    /// The file header is *not* read until [`CafReader::read_file_header`] is called.
    #[must_use]
    pub fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            pos: 0,
            file_header_parsed: false,
        }
    }

    /// Parse and return the CAF file header.
    ///
    /// Must be called once before [`CafReader::next_chunk`].
    ///
    /// # Errors
    ///
    /// Returns [`CafError`] if the header is invalid.
    pub fn read_file_header(&mut self) -> Result<CafFileHeader, CafError> {
        let hdr = parse_caf_file_header(&self.buf[self.pos..])?;
        self.pos += CAF_FILE_HEADER_SIZE;
        self.file_header_parsed = true;
        Ok(hdr)
    }

    /// Read the next chunk.
    ///
    /// Returns `Ok(None)` at end of buffer, or `Ok(Some(chunk))`.
    ///
    /// # Errors
    ///
    /// Returns [`CafError`] on malformed data.
    pub fn next_chunk(&mut self) -> Result<Option<CafChunk>, CafError> {
        if self.pos >= self.buf.len() {
            return Ok(None);
        }
        let chunk_hdr_buf = &self.buf[self.pos..];
        if chunk_hdr_buf.len() < CAF_CHUNK_HEADER_SIZE {
            // Trailing bytes — not enough for another header; treat as EOF.
            return Ok(None);
        }
        let header = parse_caf_chunk_header(chunk_hdr_buf)?;
        self.pos += CAF_CHUNK_HEADER_SIZE;

        let body_size: usize = if header.chunk_size < 0 {
            // Open-ended chunk: consume to end of buffer.
            self.buf.len().saturating_sub(self.pos)
        } else {
            header
                .chunk_size
                .try_into()
                .map_err(|_| CafError::Overflow)?
        };

        if self.pos + body_size > self.buf.len() {
            return Err(CafError::ChunkOverflow);
        }
        let body = self.buf[self.pos..self.pos + body_size].to_vec();
        self.pos += body_size;
        Ok(Some(CafChunk { header, body }))
    }

    /// Return the current byte offset in the input buffer.
    #[must_use]
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Collect all remaining chunks into a `Vec`.
    ///
    /// # Errors
    ///
    /// Returns the first error encountered.
    pub fn collect_chunks(&mut self) -> Result<Vec<CafChunk>, CafError> {
        let mut out = Vec::new();
        while let Some(chunk) = self.next_chunk()? {
            out.push(chunk);
        }
        Ok(out)
    }
}

// ─── CafWriter ────────────────────────────────────────────────────────────────

/// In-memory CAF file writer.
///
/// Produces a complete CAF file with a `desc` chunk (required, first), an
/// optional `kuki` magic cookie, and a `data` chunk containing audio samples.
pub struct CafWriter {
    desc: CafAudioDescription,
    magic_cookie: Option<Vec<u8>>,
    audio_data: Vec<u8>,
    extra_chunks: Vec<CafChunk>,
}

impl CafWriter {
    /// Create a writer with the given audio description.
    #[must_use]
    pub fn new(desc: CafAudioDescription) -> Self {
        Self {
            desc,
            magic_cookie: None,
            audio_data: Vec::new(),
            extra_chunks: Vec::new(),
        }
    }

    /// Set the magic cookie (codec-private data, `kuki` chunk).
    pub fn set_magic_cookie(&mut self, cookie: Vec<u8>) {
        self.magic_cookie = Some(cookie);
    }

    /// Append raw audio sample bytes to the `data` chunk.
    pub fn append_audio_data(&mut self, data: &[u8]) {
        self.audio_data.extend_from_slice(data);
    }

    /// Add a custom chunk.
    pub fn add_chunk(&mut self, chunk: CafChunk) {
        self.extra_chunks.push(chunk);
    }

    /// Serialize the entire CAF file to a byte vector.
    #[must_use]
    pub fn finish(self) -> Vec<u8> {
        let mut out = Vec::new();

        // File header.
        out.extend_from_slice(&serialize_caf_file_header(&CafFileHeader::new()));

        // `desc` chunk (required first).
        let desc_body = serialize_audio_description(&self.desc);
        let desc_hdr = CafChunkHeader::new(CafChunkType::AudioDescription, CAF_ASBD_SIZE as i64);
        out.extend_from_slice(&serialize_caf_chunk_header(&desc_hdr));
        out.extend_from_slice(&desc_body);

        // `kuki` magic cookie chunk (optional).
        if let Some(cookie) = &self.magic_cookie {
            let kuki_hdr = CafChunkHeader::new(CafChunkType::MagicCookie, cookie.len() as i64);
            out.extend_from_slice(&serialize_caf_chunk_header(&kuki_hdr));
            out.extend_from_slice(cookie);
        }

        // Extra user chunks.
        for chunk in &self.extra_chunks {
            out.extend_from_slice(&serialize_caf_chunk_header(&chunk.header));
            out.extend_from_slice(&chunk.body);
        }

        // `data` chunk — 4-byte edit count (u32 big-endian, value 0) followed
        // by audio bytes.  Use open-ended size (-1) per spec for stream compat.
        let data_chunk_hdr = CafChunkHeader::new(CafChunkType::AudioData, -1i64);
        out.extend_from_slice(&serialize_caf_chunk_header(&data_chunk_hdr));
        out.extend_from_slice(&0u32.to_be_bytes()); // edit count
        out.extend_from_slice(&self.audio_data);

        out
    }
}

// ─── Helper: read audio description from parsed chunks ───────────────────────

/// Extract and parse the `desc` chunk from a list of chunks.
///
/// # Errors
///
/// Returns [`CafError::MissingChunk`] if no `desc` chunk is present, or
/// [`CafError::InvalidDescSize`] if the body is not exactly 32 bytes.
pub fn extract_audio_description(chunks: &[CafChunk]) -> Result<CafAudioDescription, CafError> {
    let desc_chunk = chunks
        .iter()
        .find(|c| c.header.chunk_type == CafChunkType::AudioDescription)
        .ok_or_else(|| CafError::MissingChunk("desc".into()))?;

    if desc_chunk.header.chunk_size != CAF_ASBD_SIZE as i64 {
        return Err(CafError::InvalidDescSize(desc_chunk.header.chunk_size));
    }

    parse_audio_description(&desc_chunk.body)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── file header ──

    #[test]
    fn test_file_header_round_trip() {
        let hdr = CafFileHeader::new();
        let bytes = serialize_caf_file_header(&hdr);
        let parsed = parse_caf_file_header(&bytes).expect("parse failed");
        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.flags, 0);
    }

    #[test]
    fn test_file_header_invalid_magic() {
        let mut bytes = serialize_caf_file_header(&CafFileHeader::new());
        bytes[0] = b'X';
        assert!(matches!(
            parse_caf_file_header(&bytes),
            Err(CafError::InvalidMagic(_))
        ));
    }

    #[test]
    fn test_file_header_unsupported_version() {
        let mut bytes = serialize_caf_file_header(&CafFileHeader::new());
        bytes[4] = 0;
        bytes[5] = 2; // version = 2
        assert!(matches!(
            parse_caf_file_header(&bytes),
            Err(CafError::UnsupportedVersion(2))
        ));
    }

    // ── chunk header ──

    #[test]
    fn test_chunk_header_round_trip() {
        let hdr = CafChunkHeader::new(CafChunkType::AudioDescription, 32);
        let bytes = serialize_caf_chunk_header(&hdr);
        let parsed = parse_caf_chunk_header(&bytes).expect("parse failed");
        assert_eq!(parsed.chunk_type, CafChunkType::AudioDescription);
        assert_eq!(parsed.chunk_size, 32);
    }

    #[test]
    fn test_chunk_type_unknown() {
        let hdr = CafChunkHeader::new(CafChunkType::Unknown(*b"xyzw"), 0);
        let bytes = serialize_caf_chunk_header(&hdr);
        let parsed = parse_caf_chunk_header(&bytes).expect("parse failed");
        assert_eq!(parsed.chunk_type, CafChunkType::Unknown(*b"xyzw"));
    }

    // ── audio description ──

    #[test]
    fn test_audio_description_round_trip() {
        let desc = CafAudioDescription::pcm_f32_mono(44_100.0);
        let bytes = serialize_audio_description(&desc);
        let parsed = parse_audio_description(&bytes).expect("parse failed");
        assert!((parsed.sample_rate - 44_100.0).abs() < f64::EPSILON);
        assert_eq!(parsed.format_id, *b"lpcm");
        assert_eq!(parsed.channels_per_frame, 1);
        assert_eq!(parsed.bits_per_channel, 32);
    }

    #[test]
    fn test_audio_description_stereo() {
        let desc = CafAudioDescription::pcm_i16_stereo(48_000.0);
        let bytes = serialize_audio_description(&desc);
        let parsed = parse_audio_description(&bytes).expect("parse failed");
        assert_eq!(parsed.channels_per_frame, 2);
        assert_eq!(parsed.bits_per_channel, 16);
        assert_eq!(parsed.bytes_per_packet, 4);
    }

    // ── writer + reader ──

    #[test]
    fn test_writer_produces_valid_file() {
        let desc = CafAudioDescription::pcm_f32_mono(48_000.0);
        let mut writer = CafWriter::new(desc);
        writer.append_audio_data(&[0u8; 32]);
        let bytes = writer.finish();

        // Must start with "caff".
        assert_eq!(&bytes[..4], b"caff");
        // Must be at least: file_hdr(8) + chunk_hdr(12) + desc(32) + data_hdr(12) + edit_cnt(4) + audio(32).
        assert!(bytes.len() >= 8 + 12 + 32 + 12 + 4 + 32);
    }

    #[test]
    fn test_reader_round_trip() {
        let desc = CafAudioDescription::pcm_f32_mono(22_050.0);
        let mut writer = CafWriter::new(desc.clone());
        writer.append_audio_data(&[0xABu8; 16]);
        let bytes = writer.finish();

        let mut reader = CafReader::new(&bytes);
        let fhdr = reader.read_file_header().expect("file header");
        assert_eq!(fhdr.version, 1);

        let chunks = reader.collect_chunks().expect("collect chunks");
        assert!(!chunks.is_empty());

        let parsed_desc = extract_audio_description(&chunks).expect("desc chunk");
        assert!((parsed_desc.sample_rate - 22_050.0).abs() < f64::EPSILON);
        assert_eq!(parsed_desc.format_id, *b"lpcm");
    }

    #[test]
    fn test_writer_with_magic_cookie() {
        let desc = CafAudioDescription::pcm_f32_mono(48_000.0);
        let mut writer = CafWriter::new(desc);
        writer.set_magic_cookie(vec![0xDE, 0xAD, 0xBE, 0xEF]);
        writer.append_audio_data(&[0u8; 8]);
        let bytes = writer.finish();

        let mut reader = CafReader::new(&bytes);
        reader.read_file_header().expect("file header");
        let chunks = reader.collect_chunks().expect("collect chunks");

        let kuki = chunks
            .iter()
            .find(|c| c.header.chunk_type == CafChunkType::MagicCookie);
        assert!(kuki.is_some());
        assert_eq!(kuki.expect("kuki").body, vec![0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn test_chunk_type_display() {
        assert_eq!(CafChunkType::AudioDescription.to_string(), "'desc'");
        assert_eq!(CafChunkType::AudioData.to_string(), "'data'");
    }

    #[test]
    fn test_missing_desc_chunk_error() {
        let chunks: Vec<CafChunk> = vec![];
        assert!(matches!(
            extract_audio_description(&chunks),
            Err(CafError::MissingChunk(_))
        ));
    }

    #[test]
    fn test_buffer_too_short_file_header() {
        assert!(matches!(
            parse_caf_file_header(&[0u8; 4]),
            Err(CafError::BufferTooShort { .. })
        ));
    }

    #[test]
    fn test_data_chunk_edit_count() {
        // The `data` chunk body starts with a 4-byte big-endian edit count of 0.
        let desc = CafAudioDescription::pcm_f32_mono(48_000.0);
        let mut writer = CafWriter::new(desc);
        writer.append_audio_data(&[1u8, 2, 3, 4]);
        let bytes = writer.finish();

        let mut reader = CafReader::new(&bytes);
        reader.read_file_header().expect("file header");
        let chunks = reader.collect_chunks().expect("chunks");

        let data_chunk = chunks
            .iter()
            .find(|c| c.header.chunk_type == CafChunkType::AudioData)
            .expect("data chunk");
        // First 4 bytes of body = edit count (0), next 4 = audio bytes.
        assert_eq!(&data_chunk.body[..4], &[0, 0, 0, 0]);
        assert_eq!(&data_chunk.body[4..], &[1u8, 2, 3, 4]);
    }
}
