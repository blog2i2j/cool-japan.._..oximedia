//! WebP RIFF container parser and writer.
//!
//! WebP uses a RIFF-based container format. The file structure is:
//! - `"RIFF"` (4 bytes) + file_size (4 bytes LE u32) + `"WEBP"` (4 bytes)
//! - Then one or more chunks, each with:
//!   - FourCC (4 bytes ASCII)
//!   - Chunk size (4 bytes LE u32)
//!   - Chunk data (size bytes, padded to even boundary)

use crate::error::{CodecError, CodecResult};

// ── Constants ──────────────────────────────────────────────────────────────────

/// RIFF header magic bytes.
const RIFF_MAGIC: &[u8; 4] = b"RIFF";
/// WebP form type.
const WEBP_MAGIC: &[u8; 4] = b"WEBP";
/// RIFF header size (RIFF tag + file size + WEBP tag).
const RIFF_HEADER_SIZE: usize = 12;
/// Chunk header size (FourCC + chunk size).
const CHUNK_HEADER_SIZE: usize = 8;
/// VP8 sync code bytes.
const VP8_SYNC_CODE: [u8; 3] = [0x9D, 0x01, 0x2A];
/// VP8L signature byte.
const VP8L_SIGNATURE: u8 = 0x2F;
/// VP8X chunk data size (flags byte + 3 reserved + 3-byte width-1 + 3-byte height-1 = 10).
const VP8X_CHUNK_DATA_SIZE: usize = 10;

// ── FourCC constants ───────────────────────────────────────────────────────────

const FOURCC_VP8: [u8; 4] = *b"VP8 ";
const FOURCC_VP8L: [u8; 4] = *b"VP8L";
const FOURCC_VP8X: [u8; 4] = *b"VP8X";
const FOURCC_ALPH: [u8; 4] = *b"ALPH";
const FOURCC_ANIM: [u8; 4] = *b"ANIM";
const FOURCC_ANMF: [u8; 4] = *b"ANMF";
const FOURCC_ICCP: [u8; 4] = *b"ICCP";
const FOURCC_EXIF: [u8; 4] = *b"EXIF";
const FOURCC_XMP: [u8; 4] = *b"XMP ";

// ── VP8X feature flag bits ─────────────────────────────────────────────────────

const VP8X_FLAG_ANIMATION: u8 = 1 << 1;
const VP8X_FLAG_XMP: u8 = 1 << 2;
const VP8X_FLAG_EXIF: u8 = 1 << 3;
const VP8X_FLAG_ALPHA: u8 = 1 << 4;
const VP8X_FLAG_ICC: u8 = 1 << 5;

// ── Chunk Type ─────────────────────────────────────────────────────────────────

/// WebP chunk types as defined by the RIFF-based WebP specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkType {
    /// `"VP8 "` — Lossy bitstream (VP8 format).
    Vp8,
    /// `"VP8L"` — Lossless bitstream (VP8L format).
    Vp8L,
    /// `"VP8X"` — Extended format header (feature flags + canvas size).
    Vp8X,
    /// `"ALPH"` — Alpha channel data.
    Alph,
    /// `"ANIM"` — Animation parameters (background color, loop count).
    Anim,
    /// `"ANMF"` — Animation frame.
    Anmf,
    /// `"ICCP"` — ICC color profile.
    Iccp,
    /// `"EXIF"` — EXIF metadata.
    Exif,
    /// `"XMP "` — XMP metadata.
    Xmp,
    /// Any other FourCC not recognized by this parser.
    Unknown([u8; 4]),
}

impl ChunkType {
    /// Create a `ChunkType` from a 4-byte FourCC.
    fn from_fourcc(fourcc: [u8; 4]) -> Self {
        match fourcc {
            FOURCC_VP8 => ChunkType::Vp8,
            FOURCC_VP8L => ChunkType::Vp8L,
            FOURCC_VP8X => ChunkType::Vp8X,
            FOURCC_ALPH => ChunkType::Alph,
            FOURCC_ANIM => ChunkType::Anim,
            FOURCC_ANMF => ChunkType::Anmf,
            FOURCC_ICCP => ChunkType::Iccp,
            FOURCC_EXIF => ChunkType::Exif,
            FOURCC_XMP => ChunkType::Xmp,
            other => ChunkType::Unknown(other),
        }
    }

    /// Convert this chunk type back to its 4-byte FourCC.
    fn to_fourcc(self) -> [u8; 4] {
        match self {
            ChunkType::Vp8 => FOURCC_VP8,
            ChunkType::Vp8L => FOURCC_VP8L,
            ChunkType::Vp8X => FOURCC_VP8X,
            ChunkType::Alph => FOURCC_ALPH,
            ChunkType::Anim => FOURCC_ANIM,
            ChunkType::Anmf => FOURCC_ANMF,
            ChunkType::Iccp => FOURCC_ICCP,
            ChunkType::Exif => FOURCC_EXIF,
            ChunkType::Xmp => FOURCC_XMP,
            ChunkType::Unknown(cc) => cc,
        }
    }
}

impl std::fmt::Display for ChunkType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let fourcc = self.to_fourcc();
        let s = String::from_utf8_lossy(&fourcc);
        write!(f, "{s}")
    }
}

// ── RiffChunk ──────────────────────────────────────────────────────────────────

/// A parsed RIFF chunk with its type tag and raw payload data.
#[derive(Debug, Clone)]
pub struct RiffChunk {
    /// The chunk type identified by its FourCC tag.
    pub chunk_type: ChunkType,
    /// The raw payload bytes of the chunk (excluding padding).
    pub data: Vec<u8>,
}

// ── WebPEncoding ───────────────────────────────────────────────────────────────

/// WebP container encoding type, determined by the first chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebPEncoding {
    /// Simple lossy (first chunk is `VP8 `).
    Lossy,
    /// Simple lossless (first chunk is `VP8L`).
    Lossless,
    /// Extended format (first chunk is `VP8X`).
    Extended,
}

// ── VP8X Features ──────────────────────────────────────────────────────────────

/// Feature flags and canvas size parsed from a VP8X chunk.
#[derive(Debug, Clone, Copy, Default)]
pub struct Vp8xFeatures {
    /// Whether animation chunks may follow.
    pub has_animation: bool,
    /// Whether the file contains XMP metadata.
    pub has_xmp: bool,
    /// Whether the file contains EXIF metadata.
    pub has_exif: bool,
    /// Whether the file contains alpha channel data.
    pub has_alpha: bool,
    /// Whether the file contains an ICC color profile.
    pub has_icc: bool,
    /// Canvas width in pixels (1-based; stored as width-1 in VP8X).
    pub canvas_width: u32,
    /// Canvas height in pixels (1-based; stored as height-1 in VP8X).
    pub canvas_height: u32,
}

impl Vp8xFeatures {
    /// Parse VP8X features from the raw chunk data (expected 10 bytes).
    fn parse(data: &[u8]) -> CodecResult<Self> {
        if data.len() < VP8X_CHUNK_DATA_SIZE {
            return Err(CodecError::InvalidBitstream(format!(
                "VP8X chunk too small: expected at least {VP8X_CHUNK_DATA_SIZE} bytes, got {}",
                data.len()
            )));
        }

        let flags = data[0];

        // Canvas width-1 is stored as a 24-bit LE value at bytes 4..7
        let canvas_width = u32::from(data[4])
            | (u32::from(data[5]) << 8)
            | (u32::from(data[6]) << 16);
        let canvas_width = canvas_width + 1;

        // Canvas height-1 is stored as a 24-bit LE value at bytes 7..10
        let canvas_height = u32::from(data[7])
            | (u32::from(data[8]) << 8)
            | (u32::from(data[9]) << 16);
        let canvas_height = canvas_height + 1;

        Ok(Self {
            has_animation: (flags & VP8X_FLAG_ANIMATION) != 0,
            has_xmp: (flags & VP8X_FLAG_XMP) != 0,
            has_exif: (flags & VP8X_FLAG_EXIF) != 0,
            has_alpha: (flags & VP8X_FLAG_ALPHA) != 0,
            has_icc: (flags & VP8X_FLAG_ICC) != 0,
            canvas_width,
            canvas_height,
        })
    }

    /// Encode VP8X features into a 10-byte chunk payload.
    fn encode(&self) -> [u8; VP8X_CHUNK_DATA_SIZE] {
        let mut buf = [0u8; VP8X_CHUNK_DATA_SIZE];

        let mut flags: u8 = 0;
        if self.has_animation {
            flags |= VP8X_FLAG_ANIMATION;
        }
        if self.has_xmp {
            flags |= VP8X_FLAG_XMP;
        }
        if self.has_exif {
            flags |= VP8X_FLAG_EXIF;
        }
        if self.has_alpha {
            flags |= VP8X_FLAG_ALPHA;
        }
        if self.has_icc {
            flags |= VP8X_FLAG_ICC;
        }
        buf[0] = flags;
        // bytes 1..4 are reserved (zero)

        let w = self.canvas_width.saturating_sub(1);
        buf[4] = (w & 0xFF) as u8;
        buf[5] = ((w >> 8) & 0xFF) as u8;
        buf[6] = ((w >> 16) & 0xFF) as u8;

        let h = self.canvas_height.saturating_sub(1);
        buf[7] = (h & 0xFF) as u8;
        buf[8] = ((h >> 8) & 0xFF) as u8;
        buf[9] = ((h >> 16) & 0xFF) as u8;

        buf
    }
}

// ── WebPContainer ──────────────────────────────────────────────────────────────

/// A fully parsed WebP RIFF container.
#[derive(Debug, Clone)]
pub struct WebPContainer {
    /// The encoding type (lossy, lossless, or extended).
    pub encoding: WebPEncoding,
    /// VP8X feature flags, present only for extended format.
    pub features: Option<Vp8xFeatures>,
    /// All chunks in the container, in order.
    pub chunks: Vec<RiffChunk>,
}

impl WebPContainer {
    /// Parse a WebP file from a byte slice.
    ///
    /// Validates the RIFF header and iterates through all chunks.
    pub fn parse(data: &[u8]) -> CodecResult<Self> {
        if data.len() < RIFF_HEADER_SIZE {
            return Err(CodecError::InvalidBitstream(
                "Data too small for RIFF header".into(),
            ));
        }

        // Validate RIFF magic
        if &data[0..4] != RIFF_MAGIC {
            return Err(CodecError::InvalidBitstream(
                "Missing RIFF magic bytes".into(),
            ));
        }

        // Read declared file size (bytes after the initial 8 bytes)
        let file_size = read_u32_le(&data[4..8]);
        let declared_total = file_size as usize + 8; // +8 for RIFF tag + size field

        // Validate WEBP form type
        if &data[8..12] != WEBP_MAGIC {
            return Err(CodecError::InvalidBitstream(
                "Missing WEBP form type".into(),
            ));
        }

        // The actual data we can parse (don't read past the buffer)
        let payload_end = declared_total.min(data.len());

        // Parse chunks
        let mut offset = RIFF_HEADER_SIZE;
        let mut chunks = Vec::new();

        while offset + CHUNK_HEADER_SIZE <= payload_end {
            let mut fourcc = [0u8; 4];
            fourcc.copy_from_slice(&data[offset..offset + 4]);
            let chunk_size = read_u32_le(&data[offset + 4..offset + 8]) as usize;
            offset += CHUNK_HEADER_SIZE;

            // Guard against truncated data
            if offset + chunk_size > payload_end {
                return Err(CodecError::InvalidBitstream(format!(
                    "Chunk '{}' at offset {} declares size {} but only {} bytes remain",
                    String::from_utf8_lossy(&fourcc),
                    offset - CHUNK_HEADER_SIZE,
                    chunk_size,
                    payload_end.saturating_sub(offset),
                )));
            }

            let chunk_data = data[offset..offset + chunk_size].to_vec();
            chunks.push(RiffChunk {
                chunk_type: ChunkType::from_fourcc(fourcc),
                data: chunk_data,
            });

            // Advance past chunk data, with even-byte padding
            offset += chunk_size;
            if chunk_size % 2 != 0 {
                offset += 1;
            }
        }

        if chunks.is_empty() {
            return Err(CodecError::InvalidBitstream(
                "No chunks found in WebP container".into(),
            ));
        }

        // Determine encoding type from the first chunk
        let encoding = match chunks[0].chunk_type {
            ChunkType::Vp8 => WebPEncoding::Lossy,
            ChunkType::Vp8L => WebPEncoding::Lossless,
            ChunkType::Vp8X => WebPEncoding::Extended,
            other => {
                return Err(CodecError::InvalidBitstream(format!(
                    "Unexpected first chunk type: {other}"
                )));
            }
        };

        // Parse VP8X features if present
        let features = if encoding == WebPEncoding::Extended {
            Some(Vp8xFeatures::parse(&chunks[0].data)?)
        } else {
            None
        };

        Ok(Self {
            encoding,
            features,
            chunks,
        })
    }

    /// Find the VP8 or VP8L bitstream chunk.
    ///
    /// For simple files, this is the first (and only) chunk.
    /// For extended files, this searches for the first VP8/VP8L chunk.
    pub fn bitstream_chunk(&self) -> Option<&RiffChunk> {
        self.chunks
            .iter()
            .find(|c| c.chunk_type == ChunkType::Vp8 || c.chunk_type == ChunkType::Vp8L)
    }

    /// Find the alpha chunk (`ALPH`), if present.
    pub fn alpha_chunk(&self) -> Option<&RiffChunk> {
        self.chunks
            .iter()
            .find(|c| c.chunk_type == ChunkType::Alph)
    }

    /// Find the ICC profile chunk, if present.
    pub fn icc_chunk(&self) -> Option<&RiffChunk> {
        self.chunks
            .iter()
            .find(|c| c.chunk_type == ChunkType::Iccp)
    }

    /// Find the EXIF metadata chunk, if present.
    pub fn exif_chunk(&self) -> Option<&RiffChunk> {
        self.chunks
            .iter()
            .find(|c| c.chunk_type == ChunkType::Exif)
    }

    /// Find the XMP metadata chunk, if present.
    pub fn xmp_chunk(&self) -> Option<&RiffChunk> {
        self.chunks
            .iter()
            .find(|c| c.chunk_type == ChunkType::Xmp)
    }

    /// Find the animation parameters chunk, if present.
    pub fn anim_chunk(&self) -> Option<&RiffChunk> {
        self.chunks
            .iter()
            .find(|c| c.chunk_type == ChunkType::Anim)
    }

    /// Collect all animation frame chunks.
    pub fn animation_frames(&self) -> Vec<&RiffChunk> {
        self.chunks
            .iter()
            .filter(|c| c.chunk_type == ChunkType::Anmf)
            .collect()
    }

    /// Get canvas dimensions.
    ///
    /// For VP8X extended format, uses the canvas size from the header.
    /// For simple lossy, parses the VP8 frame header.
    /// For simple lossless, parses the VP8L signature header.
    pub fn dimensions(&self) -> CodecResult<(u32, u32)> {
        // VP8X canvas size takes priority
        if let Some(ref features) = self.features {
            return Ok((features.canvas_width, features.canvas_height));
        }

        // Otherwise parse from the bitstream chunk
        let bs = self.bitstream_chunk().ok_or_else(|| {
            CodecError::InvalidBitstream("No bitstream chunk found".into())
        })?;

        match bs.chunk_type {
            ChunkType::Vp8 => parse_vp8_dimensions(&bs.data),
            ChunkType::Vp8L => parse_vp8l_dimensions(&bs.data),
            _ => Err(CodecError::InvalidBitstream(
                "Bitstream chunk is neither VP8 nor VP8L".into(),
            )),
        }
    }
}

// ── VP8 / VP8L dimension parsing ───────────────────────────────────────────────

/// Parse width and height from a VP8 lossy bitstream header.
///
/// VP8 frame header layout:
/// - Bytes 0-2: frame tag (3 bytes)
/// - Bytes 3-5: sync code 0x9D 0x01 0x2A
/// - Bytes 6-7: width (LE u16, lower 14 bits = width, upper 2 = horizontal scale)
/// - Bytes 8-9: height (LE u16, lower 14 bits = height, upper 2 = vertical scale)
fn parse_vp8_dimensions(data: &[u8]) -> CodecResult<(u32, u32)> {
    if data.len() < 10 {
        return Err(CodecError::InvalidBitstream(
            "VP8 bitstream too small for frame header".into(),
        ));
    }

    // Validate sync code at bytes 3..6
    if data[3] != VP8_SYNC_CODE[0]
        || data[4] != VP8_SYNC_CODE[1]
        || data[5] != VP8_SYNC_CODE[2]
    {
        return Err(CodecError::InvalidBitstream(
            "VP8 sync code not found (expected 0x9D 0x01 0x2A)".into(),
        ));
    }

    let raw_width = u16::from_le_bytes([data[6], data[7]]);
    let raw_height = u16::from_le_bytes([data[8], data[9]]);

    // Lower 14 bits are the actual dimension
    let width = u32::from(raw_width & 0x3FFF);
    let height = u32::from(raw_height & 0x3FFF);

    if width == 0 || height == 0 {
        return Err(CodecError::InvalidBitstream(
            "VP8 dimensions cannot be zero".into(),
        ));
    }

    Ok((width, height))
}

/// Parse width and height from a VP8L lossless bitstream header.
///
/// VP8L header layout:
/// - Byte 0: signature 0x2F
/// - Bytes 1-4: 32 bits containing:
///   - bits 0..13  (14 bits): width - 1
///   - bits 14..27 (14 bits): height - 1
///   - bit 28:     alpha_is_used
///   - bits 29..31 (3 bits): version (must be 0)
fn parse_vp8l_dimensions(data: &[u8]) -> CodecResult<(u32, u32)> {
    if data.len() < 5 {
        return Err(CodecError::InvalidBitstream(
            "VP8L bitstream too small for header".into(),
        ));
    }

    if data[0] != VP8L_SIGNATURE {
        return Err(CodecError::InvalidBitstream(format!(
            "VP8L signature mismatch: expected 0x{VP8L_SIGNATURE:02X}, got 0x{:02X}",
            data[0]
        )));
    }

    let bits = u32::from_le_bytes([data[1], data[2], data[3], data[4]]);

    let width = (bits & 0x3FFF) + 1;          // 14 bits
    let height = ((bits >> 14) & 0x3FFF) + 1;  // 14 bits

    if width == 0 || height == 0 {
        return Err(CodecError::InvalidBitstream(
            "VP8L dimensions cannot be zero".into(),
        ));
    }

    Ok((width, height))
}

// ── WebPWriter ─────────────────────────────────────────────────────────────────

/// Writer for constructing WebP RIFF containers from encoded bitstream data.
pub struct WebPWriter;

impl WebPWriter {
    /// Write a simple lossy WebP file (RIFF header + VP8 chunk).
    pub fn write_lossy(vp8_data: &[u8]) -> Vec<u8> {
        Self::write_single_chunk(&FOURCC_VP8, vp8_data)
    }

    /// Write a simple lossless WebP file (RIFF header + VP8L chunk).
    pub fn write_lossless(vp8l_data: &[u8]) -> Vec<u8> {
        Self::write_single_chunk(&FOURCC_VP8L, vp8l_data)
    }

    /// Write an extended WebP file with optional alpha.
    ///
    /// Produces: RIFF header, VP8X chunk, optional ALPH chunk, VP8 chunk.
    pub fn write_extended(
        vp8_data: &[u8],
        alpha_data: Option<&[u8]>,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let features = Vp8xFeatures {
            has_alpha: alpha_data.is_some(),
            canvas_width: width,
            canvas_height: height,
            ..Vp8xFeatures::default()
        };

        let vp8x_payload = features.encode();

        // Calculate total file size
        let mut body_size: usize = 4; // "WEBP" form type
        body_size += chunk_wire_size(&vp8x_payload);
        if let Some(alpha) = alpha_data {
            body_size += chunk_wire_size(alpha);
        }
        body_size += chunk_wire_size(vp8_data);

        let mut buf = Vec::with_capacity(8 + body_size);

        // RIFF header
        buf.extend_from_slice(RIFF_MAGIC);
        buf.extend_from_slice(&(body_size as u32).to_le_bytes());
        buf.extend_from_slice(WEBP_MAGIC);

        // VP8X chunk
        write_chunk(&mut buf, &FOURCC_VP8X, &vp8x_payload);

        // Optional ALPH chunk
        if let Some(alpha) = alpha_data {
            write_chunk(&mut buf, &FOURCC_ALPH, alpha);
        }

        // VP8 bitstream chunk
        write_chunk(&mut buf, &FOURCC_VP8, vp8_data);

        buf
    }

    /// Write an extended WebP file from a list of chunks.
    ///
    /// The caller is responsible for providing a valid VP8X chunk as the first entry.
    pub fn write_chunks(chunks: &[RiffChunk]) -> Vec<u8> {
        let mut body_size: usize = 4; // "WEBP"
        for chunk in chunks {
            body_size += chunk_wire_size(&chunk.data);
        }

        let mut buf = Vec::with_capacity(8 + body_size);
        buf.extend_from_slice(RIFF_MAGIC);
        buf.extend_from_slice(&(body_size as u32).to_le_bytes());
        buf.extend_from_slice(WEBP_MAGIC);

        for chunk in chunks {
            let fourcc = chunk.chunk_type.to_fourcc();
            write_chunk(&mut buf, &fourcc, &chunk.data);
        }

        buf
    }

    /// Internal: write a single-chunk WebP file.
    fn write_single_chunk(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let body_size = 4 + chunk_wire_size(payload); // "WEBP" + chunk
        let mut buf = Vec::with_capacity(8 + body_size);

        buf.extend_from_slice(RIFF_MAGIC);
        buf.extend_from_slice(&(body_size as u32).to_le_bytes());
        buf.extend_from_slice(WEBP_MAGIC);

        write_chunk(&mut buf, fourcc, payload);
        buf
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────────

/// Read a little-endian u32 from a 4-byte slice.
fn read_u32_le(data: &[u8]) -> u32 {
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&data[..4]);
    u32::from_le_bytes(buf)
}

/// Calculate the wire size of a single chunk (header + payload + padding).
fn chunk_wire_size(payload: &[u8]) -> usize {
    let padded = if payload.len() % 2 != 0 {
        payload.len() + 1
    } else {
        payload.len()
    };
    CHUNK_HEADER_SIZE + padded
}

/// Write a chunk (FourCC + LE size + data + optional pad byte) to `buf`.
fn write_chunk(buf: &mut Vec<u8>, fourcc: &[u8; 4], data: &[u8]) {
    buf.extend_from_slice(fourcc);
    buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
    buf.extend_from_slice(data);
    if data.len() % 2 != 0 {
        buf.push(0); // pad to even boundary
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ────────────────────────────────────────────────────────

    /// Build a minimal valid VP8 bitstream header with the given dimensions.
    fn make_vp8_header(width: u16, height: u16) -> Vec<u8> {
        let mut data = vec![0u8; 10];
        // Frame tag (3 bytes, keyframe)
        data[0] = 0x00;
        data[1] = 0x00;
        data[2] = 0x00;
        // Sync code
        data[3] = 0x9D;
        data[4] = 0x01;
        data[5] = 0x2A;
        // Width (LE, lower 14 bits)
        let w_bytes = width.to_le_bytes();
        data[6] = w_bytes[0];
        data[7] = w_bytes[1];
        // Height (LE, lower 14 bits)
        let h_bytes = height.to_le_bytes();
        data[8] = h_bytes[0];
        data[9] = h_bytes[1];
        data
    }

    /// Build a minimal valid VP8L bitstream header with the given dimensions.
    fn make_vp8l_header(width: u32, height: u32) -> Vec<u8> {
        let mut data = vec![0u8; 5];
        data[0] = VP8L_SIGNATURE;
        let w_minus_1 = (width - 1) & 0x3FFF;
        let h_minus_1 = (height - 1) & 0x3FFF;
        let bits: u32 = w_minus_1 | (h_minus_1 << 14);
        let b = bits.to_le_bytes();
        data[1] = b[0];
        data[2] = b[1];
        data[3] = b[2];
        data[4] = b[3];
        data
    }

    /// Build a simple lossy WebP file from raw VP8 data.
    fn make_simple_lossy(width: u16, height: u16) -> Vec<u8> {
        let vp8 = make_vp8_header(width, height);
        WebPWriter::write_lossy(&vp8)
    }

    /// Build a simple lossless WebP file from raw VP8L data.
    fn make_simple_lossless(width: u32, height: u32) -> Vec<u8> {
        let vp8l = make_vp8l_header(width, height);
        WebPWriter::write_lossless(&vp8l)
    }

    // ── ChunkType ──────────────────────────────────────────────────────

    #[test]
    fn test_chunk_type_roundtrip() {
        let types = [
            ChunkType::Vp8,
            ChunkType::Vp8L,
            ChunkType::Vp8X,
            ChunkType::Alph,
            ChunkType::Anim,
            ChunkType::Anmf,
            ChunkType::Iccp,
            ChunkType::Exif,
            ChunkType::Xmp,
            ChunkType::Unknown(*b"TEST"),
        ];

        for ct in &types {
            let fourcc = ct.to_fourcc();
            let recovered = ChunkType::from_fourcc(fourcc);
            assert_eq!(*ct, recovered);
        }
    }

    #[test]
    fn test_chunk_type_display() {
        assert_eq!(ChunkType::Vp8.to_string(), "VP8 ");
        assert_eq!(ChunkType::Vp8L.to_string(), "VP8L");
        assert_eq!(ChunkType::Xmp.to_string(), "XMP ");
        assert_eq!(ChunkType::Unknown(*b"TSET").to_string(), "TSET");
    }

    // ── VP8X Features ──────────────────────────────────────────────────

    #[test]
    fn test_vp8x_features_parse_all_flags() {
        let mut data = [0u8; 10];
        data[0] = VP8X_FLAG_ANIMATION | VP8X_FLAG_XMP | VP8X_FLAG_EXIF
            | VP8X_FLAG_ALPHA | VP8X_FLAG_ICC;
        // canvas width-1 = 1919 (0x077F) => width 1920
        data[4] = 0x7F;
        data[5] = 0x07;
        data[6] = 0x00;
        // canvas height-1 = 1079 (0x0437) => height 1080
        data[7] = 0x37;
        data[8] = 0x04;
        data[9] = 0x00;

        let feat = Vp8xFeatures::parse(&data).expect("should parse");
        assert!(feat.has_animation);
        assert!(feat.has_xmp);
        assert!(feat.has_exif);
        assert!(feat.has_alpha);
        assert!(feat.has_icc);
        assert_eq!(feat.canvas_width, 1920);
        assert_eq!(feat.canvas_height, 1080);
    }

    #[test]
    fn test_vp8x_features_parse_no_flags() {
        let data = [0u8; 10];
        let feat = Vp8xFeatures::parse(&data).expect("should parse");
        assert!(!feat.has_animation);
        assert!(!feat.has_xmp);
        assert!(!feat.has_exif);
        assert!(!feat.has_alpha);
        assert!(!feat.has_icc);
        assert_eq!(feat.canvas_width, 1);
        assert_eq!(feat.canvas_height, 1);
    }

    #[test]
    fn test_vp8x_features_roundtrip() {
        let original = Vp8xFeatures {
            has_animation: true,
            has_xmp: false,
            has_exif: true,
            has_alpha: true,
            has_icc: false,
            canvas_width: 3840,
            canvas_height: 2160,
        };

        let encoded = original.encode();
        let decoded = Vp8xFeatures::parse(&encoded).expect("should parse");

        assert_eq!(original.has_animation, decoded.has_animation);
        assert_eq!(original.has_xmp, decoded.has_xmp);
        assert_eq!(original.has_exif, decoded.has_exif);
        assert_eq!(original.has_alpha, decoded.has_alpha);
        assert_eq!(original.has_icc, decoded.has_icc);
        assert_eq!(original.canvas_width, decoded.canvas_width);
        assert_eq!(original.canvas_height, decoded.canvas_height);
    }

    #[test]
    fn test_vp8x_features_parse_too_small() {
        let data = [0u8; 5];
        let result = Vp8xFeatures::parse(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_vp8x_max_canvas_size() {
        // 24-bit max = 16_777_215, stored as width-1 => width = 16_777_216
        let feat = Vp8xFeatures {
            canvas_width: 16_777_216,
            canvas_height: 16_777_216,
            ..Vp8xFeatures::default()
        };
        let encoded = feat.encode();
        let decoded = Vp8xFeatures::parse(&encoded).expect("should parse");
        assert_eq!(decoded.canvas_width, 16_777_216);
        assert_eq!(decoded.canvas_height, 16_777_216);
    }

    // ── VP8 Dimensions ─────────────────────────────────────────────────

    #[test]
    fn test_vp8_dimensions_basic() {
        let data = make_vp8_header(640, 480);
        let (w, h) = parse_vp8_dimensions(&data).expect("should parse");
        assert_eq!(w, 640);
        assert_eq!(h, 480);
    }

    #[test]
    fn test_vp8_dimensions_with_scale_bits() {
        // Set scale bits in upper 2 bits of the 16-bit values
        let mut data = make_vp8_header(320, 240);
        // Add horizontal scale = 1 (bits 14-15)
        data[7] |= 0x40; // set bit 14
        // Add vertical scale = 2 (bits 14-15)
        data[9] |= 0x80; // set bit 15

        let (w, h) = parse_vp8_dimensions(&data).expect("should parse");
        // Width/height should ignore scale bits (only lower 14)
        assert_eq!(w, 320);
        assert_eq!(h, 240);
    }

    #[test]
    fn test_vp8_dimensions_too_small() {
        let data = [0u8; 5];
        assert!(parse_vp8_dimensions(&data).is_err());
    }

    #[test]
    fn test_vp8_dimensions_bad_sync() {
        let mut data = make_vp8_header(100, 100);
        data[3] = 0x00; // corrupt sync code
        assert!(parse_vp8_dimensions(&data).is_err());
    }

    #[test]
    fn test_vp8_dimensions_zero_width() {
        let mut data = make_vp8_header(0, 100);
        // Width 0 in lower 14 bits
        data[6] = 0;
        data[7] = 0;
        assert!(parse_vp8_dimensions(&data).is_err());
    }

    // ── VP8L Dimensions ────────────────────────────────────────────────

    #[test]
    fn test_vp8l_dimensions_basic() {
        let data = make_vp8l_header(800, 600);
        let (w, h) = parse_vp8l_dimensions(&data).expect("should parse");
        assert_eq!(w, 800);
        assert_eq!(h, 600);
    }

    #[test]
    fn test_vp8l_dimensions_one_pixel() {
        let data = make_vp8l_header(1, 1);
        let (w, h) = parse_vp8l_dimensions(&data).expect("should parse");
        assert_eq!(w, 1);
        assert_eq!(h, 1);
    }

    #[test]
    fn test_vp8l_dimensions_max_14bit() {
        // Max 14-bit value: 16383 (0x3FFF) + 1 = 16384
        let data = make_vp8l_header(16384, 16384);
        let (w, h) = parse_vp8l_dimensions(&data).expect("should parse");
        assert_eq!(w, 16384);
        assert_eq!(h, 16384);
    }

    #[test]
    fn test_vp8l_dimensions_too_small() {
        let data = [VP8L_SIGNATURE, 0, 0];
        assert!(parse_vp8l_dimensions(&data).is_err());
    }

    #[test]
    fn test_vp8l_dimensions_bad_signature() {
        let mut data = make_vp8l_header(100, 100);
        data[0] = 0xFF;
        assert!(parse_vp8l_dimensions(&data).is_err());
    }

    // ── WebPContainer::parse ───────────────────────────────────────────

    #[test]
    fn test_parse_simple_lossy() {
        let webp = make_simple_lossy(320, 240);
        let container = WebPContainer::parse(&webp).expect("should parse");

        assert_eq!(container.encoding, WebPEncoding::Lossy);
        assert!(container.features.is_none());
        assert_eq!(container.chunks.len(), 1);
        assert_eq!(container.chunks[0].chunk_type, ChunkType::Vp8);

        let (w, h) = container.dimensions().expect("should get dimensions");
        assert_eq!(w, 320);
        assert_eq!(h, 240);
    }

    #[test]
    fn test_parse_simple_lossless() {
        let webp = make_simple_lossless(1024, 768);
        let container = WebPContainer::parse(&webp).expect("should parse");

        assert_eq!(container.encoding, WebPEncoding::Lossless);
        assert!(container.features.is_none());
        assert_eq!(container.chunks.len(), 1);
        assert_eq!(container.chunks[0].chunk_type, ChunkType::Vp8L);

        let (w, h) = container.dimensions().expect("should get dimensions");
        assert_eq!(w, 1024);
        assert_eq!(h, 768);
    }

    #[test]
    fn test_parse_extended_with_alpha() {
        let vp8 = make_vp8_header(640, 480);
        let alpha = vec![0xAA; 100];
        let webp = WebPWriter::write_extended(&vp8, Some(&alpha), 640, 480);
        let container = WebPContainer::parse(&webp).expect("should parse");

        assert_eq!(container.encoding, WebPEncoding::Extended);
        let features = container.features.expect("should have features");
        assert!(features.has_alpha);
        assert!(!features.has_animation);
        assert_eq!(features.canvas_width, 640);
        assert_eq!(features.canvas_height, 480);

        assert_eq!(container.chunks.len(), 3); // VP8X, ALPH, VP8
        assert!(container.alpha_chunk().is_some());
        assert_eq!(container.alpha_chunk().map(|c| c.data.len()), Some(100));

        let bs = container.bitstream_chunk().expect("should have bitstream");
        assert_eq!(bs.chunk_type, ChunkType::Vp8);
    }

    #[test]
    fn test_parse_extended_no_alpha() {
        let vp8 = make_vp8_header(1920, 1080);
        let webp = WebPWriter::write_extended(&vp8, None, 1920, 1080);
        let container = WebPContainer::parse(&webp).expect("should parse");

        assert_eq!(container.encoding, WebPEncoding::Extended);
        let features = container.features.expect("should have features");
        assert!(!features.has_alpha);
        assert_eq!(features.canvas_width, 1920);
        assert_eq!(features.canvas_height, 1080);

        assert_eq!(container.chunks.len(), 2); // VP8X, VP8
        assert!(container.alpha_chunk().is_none());
    }

    #[test]
    fn test_parse_too_small() {
        let data = [0u8; 8];
        assert!(WebPContainer::parse(&data).is_err());
    }

    #[test]
    fn test_parse_bad_riff_magic() {
        let mut webp = make_simple_lossy(10, 10);
        webp[0] = b'X';
        assert!(WebPContainer::parse(&webp).is_err());
    }

    #[test]
    fn test_parse_bad_webp_magic() {
        let mut webp = make_simple_lossy(10, 10);
        webp[8] = b'X';
        assert!(WebPContainer::parse(&webp).is_err());
    }

    #[test]
    fn test_parse_empty_payload() {
        // Valid RIFF/WEBP header but no chunks
        let mut data = Vec::new();
        data.extend_from_slice(RIFF_MAGIC);
        data.extend_from_slice(&4u32.to_le_bytes()); // file size = 4 (just "WEBP")
        data.extend_from_slice(WEBP_MAGIC);
        assert!(WebPContainer::parse(&data).is_err());
    }

    // ── WebPWriter ─────────────────────────────────────────────────────

    #[test]
    fn test_write_lossy_roundtrip() {
        let vp8 = make_vp8_header(256, 256);
        let webp = WebPWriter::write_lossy(&vp8);
        let container = WebPContainer::parse(&webp).expect("should parse");

        assert_eq!(container.encoding, WebPEncoding::Lossy);
        let bs = container.bitstream_chunk().expect("bitstream");
        assert_eq!(bs.data, vp8);
    }

    #[test]
    fn test_write_lossless_roundtrip() {
        let vp8l = make_vp8l_header(512, 512);
        let webp = WebPWriter::write_lossless(&vp8l);
        let container = WebPContainer::parse(&webp).expect("should parse");

        assert_eq!(container.encoding, WebPEncoding::Lossless);
        let bs = container.bitstream_chunk().expect("bitstream");
        assert_eq!(bs.data, vp8l);
    }

    #[test]
    fn test_write_extended_roundtrip() {
        let vp8 = make_vp8_header(1280, 720);
        let alpha = vec![0xFF; 50];
        let webp = WebPWriter::write_extended(&vp8, Some(&alpha), 1280, 720);
        let container = WebPContainer::parse(&webp).expect("should parse");

        assert_eq!(container.encoding, WebPEncoding::Extended);
        let feat = container.features.expect("features");
        assert!(feat.has_alpha);
        assert_eq!(feat.canvas_width, 1280);
        assert_eq!(feat.canvas_height, 720);

        let bs = container.bitstream_chunk().expect("bitstream");
        assert_eq!(bs.data, vp8);

        let alph = container.alpha_chunk().expect("alpha");
        assert_eq!(alph.data, alpha);
    }

    #[test]
    fn test_write_odd_sized_payload_padding() {
        // Odd-length payload should be padded to even boundary
        let vp8 = vec![0x9D, 0x01, 0x2A, 0x9D, 0x01, 0x2A, 0x01, 0x00, 0x01, 0x00, 0xAB];
        // 11 bytes = odd, should be padded
        let webp = WebPWriter::write_lossy(&vp8);

        // Total: 12 (header) + 8 (chunk header) + 11 (data) + 1 (pad) = 32
        assert_eq!(webp.len(), 32);

        // Verify we can parse it back
        // (The VP8 header in this data has correct sync code at the right offset)
        let container = WebPContainer::parse(&webp).expect("should parse padded");
        let bs = container.bitstream_chunk().expect("bitstream");
        assert_eq!(bs.data, vp8);
    }

    #[test]
    fn test_write_chunks_custom() {
        let chunks = vec![
            RiffChunk {
                chunk_type: ChunkType::Vp8X,
                data: Vp8xFeatures {
                    has_icc: true,
                    canvas_width: 100,
                    canvas_height: 100,
                    ..Vp8xFeatures::default()
                }
                .encode()
                .to_vec(),
            },
            RiffChunk {
                chunk_type: ChunkType::Iccp,
                data: vec![0x01, 0x02, 0x03],
            },
            RiffChunk {
                chunk_type: ChunkType::Vp8,
                data: make_vp8_header(100, 100),
            },
        ];

        let webp = WebPWriter::write_chunks(&chunks);
        let container = WebPContainer::parse(&webp).expect("should parse");

        assert_eq!(container.encoding, WebPEncoding::Extended);
        assert_eq!(container.chunks.len(), 3);
        let feat = container.features.expect("features");
        assert!(feat.has_icc);
        assert_eq!(feat.canvas_width, 100);
        assert_eq!(feat.canvas_height, 100);

        let icc = container.icc_chunk().expect("icc");
        assert_eq!(icc.data, vec![0x01, 0x02, 0x03]);
    }

    // ── Accessor methods ───────────────────────────────────────────────

    #[test]
    fn test_accessor_methods_none() {
        let webp = make_simple_lossy(10, 10);
        let container = WebPContainer::parse(&webp).expect("should parse");
        assert!(container.alpha_chunk().is_none());
        assert!(container.icc_chunk().is_none());
        assert!(container.exif_chunk().is_none());
        assert!(container.xmp_chunk().is_none());
        assert!(container.anim_chunk().is_none());
        assert!(container.animation_frames().is_empty());
    }

    #[test]
    fn test_dimensions_from_vp8x() {
        let vp8 = make_vp8_header(100, 100);
        // VP8X says 640x480, VP8 header says 100x100 — VP8X takes priority
        let webp = WebPWriter::write_extended(&vp8, None, 640, 480);
        let container = WebPContainer::parse(&webp).expect("should parse");
        let (w, h) = container.dimensions().expect("dimensions");
        assert_eq!(w, 640);
        assert_eq!(h, 480);
    }

    #[test]
    fn test_dimensions_from_vp8_bitstream() {
        let webp = make_simple_lossy(1920, 1080);
        let container = WebPContainer::parse(&webp).expect("should parse");
        let (w, h) = container.dimensions().expect("dimensions");
        assert_eq!(w, 1920);
        assert_eq!(h, 1080);
    }

    #[test]
    fn test_dimensions_from_vp8l_bitstream() {
        let webp = make_simple_lossless(4096, 2048);
        let container = WebPContainer::parse(&webp).expect("should parse");
        let (w, h) = container.dimensions().expect("dimensions");
        assert_eq!(w, 4096);
        assert_eq!(h, 2048);
    }

    // ── Edge cases ─────────────────────────────────────────────────────

    #[test]
    fn test_unknown_chunk_type_preserved() {
        let chunks = vec![
            RiffChunk {
                chunk_type: ChunkType::Vp8,
                data: make_vp8_header(10, 10),
            },
        ];
        let mut webp = WebPWriter::write_chunks(&chunks);

        // Manually append an unknown chunk "ZZZZ" with 4 bytes of data
        // But we need to fix the RIFF file size first
        let old_file_size = read_u32_le(&webp[4..8]);
        let extra_chunk_size: u32 = 8 + 4; // header + data
        let new_file_size = old_file_size + extra_chunk_size;
        webp[4..8].copy_from_slice(&new_file_size.to_le_bytes());
        webp.extend_from_slice(b"ZZZZ");
        webp.extend_from_slice(&4u32.to_le_bytes());
        webp.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);

        let container = WebPContainer::parse(&webp).expect("should parse");
        assert_eq!(container.chunks.len(), 2);
        assert_eq!(
            container.chunks[1].chunk_type,
            ChunkType::Unknown(*b"ZZZZ")
        );
        assert_eq!(container.chunks[1].data, vec![0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn test_truncated_chunk_error() {
        let mut webp = make_simple_lossy(10, 10);
        // Corrupt: set chunk size to be larger than actual data
        let chunk_size_offset = RIFF_HEADER_SIZE + 4;
        webp[chunk_size_offset..chunk_size_offset + 4]
            .copy_from_slice(&9999u32.to_le_bytes());
        assert!(WebPContainer::parse(&webp).is_err());
    }

    #[test]
    fn test_multiple_chunks_with_metadata() {
        let vp8x_data = Vp8xFeatures {
            has_exif: true,
            has_xmp: true,
            canvas_width: 200,
            canvas_height: 150,
            ..Vp8xFeatures::default()
        }
        .encode();

        let chunks = vec![
            RiffChunk {
                chunk_type: ChunkType::Vp8X,
                data: vp8x_data.to_vec(),
            },
            RiffChunk {
                chunk_type: ChunkType::Exif,
                data: vec![0x45, 0x78, 0x69, 0x66], // "Exif"
            },
            RiffChunk {
                chunk_type: ChunkType::Xmp,
                data: b"<x:xmpmeta>test</x:xmpmeta>".to_vec(),
            },
            RiffChunk {
                chunk_type: ChunkType::Vp8,
                data: make_vp8_header(200, 150),
            },
        ];

        let webp = WebPWriter::write_chunks(&chunks);
        let container = WebPContainer::parse(&webp).expect("should parse");

        assert_eq!(container.encoding, WebPEncoding::Extended);
        assert_eq!(container.chunks.len(), 4);

        let feat = container.features.expect("features");
        assert!(feat.has_exif);
        assert!(feat.has_xmp);

        let exif = container.exif_chunk().expect("exif");
        assert_eq!(exif.data, vec![0x45, 0x78, 0x69, 0x66]);

        let xmp = container.xmp_chunk().expect("xmp");
        assert_eq!(xmp.data, b"<x:xmpmeta>test</x:xmpmeta>");

        let (w, h) = container.dimensions().expect("dimensions");
        assert_eq!(w, 200);
        assert_eq!(h, 150);
    }

    #[test]
    fn test_even_payload_no_padding() {
        // Even-length payload should NOT have padding byte
        let vp8 = make_vp8_header(10, 10); // 10 bytes = even
        let webp = WebPWriter::write_lossy(&vp8);
        // Total: 12 (header) + 8 (chunk header) + 10 (data) = 30
        assert_eq!(webp.len(), 30);
    }

    #[test]
    fn test_file_size_field_accuracy() {
        let vp8 = make_vp8_header(10, 10);
        let webp = WebPWriter::write_lossy(&vp8);
        let declared = read_u32_le(&webp[4..8]) as usize;
        // declared = total - 8 (RIFF + size field)
        assert_eq!(declared + 8, webp.len());
    }

    #[test]
    fn test_extended_file_size_field_accuracy() {
        let vp8 = make_vp8_header(100, 100);
        let alpha = vec![0x42; 7]; // odd length
        let webp = WebPWriter::write_extended(&vp8, Some(&alpha), 100, 100);
        let declared = read_u32_le(&webp[4..8]) as usize;
        assert_eq!(declared + 8, webp.len());
    }
}
