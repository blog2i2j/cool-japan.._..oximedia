//! ID3v2.4 UTF-8 text encoding preference module.
//!
//! ID3v2.4 introduced UTF-8 (encoding byte `0x03`) as a valid text encoding
//! for text frames. This module provides types and utilities for reading and
//! writing UTF-8–encoded ID3v2 text frames, with fallback to UTF-16 for
//! ID3v2.3 compatibility.
//!
//! # Text Encoding Bytes (ID3v2 spec)
//!
//! | Byte | Encoding      | ID3v2.3 | ID3v2.4 |
//! |------|---------------|---------|---------|
//! | 0x00 | ISO-8859-1    | Yes     | Yes     |
//! | 0x01 | UTF-16 w/ BOM | Yes     | Yes     |
//! | 0x02 | UTF-16BE      | No      | Yes     |
//! | 0x03 | UTF-8         | No      | Yes     |
//!
//! # Example
//!
//! ```
//! use oximedia_metadata::id3v2_utf8::{
//!     TextEncodingPreference, Utf8TextFrameWriter, Utf8TextFrameReader,
//! };
//!
//! // Write a UTF-8 text frame for ID3v2.4
//! let writer = Utf8TextFrameWriter::new(TextEncodingPreference::Utf8First);
//! let frame = writer.write_text_frame("TIT2", "Hello World", 4)
//!     .expect("should write");
//! assert_eq!(frame[0], 0x03); // UTF-8 encoding byte
//!
//! // Read it back
//! let reader = Utf8TextFrameReader::new();
//! let text = reader.read_text_frame(&frame).expect("should read");
//! assert_eq!(text, "Hello World");
//! ```

use crate::Error;

// ---- Encoding byte constants ----

/// ISO-8859-1 / Latin-1 encoding byte.
const ENCODING_LATIN1: u8 = 0x00;
/// UTF-16 with BOM encoding byte.
const ENCODING_UTF16_BOM: u8 = 0x01;
/// UTF-16BE without BOM encoding byte (v2.4 only).
const ENCODING_UTF16BE: u8 = 0x02;
/// UTF-8 encoding byte (v2.4 only).
const ENCODING_UTF8: u8 = 0x03;

/// UTF-16 LE BOM bytes.
const UTF16_LE_BOM: [u8; 2] = [0xFF, 0xFE];

// ---- Text Encoding Preference ----

/// Strategy for choosing text encoding when writing ID3v2 text frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextEncodingPreference {
    /// Prefer UTF-8 (encoding byte 0x03). If the target version is < 2.4,
    /// fall back to UTF-16 with BOM.
    Utf8First,
    /// Prefer UTF-16 with BOM (encoding byte 0x01). Works with both
    /// ID3v2.3 and v2.4.
    Utf16First,
    /// UTF-8 only — return an error if the target version does not support it.
    Utf8Only,
}

impl Default for TextEncodingPreference {
    fn default() -> Self {
        Self::Utf8First
    }
}

impl TextEncodingPreference {
    /// Resolve the encoding byte for a given ID3v2 version.
    ///
    /// # Errors
    ///
    /// Returns `Error::Unsupported` if `Utf8Only` is used with a version < 4.
    pub fn resolve_encoding_byte(&self, version: u8) -> Result<u8, Error> {
        match self {
            Self::Utf8First => {
                if version >= 4 {
                    Ok(ENCODING_UTF8)
                } else {
                    Ok(ENCODING_UTF16_BOM)
                }
            }
            Self::Utf16First => Ok(ENCODING_UTF16_BOM),
            Self::Utf8Only => {
                if version >= 4 {
                    Ok(ENCODING_UTF8)
                } else {
                    Err(Error::Unsupported(
                        "UTF-8 encoding is only supported in ID3v2.4 and later".to_string(),
                    ))
                }
            }
        }
    }
}

// ---- UTF-8 Text Frame Writer ----

/// Writer for ID3v2 text frames with configurable encoding preference.
///
/// Produces the raw frame *body* (encoding byte + encoded text), without
/// the 10-byte ID3v2 frame header.
#[derive(Debug, Clone)]
pub struct Utf8TextFrameWriter {
    /// The encoding preference strategy.
    pub preference: TextEncodingPreference,
}

impl Utf8TextFrameWriter {
    /// Create a new writer with the given encoding preference.
    pub fn new(preference: TextEncodingPreference) -> Self {
        Self { preference }
    }

    /// Write a text frame body for the given frame ID and text content.
    ///
    /// The `version` parameter is the ID3v2 minor version (3 or 4).
    ///
    /// Returns the frame body bytes: `[encoding_byte] [encoded_text]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the encoding preference is incompatible with
    /// the target version (e.g., `Utf8Only` with version 3).
    pub fn write_text_frame(
        &self,
        _frame_id: &str,
        text: &str,
        version: u8,
    ) -> Result<Vec<u8>, Error> {
        let encoding_byte = self.preference.resolve_encoding_byte(version)?;
        let mut buf = Vec::new();
        buf.push(encoding_byte);

        match encoding_byte {
            ENCODING_UTF8 => {
                buf.extend_from_slice(text.as_bytes());
            }
            ENCODING_UTF16_BOM => {
                // Write UTF-16 LE with BOM
                buf.extend_from_slice(&UTF16_LE_BOM);
                for code_unit in text.encode_utf16() {
                    buf.extend_from_slice(&code_unit.to_le_bytes());
                }
            }
            ENCODING_UTF16BE => {
                for code_unit in text.encode_utf16() {
                    buf.extend_from_slice(&code_unit.to_be_bytes());
                }
            }
            ENCODING_LATIN1 => {
                // Only safe for ASCII/Latin-1 subset
                for ch in text.chars() {
                    let byte = u8::try_from(ch as u32).unwrap_or(b'?');
                    buf.push(byte);
                }
            }
            _ => {
                return Err(Error::EncodingError(format!(
                    "Unknown encoding byte: {encoding_byte:#04x}"
                )));
            }
        }

        Ok(buf)
    }

    /// Write multiple text values as a single text frame (null-separated
    /// for multi-value frames like TPE1, TCON, etc.).
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails.
    pub fn write_multi_text_frame(
        &self,
        frame_id: &str,
        texts: &[&str],
        version: u8,
    ) -> Result<Vec<u8>, Error> {
        if texts.is_empty() {
            return self.write_text_frame(frame_id, "", version);
        }

        let encoding_byte = self.preference.resolve_encoding_byte(version)?;
        let mut buf = Vec::new();
        buf.push(encoding_byte);

        for (i, text) in texts.iter().enumerate() {
            if i > 0 {
                // Null separator between values
                match encoding_byte {
                    ENCODING_UTF8 | ENCODING_LATIN1 => buf.push(0x00),
                    ENCODING_UTF16_BOM | ENCODING_UTF16BE => buf.extend_from_slice(&[0x00, 0x00]),
                    _ => buf.push(0x00),
                }
            }

            match encoding_byte {
                ENCODING_UTF8 => {
                    buf.extend_from_slice(text.as_bytes());
                }
                ENCODING_UTF16_BOM => {
                    if i == 0 {
                        buf.extend_from_slice(&UTF16_LE_BOM);
                    }
                    for code_unit in text.encode_utf16() {
                        buf.extend_from_slice(&code_unit.to_le_bytes());
                    }
                }
                ENCODING_UTF16BE => {
                    for code_unit in text.encode_utf16() {
                        buf.extend_from_slice(&code_unit.to_be_bytes());
                    }
                }
                ENCODING_LATIN1 => {
                    for ch in text.chars() {
                        let byte = u8::try_from(ch as u32).unwrap_or(b'?');
                        buf.push(byte);
                    }
                }
                _ => {}
            }
        }

        Ok(buf)
    }
}

impl Default for Utf8TextFrameWriter {
    fn default() -> Self {
        Self::new(TextEncodingPreference::Utf8First)
    }
}

// ---- UTF-8 Text Frame Reader ----

/// Reader for ID3v2 text frame bodies.
///
/// Decodes text from a frame body based on the leading encoding byte.
#[derive(Debug, Clone)]
pub struct Utf8TextFrameReader;

impl Utf8TextFrameReader {
    /// Create a new reader.
    pub fn new() -> Self {
        Self
    }

    /// Read a text frame body, returning the decoded text.
    ///
    /// The input `data` must begin with the encoding byte.
    ///
    /// # Errors
    ///
    /// Returns an error if the encoding byte is invalid or the data
    /// cannot be decoded.
    pub fn read_text_frame(&self, data: &[u8]) -> Result<String, Error> {
        if data.is_empty() {
            return Err(Error::ParseError(
                "Empty text frame data".to_string(),
            ));
        }

        let encoding_byte = data[0];
        let payload = &data[1..];

        match encoding_byte {
            ENCODING_UTF8 => {
                let text = strip_trailing_nulls_u8(payload);
                std::str::from_utf8(text)
                    .map(|s| s.to_string())
                    .map_err(|e| Error::EncodingError(format!("Invalid UTF-8: {e}")))
            }
            ENCODING_LATIN1 => {
                let text = strip_trailing_nulls_u8(payload);
                Ok(text.iter().map(|&b| b as char).collect())
            }
            ENCODING_UTF16_BOM => decode_utf16_with_bom(payload),
            ENCODING_UTF16BE => decode_utf16be(payload),
            other => Err(Error::EncodingError(format!(
                "Unknown text encoding byte: {other:#04x}"
            ))),
        }
    }

    /// Read a multi-value text frame, splitting on null separators.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails.
    pub fn read_multi_text_frame(&self, data: &[u8]) -> Result<Vec<String>, Error> {
        if data.is_empty() {
            return Err(Error::ParseError("Empty text frame data".to_string()));
        }

        let encoding_byte = data[0];
        let payload = &data[1..];

        match encoding_byte {
            ENCODING_UTF8 | ENCODING_LATIN1 => {
                let parts: Vec<&[u8]> = split_on_null_u8(payload);
                let mut results = Vec::new();
                for part in parts {
                    if encoding_byte == ENCODING_UTF8 {
                        let s = std::str::from_utf8(part)
                            .map_err(|e| Error::EncodingError(format!("Invalid UTF-8: {e}")))?;
                        results.push(s.to_string());
                    } else {
                        results.push(part.iter().map(|&b| b as char).collect());
                    }
                }
                Ok(results)
            }
            ENCODING_UTF16_BOM => {
                // The BOM appears once at the start
                let full_text = decode_utf16_with_bom(payload)?;
                Ok(full_text
                    .split('\0')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .collect())
            }
            ENCODING_UTF16BE => {
                let full_text = decode_utf16be(payload)?;
                Ok(full_text
                    .split('\0')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .collect())
            }
            other => Err(Error::EncodingError(format!(
                "Unknown text encoding byte: {other:#04x}"
            ))),
        }
    }
}

impl Default for Utf8TextFrameReader {
    fn default() -> Self {
        Self::new()
    }
}

// ---- Helper functions ----

/// Strip trailing null bytes from a byte slice.
fn strip_trailing_nulls_u8(data: &[u8]) -> &[u8] {
    let mut end = data.len();
    while end > 0 && data[end - 1] == 0 {
        end -= 1;
    }
    &data[..end]
}

/// Split a byte slice on single null bytes (0x00).
fn split_on_null_u8(data: &[u8]) -> Vec<&[u8]> {
    let mut parts = Vec::new();
    let mut start = 0;
    for (i, &b) in data.iter().enumerate() {
        if b == 0x00 {
            if i > start {
                parts.push(&data[start..i]);
            }
            start = i + 1;
        }
    }
    if start < data.len() {
        parts.push(&data[start..]);
    }
    parts
}

/// Decode UTF-16 with BOM from raw bytes.
fn decode_utf16_with_bom(data: &[u8]) -> Result<String, Error> {
    if data.len() < 2 {
        return Ok(String::new());
    }

    let (is_le, payload) = if data[0] == 0xFF && data[1] == 0xFE {
        (true, &data[2..])
    } else if data[0] == 0xFE && data[1] == 0xFF {
        (false, &data[2..])
    } else {
        // Assume LE if no BOM present
        (true, data)
    };

    decode_utf16_bytes(payload, is_le)
}

/// Decode UTF-16BE from raw bytes (no BOM expected).
fn decode_utf16be(data: &[u8]) -> Result<String, Error> {
    decode_utf16_bytes(data, false)
}

/// Decode UTF-16 bytes with explicit endianness.
fn decode_utf16_bytes(data: &[u8], little_endian: bool) -> Result<String, Error> {
    // Ensure we have an even number of bytes
    let len = data.len() & !1;
    let mut code_units = Vec::with_capacity(len / 2);

    for chunk in data[..len].chunks_exact(2) {
        let unit = if little_endian {
            u16::from_le_bytes([chunk[0], chunk[1]])
        } else {
            u16::from_be_bytes([chunk[0], chunk[1]])
        };
        code_units.push(unit);
    }

    // Strip trailing nulls
    while code_units.last() == Some(&0) {
        code_units.pop();
    }

    String::from_utf16(&code_units)
        .map_err(|e| Error::EncodingError(format!("Invalid UTF-16: {e}")))
}

// ---- Id3Error ----

/// Errors that can occur during ID3v2 binary encoding/decoding.
#[derive(Debug, PartialEq, Eq)]
pub enum Id3Error {
    /// The byte sequence does not begin with the `ID3` magic bytes.
    InvalidHeader,
    /// The ID3v2 minor version is not supported.
    UnsupportedVersion(u8),
    /// The data is shorter than expected.
    TruncatedData,
}

impl std::fmt::Display for Id3Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidHeader => write!(f, "Invalid ID3v2 header (missing 'ID3' magic bytes)"),
            Self::UnsupportedVersion(v) => write!(f, "Unsupported ID3v2 version: 2.{v}"),
            Self::TruncatedData => write!(f, "Truncated ID3v2 data"),
        }
    }
}

impl std::error::Error for Id3Error {}

// ---- Id3TextEncoding ----

/// ID3v2 text encoding indicator byte values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Id3TextEncoding {
    /// ISO-8859-1 / Latin-1 (encoding byte 0x00).
    Latin1 = 0,
    /// UTF-16 with BOM (encoding byte 0x01).
    Utf16 = 1,
    /// UTF-16BE without BOM (encoding byte 0x02, ID3v2.4 only).
    Utf16Be = 2,
    /// UTF-8 (encoding byte 0x03, ID3v2.4 only).
    Utf8 = 3,
}

impl Id3TextEncoding {
    /// Return the encoding byte value.
    pub fn byte(self) -> u8 {
        self as u8
    }

    /// Parse from an encoding byte.
    ///
    /// # Errors
    ///
    /// Returns `Id3Error::InvalidHeader` for unrecognised encoding bytes.
    pub fn from_byte(b: u8) -> Result<Self, Id3Error> {
        match b {
            0 => Ok(Self::Latin1),
            1 => Ok(Self::Utf16),
            2 => Ok(Self::Utf16Be),
            3 => Ok(Self::Utf8),
            _ => Err(Id3Error::InvalidHeader),
        }
    }
}

// ---- Id3Frame ----

/// A single ID3v2 text frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Id3Frame {
    /// Four-character frame identifier (e.g., `"TIT2"`, `"TPE1"`).
    pub frame_id: String,
    /// Text encoding used in this frame.
    pub encoding: Id3TextEncoding,
    /// Decoded text content.
    pub content: String,
}

impl Id3Frame {
    /// Create a new ID3 text frame.
    pub fn new(frame_id: impl Into<String>, encoding: Id3TextEncoding, content: impl Into<String>) -> Self {
        Self { frame_id: frame_id.into(), encoding, content: content.into() }
    }
}

// ---- Id3v2Tag ----

/// An ID3v2 tag containing a list of text frames.
///
/// Supports ID3v2.3 (version `(2, 3)`) and ID3v2.4 (version `(2, 4)`).
/// Text encoding preference differs: v2.4 prefers UTF-8, v2.3 uses UTF-16.
#[derive(Debug, Clone)]
pub struct Id3v2Tag {
    /// ID3v2 version as `(major, minor)`.  `major` is always 2; `minor` is
    /// typically 3 or 4.
    pub version: (u8, u8),
    /// Text frames in this tag.
    pub frames: Vec<Id3Frame>,
}

impl Id3v2Tag {
    /// Create a new empty tag for the given version.
    pub fn new(version: (u8, u8)) -> Self {
        Self { version, frames: Vec::new() }
    }

    /// Get the preferred text encoding for this tag version.
    fn preferred_encoding(&self) -> Id3TextEncoding {
        if self.version.1 >= 4 {
            Id3TextEncoding::Utf8
        } else {
            Id3TextEncoding::Utf16
        }
    }

    /// Get a frame by its 4-character ID.
    pub fn get_frame(&self, id: &str) -> Option<&Id3Frame> {
        self.frames.iter().find(|f| f.frame_id == id)
    }

    /// Remove any existing frame with the given ID and insert a new one.
    fn set_text_frame(&mut self, id: &str, text: &str) {
        self.frames.retain(|f| f.frame_id != id);
        self.frames.push(Id3Frame::new(id, self.preferred_encoding(), text));
    }

    /// Set the title (TIT2).
    pub fn set_title(&mut self, title: &str) {
        self.set_text_frame("TIT2", title);
    }

    /// Set the lead artist (TPE1).
    pub fn set_artist(&mut self, artist: &str) {
        self.set_text_frame("TPE1", artist);
    }

    /// Set the album (TALB).
    pub fn set_album(&mut self, album: &str) {
        self.set_text_frame("TALB", album);
    }

    /// Set the recording year (TDRC in v2.4, TYER in v2.3).
    pub fn set_year(&mut self, year: u16) {
        let id = if self.version.1 >= 4 { "TDRC" } else { "TYER" };
        self.set_text_frame(id, &year.to_string());
    }

    /// Set the content type / genre (TCON).
    pub fn set_genre(&mut self, genre: &str) {
        self.set_text_frame("TCON", genre);
    }

    /// Get the title, if present.
    pub fn title(&self) -> Option<&str> {
        self.get_frame("TIT2").map(|f| f.content.as_str())
    }

    /// Get the lead artist, if present.
    pub fn artist(&self) -> Option<&str> {
        self.get_frame("TPE1").map(|f| f.content.as_str())
    }

    /// Get the album, if present.
    pub fn album(&self) -> Option<&str> {
        self.get_frame("TALB").map(|f| f.content.as_str())
    }

    /// Get the year as a string (checks both TDRC and TYER).
    pub fn year_str(&self) -> Option<&str> {
        self.get_frame("TDRC")
            .or_else(|| self.get_frame("TYER"))
            .map(|f| f.content.as_str())
    }

    /// Get the genre, if present.
    pub fn genre(&self) -> Option<&str> {
        self.get_frame("TCON").map(|f| f.content.as_str())
    }
}

// ---- Id3Encoder ----

/// Encodes `Id3Frame` and `Id3v2Tag` structs to binary ID3v2 format.
pub struct Id3Encoder;

impl Id3Encoder {
    /// Encode a single `Id3Frame` to its binary representation.
    ///
    /// The result is the full 10-byte frame header followed by the frame body.
    pub fn encode_frame(frame: &Id3Frame) -> Vec<u8> {
        let body = Self::encode_frame_body(frame);
        let mut out = Vec::with_capacity(10 + body.len());
        // Frame ID (4 bytes, ASCII padded with spaces if needed)
        let id_bytes: Vec<u8> = frame.frame_id.bytes().take(4).collect();
        out.extend_from_slice(&id_bytes);
        // Pad to 4 bytes if shorter
        for _ in id_bytes.len()..4 {
            out.push(b' ');
        }
        // Frame size as 4-byte big-endian (plain, not syncsafe)
        let size = body.len() as u32;
        out.extend_from_slice(&size.to_be_bytes());
        // Flags (2 bytes, always 0 for basic frames)
        out.extend_from_slice(&[0x00, 0x00]);
        // Frame body
        out.extend_from_slice(&body);
        out
    }

    /// Encode the body of a text frame (encoding byte + encoded text).
    fn encode_frame_body(frame: &Id3Frame) -> Vec<u8> {
        let mut body = Vec::new();
        body.push(frame.encoding.byte());
        match frame.encoding {
            Id3TextEncoding::Utf8 => {
                body.extend_from_slice(frame.content.as_bytes());
            }
            Id3TextEncoding::Latin1 => {
                for ch in frame.content.chars() {
                    body.push(u8::try_from(ch as u32).unwrap_or(b'?'));
                }
            }
            Id3TextEncoding::Utf16 => {
                // BOM + UTF-16 LE
                body.extend_from_slice(&[0xFF, 0xFE]);
                for unit in frame.content.encode_utf16() {
                    body.extend_from_slice(&unit.to_le_bytes());
                }
            }
            Id3TextEncoding::Utf16Be => {
                for unit in frame.content.encode_utf16() {
                    body.extend_from_slice(&unit.to_be_bytes());
                }
            }
        }
        body
    }

    /// Encode a complete `Id3v2Tag` to binary, including the 10-byte ID3v2 header.
    ///
    /// The tag size field in the header uses syncsafe integer encoding per spec.
    pub fn encode_tag(tag: &Id3v2Tag) -> Vec<u8> {
        // Encode all frames
        let frames_bytes: Vec<u8> = tag.frames.iter().flat_map(|f| Self::encode_frame(f)).collect();

        let tag_size = frames_bytes.len() as u32;
        // ID3v2 header: 3 bytes magic + 2 bytes version + 1 byte flags + 4 bytes syncsafe size
        let mut out = Vec::with_capacity(10 + frames_bytes.len());
        // Magic
        out.extend_from_slice(b"ID3");
        // Version: major.minor
        out.push(tag.version.0);
        out.push(tag.version.1);
        // Flags (0 = no unsync, no extended header, etc.)
        out.push(0x00);
        // Syncsafe size
        out.extend_from_slice(&to_syncsafe(tag_size));
        out.extend_from_slice(&frames_bytes);
        out
    }
}

// ---- Id3Decoder ----

/// Decodes binary ID3v2 data.
pub struct Id3Decoder;

impl Id3Decoder {
    /// Decode the ID3v2 tag header and return `(major_version, minor_version, tag_size)`.
    ///
    /// `tag_size` is the syncsafe-decoded number of bytes following the header.
    ///
    /// # Errors
    ///
    /// - `Id3Error::TruncatedData` — fewer than 10 bytes provided.
    /// - `Id3Error::InvalidHeader` — bytes 0–2 are not `ID3`.
    /// - `Id3Error::UnsupportedVersion` — minor version is not 3 or 4.
    pub fn decode_header(bytes: &[u8]) -> Result<(u8, u8, u32), Id3Error> {
        if bytes.len() < 10 {
            return Err(Id3Error::TruncatedData);
        }
        if &bytes[0..3] != b"ID3" {
            return Err(Id3Error::InvalidHeader);
        }
        let major = bytes[3];
        let minor = bytes[4];
        if minor != 3 && minor != 4 {
            return Err(Id3Error::UnsupportedVersion(minor));
        }
        let size = from_syncsafe([bytes[6], bytes[7], bytes[8], bytes[9]]);
        Ok((major, minor, size))
    }

    /// Decode all text frames from a complete tag binary (starting at the ID3 header).
    ///
    /// # Errors
    ///
    /// Returns `Id3Error` variants if the header is invalid.
    pub fn decode_tag(bytes: &[u8]) -> Result<Id3v2Tag, Id3Error> {
        let (major, minor, _tag_size) = Self::decode_header(bytes)?;
        let mut tag = Id3v2Tag::new((major, minor));
        let mut offset = 10usize;

        while offset + 10 <= bytes.len() {
            let frame_id_bytes = &bytes[offset..offset + 4];
            // Stop at padding
            if frame_id_bytes.iter().all(|&b| b == 0) { break; }

            let frame_id = match std::str::from_utf8(frame_id_bytes) {
                Ok(s) => s.trim_end().to_string(),
                Err(_) => break,
            };
            let size = u32::from_be_bytes([
                bytes[offset + 4],
                bytes[offset + 5],
                bytes[offset + 6],
                bytes[offset + 7],
            ]) as usize;
            // Skip flags (2 bytes)
            offset += 10;

            if offset + size > bytes.len() { break; }
            let body = &bytes[offset..offset + size];
            offset += size;

            if body.is_empty() { continue; }
            let encoding = match Id3TextEncoding::from_byte(body[0]) {
                Ok(e) => e,
                Err(_) => continue,
            };

            let reader = Utf8TextFrameReader::new();
            if let Ok(content) = reader.read_text_frame(body) {
                tag.frames.push(Id3Frame::new(frame_id, encoding, content));
            }
        }
        Ok(tag)
    }
}

// ---- Syncsafe integer helpers ----

/// Encode a 28-bit value as a 4-byte syncsafe integer.
fn to_syncsafe(n: u32) -> [u8; 4] {
    [
        ((n >> 21) & 0x7F) as u8,
        ((n >> 14) & 0x7F) as u8,
        ((n >> 7) & 0x7F) as u8,
        (n & 0x7F) as u8,
    ]
}

/// Decode a 4-byte syncsafe integer.
fn from_syncsafe(b: [u8; 4]) -> u32 {
    (u32::from(b[0]) << 21)
        | (u32::from(b[1]) << 14)
        | (u32::from(b[2]) << 7)
        | u32::from(b[3])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoding_preference_utf8_first_v24() {
        let pref = TextEncodingPreference::Utf8First;
        assert_eq!(pref.resolve_encoding_byte(4).expect("ok"), ENCODING_UTF8);
    }

    #[test]
    fn test_encoding_preference_utf8_first_v23_fallback() {
        let pref = TextEncodingPreference::Utf8First;
        // v2.3 should fall back to UTF-16
        assert_eq!(
            pref.resolve_encoding_byte(3).expect("ok"),
            ENCODING_UTF16_BOM
        );
    }

    #[test]
    fn test_encoding_preference_utf8_only_v23_error() {
        let pref = TextEncodingPreference::Utf8Only;
        assert!(pref.resolve_encoding_byte(3).is_err());
    }

    #[test]
    fn test_encoding_preference_utf8_only_v24_ok() {
        let pref = TextEncodingPreference::Utf8Only;
        assert_eq!(pref.resolve_encoding_byte(4).expect("ok"), ENCODING_UTF8);
    }

    #[test]
    fn test_encoding_preference_utf16_first() {
        let pref = TextEncodingPreference::Utf16First;
        assert_eq!(
            pref.resolve_encoding_byte(3).expect("ok"),
            ENCODING_UTF16_BOM
        );
        assert_eq!(
            pref.resolve_encoding_byte(4).expect("ok"),
            ENCODING_UTF16_BOM
        );
    }

    #[test]
    fn test_utf8_round_trip_ascii() {
        let writer = Utf8TextFrameWriter::new(TextEncodingPreference::Utf8First);
        let frame = writer
            .write_text_frame("TIT2", "Hello World", 4)
            .expect("write");

        // Encoding byte should be 0x03
        assert_eq!(frame[0], ENCODING_UTF8);
        assert_eq!(&frame[1..], b"Hello World");

        let reader = Utf8TextFrameReader::new();
        let text = reader.read_text_frame(&frame).expect("read");
        assert_eq!(text, "Hello World");
    }

    #[test]
    fn test_utf8_round_trip_japanese() {
        let writer = Utf8TextFrameWriter::new(TextEncodingPreference::Utf8First);
        let japanese = "\u{3053}\u{3093}\u{306B}\u{3061}\u{306F}\u{4E16}\u{754C}"; // こんにちは世界
        let frame = writer
            .write_text_frame("TIT2", japanese, 4)
            .expect("write");

        assert_eq!(frame[0], ENCODING_UTF8);

        let reader = Utf8TextFrameReader::new();
        let text = reader.read_text_frame(&frame).expect("read");
        assert_eq!(text, japanese);
    }

    #[test]
    fn test_utf8_round_trip_emoji() {
        let writer = Utf8TextFrameWriter::new(TextEncodingPreference::Utf8First);
        let emoji = "Music \u{1F3B5} is life \u{2764}\u{FE0F}";
        let frame = writer
            .write_text_frame("TIT2", emoji, 4)
            .expect("write");

        assert_eq!(frame[0], ENCODING_UTF8);

        let reader = Utf8TextFrameReader::new();
        let text = reader.read_text_frame(&frame).expect("read");
        assert_eq!(text, emoji);
    }

    #[test]
    fn test_utf8_round_trip_empty_string() {
        let writer = Utf8TextFrameWriter::new(TextEncodingPreference::Utf8First);
        let frame = writer
            .write_text_frame("TIT2", "", 4)
            .expect("write");

        assert_eq!(frame.len(), 1); // just the encoding byte
        assert_eq!(frame[0], ENCODING_UTF8);

        let reader = Utf8TextFrameReader::new();
        let text = reader.read_text_frame(&frame).expect("read");
        assert_eq!(text, "");
    }

    #[test]
    fn test_utf16_fallback_for_v23() {
        let writer = Utf8TextFrameWriter::new(TextEncodingPreference::Utf8First);
        let frame = writer
            .write_text_frame("TIT2", "Hello", 3)
            .expect("write v2.3");

        // Should fall back to UTF-16 with BOM
        assert_eq!(frame[0], ENCODING_UTF16_BOM);
        // BOM should follow
        assert_eq!(frame[1], 0xFF);
        assert_eq!(frame[2], 0xFE);

        let reader = Utf8TextFrameReader::new();
        let text = reader.read_text_frame(&frame).expect("read");
        assert_eq!(text, "Hello");
    }

    #[test]
    fn test_utf16_round_trip_japanese() {
        let writer = Utf8TextFrameWriter::new(TextEncodingPreference::Utf16First);
        let japanese = "\u{65E5}\u{672C}\u{8A9E}"; // 日本語
        let frame = writer
            .write_text_frame("TIT2", japanese, 4)
            .expect("write");

        assert_eq!(frame[0], ENCODING_UTF16_BOM);

        let reader = Utf8TextFrameReader::new();
        let text = reader.read_text_frame(&frame).expect("read");
        assert_eq!(text, japanese);
    }

    #[test]
    fn test_multi_text_frame_utf8() {
        let writer = Utf8TextFrameWriter::new(TextEncodingPreference::Utf8First);
        let texts = &["Rock", "Pop", "Jazz"];
        let frame = writer
            .write_multi_text_frame("TCON", texts, 4)
            .expect("write");

        assert_eq!(frame[0], ENCODING_UTF8);

        let reader = Utf8TextFrameReader::new();
        let values = reader.read_multi_text_frame(&frame).expect("read");
        assert_eq!(values, vec!["Rock", "Pop", "Jazz"]);
    }

    #[test]
    fn test_read_empty_frame_error() {
        let reader = Utf8TextFrameReader::new();
        assert!(reader.read_text_frame(&[]).is_err());
    }

    #[test]
    fn test_read_unknown_encoding_byte_error() {
        let reader = Utf8TextFrameReader::new();
        assert!(reader.read_text_frame(&[0xFF, 0x41, 0x42]).is_err());
    }

    #[test]
    fn test_latin1_encoding() {
        // Manually construct a Latin-1 frame
        let data = vec![ENCODING_LATIN1, b'H', b'e', b'l', b'l', b'o'];
        let reader = Utf8TextFrameReader::new();
        let text = reader.read_text_frame(&data).expect("read");
        assert_eq!(text, "Hello");
    }

    #[test]
    fn test_utf16be_encoding() {
        // Manually construct a UTF-16BE frame
        let mut data = vec![ENCODING_UTF16BE];
        for &ch in &[0x0048u16, 0x0069u16] {
            // "Hi"
            data.extend_from_slice(&ch.to_be_bytes());
        }

        let reader = Utf8TextFrameReader::new();
        let text = reader.read_text_frame(&data).expect("read");
        assert_eq!(text, "Hi");
    }

    #[test]
    fn test_default_preference() {
        let pref = TextEncodingPreference::default();
        assert_eq!(pref, TextEncodingPreference::Utf8First);
    }

    #[test]
    fn test_default_writer() {
        let writer = Utf8TextFrameWriter::default();
        assert_eq!(writer.preference, TextEncodingPreference::Utf8First);
    }

    #[test]
    fn test_default_reader() {
        let _reader = Utf8TextFrameReader::default();
        // Just ensuring Default works
    }

    #[test]
    fn test_strip_trailing_nulls() {
        assert_eq!(strip_trailing_nulls_u8(b"hello\0\0"), b"hello");
        assert_eq!(strip_trailing_nulls_u8(b"hello"), b"hello");
        assert_eq!(strip_trailing_nulls_u8(b"\0"), b"" as &[u8]);
        assert_eq!(strip_trailing_nulls_u8(b""), b"" as &[u8]);
    }

    #[test]
    fn test_mixed_script_round_trip() {
        // Korean + Arabic + Latin
        let mixed = "Hello \u{D55C}\u{AD6D}\u{C5B4} \u{0645}\u{0631}\u{062D}\u{0628}\u{0627}";
        let writer = Utf8TextFrameWriter::new(TextEncodingPreference::Utf8First);
        let frame = writer
            .write_text_frame("TIT2", mixed, 4)
            .expect("write");

        let reader = Utf8TextFrameReader::new();
        let text = reader.read_text_frame(&frame).expect("read");
        assert_eq!(text, mixed);
    }

    // ---- Id3TextEncoding / Id3Frame / Id3v2Tag / Id3Encoder / Id3Decoder tests ----

    #[test]
    fn test_id3_text_encoding_byte_values() {
        assert_eq!(Id3TextEncoding::Latin1.byte(), 0);
        assert_eq!(Id3TextEncoding::Utf16.byte(), 1);
        assert_eq!(Id3TextEncoding::Utf16Be.byte(), 2);
        assert_eq!(Id3TextEncoding::Utf8.byte(), 3);
    }

    #[test]
    fn test_id3_text_encoding_from_byte_round_trip() {
        for b in 0u8..=3 {
            let enc = Id3TextEncoding::from_byte(b).expect("valid");
            assert_eq!(enc.byte(), b);
        }
        assert!(Id3TextEncoding::from_byte(4).is_err());
    }

    #[test]
    fn test_id3v2_tag_v24_prefers_utf8() {
        let mut tag = Id3v2Tag::new((2, 4));
        tag.set_title("My Song");
        let frame = tag.get_frame("TIT2").expect("TIT2 exists");
        assert_eq!(frame.encoding, Id3TextEncoding::Utf8);
        assert_eq!(tag.title(), Some("My Song"));
    }

    #[test]
    fn test_id3v2_tag_v23_prefers_utf16() {
        let mut tag = Id3v2Tag::new((2, 3));
        tag.set_title("My Song");
        let frame = tag.get_frame("TIT2").expect("TIT2 exists");
        assert_eq!(frame.encoding, Id3TextEncoding::Utf16);
        assert_eq!(tag.title(), Some("My Song"));
    }

    #[test]
    fn test_id3v2_tag_title_artist_album_round_trip() {
        let mut tag = Id3v2Tag::new((2, 4));
        tag.set_title("Test Title");
        tag.set_artist("Test Artist");
        tag.set_album("Test Album");
        tag.set_year(2025);
        tag.set_genre("Rock");

        assert_eq!(tag.title(), Some("Test Title"));
        assert_eq!(tag.artist(), Some("Test Artist"));
        assert_eq!(tag.album(), Some("Test Album"));
        assert_eq!(tag.year_str(), Some("2025"));
        assert_eq!(tag.genre(), Some("Rock"));
    }

    #[test]
    fn test_id3_encoder_encode_frame_header_and_body() {
        let frame = Id3Frame::new("TIT2", Id3TextEncoding::Utf8, "Hello");
        let encoded = Id3Encoder::encode_frame(&frame);
        // First 4 bytes: frame ID
        assert_eq!(&encoded[0..4], b"TIT2");
        // Bytes 4-7: size in big-endian (body = 1 encoding byte + 5 UTF-8 bytes = 6)
        let size = u32::from_be_bytes([encoded[4], encoded[5], encoded[6], encoded[7]]);
        assert_eq!(size, 6);
        // Bytes 8-9: flags (0)
        assert_eq!(encoded[8], 0);
        assert_eq!(encoded[9], 0);
        // Byte 10: encoding = 0x03 (UTF-8)
        assert_eq!(encoded[10], 0x03);
        // Bytes 11-15: "Hello"
        assert_eq!(&encoded[11..], b"Hello");
    }

    #[test]
    fn test_id3_encoder_encode_tag_magic_bytes() {
        let tag = Id3v2Tag::new((2, 4));
        let encoded = Id3Encoder::encode_tag(&tag);
        // First 3 bytes must be "ID3"
        assert_eq!(&encoded[0..3], b"ID3");
        // Byte 3: major version = 2
        assert_eq!(encoded[3], 2);
        // Byte 4: minor version = 4
        assert_eq!(encoded[4], 4);
    }

    #[test]
    fn test_id3_decoder_decode_header_valid() {
        let mut tag = Id3v2Tag::new((2, 4));
        tag.set_title("Decode Me");
        let encoded = Id3Encoder::encode_tag(&tag);
        let (major, minor, size) = Id3Decoder::decode_header(&encoded).expect("should decode");
        assert_eq!(major, 2);
        assert_eq!(minor, 4);
        assert!(size > 0);
    }

    #[test]
    fn test_id3_decoder_decode_header_invalid_magic() {
        let bytes = b"XMP\x02\x04\x00\x00\x00\x00\x00";
        let result = Id3Decoder::decode_header(bytes);
        assert_eq!(result, Err(Id3Error::InvalidHeader));
    }

    #[test]
    fn test_id3_decoder_decode_header_truncated() {
        let bytes = b"ID3\x02\x04";
        let result = Id3Decoder::decode_header(bytes);
        assert_eq!(result, Err(Id3Error::TruncatedData));
    }

    #[test]
    fn test_id3_encoder_decoder_tag_round_trip() {
        let mut tag = Id3v2Tag::new((2, 4));
        tag.set_title("Round Trip");
        tag.set_artist("OxiMedia");
        tag.set_album("Test Suite");
        tag.set_genre("Electronic");

        let encoded = Id3Encoder::encode_tag(&tag);
        let decoded = Id3Decoder::decode_tag(&encoded).expect("should decode");

        assert_eq!(decoded.title(), Some("Round Trip"));
        assert_eq!(decoded.artist(), Some("OxiMedia"));
        assert_eq!(decoded.album(), Some("Test Suite"));
        assert_eq!(decoded.genre(), Some("Electronic"));
    }

    #[test]
    fn test_id3_syncsafe_round_trip() {
        for &n in &[0u32, 127, 128, 1024, 65535, 2_097_151] {
            let encoded = to_syncsafe(n);
            let decoded = from_syncsafe(encoded);
            assert_eq!(decoded, n, "syncsafe round-trip failed for {n}");
        }
    }

    #[test]
    fn test_id3_error_display() {
        assert!(format!("{}", Id3Error::InvalidHeader).contains("ID3"));
        assert!(format!("{}", Id3Error::UnsupportedVersion(5)).contains('5'));
        assert!(format!("{}", Id3Error::TruncatedData).contains("Truncated"));
    }
}
