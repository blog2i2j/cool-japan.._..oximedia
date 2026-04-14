//! OGG page format parsing and serialization.
//!
//! This module implements the OGG page format as specified in RFC 3533.
//! An OGG bitstream is divided into pages; each page carries a header
//! describing its position in the logical bitstream plus up to 255 segments
//! of packet data.
//!
//! # CRC-32
//!
//! OGG uses CRC-32 with the polynomial `0x04c11db7` (big-endian/normal form),
//! reflected both in and out — this is the standard CRC-32 used in ZIP/Ethernet
//! but applied to the full page bytes with the four checksum bytes zeroed.
//!
//! # Example
//!
//! ```
//! use oximedia_container::ogg_page::{OggPageHeader, OggPage, OggPageType, serialize_ogg_page, parse_ogg_page};
//!
//! let header = OggPageHeader {
//!     version: 0,
//!     header_type: 0x02, // BOS
//!     granule_pos: 0,
//!     serial: 0x1234,
//!     seq_num: 0,
//!     checksum: 0,
//!     segment_count: 1,
//! };
//! let page = OggPage {
//!     header,
//!     segment_table: vec![5],
//!     page_data: vec![0x4f, 0x67, 0x67, 0x53, 0x01],
//! };
//! let bytes = serialize_ogg_page(&page);
//! assert!(bytes.len() >= 27);
//! ```

use std::fmt;
use thiserror::Error;

/// Magic bytes at the start of every OGG page.
pub const OGG_PAGE_MAGIC: &[u8; 4] = b"OggS";

/// Fixed size of the page header (without segment table).
///
/// `4` (magic) + `1` (version) + `1` (header_type) + `8` (granule_pos) +
/// `4` (serial) + `4` (seq_num) + `4` (checksum) + `1` (segment_count) = 27.
pub const OGG_PAGE_HEADER_SIZE: usize = 27;

/// OGG CRC-32 polynomial (normal / non-reflected form).
///
/// OGG uses CRC-32 with polynomial `0x04c11db7`, initial value 0, no input or
/// output reflection, and no final XOR — this is distinct from the standard
/// IEEE 802.3 CRC (which uses initial value `0xFFFFFFFF` and reflection).
/// The algorithm is sometimes called "CRC-32/OGG" or "CRC-32/BZIP2-0".
const OGG_CRC32_POLY: u32 = 0x04c1_1db7;

/// Errors that can occur during OGG page parsing.
#[derive(Debug, Error)]
pub enum OggError {
    /// The byte slice is too short to contain a valid page header.
    #[error("buffer too short: need at least {needed} bytes, got {got}")]
    BufferTooShort { needed: usize, got: usize },

    /// Magic bytes are not "OggS".
    #[error("invalid OGG magic: expected b\"OggS\", got {0:?}")]
    InvalidMagic([u8; 4]),

    /// OGG stream version must be 0.
    #[error("unsupported OGG version: {0}")]
    UnsupportedVersion(u8),

    /// Computed CRC-32 does not match the stored checksum.
    #[error("CRC-32 mismatch: computed {computed:#010x}, stored {stored:#010x}")]
    ChecksumMismatch { computed: u32, stored: u32 },

    /// Segment table or payload extends past the buffer end.
    #[error("page data extends past buffer end")]
    DataOverflow,
}

/// Page type derived from the `header_type` flags byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OggPageType {
    /// Beginning of stream — first page for a logical bitstream.
    Bos,
    /// End of stream — last page for a logical bitstream.
    Eos,
    /// Continuation page — contains the rest of a packet started on a previous page.
    Continuation,
    /// Normal page — contains one or more complete packets (no BOS/EOS/continuation flag).
    Normal,
}

impl fmt::Display for OggPageType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bos => f.write_str("BOS"),
            Self::Eos => f.write_str("EOS"),
            Self::Continuation => f.write_str("Continuation"),
            Self::Normal => f.write_str("Normal"),
        }
    }
}

/// Bit flag: continuation packet.
pub const HEADER_TYPE_CONTINUATION: u8 = 0x01;
/// Bit flag: beginning of stream.
pub const HEADER_TYPE_BOS: u8 = 0x02;
/// Bit flag: end of stream.
pub const HEADER_TYPE_EOS: u8 = 0x04;

impl OggPageType {
    /// Derive the page type from the raw `header_type` flags byte.
    ///
    /// When multiple flags are set the priority order is BOS > EOS > Continuation > Normal.
    #[must_use]
    pub const fn from_flags(flags: u8) -> Self {
        if flags & HEADER_TYPE_BOS != 0 {
            Self::Bos
        } else if flags & HEADER_TYPE_EOS != 0 {
            Self::Eos
        } else if flags & HEADER_TYPE_CONTINUATION != 0 {
            Self::Continuation
        } else {
            Self::Normal
        }
    }

    /// Returns `true` if the continuation flag is set in `flags`.
    #[must_use]
    pub const fn is_continued(flags: u8) -> bool {
        flags & HEADER_TYPE_CONTINUATION != 0
    }

    /// Returns `true` if this is a beginning-of-stream page.
    #[must_use]
    pub const fn is_bos(self) -> bool {
        matches!(self, Self::Bos)
    }

    /// Returns `true` if this is an end-of-stream page.
    #[must_use]
    pub const fn is_eos(self) -> bool {
        matches!(self, Self::Eos)
    }
}

/// OGG CRC-32 engine.
///
/// Implements the standard CRC-32 algorithm (IEEE 802.3 polynomial) which is
/// the algorithm OGG uses for its page checksum.  The lookup table is built
/// once and reused for each computation.
pub struct OggCrc32 {
    table: [u32; 256],
}

impl OggCrc32 {
    /// Build a new CRC-32 engine with the OGG lookup table.
    ///
    /// The table is built using the normal (non-reflected) polynomial
    /// `0x04c11db7` with MSB-first shift direction.
    #[must_use]
    pub fn new() -> Self {
        let mut table = [0u32; 256];
        for (i, entry) in table.iter_mut().enumerate() {
            let mut crc = (i as u32) << 24;
            for _ in 0..8 {
                if crc & 0x8000_0000 != 0 {
                    crc = (crc << 1) ^ OGG_CRC32_POLY;
                } else {
                    crc <<= 1;
                }
            }
            *entry = crc;
        }
        Self { table }
    }

    /// Compute the OGG CRC-32 over `data`.
    ///
    /// Uses initial value 0, no reflection, and no final XOR, matching the
    /// OGG specification (RFC 3533 §6.3).  The four checksum bytes in the
    /// page must be zeroed *before* feeding the page bytes into this function.
    #[must_use]
    pub fn compute(&self, data: &[u8]) -> u32 {
        let mut crc: u32 = 0;
        for &byte in data {
            let index = ((crc >> 24) ^ u32::from(byte)) as usize & 0xFF;
            crc = (crc << 8) ^ self.table[index];
        }
        crc
    }
}

impl Default for OggCrc32 {
    fn default() -> Self {
        Self::new()
    }
}

/// OGG page header (27 bytes on the wire).
///
/// Layout (RFC 3533 §6):
/// ```text
/// offset  size  field
///    0     4    capture_pattern ("OggS")
///    4     1    stream_structure_version  (must be 0)
///    5     1    header_type_flag
///    6     8    absolute_granule_position (little-endian i64)
///   14     4    stream_serial_number      (little-endian u32)
///   18     4    page_sequence_no          (little-endian u32)
///   22     4    CRC_checksum              (little-endian u32)
///   26     1    number_page_segments
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OggPageHeader {
    /// Stream structure version — must be 0.
    pub version: u8,
    /// Header type flags (`HEADER_TYPE_*` constants).
    pub header_type: u8,
    /// Absolute granule position (codec-specific; -1 / `u64::MAX` means "not yet known").
    pub granule_pos: i64,
    /// Logical bitstream serial number.
    pub serial: u32,
    /// Page sequence number (monotonically increasing per logical bitstream).
    pub seq_num: u32,
    /// CRC-32 checksum of the complete page with this field zeroed.
    pub checksum: u32,
    /// Number of entries in the lacing (segment) table that follows the header.
    pub segment_count: u8,
}

impl OggPageHeader {
    /// Derive the [`OggPageType`] for this header.
    #[must_use]
    pub fn page_type(&self) -> OggPageType {
        OggPageType::from_flags(self.header_type)
    }

    /// Returns `true` if the continued-packet flag is set.
    #[must_use]
    pub fn is_continued(&self) -> bool {
        OggPageType::is_continued(self.header_type)
    }
}

/// A complete OGG page: header, lacing table, and payload bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OggPage {
    /// Parsed header fields.
    pub header: OggPageHeader,
    /// Lacing (segment) table — one byte per segment giving its size (0–255).
    ///
    /// The total payload size is the sum of all entries.
    pub segment_table: Vec<u8>,
    /// Raw payload bytes (segments concatenated).
    pub page_data: Vec<u8>,
}

impl OggPage {
    /// Returns the [`OggPageType`] for this page.
    #[must_use]
    pub fn page_type(&self) -> OggPageType {
        self.header.page_type()
    }

    /// Returns the total number of payload bytes in this page.
    #[must_use]
    pub fn payload_len(&self) -> usize {
        self.segment_table.iter().map(|&s| s as usize).sum()
    }

    /// Iterate over the logical packets contained in this page.
    ///
    /// A packet boundary is indicated by a segment whose lace value is < 255.
    /// If the last segment has value 255 the packet continues on the next page.
    pub fn packets(&self) -> impl Iterator<Item = &[u8]> {
        OggPacketIter::new(&self.segment_table, &self.page_data)
    }
}

/// Iterator over packet slices in a single OGG page.
struct OggPacketIter<'a> {
    segtab: &'a [u8],
    data: &'a [u8],
    seg_pos: usize,
    data_pos: usize,
}

impl<'a> OggPacketIter<'a> {
    fn new(segtab: &'a [u8], data: &'a [u8]) -> Self {
        Self {
            segtab,
            data,
            seg_pos: 0,
            data_pos: 0,
        }
    }
}

impl<'a> Iterator for OggPacketIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.seg_pos >= self.segtab.len() {
            return None;
        }
        let start = self.data_pos;
        loop {
            if self.seg_pos >= self.segtab.len() {
                break;
            }
            let seg = self.segtab[self.seg_pos];
            self.seg_pos += 1;
            self.data_pos += seg as usize;
            if seg < 255 {
                break; // end of this packet
            }
        }
        let end = self.data_pos.min(self.data.len());
        Some(&self.data[start..end])
    }
}

// ─── Parsing ─────────────────────────────────────────────────────────────────

/// Parse a single OGG page from `buf`.
///
/// Returns `(page, consumed_bytes)` where `consumed_bytes` is the number of
/// bytes read from `buf`.  The function verifies the CRC-32 checksum.
///
/// # Errors
///
/// Returns [`OggError`] if the buffer is too short, the magic is wrong, the
/// version is non-zero, or the checksum does not match.
pub fn parse_ogg_page(buf: &[u8]) -> Result<(OggPage, usize), OggError> {
    // Need at least the fixed header.
    if buf.len() < OGG_PAGE_HEADER_SIZE {
        return Err(OggError::BufferTooShort {
            needed: OGG_PAGE_HEADER_SIZE,
            got: buf.len(),
        });
    }

    // Magic check.
    let magic: [u8; 4] = buf[0..4].try_into().unwrap_or([0; 4]);
    if &magic != OGG_PAGE_MAGIC {
        return Err(OggError::InvalidMagic(magic));
    }

    let version = buf[4];
    if version != 0 {
        return Err(OggError::UnsupportedVersion(version));
    }

    let header_type = buf[5];
    let granule_pos =
        i64::from_le_bytes(
            buf[6..14]
                .try_into()
                .map_err(|_| OggError::BufferTooShort {
                    needed: 14,
                    got: buf.len(),
                })?,
        );
    let serial =
        u32::from_le_bytes(
            buf[14..18]
                .try_into()
                .map_err(|_| OggError::BufferTooShort {
                    needed: 18,
                    got: buf.len(),
                })?,
        );
    let seq_num =
        u32::from_le_bytes(
            buf[18..22]
                .try_into()
                .map_err(|_| OggError::BufferTooShort {
                    needed: 22,
                    got: buf.len(),
                })?,
        );
    let stored_checksum =
        u32::from_le_bytes(
            buf[22..26]
                .try_into()
                .map_err(|_| OggError::BufferTooShort {
                    needed: 26,
                    got: buf.len(),
                })?,
        );
    let segment_count = buf[26];

    // Segment table.
    let seg_table_end = OGG_PAGE_HEADER_SIZE + segment_count as usize;
    if buf.len() < seg_table_end {
        return Err(OggError::BufferTooShort {
            needed: seg_table_end,
            got: buf.len(),
        });
    }
    let segment_table: Vec<u8> = buf[OGG_PAGE_HEADER_SIZE..seg_table_end].to_vec();

    // Payload.
    let payload_len: usize = segment_table.iter().map(|&s| s as usize).sum();
    let page_end = seg_table_end + payload_len;
    if buf.len() < page_end {
        return Err(OggError::DataOverflow);
    }
    let page_data: Vec<u8> = buf[seg_table_end..page_end].to_vec();

    // CRC-32 verification: zero out the four checksum bytes, compute, compare.
    let crc_engine = OggCrc32::new();
    let computed = {
        // Build a temporary buffer with checksum zeroed.
        let mut tmp = buf[..page_end].to_vec();
        tmp[22] = 0;
        tmp[23] = 0;
        tmp[24] = 0;
        tmp[25] = 0;
        crc_engine.compute(&tmp)
    };

    if computed != stored_checksum {
        return Err(OggError::ChecksumMismatch {
            computed,
            stored: stored_checksum,
        });
    }

    let header = OggPageHeader {
        version,
        header_type,
        granule_pos,
        serial,
        seq_num,
        checksum: stored_checksum,
        segment_count,
    };
    let page = OggPage {
        header,
        segment_table,
        page_data,
    };
    Ok((page, page_end))
}

// ─── Serialization ───────────────────────────────────────────────────────────

/// Serialize an [`OggPage`] into a byte vector, computing and inserting the
/// correct CRC-32 checksum.
///
/// The `checksum` field of `page.header` is **ignored** — the correct value is
/// always computed and written.
#[must_use]
pub fn serialize_ogg_page(page: &OggPage) -> Vec<u8> {
    let payload_len: usize = page.segment_table.iter().map(|&s| s as usize).sum();
    let total = OGG_PAGE_HEADER_SIZE + page.segment_table.len() + payload_len;
    let mut out = Vec::with_capacity(total);

    // Magic
    out.extend_from_slice(OGG_PAGE_MAGIC);
    // Version
    out.push(page.header.version);
    // Header type
    out.push(page.header.header_type);
    // Granule position (LE i64)
    out.extend_from_slice(&page.header.granule_pos.to_le_bytes());
    // Serial (LE u32)
    out.extend_from_slice(&page.header.serial.to_le_bytes());
    // Sequence number (LE u32)
    out.extend_from_slice(&page.header.seq_num.to_le_bytes());
    // Checksum placeholder (4 zero bytes)
    let csum_offset = out.len();
    out.extend_from_slice(&[0u8; 4]);
    // Segment count
    out.push(page.header.segment_count);
    // Segment table
    out.extend_from_slice(&page.segment_table);
    // Payload
    out.extend_from_slice(&page.page_data);

    // Compute CRC-32 over entire page with checksum bytes zeroed (already zero).
    let crc_engine = OggCrc32::new();
    let checksum = crc_engine.compute(&out);
    // Write the computed checksum.
    out[csum_offset..csum_offset + 4].copy_from_slice(&checksum.to_le_bytes());

    out
}

// ─── Builder ─────────────────────────────────────────────────────────────────

/// Incremental OGG page builder.
///
/// Accumulates packet data and produces [`OggPage`] values whose segment
/// tables comply with RFC 3533 lacing rules.  A page holds at most 255
/// segments × 255 bytes = 65 025 bytes of payload.
pub struct OggPageBuilder {
    serial: u32,
    seq_num: u32,
    granule_pos: i64,
    header_type: u8,
    /// Segments accumulated so far.
    segments: Vec<u8>,
    /// Payload bytes accumulated so far.
    payload: Vec<u8>,
}

impl OggPageBuilder {
    /// Create a new builder for logical bitstream `serial`.
    #[must_use]
    pub fn new(serial: u32) -> Self {
        Self {
            serial,
            seq_num: 0,
            granule_pos: -1,
            header_type: 0,
            segments: Vec::new(),
            payload: Vec::new(),
        }
    }

    /// Set the granule position for the *next* page to be emitted.
    pub fn set_granule_pos(&mut self, gp: i64) {
        self.granule_pos = gp;
    }

    /// Set header type flags (BOS / EOS / continuation) for the next page.
    pub fn set_header_type(&mut self, flags: u8) {
        self.header_type = flags;
    }

    /// Append a complete packet, splitting into lace segments per RFC 3533.
    ///
    /// Returns an error if adding the packet would exceed the maximum page
    /// capacity (255 segments).
    pub fn append_packet(&mut self, data: &[u8]) -> Result<(), OggError> {
        // Build lacing values for this packet.
        let mut remaining = data.len();
        let mut laces: Vec<u8> = Vec::new();
        loop {
            if remaining >= 255 {
                laces.push(255);
                remaining -= 255;
            } else {
                laces.push(remaining as u8);
                break;
            }
        }

        let free_segs = 255usize.saturating_sub(self.segments.len());
        if laces.len() > free_segs {
            return Err(OggError::DataOverflow);
        }

        self.segments.extend_from_slice(&laces);
        self.payload.extend_from_slice(data);
        Ok(())
    }

    /// Flush accumulated data into an [`OggPage`] and return it.
    ///
    /// Returns `None` if no data has been accumulated.
    pub fn flush(&mut self) -> Option<OggPage> {
        if self.segments.is_empty() {
            return None;
        }
        let segment_count = self.segments.len() as u8;
        let header = OggPageHeader {
            version: 0,
            header_type: self.header_type,
            granule_pos: self.granule_pos,
            serial: self.serial,
            seq_num: self.seq_num,
            checksum: 0, // will be filled by serialize
            segment_count,
        };
        let page = OggPage {
            header,
            segment_table: std::mem::take(&mut self.segments),
            page_data: std::mem::take(&mut self.payload),
        };
        self.seq_num = self.seq_num.wrapping_add(1);
        self.header_type = 0; // reset flags (BOS/EOS are one-shot)
        Some(page)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid OGG page and round-trip it through serialize → parse.
    fn make_test_page(serial: u32, seq: u32, data: &[u8], flags: u8) -> OggPage {
        let segment_table = vec![data.len() as u8];
        let header = OggPageHeader {
            version: 0,
            header_type: flags,
            granule_pos: 0,
            serial,
            seq_num: seq,
            checksum: 0,
            segment_count: 1,
        };
        OggPage {
            header,
            segment_table,
            page_data: data.to_vec(),
        }
    }

    #[test]
    fn test_crc32_known_vector() {
        // OGG CRC-32 uses poly=0x04c11db7, init=0, non-reflected, no final XOR.
        // Known value for b"123456789" under this algorithm: 0x89A1897F.
        let engine = OggCrc32::new();
        let digest = engine.compute(b"123456789");
        assert_eq!(digest, 0x89A1_897F);
    }

    #[test]
    fn test_crc32_empty_is_zero() {
        let engine = OggCrc32::new();
        assert_eq!(engine.compute(&[]), 0);
    }

    #[test]
    fn test_serialize_produces_oggs_magic() {
        let page = make_test_page(1, 0, b"hello", HEADER_TYPE_BOS);
        let bytes = serialize_ogg_page(&page);
        assert_eq!(&bytes[..4], b"OggS");
    }

    #[test]
    fn test_round_trip_simple() {
        let page = make_test_page(42, 7, b"world", 0);
        let bytes = serialize_ogg_page(&page);
        let (parsed, consumed) = parse_ogg_page(&bytes).expect("parse failed");
        assert_eq!(consumed, bytes.len());
        assert_eq!(parsed.header.serial, 42);
        assert_eq!(parsed.header.seq_num, 7);
        assert_eq!(parsed.page_data, b"world");
        assert_eq!(parsed.header.header_type, 0);
    }

    #[test]
    fn test_round_trip_bos_eos() {
        let page = make_test_page(99, 0, b"bos", HEADER_TYPE_BOS);
        let bytes = serialize_ogg_page(&page);
        let (parsed, _) = parse_ogg_page(&bytes).expect("parse failed");
        assert_eq!(parsed.page_type(), OggPageType::Bos);

        let eos_page = make_test_page(99, 1, b"eos", HEADER_TYPE_EOS);
        let eos_bytes = serialize_ogg_page(&eos_page);
        let (parsed_eos, _) = parse_ogg_page(&eos_bytes).expect("parse failed");
        assert_eq!(parsed_eos.page_type(), OggPageType::Eos);
    }

    #[test]
    fn test_checksum_mismatch_detected() {
        let page = make_test_page(1, 0, b"test", 0);
        let mut bytes = serialize_ogg_page(&page);
        // Corrupt checksum byte.
        bytes[22] ^= 0xFF;
        let result = parse_ogg_page(&bytes);
        assert!(matches!(result, Err(OggError::ChecksumMismatch { .. })));
    }

    #[test]
    fn test_invalid_magic_detected() {
        let page = make_test_page(1, 0, b"data", 0);
        let mut bytes = serialize_ogg_page(&page);
        bytes[0] = b'X';
        let result = parse_ogg_page(&bytes);
        assert!(matches!(result, Err(OggError::InvalidMagic(_))));
    }

    #[test]
    fn test_buffer_too_short() {
        let result = parse_ogg_page(&[0u8; 10]);
        assert!(matches!(result, Err(OggError::BufferTooShort { .. })));
    }

    #[test]
    fn test_page_type_derivation() {
        assert_eq!(OggPageType::from_flags(HEADER_TYPE_BOS), OggPageType::Bos);
        assert_eq!(OggPageType::from_flags(HEADER_TYPE_EOS), OggPageType::Eos);
        assert_eq!(
            OggPageType::from_flags(HEADER_TYPE_CONTINUATION),
            OggPageType::Continuation
        );
        assert_eq!(OggPageType::from_flags(0), OggPageType::Normal);
        // BOS wins over EOS when both set.
        assert_eq!(
            OggPageType::from_flags(HEADER_TYPE_BOS | HEADER_TYPE_EOS),
            OggPageType::Bos
        );
    }

    #[test]
    fn test_packet_iterator_single() {
        let data = b"hello world";
        let page = make_test_page(1, 0, data, 0);
        let packets: Vec<&[u8]> = page.packets().collect();
        assert_eq!(packets.len(), 1);
        assert_eq!(packets[0], data);
    }

    #[test]
    fn test_packet_iterator_multiple() {
        // Two packets: 3 bytes + 4 bytes.
        let segment_table = vec![3, 4];
        let page_data = b"abcdefg".to_vec();
        let header = OggPageHeader {
            version: 0,
            header_type: 0,
            granule_pos: 0,
            serial: 1,
            seq_num: 0,
            checksum: 0,
            segment_count: 2,
        };
        let page = OggPage {
            header,
            segment_table,
            page_data,
        };
        let packets: Vec<&[u8]> = page.packets().collect();
        assert_eq!(packets.len(), 2);
        assert_eq!(packets[0], b"abc");
        assert_eq!(packets[1], b"defg");
    }

    #[test]
    fn test_builder_flush_produces_valid_page() {
        let mut builder = OggPageBuilder::new(0x5678);
        builder.set_header_type(HEADER_TYPE_BOS);
        builder.set_granule_pos(0);
        builder.append_packet(b"opus head").expect("append failed");
        let page = builder.flush().expect("flush returned None");

        // Round-trip through serialize / parse.
        let bytes = serialize_ogg_page(&page);
        let (parsed, _) = parse_ogg_page(&bytes).expect("parse failed");
        assert_eq!(parsed.header.serial, 0x5678);
        assert_eq!(parsed.page_type(), OggPageType::Bos);
        let pkts: Vec<&[u8]> = parsed.packets().collect();
        assert_eq!(pkts[0], b"opus head");
    }

    #[test]
    fn test_payload_len() {
        let page = make_test_page(1, 0, b"abcde", 0);
        assert_eq!(page.payload_len(), 5);
    }

    #[test]
    fn test_granule_pos_negative_one() {
        // granule_pos = -1 (i.e. 0xFFFFFFFFFFFFFFFF) is the "unset" sentinel.
        let mut page = make_test_page(1, 0, b"x", 0);
        page.header.granule_pos = -1;
        let bytes = serialize_ogg_page(&page);
        let (parsed, _) = parse_ogg_page(&bytes).expect("parse failed");
        assert_eq!(parsed.header.granule_pos, -1i64);
    }
}
