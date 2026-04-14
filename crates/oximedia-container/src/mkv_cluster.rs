#![allow(dead_code)]
//! Matroska cluster parsing.
//!
//! This module provides parsing of Matroska/WebM cluster structures,
//! including cluster headers, `SimpleBlock` and `BlockGroup` elements,
//! and the four lacing schemes (None, Xiph, EBML, Fixed-size).
//!
//! # Matroska Cluster Structure
//!
//! A Matroska file is organised into *clusters*, each containing a
//! base timecode and a sequence of *block* elements:
//!
//! ```text
//! Cluster (1F 43 B6 75)
//! ├─ Timecode      (E7) — cluster base timecode in ms
//! ├─ Position      (A7) — byte position of this cluster
//! ├─ PrevSize      (AB) — size of the preceding cluster
//! ├─ SimpleBlock   (A3) — a single-frame (or laced) block
//! └─ BlockGroup    (A0)
//!    ├─ Block          (A1)
//!    ├─ BlockDuration  (9B)
//!    ├─ ReferenceBlock (FB)
//!    └─ BlockAdditions (75 A1)
//! ```
//!
//! # Lacing
//!
//! Blocks may carry multiple audio frames via *lacing*.  Four modes exist:
//!
//! | bits 5-6 | Mode      |
//! |----------|-----------|
//! | `00`     | No lacing |
//! | `01`     | Fixed-size|
//! | `10`     | Xiph      |
//! | `11`     | EBML      |
//!
//! # Example
//!
//! ```rust
//! use oximedia_container::mkv_cluster::{parse_simple_block, parse_cluster_header};
//!
//! // A minimal SimpleBlock: track 1, timecode 0, no lacing, 4 bytes of data
//! let raw: &[u8] = &[0x81, 0x00, 0x00, 0x00, 0xDE, 0xAD, 0xBE, 0xEF];
//! let block = parse_simple_block(raw).unwrap();
//! assert_eq!(block.track_number, 1);
//! assert_eq!(block.timecode, 0);
//! ```

use bitflags::bitflags;
use thiserror::Error;

// ── EBML element IDs used inside Cluster ──────────────────────────────────────

/// EBML element ID: Cluster Timecode (milliseconds).
pub const EBML_ID_TIMECODE: u32 = 0xE7;
/// EBML element ID: Cluster byte position.
pub const EBML_ID_POSITION: u32 = 0xA7;
/// EBML element ID: Size of the previous cluster.
pub const EBML_ID_PREV_SIZE: u32 = 0xAB;
/// EBML element ID: `SimpleBlock`.
pub const EBML_ID_SIMPLE_BLOCK: u32 = 0xA3;
/// EBML element ID: `BlockGroup`.
pub const EBML_ID_BLOCK_GROUP: u32 = 0xA0;
/// EBML element ID: `Block` inside a `BlockGroup`.
pub const EBML_ID_BLOCK: u32 = 0xA1;
/// EBML element ID: `BlockDuration`.
pub const EBML_ID_BLOCK_DURATION: u32 = 0x9B;
/// EBML element ID: `ReferenceBlock`.
pub const EBML_ID_REFERENCE_BLOCK: u32 = 0xFB;
/// EBML element ID: `BlockAdditions`.
pub const EBML_ID_BLOCK_ADDITIONS: u32 = 0x75A1;

// ── Special vint sentinels ─────────────────────────────────────────────────────

/// The "unknown size" sentinel value for a 1-byte vint (all data bits set).
const VINT_UNKNOWN_1: u64 = 0x7F;
/// The "unknown size" sentinel for a 2-byte vint.
const VINT_UNKNOWN_2: u64 = 0x3FFF;
/// The "unknown size" sentinel for a 3-byte vint.
const VINT_UNKNOWN_3: u64 = 0x1F_FFFF;
/// The "unknown size" sentinel for a 4-byte vint.
const VINT_UNKNOWN_4: u64 = 0x0FFF_FFFF;
/// The "unknown size" sentinel for a 5-byte vint.
const VINT_UNKNOWN_5: u64 = 0x07_FFFF_FFFF;
/// The "unknown size" sentinel for a 6-byte vint.
const VINT_UNKNOWN_6: u64 = 0x03_FFFF_FFFF_FFFF;
/// The "unknown size" sentinel for a 7-byte vint.
const VINT_UNKNOWN_7: u64 = 0x01_FFFF_FFFF_FFFF_FF;
/// The "unknown size" sentinel for an 8-byte vint (all data bits set).
const VINT_UNKNOWN_8: u64 = 0x00FF_FFFF_FFFF_FFFF;

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors that can occur while parsing Matroska cluster data.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ClusterError {
    /// The input ended before enough bytes were available.
    #[error("unexpected end of data (need {need} bytes, have {have})")]
    UnexpectedEof {
        /// Bytes required.
        need: usize,
        /// Bytes available.
        have: usize,
    },

    /// A variable-length integer (vint) started with the byte `0x00`,
    /// which would indicate a 9-byte width — not defined in the spec.
    #[error("invalid EBML vint: leading byte is 0x00")]
    InvalidVint,

    /// The vint encoded a zero track number, which Matroska forbids.
    #[error("block track number must be ≥ 1")]
    TrackNumberZero,

    /// A lace header was malformed (e.g., Xiph sum exceeded total data).
    #[error("invalid lace data: {reason}")]
    InvalidLace {
        /// Human-readable reason.
        reason: &'static str,
    },

    /// Fixed-size lacing requires that the data divides evenly.
    #[error("fixed lace: data length {data_len} is not divisible by frame count {frame_count}")]
    FixedLaceNotDivisible {
        /// Total data bytes.
        data_len: usize,
        /// Number of frames.
        frame_count: usize,
    },

    /// Absolute timecode computation overflowed `u64`.
    #[error("timecode overflow: cluster={cluster}, block={block}")]
    TimecodeOverflow {
        /// Cluster base timecode.
        cluster: u64,
        /// Block relative timecode (may be negative).
        block: i16,
    },

    /// An EBML element ID did not match any recognised cluster sub-element.
    #[error("unknown cluster element ID: 0x{id:X}")]
    UnknownElementId {
        /// The unrecognised ID.
        id: u32,
    },

    /// An element's data-size vint could not be parsed.
    #[error("malformed element size vint at offset {offset}")]
    MalformedSize {
        /// Byte offset where the size field was expected.
        offset: usize,
    },

    /// Lace frame count byte was 0, which implies at least 1 frame — but a
    /// count of 0 would mean the header is corrupt.
    #[error("lace frame count must be ≥ 1")]
    ZeroFrameCount,
}

// ── Lace type ─────────────────────────────────────────────────────────────────

/// Lacing mode embedded in a Matroska block flags byte.
///
/// Lacing allows multiple small frames (typically audio) to be bundled
/// inside a single block, reducing per-frame overhead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LaceType {
    /// No lacing — the block holds exactly one frame.
    None,
    /// Xiph lacing — frame sizes encoded as sequences of bytes summed until
    /// a byte < 255 is encountered; last frame size is implicit.
    Xiph,
    /// Fixed-size lacing — all frames have equal size (total / count).
    Fixed,
    /// EBML lacing — first frame size is an EBML vint; subsequent sizes are
    /// signed-vint deltas relative to the previous frame size; last size
    /// is implicit.
    Ebml,
}

impl LaceType {
    /// Decode the lace type from bits 5-6 of the block flags byte.
    ///
    /// Per the Matroska spec the mapping is:
    ///
    /// | bits 6-5 | Lace type  |
    /// |----------|------------|
    /// | `0 0`    | None       |
    /// | `0 1`    | Fixed-size |
    /// | `1 0`    | Xiph       |
    /// | `1 1`    | EBML       |
    #[must_use]
    pub const fn from_flags_byte(flags: u8) -> Self {
        match (flags >> 1) & 0b11 {
            0b00 => Self::None,
            0b01 => Self::Fixed,
            0b10 => Self::Xiph,
            _ => Self::Ebml, // 0b11
        }
    }
}

// ── Block flags ───────────────────────────────────────────────────────────────

bitflags! {
    /// Flags stored in the third byte of a Matroska block header.
    ///
    /// Bit layout (MSB to LSB):
    ///
    /// ```text
    /// 7   6   5   4   3   2   1   0
    /// KF  0   L1  L0  INV 0   0   DC
    /// ```
    ///
    /// * `KF`  — keyframe (only meaningful in `SimpleBlock`)
    /// * `L1,L0` — lacing type (see [`LaceType`])
    /// * `INV` — invisible (should not be rendered)
    /// * `DC`  — discardable (may be dropped during seeking)
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
    pub struct BlockFlags: u8 {
        /// Block is a keyframe (can be decoded without prior reference frames).
        const KEYFRAME    = 0b1000_0000;
        /// Block should not be rendered (invisible).
        const INVISIBLE   = 0b0000_1000;
        /// Block may be discarded during playback without visible artefacts.
        const DISCARDABLE = 0b0000_0001;
    }
}

// ── SimpleBlock ───────────────────────────────────────────────────────────────

/// A parsed Matroska `SimpleBlock` element.
///
/// A `SimpleBlock` encodes one or more frames for a single track.  The
/// `frame_sizes` field lists the byte length of each laced frame; if lacing
/// is [`LaceType::None`] it contains exactly one entry equal to `data_len`.
///
/// `data_offset` points to the first payload byte *relative to the start of
/// the input slice* passed to [`parse_simple_block`], so callers can slice
/// the original byte buffer to extract frame data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimpleBlock {
    /// Track number (1-based) that this block belongs to.
    pub track_number: u64,
    /// Timecode relative to the containing cluster's base timecode (ms).
    pub timecode: i16,
    /// Parsed block flags.
    pub flags: BlockFlags,
    /// Lacing mode.
    pub lace_type: LaceType,
    /// Byte length of each laced frame.  For no-lacing blocks this has
    /// exactly one entry.
    pub frame_sizes: Vec<usize>,
    /// Byte offset of the first frame payload within the original input.
    pub data_offset: usize,
    /// Total byte length of all frame payloads combined.
    pub data_len: usize,
}

impl SimpleBlock {
    /// Computes the absolute timecode by adding the cluster base and this
    /// block's relative timecode.
    ///
    /// The relative timecode is a signed 16-bit integer — a negative value
    /// indicates a block that precedes the cluster's nominal start.
    ///
    /// Returns [`ClusterError::TimecodeOverflow`] if the arithmetic overflows
    /// `u64`.
    ///
    /// # Errors
    ///
    /// Returns an error if `cluster_timecode + block_timecode` would overflow
    /// `u64` or if the result would be negative (cluster < |block_timecode|
    /// for a negative block_timecode).
    pub fn absolute_timecode(&self, cluster_timecode: u64) -> Result<u64, ClusterError> {
        compute_absolute_timecode(cluster_timecode, self.timecode)
    }
}

// ── BlockGroup ────────────────────────────────────────────────────────────────

/// An entry within a `BlockGroup` beyond the mandatory `Block` element.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockGroupEntry {
    /// A reference to another block (negative = backward, positive = forward).
    ReferenceBlock(i64),
    /// Duration of the block in track timescale units.
    BlockDuration(u64),
    /// Raw `BlockAdditions` payload.
    BlockAdditions(Vec<u8>),
}

/// A parsed Matroska `BlockGroup` element.
///
/// `BlockGroup` wraps a `Block` (identical wire format to `SimpleBlock` except
/// the keyframe flag is not used) with optional duration, reference blocks
/// for B-frame support, and extension data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockGroup {
    /// The wrapped block.
    pub block: SimpleBlock,
    /// Optional explicit duration in track timescale units.
    pub duration: Option<u64>,
    /// Forward/backward reference delta values.
    pub reference_blocks: Vec<i64>,
    /// Raw `BlockAdditions` payload, if present.
    pub additions: Vec<u8>,
}

// ── ClusterHeader ─────────────────────────────────────────────────────────────

/// Parsed header fields of a Matroska `Cluster` element.
///
/// The cluster header contains the base timecode for all blocks within the
/// cluster, plus optional positioning hints used for seeking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusterHeader {
    /// Base timecode for this cluster in milliseconds (default timescale).
    ///
    /// All block timecodes within the cluster are relative to this value.
    pub timecode: u64,
    /// Absolute byte position of this cluster within the segment, if present.
    pub position: Option<u64>,
    /// Size of the preceding cluster in bytes, if present.
    pub prev_size: Option<u64>,
}

// ── ClusterBlock ─────────────────────────────────────────────────────────────

/// A block element found within a Matroska cluster — either a `SimpleBlock`
/// or a `BlockGroup`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClusterBlock {
    /// A standalone `SimpleBlock`.
    Simple(SimpleBlock),
    /// A `BlockGroup` with optional duration and reference information.
    Group(BlockGroup),
}

impl ClusterBlock {
    /// Returns the inner [`SimpleBlock`] regardless of variant.
    #[must_use]
    pub fn block(&self) -> &SimpleBlock {
        match self {
            Self::Simple(b) => b,
            Self::Group(g) => &g.block,
        }
    }

    /// Computes the absolute timecode for this block.
    ///
    /// # Errors
    ///
    /// Returns [`ClusterError::TimecodeOverflow`] if the addition overflows.
    pub fn absolute_timecode(&self, cluster_timecode: u64) -> Result<u64, ClusterError> {
        self.block().absolute_timecode(cluster_timecode)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Core parsing functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Parse an EBML variable-length unsigned integer (vint).
///
/// EBML vints use a unary prefix to indicate their width:
///
/// | Leading byte mask | Width | Data bits |
/// |-------------------|-------|-----------|
/// | `1xxx xxxx`       | 1     | 7         |
/// | `01xx xxxx`       | 2     | 14        |
/// | `001x xxxx`       | 3     | 21        |
/// | …                 | …     | …         |
/// | `0000 0001`       | 8     | 56        |
///
/// The width marker bit is *stripped* from the returned value.
///
/// Returns `(value, bytes_consumed)` on success.
///
/// # Errors
///
/// * [`ClusterError::UnexpectedEof`] — fewer bytes than the encoded width.
/// * [`ClusterError::InvalidVint`]   — leading byte is `0x00` (9-byte width,
///   undefined by the spec).
pub fn parse_vint(data: &[u8]) -> Result<(u64, usize), ClusterError> {
    let first = data
        .first()
        .copied()
        .ok_or(ClusterError::UnexpectedEof { need: 1, have: 0 })?;

    // Determine width from the position of the most-significant set bit.
    let width = first.leading_zeros() as usize + 1;

    if width > 8 {
        return Err(ClusterError::InvalidVint);
    }
    if data.len() < width {
        return Err(ClusterError::UnexpectedEof {
            need: width,
            have: data.len(),
        });
    }

    // Strip the width-marker bit from the first byte.
    let mask = 0xFF_u8 >> width;
    let mut value = u64::from(first & mask);

    for &byte in &data[1..width] {
        value = (value << 8) | u64::from(byte);
    }

    Ok((value, width))
}

/// Parse an EBML variable-length *signed* integer.
///
/// The signed variant is used for `ReferenceBlock` delta values and EBML
/// lace size deltas.  The value is computed by parsing an unsigned vint and
/// then subtracting the bias `2^(7·width − 1) − 1`.
///
/// Returns `(value, bytes_consumed)` on success.
///
/// # Errors
///
/// Propagates errors from [`parse_vint`].
pub fn parse_vint_signed(data: &[u8]) -> Result<(i64, usize), ClusterError> {
    let (raw, width) = parse_vint(data)?;

    // Bias = 2^(7*width - 1) - 1
    let bias: u64 = (1_u64 << (7 * width - 1)).saturating_sub(1);
    let signed = (raw as i64) - (bias as i64);

    Ok((signed, width))
}

/// Parse the block flags byte and return `(BlockFlags, LaceType)`.
///
/// The flags byte layout:
///
/// ```text
/// bit 7 : keyframe
/// bit 6 : reserved (0)
/// bit 5 : lace bit 1
/// bit 4 : lace bit 0
/// bit 3 : invisible
/// bit 2 : reserved (0)
/// bit 1 : reserved (0)
/// bit 0 : discardable
/// ```
///
/// The lace type is encoded in bits 5-4 (shifted right by 1 from bits 6-5
/// used in this module's `LaceType::from_flags_byte`).
///
/// **Note**: bits 6-5 as used by `LaceType::from_flags_byte` correspond to
/// the bit positions 5-4 of the byte (the Matroska spec labels bits 0-7 from
/// LSB to MSB, while Rust counts from MSB). This implementation follows the
/// Matroska spec's bit numbering where bit 0 is LSB.
#[must_use]
pub fn parse_block_flags(byte: u8) -> (BlockFlags, LaceType) {
    let mut flags = BlockFlags::empty();

    if byte & 0x80 != 0 {
        flags |= BlockFlags::KEYFRAME;
    }
    if byte & 0x08 != 0 {
        flags |= BlockFlags::INVISIBLE;
    }
    if byte & 0x01 != 0 {
        flags |= BlockFlags::DISCARDABLE;
    }

    // Lace type: bits 5-6 (i.e., bits 1-2 when counting from LSB=0),
    // which is (byte >> 1) & 0b11 in the Matroska specification.
    let lace_type = LaceType::from_flags_byte(byte);

    (flags, lace_type)
}

/// Decode frame sizes from a Xiph-laced block payload.
///
/// Xiph lacing encodes `frame_count - 1` frame sizes (the last is implicit).
/// Each size is the sum of a run of bytes; the run ends at the first byte < 255.
///
/// `data`        — the lace header bytes (starting immediately after the
///                 frame-count byte).
/// `frame_count` — total number of frames (including the implicit last one).
/// `total_data`  — total byte length of all frame payloads combined (used to
///                 derive the final frame size and validate the result).
///
/// Returns `(frame_sizes, header_bytes_consumed)`.
///
/// # Errors
///
/// * [`ClusterError::InvalidLace`] if the sizes exceed `total_data` or
///   the header is truncated.
pub fn decode_xiph_lace_sizes(
    data: &[u8],
    frame_count: usize,
    total_data: usize,
) -> Result<(Vec<usize>, usize), ClusterError> {
    if frame_count == 0 {
        return Err(ClusterError::ZeroFrameCount);
    }
    if frame_count == 1 {
        return Ok((vec![total_data], 0));
    }

    let mut sizes = Vec::with_capacity(frame_count);
    let mut pos = 0usize;
    let explicit_count = frame_count - 1;

    for _ in 0..explicit_count {
        let mut size: usize = 0;
        loop {
            let byte = data.get(pos).copied().ok_or(ClusterError::UnexpectedEof {
                need: pos + 1,
                have: data.len(),
            })?;
            pos += 1;
            size = size
                .checked_add(byte as usize)
                .ok_or(ClusterError::InvalidLace {
                    reason: "Xiph frame size overflowed usize",
                })?;
            if byte < 255 {
                break;
            }
        }
        sizes.push(size);
    }

    // Compute the implicit last frame size.
    let explicit_total: usize = sizes.iter().sum();
    let last_size = total_data
        .checked_sub(explicit_total)
        .ok_or(ClusterError::InvalidLace {
            reason: "Xiph explicit frame sizes exceed total data length",
        })?;
    sizes.push(last_size);

    Ok((sizes, pos))
}

/// Decode frame sizes from an EBML-laced block payload.
///
/// EBML lacing encodes `frame_count - 1` sizes; the last is implicit.
/// The first size is an unsigned EBML vint; subsequent sizes are *signed*
/// vint deltas relative to the previous frame size.
///
/// Returns `(frame_sizes, header_bytes_consumed)`.
///
/// # Errors
///
/// * [`ClusterError::InvalidLace`] if a delta would make a frame size negative
///   or if explicit sizes exceed `total_data`.
pub fn decode_ebml_lace_sizes(
    data: &[u8],
    frame_count: usize,
    total_data: usize,
) -> Result<(Vec<usize>, usize), ClusterError> {
    if frame_count == 0 {
        return Err(ClusterError::ZeroFrameCount);
    }
    if frame_count == 1 {
        return Ok((vec![total_data], 0));
    }

    let mut sizes = Vec::with_capacity(frame_count);
    let mut pos = 0usize;

    // First frame size: unsigned vint.
    let (first_size_raw, w) = parse_vint(&data[pos..]).map_err(|_| ClusterError::InvalidLace {
        reason: "EBML lace: cannot parse first frame size vint",
    })?;
    pos += w;
    let first_size = first_size_raw as usize;
    sizes.push(first_size);

    // Remaining explicit sizes: signed vint deltas.
    let mut prev_size = first_size as i64;
    for _ in 1..(frame_count - 1) {
        let (delta, w) =
            parse_vint_signed(&data[pos..]).map_err(|_| ClusterError::InvalidLace {
                reason: "EBML lace: cannot parse frame size delta vint",
            })?;
        pos += w;
        let cur_size = prev_size + delta;
        if cur_size < 0 {
            return Err(ClusterError::InvalidLace {
                reason: "EBML lace: frame size became negative after delta",
            });
        }
        sizes.push(cur_size as usize);
        prev_size = cur_size;
    }

    // Implicit last frame.
    let explicit_total: usize = sizes.iter().sum();
    let last_size = total_data
        .checked_sub(explicit_total)
        .ok_or(ClusterError::InvalidLace {
            reason: "EBML lace: explicit frame sizes exceed total data length",
        })?;
    sizes.push(last_size);

    Ok((sizes, pos))
}

/// Decode frame sizes for a fixed-size laced block.
///
/// All frames have equal size.  This function verifies that `total_data` is
/// evenly divisible by `frame_count`.
///
/// # Errors
///
/// * [`ClusterError::ZeroFrameCount`] — `frame_count` is 0.
/// * [`ClusterError::FixedLaceNotDivisible`] — `total_data % frame_count ≠ 0`.
pub fn decode_fixed_lace_sizes(
    frame_count: usize,
    total_data: usize,
) -> Result<Vec<usize>, ClusterError> {
    if frame_count == 0 {
        return Err(ClusterError::ZeroFrameCount);
    }
    if total_data % frame_count != 0 {
        return Err(ClusterError::FixedLaceNotDivisible {
            data_len: total_data,
            frame_count,
        });
    }
    let frame_size = total_data / frame_count;
    Ok(vec![frame_size; frame_count])
}

/// Parse a Matroska `SimpleBlock` (or bare `Block`) from a raw byte slice.
///
/// Wire format:
///
/// ```text
/// [vint: track_number] [i16 BE: timecode] [u8: flags]
/// [optional lace header] [frame payloads…]
/// ```
///
/// The returned [`SimpleBlock`] records the offset and sizes of the payload
/// data within the original slice so that callers can extract frames without
/// copying.
///
/// # Errors
///
/// Returns a [`ClusterError`] if the input is truncated or structurally
/// invalid.
pub fn parse_simple_block(data: &[u8]) -> Result<SimpleBlock, ClusterError> {
    let mut pos = 0usize;

    // 1. Track number (vint).
    let (track_number, w) = parse_vint(data)?;
    pos += w;
    if track_number == 0 {
        return Err(ClusterError::TrackNumberZero);
    }

    // 2. Timecode (signed 16-bit big-endian).
    if data.len() < pos + 2 {
        return Err(ClusterError::UnexpectedEof {
            need: pos + 2,
            have: data.len(),
        });
    }
    let timecode = i16::from_be_bytes([data[pos], data[pos + 1]]);
    pos += 2;

    // 3. Flags byte.
    let flags_byte = *data.get(pos).ok_or(ClusterError::UnexpectedEof {
        need: pos + 1,
        have: data.len(),
    })?;
    pos += 1;

    let (flags, lace_type) = parse_block_flags(flags_byte);

    // The payload starts here unless there is a lace header.
    let payload_total = data.len() - pos;

    let (frame_sizes, data_offset) = match lace_type {
        LaceType::None => (vec![payload_total], pos),
        LaceType::Xiph | LaceType::Ebml | LaceType::Fixed => {
            // Laced blocks: next byte is (frame_count - 1).
            let lace_count_byte = *data.get(pos).ok_or(ClusterError::UnexpectedEof {
                need: pos + 1,
                have: data.len(),
            })?;
            pos += 1;
            let frame_count = lace_count_byte as usize + 1;

            // Remaining bytes form the lace header followed by payloads.
            let lace_and_payload = &data[pos..];

            match lace_type {
                LaceType::Xiph => {
                    let (sizes, hdr_len) =
                        decode_xiph_lace_sizes(lace_and_payload, frame_count, payload_total - 1)?;
                    (sizes, pos + hdr_len)
                }
                LaceType::Ebml => {
                    let payload_bytes = payload_total - 1; // subtract the frame-count byte
                    let (sizes, hdr_len) =
                        decode_ebml_lace_sizes(lace_and_payload, frame_count, payload_bytes)?;
                    (sizes, pos + hdr_len)
                }
                LaceType::Fixed => {
                    // Fixed lacing has no extra header bytes.
                    let payload_bytes = payload_total - 1;
                    let sizes = decode_fixed_lace_sizes(frame_count, payload_bytes)?;
                    (sizes, pos)
                }
                LaceType::None => unreachable!(),
            }
        }
    };

    let data_len = data.len() - data_offset;

    Ok(SimpleBlock {
        track_number,
        timecode,
        flags,
        lace_type,
        frame_sizes,
        data_offset,
        data_len,
    })
}

// ── EBML element ID and size parsing helpers ──────────────────────────────────

/// Parse an EBML element ID from `data[pos..]`.
///
/// EBML element IDs follow the same vint width encoding as data sizes *except*
/// that the width-marker bit is **not** stripped — the full byte sequence
/// (including the marker) forms the canonical ID value.
///
/// Returns `(id_as_u32, bytes_consumed)`.
fn parse_element_id(data: &[u8], pos: usize) -> Result<(u32, usize), ClusterError> {
    let slice = data.get(pos..).ok_or(ClusterError::UnexpectedEof {
        need: pos + 1,
        have: data.len(),
    })?;

    let first = slice
        .first()
        .copied()
        .ok_or(ClusterError::UnexpectedEof { need: 1, have: 0 })?;

    let width = first.leading_zeros() as usize + 1;
    if width > 4 {
        // Cluster sub-elements use at most 3-byte IDs; 4 is a safe upper limit.
        return Err(ClusterError::MalformedSize { offset: pos });
    }
    if slice.len() < width {
        return Err(ClusterError::UnexpectedEof {
            need: pos + width,
            have: data.len(),
        });
    }

    // The ID includes the marker bit — accumulate all bytes as-is.
    let mut id = u32::from(first);
    for &b in &slice[1..width] {
        id = (id << 8) | u32::from(b);
    }

    Ok((id, width))
}

/// Parse an EBML data-size vint from `data[pos..]`.
///
/// This is the standard vint format (marker bit *stripped*).
/// Returns `(size_in_bytes, bytes_consumed)`.
fn parse_element_size(data: &[u8], pos: usize) -> Result<(u64, usize), ClusterError> {
    let slice = data.get(pos..).ok_or(ClusterError::UnexpectedEof {
        need: pos + 1,
        have: data.len(),
    })?;
    parse_vint(slice).map_err(|_| ClusterError::MalformedSize { offset: pos })
}

/// Read a big-endian unsigned integer of `width` bytes from `data`.
fn read_uint_be(data: &[u8], width: usize) -> Result<u64, ClusterError> {
    if data.len() < width {
        return Err(ClusterError::UnexpectedEof {
            need: width,
            have: data.len(),
        });
    }
    let mut v = 0u64;
    for &b in &data[..width] {
        v = (v << 8) | u64::from(b);
    }
    Ok(v)
}

/// Read a big-endian signed integer of `width` bytes from `data`.
fn read_sint_be(data: &[u8], width: usize) -> Result<i64, ClusterError> {
    let raw = read_uint_be(data, width)?;
    // Sign-extend from `width` bytes.
    let shift = (8 - width) * 8;
    Ok(((raw << shift) as i64) >> shift)
}

// ── Cluster header parsing ────────────────────────────────────────────────────

/// Parse the sub-elements of a Matroska `Cluster` element body.
///
/// `data` should be the raw bytes of the Cluster body (i.e., everything
/// *after* the Cluster element's own ID and size fields).
///
/// The parser scans for `Timecode` (0xE7), `Position` (0xA7), and `PrevSize`
/// (0xAB) elements and stops as soon as all three are found or the data is
/// exhausted.
///
/// Unknown elements are skipped using their encoded size.
///
/// # Errors
///
/// Returns a [`ClusterError`] if required fields are absent or the data is
/// structurally corrupt.  A missing `Timecode` element (which is mandatory
/// in Matroska) causes [`ClusterError::MalformedSize`] with `offset = 0`.
pub fn parse_cluster_header(data: &[u8]) -> Result<ClusterHeader, ClusterError> {
    let mut timecode: Option<u64> = None;
    let mut position: Option<u64> = None;
    let mut prev_size: Option<u64> = None;

    let mut pos = 0usize;

    while pos < data.len() {
        // Parse element ID.
        let (id, id_len) = parse_element_id(data, pos)?;
        pos += id_len;

        // Parse element size.
        let (elem_size, size_len) = parse_element_size(data, pos)?;
        pos += size_len;

        let elem_size = elem_size as usize;

        // Guard against reading beyond the buffer.
        if pos + elem_size > data.len() {
            return Err(ClusterError::UnexpectedEof {
                need: pos + elem_size,
                have: data.len(),
            });
        }

        let elem_data = &data[pos..pos + elem_size];

        match id {
            id if id == EBML_ID_TIMECODE => {
                timecode = Some(read_uint_be(elem_data, elem_size)?);
            }
            id if id == EBML_ID_POSITION => {
                position = Some(read_uint_be(elem_data, elem_size)?);
            }
            id if id == EBML_ID_PREV_SIZE => {
                prev_size = Some(read_uint_be(elem_data, elem_size)?);
            }
            id if id == EBML_ID_SIMPLE_BLOCK
                || id == EBML_ID_BLOCK_GROUP
                || id == EBML_ID_BLOCK =>
            {
                // Block elements signal the end of the header region.
                break;
            }
            _ => {
                // Unknown or unneeded element — skip.
            }
        }

        pos += elem_size;

        // Short-circuit once all header fields are present.
        if timecode.is_some() && position.is_some() && prev_size.is_some() {
            break;
        }
    }

    let timecode = timecode.ok_or(ClusterError::MalformedSize { offset: 0 })?;

    Ok(ClusterHeader {
        timecode,
        position,
        prev_size,
    })
}

// ── Timecode helpers ──────────────────────────────────────────────────────────

/// Compute the absolute timecode `cluster_timecode + block_timecode`.
///
/// The block timecode is a signed 16-bit integer (it can be negative).
/// If `block_timecode` is negative and its absolute value exceeds
/// `cluster_timecode`, or if the addition would overflow `u64`, the function
/// returns [`ClusterError::TimecodeOverflow`].
pub fn compute_absolute_timecode(
    cluster_timecode: u64,
    block_timecode: i16,
) -> Result<u64, ClusterError> {
    if block_timecode >= 0 {
        cluster_timecode
            .checked_add(block_timecode as u64)
            .ok_or(ClusterError::TimecodeOverflow {
                cluster: cluster_timecode,
                block: block_timecode,
            })
    } else {
        let abs_delta = block_timecode.unsigned_abs() as u64;
        cluster_timecode
            .checked_sub(abs_delta)
            .ok_or(ClusterError::TimecodeOverflow {
                cluster: cluster_timecode,
                block: block_timecode,
            })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════════════
#[cfg(test)]
mod tests {
    use super::*;

    // ── 1. parse_vint — 1-byte value ─────────────────────────────────────────

    #[test]
    fn test_parse_vint_1byte() {
        // 0x85 = 1000_0101: width=1, marker stripped → value = 0x05
        let (val, width) = parse_vint(&[0x85]).unwrap();
        assert_eq!(width, 1);
        assert_eq!(val, 5);
    }

    // ── 2. parse_vint — 2-byte value ─────────────────────────────────────────

    #[test]
    fn test_parse_vint_2byte() {
        // 0x40 0x05: leading byte 0x40 = 0100_0000 → width=2,
        // marker stripped from first byte: 0x40 & 0x3F = 0x00, then 0x05
        // value = (0x00 << 8) | 0x05 = 5
        let (val, width) = parse_vint(&[0x40, 0x05]).unwrap();
        assert_eq!(width, 2);
        assert_eq!(val, 5);
    }

    // ── 3. parse_vint — maximum 1-byte (unknown size marker) ─────────────────

    #[test]
    fn test_parse_vint_max_1byte() {
        // 0xFF = 1111_1111: width=1, marker stripped → value = 0x7F
        // This is the "unknown size" sentinel for 1-byte vints.
        let (val, width) = parse_vint(&[0xFF]).unwrap();
        assert_eq!(width, 1);
        assert_eq!(val, VINT_UNKNOWN_1);
    }

    // ── 4. parse_block_flags — keyframe, no lacing ───────────────────────────

    #[test]
    fn test_parse_block_flags_keyframe_no_lace() {
        // 0x80 = 1000_0000: keyframe set, lace=00 (none), invisible=0, discardable=0
        let (flags, lace) = parse_block_flags(0x80);
        assert!(flags.contains(BlockFlags::KEYFRAME));
        assert!(!flags.contains(BlockFlags::INVISIBLE));
        assert!(!flags.contains(BlockFlags::DISCARDABLE));
        assert_eq!(lace, LaceType::None);
    }

    // ── 5. parse_block_flags — xiph lacing + discardable ─────────────────────

    #[test]
    fn test_parse_block_flags_xiph_discardable() {
        // bits: KF=0, L1=1, L0=0, INV=0 → 0b0000_0100 | 0b0000_0001 = 0x05
        // Wait — lace bits 5-4 in the byte. For Xiph: (byte >> 1) & 0b11 = 0b10
        // So bits 2-1 = 10 → byte contributes 0b0000_0100 = 0x04 for lace.
        // Plus discardable bit 0: 0x01. No keyframe. → 0x05
        let (flags, lace) = parse_block_flags(0x05);
        assert!(!flags.contains(BlockFlags::KEYFRAME));
        assert!(flags.contains(BlockFlags::DISCARDABLE));
        assert_eq!(lace, LaceType::Xiph);
    }

    // ── 6. decode_xiph_lace_sizes — 3 frames ─────────────────────────────────

    #[test]
    fn test_decode_xiph_lace_sizes_three_frames() {
        // 3 frames: sizes are 100, 200, implicit = 400 - 100 - 200 = 100.
        // Xiph encoding of 100: single byte 100 (< 255).
        // Xiph encoding of 200: single byte 200 (< 255).
        let header = &[100u8, 200u8];
        let total_data = 400usize;
        let (sizes, hdr_consumed) = decode_xiph_lace_sizes(header, 3, total_data).unwrap();
        assert_eq!(hdr_consumed, 2); // 2 bytes consumed for the header
        assert_eq!(sizes, vec![100, 200, 100]);
    }

    // ── 7. decode_fixed_lace_sizes — success and error ───────────────────────

    #[test]
    fn test_decode_fixed_lace_sizes_success() {
        let sizes = decode_fixed_lace_sizes(4, 400).unwrap();
        assert_eq!(sizes, vec![100, 100, 100, 100]);
    }

    #[test]
    fn test_decode_fixed_lace_sizes_not_divisible() {
        let err = decode_fixed_lace_sizes(3, 100).unwrap_err();
        assert!(matches!(
            err,
            ClusterError::FixedLaceNotDivisible {
                data_len: 100,
                frame_count: 3
            }
        ));
    }

    #[test]
    fn test_decode_fixed_lace_sizes_zero_count() {
        let err = decode_fixed_lace_sizes(0, 100).unwrap_err();
        assert_eq!(err, ClusterError::ZeroFrameCount);
    }

    // ── 8. parse_simple_block — minimal no-lace block ────────────────────────

    #[test]
    fn test_parse_simple_block_no_lace() {
        // Track 1 (vint: 0x81), timecode 0 (0x00 0x00), flags 0x80 (keyframe, no lace),
        // then 4 bytes of payload.
        let raw: &[u8] = &[0x81, 0x00, 0x00, 0x80, 0xDE, 0xAD, 0xBE, 0xEF];
        let block = parse_simple_block(raw).unwrap();
        assert_eq!(block.track_number, 1);
        assert_eq!(block.timecode, 0);
        assert!(block.flags.contains(BlockFlags::KEYFRAME));
        assert_eq!(block.lace_type, LaceType::None);
        assert_eq!(block.frame_sizes, vec![4]);
        assert_eq!(block.data_offset, 4); // 1 (vint) + 2 (tc) + 1 (flags)
        assert_eq!(block.data_len, 4);
    }

    // ── 9. absolute_timecode — overflow detection ─────────────────────────────

    #[test]
    fn test_absolute_timecode_overflow() {
        let err = compute_absolute_timecode(u64::MAX, 1).unwrap_err();
        assert!(matches!(err, ClusterError::TimecodeOverflow { .. }));
    }

    #[test]
    fn test_absolute_timecode_underflow() {
        // cluster=5, block=-10 → 5 - 10 would underflow u64
        let err = compute_absolute_timecode(5, -10).unwrap_err();
        assert!(matches!(err, ClusterError::TimecodeOverflow { .. }));
    }

    #[test]
    fn test_absolute_timecode_ok_positive() {
        let abs = compute_absolute_timecode(1000, 42).unwrap();
        assert_eq!(abs, 1042);
    }

    #[test]
    fn test_absolute_timecode_ok_negative() {
        // cluster=1000, block=-100 → 900
        let abs = compute_absolute_timecode(1000, -100).unwrap();
        assert_eq!(abs, 900);
    }

    // ── 10. parse_cluster_header — with Timecode sub-element ─────────────────

    #[test]
    fn test_parse_cluster_header_timecode_only() {
        // Craft a minimal cluster body with only a Timecode element.
        // Timecode element ID: 0xE7 (1-byte ID with marker bit set)
        // Data size: 0x82 (vint: width=1, value=2) → 2-byte payload
        // Payload: 0x00 0x64 = 100 in big-endian u16
        let data: &[u8] = &[0xE7, 0x82, 0x00, 0x64];
        let header = parse_cluster_header(data).unwrap();
        assert_eq!(header.timecode, 100);
        assert!(header.position.is_none());
        assert!(header.prev_size.is_none());
    }

    // ── 11. parse_cluster_header — Timecode + Position ───────────────────────

    #[test]
    fn test_parse_cluster_header_with_position() {
        // Timecode = 200 (0x00C8), Position = 4096 (0x1000)
        // Timecode: ID=0xE7, size=0x82 (2 bytes), data=[0x00, 0xC8]
        // Position: ID=0xA7, size=0x82 (2 bytes), data=[0x10, 0x00]
        let data: &[u8] = &[
            0xE7, 0x82, 0x00, 0xC8, // Timecode = 200
            0xA7, 0x82, 0x10, 0x00, // Position = 4096
        ];
        let header = parse_cluster_header(data).unwrap();
        assert_eq!(header.timecode, 200);
        assert_eq!(header.position, Some(4096));
        assert!(header.prev_size.is_none());
    }

    // ── 12. parse_vint — 3-byte value ────────────────────────────────────────

    #[test]
    fn test_parse_vint_3byte() {
        // 0x20 0x00 0x10: leading byte 0x20 = 0010_0000 → width=3
        // marker stripped: 0x20 & 0x1F = 0x00, then 0x00, then 0x10
        // value = 0x00_0010 = 16
        let (val, width) = parse_vint(&[0x20, 0x00, 0x10]).unwrap();
        assert_eq!(width, 3);
        assert_eq!(val, 16);
    }

    // ── 13. parse_vint — unexpected EOF ──────────────────────────────────────

    #[test]
    fn test_parse_vint_eof() {
        // 0x40 signals a 2-byte vint but only 1 byte provided
        let err = parse_vint(&[0x40]).unwrap_err();
        assert!(matches!(
            err,
            ClusterError::UnexpectedEof { need: 2, have: 1 }
        ));
    }

    // ── 14. parse_vint_signed — round-trip ───────────────────────────────────

    #[test]
    fn test_parse_vint_signed_zero() {
        // For width=1, bias = 2^6 - 1 = 63. A value of 63 decodes to signed 0.
        // vint encoding of 63 in 1 byte: 0x80 | 63 = 0xBF
        let (val, w) = parse_vint_signed(&[0xBF]).unwrap();
        assert_eq!(w, 1);
        assert_eq!(val, 0);
    }

    // ── 15. ClusterBlock::absolute_timecode delegation ───────────────────────

    #[test]
    fn test_cluster_block_absolute_timecode() {
        let block = SimpleBlock {
            track_number: 1,
            timecode: 50,
            flags: BlockFlags::KEYFRAME,
            lace_type: LaceType::None,
            frame_sizes: vec![10],
            data_offset: 4,
            data_len: 10,
        };
        let cb = ClusterBlock::Simple(block);
        assert_eq!(cb.absolute_timecode(1000).unwrap(), 1050);
    }

    // ── 16. parse_simple_block — track number zero error ─────────────────────

    #[test]
    fn test_parse_simple_block_track_zero() {
        // 0x80 as vint: width=1, value=0 — track number 0 is invalid
        let raw: &[u8] = &[0x80, 0x00, 0x00, 0x00, 0x01, 0x02];
        let err = parse_simple_block(raw).unwrap_err();
        assert_eq!(err, ClusterError::TrackNumberZero);
    }

    // ── 17. decode_ebml_lace_sizes — 3 equal-size frames ─────────────────────

    #[test]
    fn test_decode_ebml_lace_sizes_equal() {
        // 3 frames each 100 bytes = 300 bytes total.
        // First frame size: vint(100) = 0xE4 (1-byte: 0x80 | 100 = 0xE4)
        // Second size delta: 0 (signed vint: for width=1, bias=63, encode 63 → 0xBF)
        // Third frame: implicit = 300 - 100 - 100 = 100.
        let header = &[0xE4, 0xBF];
        let (sizes, hdr_len) = decode_ebml_lace_sizes(header, 3, 300).unwrap();
        assert_eq!(hdr_len, 2);
        assert_eq!(sizes, vec![100, 100, 100]);
    }

    // ── 18. LaceType::from_flags_byte round-trip ─────────────────────────────

    #[test]
    fn test_lace_type_from_flags() {
        // bits 2-1 in the byte encode lace type
        // None:  (byte >> 1) & 0b11 = 0b00 → byte with those bits = 0x00
        // Fixed: (byte >> 1) & 0b11 = 0b01 → byte = 0x02
        // Xiph:  (byte >> 1) & 0b11 = 0b10 → byte = 0x04
        // Ebml:  (byte >> 1) & 0b11 = 0b11 → byte = 0x06
        assert_eq!(LaceType::from_flags_byte(0x00), LaceType::None);
        assert_eq!(LaceType::from_flags_byte(0x02), LaceType::Fixed);
        assert_eq!(LaceType::from_flags_byte(0x04), LaceType::Xiph);
        assert_eq!(LaceType::from_flags_byte(0x06), LaceType::Ebml);
    }
}
