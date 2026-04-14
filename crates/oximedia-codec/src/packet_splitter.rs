//! Packet splitting and fragment reassembly for codec bitstreams.
//!
//! Provides utilities to split oversized NAL units / codec packets into
//! MTU-bounded fragments and to reassemble them back to the original payload.
//!
//! The fragmentation scheme is transport-agnostic: each fragment carries a
//! small header so that a receiver can reconstruct the original packet from
//! an unordered set of fragments.

use std::fmt;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur during packet splitting or reassembly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SplitterError {
    /// The maximum packet size is too small to hold even the fragment header.
    MaxSizeTooSmall {
        /// The requested max packet size.
        max_size: usize,
        /// The minimum required (header size).
        min_required: usize,
    },
    /// A fragment header is malformed (too short or invalid field values).
    MalformedFragmentHeader {
        /// Byte offset of the fragment within the reassembly buffer.
        offset: usize,
    },
    /// One or more fragments are missing; reassembly cannot complete.
    MissingFragments {
        /// Total expected fragments.
        total: u16,
        /// Number received.
        received: usize,
    },
    /// Fragment indices are duplicated or inconsistent.
    InconsistentFragments,
    /// The input packet is empty.
    EmptyPacket,
    /// The declared total fragment count exceeds a safety limit.
    TooManyFragments(u16),
}

impl fmt::Display for SplitterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MaxSizeTooSmall {
                max_size,
                min_required,
            } => {
                write!(
                    f,
                    "max packet size {max_size} is smaller than fragment header size {min_required}"
                )
            }
            Self::MalformedFragmentHeader { offset } => {
                write!(f, "malformed fragment header at offset {offset}")
            }
            Self::MissingFragments { total, received } => {
                write!(
                    f,
                    "missing fragments: expected {total}, received {received}"
                )
            }
            Self::InconsistentFragments => write!(f, "inconsistent fragment indices"),
            Self::EmptyPacket => write!(f, "packet is empty"),
            Self::TooManyFragments(n) => write!(f, "too many fragments: {n}"),
        }
    }
}

impl std::error::Error for SplitterError {}

/// Result type for packet splitter operations.
pub type SplitterResult<T> = Result<T, SplitterError>;

// ---------------------------------------------------------------------------
// Fragment header layout
//
// Each fragment is prefixed with a 6-byte header:
//
//   Bytes 0-1: packet_id     (u16 BE) — identifies which original packet
//   Bytes 2-3: fragment_index (u16 BE) — zero-based index of this fragment
//   Bytes 4-5: total_fragments (u16 BE) — total number of fragments for this packet
//
// Followed by the fragment payload.
// ---------------------------------------------------------------------------

/// Size of the per-fragment header in bytes.
pub const FRAGMENT_HEADER_SIZE: usize = 6;

/// Safety cap on the number of fragments a single packet may be split into.
const MAX_FRAGMENTS: u16 = 4096;

/// A single fragment of a split packet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fragment {
    /// Identifier of the source packet (shared by all fragments of the same packet).
    pub packet_id: u16,
    /// Zero-based index of this fragment.
    pub fragment_index: u16,
    /// Total number of fragments that make up the original packet.
    pub total_fragments: u16,
    /// Fragment payload bytes (does NOT include the header).
    pub payload: Vec<u8>,
}

impl Fragment {
    /// Serialise this fragment (header + payload) into a byte vector.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(FRAGMENT_HEADER_SIZE + self.payload.len());
        out.extend_from_slice(&self.packet_id.to_be_bytes());
        out.extend_from_slice(&self.fragment_index.to_be_bytes());
        out.extend_from_slice(&self.total_fragments.to_be_bytes());
        out.extend_from_slice(&self.payload);
        out
    }

    /// Deserialise a fragment from raw bytes (including header).
    pub fn from_bytes(data: &[u8]) -> SplitterResult<Self> {
        if data.len() < FRAGMENT_HEADER_SIZE {
            return Err(SplitterError::MalformedFragmentHeader { offset: 0 });
        }
        let packet_id = u16::from_be_bytes([data[0], data[1]]);
        let fragment_index = u16::from_be_bytes([data[2], data[3]]);
        let total_fragments = u16::from_be_bytes([data[4], data[5]]);
        if total_fragments == 0 {
            return Err(SplitterError::MalformedFragmentHeader { offset: 0 });
        }
        if fragment_index >= total_fragments {
            return Err(SplitterError::InconsistentFragments);
        }
        Ok(Self {
            packet_id,
            fragment_index,
            total_fragments,
            payload: data[FRAGMENT_HEADER_SIZE..].to_vec(),
        })
    }
}

// ---------------------------------------------------------------------------
// Packet splitter
// ---------------------------------------------------------------------------

/// Configuration for the packet splitter.
#[derive(Debug, Clone)]
pub struct SplitterConfig {
    /// Maximum number of bytes per output fragment (including the fragment header).
    pub max_packet_size: usize,
}

impl SplitterConfig {
    /// Create a new splitter configuration.
    pub fn new(max_packet_size: usize) -> SplitterResult<Self> {
        if max_packet_size <= FRAGMENT_HEADER_SIZE {
            return Err(SplitterError::MaxSizeTooSmall {
                max_size: max_packet_size,
                min_required: FRAGMENT_HEADER_SIZE + 1,
            });
        }
        Ok(Self { max_packet_size })
    }

    /// Maximum payload bytes per fragment (after accounting for the header).
    pub fn max_payload_per_fragment(&self) -> usize {
        self.max_packet_size - FRAGMENT_HEADER_SIZE
    }
}

/// Split a packet into fragments that each fit within `config.max_packet_size`.
///
/// If the packet already fits in a single fragment it is still wrapped in a
/// single-element `Vec<Fragment>` for uniform handling.
///
/// # Parameters
///
/// - `packet_id`: Caller-supplied identifier that groups all fragments of a packet.
/// - `data`: The raw packet payload (e.g., one or more NAL units in AnnexB/AVCC).
/// - `config`: Splitter configuration.
pub fn split_packet(
    packet_id: u16,
    data: &[u8],
    config: &SplitterConfig,
) -> SplitterResult<Vec<Fragment>> {
    if data.is_empty() {
        return Err(SplitterError::EmptyPacket);
    }

    let max_payload = config.max_payload_per_fragment();
    // Integer ceiling division.
    let total_fragments = (data.len() + max_payload - 1) / max_payload;

    if total_fragments > MAX_FRAGMENTS as usize {
        return Err(SplitterError::TooManyFragments(total_fragments as u16));
    }

    let total_u16 = total_fragments as u16;
    let mut fragments = Vec::with_capacity(total_fragments);

    for (idx, chunk) in data.chunks(max_payload).enumerate() {
        fragments.push(Fragment {
            packet_id,
            fragment_index: idx as u16,
            total_fragments: total_u16,
            payload: chunk.to_vec(),
        });
    }

    Ok(fragments)
}

// ---------------------------------------------------------------------------
// Fragment reassembly
// ---------------------------------------------------------------------------

/// Reassemble a complete packet from an unordered collection of fragments.
///
/// All fragments must share the same `packet_id` and agree on `total_fragments`.
/// The function tolerates duplicates (last-write-wins by fragment index).
pub fn reassemble_fragments(fragments: &[Fragment]) -> SplitterResult<Vec<u8>> {
    if fragments.is_empty() {
        return Err(SplitterError::EmptyPacket);
    }

    let total = fragments[0].total_fragments;
    let packet_id = fragments[0].packet_id;

    if total == 0 {
        return Err(SplitterError::MalformedFragmentHeader { offset: 0 });
    }
    if total > MAX_FRAGMENTS {
        return Err(SplitterError::TooManyFragments(total));
    }

    // Validate consistency across all fragments.
    for (i, frag) in fragments.iter().enumerate() {
        if frag.packet_id != packet_id || frag.total_fragments != total {
            return Err(SplitterError::InconsistentFragments);
        }
        if frag.fragment_index >= total {
            return Err(SplitterError::MalformedFragmentHeader { offset: i });
        }
    }

    // Build a slot array; duplicate fragments overwrite earlier entries.
    let mut slots: Vec<Option<&[u8]>> = vec![None; total as usize];
    for frag in fragments {
        slots[frag.fragment_index as usize] = Some(&frag.payload);
    }

    // Check completeness.
    let received = slots.iter().filter(|s| s.is_some()).count();
    if received < total as usize {
        return Err(SplitterError::MissingFragments { total, received });
    }

    let total_bytes: usize = slots.iter().filter_map(|s| *s).map(|s| s.len()).sum();
    let mut out = Vec::with_capacity(total_bytes);
    for slot in slots {
        // Safety: we checked completeness above.
        if let Some(payload) = slot {
            out.extend_from_slice(payload);
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// NAL unit size enforcement
// ---------------------------------------------------------------------------

/// Split a single large NAL unit so that each piece does not exceed `max_nal_size`.
///
/// This is a byte-level split that does **not** attempt to find valid EBSP
/// boundaries; use it only when the codec or transport allows arbitrary slicing
/// (e.g., RTP packetization with FU-A style fragmentation).
///
/// The returned slices borrow from the input slice.
pub fn split_nal_unit(nal: &[u8], max_nal_size: usize) -> SplitterResult<Vec<&[u8]>> {
    if nal.is_empty() {
        return Err(SplitterError::EmptyPacket);
    }
    if max_nal_size == 0 {
        return Err(SplitterError::MaxSizeTooSmall {
            max_size: 0,
            min_required: 1,
        });
    }
    Ok(nal.chunks(max_nal_size).collect())
}

/// Enforce a maximum payload size across a list of AnnexB NAL units.
///
/// Each NAL unit that fits within `max_size` is kept as-is.  Larger NAL units
/// are split by [`split_nal_unit`].  The result is a flat list of byte slices,
/// each guaranteed to be ≤ `max_size` bytes.
pub fn enforce_max_nal_size<'a>(
    nals: &[&'a [u8]],
    max_size: usize,
) -> SplitterResult<Vec<&'a [u8]>> {
    if max_size == 0 {
        return Err(SplitterError::MaxSizeTooSmall {
            max_size: 0,
            min_required: 1,
        });
    }
    let mut result = Vec::new();
    for &nal in nals {
        if nal.len() <= max_size {
            result.push(nal);
        } else {
            let pieces = split_nal_unit(nal, max_size)?;
            result.extend(pieces);
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Serialise / deserialise helpers for raw fragment bytes
// ---------------------------------------------------------------------------

/// Encode a list of fragments to a single flat byte buffer.
///
/// Each fragment is prefixed with a 2-byte big-endian length field so that the
/// buffer can be decoded without out-of-band information.
pub fn encode_fragment_stream(fragments: &[Fragment]) -> Vec<u8> {
    let total_bytes: usize = fragments
        .iter()
        .map(|f| 2 + FRAGMENT_HEADER_SIZE + f.payload.len())
        .sum();
    let mut out = Vec::with_capacity(total_bytes);
    for frag in fragments {
        let frag_bytes = frag.to_bytes();
        let frag_len = frag_bytes.len() as u16;
        out.extend_from_slice(&frag_len.to_be_bytes());
        out.extend_from_slice(&frag_bytes);
    }
    out
}

/// Decode a flat fragment stream previously produced by [`encode_fragment_stream`].
pub fn decode_fragment_stream(data: &[u8]) -> SplitterResult<Vec<Fragment>> {
    let mut fragments = Vec::new();
    let mut offset = 0usize;
    let len = data.len();

    while offset < len {
        if offset + 2 > len {
            return Err(SplitterError::MalformedFragmentHeader { offset });
        }
        let frag_len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2;
        if offset + frag_len > len {
            return Err(SplitterError::MalformedFragmentHeader { offset });
        }
        let frag = Fragment::from_bytes(&data[offset..offset + frag_len])?;
        fragments.push(frag);
        offset += frag_len;
    }

    Ok(fragments)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(max: usize) -> SplitterConfig {
        SplitterConfig::new(max).unwrap()
    }

    #[test]
    fn test_split_single_fragment() {
        let data = b"hello world";
        let cfg = make_config(64);
        let frags = split_packet(1, data, &cfg).unwrap();
        assert_eq!(frags.len(), 1);
        assert_eq!(frags[0].packet_id, 1);
        assert_eq!(frags[0].fragment_index, 0);
        assert_eq!(frags[0].total_fragments, 1);
        assert_eq!(frags[0].payload, data);
    }

    #[test]
    fn test_split_multiple_fragments() {
        // max_packet_size = FRAGMENT_HEADER_SIZE + 4 => payload_per_frag = 4
        let cfg = make_config(FRAGMENT_HEADER_SIZE + 4);
        let data: Vec<u8> = (0..10).collect();
        let frags = split_packet(42, &data, &cfg).unwrap();
        assert_eq!(frags.len(), 3); // 4 + 4 + 2
        assert!(frags.iter().all(|f| f.packet_id == 42));
        assert!(frags.iter().all(|f| f.total_fragments == 3));
        for (i, f) in frags.iter().enumerate() {
            assert_eq!(f.fragment_index, i as u16);
        }
    }

    #[test]
    fn test_reassemble_ordered() {
        let data: Vec<u8> = (0u8..100).collect();
        let cfg = make_config(FRAGMENT_HEADER_SIZE + 10);
        let frags = split_packet(7, &data, &cfg).unwrap();
        let reassembled = reassemble_fragments(&frags).unwrap();
        assert_eq!(reassembled, data);
    }

    #[test]
    fn test_reassemble_unordered() {
        let data: Vec<u8> = (0u8..30).collect();
        let cfg = make_config(FRAGMENT_HEADER_SIZE + 10);
        let mut frags = split_packet(99, &data, &cfg).unwrap();
        // Reverse order.
        frags.reverse();
        let reassembled = reassemble_fragments(&frags).unwrap();
        assert_eq!(reassembled, data);
    }

    #[test]
    fn test_reassemble_missing_fragment_error() {
        let data: Vec<u8> = (0u8..20).collect();
        let cfg = make_config(FRAGMENT_HEADER_SIZE + 5);
        let frags = split_packet(1, &data, &cfg).unwrap();
        // Drop second fragment.
        let partial: Vec<Fragment> = frags
            .into_iter()
            .filter(|f| f.fragment_index != 1)
            .collect();
        let err = reassemble_fragments(&partial).unwrap_err();
        assert!(matches!(err, SplitterError::MissingFragments { .. }));
    }

    #[test]
    fn test_fragment_serialise_deserialise() {
        let frag = Fragment {
            packet_id: 5,
            fragment_index: 0,
            total_fragments: 1,
            payload: vec![0xDE, 0xAD, 0xBE, 0xEF],
        };
        let bytes = frag.to_bytes();
        let decoded = Fragment::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, frag);
    }

    #[test]
    fn test_encode_decode_fragment_stream() {
        let data: Vec<u8> = (0u8..50).collect();
        let cfg = make_config(FRAGMENT_HEADER_SIZE + 10);
        let frags = split_packet(3, &data, &cfg).unwrap();
        let stream = encode_fragment_stream(&frags);
        let decoded_frags = decode_fragment_stream(&stream).unwrap();
        let reassembled = reassemble_fragments(&decoded_frags).unwrap();
        assert_eq!(reassembled, data);
    }

    #[test]
    fn test_split_nal_unit() {
        let nal = [0xAAu8; 100];
        let pieces = split_nal_unit(&nal, 30).unwrap();
        // 100 / 30 = 3 full + 1 partial
        assert_eq!(pieces.len(), 4);
        assert_eq!(pieces[0].len(), 30);
        assert_eq!(pieces[3].len(), 10);
    }

    #[test]
    fn test_enforce_max_nal_size() {
        let small = [0x01u8; 10];
        let large = [0x02u8; 50];
        let nals: Vec<&[u8]> = vec![&small, &large];
        let out = enforce_max_nal_size(&nals, 20).unwrap();
        // small fits as-is; large splits into 50/20 = 3 pieces
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].len(), 10);
        assert!(out[1..].iter().all(|s| s.len() <= 20));
    }

    #[test]
    fn test_config_too_small_error() {
        let err = SplitterConfig::new(FRAGMENT_HEADER_SIZE).unwrap_err();
        assert!(matches!(err, SplitterError::MaxSizeTooSmall { .. }));
    }

    #[test]
    fn test_empty_packet_split_error() {
        let cfg = make_config(64);
        let err = split_packet(0, &[], &cfg).unwrap_err();
        assert_eq!(err, SplitterError::EmptyPacket);
    }

    #[test]
    fn test_inconsistent_fragments_error() {
        let frag_a = Fragment {
            packet_id: 1,
            fragment_index: 0,
            total_fragments: 2,
            payload: vec![0x01],
        };
        // Different packet_id
        let frag_b = Fragment {
            packet_id: 2,
            fragment_index: 1,
            total_fragments: 2,
            payload: vec![0x02],
        };
        let err = reassemble_fragments(&[frag_a, frag_b]).unwrap_err();
        assert_eq!(err, SplitterError::InconsistentFragments);
    }
}
