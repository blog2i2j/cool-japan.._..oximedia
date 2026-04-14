//! CMAF chunk sequencer for Low-Latency streaming.
//!
//! [`CmafSequencer`] accumulates raw media data and emits CMAF-compatible
//! chunk objects ([`CmafChunkOut`]) when enough data has been accumulated to
//! form a chunk of the configured target duration.
//!
//! CMAF (Common Media Application Format, ISO 23000-19) builds on fragmented
//! MP4 (fMP4) and defines a chunk as a `moof + mdat` pair that can be
//! transferred independently for low-latency delivery.
//!
//! # Example
//!
//! ```rust
//! use oximedia_stream::cmaf_sequencer::CmafSequencer;
//!
//! let mut seq = CmafSequencer::new(2000); // 2-second chunks
//!
//! // Simulate feeding 500 ms worth of data per call
//! let data = vec![0u8; 1024];
//! let chunk1 = seq.push_data(&data, 500); // Not enough yet → None
//! assert!(chunk1.is_none());
//!
//! for _ in 0..3 {
//!     seq.push_data(&data, 500);
//! }
//! // After 2000 ms accumulated the sequencer will emit a chunk
//! ```

// ─── CmafChunkOut ─────────────────────────────────────────────────────────────

/// A single emitted CMAF chunk.
///
/// Each chunk is a self-contained `moof + mdat` fMP4 unit that can be
/// transferred to a client independently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CmafChunkOut {
    /// Monotonically increasing chunk sequence number (1-indexed).
    pub sequence: u64,
    /// Accumulated duration of this chunk in milliseconds.
    pub duration_ms: u64,
    /// The chunk payload wrapped in a minimal `moof + mdat` box structure.
    pub data: Vec<u8>,
    /// Whether this chunk represents the end of a complete CMAF segment.
    pub is_segment_end: bool,
}

// ─── CmafSequencer ────────────────────────────────────────────────────────────

/// Accumulates media data and emits [`CmafChunkOut`]s at regular intervals.
///
/// A chunk boundary is created whenever the accumulated duration reaches or
/// exceeds `chunk_duration_ms`.  The final chunk in a segment can be forced
/// via [`flush`](CmafSequencer::flush).
pub struct CmafSequencer {
    /// Target chunk duration in milliseconds.
    pub chunk_duration_ms: u64,
    /// Number of media units (calls to `push_data`) per segment, used to
    /// detect segment boundaries.  0 means unbounded — no automatic segment end.
    pub units_per_segment: u32,
    /// Current accumulated raw payload for the in-progress chunk.
    pending_data: Vec<u8>,
    /// Accumulated duration in milliseconds for the in-progress chunk.
    pending_duration_ms: u64,
    /// Monotonically increasing chunk counter (1-indexed).
    next_sequence: u64,
    /// Number of complete chunks emitted in the current segment.
    chunks_in_segment: u32,
}

impl CmafSequencer {
    /// Create a sequencer with the given target `chunk_duration_ms`.
    pub fn new(chunk_duration_ms: u64) -> Self {
        Self {
            chunk_duration_ms,
            units_per_segment: 0,
            pending_data: Vec::new(),
            pending_duration_ms: 0,
            next_sequence: 1,
            chunks_in_segment: 0,
        }
    }

    /// Override the number of chunks per segment (0 = no automatic segment end).
    pub fn with_units_per_segment(mut self, units: u32) -> Self {
        self.units_per_segment = units;
        self
    }

    /// Feed `data` bytes representing `duration_ms` milliseconds of content.
    ///
    /// Returns `Some(CmafChunkOut)` if the accumulated duration has reached
    /// or exceeded `chunk_duration_ms`, otherwise `None`.
    pub fn push_data(&mut self, data: &[u8], duration_ms: u64) -> Option<CmafChunkOut> {
        self.pending_data.extend_from_slice(data);
        self.pending_duration_ms = self.pending_duration_ms.saturating_add(duration_ms);

        if self.pending_duration_ms >= self.chunk_duration_ms {
            Some(self.emit_chunk(false))
        } else {
            None
        }
    }

    /// Force emission of whatever data is currently buffered as a final chunk.
    ///
    /// Returns `None` when there is no buffered data.
    pub fn flush(&mut self) -> Option<CmafChunkOut> {
        if self.pending_data.is_empty() {
            None
        } else {
            Some(self.emit_chunk(true))
        }
    }

    /// Current accumulated duration in the pending chunk (milliseconds).
    pub fn pending_duration_ms(&self) -> u64 {
        self.pending_duration_ms
    }

    /// Number of bytes currently buffered for the pending chunk.
    pub fn pending_bytes(&self) -> usize {
        self.pending_data.len()
    }

    /// Next chunk sequence number to be emitted.
    pub fn next_sequence(&self) -> u64 {
        self.next_sequence
    }

    // ── Private ───────────────────────────────────────────────────────────────

    fn emit_chunk(&mut self, forced: bool) -> CmafChunkOut {
        let seq = self.next_sequence;
        self.next_sequence = self.next_sequence.saturating_add(1);

        let payload = std::mem::take(&mut self.pending_data);
        let duration_ms = self.pending_duration_ms;
        self.pending_duration_ms = 0;

        // Determine segment-end flag.
        self.chunks_in_segment += 1;
        let is_segment_end = forced
            || (self.units_per_segment > 0 && self.chunks_in_segment >= self.units_per_segment);
        if is_segment_end {
            self.chunks_in_segment = 0;
        }

        // Wrap in minimal fMP4 moof+mdat boxes.
        let data = build_moof_mdat(seq, &payload);

        CmafChunkOut {
            sequence: seq,
            duration_ms,
            data,
            is_segment_end,
        }
    }
}

// ─── fMP4 minimal box builder ─────────────────────────────────────────────────

/// Build a minimal `moof + mdat` fMP4 structure for a CMAF chunk.
///
/// Box layout:
/// ```text
/// moof {
///   mfhd { sequence_number }
/// }
/// mdat { payload }
/// ```
fn build_moof_mdat(sequence: u64, payload: &[u8]) -> Vec<u8> {
    // mfhd: 8 (box header) + 4 (version+flags) + 4 (seq_num) = 16 bytes
    let mfhd_size: u32 = 16;
    let seq_u32 = (sequence & 0xFFFF_FFFF) as u32;

    // moof: 8 (box header) + mfhd = 24 bytes
    let moof_size: u32 = 8 + mfhd_size;

    // mdat: 8 (box header) + payload
    let mdat_payload_len = payload.len() as u32;
    let mdat_size: u32 = 8u32.saturating_add(mdat_payload_len);

    let mut out = Vec::with_capacity((moof_size + mdat_size) as usize);

    // moof
    out.extend_from_slice(&moof_size.to_be_bytes());
    out.extend_from_slice(b"moof");
    // mfhd inside moof
    out.extend_from_slice(&mfhd_size.to_be_bytes());
    out.extend_from_slice(b"mfhd");
    out.extend_from_slice(&0u32.to_be_bytes()); // version + flags
    out.extend_from_slice(&seq_u32.to_be_bytes());

    // mdat
    out.extend_from_slice(&mdat_size.to_be_bytes());
    out.extend_from_slice(b"mdat");
    out.extend_from_slice(payload);

    out
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_chunk_before_duration_reached() {
        let mut seq = CmafSequencer::new(2000);
        let result = seq.push_data(&[1u8; 512], 500);
        assert!(result.is_none(), "not enough duration accumulated yet");
    }

    #[test]
    fn test_chunk_emitted_when_duration_reached() {
        let mut seq = CmafSequencer::new(2000);
        seq.push_data(&[1u8; 128], 500);
        seq.push_data(&[2u8; 128], 500);
        seq.push_data(&[3u8; 128], 500);
        let chunk = seq.push_data(&[4u8; 128], 500);
        assert!(chunk.is_some(), "chunk should be emitted at 2000ms");
        let c = chunk.expect("chunk");
        assert_eq!(c.duration_ms, 2000);
        assert_eq!(c.sequence, 1);
    }

    #[test]
    fn test_sequence_increments() {
        let mut seq = CmafSequencer::new(1000);
        let c1 = seq.push_data(&[0u8; 64], 1000).expect("chunk 1");
        let c2 = seq.push_data(&[0u8; 64], 1000).expect("chunk 2");
        assert_eq!(c1.sequence, 1);
        assert_eq!(c2.sequence, 2);
    }

    #[test]
    fn test_flush_emits_partial_chunk() {
        let mut seq = CmafSequencer::new(5000);
        seq.push_data(&[0u8; 32], 200);
        seq.push_data(&[0u8; 32], 300);
        let chunk = seq.flush().expect("flushed chunk");
        assert_eq!(chunk.duration_ms, 500);
        assert!(
            chunk.is_segment_end,
            "flushed chunk should mark segment end"
        );
    }

    #[test]
    fn test_flush_returns_none_when_empty() {
        let mut seq = CmafSequencer::new(2000);
        assert!(seq.flush().is_none());
    }

    #[test]
    fn test_data_contains_moof_mdat_boxes() {
        let mut seq = CmafSequencer::new(1000);
        let chunk = seq.push_data(&[0xAB; 100], 1000).expect("chunk");
        // Check moof box marker at offset 4
        assert_eq!(&chunk.data[4..8], b"moof");
        // mdat box starts after moof (24 bytes); fourcc is at offset 28 (after 4-byte size)
        assert_eq!(&chunk.data[28..32], b"mdat");
    }

    #[test]
    fn test_pending_duration_resets_after_emit() {
        let mut seq = CmafSequencer::new(1000);
        seq.push_data(&[0u8; 64], 1000).expect("emit");
        assert_eq!(seq.pending_duration_ms(), 0);
        assert_eq!(seq.pending_bytes(), 0);
    }

    #[test]
    fn test_segment_end_flag_with_units_per_segment() {
        let mut seq = CmafSequencer::new(1000).with_units_per_segment(3);
        let c1 = seq.push_data(&[0u8; 32], 1000).expect("c1");
        assert!(!c1.is_segment_end);
        let c2 = seq.push_data(&[0u8; 32], 1000).expect("c2");
        assert!(!c2.is_segment_end);
        let c3 = seq.push_data(&[0u8; 32], 1000).expect("c3");
        assert!(
            c3.is_segment_end,
            "3rd chunk of 3-per-segment should end segment"
        );
    }
}
