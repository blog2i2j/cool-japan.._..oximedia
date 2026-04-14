//! Compact binary framing for WebSocket sync messages.
//!
//! JSON serialization is convenient but expensive for high-frequency operations:
//! a simple cursor-position update is ~60 bytes as JSON vs ~8-12 bytes as a
//! binary frame.  At 100 ms poll intervals across 10 concurrent editors, that
//! overhead accumulates to megabytes per minute — far in excess of what a
//! typical sync channel requires.
//!
//! This module provides a compact binary frame format (`BinaryFrame`) and a
//! batching layer (`BatchedFramer`) that accumulates frames and flushes them as
//! a single WebSocket binary message.  The combination eliminates per-message
//! framing overhead while keeping the wire format simple enough to parse in
//! O(1) per frame.
//!
//! # Wire Format
//!
//! ```text
//! ┌───────────┬───────────┬─────────────────┐
//! │ type (2B) │  len (2B) │  payload (N)    │
//! └───────────┴───────────┴─────────────────┘
//! ```
//!
//! Both `type` and `len` are little-endian unsigned 16-bit integers.
//! `len` is the byte length of the payload that follows.
//!
//! The 4-byte header overhead means a heartbeat (`FrameType::Heartbeat`) with
//! an empty payload costs exactly 4 bytes vs. `{"type":"Ping"}` (15 bytes JSON).
//!
//! # Batching
//!
//! `BatchedFramer` accumulates frames until either:
//! - A call to `flush()` drains the pending queue, or
//! - The accumulated size reaches `max_batch_bytes` (auto-flush on `add()`).
//!
//! The returned `Vec<u8>` is a contiguous buffer of back-to-back frames that
//! can be decoded with `BatchedFramer::decode_batch`.

use std::sync::atomic::{AtomicU64, Ordering};

// ─── FrameType ───────────────────────────────────────────────────────────────

/// Discriminant for the binary wire frame type.
///
/// Each variant maps to a 16-bit little-endian integer on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum FrameType {
    /// Cursor or playhead position update (minimal payload: ~4-8 bytes).
    CursorMove = 0x0001,
    /// Single edit operation payload.
    Edit = 0x0002,
    /// Heartbeat — keeps the WebSocket connection alive (empty payload).
    Heartbeat = 0x0003,
    /// Acknowledgement of a previously received frame sequence.
    Ack = 0x0004,
    /// A batch of edit operations encoded back-to-back.
    BatchedEdits = 0x0005,
    /// Presence update (user connected / disconnected / idle).
    PresenceUpdate = 0x0006,
    /// Frame type not recognised by this version of the protocol.
    Unknown = 0xFFFF,
}

impl FrameType {
    /// Convert a raw `u16` wire value to the corresponding `FrameType`.
    pub fn from_u16(v: u16) -> Self {
        match v {
            0x0001 => Self::CursorMove,
            0x0002 => Self::Edit,
            0x0003 => Self::Heartbeat,
            0x0004 => Self::Ack,
            0x0005 => Self::BatchedEdits,
            0x0006 => Self::PresenceUpdate,
            _ => Self::Unknown,
        }
    }

    /// Return the numeric wire value.
    pub fn as_u16(self) -> u16 {
        self as u16
    }
}

// ─── BinaryFrame ─────────────────────────────────────────────────────────────

/// A single binary frame ready for transmission over a WebSocket connection.
///
/// The frame encodes the message type and an arbitrary byte payload.  Encoding
/// and decoding are O(n) in the payload length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryFrame {
    /// The semantic type of this frame.
    pub frame_type: FrameType,
    /// Raw bytes of the frame payload (may be empty, e.g. for `Heartbeat`).
    pub payload: Vec<u8>,
}

impl BinaryFrame {
    /// Create a new frame with the given type and payload.
    pub fn new(frame_type: FrameType, payload: Vec<u8>) -> Self {
        Self {
            frame_type,
            payload,
        }
    }

    /// Create a heartbeat frame (no payload).
    pub fn heartbeat() -> Self {
        Self::new(FrameType::Heartbeat, Vec::new())
    }

    /// Total encoded byte size: 4-byte header + payload length.
    #[must_use]
    pub fn encoded_size(&self) -> usize {
        4 + self.payload.len()
    }

    /// Encode the frame into a byte buffer: `[type_u16_le][len_u16_le][payload]`.
    ///
    /// Panics if `self.payload.len() > u16::MAX` (65 535 bytes).  In practice
    /// sync payloads are far smaller; callers should split large payloads if needed.
    pub fn encode(&self) -> Vec<u8> {
        let payload_len = self.payload.len();
        assert!(
            u16::try_from(payload_len).is_ok(),
            "payload too large for binary frame: {payload_len} bytes"
        );

        let mut out = Vec::with_capacity(4 + payload_len);
        out.extend_from_slice(&self.frame_type.as_u16().to_le_bytes());
        // SAFETY: asserted above that payload_len fits in u16
        out.extend_from_slice(&(payload_len as u16).to_le_bytes());
        out.extend_from_slice(&self.payload);
        out
    }

    /// Attempt to decode one frame from the beginning of `data`.
    ///
    /// Returns `Some((frame, bytes_consumed))` on success, or `None` if `data`
    /// is too short to contain a complete frame.  The caller should advance its
    /// slice by `bytes_consumed` before calling `decode` again.
    pub fn decode(data: &[u8]) -> Option<(Self, usize)> {
        if data.len() < 4 {
            return None;
        }
        let type_u16 = u16::from_le_bytes([data[0], data[1]]);
        let payload_len = u16::from_le_bytes([data[2], data[3]]) as usize;
        let total = 4 + payload_len;
        if data.len() < total {
            return None;
        }
        let payload = data[4..total].to_vec();
        let frame = Self {
            frame_type: FrameType::from_u16(type_u16),
            payload,
        };
        Some((frame, total))
    }
}

// ─── BatchedFramer ───────────────────────────────────────────────────────────

/// Accumulates binary frames and encodes them into a single WebSocket message.
///
/// Instead of sending one WebSocket message per operation (each with its own
/// framing overhead), `BatchedFramer` accumulates operations up to a byte
/// budget and then flushes them as a single contiguous binary message.
///
/// # Auto-flush
///
/// If the accumulated size would reach `max_batch_bytes`, `add()` automatically
/// calls `flush()` and returns the flushed data before queuing the new frame.
///
/// # Statistics
///
/// `total_encoded_bytes()` and `total_batches()` track overall throughput;
/// `avg_batch_size()` computes the mean bytes-per-flush over the lifetime of
/// the framer.
pub struct BatchedFramer {
    pending: Vec<BinaryFrame>,
    max_batch_bytes: usize,
    current_bytes: usize,
    total_encoded: AtomicU64,
    total_batches: AtomicU64,
}

impl BatchedFramer {
    /// Create a new framer with the given flush threshold in bytes.
    ///
    /// `max_batch_bytes = 0` means every `add()` call will auto-flush.
    pub fn new(max_batch_bytes: usize) -> Self {
        Self {
            pending: Vec::new(),
            max_batch_bytes,
            current_bytes: 0,
            total_encoded: AtomicU64::new(0),
            total_batches: AtomicU64::new(0),
        }
    }

    /// Add a frame to the pending batch.
    ///
    /// If the accumulated byte count reaches `max_batch_bytes`, the pending
    /// batch (including the new frame) is flushed automatically and the
    /// encoded bytes are returned.  Otherwise returns `None`.
    pub fn add(&mut self, frame: BinaryFrame) -> Option<Vec<u8>> {
        let frame_size = frame.encoded_size();
        self.current_bytes += frame_size;
        self.pending.push(frame);

        if self.current_bytes >= self.max_batch_bytes {
            Some(self.flush())
        } else {
            None
        }
    }

    /// Flush all pending frames into a single encoded byte buffer.
    ///
    /// Updates the lifetime statistics (`total_encoded`, `total_batches`).
    /// Returns an empty `Vec` if there are no pending frames.
    pub fn flush(&mut self) -> Vec<u8> {
        if self.pending.is_empty() {
            return Vec::new();
        }

        let mut out = Vec::with_capacity(self.current_bytes);
        for frame in self.pending.drain(..) {
            out.extend_from_slice(&frame.encode());
        }

        let out_len = out.len() as u64;
        self.total_encoded.fetch_add(out_len, Ordering::Relaxed);
        self.total_batches.fetch_add(1, Ordering::Relaxed);
        self.current_bytes = 0;

        out
    }

    /// Decode a contiguous byte buffer produced by `flush()` back into frames.
    ///
    /// Unknown frame types are preserved as `FrameType::Unknown` frames rather
    /// than discarded, allowing the caller to log or forward them.
    pub fn decode_batch(data: &[u8]) -> Vec<BinaryFrame> {
        let mut frames = Vec::new();
        let mut offset = 0;
        while offset < data.len() {
            match BinaryFrame::decode(&data[offset..]) {
                Some((frame, consumed)) => {
                    frames.push(frame);
                    offset += consumed;
                }
                None => break,
            }
        }
        frames
    }

    /// Number of frames currently pending (not yet flushed).
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Whether there are no pending frames.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Total bytes encoded across all `flush()` calls.
    #[must_use]
    pub fn total_encoded_bytes(&self) -> u64 {
        self.total_encoded.load(Ordering::Relaxed)
    }

    /// Total number of `flush()` calls that produced output.
    #[must_use]
    pub fn total_batches(&self) -> u64 {
        self.total_batches.load(Ordering::Relaxed)
    }

    /// Mean bytes per flushed batch over the framer's lifetime.
    ///
    /// Returns `0.0` if no batches have been flushed yet.
    #[must_use]
    pub fn avg_batch_size(&self) -> f64 {
        let batches = self.total_batches.load(Ordering::Relaxed);
        if batches == 0 {
            return 0.0;
        }
        let encoded = self.total_encoded.load(Ordering::Relaxed);
        encoded as f64 / batches as f64
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // ── FrameType ────────────────────────────────────────────────────────────

    #[test]
    fn test_frame_type_cursor_move_round_trip() {
        assert_eq!(FrameType::from_u16(0x0001), FrameType::CursorMove);
        assert_eq!(FrameType::CursorMove.as_u16(), 0x0001);
    }

    #[test]
    fn test_frame_type_edit_round_trip() {
        assert_eq!(FrameType::from_u16(0x0002), FrameType::Edit);
        assert_eq!(FrameType::Edit.as_u16(), 0x0002);
    }

    #[test]
    fn test_frame_type_heartbeat_round_trip() {
        assert_eq!(FrameType::from_u16(0x0003), FrameType::Heartbeat);
        assert_eq!(FrameType::Heartbeat.as_u16(), 0x0003);
    }

    #[test]
    fn test_frame_type_ack_round_trip() {
        assert_eq!(FrameType::from_u16(0x0004), FrameType::Ack);
        assert_eq!(FrameType::Ack.as_u16(), 0x0004);
    }

    #[test]
    fn test_frame_type_batched_edits_round_trip() {
        assert_eq!(FrameType::from_u16(0x0005), FrameType::BatchedEdits);
        assert_eq!(FrameType::BatchedEdits.as_u16(), 0x0005);
    }

    #[test]
    fn test_frame_type_presence_update_round_trip() {
        assert_eq!(FrameType::from_u16(0x0006), FrameType::PresenceUpdate);
        assert_eq!(FrameType::PresenceUpdate.as_u16(), 0x0006);
    }

    #[test]
    fn test_frame_type_unknown_for_unrecognised_value() {
        assert_eq!(FrameType::from_u16(0xABCD), FrameType::Unknown);
    }

    // ── BinaryFrame encode / decode ──────────────────────────────────────────

    #[test]
    fn test_encode_decode_cursor_move() {
        let frame = BinaryFrame::new(FrameType::CursorMove, vec![0x00, 0x00, 0x03, 0xE8]);
        let encoded = frame.encode();
        let (decoded, consumed) = BinaryFrame::decode(&encoded).expect("should decode");
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.frame_type, FrameType::CursorMove);
        assert_eq!(decoded.payload, frame.payload);
    }

    #[test]
    fn test_encode_decode_empty_heartbeat() {
        let frame = BinaryFrame::heartbeat();
        assert_eq!(frame.payload.len(), 0);
        let encoded = frame.encode();
        assert_eq!(encoded.len(), 4); // header only
        let (decoded, consumed) = BinaryFrame::decode(&encoded).expect("decode heartbeat");
        assert_eq!(consumed, 4);
        assert_eq!(decoded.frame_type, FrameType::Heartbeat);
        assert!(decoded.payload.is_empty());
    }

    #[test]
    fn test_encode_decode_edit_with_payload() {
        let payload = b"insert:42:hello".to_vec();
        let frame = BinaryFrame::new(FrameType::Edit, payload.clone());
        let encoded = frame.encode();
        let (decoded, _) = BinaryFrame::decode(&encoded).expect("decode edit");
        assert_eq!(decoded.frame_type, FrameType::Edit);
        assert_eq!(decoded.payload, payload);
    }

    #[test]
    fn test_decode_too_short_header_returns_none() {
        // Only 3 bytes — not enough for the 4-byte header
        let short = vec![0x01, 0x00, 0x05];
        assert!(BinaryFrame::decode(&short).is_none());
    }

    #[test]
    fn test_decode_truncated_payload_returns_none() {
        // Header says payload len = 10, but only 3 payload bytes follow
        let mut data = vec![0x02, 0x00]; // type = Edit
        data.extend_from_slice(&10u16.to_le_bytes()); // len = 10
        data.extend_from_slice(&[0x01, 0x02, 0x03]); // only 3 bytes
        assert!(BinaryFrame::decode(&data).is_none());
    }

    #[test]
    fn test_encoded_size_matches_actual_encode_length() {
        let frame = BinaryFrame::new(FrameType::PresenceUpdate, vec![1, 2, 3, 4, 5]);
        assert_eq!(frame.encoded_size(), frame.encode().len());
    }

    #[test]
    fn test_binary_frame_is_clone() {
        let frame = BinaryFrame::new(FrameType::Ack, vec![0xCA, 0xFE]);
        let cloned = frame.clone();
        assert_eq!(frame, cloned);
    }

    // ── BatchedFramer ────────────────────────────────────────────────────────

    #[test]
    fn test_batcher_starts_empty() {
        let framer = BatchedFramer::new(4096);
        assert_eq!(framer.pending_count(), 0);
        assert!(framer.is_empty());
        assert_eq!(framer.total_encoded_bytes(), 0);
        assert_eq!(framer.total_batches(), 0);
    }

    #[test]
    fn test_batcher_flush_with_multiple_frames() {
        let mut framer = BatchedFramer::new(65536); // large limit — no auto-flush
        framer.add(BinaryFrame::new(FrameType::Edit, b"op1".to_vec()));
        framer.add(BinaryFrame::new(FrameType::Edit, b"op2".to_vec()));
        framer.add(BinaryFrame::heartbeat());

        assert_eq!(framer.pending_count(), 3);

        let data = framer.flush();
        assert!(framer.is_empty());
        assert_eq!(framer.total_batches(), 1);

        let decoded = BatchedFramer::decode_batch(&data);
        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0].frame_type, FrameType::Edit);
        assert_eq!(decoded[0].payload, b"op1");
        assert_eq!(decoded[2].frame_type, FrameType::Heartbeat);
    }

    #[test]
    fn test_batcher_auto_flush_when_limit_reached() {
        // Limit = 12 bytes; a heartbeat (4 bytes) + edit with 7-byte payload (11 bytes) = 15 bytes total → triggers flush
        let mut framer = BatchedFramer::new(12);

        // First frame: heartbeat = 4 bytes; 4 < 12, no flush
        let r1 = framer.add(BinaryFrame::heartbeat());
        assert!(r1.is_none(), "no flush yet — under limit");

        // Second frame: edit with 9-byte payload = 13 bytes encoded; 4+13=17 >= 12 → auto-flush
        let r2 = framer.add(BinaryFrame::new(FrameType::Edit, b"XXXXXXXXX".to_vec()));
        assert!(r2.is_some(), "should auto-flush when limit exceeded");

        // Both frames should appear in the flushed buffer
        let data = r2.expect("auto-flush result");
        let decoded = BatchedFramer::decode_batch(&data);
        assert_eq!(decoded.len(), 2);
    }

    #[test]
    fn test_batcher_empty_flush_returns_empty_vec() {
        let mut framer = BatchedFramer::new(4096);
        let data = framer.flush();
        assert!(data.is_empty());
        // Empty flush must NOT increment batch counter
        assert_eq!(framer.total_batches(), 0);
    }

    #[test]
    fn test_batcher_total_encoded_bytes_accumulates() {
        let mut framer = BatchedFramer::new(65536);
        framer.add(BinaryFrame::new(FrameType::Edit, b"hello".to_vec())); // 4+5=9
        let data = framer.flush();
        assert_eq!(framer.total_encoded_bytes(), data.len() as u64);
    }

    #[test]
    fn test_batcher_avg_batch_size_after_two_flushes() {
        let mut framer = BatchedFramer::new(65536);

        // Flush 1: 1 heartbeat = 4 bytes
        framer.add(BinaryFrame::heartbeat());
        let _d1 = framer.flush();

        // Flush 2: 1 edit with 8-byte payload = 12 bytes
        framer.add(BinaryFrame::new(FrameType::Edit, b"12345678".to_vec()));
        let _d2 = framer.flush();

        assert_eq!(framer.total_batches(), 2);
        // avg = (4 + 12) / 2 = 8.0
        let avg = framer.avg_batch_size();
        assert!((avg - 8.0).abs() < f64::EPSILON, "avg={avg}");
    }

    #[test]
    fn test_decode_batch_multiple_concatenated_frames() {
        let f1 = BinaryFrame::new(FrameType::CursorMove, vec![0x01, 0x02, 0x03, 0x04]);
        let f2 = BinaryFrame::heartbeat();
        let f3 = BinaryFrame::new(FrameType::Ack, vec![0xFF]);

        let mut buffer = Vec::new();
        buffer.extend_from_slice(&f1.encode());
        buffer.extend_from_slice(&f2.encode());
        buffer.extend_from_slice(&f3.encode());

        let frames = BatchedFramer::decode_batch(&buffer);
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0], f1);
        assert_eq!(frames[1], f2);
        assert_eq!(frames[2], f3);
    }

    #[test]
    fn test_binary_frame_more_compact_than_json() {
        // A cursor-move with a 4-byte position payload = 8 bytes binary.
        // The equivalent JSON: {"type":"CursorMove","payload":[1,2,3,4]} = 44 bytes.
        let binary_size =
            BinaryFrame::new(FrameType::CursorMove, vec![0, 0, 3, 232]).encoded_size();
        let json_equiv = r#"{"type":"CursorMove","payload":[0,0,3,232]}"#.len();
        assert!(
            binary_size < json_equiv,
            "binary ({binary_size}B) should be smaller than JSON ({json_equiv}B)"
        );
    }

    /// Concurrent encode → decode round-trip across 4 threads.
    #[test]
    fn test_concurrent_encode_decode_round_trip() {
        // Wrap payload in Arc so threads can share it for comparison
        let payload: Arc<Vec<u8>> = Arc::new(b"concurrent test payload".to_vec());

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let p = Arc::clone(&payload);
                std::thread::spawn(move || {
                    let frame_type = match i % 3 {
                        0 => FrameType::Edit,
                        1 => FrameType::CursorMove,
                        _ => FrameType::Ack,
                    };
                    let frame = BinaryFrame::new(frame_type, p.as_ref().clone());
                    let encoded = frame.encode();
                    let (decoded, consumed) =
                        BinaryFrame::decode(&encoded).expect("concurrent decode should succeed");
                    assert_eq!(consumed, encoded.len());
                    assert_eq!(decoded.frame_type, frame_type);
                    assert_eq!(decoded.payload, *p);
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread should not panic");
        }
    }

    #[test]
    fn test_decode_unknown_frame_type_preserved() {
        // Build a raw frame with type 0xDEAD
        let mut data = Vec::new();
        data.extend_from_slice(&0xDEADu16.to_le_bytes());
        data.extend_from_slice(&2u16.to_le_bytes());
        data.extend_from_slice(&[0x01, 0x02]);

        let (frame, _) = BinaryFrame::decode(&data).expect("decode should succeed");
        assert_eq!(frame.frame_type, FrameType::Unknown);
        assert_eq!(frame.payload, vec![0x01, 0x02]);
    }
}
