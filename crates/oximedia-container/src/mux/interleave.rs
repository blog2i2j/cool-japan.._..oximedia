//! Stream interleaving for container muxers.
//!
//! Ensures packets from multiple streams are interleaved in timestamp order
//! so that media players can read them efficiently.

#![allow(dead_code)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A packet queued for interleaved writing.
#[derive(Debug, Clone)]
pub struct InterleavedPacket {
    /// Stream index the packet belongs to.
    pub stream_index: usize,
    /// Presentation timestamp (in stream timebase units).
    pub pts: i64,
    /// Decode timestamp, if different from PTS.
    pub dts: Option<i64>,
    /// Raw payload bytes.
    pub data: Vec<u8>,
    /// Whether this is a keyframe.
    pub is_keyframe: bool,
}

impl InterleavedPacket {
    /// Create a new packet.
    #[must_use]
    pub fn new(stream_index: usize, pts: i64, data: Vec<u8>, is_keyframe: bool) -> Self {
        Self {
            stream_index,
            pts,
            dts: None,
            data,
            is_keyframe,
        }
    }

    /// Effective sorting key: use DTS when present, else PTS.
    #[must_use]
    fn sort_key(&self) -> i64 {
        self.dts.unwrap_or(self.pts)
    }
}

// BinaryHeap is a max-heap; we want a min-heap by sort_key.
impl PartialEq for InterleavedPacket {
    fn eq(&self, other: &Self) -> bool {
        self.sort_key() == other.sort_key()
    }
}

impl Eq for InterleavedPacket {}

impl PartialOrd for InterleavedPacket {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for InterleavedPacket {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse so BinaryHeap becomes a min-heap
        other.sort_key().cmp(&self.sort_key())
    }
}

/// Interleaver buffers packets from multiple streams and emits them in
/// ascending timestamp order.
pub struct Interleaver {
    /// Internal priority queue (min-heap by `sort_key`).
    heap: BinaryHeap<InterleavedPacket>,
    /// Maximum number of packets buffered before forcing a flush.
    max_buffer: usize,
}

impl Interleaver {
    /// Create a new interleaver.
    ///
    /// `max_buffer` controls how many packets are held before the oldest is
    /// forced out regardless of ordering.
    #[must_use]
    pub fn new(max_buffer: usize) -> Self {
        Self {
            heap: BinaryHeap::new(),
            max_buffer: max_buffer.max(1),
        }
    }

    /// Push a packet into the interleaver buffer.
    pub fn push(&mut self, packet: InterleavedPacket) {
        self.heap.push(packet);
    }

    /// Pop the packet with the lowest timestamp, if any.
    #[must_use]
    pub fn pop(&mut self) -> Option<InterleavedPacket> {
        self.heap.pop()
    }

    /// Pop the lowest-timestamp packet only if the buffer exceeds `max_buffer`.
    ///
    /// This forces output when too many packets accumulate.
    pub fn pop_if_full(&mut self) -> Option<InterleavedPacket> {
        if self.heap.len() >= self.max_buffer {
            self.heap.pop()
        } else {
            None
        }
    }

    /// Drain all remaining packets in timestamp order.
    pub fn drain(&mut self) -> Vec<InterleavedPacket> {
        let mut out = Vec::with_capacity(self.heap.len());
        while let Some(p) = self.heap.pop() {
            out.push(p);
        }
        out
    }

    /// Number of packets currently buffered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Returns `true` when the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

/// PTS/DTS mapping utilities.
pub mod pts_map {
    /// Rescale a timestamp from one timebase to another.
    ///
    /// `src_tb` and `dst_tb` are expressed as `(numerator, denominator)`.
    ///
    /// # Panics
    ///
    /// Panics if either denominator is zero.
    #[must_use]
    pub fn rescale(ts: i64, src_tb: (i64, i64), dst_tb: (i64, i64)) -> i64 {
        assert!(src_tb.1 != 0, "src timebase denominator must not be zero");
        assert!(dst_tb.1 != 0, "dst timebase denominator must not be zero");
        // ts * (src_num / src_den) / (dst_num / dst_den)
        // = ts * src_num * dst_den / (src_den * dst_num)
        let numer = ts * src_tb.0 * dst_tb.1;
        let denom = src_tb.1 * dst_tb.0;
        numer / denom
    }

    /// Validate that DTS <= PTS for a packet.
    #[must_use]
    pub fn dts_before_pts(dts: i64, pts: i64) -> bool {
        dts <= pts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pkt(stream: usize, pts: i64, is_key: bool) -> InterleavedPacket {
        InterleavedPacket::new(stream, pts, vec![0u8; 4], is_key)
    }

    #[test]
    fn test_packet_new() {
        let p = pkt(0, 1000, true);
        assert_eq!(p.stream_index, 0);
        assert_eq!(p.pts, 1000);
        assert!(p.is_keyframe);
        assert_eq!(p.data.len(), 4);
    }

    #[test]
    fn test_packet_sort_key_uses_dts() {
        let mut p = pkt(0, 1000, false);
        p.dts = Some(990);
        assert_eq!(p.sort_key(), 990);
    }

    #[test]
    fn test_packet_sort_key_falls_back_to_pts() {
        let p = pkt(0, 500, false);
        assert_eq!(p.sort_key(), 500);
    }

    #[test]
    fn test_interleaver_push_pop_order() {
        let mut il = Interleaver::new(10);
        il.push(pkt(0, 300, false));
        il.push(pkt(1, 100, true));
        il.push(pkt(0, 200, false));

        let first = il.pop().expect("operation should succeed");
        assert_eq!(first.pts, 100);
        let second = il.pop().expect("operation should succeed");
        assert_eq!(second.pts, 200);
        let third = il.pop().expect("operation should succeed");
        assert_eq!(third.pts, 300);
    }

    #[test]
    fn test_interleaver_pop_empty() {
        let mut il = Interleaver::new(5);
        assert!(il.pop().is_none());
    }

    #[test]
    fn test_interleaver_len_and_empty() {
        let mut il = Interleaver::new(5);
        assert!(il.is_empty());
        il.push(pkt(0, 0, false));
        assert_eq!(il.len(), 1);
        assert!(!il.is_empty());
    }

    #[test]
    fn test_interleaver_drain() {
        let mut il = Interleaver::new(10);
        il.push(pkt(1, 50, false));
        il.push(pkt(0, 10, true));
        il.push(pkt(1, 30, false));
        let drained = il.drain();
        assert_eq!(drained.len(), 3);
        // Must be in ascending PTS order
        assert_eq!(drained[0].pts, 10);
        assert_eq!(drained[1].pts, 30);
        assert_eq!(drained[2].pts, 50);
        assert!(il.is_empty());
    }

    #[test]
    fn test_interleaver_pop_if_full() {
        let mut il = Interleaver::new(3);
        il.push(pkt(0, 1, false));
        il.push(pkt(0, 2, false));
        // Not yet full (len < 3)
        assert!(il.pop_if_full().is_none());
        il.push(pkt(0, 3, false));
        // Now full: pop_if_full should return the lowest
        let p = il.pop_if_full().expect("operation should succeed");
        assert_eq!(p.pts, 1);
    }

    #[test]
    fn test_rescale_same_timebase() {
        let ts = pts_map::rescale(1000, (1, 1000), (1, 1000));
        assert_eq!(ts, 1000);
    }

    #[test]
    fn test_rescale_ms_to_90khz() {
        // 1000 ms -> 90 000 ticks at 90 kHz
        let ts = pts_map::rescale(1000, (1, 1000), (1, 90_000));
        assert_eq!(ts, 90_000);
    }

    #[test]
    fn test_rescale_90khz_to_ms() {
        let ts = pts_map::rescale(90_000, (1, 90_000), (1, 1000));
        assert_eq!(ts, 1000);
    }

    #[test]
    fn test_dts_before_pts_valid() {
        assert!(pts_map::dts_before_pts(990, 1000));
        assert!(pts_map::dts_before_pts(1000, 1000));
    }

    #[test]
    fn test_dts_before_pts_invalid() {
        assert!(!pts_map::dts_before_pts(1001, 1000));
    }

    #[test]
    fn test_interleaver_multi_stream() {
        let mut il = Interleaver::new(20);
        // Simulate two streams: video (stream 0) and audio (stream 1)
        for i in (0..5).map(|x| x * 40) {
            il.push(pkt(0, i, i == 0)); // video keyframe at 0
        }
        for i in (0..10).map(|x| x * 20) {
            il.push(pkt(1, i, true)); // audio every 20 units
        }
        let drained = il.drain();
        // First packet must have pts == 0
        assert_eq!(drained[0].pts, 0);
        // Packets must be in non-decreasing order
        for w in drained.windows(2) {
            assert!(
                w[0].pts <= w[1].pts,
                "out of order: {} > {}",
                w[0].pts,
                w[1].pts
            );
        }
    }
}
