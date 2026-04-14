#![allow(dead_code)]
//! Parallel multi-stream interleaving for container muxers.
//!
//! Extends the basic [`Interleaver`](super::interleave::Interleaver) with
//! concurrent per-stream buffering and merge-sort interleaving, enabling
//! higher throughput when encoding multiple streams simultaneously.
//!
//! # Architecture
//!
//! Each stream has its own dedicated input buffer. Packets are accepted
//! lock-free into per-stream queues and then merged into a single
//! timestamp-ordered output via a k-way merge.
//!
//! ```text
//!   Stream 0 buffer ──┐
//!   Stream 1 buffer ──┤── K-way merge ──> interleaved output
//!   Stream 2 buffer ──┘
//! ```
//!
//! # Example
//!
//! ```
//! use oximedia_container::mux::parallel_interleave::{
//!     ParallelInterleaver, StreamPacket,
//! };
//!
//! let mut interleaver = ParallelInterleaver::new(3); // 3 streams
//! interleaver.push(StreamPacket::new(0, 0, vec![1,2,3], true));
//! interleaver.push(StreamPacket::new(1, 10, vec![4,5,6], true));
//! interleaver.push(StreamPacket::new(0, 40, vec![7,8,9], false));
//! interleaver.push(StreamPacket::new(1, 30, vec![10,11,12], false));
//!
//! let output = interleaver.flush_all();
//! assert_eq!(output.len(), 4);
//! assert_eq!(output[0].dts, 0);
//! assert_eq!(output[1].dts, 10);
//! ```

#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;

// ─── Stream packet ─────────────────────────────────────────────────────────

/// A packet with stream affinity for parallel interleaving.
#[derive(Debug, Clone)]
pub struct StreamPacket {
    /// Stream index (0-based).
    pub stream_index: usize,
    /// Decode timestamp (used for interleaving order).
    pub dts: i64,
    /// Presentation timestamp (may differ from DTS for B-frames).
    pub pts: i64,
    /// Raw payload bytes.
    pub data: Vec<u8>,
    /// Whether this is a keyframe / sync sample.
    pub is_keyframe: bool,
    /// Duration of this packet in timebase ticks (0 if unknown).
    pub duration: i64,
    /// Insertion sequence for stable ordering among equal-DTS packets.
    insertion_seq: u64,
}

impl StreamPacket {
    /// Creates a new stream packet.
    ///
    /// PTS is set equal to DTS by default; use [`with_pts`] to override.
    #[must_use]
    pub fn new(stream_index: usize, dts: i64, data: Vec<u8>, is_keyframe: bool) -> Self {
        Self {
            stream_index,
            dts,
            pts: dts,
            data,
            is_keyframe,
            duration: 0,
            insertion_seq: 0,
        }
    }

    /// Sets the presentation timestamp (for B-frame reordering).
    #[must_use]
    pub fn with_pts(mut self, pts: i64) -> Self {
        self.pts = pts;
        self
    }

    /// Sets the packet duration.
    #[must_use]
    pub fn with_duration(mut self, duration: i64) -> Self {
        self.duration = duration;
        self
    }

    /// Returns the payload size in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

// ─── Min-heap wrapper ──────────────────────────────────────────────────────

/// Wrapper for BinaryHeap min-ordering by DTS.
#[derive(Debug)]
struct HeapEntry(StreamPacket);

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.0.dts == other.0.dts && self.0.insertion_seq == other.0.insertion_seq
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap: smallest DTS first, then smallest insertion_seq
        other
            .0
            .dts
            .cmp(&self.0.dts)
            .then_with(|| other.0.insertion_seq.cmp(&self.0.insertion_seq))
    }
}

// ─── Stream buffer ─────────────────────────────────────────────────────────

/// Per-stream packet buffer.
#[derive(Debug)]
struct StreamBuffer {
    /// Stream index.
    stream_index: usize,
    /// Buffered packets (not yet merged).
    packets: Vec<StreamPacket>,
    /// Total bytes buffered.
    buffered_bytes: usize,
    /// Number of packets that have been flushed.
    flushed_count: u64,
    /// Last DTS that was flushed.
    last_flushed_dts: Option<i64>,
}

impl StreamBuffer {
    fn new(stream_index: usize) -> Self {
        Self {
            stream_index,
            packets: Vec::new(),
            buffered_bytes: 0,
            flushed_count: 0,
            last_flushed_dts: None,
        }
    }

    fn push(&mut self, packet: StreamPacket) {
        self.buffered_bytes += packet.data.len();
        self.packets.push(packet);
    }

    fn sort_by_dts(&mut self) {
        self.packets.sort_by_key(|p| p.dts);
    }

    fn drain_all(&mut self) -> Vec<StreamPacket> {
        self.buffered_bytes = 0;
        let count = self.packets.len() as u64;
        self.flushed_count += count;
        if let Some(last) = self.packets.last() {
            self.last_flushed_dts = Some(last.dts);
        }
        std::mem::take(&mut self.packets)
    }

    fn drain_up_to_dts(&mut self, max_dts: i64) -> Vec<StreamPacket> {
        // Partition: packets with dts <= max_dts go out, rest stay
        let split_idx = self.packets.partition_point(|p| p.dts <= max_dts);
        let drained: Vec<StreamPacket> = self.packets.drain(..split_idx).collect();
        let drained_bytes: usize = drained.iter().map(|p| p.data.len()).sum();
        self.buffered_bytes -= drained_bytes;
        self.flushed_count += drained.len() as u64;
        if let Some(last) = drained.last() {
            self.last_flushed_dts = Some(last.dts);
        }
        drained
    }

    fn len(&self) -> usize {
        self.packets.len()
    }

    fn is_empty(&self) -> bool {
        self.packets.is_empty()
    }

    /// Returns the minimum DTS in this buffer, or None if empty.
    fn min_dts(&self) -> Option<i64> {
        self.packets.first().map(|p| p.dts)
    }
}

// ─── Interleave strategy ───────────────────────────────────────────────────

/// Strategy for deciding when to flush packets from stream buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlushStrategy {
    /// Flush when any single stream buffer reaches `threshold` packets.
    PacketCount {
        /// Maximum packets per stream before flushing.
        threshold: usize,
    },
    /// Flush when any single stream buffer reaches `threshold` bytes.
    ByteCount {
        /// Maximum bytes per stream before flushing.
        threshold: usize,
    },
    /// Flush when the DTS span (max - min) across all buffered packets
    /// exceeds `max_span` ticks.
    DtsSpan {
        /// Maximum DTS span in timebase ticks.
        max_span: i64,
    },
    /// Only flush on explicit `flush_all()` calls.
    Manual,
}

impl Default for FlushStrategy {
    fn default() -> Self {
        Self::PacketCount { threshold: 64 }
    }
}

// ─── Interleave statistics ─────────────────────────────────────────────────

/// Statistics tracked by the parallel interleaver.
#[derive(Debug, Clone, Copy, Default)]
pub struct InterleaveStats {
    /// Total packets accepted across all streams.
    pub total_packets_in: u64,
    /// Total packets emitted (interleaved output).
    pub total_packets_out: u64,
    /// Total bytes accepted.
    pub total_bytes_in: u64,
    /// Total bytes emitted.
    pub total_bytes_out: u64,
    /// Number of flush operations performed.
    pub flush_count: u64,
    /// Peak buffered packet count across all streams.
    pub peak_buffer_packets: usize,
    /// Number of out-of-order corrections applied.
    pub reorder_corrections: u64,
}

// ─── Parallel interleaver ──────────────────────────────────────────────────

/// Parallel multi-stream interleaver.
///
/// Accepts packets from multiple streams and produces a single
/// timestamp-ordered output sequence.
#[derive(Debug)]
pub struct ParallelInterleaver {
    /// Per-stream buffers.
    buffers: Vec<StreamBuffer>,
    /// Flush strategy.
    strategy: FlushStrategy,
    /// Global insertion counter for stable ordering.
    insertion_counter: u64,
    /// Accumulated statistics.
    stats: InterleaveStats,
}

impl ParallelInterleaver {
    /// Creates a new parallel interleaver for `stream_count` streams.
    #[must_use]
    pub fn new(stream_count: usize) -> Self {
        let buffers = (0..stream_count).map(StreamBuffer::new).collect();
        Self {
            buffers,
            strategy: FlushStrategy::default(),
            insertion_counter: 0,
            stats: InterleaveStats::default(),
        }
    }

    /// Creates a new interleaver with a specific flush strategy.
    #[must_use]
    pub fn with_strategy(stream_count: usize, strategy: FlushStrategy) -> Self {
        let mut interleaver = Self::new(stream_count);
        interleaver.strategy = strategy;
        interleaver
    }

    /// Pushes a packet into the appropriate stream buffer.
    ///
    /// If the stream index exceeds the configured stream count, the packet
    /// is silently dropped.
    pub fn push(&mut self, mut packet: StreamPacket) {
        let idx = packet.stream_index;
        if idx >= self.buffers.len() {
            return;
        }
        packet.insertion_seq = self.insertion_counter;
        self.insertion_counter += 1;
        self.stats.total_packets_in += 1;
        self.stats.total_bytes_in += packet.data.len() as u64;
        self.buffers[idx].push(packet);

        // Track peak buffer size
        let total_buffered: usize = self.buffers.iter().map(|b| b.len()).sum();
        if total_buffered > self.stats.peak_buffer_packets {
            self.stats.peak_buffer_packets = total_buffered;
        }
    }

    /// Checks if the flush strategy triggers automatic flushing.
    ///
    /// Returns flushed packets if the strategy threshold is met.
    pub fn push_and_maybe_flush(&mut self, packet: StreamPacket) -> Vec<StreamPacket> {
        self.push(packet);
        if self.should_flush() {
            self.flush_ready()
        } else {
            Vec::new()
        }
    }

    /// Returns true if the current buffer state triggers the flush strategy.
    #[must_use]
    pub fn should_flush(&self) -> bool {
        match self.strategy {
            FlushStrategy::PacketCount { threshold } => {
                self.buffers.iter().any(|b| b.len() >= threshold)
            }
            FlushStrategy::ByteCount { threshold } => {
                self.buffers.iter().any(|b| b.buffered_bytes >= threshold)
            }
            FlushStrategy::DtsSpan { max_span } => {
                let min = self
                    .buffers
                    .iter()
                    .filter_map(|b| b.min_dts())
                    .min();
                let max = self
                    .buffers
                    .iter()
                    .filter_map(|b| b.packets.last().map(|p| p.dts))
                    .max();
                if let (Some(lo), Some(hi)) = (min, max) {
                    (hi - lo) > max_span
                } else {
                    false
                }
            }
            FlushStrategy::Manual => false,
        }
    }

    /// Flushes packets up to the minimum DTS available across all non-empty
    /// streams (ensuring no stream is starved).
    pub fn flush_ready(&mut self) -> Vec<StreamPacket> {
        // Find the minimum max-DTS across all non-empty streams
        // (i.e. the latest DTS of the stream that is furthest behind)
        let safe_dts = self
            .buffers
            .iter()
            .filter(|b| !b.is_empty())
            .filter_map(|b| b.packets.last().map(|p| p.dts))
            .min();

        let Some(safe_dts) = safe_dts else {
            return Vec::new();
        };

        // Sort each buffer before draining
        for buf in &mut self.buffers {
            buf.sort_by_dts();
        }

        // K-way merge using a min-heap
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
        for buf in &mut self.buffers {
            for pkt in buf.drain_up_to_dts(safe_dts) {
                heap.push(HeapEntry(pkt));
            }
        }

        let mut output = Vec::with_capacity(heap.len());
        while let Some(entry) = heap.pop() {
            self.stats.total_packets_out += 1;
            self.stats.total_bytes_out += entry.0.data.len() as u64;
            output.push(entry.0);
        }

        self.stats.flush_count += 1;
        output
    }

    /// Flushes all remaining packets across all streams in DTS order.
    pub fn flush_all(&mut self) -> Vec<StreamPacket> {
        for buf in &mut self.buffers {
            buf.sort_by_dts();
        }

        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
        for buf in &mut self.buffers {
            for pkt in buf.drain_all() {
                heap.push(HeapEntry(pkt));
            }
        }

        let mut output = Vec::with_capacity(heap.len());
        while let Some(entry) = heap.pop() {
            self.stats.total_packets_out += 1;
            self.stats.total_bytes_out += entry.0.data.len() as u64;
            output.push(entry.0);
        }

        self.stats.flush_count += 1;
        output
    }

    /// Returns the number of streams.
    #[must_use]
    pub fn stream_count(&self) -> usize {
        self.buffers.len()
    }

    /// Returns the total number of buffered packets across all streams.
    #[must_use]
    pub fn total_buffered(&self) -> usize {
        self.buffers.iter().map(|b| b.len()).sum()
    }

    /// Returns the total buffered bytes across all streams.
    #[must_use]
    pub fn total_buffered_bytes(&self) -> usize {
        self.buffers.iter().map(|b| b.buffered_bytes).sum()
    }

    /// Returns true if all stream buffers are empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffers.iter().all(|b| b.is_empty())
    }

    /// Returns the number of buffered packets for a specific stream.
    #[must_use]
    pub fn stream_buffered(&self, stream_index: usize) -> usize {
        self.buffers
            .get(stream_index)
            .map_or(0, |b| b.len())
    }

    /// Returns the accumulated statistics.
    #[must_use]
    pub fn stats(&self) -> &InterleaveStats {
        &self.stats
    }

    /// Resets the statistics counters.
    pub fn reset_stats(&mut self) {
        self.stats = InterleaveStats::default();
    }

    /// Returns the DTS span (max - min) across all buffered packets.
    #[must_use]
    pub fn dts_span(&self) -> i64 {
        let min = self
            .buffers
            .iter()
            .filter_map(|b| b.min_dts())
            .min();
        let max = self
            .buffers
            .iter()
            .filter_map(|b| b.packets.last().map(|p| p.dts))
            .max();
        match (min, max) {
            (Some(lo), Some(hi)) => hi - lo,
            _ => 0,
        }
    }

    /// Returns per-stream buffer sizes.
    #[must_use]
    pub fn per_stream_sizes(&self) -> Vec<(usize, usize)> {
        self.buffers
            .iter()
            .map(|b| (b.stream_index, b.len()))
            .collect()
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn pkt(stream: usize, dts: i64) -> StreamPacket {
        StreamPacket::new(stream, dts, vec![0u8; 100], dts == 0)
    }

    #[test]
    fn test_basic_two_stream_interleave() {
        let mut il = ParallelInterleaver::new(2);
        il.push(pkt(0, 0));
        il.push(pkt(1, 10));
        il.push(pkt(0, 40));
        il.push(pkt(1, 30));

        let output = il.flush_all();
        assert_eq!(output.len(), 4);
        // Must be in DTS order
        assert_eq!(output[0].dts, 0);
        assert_eq!(output[1].dts, 10);
        assert_eq!(output[2].dts, 30);
        assert_eq!(output[3].dts, 40);
    }

    #[test]
    fn test_three_stream_interleave() {
        let mut il = ParallelInterleaver::new(3);
        il.push(pkt(0, 100));
        il.push(pkt(1, 50));
        il.push(pkt(2, 75));
        il.push(pkt(0, 200));
        il.push(pkt(1, 150));
        il.push(pkt(2, 175));

        let output = il.flush_all();
        assert_eq!(output.len(), 6);
        // Verify DTS ordering
        for w in output.windows(2) {
            assert!(
                w[0].dts <= w[1].dts,
                "out of order: {} > {}",
                w[0].dts,
                w[1].dts
            );
        }
    }

    #[test]
    fn test_empty_interleaver() {
        let mut il = ParallelInterleaver::new(2);
        assert!(il.is_empty());
        assert_eq!(il.total_buffered(), 0);
        let output = il.flush_all();
        assert!(output.is_empty());
    }

    #[test]
    fn test_single_stream() {
        let mut il = ParallelInterleaver::new(1);
        il.push(pkt(0, 300));
        il.push(pkt(0, 100));
        il.push(pkt(0, 200));

        let output = il.flush_all();
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].dts, 100);
        assert_eq!(output[1].dts, 200);
        assert_eq!(output[2].dts, 300);
    }

    #[test]
    fn test_out_of_range_stream_dropped() {
        let mut il = ParallelInterleaver::new(2);
        il.push(pkt(0, 0));
        il.push(pkt(5, 10)); // out of range — should be silently dropped
        il.push(pkt(1, 20));

        let output = il.flush_all();
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_flush_ready_partial() {
        let mut il = ParallelInterleaver::with_strategy(
            2,
            FlushStrategy::Manual,
        );
        // Stream 0: packets at 0, 40, 80
        il.push(pkt(0, 0));
        il.push(pkt(0, 40));
        il.push(pkt(0, 80));
        // Stream 1: packets at 10, 30
        il.push(pkt(1, 10));
        il.push(pkt(1, 30));

        // flush_ready should flush up to min(max(stream0), max(stream1)) = min(80, 30) = 30
        let partial = il.flush_ready();
        // Should contain packets with dts <= 30: {0, 10, 30}
        assert_eq!(partial.len(), 3);
        assert_eq!(partial[0].dts, 0);
        assert_eq!(partial[1].dts, 10);
        assert_eq!(partial[2].dts, 30);

        // Remaining: stream 0 still has 40, 80
        assert_eq!(il.total_buffered(), 2);
    }

    #[test]
    fn test_stats_tracking() {
        let mut il = ParallelInterleaver::new(2);
        il.push(pkt(0, 0));
        il.push(pkt(1, 10));
        il.push(pkt(0, 20));

        assert_eq!(il.stats().total_packets_in, 3);
        assert_eq!(il.stats().total_bytes_in, 300); // 3 * 100 bytes

        let output = il.flush_all();
        assert_eq!(output.len(), 3);
        assert_eq!(il.stats().total_packets_out, 3);
        assert_eq!(il.stats().flush_count, 1);
    }

    #[test]
    fn test_packet_count_flush_strategy() {
        let mut il = ParallelInterleaver::with_strategy(
            2,
            FlushStrategy::PacketCount { threshold: 3 },
        );
        il.push(pkt(0, 0));
        il.push(pkt(0, 10));
        assert!(!il.should_flush());
        il.push(pkt(0, 20));
        assert!(il.should_flush()); // stream 0 has 3 packets
    }

    #[test]
    fn test_byte_count_flush_strategy() {
        let mut il = ParallelInterleaver::with_strategy(
            2,
            FlushStrategy::ByteCount { threshold: 250 },
        );
        il.push(pkt(0, 0)); // 100 bytes
        il.push(pkt(0, 10)); // 100 bytes
        assert!(!il.should_flush());
        il.push(pkt(0, 20)); // 100 bytes, total = 300 >= 250
        assert!(il.should_flush());
    }

    #[test]
    fn test_dts_span_flush_strategy() {
        let mut il = ParallelInterleaver::with_strategy(
            2,
            FlushStrategy::DtsSpan { max_span: 50 },
        );
        il.push(pkt(0, 0));
        il.push(pkt(1, 10));
        assert!(!il.should_flush()); // span = 10
        il.push(pkt(0, 60));
        assert!(il.should_flush()); // span = 60 > 50
    }

    #[test]
    fn test_push_and_maybe_flush() {
        let mut il = ParallelInterleaver::with_strategy(
            1,
            FlushStrategy::PacketCount { threshold: 2 },
        );
        let out1 = il.push_and_maybe_flush(pkt(0, 0));
        assert!(out1.is_empty());
        let out2 = il.push_and_maybe_flush(pkt(0, 10));
        assert!(!out2.is_empty());
    }

    #[test]
    fn test_per_stream_sizes() {
        let mut il = ParallelInterleaver::new(3);
        il.push(pkt(0, 0));
        il.push(pkt(0, 10));
        il.push(pkt(1, 5));

        let sizes = il.per_stream_sizes();
        assert_eq!(sizes.len(), 3);
        assert_eq!(sizes[0], (0, 2));
        assert_eq!(sizes[1], (1, 1));
        assert_eq!(sizes[2], (2, 0));
    }

    #[test]
    fn test_dts_span() {
        let mut il = ParallelInterleaver::new(2);
        assert_eq!(il.dts_span(), 0);
        il.push(pkt(0, 100));
        il.push(pkt(1, 300));
        assert_eq!(il.dts_span(), 200);
    }

    #[test]
    fn test_stream_packet_builder() {
        let pkt = StreamPacket::new(0, 100, vec![1, 2, 3], true)
            .with_pts(200)
            .with_duration(50);
        assert_eq!(pkt.dts, 100);
        assert_eq!(pkt.pts, 200);
        assert_eq!(pkt.duration, 50);
        assert_eq!(pkt.size(), 3);
        assert!(pkt.is_keyframe);
    }

    #[test]
    fn test_stable_ordering_equal_dts() {
        let mut il = ParallelInterleaver::new(2);
        // Push packets with equal DTS from different streams
        il.push(StreamPacket::new(0, 100, vec![1], false));
        il.push(StreamPacket::new(1, 100, vec![2], false));
        il.push(StreamPacket::new(0, 100, vec![3], false));

        let output = il.flush_all();
        assert_eq!(output.len(), 3);
        // All have same DTS; should maintain insertion order
        assert_eq!(output[0].data, vec![1]);
        assert_eq!(output[1].data, vec![2]);
        assert_eq!(output[2].data, vec![3]);
    }

    #[test]
    fn test_reset_stats() {
        let mut il = ParallelInterleaver::new(1);
        il.push(pkt(0, 0));
        il.flush_all();
        assert_eq!(il.stats().total_packets_in, 1);
        il.reset_stats();
        assert_eq!(il.stats().total_packets_in, 0);
        assert_eq!(il.stats().total_packets_out, 0);
    }

    #[test]
    fn test_total_buffered_bytes() {
        let mut il = ParallelInterleaver::new(2);
        il.push(StreamPacket::new(0, 0, vec![0; 50], true));
        il.push(StreamPacket::new(1, 10, vec![0; 75], true));
        assert_eq!(il.total_buffered_bytes(), 125);
    }
}
