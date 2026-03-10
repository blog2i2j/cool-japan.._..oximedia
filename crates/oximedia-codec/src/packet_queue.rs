#![allow(dead_code)]
//! Codec packet queuing and reordering.
//!
//! Video codecs often produce packets out of display order (B-frames cause
//! decode order != presentation order). This module provides a priority queue
//! that reorders coded packets by PTS (presentation timestamp) or DTS (decode
//! timestamp) before they are fed to a muxer or consumer.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A coded packet with timestamps for queue ordering.
#[derive(Clone, Debug)]
pub struct QueuedPacket {
    /// Presentation timestamp in timebase units.
    pub pts: i64,
    /// Decode timestamp in timebase units.
    pub dts: i64,
    /// Duration in timebase units.
    pub duration: u64,
    /// Stream index this packet belongs to.
    pub stream_index: u32,
    /// Whether this is a keyframe.
    pub is_keyframe: bool,
    /// Raw payload data.
    pub data: Vec<u8>,
    /// Sequence counter for stable sorting when timestamps tie.
    sequence: u64,
}

impl QueuedPacket {
    /// Create a new queued packet.
    pub fn new(pts: i64, dts: i64, data: Vec<u8>) -> Self {
        Self {
            pts,
            dts,
            duration: 0,
            stream_index: 0,
            is_keyframe: false,
            data,
            sequence: 0,
        }
    }

    /// Set the duration.
    pub fn with_duration(mut self, duration: u64) -> Self {
        self.duration = duration;
        self
    }

    /// Set the stream index.
    pub fn with_stream_index(mut self, index: u32) -> Self {
        self.stream_index = index;
        self
    }

    /// Set the keyframe flag.
    pub fn with_keyframe(mut self, is_keyframe: bool) -> Self {
        self.is_keyframe = is_keyframe;
        self
    }

    /// Packet size in bytes.
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

impl PartialEq for QueuedPacket {
    fn eq(&self, other: &Self) -> bool {
        self.pts == other.pts && self.sequence == other.sequence
    }
}

impl Eq for QueuedPacket {}

/// Ordering strategy for packet queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueOrder {
    /// Order by PTS (presentation timestamp) ascending.
    Pts,
    /// Order by DTS (decode timestamp) ascending.
    Dts,
}

impl Default for QueueOrder {
    fn default() -> Self {
        Self::Pts
    }
}

/// Wrapper for min-heap ordering (BinaryHeap is max-heap by default).
struct MinPacket {
    packet: QueuedPacket,
    order: QueueOrder,
}

impl PartialEq for MinPacket {
    fn eq(&self, other: &Self) -> bool {
        self.packet.eq(&other.packet)
    }
}

impl Eq for MinPacket {}

impl PartialOrd for MinPacket {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinPacket {
    fn cmp(&self, other: &Self) -> Ordering {
        let self_ts = match self.order {
            QueueOrder::Pts => self.packet.pts,
            QueueOrder::Dts => self.packet.dts,
        };
        let other_ts = match self.order {
            QueueOrder::Pts => other.packet.pts,
            QueueOrder::Dts => other.packet.dts,
        };
        // Reverse for min-heap behavior
        other_ts
            .cmp(&self_ts)
            .then_with(|| other.packet.sequence.cmp(&self.packet.sequence))
    }
}

/// Configuration for the packet queue.
#[derive(Clone, Debug)]
pub struct PacketQueueConfig {
    /// Maximum number of packets to buffer.
    pub max_packets: usize,
    /// Maximum total byte size to buffer.
    pub max_bytes: usize,
    /// Ordering strategy.
    pub order: QueueOrder,
}

impl Default for PacketQueueConfig {
    fn default() -> Self {
        Self {
            max_packets: 256,
            max_bytes: 64 * 1024 * 1024,
            order: QueueOrder::Pts,
        }
    }
}

/// Statistics for the packet queue.
#[derive(Clone, Debug, Default)]
pub struct QueueStats {
    /// Total packets enqueued.
    pub total_enqueued: u64,
    /// Total packets dequeued.
    pub total_dequeued: u64,
    /// Total packets dropped due to overflow.
    pub total_dropped: u64,
    /// Total bytes enqueued.
    pub total_bytes_in: u64,
    /// Total bytes dequeued.
    pub total_bytes_out: u64,
}

impl QueueStats {
    /// Number of packets currently in the queue.
    pub fn pending(&self) -> u64 {
        self.total_enqueued - self.total_dequeued - self.total_dropped
    }
}

/// A reordering packet queue.
///
/// Packets are inserted in arbitrary order and extracted in PTS or DTS order.
pub struct PacketQueue {
    heap: BinaryHeap<MinPacket>,
    config: PacketQueueConfig,
    total_bytes: usize,
    sequence_counter: u64,
    stats: QueueStats,
}

impl PacketQueue {
    /// Create a new packet queue with default configuration.
    pub fn new() -> Self {
        Self::with_config(PacketQueueConfig::default())
    }

    /// Create a new packet queue with custom configuration.
    pub fn with_config(config: PacketQueueConfig) -> Self {
        Self {
            heap: BinaryHeap::new(),
            config,
            total_bytes: 0,
            sequence_counter: 0,
            stats: QueueStats::default(),
        }
    }

    /// Push a packet into the queue. Returns true if accepted, false if dropped.
    pub fn push(&mut self, mut packet: QueuedPacket) -> bool {
        let pkt_size = packet.size();
        if self.heap.len() >= self.config.max_packets
            || self.total_bytes + pkt_size > self.config.max_bytes
        {
            self.stats.total_dropped += 1;
            return false;
        }
        packet.sequence = self.sequence_counter;
        self.sequence_counter += 1;
        self.total_bytes += pkt_size;
        self.stats.total_enqueued += 1;
        self.stats.total_bytes_in += pkt_size as u64;
        self.heap.push(MinPacket {
            packet,
            order: self.config.order,
        });
        true
    }

    /// Pop the next packet in timestamp order.
    pub fn pop(&mut self) -> Option<QueuedPacket> {
        let min_pkt = self.heap.pop()?;
        let pkt = min_pkt.packet;
        self.total_bytes -= pkt.size();
        self.stats.total_dequeued += 1;
        self.stats.total_bytes_out += pkt.size() as u64;
        Some(pkt)
    }

    /// Peek at the next packet without removing it.
    pub fn peek_pts(&self) -> Option<i64> {
        self.heap.peek().map(|p| p.packet.pts)
    }

    /// Number of packets in the queue.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Total bytes buffered.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Queue statistics.
    pub fn stats(&self) -> &QueueStats {
        &self.stats
    }

    /// Drain all packets in timestamp order.
    pub fn drain(&mut self) -> Vec<QueuedPacket> {
        let mut out = Vec::with_capacity(self.heap.len());
        while let Some(pkt) = self.pop() {
            out.push(pkt);
        }
        out
    }

    /// Clear the queue.
    pub fn clear(&mut self) {
        self.heap.clear();
        self.total_bytes = 0;
    }
}

impl Default for PacketQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queued_packet_new() {
        let pkt = QueuedPacket::new(100, 90, vec![1, 2, 3]);
        assert_eq!(pkt.pts, 100);
        assert_eq!(pkt.dts, 90);
        assert_eq!(pkt.size(), 3);
        assert!(!pkt.is_keyframe);
    }

    #[test]
    fn test_queued_packet_builder() {
        let pkt = QueuedPacket::new(10, 5, vec![0; 10])
            .with_duration(33)
            .with_stream_index(1)
            .with_keyframe(true);
        assert_eq!(pkt.duration, 33);
        assert_eq!(pkt.stream_index, 1);
        assert!(pkt.is_keyframe);
    }

    #[test]
    fn test_queue_order_default() {
        assert_eq!(QueueOrder::default(), QueueOrder::Pts);
    }

    #[test]
    fn test_empty_queue() {
        let queue = PacketQueue::new();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
        assert_eq!(queue.total_bytes(), 0);
    }

    #[test]
    fn test_push_and_pop_single() {
        let mut queue = PacketQueue::new();
        let pkt = QueuedPacket::new(100, 100, vec![42]);
        assert!(queue.push(pkt));
        assert_eq!(queue.len(), 1);

        let out = queue.pop().expect("pop should return item");
        assert_eq!(out.pts, 100);
        assert_eq!(out.data, vec![42]);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_pts_ordering() {
        let mut queue = PacketQueue::new();
        queue.push(QueuedPacket::new(300, 300, vec![3]));
        queue.push(QueuedPacket::new(100, 100, vec![1]));
        queue.push(QueuedPacket::new(200, 200, vec![2]));

        assert_eq!(queue.pop().expect("pop should return item").pts, 100);
        assert_eq!(queue.pop().expect("pop should return item").pts, 200);
        assert_eq!(queue.pop().expect("pop should return item").pts, 300);
    }

    #[test]
    fn test_dts_ordering() {
        let config = PacketQueueConfig {
            order: QueueOrder::Dts,
            ..Default::default()
        };
        let mut queue = PacketQueue::with_config(config);
        queue.push(QueuedPacket::new(300, 200, vec![2]));
        queue.push(QueuedPacket::new(100, 100, vec![1]));
        queue.push(QueuedPacket::new(200, 300, vec![3]));

        assert_eq!(queue.pop().expect("pop should return item").dts, 100);
        assert_eq!(queue.pop().expect("pop should return item").dts, 200);
        assert_eq!(queue.pop().expect("pop should return item").dts, 300);
    }

    #[test]
    fn test_max_packets_overflow() {
        let config = PacketQueueConfig {
            max_packets: 2,
            ..Default::default()
        };
        let mut queue = PacketQueue::with_config(config);
        assert!(queue.push(QueuedPacket::new(1, 1, vec![1])));
        assert!(queue.push(QueuedPacket::new(2, 2, vec![2])));
        assert!(!queue.push(QueuedPacket::new(3, 3, vec![3])));
        assert_eq!(queue.stats().total_dropped, 1);
    }

    #[test]
    fn test_max_bytes_overflow() {
        let config = PacketQueueConfig {
            max_bytes: 5,
            ..Default::default()
        };
        let mut queue = PacketQueue::with_config(config);
        assert!(queue.push(QueuedPacket::new(1, 1, vec![0; 3])));
        assert!(!queue.push(QueuedPacket::new(2, 2, vec![0; 3])));
        assert_eq!(queue.total_bytes(), 3);
    }

    #[test]
    fn test_drain() {
        let mut queue = PacketQueue::new();
        queue.push(QueuedPacket::new(30, 30, vec![3]));
        queue.push(QueuedPacket::new(10, 10, vec![1]));
        queue.push(QueuedPacket::new(20, 20, vec![2]));

        let drained = queue.drain();
        assert_eq!(drained.len(), 3);
        assert_eq!(drained[0].pts, 10);
        assert_eq!(drained[1].pts, 20);
        assert_eq!(drained[2].pts, 30);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_peek_pts() {
        let mut queue = PacketQueue::new();
        assert!(queue.peek_pts().is_none());
        queue.push(QueuedPacket::new(50, 50, vec![]));
        queue.push(QueuedPacket::new(10, 10, vec![]));
        assert_eq!(queue.peek_pts(), Some(10));
    }

    #[test]
    fn test_stats() {
        let mut queue = PacketQueue::new();
        queue.push(QueuedPacket::new(1, 1, vec![0; 10]));
        queue.push(QueuedPacket::new(2, 2, vec![0; 20]));
        let _ = queue.pop();
        let stats = queue.stats();
        assert_eq!(stats.total_enqueued, 2);
        assert_eq!(stats.total_dequeued, 1);
        assert_eq!(stats.total_bytes_in, 30);
        assert_eq!(stats.total_bytes_out, 10);
        assert_eq!(stats.pending(), 1);
    }

    #[test]
    fn test_clear() {
        let mut queue = PacketQueue::new();
        queue.push(QueuedPacket::new(1, 1, vec![0; 100]));
        queue.push(QueuedPacket::new(2, 2, vec![0; 200]));
        queue.clear();
        assert!(queue.is_empty());
        assert_eq!(queue.total_bytes(), 0);
    }
}
