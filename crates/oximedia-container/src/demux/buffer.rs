//! Packet buffering for demuxers.
//!
//! Provides per-stream packet queues with seek support and configurable
//! maximum buffer depths.

#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};

/// A buffered packet for a single stream.
#[derive(Debug, Clone)]
pub struct BufferedPacket {
    /// Stream index this packet belongs to.
    pub stream_index: usize,
    /// Presentation timestamp.
    pub pts: i64,
    /// Optional decode timestamp.
    pub dts: Option<i64>,
    /// Raw compressed data.
    pub data: Vec<u8>,
    /// Whether this packet contains a keyframe.
    pub is_keyframe: bool,
    /// Duration in stream timebase units.
    pub duration: Option<i64>,
}

impl BufferedPacket {
    /// Create a new buffered packet.
    #[must_use]
    pub fn new(stream_index: usize, pts: i64, data: Vec<u8>, is_keyframe: bool) -> Self {
        Self {
            stream_index,
            pts,
            dts: None,
            data,
            is_keyframe,
            duration: None,
        }
    }
}

/// Per-stream packet queue.
struct StreamQueue {
    queue: VecDeque<BufferedPacket>,
    max_depth: usize,
}

impl StreamQueue {
    fn new(max_depth: usize) -> Self {
        Self {
            queue: VecDeque::new(),
            max_depth: max_depth.max(1),
        }
    }

    /// Push a packet; returns evicted packet if the queue was full.
    fn push(&mut self, pkt: BufferedPacket) -> Option<BufferedPacket> {
        let evicted = if self.queue.len() >= self.max_depth {
            self.queue.pop_front()
        } else {
            None
        };
        self.queue.push_back(pkt);
        evicted
    }

    fn pop(&mut self) -> Option<BufferedPacket> {
        self.queue.pop_front()
    }

    fn peek(&self) -> Option<&BufferedPacket> {
        self.queue.front()
    }

    fn len(&self) -> usize {
        self.queue.len()
    }

    fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    fn clear(&mut self) {
        self.queue.clear();
    }

    /// Discard all packets whose PTS < `target_pts`.
    fn discard_before(&mut self, target_pts: i64) {
        while let Some(front) = self.queue.front() {
            if front.pts < target_pts {
                self.queue.pop_front();
            } else {
                break;
            }
        }
    }

    /// Find the position of the first keyframe at or after `target_pts`.
    fn first_keyframe_pos_from(&self, target_pts: i64) -> Option<usize> {
        self.queue
            .iter()
            .enumerate()
            .find(|(_, p)| p.pts >= target_pts && p.is_keyframe)
            .map(|(i, _)| i)
    }
}

/// Multi-stream packet buffer supporting seek operations.
pub struct PacketBuffer {
    streams: HashMap<usize, StreamQueue>,
    default_depth: usize,
}

impl PacketBuffer {
    /// Create a new buffer with the given per-stream maximum depth.
    #[must_use]
    pub fn new(default_depth: usize) -> Self {
        Self {
            streams: HashMap::new(),
            default_depth,
        }
    }

    /// Push a packet into the appropriate stream queue.
    ///
    /// Returns any evicted packet (when the queue was at capacity).
    pub fn push(&mut self, pkt: BufferedPacket) -> Option<BufferedPacket> {
        let depth = self.default_depth;
        self.streams
            .entry(pkt.stream_index)
            .or_insert_with(|| StreamQueue::new(depth))
            .push(pkt)
    }

    /// Pop the next packet from a specific stream.
    pub fn pop_stream(&mut self, stream_index: usize) -> Option<BufferedPacket> {
        self.streams.get_mut(&stream_index)?.pop()
    }

    /// Pop the packet with the lowest PTS across all streams.
    pub fn pop_lowest_pts(&mut self) -> Option<BufferedPacket> {
        let stream_idx = self
            .streams
            .iter()
            .filter_map(|(idx, q)| q.peek().map(|p| (*idx, p.pts)))
            .min_by_key(|&(_, pts)| pts)
            .map(|(idx, _)| idx)?;
        self.streams.get_mut(&stream_idx)?.pop()
    }

    /// Flush all buffers (e.g., after a seek).
    pub fn flush(&mut self) {
        for q in self.streams.values_mut() {
            q.clear();
        }
    }

    /// Perform a seek: discard packets before `target_pts` on all streams.
    pub fn seek_to_pts(&mut self, target_pts: i64) {
        for q in self.streams.values_mut() {
            q.discard_before(target_pts);
        }
    }

    /// Total number of buffered packets across all streams.
    #[must_use]
    pub fn total_buffered(&self) -> usize {
        self.streams.values().map(StreamQueue::len).sum()
    }

    /// Number of packets buffered for a specific stream.
    #[must_use]
    pub fn stream_depth(&self, stream_index: usize) -> usize {
        self.streams.get(&stream_index).map_or(0, StreamQueue::len)
    }

    /// Returns `true` if all stream queues are empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.streams.values().all(StreamQueue::is_empty)
    }

    /// Find the PTS of the first keyframe at or after `target_pts` on the
    /// given stream, without consuming any packets.
    #[must_use]
    pub fn first_keyframe_pts(&self, stream_index: usize, target_pts: i64) -> Option<i64> {
        let q = self.streams.get(&stream_index)?;
        let pos = q.first_keyframe_pos_from(target_pts)?;
        q.queue.get(pos).map(|p| p.pts)
    }
}

impl Default for PacketBuffer {
    fn default() -> Self {
        Self::new(512)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pkt(stream: usize, pts: i64, keyframe: bool) -> BufferedPacket {
        BufferedPacket::new(stream, pts, vec![0u8; 8], keyframe)
    }

    #[test]
    fn test_buffered_packet_new() {
        let p = make_pkt(0, 1000, true);
        assert_eq!(p.stream_index, 0);
        assert_eq!(p.pts, 1000);
        assert!(p.is_keyframe);
        assert_eq!(p.data.len(), 8);
    }

    #[test]
    fn test_push_and_pop_stream() {
        let mut buf = PacketBuffer::new(16);
        buf.push(make_pkt(0, 100, true));
        buf.push(make_pkt(0, 200, false));
        let p = buf.pop_stream(0).expect("operation should succeed");
        assert_eq!(p.pts, 100);
        assert_eq!(buf.stream_depth(0), 1);
    }

    #[test]
    fn test_pop_stream_empty() {
        let mut buf = PacketBuffer::new(16);
        assert!(buf.pop_stream(0).is_none());
    }

    #[test]
    fn test_pop_lowest_pts() {
        let mut buf = PacketBuffer::new(16);
        buf.push(make_pkt(0, 300, false));
        buf.push(make_pkt(1, 100, true));
        buf.push(make_pkt(1, 200, false));

        let p = buf.pop_lowest_pts().expect("operation should succeed");
        assert_eq!(p.pts, 100);
        assert_eq!(p.stream_index, 1);
    }

    #[test]
    fn test_flush() {
        let mut buf = PacketBuffer::new(16);
        buf.push(make_pkt(0, 100, true));
        buf.push(make_pkt(1, 200, false));
        assert_eq!(buf.total_buffered(), 2);
        buf.flush();
        assert_eq!(buf.total_buffered(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_seek_to_pts() {
        let mut buf = PacketBuffer::new(16);
        buf.push(make_pkt(0, 50, true));
        buf.push(make_pkt(0, 150, false));
        buf.push(make_pkt(0, 250, false));
        buf.seek_to_pts(100);
        // Packet at 50 should be discarded
        let p = buf.pop_stream(0).expect("operation should succeed");
        assert_eq!(p.pts, 150);
    }

    #[test]
    fn test_eviction_on_overflow() {
        let mut buf = PacketBuffer::new(3);
        buf.push(make_pkt(0, 10, true));
        buf.push(make_pkt(0, 20, false));
        buf.push(make_pkt(0, 30, false));
        // This push should evict pts=10
        let evicted = buf.push(make_pkt(0, 40, false));
        assert!(evicted.is_some());
        assert_eq!(evicted.expect("operation should succeed").pts, 10);
        assert_eq!(buf.stream_depth(0), 3);
    }

    #[test]
    fn test_total_buffered() {
        let mut buf = PacketBuffer::new(16);
        buf.push(make_pkt(0, 100, false));
        buf.push(make_pkt(1, 200, false));
        buf.push(make_pkt(2, 300, false));
        assert_eq!(buf.total_buffered(), 3);
    }

    #[test]
    fn test_is_empty_initially() {
        let buf = PacketBuffer::new(16);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_first_keyframe_pts() {
        let mut buf = PacketBuffer::new(16);
        buf.push(make_pkt(0, 0, true));
        buf.push(make_pkt(0, 100, false));
        buf.push(make_pkt(0, 200, true));
        buf.push(make_pkt(0, 300, false));

        // Looking for keyframe >= 150: should find 200
        let kf = buf.first_keyframe_pts(0, 150);
        assert_eq!(kf, Some(200));
    }

    #[test]
    fn test_first_keyframe_pts_none() {
        let mut buf = PacketBuffer::new(16);
        buf.push(make_pkt(0, 0, false));
        buf.push(make_pkt(0, 100, false));
        let kf = buf.first_keyframe_pts(0, 0);
        assert!(kf.is_none()); // No keyframe in queue
    }

    #[test]
    fn test_default_buffer() {
        let buf: PacketBuffer = Default::default();
        assert!(buf.is_empty());
    }

    #[test]
    fn test_multi_stream_interleave() {
        let mut buf = PacketBuffer::new(64);
        // Video stream
        for i in [0i64, 40, 80, 120] {
            buf.push(make_pkt(0, i, i == 0));
        }
        // Audio stream
        for i in [0i64, 20, 40, 60, 80] {
            buf.push(make_pkt(1, i, true));
        }
        let mut last_pts = -1i64;
        while let Some(p) = buf.pop_lowest_pts() {
            assert!(p.pts >= last_pts, "out of order");
            last_pts = p.pts;
        }
    }
}
