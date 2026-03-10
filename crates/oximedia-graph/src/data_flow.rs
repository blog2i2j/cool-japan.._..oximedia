//! Graph data flow management: buffered port queues with backpressure.
//!
//! This module handles the movement of [`DataPacket`]s between graph nodes
//! through bounded [`PortBuffer`]s.  A [`DataFlowController`] coordinates
//! all buffers and enforces a configurable backpressure [`BackpressurePolicy`].

use std::collections::VecDeque;

// ── DataPacket ────────────────────────────────────────────────────────────────

/// A unit of data travelling between two ports in the processing graph.
#[derive(Debug, Clone)]
pub struct DataPacket {
    /// ID of the node that produced this packet.
    pub node_id: u64,
    /// Output port index on the producing node.
    pub port: u32,
    /// Presentation timestamp in milliseconds.
    pub timestamp_ms: u64,
    /// Raw payload bytes.
    pub data: Vec<u8>,
    /// If `true` this is the last packet in the stream.
    pub is_eos: bool,
}

impl DataPacket {
    /// Creates a normal (non-EOS) data packet.
    pub fn new(node_id: u64, port: u32, timestamp_ms: u64, data: Vec<u8>) -> Self {
        Self {
            node_id,
            port,
            timestamp_ms,
            data,
            is_eos: false,
        }
    }

    /// Creates an end-of-stream sentinel packet with an empty payload.
    pub fn eos(node_id: u64, port: u32, timestamp_ms: u64) -> Self {
        Self {
            node_id,
            port,
            timestamp_ms,
            data: Vec::new(),
            is_eos: true,
        }
    }

    /// Returns the size of the payload in bytes.
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if this is an end-of-stream packet.
    pub fn is_end_of_stream(&self) -> bool {
        self.is_eos
    }
}

// ── PortBuffer ────────────────────────────────────────────────────────────────

/// A bounded FIFO queue of [`DataPacket`]s associated with one (node, port) pair.
#[derive(Debug)]
pub struct PortBuffer {
    /// Node ID that owns this buffer.
    pub node_id: u64,
    /// Port index this buffer serves.
    pub port: u32,
    /// Queued packets, oldest at the front.
    pub buffer: VecDeque<DataPacket>,
    /// Maximum number of packets this buffer may hold.
    pub max_size: usize,
}

impl PortBuffer {
    /// Creates a new, empty port buffer with the given capacity.
    pub fn new(node_id: u64, port: u32, max_size: usize) -> Self {
        Self {
            node_id,
            port,
            buffer: VecDeque::new(),
            max_size,
        }
    }

    /// Enqueues `packet` if there is room.  Returns `true` on success.
    pub fn push(&mut self, packet: DataPacket) -> bool {
        if self.buffer.len() >= self.max_size {
            return false;
        }
        self.buffer.push_back(packet);
        true
    }

    /// Removes and returns the oldest packet, or `None` if empty.
    pub fn pop(&mut self) -> Option<DataPacket> {
        self.buffer.pop_front()
    }

    /// Returns `true` if the buffer has reached its maximum capacity.
    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.max_size
    }

    /// Returns the current number of packets in the buffer.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns `true` if the buffer contains no packets.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

// ── BackpressurePolicy ────────────────────────────────────────────────────────

/// Strategy to apply when a [`PortBuffer`] is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackpressurePolicy {
    /// Silently discard the incoming packet.
    Drop,
    /// Signal the producer to block (caller must retry).
    Block,
    /// Resize the buffer to accommodate the packet.
    Resize,
}

impl BackpressurePolicy {
    /// Returns `true` if the policy instructs the caller to drop the packet.
    pub fn should_drop(&self) -> bool {
        matches!(self, Self::Drop)
    }
}

// ── DataFlowController ────────────────────────────────────────────────────────

/// Coordinates all port buffers in the graph and enforces backpressure.
#[derive(Debug)]
pub struct DataFlowController {
    /// All port buffers managed by this controller.
    pub buffers: Vec<PortBuffer>,
    /// Backpressure strategy applied to full buffers.
    pub policy: BackpressurePolicy,
}

impl DataFlowController {
    /// Creates a new controller with no buffers and the given policy.
    pub fn new(policy: BackpressurePolicy) -> Self {
        Self {
            buffers: Vec::new(),
            policy,
        }
    }

    /// Adds a new buffer to this controller.
    pub fn add_buffer(&mut self, buffer: PortBuffer) {
        self.buffers.push(buffer);
    }

    /// Returns a mutable reference to the buffer for `(node_id, port)`.
    pub fn get_buffer(&mut self, node_id: u64, port: u32) -> Option<&mut PortBuffer> {
        self.buffers
            .iter_mut()
            .find(|b| b.node_id == node_id && b.port == port)
    }

    /// Returns the total number of packets currently held across all buffers.
    pub fn total_packets_in_flight(&self) -> usize {
        self.buffers.iter().map(|b| b.len()).sum()
    }

    /// Drains the buffer for `(node_id, port)`.  Returns `true` if the buffer
    /// existed (it may have already been empty).
    pub fn clear_buffer(&mut self, node_id: u64, port: u32) -> bool {
        if let Some(buf) = self.get_buffer(node_id, port) {
            buf.buffer.clear();
            true
        } else {
            false
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // ── DataPacket ────────────────────────────────────────────────────────────

    #[test]
    fn packet_size_bytes_returns_payload_length() {
        let p = DataPacket::new(1, 0, 1000, vec![0u8; 64]);
        assert_eq!(p.size_bytes(), 64);
    }

    #[test]
    fn packet_is_not_eos_by_default() {
        let p = DataPacket::new(1, 0, 0, vec![]);
        assert!(!p.is_end_of_stream());
    }

    #[test]
    fn eos_packet_is_end_of_stream() {
        let p = DataPacket::eos(2, 1, 5000);
        assert!(p.is_end_of_stream());
        assert_eq!(p.size_bytes(), 0);
    }

    #[test]
    fn packet_stores_timestamp() {
        let p = DataPacket::new(3, 0, 42_000, vec![1, 2, 3]);
        assert_eq!(p.timestamp_ms, 42_000);
    }

    // ── PortBuffer ────────────────────────────────────────────────────────────

    #[test]
    fn buffer_push_pop_roundtrip() {
        let mut buf = PortBuffer::new(1, 0, 4);
        let p = DataPacket::new(1, 0, 0, vec![9]);
        assert!(buf.push(p));
        let out = buf.pop().expect("pop should succeed");
        assert_eq!(out.data, vec![9]);
    }

    #[test]
    fn buffer_push_rejects_when_full() {
        let mut buf = PortBuffer::new(1, 0, 2);
        assert!(buf.push(DataPacket::new(1, 0, 0, vec![])));
        assert!(buf.push(DataPacket::new(1, 0, 1, vec![])));
        assert!(!buf.push(DataPacket::new(1, 0, 2, vec![]))); // full
    }

    #[test]
    fn buffer_is_full_true_at_capacity() {
        let mut buf = PortBuffer::new(1, 0, 1);
        buf.push(DataPacket::new(1, 0, 0, vec![]));
        assert!(buf.is_full());
    }

    #[test]
    fn buffer_len_reflects_queue_depth() {
        let mut buf = PortBuffer::new(2, 0, 10);
        for i in 0..5 {
            buf.push(DataPacket::new(2, 0, i, vec![]));
        }
        assert_eq!(buf.len(), 5);
    }

    #[test]
    fn buffer_pop_empty_returns_none() {
        let mut buf = PortBuffer::new(1, 0, 4);
        assert!(buf.pop().is_none());
    }

    // ── BackpressurePolicy ────────────────────────────────────────────────────

    #[test]
    fn drop_policy_should_drop_is_true() {
        assert!(BackpressurePolicy::Drop.should_drop());
    }

    #[test]
    fn block_policy_should_drop_is_false() {
        assert!(!BackpressurePolicy::Block.should_drop());
    }

    #[test]
    fn resize_policy_should_drop_is_false() {
        assert!(!BackpressurePolicy::Resize.should_drop());
    }

    // ── DataFlowController ────────────────────────────────────────────────────

    #[test]
    fn controller_total_packets_sums_all_buffers() {
        let mut ctrl = DataFlowController::new(BackpressurePolicy::Drop);
        let mut b1 = PortBuffer::new(1, 0, 8);
        let mut b2 = PortBuffer::new(2, 0, 8);
        b1.push(DataPacket::new(1, 0, 0, vec![]));
        b1.push(DataPacket::new(1, 0, 1, vec![]));
        b2.push(DataPacket::new(2, 0, 0, vec![]));
        ctrl.add_buffer(b1);
        ctrl.add_buffer(b2);
        assert_eq!(ctrl.total_packets_in_flight(), 3);
    }

    #[test]
    fn controller_get_buffer_returns_correct_buffer() {
        let mut ctrl = DataFlowController::new(BackpressurePolicy::Block);
        ctrl.add_buffer(PortBuffer::new(5, 2, 4));
        assert!(ctrl.get_buffer(5, 2).is_some());
        assert!(ctrl.get_buffer(5, 3).is_none());
    }

    #[test]
    fn controller_clear_buffer_empties_queue() {
        let mut ctrl = DataFlowController::new(BackpressurePolicy::Drop);
        let mut buf = PortBuffer::new(1, 0, 8);
        buf.push(DataPacket::new(1, 0, 0, vec![1, 2, 3]));
        ctrl.add_buffer(buf);
        assert!(ctrl.clear_buffer(1, 0));
        assert_eq!(ctrl.total_packets_in_flight(), 0);
    }

    #[test]
    fn controller_clear_buffer_false_for_missing() {
        let mut ctrl = DataFlowController::new(BackpressurePolicy::Drop);
        assert!(!ctrl.clear_buffer(99, 0));
    }
}
