//! NDI transport layer abstraction.
//!
//! Provides a unified transport API over different underlying carriers
//! (TCP reliable, UDP multicast, RUDP low-latency).  The transport manages
//! framing, congestion signalling, and basic packet sequencing.

#![allow(dead_code)]

use std::collections::VecDeque;
use std::fmt;
use std::time::{Duration, Instant};

/// Supported transport modes for NDI streaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransportMode {
    /// Reliable ordered TCP stream.
    TcpUnicast,
    /// Low-latency unreliable UDP unicast.
    UdpUnicast,
    /// UDP multicast for one-to-many delivery.
    UdpMulticast,
    /// Reliable UDP with congestion control (RUDP-like).
    ReliableUdp,
}

impl fmt::Display for TransportMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TcpUnicast => write!(f, "TCP/Unicast"),
            Self::UdpUnicast => write!(f, "UDP/Unicast"),
            Self::UdpMulticast => write!(f, "UDP/Multicast"),
            Self::ReliableUdp => write!(f, "RUDP"),
        }
    }
}

/// Priority level for transport packets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PacketPriority {
    /// Best-effort (metadata, tally)
    Low,
    /// Normal (audio)
    Normal,
    /// Elevated (video I-frames)
    High,
    /// Critical (control messages)
    Critical,
}

/// A sequenced packet ready for transport.
#[derive(Debug, Clone)]
pub struct TransportPacket {
    /// Monotonically increasing sequence number.
    pub sequence: u64,
    /// Priority level.
    pub priority: PacketPriority,
    /// Payload bytes.
    pub payload: Vec<u8>,
    /// Timestamp when the packet was created.
    pub created_at: Instant,
}

impl TransportPacket {
    /// Create a new transport packet.
    #[must_use]
    pub fn new(sequence: u64, priority: PacketPriority, payload: Vec<u8>) -> Self {
        Self {
            sequence,
            priority,
            payload,
            created_at: Instant::now(),
        }
    }

    /// Payload size in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.payload.len()
    }

    /// Age of this packet since creation.
    #[must_use]
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

/// Congestion signal reported by the transport layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CongestionSignal {
    /// No congestion detected.
    Clear,
    /// Mild congestion — consider reducing bitrate.
    Warning,
    /// Severe congestion — drop non-essential frames.
    Critical,
}

/// Configuration for the transport layer.
#[derive(Debug, Clone)]
pub struct TransportConfig {
    /// Transport mode to use.
    pub mode: TransportMode,
    /// Maximum transmission unit in bytes.
    pub mtu: usize,
    /// Send buffer capacity (number of packets).
    pub send_buffer_size: usize,
    /// Receive buffer capacity (number of packets).
    pub recv_buffer_size: usize,
    /// Timeout for considering a connection stale.
    pub stale_timeout: Duration,
    /// Whether to enable congestion detection.
    pub congestion_detection: bool,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            mode: TransportMode::TcpUnicast,
            mtu: 1400,
            send_buffer_size: 256,
            recv_buffer_size: 256,
            stale_timeout: Duration::from_secs(5),
            congestion_detection: true,
        }
    }
}

/// Running statistics for a transport session.
#[derive(Debug, Clone, Default)]
pub struct TransportStats {
    /// Total packets sent.
    pub packets_sent: u64,
    /// Total packets received.
    pub packets_received: u64,
    /// Total bytes sent.
    pub bytes_sent: u64,
    /// Total bytes received.
    pub bytes_received: u64,
    /// Packets dropped due to buffer overflow.
    pub packets_dropped: u64,
    /// Number of congestion events.
    pub congestion_events: u64,
}

impl TransportStats {
    /// Packet loss ratio (dropped / sent).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn loss_ratio(&self) -> f64 {
        if self.packets_sent == 0 {
            return 0.0;
        }
        self.packets_dropped as f64 / self.packets_sent as f64
    }

    /// Average packet size on the send side.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_send_packet_size(&self) -> f64 {
        if self.packets_sent == 0 {
            return 0.0;
        }
        self.bytes_sent as f64 / self.packets_sent as f64
    }

    /// Average packet size on the receive side.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_recv_packet_size(&self) -> f64 {
        if self.packets_received == 0 {
            return 0.0;
        }
        self.bytes_received as f64 / self.packets_received as f64
    }
}

/// A software-level transport session that manages packet sequencing,
/// buffering, and congestion estimation.
#[derive(Debug)]
pub struct TransportSession {
    /// Session configuration.
    config: TransportConfig,
    /// Outbound packet queue.
    send_queue: VecDeque<TransportPacket>,
    /// Inbound reassembly queue keyed by sequence number.
    recv_queue: VecDeque<TransportPacket>,
    /// Next outgoing sequence number.
    next_seq: u64,
    /// Running statistics.
    stats: TransportStats,
    /// Current congestion signal.
    congestion: CongestionSignal,
}

impl TransportSession {
    /// Create a new transport session with the given configuration.
    #[must_use]
    pub fn new(config: TransportConfig) -> Self {
        Self {
            config,
            send_queue: VecDeque::new(),
            recv_queue: VecDeque::new(),
            next_seq: 0,
            stats: TransportStats::default(),
            congestion: CongestionSignal::Clear,
        }
    }

    /// Transport mode of this session.
    #[must_use]
    pub fn mode(&self) -> TransportMode {
        self.config.mode
    }

    /// Enqueue a payload for sending with the given priority.
    ///
    /// Returns the assigned sequence number, or `None` if the buffer is full
    /// (packet dropped).
    pub fn enqueue_send(&mut self, payload: Vec<u8>, priority: PacketPriority) -> Option<u64> {
        if self.send_queue.len() >= self.config.send_buffer_size {
            self.stats.packets_dropped += 1;
            self.check_congestion();
            return None;
        }
        let seq = self.next_seq;
        self.next_seq += 1;
        let pkt = TransportPacket::new(seq, priority, payload);
        self.stats.bytes_sent += pkt.size() as u64;
        self.stats.packets_sent += 1;
        self.send_queue.push_back(pkt);
        Some(seq)
    }

    /// Dequeue the next outbound packet, if any.
    pub fn dequeue_send(&mut self) -> Option<TransportPacket> {
        self.send_queue.pop_front()
    }

    /// Feed a received packet into the session.
    ///
    /// Returns `false` if the receive buffer is full (packet dropped).
    pub fn feed_received(&mut self, pkt: TransportPacket) -> bool {
        if self.recv_queue.len() >= self.config.recv_buffer_size {
            self.stats.packets_dropped += 1;
            return false;
        }
        self.stats.bytes_received += pkt.size() as u64;
        self.stats.packets_received += 1;
        self.recv_queue.push_back(pkt);
        true
    }

    /// Dequeue the next received packet.
    pub fn dequeue_received(&mut self) -> Option<TransportPacket> {
        self.recv_queue.pop_front()
    }

    /// Number of pending outbound packets.
    #[must_use]
    pub fn send_queue_len(&self) -> usize {
        self.send_queue.len()
    }

    /// Number of pending inbound packets.
    #[must_use]
    pub fn recv_queue_len(&self) -> usize {
        self.recv_queue.len()
    }

    /// Current congestion signal.
    #[must_use]
    pub fn congestion(&self) -> CongestionSignal {
        self.congestion
    }

    /// Read-only view of the running statistics.
    #[must_use]
    pub fn stats(&self) -> &TransportStats {
        &self.stats
    }

    /// Update congestion estimate based on queue fill levels.
    fn check_congestion(&mut self) {
        if !self.config.congestion_detection {
            self.congestion = CongestionSignal::Clear;
            return;
        }
        let fill = self.send_queue.len() as f64 / self.config.send_buffer_size.max(1) as f64;
        self.congestion = if fill > 0.9 {
            self.stats.congestion_events += 1;
            CongestionSignal::Critical
        } else if fill > 0.6 {
            CongestionSignal::Warning
        } else {
            CongestionSignal::Clear
        };
    }

    /// Flush all pending send packets (e.g. on disconnect).
    pub fn flush_send(&mut self) {
        self.send_queue.clear();
    }

    /// Flush all pending receive packets.
    pub fn flush_recv(&mut self) {
        self.recv_queue.clear();
    }

    /// Reset the session to initial state.
    pub fn reset(&mut self) {
        self.send_queue.clear();
        self.recv_queue.clear();
        self.next_seq = 0;
        self.stats = TransportStats::default();
        self.congestion = CongestionSignal::Clear;
    }
}

/// Fragment a large payload into MTU-sized chunks.
///
/// Each chunk is tagged with the same `base_sequence` and a fragment
/// index so the receiver can reassemble.
#[derive(Debug, Clone)]
pub struct Fragment {
    /// Base sequence number of the original packet.
    pub base_sequence: u64,
    /// Zero-based fragment index.
    pub fragment_index: u16,
    /// Total fragment count.
    pub total_fragments: u16,
    /// Fragment payload.
    pub data: Vec<u8>,
}

/// Split `payload` into fragments of at most `mtu` bytes each.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn fragment_payload(payload: &[u8], mtu: usize, base_sequence: u64) -> Vec<Fragment> {
    if mtu == 0 {
        return Vec::new();
    }
    let chunks: Vec<&[u8]> = payload.chunks(mtu).collect();
    let total = chunks.len().min(u16::MAX as usize) as u16;
    chunks
        .into_iter()
        .enumerate()
        .map(|(i, chunk)| Fragment {
            base_sequence,
            fragment_index: i as u16,
            total_fragments: total,
            data: chunk.to_vec(),
        })
        .collect()
}

/// Reassemble fragments into the original payload.
///
/// Returns `None` if fragments are missing or inconsistent.
#[must_use]
pub fn reassemble_fragments(fragments: &mut [Fragment]) -> Option<Vec<u8>> {
    if fragments.is_empty() {
        return None;
    }
    let total = fragments[0].total_fragments;
    if fragments.len() != total as usize {
        return None;
    }
    fragments.sort_by_key(|f| f.fragment_index);
    for (i, f) in fragments.iter().enumerate() {
        if f.fragment_index != i as u16 {
            return None;
        }
    }
    let mut result = Vec::new();
    for f in fragments.iter() {
        result.extend_from_slice(&f.data);
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transport_mode_display() {
        assert_eq!(TransportMode::TcpUnicast.to_string(), "TCP/Unicast");
        assert_eq!(TransportMode::ReliableUdp.to_string(), "RUDP");
    }

    #[test]
    fn test_packet_creation() {
        let pkt = TransportPacket::new(42, PacketPriority::High, vec![1, 2, 3]);
        assert_eq!(pkt.sequence, 42);
        assert_eq!(pkt.size(), 3);
        assert_eq!(pkt.priority, PacketPriority::High);
    }

    #[test]
    fn test_default_config() {
        let cfg = TransportConfig::default();
        assert_eq!(cfg.mode, TransportMode::TcpUnicast);
        assert_eq!(cfg.mtu, 1400);
        assert!(cfg.congestion_detection);
    }

    #[test]
    fn test_session_enqueue_dequeue_send() {
        let mut session = TransportSession::new(TransportConfig::default());
        let seq = session.enqueue_send(vec![10, 20], PacketPriority::Normal);
        assert_eq!(seq, Some(0));
        assert_eq!(session.send_queue_len(), 1);
        let pkt = session
            .dequeue_send()
            .expect("expected packet in send queue");
        assert_eq!(pkt.sequence, 0);
        assert_eq!(pkt.payload, vec![10, 20]);
    }

    #[test]
    fn test_session_send_buffer_overflow() {
        let cfg = TransportConfig {
            send_buffer_size: 2,
            ..TransportConfig::default()
        };
        let mut session = TransportSession::new(cfg);
        assert!(session.enqueue_send(vec![1], PacketPriority::Low).is_some());
        assert!(session.enqueue_send(vec![2], PacketPriority::Low).is_some());
        // Buffer full — should drop
        assert!(session.enqueue_send(vec![3], PacketPriority::Low).is_none());
        assert_eq!(session.stats().packets_dropped, 1);
    }

    #[test]
    fn test_session_receive() {
        let mut session = TransportSession::new(TransportConfig::default());
        let pkt = TransportPacket::new(0, PacketPriority::Normal, vec![99]);
        assert!(session.feed_received(pkt));
        assert_eq!(session.recv_queue_len(), 1);
        let out = session
            .dequeue_received()
            .expect("expected packet in recv queue");
        assert_eq!(out.payload, vec![99]);
    }

    #[test]
    fn test_session_recv_overflow() {
        let cfg = TransportConfig {
            recv_buffer_size: 1,
            ..TransportConfig::default()
        };
        let mut session = TransportSession::new(cfg);
        let p1 = TransportPacket::new(0, PacketPriority::Normal, vec![1]);
        let p2 = TransportPacket::new(1, PacketPriority::Normal, vec![2]);
        assert!(session.feed_received(p1));
        assert!(!session.feed_received(p2));
    }

    #[test]
    fn test_session_reset() {
        let mut session = TransportSession::new(TransportConfig::default());
        session.enqueue_send(vec![1], PacketPriority::Normal);
        session.reset();
        assert_eq!(session.send_queue_len(), 0);
        assert_eq!(session.stats().packets_sent, 0);
    }

    #[test]
    fn test_stats_loss_ratio() {
        let stats = TransportStats {
            packets_sent: 100,
            packets_dropped: 10,
            ..TransportStats::default()
        };
        assert!((stats.loss_ratio() - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_stats_avg_packet_size() {
        let stats = TransportStats {
            packets_sent: 4,
            bytes_sent: 400,
            ..TransportStats::default()
        };
        assert!((stats.avg_send_packet_size() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_fragment_payload_basic() {
        let payload = vec![0u8; 10];
        let frags = fragment_payload(&payload, 4, 100);
        assert_eq!(frags.len(), 3);
        assert_eq!(frags[0].total_fragments, 3);
        assert_eq!(frags[0].data.len(), 4);
        assert_eq!(frags[2].data.len(), 2);
    }

    #[test]
    fn test_fragment_zero_mtu() {
        let frags = fragment_payload(&[1, 2, 3], 0, 0);
        assert!(frags.is_empty());
    }

    #[test]
    fn test_reassemble_roundtrip() {
        let original = vec![1u8, 2, 3, 4, 5, 6, 7];
        let mut frags = fragment_payload(&original, 3, 0);
        let reassembled = reassemble_fragments(&mut frags).expect("reassembly should succeed");
        assert_eq!(reassembled, original);
    }

    #[test]
    fn test_reassemble_missing_fragment() {
        let original = vec![1u8, 2, 3, 4, 5];
        let mut frags = fragment_payload(&original, 2, 0);
        frags.remove(1); // remove middle fragment
        assert!(reassemble_fragments(&mut frags).is_none());
    }

    #[test]
    fn test_priority_ordering() {
        assert!(PacketPriority::Low < PacketPriority::Normal);
        assert!(PacketPriority::Normal < PacketPriority::High);
        assert!(PacketPriority::High < PacketPriority::Critical);
    }
}
