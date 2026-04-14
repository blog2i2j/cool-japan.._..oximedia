//! SRT (Secure Reliable Transport) protocol ingest for the OxiMedia streaming pipeline.
//!
//! Provides a pure-Rust simulation of SRT protocol ingest as input to the streaming
//! pipeline.  SRT is a UDP-based transport protocol that adds reliability, encryption,
//! and flow control on top of UDP for contribution/ingest workflows.
//!
//! # Key types
//!
//! | Type | Purpose |
//! |---|---|
//! | [`SrtConfig`] | Connection configuration (mode, encryption, latency) |
//! | [`SrtPacket`] | Low-level SRT packet representation |
//! | [`SrtStream`] | Active ingest stream with statistics |
//! | [`SrtIngest`] | Multi-stream ingest manager |

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

// ─── SrtMode ─────────────────────────────────────────────────────────────────

/// SRT connection mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SrtMode {
    /// Caller (client) mode — initiates the connection.
    Caller,
    /// Listener (server) mode — accepts incoming connections.
    Listener,
    /// Rendezvous mode — both sides connect simultaneously.
    Rendezvous,
}

impl Default for SrtMode {
    fn default() -> Self {
        Self::Listener
    }
}

// ─── SrtEncryption ───────────────────────────────────────────────────────────

/// SRT payload encryption configuration.
#[derive(Debug, Clone)]
pub enum SrtEncryption {
    /// No encryption.
    None,
    /// AES-128 encryption with the given passphrase.
    Aes128 {
        /// The shared encryption passphrase.
        passphrase: String,
    },
    /// AES-256 encryption with the given passphrase.
    Aes256 {
        /// The shared encryption passphrase.
        passphrase: String,
    },
}

impl Default for SrtEncryption {
    fn default() -> Self {
        Self::None
    }
}

// ─── SrtConfig ───────────────────────────────────────────────────────────────

/// Configuration for an SRT ingest connection.
#[derive(Debug, Clone)]
pub struct SrtConfig {
    /// Bind address (e.g. `"0.0.0.0:9000"`).
    pub bind_addr: String,
    /// Connection mode.
    pub mode: SrtMode,
    /// SRT latency — the size of the receiver buffer in milliseconds.
    /// Higher values tolerate more network jitter.  Typical range: 120–800 ms.
    pub latency_ms: u32,
    /// Maximum bandwidth in bytes per second (0 = unlimited).
    pub max_bandwidth_bps: u64,
    /// Encryption configuration.
    pub encryption: SrtEncryption,
    /// Input bandwidth in bytes per second (used for rate estimation).
    pub input_bandwidth_bps: u64,
    /// Overhead bandwidth as a percentage of `input_bandwidth_bps`.
    pub overhead_bandwidth_pct: u8,
    /// Maximum SRT payload size in bytes (must be ≤ 1456 for Ethernet MTU).
    pub payload_size: usize,
    /// Stream ID (optional metadata string sent during connection handshake).
    pub stream_id: Option<String>,
    /// Connection timeout in milliseconds.
    pub connect_timeout_ms: u64,
    /// Peer idle timeout in milliseconds (connection dropped when idle).
    pub peer_idle_timeout_ms: u64,
}

impl Default for SrtConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0:9000".into(),
            mode: SrtMode::default(),
            latency_ms: 200,
            max_bandwidth_bps: 0,
            encryption: SrtEncryption::default(),
            input_bandwidth_bps: 0,
            overhead_bandwidth_pct: 25,
            payload_size: 1316,
            stream_id: None,
            connect_timeout_ms: 3_000,
            peer_idle_timeout_ms: 5_000,
        }
    }
}

// ─── SrtPacketType ───────────────────────────────────────────────────────────

/// SRT packet type (high-level classification).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SrtPacketType {
    /// Data packet carrying media payload.
    Data,
    /// Control packet (ACK, NAK, SHUTDOWN, etc.).
    Control,
}

// ─── SrtPacket ───────────────────────────────────────────────────────────────

/// A single SRT packet.
#[derive(Debug, Clone)]
pub struct SrtPacket {
    /// Packet type.
    pub packet_type: SrtPacketType,
    /// Packet sequence number (32-bit, wraps at 0x7FFF_FFFF per SRT spec).
    pub sequence_number: u32,
    /// Timestamp (microseconds from session start).
    pub timestamp_us: u32,
    /// Destination socket ID.
    pub socket_id: u32,
    /// Raw payload bytes.
    pub payload: Vec<u8>,
    /// Whether this packet was re-transmitted (ARQ).
    pub is_retransmit: bool,
    /// Whether this is the first fragment of an encapsulated TS packet boundary.
    pub is_boundary: bool,
}

impl SrtPacket {
    /// Create a data packet.
    pub fn data(sequence_number: u32, timestamp_us: u32, socket_id: u32, payload: Vec<u8>) -> Self {
        Self {
            packet_type: SrtPacketType::Data,
            sequence_number,
            timestamp_us,
            socket_id,
            payload,
            is_retransmit: false,
            is_boundary: false,
        }
    }

    /// Serialise to a flat byte buffer (simplified SRT wire format for testing).
    ///
    /// Layout (28 bytes header + payload):
    /// ```text
    /// [0]    = packet_type (0 = data, 1 = control)
    /// [1-3]  = flags (is_retransmit[0], is_boundary[1])
    /// [4-7]  = sequence_number (big-endian)
    /// [8-11] = timestamp_us   (big-endian)
    /// [12-15]= socket_id      (big-endian)
    /// [16-27]= reserved
    /// [28..] = payload
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(28 + self.payload.len());
        buf.push(if self.packet_type == SrtPacketType::Data {
            0u8
        } else {
            1u8
        });
        let mut flags = 0u8;
        if self.is_retransmit {
            flags |= 0x01;
        }
        if self.is_boundary {
            flags |= 0x02;
        }
        buf.push(flags);
        buf.push(0); // reserved
        buf.push(0);
        buf.extend_from_slice(&self.sequence_number.to_be_bytes());
        buf.extend_from_slice(&self.timestamp_us.to_be_bytes());
        buf.extend_from_slice(&self.socket_id.to_be_bytes());
        buf.extend_from_slice(&[0u8; 12]); // reserved
        buf.extend_from_slice(&self.payload);
        buf
    }

    /// Parse a flat byte buffer back into an [`SrtPacket`].
    ///
    /// Returns `None` if `bytes` is shorter than the 28-byte header.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 28 {
            return None;
        }
        let packet_type = if bytes[0] == 0 {
            SrtPacketType::Data
        } else {
            SrtPacketType::Control
        };
        let flags = bytes[1];
        let is_retransmit = (flags & 0x01) != 0;
        let is_boundary = (flags & 0x02) != 0;
        let seq = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let ts = u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let sid = u32::from_be_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        let payload = bytes[28..].to_vec();
        Some(Self {
            packet_type,
            sequence_number: seq,
            timestamp_us: ts,
            socket_id: sid,
            payload,
            is_retransmit,
            is_boundary,
        })
    }
}

// ─── SrtStreamStats ──────────────────────────────────────────────────────────

/// Real-time statistics for an SRT ingest stream.
#[derive(Debug, Clone, Default)]
pub struct SrtStreamStats {
    /// Total packets received (data + retransmitted).
    pub packets_received: u64,
    /// Total bytes received (payload only).
    pub bytes_received: u64,
    /// Packets lost (gap in sequence numbers, not recovered by ARQ).
    pub packets_lost: u64,
    /// Packets retransmitted (re-delivered by ARQ).
    pub packets_retransmitted: u64,
    /// Current estimated round-trip time in microseconds.
    pub rtt_us: u32,
    /// Receive buffer fill level in bytes.
    pub buffer_fill_bytes: usize,
    /// Estimated bitrate in kbps (EWMA over last ~1 s).
    pub bitrate_kbps: f64,
}

// ─── SrtStream ───────────────────────────────────────────────────────────────

/// An active SRT ingest stream.
///
/// In a production implementation this would wrap an OS socket and a running
/// async task.  Here it provides a simulated receive queue suitable for testing
/// and pipeline integration.
pub struct SrtStream {
    /// Unique socket ID assigned to this stream.
    pub socket_id: u32,
    /// Optional stream ID string (from handshake).
    pub stream_id: Option<String>,
    /// Configuration used to open this stream.
    pub config: SrtConfig,
    /// Monotonic time when this stream was opened.
    pub opened_at: Instant,
    /// Receive queue — packets ordered by arrival (not sequence number).
    receive_queue: VecDeque<SrtPacket>,
    /// Next expected sequence number (for loss detection).
    next_expected_seq: u32,
    /// Accumulated statistics.
    pub stats: SrtStreamStats,
    /// Bitrate EWMA state: (last_update, bytes_in_window).
    bitrate_state: (Instant, u64),
}

impl SrtStream {
    /// Create a new stream (opened / accepted).
    pub fn new(socket_id: u32, config: SrtConfig, stream_id: Option<String>) -> Self {
        Self {
            socket_id,
            stream_id,
            config,
            opened_at: Instant::now(),
            receive_queue: VecDeque::new(),
            next_expected_seq: 0,
            stats: SrtStreamStats::default(),
            bitrate_state: (Instant::now(), 0),
        }
    }

    /// Feed a received [`SrtPacket`] into the stream.
    ///
    /// Updates statistics and enqueues payload bytes.
    pub fn receive_packet(&mut self, packet: SrtPacket) {
        if packet.packet_type != SrtPacketType::Data {
            return; // Ignore control packets for now.
        }

        // Sequence number loss detection (simplified: wrapping 31-bit space).
        let seq = packet.sequence_number & 0x7FFF_FFFF;
        let expected = self.next_expected_seq & 0x7FFF_FFFF;
        if seq != expected && !packet.is_retransmit {
            // Estimate lost packets in the gap.
            let lost = seq.wrapping_sub(expected) & 0x7FFF_FFFF;
            if lost < 1000 {
                // Sanity cap to avoid false positives on wrap.
                self.stats.packets_lost += u64::from(lost);
            }
        }

        self.next_expected_seq = (seq + 1) & 0x7FFF_FFFF;

        if packet.is_retransmit {
            self.stats.packets_retransmitted += 1;
        }

        self.stats.packets_received += 1;
        self.stats.bytes_received += packet.payload.len() as u64;
        self.stats.buffer_fill_bytes += packet.payload.len();

        // Update bitrate EWMA every second.
        self.bitrate_state.1 += packet.payload.len() as u64;
        let elapsed = self.bitrate_state.0.elapsed();
        if elapsed >= Duration::from_secs(1) {
            let kbps = (self.bitrate_state.1 as f64 * 8.0) / (elapsed.as_secs_f64() * 1000.0);
            // EWMA α = 0.3
            self.stats.bitrate_kbps = self.stats.bitrate_kbps * 0.7 + kbps * 0.3;
            self.bitrate_state = (Instant::now(), 0);
        }

        self.receive_queue.push_back(packet);
    }

    /// Drain all available payload bytes from the receive queue.
    ///
    /// Returns concatenated payload from all queued packets.
    pub fn drain_payload(&mut self) -> Vec<u8> {
        let total: usize = self.receive_queue.iter().map(|p| p.payload.len()).sum();
        let mut buf = Vec::with_capacity(total);
        while let Some(pkt) = self.receive_queue.pop_front() {
            self.stats.buffer_fill_bytes = self
                .stats
                .buffer_fill_bytes
                .saturating_sub(pkt.payload.len());
            buf.extend_from_slice(&pkt.payload);
        }
        buf
    }

    /// Number of packets currently in the receive queue.
    pub fn queue_depth(&self) -> usize {
        self.receive_queue.len()
    }

    /// Wall-clock age of this stream.
    pub fn age(&self) -> Duration {
        self.opened_at.elapsed()
    }

    /// Update the RTT estimate (from ACK/ACKACK exchange, microseconds).
    pub fn update_rtt(&mut self, rtt_us: u32) {
        // EWMA α = 0.125 (classic TCP smoothing)
        let current = self.stats.rtt_us as f64;
        let new_sample = rtt_us as f64;
        self.stats.rtt_us = (current * 0.875 + new_sample * 0.125) as u32;
    }
}

// ─── SrtIngest ───────────────────────────────────────────────────────────────

/// Multi-stream SRT ingest manager.
///
/// Manages a collection of active [`SrtStream`]s and routes incoming packets
/// to the correct stream by socket ID.  In production this would be backed by
/// actual UDP sockets; in this implementation it is driven by calls to
/// [`SrtIngest::inject_packet`] for testing.
pub struct SrtIngest {
    /// Active streams, keyed by socket ID.
    streams: HashMap<u32, SrtStream>,
    /// Shared configuration template for new streams.
    config: SrtConfig,
    /// Next socket ID to assign.
    next_socket_id: u32,
    /// Maximum concurrent streams.
    pub max_streams: usize,
}

impl SrtIngest {
    /// Create a new ingest manager.
    pub fn new(config: SrtConfig) -> Self {
        Self {
            streams: HashMap::new(),
            config,
            next_socket_id: 1,
            max_streams: 100,
        }
    }

    /// Accept a new ingest stream (simulates a successful SRT handshake).
    ///
    /// Returns the assigned socket ID, or `None` if the maximum stream count
    /// has been reached.
    pub fn accept_stream(&mut self, stream_id: Option<String>) -> Option<u32> {
        if self.streams.len() >= self.max_streams {
            return None;
        }
        let sid = self.next_socket_id;
        self.next_socket_id = self.next_socket_id.wrapping_add(1).max(1);
        let stream = SrtStream::new(sid, self.config.clone(), stream_id);
        self.streams.insert(sid, stream);
        Some(sid)
    }

    /// Inject a received packet for a specific socket ID.
    ///
    /// Returns `true` if the packet was delivered, `false` if the socket ID
    /// is unknown (e.g., the stream was closed).
    pub fn inject_packet(&mut self, socket_id: u32, packet: SrtPacket) -> bool {
        if let Some(stream) = self.streams.get_mut(&socket_id) {
            stream.receive_packet(packet);
            true
        } else {
            false
        }
    }

    /// Drain all payload from a stream.
    pub fn drain_stream(&mut self, socket_id: u32) -> Option<Vec<u8>> {
        self.streams.get_mut(&socket_id).map(|s| s.drain_payload())
    }

    /// Close and remove a stream.
    pub fn close_stream(&mut self, socket_id: u32) -> bool {
        self.streams.remove(&socket_id).is_some()
    }

    /// Iterate over active streams and their stats.
    pub fn stream_stats(&self) -> impl Iterator<Item = (u32, &SrtStreamStats)> {
        self.streams.iter().map(|(id, s)| (*id, &s.stats))
    }

    /// Number of active streams.
    pub fn active_count(&self) -> usize {
        self.streams.len()
    }

    /// Evict streams that have been idle for longer than `peer_idle_timeout_ms`
    /// (from the configuration template).
    ///
    /// Returns the socket IDs of the evicted streams.
    pub fn evict_idle_streams(&mut self) -> Vec<u32> {
        let timeout = Duration::from_millis(self.config.peer_idle_timeout_ms);
        let idle: Vec<u32> = self
            .streams
            .iter()
            .filter(|(_, s)| s.opened_at.elapsed() > timeout && s.queue_depth() == 0)
            .map(|(id, _)| *id)
            .collect();
        for id in &idle {
            self.streams.remove(id);
        }
        idle
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data_packet(seq: u32, payload: &[u8]) -> SrtPacket {
        SrtPacket::data(seq, seq * 10, 1, payload.to_vec())
    }

    #[test]
    fn test_srt_config_defaults() {
        let cfg = SrtConfig::default();
        assert_eq!(cfg.latency_ms, 200);
        assert_eq!(cfg.payload_size, 1316);
        assert!(matches!(cfg.mode, SrtMode::Listener));
        assert!(matches!(cfg.encryption, SrtEncryption::None));
    }

    #[test]
    fn test_srt_packet_roundtrip() {
        let pkt = SrtPacket {
            packet_type: SrtPacketType::Data,
            sequence_number: 42,
            timestamp_us: 12345,
            socket_id: 7,
            payload: b"hello".to_vec(),
            is_retransmit: false,
            is_boundary: true,
        };
        let bytes = pkt.to_bytes();
        let parsed = SrtPacket::from_bytes(&bytes).expect("parse should succeed");
        assert_eq!(parsed.sequence_number, 42);
        assert_eq!(parsed.timestamp_us, 12345);
        assert_eq!(parsed.socket_id, 7);
        assert_eq!(parsed.payload, b"hello");
        assert!(parsed.is_boundary);
        assert!(!parsed.is_retransmit);
    }

    #[test]
    fn test_srt_packet_from_bytes_too_short() {
        assert!(SrtPacket::from_bytes(&[0u8; 10]).is_none());
        assert!(SrtPacket::from_bytes(&[]).is_none());
    }

    #[test]
    fn test_stream_receive_and_drain() {
        let mut stream = SrtStream::new(1, SrtConfig::default(), None);
        stream.receive_packet(make_data_packet(0, b"segment_a"));
        stream.receive_packet(make_data_packet(1, b"segment_b"));
        assert_eq!(stream.queue_depth(), 2);
        let payload = stream.drain_payload();
        assert_eq!(payload, b"segment_asegment_b");
        assert_eq!(stream.queue_depth(), 0);
        assert_eq!(stream.stats.packets_received, 2);
        assert_eq!(stream.stats.bytes_received, 18);
    }

    #[test]
    fn test_stream_loss_detection() {
        let mut stream = SrtStream::new(1, SrtConfig::default(), None);
        // Packets 0 and 2 (gap = 1 lost packet between).
        stream.receive_packet(make_data_packet(0, b"a"));
        stream.receive_packet(make_data_packet(2, b"c")); // seq 1 is missing
        assert_eq!(stream.stats.packets_lost, 1);
    }

    #[test]
    fn test_stream_retransmit_flag() {
        let mut stream = SrtStream::new(1, SrtConfig::default(), None);
        let mut pkt = make_data_packet(5, b"retx");
        pkt.is_retransmit = true;
        stream.receive_packet(pkt);
        assert_eq!(stream.stats.packets_retransmitted, 1);
        assert_eq!(stream.stats.packets_lost, 0); // No loss because retransmit flag.
    }

    #[test]
    fn test_stream_rtt_ewma() {
        let mut stream = SrtStream::new(1, SrtConfig::default(), None);
        stream.update_rtt(1000);
        assert_eq!(stream.stats.rtt_us, 125); // 0 * 0.875 + 1000 * 0.125
        stream.update_rtt(1000);
        // 125 * 0.875 + 1000 * 0.125 = 109.375 + 125 = 234.375
        assert!(stream.stats.rtt_us >= 230 && stream.stats.rtt_us <= 240);
    }

    #[test]
    fn test_ingest_accept_and_inject() {
        let mut ingest = SrtIngest::new(SrtConfig::default());
        let sid = ingest
            .accept_stream(Some("live/stream1".into()))
            .expect("should accept");
        let ok = ingest.inject_packet(sid, make_data_packet(0, b"ts_data"));
        assert!(ok);
        let payload = ingest.drain_stream(sid).expect("drain");
        assert_eq!(payload, b"ts_data");
    }

    #[test]
    fn test_ingest_unknown_socket() {
        let mut ingest = SrtIngest::new(SrtConfig::default());
        assert!(!ingest.inject_packet(999, make_data_packet(0, b"x")));
    }

    #[test]
    fn test_ingest_close_stream() {
        let mut ingest = SrtIngest::new(SrtConfig::default());
        let sid = ingest.accept_stream(None).expect("accept");
        assert_eq!(ingest.active_count(), 1);
        assert!(ingest.close_stream(sid));
        assert_eq!(ingest.active_count(), 0);
        assert!(!ingest.close_stream(sid)); // Already closed.
    }

    #[test]
    fn test_ingest_max_streams() {
        let cfg = SrtConfig::default();
        let mut ingest = SrtIngest::new(cfg);
        ingest.max_streams = 2;
        assert!(ingest.accept_stream(None).is_some());
        assert!(ingest.accept_stream(None).is_some());
        assert!(ingest.accept_stream(None).is_none()); // Max reached.
    }

    #[test]
    fn test_ingest_stream_stats_iteration() {
        let mut ingest = SrtIngest::new(SrtConfig::default());
        let sid = ingest.accept_stream(None).expect("accept");
        ingest.inject_packet(sid, make_data_packet(0, b"abc"));
        let stats: Vec<_> = ingest.stream_stats().collect();
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].1.packets_received, 1);
    }
}
