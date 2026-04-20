//! QUIC transport abstraction layer.
//!
//! This module provides a pure-Rust, zero-external-dependency API abstraction
//! for QUIC-based media transport.  It models the QUIC connection and stream
//! lifecycle, flow control, and connection statistics without implementing the
//! actual QUIC protocol handshake — it serves as the scaffolding layer over
//! which a real QUIC implementation (e.g., quinn, quiche) can be integrated.
//!
//! # Key types
//!
//! - [`QuicConfig`] — transport configuration
//! - [`QuicConnection`] — manages streams for a single peer connection
//! - [`QuicStream`] — logical QUIC stream with direction and priority
//! - [`QuicStats`] — RTT estimation, packet loss, congestion window
//! - [`HlsOverQuic`] — convenience wrapper for LL-HLS over QUIC delivery
//!
//! # Example
//!
//! ```rust
//! use oximedia_net::quic::{QuicConfig, QuicConnection, StreamDirection};
//! use std::net::SocketAddr;
//!
//! let local: SocketAddr  = "127.0.0.1:0".parse().expect("valid addr");
//! let remote: SocketAddr = "203.0.113.1:4433".parse().expect("valid addr");
//! let mut conn = QuicConnection::new(QuicConfig::default(), local, remote);
//!
//! let stream_id = conn.open_stream(StreamDirection::UniSend, 128)
//!     .expect("stream opened");
//! assert_eq!(conn.active_streams().len(), 1);
//! ```

use std::collections::HashMap;
use std::net::SocketAddr;

// ─── Error type ───────────────────────────────────────────────────────────────

/// Errors that can occur during QUIC transport operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuicTransportError {
    /// The connection has been closed (locally or by the peer).
    ConnectionClosed,
    /// The specified stream was reset by the peer.
    StreamReset(u64),
    /// A flow-control limit has been violated.
    FlowControlViolation,
    /// A generic QUIC protocol error with a human-readable description.
    ProtocolError(String),
    /// The requested stream does not exist.
    StreamNotFound(u64),
    /// No more stream IDs are available.
    StreamLimitReached,
}

impl std::fmt::Display for QuicTransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConnectionClosed => write!(f, "QUIC connection closed"),
            Self::StreamReset(id) => write!(f, "QUIC stream {id} was reset by peer"),
            Self::FlowControlViolation => write!(f, "QUIC flow-control limit violated"),
            Self::ProtocolError(msg) => write!(f, "QUIC protocol error: {msg}"),
            Self::StreamNotFound(id) => write!(f, "QUIC stream {id} not found"),
            Self::StreamLimitReached => {
                write!(f, "QUIC max_streams limit reached")
            }
        }
    }
}

impl std::error::Error for QuicTransportError {}

// ─── Stream direction ─────────────────────────────────────────────────────────

/// Direction of a QUIC stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamDirection {
    /// Data can flow in both directions.
    Bidirectional,
    /// Only the local endpoint sends on this stream.
    UniSend,
    /// Only the remote endpoint sends on this stream.
    UniReceive,
}

impl StreamDirection {
    /// Returns `true` if this stream can send data locally.
    #[must_use]
    pub fn can_send(&self) -> bool {
        matches!(self, Self::Bidirectional | Self::UniSend)
    }

    /// Returns `true` if this stream can receive data.
    #[must_use]
    pub fn can_receive(&self) -> bool {
        matches!(self, Self::Bidirectional | Self::UniReceive)
    }
}

// ─── Stream ───────────────────────────────────────────────────────────────────

/// A logical QUIC stream.
///
/// QUIC stream IDs follow RFC 9000 §2.1:
/// - Bits 0–1 encode initiator (0 = client) and type (0 = bidirectional).
#[derive(Debug, Clone)]
pub struct QuicStream {
    /// Unique stream identifier within the connection.
    pub stream_id: u64,
    /// Direction of data flow.
    pub direction: StreamDirection,
    /// Application-level priority (0 = highest, 255 = lowest).
    pub priority: u8,
    /// Number of bytes sent on this stream.
    pub bytes_sent: u64,
    /// Number of bytes received on this stream.
    pub bytes_received: u64,
    /// Whether the stream has been half-closed (local side finished).
    pub fin_sent: bool,
}

impl QuicStream {
    fn new(stream_id: u64, direction: StreamDirection, priority: u8) -> Self {
        Self {
            stream_id,
            direction,
            priority,
            bytes_sent: 0,
            bytes_received: u64::MIN,
            fin_sent: false,
        }
    }
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// QUIC transport configuration.
#[derive(Debug, Clone)]
pub struct QuicConfig {
    /// Maximum concurrent streams (both directions combined).
    pub max_streams: u32,
    /// Connection idle timeout in milliseconds (0 = disabled).
    pub idle_timeout_ms: u64,
    /// Keep-alive probe interval in milliseconds (0 = disabled).
    pub keep_alive_interval_ms: u64,
    /// Connection-level flow control initial limit in bytes.
    pub initial_max_data: u64,
    /// Per-stream flow control initial limit in bytes.
    pub initial_max_stream_data: u64,
    /// Maximum UDP payload size in bytes (QUIC PMTU).
    pub max_udp_payload_size: u16,
    /// Whether DATAGRAM extension (RFC 9221) is enabled.
    pub enable_datagrams: bool,
}

impl Default for QuicConfig {
    fn default() -> Self {
        Self {
            max_streams: 100,
            idle_timeout_ms: 30_000,
            keep_alive_interval_ms: 10_000,
            initial_max_data: 10 * 1024 * 1024,       // 10 MiB
            initial_max_stream_data: 1 * 1024 * 1024, // 1 MiB
            max_udp_payload_size: 1200,
            enable_datagrams: false,
        }
    }
}

// ─── Statistics ───────────────────────────────────────────────────────────────

/// Connection-level QUIC statistics.
#[derive(Debug, Clone, Default)]
pub struct QuicStats {
    /// Smoothed round-trip time estimate in microseconds.
    pub rtt_us: u64,
    /// RTT variance (rttvar) in microseconds.
    pub rtt_var_us: u64,
    /// Estimated packet-loss rate in the range [0.0, 1.0].
    pub packet_loss_rate: f64,
    /// Current congestion window in bytes.
    pub cwnd_bytes: u64,
    /// Total packets sent.
    pub packets_sent: u64,
    /// Total packets received.
    pub packets_received: u64,
    /// Total bytes sent.
    pub bytes_sent: u64,
    /// Total bytes received.
    pub bytes_received: u64,
    /// Number of packet loss events (congestion signals).
    pub loss_events: u32,
}

impl QuicStats {
    /// Returns the smoothed RTT as a `std::time::Duration`.
    #[must_use]
    pub fn rtt(&self) -> std::time::Duration {
        std::time::Duration::from_micros(self.rtt_us)
    }

    /// Returns the minimum RTT estimate (here approximated as RTT − rttvar).
    #[must_use]
    pub fn min_rtt_us(&self) -> u64 {
        self.rtt_us.saturating_sub(self.rtt_var_us)
    }
}

// ─── Connection ───────────────────────────────────────────────────────────────

/// A simulated QUIC connection managing streams, flow control, and stats.
///
/// This is the primary API surface for QUIC-based media transport.
pub struct QuicConnection {
    config: QuicConfig,
    /// Local socket address.
    pub local_addr: SocketAddr,
    /// Remote peer socket address.
    pub remote_addr: SocketAddr,
    /// Opaque connection identifier.
    pub connection_id: u64,
    /// Active streams keyed by stream ID.
    streams: HashMap<u64, QuicStream>,
    /// Next stream ID to allocate.
    next_stream_id: u64,
    /// Whether the connection is still open.
    closed: bool,
    /// Mutable statistics (simulated in this scaffold).
    stats: QuicStats,
    /// Remaining connection-level flow control credit (bytes).
    flow_credit: u64,
}

impl QuicConnection {
    /// Creates a new QUIC connection scaffold.
    #[must_use]
    pub fn new(config: QuicConfig, local_addr: SocketAddr, remote_addr: SocketAddr) -> Self {
        let flow_credit = config.initial_max_data;
        // Generate a deterministic connection ID from address bytes for tests.
        let connection_id = {
            let b = remote_addr.ip().to_string();
            b.bytes().fold(0u64, |acc, byte| {
                acc.wrapping_mul(31).wrapping_add(byte as u64)
            })
        };
        Self {
            config,
            local_addr,
            remote_addr,
            connection_id,
            streams: HashMap::new(),
            next_stream_id: 0,
            closed: false,
            stats: QuicStats::default(),
            flow_credit,
        }
    }

    /// Opens a new QUIC stream with the given direction and priority.
    ///
    /// Returns the new stream's ID on success.
    ///
    /// # Errors
    ///
    /// - [`QuicTransportError::ConnectionClosed`] if the connection is closed.
    /// - [`QuicTransportError::StreamLimitReached`] if `max_streams` is exhausted.
    pub fn open_stream(
        &mut self,
        direction: StreamDirection,
        priority: u8,
    ) -> Result<u64, QuicTransportError> {
        if self.closed {
            return Err(QuicTransportError::ConnectionClosed);
        }
        if self.streams.len() >= self.config.max_streams as usize {
            return Err(QuicTransportError::StreamLimitReached);
        }

        let stream_id = self.next_stream_id;
        self.next_stream_id += 1;

        let stream = QuicStream::new(stream_id, direction, priority);
        self.streams.insert(stream_id, stream);
        Ok(stream_id)
    }

    /// Closes an existing stream, releasing its resources.
    ///
    /// # Errors
    ///
    /// - [`QuicTransportError::StreamNotFound`] if the stream ID is unknown.
    /// - [`QuicTransportError::ConnectionClosed`] if the connection is closed.
    pub fn close_stream(&mut self, stream_id: u64) -> Result<(), QuicTransportError> {
        if self.closed {
            return Err(QuicTransportError::ConnectionClosed);
        }
        if self.streams.remove(&stream_id).is_none() {
            return Err(QuicTransportError::StreamNotFound(stream_id));
        }
        Ok(())
    }

    /// Returns references to all currently open streams.
    #[must_use]
    pub fn active_streams(&self) -> Vec<&QuicStream> {
        self.streams.values().collect()
    }

    /// Returns a mutable reference to a stream by ID.
    pub fn stream_mut(&mut self, stream_id: u64) -> Option<&mut QuicStream> {
        self.streams.get_mut(&stream_id)
    }

    /// Simulates sending `bytes` on `stream_id`, consuming flow-control credit.
    ///
    /// # Errors
    ///
    /// - [`QuicTransportError::StreamNotFound`] if the stream does not exist.
    /// - [`QuicTransportError::FlowControlViolation`] if the connection-level
    ///   flow credit is exhausted.
    pub fn send_bytes(&mut self, stream_id: u64, bytes: u64) -> Result<(), QuicTransportError> {
        if self.closed {
            return Err(QuicTransportError::ConnectionClosed);
        }
        if !self.streams.contains_key(&stream_id) {
            return Err(QuicTransportError::StreamNotFound(stream_id));
        }
        if bytes > self.flow_credit {
            return Err(QuicTransportError::FlowControlViolation);
        }
        self.flow_credit -= bytes;
        let stream = self
            .streams
            .get_mut(&stream_id)
            .ok_or(QuicTransportError::StreamNotFound(stream_id))?;
        stream.bytes_sent += bytes;
        self.stats.bytes_sent += bytes;
        self.stats.packets_sent += 1;
        Ok(())
    }

    /// Simulates receiving `bytes` on `stream_id`.
    ///
    /// # Errors
    ///
    /// - [`QuicTransportError::StreamNotFound`] if the stream does not exist.
    pub fn receive_bytes(&mut self, stream_id: u64, bytes: u64) -> Result<(), QuicTransportError> {
        if self.closed {
            return Err(QuicTransportError::ConnectionClosed);
        }
        let stream = self
            .streams
            .get_mut(&stream_id)
            .ok_or(QuicTransportError::StreamNotFound(stream_id))?;
        stream.bytes_received = stream.bytes_received.saturating_add(bytes);
        self.stats.bytes_received += bytes;
        self.stats.packets_received += 1;
        Ok(())
    }

    /// Updates the simulated RTT estimate using RFC 6298 EWMA.
    ///
    /// `sample_us` is the one-way RTT sample in microseconds.
    pub fn update_rtt(&mut self, sample_us: u64) {
        if self.stats.rtt_us == 0 {
            self.stats.rtt_us = sample_us;
            self.stats.rtt_var_us = sample_us / 2;
        } else {
            // RFC 6298: RTTVAR = (1 - β) * RTTVAR + β * |SRTT - R'|
            //           SRTT   = (1 - α) * SRTT   + α * R'
            // α = 1/8, β = 1/4 (standard QUIC values).
            let diff = if sample_us > self.stats.rtt_us {
                sample_us - self.stats.rtt_us
            } else {
                self.stats.rtt_us - sample_us
            };
            self.stats.rtt_var_us = (3 * self.stats.rtt_var_us + diff) / 4;
            self.stats.rtt_us = (7 * self.stats.rtt_us + sample_us) / 8;
        }
    }

    /// Signals a congestion event (packet loss detected).
    ///
    /// Halves the congestion window (multiplicative decrease).
    pub fn signal_loss_event(&mut self) {
        self.stats.loss_events += 1;
        self.stats.packet_loss_rate =
            self.stats.loss_events as f64 / (self.stats.packets_sent.max(1) as f64);
        // Halve the CWND (minimum 2 × max_udp_payload_size).
        let min_cwnd = 2 * self.config.max_udp_payload_size as u64;
        self.stats.cwnd_bytes = (self.stats.cwnd_bytes / 2).max(min_cwnd);
    }

    /// Increases the congestion window for slow-start / congestion avoidance.
    pub fn increase_cwnd(&mut self, bytes: u64) {
        self.stats.cwnd_bytes += bytes;
    }

    /// Closes the connection, preventing further stream operations.
    pub fn close(&mut self) {
        self.closed = true;
        self.streams.clear();
    }

    /// Returns `true` if the connection is open.
    #[must_use]
    pub fn is_open(&self) -> bool {
        !self.closed
    }

    /// Returns a snapshot of the current connection statistics.
    #[must_use]
    pub fn connection_stats(&self) -> QuicStats {
        self.stats.clone()
    }

    /// Returns the remaining connection-level flow control credit.
    #[must_use]
    pub fn flow_credit(&self) -> u64 {
        self.flow_credit
    }

    /// Advances the connection-level max data (simulates MAX_DATA frame).
    pub fn extend_flow_credit(&mut self, additional: u64) {
        self.flow_credit = self.flow_credit.saturating_add(additional);
    }
}

// ─── HLS-over-QUIC wrapper ────────────────────────────────────────────────────

/// Wraps a [`QuicConnection`] to deliver LL-HLS segments and parts over
/// dedicated QUIC streams.
///
/// Each segment or partial segment is assigned its own stream so that
/// HTTP/3-style independent delivery and prioritisation can be modelled.
pub struct HlsOverQuic {
    conn: QuicConnection,
    /// Maps `segment_uri` → `stream_id`.
    segment_streams: HashMap<String, u64>,
    /// Maps `part_uri` → `stream_id`.
    part_streams: HashMap<String, u64>,
}

impl HlsOverQuic {
    /// Creates a new `HlsOverQuic` wrapper over an existing connection.
    #[must_use]
    pub fn new(conn: QuicConnection) -> Self {
        Self {
            conn,
            segment_streams: HashMap::new(),
            part_streams: HashMap::new(),
        }
    }

    /// Requests a new QUIC stream for the given segment URI.
    ///
    /// If a stream for this URI already exists, its ID is returned without
    /// allocating a new stream.
    ///
    /// # Errors
    ///
    /// Propagates any [`QuicTransportError`] from stream allocation.
    pub fn request_segment(&mut self, uri: &str) -> Result<u64, QuicTransportError> {
        if let Some(&id) = self.segment_streams.get(uri) {
            return Ok(id);
        }
        // Segments use high priority (0 = highest).
        let id = self.conn.open_stream(StreamDirection::UniReceive, 0)?;
        self.segment_streams.insert(uri.to_owned(), id);
        Ok(id)
    }

    /// Requests a new QUIC stream for the given partial segment URI.
    ///
    /// Parts are delivered at slightly lower priority than full segments to
    /// allow segment retransmissions to preempt in-progress part streams.
    ///
    /// # Errors
    ///
    /// Propagates any [`QuicTransportError`] from stream allocation.
    pub fn request_part(&mut self, uri: &str) -> Result<u64, QuicTransportError> {
        if let Some(&id) = self.part_streams.get(uri) {
            return Ok(id);
        }
        // Parts use medium priority.
        let id = self.conn.open_stream(StreamDirection::UniReceive, 64)?;
        self.part_streams.insert(uri.to_owned(), id);
        Ok(id)
    }

    /// Returns a reference to the underlying connection.
    #[must_use]
    pub fn connection(&self) -> &QuicConnection {
        &self.conn
    }

    /// Returns a mutable reference to the underlying connection.
    pub fn connection_mut(&mut self) -> &mut QuicConnection {
        &mut self.conn
    }

    /// Returns all currently mapped segment URIs and their stream IDs.
    #[must_use]
    pub fn segment_stream_ids(&self) -> &HashMap<String, u64> {
        &self.segment_streams
    }

    /// Returns all currently mapped part URIs and their stream IDs.
    #[must_use]
    pub fn part_stream_ids(&self) -> &HashMap<String, u64> {
        &self.part_streams
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_conn() -> QuicConnection {
        let local: SocketAddr = "127.0.0.1:0".parse().expect("valid addr");
        let remote: SocketAddr = "127.0.0.1:4433".parse().expect("valid addr");
        QuicConnection::new(QuicConfig::default(), local, remote)
    }

    // 1. Default config has sensible max_streams
    #[test]
    fn test_default_config_max_streams() {
        let cfg = QuicConfig::default();
        assert_eq!(cfg.max_streams, 100);
    }

    // 2. New connection is open
    #[test]
    fn test_new_connection_is_open() {
        let conn = make_conn();
        assert!(conn.is_open());
    }

    // 3. open_stream returns unique IDs
    #[test]
    fn test_open_stream_unique_ids() {
        let mut conn = make_conn();
        let id0 = conn
            .open_stream(StreamDirection::Bidirectional, 0)
            .expect("ok");
        let id1 = conn.open_stream(StreamDirection::UniSend, 0).expect("ok");
        assert_ne!(id0, id1);
    }

    // 4. active_streams count matches opened streams
    #[test]
    fn test_active_streams_count() {
        let mut conn = make_conn();
        conn.open_stream(StreamDirection::UniSend, 0).expect("ok");
        conn.open_stream(StreamDirection::UniReceive, 0)
            .expect("ok");
        assert_eq!(conn.active_streams().len(), 2);
    }

    // 5. close_stream removes the stream
    #[test]
    fn test_close_stream_removes_it() {
        let mut conn = make_conn();
        let id = conn.open_stream(StreamDirection::UniSend, 0).expect("ok");
        conn.close_stream(id).expect("close ok");
        assert_eq!(conn.active_streams().len(), 0);
    }

    // 6. close_stream on unknown ID returns StreamNotFound
    #[test]
    fn test_close_unknown_stream_error() {
        let mut conn = make_conn();
        let err = conn.close_stream(999).expect_err("should fail");
        assert_eq!(err, QuicTransportError::StreamNotFound(999));
    }

    // 7. send_bytes deducts from flow credit
    #[test]
    fn test_send_bytes_deducts_flow_credit() {
        let mut conn = make_conn();
        let id = conn.open_stream(StreamDirection::UniSend, 0).expect("ok");
        let initial = conn.flow_credit();
        conn.send_bytes(id, 1024).expect("ok");
        assert_eq!(conn.flow_credit(), initial - 1024);
    }

    // 8. send_bytes over credit returns FlowControlViolation
    #[test]
    fn test_send_bytes_flow_control_violation() {
        let mut conn = make_conn();
        let id = conn.open_stream(StreamDirection::UniSend, 0).expect("ok");
        let err = conn.send_bytes(id, u64::MAX).expect_err("must fail");
        assert_eq!(err, QuicTransportError::FlowControlViolation);
    }

    // 9. update_rtt initialises SRTT from first sample
    #[test]
    fn test_update_rtt_first_sample() {
        let mut conn = make_conn();
        conn.update_rtt(20_000);
        assert_eq!(conn.connection_stats().rtt_us, 20_000);
        assert_eq!(conn.connection_stats().rtt_var_us, 10_000);
    }

    // 10. update_rtt converges toward new value
    #[test]
    fn test_update_rtt_convergence() {
        let mut conn = make_conn();
        conn.update_rtt(20_000);
        for _ in 0..16 {
            conn.update_rtt(30_000);
        }
        // After many samples, SRTT should be close to 30 ms.
        let stats = conn.connection_stats();
        assert!(stats.rtt_us > 25_000, "rtt={}", stats.rtt_us);
    }

    // 11. signal_loss_event halves cwnd (after initial growth)
    #[test]
    fn test_signal_loss_halves_cwnd() {
        let mut conn = make_conn();
        conn.increase_cwnd(100_000);
        let before = conn.connection_stats().cwnd_bytes;
        conn.signal_loss_event();
        let after = conn.connection_stats().cwnd_bytes;
        assert!(after <= before / 2 + 2400); // within rounding + min_cwnd guard
    }

    // 12. close() marks connection closed and clears streams
    #[test]
    fn test_close_connection() {
        let mut conn = make_conn();
        conn.open_stream(StreamDirection::UniSend, 0).expect("ok");
        conn.close();
        assert!(!conn.is_open());
        assert_eq!(conn.active_streams().len(), 0);
    }

    // 13. open_stream on closed connection returns ConnectionClosed
    #[test]
    fn test_open_stream_on_closed_connection() {
        let mut conn = make_conn();
        conn.close();
        let err = conn
            .open_stream(StreamDirection::UniSend, 0)
            .expect_err("fail");
        assert_eq!(err, QuicTransportError::ConnectionClosed);
    }

    // 14. StreamDirection::can_send/can_receive semantics
    #[test]
    fn test_stream_direction_semantics() {
        assert!(StreamDirection::Bidirectional.can_send());
        assert!(StreamDirection::Bidirectional.can_receive());
        assert!(StreamDirection::UniSend.can_send());
        assert!(!StreamDirection::UniSend.can_receive());
        assert!(!StreamDirection::UniReceive.can_send());
        assert!(StreamDirection::UniReceive.can_receive());
    }

    // 15. QuicStats rtt() returns correct Duration
    #[test]
    fn test_stats_rtt_duration() {
        let stats = QuicStats {
            rtt_us: 15_000,
            ..Default::default()
        };
        assert_eq!(stats.rtt(), std::time::Duration::from_micros(15_000));
    }

    // 16. extend_flow_credit increases credit
    #[test]
    fn test_extend_flow_credit() {
        let mut conn = make_conn();
        let initial = conn.flow_credit();
        conn.extend_flow_credit(1_000_000);
        assert_eq!(conn.flow_credit(), initial + 1_000_000);
    }

    // 17. max_streams limit is enforced
    #[test]
    fn test_max_streams_limit() {
        let cfg = QuicConfig {
            max_streams: 2,
            ..QuicConfig::default()
        };
        let local: SocketAddr = "127.0.0.1:0".parse().expect("valid");
        let remote: SocketAddr = "127.0.0.1:4433".parse().expect("valid");
        let mut conn = QuicConnection::new(cfg, local, remote);
        conn.open_stream(StreamDirection::UniSend, 0).expect("ok");
        conn.open_stream(StreamDirection::UniSend, 0).expect("ok");
        let err = conn
            .open_stream(StreamDirection::UniSend, 0)
            .expect_err("must fail");
        assert_eq!(err, QuicTransportError::StreamLimitReached);
    }

    // 18. HlsOverQuic::request_segment opens stream and caches URI
    #[test]
    fn test_hls_over_quic_request_segment() {
        let conn = make_conn();
        let mut hoq = HlsOverQuic::new(conn);
        let id = hoq.request_segment("seg0.ts").expect("ok");
        assert_eq!(hoq.segment_stream_ids().len(), 1);
        // Second call with same URI returns the same stream ID.
        let id2 = hoq.request_segment("seg0.ts").expect("ok");
        assert_eq!(id, id2);
    }

    // 19. HlsOverQuic::request_part uses different priority than segment
    #[test]
    fn test_hls_over_quic_request_part_priority() {
        let conn = make_conn();
        let mut hoq = HlsOverQuic::new(conn);
        let seg_id = hoq.request_segment("seg0.ts").expect("ok");
        let part_id = hoq.request_part("part0.mp4").expect("ok");
        assert_ne!(seg_id, part_id);
        // Check priority via stream details.
        let streams = hoq.connection().active_streams();
        let seg_stream = streams
            .iter()
            .find(|s| s.stream_id == seg_id)
            .expect("found");
        let part_stream = streams
            .iter()
            .find(|s| s.stream_id == part_id)
            .expect("found");
        assert!(seg_stream.priority < part_stream.priority); // lower value = higher priority
    }

    // 20. QuicStats min_rtt saturates to zero when rttvar > rtt
    #[test]
    fn test_stats_min_rtt_saturates() {
        let stats = QuicStats {
            rtt_us: 5_000,
            rtt_var_us: 10_000,
            ..Default::default()
        };
        assert_eq!(stats.min_rtt_us(), 0);
    }
}
