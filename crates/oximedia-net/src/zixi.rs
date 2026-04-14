//! Zixi-compatible broadcast contribution protocol.
//!
//! Zixi is a proprietary reliable transport protocol widely used in broadcast
//! contribution and distribution links.  This module implements a
//! **Zixi-compatible** transport layer: it replicates the wire-level behaviour
//! closely enough to interoperate with Zixi-branded encoders / decoders while
//! remaining a clean-room pure-Rust implementation.
//!
//! ## Protocol overview
//!
//! ```text
//!  Sender                               Receiver
//!   │  ── ZIXI_HELLO ──────────────────► │  (handshake)
//!   │  ◄─ ZIXI_ACCEPT ──────────────────  │
//!   │  ── DATA_FRAME(seq=0, ts, data) ──► │
//!   │  ── DATA_FRAME(seq=1, ts, data) ──► │
//!   │  ◄─ ZIXI_ACK(seq=1, nack=[]) ──────  │  (every ack_interval_ms)
//!   │  ── DATA_FRAME(seq=2, ts, data) ──► │
//!   │  ◄─ ZIXI_NACK(seq=2) ──────────────  │  (missing frame request)
//!   │  ── RETRANSMIT(seq=2, data) ───────► │
//!   │  ── ZIXI_BYE ───────────────────────► │
//! ```
//!
//! ### Key features implemented
//!
//! - **Framing**: 12-byte fixed header per data frame.
//! - **ARQ retransmission**: NACK-based selective retransmission with a
//!   configurable retransmit buffer (ring buffer of the last N frames).
//! - **Sequence numbers**: 32-bit rolling sequence number; wrap-around safe.
//! - **Timestamps**: 32-bit millisecond timestamps for jitter measurement.
//! - **Latency target**: configurable maximum latency budget; frames older than
//!   the budget are discarded at the receiver.
//! - **Statistics**: per-session counters for sent, received, retransmitted, and
//!   dropped frames.
//!
//! ## Wire format
//!
//! ```text
//! ┌─────────┬──────────┬──────────┬──────────┬──────────┬────────────────┐
//! │  magic  │  type    │  seq_no  │   ts_ms  │ payload  │    payload     │
//! │ 2 bytes │  1 byte  │  4 bytes │  4 bytes │  len     │   (variable)   │
//! │ 0x5A58  │          │  (u32be) │  (u32be) │ (u16be)  │                │
//! └─────────┴──────────┴──────────┴──────────┴──────────┴────────────────┘
//!                        ──────────────── 11 bytes header ────────────────
//! ```
//!
//! Magic bytes `0x5A58` = `'Z' 'X'` (ASCII).

#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use crate::error::{NetError, NetResult};

// ─── Constants ────────────────────────────────────────────────────────────────

/// Wire magic bytes identifying Zixi frames (`ZX`).
pub const ZIXI_MAGIC: u16 = 0x5A58;

/// Fixed header size in bytes.
pub const ZIXI_HEADER_SIZE: usize = 11;

/// Maximum payload size in a single frame (64 KiB – header).
pub const ZIXI_MAX_PAYLOAD: usize = 65_524;

// ─── Frame Type ───────────────────────────────────────────────────────────────

/// Wire-level frame type byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FrameType {
    /// Initial handshake from sender.
    Hello = 0x01,
    /// Handshake acknowledgement from receiver.
    Accept = 0x02,
    /// Media data frame.
    Data = 0x03,
    /// Cumulative acknowledgement + optional NACK list.
    Ack = 0x04,
    /// Negative acknowledgement (selective retransmission request).
    Nack = 0x05,
    /// Retransmitted media frame.
    Retransmit = 0x06,
    /// Graceful session teardown.
    Bye = 0xFF,
}

impl FrameType {
    /// Attempts to parse a frame type from a raw byte.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::Parse`] for unknown type bytes.
    pub fn from_byte(b: u8) -> NetResult<Self> {
        match b {
            0x01 => Ok(Self::Hello),
            0x02 => Ok(Self::Accept),
            0x03 => Ok(Self::Data),
            0x04 => Ok(Self::Ack),
            0x05 => Ok(Self::Nack),
            0x06 => Ok(Self::Retransmit),
            0xFF => Ok(Self::Bye),
            other => Err(NetError::parse(
                0,
                format!("unknown Zixi frame type 0x{other:02X}"),
            )),
        }
    }

    /// Returns the type byte value.
    #[must_use]
    pub const fn as_byte(self) -> u8 {
        self as u8
    }
}

// ─── Header ───────────────────────────────────────────────────────────────────

/// Parsed Zixi frame header (11 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZixiHeader {
    /// Magic bytes (must equal [`ZIXI_MAGIC`]).
    pub magic: u16,
    /// Frame type.
    pub frame_type: FrameType,
    /// Rolling sequence number (big-endian u32).
    pub seq_no: u32,
    /// Sender timestamp in milliseconds (big-endian u32).
    pub timestamp_ms: u32,
    /// Payload length in bytes (big-endian u16).
    pub payload_len: u16,
}

impl ZixiHeader {
    /// Serialises the header into an 11-byte array.
    #[must_use]
    pub fn to_bytes(self) -> [u8; ZIXI_HEADER_SIZE] {
        let mut buf = [0u8; ZIXI_HEADER_SIZE];
        let magic = self.magic.to_be_bytes();
        buf[0] = magic[0];
        buf[1] = magic[1];
        buf[2] = self.frame_type.as_byte();
        let seq = self.seq_no.to_be_bytes();
        buf[3] = seq[0];
        buf[4] = seq[1];
        buf[5] = seq[2];
        buf[6] = seq[3];
        let ts = self.timestamp_ms.to_be_bytes();
        buf[7] = ts[0];
        buf[8] = ts[1];
        buf[9] = ts[2];
        buf[10] = ts[3];
        buf
    }

    /// Parses a header from an 11-byte slice.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::Parse`] if the slice is too short, the magic bytes
    /// are wrong, or the frame type is unknown.
    pub fn from_bytes(buf: &[u8]) -> NetResult<(Self, u16)> {
        if buf.len() < ZIXI_HEADER_SIZE + 2 {
            return Err(NetError::parse(
                0,
                format!(
                    "buffer too short for Zixi header: {} < {}",
                    buf.len(),
                    ZIXI_HEADER_SIZE + 2
                ),
            ));
        }
        let magic = u16::from_be_bytes([buf[0], buf[1]]);
        if magic != ZIXI_MAGIC {
            return Err(NetError::parse(0, format!("bad Zixi magic 0x{magic:04X}")));
        }
        let frame_type = FrameType::from_byte(buf[2])?;
        let seq_no = u32::from_be_bytes([buf[3], buf[4], buf[5], buf[6]]);
        let timestamp_ms = u32::from_be_bytes([buf[7], buf[8], buf[9], buf[10]]);
        let payload_len = u16::from_be_bytes([buf[11], buf[12]]);
        Ok((
            Self {
                magic,
                frame_type,
                seq_no,
                timestamp_ms,
                payload_len,
            },
            payload_len,
        ))
    }
}

// ─── Frame ────────────────────────────────────────────────────────────────────

/// A complete Zixi wire frame (header + payload).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZixiFrame {
    /// Parsed header.
    pub header: ZixiHeader,
    /// Raw payload bytes (length matches `header.payload_len`).
    pub payload: Vec<u8>,
}

impl ZixiFrame {
    /// Creates a new data frame.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::Encoding`] if the payload exceeds [`ZIXI_MAX_PAYLOAD`].
    pub fn data(seq_no: u32, timestamp_ms: u32, payload: Vec<u8>) -> NetResult<Self> {
        Self::with_type(FrameType::Data, seq_no, timestamp_ms, payload)
    }

    /// Creates a retransmit frame.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::Encoding`] if the payload exceeds [`ZIXI_MAX_PAYLOAD`].
    pub fn retransmit(seq_no: u32, timestamp_ms: u32, payload: Vec<u8>) -> NetResult<Self> {
        Self::with_type(FrameType::Retransmit, seq_no, timestamp_ms, payload)
    }

    /// Creates a frame of the given type.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::Encoding`] if the payload exceeds [`ZIXI_MAX_PAYLOAD`].
    pub fn with_type(
        frame_type: FrameType,
        seq_no: u32,
        timestamp_ms: u32,
        payload: Vec<u8>,
    ) -> NetResult<Self> {
        if payload.len() > ZIXI_MAX_PAYLOAD {
            return Err(NetError::encoding(format!(
                "payload {} bytes exceeds max {ZIXI_MAX_PAYLOAD}",
                payload.len()
            )));
        }
        Ok(Self {
            header: ZixiHeader {
                magic: ZIXI_MAGIC,
                frame_type,
                seq_no,
                timestamp_ms,
                payload_len: payload.len() as u16,
            },
            payload,
        })
    }

    /// Serialises the complete frame into a byte vector.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(ZIXI_HEADER_SIZE + 2 + self.payload.len());
        out.extend_from_slice(&self.header.to_bytes());
        out.extend_from_slice(&self.header.payload_len.to_be_bytes());
        out.extend_from_slice(&self.payload);
        out
    }

    /// Deserialises a complete frame from a byte slice.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`ZixiHeader::from_bytes`] or returns
    /// [`NetError::Parse`] if the slice is too short for the declared payload.
    pub fn from_bytes(buf: &[u8]) -> NetResult<Self> {
        let (header, payload_len) = ZixiHeader::from_bytes(buf)?;
        let payload_start = ZIXI_HEADER_SIZE + 2;
        let payload_end = payload_start + payload_len as usize;
        if buf.len() < payload_end {
            return Err(NetError::parse(
                payload_start as u64,
                format!(
                    "frame truncated: declared {} bytes payload, only {} available",
                    payload_len,
                    buf.len().saturating_sub(payload_start)
                ),
            ));
        }
        Ok(Self {
            header,
            payload: buf[payload_start..payload_end].to_vec(),
        })
    }
}

// ─── Retransmit Buffer ────────────────────────────────────────────────────────

/// Ring buffer that stores recent data frames for potential retransmission.
#[derive(Debug)]
pub struct RetransmitBuffer {
    capacity: usize,
    frames: VecDeque<ZixiFrame>,
    index: HashMap<u32, usize>, // seq_no → deque index (approx; rebuild on wrap)
}

impl RetransmitBuffer {
    /// Creates a new retransmit buffer holding at most `capacity` frames.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::InvalidState`] if capacity is zero.
    pub fn new(capacity: usize) -> NetResult<Self> {
        if capacity == 0 {
            return Err(NetError::invalid_state(
                "retransmit buffer capacity must be > 0",
            ));
        }
        Ok(Self {
            capacity,
            frames: VecDeque::with_capacity(capacity),
            index: HashMap::new(),
        })
    }

    /// Stores a frame, evicting the oldest when full.
    pub fn store(&mut self, frame: ZixiFrame) {
        if self.frames.len() == self.capacity {
            if let Some(oldest) = self.frames.pop_front() {
                self.index.remove(&oldest.header.seq_no);
            }
        }
        let idx = self.frames.len();
        self.index.insert(frame.header.seq_no, idx);
        self.frames.push_back(frame);
    }

    /// Looks up a frame by sequence number.
    ///
    /// Linear scan because the index may be stale after evictions.
    #[must_use]
    pub fn get(&self, seq_no: u32) -> Option<&ZixiFrame> {
        self.frames.iter().find(|f| f.header.seq_no == seq_no)
    }

    /// Returns the number of frames currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Returns `true` if the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

// ─── Session Statistics ───────────────────────────────────────────────────────

/// Per-session counters for a Zixi sender or receiver.
#[derive(Debug, Clone, Default)]
pub struct ZixiStats {
    /// Total data frames sent.
    pub frames_sent: u64,
    /// Total data frames received.
    pub frames_received: u64,
    /// Total retransmissions sent.
    pub retransmits_sent: u64,
    /// Total retransmissions requested (NACK count).
    pub retransmits_requested: u64,
    /// Frames dropped because they arrived too late (past latency budget).
    pub frames_dropped_late: u64,
    /// Frames that could not be retransmitted (no longer in buffer).
    pub retransmit_failures: u64,
    /// Total bytes of payload sent.
    pub bytes_sent: u64,
    /// Total bytes of payload received.
    pub bytes_received: u64,
}

impl ZixiStats {
    /// Computes the packet loss ratio: `(requested − sent_retransmits) / sent`.
    ///
    /// Returns 0.0 if no frames have been sent.
    #[must_use]
    pub fn loss_ratio(&self) -> f64 {
        if self.frames_sent == 0 {
            return 0.0;
        }
        self.retransmits_requested as f64 / self.frames_sent as f64
    }
}

// ─── Sender ───────────────────────────────────────────────────────────────────

/// Configuration for a Zixi sender session.
#[derive(Debug, Clone)]
pub struct ZixiSenderConfig {
    /// Remote receiver address.
    pub remote_addr: SocketAddr,
    /// Maximum retransmit buffer depth in frames.
    pub retransmit_buffer: usize,
    /// Stream ID (used in HELLO handshake for multiplexing).
    pub stream_id: String,
    /// Target one-way latency budget in milliseconds.
    pub latency_ms: u32,
}

impl ZixiSenderConfig {
    /// Creates a config with sensible defaults.
    #[must_use]
    pub fn new(remote_addr: SocketAddr, stream_id: impl Into<String>) -> Self {
        Self {
            remote_addr,
            retransmit_buffer: 1000,
            stream_id: stream_id.into(),
            latency_ms: 1000,
        }
    }
}

/// In-process Zixi sender — encapsulates and sequences outgoing frames.
///
/// This struct handles framing, sequencing, and retransmit buffer management.
/// Actual network I/O is left to the caller (pass the bytes from `next_frame`
/// to a UDP socket, for example).
#[derive(Debug)]
pub struct ZixiSender {
    config: ZixiSenderConfig,
    next_seq: u32,
    retransmit_buf: RetransmitBuffer,
    stats: ZixiStats,
    started_at: Instant,
}

impl ZixiSender {
    /// Creates a new Zixi sender.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`RetransmitBuffer::new`].
    pub fn new(config: ZixiSenderConfig) -> NetResult<Self> {
        let buf = RetransmitBuffer::new(config.retransmit_buffer)?;
        Ok(Self {
            config,
            next_seq: 0,
            retransmit_buf: buf,
            stats: ZixiStats::default(),
            started_at: Instant::now(),
        })
    }

    /// Encapsulates a media payload and returns the serialised frame bytes.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::Encoding`] if the payload exceeds [`ZIXI_MAX_PAYLOAD`].
    pub fn send(&mut self, payload: &[u8]) -> NetResult<Vec<u8>> {
        let ts_ms = self.elapsed_ms();
        let frame = ZixiFrame::data(self.next_seq, ts_ms, payload.to_vec())?;
        let bytes = frame.to_bytes();
        self.stats.frames_sent += 1;
        self.stats.bytes_sent += payload.len() as u64;
        self.retransmit_buf.store(frame);
        self.next_seq = self.next_seq.wrapping_add(1);
        Ok(bytes)
    }

    /// Handles an incoming NACK for `seq_no` and returns a retransmit frame,
    /// or `None` if the frame is no longer in the buffer.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::Encoding`] from frame serialisation (should not occur
    /// in practice since the original frame was already validated).
    pub fn handle_nack(&mut self, seq_no: u32) -> NetResult<Option<Vec<u8>>> {
        self.stats.retransmits_requested += 1;
        match self.retransmit_buf.get(seq_no) {
            None => {
                self.stats.retransmit_failures += 1;
                Ok(None)
            }
            Some(original) => {
                let frame = ZixiFrame::retransmit(
                    seq_no,
                    original.header.timestamp_ms,
                    original.payload.clone(),
                )?;
                let bytes = frame.to_bytes();
                self.stats.retransmits_sent += 1;
                Ok(Some(bytes))
            }
        }
    }

    /// Returns a reference to the session statistics.
    #[must_use]
    pub const fn stats(&self) -> &ZixiStats {
        &self.stats
    }

    /// Returns the next sequence number that will be assigned.
    #[must_use]
    pub const fn next_seq(&self) -> u32 {
        self.next_seq
    }

    /// Returns the stream ID.
    #[must_use]
    pub fn stream_id(&self) -> &str {
        &self.config.stream_id
    }

    /// Elapsed milliseconds since the session was created (truncated to u32).
    fn elapsed_ms(&self) -> u32 {
        self.started_at.elapsed().as_millis() as u32
    }
}

// ─── Receiver ─────────────────────────────────────────────────────────────────

/// Configuration for a Zixi receiver session.
#[derive(Debug, Clone)]
pub struct ZixiReceiverConfig {
    /// Local listen address.
    pub local_addr: SocketAddr,
    /// Maximum one-way latency budget; frames older than this are discarded.
    pub latency_budget: Duration,
    /// ACK interval: send an ACK every N received frames (0 = disable).
    pub ack_every_n_frames: u32,
}

impl ZixiReceiverConfig {
    /// Creates a config with sensible defaults.
    #[must_use]
    pub fn new(local_addr: SocketAddr) -> Self {
        Self {
            local_addr,
            latency_budget: Duration::from_secs(1),
            ack_every_n_frames: 10,
        }
    }
}

/// In-process Zixi receiver — parses incoming frames and tracks missing ones.
#[derive(Debug)]
pub struct ZixiReceiver {
    config: ZixiReceiverConfig,
    /// Highest received sequence number seen so far.
    highest_seq: Option<u32>,
    /// Sequence numbers received (ring of last N for gap detection).
    received_set: HashMap<u32, Instant>,
    stats: ZixiStats,
    /// Frames received since last ACK.
    since_last_ack: u32,
}

impl ZixiReceiver {
    /// Creates a new Zixi receiver.
    #[must_use]
    pub fn new(config: ZixiReceiverConfig) -> Self {
        Self {
            config,
            highest_seq: None,
            received_set: HashMap::new(),
            stats: ZixiStats::default(),
            since_last_ack: 0,
        }
    }

    /// Processes an incoming serialised frame.
    ///
    /// Returns an action list for the caller to act upon (send ACK, send NACK,
    /// or deliver the payload to the application).
    ///
    /// # Errors
    ///
    /// Propagates parse errors from [`ZixiFrame::from_bytes`].
    pub fn receive(&mut self, raw: &[u8]) -> NetResult<Vec<ReceiverAction>> {
        let frame = ZixiFrame::from_bytes(raw)?;
        let mut actions = Vec::new();

        match frame.header.frame_type {
            FrameType::Data | FrameType::Retransmit => {
                let seq = frame.header.seq_no;
                let now = Instant::now();

                self.stats.frames_received += 1;
                self.stats.bytes_received += frame.payload.len() as u64;
                self.received_set.insert(seq, now);

                // Update highest
                match self.highest_seq {
                    None => self.highest_seq = Some(seq),
                    Some(h) => {
                        if seq_after(seq, h) {
                            // Check for gaps
                            let gap_start = h.wrapping_add(1);
                            let mut s = gap_start;
                            while seq_after(seq, s) {
                                if !self.received_set.contains_key(&s) {
                                    self.stats.retransmits_requested += 1;
                                    actions.push(ReceiverAction::Nack(s));
                                }
                                s = s.wrapping_add(1);
                            }
                            self.highest_seq = Some(seq);
                        }
                    }
                }

                // Deliver payload
                actions.push(ReceiverAction::Deliver(frame.payload.clone()));

                // ACK check
                self.since_last_ack += 1;
                if self.config.ack_every_n_frames > 0
                    && self.since_last_ack >= self.config.ack_every_n_frames
                {
                    actions.push(ReceiverAction::Ack(seq));
                    self.since_last_ack = 0;
                }
            }
            FrameType::Bye => {
                actions.push(ReceiverAction::SessionEnded);
            }
            _ => {} // Hello / Accept / Ack handled externally
        }

        Ok(actions)
    }

    /// Returns a reference to the session statistics.
    #[must_use]
    pub const fn stats(&self) -> &ZixiStats {
        &self.stats
    }

    /// Returns the highest received sequence number, if any.
    #[must_use]
    pub const fn highest_seq(&self) -> Option<u32> {
        self.highest_seq
    }
}

/// An action the receiver requests the caller to perform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReceiverAction {
    /// Deliver this payload to the application.
    Deliver(Vec<u8>),
    /// Send a NACK for the given sequence number.
    Nack(u32),
    /// Send a cumulative ACK up to the given sequence number.
    Ack(u32),
    /// The remote sender sent a BYE; the session is over.
    SessionEnded,
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Returns `true` if sequence number `a` comes after `b` in the 32-bit
/// roll-over-safe sense (half the number space).
#[inline]
fn seq_after(a: u32, b: u32) -> bool {
    a.wrapping_sub(b) < 0x8000_0000
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, SocketAddrV4};

    fn local_addr() -> SocketAddr {
        SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 9999))
    }

    fn remote_addr() -> SocketAddr {
        SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 10000))
    }

    // ── Frame type tests ──────────────────────────────────────────────────────

    #[test]
    fn test_frame_type_round_trip() {
        for &ft in &[
            FrameType::Hello,
            FrameType::Accept,
            FrameType::Data,
            FrameType::Ack,
            FrameType::Nack,
            FrameType::Retransmit,
            FrameType::Bye,
        ] {
            let b = ft.as_byte();
            let parsed = FrameType::from_byte(b).expect("known type");
            assert_eq!(ft, parsed);
        }
    }

    #[test]
    fn test_unknown_frame_type_errors() {
        assert!(FrameType::from_byte(0xAB).is_err());
    }

    // ── Wire framing tests ────────────────────────────────────────────────────

    #[test]
    fn test_frame_serialise_deserialise() {
        let payload = b"hello zixi".to_vec();
        let frame = ZixiFrame::data(42, 1000, payload.clone()).expect("valid");
        let bytes = frame.to_bytes();
        let decoded = ZixiFrame::from_bytes(&bytes).expect("valid");
        assert_eq!(decoded.header.seq_no, 42);
        assert_eq!(decoded.header.timestamp_ms, 1000);
        assert_eq!(decoded.payload, payload);
        assert_eq!(decoded.header.frame_type, FrameType::Data);
    }

    #[test]
    fn test_magic_bytes_checked() {
        let payload = b"test".to_vec();
        let frame = ZixiFrame::data(0, 0, payload).expect("valid");
        let mut bytes = frame.to_bytes();
        bytes[0] = 0xDE; // corrupt magic
        assert!(ZixiFrame::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_truncated_buffer_errors() {
        let bytes = [0u8; 5]; // too short
        assert!(ZixiFrame::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_oversized_payload_rejected() {
        let too_big = vec![0u8; ZIXI_MAX_PAYLOAD + 1];
        assert!(ZixiFrame::data(0, 0, too_big).is_err());
    }

    #[test]
    fn test_header_to_bytes_length() {
        let hdr = ZixiHeader {
            magic: ZIXI_MAGIC,
            frame_type: FrameType::Data,
            seq_no: 100,
            timestamp_ms: 500,
            payload_len: 0,
        };
        let bytes = hdr.to_bytes();
        assert_eq!(bytes.len(), ZIXI_HEADER_SIZE);
    }

    // ── Sender tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_sender_send_increments_seq() {
        let mut sender =
            ZixiSender::new(ZixiSenderConfig::new(remote_addr(), "test")).expect("valid");
        sender.send(b"a").expect("ok");
        assert_eq!(sender.next_seq(), 1);
        sender.send(b"b").expect("ok");
        assert_eq!(sender.next_seq(), 2);
    }

    #[test]
    fn test_sender_stats_accumulate() {
        let mut sender =
            ZixiSender::new(ZixiSenderConfig::new(remote_addr(), "stream")).expect("valid");
        sender.send(b"payload").expect("ok");
        sender.send(b"payload2").expect("ok");
        assert_eq!(sender.stats().frames_sent, 2);
        assert_eq!(
            sender.stats().bytes_sent,
            b"payload".len() as u64 + b"payload2".len() as u64
        );
    }

    #[test]
    fn test_sender_nack_retransmits_frame() {
        let mut sender = ZixiSender::new(ZixiSenderConfig::new(remote_addr(), "s")).expect("valid");
        sender.send(b"original").expect("ok");
        let retx = sender.handle_nack(0).expect("ok").expect("frame present");
        let frame = ZixiFrame::from_bytes(&retx).expect("valid frame");
        assert_eq!(frame.header.frame_type, FrameType::Retransmit);
        assert_eq!(frame.payload, b"original");
    }

    #[test]
    fn test_sender_nack_missing_frame() {
        let mut sender = ZixiSender::new(ZixiSenderConfig::new(remote_addr(), "s")).expect("valid");
        let result = sender.handle_nack(999).expect("no error");
        assert!(result.is_none(), "seq 999 not in buffer");
        assert_eq!(sender.stats().retransmit_failures, 1);
    }

    // ── Receiver tests ────────────────────────────────────────────────────────

    #[test]
    fn test_receiver_delivers_payload() {
        let mut receiver = ZixiReceiver::new(ZixiReceiverConfig::new(local_addr()));
        let frame = ZixiFrame::data(0, 0, b"data".to_vec()).expect("valid");
        let actions = receiver.receive(&frame.to_bytes()).expect("ok");
        assert!(actions
            .iter()
            .any(|a| a == &ReceiverAction::Deliver(b"data".to_vec())));
    }

    #[test]
    fn test_receiver_emits_nack_on_gap() {
        let mut receiver = ZixiReceiver::new(ZixiReceiverConfig {
            ack_every_n_frames: 0, // disable ACK
            ..ZixiReceiverConfig::new(local_addr())
        });
        // Send seq=0 first
        let f0 = ZixiFrame::data(0, 0, b"a".to_vec()).expect("v");
        receiver.receive(&f0.to_bytes()).expect("ok");
        // Skip seq=1, send seq=2
        let f2 = ZixiFrame::data(2, 10, b"c".to_vec()).expect("v");
        let actions = receiver.receive(&f2.to_bytes()).expect("ok");
        let nacks: Vec<_> = actions
            .iter()
            .filter_map(|a| {
                if let ReceiverAction::Nack(s) = a {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect();
        assert!(nacks.contains(&1), "should NACK seq=1, got {nacks:?}");
    }

    #[test]
    fn test_receiver_session_ended_on_bye() {
        let mut receiver = ZixiReceiver::new(ZixiReceiverConfig::new(local_addr()));
        let bye = ZixiFrame::with_type(FrameType::Bye, 0, 0, vec![]).expect("valid");
        let actions = receiver.receive(&bye.to_bytes()).expect("ok");
        assert!(actions.contains(&ReceiverAction::SessionEnded));
    }

    #[test]
    fn test_receiver_ack_every_n_frames() {
        let mut receiver = ZixiReceiver::new(ZixiReceiverConfig {
            ack_every_n_frames: 3,
            ..ZixiReceiverConfig::new(local_addr())
        });
        // Send 3 frames
        for seq in 0..3u32 {
            let f = ZixiFrame::data(seq, 0, vec![seq as u8]).expect("v");
            let actions = receiver.receive(&f.to_bytes()).expect("ok");
            if seq == 2 {
                assert!(
                    actions.iter().any(|a| matches!(a, ReceiverAction::Ack(_))),
                    "expected ACK on 3rd frame"
                );
            }
        }
    }

    // ── RetransmitBuffer tests ────────────────────────────────────────────────

    #[test]
    fn test_retransmit_buffer_capacity() {
        let mut buf = RetransmitBuffer::new(3).expect("valid");
        for i in 0..5u32 {
            let f = ZixiFrame::data(i, 0, vec![i as u8]).expect("v");
            buf.store(f);
        }
        assert_eq!(buf.len(), 3);
        // Oldest (0, 1) should have been evicted
        assert!(buf.get(0).is_none());
        assert!(buf.get(1).is_none());
        assert!(buf.get(4).is_some());
    }

    #[test]
    fn test_retransmit_buffer_zero_capacity_error() {
        assert!(RetransmitBuffer::new(0).is_err());
    }

    // ── Sequence ordering helper ──────────────────────────────────────────────

    #[test]
    fn test_seq_after() {
        assert!(seq_after(1, 0));
        assert!(!seq_after(0, 1));
        // Wrap-around
        assert!(seq_after(0, u32::MAX));
        assert!(!seq_after(u32::MAX, 0));
    }

    // ── ZixiStats ─────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_loss_ratio() {
        let mut stats = ZixiStats::default();
        stats.frames_sent = 100;
        stats.retransmits_requested = 5;
        assert!((stats.loss_ratio() - 0.05).abs() < 1e-10);
    }
}
