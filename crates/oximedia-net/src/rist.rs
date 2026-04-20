//! RIST (Reliable Internet Stream Transport) protocol implementation.
//!
//! RIST is a standardised protocol (VSF TR-06-1 / SMPTE ST 2022-17) designed
//! for reliable media transport over unmanaged networks.  It builds on top of
//! RTP/UDP and adds:
//!
//! - **Null-packet deletion** — removes MPEG-TS stuffing packets to save bandwidth.
//! - **Selective ARQ** — selective retransmission via RTCP NACK (RFC 4585).
//! - **Sequence-based loss detection** — sequence numbers on every RTP packet.
//! - **Buffer management** — configurable buffer for reorder and jitter.
//! - **Simple Profile (TR-06-2)** and **Main Profile (TR-06-1)** support.
//!
//! This implementation provides:
//! - Packet encapsulation / de-encapsulation
//! - NACK generation and retransmission tracking
//! - Reorder buffer with configurable latency
//! - RIST flow statistics

#![allow(dead_code)]

use std::collections::{BTreeMap, VecDeque};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::{Duration, Instant};

// ─── Profile ─────────────────────────────────────────────────────────────────

/// RIST protocol profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RistProfile {
    /// Simple profile (TR-06-2): best-effort + selective retransmission.
    Simple,
    /// Main profile (TR-06-1): adds bonding and DTLS encryption.
    Main,
    /// Advanced profile: adds QUIC transport.
    Advanced,
}

impl RistProfile {
    /// Returns the profile name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Simple => "RIST Simple Profile",
            Self::Main => "RIST Main Profile",
            Self::Advanced => "RIST Advanced Profile",
        }
    }
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// RIST flow configuration.
#[derive(Debug, Clone)]
pub struct RistConfig {
    /// Protocol profile.
    pub profile: RistProfile,
    /// Reorder / jitter buffer duration.
    pub buffer_duration: Duration,
    /// Maximum number of NACK retransmissions per packet.
    pub max_retransmissions: u32,
    /// NACK request interval.
    pub nack_interval: Duration,
    /// Whether null-packet deletion is enabled.
    pub null_packet_deletion: bool,
    /// Maximum sequence-number gap before declaring loss.
    pub max_seq_gap: u16,
    /// Local bind address.
    pub bind_addr: SocketAddr,
    /// Remote peer address (sender or receiver).
    pub peer_addr: Option<SocketAddr>,
}

impl Default for RistConfig {
    fn default() -> Self {
        Self {
            profile: RistProfile::Simple,
            buffer_duration: Duration::from_millis(500),
            max_retransmissions: 5,
            nack_interval: Duration::from_millis(20),
            null_packet_deletion: true,
            max_seq_gap: 200,
            bind_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0),
            peer_addr: None,
        }
    }
}

impl RistConfig {
    /// Creates a new configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the buffer duration.
    #[must_use]
    pub const fn with_buffer(mut self, d: Duration) -> Self {
        self.buffer_duration = d;
        self
    }

    /// Sets the peer address.
    #[must_use]
    pub fn with_peer(mut self, addr: SocketAddr) -> Self {
        self.peer_addr = Some(addr);
        self
    }

    /// Disables null-packet deletion.
    #[must_use]
    pub const fn without_npd(mut self) -> Self {
        self.null_packet_deletion = false;
        self
    }
}

// ─── RTP Packet ──────────────────────────────────────────────────────────────

/// Minimal RTP header fields used by RIST.
#[derive(Debug, Clone)]
pub struct RistPacket {
    /// RTP sequence number (16-bit, wrapping).
    pub sequence: u16,
    /// RTP timestamp.
    pub timestamp: u32,
    /// Synchronization source identifier.
    pub ssrc: u32,
    /// Payload type.
    pub payload_type: u8,
    /// Payload data.
    pub payload: Vec<u8>,
    /// Whether this packet was retransmitted (RTX).
    pub is_retransmission: bool,
    /// Arrival time at the receiver.
    pub arrived_at: Instant,
}

impl RistPacket {
    /// Creates a new RIST/RTP packet.
    #[must_use]
    pub fn new(sequence: u16, timestamp: u32, ssrc: u32, payload: Vec<u8>) -> Self {
        Self {
            sequence,
            timestamp,
            ssrc,
            payload_type: 33, // MPEG-TS
            payload,
            is_retransmission: false,
            arrived_at: Instant::now(),
        }
    }

    /// Serialises to a minimal RTP wire format (12-byte fixed header + payload).
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(12 + self.payload.len());
        // Byte 0: V=2, P=0, X=0, CC=0
        buf.push(0x80);
        // Byte 1: M=0, PT
        buf.push(self.payload_type & 0x7F);
        // Bytes 2-3: sequence number
        buf.extend_from_slice(&self.sequence.to_be_bytes());
        // Bytes 4-7: timestamp
        buf.extend_from_slice(&self.timestamp.to_be_bytes());
        // Bytes 8-11: SSRC
        buf.extend_from_slice(&self.ssrc.to_be_bytes());
        buf.extend_from_slice(&self.payload);
        buf
    }

    /// Parses an RTP packet from raw bytes.
    ///
    /// Returns `None` if the buffer is too short or the version field is wrong.
    #[must_use]
    pub fn decode(data: &[u8]) -> Option<Self> {
        if data.len() < 12 {
            return None;
        }
        let version = (data[0] >> 6) & 0x03;
        if version != 2 {
            return None;
        }
        let payload_type = data[1] & 0x7F;
        let sequence = u16::from_be_bytes([data[2], data[3]]);
        let timestamp = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
        let ssrc = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);
        let payload = data[12..].to_vec();
        Some(Self {
            sequence,
            timestamp,
            ssrc,
            payload_type,
            payload,
            is_retransmission: false,
            arrived_at: Instant::now(),
        })
    }
}

// ─── NACK ─────────────────────────────────────────────────────────────────────

/// A RIST NACK request for one or more missing packets.
#[derive(Debug, Clone)]
pub struct RistNack {
    /// SSRC of the flow being retransmitted.
    pub ssrc: u32,
    /// First missing sequence number.
    pub seq_start: u16,
    /// Number of consecutive missing packets after `seq_start`.
    pub seq_count: u16,
    /// When this NACK was generated.
    pub generated_at: Instant,
    /// Number of times this NACK has been sent.
    pub send_count: u32,
}

impl RistNack {
    /// Creates a NACK for a single missing sequence number.
    #[must_use]
    pub fn single(ssrc: u32, seq: u16) -> Self {
        Self {
            ssrc,
            seq_start: seq,
            seq_count: 1,
            generated_at: Instant::now(),
            send_count: 0,
        }
    }

    /// Creates a NACK for a range of missing sequences.
    #[must_use]
    pub fn range(ssrc: u32, seq_start: u16, seq_count: u16) -> Self {
        Self {
            ssrc,
            seq_start,
            seq_count,
            generated_at: Instant::now(),
            send_count: 0,
        }
    }

    /// Encodes as an RTCP NACK feedback packet (RFC 4585 §6.2.1).
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        // Fixed: V=2 P=0 FMT=1 (NACK), PT=205 (RTPFB), length=3 words.
        let mut buf = Vec::with_capacity(16);
        buf.push(0x81); // V=2, P=0, FMT=1
        buf.push(205); // RTPFB
        buf.extend_from_slice(&3u16.to_be_bytes()); // length in 32-bit words minus 1
        buf.extend_from_slice(&self.ssrc.to_be_bytes()); // sender SSRC (reuse)
        buf.extend_from_slice(&self.ssrc.to_be_bytes()); // media SSRC
                                                         // FCI: PID | BLP
        buf.extend_from_slice(&self.seq_start.to_be_bytes());
        // BLP: bitmask of up to 16 additional missing packets
        let blp: u16 = if self.seq_count > 1 {
            let ones = self.seq_count.saturating_sub(1).min(16);
            (1u16 << ones) - 1
        } else {
            0
        };
        buf.extend_from_slice(&blp.to_be_bytes());
        buf
    }
}

// ─── Reorder / Jitter Buffer ──────────────────────────────────────────────────

/// Reorder buffer for RIST receivers.
///
/// Holds incoming packets in sequence-number order and releases them
/// after the configured latency window expires, issuing NACKs for gaps.
pub struct RistReorderBuffer {
    /// Buffered packets keyed by sequence number.
    buffer: BTreeMap<u16, RistPacket>,
    /// Pending NACKs (seq → NACK).
    pending_nacks: BTreeMap<u16, RistNack>,
    /// Highest contiguous sequence number delivered.
    highest_delivered: Option<u16>,
    /// Buffer duration.
    buffer_duration: Duration,
    /// NACK interval.
    nack_interval: Duration,
    /// Max retransmissions.
    max_retransmissions: u32,
    /// SSRC this buffer tracks.
    ssrc: u32,
}

impl RistReorderBuffer {
    /// Creates a new reorder buffer.
    #[must_use]
    pub fn new(config: &RistConfig, ssrc: u32) -> Self {
        Self {
            buffer: BTreeMap::new(),
            pending_nacks: BTreeMap::new(),
            highest_delivered: None,
            buffer_duration: config.buffer_duration,
            nack_interval: config.nack_interval,
            max_retransmissions: config.max_retransmissions,
            ssrc,
        }
    }

    /// Inserts a received packet into the buffer.
    pub fn insert(&mut self, pkt: RistPacket) {
        let seq = pkt.sequence;
        // Cancel any pending NACK for this sequence (it arrived).
        self.pending_nacks.remove(&seq);
        self.buffer.insert(seq, pkt);

        // Issue NACKs for any gaps between last_delivered and the new packet.
        self.detect_gaps(seq);
    }

    /// Drains packets ready for delivery.
    ///
    /// A packet is ready when it is the next expected in sequence AND
    /// either the slot is filled or the buffer deadline has elapsed.
    pub fn drain_ready(&mut self) -> Vec<RistPacket> {
        let mut out = Vec::new();
        loop {
            let next_seq = match self.highest_delivered {
                None => {
                    if let Some((&seq, _)) = self.buffer.iter().next() {
                        seq
                    } else {
                        break;
                    }
                }
                Some(last) => last.wrapping_add(1),
            };

            if let Some(pkt) = self.buffer.remove(&next_seq) {
                self.highest_delivered = Some(next_seq);
                out.push(pkt);
            } else {
                // Check if the deadline for this slot has expired.
                // If we have a later packet that was buffered long ago, skip the gap.
                let gap_expired = self
                    .buffer
                    .iter()
                    .next()
                    .map(|(_, p)| p.arrived_at.elapsed() >= self.buffer_duration)
                    .unwrap_or(false);

                if gap_expired {
                    // Skip the missing sequence.
                    self.highest_delivered = Some(next_seq);
                } else {
                    break;
                }
            }
        }
        out
    }

    /// Returns NACKs that should be (re-)sent.
    pub fn pending_nacks(&mut self) -> Vec<RistNack> {
        let interval = self.nack_interval;
        let max_retx = self.max_retransmissions;

        let mut to_send = Vec::new();
        let mut to_remove = Vec::new();

        for (&seq, nack) in &mut self.pending_nacks {
            if nack.send_count >= max_retx {
                to_remove.push(seq);
                continue;
            }
            if nack.send_count == 0 || nack.generated_at.elapsed() >= interval {
                nack.send_count += 1;
                nack.generated_at = Instant::now();
                to_send.push(nack.clone());
            }
        }
        for seq in to_remove {
            self.pending_nacks.remove(&seq);
        }
        to_send
    }

    /// Returns the number of packets currently buffered.
    #[must_use]
    pub fn buffered_count(&self) -> usize {
        self.buffer.len()
    }

    /// Returns the number of pending NACKs.
    #[must_use]
    pub fn pending_nack_count(&self) -> usize {
        self.pending_nacks.len()
    }

    // ── Private ───────────────────────────────────────────────────────────────

    fn detect_gaps(&mut self, arrived_seq: u16) {
        let expected_start = match self.highest_delivered {
            None => return, // No baseline yet; gaps will be detected after first delivery.
            Some(last) => last.wrapping_add(1),
        };

        // Walk from expected_start up to (but not including) arrived_seq.
        let mut seq = expected_start;
        while seq != arrived_seq {
            if !self.buffer.contains_key(&seq) && !self.pending_nacks.contains_key(&seq) {
                self.pending_nacks
                    .insert(seq, RistNack::single(self.ssrc, seq));
            }
            seq = seq.wrapping_add(1);
        }
    }
}

// ─── Null-Packet Deletion ─────────────────────────────────────────────────────

/// Null-packet deletion / restoration utilities.
///
/// MPEG-TS null packets (PID 0x1FFF) are pure stuffing; removing them
/// before transmission and re-inserting them at the receiver reduces the
/// required network bitrate.
pub struct NullPacketFilter;

impl NullPacketFilter {
    /// Removes MPEG-TS null packets from an array of 188-byte MPEG-TS packets.
    ///
    /// Returns the filtered payload (fewer bytes when nulls are present).
    #[must_use]
    pub fn delete(ts_stream: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(ts_stream.len());
        for chunk in ts_stream.chunks_exact(188) {
            // Null PID = 0x1FFF: bytes 1-2 with top 5 bits of byte 1 + all of byte 2.
            let pid = u16::from(chunk[1] & 0x1F) << 8 | u16::from(chunk[2]);
            if chunk[0] == 0x47 && pid == 0x1FFF {
                continue; // Skip null packet.
            }
            out.extend_from_slice(chunk);
        }
        out
    }

    /// Re-inserts null packets to restore a 188-byte-aligned TS stream.
    ///
    /// `target_len` is the desired output length in bytes (rounded to 188).
    #[must_use]
    pub fn restore(ts_stream: &[u8], target_len: usize) -> Vec<u8> {
        let target_packets = target_len / 188;
        let input_packets = ts_stream.len() / 188;
        let nulls_needed = target_packets.saturating_sub(input_packets);

        let mut out = Vec::with_capacity(target_len);
        out.extend_from_slice(ts_stream);

        // Append null packets at the end.
        for _ in 0..nulls_needed {
            // 0x47 sync, PID 0x1FFF, CC=0, payload
            let mut null = [0u8; 188];
            null[0] = 0x47;
            null[1] = 0x1F;
            null[2] = 0xFF;
            null[3] = 0x10; // adaptation field control = payload only
            out.extend_from_slice(&null);
        }

        out
    }
}

// ─── Flow Statistics ──────────────────────────────────────────────────────────

/// Per-flow RIST receiver statistics.
#[derive(Debug, Clone, Default)]
pub struct RistFlowStats {
    /// Total packets received (including retransmissions).
    pub packets_received: u64,
    /// Total retransmitted packets received.
    pub retransmissions_received: u64,
    /// Total packets lost (NACKed but never recovered).
    pub packets_lost: u64,
    /// Total NACKs sent.
    pub nacks_sent: u64,
    /// Estimated bitrate in bits per second.
    pub estimated_bitrate_bps: f64,
    /// Packet loss ratio (0.0 – 1.0).
    pub loss_ratio: f64,
}

impl RistFlowStats {
    /// Updates the loss ratio.
    pub fn update_loss_ratio(&mut self) {
        let total = self.packets_received + self.packets_lost;
        if total > 0 {
            self.loss_ratio = self.packets_lost as f64 / total as f64;
        }
    }
}

// ─── RIST Sender ─────────────────────────────────────────────────────────────

/// RIST sender state machine.
///
/// Tracks sent packets in a retransmission buffer and handles incoming NACKs.
pub struct RistSender {
    /// Configuration.
    config: RistConfig,
    /// Next RTP sequence number.
    next_seq: u16,
    /// RTP timestamp clock (90 kHz for video).
    timestamp: u32,
    /// SSRC for this flow.
    ssrc: u32,
    /// Retransmission buffer: seq → (packet, send_time).
    retransmit_buffer: BTreeMap<u16, (RistPacket, Instant)>,
    /// Retransmission buffer size limit.
    max_buffer_packets: usize,
    /// Statistics.
    stats: RistFlowStats,
}

impl RistSender {
    /// Creates a new RIST sender.
    #[must_use]
    pub fn new(config: RistConfig, ssrc: u32) -> Self {
        Self {
            config,
            next_seq: 0,
            timestamp: 0,
            ssrc,
            retransmit_buffer: BTreeMap::new(),
            max_buffer_packets: 4096,
            stats: RistFlowStats::default(),
        }
    }

    /// Creates and queues a packet for sending.
    ///
    /// Returns the encoded bytes that should be sent over the network.
    pub fn send_packet(&mut self, payload: Vec<u8>) -> Vec<u8> {
        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);

        let pkt = RistPacket::new(seq, self.timestamp, self.ssrc, payload);
        self.timestamp = self.timestamp.wrapping_add(3003); // ~33 ms @ 90 kHz

        let encoded = pkt.encode();

        // Keep in retransmission buffer.
        while self.retransmit_buffer.len() >= self.max_buffer_packets {
            if let Some((&first_seq, _)) = self.retransmit_buffer.iter().next() {
                self.retransmit_buffer.remove(&first_seq);
            }
        }
        self.retransmit_buffer.insert(seq, (pkt, Instant::now()));
        self.stats.packets_received += 1;

        encoded
    }

    /// Handles an incoming NACK, returning packets to retransmit (if available).
    pub fn handle_nack(&mut self, nack: &RistNack) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        for i in 0..nack.seq_count {
            let seq = nack.seq_start.wrapping_add(i);
            if let Some((pkt, _)) = self.retransmit_buffer.get(&seq) {
                let mut retx = pkt.clone();
                retx.is_retransmission = true;
                out.push(retx.encode());
                self.stats.retransmissions_received += 1;
            }
        }
        out
    }

    /// Evicts retransmission buffer entries older than the buffer duration.
    pub fn evict_old_packets(&mut self) {
        let cutoff = self.config.buffer_duration;
        self.retransmit_buffer
            .retain(|_, (_, sent_at)| sent_at.elapsed() < cutoff);
    }

    /// Returns current statistics.
    #[must_use]
    pub fn stats(&self) -> &RistFlowStats {
        &self.stats
    }

    /// Returns the retransmission buffer size.
    #[must_use]
    pub fn buffer_size(&self) -> usize {
        self.retransmit_buffer.len()
    }
}

// ─── RIST Receiver ───────────────────────────────────────────────────────────

/// RIST receiver state machine.
///
/// Processes incoming RTP packets, manages the reorder buffer, and generates
/// NACK requests for missing packets.
pub struct RistReceiver {
    /// Configuration.
    config: RistConfig,
    /// Reorder / jitter buffer.
    buffer: RistReorderBuffer,
    /// Statistics.
    stats: RistFlowStats,
    /// Received packet queue (ready for the application).
    ready_queue: VecDeque<RistPacket>,
}

impl RistReceiver {
    /// Creates a new RIST receiver.
    #[must_use]
    pub fn new(config: RistConfig, ssrc: u32) -> Self {
        let buffer = RistReorderBuffer::new(&config, ssrc);
        Self {
            config,
            buffer,
            stats: RistFlowStats::default(),
            ready_queue: VecDeque::new(),
        }
    }

    /// Processes a raw UDP datagram, returns any application-ready payloads.
    pub fn receive_datagram(&mut self, data: &[u8]) -> Vec<Vec<u8>> {
        let pkt = match RistPacket::decode(data) {
            Some(p) => p,
            None => return Vec::new(),
        };

        if pkt.is_retransmission {
            self.stats.retransmissions_received += 1;
        }
        self.stats.packets_received += 1;

        self.buffer.insert(pkt);
        let ready = self.buffer.drain_ready();
        let mut payloads = Vec::with_capacity(ready.len());
        for p in ready {
            if self.config.null_packet_deletion {
                payloads.push(NullPacketFilter::restore(&p.payload, p.payload.len()));
            } else {
                payloads.push(p.payload.clone());
            }
            self.ready_queue.push_back(p);
        }
        payloads
    }

    /// Returns pending NACK packets that should be sent to the sender.
    pub fn nacks_to_send(&mut self) -> Vec<RistNack> {
        let nacks = self.buffer.pending_nacks();
        self.stats.nacks_sent += nacks.len() as u64;
        nacks
    }

    /// Returns the number of packets in the ready queue.
    #[must_use]
    pub fn ready_count(&self) -> usize {
        self.ready_queue.len()
    }

    /// Returns current statistics.
    #[must_use]
    pub fn stats(&self) -> &RistFlowStats {
        &self.stats
    }

    /// Returns the number of buffered (not yet delivered) packets.
    #[must_use]
    pub fn buffered_count(&self) -> usize {
        self.buffer.buffered_count()
    }
}

// ─── Task-Specified Public API ────────────────────────────────────────────────
//
// The following types and free functions implement the API specified in the
// RIST implementation task.  They sit alongside the richer `RistSender` /
// `RistReceiver` / `RistReorderBuffer` machinery above, sharing packet
// encoding helpers where possible.

use std::collections::HashSet;

use crate::error::NetError;

// ── RistPacketHeader ──────────────────────────────────────────────────────────

/// Minimal RTP header fields as specified by the RIST task API.
///
/// This is a plain-data view of the 12-byte fixed RTP header used in RIST
/// encapsulation; use [`parse_rtp_header`] and [`build_rtp_header`] to
/// convert to/from the wire format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RistPacketHeader {
    /// RTP sequence number (16-bit, wrapping).
    pub rtp_seq: u16,
    /// Synchronisation source identifier.
    pub ssrc: u32,
    /// RTP timestamp (clock units).
    pub timestamp: u32,
    /// RTP payload type (7-bit value).
    pub payload_type: u8,
    /// RTP marker bit.
    pub marker: bool,
}

/// Parses a minimal 12-byte RTP header.
///
/// Returns `Err` if `data` is shorter than 12 bytes or the RTP version
/// field is not 2.
pub fn parse_rtp_header(data: &[u8]) -> Result<RistPacketHeader, NetError> {
    if data.len() < 12 {
        return Err(NetError::Protocol(
            "RTP header too short (need 12 bytes)".into(),
        ));
    }
    let version = (data[0] >> 6) & 0x03;
    if version != 2 {
        return Err(NetError::Protocol(format!(
            "RTP version {version} unsupported (expected 2)"
        )));
    }
    let marker = (data[1] & 0x80) != 0;
    let payload_type = data[1] & 0x7F;
    let rtp_seq = u16::from_be_bytes([data[2], data[3]]);
    let timestamp = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
    let ssrc = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);
    Ok(RistPacketHeader {
        rtp_seq,
        ssrc,
        timestamp,
        payload_type,
        marker,
    })
}

/// Serialises a [`RistPacketHeader`] into the 12-byte fixed RTP header.
///
/// Extension, CSRC, and padding bits are always set to zero.  The version
/// field is set to 2 as required by RIST.
#[must_use]
pub fn build_rtp_header(header: &RistPacketHeader) -> [u8; 12] {
    let mut buf = [0u8; 12];
    // Byte 0: V=2, P=0, X=0, CC=0
    buf[0] = 0x80;
    // Byte 1: M bit | PT
    buf[1] = (u8::from(header.marker) << 7) | (header.payload_type & 0x7F);
    // Bytes 2-3: sequence number
    let seq_bytes = header.rtp_seq.to_be_bytes();
    buf[2] = seq_bytes[0];
    buf[3] = seq_bytes[1];
    // Bytes 4-7: timestamp
    let ts_bytes = header.timestamp.to_be_bytes();
    buf[4] = ts_bytes[0];
    buf[5] = ts_bytes[1];
    buf[6] = ts_bytes[2];
    buf[7] = ts_bytes[3];
    // Bytes 8-11: SSRC
    let ssrc_bytes = header.ssrc.to_be_bytes();
    buf[8] = ssrc_bytes[0];
    buf[9] = ssrc_bytes[1];
    buf[10] = ssrc_bytes[2];
    buf[11] = ssrc_bytes[3];
    buf
}

// ── Simplified task-API NACK ──────────────────────────────────────────────────

/// RIST NACK carrying an explicit list of sequence numbers to retransmit.
///
/// This is the task-API companion to [`RistNack`] which uses the compact
/// RFC 4585 BLP encoding.  For large-scale retransmission requests use the
/// range-based [`RistNack`] above.
#[derive(Debug, Clone)]
pub struct RistNackList {
    /// SSRC of the flow being retransmitted.
    pub ssrc: u32,
    /// Explicit list of requested retransmit sequence numbers.
    pub seq_numbers: Vec<u16>,
}

impl RistNackList {
    /// Creates a new NACK list.
    #[must_use]
    pub fn new(ssrc: u32, seq_numbers: Vec<u16>) -> Self {
        Self { ssrc, seq_numbers }
    }
}

// ── Task-API RistConfig ───────────────────────────────────────────────────────

/// Simplified RIST configuration as specified by the task API.
///
/// For full configuration including binding addresses and per-flow knobs see
/// [`RistConfig`] above.
#[derive(Debug, Clone)]
pub struct RistSimpleConfig {
    /// RIST protocol profile.
    pub profile: RistProfile,
    /// Retransmission buffer size in milliseconds.
    pub buffer_size_ms: u32,
    /// Maximum number of NACK retransmits per packet.
    pub max_retransmits: u32,
    /// Fraction of total bandwidth reserved for retransmissions (e.g. 0.05 = 5 %).
    pub overhead_bandwidth: f32,
}

impl Default for RistSimpleConfig {
    fn default() -> Self {
        Self {
            profile: RistProfile::Simple,
            buffer_size_ms: 500,
            max_retransmits: 5,
            overhead_bandwidth: 0.05,
        }
    }
}

impl RistSimpleConfig {
    /// Creates a new simple configuration with sensible defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Converts to the full [`RistConfig`] used by the sender/receiver internals.
    #[must_use]
    pub fn into_full_config(self) -> RistConfig {
        RistConfig {
            profile: self.profile,
            buffer_duration: Duration::from_millis(u64::from(self.buffer_size_ms)),
            max_retransmissions: self.max_retransmits,
            ..RistConfig::default()
        }
    }
}

// ── Task-API RistSender ───────────────────────────────────────────────────────

/// Task-API RIST sender.
///
/// Wraps the full [`RistSender`] and exposes the method signatures required
/// by the implementation task:
/// - [`Self::packetize`] — wraps a payload in an RTP/RIST packet.
/// - [`Self::handle_nack`] — returns retransmit payloads for a [`RistNackList`].
/// - [`Self::prune_buffer`] — removes packets older than `buffer_size_ms`.
pub struct RistTaskSender {
    inner: RistSender,
    ssrc: u32,
}

impl RistTaskSender {
    /// Creates a new sender from a simplified configuration.
    #[must_use]
    pub fn new(config: RistSimpleConfig, ssrc: u32) -> Self {
        let full_cfg = config.into_full_config();
        Self {
            inner: RistSender::new(full_cfg, ssrc),
            ssrc,
        }
    }

    /// Encapsulates `payload` in a RIST/RTP packet and returns the encoded bytes.
    ///
    /// The `timestamp` parameter sets the RTP timestamp field directly (caller
    /// is responsible for deriving it from the media clock).
    pub fn packetize(&mut self, payload: &[u8], timestamp: u32) -> Vec<u8> {
        // Override the internal timestamp for this packet.
        self.inner.timestamp = timestamp;
        self.inner.send_packet(payload.to_vec())
    }

    /// Returns retransmit payloads for all sequence numbers listed in `nack`.
    ///
    /// Sequences not present in the retransmission buffer are silently skipped.
    #[must_use]
    pub fn handle_nack(&self, nack: &RistNackList) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        for &seq in &nack.seq_numbers {
            if let Some((pkt, _)) = self.inner.retransmit_buffer.get(&seq) {
                let mut retx = pkt.clone();
                retx.is_retransmission = true;
                out.push(retx.encode());
            }
        }
        out
    }

    /// Removes retransmission-buffer entries whose RTP timestamp would place
    /// them beyond `max_age_ms` given `clock_rate` ticks per second.
    ///
    /// The age is computed as:
    /// ```text
    /// age_ms = (current_ts - pkt_ts) * 1000 / clock_rate
    /// ```
    /// where `current_ts` is the sender's latest RTP timestamp.
    pub fn prune_buffer(&mut self, max_age_ms: u32, clock_rate: u32) {
        if clock_rate == 0 {
            return;
        }
        // Snapshot the current timestamp before borrowing buffer mutably.
        let current_ts = self.inner.timestamp;
        let clock_rate_u64 = u64::from(clock_rate);
        let max_age_ms_u64 = u64::from(max_age_ms);
        self.inner.retransmit_buffer.retain(|_, (pkt, _)| {
            // Wrapping subtraction handles 32-bit timestamp rollover.
            let age_ticks = current_ts.wrapping_sub(pkt.timestamp) as u64;
            let age_ms = age_ticks.saturating_mul(1000) / clock_rate_u64;
            age_ms <= max_age_ms_u64
        });
    }

    /// Returns the current retransmission buffer depth.
    #[must_use]
    pub fn buffer_size(&self) -> usize {
        self.inner.buffer_size()
    }

    /// Returns current flow statistics.
    #[must_use]
    pub fn stats(&self) -> &RistFlowStats {
        self.inner.stats()
    }
}

// ── Task-API RistReceiver ─────────────────────────────────────────────────────

/// Task-API RIST receiver.
///
/// Maintains a set of received sequence numbers and a queue of detected
/// gaps, generating [`RistNackList`] messages for the sender.
pub struct RistTaskReceiver {
    /// Configuration.
    config: RistSimpleConfig,
    /// Set of received sequence numbers (bounded to `sequence_history`).
    received: HashSet<u16>,
    /// Next expected in-order sequence number.
    expected_seq: u16,
    /// Missing sequences with discovery timestamps for NACK generation.
    missing: VecDeque<(u16, Instant)>,
    /// SSRC this receiver is tracking.
    ssrc: u32,
    /// Whether the first packet has been received (to initialise `expected_seq`).
    initialised: bool,
}

impl RistTaskReceiver {
    /// Creates a new task-API receiver.
    #[must_use]
    pub fn new(config: RistSimpleConfig, ssrc: u32) -> Self {
        Self {
            config,
            received: HashSet::new(),
            expected_seq: 0,
            missing: VecDeque::new(),
            ssrc,
            initialised: false,
        }
    }

    /// Processes a raw RTP-encapsulated datagram.
    ///
    /// Returns `Ok(Some(payload))` for the next in-order packet, `Ok(None)`
    /// if the packet was buffered out-of-order, or `Err` on a parse failure.
    pub fn receive_packet(&mut self, data: &[u8]) -> Result<Option<Vec<u8>>, NetError> {
        let hdr = parse_rtp_header(data)?;
        if data.len() < 12 {
            return Err(NetError::Protocol("packet too short".into()));
        }
        let payload = data[12..].to_vec();
        let seq = hdr.rtp_seq;

        // Initialise expected sequence from first packet.
        if !self.initialised {
            self.expected_seq = seq;
            self.initialised = true;
        }

        // Duplicate detection.
        if self.received.contains(&seq) {
            return Ok(None);
        }
        self.received.insert(seq);

        // Cancel any outstanding NACK for this sequence.
        self.missing.retain(|(s, _)| *s != seq);

        if seq == self.expected_seq {
            self.expected_seq = self.expected_seq.wrapping_add(1);
            // Advance past any subsequently received out-of-order packets.
            while self.received.contains(&self.expected_seq) {
                self.expected_seq = self.expected_seq.wrapping_add(1);
            }
            Ok(Some(payload))
        } else {
            // Gap: issue NACKs for missing sequences between expected and seq.
            let mut gap_seq = self.expected_seq;
            while gap_seq != seq {
                if !self.received.contains(&gap_seq)
                    && !self.missing.iter().any(|(s, _)| *s == gap_seq)
                {
                    self.missing.push_back((gap_seq, Instant::now()));
                }
                gap_seq = gap_seq.wrapping_add(1);
            }
            Ok(None)
        }
    }

    /// Returns [`RistNackList`] messages for all currently missing sequences.
    ///
    /// Entries that have exceeded the buffer window are pruned before
    /// generating the list.
    pub fn generate_nacks(&mut self) -> Vec<RistNackList> {
        let timeout = Duration::from_millis(u64::from(self.config.buffer_size_ms));
        // Prune timed-out missing entries.
        self.missing.retain(|(_, t)| t.elapsed() < timeout);

        if self.missing.is_empty() {
            return Vec::new();
        }
        let seq_numbers: Vec<u16> = self.missing.iter().map(|(s, _)| *s).collect();
        vec![RistNackList::new(self.ssrc, seq_numbers)]
    }

    /// Returns the number of currently tracked missing sequences.
    #[must_use]
    pub fn missing_count(&self) -> usize {
        self.missing.len()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> RistConfig {
        RistConfig::default()
    }

    // 1. Profile names
    #[test]
    fn test_rist_profile_names() {
        assert_eq!(RistProfile::Simple.name(), "RIST Simple Profile");
        assert_eq!(RistProfile::Main.name(), "RIST Main Profile");
        assert_eq!(RistProfile::Advanced.name(), "RIST Advanced Profile");
    }

    // 2. Config defaults
    #[test]
    fn test_rist_config_default() {
        let cfg = default_config();
        assert_eq!(cfg.profile, RistProfile::Simple);
        assert!(cfg.null_packet_deletion);
        assert_eq!(cfg.max_retransmissions, 5);
    }

    // 3. Config builder
    #[test]
    fn test_rist_config_builder() {
        let cfg = RistConfig::new()
            .with_buffer(Duration::from_millis(200))
            .without_npd();
        assert!(!cfg.null_packet_deletion);
        assert_eq!(cfg.buffer_duration, Duration::from_millis(200));
    }

    // 4. RistPacket encode / decode round-trip
    #[test]
    fn test_rist_packet_encode_decode() {
        let payload = vec![0x47, 0x1F, 0xFF, 0x10]; // one TS null packet
        let pkt = RistPacket::new(42, 12345, 0xDEAD_BEEF, payload.clone());
        let encoded = pkt.encode();
        let decoded = RistPacket::decode(&encoded).expect("decode should succeed");

        assert_eq!(decoded.sequence, 42);
        assert_eq!(decoded.timestamp, 12345);
        assert_eq!(decoded.ssrc, 0xDEAD_BEEF);
        assert_eq!(decoded.payload, payload);
    }

    // 5. RistPacket decode invalid version
    #[test]
    fn test_rist_packet_decode_invalid_version() {
        let mut raw = vec![0u8; 16];
        raw[0] = 0x00; // V=0
        assert!(RistPacket::decode(&raw).is_none());
    }

    // 6. RistPacket decode too short
    #[test]
    fn test_rist_packet_decode_too_short() {
        let raw = vec![0x80; 8]; // only 8 bytes
        assert!(RistPacket::decode(&raw).is_none());
    }

    // 7. RistNack single encode
    #[test]
    fn test_rist_nack_single_encode() {
        let nack = RistNack::single(0xABCD_1234, 100);
        let encoded = nack.encode();
        assert_eq!(encoded.len(), 16);
        // Check FMT=1 (NACK)
        assert_eq!(encoded[0], 0x81);
        // PT=205
        assert_eq!(encoded[1], 205);
    }

    // 8. RistNack range
    #[test]
    fn test_rist_nack_range() {
        let nack = RistNack::range(0, 10, 5);
        assert_eq!(nack.seq_start, 10);
        assert_eq!(nack.seq_count, 5);
    }

    // 9. Null-packet deletion removes null packets
    #[test]
    fn test_null_packet_deletion() {
        // One real TS packet (PID = 0) + one null (PID = 0x1FFF)
        let mut stream = vec![0u8; 376]; // 2 × 188
        stream[0] = 0x47;
        stream[1] = 0x00; // PID MSB
        stream[2] = 0x00; // PID LSB → PID = 0 (PAT)
                          // Null packet starts at offset 188
        stream[188] = 0x47;
        stream[189] = 0x1F;
        stream[190] = 0xFF;

        let filtered = NullPacketFilter::delete(&stream);
        assert_eq!(filtered.len(), 188); // Only the PAT survives
    }

    // 10. Null-packet restoration pads correctly
    #[test]
    fn test_null_packet_restore() {
        let ts = vec![0x47u8; 188]; // one real packet
        let restored = NullPacketFilter::restore(&ts, 376);
        assert_eq!(restored.len(), 376);
        // Second packet should be a null
        assert_eq!(restored[188], 0x47);
        assert_eq!(restored[189], 0x1F);
        assert_eq!(restored[190], 0xFF);
    }

    // 11. RistSender send / buffer
    #[test]
    fn test_rist_sender_send_packet() {
        let mut sender = RistSender::new(default_config(), 1);
        let encoded = sender.send_packet(vec![0x47; 188]);
        assert!(!encoded.is_empty());
        assert_eq!(sender.buffer_size(), 1);
        assert_eq!(sender.stats().packets_received, 1);
    }

    // 12. RistSender sequence wrapping
    #[test]
    fn test_rist_sender_sequence_wraps() {
        let mut sender = RistSender::new(default_config(), 1);
        sender.next_seq = u16::MAX;
        sender.send_packet(vec![1]);
        sender.send_packet(vec![2]);
        assert_eq!(sender.next_seq, 1); // wrapped through 0
    }

    // 13. RistSender handle NACK retransmits buffered packet
    #[test]
    fn test_rist_sender_handle_nack() {
        let mut sender = RistSender::new(default_config(), 5);
        sender.send_packet(vec![0u8; 10]);
        let nack = RistNack::single(5, 0);
        let retx = sender.handle_nack(&nack);
        assert_eq!(retx.len(), 1);
    }

    // 14. RistSender NACK for missing seq returns empty
    #[test]
    fn test_rist_sender_nack_missing() {
        let mut sender = RistSender::new(default_config(), 5);
        let nack = RistNack::single(5, 99); // seq 99 never sent
        let retx = sender.handle_nack(&nack);
        assert!(retx.is_empty());
    }

    // 15. RistReorderBuffer insert and drain
    #[test]
    fn test_reorder_buffer_insert_drain() {
        let cfg = default_config();
        let mut buf = RistReorderBuffer::new(&cfg, 1);

        // Insert seq 0
        let pkt0 = RistPacket::new(0, 0, 1, vec![0]);
        buf.insert(pkt0);
        let ready = buf.drain_ready();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].sequence, 0);
    }

    // 16. RistReorderBuffer gap detection generates NACK
    #[test]
    fn test_reorder_buffer_gap_nack() {
        let cfg = RistConfig::new().with_buffer(Duration::from_secs(10)); // long timeout so nothing expires
        let mut buf = RistReorderBuffer::new(&cfg, 1);

        // Deliver seq 0 first to set baseline
        buf.insert(RistPacket::new(0, 0, 1, vec![0]));
        let _ = buf.drain_ready();

        // Now insert seq 2 (gap at seq 1)
        buf.insert(RistPacket::new(2, 0, 1, vec![2]));
        let _ = buf.drain_ready(); // seq 1 still missing so won't drain seq 2

        // NACK for seq 1 should be pending
        assert!(buf.pending_nack_count() > 0);
    }

    // 17. RistReceiver receive_datagram
    #[test]
    fn test_rist_receiver_receive_datagram() {
        let mut sender = RistSender::new(default_config(), 7);
        let mut receiver = RistReceiver::new(default_config(), 7);

        let encoded = sender.send_packet(vec![0x47; 188]);
        let payloads = receiver.receive_datagram(&encoded);
        assert_eq!(payloads.len(), 1);
        assert_eq!(receiver.stats().packets_received, 1);
    }

    // 18. RistFlowStats loss_ratio update
    #[test]
    fn test_rist_flow_stats_loss_ratio() {
        let mut stats = RistFlowStats {
            packets_received: 90,
            packets_lost: 10,
            ..Default::default()
        };
        stats.update_loss_ratio();
        assert!((stats.loss_ratio - 0.1).abs() < 1e-9);
    }

    // 19. RistFlowStats zero division guard
    #[test]
    fn test_rist_flow_stats_zero_total() {
        let mut stats = RistFlowStats::default();
        stats.update_loss_ratio();
        assert_eq!(stats.loss_ratio, 0.0);
    }

    // 20. RistSender evict old packets
    #[test]
    fn test_rist_sender_evict_old_packets() {
        let cfg = RistConfig::new().with_buffer(Duration::from_nanos(1));
        let mut sender = RistSender::new(cfg, 1);
        sender.send_packet(vec![0u8; 188]);
        // Sleep not needed — with 1 ns buffer anything sent before the call is already old
        std::thread::sleep(Duration::from_millis(1));
        sender.evict_old_packets();
        assert_eq!(sender.buffer_size(), 0);
    }

    // ── Task-API tests ────────────────────────────────────────────────────────

    // 21. parse_rtp_header round-trip with build_rtp_header
    #[test]
    fn test_parse_build_rtp_header_roundtrip() {
        let hdr = RistPacketHeader {
            rtp_seq: 0x1234,
            ssrc: 0xDEAD_BEEF,
            timestamp: 0x0011_2233,
            payload_type: 33,
            marker: true,
        };
        let raw = build_rtp_header(&hdr);
        let parsed = parse_rtp_header(&raw).expect("should parse");
        assert_eq!(parsed.rtp_seq, hdr.rtp_seq);
        assert_eq!(parsed.ssrc, hdr.ssrc);
        assert_eq!(parsed.timestamp, hdr.timestamp);
        assert_eq!(parsed.payload_type, hdr.payload_type);
        assert_eq!(parsed.marker, hdr.marker);
    }

    // 22. parse_rtp_header rejects too-short buffers
    #[test]
    fn test_parse_rtp_header_too_short() {
        let err = parse_rtp_header(&[0x80u8; 8]);
        assert!(err.is_err());
    }

    // 23. parse_rtp_header rejects wrong RTP version
    #[test]
    fn test_parse_rtp_header_bad_version() {
        let mut raw = [0u8; 12];
        raw[0] = 0x00; // V=0
        let err = parse_rtp_header(&raw);
        assert!(err.is_err());
    }

    // 24. build_rtp_header sets version=2
    #[test]
    fn test_build_rtp_header_version() {
        let hdr = RistPacketHeader {
            rtp_seq: 1,
            ssrc: 0,
            timestamp: 0,
            payload_type: 96,
            marker: false,
        };
        let raw = build_rtp_header(&hdr);
        assert_eq!((raw[0] >> 6) & 0x03, 2);
    }

    // 25. build_rtp_header encodes marker bit
    #[test]
    fn test_build_rtp_header_marker() {
        let hdr = RistPacketHeader {
            rtp_seq: 0,
            ssrc: 0,
            timestamp: 0,
            payload_type: 96,
            marker: true,
        };
        let raw = build_rtp_header(&hdr);
        assert_eq!(raw[1] & 0x80, 0x80);
    }

    // 26. RistSimpleConfig defaults
    #[test]
    fn test_rist_simple_config_defaults() {
        let cfg = RistSimpleConfig::default();
        assert_eq!(cfg.buffer_size_ms, 500);
        assert_eq!(cfg.max_retransmits, 5);
        assert!((cfg.overhead_bandwidth - 0.05).abs() < 1e-6);
    }

    // 27. RistSimpleConfig converts to full config
    #[test]
    fn test_rist_simple_config_into_full() {
        let simple = RistSimpleConfig {
            buffer_size_ms: 200,
            max_retransmits: 3,
            ..Default::default()
        };
        let full = simple.into_full_config();
        assert_eq!(full.buffer_duration, Duration::from_millis(200));
        assert_eq!(full.max_retransmissions, 3);
    }

    // 28. RistNackList construction
    #[test]
    fn test_rist_nack_list_new() {
        let nack = RistNackList::new(42, vec![10, 11, 12]);
        assert_eq!(nack.ssrc, 42);
        assert_eq!(nack.seq_numbers, vec![10, 11, 12]);
    }

    // 29. RistTaskSender packetize and handle_nack
    #[test]
    fn test_rist_task_sender_packetize_and_handle_nack() {
        let cfg = RistSimpleConfig::default();
        let mut sender = RistTaskSender::new(cfg, 0xABCD);
        let pkt = sender.packetize(&[0x47u8; 188], 90_000);
        assert!(!pkt.is_empty());
        assert_eq!(sender.buffer_size(), 1);

        // Retransmit seq 0 via explicit nack list.
        let nack = RistNackList::new(0xABCD, vec![0]);
        let retx = sender.handle_nack(&nack);
        assert_eq!(retx.len(), 1);
    }

    // 30. RistTaskSender handle_nack for unknown seq returns empty
    #[test]
    fn test_rist_task_sender_handle_nack_unknown() {
        let cfg = RistSimpleConfig::default();
        let sender = RistTaskSender::new(cfg, 1);
        let nack = RistNackList::new(1, vec![99]);
        let retx = sender.handle_nack(&nack);
        assert!(retx.is_empty());
    }

    // 31. RistTaskSender prune_buffer removes old packets
    #[test]
    fn test_rist_task_sender_prune_buffer() {
        // Use 90 kHz clock.  After sending at ts=0 and advancing to ts=9001,
        // the age is 9001 ticks ≈ 100 ms.  Prune at max_age_ms=50.
        let cfg = RistSimpleConfig {
            buffer_size_ms: 500,
            ..Default::default()
        };
        let mut sender = RistTaskSender::new(cfg, 1);
        sender.packetize(&[0u8; 4], 0); // seq 0, ts 0
                                        // Advance internal timestamp to 9001 ticks (≈100 ms @ 90 kHz).
        sender.inner.timestamp = 9001;
        sender.prune_buffer(50, 90_000);
        assert_eq!(sender.buffer_size(), 0);
    }

    // 32. RistTaskSender prune_buffer keeps recent packets
    #[test]
    fn test_rist_task_sender_prune_buffer_keeps_recent() {
        let cfg = RistSimpleConfig::default();
        let mut sender = RistTaskSender::new(cfg, 1);
        sender.packetize(&[0u8; 4], 90_000); // ts = 90_000
                                             // Current ts is still 90_000 — age = 0 ms.
        sender.prune_buffer(500, 90_000);
        assert_eq!(sender.buffer_size(), 1);
    }

    // 33. RistTaskReceiver in-order delivery
    #[test]
    fn test_rist_task_receiver_in_order() {
        let mut tx = RistTaskSender::new(RistSimpleConfig::default(), 7);
        let mut rx = RistTaskReceiver::new(RistSimpleConfig::default(), 7);
        let pkt0 = tx.packetize(&[0xAAu8; 4], 0);
        let result = rx.receive_packet(&pkt0).expect("no error");
        assert!(result.is_some());
    }

    // 34. RistTaskReceiver duplicate returns None
    #[test]
    fn test_rist_task_receiver_duplicate() {
        let mut tx = RistTaskSender::new(RistSimpleConfig::default(), 7);
        let mut rx = RistTaskReceiver::new(RistSimpleConfig::default(), 7);
        let pkt0 = tx.packetize(&[0xBBu8; 4], 0);
        let _ = rx.receive_packet(&pkt0).expect("first ok");
        let result = rx.receive_packet(&pkt0).expect("duplicate ok");
        assert!(result.is_none());
    }

    // 35. RistTaskReceiver generates NACK for gap
    #[test]
    fn test_rist_task_receiver_generate_nacks_gap() {
        // Manually craft two RTP packets with seq 0 and seq 2 (gap at 1).
        let hdr0 = RistPacketHeader {
            rtp_seq: 0,
            ssrc: 1,
            timestamp: 0,
            payload_type: 33,
            marker: false,
        };
        let hdr2 = RistPacketHeader {
            rtp_seq: 2,
            ssrc: 1,
            timestamp: 90,
            payload_type: 33,
            marker: false,
        };
        let make_pkt = |hdr: &RistPacketHeader| {
            let header_bytes = build_rtp_header(hdr);
            let mut v: Vec<u8> = header_bytes.to_vec();
            v.extend_from_slice(&[0xFFu8; 4]);
            v
        };
        let mut rx = RistTaskReceiver::new(RistSimpleConfig::default(), 1);
        let _ = rx.receive_packet(&make_pkt(&hdr0)).expect("seq 0 ok");
        let _ = rx.receive_packet(&make_pkt(&hdr2)).expect("seq 2 ok");
        let nacks = rx.generate_nacks();
        // Should request seq 1.
        assert!(!nacks.is_empty());
        assert!(nacks[0].seq_numbers.contains(&1));
    }

    // 36. RistTaskReceiver missing_count tracks gaps
    #[test]
    fn test_rist_task_receiver_missing_count() {
        let hdr0 = RistPacketHeader {
            rtp_seq: 0,
            ssrc: 5,
            timestamp: 0,
            payload_type: 33,
            marker: false,
        };
        let hdr3 = RistPacketHeader {
            rtp_seq: 3,
            ssrc: 5,
            timestamp: 270,
            payload_type: 33,
            marker: false,
        };
        let make_pkt = |hdr: &RistPacketHeader| {
            let mut v: Vec<u8> = build_rtp_header(hdr).to_vec();
            v.extend_from_slice(&[0u8; 4]);
            v
        };
        let mut rx = RistTaskReceiver::new(RistSimpleConfig::default(), 5);
        let _ = rx.receive_packet(&make_pkt(&hdr0));
        let _ = rx.receive_packet(&make_pkt(&hdr3));
        // Gap: seqs 1 and 2 are missing.
        assert_eq!(rx.missing_count(), 2);
    }
}
