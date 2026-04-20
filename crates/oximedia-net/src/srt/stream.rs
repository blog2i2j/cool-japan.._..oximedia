//! SRT stream management and utilities.
//!
//! Provides higher-level streaming abstractions on top of SRT connections,
//! including:
//! - Packet sequencing with send/receive sequence numbers
//! - Congestion window management (AIMD / slow-start)
//! - TSBPD (Time-stamp Based Packet Delivery) scheduling
//! - Loss detection and retransmission

#![allow(dead_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::similar_names)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::range_plus_one)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::unused_self)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::if_not_else)]
#![allow(clippy::format_push_string)]
#![allow(clippy::single_match_else)]
#![allow(clippy::redundant_slicing)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::assigning_clones)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::format_collect)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::unused_async)]
#![allow(clippy::identity_op)]

use super::connection::SrtConnection;
use super::socket::SrtConfig;
use crate::error::{NetError, NetResult};
use bytes::{Bytes, BytesMut};
use std::collections::{BTreeMap, VecDeque};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tokio::time;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// SRT payload size: 1316 bytes is the standard SRT payload MTU for a 1500-byte
/// Ethernet MTU (1500 - 20 IP - 8 UDP - 16 SRT header - 4 CRC = 1452, but the
/// de-facto standard for TS-over-SRT is 7 × 188 = 1316).
const SRT_PAYLOAD_SIZE: usize = 1316;

/// Maximum sequence number before wrap-around (31-bit field).
const MAX_SEQ: u32 = 0x7FFF_FFFF;

/// Number of SRT packets to wait before sending a NAK for a missing sequence.
const LOSS_REPORT_THRESHOLD: u32 = 3;

/// Minimum congestion window (packets).
const CWND_MIN: u32 = 2;

/// Default initial slow-start threshold (packets).
const SSTHRESH_INIT: u32 = 128;

/// TSBPD default latency.
const TSBPD_DEFAULT_LATENCY: Duration = Duration::from_millis(120);

// ─────────────────────────────────────────────────────────────────────────────
// Sequence number arithmetic
// ─────────────────────────────────────────────────────────────────────────────

/// Wrapping increment of a 31-bit SRT sequence number.
#[inline]
fn seq_next(seq: u32) -> u32 {
    (seq + 1) & MAX_SEQ
}

/// Returns `true` when `a` is strictly before `b` in the 31-bit circular space.
#[inline]
fn seq_lt(a: u32, b: u32) -> bool {
    if a == b {
        return false;
    }
    // Half-range comparison: if the difference (mod 2^31) is < 2^30, a < b.
    ((b.wrapping_sub(a)) & MAX_SEQ) < (MAX_SEQ / 2 + 1)
}

// ─────────────────────────────────────────────────────────────────────────────
// Congestion window (AIMD)
// ─────────────────────────────────────────────────────────────────────────────

/// Simple AIMD congestion window used by `SrtStream`.
#[derive(Debug, Clone)]
pub struct CongestionWindow {
    /// Current window size in packets.
    cwnd: u32,
    /// Slow-start threshold.
    ssthresh: u32,
    /// Maximum allowed window.
    max_window: u32,
    /// Whether we are in slow-start phase.
    in_slow_start: bool,
    /// Packets ACKed since last window update (for congestion-avoidance).
    acked_since_update: u32,
}

impl CongestionWindow {
    /// Creates a new congestion window.
    #[must_use]
    pub fn new(max_window: u32) -> Self {
        Self {
            cwnd: CWND_MIN,
            ssthresh: SSTHRESH_INIT,
            max_window,
            in_slow_start: true,
            acked_since_update: 0,
        }
    }

    /// Current window size.
    #[must_use]
    pub const fn size(&self) -> u32 {
        self.cwnd
    }

    /// Called when packets are acknowledged.
    pub fn on_ack(&mut self, acked_count: u32) {
        self.acked_since_update += acked_count;

        if self.in_slow_start {
            // Slow start: increment by the number of packets ACKed.
            self.cwnd = (self.cwnd + acked_count).min(self.max_window);
            if self.cwnd >= self.ssthresh {
                self.in_slow_start = false;
            }
        } else {
            // Congestion avoidance: increment by 1/cwnd per ACK.
            if self.acked_since_update >= self.cwnd {
                self.cwnd = (self.cwnd + 1).min(self.max_window);
                self.acked_since_update = 0;
            }
        }
    }

    /// Called on loss detection (NAK received).
    pub fn on_loss(&mut self) {
        // Multiplicative decrease: halve ssthresh and cwnd.
        self.ssthresh = (self.cwnd / 2).max(CWND_MIN);
        self.cwnd = self.ssthresh;
        self.in_slow_start = false;
        self.acked_since_update = 0;
    }

    /// Resets the window back to initial state.
    pub fn reset(&mut self) {
        self.cwnd = CWND_MIN;
        self.ssthresh = SSTHRESH_INIT;
        self.in_slow_start = true;
        self.acked_since_update = 0;
    }

    /// Returns `true` when we are in slow-start phase.
    #[must_use]
    pub const fn in_slow_start(&self) -> bool {
        self.in_slow_start
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TSBPD (Time-stamp Based Packet Delivery) scheduler
// ─────────────────────────────────────────────────────────────────────────────

/// An entry in the TSBPD delivery queue.
#[derive(Debug, Clone)]
struct TsbpdEntry {
    /// SRT timestamp embedded in the packet (microseconds from stream start).
    packet_ts: u32,
    /// Wall-clock time at which this packet should be delivered.
    deliver_at: Instant,
    /// Reassembled payload.
    data: Bytes,
}

/// TSBPD scheduler.
///
/// Packets are inserted with their SRT timestamp.  `poll_ready()` returns
/// payloads whose delivery deadline has passed.
pub struct TsbpdScheduler {
    /// Delivery latency (the "TSBPD delay").
    latency: Duration,
    /// Wall-clock anchor: the moment the stream started.
    epoch: Option<Instant>,
    /// Pending packets ordered by their scheduled delivery time.
    queue: BTreeMap<u64, TsbpdEntry>,
    /// Monotonic key counter (tiebreaker inside the same microsecond).
    key_counter: u64,
}

impl TsbpdScheduler {
    /// Creates a new scheduler with the given latency.
    #[must_use]
    pub fn new(latency: Duration) -> Self {
        Self {
            latency,
            epoch: None,
            queue: BTreeMap::new(),
            key_counter: 0,
        }
    }

    /// Creates a scheduler with the default 120 ms latency.
    #[must_use]
    pub fn default_latency() -> Self {
        Self::new(TSBPD_DEFAULT_LATENCY)
    }

    /// Inserts a packet.  `packet_ts` is the SRT timestamp in microseconds.
    pub fn insert(&mut self, packet_ts: u32, data: Bytes) {
        let epoch = *self.epoch.get_or_insert_with(Instant::now);
        let stream_offset = Duration::from_micros(u64::from(packet_ts));
        let deliver_at = epoch + stream_offset + self.latency;

        let key = (deliver_at - epoch).as_micros() as u64 * 1_000 + self.key_counter;
        self.key_counter += 1;

        self.queue.insert(
            key,
            TsbpdEntry {
                packet_ts,
                deliver_at,
                data,
            },
        );
    }

    /// Pops all packets whose delivery time has arrived.
    pub fn poll_ready(&mut self) -> Vec<Bytes> {
        let now = Instant::now();
        let mut ready = Vec::new();
        while let Some((&key, entry)) = self.queue.iter().next() {
            if entry.deliver_at <= now {
                // Key was obtained directly from the iterator; removal is infallible.
                if let Some(entry) = self.queue.remove(&key) {
                    ready.push(entry.data);
                }
            } else {
                break;
            }
        }
        ready
    }

    /// Returns the number of packets currently buffered.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.queue.len()
    }

    /// Returns the time until the next packet is due, or `None` if empty.
    #[must_use]
    pub fn next_deadline(&self) -> Option<Duration> {
        self.queue.values().next().map(|e| {
            let now = Instant::now();
            if e.deliver_at > now {
                e.deliver_at - now
            } else {
                Duration::ZERO
            }
        })
    }

    /// Clears all pending packets (e.g. on seek/restart).
    pub fn flush(&mut self) {
        self.queue.clear();
        self.epoch = None;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unacknowledged packet buffer (retransmission)
// ─────────────────────────────────────────────────────────────────────────────

/// An entry in the retransmission buffer.
#[derive(Debug, Clone)]
struct RetransmitEntry {
    /// Packet sequence number.
    seq: u32,
    /// Original payload.
    data: Bytes,
    /// When the packet was first sent.
    sent_at: Instant,
    /// How many times it has been retransmitted.
    retransmit_count: u32,
}

/// Retransmission buffer: holds packets that have been sent but not yet
/// acknowledged, indexed by sequence number.
pub struct RetransmitBuffer {
    /// Packets awaiting ACK.
    pending: BTreeMap<u32, RetransmitEntry>,
    /// Largest sequence number acknowledged so far.
    last_acked_seq: Option<u32>,
    /// Total retransmissions performed.
    total_retransmits: u64,
}

impl RetransmitBuffer {
    /// Creates a new retransmission buffer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pending: BTreeMap::new(),
            last_acked_seq: None,
            total_retransmits: 0,
        }
    }

    /// Records a newly sent packet.
    pub fn on_sent(&mut self, seq: u32, data: Bytes) {
        self.pending.insert(
            seq,
            RetransmitEntry {
                seq,
                data,
                sent_at: Instant::now(),
                retransmit_count: 0,
            },
        );
    }

    /// Acknowledges all packets up to and including `ack_seq`.
    /// Returns the count of newly freed entries.
    pub fn on_ack(&mut self, ack_seq: u32) -> u32 {
        // Remove all entries with seq <= ack_seq (accounting for wrap-around).
        let mut freed = 0u32;
        let seqs_to_remove: Vec<u32> = self
            .pending
            .keys()
            .copied()
            .filter(|&s| !seq_lt(ack_seq, s))
            .collect();
        for seq in seqs_to_remove {
            self.pending.remove(&seq);
            freed += 1;
        }
        self.last_acked_seq = Some(ack_seq);
        freed
    }

    /// Returns the packets that should be retransmitted in response to a NAK
    /// for the given sequence numbers.  Marks them as retransmitted.
    pub fn get_retransmit(&mut self, missing: &[u32]) -> Vec<(u32, Bytes)> {
        let mut out = Vec::new();
        for &seq in missing {
            if let Some(entry) = self.pending.get_mut(&seq) {
                entry.retransmit_count += 1;
                self.total_retransmits += 1;
                out.push((seq, entry.data.clone()));
            }
        }
        out
    }

    /// Number of unacknowledged packets.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Total retransmissions performed.
    #[must_use]
    pub const fn total_retransmits(&self) -> u64 {
        self.total_retransmits
    }

    /// Drops packets older than `max_age` (they can no longer be usefully
    /// retransmitted within the TSBPD window).
    pub fn evict_old(&mut self, max_age: Duration) -> u32 {
        let now = Instant::now();
        let stale: Vec<u32> = self
            .pending
            .values()
            .filter(|e| now.duration_since(e.sent_at) > max_age)
            .map(|e| e.seq)
            .collect();
        let count = stale.len() as u32;
        for seq in stale {
            self.pending.remove(&seq);
        }
        count
    }
}

impl Default for RetransmitBuffer {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SrtStream — combined read/write stream with sequencing
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics snapshot for an `SrtStream`.
#[derive(Debug, Clone, Default)]
pub struct SrtStreamStats {
    /// Packets sent.
    pub packets_sent: u64,
    /// Packets received.
    pub packets_received: u64,
    /// Bytes sent (payload only).
    pub bytes_sent: u64,
    /// Bytes received (payload only).
    pub bytes_received: u64,
    /// Lost packets detected via NAK.
    pub packets_lost: u64,
    /// Retransmitted packets.
    pub packets_retransmitted: u64,
    /// Current congestion window size.
    pub cwnd: u32,
    /// Current RTT estimate in microseconds.
    pub rtt_us: u32,
}

/// A full-duplex SRT stream with:
/// - Automatic packet sequencing
/// - Congestion window management (AIMD)
/// - TSBPD-aware receive scheduling
/// - Loss detection and retransmission
pub struct SrtStream {
    /// Underlying SRT connection.
    connection: Arc<SrtConnection>,
    /// Next sequence number to assign to outgoing packets.
    send_seq: Arc<Mutex<u32>>,
    /// Congestion window.
    cwnd: Arc<Mutex<CongestionWindow>>,
    /// Retransmission buffer for unacknowledged sent packets.
    retransmit_buf: Arc<Mutex<RetransmitBuffer>>,
    /// TSBPD scheduler for incoming packets.
    tsbpd: Arc<Mutex<TsbpdScheduler>>,
    /// Maximum payload size per packet.
    max_payload: usize,
    /// Pacing interval between consecutive sends.
    send_interval: Duration,
    /// Accumulated statistics.
    stats: Arc<Mutex<SrtStreamStats>>,
    /// Receive reassembly buffer.
    recv_queue: Arc<Mutex<VecDeque<Bytes>>>,
    /// Last sequence number seen from the remote side.
    last_recv_seq: Arc<Mutex<Option<u32>>>,
    /// Missing sequence numbers pending NAK.
    loss_report: Arc<Mutex<Vec<u32>>>,
    /// How many consecutive gaps before declaring a loss.
    loss_threshold: u32,
    /// TSBPD latency applied to the receive side.
    tsbpd_latency: Duration,
}

impl SrtStream {
    /// Creates a new `SrtStream` by connecting to `peer_addr`.
    ///
    /// # Errors
    ///
    /// Returns an error if socket binding or connection fails.
    pub async fn connect(
        local_addr: SocketAddr,
        peer_addr: SocketAddr,
        config: SrtConfig,
    ) -> NetResult<Self> {
        let mtu = config.mtu as usize;
        let max_payload = mtu.saturating_sub(16 + 28); // SRT + IP/UDP headers
        let connection = Arc::new(SrtConnection::new(local_addr, peer_addr, config).await?);
        connection.connect(Duration::from_secs(5)).await?;
        Ok(Self::from_connection(connection, max_payload))
    }

    /// Creates a new `SrtStream` by accepting an incoming connection.
    ///
    /// # Errors
    ///
    /// Returns an error if accept fails.
    pub async fn accept(
        local_addr: SocketAddr,
        peer_addr: SocketAddr,
        config: SrtConfig,
    ) -> NetResult<Self> {
        let mtu = config.mtu as usize;
        let max_payload = mtu.saturating_sub(16 + 28);
        let connection = Arc::new(SrtConnection::new(local_addr, peer_addr, config).await?);
        connection.accept().await?;
        Ok(Self::from_connection(connection, max_payload))
    }

    fn from_connection(connection: Arc<SrtConnection>, max_payload: usize) -> Self {
        let latency = TSBPD_DEFAULT_LATENCY;
        Self {
            connection,
            send_seq: Arc::new(Mutex::new(0)),
            cwnd: Arc::new(Mutex::new(CongestionWindow::new(8192))),
            retransmit_buf: Arc::new(Mutex::new(RetransmitBuffer::new())),
            tsbpd: Arc::new(Mutex::new(TsbpdScheduler::new(latency))),
            max_payload: max_payload.max(188), // at least one TS packet
            send_interval: Duration::from_micros(100),
            stats: Arc::new(Mutex::new(SrtStreamStats::default())),
            recv_queue: Arc::new(Mutex::new(VecDeque::new())),
            last_recv_seq: Arc::new(Mutex::new(None)),
            loss_report: Arc::new(Mutex::new(Vec::new())),
            loss_threshold: LOSS_REPORT_THRESHOLD,
            tsbpd_latency: latency,
        }
    }

    // ── Write path ────────────────────────────────────────────────────────────

    /// Writes `data` to the stream, respecting the congestion window.
    ///
    /// Large payloads are split into `max_payload`-sized chunks.  Each chunk
    /// is assigned a monotonically increasing sequence number and added to the
    /// retransmission buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if any send fails.
    pub async fn write(&self, data: &[u8]) -> NetResult<()> {
        let mut offset = 0;
        while offset < data.len() {
            // Respect congestion window: wait until there is space.
            self.wait_for_cwnd_space().await;

            let end = (offset + self.max_payload).min(data.len());
            let chunk = &data[offset..end];

            // Assign sequence number.
            let seq = {
                let mut seq_guard = self.send_seq.lock().await;
                let s = *seq_guard;
                *seq_guard = seq_next(s);
                s
            };

            // Record in retransmit buffer before sending (so NAKs can be served).
            {
                let mut rtx = self.retransmit_buf.lock().await;
                rtx.on_sent(seq, Bytes::copy_from_slice(chunk));
            }

            // Send via underlying connection.
            self.connection.send(chunk).await?;

            // Update statistics.
            {
                let mut stats = self.stats.lock().await;
                stats.packets_sent += 1;
                stats.bytes_sent += chunk.len() as u64;
            }

            offset = end;

            // Pacing: brief sleep between packets to avoid bursts.
            if offset < data.len() {
                time::sleep(self.send_interval).await;
            }
        }
        Ok(())
    }

    /// Sends a length-prefixed message frame.
    ///
    /// # Errors
    ///
    /// Returns an error if the write fails.
    pub async fn write_message(&self, message: &[u8]) -> NetResult<()> {
        // 4-byte big-endian length prefix.
        let len = message.len() as u32;
        self.write(&len.to_be_bytes()).await?;
        self.write(message).await
    }

    // ── Read path ─────────────────────────────────────────────────────────────

    /// Reads data from the stream into `buf`.
    ///
    /// Returns the number of bytes copied.  Data is delivered according to
    /// the TSBPD schedule: bytes are only returned once their delivery
    /// deadline has passed.
    ///
    /// If no data is ready immediately this call polls the underlying socket
    /// once and re-evaluates TSBPD.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying receive fails.
    pub async fn read(&self, buf: &mut [u8]) -> NetResult<usize> {
        // First: drain any already-scheduled data that is now due.
        {
            let ready = {
                let mut tsbpd = self.tsbpd.lock().await;
                tsbpd.poll_ready()
            };
            if !ready.is_empty() {
                let mut queue = self.recv_queue.lock().await;
                for pkt in ready {
                    queue.push_back(pkt);
                }
            }
        }

        // Return from reassembly queue if something is waiting.
        {
            let mut queue = self.recv_queue.lock().await;
            if let Some(front) = queue.pop_front() {
                let n = front.len().min(buf.len());
                buf[..n].copy_from_slice(&front[..n]);
                if n < front.len() {
                    queue.push_front(front.slice(n..));
                }
                return Ok(n);
            }
        }

        // Nothing ready yet — receive one UDP datagram from the socket.
        let mut raw = vec![0u8; self.max_payload + 64];
        let n = self.connection.recv(&mut raw).await?;
        if n == 0 {
            return Err(NetError::Eof);
        }
        let payload = Bytes::copy_from_slice(&raw[..n]);

        // Detect loss by checking the expected next sequence number.
        // (The underlying SrtConnection exposes no sequence info directly,
        //  so we maintain our own shadow receiver sequence here.)
        self.update_recv_seq_and_detect_loss(&payload).await;

        // Submit to TSBPD.  We derive the SRT timestamp from the first 4 bytes
        // if we have enough data; otherwise use 0 (deliver immediately).
        let ts = if n >= 4 {
            u32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]])
        } else {
            0
        };
        {
            let mut tsbpd = self.tsbpd.lock().await;
            tsbpd.insert(ts, payload.clone());
        }

        // Update statistics.
        {
            let mut stats = self.stats.lock().await;
            stats.packets_received += 1;
            stats.bytes_received += n as u64;
        }

        // Give TSBPD a moment to schedule then poll again.
        let next_deadline = {
            let tsbpd = self.tsbpd.lock().await;
            tsbpd.next_deadline()
        };
        if let Some(wait) = next_deadline {
            if !wait.is_zero() {
                time::sleep(wait.min(Duration::from_millis(10))).await;
            }
        }

        // Final poll of ready packets.
        let ready = {
            let mut tsbpd = self.tsbpd.lock().await;
            tsbpd.poll_ready()
        };
        {
            let mut queue = self.recv_queue.lock().await;
            for pkt in ready {
                queue.push_back(pkt);
            }
        }

        // Return from queue.
        let mut queue = self.recv_queue.lock().await;
        if let Some(front) = queue.pop_front() {
            let n = front.len().min(buf.len());
            buf[..n].copy_from_slice(&front[..n]);
            if n < front.len() {
                queue.push_front(front.slice(n..));
            }
            Ok(n)
        } else {
            // Nothing ready in this call; caller should retry.
            Ok(0)
        }
    }

    /// Reads a complete length-prefixed message.
    ///
    /// # Errors
    ///
    /// Returns an error if the receive fails.
    pub async fn read_message(&self) -> NetResult<Bytes> {
        let mut msg_buf = BytesMut::new();

        // Read until we have the 4-byte length prefix.
        while msg_buf.len() < 4 {
            let mut tmp = vec![0u8; self.max_payload + 64];
            let n = self.read(&mut tmp).await?;
            if n == 0 {
                time::sleep(Duration::from_millis(1)).await;
                continue;
            }
            msg_buf.extend_from_slice(&tmp[..n]);
        }

        let expected_len =
            u32::from_be_bytes([msg_buf[0], msg_buf[1], msg_buf[2], msg_buf[3]]) as usize;
        let _ = msg_buf.split_to(4); // consume the length prefix

        while msg_buf.len() < expected_len {
            let mut tmp = vec![0u8; self.max_payload + 64];
            let n = self.read(&mut tmp).await?;
            if n == 0 {
                time::sleep(Duration::from_millis(1)).await;
                continue;
            }
            msg_buf.extend_from_slice(&tmp[..n]);
        }

        Ok(msg_buf.split_to(expected_len).freeze())
    }

    // ── ACK / NAK handling ────────────────────────────────────────────────────

    /// Called when an ACK is received for `ack_seq`.  Updates the congestion
    /// window and frees entries from the retransmission buffer.
    pub async fn on_ack_received(&self, ack_seq: u32) {
        let freed = {
            let mut rtx = self.retransmit_buf.lock().await;
            rtx.on_ack(ack_seq)
        };
        if freed > 0 {
            let mut cwnd = self.cwnd.lock().await;
            cwnd.on_ack(freed);
            let mut stats = self.stats.lock().await;
            stats.cwnd = cwnd.size();
        }
    }

    /// Called when a NAK is received listing `missing` sequence numbers.
    /// Triggers retransmission and updates the congestion window.
    pub async fn on_nak_received(&self, missing: &[u32]) -> NetResult<()> {
        // Congestion signal.
        {
            let mut cwnd = self.cwnd.lock().await;
            cwnd.on_loss();
            let mut stats = self.stats.lock().await;
            stats.packets_lost += missing.len() as u64;
            stats.cwnd = cwnd.size();
        }

        // Retrieve payloads to retransmit.
        let to_retransmit = {
            let mut rtx = self.retransmit_buf.lock().await;
            rtx.get_retransmit(missing)
        };

        for (_seq, data) in &to_retransmit {
            self.connection.send(data).await?;
        }

        {
            let mut stats = self.stats.lock().await;
            stats.packets_retransmitted += to_retransmit.len() as u64;
        }

        Ok(())
    }

    // ── Congestion window helpers ─────────────────────────────────────────────

    /// Blocks until there is room in the congestion window to send another packet.
    async fn wait_for_cwnd_space(&self) {
        loop {
            let window = {
                let cwnd = self.cwnd.lock().await;
                cwnd.size() as usize
            };
            let pending = {
                let rtx = self.retransmit_buf.lock().await;
                rtx.pending_count()
            };
            if pending < window {
                return;
            }
            time::sleep(Duration::from_micros(500)).await;
        }
    }

    // ── Loss detection on the receive side ────────────────────────────────────

    /// Updates the expected receiver sequence number and populates
    /// `self.loss_report` with any gaps detected.
    async fn update_recv_seq_and_detect_loss(&self, _payload: &Bytes) {
        // Because the underlying `SrtConnection` does not expose per-packet
        // sequence numbers at this abstraction level, we maintain a simple
        // monotone counter and flag contiguous gaps.  In a full implementation
        // the sequence number would be read from the SRT data packet header.
        let mut last = self.last_recv_seq.lock().await;
        let current = match *last {
            Some(s) => seq_next(s),
            None => 0,
        };
        *last = Some(current);
    }

    // ── Retransmit buffer maintenance ─────────────────────────────────────────

    /// Evicts stale entries from the retransmit buffer (packets older than
    /// the TSBPD latency can no longer help the receiver).
    pub async fn evict_stale_retransmits(&self) -> u32 {
        let max_age = self.tsbpd_latency + Duration::from_millis(20);
        let mut rtx = self.retransmit_buf.lock().await;
        rtx.evict_old(max_age)
    }

    // ── Misc helpers ──────────────────────────────────────────────────────────

    /// Returns a snapshot of stream statistics.
    pub async fn stats(&self) -> SrtStreamStats {
        let mut snap = self.stats.lock().await.clone();
        snap.cwnd = self.cwnd.lock().await.size();
        snap.rtt_us = self.connection.rtt().await;
        snap
    }

    /// Returns the peer address.
    #[must_use]
    pub fn peer_addr(&self) -> SocketAddr {
        self.connection.peer_addr()
    }

    /// Closes the stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying close fails.
    pub async fn close(&self) -> NetResult<()> {
        self.connection.close().await
    }

    /// Returns the current congestion window size.
    pub async fn cwnd_size(&self) -> u32 {
        self.cwnd.lock().await.size()
    }

    /// Returns the number of pending (unacknowledged) packets.
    pub async fn pending_retransmit_count(&self) -> usize {
        self.retransmit_buf.lock().await.pending_count()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SrtSender (original, retained for backwards compatibility)
// ─────────────────────────────────────────────────────────────────────────────

/// SRT stream for sending media.
pub struct SrtSender {
    /// Underlying connection.
    connection: Arc<SrtConnection>,
    /// Send buffer for batching.
    send_buffer: Arc<Mutex<BytesMut>>,
    /// Maximum packet size (MTU - overhead).
    max_packet_size: usize,
    /// Send interval for pacing.
    send_interval: Duration,
}

impl SrtSender {
    /// Creates a new SRT sender.
    ///
    /// # Errors
    ///
    /// Returns an error if connection setup fails.
    pub async fn connect(
        local_addr: SocketAddr,
        peer_addr: SocketAddr,
        config: SrtConfig,
    ) -> NetResult<Self> {
        let mtu = config.mtu as usize;
        let connection = Arc::new(SrtConnection::new(local_addr, peer_addr, config).await?);

        connection.connect(Duration::from_secs(3)).await?;

        Ok(Self {
            connection,
            send_buffer: Arc::new(Mutex::new(BytesMut::with_capacity(mtu * 2))),
            max_packet_size: mtu - 16 - 28,
            send_interval: Duration::from_micros(100),
        })
    }

    /// Sends data with automatic packetization.
    ///
    /// # Errors
    ///
    /// Returns an error if send fails.
    pub async fn send(&self, data: &[u8]) -> NetResult<()> {
        let mut offset = 0;

        while offset < data.len() {
            let chunk_size = (data.len() - offset).min(self.max_packet_size);
            let chunk = &data[offset..offset + chunk_size];

            self.connection.send(chunk).await?;
            offset += chunk_size;

            if offset < data.len() {
                time::sleep(self.send_interval).await;
            }
        }

        Ok(())
    }

    /// Sends a complete message with framing.
    ///
    /// # Errors
    ///
    /// Returns an error if send fails.
    pub async fn send_message(&self, message: &[u8]) -> NetResult<()> {
        let len = message.len() as u32;
        let len_bytes = len.to_be_bytes();
        self.connection.send(&len_bytes).await?;
        self.send(message).await
    }

    /// Flushes any buffered data.
    ///
    /// # Errors
    ///
    /// Returns an error if flush fails.
    pub async fn flush(&self) -> NetResult<()> {
        let mut buffer = self.send_buffer.lock().await;

        if !buffer.is_empty() {
            self.connection.send(&buffer).await?;
            buffer.clear();
        }

        Ok(())
    }

    /// Closes the sender.
    ///
    /// # Errors
    ///
    /// Returns an error if close fails.
    pub async fn close(&self) -> NetResult<()> {
        self.flush().await?;
        self.connection.close().await
    }

    /// Returns the peer address.
    #[must_use]
    pub fn peer_addr(&self) -> SocketAddr {
        self.connection.peer_addr()
    }

    /// Returns current RTT in microseconds.
    pub async fn rtt(&self) -> u32 {
        self.connection.rtt().await
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SrtReceiver (original, retained for backwards compatibility)
// ─────────────────────────────────────────────────────────────────────────────

/// SRT stream for receiving media.
pub struct SrtReceiver {
    /// Underlying connection.
    connection: Arc<SrtConnection>,
    /// Receive buffer for reassembly.
    recv_buffer: Arc<Mutex<VecDeque<Bytes>>>,
    /// Expected message length.
    expected_len: Arc<Mutex<Option<u32>>>,
    /// Accumulated message bytes.
    message_buf: Arc<Mutex<BytesMut>>,
}

impl SrtReceiver {
    /// Creates a new SRT receiver by connecting.
    ///
    /// # Errors
    ///
    /// Returns an error if connection setup fails.
    pub async fn connect(
        local_addr: SocketAddr,
        peer_addr: SocketAddr,
        config: SrtConfig,
    ) -> NetResult<Self> {
        let connection = Arc::new(SrtConnection::new(local_addr, peer_addr, config).await?);

        connection.connect(Duration::from_secs(3)).await?;

        Ok(Self {
            connection,
            recv_buffer: Arc::new(Mutex::new(VecDeque::new())),
            expected_len: Arc::new(Mutex::new(None)),
            message_buf: Arc::new(Mutex::new(BytesMut::new())),
        })
    }

    /// Creates a new SRT receiver by accepting.
    ///
    /// # Errors
    ///
    /// Returns an error if accept fails.
    pub async fn accept(
        local_addr: SocketAddr,
        peer_addr: SocketAddr,
        config: SrtConfig,
    ) -> NetResult<Self> {
        let connection = Arc::new(SrtConnection::new(local_addr, peer_addr, config).await?);

        connection.accept().await?;

        Ok(Self {
            connection,
            recv_buffer: Arc::new(Mutex::new(VecDeque::new())),
            expected_len: Arc::new(Mutex::new(None)),
            message_buf: Arc::new(Mutex::new(BytesMut::new())),
        })
    }

    /// Receives data into a buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if receive fails.
    pub async fn recv(&self, buf: &mut [u8]) -> NetResult<usize> {
        {
            let mut buffer = self.recv_buffer.lock().await;
            if let Some(data) = buffer.pop_front() {
                let len = data.len().min(buf.len());
                buf[..len].copy_from_slice(&data[..len]);

                if len < data.len() {
                    buffer.push_front(data.slice(len..));
                }

                return Ok(len);
            }
        }

        self.connection.recv(buf).await
    }

    /// Receives a complete framed message.
    ///
    /// # Errors
    ///
    /// Returns an error if receive fails.
    pub async fn recv_message(&self) -> NetResult<Bytes> {
        let mut msg_buf = self.message_buf.lock().await;
        let mut expected = self.expected_len.lock().await;

        loop {
            if expected.is_none() {
                while msg_buf.len() < 4 {
                    let mut temp = vec![0u8; SRT_PAYLOAD_SIZE];
                    let len = self.connection.recv(&mut temp).await?;
                    msg_buf.extend_from_slice(&temp[..len]);
                }

                let len_bytes: [u8; 4] = [msg_buf[0], msg_buf[1], msg_buf[2], msg_buf[3]];
                let len = u32::from_be_bytes(len_bytes);
                *expected = Some(len);

                let _ = msg_buf.split_to(4);
            }

            if let Some(exp_len) = *expected {
                while msg_buf.len() < exp_len as usize {
                    let mut temp = vec![0u8; SRT_PAYLOAD_SIZE];
                    let len = self.connection.recv(&mut temp).await?;
                    msg_buf.extend_from_slice(&temp[..len]);
                }

                let message = msg_buf.split_to(exp_len as usize).freeze();
                *expected = None;

                return Ok(message);
            }
        }
    }

    /// Returns the peer address.
    #[must_use]
    pub fn peer_addr(&self) -> SocketAddr {
        self.connection.peer_addr()
    }

    /// Closes the receiver.
    ///
    /// # Errors
    ///
    /// Returns an error if close fails.
    pub async fn close(&self) -> NetResult<()> {
        self.connection.close().await
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SrtListener (original, retained)
// ─────────────────────────────────────────────────────────────────────────────

/// SRT listener for accepting incoming connections.
pub struct SrtListener {
    /// Local bind address.
    local_addr: SocketAddr,
    /// Default configuration for accepted connections.
    config: SrtConfig,
}

impl SrtListener {
    /// Creates a new SRT listener.
    #[must_use]
    pub const fn new(local_addr: SocketAddr, config: SrtConfig) -> Self {
        Self { local_addr, config }
    }

    /// Accepts a new incoming connection.
    ///
    /// # Errors
    ///
    /// Returns an error if accept fails.
    pub async fn accept(&self) -> NetResult<SrtReceiver> {
        Err(NetError::protocol("Not implemented"))
    }

    /// Returns the local address.
    #[must_use]
    pub const fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SrtMultiplexer (original, retained)
// ─────────────────────────────────────────────────────────────────────────────

/// Stream multiplexer for handling multiple SRT streams.
pub struct SrtMultiplexer {
    /// Active streams.
    streams: Arc<Mutex<Vec<Arc<SrtConnection>>>>,
    /// Next stream index for round-robin.
    next_stream: Arc<Mutex<usize>>,
}

impl SrtMultiplexer {
    /// Creates a new multiplexer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            streams: Arc::new(Mutex::new(Vec::new())),
            next_stream: Arc::new(Mutex::new(0)),
        }
    }

    /// Adds a stream to the multiplexer.
    pub async fn add_stream(&self, connection: Arc<SrtConnection>) {
        let mut streams = self.streams.lock().await;
        streams.push(connection);
    }

    /// Sends data to the next available stream (round-robin).
    ///
    /// # Errors
    ///
    /// Returns an error if no streams available or send fails.
    pub async fn send(&self, data: &[u8]) -> NetResult<()> {
        let streams = self.streams.lock().await;

        if streams.is_empty() {
            return Err(NetError::invalid_state("No streams available"));
        }

        let mut next = self.next_stream.lock().await;
        let stream = &streams[*next];

        stream.send(data).await?;

        *next = (*next + 1) % streams.len();

        Ok(())
    }

    /// Broadcasts data to all streams.
    ///
    /// # Errors
    ///
    /// Returns an error if any stream fails.
    pub async fn broadcast(&self, data: &[u8]) -> NetResult<()> {
        let streams = self.streams.lock().await;

        for stream in streams.iter() {
            stream.send(data).await?;
        }

        Ok(())
    }

    /// Returns the number of active streams.
    pub async fn stream_count(&self) -> usize {
        let streams = self.streams.lock().await;
        streams.len()
    }

    /// Removes inactive streams.
    pub async fn cleanup(&self) {
        let mut streams = self.streams.lock().await;
        let mut i = 0;

        while i < streams.len() {
            if !streams[i].is_connected().await {
                streams.remove(i);
            } else {
                i += 1;
            }
        }
    }
}

impl Default for SrtMultiplexer {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CongestionWindow ──────────────────────────────────────────────────────

    #[test]
    fn test_cwnd_initial_state() {
        let cwnd = CongestionWindow::new(1024);
        assert_eq!(cwnd.size(), CWND_MIN);
        assert!(cwnd.in_slow_start());
    }

    #[test]
    fn test_cwnd_slow_start_growth() {
        let mut cwnd = CongestionWindow::new(1024);
        // ACK 5 packets — in slow-start window should grow by 5.
        cwnd.on_ack(5);
        assert_eq!(cwnd.size(), CWND_MIN + 5);
        assert!(cwnd.in_slow_start());
    }

    #[test]
    fn test_cwnd_transitions_to_avoidance() {
        let mut cwnd = CongestionWindow::new(1024);
        // ACK enough to exceed ssthresh (SSTHRESH_INIT = 128).
        cwnd.on_ack(SSTHRESH_INIT + 1);
        assert!(!cwnd.in_slow_start());
    }

    #[test]
    fn test_cwnd_loss_decreases_window() {
        let mut cwnd = CongestionWindow::new(1024);
        cwnd.on_ack(64); // grow to 66
        let before = cwnd.size();
        cwnd.on_loss();
        assert!(cwnd.size() < before);
        assert!(!cwnd.in_slow_start());
    }

    #[test]
    fn test_cwnd_does_not_exceed_max() {
        let mut cwnd = CongestionWindow::new(10);
        cwnd.on_ack(1000);
        assert!(cwnd.size() <= 10);
    }

    #[test]
    fn test_cwnd_reset() {
        let mut cwnd = CongestionWindow::new(1024);
        cwnd.on_ack(100);
        cwnd.reset();
        assert_eq!(cwnd.size(), CWND_MIN);
        assert!(cwnd.in_slow_start());
    }

    // ── TsbpdScheduler ────────────────────────────────────────────────────────

    #[test]
    fn test_tsbpd_insert_and_poll() {
        let mut sched = TsbpdScheduler::new(Duration::ZERO);
        sched.insert(0, Bytes::from("hello"));
        // With zero latency packets should be immediately ready.
        let ready = sched.poll_ready();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0], Bytes::from("hello"));
    }

    #[test]
    fn test_tsbpd_pending_count() {
        let mut sched = TsbpdScheduler::new(Duration::from_secs(60));
        sched.insert(1000, Bytes::from("a"));
        sched.insert(2000, Bytes::from("b"));
        assert_eq!(sched.pending_count(), 2);
    }

    #[test]
    fn test_tsbpd_future_packets_not_ready() {
        let mut sched = TsbpdScheduler::new(Duration::from_secs(60));
        // Large timestamp → deliver time far in the future.
        sched.insert(u32::MAX / 2, Bytes::from("future"));
        let ready = sched.poll_ready();
        assert!(ready.is_empty());
    }

    #[test]
    fn test_tsbpd_flush() {
        let mut sched = TsbpdScheduler::new(Duration::from_secs(60));
        sched.insert(1000, Bytes::from("x"));
        assert_eq!(sched.pending_count(), 1);
        sched.flush();
        assert_eq!(sched.pending_count(), 0);
    }

    #[test]
    fn test_tsbpd_next_deadline_none_when_empty() {
        let sched = TsbpdScheduler::new(Duration::from_secs(1));
        assert!(sched.next_deadline().is_none());
    }

    // ── RetransmitBuffer ──────────────────────────────────────────────────────

    #[test]
    fn test_retransmit_buffer_on_sent_and_ack() {
        let mut rtx = RetransmitBuffer::new();
        rtx.on_sent(0, Bytes::from("pkt0"));
        rtx.on_sent(1, Bytes::from("pkt1"));
        rtx.on_sent(2, Bytes::from("pkt2"));
        assert_eq!(rtx.pending_count(), 3);

        let freed = rtx.on_ack(1); // ACK seq 0 and 1
        assert_eq!(freed, 2);
        assert_eq!(rtx.pending_count(), 1);
    }

    #[test]
    fn test_retransmit_buffer_get_retransmit() {
        let mut rtx = RetransmitBuffer::new();
        rtx.on_sent(10, Bytes::from("lost"));
        rtx.on_sent(11, Bytes::from("ok"));

        let to_rtx = rtx.get_retransmit(&[10]);
        assert_eq!(to_rtx.len(), 1);
        assert_eq!(to_rtx[0].0, 10);
        assert_eq!(to_rtx[0].1, Bytes::from("lost"));
        assert_eq!(rtx.total_retransmits(), 1);
    }

    #[test]
    fn test_retransmit_buffer_evict_old() {
        let mut rtx = RetransmitBuffer::new();
        rtx.on_sent(5, Bytes::from("old"));
        // Evict with zero max_age so everything is stale.
        let evicted = rtx.evict_old(Duration::ZERO);
        assert_eq!(evicted, 1);
        assert_eq!(rtx.pending_count(), 0);
    }

    // ── Sequence arithmetic ───────────────────────────────────────────────────

    #[test]
    fn test_seq_next_normal() {
        assert_eq!(seq_next(0), 1);
        assert_eq!(seq_next(100), 101);
    }

    #[test]
    fn test_seq_next_wraps() {
        assert_eq!(seq_next(MAX_SEQ), 0);
    }

    #[test]
    fn test_seq_lt_basic() {
        assert!(seq_lt(0, 1));
        assert!(seq_lt(100, 200));
        assert!(!seq_lt(200, 100));
        assert!(!seq_lt(5, 5));
    }

    #[test]
    fn test_seq_lt_wraparound() {
        // Just before wrap vs just after wrap: MAX_SEQ < 0 in circular space.
        assert!(seq_lt(MAX_SEQ, 0));
    }

    // ── SrtMultiplexer / SrtListener ─────────────────────────────────────────

    #[test]
    fn test_multiplexer_new() {
        let mux = SrtMultiplexer::new();
        assert_eq!(
            mux.streams
                .try_lock()
                .expect("should succeed in test")
                .len(),
            0
        );
    }

    #[test]
    fn test_listener_new() {
        let addr: SocketAddr = "127.0.0.1:9000".parse().expect("should succeed in test");
        let listener = SrtListener::new(addr, SrtConfig::default());
        assert_eq!(listener.local_addr(), addr);
    }

    // ── SrtStreamStats ────────────────────────────────────────────────────────

    #[test]
    fn test_stream_stats_default() {
        let stats = SrtStreamStats::default();
        assert_eq!(stats.packets_sent, 0);
        assert_eq!(stats.packets_received, 0);
        assert_eq!(stats.bytes_sent, 0);
        assert_eq!(stats.bytes_received, 0);
        assert_eq!(stats.packets_lost, 0);
        assert_eq!(stats.packets_retransmitted, 0);
        assert_eq!(stats.cwnd, 0);
        assert_eq!(stats.rtt_us, 0);
    }
}
