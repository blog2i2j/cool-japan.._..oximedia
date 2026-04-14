//! Scatter/gather I/O for high-throughput UDP media transport.
//!
//! This module provides batch send/receive operations that reduce per-packet
//! syscall overhead by coalescing multiple datagrams into fewer kernel calls.
//! On Linux, the underlying approach mirrors `sendmmsg`/`recvmmsg`; on other
//! platforms we fall back to a tight batched loop that still amortises buffer
//! management and error handling.
//!
//! # Design
//!
//! ```text
//! ┌──────────────┐          ┌──────────────┐
//! │  MediaFrame  │──split──>│ PacketBatch  │──scatter_send──>  network
//! └──────────────┘          └──────────────┘
//!
//!       network  ──gather_recv──>│ PacketBatch │──reassemble──>│ MediaFrame │
//! ```

#![allow(dead_code)]

use crate::error::{VideoIpError, VideoIpResult};
use crate::packet::{Packet, PacketHeader, MAX_PACKET_SIZE};
use bytes::{BufMut, Bytes, BytesMut};
use std::net::SocketAddr;

// ---------------------------------------------------------------------------
// Batch descriptor
// ---------------------------------------------------------------------------

/// A single element in a scatter/gather batch.
#[derive(Debug, Clone)]
pub struct IoEntry {
    /// Pre-serialised packet bytes (header + payload).
    pub data: Bytes,
    /// Remote address for send, or origin address for receive.
    pub addr: SocketAddr,
}

/// A batch of datagrams ready for scatter send or produced by gather receive.
#[derive(Debug, Clone)]
pub struct PacketBatch {
    entries: Vec<IoEntry>,
    /// Maximum entries this batch will accept.
    capacity: usize,
}

impl PacketBatch {
    /// Creates an empty batch with the given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Adds a datagram to the batch.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch is already full.
    pub fn push(&mut self, data: Bytes, addr: SocketAddr) -> VideoIpResult<()> {
        if self.entries.len() >= self.capacity {
            return Err(VideoIpError::BufferOverflow);
        }
        self.entries.push(IoEntry { data, addr });
        Ok(())
    }

    /// Adds a serialised [`Packet`] to the batch.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch is full.
    pub fn push_packet(&mut self, packet: &Packet, addr: SocketAddr) -> VideoIpResult<()> {
        let mut buf = BytesMut::with_capacity(PacketHeader::SIZE + packet.payload.len());
        packet.header.encode(&mut buf);
        buf.put_slice(&packet.payload);
        self.push(buf.freeze(), addr)
    }

    /// Returns the number of entries in the batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the batch is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the remaining capacity.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.capacity.saturating_sub(self.entries.len())
    }

    /// Iterates over the entries.
    pub fn iter(&self) -> impl Iterator<Item = &IoEntry> {
        self.entries.iter()
    }

    /// Drains the batch, returning owned entries.
    pub fn drain(&mut self) -> Vec<IoEntry> {
        std::mem::take(&mut self.entries)
    }

    /// Returns the total byte count across all entries.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.data.len()).sum()
    }

    /// Clears the batch for reuse.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// ---------------------------------------------------------------------------
// Scatter/gather engine
// ---------------------------------------------------------------------------

/// Statistics for a scatter-send or gather-receive operation.
#[derive(Debug, Clone, Default)]
pub struct BatchIoStats {
    /// Number of datagrams successfully processed.
    pub datagrams_ok: usize,
    /// Number of datagrams that failed.
    pub datagrams_failed: usize,
    /// Total bytes transferred.
    pub bytes_transferred: usize,
}

/// Configuration for the scatter/gather I/O engine.
#[derive(Debug, Clone)]
pub struct ScatterGatherConfig {
    /// Maximum batch size (number of datagrams per syscall batch).
    pub max_batch_size: usize,
    /// Maximum datagram size.
    pub max_datagram_size: usize,
    /// Whether to enable GSO (Generic Segmentation Offload) hints.
    pub enable_gso: bool,
    /// Whether to enable GRO (Generic Receive Offload) hints.
    pub enable_gro: bool,
}

impl Default for ScatterGatherConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            max_datagram_size: MAX_PACKET_SIZE,
            enable_gso: false,
            enable_gro: false,
        }
    }
}

impl ScatterGatherConfig {
    /// Creates a config tuned for HD 1080p60 streams.
    #[must_use]
    pub fn hd_1080p60() -> Self {
        Self {
            max_batch_size: 128,
            max_datagram_size: MAX_PACKET_SIZE,
            enable_gso: true,
            enable_gro: true,
        }
    }

    /// Creates a config tuned for UHD 4K streams.
    #[must_use]
    pub fn uhd_4k() -> Self {
        Self {
            max_batch_size: 256,
            max_datagram_size: MAX_PACKET_SIZE,
            enable_gso: true,
            enable_gro: true,
        }
    }

    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if configuration values are out of range.
    pub fn validate(&self) -> VideoIpResult<()> {
        if self.max_batch_size == 0 {
            return Err(VideoIpError::InvalidVideoConfig(
                "max_batch_size must be > 0".into(),
            ));
        }
        if self.max_datagram_size == 0 || self.max_datagram_size > 65535 {
            return Err(VideoIpError::InvalidVideoConfig(
                "max_datagram_size must be 1..=65535".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Scatter-send helper
// ---------------------------------------------------------------------------

/// Pre-computes a serialisation plan for a frame split across many packets.
///
/// Given a raw payload and an MTU, this returns a vector of payload chunks
/// sized so each fits within one datagram (accounting for the packet header).
pub fn split_frame_payload(payload: &[u8], max_datagram_payload: usize) -> Vec<Bytes> {
    if max_datagram_payload == 0 {
        return Vec::new();
    }
    payload
        .chunks(max_datagram_payload)
        .map(|chunk| Bytes::copy_from_slice(chunk))
        .collect()
}

/// Merges multiple payload fragments back into the original frame payload.
///
/// # Errors
///
/// Returns an error if the total reassembled size exceeds `max_frame_size`.
pub fn reassemble_frame_payload(
    fragments: &[Bytes],
    max_frame_size: usize,
) -> VideoIpResult<Bytes> {
    let total: usize = fragments.iter().map(|f| f.len()).sum();
    if total > max_frame_size {
        return Err(VideoIpError::PacketTooLarge {
            size: total,
            max: max_frame_size,
        });
    }
    let mut buf = BytesMut::with_capacity(total);
    for frag in fragments {
        buf.put_slice(frag);
    }
    Ok(buf.freeze())
}

// ---------------------------------------------------------------------------
// Batch send / recv simulators (pure-Rust, no real socket)
// ---------------------------------------------------------------------------

/// Simulates a scatter-send by serialising a batch and collecting the output.
///
/// In production this would call `sendmmsg` (Linux) or a tight
/// `send_to` loop on other platforms.  Here we return the stats and the
/// serialised buffers so calling code can verify correctness without
/// binding real sockets.
pub fn scatter_send_sim(batch: &PacketBatch) -> BatchIoStats {
    let mut stats = BatchIoStats::default();
    for entry in batch.iter() {
        if entry.data.is_empty() {
            stats.datagrams_failed += 1;
        } else {
            stats.datagrams_ok += 1;
            stats.bytes_transferred += entry.data.len();
        }
    }
    stats
}

/// Simulates a gather-receive by splitting a large buffer into
/// individual datagrams according to the given sizes.
///
/// # Errors
///
/// Returns an error if the buffer is too short for the declared sizes.
pub fn gather_recv_sim(
    raw: &[u8],
    sizes: &[usize],
    origin: SocketAddr,
    config: &ScatterGatherConfig,
) -> VideoIpResult<PacketBatch> {
    let mut batch = PacketBatch::new(config.max_batch_size);
    let mut offset = 0usize;
    for &sz in sizes {
        let end = offset.checked_add(sz).ok_or_else(|| {
            VideoIpError::InvalidPacket("size overflow in gather_recv_sim".into())
        })?;
        if end > raw.len() {
            return Err(VideoIpError::InvalidPacket(format!(
                "buffer too short: need {end}, have {}",
                raw.len()
            )));
        }
        batch.push(Bytes::copy_from_slice(&raw[offset..end]), origin)?;
        offset = end;
    }
    Ok(batch)
}

// ---------------------------------------------------------------------------
// Pacing / rate-limited batch sender
// ---------------------------------------------------------------------------

/// Rate-limited batch sender that spaces out batches to avoid micro-bursts.
#[derive(Debug)]
pub struct PacedBatchSender {
    /// Configured inter-batch gap in microseconds.
    inter_batch_gap_us: u64,
    /// Maximum datagrams per batch.
    max_batch: usize,
    /// Running count of datagrams sent.
    total_sent: u64,
    /// Running count of bytes sent.
    total_bytes: u64,
}

impl PacedBatchSender {
    /// Creates a new paced batch sender.
    ///
    /// # Arguments
    ///
    /// * `inter_batch_gap_us` - Minimum microseconds between batch sends.
    /// * `max_batch` - Maximum datagrams per batch.
    #[must_use]
    pub fn new(inter_batch_gap_us: u64, max_batch: usize) -> Self {
        Self {
            inter_batch_gap_us,
            max_batch,
            total_sent: 0,
            total_bytes: 0,
        }
    }

    /// Returns the configured inter-batch gap.
    #[must_use]
    pub fn inter_batch_gap_us(&self) -> u64 {
        self.inter_batch_gap_us
    }

    /// Returns the maximum batch size.
    #[must_use]
    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    /// Partitions a large batch into sub-batches respecting `max_batch`.
    pub fn partition(&self, batch: &PacketBatch) -> Vec<PacketBatch> {
        let entries: Vec<&IoEntry> = batch.iter().collect();
        let chunks: Vec<&[&IoEntry]> = entries.chunks(self.max_batch).collect();
        chunks
            .into_iter()
            .map(|chunk| {
                let mut sub = PacketBatch::new(self.max_batch);
                for entry in chunk {
                    // Capacity is guaranteed because chunk.len() <= max_batch
                    let _ = sub.push(entry.data.clone(), entry.addr);
                }
                sub
            })
            .collect()
    }

    /// Records that a batch has been sent, updating counters.
    pub fn record_send(&mut self, stats: &BatchIoStats) {
        self.total_sent += stats.datagrams_ok as u64;
        self.total_bytes += stats.bytes_transferred as u64;
    }

    /// Returns total datagrams sent.
    #[must_use]
    pub fn total_sent(&self) -> u64 {
        self.total_sent
    }

    /// Returns total bytes sent.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }
}

// ---------------------------------------------------------------------------
// Receive coalescer
// ---------------------------------------------------------------------------

/// Coalesces individually received datagrams into batches for more
/// efficient downstream processing.
#[derive(Debug)]
pub struct ReceiveCoalescer {
    pending: PacketBatch,
    flush_threshold: usize,
    total_flushed: u64,
}

impl ReceiveCoalescer {
    /// Creates a new coalescer with the given flush threshold.
    #[must_use]
    pub fn new(flush_threshold: usize) -> Self {
        Self {
            pending: PacketBatch::new(flush_threshold),
            flush_threshold,
            total_flushed: 0,
        }
    }

    /// Ingests a single datagram.  Returns `Some(batch)` when the
    /// threshold is reached, triggering a flush.
    ///
    /// # Errors
    ///
    /// Returns an error if the pending batch overflows.
    pub fn ingest(&mut self, data: Bytes, addr: SocketAddr) -> VideoIpResult<Option<PacketBatch>> {
        self.pending.push(data, addr)?;
        if self.pending.len() >= self.flush_threshold {
            Ok(Some(self.flush()))
        } else {
            Ok(None)
        }
    }

    /// Forces a flush of whatever is pending.
    pub fn flush(&mut self) -> PacketBatch {
        let flushed = std::mem::replace(
            &mut self.pending,
            PacketBatch::new(self.flush_threshold),
        );
        self.total_flushed += 1;
        flushed
    }

    /// Returns how many datagrams are currently pending.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Returns total number of flushes.
    #[must_use]
    pub fn total_flushed(&self) -> u64 {
        self.total_flushed
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, SocketAddrV4};

    fn test_addr() -> SocketAddr {
        SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 5000))
    }

    #[test]
    fn test_packet_batch_push_and_len() {
        let mut batch = PacketBatch::new(4);
        assert!(batch.is_empty());
        assert_eq!(batch.remaining(), 4);

        batch.push(Bytes::from_static(b"hello"), test_addr()).ok();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch.remaining(), 3);
    }

    #[test]
    fn test_packet_batch_overflow() {
        let mut batch = PacketBatch::new(2);
        batch.push(Bytes::from_static(b"a"), test_addr()).ok();
        batch.push(Bytes::from_static(b"b"), test_addr()).ok();
        let result = batch.push(Bytes::from_static(b"c"), test_addr());
        assert!(result.is_err());
    }

    #[test]
    fn test_packet_batch_total_bytes() {
        let mut batch = PacketBatch::new(8);
        batch.push(Bytes::from_static(b"abc"), test_addr()).ok();
        batch.push(Bytes::from_static(b"defgh"), test_addr()).ok();
        assert_eq!(batch.total_bytes(), 8);
    }

    #[test]
    fn test_packet_batch_drain() {
        let mut batch = PacketBatch::new(4);
        batch.push(Bytes::from_static(b"x"), test_addr()).ok();
        batch.push(Bytes::from_static(b"y"), test_addr()).ok();
        let entries = batch.drain();
        assert_eq!(entries.len(), 2);
        assert!(batch.is_empty());
    }

    #[test]
    fn test_split_frame_payload() {
        let payload = vec![0u8; 1000];
        let chunks = split_frame_payload(&payload, 300);
        assert_eq!(chunks.len(), 4); // 300+300+300+100
        assert_eq!(chunks[0].len(), 300);
        assert_eq!(chunks[3].len(), 100);
    }

    #[test]
    fn test_split_frame_payload_zero_mtu() {
        let payload = vec![0u8; 100];
        let chunks = split_frame_payload(&payload, 0);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_reassemble_frame_payload() {
        let frags = vec![
            Bytes::from_static(b"hel"),
            Bytes::from_static(b"lo "),
            Bytes::from_static(b"world"),
        ];
        let result = reassemble_frame_payload(&frags, 1024);
        assert!(result.is_ok());
        assert_eq!(result.expect("reassemble"), Bytes::from_static(b"hello world"));
    }

    #[test]
    fn test_reassemble_frame_payload_exceeds_max() {
        let frags = vec![Bytes::from(vec![0u8; 600]), Bytes::from(vec![0u8; 600])];
        let result = reassemble_frame_payload(&frags, 1000);
        assert!(result.is_err());
    }

    #[test]
    fn test_scatter_send_sim() {
        let mut batch = PacketBatch::new(8);
        batch.push(Bytes::from_static(b"packet1"), test_addr()).ok();
        batch.push(Bytes::from_static(b"packet2"), test_addr()).ok();
        batch.push(Bytes::from(Vec::new()), test_addr()).ok(); // empty -> fail
        let stats = scatter_send_sim(&batch);
        assert_eq!(stats.datagrams_ok, 2);
        assert_eq!(stats.datagrams_failed, 1);
        assert_eq!(stats.bytes_transferred, 14);
    }

    #[test]
    fn test_gather_recv_sim() {
        let raw = b"aaabbbcccc";
        let sizes = [3, 3, 4];
        let config = ScatterGatherConfig::default();
        let batch = gather_recv_sim(raw, &sizes, test_addr(), &config);
        assert!(batch.is_ok());
        let batch = batch.expect("gather_recv_sim");
        assert_eq!(batch.len(), 3);
        let entries: Vec<&IoEntry> = batch.iter().collect();
        assert_eq!(&entries[0].data[..], b"aaa");
        assert_eq!(&entries[1].data[..], b"bbb");
        assert_eq!(&entries[2].data[..], b"cccc");
    }

    #[test]
    fn test_gather_recv_sim_short_buffer() {
        let raw = b"short";
        let sizes = [3, 10]; // 10 exceeds remaining
        let config = ScatterGatherConfig::default();
        let result = gather_recv_sim(raw, &sizes, test_addr(), &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_paced_batch_sender_partition() {
        let sender = PacedBatchSender::new(100, 3);
        let mut batch = PacketBatch::new(8);
        for i in 0..7 {
            batch
                .push(Bytes::from(vec![i; 10]), test_addr())
                .ok();
        }
        let sub_batches = sender.partition(&batch);
        assert_eq!(sub_batches.len(), 3); // 3+3+1
        assert_eq!(sub_batches[0].len(), 3);
        assert_eq!(sub_batches[1].len(), 3);
        assert_eq!(sub_batches[2].len(), 1);
    }

    #[test]
    fn test_paced_batch_sender_record() {
        let mut sender = PacedBatchSender::new(50, 16);
        let stats = BatchIoStats {
            datagrams_ok: 10,
            datagrams_failed: 0,
            bytes_transferred: 5000,
        };
        sender.record_send(&stats);
        assert_eq!(sender.total_sent(), 10);
        assert_eq!(sender.total_bytes(), 5000);
    }

    #[test]
    fn test_receive_coalescer_threshold_flush() {
        let mut coal = ReceiveCoalescer::new(3);
        let r1 = coal.ingest(Bytes::from_static(b"a"), test_addr());
        assert!(r1.is_ok());
        assert!(r1.expect("r1").is_none());

        let r2 = coal.ingest(Bytes::from_static(b"b"), test_addr());
        assert!(r2.is_ok());
        assert!(r2.expect("r2").is_none());

        let r3 = coal.ingest(Bytes::from_static(b"c"), test_addr());
        assert!(r3.is_ok());
        let flushed = r3.expect("r3");
        assert!(flushed.is_some());
        assert_eq!(flushed.expect("flushed batch").len(), 3);
        assert_eq!(coal.pending_count(), 0);
        assert_eq!(coal.total_flushed(), 1);
    }

    #[test]
    fn test_receive_coalescer_manual_flush() {
        let mut coal = ReceiveCoalescer::new(10);
        coal.ingest(Bytes::from_static(b"x"), test_addr()).ok();
        coal.ingest(Bytes::from_static(b"y"), test_addr()).ok();
        assert_eq!(coal.pending_count(), 2);

        let batch = coal.flush();
        assert_eq!(batch.len(), 2);
        assert_eq!(coal.pending_count(), 0);
    }

    #[test]
    fn test_scatter_gather_config_validate() {
        let good = ScatterGatherConfig::default();
        assert!(good.validate().is_ok());

        let bad = ScatterGatherConfig {
            max_batch_size: 0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        let bad2 = ScatterGatherConfig {
            max_datagram_size: 0,
            ..Default::default()
        };
        assert!(bad2.validate().is_err());
    }

    #[test]
    fn test_scatter_gather_config_presets() {
        let hd = ScatterGatherConfig::hd_1080p60();
        assert_eq!(hd.max_batch_size, 128);
        assert!(hd.enable_gso);

        let uhd = ScatterGatherConfig::uhd_4k();
        assert_eq!(uhd.max_batch_size, 256);
        assert!(uhd.enable_gro);
    }
}
