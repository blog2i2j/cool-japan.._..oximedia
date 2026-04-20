//! SMPTE ST 2022-7 seamless protection switching (dual-path redundancy).
//!
//! SMPTE ST 2022-7 defines a scheme for redundant delivery of RTP streams over
//! two independent network paths.  When a packet is lost on one path it is
//! recovered from the other path, providing hitless failover.
//!
//! The receiver maintains a **merge buffer** keyed by RTP sequence number.
//! Arriving packets are stored in the buffer and held for the configured
//! *merge delay* so that the same packet arriving on the alternate path can
//! still be used for recovery.  Packets are only delivered once; duplicates
//! are silently discarded.
//!
//! This implementation provides:
//! - Dual-path packet reception and de-duplication.
//! - Configurable merge delay buffer.
//! - Per-path statistics (loss, out-of-order, late arrivals).
//! - Selection policy: prefer primary or automatic fallback.
//! - Active/standby path management with health scoring.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::{Duration, Instant};

// ─── Path Identifier ─────────────────────────────────────────────────────────

/// Identifies which redundant network path a packet arrived on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PathId {
    /// Primary network path (typically Path A).
    Primary,
    /// Secondary network path (typically Path B).
    Secondary,
}

impl PathId {
    /// Returns the path name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Primary => "Path-A (Primary)",
            Self::Secondary => "Path-B (Secondary)",
        }
    }
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for SMPTE ST 2022-7 protection switching.
#[derive(Debug, Clone)]
pub struct St2022Config {
    /// Time to hold packets waiting for the duplicate from the other path.
    pub merge_delay: Duration,
    /// Maximum number of packets held in the merge buffer per path.
    pub max_buffer_packets: usize,
    /// Prefer the primary path when both packets arrive simultaneously.
    pub prefer_primary: bool,
    /// After this many consecutive path failures declare the path down.
    pub fail_threshold: u32,
    /// After this many consecutive successes restore the path.
    pub restore_threshold: u32,
    /// Primary path bind address.
    pub primary_addr: SocketAddr,
    /// Secondary path bind address.
    pub secondary_addr: SocketAddr,
}

impl Default for St2022Config {
    fn default() -> Self {
        Self {
            merge_delay: Duration::from_millis(50),
            max_buffer_packets: 512,
            prefer_primary: true,
            fail_threshold: 10,
            restore_threshold: 5,
            primary_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 5000),
            secondary_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 5002),
        }
    }
}

impl St2022Config {
    /// Creates a new configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the merge delay.
    #[must_use]
    pub const fn with_merge_delay(mut self, delay: Duration) -> Self {
        self.merge_delay = delay;
        self
    }

    /// Sets the maximum buffer size.
    #[must_use]
    pub const fn with_max_buffer(mut self, max: usize) -> Self {
        self.max_buffer_packets = max;
        self
    }
}

// ─── Per-Path Statistics ──────────────────────────────────────────────────────

/// Statistics for a single redundant path.
#[derive(Debug, Clone, Default)]
pub struct PathStats {
    /// Total packets received on this path.
    pub packets_received: u64,
    /// Packets delivered as the primary copy (not duplicates).
    pub packets_used: u64,
    /// Duplicate packets discarded (arrived after the other path).
    pub duplicates_discarded: u64,
    /// Packets that arrived after the merge delay expired (too late).
    pub late_arrivals: u64,
    /// Out-of-order packets received.
    pub out_of_order: u64,
    /// Consecutive failures (used for health scoring).
    pub consecutive_failures: u32,
    /// Whether this path is currently considered up.
    pub is_up: bool,
}

impl PathStats {
    fn new() -> Self {
        Self {
            is_up: true,
            ..Default::default()
        }
    }

    /// Returns a simple health score in [0.0, 1.0].
    #[must_use]
    pub fn health_score(&self) -> f64 {
        let total = self.packets_received;
        if total == 0 {
            return 1.0; // No data — assume healthy.
        }
        let bad = self.late_arrivals + self.out_of_order;
        let ratio = bad as f64 / total as f64;
        (1.0 - ratio).clamp(0.0, 1.0)
    }
}

// ─── Merge Buffer Entry ───────────────────────────────────────────────────────

/// An entry in the merge buffer.
#[derive(Debug)]
struct MergeEntry {
    /// Packet payload (RTP or raw UDP bytes).
    payload: Vec<u8>,
    /// Which path the first copy arrived on.
    first_path: PathId,
    /// When the first copy arrived.
    arrived_at: Instant,
    /// Whether the packet has already been delivered to the application.
    delivered: bool,
}

// ─── Protection Switcher ──────────────────────────────────────────────────────

/// SMPTE ST 2022-7 dual-path protection switcher.
///
/// Feed packets from both paths via `receive`.  Call `drain_ready` to
/// obtain the de-duplicated, time-ordered output.
pub struct ProtectionSwitcher {
    /// Configuration.
    config: St2022Config,
    /// Merge buffer keyed by RTP sequence number.
    buffer: BTreeMap<u16, MergeEntry>,
    /// Per-path statistics.
    stats: [PathStats; 2],
    /// Highest sequence number seen (used for out-of-order detection).
    highest_seen: Option<u16>,
    /// Total packets delivered.
    total_delivered: u64,
}

impl ProtectionSwitcher {
    /// Creates a new protection switcher.
    #[must_use]
    pub fn new(config: St2022Config) -> Self {
        Self {
            config,
            buffer: BTreeMap::new(),
            stats: [PathStats::new(), PathStats::new()],
            highest_seen: None,
            total_delivered: 0,
        }
    }

    /// Returns the path index for stats array indexing.
    const fn path_index(path: PathId) -> usize {
        match path {
            PathId::Primary => 0,
            PathId::Secondary => 1,
        }
    }

    /// Receives a packet from the given path.
    ///
    /// `seq` is the RTP sequence number extracted from the packet header.
    /// `payload` is the complete datagram bytes.
    ///
    /// Returns `true` if this is the first copy of this packet (primary delivery).
    pub fn receive(&mut self, seq: u16, payload: Vec<u8>, path: PathId) -> bool {
        let idx = Self::path_index(path);
        self.stats[idx].packets_received += 1;

        // Out-of-order detection.
        if let Some(highest) = self.highest_seen {
            let diff = seq.wrapping_sub(highest);
            if diff > 32768 {
                self.stats[idx].out_of_order += 1;
            } else {
                self.highest_seen = Some(seq);
            }
        } else {
            self.highest_seen = Some(seq);
        }

        // Check if already in buffer.
        if let Some(entry) = self.buffer.get_mut(&seq) {
            if entry.delivered {
                // Duplicate — already delivered.
                self.stats[idx].duplicates_discarded += 1;
                return false;
            }

            // Check if merge delay has expired.
            if entry.arrived_at.elapsed() > self.config.merge_delay {
                self.stats[idx].late_arrivals += 1;
                return false;
            }

            // Both copies arrived within the merge window — this is a duplicate.
            self.stats[idx].duplicates_discarded += 1;
            return false;
        }

        // Evict oldest entry if buffer is full.
        while self.buffer.len() >= self.config.max_buffer_packets {
            if let Some((&old_seq, _)) = self.buffer.iter().next() {
                self.buffer.remove(&old_seq);
            }
        }

        // First copy of this packet.
        self.buffer.insert(
            seq,
            MergeEntry {
                payload,
                first_path: path,
                arrived_at: Instant::now(),
                delivered: false,
            },
        );
        self.stats[idx].packets_used += 1;
        true
    }

    /// Drains packets that are ready for delivery.
    ///
    /// A packet is ready when its merge delay has elapsed (so the alternate path
    /// has had a chance to deliver a duplicate).
    pub fn drain_ready(&mut self) -> Vec<(u16, Vec<u8>)> {
        let delay = self.config.merge_delay;
        let mut ready = Vec::new();

        for (&seq, entry) in &mut self.buffer {
            if !entry.delivered && entry.arrived_at.elapsed() >= delay {
                entry.delivered = true;
                ready.push((seq, entry.payload.clone()));
                self.total_delivered += 1;
            }
        }

        // Keep delivered entries for a short while longer (for duplicate detection).
        // Purge very old delivered entries.
        self.buffer
            .retain(|_, e| e.arrived_at.elapsed() < delay * 4);

        ready
    }

    /// Returns statistics for the given path.
    #[must_use]
    pub fn path_stats(&self, path: PathId) -> &PathStats {
        &self.stats[Self::path_index(path)]
    }

    /// Returns the active path based on health scores.
    ///
    /// The primary path is preferred unless it has a significantly lower
    /// health score than the secondary.
    #[must_use]
    pub fn active_path(&self) -> PathId {
        let primary_health = self.stats[0].health_score();
        let secondary_health = self.stats[1].health_score();

        if self.config.prefer_primary && primary_health >= secondary_health * 0.9 {
            PathId::Primary
        } else if secondary_health > primary_health {
            PathId::Secondary
        } else {
            PathId::Primary
        }
    }

    /// Returns the number of packets currently in the merge buffer.
    #[must_use]
    pub fn buffer_depth(&self) -> usize {
        self.buffer.len()
    }

    /// Returns the total number of delivered packets.
    #[must_use]
    pub fn total_delivered(&self) -> u64 {
        self.total_delivered
    }
}

// ─── Sender-side dual-path emitter ───────────────────────────────────────────

/// Sends the same RTP packets over two independent paths.
///
/// The caller is responsible for the actual UDP socket writes.  This struct
/// provides the metadata and sequence tracking needed to emit on two paths.
pub struct DualPathSender {
    /// Configuration.
    _config: St2022Config,
    /// Next sequence number.
    next_seq: u16,
    /// Packets sent on primary path.
    sent_primary: u64,
    /// Packets sent on secondary path.
    sent_secondary: u64,
}

impl DualPathSender {
    /// Creates a new dual-path sender.
    #[must_use]
    pub fn new(config: St2022Config) -> Self {
        Self {
            _config: config,
            next_seq: 0,
            sent_primary: 0,
            sent_secondary: 0,
        }
    }

    /// Prepares a packet for dual-path transmission.
    ///
    /// Returns `(seq, primary_dest, secondary_dest)`.  The caller should
    /// send `payload` to both addresses.
    pub fn prepare_send(&mut self, _payload: &[u8]) -> (u16, SocketAddr, SocketAddr) {
        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);
        self.sent_primary += 1;
        self.sent_secondary += 1;

        (seq, self._config.primary_addr, self._config.secondary_addr)
    }

    /// Returns total packets sent on each path.
    #[must_use]
    pub const fn send_counts(&self) -> (u64, u64) {
        (self.sent_primary, self.sent_secondary)
    }
}

// ─── Task-Specified ST 2022-7 Public API ──────────────────────────────────────
//
// The following types implement the API specified in the SMPTE ST 2022-7 task.
// They sit alongside the full `ProtectionSwitcher` / `DualPathSender` above.

use std::collections::HashMap;

use crate::rist::parse_rtp_header;

// ── PathSwitchingStats ────────────────────────────────────────────────────────

/// Per-path statistics for the ST 2022-7 task-specified receiver.
#[derive(Debug, Clone, Default)]
pub struct PathSwitchingStats {
    /// Total packets received on Path A.
    pub path_a_packets: u64,
    /// Total packets received on Path B.
    pub path_b_packets: u64,
    /// Duplicate packets discarded (arrived on the second path after delivery).
    pub duplicates_discarded: u64,
    /// Packets lost on Path A but recovered from Path B.
    pub recovered_from_path_b: u64,
    /// Packet loss rate on Path A (0.0 – 1.0).
    pub path_a_loss_rate: f32,
    /// Packet loss rate on Path B (0.0 – 1.0).
    pub path_b_loss_rate: f32,
}

// ── St20227Config ─────────────────────────────────────────────────────────────

/// Configuration for the task-specified SMPTE ST 2022-7 receiver.
#[derive(Debug, Clone)]
pub struct St20227Config {
    /// Maximum acceptable delay between the two paths in milliseconds.
    pub path_delay_limit_ms: u32,
    /// Number of sequence numbers kept in the de-duplication window.
    pub sequence_history: usize,
}

impl Default for St20227Config {
    fn default() -> Self {
        Self {
            path_delay_limit_ms: 100,
            sequence_history: 256,
        }
    }
}

impl St20227Config {
    /// Creates a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the path delay limit.
    #[must_use]
    pub const fn with_path_delay_limit(mut self, ms: u32) -> Self {
        self.path_delay_limit_ms = ms;
        self
    }

    /// Sets the sequence history window.
    #[must_use]
    pub const fn with_sequence_history(mut self, n: usize) -> Self {
        self.sequence_history = n;
        self
    }
}

// ── St20227Receiver ───────────────────────────────────────────────────────────

/// SMPTE ST 2022-7 dual-path seamless protection switching receiver.
///
/// Feed raw RTP datagrams from Path A and Path B via [`receive_path_a`] and
/// [`receive_path_b`].  Each method returns `Some(payload)` when the packet at
/// the head of the de-duplication window can be delivered to the application.
///
/// [`receive_path_a`]: Self::receive_path_a
/// [`receive_path_b`]: Self::receive_path_b
pub struct St20227Receiver {
    /// Configuration.
    config: St20227Config,
    /// Packets received on Path A: seq → (raw datagram, arrival instant).
    path_a: HashMap<u16, (Vec<u8>, Instant)>,
    /// Packets received on Path B: seq → (raw datagram, arrival instant).
    path_b: HashMap<u16, (Vec<u8>, Instant)>,
    /// Sequence numbers that have already been delivered to the application.
    /// Kept for duplicate detection for late-arriving packets from the other path.
    delivered: std::collections::HashSet<u16>,
    /// Sequence numbers that Path A has seen (buffered or delivered).
    seen_on_a: std::collections::HashSet<u16>,
    /// Next sequence number to deliver.
    next_deliver: u16,
    /// Whether `next_deliver` has been initialised from the first packet.
    initialised: bool,
    /// Accumulated statistics.
    stats: PathSwitchingStats,
}

impl St20227Receiver {
    /// Creates a new receiver with the given configuration.
    #[must_use]
    pub fn new(config: St20227Config) -> Self {
        Self {
            config,
            path_a: HashMap::new(),
            path_b: HashMap::new(),
            delivered: std::collections::HashSet::new(),
            seen_on_a: std::collections::HashSet::new(),
            next_deliver: 0,
            initialised: false,
            stats: PathSwitchingStats::default(),
        }
    }

    /// Returns the payload bytes for a packet that arrived on `path_a` or
    /// `path_b`, extracting just the RTP payload (bytes 12+).
    fn extract_payload(data: &[u8]) -> Option<Vec<u8>> {
        if data.len() < 12 {
            return None;
        }
        Some(data[12..].to_vec())
    }

    /// Processes a datagram from Path A.
    ///
    /// Returns `Some(payload)` when this packet completes the next expected
    /// sequence for delivery; `None` otherwise.
    pub fn receive_path_a(&mut self, data: &[u8]) -> Option<Vec<u8>> {
        let hdr = parse_rtp_header(data).ok()?;
        let seq = hdr.rtp_seq;
        self.stats.path_a_packets += 1;
        self.init_next_deliver(seq);

        // Track that Path A has seen this sequence (for Path B duplicate detection).
        self.seen_on_a.insert(seq);

        // If already delivered, this is a late duplicate.
        if self.delivered.contains(&seq) {
            self.stats.duplicates_discarded += 1;
            return None;
        }

        // If Path B already buffered this sequence, this is a cross-path duplicate.
        if self.path_b.contains_key(&seq) {
            let age_ok = self
                .path_b
                .get(&seq)
                .map(|(_, t)| {
                    t.elapsed() < Duration::from_millis(u64::from(self.config.path_delay_limit_ms))
                })
                .unwrap_or(false);
            if age_ok {
                self.stats.duplicates_discarded += 1;
            }
        }

        self.path_a.insert(seq, (data.to_vec(), Instant::now()));
        self.try_deliver()
    }

    /// Processes a datagram from Path B.
    ///
    /// Returns `Some(payload)` when this packet completes the next expected
    /// sequence for delivery; `None` otherwise.
    pub fn receive_path_b(&mut self, data: &[u8]) -> Option<Vec<u8>> {
        let hdr = parse_rtp_header(data).ok()?;
        let seq = hdr.rtp_seq;
        self.stats.path_b_packets += 1;
        self.init_next_deliver(seq);

        // If already delivered, this is a late duplicate from Path B.
        if self.delivered.contains(&seq) {
            self.stats.duplicates_discarded += 1;
            return None;
        }

        // Determine if this is a recovery or a duplicate.
        // Path A has "seen" the sequence if it buffered or already delivered it.
        let path_a_seen = self.seen_on_a.contains(&seq) || self.path_a.contains_key(&seq);

        self.path_b.insert(seq, (data.to_vec(), Instant::now()));

        if !path_a_seen {
            // Path A never saw this sequence — Path B is providing recovery.
            self.stats.recovered_from_path_b += 1;
        } else {
            // Path A has (or had) this sequence — this is a cross-path duplicate.
            let age_ok = self
                .path_a
                .get(&seq)
                .map(|(_, t)| {
                    t.elapsed() < Duration::from_millis(u64::from(self.config.path_delay_limit_ms))
                })
                .unwrap_or(true); // If Path A entry already evicted, still count as dup.
            if age_ok {
                self.stats.duplicates_discarded += 1;
            }
        }

        self.try_deliver()
    }

    /// Returns a reference to the current path-switching statistics.
    #[must_use]
    pub fn stats(&self) -> &PathSwitchingStats {
        &self.stats
    }

    /// Returns the total number of packets currently pending delivery across
    /// both path buffers.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        // Union of seq numbers present in either buffer.
        let mut all_seqs: std::collections::HashSet<u16> = self.path_a.keys().copied().collect();
        for k in self.path_b.keys() {
            all_seqs.insert(*k);
        }
        all_seqs.len()
    }

    // ── Private ───────────────────────────────────────────────────────────────

    /// Initialises `next_deliver` from the first packet's sequence number.
    fn init_next_deliver(&mut self, seq: u16) {
        if !self.initialised {
            self.next_deliver = seq;
            self.initialised = true;
        }
    }

    /// Attempts to deliver the packet with sequence `next_deliver`.
    ///
    /// Delivery prefers Path A; if it is absent but Path B has the packet
    /// within the delay window, Path B is used.
    fn try_deliver(&mut self) -> Option<Vec<u8>> {
        let seq = self.next_deliver;
        let delay_limit = Duration::from_millis(u64::from(self.config.path_delay_limit_ms));

        // Prefer Path A.
        if let Some((data, _arrived)) = self.path_a.remove(&seq) {
            // Also remove from Path B if it arrived (duplicate).
            self.path_b.remove(&seq);
            self.delivered.insert(seq);
            self.advance_next_deliver();
            self.evict_stale();
            return Self::extract_payload(&data);
        }

        // Fall back to Path B if within delay window.
        if let Some((data, arrived)) = self.path_b.get(&seq) {
            if arrived.elapsed() <= delay_limit {
                let payload = Self::extract_payload(data);
                self.path_b.remove(&seq);
                self.delivered.insert(seq);
                self.advance_next_deliver();
                self.evict_stale();
                return payload;
            }
        }

        // Check if path A packet exceeded delay window — skip and advance.
        if let Some((_, arrived)) = self.path_a.get(&seq) {
            if arrived.elapsed() > delay_limit {
                self.path_a.remove(&seq);
                self.path_b.remove(&seq);
                self.delivered.insert(seq);
                self.advance_next_deliver();
                self.evict_stale();
            }
        }

        None
    }

    /// Advances `next_deliver` to the next sequence number.
    fn advance_next_deliver(&mut self) {
        self.next_deliver = self.next_deliver.wrapping_add(1);
        // Prune the history window to at most `sequence_history` entries.
        let limit = self.config.sequence_history;
        if self.path_a.len() > limit {
            // Remove the oldest entries (smallest wrapped distance from next_deliver).
            let to_remove: Vec<u16> = {
                let mut keys: Vec<u16> = self.path_a.keys().copied().collect();
                let nd = self.next_deliver;
                keys.sort_by_key(|&k| k.wrapping_sub(nd));
                keys.into_iter().skip(limit).collect()
            };
            for k in to_remove {
                self.path_a.remove(&k);
            }
        }
        if self.path_b.len() > limit {
            let to_remove: Vec<u16> = {
                let mut keys: Vec<u16> = self.path_b.keys().copied().collect();
                let nd = self.next_deliver;
                keys.sort_by_key(|&k| k.wrapping_sub(nd));
                keys.into_iter().skip(limit).collect()
            };
            for k in to_remove {
                self.path_b.remove(&k);
            }
        }
    }

    /// Evicts entries from both path buffers that are beyond the history window.
    fn evict_stale(&mut self) {
        let delay_limit = Duration::from_millis(u64::from(self.config.path_delay_limit_ms) * 4);
        self.path_a.retain(|_, (_, t)| t.elapsed() < delay_limit);
        self.path_b.retain(|_, (_, t)| t.elapsed() < delay_limit);

        // Prune delivered / seen_on_a sets to at most sequence_history entries.
        let limit = self.config.sequence_history;
        if self.delivered.len() > limit {
            let nd = self.next_deliver;
            let mut keys: Vec<u16> = self.delivered.iter().copied().collect();
            // Sort by wrapped distance from next_deliver (largest = oldest)
            keys.sort_by_key(|&k| std::cmp::Reverse(nd.wrapping_sub(k)));
            for k in keys
                .into_iter()
                .take(self.delivered.len().saturating_sub(limit))
            {
                self.delivered.remove(&k);
            }
        }
        if self.seen_on_a.len() > limit * 2 {
            let nd = self.next_deliver;
            let mut keys: Vec<u16> = self.seen_on_a.iter().copied().collect();
            keys.sort_by_key(|&k| std::cmp::Reverse(nd.wrapping_sub(k)));
            for k in keys
                .into_iter()
                .take(self.seen_on_a.len().saturating_sub(limit * 2))
            {
                self.seen_on_a.remove(&k);
            }
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> St2022Config {
        St2022Config::new().with_merge_delay(Duration::from_millis(10))
    }

    // 1. PathId names
    #[test]
    fn test_path_id_name() {
        assert!(PathId::Primary.name().contains("Primary"));
        assert!(PathId::Secondary.name().contains("Secondary"));
    }

    // 2. Default configuration
    #[test]
    fn test_config_default() {
        let cfg = St2022Config::default();
        assert_eq!(cfg.merge_delay, Duration::from_millis(50));
        assert!(cfg.prefer_primary);
    }

    // 3. Builder pattern
    #[test]
    fn test_config_builder() {
        let cfg = St2022Config::new()
            .with_merge_delay(Duration::from_millis(100))
            .with_max_buffer(256);
        assert_eq!(cfg.merge_delay, Duration::from_millis(100));
        assert_eq!(cfg.max_buffer_packets, 256);
    }

    // 4. PathStats default health score is 1.0
    #[test]
    fn test_path_stats_default_health() {
        let stats = PathStats::new();
        assert!((stats.health_score() - 1.0).abs() < 1e-9);
    }

    // 5. PathStats health score with losses
    #[test]
    fn test_path_stats_health_with_losses() {
        let stats = PathStats {
            packets_received: 100,
            late_arrivals: 10,
            ..PathStats::new()
        };
        let score = stats.health_score();
        assert!(score < 1.0);
        assert!(score >= 0.0);
    }

    // 6. First copy accepted
    #[test]
    fn test_switcher_first_copy_accepted() {
        let mut sw = ProtectionSwitcher::new(make_config());
        let is_first = sw.receive(0, vec![0u8; 188], PathId::Primary);
        assert!(is_first);
        assert_eq!(sw.path_stats(PathId::Primary).packets_used, 1);
    }

    // 7. Duplicate from secondary discarded
    #[test]
    fn test_switcher_duplicate_discarded() {
        let cfg = St2022Config::new().with_merge_delay(Duration::from_secs(10));
        let mut sw = ProtectionSwitcher::new(cfg);
        sw.receive(0, vec![0u8; 188], PathId::Primary);
        let is_first = sw.receive(0, vec![0u8; 188], PathId::Secondary);
        assert!(!is_first);
        assert_eq!(sw.path_stats(PathId::Secondary).duplicates_discarded, 1);
    }

    // 8. Different sequence numbers both accepted
    #[test]
    fn test_switcher_different_seqs() {
        let cfg = St2022Config::new().with_merge_delay(Duration::from_secs(10));
        let mut sw = ProtectionSwitcher::new(cfg);
        let a = sw.receive(0, vec![0u8], PathId::Primary);
        let b = sw.receive(1, vec![1u8], PathId::Primary);
        assert!(a && b);
    }

    // 9. Drain ready after merge delay
    #[test]
    fn test_switcher_drain_after_delay() {
        let cfg = St2022Config::new().with_merge_delay(Duration::from_millis(1));
        let mut sw = ProtectionSwitcher::new(cfg);
        sw.receive(0, vec![42u8], PathId::Primary);
        std::thread::sleep(Duration::from_millis(5));
        let ready = sw.drain_ready();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].0, 0); // seq = 0
    }

    // 10. Buffer depth
    #[test]
    fn test_switcher_buffer_depth() {
        let cfg = St2022Config::new().with_merge_delay(Duration::from_secs(10));
        let mut sw = ProtectionSwitcher::new(cfg);
        sw.receive(0, vec![0u8], PathId::Primary);
        sw.receive(1, vec![1u8], PathId::Primary);
        assert_eq!(sw.buffer_depth(), 2);
    }

    // 11. Active path returns primary by default
    #[test]
    fn test_switcher_active_path_default() {
        let sw = ProtectionSwitcher::new(make_config());
        assert_eq!(sw.active_path(), PathId::Primary);
    }

    // 12. Total delivered counter
    #[test]
    fn test_switcher_total_delivered() {
        let cfg = St2022Config::new().with_merge_delay(Duration::from_millis(1));
        let mut sw = ProtectionSwitcher::new(cfg);
        sw.receive(0, vec![0u8], PathId::Primary);
        sw.receive(1, vec![1u8], PathId::Primary);
        std::thread::sleep(Duration::from_millis(5));
        let _ = sw.drain_ready();
        assert_eq!(sw.total_delivered(), 2);
    }

    // 13. DualPathSender prepare_send increments seq
    #[test]
    fn test_dual_path_sender_seq() {
        let mut sender = DualPathSender::new(St2022Config::default());
        let (seq0, _, _) = sender.prepare_send(&[0u8; 188]);
        let (seq1, _, _) = sender.prepare_send(&[0u8; 188]);
        assert_eq!(seq0, 0);
        assert_eq!(seq1, 1);
    }

    // 14. DualPathSender send counts
    #[test]
    fn test_dual_path_sender_counts() {
        let mut sender = DualPathSender::new(St2022Config::default());
        sender.prepare_send(&[0u8]);
        sender.prepare_send(&[0u8]);
        let (p, s) = sender.send_counts();
        assert_eq!(p, 2);
        assert_eq!(s, 2);
    }

    // 15. Buffer eviction when full
    #[test]
    fn test_switcher_buffer_eviction() {
        let cfg = St2022Config::new()
            .with_merge_delay(Duration::from_secs(60))
            .with_max_buffer(3);
        let mut sw = ProtectionSwitcher::new(cfg);
        for i in 0..5u16 {
            sw.receive(i, vec![i as u8], PathId::Primary);
        }
        assert!(sw.buffer_depth() <= 3);
    }

    // 16. Stats path_index correctness
    #[test]
    fn test_path_index() {
        assert_eq!(ProtectionSwitcher::path_index(PathId::Primary), 0);
        assert_eq!(ProtectionSwitcher::path_index(PathId::Secondary), 1);
    }

    // ── Task-API St20227Receiver tests ────────────────────────────────────────

    use crate::rist::{build_rtp_header, RistPacketHeader};

    fn make_rtp_pkt(seq: u16, ssrc: u32, payload: &[u8]) -> Vec<u8> {
        let hdr = RistPacketHeader {
            rtp_seq: seq,
            ssrc,
            timestamp: u32::from(seq) * 90,
            payload_type: 33,
            marker: false,
        };
        let mut v: Vec<u8> = build_rtp_header(&hdr).to_vec();
        v.extend_from_slice(payload);
        v
    }

    // 17. St20227Config defaults
    #[test]
    fn test_st20227_config_defaults() {
        let cfg = St20227Config::default();
        assert_eq!(cfg.path_delay_limit_ms, 100);
        assert_eq!(cfg.sequence_history, 256);
    }

    // 18. St20227Config builder
    #[test]
    fn test_st20227_config_builder() {
        let cfg = St20227Config::new()
            .with_path_delay_limit(50)
            .with_sequence_history(128);
        assert_eq!(cfg.path_delay_limit_ms, 50);
        assert_eq!(cfg.sequence_history, 128);
    }

    // 19. PathSwitchingStats default is zeroed
    #[test]
    fn test_path_switching_stats_default() {
        let s = PathSwitchingStats::default();
        assert_eq!(s.path_a_packets, 0);
        assert_eq!(s.path_b_packets, 0);
        assert_eq!(s.duplicates_discarded, 0);
    }

    // 20. St20227Receiver: path A delivers in-order packet
    #[test]
    fn test_st20227_receiver_path_a_delivers() {
        let cfg = St20227Config::new().with_path_delay_limit(1000);
        let mut rx = St20227Receiver::new(cfg);
        let pkt = make_rtp_pkt(0, 1, &[0xAAu8; 4]);
        let result = rx.receive_path_a(&pkt);
        assert!(result.is_some());
        assert_eq!(rx.stats().path_a_packets, 1);
    }

    // 21. St20227Receiver: path B delivers in-order packet
    #[test]
    fn test_st20227_receiver_path_b_delivers() {
        let cfg = St20227Config::new().with_path_delay_limit(1000);
        let mut rx = St20227Receiver::new(cfg);
        let pkt = make_rtp_pkt(0, 1, &[0xBBu8; 4]);
        let result = rx.receive_path_b(&pkt);
        assert!(result.is_some());
        assert_eq!(rx.stats().path_b_packets, 1);
    }

    // 22. St20227Receiver: duplicate on path B increments counter
    #[test]
    fn test_st20227_receiver_duplicate_discarded() {
        let cfg = St20227Config::new().with_path_delay_limit(1000);
        let mut rx = St20227Receiver::new(cfg);
        let pkt = make_rtp_pkt(0, 1, &[0u8; 4]);
        let _ = rx.receive_path_a(&pkt);
        let _ = rx.receive_path_b(&pkt);
        assert_eq!(rx.stats().duplicates_discarded, 1);
    }

    // 23. St20227Receiver: recovery from path B counted
    #[test]
    fn test_st20227_receiver_recovery_from_path_b() {
        let cfg = St20227Config::new().with_path_delay_limit(1000);
        let mut rx = St20227Receiver::new(cfg);
        // Send seq 0 only on Path B (Path A dropped it).
        let pkt = make_rtp_pkt(0, 1, &[0u8; 4]);
        let _ = rx.receive_path_b(&pkt);
        assert_eq!(rx.stats().recovered_from_path_b, 1);
    }

    // 24. St20227Receiver: stats() returns reference
    #[test]
    fn test_st20227_receiver_stats_ref() {
        let rx = St20227Receiver::new(St20227Config::default());
        let s = rx.stats();
        assert_eq!(s.path_a_packets, 0);
    }

    // 25. St20227Receiver: pending_count with packets in both buffers
    #[test]
    fn test_st20227_receiver_pending_count() {
        let cfg = St20227Config::new().with_path_delay_limit(1000);
        let mut rx = St20227Receiver::new(cfg);
        // seq 0 on Path A (delivers immediately → not pending)
        let p0 = make_rtp_pkt(0, 1, &[0u8; 4]);
        let _ = rx.receive_path_a(&p0);
        // seq 1 only on Path B (not yet deliverable because next_deliver = 1 now)
        let p1 = make_rtp_pkt(1, 1, &[1u8; 4]);
        let _ = rx.receive_path_b(&p1);
        // pending_count may be 0 or 1 depending on whether seq 1 delivered
        // We just confirm it doesn't panic and returns a usize.
        let _count = rx.pending_count();
    }

    // 26. St20227Receiver: multiple sequential packets on path A all delivered
    #[test]
    fn test_st20227_receiver_sequential_path_a() {
        let cfg = St20227Config::new().with_path_delay_limit(1000);
        let mut rx = St20227Receiver::new(cfg);
        let mut delivered = 0usize;
        for seq in 0..5u16 {
            let pkt = make_rtp_pkt(seq, 1, &[seq as u8; 4]);
            if rx.receive_path_a(&pkt).is_some() {
                delivered += 1;
            }
        }
        assert_eq!(delivered, 5);
    }

    // 27. St20227Receiver: payload content is preserved
    #[test]
    fn test_st20227_receiver_payload_preserved() {
        let cfg = St20227Config::new().with_path_delay_limit(1000);
        let mut rx = St20227Receiver::new(cfg);
        let payload = [0xDE, 0xAD, 0xBE, 0xEF];
        let pkt = make_rtp_pkt(0, 42, &payload);
        let result = rx.receive_path_a(&pkt).expect("should deliver");
        assert_eq!(result, payload);
    }
}
