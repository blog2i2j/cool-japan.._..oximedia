//! Packet-level repair and loss recovery for media streams.
//!
//! This module provides tools for detecting and repairing packet-level
//! corruption in media streams, including estimation of packet loss
//! and selection of repair strategies.

#![allow(dead_code)]

/// Pattern of packet loss observed in a media stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PacketLossPattern {
    /// Single isolated packet losses spread evenly.
    Random,
    /// Bursts of consecutive packet loss.
    Burst {
        /// Average burst length in packets.
        avg_burst_len: u32,
    },
    /// Periodic packet loss at a fixed interval.
    Periodic {
        /// Period in packets between losses.
        period: u32,
    },
    /// No discernible pattern.
    Unknown,
}

/// Strategy used to repair missing or corrupted packets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RepairStrategy {
    /// Interpolate from neighbouring packets.
    Interpolation,
    /// Repeat the previous good packet.
    PacketRepetition,
    /// Insert silence (audio) or black frame (video).
    Concealment,
    /// Use forward error correction data if available.
    ForwardErrorCorrection,
    /// Leave the gap unfilled (drop).
    Drop,
}

/// Configuration for packet repair operations.
#[derive(Debug, Clone)]
pub struct PacketRepairConfig {
    /// Maximum consecutive missing packets to repair.
    pub max_repair_run: usize,
    /// Strategy to use when loss pattern is random.
    pub random_strategy: RepairStrategy,
    /// Strategy to use when loss pattern is burst.
    pub burst_strategy: RepairStrategy,
    /// Whether to attempt FEC decoding before falling back.
    pub try_fec_first: bool,
}

impl Default for PacketRepairConfig {
    fn default() -> Self {
        Self {
            max_repair_run: 8,
            random_strategy: RepairStrategy::Interpolation,
            burst_strategy: RepairStrategy::PacketRepetition,
            try_fec_first: true,
        }
    }
}

/// Record of a single packet loss event.
#[derive(Debug, Clone)]
pub struct PacketLossEvent {
    /// Sequence number of the first missing packet.
    pub seq_start: u32,
    /// Number of consecutive packets missing.
    pub count: u32,
    /// Timestamp (in stream time units) of the gap.
    pub timestamp: u64,
    /// Whether the gap was repaired.
    pub repaired: bool,
}

/// Engine that analyses and repairs packet-level losses.
#[derive(Debug, Default)]
pub struct PacketRepairer {
    config: PacketRepairConfig,
    loss_events: Vec<PacketLossEvent>,
    total_packets_seen: u64,
    total_packets_lost: u64,
}

impl PacketRepairer {
    /// Create a new `PacketRepairer` with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new `PacketRepairer` with custom configuration.
    pub fn with_config(config: PacketRepairConfig) -> Self {
        Self {
            config,
            ..Default::default()
        }
    }

    /// Record a span of packets, noting any gaps in `seq_numbers`.
    ///
    /// `seq_numbers` must contain the sequence numbers of packets that were
    /// actually received, in ascending order.
    pub fn record_received(&mut self, seq_numbers: &[u32], timestamp_base: u64) {
        if seq_numbers.is_empty() {
            return;
        }
        let first = seq_numbers[0];
        let last = match seq_numbers.last() {
            Some(&v) => v,
            None => return,
        };
        let expected_count = (last - first + 1) as u64;
        self.total_packets_seen += expected_count;
        self.total_packets_lost += expected_count - seq_numbers.len() as u64;

        // Detect gaps
        let mut prev = first;
        for &seq in seq_numbers.iter().skip(1) {
            if seq > prev + 1 {
                self.loss_events.push(PacketLossEvent {
                    seq_start: prev + 1,
                    count: seq - prev - 1,
                    timestamp: timestamp_base + u64::from(prev + 1),
                    repaired: false,
                });
            }
            prev = seq;
        }
    }

    /// Estimate packet loss percentage over all recorded traffic.
    ///
    /// Returns a value in `[0.0, 100.0]`.
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate_loss_pct(&self) -> f64 {
        if self.total_packets_seen == 0 {
            return 0.0;
        }
        (self.total_packets_lost as f64 / self.total_packets_seen as f64) * 100.0
    }

    /// Classify the observed loss pattern.
    pub fn classify_pattern(&self) -> PacketLossPattern {
        if self.loss_events.is_empty() {
            return PacketLossPattern::Unknown;
        }
        let burst_events: Vec<_> = self.loss_events.iter().filter(|e| e.count > 2).collect();
        if burst_events.len() * 2 >= self.loss_events.len() {
            let avg = burst_events.iter().map(|e| e.count).sum::<u32>() / burst_events.len() as u32;
            return PacketLossPattern::Burst { avg_burst_len: avg };
        }
        PacketLossPattern::Random
    }

    /// Choose the best repair strategy for the current loss pattern.
    pub fn choose_strategy(&self) -> &RepairStrategy {
        match self.classify_pattern() {
            PacketLossPattern::Burst { .. } => &self.config.burst_strategy,
            _ => &self.config.random_strategy,
        }
    }

    /// Mark all outstanding loss events as repaired.
    pub fn mark_repaired(&mut self) {
        for event in &mut self.loss_events {
            event.repaired = true;
        }
    }

    /// Return the number of unrepaired loss events.
    pub fn unrepaired_count(&self) -> usize {
        self.loss_events.iter().filter(|e| !e.repaired).count()
    }

    /// Return total packets seen so far.
    pub fn total_packets_seen(&self) -> u64 {
        self.total_packets_seen
    }

    /// Return total packets lost so far.
    pub fn total_packets_lost(&self) -> u64 {
        self.total_packets_lost
    }

    /// All recorded loss events.
    pub fn loss_events(&self) -> &[PacketLossEvent] {
        &self.loss_events
    }

    /// Reset all internal counters and event history.
    pub fn reset(&mut self) {
        self.loss_events.clear();
        self.total_packets_seen = 0;
        self.total_packets_lost = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_loss() {
        let mut r = PacketRepairer::new();
        r.record_received(&[0, 1, 2, 3, 4], 0);
        assert_eq!(r.total_packets_lost(), 0);
        assert!((r.estimate_loss_pct() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_single_gap() {
        let mut r = PacketRepairer::new();
        // Missing seq 2
        r.record_received(&[0, 1, 3, 4], 0);
        assert_eq!(r.total_packets_lost(), 1);
        assert_eq!(r.loss_events().len(), 1);
        assert_eq!(r.loss_events()[0].count, 1);
    }

    #[test]
    fn test_loss_percentage() {
        let mut r = PacketRepairer::new();
        // 5 seen (0-4), 2 missing (2, 3)
        r.record_received(&[0, 1, 4], 0);
        let pct = r.estimate_loss_pct();
        assert!((pct - 40.0).abs() < 1e-6, "expected 40%, got {pct}");
    }

    #[test]
    fn test_zero_packets_no_panic() {
        let r = PacketRepairer::new();
        assert_eq!(r.estimate_loss_pct(), 0.0);
    }

    #[test]
    fn test_burst_pattern_classification() {
        let mut r = PacketRepairer::new();
        // Large burst: 0-2 received, 3-10 missing, 11 received
        r.record_received(&[0, 1, 2, 11], 0);
        let pattern = r.classify_pattern();
        assert!(
            matches!(pattern, PacketLossPattern::Burst { .. }),
            "expected Burst, got {pattern:?}"
        );
    }

    #[test]
    fn test_random_pattern_classification() {
        let mut r = PacketRepairer::new();
        // Single packet gaps
        r.record_received(&[0, 2, 4, 6, 8], 0);
        let pattern = r.classify_pattern();
        assert_eq!(pattern, PacketLossPattern::Random);
    }

    #[test]
    fn test_mark_repaired() {
        let mut r = PacketRepairer::new();
        r.record_received(&[0, 2, 4], 0);
        assert_eq!(r.unrepaired_count(), 2);
        r.mark_repaired();
        assert_eq!(r.unrepaired_count(), 0);
    }

    #[test]
    fn test_reset() {
        let mut r = PacketRepairer::new();
        r.record_received(&[0, 2, 4], 0);
        r.reset();
        assert_eq!(r.total_packets_seen(), 0);
        assert_eq!(r.total_packets_lost(), 0);
        assert!(r.loss_events().is_empty());
    }

    #[test]
    fn test_choose_strategy_random() {
        let mut r = PacketRepairer::new();
        r.record_received(&[0, 2, 4, 6], 0);
        assert_eq!(r.choose_strategy(), &RepairStrategy::Interpolation);
    }

    #[test]
    fn test_config_custom() {
        let cfg = PacketRepairConfig {
            max_repair_run: 4,
            random_strategy: RepairStrategy::Concealment,
            burst_strategy: RepairStrategy::Drop,
            try_fec_first: false,
        };
        let r = PacketRepairer::with_config(cfg.clone());
        assert_eq!(r.config.max_repair_run, 4);
        assert_eq!(r.config.random_strategy, RepairStrategy::Concealment);
    }

    #[test]
    fn test_loss_event_timestamp() {
        let mut r = PacketRepairer::new();
        r.record_received(&[10, 15], 1000);
        assert!(!r.loss_events().is_empty());
        // The gap starts at seq 11
        assert_eq!(r.loss_events()[0].seq_start, 11);
    }

    #[test]
    fn test_multiple_gaps() {
        let mut r = PacketRepairer::new();
        // Gaps: 1, 3, 5
        r.record_received(&[0, 2, 4, 6], 0);
        assert_eq!(r.loss_events().len(), 3);
        assert_eq!(r.total_packets_lost(), 3);
    }

    #[test]
    fn test_pattern_unknown_no_events() {
        let r = PacketRepairer::new();
        assert_eq!(r.classify_pattern(), PacketLossPattern::Unknown);
    }

    #[test]
    fn test_repair_strategy_fec() {
        let cfg = PacketRepairConfig {
            random_strategy: RepairStrategy::ForwardErrorCorrection,
            ..Default::default()
        };
        let mut r = PacketRepairer::with_config(cfg);
        r.record_received(&[0, 2], 0);
        assert_eq!(r.choose_strategy(), &RepairStrategy::ForwardErrorCorrection);
    }

    #[test]
    fn test_accumulate_multiple_calls() {
        let mut r = PacketRepairer::new();
        r.record_received(&[0, 1, 2], 0);
        r.record_received(&[10, 11, 12], 100);
        // Each span: 3 seen, 0 lost
        assert_eq!(r.total_packets_seen(), 6);
        assert_eq!(r.total_packets_lost(), 0);
    }
}
