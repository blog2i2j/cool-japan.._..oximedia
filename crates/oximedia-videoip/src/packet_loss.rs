#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]

//! Packet loss detection, tracking, and recovery metrics.
//!
//! This module tracks incoming packet sequence numbers to detect gaps (lost
//! packets), compute loss rates over sliding windows, and generate loss event
//! reports. It is used by the receiver side to monitor stream health and
//! trigger FEC recovery or retransmission requests.

use std::collections::VecDeque;

/// Default sliding window size for loss rate computation.
const DEFAULT_WINDOW_SIZE: usize = 1000;

/// Maximum gap before treating it as a sequence reset rather than loss.
const MAX_GAP_BEFORE_RESET: u64 = 10_000;

/// A detected packet loss event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LossEvent {
    /// First missing sequence number.
    pub start_seq: u64,
    /// Number of consecutive missing packets.
    pub count: u64,
    /// Timestamp (monotonic counter) when the loss was detected.
    pub detection_time: u64,
}

impl LossEvent {
    /// Create a new loss event.
    pub fn new(start_seq: u64, count: u64, detection_time: u64) -> Self {
        Self {
            start_seq,
            count,
            detection_time,
        }
    }

    /// Return the last missing sequence number (inclusive).
    pub fn end_seq(&self) -> u64 {
        self.start_seq + self.count.saturating_sub(1)
    }
}

/// Aggregate loss statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct LossStats {
    /// Total packets received.
    pub total_received: u64,
    /// Total packets lost.
    pub total_lost: u64,
    /// Total packets expected (received + lost).
    pub total_expected: u64,
    /// Overall loss rate (0.0 to 1.0).
    pub loss_rate: f64,
    /// Loss rate in the current sliding window.
    pub window_loss_rate: f64,
    /// Number of distinct loss events.
    pub loss_event_count: usize,
    /// Average burst length (consecutive losses per event).
    pub avg_burst_length: f64,
    /// Maximum burst length.
    pub max_burst_length: u64,
}

/// Severity classification for packet loss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossSeverity {
    /// No loss detected.
    None,
    /// Minor loss (<0.1%), typically recoverable by FEC.
    Minor,
    /// Moderate loss (0.1%-1%), may cause visible artifacts.
    Moderate,
    /// Severe loss (1%-5%), likely to cause noticeable degradation.
    Severe,
    /// Critical loss (>5%), stream is severely degraded.
    Critical,
}

impl LossSeverity {
    /// Classify from a loss rate (0.0 to 1.0).
    pub fn from_rate(rate: f64) -> Self {
        if rate <= 0.0 {
            Self::None
        } else if rate < 0.001 {
            Self::Minor
        } else if rate < 0.01 {
            Self::Moderate
        } else if rate < 0.05 {
            Self::Severe
        } else {
            Self::Critical
        }
    }

    /// Returns a human-readable description.
    pub fn description(self) -> &'static str {
        match self {
            Self::None => "no loss",
            Self::Minor => "minor (<0.1%, FEC recoverable)",
            Self::Moderate => "moderate (0.1%-1%, possible artifacts)",
            Self::Severe => "severe (1%-5%, noticeable degradation)",
            Self::Critical => "critical (>5%, severely degraded)",
        }
    }
}

/// Tracker for packet loss detection and statistics.
#[derive(Debug)]
pub struct PacketLossTracker {
    /// Next expected sequence number.
    next_expected_seq: Option<u64>,
    /// Total packets received.
    total_received: u64,
    /// Total packets lost.
    total_lost: u64,
    /// Sliding window of received/lost flags (true = received).
    window: VecDeque<bool>,
    /// Window size.
    window_size: usize,
    /// Recorded loss events.
    loss_events: Vec<LossEvent>,
    /// Monotonic time counter.
    time_counter: u64,
}

impl PacketLossTracker {
    /// Create a new packet loss tracker.
    pub fn new(window_size: usize) -> Self {
        let ws = if window_size == 0 {
            DEFAULT_WINDOW_SIZE
        } else {
            window_size
        };
        Self {
            next_expected_seq: None,
            total_received: 0,
            total_lost: 0,
            window: VecDeque::with_capacity(ws),
            window_size: ws,
            loss_events: Vec::new(),
            time_counter: 0,
        }
    }

    /// Create with default window size.
    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_WINDOW_SIZE)
    }

    /// Record an incoming packet with the given sequence number.
    pub fn record_packet(&mut self, seq: u64) {
        self.time_counter += 1;

        match self.next_expected_seq {
            None => {
                // First packet — initialize.
                self.next_expected_seq = Some(seq + 1);
                self.total_received += 1;
                self.push_window(true);
            }
            Some(expected) => {
                if seq == expected {
                    // In order.
                    self.next_expected_seq = Some(seq + 1);
                    self.total_received += 1;
                    self.push_window(true);
                } else if seq > expected {
                    let gap = seq - expected;
                    if gap > MAX_GAP_BEFORE_RESET {
                        // Sequence reset — treat as fresh start.
                        self.next_expected_seq = Some(seq + 1);
                        self.total_received += 1;
                        self.push_window(true);
                    } else {
                        // Lost packets in the gap.
                        self.total_lost += gap;
                        self.loss_events
                            .push(LossEvent::new(expected, gap, self.time_counter));
                        for _ in 0..gap {
                            self.push_window(false);
                        }
                        self.next_expected_seq = Some(seq + 1);
                        self.total_received += 1;
                        self.push_window(true);
                    }
                } else {
                    // Out of order or duplicate — count as received, don't move expected.
                    self.total_received += 1;
                    self.push_window(true);
                }
            }
        }
    }

    /// Push a received/lost flag into the sliding window.
    fn push_window(&mut self, received: bool) {
        self.window.push_back(received);
        if self.window.len() > self.window_size {
            self.window.pop_front();
        }
    }

    /// Compute the current sliding-window loss rate.
    pub fn window_loss_rate(&self) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        let lost = self.window.iter().filter(|&&r| !r).count();
        lost as f64 / self.window.len() as f64
    }

    /// Compute overall loss rate.
    pub fn overall_loss_rate(&self) -> f64 {
        let total = self.total_received + self.total_lost;
        if total == 0 {
            return 0.0;
        }
        self.total_lost as f64 / total as f64
    }

    /// Get the current loss severity.
    pub fn severity(&self) -> LossSeverity {
        LossSeverity::from_rate(self.window_loss_rate())
    }

    /// Compute full loss statistics.
    pub fn compute_stats(&self) -> LossStats {
        let total_expected = self.total_received + self.total_lost;
        let avg_burst = if self.loss_events.is_empty() {
            0.0
        } else {
            let total_burst: u64 = self.loss_events.iter().map(|e| e.count).sum();
            total_burst as f64 / self.loss_events.len() as f64
        };
        let max_burst = self.loss_events.iter().map(|e| e.count).max().unwrap_or(0);

        LossStats {
            total_received: self.total_received,
            total_lost: self.total_lost,
            total_expected,
            loss_rate: self.overall_loss_rate(),
            window_loss_rate: self.window_loss_rate(),
            loss_event_count: self.loss_events.len(),
            avg_burst_length: avg_burst,
            max_burst_length: max_burst,
        }
    }

    /// Return all recorded loss events.
    pub fn loss_events(&self) -> &[LossEvent] {
        &self.loss_events
    }

    /// Return total received packet count.
    pub fn total_received(&self) -> u64 {
        self.total_received
    }

    /// Return total lost packet count.
    pub fn total_lost(&self) -> u64 {
        self.total_lost
    }

    /// Reset the tracker.
    pub fn reset(&mut self) {
        self.next_expected_seq = None;
        self.total_received = 0;
        self.total_lost = 0;
        self.window.clear();
        self.loss_events.clear();
        self.time_counter = 0;
    }
}

impl Default for PacketLossTracker {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loss_event_creation() {
        let e = LossEvent::new(10, 3, 100);
        assert_eq!(e.start_seq, 10);
        assert_eq!(e.count, 3);
        assert_eq!(e.end_seq(), 12);
    }

    #[test]
    fn test_empty_tracker() {
        let t = PacketLossTracker::with_defaults();
        assert_eq!(t.total_received(), 0);
        assert_eq!(t.total_lost(), 0);
        assert!((t.overall_loss_rate()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sequential_packets_no_loss() {
        let mut t = PacketLossTracker::with_defaults();
        for i in 0..100 {
            t.record_packet(i);
        }
        assert_eq!(t.total_received(), 100);
        assert_eq!(t.total_lost(), 0);
        assert!((t.overall_loss_rate()).abs() < f64::EPSILON);
        assert_eq!(t.severity(), LossSeverity::None);
    }

    #[test]
    fn test_single_loss() {
        let mut t = PacketLossTracker::with_defaults();
        t.record_packet(0);
        t.record_packet(1);
        // Skip 2
        t.record_packet(3);
        assert_eq!(t.total_lost(), 1);
        assert_eq!(t.loss_events().len(), 1);
        assert_eq!(t.loss_events()[0].start_seq, 2);
        assert_eq!(t.loss_events()[0].count, 1);
    }

    #[test]
    fn test_burst_loss() {
        let mut t = PacketLossTracker::with_defaults();
        t.record_packet(0);
        // Skip 1,2,3,4
        t.record_packet(5);
        assert_eq!(t.total_lost(), 4);
        assert_eq!(t.loss_events().len(), 1);
        assert_eq!(t.loss_events()[0].count, 4);
    }

    #[test]
    fn test_loss_rate_calculation() {
        let mut t = PacketLossTracker::new(100);
        for i in 0..10 {
            t.record_packet(i);
        }
        // Skip 10
        t.record_packet(11);
        // 11 received, 1 lost => rate = 1/12
        let rate = t.overall_loss_rate();
        assert!((rate - 1.0 / 12.0).abs() < 0.001);
    }

    #[test]
    fn test_window_loss_rate() {
        let mut t = PacketLossTracker::new(10);
        for i in 0..5 {
            t.record_packet(i);
        }
        // Skip 5
        t.record_packet(6);
        // Window has 5 received + 1 lost + 1 received = 7 entries, 1 lost
        let wlr = t.window_loss_rate();
        assert!(wlr > 0.0);
        assert!(wlr < 0.5);
    }

    #[test]
    fn test_severity_none() {
        let mut t = PacketLossTracker::with_defaults();
        for i in 0..100 {
            t.record_packet(i);
        }
        assert_eq!(t.severity(), LossSeverity::None);
    }

    #[test]
    fn test_severity_classification() {
        assert_eq!(LossSeverity::from_rate(0.0), LossSeverity::None);
        assert_eq!(LossSeverity::from_rate(0.0005), LossSeverity::Minor);
        assert_eq!(LossSeverity::from_rate(0.005), LossSeverity::Moderate);
        assert_eq!(LossSeverity::from_rate(0.03), LossSeverity::Severe);
        assert_eq!(LossSeverity::from_rate(0.1), LossSeverity::Critical);
    }

    #[test]
    fn test_severity_description() {
        assert!(!LossSeverity::None.description().is_empty());
        assert!(!LossSeverity::Critical.description().is_empty());
    }

    #[test]
    fn test_stats_computation() {
        let mut t = PacketLossTracker::new(100);
        for i in 0..10 {
            t.record_packet(i);
        }
        // Burst of 3 lost
        t.record_packet(13);
        // Single loss
        t.record_packet(15);
        let stats = t.compute_stats();
        assert_eq!(stats.total_received, 12);
        assert_eq!(stats.total_lost, 4);
        assert_eq!(stats.loss_event_count, 2);
        assert_eq!(stats.max_burst_length, 3);
    }

    #[test]
    fn test_sequence_reset() {
        let mut t = PacketLossTracker::with_defaults();
        t.record_packet(0);
        t.record_packet(1);
        // Big gap — treated as reset, not loss
        t.record_packet(100_000);
        assert_eq!(t.total_lost(), 0);
        assert_eq!(t.total_received(), 3);
    }

    #[test]
    fn test_reset() {
        let mut t = PacketLossTracker::with_defaults();
        t.record_packet(0);
        t.record_packet(2); // 1 lost
        t.reset();
        assert_eq!(t.total_received(), 0);
        assert_eq!(t.total_lost(), 0);
        assert!(t.loss_events().is_empty());
    }
}
