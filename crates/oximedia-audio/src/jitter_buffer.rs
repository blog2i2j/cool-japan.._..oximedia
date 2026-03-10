//! Network jitter buffer for smoothing bursty audio packet delivery.
#![allow(dead_code)]

use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Operational state of the jitter buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JitterBufferState {
    /// Collecting packets before playback starts.
    Filling,
    /// Normal operation.
    Ok,
    /// Buffer is starved and playback is stalling.
    Underrun,
    /// Buffer is full and discarding late packets.
    Overrun,
}

impl JitterBufferState {
    /// Returns `true` when the buffer is in a healthy operational state.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        matches!(self, JitterBufferState::Ok)
    }

    /// Human-readable state label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            JitterBufferState::Filling => "Filling",
            JitterBufferState::Ok => "OK",
            JitterBufferState::Underrun => "Underrun",
            JitterBufferState::Overrun => "Overrun",
        }
    }
}

/// A single packet entry inside the jitter buffer.
#[derive(Debug, Clone)]
pub struct JitterEntry {
    /// RTP-style sequence number.
    pub seq: u32,
    /// Presentation timestamp in samples.
    pub pts_samples: u64,
    /// Sample rate of the payload (Hz).
    pub sample_rate: u32,
    /// PCM payload samples (f32, interleaved).
    pub payload: Vec<f32>,
    /// Wall-clock arrival time expressed as milliseconds since an arbitrary epoch.
    pub arrival_ms: u64,
    /// Expected playout time in milliseconds from the same epoch.
    pub playout_ms: u64,
}

impl JitterEntry {
    /// Create a new entry.
    #[must_use]
    pub fn new(
        seq: u32,
        pts_samples: u64,
        sample_rate: u32,
        payload: Vec<f32>,
        arrival_ms: u64,
        playout_ms: u64,
    ) -> Self {
        Self {
            seq,
            pts_samples,
            sample_rate,
            payload,
            arrival_ms,
            playout_ms,
        }
    }

    /// Returns `true` when this packet arrived after its expected playout time.
    #[must_use]
    pub fn is_late(&self) -> bool {
        self.arrival_ms > self.playout_ms
    }

    /// Latency of this packet in milliseconds (positive = late, negative = early).
    #[allow(clippy::cast_possible_wrap)]
    #[must_use]
    pub fn latency_ms(&self) -> i64 {
        self.arrival_ms as i64 - self.playout_ms as i64
    }
}

// Implement Ord/PartialOrd so packets can live in a min-heap ordered by pts.
impl PartialEq for JitterEntry {
    fn eq(&self, other: &Self) -> bool {
        self.pts_samples == other.pts_samples
    }
}
impl Eq for JitterEntry {}
impl PartialOrd for JitterEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for JitterEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.pts_samples.cmp(&other.pts_samples)
    }
}

/// Cumulative statistics for a jitter buffer session.
#[derive(Debug, Clone, Default)]
pub struct JitterStats {
    /// Total packets inserted.
    pub total_inserted: u64,
    /// Total packets drained.
    pub total_drained: u64,
    /// Total packets that arrived late.
    pub total_late: u64,
    /// Total packets discarded because the buffer was full.
    pub total_discarded: u64,
}

impl JitterStats {
    /// Percentage of packets that arrived late (0.0 – 100.0).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn late_pct(&self) -> f64 {
        if self.total_inserted == 0 {
            return 0.0;
        }
        self.total_late as f64 / self.total_inserted as f64 * 100.0
    }
}

/// Adaptive jitter buffer that re-orders out-of-sequence packets.
///
/// Packets are stored in a min-heap keyed by PTS.  When `drain` is called the
/// buffer emits packets in ascending timestamp order up to the requested playout
/// time.
pub struct JitterBuffer {
    /// Min-heap: smallest `pts_samples` first.
    heap: BinaryHeap<Reverse<JitterEntry>>,
    /// Maximum number of packets to hold.
    capacity: usize,
    /// Sample rate used for time calculations.
    sample_rate: u32,
    /// Current operational state.
    state: JitterBufferState,
    /// Minimum fill threshold before transitioning out of `Filling`.
    min_fill: usize,
    stats: JitterStats,
}

impl JitterBuffer {
    /// Create a new jitter buffer.
    ///
    /// - `capacity`: maximum packet count (packets are discarded when full).
    /// - `min_fill`: number of packets to accumulate before entering `Ok` state.
    /// - `sample_rate`: sample rate in Hz (used for depth calculations).
    #[must_use]
    pub fn new(capacity: usize, min_fill: usize, sample_rate: u32) -> Self {
        Self {
            heap: BinaryHeap::new(),
            capacity,
            sample_rate,
            state: JitterBufferState::Filling,
            min_fill: min_fill.max(1),
            stats: JitterStats::default(),
        }
    }

    /// Insert a packet into the buffer.
    ///
    /// Returns `false` when the buffer is full and the packet was discarded.
    pub fn insert(&mut self, entry: JitterEntry) -> bool {
        if entry.is_late() {
            self.stats.total_late += 1;
        }
        if self.heap.len() >= self.capacity {
            self.stats.total_discarded += 1;
            self.state = JitterBufferState::Overrun;
            return false;
        }
        self.stats.total_inserted += 1;
        self.heap.push(Reverse(entry));

        // Transition out of Filling once we have enough packets.
        if self.state == JitterBufferState::Filling && self.heap.len() >= self.min_fill {
            self.state = JitterBufferState::Ok;
        } else if self.state == JitterBufferState::Overrun && self.heap.len() < self.capacity {
            self.state = JitterBufferState::Ok;
        }

        true
    }

    /// Drain all packets whose `pts_samples` is ≤ `up_to_pts`.
    ///
    /// Returns the drained entries in ascending timestamp order.
    pub fn drain(&mut self, up_to_pts: u64) -> Vec<JitterEntry> {
        let mut out = Vec::new();
        loop {
            match self.heap.peek() {
                Some(Reverse(e)) if e.pts_samples <= up_to_pts => {
                    if let Some(Reverse(entry)) = self.heap.pop() {
                        out.push(entry);
                        self.stats.total_drained += 1;
                    }
                }
                _ => break,
            }
        }
        if self.heap.is_empty() && self.state == JitterBufferState::Ok {
            self.state = JitterBufferState::Underrun;
        }
        out
    }

    /// Approximate depth of the buffer in milliseconds.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn depth_ms(&self) -> f64 {
        if self.sample_rate == 0 || self.heap.is_empty() {
            return 0.0;
        }
        // Sum up durations of all buffered entries.
        let total_samples: usize = self.heap.iter().map(|Reverse(e)| e.payload.len()).sum();
        total_samples as f64 / self.sample_rate as f64 * 1_000.0
    }

    /// Current state of the buffer.
    #[must_use]
    pub fn state(&self) -> JitterBufferState {
        self.state
    }

    /// Reference to accumulated statistics.
    #[must_use]
    pub fn stats(&self) -> &JitterStats {
        &self.stats
    }

    /// Number of packets currently held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Returns `true` when no packets are held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(seq: u32, pts: u64, samples: usize, arrival: u64, playout: u64) -> JitterEntry {
        JitterEntry::new(seq, pts, 48_000, vec![0.0; samples], arrival, playout)
    }

    #[test]
    fn test_state_is_ok() {
        assert!(JitterBufferState::Ok.is_ok());
        assert!(!JitterBufferState::Filling.is_ok());
        assert!(!JitterBufferState::Underrun.is_ok());
        assert!(!JitterBufferState::Overrun.is_ok());
    }

    #[test]
    fn test_state_labels() {
        assert_eq!(JitterBufferState::Filling.label(), "Filling");
        assert_eq!(JitterBufferState::Underrun.label(), "Underrun");
    }

    #[test]
    fn test_entry_is_late_true() {
        let e = entry(0, 0, 480, 100, 50); // arrived at 100, should play at 50
        assert!(e.is_late());
    }

    #[test]
    fn test_entry_is_late_false() {
        let e = entry(0, 0, 480, 30, 50); // arrived at 30, should play at 50
        assert!(!e.is_late());
    }

    #[test]
    fn test_entry_latency_ms_positive() {
        let e = entry(0, 0, 480, 100, 50);
        assert_eq!(e.latency_ms(), 50);
    }

    #[test]
    fn test_entry_latency_ms_negative() {
        let e = entry(0, 0, 480, 20, 50);
        assert_eq!(e.latency_ms(), -30);
    }

    #[test]
    fn test_jitter_buffer_filling_state_initially() {
        let jb = JitterBuffer::new(16, 4, 48_000);
        assert_eq!(jb.state(), JitterBufferState::Filling);
    }

    #[test]
    fn test_jitter_buffer_transitions_to_ok() {
        let mut jb = JitterBuffer::new(16, 2, 48_000);
        jb.insert(entry(0, 0, 480, 0, 10));
        assert_eq!(jb.state(), JitterBufferState::Filling);
        jb.insert(entry(1, 480, 480, 10, 20));
        assert_eq!(jb.state(), JitterBufferState::Ok);
    }

    #[test]
    fn test_jitter_buffer_insert_returns_false_when_full() {
        let mut jb = JitterBuffer::new(2, 1, 48_000);
        assert!(jb.insert(entry(0, 0, 480, 0, 10)));
        assert!(jb.insert(entry(1, 480, 480, 0, 10)));
        assert!(!jb.insert(entry(2, 960, 480, 0, 10)));
    }

    #[test]
    fn test_jitter_buffer_drain_in_order() {
        let mut jb = JitterBuffer::new(16, 1, 48_000);
        jb.insert(entry(2, 960, 480, 0, 10));
        jb.insert(entry(0, 0, 480, 0, 10));
        jb.insert(entry(1, 480, 480, 0, 10));
        let drained = jb.drain(1440); // drain all three
        assert_eq!(drained.len(), 3);
        assert_eq!(drained[0].pts_samples, 0);
        assert_eq!(drained[1].pts_samples, 480);
        assert_eq!(drained[2].pts_samples, 960);
    }

    #[test]
    fn test_jitter_buffer_drain_partial() {
        let mut jb = JitterBuffer::new(16, 1, 48_000);
        jb.insert(entry(0, 0, 480, 0, 10));
        jb.insert(entry(1, 480, 480, 0, 10));
        let drained = jb.drain(0); // only the first
        assert_eq!(drained.len(), 1);
        assert_eq!(jb.len(), 1);
    }

    #[test]
    fn test_jitter_buffer_depth_ms() {
        let mut jb = JitterBuffer::new(16, 1, 48_000);
        jb.insert(entry(0, 0, 480, 0, 10)); // 10ms
        jb.insert(entry(1, 480, 480, 0, 10)); // 10ms
        let depth = jb.depth_ms();
        assert!((depth - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_stats_late_pct_zero_when_no_inserts() {
        let stats = JitterStats::default();
        assert_eq!(stats.late_pct(), 0.0);
    }

    #[test]
    fn test_stats_late_pct_calculation() {
        let stats = JitterStats {
            total_inserted: 10,
            total_late: 3,
            ..Default::default()
        };
        assert!((stats.late_pct() - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_jitter_buffer_is_empty_initially() {
        let jb = JitterBuffer::new(16, 4, 48_000);
        assert!(jb.is_empty());
    }

    #[test]
    fn test_jitter_buffer_underrun_after_drain_empty() {
        let mut jb = JitterBuffer::new(16, 1, 48_000);
        jb.insert(entry(0, 0, 480, 0, 10));
        jb.drain(0); // drain the only packet
        assert_eq!(jb.state(), JitterBufferState::Underrun);
    }
}
