//! Double-buffered GPU command submission.
//!
//! Overlaps CPU work (encoding the next frame's commands) with GPU work
//! (executing the current frame's commands) using a ping-pong buffer scheme.
//!
//! # Architecture
//!
//! ```text
//! Frame N:   CPU encodes buffer A   ───▶  GPU executes buffer A
//! Frame N+1: CPU encodes buffer B   ───▶  GPU executes buffer B
//! Frame N+2: CPU encodes buffer A   ───▶  GPU executes buffer A  (recycled)
//!            └──── overlapped ─────┘
//! ```
//!
//! The [`DoubleBufferSubmitter`] manages two command slots and alternates
//! between them.  While the GPU processes slot A, the CPU can prepare
//! commands in slot B, minimising idle time on both sides.
//!
//! # Status
//!
//! This module provides the scheduling and statistics logic.  Actual GPU
//! command execution is delegated to the caller via closures.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ─── Slot state ─────────────────────────────────────────────────────────────

/// State of a single command buffer slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    /// The slot is available for the CPU to encode commands.
    Idle,
    /// The CPU is currently encoding commands into this slot.
    Encoding,
    /// Commands have been submitted to the GPU and are in flight.
    InFlight,
    /// The GPU has finished executing; results can be read back.
    Complete,
}

impl std::fmt::Display for SlotState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Idle => write!(f, "idle"),
            Self::Encoding => write!(f, "encoding"),
            Self::InFlight => write!(f, "in-flight"),
            Self::Complete => write!(f, "complete"),
        }
    }
}

// ─── Slot ───────────────────────────────────────────────────────────────────

/// A command buffer slot with timing information.
#[derive(Debug, Clone)]
pub struct CommandSlot {
    /// Current state.
    state: SlotState,
    /// Frame index assigned to this slot (monotonically increasing).
    frame_index: u64,
    /// When encoding started for this slot.
    encode_start: Option<Instant>,
    /// Duration of the encoding phase.
    encode_duration: Option<Duration>,
    /// When submission to the GPU occurred.
    submit_time: Option<Instant>,
    /// Duration from submission to completion.
    gpu_duration: Option<Duration>,
    /// Opaque payload attached by the caller during encoding.
    payload_size: usize,
}

impl CommandSlot {
    fn new() -> Self {
        Self {
            state: SlotState::Idle,
            frame_index: 0,
            encode_start: None,
            encode_duration: None,
            submit_time: None,
            gpu_duration: None,
            payload_size: 0,
        }
    }

    /// Current state of this slot.
    #[must_use]
    pub fn state(&self) -> SlotState {
        self.state
    }

    /// Frame index assigned to this slot.
    #[must_use]
    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    /// Duration of the last encode phase, if available.
    #[must_use]
    pub fn encode_duration(&self) -> Option<Duration> {
        self.encode_duration
    }

    /// Duration of the last GPU execution phase, if available.
    #[must_use]
    pub fn gpu_duration(&self) -> Option<Duration> {
        self.gpu_duration
    }

    /// Size of the payload attached during encoding.
    #[must_use]
    pub fn payload_size(&self) -> usize {
        self.payload_size
    }
}

// ─── Statistics ─────────────────────────────────────────────────────────────

/// Performance statistics for the double-buffer submitter.
#[derive(Debug, Clone, Default)]
pub struct DoubleBufferStats {
    /// Total frames submitted.
    pub total_frames: u64,
    /// Total frames completed.
    pub completed_frames: u64,
    /// Total frames that failed during GPU execution.
    pub failed_frames: u64,
    /// Total CPU encoding time across all frames.
    pub total_encode_time: Duration,
    /// Total GPU execution time across all frames.
    pub total_gpu_time: Duration,
    /// Number of times the CPU had to stall waiting for a free slot.
    pub cpu_stalls: u64,
    /// Rolling window of recent frame latencies (encode + GPU).
    recent_latencies: VecDeque<Duration>,
}

impl DoubleBufferStats {
    const MAX_RECENT: usize = 64;

    fn record_latency(&mut self, latency: Duration) {
        if self.recent_latencies.len() >= Self::MAX_RECENT {
            self.recent_latencies.pop_front();
        }
        self.recent_latencies.push_back(latency);
    }

    /// Average frame latency over the recent window.
    #[must_use]
    pub fn avg_latency(&self) -> Duration {
        if self.recent_latencies.is_empty() {
            return Duration::ZERO;
        }
        let total: Duration = self.recent_latencies.iter().sum();
        total / self.recent_latencies.len() as u32
    }

    /// Maximum frame latency in the recent window.
    #[must_use]
    pub fn max_latency(&self) -> Duration {
        self.recent_latencies
            .iter()
            .max()
            .copied()
            .unwrap_or(Duration::ZERO)
    }

    /// Minimum frame latency in the recent window.
    #[must_use]
    pub fn min_latency(&self) -> Duration {
        self.recent_latencies
            .iter()
            .min()
            .copied()
            .unwrap_or(Duration::ZERO)
    }

    /// Estimated CPU utilisation (encoding time / total time).
    #[must_use]
    pub fn cpu_utilisation(&self) -> f64 {
        let total = self.total_encode_time + self.total_gpu_time;
        if total.is_zero() {
            return 0.0;
        }
        self.total_encode_time.as_secs_f64() / total.as_secs_f64()
    }

    /// Estimated GPU utilisation (GPU time / total time).
    #[must_use]
    pub fn gpu_utilisation(&self) -> f64 {
        let total = self.total_encode_time + self.total_gpu_time;
        if total.is_zero() {
            return 0.0;
        }
        self.total_gpu_time.as_secs_f64() / total.as_secs_f64()
    }

    /// Estimated throughput in frames per second (based on recent window).
    #[must_use]
    pub fn estimated_fps(&self) -> f64 {
        let avg = self.avg_latency();
        if avg.is_zero() {
            return 0.0;
        }
        1.0 / avg.as_secs_f64()
    }
}

// ─── Error ──────────────────────────────────────────────────────────────────

/// Errors from the double-buffer submitter.
#[derive(Debug, Clone)]
pub enum DoubleBufferError {
    /// No idle slot is available (both are in-flight or encoding).
    NoFreeSlot,
    /// Attempted to submit a slot that is not in the Encoding state.
    InvalidSlotState {
        expected: SlotState,
        actual: SlotState,
    },
    /// The GPU work closure returned an error.
    GpuWorkFailed(String),
}

impl std::fmt::Display for DoubleBufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoFreeSlot => write!(f, "no free command slot available"),
            Self::InvalidSlotState { expected, actual } => {
                write!(f, "slot in state {actual}, expected {expected}")
            }
            Self::GpuWorkFailed(msg) => write!(f, "GPU work failed: {msg}"),
        }
    }
}

impl std::error::Error for DoubleBufferError {}

type Result<T> = std::result::Result<T, DoubleBufferError>;

// ─── DoubleBufferSubmitter ──────────────────────────────────────────────────

/// Double-buffered command submission manager.
///
/// Manages two [`CommandSlot`]s in a ping-pong fashion to enable overlapping
/// CPU encoding and GPU execution.
pub struct DoubleBufferSubmitter {
    slots: [CommandSlot; 2],
    /// Index of the slot currently being used for encoding (0 or 1).
    active_slot: usize,
    /// Monotonically increasing frame counter.
    frame_counter: u64,
    /// Accumulated statistics.
    stats: DoubleBufferStats,
}

impl DoubleBufferSubmitter {
    /// Create a new double-buffer submitter.
    #[must_use]
    pub fn new() -> Self {
        Self {
            slots: [CommandSlot::new(), CommandSlot::new()],
            active_slot: 0,
            frame_counter: 0,
            stats: DoubleBufferStats::default(),
        }
    }

    /// Begin encoding a new frame in the next available slot.
    ///
    /// Returns the slot index (0 or 1) that was acquired.
    ///
    /// # Errors
    ///
    /// Returns [`DoubleBufferError::NoFreeSlot`] if both slots are busy.
    pub fn begin_encode(&mut self) -> Result<usize> {
        // Find an idle or complete slot.
        let slot_idx = self.find_free_slot()?;
        let slot = &mut self.slots[slot_idx];
        slot.state = SlotState::Encoding;
        slot.frame_index = self.frame_counter;
        slot.encode_start = Some(Instant::now());
        slot.encode_duration = None;
        slot.submit_time = None;
        slot.gpu_duration = None;
        slot.payload_size = 0;
        self.active_slot = slot_idx;
        self.frame_counter += 1;
        Ok(slot_idx)
    }

    /// Set the payload size for the current encoding slot.
    ///
    /// This is informational and can be used for statistics.
    pub fn set_payload_size(&mut self, slot_idx: usize, size: usize) {
        if slot_idx < 2 {
            self.slots[slot_idx].payload_size = size;
        }
    }

    /// Finish encoding and submit the slot for GPU execution.
    ///
    /// The `gpu_work` closure receives the slot index and should perform
    /// the actual GPU command submission.
    ///
    /// # Errors
    ///
    /// Returns an error if the slot is not in the Encoding state or if
    /// `gpu_work` fails.
    pub fn submit<F>(&mut self, slot_idx: usize, gpu_work: F) -> Result<()>
    where
        F: FnOnce(usize) -> std::result::Result<(), String>,
    {
        if slot_idx >= 2 {
            return Err(DoubleBufferError::InvalidSlotState {
                expected: SlotState::Encoding,
                actual: SlotState::Idle,
            });
        }
        let slot = &mut self.slots[slot_idx];
        if slot.state != SlotState::Encoding {
            return Err(DoubleBufferError::InvalidSlotState {
                expected: SlotState::Encoding,
                actual: slot.state,
            });
        }

        // Record encode duration.
        if let Some(start) = slot.encode_start {
            let dur = start.elapsed();
            slot.encode_duration = Some(dur);
            self.stats.total_encode_time += dur;
        }

        slot.submit_time = Some(Instant::now());
        slot.state = SlotState::InFlight;
        self.stats.total_frames += 1;

        // Execute the GPU work.
        if let Err(msg) = gpu_work(slot_idx) {
            slot.state = SlotState::Idle;
            self.stats.failed_frames += 1;
            return Err(DoubleBufferError::GpuWorkFailed(msg));
        }

        Ok(())
    }

    /// Mark a slot as complete (GPU execution finished).
    ///
    /// Should be called when the GPU fence/completion signal fires.
    pub fn mark_complete(&mut self, slot_idx: usize) {
        if slot_idx >= 2 {
            return;
        }
        let slot = &mut self.slots[slot_idx];
        if slot.state != SlotState::InFlight {
            return;
        }

        if let Some(submit) = slot.submit_time {
            let gpu_dur = submit.elapsed();
            slot.gpu_duration = Some(gpu_dur);
            self.stats.total_gpu_time += gpu_dur;

            // Total latency = encode + GPU
            let total = slot.encode_duration.unwrap_or(Duration::ZERO) + gpu_dur;
            self.stats.record_latency(total);
        }

        slot.state = SlotState::Complete;
        self.stats.completed_frames += 1;
    }

    /// Convenience: encode, submit, and immediately mark complete.
    ///
    /// Useful for synchronous (non-pipelined) operation.
    ///
    /// # Errors
    ///
    /// Returns an error if no slot is free or if GPU work fails.
    pub fn submit_sync<F>(&mut self, payload_size: usize, gpu_work: F) -> Result<u64>
    where
        F: FnOnce(usize) -> std::result::Result<(), String>,
    {
        let slot_idx = self.begin_encode()?;
        self.set_payload_size(slot_idx, payload_size);
        self.submit(slot_idx, gpu_work)?;
        self.mark_complete(slot_idx);
        Ok(self.slots[slot_idx].frame_index)
    }

    /// Get a snapshot of the current statistics.
    #[must_use]
    pub fn stats(&self) -> &DoubleBufferStats {
        &self.stats
    }

    /// Get the state of a specific slot.
    #[must_use]
    pub fn slot(&self, idx: usize) -> Option<&CommandSlot> {
        self.slots.get(idx)
    }

    /// Get the current active (encoding) slot index.
    #[must_use]
    pub fn active_slot(&self) -> usize {
        self.active_slot
    }

    /// The total number of frames that have been assigned frame indices.
    #[must_use]
    pub fn frame_counter(&self) -> u64 {
        self.frame_counter
    }

    /// Reset both slots to idle and clear statistics.
    pub fn reset(&mut self) {
        self.slots = [CommandSlot::new(), CommandSlot::new()];
        self.active_slot = 0;
        self.frame_counter = 0;
        self.stats = DoubleBufferStats::default();
    }

    /// Force-reclaim a complete or idle slot (for error recovery).
    pub fn reclaim(&mut self, slot_idx: usize) {
        if slot_idx < 2 {
            self.slots[slot_idx].state = SlotState::Idle;
        }
    }

    // ── Private ─────────────────────────────────────────────────────────────

    fn find_free_slot(&mut self) -> Result<usize> {
        // Prefer the slot that is not the active one.
        let other = 1 - self.active_slot;

        // Check other slot first.
        if self.slots[other].state == SlotState::Idle
            || self.slots[other].state == SlotState::Complete
        {
            return Ok(other);
        }

        // Check active slot.
        if self.slots[self.active_slot].state == SlotState::Idle
            || self.slots[self.active_slot].state == SlotState::Complete
        {
            return Ok(self.active_slot);
        }

        self.stats.cpu_stalls += 1;
        Err(DoubleBufferError::NoFreeSlot)
    }
}

impl Default for DoubleBufferSubmitter {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_submitter_is_idle() {
        let sub = DoubleBufferSubmitter::new();
        assert_eq!(sub.slot(0).map(|s| s.state()), Some(SlotState::Idle));
        assert_eq!(sub.slot(1).map(|s| s.state()), Some(SlotState::Idle));
        assert_eq!(sub.frame_counter(), 0);
    }

    #[test]
    fn test_begin_encode_transitions_to_encoding() {
        let mut sub = DoubleBufferSubmitter::new();
        let idx = sub.begin_encode().expect("begin encode");
        assert_eq!(sub.slot(idx).map(|s| s.state()), Some(SlotState::Encoding));
    }

    #[test]
    fn test_submit_transitions_to_in_flight() {
        let mut sub = DoubleBufferSubmitter::new();
        let idx = sub.begin_encode().expect("begin encode");
        sub.submit(idx, |_| Ok(())).expect("submit");
        assert_eq!(sub.slot(idx).map(|s| s.state()), Some(SlotState::InFlight));
    }

    #[test]
    fn test_mark_complete_transitions() {
        let mut sub = DoubleBufferSubmitter::new();
        let idx = sub.begin_encode().expect("begin encode");
        sub.submit(idx, |_| Ok(())).expect("submit");
        sub.mark_complete(idx);
        assert_eq!(sub.slot(idx).map(|s| s.state()), Some(SlotState::Complete));
    }

    #[test]
    fn test_submit_sync_full_cycle() {
        let mut sub = DoubleBufferSubmitter::new();
        let frame_id = sub.submit_sync(1024, |_| Ok(())).expect("sync submit");
        assert_eq!(frame_id, 0);
        assert_eq!(sub.stats().total_frames, 1);
        assert_eq!(sub.stats().completed_frames, 1);
    }

    #[test]
    fn test_double_buffer_alternates_slots() {
        let mut sub = DoubleBufferSubmitter::new();
        let idx0 = sub.begin_encode().expect("encode 0");
        sub.submit(idx0, |_| Ok(())).expect("submit 0");
        sub.mark_complete(idx0);

        let idx1 = sub.begin_encode().expect("encode 1");
        assert_ne!(idx0, idx1, "should alternate to the other slot");
        sub.submit(idx1, |_| Ok(())).expect("submit 1");
        sub.mark_complete(idx1);

        assert_eq!(sub.stats().total_frames, 2);
        assert_eq!(sub.stats().completed_frames, 2);
    }

    #[test]
    fn test_no_free_slot_when_both_in_flight() {
        let mut sub = DoubleBufferSubmitter::new();
        let idx0 = sub.begin_encode().expect("encode 0");
        sub.submit(idx0, |_| Ok(())).expect("submit 0");
        // idx0 is InFlight

        let idx1 = sub.begin_encode().expect("encode 1");
        sub.submit(idx1, |_| Ok(())).expect("submit 1");
        // idx1 is InFlight

        // Both slots busy
        let result = sub.begin_encode();
        assert!(result.is_err());
        assert_eq!(sub.stats().cpu_stalls, 1);
    }

    #[test]
    fn test_gpu_work_failure_records_stats() {
        let mut sub = DoubleBufferSubmitter::new();
        let idx = sub.begin_encode().expect("encode");
        let result = sub.submit(idx, |_| Err("simulated failure".to_string()));
        assert!(result.is_err());
        assert_eq!(sub.stats().failed_frames, 1);
        // Slot should be reset to idle on failure
        assert_eq!(sub.slot(idx).map(|s| s.state()), Some(SlotState::Idle));
    }

    #[test]
    fn test_submit_wrong_state_returns_error() {
        let mut sub = DoubleBufferSubmitter::new();
        // Slot 0 is idle, not encoding — submit should fail.
        let result = sub.submit(0, |_| Ok(()));
        assert!(result.is_err());
    }

    #[test]
    fn test_payload_size_tracking() {
        let mut sub = DoubleBufferSubmitter::new();
        let idx = sub.begin_encode().expect("encode");
        sub.set_payload_size(idx, 4096);
        assert_eq!(sub.slot(idx).map(|s| s.payload_size()), Some(4096));
    }

    #[test]
    fn test_reset_clears_everything() {
        let mut sub = DoubleBufferSubmitter::new();
        let _ = sub.submit_sync(100, |_| Ok(()));
        let _ = sub.submit_sync(200, |_| Ok(()));
        sub.reset();
        assert_eq!(sub.frame_counter(), 0);
        assert_eq!(sub.stats().total_frames, 0);
        assert_eq!(sub.slot(0).map(|s| s.state()), Some(SlotState::Idle));
        assert_eq!(sub.slot(1).map(|s| s.state()), Some(SlotState::Idle));
    }

    #[test]
    fn test_reclaim_resets_slot() {
        let mut sub = DoubleBufferSubmitter::new();
        let idx = sub.begin_encode().expect("encode");
        sub.submit(idx, |_| Ok(())).expect("submit");
        assert_eq!(sub.slot(idx).map(|s| s.state()), Some(SlotState::InFlight));
        sub.reclaim(idx);
        assert_eq!(sub.slot(idx).map(|s| s.state()), Some(SlotState::Idle));
    }

    #[test]
    fn test_frame_index_increments() {
        let mut sub = DoubleBufferSubmitter::new();
        let f0 = sub.submit_sync(10, |_| Ok(())).expect("f0");
        let f1 = sub.submit_sync(10, |_| Ok(())).expect("f1");
        let f2 = sub.submit_sync(10, |_| Ok(())).expect("f2");
        assert_eq!(f0, 0);
        assert_eq!(f1, 1);
        assert_eq!(f2, 2);
    }

    #[test]
    fn test_stats_avg_latency_not_zero_after_frames() {
        let mut sub = DoubleBufferSubmitter::new();
        for _ in 0..5 {
            let _ = sub.submit_sync(100, |_| Ok(()));
        }
        // Latencies might be very small but should be recorded
        assert_eq!(sub.stats().completed_frames, 5);
    }

    #[test]
    fn test_stats_utilisation_bounded() {
        let mut sub = DoubleBufferSubmitter::new();
        for _ in 0..10 {
            let _ = sub.submit_sync(100, |_| Ok(()));
        }
        let cpu_u = sub.stats().cpu_utilisation();
        let gpu_u = sub.stats().gpu_utilisation();
        assert!(
            cpu_u >= 0.0 && cpu_u <= 1.0,
            "cpu utilisation out of range: {cpu_u}"
        );
        assert!(
            gpu_u >= 0.0 && gpu_u <= 1.0,
            "gpu utilisation out of range: {gpu_u}"
        );
    }

    #[test]
    fn test_slot_out_of_range() {
        let sub = DoubleBufferSubmitter::new();
        assert!(sub.slot(2).is_none());
        assert!(sub.slot(99).is_none());
    }

    #[test]
    fn test_mark_complete_ignored_for_non_inflight() {
        let mut sub = DoubleBufferSubmitter::new();
        // Slot 0 is Idle — mark_complete should be a no-op
        sub.mark_complete(0);
        assert_eq!(sub.slot(0).map(|s| s.state()), Some(SlotState::Idle));
        assert_eq!(sub.stats().completed_frames, 0);
    }

    #[test]
    fn test_default_impl() {
        let sub = DoubleBufferSubmitter::default();
        assert_eq!(sub.frame_counter(), 0);
    }

    #[test]
    fn test_slot_state_display() {
        assert_eq!(format!("{}", SlotState::Idle), "idle");
        assert_eq!(format!("{}", SlotState::Encoding), "encoding");
        assert_eq!(format!("{}", SlotState::InFlight), "in-flight");
        assert_eq!(format!("{}", SlotState::Complete), "complete");
    }

    #[test]
    fn test_max_min_latency_with_empty_stats() {
        let stats = DoubleBufferStats::default();
        assert_eq!(stats.max_latency(), Duration::ZERO);
        assert_eq!(stats.min_latency(), Duration::ZERO);
        assert_eq!(stats.avg_latency(), Duration::ZERO);
    }

    #[test]
    fn test_estimated_fps_zero_when_no_frames() {
        let stats = DoubleBufferStats::default();
        assert_eq!(stats.estimated_fps(), 0.0);
    }
}
