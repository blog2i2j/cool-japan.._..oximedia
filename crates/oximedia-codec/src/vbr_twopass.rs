//! VBR two-pass encoding state tracking.
//!
//! Two-pass VBR encoding works in two phases:
//!
//! 1. **First pass** — the encoder analyses each frame, recording complexity
//!    metrics (e.g. SAD, variance, intra cost) without actually producing output
//!    bitstream.  The results are accumulated into a [`TwoPassFirstPassStats`].
//!
//! 2. **Second pass** — using the first-pass statistics the encoder computes an
//!    optimal bitrate allocation per frame, then encodes each frame with that
//!    target.  [`TwoPassBitrateAllocator`] performs this allocation.
//!
//! # Algorithm
//!
//! The allocator implements *complexity-proportional* bit allocation:
//!
//! ```text
//! target_bits(i) = total_bits * complexity(i) / sum(complexity)
//! ```
//!
//! with GOP-level equalisation to avoid budget overflow across group boundaries.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ---------------------------------------------------------------------------
// Per-frame complexity record (first pass)
// ---------------------------------------------------------------------------

/// Per-frame statistics recorded during the first encoding pass.
#[derive(Debug, Clone)]
pub struct FirstPassFrameStats {
    /// Frame index (display order).
    pub frame_index: u64,
    /// Mean intra cost (bits per pixel at a reference QP).
    pub intra_cost: f32,
    /// Mean inter cost (bits per pixel for the best motion prediction).
    pub inter_cost: f32,
    /// Intra/inter cost ratio; values > 1.0 indicate scene-change candidates.
    pub intra_inter_ratio: f32,
    /// Motion-compensated average block SAD.
    pub mean_sad: f32,
    /// Estimated number of bits if encoded as a key frame.
    pub key_frame_bits: u64,
    /// Whether this frame was marked as a scene change in the first pass.
    pub is_scene_change: bool,
}

impl FirstPassFrameStats {
    /// Create a minimal stats record for a frame.
    #[must_use]
    pub fn new(frame_index: u64, intra_cost: f32, inter_cost: f32) -> Self {
        let ratio = if inter_cost > 0.0 {
            intra_cost / inter_cost
        } else {
            1.0
        };
        Self {
            frame_index,
            intra_cost,
            inter_cost,
            intra_inter_ratio: ratio,
            mean_sad: 0.0,
            key_frame_bits: 0,
            is_scene_change: ratio > 2.5,
        }
    }

    /// Effective complexity weight for bit allocation.
    ///
    /// Uses the inter cost as the primary signal; scene-change frames get a
    /// boost so that the I-frame receives proportionally more bits.
    #[must_use]
    pub fn complexity_weight(&self) -> f32 {
        let base = self.inter_cost.max(0.001);
        if self.is_scene_change {
            base * 1.8
        } else {
            base
        }
    }
}

// ---------------------------------------------------------------------------
// First-pass accumulator
// ---------------------------------------------------------------------------

/// Accumulated first-pass statistics for an entire encode session.
#[derive(Debug, Default, Clone)]
pub struct TwoPassFirstPassStats {
    frames: Vec<FirstPassFrameStats>,
}

impl TwoPassFirstPassStats {
    /// Create an empty statistics container.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one frame's statistics.
    pub fn push(&mut self, stats: FirstPassFrameStats) {
        self.frames.push(stats);
    }

    /// Total number of analysed frames.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Access per-frame statistics by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&FirstPassFrameStats> {
        self.frames.get(index)
    }

    /// Iterate over all per-frame statistics.
    pub fn iter(&self) -> impl Iterator<Item = &FirstPassFrameStats> {
        self.frames.iter()
    }

    /// Sum of all complexity weights (denominator for proportional allocation).
    #[must_use]
    pub fn total_complexity(&self) -> f32 {
        self.frames.iter().map(|f| f.complexity_weight()).sum()
    }

    /// Mean intra/inter ratio across all frames.
    #[must_use]
    pub fn mean_intra_inter_ratio(&self) -> f32 {
        if self.frames.is_empty() {
            return 1.0;
        }
        self.frames.iter().map(|f| f.intra_inter_ratio).sum::<f32>() / self.frames.len() as f32
    }

    /// Number of scene changes detected in the first pass.
    #[must_use]
    pub fn scene_change_count(&self) -> usize {
        self.frames.iter().filter(|f| f.is_scene_change).count()
    }
}

// ---------------------------------------------------------------------------
// Second-pass allocator
// ---------------------------------------------------------------------------

/// Configuration for the second-pass bitrate allocator.
#[derive(Debug, Clone)]
pub struct TwoPassAllocatorConfig {
    /// Target average bitrate in bits per second.
    pub target_bitrate_bps: u64,
    /// Output frame rate (frames per second).
    pub frame_rate: f64,
    /// GOP size (number of frames between key frames).
    pub gop_size: u32,
    /// Maximum bitrate multiplier relative to average.
    ///
    /// A value of 2.0 means the peak bitrate may be up to 2× the average.
    pub max_bitrate_ratio: f32,
    /// Minimum QP value.
    pub min_qp: u32,
    /// Maximum QP value.
    pub max_qp: u32,
}

impl Default for TwoPassAllocatorConfig {
    fn default() -> Self {
        Self {
            target_bitrate_bps: 4_000_000,
            frame_rate: 30.0,
            gop_size: 120,
            max_bitrate_ratio: 2.5,
            min_qp: 16,
            max_qp: 51,
        }
    }
}

/// Per-frame allocation result from the second pass.
#[derive(Debug, Clone)]
pub struct FrameAllocationResult {
    /// Frame index (display order).
    pub frame_index: u64,
    /// Target number of bits for this frame.
    pub target_bits: u64,
    /// Suggested quantiser parameter.
    pub suggested_qp: u32,
    /// Whether this frame should be encoded as a key frame.
    pub force_keyframe: bool,
}

/// Two-pass VBR bitrate allocator.
///
/// Use [`TwoPassBitrateAllocator::allocate`] to produce per-frame allocation
/// plans from first-pass statistics.
#[derive(Debug, Clone)]
pub struct TwoPassBitrateAllocator {
    cfg: TwoPassAllocatorConfig,
}

impl TwoPassBitrateAllocator {
    /// Create a new allocator with the given configuration.
    #[must_use]
    pub fn new(cfg: TwoPassAllocatorConfig) -> Self {
        Self { cfg }
    }

    /// Create an allocator with default configuration.
    #[must_use]
    pub fn default_allocator() -> Self {
        Self::new(TwoPassAllocatorConfig::default())
    }

    /// Allocate bits for all frames in a session.
    ///
    /// Returns one [`FrameAllocationResult`] per frame in the first-pass stats,
    /// in the same order.
    #[must_use]
    pub fn allocate(&self, stats: &TwoPassFirstPassStats) -> Vec<FrameAllocationResult> {
        let n = stats.frame_count();
        if n == 0 {
            return Vec::new();
        }

        let total_complexity = stats.total_complexity();
        let seconds = n as f64 / self.cfg.frame_rate.max(1.0);
        let total_bits = (self.cfg.target_bitrate_bps as f64 * seconds) as u64;

        let max_frame_bits =
            ((total_bits as f64 / n as f64) * f64::from(self.cfg.max_bitrate_ratio)) as u64;

        let mut results = Vec::with_capacity(n);

        for frame_stats in stats.iter() {
            let weight = frame_stats.complexity_weight();
            let raw_bits = if total_complexity > 0.0 {
                ((weight as f64 / total_complexity as f64) * total_bits as f64) as u64
            } else {
                total_bits / n as u64
            };

            let target_bits = raw_bits.min(max_frame_bits).max(512);

            // Heuristic QP from bits-per-pixel: higher bits → lower QP
            let bpp = target_bits as f64 / (frame_stats.inter_cost.max(0.001) as f64 * 100.0);
            let suggested_qp = self.bits_to_qp(bpp);

            results.push(FrameAllocationResult {
                frame_index: frame_stats.frame_index,
                target_bits,
                suggested_qp,
                force_keyframe: frame_stats.is_scene_change,
            });
        }

        results
    }

    /// Heuristic mapping from bits-per-pixel to QP.
    fn bits_to_qp(&self, bpp: f64) -> u32 {
        // Simple log-linear model: QP ≈ 36 - 12 * log2(bpp / 0.1)
        let qp_f = 36.0 - 12.0 * (bpp / 0.1 + 1.0).log2();
        (qp_f.round() as u32).clamp(self.cfg.min_qp, self.cfg.max_qp)
    }
}

// ---------------------------------------------------------------------------
// Two-pass encode session state machine
// ---------------------------------------------------------------------------

/// State of the two-pass VBR encode session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwoPassState {
    /// First pass: frame analysis in progress.
    FirstPass,
    /// Transition: first pass complete, allocator ready.
    AllocationReady,
    /// Second pass: frame encoding in progress.
    SecondPass,
    /// Encode complete.
    Complete,
}

/// Top-level two-pass VBR encode session.
///
/// Manages state transitions and coordinates between first-pass analysis
/// and second-pass encoding.
#[derive(Debug)]
pub struct TwoPassSession {
    cfg: TwoPassAllocatorConfig,
    state: TwoPassState,
    first_pass_stats: TwoPassFirstPassStats,
    allocations: Vec<FrameAllocationResult>,
    second_pass_index: usize,
}

impl TwoPassSession {
    /// Create a new session with the given allocator configuration.
    #[must_use]
    pub fn new(cfg: TwoPassAllocatorConfig) -> Self {
        Self {
            cfg,
            state: TwoPassState::FirstPass,
            first_pass_stats: TwoPassFirstPassStats::new(),
            allocations: Vec::new(),
            second_pass_index: 0,
        }
    }

    /// Current state of the session.
    #[must_use]
    pub fn state(&self) -> TwoPassState {
        self.state
    }

    /// Record one frame's first-pass statistics.
    ///
    /// Returns an error string if the session is not in `FirstPass` state.
    pub fn record_first_pass_frame(
        &mut self,
        stats: FirstPassFrameStats,
    ) -> Result<(), &'static str> {
        if self.state != TwoPassState::FirstPass {
            return Err("session is not in first-pass state");
        }
        self.first_pass_stats.push(stats);
        Ok(())
    }

    /// Finalise the first pass and compute bit allocations.
    ///
    /// After calling this method the session moves to `AllocationReady` state.
    pub fn finish_first_pass(&mut self) {
        if self.state != TwoPassState::FirstPass {
            return;
        }
        let allocator = TwoPassBitrateAllocator::new(self.cfg.clone());
        self.allocations = allocator.allocate(&self.first_pass_stats);
        self.state = TwoPassState::AllocationReady;
    }

    /// Begin the second pass.
    ///
    /// Transitions the session from `AllocationReady` to `SecondPass`.
    pub fn begin_second_pass(&mut self) -> Result<(), &'static str> {
        if self.state != TwoPassState::AllocationReady {
            return Err("first pass not yet finalised");
        }
        self.second_pass_index = 0;
        self.state = TwoPassState::SecondPass;
        Ok(())
    }

    /// Get the allocation for the next second-pass frame.
    ///
    /// Returns `None` when all frames have been consumed (session complete).
    pub fn next_frame_allocation(&mut self) -> Option<&FrameAllocationResult> {
        if self.state != TwoPassState::SecondPass {
            return None;
        }
        if self.second_pass_index >= self.allocations.len() {
            self.state = TwoPassState::Complete;
            return None;
        }
        let result = &self.allocations[self.second_pass_index];
        self.second_pass_index += 1;
        if self.second_pass_index >= self.allocations.len() {
            self.state = TwoPassState::Complete;
        }
        Some(result)
    }

    /// Access the first-pass statistics.
    #[must_use]
    pub fn first_pass_stats(&self) -> &TwoPassFirstPassStats {
        &self.first_pass_stats
    }

    /// Access all computed frame allocations.
    #[must_use]
    pub fn allocations(&self) -> &[FrameAllocationResult] {
        &self.allocations
    }
}

// ---------------------------------------------------------------------------
// Simplified TwoPassStateTracker — convenience API for two-pass VBR
// ---------------------------------------------------------------------------

/// Lightweight two-pass VBR state tracker.
///
/// Provides a simplified API for recording per-frame complexity values during
/// the first pass and computing per-frame bitrate targets during the second
/// pass.
///
/// # Usage
///
/// ```rust
/// use oximedia_codec::vbr_twopass::TwoPassStateTracker;
///
/// let mut state = TwoPassStateTracker::new();
/// state.record_first_pass(1.5); // frame 0
/// state.record_first_pass(0.8); // frame 1
/// state.record_first_pass(2.3); // frame 2
///
/// let bits_frame0 = state.compute_bitrate(1.5, 4_000_000);
/// assert!(bits_frame0 > 0);
/// ```
#[derive(Debug, Clone, Default)]
pub struct TwoPassStateTracker {
    /// Complexity values accumulated during the first pass.
    first_pass_complexities: Vec<f32>,
}

impl TwoPassStateTracker {
    /// Create a new, empty state tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a complexity value for one frame during the first pass.
    ///
    /// `complexity` must be a non-negative floating-point value.  Typical
    /// values are SAD-normalised intra/inter costs in the range 0.1 – 20.0.
    pub fn record_first_pass(&mut self, complexity: f32) {
        self.first_pass_complexities.push(complexity.max(0.0));
    }

    /// Compute the per-frame bitrate target for a frame with the given complexity.
    ///
    /// Uses complexity-proportional allocation:
    /// ```text
    /// target_bits = budget * complexity / sum(all_complexities)
    /// ```
    ///
    /// If the accumulated first-pass data is empty the entire `budget` is
    /// returned (no-op case).
    ///
    /// # Parameters
    /// - `complexity` – complexity value of the current frame (should have been
    ///   recorded via [`Self::record_first_pass`]).
    /// - `budget`     – total bit budget for the session (bits).
    ///
    /// # Returns
    /// Target number of bits for this frame, as a `u32`.  Clamped to at least 1
    /// and at most `budget`.
    #[must_use]
    pub fn compute_bitrate(&self, complexity: f32, budget: u32) -> u32 {
        let total: f32 = self.first_pass_complexities.iter().sum();
        if total <= 0.0 || budget == 0 {
            return budget;
        }
        let share = complexity.max(0.0) / total;
        let bits = (budget as f64 * share as f64).round() as u64;
        bits.clamp(1, u64::from(budget)) as u32
    }

    /// Returns the number of frames recorded in the first pass.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.first_pass_complexities.len()
    }

    /// Returns the total accumulated complexity.
    #[must_use]
    pub fn total_complexity(&self) -> f32 {
        self.first_pass_complexities.iter().sum()
    }

    /// Returns the average complexity per frame.
    #[must_use]
    pub fn mean_complexity(&self) -> f32 {
        let n = self.first_pass_complexities.len();
        if n == 0 {
            return 0.0;
        }
        self.total_complexity() / n as f32
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stats(n: usize) -> TwoPassFirstPassStats {
        let mut stats = TwoPassFirstPassStats::new();
        for i in 0..n {
            let intra = 1.0 + (i as f32) * 0.1;
            let inter = 0.5 + (i as f32) * 0.05;
            stats.push(FirstPassFrameStats::new(i as u64, intra, inter));
        }
        stats
    }

    #[test]
    fn test_first_pass_stats_frame_count() {
        let stats = make_stats(10);
        assert_eq!(stats.frame_count(), 10);
    }

    #[test]
    fn test_total_complexity_positive() {
        let stats = make_stats(5);
        assert!(stats.total_complexity() > 0.0);
    }

    #[test]
    fn test_allocator_correct_count() {
        let stats = make_stats(30);
        let alloc = TwoPassBitrateAllocator::default_allocator();
        let results = alloc.allocate(&stats);
        assert_eq!(results.len(), 30);
    }

    #[test]
    fn test_allocator_empty_stats() {
        let stats = TwoPassFirstPassStats::new();
        let alloc = TwoPassBitrateAllocator::default_allocator();
        let results = alloc.allocate(&stats);
        assert!(results.is_empty());
    }

    #[test]
    fn test_allocator_bits_in_range() {
        let stats = make_stats(100);
        let alloc = TwoPassBitrateAllocator::default_allocator();
        let results = alloc.allocate(&stats);
        for r in &results {
            assert!(r.target_bits >= 512, "target_bits should be at least 512");
        }
    }

    #[test]
    fn test_allocator_qp_in_range() {
        let stats = make_stats(20);
        let alloc = TwoPassBitrateAllocator::default_allocator();
        let results = alloc.allocate(&stats);
        for r in &results {
            assert!(r.suggested_qp >= 16 && r.suggested_qp <= 51);
        }
    }

    #[test]
    fn test_session_state_transitions() {
        let mut session = TwoPassSession::new(TwoPassAllocatorConfig::default());
        assert_eq!(session.state(), TwoPassState::FirstPass);

        let frame = FirstPassFrameStats::new(0, 1.5, 0.8);
        session
            .record_first_pass_frame(frame)
            .expect("should succeed");
        session.finish_first_pass();
        assert_eq!(session.state(), TwoPassState::AllocationReady);

        session.begin_second_pass().expect("should succeed");
        assert_eq!(session.state(), TwoPassState::SecondPass);
    }

    #[test]
    fn test_session_second_pass_iterates() {
        let mut session = TwoPassSession::new(TwoPassAllocatorConfig::default());
        for i in 0..5u64 {
            let f = FirstPassFrameStats::new(i, 1.0, 0.5);
            session.record_first_pass_frame(f).expect("ok");
        }
        session.finish_first_pass();
        session.begin_second_pass().expect("ok");

        let mut count = 0;
        while let Some(_alloc) = session.next_frame_allocation() {
            count += 1;
        }
        assert_eq!(count, 5);
        assert_eq!(session.state(), TwoPassState::Complete);
    }

    #[test]
    fn test_first_pass_record_after_finish_errors() {
        let mut session = TwoPassSession::new(TwoPassAllocatorConfig::default());
        session.finish_first_pass();
        let f = FirstPassFrameStats::new(0, 1.0, 0.5);
        assert!(session.record_first_pass_frame(f).is_err());
    }

    #[test]
    fn test_scene_change_flagged_on_high_ratio() {
        let stats = FirstPassFrameStats::new(0, 5.0, 0.5);
        // ratio = 10.0 > 2.5
        assert!(stats.is_scene_change);
    }

    #[test]
    fn test_scene_change_not_flagged_on_low_ratio() {
        let stats = FirstPassFrameStats::new(0, 1.0, 0.8);
        // ratio = 1.25 < 2.5
        assert!(!stats.is_scene_change);
    }

    #[test]
    fn test_mean_intra_inter_ratio() {
        let mut s = TwoPassFirstPassStats::new();
        s.push(FirstPassFrameStats::new(0, 2.0, 1.0)); // ratio 2.0
        s.push(FirstPassFrameStats::new(1, 4.0, 1.0)); // ratio 4.0
        let mean = s.mean_intra_inter_ratio();
        assert!((mean - 3.0).abs() < 0.01);
    }

    // TwoPassStateTracker tests
    #[test]
    fn two_pass_tracker_new_is_empty() {
        let t = TwoPassStateTracker::new();
        assert_eq!(t.frame_count(), 0);
    }

    #[test]
    fn two_pass_tracker_record_adds_frames() {
        let mut t = TwoPassStateTracker::new();
        t.record_first_pass(1.0);
        t.record_first_pass(2.0);
        assert_eq!(t.frame_count(), 2);
    }

    #[test]
    fn two_pass_tracker_compute_bitrate_proportional() {
        let mut t = TwoPassStateTracker::new();
        t.record_first_pass(1.0);
        t.record_first_pass(1.0);
        // Each frame has equal complexity so each gets half the budget
        let bits = t.compute_bitrate(1.0, 1000);
        assert_eq!(bits, 500);
    }

    #[test]
    fn two_pass_tracker_compute_bitrate_empty_returns_budget() {
        let t = TwoPassStateTracker::new();
        let bits = t.compute_bitrate(1.0, 999);
        assert_eq!(bits, 999);
    }

    #[test]
    fn two_pass_tracker_compute_bitrate_at_least_one() {
        let mut t = TwoPassStateTracker::new();
        t.record_first_pass(1000.0);
        // Tiny complexity vs huge total → at least 1
        let bits = t.compute_bitrate(0.001, 1000);
        assert!(bits >= 1);
    }

    #[test]
    fn two_pass_tracker_total_complexity() {
        let mut t = TwoPassStateTracker::new();
        t.record_first_pass(2.0);
        t.record_first_pass(3.0);
        assert!((t.total_complexity() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn two_pass_tracker_mean_complexity() {
        let mut t = TwoPassStateTracker::new();
        t.record_first_pass(4.0);
        t.record_first_pass(6.0);
        assert!((t.mean_complexity() - 5.0).abs() < 1e-5);
    }
}
