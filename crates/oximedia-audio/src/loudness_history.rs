//! Rolling loudness history — per-block LUFS tracking, Loudness Range (LRA)
//! measurement, and history export.
//!
//! This module implements EBU Tech 3342 / EBU R128 Loudness Range measurement
//! on top of a configurable rolling history of momentary LUFS blocks.
//!
//! # Concepts
//!
//! * **Block loudness**: A short-term integrated LUFS value for a fixed block
//!   of audio (typically 3 s, sliding or non-overlapping).
//! * **Loudness Range (LRA)**: The difference in LU between the 10th and 95th
//!   percentile of block loudness values that pass the absolute gate
//!   (−70 LUFS) and the relative gate (10 LU below the ungated mean).
//! * **History window**: How many blocks to retain in memory.  Old blocks are
//!   evicted when the window is full.
//!
//! # Example
//!
//! ```
//! use oximedia_audio::loudness_history::{LoudnessHistoryConfig, LoudnessHistoryTracker};
//!
//! let config = LoudnessHistoryConfig::default();
//! let mut tracker = LoudnessHistoryTracker::new(config);
//!
//! // Push some block loudness values (LUFS).
//! for lufs in [-23.0_f64, -22.5, -23.5, -24.0, -21.0] {
//!     tracker.push_block(lufs);
//! }
//!
//! let lra = tracker.loudness_range();
//! println!("LRA = {:.1} LU", lra);
//!
//! let export = tracker.export_csv();
//! println!("{}", export);
//! ```

#![forbid(unsafe_code)]

use std::collections::VecDeque;

// ────────────────────────────────────────────────────────────────────────────
// Constants (EBU R128 / EBU Tech 3342)
// ────────────────────────────────────────────────────────────────────────────

/// Absolute gating threshold (LUFS).
const ABSOLUTE_GATE_LUFS: f64 = -70.0;

/// Relative gate offset from ungated mean (LU).
const RELATIVE_GATE_OFFSET_LU: f64 = 10.0;

/// Lower percentile for LRA computation (10th).
const LRA_LOW_PERCENTILE: f64 = 0.10;

/// Upper percentile for LRA computation (95th).
const LRA_HIGH_PERCENTILE: f64 = 0.95;

// ────────────────────────────────────────────────────────────────────────────
// Configuration
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for `LoudnessHistoryTracker`.
#[derive(Clone, Debug)]
pub struct LoudnessHistoryConfig {
    /// Maximum number of LUFS blocks to retain.
    ///
    /// When the ring is full the oldest block is evicted.  Set to 0 for
    /// unlimited retention (bounded only by available memory).
    pub max_blocks: usize,

    /// Block duration in seconds (used only for timestamp generation when
    /// exporting; the tracker itself is sample-agnostic).
    pub block_duration_secs: f64,
}

impl Default for LoudnessHistoryConfig {
    fn default() -> Self {
        Self {
            max_blocks: 10_000, // ~8 h at 3 s / block
            block_duration_secs: 3.0,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Block record
// ────────────────────────────────────────────────────────────────────────────

/// A single block's loudness record stored in the history.
#[derive(Clone, Debug)]
pub struct LoudnessBlock {
    /// Block sequence number (0-indexed, never resets on eviction).
    pub index: u64,
    /// Block loudness in LUFS.
    pub lufs: f64,
    /// Approximate timestamp offset from the start in seconds.
    pub timestamp_secs: f64,
}

// ────────────────────────────────────────────────────────────────────────────
// Statistics snapshot
// ────────────────────────────────────────────────────────────────────────────

/// Snapshot of loudness statistics computed from the retained history.
#[derive(Clone, Debug)]
pub struct LoudnessHistoryStats {
    /// Number of blocks currently retained.
    pub block_count: usize,
    /// Minimum LUFS across all retained blocks (ignoring −∞).
    pub min_lufs: f64,
    /// Maximum LUFS across all retained blocks.
    pub max_lufs: f64,
    /// Arithmetic mean LUFS across all retained blocks.
    pub mean_lufs: f64,
    /// Loudness range (LRA) in LU, computed per EBU Tech 3342.
    pub loudness_range_lu: f64,
    /// Number of blocks above the absolute gate (−70 LUFS).
    pub gated_block_count: usize,
}

// ────────────────────────────────────────────────────────────────────────────
// Tracker
// ────────────────────────────────────────────────────────────────────────────

/// Rolling loudness history tracker.
///
/// Feed it one block-LUFS value per analysis period and query statistics or
/// export the data at any time.
pub struct LoudnessHistoryTracker {
    config: LoudnessHistoryConfig,
    /// Ring of retained blocks (front = oldest).
    blocks: VecDeque<LoudnessBlock>,
    /// Monotonically increasing block counter (persists across evictions).
    next_index: u64,
}

impl LoudnessHistoryTracker {
    /// Create a new tracker with the given configuration.
    #[must_use]
    pub fn new(config: LoudnessHistoryConfig) -> Self {
        let cap = if config.max_blocks == 0 {
            64
        } else {
            config.max_blocks
        };
        Self {
            config,
            blocks: VecDeque::with_capacity(cap.min(4096)),
            next_index: 0,
        }
    }

    /// Push a new block loudness value in LUFS.
    ///
    /// If the history window is full the oldest block is evicted first.
    pub fn push_block(&mut self, lufs: f64) {
        let index = self.next_index;
        let timestamp_secs = index as f64 * self.config.block_duration_secs;
        self.next_index += 1;

        if self.config.max_blocks > 0
            && self.blocks.len() >= self.config.max_blocks
        {
            self.blocks.pop_front();
        }

        self.blocks.push_back(LoudnessBlock {
            index,
            lufs,
            timestamp_secs,
        });
    }

    /// Push multiple block loudness values at once.
    pub fn push_blocks(&mut self, values: &[f64]) {
        for &v in values {
            self.push_block(v);
        }
    }

    /// Return the number of blocks currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Return true when no blocks have been pushed (or all have been cleared).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Access the stored blocks in chronological order (oldest first).
    #[must_use]
    pub fn blocks(&self) -> &VecDeque<LoudnessBlock> {
        &self.blocks
    }

    /// Compute the Loudness Range (LRA) in LU per EBU Tech 3342.
    ///
    /// Returns 0.0 when there are fewer than 2 gated blocks.
    #[must_use]
    pub fn loudness_range(&self) -> f64 {
        let gated = self.gated_values();
        compute_lra(&gated)
    }

    /// Compute a statistics snapshot from the current history.
    #[must_use]
    pub fn stats(&self) -> LoudnessHistoryStats {
        if self.blocks.is_empty() {
            return LoudnessHistoryStats {
                block_count: 0,
                min_lufs: f64::NEG_INFINITY,
                max_lufs: f64::NEG_INFINITY,
                mean_lufs: f64::NEG_INFINITY,
                loudness_range_lu: 0.0,
                gated_block_count: 0,
            };
        }

        let finite_values: Vec<f64> = self
            .blocks
            .iter()
            .map(|b| b.lufs)
            .filter(|v| v.is_finite())
            .collect();

        let min_lufs = finite_values
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_lufs = finite_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mean_lufs = if finite_values.is_empty() {
            f64::NEG_INFINITY
        } else {
            finite_values.iter().sum::<f64>() / finite_values.len() as f64
        };

        let gated = self.gated_values();
        let gated_block_count = gated.len();
        let loudness_range_lu = compute_lra(&gated);

        LoudnessHistoryStats {
            block_count: self.blocks.len(),
            min_lufs,
            max_lufs,
            mean_lufs,
            loudness_range_lu,
            gated_block_count,
        }
    }

    /// Export the block history as a CSV string.
    ///
    /// Columns: `index,timestamp_secs,lufs`
    #[must_use]
    pub fn export_csv(&self) -> String {
        let mut out = String::from("index,timestamp_secs,lufs\n");
        for b in &self.blocks {
            out.push_str(&format!(
                "{},{:.3},{:.2}\n",
                b.index, b.timestamp_secs, b.lufs
            ));
        }
        out
    }

    /// Export the block loudness values as a plain `Vec<f64>` (LUFS),
    /// oldest block first.
    #[must_use]
    pub fn export_values(&self) -> Vec<f64> {
        self.blocks.iter().map(|b| b.lufs).collect()
    }

    /// Clear all retained blocks and reset the block counter.
    pub fn clear(&mut self) {
        self.blocks.clear();
        self.next_index = 0;
    }

    /// Find the index of the loudest block currently in history.
    #[must_use]
    pub fn loudest_block_index(&self) -> Option<u64> {
        self.blocks
            .iter()
            .filter(|b| b.lufs.is_finite())
            .max_by(|a, b| a.lufs.partial_cmp(&b.lufs).unwrap_or(std::cmp::Ordering::Equal))
            .map(|b| b.index)
    }

    /// Find the index of the quietest block currently in history.
    #[must_use]
    pub fn quietest_block_index(&self) -> Option<u64> {
        self.blocks
            .iter()
            .filter(|b| b.lufs.is_finite())
            .min_by(|a, b| a.lufs.partial_cmp(&b.lufs).unwrap_or(std::cmp::Ordering::Equal))
            .map(|b| b.index)
    }

    // ── private helpers ──────────────────────────────────────────────────────

    /// Collect block loudness values that pass the absolute gate.
    fn gated_values(&self) -> Vec<f64> {
        self.blocks
            .iter()
            .map(|b| b.lufs)
            .filter(|&v| v.is_finite() && v > ABSOLUTE_GATE_LUFS)
            .collect()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// LRA computation (EBU Tech 3342 §3)
// ────────────────────────────────────────────────────────────────────────────

/// Compute LRA from a slice of absolutely-gated LUFS values.
///
/// Applies the relative gate (mean − 10 LU), then returns the difference
/// between the 95th and 10th percentiles of the remaining values.
fn compute_lra(gated: &[f64]) -> f64 {
    if gated.len() < 2 {
        return 0.0;
    }

    // Ungated mean (from the already-absolutely-gated set).
    let mean = gated.iter().sum::<f64>() / gated.len() as f64;
    let relative_gate = mean - RELATIVE_GATE_OFFSET_LU;

    // Apply relative gate.
    let mut relative_gated: Vec<f64> = gated
        .iter()
        .cloned()
        .filter(|&v| v > relative_gate)
        .collect();

    if relative_gated.len() < 2 {
        return 0.0;
    }

    relative_gated.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = relative_gated.len();
    let low_idx = ((n as f64 * LRA_LOW_PERCENTILE) as usize).min(n - 1);
    let high_idx = ((n as f64 * LRA_HIGH_PERCENTILE) as usize).min(n - 1);

    (relative_gated[high_idx] - relative_gated[low_idx]).max(0.0)
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tracker() -> LoudnessHistoryTracker {
        LoudnessHistoryTracker::new(LoudnessHistoryConfig::default())
    }

    #[test]
    fn test_empty_tracker() {
        let tracker = make_tracker();
        assert!(tracker.is_empty());
        assert_eq!(tracker.len(), 0);
        let stats = tracker.stats();
        assert_eq!(stats.block_count, 0);
        assert_eq!(stats.loudness_range_lu, 0.0);
    }

    #[test]
    fn test_push_and_len() {
        let mut tracker = make_tracker();
        tracker.push_block(-23.0);
        tracker.push_block(-22.5);
        assert_eq!(tracker.len(), 2);
        assert!(!tracker.is_empty());
    }

    #[test]
    fn test_max_blocks_evicts_oldest() {
        let config = LoudnessHistoryConfig {
            max_blocks: 3,
            block_duration_secs: 1.0,
        };
        let mut tracker = LoudnessHistoryTracker::new(config);
        for i in 0..5 {
            tracker.push_block(-20.0 - i as f64);
        }
        assert_eq!(tracker.len(), 3);
        // The 3 retained blocks should be the last 3 pushed.
        let values = tracker.export_values();
        assert_eq!(values, vec![-22.0, -23.0, -24.0]);
    }

    #[test]
    fn test_lra_constant_signal_is_zero() {
        // All identical → 95th pct == 10th pct → LRA = 0.
        let mut tracker = make_tracker();
        tracker.push_blocks(&vec![-23.0f64; 100]);
        let lra = tracker.loudness_range();
        assert!(lra < 0.01, "lra={}", lra);
    }

    #[test]
    fn test_lra_wide_dynamic_range() {
        // Blocks range from -30 to -10 LUFS (20 LU range).
        // The gated LRA should be a substantial positive value.
        let mut tracker = make_tracker();
        let values: Vec<f64> = (0..200)
            .map(|i| -30.0 + (i as f64 * 20.0 / 199.0))
            .collect();
        tracker.push_blocks(&values);
        let lra = tracker.loudness_range();
        assert!(lra > 5.0, "lra={}", lra);
    }

    #[test]
    fn test_stats_min_max() {
        let mut tracker = make_tracker();
        tracker.push_blocks(&[-25.0, -20.0, -30.0]);
        let stats = tracker.stats();
        assert!((stats.min_lufs - (-30.0)).abs() < 0.001);
        assert!((stats.max_lufs - (-20.0)).abs() < 0.001);
    }

    #[test]
    fn test_clear_resets_state() {
        let mut tracker = make_tracker();
        tracker.push_blocks(&[-23.0, -22.0, -24.0]);
        tracker.clear();
        assert!(tracker.is_empty());
        assert_eq!(tracker.len(), 0);
        assert_eq!(tracker.next_index, 0);
    }

    #[test]
    fn test_export_csv_header() {
        let mut tracker = make_tracker();
        tracker.push_block(-23.0);
        let csv = tracker.export_csv();
        assert!(csv.starts_with("index,timestamp_secs,lufs\n"));
        assert!(csv.contains("-23.00"));
    }

    #[test]
    fn test_export_values_order() {
        let mut tracker = make_tracker();
        let expected = vec![-23.0, -22.0, -24.0];
        tracker.push_blocks(&expected);
        assert_eq!(tracker.export_values(), expected);
    }

    #[test]
    fn test_loudest_and_quietest_block() {
        let mut tracker = make_tracker();
        tracker.push_blocks(&[-25.0, -18.0, -30.0]);
        let loudest = tracker.loudest_block_index().expect("non-empty");
        let quietest = tracker.quietest_block_index().expect("non-empty");
        // Index 1 was the loudest (-18 LUFS), index 2 the quietest (-30 LUFS).
        assert_eq!(loudest, 1);
        assert_eq!(quietest, 2);
    }

    #[test]
    fn test_absolute_gate_filters_very_quiet_blocks() {
        let mut tracker = make_tracker();
        // Mix of blocks: some above −70, some at −80 (below gate).
        tracker.push_blocks(&[-23.0, -80.0, -22.5, -80.0, -24.0]);
        // Stats gated_block_count should exclude the −80 blocks.
        let stats = tracker.stats();
        assert_eq!(stats.gated_block_count, 3);
    }

    #[test]
    fn test_block_timestamps_increase() {
        let config = LoudnessHistoryConfig {
            block_duration_secs: 3.0,
            ..Default::default()
        };
        let mut tracker = LoudnessHistoryTracker::new(config);
        for _ in 0..5 {
            tracker.push_block(-23.0);
        }
        let blocks: Vec<&LoudnessBlock> = tracker.blocks().iter().collect();
        for pair in blocks.windows(2) {
            assert!(pair[1].timestamp_secs > pair[0].timestamp_secs);
        }
    }

    #[test]
    fn test_stats_mean_lufs() {
        let mut tracker = make_tracker();
        tracker.push_blocks(&[-20.0, -24.0, -22.0]);
        let stats = tracker.stats();
        // Mean of -20, -24, -22 is -22.
        assert!((stats.mean_lufs - (-22.0)).abs() < 0.001, "mean={}", stats.mean_lufs);
    }
}
