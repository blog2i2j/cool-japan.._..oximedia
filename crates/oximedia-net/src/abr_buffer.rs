//! Buffer-based ABR (BBA) strategy — selects quality based on buffer occupancy.
//!
//! This is a standalone, lightweight BBA implementation complementary to the
//! full `crate::abr::bba::BbaController`.  It operates purely on buffer
//! fill level (in seconds) without requiring a bandwidth estimator, making
//! it suitable for simple integration scenarios.
//!
//! # Algorithm
//!
//! The player buffer is divided into three zones:
//!
//! - **Reservoir** (0 → `reservoir_secs`): emergency zone — always select
//!   lowest quality to avoid stalls.
//! - **Cushion** (`reservoir_secs` → `cushion_secs`): linear ramp — quality
//!   scales proportionally with buffer fill.
//! - **Upper reservoir** (`cushion_secs` → `upper_reservoir_secs`): steady
//!   state — select highest quality.
//!
//! Hysteresis prevents oscillation: a downswitch only fires if the buffer
//! drops at least 2 seconds below the threshold that would trigger the
//! current quality level's upswitch.

#![allow(dead_code)]

// ─── Configuration ──────────────────────────────────────────────────────────

/// Buffer-based ABR configuration.
#[derive(Debug, Clone)]
pub struct BufferAbrConfig {
    /// Reservoir threshold in seconds — below this, select lowest quality.
    pub reservoir_secs: f32,
    /// Cushion threshold in seconds — between reservoir and cushion, ramp
    /// quality linearly.
    pub cushion_secs: f32,
    /// Upper reservoir — above this, select highest quality.
    pub upper_reservoir_secs: f32,
    /// Available quality levels (bitrates in kbps, sorted ascending).
    pub quality_levels: Vec<u32>,
}

impl Default for BufferAbrConfig {
    fn default() -> Self {
        Self {
            reservoir_secs: 5.0,
            cushion_secs: 15.0,
            upper_reservoir_secs: 30.0,
            quality_levels: vec![500, 1000, 2000, 4000, 6000],
        }
    }
}

// ─── Controller ─────────────────────────────────────────────────────────────

/// Buffer-based ABR controller.
///
/// Selects quality level purely from the current buffer occupancy.
pub struct BufferAbr {
    config: BufferAbrConfig,
    current_quality_index: usize,
    /// History of quality decisions for stability tracking.
    decision_history: Vec<usize>,
    /// Hysteresis margin in seconds — the buffer must drop this far below
    /// the upswitch threshold before a downswitch fires.
    hysteresis_secs: f32,
}

impl BufferAbr {
    /// Creates a new buffer-based ABR controller.
    #[must_use]
    pub fn new(config: BufferAbrConfig) -> Self {
        Self {
            config,
            current_quality_index: 0,
            decision_history: Vec::new(),
            hysteresis_secs: 2.0,
        }
    }

    /// Selects quality level given current buffer occupancy in seconds.
    ///
    /// Returns the index into `quality_levels`.
    pub fn select_quality(&mut self, buffer_secs: f32) -> usize {
        let num_levels = self.config.quality_levels.len();
        if num_levels == 0 {
            self.current_quality_index = 0;
            self.decision_history.push(0);
            return 0;
        }

        let max_idx = num_levels - 1;

        let raw_idx = if buffer_secs <= self.config.reservoir_secs {
            // Emergency: lowest quality.
            0
        } else if buffer_secs >= self.config.upper_reservoir_secs {
            // Plenty of buffer: highest quality.
            max_idx
        } else {
            // Linear ramp in the cushion zone [reservoir, cushion].
            // Above cushion but below upper_reservoir stays at highest cushion
            // mapping (which is max_idx when cushion < upper_reservoir).
            let range = self.config.cushion_secs - self.config.reservoir_secs;
            if range <= 0.0 {
                0
            } else {
                let t = ((buffer_secs - self.config.reservoir_secs) / range).clamp(0.0, 1.0);
                let idx = (t * max_idx as f32).round() as usize;
                idx.min(max_idx)
            }
        };

        // Apply hysteresis: don't switch *down* unless the buffer drops
        // at least `hysteresis_secs` below the threshold that would cause
        // the current level's upswitch.
        let new_idx = if raw_idx < self.current_quality_index {
            // Compute the buffer level that would select our current quality
            // during an upswitch.
            let range = self.config.cushion_secs - self.config.reservoir_secs;
            let current_threshold = if range > 0.0 && max_idx > 0 {
                self.config.reservoir_secs
                    + (self.current_quality_index as f32 / max_idx as f32) * range
            } else {
                self.config.reservoir_secs
            };
            // Only downswitch if we're sufficiently below
            if buffer_secs < current_threshold - self.hysteresis_secs {
                raw_idx
            } else {
                self.current_quality_index
            }
        } else {
            raw_idx
        };

        self.current_quality_index = new_idx;
        self.decision_history.push(new_idx);
        new_idx
    }

    /// Gets the bitrate (kbps) for the currently selected quality level.
    #[must_use]
    pub fn selected_bitrate(&self) -> u32 {
        self.config
            .quality_levels
            .get(self.current_quality_index)
            .copied()
            .unwrap_or(0)
    }

    /// Current quality index.
    #[must_use]
    pub fn current_quality(&self) -> usize {
        self.current_quality_index
    }

    /// Number of quality switches in the last N decisions.
    #[must_use]
    pub fn switch_count(&self, last_n: usize) -> u32 {
        let history = &self.decision_history;
        if history.len() < 2 {
            return 0;
        }
        let start = history.len().saturating_sub(last_n);
        let window = &history[start..];
        let mut count = 0u32;
        for pair in window.windows(2) {
            if pair[0] != pair[1] {
                count += 1;
            }
        }
        count
    }

    /// Resets the controller to initial state.
    pub fn reset(&mut self) {
        self.current_quality_index = 0;
        self.decision_history.clear();
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_abr() -> BufferAbr {
        BufferAbr::new(BufferAbrConfig::default())
    }

    // 1. buffer < reservoir → lowest quality
    #[test]
    fn test_abr_low_buffer_lowest_quality() {
        let mut abr = default_abr();
        let idx = abr.select_quality(2.0); // well below reservoir (5.0)
        assert_eq!(idx, 0);
    }

    // 2. buffer > upper_reservoir → highest quality
    #[test]
    fn test_abr_high_buffer_highest_quality() {
        let mut abr = default_abr();
        let idx = abr.select_quality(35.0); // above upper_reservoir (30.0)
        let max_idx = abr.config.quality_levels.len() - 1;
        assert_eq!(idx, max_idx);
    }

    // 3. buffer in cushion zone → middle quality
    #[test]
    fn test_abr_mid_buffer_ramps() {
        let mut abr = default_abr();
        // Midpoint of [5.0, 15.0] = 10.0 → 50% → index 2 of [0..4]
        let idx = abr.select_quality(10.0);
        assert!(idx > 0, "should be above lowest");
        let max_idx = abr.config.quality_levels.len() - 1;
        assert!(idx < max_idx, "should be below highest");
    }

    // 4. Tracks quality switches
    #[test]
    fn test_abr_switch_count() {
        let mut abr = default_abr();
        abr.select_quality(2.0); // → 0
        abr.select_quality(35.0); // → 4 (switch)
        abr.select_quality(35.0); // → 4 (no switch)
        abr.select_quality(2.0); // → 0 (switch)
        assert_eq!(abr.switch_count(10), 2);
    }

    // 5. Returns correct bitrate for index
    #[test]
    fn test_abr_selected_bitrate() {
        let mut abr = default_abr();
        abr.select_quality(2.0); // → lowest (index 0)
        assert_eq!(abr.selected_bitrate(), 500);
        abr.select_quality(35.0); // → highest (index 4)
        assert_eq!(abr.selected_bitrate(), 6000);
    }

    // 6. Default config is reasonable
    #[test]
    fn test_abr_default_config() {
        let cfg = BufferAbrConfig::default();
        assert!(cfg.reservoir_secs > 0.0);
        assert!(cfg.cushion_secs > cfg.reservoir_secs);
        assert!(cfg.upper_reservoir_secs > cfg.cushion_secs);
        assert!(!cfg.quality_levels.is_empty());
        // Quality levels should be sorted ascending
        for w in cfg.quality_levels.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    // 7. Reset returns to initial state
    #[test]
    fn test_abr_reset() {
        let mut abr = default_abr();
        abr.select_quality(35.0); // moves to highest
        assert!(abr.current_quality() > 0);
        abr.reset();
        assert_eq!(abr.current_quality(), 0);
        assert_eq!(abr.switch_count(100), 0);
    }

    // 8. Stability — doesn't oscillate on borderline buffer
    #[test]
    fn test_abr_stability() {
        let mut abr = default_abr();
        // First ramp up to a mid-level
        abr.select_quality(12.0);
        let level = abr.current_quality();
        // Now oscillate buffer just slightly below the current level's threshold
        // The hysteresis should prevent constant switching.
        for _ in 0..10 {
            abr.select_quality(11.5);
            abr.select_quality(12.5);
        }
        // Should not have many switches — hysteresis prevents oscillation
        let switches = abr.switch_count(20);
        assert!(
            switches <= 2,
            "too many switches ({switches}), hysteresis should prevent oscillation"
        );
        let _ = level; // used above
    }
}
