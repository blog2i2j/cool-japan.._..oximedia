#![allow(dead_code)]

//! Anti-cheat stream integrity verification for gaming streams.
//!
//! Provides mechanisms to detect and verify stream integrity, including
//! frame hash chains, pixel injection watermarks, and timing anomaly detection.

use std::collections::VecDeque;

/// Maximum number of frames to keep in the verification history.
const MAX_HISTORY_SIZE: usize = 1024;

/// Minimum frames required to establish a valid baseline.
const MIN_BASELINE_FRAMES: usize = 30;

/// Hash chain entry for frame verification.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameHashEntry {
    /// Frame sequence number.
    pub frame_id: u64,
    /// Hash of the frame pixel data.
    pub pixel_hash: u64,
    /// Hash of the previous entry (chain link).
    pub prev_hash: u64,
    /// Capture timestamp in microseconds.
    pub timestamp_us: u64,
}

impl FrameHashEntry {
    /// Create a new frame hash entry.
    #[must_use]
    pub fn new(frame_id: u64, pixel_hash: u64, prev_hash: u64, timestamp_us: u64) -> Self {
        Self {
            frame_id,
            pixel_hash,
            prev_hash,
            timestamp_us,
        }
    }

    /// Compute a combined hash of this entry for chaining.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn combined_hash(&self) -> u64 {
        let mut h = self.pixel_hash;
        h = h.wrapping_mul(6_364_136_223_846_793_005);
        h = h.wrapping_add(self.prev_hash);
        h = h.wrapping_mul(6_364_136_223_846_793_005);
        h = h.wrapping_add(self.frame_id);
        h
    }
}

/// Result of an integrity check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntegrityVerdict {
    /// Stream appears genuine.
    Genuine,
    /// Suspicious timing anomaly detected.
    SuspiciousTiming,
    /// Hash chain is broken (frame tampering).
    ChainBroken,
    /// Frame injection detected (duplicate pixel hashes with wrong timing).
    FrameInjection,
    /// Insufficient data to make a determination.
    InsufficientData,
}

/// Timing anomaly detector using frame intervals.
#[derive(Debug, Clone)]
pub struct TimingAnalyzer {
    /// Expected interval in microseconds between frames.
    expected_interval_us: u64,
    /// Tolerance as a fraction (0.0..1.0).
    tolerance: f64,
    /// Recent intervals for statistical analysis.
    intervals: VecDeque<u64>,
    /// Maximum number of intervals to store.
    max_intervals: usize,
}

impl TimingAnalyzer {
    /// Create a new timing analyzer for the given target FPS.
    ///
    /// # Arguments
    ///
    /// * `target_fps` - Expected frames per second
    /// * `tolerance` - Fraction of allowed deviation (e.g. 0.15 for 15%)
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn new(target_fps: u32, tolerance: f64) -> Self {
        let expected_interval_us = if target_fps > 0 {
            (1_000_000.0 / target_fps as f64) as u64
        } else {
            16_667 // Default to ~60 fps
        };
        Self {
            expected_interval_us,
            tolerance: tolerance.clamp(0.01, 1.0),
            intervals: VecDeque::with_capacity(256),
            max_intervals: 256,
        }
    }

    /// Record a frame interval (difference between consecutive timestamps in us).
    pub fn record_interval(&mut self, interval_us: u64) {
        if self.intervals.len() >= self.max_intervals {
            self.intervals.pop_front();
        }
        self.intervals.push_back(interval_us);
    }

    /// Check whether the current timing distribution appears anomalous.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn is_anomalous(&self) -> bool {
        if self.intervals.len() < MIN_BASELINE_FRAMES {
            return false;
        }
        let mean = self.mean_interval();
        let expected = self.expected_interval_us as f64;
        let deviation = (mean - expected).abs() / expected;
        deviation > self.tolerance
    }

    /// Compute the mean interval.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn mean_interval(&self) -> f64 {
        if self.intervals.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.intervals.iter().sum();
        sum as f64 / self.intervals.len() as f64
    }

    /// Compute the standard deviation of intervals.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn std_deviation(&self) -> f64 {
        if self.intervals.len() < 2 {
            return 0.0;
        }
        let mean = self.mean_interval();
        let variance: f64 = self
            .intervals
            .iter()
            .map(|&v| {
                let diff = v as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / (self.intervals.len() - 1) as f64;
        variance.sqrt()
    }

    /// Return the number of recorded intervals.
    #[must_use]
    pub fn interval_count(&self) -> usize {
        self.intervals.len()
    }
}

/// Pixel injection watermark for stream provenance.
#[derive(Debug, Clone)]
pub struct PixelWatermark {
    /// Seed for watermark generation.
    seed: u64,
    /// Watermark strength (0.0..1.0).
    strength: f64,
    /// Width of the watermark grid.
    grid_w: u32,
    /// Height of the watermark grid.
    grid_h: u32,
}

impl PixelWatermark {
    /// Create a new pixel watermark generator.
    ///
    /// # Arguments
    ///
    /// * `seed` - Unique seed for this stream
    /// * `strength` - Watermark visibility (0.0 invisible, 1.0 maximum)
    /// * `grid_w` - Number of horizontal grid cells
    /// * `grid_h` - Number of vertical grid cells
    #[must_use]
    pub fn new(seed: u64, strength: f64, grid_w: u32, grid_h: u32) -> Self {
        Self {
            seed,
            strength: strength.clamp(0.0, 1.0),
            grid_w: grid_w.max(1),
            grid_h: grid_h.max(1),
        }
    }

    /// Generate watermark pattern for a given frame index.
    ///
    /// Returns a vector of (grid_x, grid_y, delta) tuples where delta is the
    /// pixel value adjustment to embed.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn generate_pattern(&self, frame_index: u64) -> Vec<(u32, u32, i8)> {
        let mut pattern = Vec::new();
        let mut state = self.seed.wrapping_add(frame_index);
        for gy in 0..self.grid_h {
            for gx in 0..self.grid_w {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let bit = (state >> 33) & 1;
                let delta = if bit == 1 {
                    (self.strength * 2.0) as i8
                } else {
                    -((self.strength * 2.0) as i8)
                };
                pattern.push((gx, gy, delta));
            }
        }
        pattern
    }

    /// Verify a watermark pattern against expected values.
    ///
    /// Returns the fraction of matching cells (0.0..1.0).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn verify_pattern(&self, frame_index: u64, observed: &[(u32, u32, i8)]) -> f64 {
        let expected = self.generate_pattern(frame_index);
        if expected.is_empty() || observed.is_empty() {
            return 0.0;
        }
        let mut matches = 0usize;
        let total = expected.len().min(observed.len());
        for i in 0..total {
            if expected[i] == observed[i] {
                matches += 1;
            }
        }
        matches as f64 / total as f64
    }

    /// Return the watermark strength.
    #[must_use]
    pub fn strength(&self) -> f64 {
        self.strength
    }
}

/// Stream integrity verifier combining hash chain and timing analysis.
#[derive(Debug)]
pub struct IntegrityVerifier {
    /// Hash chain history.
    chain: VecDeque<FrameHashEntry>,
    /// Timing analyzer.
    timing: TimingAnalyzer,
    /// Last recorded timestamp.
    last_timestamp_us: Option<u64>,
    /// Number of chain breaks detected.
    chain_breaks: u64,
    /// Number of timing anomalies detected.
    timing_anomalies: u64,
}

impl IntegrityVerifier {
    /// Create a new integrity verifier.
    ///
    /// # Arguments
    ///
    /// * `target_fps` - Expected frame rate
    /// * `timing_tolerance` - Fraction tolerance for timing checks
    #[must_use]
    pub fn new(target_fps: u32, timing_tolerance: f64) -> Self {
        Self {
            chain: VecDeque::with_capacity(MAX_HISTORY_SIZE),
            timing: TimingAnalyzer::new(target_fps, timing_tolerance),
            last_timestamp_us: None,
            chain_breaks: 0,
            timing_anomalies: 0,
        }
    }

    /// Submit a new frame for verification.
    ///
    /// Returns the integrity verdict after incorporating this frame.
    pub fn submit_frame(&mut self, entry: FrameHashEntry) -> IntegrityVerdict {
        // Check hash chain continuity
        if let Some(last) = self.chain.back() {
            if entry.prev_hash != last.combined_hash() {
                self.chain_breaks += 1;
            }
        }

        // Record timing
        if let Some(last_ts) = self.last_timestamp_us {
            if entry.timestamp_us > last_ts {
                self.timing.record_interval(entry.timestamp_us - last_ts);
            }
        }
        self.last_timestamp_us = Some(entry.timestamp_us);

        // Maintain history size
        if self.chain.len() >= MAX_HISTORY_SIZE {
            self.chain.pop_front();
        }
        self.chain.push_back(entry);

        self.verdict()
    }

    /// Get the current integrity verdict.
    #[must_use]
    pub fn verdict(&self) -> IntegrityVerdict {
        if self.chain.len() < MIN_BASELINE_FRAMES {
            return IntegrityVerdict::InsufficientData;
        }
        if self.chain_breaks > 0 {
            return IntegrityVerdict::ChainBroken;
        }
        if self.timing.is_anomalous() {
            return IntegrityVerdict::SuspiciousTiming;
        }
        IntegrityVerdict::Genuine
    }

    /// Return the total number of chain breaks detected.
    #[must_use]
    pub fn chain_break_count(&self) -> u64 {
        self.chain_breaks
    }

    /// Return the number of frames in the history.
    #[must_use]
    pub fn history_len(&self) -> usize {
        self.chain.len()
    }

    /// Reset the verifier state.
    pub fn reset(&mut self) {
        self.chain.clear();
        self.timing = TimingAnalyzer::new(60, 0.15);
        self.last_timestamp_us = None;
        self.chain_breaks = 0;
        self.timing_anomalies = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_hash_entry_creation() {
        let entry = FrameHashEntry::new(1, 0xDEAD, 0, 1000);
        assert_eq!(entry.frame_id, 1);
        assert_eq!(entry.pixel_hash, 0xDEAD);
        assert_eq!(entry.prev_hash, 0);
        assert_eq!(entry.timestamp_us, 1000);
    }

    #[test]
    fn test_combined_hash_deterministic() {
        let a = FrameHashEntry::new(1, 100, 0, 1000);
        let b = FrameHashEntry::new(1, 100, 0, 1000);
        assert_eq!(a.combined_hash(), b.combined_hash());
    }

    #[test]
    fn test_combined_hash_differs_with_different_input() {
        let a = FrameHashEntry::new(1, 100, 0, 1000);
        let b = FrameHashEntry::new(2, 100, 0, 1000);
        assert_ne!(a.combined_hash(), b.combined_hash());
    }

    #[test]
    fn test_timing_analyzer_basic() {
        let mut ta = TimingAnalyzer::new(60, 0.15);
        assert_eq!(ta.interval_count(), 0);
        ta.record_interval(16_667);
        assert_eq!(ta.interval_count(), 1);
    }

    #[test]
    fn test_timing_analyzer_mean() {
        let mut ta = TimingAnalyzer::new(60, 0.15);
        ta.record_interval(16_000);
        ta.record_interval(17_000);
        let mean = ta.mean_interval();
        assert!((mean - 16_500.0).abs() < 0.1);
    }

    #[test]
    fn test_timing_analyzer_std_deviation() {
        let mut ta = TimingAnalyzer::new(60, 0.15);
        for _ in 0..100 {
            ta.record_interval(16_667);
        }
        assert!(ta.std_deviation() < 1.0);
    }

    #[test]
    fn test_timing_analyzer_anomalous_detection() {
        let mut ta = TimingAnalyzer::new(60, 0.10);
        // Feed intervals that are way off (double the expected)
        for _ in 0..50 {
            ta.record_interval(33_333);
        }
        assert!(ta.is_anomalous());
    }

    #[test]
    fn test_timing_analyzer_not_anomalous_with_good_data() {
        let mut ta = TimingAnalyzer::new(60, 0.15);
        for _ in 0..50 {
            ta.record_interval(16_667);
        }
        assert!(!ta.is_anomalous());
    }

    #[test]
    fn test_timing_analyzer_insufficient_data() {
        let mut ta = TimingAnalyzer::new(60, 0.15);
        ta.record_interval(50_000);
        assert!(!ta.is_anomalous()); // Not enough data to judge
    }

    #[test]
    fn test_pixel_watermark_generate() {
        let wm = PixelWatermark::new(42, 0.5, 4, 4);
        let pattern = wm.generate_pattern(0);
        assert_eq!(pattern.len(), 16); // 4x4
    }

    #[test]
    fn test_pixel_watermark_deterministic() {
        let wm = PixelWatermark::new(42, 0.5, 4, 4);
        let p1 = wm.generate_pattern(10);
        let p2 = wm.generate_pattern(10);
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_pixel_watermark_verify_perfect() {
        let wm = PixelWatermark::new(42, 0.5, 4, 4);
        let pattern = wm.generate_pattern(5);
        let score = wm.verify_pattern(5, &pattern);
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pixel_watermark_verify_empty() {
        let wm = PixelWatermark::new(42, 0.5, 4, 4);
        let score = wm.verify_pattern(5, &[]);
        assert!((score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pixel_watermark_strength_clamped() {
        let wm = PixelWatermark::new(1, 5.0, 2, 2);
        assert!((wm.strength() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_integrity_verifier_insufficient_data() {
        let verifier = IntegrityVerifier::new(60, 0.15);
        assert_eq!(verifier.verdict(), IntegrityVerdict::InsufficientData);
    }

    #[test]
    fn test_integrity_verifier_genuine_stream() {
        let mut verifier = IntegrityVerifier::new(60, 0.15);
        let mut prev_hash = 0u64;
        for i in 0..50 {
            let entry = FrameHashEntry::new(i, i * 1000 + 1, prev_hash, i * 16_667);
            prev_hash = entry.combined_hash();
            let _ = verifier.submit_frame(entry);
        }
        assert_eq!(verifier.verdict(), IntegrityVerdict::Genuine);
    }

    #[test]
    fn test_integrity_verifier_chain_broken() {
        let mut verifier = IntegrityVerifier::new(60, 0.15);
        for i in 0..50 {
            // prev_hash is always wrong (0) except for the first frame
            let entry = FrameHashEntry::new(i, i * 1000 + 1, 0, i * 16_667);
            let _ = verifier.submit_frame(entry);
        }
        assert_eq!(verifier.verdict(), IntegrityVerdict::ChainBroken);
    }

    #[test]
    fn test_integrity_verifier_reset() {
        let mut verifier = IntegrityVerifier::new(60, 0.15);
        let entry = FrameHashEntry::new(0, 100, 0, 0);
        let _ = verifier.submit_frame(entry);
        assert_eq!(verifier.history_len(), 1);
        verifier.reset();
        assert_eq!(verifier.history_len(), 0);
        assert_eq!(verifier.chain_break_count(), 0);
    }
}
