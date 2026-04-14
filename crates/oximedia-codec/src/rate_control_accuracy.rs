//! Rate control accuracy verification.
//!
//! This module provides utilities for testing that rate control implementations
//! achieve their target bitrate within acceptable tolerances. It measures actual
//! output size against the configured target and computes deviation metrics.
//!
//! # Accuracy Criteria
//!
//! - **CBR mode**: Output bitrate should stay within ±5% of target over a
//!   sliding window of at least 1 second.
//! - **VBR mode**: Average bitrate over the entire sequence should stay within
//!   ±10% of target.
//! - **CRF mode**: Quality should remain stable (± 2 QP) across frames of
//!   similar complexity.
//!
//! # Usage
//!
//! ```rust
//! use oximedia_codec::rate_control_accuracy::{
//!     RateControlVerifier, RcVerifyMode, VerificationResult,
//! };
//!
//! let mut verifier = RateControlVerifier::new(
//!     2_000_000,    // 2 Mbps target
//!     30.0,         // 30 fps
//!     RcVerifyMode::Cbr { tolerance: 0.05 },
//! );
//!
//! // Feed frame sizes after encoding
//! for _ in 0..90 {
//!     verifier.record_frame(8000, false); // ~8000 bytes per frame
//! }
//!
//! let result = verifier.verify();
//! assert!(result.passes, "CBR should be within tolerance: {}", result.summary());
//! ```

/// Rate control verification mode.
#[derive(Debug, Clone)]
pub enum RcVerifyMode {
    /// Constant Bitrate: measured bitrate must stay within `tolerance` fraction
    /// of target over any 1-second sliding window.
    Cbr {
        /// Fractional tolerance (e.g. 0.05 for ±5%).
        tolerance: f64,
    },
    /// Variable Bitrate: average bitrate over the full sequence must stay within
    /// `tolerance` of target.
    Vbr {
        /// Fractional tolerance (e.g. 0.10 for ±10%).
        tolerance: f64,
    },
    /// Constant Rate Factor: not bitrate-based but QP-stability-based.
    Crf {
        /// Maximum allowed QP deviation from the median.
        max_qp_deviation: u8,
    },
}

/// A single recorded frame's statistics.
#[derive(Debug, Clone)]
struct FrameRecord {
    /// Size of encoded frame in bytes.
    size_bytes: u32,
    /// Whether this frame was a keyframe.
    is_keyframe: bool,
    /// Optional QP value (for CRF verification).
    qp: Option<u8>,
}

/// Verifies that an encoder's rate control meets its targets.
#[derive(Debug)]
pub struct RateControlVerifier {
    /// Target bitrate in bits per second.
    target_bitrate: u64,
    /// Frame rate in fps.
    framerate: f64,
    /// Verification mode.
    mode: RcVerifyMode,
    /// Recorded frame data.
    frames: Vec<FrameRecord>,
}

impl RateControlVerifier {
    /// Create a new rate control verifier.
    #[must_use]
    pub fn new(target_bitrate: u64, framerate: f64, mode: RcVerifyMode) -> Self {
        Self {
            target_bitrate,
            framerate,
            mode,
            frames: Vec::new(),
        }
    }

    /// Record one encoded frame.
    pub fn record_frame(&mut self, size_bytes: u32, is_keyframe: bool) {
        self.frames.push(FrameRecord {
            size_bytes,
            is_keyframe,
            qp: None,
        });
    }

    /// Record one encoded frame with its QP value (for CRF mode).
    pub fn record_frame_with_qp(&mut self, size_bytes: u32, is_keyframe: bool, qp: u8) {
        self.frames.push(FrameRecord {
            size_bytes,
            is_keyframe,
            qp: Some(qp),
        });
    }

    /// Get the total number of recorded frames.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Compute the overall average bitrate of the recorded sequence.
    #[must_use]
    pub fn average_bitrate(&self) -> f64 {
        if self.frames.is_empty() || self.framerate <= 0.0 {
            return 0.0;
        }
        let total_bits: u64 = self
            .frames
            .iter()
            .map(|f| u64::from(f.size_bytes) * 8)
            .sum();
        let duration_seconds = self.frames.len() as f64 / self.framerate;
        total_bits as f64 / duration_seconds
    }

    /// Compute the deviation of average bitrate from target.
    ///
    /// Returns a fraction: `(actual - target) / target`.
    /// Positive means over-target, negative means under-target.
    #[must_use]
    pub fn bitrate_deviation(&self) -> f64 {
        let avg = self.average_bitrate();
        if self.target_bitrate == 0 {
            return 0.0;
        }
        (avg - self.target_bitrate as f64) / self.target_bitrate as f64
    }

    /// Compute bitrate for a sliding window of `window_frames` frames
    /// starting at each frame position.
    fn sliding_window_bitrates(&self, window_frames: usize) -> Vec<f64> {
        if self.frames.len() < window_frames || window_frames == 0 {
            return vec![];
        }
        let mut results = Vec::with_capacity(self.frames.len() - window_frames + 1);
        let duration_seconds = window_frames as f64 / self.framerate;

        // Compute initial window sum
        let mut window_bits: u64 = self.frames[..window_frames]
            .iter()
            .map(|f| u64::from(f.size_bytes) * 8)
            .sum();
        results.push(window_bits as f64 / duration_seconds);

        // Slide the window
        for i in window_frames..self.frames.len() {
            window_bits += u64::from(self.frames[i].size_bytes) * 8;
            window_bits -= u64::from(self.frames[i - window_frames].size_bytes) * 8;
            results.push(window_bits as f64 / duration_seconds);
        }
        results
    }

    /// Run verification and return a detailed result.
    #[must_use]
    pub fn verify(&self) -> VerificationResult {
        match &self.mode {
            RcVerifyMode::Cbr { tolerance } => self.verify_cbr(*tolerance),
            RcVerifyMode::Vbr { tolerance } => self.verify_vbr(*tolerance),
            RcVerifyMode::Crf { max_qp_deviation } => self.verify_crf(*max_qp_deviation),
        }
    }

    fn verify_cbr(&self, tolerance: f64) -> VerificationResult {
        let window_frames = (self.framerate.ceil() as usize).max(1); // ~1 second
        let window_bitrates = self.sliding_window_bitrates(window_frames);

        if window_bitrates.is_empty() {
            return VerificationResult {
                passes: false,
                average_bitrate: 0.0,
                target_bitrate: self.target_bitrate as f64,
                max_deviation: 0.0,
                min_window_bitrate: 0.0,
                max_window_bitrate: 0.0,
                details: "Not enough frames for 1-second window".to_string(),
            };
        }

        let target = self.target_bitrate as f64;
        let mut max_deviation = 0.0_f64;
        let mut min_br = f64::MAX;
        let mut max_br = f64::MIN;

        for &br in &window_bitrates {
            let dev = ((br - target) / target).abs();
            if dev > max_deviation {
                max_deviation = dev;
            }
            if br < min_br {
                min_br = br;
            }
            if br > max_br {
                max_br = br;
            }
        }

        let passes = max_deviation <= tolerance;
        let avg = self.average_bitrate();

        VerificationResult {
            passes,
            average_bitrate: avg,
            target_bitrate: target,
            max_deviation,
            min_window_bitrate: min_br,
            max_window_bitrate: max_br,
            details: format!(
                "CBR: max window deviation={:.2}% (tolerance={:.2}%)",
                max_deviation * 100.0,
                tolerance * 100.0
            ),
        }
    }

    fn verify_vbr(&self, tolerance: f64) -> VerificationResult {
        let avg = self.average_bitrate();
        let target = self.target_bitrate as f64;
        let deviation = if target > 0.0 {
            ((avg - target) / target).abs()
        } else {
            0.0
        };

        VerificationResult {
            passes: deviation <= tolerance,
            average_bitrate: avg,
            target_bitrate: target,
            max_deviation: deviation,
            min_window_bitrate: avg,
            max_window_bitrate: avg,
            details: format!(
                "VBR: avg deviation={:.2}% (tolerance={:.2}%)",
                deviation * 100.0,
                tolerance * 100.0
            ),
        }
    }

    fn verify_crf(&self, max_qp_deviation: u8) -> VerificationResult {
        let qp_values: Vec<u8> = self.frames.iter().filter_map(|f| f.qp).collect();

        if qp_values.is_empty() {
            return VerificationResult {
                passes: false,
                average_bitrate: self.average_bitrate(),
                target_bitrate: self.target_bitrate as f64,
                max_deviation: 0.0,
                min_window_bitrate: 0.0,
                max_window_bitrate: 0.0,
                details: "CRF: no QP values recorded".to_string(),
            };
        }

        let mut sorted_qp = qp_values.clone();
        sorted_qp.sort_unstable();
        let median_qp = sorted_qp[sorted_qp.len() / 2];

        let max_dev = qp_values
            .iter()
            .map(|&q| (q as i16 - median_qp as i16).unsigned_abs() as u8)
            .max()
            .unwrap_or(0);

        let passes = max_dev <= max_qp_deviation;

        VerificationResult {
            passes,
            average_bitrate: self.average_bitrate(),
            target_bitrate: self.target_bitrate as f64,
            max_deviation: f64::from(max_dev),
            min_window_bitrate: 0.0,
            max_window_bitrate: 0.0,
            details: format!(
                "CRF: max QP deviation={} (limit={}), median QP={}",
                max_dev, max_qp_deviation, median_qp
            ),
        }
    }

    /// Reset the verifier, clearing all recorded frames.
    pub fn reset(&mut self) {
        self.frames.clear();
    }
}

/// Result of rate control verification.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Whether the rate control met its target within tolerance.
    pub passes: bool,
    /// Measured average bitrate over the full sequence.
    pub average_bitrate: f64,
    /// Configured target bitrate.
    pub target_bitrate: f64,
    /// Maximum measured deviation (fractional for CBR/VBR, QP units for CRF).
    pub max_deviation: f64,
    /// Minimum bitrate observed in any 1-second window (CBR only).
    pub min_window_bitrate: f64,
    /// Maximum bitrate observed in any 1-second window (CBR only).
    pub max_window_bitrate: f64,
    /// Human-readable summary.
    pub details: String,
}

impl VerificationResult {
    /// Return a human-readable summary string.
    #[must_use]
    pub fn summary(&self) -> &str {
        &self.details
    }
}

// =============================================================================
// Tests — Rate control accuracy (CBR within 5%, VBR within 10%)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── CBR mode tests ──────────────────────────────────────────────────────

    #[test]
    fn test_cbr_perfect_bitrate() {
        let target = 2_000_000u64; // 2 Mbps
        let fps = 30.0;
        let mut v = RateControlVerifier::new(target, fps, RcVerifyMode::Cbr { tolerance: 0.05 });

        // Each frame: 2_000_000 / 30 / 8 ≈ 8333 bytes
        let frame_bytes = (target as f64 / fps / 8.0) as u32;
        for _ in 0..90 {
            v.record_frame(frame_bytes, false);
        }

        let result = v.verify();
        assert!(
            result.passes,
            "perfect CBR should pass: {}",
            result.summary()
        );
        assert!(result.max_deviation < 0.01);
    }

    #[test]
    fn test_cbr_within_5_percent() {
        let target = 1_000_000u64;
        let fps = 24.0;
        let mut v = RateControlVerifier::new(target, fps, RcVerifyMode::Cbr { tolerance: 0.05 });

        let base_bytes = (target as f64 / fps / 8.0) as u32;
        // Alternate slightly above and below target
        for i in 0..120 {
            let variation = if i % 2 == 0 {
                (base_bytes as f64 * 1.04) as u32
            } else {
                (base_bytes as f64 * 0.96) as u32
            };
            v.record_frame(variation, i % 24 == 0);
        }

        let result = v.verify();
        assert!(
            result.passes,
            "±4% variation should be within 5% tolerance: {}",
            result.summary()
        );
    }

    #[test]
    fn test_cbr_exceeds_tolerance() {
        let target = 2_000_000u64;
        let fps = 30.0;
        let mut v = RateControlVerifier::new(target, fps, RcVerifyMode::Cbr { tolerance: 0.05 });

        let base_bytes = (target as f64 / fps / 8.0) as u32;
        // First half: double the target bitrate
        for _ in 0..45 {
            v.record_frame(base_bytes * 2, false);
        }
        // Second half: normal
        for _ in 0..45 {
            v.record_frame(base_bytes, false);
        }

        let result = v.verify();
        assert!(
            !result.passes,
            "2x bitrate burst should exceed 5% tolerance: {}",
            result.summary()
        );
    }

    #[test]
    fn test_cbr_not_enough_frames() {
        let mut v =
            RateControlVerifier::new(1_000_000, 30.0, RcVerifyMode::Cbr { tolerance: 0.05 });
        v.record_frame(5000, false);
        let result = v.verify();
        assert!(
            !result.passes,
            "too few frames should fail: {}",
            result.summary()
        );
    }

    // ── VBR mode tests ──────────────────────────────────────────────────────

    #[test]
    fn test_vbr_within_tolerance() {
        let target = 4_000_000u64;
        let fps = 60.0;
        let mut v = RateControlVerifier::new(target, fps, RcVerifyMode::Vbr { tolerance: 0.10 });

        let base_bytes = (target as f64 / fps / 8.0) as u32;
        // Vary frame sizes but keep average near target
        for i in 0..300 {
            let size = if i % 60 == 0 {
                base_bytes * 3 // keyframe burst
            } else {
                (base_bytes as f64 * 0.95) as u32 // compensate
            };
            v.record_frame(size, i % 60 == 0);
        }

        let result = v.verify();
        let deviation = result.max_deviation;
        // The keyframe bursts should be averaged out
        assert!(
            deviation < 0.15,
            "VBR with averaged bursts should be near target, deviation={:.2}%",
            deviation * 100.0
        );
    }

    #[test]
    fn test_vbr_over_target() {
        let target = 1_000_000u64;
        let fps = 30.0;
        let mut v = RateControlVerifier::new(target, fps, RcVerifyMode::Vbr { tolerance: 0.10 });

        // All frames 50% larger than target
        let over_bytes = ((target as f64 / fps / 8.0) * 1.5) as u32;
        for _ in 0..90 {
            v.record_frame(over_bytes, false);
        }

        let result = v.verify();
        assert!(
            !result.passes,
            "50% over target should fail 10% tolerance: {}",
            result.summary()
        );
    }

    // ── CRF mode tests ──────────────────────────────────────────────────────

    #[test]
    fn test_crf_stable_qp() {
        let mut v = RateControlVerifier::new(
            0,
            30.0,
            RcVerifyMode::Crf {
                max_qp_deviation: 2,
            },
        );

        // All frames use QP 28±1
        for i in 0..60 {
            let qp = if i % 3 == 0 { 27 } else { 28 };
            v.record_frame_with_qp(5000, false, qp);
        }

        let result = v.verify();
        assert!(result.passes, "stable QP should pass: {}", result.summary());
    }

    #[test]
    fn test_crf_unstable_qp() {
        let mut v = RateControlVerifier::new(
            0,
            30.0,
            RcVerifyMode::Crf {
                max_qp_deviation: 2,
            },
        );

        // QP swings wildly
        for i in 0..60 {
            let qp = if i % 2 == 0 { 20 } else { 40 };
            v.record_frame_with_qp(5000, false, qp);
        }

        let result = v.verify();
        assert!(
            !result.passes,
            "QP swing of 20 should fail deviation limit of 2: {}",
            result.summary()
        );
    }

    #[test]
    fn test_crf_no_qp_data() {
        let mut v = RateControlVerifier::new(
            0,
            30.0,
            RcVerifyMode::Crf {
                max_qp_deviation: 2,
            },
        );
        v.record_frame(5000, false);
        let result = v.verify();
        assert!(!result.passes, "no QP data should fail");
    }

    // ── Utility method tests ────────────────────────────────────────────────

    #[test]
    fn test_average_bitrate_calculation() {
        let mut v =
            RateControlVerifier::new(1_000_000, 10.0, RcVerifyMode::Vbr { tolerance: 0.10 });
        // 10 frames at 10 fps = 1 second, each 12500 bytes = 100000 bits
        for _ in 0..10 {
            v.record_frame(12500, false);
        }
        let avg = v.average_bitrate();
        assert!(
            (avg - 1_000_000.0).abs() < 1.0,
            "average bitrate should be 1 Mbps, got {avg}"
        );
    }

    #[test]
    fn test_bitrate_deviation() {
        let mut v =
            RateControlVerifier::new(1_000_000, 10.0, RcVerifyMode::Vbr { tolerance: 0.10 });
        // Produce exactly 1.1 Mbps (10% over)
        let bytes_per_frame = (1_100_000.0 / 10.0 / 8.0) as u32;
        for _ in 0..10 {
            v.record_frame(bytes_per_frame, false);
        }
        let dev = v.bitrate_deviation();
        assert!(
            (dev - 0.1).abs() < 0.01,
            "deviation should be ~10%, got {:.2}%",
            dev * 100.0
        );
    }

    #[test]
    fn test_frame_count() {
        let mut v =
            RateControlVerifier::new(1_000_000, 30.0, RcVerifyMode::Cbr { tolerance: 0.05 });
        for _ in 0..42 {
            v.record_frame(1000, false);
        }
        assert_eq!(v.frame_count(), 42);
    }

    #[test]
    fn test_reset() {
        let mut v =
            RateControlVerifier::new(1_000_000, 30.0, RcVerifyMode::Cbr { tolerance: 0.05 });
        v.record_frame(1000, false);
        v.reset();
        assert_eq!(v.frame_count(), 0);
        assert!(v.average_bitrate() < f64::EPSILON);
    }
}
