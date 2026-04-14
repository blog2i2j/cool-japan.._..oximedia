//! Adaptive scene change detection using content complexity histograms.
//!
//! This module extends the basic scene detection from [`crate::scene_detection`]
//! with a **content-complexity-aware** adaptive threshold: scenes with high
//! histogram entropy (fine-grained content, film grain, fast motion) require a
//! larger delta to be classified as a cut, while simple, low-variance content
//! (title cards, solid backgrounds) can be detected with a tighter threshold.
//!
//! # Algorithm
//!
//! 1. Compute a normalised 256-bin luma histogram for each frame.
//! 2. Estimate **histogram entropy** `H = -Σ p·log2(p)` as a proxy for spatial
//!    complexity.  Maximum entropy (all bins equally populated) is log2(256) ≈ 8 bits.
//! 3. Maintain an **exponential moving average** of entropy across the last
//!    `window` frames to track the running scene complexity.
//! 4. Scale the cut-detection threshold linearly between `threshold_min` and
//!    `threshold_max` using the normalised EMA entropy.
//! 5. Compute the **χ² distance** between consecutive histograms and compare it
//!    against the current adaptive threshold.
//!
//! # Example
//!
//! ```rust
//! use oximedia_video::adaptive_complexity_detect::{
//!     AdaptiveComplexityDetector, DetectorConfig, ComplexitySceneChange,
//! };
//!
//! let cfg = DetectorConfig::default();
//! let mut detector = AdaptiveComplexityDetector::new(cfg);
//!
//! // Feed synthetic 8×8 frames
//! let flat   = vec![128u8; 64];
//! let bright = vec![200u8; 64];
//!
//! detector.push_frame(&flat,   8, 8, 0);
//! let result = detector.push_frame(&bright, 8, 8, 1);
//! // `result` may contain a scene-change event if delta exceeds threshold.
//! let _ = result;
//! ```

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by [`AdaptiveComplexityDetector`].
#[derive(Debug, thiserror::Error)]
pub enum AdaptiveDetectError {
    /// Frame dimensions are inconsistent with previously pushed frames.
    #[error("frame dimension mismatch: expected {expected_w}×{expected_h}, got {got_w}×{got_h}")]
    DimensionMismatch {
        /// Expected width.
        expected_w: u32,
        /// Expected height.
        expected_h: u32,
        /// Provided width.
        got_w: u32,
        /// Provided height.
        got_h: u32,
    },
    /// The frame data length does not match `width × height`.
    #[error("frame data length {got} does not match {width}×{height}={expected}")]
    DataLengthMismatch {
        /// Actual data length.
        got: usize,
        /// Expected data length.
        expected: usize,
        /// Frame width.
        width: u32,
        /// Frame height.
        height: u32,
    },
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`AdaptiveComplexityDetector`].
#[derive(Debug, Clone)]
pub struct DetectorConfig {
    /// Minimum χ² threshold used for very simple (low-entropy) content.
    ///
    /// Typical range: `0.03` – `0.10`.  Default: `0.04`.
    pub threshold_min: f64,
    /// Maximum χ² threshold used for very complex (high-entropy) content.
    ///
    /// Typical range: `0.15` – `0.40`.  Default: `0.25`.
    pub threshold_max: f64,
    /// EMA smoothing factor for entropy tracking (`α` in \[0,1\]).
    ///
    /// Smaller values make the running average react more slowly.  Default: `0.15`.
    pub ema_alpha: f64,
    /// Minimum confidence (0–1) required to emit a scene-change event.
    ///
    /// Default: `0.35`.
    pub min_confidence: f32,
    /// Maximum entropy cap used to normalise the complexity score.
    ///
    /// Theoretical maximum for 256 bins is `log2(256) = 8.0`.  Default: `8.0`.
    pub max_entropy: f64,
    /// Number of frames that must elapse between two consecutive cuts.
    ///
    /// Prevents double-detections on very short sharp flashes.  Default: `5`.
    pub min_scene_gap: u64,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            threshold_min: 0.04,
            threshold_max: 0.25,
            ema_alpha: 0.15,
            min_confidence: 0.35,
            max_entropy: 8.0,
            min_scene_gap: 5,
        }
    }
}

// ---------------------------------------------------------------------------
// Output type
// ---------------------------------------------------------------------------

/// A scene change detected by [`AdaptiveComplexityDetector`].
#[derive(Debug, Clone)]
pub struct ComplexitySceneChange {
    /// Frame index of the first frame of the *new* scene.
    pub frame_number: u64,
    /// χ² distance that triggered the detection.
    pub chi_squared: f64,
    /// Adaptive threshold that was active at this frame.
    pub threshold: f64,
    /// Detection confidence in [0.0, 1.0].
    pub confidence: f32,
    /// Estimated entropy of the *current* frame (proxy for complexity).
    pub entropy: f64,
    /// Whether content complexity was considered high at the moment of detection.
    pub high_complexity: bool,
}

// ---------------------------------------------------------------------------
// Per-frame histogram summary
// ---------------------------------------------------------------------------

/// Internal per-frame summary stored in the history ring-buffer.
#[derive(Debug, Clone)]
struct FrameSummary {
    /// Normalised luma histogram (256 bins, Σ = 1.0).
    histogram: [f64; 256],
    /// Shannon entropy of the histogram (retained for future diagnostics).
    #[allow(dead_code)]
    entropy: f64,
    /// Frame index (retained for future diagnostics).
    #[allow(dead_code)]
    frame_number: u64,
}

// ---------------------------------------------------------------------------
// Detector implementation
// ---------------------------------------------------------------------------

/// Adaptive scene change detector driven by per-frame content complexity.
///
/// Feed raw luma (grayscale) frame data through [`Self::push_frame`]; the
/// detector returns a [`ComplexitySceneChange`] whenever it determines a cut
/// occurred.
#[derive(Debug)]
pub struct AdaptiveComplexityDetector {
    config: DetectorConfig,
    /// Ring-buffer of the last N frame summaries (N = `window`, currently 1).
    history: VecDeque<FrameSummary>,
    /// Exponential moving average of entropy across recent frames.
    ema_entropy: f64,
    /// Whether the EMA has been seeded with at least one frame.
    ema_seeded: bool,
    /// Index of the last detected cut frame (for gap enforcement).
    last_cut_frame: Option<u64>,
    /// Width required for all frames (set on first push).
    width: Option<u32>,
    /// Height required for all frames (set on first push).
    height: Option<u32>,
}

impl AdaptiveComplexityDetector {
    /// Create a new detector with the given [`DetectorConfig`].
    pub fn new(config: DetectorConfig) -> Self {
        Self {
            config,
            history: VecDeque::with_capacity(2),
            ema_entropy: 0.0,
            ema_seeded: false,
            last_cut_frame: None,
            width: None,
            height: None,
        }
    }

    /// Create a new detector with default configuration.
    pub fn default_config() -> Self {
        Self::new(DetectorConfig::default())
    }

    /// Push a new luma frame and return a scene change event if one is detected.
    ///
    /// `frame_data` must be a packed planar grayscale buffer of exactly
    /// `width × height` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`AdaptiveDetectError::DimensionMismatch`] if the width/height
    /// differs from the first frame, or
    /// [`AdaptiveDetectError::DataLengthMismatch`] if `frame_data.len()` ≠
    /// `width × height`.
    pub fn push_frame(
        &mut self,
        frame_data: &[u8],
        width: u32,
        height: u32,
        frame_number: u64,
    ) -> Result<Option<ComplexitySceneChange>, AdaptiveDetectError> {
        let expected_len = (width as usize).saturating_mul(height as usize);
        if frame_data.len() != expected_len {
            return Err(AdaptiveDetectError::DataLengthMismatch {
                got: frame_data.len(),
                expected: expected_len,
                width,
                height,
            });
        }
        // Enforce consistent dimensions.
        match (self.width, self.height) {
            (None, None) => {
                self.width = Some(width);
                self.height = Some(height);
            }
            (Some(w), Some(h)) if w != width || h != height => {
                return Err(AdaptiveDetectError::DimensionMismatch {
                    expected_w: w,
                    expected_h: h,
                    got_w: width,
                    got_h: height,
                });
            }
            _ => {}
        }

        let histogram = build_histogram(frame_data);
        let entropy = shannon_entropy(&histogram);

        // Update EMA entropy.
        if self.ema_seeded {
            let alpha = self.config.ema_alpha;
            self.ema_entropy = alpha * entropy + (1.0 - alpha) * self.ema_entropy;
        } else {
            self.ema_entropy = entropy;
            self.ema_seeded = true;
        }

        let summary = FrameSummary {
            histogram,
            entropy,
            frame_number,
        };

        let result = if let Some(prev) = self.history.back() {
            let chi2 = chi_squared_distance(&prev.histogram, &summary.histogram);
            let threshold = self.current_threshold();
            let event = self.evaluate_cut(chi2, threshold, frame_number, entropy);
            event
        } else {
            None
        };

        // Keep only the most recent frame in history.
        if self.history.len() >= 1 {
            self.history.pop_front();
        }
        self.history.push_back(summary);

        Ok(result)
    }

    /// Return the currently active adaptive cut threshold based on EMA entropy.
    pub fn current_threshold(&self) -> f64 {
        let normalised = (self.ema_entropy / self.config.max_entropy).clamp(0.0, 1.0);
        self.config.threshold_min
            + normalised * (self.config.threshold_max - self.config.threshold_min)
    }

    /// Return the current EMA entropy estimate.
    pub fn ema_entropy(&self) -> f64 {
        self.ema_entropy
    }

    /// Check whether a given χ² value exceeds the adaptive threshold and emit
    /// a [`ComplexitySceneChange`] if the minimum confidence is met.
    fn evaluate_cut(
        &mut self,
        chi2: f64,
        threshold: f64,
        frame_number: u64,
        entropy: f64,
    ) -> Option<ComplexitySceneChange> {
        if chi2 < threshold {
            return None;
        }
        // Gap guard: suppress detections too close together.
        if let Some(last) = self.last_cut_frame {
            if frame_number.saturating_sub(last) < self.config.min_scene_gap {
                return None;
            }
        }
        // Confidence: how far chi2 exceeds the threshold, capped at 1.0.
        let excess = (chi2 - threshold) / threshold.max(1e-12);
        let confidence = (excess as f32).clamp(0.0, 1.0);
        if confidence < self.config.min_confidence {
            return None;
        }
        let high_complexity = entropy > self.config.max_entropy * 0.6;
        self.last_cut_frame = Some(frame_number);
        Some(ComplexitySceneChange {
            frame_number,
            chi_squared: chi2,
            threshold,
            confidence,
            entropy,
            high_complexity,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build a normalised 256-bin histogram from a luma buffer.
fn build_histogram(data: &[u8]) -> [f64; 256] {
    let mut counts = [0u64; 256];
    for &b in data {
        counts[b as usize] += 1;
    }
    let total = data.len() as f64;
    let mut hist = [0.0f64; 256];
    if total > 0.0 {
        for (h, &c) in hist.iter_mut().zip(counts.iter()) {
            *h = c as f64 / total;
        }
    }
    hist
}

/// Shannon entropy of a normalised histogram (base 2, bits).
fn shannon_entropy(hist: &[f64; 256]) -> f64 {
    hist.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.log2())
        .sum()
}

/// χ² distance between two normalised histograms.
///
/// `d(h1, h2) = Σ (h1[i] - h2[i])² / (h1[i] + h2[i] + ε)`
fn chi_squared_distance(a: &[f64; 256], b: &[f64; 256]) -> f64 {
    let eps = 1e-10;
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let diff = ai - bi;
            diff * diff / (ai + bi + eps)
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a synthetic solid-colour frame of given dimensions.
    fn solid_frame(width: u32, height: u32, value: u8) -> Vec<u8> {
        vec![value; (width * height) as usize]
    }

    /// Create a frame with uniform random-ish noise using a simple LCG.
    fn noise_frame(width: u32, height: u32, seed: u64) -> Vec<u8> {
        let mut lcg = seed.wrapping_add(6364136223846793005);
        let mut data = Vec::with_capacity((width * height) as usize);
        for _ in 0..(width * height) {
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            data.push((lcg >> 33) as u8);
        }
        data
    }

    /// Build a ramp frame: pixel[i] = i % 256.
    fn ramp_frame(width: u32, height: u32) -> Vec<u8> {
        (0..(width * height) as usize)
            .map(|i| (i % 256) as u8)
            .collect()
    }

    // ------------------------------------------------------------------
    // 1. Constructor and first-frame seeding
    // ------------------------------------------------------------------
    #[test]
    fn test_default_config_is_valid() {
        let cfg = DetectorConfig::default();
        assert!(cfg.threshold_min < cfg.threshold_max);
        assert!(cfg.ema_alpha > 0.0 && cfg.ema_alpha <= 1.0);
        assert!(cfg.min_confidence >= 0.0 && cfg.min_confidence <= 1.0);
        assert!(cfg.max_entropy > 0.0);
        assert!(cfg.min_scene_gap > 0);
    }

    #[test]
    fn test_first_frame_never_emits_cut() {
        let mut det = AdaptiveComplexityDetector::default_config();
        let frame = solid_frame(16, 16, 128);
        let res = det.push_frame(&frame, 16, 16, 0).expect("first frame ok");
        assert!(res.is_none(), "first frame should not emit a cut");
    }

    // ------------------------------------------------------------------
    // 2. Identical frames → no cut
    // ------------------------------------------------------------------
    #[test]
    fn test_identical_frames_no_cut() {
        let mut det = AdaptiveComplexityDetector::default_config();
        let frame = solid_frame(32, 32, 64);
        det.push_frame(&frame, 32, 32, 0).expect("frame 0 ok");
        let res = det.push_frame(&frame, 32, 32, 1).expect("frame 1 ok");
        assert!(
            res.is_none(),
            "identical consecutive frames must not trigger cut"
        );
    }

    // ------------------------------------------------------------------
    // 3. Hard cut between black and white solid frames
    // ------------------------------------------------------------------
    #[test]
    fn test_hard_cut_black_to_white() {
        let cfg = DetectorConfig {
            min_confidence: 0.01,
            min_scene_gap: 0,
            ..DetectorConfig::default()
        };
        let mut det = AdaptiveComplexityDetector::new(cfg);
        let black = solid_frame(64, 64, 0);
        let white = solid_frame(64, 64, 255);
        det.push_frame(&black, 64, 64, 0).expect("frame 0 ok");
        let res = det.push_frame(&white, 64, 64, 1).expect("frame 1 ok");
        assert!(res.is_some(), "black→white should be detected as a cut");
        let ev = res.unwrap();
        assert_eq!(ev.frame_number, 1);
        assert!(ev.confidence > 0.0);
        assert!(ev.chi_squared > 0.0);
    }

    // ------------------------------------------------------------------
    // 4. Chi-squared is zero for identical histograms
    // ------------------------------------------------------------------
    #[test]
    fn test_chi_squared_zero_identical() {
        let hist = [1.0_f64 / 256.0; 256];
        let d = super::chi_squared_distance(&hist, &hist);
        assert!(
            d < 1e-12,
            "chi-squared of identical histograms should be ~0"
        );
    }

    // ------------------------------------------------------------------
    // 5. Shannon entropy maximum for uniform histogram
    // ------------------------------------------------------------------
    #[test]
    fn test_entropy_max_for_uniform() {
        let hist = [1.0_f64 / 256.0; 256];
        let e = super::shannon_entropy(&hist);
        // log2(256) = 8.0
        let expected = 8.0f64;
        assert!(
            (e - expected).abs() < 1e-6,
            "uniform histogram entropy should be ≈ 8.0 bits, got {e}"
        );
    }

    // ------------------------------------------------------------------
    // 6. Shannon entropy zero for a spike histogram
    // ------------------------------------------------------------------
    #[test]
    fn test_entropy_zero_for_spike() {
        let mut hist = [0.0f64; 256];
        hist[128] = 1.0;
        let e = super::shannon_entropy(&hist);
        assert!(e < 1e-12, "spike histogram entropy must be 0");
    }

    // ------------------------------------------------------------------
    // 7. Adaptive threshold scales with EMA entropy
    // ------------------------------------------------------------------
    #[test]
    fn test_adaptive_threshold_varies_with_entropy() {
        let cfg = DetectorConfig::default();
        let mut det_simple = AdaptiveComplexityDetector::new(cfg.clone());
        let mut det_complex = AdaptiveComplexityDetector::new(cfg.clone());

        // Feed simple (spike) frames to det_simple.
        let spike = solid_frame(8, 8, 128);
        for i in 0..20u64 {
            det_simple.push_frame(&spike, 8, 8, i).unwrap();
        }

        // Feed noisy (high-entropy) frames to det_complex.
        for i in 0..20u64 {
            let noisy = noise_frame(8, 8, i);
            det_complex.push_frame(&noisy, 8, 8, i).unwrap();
        }

        let t_simple = det_simple.current_threshold();
        let t_complex = det_complex.current_threshold();
        assert!(
            t_complex > t_simple,
            "complex content should have a higher threshold ({t_complex}) than simple ({t_simple})"
        );
    }

    // ------------------------------------------------------------------
    // 8. Dimension mismatch error
    // ------------------------------------------------------------------
    #[test]
    fn test_dimension_mismatch_error() {
        let mut det = AdaptiveComplexityDetector::default_config();
        let f1 = solid_frame(8, 8, 100);
        det.push_frame(&f1, 8, 8, 0).expect("first frame ok");
        let f2 = solid_frame(16, 16, 200);
        let err = det.push_frame(&f2, 16, 16, 1);
        assert!(
            matches!(err, Err(AdaptiveDetectError::DimensionMismatch { .. })),
            "should return DimensionMismatch"
        );
    }

    // ------------------------------------------------------------------
    // 9. Data length mismatch error
    // ------------------------------------------------------------------
    #[test]
    fn test_data_length_mismatch_error() {
        let mut det = AdaptiveComplexityDetector::default_config();
        let short = vec![0u8; 10]; // 10 ≠ 8×8=64
        let err = det.push_frame(&short, 8, 8, 0);
        assert!(
            matches!(err, Err(AdaptiveDetectError::DataLengthMismatch { .. })),
            "should return DataLengthMismatch"
        );
    }

    // ------------------------------------------------------------------
    // 10. Min-scene-gap suppresses rapid re-detection
    // ------------------------------------------------------------------
    #[test]
    fn test_min_scene_gap_suppresses_cuts() {
        let cfg = DetectorConfig {
            min_confidence: 0.01,
            min_scene_gap: 5,
            ..DetectorConfig::default()
        };
        let mut det = AdaptiveComplexityDetector::new(cfg);

        // Alternating black/white frames — every transition is a hard cut.
        let black = solid_frame(32, 32, 0);
        let white = solid_frame(32, 32, 255);
        let mut cuts = Vec::new();
        for i in 0..10u64 {
            let frame = if i % 2 == 0 { &black } else { &white };
            if let Ok(Some(ev)) = det.push_frame(frame, 32, 32, i) {
                cuts.push(ev.frame_number);
            }
        }
        // With gap=5 we should not see cuts at consecutive frames.
        for w in cuts.windows(2) {
            assert!(
                w[1] - w[0] >= 5,
                "cuts at {} and {} violate min_scene_gap=5",
                w[0],
                w[1]
            );
        }
    }

    // ------------------------------------------------------------------
    // 11. Ramp frame produces non-trivial entropy
    // ------------------------------------------------------------------
    #[test]
    fn test_ramp_frame_entropy() {
        let frame = ramp_frame(32, 32);
        let hist = build_histogram(&frame);
        let e = shannon_entropy(&hist);
        // A ramp across 256 values should be near-maximum entropy.
        assert!(
            e > 7.0,
            "ramp frame entropy should be high (>7.0 bits), got {e}"
        );
    }

    // ------------------------------------------------------------------
    // 12. EMA seeding and update correctness
    // ------------------------------------------------------------------
    #[test]
    fn test_ema_converges_toward_new_entropy() {
        let cfg = DetectorConfig {
            ema_alpha: 1.0, // full update → EMA = last frame's entropy
            ..DetectorConfig::default()
        };
        let mut det = AdaptiveComplexityDetector::new(cfg);
        let noisy = noise_frame(32, 32, 42);
        let noisy_hist = build_histogram(&noisy);
        let expected_entropy = shannon_entropy(&noisy_hist);

        det.push_frame(&noisy, 32, 32, 0).unwrap();
        det.push_frame(&noisy, 32, 32, 1).unwrap();
        // With alpha=1.0, ema should equal the last frame's entropy.
        let diff = (det.ema_entropy() - expected_entropy).abs();
        assert!(diff < 1e-9, "EMA should equal last entropy with alpha=1.0");
    }
}
