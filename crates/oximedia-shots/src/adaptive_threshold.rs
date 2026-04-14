//! Adaptive threshold tuning for cut detection based on content complexity.
//!
//! In practice, a fixed histogram-difference threshold for hard-cut detection
//! performs poorly across diverse content types:
//!
//! - **Action / sport** — rapid motion, high inter-frame differences even
//!   within a shot; a low threshold produces false positives.
//! - **Dialogue / interview** — slow movement; a high threshold misses genuine
//!   cuts between talking heads.
//! - **Animation / VFX** — abrupt palette changes within a shot; thresholds
//!   must account for temporal variance patterns.
//!
//! This module implements an [`AdaptiveThresholdTuner`] that:
//!
//! 1. **Estimates content complexity** from a sliding window of per-frame
//!    inter-frame difference (IFD) statistics.
//! 2. **Classifies the local content type** as `Action`, `Dialogue`, or
//!    `General` based on those statistics.
//! 3. **Recommends a cut threshold** that adjusts upward for high-motion
//!    content and downward for static content, keeping the false-positive
//!    rate roughly stable across genres.
//!
//! The tuner is purely statistical and requires no machine-learning runtime.
//! It maintains a circular buffer of recent IFD values and derives threshold
//! recommendations using robust statistics (median + scaled MAD).

use crate::error::{ShotError, ShotResult};
use crate::frame_buffer::FrameBuffer;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Content complexity classification for the local temporal window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContentComplexity {
    /// Fast action, sport, or VFX — high inter-frame variance.
    Action,
    /// Talking-head, interview, or slow drama — low inter-frame variance.
    Dialogue,
    /// Neither extreme — mixed or general content.
    General,
}

impl ContentComplexity {
    /// Human-readable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Action => "Action",
            Self::Dialogue => "Dialogue",
            Self::General => "General",
        }
    }
}

/// Recommended threshold and associated diagnostics from the tuner.
#[derive(Debug, Clone)]
pub struct ThresholdRecommendation {
    /// Recommended hard-cut threshold (0.0–1.0).
    pub threshold: f32,
    /// Detected content complexity class driving this recommendation.
    pub complexity: ContentComplexity,
    /// Window median IFD (inter-frame difference), 0.0–1.0.
    pub window_median_ifd: f32,
    /// Window MAD (median absolute deviation) of IFD, 0.0–1.0.
    pub window_mad_ifd: f32,
    /// Number of frames in the current window.
    pub window_size: usize,
    /// Multiplier applied to the base threshold for the detected complexity.
    pub complexity_multiplier: f32,
}

/// Configuration for [`AdaptiveThresholdTuner`].
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Base (default) cut detection threshold when content is `General`.
    pub base_threshold: f32,
    /// Multiplier applied to `base_threshold` for `Action` content.
    pub action_multiplier: f32,
    /// Multiplier applied to `base_threshold` for `Dialogue` content.
    pub dialogue_multiplier: f32,
    /// Window size (number of recent IFD samples to keep).
    pub window_size: usize,
    /// IFD median above this value classifies content as `Action`.
    pub action_ifd_threshold: f32,
    /// IFD median below this value classifies content as `Dialogue`.
    pub dialogue_ifd_threshold: f32,
    /// Minimum allowed recommended threshold.
    pub min_threshold: f32,
    /// Maximum allowed recommended threshold.
    pub max_threshold: f32,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            base_threshold: 0.30,
            action_multiplier: 1.60,
            dialogue_multiplier: 0.65,
            window_size: 30,
            action_ifd_threshold: 0.12,
            dialogue_ifd_threshold: 0.03,
            min_threshold: 0.10,
            max_threshold: 0.85,
        }
    }
}

// ---------------------------------------------------------------------------
// Tuner
// ---------------------------------------------------------------------------

/// Adaptive threshold tuner for cut detection.
///
/// Maintains a fixed-capacity circular buffer of recent inter-frame
/// difference (IFD) measurements. On each call to [`push_ifd`] the buffer
/// is updated; calling [`recommend`] derives the current best threshold.
///
/// [`push_ifd`]: AdaptiveThresholdTuner::push_ifd
/// [`recommend`]: AdaptiveThresholdTuner::recommend
pub struct AdaptiveThresholdTuner {
    config: AdaptiveConfig,
    /// Circular buffer of recent IFD values.
    window: Vec<f32>,
    /// Write head in the circular buffer.
    head: usize,
    /// How many entries have been written (saturates at `config.window_size`).
    filled: usize,
}

impl Default for AdaptiveThresholdTuner {
    fn default() -> Self {
        Self::new(AdaptiveConfig::default())
    }
}

impl AdaptiveThresholdTuner {
    /// Create a new tuner with the given configuration.
    #[must_use]
    pub fn new(config: AdaptiveConfig) -> Self {
        let cap = config.window_size.max(1);
        Self {
            window: vec![0.0_f32; cap],
            head: 0,
            filled: 0,
            config,
        }
    }

    /// Record a new inter-frame difference value.
    ///
    /// `ifd` should be in [0, 1] — typically the mean absolute pixel
    /// difference normalised by 255.
    ///
    /// # Errors
    ///
    /// Returns [`ShotError::InvalidParameters`] if `ifd` is not finite.
    pub fn push_ifd(&mut self, ifd: f32) -> ShotResult<()> {
        if !ifd.is_finite() {
            return Err(ShotError::InvalidParameters(format!(
                "IFD value must be finite, got {ifd}"
            )));
        }
        let cap = self.window.len();
        self.window[self.head % cap] = ifd.clamp(0.0, 1.0);
        self.head = self.head.wrapping_add(1);
        self.filled = self.filled.saturating_add(1).min(cap);
        Ok(())
    }

    /// Compute an IFD between two frames and record it.
    ///
    /// Uses the mean absolute luminance difference (normalised to [0, 1]).
    ///
    /// # Errors
    ///
    /// Returns [`ShotError::InvalidFrame`] if frames have incompatible
    /// dimensions or fewer than 3 channels.
    pub fn push_frames(&mut self, prev: &FrameBuffer, curr: &FrameBuffer) -> ShotResult<f32> {
        let ifd = mean_abs_luma_diff(prev, curr)?;
        self.push_ifd(ifd)?;
        Ok(ifd)
    }

    /// Recommend a cut threshold based on the current window.
    ///
    /// If fewer than 3 IFD values have been recorded the base threshold is
    /// returned directly (not enough data to adapt).
    #[must_use]
    pub fn recommend(&self) -> ThresholdRecommendation {
        let count = self.filled;
        if count < 3 {
            return ThresholdRecommendation {
                threshold: self.config.base_threshold,
                complexity: ContentComplexity::General,
                window_median_ifd: 0.0,
                window_mad_ifd: 0.0,
                window_size: count,
                complexity_multiplier: 1.0,
            };
        }

        let samples = self.current_samples();
        let median = robust_median(&samples);
        let mad = robust_mad(&samples, median);

        let complexity = self.classify_complexity(median);
        let multiplier = match complexity {
            ContentComplexity::Action => self.config.action_multiplier,
            ContentComplexity::Dialogue => self.config.dialogue_multiplier,
            ContentComplexity::General => 1.0,
        };

        // The recommended threshold scales with the base threshold and the
        // complexity multiplier, but also widens when the MAD is high (noisy
        // content) using a +0.5×MAD bias.
        let raw = self.config.base_threshold * multiplier + 0.5 * mad;
        let threshold = raw.clamp(self.config.min_threshold, self.config.max_threshold);

        ThresholdRecommendation {
            threshold,
            complexity,
            window_median_ifd: median,
            window_mad_ifd: mad,
            window_size: count,
            complexity_multiplier: multiplier,
        }
    }

    /// Reset the tuner state (clear the window).
    pub fn reset(&mut self) {
        for v in &mut self.window {
            *v = 0.0;
        }
        self.head = 0;
        self.filled = 0;
    }

    /// Number of IFD samples currently in the window.
    #[must_use]
    pub fn window_fill(&self) -> usize {
        self.filled
    }

    /// Access the current configuration.
    #[must_use]
    pub fn config(&self) -> &AdaptiveConfig {
        &self.config
    }

    // ---- Private helpers ----

    /// Return a sorted copy of the current valid samples.
    fn current_samples(&self) -> Vec<f32> {
        let count = self.filled;
        let cap = self.window.len();
        let mut out = Vec::with_capacity(count);
        // Read the `count` most-recent values in chronological order.
        // `head` points to the next write position (oldest if full).
        let start = if count < cap { 0 } else { self.head % cap };
        for i in 0..count {
            out.push(self.window[(start + i) % cap]);
        }
        out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    fn classify_complexity(&self, median_ifd: f32) -> ContentComplexity {
        if median_ifd >= self.config.action_ifd_threshold {
            ContentComplexity::Action
        } else if median_ifd <= self.config.dialogue_ifd_threshold {
            ContentComplexity::Dialogue
        } else {
            ContentComplexity::General
        }
    }
}

// ---------------------------------------------------------------------------
// Statistical helpers
// ---------------------------------------------------------------------------

/// Median of a *sorted* slice (panics-free; returns 0 for empty).
fn robust_median(sorted: &[f32]) -> f32 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Median absolute deviation from the median of a *sorted* slice.
fn robust_mad(sorted: &[f32], median: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let mut deviations: Vec<f32> = sorted.iter().map(|&v| (v - median).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    robust_median(&deviations)
}

// ---------------------------------------------------------------------------
// Frame IFD computation
// ---------------------------------------------------------------------------

/// Compute the mean absolute luminance difference between two frames.
///
/// Luminance uses BT.601 coefficients. The result is normalised to [0, 1]
/// by dividing by 255.
///
/// # Errors
///
/// Returns [`ShotError::InvalidFrame`] if dimensions do not match or if
/// either frame has fewer than 3 channels.
pub fn mean_abs_luma_diff(prev: &FrameBuffer, curr: &FrameBuffer) -> ShotResult<f32> {
    let (h1, w1, c1) = prev.dim();
    let (h2, w2, c2) = curr.dim();
    if c1 < 3 {
        return Err(ShotError::InvalidFrame(
            "prev frame must have at least 3 channels".to_string(),
        ));
    }
    if c2 < 3 {
        return Err(ShotError::InvalidFrame(
            "curr frame must have at least 3 channels".to_string(),
        ));
    }
    if h1 != h2 || w1 != w2 {
        return Err(ShotError::InvalidFrame(format!(
            "frame dimensions must match: ({h1},{w1}) vs ({h2},{w2})"
        )));
    }

    let n = h1 * w1;
    if n == 0 {
        return Ok(0.0);
    }

    let mut sum = 0.0_f64;
    for y in 0..h1 {
        for x in 0..w1 {
            let luma1 = f64::from(prev.get(y, x, 0)) * 0.299
                + f64::from(prev.get(y, x, 1)) * 0.587
                + f64::from(prev.get(y, x, 2)) * 0.114;
            let luma2 = f64::from(curr.get(y, x, 0)) * 0.299
                + f64::from(curr.get(y, x, 1)) * 0.587
                + f64::from(curr.get(y, x, 2)) * 0.114;
            sum += (luma1 - luma2).abs();
        }
    }

    Ok((sum / (n as f64 * 255.0)) as f32)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_buffer::FrameBuffer;

    // ---- Helpers ----

    fn flat_frame(h: usize, w: usize, v: u8) -> FrameBuffer {
        FrameBuffer::from_elem(h, w, 3, v)
    }

    fn push_n(tuner: &mut AdaptiveThresholdTuner, val: f32, n: usize) {
        for _ in 0..n {
            tuner.push_ifd(val).unwrap();
        }
    }

    // ---- ContentComplexity ----

    #[test]
    fn test_complexity_name() {
        assert_eq!(ContentComplexity::Action.name(), "Action");
        assert_eq!(ContentComplexity::Dialogue.name(), "Dialogue");
        assert_eq!(ContentComplexity::General.name(), "General");
    }

    // ---- AdaptiveConfig defaults ----

    #[test]
    fn test_config_defaults_sensible() {
        let cfg = AdaptiveConfig::default();
        assert!(cfg.base_threshold > 0.0 && cfg.base_threshold < 1.0);
        assert!(cfg.action_multiplier > 1.0);
        assert!(cfg.dialogue_multiplier < 1.0);
        assert!(cfg.window_size >= 10);
        assert!(cfg.min_threshold < cfg.max_threshold);
    }

    // ---- push_ifd errors ----

    #[test]
    fn test_push_ifd_nan_rejected() {
        let mut tuner = AdaptiveThresholdTuner::default();
        assert!(tuner.push_ifd(f32::NAN).is_err());
    }

    #[test]
    fn test_push_ifd_inf_rejected() {
        let mut tuner = AdaptiveThresholdTuner::default();
        assert!(tuner.push_ifd(f32::INFINITY).is_err());
    }

    // ---- Cold start returns base threshold ----

    #[test]
    fn test_recommend_cold_returns_base() {
        let tuner = AdaptiveThresholdTuner::default();
        let rec = tuner.recommend();
        assert_eq!(rec.complexity, ContentComplexity::General);
        assert!((rec.threshold - AdaptiveConfig::default().base_threshold).abs() < 1e-5);
        assert_eq!(rec.window_size, 0);
    }

    // ---- Action content raises threshold ----

    #[test]
    fn test_recommend_action_raises_threshold() {
        let mut tuner = AdaptiveThresholdTuner::default();
        // Push many high-IFD values (action content)
        push_n(&mut tuner, 0.20, 20);
        let rec = tuner.recommend();
        assert_eq!(rec.complexity, ContentComplexity::Action);
        assert!(
            rec.threshold > AdaptiveConfig::default().base_threshold,
            "action threshold should be higher than base: {}",
            rec.threshold
        );
    }

    // ---- Dialogue content lowers threshold ----

    #[test]
    fn test_recommend_dialogue_lowers_threshold() {
        let mut tuner = AdaptiveThresholdTuner::default();
        // Push many low-IFD values (dialogue content)
        push_n(&mut tuner, 0.01, 20);
        let rec = tuner.recommend();
        assert_eq!(rec.complexity, ContentComplexity::Dialogue);
        assert!(
            rec.threshold < AdaptiveConfig::default().base_threshold,
            "dialogue threshold should be lower than base: {}",
            rec.threshold
        );
    }

    // ---- Threshold stays in [min, max] bounds ----

    #[test]
    fn test_threshold_clamped_to_bounds() {
        let cfg = AdaptiveConfig {
            action_multiplier: 100.0, // extreme to test clamping
            ..AdaptiveConfig::default()
        };
        let mut tuner = AdaptiveThresholdTuner::new(cfg.clone());
        push_n(&mut tuner, 0.50, 20);
        let rec = tuner.recommend();
        assert!(
            rec.threshold >= cfg.min_threshold && rec.threshold <= cfg.max_threshold,
            "threshold out of configured bounds: {}",
            rec.threshold
        );
    }

    // ---- Reset clears state ----

    #[test]
    fn test_reset_clears_window() {
        let mut tuner = AdaptiveThresholdTuner::default();
        push_n(&mut tuner, 0.20, 15);
        assert_eq!(tuner.window_fill(), 15);
        tuner.reset();
        assert_eq!(tuner.window_fill(), 0);
        let rec = tuner.recommend();
        assert_eq!(rec.window_size, 0);
    }

    // ---- push_frames computes IFD ----

    #[test]
    fn test_push_frames_identical_gives_zero_ifd() {
        let mut tuner = AdaptiveThresholdTuner::default();
        let f1 = flat_frame(32, 32, 100);
        let f2 = flat_frame(32, 32, 100);
        let ifd = tuner.push_frames(&f1, &f2).expect("identical frames");
        assert!(ifd < 1e-5, "identical frames should have near-zero IFD");
    }

    #[test]
    fn test_push_frames_extreme_diff_gives_high_ifd() {
        let mut tuner = AdaptiveThresholdTuner::default();
        let f1 = flat_frame(32, 32, 0);
        let f2 = flat_frame(32, 32, 255);
        let ifd = tuner.push_frames(&f1, &f2).expect("extreme diff frames");
        assert!(
            ifd > 0.9,
            "max-contrast frames should have IFD near 1.0: {ifd}"
        );
    }

    #[test]
    fn test_push_frames_dimension_mismatch_error() {
        let mut tuner = AdaptiveThresholdTuner::default();
        let f1 = flat_frame(32, 32, 128);
        let f2 = flat_frame(16, 16, 128);
        assert!(tuner.push_frames(&f1, &f2).is_err());
    }

    // ---- mean_abs_luma_diff standalone ----

    #[test]
    fn test_mean_abs_luma_diff_few_channels_error() {
        let f1 = FrameBuffer::zeros(16, 16, 1);
        let f2 = FrameBuffer::zeros(16, 16, 1);
        assert!(mean_abs_luma_diff(&f1, &f2).is_err());
    }

    // ---- circular buffer overwrites oldest ----

    #[test]
    fn test_circular_buffer_capacity() {
        let cfg = AdaptiveConfig {
            window_size: 5,
            ..AdaptiveConfig::default()
        };
        let mut tuner = AdaptiveThresholdTuner::new(cfg);
        // Fill beyond capacity: oldest should be overwritten
        for i in 0..10 {
            tuner.push_ifd(i as f32 * 0.01).expect("valid ifd in test");
        }
        assert_eq!(
            tuner.window_fill(),
            5,
            "window should be capped at capacity"
        );
    }

    // ---- robust_median ----

    #[test]
    fn test_robust_median_odd() {
        let sorted = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        assert!((robust_median(&sorted) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_robust_median_even() {
        let sorted = vec![1.0_f32, 2.0, 3.0, 4.0];
        assert!((robust_median(&sorted) - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_robust_median_empty() {
        assert!((robust_median(&[]) - 0.0).abs() < 1e-6);
    }

    // ---- robust_mad ----

    #[test]
    fn test_robust_mad_constant_series() {
        let sorted = vec![0.5_f32; 10];
        let mad = robust_mad(&sorted, 0.5);
        assert!(mad < 1e-6, "constant series MAD should be ~0");
    }
}
