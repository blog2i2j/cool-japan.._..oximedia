#![allow(dead_code)]
//! Temporal uniformity analysis: per-frame luminance tracking and flicker detection.

/// Per-frame luminance measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameLuminance {
    /// Zero-based frame index.
    pub frame_index: u64,
    /// Average luminance (normalised, 0.0–1.0).
    pub average_luma: f64,
    /// Minimum luminance value in the frame.
    pub min_luma: f64,
    /// Maximum luminance value in the frame.
    pub max_luma: f64,
}

impl FrameLuminance {
    /// Creates a new `FrameLuminance`.
    #[must_use]
    pub fn new(frame_index: u64, average_luma: f64, min_luma: f64, max_luma: f64) -> Self {
        Self {
            frame_index,
            average_luma,
            min_luma,
            max_luma,
        }
    }

    /// Returns the absolute variation in average luminance from another frame.
    #[must_use]
    pub fn variation_from(&self, other: &Self) -> f64 {
        (self.average_luma - other.average_luma).abs()
    }

    /// Returns the intra-frame dynamic range (max − min).
    #[must_use]
    pub fn dynamic_range(&self) -> f64 {
        self.max_luma - self.min_luma
    }
}

/// Severity level for detected temporal flicker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FlickerSeverity {
    /// Variation below 0.5 % — imperceptible.
    None,
    /// Variation 0.5–2 % — minor, may be perceptible in dark scenes.
    Minor,
    /// Variation 2–5 % — moderate, visible in most content.
    Moderate,
    /// Variation > 5 % — severe, clearly visible flicker.
    Severe,
}

impl FlickerSeverity {
    /// Converts a max frame-to-frame variation (0.0–1.0) to a severity level.
    #[must_use]
    pub fn from_variation(variation: f64) -> Self {
        if variation < 0.005 {
            Self::None
        } else if variation < 0.02 {
            Self::Minor
        } else if variation < 0.05 {
            Self::Moderate
        } else {
            Self::Severe
        }
    }
}

/// Result of a temporal flicker analysis pass.
#[derive(Debug, Clone)]
pub struct TemporalFlicker {
    /// Maximum frame-to-frame luminance variation observed.
    pub max_variation: f64,
    /// Average frame-to-frame luminance variation.
    pub avg_variation: f64,
    /// Number of frames that exceeded the flicker threshold.
    pub flicker_frame_count: usize,
    /// Total frames analysed.
    pub total_frames: usize,
    /// Variation threshold used to classify a frame as flickering.
    pub threshold: f64,
}

impl TemporalFlicker {
    /// Returns the severity classification for the observed flicker.
    #[must_use]
    pub fn severity(&self) -> FlickerSeverity {
        FlickerSeverity::from_variation(self.max_variation)
    }

    /// Returns `true` if any frames exceeded the threshold.
    #[must_use]
    pub fn has_flicker(&self) -> bool {
        self.flicker_frame_count > 0
    }

    /// Returns the percentage of frames that flickered (0.0–100.0).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn flicker_pct(&self) -> f64 {
        if self.total_frames == 0 {
            return 0.0;
        }
        (self.flicker_frame_count as f64 / self.total_frames as f64) * 100.0
    }
}

/// Accumulates per-frame luminance data and detects temporal flicker.
#[derive(Debug, Clone)]
pub struct TemporalUniformityAnalyzer {
    frames: Vec<FrameLuminance>,
    /// Frame-to-frame variation that constitutes flicker (default 0.02).
    pub flicker_threshold: f64,
}

impl TemporalUniformityAnalyzer {
    /// Creates a new analyzer with the default flicker threshold (2 %).
    #[must_use]
    pub fn new() -> Self {
        Self {
            frames: Vec::new(),
            flicker_threshold: 0.02,
        }
    }

    /// Creates a new analyzer with a custom flicker threshold.
    #[must_use]
    pub fn with_threshold(threshold: f64) -> Self {
        Self {
            frames: Vec::new(),
            flicker_threshold: threshold,
        }
    }

    /// Adds a `FrameLuminance` measurement.
    pub fn add_frame(&mut self, frame: FrameLuminance) {
        self.frames.push(frame);
    }

    /// Returns the number of frames collected so far.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Returns a reference to the collected frame measurements.
    #[must_use]
    pub fn frames(&self) -> &[FrameLuminance] {
        &self.frames
    }

    /// Runs temporal flicker detection on the accumulated frames.
    ///
    /// Returns `None` if fewer than 2 frames have been added.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn detect_flicker(&self) -> Option<TemporalFlicker> {
        if self.frames.len() < 2 {
            return None;
        }

        let mut max_variation = 0.0_f64;
        let mut total_variation = 0.0_f64;
        let mut flicker_frame_count = 0_usize;

        for window in self.frames.windows(2) {
            let var = window[0].variation_from(&window[1]);
            total_variation += var;
            if var > max_variation {
                max_variation = var;
            }
            if var > self.flicker_threshold {
                flicker_frame_count += 1;
            }
        }

        let pair_count = (self.frames.len() - 1) as f64;
        let avg_variation = total_variation / pair_count;

        Some(TemporalFlicker {
            max_variation,
            avg_variation,
            flicker_frame_count,
            total_frames: self.frames.len(),
            threshold: self.flicker_threshold,
        })
    }

    /// Returns the overall average luminance across all frames.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn mean_luminance(&self) -> f64 {
        if self.frames.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.frames.iter().map(|f| f.average_luma).sum();
        sum / self.frames.len() as f64
    }
}

impl Default for TemporalUniformityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- FrameLuminance ---

    #[test]
    fn variation_from_same_frame() {
        let f = FrameLuminance::new(0, 0.5, 0.1, 0.9);
        assert!((f.variation_from(&f) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn variation_from_different_frames() {
        let a = FrameLuminance::new(0, 0.5, 0.1, 0.9);
        let b = FrameLuminance::new(1, 0.6, 0.1, 0.9);
        assert!((a.variation_from(&b) - 0.1).abs() < 1e-12);
    }

    #[test]
    fn dynamic_range_correct() {
        let f = FrameLuminance::new(0, 0.5, 0.1, 0.9);
        assert!((f.dynamic_range() - 0.8).abs() < 1e-12);
    }

    // --- FlickerSeverity ---

    #[test]
    fn severity_none_below_half_percent() {
        assert_eq!(
            FlickerSeverity::from_variation(0.004),
            FlickerSeverity::None
        );
    }

    #[test]
    fn severity_minor_range() {
        assert_eq!(
            FlickerSeverity::from_variation(0.01),
            FlickerSeverity::Minor
        );
    }

    #[test]
    fn severity_moderate_range() {
        assert_eq!(
            FlickerSeverity::from_variation(0.03),
            FlickerSeverity::Moderate
        );
    }

    #[test]
    fn severity_severe_above_five_percent() {
        assert_eq!(
            FlickerSeverity::from_variation(0.06),
            FlickerSeverity::Severe
        );
    }

    // --- TemporalUniformityAnalyzer ---

    #[test]
    fn detect_flicker_returns_none_with_one_frame() {
        let mut analyzer = TemporalUniformityAnalyzer::new();
        analyzer.add_frame(FrameLuminance::new(0, 0.5, 0.1, 0.9));
        assert!(analyzer.detect_flicker().is_none());
    }

    #[test]
    fn detect_flicker_no_variation() {
        let mut analyzer = TemporalUniformityAnalyzer::new();
        for i in 0..5 {
            analyzer.add_frame(FrameLuminance::new(i, 0.5, 0.1, 0.9));
        }
        let flicker = analyzer
            .detect_flicker()
            .expect("flicker detection should succeed");
        assert_eq!(flicker.severity(), FlickerSeverity::None);
        assert!(!flicker.has_flicker());
    }

    #[test]
    fn detect_flicker_large_variation() {
        let mut analyzer = TemporalUniformityAnalyzer::new();
        analyzer.add_frame(FrameLuminance::new(0, 0.4, 0.0, 1.0));
        analyzer.add_frame(FrameLuminance::new(1, 0.9, 0.0, 1.0)); // +0.5 variation
        let flicker = analyzer
            .detect_flicker()
            .expect("flicker detection should succeed");
        assert_eq!(flicker.severity(), FlickerSeverity::Severe);
        assert!(flicker.has_flicker());
    }

    #[test]
    fn flicker_pct_half_frames() {
        let mut analyzer = TemporalUniformityAnalyzer::with_threshold(0.05);
        // Pairs: (0→0.5) variation = 0.5 (flicker), (0.5→0.5) variation = 0 (ok)
        analyzer.add_frame(FrameLuminance::new(0, 0.0, 0.0, 1.0));
        analyzer.add_frame(FrameLuminance::new(1, 0.5, 0.0, 1.0));
        analyzer.add_frame(FrameLuminance::new(2, 0.5, 0.0, 1.0));
        let flicker = analyzer
            .detect_flicker()
            .expect("flicker detection should succeed");
        // 1 out of 2 consecutive pairs flickered; total_frames = 3
        assert!(flicker.flicker_pct() > 0.0);
    }

    #[test]
    fn mean_luminance_correct() {
        let mut analyzer = TemporalUniformityAnalyzer::new();
        analyzer.add_frame(FrameLuminance::new(0, 0.2, 0.0, 1.0));
        analyzer.add_frame(FrameLuminance::new(1, 0.8, 0.0, 1.0));
        assert!((analyzer.mean_luminance() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn mean_luminance_empty_is_zero() {
        let analyzer = TemporalUniformityAnalyzer::new();
        assert_eq!(analyzer.mean_luminance(), 0.0);
    }

    #[test]
    fn frame_count_tracks_additions() {
        let mut analyzer = TemporalUniformityAnalyzer::new();
        assert_eq!(analyzer.frame_count(), 0);
        analyzer.add_frame(FrameLuminance::new(0, 0.5, 0.0, 1.0));
        assert_eq!(analyzer.frame_count(), 1);
    }
}
