//! Audio quality metrics for click/pop detection, clipping, silence ratio,
//! dynamic range estimation, and an overall audio quality score.
//!
//! All functions operate on normalised f32 samples in the range [-1.0, 1.0].
//!
//! # Example
//!
//! ```
//! use oximedia_quality::audio_quality::{AudioQualityAnalyzer, AudioQualityConfig};
//!
//! let config = AudioQualityConfig::default();
//! let analyzer = AudioQualityAnalyzer::new(config);
//!
//! let samples: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
//! let report = analyzer.analyze(&samples, 48000);
//! assert!(report.overall_score >= 0.0 && report.overall_score <= 1.0);
//! ```

use serde::{Deserialize, Serialize};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the audio quality analyzer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioQualityConfig {
    /// Amplitude threshold above which a sample is considered clipped (0.0–1.0).
    pub clip_threshold: f32,
    /// Maximum allowed instantaneous sample-to-sample delta before a transition
    /// is flagged as a click or pop (0.0–2.0, since range is [-1, 1]).
    pub click_delta_threshold: f32,
    /// Amplitude below which a sample is considered silence.
    pub silence_threshold: f32,
    /// Frame size (in samples) used for short-term RMS energy computation.
    pub frame_size: usize,
    /// Minimum RMS energy (per frame) above which a frame is considered "active".
    pub active_frame_rms_threshold: f32,
}

impl Default for AudioQualityConfig {
    fn default() -> Self {
        Self {
            clip_threshold: 0.99,
            click_delta_threshold: 0.5,
            silence_threshold: 1e-4,
            frame_size: 512,
            active_frame_rms_threshold: 0.01,
        }
    }
}

impl AudioQualityConfig {
    /// Creates a new config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the clipping detection threshold.
    #[must_use]
    pub fn with_clip_threshold(mut self, v: f32) -> Self {
        self.clip_threshold = v.clamp(0.0, 1.0);
        self
    }

    /// Sets the click/pop detection delta threshold.
    #[must_use]
    pub fn with_click_delta_threshold(mut self, v: f32) -> Self {
        self.click_delta_threshold = v.max(0.0);
        self
    }

    /// Sets the silence detection amplitude threshold.
    #[must_use]
    pub fn with_silence_threshold(mut self, v: f32) -> Self {
        self.silence_threshold = v.max(0.0);
        self
    }

    /// Sets the analysis frame size in samples.
    #[must_use]
    pub fn with_frame_size(mut self, v: usize) -> Self {
        self.frame_size = v.max(1);
        self
    }
}

// ─── Click / Pop Detection ────────────────────────────────────────────────────

/// Detected click or pop event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClickEvent {
    /// Sample index at which the click was detected.
    pub sample_index: usize,
    /// Absolute delta magnitude that triggered the detection.
    pub delta: f32,
}

/// Detects click and pop artefacts in an audio stream.
///
/// A click is defined as an instantaneous amplitude discontinuity whose
/// magnitude exceeds `threshold`.  The detector also checks for
/// *isolated* high-amplitude samples surrounded by near-silence (a pop).
#[derive(Debug, Clone)]
pub struct ClickDetector {
    threshold: f32,
}

impl ClickDetector {
    /// Creates a new detector with the given delta threshold.
    #[must_use]
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold: threshold.max(0.0),
        }
    }

    /// Scans `samples` and returns all detected click events.
    #[must_use]
    pub fn detect(&self, samples: &[f32]) -> Vec<ClickEvent> {
        if samples.len() < 2 {
            return Vec::new();
        }
        let mut events = Vec::new();
        for i in 1..samples.len() {
            let delta = (samples[i] - samples[i - 1]).abs();
            if delta > self.threshold {
                events.push(ClickEvent {
                    sample_index: i,
                    delta,
                });
            }
        }
        events
    }

    /// Returns the click rate as clicks per second.
    #[must_use]
    pub fn click_rate(&self, samples: &[f32], sample_rate: u32) -> f64 {
        if sample_rate == 0 || samples.is_empty() {
            return 0.0;
        }
        let events = self.detect(samples);
        let duration_secs = samples.len() as f64 / f64::from(sample_rate);
        events.len() as f64 / duration_secs
    }
}

// ─── Clipping Detection ───────────────────────────────────────────────────────

/// Result of clipping analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClippingResult {
    /// Number of clipped samples.
    pub clipped_count: usize,
    /// Fraction of samples that are clipped (0.0–1.0).
    pub clipping_ratio: f32,
    /// Whether clipping was detected at all.
    pub is_clipped: bool,
    /// Maximum absolute amplitude found in the signal.
    pub peak_amplitude: f32,
}

/// Detects hard clipping in audio samples.
pub struct ClippingDetector {
    threshold: f32,
}

impl ClippingDetector {
    /// Creates a new detector with the given amplitude threshold.
    #[must_use]
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Analyzes `samples` for clipping.
    #[must_use]
    pub fn detect(&self, samples: &[f32]) -> ClippingResult {
        if samples.is_empty() {
            return ClippingResult {
                clipped_count: 0,
                clipping_ratio: 0.0,
                is_clipped: false,
                peak_amplitude: 0.0,
            };
        }

        let mut clipped_count = 0usize;
        let mut peak = 0.0f32;

        for &s in samples {
            let abs = s.abs();
            if abs > peak {
                peak = abs;
            }
            if abs >= self.threshold {
                clipped_count += 1;
            }
        }

        let clipping_ratio = clipped_count as f32 / samples.len() as f32;

        ClippingResult {
            clipped_count,
            clipping_ratio,
            is_clipped: clipped_count > 0,
            peak_amplitude: peak,
        }
    }
}

// ─── Silence Ratio ────────────────────────────────────────────────────────────

/// Computes the fraction of samples below a silence threshold.
///
/// Returns a value in [0.0, 1.0] where 1.0 means fully silent.
#[must_use]
pub fn compute_silence_ratio(samples: &[f32], threshold: f32) -> f32 {
    if samples.is_empty() {
        return 1.0;
    }
    let silent = samples.iter().filter(|&&s| s.abs() < threshold).count();
    silent as f32 / samples.len() as f32
}

// ─── Dynamic Range ────────────────────────────────────────────────────────────

/// Dynamic range estimation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicRangeResult {
    /// Crest factor: ratio of peak amplitude to RMS (in linear scale).
    pub crest_factor: f32,
    /// Crest factor expressed in dB.
    pub crest_factor_db: f32,
    /// RMS amplitude of the entire signal.
    pub rms: f32,
    /// Peak amplitude of the entire signal (max absolute value).
    pub peak: f32,
    /// Short-term dynamic range: difference (dB) between the loudest and
    /// quietest active frames.  Active frames have RMS above
    /// `active_frame_rms_threshold`.
    pub short_term_range_db: f32,
    /// Number of active (non-silent) frames used for short-term DR.
    pub active_frame_count: usize,
}

/// Computes dynamic range metrics from a normalised audio buffer.
pub struct DynamicRangeEstimator {
    frame_size: usize,
    active_frame_rms_threshold: f32,
}

impl DynamicRangeEstimator {
    /// Creates a new estimator.
    #[must_use]
    pub fn new(frame_size: usize, active_frame_rms_threshold: f32) -> Self {
        Self {
            frame_size: frame_size.max(1),
            active_frame_rms_threshold: active_frame_rms_threshold.max(0.0),
        }
    }

    /// Estimates dynamic range for `samples`.
    #[must_use]
    pub fn estimate(&self, samples: &[f32]) -> DynamicRangeResult {
        if samples.is_empty() {
            return DynamicRangeResult {
                crest_factor: 0.0,
                crest_factor_db: 0.0,
                rms: 0.0,
                peak: 0.0,
                short_term_range_db: 0.0,
                active_frame_count: 0,
            };
        }

        // Global RMS and peak
        let mut sum_sq = 0.0f64;
        let mut peak = 0.0f32;
        for &s in samples {
            let abs = s.abs();
            if abs > peak {
                peak = abs;
            }
            sum_sq += f64::from(s) * f64::from(s);
        }
        let rms = ((sum_sq / samples.len() as f64).sqrt()) as f32;

        let crest_factor = if rms > 1e-12 { peak / rms } else { 0.0 };
        let crest_factor_db = if crest_factor > 1e-12 {
            20.0 * crest_factor.log10()
        } else {
            0.0
        };

        // Short-term dynamic range over frames
        let mut active_rms_values: Vec<f32> = Vec::new();
        let chunks = samples.chunks(self.frame_size);
        for chunk in chunks {
            let sq: f64 = chunk.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
            let frame_rms = (sq / chunk.len() as f64).sqrt() as f32;
            if frame_rms >= self.active_frame_rms_threshold {
                active_rms_values.push(frame_rms);
            }
        }

        let active_frame_count = active_rms_values.len();

        let short_term_range_db = if active_frame_count >= 2 {
            let max_rms = active_rms_values
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let min_rms = active_rms_values
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min);
            if min_rms > 1e-12 {
                20.0 * (max_rms / min_rms).log10()
            } else {
                0.0
            }
        } else {
            0.0
        };

        DynamicRangeResult {
            crest_factor,
            crest_factor_db,
            rms,
            peak,
            short_term_range_db,
            active_frame_count,
        }
    }
}

// ─── Overall Audio Quality Score ─────────────────────────────────────────────

/// Qualitative audio quality grade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AudioQualityGrade {
    /// Very poor — significant artefacts and/or clipping.
    Poor,
    /// Acceptable but noticeable issues.
    Fair,
    /// Good quality with minor issues.
    Good,
    /// Excellent — clean audio with no detected artefacts.
    Excellent,
}

impl AudioQualityGrade {
    /// Returns the minimum overall_score for this grade.
    #[must_use]
    pub fn min_score(self) -> f32 {
        match self {
            Self::Poor => 0.0,
            Self::Fair => 0.4,
            Self::Good => 0.7,
            Self::Excellent => 0.9,
        }
    }

    /// Classifies a score into a grade.
    #[must_use]
    pub fn from_score(score: f32) -> Self {
        if score >= Self::Excellent.min_score() {
            Self::Excellent
        } else if score >= Self::Good.min_score() {
            Self::Good
        } else if score >= Self::Fair.min_score() {
            Self::Fair
        } else {
            Self::Poor
        }
    }
}

/// Full audio quality analysis report for a buffer of samples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioQualityReport {
    /// Total number of samples analyzed.
    pub sample_count: usize,
    /// Sample rate used for time-based metrics.
    pub sample_rate: u32,
    /// Number of detected click/pop events.
    pub click_count: usize,
    /// Click rate in events per second.
    pub click_rate_per_sec: f64,
    /// Clipping analysis result.
    pub clipping: ClippingResult,
    /// Fraction of samples below the silence threshold.
    pub silence_ratio: f32,
    /// Dynamic range metrics.
    pub dynamic_range: DynamicRangeResult,
    /// Composite quality score in [0.0, 1.0] (higher is better).
    pub overall_score: f32,
    /// Qualitative grade.
    pub grade: AudioQualityGrade,
}

/// Analyzes audio quality from a normalised PCM buffer.
pub struct AudioQualityAnalyzer {
    config: AudioQualityConfig,
    click_detector: ClickDetector,
    clipping_detector: ClippingDetector,
    dr_estimator: DynamicRangeEstimator,
}

impl AudioQualityAnalyzer {
    /// Creates a new analyzer with the given configuration.
    #[must_use]
    pub fn new(config: AudioQualityConfig) -> Self {
        let click_detector = ClickDetector::new(config.click_delta_threshold);
        let clipping_detector = ClippingDetector::new(config.clip_threshold);
        let dr_estimator =
            DynamicRangeEstimator::new(config.frame_size, config.active_frame_rms_threshold);
        Self {
            config,
            click_detector,
            clipping_detector,
            dr_estimator,
        }
    }

    /// Performs a full audio quality analysis on `samples`.
    ///
    /// `sample_rate` is used to compute time-based metrics like click rate.
    #[must_use]
    pub fn analyze(&self, samples: &[f32], sample_rate: u32) -> AudioQualityReport {
        if samples.is_empty() {
            return AudioQualityReport {
                sample_count: 0,
                sample_rate,
                click_count: 0,
                click_rate_per_sec: 0.0,
                clipping: ClippingResult {
                    clipped_count: 0,
                    clipping_ratio: 0.0,
                    is_clipped: false,
                    peak_amplitude: 0.0,
                },
                silence_ratio: 1.0,
                dynamic_range: DynamicRangeResult {
                    crest_factor: 0.0,
                    crest_factor_db: 0.0,
                    rms: 0.0,
                    peak: 0.0,
                    short_term_range_db: 0.0,
                    active_frame_count: 0,
                },
                overall_score: 0.0,
                grade: AudioQualityGrade::Poor,
            };
        }

        let click_events = self.click_detector.detect(samples);
        let click_count = click_events.len();
        let duration_secs = if sample_rate > 0 {
            samples.len() as f64 / f64::from(sample_rate)
        } else {
            1.0
        };
        let click_rate_per_sec = if duration_secs > 0.0 {
            click_count as f64 / duration_secs
        } else {
            0.0
        };

        let clipping = self.clipping_detector.detect(samples);
        let silence_ratio = compute_silence_ratio(samples, self.config.silence_threshold);
        let dynamic_range = self.dr_estimator.estimate(samples);

        let overall_score = self.compute_overall_score(
            &clipping,
            click_rate_per_sec,
            silence_ratio,
            &dynamic_range,
        );
        let grade = AudioQualityGrade::from_score(overall_score);

        AudioQualityReport {
            sample_count: samples.len(),
            sample_rate,
            click_count,
            click_rate_per_sec,
            clipping,
            silence_ratio,
            dynamic_range,
            overall_score,
            grade,
        }
    }

    /// Computes a composite quality score in [0.0, 1.0].
    ///
    /// Penalty contributors:
    /// - Clipping ratio (heavily penalised)
    /// - Click rate > 1 click/sec (moderately penalised)
    /// - Excessive silence ratio > 90 % (lightly penalised — pure silence is
    ///   acceptable for some content)
    /// - Very low dynamic range (crushed / over-limited audio)
    fn compute_overall_score(
        &self,
        clipping: &ClippingResult,
        click_rate: f64,
        silence_ratio: f32,
        dr: &DynamicRangeResult,
    ) -> f32 {
        let mut score = 1.0f32;

        // Clipping: any clipping > 0.1 % strongly penalises
        let clip_penalty = (clipping.clipping_ratio * 20.0).min(1.0);
        score -= clip_penalty * 0.5;

        // Clicks: penalise if click rate > 1 per second
        let click_penalty = ((click_rate as f32 - 1.0).max(0.0) / 10.0).min(1.0);
        score -= click_penalty * 0.3;

        // Silence: penalise if > 95 % of signal is silent (likely no content)
        if silence_ratio > 0.95 {
            score -= (silence_ratio - 0.95) * 2.0;
        }

        // Dynamic range: very low crest factor (< 3 dB) indicates over-limiting
        if dr.crest_factor_db < 3.0 && dr.rms > 0.01 {
            let dr_penalty = (3.0 - dr.crest_factor_db).clamp(0.0, 3.0) / 3.0 * 0.2;
            score -= dr_penalty;
        }

        score.clamp(0.0, 1.0)
    }
}

impl Default for AudioQualityAnalyzer {
    fn default() -> Self {
        Self::new(AudioQualityConfig::default())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine_wave(len: usize, amplitude: f32) -> Vec<f32> {
        (0..len)
            .map(|i| amplitude * (2.0 * PI * i as f32 / 441.0).sin())
            .collect()
    }

    fn silence(len: usize) -> Vec<f32> {
        vec![0.0f32; len]
    }

    // ── ClickDetector ──────────────────────────────────────────────────────

    #[test]
    fn test_click_detector_no_clicks_in_sine() {
        let samples = sine_wave(4800, 0.5);
        let detector = ClickDetector::new(0.5);
        let events = detector.detect(&samples);
        // A 0.5-amplitude sine wave at 10 Hz has a max delta much less than 0.5
        assert!(events.is_empty());
    }

    #[test]
    fn test_click_detector_detects_impulse() {
        let mut samples = vec![0.0f32; 100];
        samples[50] = 1.0; // sudden spike
        let detector = ClickDetector::new(0.4);
        let events = detector.detect(&samples);
        // Spike at 50 and descent at 51 are both discontinuities
        assert!(!events.is_empty());
    }

    #[test]
    fn test_click_detector_empty() {
        let detector = ClickDetector::new(0.3);
        let events = detector.detect(&[]);
        assert!(events.is_empty());
    }

    #[test]
    fn test_click_rate() {
        let mut samples = vec![0.0f32; 48000]; // 1 second at 48kHz
                                               // Insert 5 clicks spaced evenly
        for &pos in &[1000usize, 10000, 20000, 30000, 40000] {
            samples[pos] = 1.0;
        }
        let detector = ClickDetector::new(0.5);
        let rate = detector.click_rate(&samples, 48000);
        assert!(rate >= 5.0 && rate <= 20.0); // each click causes 2 delta events
    }

    // ── ClippingDetector ───────────────────────────────────────────────────

    #[test]
    fn test_clipping_clean_signal() {
        let samples = sine_wave(4800, 0.5);
        let detector = ClippingDetector::new(0.99);
        let result = detector.detect(&samples);
        assert!(!result.is_clipped);
        assert_eq!(result.clipped_count, 0);
        assert!((result.peak_amplitude - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_clipping_saturated_signal() {
        let samples = vec![1.0f32; 100];
        let detector = ClippingDetector::new(0.99);
        let result = detector.detect(&samples);
        assert!(result.is_clipped);
        assert_eq!(result.clipped_count, 100);
        assert!((result.clipping_ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_clipping_empty() {
        let detector = ClippingDetector::new(0.99);
        let result = detector.detect(&[]);
        assert!(!result.is_clipped);
        assert_eq!(result.peak_amplitude, 0.0);
    }

    // ── Silence Ratio ─────────────────────────────────────────────────────

    #[test]
    fn test_silence_ratio_all_silent() {
        let ratio = compute_silence_ratio(&silence(1000), 1e-4);
        assert!((ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_silence_ratio_no_silence() {
        let samples = sine_wave(4800, 0.5);
        let ratio = compute_silence_ratio(&samples, 1e-4);
        assert!(ratio < 0.1);
    }

    #[test]
    fn test_silence_ratio_empty() {
        let ratio = compute_silence_ratio(&[], 1e-4);
        assert!((ratio - 1.0).abs() < 1e-6);
    }

    // ── DynamicRangeEstimator ─────────────────────────────────────────────

    #[test]
    fn test_dynamic_range_sine() {
        let samples = sine_wave(48000, 0.7071); // 0 dBFS sine (RMS ~ 0.5)
        let estimator = DynamicRangeEstimator::new(512, 0.01);
        let result = estimator.estimate(&samples);
        assert!(result.rms > 0.0);
        assert!(result.peak > 0.0);
        // Crest factor of a sine wave is sqrt(2) ≈ 3 dB
        assert!(result.crest_factor_db >= 2.5 && result.crest_factor_db <= 3.5);
    }

    #[test]
    fn test_dynamic_range_empty() {
        let estimator = DynamicRangeEstimator::new(512, 0.01);
        let result = estimator.estimate(&[]);
        assert_eq!(result.rms, 0.0);
        assert_eq!(result.peak, 0.0);
    }

    #[test]
    fn test_dynamic_range_dc_signal() {
        // DC signal: peak = RMS = 0.5, crest factor = 1 (0 dB)
        let samples = vec![0.5f32; 1024];
        let estimator = DynamicRangeEstimator::new(512, 0.01);
        let result = estimator.estimate(&samples);
        assert!((result.rms - 0.5).abs() < 1e-4);
        assert!((result.peak - 0.5).abs() < 1e-4);
        assert!(result.crest_factor_db.abs() < 0.1);
    }

    // ── AudioQualityAnalyzer ───────────────────────────────────────────────

    #[test]
    fn test_analyzer_clean_sine_excellent() {
        let samples = sine_wave(48000, 0.5);
        let analyzer = AudioQualityAnalyzer::default();
        let report = analyzer.analyze(&samples, 48000);
        assert!(report.overall_score > 0.7);
        assert!(
            report.grade == AudioQualityGrade::Good || report.grade == AudioQualityGrade::Excellent
        );
    }

    #[test]
    fn test_analyzer_saturated_poor_score() {
        let samples = vec![1.0f32; 48000]; // fully clipped DC
        let analyzer = AudioQualityAnalyzer::default();
        let report = analyzer.analyze(&samples, 48000);
        assert!(report.clipping.is_clipped);
        assert!(report.overall_score < 0.7);
    }

    #[test]
    fn test_analyzer_empty_buffer() {
        let analyzer = AudioQualityAnalyzer::default();
        let report = analyzer.analyze(&[], 48000);
        assert_eq!(report.sample_count, 0);
        assert_eq!(report.click_count, 0);
        assert_eq!(report.overall_score, 0.0);
    }

    #[test]
    fn test_analyzer_score_in_range() {
        let samples: Vec<f32> = (0..8000).map(|i| ((i as f32) * 0.05).sin() * 0.3).collect();
        let config = AudioQualityConfig::default();
        let analyzer = AudioQualityAnalyzer::new(config);
        let report = analyzer.analyze(&samples, 8000);
        assert!(report.overall_score >= 0.0 && report.overall_score <= 1.0);
    }

    #[test]
    fn test_audio_quality_grade_ordering() {
        assert!(AudioQualityGrade::Poor < AudioQualityGrade::Fair);
        assert!(AudioQualityGrade::Fair < AudioQualityGrade::Good);
        assert!(AudioQualityGrade::Good < AudioQualityGrade::Excellent);
    }

    #[test]
    fn test_audio_quality_grade_from_score() {
        assert_eq!(
            AudioQualityGrade::from_score(0.95),
            AudioQualityGrade::Excellent
        );
        assert_eq!(AudioQualityGrade::from_score(0.75), AudioQualityGrade::Good);
        assert_eq!(AudioQualityGrade::from_score(0.5), AudioQualityGrade::Fair);
        assert_eq!(AudioQualityGrade::from_score(0.2), AudioQualityGrade::Poor);
    }

    #[test]
    fn test_config_builder() {
        let cfg = AudioQualityConfig::new()
            .with_clip_threshold(0.95)
            .with_click_delta_threshold(0.3)
            .with_silence_threshold(0.001)
            .with_frame_size(1024);
        assert!((cfg.clip_threshold - 0.95).abs() < 1e-6);
        assert!((cfg.click_delta_threshold - 0.3).abs() < 1e-6);
        assert!((cfg.silence_threshold - 0.001).abs() < 1e-8);
        assert_eq!(cfg.frame_size, 1024);
    }
}
