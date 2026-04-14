//! Audio-visual correlation module for detecting synchronisation between audio events
//! and visual changes.
//!
//! Computes cross-correlation between audio onset envelopes and frame-difference
//! visual-change envelopes to find temporal alignment, detect audio-visual sync
//! errors (lip-sync drift), and score scene cuts against music beats — all using
//! patent-free signal-processing techniques.
//!
//! # Algorithms
//! - Normalised cross-correlation (NCC) for lag estimation
//! - Peak picking on audio onset envelope
//! - Frame-difference magnitude as visual onset proxy
//! - Rolling-window Pearson correlation for short-term sync scoring

use crate::error::{SceneError, SceneResult};
use serde::{Deserialize, Serialize};

/// A discrete time-series of evenly-sampled float values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    /// Samples (one per frame or audio analysis window).
    pub samples: Vec<f32>,
    /// Sample rate in Hz (frames per second or audio windows per second).
    pub sample_rate: f32,
}

impl TimeSeries {
    /// Create a new time-series.
    pub fn new(samples: Vec<f32>, sample_rate: f32) -> SceneResult<Self> {
        if sample_rate <= 0.0 {
            return Err(SceneError::InvalidParameter(
                "sample_rate must be > 0".into(),
            ));
        }
        Ok(Self {
            samples,
            sample_rate,
        })
    }

    /// Duration in seconds.
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate
    }

    /// Mean value of the series.
    #[must_use]
    pub fn mean(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.iter().sum::<f32>() / self.samples.len() as f32
    }

    /// Variance of the series.
    #[must_use]
    pub fn variance(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let m = self.mean();
        self.samples.iter().map(|&x| (x - m).powi(2)).sum::<f32>() / self.samples.len() as f32
    }

    /// Standard deviation.
    #[must_use]
    pub fn std_dev(&self) -> f32 {
        self.variance().sqrt()
    }

    /// Normalise samples to zero mean and unit variance in-place.
    pub fn normalise(&mut self) {
        let m = self.mean();
        let s = self.std_dev();
        if s < 1e-9 {
            for v in &mut self.samples {
                *v = 0.0;
            }
        } else {
            for v in &mut self.samples {
                *v = (*v - m) / s;
            }
        }
    }
}

/// A detected audio event (onset or beat).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioEvent {
    /// Time offset in seconds from the start of the analysis window.
    pub time_offset: f32,
    /// Strength of the event (0.0–1.0).
    pub strength: f32,
    /// Event category.
    pub kind: AudioEventKind,
}

/// Category of audio event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioEventKind {
    /// A percussive or transient onset.
    Onset,
    /// A beat from a rhythmic pattern.
    Beat,
    /// A sudden level change (cut to silence or loud passage).
    LevelChange,
}

impl AudioEventKind {
    /// Human-readable label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Onset => "onset",
            Self::Beat => "beat",
            Self::LevelChange => "level-change",
        }
    }
}

/// A detected visual event (scene cut or significant frame change).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualEvent {
    /// Frame index at which the event occurs.
    pub frame_index: u64,
    /// Time offset in seconds.
    pub time_offset: f32,
    /// Change magnitude (0.0–1.0).
    pub magnitude: f32,
}

/// Correlation lag estimate and confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationResult {
    /// Best-matching lag in seconds (positive = audio leads video).
    pub lag_seconds: f32,
    /// Peak normalised cross-correlation coefficient (0.0–1.0).
    pub peak_ncc: f32,
    /// Whether the lag is within an acceptable sync window.
    pub in_sync: bool,
}

/// Short-term (rolling-window) Pearson correlation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollingCorrelation {
    /// Per-window correlation coefficients.
    pub coefficients: Vec<f32>,
    /// Window size in samples.
    pub window_size: usize,
    /// Mean Pearson r over all windows.
    pub mean_r: f32,
    /// Percentage of windows with |r| > threshold.
    pub sync_coverage: f32,
}

/// Configuration for audio-visual correlation analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvCorrelationConfig {
    /// Maximum lag (in seconds) to search in cross-correlation.
    pub max_lag_seconds: f32,
    /// Lag tolerance for "in sync" judgement (in seconds).
    pub sync_tolerance_seconds: f32,
    /// Minimum NCC peak value to consider correlated.
    pub min_ncc_threshold: f32,
    /// Rolling window size in samples for short-term analysis.
    pub rolling_window_samples: usize,
    /// Minimum |r| for a window to count as "in sync".
    pub rolling_sync_threshold: f32,
    /// Minimum peak prominence ratio for onset detection.
    pub onset_peak_ratio: f32,
}

impl Default for AvCorrelationConfig {
    fn default() -> Self {
        Self {
            max_lag_seconds: 0.5,
            sync_tolerance_seconds: 0.04, // ~1 frame at 25 fps
            min_ncc_threshold: 0.3,
            rolling_window_samples: 32,
            rolling_sync_threshold: 0.4,
            onset_peak_ratio: 0.5,
        }
    }
}

impl AvCorrelationConfig {
    /// Validate configuration.
    pub fn validate(&self) -> SceneResult<()> {
        if self.max_lag_seconds <= 0.0 {
            return Err(SceneError::InvalidParameter(
                "max_lag_seconds must be > 0".into(),
            ));
        }
        if self.sync_tolerance_seconds < 0.0 {
            return Err(SceneError::InvalidParameter(
                "sync_tolerance_seconds must be >= 0".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.min_ncc_threshold) {
            return Err(SceneError::InvalidParameter(
                "min_ncc_threshold must be in [0, 1]".into(),
            ));
        }
        if self.rolling_window_samples < 2 {
            return Err(SceneError::InvalidParameter(
                "rolling_window_samples must be >= 2".into(),
            ));
        }
        Ok(())
    }
}

/// Audio-visual correlation analyser.
pub struct AvCorrelator {
    config: AvCorrelationConfig,
}

impl Default for AvCorrelator {
    fn default() -> Self {
        Self::new()
    }
}

impl AvCorrelator {
    /// Create with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: AvCorrelationConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: AvCorrelationConfig) -> SceneResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Compute normalised cross-correlation between audio and visual envelopes.
    ///
    /// Both series must have the same `sample_rate`.
    pub fn cross_correlate(
        &self,
        audio: &TimeSeries,
        visual: &TimeSeries,
    ) -> SceneResult<CorrelationResult> {
        if (audio.sample_rate - visual.sample_rate).abs() > 0.01 {
            return Err(SceneError::InvalidParameter(format!(
                "sample rates differ: audio={} visual={}",
                audio.sample_rate, visual.sample_rate
            )));
        }
        if audio.samples.is_empty() || visual.samples.is_empty() {
            return Err(SceneError::InsufficientData(
                "both time series must be non-empty".into(),
            ));
        }

        let sr = audio.sample_rate;
        let max_lag_samples = (self.config.max_lag_seconds * sr).ceil() as usize;

        // Normalise copies.
        let mut a = audio.clone();
        let mut v = visual.clone();
        a.normalise();
        v.normalise();

        let (best_lag, peak_ncc) = ncc_search(&a.samples, &v.samples, max_lag_samples);

        let lag_seconds = best_lag as f32 / sr;
        let in_sync = lag_seconds.abs() <= self.config.sync_tolerance_seconds
            && peak_ncc >= self.config.min_ncc_threshold;

        Ok(CorrelationResult {
            lag_seconds,
            peak_ncc,
            in_sync,
        })
    }

    /// Compute rolling Pearson correlation between two aligned time-series.
    pub fn rolling_correlation(
        &self,
        a: &TimeSeries,
        b: &TimeSeries,
    ) -> SceneResult<RollingCorrelation> {
        let n = a.samples.len().min(b.samples.len());
        let w = self.config.rolling_window_samples;
        if n < w {
            return Err(SceneError::InsufficientData(format!(
                "need >= {w} samples, got {n}"
            )));
        }

        let mut coefficients = Vec::with_capacity(n - w + 1);
        for start in 0..=(n - w) {
            let r = pearson_r(&a.samples[start..start + w], &b.samples[start..start + w]);
            coefficients.push(r);
        }

        let mean_r = coefficients.iter().sum::<f32>() / coefficients.len() as f32;
        let thresh = self.config.rolling_sync_threshold;
        let sync_windows = coefficients.iter().filter(|&&r| r.abs() >= thresh).count();
        let sync_coverage = sync_windows as f32 / coefficients.len() as f32;

        Ok(RollingCorrelation {
            coefficients,
            window_size: w,
            mean_r,
            sync_coverage,
        })
    }

    /// Detect audio onset events from an energy envelope using peak picking.
    pub fn detect_audio_onsets(&self, envelope: &TimeSeries) -> SceneResult<Vec<AudioEvent>> {
        if envelope.samples.len() < 3 {
            return Err(SceneError::InsufficientData(
                "envelope must have at least 3 samples".into(),
            ));
        }
        let max_val = envelope
            .samples
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let threshold = max_val * self.config.onset_peak_ratio;

        let mut events = Vec::new();
        let samples = &envelope.samples;
        for i in 1..samples.len() - 1 {
            let prev = samples[i - 1];
            let cur = samples[i];
            let next = samples[i + 1];
            if cur > threshold && cur >= prev && cur >= next {
                events.push(AudioEvent {
                    time_offset: i as f32 / envelope.sample_rate,
                    strength: if max_val > 0.0 { cur / max_val } else { 0.0 },
                    kind: AudioEventKind::Onset,
                });
            }
        }
        Ok(events)
    }

    /// Detect visual change events from a frame-difference magnitude series.
    pub fn detect_visual_changes(&self, frame_diffs: &TimeSeries) -> SceneResult<Vec<VisualEvent>> {
        if frame_diffs.samples.len() < 3 {
            return Err(SceneError::InsufficientData(
                "frame_diffs must have at least 3 samples".into(),
            ));
        }
        let max_val = frame_diffs
            .samples
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let threshold = max_val * self.config.onset_peak_ratio;

        let mut events = Vec::new();
        let samples = &frame_diffs.samples;
        for i in 1..samples.len() - 1 {
            let prev = samples[i - 1];
            let cur = samples[i];
            let next = samples[i + 1];
            if cur > threshold && cur >= prev && cur >= next {
                events.push(VisualEvent {
                    frame_index: i as u64,
                    time_offset: i as f32 / frame_diffs.sample_rate,
                    magnitude: if max_val > 0.0 { cur / max_val } else { 0.0 },
                });
            }
        }
        Ok(events)
    }

    /// Align detected audio events to their nearest visual events.
    ///
    /// Returns pairs `(audio_event_index, visual_event_index, lag_seconds)` for
    /// all audio events within `sync_tolerance_seconds` of a visual event.
    #[must_use]
    pub fn align_events(
        &self,
        audio_events: &[AudioEvent],
        visual_events: &[VisualEvent],
    ) -> Vec<(usize, usize, f32)> {
        let tolerance = self.config.sync_tolerance_seconds;
        let mut aligned = Vec::new();
        for (ai, ae) in audio_events.iter().enumerate() {
            let best = visual_events
                .iter()
                .enumerate()
                .map(|(vi, ve)| {
                    let lag = ae.time_offset - ve.time_offset;
                    (vi, lag, lag.abs())
                })
                .filter(|(_, _, dist)| *dist <= tolerance)
                .min_by(|(_, _, d1), (_, _, d2)| {
                    d1.partial_cmp(d2).unwrap_or(std::cmp::Ordering::Equal)
                });
            if let Some((vi, lag, _)) = best {
                aligned.push((ai, vi, lag));
            }
        }
        aligned
    }
}

// ── internal helpers ──────────────────────────────────────────────────────────

/// Compute normalised cross-correlation for lags in `[-max_lag, +max_lag]`.
/// Returns `(best_lag_samples, peak_ncc)`.
fn ncc_search(a: &[f32], b: &[f32], max_lag: usize) -> (i64, f32) {
    let len = a.len().min(b.len());
    if len == 0 {
        return (0, 0.0);
    }

    let max_lag = max_lag.min(len - 1);
    let mut best_ncc = f32::NEG_INFINITY;
    let mut best_lag: i64 = 0;

    for lag in -(max_lag as i64)..=(max_lag as i64) {
        let ncc = ncc_at_lag(a, b, lag, len);
        if ncc > best_ncc {
            best_ncc = ncc;
            best_lag = lag;
        }
    }
    (best_lag, best_ncc.max(0.0).min(1.0))
}

/// Cross-correlation at a specific lag (positive lag = `a` leads `b`).
fn ncc_at_lag(a: &[f32], b: &[f32], lag: i64, len: usize) -> f32 {
    let mut sum_ab = 0.0f32;
    let mut sum_a2 = 0.0f32;
    let mut sum_b2 = 0.0f32;
    let mut count = 0usize;

    for i in 0..len {
        let j = i as i64 + lag;
        if j < 0 || j as usize >= len {
            continue;
        }
        let av = a[i];
        let bv = b[j as usize];
        sum_ab += av * bv;
        sum_a2 += av * av;
        sum_b2 += bv * bv;
        count += 1;
    }

    if count == 0 || sum_a2 == 0.0 || sum_b2 == 0.0 {
        return 0.0;
    }
    sum_ab / (sum_a2.sqrt() * sum_b2.sqrt())
}

/// Pearson correlation coefficient for two equal-length slices.
fn pearson_r(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len()) as f32;
    if n < 2.0 {
        return 0.0;
    }
    let mean_a: f32 = a.iter().sum::<f32>() / n;
    let mean_b: f32 = b.iter().sum::<f32>() / n;
    let mut cov = 0.0f32;
    let mut var_a = 0.0f32;
    let mut var_b = 0.0f32;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let da = ai - mean_a;
        let db = bi - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    if var_a == 0.0 || var_b == 0.0 {
        return 0.0;
    }
    cov / (var_a.sqrt() * var_b.sqrt())
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sine(freq: f32, sr: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
            .collect()
    }

    fn make_impulse_train(period: usize, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| if i % period == 0 { 1.0 } else { 0.0 })
            .collect()
    }

    #[test]
    fn test_timeseries_mean_variance() {
        let ts = TimeSeries::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], 25.0).unwrap();
        assert!((ts.mean() - 3.0).abs() < 1e-5);
        assert!((ts.variance() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_timeseries_invalid_sample_rate() {
        assert!(TimeSeries::new(vec![1.0], 0.0).is_err());
        assert!(TimeSeries::new(vec![1.0], -1.0).is_err());
    }

    #[test]
    fn test_cross_correlate_zero_lag() {
        let corr = AvCorrelator::new();
        let samples: Vec<f32> = make_sine(1.0, 25.0, 100);
        let audio = TimeSeries::new(samples.clone(), 25.0).unwrap();
        let visual = TimeSeries::new(samples, 25.0).unwrap();
        let result = corr.cross_correlate(&audio, &visual).unwrap();
        // Identical signals → lag should be 0
        assert_eq!(result.lag_seconds, 0.0);
        assert!(result.peak_ncc > 0.9, "peak_ncc={}", result.peak_ncc);
    }

    #[test]
    fn test_cross_correlate_sample_rate_mismatch() {
        let corr = AvCorrelator::new();
        let audio = TimeSeries::new(vec![1.0; 50], 25.0).unwrap();
        let visual = TimeSeries::new(vec![1.0; 50], 30.0).unwrap();
        assert!(corr.cross_correlate(&audio, &visual).is_err());
    }

    #[test]
    fn test_detect_audio_onsets() {
        let corr = AvCorrelator::new();
        // Impulse train: peaks at indices 0, 10, 20, 30 …
        let samples = make_impulse_train(10, 50);
        let ts = TimeSeries::new(samples, 25.0).unwrap();
        let events = corr.detect_audio_onsets(&ts).unwrap();
        // Should detect multiple onsets
        assert!(!events.is_empty(), "expected onsets, got none");
    }

    #[test]
    fn test_detect_visual_changes() {
        let corr = AvCorrelator::new();
        let samples = make_impulse_train(8, 48);
        let ts = TimeSeries::new(samples, 24.0).unwrap();
        let events = corr.detect_visual_changes(&ts).unwrap();
        assert!(!events.is_empty());
    }

    #[test]
    fn test_align_events_within_tolerance() {
        let config = AvCorrelationConfig {
            sync_tolerance_seconds: 0.1,
            ..Default::default()
        };
        let corr = AvCorrelator::with_config(config).unwrap();
        let audio_events = vec![
            AudioEvent {
                time_offset: 1.0,
                strength: 0.9,
                kind: AudioEventKind::Onset,
            },
            AudioEvent {
                time_offset: 2.0,
                strength: 0.8,
                kind: AudioEventKind::Beat,
            },
        ];
        let visual_events = vec![
            VisualEvent {
                frame_index: 25,
                time_offset: 1.02,
                magnitude: 0.7,
            },
            VisualEvent {
                frame_index: 55,
                time_offset: 2.5,
                magnitude: 0.6,
            }, // too far
        ];
        let aligned = corr.align_events(&audio_events, &visual_events);
        assert_eq!(aligned.len(), 1, "only one pair within 0.1 s tolerance");
        assert_eq!(aligned[0].0, 0); // audio event 0
        assert_eq!(aligned[0].1, 0); // visual event 0
    }

    #[test]
    fn test_rolling_correlation_all_positive() {
        let corr = AvCorrelator::new();
        let n = 100;
        let signal: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let a = TimeSeries::new(signal.clone(), 25.0).unwrap();
        let b = TimeSeries::new(signal, 25.0).unwrap();
        let rc = corr.rolling_correlation(&a, &b).unwrap();
        // Identical signals → all r ≈ 1
        assert!(rc.mean_r > 0.99, "mean_r={}", rc.mean_r);
        assert!(rc.sync_coverage > 0.9);
    }

    #[test]
    fn test_config_validation() {
        let bad = AvCorrelationConfig {
            max_lag_seconds: -0.1,
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        let bad2 = AvCorrelationConfig {
            rolling_window_samples: 1,
            ..Default::default()
        };
        assert!(bad2.validate().is_err());

        let good = AvCorrelationConfig::default();
        assert!(good.validate().is_ok());
    }

    #[test]
    fn test_pearson_r_perfect_correlation() {
        let a: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let b: Vec<f32> = a.iter().map(|&x| x * 2.0 + 3.0).collect();
        let r = pearson_r(&a, &b);
        assert!((r - 1.0).abs() < 1e-4, "r={r}");
    }
}
