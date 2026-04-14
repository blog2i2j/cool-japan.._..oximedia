//! Silence and low-energy region detection for automatic audio trimming.
//!
//! This module identifies silence regions in audio using RMS energy analysis with
//! configurable hysteresis (attack/release) so that brief transients do not
//! fragment what are perceptually single silences, and provides an [`AutoTrimmer`]
//! that strips leading/trailing silence from a clip.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A contiguous silent region within an audio buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct SilenceRegion {
    /// Index of the first silent sample (inclusive).
    pub start_sample: u64,
    /// Index one past the last silent sample (exclusive).
    pub end_sample: u64,
    /// Root-mean-square level of the region in dBFS (≤ 0).
    pub rms_db: f32,
}

impl SilenceRegion {
    /// Duration in samples.
    #[must_use]
    pub fn duration_samples(&self) -> u64 {
        self.end_sample.saturating_sub(self.start_sample)
    }

    /// Duration in milliseconds given `sample_rate`.
    #[must_use]
    pub fn duration_ms(&self, sample_rate: u32) -> f32 {
        let samples = self.duration_samples() as f32;
        samples / sample_rate as f32 * 1000.0
    }
}

/// Configuration for [`SilenceDetector`].
#[derive(Debug, Clone)]
pub struct SilenceDetectorConfig {
    /// Energy level below which audio is considered silent (dBFS, e.g. −40.0).
    pub threshold_db: f32,
    /// Minimum contiguous duration (ms) for a region to be reported.
    pub min_duration_ms: f32,
    /// Attack time (ms): how quickly the detector transitions *into* silence.
    pub attack_ms: f32,
    /// Release time (ms): how quickly the detector transitions *out of* silence.
    pub release_ms: f32,
    /// Sample rate of the audio being analysed.
    pub sample_rate: u32,
}

impl Default for SilenceDetectorConfig {
    fn default() -> Self {
        Self {
            threshold_db: -40.0,
            min_duration_ms: 100.0,
            attack_ms: 10.0,
            release_ms: 50.0,
            sample_rate: 48_000,
        }
    }
}

impl SilenceDetectorConfig {
    /// Create a new config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the silence threshold in dBFS.
    #[must_use]
    pub fn with_threshold_db(mut self, db: f32) -> Self {
        self.threshold_db = db;
        self
    }

    /// Set the minimum silence duration in milliseconds.
    #[must_use]
    pub fn with_min_duration_ms(mut self, ms: f32) -> Self {
        self.min_duration_ms = ms;
        self
    }

    /// Set attack and release times in milliseconds.
    #[must_use]
    pub fn with_attack_release_ms(mut self, attack: f32, release: f32) -> Self {
        self.attack_ms = attack;
        self.release_ms = release;
        self
    }

    /// Set the sample rate.
    #[must_use]
    pub fn with_sample_rate(mut self, sr: u32) -> Self {
        self.sample_rate = sr;
        self
    }
}

// ---------------------------------------------------------------------------
// SilenceDetector
// ---------------------------------------------------------------------------

/// Detects silence/low-energy regions in a mono audio sample buffer.
///
/// Uses a hop-based RMS analysis followed by hysteresis filtering with
/// independent attack and release time constants, so that short noisy
/// frames within an otherwise silent gap are not split into separate regions.
#[derive(Debug, Clone)]
pub struct SilenceDetector {
    /// Configuration.
    pub config: SilenceDetectorConfig,
}

impl SilenceDetector {
    /// Create a new detector with the given configuration.
    #[must_use]
    pub fn new(config: SilenceDetectorConfig) -> Self {
        Self { config }
    }

    /// Create a detector with default configuration.
    #[must_use]
    pub fn default_detector() -> Self {
        Self::new(SilenceDetectorConfig::default())
    }

    // -- helpers ------------------------------------------------------------

    /// Size (in samples) of the RMS analysis window.
    fn window_samples(&self) -> usize {
        // Use a 10 ms window for a good balance between time resolution and stability.
        let ms = 10.0_f32;
        ((ms / 1000.0) * self.config.sample_rate as f32).round() as usize
    }

    /// Convert attack/release time in ms to a smoothing coefficient per window.
    ///
    /// Returns the coefficient `c` such that the EMA is `y = c * y_prev + (1-c) * x`.
    fn time_to_coeff(&self, time_ms: f32) -> f32 {
        let n_windows = (time_ms / 10.0_f32).max(1.0); // 10 ms per window
        let tau = n_windows;
        (-1.0_f32 / tau).exp()
    }

    /// Compute per-window RMS in dBFS.
    fn compute_rms_frames(&self, samples: &[f32]) -> Vec<f32> {
        let win = self.window_samples().max(1);
        let n_frames = (samples.len() + win - 1) / win;
        let mut frames = Vec::with_capacity(n_frames);

        for i in 0..n_frames {
            let start = i * win;
            let end = (start + win).min(samples.len());
            let slice = &samples[start..end];
            let mean_sq: f32 = slice.iter().map(|&s| s * s).sum::<f32>() / slice.len() as f32;
            let rms = mean_sq.sqrt();
            let db = if rms < 1e-10 {
                -120.0_f32
            } else {
                20.0 * rms.log10()
            };
            frames.push(db);
        }

        frames
    }

    /// Apply hysteresis smoothing to a series of per-frame dBFS values.
    ///
    /// Uses independent attack (into silence) and release (out of silence) coefficients
    /// to smooth sharp energy transitions before thresholding.
    fn apply_hysteresis(&self, frames: &[f32]) -> Vec<f32> {
        if frames.is_empty() {
            return Vec::new();
        }

        let attack_coeff = self.time_to_coeff(self.config.attack_ms);
        let release_coeff = self.time_to_coeff(self.config.release_ms);
        let mut smoothed = Vec::with_capacity(frames.len());
        let mut prev = frames[0];

        smoothed.push(prev);
        for &current in &frames[1..] {
            // Going quieter (attack) uses the attack coefficient; going louder uses release.
            let coeff = if current < prev {
                attack_coeff
            } else {
                release_coeff
            };
            let next = coeff * prev + (1.0 - coeff) * current;
            smoothed.push(next);
            prev = next;
        }

        smoothed
    }

    // -- public API ---------------------------------------------------------

    /// Detect silence regions in `samples`.
    ///
    /// Regions shorter than [`SilenceDetectorConfig::min_duration_ms`] are discarded.
    #[must_use]
    pub fn detect(&self, samples: &[f32]) -> Vec<SilenceRegion> {
        if samples.is_empty() {
            return Vec::new();
        }

        let win = self.window_samples().max(1);
        let frames_db = self.compute_rms_frames(samples);
        let smoothed = self.apply_hysteresis(&frames_db);

        let threshold = self.config.threshold_db;
        let min_samples = ((self.config.min_duration_ms / 1000.0) * self.config.sample_rate as f32)
            .round() as u64;

        let mut regions: Vec<SilenceRegion> = Vec::new();
        let mut in_silence = false;
        let mut silence_start: u64 = 0;
        let mut silence_sum_sq = 0.0_f64;
        let mut silence_frame_count: usize = 0;

        for (i, &db) in smoothed.iter().enumerate() {
            let frame_start = (i * win) as u64;
            let frame_end = (((i + 1) * win).min(samples.len())) as u64;

            if db <= threshold {
                if !in_silence {
                    in_silence = true;
                    silence_start = frame_start;
                    silence_sum_sq = 0.0;
                    silence_frame_count = 0;
                }
                // Accumulate per-frame RMS for the region's average dBFS.
                let lin = 10.0_f64.powf(db as f64 / 20.0);
                silence_sum_sq += lin * lin;
                silence_frame_count += 1;
            } else if in_silence {
                in_silence = false;
                let duration = frame_start.saturating_sub(silence_start);
                if duration >= min_samples {
                    let avg_rms = if silence_frame_count > 0 {
                        (silence_sum_sq / silence_frame_count as f64).sqrt()
                    } else {
                        0.0
                    };
                    let rms_db = if avg_rms < 1e-10 {
                        -120.0_f32
                    } else {
                        (20.0 * avg_rms.log10()) as f32
                    };
                    regions.push(SilenceRegion {
                        start_sample: silence_start,
                        end_sample: frame_start,
                        rms_db,
                    });
                }
                let _ = frame_end; // suppress unused warning
            }
        }

        // Handle trailing silence
        if in_silence {
            let end = samples.len() as u64;
            let duration = end.saturating_sub(silence_start);
            if duration >= min_samples {
                let avg_rms = if silence_frame_count > 0 {
                    (silence_sum_sq / silence_frame_count as f64).sqrt()
                } else {
                    0.0
                };
                let rms_db = if avg_rms < 1e-10 {
                    -120.0_f32
                } else {
                    (20.0 * avg_rms.log10()) as f32
                };
                regions.push(SilenceRegion {
                    start_sample: silence_start,
                    end_sample: end,
                    rms_db,
                });
            }
        }

        regions
    }

    /// Detect speech gaps — silence regions longer than 500 ms.
    ///
    /// This is useful for finding natural pause points between utterances.
    #[must_use]
    pub fn detect_speech_gaps(&self, samples: &[f32]) -> Vec<SilenceRegion> {
        let all = self.detect(samples);
        let min_gap_samples = (0.5 * self.config.sample_rate as f64).round() as u64; // 500 ms
        all.into_iter()
            .filter(|r| r.duration_samples() >= min_gap_samples)
            .collect()
    }
}

impl Default for SilenceDetector {
    fn default() -> Self {
        Self::default_detector()
    }
}

// ---------------------------------------------------------------------------
// AutoTrimmer
// ---------------------------------------------------------------------------

/// Trims leading and trailing silence from an audio sample buffer.
#[derive(Debug, Clone)]
pub struct AutoTrimmer;

impl AutoTrimmer {
    /// Return the index of the first sample whose RMS window exceeds `silence_db`.
    ///
    /// Returns `0` if the entire buffer is non-silent, or `samples.len()` if
    /// the entire buffer is silent.
    #[must_use]
    pub fn trim_leading(samples: &[f32], silence_db: f32) -> usize {
        if samples.is_empty() {
            return 0;
        }
        let win = 480_usize; // ~10 ms at 48 kHz
        let n_frames = (samples.len() + win - 1) / win;
        for i in 0..n_frames {
            let start = i * win;
            let end = (start + win).min(samples.len());
            let slice = &samples[start..end];
            let mean_sq: f32 = slice.iter().map(|&s| s * s).sum::<f32>() / slice.len() as f32;
            let rms = mean_sq.sqrt();
            let db = if rms < 1e-10 {
                -120.0_f32
            } else {
                20.0 * rms.log10()
            };
            if db > silence_db {
                return start;
            }
        }
        samples.len()
    }

    /// Return one past the index of the last sample whose RMS window exceeds `silence_db`.
    ///
    /// Returns `samples.len()` if the entire buffer is non-silent, or `0` if
    /// the entire buffer is silent.
    #[must_use]
    pub fn trim_trailing(samples: &[f32], silence_db: f32) -> usize {
        if samples.is_empty() {
            return 0;
        }
        let win = 480_usize;
        let n_frames = (samples.len() + win - 1) / win;
        for i in (0..n_frames).rev() {
            let start = i * win;
            let end = (start + win).min(samples.len());
            let slice = &samples[start..end];
            let mean_sq: f32 = slice.iter().map(|&s| s * s).sum::<f32>() / slice.len() as f32;
            let rms = mean_sq.sqrt();
            let db = if rms < 1e-10 {
                -120.0_f32
            } else {
                20.0 * rms.log10()
            };
            if db > silence_db {
                return end;
            }
        }
        0
    }

    /// Return the sub-slice of `samples` with leading and trailing silence removed.
    ///
    /// Returns an empty slice if the entire buffer is silent.
    #[must_use]
    pub fn trim<'a>(samples: &'a [f32], silence_db: f32) -> &'a [f32] {
        let start = Self::trim_leading(samples, silence_db);
        if start >= samples.len() {
            return &samples[0..0];
        }
        let end = Self::trim_trailing(samples, silence_db);
        if end <= start {
            return &samples[0..0];
        }
        &samples[start..end]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helpers ----------------------------------------------------------------

    fn make_silence(n: usize) -> Vec<f32> {
        vec![0.0_f32; n]
    }

    fn make_tone(amplitude: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| amplitude * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin())
            .collect()
    }

    fn default_cfg() -> SilenceDetectorConfig {
        SilenceDetectorConfig {
            threshold_db: -40.0,
            min_duration_ms: 50.0,
            attack_ms: 5.0,
            release_ms: 10.0,
            sample_rate: 48_000,
        }
    }

    // SilenceDetector tests --------------------------------------------------

    #[test]
    fn test_all_silence_yields_single_region() {
        let det = SilenceDetector::new(default_cfg());
        let samples = make_silence(48_000); // 1 s of silence
        let regions = det.detect(&samples);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].start_sample, 0);
        assert_eq!(regions[0].end_sample, 48_000);
    }

    #[test]
    fn test_no_silence_yields_empty() {
        let det = SilenceDetector::new(default_cfg());
        let samples = make_tone(0.5, 48_000); // loud tone for 1 s
        let regions = det.detect(&samples);
        assert!(regions.is_empty(), "expected no silence, got {regions:?}");
    }

    #[test]
    fn test_silence_in_middle() {
        let det = SilenceDetector::new(default_cfg());
        let mut samples = make_tone(0.5, 24_000); // 0.5 s tone
        samples.extend(make_silence(24_000)); // 0.5 s silence
        samples.extend(make_tone(0.5, 24_000)); // 0.5 s tone
        let regions = det.detect(&samples);
        assert_eq!(regions.len(), 1);
        let r = &regions[0];
        // The silence should start around 24 000 and end around 48 000
        assert!(r.start_sample >= 22_000, "start={}", r.start_sample);
        assert!(r.end_sample <= 50_000, "end={}", r.end_sample);
        assert!(r.rms_db <= -40.0);
    }

    #[test]
    fn test_min_duration_filters_short_silence() {
        let mut cfg = default_cfg();
        cfg.min_duration_ms = 200.0; // only report silences >= 200 ms
        let det = SilenceDetector::new(cfg);
        let mut samples = make_tone(0.5, 24_000);
        samples.extend(make_silence(2_400)); // only 50 ms silence → filtered out
        samples.extend(make_tone(0.5, 24_000));
        let regions = det.detect(&samples);
        assert!(
            regions.is_empty(),
            "short silence should be filtered: {regions:?}"
        );
    }

    #[test]
    fn test_speech_gap_detection_filters_short() {
        let det = SilenceDetector::new(default_cfg());
        let mut samples = make_tone(0.5, 24_000);
        samples.extend(make_silence(9_600)); // 200 ms gap — below 500 ms threshold
        samples.extend(make_tone(0.5, 24_000));
        let gaps = det.detect_speech_gaps(&samples);
        assert!(
            gaps.is_empty(),
            "200 ms gap should not be a speech gap: {gaps:?}"
        );
    }

    #[test]
    fn test_speech_gap_detection_keeps_long_gap() {
        let det = SilenceDetector::new(default_cfg());
        let mut samples = make_tone(0.5, 24_000);
        samples.extend(make_silence(48_000)); // 1 s gap ≥ 500 ms
        samples.extend(make_tone(0.5, 24_000));
        let gaps = det.detect_speech_gaps(&samples);
        assert_eq!(gaps.len(), 1, "1 s gap should be a speech gap: {gaps:?}");
        assert!(gaps[0].duration_ms(48_000) >= 500.0);
    }

    #[test]
    fn test_empty_samples_yields_no_regions() {
        let det = SilenceDetector::default_detector();
        assert!(det.detect(&[]).is_empty());
        assert!(det.detect_speech_gaps(&[]).is_empty());
    }

    #[test]
    fn test_silence_region_duration() {
        let r = SilenceRegion {
            start_sample: 0,
            end_sample: 48_000,
            rms_db: -80.0,
        };
        assert_eq!(r.duration_samples(), 48_000);
        assert!((r.duration_ms(48_000) - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_rms_db_of_silence_region_is_very_low() {
        let det = SilenceDetector::new(default_cfg());
        let samples = make_silence(48_000);
        let regions = det.detect(&samples);
        if let Some(r) = regions.first() {
            assert!(r.rms_db <= -40.0, "rms_db={}", r.rms_db);
        }
    }

    // AutoTrimmer tests ------------------------------------------------------

    #[test]
    fn test_trim_leading_no_silence() {
        let samples = make_tone(0.5, 1000);
        let idx = AutoTrimmer::trim_leading(&samples, -40.0);
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_trim_leading_all_silence() {
        let samples = make_silence(1000);
        let idx = AutoTrimmer::trim_leading(&samples, -40.0);
        assert_eq!(idx, samples.len());
    }

    #[test]
    fn test_trim_leading_silence_then_tone() {
        let mut samples = make_silence(4800); // 0.1 s silence at 48 kHz
        samples.extend(make_tone(0.5, 48_000));
        let idx = AutoTrimmer::trim_leading(&samples, -40.0);
        assert!(idx <= 4800, "should skip leading silence, idx={idx}");
        assert!(idx < samples.len());
    }

    #[test]
    fn test_trim_trailing_all_silence() {
        let samples = make_silence(1000);
        let idx = AutoTrimmer::trim_trailing(&samples, -40.0);
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_trim_trailing_tone_then_silence() {
        let mut samples = make_tone(0.5, 48_000);
        samples.extend(make_silence(4800)); // trailing silence
        let idx = AutoTrimmer::trim_trailing(&samples, -40.0);
        assert!(idx <= 48_000, "should trim trailing silence, idx={idx}");
        assert!(idx > 0);
    }

    #[test]
    fn test_trim_all_silent_returns_empty() {
        let samples = make_silence(10_000);
        let trimmed = AutoTrimmer::trim(&samples, -40.0);
        assert!(trimmed.is_empty());
    }

    #[test]
    fn test_trim_removes_leading_and_trailing() {
        let mut samples = make_silence(4800);
        let tone = make_tone(0.5, 48_000);
        samples.extend_from_slice(&tone);
        samples.extend(make_silence(4800));
        let trimmed = AutoTrimmer::trim(&samples, -40.0);
        // Should be shorter than original and non-empty
        assert!(!trimmed.is_empty());
        assert!(trimmed.len() < samples.len());
    }
}
