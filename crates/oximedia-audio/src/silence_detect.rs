#![allow(dead_code)]
//! Silence detection and audio segmentation.
//!
//! This module detects silent regions in audio signals, enabling:
//!
//! - **Silence trimming**: Remove leading and trailing silence from recordings.
//! - **Audio segmentation**: Split audio at silence boundaries for chapter detection.
//! - **Voice activity detection (VAD)**: Distinguish speech from non-speech.
//! - **Pause detection**: Find pauses in spoken word content.
//!
//! # Algorithm
//!
//! The detector uses a combination of RMS level measurement, configurable
//! thresholds, and minimum duration requirements to classify audio segments
//! as silence or activity.

/// A detected region of silence or activity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioRegion {
    /// Start position in samples.
    pub start_sample: usize,
    /// End position in samples (exclusive).
    pub end_sample: usize,
    /// Whether this region is silence.
    pub is_silence: bool,
    /// Average RMS level of the region in dB.
    pub avg_level_db: f64,
    /// Peak level of the region in dB.
    pub peak_level_db: f64,
}

impl AudioRegion {
    /// Creates a new audio region.
    pub fn new(
        start_sample: usize,
        end_sample: usize,
        is_silence: bool,
        avg_level_db: f64,
        peak_level_db: f64,
    ) -> Self {
        Self {
            start_sample,
            end_sample,
            is_silence,
            avg_level_db,
            peak_level_db,
        }
    }

    /// Returns the duration in samples.
    pub fn duration_samples(&self) -> usize {
        self.end_sample.saturating_sub(self.start_sample)
    }

    /// Returns the duration in seconds at the given sample rate.
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_secs(&self, sample_rate: f64) -> f64 {
        self.duration_samples() as f64 / sample_rate
    }
}

/// Configuration for silence detection.
#[derive(Debug, Clone)]
pub struct SilenceDetectConfig {
    /// Silence threshold in dB (RMS level below this is considered silence).
    pub threshold_db: f64,
    /// Minimum silence duration in seconds to register as a silence region.
    pub min_silence_secs: f64,
    /// Minimum activity duration in seconds to register as an active region.
    pub min_activity_secs: f64,
    /// Analysis window size in samples for RMS computation.
    pub window_size: usize,
    /// Hop size in samples between analysis windows.
    pub hop_size: usize,
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Padding in seconds to add around activity boundaries.
    pub padding_secs: f64,
}

impl Default for SilenceDetectConfig {
    fn default() -> Self {
        Self {
            threshold_db: -40.0,
            min_silence_secs: 0.3,
            min_activity_secs: 0.1,
            window_size: 1024,
            hop_size: 512,
            sample_rate: 48000.0,
            padding_secs: 0.01,
        }
    }
}

/// Silence detector and audio segmenter.
#[derive(Debug, Clone)]
pub struct SilenceDetector {
    /// Configuration.
    config: SilenceDetectConfig,
}

impl SilenceDetector {
    /// Creates a new silence detector with the given configuration.
    pub fn new(config: SilenceDetectConfig) -> Self {
        Self { config }
    }

    /// Creates a silence detector with default settings.
    pub fn default_detector() -> Self {
        Self::new(SilenceDetectConfig::default())
    }

    /// Detects silence and activity regions in the given audio samples.
    #[allow(clippy::cast_precision_loss)]
    pub fn detect(&self, samples: &[f64]) -> Vec<AudioRegion> {
        if samples.is_empty() {
            return Vec::new();
        }

        let min_silence_samples = (self.config.min_silence_secs * self.config.sample_rate) as usize;
        let min_activity_samples =
            (self.config.min_activity_secs * self.config.sample_rate) as usize;

        // Compute per-window RMS levels
        let mut window_silent = Vec::new();
        let mut pos = 0;
        while pos + self.config.window_size <= samples.len() {
            let window = &samples[pos..pos + self.config.window_size];
            let rms_db = rms_to_db(compute_rms(window));
            window_silent.push(rms_db < self.config.threshold_db);
            pos += self.config.hop_size;
        }

        if window_silent.is_empty() {
            // Entire signal is shorter than one window
            let rms_db = rms_to_db(compute_rms(samples));
            let peak_db = peak_to_db(compute_peak(samples));
            let is_silence = rms_db < self.config.threshold_db;
            return vec![AudioRegion::new(
                0,
                samples.len(),
                is_silence,
                rms_db,
                peak_db,
            )];
        }

        // Build raw regions from window classifications
        let mut raw_regions: Vec<(usize, usize, bool)> = Vec::new();
        let mut region_start = 0;
        let mut current_silent = window_silent[0];

        for (i, &silent) in window_silent.iter().enumerate().skip(1) {
            if silent != current_silent {
                let start_sample = region_start * self.config.hop_size;
                let end_sample = (i * self.config.hop_size).min(samples.len());
                raw_regions.push((start_sample, end_sample, current_silent));
                region_start = i;
                current_silent = silent;
            }
        }
        // Final region
        let start_sample = region_start * self.config.hop_size;
        raw_regions.push((start_sample, samples.len(), current_silent));

        // Merge short regions into their neighbors
        let mut merged: Vec<(usize, usize, bool)> = Vec::new();
        for (start, end, is_silence) in &raw_regions {
            let duration = end - start;
            let min_dur = if *is_silence {
                min_silence_samples
            } else {
                min_activity_samples
            };
            if duration < min_dur && !merged.is_empty() {
                // Merge with previous
                if let Some(last) = merged.last_mut() {
                    last.1 = *end;
                }
            } else {
                // Check if we can merge with previous of same type
                if let Some(last) = merged.last_mut() {
                    if last.2 == *is_silence {
                        last.1 = *end;
                    } else {
                        merged.push((*start, *end, *is_silence));
                    }
                } else {
                    merged.push((*start, *end, *is_silence));
                }
            }
        }

        // Convert to AudioRegions with level measurements
        merged
            .iter()
            .map(|&(start, end, is_silence)| {
                let segment = &samples[start..end.min(samples.len())];
                let rms_db = rms_to_db(compute_rms(segment));
                let peak_db = peak_to_db(compute_peak(segment));
                AudioRegion::new(start, end, is_silence, rms_db, peak_db)
            })
            .collect()
    }

    /// Returns only the silence regions.
    pub fn detect_silence(&self, samples: &[f64]) -> Vec<AudioRegion> {
        self.detect(samples)
            .into_iter()
            .filter(|r| r.is_silence)
            .collect()
    }

    /// Returns only the activity (non-silence) regions.
    pub fn detect_activity(&self, samples: &[f64]) -> Vec<AudioRegion> {
        self.detect(samples)
            .into_iter()
            .filter(|r| !r.is_silence)
            .collect()
    }

    /// Finds the first non-silent sample position (for trimming leading silence).
    pub fn find_leading_edge(&self, samples: &[f64]) -> usize {
        let regions = self.detect(samples);
        for region in &regions {
            if !region.is_silence {
                return region.start_sample;
            }
        }
        samples.len()
    }

    /// Finds the last non-silent sample position (for trimming trailing silence).
    pub fn find_trailing_edge(&self, samples: &[f64]) -> usize {
        let regions = self.detect(samples);
        for region in regions.iter().rev() {
            if !region.is_silence {
                return region.end_sample;
            }
        }
        0
    }

    /// Returns the trim points (start, end) to remove leading/trailing silence.
    pub fn trim_points(&self, samples: &[f64]) -> (usize, usize) {
        let start = self.find_leading_edge(samples);
        let end = self.find_trailing_edge(samples);
        if start >= end {
            (0, samples.len())
        } else {
            (start, end)
        }
    }
}

/// Computes the RMS (Root Mean Square) of a sample buffer.
#[allow(clippy::cast_precision_loss)]
pub fn compute_rms(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f64).sqrt()
}

/// Computes the peak absolute value of a sample buffer.
pub fn compute_peak(samples: &[f64]) -> f64 {
    samples.iter().map(|s| s.abs()).fold(0.0_f64, f64::max)
}

/// Converts a linear RMS value to decibels.
pub fn rms_to_db(rms: f64) -> f64 {
    if rms <= 0.0 {
        -f64::INFINITY
    } else {
        20.0 * rms.log10()
    }
}

/// Converts a linear peak value to decibels.
pub fn peak_to_db(peak: f64) -> f64 {
    if peak <= 0.0 {
        -f64::INFINITY
    } else {
        20.0 * peak.log10()
    }
}

/// Checks whether a block of samples is silent (below threshold).
pub fn is_block_silent(samples: &[f64], threshold_db: f64) -> bool {
    rms_to_db(compute_rms(samples)) < threshold_db
}

/// Counts the number of zero crossings in a buffer.
///
/// Zero crossings can be used as a simple voice activity indicator
/// (speech typically has a moderate zero-crossing rate).
pub fn zero_crossing_count(samples: &[f64]) -> usize {
    if samples.len() < 2 {
        return 0;
    }
    let mut count = 0;
    for i in 1..samples.len() {
        if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
            count += 1;
        }
    }
    count
}

/// Computes the zero-crossing rate (crossings per sample).
#[allow(clippy::cast_precision_loss)]
pub fn zero_crossing_rate(samples: &[f64]) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }
    zero_crossing_count(samples) as f64 / (samples.len() - 1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_region_duration() {
        let region = AudioRegion::new(100, 600, true, -50.0, -45.0);
        assert_eq!(region.duration_samples(), 500);
    }

    #[test]
    fn test_audio_region_duration_secs() {
        let region = AudioRegion::new(0, 48000, false, -10.0, -5.0);
        assert!((region.duration_secs(48000.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_rms_sine() {
        // RMS of a sine wave with amplitude 1.0 is 1/sqrt(2)
        let n = 10000;
        let samples: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / n as f64).sin())
            .collect();
        let rms = compute_rms(&samples);
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((rms - expected).abs() < 0.01);
    }

    #[test]
    fn test_compute_rms_empty() {
        assert!((compute_rms(&[]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_peak() {
        let samples = vec![0.1, -0.5, 0.3, 0.8, -0.2];
        assert!((compute_peak(&samples) - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_rms_to_db() {
        assert!((rms_to_db(1.0) - 0.0).abs() < 1e-10);
        assert!((rms_to_db(0.1) - (-20.0)).abs() < 0.01);
        assert_eq!(rms_to_db(0.0), -f64::INFINITY);
    }

    #[test]
    fn test_is_block_silent() {
        let silence = vec![0.0001; 1024];
        assert!(is_block_silent(&silence, -40.0));
        let loud = vec![0.5; 1024];
        assert!(!is_block_silent(&loud, -40.0));
    }

    #[test]
    fn test_zero_crossing_count() {
        let samples = vec![1.0, -1.0, 1.0, -1.0];
        assert_eq!(zero_crossing_count(&samples), 3);
    }

    #[test]
    fn test_zero_crossing_rate() {
        let samples = vec![1.0, -1.0, 1.0, -1.0];
        assert!((zero_crossing_rate(&samples) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_detect_pure_silence() {
        let config = SilenceDetectConfig {
            threshold_db: -40.0,
            window_size: 256,
            hop_size: 128,
            min_silence_secs: 0.0,
            min_activity_secs: 0.0,
            sample_rate: 48000.0,
            padding_secs: 0.0,
        };
        let detector = SilenceDetector::new(config);
        let samples = vec![0.0; 2048];
        let regions = detector.detect(&samples);
        assert!(!regions.is_empty());
        assert!(regions[0].is_silence);
    }

    #[test]
    fn test_detect_loud_signal() {
        let config = SilenceDetectConfig {
            threshold_db: -40.0,
            window_size: 256,
            hop_size: 128,
            min_silence_secs: 0.0,
            min_activity_secs: 0.0,
            sample_rate: 48000.0,
            padding_secs: 0.0,
        };
        let detector = SilenceDetector::new(config);
        let samples = vec![0.5; 2048];
        let regions = detector.detect(&samples);
        assert!(!regions.is_empty());
        assert!(!regions[0].is_silence);
    }

    #[test]
    fn test_detect_silence_then_activity() {
        let config = SilenceDetectConfig {
            threshold_db: -30.0,
            window_size: 256,
            hop_size: 128,
            min_silence_secs: 0.0,
            min_activity_secs: 0.0,
            sample_rate: 48000.0,
            padding_secs: 0.0,
        };
        let detector = SilenceDetector::new(config);
        let mut samples = vec![0.0001; 4096];
        for s in samples[2048..].iter_mut() {
            *s = 0.5;
        }
        let regions = detector.detect(&samples);
        assert!(regions.len() >= 2);
    }

    #[test]
    fn test_trim_points_with_leading_silence() {
        let config = SilenceDetectConfig {
            threshold_db: -30.0,
            window_size: 256,
            hop_size: 128,
            min_silence_secs: 0.0,
            min_activity_secs: 0.0,
            sample_rate: 48000.0,
            padding_secs: 0.0,
        };
        let detector = SilenceDetector::new(config);
        let mut samples = vec![0.00001; 4096];
        for s in samples[1024..3072].iter_mut() {
            *s = 0.5;
        }
        let (start, end) = detector.trim_points(&samples);
        // Start should be near 1024, end near 3072
        assert!(start < 2048);
        assert!(end > 1024);
    }

    #[test]
    fn test_default_config() {
        let cfg = SilenceDetectConfig::default();
        assert!((cfg.threshold_db - (-40.0)).abs() < 1e-10);
        assert_eq!(cfg.window_size, 1024);
    }
}
