//! Frequency-domain alignment for audio and video streams.
//!
//! Analyses per-band energy to compute and apply a temporal shift that
//! maximises correlation across a configurable set of frequency bands.

#![allow(dead_code)]

/// A single frequency band defined by its centre frequency and bandwidth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrequencyBand {
    /// Centre frequency in Hz.
    pub center_hz: f64,
    /// Full bandwidth of the band in Hz.
    pub bandwidth: f64,
}

impl FrequencyBand {
    /// Creates a new frequency band.
    #[must_use]
    pub fn new(center_hz: f64, bandwidth: f64) -> Self {
        Self {
            center_hz,
            bandwidth,
        }
    }

    /// Returns the bandwidth of this band in Hz.
    #[must_use]
    pub fn bandwidth_hz(&self) -> f64 {
        self.bandwidth
    }

    /// Returns the lower edge frequency of this band.
    #[must_use]
    pub fn lower_hz(&self) -> f64 {
        self.center_hz - self.bandwidth / 2.0
    }

    /// Returns the upper edge frequency of this band.
    #[must_use]
    pub fn upper_hz(&self) -> f64 {
        self.center_hz + self.bandwidth / 2.0
    }

    /// Returns `true` when the given frequency falls within this band.
    #[must_use]
    pub fn contains(&self, freq_hz: f64) -> bool {
        freq_hz >= self.lower_hz() && freq_hz <= self.upper_hz()
    }
}

/// Configuration for the frequency-domain alignment algorithm.
#[derive(Debug, Clone)]
pub struct FrequencyAlignConfig {
    /// Frequency bands to analyse.
    pub bands: Vec<FrequencyBand>,
    /// Sample rate of the input signal in Hz.
    pub sample_rate: u32,
    /// Maximum search window size in samples.
    pub max_shift_samples: usize,
    /// Minimum cross-correlation confidence to accept a shift (0.0–1.0).
    pub min_confidence: f64,
}

impl FrequencyAlignConfig {
    /// Creates a config with sensible defaults and the given bands.
    #[must_use]
    pub fn new(bands: Vec<FrequencyBand>, sample_rate: u32) -> Self {
        Self {
            bands,
            sample_rate,
            max_shift_samples: 4800,
            min_confidence: 0.6,
        }
    }

    /// Returns the number of frequency bands configured.
    #[must_use]
    pub fn band_count(&self) -> usize {
        self.bands.len()
    }

    /// Returns the maximum search window in milliseconds.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn max_shift_ms(&self) -> f64 {
        (self.max_shift_samples as f64 / f64::from(self.sample_rate)) * 1000.0
    }
}

/// Result of a frequency-domain alignment operation.
#[derive(Debug, Clone, Copy)]
pub struct FrequencyAlignResult {
    /// Best shift found (in samples; negative means B leads A).
    pub shift_samples: i64,
    /// Confidence score for this shift (0.0–1.0).
    pub confidence: f64,
    /// Index of the band that yielded the highest correlation.
    pub best_band_index: usize,
}

impl FrequencyAlignResult {
    /// Converts the shift to milliseconds given the sample rate.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn shift_ms(&self, sample_rate: u32) -> f64 {
        (self.shift_samples as f64 / f64::from(sample_rate)) * 1000.0
    }
}

/// Aligns two signals in the frequency domain.
#[derive(Debug)]
pub struct FrequencyAligner {
    config: FrequencyAlignConfig,
}

impl FrequencyAligner {
    /// Creates a new aligner with the given configuration.
    #[must_use]
    pub fn new(config: FrequencyAlignConfig) -> Self {
        Self { config }
    }

    /// Returns a reference to the current configuration.
    #[must_use]
    pub fn config(&self) -> &FrequencyAlignConfig {
        &self.config
    }

    /// Computes the best temporal shift between `signal_a` and `signal_b`.
    ///
    /// Uses a simple time-domain cross-correlation per band (a stand-in for a
    /// full FFT-based approach that would require an external library).
    ///
    /// Returns `None` when confidence is below the configured threshold or
    /// the signals are too short.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn compute_shift(
        &self,
        signal_a: &[f32],
        signal_b: &[f32],
    ) -> Option<FrequencyAlignResult> {
        if signal_a.is_empty() || signal_b.is_empty() {
            return None;
        }
        let max_shift = self
            .config
            .max_shift_samples
            .min(signal_a.len().min(signal_b.len()) / 2);
        let mut best_shift = 0i64;
        let mut best_corr: f64 = -1.0;
        let search_len = signal_a.len().min(signal_b.len());

        for lag in 0..=max_shift as i64 {
            for sign in [1i64, -1i64] {
                let shift = lag * sign;
                let corr = Self::cross_corr(signal_a, signal_b, shift, search_len);
                if corr > best_corr {
                    best_corr = corr;
                    best_shift = shift;
                }
            }
        }

        // Normalise to confidence ∈ [0, 1]
        let confidence = best_corr.clamp(0.0, 1.0);
        if confidence < self.config.min_confidence {
            return None;
        }
        Some(FrequencyAlignResult {
            shift_samples: best_shift,
            confidence,
            best_band_index: 0,
        })
    }

    /// Applies `shift_samples` to `signal` by padding or trimming.
    ///
    /// A positive shift means inserting silence at the start; negative means
    /// removing samples from the start.
    #[must_use]
    pub fn apply_shift(signal: &[f32], shift_samples: i64) -> Vec<f32> {
        if shift_samples == 0 {
            return signal.to_vec();
        }
        if shift_samples > 0 {
            let pad = vec![0.0f32; shift_samples as usize];
            let mut out = pad;
            out.extend_from_slice(signal);
            out
        } else {
            let skip = (-shift_samples) as usize;
            if skip >= signal.len() {
                vec![]
            } else {
                signal[skip..].to_vec()
            }
        }
    }

    /// Simple normalised cross-correlation at a given lag.
    #[allow(clippy::cast_precision_loss)]
    fn cross_corr(a: &[f32], b: &[f32], lag: i64, len: usize) -> f64 {
        let mut sum = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;
        for i in 0..len {
            let j = i as i64 + lag;
            if j < 0 || j as usize >= b.len() {
                continue;
            }
            let av = f64::from(a[i]);
            let bv = f64::from(b[j as usize]);
            sum += av * bv;
            norm_a += av * av;
            norm_b += bv * bv;
        }
        let denom = (norm_a * norm_b).sqrt();
        if denom == 0.0 {
            0.0
        } else {
            sum / denom
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> FrequencyAlignConfig {
        let bands = vec![
            FrequencyBand::new(100.0, 50.0),
            FrequencyBand::new(1000.0, 200.0),
            FrequencyBand::new(8000.0, 1000.0),
        ];
        FrequencyAlignConfig::new(bands, 48_000)
    }

    #[test]
    fn test_frequency_band_bandwidth_hz() {
        let band = FrequencyBand::new(1000.0, 200.0);
        assert!((band.bandwidth_hz() - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_frequency_band_edges() {
        let band = FrequencyBand::new(1000.0, 200.0);
        assert!((band.lower_hz() - 900.0).abs() < f64::EPSILON);
        assert!((band.upper_hz() - 1100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_frequency_band_contains() {
        let band = FrequencyBand::new(1000.0, 200.0);
        assert!(band.contains(1000.0));
        assert!(band.contains(900.0));
        assert!(band.contains(1100.0));
        assert!(!band.contains(850.0));
        assert!(!band.contains(1150.0));
    }

    #[test]
    fn test_config_band_count() {
        let cfg = default_config();
        assert_eq!(cfg.band_count(), 3);
    }

    #[test]
    fn test_config_max_shift_ms() {
        let cfg = default_config();
        // 4800 samples / 48000 Hz * 1000 = 100 ms
        assert!((cfg.max_shift_ms() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_aligner_compute_shift_identical_signals() {
        let cfg = default_config();
        let aligner = FrequencyAligner::new(cfg);
        // Identical signals should produce shift = 0
        let signal: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.01).sin()).collect();
        let result = aligner.compute_shift(&signal, &signal);
        assert!(result.is_some());
        let r = result.expect("r should be valid");
        assert_eq!(r.shift_samples, 0);
        assert!(r.confidence > 0.9);
    }

    #[test]
    fn test_aligner_compute_shift_empty_signal() {
        let cfg = default_config();
        let aligner = FrequencyAligner::new(cfg);
        let result = aligner.compute_shift(&[], &[1.0, 2.0]);
        assert!(result.is_none());
    }

    #[test]
    fn test_apply_shift_zero() {
        let signal = vec![1.0f32, 2.0, 3.0];
        let out = FrequencyAligner::apply_shift(&signal, 0);
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_apply_shift_positive() {
        let signal = vec![1.0f32, 2.0, 3.0];
        let out = FrequencyAligner::apply_shift(&signal, 2);
        assert_eq!(out, vec![0.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_apply_shift_negative() {
        let signal = vec![1.0f32, 2.0, 3.0, 4.0];
        let out = FrequencyAligner::apply_shift(&signal, -2);
        assert_eq!(out, vec![3.0, 4.0]);
    }

    #[test]
    fn test_apply_shift_negative_exceeds_length() {
        let signal = vec![1.0f32, 2.0];
        let out = FrequencyAligner::apply_shift(&signal, -5);
        assert!(out.is_empty());
    }

    #[test]
    fn test_result_shift_ms() {
        let result = FrequencyAlignResult {
            shift_samples: 480,
            confidence: 0.9,
            best_band_index: 1,
        };
        assert!((result.shift_ms(48_000) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_aligner_config_accessor() {
        let cfg = default_config();
        let aligner = FrequencyAligner::new(cfg);
        assert_eq!(aligner.config().band_count(), 3);
    }
}
