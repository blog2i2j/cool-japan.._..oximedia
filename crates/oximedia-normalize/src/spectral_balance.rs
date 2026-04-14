#![allow(dead_code)]
//! Spectral balance normalization for frequency-band loudness control.
//!
//! This module provides multi-band spectral analysis and normalization to ensure
//! that audio content has a balanced frequency distribution. It is useful for
//! maintaining consistent tonal quality across different source materials.

use std::collections::HashMap;

/// Number of octave bands in the standard ISO set.
const NUM_OCTAVE_BANDS: usize = 10;

/// Default center frequencies for octave bands (Hz).
const OCTAVE_CENTERS: [f64; NUM_OCTAVE_BANDS] = [
    31.25, 62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0,
];

/// A single frequency band with configurable boundaries.
#[derive(Debug, Clone)]
pub struct FrequencyBand {
    /// Center frequency in Hz.
    pub center_hz: f64,
    /// Lower cutoff frequency in Hz.
    pub low_hz: f64,
    /// Upper cutoff frequency in Hz.
    pub high_hz: f64,
    /// Measured energy level in dB.
    pub energy_db: f64,
    /// Target energy level in dB.
    pub target_db: f64,
    /// Computed gain correction in dB.
    pub correction_db: f64,
}

impl FrequencyBand {
    /// Create a new frequency band.
    pub fn new(center_hz: f64, low_hz: f64, high_hz: f64) -> Self {
        Self {
            center_hz,
            low_hz,
            high_hz,
            energy_db: -100.0,
            target_db: 0.0,
            correction_db: 0.0,
        }
    }

    /// Compute the bandwidth in Hz.
    pub fn bandwidth(&self) -> f64 {
        self.high_hz - self.low_hz
    }

    /// Compute the Q factor.
    pub fn q_factor(&self) -> f64 {
        if self.bandwidth() > 0.0 {
            self.center_hz / self.bandwidth()
        } else {
            1.0
        }
    }

    /// Check whether a frequency falls within this band.
    pub fn contains(&self, freq_hz: f64) -> bool {
        freq_hz >= self.low_hz && freq_hz < self.high_hz
    }
}

/// Configuration for spectral balance normalization.
#[derive(Debug, Clone)]
pub struct SpectralBalanceConfig {
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Number of channels.
    pub channels: usize,
    /// FFT size for spectral analysis.
    pub fft_size: usize,
    /// Hop size (overlap) for spectral analysis.
    pub hop_size: usize,
    /// Maximum correction gain in dB per band.
    pub max_correction_db: f64,
    /// Smoothing factor (0.0 = no smoothing, 1.0 = full smoothing).
    pub smoothing: f64,
    /// Whether to use A-weighting for perceived loudness.
    pub a_weighting: bool,
}

impl Default for SpectralBalanceConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000.0,
            channels: 2,
            fft_size: 4096,
            hop_size: 2048,
            max_correction_db: 6.0,
            smoothing: 0.8,
            a_weighting: true,
        }
    }
}

impl SpectralBalanceConfig {
    /// Create a new configuration with the given sample rate and channels.
    pub fn new(sample_rate: f64, channels: usize) -> Self {
        Self {
            sample_rate,
            channels,
            ..Default::default()
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.sample_rate < 8000.0 || self.sample_rate > 192_000.0 {
            return Err(format!("Invalid sample rate: {}", self.sample_rate));
        }
        if self.channels == 0 || self.channels > 16 {
            return Err(format!("Invalid channel count: {}", self.channels));
        }
        if self.fft_size < 256 || !self.fft_size.is_power_of_two() {
            return Err(format!(
                "FFT size must be a power of 2 >= 256, got {}",
                self.fft_size
            ));
        }
        if self.hop_size == 0 || self.hop_size > self.fft_size {
            return Err(format!(
                "Hop size must be in (0, {}], got {}",
                self.fft_size, self.hop_size
            ));
        }
        if self.max_correction_db < 0.0 || self.max_correction_db > 24.0 {
            return Err(format!(
                "Max correction must be in [0, 24] dB, got {}",
                self.max_correction_db
            ));
        }
        if !(0.0..=1.0).contains(&self.smoothing) {
            return Err(format!(
                "Smoothing must be in [0, 1], got {}",
                self.smoothing
            ));
        }
        Ok(())
    }
}

/// A-weighting coefficient lookup for standard octave bands.
fn a_weight_correction(center_hz: f64) -> f64 {
    // Approximate A-weighting corrections for octave-band centers
    let table: &[(f64, f64)] = &[
        (31.25, -39.4),
        (62.5, -26.2),
        (125.0, -16.1),
        (250.0, -8.6),
        (500.0, -3.2),
        (1000.0, 0.0),
        (2000.0, 1.2),
        (4000.0, 1.0),
        (8000.0, -1.1),
        (16000.0, -6.6),
    ];

    // Find the closest entry
    let mut best = 0.0;
    let mut best_dist = f64::MAX;
    for &(freq, weight) in table {
        let dist = (freq - center_hz).abs();
        if dist < best_dist {
            best_dist = dist;
            best = weight;
        }
    }
    best
}

/// Target spectral profile for normalization.
#[derive(Debug, Clone)]
pub enum SpectralTarget {
    /// Flat spectral profile (equal energy per band).
    Flat,
    /// Pink noise profile (-3 dB/octave).
    Pink,
    /// Broadcast speech profile (emphasis on 1-4 kHz).
    Speech,
    /// Music mastering profile.
    Music,
    /// Custom per-band targets (band index -> target dB).
    Custom(HashMap<usize, f64>),
}

impl SpectralTarget {
    /// Get the target level offset for a given band index.
    pub fn target_offset(&self, band_index: usize, center_hz: f64) -> f64 {
        match self {
            Self::Flat => 0.0,
            Self::Pink => {
                // -3 dB per octave relative to 1 kHz
                if center_hz > 0.0 {
                    -3.0 * (center_hz / 1000.0).log2()
                } else {
                    0.0
                }
            }
            Self::Speech => {
                // Emphasis on 1-4 kHz for speech intelligibility
                if (1000.0..=4000.0).contains(&center_hz) {
                    3.0
                } else if center_hz < 250.0 {
                    -6.0
                } else {
                    0.0
                }
            }
            Self::Music => {
                // Slight mid-scoop, sub/presence boost
                if center_hz < 80.0 {
                    2.0
                } else if (200.0..=800.0).contains(&center_hz) {
                    -2.0
                } else if (2000.0..=6000.0).contains(&center_hz) {
                    1.5
                } else {
                    0.0
                }
            }
            Self::Custom(map) => map.get(&band_index).copied().unwrap_or(0.0),
        }
    }
}

/// Spectral balance analyzer and normalizer.
#[derive(Debug)]
pub struct SpectralBalanceProcessor {
    /// Configuration.
    config: SpectralBalanceConfig,
    /// Frequency bands.
    bands: Vec<FrequencyBand>,
    /// Target spectral profile.
    target: SpectralTarget,
    /// Number of frames analyzed.
    frames_analyzed: u64,
    /// Running average energy per band.
    avg_energy: Vec<f64>,
}

impl SpectralBalanceProcessor {
    /// Create a new spectral balance processor.
    pub fn new(config: SpectralBalanceConfig, target: SpectralTarget) -> Result<Self, String> {
        config.validate()?;

        let nyquist = config.sample_rate / 2.0;
        let mut bands = Vec::new();
        for (i, &center) in OCTAVE_CENTERS.iter().enumerate() {
            if center < nyquist {
                let low = center / 2.0_f64.sqrt();
                let high = (center * 2.0_f64.sqrt()).min(nyquist);
                bands.push(FrequencyBand::new(center, low, high));
                // Set target based on profile
                if let Some(band) = bands.last_mut() {
                    band.target_db = target.target_offset(i, center);
                }
            }
        }

        let num_bands = bands.len();
        Ok(Self {
            config,
            bands,
            target,
            frames_analyzed: 0,
            avg_energy: vec![-100.0; num_bands],
        })
    }

    /// Get the frequency bands.
    pub fn bands(&self) -> &[FrequencyBand] {
        &self.bands
    }

    /// Get the number of bands.
    pub fn num_bands(&self) -> usize {
        self.bands.len()
    }

    /// Get the number of frames analyzed.
    pub fn frames_analyzed(&self) -> u64 {
        self.frames_analyzed
    }

    /// Analyze a block of audio samples and update band energies.
    ///
    /// Samples are interleaved for multi-channel audio.
    pub fn analyze(&mut self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }

        let channels = self.config.channels;
        let frame_count = samples.len() / channels;
        if frame_count == 0 {
            return;
        }

        // Compute RMS energy of the block (simple time-domain approximation)
        let mut sum_sq = 0.0_f64;
        for &s in samples {
            let v = f64::from(s);
            sum_sq += v * v;
        }
        let rms = (sum_sq / samples.len() as f64).sqrt();
        let rms_db = if rms > 1e-10 {
            20.0 * rms.log10()
        } else {
            -100.0
        };

        // Distribute energy across bands (simplified model)
        // In a real implementation this would use FFT; here we approximate
        let num_bands = self.bands.len();
        for (i, band) in self.bands.iter_mut().enumerate() {
            // Simple model: energy falls off with distance from center of spectrum
            let norm_pos = i as f64 / num_bands.max(1) as f64;
            let spectral_shape = 1.0 - (norm_pos - 0.5).abs() * 0.5;
            let band_energy = rms_db + 20.0 * spectral_shape.log10().max(-20.0);

            // Exponential smoothing
            let alpha = 1.0 - self.config.smoothing;
            if self.frames_analyzed == 0 {
                self.avg_energy[i] = band_energy;
            } else {
                self.avg_energy[i] =
                    self.avg_energy[i] * self.config.smoothing + band_energy * alpha;
            }
            band.energy_db = self.avg_energy[i];
        }

        self.frames_analyzed += 1;
    }

    /// Compute per-band correction gains.
    pub fn compute_corrections(&mut self) -> Vec<f64> {
        let max_corr = self.config.max_correction_db;
        let mut corrections = Vec::with_capacity(self.bands.len());

        for (i, band) in self.bands.iter_mut().enumerate() {
            let mut target = band.target_db;
            if self.config.a_weighting {
                target += a_weight_correction(band.center_hz);
            }

            let diff = target - band.energy_db;
            let correction = diff.clamp(-max_corr, max_corr);
            band.correction_db = correction;
            corrections.push(correction);
            let _ = i; // suppress unused warning
        }

        corrections
    }

    /// Apply computed corrections to audio samples (in-place).
    ///
    /// This applies a simple broadband gain derived from the average correction.
    /// A production implementation would use per-band parametric EQ.
    pub fn apply_broadband_correction(&self, samples: &mut [f32]) {
        if self.bands.is_empty() || samples.is_empty() {
            return;
        }

        let avg_correction: f64 =
            self.bands.iter().map(|b| b.correction_db).sum::<f64>() / self.bands.len() as f64;

        let gain = 10.0_f64.powf(avg_correction / 20.0);
        for s in samples.iter_mut() {
            *s = (f64::from(*s) * gain) as f32;
        }
    }

    /// Reset all analysis state.
    pub fn reset(&mut self) {
        self.frames_analyzed = 0;
        for (i, band) in self.bands.iter_mut().enumerate() {
            band.energy_db = -100.0;
            band.correction_db = 0.0;
            self.avg_energy[i] = -100.0;
        }
    }

    /// Get a summary report of the spectral analysis.
    pub fn report(&self) -> SpectralReport {
        let band_reports: Vec<BandReport> = self
            .bands
            .iter()
            .map(|b| BandReport {
                center_hz: b.center_hz,
                energy_db: b.energy_db,
                target_db: b.target_db,
                correction_db: b.correction_db,
            })
            .collect();

        let avg_energy = if band_reports.is_empty() {
            -100.0
        } else {
            band_reports.iter().map(|b| b.energy_db).sum::<f64>() / band_reports.len() as f64
        };

        SpectralReport {
            bands: band_reports,
            avg_energy_db: avg_energy,
            frames_analyzed: self.frames_analyzed,
        }
    }
}

/// Report for a single frequency band.
#[derive(Debug, Clone)]
pub struct BandReport {
    /// Center frequency in Hz.
    pub center_hz: f64,
    /// Measured energy in dB.
    pub energy_db: f64,
    /// Target energy in dB.
    pub target_db: f64,
    /// Computed correction in dB.
    pub correction_db: f64,
}

/// Summary report for spectral balance analysis.
#[derive(Debug, Clone)]
pub struct SpectralReport {
    /// Per-band reports.
    pub bands: Vec<BandReport>,
    /// Average energy across all bands in dB.
    pub avg_energy_db: f64,
    /// Total frames analyzed.
    pub frames_analyzed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequency_band_creation() {
        let band = FrequencyBand::new(1000.0, 707.0, 1414.0);
        assert!((band.center_hz - 1000.0).abs() < f64::EPSILON);
        assert!((band.low_hz - 707.0).abs() < f64::EPSILON);
        assert!((band.energy_db - (-100.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_frequency_band_bandwidth() {
        let band = FrequencyBand::new(1000.0, 707.0, 1414.0);
        assert!((band.bandwidth() - 707.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_frequency_band_q_factor() {
        let band = FrequencyBand::new(1000.0, 707.0, 1414.0);
        let q = band.q_factor();
        assert!((q - 1000.0 / 707.0).abs() < 0.01);
    }

    #[test]
    fn test_frequency_band_contains() {
        let band = FrequencyBand::new(1000.0, 707.0, 1414.0);
        assert!(band.contains(1000.0));
        assert!(band.contains(707.0));
        assert!(!band.contains(1414.0)); // exclusive upper
        assert!(!band.contains(500.0));
    }

    #[test]
    fn test_config_default() {
        let config = SpectralBalanceConfig::default();
        assert!((config.sample_rate - 48000.0).abs() < f64::EPSILON);
        assert_eq!(config.channels, 2);
        assert_eq!(config.fft_size, 4096);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_invalid_sample_rate() {
        let mut config = SpectralBalanceConfig::default();
        config.sample_rate = 100.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_invalid_fft_size() {
        let mut config = SpectralBalanceConfig::default();
        config.fft_size = 300; // not a power of 2
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_spectral_target_flat() {
        let target = SpectralTarget::Flat;
        assert!((target.target_offset(0, 31.25)).abs() < f64::EPSILON);
        assert!((target.target_offset(5, 1000.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_spectral_target_pink() {
        let target = SpectralTarget::Pink;
        // At 1 kHz the offset should be 0
        assert!((target.target_offset(5, 1000.0)).abs() < f64::EPSILON);
        // At 2 kHz the offset should be -3 dB
        let offset = target.target_offset(6, 2000.0);
        assert!((offset - (-3.0)).abs() < 0.01);
    }

    #[test]
    fn test_spectral_target_speech() {
        let target = SpectralTarget::Speech;
        assert!((target.target_offset(5, 1000.0) - 3.0).abs() < f64::EPSILON);
        assert!((target.target_offset(0, 31.25) - (-6.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_spectral_target_custom() {
        let mut map = HashMap::new();
        map.insert(3, 5.0);
        let target = SpectralTarget::Custom(map);
        assert!((target.target_offset(3, 250.0) - 5.0).abs() < f64::EPSILON);
        assert!((target.target_offset(0, 31.25)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_processor_creation() {
        let config = SpectralBalanceConfig::default();
        let proc = SpectralBalanceProcessor::new(config, SpectralTarget::Flat);
        assert!(proc.is_ok());
        let proc = proc.expect("should succeed in test");
        assert!(proc.num_bands() > 0);
        assert_eq!(proc.frames_analyzed(), 0);
    }

    #[test]
    fn test_processor_analyze() {
        let config = SpectralBalanceConfig::new(48000.0, 1);
        let mut proc = SpectralBalanceProcessor::new(config, SpectralTarget::Flat)
            .expect("should succeed in test");

        let samples: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.01).sin()).collect();
        proc.analyze(&samples);
        assert_eq!(proc.frames_analyzed(), 1);

        // Energies should be updated
        for band in proc.bands() {
            assert!(band.energy_db > -100.0);
        }
    }

    #[test]
    fn test_processor_compute_corrections() {
        let config = SpectralBalanceConfig::new(48000.0, 1);
        let mut proc = SpectralBalanceProcessor::new(config, SpectralTarget::Flat)
            .expect("should succeed in test");

        let samples: Vec<f32> = vec![0.5; 4096];
        proc.analyze(&samples);
        let corrections = proc.compute_corrections();
        assert_eq!(corrections.len(), proc.num_bands());
    }

    #[test]
    fn test_processor_apply_broadband() {
        let config = SpectralBalanceConfig::new(48000.0, 1);
        let mut proc = SpectralBalanceProcessor::new(config, SpectralTarget::Flat)
            .expect("should succeed in test");

        let samples: Vec<f32> = vec![0.5; 4096];
        proc.analyze(&samples);
        proc.compute_corrections();

        let mut output = vec![0.5_f32; 16];
        proc.apply_broadband_correction(&mut output);
        // Samples should be modified (gain applied)
        // The exact value depends on the correction
        assert!(output.iter().all(|&s| s != 0.0));
    }

    #[test]
    fn test_processor_reset() {
        let config = SpectralBalanceConfig::new(48000.0, 1);
        let mut proc = SpectralBalanceProcessor::new(config, SpectralTarget::Flat)
            .expect("should succeed in test");

        let samples: Vec<f32> = vec![0.5; 4096];
        proc.analyze(&samples);
        assert_eq!(proc.frames_analyzed(), 1);

        proc.reset();
        assert_eq!(proc.frames_analyzed(), 0);
        for band in proc.bands() {
            assert!((band.energy_db - (-100.0)).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_processor_report() {
        let config = SpectralBalanceConfig::new(48000.0, 1);
        let mut proc = SpectralBalanceProcessor::new(config, SpectralTarget::Flat)
            .expect("should succeed in test");

        let samples: Vec<f32> = vec![0.3; 4096];
        proc.analyze(&samples);
        let report = proc.report();
        assert!(!report.bands.is_empty());
        assert_eq!(report.frames_analyzed, 1);
    }

    #[test]
    fn test_a_weight_correction() {
        let w1k = a_weight_correction(1000.0);
        assert!((w1k).abs() < f64::EPSILON);

        let w31 = a_weight_correction(31.25);
        assert!((w31 - (-39.4)).abs() < f64::EPSILON);
    }
}
