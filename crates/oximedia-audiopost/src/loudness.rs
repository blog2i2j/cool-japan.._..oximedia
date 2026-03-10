#![allow(dead_code)]
//! Loudness management and standards compliance.

use crate::error::{AudioPostError, AudioPostResult};
use serde::{Deserialize, Serialize};

/// Loudness standard
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LoudnessStandard {
    /// EBU R128 (-23 LUFS)
    EbuR128,
    /// ATSC A/85 (-24 LKFS)
    AtscA85,
    /// Netflix (-27 LUFS ±2)
    Netflix,
    /// Spotify (-14 LUFS)
    Spotify,
    /// Apple Music (-16 LUFS)
    AppleMusic,
    /// YouTube (-14 LUFS)
    YouTube,
    /// Custom target
    Custom(f32),
}

impl LoudnessStandard {
    /// Get target loudness in LUFS
    #[must_use]
    pub fn target_lufs(&self) -> f32 {
        match self {
            Self::EbuR128 => -23.0,
            Self::AtscA85 => -24.0,
            Self::Netflix => -27.0,
            Self::Spotify => -14.0,
            Self::AppleMusic => -16.0,
            Self::YouTube => -14.0,
            Self::Custom(target) => *target,
        }
    }

    /// Get tolerance in LU
    #[must_use]
    pub fn tolerance(&self) -> f32 {
        match self {
            Self::Netflix => 2.0,
            _ => 1.0,
        }
    }

    /// Get maximum true peak in dBTP
    #[must_use]
    pub fn max_true_peak(&self) -> f32 {
        match self {
            Self::EbuR128 => -1.0,
            Self::AtscA85 => -2.0,
            Self::Netflix => -2.0,
            Self::Spotify => -1.0,
            Self::AppleMusic => -1.0,
            Self::YouTube => -1.0,
            Self::Custom(_) => -1.0,
        }
    }
}

/// Loudness meter
#[derive(Debug)]
pub struct LoudnessMeter {
    sample_rate: u32,
    standard: LoudnessStandard,
    momentary_lufs: f32,
    short_term_lufs: f32,
    integrated_lufs: f32,
    max_true_peak: f32,
    loudness_range: f32,
}

impl LoudnessMeter {
    /// Create a new loudness meter
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate is invalid
    pub fn new(sample_rate: u32, standard: LoudnessStandard) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }

        Ok(Self {
            sample_rate,
            standard,
            momentary_lufs: -70.0,
            short_term_lufs: -70.0,
            integrated_lufs: -70.0,
            max_true_peak: -70.0,
            loudness_range: 0.0,
        })
    }

    /// Get momentary loudness (400ms)
    #[must_use]
    pub fn get_momentary_lufs(&self) -> f32 {
        self.momentary_lufs
    }

    /// Get short-term loudness (3s)
    #[must_use]
    pub fn get_short_term_lufs(&self) -> f32 {
        self.short_term_lufs
    }

    /// Get integrated loudness
    #[must_use]
    pub fn get_integrated_lufs(&self) -> f32 {
        self.integrated_lufs
    }

    /// Get maximum true peak
    #[must_use]
    pub fn get_max_true_peak(&self) -> f32 {
        self.max_true_peak
    }

    /// Get loudness range (LRA)
    #[must_use]
    pub fn get_loudness_range(&self) -> f32 {
        self.loudness_range
    }

    /// Check if compliant with standard
    #[must_use]
    pub fn is_compliant(&self) -> bool {
        let target = self.standard.target_lufs();
        let tolerance = self.standard.tolerance();
        let max_peak = self.standard.max_true_peak();

        let loudness_ok = (self.integrated_lufs - target).abs() <= tolerance;
        let peak_ok = self.max_true_peak <= max_peak;

        loudness_ok && peak_ok
    }

    /// Get compliance report
    #[must_use]
    pub fn get_compliance_report(&self) -> ComplianceReport {
        ComplianceReport {
            standard: self.standard,
            integrated_lufs: self.integrated_lufs,
            target_lufs: self.standard.target_lufs(),
            max_true_peak: self.max_true_peak,
            max_allowed_peak: self.standard.max_true_peak(),
            loudness_range: self.loudness_range,
            compliant: self.is_compliant(),
        }
    }

    /// Process audio and update measurements
    pub fn process(&mut self, audio: &[f32]) {
        if audio.is_empty() {
            return;
        }

        // Calculate RMS for momentary loudness
        let rms: f32 = audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32;
        let rms = rms.sqrt();

        // Convert to LUFS (simplified calculation)
        self.momentary_lufs = if rms > 0.0 {
            20.0 * rms.log10() - 0.691
        } else {
            -70.0
        };

        // Update true peak
        for &sample in audio {
            let peak_db = 20.0 * sample.abs().log10();
            if peak_db > self.max_true_peak {
                self.max_true_peak = peak_db;
            }
        }

        // Update integrated (simplified)
        self.integrated_lufs = self.momentary_lufs;
    }

    /// Reset measurements
    pub fn reset(&mut self) {
        self.momentary_lufs = -70.0;
        self.short_term_lufs = -70.0;
        self.integrated_lufs = -70.0;
        self.max_true_peak = -70.0;
        self.loudness_range = 0.0;
    }
}

/// Compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    /// Standard used
    pub standard: LoudnessStandard,
    /// Measured integrated loudness
    pub integrated_lufs: f32,
    /// Target loudness
    pub target_lufs: f32,
    /// Maximum true peak measured
    pub max_true_peak: f32,
    /// Maximum allowed true peak
    pub max_allowed_peak: f32,
    /// Loudness range
    pub loudness_range: f32,
    /// Compliance status
    pub compliant: bool,
}

impl ComplianceReport {
    /// Get loudness delta from target
    #[must_use]
    pub fn loudness_delta(&self) -> f32 {
        self.integrated_lufs - self.target_lufs
    }

    /// Get peak delta from maximum
    #[must_use]
    pub fn peak_delta(&self) -> f32 {
        self.max_true_peak - self.max_allowed_peak
    }
}

/// Loudness normalizer
#[derive(Debug)]
pub struct LoudnessNormalizer {
    sample_rate: u32,
    target_lufs: f32,
    max_true_peak: f32,
}

impl LoudnessNormalizer {
    /// Create a new loudness normalizer
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate or target is invalid
    pub fn new(sample_rate: u32, target_lufs: f32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        if target_lufs > 0.0 {
            return Err(AudioPostError::InvalidLoudnessTarget(target_lufs));
        }

        Ok(Self {
            sample_rate,
            target_lufs,
            max_true_peak: -1.0,
        })
    }

    /// Set maximum true peak
    pub fn set_max_true_peak(&mut self, max_peak: f32) {
        self.max_true_peak = max_peak;
    }

    /// Calculate required gain to reach target
    #[must_use]
    pub fn calculate_gain(&self, current_lufs: f32) -> f32 {
        self.target_lufs - current_lufs
    }

    /// Normalize audio to target loudness
    pub fn normalize(&self, input: &[f32], output: &mut [f32], current_lufs: f32) {
        let gain_db = self.calculate_gain(current_lufs);
        let gain_linear = 10.0_f32.powf(gain_db / 20.0);

        for (out, &inp) in output.iter_mut().zip(input.iter()) {
            *out = inp * gain_linear;

            // Apply true peak limiting
            let peak_linear = 10.0_f32.powf(self.max_true_peak / 20.0);
            if out.abs() > peak_linear {
                *out = out.signum() * peak_linear;
            }
        }
    }
}

/// Automatic gain adjustment
#[derive(Debug)]
pub struct AutoGain {
    sample_rate: u32,
    target_lufs: f32,
    attack_time: f32,
    release_time: f32,
}

impl AutoGain {
    /// Create a new auto gain processor
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate or target is invalid
    pub fn new(sample_rate: u32, target_lufs: f32) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        if target_lufs > 0.0 {
            return Err(AudioPostError::InvalidLoudnessTarget(target_lufs));
        }

        Ok(Self {
            sample_rate,
            target_lufs,
            attack_time: 100.0,
            release_time: 1000.0,
        })
    }

    /// Set attack time in milliseconds
    pub fn set_attack_time(&mut self, attack_ms: f32) {
        self.attack_time = attack_ms.max(0.0);
    }

    /// Set release time in milliseconds
    pub fn set_release_time(&mut self, release_ms: f32) {
        self.release_time = release_ms.max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loudness_standards() {
        assert_eq!(LoudnessStandard::EbuR128.target_lufs(), -23.0);
        assert_eq!(LoudnessStandard::AtscA85.target_lufs(), -24.0);
        assert_eq!(LoudnessStandard::Netflix.target_lufs(), -27.0);
    }

    #[test]
    fn test_custom_standard() {
        let custom = LoudnessStandard::Custom(-20.0);
        assert_eq!(custom.target_lufs(), -20.0);
    }

    #[test]
    fn test_loudness_meter_creation() {
        let meter = LoudnessMeter::new(48000, LoudnessStandard::EbuR128).expect("failed to create");
        assert_eq!(meter.sample_rate, 48000);
    }

    #[test]
    fn test_loudness_meter_process() {
        let mut meter =
            LoudnessMeter::new(48000, LoudnessStandard::EbuR128).expect("failed to create");
        let audio = vec![0.1_f32; 1000];
        meter.process(&audio);
        assert!(meter.get_momentary_lufs() > -70.0);
    }

    #[test]
    fn test_loudness_meter_reset() {
        let mut meter =
            LoudnessMeter::new(48000, LoudnessStandard::EbuR128).expect("failed to create");
        let audio = vec![0.1_f32; 1000];
        meter.process(&audio);
        meter.reset();
        assert_eq!(meter.get_integrated_lufs(), -70.0);
    }

    #[test]
    fn test_compliance_report() {
        let meter = LoudnessMeter::new(48000, LoudnessStandard::EbuR128).expect("failed to create");
        let report = meter.get_compliance_report();
        assert_eq!(report.target_lufs, -23.0);
    }

    #[test]
    fn test_loudness_normalizer() {
        let normalizer = LoudnessNormalizer::new(48000, -23.0).expect("failed to create");
        let gain = normalizer.calculate_gain(-26.0);
        assert!((gain - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_audio() {
        let normalizer = LoudnessNormalizer::new(48000, -23.0).expect("failed to create");
        let input = vec![0.1_f32; 1000];
        let mut output = vec![0.0_f32; 1000];
        normalizer.normalize(&input, &mut output, -26.0);
        assert!(output[0] > input[0]);
    }

    #[test]
    fn test_auto_gain() {
        let mut auto_gain = AutoGain::new(48000, -23.0).expect("failed to create");
        auto_gain.set_attack_time(50.0);
        auto_gain.set_release_time(500.0);
        assert_eq!(auto_gain.attack_time, 50.0);
    }

    #[test]
    fn test_invalid_target_lufs() {
        assert!(LoudnessNormalizer::new(48000, 5.0).is_err());
    }

    #[test]
    fn test_compliance_report_deltas() {
        let report = ComplianceReport {
            standard: LoudnessStandard::EbuR128,
            integrated_lufs: -24.0,
            target_lufs: -23.0,
            max_true_peak: -0.5,
            max_allowed_peak: -1.0,
            loudness_range: 10.0,
            compliant: false,
        };

        assert_eq!(report.loudness_delta(), -1.0);
        assert_eq!(report.peak_delta(), 0.5);
    }

    #[test]
    fn test_netflix_tolerance() {
        assert_eq!(LoudnessStandard::Netflix.tolerance(), 2.0);
        assert_eq!(LoudnessStandard::EbuR128.tolerance(), 1.0);
    }
}
