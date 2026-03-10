//! Dynamic range analysis.

use crate::{amplitude_to_db, AnalysisConfig, Result};

/// Dynamics analyzer for measuring dynamic range and loudness variations.
pub struct DynamicsAnalyzer {
    config: AnalysisConfig,
}

impl DynamicsAnalyzer {
    /// Create a new dynamics analyzer.
    #[must_use]
    pub fn new(config: AnalysisConfig) -> Self {
        Self { config }
    }

    /// Analyze dynamic range of audio samples.
    pub fn analyze(&self, samples: &[f32], _sample_rate: f32) -> Result<DynamicsResult> {
        if samples.is_empty() {
            return Ok(DynamicsResult::default());
        }

        // Compute RMS over time
        let rms_values = super::rms::rms_over_time(samples, self.config.hop_size);

        // Peak level
        let peak = samples.iter().map(|&x| x.abs()).fold(0.0_f32, f32::max);

        // RMS level (overall)
        let rms = super::rms::rms_level(samples);

        // Crest factor
        let crest = super::crest::crest_factor(samples);

        // Dynamic range (difference between loudest and quietest parts)
        let max_rms = rms_values.iter().copied().fold(0.0_f32, f32::max);
        let min_rms = rms_values
            .iter()
            .copied()
            .filter(|&x| x > 1e-6)
            .fold(f32::INFINITY, f32::min);

        let dynamic_range_db = if min_rms > 0.0 && min_rms.is_finite() {
            amplitude_to_db(max_rms) - amplitude_to_db(min_rms)
        } else {
            0.0
        };

        // Loudness variation (standard deviation of RMS)
        let mean_rms = rms_values.iter().sum::<f32>() / rms_values.len() as f32;
        let variance = rms_values
            .iter()
            .map(|&x| (x - mean_rms).powi(2))
            .sum::<f32>()
            / rms_values.len() as f32;
        let loudness_variation = variance.sqrt();

        Ok(DynamicsResult {
            peak,
            rms,
            crest,
            dynamic_range_db,
            loudness_variation,
            rms_over_time: rms_values,
        })
    }
}

/// Dynamic range analysis result.
#[derive(Debug, Clone)]
pub struct DynamicsResult {
    /// Peak amplitude (0-1)
    pub peak: f32,
    /// RMS level (0-1)
    pub rms: f32,
    /// Crest factor (peak-to-RMS ratio)
    pub crest: f32,
    /// Dynamic range in dB
    pub dynamic_range_db: f32,
    /// Loudness variation (standard deviation of RMS)
    pub loudness_variation: f32,
    /// RMS level over time
    pub rms_over_time: Vec<f32>,
}

impl Default for DynamicsResult {
    fn default() -> Self {
        Self {
            peak: 0.0,
            rms: 0.0,
            crest: 0.0,
            dynamic_range_db: 0.0,
            loudness_variation: 0.0,
            rms_over_time: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamics_analyzer() {
        let config = AnalysisConfig::default();
        let analyzer = DynamicsAnalyzer::new(config);

        // Generate test signal with varying amplitude
        let mut samples = Vec::new();
        for i in 0..44100 {
            let amp = (i as f32 / 10000.0).sin().abs() * 0.5 + 0.1;
            samples.push(amp * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin());
        }

        let result = analyzer
            .analyze(&samples, 44100.0)
            .expect("analysis should succeed");

        assert!(result.peak > 0.0);
        assert!(result.rms > 0.0);
        assert!(result.crest > 1.0);
        assert!(result.dynamic_range_db > 0.0);
    }
}
