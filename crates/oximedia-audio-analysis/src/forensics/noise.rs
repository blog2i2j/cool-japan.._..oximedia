//! Background noise consistency analysis.

use crate::{AnalysisConfig, Result};

/// Noise analyzer for forensic analysis.
pub struct NoiseAnalyzer {
    config: AnalysisConfig,
}

impl NoiseAnalyzer {
    /// Create a new noise analyzer.
    #[must_use]
    pub fn new(config: AnalysisConfig) -> Self {
        Self { config }
    }

    /// Analyze noise consistency across audio.
    pub fn analyze_consistency(
        &self,
        samples: &[f32],
        _sample_rate: f32,
    ) -> Result<NoiseConsistency> {
        // Divide audio into segments and analyze noise floor in each
        let segment_size = self.config.fft_size * 10;
        let num_segments = samples.len() / segment_size;

        if num_segments < 2 {
            return Ok(NoiseConsistency {
                is_consistent: true,
                noise_floor_variation: 0.0,
            });
        }

        let mut noise_floors = Vec::new();

        for i in 0..num_segments {
            let start = i * segment_size;
            let end = ((i + 1) * segment_size).min(samples.len());
            let segment = &samples[start..end];

            let noise_floor = self.estimate_noise_floor(segment);
            noise_floors.push(noise_floor);
        }

        // Compute variation in noise floors
        let mean = noise_floors.iter().sum::<f32>() / noise_floors.len() as f32;
        let variance = noise_floors
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / noise_floors.len() as f32;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = if mean > 0.0 { std_dev / mean } else { 0.0 };

        // Noise should be relatively consistent
        let is_consistent = coefficient_of_variation < 0.5;

        Ok(NoiseConsistency {
            is_consistent,
            noise_floor_variation: coefficient_of_variation,
        })
    }

    /// Estimate noise floor from audio segment.
    #[allow(clippy::unused_self)]
    fn estimate_noise_floor(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        // Use lower percentile of amplitudes as noise floor estimate
        let mut amplitudes: Vec<f32> = samples.iter().map(|&x| x.abs()).collect();
        amplitudes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // 10th percentile as noise floor
        let percentile_idx = (amplitudes.len() as f32 * 0.1) as usize;
        amplitudes[percentile_idx]
    }
}

/// Noise consistency analysis result.
#[derive(Debug, Clone)]
pub struct NoiseConsistency {
    /// Whether noise floor is consistent
    pub is_consistent: bool,
    /// Coefficient of variation of noise floor
    pub noise_floor_variation: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_analyzer() {
        let config = AnalysisConfig::default();
        let analyzer = NoiseAnalyzer::new(config);

        // Generate signal with consistent noise
        let mut samples = Vec::new();
        for _ in 0..100000 {
            samples.push(0.01); // Constant low-level noise
        }

        let result = analyzer.analyze_consistency(&samples, 44100.0);
        assert!(result.is_ok());

        let consistency = result.expect("expected successful result");
        assert!(consistency.is_consistent);
    }

    #[test]
    fn test_noise_floor_estimation() {
        let config = AnalysisConfig::default();
        let analyzer = NoiseAnalyzer::new(config);

        let samples = vec![0.01, 0.02, 0.01, 0.5, 0.8, 0.01]; // Mostly low, some peaks
        let floor = analyzer.estimate_noise_floor(&samples);

        assert!(floor < 0.1); // Should be close to low values
    }
}
