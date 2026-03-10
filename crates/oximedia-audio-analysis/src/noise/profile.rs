//! Noise profiling.

use crate::spectral::SpectralAnalyzer;
use crate::{AnalysisConfig, Result};

/// Noise profiler for characterizing noise.
pub struct NoiseProfiler {
    spectral_analyzer: SpectralAnalyzer,
}

impl NoiseProfiler {
    /// Create a new noise profiler.
    #[must_use]
    pub fn new(config: AnalysisConfig) -> Self {
        Self {
            spectral_analyzer: SpectralAnalyzer::new(config),
        }
    }

    /// Profile noise characteristics.
    pub fn profile(&self, samples: &[f32], sample_rate: f32) -> Result<NoiseProfile> {
        // Analyze spectral characteristics
        let spectral = self.spectral_analyzer.analyze(samples, sample_rate)?;

        // Compute noise type
        let noise_type = super::classify::classify_noise(&spectral);

        // Compute noise level (RMS)
        let noise_level = crate::compute_rms(samples);

        // Estimate noise floor
        let noise_floor = self.estimate_noise_floor(samples);

        Ok(NoiseProfile {
            noise_type,
            noise_level,
            noise_floor,
            spectral_flatness: spectral.flatness,
            spectral_centroid: spectral.centroid,
        })
    }

    /// Estimate noise floor.
    #[allow(clippy::unused_self)]
    fn estimate_noise_floor(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let mut amplitudes: Vec<f32> = samples.iter().map(|&x| x.abs()).collect();
        amplitudes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Use 10th percentile as noise floor
        let idx = (amplitudes.len() as f32 * 0.1) as usize;
        amplitudes[idx]
    }
}

/// Noise profile.
#[derive(Debug, Clone)]
pub struct NoiseProfile {
    /// Type of noise
    pub noise_type: super::classify::NoiseType,
    /// Noise level (RMS)
    pub noise_level: f32,
    /// Noise floor estimate
    pub noise_floor: f32,
    /// Spectral flatness
    pub spectral_flatness: f32,
    /// Spectral centroid
    pub spectral_centroid: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_profiler() {
        let config = AnalysisConfig::default();
        let profiler = NoiseProfiler::new(config);

        // White noise
        let samples = vec![0.1; 8192];
        let profile = profiler.profile(&samples, 44100.0);
        assert!(profile.is_ok());
    }
}
