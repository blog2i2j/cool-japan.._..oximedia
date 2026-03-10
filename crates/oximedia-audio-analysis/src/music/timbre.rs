//! Timbral analysis for sound quality characterization.

use crate::spectral::SpectralAnalyzer;
use crate::{AnalysisConfig, Result};

/// Timbral analyzer for sound quality features.
pub struct TimbralAnalyzer {
    spectral_analyzer: SpectralAnalyzer,
}

impl TimbralAnalyzer {
    /// Create a new timbral analyzer.
    #[must_use]
    pub fn new(config: AnalysisConfig) -> Self {
        Self {
            spectral_analyzer: SpectralAnalyzer::new(config),
        }
    }

    /// Analyze timbral features.
    pub fn analyze(&self, samples: &[f32], sample_rate: f32) -> Result<TimbralFeatures> {
        let spectral = self.spectral_analyzer.analyze(samples, sample_rate)?;

        // Brightness: ratio of energy above/below a threshold frequency
        let brightness = self.compute_brightness(&spectral.magnitude_spectrum, sample_rate);

        // Roughness: amplitude modulation in critical bands
        let roughness = self.compute_roughness(&spectral.magnitude_spectrum);

        // Warmth: emphasis on lower frequencies
        let warmth = self.compute_warmth(&spectral.magnitude_spectrum, sample_rate);

        Ok(TimbralFeatures {
            brightness,
            warmth,
            roughness,
            spectral_centroid: spectral.centroid,
            spectral_flatness: spectral.flatness,
        })
    }

    /// Compute brightness (high-frequency energy ratio).
    #[allow(clippy::unused_self)]
    fn compute_brightness(&self, spectrum: &[f32], sample_rate: f32) -> f32 {
        let threshold_freq = 1500.0; // Hz
        let threshold_bin = (threshold_freq * spectrum.len() as f32 / (sample_rate / 2.0)) as usize;

        if threshold_bin >= spectrum.len() {
            return 0.0;
        }

        let low_energy: f32 = spectrum[..threshold_bin].iter().map(|&x| x * x).sum();
        let high_energy: f32 = spectrum[threshold_bin..].iter().map(|&x| x * x).sum();

        let total = low_energy + high_energy;
        if total > 0.0 {
            high_energy / total
        } else {
            0.0
        }
    }

    /// Compute roughness (sensory dissonance).
    #[allow(clippy::unused_self)]
    fn compute_roughness(&self, spectrum: &[f32]) -> f32 {
        // Simplified roughness: variance in spectral magnitudes
        if spectrum.is_empty() {
            return 0.0;
        }

        let mean = spectrum.iter().sum::<f32>() / spectrum.len() as f32;
        let variance =
            spectrum.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / spectrum.len() as f32;

        variance.sqrt() / (mean + 1e-6)
    }

    /// Compute warmth (low-frequency emphasis).
    #[allow(clippy::unused_self)]
    fn compute_warmth(&self, spectrum: &[f32], sample_rate: f32) -> f32 {
        let warmth_freq = 500.0; // Hz
        let warmth_bin = (warmth_freq * spectrum.len() as f32 / (sample_rate / 2.0)) as usize;

        if warmth_bin >= spectrum.len() {
            return 1.0;
        }

        let low_energy: f32 = spectrum[..warmth_bin].iter().map(|&x| x * x).sum();
        let total_energy: f32 = spectrum.iter().map(|&x| x * x).sum();

        if total_energy > 0.0 {
            low_energy / total_energy
        } else {
            0.0
        }
    }
}

/// Timbral features.
#[derive(Debug, Clone)]
pub struct TimbralFeatures {
    /// Brightness (0-1, higher = brighter)
    pub brightness: f32,
    /// Warmth (0-1, higher = warmer)
    pub warmth: f32,
    /// Roughness (sensory dissonance)
    pub roughness: f32,
    /// Spectral centroid in Hz
    pub spectral_centroid: f32,
    /// Spectral flatness (0-1)
    pub spectral_flatness: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timbral_analyzer() {
        let config = AnalysisConfig::default();
        let analyzer = TimbralAnalyzer::new(config);

        // Generate bright sound (high frequency)
        let sample_rate = 44100.0;
        let samples: Vec<f32> = (0..8192)
            .map(|i| (2.0 * std::f32::consts::PI * 3000.0 * i as f32 / sample_rate).sin())
            .collect();

        let result = analyzer
            .analyze(&samples, sample_rate)
            .expect("analysis should succeed");

        // High frequency should result in high brightness, low warmth
        assert!(result.brightness > 0.5);
        assert!(result.warmth < 0.5);
    }
}
