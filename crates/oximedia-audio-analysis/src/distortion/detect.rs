//! Distortion detection.

use crate::{AnalysisConfig, Result};

/// Distortion detector.
pub struct DistortionDetector {
    #[allow(dead_code)]
    config: AnalysisConfig,
}

impl DistortionDetector {
    /// Create a new distortion detector.
    #[must_use]
    pub fn new(config: AnalysisConfig) -> Self {
        Self { config }
    }

    /// Detect distortion in audio samples.
    pub fn detect(&self, samples: &[f32], sample_rate: f32) -> Result<DistortionResult> {
        // Detect clipping
        let clipping = super::clipping::detect_clipping(samples, 0.99);

        // Compute THD
        let thd = super::thd::total_harmonic_distortion(samples, sample_rate);

        // Overall distortion score
        let distortion_score = (clipping.clipping_ratio * 0.5 + thd * 0.5).min(1.0);

        Ok(DistortionResult {
            has_distortion: distortion_score > 0.05,
            distortion_score,
            thd,
            clipping_detected: clipping.has_clipping,
        })
    }
}

/// Distortion detection result.
#[derive(Debug, Clone)]
pub struct DistortionResult {
    /// Whether distortion is detected
    pub has_distortion: bool,
    /// Overall distortion score (0-1)
    pub distortion_score: f32,
    /// Total harmonic distortion
    pub thd: f32,
    /// Whether clipping is detected
    pub clipping_detected: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distortion_detector() {
        let config = AnalysisConfig::default();
        let detector = DistortionDetector::new(config);

        // Generate clean signal
        let sample_rate = 44100.0;
        let samples: Vec<f32> = (0..4096)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate).sin() * 0.5)
            .collect();

        let result = detector
            .detect(&samples, sample_rate)
            .expect("detection should succeed");
        assert!(!result.has_distortion || result.distortion_score < 0.1);
    }
}
