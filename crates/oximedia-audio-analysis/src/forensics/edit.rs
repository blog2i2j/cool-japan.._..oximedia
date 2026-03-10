//! Edit detection (cuts, splices, insertions).

use crate::spectral::SpectralAnalyzer;
use crate::{AnalysisConfig, Result};

/// Edit detector for detecting cuts and splices.
pub struct EditDetector {
    spectral_analyzer: SpectralAnalyzer,
    hop_size: usize,
}

impl EditDetector {
    /// Create a new edit detector.
    #[must_use]
    pub fn new(config: AnalysisConfig) -> Self {
        let hop_size = config.hop_size;
        Self {
            spectral_analyzer: SpectralAnalyzer::new(config),
            hop_size,
        }
    }

    /// Detect edits in audio.
    pub fn detect(&self, samples: &[f32], sample_rate: f32) -> Result<EditResult> {
        // Look for discontinuities in:
        // 1. Amplitude envelope
        // 2. Spectral characteristics
        // 3. Phase continuity

        let edit_times = self.detect_discontinuities(samples, sample_rate)?;

        Ok(EditResult {
            num_edits: edit_times.len(),
            edit_times,
        })
    }

    /// Detect discontinuities indicating edits.
    fn detect_discontinuities(&self, samples: &[f32], sample_rate: f32) -> Result<Vec<f32>> {
        let window_size = 2048;
        let mut edits = Vec::new();

        if samples.len() < window_size * 2 {
            return Ok(edits);
        }

        // Compute spectral features over time
        let num_frames = (samples.len() - window_size) / self.hop_size;
        let mut spectral_centroids = Vec::new();
        let mut energies = Vec::new();

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.hop_size;
            let end = (start + window_size).min(samples.len());

            if end - start < window_size {
                break;
            }

            let frame = &samples[start..end];

            // Compute energy
            let energy: f32 = frame.iter().map(|&x| x * x).sum();
            energies.push(energy);

            // Compute spectral centroid
            let features = self.spectral_analyzer.analyze_frame(frame, sample_rate)?;
            spectral_centroids.push(features.centroid);
        }

        // Look for sudden changes
        let threshold = 3.0; // Standard deviations

        // Check energy discontinuities
        for i in 1..energies.len() {
            let diff = (energies[i] - energies[i - 1]).abs();
            let mean = (energies[i] + energies[i - 1]) / 2.0;

            if mean > 0.0 && diff / mean > threshold {
                let time = (i * self.hop_size) as f32 / sample_rate;
                edits.push(time);
            }
        }

        // Check spectral centroid discontinuities
        for i in 1..spectral_centroids.len() {
            let diff = (spectral_centroids[i] - spectral_centroids[i - 1]).abs();

            if diff > 500.0 {
                // Large change in spectral centroid
                let time = (i * self.hop_size) as f32 / sample_rate;
                if !edits.contains(&time) {
                    edits.push(time);
                }
            }
        }

        edits.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        Ok(edits)
    }
}

/// Edit detection result.
#[derive(Debug, Clone)]
pub struct EditResult {
    /// Number of detected edits
    pub num_edits: usize,
    /// Times of detected edits in seconds
    pub edit_times: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edit_detector() {
        let config = AnalysisConfig::default();
        let detector = EditDetector::new(config);

        // Generate signal with splice
        let sample_rate = 44100.0;
        let mut samples = Vec::new();

        // First part: 440 Hz
        for i in 0..22050 {
            samples.push((2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate).sin() * 0.5);
        }

        // Second part: 880 Hz (sudden change)
        for i in 0..22050 {
            samples.push((2.0 * std::f32::consts::PI * 880.0 * i as f32 / sample_rate).sin() * 0.5);
        }

        let result = detector.detect(&samples, sample_rate);
        assert!(result.is_ok());
    }
}
