//! Transient detection using onset strength.

use crate::{AnalysisConfig, Result};

/// Transient detector for detecting attacks and transients.
pub struct TransientDetector {
    config: AnalysisConfig,
    threshold: f32,
}

impl TransientDetector {
    /// Create a new transient detector.
    #[must_use]
    pub fn new(config: AnalysisConfig) -> Self {
        Self {
            config,
            threshold: 0.3, // Onset detection threshold
        }
    }

    /// Detect transients in audio samples.
    pub fn detect(&self, samples: &[f32], sample_rate: f32) -> Result<TransientResult> {
        if samples.is_empty() {
            return Ok(TransientResult::default());
        }

        // Compute onset strength function
        let onset_strength = self.compute_onset_strength(samples);

        // Peak picking to find transients
        let transient_times = self.pick_peaks(&onset_strength, sample_rate);

        // Compute average transient strength
        let avg_strength = if onset_strength.is_empty() {
            0.0
        } else {
            onset_strength.iter().sum::<f32>() / onset_strength.len() as f32
        };

        let num_transients = transient_times.len();

        Ok(TransientResult {
            transient_times,
            onset_strength,
            num_transients,
            avg_strength,
        })
    }

    /// Compute onset strength function using spectral flux.
    fn compute_onset_strength(&self, samples: &[f32]) -> Vec<f32> {
        let window_size = 512;
        let hop_size = self.config.hop_size;

        if samples.len() < window_size * 2 {
            return vec![];
        }

        let num_frames = (samples.len() - window_size) / hop_size;
        let mut onset_strength = Vec::with_capacity(num_frames);

        let mut prev_spectrum = vec![0.0; window_size / 2];

        for frame_idx in 0..num_frames {
            let start = frame_idx * hop_size;
            let end = start + window_size;

            if end > samples.len() {
                break;
            }

            // Compute energy in frequency bands
            let curr_spectrum = self.compute_energy_bands(&samples[start..end]);

            // Spectral flux (sum of positive differences)
            let mut flux = 0.0;
            for (i, &curr_energy) in curr_spectrum.iter().enumerate() {
                let diff = curr_energy - prev_spectrum[i];
                if diff > 0.0 {
                    flux += diff;
                }
            }

            onset_strength.push(flux);
            prev_spectrum = curr_spectrum;
        }

        // Normalize onset strength
        let max_onset = onset_strength.iter().copied().fold(0.0_f32, f32::max);
        if max_onset > 0.0 {
            for strength in &mut onset_strength {
                *strength /= max_onset;
            }
        }

        onset_strength
    }

    /// Compute energy in frequency bands.
    #[allow(clippy::unused_self)]
    fn compute_energy_bands(&self, frame: &[f32]) -> Vec<f32> {
        let num_bands = frame.len() / 2;
        let mut bands = vec![0.0; num_bands];

        let band_size = frame.len() / num_bands;

        for (i, band) in bands.iter_mut().enumerate() {
            let start = i * band_size;
            let end = ((i + 1) * band_size).min(frame.len());

            *band = frame[start..end].iter().map(|&x| x * x).sum::<f32>().sqrt();
        }

        bands
    }

    /// Pick peaks in onset strength function.
    fn pick_peaks(&self, onset_strength: &[f32], sample_rate: f32) -> Vec<f32> {
        if onset_strength.len() < 3 {
            return vec![];
        }

        let mut peaks = Vec::new();
        let hop_duration = self.config.hop_size as f32 / sample_rate;

        for i in 1..(onset_strength.len() - 1) {
            if onset_strength[i] > self.threshold
                && onset_strength[i] > onset_strength[i - 1]
                && onset_strength[i] > onset_strength[i + 1]
            {
                let time = i as f32 * hop_duration;
                peaks.push(time);
            }
        }

        peaks
    }

    /// Set detection threshold.
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.0, 1.0);
    }
}

/// Transient detection result.
#[derive(Debug, Clone)]
pub struct TransientResult {
    /// Times of detected transients in seconds
    pub transient_times: Vec<f32>,
    /// Onset strength function
    pub onset_strength: Vec<f32>,
    /// Number of transients detected
    pub num_transients: usize,
    /// Average onset strength
    pub avg_strength: f32,
}

impl Default for TransientResult {
    fn default() -> Self {
        Self {
            transient_times: Vec::new(),
            onset_strength: Vec::new(),
            num_transients: 0,
            avg_strength: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transient_detector() {
        let config = AnalysisConfig::default();
        let mut detector = TransientDetector::new(config);
        detector.set_threshold(0.5);

        // Generate signal with sharp attacks
        let mut samples = vec![0.0; 44100];

        // Add some transients
        for &pos in &[1000, 10000, 20000, 30000] {
            if pos < samples.len() {
                samples[pos] = 1.0;
                // Decay
                for i in 1..100 {
                    if pos + i < samples.len() {
                        samples[pos + i] = (1.0 - i as f32 / 100.0) * 0.5;
                    }
                }
            }
        }

        let result = detector
            .detect(&samples, 44100.0)
            .expect("detection should succeed");
        assert!(result.num_transients > 0);
    }
}
