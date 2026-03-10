//! Main spectral analysis implementation.

use crate::{generate_window, AnalysisConfig, AnalysisError, Result};
use rustfft::{num_complex::Complex, FftPlanner};
use std::sync::Arc;

/// Spectral analyzer for frequency-domain analysis.
pub struct SpectralAnalyzer {
    config: AnalysisConfig,
    fft: Arc<dyn rustfft::Fft<f32>>,
    window: Vec<f32>,
}

impl SpectralAnalyzer {
    /// Create a new spectral analyzer with the given configuration.
    #[must_use]
    pub fn new(config: AnalysisConfig) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(config.fft_size);
        let window = generate_window(config.window_type, config.fft_size);

        Self {
            config,
            fft,
            window,
        }
    }

    /// Perform spectral analysis on audio samples.
    pub fn analyze(&self, samples: &[f32], sample_rate: f32) -> Result<SpectralFeatures> {
        if samples.len() < self.config.fft_size {
            return Err(AnalysisError::InsufficientSamples {
                needed: self.config.fft_size,
                got: samples.len(),
            });
        }

        // Compute average spectrum over all frames
        let num_frames = (samples.len() - self.config.fft_size) / self.config.hop_size + 1;
        let mut avg_magnitude = vec![0.0_f32; self.config.fft_size / 2 + 1];
        let mut prev_magnitude: Option<Vec<f32>> = None;
        let mut total_flux = 0.0_f32;
        let mut flux_count = 0usize;

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.config.hop_size;
            let end = start + self.config.fft_size;
            if end > samples.len() {
                break;
            }

            let frame_spectrum = self.compute_spectrum(&samples[start..end])?;

            // Accumulate flux
            if let Some(ref prev) = prev_magnitude {
                total_flux += super::flux::spectral_flux(&frame_spectrum, prev);
                flux_count += 1;
            }

            for (i, &mag) in frame_spectrum.iter().enumerate() {
                avg_magnitude[i] += mag;
            }

            prev_magnitude = Some(frame_spectrum);
        }

        // Average the magnitude spectrum
        let actual_frames = num_frames.max(1) as f32;
        for mag in &mut avg_magnitude {
            *mag /= actual_frames;
        }

        // Compute spectral features
        let centroid = super::centroid::spectral_centroid(&avg_magnitude, sample_rate);
        let flatness = super::flatness::spectral_flatness(&avg_magnitude);
        let crest = super::crest::spectral_crest(&avg_magnitude);
        let bandwidth = super::bandwidth::spectral_bandwidth(&avg_magnitude, sample_rate, centroid);
        let rolloff = super::rolloff::spectral_rolloff_85(&avg_magnitude, sample_rate);
        let flux = if flux_count > 0 {
            total_flux / flux_count as f32
        } else {
            0.0
        };

        Ok(SpectralFeatures {
            centroid,
            flatness,
            crest,
            bandwidth,
            rolloff,
            flux,
            magnitude_spectrum: avg_magnitude,
        })
    }

    /// Analyze a single frame for real-time processing.
    pub fn analyze_frame(&self, samples: &[f32], sample_rate: f32) -> Result<SpectralFeatures> {
        let magnitude = self.compute_spectrum(samples)?;

        let centroid = super::centroid::spectral_centroid(&magnitude, sample_rate);
        let flatness = super::flatness::spectral_flatness(&magnitude);
        let crest = super::crest::spectral_crest(&magnitude);
        let bandwidth = super::bandwidth::spectral_bandwidth(&magnitude, sample_rate, centroid);
        let rolloff = super::rolloff::spectral_rolloff_85(&magnitude, sample_rate);

        Ok(SpectralFeatures {
            centroid,
            flatness,
            crest,
            bandwidth,
            rolloff,
            flux: 0.0,
            magnitude_spectrum: magnitude,
        })
    }

    /// Compute magnitude spectrum for a frame.
    fn compute_spectrum(&self, samples: &[f32]) -> Result<Vec<f32>> {
        if samples.len() != self.config.fft_size {
            return Err(AnalysisError::InvalidInput(format!(
                "Expected {} samples, got {}",
                self.config.fft_size,
                samples.len()
            )));
        }

        // Apply window and convert to complex
        let mut buffer: Vec<Complex<f32>> = samples
            .iter()
            .zip(&self.window)
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect();

        // Perform FFT
        self.fft.process(&mut buffer);

        // Compute magnitude spectrum (only positive frequencies)
        let magnitude: Vec<f32> = buffer[..=(self.config.fft_size / 2)]
            .iter()
            .map(|c| c.norm())
            .collect();

        Ok(magnitude)
    }
}

/// Spectral features extracted from audio.
#[derive(Debug, Clone)]
pub struct SpectralFeatures {
    /// Spectral centroid in Hz (center of mass of spectrum)
    pub centroid: f32,
    /// Spectral flatness (0-1, higher = more noise-like)
    pub flatness: f32,
    /// Spectral crest factor (peak-to-average ratio)
    pub crest: f32,
    /// Spectral bandwidth in Hz
    pub bandwidth: f32,
    /// Spectral rolloff frequency in Hz (85% energy threshold)
    pub rolloff: f32,
    /// Spectral flux (change from previous frame)
    pub flux: f32,
    /// Full magnitude spectrum
    pub magnitude_spectrum: Vec<f32>,
}

impl Default for SpectralFeatures {
    fn default() -> Self {
        Self {
            centroid: 0.0,
            flatness: 0.0,
            crest: 0.0,
            bandwidth: 0.0,
            rolloff: 0.0,
            flux: 0.0,
            magnitude_spectrum: Vec::new(),
        }
    }
}

/// Convert FFT bin index to frequency in Hz.
#[must_use]
pub fn bin_to_frequency(bin: usize, sample_rate: f32, fft_size: usize) -> f32 {
    bin as f32 * sample_rate / fft_size as f32
}

/// Convert frequency to FFT bin index.
#[must_use]
pub fn frequency_to_bin(frequency: f32, sample_rate: f32, fft_size: usize) -> usize {
    ((frequency * fft_size as f32 / sample_rate).round() as usize).min(fft_size / 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_analyzer() {
        let config = AnalysisConfig::default();
        let analyzer = SpectralAnalyzer::new(config);

        // Generate 1 second of 440 Hz sine wave
        let sample_rate = 44100.0;
        let duration = 1.0;
        let frequency = 440.0;
        let samples: Vec<f32> = (0..(sample_rate * duration) as usize)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (2.0 * std::f32::consts::PI * frequency * t).sin()
            })
            .collect();

        let result = analyzer
            .analyze(&samples, sample_rate)
            .expect("analysis should succeed");

        // Just verify we get reasonable values
        assert!(result.centroid > 0.0 && result.centroid < sample_rate / 2.0);
        // Flatness should be between 0 and 1
        assert!(result.flatness >= 0.0 && result.flatness <= 1.0);
    }

    #[test]
    fn test_bin_frequency_conversion() {
        let sample_rate = 44100.0;
        let fft_size = 2048;

        let freq = 440.0;
        let bin = frequency_to_bin(freq, sample_rate, fft_size);
        let back = bin_to_frequency(bin, sample_rate, fft_size);

        assert!((freq - back).abs() < sample_rate / fft_size as f32);
    }
}
