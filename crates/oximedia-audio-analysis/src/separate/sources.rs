//! Source separation using harmonic-percussive decomposition.

use crate::{AnalysisConfig, AnalysisError, Result};
use rustfft::{num_complex::Complex, FftPlanner};
use std::sync::Arc;

/// Source separator for separating audio sources.
pub struct SourceSeparator {
    config: AnalysisConfig,
    fft: Arc<dyn rustfft::Fft<f32>>,
    ifft: Arc<dyn rustfft::Fft<f32>>,
}

impl SourceSeparator {
    /// Create a new source separator.
    #[must_use]
    pub fn new(config: AnalysisConfig) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(config.fft_size);
        let ifft = planner.plan_fft_inverse(config.fft_size);

        Self { config, fft, ifft }
    }

    /// Separate harmonic and percussive components.
    pub fn separate_harmonic_percussive(
        &self,
        samples: &[f32],
        _sample_rate: f32,
    ) -> Result<SeparationResult> {
        if samples.len() < self.config.fft_size {
            return Err(AnalysisError::InsufficientSamples {
                needed: self.config.fft_size,
                got: samples.len(),
            });
        }

        // Compute spectrogram
        let spectrogram = self.compute_spectrogram(samples)?;

        // Apply median filtering
        let harmonic_spec = self.median_filter_horizontal(&spectrogram);
        let percussive_spec = self.median_filter_vertical(&spectrogram);

        // Synthesize separated sources
        let harmonic = self.synthesize(&harmonic_spec)?;
        let percussive = self.synthesize(&percussive_spec)?;

        Ok(SeparationResult {
            harmonic,
            percussive,
            residual: vec![],
        })
    }

    /// Compute spectrogram.
    #[allow(clippy::unnecessary_wraps)]
    fn compute_spectrogram(&self, samples: &[f32]) -> Result<Vec<Vec<Complex<f32>>>> {
        let hop_size = self.config.hop_size;
        let window_size = self.config.fft_size;

        let num_frames = (samples.len() - window_size) / hop_size + 1;
        let mut spectrogram = Vec::with_capacity(num_frames);

        for frame_idx in 0..num_frames {
            let start = frame_idx * hop_size;
            let end = start + window_size;

            if end > samples.len() {
                break;
            }

            let mut frame: Vec<Complex<f32>> = samples[start..end]
                .iter()
                .map(|&s| Complex::new(s, 0.0))
                .collect();

            self.fft.process(&mut frame);
            spectrogram.push(frame);
        }

        Ok(spectrogram)
    }

    /// Apply horizontal median filtering (enhances harmonic content).
    #[allow(clippy::unused_self)]
    fn median_filter_horizontal(
        &self,
        spectrogram: &[Vec<Complex<f32>>],
    ) -> Vec<Vec<Complex<f32>>> {
        let kernel_size = 17;
        let mut filtered = spectrogram.to_vec();

        for (time_idx, frame) in filtered.iter_mut().enumerate() {
            for freq_idx in 0..frame.len() {
                let start = time_idx.saturating_sub(kernel_size / 2);
                let end = (time_idx + kernel_size / 2 + 1).min(spectrogram.len());

                let mut values: Vec<f32> = (start..end)
                    .map(|t| spectrogram[t][freq_idx].norm())
                    .collect();

                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median = values[values.len() / 2];

                // Preserve phase, update magnitude
                let phase = frame[freq_idx].arg();
                frame[freq_idx] = Complex::from_polar(median, phase);
            }
        }

        filtered
    }

    /// Apply vertical median filtering (enhances percussive content).
    #[allow(clippy::unused_self)]
    fn median_filter_vertical(&self, spectrogram: &[Vec<Complex<f32>>]) -> Vec<Vec<Complex<f32>>> {
        let kernel_size = 17;
        let mut filtered = spectrogram.to_vec();

        for (time_idx, frame) in filtered.iter_mut().enumerate() {
            for freq_idx in 0..frame.len() {
                let start = freq_idx.saturating_sub(kernel_size / 2);
                let end = (freq_idx + kernel_size / 2 + 1).min(frame.len());

                let mut values: Vec<f32> = (start..end)
                    .map(|f| spectrogram[time_idx][f].norm())
                    .collect();

                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median = values[values.len() / 2];

                // Preserve phase, update magnitude
                let phase = frame[freq_idx].arg();
                frame[freq_idx] = Complex::from_polar(median, phase);
            }
        }

        filtered
    }

    /// Synthesize audio from spectrogram using overlap-add.
    #[allow(clippy::unnecessary_wraps)]
    fn synthesize(&self, spectrogram: &[Vec<Complex<f32>>]) -> Result<Vec<f32>> {
        let hop_size = self.config.hop_size;
        let window_size = self.config.fft_size;
        let output_len = (spectrogram.len() - 1) * hop_size + window_size;

        let mut output = vec![0.0; output_len];
        let mut window_sum = vec![0.0; output_len];

        for (frame_idx, frame) in spectrogram.iter().enumerate() {
            let mut ifft_frame = frame.clone();
            self.ifft.process(&mut ifft_frame);

            let start = frame_idx * hop_size;

            for (i, &value) in ifft_frame.iter().enumerate().take(window_size) {
                if start + i < output.len() {
                    output[start + i] += value.re / window_size as f32;
                    window_sum[start + i] += 1.0;
                }
            }
        }

        // Normalize by window sum
        for (i, &sum) in window_sum.iter().enumerate() {
            if sum > 0.0 {
                output[i] /= sum;
            }
        }

        Ok(output)
    }
}

/// Source separation result.
#[derive(Debug, Clone)]
pub struct SeparationResult {
    /// Harmonic component
    pub harmonic: Vec<f32>,
    /// Percussive component
    pub percussive: Vec<f32>,
    /// Residual component
    pub residual: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_separator() {
        let config = AnalysisConfig::default();
        let separator = SourceSeparator::new(config);

        // Generate mixed signal (sine + impulses)
        let sample_rate = 44100.0;
        let mut samples = vec![0.0; 8192];

        for i in 0..samples.len() {
            // Harmonic component
            samples[i] += (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate).sin() * 0.3;

            // Percussive component
            if i % 1000 == 0 {
                samples[i] += 0.8;
            }
        }

        let result = separator.separate_harmonic_percussive(&samples, sample_rate);
        assert!(result.is_ok());

        let separation = result.expect("expected successful result");
        assert_eq!(separation.harmonic.len(), samples.len());
        assert_eq!(separation.percussive.len(), samples.len());
    }
}
