//! Audio analysis and quality assessment.
//!
//! This module provides comprehensive audio analysis:
//! - **Silence Detection** - Detect silent segments
//! - **Clipping Detection** - Digital clipping and distortion
//! - **Phase Correlation** - Stereo phase issues
//! - **Spectral Analysis** - Frequency content analysis
//! - **Dynamic Range** - Peak-to-RMS ratio
//!
//! # Implementation
//!
//! Uses FFT for spectral analysis and waveform analysis for time-domain metrics.

use crate::black::{SilenceDetector, SilenceSegment};
use crate::AnalysisResult;
use rustfft::{num_complex::Complex, FftPlanner};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Audio analysis results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioAnalysis {
    /// Detected silence segments
    pub silence_segments: Vec<SilenceSegment>,
    /// Clipping events
    pub clipping_events: Vec<ClippingEvent>,
    /// Phase correlation (stereo only, -1.0 to 1.0)
    pub phase_correlation: Option<f64>,
    /// Spectral analysis
    pub spectral_stats: SpectralStats,
    /// Dynamic range (dB)
    pub dynamic_range_db: f64,
    /// Peak level (dBFS)
    pub peak_dbfs: f64,
    /// RMS level (dBFS)
    pub rms_dbfs: f64,
}

/// Clipping event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClippingEvent {
    /// Sample index where clipping started
    pub start_sample: usize,
    /// Duration in samples
    pub duration: usize,
    /// Peak level during clipping
    pub peak_level: f64,
}

/// Spectral statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralStats {
    /// Spectral centroid (Hz)
    pub centroid_hz: f64,
    /// Spectral flatness (0.0-1.0)
    pub flatness: f64,
    /// Spectral rolloff (Hz)
    pub rolloff_hz: f64,
    /// Low frequency energy ratio (0.0-1.0)
    pub low_freq_ratio: f64,
    /// High frequency energy ratio (0.0-1.0)
    pub high_freq_ratio: f64,
}

/// Audio analyzer.
pub struct AudioAnalyzer {
    silence_detector: SilenceDetector,
    clipping_events: Vec<ClippingEvent>,
    sample_rate: u32,
    peak_level: f64,
    rms_sum: f64,
    sample_count: usize,
    spectral_data: Vec<Vec<f64>>,
    stereo_samples: Vec<(f32, f32)>,
}

impl AudioAnalyzer {
    /// Create a new audio analyzer.
    #[must_use]
    pub fn new(silence_threshold_db: f64, _silence_min_duration: Duration) -> Self {
        let min_duration_samples = 48000; // Default, will be updated
        Self {
            silence_detector: SilenceDetector::new(
                silence_threshold_db,
                min_duration_samples,
                48000,
            ),
            clipping_events: Vec::new(),
            sample_rate: 48000,
            peak_level: 0.0,
            rms_sum: 0.0,
            sample_count: 0,
            spectral_data: Vec::new(),
            stereo_samples: Vec::new(),
        }
    }

    /// Process audio samples (interleaved f32, -1.0 to 1.0).
    pub fn process_samples(&mut self, samples: &[f32], sample_rate: u32) -> AnalysisResult<()> {
        self.sample_rate = sample_rate;

        // Process silence detection
        self.silence_detector.process_samples(samples)?;

        // Detect clipping
        self.detect_clipping(samples);

        // Update peak and RMS
        for &sample in samples {
            let abs_sample = f64::from(sample.abs());
            if abs_sample > self.peak_level {
                self.peak_level = abs_sample;
            }
            self.rms_sum += f64::from(sample * sample);
            self.sample_count += 1;
        }

        // Collect spectral data (process in chunks)
        const FFT_SIZE: usize = 2048;
        if samples.len() >= FFT_SIZE {
            let spectrum = compute_spectrum(&samples[..FFT_SIZE], sample_rate);
            self.spectral_data.push(spectrum);
        }

        // Collect stereo samples for phase correlation
        if samples.len() >= 2 && samples.len() % 2 == 0 {
            for i in (0..samples.len()).step_by(2) {
                self.stereo_samples.push((samples[i], samples[i + 1]));
                if self.stereo_samples.len() >= 10000 {
                    // Limit memory usage
                    break;
                }
            }
        }

        Ok(())
    }

    /// Detect clipping in audio samples.
    fn detect_clipping(&mut self, samples: &[f32]) {
        const CLIP_THRESHOLD: f32 = 0.99;
        let mut in_clip = false;
        let mut clip_start = 0;
        let mut clip_peak = 0.0f32;

        for (i, &sample) in samples.iter().enumerate() {
            let abs_sample = sample.abs();

            if abs_sample >= CLIP_THRESHOLD {
                if in_clip {
                    clip_peak = clip_peak.max(abs_sample);
                } else {
                    clip_start = self.sample_count + i;
                    clip_peak = abs_sample;
                    in_clip = true;
                }
            } else if in_clip {
                // End of clipping event
                let duration = (self.sample_count + i) - clip_start;
                if duration > 10 {
                    // Only report significant clips
                    self.clipping_events.push(ClippingEvent {
                        start_sample: clip_start,
                        duration,
                        peak_level: f64::from(clip_peak),
                    });
                }
                in_clip = false;
            }
        }
    }

    /// Finalize and return audio analysis.
    pub fn finalize(self) -> AudioAnalysis {
        let silence_segments = self.silence_detector.finalize();

        // Compute RMS
        let rms = if self.sample_count > 0 {
            (self.rms_sum / self.sample_count as f64).sqrt()
        } else {
            0.0
        };

        // Convert to dBFS
        let peak_dbfs = if self.peak_level > 0.0 {
            20.0 * self.peak_level.log10()
        } else {
            -100.0
        };

        let rms_dbfs = if rms > 0.0 {
            20.0 * rms.log10()
        } else {
            -100.0
        };

        let dynamic_range_db = peak_dbfs - rms_dbfs;

        // Compute spectral statistics
        let spectral_stats = if self.spectral_data.is_empty() {
            SpectralStats {
                centroid_hz: 0.0,
                flatness: 0.0,
                rolloff_hz: 0.0,
                low_freq_ratio: 0.0,
                high_freq_ratio: 0.0,
            }
        } else {
            compute_spectral_stats(&self.spectral_data, self.sample_rate)
        };

        // Compute phase correlation for stereo
        let phase_correlation = if self.stereo_samples.len() > 100 {
            Some(compute_phase_correlation(&self.stereo_samples))
        } else {
            None
        };

        AudioAnalysis {
            silence_segments,
            clipping_events: self.clipping_events,
            phase_correlation,
            spectral_stats,
            dynamic_range_db,
            peak_dbfs,
            rms_dbfs,
        }
    }
}

/// Compute spectrum using FFT.
fn compute_spectrum(samples: &[f32], _sample_rate: u32) -> Vec<f64> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(samples.len());

    // Convert to complex
    let mut buffer: Vec<Complex<f64>> = samples
        .iter()
        .map(|&s| Complex {
            re: f64::from(s),
            im: 0.0,
        })
        .collect();

    // Apply Hanning window
    for (i, sample) in buffer.iter_mut().enumerate() {
        let window =
            0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / samples.len() as f64).cos());
        sample.re *= window;
    }

    // Perform FFT
    fft.process(&mut buffer);

    // Compute magnitude spectrum (only first half, due to symmetry)
    let half_len = buffer.len() / 2;
    buffer[..half_len]
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im).sqrt())
        .collect()
}

/// Compute spectral statistics.
fn compute_spectral_stats(spectral_data: &[Vec<f64>], sample_rate: u32) -> SpectralStats {
    // Average all spectra
    let spectrum_len = spectral_data[0].len();
    let mut avg_spectrum = vec![0.0; spectrum_len];

    for spectrum in spectral_data {
        for (i, &mag) in spectrum.iter().enumerate() {
            avg_spectrum[i] += mag;
        }
    }

    for mag in &mut avg_spectrum {
        *mag /= spectral_data.len() as f64;
    }

    let total_energy: f64 = avg_spectrum.iter().sum();

    if total_energy < f64::EPSILON {
        return SpectralStats {
            centroid_hz: 0.0,
            flatness: 0.0,
            rolloff_hz: 0.0,
            low_freq_ratio: 0.0,
            high_freq_ratio: 0.0,
        };
    }

    // Compute spectral centroid
    let mut weighted_sum = 0.0;
    for (i, &mag) in avg_spectrum.iter().enumerate() {
        let freq = i as f64 * f64::from(sample_rate) / (2 * spectrum_len) as f64;
        weighted_sum += freq * mag;
    }
    let centroid_hz = weighted_sum / total_energy;

    // Compute spectral flatness (geometric mean / arithmetic mean)
    let geometric_mean =
        avg_spectrum.iter().map(|&x| (x + 1e-10).ln()).sum::<f64>() / spectrum_len as f64;
    let geometric_mean = geometric_mean.exp();
    let arithmetic_mean = total_energy / spectrum_len as f64;
    let flatness = if arithmetic_mean > 0.0 {
        (geometric_mean / arithmetic_mean).min(1.0)
    } else {
        0.0
    };

    // Compute spectral rolloff (95% energy threshold)
    let mut cumulative = 0.0;
    let threshold = total_energy * 0.95;
    let mut rolloff_hz = 0.0;
    for (i, &mag) in avg_spectrum.iter().enumerate() {
        cumulative += mag;
        if cumulative >= threshold {
            rolloff_hz = i as f64 * f64::from(sample_rate) / (2 * spectrum_len) as f64;
            break;
        }
    }

    // Compute frequency band ratios
    let low_cutoff = (500.0 * (2 * spectrum_len) as f64 / f64::from(sample_rate)) as usize;
    let high_cutoff = (4000.0 * (2 * spectrum_len) as f64 / f64::from(sample_rate)) as usize;

    let low_freq_energy: f64 = avg_spectrum[..low_cutoff.min(spectrum_len)].iter().sum();
    let high_freq_energy: f64 = avg_spectrum[high_cutoff.min(spectrum_len)..].iter().sum();

    SpectralStats {
        centroid_hz,
        flatness,
        rolloff_hz,
        low_freq_ratio: low_freq_energy / total_energy,
        high_freq_ratio: high_freq_energy / total_energy,
    }
}

/// Compute phase correlation for stereo audio.
fn compute_phase_correlation(samples: &[(f32, f32)]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }

    let mut sum_lr = 0.0;
    let mut sum_l2 = 0.0;
    let mut sum_r2 = 0.0;

    for &(l, r) in samples {
        sum_lr += f64::from(l * r);
        sum_l2 += f64::from(l * l);
        sum_r2 += f64::from(r * r);
    }

    let denominator = (sum_l2 * sum_r2).sqrt();
    if denominator > f64::EPSILON {
        (sum_lr / denominator).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_analyzer() {
        let analyzer = AudioAnalyzer::new(-60.0, Duration::from_millis(500));

        // Create test audio
        let samples = vec![0.5f32; 48000];
        let mut analyzer_mut = analyzer;
        analyzer_mut
            .process_samples(&samples, 48000)
            .expect("sample processing should succeed");

        let analysis = analyzer_mut.finalize();
        assert!(analysis.peak_dbfs < 0.0);
        assert!(analysis.dynamic_range_db >= 0.0);
    }

    #[test]
    fn test_clipping_detection() {
        let mut analyzer = AudioAnalyzer::new(-60.0, Duration::from_millis(500));

        // Create clipped audio
        let mut samples = vec![0.5f32; 1000];
        for i in 100..200 {
            samples[i] = 1.0; // Clipped
        }

        analyzer
            .process_samples(&samples, 48000)
            .expect("sample processing should succeed");
        let analysis = analyzer.finalize();

        assert!(!analysis.clipping_events.is_empty());
    }

    #[test]
    fn test_phase_correlation() {
        // Perfect correlation (identical channels)
        let samples: Vec<(f32, f32)> = (0..1000)
            .map(|i| {
                let val = (i as f32 * 0.01).sin();
                (val, val)
            })
            .collect();
        let corr = compute_phase_correlation(&samples);
        assert!(corr > 0.99);

        // Perfect anti-correlation (inverted channels)
        let samples2: Vec<(f32, f32)> = (0..1000)
            .map(|i| {
                let val = (i as f32 * 0.01).sin();
                (val, -val)
            })
            .collect();
        let corr2 = compute_phase_correlation(&samples2);
        assert!(corr2 < -0.99);
    }

    #[test]
    fn test_spectrum_computation() {
        // Create a simple sine wave
        let sample_rate = 48000;
        let freq = 1000.0;
        let samples: Vec<f32> = (0..2048)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
            .collect();

        let spectrum = compute_spectrum(&samples, sample_rate);
        assert_eq!(spectrum.len(), 1024);

        // Peak should be near 1000 Hz bin
        let peak_bin = spectrum
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let peak_freq = peak_bin as f64 * sample_rate as f64 / 2048.0;
        assert!((peak_freq - 1000.0).abs() < 50.0);
    }

    #[test]
    fn test_spectral_stats() {
        let sample_rate = 48000;
        let spectrum = vec![vec![1.0; 1024]]; // Flat spectrum
        let stats = compute_spectral_stats(&spectrum, sample_rate);

        assert!(stats.flatness > 0.9); // Should be high for flat spectrum
        assert!(stats.centroid_hz > 0.0);
    }

    #[test]
    fn test_dynamic_range() {
        let mut analyzer = AudioAnalyzer::new(-60.0, Duration::from_millis(500));

        // Low dynamic range (constant level)
        let samples = vec![0.5f32; 48000];
        analyzer
            .process_samples(&samples, 48000)
            .expect("sample processing should succeed");
        let analysis = analyzer.finalize();

        assert!(analysis.dynamic_range_db < 1.0); // Very low dynamic range
    }

    #[test]
    fn test_empty_audio() {
        let analyzer = AudioAnalyzer::new(-60.0, Duration::from_millis(500));
        let analysis = analyzer.finalize();
        assert_eq!(analysis.peak_dbfs, -100.0);
        assert_eq!(analysis.rms_dbfs, -100.0);
    }
}
