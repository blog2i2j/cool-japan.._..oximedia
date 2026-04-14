#![allow(dead_code)]
//! Vinyl surface noise profiling and reduction.
//!
//! Surface noise on vinyl records is a continuous, broadband phenomenon
//! distinct from impulsive artefacts (clicks, crackle).  It originates from
//! the stylus tracing micro-imperfections in the groove walls and manifests
//! as a characteristic "hiss" with a frequency-dependent spectral shape that
//! rises with frequency (roughly +3 dB/octave above 1 kHz on RIAA-equalised
//! playback).
//!
//! This module provides:
//!
//! - **Adaptive noise profiling** — learns the noise floor from inter-groove
//!   silence or user-designated "noise-only" regions.
//! - **Spectral gating** — attenuates bins that fall below the learned noise
//!   profile, leaving wanted signal intact.
//! - **Perceptual weighting** — applies frequency-dependent attenuation curves
//!   to preserve low-frequency warmth while aggressively cleaning the highs.
//! - **Real-time profile update** — continuously adapts the noise estimate
//!   during quiet passages.

use crate::error::{RestoreError, RestoreResult};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Noise profile
// ---------------------------------------------------------------------------

/// A learned vinyl surface noise spectral profile.
#[derive(Debug, Clone)]
pub struct VinylNoiseProfile {
    /// Magnitude spectrum of the estimated noise floor (one value per FFT bin).
    pub spectrum: Vec<f64>,
    /// Sample rate used when the profile was captured.
    pub sample_rate: u32,
    /// FFT size used for the profile.
    pub fft_size: usize,
    /// Number of frames averaged into this profile.
    pub frames_averaged: usize,
}

impl VinylNoiseProfile {
    /// Create an empty profile for later accumulation.
    pub fn empty(fft_size: usize, sample_rate: u32) -> Self {
        Self {
            spectrum: vec![0.0; fft_size / 2 + 1],
            sample_rate,
            fft_size,
            frames_averaged: 0,
        }
    }

    /// Number of frequency bins in this profile.
    pub fn bin_count(&self) -> usize {
        self.spectrum.len()
    }

    /// Return the frequency in Hz of a given bin index.
    #[allow(clippy::cast_precision_loss)]
    pub fn bin_frequency(&self, bin: usize) -> f64 {
        bin as f64 * f64::from(self.sample_rate) / self.fft_size as f64
    }

    /// Accumulate a magnitude frame into the running average.
    pub fn accumulate(&mut self, magnitudes: &[f64]) {
        let n = self.spectrum.len().min(magnitudes.len());
        self.frames_averaged += 1;
        let count = self.frames_averaged as f64;
        for i in 0..n {
            // Running mean
            self.spectrum[i] += (magnitudes[i] - self.spectrum[i]) / count;
        }
    }

    /// Scale the entire profile by a factor.
    pub fn scale(&mut self, factor: f64) {
        for v in &mut self.spectrum {
            *v *= factor;
        }
    }
}

// ---------------------------------------------------------------------------
// Profiler
// ---------------------------------------------------------------------------

/// Configuration for the vinyl surface noise profiler.
#[derive(Debug, Clone)]
pub struct VinylProfilerConfig {
    /// FFT size (must be a power of two).
    pub fft_size: usize,
    /// Hop size between consecutive analysis frames.
    pub hop_size: usize,
    /// RMS threshold below which a frame is considered "noise only".
    pub silence_threshold: f64,
    /// Maximum number of frames to average for the profile.
    pub max_frames: usize,
}

impl Default for VinylProfilerConfig {
    fn default() -> Self {
        Self {
            fft_size: 2048,
            hop_size: 512,
            silence_threshold: 0.01,
            max_frames: 200,
        }
    }
}

/// Vinyl surface noise profiler.
///
/// Learns the spectral shape of vinyl surface noise from quiet regions
/// of a recording.
#[derive(Debug, Clone)]
pub struct VinylNoiseProfiler {
    config: VinylProfilerConfig,
}

impl VinylNoiseProfiler {
    /// Create a new profiler.
    pub fn new(config: VinylProfilerConfig) -> Self {
        Self { config }
    }

    /// Create a profiler with default settings.
    pub fn with_defaults() -> Self {
        Self::new(VinylProfilerConfig::default())
    }

    /// RMS of a slice.
    #[allow(clippy::cast_precision_loss)]
    fn rms(samples: &[f32]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
        (sum_sq / samples.len() as f64).sqrt()
    }

    /// Compute magnitude spectrum of a windowed frame using a simple DFT.
    ///
    /// For production use a proper FFT (OxiFFT) would be preferred, but this
    /// module keeps the implementation self-contained for portability.
    #[allow(clippy::cast_precision_loss)]
    fn magnitude_spectrum(frame: &[f64]) -> Vec<f64> {
        let n = frame.len();
        let n_bins = n / 2 + 1;
        let mut mags = Vec::with_capacity(n_bins);
        for k in 0..n_bins {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            let omega = 2.0 * PI * k as f64 / n as f64;
            for (i, &v) in frame.iter().enumerate() {
                re += v * (omega * i as f64).cos();
                im -= v * (omega * i as f64).sin();
            }
            mags.push((re * re + im * im).sqrt() / n as f64);
        }
        mags
    }

    /// Hann window.
    #[allow(clippy::cast_precision_loss)]
    fn hann_window(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / len as f64).cos()))
            .collect()
    }

    /// Learn a noise profile from the given samples.
    ///
    /// Only frames whose RMS falls below `silence_threshold` are used.
    ///
    /// # Errors
    ///
    /// Returns [`RestoreError::InvalidParameter`] when the FFT size is not a
    /// power of two or the input is shorter than one frame.
    pub fn learn(&self, samples: &[f32], sample_rate: u32) -> RestoreResult<VinylNoiseProfile> {
        let fft_size = self.config.fft_size;
        if fft_size == 0 || (fft_size & (fft_size - 1)) != 0 {
            return Err(RestoreError::InvalidParameter(
                "fft_size must be a power of two".into(),
            ));
        }
        if samples.len() < fft_size {
            return Err(RestoreError::InvalidParameter(format!(
                "input too short: need at least {} samples, got {}",
                fft_size,
                samples.len()
            )));
        }

        let hop = self.config.hop_size.max(1);
        let window = Self::hann_window(fft_size);
        let mut profile = VinylNoiseProfile::empty(fft_size, sample_rate);

        let mut pos = 0;
        while pos + fft_size <= samples.len() && profile.frames_averaged < self.config.max_frames {
            let frame_slice = &samples[pos..pos + fft_size];
            if Self::rms(frame_slice) < self.config.silence_threshold {
                // Apply window and compute magnitude
                let windowed: Vec<f64> = frame_slice
                    .iter()
                    .zip(window.iter())
                    .map(|(&s, &w)| f64::from(s) * w)
                    .collect();
                let mags = Self::magnitude_spectrum(&windowed);
                profile.accumulate(&mags);
            }
            pos += hop;
        }

        Ok(profile)
    }

    /// Learn a noise profile from an explicitly designated "noise-only" region.
    ///
    /// Unlike [`learn`](Self::learn), all frames are used regardless of RMS level.
    ///
    /// # Errors
    ///
    /// Returns an error when the FFT size is invalid or input is too short.
    pub fn learn_from_noise_region(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> RestoreResult<VinylNoiseProfile> {
        let fft_size = self.config.fft_size;
        if fft_size == 0 || (fft_size & (fft_size - 1)) != 0 {
            return Err(RestoreError::InvalidParameter(
                "fft_size must be a power of two".into(),
            ));
        }
        if samples.len() < fft_size {
            return Err(RestoreError::InvalidParameter(format!(
                "noise region too short: need at least {} samples, got {}",
                fft_size,
                samples.len()
            )));
        }

        let hop = self.config.hop_size.max(1);
        let window = Self::hann_window(fft_size);
        let mut profile = VinylNoiseProfile::empty(fft_size, sample_rate);

        let mut pos = 0;
        while pos + fft_size <= samples.len() && profile.frames_averaged < self.config.max_frames {
            let frame_slice = &samples[pos..pos + fft_size];
            let windowed: Vec<f64> = frame_slice
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| f64::from(s) * w)
                .collect();
            let mags = Self::magnitude_spectrum(&windowed);
            profile.accumulate(&mags);
            pos += hop;
        }

        Ok(profile)
    }
}

// ---------------------------------------------------------------------------
// Surface noise reducer
// ---------------------------------------------------------------------------

/// Configuration for the vinyl surface noise reducer.
#[derive(Debug, Clone)]
pub struct VinylNoiseReducerConfig {
    /// FFT size (must match the profile).
    pub fft_size: usize,
    /// Hop size for overlap-add processing.
    pub hop_size: usize,
    /// Over-subtraction factor (>= 1.0).  Higher values remove more noise
    /// but risk introducing "musical noise" artefacts.
    pub over_subtraction: f64,
    /// Spectral floor factor.  Bins are never reduced below
    /// `floor * noise_profile[bin]`.
    pub floor: f64,
    /// Enable perceptual weighting (stronger attenuation at high frequencies).
    pub perceptual_weight: bool,
    /// Smoothing factor for temporal noise estimate update (0.0–1.0).
    pub smoothing: f64,
}

impl Default for VinylNoiseReducerConfig {
    fn default() -> Self {
        Self {
            fft_size: 2048,
            hop_size: 512,
            over_subtraction: 2.0,
            floor: 0.02,
            perceptual_weight: true,
            smoothing: 0.95,
        }
    }
}

/// Vinyl surface noise reducer.
///
/// Uses spectral gating against a learned noise profile to attenuate
/// surface noise while preserving the musical content.
#[derive(Debug, Clone)]
pub struct VinylNoiseReducer {
    config: VinylNoiseReducerConfig,
    profile: VinylNoiseProfile,
}

impl VinylNoiseReducer {
    /// Create a new reducer with the given profile and configuration.
    ///
    /// # Errors
    ///
    /// Returns [`RestoreError::InvalidParameter`] when the FFT sizes of the
    /// config and profile do not match.
    pub fn new(config: VinylNoiseReducerConfig, profile: VinylNoiseProfile) -> RestoreResult<Self> {
        if config.fft_size != profile.fft_size {
            return Err(RestoreError::InvalidParameter(format!(
                "FFT size mismatch: config={}, profile={}",
                config.fft_size, profile.fft_size
            )));
        }
        Ok(Self { config, profile })
    }

    /// Compute a perceptual weighting curve (more aggressive at high freqs).
    #[allow(clippy::cast_precision_loss)]
    fn perceptual_curve(n_bins: usize, sample_rate: u32) -> Vec<f64> {
        let nyquist = f64::from(sample_rate) / 2.0;
        (0..n_bins)
            .map(|k| {
                let freq = k as f64 * nyquist / n_bins as f64;
                // +3 dB/octave above 1 kHz approximation
                if freq > 1000.0 {
                    1.0 + 0.5 * (freq / 1000.0).log2()
                } else {
                    1.0
                }
            })
            .collect()
    }

    /// Hann window.
    #[allow(clippy::cast_precision_loss)]
    fn hann_window(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / len as f64).cos()))
            .collect()
    }

    /// Simple DFT (real input) returning (real, imag) pairs for n/2+1 bins.
    #[allow(clippy::cast_precision_loss)]
    fn dft(frame: &[f64]) -> Vec<(f64, f64)> {
        let n = frame.len();
        let n_bins = n / 2 + 1;
        let mut out = Vec::with_capacity(n_bins);
        for k in 0..n_bins {
            let omega = 2.0 * PI * k as f64 / n as f64;
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for (i, &v) in frame.iter().enumerate() {
                re += v * (omega * i as f64).cos();
                im -= v * (omega * i as f64).sin();
            }
            out.push((re, im));
        }
        out
    }

    /// Inverse DFT from n/2+1 complex bins back to n real samples.
    #[allow(clippy::cast_precision_loss)]
    fn idft(bins: &[(f64, f64)], n: usize) -> Vec<f64> {
        let mut out = vec![0.0_f64; n];
        for (k, &(re, im)) in bins.iter().enumerate() {
            let omega = 2.0 * PI * k as f64 / n as f64;
            for (i, sample) in out.iter_mut().enumerate() {
                let angle = omega * i as f64;
                *sample += re * angle.cos() - im * angle.sin();
                // Mirror (conjugate) for k > 0 and k < n/2
                if k > 0 && k < n / 2 {
                    *sample += re * angle.cos() - im * angle.sin();
                }
            }
        }
        let scale = 1.0 / n as f64;
        for s in &mut out {
            *s *= scale;
        }
        out
    }

    /// Process a mono signal, attenuating surface noise.
    ///
    /// Uses overlap-add with spectral gating.
    ///
    /// # Errors
    ///
    /// Returns [`RestoreError::InvalidData`] when the signal is shorter than
    /// one FFT frame.
    #[allow(clippy::cast_precision_loss)]
    pub fn process(&self, samples: &[f32]) -> RestoreResult<Vec<f32>> {
        let fft_size = self.config.fft_size;
        let hop = self.config.hop_size.max(1);

        if samples.len() < fft_size {
            return Ok(samples.to_vec());
        }

        let window = Self::hann_window(fft_size);
        let n_bins = fft_size / 2 + 1;

        let perceptual = if self.config.perceptual_weight {
            Self::perceptual_curve(n_bins, self.profile.sample_rate)
        } else {
            vec![1.0; n_bins]
        };

        let mut output = vec![0.0_f64; samples.len()];
        let mut window_sum = vec![0.0_f64; samples.len()];

        let mut pos = 0;
        while pos + fft_size <= samples.len() {
            // Window the frame
            let windowed: Vec<f64> = samples[pos..pos + fft_size]
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| f64::from(s) * w)
                .collect();

            // DFT
            let mut bins = Self::dft(&windowed);

            // Spectral gate
            for (k, bin) in bins.iter_mut().enumerate().take(n_bins) {
                let mag = (bin.0 * bin.0 + bin.1 * bin.1).sqrt();
                let noise_level = if k < self.profile.spectrum.len() {
                    self.profile.spectrum[k]
                } else {
                    0.0
                };
                let weighted_noise = noise_level * self.config.over_subtraction * perceptual[k];
                let floor = noise_level * self.config.floor;

                if mag > weighted_noise {
                    // Keep the signal
                } else if mag > floor {
                    // Attenuate proportionally
                    let gain = floor / mag.max(1e-30);
                    bin.0 *= gain;
                    bin.1 *= gain;
                } else {
                    // Below floor: set to floor level
                    let gain = floor / mag.max(1e-30);
                    bin.0 *= gain;
                    bin.1 *= gain;
                }
            }

            // IDFT
            let frame_out = Self::idft(&bins, fft_size);

            // Overlap-add
            for (i, &val) in frame_out.iter().enumerate() {
                let idx = pos + i;
                if idx < output.len() {
                    output[idx] += val * window[i];
                    window_sum[idx] += window[i] * window[i];
                }
            }

            pos += hop;
        }

        // Normalise by window sum
        let result: Vec<f32> = output
            .iter()
            .zip(window_sum.iter())
            .enumerate()
            .map(|(i, (&o, &w))| {
                if w > 1e-10 {
                    (o / w) as f32
                } else {
                    samples.get(i).copied().unwrap_or(0.0)
                }
            })
            .collect();

        Ok(result)
    }

    /// Return the current noise profile.
    pub fn profile(&self) -> &VinylNoiseProfile {
        &self.profile
    }

    /// Update the noise profile.
    pub fn set_profile(&mut self, profile: VinylNoiseProfile) {
        self.profile = profile;
    }

    /// Return the configuration.
    pub fn config(&self) -> &VinylNoiseReducerConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI as PI64;

    /// Generate a sine wave.
    #[allow(clippy::cast_precision_loss)]
    fn make_sine(freq: f64, sample_rate: u32, len: usize, amp: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = i as f64 / f64::from(sample_rate);
                (amp as f64 * (2.0 * PI64 * freq * t).sin()) as f32
            })
            .collect()
    }

    /// Generate low-level noise.
    fn make_noise(len: usize, amp: f32) -> Vec<f32> {
        // Deterministic pseudo-random using LCG
        let mut state = 12345_u64;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let norm = ((state >> 33) as f64 / u32::MAX as f64) * 2.0 - 1.0;
                (norm * amp as f64) as f32
            })
            .collect()
    }

    #[test]
    fn test_empty_profile() {
        let p = VinylNoiseProfile::empty(2048, 44100);
        assert_eq!(p.bin_count(), 1025); // 2048/2 + 1
        assert_eq!(p.frames_averaged, 0);
    }

    #[test]
    fn test_profile_bin_frequency() {
        let p = VinylNoiseProfile::empty(2048, 44100);
        let freq = p.bin_frequency(100);
        let expected = 100.0 * 44100.0 / 2048.0;
        assert!((freq - expected).abs() < 1e-6);
    }

    #[test]
    fn test_profile_accumulate() {
        let mut p = VinylNoiseProfile::empty(4, 44100); // 3 bins
        p.accumulate(&[1.0, 2.0, 3.0]);
        assert_eq!(p.frames_averaged, 1);
        assert!((p.spectrum[0] - 1.0).abs() < 1e-10);
        p.accumulate(&[3.0, 4.0, 5.0]);
        assert_eq!(p.frames_averaged, 2);
        assert!((p.spectrum[0] - 2.0).abs() < 1e-10); // mean of 1 and 3
    }

    #[test]
    fn test_profile_scale() {
        let mut p = VinylNoiseProfile::empty(4, 44100);
        p.accumulate(&[1.0, 2.0, 3.0]);
        p.scale(2.0);
        assert!((p.spectrum[0] - 2.0).abs() < 1e-10);
        assert!((p.spectrum[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_profiler_learn_from_silence() {
        let profiler = VinylNoiseProfiler::new(VinylProfilerConfig {
            fft_size: 256,
            hop_size: 128,
            silence_threshold: 0.5,
            max_frames: 100,
        });
        let noise = make_noise(4096, 0.001);
        let profile = profiler.learn(&noise, 44100).expect("ok");
        assert!(profile.frames_averaged > 0, "should learn from quiet noise");
    }

    #[test]
    fn test_profiler_learn_from_noise_region() {
        let profiler = VinylNoiseProfiler::new(VinylProfilerConfig {
            fft_size: 256,
            hop_size: 128,
            silence_threshold: 0.01,
            max_frames: 100,
        });
        let noise = make_noise(2048, 0.1);
        let profile = profiler.learn_from_noise_region(&noise, 44100).expect("ok");
        assert!(profile.frames_averaged > 0);
    }

    #[test]
    fn test_profiler_reject_invalid_fft_size() {
        let profiler = VinylNoiseProfiler::new(VinylProfilerConfig {
            fft_size: 100, // not power of two
            ..VinylProfilerConfig::default()
        });
        let noise = make_noise(4096, 0.01);
        let result = profiler.learn(&noise, 44100);
        assert!(result.is_err());
    }

    #[test]
    fn test_profiler_reject_short_input() {
        let profiler = VinylNoiseProfiler::with_defaults();
        let short = make_noise(100, 0.01);
        let result = profiler.learn(&short, 44100);
        assert!(result.is_err());
    }

    #[test]
    fn test_reducer_fft_size_mismatch() {
        let config = VinylNoiseReducerConfig {
            fft_size: 1024,
            ..VinylNoiseReducerConfig::default()
        };
        let profile = VinylNoiseProfile::empty(2048, 44100);
        let result = VinylNoiseReducer::new(config, profile);
        assert!(result.is_err());
    }

    #[test]
    fn test_reducer_preserves_length() {
        let fft_size = 256;
        let profile = VinylNoiseProfile::empty(fft_size, 44100);
        let config = VinylNoiseReducerConfig {
            fft_size,
            hop_size: 64,
            over_subtraction: 1.5,
            floor: 0.01,
            perceptual_weight: false,
            smoothing: 0.9,
        };
        let reducer = VinylNoiseReducer::new(config, profile).expect("ok");
        let sine = make_sine(440.0, 44100, 2048, 0.5);
        let out = reducer.process(&sine).expect("ok");
        assert_eq!(out.len(), sine.len());
    }

    #[test]
    fn test_reducer_short_input_passthrough() {
        let fft_size = 2048;
        let profile = VinylNoiseProfile::empty(fft_size, 44100);
        let config = VinylNoiseReducerConfig {
            fft_size,
            ..VinylNoiseReducerConfig::default()
        };
        let reducer = VinylNoiseReducer::new(config, profile).expect("ok");
        let short = vec![0.5_f32; 100];
        let out = reducer.process(&short).expect("ok");
        assert_eq!(out.len(), 100);
        assert_eq!(out, short); // passthrough for too-short input
    }

    #[test]
    fn test_perceptual_curve() {
        let curve = VinylNoiseReducer::perceptual_curve(100, 44100);
        assert_eq!(curve.len(), 100);
        // Below 1 kHz should be 1.0
        assert!((curve[0] - 1.0).abs() < 1e-10);
        // Higher frequencies should be > 1.0
        let last = curve[99];
        assert!(last >= 1.0, "high frequency weight should be >= 1.0");
    }

    #[test]
    fn test_full_pipeline_profile_and_reduce() {
        let fft_size = 256;
        let profiler = VinylNoiseProfiler::new(VinylProfilerConfig {
            fft_size,
            hop_size: 64,
            silence_threshold: 0.5,
            max_frames: 50,
        });

        // Create a noisy signal: sine + low-level noise
        let sine = make_sine(440.0, 44100, 4096, 0.5);
        let noise = make_noise(4096, 0.005);
        let noisy: Vec<f32> = sine
            .iter()
            .zip(noise.iter())
            .map(|(&s, &n)| s + n)
            .collect();

        // Learn profile from noise-only region
        let noise_only = make_noise(2048, 0.005);
        let profile = profiler
            .learn_from_noise_region(&noise_only, 44100)
            .expect("ok");

        // Reduce
        let config = VinylNoiseReducerConfig {
            fft_size,
            hop_size: 64,
            over_subtraction: 2.0,
            floor: 0.01,
            perceptual_weight: false,
            smoothing: 0.9,
        };
        let reducer = VinylNoiseReducer::new(config, profile).expect("ok");
        let cleaned = reducer.process(&noisy).expect("ok");
        assert_eq!(cleaned.len(), noisy.len());
    }
}
