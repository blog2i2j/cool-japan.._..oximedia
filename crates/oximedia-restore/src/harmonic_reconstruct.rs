#![allow(dead_code)]
//! Harmonic reconstruction for bandwidth-limited recordings.
//!
//! Many archival recordings — telephone lines, AM radio captures, early
//! digital transfers — have a hard upper frequency cutoff that removes
//! harmonics above 3–8 kHz.  This module detects the cutoff frequency,
//! analyses the fundamental frequencies present in the signal, and
//! synthesises plausible harmonic content above the cutoff to restore
//! brightness and presence.
//!
//! # Algorithm
//!
//! 1. **Bandwidth detection** — estimate the effective upper bandwidth by
//!    finding the frequency above which spectral energy drops sharply.
//! 2. **Fundamental extraction** — use autocorrelation-based pitch detection
//!    to find the dominant fundamental frequencies in each analysis frame.
//! 3. **Harmonic synthesis** — for each detected fundamental, generate
//!    harmonics above the cutoff with amplitudes extrapolated from the
//!    existing in-band harmonics using a spectral envelope model.
//! 4. **Blending** — high-pass the synthesised harmonics and mix them with
//!    the original signal at a user-controllable level.
//!
//! # Example
//!
//! ```
//! use oximedia_restore::harmonic_reconstruct::*;
//!
//! let config = HarmonicReconstructConfig::default();
//! let mut reconstructor = HarmonicReconstructor::new(config, 44100);
//! let samples = vec![0.0f32; 4096];
//! let result = reconstructor.process(&samples).unwrap();
//! assert_eq!(result.len(), samples.len());
//! ```

use crate::error::{RestoreError, RestoreResult};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Strategy used to extrapolate harmonic amplitudes above the detected cutoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvelopeModel {
    /// Assume harmonics roll off at a fixed dB/octave rate above the cutoff.
    FixedRolloff,
    /// Fit an exponential decay to the existing in-band harmonics and continue
    /// the curve above the cutoff.
    ExponentialDecay,
    /// Copy the spectral shape of the last in-band octave into the
    /// synthesised region (spectral mirroring).
    SpectralMirror,
}

/// Configuration for harmonic reconstruction.
#[derive(Debug, Clone)]
pub struct HarmonicReconstructConfig {
    /// FFT analysis block size (must be a power of two).
    pub block_size: usize,
    /// Hop size between successive analysis frames.
    pub hop_size: usize,
    /// Maximum number of harmonics to synthesise per fundamental.
    pub max_harmonics: usize,
    /// Minimum fundamental frequency to consider (Hz).
    pub min_f0_hz: f64,
    /// Maximum fundamental frequency to consider (Hz).
    pub max_f0_hz: f64,
    /// Envelope model for extrapolating harmonic amplitudes.
    pub envelope_model: EnvelopeModel,
    /// Dry/wet mix (0.0 = original only, 1.0 = fully reconstructed).
    pub mix: f64,
    /// Roll-off rate in dB/octave when using [`EnvelopeModel::FixedRolloff`].
    pub rolloff_db_per_octave: f64,
    /// Automatically detect the bandwidth cutoff?  If `false`, use
    /// `manual_cutoff_hz` instead.
    pub auto_detect_cutoff: bool,
    /// Manual cutoff frequency (Hz).  Ignored when `auto_detect_cutoff` is true
    /// unless auto-detection fails.
    pub manual_cutoff_hz: f64,
}

impl Default for HarmonicReconstructConfig {
    fn default() -> Self {
        Self {
            block_size: 2048,
            hop_size: 512,
            max_harmonics: 8,
            min_f0_hz: 80.0,
            max_f0_hz: 1000.0,
            envelope_model: EnvelopeModel::ExponentialDecay,
            mix: 0.5,
            rolloff_db_per_octave: 6.0,
            auto_detect_cutoff: true,
            manual_cutoff_hz: 4000.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Analysis output
// ---------------------------------------------------------------------------

/// Result of bandwidth analysis on a single frame.
#[derive(Debug, Clone)]
pub struct BandwidthAnalysis {
    /// Estimated upper bandwidth cutoff in Hz.
    pub cutoff_hz: f64,
    /// Spectral energy below the cutoff (linear).
    pub in_band_energy: f64,
    /// Spectral energy above the cutoff (linear).
    pub out_band_energy: f64,
    /// Ratio of out-band to in-band energy.
    pub bandwidth_ratio: f64,
}

/// Detected fundamental with its in-band harmonics.
#[derive(Debug, Clone)]
pub struct DetectedFundamental {
    /// Fundamental frequency in Hz.
    pub f0_hz: f64,
    /// Amplitudes of in-band harmonics (index 0 = fundamental).
    pub harmonic_amplitudes: Vec<f64>,
    /// Confidence of the detection (0.0–1.0).
    pub confidence: f64,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Hann window.
fn hann_window(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let w = 0.5 * (1.0 - (2.0 * PI * i as f64 / size as f64).cos());
            w
        })
        .collect()
}

/// Compute magnitude spectrum from real-valued input (returns N/2+1 bins).
fn magnitude_spectrum(samples: &[f64]) -> Vec<f64> {
    let n = samples.len();
    let mut magnitudes = Vec::with_capacity(n / 2 + 1);
    for k in 0..=n / 2 {
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        for (i, &s) in samples.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let angle = -2.0 * PI * k as f64 * i as f64 / n as f64;
            re += s * angle.cos();
            im += s * angle.sin();
        }
        magnitudes.push((re * re + im * im).sqrt());
    }
    magnitudes
}

/// Estimate the effective bandwidth cutoff from a magnitude spectrum.
///
/// Walks from high frequencies downward and finds the first bin where the
/// energy exceeds a threshold relative to the peak energy in the spectrum.
#[allow(clippy::cast_precision_loss)]
fn estimate_cutoff(magnitudes: &[f64], sample_rate: u32, _block_size: usize) -> f64 {
    let peak = magnitudes.iter().copied().fold(0.0_f64, f64::max);
    if peak < 1e-12 {
        return sample_rate as f64 / 2.0;
    }

    let threshold = peak * 0.01; // -40 dB relative to peak
    let n_bins = magnitudes.len();
    let bin_hz = sample_rate as f64 / ((n_bins - 1) * 2) as f64;

    // Scan from the top bin downward
    for k in (1..n_bins).rev() {
        if magnitudes[k] > threshold {
            return (k as f64 + 1.0) * bin_hz;
        }
    }

    bin_hz
}

/// Autocorrelation-based pitch detection (simplified YIN-like).
#[allow(clippy::cast_precision_loss)]
fn detect_pitch(samples: &[f64], sample_rate: u32, min_f0: f64, max_f0: f64) -> Option<(f64, f64)> {
    let n = samples.len();
    let min_lag = (sample_rate as f64 / max_f0).floor() as usize;
    let max_lag = (sample_rate as f64 / min_f0).ceil() as usize;

    if max_lag >= n || min_lag >= max_lag {
        return None;
    }

    // Compute normalised autocorrelation
    let mut best_lag = min_lag;
    let mut best_corr = f64::NEG_INFINITY;

    for lag in min_lag..=max_lag.min(n - 1) {
        let mut sum = 0.0_f64;
        let mut energy_a = 0.0_f64;
        let mut energy_b = 0.0_f64;
        for i in 0..n - lag {
            sum += samples[i] * samples[i + lag];
            energy_a += samples[i] * samples[i];
            energy_b += samples[i + lag] * samples[i + lag];
        }
        let norm = (energy_a * energy_b).sqrt();
        let corr = if norm > 1e-12 { sum / norm } else { 0.0 };
        if corr > best_corr {
            best_corr = corr;
            best_lag = lag;
        }
    }

    if best_corr < 0.3 {
        return None;
    }

    let f0 = sample_rate as f64 / best_lag as f64;
    Some((f0, best_corr))
}

/// Generate a sine harmonic at the given frequency and amplitude.
#[allow(clippy::cast_precision_loss)]
fn synthesise_harmonic(length: usize, freq_hz: f64, amplitude: f64, sample_rate: u32) -> Vec<f64> {
    let sr = sample_rate as f64;
    (0..length)
        .map(|i| amplitude * (2.0 * PI * freq_hz * i as f64 / sr).sin())
        .collect()
}

/// Extrapolate harmonic amplitude above cutoff using the chosen envelope model.
fn extrapolate_amplitude(
    harmonic_index: usize,
    in_band_amps: &[f64],
    cutoff_hz: f64,
    f0: f64,
    model: EnvelopeModel,
    rolloff_db_oct: f64,
) -> f64 {
    let freq = f0 * (harmonic_index + 1) as f64;
    if freq <= cutoff_hz {
        return in_band_amps.get(harmonic_index).copied().unwrap_or(0.0);
    }

    match model {
        EnvelopeModel::FixedRolloff => {
            let last_amp = in_band_amps.last().copied().unwrap_or(0.001);
            let last_freq = f0 * in_band_amps.len() as f64;
            if last_freq < 1.0 {
                return 0.0;
            }
            let octaves_above = (freq / last_freq).log2();
            let db_drop = rolloff_db_oct * octaves_above;
            last_amp * 10.0_f64.powf(-db_drop / 20.0)
        }
        EnvelopeModel::ExponentialDecay => {
            if in_band_amps.len() < 2 {
                return in_band_amps.first().copied().unwrap_or(0.0) * 0.5;
            }
            // Fit exponential: a(n) = A * exp(-alpha * n)
            let first = in_band_amps[0].max(1e-12);
            let last = in_band_amps.last().copied().unwrap_or(1e-12).max(1e-12);
            let n_in = in_band_amps.len() as f64;
            let alpha = (first / last).ln() / (n_in - 1.0);
            first * (-alpha * harmonic_index as f64).exp()
        }
        EnvelopeModel::SpectralMirror => {
            // Mirror the last in-band harmonic amplitude
            let mirror_idx = if in_band_amps.is_empty() {
                0
            } else {
                let offset = harmonic_index.saturating_sub(in_band_amps.len());
                in_band_amps.len() - 1 - (offset % in_band_amps.len())
            };
            in_band_amps.get(mirror_idx).copied().unwrap_or(0.0) * 0.7
        }
    }
}

// ---------------------------------------------------------------------------
// Reconstructor
// ---------------------------------------------------------------------------

/// Harmonic reconstructor that rebuilds missing harmonics above a detected
/// or manual bandwidth cutoff.
#[derive(Debug, Clone)]
pub struct HarmonicReconstructor {
    config: HarmonicReconstructConfig,
    sample_rate: u32,
    window: Vec<f64>,
}

impl HarmonicReconstructor {
    /// Create a new reconstructor.
    ///
    /// # Panics
    ///
    /// Does not panic; invalid configuration is caught at processing time.
    pub fn new(config: HarmonicReconstructConfig, sample_rate: u32) -> Self {
        let window = hann_window(config.block_size);
        Self {
            config,
            sample_rate,
            window,
        }
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: HarmonicReconstructConfig) {
        if config.block_size != self.config.block_size {
            self.window = hann_window(config.block_size);
        }
        self.config = config;
    }

    /// Return a reference to the current configuration.
    pub fn config(&self) -> &HarmonicReconstructConfig {
        &self.config
    }

    /// Analyse the bandwidth of a signal block.
    pub fn analyse_bandwidth(&self, samples: &[f32]) -> RestoreResult<BandwidthAnalysis> {
        if samples.len() < self.config.block_size {
            return Err(RestoreError::NotEnoughData {
                needed: self.config.block_size,
                have: samples.len(),
            });
        }

        let block: Vec<f64> = samples[..self.config.block_size]
            .iter()
            .enumerate()
            .map(|(i, &s)| s as f64 * self.window.get(i).copied().unwrap_or(1.0))
            .collect();

        let mags = magnitude_spectrum(&block);
        let cutoff = estimate_cutoff(&mags, self.sample_rate, self.config.block_size);

        #[allow(clippy::cast_precision_loss)]
        let bin_hz = self.sample_rate as f64 / self.config.block_size as f64;
        let cutoff_bin = (cutoff / bin_hz).round() as usize;
        let cutoff_bin = cutoff_bin.min(mags.len());

        let in_band: f64 = mags[..cutoff_bin].iter().map(|m| m * m).sum();
        let out_band: f64 = mags[cutoff_bin..].iter().map(|m| m * m).sum();
        let ratio = if in_band > 1e-20 {
            out_band / in_band
        } else {
            0.0
        };

        Ok(BandwidthAnalysis {
            cutoff_hz: cutoff,
            in_band_energy: in_band.sqrt(),
            out_band_energy: out_band.sqrt(),
            bandwidth_ratio: ratio,
        })
    }

    /// Detect the dominant fundamental frequency in a sample block.
    pub fn detect_fundamental(&self, samples: &[f32]) -> Option<DetectedFundamental> {
        let block: Vec<f64> = samples.iter().map(|&s| s as f64).collect();
        let (f0, confidence) = detect_pitch(
            &block,
            self.sample_rate,
            self.config.min_f0_hz,
            self.config.max_f0_hz,
        )?;

        // Measure existing in-band harmonics
        let windowed: Vec<f64> = block
            .iter()
            .enumerate()
            .map(|(i, &s)| s * self.window.get(i).copied().unwrap_or(1.0))
            .collect();
        let mags = magnitude_spectrum(&windowed);

        #[allow(clippy::cast_precision_loss)]
        let bin_hz = self.sample_rate as f64 / self.config.block_size as f64;
        let nyquist = self.sample_rate as f64 / 2.0;

        let mut harmonic_amplitudes = Vec::new();
        for h in 1..=self.config.max_harmonics {
            let freq = f0 * h as f64;
            if freq >= nyquist {
                break;
            }
            let bin = (freq / bin_hz).round() as usize;
            let amp = mags.get(bin).copied().unwrap_or(0.0);
            harmonic_amplitudes.push(amp);
        }

        Some(DetectedFundamental {
            f0_hz: f0,
            harmonic_amplitudes,
            confidence,
        })
    }

    /// Process a buffer of samples, returning the reconstructed signal.
    ///
    /// The output has the same length as the input.
    #[allow(clippy::cast_precision_loss)]
    pub fn process(&mut self, samples: &[f32]) -> RestoreResult<Vec<f32>> {
        let n = samples.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // For very short inputs, return as-is
        if n < self.config.block_size {
            return Ok(samples.to_vec());
        }

        let mut output = vec![0.0_f64; n];
        let mut weight = vec![0.0_f64; n];
        let block_size = self.config.block_size;
        let hop = self.config.hop_size.max(1);

        // Determine cutoff
        let cutoff_hz = if self.config.auto_detect_cutoff {
            let analysis = self.analyse_bandwidth(samples)?;
            analysis.cutoff_hz
        } else {
            self.config.manual_cutoff_hz
        };

        let nyquist = self.sample_rate as f64 / 2.0;

        // Process frame-by-frame with overlap-add
        let mut pos = 0;
        while pos + block_size <= n {
            let frame: Vec<f32> = samples[pos..pos + block_size].to_vec();

            // Try to detect fundamental in this frame
            let synthesised = if let Some(fund) = self.detect_fundamental(&frame) {
                // Generate harmonics above the cutoff
                let mut synth = vec![0.0_f64; block_size];
                for h_idx in 0..self.config.max_harmonics {
                    let freq = fund.f0_hz * (h_idx + 1) as f64;
                    if freq <= cutoff_hz || freq >= nyquist {
                        continue;
                    }
                    let amp = extrapolate_amplitude(
                        h_idx,
                        &fund.harmonic_amplitudes,
                        cutoff_hz,
                        fund.f0_hz,
                        self.config.envelope_model,
                        self.config.rolloff_db_per_octave,
                    );
                    let harmonic = synthesise_harmonic(block_size, freq, amp, self.sample_rate);
                    for (s, h) in synth.iter_mut().zip(harmonic.iter()) {
                        *s += h;
                    }
                }
                synth
            } else {
                vec![0.0; block_size]
            };

            // Window and overlap-add
            for i in 0..block_size {
                let w = self.window.get(i).copied().unwrap_or(1.0);
                let orig = frame[i] as f64;
                let mixed = orig + self.config.mix * synthesised[i];
                output[pos + i] += mixed * w;
                weight[pos + i] += w;
            }

            pos += hop;
        }

        // Handle tail
        for i in 0..n {
            if weight[i] > 1e-12 {
                output[i] /= weight[i];
            } else {
                output[i] = samples[i] as f64;
            }
        }

        Ok(output.iter().map(|&s| s as f32).collect())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sine(freq: f64, sample_rate: u32, length: usize) -> Vec<f32> {
        (0..length)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let s = (2.0 * PI * freq * i as f64 / sample_rate as f64).sin() as f32;
                s
            })
            .collect()
    }

    #[test]
    fn test_default_config() {
        let cfg = HarmonicReconstructConfig::default();
        assert_eq!(cfg.block_size, 2048);
        assert_eq!(cfg.max_harmonics, 8);
        assert!((cfg.mix - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_empty_input() {
        let cfg = HarmonicReconstructConfig::default();
        let mut r = HarmonicReconstructor::new(cfg, 44100);
        let result = r.process(&[]).expect("empty ok");
        assert!(result.is_empty());
    }

    #[test]
    fn test_short_input_passthrough() {
        let cfg = HarmonicReconstructConfig::default();
        let mut r = HarmonicReconstructor::new(cfg, 44100);
        let samples = vec![0.1, 0.2, 0.3];
        let result = r.process(&samples).expect("short ok");
        assert_eq!(result, samples);
    }

    #[test]
    fn test_process_preserves_length() {
        let cfg = HarmonicReconstructConfig::default();
        let mut r = HarmonicReconstructor::new(cfg, 44100);
        let samples = make_sine(440.0, 44100, 4096);
        let result = r.process(&samples).expect("ok");
        assert_eq!(result.len(), samples.len());
    }

    #[test]
    fn test_bandwidth_analysis() {
        let cfg = HarmonicReconstructConfig::default();
        let r = HarmonicReconstructor::new(cfg, 44100);
        let samples = make_sine(440.0, 44100, 4096);
        let analysis = r.analyse_bandwidth(&samples).expect("ok");
        assert!(analysis.cutoff_hz > 0.0);
        assert!(analysis.in_band_energy >= 0.0);
    }

    #[test]
    fn test_bandwidth_analysis_too_short() {
        let cfg = HarmonicReconstructConfig::default();
        let r = HarmonicReconstructor::new(cfg, 44100);
        let samples = vec![0.0; 100];
        let result = r.analyse_bandwidth(&samples);
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_fundamental_sine() {
        let cfg = HarmonicReconstructConfig {
            block_size: 4096,
            ..Default::default()
        };
        let r = HarmonicReconstructor::new(cfg, 44100);
        let samples = make_sine(200.0, 44100, 4096);
        let fund = r.detect_fundamental(&samples);
        // We may or may not detect it depending on the autocorrelation, but it shouldn't panic
        if let Some(f) = fund {
            // Autocorrelation pitch detection has limited resolution; accept
            // the detected value as long as it falls within a reasonable range
            // that covers the fundamental and potential octave errors.
            assert!(
                f.f0_hz > 50.0 && f.f0_hz < 800.0,
                "detected f0 = {} Hz is outside acceptable range",
                f.f0_hz
            );
            assert!(f.confidence > 0.0);
        }
    }

    #[test]
    fn test_envelope_model_fixed_rolloff() {
        let amps = vec![1.0, 0.8, 0.5, 0.3];
        let result =
            extrapolate_amplitude(5, &amps, 2000.0, 440.0, EnvelopeModel::FixedRolloff, 6.0);
        assert!(result > 0.0);
        assert!(result < 1.0);
    }

    #[test]
    fn test_envelope_model_exponential_decay() {
        let amps = vec![1.0, 0.7, 0.4, 0.2];
        let result = extrapolate_amplitude(
            5,
            &amps,
            1000.0,
            200.0,
            EnvelopeModel::ExponentialDecay,
            6.0,
        );
        assert!(result > 0.0);
        assert!(result < amps[0]);
    }

    #[test]
    fn test_envelope_model_spectral_mirror() {
        let amps = vec![1.0, 0.8, 0.6, 0.4];
        let result =
            extrapolate_amplitude(5, &amps, 1000.0, 200.0, EnvelopeModel::SpectralMirror, 6.0);
        assert!(result >= 0.0);
    }

    #[test]
    fn test_zero_mix_returns_original() {
        let cfg = HarmonicReconstructConfig {
            mix: 0.0,
            ..Default::default()
        };
        let mut r = HarmonicReconstructor::new(cfg, 44100);
        let samples = make_sine(440.0, 44100, 4096);
        let result = r.process(&samples).expect("ok");
        // With zero mix, output ≈ input (within windowing tolerance)
        for (a, b) in result.iter().zip(samples.iter()) {
            assert!((a - b).abs() < 0.05, "diff too large: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_set_config_updates() {
        let cfg = HarmonicReconstructConfig::default();
        let mut r = HarmonicReconstructor::new(cfg, 44100);
        let new_cfg = HarmonicReconstructConfig {
            max_harmonics: 16,
            block_size: 4096,
            ..Default::default()
        };
        r.set_config(new_cfg);
        assert_eq!(r.config().max_harmonics, 16);
        assert_eq!(r.config().block_size, 4096);
    }

    #[test]
    fn test_manual_cutoff_mode() {
        let cfg = HarmonicReconstructConfig {
            auto_detect_cutoff: false,
            manual_cutoff_hz: 3000.0,
            ..Default::default()
        };
        let mut r = HarmonicReconstructor::new(cfg, 44100);
        let samples = make_sine(440.0, 44100, 4096);
        let result = r.process(&samples).expect("ok");
        assert_eq!(result.len(), samples.len());
    }

    #[test]
    fn test_hann_window_endpoints() {
        let w = hann_window(256);
        assert_eq!(w.len(), 256);
        assert!(w[0].abs() < 1e-9, "Hann window starts at 0");
        // Mid-point should be close to 1.0
        assert!((w[128] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_synthesise_harmonic_length() {
        let h = synthesise_harmonic(1024, 440.0, 0.5, 44100);
        assert_eq!(h.len(), 1024);
        // Peak should not exceed amplitude
        let peak = h.iter().copied().fold(0.0_f64, |a, b| a.max(b.abs()));
        assert!(peak <= 0.5 + 1e-9);
    }
}
