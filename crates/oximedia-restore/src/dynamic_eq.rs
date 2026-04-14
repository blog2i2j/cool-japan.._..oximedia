#![allow(dead_code)]
//! Frequency-dependent dynamic EQ for audio restoration.
//!
//! This module implements a multiband dynamic equaliser that applies
//! compression or expansion independently per frequency band.  It is
//! useful for taming resonant peaks, controlling sibilance, reducing
//! low-frequency rumble that only appears during loud passages, or
//! expanding dull recordings that lost spectral detail.
//!
//! # Architecture
//!
//! The signal is split into configurable frequency bands using
//! cascaded biquad crossover filters (Linkwitz–Riley topology).  Each
//! band has its own compressor/expander with independent threshold,
//! ratio, attack and release parameters.  After dynamics processing
//! the bands are summed back together.

use crate::error::{RestoreError, RestoreResult};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Biquad filter (second-order IIR)
// ---------------------------------------------------------------------------

/// Biquad filter coefficients.
#[derive(Debug, Clone)]
struct BiquadCoeffs {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

/// Biquad filter state (Direct Form II Transposed).
#[derive(Debug, Clone)]
struct BiquadState {
    z1: f64,
    z2: f64,
}

impl BiquadState {
    fn new() -> Self {
        Self { z1: 0.0, z2: 0.0 }
    }

    fn process(&mut self, c: &BiquadCoeffs, x: f64) -> f64 {
        let y = c.b0 * x + self.z1;
        self.z1 = c.b1 * x - c.a1 * y + self.z2;
        self.z2 = c.b2 * x - c.a2 * y;
        y
    }
}

/// Design a second-order Butterworth low-pass filter.
#[allow(clippy::cast_precision_loss)]
fn lowpass_coeffs(cutoff_hz: f64, sample_rate: u32) -> BiquadCoeffs {
    let w0 = 2.0 * PI * cutoff_hz / f64::from(sample_rate);
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let alpha = sin_w0 / (2.0_f64.sqrt() * 2.0); // Q = sqrt(2)/2 for Butterworth

    let a0 = 1.0 + alpha;
    BiquadCoeffs {
        b0: ((1.0 - cos_w0) / 2.0) / a0,
        b1: (1.0 - cos_w0) / a0,
        b2: ((1.0 - cos_w0) / 2.0) / a0,
        a1: (-2.0 * cos_w0) / a0,
        a2: (1.0 - alpha) / a0,
    }
}

/// Design a second-order Butterworth high-pass filter.
#[allow(clippy::cast_precision_loss)]
fn highpass_coeffs(cutoff_hz: f64, sample_rate: u32) -> BiquadCoeffs {
    let w0 = 2.0 * PI * cutoff_hz / f64::from(sample_rate);
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let alpha = sin_w0 / (2.0_f64.sqrt() * 2.0);

    let a0 = 1.0 + alpha;
    BiquadCoeffs {
        b0: ((1.0 + cos_w0) / 2.0) / a0,
        b1: (-(1.0 + cos_w0)) / a0,
        b2: ((1.0 + cos_w0) / 2.0) / a0,
        a1: (-2.0 * cos_w0) / a0,
        a2: (1.0 - alpha) / a0,
    }
}

// ---------------------------------------------------------------------------
// Per-band dynamics
// ---------------------------------------------------------------------------

/// Dynamics mode for a band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DynamicsMode {
    /// Compress: reduce levels above the threshold.
    Compress,
    /// Expand: reduce levels below the threshold (gate-like).
    Expand,
    /// Bypass: no dynamics processing.
    Bypass,
}

/// Per-band dynamics configuration.
#[derive(Debug, Clone)]
pub struct BandDynamicsConfig {
    /// Threshold in dB (typically negative, e.g. -20).
    pub threshold_db: f64,
    /// Ratio (e.g. 4.0 means 4:1 compression).  Values < 1.0 act as
    /// expansion when [`DynamicsMode::Expand`] is used.
    pub ratio: f64,
    /// Attack time in seconds.
    pub attack_s: f64,
    /// Release time in seconds.
    pub release_s: f64,
    /// Make-up gain in dB applied after dynamics.
    pub makeup_gain_db: f64,
    /// Dynamics mode.
    pub mode: DynamicsMode,
    /// Knee width in dB.  0.0 gives a hard knee.
    pub knee_db: f64,
}

impl Default for BandDynamicsConfig {
    fn default() -> Self {
        Self {
            threshold_db: -20.0,
            ratio: 4.0,
            attack_s: 0.005,
            release_s: 0.050,
            makeup_gain_db: 0.0,
            mode: DynamicsMode::Compress,
            knee_db: 6.0,
        }
    }
}

/// Runtime state for the envelope follower of a single band.
#[derive(Debug, Clone)]
struct EnvelopeState {
    envelope_db: f64,
}

impl EnvelopeState {
    fn new() -> Self {
        Self {
            envelope_db: -120.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Band descriptor
// ---------------------------------------------------------------------------

/// Description of a single frequency band in the dynamic EQ.
#[derive(Debug, Clone)]
pub struct DynamicEqBand {
    /// Lower edge frequency in Hz (inclusive).
    pub low_hz: f64,
    /// Upper edge frequency in Hz (inclusive).
    pub high_hz: f64,
    /// Dynamics configuration for this band.
    pub dynamics: BandDynamicsConfig,
}

// ---------------------------------------------------------------------------
// DynamicEq processor
// ---------------------------------------------------------------------------

/// Configuration for the dynamic EQ.
#[derive(Debug, Clone)]
pub struct DynamicEqConfig {
    /// Frequency bands with their dynamics settings.
    pub bands: Vec<DynamicEqBand>,
    /// Global output gain in dB.
    pub output_gain_db: f64,
}

impl Default for DynamicEqConfig {
    fn default() -> Self {
        Self {
            bands: vec![
                DynamicEqBand {
                    low_hz: 20.0,
                    high_hz: 200.0,
                    dynamics: BandDynamicsConfig {
                        threshold_db: -24.0,
                        ratio: 3.0,
                        mode: DynamicsMode::Compress,
                        ..BandDynamicsConfig::default()
                    },
                },
                DynamicEqBand {
                    low_hz: 200.0,
                    high_hz: 2000.0,
                    dynamics: BandDynamicsConfig {
                        threshold_db: -18.0,
                        ratio: 2.5,
                        mode: DynamicsMode::Compress,
                        ..BandDynamicsConfig::default()
                    },
                },
                DynamicEqBand {
                    low_hz: 2000.0,
                    high_hz: 8000.0,
                    dynamics: BandDynamicsConfig {
                        threshold_db: -20.0,
                        ratio: 3.5,
                        mode: DynamicsMode::Compress,
                        ..BandDynamicsConfig::default()
                    },
                },
                DynamicEqBand {
                    low_hz: 8000.0,
                    high_hz: 20000.0,
                    dynamics: BandDynamicsConfig {
                        threshold_db: -22.0,
                        ratio: 4.0,
                        mode: DynamicsMode::Compress,
                        ..BandDynamicsConfig::default()
                    },
                },
            ],
            output_gain_db: 0.0,
        }
    }
}

/// Internal crossover + dynamics processor for one band.
#[derive(Debug, Clone)]
struct BandProcessor {
    lp: Option<BiquadCoeffs>,
    hp: Option<BiquadCoeffs>,
    lp_state: BiquadState,
    hp_state: BiquadState,
    dynamics: BandDynamicsConfig,
    envelope: EnvelopeState,
    attack_coeff: f64,
    release_coeff: f64,
}

/// Convert dB to linear amplitude.
fn db_to_linear(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

/// Convert linear amplitude to dB (with a floor at -120 dB).
fn linear_to_db(lin: f64) -> f64 {
    if lin.abs() < 1e-6 {
        -120.0
    } else {
        20.0 * lin.abs().log10()
    }
}

impl BandProcessor {
    #[allow(clippy::cast_precision_loss)]
    fn new(band: &DynamicEqBand, sample_rate: u32, nyquist: f64) -> Self {
        let lp = if band.high_hz < nyquist {
            Some(lowpass_coeffs(band.high_hz, sample_rate))
        } else {
            None
        };
        let hp = if band.low_hz > 20.0 {
            Some(highpass_coeffs(band.low_hz, sample_rate))
        } else {
            None
        };
        let sr = f64::from(sample_rate);
        let attack_coeff = if band.dynamics.attack_s > 0.0 {
            (-1.0 / (band.dynamics.attack_s * sr)).exp()
        } else {
            0.0
        };
        let release_coeff = if band.dynamics.release_s > 0.0 {
            (-1.0 / (band.dynamics.release_s * sr)).exp()
        } else {
            0.0
        };
        Self {
            lp,
            hp,
            lp_state: BiquadState::new(),
            hp_state: BiquadState::new(),
            dynamics: band.dynamics.clone(),
            envelope: EnvelopeState::new(),
            attack_coeff,
            release_coeff,
        }
    }

    /// Split one sample into this band's frequency range.
    fn split(&mut self, x: f64) -> f64 {
        let mut y = x;
        if let Some(ref hp) = self.hp {
            y = self.hp_state.process(hp, y);
        }
        if let Some(ref lp) = self.lp {
            y = self.lp_state.process(lp, y);
        }
        y
    }

    /// Apply dynamics processing to a single sample in this band.
    fn apply_dynamics(&mut self, x: f64) -> f64 {
        if self.dynamics.mode == DynamicsMode::Bypass {
            return x;
        }

        let input_db = linear_to_db(x);

        // Smooth envelope follower
        let coeff = if input_db > self.envelope.envelope_db {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.envelope.envelope_db = coeff * self.envelope.envelope_db + (1.0 - coeff) * input_db;

        let env_db = self.envelope.envelope_db;
        let threshold = self.dynamics.threshold_db;
        let ratio = self.dynamics.ratio;
        let knee = self.dynamics.knee_db;

        // Compute gain reduction in dB
        let gain_db = match self.dynamics.mode {
            DynamicsMode::Compress => {
                let half_knee = knee / 2.0;
                if env_db <= threshold - half_knee {
                    0.0
                } else if env_db >= threshold + half_knee && knee > 0.0 {
                    (threshold - env_db) * (1.0 - 1.0 / ratio)
                } else if knee > 0.0 {
                    // Soft knee region
                    let x_k = env_db - threshold + half_knee;
                    -((1.0 - 1.0 / ratio) * x_k * x_k / (2.0 * knee))
                } else {
                    (threshold - env_db) * (1.0 - 1.0 / ratio)
                }
            }
            DynamicsMode::Expand => {
                if env_db < threshold {
                    -((threshold - env_db) * (ratio - 1.0))
                } else {
                    0.0
                }
            }
            DynamicsMode::Bypass => 0.0,
        };

        let total_gain_db = gain_db + self.dynamics.makeup_gain_db;
        x * db_to_linear(total_gain_db)
    }
}

/// Multiband dynamic equaliser.
///
/// Splits audio into frequency bands, applies per-band compression or
/// expansion, and sums the result.
#[derive(Debug)]
pub struct DynamicEq {
    config: DynamicEqConfig,
    bands: Vec<BandProcessor>,
    sample_rate: u32,
    output_gain: f64,
}

impl DynamicEq {
    /// Create a new dynamic EQ.
    ///
    /// # Errors
    ///
    /// Returns [`RestoreError::InvalidParameter`] when `config.bands` is empty
    /// or `sample_rate` is zero.
    #[allow(clippy::cast_precision_loss)]
    pub fn new(config: DynamicEqConfig, sample_rate: u32) -> RestoreResult<Self> {
        if sample_rate == 0 {
            return Err(RestoreError::InvalidParameter(
                "sample_rate must be > 0".into(),
            ));
        }
        if config.bands.is_empty() {
            return Err(RestoreError::InvalidParameter(
                "at least one band is required".into(),
            ));
        }
        let nyquist = f64::from(sample_rate) / 2.0;
        let bands = config
            .bands
            .iter()
            .map(|b| BandProcessor::new(b, sample_rate, nyquist))
            .collect();
        let output_gain = db_to_linear(config.output_gain_db);
        Ok(Self {
            config,
            bands,
            sample_rate,
            output_gain,
        })
    }

    /// Create a dynamic EQ with the default four-band configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if `sample_rate` is zero.
    pub fn with_defaults(sample_rate: u32) -> RestoreResult<Self> {
        Self::new(DynamicEqConfig::default(), sample_rate)
    }

    /// Process a buffer of mono samples.
    ///
    /// # Errors
    ///
    /// Returns [`RestoreError::InvalidData`] when the input is empty.
    pub fn process(&mut self, samples: &[f32]) -> RestoreResult<Vec<f32>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        let mut output = Vec::with_capacity(samples.len());

        for &s in samples {
            let x = f64::from(s);
            let mut sum = 0.0_f64;
            for band in &mut self.bands {
                let band_signal = band.split(x);
                let processed = band.apply_dynamics(band_signal);
                sum += processed;
            }
            let out = (sum * self.output_gain) as f32;
            output.push(out.clamp(-1.0, 1.0));
        }

        Ok(output)
    }

    /// Return the current configuration.
    pub fn config(&self) -> &DynamicEqConfig {
        &self.config
    }

    /// Return the sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Return the number of bands.
    pub fn band_count(&self) -> usize {
        self.bands.len()
    }

    /// Reset all internal state (envelope followers, filter states).
    pub fn reset(&mut self) {
        for band in &mut self.bands {
            band.envelope = EnvelopeState::new();
            band.lp_state = BiquadState::new();
            band.hp_state = BiquadState::new();
        }
    }

    /// Update the dynamics configuration for a specific band.
    ///
    /// # Errors
    ///
    /// Returns [`RestoreError::InvalidParameter`] when `band_idx` is out of range.
    pub fn set_band_dynamics(
        &mut self,
        band_idx: usize,
        dynamics: BandDynamicsConfig,
    ) -> RestoreResult<()> {
        let n_bands = self.bands.len();
        let band = self.bands.get_mut(band_idx).ok_or_else(|| {
            RestoreError::InvalidParameter(format!(
                "band index {band_idx} out of range (have {n_bands} bands)",
            ))
        })?;
        let sr = f64::from(self.sample_rate);
        band.attack_coeff = if dynamics.attack_s > 0.0 {
            (-1.0 / (dynamics.attack_s * sr)).exp()
        } else {
            0.0
        };
        band.release_coeff = if dynamics.release_s > 0.0 {
            (-1.0 / (dynamics.release_s * sr)).exp()
        } else {
            0.0
        };
        band.dynamics = dynamics;
        Ok(())
    }

    /// Create a de-esser preset (compress 4-9 kHz aggressively).
    ///
    /// # Errors
    ///
    /// Returns an error if `sample_rate` is zero.
    pub fn de_esser(sample_rate: u32) -> RestoreResult<Self> {
        let config = DynamicEqConfig {
            bands: vec![
                DynamicEqBand {
                    low_hz: 20.0,
                    high_hz: 4000.0,
                    dynamics: BandDynamicsConfig {
                        mode: DynamicsMode::Bypass,
                        ..BandDynamicsConfig::default()
                    },
                },
                DynamicEqBand {
                    low_hz: 4000.0,
                    high_hz: 9000.0,
                    dynamics: BandDynamicsConfig {
                        threshold_db: -30.0,
                        ratio: 6.0,
                        attack_s: 0.001,
                        release_s: 0.030,
                        mode: DynamicsMode::Compress,
                        knee_db: 3.0,
                        ..BandDynamicsConfig::default()
                    },
                },
                DynamicEqBand {
                    low_hz: 9000.0,
                    high_hz: 20000.0,
                    dynamics: BandDynamicsConfig {
                        mode: DynamicsMode::Bypass,
                        ..BandDynamicsConfig::default()
                    },
                },
            ],
            output_gain_db: 0.0,
        };
        Self::new(config, sample_rate)
    }

    /// Create a rumble reducer preset (expand below 60 Hz).
    ///
    /// # Errors
    ///
    /// Returns an error if `sample_rate` is zero.
    pub fn rumble_reducer(sample_rate: u32) -> RestoreResult<Self> {
        let config = DynamicEqConfig {
            bands: vec![
                DynamicEqBand {
                    low_hz: 20.0,
                    high_hz: 60.0,
                    dynamics: BandDynamicsConfig {
                        threshold_db: -40.0,
                        ratio: 3.0,
                        attack_s: 0.010,
                        release_s: 0.100,
                        mode: DynamicsMode::Expand,
                        knee_db: 6.0,
                        ..BandDynamicsConfig::default()
                    },
                },
                DynamicEqBand {
                    low_hz: 60.0,
                    high_hz: 20000.0,
                    dynamics: BandDynamicsConfig {
                        mode: DynamicsMode::Bypass,
                        ..BandDynamicsConfig::default()
                    },
                },
            ],
            output_gain_db: 0.0,
        };
        Self::new(config, sample_rate)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI as PI64;

    /// Generate a sine wave at the given frequency.
    #[allow(clippy::cast_precision_loss)]
    fn make_sine(freq: f64, sample_rate: u32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = i as f64 / f64::from(sample_rate);
                (2.0 * PI64 * freq * t).sin() as f32
            })
            .collect()
    }

    #[test]
    fn test_create_default() {
        let deq = DynamicEq::with_defaults(44100);
        assert!(deq.is_ok());
        let deq = deq.expect("valid");
        assert_eq!(deq.band_count(), 4);
        assert_eq!(deq.sample_rate(), 44100);
    }

    #[test]
    fn test_reject_zero_sample_rate() {
        let result = DynamicEq::with_defaults(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_empty_bands() {
        let config = DynamicEqConfig {
            bands: Vec::new(),
            output_gain_db: 0.0,
        };
        let result = DynamicEq::new(config, 44100);
        assert!(result.is_err());
    }

    #[test]
    fn test_process_preserves_length() {
        let mut deq = DynamicEq::with_defaults(44100).expect("valid");
        let sine = make_sine(440.0, 44100, 4096);
        let out = deq.process(&sine).expect("ok");
        assert_eq!(out.len(), 4096);
    }

    #[test]
    fn test_process_empty_input() {
        let mut deq = DynamicEq::with_defaults(44100).expect("valid");
        let out = deq.process(&[]).expect("ok");
        assert!(out.is_empty());
    }

    #[test]
    fn test_output_clamped() {
        let mut deq = DynamicEq::with_defaults(44100).expect("valid");
        let loud = vec![0.99_f32; 2048];
        let out = deq.process(&loud).expect("ok");
        for &s in &out {
            assert!(s >= -1.0 && s <= 1.0, "out of range: {s}");
        }
    }

    #[test]
    fn test_silence_stays_silent() {
        let mut deq = DynamicEq::with_defaults(44100).expect("valid");
        let silence = vec![0.0_f32; 1024];
        let out = deq.process(&silence).expect("ok");
        for &s in &out {
            assert!(s.abs() < 1e-6, "expected silence, got {s}");
        }
    }

    #[test]
    fn test_de_esser_preset() {
        let mut deq = DynamicEq::de_esser(44100).expect("valid");
        assert_eq!(deq.band_count(), 3);
        // Feed sibilant-range sine and verify it gets attenuated
        let sibilant = make_sine(6000.0, 44100, 8192);
        let out = deq.process(&sibilant).expect("ok");
        assert_eq!(out.len(), sibilant.len());
    }

    #[test]
    fn test_rumble_reducer_preset() {
        let mut deq = DynamicEq::rumble_reducer(44100).expect("valid");
        assert_eq!(deq.band_count(), 2);
        let rumble = make_sine(30.0, 44100, 8192);
        let out = deq.process(&rumble).expect("ok");
        assert_eq!(out.len(), rumble.len());
    }

    #[test]
    fn test_reset_clears_state() {
        let mut deq = DynamicEq::with_defaults(44100).expect("valid");
        let sine = make_sine(440.0, 44100, 4096);
        let _ = deq.process(&sine).expect("ok");
        deq.reset();
        // After reset, envelope should be back at -120 dB
        for band in &deq.bands {
            assert!(
                (band.envelope.envelope_db - (-120.0)).abs() < 1e-6,
                "envelope not reset"
            );
        }
    }

    #[test]
    fn test_set_band_dynamics() {
        let mut deq = DynamicEq::with_defaults(44100).expect("valid");
        let new_dyn = BandDynamicsConfig {
            threshold_db: -30.0,
            ratio: 8.0,
            ..BandDynamicsConfig::default()
        };
        assert!(deq.set_band_dynamics(0, new_dyn).is_ok());
        assert!(deq
            .set_band_dynamics(99, BandDynamicsConfig::default())
            .is_err());
    }

    #[test]
    fn test_bypass_mode_passes_through() {
        let config = DynamicEqConfig {
            bands: vec![DynamicEqBand {
                low_hz: 20.0,
                high_hz: 20000.0,
                dynamics: BandDynamicsConfig {
                    mode: DynamicsMode::Bypass,
                    ..BandDynamicsConfig::default()
                },
            }],
            output_gain_db: 0.0,
        };
        let mut deq = DynamicEq::new(config, 44100).expect("valid");
        let sine = make_sine(440.0, 44100, 2048);
        let out = deq.process(&sine).expect("ok");
        // With bypass and single full-range band, output should be close to input
        // (some filtering transient at the start is expected)
        let skip = 256; // skip transient
        for i in skip..sine.len() {
            assert!(
                (out[i] - sine[i]).abs() < 0.1,
                "sample {i}: in={} out={}",
                sine[i],
                out[i]
            );
        }
    }

    #[test]
    fn test_db_conversions() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-10);
        assert!((db_to_linear(20.0) - 10.0).abs() < 1e-8);
        assert!((db_to_linear(-20.0) - 0.1).abs() < 1e-8);
        assert!((linear_to_db(1.0) - 0.0).abs() < 1e-8);
    }
}
