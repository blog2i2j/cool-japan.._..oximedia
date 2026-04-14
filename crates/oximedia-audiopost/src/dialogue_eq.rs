#![allow(dead_code)]
//! Dialogue EQ for broadcast and post-production.
//!
//! Provides a parametric EQ chain tuned for spoken-word content:
//! - **High-pass filter (HPF)**: removes low-frequency rumble (default 80 Hz, 12 dB/oct Butterworth)
//! - **Low-mid cut ("de-muddying")**: attenuates boxy resonances in the 200–500 Hz region
//! - **Presence boost**: bell EQ centred at 2–5 kHz for intelligibility and clarity
//! - **Air shelf**: high-shelf above 10 kHz adds sparkle for broadcast delivery
//!
//! All filters are implemented as biquad second-order sections (SOS) and cascaded at block
//! processing time using the Direct Form II transposed topology, which is numerically stable.

use crate::error::{AudioPostError, AudioPostResult};

// ---------------------------------------------------------------------------
// Biquad helper
// ---------------------------------------------------------------------------

/// Transposed Direct Form II biquad filter state.
#[derive(Debug, Clone, Default)]
struct Biquad {
    /// Feed-forward coefficients [b0, b1, b2]
    b: [f64; 3],
    /// Feedback coefficients [a1, a2] (a0 normalised to 1)
    a: [f64; 2],
    /// Delay-line state
    s: [f64; 2],
}

impl Biquad {
    /// Process a single sample.
    #[inline]
    fn process(&mut self, x: f64) -> f64 {
        let y = self.b[0] * x + self.s[0];
        self.s[0] = self.b[1] * x - self.a[0] * y + self.s[1];
        self.s[1] = self.b[2] * x - self.a[1] * y;
        y
    }

    /// Reset delay-line state (e.g. between unrelated segments).
    fn reset(&mut self) {
        self.s = [0.0; 2];
    }
}

// ---------------------------------------------------------------------------
// Coefficient design helpers
// ---------------------------------------------------------------------------

/// Design a 2nd-order Butterworth high-pass filter.
///
/// # Arguments
/// * `fc` – cutoff frequency in Hz
/// * `fs` – sample rate in Hz
fn hpf_butterworth(fc: f64, fs: f64) -> Biquad {
    // Bilinear transform of 2nd-order Butterworth HPF
    let omega = std::f64::consts::PI * fc / fs;
    let cos_w = omega.cos();
    let sin_w = omega.sin();
    let alpha = sin_w / std::f64::consts::SQRT_2; // Q = 1/sqrt(2) = Butterworth

    let b0 = (1.0 + cos_w) / 2.0;
    let b1 = -(1.0 + cos_w);
    let b2 = (1.0 + cos_w) / 2.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w;
    let a2 = 1.0 - alpha;

    Biquad {
        b: [b0 / a0, b1 / a0, b2 / a0],
        a: [a1 / a0, a2 / a0],
        s: [0.0; 2],
    }
}

/// Design a 2nd-order peaking (bell) EQ filter.
///
/// # Arguments
/// * `fc` – centre frequency in Hz
/// * `gain_db` – boost/cut in dB
/// * `q` – quality factor (bandwidth)
/// * `fs` – sample rate in Hz
fn peak_eq(fc: f64, gain_db: f64, q: f64, fs: f64) -> Biquad {
    let omega = 2.0 * std::f64::consts::PI * fc / fs;
    let cos_w = omega.cos();
    let sin_w = omega.sin();
    let alpha = sin_w / (2.0 * q);
    let a_gain = 10.0_f64.powf(gain_db / 40.0); // sqrt(linear gain)

    let b0 = 1.0 + alpha * a_gain;
    let b1 = -2.0 * cos_w;
    let b2 = 1.0 - alpha * a_gain;
    let a0 = 1.0 + alpha / a_gain;
    let a1 = -2.0 * cos_w;
    let a2 = 1.0 - alpha / a_gain;

    Biquad {
        b: [b0 / a0, b1 / a0, b2 / a0],
        a: [a1 / a0, a2 / a0],
        s: [0.0; 2],
    }
}

/// Design a 2nd-order high-shelf filter (Audio EQ Cookbook, Zölzer).
///
/// # Arguments
/// * `fc` – shelf knee frequency in Hz
/// * `gain_db` – shelf gain in dB (positive = boost)
/// * `q` – slope parameter (0.707 for maximally-flat)
/// * `fs` – sample rate in Hz
fn high_shelf(fc: f64, gain_db: f64, q: f64, fs: f64) -> Biquad {
    let omega = 2.0 * std::f64::consts::PI * fc / fs;
    let cos_w = omega.cos();
    let sin_w = omega.sin();
    let a_gain = 10.0_f64.powf(gain_db / 40.0);
    let alpha = sin_w / (2.0 * q) * (a_gain + 1.0 / a_gain).sqrt();

    let b0 = a_gain * ((a_gain + 1.0) + (a_gain - 1.0) * cos_w + 2.0 * a_gain.sqrt() * alpha);
    let b1 = -2.0 * a_gain * ((a_gain - 1.0) + (a_gain + 1.0) * cos_w);
    let b2 = a_gain * ((a_gain + 1.0) + (a_gain - 1.0) * cos_w - 2.0 * a_gain.sqrt() * alpha);
    let a0 = (a_gain + 1.0) - (a_gain - 1.0) * cos_w + 2.0 * a_gain.sqrt() * alpha;
    let a1 = 2.0 * ((a_gain - 1.0) - (a_gain + 1.0) * cos_w);
    let a2 = (a_gain + 1.0) - (a_gain - 1.0) * cos_w - 2.0 * a_gain.sqrt() * alpha;

    Biquad {
        b: [b0 / a0, b1 / a0, b2 / a0],
        a: [a1 / a0, a2 / a0],
        s: [0.0; 2],
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Configuration for the dialogue EQ chain.
#[derive(Debug, Clone)]
pub struct DialogueEqConfig {
    /// High-pass filter cutoff in Hz (default 80 Hz).
    pub hpf_cutoff_hz: f32,
    /// Whether the HPF is enabled.
    pub hpf_enabled: bool,

    /// Low-mid bell cut centre frequency in Hz (default 330 Hz, de-muddying).
    pub low_mid_freq_hz: f32,
    /// Low-mid gain in dB — normally negative (default -3 dB).
    pub low_mid_gain_db: f32,
    /// Low-mid Q factor (default 0.7).
    pub low_mid_q: f32,
    /// Whether the low-mid band is enabled.
    pub low_mid_enabled: bool,

    /// Presence bell centre frequency in Hz (default 3 000 Hz).
    pub presence_freq_hz: f32,
    /// Presence gain in dB (default +4 dB).
    pub presence_gain_db: f32,
    /// Presence Q factor (default 1.0).
    pub presence_q: f32,
    /// Whether the presence band is enabled.
    pub presence_enabled: bool,

    /// Air high-shelf knee in Hz (default 12 000 Hz).
    pub air_shelf_hz: f32,
    /// Air shelf gain in dB (default +2 dB).
    pub air_gain_db: f32,
    /// Whether the air shelf is enabled.
    pub air_enabled: bool,
}

impl Default for DialogueEqConfig {
    fn default() -> Self {
        Self {
            hpf_cutoff_hz: 80.0,
            hpf_enabled: true,
            low_mid_freq_hz: 330.0,
            low_mid_gain_db: -3.0,
            low_mid_q: 0.7,
            low_mid_enabled: true,
            presence_freq_hz: 3000.0,
            presence_gain_db: 4.0,
            presence_q: 1.0,
            presence_enabled: true,
            air_shelf_hz: 12000.0,
            air_gain_db: 2.0,
            air_enabled: true,
        }
    }
}

/// Frequency analysis result for a processed block.
#[derive(Debug, Clone)]
pub struct DialogueEqAnalysis {
    /// RMS level of the input signal (linear).
    pub input_rms: f32,
    /// RMS level of the output signal (linear).
    pub output_rms: f32,
    /// Effective gain change in dB.
    pub gain_change_db: f32,
}

/// Multi-band dialogue EQ processor.
///
/// Uses a cascade of biquad filters tuned for intelligibility and broadcast compliance.
/// One instance processes a single audio channel; duplicate for multi-channel audio.
#[derive(Debug)]
pub struct DialogueEq {
    sample_rate: u32,
    config: DialogueEqConfig,
    hpf: Biquad,
    low_mid: Biquad,
    presence: Biquad,
    air: Biquad,
}

impl DialogueEq {
    /// Create a new dialogue EQ with the given sample rate and default configuration.
    ///
    /// # Errors
    ///
    /// Returns [`AudioPostError::InvalidSampleRate`] if `sample_rate` is zero.
    pub fn new(sample_rate: u32) -> AudioPostResult<Self> {
        Self::with_config(sample_rate, DialogueEqConfig::default())
    }

    /// Create a new dialogue EQ with the given sample rate and custom configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if `sample_rate` is zero or any frequency / Q / gain parameter is
    /// invalid.
    pub fn with_config(sample_rate: u32, config: DialogueEqConfig) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        Self::validate_config(&config, sample_rate)?;

        let fs = sample_rate as f64;
        let hpf = hpf_butterworth(config.hpf_cutoff_hz as f64, fs);
        let low_mid = peak_eq(
            config.low_mid_freq_hz as f64,
            config.low_mid_gain_db as f64,
            config.low_mid_q as f64,
            fs,
        );
        let presence = peak_eq(
            config.presence_freq_hz as f64,
            config.presence_gain_db as f64,
            config.presence_q as f64,
            fs,
        );
        let air = high_shelf(
            config.air_shelf_hz as f64,
            config.air_gain_db as f64,
            0.707,
            fs,
        );

        Ok(Self {
            sample_rate,
            config,
            hpf,
            low_mid,
            presence,
            air,
        })
    }

    fn validate_config(cfg: &DialogueEqConfig, sample_rate: u32) -> AudioPostResult<()> {
        let nyquist = sample_rate as f32 / 2.0;
        if cfg.hpf_cutoff_hz <= 0.0 || cfg.hpf_cutoff_hz >= nyquist {
            return Err(AudioPostError::InvalidFrequency(cfg.hpf_cutoff_hz));
        }
        if cfg.low_mid_freq_hz <= 0.0 || cfg.low_mid_freq_hz >= nyquist {
            return Err(AudioPostError::InvalidFrequency(cfg.low_mid_freq_hz));
        }
        if cfg.low_mid_q <= 0.0 {
            return Err(AudioPostError::InvalidQ(cfg.low_mid_q));
        }
        if cfg.presence_freq_hz <= 0.0 || cfg.presence_freq_hz >= nyquist {
            return Err(AudioPostError::InvalidFrequency(cfg.presence_freq_hz));
        }
        if cfg.presence_q <= 0.0 {
            return Err(AudioPostError::InvalidQ(cfg.presence_q));
        }
        if cfg.air_shelf_hz <= 0.0 || cfg.air_shelf_hz >= nyquist {
            return Err(AudioPostError::InvalidFrequency(cfg.air_shelf_hz));
        }
        Ok(())
    }

    /// Return the current configuration.
    #[must_use]
    pub fn config(&self) -> &DialogueEqConfig {
        &self.config
    }

    /// Return the sample rate this instance was created with.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Update the configuration and re-compute filter coefficients.
    ///
    /// # Errors
    ///
    /// Returns an error if any parameter in `config` is invalid.
    pub fn set_config(&mut self, config: DialogueEqConfig) -> AudioPostResult<()> {
        Self::validate_config(&config, self.sample_rate)?;
        let fs = self.sample_rate as f64;
        self.hpf = hpf_butterworth(config.hpf_cutoff_hz as f64, fs);
        self.low_mid = peak_eq(
            config.low_mid_freq_hz as f64,
            config.low_mid_gain_db as f64,
            config.low_mid_q as f64,
            fs,
        );
        self.presence = peak_eq(
            config.presence_freq_hz as f64,
            config.presence_gain_db as f64,
            config.presence_q as f64,
            fs,
        );
        self.air = high_shelf(
            config.air_shelf_hz as f64,
            config.air_gain_db as f64,
            0.707,
            fs,
        );
        self.config = config;
        Ok(())
    }

    /// Reset all filter delay-line state.
    ///
    /// Call this between unrelated audio segments to avoid transients from stale state.
    pub fn reset(&mut self) {
        self.hpf.reset();
        self.low_mid.reset();
        self.presence.reset();
        self.air.reset();
    }

    /// Process a mono audio block **in-place**.
    ///
    /// Each enabled biquad stage is applied sequentially: HPF → low-mid → presence → air shelf.
    pub fn process_block(&mut self, samples: &mut [f32]) {
        for x in samples.iter_mut() {
            let mut s = *x as f64;
            if self.config.hpf_enabled {
                s = self.hpf.process(s);
            }
            if self.config.low_mid_enabled {
                s = self.low_mid.process(s);
            }
            if self.config.presence_enabled {
                s = self.presence.process(s);
            }
            if self.config.air_enabled {
                s = self.air.process(s);
            }
            *x = s as f32;
        }
    }

    /// Process a block and return analysis statistics.
    pub fn process_block_with_analysis(
        &mut self,
        samples: &mut [f32],
    ) -> DialogueEqAnalysis {
        let input_rms = rms(samples);
        self.process_block(samples);
        let output_rms = rms(samples);
        let gain_change_db = if input_rms > 1e-12 && output_rms > 1e-12 {
            20.0 * (output_rms / input_rms).log10()
        } else {
            0.0
        };
        DialogueEqAnalysis {
            input_rms,
            output_rms,
            gain_change_db,
        }
    }

    /// Convenience: process a stereo interleaved buffer.
    ///
    /// `channels` must be 2; a separate EQ instance is maintained per call via temporary
    /// per-channel splits.  For persistent per-channel state across blocks, prefer maintaining
    /// two `DialogueEq` instances.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer length is not an even multiple of 2.
    pub fn process_stereo_interleaved(
        &mut self,
        samples: &mut [f32],
        right_eq: &mut DialogueEq,
    ) -> AudioPostResult<()> {
        if samples.len() % 2 != 0 {
            return Err(AudioPostError::InvalidBufferSize(samples.len()));
        }
        for chunk in samples.chunks_exact_mut(2) {
            let mut l = chunk[0] as f64;
            let mut r = chunk[1] as f64;
            if self.config.hpf_enabled {
                l = self.hpf.process(l);
            }
            if self.config.low_mid_enabled {
                l = self.low_mid.process(l);
            }
            if self.config.presence_enabled {
                l = self.presence.process(l);
            }
            if self.config.air_enabled {
                l = self.air.process(l);
            }
            if right_eq.config.hpf_enabled {
                r = right_eq.hpf.process(r);
            }
            if right_eq.config.low_mid_enabled {
                r = right_eq.low_mid.process(r);
            }
            if right_eq.config.presence_enabled {
                r = right_eq.presence.process(r);
            }
            if right_eq.config.air_enabled {
                r = right_eq.air.process(r);
            }
            chunk[0] = l as f32;
            chunk[1] = r as f32;
        }
        Ok(())
    }
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&x| x * x).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_block(freq: f32, sample_rate: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin()
            })
            .collect()
    }

    #[test]
    fn test_new_default() {
        let eq = DialogueEq::new(48000).unwrap();
        assert_eq!(eq.sample_rate(), 48000);
        assert!(eq.config().hpf_enabled);
        assert!(eq.config().presence_enabled);
    }

    #[test]
    fn test_invalid_sample_rate() {
        let result = DialogueEq::new(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_hpf_attenuates_low_frequency() {
        let mut eq = DialogueEq::new(48000).unwrap();
        // Only enable HPF
        let mut cfg = eq.config().clone();
        cfg.low_mid_enabled = false;
        cfg.presence_enabled = false;
        cfg.air_enabled = false;
        eq.set_config(cfg).unwrap();

        // 20 Hz signal should be attenuated by an 80 Hz HPF.
        // Use a long block so the biquad settles well past the initial transient;
        // measure only the final quarter to avoid the onset region.
        let mut block = sine_block(20.0, 48000, 16384);
        eq.process_block(&mut block);
        let steady_state = &block[12288..];
        let out_rms = rms(steady_state);
        // 2nd-order Butterworth: 20 Hz is 2 octaves below 80 Hz → ≥ 24 dB attenuation
        // which corresponds to a linear ratio < 0.063.  Allow a generous margin of 0.25.
        assert!(out_rms < 0.25, "20 Hz should be heavily attenuated by 80 Hz HPF, got steady-state rms={out_rms}");
    }

    #[test]
    fn test_hpf_passes_high_frequency() {
        let mut eq = DialogueEq::new(48000).unwrap();
        let mut cfg = eq.config().clone();
        cfg.low_mid_enabled = false;
        cfg.presence_enabled = false;
        cfg.air_enabled = false;
        eq.set_config(cfg).unwrap();

        // 1 kHz is well above cutoff and should pass through with ~unity gain
        let mut block = sine_block(1000.0, 48000, 4096);
        let before_rms = rms(&block);
        eq.process_block(&mut block);
        let after_rms = rms(&block);
        let ratio = after_rms / before_rms;
        assert!(ratio > 0.95, "1 kHz should pass HPF with near-unity gain, ratio={ratio}");
    }

    #[test]
    fn test_presence_boost_increases_level() {
        let sr = 48000u32;
        let mut eq = DialogueEq::new(sr).unwrap();
        let mut cfg = eq.config().clone();
        cfg.hpf_enabled = false;
        cfg.low_mid_enabled = false;
        cfg.air_enabled = false;
        cfg.presence_freq_hz = 3000.0;
        cfg.presence_gain_db = 6.0;
        cfg.presence_q = 1.0;
        eq.set_config(cfg).unwrap();

        // Sine at presence frequency should be boosted
        let mut block = sine_block(3000.0, sr, 4096);
        let before_rms = rms(&block);
        eq.process_block(&mut block);
        let after_rms = rms(&block);
        assert!(
            after_rms > before_rms * 1.3,
            "3 kHz should be boosted by presence band, before={before_rms} after={after_rms}"
        );
    }

    #[test]
    fn test_low_mid_cut_reduces_level() {
        let sr = 48000u32;
        let mut eq = DialogueEq::new(sr).unwrap();
        let mut cfg = eq.config().clone();
        cfg.hpf_enabled = false;
        cfg.presence_enabled = false;
        cfg.air_enabled = false;
        cfg.low_mid_freq_hz = 330.0;
        cfg.low_mid_gain_db = -6.0;
        cfg.low_mid_q = 0.7;
        eq.set_config(cfg).unwrap();

        let mut block = sine_block(330.0, sr, 4096);
        let before_rms = rms(&block);
        eq.process_block(&mut block);
        let after_rms = rms(&block);
        assert!(
            after_rms < before_rms * 0.8,
            "330 Hz should be attenuated by low-mid cut, before={before_rms} after={after_rms}"
        );
    }

    #[test]
    fn test_reset_clears_state() {
        let mut eq = DialogueEq::new(48000).unwrap();
        let mut block = sine_block(1000.0, 48000, 512);
        eq.process_block(&mut block);
        eq.reset();
        // After reset internal state should be zero; processing silence yields silence
        let mut silent = vec![0.0f32; 512];
        eq.process_block(&mut silent);
        let rms_out = rms(&silent);
        assert!(rms_out < 1e-10, "After reset, silence should produce silence");
    }

    #[test]
    fn test_analysis_gain_change_sign() {
        let sr = 48000u32;
        let mut eq = DialogueEq::new(sr).unwrap();
        // Boost 3 kHz only
        let mut cfg = eq.config().clone();
        cfg.hpf_enabled = false;
        cfg.low_mid_enabled = false;
        cfg.air_enabled = false;
        cfg.presence_gain_db = 6.0;
        cfg.presence_freq_hz = 3000.0;
        eq.set_config(cfg).unwrap();

        let mut block = sine_block(3000.0, sr, 4096);
        let analysis = eq.process_block_with_analysis(&mut block);
        assert!(
            analysis.gain_change_db > 0.0,
            "Boosting presence should yield positive gain change, got {}",
            analysis.gain_change_db
        );
    }

    #[test]
    fn test_stereo_interleaved() {
        let sr = 48000u32;
        let mut l_eq = DialogueEq::new(sr).unwrap();
        let mut r_eq = DialogueEq::new(sr).unwrap();
        let n_frames = 512;
        let mut interleaved: Vec<f32> = (0..n_frames * 2)
            .map(|i| {
                let frame = i / 2;
                (2.0 * std::f32::consts::PI * 1000.0 * frame as f32 / sr as f32).sin()
            })
            .collect();
        let result = l_eq.process_stereo_interleaved(&mut interleaved, &mut r_eq);
        assert!(result.is_ok());
    }

    #[test]
    fn test_stereo_odd_length_error() {
        let sr = 48000u32;
        let mut l_eq = DialogueEq::new(sr).unwrap();
        let mut r_eq = DialogueEq::new(sr).unwrap();
        let mut bad = vec![0.0f32; 5];
        let result = l_eq.process_stereo_interleaved(&mut bad, &mut r_eq);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_frequency_rejected() {
        let mut eq = DialogueEq::new(48000).unwrap();
        let mut cfg = eq.config().clone();
        cfg.hpf_cutoff_hz = 0.0; // invalid
        assert!(eq.set_config(cfg).is_err());
    }
}
