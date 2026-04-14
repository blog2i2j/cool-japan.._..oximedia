#![allow(dead_code)]
//! Stereo width restoration for collapsed or narrow stereo fields.
//!
//! Many archival recordings — early stereo masters, cassette dubs, broadcast
//! captures — suffer from a collapsed or artificially narrow stereo image.
//! This module provides tools to diagnose and restore the stereo width using
//! mid/side processing, Haas-effect decorrelation, and frequency-dependent
//! widening.
//!
//! # Techniques
//!
//! - **Mid/Side rebalancing** — adjusts the relative level of the mid (mono)
//!   and side (stereo difference) components.
//! - **Frequency-dependent widening** — applies different width factors to
//!   low, mid, and high frequency bands so that bass remains centred while
//!   higher frequencies are spread.
//! - **Haas decorrelation** — introduces a small inter-channel delay to
//!   one channel, increasing perceived width without altering the spectral
//!   content.
//! - **Phase decorrelation** — applies an all-pass filter to one channel to
//!   create frequency-dependent phase shift for a more natural widening.
//!
//! # Example
//!
//! ```
//! use oximedia_restore::stereo_width::*;
//!
//! let config = StereoWidthConfig::default();
//! let mut processor = StereoWidthProcessor::new(config, 44100);
//! let left  = vec![0.0f32; 1024];
//! let right = vec![0.0f32; 1024];
//! let (out_l, out_r) = processor.process(&left, &right).unwrap();
//! assert_eq!(out_l.len(), left.len());
//! ```

use crate::error::{RestoreError, RestoreResult};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Widening strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WideningMode {
    /// Simple mid/side level rebalancing.
    MidSide,
    /// Frequency-dependent widening with 3 bands.
    MultiBand,
    /// Haas-effect decorrelation (small delay on one channel).
    HaasDelay,
    /// All-pass phase decorrelation.
    PhaseDecorrelation,
}

/// Configuration for stereo width restoration.
#[derive(Debug, Clone)]
pub struct StereoWidthConfig {
    /// Widening mode to use.
    pub mode: WideningMode,
    /// Target stereo width factor.
    /// 0.0 = mono, 1.0 = original, >1.0 = wider than original.
    pub width_factor: f64,
    /// Haas delay in samples (used in [`WideningMode::HaasDelay`]).
    pub haas_delay_samples: usize,
    /// Low-band crossover frequency (Hz) for multiband mode.
    pub low_crossover_hz: f64,
    /// High-band crossover frequency (Hz) for multiband mode.
    pub high_crossover_hz: f64,
    /// Width factor for the low band in multiband mode.
    pub low_band_width: f64,
    /// Width factor for the mid band in multiband mode.
    pub mid_band_width: f64,
    /// Width factor for the high band in multiband mode.
    pub high_band_width: f64,
    /// All-pass filter coefficient for phase decorrelation (0.0–1.0).
    pub allpass_coeff: f64,
    /// Preserve mono compatibility by limiting the maximum side gain.
    pub mono_safe: bool,
    /// Maximum allowed side gain multiplier when `mono_safe` is true.
    pub max_side_gain: f64,
}

impl Default for StereoWidthConfig {
    fn default() -> Self {
        Self {
            mode: WideningMode::MidSide,
            width_factor: 1.5,
            haas_delay_samples: 15,
            low_crossover_hz: 200.0,
            high_crossover_hz: 5000.0,
            low_band_width: 0.8,
            mid_band_width: 1.5,
            high_band_width: 2.0,
            allpass_coeff: 0.6,
            mono_safe: true,
            max_side_gain: 3.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Stereo width analysis
// ---------------------------------------------------------------------------

/// Analysis of the current stereo width characteristics.
#[derive(Debug, Clone)]
pub struct StereoWidthAnalysis {
    /// Current width ratio (side RMS / mid RMS).
    pub width_ratio: f64,
    /// Cross-correlation coefficient (-1.0 to 1.0).
    pub correlation: f64,
    /// Mid-channel RMS level.
    pub mid_rms: f64,
    /// Side-channel RMS level.
    pub side_rms: f64,
    /// Whether the stereo field is considered "collapsed" (width_ratio < 0.1).
    pub is_collapsed: bool,
    /// Whether the signal is essentially mono.
    pub is_mono: bool,
}

// ---------------------------------------------------------------------------
// M/S helpers
// ---------------------------------------------------------------------------

/// Encode left/right to mid/side.
fn encode_ms(left: &[f32], right: &[f32]) -> (Vec<f64>, Vec<f64>) {
    let n = left.len().min(right.len());
    let mut mid = Vec::with_capacity(n);
    let mut side = Vec::with_capacity(n);
    for i in 0..n {
        let l = left[i] as f64;
        let r = right[i] as f64;
        mid.push((l + r) * 0.5);
        side.push((l - r) * 0.5);
    }
    (mid, side)
}

/// Decode mid/side back to left/right.
fn decode_ms(mid: &[f64], side: &[f64]) -> (Vec<f32>, Vec<f32>) {
    let n = mid.len().min(side.len());
    let mut left = Vec::with_capacity(n);
    let mut right = Vec::with_capacity(n);
    for i in 0..n {
        left.push((mid[i] + side[i]) as f32);
        right.push((mid[i] - side[i]) as f32);
    }
    (left, right)
}

/// Compute RMS of a f64 slice.
fn rms_f64(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let sum: f64 = data.iter().map(|s| s * s).sum();
    (sum / data.len() as f64).sqrt()
}

/// Compute cross-correlation between two f32 slices.
fn cross_corr(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut sum_ab = 0.0_f64;
    let mut sum_aa = 0.0_f64;
    let mut sum_bb = 0.0_f64;
    for i in 0..n {
        let va = a[i] as f64;
        let vb = b[i] as f64;
        sum_ab += va * vb;
        sum_aa += va * va;
        sum_bb += vb * vb;
    }
    let denom = (sum_aa * sum_bb).sqrt();
    if denom < 1e-20 {
        0.0
    } else {
        sum_ab / denom
    }
}

// ---------------------------------------------------------------------------
// Simple one-pole filter for band splitting
// ---------------------------------------------------------------------------

/// Single-pole low-pass filter.
#[derive(Debug, Clone)]
struct OnePoleLP {
    coeff: f64,
    state: f64,
}

impl OnePoleLP {
    #[allow(clippy::cast_precision_loss)]
    fn new(cutoff_hz: f64, sample_rate: u32) -> Self {
        let w = (2.0 * PI * cutoff_hz / sample_rate as f64).min(PI * 0.99);
        let coeff = 1.0 - (-w).exp();
        Self { coeff, state: 0.0 }
    }

    fn reset(&mut self) {
        self.state = 0.0;
    }

    fn process_sample(&mut self, x: f64) -> f64 {
        self.state += self.coeff * (x - self.state);
        self.state
    }
}

// ---------------------------------------------------------------------------
// All-pass filter for phase decorrelation
// ---------------------------------------------------------------------------

/// First-order all-pass filter.
#[derive(Debug, Clone)]
struct AllPassFilter {
    coeff: f64,
    x_prev: f64,
    y_prev: f64,
}

impl AllPassFilter {
    fn new(coeff: f64) -> Self {
        Self {
            coeff,
            x_prev: 0.0,
            y_prev: 0.0,
        }
    }

    fn process_sample(&mut self, x: f64) -> f64 {
        let y = self.coeff * (x - self.y_prev) + self.x_prev;
        self.x_prev = x;
        self.y_prev = y;
        y
    }
}

// ---------------------------------------------------------------------------
// Processor
// ---------------------------------------------------------------------------

/// Stereo width restoration processor.
#[derive(Debug, Clone)]
pub struct StereoWidthProcessor {
    config: StereoWidthConfig,
    sample_rate: u32,
    low_lp_l: OnePoleLP,
    low_lp_r: OnePoleLP,
    high_lp_l: OnePoleLP,
    high_lp_r: OnePoleLP,
    allpass: AllPassFilter,
}

impl StereoWidthProcessor {
    /// Create a new stereo width processor.
    pub fn new(config: StereoWidthConfig, sample_rate: u32) -> Self {
        let low_lp_l = OnePoleLP::new(config.low_crossover_hz, sample_rate);
        let low_lp_r = OnePoleLP::new(config.low_crossover_hz, sample_rate);
        let high_lp_l = OnePoleLP::new(config.high_crossover_hz, sample_rate);
        let high_lp_r = OnePoleLP::new(config.high_crossover_hz, sample_rate);
        let allpass = AllPassFilter::new(config.allpass_coeff);
        Self {
            config,
            sample_rate,
            low_lp_l,
            low_lp_r,
            high_lp_l,
            high_lp_r,
            allpass,
        }
    }

    /// Update configuration.
    pub fn set_config(&mut self, config: StereoWidthConfig) {
        self.low_lp_l = OnePoleLP::new(config.low_crossover_hz, self.sample_rate);
        self.low_lp_r = OnePoleLP::new(config.low_crossover_hz, self.sample_rate);
        self.high_lp_l = OnePoleLP::new(config.high_crossover_hz, self.sample_rate);
        self.high_lp_r = OnePoleLP::new(config.high_crossover_hz, self.sample_rate);
        self.allpass = AllPassFilter::new(config.allpass_coeff);
        self.config = config;
    }

    /// Return a reference to the current configuration.
    pub fn config(&self) -> &StereoWidthConfig {
        &self.config
    }

    /// Analyse the stereo width of a signal pair.
    pub fn analyse(&self, left: &[f32], right: &[f32]) -> StereoWidthAnalysis {
        let (mid, side) = encode_ms(left, right);
        let mid_rms = rms_f64(&mid);
        let side_rms = rms_f64(&side);
        let correlation = cross_corr(left, right);
        let width_ratio = if mid_rms > 1e-10 {
            side_rms / mid_rms
        } else {
            0.0
        };

        StereoWidthAnalysis {
            width_ratio,
            correlation,
            mid_rms,
            side_rms,
            is_collapsed: width_ratio < 0.1,
            is_mono: width_ratio < 0.01,
        }
    }

    /// Process a stereo pair, returning the width-adjusted output.
    ///
    /// # Errors
    ///
    /// Returns [`RestoreError::InvalidParameter`] if the channels have
    /// different lengths.
    pub fn process(&mut self, left: &[f32], right: &[f32]) -> RestoreResult<(Vec<f32>, Vec<f32>)> {
        if left.len() != right.len() {
            return Err(RestoreError::InvalidParameter(format!(
                "channel length mismatch: left={}, right={}",
                left.len(),
                right.len()
            )));
        }

        if left.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        match self.config.mode {
            WideningMode::MidSide => self.process_mid_side(left, right),
            WideningMode::MultiBand => self.process_multiband(left, right),
            WideningMode::HaasDelay => self.process_haas(left, right),
            WideningMode::PhaseDecorrelation => self.process_phase_decorrelation(left, right),
        }
    }

    /// Simple mid/side width adjustment.
    fn process_mid_side(&self, left: &[f32], right: &[f32]) -> RestoreResult<(Vec<f32>, Vec<f32>)> {
        let (mid, side) = encode_ms(left, right);
        let mut side_gain = self.config.width_factor;

        if self.config.mono_safe {
            side_gain = side_gain.min(self.config.max_side_gain);
        }

        let scaled_side: Vec<f64> = side.iter().map(|&s| s * side_gain).collect();
        let (out_l, out_r) = decode_ms(&mid, &scaled_side);
        Ok((out_l, out_r))
    }

    /// Frequency-dependent multiband widening.
    fn process_multiband(
        &mut self,
        left: &[f32],
        right: &[f32],
    ) -> RestoreResult<(Vec<f32>, Vec<f32>)> {
        let n = left.len();
        let mut out_l = vec![0.0_f32; n];
        let mut out_r = vec![0.0_f32; n];

        self.low_lp_l.reset();
        self.low_lp_r.reset();
        self.high_lp_l.reset();
        self.high_lp_r.reset();

        for i in 0..n {
            let l = left[i] as f64;
            let r = right[i] as f64;

            // 3-band split
            let low_l = self.low_lp_l.process_sample(l);
            let low_r = self.low_lp_r.process_sample(r);

            let below_high_l = self.high_lp_l.process_sample(l);
            let below_high_r = self.high_lp_r.process_sample(r);

            let mid_l = below_high_l - low_l;
            let mid_r = below_high_r - low_r;

            let high_l = l - below_high_l;
            let high_r = r - below_high_r;

            // Width per band via M/S
            let apply_width = |bl: f64, br: f64, width: f64| -> (f64, f64) {
                let m = (bl + br) * 0.5;
                let s = (bl - br) * 0.5;
                let s_scaled = s * width;
                (m + s_scaled, m - s_scaled)
            };

            let (ll, lr) = apply_width(low_l, low_r, self.config.low_band_width);
            let (ml, mr) = apply_width(mid_l, mid_r, self.config.mid_band_width);
            let (hl, hr) = apply_width(high_l, high_r, self.config.high_band_width);

            out_l[i] = (ll + ml + hl) as f32;
            out_r[i] = (lr + mr + hr) as f32;
        }

        Ok((out_l, out_r))
    }

    /// Haas-effect delay-based widening.
    fn process_haas(&self, left: &[f32], right: &[f32]) -> RestoreResult<(Vec<f32>, Vec<f32>)> {
        let n = left.len();
        let delay = self.config.haas_delay_samples;

        let out_l = left.to_vec();
        let mut out_r = vec![0.0_f32; n];

        for i in 0..n {
            if i >= delay {
                out_r[i] = right[i - delay];
            } else {
                out_r[i] = right[i];
            }
        }

        Ok((out_l, out_r))
    }

    /// Phase decorrelation-based widening.
    fn process_phase_decorrelation(
        &mut self,
        left: &[f32],
        right: &[f32],
    ) -> RestoreResult<(Vec<f32>, Vec<f32>)> {
        let n = left.len();
        let out_l = left.to_vec();
        let mut out_r = vec![0.0_f32; n];

        for i in 0..n {
            let filtered = self.allpass.process_sample(right[i] as f64);
            out_r[i] = filtered as f32;
        }

        Ok((out_l, out_r))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stereo_sine(freq: f64, sr: u32, n: usize) -> (Vec<f32>, Vec<f32>) {
        let left: Vec<f32> = (0..n)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let s = (2.0 * PI * freq * i as f64 / sr as f64).sin() as f32;
                s
            })
            .collect();
        let right = left.clone();
        (left, right)
    }

    #[test]
    fn test_default_config() {
        let cfg = StereoWidthConfig::default();
        assert_eq!(cfg.mode, WideningMode::MidSide);
        assert!((cfg.width_factor - 1.5).abs() < 1e-9);
        assert!(cfg.mono_safe);
    }

    #[test]
    fn test_empty_input() {
        let cfg = StereoWidthConfig::default();
        let mut p = StereoWidthProcessor::new(cfg, 44100);
        let (l, r) = p.process(&[], &[]).expect("ok");
        assert!(l.is_empty());
        assert!(r.is_empty());
    }

    #[test]
    fn test_length_mismatch_error() {
        let cfg = StereoWidthConfig::default();
        let mut p = StereoWidthProcessor::new(cfg, 44100);
        let result = p.process(&[0.0; 10], &[0.0; 5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_mid_side_preserves_length() {
        let cfg = StereoWidthConfig::default();
        let mut p = StereoWidthProcessor::new(cfg, 44100);
        let (left, right) = make_stereo_sine(440.0, 44100, 2048);
        let (out_l, out_r) = p.process(&left, &right).expect("ok");
        assert_eq!(out_l.len(), left.len());
        assert_eq!(out_r.len(), right.len());
    }

    #[test]
    fn test_width_factor_one_is_identity() {
        let cfg = StereoWidthConfig {
            width_factor: 1.0,
            ..Default::default()
        };
        let mut p = StereoWidthProcessor::new(cfg, 44100);
        let left: Vec<f32> = (0..256).map(|i| (i as f32) * 0.001).collect();
        let right: Vec<f32> = (0..256).map(|i| (i as f32) * 0.002).collect();
        let (out_l, out_r) = p.process(&left, &right).expect("ok");
        for (a, b) in out_l.iter().zip(left.iter()) {
            assert!((a - b).abs() < 1e-5, "left mismatch");
        }
        for (a, b) in out_r.iter().zip(right.iter()) {
            assert!((a - b).abs() < 1e-5, "right mismatch");
        }
    }

    #[test]
    fn test_mono_signal_widening() {
        let cfg = StereoWidthConfig {
            width_factor: 2.0,
            ..Default::default()
        };
        let mut p = StereoWidthProcessor::new(cfg, 44100);
        // Identical channels (mono)
        let signal: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin()).collect();
        let (out_l, out_r) = p.process(&signal, &signal).expect("ok");
        // For a perfectly mono signal, side = 0 so widening factor doesn't change output
        for (a, b) in out_l.iter().zip(out_r.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "mono signal should remain mono after M/S widening"
            );
        }
    }

    #[test]
    fn test_analyse_mono_signal() {
        let cfg = StereoWidthConfig::default();
        let p = StereoWidthProcessor::new(cfg, 44100);
        let signal: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
        let analysis = p.analyse(&signal, &signal);
        assert!(analysis.is_mono || analysis.is_collapsed);
        assert!(analysis.width_ratio < 0.01);
        assert!(analysis.correlation > 0.99);
    }

    #[test]
    fn test_analyse_wide_signal() {
        let cfg = StereoWidthConfig::default();
        let p = StereoWidthProcessor::new(cfg, 44100);
        let left: Vec<f32> = (0..1024)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let s = (2.0 * PI * 440.0 * i as f64 / 44100.0).sin() as f32;
                s
            })
            .collect();
        let right: Vec<f32> = (0..1024)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let s = (2.0 * PI * 440.0 * i as f64 / 44100.0 + 0.5).sin() as f32;
                s
            })
            .collect();
        let analysis = p.analyse(&left, &right);
        assert!(analysis.width_ratio > 0.0);
        assert!(!analysis.is_mono);
    }

    #[test]
    fn test_multiband_mode() {
        let cfg = StereoWidthConfig {
            mode: WideningMode::MultiBand,
            ..Default::default()
        };
        let mut p = StereoWidthProcessor::new(cfg, 44100);
        let left: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.01).sin()).collect();
        let right: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.012).sin()).collect();
        let (out_l, out_r) = p.process(&left, &right).expect("ok");
        assert_eq!(out_l.len(), 2048);
        assert_eq!(out_r.len(), 2048);
    }

    #[test]
    fn test_haas_mode() {
        let cfg = StereoWidthConfig {
            mode: WideningMode::HaasDelay,
            haas_delay_samples: 10,
            ..Default::default()
        };
        let mut p = StereoWidthProcessor::new(cfg, 44100);
        let left: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let right = left.clone();
        let (out_l, out_r) = p.process(&left, &right).expect("ok");
        // Left should be unchanged
        assert_eq!(out_l, left);
        // Right should be delayed by 10 samples
        for i in 10..128 {
            assert!(
                (out_r[i] - right[i - 10]).abs() < 1e-6,
                "Haas delay mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn test_phase_decorrelation_mode() {
        let cfg = StereoWidthConfig {
            mode: WideningMode::PhaseDecorrelation,
            ..Default::default()
        };
        let mut p = StereoWidthProcessor::new(cfg, 44100);
        let left: Vec<f32> = (0..512).map(|i| (i as f32 * 0.02).sin()).collect();
        let right = left.clone();
        let (out_l, out_r) = p.process(&left, &right).expect("ok");
        assert_eq!(out_l.len(), 512);
        assert_eq!(out_r.len(), 512);
        // Left should be unchanged
        assert_eq!(out_l, left);
        // Right should differ due to all-pass filtering
        let mut differs = false;
        for i in 1..512 {
            if (out_r[i] - right[i]).abs() > 1e-6 {
                differs = true;
                break;
            }
        }
        assert!(
            differs,
            "phase decorrelation should alter the right channel"
        );
    }

    #[test]
    fn test_set_config() {
        let cfg = StereoWidthConfig::default();
        let mut p = StereoWidthProcessor::new(cfg, 44100);
        let new_cfg = StereoWidthConfig {
            width_factor: 3.0,
            mode: WideningMode::MultiBand,
            ..Default::default()
        };
        p.set_config(new_cfg);
        assert_eq!(p.config().mode, WideningMode::MultiBand);
        assert!((p.config().width_factor - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_encode_decode_ms_roundtrip() {
        let left: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let right: Vec<f32> = (0..64).map(|i| i as f32 * 0.05 + 0.2).collect();
        let (mid, side) = encode_ms(&left, &right);
        let (dec_l, dec_r) = decode_ms(&mid, &side);
        for i in 0..64 {
            assert!((dec_l[i] - left[i]).abs() < 1e-5, "L roundtrip at {i}");
            assert!((dec_r[i] - right[i]).abs() < 1e-5, "R roundtrip at {i}");
        }
    }

    #[test]
    fn test_mono_safe_limits_gain() {
        let cfg = StereoWidthConfig {
            width_factor: 100.0,
            mono_safe: true,
            max_side_gain: 3.0,
            ..Default::default()
        };
        let mut p = StereoWidthProcessor::new(cfg, 44100);
        let left: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
        let right: Vec<f32> = (0..256).map(|i| (i as f32 * 0.012).cos()).collect();
        let result = p.process(&left, &right);
        assert!(result.is_ok());
    }
}
