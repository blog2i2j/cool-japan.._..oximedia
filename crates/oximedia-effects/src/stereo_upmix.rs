//! Stereo-to-surround upmixer for 5.1 and 7.1 channel configurations.
//!
//! Implements a matrix-based psychoacoustic upmixer that derives additional
//! surround channels from a stereo source using phase/amplitude cues, centre
//! channel extraction via mid/side decomposition, LFE (subwoofer) extraction
//! using a crossover filter, and decorrelated rear-channel synthesis.
//!
//! # Channel ordering
//!
//! The output arrays follow the standard ITU/Dolby/DTS channel ordering:
//!
//! ## 5.1
//! `[L, R, C, LFE, Ls, Rs]`
//!
//! ## 7.1
//! `[L, R, C, LFE, Lss, Rss, Lrs, Rrs]`
//!
//! where `Lss`/`Rss` = left/right side surrounds and `Lrs`/`Rrs` = left/right
//! rear surrounds.
//!
//! # Algorithm
//!
//! 1. **Mid/Side decomposition**: `M = (L+R)/√2`, `S = (L-R)/√2`
//! 2. **Centre extraction**: centre ≈ `M` attenuated by `centre_level`
//! 3. **LFE extraction**: low-pass (80 Hz) filtered mono signal
//! 4. **Surround extraction**: allpass-filtered Side signal with decorrelation
//! 5. **Rear (7.1)**: decorrelated version of the surround with additional delay
//!
//! # Example
//!
//! ```
//! use oximedia_effects::stereo_upmix::{StereoUpmixer, UpmixFormat, UpmixConfig};
//!
//! let config = UpmixConfig::default();
//! let mut upmixer = StereoUpmixer::new(config, 48_000.0);
//!
//! let (l_in, r_in) = (0.5_f32, -0.3_f32);
//! let out = upmixer.process_sample_5_1(l_in, r_in);
//! // out: [L, R, C, LFE, Ls, Rs]
//! assert_eq!(out.len(), 6);
//! ```

#![allow(dead_code)]

use std::f32::consts::PI;

// ─── Helper: one-pole lowpass filter ─────────────────────────────────────────

/// First-order IIR lowpass filter (one-pole).
#[derive(Debug, Clone)]
struct OnePole {
    /// Coefficient for the feedback term.
    coeff: f32,
    /// Previous output sample.
    state: f32,
}

impl OnePole {
    /// Create a lowpass one-pole filter with the given cutoff frequency.
    fn new_lp(cutoff_hz: f32, sample_rate: f32) -> Self {
        let omega = 2.0 * PI * cutoff_hz.max(1.0) / sample_rate.max(1.0);
        let coeff = (-omega).exp();
        Self { coeff, state: 0.0 }
    }

    /// Create a highpass one-pole filter with the given cutoff frequency.
    fn new_hp(cutoff_hz: f32, sample_rate: f32) -> Self {
        let lp = Self::new_lp(cutoff_hz, sample_rate);
        // Mirror: for HP we store the LP coeff and invert on output
        lp
    }

    /// Process a sample through the lowpass filter.
    #[inline]
    fn process_lp(&mut self, x: f32) -> f32 {
        self.state = x * (1.0 - self.coeff) + self.state * self.coeff;
        self.state
    }

    /// Process a sample through a highpass derived from this pole.
    #[inline]
    fn process_hp(&mut self, x: f32) -> f32 {
        let lp = self.process_lp(x);
        x - lp
    }

    /// Reset filter state.
    fn reset(&mut self) {
        self.state = 0.0;
    }
}

// ─── Allpass filter for decorrelation ────────────────────────────────────────

/// First-order allpass filter.
///
/// Used to decorrelate side/surround channels so they sound spatially separate
/// from the front channels without introducing comb filtering artefacts.
#[derive(Debug, Clone)]
struct AllPass {
    coeff: f32,
    state: f32,
}

impl AllPass {
    /// Create an allpass with coefficient `g` (typically 0.3–0.7).
    fn new(g: f32) -> Self {
        Self {
            coeff: g.clamp(-0.999, 0.999),
            state: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        // y[n] = -g*x[n] + x[n-1] + g*y[n-1]
        let y = -self.coeff * x + self.state;
        self.state = x + self.coeff * y;
        y
    }

    fn reset(&mut self) {
        self.state = 0.0;
    }
}

// ─── Short delay line for rear-channel decorrelation ─────────────────────────

/// Simple ring-buffer delay line.
#[derive(Debug, Clone)]
struct ShortDelay {
    buf: Vec<f32>,
    pos: usize,
}

impl ShortDelay {
    fn new(max_samples: usize) -> Self {
        Self {
            buf: vec![0.0; max_samples.max(1)],
            pos: 0,
        }
    }

    fn process(&mut self, x: f32) -> f32 {
        let out = self.buf[self.pos];
        self.buf[self.pos] = x;
        self.pos = (self.pos + 1) % self.buf.len();
        out
    }

    fn reset(&mut self) {
        self.buf.fill(0.0);
        self.pos = 0;
    }
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Output format for the upmixer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpmixFormat {
    /// 5.1 surround (6 channels): L, R, C, LFE, Ls, Rs
    Surround5_1,
    /// 7.1 surround (8 channels): L, R, C, LFE, Lss, Rss, Lrs, Rrs
    Surround7_1,
}

/// Configuration for the stereo upmixer.
#[derive(Debug, Clone)]
pub struct UpmixConfig {
    /// Centre channel level `[0.0, 1.0]`.
    ///
    /// Controls how much of the mid (M = L+R) signal is routed to the centre
    /// channel.  `0.0` = phantom centre (no centre channel output), `1.0` =
    /// full centre extraction.
    pub centre_level: f32,

    /// LFE (subwoofer) level `[0.0, 1.0]`.
    ///
    /// Controls the gain applied to the extracted bass content routed to LFE.
    pub lfe_level: f32,

    /// Surround level `[0.0, 1.0]`.
    ///
    /// Controls how much of the side (S = L−R) signal is mixed into surround
    /// channels.
    pub surround_level: f32,

    /// Rear attenuation for 7.1 mode `[0.0, 1.0]`.
    ///
    /// Applied to the rear (Lrs/Rrs) channels relative to the side surrounds.
    pub rear_attenuation: f32,

    /// LFE crossover frequency in Hz.  Signals below this are routed to LFE.
    pub lfe_crossover_hz: f32,

    /// Delay in milliseconds applied to the rear channels in 7.1 to improve
    /// spaciousness.  Typical: 10–30 ms.
    pub rear_delay_ms: f32,
}

impl Default for UpmixConfig {
    fn default() -> Self {
        Self {
            centre_level: 0.7,
            lfe_level: 0.8,
            surround_level: 0.7,
            rear_attenuation: 0.6,
            lfe_crossover_hz: 80.0,
            rear_delay_ms: 20.0,
        }
    }
}

impl UpmixConfig {
    /// Preset for cinema-style upmixing (wide, enveloping surround field).
    #[must_use]
    pub fn cinema() -> Self {
        Self {
            centre_level: 0.85,
            lfe_level: 1.0,
            surround_level: 0.8,
            rear_attenuation: 0.7,
            lfe_crossover_hz: 80.0,
            rear_delay_ms: 25.0,
        }
    }

    /// Preset for music upmixing (subtle, transparent surround).
    #[must_use]
    pub fn music() -> Self {
        Self {
            centre_level: 0.4,
            lfe_level: 0.5,
            surround_level: 0.5,
            rear_attenuation: 0.4,
            lfe_crossover_hz: 60.0,
            rear_delay_ms: 15.0,
        }
    }

    /// Preset for broadcast/TV content.
    #[must_use]
    pub fn broadcast() -> Self {
        Self {
            centre_level: 1.0,
            lfe_level: 0.7,
            surround_level: 0.6,
            rear_attenuation: 0.5,
            lfe_crossover_hz: 80.0,
            rear_delay_ms: 20.0,
        }
    }
}

// ─── Upmixer ──────────────────────────────────────────────────────────────────

/// Stereo-to-surround upmixer supporting 5.1 and 7.1 output formats.
///
/// # Thread safety
///
/// Not `Sync` or `Send` by default (mutable state). Create a separate instance
/// per audio thread.
pub struct StereoUpmixer {
    config: UpmixConfig,
    sample_rate: f32,

    /// Lowpass filter for LFE extraction.
    lfe_lp: OnePole,

    /// Allpass decorrelators for left and right surround channels.
    surround_ap_l: AllPass,
    surround_ap_r: AllPass,

    /// Allpass decorrelators for 7.1 rear channels (extra decorrelation).
    rear_ap_l: AllPass,
    rear_ap_r: AllPass,

    /// Delay line for left rear channel (7.1 only).
    rear_delay_l: ShortDelay,
    /// Delay line for right rear channel (7.1 only).
    rear_delay_r: ShortDelay,

    /// Smoother for centre level (prevent zipper noise).
    smooth_centre: f32,
    /// Smoother for LFE level.
    smooth_lfe: f32,
    /// Smoother for surround level.
    smooth_surround: f32,

    /// One-pole smoothing coefficient (≈ 5 ms at the current sample rate).
    smooth_coeff: f32,
}

impl StereoUpmixer {
    /// Create a new stereo upmixer.
    ///
    /// # Arguments
    /// * `config` - Upmixer configuration
    /// * `sample_rate` - Sample rate in Hz
    #[must_use]
    pub fn new(config: UpmixConfig, sample_rate: f32) -> Self {
        let sr = sample_rate.max(1.0);
        let lfe_lp = OnePole::new_lp(config.lfe_crossover_hz, sr);

        // Decorrelation allpass coefficients chosen for each channel to ensure
        // independent phase responses (prime-number-derived).
        let surround_ap_l = AllPass::new(0.4142); // ≈ √2 − 1
        let surround_ap_r = AllPass::new(-0.3820); // ≈ -(√5 − 2)
        let rear_ap_l = AllPass::new(0.5176); // additional decorrelation
        let rear_ap_r = AllPass::new(-0.4641);

        // Convert rear delay from ms to samples.
        let rear_samples = ((config.rear_delay_ms * sr / 1000.0) as usize).max(1);
        let rear_delay_l = ShortDelay::new(rear_samples);
        let rear_delay_r = ShortDelay::new(rear_samples);

        // Parameter smoother: ~5 ms time constant.
        let smooth_coeff = (-1.0_f32 / (0.005 * sr)).exp();

        Self {
            smooth_centre: config.centre_level,
            smooth_lfe: config.lfe_level,
            smooth_surround: config.surround_level,
            config,
            sample_rate: sr,
            lfe_lp,
            surround_ap_l,
            surround_ap_r,
            rear_ap_l,
            rear_ap_r,
            rear_delay_l,
            rear_delay_r,
            smooth_coeff,
        }
    }

    // ── Parameter setters ─────────────────────────────────────────────────

    /// Set the centre channel extraction level `[0.0, 1.0]`.
    pub fn set_centre_level(&mut self, level: f32) {
        self.config.centre_level = level.clamp(0.0, 1.0);
    }

    /// Set the LFE channel level `[0.0, 1.0]`.
    pub fn set_lfe_level(&mut self, level: f32) {
        self.config.lfe_level = level.clamp(0.0, 1.0);
    }

    /// Set the surround channel level `[0.0, 1.0]`.
    pub fn set_surround_level(&mut self, level: f32) {
        self.config.surround_level = level.clamp(0.0, 1.0);
    }

    /// Set the rear channel attenuation in 7.1 mode `[0.0, 1.0]`.
    pub fn set_rear_attenuation(&mut self, attenuation: f32) {
        self.config.rear_attenuation = attenuation.clamp(0.0, 1.0);
    }

    /// Set the sample rate and reinitialise filters.
    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        let sr = sample_rate.max(1.0);
        self.sample_rate = sr;
        self.lfe_lp = OnePole::new_lp(self.config.lfe_crossover_hz, sr);
        self.smooth_coeff = (-1.0_f32 / (0.005 * sr)).exp();
        let rear_samples = ((self.config.rear_delay_ms * sr / 1000.0) as usize).max(1);
        self.rear_delay_l = ShortDelay::new(rear_samples);
        self.rear_delay_r = ShortDelay::new(rear_samples);
    }

    /// Reset all filter and delay states.
    pub fn reset(&mut self) {
        self.lfe_lp.reset();
        self.surround_ap_l.reset();
        self.surround_ap_r.reset();
        self.rear_ap_l.reset();
        self.rear_ap_r.reset();
        self.rear_delay_l.reset();
        self.rear_delay_r.reset();
    }

    // ── Core decomposition ────────────────────────────────────────────────

    /// Advance smoothed parameters one sample.
    #[inline]
    fn advance_smoothers(&mut self) {
        let c = self.smooth_coeff;
        self.smooth_centre = self.config.centre_level * (1.0 - c) + self.smooth_centre * c;
        self.smooth_lfe = self.config.lfe_level * (1.0 - c) + self.smooth_lfe * c;
        self.smooth_surround = self.config.surround_level * (1.0 - c) + self.smooth_surround * c;
    }

    /// Decompose one stereo sample into its mid/side components and derive
    /// the shared signals used by both 5.1 and 7.1 outputs.
    ///
    /// Returns `(l_out, r_out, centre, lfe, surr_l, surr_r)`.
    #[inline]
    fn decompose(&mut self, l: f32, r: f32) -> (f32, f32, f32, f32, f32, f32) {
        self.advance_smoothers();

        let sqrt2_inv = std::f32::consts::FRAC_1_SQRT_2;

        // Mid/Side decomposition.
        let mid = (l + r) * sqrt2_inv;
        let side = (l - r) * sqrt2_inv;

        // Centre: mid attenuated by centre_level.
        // The residual (1 − centre_level) of mid energy remains in L/R.
        let c_gain = self.smooth_centre;
        let centre = mid * c_gain;
        let residual_mid = mid * (1.0 - c_gain) * sqrt2_inv;

        // L/R output: dry input minus centre bleed plus residual mid.
        let l_out = l - centre * sqrt2_inv * 0.5 + residual_mid * 0.5;
        let r_out = r - centre * sqrt2_inv * 0.5 + residual_mid * 0.5;

        // LFE: low-passed mono signal.
        let mono_lp = self.lfe_lp.process_lp(mid);
        let lfe = mono_lp * self.smooth_lfe;

        // Surround: allpass-decorrelated side signal.
        let surr_l = self.surround_ap_l.process(side) * self.smooth_surround;
        let surr_r = self.surround_ap_r.process(-side) * self.smooth_surround;

        (l_out, r_out, centre, lfe, surr_l, surr_r)
    }

    // ── Public processing API ─────────────────────────────────────────────

    /// Process one stereo sample pair and return a 6-channel 5.1 frame.
    ///
    /// Output channel ordering: `[L, R, C, LFE, Ls, Rs]`
    #[must_use]
    pub fn process_sample_5_1(&mut self, l: f32, r: f32) -> [f32; 6] {
        let (l_out, r_out, centre, lfe, surr_l, surr_r) = self.decompose(l, r);
        [l_out, r_out, centre, lfe, surr_l, surr_r]
    }

    /// Process one stereo sample pair and return an 8-channel 7.1 frame.
    ///
    /// Output channel ordering: `[L, R, C, LFE, Lss, Rss, Lrs, Rrs]`
    #[must_use]
    pub fn process_sample_7_1(&mut self, l: f32, r: f32) -> [f32; 8] {
        let (l_out, r_out, centre, lfe, surr_l, surr_r) = self.decompose(l, r);

        // Rear surrounds: further decorrelated and delayed surround signal.
        let rear_l_raw = self.rear_ap_l.process(surr_l) * self.config.rear_attenuation;
        let rear_r_raw = self.rear_ap_r.process(surr_r) * self.config.rear_attenuation;

        let rear_l = self.rear_delay_l.process(rear_l_raw);
        let rear_r = self.rear_delay_r.process(rear_r_raw);

        [l_out, r_out, centre, lfe, surr_l, surr_r, rear_l, rear_r]
    }

    /// Process a block of stereo samples into a 5.1 multi-channel output.
    ///
    /// `stereo` must have an even number of samples (interleaved L/R).
    /// Returns an interleaved 6-channel `Vec<f32>` with ordering
    /// `[L0, R0, C0, LFE0, Ls0, Rs0, L1, R1, …]`.
    #[must_use]
    pub fn process_block_5_1(&mut self, stereo: &[f32]) -> Vec<f32> {
        let frames = stereo.len() / 2;
        let mut out = Vec::with_capacity(frames * 6);
        for i in 0..frames {
            let l = stereo[i * 2];
            let r = stereo[i * 2 + 1];
            let frame = self.process_sample_5_1(l, r);
            out.extend_from_slice(&frame);
        }
        out
    }

    /// Process a block of stereo samples into a 7.1 multi-channel output.
    ///
    /// `stereo` must have an even number of samples (interleaved L/R).
    /// Returns an interleaved 8-channel `Vec<f32>`.
    #[must_use]
    pub fn process_block_7_1(&mut self, stereo: &[f32]) -> Vec<f32> {
        let frames = stereo.len() / 2;
        let mut out = Vec::with_capacity(frames * 8);
        for i in 0..frames {
            let l = stereo[i * 2];
            let r = stereo[i * 2 + 1];
            let frame = self.process_sample_7_1(l, r);
            out.extend_from_slice(&frame);
        }
        out
    }

    /// Get the current sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &UpmixConfig {
        &self.config
    }

    /// Upmix format selector: process sample and return a `Vec<f32>`.
    ///
    /// Returns 6 channels for 5.1, 8 channels for 7.1.
    #[must_use]
    pub fn process_sample_fmt(&mut self, l: f32, r: f32, format: UpmixFormat) -> Vec<f32> {
        match format {
            UpmixFormat::Surround5_1 => self.process_sample_5_1(l, r).to_vec(),
            UpmixFormat::Surround7_1 => self.process_sample_7_1(l, r).to_vec(),
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_upmixer() -> StereoUpmixer {
        StereoUpmixer::new(UpmixConfig::default(), 48_000.0)
    }

    /// Compute RMS of a slice.
    fn rms(buf: &[f32]) -> f32 {
        if buf.is_empty() {
            return 0.0;
        }
        (buf.iter().map(|&x| x * x).sum::<f32>() / buf.len() as f32).sqrt()
    }

    // ── Basic output structure ─────────────────────────────────────────────

    #[test]
    fn test_5_1_output_has_6_channels() {
        let mut um = make_upmixer();
        let frame = um.process_sample_5_1(0.5, -0.3);
        assert_eq!(frame.len(), 6, "5.1 frame must have exactly 6 channels");
    }

    #[test]
    fn test_7_1_output_has_8_channels() {
        let mut um = make_upmixer();
        let frame = um.process_sample_7_1(0.5, -0.3);
        assert_eq!(frame.len(), 8, "7.1 frame must have exactly 8 channels");
    }

    #[test]
    fn test_all_output_samples_finite() {
        let mut um = make_upmixer();
        for i in 0..512 {
            let phase = i as f32 / 48.0;
            let l = (phase * 0.3).sin() * 0.7;
            let r = (phase * 0.4).sin() * 0.5;
            let frame6 = um.process_sample_5_1(l, r);
            for (ch, &s) in frame6.iter().enumerate() {
                assert!(s.is_finite(), "5.1 channel {ch} sample {i} not finite: {s}");
            }
            let frame8 = um.process_sample_7_1(l, r);
            for (ch, &s) in frame8.iter().enumerate() {
                assert!(s.is_finite(), "7.1 channel {ch} sample {i} not finite: {s}");
            }
        }
    }

    // ── Silence pass-through ───────────────────────────────────────────────

    #[test]
    fn test_silence_in_produces_near_silence_out() {
        let mut um = make_upmixer();
        // Feed silence for 256 samples to let filters settle, then check
        for _ in 0..256 {
            let frame = um.process_sample_5_1(0.0, 0.0);
            for &s in &frame {
                assert!(
                    s.abs() < 1e-6,
                    "silence in should produce ~silence out, got {s}"
                );
            }
        }
    }

    // ── Centre channel extraction ──────────────────────────────────────────

    #[test]
    fn test_mono_signal_extracts_centre() {
        // A mono signal (L == R) should produce a non-zero centre channel.
        let config = UpmixConfig {
            centre_level: 1.0,
            ..Default::default()
        };
        let mut um = StereoUpmixer::new(config, 48_000.0);

        // Run 128 samples to let the smoother settle.
        let mono = 0.5_f32;
        let mut centre_rms_acc = 0.0_f32;
        for i in 0..256 {
            let frame = um.process_sample_5_1(mono, mono);
            if i >= 128 {
                centre_rms_acc += frame[2] * frame[2]; // channel 2 = C
            }
        }
        let centre_rms = (centre_rms_acc / 128.0).sqrt();
        assert!(
            centre_rms > 0.1,
            "Mono signal should produce significant centre output, got rms={centre_rms:.4}"
        );
    }

    #[test]
    fn test_out_of_phase_signal_produces_no_centre() {
        // L = -R → mid = 0 → centre should be ~0
        let config = UpmixConfig {
            centre_level: 1.0,
            ..Default::default()
        };
        let mut um = StereoUpmixer::new(config, 48_000.0);
        let mut centre_acc = 0.0_f32;
        for i in 0..256 {
            let phase = i as f32 / 100.0;
            let sig = phase.sin() * 0.5;
            let frame = um.process_sample_5_1(sig, -sig);
            centre_acc += frame[2].abs();
        }
        let centre_avg = centre_acc / 256.0;
        assert!(
            centre_avg < 0.05,
            "Out-of-phase signal should produce near-zero centre, got avg={centre_avg:.4}"
        );
    }

    // ── LFE extraction ─────────────────────────────────────────────────────

    #[test]
    fn test_lfe_is_lowpass_of_input() {
        // High frequency signal should produce low LFE output.
        let config = UpmixConfig {
            lfe_level: 1.0,
            lfe_crossover_hz: 80.0,
            ..Default::default()
        };
        let mut um = StereoUpmixer::new(config, 48_000.0);

        // 1 kHz sine (well above 80 Hz crossover)
        let sr = 48_000.0_f32;
        let freq = 1000.0_f32;
        let mut lfe_acc = 0.0_f32;
        for i in 128..256 {
            let s = (2.0 * PI * freq * i as f32 / sr).sin() * 0.8;
            let frame = um.process_sample_5_1(s, s);
            lfe_acc += frame[3] * frame[3]; // channel 3 = LFE
        }
        let lfe_rms = (lfe_acc / 128.0).sqrt();
        assert!(
            lfe_rms < 0.2,
            "1kHz signal should produce low LFE energy, got rms={lfe_rms:.4}"
        );
    }

    #[test]
    fn test_lfe_passes_low_freq() {
        // 20 Hz signal should pass through the LFE crossover filter.
        let config = UpmixConfig {
            lfe_level: 1.0,
            lfe_crossover_hz: 80.0,
            ..Default::default()
        };
        let mut um = StereoUpmixer::new(config, 48_000.0);

        let sr = 48_000.0_f32;
        let freq = 20.0_f32;
        let mut lfe_acc = 0.0_f32;
        // Allow more settling time for very low frequency filters.
        for i in 4800..9600 {
            let s = (2.0 * PI * freq * i as f32 / sr).sin() * 0.8;
            let frame = um.process_sample_5_1(s, s);
            lfe_acc += frame[3] * frame[3];
        }
        let lfe_rms = (lfe_acc / 4800.0).sqrt();
        assert!(
            lfe_rms > 0.1,
            "20 Hz signal should produce significant LFE energy after settling, got rms={lfe_rms:.4}"
        );
    }

    // ── Surround channels ─────────────────────────────────────────────────

    #[test]
    fn test_surround_nonzero_for_stereo_signal() {
        // A pure stereo signal (L ≠ R, out of phase) should produce surround output.
        let mut um = make_upmixer();
        let mut surr_acc = 0.0_f32;
        for i in 0..512 {
            let phase = i as f32 / 100.0;
            let frame = um.process_sample_5_1(phase.sin() * 0.5, phase.cos() * 0.4);
            surr_acc += frame[4] * frame[4] + frame[5] * frame[5];
        }
        let surr_rms = (surr_acc / 1024.0).sqrt();
        assert!(
            surr_rms > 0.01,
            "Stereo signal should produce non-zero surround output, got rms={surr_rms:.4}"
        );
    }

    #[test]
    fn test_7_1_has_decorrelated_rear() {
        // 7.1 rear channels should differ from 5.1 surround channels.
        let mut um = make_upmixer();
        let mut side_sum = 0.0_f32;
        let mut rear_sum = 0.0_f32;
        for i in 0..512 {
            let phase = i as f32 / 100.0;
            let l = phase.sin() * 0.5;
            let r = phase.cos() * 0.4;
            let frame8 = um.process_sample_7_1(l, r);
            side_sum += frame8[4].abs(); // Lss
            rear_sum += frame8[6].abs(); // Lrs
        }
        // Rear and side should not be identical (delay + decorrelation).
        let diff = (side_sum - rear_sum).abs();
        assert!(
            diff > 0.0 || (side_sum == 0.0 && rear_sum == 0.0),
            "7.1 rear and side channels should differ after decorrelation"
        );
    }

    // ── Block processing ──────────────────────────────────────────────────

    #[test]
    fn test_block_5_1_output_size() {
        let mut um = make_upmixer();
        let stereo: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let out = um.process_block_5_1(&stereo);
        // 128 frames × 6 channels = 768
        assert_eq!(out.len(), 128 * 6, "5.1 block output size mismatch");
    }

    #[test]
    fn test_block_7_1_output_size() {
        let mut um = make_upmixer();
        let stereo: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let out = um.process_block_7_1(&stereo);
        // 128 frames × 8 channels = 1024
        assert_eq!(out.len(), 128 * 8, "7.1 block output size mismatch");
    }

    // ── Presets ───────────────────────────────────────────────────────────

    #[test]
    fn test_presets_compile_and_produce_finite_output() {
        for config in [
            UpmixConfig::cinema(),
            UpmixConfig::music(),
            UpmixConfig::broadcast(),
        ] {
            let mut um = StereoUpmixer::new(config, 48_000.0);
            let frame = um.process_sample_5_1(0.3, -0.2);
            for &s in &frame {
                assert!(s.is_finite(), "preset produced non-finite output: {s}");
            }
        }
    }

    // ── Format selector ───────────────────────────────────────────────────

    #[test]
    fn test_fmt_selector_5_1() {
        let mut um = make_upmixer();
        let out = um.process_sample_fmt(0.4, 0.2, UpmixFormat::Surround5_1);
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn test_fmt_selector_7_1() {
        let mut um = make_upmixer();
        let out = um.process_sample_fmt(0.4, 0.2, UpmixFormat::Surround7_1);
        assert_eq!(out.len(), 8);
    }

    // ── Reset ─────────────────────────────────────────────────────────────

    #[test]
    fn test_reset_clears_delay_state() {
        let mut um = make_upmixer();
        // Drive with loud signal to fill delay buffers.
        for _ in 0..1024 {
            um.process_sample_7_1(0.9, -0.9);
        }
        um.reset();
        // After reset, silence should produce near-zero output.
        let frame = um.process_sample_7_1(0.0, 0.0);
        for (ch, &s) in frame.iter().enumerate() {
            assert!(s.abs() < 1e-6, "channel {ch} not silent after reset: {s}");
        }
    }

    // ── Parameter updates ─────────────────────────────────────────────────

    #[test]
    fn test_set_centre_level_affects_output() {
        let mut um_full = StereoUpmixer::new(
            UpmixConfig {
                centre_level: 1.0,
                ..Default::default()
            },
            48_000.0,
        );
        let mut um_none = StereoUpmixer::new(
            UpmixConfig {
                centre_level: 0.0,
                ..Default::default()
            },
            48_000.0,
        );

        // Settle both
        for _ in 0..256 {
            um_full.process_sample_5_1(0.5, 0.5);
            um_none.process_sample_5_1(0.5, 0.5);
        }

        let c_full = um_full.process_sample_5_1(0.5, 0.5)[2];
        let c_none = um_none.process_sample_5_1(0.5, 0.5)[2];
        assert!(
            c_full.abs() > c_none.abs(),
            "centre_level=1 should produce more centre than level=0: full={c_full:.4}, none={c_none:.4}"
        );
    }

    #[test]
    fn test_energy_conservation() {
        // Total output energy should not exceed a reasonable multiple of input
        // energy (upmixing redistributes, not amplifies beyond certain bounds).
        let mut um = make_upmixer();
        let mut in_energy = 0.0_f32;
        let mut out_energy = 0.0_f32;
        let n = 4096;
        for i in 0..n {
            let phase = i as f32 / 96.0;
            let l = phase.sin() * 0.6;
            let r = (phase * 1.1 + 0.3).sin() * 0.5;
            in_energy += l * l + r * r;
            let frame = um.process_sample_5_1(l, r);
            for &s in &frame {
                out_energy += s * s;
            }
        }
        // Allow up to 5x input energy across 6 channels.
        assert!(
            out_energy < in_energy * 5.0 + 1e-3,
            "output energy {out_energy:.4} far exceeds 5× input energy {in_energy:.4}"
        );
    }
}
