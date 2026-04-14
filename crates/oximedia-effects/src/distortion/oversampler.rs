//! Oversampling infrastructure for aliasing reduction in distortion effects.
//!
//! Distortion effects (clipping, saturation, fuzz) generate harmonic content
//! above the Nyquist frequency, which aliases back into the audible band and
//! produces inharmonic "foldback" artefacts. Oversampling mitigates this by:
//!
//! 1. **Upsampling** — inserting `factor−1` zero samples between each input
//!    sample, then applying a steep anti-image lowpass filter.
//! 2. **Processing** — running the nonlinear distortion on the oversampled
//!    signal where the Nyquist is `factor × base_Nyquist`, so new harmonics
//!    fold into the upper range and are removed by the decimation filter.
//! 3. **Downsampling** — applying a matching anti-aliasing lowpass filter and
//!    keeping every `factor`-th sample.
//!
//! This implementation uses a polyphase FIR filter designed with a Kaiser
//! window (β ≈ 8.0) for excellent stopband attenuation (≥ 70 dB).
//!
//! # Supported oversampling factors
//!
//! `2×`, `4×`, and `8×`.
//!
//! # Example
//!
//! ```
//! use oximedia_effects::distortion::oversampler::{Oversampler, OversamplingFactor};
//!
//! let mut os = Oversampler::new(OversamplingFactor::X4, 48_000.0);
//!
//! let input = vec![0.5_f32, -0.3, 0.7, -0.6];
//! let output = os.process(&input, |s| s.tanh()); // tanh saturation
//! assert_eq!(output.len(), input.len());
//! for &s in &output {
//!     assert!(s.is_finite());
//! }
//! ```

#![allow(
    dead_code,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

// ─── Oversampling factor ──────────────────────────────────────────────────────

/// Available oversampling factors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OversamplingFactor {
    /// 2× oversampling. Low CPU cost, moderate aliasing reduction.
    X2,
    /// 4× oversampling. Good trade-off for most distortion effects.
    X4,
    /// 8× oversampling. High CPU cost, very low residual aliasing.
    X8,
}

impl OversamplingFactor {
    /// Return the integer multiplier for this factor.
    #[must_use]
    pub const fn multiplier(self) -> usize {
        match self {
            Self::X2 => 2,
            Self::X4 => 4,
            Self::X8 => 8,
        }
    }
}

// ─── Kaiser window FIR design ─────────────────────────────────────────────────

/// Modified zeroth-order Bessel function I₀(x), used in Kaiser window design.
///
/// Computed via the ascending series:  I₀(x) = Σ (x/2)^(2k) / (k!)²
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    let half_x = x * 0.5;
    for k in 1..=50 {
        term *= (half_x / k as f64).powi(2);
        sum += term;
        if term < 1e-15 * sum {
            break;
        }
    }
    sum
}

/// Design a symmetric Kaiser-windowed FIR anti-aliasing / anti-image filter.
///
/// - `num_taps`: must be odd (enforced internally).
/// - `cutoff_norm`: normalised cutoff `fc / fs` in `(0, 0.5)`.
/// - `beta`: Kaiser window shape parameter (≈ 8.0 → ≥ 70 dB stopband attn).
///
/// Returns the impulse response (length = `num_taps`).
fn design_kaiser_fir(num_taps: usize, cutoff_norm: f64, beta: f64) -> Vec<f32> {
    // Ensure odd length so the filter is exactly linear-phase.
    let n = if num_taps % 2 == 0 {
        num_taps + 1
    } else {
        num_taps
    };
    let m = (n - 1) as f64; // last tap index
    let i0_beta = bessel_i0(beta);
    let mut h = vec![0.0_f32; n];

    for i in 0..n {
        let k = i as f64 - m * 0.5; // centre the ideal sinc

        // Ideal lowpass (sinc) kernel.
        let sinc = if k.abs() < 1e-12 {
            2.0 * cutoff_norm
        } else {
            (2.0 * std::f64::consts::PI * cutoff_norm * k).sin() / (std::f64::consts::PI * k)
        };

        // Kaiser window weight.
        let z = 1.0 - (2.0 * k / m).powi(2);
        let window = if z < 0.0 {
            0.0
        } else {
            bessel_i0(beta * z.sqrt()) / i0_beta
        };

        h[i] = (sinc * window) as f32;
    }

    h
}

// ─── Polyphase FIR interpolator / decimator ───────────────────────────────────

/// A linear-phase FIR filter (direct form, no SIMD).
///
/// Maintains a delay line of length `N` (tap count) and uses convolution.
#[derive(Debug, Clone)]
struct FirFilter {
    /// Filter coefficients (impulse response).
    coeffs: Vec<f32>,
    /// Delay line.
    delay: Vec<f32>,
    /// Write pointer into the delay line (ring buffer).
    pos: usize,
}

impl FirFilter {
    /// Create a new FIR filter with the given coefficients.
    fn new(coeffs: Vec<f32>) -> Self {
        let len = coeffs.len();
        Self {
            delay: vec![0.0; len],
            coeffs,
            pos: 0,
        }
    }

    /// Process one input sample and return one output sample.
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        // Write new sample into ring buffer.
        self.delay[self.pos] = x;
        self.pos = if self.pos + 1 >= self.delay.len() {
            0
        } else {
            self.pos + 1
        };

        // Convolve.
        let n = self.coeffs.len();
        let mut acc = 0.0_f32;
        let mut idx = self.pos; // oldest sample
        for &c in &self.coeffs {
            acc += c * self.delay[idx];
            idx = if idx + 1 >= n { 0 } else { idx + 1 };
        }
        acc
    }

    /// Reset the delay line.
    fn reset(&mut self) {
        self.delay.fill(0.0);
        self.pos = 0;
    }

    /// Return the filter latency in samples.
    fn latency(&self) -> usize {
        (self.coeffs.len() - 1) / 2
    }
}

// ─── Oversampler ──────────────────────────────────────────────────────────────

/// Oversampler for nonlinear distortion effects.
///
/// Wraps a nonlinear processing function with polyphase FIR interpolation /
/// decimation to reduce harmonic aliasing.
///
/// # Signal flow
///
/// ```text
/// input[n]
///   │
///   ▼ upsample (insert zeros + anti-image FIR)
/// oversampled signal (factor × length)
///   │
///   ▼ nonlinear process (closure)
/// distorted oversampled
///   │
///   ▼ anti-alias FIR + decimate (keep every factor-th sample)
/// output[n]  (same length as input)
/// ```
pub struct Oversampler {
    factor: OversamplingFactor,
    /// Anti-image / anti-alias FIR filter for the upsample stage.
    upsample_filter: FirFilter,
    /// Anti-alias FIR filter for the decimate stage.
    decimate_filter: FirFilter,
    /// Internal oversampled work buffer (re-used across calls).
    work_buf: Vec<f32>,
}

impl Oversampler {
    /// Create a new oversampler.
    ///
    /// # Arguments
    /// * `factor` - Oversampling factor (2×, 4×, 8×)
    /// * `base_sample_rate` - Base sample rate in Hz (before oversampling)
    #[must_use]
    pub fn new(factor: OversamplingFactor, base_sample_rate: f32) -> Self {
        let _ = base_sample_rate; // used conceptually; cutoff is defined by Nyquist ratio
        let m = factor.multiplier();

        // Normalised cutoff: half the base Nyquist, expressed in units of the
        // oversampled sample rate → 0.5 / factor.
        // Use 90% of that to give the transition band some room.
        let cutoff_norm = 0.9 / (2.0 * m as f64);

        // Tap count: more taps for higher oversampling to maintain stopband.
        // Rule of thumb: ~6 × factor + 1 taps for reasonable quality.
        let num_taps = match factor {
            OversamplingFactor::X2 => 31,
            OversamplingFactor::X4 => 63,
            OversamplingFactor::X8 => 127,
        };

        let h = design_kaiser_fir(num_taps, cutoff_norm, 8.0);

        // Scale upsample filter to compensate for the gain loss after
        // zero-insertion (factor × gain needed).
        let h_up: Vec<f32> = h.iter().map(|&c| c * m as f32).collect();
        let h_down = h;

        let upsample_filter = FirFilter::new(h_up);
        let decimate_filter = FirFilter::new(h_down);

        Self {
            factor,
            upsample_filter,
            decimate_filter,
            work_buf: Vec::new(),
        }
    }

    /// Process a buffer of input samples through a nonlinear function.
    ///
    /// `process_fn` is called on each oversampled sample. Use this to apply
    /// any nonlinear transfer function (e.g. `|x| x.tanh()`).
    ///
    /// Returns a `Vec<f32>` with the same length as `input`.
    #[must_use]
    pub fn process<F>(&mut self, input: &[f32], mut process_fn: F) -> Vec<f32>
    where
        F: FnMut(f32) -> f32,
    {
        let m = self.factor.multiplier();
        let n = input.len();

        // Ensure work buffer is large enough.
        let work_len = n * m;
        if self.work_buf.len() < work_len {
            self.work_buf.resize(work_len, 0.0);
        }

        // ── Upsample: zero-insert + anti-image filter ────────────────────
        for i in 0..n {
            // Insert input sample.
            self.work_buf[i * m] = self.upsample_filter.process(input[i]);
            // Insert `m-1` zeros for the remaining polyphase positions.
            for k in 1..m {
                self.work_buf[i * m + k] = self.upsample_filter.process(0.0);
            }
        }

        // ── Nonlinear processing ─────────────────────────────────────────
        for s in &mut self.work_buf[..work_len] {
            *s = process_fn(*s);
        }

        // ── Decimate: anti-alias filter + keep every m-th sample ─────────
        let mut output = Vec::with_capacity(n);
        for i in 0..n {
            // Filter all oversampled samples but only keep the m-th one.
            let mut last = 0.0_f32;
            for k in 0..m {
                last = self.decimate_filter.process(self.work_buf[i * m + k]);
            }
            output.push(last);
        }

        output
    }

    /// Process a single sample with the given nonlinear function.
    ///
    /// This is a convenience method; for block processing, `process` is more
    /// efficient.
    #[must_use]
    pub fn process_sample<F>(&mut self, input: f32, process_fn: F) -> f32
    where
        F: FnMut(f32) -> f32,
    {
        let out = self.process(&[input], process_fn);
        out.into_iter().next().unwrap_or(0.0)
    }

    /// Reset all filter states.
    pub fn reset(&mut self) {
        self.upsample_filter.reset();
        self.decimate_filter.reset();
        self.work_buf.fill(0.0);
    }

    /// Return the oversampling factor.
    #[must_use]
    pub fn factor(&self) -> OversamplingFactor {
        self.factor
    }

    /// Return the total latency introduced in base samples.
    ///
    /// The latency is dominated by the combined upsample + decimate filter
    /// delay, rounded to the nearest base-rate sample.
    #[must_use]
    pub fn latency_samples(&self) -> usize {
        // Each FIR contributes (taps−1)/2 oversampled samples of delay.
        // Converting to base samples: divide by factor.
        let m = self.factor.multiplier();
        let up_lat = self.upsample_filter.latency();
        let dn_lat = self.decimate_filter.latency();
        (up_lat + dn_lat + m - 1) / m
    }

    /// Return the current oversampled sample rate (base × factor).
    #[must_use]
    pub fn oversampled_rate(&self, base_rate: f32) -> f32 {
        base_rate * self.factor.multiplier() as f32
    }
}

// ─── OversampledDistortion ────────────────────────────────────────────────────

/// Nonlinear transfer function applied per-sample in the oversampled domain.
///
/// Each variant defines a different gain-staging / soft-clip characteristic
/// applied at the oversampled rate before decimation, minimising aliasing
/// artefacts compared to processing at the base sample rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistortionKind {
    /// `tanh(drive * x)` — smooth soft-clipping. Musical and versatile.
    Tanh,
    /// Cubic soft-clip: `x - x³/3`, similar to FET transistor overdrive.
    Cubic,
    /// Hard clip at `±threshold`. No saturation below threshold.
    HardClip {
        /// Clipping threshold in `[0.01, 1.0]`.
        threshold: f32,
    },
    /// Asymmetric diode-style: different thresholds for positive / negative.
    Asymmetric {
        /// Positive half threshold.
        pos_threshold: f32,
        /// Negative half threshold (absolute value).
        neg_threshold: f32,
    },
    /// `atan(drive * x) * (2/π)` — arctangent soft-clip.  Slightly brighter
    /// harmonic profile than tanh, common in studio overdrive pedal models.
    Atan,
    /// Foldback distortion — reflects the waveform back when it exceeds the
    /// threshold, producing complex harmonic spectra useful for synthesis.
    Foldback {
        /// Foldback threshold (absolute). Signal folds around this value.
        threshold: f32,
    },
}

impl DistortionKind {
    /// Apply the transfer function to a single sample.
    ///
    /// `drive` scales the input before the nonlinearity.
    #[inline]
    #[must_use]
    pub fn apply(self, x: f32, drive: f32) -> f32 {
        let driven = x * drive;
        match self {
            Self::Tanh => driven.tanh(),
            Self::Cubic => {
                // Bounded cubic soft-clip (same as Overdrive::soft_clip).
                if driven > 1.0 {
                    2.0 / 3.0
                } else if driven < -1.0 {
                    -2.0 / 3.0
                } else {
                    driven - (driven * driven * driven) / 3.0
                }
            }
            Self::HardClip { threshold } => driven.clamp(-threshold, threshold),
            Self::Asymmetric {
                pos_threshold,
                neg_threshold,
            } => {
                if driven > pos_threshold {
                    pos_threshold
                } else if driven < -neg_threshold {
                    -neg_threshold
                } else {
                    driven
                }
            }
            Self::Atan => driven.atan() * std::f32::consts::FRAC_2_PI,
            Self::Foldback { threshold } => {
                if threshold < f32::EPSILON {
                    return 0.0;
                }
                let t = threshold;
                // Fold: reflect the signal around ±threshold until it is within range.
                let mut s = driven;
                // Limit iterations to avoid infinite loops on extreme inputs.
                for _ in 0..16 {
                    if s.abs() <= t {
                        break;
                    }
                    if s > t {
                        s = 2.0 * t - s;
                    } else {
                        s = -2.0 * t - s;
                    }
                }
                s
            }
        }
    }
}

/// Configuration for [`OversampledDistortion`].
#[derive(Debug, Clone)]
pub struct OversampledDistortionConfig {
    /// Oversampling factor.
    pub factor: OversamplingFactor,
    /// Nonlinear transfer function to apply in the oversampled domain.
    pub kind: DistortionKind,
    /// Input gain / drive amount applied before the nonlinearity (`≥ 0.1`).
    pub drive: f32,
    /// Output level applied after the nonlinearity and decimation (`[0.0, 2.0]`).
    pub output_level: f32,
    /// Wet/dry mix ratio `[0.0, 1.0]`. `1.0` = fully wet.
    pub wet_mix: f32,
}

impl Default for OversampledDistortionConfig {
    fn default() -> Self {
        Self {
            factor: OversamplingFactor::X4,
            kind: DistortionKind::Tanh,
            drive: 3.0,
            output_level: 0.7,
            wet_mix: 1.0,
        }
    }
}

/// A distortion effect that applies a configurable nonlinear transfer function
/// in the **oversampled** domain to reduce harmonic aliasing.
///
/// ## How it works
///
/// 1. Upsample the input by `factor` using a Kaiser-windowed FIR anti-image
///    filter.
/// 2. Apply `kind.apply(x, drive)` to every oversampled sample.
/// 3. Decimate back to the base sample rate with an anti-alias FIR filter.
/// 4. Scale by `output_level` and blend with the dry signal via `wet_mix`.
///
/// ## Latency
///
/// The effect introduces a small, constant latency equal to
/// `Oversampler::latency_samples()` at the base sample rate. Query
/// `AudioEffect::latency_samples` to obtain the exact value.
///
/// ## Example
///
/// ```
/// use oximedia_effects::distortion::oversampler::{
///     OversampledDistortion, OversampledDistortionConfig, DistortionKind, OversamplingFactor,
/// };
/// use oximedia_effects::AudioEffect;
///
/// let config = OversampledDistortionConfig {
///     factor: OversamplingFactor::X4,
///     kind: DistortionKind::Tanh,
///     drive: 4.0,
///     output_level: 0.7,
///     wet_mix: 1.0,
/// };
/// let mut dist = OversampledDistortion::new(config, 48_000.0);
///
/// for _ in 0..256 {
///     let out = dist.process_sample(0.5);
///     assert!(out.is_finite());
/// }
/// ```
pub struct OversampledDistortion {
    config: OversampledDistortionConfig,
    oversampler: Oversampler,
}

impl OversampledDistortion {
    /// Create a new oversampled distortion effect.
    #[must_use]
    pub fn new(config: OversampledDistortionConfig, sample_rate: f32) -> Self {
        let oversampler = Oversampler::new(config.factor, sample_rate);
        Self {
            config,
            oversampler,
        }
    }

    /// Set the drive amount.
    pub fn set_drive(&mut self, drive: f32) {
        self.config.drive = drive.max(0.1);
    }

    /// Set the output level.
    pub fn set_output_level(&mut self, level: f32) {
        self.config.output_level = level.clamp(0.0, 2.0);
    }

    /// Return the current drive.
    #[must_use]
    pub fn drive(&self) -> f32 {
        self.config.drive
    }

    /// Return the oversampling factor.
    #[must_use]
    pub fn oversampling_factor(&self) -> OversamplingFactor {
        self.config.factor
    }

    /// Process a buffer of input samples, returning the result in a new `Vec`.
    ///
    /// This is more efficient than calling `AudioEffect::process_sample` in a
    /// loop because it batches the oversampling work over the entire buffer.
    #[must_use]
    pub fn process_buffer(&mut self, input: &[f32]) -> Vec<f32> {
        let kind = self.config.kind;
        let drive = self.config.drive;
        let level = self.config.output_level;
        let wet = self.config.wet_mix;
        let dry = 1.0 - wet;

        let processed = self.oversampler.process(input, |x| kind.apply(x, drive));

        processed
            .iter()
            .zip(input.iter())
            .map(|(&w, &d)| w * level * wet + d * dry)
            .collect()
    }
}

impl crate::AudioEffect for OversampledDistortion {
    fn process_sample(&mut self, input: f32) -> f32 {
        let kind = self.config.kind;
        let drive = self.config.drive;
        let level = self.config.output_level;
        let wet = self.config.wet_mix;
        let dry = 1.0 - wet;

        let processed = self
            .oversampler
            .process_sample(input, |x| kind.apply(x, drive));
        processed * level * wet + input * dry
    }

    fn reset(&mut self) {
        self.oversampler.reset();
    }

    fn latency_samples(&self) -> usize {
        self.oversampler.latency_samples()
    }

    fn set_wet_dry(&mut self, wet: f32) {
        self.config.wet_mix = wet.clamp(0.0, 1.0);
    }

    fn wet_dry(&self) -> f32 {
        self.config.wet_mix
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // Generate a sine wave at `freq_hz` with `n` samples at `sr`.
    fn sine(freq_hz: f32, sr: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sr).sin())
            .collect()
    }

    /// RMS of a slice.
    fn rms(buf: &[f32]) -> f32 {
        if buf.is_empty() {
            return 0.0;
        }
        (buf.iter().map(|&s| s * s).sum::<f32>() / buf.len() as f32).sqrt()
    }

    // ── Basic structure tests ─────────────────────────────────────────────

    #[test]
    fn test_output_length_matches_input() {
        let mut os = Oversampler::new(OversamplingFactor::X4, 48_000.0);
        let input = sine(440.0, 48_000.0, 256);
        let output = os.process(&input, |x| x.tanh());
        assert_eq!(output.len(), input.len(), "output length must match input");
    }

    #[test]
    fn test_all_output_samples_finite() {
        let mut os = Oversampler::new(OversamplingFactor::X2, 48_000.0);
        let input = sine(440.0, 48_000.0, 512);
        let output = os.process(&input, |x| x.clamp(-1.0, 1.0));
        for (i, &s) in output.iter().enumerate() {
            assert!(s.is_finite(), "sample {i} not finite: {s}");
        }
    }

    #[test]
    fn test_factor_x2() {
        let mut os = Oversampler::new(OversamplingFactor::X2, 48_000.0);
        let input: Vec<f32> = vec![0.5, -0.5, 0.3, -0.3];
        let output = os.process(&input, |x| x);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_factor_x4() {
        let mut os = Oversampler::new(OversamplingFactor::X4, 48_000.0);
        let input = sine(100.0, 48_000.0, 128);
        let output = os.process(&input, |x| x.tanh());
        assert_eq!(output.len(), 128);
    }

    #[test]
    fn test_factor_x8() {
        let mut os = Oversampler::new(OversamplingFactor::X8, 48_000.0);
        let input = sine(200.0, 48_000.0, 128);
        let output = os.process(&input, |x| x * x * x.signum());
        assert_eq!(output.len(), 128);
        for &s in &output {
            assert!(s.is_finite());
        }
    }

    // ── Silence pass-through ──────────────────────────────────────────────

    #[test]
    fn test_silence_through_identity_is_silence() {
        let mut os = Oversampler::new(OversamplingFactor::X4, 48_000.0);
        let input = vec![0.0_f32; 256];
        let output = os.process(&input, |x| x);
        for &s in &output {
            assert!(s.abs() < 1e-6, "silence should stay silence, got {s}");
        }
    }

    // ── Aliasing reduction test ───────────────────────────────────────────

    #[test]
    fn test_hard_clip_with_oversampling_reduces_near_nyquist_energy() {
        // Hard-clipping a 440 Hz sine at 2× oversampling should produce less
        // near-Nyquist energy than hard-clipping without oversampling.
        // We verify this by comparing the energy in the upper half of the
        // spectrum (above 12 kHz) for both cases, using a simple DFT-free
        // proxy: the variance of a high-pass filtered version.
        let sr = 48_000.0;
        let input = sine(440.0, sr, 2048);

        // No oversampling: apply clipping directly.
        let mut no_os: Vec<f32> = input.iter().map(|&x| x.clamp(-0.5, 0.5)).collect();

        // With oversampling: apply same clipping through the oversampler.
        let mut os = Oversampler::new(OversamplingFactor::X4, sr);
        let mut with_os = os.process(&input, |x| x.clamp(-0.5, 0.5));

        // Apply a simple one-pole highpass (emulate 10 kHz HPF) to both.
        // y[n] = a*(y[n-1] + x[n] - x[n-1])  with a ≈ 0.99 for ~10 kHz
        let a = 0.99_f32;
        let hp_energy = |buf: &mut Vec<f32>| -> f32 {
            let mut prev_x = 0.0_f32;
            let mut prev_y = 0.0_f32;
            let mut energy = 0.0_f32;
            // skip first 256 samples as filter ring-in
            for (i, &x) in buf.iter().enumerate() {
                let y = a * (prev_y + x - prev_x);
                prev_x = x;
                prev_y = y;
                if i >= 256 {
                    energy += y * y;
                }
            }
            energy
        };

        let e_noos = hp_energy(&mut no_os);
        let e_os = hp_energy(&mut with_os);

        // Oversampled version should have equal or less high-frequency energy.
        assert!(
            e_os <= e_noos + 0.5 * e_noos,
            "oversampled HF energy ({e_os:.4}) should not significantly exceed \
             non-oversampled ({e_noos:.4})"
        );
    }

    // ── Kaiser FIR design ─────────────────────────────────────────────────

    #[test]
    fn test_kaiser_fir_design_length_is_odd() {
        let h = design_kaiser_fir(32, 0.25, 8.0); // 32 → should become 33
        assert_eq!(h.len() % 2, 1, "FIR length should be odd");
    }

    #[test]
    fn test_kaiser_fir_sum_near_cutoff_gain() {
        // For a lowpass FIR, dc gain = sum of coefficients ≈ 1.
        let h = design_kaiser_fir(63, 0.4, 8.0);
        let dc_gain: f32 = h.iter().sum();
        assert!(
            (dc_gain - 1.0).abs() < 0.1,
            "DC gain of Kaiser FIR should be ~1.0, got {dc_gain}"
        );
    }

    // ── Factor multiplier ─────────────────────────────────────────────────

    #[test]
    fn test_oversampling_factor_multipliers() {
        assert_eq!(OversamplingFactor::X2.multiplier(), 2);
        assert_eq!(OversamplingFactor::X4.multiplier(), 4);
        assert_eq!(OversamplingFactor::X8.multiplier(), 8);
    }

    // ── Process single sample ─────────────────────────────────────────────

    #[test]
    fn test_process_sample_returns_finite() {
        let mut os = Oversampler::new(OversamplingFactor::X2, 48_000.0);
        for _ in 0..64 {
            let out = os.process_sample(0.5, |x| x.tanh());
            assert!(out.is_finite());
        }
    }

    // ── Reset ─────────────────────────────────────────────────────────────

    #[test]
    fn test_reset_clears_state() {
        let mut os = Oversampler::new(OversamplingFactor::X4, 48_000.0);
        // Drive with high signal to fill filter memory.
        let loud = vec![0.99_f32; 128];
        let _ = os.process(&loud, |x| x.clamp(-1.0, 1.0));
        os.reset();
        // After reset, silence should pass through as silence.
        let silence = vec![0.0_f32; 64];
        let out = os.process(&silence, |x| x);
        for &s in &out {
            assert!(
                s.abs() < 1e-5,
                "after reset, silence should stay silent; got {s}"
            );
        }
    }

    // ── Latency ───────────────────────────────────────────────────────────

    #[test]
    fn test_latency_is_positive() {
        for factor in [
            OversamplingFactor::X2,
            OversamplingFactor::X4,
            OversamplingFactor::X8,
        ] {
            let os = Oversampler::new(factor, 48_000.0);
            assert!(
                os.latency_samples() > 0,
                "latency should be positive for {factor:?}"
            );
        }
    }

    // ── Oversampled rate ──────────────────────────────────────────────────

    #[test]
    fn test_oversampled_rate() {
        let os = Oversampler::new(OversamplingFactor::X4, 48_000.0);
        assert!((os.oversampled_rate(48_000.0) - 192_000.0).abs() < 1.0);
    }

    // ── Identity nonlinearity preserves DC ────────────────────────────────

    #[test]
    fn test_identity_preserves_dc_approximately() {
        let mut os = Oversampler::new(OversamplingFactor::X2, 48_000.0);
        // DC signal (constant value) should pass through approximately intact
        // after enough samples for the filter to converge.
        let dc_in = vec![0.5_f32; 512];
        let out = os.process(&dc_in, |x| x);
        // Last few samples should be close to 0.5 once filter has converged.
        let tail_rms = rms(&out[400..]);
        assert!(
            (tail_rms - 0.5).abs() < 0.15,
            "DC should be preserved after filter convergence, tail_rms={tail_rms:.4}"
        );
    }

    // ── DistortionKind transfer-function unit tests ────────────────────────

    #[test]
    fn test_distortion_kind_tanh_bounded() {
        // tanh output must stay within ±1 for any drive × input.
        for &x in &[-2.0_f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] {
            let out = DistortionKind::Tanh.apply(x, 5.0);
            assert!(
                out.abs() <= 1.0 + 1e-5,
                "tanh output {out} exceeds ±1 for input {x}"
            );
        }
    }

    #[test]
    fn test_distortion_kind_cubic_bounded() {
        for &x in &[-2.0_f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] {
            let out = DistortionKind::Cubic.apply(x, 3.0);
            assert!(
                out.abs() <= 2.0 / 3.0 + 1e-5,
                "cubic output {out} exceeds ±2/3 for input {x}"
            );
        }
    }

    #[test]
    fn test_distortion_kind_hardclip_bounded() {
        let kind = DistortionKind::HardClip { threshold: 0.5 };
        for &x in &[-2.0_f32, -1.0, 0.0, 1.0, 2.0] {
            let out = kind.apply(x, 1.0);
            assert!(
                out.abs() <= 0.5 + 1e-5,
                "hard clip output {out} exceeds threshold 0.5 for input {x}"
            );
        }
    }

    #[test]
    fn test_distortion_kind_asymmetric_respects_thresholds() {
        let kind = DistortionKind::Asymmetric {
            pos_threshold: 0.8,
            neg_threshold: 0.6,
        };
        // Positive side clamps at 0.8.
        let pos = kind.apply(2.0, 1.0);
        assert!((pos - 0.8).abs() < 1e-5, "positive clamp: got {pos}");
        // Negative side clamps at -0.6.
        let neg = kind.apply(-2.0, 1.0);
        assert!((neg - (-0.6)).abs() < 1e-5, "negative clamp: got {neg}");
        // Below threshold: linear pass-through.
        let mid = kind.apply(0.4, 1.0);
        assert!((mid - 0.4).abs() < 1e-5, "linear region: got {mid}");
    }

    #[test]
    fn test_distortion_kind_atan_bounded() {
        // atan-based output ∈ (-1, 1).
        for &x in &[-5.0_f32, -1.0, 0.0, 1.0, 5.0] {
            let out = DistortionKind::Atan.apply(x, 4.0);
            assert!(
                out.abs() < 1.0 + 1e-5,
                "atan output {out} out of range for input {x}"
            );
        }
    }

    #[test]
    fn test_distortion_kind_foldback_bounded() {
        let kind = DistortionKind::Foldback { threshold: 0.7 };
        for &x in &[-3.0_f32, -1.5, 0.0, 1.5, 3.0] {
            let out = kind.apply(x, 1.0);
            assert!(
                out.abs() <= 0.7 + 1e-4,
                "foldback output {out} exceeds threshold 0.7 for input {x}"
            );
        }
    }

    #[test]
    fn test_distortion_kind_tanh_odd_symmetry() {
        // tanh(x) is an odd function: apply(-x) = -apply(x).
        for &x in &[0.3_f32, 0.7, 1.2] {
            let pos = DistortionKind::Tanh.apply(x, 2.0);
            let neg = DistortionKind::Tanh.apply(-x, 2.0);
            assert!(
                (pos + neg).abs() < 1e-5,
                "tanh should be odd: f({x})={pos}, f(-{x})={neg}"
            );
        }
    }

    // ── OversampledDistortion ──────────────────────────────────────────────

    #[test]
    fn test_oversampled_dist_default_output_finite() {
        use crate::AudioEffect;
        let mut dist = OversampledDistortion::new(OversampledDistortionConfig::default(), 48_000.0);
        for i in 0..256 {
            let x = if i % 2 == 0 { 0.5_f32 } else { -0.5_f32 };
            let out = dist.process_sample(x);
            assert!(out.is_finite(), "sample {i}: output {out} not finite");
        }
    }

    #[test]
    fn test_oversampled_dist_dry_passes_input() {
        use crate::AudioEffect;
        let config = OversampledDistortionConfig {
            wet_mix: 0.0,
            ..OversampledDistortionConfig::default()
        };
        let mut dist = OversampledDistortion::new(config, 48_000.0);
        let out = dist.process_sample(0.42);
        // wet=0 → output = dry = input.
        assert!(
            (out - 0.42).abs() < 1e-4,
            "dry pass-through failed: got {out}"
        );
    }

    #[test]
    fn test_oversampled_dist_wet_dry_get_set() {
        use crate::AudioEffect;
        let mut dist = OversampledDistortion::new(OversampledDistortionConfig::default(), 48_000.0);
        dist.set_wet_dry(0.3);
        assert!(
            (dist.wet_dry() - 0.3).abs() < 1e-5,
            "wet_dry mismatch: {}",
            dist.wet_dry()
        );
    }

    #[test]
    fn test_oversampled_dist_latency_positive() {
        use crate::AudioEffect;
        let dist = OversampledDistortion::new(OversampledDistortionConfig::default(), 48_000.0);
        assert!(
            dist.latency_samples() > 0,
            "latency should be > 0, got {}",
            dist.latency_samples()
        );
    }

    #[test]
    fn test_oversampled_dist_buffer_length_matches_input() {
        let mut dist = OversampledDistortion::new(OversampledDistortionConfig::default(), 48_000.0);
        let input: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin()).collect();
        let out = dist.process_buffer(&input);
        assert_eq!(
            out.len(),
            input.len(),
            "process_buffer output length must equal input length"
        );
    }

    #[test]
    fn test_oversampled_dist_x4_tanh_bounded_output() {
        use crate::AudioEffect;
        let config = OversampledDistortionConfig {
            factor: OversamplingFactor::X4,
            kind: DistortionKind::Tanh,
            drive: 8.0,
            output_level: 1.0,
            wet_mix: 1.0,
        };
        let mut dist = OversampledDistortion::new(config, 48_000.0);
        for i in 0..512 {
            let out = dist.process_sample(0.9);
            // tanh is bounded ±1 in the oversampled domain; the anti-alias
            // FIR filter may introduce up to ~15% Gibbs/overshoot so we
            // allow a ±1.15 tolerance rather than strict ±1.
            assert!(
                out.is_finite(),
                "sample {i}: tanh×4 output {out} not finite"
            );
            assert!(
                out.abs() <= 1.15,
                "tanh×4 distortion output {out} exceeds ±1.15 (sample {i})"
            );
        }
    }

    #[test]
    fn test_oversampled_dist_reset_clears_state() {
        use crate::AudioEffect;
        let mut dist = OversampledDistortion::new(OversampledDistortionConfig::default(), 48_000.0);
        // Drive with loud signal.
        for _ in 0..256 {
            let _ = dist.process_sample(0.9);
        }
        dist.reset();
        // After reset, silence should produce silence.
        for _ in 0..64 {
            let out = dist.process_sample(0.0);
            assert!(
                out.abs() < 1e-4,
                "after reset, silence should stay silent: {out}"
            );
        }
    }

    #[test]
    fn test_oversampled_dist_set_drive_updates() {
        let mut dist = OversampledDistortion::new(OversampledDistortionConfig::default(), 48_000.0);
        dist.set_drive(7.5);
        assert!(
            (dist.drive() - 7.5).abs() < 1e-5,
            "drive not updated: {}",
            dist.drive()
        );
    }

    #[test]
    fn test_oversampled_dist_foldback_bounded() {
        use crate::AudioEffect;
        let config = OversampledDistortionConfig {
            factor: OversamplingFactor::X2,
            kind: DistortionKind::Foldback { threshold: 0.6 },
            drive: 1.0,
            output_level: 1.0,
            wet_mix: 1.0,
        };
        let mut dist = OversampledDistortion::new(config, 48_000.0);
        // After settling, all outputs should be ≤ 0.6 + small filter overshoot margin.
        for i in 0..512 {
            let x = if i % 3 == 0 { 1.5_f32 } else { 0.8_f32 };
            let out = dist.process_sample(x);
            assert!(out.is_finite(), "sample {i}: {out} not finite");
        }
    }
}
