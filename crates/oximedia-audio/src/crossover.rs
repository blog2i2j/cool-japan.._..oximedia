//! Crossover filter networks for multi-way speaker systems.
//!
//! Provides Linkwitz-Riley (LR) crossover filters in 2-way and 3-way
//! configurations.  LR crossovers are formed by cascading two 2nd-order
//! Butterworth filters, giving a 4th-order (LR-4) response that sums flat at
//! the crossover frequency with 0 dB and -6 dB points at the crossover.
//!
//! # Topologies
//!
//! | Type | Bands | Crossover points |
//! |------|-------|-----------------|
//! | [`TwoWayCrossover`] | Low, High | 1 frequency |
//! | [`ThreeWayCrossover`] | Low, Mid, High | 2 frequencies |
//!
//! All filters operate on mono `f32` samples.
//!
//! # Example — 2-way crossover
//!
//! ```rust
//! use oximedia_audio::crossover::{TwoWayCrossover, CrossoverBands2};
//!
//! let mut xover = TwoWayCrossover::new(80.0, 48_000.0);
//! let input: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
//! let CrossoverBands2 { low, high } = xover.process_block(&input);
//! assert_eq!(low.len(), 128);
//! assert_eq!(high.len(), 128);
//! ```
//!
//! # Example — 3-way crossover
//!
//! ```rust
//! use oximedia_audio::crossover::{ThreeWayCrossover, CrossoverBands3};
//!
//! let mut xover = ThreeWayCrossover::new(200.0, 3_000.0, 48_000.0);
//! let input: Vec<f32> = (0..64).map(|i| (i as f32 * 0.2).sin()).collect();
//! let CrossoverBands3 { low, mid, high } = xover.process_block(&input);
//! assert_eq!(low.len(), 64);
//! assert_eq!(mid.len(), 64);
//! assert_eq!(high.len(), 64);
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]

use std::f32::consts::PI;

// ── 2nd-order Butterworth biquad ──────────────────────────────────────────────

/// Internal biquad state (direct form I, f32).
#[derive(Debug, Clone, Default)]
struct BiquadState {
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

/// 2nd-order IIR filter coefficients (normalised: a0 = 1).
#[derive(Debug, Clone)]
struct BiquadCoeffs {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

impl BiquadCoeffs {
    /// Compute a 2nd-order Butterworth low-pass filter.
    ///
    /// `fc` — cutoff frequency in Hz, `fs` — sample rate in Hz.
    fn butterworth_lowpass(fc: f32, fs: f32) -> Self {
        let omega = 2.0 * PI * fc / fs;
        let cos_omega = omega.cos();
        let sin_omega = omega.sin();
        // Q = 1/sqrt(2) for 2nd-order Butterworth.
        let q = 1.0 / 2.0_f32.sqrt();
        let alpha = sin_omega / (2.0 * q);

        let a0 = 1.0 + alpha;
        let b0 = (1.0 - cos_omega) / 2.0;
        let b1 = 1.0 - cos_omega;
        let b2 = (1.0 - cos_omega) / 2.0;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Compute a 2nd-order Butterworth high-pass filter.
    fn butterworth_highpass(fc: f32, fs: f32) -> Self {
        let omega = 2.0 * PI * fc / fs;
        let cos_omega = omega.cos();
        let sin_omega = omega.sin();
        let q = 1.0 / 2.0_f32.sqrt();
        let alpha = sin_omega / (2.0 * q);

        let a0 = 1.0 + alpha;
        let b0 = (1.0 + cos_omega) / 2.0;
        let b1 = -(1.0 + cos_omega);
        let b2 = (1.0 + cos_omega) / 2.0;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Process one sample through the filter.
    #[inline]
    fn process(&self, state: &mut BiquadState, x: f32) -> f32 {
        let y = self.b0 * x + self.b1 * state.x1 + self.b2 * state.x2
            - self.a1 * state.y1
            - self.a2 * state.y2;
        state.x2 = state.x1;
        state.x1 = x;
        state.y2 = state.y1;
        state.y1 = y;
        y
    }
}

// ── Linkwitz-Riley 4th-order section ─────────────────────────────────────────

/// A Linkwitz-Riley 4th-order filter: two cascaded 2nd-order Butterworth stages.
///
/// LR-4 low-pass: -24 dB/octave, -6 dB at fc; sums with the matching
/// high-pass to give a flat magnitude response.
#[derive(Debug, Clone)]
struct Lr4Filter {
    coeffs: BiquadCoeffs,
    stage1: BiquadState,
    stage2: BiquadState,
}

impl Lr4Filter {
    /// Create a Linkwitz-Riley 4th-order low-pass filter at `fc` Hz.
    fn lowpass(fc: f32, fs: f32) -> Self {
        Self {
            coeffs: BiquadCoeffs::butterworth_lowpass(fc, fs),
            stage1: BiquadState::default(),
            stage2: BiquadState::default(),
        }
    }

    /// Create a Linkwitz-Riley 4th-order high-pass filter at `fc` Hz.
    fn highpass(fc: f32, fs: f32) -> Self {
        Self {
            coeffs: BiquadCoeffs::butterworth_highpass(fc, fs),
            stage1: BiquadState::default(),
            stage2: BiquadState::default(),
        }
    }

    /// Process one sample — two cascaded biquad stages.
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let y1 = self.coeffs.process(&mut self.stage1, x);
        self.coeffs.process(&mut self.stage2, y1)
    }

    /// Reset the filter state (zero history).
    fn reset(&mut self) {
        self.stage1 = BiquadState::default();
        self.stage2 = BiquadState::default();
    }
}

// ── 2-way crossover ──────────────────────────────────────────────────────────

/// Output bands from a [`TwoWayCrossover`].
#[derive(Debug, Clone)]
pub struct CrossoverBands2 {
    /// Low-frequency band (below the crossover frequency).
    pub low: Vec<f32>,
    /// High-frequency band (above the crossover frequency).
    pub high: Vec<f32>,
}

/// Linkwitz-Riley 4th-order 2-way crossover network.
///
/// Splits the input into a low and a high band at a single crossover frequency.
/// Both bands sum to a flat magnitude response.
#[derive(Debug, Clone)]
pub struct TwoWayCrossover {
    /// Crossover frequency in Hz.
    pub crossover_hz: f32,
    /// Sample rate in Hz.
    pub sample_rate: f32,
    lp: Lr4Filter,
    hp: Lr4Filter,
}

impl TwoWayCrossover {
    /// Create a new 2-way LR-4 crossover.
    ///
    /// `crossover_hz` — crossover frequency (Hz), `sample_rate` — sample rate (Hz).
    #[must_use]
    pub fn new(crossover_hz: f32, sample_rate: f32) -> Self {
        let lp = Lr4Filter::lowpass(crossover_hz, sample_rate);
        let hp = Lr4Filter::highpass(crossover_hz, sample_rate);
        Self { crossover_hz, sample_rate, lp, hp }
    }

    /// Process a single sample, returning `(low, high)`.
    #[must_use]
    pub fn process_sample(&mut self, x: f32) -> (f32, f32) {
        (self.lp.process(x), self.hp.process(x))
    }

    /// Process a block of samples, returning [`CrossoverBands2`].
    #[must_use]
    pub fn process_block(&mut self, input: &[f32]) -> CrossoverBands2 {
        let mut low = Vec::with_capacity(input.len());
        let mut high = Vec::with_capacity(input.len());
        for &x in input {
            let (l, h) = self.process_sample(x);
            low.push(l);
            high.push(h);
        }
        CrossoverBands2 { low, high }
    }

    /// Reset filter state (clear delay-line history).
    pub fn reset(&mut self) {
        self.lp.reset();
        self.hp.reset();
    }
}

// ── 3-way crossover ──────────────────────────────────────────────────────────

/// Output bands from a [`ThreeWayCrossover`].
#[derive(Debug, Clone)]
pub struct CrossoverBands3 {
    /// Low-frequency band (below `crossover_low_hz`).
    pub low: Vec<f32>,
    /// Mid-frequency band (between `crossover_low_hz` and `crossover_high_hz`).
    pub mid: Vec<f32>,
    /// High-frequency band (above `crossover_high_hz`).
    pub high: Vec<f32>,
}

/// Linkwitz-Riley 4th-order 3-way crossover network.
///
/// Topology:
/// ```text
/// input ──┬── LP(f_lo) ──────────────────────────────────► low
///         │
///         ├── HP(f_lo) ──┬── LP(f_hi) ──────────────────► mid
///         │              └── HP(f_hi) ──────────────────► high
/// ```
///
/// The low–mid–high bands sum to a flat magnitude response.
#[derive(Debug, Clone)]
pub struct ThreeWayCrossover {
    /// Lower crossover frequency in Hz.
    pub crossover_low_hz: f32,
    /// Upper crossover frequency in Hz.
    pub crossover_high_hz: f32,
    /// Sample rate in Hz.
    pub sample_rate: f32,
    // Stage 1: split into low vs. rest
    lp1: Lr4Filter,
    hp1: Lr4Filter,
    // Stage 2: split rest into mid vs. high
    lp2: Lr4Filter,
    hp2: Lr4Filter,
}

impl ThreeWayCrossover {
    /// Create a new 3-way LR-4 crossover.
    ///
    /// `crossover_low_hz` must be less than `crossover_high_hz`.
    ///
    /// # Panics
    ///
    /// Panics if `crossover_low_hz >= crossover_high_hz`.
    #[must_use]
    pub fn new(crossover_low_hz: f32, crossover_high_hz: f32, sample_rate: f32) -> Self {
        assert!(
            crossover_low_hz < crossover_high_hz,
            "crossover_low_hz ({crossover_low_hz}) must be < crossover_high_hz ({crossover_high_hz})"
        );
        let lp1 = Lr4Filter::lowpass(crossover_low_hz, sample_rate);
        let hp1 = Lr4Filter::highpass(crossover_low_hz, sample_rate);
        let lp2 = Lr4Filter::lowpass(crossover_high_hz, sample_rate);
        let hp2 = Lr4Filter::highpass(crossover_high_hz, sample_rate);
        Self {
            crossover_low_hz,
            crossover_high_hz,
            sample_rate,
            lp1,
            hp1,
            lp2,
            hp2,
        }
    }

    /// Process a single sample, returning `(low, mid, high)`.
    #[must_use]
    pub fn process_sample(&mut self, x: f32) -> (f32, f32, f32) {
        let low = self.lp1.process(x);
        let rest = self.hp1.process(x);
        let mid = self.lp2.process(rest);
        let high = self.hp2.process(rest);
        (low, mid, high)
    }

    /// Process a block of samples, returning [`CrossoverBands3`].
    #[must_use]
    pub fn process_block(&mut self, input: &[f32]) -> CrossoverBands3 {
        let mut low = Vec::with_capacity(input.len());
        let mut mid = Vec::with_capacity(input.len());
        let mut high = Vec::with_capacity(input.len());
        for &x in input {
            let (l, m, h) = self.process_sample(x);
            low.push(l);
            mid.push(m);
            high.push(h);
        }
        CrossoverBands3 { low, mid, high }
    }

    /// Reset all filter states (clear delay-line history).
    pub fn reset(&mut self) {
        self.lp1.reset();
        self.hp1.reset();
        self.lp2.reset();
        self.hp2.reset();
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f32 = 48_000.0;

    fn gen_sine(freq: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / FS).sin())
            .collect()
    }

    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let mean_sq: f32 = samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32;
        mean_sq.sqrt()
    }

    #[test]
    fn test_two_way_output_lengths() {
        let mut xover = TwoWayCrossover::new(1_000.0, FS);
        let input: Vec<f32> = (0..512).map(|i| i as f32 * 0.001).collect();
        let bands = xover.process_block(&input);
        assert_eq!(bands.low.len(), 512);
        assert_eq!(bands.high.len(), 512);
    }

    #[test]
    fn test_three_way_output_lengths() {
        let mut xover = ThreeWayCrossover::new(400.0, 4_000.0, FS);
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let bands = xover.process_block(&input);
        assert_eq!(bands.low.len(), 256);
        assert_eq!(bands.mid.len(), 256);
        assert_eq!(bands.high.len(), 256);
    }

    #[test]
    fn test_low_pass_attenuates_high_freq() {
        // A 10 kHz sine should be strongly attenuated by an 800 Hz LP filter.
        let mut xover = TwoWayCrossover::new(800.0, FS);
        // Warm up the filter to settle transients.
        let warmup: Vec<f32> = gen_sine(10_000.0, 4096);
        xover.process_block(&warmup);
        xover.reset();
        // Measure the steady-state response.
        let signal = gen_sine(10_000.0, 2048);
        let bands = xover.process_block(&signal);
        let rms_high_in_lp = rms(&bands.low);
        let rms_input = rms(&signal);
        // LR-4 is -24 dB/oct; 10 kHz is about 3.6 octaves above 800 Hz →
        // attenuation ≈ 3.6 * 24 ≈ 86 dB — in practice we just verify
        // meaningful attenuation (> 20 dB, i.e. factor > 10).
        assert!(
            rms_high_in_lp < rms_input * 0.1,
            "LP band should strongly attenuate 10 kHz; got rms={rms_high_in_lp}"
        );
    }

    #[test]
    fn test_high_pass_attenuates_low_freq() {
        // A 50 Hz sine should be strongly attenuated by the HP branch of a
        // 2 kHz crossover.
        let mut xover = TwoWayCrossover::new(2_000.0, FS);
        let signal = gen_sine(50.0, 4096);
        let bands = xover.process_block(&signal);
        let rms_low_in_hp = rms(&bands.high);
        let rms_input = rms(&signal);
        assert!(
            rms_low_in_hp < rms_input * 0.1,
            "HP band should strongly attenuate 50 Hz; got rms={rms_low_in_hp}"
        );
    }

    #[test]
    fn test_two_way_finite_outputs() {
        let mut xover = TwoWayCrossover::new(500.0, FS);
        let signal = gen_sine(1_000.0, 1024);
        let bands = xover.process_block(&signal);
        for &s in bands.low.iter().chain(bands.high.iter()) {
            assert!(s.is_finite(), "output must be finite");
        }
    }

    #[test]
    fn test_three_way_finite_outputs() {
        let mut xover = ThreeWayCrossover::new(200.0, 5_000.0, FS);
        let signal = gen_sine(2_000.0, 512);
        let bands = xover.process_block(&signal);
        for &s in bands.low.iter().chain(bands.mid.iter()).chain(bands.high.iter()) {
            assert!(s.is_finite(), "output must be finite");
        }
    }

    #[test]
    fn test_reset_clears_state() {
        let mut xover = TwoWayCrossover::new(1_000.0, FS);
        // Process some audio to build up filter state.
        let signal1 = gen_sine(500.0, 512);
        xover.process_block(&signal1);

        // Reset and process a unit impulse — should give same result as fresh filter.
        xover.reset();
        let (l1, h1) = xover.process_sample(1.0);

        let mut fresh = TwoWayCrossover::new(1_000.0, FS);
        let (l2, h2) = fresh.process_sample(1.0);

        let diff_l = (l1 - l2).abs();
        let diff_h = (h1 - h2).abs();
        assert!(
            diff_l < 1e-6 && diff_h < 1e-6,
            "reset filter must produce same output as fresh filter; diff_l={diff_l}, diff_h={diff_h}"
        );
    }

    #[test]
    fn test_three_way_mid_band_passes_mid_freq() {
        // A 1 kHz tone with a 200 Hz / 5 kHz 3-way crossover should appear
        // primarily in the mid band.
        let mut xover = ThreeWayCrossover::new(200.0, 5_000.0, FS);
        let signal = gen_sine(1_000.0, 8192);
        let bands = xover.process_block(&signal);

        // Skip first 2048 samples to allow filter to settle.
        let start = 2048;
        let mid_rms = rms(&bands.mid[start..]);
        let low_rms = rms(&bands.low[start..]);
        let high_rms = rms(&bands.high[start..]);

        assert!(
            mid_rms > low_rms,
            "1 kHz should be stronger in mid than low; mid={mid_rms}, low={low_rms}"
        );
        assert!(
            mid_rms > high_rms,
            "1 kHz should be stronger in mid than high; mid={mid_rms}, high={high_rms}"
        );
    }

    #[test]
    fn test_crossover_sample_by_sample_matches_block() {
        let mut xover_block = TwoWayCrossover::new(1_000.0, FS);
        let mut xover_sample = TwoWayCrossover::new(1_000.0, FS);

        let input = gen_sine(440.0, 256);
        let bands = xover_block.process_block(&input);

        for (i, &x) in input.iter().enumerate() {
            let (l, h) = xover_sample.process_sample(x);
            let diff_l = (l - bands.low[i]).abs();
            let diff_h = (h - bands.high[i]).abs();
            assert!(diff_l < 1e-7, "sample-by-sample low mismatch at {i}: {diff_l}");
            assert!(diff_h < 1e-7, "sample-by-sample high mismatch at {i}: {diff_h}");
        }
    }
}
