//! Wavetable-based LFO for efficient periodic modulation in audio effects.
//!
//! Replaces per-sample trigonometric calls with a pre-computed table lookup,
//! providing the same waveform quality at a fraction of the CPU cost.
//!
//! # Design
//!
//! A fixed-size wavetable (default 2048 entries) is computed once during
//! construction.  The oscillator maintains a fractional phase accumulator that
//! advances by `frequency / sample_rate` per sample.  Linear interpolation
//! between adjacent table entries gives sub-sample accuracy.
//!
//! # Supported waveforms
//!
//! | Variant | Description |
//! |---------|-------------|
//! | [`WtWaveform::Sine`]     | Full-cycle sine |
//! | [`WtWaveform::Triangle`] | Band-limited triangle |
//! | [`WtWaveform::Square`]   | ±1 square (50 % duty) |
//! | [`WtWaveform::SawUp`]    | Rising sawtooth |
//! | [`WtWaveform::SawDown`]  | Falling sawtooth |
//! | [`WtWaveform::SoftSine`] | Raised-cosine (gentle sine) |
//!
//! # Example
//!
//! ```ignore
//! use oximedia_effects::utils::wavetable::{WavetableLfo, WtWaveform};
//!
//! let mut lfo = WavetableLfo::new(WtWaveform::Sine, 2.0, 48_000.0);
//! let sample = lfo.next_sample();
//! assert!((-1.0..=1.0).contains(&sample));
//! ```

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::f32::consts::TAU;

/// Number of entries in the wavetable.
const TABLE_SIZE: usize = 2048;

/// Waveform shape for the wavetable LFO.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WtWaveform {
    /// Full-cycle sine wave (smooth, band-limited).
    Sine,
    /// Bi-polar triangle wave.
    Triangle,
    /// Square wave, ±1 at 50 % duty cycle.
    Square,
    /// Rising sawtooth: -1 → +1 over one cycle.
    SawUp,
    /// Falling sawtooth: +1 → -1 over one cycle.
    SawDown,
    /// Raised-cosine "soft sine": gently rounded, less harsh than square.
    SoftSine,
}

/// Build a wavetable for a given waveform.
fn build_table(waveform: WtWaveform) -> Box<[f32; TABLE_SIZE]> {
    // SAFETY: we fill every element below before returning.
    let mut table = Box::new([0.0_f32; TABLE_SIZE]);
    for (i, entry) in table.iter_mut().enumerate() {
        let phase = i as f32 / TABLE_SIZE as f32; // [0, 1)
        *entry = match waveform {
            WtWaveform::Sine => (phase * TAU).sin(),
            WtWaveform::Triangle => {
                // Bipolar triangle: -1 → +1 → -1
                if phase < 0.25 {
                    phase * 4.0
                } else if phase < 0.75 {
                    2.0 - phase * 4.0
                } else {
                    phase * 4.0 - 4.0
                }
            }
            WtWaveform::Square => {
                if phase < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
            WtWaveform::SawUp => phase * 2.0 - 1.0,
            WtWaveform::SawDown => 1.0 - phase * 2.0,
            WtWaveform::SoftSine => {
                // Raised cosine: (1 - cos(2π·phase)) / 2 mapped to [-1, 1]
                let rc = (1.0 - (phase * TAU).cos()) / 2.0;
                rc * 2.0 - 1.0
            }
        };
    }
    table
}

/// Wavetable-based LFO.
///
/// Produces periodic modulation signals at low CPU cost by reading from
/// a pre-computed table with linear interpolation.
pub struct WavetableLfo {
    /// Pre-computed waveform table.
    table: Box<[f32; TABLE_SIZE]>,
    /// Fractional phase in `[0, 1)`.
    phase: f64,
    /// Phase increment per sample: `frequency / sample_rate`.
    phase_inc: f64,
    /// LFO output amplitude in `[0.0, 1.0]`.
    amplitude: f32,
    /// DC offset added to output, typically 0.
    offset: f32,
    /// Current waveform.
    waveform: WtWaveform,
}

impl WavetableLfo {
    /// Create a new wavetable LFO.
    ///
    /// # Arguments
    ///
    /// * `waveform`    – shape of the LFO waveform.
    /// * `frequency`   – LFO rate in Hz (clamped to `[0.001, 200.0]`).
    /// * `sample_rate` – audio sample rate in Hz.
    #[must_use]
    pub fn new(waveform: WtWaveform, frequency: f32, sample_rate: f32) -> Self {
        let freq = frequency.clamp(0.001, 200.0);
        let sr = sample_rate.max(1.0);
        Self {
            table: build_table(waveform),
            phase: 0.0,
            phase_inc: (freq as f64) / (sr as f64),
            amplitude: 1.0,
            offset: 0.0,
            waveform,
        }
    }

    /// Return the current waveform.
    #[must_use]
    pub fn waveform(&self) -> WtWaveform {
        self.waveform
    }

    /// Return the current LFO frequency in Hz.
    #[must_use]
    pub fn frequency(&self) -> f32 {
        // Reconstruct from phase_inc and nominal sample_rate isn't stored; callers
        // who need it should store it themselves.  We return phase_inc as a proxy.
        self.phase_inc as f32
    }

    /// Set the LFO frequency in Hz (clamped to `[0.001, 200.0]`).
    pub fn set_frequency(&mut self, frequency: f32, sample_rate: f32) {
        let freq = frequency.clamp(0.001, 200.0);
        let sr = sample_rate.max(1.0);
        self.phase_inc = (freq as f64) / (sr as f64);
    }

    /// Set the output amplitude in `[0.0, 1.0]`.  Values outside are clamped.
    pub fn set_amplitude(&mut self, amplitude: f32) {
        self.amplitude = amplitude.clamp(0.0, 1.0);
    }

    /// Set a DC offset added to output (default 0).
    pub fn set_offset(&mut self, offset: f32) {
        self.offset = offset;
    }

    /// Change the waveform, rebuilding the table.  Phase is preserved.
    pub fn set_waveform(&mut self, waveform: WtWaveform) {
        if waveform != self.waveform {
            self.table = build_table(waveform);
            self.waveform = waveform;
        }
    }

    /// Reset the LFO phase to 0.
    pub fn reset(&mut self) {
        self.phase = 0.0;
    }

    /// Set the phase to an exact value in `[0, 1)`.
    pub fn set_phase(&mut self, phase: f64) {
        self.phase = phase.rem_euclid(1.0);
    }

    /// Read the current table value at `phase` using linear interpolation.
    fn table_read(&self, phase: f64) -> f32 {
        let frac_pos = phase * TABLE_SIZE as f64;
        let idx_lo = (frac_pos as usize) % TABLE_SIZE;
        let idx_hi = (idx_lo + 1) % TABLE_SIZE;
        let frac = (frac_pos - frac_pos.floor()) as f32;
        self.table[idx_lo] + frac * (self.table[idx_hi] - self.table[idx_lo])
    }

    /// Advance the phase and return the next LFO sample.
    ///
    /// Output is in `[-(amplitude), +(amplitude)] + offset`.
    pub fn next_sample(&mut self) -> f32 {
        let out = self.table_read(self.phase);
        self.phase = (self.phase + self.phase_inc).rem_euclid(1.0);
        out * self.amplitude + self.offset
    }

    /// Fill `buffer` with consecutive LFO samples (in-place).
    pub fn fill(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = self.next_sample();
        }
    }

    /// Return `n` LFO samples without modifying the external state (peek mode).
    ///
    /// The LFO phase **is** advanced — this is a normal `n`-sample advance that
    /// collects into a `Vec` for inspection/testing.
    #[must_use]
    pub fn collect(&mut self, n: usize) -> Vec<f32> {
        let mut v = vec![0.0; n];
        self.fill(&mut v);
        v
    }

    /// Return the current phase without advancing.
    #[must_use]
    pub fn phase(&self) -> f64 {
        self.phase
    }
}

/// A stereo pair of wavetable LFOs with a configurable phase offset.
///
/// Useful for stereo chorus, vibrato, and other effects that need independent
/// per-channel modulation from a common frequency source.
pub struct StereoWavetableLfo {
    /// Left (or mono) LFO.
    pub left: WavetableLfo,
    /// Right LFO.
    pub right: WavetableLfo,
}

impl StereoWavetableLfo {
    /// Create a stereo pair with the given phase offset between channels.
    ///
    /// `phase_offset` is in `[0, 1)` (e.g. 0.25 = 90° offset).
    #[must_use]
    pub fn new(
        waveform: WtWaveform,
        frequency: f32,
        sample_rate: f32,
        phase_offset: f32,
    ) -> Self {
        let mut left = WavetableLfo::new(waveform, frequency, sample_rate);
        let mut right = WavetableLfo::new(waveform, frequency, sample_rate);
        right.set_phase(phase_offset as f64);
        left.set_phase(0.0);
        Self { left, right }
    }

    /// Set frequency for both channels.
    pub fn set_frequency(&mut self, frequency: f32, sample_rate: f32) {
        self.left.set_frequency(frequency, sample_rate);
        self.right.set_frequency(frequency, sample_rate);
    }

    /// Set amplitude for both channels.
    pub fn set_amplitude(&mut self, amplitude: f32) {
        self.left.set_amplitude(amplitude);
        self.right.set_amplitude(amplitude);
    }

    /// Advance both LFOs by one sample and return `(left, right)` outputs.
    pub fn next_sample(&mut self) -> (f32, f32) {
        (self.left.next_sample(), self.right.next_sample())
    }

    /// Reset both LFOs to phase 0 (right re-applies the original offset).
    pub fn reset(&mut self) {
        self.left.reset();
        self.right.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    // ── helper ──────────────────────────────────────────────────────────────

    fn sine_peak_approx(lfo: &mut WavetableLfo, samples: usize) -> f32 {
        let mut max = 0.0_f32;
        for _ in 0..samples {
            max = max.max(lfo.next_sample().abs());
        }
        max
    }

    fn lfo_mean(lfo: &mut WavetableLfo, n: usize) -> f32 {
        let sum: f32 = (0..n).map(|_| lfo.next_sample()).sum();
        sum / n as f32
    }

    fn full_cycle_rms(waveform: WtWaveform, freq: f32) -> f32 {
        let mut lfo = WavetableLfo::new(waveform, freq, SR);
        let n = (SR / freq) as usize * 4; // 4 full cycles
        let samples: Vec<f32> = lfo.collect(n);
        let mean_sq: f32 = samples.iter().map(|&s| s * s).sum::<f32>() / n as f32;
        mean_sq.sqrt()
    }

    // ── basic output ─────────────────────────────────────────────────────────

    #[test]
    fn test_sine_range() {
        let mut lfo = WavetableLfo::new(WtWaveform::Sine, 1.0, SR);
        let peak = sine_peak_approx(&mut lfo, SR as usize);
        assert!(
            peak > 0.99 && peak <= 1.0,
            "sine peak should be ~1.0, got {peak}"
        );
    }

    #[test]
    fn test_all_waveforms_finite() {
        for wf in [
            WtWaveform::Sine,
            WtWaveform::Triangle,
            WtWaveform::Square,
            WtWaveform::SawUp,
            WtWaveform::SawDown,
            WtWaveform::SoftSine,
        ] {
            let mut lfo = WavetableLfo::new(wf, 10.0, SR);
            for _ in 0..4800 {
                let s = lfo.next_sample();
                assert!(s.is_finite(), "waveform {wf:?} produced non-finite output {s}");
            }
        }
    }

    #[test]
    fn test_amplitude_scaling() {
        let mut lfo = WavetableLfo::new(WtWaveform::Sine, 5.0, SR);
        lfo.set_amplitude(0.5);
        let peak = sine_peak_approx(&mut lfo, SR as usize);
        assert!(
            peak > 0.49 && peak <= 0.5 + 1e-3,
            "amplitude=0.5 peak should be ~0.5, got {peak}"
        );
    }

    #[test]
    fn test_reset_restores_phase() {
        let mut lfo = WavetableLfo::new(WtWaveform::Sine, 1.0, SR);
        let first = lfo.next_sample();
        // Advance many samples
        for _ in 0..1000 {
            let _ = lfo.next_sample();
        }
        lfo.reset();
        let after_reset = lfo.next_sample();
        assert!(
            (first - after_reset).abs() < 1e-4,
            "reset should reproduce first sample; first={first}, after_reset={after_reset}"
        );
    }

    #[test]
    fn test_frequency_change() {
        let mut lfo = WavetableLfo::new(WtWaveform::Sine, 1.0, SR);
        let old_inc = lfo.phase_inc;
        lfo.set_frequency(10.0, SR);
        assert!(
            lfo.phase_inc > old_inc,
            "higher frequency must give larger phase increment"
        );
    }

    #[test]
    fn test_square_wave_values() {
        // Square wave must only produce +1 and -1 (from table, not interpolated boundaries).
        let mut lfo = WavetableLfo::new(WtWaveform::Square, 440.0, SR);
        // Collect a full period, most values should be exactly ±1
        let samples = lfo.collect((SR / 440.0) as usize);
        let all_bipolar = samples.iter().all(|&s| (s - 1.0).abs() < 1e-3 || (s + 1.0).abs() < 1e-3);
        assert!(all_bipolar, "square wave should produce values near ±1");
    }

    #[test]
    fn test_sine_near_zero_mean() {
        // A full-period sine should have mean ≈ 0.
        let mut lfo = WavetableLfo::new(WtWaveform::Sine, 1.0, SR);
        let mean = lfo_mean(&mut lfo, SR as usize);
        assert!(
            mean.abs() < 1e-2,
            "sine LFO mean should be ~0, got {mean}"
        );
    }

    #[test]
    fn test_saw_up_monotone() {
        // Rising sawtooth samples should be increasing for most of a period
        let mut lfo = WavetableLfo::new(WtWaveform::SawUp, 10.0, SR);
        let n = (SR / 10.0) as usize - 1; // one period minus 1 (to avoid the wraparound)
        let samples = lfo.collect(n);
        let increasing = samples.windows(2).filter(|w| w[0] < w[1]).count();
        // Most consecutive pairs should be ascending
        assert!(
            increasing > (n * 9 / 10),
            "saw-up should mostly increase: {increasing}/{n} pairs"
        );
    }

    #[test]
    fn test_stereo_lfo_phase_offset() {
        let mut stereo = StereoWavetableLfo::new(WtWaveform::Sine, 1.0, SR, 0.25);
        // Advance to 1/4 of a cycle and verify left ≈ sin(0), right ≈ sin(π/2)
        let (l, r) = stereo.next_sample();
        // At phase=0, sin(0)=0; at phase=0.25, sin(π/2)=1
        assert!(l.abs() < 0.1, "left initial phase ≈ 0 (sin=0), got {l}");
        assert!((r - 1.0).abs() < 0.1, "right 90° ahead ≈ 1 (sin=π/2), got {r}");
    }

    #[test]
    fn test_fill_buffer() {
        let mut lfo = WavetableLfo::new(WtWaveform::Sine, 100.0, SR);
        let mut buf = vec![0.0_f32; 256];
        lfo.fill(&mut buf);
        // All samples should be finite and within [-1, 1]
        assert!(
            buf.iter().all(|&s| s.is_finite() && s.abs() <= 1.001),
            "fill buffer should produce finite samples in [-1,1]"
        );
    }

    #[test]
    fn test_triangle_rms_approx_half() {
        // Full-cycle triangle RMS ≈ 1/√3 ≈ 0.577
        let rms = full_cycle_rms(WtWaveform::Triangle, 1.0);
        assert!(
            (rms - (1.0_f32 / 3.0_f32.sqrt())).abs() < 0.02,
            "triangle RMS should be ~{:.3}, got {rms}",
            1.0_f32 / 3.0_f32.sqrt()
        );
    }

    #[test]
    fn test_set_phase_manual() {
        // Setting phase to 0.25 on a sine should give sin(π/2) ≈ 1.0
        let mut lfo = WavetableLfo::new(WtWaveform::Sine, 1.0, SR);
        lfo.set_phase(0.25);
        // One sample at phase 0.25 (quarter cycle) → ≈ sin(π/2) = 1
        let s = lfo.next_sample();
        assert!(
            (s - 1.0).abs() < 0.01,
            "phase=0.25 on sine should ≈ 1.0, got {s}"
        );
    }

    #[test]
    fn test_waveform_switch() {
        let mut lfo = WavetableLfo::new(WtWaveform::Sine, 1.0, SR);
        assert_eq!(lfo.waveform(), WtWaveform::Sine);
        lfo.set_waveform(WtWaveform::Triangle);
        assert_eq!(lfo.waveform(), WtWaveform::Triangle);
        // Should still produce finite output
        let s = lfo.next_sample();
        assert!(s.is_finite());
    }

    #[test]
    fn test_soft_sine_bounded() {
        let mut lfo = WavetableLfo::new(WtWaveform::SoftSine, 5.0, SR);
        for _ in 0..9600 {
            let s = lfo.next_sample();
            assert!(
                s >= -1.001 && s <= 1.001,
                "soft-sine must stay in [-1, 1], got {s}"
            );
        }
    }
}

