//! 5-band parametric equaliser with RBJ biquad (Audio EQ Cookbook) formulas.
//!
//! This module provides a richer EQ API than `eq::mod` (which stores pairs of
//! `(EqBand, BiquadFilter)`).  Key differences:
//!
//! - Bands carry an `enabled` flag so they can be bypassed without removal.
//! - Per-band type is expressed as [`EqBandType`] (a superset of `BandType`).
//! - [`BiquadState`] exposes Direct Form I state explicitly, making it easy to
//!   serialise / reset individual filter histories.
//! - [`ParametricEq`] stores `sample_rate` so callers do not repeat it on every
//!   `add_band` call.

#![allow(clippy::cast_precision_loss)]

use std::f32::consts::PI;

/// Band filter type for the parametric EQ.
#[derive(Debug, Clone, PartialEq)]
pub enum EqBandType {
    /// Gentle low-frequency shelving boost or cut.
    LowShelf,
    /// Gentle high-frequency shelving boost or cut.
    HighShelf,
    /// Bell (peak) filter — symmetric boost or cut around a centre frequency.
    Peaking,
    /// Narrow band-reject (notch) filter.
    Notch,
    /// Second-order low-pass (Butterworth) filter.
    LowPass,
    /// Second-order high-pass (Butterworth) filter.
    HighPass,
    /// All-pass filter (unity magnitude, phase shift only).
    AllPass,
}

/// A single parametric EQ band definition.
#[derive(Debug, Clone)]
pub struct EqBand {
    /// Filter shape / type.
    pub band_type: EqBandType,
    /// Centre or cutoff frequency in Hz.
    pub frequency_hz: f32,
    /// Gain in dB (positive = boost, negative = cut).
    /// Relevant for [`EqBandType::Peaking`], [`EqBandType::LowShelf`],
    /// and [`EqBandType::HighShelf`].
    pub gain_db: f32,
    /// Quality factor controlling bandwidth.
    pub q: f32,
    /// Whether this band is active.  A disabled band passes the signal unchanged.
    pub enabled: bool,
}

impl EqBand {
    // ──────────────────────────────────────────── constructors ────────────────

    /// Bell / peak filter with the given centre frequency, gain and Q.
    #[must_use]
    pub fn peaking(freq_hz: f32, gain_db: f32, q: f32) -> Self {
        Self {
            band_type: EqBandType::Peaking,
            frequency_hz: freq_hz,
            gain_db,
            q,
            enabled: true,
        }
    }

    /// Low-shelf filter (Butterworth Q ≈ 0.707).
    #[must_use]
    pub fn low_shelf(freq_hz: f32, gain_db: f32) -> Self {
        Self {
            band_type: EqBandType::LowShelf,
            frequency_hz: freq_hz,
            gain_db,
            q: 0.707,
            enabled: true,
        }
    }

    /// High-shelf filter (Butterworth Q ≈ 0.707).
    #[must_use]
    pub fn high_shelf(freq_hz: f32, gain_db: f32) -> Self {
        Self {
            band_type: EqBandType::HighShelf,
            frequency_hz: freq_hz,
            gain_db,
            q: 0.707,
            enabled: true,
        }
    }

    /// Band-reject (notch) filter.
    #[must_use]
    pub fn notch(freq_hz: f32, q: f32) -> Self {
        Self {
            band_type: EqBandType::Notch,
            frequency_hz: freq_hz,
            gain_db: 0.0,
            q,
            enabled: true,
        }
    }

    /// Second-order low-pass filter.
    #[must_use]
    pub fn low_pass(freq_hz: f32, q: f32) -> Self {
        Self {
            band_type: EqBandType::LowPass,
            frequency_hz: freq_hz,
            gain_db: 0.0,
            q,
            enabled: true,
        }
    }

    /// Second-order high-pass filter.
    #[must_use]
    pub fn high_pass(freq_hz: f32, q: f32) -> Self {
        Self {
            band_type: EqBandType::HighPass,
            frequency_hz: freq_hz,
            gain_db: 0.0,
            q,
            enabled: true,
        }
    }

    /// Create a peaking (bell) EQ band — convenience alias for [`Self::peaking`].
    ///
    /// Matches the interface: `ParametricEqBand::new(freq, gain_db, q)`.
    #[must_use]
    pub fn new(freq_hz: f32, gain_db: f32, q: f32) -> Self {
        Self::peaking(freq_hz, gain_db, q)
    }

    /// Apply this single EQ band to a buffer of samples, returning processed output.
    ///
    /// Creates a temporary [`BiquadState`] internally so this is stateless across
    /// calls — useful for offline processing or testing.  For real-time use,
    /// prefer [`ParametricEq`] which preserves filter state between calls.
    ///
    /// # Arguments
    ///
    /// * `samples` — input buffer.
    /// * `sample_rate` — audio sample rate in Hz (must be > 0).
    #[must_use]
    pub fn apply(&self, samples: &[f32], sample_rate: u32) -> Vec<f32> {
        let sr = sample_rate as f32;
        let coeffs = self.compute_biquad(sr);
        let mut state = BiquadState::default();
        if !self.enabled {
            return samples.to_vec();
        }
        samples
            .iter()
            .map(|&s| state.process_sample(s, &coeffs))
            .collect()
    }

    // ──────────────────────────────────────── coefficient computation ─────────

    /// Compute RBJ biquad coefficients `[b0, b1, b2, a0, a1, a2]`.
    ///
    /// Uses the Audio EQ Cookbook formulas by Robert Bristow-Johnson.
    /// The returned array stores the *un-normalised* coefficients so that
    /// callers (e.g. [`BiquadState`]) can normalise by `a0` on each call.
    #[must_use]
    pub fn compute_biquad(&self, sample_rate: f32) -> [f32; 6] {
        let w0 = 2.0 * PI * self.frequency_hz / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * self.q.max(f32::EPSILON));

        match self.band_type {
            EqBandType::Peaking => {
                let a = 10.0_f32.powf(self.gain_db / 40.0);
                let b0 = 1.0 + alpha * a;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0 - alpha * a;
                let a0 = 1.0 + alpha / a;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha / a;
                [b0, b1, b2, a0, a1, a2]
            }
            EqBandType::LowShelf => {
                let a = 10.0_f32.powf(self.gain_db / 40.0);
                let sqrt_a = a.sqrt();
                let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha);
                let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
                let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha);
                let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha;
                let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
                let a2 = (a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha;
                [b0, b1, b2, a0, a1, a2]
            }
            EqBandType::HighShelf => {
                let a = 10.0_f32.powf(self.gain_db / 40.0);
                let sqrt_a = a.sqrt();
                let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha);
                let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
                let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha);
                let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha;
                let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
                let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha;
                [b0, b1, b2, a0, a1, a2]
            }
            EqBandType::Notch => {
                let b0 = 1.0;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                [b0, b1, b2, a0, a1, a2]
            }
            EqBandType::LowPass => {
                let b0 = (1.0 - cos_w0) / 2.0;
                let b1 = 1.0 - cos_w0;
                let b2 = (1.0 - cos_w0) / 2.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                [b0, b1, b2, a0, a1, a2]
            }
            EqBandType::HighPass => {
                let b0 = (1.0 + cos_w0) / 2.0;
                let b1 = -(1.0 + cos_w0);
                let b2 = (1.0 + cos_w0) / 2.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                [b0, b1, b2, a0, a1, a2]
            }
            EqBandType::AllPass => {
                let b0 = 1.0 - alpha;
                let b1 = -2.0 * cos_w0;
                let b2 = 1.0 + alpha;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_w0;
                let a2 = 1.0 - alpha;
                [b0, b1, b2, a0, a1, a2]
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════ BiquadState ═════

/// Direct Form I biquad filter state.
///
/// Implements the difference equation:
/// `y[n] = (b0/a0)·x[n] + (b1/a0)·x[n-1] + (b2/a0)·x[n-2]
///         - (a1/a0)·y[n-1] - (a2/a0)·y[n-2]`
#[derive(Debug, Clone, Default)]
pub struct BiquadState {
    /// Previous input sample x[n-1].
    x1: f32,
    /// Two-samples-ago input x[n-2].
    x2: f32,
    /// Previous output sample y[n-1].
    y1: f32,
    /// Two-samples-ago output y[n-2].
    y2: f32,
}

impl BiquadState {
    /// Process one sample using the provided coefficient array `[b0,b1,b2,a0,a1,a2]`.
    #[inline]
    pub fn process_sample(&mut self, x: f32, coeffs: &[f32; 6]) -> f32 {
        let [b0, b1, b2, a0, a1, a2] = *coeffs;
        // Guard against degenerate a0 (should never happen for valid biquads)
        let a0_safe = if a0.abs() < f32::EPSILON { 1.0 } else { a0 };
        let y = (b0 / a0_safe) * x
            + (b1 / a0_safe) * self.x1
            + (b2 / a0_safe) * self.x2
            - (a1 / a0_safe) * self.y1
            - (a2 / a0_safe) * self.y2;

        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }

    /// Reset all internal state to zero.
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

// ═══════════════════════════════════════════════════════ ParametricEq ════════

/// Multi-band parametric equaliser (Direct Form I, RBJ biquad).
///
/// Bands are processed in order.  Disabled bands pass the signal unchanged.
///
/// # Example
/// ```
/// use oximedia_effects::eq::parametric::{ParametricEq, EqBand};
///
/// let mut eq = ParametricEq::new(48_000.0)
///     .with_band(EqBand::peaking(1000.0, 6.0, 1.0))
///     .with_band(EqBand::high_shelf(8000.0, -3.0));
///
/// let output = eq.process_buffer(&vec![0.5_f32; 256]);
/// assert_eq!(output.len(), 256);
/// ```
pub struct ParametricEq {
    /// Active band definitions.
    pub bands: Vec<EqBand>,
    /// Per-band filter state (parallel to `bands`).
    states: Vec<BiquadState>,
    /// Sample rate used to compute biquad coefficients.
    pub sample_rate: f32,
}

impl ParametricEq {
    /// Create a new empty parametric EQ.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self {
            bands: Vec::new(),
            states: Vec::new(),
            sample_rate,
        }
    }

    /// Builder helper — append a band and return `self`.
    #[must_use]
    pub fn with_band(mut self, band: EqBand) -> Self {
        self.add_band(band);
        self
    }

    /// Append a band to the equaliser.
    pub fn add_band(&mut self, band: EqBand) {
        self.bands.push(band);
        self.states.push(BiquadState::default());
    }

    /// Set the gain (in dB) for an existing band by index.
    ///
    /// Returns `Err` if `index` is out of range.
    pub fn set_band_gain(&mut self, index: usize, gain_db: f32) -> Result<(), String> {
        if let Some(band) = self.bands.get_mut(index) {
            band.gain_db = gain_db;
            Ok(())
        } else {
            Err(format!(
                "Band index {index} out of range (have {} bands)",
                self.bands.len()
            ))
        }
    }

    /// Enable or disable a band by index.
    ///
    /// Returns `Err` if `index` is out of range.
    pub fn set_band_enabled(&mut self, index: usize, enabled: bool) -> Result<(), String> {
        if let Some(band) = self.bands.get_mut(index) {
            band.enabled = enabled;
            Ok(())
        } else {
            Err(format!(
                "Band index {index} out of range (have {} bands)",
                self.bands.len()
            ))
        }
    }

    /// Process a single mono sample through all enabled bands in sequence.
    pub fn process_sample(&mut self, mut sample: f32) -> f32 {
        for (band, state) in self.bands.iter().zip(self.states.iter_mut()) {
            if band.enabled {
                let coeffs = band.compute_biquad(self.sample_rate);
                sample = state.process_sample(sample, &coeffs);
            }
        }
        sample
    }

    /// Process a buffer of mono samples, returning a new `Vec<f32>`.
    #[must_use]
    pub fn process_buffer(&mut self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&s| self.process_sample(s)).collect()
    }

    /// Reset all filter states.
    pub fn reset(&mut self) {
        for state in &mut self.states {
            state.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    fn make_sine(freq_hz: f32, sr: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sr).sin())
            .collect()
    }

    fn rms(buf: &[f32]) -> f32 {
        (buf.iter().map(|&s| s * s).sum::<f32>() / buf.len() as f32).sqrt()
    }

    // ── basic sanity ──────────────────────────────────────────────────────────

    #[test]
    fn test_flat_eq_passes_signal_unchanged() {
        // A peaking band with gain=0 dB is a unity filter: output == input.
        let mut eq = ParametricEq::new(SR).with_band(EqBand::peaking(1000.0, 0.0, 1.0));
        let input: Vec<f32> = make_sine(440.0, SR, 512);
        let output = eq.process_buffer(&input);
        for (i, (&a, &b)) in input.iter().zip(output.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "Flat EQ should be unity, sample {i}: in={a}, out={b}"
            );
        }
    }

    #[test]
    fn test_peaking_band_boosts_at_center_freq() {
        // Settle the filter first, then measure RMS at centre frequency.
        let mut eq = ParametricEq::new(SR).with_band(EqBand::peaking(1000.0, 12.0, 1.0));
        let settle = make_sine(1000.0, SR, 2048);
        let _ = eq.process_buffer(&settle);

        let input = make_sine(1000.0, SR, 512);
        let output = eq.process_buffer(&input);

        assert!(
            rms(&output) > rms(&input),
            "Peak +12 dB at 1 kHz should increase RMS of 1 kHz sine"
        );
    }

    #[test]
    fn test_notch_reduces_at_center_freq() {
        // Use a wider notch (lower Q) so the attenuation is deep enough across
        // the entire measurement window without needing extensive settling.
        let mut eq = ParametricEq::new(SR).with_band(EqBand::notch(1000.0, 1.0));
        // Give the filter plenty of time to settle with the same frequency.
        let settle = make_sine(1000.0, SR, 8192);
        let _ = eq.process_buffer(&settle);

        let input = make_sine(1000.0, SR, 1024);
        let output = eq.process_buffer(&input);

        assert!(
            rms(&output) < rms(&input) * 0.8,
            "Notch at 1 kHz should reduce RMS: in={:.4}, out={:.4}",
            rms(&input),
            rms(&output)
        );
    }

    #[test]
    fn test_low_shelf_affects_low_frequencies() {
        let mut eq = ParametricEq::new(SR).with_band(EqBand::low_shelf(200.0, 6.0));
        let settle = make_sine(100.0, SR, 2048);
        let _ = eq.process_buffer(&settle);

        let input = make_sine(100.0, SR, 512);
        let output = eq.process_buffer(&input);

        // A +6 dB low-shelf at 200 Hz should boost a 100 Hz sine.
        assert!(
            rms(&output) > rms(&input),
            "Low shelf +6 dB should boost 100 Hz: in={:.4}, out={:.4}",
            rms(&input),
            rms(&output)
        );
    }

    #[test]
    fn test_process_buffer_length_correct() {
        let mut eq = ParametricEq::new(SR).with_band(EqBand::peaking(1000.0, 6.0, 1.0));
        let input = vec![0.5_f32; 128];
        let output = eq.process_buffer(&input);
        assert_eq!(output.len(), 128);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut eq = ParametricEq::new(SR).with_band(EqBand::peaking(1000.0, 6.0, 1.0));
        // Prime the filter with non-zero signal.
        let _ = eq.process_buffer(&vec![1.0_f32; 64]);
        eq.reset();
        // After reset the first output for a zero input should be exactly zero.
        assert_eq!(eq.process_sample(0.0), 0.0, "reset should clear history");
    }

    #[test]
    fn test_disabled_band_bypasses_signal() {
        let band = EqBand {
            band_type: EqBandType::Peaking,
            frequency_hz: 1000.0,
            gain_db: 20.0,
            q: 1.0,
            enabled: false,
        };
        let mut eq = ParametricEq::new(SR).with_band(band);
        let input: Vec<f32> = make_sine(1000.0, SR, 256);
        let output = eq.process_buffer(&input);
        for (i, (&a, &b)) in input.iter().zip(output.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "Disabled band should pass unchanged, sample {i}: in={a}, out={b}"
            );
        }
    }

    #[test]
    fn test_set_band_gain_updates_correctly() {
        let mut eq = ParametricEq::new(SR).with_band(EqBand::peaking(1000.0, 0.0, 1.0));
        eq.set_band_gain(0, 6.0).expect("index 0 should be valid");
        assert!((eq.bands[0].gain_db - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_set_band_gain_out_of_range() {
        let mut eq = ParametricEq::new(SR);
        let result = eq.set_band_gain(5, 3.0);
        assert!(result.is_err(), "Out-of-range index should return Err");
    }

    #[test]
    fn test_set_band_enabled_toggles() {
        let mut eq = ParametricEq::new(SR).with_band(EqBand::peaking(1000.0, 12.0, 1.0));
        eq.set_band_enabled(0, false).expect("index 0 should be valid");
        assert!(!eq.bands[0].enabled);
        eq.set_band_enabled(0, true).expect("index 0 should be valid");
        assert!(eq.bands[0].enabled);
    }

    #[test]
    fn test_all_outputs_finite() {
        let mut eq = ParametricEq::new(SR)
            .with_band(EqBand::low_pass(5000.0, 0.707))
            .with_band(EqBand::high_pass(80.0, 0.707))
            .with_band(EqBand::peaking(1000.0, 3.0, 1.5))
            .with_band(EqBand::notch(2000.0, 8.0));

        let sine = make_sine(500.0, SR, 1024);
        let output = eq.process_buffer(&sine);
        for (i, &s) in output.iter().enumerate() {
            assert!(s.is_finite(), "Output at sample {i} is not finite: {s}");
        }
    }
}
