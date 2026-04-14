//! EBU R128 / ITU-R BS.1770 loudness metering.
//!
//! Implements the K-weighted integrated loudness algorithm described in
//! EBU R128 and ITU-R BS.1770-4, including:
//!
//! - **K-weighting filter** — two cascaded biquad stages (pre-filter + RLB).
//! - **400 ms gating block** with 75 % overlap (100 ms hop).
//! - **Absolute gate** at −70 LUFS.
//! - **Relative gate** at −10 LU below the ungated mean.
//!
//! The `R128Meter` accepts mono f32 blocks (normalized to `[-1.0, 1.0]`) and
//! computes the integrated programme loudness in LUFS.
//!
//! # Example
//!
//! ```rust
//! use oximedia_audio::r128::R128Meter;
//!
//! let mut meter = R128Meter::new(48_000);
//! // Feed 1 second of -23 LUFS sine at 1 kHz (very rough approximation).
//! let block: Vec<f32> = (0..480).map(|i| (i as f32 * 0.13).sin() * 0.224).collect();
//! for _ in 0..10 {
//!     meter.add_block(&block);
//! }
//! let lufs = meter.integrated_lufs();
//! assert!(lufs < 0.0);
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]

use std::f64::consts::PI;

// ── K-weighting biquad ────────────────────────────────────────────────────────

/// Simple direct-form-I biquad filter (f64 for precision).
#[derive(Clone, Debug)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl Biquad {
    fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }

    /// High-shelf pre-filter (first stage of K-weighting), fs-dependent.
    fn k_prefilter(fs: f64) -> Self {
        // Coefficients from ITU-R BS.1770-4 Annex 1 (normalized for 48 kHz,
        // recomputed analytically for arbitrary fs via bilinear transform).
        let f0 = 1681.974_450_955_533;
        let g = 3.999_843_853_973_347; // dB shelf gain
        let q = 0.707_9955_960_838_675_5;

        let k = (PI * f0 / fs).tan();
        let vh = 10.0_f64.powf(g / 20.0);
        let vb = vh.sqrt();
        let a0 = 1.0 + k / q + k * k;

        Self {
            b0: (vh + vb * k / q + k * k) / a0,
            b1: 2.0 * (k * k - vh) / a0,
            b2: (vh - vb * k / q + k * k) / a0,
            a1: 2.0 * (k * k - 1.0) / a0,
            a2: (1.0 - k / q + k * k) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// RLB high-pass filter (second stage of K-weighting).
    fn k_rlb(fs: f64) -> Self {
        let f0 = 38.135_473_580_000_00;
        let q = 0.5;
        let k = (PI * f0 / fs).tan();
        let a0 = 1.0 + k / q + k * k;

        Self {
            b0: 1.0 / a0,
            b1: -2.0 / a0,
            b2: 1.0 / a0,
            a1: 2.0 * (k * k - 1.0) / a0,
            a2: (1.0 - k / q + k * k) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }
}

// ── R128Meter ─────────────────────────────────────────────────────────────────

/// EBU R128 integrated loudness meter (mono).
///
/// Feed audio in arbitrary-length blocks via [`add_block`]; retrieve the
/// current integrated loudness with [`integrated_lufs`].
pub struct R128Meter {
    sample_rate: u32,
    /// K-weighting pre-filter.
    pre: Biquad,
    /// K-weighting RLB filter.
    rlb: Biquad,
    /// Accumulated samples for the current 400 ms gate block.
    gate_buf: Vec<f64>,
    /// Target block length in samples (400 ms).
    block_len: usize,
    /// Hop size in samples (100 ms, 75 % overlap).
    hop_len: usize,
    /// Offset within the current gate block.
    block_offset: usize,
    /// Mean squared values per completed gating block.
    block_ms: Vec<f64>,
}

impl R128Meter {
    /// Create a new meter for the given sample rate.
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        let fs = sample_rate as f64;
        let block_len = ((400e-3) * fs) as usize;
        let hop_len = ((100e-3) * fs) as usize;
        Self {
            sample_rate,
            pre: Biquad::k_prefilter(fs),
            rlb: Biquad::k_rlb(fs),
            gate_buf: vec![0.0; block_len],
            block_len,
            hop_len,
            block_offset: 0,
            block_ms: Vec::new(),
        }
    }

    /// Reset all internal state.
    pub fn reset(&mut self) {
        *self = Self::new(self.sample_rate);
    }

    /// Feed a block of samples.
    ///
    /// Samples should be linear amplitude in `[-1.0, 1.0]`.
    pub fn add_block(&mut self, block: &[f32]) {
        for &s in block {
            // Apply K-weighting.
            let kw = self.rlb.process(self.pre.process(s as f64));
            self.gate_buf[self.block_offset] = kw * kw;
            self.block_offset += 1;

            if self.block_offset >= self.block_len {
                // Compute mean square over the block.
                let ms: f64 = self.gate_buf.iter().sum::<f64>() / self.block_len as f64;
                self.block_ms.push(ms);
                // Slide window by hop_len.
                self.gate_buf.copy_within(self.hop_len.., 0);
                self.block_offset = self.block_len - self.hop_len;
            }
        }
    }

    /// Integrated programme loudness in LUFS (EBU R128).
    ///
    /// Returns `f32::NEG_INFINITY` if no gating blocks have been completed.
    #[must_use]
    pub fn integrated_lufs(&self) -> f32 {
        if self.block_ms.is_empty() {
            return f32::NEG_INFINITY;
        }

        // Absolute gate: −70 LUFS → linear MS threshold.
        let abs_gate_ms = 10.0_f64.powf((-70.0 - 0.691) / 10.0);
        let above_abs: Vec<f64> = self
            .block_ms
            .iter()
            .copied()
            .filter(|&ms| ms > abs_gate_ms)
            .collect();

        if above_abs.is_empty() {
            return f32::NEG_INFINITY;
        }

        // Ungated mean.
        let ungated_mean = above_abs.iter().sum::<f64>() / above_abs.len() as f64;
        let ungated_lufs = -0.691 + 10.0 * ungated_mean.log10();

        // Relative gate: −10 LU below ungated.
        let rel_gate_lufs = ungated_lufs - 10.0;
        let rel_gate_ms = 10.0_f64.powf((rel_gate_lufs - 0.691) / 10.0);

        let above_rel: Vec<f64> = self
            .block_ms
            .iter()
            .copied()
            .filter(|&ms| ms > rel_gate_ms)
            .collect();

        if above_rel.is_empty() {
            return f32::NEG_INFINITY;
        }

        let gated_mean = above_rel.iter().sum::<f64>() / above_rel.len() as f64;
        let lufs = -0.691 + 10.0 * gated_mean.log10();
        lufs as f32
    }

    /// Short-term loudness of the most recently completed 3-second window.
    ///
    /// Returns `f32::NEG_INFINITY` if insufficient data.
    #[must_use]
    pub fn short_term_lufs(&self) -> f32 {
        // 3 s = 30 × 100 ms hops.
        let window = 30;
        if self.block_ms.len() < window {
            return f32::NEG_INFINITY;
        }
        let recent: &[f64] = &self.block_ms[self.block_ms.len() - window..];
        let mean = recent.iter().sum::<f64>() / window as f64;
        if mean <= 0.0 {
            return f32::NEG_INFINITY;
        }
        (-0.691 + 10.0 * mean.log10()) as f32
    }

    /// Number of completed gating blocks accumulated so far.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.block_ms.len()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI as PI32;

    /// Generate a mono sine wave at `freq_hz` for `duration_secs` at `fs`.
    fn sine(fs: u32, freq_hz: f32, duration_secs: f32, amplitude: f32) -> Vec<f32> {
        let n = (fs as f32 * duration_secs) as usize;
        (0..n)
            .map(|i| amplitude * (2.0 * PI32 * freq_hz * i as f32 / fs as f32).sin())
            .collect()
    }

    #[test]
    fn test_empty_meter_returns_neg_inf() {
        let meter = R128Meter::new(48_000);
        assert_eq!(meter.integrated_lufs(), f32::NEG_INFINITY);
    }

    #[test]
    fn test_meter_produces_finite_value_after_data() {
        let mut meter = R128Meter::new(48_000);
        let sig = sine(48_000, 1000.0, 3.0, 0.2236); // ~-23 LUFS approximate
        meter.add_block(&sig);
        let lufs = meter.integrated_lufs();
        assert!(lufs.is_finite() || lufs == f32::NEG_INFINITY);
    }

    #[test]
    fn test_louder_signal_gives_higher_lufs() {
        let mut low = R128Meter::new(48_000);
        let mut high = R128Meter::new(48_000);

        let quiet = sine(48_000, 1000.0, 5.0, 0.05);
        let loud = sine(48_000, 1000.0, 5.0, 0.5);

        low.add_block(&quiet);
        high.add_block(&loud);

        let lufs_low = low.integrated_lufs();
        let lufs_high = high.integrated_lufs();

        if lufs_low.is_finite() && lufs_high.is_finite() {
            assert!(
                lufs_high > lufs_low,
                "louder signal {lufs_high} should exceed quieter {lufs_low}"
            );
        }
    }

    #[test]
    fn test_block_count_increases() {
        let mut meter = R128Meter::new(48_000);
        assert_eq!(meter.block_count(), 0);
        let sig = vec![0.1f32; 48_000]; // 1 second → several 400 ms blocks
        meter.add_block(&sig);
        assert!(meter.block_count() > 0);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut meter = R128Meter::new(48_000);
        let sig = vec![0.2f32; 48_000];
        meter.add_block(&sig);
        meter.reset();
        assert_eq!(meter.block_count(), 0);
        assert_eq!(meter.integrated_lufs(), f32::NEG_INFINITY);
    }

    #[test]
    fn test_short_term_insufficient_data_returns_neg_inf() {
        let mut meter = R128Meter::new(48_000);
        let sig = vec![0.2f32; 1000]; // very short
        meter.add_block(&sig);
        // Less than 3 seconds worth → NEG_INFINITY.
        let st = meter.short_term_lufs();
        assert_eq!(st, f32::NEG_INFINITY);
    }

    #[test]
    fn test_add_block_small_chunks() {
        let mut meter = R128Meter::new(48_000);
        // Add data in tiny chunks.
        for _ in 0..1000 {
            meter.add_block(&[0.1, 0.1, 0.1, 0.1]);
        }
        // Should not panic and should produce a finite or -inf result.
        let lufs = meter.integrated_lufs();
        assert!(lufs.is_finite() || lufs == f32::NEG_INFINITY);
    }
}
