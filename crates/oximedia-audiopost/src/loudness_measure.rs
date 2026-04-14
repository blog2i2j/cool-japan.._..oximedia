//! ITU-R BS.1770-4 loudness measurement.
//!
//! Provides a lightweight, self-contained implementation of integrated LUFS
//! measurement following ITU-R BS.1770-4 and EBU R128 gating semantics.
//!
//! # Algorithm
//!
//! 1. Apply the ITU K-weighting pre-filter (two biquad stages).
//! 2. Compute mean-square of the K-weighted signal in 400 ms blocks (overlapped
//!    75 % for 100 ms hop).
//! 3. Apply absolute gate (–70 LUFS).
//! 4. Compute preliminary loudness from gated blocks.
//! 5. Apply relative gate (–10 LU below preliminary).
//! 6. Return integrated loudness from blocks passing both gates.

/// Integrated loudness measurement following ITU-R BS.1770-4.
pub struct LoudnessMeasurement;

impl LoudnessMeasurement {
    /// Compute integrated loudness in LUFS for a mono PCM signal.
    ///
    /// # Arguments
    ///
    /// * `samples`     — Mono PCM samples normalised to ±1.0.
    /// * `sample_rate` — Sample rate in Hz.
    ///
    /// # Returns
    ///
    /// Integrated loudness in LUFS (negative value, typically –23 to –5).
    /// Returns `-f32::INFINITY` when no samples pass the gating conditions.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_audiopost::loudness_measure::LoudnessMeasurement;
    ///
    /// // Sine wave at –23 LUFS ≈ amplitude 0.0708
    /// let sr = 48_000u32;
    /// let secs = 5.0f32;
    /// let amp = 10f32.powf((-23.0 - 0.691) / 20.0); // approx –23 LUFS
    /// let samples: Vec<f32> = (0..((sr as f32 * secs) as usize))
    ///     .map(|i| amp * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr as f32).sin())
    ///     .collect();
    /// let lufs = LoudnessMeasurement::compute_lufs(&samples, sr);
    /// assert!(lufs < -20.0 && lufs > -26.0, "lufs={lufs}");
    /// ```
    #[must_use]
    pub fn compute_lufs(samples: &[f32], sample_rate: u32) -> f32 {
        if samples.is_empty() || sample_rate == 0 {
            return f32::NEG_INFINITY;
        }

        let sr = sample_rate as f64;

        // ── Stage 1: K-weighting pre-filter (two biquad stages) ──────────────
        let filtered = Self::apply_k_weighting(samples, sr);

        // ── Stage 2: Mean-square in 400 ms blocks, 100 ms hop ────────────────
        let block_len = ((0.4 * sr).round() as usize).max(1);
        let hop_len = ((0.1 * sr).round() as usize).max(1);

        let mut block_ms: Vec<f64> = Vec::new();
        let mut start = 0usize;
        while start + block_len <= filtered.len() {
            let ms: f64 = filtered[start..start + block_len]
                .iter()
                .map(|&x| (x as f64) * (x as f64))
                .sum::<f64>()
                / block_len as f64;
            block_ms.push(ms);
            start += hop_len;
        }

        if block_ms.is_empty() {
            // Signal too short for a full block — measure the whole buffer
            let ms: f64 = filtered.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>()
                / filtered.len() as f64;
            return Self::ms_to_lufs(ms) as f32;
        }

        // ── Stage 3: Absolute gate — –70 LUFS ────────────────────────────────
        // –70 LUFS → mean-square threshold
        let abs_threshold = 10f64.powf((-70.0_f64 - 0.691) / 10.0);
        let gated_abs: Vec<f64> = block_ms.iter().copied().filter(|&ms| ms > abs_threshold).collect();

        if gated_abs.is_empty() {
            return f32::NEG_INFINITY;
        }

        // ── Stage 4: Preliminary loudness ─────────────────────────────────────
        let prelim_ms = gated_abs.iter().sum::<f64>() / gated_abs.len() as f64;
        let prelim_lufs = Self::ms_to_lufs(prelim_ms);

        // ── Stage 5: Relative gate — –10 LU below preliminary ────────────────
        let rel_threshold = 10f64.powf((prelim_lufs - 10.0 - 0.691) / 10.0);
        let gated_rel: Vec<f64> = block_ms
            .iter()
            .copied()
            .filter(|&ms| ms > abs_threshold && ms > rel_threshold)
            .collect();

        if gated_rel.is_empty() {
            return prelim_lufs as f32;
        }

        // ── Stage 6: Integrated loudness ──────────────────────────────────────
        let integrated_ms = gated_rel.iter().sum::<f64>() / gated_rel.len() as f64;
        Self::ms_to_lufs(integrated_ms) as f32
    }

    /// Apply ITU-R BS.1770 K-weighting pre-filter to a mono signal.
    ///
    /// The K-weighting consists of two cascaded biquad IIR filters:
    /// * Stage 1: High-frequency shelf (+4 dB at ~2 kHz)
    /// * Stage 2: High-pass (100 Hz, 2nd order Butterworth)
    ///
    /// Coefficients are computed analytically for the given sample rate.
    fn apply_k_weighting(samples: &[f32], sr: f64) -> Vec<f32> {
        // Pre-filter stage 1 (high-shelf, ~4 dB at 2 kHz)
        // Using standard ITU coefficients for 48 kHz; adjusted by bilinear
        // transform for other sample rates.
        let (b1, a1) = Self::shelf_coeffs(sr);
        let stage1 = Self::biquad_filter(samples, &b1, &a1);

        // Pre-filter stage 2 (2nd-order high-pass at 38.1 Hz)
        let (b2, a2) = Self::hp_coeffs(sr);
        Self::biquad_filter(&stage1, &b2, &a2)
    }

    /// Compute high-shelf biquad coefficients for K-weighting stage 1.
    fn shelf_coeffs(sr: f64) -> ([f64; 3], [f64; 3]) {
        // Approximated ITU-R BS.1770 stage 1 high-shelf
        // Db=+4.0, Fc=1502.1 Hz  (for 48 kHz).  Bilinear-transform scaled.
        let db = 4.0_f64;
        let fc = 1502.1_f64;
        let kv = (std::f64::consts::PI * fc / sr).tan();
        let v0 = 10f64.powf(db / 20.0);
        let sqrt_v0 = v0.sqrt();

        // Boost shelf
        let denom = 1.0 + std::f64::consts::SQRT_2 * kv + kv * kv;
        let b0 = (v0 + std::f64::consts::SQRT_2 * sqrt_v0 * kv + kv * kv) / denom;
        let b1 = 2.0 * (kv * kv - v0) / denom;
        let b2 = (v0 - std::f64::consts::SQRT_2 * sqrt_v0 * kv + kv * kv) / denom;
        let a1 = 2.0 * (kv * kv - 1.0) / denom;
        let a2 = (1.0 - std::f64::consts::SQRT_2 * kv + kv * kv) / denom;

        ([b0, b1, b2], [1.0, a1, a2])
    }

    /// Compute 2nd-order high-pass biquad coefficients for K-weighting stage 2.
    fn hp_coeffs(sr: f64) -> ([f64; 3], [f64; 3]) {
        // Butterworth 2nd-order HP at Fc = 38.13 Hz (ITU stage 2)
        let fc = 38.13_f64;
        let kv = (std::f64::consts::PI * fc / sr).tan();
        let denom = kv * kv + std::f64::consts::SQRT_2 * kv + 1.0;
        let b0 = 1.0 / denom;
        let b1 = -2.0 / denom;
        let b2 = 1.0 / denom;
        let a1 = 2.0 * (kv * kv - 1.0) / denom;
        let a2 = (kv * kv - std::f64::consts::SQRT_2 * kv + 1.0) / denom;
        ([b0, b1, b2], [1.0, a1, a2])
    }

    /// Apply a Direct Form II biquad IIR filter to `input`.
    fn biquad_filter(input: &[f32], b: &[f64; 3], a: &[f64; 3]) -> Vec<f32> {
        let mut out = Vec::with_capacity(input.len());
        let mut w1 = 0.0f64;
        let mut w2 = 0.0f64;

        for &x in input {
            let x = x as f64;
            let w = x - a[1] * w1 - a[2] * w2;
            let y = b[0] * w + b[1] * w1 + b[2] * w2;
            w2 = w1;
            w1 = w;
            out.push(y as f32);
        }
        out
    }

    /// Convert mean-square to LUFS.
    ///
    /// LUFS = –0.691 + 10·log10(mean_square)
    fn ms_to_lufs(ms: f64) -> f64 {
        if ms <= 0.0 {
            return f64::NEG_INFINITY;
        }
        -0.691 + 10.0 * ms.log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 48_000;

    fn sine_wave(freq_hz: f32, amplitude: f32, duration_secs: f32, sr: u32) -> Vec<f32> {
        let n = (sr as f32 * duration_secs) as usize;
        (0..n)
            .map(|i| {
                amplitude
                    * (2.0 * std::f32::consts::PI * freq_hz * i as f32 / sr as f32).sin()
            })
            .collect()
    }

    #[test]
    fn empty_input_returns_neg_infinity() {
        assert_eq!(LoudnessMeasurement::compute_lufs(&[], SR), f32::NEG_INFINITY);
    }

    #[test]
    fn silence_returns_neg_infinity() {
        let samples = vec![0.0f32; SR as usize * 5];
        assert_eq!(
            LoudnessMeasurement::compute_lufs(&samples, SR),
            f32::NEG_INFINITY
        );
    }

    #[test]
    fn loud_signal_above_neg_5_lufs() {
        // Full-scale sine wave should be very loud
        let samples = sine_wave(1000.0, 1.0, 5.0, SR);
        let lufs = LoudnessMeasurement::compute_lufs(&samples, SR);
        assert!(lufs > -5.0, "lufs={lufs} should be > -5 for full-scale");
    }

    #[test]
    fn quiet_signal_below_neg_30_lufs() {
        // Very quiet sine: amplitude 0.001 → roughly –60 LUFS
        let samples = sine_wave(1000.0, 0.001, 5.0, SR);
        let lufs = LoudnessMeasurement::compute_lufs(&samples, SR);
        // Either measured quiet or gated to -inf
        assert!(
            lufs < -30.0 || lufs == f32::NEG_INFINITY,
            "lufs={lufs} should be very quiet"
        );
    }

    #[test]
    fn loudness_increases_with_amplitude() {
        let quiet = sine_wave(1000.0, 0.1, 3.0, SR);
        let loud = sine_wave(1000.0, 0.8, 3.0, SR);
        let l_quiet = LoudnessMeasurement::compute_lufs(&quiet, SR);
        let l_loud = LoudnessMeasurement::compute_lufs(&loud, SR);
        assert!(
            l_loud > l_quiet,
            "louder signal should have higher LUFS: {l_loud} vs {l_quiet}"
        );
    }

    #[test]
    fn target_minus_23_lufs_within_tolerance() {
        // Amplitude calibrated for approximately –23 LUFS
        let amp = 10f32.powf((-23.0_f32 - 0.691) / 20.0);
        let samples = sine_wave(1000.0, amp, 10.0, SR);
        let lufs = LoudnessMeasurement::compute_lufs(&samples, SR);
        assert!(
            lufs > -30.0 && lufs < -16.0,
            "lufs={lufs} not in expected range [-30, -16] for ~-23 LUFS target"
        );
    }

    #[test]
    fn zero_sample_rate_returns_neg_infinity() {
        let samples = vec![0.5f32; 1000];
        assert_eq!(LoudnessMeasurement::compute_lufs(&samples, 0), f32::NEG_INFINITY);
    }

    #[test]
    fn very_short_signal_does_not_panic() {
        let samples = vec![0.5f32; 10];
        let _ = LoudnessMeasurement::compute_lufs(&samples, SR);
    }
}
