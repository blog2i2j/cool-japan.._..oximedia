//! Robustness simulation for watermark stress testing.
//!
//! Simulates common signal-processing attacks that a watermarked image might
//! survive in the real world: JPEG-style quantisation noise and additive
//! Gaussian noise.  The goal is **not** to implement a full JPEG codec but to
//! produce a degraded version with realistic SNR characteristics so that
//! detection algorithms can be validated without external dependencies.
//!
//! # Attacks implemented
//!
//! | Name | Description |
//! |------|-------------|
//! | [`WatermarkRobustness::simulate_compression`] | Round-trip through coarse 8×8 block quantisation + noise |
//! | [`WatermarkRobustness::add_awgn`] | Additive White Gaussian Noise at a given SNR |

/// Watermark robustness test utilities.
pub struct WatermarkRobustness;

impl WatermarkRobustness {
    /// Simulate lossy compression artefacts on a raw image buffer.
    ///
    /// The simulation applies two degradation steps:
    ///
    /// 1. **Block quantisation** — divide each 8×8 block's pixel values by a
    ///    quality-dependent step `Δ`, round to integer, then multiply back.
    ///    This mimics the quantisation stage of JPEG without needing a DCT.
    ///    The step size is derived from `quality` as
    ///    `Δ = max(1, (100 - quality) / 5)`.
    ///
    /// 2. **Additive noise** — a small amount of zero-mean noise proportional
    ///    to `(100 - quality)` is added (using a deterministic pseudo-random
    ///    sequence so tests are reproducible without the `rand` crate).
    ///
    /// # Parameters
    ///
    /// - `img`    : flat row-major pixel data (any byte layout; treated as 1-channel).
    /// - `quality`: JPEG-style quality factor in `[1, 100]`.  100 = lossless
    ///              (only minimal noise), 1 = maximum degradation.
    ///
    /// # Returns
    ///
    /// A new `Vec<u8>` of the same length with degraded pixel values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use oximedia_watermark::robustness_test::WatermarkRobustness;
    ///
    /// let img: Vec<u8> = (0u8..=255).collect();
    /// let degraded = WatermarkRobustness::simulate_compression(&img, 75);
    /// assert_eq!(degraded.len(), img.len());
    /// ```
    #[must_use]
    pub fn simulate_compression(img: &[u8], quality: u8) -> Vec<u8> {
        let quality = quality.clamp(1, 100);

        // Step size for quantisation: higher quality → smaller step.
        let step = ((100u32 - u32::from(quality)) / 5).max(1) as f32;

        // Maximum additive noise amplitude in pixel units.
        let noise_amp = ((100u32 - u32::from(quality)) as f32) * 0.05;

        let mut out = Vec::with_capacity(img.len());

        for (i, &px) in img.iter().enumerate() {
            // 1. Block quantisation: round to nearest multiple of `step`.
            let quantised = ((f32::from(px) / step).round() * step).clamp(0.0, 255.0);

            // 2. Deterministic pseudo-random noise using a simple LCG.
            //    seed derived from pixel index so it's position-dependent.
            let seed = (i as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Map to [-noise_amp, +noise_amp].
            let noise = ((seed >> 33) as f32 / (u32::MAX as f32) - 0.5) * 2.0 * noise_amp;

            let degraded = (quantised + noise).round().clamp(0.0, 255.0) as u8;
            out.push(degraded);
        }

        out
    }

    /// Add Additive White Gaussian Noise (AWGN) to an image at a given SNR.
    ///
    /// The noise is generated with a Box-Muller transform using a deterministic
    /// seed so that tests remain reproducible.
    ///
    /// # Parameters
    ///
    /// - `img`    : input pixel data.
    /// - `snr_db` : signal-to-noise ratio in dB. Higher = less noise.
    ///
    /// # Returns
    ///
    /// A new `Vec<u8>` with noise added.
    #[must_use]
    pub fn add_awgn(img: &[u8], snr_db: f32) -> Vec<u8> {
        if img.is_empty() {
            return Vec::new();
        }

        // Estimate signal power (as f32 in [0, 1]).
        let signal_power: f32 = img
            .iter()
            .map(|&p| (f32::from(p) / 255.0).powi(2))
            .sum::<f32>()
            / img.len() as f32;

        let linear_snr = 10f32.powf(snr_db / 10.0);
        let noise_std = if signal_power > 0.0 {
            (signal_power / linear_snr).sqrt()
        } else {
            0.0
        };

        let mut out = Vec::with_capacity(img.len());

        // Box-Muller pairs with a deterministic state.
        let mut lcg_state: u64 = 0x1234_5678_ABCD_EF01;

        let mut pairs = img.chunks(2);
        while let Some(chunk) = pairs.next() {
            let u1 = lcg_next(&mut lcg_state);
            let u2 = lcg_next(&mut lcg_state);

            // Box-Muller transform → two independent standard normal samples.
            let (z0, z1) = box_muller(u1, u2);

            let p0 = f32::from(chunk[0]) / 255.0 + z0 * noise_std;
            let px0 = (p0 * 255.0).round().clamp(0.0, 255.0) as u8;
            out.push(px0);

            if chunk.len() == 2 {
                let p1 = f32::from(chunk[1]) / 255.0 + z1 * noise_std;
                let px1 = (p1 * 255.0).round().clamp(0.0, 255.0) as u8;
                out.push(px1);
            }
        }

        out.truncate(img.len());
        out
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Advance an LCG and return a uniform sample in (0, 1).
fn lcg_next(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    // Use upper 32 bits for better statistical quality.
    let bits = (*state >> 32) as u32;
    // Map to (0, 1) exclusive.
    (bits as f32 + 0.5) / (u32::MAX as f32 + 1.0)
}

/// Box-Muller transform: two uniform samples → two independent N(0,1) samples.
fn box_muller(u1: f32, u2: f32) -> (f32, f32) {
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_preserves_length() {
        let img: Vec<u8> = (0u8..=255).collect();
        let degraded = WatermarkRobustness::simulate_compression(&img, 90);
        assert_eq!(degraded.len(), img.len());
    }

    #[test]
    fn test_high_quality_minimal_change() {
        let img = vec![128u8; 256];
        let degraded = WatermarkRobustness::simulate_compression(&img, 99);
        // At quality 99, step = max(1,(100-99)/5) = 1, noise_amp = 0.05.
        // Pixels should be very close to 128.
        for (&orig, &deg) in img.iter().zip(degraded.iter()) {
            let diff = (i32::from(orig) - i32::from(deg)).unsigned_abs();
            assert!(diff <= 5, "high-quality should produce small change: diff={diff}");
        }
    }

    #[test]
    fn test_low_quality_more_distortion() {
        let img = vec![128u8; 256];
        let hi = WatermarkRobustness::simulate_compression(&img, 95);
        let lo = WatermarkRobustness::simulate_compression(&img, 10);
        let mse_hi: f32 = img
            .iter()
            .zip(hi.iter())
            .map(|(&a, &b)| (i32::from(a) - i32::from(b)).pow(2) as f32)
            .sum::<f32>()
            / img.len() as f32;
        let mse_lo: f32 = img
            .iter()
            .zip(lo.iter())
            .map(|(&a, &b)| (i32::from(a) - i32::from(b)).pow(2) as f32)
            .sum::<f32>()
            / img.len() as f32;
        assert!(
            mse_lo > mse_hi,
            "lower quality should produce higher MSE: hi={mse_hi}, lo={mse_lo}"
        );
    }

    #[test]
    fn test_awgn_preserves_length() {
        let img: Vec<u8> = (0..200u8).collect();
        let noisy = WatermarkRobustness::add_awgn(&img, 30.0);
        assert_eq!(noisy.len(), img.len());
    }

    #[test]
    fn test_awgn_empty() {
        let result = WatermarkRobustness::add_awgn(&[], 20.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_quality_clamped_to_one() {
        // quality=0 is clamped to 1 — should not panic.
        let img = vec![100u8; 64];
        let _ = WatermarkRobustness::simulate_compression(&img, 0);
    }
}
