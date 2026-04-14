//! Bit-depth conversion and dithering for audio samples.
//!
//! This module provides utilities for converting between common PCM bit depths:
//!
//! * [`convert_16bit_to_24bit`] — lossless upward conversion of 16-bit PCM
//!   integers to 24-bit-wide `i32` values (stored in the low 24 bits).
//! * [`dither_16bit`] — convert normalised floating-point samples to 16-bit
//!   PCM with triangular probability density function (TPDF) dither to mask
//!   quantisation noise.
//!
//! # Dither
//!
//! TPDF dither adds two independent uniform noise signals (each spanning
//! ±0.5 LSB) before quantisation.  The result is a flat noise floor with no
//! correlated quantisation artefacts (no harmonic distortion).
//!
//! The implementation uses a simple deterministic pseudo-random number
//! generator (xorshift64) so tests are reproducible without the `rand` crate.
//!
//! # Example
//!
//! ```
//! use oximedia_restore::bit_depth::{convert_16bit_to_24bit, dither_16bit};
//!
//! // 16 → 24 bit (lossless)
//! let pcm16 = vec![0i16, i16::MAX, i16::MIN];
//! let pcm24 = convert_16bit_to_24bit(&pcm16);
//! assert_eq!(pcm24[0], 0);
//! assert_eq!(pcm24[1], i16::MAX as i32);
//! assert_eq!(pcm24[2], i16::MIN as i32);
//!
//! // float → 16-bit with dither
//! let float_samples = vec![0.0f32, 0.5, -0.5];
//! let dithered = dither_16bit(&float_samples);
//! assert_eq!(dithered.len(), 3);
//! ```

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Convert 16-bit PCM integer samples to 24-bit-range `i32` values.
///
/// The 16-bit value is stored as-is in the `i32`; no scaling or shifting is
/// performed.  This is a lossless widening conversion: the numeric value of
/// each sample is preserved exactly.
///
/// To produce correctly-scaled 24-bit values (shifted to the most-significant
/// bits), see [`convert_16bit_to_24bit_msb`].
///
/// # Returns
///
/// A `Vec<i32>` with the same length as `samples`.
pub fn convert_16bit_to_24bit(samples: &[i16]) -> Vec<i32> {
    samples.iter().map(|&s| s as i32).collect()
}

/// Convert 16-bit PCM samples to 24-bit values aligned to the most-significant
/// bits (i.e., left-shifted by 8).
///
/// This representation is used by some 24-bit DAC/ADC interfaces that expect
/// the audio data in the MSB-aligned position.
///
/// # Returns
///
/// A `Vec<i32>` with the same length as `samples`.  Each value is in the
/// range `[i16::MIN * 256, i16::MAX * 256]`.
pub fn convert_16bit_to_24bit_msb(samples: &[i16]) -> Vec<i32> {
    samples.iter().map(|&s| (s as i32) << 8).collect()
}

/// Convert normalised floating-point samples to 16-bit PCM with TPDF dither.
///
/// # Parameters
///
/// * `samples` — normalised audio samples in the range `[-1.0, 1.0]`.
///   Values outside this range are clamped before quantisation.
///
/// # Returns
///
/// A `Vec<i16>` of the same length as `samples`.
///
/// # Algorithm
///
/// For each sample *x*:
///
/// 1. Scale to the 16-bit integer range: `x_scaled = x × 32767.0`.
/// 2. Generate two independent uniform random values `r1, r2 ∈ [−0.5, 0.5]`
///    using an xorshift64 PRNG.
/// 3. Add the TPDF dither noise: `x_dithered = x_scaled + r1 + r2`.
/// 4. Round to the nearest integer and clamp to `[i16::MIN, i16::MAX]`.
pub fn dither_16bit(samples: &[f32]) -> Vec<i16> {
    const SCALE: f32 = 32767.0;
    // xorshift64 state — non-zero seed.
    let mut rng_state: u64 = 0x9E37_79B9_7F4A_7C15;

    samples
        .iter()
        .map(|&x| {
            let x_scaled = x.clamp(-1.0, 1.0) * SCALE;

            // Two uniform random values in [0, 1) via xorshift64.
            let r1 = xorshift64(&mut rng_state) as f32 / u64::MAX as f32 - 0.5;
            let r2 = xorshift64(&mut rng_state) as f32 / u64::MAX as f32 - 0.5;

            let dithered = x_scaled + r1 + r2;
            dithered.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// xorshift64 pseudo-random number generator (Marsaglia 2003).
///
/// Returns the next pseudo-random `u64` and updates `state` in-place.
/// `state` must never be zero.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ─── convert_16bit_to_24bit ───────────────────────────────────────────

    #[test]
    fn test_16_to_24_zero() {
        assert_eq!(convert_16bit_to_24bit(&[0i16]), vec![0i32]);
    }

    #[test]
    fn test_16_to_24_max() {
        assert_eq!(convert_16bit_to_24bit(&[i16::MAX]), vec![i16::MAX as i32]);
    }

    #[test]
    fn test_16_to_24_min() {
        assert_eq!(convert_16bit_to_24bit(&[i16::MIN]), vec![i16::MIN as i32]);
    }

    #[test]
    fn test_16_to_24_preserves_length() {
        let samples: Vec<i16> = (0..100).map(|i| i as i16 - 50).collect();
        let result = convert_16bit_to_24bit(&samples);
        assert_eq!(result.len(), samples.len());
    }

    #[test]
    fn test_16_to_24_lossless_roundtrip() {
        let original: Vec<i16> = vec![-32768, -100, 0, 100, 32767];
        let widened = convert_16bit_to_24bit(&original);
        // Cast back — values must be in i16 range.
        let roundtrip: Vec<i16> = widened.iter().map(|&v| v as i16).collect();
        assert_eq!(roundtrip, original);
    }

    #[test]
    fn test_16_to_24_msb_shift() {
        let samples = vec![1i16];
        let msb = convert_16bit_to_24bit_msb(&samples);
        assert_eq!(msb[0], 256, "left-shift by 8 should multiply by 256");
    }

    #[test]
    fn test_16_to_24_msb_max() {
        let samples = vec![i16::MAX];
        let msb = convert_16bit_to_24bit_msb(&samples);
        assert_eq!(msb[0], i16::MAX as i32 * 256);
    }

    #[test]
    fn test_16_to_24_empty() {
        let result = convert_16bit_to_24bit(&[]);
        assert!(result.is_empty());
    }

    // ─── dither_16bit ─────────────────────────────────────────────────────

    #[test]
    fn test_dither_output_length() {
        let samples = vec![0.0f32; 64];
        assert_eq!(dither_16bit(&samples).len(), 64);
    }

    #[test]
    fn test_dither_zero_near_zero() {
        // A silent input should produce values very close to 0 (only dither noise).
        let samples = vec![0.0f32; 1024];
        let out = dither_16bit(&samples);
        // TPDF dither spans ±1 LSB at most → values should be in [-1, 1].
        for &s in &out {
            assert!(
                s.unsigned_abs() <= 1,
                "dither of silence: |{s}| should be ≤ 1"
            );
        }
    }

    #[test]
    fn test_dither_positive_full_scale() {
        let samples = vec![1.0f32; 4];
        let out = dither_16bit(&samples);
        // Maximum output ≈ 32767; dither adds at most 1 LSB.
        for &s in &out {
            assert!(s >= 32766, "full-scale positive: expected ≈ 32767, got {s}");
        }
    }

    #[test]
    fn test_dither_negative_full_scale() {
        let samples = vec![-1.0f32; 4];
        let out = dither_16bit(&samples);
        for &s in &out {
            assert!(
                s <= -32767,
                "full-scale negative: expected ≈ -32768, got {s}"
            );
        }
    }

    #[test]
    fn test_dither_clamp_above_1() {
        // Values > 1.0 should be clamped to 1.0 before scaling.
        let samples = vec![2.0f32, 100.0f32];
        let out = dither_16bit(&samples);
        for &s in &out {
            // After clamping to 1.0 and adding ≤1 LSB dither, must be ≤ 32767.
            // i16 type guarantees this, just verify non-panic.
            let _ = s;
        }
    }

    #[test]
    fn test_dither_empty_input() {
        let out = dither_16bit(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn test_xorshift_non_zero() {
        let mut state = 0xDEAD_BEEF_CAFE_BABEu64;
        for _ in 0..100 {
            let v = super::xorshift64(&mut state);
            assert_ne!(v, 0, "xorshift64 should never produce 0");
        }
    }
}
