//! LUT strength / blend utilities (f32 API).
//!
//! Provides functions for blending between the original pixel value and the
//! LUT-transformed value, allowing the strength (opacity) of a LUT to be
//! dialled in from 0 (no effect) to 1 (full effect).

/// Blend an original RGB pixel value with the LUT-transformed output by a
/// `strength` factor.
///
/// `strength = 0.0` returns `original` unchanged.
/// `strength = 1.0` returns `lut_out` unchanged.
/// Intermediate values produce a linear blend.
///
/// All channel values are clamped to `[0.0, 1.0]` before blending.
///
/// # Arguments
///
/// * `original` - Input pixel in linear `[0.0, 1.0]` space.
/// * `lut_out`  - LUT-transformed pixel.
/// * `strength` - Mix factor in `[0.0, 1.0]` (clamped).
#[must_use]
pub fn blend_lut_result(original: [f32; 3], lut_out: [f32; 3], strength: f32) -> [f32; 3] {
    let t = strength.clamp(0.0, 1.0);
    let one_minus_t = 1.0 - t;
    [
        original[0] * one_minus_t + lut_out[0] * t,
        original[1] * one_minus_t + lut_out[1] * t,
        original[2] * one_minus_t + lut_out[2] * t,
    ]
}

/// Blend an entire scanline (or image buffer) of pixels with their LUT output.
///
/// `original` and `lut_out` must be RGB-interleaved (`[r0 g0 b0 r1 g1 b1 ...]`)
/// and of equal length (must be a multiple of 3).  Returns a new `Vec<f32>`.
///
/// # Arguments
///
/// * `original` - Source pixel buffer (RGB interleaved).
/// * `lut_out`  - LUT-transformed buffer (same layout).
/// * `strength` - Blend factor `[0.0, 1.0]`.
///
/// # Returns
///
/// Blended buffer of the same length, or empty `Vec` if lengths differ or are
/// not multiples of 3.
#[must_use]
pub fn blend_lut_result_buffer(original: &[f32], lut_out: &[f32], strength: f32) -> Vec<f32> {
    if original.len() != lut_out.len() || original.len() % 3 != 0 {
        return Vec::new();
    }

    let t = strength.clamp(0.0, 1.0);
    let one_minus_t = 1.0 - t;

    original
        .iter()
        .zip(lut_out.iter())
        .map(|(&o, &l)| o * one_minus_t + l * t)
        .collect()
}

/// Apply a strength adjustment to a single channel value.
///
/// `strength = 0.0` returns `original`; `strength = 1.0` returns `lut_out`.
#[must_use]
#[inline]
pub fn blend_lut_channel(original: f32, lut_out: f32, strength: f32) -> f32 {
    let t = strength.clamp(0.0, 1.0);
    original * (1.0 - t) + lut_out * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blend_strength_zero_returns_original() {
        let orig = [0.2_f32, 0.4, 0.6];
        let lut = [0.8_f32, 0.6, 0.4];
        let result = blend_lut_result(orig, lut, 0.0);
        assert!((result[0] - orig[0]).abs() < 1e-6);
        assert!((result[1] - orig[1]).abs() < 1e-6);
        assert!((result[2] - orig[2]).abs() < 1e-6);
    }

    #[test]
    fn test_blend_strength_one_returns_lut_out() {
        let orig = [0.2_f32, 0.4, 0.6];
        let lut = [0.8_f32, 0.6, 0.4];
        let result = blend_lut_result(orig, lut, 1.0);
        assert!((result[0] - lut[0]).abs() < 1e-6);
        assert!((result[1] - lut[1]).abs() < 1e-6);
        assert!((result[2] - lut[2]).abs() < 1e-6);
    }

    #[test]
    fn test_blend_strength_midpoint() {
        let orig = [0.0_f32, 0.0, 0.0];
        let lut = [1.0_f32, 1.0, 1.0];
        let result = blend_lut_result(orig, lut, 0.5);
        assert!((result[0] - 0.5).abs() < 1e-6);
        assert!((result[1] - 0.5).abs() < 1e-6);
        assert!((result[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_blend_strength_clamped_negative() {
        let orig = [0.3_f32; 3];
        let lut = [0.7_f32; 3];
        // Strength below 0 should behave as 0
        let result = blend_lut_result(orig, lut, -0.5);
        assert!((result[0] - orig[0]).abs() < 1e-6);
    }

    #[test]
    fn test_blend_strength_clamped_over_one() {
        let orig = [0.3_f32; 3];
        let lut = [0.7_f32; 3];
        // Strength above 1 should behave as 1
        let result = blend_lut_result(orig, lut, 2.0);
        assert!((result[0] - lut[0]).abs() < 1e-6);
    }

    #[test]
    fn test_blend_buffer_identity() {
        let buf: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let lut_buf: Vec<f32> = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4];
        let result = blend_lut_result_buffer(&buf, &lut_buf, 0.0);
        assert_eq!(result.len(), buf.len());
        for (r, &o) in result.iter().zip(buf.iter()) {
            assert!((r - o).abs() < 1e-6);
        }
    }

    #[test]
    fn test_blend_buffer_full_strength() {
        let buf: Vec<f32> = vec![0.1, 0.2, 0.3];
        let lut_buf: Vec<f32> = vec![0.9, 0.8, 0.7];
        let result = blend_lut_result_buffer(&buf, &lut_buf, 1.0);
        for (r, &l) in result.iter().zip(lut_buf.iter()) {
            assert!((r - l).abs() < 1e-6);
        }
    }

    #[test]
    fn test_blend_buffer_mismatch_returns_empty() {
        let buf: Vec<f32> = vec![0.1, 0.2, 0.3];
        let lut_buf: Vec<f32> = vec![0.9, 0.8];
        assert!(blend_lut_result_buffer(&buf, &lut_buf, 0.5).is_empty());
    }

    #[test]
    fn test_blend_buffer_non_multiple_of_3() {
        let buf: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
        let lut_buf: Vec<f32> = vec![0.9, 0.8, 0.7, 0.6];
        // Both same length but not multiple of 3
        assert!(blend_lut_result_buffer(&buf, &lut_buf, 0.5).is_empty());
    }

    #[test]
    fn test_blend_channel() {
        assert!((blend_lut_channel(0.0, 1.0, 0.5) - 0.5).abs() < 1e-6);
        assert!((blend_lut_channel(0.2, 0.8, 0.0) - 0.2).abs() < 1e-6);
        assert!((blend_lut_channel(0.2, 0.8, 1.0) - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_blend_same_values() {
        // When original == lut_out, strength doesn't matter
        let v = [0.5_f32; 3];
        for &s in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            let result = blend_lut_result(v, v, s);
            assert!((result[0] - 0.5).abs() < 1e-6);
        }
    }
}
