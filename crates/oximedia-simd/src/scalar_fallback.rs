//! Pure-scalar fallback implementations for all performance-critical SIMD kernels.
//!
//! These implementations are used:
//! * when the CPU does not support the required SIMD extension (detected via
//!   [`crate::detect_cpu_features`]), and
//! * as reference implementations in tests to verify correctness of any future
//!   SIMD-accelerated variants.
//!
//! Every function in this module is intentionally straightforward — no clever
//! tricks — so that the behaviour is easy to reason about and review.

use crate::SimdError;

// ─── Arithmetic ───────────────────────────────────────────────────────────────

/// Add corresponding `f32` elements of two equal-length slices into `out`.
///
/// `out[i] = a[i] + b[i]`
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if the slices have different lengths
/// or `out` is not large enough.
pub fn scalar_add_f32(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<(), SimdError> {
    if a.len() != b.len() || out.len() < a.len() {
        return Err(SimdError::InvalidBufferSize);
    }
    for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
        out[i] = ai + bi;
    }
    Ok(())
}

/// Multiply corresponding `f32` elements of two equal-length slices into `out`.
///
/// `out[i] = a[i] * b[i]`
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if the slices have different lengths
/// or `out` is not large enough.
pub fn scalar_mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<(), SimdError> {
    if a.len() != b.len() || out.len() < a.len() {
        return Err(SimdError::InvalidBufferSize);
    }
    for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
        out[i] = ai * bi;
    }
    Ok(())
}

/// Compute the dot product of two equal-length `f32` slices.
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if the slices have different lengths.
pub fn scalar_dot_f32(a: &[f32], b: &[f32]) -> Result<f32, SimdError> {
    if a.len() != b.len() {
        return Err(SimdError::InvalidBufferSize);
    }
    let mut sum = 0.0f32;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        sum += ai * bi;
    }
    Ok(sum)
}

// ─── Pixel operations ────────────────────────────────────────────────────────

/// Clamp each element of `src` to `[lo, hi]` and write to `dst`.
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if `dst` is shorter than `src`.
pub fn scalar_clamp_u8(src: &[u8], dst: &mut [u8], lo: u8, hi: u8) -> Result<(), SimdError> {
    if dst.len() < src.len() {
        return Err(SimdError::InvalidBufferSize);
    }
    for (i, &s) in src.iter().enumerate() {
        dst[i] = s.clamp(lo, hi);
    }
    Ok(())
}

/// Alpha-blend two RGBA buffers: `out[i] = a[i] * alpha + b[i] * (1 - alpha)`.
///
/// `alpha` is in `[0.0, 1.0]`.
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if buffer lengths differ or `out`
/// is too short.
pub fn scalar_alpha_blend_rgba(
    a: &[u8],
    b: &[u8],
    out: &mut [u8],
    alpha: f32,
) -> Result<(), SimdError> {
    if a.len() != b.len() || out.len() < a.len() {
        return Err(SimdError::InvalidBufferSize);
    }
    let alpha = alpha.clamp(0.0, 1.0);
    let inv = 1.0 - alpha;
    for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
        out[i] = (ai as f32 * alpha + bi as f32 * inv).round() as u8;
    }
    Ok(())
}

// ─── SAD (Sum of Absolute Differences) ───────────────────────────────────────

/// Compute the SAD between two equal-length `u8` slices.
///
/// `result = sum |a[i] - b[i]|`
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if the slices have different lengths.
pub fn scalar_sad_u8(a: &[u8], b: &[u8]) -> Result<u32, SimdError> {
    if a.len() != b.len() {
        return Err(SimdError::InvalidBufferSize);
    }
    let mut sum = 0u32;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        sum += ai.abs_diff(bi) as u32;
    }
    Ok(sum)
}

// ─── YUV → RGBA conversion ────────────────────────────────────────────────────

/// Convert a YUV 4:2:0 planar frame to RGBA (BT.601 full-range).
///
/// This is a thin wrapper around [`crate::yuv_rgb::yuv420_to_rgba`] exposed
/// here as the canonical scalar fallback entry point.
#[must_use]
pub fn scalar_yuv420_to_rgba(y: &[u8], u: &[u8], v: &[u8], w: u32, h: u32) -> Vec<u8> {
    crate::yuv_rgb::yuv420_to_rgba(y, u, v, w, h)
}

// ─── Histogram computation ────────────────────────────────────────────────────

/// Compute a 256-bucket histogram of a luma buffer.
///
/// This is the simplest possible implementation — one bucket increment per
/// byte — used as a reference and verified against the unrolled version in
/// [`crate::hist_simd`].
#[must_use]
pub fn scalar_histogram(luma: &[u8]) -> [u32; 256] {
    let mut hist = [0u32; 256];
    for &b in luma {
        hist[b as usize] += 1;
    }
    hist
}

// ─── Motion blur / convolution helpers ───────────────────────────────────────

/// Apply a 1-D box blur of width `kernel_size` to a `f32` signal.
///
/// `kernel_size` must be odd and ≥ 1; out-of-bounds positions are clamped.
/// The output has the same length as the input.
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if `kernel_size` is 0 or even.
pub fn scalar_box_blur_f32(src: &[f32], kernel_size: usize) -> Result<Vec<f32>, SimdError> {
    if kernel_size == 0 || kernel_size % 2 == 0 {
        return Err(SimdError::InvalidBufferSize);
    }
    let n = src.len();
    let half = (kernel_size / 2) as i64;
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        let mut sum = 0.0f32;
        for k in -half..=half {
            let j = (i as i64 + k).clamp(0, n as i64 - 1) as usize;
            sum += src[j];
        }
        out.push(sum / kernel_size as f32);
    }
    Ok(out)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── scalar_add_f32 ────────────────────────────────────────────────────────

    #[test]
    fn add_f32_basic() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        let mut out = [0.0f32; 3];
        scalar_add_f32(&a, &b, &mut out).expect("ok");
        assert_eq!(out, [5.0, 7.0, 9.0]);
    }

    #[test]
    fn add_f32_length_mismatch() {
        let a = [1.0f32];
        let b = [1.0f32, 2.0];
        let mut out = [0.0f32; 2];
        assert!(scalar_add_f32(&a, &b, &mut out).is_err());
    }

    // ── scalar_mul_f32 ────────────────────────────────────────────────────────

    #[test]
    fn mul_f32_basic() {
        let a = [2.0f32, 3.0];
        let b = [4.0f32, 5.0];
        let mut out = [0.0f32; 2];
        scalar_mul_f32(&a, &b, &mut out).expect("ok");
        assert_eq!(out, [8.0, 15.0]);
    }

    // ── scalar_dot_f32 ────────────────────────────────────────────────────────

    #[test]
    fn dot_f32_basic() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        let dot = scalar_dot_f32(&a, &b).expect("ok");
        assert!((dot - 32.0).abs() < 1e-5);
    }

    #[test]
    fn dot_f32_empty() {
        let dot = scalar_dot_f32(&[], &[]).expect("ok");
        assert!(dot.abs() < 1e-9);
    }

    // ── scalar_clamp_u8 ───────────────────────────────────────────────────────

    #[test]
    fn clamp_u8_basic() {
        let src = [0u8, 64, 128, 192, 255];
        let mut dst = [0u8; 5];
        scalar_clamp_u8(&src, &mut dst, 64, 192).expect("ok");
        assert_eq!(dst, [64, 64, 128, 192, 192]);
    }

    // ── scalar_alpha_blend_rgba ───────────────────────────────────────────────

    #[test]
    fn alpha_blend_full_a() {
        let a = [200u8, 100, 50, 255];
        let b = [0u8, 0, 0, 255];
        let mut out = [0u8; 4];
        scalar_alpha_blend_rgba(&a, &b, &mut out, 1.0).expect("ok");
        assert_eq!(out, [200, 100, 50, 255]);
    }

    #[test]
    fn alpha_blend_full_b() {
        let a = [200u8, 100, 50, 255];
        let b = [10u8, 20, 30, 255];
        let mut out = [0u8; 4];
        scalar_alpha_blend_rgba(&a, &b, &mut out, 0.0).expect("ok");
        assert_eq!(out, [10, 20, 30, 255]);
    }

    // ── scalar_sad_u8 ─────────────────────────────────────────────────────────

    #[test]
    fn sad_u8_identical_is_zero() {
        let a = [1u8, 2, 3, 4, 5];
        assert_eq!(scalar_sad_u8(&a, &a).expect("ok"), 0);
    }

    #[test]
    fn sad_u8_known_value() {
        let a = [0u8; 4];
        let b = [10u8; 4];
        assert_eq!(scalar_sad_u8(&a, &b).expect("ok"), 40);
    }

    // ── scalar_histogram ──────────────────────────────────────────────────────

    #[test]
    fn scalar_histogram_matches_unrolled() {
        use crate::hist_simd::compute_histogram_fast;
        let data: Vec<u8> = (0..1000u16).map(|i| (i % 256) as u8).collect();
        let scalar = scalar_histogram(&data);
        let fast = compute_histogram_fast(&data);
        assert_eq!(scalar, fast);
    }

    // ── scalar_yuv420_to_rgba ─────────────────────────────────────────────────

    #[test]
    fn scalar_yuv420_output_size() {
        let w = 4u32;
        let h = 4u32;
        let y = vec![128u8; (w * h) as usize];
        let u = vec![128u8; (w / 2 * h / 2) as usize];
        let v = vec![128u8; (w / 2 * h / 2) as usize];
        let out = scalar_yuv420_to_rgba(&y, &u, &v, w, h);
        assert_eq!(out.len(), (w * h * 4) as usize);
    }

    // ── scalar_box_blur_f32 ───────────────────────────────────────────────────

    #[test]
    fn box_blur_flat_signal_unchanged() {
        let src = vec![5.0f32; 10];
        let out = scalar_box_blur_f32(&src, 3).expect("ok");
        for v in &out {
            assert!((v - 5.0).abs() < 1e-5);
        }
    }

    #[test]
    fn box_blur_even_kernel_errors() {
        let src = vec![1.0f32; 10];
        assert!(scalar_box_blur_f32(&src, 2).is_err());
    }
}
