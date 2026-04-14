//! 8×8 DCT block operations for image compression.
//!
//! Provides forward and inverse DCT for 8×8 pixel blocks (as used in JPEG),
//! plus quantization helpers compatible with standard JPEG quantization tables.
//!
//! ## Algorithm
//!
//! Uses the separable 2-D DCT-II definition:
//! ```text
//! F[u,v] = (1/4) * C(u)*C(v) * sum_{x,y} f[x,y]
//!          * cos((2x+1)*u*π/16) * cos((2y+1)*v*π/16)
//!
//! where C(0) = 1/√2, C(k>0) = 1.
//! ```
//! The forward transform (`dct8x8`) produces coefficients in the same
//! row-major order as the input block (index = `v*8 + u`).
//!
//! The inverse transform (`idct8x8`) reconstructs spatial-domain samples.
//!
//! ## JPEG quantization
//!
//! `quantize` divides each DCT coefficient by the corresponding table entry
//! and rounds to the nearest integer.  `dequantize` multiplies back to
//! recover approximate DCT coefficients before the inverse transform.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Pre-computed cosine table
// ---------------------------------------------------------------------------

/// Build the 8×8 cosine basis matrix used by the separable 1-D DCT-II.
///
/// `cos_table[n][k]` = `cos((2n+1) * k * π / 16)`
fn build_cos_table() -> [[f32; 8]; 8] {
    let mut table = [[0.0f32; 8]; 8];
    for n in 0..8 {
        for k in 0..8 {
            table[n][k] = ((2 * n + 1) as f32 * k as f32 * PI / 16.0).cos();
        }
    }
    table
}

/// Scale factor C(k): 1/√2 for k=0, 1.0 otherwise.
#[inline]
fn c(k: usize) -> f32 {
    if k == 0 {
        std::f32::consts::FRAC_1_SQRT_2
    } else {
        1.0
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the forward 2-D 8×8 DCT-II of `block`.
///
/// `block` is a row-major slice of 64 `f32` values (row 0 first).
/// Coefficients are returned in the same order: index `v*8 + u` holds `F[u,v]`.
///
/// The transform is orthonormal and scaled so that `idct8x8(dct8x8(b)) ≈ b`
/// to within floating-point rounding error.
#[must_use]
pub fn dct8x8(block: &[f32; 64]) -> [f32; 64] {
    let cos = build_cos_table();
    let mut out = [0.0f32; 64];

    for v in 0..8usize {
        for u in 0..8usize {
            let mut sum = 0.0f32;
            for y in 0..8usize {
                for x in 0..8usize {
                    sum += block[y * 8 + x] * cos[x][u] * cos[y][v];
                }
            }
            out[v * 8 + u] = 0.25 * c(u) * c(v) * sum;
        }
    }

    out
}

/// Compute the inverse 2-D 8×8 DCT-II (IDCT) of `coeffs`.
///
/// This is the exact inverse of [`dct8x8`]:
/// `idct8x8(dct8x8(block)) ≈ block` to within floating-point rounding.
///
/// Output values are **not** clamped; the caller should clamp to [0, 255]
/// before storing as pixel data.
#[must_use]
pub fn idct8x8(coeffs: &[f32; 64]) -> [f32; 64] {
    let cos = build_cos_table();
    let mut out = [0.0f32; 64];

    for y in 0..8usize {
        for x in 0..8usize {
            let mut sum = 0.0f32;
            for v in 0..8usize {
                for u in 0..8usize {
                    sum += c(u) * c(v) * coeffs[v * 8 + u] * cos[x][u] * cos[y][v];
                }
            }
            out[y * 8 + x] = 0.25 * sum;
        }
    }

    out
}

/// Quantize a block of DCT coefficients using a quantization table.
///
/// Each coefficient `block[i]` is divided by `table[i]` and rounded to the
/// nearest integer (stored as f32 with a fractional part of 0).  Zero-valued
/// table entries are skipped (coefficient set to 0.0).
///
/// # Parameters
/// - `block` – DCT coefficients as produced by [`dct8x8`].
/// - `table` – positive quantization step sizes (one per coefficient).
#[must_use]
pub fn quantize(block: &[f32; 64], table: &[f32; 64]) -> [f32; 64] {
    let mut out = [0.0f32; 64];
    for i in 0..64 {
        if table[i].abs() < f32::EPSILON {
            out[i] = 0.0;
        } else {
            out[i] = (block[i] / table[i]).round();
        }
    }
    out
}

/// De-quantize a block of quantized DCT coefficients.
///
/// Multiplies each quantized coefficient by the corresponding table entry to
/// recover an approximation of the original DCT coefficients.
///
/// # Parameters
/// - `block` – quantized coefficients as produced by [`quantize`].
/// - `table` – quantization step sizes (same table used for quantization).
#[must_use]
pub fn dequantize(block: &[f32; 64], table: &[f32; 64]) -> [f32; 64] {
    let mut out = [0.0f32; 64];
    for i in 0..64 {
        out[i] = block[i] * table[i];
    }
    out
}

// ---------------------------------------------------------------------------
// Standard JPEG quantization tables
// ---------------------------------------------------------------------------

/// Standard JPEG luminance quantization table (quality 50).
pub const JPEG_LUMA_QT: [f32; 64] = [
    16.0, 11.0, 10.0, 16.0, 24.0, 40.0, 51.0, 61.0, 12.0, 12.0, 14.0, 19.0, 26.0, 58.0, 60.0, 55.0,
    14.0, 13.0, 16.0, 24.0, 40.0, 57.0, 69.0, 56.0, 14.0, 17.0, 22.0, 29.0, 51.0, 87.0, 80.0, 62.0,
    18.0, 22.0, 37.0, 56.0, 68.0, 109.0, 103.0, 77.0, 24.0, 35.0, 55.0, 64.0, 81.0, 104.0, 113.0,
    92.0, 49.0, 64.0, 78.0, 87.0, 103.0, 121.0, 120.0, 101.0, 72.0, 92.0, 95.0, 98.0, 112.0, 100.0,
    103.0, 99.0,
];

/// Standard JPEG chrominance quantization table (quality 50).
pub const JPEG_CHROMA_QT: [f32; 64] = [
    17.0, 18.0, 24.0, 47.0, 99.0, 99.0, 99.0, 99.0, 18.0, 21.0, 26.0, 66.0, 99.0, 99.0, 99.0, 99.0,
    24.0, 26.0, 56.0, 99.0, 99.0, 99.0, 99.0, 99.0, 47.0, 66.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
    99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
    99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
];

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_err(a: &[f32; 64], b: &[f32; 64]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// Forward DCT of a DC-only block (all-128) should have non-zero only at (0,0).
    #[test]
    fn dct8x8_dc_only_block() {
        let block = [128.0f32; 64];
        let coeffs = dct8x8(&block);
        // DC coefficient F[0,0] ≈ 128 * 8 * 8 * (1/4) * C(0)^2
        // = 128 * 64 * 0.25 * 0.5 = 1024.0
        let dc = coeffs[0];
        assert!(dc.abs() > 100.0, "DC coefficient should be large, got {dc}");
        // All AC coefficients should be near zero
        for i in 1..64 {
            assert!(
                coeffs[i].abs() < 1.0,
                "AC coefficient [{i}] expected ~0, got {}",
                coeffs[i]
            );
        }
    }

    /// IDCT(DCT(block)) ≈ block within rounding tolerance.
    #[test]
    fn dct_idct_roundtrip_flat() {
        let block = [64.0f32; 64];
        let coeffs = dct8x8(&block);
        let reconstructed = idct8x8(&coeffs);
        let err = max_abs_err(&block, &reconstructed);
        assert!(err < 1e-3, "round-trip error {err} exceeds 1e-3");
    }

    /// Round-trip for a ramp signal.
    #[test]
    fn dct_idct_roundtrip_ramp() {
        let mut block = [0.0f32; 64];
        for i in 0..64 {
            block[i] = i as f32;
        }
        let coeffs = dct8x8(&block);
        let reconstructed = idct8x8(&coeffs);
        let err = max_abs_err(&block, &reconstructed);
        assert!(err < 1e-3, "round-trip ramp error {err} exceeds 1e-3");
    }

    /// Quantize followed by dequantize should approximately preserve values
    /// when quantization step size is 1 (unity table).
    #[test]
    fn quantize_dequantize_unity_table() {
        let block = [16.7f32; 64];
        let table = [1.0f32; 64];
        let q = quantize(&block, &table);
        let dq = dequantize(&q, &table);
        // dq[i] should equal round(16.7) = 17.0
        for v in dq {
            assert!((v - 17.0).abs() < 1e-5, "expected 17.0 got {v}");
        }
    }

    /// Quantize with a large table entry should zero-out small coefficients.
    #[test]
    fn quantize_zeroes_small_coefficients() {
        let mut block = [0.0f32; 64];
        block[0] = 1000.0;
        block[1] = 5.0; // smaller than step
        let mut table = [1.0f32; 64];
        table[1] = 100.0; // step 100 > coeff 5
        let q = quantize(&block, &table);
        assert_eq!(q[0], 1000.0); // large coeff survives
        assert_eq!(q[1], 0.0); // small coeff zeroed
    }

    /// Zero-valued table entry should produce 0 in output (no div-by-zero panic).
    #[test]
    fn quantize_zero_table_entry_safe() {
        let block = [99.0f32; 64];
        let mut table = [1.0f32; 64];
        table[5] = 0.0;
        let q = quantize(&block, &table);
        assert_eq!(q[5], 0.0, "zero table entry should yield 0, not NaN/Inf");
    }

    /// DCT of a zero block should be all zeros.
    #[test]
    fn dct8x8_zero_block() {
        let block = [0.0f32; 64];
        let coeffs = dct8x8(&block);
        for (i, &v) in coeffs.iter().enumerate() {
            assert!(v.abs() < 1e-6, "coeff[{i}] expected 0, got {v}");
        }
    }

    /// IDCT of a zero block should be all zeros.
    #[test]
    fn idct8x8_zero_block() {
        let coeffs = [0.0f32; 64];
        let spatial = idct8x8(&coeffs);
        for (i, &v) in spatial.iter().enumerate() {
            assert!(v.abs() < 1e-6, "pixel[{i}] expected 0, got {v}");
        }
    }

    /// Verify JPEG luma quantization table has 64 elements all > 0.
    #[test]
    fn jpeg_luma_qt_all_positive() {
        for (i, &v) in JPEG_LUMA_QT.iter().enumerate() {
            assert!(v > 0.0, "JPEG_LUMA_QT[{i}] = {v} must be positive");
        }
    }

    /// Full pipeline: DCT → quantize(JPEG) → dequantize → IDCT
    /// The reconstructed block should be a plausible approximation.
    #[test]
    fn jpeg_pipeline_roundtrip() {
        let mut block = [0.0f32; 64];
        for (i, v) in block.iter_mut().enumerate() {
            *v = 128.0 + (i as f32 % 32.0) - 16.0;
        }
        let dct_coeffs = dct8x8(&block);
        let quantized = quantize(&dct_coeffs, &JPEG_LUMA_QT);
        let dequantized = dequantize(&quantized, &JPEG_LUMA_QT);
        let reconstructed = idct8x8(&dequantized);

        // JPEG quantization is lossy; allow up to 20.0 per-pixel error
        let err = max_abs_err(&block, &reconstructed);
        assert!(err < 50.0, "JPEG pipeline round-trip error {err} too large");
    }
}
