//! APV DCT transform and quantization.
//!
//! Implements the 8x8 Discrete Cosine Transform (Type-II) for the APV codec
//! using `f64` precision as specified by the APV standard. APV uses a slightly
//! different normalization factor than baseline JPEG — the orthonormal variant
//! where the DC scaling factor is `1/sqrt(2)` and the transform matrix is
//! self-inverse.
//!
//! # Quantization
//!
//! APV derives its quantization matrix from a flat base table scaled by the
//! quantization parameter (QP). The QP maps exponentially: each increment of 6
//! approximately doubles the quantization step size, similar to H.26x codecs.

use std::f64::consts::PI;

/// Standard 8x8 zigzag scan order (JPEG/MPEG standard).
///
/// Maps linear index 0..63 to the zigzag-ordered position within the 8x8 block.
pub const ZIGZAG_ORDER: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Inverse zigzag: given a position within the 8x8 block, returns the zigzag
/// scan index. Precomputed from [`ZIGZAG_ORDER`].
pub const INVERSE_ZIGZAG: [usize; 64] = compute_inverse_zigzag();

/// Compute the inverse zigzag table at compile time.
const fn compute_inverse_zigzag() -> [usize; 64] {
    let mut inv = [0usize; 64];
    let mut i = 0;
    while i < 64 {
        inv[ZIGZAG_ORDER[i]] = i;
        i += 1;
    }
    inv
}

/// APV base quantization matrix.
///
/// A flat matrix where all positions have the same base step size.
/// The actual quantization step for each coefficient is derived from this
/// base scaled by the QP-dependent factor. APV uses a flat base (unlike
/// JPEG which uses perceptually weighted tables) because the QP already
/// controls quality precisely.
const APV_BASE_QUANT: u16 = 16;

/// Apply the forward 8x8 DCT (Type-II, orthonormal) to a block of samples.
///
/// Input `block` contains 64 sample values in raster order (row-major).
/// After the call, `block` contains the 64 DCT coefficients.
///
/// The input samples are level-shifted by subtracting `dc_offset` (typically
/// 128 for 8-bit, 512 for 10-bit, etc.) before the transform.
pub fn forward_dct_8x8(block: &mut [f64; 64], dc_offset: f64) {
    // Level shift
    for v in block.iter_mut() {
        *v -= dc_offset;
    }

    // 1D DCT on rows
    for row in 0..8 {
        let start = row * 8;
        let end = start + 8;
        dct_1d_f64(&mut block[start..end]);
    }

    // 1D DCT on columns (extract column, transform, put back)
    let mut col_buf = [0.0f64; 8];
    for col in 0..8 {
        for row in 0..8 {
            col_buf[row] = block[row * 8 + col];
        }
        dct_1d_f64(&mut col_buf);
        for row in 0..8 {
            block[row * 8 + col] = col_buf[row];
        }
    }
}

/// Apply the inverse 8x8 DCT (Type-III, orthonormal) and level-unshift.
///
/// Input `block` contains 64 DCT coefficients. After the call, `block`
/// contains the reconstructed sample values with `dc_offset` added back.
pub fn inverse_dct_8x8(block: &mut [f64; 64], dc_offset: f64) {
    // 1D IDCT on rows
    for row in 0..8 {
        let start = row * 8;
        let end = start + 8;
        idct_1d_f64(&mut block[start..end]);
    }

    // 1D IDCT on columns
    let mut col_buf = [0.0f64; 8];
    for col in 0..8 {
        for row in 0..8 {
            col_buf[row] = block[row * 8 + col];
        }
        idct_1d_f64(&mut col_buf);
        for row in 0..8 {
            block[row * 8 + col] = col_buf[row];
        }
    }

    // Level unshift
    for v in block.iter_mut() {
        *v += dc_offset;
    }
}

/// 1D forward DCT (Type-II) on 8 values, orthonormal scaling.
fn dct_1d_f64(x: &mut [f64]) {
    let n = x.len() as f64;
    let mut out = [0.0f64; 8];
    for k in 0..8 {
        let ck = if k == 0 { 1.0 / 2.0_f64.sqrt() } else { 1.0 };
        let mut sum = 0.0f64;
        for (i, &xi) in x.iter().enumerate() {
            sum += xi * ((PI * k as f64 * (2.0 * i as f64 + 1.0)) / (2.0 * n)).cos();
        }
        out[k] = (2.0 / n).sqrt() * ck * sum;
    }
    x[..8].copy_from_slice(&out);
}

/// 1D inverse DCT (Type-III) on 8 values, orthonormal scaling.
fn idct_1d_f64(x: &mut [f64]) {
    let n = x.len() as f64;
    let mut out = [0.0f64; 8];
    for i in 0..8 {
        let mut sum = 0.0f64;
        for (k, &xk) in x.iter().enumerate() {
            let ck = if k == 0 { 1.0 / 2.0_f64.sqrt() } else { 1.0 };
            sum += ck * xk * ((PI * k as f64 * (2.0 * i as f64 + 1.0)) / (2.0 * n)).cos();
        }
        out[i] = (2.0 / n).sqrt() * sum;
    }
    x[..8].copy_from_slice(&out);
}

/// Compute the quantization step size for a given QP.
///
/// The QP-to-step mapping follows an exponential curve:
///   step = base * 2^((qp - 4) / 6)
///
/// This means every 6-step QP increment doubles the step size.
/// QP=0 yields the finest quantization (highest quality).
fn qp_to_step(qp: u8) -> f64 {
    let base = APV_BASE_QUANT as f64;
    base * 2.0_f64.powf((qp as f64 - 4.0) / 6.0)
}

/// Generate a 64-element quantization matrix for a given QP.
///
/// APV uses a flat quantization matrix where all coefficients share the
/// same base step, scaled by frequency-dependent weighting. DC and low
/// frequency coefficients get finer quantization (smaller steps) while
/// high frequency coefficients get coarser steps.
#[must_use]
pub fn generate_quant_matrix(qp: u8) -> [f64; 64] {
    let step = qp_to_step(qp);
    let mut matrix = [0.0f64; 64];

    for (i, val) in matrix.iter_mut().enumerate() {
        let row = i / 8;
        let col = i % 8;
        // Frequency-dependent scaling: higher frequency = larger quant step
        let freq_scale = 1.0 + 0.05 * (row + col) as f64;
        *val = (step * freq_scale).max(1.0);
    }

    matrix
}

/// Quantize a block of DCT coefficients using the given quantization matrix.
///
/// Returns the quantized integer coefficients (rounded to nearest).
pub fn quantize_block(coeffs: &[f64; 64], quant_matrix: &[f64; 64]) -> [i32; 64] {
    let mut quantized = [0i32; 64];
    for i in 0..64 {
        quantized[i] = (coeffs[i] / quant_matrix[i]).round() as i32;
    }
    quantized
}

/// Dequantize a block of integer coefficients back to f64 DCT coefficients.
pub fn dequantize_block(quantized: &[i32; 64], quant_matrix: &[f64; 64]) -> [f64; 64] {
    let mut coeffs = [0.0f64; 64];
    for i in 0..64 {
        coeffs[i] = quantized[i] as f64 * quant_matrix[i];
    }
    coeffs
}

/// Perform a zigzag scan on a raster-ordered 8x8 block.
///
/// Reorders coefficients from 2D raster order to 1D zigzag order,
/// which groups low-frequency coefficients first and tends to produce
/// long runs of zeros at the end.
pub fn zigzag_scan(block: &[i32; 64]) -> [i32; 64] {
    let mut scanned = [0i32; 64];
    for (i, &pos) in ZIGZAG_ORDER.iter().enumerate() {
        scanned[i] = block[pos];
    }
    scanned
}

/// Perform the inverse zigzag scan (zigzag order → raster order).
pub fn inverse_zigzag_scan(scanned: &[i32; 64]) -> [i32; 64] {
    let mut block = [0i32; 64];
    for (i, &pos) in ZIGZAG_ORDER.iter().enumerate() {
        block[pos] = scanned[i];
    }
    block
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for DCT round-trip reconstruction (f64 precision).
    const ROUNDTRIP_TOLERANCE: f64 = 1e-8;

    #[test]
    fn test_forward_dct_flat_block() {
        // A flat block (all same value) should produce only a DC coefficient.
        let val = 200.0;
        let dc_offset = 128.0;
        let mut block = [val; 64];
        forward_dct_8x8(&mut block, dc_offset);

        // DC coefficient should be (200 - 128) * 8 = 576 (with orthonormal scaling)
        let expected_dc = (val - dc_offset) * 8.0;
        assert!(
            (block[0] - expected_dc).abs() < 1e-6,
            "DC coefficient: expected {expected_dc}, got {}",
            block[0]
        );

        // All AC coefficients should be zero
        for (i, &c) in block.iter().enumerate().skip(1) {
            assert!(
                c.abs() < 1e-8,
                "AC coefficient [{i}] should be zero, got {c}"
            );
        }
    }

    #[test]
    fn test_forward_inverse_roundtrip() {
        let dc_offset = 128.0;
        // Create a known pattern
        let mut original = [0.0f64; 64];
        for i in 0..64 {
            original[i] = 50.0 + (i as f64 * 3.0) % 200.0;
        }

        let mut block = original;
        forward_dct_8x8(&mut block, dc_offset);
        inverse_dct_8x8(&mut block, dc_offset);

        for i in 0..64 {
            assert!(
                (block[i] - original[i]).abs() < ROUNDTRIP_TOLERANCE,
                "Mismatch at [{i}]: expected {}, got {} (diff {})",
                original[i],
                block[i],
                (block[i] - original[i]).abs()
            );
        }
    }

    #[test]
    fn test_forward_inverse_all_zeros() {
        let dc_offset = 128.0;
        let mut block = [128.0f64; 64]; // After level shift → all zeros
        forward_dct_8x8(&mut block, dc_offset);
        // All coefficients should be zero
        for &c in &block {
            assert!(c.abs() < 1e-10);
        }
        inverse_dct_8x8(&mut block, dc_offset);
        // Should reconstruct to 128
        for &v in &block {
            assert!((v - 128.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_forward_inverse_checkerboard() {
        let dc_offset = 128.0;
        let mut original = [0.0f64; 64];
        for i in 0..64 {
            let row = i / 8;
            let col = i % 8;
            original[i] = if (row + col) % 2 == 0 { 200.0 } else { 50.0 };
        }

        let mut block = original;
        forward_dct_8x8(&mut block, dc_offset);
        inverse_dct_8x8(&mut block, dc_offset);

        for i in 0..64 {
            assert!(
                (block[i] - original[i]).abs() < ROUNDTRIP_TOLERANCE,
                "Checkerboard mismatch at [{i}]"
            );
        }
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let dc_offset = 128.0;
        let mut block = [0.0f64; 64];
        for i in 0..64 {
            block[i] = 100.0 + (i as f64 * 2.5) % 150.0;
        }

        forward_dct_8x8(&mut block, dc_offset);
        let quant_matrix = generate_quant_matrix(0); // Finest quantization
        let quantized = quantize_block(&block, &quant_matrix);
        let dequantized = dequantize_block(&quantized, &quant_matrix);
        inverse_dct_8x8(&mut block, dc_offset);

        // With QP=0 (finest), reconstruction should be very close
        let mut recon = dequantized;
        inverse_dct_8x8(&mut recon, dc_offset);

        // Check reconstruction is within a few sample values
        let original_block: Vec<f64> = (0..64).map(|i| 100.0 + (i as f64 * 2.5) % 150.0).collect();
        for i in 0..64 {
            let diff = (recon[i] - original_block[i]).abs();
            assert!(
                diff < 20.0,
                "Quant roundtrip too lossy at [{i}]: diff={diff}"
            );
        }
    }

    #[test]
    fn test_quant_matrix_qp_scaling() {
        let q0 = generate_quant_matrix(0);
        let q30 = generate_quant_matrix(30);
        let q63 = generate_quant_matrix(63);

        // Higher QP should produce larger quantization steps
        for i in 0..64 {
            assert!(
                q30[i] > q0[i],
                "QP=30 step should be larger than QP=0 at [{i}]"
            );
            assert!(
                q63[i] > q30[i],
                "QP=63 step should be larger than QP=30 at [{i}]"
            );
        }
    }

    #[test]
    fn test_quant_matrix_frequency_scaling() {
        let matrix = generate_quant_matrix(22);
        // DC (position [0][0]) should have smaller step than high-freq corner [7][7]
        assert!(
            matrix[0] < matrix[63],
            "DC step {} should be less than HF step {}",
            matrix[0],
            matrix[63]
        );
    }

    #[test]
    fn test_zigzag_scan_identity() {
        // If we zigzag and then inverse-zigzag, we should get back the original
        let mut block = [0i32; 64];
        for i in 0..64 {
            block[i] = i as i32;
        }

        let scanned = zigzag_scan(&block);
        let restored = inverse_zigzag_scan(&scanned);
        assert_eq!(block, restored);
    }

    #[test]
    fn test_zigzag_order_first_elements() {
        // First few zigzag positions should be: DC, (0,1), (1,0), (2,0), (1,1), (0,2)
        assert_eq!(ZIGZAG_ORDER[0], 0); // (0,0) = DC
        assert_eq!(ZIGZAG_ORDER[1], 1); // (0,1)
        assert_eq!(ZIGZAG_ORDER[2], 8); // (1,0)
        assert_eq!(ZIGZAG_ORDER[3], 16); // (2,0)
        assert_eq!(ZIGZAG_ORDER[4], 9); // (1,1)
        assert_eq!(ZIGZAG_ORDER[5], 2); // (0,2)
    }

    #[test]
    fn test_zigzag_scan_zeros_at_end() {
        // Block with only DC and a few low-frequency coefficients should
        // have zeros grouped at the end after zigzag scanning.
        let mut block = [0i32; 64];
        block[0] = 100; // DC
        block[1] = 10; // (0,1)
        block[8] = 5; // (1,0)

        let scanned = zigzag_scan(&block);
        assert_eq!(scanned[0], 100); // DC is first in zigzag
        assert_eq!(scanned[1], 10); // (0,1)
        assert_eq!(scanned[2], 5); // (1,0)
                                   // Everything else should be zero
        for &v in scanned.iter().skip(3) {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn test_inverse_zigzag_table() {
        // Verify inverse_zigzag is indeed the inverse of zigzag_order
        for i in 0..64 {
            assert_eq!(INVERSE_ZIGZAG[ZIGZAG_ORDER[i]], i);
        }
    }

    #[test]
    fn test_quantize_block_zero_input() {
        let coeffs = [0.0f64; 64];
        let quant_matrix = generate_quant_matrix(22);
        let quantized = quantize_block(&coeffs, &quant_matrix);
        for &v in &quantized {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn test_dequantize_block_zero_input() {
        let quantized = [0i32; 64];
        let quant_matrix = generate_quant_matrix(22);
        let dequantized = dequantize_block(&quantized, &quant_matrix);
        for &v in &dequantized {
            assert!((v - 0.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_dct_energy_conservation() {
        // Parseval's theorem: energy in spatial domain ≈ energy in frequency domain
        let dc_offset = 128.0;
        let mut block = [0.0f64; 64];
        for i in 0..64 {
            block[i] = 50.0 + (i as f64 * 7.0) % 200.0;
        }

        // Compute spatial energy (after level shift)
        let spatial_energy: f64 = block.iter().map(|&v| (v - dc_offset).powi(2)).sum();

        forward_dct_8x8(&mut block, dc_offset);

        // Compute frequency energy
        let freq_energy: f64 = block.iter().map(|&v| v.powi(2)).sum();

        // Should be approximately equal (orthonormal DCT preserves energy)
        let ratio = spatial_energy / freq_energy;
        assert!(
            (ratio - 1.0).abs() < 1e-8,
            "Energy ratio: {ratio} (should be ~1.0)"
        );
    }

    #[test]
    fn test_qp_step_monotonic() {
        let mut prev = qp_to_step(0);
        for qp in 1..=63 {
            let current = qp_to_step(qp);
            assert!(
                current > prev,
                "QP step should increase: QP={qp}, prev={prev}, current={current}"
            );
            prev = current;
        }
    }

    #[test]
    fn test_qp_step_doubling() {
        // Every 6 QP steps should approximately double the step size
        let s0 = qp_to_step(10);
        let s6 = qp_to_step(16);
        let ratio = s6 / s0;
        assert!(
            (ratio - 2.0).abs() < 0.01,
            "6-step QP ratio should be ~2.0, got {ratio}"
        );
    }

    #[test]
    fn test_forward_dct_gradient() {
        let dc_offset = 128.0;
        // Horizontal gradient: values increase left to right
        let mut block = [0.0f64; 64];
        for i in 0..64 {
            let col = (i % 8) as f64;
            block[i] = 100.0 + col * 20.0;
        }

        let original = block;
        forward_dct_8x8(&mut block, dc_offset);

        // DC should be non-zero
        assert!(block[0].abs() > 1.0);

        // Roundtrip
        inverse_dct_8x8(&mut block, dc_offset);
        for i in 0..64 {
            assert!(
                (block[i] - original[i]).abs() < ROUNDTRIP_TOLERANCE,
                "Gradient mismatch at [{i}]"
            );
        }
    }
}
