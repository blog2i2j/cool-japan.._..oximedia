//! Butterfly-decomposed DCT-II / DCT-III for power-of-two block sizes.
//!
//! This module implements a fast O(N log N) DCT-II using the "via-2N-DFT" method:
//! the N-point DCT-II is computed as the real part of a 2N-point DFT of a
//! symmetrically extended sequence, scaled by appropriate twiddle factors.
//!
//! # Mathematical foundation
//!
//! Given `x[n]` for n = 0..N−1 the DCT-II sum is:
//!
//! ```text
//! S[k] = Σ_{n=0}^{N-1} x[n] cos(π(2n+1)k / (2N))
//! ```
//!
//! Construct the 2N-point symmetric sequence `y[n] = x[n]` and `y[2N-1-n] = x[n]`.
//! Its DFT satisfies:
//!
//! ```text
//! Y[k] = e^{-jπk/(2N)} · 2 S[k]
//! ```
//!
//! so `S[k] = Re(Y[k] · e^{+jπk/(2N)}) / 2`.
//!
//! The normalised DCT-II then applies the usual α(k) factor.
//! The normalised DCT-III (inverse) reverses these steps.
//!
//! # Public API
//!
//! - [`dct_butterfly_1d`]         — forward 1-D DCT-II (length N, power of 2).
//! - [`idct_butterfly_1d`]        — inverse 1-D (DCT-III, length N, power of 2).
//! - [`dct_butterfly_2d`]         — separable 2-D forward DCT-II (N×N block).
//! - [`idct_butterfly_2d`]        — separable 2-D inverse DCT-II (N×N block).
//! - [`forward_dct_butterfly_32`] — 32×32 i16 forward DCT.
//! - [`inverse_dct_butterfly_32`] — 32×32 i16 inverse DCT.
//! - [`forward_dct_butterfly_64`] — 64×64 i16 forward DCT.
//! - [`inverse_dct_butterfly_64`] — 64×64 i16 inverse DCT.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]

use crate::{Result, SimdError};
use std::f64::consts::PI;

// ── Complex number (f64) ──────────────────────────────────────────────────────

#[derive(Clone, Copy, Default)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    #[inline]
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
    #[inline]
    fn scale(self, s: f64) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }
    #[inline]
    fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
}

// ── Cooley–Tukey radix-2 DIT FFT ─────────────────────────────────────────────

/// In-place radix-2 DIT FFT.  `data.len()` must be a power of two.
/// When `inverse` is true, the output is divided by N (normalised IFFT).
fn fft_radix2(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    debug_assert!(n.is_power_of_two(), "FFT length must be a power of two");

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }

    // Butterfly stages
    let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
    let mut len = 2usize;
    while len <= n {
        let half = len >> 1;
        let ang = sign * PI / half as f64;
        let w_unit = Complex::new(ang.cos(), ang.sin());
        let mut k = 0usize;
        while k < n {
            let mut w = Complex::new(1.0, 0.0);
            for l in 0..half {
                let u = data[k + l];
                let v = data[k + l + half].mul(w);
                data[k + l] = u.add(v);
                data[k + l + half] = u.sub(v);
                w = w.mul(w_unit);
            }
            k += len;
        }
        len <<= 1;
    }

    if inverse {
        let inv_n = 1.0 / n as f64;
        for x in data.iter_mut() {
            *x = x.scale(inv_n);
        }
    }
}

// ── 1-D DCT-II via 2N-point DFT (correct derivation) ────────────────────────

/// Compute the un-normalised forward DCT-II sum for a slice.
///
/// Returns a vector of length `n` with coefficients
/// `S[k] = Σ_{i=0}^{N-1} src[i] cos(π(2i+1)k/(2N))`.
fn dct_ii_sums(src: &[f64]) -> Vec<f64> {
    let n = src.len();
    debug_assert!(n.is_power_of_two() && n >= 2);

    // Build 2N symmetric extension: y = [x[0], x[1], ..., x[N-1], x[N-1], x[N-2], ..., x[0]]
    let two_n = 2 * n;
    let mut y = vec![Complex::default(); two_n];
    for i in 0..n {
        y[i] = Complex::new(src[i], 0.0);
        y[two_n - 1 - i] = Complex::new(src[i], 0.0);
    }

    // Forward 2N-point FFT
    fft_radix2(&mut y, false);

    // Extract: Y[k] = e^{+jπk/(2N)} * 2 * S[k]  →  S[k] = Re(Y[k] * e^{-jπk/(2N)}) / 2
    (0..n)
        .map(|k| {
            let theta = -PI * k as f64 / (two_n as f64);
            let tw = Complex::new(theta.cos(), theta.sin());
            y[k].mul(tw).re / 2.0
        })
        .collect()
}

/// Apply the standard DCT-II normalisation to un-normalised sums.
fn apply_dct_normalisation(sums: &[f64]) -> Vec<f64> {
    let n = sums.len();
    let inv_sqrt_n = 1.0 / (n as f64).sqrt();
    let sqrt_2_over_n = (2.0 / n as f64).sqrt();
    (0..n)
        .map(|k| {
            if k == 0 {
                sums[k] * inv_sqrt_n
            } else {
                sums[k] * sqrt_2_over_n
            }
        })
        .collect()
}

/// Undo DCT-II normalisation (for the inverse path).
fn remove_dct_normalisation(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let sqrt_n = (n as f64).sqrt();
    let sqrt_n_over_2 = (n as f64 / 2.0).sqrt();
    (0..n)
        .map(|k| {
            if k == 0 {
                data[k] * sqrt_n
            } else {
                data[k] * sqrt_n_over_2
            }
        })
        .collect()
}

/// Compute un-normalised DCT-III (inverse DCT-II) from un-normalised DCT-II coefficients.
///
/// This is the inverse of `dct_ii_sums`: given `S[k]` it recovers `x[n]`.
fn idct_ii_from_sums(sums: &[f64]) -> Vec<f64> {
    let n = sums.len();
    debug_assert!(n.is_power_of_two() && n >= 2);

    // Reconstruct the 2N-point Hermitian DFT from the DCT-II sums.
    // Y[k] = e^{-jπk/(2N)} * 2 * S[k]   (from forward derivation)
    // The DFT of a real symmetric signal is Hermitian: Y[2N-k] = conj(Y[k]).
    let two_n = 2 * n;
    let mut y = vec![Complex::default(); two_n];
    // Y[k] = e^{+jπk/(2N)} * 2 * S[k]
    for k in 0..n {
        let theta = PI * k as f64 / two_n as f64;
        let tw = Complex::new(theta.cos(), theta.sin());
        let val = tw.scale(2.0 * sums[k]);
        y[k] = val;
        if k > 0 {
            // Hermitian symmetry for real input: Y[2N-k] = conj(Y[k])
            y[two_n - k] = val.conj();
        }
    }
    // y[N] = 0 (Nyquist; zero for the symmetric extension with real input)

    // Inverse 2N-point FFT
    fft_radix2(&mut y, true);

    // The IFFT result has the symmetric extension; first N samples = original x[n].
    // Scale: IFFT already divides by 2N, so multiply by 2N to undo, then divide by N
    // because the DCT-II forward path summed over N samples:
    // Actually: from IFFT of y we get back the 2N symmetric signal y_time,
    // where y_time[i] = x[i] for i < N (by construction). The IFFT normalises by
    // dividing by 2N, so the output is x[i] / (2N) * 2N = x[i]. But the way we
    // built Y, the IFFT should recover the 2N extended signal exactly.
    // However we set Y[N] = 0 which may introduce a Gibbs-like artifact.
    // For a symmetric sequence the Nyquist is always real and we need to include it.
    // Since the signal is symmetric, Y[N] is real; and since X_unnorm[N] = 0
    // (DCT-II only has coefficients for k=0..N-1), Y[N] = e^{-jπ/2}*2*0 = 0.
    // So leaving y[N]=0 is correct.
    (0..n).map(|i| y[i].re).collect()
}

// ── Public 1-D DCT API ────────────────────────────────────────────────────────

/// Compute the normalised forward DCT-II of `data` in-place using FFT butterflies.
///
/// `data.len()` must be a power of two ≥ 2.  On return `data[k]` holds the
/// normalised DCT-II coefficient with `α(0) = 1/√N` and `α(k>0) = √(2/N)`.
///
/// # Errors
///
/// Returns [`SimdError::UnsupportedOperation`] if the length is not a power of two or < 2.
pub fn dct_butterfly_1d(data: &mut [f64]) -> Result<()> {
    let n = data.len();
    if n < 2 || !n.is_power_of_two() {
        return Err(SimdError::UnsupportedOperation);
    }
    let sums = dct_ii_sums(data);
    let normed = apply_dct_normalisation(&sums);
    data.copy_from_slice(&normed);
    Ok(())
}

/// Compute the normalised inverse DCT (DCT-III) of `data` in-place.
///
/// Exact inverse of [`dct_butterfly_1d`].  `data.len()` must be a power of two ≥ 2.
///
/// # Errors
///
/// Returns [`SimdError::UnsupportedOperation`] if the length is not a power of two or < 2.
pub fn idct_butterfly_1d(data: &mut [f64]) -> Result<()> {
    let n = data.len();
    if n < 2 || !n.is_power_of_two() {
        return Err(SimdError::UnsupportedOperation);
    }
    // Undo normalisation → get un-normalised DCT-II coefficients
    let sums = remove_dct_normalisation(data);
    // Inverse DCT-II from sums
    let recovered = idct_ii_from_sums(&sums);
    data.copy_from_slice(&recovered);
    Ok(())
}

// ── 2-D separable DCT ─────────────────────────────────────────────────────────

/// Separable 2-D forward DCT-II for an `n × n` block of `f64`.
///
/// Applies the 1-D butterfly DCT to each row, then to each column.
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if `block.len() != n * n`.
/// Returns [`SimdError::UnsupportedOperation`] if `n` is not a power of two or `n < 2`.
pub fn dct_butterfly_2d(block: &mut [f64], n: usize) -> Result<()> {
    if block.len() != n * n {
        return Err(SimdError::InvalidBufferSize);
    }
    if n < 2 || !n.is_power_of_two() {
        return Err(SimdError::UnsupportedOperation);
    }

    let mut row = vec![0.0f64; n];
    for r in 0..n {
        row.copy_from_slice(&block[r * n..(r + 1) * n]);
        dct_butterfly_1d(&mut row)?;
        block[r * n..(r + 1) * n].copy_from_slice(&row);
    }

    let mut col = vec![0.0f64; n];
    for c in 0..n {
        for r in 0..n {
            col[r] = block[r * n + c];
        }
        dct_butterfly_1d(&mut col)?;
        for r in 0..n {
            block[r * n + c] = col[r];
        }
    }
    Ok(())
}

/// Separable 2-D inverse DCT-II for an `n × n` block of `f64`.
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if `block.len() != n * n`.
/// Returns [`SimdError::UnsupportedOperation`] if `n` is not a power of two or `n < 2`.
pub fn idct_butterfly_2d(block: &mut [f64], n: usize) -> Result<()> {
    if block.len() != n * n {
        return Err(SimdError::InvalidBufferSize);
    }
    if n < 2 || !n.is_power_of_two() {
        return Err(SimdError::UnsupportedOperation);
    }

    let mut row = vec![0.0f64; n];
    for r in 0..n {
        row.copy_from_slice(&block[r * n..(r + 1) * n]);
        idct_butterfly_1d(&mut row)?;
        block[r * n..(r + 1) * n].copy_from_slice(&row);
    }

    let mut col = vec![0.0f64; n];
    for c in 0..n {
        for r in 0..n {
            col[r] = block[r * n + c];
        }
        idct_butterfly_1d(&mut col)?;
        for r in 0..n {
            block[r * n + c] = col[r];
        }
    }
    Ok(())
}

// ── i16 convenience wrappers ──────────────────────────────────────────────────

/// Forward DCT-II for a 32×32 block of `i16` samples (O(N log N) butterfly path).
///
/// Equivalent to [`crate::forward_dct`] with [`crate::DctSize::Dct32x32`].
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if `src` or `dst` have fewer than 1024 elements.
pub fn forward_dct_butterfly_32(src: &[i16], dst: &mut [i16]) -> Result<()> {
    const N: usize = 32;
    const NN: usize = N * N;
    if src.len() < NN || dst.len() < NN {
        return Err(SimdError::InvalidBufferSize);
    }
    let mut block: Vec<f64> = src[..NN].iter().map(|&v| f64::from(v)).collect();
    dct_butterfly_2d(&mut block, N)?;
    for (d, &b) in dst[..NN].iter_mut().zip(block.iter()) {
        *d = b.round().clamp(-32768.0, 32767.0) as i16;
    }
    Ok(())
}

/// Inverse DCT-II for a 32×32 block of `i16` coefficients (butterfly path).
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if `src` or `dst` have fewer than 1024 elements.
pub fn inverse_dct_butterfly_32(src: &[i16], dst: &mut [i16]) -> Result<()> {
    const N: usize = 32;
    const NN: usize = N * N;
    if src.len() < NN || dst.len() < NN {
        return Err(SimdError::InvalidBufferSize);
    }
    let mut block: Vec<f64> = src[..NN].iter().map(|&v| f64::from(v)).collect();
    idct_butterfly_2d(&mut block, N)?;
    for (d, &b) in dst[..NN].iter_mut().zip(block.iter()) {
        *d = b.round().clamp(-32768.0, 32767.0) as i16;
    }
    Ok(())
}

/// Forward DCT-II for a 64×64 block of `i16` samples (butterfly path).
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if `src` or `dst` have fewer than 4096 elements.
pub fn forward_dct_butterfly_64(src: &[i16], dst: &mut [i16]) -> Result<()> {
    const N: usize = 64;
    const NN: usize = N * N;
    if src.len() < NN || dst.len() < NN {
        return Err(SimdError::InvalidBufferSize);
    }
    let mut block: Vec<f64> = src[..NN].iter().map(|&v| f64::from(v)).collect();
    dct_butterfly_2d(&mut block, N)?;
    for (d, &b) in dst[..NN].iter_mut().zip(block.iter()) {
        *d = b.round().clamp(-32768.0, 32767.0) as i16;
    }
    Ok(())
}

/// Inverse DCT-II for a 64×64 block of `i16` coefficients (butterfly path).
///
/// # Errors
///
/// Returns [`SimdError::InvalidBufferSize`] if `src` or `dst` have fewer than 4096 elements.
pub fn inverse_dct_butterfly_64(src: &[i16], dst: &mut [i16]) -> Result<()> {
    const N: usize = 64;
    const NN: usize = N * N;
    if src.len() < NN || dst.len() < NN {
        return Err(SimdError::InvalidBufferSize);
    }
    let mut block: Vec<f64> = src[..NN].iter().map(|&v| f64::from(v)).collect();
    idct_butterfly_2d(&mut block, N)?;
    for (d, &b) in dst[..NN].iter_mut().zip(block.iter()) {
        *d = b.round().clamp(-32768.0, 32767.0) as i16;
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── 1-D DCT-II correctness ───────────────────────────────────────────────

    #[test]
    fn dct_butterfly_1d_dc_input_gives_single_nonzero_coeff() {
        // Constant input → only DC (k=0) should be non-zero.
        // With normalisation: X[0] = (1/√N) · Σ c = (1/√N) · N·c = c·√N.
        let n = 8usize;
        let c = 64.0f64;
        let mut data = vec![c; n];
        dct_butterfly_1d(&mut data).expect("DCT butterfly 1D on constant input should succeed");
        let expected_dc = c * (n as f64).sqrt();
        assert!(
            (data[0] - expected_dc).abs() < 1e-6,
            "DC coefficient should be {expected_dc}, got {}",
            data[0]
        );
        for k in 1..n {
            assert!(
                data[k].abs() < 1e-8,
                "AC coeff[{k}] should be ~0, got {}",
                data[k]
            );
        }
    }

    #[test]
    fn dct_butterfly_1d_roundtrip_n8() {
        let original = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0_f64];
        let mut data = original;
        dct_butterfly_1d(&mut data).expect("forward DCT butterfly 1D n=8 should succeed");
        idct_butterfly_1d(&mut data).expect("inverse DCT butterfly 1D n=8 should succeed");
        for (i, (&orig, &rec)) in original.iter().zip(data.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-8,
                "1D roundtrip n=8 at {i}: orig={orig} rec={rec}"
            );
        }
    }

    #[test]
    fn dct_butterfly_1d_roundtrip_n16() {
        let original: Vec<f64> = (0..16).map(|i| (i as f64 * 7.5) - 50.0).collect();
        let mut data = original.clone();
        dct_butterfly_1d(&mut data).expect("forward DCT butterfly 1D n=16 should succeed");
        idct_butterfly_1d(&mut data).expect("inverse DCT butterfly 1D n=16 should succeed");
        for (i, (&orig, &rec)) in original.iter().zip(data.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-8,
                "1D roundtrip n=16 at {i}: orig={orig} rec={rec}"
            );
        }
    }

    #[test]
    fn dct_butterfly_1d_roundtrip_n32() {
        let original: Vec<f64> = (0..32).map(|i| (i * 7 % 100) as f64 - 50.0).collect();
        let mut data = original.clone();
        dct_butterfly_1d(&mut data).expect("forward DCT butterfly 1D n=32 should succeed");
        idct_butterfly_1d(&mut data).expect("inverse DCT butterfly 1D n=32 should succeed");
        for (i, (&orig, &rec)) in original.iter().zip(data.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-6,
                "1D roundtrip n=32 at {i}: orig={orig} rec={rec}"
            );
        }
    }

    #[test]
    fn dct_butterfly_1d_rejects_non_power_of_two() {
        let mut data = vec![0.0f64; 7];
        assert!(
            dct_butterfly_1d(&mut data).is_err(),
            "Should reject length 7"
        );
    }

    #[test]
    fn dct_butterfly_1d_rejects_length_one() {
        let mut data = vec![42.0f64];
        assert!(
            dct_butterfly_1d(&mut data).is_err(),
            "Should reject length 1"
        );
    }

    #[test]
    fn dct_butterfly_1d_zero_input_gives_zero_output() {
        let mut data = vec![0.0f64; 8];
        dct_butterfly_1d(&mut data).expect("DCT of zeros should succeed");
        for (k, &v) in data.iter().enumerate() {
            assert!(v.abs() < 1e-12, "coeff[{k}] should be 0, got {v}");
        }
    }

    #[test]
    fn idct_butterfly_1d_rejects_non_power_of_two() {
        let mut data = vec![0.0f64; 5];
        assert!(idct_butterfly_1d(&mut data).is_err());
    }

    // ── 1-D matches naive reference ──────────────────────────────────────────

    #[test]
    fn dct_butterfly_1d_matches_naive_n8() {
        let input = [5.0, 10.0, -3.0, 7.0, 2.0, -8.0, 1.0, 4.0_f64];
        let n = 8usize;

        // Naive O(N²) DCT-II reference
        let mut naive = vec![0.0f64; n];
        for k in 0..n {
            let norm = if k == 0 {
                1.0 / (n as f64).sqrt()
            } else {
                (2.0 / n as f64).sqrt()
            };
            let sum: f64 = (0..n)
                .map(|i| input[i] * (PI * (2 * i + 1) as f64 * k as f64 / (2 * n) as f64).cos())
                .sum();
            naive[k] = norm * sum;
        }

        let mut butterfly = input;
        dct_butterfly_1d(&mut butterfly).expect("DCT butterfly should succeed");

        for k in 0..n {
            assert!(
                (butterfly[k] - naive[k]).abs() < 1e-8,
                "k={k}: butterfly={} naive={}",
                butterfly[k],
                naive[k]
            );
        }
    }

    // ── 2-D DCT correctness ───────────────────────────────────────────────────

    #[test]
    fn dct_butterfly_2d_constant_block_gives_single_dc() {
        let n = 8usize;
        let c = 100.0f64;
        let mut block = vec![c; n * n];
        dct_butterfly_2d(&mut block, n).expect("2D butterfly DCT on constant block should succeed");
        // DC = c * sqrt(N) (from 1D row) then * sqrt(N) (from 1D col) = c * N
        let dc = block[0];
        assert!(dc.abs() > 1.0, "DC coefficient should be non-zero: {dc}");
        for (i, &v) in block.iter().enumerate().skip(1) {
            assert!(
                v.abs() < 1e-8,
                "Non-DC coeff at flat index {i} should be ~0, got {v}"
            );
        }
    }

    #[test]
    fn dct_butterfly_2d_roundtrip_n8() {
        let n = 8usize;
        let original: Vec<f64> = (0..n * n).map(|i| (i * 3 % 200) as f64 - 100.0).collect();
        let mut block = original.clone();
        dct_butterfly_2d(&mut block, n).expect("2D butterfly forward should succeed");
        idct_butterfly_2d(&mut block, n).expect("2D butterfly inverse should succeed");
        for (i, (&orig, &rec)) in original.iter().zip(block.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-8,
                "2D roundtrip n=8 at {i}: orig={orig} rec={rec}"
            );
        }
    }

    #[test]
    fn dct_butterfly_2d_rejects_wrong_buffer_size() {
        let mut block = vec![0.0f64; 63]; // 63 < 8×8 = 64
        assert!(dct_butterfly_2d(&mut block, 8).is_err());
    }

    // ── i16 32×32 wrapper ────────────────────────────────────────────────────

    #[test]
    fn forward_butterfly_32_roundtrip() {
        let input: Vec<i16> = (0..1024)
            .map(|i| ((i * 7 + 13) % 201) as i16 - 100)
            .collect();
        let mut coeffs = vec![0i16; 1024];
        let mut recovered = vec![0i16; 1024];
        forward_dct_butterfly_32(&input, &mut coeffs)
            .expect("forward DCT butterfly 32 should succeed");
        inverse_dct_butterfly_32(&coeffs, &mut recovered)
            .expect("inverse DCT butterfly 32 should succeed");
        for (i, (&orig, &rec)) in input.iter().zip(recovered.iter()).enumerate() {
            let diff = (i32::from(orig) - i32::from(rec)).abs();
            assert!(
                diff <= 2,
                "32×32 roundtrip at {i}: orig={orig} rec={rec} diff={diff}"
            );
        }
    }

    #[test]
    fn forward_butterfly_32_zero_input_gives_zero_output() {
        let input = vec![0i16; 1024];
        let mut coeffs = vec![99i16; 1024]; // non-zero init to detect bugs
        forward_dct_butterfly_32(&input, &mut coeffs)
            .expect("DCT butterfly 32 of zeros should succeed");
        for (i, &v) in coeffs.iter().enumerate() {
            assert_eq!(v, 0, "DCT of zeros: coeff[{i}] = {v}");
        }
    }

    #[test]
    fn forward_butterfly_32_buffer_too_small_returns_error() {
        let src = vec![0i16; 100]; // 100 < 1024
        let mut dst = vec![0i16; 1024];
        assert!(forward_dct_butterfly_32(&src, &mut dst).is_err());
    }

    #[test]
    fn forward_butterfly_32_matches_generic_scalar() {
        let input: Vec<i16> = (0..1024).map(|i| ((i * 3 + 7) % 101) as i16 - 50).collect();
        let mut out_butterfly = vec![0i16; 1024];
        let mut out_generic = vec![0i16; 1024];
        forward_dct_butterfly_32(&input, &mut out_butterfly)
            .expect("butterfly DCT 32 should succeed");
        crate::scalar::forward_dct_scalar(&input, &mut out_generic, crate::DctSize::Dct32x32)
            .expect("generic scalar DCT 32 should succeed");
        let max_diff = out_butterfly
            .iter()
            .zip(out_generic.iter())
            .map(|(&a, &b)| (i32::from(a) - i32::from(b)).abs())
            .max()
            .unwrap_or(0);
        assert!(
            max_diff <= 2,
            "Butterfly vs generic scalar max diff = {max_diff} (expected ≤ 2)"
        );
    }

    // ── i16 64×64 wrapper ────────────────────────────────────────────────────

    #[test]
    fn forward_butterfly_64_roundtrip() {
        // Small values to keep i16 headroom through 64×64 DCT
        let input: Vec<i16> = (0..4096).map(|i| ((i % 50) as i16) - 25).collect();
        let mut coeffs = vec![0i16; 4096];
        let mut recovered = vec![0i16; 4096];
        forward_dct_butterfly_64(&input, &mut coeffs)
            .expect("forward DCT butterfly 64 should succeed");
        inverse_dct_butterfly_64(&coeffs, &mut recovered)
            .expect("inverse DCT butterfly 64 should succeed");
        for (i, (&orig, &rec)) in input.iter().zip(recovered.iter()).enumerate() {
            let diff = (i32::from(orig) - i32::from(rec)).abs();
            assert!(
                diff <= 3,
                "64×64 roundtrip at {i}: orig={orig} rec={rec} diff={diff}"
            );
        }
    }

    #[test]
    fn forward_butterfly_64_buffer_too_small_returns_error() {
        let src = vec![0i16; 100];
        let mut dst = vec![0i16; 4096];
        assert!(forward_dct_butterfly_64(&src, &mut dst).is_err());
    }
}
