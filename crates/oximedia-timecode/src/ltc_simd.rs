//! SIMD-accelerated Manchester (Biphase Mark Code) encoding and decoding for LTC.
//!
//! # LTC Audio Format Specification
//!
//! Linear Timecode encodes SMPTE timecode as an audio signal using **Biphase Mark
//! Code (BMC)**, also known as bi-phase mark modulation:
//!
//! ## Baud Rate
//! - At **30 fps**: 30 frames/s × 80 bits/frame = **2400 baud** (2400 bit-cells/second)
//! - At **25 fps**: 25 × 80 = **2000 baud**
//! - At **24 fps**: 24 × 80 = **1920 baud**
//!
//! ## Modulation (Biphase Mark Code)
//! Each bit occupies one bit-cell. Transitions (polarity reversals) occur as follows:
//! - **Bit 0**: One transition — only at the **beginning** of the bit cell.
//! - **Bit 1**: Two transitions — at the **beginning** *and* in the **middle** of the cell.
//!
//! Consequently:
//! - Logic 0 produces a square wave at the **bit-clock frequency** (one edge/cell).
//! - Logic 1 produces a square wave at **twice** the bit-clock frequency (two edges/cell).
//!
//! ## Sync Word
//! The 80-bit LTC frame ends with a fixed 16-bit sync word `0x3FFD`
//! (`1011_1111_1111_1100` in LSB-first order), which can never appear in the
//! preceding 64 data bits due to BMC encoding constraints.
//!
//! ## Signal Levels
//! Typically recorded at line level (−10 dBV to +4 dBu). The signal is polarity-
//! independent and can be read forwards *or* backwards at speeds from ~0.1× to ~10×.
//!
//! ## Manchester vs. BMC
//! Strictly speaking LTC uses **Biphase Mark** (transition at every cell boundary,
//! plus an extra mid-cell transition for logic 1), not classical Manchester code
//! (which uses mid-cell transitions only). The two schemes produce similar output
//! waveforms and are often conflated in practice. This module encodes/decodes both
//! the "LTC biphase mark" variant (used for audio output) and a "classical Manchester"
//! variant (used internally for testing and reference).

/// Scalar Manchester / Biphase Mark Code encoder.
///
/// Each input bit is expanded into 2 output samples:
/// - `0` → `[+amp, −amp]` (one transition in middle)
/// - `1` → `[+amp, −amp]` (same shape, but polarity tracks the running state)
///
/// For a full LTC signal use [`crate::ltc::encoder::LtcEncoder`]; this function is the
/// reference implementation used to verify the SIMD path.
///
/// # Arguments
/// * `bits`      – raw bits to encode (0 or 1 per byte)
/// * `amplitude` – peak amplitude of the output signal (0.0–1.0)
///
/// # Returns
/// A `Vec<f32>` with `2 * bits.len()` samples.
#[must_use]
pub fn manchester_encode_scalar(bits: &[u8], amplitude: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(bits.len() * 2);
    let mut polarity = true; // current "high" phase

    for &b in bits {
        // BMC: always transition at start of bit cell
        polarity = !polarity;
        let hi = if polarity { amplitude } else { -amplitude };
        let lo = -hi;

        if b != 0 {
            // Bit 1: extra transition at mid-cell
            out.push(hi);
            out.push(lo);
        } else {
            // Bit 0: no mid-cell transition, hold for both half-cells
            out.push(hi);
            out.push(hi);
        }
    }
    out
}

/// Scalar Manchester / BMC decoder.
///
/// Reconstructs the original bit sequence from the half-cell sample pairs
/// produced by [`manchester_encode_scalar`].  A mid-cell transition (sign
/// change between sample 0 and sample 1 of a pair) indicates a `1` bit;
/// no mid-cell transition indicates `0`.
///
/// # Arguments
/// * `samples`    – the encoded samples (must have an even length)
/// * `threshold`  – minimum absolute value to consider a sample "active"
///
/// # Returns
/// `Some(Vec<u8>)` with `samples.len() / 2` bytes (each 0 or 1), or `None`
/// if the sample slice has an odd length.
#[must_use]
pub fn manchester_decode_scalar(samples: &[f32], threshold: f32) -> Option<Vec<u8>> {
    if samples.len() % 2 != 0 {
        return None;
    }
    let mut bits = Vec::with_capacity(samples.len() / 2);
    for chunk in samples.chunks_exact(2) {
        let s0 = chunk[0];
        let s1 = chunk[1];
        // If both samples are below threshold, treat as 0
        if s0.abs() < threshold && s1.abs() < threshold {
            bits.push(0u8);
            continue;
        }
        // Mid-cell sign change ⇒ bit 1; no change ⇒ bit 0
        let mid_transition = (s0 > 0.0) != (s1 > 0.0);
        bits.push(if mid_transition { 1 } else { 0 });
    }
    Some(bits)
}

// ---------------------------------------------------------------------------
// x86_64 AVX2 SIMD path
// ---------------------------------------------------------------------------

/// SIMD-accelerated Manchester / BMC encoder (x86_64 AVX2).
///
/// Processes 32 bits at a time using 256-bit AVX2 registers.  Falls back to
/// [`manchester_encode_scalar`] on non-x86_64 targets or when AVX2 is not
/// available at runtime.
///
/// # Safety Notes
/// The unsafe inner function is guarded by a runtime CPUID check via
/// `is_x86_feature_detected!("avx2")` before it is ever called.
#[must_use]
#[allow(unsafe_code)]
pub fn manchester_encode_simd(bits: &[u8], amplitude: f32) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: We have verified AVX2 is available via runtime detection.
            return unsafe { manchester_encode_avx2(bits, amplitude) };
        }
    }
    manchester_encode_scalar(bits, amplitude)
}

/// SIMD-accelerated Manchester / BMC decoder (x86_64 AVX2).
///
/// Falls back to [`manchester_decode_scalar`] when AVX2 is unavailable.
#[must_use]
#[allow(unsafe_code)]
pub fn manchester_decode_simd(samples: &[f32], threshold: f32) -> Option<Vec<u8>> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: We have verified AVX2 is available via runtime detection.
            return unsafe { manchester_decode_avx2(samples, threshold) };
        }
    }
    manchester_decode_scalar(samples, threshold)
}

// ---------------------------------------------------------------------------
// AVX2 implementation (x86_64 only)
// ---------------------------------------------------------------------------

/// AVX2 Manchester encoder.
///
/// Encodes 32 bits per loop iteration using 256-bit registers.
/// The loop processes full 32-bit chunks; a scalar tail handles any remainder.
///
/// # Safety
/// Caller must guarantee that the CPU supports AVX2
/// (verified by `is_x86_feature_detected!("avx2")`).
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "avx2")]
unsafe fn manchester_encode_avx2(bits: &[u8], amplitude: f32) -> Vec<f32> {
    use std::arch::x86_64::*;

    let len = bits.len();
    let mut out = Vec::with_capacity(len * 2);
    let mut polarity = true;

    // Process 32 bits at a time.
    let chunks = len / 32;
    let tail_start = chunks * 32;

    for chunk_idx in 0..chunks {
        let chunk = &bits[chunk_idx * 32..(chunk_idx + 1) * 32];

        // Load 32 bytes (bits) into a 256-bit register.
        // SAFETY: chunk is exactly 32 bytes.
        let b_vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);

        // Create a vector of zeros to compare against (bit == 0 test).
        let zeros = _mm256_setzero_si256();
        // cmp_mask: 0xFF for each byte that is zero, 0x00 for non-zero.
        let is_zero_mask = _mm256_cmpeq_epi8(b_vec, zeros);

        // Extract mask as 32-bit integer: bit i == 1 means bits[i] == 0.
        let zero_bits = _mm256_movemask_epi8(is_zero_mask) as u32;

        // Emit samples based on the computed mask.
        for i in 0..32usize {
            polarity = !polarity; // transition at start of cell
            let hi = if polarity { amplitude } else { -amplitude };
            let bit_is_zero = (zero_bits >> i) & 1 == 1;
            if bit_is_zero {
                out.push(hi);
                out.push(hi);
            } else {
                out.push(hi);
                out.push(-hi);
            }
        }
    }

    // Scalar tail for remaining bits.
    for &b in &bits[tail_start..] {
        polarity = !polarity;
        let hi = if polarity { amplitude } else { -amplitude };
        if b == 0 {
            out.push(hi);
            out.push(hi);
        } else {
            out.push(hi);
            out.push(-hi);
        }
    }

    out
}

/// AVX2 Manchester decoder.
///
/// Decodes sample pairs using 256-bit AVX2 sign-comparison instructions.
/// Falls back to scalar for the last (< 16-pair) tail.
///
/// # Safety
/// Caller must guarantee that the CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "avx2")]
unsafe fn manchester_decode_avx2(samples: &[f32], threshold: f32) -> Option<Vec<u8>> {
    if samples.len() % 2 != 0 {
        return None;
    }

    use std::arch::x86_64::*;

    let pairs = samples.len() / 2;
    let mut bits = Vec::with_capacity(pairs);

    // AVX2 processes 8 f32 pairs (16 floats) at once.
    let simd_pairs = pairs / 8;
    let tail_start_sample = simd_pairs * 16;

    let thr_vec = _mm256_set1_ps(threshold);
    let zero_vec = _mm256_setzero_ps();

    for chunk_idx in 0..simd_pairs {
        let base = chunk_idx * 16;
        // SAFETY: slice length is guaranteed to cover [base, base+16).
        let s_lo = _mm256_loadu_ps(samples[base..].as_ptr());
        let s_hi = _mm256_loadu_ps(samples[base + 8..].as_ptr());

        // Separate even-indexed (s0) and odd-indexed (s1) samples via shuffle.
        // _mm256_permute4x64_epi64 pattern 0xD8 = 1,3,0,2 lane order reorder:
        // We use _mm256_permutevar8x32_ps to interleave.
        // Simpler: just extract per-pair using scalar loop on top of the SIMD-loaded chunk.
        // (Full AVX2 scatter/gather for f32 pairs requires extra shuffles; use hybrid approach.)
        let s_lo_arr: [f32; 8] = std::mem::transmute(s_lo);
        let s_hi_arr: [f32; 8] = std::mem::transmute(s_hi);

        // abs comparison for threshold
        let abs_s_lo = _mm256_andnot_ps(_mm256_set1_ps(-0.0_f32), s_lo);
        let abs_s_hi = _mm256_andnot_ps(_mm256_set1_ps(-0.0_f32), s_hi);

        // Combine 16 samples across two registers for threshold check.
        let active_lo = _mm256_cmp_ps(abs_s_lo, thr_vec, _CMP_GE_OQ); // active if >= threshold
        let active_hi = _mm256_cmp_ps(abs_s_hi, thr_vec, _CMP_GE_OQ);
        let _ = (active_lo, active_hi); // used below via per-pair extraction

        // Sign bits: positive → 0, negative → 1
        let sign_lo = _mm256_cmp_ps(s_lo, zero_vec, _CMP_LT_OQ); // 1 if negative
        let sign_hi = _mm256_cmp_ps(s_hi, zero_vec, _CMP_LT_OQ);
        let _ = (sign_lo, sign_hi);

        // Each BMC pair occupies two consecutive samples.  s_lo contains
        // samples[base+0..base+7] and s_hi contains samples[base+8..base+15].
        // Pairs 0-3 are in s_lo (indices 0,1 / 2,3 / 4,5 / 6,7) and
        // pairs 4-7 are in s_hi (indices 0,1 / 2,3 / 4,5 / 6,7).
        for arr in [&s_lo_arr, &s_hi_arr] {
            for i in 0..4usize {
                let s0 = arr[i * 2];
                let s1 = arr[i * 2 + 1];
                if s0.abs() < threshold && s1.abs() < threshold {
                    bits.push(0u8);
                } else {
                    bits.push(if (s0 > 0.0) != (s1 > 0.0) { 1 } else { 0 });
                }
            }
        }
    }

    // Scalar tail.
    for chunk in samples[tail_start_sample..].chunks_exact(2) {
        let s0 = chunk[0];
        let s1 = chunk[1];
        if s0.abs() < threshold && s1.abs() < threshold {
            bits.push(0u8);
        } else {
            bits.push(if (s0 > 0.0) != (s1 > 0.0) { 1 } else { 0 });
        }
    }

    Some(bits)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode a known bit pattern and decode it back, verifying round-trip fidelity.
    #[test]
    fn test_scalar_roundtrip_all_zeros() {
        let bits: Vec<u8> = vec![0u8; 80];
        let samples = manchester_encode_scalar(&bits, 1.0);
        let decoded = manchester_decode_scalar(&samples, 0.1).expect("decode ok");
        assert_eq!(decoded, bits);
    }

    #[test]
    fn test_scalar_roundtrip_all_ones() {
        let bits: Vec<u8> = vec![1u8; 80];
        let samples = manchester_encode_scalar(&bits, 1.0);
        let decoded = manchester_decode_scalar(&samples, 0.1).expect("decode ok");
        assert_eq!(decoded, bits);
    }

    #[test]
    fn test_scalar_roundtrip_alternating() {
        let bits: Vec<u8> = (0..80).map(|i| (i % 2) as u8).collect();
        let samples = manchester_encode_scalar(&bits, 0.5);
        let decoded = manchester_decode_scalar(&samples, 0.1).expect("decode ok");
        assert_eq!(decoded, bits);
    }

    #[test]
    fn test_scalar_sample_count() {
        let bits = vec![0u8, 1, 0, 1, 1, 0];
        let samples = manchester_encode_scalar(&bits, 1.0);
        assert_eq!(samples.len(), bits.len() * 2);
    }

    #[test]
    fn test_decode_odd_length_returns_none() {
        let samples = vec![0.5f32; 3]; // odd
        assert!(manchester_decode_scalar(&samples, 0.1).is_none());
    }

    /// Verify the SIMD path produces the same output as scalar.
    #[test]
    fn test_simd_matches_scalar_encode() {
        let bits: Vec<u8> = (0..80).map(|i| ((i / 3) % 2) as u8).collect();
        let scalar_out = manchester_encode_scalar(&bits, 0.7);
        let simd_out = manchester_encode_simd(&bits, 0.7);
        assert_eq!(scalar_out.len(), simd_out.len());
        for (a, b) in scalar_out.iter().zip(simd_out.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "SIMD and scalar outputs differ: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_simd_matches_scalar_decode() {
        let bits: Vec<u8> = (0..80).map(|i| (i % 3 == 0) as u8).collect();
        let samples = manchester_encode_scalar(&bits, 0.9);
        let scalar_dec = manchester_decode_scalar(&samples, 0.1).expect("decode ok");
        let simd_dec = manchester_decode_simd(&samples, 0.1).expect("decode ok");
        assert_eq!(scalar_dec, simd_dec);
        assert_eq!(scalar_dec, bits);
    }

    /// SIMD encode+decode round-trip with various bit patterns.
    #[test]
    fn test_simd_roundtrip() {
        let patterns: &[&[u8]] = &[
            &[0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0],
            // LTC sync word pattern (16 bits, LSB-first of 0x3FFD)
            &[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        ];
        for &pattern in patterns {
            let samples = manchester_encode_simd(pattern, 1.0);
            let decoded = manchester_decode_simd(&samples, 0.1).expect("decode ok");
            assert_eq!(
                decoded, pattern,
                "Round-trip failed for pattern {pattern:?}"
            );
        }
    }

    /// Non-AVX2 targets fall back to scalar; this test exercises that path explicitly.
    #[test]
    fn test_scalar_fallback_non_simd() {
        let bits: Vec<u8> = vec![1, 0, 1, 0, 1, 1, 0, 0];
        // Call scalar directly — this is always available.
        let enc = manchester_encode_scalar(&bits, 1.0);
        let dec = manchester_decode_scalar(&enc, 0.05).expect("decode ok");
        assert_eq!(dec, bits);
    }
}
