//! 8×8 DCT/IDCT using the Loeffler–Ligtenberg–Moschytz algorithm.
//!
//! The Loeffler 1989 factorisation achieves the DCT-II in 11 multiplications
//! and 29 additions (the theoretical minimum for N=8).  This file provides:
//!
//! - `DctBlock` — a newtype wrapper for `[i16; 64]`
//! - `forward_dct_8x8` / `inverse_dct_8x8` — in-place 1D row+column passes
//! - `forward_dct_batch` / `inverse_dct_batch` — Rayon-parallel batch variants
//!
//! A WGSL compute shader string is also exposed for GPU offloading; the GPU
//! path uses one workgroup per 8×8 block with one thread per row/column.

use rayon::prelude::*;

/// WGSL compute shader for GPU DCT (one workgroup per 8×8 block, 8 threads).
///
/// Each thread processes one row in the forward DCT then one column in a
/// second pass (two dispatches required: `dct_row_pass` then `dct_col_pass`).
pub const DCT_SHADER_WGSL: &str = r#"
// 8x8 Loeffler DCT/IDCT compute shader.
// Dispatch: (num_blocks, 1, 1) workgroups, workgroup_size(8, 1, 1).
// Each workgroup owns one 8x8 block; each thread handles one row (or column).

// Storage: flat array of 64 i32 per block (scaled by 1024 for fixed-point).
@group(0) @binding(0) var<storage, read_write> blocks: array<i32>;
@group(0) @binding(1) var<uniform>             num_blocks: u32;

// Loeffler constants (scaled by 1024)
const C1: i32 = 1004;   // cos(pi/16) * 1024 ≈ 1004
const C2: i32 =  946;   // cos(2*pi/16) * 1024 ≈ 946
const C3: i32 =  851;   // cos(3*pi/16) * 1024 ≈ 851
const C5: i32 =  569;   // cos(5*pi/16) * 1024 ≈ 569
const C6: i32 =  391;   // cos(6*pi/16) * 1024 ≈ 391
const C7: i32 =  200;   // cos(7*pi/16) * 1024 ≈ 200
const S6: i32 =  946;   // sin(6*pi/16) = cos(2*pi/16) ≈ 946
const RC: i32 = 1024;   // scale divisor

fn loeffler_1d(x: array<i32, 8>) -> array<i32, 8> {
    // Stage 1: butterfly pairs
    var s: array<i32, 8>;
    s[0] = x[0] + x[7];
    s[1] = x[1] + x[6];
    s[2] = x[2] + x[5];
    s[3] = x[3] + x[4];
    s[4] = x[3] - x[4];
    s[5] = x[2] - x[5];
    s[6] = x[1] - x[6];
    s[7] = x[0] - x[7];

    // Stage 2: even part
    var e: array<i32, 4>;
    e[0] = s[0] + s[3];
    e[1] = s[1] + s[2];
    e[2] = s[1] - s[2];
    e[3] = s[0] - s[3];

    // Stage 3: even part finish
    var out: array<i32, 8>;
    out[0] = (e[0] + e[1]);
    out[4] = (e[0] - e[1]);
    out[2] = (C6 * e[2] + S6 * e[3]) / RC;
    out[6] = (C6 * e[3] - S6 * e[2]) / RC;

    // Stage 4: odd part rotations
    let p = s[4] + s[7];
    let q = s[5] + s[6];
    let r = s[4] + s[6];
    let t = s[5] + s[7];
    let k = (C3 * (p + q)) / RC;

    let o0 = k - (C7 * p) / RC;
    let o1 = k - (C1 * q) / RC;
    let o2 = (C3 * (r + t)) / RC - (C5 * t) / RC;
    let o3 = (C3 * (r + t)) / RC - (C5 * r) / RC;

    out[7] = (o0 + (C7 * s[4]) / RC) - (C3 * s[7]) / RC;
    out[5] = (C3 * s[5]) / RC - (C5 * s[6]) / RC + o1 - o0;
    out[3] = o2;
    out[1] = (C1 * s[6]) / RC - (C7 * s[7]) / RC + o3;
    let _ = o3; // suppress warning

    return out;
}

@compute @workgroup_size(8, 1, 1)
fn dct_row_pass(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
) {
    let block_idx = gid.x / 8u;
    if (block_idx >= num_blocks) { return; }
    let row = lid.x;
    let base = block_idx * 64u + row * 8u;

    var x: array<i32, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        x[i] = blocks[base + i];
    }
    let y = loeffler_1d(x);
    for (var i = 0u; i < 8u; i = i + 1u) {
        blocks[base + i] = y[i];
    }
}

@compute @workgroup_size(8, 1, 1)
fn dct_col_pass(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
) {
    let block_idx = gid.x / 8u;
    if (block_idx >= num_blocks) { return; }
    let col = lid.x;
    let base = block_idx * 64u;

    var x: array<i32, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        x[i] = blocks[base + i * 8u + col];
    }
    let y = loeffler_1d(x);
    for (var i = 0u; i < 8u; i = i + 1u) {
        blocks[base + i * 8u + col] = y[i];
    }
}
"#;

/// A single 8×8 DCT block stored in row-major order (64 `i16` coefficients).
#[derive(Debug, Clone, PartialEq)]
pub struct DctBlock(pub [i16; 64]);

impl DctBlock {
    /// Create a block filled with zeroes.
    #[must_use]
    pub fn zeroed() -> Self {
        Self([0i16; 64])
    }

    /// Create a block where every coefficient equals `value`.
    #[must_use]
    pub fn filled(value: i16) -> Self {
        Self([value; 64])
    }

    /// Read the coefficient at row `r`, column `c`.
    #[must_use]
    pub fn get(&self, r: usize, c: usize) -> i16 {
        self.0[r * 8 + c]
    }

    /// Write the coefficient at row `r`, column `c`.
    pub fn set(&mut self, r: usize, c: usize, v: i16) {
        self.0[r * 8 + c] = v;
    }
}

// ── Loeffler 1-D DCT ─────────────────────────────────────────────────────────
//
// This is the Loeffler–Ligtenberg–Moschytz (LLM) factorisation of the length-8
// DCT-II.  Scaling by 1/√8 produces an orthonormal transform.
//
// For JPEG compatibility we use the AAN (Arai-Agui-Nakajima) simplification
// which incorporates the normalisation into the dequantisation step and works
// purely with integer arithmetic via a 12-bit fixed-point representation.

/// Fixed-point scale (2^12 = 4096).
const FIX_SCALE: i32 = 4096;

#[inline]
fn fix(x: f64) -> i32 {
    (x * FIX_SCALE as f64).round() as i32
}

/// Loeffler 1-D forward DCT on 8 `i32` values (in-place).
///
/// Scaling: output[k] ≈ DCT[k] * 2 (not fully normalised; caller may divide
/// by 8 after both row and column passes for a valid DCT-II).
fn loeffler_1d_forward(v: &mut [i32; 8]) {
    // Precomputed Loeffler constants (fixed-point, scale 4096)
    let w1 = fix(0.707_106_781); // cos(pi/4) = 1/sqrt(2)
    let w2 = fix(0.541_196_100); // cos(3*pi/8)
    let w3 = fix(1.306_562_965); // cos(pi/8)
    let w4 = fix(0.382_683_432); // sin(pi/8) / sqrt(2)  ≈ sin(3π/8)·w
    let w5 = fix(1.847_759_065); // cos(pi/8) * sqrt(2)

    // Stage 1: butterfly
    let s0 = v[0] + v[7];
    let s1 = v[1] + v[6];
    let s2 = v[2] + v[5];
    let s3 = v[3] + v[4];
    let s4 = v[3] - v[4];
    let s5 = v[2] - v[5];
    let s6 = v[1] - v[6];
    let s7 = v[0] - v[7];

    // Even half
    let e0 = s0 + s3;
    let e1 = s1 + s2;
    let e2 = s1 - s2;
    let e3 = s0 - s3;

    v[0] = e0 + e1;
    v[4] = e0 - e1;
    v[2] = (e2 * w2 + e3 * w3) / FIX_SCALE; // approximation
    v[6] = (e3 * w2 - e2 * w3) / FIX_SCALE;

    // Odd half (Loeffler rotation butterfly)
    let p = s4 + s7;
    let q = s5 + s6;
    let r = s4 + s6;
    let t = s5 + s7;
    let k = (p + q) * w1 / FIX_SCALE;

    v[5] = k - p * w4 / FIX_SCALE;
    v[3] = k - q * w4 / FIX_SCALE;
    // Two-point IDCT-style rotation for 1,7
    let z1 = (r + t) * w5 / FIX_SCALE / 2;
    v[7] = z1 - t * fix(1.175_875_602) / FIX_SCALE;
    v[1] = z1 - r * fix(0.275_899_379) / FIX_SCALE;
}

/// Loeffler 1-D inverse DCT on 8 `i32` values (in-place).
fn loeffler_1d_inverse(v: &mut [i32; 8]) {
    // Use the transpose of the forward butterfly (DCT-III = DCT-II^T).
    let w1 = fix(0.707_106_781);
    let w2 = fix(0.541_196_100);
    let w3 = fix(1.306_562_965);
    let w5 = fix(1.847_759_065);
    let wt = fix(1.175_875_602);
    let wu = fix(0.275_899_379);

    // Stage 1: even inverse
    let e0 = v[0];
    let e1 = v[4];
    let tmp0 = e0 + e1;
    let tmp1 = e0 - e1;
    let tmp2 = (v[2] * w2 + v[6] * w3) / FIX_SCALE;
    let tmp3 = (v[2] * w3 - v[6] * w2) / FIX_SCALE;
    let e_a = tmp0 + tmp2;
    let e_b = tmp1 + tmp3;
    let e_c = tmp1 - tmp3;
    let e_d = tmp0 - tmp2;

    // Stage 2: odd inverse
    let o1 = v[1];
    let o3 = v[3];
    let o5 = v[5];
    let o7 = v[7];
    let z1 = (o1 + o7) * w5 / FIX_SCALE / 2;
    let z2 = (o3 + o5) * w1 / FIX_SCALE;
    let z3 = o1 + o3;
    let z4 = o5 + o7;
    let z5 = (z3 + z4) * fix(1.175_875_602) / FIX_SCALE;
    let tmp10 = o1 * fix(0.298_631_336) / FIX_SCALE + z5 - z4 * wt / FIX_SCALE;
    let tmp12 = o3 * fix(2.053_119_869) / FIX_SCALE + z5 - z3 * wt / FIX_SCALE;
    let tmp13 = o5 * fix(3.072_711_026) / FIX_SCALE + z1 - z2;
    let tmp11 = o7 * fix(1.501_321_110) / FIX_SCALE + z1 - z2;
    let _ = wu;
    let _ = wt;
    let _ = w2;
    let _ = w3;

    v[0] = e_a + tmp10;
    v[7] = e_a - tmp10;
    v[1] = e_b + tmp11;
    v[6] = e_b - tmp11;
    v[2] = e_c + tmp12;
    v[5] = e_c - tmp12;
    v[3] = e_d + tmp13;
    v[4] = e_d - tmp13;
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Apply the forward 8×8 DCT-II to `block` in-place (rows then columns).
///
/// The Loeffler algorithm is applied row-by-row, then column-by-column.
/// Values are shifted right by 3 after both passes to account for the √8
/// normalisation inherent in the 2-pass structure.
pub fn forward_dct_8x8(block: &mut DctBlock) {
    let mut buf = [0i32; 64];

    // Lift i16 → i32
    for i in 0..64 {
        buf[i] = block.0[i] as i32;
    }

    // Row pass
    for row in 0..8 {
        let mut row_buf = [0i32; 8];
        row_buf.copy_from_slice(&buf[row * 8..row * 8 + 8]);
        loeffler_1d_forward(&mut row_buf);
        buf[row * 8..row * 8 + 8].copy_from_slice(&row_buf);
    }

    // Column pass
    for col in 0..8 {
        let mut col_buf = [0i32; 8];
        for i in 0..8 {
            col_buf[i] = buf[i * 8 + col];
        }
        loeffler_1d_forward(&mut col_buf);
        for i in 0..8 {
            buf[i * 8 + col] = col_buf[i];
        }
    }

    // Normalise: divide by 8 (2^3) — two passes each multiply by ~2
    for i in 0..64 {
        block.0[i] = (buf[i] >> 3).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
}

/// Apply the inverse 8×8 DCT (DCT-III) to `block` in-place (columns then rows).
pub fn inverse_dct_8x8(block: &mut DctBlock) {
    let mut buf = [0i32; 64];
    for i in 0..64 {
        buf[i] = block.0[i] as i32;
    }

    // Column pass first (transpose of forward)
    for col in 0..8 {
        let mut col_buf = [0i32; 8];
        for i in 0..8 {
            col_buf[i] = buf[i * 8 + col];
        }
        loeffler_1d_inverse(&mut col_buf);
        for i in 0..8 {
            buf[i * 8 + col] = col_buf[i];
        }
    }

    // Row pass
    for row in 0..8 {
        let mut row_buf = [0i32; 8];
        row_buf.copy_from_slice(&buf[row * 8..row * 8 + 8]);
        loeffler_1d_inverse(&mut row_buf);
        buf[row * 8..row * 8 + 8].copy_from_slice(&row_buf);
    }

    // Normalise: divide by 8
    for i in 0..64 {
        block.0[i] = (buf[i] >> 3).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
}

/// Batch forward DCT using Rayon parallelism.
///
/// Each block is processed independently on a worker thread.
pub fn forward_dct_batch(blocks: &mut [DctBlock]) {
    blocks.par_iter_mut().for_each(|b| forward_dct_8x8(b));
}

/// Batch inverse DCT using Rayon parallelism.
pub fn inverse_dct_batch(blocks: &mut [DctBlock]) {
    blocks.par_iter_mut().for_each(|b| inverse_dct_8x8(b));
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Maximum absolute error we allow for a DCT round-trip (quantisation
    /// from i32→i16 introduces at most a few LSBs of rounding error).
    const MAX_ROUND_TRIP_ERR: i16 = 4;

    #[test]
    fn test_all_zero_forward_dct_is_zero() {
        let mut block = DctBlock::zeroed();
        forward_dct_8x8(&mut block);
        for &v in &block.0 {
            assert_eq!(v, 0, "forward DCT of zero block must be zero");
        }
    }

    #[test]
    fn test_all_zero_inverse_dct_is_zero() {
        let mut block = DctBlock::zeroed();
        inverse_dct_8x8(&mut block);
        for &v in &block.0 {
            assert_eq!(v, 0, "inverse DCT of zero block must be zero");
        }
    }

    #[test]
    fn test_dc_only_block_round_trip() {
        // A DC-only block: coefficient [0] = 64, rest = 0.
        let mut freq = DctBlock::zeroed();
        freq.0[0] = 64;
        let mut check = freq.clone();
        inverse_dct_8x8(&mut check);
        forward_dct_8x8(&mut check);
        // Should return very close to the original [64, 0, 0, …]
        for (i, (&orig, &rt)) in freq.0.iter().zip(check.0.iter()).enumerate() {
            let err = (orig - rt).abs();
            assert!(
                err <= MAX_ROUND_TRIP_ERR,
                "DC round-trip mismatch at [{i}]: orig={orig} round-trip={rt} err={err}"
            );
        }
    }

    #[test]
    fn test_uniform_spatial_domain_round_trip() {
        // Spatial block where every pixel = 100.
        let mut block = DctBlock::filled(100);
        let original = block.clone();
        forward_dct_8x8(&mut block);
        inverse_dct_8x8(&mut block);
        for (i, (&orig, &rt)) in original.0.iter().zip(block.0.iter()).enumerate() {
            let err = (orig - rt).abs();
            assert!(
                err <= MAX_ROUND_TRIP_ERR,
                "uniform round-trip mismatch at [{i}]: orig={orig} rt={rt} err={err}"
            );
        }
    }

    #[test]
    fn test_forward_dct_dc_coefficient_is_largest() {
        // For a block of all-same values the DC term (index 0) should dominate.
        let mut block = DctBlock::filled(128);
        forward_dct_8x8(&mut block);
        let dc = block.0[0].abs();
        for (i, &ac) in block.0.iter().enumerate().skip(1) {
            assert!(
                dc >= ac.abs(),
                "DC ({dc}) should be >= AC[{i}] ({ac}) for uniform block"
            );
        }
    }

    #[test]
    fn test_alternating_block_has_high_frequency_energy() {
        // A checkerboard (alternating +/-) should produce high-frequency DCT
        // coefficients — specifically the (7,7) AC coefficient should dominate.
        let mut block = DctBlock::zeroed();
        for r in 0..8 {
            for c in 0..8 {
                block.0[r * 8 + c] = if (r + c) % 2 == 0 { 50i16 } else { -50i16 };
            }
        }
        forward_dct_8x8(&mut block);
        // The (7,7) frequency should be non-zero for a checkerboard pattern.
        let high_freq = block.0[7 * 8 + 7].abs();
        // The DC should be near zero (the checkerboard averages to 0).
        let dc = block.0[0].abs();
        assert!(dc <= 4, "DC coeff of checkerboard should be ~0; got {dc}");
        assert!(
            high_freq > 10,
            "High-frequency coeff (7,7) should be significant; got {high_freq}"
        );
    }

    #[test]
    fn test_batch_forward_dct() {
        let mut blocks: Vec<DctBlock> = (0..4).map(|_| DctBlock::filled(50)).collect();
        let before: Vec<DctBlock> = blocks.clone();
        forward_dct_batch(&mut blocks);
        // Each block should have been transformed (DC != 50 since DCT scales it)
        for (i, (b, a)) in before.iter().zip(blocks.iter()).enumerate() {
            assert_ne!(
                b.0[0], a.0[0],
                "batch forward DCT should transform block {i}"
            );
        }
    }

    #[test]
    fn test_batch_inverse_dct() {
        // Apply forward then inverse via batch functions → round-trip.
        let mut blocks: Vec<DctBlock> = (0..8).map(|i| DctBlock::filled(i as i16 * 10)).collect();
        let originals = blocks.clone();
        forward_dct_batch(&mut blocks);
        inverse_dct_batch(&mut blocks);
        for (bi, (orig, rt)) in originals.iter().zip(blocks.iter()).enumerate() {
            for (i, (&o, &r)) in orig.0.iter().zip(rt.0.iter()).enumerate() {
                let err = (o - r).abs();
                assert!(
                    err <= MAX_ROUND_TRIP_ERR,
                    "batch round-trip block {bi}[{i}]: orig={o} rt={r} err={err}"
                );
            }
        }
    }
}
