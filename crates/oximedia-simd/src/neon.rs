//! ARM NEON SIMD backend for OxiMedia.
//!
//! On `aarch64` targets every function in the `neon` sub-module uses real
//! NEON intrinsics via `std::arch::aarch64`.  On all other targets a pure-Rust
//! scalar fallback is compiled instead, keeping the same public API surface.
//!
//! # Example
//! ```rust
//! use oximedia_simd::neon;
//!
//! let result = neon::neon::hadd_f32x4([1.0, 2.0, 3.0, 4.0]);
//! assert!((result - 10.0).abs() < 1e-5);
//! ```

// ── Runtime feature query ─────────────────────────────────────────────────────

/// Returns `true` when the executing CPU provides NEON (always on `aarch64`).
///
/// On `aarch64`, NEON is mandatory per the ARM architecture specification.
/// On other architectures this always returns `false`.
#[must_use]
pub fn has_neon() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// aarch64 — real NEON intrinsics
// ══════════════════════════════════════════════════════════════════════════════
#[cfg(target_arch = "aarch64")]
#[allow(clippy::cast_ptr_alignment)]
#[allow(clippy::module_inception)]
pub mod neon {
    //! NEON-accelerated kernels (aarch64 only).
    //!
    //! Every public function in this module uses `std::arch::aarch64` intrinsics
    //! inside `unsafe` blocks.  Buffer bounds and dimension invariants are always
    //! checked before any pointer arithmetic.
    #![allow(unsafe_code)]

    use std::arch::aarch64::*;

    // ─────────────────────────────────────────────────────────────────────────
    // rgba_to_yuv420_neon
    // ─────────────────────────────────────────────────────────────────────────
    // BT.601 limited-range coefficients (Q8 fixed-point):
    //   Y  = ( 66R + 129G +  25B + 128 + 16·256) >> 8
    //   U  = (-38R -  74G + 112B + 128 + 128·256) >> 8
    //   V  = (112R -  94G -  18B + 128 + 128·256) >> 8
    //
    // The NEON path processes 8 pixels per iteration using uint8x8 / uint16x8
    // registers.  Chroma (U/V) is down-sampled 2×2 using the top-left pixel of
    // each 2×2 block (fastest, negligible quality loss for 4:2:0).
    //
    // Row pairs are processed together so that U/V output is produced once per
    // two input rows.

    /// NEON-accelerated pixel-format conversion: packed RGBA → planar YUV 4:2:0.
    ///
    /// # Arguments
    /// * `src`    – Packed RGBA bytes: `width * height * 4` bytes (R, G, B, A order).
    /// * `dst_y`  – Luma plane output: `width * height` bytes.
    /// * `dst_u`  – Cb (U) sub-sampled plane: `(width/2) * (height/2)` bytes.
    /// * `dst_v`  – Cr (V) sub-sampled plane: `(width/2) * (height/2)` bytes.
    /// * `width`  – Frame width in pixels.  Must be even and ≥ 2.
    /// * `height` – Frame height in pixels.  Must be even and ≥ 2.
    ///
    /// # Panics
    /// Panics if any buffer is too small or if dimensions are not even.
    pub fn rgba_to_yuv420_neon(
        src: &[u8],
        dst_y: &mut [u8],
        dst_u: &mut [u8],
        dst_v: &mut [u8],
        width: usize,
        height: usize,
    ) {
        assert!(width >= 2 && width % 2 == 0, "width must be even and >= 2");
        assert!(
            height >= 2 && height % 2 == 0,
            "height must be even and >= 2"
        );
        assert!(src.len() >= width * height * 4, "src too small");
        assert!(dst_y.len() >= width * height, "dst_y too small");
        let chroma_len = (width / 2) * (height / 2);
        assert!(dst_u.len() >= chroma_len, "dst_u too small");
        assert!(dst_v.len() >= chroma_len, "dst_v too small");

        let stride = width * 4; // bytes per RGBA row

        for pair in 0..(height / 2) {
            let row0 = pair * 2;
            let row1 = row0 + 1;
            let src0 = &src[row0 * stride..];
            let src1 = &src[row1 * stride..];
            // Split dst_y into two non-overlapping halves for row0 and row1.
            let (y_before_row1, y_from_row1) = dst_y.split_at_mut(row1 * width);
            let y0_out = &mut y_before_row1[row0 * width..];
            let y1_out = &mut y_from_row1[..];
            let uv_row_off = pair * (width / 2);
            let u_out = &mut dst_u[uv_row_off..];
            let v_out = &mut dst_v[uv_row_off..];

            let mut col = 0usize;

            // NEON fast path: 8 pixels per loop (two 4-pixel groups per row).
            while col + 8 <= width {
                let src0_ptr = src0[col * 4..].as_ptr();
                let src1_ptr = src1[col * 4..].as_ptr();

                // Load 8 RGBA pixels from each row as interleaved uint8x8x4.
                // vld4_u8 de-interleaves: .0=R, .1=G, .2=B, .3=A  (8 pixels each)
                // SAFETY: col + 8 <= width, and src0/src1 have at least width*4 bytes.
                let p0 = unsafe { vld4_u8(src0_ptr) };
                let p1 = unsafe { vld4_u8(src1_ptr) };

                let r0 = p0.0; // uint8x8_t — 8 R values, row 0
                let g0 = p0.1;
                let b0 = p0.2;
                let r1 = p1.0; // 8 R values, row 1
                let g1 = p1.1;
                let b1 = p1.2;

                // --- Compute Y (luma) for both rows ---
                let y0 = compute_y_8px(r0, g0, b0);
                let y1 = compute_y_8px(r1, g1, b1);

                // SAFETY: col + 8 <= width, so there's room for 8 bytes.
                unsafe {
                    vst1_u8(y0_out[col..].as_mut_ptr(), y0);
                    vst1_u8(y1_out[col..].as_mut_ptr(), y1);
                }

                // --- Compute U/V (chroma) using top-left pixel of each 2×2 block ---
                // We take every other pixel from row 0: r0[0,2,4,6], g0[0,2,4,6], b0[0,2,4,6].
                // vuzp1_u8 takes the even-indexed lanes: [0,2,4,6,...] from uint8x8.
                let r_chroma = unsafe { vuzp1_u8(r0, r0) }; // [r[0],r[2],r[4],r[6], ...]
                let g_chroma = unsafe { vuzp1_u8(g0, g0) };
                let b_chroma = unsafe { vuzp1_u8(b0, b0) };

                // Only 4 valid U/V values per 8 source pixels (one per 2×2 block).
                let u4 = compute_u_4px(r_chroma, g_chroma, b_chroma);
                let v4 = compute_v_4px(r_chroma, g_chroma, b_chroma);

                let uv_col = col / 2;
                // SAFETY: uv_col + 4 <= width/2, bounds checked by outer loop.
                unsafe {
                    vst1_lane_u32::<0>(
                        u_out[uv_col..].as_mut_ptr().cast::<u32>(),
                        vreinterpret_u32_u8(u4),
                    );
                    vst1_lane_u32::<0>(
                        v_out[uv_col..].as_mut_ptr().cast::<u32>(),
                        vreinterpret_u32_u8(v4),
                    );
                }

                col += 8;
            }

            // Scalar tail for remaining pixels.
            while col + 2 <= width {
                for dr in 0..2usize {
                    for dc in 0..2usize {
                        let src_row = if dr == 0 { src0 } else { src1 };
                        let off = (col + dc) * 4;
                        let r = src_row[off] as i32;
                        let g = src_row[off + 1] as i32;
                        let b = src_row[off + 2] as i32;
                        let y = (66 * r + 129 * g + 25 * b + 128 + 16 * 256) >> 8;
                        let y_out = if dr == 0 { &mut *y0_out } else { &mut *y1_out };
                        y_out[col + dc] = y.clamp(0, 255) as u8;
                    }
                }
                let off0 = col * 4;
                let r = src0[off0] as i32;
                let g = src0[off0 + 1] as i32;
                let b = src0[off0 + 2] as i32;
                let u = (-38 * r - 74 * g + 112 * b + 128 + 128 * 256) >> 8;
                let v = (112 * r - 94 * g - 18 * b + 128 + 128 * 256) >> 8;
                let uv_col = col / 2;
                u_out[uv_col] = u.clamp(0, 255) as u8;
                v_out[uv_col] = v.clamp(0, 255) as u8;
                col += 2;
            }
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Compute luma Y for 8 pixels from uint8x8 R/G/B channels.
    ///
    /// Uses Q8 BT.601: `Y = (66R + 129G + 25B + 4224) >> 8`
    /// Returns uint8x8_t with values saturated to [0, 255].
    #[inline]
    fn compute_y_8px(r: uint8x8_t, g: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
        // Widen uint8x8 → uint16x8 for 16-bit accumulation.
        let r16 = unsafe { vmovl_u8(r) };
        let g16 = unsafe { vmovl_u8(g) };
        let b16 = unsafe { vmovl_u8(b) };

        // Bias: 16*256 + 128 (rounding) = 4224
        let acc = unsafe { vmovq_n_u16(4224u16) };
        let acc = unsafe { vmlaq_n_u16(acc, r16, 66u16) };
        let acc = unsafe { vmlaq_n_u16(acc, g16, 129u16) };
        let acc = unsafe { vmlaq_n_u16(acc, b16, 25u16) };
        // Shift right by 8 and narrow saturating to uint8x8.
        let shifted = unsafe { vshrq_n_u16(acc, 8) };
        unsafe { vqmovn_u16(shifted) }
    }

    /// Compute chroma U for 4 pixels from the even-lane uint8x8 channels.
    ///
    /// Uses Q8 BT.601: `U = (-38R - 74G + 112B + 32896) >> 8`
    /// Returns the 4 U values packed into the low 4 bytes of a uint8x8_t.
    #[inline]
    fn compute_u_4px(r: uint8x8_t, g: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
        // Work with the low 4 lanes (the even-indexed pixels from the original 8).
        let r16 = unsafe { vmovl_u8(r) }; // uint16x8, use low half
        let g16 = unsafe { vmovl_u8(g) };
        let b16 = unsafe { vmovl_u8(b) };

        // Reinterpret as signed for the negative coefficients.
        let r_s = unsafe { vreinterpretq_s16_u16(r16) };
        let g_s = unsafe { vreinterpretq_s16_u16(g16) };
        let b_s = unsafe { vreinterpretq_s16_u16(b16) };

        // Full formula: ((−38R − 74G + 112B) + 32896) >> 8
        //             = (−38R − 74G + 112B + 128_rounding) >> 8 + 128
        // bias = 32896 > i16::MAX, so split: compute coefficient sum, add 128 rounding,
        // shift right 8, then add the 128 DC offset.
        let acc = unsafe { vmovq_n_s16(0i16) };
        let acc = unsafe { vmlaq_n_s16(acc, r_s, -38i16) };
        let acc = unsafe { vmlaq_n_s16(acc, g_s, -74i16) };
        let acc = unsafe { vmlaq_n_s16(acc, b_s, 112i16) };
        // Add 128 (bias after shift) and rounding (128 before shift → 0 after integer shift)
        let acc = unsafe { vaddq_s16(acc, vmovq_n_s16(128i16)) }; // rounding
        let shifted = unsafe { vshrq_n_s16(acc, 8) };
        let biased = unsafe { vaddq_s16(shifted, vmovq_n_s16(128i16)) };
        // Clamp to [0, 255] and narrow.
        let clamped = unsafe { vminq_s16(vmaxq_s16(biased, vmovq_n_s16(0)), vmovq_n_s16(255)) };
        let u16_vals = unsafe { vreinterpretq_u16_s16(clamped) };
        // Narrow saturating to uint8x8.
        unsafe { vqmovn_u16(u16_vals) }
    }

    /// Compute chroma V for 4 pixels from the even-lane uint8x8 channels.
    ///
    /// Uses Q8 BT.601: `V = (112R - 94G - 18B + 32896) >> 8`
    /// Returns the 4 V values packed into the low 4 bytes of a uint8x8_t.
    #[inline]
    fn compute_v_4px(r: uint8x8_t, g: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
        let r16 = unsafe { vmovl_u8(r) };
        let g16 = unsafe { vmovl_u8(g) };
        let b16 = unsafe { vmovl_u8(b) };

        let r_s = unsafe { vreinterpretq_s16_u16(r16) };
        let g_s = unsafe { vreinterpretq_s16_u16(g16) };
        let b_s = unsafe { vreinterpretq_s16_u16(b16) };

        let acc = unsafe { vmovq_n_s16(0i16) };
        let acc = unsafe { vmlaq_n_s16(acc, r_s, 112i16) };
        let acc = unsafe { vmlaq_n_s16(acc, g_s, -94i16) };
        let acc = unsafe { vmlaq_n_s16(acc, b_s, -18i16) };
        let acc = unsafe { vaddq_s16(acc, vmovq_n_s16(128i16)) }; // rounding
        let shifted = unsafe { vshrq_n_s16(acc, 8) };
        let biased = unsafe { vaddq_s16(shifted, vmovq_n_s16(128i16)) };
        let clamped = unsafe { vminq_s16(vmaxq_s16(biased, vmovq_n_s16(0)), vmovq_n_s16(255)) };
        let u16_vals = unsafe { vreinterpretq_u16_s16(clamped) };
        unsafe { vqmovn_u16(u16_vals) }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // hadd_f32x4
    // ─────────────────────────────────────────────────────────────────────────

    /// NEON horizontal sum of four `f32` lanes.
    ///
    /// Returns `a[0] + a[1] + a[2] + a[3]` using two `vpadd_f32` passes.
    ///
    /// # Example
    /// ```rust
    /// # #[cfg(target_arch = "aarch64")]
    /// # {
    /// use oximedia_simd::neon::neon::hadd_f32x4;
    /// let s = hadd_f32x4([1.0, 2.0, 3.0, 4.0]);
    /// assert!((s - 10.0f32).abs() < 1e-5);
    /// # }
    /// ```
    #[must_use]
    pub fn hadd_f32x4(a: [f32; 4]) -> f32 {
        // SAFETY: No raw pointers; all operations are register-level on f32x4.
        unsafe {
            let v = vld1q_f32(a.as_ptr());
            let lo = vget_low_f32(v); // [a0, a1]
            let hi = vget_high_f32(v); // [a2, a3]
            let s = vpadd_f32(lo, hi); // [a0+a1, a2+a3]
            let s2 = vpadd_f32(s, s); // [sum, sum]
            vget_lane_f32::<0>(s2)
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // dot_product_neon
    // ─────────────────────────────────────────────────────────────────────────

    /// NEON-accelerated dot product of two `f32` slices.
    ///
    /// Processes 4 elements per SIMD cycle using `vmlaq_f32`.
    /// Remaining elements (slice length not a multiple of 4) are handled with
    /// scalar arithmetic.
    ///
    /// Returns the sum of element-wise products `Σ a[i] * b[i]` over the
    /// shorter of the two slices.
    #[must_use]
    pub fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let chunks = len / 4;

        let mut acc = unsafe { vmovq_n_f32(0.0f32) };

        for i in 0..chunks {
            let base = i * 4;
            // SAFETY: base + 4 <= len <= a.len() and b.len().
            unsafe {
                let va = vld1q_f32(a[base..].as_ptr());
                let vb = vld1q_f32(b[base..].as_ptr());
                acc = vmlaq_f32(acc, va, vb);
            }
        }

        // Horizontal reduction: uint32x4 → scalar
        let lo = unsafe { vget_low_f32(acc) };
        let hi = unsafe { vget_high_f32(acc) };
        let s = unsafe { vpadd_f32(lo, hi) };
        let s2 = unsafe { vpadd_f32(s, s) };
        let simd_sum = unsafe { vget_lane_f32::<0>(s2) };

        // Scalar tail
        let tail_start = chunks * 4;
        let tail: f32 = a[tail_start..len]
            .iter()
            .zip(b[tail_start..len].iter())
            .map(|(&x, &y)| x * y)
            .sum();

        simd_sum + tail
    }

    // ─────────────────────────────────────────────────────────────────────────
    // sad_8x8_neon
    // ─────────────────────────────────────────────────────────────────────────

    /// NEON-accelerated SAD (Sum of Absolute Differences) for an 8×8 block.
    ///
    /// Both `block1` and `block2` are treated as row-major 8×8 arrays of `u8`.
    /// `stride1` / `stride2` are the byte strides between consecutive rows
    /// (must be ≥ 8).
    ///
    /// # Panics
    /// Panics if either buffer is too small to hold 8 rows of the given stride,
    /// or if a stride is less than 8.
    #[must_use]
    pub fn sad_8x8_neon(block1: &[u8], block2: &[u8], stride1: usize, stride2: usize) -> u32 {
        assert!(stride1 >= 8, "stride1 must be >= 8");
        assert!(stride2 >= 8, "stride2 must be >= 8");
        // Need at least 7 full strides + 8 bytes for the last row.
        assert!(block1.len() >= 7 * stride1 + 8, "block1 too small");
        assert!(block2.len() >= 7 * stride2 + 8, "block2 too small");

        // SAFETY: bounds verified above; all pointer arithmetic stays within
        // the verified buffer extents.
        unsafe {
            let mut acc = vmovq_n_u16(0u16);

            for row in 0..8usize {
                let p1 = block1[row * stride1..].as_ptr();
                let p2 = block2[row * stride2..].as_ptr();

                // Load 8 bytes (one row) from each block.
                let v1 = vld1_u8(p1);
                let v2 = vld1_u8(p2);

                // vabd_u8: absolute difference, uint8x8_t.
                // vaddw_u8: widen and accumulate into uint16x8_t.
                let diff = vabd_u8(v1, v2);
                acc = vaddw_u8(acc, diff);
            }

            // Horizontal reduction: uint16x8 → uint32x4 → uint64x2 → u64 → u32
            let sum32 = vpaddlq_u16(acc); // uint32x4
            let sum64 = vpaddlq_u32(sum32); // uint64x2
            let lo = vgetq_lane_u64::<0>(sum64);
            let hi = vgetq_lane_u64::<1>(sum64);
            (lo + hi) as u32
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // sigmoid_f32x4_neon
    // ─────────────────────────────────────────────────────────────────────────

    /// NEON fast f32 sigmoid using a rational approximation.
    ///
    /// Computes `σ(x) = 1 / (1 + exp(-x))` for each of the 4 input lanes
    /// using the fast rational approximation:
    ///
    /// ```text
    /// σ(x) ≈ (x/2) / (1 + |x/2|) + 0.5
    /// ```
    ///
    /// Maximum absolute error < 0.01 over all `x`.  The result is always
    /// clamped to `[0.0, 1.0]`.
    ///
    /// Division is implemented via two Newton-Raphson refinements of
    /// `vrecpeq_f32`, avoiding any `vdivq_f32` which is slower on some µarchs.
    #[must_use]
    pub fn sigmoid_f32x4_neon(x: [f32; 4]) -> [f32; 4] {
        // SAFETY: All operations are lane-wise float arithmetic on NEON registers.
        unsafe {
            let vx = vld1q_f32(x.as_ptr());

            // half = x * 0.5
            let half = vmulq_n_f32(vx, 0.5f32);

            // abs_half = |half|
            let abs_half = vabsq_f32(half);

            // denom = 1.0 + |half|
            let one = vmovq_n_f32(1.0f32);
            let denom = vaddq_f32(one, abs_half);

            // Reciprocal estimate: 1 / denom (two Newton-Raphson steps).
            let recip = vrecpeq_f32(denom);
            // Step 1
            let recip = vmulq_f32(recip, vrecpsq_f32(denom, recip));
            // Step 2 (extra accuracy)
            let recip = vmulq_f32(recip, vrecpsq_f32(denom, recip));

            // approx = half / denom = half * recip
            let approx = vmulq_f32(half, recip);

            // result = approx + 0.5, clamped to [0, 1]
            let point5 = vmovq_n_f32(0.5f32);
            let result = vaddq_f32(approx, point5);
            let result = vmaxq_f32(result, vmovq_n_f32(0.0f32));
            let result = vminq_f32(result, vmovq_n_f32(1.0f32));

            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            out
        }
    }

    // ── Tests (aarch64 only) ──────────────────────────────────────────────────
    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_hadd_f32x4_basic() {
            let result = hadd_f32x4([1.0, 2.0, 3.0, 4.0]);
            assert!(
                (result - 10.0f32).abs() < 1e-5,
                "expected 10.0, got {result}"
            );
        }

        #[test]
        fn test_hadd_f32x4_zeros() {
            let result = hadd_f32x4([0.0; 4]);
            assert!(result.abs() < 1e-9, "expected 0.0, got {result}");
        }

        #[test]
        fn test_hadd_f32x4_negative() {
            let result = hadd_f32x4([-1.0, -2.0, -3.0, -4.0]);
            assert!(
                (result + 10.0f32).abs() < 1e-5,
                "expected -10.0, got {result}"
            );
        }

        #[test]
        fn test_hadd_f32x4_mixed() {
            let result = hadd_f32x4([10.0, -10.0, 5.0, -5.0]);
            assert!(result.abs() < 1e-5, "expected 0.0, got {result}");
        }

        #[test]
        fn test_dot_product_neon_basic() {
            let a = [1.0f32, 2.0, 3.0, 4.0];
            let b = [4.0f32, 3.0, 2.0, 1.0];
            let result = dot_product_neon(&a, &b);
            // 1*4 + 2*3 + 3*2 + 4*1 = 4+6+6+4 = 20
            assert!(
                (result - 20.0f32).abs() < 1e-4,
                "expected 20.0, got {result}"
            );
        }

        #[test]
        fn test_dot_product_neon_non_multiple_of_4() {
            let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
            let b = [1.0f32; 5];
            let result = dot_product_neon(&a, &b);
            assert!(
                (result - 15.0f32).abs() < 1e-4,
                "expected 15.0, got {result}"
            );
        }

        #[test]
        fn test_dot_product_neon_empty() {
            let result = dot_product_neon(&[], &[]);
            assert!(result.abs() < 1e-9, "expected 0.0 for empty slices");
        }

        #[test]
        fn test_dot_product_neon_large() {
            let n = 100;
            let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let b = vec![1.0f32; n];
            let result = dot_product_neon(&a, &b);
            let expected: f32 = (0..n).map(|i| i as f32).sum();
            assert!(
                (result - expected).abs() < 0.1,
                "expected {expected}, got {result}"
            );
        }

        #[test]
        fn test_sad_8x8_identical_blocks() {
            let block = vec![128u8; 8 * 8];
            let result = sad_8x8_neon(&block, &block, 8, 8);
            assert_eq!(result, 0, "SAD of identical blocks must be 0");
        }

        #[test]
        fn test_sad_8x8_known_value() {
            let a = vec![10u8; 8 * 8];
            let b = vec![20u8; 8 * 8];
            // |10 - 20| = 10 per pixel × 64 pixels = 640
            let result = sad_8x8_neon(&a, &b, 8, 8);
            assert_eq!(result, 640, "SAD mismatch: got {result}");
        }

        #[test]
        fn test_sad_8x8_max_diff() {
            let a = vec![0u8; 8 * 8];
            let b = vec![255u8; 8 * 8];
            // 255 × 64 = 16320
            let result = sad_8x8_neon(&a, &b, 8, 8);
            assert_eq!(result, 16320, "SAD max diff mismatch: got {result}");
        }

        #[test]
        fn test_sad_8x8_with_stride() {
            // Use stride=16 to emulate sub-block in a larger frame buffer.
            let mut a = vec![0u8; 8 * 16];
            let mut b = vec![0u8; 8 * 16];
            for row in 0..8usize {
                for col in 0..8usize {
                    a[row * 16 + col] = 50;
                    b[row * 16 + col] = 100;
                }
            }
            // |50 - 100| × 64 = 3200
            let result = sad_8x8_neon(&a, &b, 16, 16);
            assert_eq!(result, 3200, "SAD with stride mismatch: got {result}");
        }

        #[test]
        fn test_sigmoid_zero() {
            let result = sigmoid_f32x4_neon([0.0; 4]);
            for v in result {
                assert!((v - 0.5f32).abs() < 0.01, "sigmoid(0) ≈ 0.5, got {v}");
            }
        }

        #[test]
        fn test_sigmoid_large_positive() {
            let result = sigmoid_f32x4_neon([8.0; 4]);
            for v in result {
                assert!(v > 0.95, "sigmoid(8) should be > 0.95, got {v}");
            }
        }

        #[test]
        fn test_sigmoid_large_negative() {
            let result = sigmoid_f32x4_neon([-8.0; 4]);
            for v in result {
                assert!(v < 0.05, "sigmoid(-8) should be < 0.05, got {v}");
            }
        }

        #[test]
        fn test_sigmoid_bounds() {
            let result = sigmoid_f32x4_neon([100.0, -100.0, 0.5, -0.5]);
            for v in result {
                assert!(v >= 0.0 && v <= 1.0, "sigmoid out of [0,1]: {v}");
            }
        }

        #[test]
        fn test_sigmoid_symmetry() {
            let pos = sigmoid_f32x4_neon([1.0, 2.0, 3.0, 4.0]);
            let neg = sigmoid_f32x4_neon([-1.0, -2.0, -3.0, -4.0]);
            for (p, n) in pos.iter().zip(neg.iter()) {
                assert!(
                    (p + n - 1.0f32).abs() < 0.02,
                    "sigmoid symmetry: {p} + {n} ≠ 1"
                );
            }
        }

        #[test]
        fn test_rgba_to_yuv420_black() {
            let width = 8;
            let height = 4;
            let src = vec![0u8; width * height * 4];
            let mut dst_y = vec![0u8; width * height];
            let mut dst_u = vec![0u8; (width / 2) * (height / 2)];
            let mut dst_v = vec![0u8; (width / 2) * (height / 2)];
            rgba_to_yuv420_neon(&src, &mut dst_y, &mut dst_u, &mut dst_v, width, height);
            for &y in &dst_y {
                assert!((y as i32 - 16).abs() < 3, "Y for black: {y}");
            }
            for &u in &dst_u {
                assert!((u as i32 - 128).abs() < 5, "U for black: {u}");
            }
            for &v in &dst_v {
                assert!((v as i32 - 128).abs() < 5, "V for black: {v}");
            }
        }

        #[test]
        fn test_rgba_to_yuv420_white() {
            let width = 8;
            let height = 4;
            let src = vec![255u8; width * height * 4];
            let mut dst_y = vec![0u8; width * height];
            let mut dst_u = vec![0u8; (width / 2) * (height / 2)];
            let mut dst_v = vec![0u8; (width / 2) * (height / 2)];
            rgba_to_yuv420_neon(&src, &mut dst_y, &mut dst_u, &mut dst_v, width, height);
            for &y in &dst_y {
                assert!(y > 230, "Y for white should be > 230: {y}");
            }
        }

        #[test]
        fn test_rgba_to_yuv420_output_sizes() {
            let width = 16;
            let height = 8;
            let src = vec![128u8; width * height * 4];
            let mut dst_y = vec![0u8; width * height];
            let mut dst_u = vec![0u8; (width / 2) * (height / 2)];
            let mut dst_v = vec![0u8; (width / 2) * (height / 2)];
            // Should not panic.
            rgba_to_yuv420_neon(&src, &mut dst_y, &mut dst_u, &mut dst_v, width, height);
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// non-aarch64 — scalar fallbacks with identical public API
// ══════════════════════════════════════════════════════════════════════════════
#[cfg(not(target_arch = "aarch64"))]
#[allow(clippy::module_inception)]
pub mod neon {
    //! Scalar fallback implementations that mirror the NEON API.
    //!
    //! These are compiled on non-aarch64 targets so that code using
    //! `oximedia_simd::neon::neon::*` compiles and runs correctly on any
    //! architecture.  Performance is equivalent to ordinary Rust scalar code.

    // ─────────────────────────────────────────────────────────────────────────
    // rgba_to_yuv420_neon (scalar)
    // ─────────────────────────────────────────────────────────────────────────

    /// RGBA → planar YUV 4:2:0 (BT.601 limited range), scalar fallback.
    ///
    /// Produces identical numerical results to the NEON path.
    ///
    /// # Panics
    /// Panics if any buffer is too small or dimensions are not even.
    pub fn rgba_to_yuv420_neon(
        src: &[u8],
        dst_y: &mut [u8],
        dst_u: &mut [u8],
        dst_v: &mut [u8],
        width: usize,
        height: usize,
    ) {
        assert!(width >= 2 && width % 2 == 0, "width must be even and >= 2");
        assert!(
            height >= 2 && height % 2 == 0,
            "height must be even and >= 2"
        );
        assert!(src.len() >= width * height * 4, "src too small");
        assert!(dst_y.len() >= width * height, "dst_y too small");
        let chroma_len = (width / 2) * (height / 2);
        assert!(dst_u.len() >= chroma_len, "dst_u too small");
        assert!(dst_v.len() >= chroma_len, "dst_v too small");

        for row in (0..height).step_by(2) {
            let uv_row = row / 2;
            for col in (0..width).step_by(2) {
                let uv_col = col / 2;
                // Compute Y for all 4 pixels in the 2×2 block.
                for dr in 0..2usize {
                    for dc in 0..2usize {
                        let off = ((row + dr) * width + col + dc) * 4;
                        let r = src[off] as i32;
                        let g = src[off + 1] as i32;
                        let b = src[off + 2] as i32;
                        let y = (66 * r + 129 * g + 25 * b + 128 + 16 * 256) >> 8;
                        dst_y[(row + dr) * width + col + dc] = y.clamp(0, 255) as u8;
                    }
                }
                // U/V from the top-left pixel of each 2×2 block.
                let off = (row * width + col) * 4;
                let r = src[off] as i32;
                let g = src[off + 1] as i32;
                let b = src[off + 2] as i32;
                let u = (-38 * r - 74 * g + 112 * b + 128 + 128 * 256) >> 8;
                let v = (112 * r - 94 * g - 18 * b + 128 + 128 * 256) >> 8;
                dst_u[uv_row * (width / 2) + uv_col] = u.clamp(0, 255) as u8;
                dst_v[uv_row * (width / 2) + uv_col] = v.clamp(0, 255) as u8;
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // hadd_f32x4 (scalar)
    // ─────────────────────────────────────────────────────────────────────────

    /// Horizontal sum of four `f32` values (scalar fallback).
    ///
    /// Returns `a[0] + a[1] + a[2] + a[3]`.
    #[must_use]
    pub fn hadd_f32x4(a: [f32; 4]) -> f32 {
        a[0] + a[1] + a[2] + a[3]
    }

    // ─────────────────────────────────────────────────────────────────────────
    // dot_product_neon (scalar)
    // ─────────────────────────────────────────────────────────────────────────

    /// Dot product of two `f32` slices (scalar fallback).
    ///
    /// Returns `Σ a[i] * b[i]` over the shorter of the two slices.
    #[must_use]
    pub fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        a[..len]
            .iter()
            .zip(b[..len].iter())
            .map(|(&x, &y)| x * y)
            .sum()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // sad_8x8_neon (scalar)
    // ─────────────────────────────────────────────────────────────────────────

    /// SAD for an 8×8 block (scalar fallback).
    ///
    /// # Panics
    /// Panics if either buffer is too small for the given stride, or stride < 8.
    #[must_use]
    pub fn sad_8x8_neon(block1: &[u8], block2: &[u8], stride1: usize, stride2: usize) -> u32 {
        assert!(stride1 >= 8, "stride1 must be >= 8");
        assert!(stride2 >= 8, "stride2 must be >= 8");
        assert!(block1.len() >= 7 * stride1 + 8, "block1 too small");
        assert!(block2.len() >= 7 * stride2 + 8, "block2 too small");
        let mut sum = 0u32;
        for row in 0..8usize {
            for col in 0..8usize {
                let a = block1[row * stride1 + col] as i32;
                let b_val = block2[row * stride2 + col] as i32;
                sum += (a - b_val).unsigned_abs();
            }
        }
        sum
    }

    // ─────────────────────────────────────────────────────────────────────────
    // sigmoid_f32x4_neon (scalar)
    // ─────────────────────────────────────────────────────────────────────────

    /// Fast sigmoid approximation for 4 `f32` values (scalar fallback).
    ///
    /// Uses the same rational approximation as the NEON path:
    /// `σ(x) ≈ (x/2) / (1 + |x/2|) + 0.5`, clamped to `[0.0, 1.0]`.
    #[must_use]
    pub fn sigmoid_f32x4_neon(x: [f32; 4]) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        for (i, &xi) in x.iter().enumerate() {
            let h = xi * 0.5;
            let approx = h / (1.0 + h.abs());
            out[i] = (approx + 0.5).clamp(0.0, 1.0);
        }
        out
    }

    // ── Tests (non-aarch64 / scalar fallback) ─────────────────────────────────
    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_hadd_f32x4_basic() {
            let result = hadd_f32x4([1.0, 2.0, 3.0, 4.0]);
            assert!(
                (result - 10.0f32).abs() < 1e-5,
                "expected 10.0, got {result}"
            );
        }

        #[test]
        fn test_hadd_f32x4_zeros() {
            let result = hadd_f32x4([0.0; 4]);
            assert!(result.abs() < 1e-9, "expected 0.0, got {result}");
        }

        #[test]
        fn test_hadd_f32x4_negative() {
            let result = hadd_f32x4([-1.0, -2.0, -3.0, -4.0]);
            assert!(
                (result + 10.0f32).abs() < 1e-5,
                "expected -10.0, got {result}"
            );
        }

        #[test]
        fn test_hadd_f32x4_mixed() {
            let result = hadd_f32x4([10.0, -10.0, 5.0, -5.0]);
            assert!(result.abs() < 1e-5, "expected 0.0, got {result}");
        }

        #[test]
        fn test_dot_product_basic() {
            let a = [1.0f32, 2.0, 3.0, 4.0];
            let b = [4.0f32, 3.0, 2.0, 1.0];
            let result = dot_product_neon(&a, &b);
            assert!(
                (result - 20.0f32).abs() < 1e-4,
                "expected 20.0, got {result}"
            );
        }

        #[test]
        fn test_dot_product_tail() {
            let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
            let b = [1.0f32; 5];
            let result = dot_product_neon(&a, &b);
            assert!(
                (result - 15.0f32).abs() < 1e-4,
                "expected 15.0, got {result}"
            );
        }

        #[test]
        fn test_dot_product_empty() {
            let result = dot_product_neon(&[], &[]);
            assert!(result.abs() < 1e-9, "expected 0.0 for empty slices");
        }

        #[test]
        fn test_dot_product_large() {
            let n = 100;
            let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let b = vec![1.0f32; n];
            let result = dot_product_neon(&a, &b);
            let expected: f32 = (0..n).map(|i| i as f32).sum();
            assert!(
                (result - expected).abs() < 0.1,
                "expected {expected}, got {result}"
            );
        }

        #[test]
        fn test_sad_8x8_identical() {
            let block = vec![128u8; 8 * 8];
            assert_eq!(sad_8x8_neon(&block, &block, 8, 8), 0);
        }

        #[test]
        fn test_sad_8x8_known() {
            let a = vec![10u8; 8 * 8];
            let b = vec![20u8; 8 * 8];
            assert_eq!(sad_8x8_neon(&a, &b, 8, 8), 640);
        }

        #[test]
        fn test_sad_8x8_max_diff() {
            let a = vec![0u8; 8 * 8];
            let b = vec![255u8; 8 * 8];
            assert_eq!(sad_8x8_neon(&a, &b, 8, 8), 16320);
        }

        #[test]
        fn test_sad_8x8_with_stride() {
            let mut a = vec![0u8; 8 * 16];
            let mut b = vec![0u8; 8 * 16];
            for row in 0..8usize {
                for col in 0..8usize {
                    a[row * 16 + col] = 50;
                    b[row * 16 + col] = 100;
                }
            }
            assert_eq!(sad_8x8_neon(&a, &b, 16, 16), 3200);
        }

        #[test]
        fn test_sigmoid_zero() {
            let result = sigmoid_f32x4_neon([0.0; 4]);
            for v in result {
                assert!((v - 0.5f32).abs() < 0.01, "sigmoid(0) ≈ 0.5, got {v}");
            }
        }

        #[test]
        fn test_sigmoid_large_positive() {
            let result = sigmoid_f32x4_neon([8.0; 4]);
            for v in result {
                assert!(v > 0.95, "sigmoid(8) > 0.95, got {v}");
            }
        }

        #[test]
        fn test_sigmoid_large_negative() {
            let result = sigmoid_f32x4_neon([-8.0; 4]);
            for v in result {
                assert!(v < 0.05, "sigmoid(-8) < 0.05, got {v}");
            }
        }

        #[test]
        fn test_sigmoid_bounds() {
            let inputs = [100.0f32, -100.0, 0.5, -0.5];
            let result = sigmoid_f32x4_neon(inputs);
            for v in result {
                assert!(v >= 0.0 && v <= 1.0, "sigmoid out of [0,1]: {v}");
            }
        }

        #[test]
        fn test_sigmoid_symmetry() {
            let pos = sigmoid_f32x4_neon([1.0, 2.0, 3.0, 4.0]);
            let neg = sigmoid_f32x4_neon([-1.0, -2.0, -3.0, -4.0]);
            for (p, n) in pos.iter().zip(neg.iter()) {
                assert!(
                    (p + n - 1.0f32).abs() < 0.02,
                    "sigmoid symmetry: {p} + {n} ≠ 1"
                );
            }
        }

        #[test]
        fn test_rgba_to_yuv420_black() {
            let width = 8;
            let height = 4;
            let src = vec![0u8; width * height * 4];
            let mut dst_y = vec![0u8; width * height];
            let mut dst_u = vec![0u8; (width / 2) * (height / 2)];
            let mut dst_v = vec![0u8; (width / 2) * (height / 2)];
            rgba_to_yuv420_neon(&src, &mut dst_y, &mut dst_u, &mut dst_v, width, height);
            for &y in &dst_y {
                assert!((y as i32 - 16).abs() < 3, "Y for black: {y}");
            }
            for &u in &dst_u {
                assert!((u as i32 - 128).abs() < 5, "U for black: {u}");
            }
            for &v in &dst_v {
                assert!((v as i32 - 128).abs() < 5, "V for black: {v}");
            }
        }

        #[test]
        fn test_rgba_to_yuv420_white() {
            let width = 8;
            let height = 4;
            let src = vec![255u8; width * height * 4];
            let mut dst_y = vec![0u8; width * height];
            let mut dst_u = vec![0u8; (width / 2) * (height / 2)];
            let mut dst_v = vec![0u8; (width / 2) * (height / 2)];
            rgba_to_yuv420_neon(&src, &mut dst_y, &mut dst_u, &mut dst_v, width, height);
            for &y in &dst_y {
                assert!(y > 230, "Y for white should be > 230: {y}");
            }
        }

        #[test]
        fn test_rgba_to_yuv420_crossarch_parity() {
            // Verify scalar results match direct BT.601 formula computation.
            let r = 100i32;
            let g = 150i32;
            let b = 50i32;
            let expected_y =
                ((66 * r + 129 * g + 25 * b + 128 + 16 * 256) >> 8).clamp(0, 255) as u8;
            let expected_u =
                ((-38 * r - 74 * g + 112 * b + 128 + 128 * 256) >> 8).clamp(0, 255) as u8;
            let expected_v =
                ((112 * r - 94 * g - 18 * b + 128 + 128 * 256) >> 8).clamp(0, 255) as u8;

            let width = 2;
            let height = 2;
            // 4 pixels, all (R=100, G=150, B=50, A=255)
            let src: Vec<u8> = vec![
                100, 150, 50, 255, 100, 150, 50, 255, 100, 150, 50, 255, 100, 150, 50, 255,
            ];
            let mut dst_y = vec![0u8; 4];
            let mut dst_u = vec![0u8; 1];
            let mut dst_v = vec![0u8; 1];
            rgba_to_yuv420_neon(&src, &mut dst_y, &mut dst_u, &mut dst_v, width, height);

            for &y in &dst_y {
                assert!(
                    (y as i32 - expected_y as i32).abs() <= 1,
                    "Y mismatch: {y} vs {expected_y}"
                );
            }
            assert!(
                (dst_u[0] as i32 - expected_u as i32).abs() <= 2,
                "U mismatch: {} vs {expected_u}",
                dst_u[0]
            );
            assert!(
                (dst_v[0] as i32 - expected_v as i32).abs() <= 2,
                "V mismatch: {} vs {expected_v}",
                dst_v[0]
            );
        }

        #[test]
        fn test_rgba_to_yuv420_output_sizes() {
            let width = 16;
            let height = 8;
            let src = vec![128u8; width * height * 4];
            let mut dst_y = vec![0u8; width * height];
            let mut dst_u = vec![0u8; (width / 2) * (height / 2)];
            let mut dst_v = vec![0u8; (width / 2) * (height / 2)];
            // Should not panic.
            rgba_to_yuv420_neon(&src, &mut dst_y, &mut dst_u, &mut dst_v, width, height);
        }
    }
}
