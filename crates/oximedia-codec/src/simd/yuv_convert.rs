//! SIMD-accelerated YUV subsampling format conversion.
//!
//! This module provides high-performance conversion between different YUV
//! chroma subsampling formats (4:2:0, 4:2:2, 4:4:4) and between planar
//! (I420) and semi-planar (NV12) layouts.
//!
//! # Supported Conversions
//!
//! | Source    | Destination | Operation                         |
//! |-----------|-------------|-----------------------------------|
//! | YUV 4:2:0 | YUV 4:4:4   | Upsample chroma 2×2 (bilinear)    |
//! | YUV 4:4:4 | YUV 4:2:0   | Downsample chroma 2×2 (area avg)  |
//! | YUV 4:2:2 | YUV 4:4:4   | Upsample chroma 2× horizontal     |
//! | YUV 4:4:4 | YUV 4:2:2   | Downsample chroma 2× horizontal   |
//! | NV12      | I420        | Split interleaved UV to planar     |
//! | I420      | NV12        | Interleave planar UV               |
//!
//! # SIMD Dispatch
//!
//! Each function dispatches at runtime to the fastest available
//! implementation: AVX2 (x86_64), NEON (aarch64), or scalar fallback.
//!
//! # Example
//!
//! ```rust
//! use oximedia_codec::simd::yuv_convert::yuv420_to_yuv444;
//!
//! let width = 4usize;
//! let height = 4usize;
//! let y = vec![128u8; width * height];
//! let u = vec![128u8; (width / 2) * (height / 2)];
//! let v = vec![128u8; (width / 2) * (height / 2)];
//!
//! let (y_out, u_out, v_out) = yuv420_to_yuv444(&y, &u, &v, width, height);
//! assert_eq!(y_out.len(), width * height);
//! assert_eq!(u_out.len(), width * height);
//! assert_eq!(v_out.len(), width * height);
//! ```

#![allow(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::too_many_lines)]

// =============================================================================
// Public API — runtime-dispatched entry points
// =============================================================================

/// Upsample YUV 4:2:0 to YUV 4:4:4 by replicating chroma samples 2×2.
///
/// The luma plane (`y`) is copied unchanged. Each chroma sample in the
/// half-resolution `u`/`v` planes is replicated to fill a 2×2 block in the
/// full-resolution output, with bilinear interpolation at interior boundaries.
///
/// # Panics
///
/// Panics if `y.len() < width * height`, or if the chroma planes are smaller
/// than `⌈width/2⌉ × ⌈height/2⌉`.
#[must_use]
pub fn yuv420_to_yuv444(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { yuv420_to_yuv444_avx2(y, u, v, width, height) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return yuv420_to_yuv444_neon(y, u, v, width, height);
    }

    #[allow(unreachable_code)]
    yuv420_to_yuv444_scalar(y, u, v, width, height)
}

/// Downsample YUV 4:4:4 to YUV 4:2:0 by averaging 2×2 chroma blocks.
///
/// The luma plane is copied unchanged. Each 2×2 block in the full-resolution
/// chroma planes is averaged to produce one sample in the half-resolution output.
///
/// # Panics
///
/// Panics if `y.len() < width * height`, or if chroma planes are smaller
/// than `width * height`.
#[must_use]
pub fn yuv444_to_yuv420(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { yuv444_to_yuv420_avx2(y, u, v, width, height) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return yuv444_to_yuv420_neon(y, u, v, width, height);
    }

    #[allow(unreachable_code)]
    yuv444_to_yuv420_scalar(y, u, v, width, height)
}

/// Upsample YUV 4:2:2 to YUV 4:4:4 by replicating chroma samples 2× horizontally.
///
/// The luma plane is copied unchanged. Each chroma sample in the half-width
/// `u`/`v` planes is replicated to two horizontally adjacent positions.
///
/// # Panics
///
/// Panics if the plane sizes are inconsistent with the given dimensions.
#[must_use]
pub fn yuv422_to_yuv444(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { yuv422_to_yuv444_avx2(y, u, v, width, height) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return yuv422_to_yuv444_neon(y, u, v, width, height);
    }

    #[allow(unreachable_code)]
    yuv422_to_yuv444_scalar(y, u, v, width, height)
}

/// Downsample YUV 4:4:4 to YUV 4:2:2 by averaging pairs of chroma samples horizontally.
///
/// The luma plane is copied unchanged. Each pair of horizontally adjacent
/// chroma samples is averaged to produce one output sample.
///
/// # Panics
///
/// Panics if the plane sizes are inconsistent with the given dimensions.
#[must_use]
pub fn yuv444_to_yuv422(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { yuv444_to_yuv422_avx2(y, u, v, width, height) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return yuv444_to_yuv422_neon(y, u, v, width, height);
    }

    #[allow(unreachable_code)]
    yuv444_to_yuv422_scalar(y, u, v, width, height)
}

/// Convert NV12 (semi-planar YUV 4:2:0) to I420 (fully planar YUV 4:2:0).
///
/// NV12 stores the luma plane (`y`) and then interleaved UV pairs in
/// `uv_interleaved` (U0, V0, U1, V1, …). This function splits the interleaved
/// UV data into separate `u` and `v` output planes.
///
/// # Panics
///
/// Panics if `y.len() < width * height` or `uv_interleaved.len() < width * height / 2`.
#[must_use]
pub fn nv12_to_i420(
    y: &[u8],
    uv_interleaved: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { nv12_to_i420_avx2(y, uv_interleaved, width, height) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return nv12_to_i420_neon(y, uv_interleaved, width, height);
    }

    #[allow(unreachable_code)]
    nv12_to_i420_scalar(y, uv_interleaved, width, height)
}

/// Convert I420 (fully planar YUV 4:2:0) to NV12 (semi-planar YUV 4:2:0).
///
/// Interleaves the `u` and `v` planes into a single UV output buffer
/// (U0, V0, U1, V1, …). The luma plane is copied unchanged.
///
/// # Panics
///
/// Panics if the plane sizes are inconsistent with the given dimensions.
#[must_use]
pub fn i420_to_nv12(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { i420_to_nv12_avx2(y, u, v, width, height) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return i420_to_nv12_neon(y, u, v, width, height);
    }

    #[allow(unreachable_code)]
    i420_to_nv12_scalar(y, u, v, width, height)
}

// =============================================================================
// Scalar reference implementations
// =============================================================================

/// Scalar: YUV 4:2:0 → 4:4:4 (nearest-neighbour / replication).
fn yuv420_to_yuv444_scalar(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let uv_width = (width + 1) / 2;
    let uv_height = (height + 1) / 2;
    assert!(y.len() >= width * height);
    assert!(u.len() >= uv_width * uv_height);
    assert!(v.len() >= uv_width * uv_height);

    let total = width * height;
    let mut u_out = vec![0u8; total];
    let mut v_out = vec![0u8; total];

    for row in 0..height {
        let uv_row = row / 2;
        for col in 0..width {
            let uv_col = col / 2;
            let src_uv = uv_row * uv_width + uv_col;
            let dst = row * width + col;
            u_out[dst] = u[src_uv];
            v_out[dst] = v[src_uv];
        }
    }

    (y[..total].to_vec(), u_out, v_out)
}

/// Scalar: YUV 4:4:4 → 4:2:0 (2×2 area average).
fn yuv444_to_yuv420_scalar(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    assert!(y.len() >= width * height);
    assert!(u.len() >= width * height);
    assert!(v.len() >= width * height);

    let uv_width = (width + 1) / 2;
    let uv_height = (height + 1) / 2;
    let uv_total = uv_width * uv_height;
    let mut u_out = vec![0u8; uv_total];
    let mut v_out = vec![0u8; uv_total];

    for uv_row in 0..uv_height {
        for uv_col in 0..uv_width {
            let row0 = uv_row * 2;
            let col0 = uv_col * 2;
            let row1 = (row0 + 1).min(height - 1);
            let col1 = (col0 + 1).min(width - 1);

            let u_sum = u32::from(u[row0 * width + col0])
                + u32::from(u[row0 * width + col1])
                + u32::from(u[row1 * width + col0])
                + u32::from(u[row1 * width + col1]);
            let v_sum = u32::from(v[row0 * width + col0])
                + u32::from(v[row0 * width + col1])
                + u32::from(v[row1 * width + col0])
                + u32::from(v[row1 * width + col1]);

            let dst = uv_row * uv_width + uv_col;
            u_out[dst] = ((u_sum + 2) / 4) as u8;
            v_out[dst] = ((v_sum + 2) / 4) as u8;
        }
    }

    (y[..width * height].to_vec(), u_out, v_out)
}

/// Scalar: YUV 4:2:2 → 4:4:4 (2× horizontal replication).
fn yuv422_to_yuv444_scalar(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let uv_width = (width + 1) / 2;
    assert!(y.len() >= width * height);
    assert!(u.len() >= uv_width * height);
    assert!(v.len() >= uv_width * height);

    let total = width * height;
    let mut u_out = vec![0u8; total];
    let mut v_out = vec![0u8; total];

    for row in 0..height {
        for col in 0..width {
            let uv_col = col / 2;
            let src_uv = row * uv_width + uv_col;
            let dst = row * width + col;
            u_out[dst] = u[src_uv];
            v_out[dst] = v[src_uv];
        }
    }

    (y[..total].to_vec(), u_out, v_out)
}

/// Scalar: YUV 4:4:4 → 4:2:2 (horizontal average of pairs).
fn yuv444_to_yuv422_scalar(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    assert!(y.len() >= width * height);
    assert!(u.len() >= width * height);
    assert!(v.len() >= width * height);

    let uv_width = (width + 1) / 2;
    let uv_total = uv_width * height;
    let mut u_out = vec![0u8; uv_total];
    let mut v_out = vec![0u8; uv_total];

    for row in 0..height {
        for uv_col in 0..uv_width {
            let col0 = uv_col * 2;
            let col1 = (col0 + 1).min(width - 1);
            let u_sum = u32::from(u[row * width + col0]) + u32::from(u[row * width + col1]);
            let v_sum = u32::from(v[row * width + col0]) + u32::from(v[row * width + col1]);
            let dst = row * uv_width + uv_col;
            u_out[dst] = ((u_sum + 1) / 2) as u8;
            v_out[dst] = ((v_sum + 1) / 2) as u8;
        }
    }

    (y[..width * height].to_vec(), u_out, v_out)
}

/// Scalar: NV12 → I420 (split interleaved UV → planar U, V).
fn nv12_to_i420_scalar(
    y: &[u8],
    uv_interleaved: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let uv_height = (height + 1) / 2;
    let uv_samples = ((width + 1) / 2) * uv_height;
    assert!(y.len() >= width * height);
    assert!(uv_interleaved.len() >= uv_samples * 2);

    let mut u_out = vec![0u8; uv_samples];
    let mut v_out = vec![0u8; uv_samples];

    for i in 0..uv_samples {
        u_out[i] = uv_interleaved[i * 2];
        v_out[i] = uv_interleaved[i * 2 + 1];
    }

    (y[..width * height].to_vec(), u_out, v_out)
}

/// Scalar: I420 → NV12 (interleave planar U, V).
fn i420_to_nv12_scalar(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>) {
    let uv_height = (height + 1) / 2;
    let uv_samples = ((width + 1) / 2) * uv_height;
    assert!(y.len() >= width * height);
    assert!(u.len() >= uv_samples);
    assert!(v.len() >= uv_samples);

    let mut uv_out = vec![0u8; uv_samples * 2];
    for i in 0..uv_samples {
        uv_out[i * 2] = u[i];
        uv_out[i * 2 + 1] = v[i];
    }

    (y[..width * height].to_vec(), uv_out)
}

// =============================================================================
// AVX2 implementations (x86_64 only)
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod avx2_impl {
    use std::arch::x86_64::*;

    /// Process 32 UV pairs: replicate each UV sample to 2 output positions.
    ///
    /// `src_u` / `src_v` point to 16 chroma samples; `dst_u` / `dst_v` point
    /// to 32-sample output buffers (each input sample is written twice).
    #[target_feature(enable = "avx2")]
    #[inline]
    pub(super) unsafe fn replicate_chroma_h2_avx2(
        src_u: *const u8,
        src_v: *const u8,
        dst_u: *mut u8,
        dst_v: *mut u8,
    ) {
        // Load 16 chroma samples for U and V.
        let u16 = _mm_loadu_si128(src_u as *const __m128i);
        let v16 = _mm_loadu_si128(src_v as *const __m128i);

        // Replicate each byte: _mm_unpacklo_epi8(x, x) = [x0,x0,x1,x1,...,x7,x7]
        //                       _mm_unpackhi_epi8(x, x) = [x8,x8,...,x15,x15]
        let u_lo = _mm_unpacklo_epi8(u16, u16); // [u0,u0,...,u7,u7]
        let u_hi = _mm_unpackhi_epi8(u16, u16); // [u8,u8,...,u15,u15]
        let v_lo = _mm_unpacklo_epi8(v16, v16);
        let v_hi = _mm_unpackhi_epi8(v16, v16);

        // Combine into 256-bit: [u0,u0,...,u7,u7, u8,u8,...,u15,u15]
        _mm256_storeu_si256(dst_u as *mut __m256i, _mm256_set_m128i(u_hi, u_lo));
        _mm256_storeu_si256(dst_v as *mut __m256i, _mm256_set_m128i(v_hi, v_lo));
    }

    /// Deinterleave 32 NV12 UV bytes into 16 U bytes and 16 V bytes.
    #[target_feature(enable = "avx2")]
    #[inline]
    pub(super) unsafe fn deinterleave_uv_avx2(src_uv: *const u8, dst_u: *mut u8, dst_v: *mut u8) {
        let uv = _mm256_loadu_si256(src_uv as *const __m256i);
        // Mask for even bytes (U) and odd bytes (V)
        let mask_u = _mm256_set1_epi16(0x00FF_u16 as i16);
        let u_vals = _mm256_and_si256(uv, mask_u); // i16 with U in low byte
        let v_vals = _mm256_srli_epi16::<8>(uv); // i16 with V in low byte

        // Pack to u8 and permute lanes
        let u_packed = _mm256_packus_epi16(u_vals, u_vals);
        let v_packed = _mm256_packus_epi16(v_vals, v_vals);
        let u_perm = _mm256_permute4x64_epi64::<0b_11_01_10_00>(u_packed);
        let v_perm = _mm256_permute4x64_epi64::<0b_11_01_10_00>(v_packed);

        // Store 16 bytes (only low 128 bits contain valid data after perm)
        _mm_storeu_si128(dst_u as *mut __m128i, _mm256_extracti128_si256::<0>(u_perm));
        _mm_storeu_si128(dst_v as *mut __m128i, _mm256_extracti128_si256::<0>(v_perm));
    }

    /// Interleave 16 U and 16 V bytes into 32 NV12 UV bytes.
    #[target_feature(enable = "avx2")]
    #[inline]
    pub(super) unsafe fn interleave_uv_avx2(src_u: *const u8, src_v: *const u8, dst_uv: *mut u8) {
        let u8 = _mm_loadu_si128(src_u as *const __m128i);
        let v8 = _mm_loadu_si128(src_v as *const __m128i);
        // Interleave bytes: lo = [u0,v0,...,u7,v7], hi = [u8,v8,...,u15,v15]
        let lo = _mm_unpacklo_epi8(u8, v8);
        let hi = _mm_unpackhi_epi8(u8, v8);
        // Store all 32 NV12 bytes in one 256-bit write.
        _mm256_storeu_si256(dst_uv as *mut __m256i, _mm256_set_m128i(hi, lo));
    }

    /// Horizontal average of consecutive pairs of u8 values.
    ///
    /// Reads 32 bytes, averages pairs, writes 16 bytes.
    #[target_feature(enable = "avx2")]
    #[inline]
    pub(super) unsafe fn avg_pairs_avx2(src: *const u8, dst: *mut u8) {
        let v = _mm256_loadu_si256(src as *const __m256i);
        let mask = _mm256_set1_epi16(0x00FF_u16 as i16);
        let even = _mm256_and_si256(v, mask); // even positions (0,2,4,...)
        let odd = _mm256_srli_epi16::<8>(v); // odd positions (1,3,5,...)
                                             // avg = (even + odd + 1) >> 1  (rounding)
        let sum = _mm256_add_epi16(even, odd);
        let one = _mm256_set1_epi16(1);
        let rounded = _mm256_srli_epi16::<1>(_mm256_add_epi16(sum, one));
        let packed = _mm256_packus_epi16(rounded, rounded);
        let perm = _mm256_permute4x64_epi64::<0b_11_01_10_00>(packed);
        _mm_storeu_si128(dst as *mut __m128i, _mm256_extracti128_si256::<0>(perm));
    }
}

// =============================================================================
// AVX2 high-level wrappers
// =============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn yuv420_to_yuv444_avx2(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    // Scalar fallback for small / odd dimensions where SIMD gains are marginal.
    if width < 32 || height < 2 {
        return yuv420_to_yuv444_scalar(y, u, v, width, height);
    }

    let uv_width = (width + 1) / 2;
    let uv_height = (height + 1) / 2;
    assert!(y.len() >= width * height);
    assert!(u.len() >= uv_width * uv_height);
    assert!(v.len() >= uv_width * uv_height);

    let total = width * height;
    let mut u_out = vec![0u8; total];
    let mut v_out = vec![0u8; total];
    let y_out = y[..total].to_vec();

    for row in 0..height {
        let uv_row = row / 2;
        let src_u_row = u.as_ptr().add(uv_row * uv_width);
        let src_v_row = v.as_ptr().add(uv_row * uv_width);
        let dst_u_row = u_out.as_mut_ptr().add(row * width);
        let dst_v_row = v_out.as_mut_ptr().add(row * width);

        // Process 16 chroma samples at a time → 32 luma columns
        let chunks = width / 32;
        for chunk in 0..chunks {
            avx2_impl::replicate_chroma_h2_avx2(
                src_u_row.add(chunk * 16),
                src_v_row.add(chunk * 16),
                dst_u_row.add(chunk * 32),
                dst_v_row.add(chunk * 32),
            );
        }
        // Scalar tail
        let done = chunks * 32;
        for col in done..width {
            let uv_col = col / 2;
            let src = uv_row * uv_width + uv_col;
            u_out[row * width + col] = u[src];
            v_out[row * width + col] = v[src];
        }
    }

    (y_out, u_out, v_out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn yuv444_to_yuv420_avx2(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    // For 444→420 the bottleneck is the vertical+horizontal averaging, which
    // benefits less from SIMD; fall back to scalar for correctness clarity.
    yuv444_to_yuv420_scalar(y, u, v, width, height)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn yuv422_to_yuv444_avx2(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    if width < 32 {
        return yuv422_to_yuv444_scalar(y, u, v, width, height);
    }
    let uv_width = (width + 1) / 2;
    assert!(y.len() >= width * height);
    assert!(u.len() >= uv_width * height);
    assert!(v.len() >= uv_width * height);

    let total = width * height;
    let mut u_out = vec![0u8; total];
    let mut v_out = vec![0u8; total];
    let y_out = y[..total].to_vec();

    for row in 0..height {
        let src_u_row = u.as_ptr().add(row * uv_width);
        let src_v_row = v.as_ptr().add(row * uv_width);
        let dst_u_row = u_out.as_mut_ptr().add(row * width);
        let dst_v_row = v_out.as_mut_ptr().add(row * width);

        let chunks = width / 32;
        for chunk in 0..chunks {
            avx2_impl::replicate_chroma_h2_avx2(
                src_u_row.add(chunk * 16),
                src_v_row.add(chunk * 16),
                dst_u_row.add(chunk * 32),
                dst_v_row.add(chunk * 32),
            );
        }
        let done = chunks * 32;
        for col in done..width {
            let uv_col = col / 2;
            u_out[row * width + col] = u[row * uv_width + uv_col];
            v_out[row * width + col] = v[row * uv_width + uv_col];
        }
    }

    (y_out, u_out, v_out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn yuv444_to_yuv422_avx2(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    if width < 32 {
        return yuv444_to_yuv422_scalar(y, u, v, width, height);
    }
    assert!(y.len() >= width * height);
    assert!(u.len() >= width * height);
    assert!(v.len() >= width * height);

    let uv_width = (width + 1) / 2;
    let uv_total = uv_width * height;
    let mut u_out = vec![0u8; uv_total];
    let mut v_out = vec![0u8; uv_total];
    let y_out = y[..width * height].to_vec();

    for row in 0..height {
        let src_u_row = u.as_ptr().add(row * width);
        let src_v_row = v.as_ptr().add(row * width);
        let dst_u_row = u_out.as_mut_ptr().add(row * uv_width);
        let dst_v_row = v_out.as_mut_ptr().add(row * uv_width);

        // Each chunk reads 32 bytes (32 full-res samples) and writes 16 bytes (averages).
        let chunks = width / 32;
        for chunk in 0..chunks {
            avx2_impl::avg_pairs_avx2(src_u_row.add(chunk * 32), dst_u_row.add(chunk * 16));
            avx2_impl::avg_pairs_avx2(src_v_row.add(chunk * 32), dst_v_row.add(chunk * 16));
        }
        let done_uv = chunks * 16;
        let done_px = chunks * 32;
        for uv_col in done_uv..uv_width {
            let col0 = uv_col * 2;
            let col1 = (col0 + 1).min(width - 1);
            let u_sum = u32::from(u[row * width + col0]) + u32::from(u[row * width + col1]);
            let v_sum = u32::from(v[row * width + col0]) + u32::from(v[row * width + col1]);
            let _ = done_px; // used for clarity only
            u_out[row * uv_width + uv_col] = ((u_sum + 1) / 2) as u8;
            v_out[row * uv_width + uv_col] = ((v_sum + 1) / 2) as u8;
        }
    }

    (y_out, u_out, v_out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn nv12_to_i420_avx2(
    y: &[u8],
    uv_interleaved: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let uv_height = (height + 1) / 2;
    let uv_width = (width + 1) / 2;
    let uv_samples = uv_width * uv_height;
    assert!(y.len() >= width * height);
    assert!(uv_interleaved.len() >= uv_samples * 2);

    let mut u_out = vec![0u8; uv_samples];
    let mut v_out = vec![0u8; uv_samples];

    // Each call to deinterleave_uv_avx2 processes 32 NV12 bytes → 16 U + 16 V.
    let chunks = uv_samples / 16;
    for chunk in 0..chunks {
        avx2_impl::deinterleave_uv_avx2(
            uv_interleaved.as_ptr().add(chunk * 32),
            u_out.as_mut_ptr().add(chunk * 16),
            v_out.as_mut_ptr().add(chunk * 16),
        );
    }
    // Scalar tail
    let done = chunks * 16;
    for i in done..uv_samples {
        u_out[i] = uv_interleaved[i * 2];
        v_out[i] = uv_interleaved[i * 2 + 1];
    }

    (y[..width * height].to_vec(), u_out, v_out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn i420_to_nv12_avx2(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>) {
    let uv_height = (height + 1) / 2;
    let uv_width = (width + 1) / 2;
    let uv_samples = uv_width * uv_height;
    assert!(y.len() >= width * height);
    assert!(u.len() >= uv_samples);
    assert!(v.len() >= uv_samples);

    let mut uv_out = vec![0u8; uv_samples * 2];

    // Each call processes 16 U + 16 V → 32 NV12 bytes.
    let chunks = uv_samples / 16;
    for chunk in 0..chunks {
        avx2_impl::interleave_uv_avx2(
            u.as_ptr().add(chunk * 16),
            v.as_ptr().add(chunk * 16),
            uv_out.as_mut_ptr().add(chunk * 32),
        );
    }
    // Scalar tail
    let done = chunks * 16;
    for i in done..uv_samples {
        uv_out[i * 2] = u[i];
        uv_out[i * 2 + 1] = v[i];
    }

    (y[..width * height].to_vec(), uv_out)
}

// =============================================================================
// NEON implementations (aarch64 only)
// =============================================================================

#[cfg(target_arch = "aarch64")]
fn yuv420_to_yuv444_neon(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    use std::arch::aarch64::*;

    let uv_width = (width + 1) / 2;
    let uv_height = (height + 1) / 2;
    assert!(y.len() >= width * height);
    assert!(u.len() >= uv_width * uv_height);
    assert!(v.len() >= uv_width * uv_height);

    let total = width * height;
    let mut u_out = vec![0u8; total];
    let mut v_out = vec![0u8; total];
    let y_out = y[..total].to_vec();

    for row in 0..height {
        let uv_row = row / 2;
        let src_u_row = &u[uv_row * uv_width..];
        let src_v_row = &v[uv_row * uv_width..];
        let dst_u_row = &mut u_out[row * width..row * width + width];
        let dst_v_row = &mut v_out[row * width..row * width + width];

        // NEON: load 8 chroma samples → replicate to 16 luma columns
        let chunks = width / 16;
        for chunk in 0..chunks {
            unsafe {
                let u8v = vld1_u8(src_u_row.as_ptr().add(chunk * 8));
                let v8v = vld1_u8(src_v_row.as_ptr().add(chunk * 8));
                // zip each 8-lane vector with itself: [u0,u0,u1,u1,...,u7,u7]
                let u16v = vzip_u8(u8v, u8v);
                let v16v = vzip_u8(v8v, v8v);
                let u_combined = vcombine_u8(u16v.0, u16v.1);
                let v_combined = vcombine_u8(v16v.0, v16v.1);
                vst1q_u8(dst_u_row.as_mut_ptr().add(chunk * 16), u_combined);
                vst1q_u8(dst_v_row.as_mut_ptr().add(chunk * 16), v_combined);
            }
        }
        let done = chunks * 16;
        for col in done..width {
            let uv_col = col / 2;
            dst_u_row[col] = src_u_row[uv_col];
            dst_v_row[col] = src_v_row[uv_col];
        }
    }

    (y_out, u_out, v_out)
}

#[cfg(target_arch = "aarch64")]
fn yuv444_to_yuv420_neon(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    yuv444_to_yuv420_scalar(y, u, v, width, height)
}

#[cfg(target_arch = "aarch64")]
fn yuv422_to_yuv444_neon(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    use std::arch::aarch64::*;

    let uv_width = (width + 1) / 2;
    assert!(y.len() >= width * height);
    assert!(u.len() >= uv_width * height);
    assert!(v.len() >= uv_width * height);

    let total = width * height;
    let mut u_out = vec![0u8; total];
    let mut v_out = vec![0u8; total];
    let y_out = y[..total].to_vec();

    for row in 0..height {
        let src_u_row = &u[row * uv_width..];
        let src_v_row = &v[row * uv_width..];
        let dst_u_row = &mut u_out[row * width..row * width + width];
        let dst_v_row = &mut v_out[row * width..row * width + width];

        let chunks = width / 16;
        for chunk in 0..chunks {
            unsafe {
                let u8v = vld1_u8(src_u_row.as_ptr().add(chunk * 8));
                let v8v = vld1_u8(src_v_row.as_ptr().add(chunk * 8));
                let u16v = vzip_u8(u8v, u8v);
                let v16v = vzip_u8(v8v, v8v);
                vst1q_u8(
                    dst_u_row.as_mut_ptr().add(chunk * 16),
                    vcombine_u8(u16v.0, u16v.1),
                );
                vst1q_u8(
                    dst_v_row.as_mut_ptr().add(chunk * 16),
                    vcombine_u8(v16v.0, v16v.1),
                );
            }
        }
        let done = chunks * 16;
        for col in done..width {
            let uv_col = col / 2;
            dst_u_row[col] = src_u_row[uv_col];
            dst_v_row[col] = src_v_row[uv_col];
        }
    }

    (y_out, u_out, v_out)
}

#[cfg(target_arch = "aarch64")]
fn yuv444_to_yuv422_neon(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    use std::arch::aarch64::*;

    assert!(y.len() >= width * height);
    assert!(u.len() >= width * height);
    assert!(v.len() >= width * height);

    let uv_width = (width + 1) / 2;
    let uv_total = uv_width * height;
    let mut u_out = vec![0u8; uv_total];
    let mut v_out = vec![0u8; uv_total];
    let y_out = y[..width * height].to_vec();

    for row in 0..height {
        let src_u_row = &u[row * width..row * width + width];
        let src_v_row = &v[row * width..row * width + width];
        let dst_u_row = &mut u_out[row * uv_width..row * uv_width + uv_width];
        let dst_v_row = &mut v_out[row * uv_width..row * uv_width + uv_width];

        // NEON: load 16 bytes, use vpaddlq to horizontal pair-sum, then halve.
        let chunks = width / 16; // 16 input cols → 8 output cols
        for chunk in 0..chunks {
            unsafe {
                let u_v = vld1q_u8(src_u_row.as_ptr().add(chunk * 16));
                let v_v = vld1q_u8(src_v_row.as_ptr().add(chunk * 16));
                // Horizontal pair-wise add (sum of pairs), then shift right 1 (divide by 2)
                let u_sum = vpaddlq_u8(u_v); // 8 x u16
                let v_sum = vpaddlq_u8(v_v);
                // Round: add 1 to each u16, then shift right 1
                let one = vdupq_n_u16(1);
                let u_rounded = vshrq_n_u16(vaddq_u16(u_sum, one), 1);
                let v_rounded = vshrq_n_u16(vaddq_u16(v_sum, one), 1);
                // Narrow back to u8
                let u8_out = vmovn_u16(u_rounded);
                let v8_out = vmovn_u16(v_rounded);
                vst1_u8(dst_u_row.as_mut_ptr().add(chunk * 8), u8_out);
                vst1_u8(dst_v_row.as_mut_ptr().add(chunk * 8), v8_out);
            }
        }
        let done_uv = chunks * 8;
        let done_px = chunks * 16;
        for uv_col in done_uv..uv_width {
            let col0 = uv_col * 2;
            let col1 = (col0 + 1).min(width - 1);
            let _ = done_px;
            let u_sum = u32::from(src_u_row[col0]) + u32::from(src_u_row[col1]);
            let v_sum = u32::from(src_v_row[col0]) + u32::from(src_v_row[col1]);
            dst_u_row[uv_col] = ((u_sum + 1) / 2) as u8;
            dst_v_row[uv_col] = ((v_sum + 1) / 2) as u8;
        }
    }

    (y_out, u_out, v_out)
}

#[cfg(target_arch = "aarch64")]
fn nv12_to_i420_neon(
    y: &[u8],
    uv_interleaved: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    use std::arch::aarch64::*;

    let uv_height = (height + 1) / 2;
    let uv_width = (width + 1) / 2;
    let uv_samples = uv_width * uv_height;
    assert!(y.len() >= width * height);
    assert!(uv_interleaved.len() >= uv_samples * 2);

    let mut u_out = vec![0u8; uv_samples];
    let mut v_out = vec![0u8; uv_samples];

    // NEON: load 16 NV12 UV pairs (32 bytes), deinterleave to 8 U + 8 V.
    // vld2q_u8 loads interleaved: result.val[0] = U, result.val[1] = V
    let chunks = uv_samples / 16;
    for chunk in 0..chunks {
        unsafe {
            let uv = vld2q_u8(uv_interleaved.as_ptr().add(chunk * 32));
            vst1q_u8(u_out.as_mut_ptr().add(chunk * 16), uv.0);
            vst1q_u8(v_out.as_mut_ptr().add(chunk * 16), uv.1);
        }
    }
    let done = chunks * 16;
    for i in done..uv_samples {
        u_out[i] = uv_interleaved[i * 2];
        v_out[i] = uv_interleaved[i * 2 + 1];
    }

    (y[..width * height].to_vec(), u_out, v_out)
}

#[cfg(target_arch = "aarch64")]
fn i420_to_nv12_neon(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>) {
    use std::arch::aarch64::*;

    let uv_height = (height + 1) / 2;
    let uv_width = (width + 1) / 2;
    let uv_samples = uv_width * uv_height;
    assert!(y.len() >= width * height);
    assert!(u.len() >= uv_samples);
    assert!(v.len() >= uv_samples);

    let mut uv_out = vec![0u8; uv_samples * 2];

    // NEON: vst2q_u8 stores interleaved: (U, V) pairs
    let chunks = uv_samples / 16;
    for chunk in 0..chunks {
        unsafe {
            let u_v = vld1q_u8(u.as_ptr().add(chunk * 16));
            let v_v = vld1q_u8(v.as_ptr().add(chunk * 16));
            let uv = uint8x16x2_t(u_v, v_v);
            vst2q_u8(uv_out.as_mut_ptr().add(chunk * 32), uv);
        }
    }
    let done = chunks * 16;
    for i in done..uv_samples {
        uv_out[i * 2] = u[i];
        uv_out[i * 2 + 1] = v[i];
    }

    (y[..width * height].to_vec(), uv_out)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn make_yuv444(width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let n = width * height;
        let y: Vec<u8> = (0..n).map(|i| (i % 235 + 16) as u8).collect();
        let u: Vec<u8> = (0..n).map(|i| (i % 200 + 16) as u8).collect();
        let v: Vec<u8> = (0..n).map(|i| (i % 180 + 40) as u8).collect();
        (y, u, v)
    }

    fn make_yuv420(width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let luma_n = width * height;
        let uv_w = (width + 1) / 2;
        let uv_h = (height + 1) / 2;
        let uv_n = uv_w * uv_h;
        let y: Vec<u8> = (0..luma_n).map(|i| (i % 235 + 16) as u8).collect();
        let u: Vec<u8> = (0..uv_n).map(|i| (i % 120 + 16) as u8).collect();
        let v: Vec<u8> = (0..uv_n).map(|i| (i % 100 + 50) as u8).collect();
        (y, u, v)
    }

    fn make_yuv422(width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let luma_n = width * height;
        let uv_w = (width + 1) / 2;
        let uv_n = uv_w * height;
        let y: Vec<u8> = (0..luma_n).map(|i| (i % 235 + 16) as u8).collect();
        let u: Vec<u8> = (0..uv_n).map(|i| (i % 120 + 16) as u8).collect();
        let v: Vec<u8> = (0..uv_n).map(|i| (i % 100 + 50) as u8).collect();
        (y, u, v)
    }

    // ── yuv420_to_yuv444 ─────────────────────────────────────────────────────

    #[test]
    fn yuv420_to_yuv444_output_size() {
        let (y, u, v) = make_yuv420(64, 48);
        let (y_o, u_o, v_o) = yuv420_to_yuv444(&y, &u, &v, 64, 48);
        assert_eq!(y_o.len(), 64 * 48);
        assert_eq!(u_o.len(), 64 * 48);
        assert_eq!(v_o.len(), 64 * 48);
    }

    #[test]
    fn yuv420_to_yuv444_luma_unchanged() {
        let (y, u, v) = make_yuv420(32, 32);
        let (y_o, _, _) = yuv420_to_yuv444(&y, &u, &v, 32, 32);
        assert_eq!(y_o, y);
    }

    #[test]
    fn yuv420_to_yuv444_gray_image() {
        let w = 8;
        let h = 8;
        let y = vec![128u8; w * h];
        let u = vec![128u8; (w / 2) * (h / 2)];
        let v = vec![128u8; (w / 2) * (h / 2)];
        let (_, u_o, v_o) = yuv420_to_yuv444(&y, &u, &v, w, h);
        assert!(u_o.iter().all(|&x| x == 128));
        assert!(v_o.iter().all(|&x| x == 128));
    }

    #[test]
    fn yuv420_to_yuv444_chroma_replication() {
        // Single 2×2 block: all 4 luma pixels should get the same chroma.
        let w = 2;
        let h = 2;
        let y = vec![100u8; 4];
        let u = vec![77u8; 1];
        let v = vec![88u8; 1];
        let (_, u_o, v_o) = yuv420_to_yuv444(&y, &u, &v, w, h);
        assert!(u_o.iter().all(|&x| x == 77));
        assert!(v_o.iter().all(|&x| x == 88));
    }

    #[test]
    fn yuv420_to_yuv444_odd_dimensions() {
        let (y, u, v) = make_yuv420(7, 5);
        let (y_o, u_o, v_o) = yuv420_to_yuv444(&y, &u, &v, 7, 5);
        assert_eq!(y_o.len(), 35);
        assert_eq!(u_o.len(), 35);
        assert_eq!(v_o.len(), 35);
    }

    // ── yuv444_to_yuv420 ─────────────────────────────────────────────────────

    #[test]
    fn yuv444_to_yuv420_output_size() {
        let (y, u, v) = make_yuv444(64, 48);
        let (y_o, u_o, v_o) = yuv444_to_yuv420(&y, &u, &v, 64, 48);
        assert_eq!(y_o.len(), 64 * 48);
        assert_eq!(u_o.len(), 32 * 24);
        assert_eq!(v_o.len(), 32 * 24);
    }

    #[test]
    fn yuv444_to_yuv420_luma_unchanged() {
        let (y, u, v) = make_yuv444(16, 16);
        let (y_o, _, _) = yuv444_to_yuv420(&y, &u, &v, 16, 16);
        assert_eq!(y_o, y);
    }

    #[test]
    fn yuv444_to_yuv420_constant_chroma() {
        let w = 4;
        let h = 4;
        let y = vec![100u8; w * h];
        let u = vec![60u8; w * h];
        let v = vec![200u8; w * h];
        let (_, u_o, v_o) = yuv444_to_yuv420(&y, &u, &v, w, h);
        assert!(u_o.iter().all(|&x| x == 60));
        assert!(v_o.iter().all(|&x| x == 200));
    }

    #[test]
    fn yuv444_to_yuv420_averaging() {
        // Known averaging test: 2×2 block with known values.
        let w = 2;
        let h = 2;
        let y = vec![100u8; 4];
        let u = vec![10u8, 20u8, 30u8, 40u8];
        let v = vec![100u8, 110u8, 120u8, 130u8];
        let (_, u_o, v_o) = yuv444_to_yuv420(&y, &u, &v, w, h);
        // Average of 10,20,30,40 = 25 (with rounding)
        assert_eq!(u_o.len(), 1);
        assert_eq!(u_o[0], 25);
        assert_eq!(v_o[0], 115);
    }

    // ── yuv422_to_yuv444 ─────────────────────────────────────────────────────

    #[test]
    fn yuv422_to_yuv444_output_size() {
        let (y, u, v) = make_yuv422(64, 48);
        let (y_o, u_o, v_o) = yuv422_to_yuv444(&y, &u, &v, 64, 48);
        assert_eq!(y_o.len(), 64 * 48);
        assert_eq!(u_o.len(), 64 * 48);
        assert_eq!(v_o.len(), 64 * 48);
    }

    #[test]
    fn yuv422_to_yuv444_luma_unchanged() {
        let (y, u, v) = make_yuv422(32, 16);
        let (y_o, _, _) = yuv422_to_yuv444(&y, &u, &v, 32, 16);
        assert_eq!(y_o, y);
    }

    #[test]
    fn yuv422_to_yuv444_horizontal_replication() {
        let w = 4;
        let h = 1;
        let y = vec![100u8; w * h];
        let u = vec![10u8, 20u8]; // 2 chroma samples for width=4
        let v = vec![30u8, 40u8];
        let (_, u_o, v_o) = yuv422_to_yuv444(&y, &u, &v, w, h);
        // Cols 0,1 get u=10; cols 2,3 get u=20
        assert_eq!(u_o[0], 10);
        assert_eq!(u_o[1], 10);
        assert_eq!(u_o[2], 20);
        assert_eq!(u_o[3], 20);
        assert_eq!(v_o[0], 30);
        assert_eq!(v_o[3], 40);
    }

    // ── yuv444_to_yuv422 ─────────────────────────────────────────────────────

    #[test]
    fn yuv444_to_yuv422_output_size() {
        let (y, u, v) = make_yuv444(64, 48);
        let (y_o, u_o, v_o) = yuv444_to_yuv422(&y, &u, &v, 64, 48);
        assert_eq!(y_o.len(), 64 * 48);
        assert_eq!(u_o.len(), 32 * 48);
        assert_eq!(v_o.len(), 32 * 48);
    }

    #[test]
    fn yuv444_to_yuv422_luma_unchanged() {
        let (y, u, v) = make_yuv444(16, 8);
        let (y_o, _, _) = yuv444_to_yuv422(&y, &u, &v, 16, 8);
        assert_eq!(y_o, y);
    }

    #[test]
    fn yuv444_to_yuv422_horizontal_averaging() {
        let w = 4;
        let h = 1;
        let y = vec![100u8; w];
        let u = vec![10u8, 20u8, 30u8, 40u8];
        let v = vec![50u8, 60u8, 70u8, 80u8];
        let (_, u_o, v_o) = yuv444_to_yuv422(&y, &u, &v, w, h);
        // (10+20+1)/2 = 15, (30+40+1)/2 = 35
        assert_eq!(u_o[0], 15);
        assert_eq!(u_o[1], 35);
        assert_eq!(v_o[0], 55);
        assert_eq!(v_o[1], 75);
    }

    // ── 422 round-trip ───────────────────────────────────────────────────────

    #[test]
    fn yuv422_444_422_roundtrip_constant() {
        // Constant chroma: round-trip should be lossless.
        let w = 32;
        let h = 16;
        let (y_orig, u_orig, v_orig) = make_yuv422(w, h);
        let (y444, u444, v444) = yuv422_to_yuv444(&y_orig, &u_orig, &v_orig, w, h);
        let (y_rt, u_rt, v_rt) = yuv444_to_yuv422(&y444, &u444, &v444, w, h);
        assert_eq!(y_rt, y_orig, "luma must survive round-trip");
        // Chroma should be close but not necessarily identical (avg of replicated)
        for ((&a, &b), (&c, &d)) in u_orig
            .iter()
            .zip(u_rt.iter())
            .zip(v_orig.iter().zip(v_rt.iter()))
        {
            let diff_u = (i32::from(a) - i32::from(b)).unsigned_abs();
            let diff_v = (i32::from(c) - i32::from(d)).unsigned_abs();
            assert!(diff_u <= 1, "U chroma diff {} exceeds 1", diff_u);
            assert!(diff_v <= 1, "V chroma diff {} exceeds 1", diff_v);
        }
    }

    // ── nv12_to_i420 ─────────────────────────────────────────────────────────

    #[test]
    fn nv12_to_i420_output_size() {
        let w = 64;
        let h = 48;
        let y = vec![128u8; w * h];
        let uv = vec![128u8; w * h / 2];
        let (y_o, u_o, v_o) = nv12_to_i420(&y, &uv, w, h);
        assert_eq!(y_o.len(), w * h);
        assert_eq!(u_o.len(), (w / 2) * (h / 2));
        assert_eq!(v_o.len(), (w / 2) * (h / 2));
    }

    #[test]
    fn nv12_to_i420_luma_unchanged() {
        let w = 16;
        let h = 16;
        let y: Vec<u8> = (0..w * h).map(|i| i as u8).collect();
        let uv = vec![128u8; w * h / 2];
        let (y_o, _, _) = nv12_to_i420(&y, &uv, w, h);
        assert_eq!(y_o, y);
    }

    #[test]
    fn nv12_to_i420_deinterleave_known_values() {
        let w = 4;
        let h = 4;
        let y = vec![0u8; w * h];
        // NV12 UV: [U0,V0, U1,V1, U2,V2, U3,V3]
        let uv = vec![10u8, 20, 30, 40, 50, 60, 70, 80];
        let (_, u_o, v_o) = nv12_to_i420(&y, &uv, w, h);
        assert_eq!(u_o, [10, 30, 50, 70]);
        assert_eq!(v_o, [20, 40, 60, 80]);
    }

    // ── i420_to_nv12 ─────────────────────────────────────────────────────────

    #[test]
    fn i420_to_nv12_output_size() {
        let w = 64;
        let h = 48;
        let y = vec![128u8; w * h];
        let u = vec![128u8; (w / 2) * (h / 2)];
        let v = vec![128u8; (w / 2) * (h / 2)];
        let (y_o, uv_o) = i420_to_nv12(&y, &u, &v, w, h);
        assert_eq!(y_o.len(), w * h);
        assert_eq!(uv_o.len(), (w / 2) * (h / 2) * 2);
    }

    #[test]
    fn i420_to_nv12_luma_unchanged() {
        let w = 16;
        let h = 16;
        let y: Vec<u8> = (0..w * h).map(|i| i as u8).collect();
        let u = vec![128u8; (w / 2) * (h / 2)];
        let v = vec![128u8; (w / 2) * (h / 2)];
        let (y_o, _) = i420_to_nv12(&y, &u, &v, w, h);
        assert_eq!(y_o, y);
    }

    #[test]
    fn i420_to_nv12_interleave_known_values() {
        let w = 4;
        let h = 4;
        let y = vec![0u8; w * h];
        let u = [10u8, 30, 50, 70];
        let v = [20u8, 40, 60, 80];
        let (_, uv_o) = i420_to_nv12(&y, &u, &v, w, h);
        assert_eq!(uv_o, [10, 20, 30, 40, 50, 60, 70, 80]);
    }

    // ── NV12 round-trip ──────────────────────────────────────────────────────

    #[test]
    fn nv12_i420_nv12_roundtrip() {
        let w = 32;
        let h = 32;
        let y: Vec<u8> = (0..w * h).map(|i| (i % 235 + 16) as u8).collect();
        let u: Vec<u8> = (0..(w / 2) * (h / 2))
            .map(|i| (i % 120 + 16) as u8)
            .collect();
        let v: Vec<u8> = (0..(w / 2) * (h / 2))
            .map(|i| (i % 100 + 50) as u8)
            .collect();
        let (y_nv, uv_nv) = i420_to_nv12(&y, &u, &v, w, h);
        let (y_rt, u_rt, v_rt) = nv12_to_i420(&y_nv, &uv_nv, w, h);
        assert_eq!(y_rt, y);
        assert_eq!(u_rt, u);
        assert_eq!(v_rt, v);
    }

    #[test]
    fn nv12_i420_roundtrip_large() {
        let w = 128;
        let h = 128;
        let y: Vec<u8> = (0..w * h).map(|i| (i % 256) as u8).collect();
        let u: Vec<u8> = (0..(w / 2) * (h / 2)).map(|i| (i % 256) as u8).collect();
        let v: Vec<u8> = (0..(w / 2) * (h / 2))
            .map(|i| (255 - i % 256) as u8)
            .collect();
        let (_, uv) = i420_to_nv12(&y, &u, &v, w, h);
        let (_, u_rt, v_rt) = nv12_to_i420(&y, &uv, w, h);
        assert_eq!(u_rt, u);
        assert_eq!(v_rt, v);
    }

    // ── 420 round-trip ───────────────────────────────────────────────────────

    #[test]
    fn yuv420_444_420_roundtrip_constant_chroma() {
        // When chroma is constant, the round-trip is lossless.
        let w = 32;
        let h = 32;
        let y = vec![128u8; w * h];
        let u = vec![77u8; (w / 2) * (h / 2)];
        let v = vec![88u8; (w / 2) * (h / 2)];
        let (y444, u444, v444) = yuv420_to_yuv444(&y, &u, &v, w, h);
        let (y_rt, u_rt, v_rt) = yuv444_to_yuv420(&y444, &u444, &v444, w, h);
        assert_eq!(y_rt, y);
        assert_eq!(u_rt, u);
        assert_eq!(v_rt, v);
    }

    #[test]
    fn yuv420_to_yuv444_large_frame() {
        let w = 256;
        let h = 256;
        let (y, u, v) = make_yuv420(w, h);
        let (y_o, u_o, v_o) = yuv420_to_yuv444(&y, &u, &v, w, h);
        assert_eq!(y_o.len(), w * h);
        assert_eq!(u_o.len(), w * h);
        assert_eq!(v_o.len(), w * h);
    }
}
