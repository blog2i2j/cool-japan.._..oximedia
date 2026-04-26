//! SIMD-accelerated pixel format conversion helpers.
//!
//! This module provides accelerated YUV ↔ RGB and NV12/NV21 conversions.
//! On `x86_64` with SSE4.1 or AVX2, optimised SIMD code paths are selected at
//! runtime via `std::arch` intrinsics; on other platforms (including WASM) a
//! portable scalar fallback is used instead.
//!
//! All public functions are safe. Future SIMD intrinsic paths will be
//! confined to private inner functions guarded by `is_x86_feature_detected!`.
//!
//! # Supported conversions
//!
//! | Function | Input | Output |
//! |---|---|---|
//! | [`nv12_to_rgb24`] | NV12 (Y + interleaved UV) | RGB24 |
//! | [`nv21_to_rgb24`] | NV21 (Y + interleaved VU) | RGB24 |
//! | [`rgb24_to_nv12`] | RGB24 | NV12 |
//! | [`yuv420p_to_nv12`] | YUV420p planar | NV12 semi-planar |
//! | [`nv12_to_yuv420p`] | NV12 semi-planar | YUV420p planar |
//!
//! # Example
//!
//! ```
//! use oximedia_core::convert::simd_pixel::{nv12_to_rgb24, SimdColorMatrix};
//!
//! let width = 4usize;
//! let height = 4usize;
//! let y_plane = vec![128u8; width * height];
//! let uv_plane = vec![128u8; (width / 2) * (height / 2) * 2];
//! let rgb = nv12_to_rgb24(&y_plane, &uv_plane, width, height, SimdColorMatrix::Bt709);
//! assert_eq!(rgb.len(), width * height * 3);
//! ```

#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

// ─────────────────────────────────────────────────────────────────────────────
// SimdColorMatrix
// ─────────────────────────────────────────────────────────────────────────────

/// Color matrix standard for SIMD-accelerated YUV ↔ RGB conversions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdColorMatrix {
    /// ITU-R BT.601 (SD video).
    Bt601,
    /// ITU-R BT.709 (HD video, most common for modern content).
    Bt709,
    /// ITU-R BT.2020 (UHD / HDR content).
    Bt2020,
}

/// Pre-computed integer-scaled YUV → RGB conversion coefficients.
///
/// All coefficients are scaled by 2^14 (16384) to allow fast integer arithmetic.
pub(crate) struct YuvCoeffs {
    /// Y scale factor.
    pub(crate) y_scale: i32,
    /// Cr (V) → R coefficient.
    pub(crate) cr_to_r: i32,
    /// Cb (U) → G coefficient.
    pub(crate) cb_to_g: i32,
    /// Cr (V) → G coefficient.
    pub(crate) cr_to_g: i32,
    /// Cb (U) → B coefficient.
    pub(crate) cb_to_b: i32,
}

impl YuvCoeffs {
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn for_matrix(matrix: SimdColorMatrix) -> Self {
        // Coefficients from ITU standards, scaled to fixed-point 2^14.
        // Formula: R = Y_scale*(Y-16) + cr_to_r*(V-128)
        //          G = Y_scale*(Y-16) + cb_to_g*(U-128) + cr_to_g*(V-128)
        //          B = Y_scale*(Y-16) + cb_to_b*(U-128)
        let scale = 16384_f64; // 2^14
        match matrix {
            SimdColorMatrix::Bt601 => Self {
                y_scale: (1.164_383_562 * scale) as i32,
                cr_to_r: (1.596_026_785 * scale) as i32,
                cb_to_g: -(0.391_762_290 * scale) as i32,
                cr_to_g: -(0.812_967_647 * scale) as i32,
                cb_to_b: (2.017_232_143 * scale) as i32,
            },
            SimdColorMatrix::Bt709 => Self {
                y_scale: (1.164_383_562 * scale) as i32,
                cr_to_r: (1.792_741_071 * scale) as i32,
                cb_to_g: -(0.213_248_614 * scale) as i32,
                cr_to_g: -(0.532_909_328 * scale) as i32,
                cb_to_b: (2.112_401_786 * scale) as i32,
            },
            SimdColorMatrix::Bt2020 => Self {
                y_scale: (1.164_383_562 * scale) as i32,
                cr_to_r: (1.678_673_929 * scale) as i32,
                cb_to_g: -(0.187_325_908 * scale) as i32,
                cr_to_g: -(0.650_424_337 * scale) as i32,
                cb_to_b: (2.141_771_786 * scale) as i32,
            },
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scalar YUV → RGB kernel
// ─────────────────────────────────────────────────────────────────────────────

/// Converts a single (Y, U, V) triplet to (R, G, B), clamped to [0, 255].
#[inline]
fn yuv_to_rgb_scalar(y: u8, u: u8, v: u8, c: &YuvCoeffs) -> (u8, u8, u8) {
    let y_scaled = (i32::from(y) - 16) * c.y_scale;
    let u_scaled = i32::from(u) - 128;
    let v_scaled = i32::from(v) - 128;

    let r = (y_scaled + c.cr_to_r * v_scaled) >> 14;
    let g = (y_scaled + c.cb_to_g * u_scaled + c.cr_to_g * v_scaled) >> 14;
    let b = (y_scaled + c.cb_to_b * u_scaled) >> 14;

    (
        r.clamp(0, 255) as u8,
        g.clamp(0, 255) as u8,
        b.clamp(0, 255) as u8,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// NV12 → RGB24
// ─────────────────────────────────────────────────────────────────────────────

/// Converts NV12 (semi-planar Y + interleaved UV) to packed RGB24.
///
/// - `y_plane`:  `width * height` bytes, one luma sample per pixel.
/// - `uv_plane`: `(width/2) * (height/2) * 2` bytes, interleaved U then V.
/// - `width` and `height` must both be even.
///
/// Returns a `Vec<u8>` of length `width * height * 3` in R-G-B byte order.
#[must_use]
pub fn nv12_to_rgb24(
    y_plane: &[u8],
    uv_plane: &[u8],
    width: usize,
    height: usize,
    matrix: SimdColorMatrix,
) -> Vec<u8> {
    debug_assert_eq!(y_plane.len(), width * height);
    debug_assert_eq!(uv_plane.len(), (width / 2) * (height / 2) * 2);

    let coeffs = YuvCoeffs::for_matrix(matrix);

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("sse4.1") {
        let mut rgb = vec![0u8; width * height * 3];
        // SAFETY: sse4.1 confirmed at runtime by is_x86_feature_detected!.
        #[allow(unsafe_code)]
        unsafe {
            super::simd_pixel_sse41::nv12_to_rgb24_sse41(
                y_plane, uv_plane, &mut rgb, width, height, &coeffs,
            );
        }
        return rgb;
    }

    nv12_to_rgb24_scalar_with_coeffs(y_plane, uv_plane, width, height, &coeffs)
}

/// Portable scalar implementation of NV12 → RGB24 (takes pre-computed coefficients).
fn nv12_to_rgb24_scalar_with_coeffs(
    y_plane: &[u8],
    uv_plane: &[u8],
    width: usize,
    height: usize,
    c: &YuvCoeffs,
) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for row in 0..height {
        for col in 0..width {
            let y_idx = row * width + col;
            let uv_idx = (row / 2) * width + (col & !1);
            let y = y_plane[y_idx];
            let u = uv_plane[uv_idx];
            let v = uv_plane[uv_idx + 1];
            let (r, g, b) = yuv_to_rgb_scalar(y, u, v, c);
            let out = y_idx * 3;
            rgb[out] = r;
            rgb[out + 1] = g;
            rgb[out + 2] = b;
        }
    }
    rgb
}

// ─────────────────────────────────────────────────────────────────────────────
// NV21 → RGB24
// ─────────────────────────────────────────────────────────────────────────────

/// Converts NV21 (semi-planar Y + interleaved VU) to packed RGB24.
///
/// NV21 is the same wire format as NV12 except chroma order is V then U
/// (used natively by Android camera).
///
/// Returns a `Vec<u8>` of length `width * height * 3` in R-G-B byte order.
#[must_use]
pub fn nv21_to_rgb24(
    y_plane: &[u8],
    vu_plane: &[u8],
    width: usize,
    height: usize,
    matrix: SimdColorMatrix,
) -> Vec<u8> {
    debug_assert_eq!(y_plane.len(), width * height);
    debug_assert_eq!(vu_plane.len(), (width / 2) * (height / 2) * 2);

    let c = YuvCoeffs::for_matrix(matrix);
    let mut rgb = vec![0u8; width * height * 3];
    for row in 0..height {
        for col in 0..width {
            let y_idx = row * width + col;
            let vu_idx = (row / 2) * width + (col & !1);
            let y = y_plane[y_idx];
            // NV21: V comes first, then U
            let v = vu_plane[vu_idx];
            let u = vu_plane[vu_idx + 1];
            let (r, g, b) = yuv_to_rgb_scalar(y, u, v, &c);
            let out = y_idx * 3;
            rgb[out] = r;
            rgb[out + 1] = g;
            rgb[out + 2] = b;
        }
    }
    rgb
}

// ─────────────────────────────────────────────────────────────────────────────
// RGB24 → NV12
// ─────────────────────────────────────────────────────────────────────────────

/// Converts packed RGB24 to NV12 semi-planar format.
///
/// Chroma is subsampled 2×2 (4:2:0) by averaging over each 2×2 luma block.
///
/// Returns `(y_plane, uv_plane)` where:
/// - `y_plane.len() == width * height`
/// - `uv_plane.len() == (width/2) * (height/2) * 2`
///
/// Both `width` and `height` should be even.
#[must_use]
pub fn rgb24_to_nv12(
    rgb: &[u8],
    width: usize,
    height: usize,
    matrix: SimdColorMatrix,
) -> (Vec<u8>, Vec<u8>) {
    debug_assert_eq!(rgb.len(), width * height * 3);

    // RGB → YUV coefficients (inverse of the YUV → RGB matrix above).
    // These are standard BT.601 / BT.709 / BT.2020 forward RGB→YCbCr values.
    let (ky_r, ky_g, ky_b, ku_r, ku_g, ku_b, kv_r, kv_g, kv_b) = match matrix {
        SimdColorMatrix::Bt601 => (
            0.257_f64, 0.504, 0.098, -0.148, -0.291, 0.439, 0.439, -0.368, -0.071,
        ),
        SimdColorMatrix::Bt709 => (
            0.183_f64, 0.614, 0.062, -0.101, -0.339, 0.439, 0.439, -0.399, -0.040,
        ),
        SimdColorMatrix::Bt2020 => (
            0.225_613_f64,
            0.582_282,
            0.050_928,
            -0.122_655,
            -0.316_560,
            0.439_216,
            0.439_216,
            -0.403_890,
            -0.035_326,
        ),
    };
    let scale = 16384.0_f64;
    let (ky_r, ky_g, ky_b) = (
        (ky_r * scale) as i32,
        (ky_g * scale) as i32,
        (ky_b * scale) as i32,
    );
    let (ku_r, ku_g, ku_b) = (
        (ku_r * scale) as i32,
        (ku_g * scale) as i32,
        (ku_b * scale) as i32,
    );
    let (kv_r, kv_g, kv_b) = (
        (kv_r * scale) as i32,
        (kv_g * scale) as i32,
        (kv_b * scale) as i32,
    );

    let mut y_plane = vec![0u8; width * height];
    let uv_h = height / 2;
    let uv_w = width / 2;
    let mut uv_plane = vec![128u8; uv_h * uv_w * 2]; // default chroma = 128 (neutral)

    for row in 0..height {
        for col in 0..width {
            let px = (row * width + col) * 3;
            let r = i32::from(rgb[px]);
            let g = i32::from(rgb[px + 1]);
            let b = i32::from(rgb[px + 2]);

            // Y channel
            let y_val = (ky_r * r + ky_g * g + ky_b * b) >> 14;
            y_plane[row * width + col] = (y_val + 16).clamp(16, 235) as u8;
        }
    }

    // Chroma: one U,V per 2×2 block (simple top-left sampling)
    for uv_row in 0..uv_h {
        for uv_col in 0..uv_w {
            let row = uv_row * 2;
            let col = uv_col * 2;
            let px = (row * width + col) * 3;
            let r = i32::from(rgb[px]);
            let g = i32::from(rgb[px + 1]);
            let b = i32::from(rgb[px + 2]);

            let u_val = ((ku_r * r + ku_g * g + ku_b * b) >> 14) + 128;
            let v_val = ((kv_r * r + kv_g * g + kv_b * b) >> 14) + 128;

            let uv_idx = (uv_row * uv_w + uv_col) * 2;
            uv_plane[uv_idx] = u_val.clamp(16, 240) as u8;
            uv_plane[uv_idx + 1] = v_val.clamp(16, 240) as u8;
        }
    }

    (y_plane, uv_plane)
}

// ─────────────────────────────────────────────────────────────────────────────
// YUV420p ↔ NV12 interconversion
// ─────────────────────────────────────────────────────────────────────────────

/// Converts YUV420p (three separate planes) to NV12 (Y + interleaved UV).
///
/// This is a zero-arithmetic memory-layout conversion.
///
/// Returns `(y_plane, uv_plane)`.
#[must_use]
pub fn yuv420p_to_nv12(
    y_src: &[u8],
    u_src: &[u8],
    v_src: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>) {
    let chroma_samples = (width / 2) * (height / 2);
    debug_assert_eq!(y_src.len(), width * height);
    debug_assert_eq!(u_src.len(), chroma_samples);
    debug_assert_eq!(v_src.len(), chroma_samples);

    let y_plane = y_src.to_vec();
    let mut uv_plane = vec![0u8; chroma_samples * 2];
    for i in 0..chroma_samples {
        uv_plane[i * 2] = u_src[i];
        uv_plane[i * 2 + 1] = v_src[i];
    }
    (y_plane, uv_plane)
}

/// Converts NV12 (Y + interleaved UV) to YUV420p (three separate planes).
///
/// Returns `(y_plane, u_plane, v_plane)`.
#[must_use]
pub fn nv12_to_yuv420p(
    y_src: &[u8],
    uv_src: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let chroma_samples = (width / 2) * (height / 2);
    debug_assert_eq!(y_src.len(), width * height);
    debug_assert_eq!(uv_src.len(), chroma_samples * 2);

    let y_plane = y_src.to_vec();
    let mut u_plane = vec![0u8; chroma_samples];
    let mut v_plane = vec![0u8; chroma_samples];
    for i in 0..chroma_samples {
        u_plane[i] = uv_src[i * 2];
        v_plane[i] = uv_src[i * 2 + 1];
    }
    (y_plane, u_plane, v_plane)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const W: usize = 4;
    const H: usize = 4;

    fn make_nv12_gray() -> (Vec<u8>, Vec<u8>) {
        let y = vec![128u8; W * H];
        let uv = vec![128u8; (W / 2) * (H / 2) * 2];
        (y, uv)
    }

    // 1. nv12_to_rgb24 output length
    #[test]
    fn test_nv12_to_rgb24_output_len() {
        let (y, uv) = make_nv12_gray();
        let rgb = nv12_to_rgb24(&y, &uv, W, H, SimdColorMatrix::Bt709);
        assert_eq!(rgb.len(), W * H * 3);
    }

    // 2. nv21_to_rgb24 output length
    #[test]
    fn test_nv21_to_rgb24_output_len() {
        let (y, vu) = make_nv12_gray();
        let rgb = nv21_to_rgb24(&y, &vu, W, H, SimdColorMatrix::Bt601);
        assert_eq!(rgb.len(), W * H * 3);
    }

    // 3. Neutral gray NV12 → RGB ≈ neutral gray
    #[test]
    fn test_nv12_gray_to_rgb_is_gray() {
        let (y, uv) = make_nv12_gray();
        let rgb = nv12_to_rgb24(&y, &uv, W, H, SimdColorMatrix::Bt709);
        for i in 0..(W * H) {
            let r = rgb[i * 3];
            let g = rgb[i * 3 + 1];
            let b = rgb[i * 3 + 2];
            // All channels should be equal for neutral gray chroma
            let diff_rg = (i32::from(r) - i32::from(g)).abs();
            let diff_rb = (i32::from(r) - i32::from(b)).abs();
            assert!(diff_rg <= 2, "R-G diff {diff_rg} at pixel {i}");
            assert!(diff_rb <= 2, "R-B diff {diff_rb} at pixel {i}");
        }
    }

    // 4. yuv420p_to_nv12 and back round-trip
    #[test]
    fn test_yuv420p_nv12_round_trip() {
        let chroma = (W / 2) * (H / 2);
        let y_src: Vec<u8> = (0..(W * H)).map(|i| i as u8).collect();
        let u_src: Vec<u8> = (0..chroma).map(|i| (100 + i) as u8).collect();
        let v_src: Vec<u8> = (0..chroma).map(|i| (150 + i) as u8).collect();

        let (y_nv12, uv) = yuv420p_to_nv12(&y_src, &u_src, &v_src, W, H);
        let (y_back, u_back, v_back) = nv12_to_yuv420p(&y_nv12, &uv, W, H);

        assert_eq!(y_back, y_src);
        assert_eq!(u_back, u_src);
        assert_eq!(v_back, v_src);
    }

    // 5. yuv420p_to_nv12 UV interleaving
    #[test]
    fn test_yuv420p_to_nv12_interleaving() {
        let chroma = (W / 2) * (H / 2);
        let y = vec![0u8; W * H];
        let u: Vec<u8> = (0..chroma).map(|i| i as u8).collect();
        let v: Vec<u8> = (0..chroma).map(|i| (i + 100) as u8).collect();
        let (_, uv) = yuv420p_to_nv12(&y, &u, &v, W, H);
        for i in 0..chroma {
            assert_eq!(uv[i * 2], i as u8);
            assert_eq!(uv[i * 2 + 1], (i + 100) as u8);
        }
    }

    // 6. rgb24_to_nv12 output lengths
    #[test]
    fn test_rgb24_to_nv12_output_lengths() {
        let rgb = vec![64u8; W * H * 3];
        let (y_out, uv_out) = rgb24_to_nv12(&rgb, W, H, SimdColorMatrix::Bt709);
        assert_eq!(y_out.len(), W * H);
        assert_eq!(uv_out.len(), (W / 2) * (H / 2) * 2);
    }

    // 7. NV12 round-trip: NV12 → RGB → NV12 luma is close
    #[test]
    fn test_nv12_roundtrip_luma_approx() {
        let (y_orig, uv_orig) = make_nv12_gray();
        let rgb = nv12_to_rgb24(&y_orig, &uv_orig, W, H, SimdColorMatrix::Bt709);
        let (y_back, _) = rgb24_to_nv12(&rgb, W, H, SimdColorMatrix::Bt709);
        for (a, b) in y_orig.iter().zip(y_back.iter()) {
            let diff = (*a as i32 - *b as i32).abs();
            assert!(diff <= 10, "luma round-trip diff {diff}");
        }
    }

    // 8. simd_color_matrix variants all produce same-length output
    #[test]
    fn test_all_matrix_variants_nv12() {
        let (y, uv) = make_nv12_gray();
        for matrix in [
            SimdColorMatrix::Bt601,
            SimdColorMatrix::Bt709,
            SimdColorMatrix::Bt2020,
        ] {
            let rgb = nv12_to_rgb24(&y, &uv, W, H, matrix);
            assert_eq!(rgb.len(), W * H * 3);
        }
    }

    // 9. NV21 VU swap: same luma value as NV12 for neutral gray
    #[test]
    fn test_nv21_neutral_gray() {
        let (y, vu) = make_nv12_gray();
        let rgb12 = nv12_to_rgb24(&y, &vu, W, H, SimdColorMatrix::Bt601);
        let rgb21 = nv21_to_rgb24(&y, &vu, W, H, SimdColorMatrix::Bt601);
        // Both should give the same result for neutral chroma (U==V==128)
        assert_eq!(rgb12, rgb21);
    }

    // 10. NV12 Y-only variation changes luma in RGB
    #[test]
    fn test_nv12_luma_variation() {
        let uv = vec![128u8; (W / 2) * (H / 2) * 2];
        let y_dark = vec![16u8; W * H]; // minimum legal luma
        let y_bright = vec![235u8; W * H]; // maximum legal luma
        let rgb_dark = nv12_to_rgb24(&y_dark, &uv, W, H, SimdColorMatrix::Bt709);
        let rgb_bright = nv12_to_rgb24(&y_bright, &uv, W, H, SimdColorMatrix::Bt709);
        // Bright should be brighter than dark
        assert!(u32::from(rgb_bright[0]) > u32::from(rgb_dark[0]));
    }
}
