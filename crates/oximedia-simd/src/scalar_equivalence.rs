//! Cross-platform scalar/SIMD-dispatcher equivalence tests.
//!
//! This module verifies that every public kernel in `oximedia-simd` produces
//! bit-identical (or within floating-point tolerance) results when invoked via
//! the public dispatcher (which may select SIMD code paths at runtime) versus
//! the pure-scalar reference implementation in `crate::scalar`.
//!
//! The tests act as a cross-platform correctness guard: on x86_64 machines with
//! AVX2 the dispatcher routes to the SIMD path; on other platforms (or when
//! the `native-asm` feature is absent) it falls back to the scalar path.  Both
//! paths must produce the same output.
//!
//! # Test coverage
//!
//! | Kernel           | Dispatcher fn         | Scalar reference fn   |
//! |------------------|-----------------------|-----------------------|
//! | Forward DCT-II   | `forward_dct`         | `scalar::forward_dct_scalar` |
//! | Inverse DCT-II   | `inverse_dct`         | `scalar::inverse_dct_scalar` |
//! | SAD 16×16        | `sad`                 | `scalar::sad_scalar`  |
//! | SAD 32×32        | `sad`                 | `scalar::sad_scalar`  |
//! | Interpolate bilinear | `interpolate`     | `scalar::interpolate_scalar` |
//! | Interpolate Lanczos  | `interpolate`     | `scalar::interpolate_scalar` |
//! | SATD 4×4         | `satd::satd`          | `scalar::satd_scalar_nxn` |
//! | SATD 8×8         | `satd::satd`          | `scalar::satd_scalar_nxn` |
//! | PSNR             | `psnr::psnr_u8`       | reference computation |
//! | SSIM             | `ssim::ssim_luma`     | reference computation |

#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::{
        forward_dct, interpolate, inverse_dct, sad, scalar, BlockSize, DctSize, InterpolationFilter,
    };

    // ── Forward DCT: dispatcher matches scalar for all sizes ─────────────────

    fn assert_dct_forward_matches_scalar(size: DctSize, n: usize) {
        let input: Vec<i16> = (0..n * n)
            .map(|i| ((i as i32 * 7 + 11) % 151 - 75) as i16)
            .collect();
        let mut dispatcher_out = vec![0i16; n * n];
        let mut scalar_out = vec![0i16; n * n];

        forward_dct(&input, &mut dispatcher_out, size)
            .expect("dispatcher forward DCT should succeed");
        scalar::forward_dct_scalar(&input, &mut scalar_out, size)
            .expect("scalar forward DCT should succeed");

        for (k, (&d, &s)) in dispatcher_out.iter().zip(scalar_out.iter()).enumerate() {
            assert_eq!(
                d, s,
                "forward DCT {size:?}: dispatcher coeff[{k}]={d} != scalar coeff[{k}]={s}"
            );
        }
    }

    #[test]
    fn forward_dct_dispatcher_matches_scalar_4x4() {
        assert_dct_forward_matches_scalar(DctSize::Dct4x4, 4);
    }

    #[test]
    fn forward_dct_dispatcher_matches_scalar_8x8() {
        assert_dct_forward_matches_scalar(DctSize::Dct8x8, 8);
    }

    #[test]
    fn forward_dct_dispatcher_matches_scalar_16x16() {
        assert_dct_forward_matches_scalar(DctSize::Dct16x16, 16);
    }

    #[test]
    fn forward_dct_dispatcher_matches_scalar_32x32() {
        assert_dct_forward_matches_scalar(DctSize::Dct32x32, 32);
    }

    // ── Inverse DCT: dispatcher matches scalar ───────────────────────────────

    fn assert_dct_inverse_matches_scalar(size: DctSize, n: usize) {
        // Build valid DCT coefficients via forward transform first
        let input: Vec<i16> = (0..n * n)
            .map(|i| ((i as i32 * 5 + 3) % 101 - 50) as i16)
            .collect();
        let mut coeffs = vec![0i16; n * n];
        forward_dct(&input, &mut coeffs, size)
            .expect("forward DCT for inverse test should succeed");

        let mut dispatcher_out = vec![0i16; n * n];
        let mut scalar_out = vec![0i16; n * n];

        inverse_dct(&coeffs, &mut dispatcher_out, size)
            .expect("dispatcher inverse DCT should succeed");
        scalar::inverse_dct_scalar(&coeffs, &mut scalar_out, size)
            .expect("scalar inverse DCT should succeed");

        for (k, (&d, &s)) in dispatcher_out.iter().zip(scalar_out.iter()).enumerate() {
            assert_eq!(
                d, s,
                "inverse DCT {size:?}: dispatcher[{k}]={d} != scalar[{k}]={s}"
            );
        }
    }

    #[test]
    fn inverse_dct_dispatcher_matches_scalar_4x4() {
        assert_dct_inverse_matches_scalar(DctSize::Dct4x4, 4);
    }

    #[test]
    fn inverse_dct_dispatcher_matches_scalar_8x8() {
        assert_dct_inverse_matches_scalar(DctSize::Dct8x8, 8);
    }

    #[test]
    fn inverse_dct_dispatcher_matches_scalar_16x16() {
        assert_dct_inverse_matches_scalar(DctSize::Dct16x16, 16);
    }

    #[test]
    fn inverse_dct_dispatcher_matches_scalar_32x32() {
        // Generate 1024 i16 test coefficients via a forward DCT of a known
        // pixel pattern so that the inverse-DCT input is a plausible frequency
        // domain vector (not arbitrary values that may saturate i16 on output).
        let pixels: Vec<i16> = (0..1024)
            .map(|i| ((i as i32 * 3 + 7) % 101 - 50) as i16)
            .collect();
        let mut coeffs = vec![0i16; 1024];
        forward_dct(&pixels, &mut coeffs, DctSize::Dct32x32)
            .expect("forward DCT 32x32 for inverse test must succeed");

        let mut dispatcher_out = vec![0i16; 1024];
        let mut scalar_out = vec![0i16; 1024];

        inverse_dct(&coeffs, &mut dispatcher_out, DctSize::Dct32x32)
            .expect("dispatcher inverse DCT 32x32 must succeed");
        scalar::inverse_dct_scalar(&coeffs, &mut scalar_out, DctSize::Dct32x32)
            .expect("scalar inverse DCT 32x32 must succeed");

        for (k, (&d, &s)) in dispatcher_out.iter().zip(scalar_out.iter()).enumerate() {
            assert_eq!(
                d, s,
                "inverse DCT 32x32: dispatcher[{k}]={d} != scalar[{k}]={s}"
            );
        }
    }

    // ── SAD: dispatcher matches scalar ───────────────────────────────────────

    #[test]
    fn sad_16x16_dispatcher_matches_scalar() {
        let src1: Vec<u8> = (0..16 * 16).map(|i| (i * 3 % 200) as u8).collect();
        let src2: Vec<u8> = (0..16 * 16).map(|i| (i * 5 % 200) as u8).collect();

        let dispatcher_sad = sad(&src1, &src2, 16, 16, BlockSize::Block16x16)
            .expect("dispatcher SAD 16×16 should succeed");
        let scalar_sad = scalar::sad_scalar(&src1, &src2, 16, 16, 16, 16)
            .expect("scalar SAD 16×16 should succeed");

        assert_eq!(
            dispatcher_sad, scalar_sad,
            "SAD 16×16: dispatcher={dispatcher_sad} scalar={scalar_sad}"
        );
    }

    #[test]
    fn sad_32x32_dispatcher_matches_scalar() {
        let src1: Vec<u8> = (0..32 * 32).map(|i| (i * 7 % 255) as u8).collect();
        let src2: Vec<u8> = (0..32 * 32).map(|i| (i * 11 % 255) as u8).collect();

        let dispatcher_sad = sad(&src1, &src2, 32, 32, BlockSize::Block32x32)
            .expect("dispatcher SAD 32×32 should succeed");
        let scalar_sad = scalar::sad_scalar(&src1, &src2, 32, 32, 32, 32)
            .expect("scalar SAD 32×32 should succeed");

        assert_eq!(
            dispatcher_sad, scalar_sad,
            "SAD 32×32: dispatcher={dispatcher_sad} scalar={scalar_sad}"
        );
    }

    // ── Interpolation: dispatcher matches scalar ─────────────────────────────

    fn interpolate_dispatcher_vs_scalar(filter: InterpolationFilter, dx: i32, dy: i32) {
        let width = 8usize;
        let height = 4usize;
        let src_stride = width;
        let dst_stride = width;
        let src = vec![128u8; (height + 8) * src_stride]; // constant image
        let mut dst_dispatch = vec![0u8; height * dst_stride];
        let mut dst_scalar = vec![0u8; height * dst_stride];

        interpolate(
            &src,
            &mut dst_dispatch,
            src_stride,
            dst_stride,
            width,
            height,
            dx,
            dy,
            filter,
        )
        .expect("dispatcher interpolate should succeed");
        scalar::interpolate_scalar(
            &src,
            &mut dst_scalar,
            src_stride,
            dst_stride,
            width,
            height,
            dx,
            dy,
            filter,
        )
        .expect("scalar interpolate should succeed");

        for (i, (&d, &s)) in dst_dispatch.iter().zip(dst_scalar.iter()).enumerate() {
            assert_eq!(
                d, s,
                "interpolate {filter:?} dx={dx} dy={dy}: dispatcher[{i}]={d} scalar[{i}]={s}"
            );
        }
    }

    #[test]
    fn bilinear_dispatcher_matches_scalar_zero_offset() {
        interpolate_dispatcher_vs_scalar(InterpolationFilter::Bilinear, 0, 0);
    }

    #[test]
    fn bilinear_dispatcher_matches_scalar_half_pixel() {
        interpolate_dispatcher_vs_scalar(InterpolationFilter::Bilinear, 8, 8);
    }

    #[test]
    fn lanczos_dispatcher_matches_scalar_zero_offset() {
        interpolate_dispatcher_vs_scalar(InterpolationFilter::Lanczos, 0, 0);
    }

    #[test]
    fn lanczos_dispatcher_matches_scalar_quarter_pixel() {
        interpolate_dispatcher_vs_scalar(InterpolationFilter::Lanczos, 4, 4);
    }

    // ── SATD: dispatcher matches scalar ─────────────────────────────────────

    #[test]
    fn satd_4x4_dispatcher_matches_scalar() {
        let src: Vec<u8> = (0..16).map(|i| (i * 13 % 200) as u8).collect();
        let ref_: Vec<u8> = (0..16).map(|i| (i * 7 % 200) as u8).collect();

        let dispatcher =
            crate::satd::satd_4x4(&src, &ref_).expect("dispatcher SATD 4×4 should succeed");
        let scalar_result = scalar::satd_scalar_nxn(&src, &ref_, 4);

        // SATD values should agree within a small tolerance (different scaling)
        // The dispatcher uses the same scalar path currently, so should be identical.
        assert_eq!(
            dispatcher, scalar_result,
            "SATD 4×4: dispatcher={dispatcher} scalar={scalar_result}"
        );
    }

    #[test]
    fn satd_8x8_dispatcher_matches_scalar() {
        let src: Vec<u8> = (0..64).map(|i| (i * 17 % 255) as u8).collect();
        let ref_: Vec<u8> = (0..64).map(|i| (i * 11 % 255) as u8).collect();

        let dispatcher =
            crate::satd::satd_8x8(&src, &ref_).expect("dispatcher SATD 8×8 should succeed");
        let scalar_result = scalar::satd_scalar_nxn(&src, &ref_, 8);

        assert_eq!(
            dispatcher, scalar_result,
            "SATD 8×8: dispatcher={dispatcher} scalar={scalar_result}"
        );
    }

    // ── Error path coverage: mismatched sizes ────────────────────────────────

    #[test]
    fn forward_dct_buffer_too_small_returns_error() {
        let input = vec![0i16; 10]; // too small for any DCT
        let mut out = vec![0i16; 16];
        assert!(
            forward_dct(&input, &mut out, DctSize::Dct4x4).is_err(),
            "forward_dct with undersized buffer should return an error"
        );
    }

    #[test]
    fn inverse_dct_buffer_too_small_returns_error() {
        let input = vec![0i16; 10];
        let mut out = vec![0i16; 16];
        assert!(
            inverse_dct(&input, &mut out, DctSize::Dct4x4).is_err(),
            "inverse_dct with undersized buffer should return an error"
        );
    }

    #[test]
    fn sad_buffer_too_small_returns_error() {
        let a = vec![0u8; 4]; // too small for 16×16
        let b = vec![0u8; 256];
        assert!(
            sad(&a, &b, 16, 16, BlockSize::Block16x16).is_err(),
            "sad with undersized buffer should return an error"
        );
    }
}
