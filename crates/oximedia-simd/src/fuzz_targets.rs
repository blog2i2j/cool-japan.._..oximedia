//! Property-based fuzz targets for SIMD kernels.
//!
//! Uses `proptest` to exercise interpolation, SAD and SATD kernels with
//! randomly generated pixel buffers and block sizes.  Every property verified
//! here must hold regardless of input — they are mathematical invariants of
//! the algorithms, not just implementation tests.
//!
//! These tests run as ordinary `#[cfg(test)]` tests so they execute in CI
//! without any special harness.  The `proptest!` macro generates hundreds of
//! random cases per property automatically.

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use crate::{
        resize::resize_bilinear,
        sad::{block_sad_4x4, block_sad_4x8, block_sad_8x4, block_sad_8x8},
        satd::{satd_4x4, satd_8x8, SatdBlockSize},
    };

    // ── bilinear interpolation properties ──────────────────────────────────────

    proptest! {
        /// Bilinear resize never panics on valid-sized inputs.
        #[test]
        fn prop_bilinear_no_panic(
            src_w in 1usize..=64usize,
            src_h in 1usize..=64usize,
            dst_w in 1usize..=64usize,
            dst_h in 1usize..=64usize,
            fill in 0u8..=255u8,
        ) {
            let src = vec![fill; src_w * src_h];
            let mut dst = vec![0u8; dst_w * dst_h];
            // Must not panic regardless of dimensions.
            let _ = resize_bilinear(&src, src_w, src_h, &mut dst, dst_w, dst_h);
        }

        /// Bilinear resize output pixels are always in [0, 255].
        #[test]
        fn prop_bilinear_output_in_range(
            src_w in 1usize..=32usize,
            src_h in 1usize..=32usize,
            dst_w in 1usize..=32usize,
            dst_h in 1usize..=32usize,
            src in prop::collection::vec(0u8..=255u8, 1..=1024usize),
        ) {
            let actual_src_len = src_w * src_h;
            if src.len() < actual_src_len {
                // proptest may generate a src vec shorter than needed; skip.
                return Ok(());
            }
            let src_trimmed = &src[..actual_src_len];
            let mut dst = vec![0u8; dst_w * dst_h];
            if resize_bilinear(src_trimmed, src_w, src_h, &mut dst, dst_w, dst_h).is_ok() {
                // All u8 values are in [0, 255] by definition — no assertion needed.
                let _ = &dst;
            }
        }

        /// Bilinear resize with a uniform constant source produces a uniform output.
        ///
        /// When every source pixel is the same value `c`, every destination pixel
        /// must also be `c` (bilinear interpolation is linear in pixel values).
        #[test]
        fn prop_bilinear_constant_source_is_constant(
            src_w in 1usize..=32usize,
            src_h in 1usize..=32usize,
            dst_w in 1usize..=32usize,
            dst_h in 1usize..=32usize,
            fill in 0u8..=255u8,
        ) {
            let src = vec![fill; src_w * src_h];
            let mut dst = vec![0u8; dst_w * dst_h];
            if resize_bilinear(&src, src_w, src_h, &mut dst, dst_w, dst_h).is_ok() {
                for &pixel in &dst {
                    prop_assert_eq!(
                        pixel, fill,
                        "constant source produced non-constant output"
                    );
                }
            }
        }

        /// Identity resize (same dimensions) produces identical output.
        #[test]
        fn prop_bilinear_identity_resize(
            w in 1usize..=32usize,
            h in 1usize..=32usize,
            src in prop::collection::vec(0u8..=255u8, 1..=1024usize),
        ) {
            let pixel_count = w * h;
            if src.len() < pixel_count {
                return Ok(());
            }
            let src_trimmed = &src[..pixel_count];
            let mut dst = vec![0u8; pixel_count];
            if resize_bilinear(src_trimmed, w, h, &mut dst, w, h).is_ok() {
                prop_assert_eq!(src_trimmed, dst.as_slice(),
                    "identity resize changed pixel values");
            }
        }

        /// Bilinear resize with empty source buffer returns error, never panics.
        #[test]
        fn prop_bilinear_short_src_returns_error(
            src_w in 2usize..=16usize,
            src_h in 2usize..=16usize,
            dst_w in 1usize..=16usize,
            dst_h in 1usize..=16usize,
        ) {
            // Provide a source buffer that is intentionally 1 byte shorter than needed.
            let too_short = vec![128u8; src_w * src_h - 1];
            let mut dst = vec![0u8; dst_w * dst_h];
            let result = resize_bilinear(&too_short, src_w, src_h, &mut dst, dst_w, dst_h);
            prop_assert!(result.is_err(), "short src buffer should return error");
        }
    }

    // ── SAD properties ─────────────────────────────────────────────────────────

    proptest! {
        /// SAD(block, block) == 0 for any identical pair of 8×8 blocks.
        #[test]
        fn prop_sad_8x8_identical_is_zero(
            fill in 0u8..=255u8,
        ) {
            let block = vec![fill; 64];
            prop_assert_eq!(block_sad_8x8(&block, &block, 8), 0,
                "SAD of identical blocks must be 0");
        }

        /// SAD is symmetric: SAD(A, B) == SAD(B, A).
        #[test]
        fn prop_sad_8x8_symmetric(
            a in prop::collection::vec(0u8..=255u8, 64usize..=64usize),
            b in prop::collection::vec(0u8..=255u8, 64usize..=64usize),
        ) {
            let ab = block_sad_8x8(&a, &b, 8);
            let ba = block_sad_8x8(&b, &a, 8);
            prop_assert_eq!(ab, ba, "SAD(A,B) must equal SAD(B,A)");
        }

        /// SAD result is always non-negative (guaranteed by u32 return type, but
        /// also check it never returns u32::MAX for well-sized inputs).
        #[test]
        fn prop_sad_8x8_valid_range(
            a in prop::collection::vec(0u8..=255u8, 64usize..=64usize),
            b in prop::collection::vec(0u8..=255u8, 64usize..=64usize),
        ) {
            let result = block_sad_8x8(&a, &b, 8);
            // u32::MAX is a sentinel for invalid input; well-sized buffers must
            // not return it.  Maximum valid SAD for 8×8 is 8*8*255 = 16320.
            prop_assert!(result <= 16_320, "SAD out of valid range: {}", result);
        }

        /// SAD of identical 4×4 blocks is always 0.
        #[test]
        fn prop_sad_4x4_identical_is_zero(
            fill in 0u8..=255u8,
        ) {
            let block = vec![fill; 16];
            prop_assert_eq!(block_sad_4x4(&block, &block, 4), 0,
                "4x4 identical SAD must be 0");
        }

        /// SAD is symmetric for 4×4 blocks.
        #[test]
        fn prop_sad_4x4_symmetric(
            a in prop::collection::vec(0u8..=255u8, 16usize..=16usize),
            b in prop::collection::vec(0u8..=255u8, 16usize..=16usize),
        ) {
            let ab = block_sad_4x4(&a, &b, 4);
            let ba = block_sad_4x4(&b, &a, 4);
            prop_assert_eq!(ab, ba, "4x4 SAD(A,B) must equal SAD(B,A)");
        }

        /// For a uniform source block and a uniform reference block with a constant
        /// offset `d`, SAD(src, ref) == N × d where N is the number of pixels.
        #[test]
        fn prop_sad_uniform_block_offset(
            base in 0u8..=200u8,
            offset in 1u8..=55u8,
        ) {
            // src = [base; 64], ref = [base + offset; 64]
            let src = vec![base; 64];
            let reference = vec![base + offset; 64];
            let expected = 64u32 * u32::from(offset);
            let actual = block_sad_8x8(&src, &reference, 8);
            prop_assert_eq!(actual, expected, "uniform-offset SAD mismatch");
        }

        /// For asymmetric 8×4 blocks, SAD of identical blocks is 0.
        #[test]
        fn prop_sad_8x4_identical_is_zero(
            fill in 0u8..=255u8,
        ) {
            let block = vec![fill; 32];
            prop_assert_eq!(block_sad_8x4(&block, 8, &block, 8), 0,
                "8x4 identical SAD must be 0");
        }

        /// For asymmetric 4×8 blocks, SAD of identical blocks is 0.
        #[test]
        fn prop_sad_4x8_identical_is_zero(
            fill in 0u8..=255u8,
        ) {
            let block = vec![fill; 32];
            prop_assert_eq!(block_sad_4x8(&block, 4, &block, 4), 0,
                "4x8 identical SAD must be 0");
        }
    }

    // ── SATD properties ────────────────────────────────────────────────────────

    proptest! {
        /// SATD(block, block) == 0 for identical 8×8 blocks.
        #[test]
        fn prop_satd_8x8_identical_is_zero(
            fill in 0u8..=255u8,
        ) {
            let block = vec![fill; 64];
            let result = satd_8x8(&block, &block);
            prop_assert_eq!(
                result,
                Ok(0),
                "SATD of identical 8x8 blocks must be 0"
            );
        }

        /// SATD(block, block) == 0 for identical 4×4 blocks.
        #[test]
        fn prop_satd_4x4_identical_is_zero(
            fill in 0u8..=255u8,
        ) {
            let block = vec![fill; 16];
            let result = satd_4x4(&block, &block);
            prop_assert_eq!(
                result,
                Ok(0),
                "SATD of identical 4x4 blocks must be 0"
            );
        }

        /// SATD is symmetric: SATD(A, B) == SATD(B, A).
        #[test]
        fn prop_satd_8x8_symmetric(
            a in prop::collection::vec(0u8..=255u8, 64usize..=64usize),
            b in prop::collection::vec(0u8..=255u8, 64usize..=64usize),
        ) {
            match (satd_8x8(&a, &b), satd_8x8(&b, &a)) {
                (Ok(ab), Ok(ba)) => {
                    prop_assert_eq!(ab, ba, "SATD must be symmetric");
                }
                (Err(_), Err(_)) => {} // Both fail consistently — acceptable.
                (ab, ba) => {
                    prop_assert!(false,
                        "SATD symmetry inconsistency: {:?} vs {:?}", ab, ba);
                }
            }
        }

        /// SATD block size `pixels()` method always returns side² pixels.
        #[test]
        fn prop_satd_block_size_pixels(
            choice in 0usize..=3usize,
        ) {
            let size = match choice {
                0 => SatdBlockSize::Block4x4,
                1 => SatdBlockSize::Block8x8,
                2 => SatdBlockSize::Block16x16,
                _ => SatdBlockSize::Block32x32,
            };
            let side = size.side();
            prop_assert_eq!(size.pixels(), side * side,
                "pixels() must equal side^2");
        }

        /// SATD for well-sized buffers never returns an implausibly large value.
        ///
        /// Any `Ok` result must be within the mathematical upper bound
        /// (a conservative over-estimate for 8×8 Hadamard).
        #[test]
        fn prop_satd_8x8_valid_range(
            a in prop::collection::vec(0u8..=255u8, 64usize..=64usize),
            b in prop::collection::vec(0u8..=255u8, 64usize..=64usize),
        ) {
            if let Ok(result) = satd_8x8(&a, &b) {
                // Hadamard of a max-contrast 8×8 block: conservative ceiling.
                prop_assert!(result <= 1_040_400,
                    "SATD 8x8 result implausibly large: {}", result);
            }
        }
    }
}
