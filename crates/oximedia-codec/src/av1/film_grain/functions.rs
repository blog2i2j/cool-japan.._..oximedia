//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

/// Maximum number of AR (auto-regressive) coefficients for luma.
pub const MAX_AR_COEFFS_LUMA: usize = 24;
/// Maximum number of AR coefficients for chroma.
pub const MAX_AR_COEFFS_CHROMA: usize = 25;
/// Maximum AR lag.
pub const MAX_AR_LAG: usize = 3;
/// Luma grain template size (large).
pub const GRAIN_TEMPLATE_SIZE_LARGE: usize = 128;
/// Luma grain template size (small).
pub const GRAIN_TEMPLATE_SIZE_SMALL: usize = 64;
/// Grain block size for processing.
pub const GRAIN_BLOCK_SIZE: usize = 32;
/// Maximum number of luma scaling points.
pub const MAX_LUMA_SCALING_POINTS: usize = 14;
/// Maximum number of chroma scaling points.
pub const MAX_CHROMA_SCALING_POINTS: usize = 10;
/// Grain seed XOR constant (per AV1 spec).
pub(super) const GRAIN_SEED_XOR: u16 = 0xB524;
/// Gaussian sequence LUT size.
pub(super) const GAUSSIAN_SEQUENCE_SIZE: usize = 2048;
/// Scaling LUT size (256 for 8-bit).
const SCALING_LUT_SIZE: usize = 256;
#[cfg(test)]
mod tests {
    use super::super::types::{
        FilmGrainParams, FilmGrainSynthesizer, GaussianSequence, GrainLcg, ScalingLut, ScalingPoint,
    };
    use super::*;
    #[test]
    fn test_scaling_point() {
        let point = ScalingPoint::new(128, 64);
        assert_eq!(point.value, 128);
        assert_eq!(point.scaling, 64);
    }
    #[test]
    fn test_film_grain_params_default() {
        let params = FilmGrainParams::default();
        assert!(!params.apply_grain);
        assert!(!params.is_enabled());
        assert_eq!(params.num_y_points, 0);
    }
    #[test]
    fn test_film_grain_params_enabled() {
        let mut params = FilmGrainParams::new();
        params.apply_grain = true;
        params.film_grain_params_present = true;
        params.add_y_point(0, 32);
        params.add_y_point(255, 64);
        assert!(params.is_enabled());
        assert_eq!(params.num_y_points, 2);
    }
    #[test]
    fn test_film_grain_params_scaling_values() {
        let mut params = FilmGrainParams::new();
        params.grain_scaling_minus_8 = 2;
        params.ar_coeff_shift_minus_6 = 3;
        assert_eq!(params.grain_scaling(), 10);
        assert_eq!(params.ar_coeff_shift(), 9);
    }
    #[test]
    fn test_film_grain_params_ar_coeffs() {
        let mut params = FilmGrainParams::new();
        params.ar_coeff_lag = 0;
        assert_eq!(params.num_ar_coeffs_y(), 0);
        assert_eq!(params.num_ar_coeffs_uv(), 0);
        params.ar_coeff_lag = 1;
        assert_eq!(params.num_ar_coeffs_y(), 4);
        assert_eq!(params.num_ar_coeffs_uv(), 5);
        params.ar_coeff_lag = 2;
        assert_eq!(params.num_ar_coeffs_y(), 12);
        assert_eq!(params.num_ar_coeffs_uv(), 13);
        params.ar_coeff_lag = 3;
        assert_eq!(params.num_ar_coeffs_y(), 24);
        assert_eq!(params.num_ar_coeffs_uv(), 25);
    }
    #[test]
    fn test_film_grain_params_add_points() {
        let mut params = FilmGrainParams::new();
        params.add_y_point(0, 10);
        params.add_y_point(128, 20);
        params.add_y_point(255, 30);
        assert_eq!(params.num_y_points, 3);
        assert_eq!(params.y_points[0], ScalingPoint::new(0, 10));
        assert_eq!(params.y_points[1], ScalingPoint::new(128, 20));
        assert_eq!(params.y_points[2], ScalingPoint::new(255, 30));
    }
    #[test]
    fn test_film_grain_params_validate() {
        let mut params = FilmGrainParams::new();
        params.add_y_point(0, 32);
        params.add_y_point(128, 48);
        params.add_y_point(255, 64);
        assert!(params.validate());
        let mut params2 = FilmGrainParams::new();
        params2.add_y_point(128, 32);
        params2.add_y_point(0, 48);
        assert!(!params2.validate());
    }
    #[test]
    fn test_grain_lcg() {
        let mut lcg = GrainLcg::new(12345);
        let val1 = lcg.next();
        let val2 = lcg.next();
        assert_ne!(val1, val2);
        let mut lcg2 = GrainLcg::new(12345);
        assert_eq!(lcg2.next(), val1);
        assert_eq!(lcg2.next(), val2);
    }
    #[test]
    fn test_grain_lcg_value_range() {
        let mut lcg = GrainLcg::new(12345);
        for _ in 0..1000 {
            let val = lcg.next_grain_value();
            assert!(val >= -2048 && val < 2048);
        }
    }
    #[test]
    fn test_gaussian_sequence() {
        let seq = GaussianSequence::generate(12345);
        assert_eq!(seq.values.len(), GAUSSIAN_SEQUENCE_SIZE);
        let val = seq.get(GAUSSIAN_SEQUENCE_SIZE + 10);
        assert_eq!(val, seq.get(10));
    }
    #[test]
    fn test_scaling_lut() {
        let points = [
            ScalingPoint::new(0, 32),
            ScalingPoint::new(128, 64),
            ScalingPoint::new(255, 96),
        ];
        let lut = ScalingLut::from_points(&points, 3, 8);
        assert_eq!(lut.values.len(), 256);
        assert_eq!(lut.get(0, 8), 32);
        assert_eq!(lut.get(255, 8), 96);
        let mid = lut.get(128, 8);
        assert!(mid >= 64 && mid <= 64);
    }
    #[test]
    fn test_film_grain_synthesizer_creation() {
        let synth = FilmGrainSynthesizer::new(8);
        assert_eq!(synth.bit_depth, 8);
        assert!(!synth.params().is_enabled());
    }
    #[test]
    fn test_film_grain_synthesizer_set_params() {
        let mut synth = FilmGrainSynthesizer::new(8);
        let mut params = FilmGrainParams::new();
        params.apply_grain = true;
        params.film_grain_params_present = true;
        params.grain_seed = 12345;
        params.add_y_point(0, 32);
        params.add_y_point(255, 64);
        synth.set_params(params);
        assert!(synth.params().is_enabled());
        assert!(synth.luma_template.is_some());
        assert!(synth.luma_scaling.is_some());
    }
    #[test]
    fn test_constants() {
        assert_eq!(MAX_AR_COEFFS_LUMA, 24);
        assert_eq!(MAX_AR_COEFFS_CHROMA, 25);
        assert_eq!(MAX_AR_LAG, 3);
        assert_eq!(GRAIN_BLOCK_SIZE, 32);
        assert_eq!(MAX_LUMA_SCALING_POINTS, 14);
        assert_eq!(MAX_CHROMA_SCALING_POINTS, 10);
    }
    #[test]
    fn test_overlap_weight_values() {
        let (w0_curr, w0_other) = FilmGrainSynthesizer::overlap_weight(0);
        assert_eq!(w0_curr, 27);
        assert_eq!(w0_other, 5);
        assert_eq!(w0_curr + w0_other, 32);
        let (w1_curr, w1_other) = FilmGrainSynthesizer::overlap_weight(1);
        assert_eq!(w1_curr, 17);
        assert_eq!(w1_other, 15);
        assert_eq!(w1_curr + w1_other, 32);
        let (w2_curr, w2_other) = FilmGrainSynthesizer::overlap_weight(2);
        assert_eq!(w2_curr, 32);
        assert_eq!(w2_other, 0);
    }
    #[test]
    fn test_overlap_disabled_uses_regular_grain() {
        use crate::frame::{Plane, VideoFrame};
        use oximedia_core::PixelFormat;
        let mut synth = FilmGrainSynthesizer::new(8);
        let mut params = FilmGrainParams::new();
        params.apply_grain = true;
        params.film_grain_params_present = true;
        params.grain_seed = 42;
        params.overlap_flag = false;
        params.add_y_point(0, 32);
        params.add_y_point(255, 64);
        synth.set_params(params);
        let y = Plane::with_dimensions(vec![128u8; 64 * 64], 64, 64, 64);
        let u = Plane::with_dimensions(vec![128u8; 32 * 32], 32, 32, 32);
        let v = Plane::with_dimensions(vec![128u8; 32 * 32], 32, 32, 32);
        let mut frame = {
            let mut f = VideoFrame::new(PixelFormat::Yuv420p, y.width, y.height);
            f.planes = vec![y, u, v];
            f
        };
        let result = synth.apply_grain_with_overlap(&mut frame);
        assert!(result.is_ok());
    }
    #[test]
    fn test_overlap_enabled_produces_output() {
        use crate::frame::{Plane, VideoFrame};
        use oximedia_core::PixelFormat;
        let mut synth = FilmGrainSynthesizer::new(8);
        let mut params = FilmGrainParams::new();
        params.apply_grain = true;
        params.film_grain_params_present = true;
        params.grain_seed = 42;
        params.overlap_flag = true;
        params.add_y_point(0, 48);
        params.add_y_point(255, 80);
        synth.set_params(params);
        let y = Plane::with_dimensions(vec![128u8; 96 * 96], 96, 96, 96);
        let u = Plane::with_dimensions(vec![128u8; 48 * 48], 48, 48, 48);
        let v = Plane::with_dimensions(vec![128u8; 48 * 48], 48, 48, 48);
        let mut frame = {
            let mut f = VideoFrame::new(PixelFormat::Yuv420p, y.width, y.height);
            f.planes = vec![y, u, v];
            f
        };
        let result = synth.apply_grain_with_overlap(&mut frame);
        assert!(result.is_ok());
        let data = frame.plane(0).data();
        let changed = data.iter().filter(|&&p| p != 128).count();
        assert!(changed > 0, "Overlap grain should modify some pixels");
    }
    #[test]
    fn test_overlap_vs_non_overlap_differ() {
        use crate::frame::{Plane, VideoFrame};
        use oximedia_core::PixelFormat;
        fn make_synth(overlap: bool) -> FilmGrainSynthesizer {
            let mut synth = FilmGrainSynthesizer::new(8);
            let mut params = FilmGrainParams::new();
            params.apply_grain = true;
            params.film_grain_params_present = true;
            params.grain_seed = 999;
            params.overlap_flag = overlap;
            params.add_y_point(0, 40);
            params.add_y_point(255, 80);
            synth.set_params(params);
            synth
        }
        fn make_frame() -> VideoFrame {
            let y = Plane::with_dimensions(vec![128u8; 96 * 96], 96, 96, 96);
            let u = Plane::with_dimensions(vec![128u8; 48 * 48], 48, 48, 48);
            let v = Plane::with_dimensions(vec![128u8; 48 * 48], 48, 48, 48);
            {
                let mut f = VideoFrame::new(PixelFormat::Yuv420p, y.width, y.height);
                f.planes = vec![y, u, v];
                f
            }
        }
        let synth_no_overlap = make_synth(false);
        let mut frame_no = make_frame();
        synth_no_overlap
            .apply_grain(&mut frame_no)
            .expect("no overlap grain");
        let synth_overlap = make_synth(true);
        let mut frame_ov = make_frame();
        synth_overlap
            .apply_grain_with_overlap(&mut frame_ov)
            .expect("overlap grain");
        let data_no = frame_no.plane(0).data();
        let data_ov = frame_ov.plane(0).data();
        let diff_count = data_no
            .iter()
            .zip(data_ov.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            diff_count > 0,
            "Overlap and non-overlap grain should differ at boundaries"
        );
    }
    #[test]
    fn test_per_block_grain_table_empty() {
        use super::super::super::film_grain_table::{BlockGrainOverride, PerBlockGrainTable};
        let base = FilmGrainParams::new();
        let table = PerBlockGrainTable::new();
        assert!(table.is_empty());
        assert_eq!(table.resolve(&base, 0, 0).grain_seed, base.grain_seed);
    }
    #[test]
    fn test_block_grain_override_clamps() {
        use super::super::super::film_grain_table::BlockGrainOverride;
        let mut base = FilmGrainParams::new();
        base.grain_scaling_minus_8 = 3;
        let mut ov = BlockGrainOverride::new(0, 0);
        ov.scaling_delta = 10;
        assert_eq!(ov.apply(&base).grain_scaling_minus_8, 3);
        ov.scaling_delta = -10;
        base.grain_scaling_minus_8 = 0;
        assert_eq!(ov.apply(&base).grain_scaling_minus_8, 0);
    }
    #[test]
    fn test_per_block_table_resolve() {
        use super::super::super::film_grain_table::{BlockGrainOverride, PerBlockGrainTable};
        let mut base = FilmGrainParams::new();
        base.grain_seed = 100;
        let mut table = PerBlockGrainTable::new();
        let mut o0 = BlockGrainOverride::new(0, 0);
        o0.seed_xor = 0xAA;
        table.set(o0);
        let mut o1 = BlockGrainOverride::new(1, 0);
        o1.seed_xor = 0x55;
        table.set(o1);
        assert_eq!(table.len(), 2);
        assert_eq!(table.resolve(&base, 0, 0).grain_seed, 100 ^ 0xAA);
        assert_eq!(table.resolve(&base, 1, 0).grain_seed, 100 ^ 0x55);
        assert_eq!(table.resolve(&base, 9, 9).grain_seed, 100);
    }
    #[test]
    fn test_apply_grain_per_block_ok() {
        use super::super::super::film_grain_table::{BlockGrainOverride, PerBlockGrainTable};
        use crate::frame::{Plane, VideoFrame};
        use oximedia_core::PixelFormat;
        let mut synth = FilmGrainSynthesizer::new(8);
        let mut params = FilmGrainParams::new();
        params.apply_grain = true;
        params.film_grain_params_present = true;
        params.grain_seed = 7;
        params.add_y_point(0, 32);
        params.add_y_point(255, 64);
        synth.set_params(params);
        let y = Plane::with_dimensions(vec![128u8; 64 * 64], 64, 64, 64);
        let u = Plane::with_dimensions(vec![128u8; 32 * 32], 32, 32, 32);
        let v = Plane::with_dimensions(vec![128u8; 32 * 32], 32, 32, 32);
        let mut frame = {
            let mut f = VideoFrame::new(PixelFormat::Yuv420p, y.width, y.height);
            f.planes = vec![y, u, v];
            f
        };
        let mut table = PerBlockGrainTable::new();
        let mut ov = BlockGrainOverride::new(0, 0);
        ov.seed_xor = 0xBEEF;
        table.set(ov);
        assert!(synth.apply_grain_per_block(&mut frame, &table).is_ok());
    }
    #[test]
    fn test_apply_grain_per_block_u16_10bit_range() {
        use super::super::super::film_grain_table::PerBlockGrainTable;
        let mut synth = FilmGrainSynthesizer::new(10);
        let mut params = FilmGrainParams::new();
        params.apply_grain = true;
        params.film_grain_params_present = true;
        params.grain_seed = 42;
        params.add_y_point(0, 32);
        params.add_y_point(255, 64);
        synth.set_params(params);
        let mut luma = vec![512u16; 64 * 64];
        assert!(synth
            .apply_grain_per_block_u16(&mut luma, 64, 64, 64, &PerBlockGrainTable::new())
            .is_ok());
        for &px in &luma {
            assert!(px <= 1023, "10-bit value out of range: {px}");
        }
    }
    #[test]
    fn test_apply_grain_per_block_u16_noop_when_disabled() {
        use super::super::super::film_grain_table::PerBlockGrainTable;
        let synth = FilmGrainSynthesizer::new(10);
        let mut luma = vec![500u16; 32 * 32];
        let orig = luma.clone();
        assert!(synth
            .apply_grain_per_block_u16(&mut luma, 32, 32, 32, &PerBlockGrainTable::new())
            .is_ok());
        assert_eq!(luma, orig);
    }
    #[test]
    fn test_apply_grain_per_block_u16_12bit_range() {
        use super::super::super::film_grain_table::PerBlockGrainTable;
        let mut synth = FilmGrainSynthesizer::new(12);
        let mut params = FilmGrainParams::new();
        params.apply_grain = true;
        params.film_grain_params_present = true;
        params.grain_seed = 99;
        params.add_y_point(0, 20);
        params.add_y_point(255, 50);
        synth.set_params(params);
        let mut luma = vec![2048u16; 32 * 32];
        assert!(synth
            .apply_grain_per_block_u16(&mut luma, 32, 32, 32, &PerBlockGrainTable::new())
            .is_ok());
        for &px in &luma {
            assert!(px <= 4095, "12-bit value out of range: {px}");
        }
    }
    #[test]
    fn test_apply_grain_per_block_u16_modifies_pixels() {
        use super::super::super::film_grain_table::PerBlockGrainTable;
        let mut synth = FilmGrainSynthesizer::new(10);
        let mut params = FilmGrainParams::new();
        params.apply_grain = true;
        params.film_grain_params_present = true;
        params.grain_seed = 12345;
        params.grain_scaling_minus_8 = 3;
        params.add_y_point(0, 64);
        params.add_y_point(255, 128);
        synth.set_params(params);
        let mut luma = vec![512u16; 64 * 64];
        let orig = luma.clone();
        assert!(synth
            .apply_grain_per_block_u16(&mut luma, 64, 64, 64, &PerBlockGrainTable::new())
            .is_ok());
        assert!(luma.iter().zip(orig.iter()).any(|(&a, &b)| a != b));
    }
}
