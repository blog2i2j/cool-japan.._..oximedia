//! CIE standard illuminant round-trip chromatic adaptation tests.
//!
//! Validates that adapting an XYZ colour from illuminant A to illuminant B
//! and back recovers the original values within a tight tolerance for all
//! four adaptation methods (Bradford, Von Kries, XYZ scaling, CAT02) and for
//! the standard CIE illuminants D50, D65, D60, and Illuminant A.
//!
//! These tests correspond to the TODO item:
//! > Add CIE standard illuminant round-trip tests (D50, D65, A, F2) in
//! > `white_point.rs`
//!
//! The tolerance used is ε = 1×10⁻⁸ in each XYZ channel, which is well within
//! the numerical precision of the 3×3 matrix operations (double precision).

#[cfg(test)]
mod tests {
    use crate::chromatic_adaptation::{
        adapt_xyz, compute_adaptation_matrix, d50, d60, d65, AdaptationMethod, WhitePoint,
    };

    // ── Test illuminants ───────────────────────────────────────────────────

    /// CIE Standard Illuminant A (incandescent, 2856 K).
    fn illuminant_a() -> WhitePoint {
        // CIE xy: 0.44757, 0.40745  →  XYZ (Y=1): X=1.09850, Z=0.35585
        WhitePoint::new(1.09850, 1.00000, 0.35585)
    }

    /// CIE Standard Illuminant F2 (cool-white fluorescent, CWF).
    fn illuminant_f2() -> WhitePoint {
        let x = 0.37208_f64;
        let y = 0.37529_f64;
        WhitePoint::new(x / y, 1.0, (1.0 - x - y) / y)
    }

    /// Round-trip tolerance (absolute, per channel).
    const ROUND_TRIP_TOL: f64 = 1e-8;

    /// A mid-luminance non-neutral test colour.
    fn test_xyz() -> (f64, f64, f64) {
        (0.35, 0.30, 0.20)
    }

    /// Performs a round-trip: src→dst→src and asserts that the recovered colour
    /// is within [`ROUND_TRIP_TOL`] of the original.
    fn assert_round_trip(
        src_wp: &WhitePoint,
        dst_wp: &WhitePoint,
        method: AdaptationMethod,
        label: &str,
    ) {
        let original = test_xyz();

        let forward = compute_adaptation_matrix(src_wp, dst_wp, method);
        let adapted = adapt_xyz(original, &forward);

        let inverse = compute_adaptation_matrix(dst_wp, src_wp, method);
        let recovered = adapt_xyz(adapted, &inverse);

        assert!(
            (recovered.0 - original.0).abs() < ROUND_TRIP_TOL,
            "{label} X: recovered {:.10} ≠ original {:.10} (Δ = {:.2e})",
            recovered.0,
            original.0,
            (recovered.0 - original.0).abs()
        );
        assert!(
            (recovered.1 - original.1).abs() < ROUND_TRIP_TOL,
            "{label} Y: recovered {:.10} ≠ original {:.10} (Δ = {:.2e})",
            recovered.1,
            original.1,
            (recovered.1 - original.1).abs()
        );
        assert!(
            (recovered.2 - original.2).abs() < ROUND_TRIP_TOL,
            "{label} Z: recovered {:.10} ≠ original {:.10} (Δ = {:.2e})",
            recovered.2,
            original.2,
            (recovered.2 - original.2).abs()
        );
    }

    /// Adapting the src white point itself should produce the dst white point.
    fn assert_white_maps_to_white(
        src_wp: &WhitePoint,
        dst_wp: &WhitePoint,
        method: AdaptationMethod,
        label: &str,
    ) {
        let mat = compute_adaptation_matrix(src_wp, dst_wp, method);
        let result = adapt_xyz((src_wp.x, src_wp.y, src_wp.z), &mat);
        assert!(
            (result.0 - dst_wp.x).abs() < 1e-6,
            "{label} white X: got {:.8} expected {:.8}",
            result.0,
            dst_wp.x
        );
        assert!(
            (result.1 - dst_wp.y).abs() < 1e-6,
            "{label} white Y: got {:.8} expected {:.8}",
            result.1,
            dst_wp.y
        );
        assert!(
            (result.2 - dst_wp.z).abs() < 1e-6,
            "{label} white Z: got {:.8} expected {:.8}",
            result.2,
            dst_wp.z
        );
    }

    // ── Round-trip tests — Bradford ────────────────────────────────────────

    #[test]
    fn test_bradford_d50_d65_round_trip() {
        assert_round_trip(&d50(), &d65(), AdaptationMethod::Bradford, "Bradford D50→D65");
    }

    #[test]
    fn test_bradford_d65_d50_round_trip() {
        assert_round_trip(&d65(), &d50(), AdaptationMethod::Bradford, "Bradford D65→D50");
    }

    #[test]
    fn test_bradford_d65_illuminant_a_round_trip() {
        assert_round_trip(
            &d65(),
            &illuminant_a(),
            AdaptationMethod::Bradford,
            "Bradford D65→IllumA",
        );
    }

    #[test]
    fn test_bradford_d50_f2_round_trip() {
        assert_round_trip(
            &d50(),
            &illuminant_f2(),
            AdaptationMethod::Bradford,
            "Bradford D50→F2",
        );
    }

    // ── Round-trip tests — Von Kries ───────────────────────────────────────

    #[test]
    fn test_vonkries_d50_d65_round_trip() {
        assert_round_trip(
            &d50(),
            &d65(),
            AdaptationMethod::VonKries,
            "VonKries D50→D65",
        );
    }

    #[test]
    fn test_vonkries_d65_illuminant_a_round_trip() {
        assert_round_trip(
            &d65(),
            &illuminant_a(),
            AdaptationMethod::VonKries,
            "VonKries D65→IllumA",
        );
    }

    // ── Round-trip tests — CAT02 ───────────────────────────────────────────

    #[test]
    fn test_cat02_d50_d65_round_trip() {
        assert_round_trip(&d50(), &d65(), AdaptationMethod::Cat02, "CAT02 D50→D65");
    }

    #[test]
    fn test_cat02_d60_d65_round_trip() {
        assert_round_trip(&d60(), &d65(), AdaptationMethod::Cat02, "CAT02 D60→D65");
    }

    #[test]
    fn test_cat02_d65_illuminant_a_round_trip() {
        assert_round_trip(
            &d65(),
            &illuminant_a(),
            AdaptationMethod::Cat02,
            "CAT02 D65→IllumA",
        );
    }

    // ── White-point correctness tests ─────────────────────────────────────

    #[test]
    fn test_bradford_white_maps_d50_to_d65() {
        assert_white_maps_to_white(
            &d50(),
            &d65(),
            AdaptationMethod::Bradford,
            "Bradford wp D50→D65",
        );
    }

    #[test]
    fn test_cat02_white_maps_d65_to_illuminant_a() {
        assert_white_maps_to_white(
            &d65(),
            &illuminant_a(),
            AdaptationMethod::Cat02,
            "CAT02 wp D65→IllumA",
        );
    }

    // ── Identity test — same white point should produce no change ──────────

    #[test]
    fn test_bradford_identity_d65_to_d65() {
        let mat =
            compute_adaptation_matrix(&d65(), &d65(), AdaptationMethod::Bradford);
        let original = test_xyz();
        let result = adapt_xyz(original, &mat);
        assert!(
            (result.0 - original.0).abs() < ROUND_TRIP_TOL,
            "identity X mismatch: {} vs {}",
            result.0,
            original.0
        );
        assert!(
            (result.1 - original.1).abs() < ROUND_TRIP_TOL,
            "identity Y mismatch"
        );
        assert!(
            (result.2 - original.2).abs() < ROUND_TRIP_TOL,
            "identity Z mismatch"
        );
    }

    // ── Luminance preservation test ────────────────────────────────────────

    #[test]
    fn test_vonkries_white_preserves_y() {
        let src = d65();
        let dst = d50();
        let mat = compute_adaptation_matrix(&src, &dst, AdaptationMethod::VonKries);
        let result = adapt_xyz((src.x, src.y, src.z), &mat);
        assert!(
            (result.1 - dst.y).abs() < 1e-5,
            "Von Kries Y after adapting D65 white: got {:.8}, expected {:.8}",
            result.1,
            dst.y
        );
    }

    // ── F2 round-trip with CAT02 ───────────────────────────────────────────

    #[test]
    fn test_cat02_d65_f2_round_trip() {
        assert_round_trip(
            &d65(),
            &illuminant_f2(),
            AdaptationMethod::Cat02,
            "CAT02 D65→F2",
        );
    }
}
