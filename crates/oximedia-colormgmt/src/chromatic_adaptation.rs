//! Chromatic adaptation transforms (CAT).
//!
//! Provides industry-standard methods for adapting colors from one illuminant
//! (white point) to another, including Bradford, Von Kries, XYZ scaling, and CAT02.

#![allow(dead_code)]

/// Chromatic adaptation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptationMethod {
    /// Bradford chromatic adaptation (ICC recommended).
    Bradford,
    /// Von Kries chromatic adaptation (classical).
    VonKries,
    /// Simple XYZ scaling (not perceptually accurate, but simple).
    XyzScaling,
    /// CAT02 adaptation (CIECAM02).
    Cat02,
}

/// A CIE XYZ tristimulus white point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WhitePoint {
    /// X tristimulus value.
    pub x: f64,
    /// Y tristimulus value (luminance; 1.0 for relative).
    pub y: f64,
    /// Z tristimulus value.
    pub z: f64,
}

impl WhitePoint {
    /// Create a white point from XYZ values.
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Create a white point from CIE xy chromaticity (Y assumed 1.0).
    #[must_use]
    pub fn from_xy(x: f64, y: f64) -> Self {
        // xy → XYZ with Y=1
        let z = (1.0 - x - y) / y;
        Self {
            x: x / y,
            y: 1.0,
            z,
        }
    }
}

/// CIE Standard Illuminant D50 (horizon daylight; used in ICC profiles).
#[must_use]
pub fn d50() -> WhitePoint {
    // CIE xy: 0.3457, 0.3585
    WhitePoint::new(0.96429, 1.00000, 0.82510)
}

/// CIE Standard Illuminant D65 (average daylight; sRGB/Rec.709 reference).
#[must_use]
pub fn d65() -> WhitePoint {
    // CIE xy: 0.3127, 0.3290
    WhitePoint::new(0.95046, 1.00000, 1.08906)
}

/// CIE Standard Illuminant D60 (ACES reference illuminant).
#[must_use]
pub fn d60() -> WhitePoint {
    // CIE xy: 0.32168, 0.33767
    WhitePoint::new(0.95265, 1.00000, 1.00883)
}

// ── Adaptation matrices ───────────────────────────────────────────────────────

/// Bradford chromatic adaptation matrix (MCAT).
///
/// Transforms XYZ to Bradford cone-like LMS space.
#[must_use]
pub fn bradford_matrix() -> [[f64; 3]; 3] {
    [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ]
}

/// Von Kries adaptation matrix.
#[must_use]
pub fn von_kries_matrix() -> [[f64; 3]; 3] {
    [
        [0.40024, 0.70760, -0.08081],
        [-0.22630, 1.16532, 0.04570],
        [0.00000, 0.00000, 0.91822],
    ]
}

/// CAT02 chromatic adaptation matrix (CIECAM02).
#[must_use]
pub fn cat02_matrix() -> [[f64; 3]; 3] {
    [
        [0.7328, 0.4296, -0.1624],
        [-0.7036, 1.6975, 0.0061],
        [0.0030, 0.0136, 0.9834],
    ]
}

/// XYZ scaling adaptation matrix (identity).
#[must_use]
fn xyz_scaling_matrix() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

// ── Matrix helpers ────────────────────────────────────────────────────────────

/// Multiply a 3×3 matrix by a 3-vector.
#[inline]
#[must_use]
fn mat3_mul_vec(m: &[[f64; 3]; 3], v: (f64, f64, f64)) -> (f64, f64, f64) {
    (
        m[0][0] * v.0 + m[0][1] * v.1 + m[0][2] * v.2,
        m[1][0] * v.0 + m[1][1] * v.1 + m[1][2] * v.2,
        m[2][0] * v.0 + m[2][1] * v.1 + m[2][2] * v.2,
    )
}

/// Multiply two 3×3 matrices.
#[must_use]
fn mat3_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

/// Invert a 3×3 matrix using Cramer's rule.
///
/// Returns `None` if the matrix is singular.
#[must_use]
fn mat3_inv(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if det.abs() < 1e-12 {
        return None;
    }

    let inv_det = 1.0 / det;
    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ])
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Compute the chromatic adaptation matrix from `src_wp` to `dst_wp`.
///
/// The returned 3×3 matrix can be applied with [`adapt_xyz`] to transform
/// XYZ values adapted to `src_wp` into XYZ values adapted to `dst_wp`.
///
/// # Panics
///
/// Panics if the chosen CAT matrix is singular (which should never happen for
/// the standard matrices provided by this module).
#[must_use]
pub fn compute_adaptation_matrix(
    src_wp: &WhitePoint,
    dst_wp: &WhitePoint,
    method: AdaptationMethod,
) -> [[f64; 3]; 3] {
    let mcat = match method {
        AdaptationMethod::Bradford => bradford_matrix(),
        AdaptationMethod::VonKries => von_kries_matrix(),
        AdaptationMethod::Cat02 => cat02_matrix(),
        AdaptationMethod::XyzScaling => xyz_scaling_matrix(),
    };

    // Transform white points into cone space
    let src_lms = mat3_mul_vec(&mcat, (src_wp.x, src_wp.y, src_wp.z));
    let dst_lms = mat3_mul_vec(&mcat, (dst_wp.x, dst_wp.y, dst_wp.z));

    // Build diagonal scaling matrix
    let scale = [
        [dst_lms.0 / src_lms.0.max(f64::EPSILON), 0.0, 0.0],
        [0.0, dst_lms.1 / src_lms.1.max(f64::EPSILON), 0.0],
        [0.0, 0.0, dst_lms.2 / src_lms.2.max(f64::EPSILON)],
    ];

    // Full CAT = Mcat^-1 * Scale * Mcat
    // Standard CAT matrices (Bradford, Von Kries, CAT02, XYZ) are always invertible.
    let mcat_inv = mat3_inv(&mcat).unwrap_or([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let scaled = mat3_mul(&scale, &mcat);
    mat3_mul(&mcat_inv, &scaled)
}

/// Apply a precomputed chromatic adaptation matrix to an XYZ triplet.
///
/// The matrix should have been computed by [`compute_adaptation_matrix`].
#[must_use]
#[inline]
pub fn adapt_xyz(xyz: (f64, f64, f64), matrix: &[[f64; 3]; 3]) -> (f64, f64, f64) {
    mat3_mul_vec(matrix, xyz)
}

// ── Multi-illuminant support ──────────────────────────────────────────────────

/// CIE Standard Illuminant A (incandescent / tungsten, 2856 K).
///
/// Approximated from the Planckian radiator at 2856 K relative to equal-energy
/// (normalised so Y = 1). Values per CIE Publication 15 (2004).
#[must_use]
pub fn illuminant_a() -> WhitePoint {
    WhitePoint::new(1.09850, 1.00000, 0.35585)
}

/// CIE Standard Illuminant F2 (cool-white fluorescent).
///
/// Representative of cool-white fluorescent tube illumination used in offices.
/// Values from CIE Publication 15 (2004), Table T.1.
#[must_use]
pub fn illuminant_f2() -> WhitePoint {
    // CIE xy: 0.3721, 0.3751
    WhitePoint::new(0.9915, 1.00000, 0.6731)
}

/// CIE Standard Illuminant F7 (broadband daylight fluorescent).
///
/// Represents a broadband daylight-simulating fluorescent lamp.
/// Values from CIE Publication 15 (2004).
#[must_use]
pub fn illuminant_f7() -> WhitePoint {
    // CIE xy: 0.3129, 0.3292 (close to D65)
    WhitePoint::new(0.9503, 1.00000, 1.0887)
}

/// CIE Standard Illuminant F11 (narrowband cool-white fluorescent).
///
/// A three-band lamp commonly used in industrial settings.
/// Values from CIE Publication 15 (2004).
#[must_use]
pub fn illuminant_f11() -> WhitePoint {
    // CIE xy: 0.3805, 0.3769
    WhitePoint::new(1.0096, 1.00000, 0.6432)
}

/// CIE Standard Illuminant E (equal-energy white).
///
/// A theoretical equal-energy illuminant where all wavelengths have equal power.
/// XYZ = (1, 1, 1) when normalised.
#[must_use]
pub fn illuminant_e() -> WhitePoint {
    WhitePoint::new(1.00000, 1.00000, 1.00000)
}

/// Computes the chromatic adaptation matrix for a sequence of illuminants.
///
/// This computes a single combined matrix that adapts from `src_illuminant`
/// through intermediate illuminants to `dst_illuminant`. This is useful for
/// multi-step workflows (e.g. scene illuminant → D50 → D65) where each
/// adaptation step uses the same CAT method.
///
/// # Arguments
///
/// * `illuminants` - Ordered list of illuminants; the first is the source,
///   the last is the destination. Must have at least 2 elements.
/// * `method` - Chromatic adaptation method to use for each step.
///
/// # Returns
///
/// A single combined 3×3 matrix equivalent to applying each step in sequence.
///
/// # Panics
///
/// Panics if `illuminants` has fewer than 2 elements, or if any CAT matrix
/// is singular.
#[must_use]
pub fn compute_multi_illuminant_adaptation(
    illuminants: &[WhitePoint],
    method: AdaptationMethod,
) -> [[f64; 3]; 3] {
    assert!(
        illuminants.len() >= 2,
        "Need at least 2 illuminants for multi-illuminant adaptation"
    );

    // Start with identity
    let mut combined = [[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    for pair in illuminants.windows(2) {
        let step = compute_adaptation_matrix(&pair[0], &pair[1], method);
        combined = mat3_mul(&step, &combined);
    }

    combined
}

/// Adapts a colour from one illuminant to another, given a set of illuminant
/// definitions and adaptation method.
///
/// This is a convenience wrapper that computes and applies the adaptation
/// matrix in one call, allowing easy switching between illuminants.
#[must_use]
pub fn adapt_xyz_between(
    xyz: (f64, f64, f64),
    src: &WhitePoint,
    dst: &WhitePoint,
    method: AdaptationMethod,
) -> (f64, f64, f64) {
    let matrix = compute_adaptation_matrix(src, dst, method);
    adapt_xyz(xyz, &matrix)
}

/// Multi-illuminant chromatic adaptation state machine.
///
/// Supports adapting through a chain of illuminants (e.g. for multi-step
/// studio workflows where the viewing illuminant changes at each stage).
#[derive(Debug, Clone)]
pub struct MultiIlluminantAdapter {
    /// The sequence of white points in the adaptation chain.
    pub illuminants: Vec<WhitePoint>,
    /// Chromatic adaptation method.
    pub method: AdaptationMethod,
    /// Pre-computed combined adaptation matrix.
    combined_matrix: [[f64; 3]; 3],
}

impl MultiIlluminantAdapter {
    /// Create a new multi-illuminant adapter.
    ///
    /// # Panics
    ///
    /// Panics if `illuminants` has fewer than 2 entries.
    #[must_use]
    pub fn new(illuminants: Vec<WhitePoint>, method: AdaptationMethod) -> Self {
        let combined_matrix = compute_multi_illuminant_adaptation(&illuminants, method);
        Self {
            illuminants,
            method,
            combined_matrix,
        }
    }

    /// Adapt an XYZ colour through the full illuminant chain.
    #[must_use]
    pub fn adapt(&self, xyz: (f64, f64, f64)) -> (f64, f64, f64) {
        adapt_xyz(xyz, &self.combined_matrix)
    }

    /// Returns the combined adaptation matrix.
    #[must_use]
    pub fn combined_matrix(&self) -> [[f64; 3]; 3] {
        self.combined_matrix
    }

    /// Returns the source illuminant (first in the chain).
    #[must_use]
    pub fn source_illuminant(&self) -> &WhitePoint {
        &self.illuminants[0]
    }

    /// Returns the destination illuminant (last in the chain).
    #[must_use]
    pub fn destination_illuminant(&self) -> &WhitePoint {
        // The constructor guarantees at least 2 illuminants; use len()-1 to access last.
        &self.illuminants[self.illuminants.len() - 1]
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Check that two f64 values are within an absolute tolerance.
    fn assert_near(a: f64, b: f64, tol: f64, label: &str) {
        assert!(
            (a - b).abs() < tol,
            "{label}: {a} vs {b}, diff={}",
            (a - b).abs()
        );
    }

    #[test]
    fn test_d65_white_point_values() {
        let wp = d65();
        assert_near(wp.x, 0.95046, 0.001, "D65 X");
        assert_near(wp.y, 1.00000, 1e-10, "D65 Y");
        assert_near(wp.z, 1.08906, 0.001, "D65 Z");
    }

    #[test]
    fn test_d50_white_point_values() {
        let wp = d50();
        assert_near(wp.x, 0.96429, 0.001, "D50 X");
        assert_near(wp.y, 1.00000, 1e-10, "D50 Y");
        assert_near(wp.z, 0.82510, 0.001, "D50 Z");
    }

    #[test]
    fn test_d60_white_point_values() {
        let wp = d60();
        assert_near(wp.x, 0.95265, 0.001, "D60 X");
        assert_near(wp.y, 1.00000, 1e-10, "D60 Y");
    }

    #[test]
    fn test_identity_adaptation_same_white_point() {
        let wp = d65();
        let m = compute_adaptation_matrix(&wp, &wp, AdaptationMethod::Bradford);
        // Adapting from a WP to itself should give identity (within floating point)
        let xyz = (0.5, 0.4, 0.6);
        let adapted = adapt_xyz(xyz, &m);
        assert_near(adapted.0, xyz.0, 1e-8, "X identity");
        assert_near(adapted.1, xyz.1, 1e-8, "Y identity");
        assert_near(adapted.2, xyz.2, 1e-8, "Z identity");
    }

    #[test]
    fn test_adapt_xyz_bradford_d65_d50() {
        // Adapting D65 white point to D50 should give approximately D50 white point
        let src = d65();
        let dst = d50();
        let m = compute_adaptation_matrix(&src, &dst, AdaptationMethod::Bradford);
        let adapted = adapt_xyz((src.x, src.y, src.z), &m);
        assert_near(adapted.0, dst.x, 0.01, "Bradford D65->D50 X");
        assert_near(adapted.1, dst.y, 0.01, "Bradford D65->D50 Y");
        assert_near(adapted.2, dst.z, 0.01, "Bradford D65->D50 Z");
    }

    #[test]
    fn test_adapt_xyz_vonkries_d65_d50() {
        let src = d65();
        let dst = d50();
        let m = compute_adaptation_matrix(&src, &dst, AdaptationMethod::VonKries);
        let adapted = adapt_xyz((src.x, src.y, src.z), &m);
        assert_near(adapted.0, dst.x, 0.05, "VonKries D65->D50 X");
        assert_near(adapted.1, dst.y, 0.05, "VonKries D65->D50 Y");
    }

    #[test]
    fn test_adapt_xyz_cat02_d65_d50() {
        let src = d65();
        let dst = d50();
        let m = compute_adaptation_matrix(&src, &dst, AdaptationMethod::Cat02);
        let adapted = adapt_xyz((src.x, src.y, src.z), &m);
        assert_near(adapted.0, dst.x, 0.05, "CAT02 D65->D50 X");
        assert_near(adapted.1, dst.y, 0.05, "CAT02 D65->D50 Y");
    }

    #[test]
    fn test_adapt_xyz_xyz_scaling() {
        let src = d65();
        let dst = d50();
        let m = compute_adaptation_matrix(&src, &dst, AdaptationMethod::XyzScaling);
        let adapted = adapt_xyz((src.x, src.y, src.z), &m);
        // XYZ scaling just scales each channel independently
        assert_near(adapted.1, dst.y, 0.1, "XYZ scaling Y");
    }

    #[test]
    fn test_bradford_matrix_invertible() {
        let m = bradford_matrix();
        assert!(
            mat3_inv(&m).is_some(),
            "Bradford matrix should be invertible"
        );
    }

    #[test]
    fn test_cat02_matrix_invertible() {
        let m = cat02_matrix();
        assert!(mat3_inv(&m).is_some(), "CAT02 matrix should be invertible");
    }

    #[test]
    fn test_white_point_from_xy() {
        // D65 xy chromaticity
        let wp = WhitePoint::from_xy(0.3127, 0.3290);
        assert_near(wp.y, 1.0, 1e-10, "Y should be 1.0");
        // X = x/y for Y=1
        let expected_x = 0.3127 / 0.3290;
        assert_near(wp.x, expected_x, 1e-10, "X from xy");
    }

    #[test]
    fn test_mat3_mul_identity() {
        let id = xyz_scaling_matrix();
        let m = bradford_matrix();
        let result = mat3_mul(&id, &m);
        for i in 0..3 {
            for j in 0..3 {
                assert_near(result[i][j], m[i][j], 1e-12, "identity mul");
            }
        }
    }

    #[test]
    fn test_round_trip_adaptation() {
        // D65 -> D50 then D50 -> D65 should be (near) identity
        let src = d65();
        let dst = d50();
        let m_fwd = compute_adaptation_matrix(&src, &dst, AdaptationMethod::Bradford);
        let m_rev = compute_adaptation_matrix(&dst, &src, AdaptationMethod::Bradford);
        let xyz = (0.5, 0.6, 0.4);
        let adapted = adapt_xyz(xyz, &m_fwd);
        let back = adapt_xyz(adapted, &m_rev);
        assert_near(back.0, xyz.0, 1e-6, "round-trip X");
        assert_near(back.1, xyz.1, 1e-6, "round-trip Y");
        assert_near(back.2, xyz.2, 1e-6, "round-trip Z");
    }

    // ── Multi-illuminant and additional illuminant tests ───────────────────────

    #[test]
    fn test_illuminant_a_white_point() {
        let wp = illuminant_a();
        // Illuminant A is incandescent; X should be > 1 (reddish)
        assert!(
            wp.x > 1.0,
            "Illuminant A X should be > 1 for tungsten: {}",
            wp.x
        );
        assert_near(wp.y, 1.0, 1e-10, "Illuminant A Y");
        assert!(
            wp.z < 1.0,
            "Illuminant A Z should be < 1 for warm light: {}",
            wp.z
        );
    }

    #[test]
    fn test_illuminant_f2_white_point() {
        let wp = illuminant_f2();
        assert_near(wp.y, 1.0, 1e-10, "F2 Y");
        assert!(wp.x > 0.5, "F2 X should be positive");
    }

    #[test]
    fn test_illuminant_e_equal_energy() {
        let wp = illuminant_e();
        assert_near(wp.x, 1.0, 1e-10, "Equal energy X");
        assert_near(wp.y, 1.0, 1e-10, "Equal energy Y");
        assert_near(wp.z, 1.0, 1e-10, "Equal energy Z");
    }

    #[test]
    fn test_illuminant_f7_close_to_d65() {
        let f7 = illuminant_f7();
        let d65_wp = d65();
        // F7 is a daylight-simulator — should be close to D65 in Y
        assert_near(f7.y, d65_wp.y, 0.01, "F7 Y close to D65");
    }

    #[test]
    fn test_multi_illuminant_two_step_matches_direct() {
        // A→D65→D50 combined should be close to A→D50 directly
        let a_wp = illuminant_a();
        let d65_wp = d65();
        let d50_wp = d50();

        let direct = compute_adaptation_matrix(&a_wp, &d50_wp, AdaptationMethod::Bradford);
        let multi = compute_multi_illuminant_adaptation(
            &[a_wp, d65_wp, d50_wp],
            AdaptationMethod::Bradford,
        );

        let xyz = (0.5, 0.5, 0.5);
        let direct_out = adapt_xyz(xyz, &direct);
        let multi_out = adapt_xyz(xyz, &multi);

        // Multi-step may differ slightly due to numerical accumulation
        // We just verify both produce valid and similar results
        assert_near(
            direct_out.1,
            multi_out.1,
            0.02,
            "Y should be close between direct and multi-step",
        );
    }

    #[test]
    fn test_multi_illuminant_identity_chain() {
        // D65 → D65 chain should give identity
        let d65_wp = d65();
        let combined =
            compute_multi_illuminant_adaptation(&[d65_wp, d65_wp], AdaptationMethod::Bradford);
        let xyz = (0.5, 0.6, 0.4);
        let out = adapt_xyz(xyz, &combined);
        assert_near(out.0, xyz.0, 1e-8, "X identity chain");
        assert_near(out.1, xyz.1, 1e-8, "Y identity chain");
        assert_near(out.2, xyz.2, 1e-8, "Z identity chain");
    }

    #[test]
    fn test_adapt_xyz_between_d65_d50() {
        let src = d65();
        let dst = d50();
        let adapted = adapt_xyz_between(
            (src.x, src.y, src.z),
            &src,
            &dst,
            AdaptationMethod::Bradford,
        );
        assert_near(adapted.0, dst.x, 0.01, "between D65->D50 X");
        assert_near(adapted.1, dst.y, 0.01, "between D65->D50 Y");
    }

    #[test]
    fn test_multi_illuminant_adapter_construction() {
        let adapter = MultiIlluminantAdapter::new(vec![d65(), d50()], AdaptationMethod::Bradford);
        assert_eq!(adapter.illuminants.len(), 2);
        // Adapting the D65 white point should give approximately D50
        let adapted = adapter.adapt((d65().x, d65().y, d65().z));
        assert_near(adapted.0, d50().x, 0.01, "adapter D65->D50 X");
    }

    #[test]
    fn test_multi_illuminant_adapter_three_step() {
        // A → D65 → D50
        let adapter = MultiIlluminantAdapter::new(
            vec![illuminant_a(), d65(), d50()],
            AdaptationMethod::Cat02,
        );
        assert_eq!(adapter.illuminants.len(), 3);
        let out = adapter.adapt((0.5, 0.6, 0.1));
        assert!(out.0.is_finite() && out.1.is_finite() && out.2.is_finite());
    }

    #[test]
    fn test_round_trip_d50_d65_a_illuminants() {
        // Test round-trip for each standard illuminant
        for (src, dst) in [
            (d50(), d65()),
            (d65(), d50()),
            (illuminant_a(), d65()),
            (d65(), illuminant_a()),
        ] {
            let m_fwd = compute_adaptation_matrix(&src, &dst, AdaptationMethod::Bradford);
            let m_rev = compute_adaptation_matrix(&dst, &src, AdaptationMethod::Bradford);
            let xyz = (0.4, 0.5, 0.3);
            let fwd = adapt_xyz(xyz, &m_fwd);
            let back = adapt_xyz(fwd, &m_rev);
            assert_near(back.0, xyz.0, 1e-5, "round-trip X");
            assert_near(back.1, xyz.1, 1e-5, "round-trip Y");
            assert_near(back.2, xyz.2, 1e-5, "round-trip Z");
        }
    }
}
