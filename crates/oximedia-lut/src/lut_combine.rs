//! LUT combination and composition utilities.
//!
//! Provides:
//! * Sequential application of multiple 3-D LUTs.
//! * Composition of a 1-D pre-curve with a 3-D body LUT.
//! * Identity detection so trivial LUTs can be skipped.

use crate::Rgb;

// ---------------------------------------------------------------------------
// Identity detection
// ---------------------------------------------------------------------------

/// Tolerance used when checking whether a LUT is the identity.
const IDENTITY_EPSILON: f64 = 1e-6;

/// Return `true` if the 3-D `lut` (stored `[r][g][b]`, size³ entries) is
/// effectively an identity transform.
#[allow(dead_code)]
#[must_use]
pub fn is_identity_lut3d(lut: &[Rgb], size: usize) -> bool {
    if lut.len() != size * size * size || size < 2 {
        return false;
    }
    let scale = (size - 1) as f64;
    for r in 0..size {
        for g in 0..size {
            for b in 0..size {
                let expected = [r as f64 / scale, g as f64 / scale, b as f64 / scale];
                let entry = &lut[r * size * size + g * size + b];
                for ch in 0..3 {
                    if (entry[ch] - expected[ch]).abs() > IDENTITY_EPSILON {
                        return false;
                    }
                }
            }
        }
    }
    true
}

/// Return `true` if the 1-D `curve` (per-channel, each channel has `size`
/// entries) is effectively an identity.
///
/// `curve[ch][i]` should equal `i as f64 / (size - 1) as f64` for all i.
#[allow(dead_code)]
#[must_use]
pub fn is_identity_lut1d(curve: &[[f64; 3]], size: usize) -> bool {
    if curve.len() != size || size < 2 {
        return false;
    }
    let scale = (size - 1) as f64;
    for (i, entry) in curve.iter().enumerate() {
        let expected = i as f64 / scale;
        for ch in 0..3 {
            if (entry[ch] - expected).abs() > IDENTITY_EPSILON {
                return false;
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// 1-D curve application
// ---------------------------------------------------------------------------

/// Apply a 1-D per-channel curve to a colour value.
///
/// `curve` – slice of length `size` where each element is `[r, g, b]`
/// representing the curve output at that normalised input position.
#[allow(dead_code)]
#[must_use]
pub fn apply_curve(curve: &[[f64; 3]], input: &Rgb) -> Rgb {
    let size = curve.len();
    assert!(size >= 2, "curve must have at least 2 entries");
    let scale = (size - 1) as f64;
    let mut out = [0.0f64; 3];
    for ch in 0..3 {
        let v = input[ch].clamp(0.0, 1.0) * scale;
        let lo = v.floor() as usize;
        let hi = (lo + 1).min(size - 1);
        let frac = v - lo as f64;
        out[ch] = curve[lo][ch] * (1.0 - frac) + curve[hi][ch] * frac;
    }
    out
}

// ---------------------------------------------------------------------------
// 3-D LUT application (trilinear)
// ---------------------------------------------------------------------------

/// Apply a 3-D LUT to a colour value using trilinear interpolation.
///
/// `lut` – flat slice of length `size³`, stored `[r][g][b]`.
#[allow(dead_code)]
#[must_use]
pub fn apply_lut3d(lut: &[Rgb], size: usize, input: &Rgb) -> Rgb {
    assert!(size >= 2, "size must be >= 2");
    assert_eq!(lut.len(), size * size * size, "LUT length mismatch");

    let scale = (size - 1) as f64;
    let r = input[0].clamp(0.0, 1.0) * scale;
    let g = input[1].clamp(0.0, 1.0) * scale;
    let b = input[2].clamp(0.0, 1.0) * scale;

    let r0 = r.floor() as usize;
    let g0 = g.floor() as usize;
    let b0 = b.floor() as usize;
    let r1 = (r0 + 1).min(size - 1);
    let g1 = (g0 + 1).min(size - 1);
    let b1 = (b0 + 1).min(size - 1);
    let dr = r - r0 as f64;
    let dg = g - g0 as f64;
    let db = b - b0 as f64;

    let idx = |ri: usize, gi: usize, bi: usize| -> Rgb { lut[ri * size * size + gi * size + bi] };

    let mut out = [0.0f64; 3];
    for ch in 0..3 {
        let c000 = idx(r0, g0, b0)[ch];
        let c100 = idx(r1, g0, b0)[ch];
        let c010 = idx(r0, g1, b0)[ch];
        let c110 = idx(r1, g1, b0)[ch];
        let c001 = idx(r0, g0, b1)[ch];
        let c101 = idx(r1, g0, b1)[ch];
        let c011 = idx(r0, g1, b1)[ch];
        let c111 = idx(r1, g1, b1)[ch];

        out[ch] = c000 * (1.0 - dr) * (1.0 - dg) * (1.0 - db)
            + c100 * dr * (1.0 - dg) * (1.0 - db)
            + c010 * (1.0 - dr) * dg * (1.0 - db)
            + c110 * dr * dg * (1.0 - db)
            + c001 * (1.0 - dr) * (1.0 - dg) * db
            + c101 * dr * (1.0 - dg) * db
            + c011 * (1.0 - dr) * dg * db
            + c111 * dr * dg * db;
    }
    out
}

// ---------------------------------------------------------------------------
// Composition
// ---------------------------------------------------------------------------

/// Apply a 1-D pre-curve followed by a 3-D LUT.
///
/// This is the standard way to compose a 1D tone / log curve with a 3D
/// creative LUT in a single pass.
#[allow(dead_code)]
#[must_use]
pub fn apply_1d_then_3d(curve: &[[f64; 3]], lut: &[Rgb], lut_size: usize, input: &Rgb) -> Rgb {
    let after_curve = apply_curve(curve, input);
    apply_lut3d(lut, lut_size, &after_curve)
}

/// Apply two 3-D LUTs sequentially: `first` is applied, then `second`.
#[allow(dead_code)]
#[must_use]
pub fn apply_sequential(
    first: &[Rgb],
    first_size: usize,
    second: &[Rgb],
    second_size: usize,
    input: &Rgb,
) -> Rgb {
    let intermediate = apply_lut3d(first, first_size, input);
    apply_lut3d(second, second_size, &intermediate)
}

/// Bake two sequential 3-D LUTs into one combined LUT.
///
/// Samples `first` at each lattice point of the output, then applies `second`
/// to produce the combined output. The resulting LUT has `out_size³` entries.
#[allow(dead_code)]
#[must_use]
pub fn bake_sequential(
    first: &[Rgb],
    first_size: usize,
    second: &[Rgb],
    second_size: usize,
    out_size: usize,
) -> Vec<Rgb> {
    let scale = (out_size - 1) as f64;
    let mut out = Vec::with_capacity(out_size * out_size * out_size);
    for r in 0..out_size {
        for g in 0..out_size {
            for b in 0..out_size {
                let input: Rgb = [r as f64 / scale, g as f64 / scale, b as f64 / scale];
                let combined = apply_sequential(first, first_size, second, second_size, &input);
                out.push(combined);
            }
        }
    }
    out
}

/// Bake a 1-D curve followed by a 3-D LUT into a new single 3-D LUT.
#[allow(dead_code)]
#[must_use]
pub fn bake_1d_then_3d(
    curve: &[[f64; 3]],
    lut: &[Rgb],
    lut_size: usize,
    out_size: usize,
) -> Vec<Rgb> {
    let scale = (out_size - 1) as f64;
    let mut out = Vec::with_capacity(out_size * out_size * out_size);
    for r in 0..out_size {
        for g in 0..out_size {
            for b in 0..out_size {
                let input: Rgb = [r as f64 / scale, g as f64 / scale, b as f64 / scale];
                out.push(apply_1d_then_3d(curve, lut, lut_size, &input));
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_lut3d(size: usize) -> Vec<Rgb> {
        let scale = (size - 1) as f64;
        let mut lut = Vec::with_capacity(size * size * size);
        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    lut.push([r as f64 / scale, g as f64 / scale, b as f64 / scale]);
                }
            }
        }
        lut
    }

    fn identity_curve(size: usize) -> Vec<[f64; 3]> {
        let scale = (size - 1) as f64;
        (0..size)
            .map(|i| {
                let v = i as f64 / scale;
                [v, v, v]
            })
            .collect()
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn rgb_approx(a: &Rgb, b: &Rgb) -> bool {
        approx_eq(a[0], b[0]) && approx_eq(a[1], b[1]) && approx_eq(a[2], b[2])
    }

    #[test]
    fn test_is_identity_lut3d_true() {
        let lut = identity_lut3d(3);
        assert!(is_identity_lut3d(&lut, 3));
    }

    #[test]
    fn test_is_identity_lut3d_false() {
        let mut lut = identity_lut3d(3);
        lut[0] = [0.1, 0.1, 0.1]; // Corrupt first entry.
        assert!(!is_identity_lut3d(&lut, 3));
    }

    #[test]
    fn test_is_identity_curve_true() {
        let curve = identity_curve(33);
        assert!(is_identity_lut1d(&curve, 33));
    }

    #[test]
    fn test_is_identity_curve_false() {
        let mut curve = identity_curve(33);
        curve[5] = [0.9, 0.9, 0.9];
        assert!(!is_identity_lut1d(&curve, 33));
    }

    #[test]
    fn test_apply_curve_identity() {
        let curve = identity_curve(33);
        let inp = [0.4, 0.6, 0.2];
        let out = apply_curve(&curve, &inp);
        assert!(rgb_approx(&out, &inp));
    }

    #[test]
    fn test_apply_lut3d_identity() {
        let lut = identity_lut3d(5);
        let inp = [0.25, 0.5, 0.75];
        let out = apply_lut3d(&lut, 5, &inp);
        assert!(rgb_approx(&out, &inp));
    }

    #[test]
    fn test_apply_lut3d_clamps_input() {
        let lut = identity_lut3d(3);
        let out = apply_lut3d(&lut, 3, &[-0.5, 0.5, 1.5]);
        assert!(out[0] >= 0.0 && out[0] <= 1.0 + 1e-9);
        assert!(out[2] >= 0.0 && out[2] <= 1.0 + 1e-9);
    }

    #[test]
    fn test_apply_1d_then_3d_identity() {
        let curve = identity_curve(33);
        let lut = identity_lut3d(5);
        let inp = [0.3, 0.7, 0.1];
        let out = apply_1d_then_3d(&curve, &lut, 5, &inp);
        // Both identity → output == input.
        for ch in 0..3 {
            assert!((out[ch] - inp[ch]).abs() < 0.01);
        }
    }

    #[test]
    fn test_apply_sequential_identity_identity() {
        let lut = identity_lut3d(3);
        let inp = [0.2, 0.5, 0.8];
        let out = apply_sequential(&lut, 3, &lut, 3, &inp);
        assert!(rgb_approx(&out, &inp));
    }

    #[test]
    fn test_bake_sequential_identity() {
        let lut = identity_lut3d(3);
        let baked = bake_sequential(&lut, 3, &lut, 3, 3);
        assert!(is_identity_lut3d(&baked, 3));
    }

    #[test]
    fn test_bake_1d_then_3d_identity() {
        let curve = identity_curve(33);
        let lut = identity_lut3d(3);
        let baked = bake_1d_then_3d(&curve, &lut, 3, 3);
        assert!(is_identity_lut3d(&baked, 3));
    }

    #[test]
    fn test_apply_curve_clamping() {
        let curve = identity_curve(33);
        // Over-range input should be clamped.
        let out = apply_curve(&curve, &[2.0, -1.0, 0.5]);
        assert!(out[0] <= 1.0 + 1e-9);
        assert!(out[1] >= -1e-9);
    }

    #[test]
    fn test_bake_sequential_produces_correct_size() {
        let lut = identity_lut3d(3);
        let baked = bake_sequential(&lut, 3, &lut, 3, 5);
        assert_eq!(baked.len(), 5 * 5 * 5);
    }

    #[test]
    fn test_apply_curve_endpoint_accuracy() {
        let curve = identity_curve(33);
        let black = apply_curve(&curve, &[0.0, 0.0, 0.0]);
        let white = apply_curve(&curve, &[1.0, 1.0, 1.0]);
        assert!(rgb_approx(&black, &[0.0, 0.0, 0.0]));
        assert!(rgb_approx(&white, &[1.0, 1.0, 1.0]));
    }
}
