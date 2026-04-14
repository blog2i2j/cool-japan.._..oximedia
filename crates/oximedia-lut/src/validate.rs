//! Simple LUT validation predicates (f32 API).
//!
//! Provides lightweight boolean checks for 1D LUT integrity, complementing
//! the richer `lut_validate` module which produces structured reports.

/// Check that a 1D LUT is monotonically non-decreasing.
///
/// Returns `true` if every consecutive pair of values satisfies
/// `values[i+1] >= values[i]` (within a small floating-point tolerance).
///
/// An empty or single-element LUT is considered monotone.
///
/// # Arguments
///
/// * `lut` - 1D LUT values to check.
#[must_use]
pub fn check_1d_monotone(lut: &[f32]) -> bool {
    const TOLERANCE: f32 = 1e-7;
    lut.windows(2).all(|w| w[1] >= w[0] - TOLERANCE)
}

/// Check that a 1D LUT contains no clipped (out-of-range) values.
///
/// Returns `true` if all values lie within `[0.0, 1.0]` (within a small
/// tolerance).  Values that are `NaN` are treated as clipped.
///
/// # Arguments
///
/// * `lut` - 1D LUT values to check.
#[must_use]
pub fn check_no_clipping(lut: &[f32]) -> bool {
    const TOLERANCE: f32 = 1e-6;
    lut.iter().all(|&v| !v.is_nan() && v >= -TOLERANCE && v <= 1.0 + TOLERANCE)
}

/// Check both monotonicity and absence of clipping.
///
/// Equivalent to `check_1d_monotone(lut) && check_no_clipping(lut)`.
#[must_use]
pub fn check_1d_valid(lut: &[f32]) -> bool {
    check_1d_monotone(lut) && check_no_clipping(lut)
}

/// Check that no values in the LUT are `NaN` or infinite.
#[must_use]
pub fn check_no_nan_inf(lut: &[f32]) -> bool {
    lut.iter().all(|&v| !v.is_nan() && !v.is_infinite())
}

/// Check that a 1D LUT is monotonically non-increasing.
///
/// Returns `true` if every consecutive pair satisfies
/// `values[i+1] <= values[i]`.
#[must_use]
pub fn check_1d_monotone_decreasing(lut: &[f32]) -> bool {
    const TOLERANCE: f32 = 1e-7;
    lut.windows(2).all(|w| w[1] <= w[0] + TOLERANCE)
}

/// Return the maximum deviation from a linear ramp across the LUT.
///
/// A linear ramp has `lut[i] = i / (n - 1)`.  Returns `None` if the LUT
/// has fewer than 2 entries.
#[must_use]
pub fn max_deviation_from_identity(lut: &[f32]) -> Option<f32> {
    let n = lut.len();
    if n < 2 {
        return None;
    }
    let scale = (n - 1) as f32;
    let max_dev = lut
        .iter()
        .enumerate()
        .map(|(i, &v)| (v - i as f32 / scale).abs())
        .fold(0.0_f32, f32::max);
    Some(max_dev)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monotone_identity_lut() {
        let lut: Vec<f32> = (0..256).map(|i| i as f32 / 255.0).collect();
        assert!(check_1d_monotone(&lut));
    }

    #[test]
    fn test_monotone_constant_lut() {
        let lut = vec![0.5_f32; 64];
        assert!(check_1d_monotone(&lut));
    }

    #[test]
    fn test_monotone_fails_on_decrease() {
        let mut lut: Vec<f32> = (0..64).map(|i| i as f32 / 63.0).collect();
        lut[32] = 0.1; // dip
        assert!(!check_1d_monotone(&lut));
    }

    #[test]
    fn test_monotone_empty_and_single() {
        assert!(check_1d_monotone(&[]));
        assert!(check_1d_monotone(&[0.5]));
    }

    #[test]
    fn test_no_clipping_identity() {
        let lut: Vec<f32> = (0..256).map(|i| i as f32 / 255.0).collect();
        assert!(check_no_clipping(&lut));
    }

    #[test]
    fn test_no_clipping_fails_above_one() {
        let lut = vec![0.0_f32, 0.5, 1.5];
        assert!(!check_no_clipping(&lut));
    }

    #[test]
    fn test_no_clipping_fails_below_zero() {
        let lut = vec![-0.1_f32, 0.5, 1.0];
        assert!(!check_no_clipping(&lut));
    }

    #[test]
    fn test_no_clipping_fails_nan() {
        let lut = vec![0.0_f32, f32::NAN, 1.0];
        assert!(!check_no_clipping(&lut));
    }

    #[test]
    fn test_no_clipping_empty() {
        assert!(check_no_clipping(&[]));
    }

    #[test]
    fn test_check_1d_valid_identity() {
        let lut: Vec<f32> = (0..128).map(|i| i as f32 / 127.0).collect();
        assert!(check_1d_valid(&lut));
    }

    #[test]
    fn test_check_1d_valid_fails_non_monotone() {
        let lut = vec![0.0_f32, 0.5, 0.3, 1.0]; // dip at index 2
        assert!(!check_1d_valid(&lut));
    }

    #[test]
    fn test_check_no_nan_inf() {
        assert!(check_no_nan_inf(&[0.0, 0.5, 1.0]));
        assert!(!check_no_nan_inf(&[0.0, f32::NAN]));
        assert!(!check_no_nan_inf(&[0.0, f32::INFINITY]));
        assert!(!check_no_nan_inf(&[0.0, f32::NEG_INFINITY]));
    }

    #[test]
    fn test_monotone_decreasing() {
        let lut = vec![1.0_f32, 0.75, 0.5, 0.25, 0.0];
        assert!(check_1d_monotone_decreasing(&lut));
        assert!(!check_1d_monotone(&lut));
    }

    #[test]
    fn test_max_deviation_identity() {
        let lut: Vec<f32> = (0..256).map(|i| i as f32 / 255.0).collect();
        let dev = max_deviation_from_identity(&lut);
        assert!(dev.is_some());
        assert!(dev.expect("should be Some") < 1e-5);
    }

    #[test]
    fn test_max_deviation_offset() {
        let lut = vec![0.1_f32, 0.6, 1.1_f32.min(1.0)]; // offset by ~0.1
        let dev = max_deviation_from_identity(&lut);
        assert!(dev.is_some());
        assert!(dev.expect("should be Some") >= 0.09);
    }

    #[test]
    fn test_max_deviation_empty() {
        assert!(max_deviation_from_identity(&[]).is_none());
        assert!(max_deviation_from_identity(&[0.5]).is_none());
    }
}
