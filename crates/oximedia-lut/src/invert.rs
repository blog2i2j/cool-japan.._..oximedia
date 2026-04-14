//! 1D LUT inversion utilities (f32 API).
//!
//! Inverts a monotone 1D LUT by finding, for each output value, the input
//! position that produces it via binary search with linear interpolation.

/// Invert a monotone 1D LUT.
///
/// Assumes the LUT is monotonically increasing or decreasing.  For each output
/// value position the function uses binary search + linear interpolation to
/// find the input that produces it.
///
/// The output LUT has the same length as the input.
///
/// # Arguments
///
/// * `lut` - Source 1D LUT values (normalised to `[0.0, 1.0]`).
///
/// # Returns
///
/// Inverted LUT of the same length, or an empty `Vec` if `lut` has fewer than
/// 2 entries.
#[must_use]
pub fn invert_1d_lut(lut: &[f32]) -> Vec<f32> {
    let n = lut.len();
    if n < 2 {
        return Vec::new();
    }

    let is_increasing = lut[n - 1] >= lut[0];
    let out_scale = (n - 1) as f32;
    let in_scale = (n - 1) as f32;

    (0..n)
        .map(|i| {
            let target = i as f32 / out_scale;
            let frac_idx = if is_increasing {
                search_increasing(lut, target)
            } else {
                search_decreasing(lut, target)
            };
            (frac_idx / in_scale).clamp(0.0, 1.0)
        })
        .collect()
}

/// Invert a monotone 1D LUT into an output LUT of a specified size.
///
/// Useful when a finer-resolution inverse is needed than the source.
///
/// # Arguments
///
/// * `lut` - Source 1D LUT (monotone, length >= 2).
/// * `out_size` - Desired output length (>= 2).
#[must_use]
pub fn invert_1d_lut_resized(lut: &[f32], out_size: usize) -> Vec<f32> {
    let n = lut.len();
    if n < 2 || out_size < 2 {
        return Vec::new();
    }

    let is_increasing = lut[n - 1] >= lut[0];
    let out_scale = (out_size - 1) as f32;
    let in_scale = (n - 1) as f32;

    (0..out_size)
        .map(|i| {
            let target = i as f32 / out_scale;
            let frac_idx = if is_increasing {
                search_increasing(lut, target)
            } else {
                search_decreasing(lut, target)
            };
            (frac_idx / in_scale).clamp(0.0, 1.0)
        })
        .collect()
}

/// Binary search in a monotonically increasing slice, returning the fractional
/// index where `values[idx] ≈ target`.
fn search_increasing(values: &[f32], target: f32) -> f32 {
    let n = values.len();

    // Clamp to valid range
    if target <= values[0] {
        return 0.0;
    }
    if target >= values[n - 1] {
        return (n - 1) as f32;
    }

    let mut lo = 0usize;
    let mut hi = n - 1;

    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if values[mid] <= target {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let denom = values[hi] - values[lo];
    if denom.abs() < f32::EPSILON {
        lo as f32
    } else {
        lo as f32 + (target - values[lo]) / denom
    }
}

/// Binary search in a monotonically decreasing slice.
fn search_decreasing(values: &[f32], target: f32) -> f32 {
    let n = values.len();

    if target >= values[0] {
        return 0.0;
    }
    if target <= values[n - 1] {
        return (n - 1) as f32;
    }

    let mut lo = 0usize;
    let mut hi = n - 1;

    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if values[mid] >= target {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let denom = values[hi] - values[lo];
    if denom.abs() < f32::EPSILON {
        lo as f32
    } else {
        lo as f32 + (target - values[lo]) / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_lut(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32 / (n - 1) as f32).collect()
    }

    fn gamma_lut(n: usize, gamma: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (i as f32 / (n - 1) as f32).powf(gamma))
            .collect()
    }

    fn apply_lut_1d(lut: &[f32], t: f32) -> f32 {
        let n = lut.len();
        if n < 2 {
            return 0.0;
        }
        let scale = (n - 1) as f32;
        let pos = t.clamp(0.0, 1.0) * scale;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(n - 1);
        let frac = pos - lo as f32;
        lut[lo] * (1.0 - frac) + lut[hi] * frac
    }

    #[test]
    fn test_invert_identity_is_identity() {
        let id = identity_lut(256);
        let inv = invert_1d_lut(&id);
        assert_eq!(inv.len(), 256);
        for (i, &v) in inv.iter().enumerate() {
            let expected = i as f32 / 255.0;
            assert!(
                (v - expected).abs() < 1e-4,
                "index {i}: expected {expected}, got {v}"
            );
        }
    }

    #[test]
    fn test_invert_gamma_roundtrip() {
        let n = 512;
        let gamma_fwd = gamma_lut(n, 2.2);
        let gamma_inv = invert_1d_lut(&gamma_fwd);

        // Apply forward then inverse; should recover the original input
        for step in 0..=10 {
            let t = step as f32 / 10.0;
            let encoded = apply_lut_1d(&gamma_fwd, t);
            let decoded = apply_lut_1d(&gamma_inv, encoded);
            assert!(
                (decoded - t).abs() < 0.02,
                "t={t}: decoded={decoded}"
            );
        }
    }

    #[test]
    fn test_invert_empty_returns_empty() {
        assert!(invert_1d_lut(&[]).is_empty());
        assert!(invert_1d_lut(&[0.5]).is_empty());
    }

    #[test]
    fn test_invert_decreasing() {
        // A decreasing LUT: 1.0, 0.75, 0.5, 0.25, 0.0
        let dec = vec![1.0_f32, 0.75, 0.5, 0.25, 0.0];
        let inv = invert_1d_lut(&dec);
        assert_eq!(inv.len(), 5);
        // inv maps [0..1] back, values should be decreasing too
        assert!(inv[0] >= inv[4]);
    }

    #[test]
    fn test_invert_resized_larger() {
        let n = 64;
        let gamma_fwd = gamma_lut(n, 2.0);
        let inv = invert_1d_lut_resized(&gamma_fwd, 256);
        assert_eq!(inv.len(), 256);
        // Roundtrip check at midpoint
        let t = 0.5_f32;
        let encoded = apply_lut_1d(&gamma_fwd, t);
        let decoded = apply_lut_1d(&inv, encoded);
        assert!((decoded - t).abs() < 0.05, "decoded={decoded}");
    }

    #[test]
    fn test_invert_resized_empty_on_small_input() {
        let tiny = vec![0.0_f32, 1.0];
        assert!(invert_1d_lut_resized(&tiny, 0).is_empty());
        assert!(invert_1d_lut_resized(&tiny, 1).is_empty());
        assert!(invert_1d_lut_resized(&[], 10).is_empty());
    }

    #[test]
    fn test_invert_clamps_output() {
        let id = identity_lut(128);
        let inv = invert_1d_lut(&id);
        for &v in &inv {
            assert!(v >= 0.0 && v <= 1.0, "out-of-range value: {v}");
        }
    }
}
