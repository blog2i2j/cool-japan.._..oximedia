//! LUT combination utilities (f32 API).
//!
//! Provides functions for combining 1D LUTs by sequential application,
//! enabling efficient pipeline of multiple colour corrections.

/// Apply LUT `a` to an input, then pass the result through LUT `b`.
///
/// Both LUTs must be non-empty. The input domain and output range are
/// normalised to `[0.0, 1.0]`.  The combined LUT has the same number of
/// entries as `a`.
///
/// # Arguments
///
/// * `a` - First 1D LUT to apply (values in `[0.0, 1.0]`).
/// * `b` - Second 1D LUT applied to the output of `a`.
///
/// # Returns
///
/// A new 1D LUT of length `a.len()` representing the sequential application
/// of `a` then `b`.
#[must_use]
pub fn combine_luts_1d(a: &[f32], b: &[f32]) -> Vec<f32> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }

    let b_len = b.len();
    let b_scale = (b_len - 1).max(1) as f32;

    a.iter()
        .map(|&val_a| {
            // Clamp the output of LUT a to [0, 1] before indexing into b
            let clamped = val_a.clamp(0.0, 1.0);
            let pos = clamped * b_scale;
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(b_len - 1);
            let frac = pos - lo as f32;
            b[lo] * (1.0 - frac) + b[hi] * frac
        })
        .collect()
}

/// Combine N 1D LUTs sequentially, applying each in order.
///
/// Returns an empty `Vec` if the slice of LUTs is empty or any LUT is empty.
/// The output length matches the length of the first LUT.
#[must_use]
pub fn combine_luts_1d_chain(luts: &[&[f32]]) -> Vec<f32> {
    match luts {
        [] => Vec::new(),
        [first, rest @ ..] => {
            let mut result: Vec<f32> = first.to_vec();
            for lut in rest {
                result = combine_luts_1d(&result, lut);
                if result.is_empty() {
                    return Vec::new();
                }
            }
            result
        }
    }
}

/// Combine two 1D per-channel LUTs (RGB interleaved: `[r0,g0,b0, r1,g1,b1, ...]`).
///
/// Each `a_rgb` and `b_rgb` slice must have length `3 * n` for some `n >= 1`.
/// Returns a combined RGB LUT of the same length.
#[must_use]
pub fn combine_luts_1d_rgb(a_rgb: &[f32], b_rgb: &[f32]) -> Vec<f32> {
    if a_rgb.len() % 3 != 0 || b_rgb.len() % 3 != 0 || a_rgb.is_empty() || b_rgb.is_empty() {
        return Vec::new();
    }

    let a_n = a_rgb.len() / 3;
    let b_n = b_rgb.len() / 3;

    // Split into per-channel slices
    let mut a_r = Vec::with_capacity(a_n);
    let mut a_g = Vec::with_capacity(a_n);
    let mut a_b = Vec::with_capacity(a_n);
    for i in 0..a_n {
        a_r.push(a_rgb[i * 3]);
        a_g.push(a_rgb[i * 3 + 1]);
        a_b.push(a_rgb[i * 3 + 2]);
    }

    let mut b_r = Vec::with_capacity(b_n);
    let mut b_g = Vec::with_capacity(b_n);
    let mut b_b = Vec::with_capacity(b_n);
    for i in 0..b_n {
        b_r.push(b_rgb[i * 3]);
        b_g.push(b_rgb[i * 3 + 1]);
        b_b.push(b_rgb[i * 3 + 2]);
    }

    let combined_r = combine_luts_1d(&a_r, &b_r);
    let combined_g = combine_luts_1d(&a_g, &b_g);
    let combined_b = combine_luts_1d(&a_b, &b_b);

    if combined_r.len() != a_n || combined_g.len() != a_n || combined_b.len() != a_n {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(a_n * 3);
    for i in 0..a_n {
        result.push(combined_r[i]);
        result.push(combined_g[i]);
        result.push(combined_b[i]);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_lut(size: usize) -> Vec<f32> {
        if size < 2 {
            return vec![0.0; size];
        }
        (0..size).map(|i| i as f32 / (size - 1) as f32).collect()
    }

    fn constant_lut(size: usize, val: f32) -> Vec<f32> {
        vec![val; size]
    }

    #[test]
    fn test_combine_identity_identity() {
        let id = identity_lut(256);
        let combined = combine_luts_1d(&id, &id);
        assert_eq!(combined.len(), 256);
        for (i, &v) in combined.iter().enumerate() {
            let expected = i as f32 / 255.0;
            assert!((v - expected).abs() < 1e-5, "index {i}: expected {expected}, got {v}");
        }
    }

    #[test]
    fn test_combine_identity_then_constant() {
        let id = identity_lut(64);
        let constant = constant_lut(64, 0.5);
        let combined = combine_luts_1d(&id, &constant);
        // Applying identity then a constant LUT should yield ~0.5 everywhere
        for &v in &combined {
            assert!((v - 0.5).abs() < 1e-5, "expected 0.5, got {v}");
        }
    }

    #[test]
    fn test_combine_constant_then_identity() {
        let constant = constant_lut(64, 0.3);
        let id = identity_lut(64);
        let combined = combine_luts_1d(&constant, &id);
        // Constant 0.3 passed through identity → 0.3
        for &v in &combined {
            assert!((v - 0.3).abs() < 1e-5, "expected 0.3, got {v}");
        }
    }

    #[test]
    fn test_combine_empty_returns_empty() {
        let id = identity_lut(64);
        assert!(combine_luts_1d(&[], &id).is_empty());
        assert!(combine_luts_1d(&id, &[]).is_empty());
    }

    #[test]
    fn test_combine_double_gamma() {
        // LUT that squares the input (gamma = 2)
        let size = 256;
        let gamma2: Vec<f32> = (0..size)
            .map(|i| {
                let v = i as f32 / (size - 1) as f32;
                v * v
            })
            .collect();
        let combined = combine_luts_1d(&gamma2, &gamma2);
        // At i=128 (≈0.502): 0.502^2 ≈ 0.252, then 0.252^2 ≈ 0.0635
        let mid = combined[128];
        let expected = {
            let v = 128.0_f32 / 255.0;
            (v * v) * (v * v)
        };
        assert!((mid - expected).abs() < 0.01, "got {mid}, expected {expected}");
    }

    #[test]
    fn test_combine_chain_three() {
        let id = identity_lut(128);
        let half: Vec<f32> = id.iter().map(|&v| v * 0.5).collect();
        let result = combine_luts_1d_chain(&[&id, &half, &id]);
        // identity → scale by 0.5 → identity: result[i] ≈ (i/127) * 0.5
        assert_eq!(result.len(), 128);
        for (i, &v) in result.iter().enumerate() {
            let expected = (i as f32 / 127.0) * 0.5;
            assert!((v - expected).abs() < 0.02, "index {i}: expected {expected}, got {v}");
        }
    }

    #[test]
    fn test_combine_chain_empty() {
        assert!(combine_luts_1d_chain(&[]).is_empty());
    }

    #[test]
    fn test_combine_luts_1d_rgb_identity() {
        let size = 64;
        let id_single = identity_lut(size);
        let mut id_rgb = Vec::with_capacity(size * 3);
        for i in 0..size {
            let v = i as f32 / (size - 1) as f32;
            id_rgb.push(v);
            id_rgb.push(v);
            id_rgb.push(v);
        }
        let combined = combine_luts_1d_rgb(&id_rgb, &id_rgb);
        assert_eq!(combined.len(), size * 3);
        // Should match identity
        let _ = id_single; // just to confirm it compiles
        for (i, chunk) in combined.chunks(3).enumerate() {
            let expected = i as f32 / (size - 1) as f32;
            assert!((chunk[0] - expected).abs() < 1e-5);
            assert!((chunk[1] - expected).abs() < 1e-5);
            assert!((chunk[2] - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_combine_luts_1d_rgb_bad_length() {
        // Length not multiple of 3
        let bad = vec![0.0_f32; 7];
        let ok = vec![0.0_f32; 9];
        assert!(combine_luts_1d_rgb(&bad, &ok).is_empty());
    }
}
