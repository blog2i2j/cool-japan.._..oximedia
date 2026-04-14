//! HLG-to-PQ single-value conversion function.
//!
//! Provides the standalone [`hlg_to_pq`] function that converts a normalised
//! HLG signal value [0, 1] to a normalised PQ signal value [0, 1] using the
//! BT.2390 transfer function chain:
//!
//!   1. HLG EOTF  (signal → scene-linear, scaled to 1000 nit reference)
//!   2. Luminance scaling (1 000 nit → 10 000 nit normalisation)
//!   3. PQ OETF   (scene-linear → PQ code word)
//!
//! For full-frame and system-gamma-adjusted conversions see
//! [`crate::pq_hlg_convert`] and [`crate::hlg_advanced`].

#![allow(dead_code)]

use crate::transfer_function::{hlg_eotf, pq_oetf};
use crate::HdrError;

// ── hlg_to_pq ─────────────────────────────────────────────────────────────────

/// Convert a normalised HLG signal value to a normalised PQ signal value.
///
/// Follows the BT.2390 reference chain:
/// - HLG reference display peak: 1 000 nits.
/// - PQ reference display peak:  10 000 nits.
///
/// # Arguments
///
/// * `hlg` — HLG signal in the range [0, 1].
///
/// # Returns
///
/// * `Ok(f32)` — PQ signal in [0, 1] on success.
/// * `Err(HdrError::InvalidLuminance)` — if `hlg` is outside [0, 1].
///
/// # Example
///
/// ```rust
/// use oximedia_hdr::hlg_to_pq::hlg_to_pq;
///
/// let pq = hlg_to_pq(0.75).expect("valid HLG value");
/// assert!(pq > 0.0 && pq <= 1.0);
/// ```
pub fn hlg_to_pq(hlg: f32) -> Result<f32, HdrError> {
    if !(0.0..=1.0).contains(&hlg) {
        return Err(HdrError::InvalidLuminance(hlg));
    }

    // Step 1 — HLG EOTF: HLG signal → scene-linear [0, 1] (1 000 nit reference)
    let scene_linear = hlg_eotf(hlg as f64)
        .map_err(|_| HdrError::InvalidLuminance(hlg))?;

    // Step 2 — Scale 1 000 nit → 10 000 nit normalisation (÷ 10)
    let pq_linear = scene_linear / 10.0;

    // Step 3 — PQ OETF: scene-linear → PQ signal [0, 1]
    let pq_signal = pq_oetf(pq_linear)
        .map_err(|_| HdrError::InvalidLuminance(pq_linear as f32))?;

    Ok(pq_signal as f32)
}

/// Convert a batch of HLG signal values to PQ, skipping invalid values.
///
/// Invalid values (outside [0, 1]) in `hlg_slice` are mapped to 0.0 in the
/// output without returning an error.
#[must_use]
pub fn hlg_to_pq_batch(hlg_slice: &[f32]) -> Vec<f32> {
    hlg_slice
        .iter()
        .map(|&v| hlg_to_pq(v).unwrap_or(0.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hlg_to_pq_zero_is_zero() {
        let pq = hlg_to_pq(0.0).expect("valid");
        assert!(pq.abs() < 1e-3, "hlg=0 should produce near-zero PQ, got {pq}");
    }

    #[test]
    fn test_hlg_to_pq_one_in_range() {
        let pq = hlg_to_pq(1.0).expect("valid");
        assert!(pq > 0.0 && pq <= 1.0, "hlg=1 PQ out of range: {pq}");
    }

    #[test]
    fn test_hlg_to_pq_mid_range() {
        let pq = hlg_to_pq(0.5).expect("valid");
        assert!(pq > 0.0 && pq < 1.0, "mid-range PQ should be in (0,1), got {pq}");
    }

    #[test]
    fn test_hlg_to_pq_monotonically_increases() {
        let values = [0.0f32, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0];
        let mut prev = -1.0f32;
        for &v in &values {
            let pq = hlg_to_pq(v).expect("valid");
            assert!(pq >= prev, "not monotonic at hlg={v}: pq={pq} < prev={prev}");
            prev = pq;
        }
    }

    #[test]
    fn test_hlg_to_pq_rejects_negative() {
        assert!(hlg_to_pq(-0.1).is_err());
    }

    #[test]
    fn test_hlg_to_pq_rejects_above_one() {
        assert!(hlg_to_pq(1.1).is_err());
    }

    #[test]
    fn test_hlg_to_pq_output_in_unit_range() {
        for i in 0..=20 {
            let hlg = i as f32 / 20.0;
            let pq = hlg_to_pq(hlg).expect("valid");
            assert!(
                (0.0..=1.0).contains(&pq),
                "PQ out of [0,1] for hlg={hlg}: pq={pq}"
            );
        }
    }

    #[test]
    fn test_hlg_to_pq_batch_length() {
        let input = vec![0.0f32, 0.25, 0.5, 0.75, 1.0];
        let output = hlg_to_pq_batch(&input);
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_hlg_to_pq_batch_invalid_mapped_to_zero() {
        let input = vec![-0.5f32, 0.5, 2.0];
        let output = hlg_to_pq_batch(&input);
        assert_eq!(output[0], 0.0, "negative hlg → 0.0");
        assert_eq!(output[2], 0.0, "hlg > 1.0 → 0.0");
        assert!(output[1] > 0.0, "valid hlg → nonzero PQ");
    }
}
