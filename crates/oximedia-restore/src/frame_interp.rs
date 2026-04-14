//! Video frame interpolation for missing frame reconstruction.
//!
//! When a video frame is lost (e.g., due to a tape dropout, packet loss, or
//! corrupted sector) the simplest repair strategy is to blend the surrounding
//! good frames.  This module provides a pixel-level blending interpolator that
//! reconstructs the missing frame as the arithmetic mean of the previous and
//! next frames.
//!
//! For each pixel position *i* the output is:
//!
//! ```text
//! out[i] = (prev[i] + next[i] + 1) / 2   (integer mid-point, no bias)
//! ```
//!
//! # Example
//!
//! ```
//! use oximedia_restore::frame_interp::interpolate_missing_frame;
//!
//! let prev = vec![0u8, 100, 200];
//! let next = vec![50u8, 150, 250];
//! let interp = interpolate_missing_frame(&prev, &next, 3);
//! assert_eq!(interp[0], 25);   // (0 + 50) / 2
//! assert_eq!(interp[1], 125);  // (100 + 150) / 2
//! assert_eq!(interp[2], 225);  // (200 + 250) / 2
//! ```

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Reconstruct a missing video frame by blending `prev` and `next`.
///
/// # Parameters
///
/// * `prev` — previous good frame pixel buffer (any format, 1 byte per element).
/// * `next` — next good frame pixel buffer (same length and format as `prev`).
/// * `len`  — number of elements to process.
///
/// # Behaviour
///
/// * If both `prev` and `next` have at least `len` bytes, each output byte is
///   the rounded average of the corresponding bytes.
/// * If only one source is long enough, its value is used verbatim.
/// * Missing bytes (when neither source is long enough) are filled with `0`.
///
/// # Returns
///
/// A `Vec<u8>` of length `len` containing the interpolated pixel data.
pub fn interpolate_missing_frame(prev: &[u8], next: &[u8], len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| {
            let p = prev.get(i).copied();
            let n = next.get(i).copied();
            match (p, n) {
                (Some(pv), Some(nv)) => {
                    // Round-to-nearest blend: (pv + nv + 1) >> 1
                    ((pv as u16 + nv as u16 + 1) >> 1) as u8
                }
                (Some(pv), None) => pv,
                (None, Some(nv)) => nv,
                (None, None) => 0,
            }
        })
        .collect()
}

/// Interpolate a missing frame using a weighted blend.
///
/// `alpha` is the weight given to `next` (0.0 = all prev, 1.0 = all next,
/// 0.5 = equal blend).  This is useful when the temporal position of the
/// missing frame is not exactly halfway between `prev` and `next`.
///
/// # Returns
///
/// A `Vec<u8>` of length `len`.
pub fn interpolate_weighted(prev: &[u8], next: &[u8], len: usize, alpha: f32) -> Vec<u8> {
    let alpha = alpha.clamp(0.0, 1.0);
    let beta = 1.0 - alpha;
    (0..len)
        .map(|i| {
            let p = prev.get(i).copied().unwrap_or(0) as f32;
            let n = next.get(i).copied().unwrap_or(0) as f32;
            (beta * p + alpha * n).round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blend_basic() {
        let prev = vec![0u8, 100, 200];
        let next = vec![50u8, 150, 250];
        let out = interpolate_missing_frame(&prev, &next, 3);
        // (0+50+1)/2=25, (100+150+1)/2=125, (200+250+1)/2=225
        assert_eq!(out, vec![25, 125, 225]);
    }

    #[test]
    fn test_same_frames_unchanged() {
        let frame = vec![128u8; 64];
        let out = interpolate_missing_frame(&frame, &frame, 64);
        assert_eq!(out, frame);
    }

    #[test]
    fn test_zero_length_returns_empty() {
        let out = interpolate_missing_frame(&[1, 2, 3], &[4, 5, 6], 0);
        assert!(out.is_empty());
    }

    #[test]
    fn test_prev_shorter_than_len_uses_next() {
        let prev: Vec<u8> = vec![0]; // only 1 byte
        let next: Vec<u8> = vec![200; 4]; // 4 bytes
        let out = interpolate_missing_frame(&prev, &next, 4);
        // Index 0: both present → (0 + 200 + 1) / 2 = 100
        assert_eq!(out[0], 100);
        // Indices 1-3: only next → 200
        for &s in &out[1..] {
            assert_eq!(s, 200, "missing prev byte should use next value");
        }
    }

    #[test]
    fn test_both_empty_gives_zeros() {
        let out = interpolate_missing_frame(&[], &[], 4);
        assert_eq!(out, vec![0u8; 4]);
    }

    #[test]
    fn test_output_length_equals_len() {
        let prev = vec![10u8; 100];
        let next = vec![20u8; 100];
        let out = interpolate_missing_frame(&prev, &next, 50);
        assert_eq!(out.len(), 50);
    }

    #[test]
    fn test_weighted_alpha_zero_is_all_prev() {
        let prev = vec![100u8; 8];
        let next = vec![200u8; 8];
        let out = interpolate_weighted(&prev, &next, 8, 0.0);
        assert_eq!(out, prev);
    }

    #[test]
    fn test_weighted_alpha_one_is_all_next() {
        let prev = vec![100u8; 8];
        let next = vec![200u8; 8];
        let out = interpolate_weighted(&prev, &next, 8, 1.0);
        assert_eq!(out, next);
    }

    #[test]
    fn test_weighted_alpha_half_is_average() {
        let prev = vec![100u8; 4];
        let next = vec![200u8; 4];
        let out = interpolate_weighted(&prev, &next, 4, 0.5);
        for &s in &out {
            assert_eq!(s, 150, "half-blend should give 150");
        }
    }

    #[test]
    fn test_blend_no_overflow() {
        // Both at maximum value → blend should stay at 255.
        let prev = vec![255u8; 4];
        let next = vec![255u8; 4];
        let out = interpolate_missing_frame(&prev, &next, 4);
        for &s in &out {
            assert_eq!(s, 255);
        }
    }

    #[test]
    fn test_rounding_up() {
        // (1 + 0 + 1) / 2 = 1  (rounds up via +1 before shift)
        let prev = vec![1u8];
        let next = vec![0u8];
        let out = interpolate_missing_frame(&prev, &next, 1);
        assert_eq!(out[0], 1, "should round up: got {}", out[0]);
    }
}
