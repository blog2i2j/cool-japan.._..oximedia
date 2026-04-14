//! Error concealment for lost or corrupted video frames.
//!
//! When a video frame is missing due to packet loss, decoding error, or
//! transmission failure, the decoder must produce a substitute frame to
//! maintain temporal continuity.  This module provides simple and effective
//! concealment strategies based on reference frame data.
//!
//! ## Strategies
//!
//! - **Frame copy** (`conceal_missing_frame`): Copies the previous reference
//!   frame verbatim.  This is the simplest and fastest strategy; effective
//!   when motion between frames is small.
//! - **Temporal blend** (`conceal_blend`): Blends between the previous and a
//!   fallback (e.g. grey or next available) frame at a configurable ratio.
//! - **Motion-compensated** (`conceal_motion_compensated`): Applies a simple
//!   global translation offset to the previous frame, useful when a dominant
//!   camera-pan motion is known.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

// ---------------------------------------------------------------------------
// Frame copy (primary API)
// ---------------------------------------------------------------------------

/// Conceal a missing frame by copying the previous reference frame.
///
/// This is the simplest error-concealment strategy: the decoder re-uses the
/// most recently decoded frame as a substitute for the lost frame.  For low-
/// motion content the result is indistinguishable; for high-motion content
/// it may produce a brief freeze.
///
/// # Parameters
/// - `prev`  – pixel buffer of the previous (reference) frame, interleaved
///             YUV or RGB u8, length `3 * w * h`.
/// - `w`     – frame width in pixels.
/// - `h`     – frame height in pixels.
///
/// # Returns
/// A `Vec<u8>` containing the concealed frame (identical to `prev`).
///
/// # Panics
/// Panics if `prev.len() != 3 * w * h`.
#[must_use]
pub fn conceal_missing_frame(prev: &[u8], w: u32, h: u32) -> Vec<u8> {
    let expected = 3 * (w as usize) * (h as usize);
    assert_eq!(
        prev.len(),
        expected,
        "conceal_missing_frame: prev length {len} != 3*w*h {expected}",
        len = prev.len()
    );
    prev.to_vec()
}

// ---------------------------------------------------------------------------
// Temporal blend
// ---------------------------------------------------------------------------

/// Conceal a missing frame by blending `prev` with a background colour.
///
/// When no future reference frame is available this replaces the missing
/// frame with a fade toward `bg_color`:
/// ```text
/// out[i] = alpha * prev[i] + (1 - alpha) * bg_color
/// ```
///
/// # Parameters
/// - `prev`     – previous frame pixel buffer, interleaved RGB u8, length
///               `3 * w * h`.
/// - `w`        – frame width in pixels.
/// - `h`        – frame height in pixels.
/// - `alpha`    – blending weight for `prev` in [0.0, 1.0].  1.0 = pure copy;
///               0.0 = pure `bg_color`.
/// - `bg_color` – background RGB colour to blend toward, as `[R, G, B]`.
///
/// # Panics
/// Panics if `prev.len() != 3 * w * h`.
#[must_use]
pub fn conceal_blend(prev: &[u8], w: u32, h: u32, alpha: f32, bg_color: [u8; 3]) -> Vec<u8> {
    let expected = 3 * (w as usize) * (h as usize);
    assert_eq!(
        prev.len(),
        expected,
        "conceal_blend: prev length {len} != 3*w*h {expected}",
        len = prev.len()
    );

    let alpha = alpha.clamp(0.0, 1.0);
    let one_minus_alpha = 1.0 - alpha;

    let mut out = Vec::with_capacity(expected);
    let n_pixels = (w * h) as usize;

    for i in 0..n_pixels {
        for ch in 0..3usize {
            let pv = prev[i * 3 + ch] as f32;
            let bg = bg_color[ch] as f32;
            let blended = alpha * pv + one_minus_alpha * bg;
            out.push(blended.round().clamp(0.0, 255.0) as u8);
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Motion-compensated concealment
// ---------------------------------------------------------------------------

/// Conceal a missing frame using a constant global motion offset.
///
/// Applies a pixel-level translation `(dx, dy)` to `prev` to approximate
/// camera-pan motion.  Pixels that are shifted out-of-bounds are filled from
/// `prev` at the clamped border (clamp-to-edge).
///
/// # Parameters
/// - `prev` – previous frame pixel buffer, interleaved RGB u8, length
///            `3 * w * h`.
/// - `w`    – frame width in pixels.
/// - `h`    – frame height in pixels.
/// - `dx`   – horizontal translation (positive = shift right).
/// - `dy`   – vertical translation (positive = shift down).
///
/// # Panics
/// Panics if `prev.len() != 3 * w * h`.
#[must_use]
pub fn conceal_motion_compensated(prev: &[u8], w: u32, h: u32, dx: i32, dy: i32) -> Vec<u8> {
    let expected = 3 * (w as usize) * (h as usize);
    assert_eq!(
        prev.len(),
        expected,
        "conceal_motion_compensated: prev length {len} != 3*w*h {expected}",
        len = prev.len()
    );

    let w_i = w as i32;
    let h_i = h as i32;
    let mut out = vec![0u8; expected];

    for y in 0..h as i32 {
        for x in 0..w_i {
            // Source coordinate in previous frame
            let src_x = (x - dx).clamp(0, w_i - 1) as usize;
            let src_y = (y - dy).clamp(0, h_i - 1) as usize;

            let src_idx = (src_y * w as usize + src_x) * 3;
            let dst_idx = (y as usize * w as usize + x as usize) * 3;

            out[dst_idx] = prev[src_idx];
            out[dst_idx + 1] = prev[src_idx + 1];
            out[dst_idx + 2] = prev[src_idx + 2];
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(w: u32, h: u32, fill: u8) -> Vec<u8> {
        vec![fill; (3 * w * h) as usize]
    }

    #[test]
    fn conceal_missing_frame_copies_prev() {
        let prev = vec![42u8; 3 * 8 * 6];
        let out = conceal_missing_frame(&prev, 8, 6);
        assert_eq!(out, prev);
    }

    #[test]
    fn conceal_missing_frame_correct_length() {
        let prev = make_frame(16, 9, 100);
        let out = conceal_missing_frame(&prev, 16, 9);
        assert_eq!(out.len(), prev.len());
    }

    #[test]
    #[should_panic(expected = "3*w*h")]
    fn conceal_missing_frame_panics_on_wrong_size() {
        let _ = conceal_missing_frame(&[0u8; 10], 4, 4);
    }

    #[test]
    fn conceal_blend_alpha_one_returns_prev() {
        let prev = vec![200u8; 3 * 4 * 4];
        let out = conceal_blend(&prev, 4, 4, 1.0, [0, 0, 0]);
        for (&p, &o) in prev.iter().zip(out.iter()) {
            assert_eq!(p, o, "alpha=1.0 should return prev unchanged");
        }
    }

    #[test]
    fn conceal_blend_alpha_zero_returns_bg() {
        let prev = vec![255u8; 3 * 4 * 4];
        let bg = [128u8, 64, 32];
        let out = conceal_blend(&prev, 4, 4, 0.0, bg);
        for i in 0..(4 * 4) as usize {
            assert_eq!(out[i * 3], bg[0]);
            assert_eq!(out[i * 3 + 1], bg[1]);
            assert_eq!(out[i * 3 + 2], bg[2]);
        }
    }

    #[test]
    fn conceal_blend_half_alpha_midpoint() {
        let prev = vec![200u8; 3 * 2 * 2];
        let out = conceal_blend(&prev, 2, 2, 0.5, [0, 0, 0]);
        for &v in &out {
            assert_eq!(v, 100, "half blend of 200 and 0 should be 100");
        }
    }

    #[test]
    fn conceal_blend_all_values_in_range() {
        let prev: Vec<u8> = (0..3 * 8 * 8).map(|i| (i * 3 % 256) as u8).collect();
        let out = conceal_blend(&prev, 8, 8, 0.7, [128, 128, 128]);
        // All values are u8, so they are always in [0, 255].
        assert!(!out.is_empty());
    }

    #[test]
    fn conceal_motion_compensated_zero_offset_copies_prev() {
        let prev: Vec<u8> = (0..3 * 8 * 8).map(|i| (i % 256) as u8).collect();
        let out = conceal_motion_compensated(&prev, 8, 8, 0, 0);
        assert_eq!(out, prev);
    }

    #[test]
    fn conceal_motion_compensated_correct_length() {
        let prev = make_frame(10, 10, 50);
        let out = conceal_motion_compensated(&prev, 10, 10, 2, -1);
        assert_eq!(out.len(), prev.len());
    }

    #[test]
    fn conceal_motion_compensated_all_values_valid() {
        let prev = make_frame(8, 6, 77);
        let out = conceal_motion_compensated(&prev, 8, 6, 3, 2);
        // All values are u8, so they are always in [0, 255].
        assert!(!out.is_empty());
    }

    #[test]
    fn conceal_motion_compensated_shift_right_fills_left_edge() {
        // A frame with gradient 0..w-1 in the horizontal direction
        let w = 8u32;
        let h = 4u32;
        let mut prev = vec![0u8; (3 * w * h) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let v = x as u8;
                prev[(y * w as usize + x) * 3] = v;
                prev[(y * w as usize + x) * 3 + 1] = v;
                prev[(y * w as usize + x) * 3 + 2] = v;
            }
        }

        let out = conceal_motion_compensated(&prev, w, h, 2, 0);

        // After shifting right by 2, the leftmost 2 columns should be clamped
        // to the source x=0 value (0)
        assert_eq!(out[0], 0, "left-edge clamped pixel should be 0");
        assert_eq!(out[3], 0, "second-pixel clamped should be 0");
    }
}
