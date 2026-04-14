//! Motion-compensated frame interpolation.
//!
//! Given two adjacent frames and a dense optical flow field, synthesises
//! an intermediate frame at fractional temporal position `t ∈ [0, 1]`.
//!
//! # Algorithm
//!
//! For each pixel `(x, y)` in the output frame:
//! 1. Look up the flow vector `(fx, fy)` at `(x, y)`.
//! 2. Sample from the previous frame at `(x - t*fx, y - t*fy)`.
//! 3. Sample from the next frame at `(x + (1-t)*fx, y + (1-t)*fy)`.
//! 4. Blend: `out[p] = t * next_sample + (1-t) * prev_sample`.
//!
//! Fractional positions are interpolated using nearest-neighbour sampling
//! (bilinear interpolation is a straightforward extension but NN is
//! sufficient for the performance-oriented stub here).

#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

/// Synthesise an intermediate grayscale frame between `prev` and `next`
/// at fractional position `t`.
///
/// # Arguments
///
/// * `prev` – previous frame pixel data (grayscale, 1 byte per pixel, row-major).
/// * `next` – next frame pixel data (same format and dimensions).
/// * `flow` – dense optical flow, one `(fx, fy)` vector per pixel
///            (same width × height layout as the frames).
/// * `t`    – temporal interpolation weight in `[0, 1]`.  `t = 0` returns
///            `prev`, `t = 1` returns `next`.
/// * `w`    – frame width in pixels.
/// * `h`    – frame height in pixels.
///
/// Returns a `Vec<u8>` of length `w * h` with the interpolated frame.
/// Returns an empty vector if input sizes are inconsistent.
pub struct MotionInterpolator;

impl MotionInterpolator {
    /// Create a new `MotionInterpolator`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Interpolate a frame between `prev` and `next` at temporal position `t`.
    ///
    /// See the module-level documentation for the algorithm description.
    ///
    /// If `prev`, `next`, or `flow` have a length that does not match `w * h`
    /// the function returns an empty vector.
    #[must_use]
    pub fn interpolate(
        &self,
        prev: &[u8],
        next: &[u8],
        flow: &[(f32, f32)],
        t: f32,
        w: u32,
        h: u32,
    ) -> Vec<u8> {
        interpolate(prev, next, flow, t, w, h)
    }
}

impl Default for MotionInterpolator {
    fn default() -> Self {
        Self::new()
    }
}

/// Free-standing interpolation function (used by both the struct and tests).
#[must_use]
pub fn interpolate(
    prev: &[u8],
    next: &[u8],
    flow: &[(f32, f32)],
    t: f32,
    w: u32,
    h: u32,
) -> Vec<u8> {
    let n = (w as usize) * (h as usize);
    if prev.len() != n || next.len() != n || flow.len() != n {
        return Vec::new();
    }

    let t = t.clamp(0.0, 1.0);
    let one_minus_t = 1.0 - t;
    let w_usize = w as usize;
    let h_usize = h as usize;

    let mut out = vec![0u8; n];

    for y in 0..h_usize {
        for x in 0..w_usize {
            let idx = y * w_usize + x;
            let (fx, fy) = flow[idx];

            // Sample previous frame at (x - t*fx, y - t*fy)
            let prev_x = x as f32 - t * fx;
            let prev_y = y as f32 - t * fy;
            let prev_val = sample_nearest(prev, w_usize, h_usize, prev_x, prev_y);

            // Sample next frame at (x + (1-t)*fx, y + (1-t)*fy)
            let next_x = x as f32 + one_minus_t * fx;
            let next_y = y as f32 + one_minus_t * fy;
            let next_val = sample_nearest(next, w_usize, h_usize, next_x, next_y);

            // Blend: out = (1-t)*prev + t*next
            let blended = one_minus_t * prev_val as f32 + t * next_val as f32;
            out[idx] = blended.round().clamp(0.0, 255.0) as u8;
        }
    }

    out
}

/// Sample a frame at fractional position `(fx, fy)` using nearest-neighbour.
fn sample_nearest(frame: &[u8], w: usize, h: usize, fx: f32, fy: f32) -> u8 {
    // Clamp to valid range
    let xi = (fx.round() as isize).clamp(0, w as isize - 1) as usize;
    let yi = (fy.round() as isize).clamp(0, h as isize - 1) as usize;
    frame[yi * w + xi]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(w: usize, h: usize, value: u8) -> Vec<u8> {
        vec![value; w * h]
    }

    fn zero_flow(n: usize) -> Vec<(f32, f32)> {
        vec![(0.0, 0.0); n]
    }

    #[test]
    fn test_interpolate_t0_returns_prev() {
        let prev = make_frame(4, 4, 50);
        let next = make_frame(4, 4, 200);
        let flow = zero_flow(16);
        let out = interpolate(&prev, &next, &flow, 0.0, 4, 4);
        assert_eq!(out.len(), 16);
        assert!(out.iter().all(|&v| v == 50), "t=0 should return prev");
    }

    #[test]
    fn test_interpolate_t1_returns_next() {
        let prev = make_frame(4, 4, 0);
        let next = make_frame(4, 4, 200);
        let flow = zero_flow(16);
        let out = interpolate(&prev, &next, &flow, 1.0, 4, 4);
        assert_eq!(out.len(), 16);
        assert!(out.iter().all(|&v| v == 200), "t=1 should return next");
    }

    #[test]
    fn test_interpolate_midpoint() {
        let prev = make_frame(4, 4, 100);
        let next = make_frame(4, 4, 200);
        let flow = zero_flow(16);
        let out = interpolate(&prev, &next, &flow, 0.5, 4, 4);
        for &v in &out {
            assert!(
                (v as i32 - 150).abs() <= 1,
                "midpoint blend should be ~150, got {v}"
            );
        }
    }

    #[test]
    fn test_interpolate_wrong_size_returns_empty() {
        let prev = vec![0u8; 10];
        let next = vec![0u8; 16]; // wrong size
        let flow = zero_flow(16);
        let out = interpolate(&prev, &next, &flow, 0.5, 4, 4);
        assert!(out.is_empty());
    }

    #[test]
    fn test_motion_interpolator_struct() {
        let interp = MotionInterpolator::new();
        let prev = make_frame(4, 4, 60);
        let next = make_frame(4, 4, 120);
        let flow = zero_flow(16);
        let out = interp.interpolate(&prev, &next, &flow, 0.5, 4, 4);
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn test_interpolate_with_flow_no_panic() {
        let w = 8usize;
        let h = 8usize;
        let prev: Vec<u8> = (0..w * h).map(|i| (i % 256) as u8).collect();
        let next: Vec<u8> = (0..w * h).map(|i| ((i + 128) % 256) as u8).collect();
        let flow: Vec<(f32, f32)> = (0..w * h)
            .map(|i| (((i % 5) as f32) - 2.0, ((i % 3) as f32) - 1.0))
            .collect();
        let out = interpolate(&prev, &next, &flow, 0.3, w as u32, h as u32);
        assert_eq!(out.len(), w * h);
    }
}
