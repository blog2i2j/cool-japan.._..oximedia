//! Per-scene spatial complexity estimation.
//!
//! Measures the visual complexity of a set of frames by computing the
//! mean absolute deviation (MAD) of adjacent luma samples.  Complex scenes
//! (lots of texture, high contrast, busy motion) produce a high score;
//! uniform scenes (solid colours, smooth gradients) produce a low score.
//!
//! The algorithm is intentionally lightweight — no FFT, no DCT — so it
//! can be used as a fast pre-pass in an encoding pipeline to decide on
//! CRF / QP adjustments without introducing significant CPU overhead.
//!
//! # Score Range
//!
//! The return value is normalised to `[0.0, 1.0]` where:
//! - `0.0` — completely uniform frame (all pixels identical).
//! - `1.0` — maximally complex (all adjacent pixels differ by 255).

#![allow(dead_code)]

/// Computes the spatial complexity of a collection of frames.
///
/// Each frame in `frames` must be a row-major **luma-only** (Y plane) buffer
/// of exactly `w × h` bytes.  The function averages the complexity across all
/// provided frames.
///
/// # Arguments
///
/// * `frames` – slice of luma frame slices.
/// * `w` – frame width in pixels.
/// * `h` – frame height in pixels.
///
/// # Returns
///
/// Normalised complexity score in `[0.0, 1.0]`.
/// Returns `0.0` when `frames` is empty or when `w` or `h` is zero.
pub struct SceneComplexity;

impl SceneComplexity {
    /// Computes the average spatial complexity of `frames`.
    ///
    /// # Panics
    ///
    /// Does not panic; silently skips frames whose length does not equal
    /// `w * h`.
    #[must_use]
    pub fn compute(frames: &[&[u8]], w: u32, h: u32) -> f32 {
        if frames.is_empty() || w == 0 || h == 0 {
            return 0.0;
        }

        let expected = w as usize * h as usize;
        let mut total_score = 0.0f64;
        let mut valid_frames: usize = 0;

        for &frame in frames {
            if frame.len() != expected {
                continue;
            }
            let score = Self::frame_complexity(frame, w, h);
            total_score += score as f64;
            valid_frames += 1;
        }

        if valid_frames == 0 {
            return 0.0;
        }
        (total_score / valid_frames as f64) as f32
    }

    /// Computes the spatial complexity of a single luma frame.
    ///
    /// Returns a value in `[0.0, 1.0]`.
    #[must_use]
    pub fn frame_complexity(frame: &[u8], w: u32, h: u32) -> f32 {
        let width = w as usize;
        let height = h as usize;
        let expected = width * height;
        if frame.len() != expected || width < 2 || height < 2 {
            return 0.0;
        }

        // Horizontal MAD: sum of |pixel[x] - pixel[x-1]| for all rows.
        let mut h_mad: u64 = 0;
        for row in 0..height {
            let row_start = row * width;
            for col in 1..width {
                let a = frame[row_start + col - 1] as i32;
                let b = frame[row_start + col] as i32;
                h_mad += (a - b).unsigned_abs() as u64;
            }
        }

        // Vertical MAD: sum of |pixel[y] - pixel[y-1]| for all columns.
        let mut v_mad: u64 = 0;
        for col in 0..width {
            for row in 1..height {
                let a = frame[(row - 1) * width + col] as i32;
                let b = frame[row * width + col] as i32;
                v_mad += (a - b).unsigned_abs() as u64;
            }
        }

        // Total comparisons.
        let h_comparisons = (width - 1) * height;
        let v_comparisons = width * (height - 1);
        let total_comparisons = (h_comparisons + v_comparisons) as f64;

        if total_comparisons == 0.0 {
            return 0.0;
        }

        let total_mad = (h_mad + v_mad) as f64;
        // Each comparison can produce at most 255 difference.
        let normalised = total_mad / (total_comparisons * 255.0);
        normalised.min(1.0) as f32
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_frames_returns_zero() {
        assert_eq!(SceneComplexity::compute(&[], 8, 8), 0.0);
    }

    #[test]
    fn zero_dimensions_returns_zero() {
        let frame = vec![0u8; 64];
        assert_eq!(SceneComplexity::compute(&[&frame], 0, 8), 0.0);
        assert_eq!(SceneComplexity::compute(&[&frame], 8, 0), 0.0);
    }

    #[test]
    fn uniform_frame_zero_complexity() {
        let frame = vec![128u8; 4 * 4];
        let s = SceneComplexity::compute(&[&frame], 4, 4);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn checkerboard_high_complexity() {
        // 4×4 checkerboard: alternating 0 and 255.
        let frame: Vec<u8> = (0u8..16)
            .map(|i| if (i / 4 + i % 4) % 2 == 0 { 0 } else { 255 })
            .collect();
        let s = SceneComplexity::compute(&[&frame], 4, 4);
        assert!(s > 0.5, "checkerboard should have high complexity, got {s}");
    }

    #[test]
    fn wrong_frame_size_skipped() {
        let bad_frame = vec![0u8; 10]; // wrong size for 4×4
        let s = SceneComplexity::compute(&[&bad_frame], 4, 4);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn multiple_frames_averaged() {
        let uniform = vec![0u8; 16];
        let checker: Vec<u8> = (0u8..16)
            .map(|i| if i % 2 == 0 { 0 } else { 255 })
            .collect();
        let s_uniform = SceneComplexity::compute(&[&uniform], 4, 4);
        let s_checker = SceneComplexity::compute(&[&checker], 4, 4);
        let s_avg = SceneComplexity::compute(&[&uniform, &checker], 4, 4);
        // Average should be between the two extremes.
        assert!(s_avg > s_uniform);
        assert!(s_avg < s_checker || (s_avg - s_checker).abs() < 1e-5);
    }

    #[test]
    fn single_pixel_frame_returns_zero() {
        let frame = vec![100u8; 1];
        let s = SceneComplexity::compute(&[&frame], 1, 1);
        // 1×1 has no adjacent pairs → complexity = 0
        assert_eq!(s, 0.0);
    }
}
