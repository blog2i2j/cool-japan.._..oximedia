//! Composite video quality score combining sharpness and noise estimates.
//!
//! Provides a single `[0.0, 1.0]` quality score that blends:
//!
//! - **Sharpness** — the mean absolute Laplacian response, normalised so that
//!   a crisp synthetic edge approaches `1.0`.
//! - **Noise** — estimated as the inverse of a local variance ratio; noisy
//!   frames have high pixel-to-pixel variance that lowers the score.
//!
//! The composite score is a weighted blend (default weights: 60 % sharpness +
//! 40 % inverse-noise).  Higher is better.
//!
//! # Usage
//!
//! ```rust
//! use oximedia_video::quality_score::VideoQualityScore;
//!
//! let w = 8u32;
//! let h = 8u32;
//! let frame = vec![128u8; (w * h) as usize];
//! let score = VideoQualityScore::compute(&frame, w, h);
//! assert!((0.0..=1.0).contains(&score));
//! ```

#![allow(dead_code)]

/// Composite video quality scorer.
pub struct VideoQualityScore;

impl VideoQualityScore {
    /// Computes the composite quality score for a **luma-only** frame.
    ///
    /// # Arguments
    ///
    /// * `frame` – row-major luma plane (`w × h` bytes).
    /// * `w` – frame width in pixels.
    /// * `h` – frame height in pixels.
    ///
    /// # Returns
    ///
    /// A score in `[0.0, 1.0]`.  Returns `0.0` for invalid input.
    #[must_use]
    pub fn compute(frame: &[u8], w: u32, h: u32) -> f32 {
        if w < 3 || h < 3 {
            return 0.0;
        }
        let expected = w as usize * h as usize;
        if frame.len() != expected {
            return 0.0;
        }

        let sharpness = Self::sharpness_score(frame, w, h);
        let noise_penalty = Self::noise_penalty(frame, w, h);

        // Weighted blend penalised by noise: high noise reduces both sharpness
        // contribution and the base score, preventing noisy frames from scoring
        // higher than uniform ones due to spurious high Laplacian response.
        let quality = (0.6 * sharpness + 0.4) * (1.0 - noise_penalty);
        quality.clamp(0.0, 1.0)
    }

    /// Returns the normalised sharpness score in `[0.0, 1.0]`.
    ///
    /// Uses the discrete Laplacian operator to measure the mean absolute
    /// second derivative of the luma signal.  A high response indicates
    /// sharp edges.
    #[must_use]
    pub fn sharpness_score(frame: &[u8], w: u32, h: u32) -> f32 {
        let width = w as usize;
        let height = h as usize;
        if width < 3 || height < 3 {
            return 0.0;
        }
        let expected = width * height;
        if frame.len() != expected {
            return 0.0;
        }

        let mut lap_sum: f64 = 0.0;
        let mut count: usize = 0;

        // 3×3 Laplacian kernel: [-1,-1,-1; -1,8,-1; -1,-1,-1] applied at
        // every interior pixel.
        for row in 1..(height - 1) {
            for col in 1..(width - 1) {
                let c = frame[row * width + col] as f64;
                let n = frame[(row - 1) * width + col] as f64;
                let s = frame[(row + 1) * width + col] as f64;
                let e = frame[row * width + col + 1] as f64;
                let w_px = frame[row * width + col - 1] as f64;
                let nw = frame[(row - 1) * width + col - 1] as f64;
                let ne = frame[(row - 1) * width + col + 1] as f64;
                let sw = frame[(row + 1) * width + col - 1] as f64;
                let se = frame[(row + 1) * width + col + 1] as f64;

                let lap = 8.0 * c - (n + s + e + w_px + nw + ne + sw + se);
                lap_sum += lap.abs();
                count += 1;
            }
        }

        if count == 0 {
            return 0.0;
        }

        let mean_lap = lap_sum / count as f64;
        // Normalise: a fully synthetic black-to-white step (255) produces
        // a Laplacian response of ≈ 255 * 8 = 2040 at the edge.  We clamp
        // the mean at 255 to keep the score in [0, 1].
        (mean_lap / 255.0).min(1.0) as f32
    }

    /// Returns the noise penalty in `[0.0, 1.0]`.
    ///
    /// Estimates high-frequency noise by computing the mean local absolute
    /// deviation of each pixel from its 8-neighbours' average.  A high
    /// penalty indicates a noisy frame.
    #[must_use]
    pub fn noise_penalty(frame: &[u8], w: u32, h: u32) -> f32 {
        let width = w as usize;
        let height = h as usize;
        if width < 3 || height < 3 {
            return 0.0;
        }
        let expected = width * height;
        if frame.len() != expected {
            return 0.0;
        }

        let mut noise_sum: f64 = 0.0;
        let mut count: usize = 0;

        for row in 1..(height - 1) {
            for col in 1..(width - 1) {
                let c = frame[row * width + col] as f64;
                // 8-neighbour average.
                let n = frame[(row - 1) * width + col] as f64;
                let s = frame[(row + 1) * width + col] as f64;
                let e = frame[row * width + col + 1] as f64;
                let w_px = frame[row * width + col - 1] as f64;
                let nw = frame[(row - 1) * width + col - 1] as f64;
                let ne = frame[(row - 1) * width + col + 1] as f64;
                let sw = frame[(row + 1) * width + col - 1] as f64;
                let se = frame[(row + 1) * width + col + 1] as f64;
                let avg = (n + s + e + w_px + nw + ne + sw + se) / 8.0;
                noise_sum += (c - avg).abs();
                count += 1;
            }
        }

        if count == 0 {
            return 0.0;
        }

        let mean_noise = noise_sum / count as f64;
        // Clamp at 128 to keep in [0, 1].
        (mean_noise / 128.0).min(1.0) as f32
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn too_small_frame_returns_zero() {
        let frame = vec![0u8; 4];
        assert_eq!(VideoQualityScore::compute(&frame, 2, 2), 0.0);
    }

    #[test]
    fn wrong_size_returns_zero() {
        let frame = vec![0u8; 10];
        assert_eq!(VideoQualityScore::compute(&frame, 8, 8), 0.0);
    }

    #[test]
    fn uniform_frame_score_in_range() {
        let frame = vec![128u8; 64];
        let s = VideoQualityScore::compute(&frame, 8, 8);
        assert!((0.0..=1.0).contains(&s));
    }

    #[test]
    fn uniform_frame_low_lap_high_quality() {
        // Uniform frame has zero Laplacian → sharpness=0, noise_penalty≈0.
        // quality = 0.6*0 + 0.4*(1-0) = 0.4
        let frame = vec![100u8; 64];
        let s = VideoQualityScore::compute(&frame, 8, 8);
        assert!((s - 0.4).abs() < 0.05, "got {s}");
    }

    #[test]
    fn noisy_frame_lower_quality() {
        // Random-ish noise: alternating 0/255
        let frame: Vec<u8> = (0..64)
            .map(|i: usize| if i % 2 == 0 { 0 } else { 255 })
            .collect();
        let noisy = VideoQualityScore::compute(&frame, 8, 8);
        let uniform = VideoQualityScore::compute(&vec![128u8; 64], 8, 8);
        // Noisy should not be higher quality than uniform (noise penalty applies).
        assert!(noisy <= uniform + 0.1, "noisy={noisy}, uniform={uniform}");
    }

    #[test]
    fn output_always_clamped_to_unit_range() {
        let frames: Vec<Vec<u8>> = vec![vec![0u8; 64], vec![255u8; 64], (0..64u8).collect()];
        for f in &frames {
            let s = VideoQualityScore::compute(f, 8, 8);
            assert!((0.0..=1.0).contains(&s), "score out of range: {s}");
        }
    }
}
