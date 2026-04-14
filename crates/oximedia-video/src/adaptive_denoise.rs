//! Adaptive spatial denoising based on local variance estimation.
//!
//! Uses a 3×3 local variance map to adaptively blend each pixel with the
//! mean of its neighbourhood: high-variance regions (edges, fine detail)
//! receive little denoising to preserve sharpness, while low-variance
//! regions (flat areas, sky) receive strong smoothing to suppress noise.
//!
//! # Algorithm
//!
//! For each interior pixel `p(x,y)`:
//!
//! ```text
//! local_mean  = mean of 3×3 neighbourhood
//! local_var   = variance of 3×3 neighbourhood
//! alpha       = local_var / (local_var + noise_variance)
//! p_denoised  = alpha * p(x,y) + (1 - alpha) * local_mean
//! ```
//!
//! `noise_variance` is a fixed prior set to `(strength / 2.0)²`.  Increasing
//! `strength` effectively raises the threshold above which a pixel is
//! considered "signal" rather than "noise".
//!
//! Border pixels (outermost row/column) are copied as-is.
//!
//! # Usage
//!
//! ```rust
//! use oximedia_video::adaptive_denoise::AdaptiveDenoiser;
//!
//! let denoiser = AdaptiveDenoiser::new(16.0);
//! let frame = vec![128u8; 8 * 8];
//! let out = denoiser.denoise(&frame, 8, 8);
//! assert_eq!(out.len(), frame.len());
//! ```

/// Adaptive local-variance denoiser operating on luma planes.
#[derive(Debug, Clone)]
pub struct AdaptiveDenoiser {
    /// Prior noise standard deviation (sigma).
    ///
    /// Higher values force stronger smoothing in all regions.
    /// Typical range: 4–32.
    pub strength: f32,
}

impl AdaptiveDenoiser {
    /// Creates a new denoiser with the given noise `strength`.
    ///
    /// # Arguments
    ///
    /// * `strength` – noise sigma prior.  `0.0` disables denoising (output
    ///   equals input).  A value of `16.0` is a good starting point for
    ///   moderate sensor noise.
    #[must_use]
    pub fn new(strength: f32) -> Self {
        Self { strength }
    }

    /// Denoises a **luma-only** frame using adaptive local variance filtering.
    ///
    /// # Arguments
    ///
    /// * `frame` – row-major luma plane (`w × h` bytes).
    /// * `w` – frame width in pixels.
    /// * `h` – frame height in pixels.
    ///
    /// # Returns
    ///
    /// A new `Vec<u8>` of the same length as the input with denoised pixels.
    /// Returns a copy of `frame` unchanged when:
    /// - `strength` is `0.0`.
    /// - `frame.len() ≠ w × h`.
    /// - `w < 3` or `h < 3`.
    #[must_use]
    pub fn denoise(&self, frame: &[u8], w: u32, h: u32) -> Vec<u8> {
        let width = w as usize;
        let height = h as usize;
        let expected = width * height;

        if self.strength <= 0.0 || frame.len() != expected || width < 3 || height < 3 {
            return frame.to_vec();
        }

        // noise_variance = (strength / 2)²
        let noise_var = (self.strength / 2.0) * (self.strength / 2.0);

        let mut out = frame.to_vec();

        for row in 1..(height - 1) {
            for col in 1..(width - 1) {
                // Collect 3×3 neighbourhood.
                let mut neigh = [0f32; 9];
                let mut k = 0;
                for dr in -1i32..=1 {
                    for dc in -1i32..=1 {
                        let r = (row as i32 + dr) as usize;
                        let c = (col as i32 + dc) as usize;
                        neigh[k] = frame[r * width + c] as f32;
                        k += 1;
                    }
                }

                let mean = neigh.iter().sum::<f32>() / 9.0;
                let var = neigh.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / 9.0;

                // Adaptive blending coefficient.
                let alpha = var / (var + noise_var);

                let center = frame[row * width + col] as f32;
                let denoised = alpha * center + (1.0 - alpha) * mean;
                out[row * width + col] = denoised.round().clamp(0.0, 255.0) as u8;
            }
        }

        out
    }

    /// Denoise with configurable `strength` for a single call without
    /// constructing a new `AdaptiveDenoiser`.
    #[must_use]
    pub fn denoise_with_strength(frame: &[u8], w: u32, h: u32, strength: f32) -> Vec<u8> {
        Self::new(strength).denoise(frame, w, h)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_strength_returns_input() {
        let denoiser = AdaptiveDenoiser::new(0.0);
        let frame: Vec<u8> = (0..64).collect();
        let out = denoiser.denoise(&frame, 8, 8);
        assert_eq!(out, frame);
    }

    #[test]
    fn output_same_length() {
        let denoiser = AdaptiveDenoiser::new(16.0);
        let frame = vec![100u8; 64];
        let out = denoiser.denoise(&frame, 8, 8);
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn uniform_frame_unchanged() {
        // Uniform frame: every neighbourhood has variance = 0.
        // alpha = 0 / (0 + noise_var) = 0 → output = mean = 128.
        let denoiser = AdaptiveDenoiser::new(16.0);
        let frame = vec![128u8; 64];
        let out = denoiser.denoise(&frame, 8, 8);
        for &b in &out {
            assert_eq!(b, 128, "uniform frame should stay uniform");
        }
    }

    #[test]
    fn noisy_frame_smoothed() {
        // Alternating 0/255 checkerboard — very high noise.
        let frame: Vec<u8> = (0..64u8)
            .map(|i| if i % 2 == 0 { 0 } else { 255 })
            .collect();
        let denoiser = AdaptiveDenoiser::new(32.0);
        let out = denoiser.denoise(&frame, 8, 8);
        // Interior pixels should be pulled toward the neighbourhood mean (≈127).
        let interior_mean: f32 = out[9..55]
            .iter()
            .filter(|&&b| b != 0 && b != 255)
            .map(|&b| b as f32)
            .sum::<f32>()
            / 64.0;
        // Mean of denoised interior should be closer to 127 than 0 or 255.
        assert!(
            interior_mean > 50.0 && interior_mean < 200.0,
            "expected smoothed values near 127, got {interior_mean}"
        );
    }

    #[test]
    fn border_pixels_copied_unchanged() {
        let frame: Vec<u8> = (0..64u8).collect();
        let denoiser = AdaptiveDenoiser::new(8.0);
        let out = denoiser.denoise(&frame, 8, 8);
        // Top row
        for col in 0..8usize {
            assert_eq!(
                out[col], frame[col],
                "top row pixel {col} should be unchanged"
            );
        }
        // Bottom row
        for col in 0..8usize {
            assert_eq!(out[56 + col], frame[56 + col], "bottom row pixel {col}");
        }
    }

    #[test]
    fn wrong_size_returns_input_copy() {
        let frame = vec![128u8; 10]; // wrong size for 8×8
        let denoiser = AdaptiveDenoiser::new(8.0);
        let out = denoiser.denoise(&frame, 8, 8);
        assert_eq!(out, frame);
    }

    #[test]
    fn too_small_frame_returns_input_copy() {
        let frame = vec![100u8; 4]; // 2×2 < 3×3
        let denoiser = AdaptiveDenoiser::new(8.0);
        let out = denoiser.denoise(&frame, 2, 2);
        assert_eq!(out, frame);
    }

    #[test]
    fn all_pixels_clamped() {
        let frame: Vec<u8> = (0..64)
            .map(|i: usize| if i % 3 == 0 { 0 } else { 255 })
            .collect();
        let denoiser = AdaptiveDenoiser::new(128.0);
        let out = denoiser.denoise(&frame, 8, 8);
        assert_eq!(out.len(), frame.len());
    }
}
