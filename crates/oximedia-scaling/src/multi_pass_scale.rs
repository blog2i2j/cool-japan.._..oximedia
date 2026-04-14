//! Multi-pass scaling for extreme scaling ratios.
//!
//! When scaling by a very large factor (e.g., 8K → 480p, a ratio of 16×) a
//! single-pass resampler may introduce aliasing or ringing artefacts.  The
//! standard mitigation is to break the scaling into multiple intermediate
//! steps, each with a ratio of at most some maximum factor.
//!
//! `MultiPassScaler` computes the sequence of intermediate resolutions
//! between the source and destination, each step reducing (or enlarging) by
//! at most `max_ratio`.  The actual per-step pixel resampling is performed by
//! nearest-neighbour interpolation so the module is self-contained.
//!
//! # Example
//!
//! ```
//! use oximedia_scaling::multi_pass_scale::MultiPassScaler;
//!
//! let scaler = MultiPassScaler::new(2.0); // max 2× per step
//! // 8 → 1: requires multiple steps (8→4→2→1)
//! let steps = scaler.compute_steps(8, 1);
//! assert!(steps.len() >= 3);
//! // Final step must reach the target.
//! assert_eq!(*steps.last().unwrap(), 1);
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

const CHANNELS: usize = 4;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Multi-pass image scaler that limits the per-step scale ratio.
#[derive(Debug, Clone)]
pub struct MultiPassScaler {
    /// Maximum scale ratio allowed between two consecutive passes.
    /// Must be > 1.0; values < 1.0 are clamped to 1.01.
    pub max_ratio: f64,
}

impl MultiPassScaler {
    /// Create a new `MultiPassScaler` with the specified maximum per-step ratio.
    pub fn new(max_ratio: f64) -> Self {
        Self {
            max_ratio: max_ratio.max(1.01),
        }
    }

    /// Compute the sequence of intermediate widths from `src` to `dst`.
    ///
    /// The returned `Vec` contains every intermediate width in order,
    /// **including** the final `dst` value.  The source width is *not*
    /// included (it is the caller's starting point).
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_scaling::multi_pass_scale::MultiPassScaler;
    ///
    /// let s = MultiPassScaler::new(2.0);
    /// let steps = s.compute_steps(32, 4);
    /// assert_eq!(*steps.last().unwrap(), 4);
    /// ```
    pub fn compute_steps(&self, src: u32, dst: u32) -> Vec<u32> {
        if src == dst || src == 0 || dst == 0 {
            return vec![dst];
        }

        let mut steps = Vec::new();
        let downscale = dst < src;
        let mut current = src as f64;
        let target = dst as f64;

        loop {
            let ratio = if downscale {
                current / target
            } else {
                target / current
            };

            if ratio <= self.max_ratio {
                // Close enough — jump directly to target.
                steps.push(dst);
                break;
            }

            // Take one step by max_ratio.
            let next = if downscale {
                (current / self.max_ratio).round().max(target).max(1.0)
            } else {
                (current * self.max_ratio).round().min(target)
            };

            let next_u = next as u32;
            steps.push(next_u);
            current = next;

            // Guard against infinite loops due to floating-point precision.
            if next_u == dst {
                break;
            }
            if steps.len() > 64 {
                steps.push(dst);
                break;
            }
        }

        steps
    }

    /// Scale an RGBA image (`4 bytes per pixel`) from `(src_w, src_h)` to
    /// `(dst_w, dst_h)` using multiple nearest-neighbour passes.
    ///
    /// Each pass reduces (or enlarges) by at most `max_ratio` in each
    /// dimension independently.
    ///
    /// # Returns
    ///
    /// An RGBA pixel buffer of length `dst_w × dst_h × 4`.
    pub fn scale(&self, src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
        if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
            return vec![0u8; (dst_w * dst_h) as usize * CHANNELS];
        }

        let w_steps = self.compute_steps(src_w, dst_w);
        let h_steps = self.compute_steps(src_h, dst_h);
        let max_steps = w_steps.len().max(h_steps.len());

        let mut current_pixels = src.to_vec();
        let mut current_w = src_w;
        let mut current_h = src_h;

        for i in 0..max_steps {
            let next_w = w_steps.get(i).copied().unwrap_or(dst_w);
            let next_h = h_steps.get(i).copied().unwrap_or(dst_h);

            current_pixels = nn_scale_rgba(&current_pixels, current_w, current_h, next_w, next_h);
            current_w = next_w;
            current_h = next_h;
        }

        current_pixels
    }
}

/// Nearest-neighbour scale of an RGBA buffer from `(sw, sh)` to `(dw, dh)`.
fn nn_scale_rgba(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    let sw = sw as usize;
    let sh = sh as usize;
    let dw_u = dw as usize;
    let dh_u = dh as usize;
    let mut out = vec![0u8; dw_u * dh_u * CHANNELS];

    for dy in 0..dh_u {
        let sy = (dy * sh / dh_u).min(sh - 1);
        for dx in 0..dw_u {
            let sx = (dx * sw / dw_u).min(sw - 1);
            let src_off = (sy * sw + sx) * CHANNELS;
            let dst_off = (dy * dw_u + dx) * CHANNELS;
            if src_off + CHANNELS <= src.len() {
                out[dst_off..dst_off + CHANNELS].copy_from_slice(&src[src_off..src_off + CHANNELS]);
            }
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

    #[test]
    fn test_compute_steps_same_size() {
        let s = MultiPassScaler::new(2.0);
        let steps = s.compute_steps(100, 100);
        assert_eq!(steps, vec![100]);
    }

    #[test]
    fn test_compute_steps_downscale_terminates_at_dst() {
        let s = MultiPassScaler::new(2.0);
        let steps = s.compute_steps(32, 4);
        assert_eq!(*steps.last().unwrap(), 4);
    }

    #[test]
    fn test_compute_steps_upscale_terminates_at_dst() {
        let s = MultiPassScaler::new(2.0);
        let steps = s.compute_steps(4, 32);
        assert_eq!(*steps.last().unwrap(), 32);
    }

    #[test]
    fn test_scale_output_size() {
        let scaler = MultiPassScaler::new(2.0);
        let src = vec![128u8; 64 * 64 * 4];
        let out = scaler.scale(&src, 64, 64, 8, 8);
        assert_eq!(out.len(), 8 * 8 * 4);
    }

    #[test]
    fn test_scale_identity() {
        let scaler = MultiPassScaler::new(2.0);
        let src = vec![200u8; 8 * 8 * 4];
        let out = scaler.scale(&src, 8, 8, 8, 8);
        assert_eq!(out, src);
    }

    #[test]
    fn test_scale_zero_dst_returns_empty_buffer() {
        let scaler = MultiPassScaler::new(2.0);
        let src = vec![0u8; 16 * 16 * 4];
        let out = scaler.scale(&src, 16, 16, 0, 8);
        assert!(out.is_empty());
    }

    #[test]
    fn test_compute_steps_large_downscale_uses_multiple_steps() {
        let s = MultiPassScaler::new(2.0);
        // 8K to 480p: factor of ~16, so should need >=4 steps
        let steps = s.compute_steps(7680, 480);
        assert!(
            steps.len() >= 3,
            "expected multiple steps for 16x downscale, got {}",
            steps.len()
        );
        assert_eq!(*steps.last().unwrap_or(&0), 480);
    }

    #[test]
    fn test_compute_steps_min_ratio_clamped() {
        // max_ratio below 1.0 should be clamped to 1.01
        let s = MultiPassScaler::new(0.5);
        assert!(s.max_ratio > 1.0);
        let steps = s.compute_steps(10, 5);
        assert_eq!(*steps.last().unwrap_or(&0), 5);
    }

    #[test]
    fn test_scale_upscale_output_size() {
        let scaler = MultiPassScaler::new(2.0);
        let src = vec![64u8; 4 * 4 * 4];
        let out = scaler.scale(&src, 4, 4, 32, 32);
        assert_eq!(out.len(), 32 * 32 * 4);
    }

    #[test]
    fn test_scale_pixel_value_preserved_on_uniform_image() {
        // A uniform image should produce a uniform output at any scale.
        let scaler = MultiPassScaler::new(2.0);
        let fill = 123u8;
        let src = vec![fill; 8 * 8 * 4];
        let out = scaler.scale(&src, 8, 8, 2, 2);
        assert_eq!(out.len(), 2 * 2 * 4);
        for &byte in &out {
            assert_eq!(byte, fill);
        }
    }

    #[test]
    fn test_compute_steps_single_step_when_within_ratio() {
        // src=10, dst=8 — ratio is 1.25 which is < 2.0 so a single jump.
        let s = MultiPassScaler::new(2.0);
        let steps = s.compute_steps(10, 8);
        assert_eq!(steps, vec![8]);
    }

    #[test]
    fn test_scale_zero_src_returns_sized_buffer() {
        // src_w=0 — function should return a zero-filled buffer of dst size.
        let scaler = MultiPassScaler::new(2.0);
        let out = scaler.scale(&[], 0, 4, 8, 4);
        assert_eq!(out.len(), 8 * 4 * 4);
        for &b in &out {
            assert_eq!(b, 0);
        }
    }
}
