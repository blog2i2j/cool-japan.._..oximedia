//! Stereo widener using mid-side (M/S) processing.
//!
//! The [`StereoWidener`] in this module is a lightweight API designed around
//! the task specification: `new(width: f32)` + `process(l, r) -> (Vec<f32>, Vec<f32>)`.
//! It is a complement to the richer [`crate::stereo_widener`] module which
//! exposes multiple widening modes.
//!
//! # Algorithm
//!
//! Mid-side processing:
//!
//! ```text
//! M = (L + R) / 2     (mono sum)
//! S = (L - R) / 2     (stereo difference)
//!
//! S' = S * width      (scale side channel by width factor)
//!
//! L' = M + S'
//! R' = M - S'
//! ```
//!
//! - `width = 0.0` → mono (M only, S = 0)
//! - `width = 1.0` → original stereo
//! - `width > 1.0` → wider than original (side boosted)
//! - `width < 0.0` → phase-inverted side (stereo reversal)
//!
//! # Example
//!
//! ```
//! use oximedia_effects::stereo_wider::StereoWidener;
//!
//! let mut w = StereoWidener::new(1.5);
//! let l = vec![0.4_f32, -0.2, 0.6];
//! let r = vec![-0.1_f32, 0.3, -0.5];
//! let (l_out, r_out) = w.process(&l, &r);
//! assert_eq!(l_out.len(), 3);
//! assert_eq!(r_out.len(), 3);
//! for (&lo, &ro) in l_out.iter().zip(r_out.iter()) {
//!     assert!(lo.is_finite());
//!     assert!(ro.is_finite());
//! }
//! ```

#![allow(dead_code)]

/// Stereo widener using mid-side (M/S) processing.
///
/// # Width values
///
/// | `width` | Effect                               |
/// |---------|--------------------------------------|
/// | `0.0`   | Full mono (S channel = 0)            |
/// | `1.0`   | Original stereo (identity)           |
/// | `2.0`   | Double stereo width                  |
/// | `-1.0`  | Inverted stereo (L/R channels swap)  |
pub struct StereoWidener {
    width: f32,
    /// Smoothed width for zipper-noise-free parameter changes.
    smooth_width: f32,
    /// One-pole smoothing coefficient (~10 ms at 48 kHz).
    smooth_coeff: f32,
    sample_rate: f32,
}

impl StereoWidener {
    /// Create a new stereo widener.
    ///
    /// `width` is the side-channel scaling factor:
    /// - `1.0` = original stereo
    /// - `0.0` = full mono
    /// - `> 1.0` = wider than original
    #[must_use]
    pub fn new(width: f32) -> Self {
        const DEFAULT_SR: f32 = 48_000.0;
        let smooth_coeff = Self::make_smooth_coeff(DEFAULT_SR);
        Self {
            width,
            smooth_width: width,
            smooth_coeff,
            sample_rate: DEFAULT_SR,
        }
    }

    /// Create a stereo widener with an explicit sample rate for smoother
    /// parameter interpolation.
    #[must_use]
    pub fn with_sample_rate(width: f32, sample_rate: f32) -> Self {
        let smooth_coeff = Self::make_smooth_coeff(sample_rate);
        Self {
            width,
            smooth_width: width,
            smooth_coeff,
            sample_rate,
        }
    }

    fn make_smooth_coeff(sample_rate: f32) -> f32 {
        // 10 ms one-pole IIR smoothing
        (-1.0_f32 / (0.010 * sample_rate.max(1.0))).exp()
    }

    /// Set a new width value. The change is applied smoothly over ~10 ms.
    pub fn set_width(&mut self, width: f32) {
        self.width = width;
    }

    /// Return the current (target) width setting.
    #[must_use]
    pub fn width(&self) -> f32 {
        self.width
    }

    /// Update the sample rate and recompute the smoothing coefficient.
    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.smooth_coeff = Self::make_smooth_coeff(sample_rate);
    }

    /// Process a single stereo sample pair.
    ///
    /// Returns `(left_out, right_out)`.
    pub fn process_sample(&mut self, left: f32, right: f32) -> (f32, f32) {
        // Smooth the width parameter.
        self.smooth_width =
            self.smooth_width * self.smooth_coeff + self.width * (1.0 - self.smooth_coeff);

        let mid = (left + right) * 0.5;
        let side = (left - right) * 0.5;
        let side_out = side * self.smooth_width;

        (mid + side_out, mid - side_out)
    }

    /// Process stereo buffers, returning `(left_out, right_out)`.
    ///
    /// The output length equals `left.len().min(right.len())`.
    #[must_use]
    pub fn process(&mut self, left: &[f32], right: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let len = left.len().min(right.len());
        let mut l_out = Vec::with_capacity(len);
        let mut r_out = Vec::with_capacity(len);
        for i in 0..len {
            let (l, r) = self.process_sample(left[i], right[i]);
            l_out.push(l);
            r_out.push(r);
        }
        (l_out, r_out)
    }

    /// Reset the parameter smoother to the current target width.
    pub fn reset(&mut self) {
        self.smooth_width = self.width;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mono_width_zero_collapses_to_center() {
        let mut w = StereoWidener::new(0.0);
        // Skip smoother settling.
        w.smooth_width = 0.0;
        let l = vec![0.6_f32, -0.4, 0.8];
        let r = vec![-0.2_f32, 0.3, -0.6];
        let (l_out, r_out) = w.process(&l, &r);
        for (&lo, &ro) in l_out.iter().zip(r_out.iter()) {
            assert!(
                (lo - ro).abs() < 1e-5,
                "width=0 should produce mono: L={lo}, R={ro}"
            );
        }
    }

    #[test]
    fn test_identity_at_width_one() {
        let mut w = StereoWidener::new(1.0);
        w.smooth_width = 1.0;
        let l = vec![0.5_f32, -0.3, 0.7];
        let r = vec![-0.1_f32, 0.2, -0.4];
        let (l_out, r_out) = w.process(&l, &r);
        for (i, (&li, &lo)) in l.iter().zip(l_out.iter()).enumerate() {
            assert!(
                (li - lo).abs() < 1e-5,
                "width=1 should be identity for L at {i}: in={li}, out={lo}"
            );
        }
        for (i, (&ri, &ro)) in r.iter().zip(r_out.iter()).enumerate() {
            assert!(
                (ri - ro).abs() < 1e-5,
                "width=1 should be identity for R at {i}: in={ri}, out={ro}"
            );
        }
    }

    #[test]
    fn test_output_length_is_min_of_inputs() {
        let mut w = StereoWidener::new(1.2);
        let l = vec![0.0_f32; 8];
        let r = vec![0.0_f32; 5];
        let (lo, ro) = w.process(&l, &r);
        assert_eq!(lo.len(), 5);
        assert_eq!(ro.len(), 5);
    }

    #[test]
    fn test_all_outputs_finite() {
        let mut w = StereoWidener::new(2.0);
        let l: Vec<f32> = (0..512).map(|i| (i as f32 * 0.02).sin()).collect();
        let r: Vec<f32> = (0..512).map(|i| (i as f32 * 0.03).cos()).collect();
        let (lo, ro) = w.process(&l, &r);
        for (i, (&lv, &rv)) in lo.iter().zip(ro.iter()).enumerate() {
            assert!(lv.is_finite(), "L sample {i} is not finite: {lv}");
            assert!(rv.is_finite(), "R sample {i} is not finite: {rv}");
        }
    }

    #[test]
    fn test_stereo_inversion_at_minus_one() {
        // width = -1 should swap L and R channels.
        let mut w = StereoWidener::new(-1.0);
        w.smooth_width = -1.0;
        let l = vec![0.6_f32, -0.4];
        let r = vec![-0.2_f32, 0.3];
        let (lo, ro) = w.process(&l, &r);
        // L_out = M + S * (-1) = M - S = R_in
        // R_out = M - S * (-1) = M + S = L_in
        for (i, (&li, &ro_v)) in l.iter().zip(ro.iter()).enumerate() {
            assert!(
                (li - ro_v).abs() < 1e-5,
                "width=-1: L_in should equal R_out at {i}: L_in={li}, R_out={ro_v}"
            );
        }
        for (i, (&ri, &lo_v)) in r.iter().zip(lo.iter()).enumerate() {
            assert!(
                (ri - lo_v).abs() < 1e-5,
                "width=-1: R_in should equal L_out at {i}: R_in={ri}, L_out={lo_v}"
            );
        }
    }

    #[test]
    fn test_set_width_changes_target() {
        let mut w = StereoWidener::new(1.0);
        w.set_width(2.0);
        assert!((w.width() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_reset_snaps_smoother() {
        let mut w = StereoWidener::new(1.5);
        w.smooth_width = 0.0;
        w.reset();
        assert!((w.smooth_width - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_wider_increases_side_energy() {
        // Processing the same stereo pair at width=2 should give more side energy
        // than at width=1.
        let n = 512;
        let l: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).sin()).collect();
        let r: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).cos()).collect();

        let mut w1 = StereoWidener::new(1.0);
        w1.smooth_width = 1.0;
        let (l1, r1) = w1.process(&l, &r);

        let mut w2 = StereoWidener::new(2.0);
        w2.smooth_width = 2.0;
        let (l2, r2) = w2.process(&l, &r);

        // Measure side energy: S = (L - R) / 2.
        let side1: f32 = l1
            .iter()
            .zip(r1.iter())
            .map(|(&a, &b)| {
                let s = (a - b) / 2.0;
                s * s
            })
            .sum::<f32>();
        let side2: f32 = l2
            .iter()
            .zip(r2.iter())
            .map(|(&a, &b)| {
                let s = (a - b) / 2.0;
                s * s
            })
            .sum::<f32>();

        assert!(
            side2 > side1,
            "width=2 should have more side energy than width=1: {side2:.4} vs {side1:.4}"
        );
    }
}
