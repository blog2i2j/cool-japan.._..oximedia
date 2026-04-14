//! Bitstream quality statistics.
//!
//! [`BitstreamStats`] accumulates per-frame quality metrics such as PSNR and
//! SSIM that can be computed between the original (uncompressed) and
//! reconstructed (compressed/decompressed) pixel buffers.
//!
//! These metrics are commonly used to evaluate codec quality and to drive
//! rate-distortion optimization loops.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

// ---------------------------------------------------------------------------
// BitstreamStats
// ---------------------------------------------------------------------------

/// Accumulated bitstream quality statistics.
///
/// Call [`BitstreamStats::update_psnr`] and/or [`BitstreamStats::update_ssim`]
/// after each encoded frame to accumulate per-frame metrics.  Aggregated
/// averages can then be retrieved via [`BitstreamStats::mean_psnr`] and
/// [`BitstreamStats::mean_ssim`].
#[derive(Debug, Clone, Default)]
pub struct BitstreamStats {
    /// Running sum of per-frame PSNR values (dB).
    psnr_sum: f64,
    /// Running sum of per-frame SSIM values [0, 1].
    ssim_sum: f64,
    /// Number of frames contributing to PSNR statistics.
    psnr_count: u64,
    /// Number of frames contributing to SSIM statistics.
    ssim_count: u64,
    /// Minimum PSNR observed (dB).
    psnr_min: f64,
    /// Maximum PSNR observed (dB).
    psnr_max: f64,
    /// Minimum SSIM observed.
    ssim_min: f64,
    /// Maximum SSIM observed.
    ssim_max: f64,
}

impl BitstreamStats {
    /// Create a new, empty statistics accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            psnr_min: f64::INFINITY,
            psnr_max: f64::NEG_INFINITY,
            ssim_min: f64::INFINITY,
            ssim_max: f64::NEG_INFINITY,
            ..Self::default()
        }
    }

    // -----------------------------------------------------------------------
    // PSNR
    // -----------------------------------------------------------------------

    /// Compute and record the PSNR between the original and reconstructed
    /// pixel buffers for one frame.
    ///
    /// PSNR is computed on the luma (Y) plane:
    /// ```text
    /// MSE = sum((orig[i] - recon[i])^2) / (w * h)
    /// PSNR = 10 * log10(255^2 / MSE)
    /// ```
    ///
    /// An MSE of 0.0 (identical buffers) yields +∞ dB; in that case `100.0`
    /// dB is recorded as a practical maximum.
    ///
    /// # Parameters
    /// - `orig`  – original uncompressed pixel buffer (u8 luma, length `w * h`).
    /// - `recon` – reconstructed pixel buffer (same layout as `orig`).
    /// - `w`     – frame width in pixels.
    /// - `h`     – frame height in pixels.
    ///
    /// # Panics
    /// Panics if `orig.len() != w * h` or `recon.len() != w * h`.
    pub fn update_psnr(&mut self, orig: &[u8], recon: &[u8], w: u32, h: u32) {
        let n = (w * h) as usize;
        assert_eq!(orig.len(), n, "update_psnr: orig length mismatch");
        assert_eq!(recon.len(), n, "update_psnr: recon length mismatch");

        let mse: f64 = orig
            .iter()
            .zip(recon.iter())
            .map(|(&a, &b)| {
                let d = a as f64 - b as f64;
                d * d
            })
            .sum::<f64>()
            / n as f64;

        let psnr = if mse < f64::EPSILON {
            100.0_f64
        } else {
            10.0 * (255.0_f64 * 255.0 / mse).log10()
        };

        self.psnr_sum += psnr;
        self.psnr_count += 1;
        if psnr < self.psnr_min {
            self.psnr_min = psnr;
        }
        if psnr > self.psnr_max {
            self.psnr_max = psnr;
        }
    }

    /// Returns the mean PSNR across all recorded frames, or `None` if no
    /// frames have been processed.
    #[must_use]
    pub fn mean_psnr(&self) -> Option<f64> {
        if self.psnr_count == 0 {
            None
        } else {
            Some(self.psnr_sum / self.psnr_count as f64)
        }
    }

    /// Returns the minimum per-frame PSNR, or `None` if no frames recorded.
    #[must_use]
    pub fn min_psnr(&self) -> Option<f64> {
        if self.psnr_count == 0 {
            None
        } else {
            Some(self.psnr_min)
        }
    }

    /// Returns the maximum per-frame PSNR, or `None` if no frames recorded.
    #[must_use]
    pub fn max_psnr(&self) -> Option<f64> {
        if self.psnr_count == 0 {
            None
        } else {
            Some(self.psnr_max)
        }
    }

    // -----------------------------------------------------------------------
    // SSIM
    // -----------------------------------------------------------------------

    /// Compute and record the mean SSIM between original and reconstructed
    /// buffers for one frame.
    ///
    /// Uses the simplified single-window SSIM formula (per-image, not
    /// per-patch) as a fast approximation:
    /// ```text
    /// μ_x, μ_y  — mean pixel values
    /// σ_x, σ_y  — standard deviations
    /// σ_xy      — cross-covariance
    /// SSIM = (2μ_xμ_y + C1)(2σ_xy + C2) / ((μ_x²+μ_y²+C1)(σ_x²+σ_y²+C2))
    /// ```
    /// with `C1 = (0.01 * 255)^2` and `C2 = (0.03 * 255)^2`.
    ///
    /// # Parameters
    /// - `orig`  – original pixel buffer (u8 luma, length `w * h`).
    /// - `recon` – reconstructed pixel buffer (same layout).
    /// - `w`     – frame width in pixels.
    /// - `h`     – frame height in pixels.
    ///
    /// # Panics
    /// Panics if `orig.len() != w * h` or `recon.len() != w * h`.
    pub fn update_ssim(&mut self, orig: &[u8], recon: &[u8], w: u32, h: u32) {
        let n = (w * h) as usize;
        assert_eq!(orig.len(), n, "update_ssim: orig length mismatch");
        assert_eq!(recon.len(), n, "update_ssim: recon length mismatch");

        let n_f = n as f64;

        let mu_x: f64 = orig.iter().map(|&v| v as f64).sum::<f64>() / n_f;
        let mu_y: f64 = recon.iter().map(|&v| v as f64).sum::<f64>() / n_f;

        let var_x: f64 = orig
            .iter()
            .map(|&v| {
                let d = v as f64 - mu_x;
                d * d
            })
            .sum::<f64>()
            / n_f;
        let var_y: f64 = recon
            .iter()
            .map(|&v| {
                let d = v as f64 - mu_y;
                d * d
            })
            .sum::<f64>()
            / n_f;

        let cov_xy: f64 = orig
            .iter()
            .zip(recon.iter())
            .map(|(&a, &b)| (a as f64 - mu_x) * (b as f64 - mu_y))
            .sum::<f64>()
            / n_f;

        const C1: f64 = (0.01 * 255.0) * (0.01 * 255.0);
        const C2: f64 = (0.03 * 255.0) * (0.03 * 255.0);

        let numerator = (2.0 * mu_x * mu_y + C1) * (2.0 * cov_xy + C2);
        let denominator = (mu_x * mu_x + mu_y * mu_y + C1) * (var_x + var_y + C2);

        let ssim = if denominator.abs() < f64::EPSILON {
            1.0
        } else {
            (numerator / denominator).clamp(-1.0, 1.0)
        };

        self.ssim_sum += ssim;
        self.ssim_count += 1;
        if ssim < self.ssim_min {
            self.ssim_min = ssim;
        }
        if ssim > self.ssim_max {
            self.ssim_max = ssim;
        }
    }

    /// Returns the mean SSIM across all recorded frames, or `None` if no
    /// frames have been processed.
    #[must_use]
    pub fn mean_ssim(&self) -> Option<f64> {
        if self.ssim_count == 0 {
            None
        } else {
            Some(self.ssim_sum / self.ssim_count as f64)
        }
    }

    /// Returns the minimum per-frame SSIM, or `None` if no frames recorded.
    #[must_use]
    pub fn min_ssim(&self) -> Option<f64> {
        if self.ssim_count == 0 {
            None
        } else {
            Some(self.ssim_min)
        }
    }

    /// Returns the maximum per-frame SSIM, or `None` if no frames recorded.
    #[must_use]
    pub fn max_ssim(&self) -> Option<f64> {
        if self.ssim_count == 0 {
            None
        } else {
            Some(self.ssim_max)
        }
    }

    // -----------------------------------------------------------------------
    // Frame counts
    // -----------------------------------------------------------------------

    /// Returns the number of frames that contributed to PSNR statistics.
    #[must_use]
    pub fn psnr_frame_count(&self) -> u64 {
        self.psnr_count
    }

    /// Returns the number of frames that contributed to SSIM statistics.
    #[must_use]
    pub fn ssim_frame_count(&self) -> u64 {
        self.ssim_count
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn identical_frame(size: usize, val: u8) -> (Vec<u8>, Vec<u8>) {
        (vec![val; size], vec![val; size])
    }

    fn noisy_frame(size: usize) -> (Vec<u8>, Vec<u8>) {
        let orig: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let recon: Vec<u8> = orig.iter().map(|&v| v.saturating_add(10)).collect();
        (orig, recon)
    }

    #[test]
    fn new_stats_no_data() {
        let s = BitstreamStats::new();
        assert!(s.mean_psnr().is_none());
        assert!(s.mean_ssim().is_none());
    }

    #[test]
    fn psnr_identical_frames_near_100() {
        let mut s = BitstreamStats::new();
        let (orig, recon) = identical_frame(64 * 64, 128);
        s.update_psnr(&orig, &recon, 64, 64);
        let psnr = s.mean_psnr().expect("should have PSNR");
        assert!(
            (psnr - 100.0).abs() < 1e-6,
            "identical PSNR should be 100 dB"
        );
    }

    #[test]
    fn psnr_positive_for_different_frames() {
        let mut s = BitstreamStats::new();
        let (orig, recon) = noisy_frame(16 * 16);
        s.update_psnr(&orig, &recon, 16, 16);
        let psnr = s.mean_psnr().expect("should have PSNR");
        assert!(
            psnr > 0.0 && psnr < 100.0,
            "PSNR {psnr} should be in (0, 100)"
        );
    }

    #[test]
    fn psnr_accumulated_over_multiple_frames() {
        let mut s = BitstreamStats::new();
        let (o1, r1) = identical_frame(16 * 16, 100);
        let (o2, r2) = noisy_frame(16 * 16);
        s.update_psnr(&o1, &r1, 16, 16);
        s.update_psnr(&o2, &r2, 16, 16);
        assert_eq!(s.psnr_frame_count(), 2);
    }

    #[test]
    fn ssim_identical_frames_near_one() {
        let mut s = BitstreamStats::new();
        let (orig, recon) = identical_frame(32 * 32, 128);
        s.update_ssim(&orig, &recon, 32, 32);
        let ssim = s.mean_ssim().expect("should have SSIM");
        assert!(ssim > 0.99, "identical SSIM should be ~1.0, got {ssim}");
    }

    #[test]
    fn ssim_drops_for_noisy_frames() {
        let mut s = BitstreamStats::new();
        let (orig, recon) = noisy_frame(32 * 32);
        s.update_ssim(&orig, &recon, 32, 32);
        let ssim = s.mean_ssim().expect("should have SSIM");
        // SSIM should drop below 1.0 for noisy frames
        assert!(ssim < 1.0, "noisy SSIM should be < 1.0");
    }

    #[test]
    fn stats_min_max_psnr() {
        let mut s = BitstreamStats::new();
        let (o1, r1) = identical_frame(16 * 16, 100); // PSNR = 100
        let (o2, r2) = noisy_frame(16 * 16); // PSNR < 100
        s.update_psnr(&o1, &r1, 16, 16);
        s.update_psnr(&o2, &r2, 16, 16);
        assert!(s.min_psnr().expect("min psnr") <= s.max_psnr().expect("max psnr"));
    }

    #[test]
    fn reset_clears_all() {
        let mut s = BitstreamStats::new();
        let (orig, recon) = identical_frame(4 * 4, 200);
        s.update_psnr(&orig, &recon, 4, 4);
        s.reset();
        assert!(s.mean_psnr().is_none());
        assert_eq!(s.psnr_frame_count(), 0);
    }

    #[test]
    #[should_panic(expected = "orig length mismatch")]
    fn update_psnr_panics_on_wrong_length() {
        let mut s = BitstreamStats::new();
        s.update_psnr(&[0u8; 10], &[0u8; 16], 4, 4);
    }
}
