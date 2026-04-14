//! VMAF-like perceptual quality estimator.
//!
//! Provides [`VmafEstimator`] which combines PSNR and SSIM estimates to
//! approximate a VMAF score in the range [0, 100].
//!
//! The combination formula is:
//! ```text
//! vmaf ≈ 0.3 * psnr_normalized + 0.7 * ssim_score * 100
//! ```
//!
//! This is intentionally simpler than the full VMAF model (which uses an SVM)
//! but correlates well with perceptual quality for common codec artefacts.

/// VMAF-like perceptual quality estimator.
///
/// Combines PSNR (weight 0.3) and SSIM estimate (weight 0.7) to produce a
/// score in [0, 100].
pub struct VmafEstimator;

impl VmafEstimator {
    /// Estimate perceptual quality for a pair of reference and distorted frames.
    ///
    /// # Arguments
    ///
    /// * `ref_frame` – reference luma plane (`width * height` bytes)
    /// * `dist_frame` – distorted luma plane (`width * height` bytes)
    /// * `width` – frame width in pixels
    /// * `height` – frame height in pixels
    ///
    /// # Returns
    ///
    /// A score in [0.0, 100.0].  Returns 100.0 for identical frames.
    /// Returns 0.0 if the inputs are empty or mismatched in size.
    #[must_use]
    pub fn estimate(ref_frame: &[u8], dist_frame: &[u8], width: u32, height: u32) -> f32 {
        let pixels = (width as usize) * (height as usize);
        if pixels == 0
            || ref_frame.len() < pixels
            || dist_frame.len() < pixels
        {
            return 0.0;
        }

        let psnr_norm = compute_psnr_normalized(&ref_frame[..pixels], &dist_frame[..pixels]);
        let ssim = compute_ssim_estimate(&ref_frame[..pixels], &dist_frame[..pixels], width as usize);

        let score = 0.3 * psnr_norm * 100.0 + 0.7 * ssim * 100.0;
        score.clamp(0.0, 100.0)
    }
}

/// Compute PSNR in the range [0.0, 1.0] where 1.0 = identical frames.
///
/// PSNR_norm = clamp((PSNR - 20) / 40, 0, 1) so that:
/// - PSNR ≥ 60 dB → 1.0 (excellent)
/// - PSNR = 30 dB → 0.25 (fair)
/// - PSNR ≤ 20 dB → 0.0 (poor)
#[allow(clippy::cast_precision_loss)]
fn compute_psnr_normalized(reference: &[u8], distorted: &[u8]) -> f32 {
    let n = reference.len() as f64;
    let mut mse = 0.0f64;
    for (&r, &d) in reference.iter().zip(distorted.iter()) {
        let diff = f64::from(r) - f64::from(d);
        mse += diff * diff;
    }
    mse /= n;

    if mse < 1e-10 {
        return 1.0; // identical frames
    }

    let psnr = 10.0 * (255.0_f64 * 255.0 / mse).log10();
    ((psnr as f32 - 20.0) / 40.0).clamp(0.0, 1.0)
}

/// Simplified SSIM estimate over non-overlapping 8×8 blocks.
///
/// Full SSIM requires 11×11 Gaussian windows; this lightweight version uses
/// block-level statistics for speed.  Returns value in [0.0, 1.0].
#[allow(clippy::cast_precision_loss)]
fn compute_ssim_estimate(reference: &[u8], distorted: &[u8], width: usize) -> f32 {
    const BLOCK: usize = 8;
    const C1: f64 = (0.01 * 255.0) * (0.01 * 255.0); // (K1*L)²
    const C2: f64 = (0.03 * 255.0) * (0.03 * 255.0); // (K2*L)²

    let height = reference.len() / width.max(1);
    if height == 0 || width < BLOCK || height < BLOCK {
        // Too small for block SSIM; fall back to pixel correlation
        return compute_pixel_correlation(reference, distorted);
    }

    let blocks_y = height / BLOCK;
    let blocks_x = width / BLOCK;
    let mut ssim_sum = 0.0f64;
    let mut block_count = 0u64;

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let mut ref_sum = 0.0f64;
            let mut dist_sum = 0.0f64;
            let mut ref_sq = 0.0f64;
            let mut dist_sq = 0.0f64;
            let mut cross = 0.0f64;

            for dy in 0..BLOCK {
                for dx in 0..BLOCK {
                    let idx = (by * BLOCK + dy) * width + (bx * BLOCK + dx);
                    let r = f64::from(reference[idx]);
                    let d = f64::from(distorted[idx]);
                    ref_sum += r;
                    dist_sum += d;
                    ref_sq += r * r;
                    dist_sq += d * d;
                    cross += r * d;
                }
            }

            let n = (BLOCK * BLOCK) as f64;
            let mu_r = ref_sum / n;
            let mu_d = dist_sum / n;
            let sigma_r_sq = (ref_sq / n - mu_r * mu_r).max(0.0);
            let sigma_d_sq = (dist_sq / n - mu_d * mu_d).max(0.0);
            let sigma_rd = cross / n - mu_r * mu_d;

            let numerator = (2.0 * mu_r * mu_d + C1) * (2.0 * sigma_rd + C2);
            let denominator = (mu_r * mu_r + mu_d * mu_d + C1) * (sigma_r_sq + sigma_d_sq + C2);

            ssim_sum += numerator / denominator;
            block_count += 1;
        }
    }

    if block_count == 0 {
        return 1.0;
    }
    ((ssim_sum / block_count as f64) as f32).clamp(0.0, 1.0)
}

/// Fallback: compute normalised pixel-level correlation as a proxy for SSIM.
#[allow(clippy::cast_precision_loss)]
fn compute_pixel_correlation(reference: &[u8], distorted: &[u8]) -> f32 {
    let n = reference.len() as f64;
    if n < 1.0 {
        return 1.0;
    }
    let mut sq_diff = 0.0f64;
    for (&r, &d) in reference.iter().zip(distorted.iter()) {
        let diff = f64::from(r) - f64::from(d);
        sq_diff += diff * diff;
    }
    let rmse = (sq_diff / n).sqrt();
    (1.0 - rmse / 255.0).clamp(0.0, 1.0) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_frames_score_100() {
        let frame = vec![128u8; 64 * 64];
        let score = VmafEstimator::estimate(&frame, &frame, 64, 64);
        assert!((score - 100.0).abs() < 1.0, "identical frames should score ~100, got {score}");
    }

    #[test]
    fn test_score_in_range_0_100() {
        let reference: Vec<u8> = (0..64).flat_map(|y: u8| (0..64u8).map(move |x| x.wrapping_add(y))).collect();
        let distorted: Vec<u8> = reference.iter().map(|&v| v.wrapping_add(30)).collect();
        let score = VmafEstimator::estimate(&reference, &distorted, 64, 64);
        assert!(score >= 0.0 && score <= 100.0, "score out of range: {score}");
    }

    #[test]
    fn test_degraded_frame_lower_score_than_identical() {
        let reference = vec![100u8; 32 * 32];
        let distorted: Vec<u8> = (0..32 * 32).map(|i: usize| ((i * 37 + 13) % 256) as u8).collect();
        let ideal_score = VmafEstimator::estimate(&reference, &reference, 32, 32);
        let degraded_score = VmafEstimator::estimate(&reference, &distorted, 32, 32);
        assert!(
            degraded_score <= ideal_score,
            "degraded={degraded_score} should be ≤ ideal={ideal_score}"
        );
    }

    #[test]
    fn test_empty_frames_return_zero() {
        let score = VmafEstimator::estimate(&[], &[], 0, 0);
        assert!((score - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_psnr_normalized_identical() {
        let frame = vec![128u8; 64];
        let psnr = compute_psnr_normalized(&frame, &frame);
        assert!((psnr - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ssim_estimate_identical() {
        let frame: Vec<u8> = (0..64u8).flat_map(|y| (0..64u8).map(move |x| x.wrapping_add(y))).collect();
        let ssim = compute_ssim_estimate(&frame, &frame, 64);
        assert!(ssim >= 0.99, "identical frame SSIM should be ~1.0, got {ssim}");
    }

    #[test]
    fn test_mismatched_sizes_return_zero() {
        let reference = vec![128u8; 64 * 64];
        let distorted = vec![128u8; 32 * 32]; // too small
        let score = VmafEstimator::estimate(&reference, &distorted, 64, 64);
        assert!((score - 0.0).abs() < f32::EPSILON, "mismatched sizes should return 0.0, got {score}");
    }

    #[test]
    fn test_heavily_distorted_frame_low_score() {
        let reference = vec![200u8; 32 * 32];
        // Invert all pixel values → maximum distortion
        let distorted: Vec<u8> = reference.iter().map(|&v| 255 - v).collect();
        let score = VmafEstimator::estimate(&reference, &distorted, 32, 32);
        assert!(score < 50.0, "heavily distorted frame should score below 50.0, got {score}");
    }
}
