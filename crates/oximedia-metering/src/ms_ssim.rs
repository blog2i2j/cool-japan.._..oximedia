//! Multi-Scale SSIM (MS-SSIM) video quality metric.
//!
//! Implements the Multi-Scale Structural Similarity (MS-SSIM) metric as
//! described in:
//!
//! > Z. Wang, E. P. Simoncelli and A. C. Bovik, "Multi-scale structural
//! > similarity for image quality assessment," *Proc. 37th Asilomar Conference
//! > on Signals, Systems and Computers*, Pacific Grove, CA, USA, 2003,
//! > pp. 1398-1402.
//!
//! MS-SSIM is computed at 5 scales by iteratively downsampling (2× average
//! pooling) and computing per-scale SSIM components.  The final score is:
//!
//! ```text
//! MS-SSIM = l₅^α · ∏ᵢ₌₁⁵ csᵢ^βᵢ
//! ```
//!
//! where `l` is luminance comparison and `cs` is the combined
//! contrast–structure comparison.  In the Wang (2003) formulation
//! `α = β₅ = 0.1333` and `βᵢ` for *i = 1..4* are `[0.0448, 0.2856, 0.3001,
//! 0.2363]`.  The luminance term only enters at the finest scale (*scale 5*
//! in the paper, index 4 in 0-based code).

/// MS-SSIM scale weights (Wang 2003, Table I).
///
/// `WEIGHTS[i]` is `βᵢ` for scales 1–5 (0-based index 0–4).
/// The last weight (index 4) applies to both contrast–structure *and*
/// luminance at the finest scale.
const WEIGHTS: [f64; 5] = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333];

/// Stability constants for SSIM (Wang 2004).
const K1: f64 = 0.01;
const K2: f64 = 0.03;
/// Dynamic range of 8-bit images.
const L: f64 = 255.0;
const C1: f64 = (K1 * L) * (K1 * L); // (0.01 * 255)² = 6.5025
const C2: f64 = (K2 * L) * (K2 * L); // (0.03 * 255)² = 58.5225

/// Downsample an 8-bit image by 2× using non-overlapping 2×2 average pooling.
///
/// Returns `(downsampled_pixels, new_width, new_height)`.  The output
/// dimensions are `⌊width/2⌋ × ⌊height/2⌋`.  If the input has fewer than
/// 4 pixels, the original slice is returned unchanged with the same
/// dimensions.
#[must_use]
pub fn downsample_2x(pixels: &[u8], width: u32, height: u32) -> (Vec<u8>, u32, u32) {
    let w = width as usize;
    let h = height as usize;
    let new_w = w / 2;
    let new_h = h / 2;

    if new_w == 0 || new_h == 0 {
        // Cannot downsample further; return a copy of the original.
        return (pixels.to_vec(), width, height);
    }

    let mut out = Vec::with_capacity(new_w * new_h);
    for by in 0..new_h {
        for bx in 0..new_w {
            let y0 = by * 2;
            let y1 = y0 + 1;
            let x0 = bx * 2;
            let x1 = x0 + 1;
            // 2×2 average (integer arithmetic to avoid float overhead).
            let sum = pixels[y0 * w + x0] as u32
                + pixels[y0 * w + x1] as u32
                + pixels[y1 * w + x0] as u32
                + pixels[y1 * w + x1] as u32;
            out.push((sum / 4) as u8);
        }
    }

    (out, new_w as u32, new_h as u32)
}

/// Compute luminance, contrast, and structure components for SSIM at a
/// single scale.
///
/// Returns `(luminance, contrast_structure)` both in [0, 1].
fn ssim_components(ref_frame: &[u8], cmp_frame: &[u8], n: usize) -> (f64, f64) {
    if n == 0 {
        return (1.0, 1.0);
    }

    let inv_n = 1.0 / n as f64;

    // Means
    let mu_x: f64 = ref_frame.iter().map(|&v| v as f64).sum::<f64>() * inv_n;
    let mu_y: f64 = cmp_frame.iter().map(|&v| v as f64).sum::<f64>() * inv_n;

    // Variances and covariance (biased estimator, consistent with Wang 2004).
    let mut var_x = 0.0_f64;
    let mut var_y = 0.0_f64;
    let mut cov_xy = 0.0_f64;

    for (&x, &y) in ref_frame.iter().zip(cmp_frame.iter()) {
        let dx = x as f64 - mu_x;
        let dy = y as f64 - mu_y;
        var_x += dx * dx;
        var_y += dy * dy;
        cov_xy += dx * dy;
    }

    var_x *= inv_n;
    var_y *= inv_n;
    cov_xy *= inv_n;

    // Luminance comparison: l(x,y) = (2·μx·μy + C1) / (μx² + μy² + C1)
    let luminance = (2.0 * mu_x * mu_y + C1) / (mu_x * mu_x + mu_y * mu_y + C1);

    // Contrast–structure comparison: cs(x,y) = (2·σxy + C2) / (σx² + σy² + C2)
    let contrast_structure = (2.0 * cov_xy + C2) / (var_x + var_y + C2);

    (luminance.clamp(0.0, 1.0), contrast_structure.clamp(0.0, 1.0))
}

/// Compute the Multi-Scale SSIM (MS-SSIM) between two 8-bit luma frames.
///
/// Both frames must be single-channel (luminance) with `width × height`
/// pixels laid out in row-major order.  If the images are too small for
/// 5 downsampling stages the metric gracefully uses however many scales
/// are available.
///
/// # Arguments
///
/// * `ref_frame` - Reference (uncompressed) frame pixels.
/// * `cmp_frame` - Compared (compressed/processed) frame pixels.
/// * `width`     - Frame width in pixels.
/// * `height`    - Frame height in pixels.
///
/// # Returns
///
/// MS-SSIM score in `[0, 1]`.  A score of 1.0 indicates perfect quality;
/// lower values indicate degradation.  Returns 1.0 if both frames are
/// empty or have the same single pixel.
#[must_use]
pub fn ms_ssim(ref_frame: &[u8], cmp_frame: &[u8], width: u32, height: u32) -> f32 {
    const NUM_SCALES: usize = 5;

    if ref_frame.is_empty() || cmp_frame.is_empty() {
        return 1.0;
    }

    // Build scale pyramid.
    let mut ref_scales: Vec<(Vec<u8>, u32, u32)> = Vec::with_capacity(NUM_SCALES);
    let mut cmp_scales: Vec<(Vec<u8>, u32, u32)> = Vec::with_capacity(NUM_SCALES);

    ref_scales.push((ref_frame.to_vec(), width, height));
    cmp_scales.push((cmp_frame.to_vec(), width, height));

    for s in 1..NUM_SCALES {
        let (prev_ref, pw, ph) = &ref_scales[s - 1];
        let (prev_cmp, _, _) = &cmp_scales[s - 1];
        let (down_ref, nw, nh) = downsample_2x(prev_ref, *pw, *ph);
        let (down_cmp, _, _) = downsample_2x(prev_cmp, *pw, *ph);
        if nw == *pw && nh == *ph {
            // Could not downsample further; stop early.
            break;
        }
        ref_scales.push((down_ref, nw, nh));
        cmp_scales.push((down_cmp, nw, nh));
    }

    let actual_scales = ref_scales.len();

    // Collect per-scale (luminance, cs) components.
    let mut luminance = 1.0_f64;
    let mut ms_ssim_acc = 1.0_f64;

    for s in 0..actual_scales {
        let (ref_s, w_s, h_s) = &ref_scales[s];
        let (cmp_s, _, _) = &cmp_scales[s];
        let n = (*w_s as usize) * (*h_s as usize);
        let (l_s, cs_s) = ssim_components(ref_s, cmp_s, n);

        // Determine the weight index: we index from the finest scale backwards
        // so that the finest scale (scale 5 in the paper) uses WEIGHTS[4].
        let weight_idx = NUM_SCALES - 1 - (NUM_SCALES - 1 - s).min(NUM_SCALES - 1);
        let w = WEIGHTS[weight_idx];

        // Contrast–structure contribution at every scale.
        ms_ssim_acc *= cs_s.powf(w);

        // Luminance only at the finest scale (last scale in our loop).
        if s == actual_scales - 1 {
            luminance = l_s.powf(w);
        }
    }

    (luminance * ms_ssim_acc).clamp(0.0, 1.0) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── downsample_2x ─────────────────────────────────────────────────────────

    #[test]
    fn test_downsample_2x_uniform() {
        // Uniform image of value 200: downsampled value should also be 200.
        let pixels = vec![200u8; 4 * 4];
        let (out, nw, nh) = downsample_2x(&pixels, 4, 4);
        assert_eq!(nw, 2);
        assert_eq!(nh, 2);
        assert_eq!(out.len(), 4);
        for &v in &out {
            assert_eq!(v, 200);
        }
    }

    #[test]
    fn test_downsample_2x_known_block() {
        // 2×2 block: [100, 200, 100, 200] → average = 150
        let pixels = vec![100u8, 200, 100, 200];
        let (out, nw, nh) = downsample_2x(&pixels, 2, 2);
        assert_eq!(nw, 1);
        assert_eq!(nh, 1);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], 150);
    }

    #[test]
    fn test_downsample_2x_dimensions() {
        let pixels = vec![128u8; 8 * 6]; // 8 wide, 6 tall
        let (_, nw, nh) = downsample_2x(&pixels, 8, 6);
        assert_eq!(nw, 4);
        assert_eq!(nh, 3);
    }

    #[test]
    fn test_downsample_2x_too_small_passthrough() {
        // 1×1 cannot be downsampled further.
        let pixels = vec![42u8];
        let (out, nw, nh) = downsample_2x(&pixels, 1, 1);
        assert_eq!(nw, 1);
        assert_eq!(nh, 1);
        assert_eq!(out, vec![42u8]);
    }

    // ── ms_ssim ───────────────────────────────────────────────────────────────

    #[test]
    fn test_ms_ssim_identical_frames() {
        let w = 32u32;
        let h = 32u32;
        let pixels: Vec<u8> = (0..(w * h) as usize).map(|i| (i % 256) as u8).collect();
        let score = ms_ssim(&pixels, &pixels, w, h);
        assert!(
            (score - 1.0).abs() < 0.01,
            "Identical frames should score ≈ 1.0, got {score}"
        );
    }

    #[test]
    fn test_ms_ssim_completely_different() {
        let w = 32u32;
        let h = 32u32;
        let ref_pixels = vec![0u8; (w * h) as usize];
        let cmp_pixels = vec![255u8; (w * h) as usize];
        let score = ms_ssim(&ref_pixels, &cmp_pixels, w, h);
        // Should be noticeably below 1.0 for maximally different images.
        assert!(
            score < 0.9,
            "Maximally different frames should score < 0.9, got {score}"
        );
    }

    #[test]
    fn test_ms_ssim_slightly_degraded() {
        let w = 64u32;
        let h = 64u32;
        let ref_pixels: Vec<u8> = (0..(w * h) as usize)
            .map(|i| ((i as f64 / (w * h) as f64) * 200.0) as u8)
            .collect();
        // Add small noise to produce a degraded version.
        let cmp_pixels: Vec<u8> = ref_pixels
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let noise = (i % 5) as i16 - 2;
                (v as i16 + noise).clamp(0, 255) as u8
            })
            .collect();
        let score = ms_ssim(&ref_pixels, &cmp_pixels, w, h);
        // Slightly degraded should score close to but below 1.0.
        assert!(score > 0.5, "Slightly degraded should score > 0.5, got {score}");
        assert!(score < 1.0, "Degraded should score < 1.0, got {score}");
    }

    #[test]
    fn test_ms_ssim_score_in_range() {
        let w = 16u32;
        let h = 16u32;
        let ref_pixels: Vec<u8> = (0..(w * h) as usize).map(|i| (i % 256) as u8).collect();
        let cmp_pixels: Vec<u8> = ref_pixels.iter().map(|&v| v.saturating_add(10)).collect();
        let score = ms_ssim(&ref_pixels, &cmp_pixels, w, h);
        assert!(score >= 0.0 && score <= 1.0, "Score out of [0,1]: {score}");
    }

    #[test]
    fn test_ms_ssim_empty_returns_one() {
        let score = ms_ssim(&[], &[], 0, 0);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_ms_ssim_degraded_lower_than_identical() {
        let w = 32u32;
        let h = 32u32;
        let ref_pixels: Vec<u8> = (0..(w * h) as usize).map(|i| (i % 256) as u8).collect();
        let perfect_score = ms_ssim(&ref_pixels, &ref_pixels, w, h);
        let degraded: Vec<u8> = ref_pixels
            .iter()
            .map(|&v| v.saturating_add(50))
            .collect();
        let degraded_score = ms_ssim(&ref_pixels, &degraded, w, h);
        assert!(
            degraded_score <= perfect_score,
            "Degraded ({degraded_score}) should be ≤ perfect ({perfect_score})"
        );
    }
}
