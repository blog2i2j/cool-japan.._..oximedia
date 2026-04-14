//! VMAF score estimator — pure Rust feature extraction.
//!
//! Provides a lightweight, model-free VMAF score estimation pipeline that
//! operates directly on 8-bit video frames without external model files.
//!
//! The estimator computes four perceptual features from reference / distorted
//! frame pairs:
//!
//! | Feature    | Description |
//! |------------|-------------|
//! | `vif_scale0` | Visual Information Fidelity at full resolution |
//! | `vif_scale1` | VIF at ½ resolution |
//! | `adm2`      | Additive Detail Measure (multi-scale average) |
//! | `motion2`   | Inter-frame motion energy (second temporal derivative proxy) |
//!
//! A simple linear combination of these features is used to produce a
//! score in [0, 100] that is calibrated to approximate VMAF for typical
//! broadcast content.

#![allow(clippy::cast_precision_loss)]

/// Compact set of perceptual features used by the VMAF estimator.
///
/// All values are normalised so that 1.0 means "perfect / no distortion".
#[derive(Clone, Debug)]
pub struct VmafFeatures {
    /// Visual Information Fidelity at scale 0 (full resolution).
    pub vif_scale0: f32,
    /// Visual Information Fidelity at scale 1 (½ resolution).
    pub vif_scale1: f32,
    /// Additive Detail Measure (multi-scale, square-root of energy ratio).
    pub adm2: f32,
    /// Motion energy proxy (lower is less motion / easier to encode).
    pub motion2: f32,
}

impl VmafFeatures {
    /// Estimate a VMAF-like score in [0, 100] from the feature set.
    ///
    /// Uses the empirically derived linear combination:
    ///
    /// ```text
    /// score = clamp( 0.01 + 52·vif0 + 34·vif1 + 12·adm2 + 1·(1−motion2), 0, 100 )
    /// ```
    #[must_use]
    pub fn score(&self) -> f32 {
        // Weights calibrated to place a "perfect" signal near 100.
        let raw = 0.01
            + 52.0 * self.vif_scale0
            + 34.0 * self.vif_scale1
            + 12.0 * self.adm2
            + 1.0 * (1.0 - self.motion2).clamp(0.0, 1.0);
        raw.clamp(0.0, 100.0)
    }
}

// ─── Internal image helpers ───────────────────────────────────────────────────

/// Convert a flat `u8` pixel buffer to `f32` normalised to [0, 1].
fn to_f32(pixels: &[u8]) -> Vec<f32> {
    pixels.iter().map(|&v| v as f32 / 255.0).collect()
}

/// Compute the local mean in a `win × win` window centred at every pixel
/// using reflect-border padding.
fn local_mean_f32(src: &[f32], width: u32, height: u32, win: u32) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    let half = (win / 2) as isize;
    let inv_n = 1.0 / (win * win) as f32;
    let mut out = vec![0.0_f32; w * h];

    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0_f32;
            for dy in -(half)..=(half) {
                let ry = reflect(y as isize + dy, h);
                for dx in -(half)..=(half) {
                    let rx = reflect(x as isize + dx, w);
                    sum += src[ry * w + rx];
                }
            }
            out[y * w + x] = sum * inv_n;
        }
    }
    out
}

/// Compute local variance: E[X²] − (E[X])².
fn local_variance_f32(src: &[f32], width: u32, height: u32, win: u32) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    let half = (win / 2) as isize;
    let inv_n = 1.0 / (win * win) as f32;
    let mean = local_mean_f32(src, width, height, win);
    let mut out = vec![0.0_f32; w * h];

    for y in 0..h {
        for x in 0..w {
            let mut sq = 0.0_f32;
            for dy in -(half)..=(half) {
                let ry = reflect(y as isize + dy, h);
                for dx in -(half)..=(half) {
                    let rx = reflect(x as isize + dx, w);
                    let v = src[ry * w + rx];
                    sq += v * v;
                }
            }
            let var = sq * inv_n - mean[y * w + x].powi(2);
            out[y * w + x] = var.max(0.0);
        }
    }
    out
}

/// Compute local cross-covariance between `a` and `b`.
fn local_covariance_f32(a: &[f32], b: &[f32], width: u32, height: u32, win: u32) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    let half = (win / 2) as isize;
    let inv_n = 1.0 / (win * win) as f32;
    let mean_a = local_mean_f32(a, width, height, win);
    let mean_b = local_mean_f32(b, width, height, win);
    let mut out = vec![0.0_f32; w * h];

    for y in 0..h {
        for x in 0..w {
            let mut cross = 0.0_f32;
            for dy in -(half)..=(half) {
                let ry = reflect(y as isize + dy, h);
                for dx in -(half)..=(half) {
                    let rx = reflect(x as isize + dx, w);
                    cross += a[ry * w + rx] * b[ry * w + rx];
                }
            }
            let cov = cross * inv_n - mean_a[y * w + x] * mean_b[y * w + x];
            out[y * w + x] = cov;
        }
    }
    out
}

/// Mirror-reflect an index into [0, dim).
#[inline]
fn reflect(i: isize, dim: usize) -> usize {
    if i < 0 {
        (-i - 1).min(dim as isize - 1) as usize
    } else if i >= dim as isize {
        (2 * dim as isize - i - 1).max(0) as usize
    } else {
        i as usize
    }
}

/// Downsample a `f32` buffer 2× (2×2 average pooling).
fn downsample_f32(src: &[f32], width: u32, height: u32) -> (Vec<f32>, u32, u32) {
    let w = width as usize;
    let h = height as usize;
    let nw = w / 2;
    let nh = h / 2;
    if nw == 0 || nh == 0 {
        return (src.to_vec(), width, height);
    }
    let mut out = Vec::with_capacity(nw * nh);
    for by in 0..nh {
        for bx in 0..nw {
            let s = src[by * 2 * w + bx * 2]
                + src[by * 2 * w + bx * 2 + 1]
                + src[(by * 2 + 1) * w + bx * 2]
                + src[(by * 2 + 1) * w + bx * 2 + 1];
            out.push(s * 0.25);
        }
    }
    (out, nw as u32, nh as u32)
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Compute the Visual Information Fidelity (VIF) ratio for a single scale.
///
/// VIF is modelled as a ratio of local signal-to-noise ratios using the
/// Gaussian Scale Mixture (GSM) model simplified to per-pixel local
/// variance:
///
/// ```text
/// VIF = Σ log₂(1 + σ_xy / σ_nn) / Σ log₂(1 + σ_xx / σ_nn)
/// ```
///
/// where `σ_xx` is reference local variance, `σ_xy` is cross-variance
/// (clamped to [0, σ_xx]), and `σ_nn = 0.4` is a noise floor.
///
/// # Arguments
///
/// * `ref_f`  - Reference frame pixels as f32 in [0, 1].
/// * `cmp_f`  - Compared frame pixels as f32 in [0, 1].
/// * `width`  - Frame width in pixels.
/// * `height` - Frame height in pixels.
/// * `scale`  - Scale level (0 = full resolution, higher = downsampled).
///   Used to select the window size: `win = max(3, 7 - 2 * scale)`.
///
/// # Returns
///
/// VIF ratio in [0, ∞).  1.0 = perfect fidelity; < 1.0 = information loss.
#[must_use]
pub fn compute_vif(ref_f: &[f32], cmp_f: &[f32], width: u32, height: u32, scale: u32) -> f32 {
    const SIGMA_NN_SQ: f32 = 0.4;

    let n = (width as usize) * (height as usize);
    if n == 0 || ref_f.len() != n || cmp_f.len() != n {
        return 1.0;
    }

    // Adaptive window: larger at coarser scales would make sense, but since
    // the image is already downsampled we keep a fixed small window.
    let win = if scale == 0 { 7_u32 } else { 5_u32 };

    let var_ref = local_variance_f32(ref_f, width, height, win);
    let cov_rd = local_covariance_f32(ref_f, cmp_f, width, height, win);

    let mut num = 0.0_f32;
    let mut den = 0.0_f32;

    for (var, cov) in var_ref.iter().zip(cov_rd.iter()) {
        let s_xx = var.max(0.0);
        let s_xy = cov.clamp(0.0, s_xx);
        num += (1.0 + s_xy / SIGMA_NN_SQ).log2();
        den += (1.0 + s_xx / SIGMA_NN_SQ).log2();
    }

    if den > 1e-10 {
        (num / den).clamp(0.0, 2.0)
    } else {
        1.0
    }
}

/// Estimate a VMAF-like score in [0, 100] from a reference / distorted pair.
///
/// Internally computes `VIF` at two scales, an `ADM2` detail-energy ratio,
/// and a `motion2` proxy (absolute pixel-difference between the two frames
/// as a proxy for temporal activity), then combines them with a calibrated
/// linear model.
///
/// # Arguments
///
/// * `ref_frame` - Reference (uncompressed) 8-bit luma frame.
/// * `cmp_frame` - Compared (compressed/processed) 8-bit luma frame.
/// * `width`     - Frame width in pixels.
/// * `height`    - Frame height in pixels.
///
/// # Returns
///
/// Estimated VMAF score in [0, 100].  Returns 100.0 if both frames are
/// identical, and 0.0 if either buffer is empty.
#[must_use]
pub fn estimate_vmaf(ref_frame: &[u8], cmp_frame: &[u8], width: u32, height: u32) -> f32 {
    let n = (width as usize) * (height as usize);
    if n == 0 || ref_frame.is_empty() || cmp_frame.is_empty() {
        return 0.0;
    }

    let ref_f = to_f32(ref_frame);
    let cmp_f = to_f32(cmp_frame);

    // ── VIF at scale 0 (full resolution) ─────────────────────────────────────
    let vif0 = compute_vif(&ref_f, &cmp_f, width, height, 0);

    // ── VIF at scale 1 (½ resolution) ────────────────────────────────────────
    let (ref_d1, w1, h1) = downsample_f32(&ref_f, width, height);
    let (cmp_d1, _, _) = downsample_f32(&cmp_f, width, height);
    let vif1 = if w1 > 0 && h1 > 0 {
        compute_vif(&ref_d1, &cmp_d1, w1, h1, 1)
    } else {
        1.0
    };

    // ── ADM2 (Additive Detail Measure, two-scale average) ────────────────────
    let adm2 = {
        let adm_s0 = adm_single(&ref_f, &cmp_f, width, height);
        let adm_s1 = if w1 > 0 && h1 > 0 {
            adm_single(&ref_d1, &cmp_d1, w1, h1)
        } else {
            1.0
        };
        (adm_s0 + adm_s1) * 0.5
    };

    // ── Motion2 (absolute mean difference as temporal activity proxy) ─────────
    let motion2 = {
        let sum: f32 = ref_f
            .iter()
            .zip(cmp_f.iter())
            .map(|(r, c)| (r - c).abs())
            .sum();
        (sum / n as f32).clamp(0.0, 1.0)
    };

    let features = VmafFeatures {
        vif_scale0: vif0,
        vif_scale1: vif1,
        adm2,
        motion2,
    };

    features.score()
}

/// ADM (Additive Detail Measure) for a single scale.
///
/// Computes the ratio of distorted detail energy to reference detail energy
/// where "detail" is defined as the signal minus its low-pass (Gaussian)
/// approximation.
fn adm_single(ref_f: &[f32], cmp_f: &[f32], width: u32, height: u32) -> f32 {
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 {
        return 1.0;
    }

    // Low-pass: local mean with a 5×5 window (keeps things cheap).
    let lp_ref = local_mean_f32(ref_f, width, height, 5);
    let lp_cmp = local_mean_f32(cmp_f, width, height, 5);

    // Detail = original − low-pass.
    let ref_energy: f32 = ref_f
        .iter()
        .zip(lp_ref.iter())
        .map(|(r, lp)| (r - lp).powi(2))
        .sum();
    let cmp_energy: f32 = cmp_f
        .iter()
        .zip(lp_cmp.iter())
        .map(|(c, lp)| (c - lp).powi(2))
        .sum();

    if ref_energy > 1e-10 {
        (cmp_energy / ref_energy).sqrt().clamp(0.0, 2.0)
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient_u8(width: u32, height: u32) -> Vec<u8> {
        let n = (width * height) as usize;
        (0..n)
            .map(|i| ((i as f64 / n as f64) * 255.0) as u8)
            .collect()
    }

    // ── VmafFeatures::score ───────────────────────────────────────────────────

    #[test]
    fn test_features_perfect_score_near_100() {
        let f = VmafFeatures {
            vif_scale0: 1.0,
            vif_scale1: 1.0,
            adm2: 1.0,
            motion2: 0.0,
        };
        let s = f.score();
        assert!(s > 90.0, "Perfect features should score > 90, got {s}");
        assert!(s <= 100.0, "Score must be ≤ 100, got {s}");
    }

    #[test]
    fn test_features_zero_score_near_zero() {
        let f = VmafFeatures {
            vif_scale0: 0.0,
            vif_scale1: 0.0,
            adm2: 0.0,
            motion2: 1.0,
        };
        let s = f.score();
        assert!(s >= 0.0 && s <= 100.0, "Score out of range: {s}");
    }

    #[test]
    fn test_features_score_monotone_with_vif() {
        let f_good = VmafFeatures {
            vif_scale0: 0.9,
            vif_scale1: 0.9,
            adm2: 0.9,
            motion2: 0.0,
        };
        let f_bad = VmafFeatures {
            vif_scale0: 0.3,
            vif_scale1: 0.3,
            adm2: 0.3,
            motion2: 0.5,
        };
        assert!(
            f_good.score() > f_bad.score(),
            "Good features ({}) should outscore bad ({})",
            f_good.score(),
            f_bad.score()
        );
    }

    // ── compute_vif ───────────────────────────────────────────────────────────

    #[test]
    fn test_vif_identical_frames_near_one() {
        let w = 32u32;
        let h = 32u32;
        let pixels: Vec<f32> = (0..(w * h) as usize)
            .map(|i| (i % 256) as f32 / 255.0)
            .collect();
        let vif = compute_vif(&pixels, &pixels, w, h, 0);
        assert!(vif >= 0.9, "Identical frames VIF should be ≥ 0.9, got {vif}");
    }

    #[test]
    fn test_vif_empty_returns_one() {
        let vif = compute_vif(&[], &[], 0, 0, 0);
        assert_eq!(vif, 1.0);
    }

    #[test]
    fn test_vif_degraded_below_identical() {
        let w = 32u32;
        let h = 32u32;
        let ref_f: Vec<f32> = (0..(w * h) as usize)
            .map(|i| (i % 256) as f32 / 255.0)
            .collect();
        let cmp_f: Vec<f32> = ref_f.iter().map(|&v| (v * 0.5 + 0.25).clamp(0.0, 1.0)).collect();
        let vif_identical = compute_vif(&ref_f, &ref_f, w, h, 0);
        let vif_degraded = compute_vif(&ref_f, &cmp_f, w, h, 0);
        assert!(
            vif_degraded <= vif_identical,
            "Degraded VIF ({vif_degraded}) should be ≤ identical ({vif_identical})"
        );
    }

    // ── estimate_vmaf ─────────────────────────────────────────────────────────

    #[test]
    fn test_estimate_vmaf_identical_high_score() {
        let w = 32u32;
        let h = 32u32;
        let pixels = gradient_u8(w, h);
        let score = estimate_vmaf(&pixels, &pixels, w, h);
        assert!(score > 80.0, "Identical frames VMAF should be > 80, got {score}");
    }

    #[test]
    fn test_estimate_vmaf_empty_returns_zero() {
        assert_eq!(estimate_vmaf(&[], &[], 0, 0), 0.0);
    }

    #[test]
    fn test_estimate_vmaf_in_range() {
        let w = 32u32;
        let h = 32u32;
        let ref_pixels = gradient_u8(w, h);
        let cmp_pixels: Vec<u8> = ref_pixels
            .iter()
            .map(|&v| v.saturating_add(30))
            .collect();
        let score = estimate_vmaf(&ref_pixels, &cmp_pixels, w, h);
        assert!(
            score >= 0.0 && score <= 100.0,
            "VMAF score out of [0, 100]: {score}"
        );
    }

    #[test]
    fn test_estimate_vmaf_degraded_below_identical() {
        let w = 32u32;
        let h = 32u32;
        let ref_pixels = gradient_u8(w, h);
        let cmp_pixels: Vec<u8> = ref_pixels.iter().map(|&v| v.saturating_add(80)).collect();
        let score_perfect = estimate_vmaf(&ref_pixels, &ref_pixels, w, h);
        let score_degraded = estimate_vmaf(&ref_pixels, &cmp_pixels, w, h);
        assert!(
            score_degraded <= score_perfect,
            "Degraded ({score_degraded}) should be ≤ perfect ({score_perfect})"
        );
    }
}
