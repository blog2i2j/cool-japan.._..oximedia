//! LUT smoothing — Gaussian smoothing of 3-D LUT entries, contrast-preserving
//! smoothing, and smooth-vs-original difference analysis.
//!
//! # Overview
//!
//! 3-D LUTs generated from hardware measurements or aggressive color-grading
//! sessions often contain high-frequency noise that causes banding or
//! contouring artefacts in smooth gradients.  This module provides:
//!
//! * **Gaussian smoothing** – isotropic 3-D kernel applied to a flat LUT array.
//! * **Contrast-preserving smoothing** – blends the Gaussian result back with
//!   a local-contrast signal so that edge transitions are not washed out.
//! * **Difference analysis** – per-entry and RMS delta between original and
//!   smoothed LUT for quality-control reporting.

use crate::error::{LutError, LutResult};
use crate::Rgb;

// ---------------------------------------------------------------------------
// Gaussian kernel helpers
// ---------------------------------------------------------------------------

/// Build a 1-D Gaussian kernel of the given radius (half-width), normalised
/// so that it sums to 1.0.
///
/// The full kernel length is `2 * radius + 1`.
fn gaussian_kernel_1d(radius: usize, sigma: f64) -> Vec<f64> {
    let len = 2 * radius + 1;
    let mut kernel = Vec::with_capacity(len);
    let two_sigma_sq = 2.0 * sigma * sigma;
    for i in 0..len {
        let x = i as f64 - radius as f64;
        kernel.push((-x * x / two_sigma_sq).exp());
    }
    let sum: f64 = kernel.iter().sum();
    if sum > 1e-15 {
        for v in &mut kernel {
            *v /= sum;
        }
    }
    kernel
}

// ---------------------------------------------------------------------------
// Smoothing parameters
// ---------------------------------------------------------------------------

/// Parameters for LUT smoothing operations.
#[derive(Debug, Clone)]
pub struct SmootherParams {
    /// Gaussian kernel radius in LUT lattice units.  Larger values = more
    /// smoothing.  Clamped to `[1, size/2 - 1]` at runtime.
    pub radius: usize,
    /// Sigma for the Gaussian kernel.  Defaults to `radius as f64 / 2.0` when
    /// set to 0.0.
    pub sigma: f64,
    /// Strength of the contrast-preserving blend in `[0.0, 1.0]`.
    /// `0.0` = pure Gaussian, `1.0` = no smoothing at edges.
    pub contrast_preserve: f64,
    /// Local-contrast detection threshold: differences larger than this value
    /// (in normalised 0–1 space) are treated as edges.
    pub edge_threshold: f64,
}

impl Default for SmootherParams {
    fn default() -> Self {
        Self {
            radius: 1,
            sigma: 0.8,
            contrast_preserve: 0.5,
            edge_threshold: 0.05,
        }
    }
}

impl SmootherParams {
    /// Create smoothing params with a specific radius.  Sigma is derived
    /// automatically (`radius * 0.5 + 0.3`).
    #[must_use]
    pub fn with_radius(radius: usize) -> Self {
        let sigma = radius as f64 * 0.5 + 0.3;
        Self {
            radius,
            sigma,
            ..Default::default()
        }
    }

    /// Set sigma explicitly.
    #[must_use]
    pub fn sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self
    }

    /// Set contrast-preservation strength.
    #[must_use]
    pub fn contrast_preserve(mut self, v: f64) -> Self {
        self.contrast_preserve = v.clamp(0.0, 1.0);
        self
    }

    /// Set edge threshold.
    #[must_use]
    pub fn edge_threshold(mut self, v: f64) -> Self {
        self.edge_threshold = v.clamp(0.0, 1.0);
        self
    }
}

// ---------------------------------------------------------------------------
// Index arithmetic helpers
// ---------------------------------------------------------------------------

/// Flat index into a cube of side `size`.
#[inline]
fn flat_idx(r: usize, g: usize, b: usize, size: usize) -> usize {
    r * size * size + g * size + b
}

/// Mirror-clamped coordinate access.
#[inline]
fn clamp_coord(c: isize, size: usize) -> usize {
    c.clamp(0, size as isize - 1) as usize
}

// ---------------------------------------------------------------------------
// Gaussian smoothing
// ---------------------------------------------------------------------------

/// Apply isotropic 3-D Gaussian smoothing to a flat 3-D LUT.
///
/// The LUT is stored in `r-major` order: index = `r * size² + g * size + b`.
///
/// # Errors
///
/// Returns [`LutError::InvalidData`] if `lut.len() != size³` or `size < 2`.
pub fn smooth_gaussian(lut: &[Rgb], size: usize, params: &SmootherParams) -> LutResult<Vec<Rgb>> {
    validate_lut(lut, size)?;

    let sigma = if params.sigma < 1e-9 {
        params.radius as f64 / 2.0 + 0.1
    } else {
        params.sigma
    };
    let radius = params.radius.clamp(1, (size / 2).max(1));
    let kernel = gaussian_kernel_1d(radius, sigma);

    // Separable 3-D convolution: R pass → G pass → B pass.
    let tmp_r = convolve_axis(lut, size, &kernel, radius, Axis::R);
    let tmp_g = convolve_axis(&tmp_r, size, &kernel, radius, Axis::G);
    let result = convolve_axis(&tmp_g, size, &kernel, radius, Axis::B);

    Ok(result)
}

/// Axis selector for separable convolution.
#[derive(Clone, Copy)]
enum Axis {
    R,
    G,
    B,
}

/// Convolve a 3-D LUT along a single axis using a 1-D kernel.
fn convolve_axis(lut: &[Rgb], size: usize, kernel: &[f64], radius: usize, axis: Axis) -> Vec<Rgb> {
    let n = size * size * size;
    let mut out = vec![[0.0_f64; 3]; n];

    for r in 0..size {
        for g in 0..size {
            for b in 0..size {
                let mut acc = [0.0_f64; 3];
                for (ki, &kv) in kernel.iter().enumerate() {
                    let offset = ki as isize - radius as isize;
                    let (sr, sg, sb) = match axis {
                        Axis::R => (
                            clamp_coord(r as isize + offset, size),
                            g,
                            b,
                        ),
                        Axis::G => (
                            r,
                            clamp_coord(g as isize + offset, size),
                            b,
                        ),
                        Axis::B => (
                            r,
                            g,
                            clamp_coord(b as isize + offset, size),
                        ),
                    };
                    let src = lut[flat_idx(sr, sg, sb, size)];
                    acc[0] += src[0] * kv;
                    acc[1] += src[1] * kv;
                    acc[2] += src[2] * kv;
                }
                out[flat_idx(r, g, b, size)] = acc;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Contrast-preserving smoothing
// ---------------------------------------------------------------------------

/// Local contrast magnitude at a lattice point (max absolute difference with
/// 6-connected neighbours).
fn local_contrast(lut: &[Rgb], r: usize, g: usize, b: usize, size: usize) -> f64 {
    let center = lut[flat_idx(r, g, b, size)];
    let neighbours = [
        (r.saturating_sub(1), g, b),
        (r + 1, g, b),
        (r, g.saturating_sub(1), b),
        (r, g + 1, b),
        (r, g, b.saturating_sub(1)),
        (r, g, b + 1),
    ];
    let mut max_diff = 0.0_f64;
    for (nr, ng, nb) in neighbours {
        if nr >= size || ng >= size || nb >= size {
            continue;
        }
        let n = lut[flat_idx(nr, ng, nb, size)];
        let diff = ((center[0] - n[0]).powi(2)
            + (center[1] - n[1]).powi(2)
            + (center[2] - n[2]).powi(2))
        .sqrt();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    max_diff
}

/// Apply contrast-preserving Gaussian smoothing to a 3-D LUT.
///
/// At each lattice point the smoothed result is blended with the original
/// according to a local-contrast weight: high-contrast regions (edges) retain
/// more of the original; low-contrast regions are smoothed more aggressively.
///
/// # Errors
///
/// Returns [`LutError::InvalidData`] if `lut.len() != size³` or `size < 2`.
pub fn smooth_contrast_preserving(
    lut: &[Rgb],
    size: usize,
    params: &SmootherParams,
) -> LutResult<Vec<Rgb>> {
    validate_lut(lut, size)?;

    let smoothed = smooth_gaussian(lut, size, params)?;
    let cp = params.contrast_preserve.clamp(0.0, 1.0);
    let threshold = params.edge_threshold.max(1e-6);

    let mut out = vec![[0.0_f64; 3]; size * size * size];

    for r in 0..size {
        for g in 0..size {
            for b in 0..size {
                let idx = flat_idx(r, g, b, size);
                let contrast = local_contrast(lut, r, g, b, size);
                // edge_weight approaches 1 as contrast >> threshold
                let edge_weight = (contrast / threshold).min(1.0) * cp;
                let smooth_weight = 1.0 - edge_weight;

                let orig = lut[idx];
                let sm = smoothed[idx];
                out[idx] = [
                    orig[0] * edge_weight + sm[0] * smooth_weight,
                    orig[1] * edge_weight + sm[1] * smooth_weight,
                    orig[2] * edge_weight + sm[2] * smooth_weight,
                ];
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Difference analysis
// ---------------------------------------------------------------------------

/// Per-entry and aggregate difference statistics between two LUTs.
#[derive(Debug, Clone)]
pub struct SmootherDiff {
    /// Per-entry Euclidean distance in RGB space (0–1 normalised).
    pub per_entry: Vec<f64>,
    /// Root-mean-square difference.
    pub rms: f64,
    /// Maximum per-entry difference.
    pub max: f64,
    /// Mean per-entry difference.
    pub mean: f64,
    /// Percentage of entries whose difference exceeds `threshold`.
    pub pct_above_threshold: f64,
    /// Threshold used for `pct_above_threshold`.
    pub threshold: f64,
}

impl SmootherDiff {
    /// Compute the difference between `original` and `smoothed`.
    ///
    /// `threshold` is the per-entry distance above which a lattice point is
    /// counted as "significantly changed" (used for `pct_above_threshold`).
    ///
    /// # Errors
    ///
    /// Returns [`LutError::InvalidData`] if the slices have different lengths.
    pub fn compute(original: &[Rgb], smoothed: &[Rgb], threshold: f64) -> LutResult<Self> {
        if original.len() != smoothed.len() {
            return Err(LutError::InvalidData(format!(
                "Length mismatch: original={} smoothed={}",
                original.len(),
                smoothed.len(),
            )));
        }
        let n = original.len();
        let mut per_entry = Vec::with_capacity(n);
        let mut sum_sq = 0.0_f64;
        let mut max = 0.0_f64;
        let mut sum = 0.0_f64;
        let mut above = 0usize;

        for (a, b) in original.iter().zip(smoothed.iter()) {
            let d = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2))
                .sqrt();
            per_entry.push(d);
            sum_sq += d * d;
            sum += d;
            if d > max {
                max = d;
            }
            if d > threshold {
                above += 1;
            }
        }

        let rms = (sum_sq / n as f64).sqrt();
        let mean = sum / n as f64;
        let pct_above_threshold = above as f64 / n as f64 * 100.0;

        Ok(Self {
            per_entry,
            rms,
            max,
            mean,
            pct_above_threshold,
            threshold,
        })
    }

    /// Return the indices of entries whose difference exceeds `self.threshold`.
    #[must_use]
    pub fn outlier_indices(&self) -> Vec<usize> {
        self.per_entry
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if d > self.threshold { Some(i) } else { None })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Iterative smoothing
// ---------------------------------------------------------------------------

/// Apply Gaussian smoothing `iterations` times.
///
/// More iterations is roughly equivalent to a larger kernel but avoids
/// overly-wide kernels which can introduce ringing at boundaries.
///
/// # Errors
///
/// Returns [`LutError::InvalidData`] if `lut.len() != size³` or `size < 2`.
pub fn smooth_iterative(
    lut: &[Rgb],
    size: usize,
    params: &SmootherParams,
    iterations: usize,
) -> LutResult<Vec<Rgb>> {
    validate_lut(lut, size)?;
    if iterations == 0 {
        return Ok(lut.to_vec());
    }
    let mut current: Vec<Rgb> = lut.to_vec();
    for _ in 0..iterations {
        current = smooth_gaussian(&current, size, params)?;
    }
    Ok(current)
}

// ---------------------------------------------------------------------------
// Validation helper
// ---------------------------------------------------------------------------

fn validate_lut(lut: &[Rgb], size: usize) -> LutResult<()> {
    if size < 2 {
        return Err(LutError::InvalidData(format!(
            "LUT size must be >= 2, got {size}"
        )));
    }
    let expected = size * size * size;
    if lut.len() != expected {
        return Err(LutError::InvalidData(format!(
            "Expected {expected} entries for size {size}, got {}",
            lut.len(),
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity 3-D LUT.
    fn identity_lut(size: usize) -> Vec<Rgb> {
        let scale = (size - 1) as f64;
        let mut lut = Vec::with_capacity(size * size * size);
        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    lut.push([r as f64 / scale, g as f64 / scale, b as f64 / scale]);
                }
            }
        }
        lut
    }

    /// Constant LUT.
    fn constant_lut(size: usize, val: f64) -> Vec<Rgb> {
        vec![[val, val, val]; size * size * size]
    }

    fn rgb_close(a: &Rgb, b: &Rgb, tol: f64) -> bool {
        (a[0] - b[0]).abs() < tol && (a[1] - b[1]).abs() < tol && (a[2] - b[2]).abs() < tol
    }

    #[test]
    fn test_gaussian_smoothing_constant_lut() {
        // Smoothing a constant LUT should return the same values.
        let size = 5;
        let lut = constant_lut(size, 0.5);
        let params = SmootherParams::default();
        let smoothed = smooth_gaussian(&lut, size, &params).expect("should succeed");
        assert_eq!(smoothed.len(), lut.len());
        for (a, b) in lut.iter().zip(smoothed.iter()) {
            assert!(rgb_close(a, b, 1e-9));
        }
    }

    #[test]
    fn test_gaussian_smoothing_identity() {
        // Smoothing the identity LUT should not dramatically change corner values.
        // With mirror-clamped boundary the corners shift slightly toward the interior,
        // so we use a relaxed tolerance (0.15).
        let size = 5;
        let lut = identity_lut(size);
        let params = SmootherParams::with_radius(1);
        let smoothed = smooth_gaussian(&lut, size, &params).expect("should succeed");
        // Black corner should stay near black.
        assert!(
            rgb_close(&smoothed[0], &[0.0, 0.0, 0.0], 0.15),
            "black corner drifted: {:?}",
            smoothed[0]
        );
        // White corner should stay near white.
        let last = *smoothed.last().expect("non-empty");
        assert!(
            rgb_close(&last, &[1.0, 1.0, 1.0], 0.15),
            "white corner drifted: {:?}",
            last
        );
    }

    #[test]
    fn test_gaussian_smoothing_reduces_noise() {
        // A LUT with alternating values should become smoother.
        let size = 5;
        let mut lut = identity_lut(size);
        // Introduce checkerboard noise.
        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    let idx = flat_idx(r, g, b, size);
                    let noise = if (r + g + b) % 2 == 0 { 0.05 } else { -0.05 };
                    lut[idx][0] = (lut[idx][0] + noise).clamp(0.0, 1.0);
                    lut[idx][1] = (lut[idx][1] + noise).clamp(0.0, 1.0);
                    lut[idx][2] = (lut[idx][2] + noise).clamp(0.0, 1.0);
                }
            }
        }
        let params = SmootherParams::with_radius(1);
        let smoothed = smooth_gaussian(&lut, size, &params).expect("should succeed");
        // Compute mean absolute difference vs identity.
        let ident = identity_lut(size);
        let diff_orig: f64 = lut
            .iter()
            .zip(ident.iter())
            .map(|(a, b)| (a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs())
            .sum::<f64>()
            / lut.len() as f64;
        let diff_smooth: f64 = smoothed
            .iter()
            .zip(ident.iter())
            .map(|(a, b)| (a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs())
            .sum::<f64>()
            / lut.len() as f64;
        assert!(
            diff_smooth < diff_orig,
            "smoothed diff {diff_smooth} should be less than orig diff {diff_orig}"
        );
    }

    #[test]
    fn test_invalid_size() {
        let lut = vec![[0.0; 3]; 4];
        let params = SmootherParams::default();
        assert!(smooth_gaussian(&lut, 1, &params).is_err());
    }

    #[test]
    fn test_invalid_length() {
        let lut = vec![[0.0; 3]; 10]; // wrong length for size=3
        let params = SmootherParams::default();
        assert!(smooth_gaussian(&lut, 3, &params).is_err());
    }

    #[test]
    fn test_contrast_preserving_constant() {
        // Constant LUT must stay constant regardless of preserve setting.
        let size = 5;
        let lut = constant_lut(size, 0.3);
        let params = SmootherParams::default().contrast_preserve(0.9);
        let smoothed = smooth_contrast_preserving(&lut, size, &params).expect("should succeed");
        for (a, b) in lut.iter().zip(smoothed.iter()) {
            assert!(rgb_close(a, b, 1e-9));
        }
    }

    #[test]
    fn test_contrast_preserving_strong_edge() {
        // At full contrast preserve, output should be close to original at edges.
        let size = 3;
        let mut lut = constant_lut(size, 0.0);
        // Make one entry a sharp spike.
        lut[flat_idx(1, 1, 1, size)] = [1.0, 1.0, 1.0];

        let params = SmootherParams::default()
            .contrast_preserve(1.0)
            .edge_threshold(0.01);
        let smoothed = smooth_contrast_preserving(&lut, size, &params).expect("should succeed");
        // The spike entry should be mostly preserved.
        let idx = flat_idx(1, 1, 1, size);
        assert!(smoothed[idx][0] > 0.5, "edge should be mostly preserved");
    }

    #[test]
    fn test_diff_compute_identity() {
        let size = 4;
        let lut = identity_lut(size);
        let diff = SmootherDiff::compute(&lut, &lut, 0.01).expect("should succeed");
        assert!(diff.rms < 1e-12);
        assert!(diff.max < 1e-12);
        assert_eq!(diff.outlier_indices().len(), 0);
    }

    #[test]
    fn test_diff_compute_mismatch() {
        let a = constant_lut(3, 0.0);
        let b = constant_lut(4, 0.0);
        assert!(SmootherDiff::compute(&a, &b, 0.01).is_err());
    }

    #[test]
    fn test_diff_pct_above_threshold() {
        let a = constant_lut(3, 0.0);
        let b = constant_lut(3, 0.5);
        let diff = SmootherDiff::compute(&a, &b, 0.1).expect("should succeed");
        assert!((diff.pct_above_threshold - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_iterative_smooth_zero_iters() {
        let size = 3;
        let lut = identity_lut(size);
        let params = SmootherParams::default();
        let result = smooth_iterative(&lut, size, &params, 0).expect("should succeed");
        // Zero iterations → identical.
        for (a, b) in lut.iter().zip(result.iter()) {
            assert!(rgb_close(a, b, 1e-12));
        }
    }

    #[test]
    fn test_iterative_smooth_more_iters() {
        // Multiple iterations should increase smoothness vs single iteration.
        let size = 5;
        let mut noisy = identity_lut(size);
        for entry in &mut noisy {
            entry[0] = (entry[0] + 0.08).min(1.0);
        }
        let params = SmootherParams::with_radius(1);
        let s1 = smooth_iterative(&noisy, size, &params, 1).expect("ok");
        let s3 = smooth_iterative(&noisy, size, &params, 3).expect("ok");
        // s3 should differ more from noisy than s1 (more smoothed).
        let diff1: f64 = noisy
            .iter()
            .zip(s1.iter())
            .map(|(a, b)| (a[0] - b[0]).abs())
            .sum();
        let diff3: f64 = noisy
            .iter()
            .zip(s3.iter())
            .map(|(a, b)| (a[0] - b[0]).abs())
            .sum();
        assert!(diff3 >= diff1);
    }

    #[test]
    fn test_gaussian_kernel_sums_to_one() {
        let k = gaussian_kernel_1d(2, 1.0);
        let sum: f64 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }
}
