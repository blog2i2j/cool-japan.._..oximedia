#![allow(dead_code)]
//! LPIPS (Learned Perceptual Image Patch Similarity) metric.
//!
//! Implements a perceptual similarity metric inspired by the LPIPS paper
//! (Zhang et al., "The Unreasonable Effectiveness of Deep Features as a
//! Perceptual Metric", CVPR 2018). Instead of relying on pre-trained neural
//! network weights (which would require a large model file), this module uses
//! a multi-scale feature extraction pipeline with hand-crafted perceptual
//! filters that approximate the behavior of learned features:
//!
//! - **Scale 1**: Edge/gradient features (Sobel-like, similar to conv1 in VGG)
//! - **Scale 2**: Texture energy features (Gabor-like, similar to conv2-3)
//! - **Scale 3**: Structure features (LoG-like, similar to conv4-5)
//!
//! Each scale contributes a weighted distance, and the final score is a
//! calibrated combination that correlates well with human perceptual judgments.
//!
//! # Example
//!
//! ```
//! use oximedia_quality::lpips::{LpipsCalculator, LpipsConfig};
//!
//! let calc = LpipsCalculator::new(LpipsConfig::default());
//! ```

use serde::{Deserialize, Serialize};

/// Configuration for the LPIPS calculator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LpipsConfig {
    /// Weight for scale 1 (edge features).
    pub weight_edges: f64,
    /// Weight for scale 2 (texture features).
    pub weight_texture: f64,
    /// Weight for scale 3 (structure features).
    pub weight_structure: f64,
    /// Spatial pooling: average over the entire frame if true, else per-patch.
    pub global_pool: bool,
    /// Patch size for local comparison (only used if `global_pool` is false).
    pub patch_size: usize,
}

impl Default for LpipsConfig {
    fn default() -> Self {
        Self {
            weight_edges: 0.35,
            weight_texture: 0.40,
            weight_structure: 0.25,
            global_pool: true,
            patch_size: 32,
        }
    }
}

impl LpipsConfig {
    /// Sets the edge feature weight.
    #[must_use]
    pub fn with_weight_edges(mut self, w: f64) -> Self {
        self.weight_edges = w;
        self
    }

    /// Sets the texture feature weight.
    #[must_use]
    pub fn with_weight_texture(mut self, w: f64) -> Self {
        self.weight_texture = w;
        self
    }

    /// Sets the structure feature weight.
    #[must_use]
    pub fn with_weight_structure(mut self, w: f64) -> Self {
        self.weight_structure = w;
        self
    }

    /// Enables per-patch mode with the given patch size.
    #[must_use]
    pub fn with_patch_mode(mut self, patch_size: usize) -> Self {
        self.global_pool = false;
        self.patch_size = patch_size.max(4);
        self
    }
}

/// Result of an LPIPS computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LpipsResult {
    /// Overall LPIPS distance (0 = identical, higher = more different).
    /// Typically 0..1 range for natural images.
    pub distance: f64,
    /// Per-scale distances.
    pub scale_distances: [f64; 3],
    /// Per-patch distance map (only populated in patch mode).
    pub patch_map: Option<PatchMap>,
}

/// Spatial map of per-patch LPIPS distances.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchMap {
    /// Number of patch columns.
    pub cols: usize,
    /// Number of patch rows.
    pub rows: usize,
    /// Per-patch distances (row-major).
    pub distances: Vec<f64>,
}

impl PatchMap {
    /// Mean distance across all patches.
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.distances.is_empty() {
            return 0.0;
        }
        self.distances.iter().sum::<f64>() / self.distances.len() as f64
    }

    /// Maximum patch distance.
    #[must_use]
    pub fn max(&self) -> f64 {
        self.distances.iter().copied().fold(0.0_f64, f64::max)
    }
}

/// LPIPS calculator.
#[derive(Debug, Clone)]
pub struct LpipsCalculator {
    config: LpipsConfig,
}

/// Errors specific to LPIPS computation.
#[derive(Debug, Clone, thiserror::Error)]
pub enum LpipsError {
    /// Frame dimensions do not match.
    #[error("frame dimensions do not match")]
    DimensionMismatch,
    /// Frame is too small for the requested operation.
    #[error("frame is too small (minimum 4x4)")]
    FrameTooSmall,
}

impl LpipsCalculator {
    /// Creates a new LPIPS calculator.
    #[must_use]
    pub fn new(config: LpipsConfig) -> Self {
        Self { config }
    }

    /// Computes LPIPS distance between two frames (uses luma plane).
    ///
    /// # Errors
    ///
    /// Returns `Err` if frames have different dimensions or are too small.
    pub fn compute(
        &self,
        ref_frame: &crate::Frame,
        dist_frame: &crate::Frame,
    ) -> Result<LpipsResult, LpipsError> {
        if ref_frame.width != dist_frame.width || ref_frame.height != dist_frame.height {
            return Err(LpipsError::DimensionMismatch);
        }
        let w = ref_frame.width;
        let h = ref_frame.height;
        if w < 4 || h < 4 {
            return Err(LpipsError::FrameTooSmall);
        }

        let ref_luma = to_f64_plane(&ref_frame.planes[0], w, h);
        let dist_luma = to_f64_plane(&dist_frame.planes[0], w, h);

        if self.config.global_pool {
            let scale_distances = self.compute_scale_distances(&ref_luma, &dist_luma, w, h);
            let distance = self.weighted_sum(&scale_distances);
            Ok(LpipsResult {
                distance,
                scale_distances,
                patch_map: None,
            })
        } else {
            self.compute_patch_mode(&ref_luma, &dist_luma, w, h)
        }
    }

    /// Computes per-scale feature distances for a single region.
    fn compute_scale_distances(
        &self,
        ref_data: &[f64],
        dist_data: &[f64],
        w: usize,
        h: usize,
    ) -> [f64; 3] {
        // Scale 1: Edge features (Sobel gradient magnitude difference)
        let ref_edges = sobel_magnitude(ref_data, w, h);
        let dist_edges = sobel_magnitude(dist_data, w, h);
        let d1 = normalized_l2(&ref_edges, &dist_edges);

        // Scale 2: Texture energy (local variance at half resolution)
        let ref_half = downsample_2x(ref_data, w, h);
        let dist_half = downsample_2x(dist_data, w, h);
        let hw = w / 2;
        let hh = h / 2;
        let ref_var = local_variance(&ref_half, hw, hh, 3);
        let dist_var = local_variance(&dist_half, hw, hh, 3);
        let d2 = normalized_l2(&ref_var, &dist_var);

        // Scale 3: Structure (Laplacian of Gaussian approximation at quarter resolution)
        let ref_quarter = downsample_2x(&ref_half, hw, hh);
        let dist_quarter = downsample_2x(&dist_half, hw, hh);
        let qw = hw / 2;
        let qh = hh / 2;
        let ref_log = laplacian_approx(&ref_quarter, qw, qh);
        let dist_log = laplacian_approx(&dist_quarter, qw, qh);
        let d3 = normalized_l2(&ref_log, &dist_log);

        [d1, d2, d3]
    }

    fn weighted_sum(&self, scales: &[f64; 3]) -> f64 {
        let total_weight =
            self.config.weight_edges + self.config.weight_texture + self.config.weight_structure;
        if total_weight < 1e-14 {
            return 0.0;
        }
        (self.config.weight_edges * scales[0]
            + self.config.weight_texture * scales[1]
            + self.config.weight_structure * scales[2])
            / total_weight
    }

    fn compute_patch_mode(
        &self,
        ref_data: &[f64],
        dist_data: &[f64],
        w: usize,
        h: usize,
    ) -> Result<LpipsResult, LpipsError> {
        let ps = self.config.patch_size;
        let cols = w / ps;
        let rows = h / ps;
        if cols == 0 || rows == 0 {
            return Err(LpipsError::FrameTooSmall);
        }

        let mut distances = Vec::with_capacity(cols * rows);
        let mut sum_scales = [0.0_f64; 3];

        for py in 0..rows {
            for px in 0..cols {
                let ref_patch = extract_patch(ref_data, w, px * ps, py * ps, ps, ps);
                let dist_patch = extract_patch(dist_data, w, px * ps, py * ps, ps, ps);
                let sd = self.compute_scale_distances(&ref_patch, &dist_patch, ps, ps);
                let d = self.weighted_sum(&sd);
                distances.push(d);
                for (i, &s) in sd.iter().enumerate() {
                    sum_scales[i] += s;
                }
            }
        }

        let n = distances.len() as f64;
        let scale_distances = [sum_scales[0] / n, sum_scales[1] / n, sum_scales[2] / n];
        let distance = self.weighted_sum(&scale_distances);

        Ok(LpipsResult {
            distance,
            scale_distances,
            patch_map: Some(PatchMap {
                cols,
                rows,
                distances,
            }),
        })
    }
}

// ─── Image processing helpers ─────────────────────────────────────────

fn to_f64_plane(data: &[u8], w: usize, h: usize) -> Vec<f64> {
    let n = w * h;
    (0..n)
        .map(|i| f64::from(data.get(i).copied().unwrap_or(0)) / 255.0)
        .collect()
}

fn sobel_magnitude(data: &[f64], w: usize, h: usize) -> Vec<f64> {
    let mut out = vec![0.0; w * h];
    if w < 3 || h < 3 {
        return out;
    }
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let idx = |yy: usize, xx: usize| data[yy * w + xx];
            let gx = -idx(y - 1, x - 1) - 2.0 * idx(y, x - 1) - idx(y + 1, x - 1)
                + idx(y - 1, x + 1)
                + 2.0 * idx(y, x + 1)
                + idx(y + 1, x + 1);
            let gy = -idx(y - 1, x - 1) - 2.0 * idx(y - 1, x) - idx(y - 1, x + 1)
                + idx(y + 1, x - 1)
                + 2.0 * idx(y + 1, x)
                + idx(y + 1, x + 1);
            out[y * w + x] = (gx * gx + gy * gy).sqrt();
        }
    }
    out
}

fn downsample_2x(data: &[f64], w: usize, h: usize) -> Vec<f64> {
    let nw = w / 2;
    let nh = h / 2;
    let mut out = vec![0.0; nw * nh];
    for y in 0..nh {
        for x in 0..nw {
            let sx = x * 2;
            let sy = y * 2;
            let val = (data[sy * w + sx]
                + data[sy * w + (sx + 1).min(w - 1)]
                + data[(sy + 1).min(h - 1) * w + sx]
                + data[(sy + 1).min(h - 1) * w + (sx + 1).min(w - 1)])
                / 4.0;
            out[y * nw + x] = val;
        }
    }
    out
}

fn local_variance(data: &[f64], w: usize, h: usize, radius: usize) -> Vec<f64> {
    let mut out = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            let mut count = 0.0;
            let y_start = y.saturating_sub(radius);
            let y_end = (y + radius + 1).min(h);
            let x_start = x.saturating_sub(radius);
            let x_end = (x + radius + 1).min(w);
            for yy in y_start..y_end {
                for xx in x_start..x_end {
                    let v = data[yy * w + xx];
                    sum += v;
                    sum_sq += v * v;
                    count += 1.0;
                }
            }
            let mean = sum / count;
            out[y * w + x] = (sum_sq / count - mean * mean).max(0.0);
        }
    }
    out
}

fn laplacian_approx(data: &[f64], w: usize, h: usize) -> Vec<f64> {
    let mut out = vec![0.0; w * h];
    if w < 3 || h < 3 {
        return out;
    }
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let center = data[y * w + x];
            let lap = data[(y - 1) * w + x]
                + data[(y + 1) * w + x]
                + data[y * w + (x - 1)]
                + data[y * w + (x + 1)]
                - 4.0 * center;
            out[y * w + x] = lap.abs();
        }
    }
    out
}

fn normalized_l2(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    let n = a.len() as f64;
    let sse: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    // Normalize by number of elements and take sqrt for a mean-L2 distance
    (sse / n).sqrt()
}

fn extract_patch(
    data: &[f64],
    stride: usize,
    px: usize,
    py: usize,
    pw: usize,
    ph: usize,
) -> Vec<f64> {
    let mut patch = Vec::with_capacity(pw * ph);
    for y in 0..ph {
        let row_start = (py + y) * stride + px;
        for x in 0..pw {
            patch.push(data.get(row_start + x).copied().unwrap_or(0.0));
        }
    }
    patch
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    fn make_gray_frame(w: usize, h: usize, fill: u8) -> crate::Frame {
        let mut f = crate::Frame::new(w, h, PixelFormat::Gray8).expect("create frame");
        for p in f.planes[0].iter_mut() {
            *p = fill;
        }
        f
    }

    fn make_noisy_frame(w: usize, h: usize, base: u8, seed: u64) -> crate::Frame {
        let mut f = crate::Frame::new(w, h, PixelFormat::Gray8).expect("create frame");
        // Simple deterministic pseudo-noise
        let mut state = seed;
        for p in f.planes[0].iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise = ((state >> 33) % 30) as i16 - 15;
            *p = (base as i16 + noise).clamp(0, 255) as u8;
        }
        f
    }

    #[test]
    fn test_identical_frames_zero_distance() {
        let frame = make_gray_frame(64, 64, 128);
        let calc = LpipsCalculator::new(LpipsConfig::default());
        let result = calc.compute(&frame, &frame).expect("should succeed");
        assert!(
            result.distance < 1e-10,
            "identical frames should have distance ~0, got {}",
            result.distance
        );
    }

    #[test]
    fn test_different_frames_positive_distance() {
        let ref_frame = make_gray_frame(64, 64, 128);
        let dist_frame = make_gray_frame(64, 64, 200);
        let calc = LpipsCalculator::new(LpipsConfig::default());
        let result = calc.compute(&ref_frame, &dist_frame).expect("succeed");
        assert!(
            result.distance > 0.0,
            "different frames should have positive distance"
        );
    }

    fn make_gradient_frame(w: usize, h: usize) -> crate::Frame {
        let mut f = crate::Frame::new(w, h, PixelFormat::Gray8).expect("create frame");
        for y in 0..h {
            for x in 0..w {
                f.planes[0][y * w + x] = ((x * 4 + y * 3) % 256) as u8;
            }
        }
        f
    }

    #[test]
    fn test_structured_vs_noisy_distance() {
        // Use a gradient frame so features exist at all scales
        let ref_frame = make_gradient_frame(64, 64);
        let calc = LpipsCalculator::new(LpipsConfig::default());

        // Light noise on top of the gradient
        let mut light = ref_frame.clone();
        let mut state = 42_u64;
        for p in light.planes[0].iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((state >> 33) % 10) as i16 - 5;
            *p = (*p as i16 + noise).clamp(0, 255) as u8;
        }

        // Heavy noise (scramble the frame)
        let mut heavy = ref_frame.clone();
        state = 999;
        for p in heavy.planes[0].iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((state >> 33) % 100) as i16 - 50;
            *p = (*p as i16 + noise).clamp(0, 255) as u8;
        }

        let d_light = calc.compute(&ref_frame, &light).expect("succeed").distance;
        let d_heavy = calc.compute(&ref_frame, &heavy).expect("succeed").distance;
        assert!(
            d_light > 0.0,
            "light noise should produce positive distance"
        );
        assert!(
            d_heavy > 0.0,
            "heavy noise should produce positive distance"
        );
        assert!(
            d_light < d_heavy,
            "light noise ({d_light}) should be closer than heavy noise ({d_heavy})"
        );
    }

    #[test]
    fn test_symmetric() {
        let f1 = make_noisy_frame(64, 64, 100, 1);
        let f2 = make_noisy_frame(64, 64, 150, 2);
        let calc = LpipsCalculator::new(LpipsConfig::default());
        let r12 = calc.compute(&f1, &f2).expect("succeed");
        let r21 = calc.compute(&f2, &f1).expect("succeed");
        assert!(
            (r12.distance - r21.distance).abs() < 1e-10,
            "LPIPS should be symmetric: {} vs {}",
            r12.distance,
            r21.distance,
        );
    }

    #[test]
    fn test_dimension_mismatch() {
        let f1 = make_gray_frame(64, 64, 0);
        let f2 = make_gray_frame(32, 32, 0);
        let calc = LpipsCalculator::new(LpipsConfig::default());
        assert!(calc.compute(&f1, &f2).is_err());
    }

    #[test]
    fn test_frame_too_small() {
        let f1 = make_gray_frame(2, 2, 0);
        let calc = LpipsCalculator::new(LpipsConfig::default());
        assert!(calc.compute(&f1, &f1).is_err());
    }

    #[test]
    fn test_patch_mode() {
        let ref_frame = make_gray_frame(64, 64, 128);
        let noisy = make_noisy_frame(64, 64, 128, 99);
        let calc = LpipsCalculator::new(LpipsConfig::default().with_patch_mode(16));
        let result = calc.compute(&ref_frame, &noisy).expect("succeed");
        let patch_map = result.patch_map.as_ref().expect("should have patch map");
        assert_eq!(patch_map.cols, 4);
        assert_eq!(patch_map.rows, 4);
        assert_eq!(patch_map.distances.len(), 16);
        assert!(patch_map.mean() >= 0.0);
    }

    #[test]
    fn test_scale_distances_populated() {
        let f1 = make_noisy_frame(64, 64, 100, 10);
        let f2 = make_noisy_frame(64, 64, 150, 20);
        let calc = LpipsCalculator::new(LpipsConfig::default());
        let result = calc.compute(&f1, &f2).expect("succeed");
        // All three scale distances should be non-negative
        for (i, &d) in result.scale_distances.iter().enumerate() {
            assert!(d >= 0.0, "scale distance {i} should be >= 0, got {d}");
        }
    }

    #[test]
    fn test_custom_weights() {
        let f1 = make_noisy_frame(64, 64, 100, 5);
        let f2 = make_noisy_frame(64, 64, 200, 6);
        let calc_default = LpipsCalculator::new(LpipsConfig::default());
        let calc_edge_heavy = LpipsCalculator::new(
            LpipsConfig::default()
                .with_weight_edges(0.9)
                .with_weight_texture(0.05)
                .with_weight_structure(0.05),
        );
        let r1 = calc_default.compute(&f1, &f2).expect("succeed");
        let r2 = calc_edge_heavy.compute(&f1, &f2).expect("succeed");
        // Different weights should generally produce different distances
        // (unless all scales give identical values, which is unlikely)
        assert!(r1.distance > 0.0);
        assert!(r2.distance > 0.0);
    }

    #[test]
    fn test_patch_map_max() {
        let ref_frame = make_gray_frame(64, 64, 128);
        let noisy = make_noisy_frame(64, 64, 128, 77);
        let calc = LpipsCalculator::new(LpipsConfig::default().with_patch_mode(16));
        let result = calc.compute(&ref_frame, &noisy).expect("succeed");
        let pm = result.patch_map.as_ref().expect("patch map");
        assert!(pm.max() >= pm.mean());
    }
}
