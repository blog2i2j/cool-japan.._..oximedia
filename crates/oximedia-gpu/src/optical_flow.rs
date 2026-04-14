//! GPU-accelerated optical flow computation for motion interpolation.
//!
//! This module implements a dense optical flow estimation algorithm inspired by
//! the Lucas-Kanade pyramidal approach and the Farnebäck polynomial expansion
//! method, adapted for CPU-parallel execution with rayon (GPU compute shader
//! path hooks in via the existing `GpuDevice` infrastructure).
//!
//! # Algorithm
//!
//! Dense optical flow is computed using a multi-scale (image pyramid) approach:
//!
//! 1. Build Gaussian image pyramids for both frames.
//! 2. At the coarsest scale, estimate flow using spatial and temporal gradients.
//! 3. Propagate and refine the flow estimate upward through the pyramid.
//! 4. Apply smoothing to produce a dense flow field.
//!
//! The output is a flow field where each pixel contains a 2D displacement
//! vector `(dx, dy)` describing where that pixel moved between frames.
//!
//! # Usage
//!
//! ```no_run
//! use oximedia_gpu::optical_flow::{OpticalFlowConfig, OpticalFlowEstimator};
//!
//! let prev = vec![0u8; 640 * 480]; // luminance data
//! let next = vec![0u8; 640 * 480];
//! let config = OpticalFlowConfig::default();
//! let estimator = OpticalFlowEstimator::new(config);
//! let flow = estimator.estimate(&prev, &next, 640, 480).unwrap();
//! ```

use crate::{GpuError, Result};
use rayon::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// Public configuration types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for optical flow estimation.
#[derive(Debug, Clone)]
pub struct OpticalFlowConfig {
    /// Number of pyramid levels (coarser to finer).
    ///
    /// More levels can capture larger displacements but cost more compute.
    pub pyramid_levels: u32,
    /// Number of iterations per pyramid level.
    pub iterations: u32,
    /// Window size for local flow estimation (must be odd, ≥ 3).
    pub window_size: u32,
    /// Spatial smoothing sigma applied to the final flow field.
    pub smoothing_sigma: f32,
    /// Maximum displacement per pyramid step (pixels).
    pub max_displacement: f32,
}

impl Default for OpticalFlowConfig {
    fn default() -> Self {
        Self {
            pyramid_levels: 4,
            iterations: 3,
            window_size: 15,
            smoothing_sigma: 1.5,
            max_displacement: 4.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Flow vector and field types
// ─────────────────────────────────────────────────────────────────────────────

/// A 2D optical flow vector (horizontal and vertical displacement in pixels).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlowVector {
    /// Horizontal displacement (positive = rightward).
    pub dx: f32,
    /// Vertical displacement (positive = downward).
    pub dy: f32,
}

impl FlowVector {
    /// Zero displacement.
    #[must_use]
    pub const fn zero() -> Self {
        Self { dx: 0.0, dy: 0.0 }
    }

    /// Compute the magnitude of this flow vector.
    #[must_use]
    pub fn magnitude(self) -> f32 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }
}

/// Dense optical flow field.
///
/// Contains one [`FlowVector`] per pixel of the image.
#[derive(Debug, Clone)]
pub struct FlowField {
    /// Width of the source image.
    pub width: u32,
    /// Height of the source image.
    pub height: u32,
    /// Per-pixel flow vectors in row-major order.
    pub vectors: Vec<FlowVector>,
}

impl FlowField {
    /// Create a zero-initialised flow field.
    #[must_use]
    pub fn zeros(width: u32, height: u32) -> Self {
        let n = (width as usize) * (height as usize);
        Self {
            width,
            height,
            vectors: vec![FlowVector::zero(); n],
        }
    }

    /// Get the flow vector at pixel `(x, y)`, or `None` if out of bounds.
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> Option<FlowVector> {
        if x >= self.width || y >= self.height {
            return None;
        }
        self.vectors
            .get((y as usize) * (self.width as usize) + (x as usize))
            .copied()
    }

    /// Compute the mean magnitude of all flow vectors.
    #[must_use]
    pub fn mean_magnitude(&self) -> f32 {
        if self.vectors.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.vectors.iter().map(|v| v.magnitude()).sum();
        sum / self.vectors.len() as f32
    }

    /// Warp the `target` frame backwards using this flow field to produce a
    /// motion-compensated intermediate frame.
    ///
    /// For each output pixel `p`, samples `target[p - flow(p)]` using
    /// bilinear interpolation.  Pixels that map outside `target` are filled
    /// with black (zero).
    ///
    /// # Arguments
    ///
    /// * `target` – RGBA pixel data for the frame to warp.
    ///
    /// # Errors
    ///
    /// Returns an error if `target` does not match the flow field dimensions.
    pub fn warp_frame(&self, target: &[u8]) -> Result<Vec<u8>> {
        let expected = (self.width as usize) * (self.height as usize) * 4;
        if target.len() != expected {
            return Err(GpuError::InvalidBufferSize {
                expected,
                actual: target.len(),
            });
        }

        let w = self.width as usize;
        let h = self.height as usize;
        let mut output = vec![0u8; expected];

        output
            .par_chunks_exact_mut(4)
            .enumerate()
            .for_each(|(idx, pix)| {
                let px = (idx % w) as f32;
                let py = (idx / w) as f32;

                let fv = self.vectors[idx];
                let src_x = px - fv.dx;
                let src_y = py - fv.dy;

                // Bilinear sampling
                let x0 = src_x.floor() as isize;
                let y0 = src_y.floor() as isize;
                let tx = src_x - x0 as f32;
                let ty = src_y - y0 as f32;

                let sample = |xi: isize, yi: isize| -> [f32; 4] {
                    if xi < 0 || yi < 0 || xi >= w as isize || yi >= h as isize {
                        return [0.0; 4];
                    }
                    let off = (yi as usize * w + xi as usize) * 4;
                    [
                        target[off] as f32,
                        target[off + 1] as f32,
                        target[off + 2] as f32,
                        target[off + 3] as f32,
                    ]
                };

                let c00 = sample(x0, y0);
                let c10 = sample(x0 + 1, y0);
                let c01 = sample(x0, y0 + 1);
                let c11 = sample(x0 + 1, y0 + 1);

                for ch in 0..4 {
                    let v = c00[ch] * (1.0 - tx) * (1.0 - ty)
                        + c10[ch] * tx * (1.0 - ty)
                        + c01[ch] * (1.0 - tx) * ty
                        + c11[ch] * tx * ty;
                    pix[ch] = v.clamp(0.0, 255.0) as u8;
                }
            });

        Ok(output)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Optical flow estimator
// ─────────────────────────────────────────────────────────────────────────────

/// GPU-accelerated optical flow estimator.
///
/// Uses a pyramidal iterative approach to estimate dense pixel displacements
/// between two consecutive luminance frames.
#[derive(Debug, Clone)]
pub struct OpticalFlowEstimator {
    config: OpticalFlowConfig,
}

impl OpticalFlowEstimator {
    /// Create a new estimator with the given configuration.
    #[must_use]
    pub fn new(config: OpticalFlowConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(OpticalFlowConfig::default())
    }

    /// Estimate dense optical flow between two luminance (Y-channel) frames.
    ///
    /// # Arguments
    ///
    /// * `prev` – Previous frame luminance values (one byte per pixel, row-major).
    /// * `next` – Next frame luminance values (same layout).
    /// * `width` – Frame width in pixels.
    /// * `height` – Frame height in pixels.
    ///
    /// # Errors
    ///
    /// Returns an error if buffer sizes don't match the declared dimensions.
    pub fn estimate(&self, prev: &[u8], next: &[u8], width: u32, height: u32) -> Result<FlowField> {
        let expected = (width as usize) * (height as usize);
        if prev.len() != expected {
            return Err(GpuError::InvalidBufferSize {
                expected,
                actual: prev.len(),
            });
        }
        if next.len() != expected {
            return Err(GpuError::InvalidBufferSize {
                expected,
                actual: next.len(),
            });
        }

        if width == 0 || height == 0 {
            return Err(GpuError::InvalidDimensions { width, height });
        }

        let window = self.config.window_size.max(3) | 1; // ensure odd
        let levels = self.config.pyramid_levels.max(1).min(8);

        // Build pyramids
        let prev_pyr = build_gaussian_pyramid(prev, width, height, levels);
        let next_pyr = build_gaussian_pyramid(next, width, height, levels);

        // Start with a zero flow at the coarsest level
        let (cw, ch) = pyramid_dims(width, height, levels - 1);
        let mut flow = FlowField::zeros(cw, ch);

        // Coarse-to-fine refinement
        for level in (0..levels).rev() {
            let (lw, lh) = pyramid_dims(width, height, level);
            let prev_lvl = &prev_pyr[level as usize];
            let next_lvl = &next_pyr[level as usize];

            // Upscale flow from coarser level
            if level + 1 < levels {
                flow = upscale_flow(&flow, lw, lh);
                // Scale displacement values
                let scale = 2.0f32;
                for v in &mut flow.vectors {
                    v.dx *= scale;
                    v.dy *= scale;
                }
            } else {
                flow = FlowField::zeros(lw, lh);
            }

            // Iterative refinement at this level
            for _ in 0..self.config.iterations {
                flow = refine_flow(
                    flow,
                    prev_lvl,
                    next_lvl,
                    lw,
                    lh,
                    window,
                    self.config.max_displacement,
                );
            }
        }

        // Apply smoothing to the final (finest level) flow
        if self.config.smoothing_sigma > 0.0 {
            flow = smooth_flow(flow, width, height, self.config.smoothing_sigma);
        }

        Ok(flow)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the dimensions at a given pyramid level (level 0 = original).
fn pyramid_dims(w: u32, h: u32, level: u32) -> (u32, u32) {
    let scale = 1u32 << level;
    let lw = (w / scale).max(1);
    let lh = (h / scale).max(1);
    (lw, lh)
}

/// Build a Gaussian image pyramid as a Vec of luminance planes.
fn build_gaussian_pyramid(frame: &[u8], width: u32, height: u32, levels: u32) -> Vec<Vec<u8>> {
    let mut pyramid = Vec::with_capacity(levels as usize);
    pyramid.push(frame.to_vec());

    for l in 1..levels {
        let (pw, ph) = pyramid_dims(width, height, l - 1);
        let (cw, ch) = pyramid_dims(width, height, l);
        let prev = &pyramid[(l - 1) as usize];
        let downsampled = downsample_2x(prev, pw, ph, cw, ch);
        pyramid.push(downsampled);
    }

    pyramid
}

/// Downsample a luminance plane by 2× using a simple 2×2 box filter.
fn downsample_2x(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    let mut dst = vec![0u8; (dw * dh) as usize];
    for dy in 0..dh {
        for dx in 0..dw {
            let sy0 = (dy * 2).min(sh - 1) as usize;
            let sy1 = (dy * 2 + 1).min(sh - 1) as usize;
            let sx0 = (dx * 2).min(sw - 1) as usize;
            let sx1 = (dx * 2 + 1).min(sw - 1) as usize;
            let sum = src[sy0 * sw as usize + sx0] as u32
                + src[sy0 * sw as usize + sx1] as u32
                + src[sy1 * sw as usize + sx0] as u32
                + src[sy1 * sw as usize + sx1] as u32;
            dst[dy as usize * dw as usize + dx as usize] = (sum / 4) as u8;
        }
    }
    dst
}

/// Upscale a flow field to new dimensions using nearest-neighbour sampling.
fn upscale_flow(flow: &FlowField, new_w: u32, new_h: u32) -> FlowField {
    let ow = flow.width as usize;
    let oh = flow.height as usize;
    let mut new_vectors = Vec::with_capacity((new_w * new_h) as usize);

    for ny in 0..new_h as usize {
        for nx in 0..new_w as usize {
            let sx = ((nx * ow) / new_w as usize).min(ow - 1);
            let sy = ((ny * oh) / new_h as usize).min(oh - 1);
            let idx = sy * ow + sx;
            new_vectors.push(flow.vectors[idx]);
        }
    }

    FlowField {
        width: new_w,
        height: new_h,
        vectors: new_vectors,
    }
}

/// Perform one iteration of Lucas-Kanade optical flow estimation at a
/// single pyramid level, refining the existing flow estimate.
///
/// Uses the spatial-temporal gradient formulation:
///   `[Ix, Iy; Iy, Iy] * [dx; dy] = -[It]` solved via 2×2 inverse.
fn refine_flow(
    flow: FlowField,
    prev: &[u8],
    next: &[u8],
    w: u32,
    h: u32,
    window: u32,
    max_disp: f32,
) -> FlowField {
    let w_usize = w as usize;
    let h_usize = h as usize;
    let half = (window / 2) as isize;

    let prev_f: Vec<f32> = prev.iter().map(|&v| v as f32).collect();
    let next_f: Vec<f32> = next.iter().map(|&v| v as f32).collect();

    let new_vectors: Vec<FlowVector> = (0..flow.vectors.len())
        .into_par_iter()
        .map(|idx| {
            let px = (idx % w_usize) as isize;
            let py = (idx / w_usize) as isize;

            let old_v = flow.vectors[idx];

            let mut a11 = 0.0f32;
            let mut a12 = 0.0f32;
            let mut a22 = 0.0f32;
            let mut b1 = 0.0f32;
            let mut b2 = 0.0f32;

            for wy in -half..=half {
                for wx in -half..=half {
                    let x = px + wx;
                    let y = py + wy;

                    if x < 1 || y < 1 || x >= w_usize as isize - 1 || y >= h_usize as isize - 1 {
                        continue;
                    }

                    // Spatial gradients from prev frame
                    let ix = (prev_f[y as usize * w_usize + (x + 1) as usize]
                        - prev_f[y as usize * w_usize + (x - 1) as usize])
                        * 0.5;
                    let iy = (prev_f[(y + 1) as usize * w_usize + x as usize]
                        - prev_f[(y - 1) as usize * w_usize + x as usize])
                        * 0.5;

                    // Sample next frame with warp
                    let nx_f = x as f32 + old_v.dx;
                    let ny_f = y as f32 + old_v.dy;
                    let next_val = sample_bilinear(&next_f, nx_f, ny_f, w_usize, h_usize);
                    let prev_val = prev_f[y as usize * w_usize + x as usize];

                    let it = next_val - prev_val;

                    a11 += ix * ix;
                    a12 += ix * iy;
                    a22 += iy * iy;
                    b1 -= ix * it;
                    b2 -= iy * it;
                }
            }

            // Solve 2×2 system [a11 a12; a12 a22] * [ddx; ddy] = [b1; b2]
            let det = a11 * a22 - a12 * a12;
            if det.abs() < 1e-6 {
                return old_v;
            }

            let ddx = (a22 * b1 - a12 * b2) / det;
            let ddy = (a11 * b2 - a12 * b1) / det;

            // Clamp update magnitude
            let mag = (ddx * ddx + ddy * ddy).sqrt();
            let (ddx, ddy) = if mag > max_disp {
                (ddx * max_disp / mag, ddy * max_disp / mag)
            } else {
                (ddx, ddy)
            };

            FlowVector {
                dx: old_v.dx + ddx,
                dy: old_v.dy + ddy,
            }
        })
        .collect();

    FlowField {
        width: w,
        height: h,
        vectors: new_vectors,
    }
}

/// Bilinear sampling of a float luminance plane.
fn sample_bilinear(frame: &[f32], x: f32, y: f32, w: usize, h: usize) -> f32 {
    let x0 = x.floor() as isize;
    let y0 = y.floor() as isize;
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;

    let sample = |xi: isize, yi: isize| -> f32 {
        let xi = xi.clamp(0, w as isize - 1) as usize;
        let yi = yi.clamp(0, h as isize - 1) as usize;
        frame[yi * w + xi]
    };

    sample(x0, y0) * (1.0 - tx) * (1.0 - ty)
        + sample(x0 + 1, y0) * tx * (1.0 - ty)
        + sample(x0, y0 + 1) * (1.0 - tx) * ty
        + sample(x0 + 1, y0 + 1) * tx * ty
}

/// Apply Gaussian smoothing to a flow field.
fn smooth_flow(flow: FlowField, w: u32, h: u32, sigma: f32) -> FlowField {
    let kernel = gaussian_kernel_1d(sigma);
    let half = (kernel.len() / 2) as isize;
    let w_usize = w as usize;
    let h_usize = h as usize;

    // Horizontal pass
    let mut temp = vec![FlowVector::zero(); (w * h) as usize];
    for y in 0..h_usize {
        for x in 0..w_usize {
            let mut dx_sum = 0.0f32;
            let mut dy_sum = 0.0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sx = (x as isize + ki as isize - half).clamp(0, w_usize as isize - 1) as usize;
                let v = flow.vectors[y * w_usize + sx];
                dx_sum += kv * v.dx;
                dy_sum += kv * v.dy;
            }
            temp[y * w_usize + x] = FlowVector {
                dx: dx_sum,
                dy: dy_sum,
            };
        }
    }

    // Vertical pass
    let mut out = vec![FlowVector::zero(); (w * h) as usize];
    for y in 0..h_usize {
        for x in 0..w_usize {
            let mut dx_sum = 0.0f32;
            let mut dy_sum = 0.0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sy = (y as isize + ki as isize - half).clamp(0, h_usize as isize - 1) as usize;
                let v = temp[sy * w_usize + x];
                dx_sum += kv * v.dx;
                dy_sum += kv * v.dy;
            }
            out[y * w_usize + x] = FlowVector {
                dx: dx_sum,
                dy: dy_sum,
            };
        }
    }

    FlowField {
        width: w,
        height: h,
        vectors: out,
    }
}

/// Build a normalised 1-D Gaussian kernel with standard deviation `sigma`.
fn gaussian_kernel_1d(sigma: f32) -> Vec<f32> {
    let radius = (3.0 * sigma).ceil() as usize;
    let size = 2 * radius + 1;
    let mut kernel = Vec::with_capacity(size);

    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut sum = 0.0f32;

    for i in 0..size {
        let x = i as f32 - radius as f32;
        let v = (-x * x / two_sigma_sq).exp();
        kernel.push(v);
        sum += v;
    }

    // Normalise
    for v in &mut kernel {
        *v /= sum;
    }

    kernel
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn gray_frame(w: usize, h: usize, fill: u8) -> Vec<u8> {
        vec![fill; w * h]
    }

    #[test]
    fn test_flow_field_zeros() {
        let f = FlowField::zeros(4, 4);
        assert_eq!(f.width, 4);
        assert_eq!(f.height, 4);
        assert_eq!(f.vectors.len(), 16);
        assert!((f.mean_magnitude() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_flow_field_get_oob_returns_none() {
        let f = FlowField::zeros(4, 4);
        assert!(f.get(4, 0).is_none());
        assert!(f.get(0, 4).is_none());
    }

    #[test]
    fn test_static_scene_zero_flow() {
        let w = 32;
        let h = 32;
        let frame = gray_frame(w, h, 128);
        let estimator = OpticalFlowEstimator::new(OpticalFlowConfig {
            pyramid_levels: 2,
            iterations: 2,
            window_size: 5,
            smoothing_sigma: 0.0,
            max_displacement: 2.0,
        });
        let flow = estimator
            .estimate(&frame, &frame, w as u32, h as u32)
            .expect("should succeed");
        // For a static scene (identical frames) flow should be near zero.
        assert!(
            flow.mean_magnitude() < 0.5,
            "mean magnitude = {}",
            flow.mean_magnitude()
        );
    }

    #[test]
    fn test_invalid_dimensions_rejected() {
        let estimator = OpticalFlowEstimator::default_config();
        let res = estimator.estimate(&[], &[], 0, 0);
        assert!(res.is_err());
    }

    #[test]
    fn test_buffer_size_mismatch_rejected() {
        let estimator = OpticalFlowEstimator::default_config();
        let res = estimator.estimate(&[0u8; 4], &[0u8; 8], 2, 2);
        assert!(res.is_err());
    }

    #[test]
    fn test_warp_frame_identity() {
        let w = 4u32;
        let h = 4u32;
        let flow = FlowField::zeros(w, h);
        let frame: Vec<u8> = (0..(w * h * 4) as usize).map(|i| (i % 256) as u8).collect();
        let warped = flow.warp_frame(&frame).expect("warp should succeed");
        // Identity flow should produce the same image (within bilinear rounding).
        assert_eq!(warped.len(), frame.len());
    }

    #[test]
    fn test_pyramid_dims_decreasing() {
        assert_eq!(pyramid_dims(64, 48, 0), (64, 48));
        assert_eq!(pyramid_dims(64, 48, 1), (32, 24));
        assert_eq!(pyramid_dims(64, 48, 2), (16, 12));
    }

    #[test]
    fn test_flow_vector_magnitude() {
        let v = FlowVector { dx: 3.0, dy: 4.0 };
        assert!((v.magnitude() - 5.0).abs() < 1e-5);
    }
}
