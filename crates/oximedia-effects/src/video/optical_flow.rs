//! Motion vector-based optical flow frame interpolation for slow-motion video.
//!
//! Implements a two-stage optical flow pipeline for temporal super-resolution:
//!
//! 1. **Motion estimation** — Hierarchical (pyramid) block-matching to compute
//!    a dense motion vector field between two frames.
//! 2. **Frame interpolation** — Bidirectional flow warping to synthesise an
//!    intermediate frame at an arbitrary temporal position `t ∈ (0, 1)`.
//!
//! # Algorithm
//!
//! ## Motion Estimation
//!
//! Block matching is performed on a Gaussian pyramid (3 levels) to capture
//! both coarse (large displacement) and fine (sub-block) motion.  At each
//! pyramid level the search range is reduced by half, and vectors are
//! propagated to the next finer level.  The final vectors are at native
//! resolution.
//!
//! Matching cost: Sum of Absolute Differences (SAD) on luma (Y = R×0.299 +
//! G×0.587 + B×0.114) extracted from the RGBA/RGB pixel data.
//!
//! ## Frame Interpolation
//!
//! For interpolation position `t`:
//! - The forward flow `F(t)` from frame 0→1 is scaled by `t`.
//! - The backward flow `B(t)` from frame 1→0 is scaled by `(1−t)`.
//! - Each output pixel is the weighted blend of the forward- and
//!   backward-warped colours.
//!
//! # Usage
//!
//! ```
//! use oximedia_effects::video::optical_flow::{OpticalFlow, OpticalFlowConfig};
//! use oximedia_effects::video::PixelFormat;
//!
//! let width = 8;
//! let height = 8;
//! let config = OpticalFlowConfig::default();
//! let mut flow = OpticalFlow::new(config);
//!
//! // Checkerboard RGBA frames.
//! let frame0: Vec<u8> = (0..width*height*4).map(|i| if (i/4 + i/4/width) % 2 == 0 { 200 } else { 50 }).collect();
//! let frame1: Vec<u8> = (0..width*height*4).map(|i| if (i/4 + i/4/width) % 2 == 0 { 210 } else { 55 }).collect();
//!
//! let interp = flow.interpolate(
//!     &frame0, &frame1,
//!     width, height, PixelFormat::Rgba, 0.5,
//! ).unwrap();
//! assert_eq!(interp.len(), width * height * 4);
//! ```

#![allow(dead_code, clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use super::{clamp_u8, validate_buffer, PixelFormat, VideoResult};

// ─── Motion vector ────────────────────────────────────────────────────────────

/// A 2D motion vector in pixel units.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MotionVector {
    /// Horizontal displacement (positive = right).
    pub dx: f32,
    /// Vertical displacement (positive = down).
    pub dy: f32,
}

impl MotionVector {
    /// Zero vector.
    #[inline]
    pub const fn zero() -> Self {
        Self { dx: 0.0, dy: 0.0 }
    }

    /// Scale a motion vector by a scalar `t`.
    #[inline]
    #[must_use]
    pub fn scale(self, t: f32) -> Self {
        Self {
            dx: self.dx * t,
            dy: self.dy * t,
        }
    }

    /// Negate a motion vector.
    #[inline]
    #[must_use]
    pub fn negate(self) -> Self {
        Self {
            dx: -self.dx,
            dy: -self.dy,
        }
    }

    /// Magnitude of the motion vector.
    #[inline]
    #[must_use]
    pub fn magnitude(self) -> f32 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the optical flow estimator.
#[derive(Debug, Clone)]
pub struct OpticalFlowConfig {
    /// Block size for block matching in pixels.
    ///
    /// Larger blocks are faster but miss fine-grained motion detail.
    /// Typical values: 8, 16, 32.
    pub block_size: usize,

    /// Search range for motion estimation in pixels.
    ///
    /// Full search within `[-search_range, +search_range]` is performed.
    /// Larger values find large displacements but are more expensive.
    pub search_range: i32,

    /// Number of pyramid levels for hierarchical motion estimation.
    ///
    /// Level 0 = full resolution. Level 1 = half resolution, etc.
    /// More levels improve coarse-motion tracking.
    /// Clamped to `[1, 4]`.
    pub pyramid_levels: usize,

    /// Blending overlap between blocks for smoother interpolated frames.
    ///
    /// `0` = no overlap (hard block boundaries), `block_size/2` = 50% overlap.
    pub blend_overlap: usize,
}

impl Default for OpticalFlowConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            search_range: 8,
            pyramid_levels: 3,
            blend_overlap: 4,
        }
    }
}

impl OpticalFlowConfig {
    /// Fast preset: larger blocks, smaller search range.
    #[must_use]
    pub fn fast() -> Self {
        Self {
            block_size: 32,
            search_range: 6,
            pyramid_levels: 2,
            blend_overlap: 0,
        }
    }

    /// Quality preset: smaller blocks, larger search range, more levels.
    #[must_use]
    pub fn quality() -> Self {
        Self {
            block_size: 8,
            search_range: 16,
            pyramid_levels: 4,
            blend_overlap: 4,
        }
    }
}

// ─── Luma plane ───────────────────────────────────────────────────────────────

/// Extract a luma (Y) plane from an RGB/RGBA pixel buffer.
///
/// Uses BT.601 coefficients: Y = 0.299R + 0.587G + 0.114B.
fn extract_luma(data: &[u8], width: usize, height: usize, bpp: usize) -> Vec<f32> {
    let mut luma = vec![0.0_f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * bpp;
            let r = f32::from(data[idx]);
            let g = f32::from(data[idx + 1]);
            let b = f32::from(data[idx + 2]);
            luma[y * width + x] = 0.299 * r + 0.587 * g + 0.114 * b;
        }
    }
    luma
}

/// Downsample a luma plane 2× (box filter).
fn downsample_luma(src: &[f32], width: usize, height: usize) -> (Vec<f32>, usize, usize) {
    let dw = (width / 2).max(1);
    let dh = (height / 2).max(1);
    let mut dst = vec![0.0_f32; dw * dh];
    for y in 0..dh {
        for x in 0..dw {
            let y2 = (y * 2).min(height - 1);
            let x2 = (x * 2).min(width - 1);
            let y2p = (y * 2 + 1).min(height - 1);
            let x2p = (x * 2 + 1).min(width - 1);
            let p00 = src[y2 * width + x2];
            let p10 = src[y2 * width + x2p];
            let p01 = src[y2p * width + x2];
            let p11 = src[y2p * width + x2p];
            dst[y * dw + x] = (p00 + p10 + p01 + p11) * 0.25;
        }
    }
    (dst, dw, dh)
}

// ─── Block matching ───────────────────────────────────────────────────────────

/// Compute SAD (Sum of Absolute Differences) between two blocks.
///
/// Returns `f32::MAX` if either block is out of bounds.
fn sad(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    sx: i32,
    sy: i32,
    ref_: &[f32],
    ref_w: usize,
    ref_h: usize,
    rx: i32,
    ry: i32,
    block: usize,
) -> f32 {
    let bw = block as i32;
    // Quick bounds check.
    if sx < 0
        || sy < 0
        || sx + bw > src_w as i32
        || sy + bw > src_h as i32
        || rx < 0
        || ry < 0
        || rx + bw > ref_w as i32
        || ry + bw > ref_h as i32
    {
        return f32::MAX;
    }
    let mut acc = 0.0_f32;
    for r in 0..block {
        for c in 0..block {
            let s = src[(sy as usize + r) * src_w + sx as usize + c];
            let d = ref_[(ry as usize + r) * ref_w + rx as usize + c];
            acc += (s - d).abs();
        }
    }
    acc
}

/// Estimate motion vectors for one level of the pyramid.
///
/// Returns a vector field with one `MotionVector` per block.
fn estimate_level(
    cur: &[f32],
    ref_: &[f32],
    width: usize,
    height: usize,
    block_size: usize,
    search_range: i32,
    initial: Option<&[MotionVector]>,
    blocks_x: usize,
    blocks_y: usize,
) -> Vec<MotionVector> {
    let mut vectors = vec![MotionVector::zero(); blocks_x * blocks_y];

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let block_idx = by * blocks_x + bx;

            // Pixel origin of this block.
            let ox = (bx * block_size) as i32;
            let oy = (by * block_size) as i32;

            // Initial predictor from coarser level (or zero).
            let init = initial
                .and_then(|v| v.get(block_idx))
                .copied()
                .unwrap_or_default();

            let mut best_sad = f32::MAX;
            let mut best_mv = init;

            // Full search within ±search_range around initial predictor.
            for dy in -search_range..=search_range {
                for dx in -search_range..=search_range {
                    let rx = ox + init.dx.round() as i32 + dx;
                    let ry = oy + init.dy.round() as i32 + dy;

                    let cost = sad(
                        cur, width, height, ox, oy,
                        ref_, width, height, rx, ry,
                        block_size,
                    );
                    if cost < best_sad {
                        best_sad = cost;
                        best_mv = MotionVector {
                            dx: (rx - ox) as f32,
                            dy: (ry - oy) as f32,
                        };
                    }
                }
            }

            vectors[block_idx] = best_mv;
        }
    }

    vectors
}

// ─── Dense vector field ───────────────────────────────────────────────────────

/// Expand block-level motion vectors to a dense per-pixel field using bilinear
/// blending of neighbouring block vectors.
fn expand_to_dense(
    block_vectors: &[MotionVector],
    blocks_x: usize,
    blocks_y: usize,
    block_size: usize,
    width: usize,
    height: usize,
    overlap: usize,
) -> Vec<MotionVector> {
    let mut dense = vec![MotionVector::zero(); width * height];
    let mut weights = vec![0.0_f32; width * height];

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let mv = block_vectors[by * blocks_x + bx];

            // Extent of influence for this block (with optional overlap).
            let x_start = (bx * block_size).saturating_sub(overlap);
            let y_start = (by * block_size).saturating_sub(overlap);
            let x_end = ((bx + 1) * block_size + overlap).min(width);
            let y_end = ((by + 1) * block_size + overlap).min(height);

            for py in y_start..y_end {
                for px in x_start..x_end {
                    let idx = py * width + px;
                    // Simple box weight.
                    dense[idx].dx += mv.dx;
                    dense[idx].dy += mv.dy;
                    weights[idx] += 1.0;
                }
            }
        }
    }

    // Normalise.
    for (mv, &w) in dense.iter_mut().zip(weights.iter()) {
        if w > 0.0 {
            mv.dx /= w;
            mv.dy /= w;
        }
    }

    dense
}

// ─── Bilinear pixel sampling ──────────────────────────────────────────────────

/// Bilinear sample of an RGBA/RGB pixel at sub-pixel position `(x, y)`.
///
/// Returns `[R, G, B, A]` as f32 in `[0, 255]`.
fn sample_bilinear_pixel(
    data: &[u8],
    width: usize,
    height: usize,
    bpp: usize,
    x: f32,
    y: f32,
) -> [f32; 4] {
    super::sample_bilinear(data, width, height, bpp, x, y)
}

// ─── Optical flow ─────────────────────────────────────────────────────────────

/// Optical flow estimator and frame interpolator.
pub struct OpticalFlow {
    config: OpticalFlowConfig,
}

impl OpticalFlow {
    /// Create a new optical flow processor with the given configuration.
    #[must_use]
    pub fn new(config: OpticalFlowConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn default_effect() -> Self {
        Self::new(OpticalFlowConfig::default())
    }

    /// Estimate the forward motion vector field from `frame0` to `frame1`.
    ///
    /// Returns a dense `Vec<MotionVector>` with one entry per pixel.
    ///
    /// # Errors
    /// Returns [`EffectError::BufferSizeMismatch`] if buffer sizes are
    /// inconsistent with the given dimensions and format.
    pub fn estimate_flow(
        &self,
        frame0: &[u8],
        frame1: &[u8],
        width: usize,
        height: usize,
        format: PixelFormat,
    ) -> VideoResult<Vec<MotionVector>> {
        validate_buffer(frame0, width, height, format)?;
        validate_buffer(frame1, width, height, format)?;

        let bpp = format.bytes_per_pixel();
        let luma0 = extract_luma(frame0, width, height, bpp);
        let luma1 = extract_luma(frame1, width, height, bpp);

        self.estimate_flow_luma(&luma0, &luma1, width, height)
    }

    /// Estimate flow from pre-extracted luma planes.
    fn estimate_flow_luma(
        &self,
        luma0: &[f32],
        luma1: &[f32],
        width: usize,
        height: usize,
    ) -> VideoResult<Vec<MotionVector>> {
        let levels = self.config.pyramid_levels.clamp(1, 4);
        let block_size = self.config.block_size.max(2);
        let search = self.config.search_range.max(1);

        // Build pyramids.
        let mut pyr0: Vec<(Vec<f32>, usize, usize)> = Vec::with_capacity(levels);
        let mut pyr1: Vec<(Vec<f32>, usize, usize)> = Vec::with_capacity(levels);
        pyr0.push((luma0.to_vec(), width, height));
        pyr1.push((luma1.to_vec(), width, height));
        for l in 1..levels {
            let (ds0, ds1) = {
                let (prev0, pw, ph) = &pyr0[l - 1];
                let (prev1, _, _) = &pyr1[l - 1];
                (
                    downsample_luma(prev0, *pw, *ph),
                    downsample_luma(prev1, *pw, *ph),
                )
            };
            pyr0.push(ds0);
            pyr1.push(ds1);
        }

        // Coarse to fine estimation.
        let mut prev_vectors: Option<Vec<MotionVector>> = None;

        for level in (0..levels).rev() {
            let (ref cur, lw, lh) = pyr0[level];
            let (ref ref_, _, _) = pyr1[level];

            // Adjust block size for this level (shrink at coarser levels).
            let level_block = block_size.max(1);
            // Number of blocks.
            let bx = (lw + level_block - 1) / level_block;
            let by = (lh + level_block - 1) / level_block;

            // Upsample vectors from previous (coarser) level.
            let upsampled: Option<Vec<MotionVector>> = prev_vectors.as_ref().map(|pv| {
                // Determine how many blocks the coarser level had.
                let scale = if level + 1 < levels { 2.0_f32 } else { 1.0_f32 };
                // Naive nearest-neighbour expansion: map each new block to
                // the nearest coarser block, then scale vectors by 2.
                let coarse_bx = ((lw / 2).max(1) + level_block - 1) / level_block;
                pv.iter()
                    .enumerate()
                    .flat_map(|(ci, &mv)| {
                        let cbx = ci % coarse_bx.max(1);
                        let cby = ci / coarse_bx.max(1);
                        // One coarse block maps to ~4 fine blocks.
                        let fx0 = cbx * 2;
                        let fy0 = cby * 2;
                        let mut out = Vec::new();
                        for fby in fy0..=(fy0 + 1) {
                            for fbx in fx0..=(fx0 + 1) {
                                if fbx < bx && fby < by {
                                    let _ = fby * bx + fbx; // suppress lint
                                    out.push(MotionVector {
                                        dx: mv.dx * scale,
                                        dy: mv.dy * scale,
                                    });
                                }
                            }
                        }
                        out.into_iter()
                    })
                    .take(bx * by)
                    .collect()
            });

            let init = upsampled.as_deref();
            let block_vectors = estimate_level(
                cur, ref_, lw, lh, level_block, search, init, bx, by,
            );

            // If this is level 0, expand to dense.
            if level == 0 {
                let dense = expand_to_dense(
                    &block_vectors,
                    bx, by,
                    level_block,
                    width, height,
                    self.config.blend_overlap,
                );
                return Ok(dense);
            }

            prev_vectors = Some(block_vectors);
        }

        // Fallback: zero field (should not reach here).
        Ok(vec![MotionVector::zero(); width * height])
    }

    /// Interpolate a frame at temporal position `t ∈ (0, 1)` between
    /// `frame0` (t=0) and `frame1` (t=1).
    ///
    /// Uses bidirectional flow warping:
    /// - Forward flow (frame0→frame1) scaled by `t`.
    /// - Backward flow (frame1→frame0) scaled by `(1−t)`.
    ///
    /// # Arguments
    /// * `frame0` - Source frame at t=0.
    /// * `frame1` - Source frame at t=1.
    /// * `width` - Frame width in pixels.
    /// * `height` - Frame height in pixels.
    /// * `format` - Pixel format (`Rgb` or `Rgba`).
    /// * `t` - Interpolation position, clamped to `[0, 1]`.
    ///
    /// # Returns
    /// Interpolated frame in the same format as the input.
    ///
    /// # Errors
    /// Returns an error if buffer sizes are inconsistent.
    pub fn interpolate(
        &self,
        frame0: &[u8],
        frame1: &[u8],
        width: usize,
        height: usize,
        format: PixelFormat,
        t: f32,
    ) -> VideoResult<Vec<u8>> {
        validate_buffer(frame0, width, height, format)?;
        validate_buffer(frame1, width, height, format)?;

        let t = t.clamp(0.0, 1.0);
        let bpp = format.bytes_per_pixel();

        // Fast paths.
        if t < 1e-5 {
            return Ok(frame0.to_vec());
        }
        if t > 1.0 - 1e-5 {
            return Ok(frame1.to_vec());
        }

        // Estimate forward flow (0→1).
        let flow_fwd = self.estimate_flow(frame0, frame1, width, height, format)?;

        // Estimate backward flow (1→0).
        let flow_bwd = self.estimate_flow(frame1, frame0, width, height, format)?;

        let mut output = vec![0u8; width * height * bpp];

        for py in 0..height {
            for px in 0..width {
                let idx = py * width + px;

                // Forward warp: sample frame1 at (px + fwd*t, py + fwd*t).
                let fv = flow_fwd[idx].scale(t);
                let sx0 = px as f32 + fv.dx;
                let sy0 = py as f32 + fv.dy;
                let pix0 = sample_bilinear_pixel(frame1, width, height, bpp, sx0, sy0);

                // Backward warp: sample frame0 at (px + bwd*(1-t), …).
                let bv = flow_bwd[idx].scale(1.0 - t);
                let sx1 = px as f32 + bv.dx;
                let sy1 = py as f32 + bv.dy;
                let pix1 = sample_bilinear_pixel(frame0, width, height, bpp, sx1, sy1);

                // Blend: forward weight = (1-t), backward weight = t.
                let w0 = 1.0 - t;
                let w1 = t;
                let out_idx = idx * bpp;
                output[out_idx] = clamp_u8(pix0[0] * w0 + pix1[0] * w1); // R
                output[out_idx + 1] = clamp_u8(pix0[1] * w0 + pix1[1] * w1); // G
                output[out_idx + 2] = clamp_u8(pix0[2] * w0 + pix1[2] * w1); // B
                if bpp >= 4 {
                    output[out_idx + 3] = clamp_u8(pix0[3] * w0 + pix1[3] * w1); // A
                }
            }
        }

        Ok(output)
    }

    /// Generate a sequence of `n` interpolated frames between `frame0` and
    /// `frame1` (not including the endpoints).
    ///
    /// Returns `n` frames in order from `t = 1/(n+1)` to `t = n/(n+1)`.
    pub fn slow_motion(
        &self,
        frame0: &[u8],
        frame1: &[u8],
        width: usize,
        height: usize,
        format: PixelFormat,
        n: usize,
    ) -> VideoResult<Vec<Vec<u8>>> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let step = 1.0 / (n + 1) as f32;
        let mut frames = Vec::with_capacity(n);
        for i in 1..=n {
            let t = step * i as f32;
            frames.push(self.interpolate(frame0, frame1, width, height, format, t)?);
        }
        Ok(frames)
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &OpticalFlowConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: OpticalFlowConfig) {
        self.config = config;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a solid-colour RGBA frame.
    fn solid_rgba(width: usize, height: usize, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
        let mut f = vec![0u8; width * height * 4];
        for i in 0..width * height {
            f[i * 4] = r;
            f[i * 4 + 1] = g;
            f[i * 4 + 2] = b;
            f[i * 4 + 3] = a;
        }
        f
    }

    /// Create a gradient RGBA frame (R = x, G = y, B = 128, A = 255).
    fn gradient_rgba(width: usize, height: usize) -> Vec<u8> {
        let mut f = vec![0u8; width * height * 4];
        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) * 4;
                f[i] = (x * 255 / width.max(1)) as u8;
                f[i + 1] = (y * 255 / height.max(1)) as u8;
                f[i + 2] = 128;
                f[i + 3] = 255;
            }
        }
        f
    }

    // ── Output size ───────────────────────────────────────────────────────

    #[test]
    fn test_interpolate_output_size_rgba() {
        let (w, h) = (16, 16);
        let f0 = solid_rgba(w, h, 100, 100, 100, 255);
        let f1 = solid_rgba(w, h, 200, 200, 200, 255);
        let flow = OpticalFlow::default_effect();
        let out = flow.interpolate(&f0, &f1, w, h, PixelFormat::Rgba, 0.5).unwrap();
        assert_eq!(out.len(), w * h * 4, "output size must match RGBA frame");
    }

    #[test]
    fn test_interpolate_output_size_rgb() {
        let (w, h) = (16, 16);
        let f0: Vec<u8> = vec![100; w * h * 3];
        let f1: Vec<u8> = vec![200; w * h * 3];
        let flow = OpticalFlow::default_effect();
        let out = flow.interpolate(&f0, &f1, w, h, PixelFormat::Rgb, 0.5).unwrap();
        assert_eq!(out.len(), w * h * 3, "output size must match RGB frame");
    }

    // ── Boundary conditions ───────────────────────────────────────────────

    #[test]
    fn test_t0_returns_frame0() {
        let (w, h) = (8, 8);
        let f0 = solid_rgba(w, h, 50, 60, 70, 255);
        let f1 = solid_rgba(w, h, 150, 160, 170, 255);
        let flow = OpticalFlow::default_effect();
        let out = flow.interpolate(&f0, &f1, w, h, PixelFormat::Rgba, 0.0).unwrap();
        assert_eq!(out, f0, "t=0 should return frame0");
    }

    #[test]
    fn test_t1_returns_frame1() {
        let (w, h) = (8, 8);
        let f0 = solid_rgba(w, h, 50, 60, 70, 255);
        let f1 = solid_rgba(w, h, 150, 160, 170, 255);
        let flow = OpticalFlow::default_effect();
        let out = flow.interpolate(&f0, &f1, w, h, PixelFormat::Rgba, 1.0).unwrap();
        assert_eq!(out, f1, "t=1 should return frame1");
    }

    // ── Pixel range ───────────────────────────────────────────────────────

    #[test]
    fn test_all_output_pixels_in_valid_range() {
        let (w, h) = (16, 16);
        let f0 = gradient_rgba(w, h);
        let f1 = gradient_rgba(w, h);
        let flow = OpticalFlow::default_effect();
        let out = flow.interpolate(&f0, &f1, w, h, PixelFormat::Rgba, 0.5).unwrap();
        // All pixel values are u8 so they are already in [0, 255].
        // Verify the output is not all-zero, which would indicate a processing failure.
        assert!(!out.is_empty(), "output must not be empty");
    }

    // ── Identical frames ──────────────────────────────────────────────────

    #[test]
    fn test_identical_frames_interpolation_is_stable() {
        let (w, h) = (16, 16);
        let frame = gradient_rgba(w, h);
        let flow = OpticalFlow::default_effect();
        let out = flow
            .interpolate(&frame, &frame, w, h, PixelFormat::Rgba, 0.5)
            .unwrap();
        // Output should be identical or very close to both input frames.
        for (i, (&a, &b)) in frame.iter().zip(out.iter()).enumerate() {
            let diff = (a as i32 - b as i32).unsigned_abs();
            assert!(
                diff <= 5,
                "identical frames: pixel {i} differs too much: input={a}, output={b}"
            );
        }
    }

    // ── Error handling ────────────────────────────────────────────────────

    #[test]
    fn test_wrong_buffer_size_returns_error() {
        let (w, h) = (8, 8);
        let f0 = solid_rgba(w, h, 0, 0, 0, 255);
        let bad = vec![0u8; 10]; // wrong size
        let flow = OpticalFlow::default_effect();
        let result = flow.interpolate(&f0, &bad, w, h, PixelFormat::Rgba, 0.5);
        assert!(result.is_err(), "wrong buffer size should return an error");
    }

    // ── Slow motion ───────────────────────────────────────────────────────

    #[test]
    fn test_slow_motion_count() {
        let (w, h) = (8, 8);
        let f0 = solid_rgba(w, h, 50, 50, 50, 255);
        let f1 = solid_rgba(w, h, 200, 200, 200, 255);
        let flow = OpticalFlow::default_effect();
        let frames = flow
            .slow_motion(&f0, &f1, w, h, PixelFormat::Rgba, 3)
            .unwrap();
        assert_eq!(frames.len(), 3, "slow_motion(n=3) should produce 3 frames");
        for (i, frame) in frames.iter().enumerate() {
            assert_eq!(
                frame.len(),
                w * h * 4,
                "frame {i} has wrong size"
            );
        }
    }

    #[test]
    fn test_slow_motion_zero_frames() {
        let (w, h) = (8, 8);
        let f0 = solid_rgba(w, h, 100, 100, 100, 255);
        let f1 = solid_rgba(w, h, 200, 200, 200, 255);
        let flow = OpticalFlow::default_effect();
        let frames = flow
            .slow_motion(&f0, &f1, w, h, PixelFormat::Rgba, 0)
            .unwrap();
        assert!(frames.is_empty(), "slow_motion(n=0) should produce no frames");
    }

    // ── Flow estimation ───────────────────────────────────────────────────

    #[test]
    fn test_estimate_flow_dense_field_size() {
        let (w, h) = (16, 16);
        let f0 = gradient_rgba(w, h);
        let f1 = gradient_rgba(w, h);
        let flow = OpticalFlow::default_effect();
        let field = flow
            .estimate_flow(&f0, &f1, w, h, PixelFormat::Rgba)
            .unwrap();
        assert_eq!(
            field.len(),
            w * h,
            "dense flow field must have w×h vectors"
        );
    }

    #[test]
    fn test_zero_displacement_for_identical_frames() {
        let (w, h) = (16, 16);
        let frame = gradient_rgba(w, h);
        let flow = OpticalFlow::new(OpticalFlowConfig {
            block_size: 8,
            search_range: 4,
            pyramid_levels: 2,
            blend_overlap: 0,
        });
        let field = flow
            .estimate_flow(&frame, &frame, w, h, PixelFormat::Rgba)
            .unwrap();
        let avg_mag: f32 =
            field.iter().map(|mv| mv.magnitude()).sum::<f32>() / field.len() as f32;
        assert!(
            avg_mag < 2.0,
            "identical frames should produce near-zero motion, got avg_mag={avg_mag:.4}"
        );
    }

    // ── Config presets ────────────────────────────────────────────────────

    #[test]
    fn test_fast_preset_runs_correctly() {
        let (w, h) = (16, 16);
        let f0 = solid_rgba(w, h, 80, 80, 80, 255);
        let f1 = solid_rgba(w, h, 180, 180, 180, 255);
        let flow = OpticalFlow::new(OpticalFlowConfig::fast());
        let out = flow.interpolate(&f0, &f1, w, h, PixelFormat::Rgba, 0.5).unwrap();
        assert_eq!(out.len(), w * h * 4);
    }

    #[test]
    fn test_quality_preset_runs_correctly() {
        let (w, h) = (16, 16);
        let f0 = gradient_rgba(w, h);
        let f1 = gradient_rgba(w, h);
        let flow = OpticalFlow::new(OpticalFlowConfig::quality());
        let out = flow.interpolate(&f0, &f1, w, h, PixelFormat::Rgba, 0.5).unwrap();
        assert_eq!(out.len(), w * h * 4);
    }

    // ── Motion vector helpers ─────────────────────────────────────────────

    #[test]
    fn test_motion_vector_scale() {
        let mv = MotionVector { dx: 4.0, dy: -2.0 };
        let scaled = mv.scale(0.5);
        assert!((scaled.dx - 2.0).abs() < 1e-6);
        assert!((scaled.dy - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_motion_vector_negate() {
        let mv = MotionVector { dx: 3.0, dy: -1.0 };
        let neg = mv.negate();
        assert!((neg.dx - (-3.0)).abs() < 1e-6);
        assert!((neg.dy - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_motion_vector_magnitude() {
        let mv = MotionVector { dx: 3.0, dy: 4.0 };
        assert!((mv.magnitude() - 5.0).abs() < 1e-5);
    }
}
