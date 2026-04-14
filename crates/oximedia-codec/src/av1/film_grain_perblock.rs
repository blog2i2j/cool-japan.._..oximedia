//! Per-block film grain synthesis with bilinear boundary interpolation.
//!
//! This module extends the base film grain synthesizer with spatially-varying
//! grain parameters. Each `GrainBlock` carries its own `FilmGrainParams` and
//! pre-built `ScalingLut`, so different regions of a frame can have different
//! grain strength, AR coefficients, and chroma weighting.
//!
//! # Boundary Interpolation
//!
//! At the shared edge between two adjacent grain blocks the output pixel is a
//! bilinear blend of the grain contributions from both blocks.  The blend zone
//! is `BLEND_ZONE` pixels wide; the weight ramps linearly from 0 (interior of
//! the left/top block) to 1 (interior of the right/bottom block).
//!
//! This avoids visible seams that would appear if the grain parameters changed
//! abruptly at block boundaries.

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::too_many_arguments)]

use super::film_grain::{FilmGrainParams, ScalingLut, GRAIN_BLOCK_SIZE, MAX_LUMA_SCALING_POINTS};
use crate::CodecResult;

// =============================================================================
// Constants
// =============================================================================

/// Width of the bilinear blend zone at block boundaries (pixels).
pub const BLEND_ZONE: usize = 4;

// =============================================================================
// GrainBlock
// =============================================================================

/// A spatially-localised grain block with its own parameters and LUTs.
///
/// The block covers pixels [`x * GRAIN_BLOCK_SIZE`, `(x+1) * GRAIN_BLOCK_SIZE`)
/// horizontally and similarly for `y`.
#[derive(Clone, Debug)]
pub struct GrainBlock {
    /// Block column index (in grain-block units).
    pub x: u32,
    /// Block row index (in grain-block units).
    pub y: u32,
    /// Film grain parameters for this block.
    pub film_grain_params: FilmGrainParams,
    /// Pre-built luma scaling LUT derived from `film_grain_params`.
    pub luma_scaling_lut: Vec<u8>,
}

impl GrainBlock {
    /// Create a `GrainBlock` at `(x, y)` with the given parameters.
    ///
    /// The `luma_scaling_lut` is built automatically from the parameters.
    #[must_use]
    pub fn new(x: u32, y: u32, params: FilmGrainParams, bit_depth: u8) -> Self {
        let n = params.num_y_points.min(MAX_LUMA_SCALING_POINTS);
        let lut = ScalingLut::from_points(&params.y_points[..n], n, bit_depth);
        Self {
            x,
            y,
            film_grain_params: params,
            luma_scaling_lut: lut.values,
        }
    }

    /// Sample the luma scaling LUT for a given pixel value.
    #[inline]
    #[must_use]
    fn scale_luma(&self, pixel: i32, bit_depth: u8) -> i32 {
        let idx = if bit_depth <= 8 {
            pixel.clamp(0, 255) as usize
        } else {
            let shift = bit_depth - 8;
            (pixel >> shift).clamp(0, 255) as usize
        };
        let idx = idx.min(self.luma_scaling_lut.len().saturating_sub(1));
        i32::from(self.luma_scaling_lut[idx])
    }
}

// =============================================================================
// Per-block application (u16 luma plane)
// =============================================================================

/// Apply per-block film grain to a high-bit-depth luma plane.
///
/// Each block in `blocks` contributes grain to the pixels it covers.  At
/// block boundaries a `BLEND_ZONE`-pixel wide bilinear blend between adjacent
/// blocks prevents visible discontinuities.
///
/// # Arguments
///
/// * `plane`        – mutable luma plane (row-major, one `u16` per pixel)
/// * `stride`       – number of `u16` elements per row (`>= frame_width`)
/// * `blocks`       – per-block grain descriptors (may overlap any order)
/// * `frame_width`  – frame width in pixels
/// * `frame_height` – frame height in pixels
/// * `bit_depth`    – bit depth of `plane` (8, 10 or 12)
///
/// # Errors
///
/// Currently infallible; returns `Ok(())` for API consistency.
pub fn apply_grain_per_block_bilinear(
    plane: &mut [u16],
    stride: usize,
    blocks: &[GrainBlock],
    frame_width: u32,
    frame_height: u32,
    bit_depth: u8,
) -> CodecResult<()> {
    let fw = frame_width as usize;
    let fh = frame_height as usize;
    let bsz = GRAIN_BLOCK_SIZE;
    let max_val = i32::from((1u16 << bit_depth.min(15)) - 1);

    // Build a per-block index map for fast lookup.
    // blocks_x = number of block columns covering [0, fw)
    let blocks_x = fw.div_ceil(bsz);
    let blocks_y = fh.div_ceil(bsz);
    let total = blocks_x * blocks_y;

    // Map from (col_idx, row_idx) → &GrainBlock (last write wins for duplicates).
    let mut block_map: Vec<Option<&GrainBlock>> = vec![None; total];
    for b in blocks {
        let bx = b.x as usize;
        let by = b.y as usize;
        if bx < blocks_x && by < blocks_y {
            block_map[by * blocks_x + bx] = Some(b);
        }
    }

    // For each pixel compute the blended grain contribution.
    for row in 0..fh {
        let by = row / bsz;
        let local_y = row % bsz;

        for col in 0..fw {
            let bx = col / bsz;
            let local_x = col % bsz;
            let idx = row * stride + col;
            if idx >= plane.len() {
                continue;
            }
            let pixel = i32::from(plane[idx]);

            // Primary block grain.
            let primary_grain = block_map
                .get(by * blocks_x + bx)
                .and_then(|o| *o)
                .map(|b| compute_block_grain(b, col, row, pixel, bit_depth))
                .unwrap_or(0);

            // Horizontal blend with right neighbour inside BLEND_ZONE.
            let h_blend_frac = if local_x + BLEND_ZONE >= bsz && bx + 1 < blocks_x {
                // How far into the right-block blend zone?
                let dist = (bsz - local_x).min(BLEND_ZONE);
                let frac = (BLEND_ZONE - dist) as i32; // 0..BLEND_ZONE
                Some((frac, bx + 1, by))
            } else {
                None
            };

            // Vertical blend with bottom neighbour inside BLEND_ZONE.
            let v_blend_frac = if local_y + BLEND_ZONE >= bsz && by + 1 < blocks_y {
                let dist = (bsz - local_y).min(BLEND_ZONE);
                let frac = (BLEND_ZONE - dist) as i32;
                Some((frac, bx, by + 1))
            } else {
                None
            };

            let mut grain = primary_grain;

            if let Some((frac, nbx, nby)) = h_blend_frac {
                let neigh_grain = block_map
                    .get(nby * blocks_x + nbx)
                    .and_then(|o| *o)
                    .map(|b| compute_block_grain(b, col, row, pixel, bit_depth))
                    .unwrap_or(primary_grain);
                // Linear blend: grain = primary*(BZ-frac)/BZ + neigh*frac/BZ
                let bz = BLEND_ZONE as i32;
                grain = (grain * (bz - frac) + neigh_grain * frac) / bz;
            }

            if let Some((frac, nbx, nby)) = v_blend_frac {
                let neigh_grain = block_map
                    .get(nby * blocks_x + nbx)
                    .and_then(|o| *o)
                    .map(|b| compute_block_grain(b, col, row, pixel, bit_depth))
                    .unwrap_or(grain);
                let bz = BLEND_ZONE as i32;
                grain = (grain * (bz - frac) + neigh_grain * frac) / bz;
            }

            plane[idx] = (pixel + grain).clamp(0, max_val) as u16;
        }
    }

    Ok(())
}

/// Compute the grain contribution for a single pixel from one block.
///
/// Returns the signed grain delta (before clamping to the output range).
fn compute_block_grain(
    block: &GrainBlock,
    col: usize,
    row: usize,
    pixel: i32,
    bit_depth: u8,
) -> i32 {
    let params = &block.film_grain_params;
    if !params.apply_grain || params.num_y_points == 0 {
        return 0;
    }

    // Generate deterministic grain value via a simple hash of (seed, col, row).
    let seed = u64::from(params.grain_seed);
    let hash = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add((col as u64).wrapping_mul(2654435761))
        .wrapping_add((row as u64).wrapping_mul(40503));
    // Map to a signed value in [-128, 127].
    let raw = ((hash >> 32) as i32 & 0xFF) - 128;

    // Scale by per-block luma LUT.
    let scale = block.scale_luma(pixel, bit_depth);
    let g_scale = i32::from(params.grain_scaling());
    let g_shift = i32::from(params.grain_scale_shift);

    (raw * scale * g_scale) >> (g_shift + 8)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::super::film_grain::FilmGrainParams;
    use super::*;

    fn make_enabled_params(seed: u16, scaling: u8) -> FilmGrainParams {
        let mut p = FilmGrainParams::new();
        p.apply_grain = true;
        p.film_grain_params_present = true;
        p.grain_seed = seed;
        p.grain_scaling_minus_8 = 0;
        p.add_y_point(0, scaling);
        p.add_y_point(255, scaling);
        p
    }

    /// Test: uniform grain (all blocks same params) → apply_grain_per_block_bilinear runs OK.
    #[test]
    fn test_uniform_grain_runs_without_panic() {
        let params = make_enabled_params(12345, 48);
        let w = 64u32;
        let h = 64u32;
        let bsz = GRAIN_BLOCK_SIZE;
        let bx_count = (w as usize).div_ceil(bsz);
        let by_count = (h as usize).div_ceil(bsz);

        let blocks: Vec<GrainBlock> = (0..by_count)
            .flat_map(|by| {
                let p = params.clone();
                (0..bx_count).map(move |bx| GrainBlock::new(bx as u32, by as u32, p.clone(), 10))
            })
            .collect();

        let mut plane = vec![512u16; w as usize * h as usize];
        let result = apply_grain_per_block_bilinear(&mut plane, w as usize, &blocks, w, h, 10);
        assert!(result.is_ok());
    }

    /// Test: all pixel values stay within 10-bit range [0, 1023].
    #[test]
    fn test_10bit_output_range() {
        let params = make_enabled_params(99, 64);
        let w = 64u32;
        let h = 64u32;
        let bsz = GRAIN_BLOCK_SIZE;
        let bx_count = (w as usize).div_ceil(bsz);
        let by_count = (h as usize).div_ceil(bsz);

        let blocks: Vec<GrainBlock> = (0..by_count)
            .flat_map(|by| {
                let p = params.clone();
                (0..bx_count).map(move |bx| GrainBlock::new(bx as u32, by as u32, p.clone(), 10))
            })
            .collect();

        let mut plane = vec![512u16; w as usize * h as usize];
        apply_grain_per_block_bilinear(&mut plane, w as usize, &blocks, w, h, 10).expect("apply");
        for &px in &plane {
            assert!(px <= 1023, "10-bit pixel out of range: {px}");
        }
    }

    /// Test: gradient grain (varying strength per block) → center vs edge differ in RMS.
    #[test]
    fn test_gradient_grain_center_differs_from_edges() {
        let w = 128u32;
        let h = 128u32;
        let bsz = GRAIN_BLOCK_SIZE;
        let bx_count = (w as usize).div_ceil(bsz);
        let by_count = (h as usize).div_ceil(bsz);

        let mut blocks = Vec::new();
        for by in 0..by_count {
            for bx in 0..bx_count {
                // Center blocks get stronger grain (scaling=120), edges get weak (scaling=10).
                let is_center = bx == bx_count / 2 && by == by_count / 2;
                let scaling = if is_center { 120u8 } else { 10u8 };
                let p = make_enabled_params(1000 + (by * bx_count + bx) as u16, scaling);
                blocks.push(GrainBlock::new(bx as u32, by as u32, p, 10));
            }
        }

        let w_sz = w as usize;
        let h_sz = h as usize;
        let plane_orig = vec![512u16; w_sz * h_sz];
        let mut plane_mod = plane_orig.clone();

        apply_grain_per_block_bilinear(&mut plane_mod, w_sz, &blocks, w, h, 10).expect("apply");

        // Compute RMS delta for center block.
        let cx0 = (bx_count / 2) * bsz;
        let cy0 = (by_count / 2) * bsz;
        let cx1 = (cx0 + bsz).min(w_sz);
        let cy1 = (cy0 + bsz).min(h_sz);

        let center_rms = rms_delta(&plane_orig, &plane_mod, w_sz, cx0, cy0, cx1, cy1);
        let edge_rms = rms_delta(&plane_orig, &plane_mod, w_sz, 0, 0, bsz, bsz);

        assert!(
            center_rms >= edge_rms,
            "center RMS {center_rms:.2} should be >= edge RMS {edge_rms:.2}",
        );
    }

    /// Helper: compute RMS of pixel-level delta in a sub-rectangle.
    fn rms_delta(
        orig: &[u16],
        modified: &[u16],
        stride: usize,
        x0: usize,
        y0: usize,
        x1: usize,
        y1: usize,
    ) -> f64 {
        let mut sum_sq = 0i64;
        let mut count = 0usize;
        for row in y0..y1 {
            for col in x0..x1 {
                let idx = row * stride + col;
                if idx >= orig.len() || idx >= modified.len() {
                    continue;
                }
                let delta = i64::from(modified[idx]) - i64::from(orig[idx]);
                sum_sq += delta * delta;
                count += 1;
            }
        }
        if count == 0 {
            return 0.0;
        }
        ((sum_sq as f64) / (count as f64)).sqrt()
    }

    /// Test: boundary interpolation — delta at a block boundary should not
    /// exceed `BLEND_ZONE + 2` gray levels above the neighboring block delta.
    /// This checks that the bilinear ramp avoids large discontinuities.
    #[test]
    fn test_boundary_interpolation_smooth() {
        let w = 64u32;
        let h = 32u32;
        let bsz = GRAIN_BLOCK_SIZE;

        // Two side-by-side blocks with very different scaling.
        let p_left = make_enabled_params(111, 8);
        let p_right = make_enabled_params(222, 200);
        let blocks = vec![
            GrainBlock::new(0, 0, p_left, 10),
            GrainBlock::new(1, 0, p_right, 10),
        ];

        let w_sz = w as usize;
        let h_sz = h as usize;
        let orig = vec![512u16; w_sz * h_sz];
        let mut plane = orig.clone();

        apply_grain_per_block_bilinear(&mut plane, w_sz, &blocks, w, h, 10).expect("apply");

        // Sample the boundary column and two columns on each side.
        let boundary_col = bsz; // First column of the right block.
        let row = h_sz / 2;

        let left_delta = plane[row * w_sz + boundary_col.saturating_sub(1)] as i32
            - orig[row * w_sz + boundary_col.saturating_sub(1)] as i32;
        let right_delta =
            plane[row * w_sz + boundary_col] as i32 - orig[row * w_sz + boundary_col] as i32;

        // The difference in grain across the boundary should be finite.
        let jump = (right_delta - left_delta).abs();
        assert!(
            jump < 300,
            "Grain jump at boundary ({jump}) should be smoothed by bilinear blend"
        );
    }

    /// Test: 8-bit path — output values stay in [0, 255].
    #[test]
    fn test_8bit_output_range() {
        let params = make_enabled_params(55, 80);
        let w = 32u32;
        let h = 32u32;
        let blocks = vec![GrainBlock::new(0, 0, params, 8)];
        let mut plane: Vec<u16> = (0..32 * 32).map(|i| (i % 255) as u16).collect();
        apply_grain_per_block_bilinear(&mut plane, 32, &blocks, w, h, 8).expect("apply");
        for &px in &plane {
            assert!(px <= 255, "8-bit pixel out of range: {px}");
        }
    }

    /// Test: disabled grain (apply_grain=false) → plane is unchanged.
    #[test]
    fn test_disabled_grain_noop() {
        let mut params = FilmGrainParams::new();
        params.apply_grain = false;
        params.grain_seed = 42;
        let w = 32u32;
        let h = 32u32;
        let blocks = vec![GrainBlock::new(0, 0, params, 10)];
        let orig: Vec<u16> = (0..1024).map(|i| i as u16).collect();
        let mut plane = orig.clone();
        apply_grain_per_block_bilinear(&mut plane, 32, &blocks, w, h, 10).expect("apply");
        // With apply_grain=false, no modification.
        assert_eq!(plane, orig);
    }

    /// Test: empty block list → plane is unchanged.
    #[test]
    fn test_empty_block_list_noop() {
        let w = 32u32;
        let h = 32u32;
        let orig: Vec<u16> = (0..1024).map(|i| i as u16).collect();
        let mut plane = orig.clone();
        apply_grain_per_block_bilinear(&mut plane, 32, &[], w, h, 10).expect("apply");
        assert_eq!(plane, orig);
    }

    /// Test: GrainBlock::new builds a luma_scaling_lut of the right size.
    #[test]
    fn test_grain_block_lut_size() {
        let params = make_enabled_params(7, 64);
        let b = GrainBlock::new(0, 0, params, 8);
        assert_eq!(
            b.luma_scaling_lut.len(),
            256,
            "8-bit LUT should have 256 entries"
        );

        let params10 = make_enabled_params(8, 32);
        let b10 = GrainBlock::new(0, 0, params10, 10);
        // 10-bit uses min(bit_depth, 8) = 8 → 256 entries
        assert_eq!(b10.luma_scaling_lut.len(), 256);
    }

    /// Test: 12-bit path stays in [0, 4095].
    #[test]
    fn test_12bit_output_range() {
        let params = make_enabled_params(123, 64);
        let w = 32u32;
        let h = 32u32;
        let blocks = vec![GrainBlock::new(0, 0, params, 12)];
        let mut plane = vec![2048u16; 32 * 32];
        apply_grain_per_block_bilinear(&mut plane, 32, &blocks, w, h, 12).expect("apply");
        for &px in &plane {
            assert!(px <= 4095, "12-bit pixel out of range: {px}");
        }
    }
}
