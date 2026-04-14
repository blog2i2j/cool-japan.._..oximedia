//! SIMD-optimized Sum of Absolute Differences (SAD) and Sum of Squared
//! Differences (SSD) for block matching in motion estimation.
//!
//! This module provides optimised implementations that exploit data-level
//! parallelism through wide integer arithmetic and chunk-based iteration.
//! On platforms that support auto-vectorisation the compiler will typically
//! emit SIMD instructions (SSE2 / AVX2 / NEON) for the inner loops.
//!
//! # Overview
//!
//! The fundamental block-matching cost functions used in video motion
//! estimation are:
//!
//! - **SAD** (Sum of Absolute Differences): `Σ |cur[i] - ref[i]|`
//! - **SSD** (Sum of Squared Differences): `Σ (cur[i] - ref[i])²`
//! - **SATD** (Sum of Absolute Transformed Differences): Hadamard-transformed
//!   SAD, which approximates rate-distortion cost more accurately than plain SAD.
//!
//! All functions accept row-major luma planes and explicit block coordinates
//! so they can be called directly from a motion estimator without any memory
//! allocation.
//!
//! # Example
//!
//! ```rust
//! use oximedia_video::simd_ops::{block_sad, block_ssd};
//!
//! let reference = vec![128u8; 16 * 16];
//! let current   = vec![130u8; 16 * 16];
//!
//! // SAD over the full 16×16 block
//! let sad = block_sad(&reference, &current, 16, 0, 0, 16, 16);
//! assert_eq!(sad, 2 * 16 * 16);
//!
//! // SSD over the full 16×16 block
//! let ssd = block_ssd(&reference, &current, 16, 0, 0, 16, 16);
//! assert_eq!(ssd, 4 * 16 * 16);
//! ```

// -----------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------

/// Width of the "SIMD lane" used for bulk processing.
///
/// The inner SAD / SSD loops accumulate into `LANE_WIDTH` accumulators in
/// parallel, giving the auto-vectoriser the best opportunity to emit wide
/// SIMD instructions.  Must be a power of two ≥ 4.
const LANE_WIDTH: usize = 8;

// -----------------------------------------------------------------------
// SAD  (Sum of Absolute Differences)
// -----------------------------------------------------------------------

/// Compute the SAD between a rectangular block in two luma planes.
///
/// # Parameters
///
/// - `ref_plane`  – Reference (previous / anchor) luma plane.  Row-major,
///   `plane_width` bytes per row.
/// - `cur_plane`  – Current luma plane, same layout.
/// - `plane_width` – Number of columns in each plane (stride).
/// - `bx`, `by`   – Top-left corner of the block in **both** planes.
/// - `block_w`, `block_h` – Block dimensions in pixels.
///
/// # Panics
///
/// Does not panic; out-of-bounds accesses return 0 via safe indexing.
pub fn block_sad(
    ref_plane: &[u8],
    cur_plane: &[u8],
    plane_width: u32,
    bx: u32,
    by: u32,
    block_w: u32,
    block_h: u32,
) -> u32 {
    block_sad_offset(
        ref_plane,
        cur_plane,
        plane_width,
        bx,
        by,
        bx,
        by,
        block_w,
        block_h,
    )
}

/// Compute the SAD between two differently-positioned blocks (full motion
/// vector evaluation).
///
/// - `(ref_bx, ref_by)` – Block origin in the reference plane.
/// - `(cur_bx, cur_by)` – Block origin in the current plane.
pub fn block_sad_offset(
    ref_plane: &[u8],
    cur_plane: &[u8],
    plane_width: u32,
    ref_bx: u32,
    ref_by: u32,
    cur_bx: u32,
    cur_by: u32,
    block_w: u32,
    block_h: u32,
) -> u32 {
    let pw = plane_width as usize;
    let bw = block_w as usize;
    let bh = block_h as usize;
    let rbx = ref_bx as usize;
    let rby = ref_by as usize;
    let cbx = cur_bx as usize;
    let cby = cur_by as usize;

    let mut total: u32 = 0;

    for row in 0..bh {
        let ref_row_start = (rby + row) * pw + rbx;
        let cur_row_start = (cby + row) * pw + cbx;

        // Collect row slices; fall back to zero-length slice if out of bounds.
        let ref_row = ref_plane
            .get(ref_row_start..ref_row_start + bw)
            .unwrap_or(&[]);
        let cur_row = cur_plane
            .get(cur_row_start..cur_row_start + bw)
            .unwrap_or(&[]);

        let safe_len = ref_row.len().min(cur_row.len());

        // Process LANE_WIDTH pixels at a time using parallel accumulators.
        let lane_end = (safe_len / LANE_WIDTH) * LANE_WIDTH;
        let mut accum = [0u32; LANE_WIDTH];

        for chunk_start in (0..lane_end).step_by(LANE_WIDTH) {
            for lane in 0..LANE_WIDTH {
                let i = chunk_start + lane;
                let diff = ref_row[i] as i32 - cur_row[i] as i32;
                accum[lane] += diff.unsigned_abs();
            }
        }

        // Reduce accumulators.
        let mut row_sum: u32 = accum.iter().sum();

        // Handle the remaining tail (< LANE_WIDTH pixels).
        for i in lane_end..safe_len {
            let diff = ref_row[i] as i32 - cur_row[i] as i32;
            row_sum += diff.unsigned_abs();
        }

        total += row_sum;
    }

    total
}

/// Compute SAD for a motion vector candidate.
///
/// A convenience wrapper around [`block_sad_offset`] that accepts signed
/// integer-pel displacements.  Returns `u32::MAX` if the displaced block
/// would lie entirely outside the plane.
pub fn sad_for_mv(
    ref_plane: &[u8],
    cur_plane: &[u8],
    plane_width: u32,
    plane_height: u32,
    cur_bx: u32,
    cur_by: u32,
    block_size: u32,
    mv_dx: i32,
    mv_dy: i32,
) -> u32 {
    let pw = plane_width as i32;
    let ph = plane_height as i32;
    let bs = block_size as i32;
    let ref_x = cur_bx as i32 + mv_dx;
    let ref_y = cur_by as i32 + mv_dy;

    // Reject if the reference block is entirely out of bounds.
    if ref_x < 0 || ref_y < 0 || ref_x + bs > pw || ref_y + bs > ph {
        return u32::MAX;
    }

    block_sad_offset(
        ref_plane,
        cur_plane,
        plane_width,
        ref_x as u32,
        ref_y as u32,
        cur_bx,
        cur_by,
        block_size,
        block_size,
    )
}

// -----------------------------------------------------------------------
// SSD  (Sum of Squared Differences)
// -----------------------------------------------------------------------

/// Compute the SSD between a rectangular block in two luma planes.
pub fn block_ssd(
    ref_plane: &[u8],
    cur_plane: &[u8],
    plane_width: u32,
    bx: u32,
    by: u32,
    block_w: u32,
    block_h: u32,
) -> u64 {
    block_ssd_offset(
        ref_plane,
        cur_plane,
        plane_width,
        bx,
        by,
        bx,
        by,
        block_w,
        block_h,
    )
}

/// Compute the SSD between two differently-positioned blocks.
pub fn block_ssd_offset(
    ref_plane: &[u8],
    cur_plane: &[u8],
    plane_width: u32,
    ref_bx: u32,
    ref_by: u32,
    cur_bx: u32,
    cur_by: u32,
    block_w: u32,
    block_h: u32,
) -> u64 {
    let pw = plane_width as usize;
    let bw = block_w as usize;
    let bh = block_h as usize;
    let rbx = ref_bx as usize;
    let rby = ref_by as usize;
    let cbx = cur_bx as usize;
    let cby = cur_by as usize;

    let mut total: u64 = 0;

    for row in 0..bh {
        let ref_row_start = (rby + row) * pw + rbx;
        let cur_row_start = (cby + row) * pw + cbx;

        let ref_row = ref_plane
            .get(ref_row_start..ref_row_start + bw)
            .unwrap_or(&[]);
        let cur_row = cur_plane
            .get(cur_row_start..cur_row_start + bw)
            .unwrap_or(&[]);

        let safe_len = ref_row.len().min(cur_row.len());

        // Parallel accumulators for SIMD-friendly inner loop.
        let lane_end = (safe_len / LANE_WIDTH) * LANE_WIDTH;
        let mut accum = [0u64; LANE_WIDTH];

        for chunk_start in (0..lane_end).step_by(LANE_WIDTH) {
            for lane in 0..LANE_WIDTH {
                let i = chunk_start + lane;
                let diff = ref_row[i] as i64 - cur_row[i] as i64;
                accum[lane] += (diff * diff) as u64;
            }
        }

        let mut row_sum: u64 = accum.iter().sum();

        for i in lane_end..safe_len {
            let diff = ref_row[i] as i64 - cur_row[i] as i64;
            row_sum += (diff * diff) as u64;
        }

        total += row_sum;
    }

    total
}

// -----------------------------------------------------------------------
// SATD  (Sum of Absolute Transformed Differences) — 4×4 Hadamard
// -----------------------------------------------------------------------

/// Compute the 4×4 Hadamard SATD for a single 4×4 block.
///
/// SATD is a better proxy for rate-distortion cost than SAD because it
/// operates in the frequency domain.  This implementation uses the fast
/// integer Hadamard transform and computes the L1 norm of the coefficients.
///
/// `residual` must contain exactly 16 `i32` values arranged row-major
/// (4 values per row, 4 rows).
///
/// Returns the SATD value (always non-negative).
pub fn hadamard_satd_4x4(residual: &[i32; 16]) -> u32 {
    // Horizontal pass (butterfly across each row)
    let mut h = [0i32; 16];
    for row in 0..4 {
        let base = row * 4;
        let a = residual[base];
        let b = residual[base + 1];
        let c = residual[base + 2];
        let d = residual[base + 3];
        h[base] = a + b + c + d;
        h[base + 1] = a - b + c - d;
        h[base + 2] = a + b - c - d;
        h[base + 3] = a - b - c + d;
    }

    // Vertical pass (butterfly down each column)
    let mut v = [0i32; 16];
    for col in 0..4 {
        let a = h[col];
        let b = h[4 + col];
        let c = h[8 + col];
        let d = h[12 + col];
        v[col] = a + b + c + d;
        v[4 + col] = a - b + c - d;
        v[8 + col] = a + b - c - d;
        v[12 + col] = a - b - c + d;
    }

    // L1 norm of the transform coefficients, divided by 2
    v.iter().map(|&x| x.unsigned_abs()).sum::<u32>() / 2
}

/// Compute SATD over an arbitrary block by tiling 4×4 sub-blocks.
///
/// Blocks whose dimensions are not multiples of 4 have their trailing pixels
/// rounded down to the nearest multiple of 4.
pub fn block_satd(
    ref_plane: &[u8],
    cur_plane: &[u8],
    plane_width: u32,
    bx: u32,
    by: u32,
    block_w: u32,
    block_h: u32,
) -> u32 {
    let pw = plane_width as usize;
    let bw4 = (block_w as usize / 4) * 4;
    let bh4 = (block_h as usize / 4) * 4;
    let bx = bx as usize;
    let by = by as usize;

    let mut total: u32 = 0;

    for tile_y in (0..bh4).step_by(4) {
        for tile_x in (0..bw4).step_by(4) {
            let mut residual = [0i32; 16];
            for row in 0..4 {
                for col in 0..4 {
                    let ref_idx = (by + tile_y + row) * pw + (bx + tile_x + col);
                    let cur_idx = (by + tile_y + row) * pw + (bx + tile_x + col);
                    let r = ref_plane.get(ref_idx).copied().unwrap_or(0) as i32;
                    let c = cur_plane.get(cur_idx).copied().unwrap_or(0) as i32;
                    residual[row * 4 + col] = r - c;
                }
            }
            total += hadamard_satd_4x4(&residual);
        }
    }

    total
}

// -----------------------------------------------------------------------
// Variance / Mean helpers
// -----------------------------------------------------------------------

/// Compute the mean and variance of a block.
///
/// Returns `(mean, variance)` where `mean` is the average pixel value and
/// `variance` is the population variance (not sample variance).
pub fn block_variance(
    plane: &[u8],
    plane_width: u32,
    bx: u32,
    by: u32,
    block_w: u32,
    block_h: u32,
) -> (f64, f64) {
    let pw = plane_width as usize;
    let bw = block_w as usize;
    let bh = block_h as usize;
    let bx = bx as usize;
    let by = by as usize;
    let n = (bw * bh) as f64;

    if n == 0.0 {
        return (0.0, 0.0);
    }

    let mut sum = 0u64;
    let mut sum_sq = 0u64;

    for row in 0..bh {
        let row_start = (by + row) * pw + bx;
        let row_slice = plane.get(row_start..row_start + bw).unwrap_or(&[]);

        // Parallel accumulators
        let lane_end = (row_slice.len() / LANE_WIDTH) * LANE_WIDTH;
        let mut s_acc = [0u32; LANE_WIDTH];
        let mut sq_acc = [0u32; LANE_WIDTH];

        for chunk_start in (0..lane_end).step_by(LANE_WIDTH) {
            for lane in 0..LANE_WIDTH {
                let px = row_slice[chunk_start + lane] as u32;
                s_acc[lane] += px;
                sq_acc[lane] += px * px;
            }
        }

        sum += s_acc.iter().map(|&v| v as u64).sum::<u64>();
        sum_sq += sq_acc.iter().map(|&v| v as u64).sum::<u64>();

        for i in lane_end..row_slice.len() {
            let px = row_slice[i] as u64;
            sum += px;
            sum_sq += px * px;
        }
    }

    let mean = sum as f64 / n;
    let variance = sum_sq as f64 / n - mean * mean;

    (mean, variance.max(0.0))
}

// -----------------------------------------------------------------------
// Batch SAD — evaluate multiple motion vector candidates at once
// -----------------------------------------------------------------------

/// Evaluate SAD for a list of `(mv_dx, mv_dy)` candidate motion vectors for
/// a single block and return the index of the minimum-SAD candidate together
/// with its cost.
///
/// Returns `None` if `candidates` is empty.
pub fn best_sad_candidate(
    ref_plane: &[u8],
    cur_plane: &[u8],
    plane_width: u32,
    plane_height: u32,
    cur_bx: u32,
    cur_by: u32,
    block_size: u32,
    candidates: &[(i32, i32)],
) -> Option<(usize, u32)> {
    let mut best_idx: usize = 0;
    let mut best_cost: u32 = u32::MAX;
    let mut found = false;

    for (idx, &(dx, dy)) in candidates.iter().enumerate() {
        let cost = sad_for_mv(
            ref_plane,
            cur_plane,
            plane_width,
            plane_height,
            cur_bx,
            cur_by,
            block_size,
            dx,
            dy,
        );
        if cost < best_cost {
            best_cost = cost;
            best_idx = idx;
            found = true;
        }
    }

    if found {
        Some((best_idx, best_cost))
    } else {
        None
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a flat luma plane of `w × h` pixels all set to `val`.
    fn flat_plane(w: usize, h: usize, val: u8) -> Vec<u8> {
        vec![val; w * h]
    }

    /// Create a ramp plane: pixel at (x, y) = (y * w + x) % 256.
    fn ramp_plane(w: usize, h: usize) -> Vec<u8> {
        (0..w * h).map(|i| (i % 256) as u8).collect()
    }

    // 1. SAD of identical flat blocks is zero.
    #[test]
    fn test_sad_identical_blocks_zero() {
        let plane = flat_plane(16, 16, 128);
        let sad = block_sad(&plane, &plane, 16, 0, 0, 16, 16);
        assert_eq!(sad, 0);
    }

    // 2. SAD of blocks differing by a constant is correct.
    #[test]
    fn test_sad_constant_difference() {
        let ref_plane = flat_plane(16, 16, 100);
        let cur_plane = flat_plane(16, 16, 120);
        // Difference = 20 per pixel, 16×16 = 256 pixels.
        let sad = block_sad(&ref_plane, &cur_plane, 16, 0, 0, 16, 16);
        assert_eq!(sad, 20 * 256);
    }

    // 3. SAD is symmetric: SAD(a,b) == SAD(b,a).
    #[test]
    fn test_sad_symmetry() {
        let a = ramp_plane(16, 16);
        let b = flat_plane(16, 16, 100);
        let sad_ab = block_sad(&a, &b, 16, 0, 0, 16, 16);
        let sad_ba = block_sad(&b, &a, 16, 0, 0, 16, 16);
        assert_eq!(sad_ab, sad_ba);
    }

    // 4. SSD of identical flat blocks is zero.
    #[test]
    fn test_ssd_identical_blocks_zero() {
        let plane = flat_plane(16, 16, 200);
        let ssd = block_ssd(&plane, &plane, 16, 0, 0, 16, 16);
        assert_eq!(ssd, 0);
    }

    // 5. SSD of blocks differing by a constant is correct.
    #[test]
    fn test_ssd_constant_difference() {
        let ref_plane = flat_plane(8, 8, 50);
        let cur_plane = flat_plane(8, 8, 54);
        // Difference = 4 per pixel, squared = 16, 8×8 = 64 pixels.
        let ssd = block_ssd(&ref_plane, &cur_plane, 8, 0, 0, 8, 8);
        assert_eq!(ssd, 16 * 64);
    }

    // 6. SSD >= SAD^2 / n (Cauchy-Schwarz inequality).
    #[test]
    fn test_ssd_ge_sad_squared_over_n() {
        let a = ramp_plane(8, 8);
        let b = flat_plane(8, 8, 100);
        let sad = block_sad(&a, &b, 8, 0, 0, 8, 8) as u64;
        let ssd = block_ssd(&a, &b, 8, 0, 0, 8, 8);
        let n = 64u64;
        // By Cauchy-Schwarz: SSD * n >= SAD^2
        assert!(ssd * n >= sad * sad);
    }

    // 7. block_sad_offset correctly reads different positions.
    #[test]
    fn test_sad_offset_different_positions() {
        let mut plane = flat_plane(32, 8, 0);
        // Fill the second 16-pixel block with value 255.
        for row in 0..8usize {
            for col in 16..32usize {
                plane[row * 32 + col] = 255;
            }
        }
        let ref_plane = flat_plane(32, 8, 128);
        // SAD: ref at (0,0) vs cur at (16,0), 8×8 block → all cur pixels are 255.
        let sad = block_sad_offset(&ref_plane, &plane, 32, 0, 0, 16, 0, 8, 8);
        // |128 - 255| = 127 per pixel, 64 pixels
        assert_eq!(sad, 127 * 64);
    }

    // 8. sad_for_mv returns u32::MAX for out-of-bounds displacements.
    #[test]
    fn test_sad_for_mv_out_of_bounds() {
        let plane = flat_plane(16, 16, 128);
        let result = sad_for_mv(&plane, &plane, 16, 16, 0, 0, 16, 32, 0);
        assert_eq!(result, u32::MAX);
    }

    // 9. sad_for_mv zero displacement equals block_sad.
    #[test]
    fn test_sad_for_mv_zero_displacement() {
        let ref_plane = flat_plane(16, 16, 100);
        let cur_plane = flat_plane(16, 16, 130);
        let mv_sad = sad_for_mv(&ref_plane, &cur_plane, 16, 16, 0, 0, 16, 0, 0);
        let direct_sad = block_sad(&ref_plane, &cur_plane, 16, 0, 0, 16, 16);
        assert_eq!(mv_sad, direct_sad);
    }

    // 10. Hadamard SATD of all-zero residual is zero.
    #[test]
    fn test_hadamard_satd_zero_residual() {
        let residual = [0i32; 16];
        assert_eq!(hadamard_satd_4x4(&residual), 0);
    }

    // 11. Hadamard SATD of constant residual: DC term only.
    #[test]
    fn test_hadamard_satd_constant_residual() {
        // All 16 residuals = 1. After horizontal + vertical Hadamard, only the
        // DC coefficient (index 0) is 16; all others are 0.
        // SATD = 16 / 2 = 8.
        let residual = [1i32; 16];
        assert_eq!(hadamard_satd_4x4(&residual), 8);
    }

    // 12. block_satd of identical blocks is zero.
    #[test]
    fn test_block_satd_identical_zero() {
        let plane = flat_plane(16, 16, 200);
        let satd = block_satd(&plane, &plane, 16, 0, 0, 16, 16);
        assert_eq!(satd, 0);
    }

    // 13. block_variance: flat plane has variance 0.
    #[test]
    fn test_block_variance_flat_zero() {
        let plane = flat_plane(16, 16, 128);
        let (mean, variance) = block_variance(&plane, 16, 0, 0, 16, 16);
        assert!((mean - 128.0).abs() < 1e-9);
        assert!(variance < 1e-9);
    }

    // 14. block_variance: two-value plane has known variance.
    #[test]
    fn test_block_variance_two_values() {
        // Alternating 0/100 plane → mean = 50, variance = (50^2 + 50^2)/2 = 2500
        let plane: Vec<u8> = (0..16 * 16)
            .map(|i| if i % 2 == 0 { 0 } else { 100 })
            .collect();
        let (mean, variance) = block_variance(&plane, 16, 0, 0, 16, 16);
        assert!((mean - 50.0).abs() < 0.5, "mean={mean}");
        assert!((variance - 2500.0).abs() < 1.0, "variance={variance}");
    }

    // 15. best_sad_candidate: identity motion vector wins for identical planes.
    #[test]
    fn test_best_sad_candidate_identity_wins() {
        let ref_plane = ramp_plane(16, 16);
        let cur_plane = ref_plane.clone();
        let candidates: Vec<(i32, i32)> = vec![(0, 0), (1, 0), (0, 1), (-1, 0)];
        let result = best_sad_candidate(&ref_plane, &cur_plane, 16, 16, 0, 0, 8, &candidates);
        let (best_idx, best_cost) = result.expect("should find a candidate");
        assert_eq!(best_idx, 0, "zero-displacement should win");
        assert_eq!(best_cost, 0, "identical planes → SAD=0");
    }

    // 16. best_sad_candidate: returns None for empty candidates.
    #[test]
    fn test_best_sad_candidate_empty_candidates() {
        let plane = flat_plane(16, 16, 128);
        let result = best_sad_candidate(&plane, &plane, 16, 16, 0, 0, 8, &[]);
        assert!(result.is_none());
    }

    // 17. SAD of 4×4 block with partial row access (non-power-of-2 width).
    #[test]
    fn test_sad_non_standard_block_size() {
        let ref_plane = flat_plane(10, 10, 50);
        let cur_plane = flat_plane(10, 10, 60);
        let sad = block_sad(&ref_plane, &cur_plane, 10, 0, 0, 5, 5);
        // |50-60| = 10 per pixel, 25 pixels
        assert_eq!(sad, 10 * 25);
    }

    // 18. SSD is always >= SAD (for blocks with at least one non-zero diff).
    #[test]
    fn test_ssd_ge_sad_nonzero() {
        let ref_plane = flat_plane(8, 8, 200);
        let cur_plane = flat_plane(8, 8, 220);
        let sad = block_sad(&ref_plane, &cur_plane, 8, 0, 0, 8, 8) as u64;
        let ssd = block_ssd(&ref_plane, &cur_plane, 8, 0, 0, 8, 8);
        assert!(ssd >= sad, "ssd={ssd} should be >= sad={sad}");
    }
}
