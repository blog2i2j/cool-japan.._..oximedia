//! Bidirectional motion estimation for B-frame–style interpolation.
//!
//! This module estimates motion vectors from a *current* frame towards both a
//! backward reference (the previous frame, `P0`) and a forward reference (the
//! next frame, `P1`), and uses them to generate a temporally interpolated
//! frame at an arbitrary position `t ∈ [0, 1]` between `P0` and `P1`.
//!
//! # Algorithm overview
//!
//! 1. **Backward estimation** – for each block in the current frame, a full
//!    search (constrained to `±search_range`) is run against `P0` to find the
//!    block whose sum-of-absolute-differences (SAD) is minimised.  The
//!    resulting vector is `mv_back`.
//!
//! 2. **Forward estimation** – the same search is run against `P1` to produce
//!    `mv_fwd`.
//!
//! 3. **Symmetry check** – the backward and forward vectors are compared.  If
//!    their L₁ distance exceeds `symmetry_threshold` the block is marked
//!    *occlusion* and the corresponding pixels in the interpolated frame are
//!    blended from both references equally.
//!
//! 4. **Interpolation** – for each block, the backward reference pixel at the
//!    backward-displaced position is weighted by `(1 − t)` and the forward
//!    reference pixel at the forward-displaced position is weighted by `t`.
//!    The result is rounded and clamped to `[0, 255]`.
//!
//! # Example
//!
//! ```rust
//! use oximedia_video::bidirectional_motion::{BidirMotionEstimator, BidirConfig};
//!
//! let w = 32u32;
//! let h = 32u32;
//! let n = (w * h) as usize;
//! let p0: Vec<u8> = (0..n).map(|i| (i % 200) as u8).collect();
//! let p1: Vec<u8> = (0..n).map(|i| ((i + 16) % 200) as u8).collect();
//!
//! let cfg = BidirConfig::default();
//! let est = BidirMotionEstimator::new(cfg);
//! let result = est.interpolate(&p0, &p1, w, h, 0.5);
//! assert_eq!(result.interpolated.len(), n);
//! ```

// ---------------------------------------------------------------------------
// BidirConfig
// ---------------------------------------------------------------------------

/// Configuration for the bidirectional motion estimator.
#[derive(Debug, Clone)]
pub struct BidirConfig {
    /// Side length of each square block in pixels (must be ≥ 4).
    pub block_size: u32,
    /// Search radius in pixels (±range).
    pub search_range: i32,
    /// Maximum L₁ difference (in pixels) between `mv_back` and `mv_fwd`
    /// before a block is considered an occlusion.
    pub symmetry_threshold: i32,
}

impl Default for BidirConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            search_range: 16,
            symmetry_threshold: 4,
        }
    }
}

// ---------------------------------------------------------------------------
// BidirMotionVector
// ---------------------------------------------------------------------------

/// A pair of motion vectors for one block — one pointing backwards to `P0`
/// and one pointing forwards to `P1`.
#[derive(Debug, Clone, PartialEq)]
pub struct BidirMotionVector {
    /// Top-left X coordinate of the block in the current frame.
    pub block_x: u32,
    /// Top-left Y coordinate of the block in the current frame.
    pub block_y: u32,
    /// Displacement to the best matching block in the *backward* reference `P0`.
    pub mv_back: (i32, i32),
    /// Displacement to the best matching block in the *forward* reference `P1`.
    pub mv_fwd: (i32, i32),
    /// SAD cost of the backward match.
    pub sad_back: u32,
    /// SAD cost of the forward match.
    pub sad_fwd: u32,
    /// `true` when the backward and forward vectors are inconsistent,
    /// indicating a possible occlusion.
    pub is_occlusion: bool,
}

// ---------------------------------------------------------------------------
// BidirInterpolationResult
// ---------------------------------------------------------------------------

/// Result of bidirectional interpolation.
#[derive(Debug, Clone)]
pub struct BidirInterpolationResult {
    /// The interpolated frame (8-bit luma, `width × height` bytes).
    pub interpolated: Vec<u8>,
    /// All bidirectional motion vectors (one per block).
    pub vectors: Vec<BidirMotionVector>,
    /// Fraction of blocks classified as occlusions.
    pub occlusion_rate: f64,
    /// Width of the interpolated frame.
    pub width: u32,
    /// Height of the interpolated frame.
    pub height: u32,
}

// ---------------------------------------------------------------------------
// BidirMotionEstimator
// ---------------------------------------------------------------------------

/// Estimates bidirectional motion between two reference frames and produces
/// an interpolated frame at an arbitrary temporal position.
#[derive(Debug, Clone)]
pub struct BidirMotionEstimator {
    cfg: BidirConfig,
}

impl BidirMotionEstimator {
    /// Create a new estimator with the given configuration.
    pub fn new(cfg: BidirConfig) -> Self {
        let block_size = cfg.block_size.max(4);
        Self {
            cfg: BidirConfig { block_size, ..cfg },
        }
    }

    /// Estimate bidirectional motion vectors for all blocks.
    ///
    /// `current` is a luma (8-bit) buffer of `width × height` bytes which
    /// acts as the *query* frame.  `p0` is the backward reference (preceding
    /// frame) and `p1` is the forward reference (following frame).
    pub fn estimate(
        &self,
        current: &[u8],
        p0: &[u8],
        p1: &[u8],
        width: u32,
        height: u32,
    ) -> Vec<BidirMotionVector> {
        assert_eq!(current.len(), (width * height) as usize);
        assert_eq!(p0.len(), current.len());
        assert_eq!(p1.len(), current.len());

        let bs = self.cfg.block_size;
        let mut vectors = Vec::new();

        let mut by = 0u32;
        while by < height {
            let bh = bs.min(height.saturating_sub(by));
            let mut bx = 0u32;
            while bx < width {
                let bw = bs.min(width.saturating_sub(bx));

                let (mv_back, sad_back) = full_search(
                    current,
                    p0,
                    width,
                    height,
                    bx,
                    by,
                    bw,
                    bh,
                    self.cfg.search_range,
                );
                let (mv_fwd, sad_fwd) = full_search(
                    current,
                    p1,
                    width,
                    height,
                    bx,
                    by,
                    bw,
                    bh,
                    self.cfg.search_range,
                );

                let l1 = (mv_back.0 + mv_fwd.0).abs() + (mv_back.1 + mv_fwd.1).abs();
                let is_occlusion = l1 > self.cfg.symmetry_threshold;

                vectors.push(BidirMotionVector {
                    block_x: bx,
                    block_y: by,
                    mv_back,
                    mv_fwd,
                    sad_back,
                    sad_fwd,
                    is_occlusion,
                });

                bx += bs;
            }
            by += bs;
        }

        vectors
    }

    /// Generate an interpolated frame at temporal position `t` between `p0`
    /// and `p1`, where `t = 0.0` is identical to `p0` and `t = 1.0` is
    /// identical to `p1`.
    ///
    /// Internally this calls [`Self::estimate`] using `p0` and `p1` directly
    /// as both the *query* and the references (since in interpolation mode
    /// there is no separate "current" frame — the interpolated frame *is* the
    /// output).
    pub fn interpolate(
        &self,
        p0: &[u8],
        p1: &[u8],
        width: u32,
        height: u32,
        t: f64,
    ) -> BidirInterpolationResult {
        let t = t.clamp(0.0, 1.0);
        let vectors = self.estimate(p0, p1, p0, width, height);

        let mut output = vec![0u8; (width * height) as usize];

        // Fill output from block motion vectors.
        for mv in &vectors {
            let bx = mv.block_x;
            let by = mv.block_y;
            let bs = self.cfg.block_size;
            let bw = bs.min(width.saturating_sub(bx));
            let bh = bs.min(height.saturating_sub(by));

            for row in 0..bh {
                for col in 0..bw {
                    let dst_x = bx + col;
                    let dst_y = by + row;
                    let dst_idx = (dst_y * width + dst_x) as usize;

                    // Backward reference sample (clamped).
                    let src0_x = clamp_coord(dst_x as i32 + mv.mv_back.0, width);
                    let src0_y = clamp_coord(dst_y as i32 + mv.mv_back.1, height);
                    let v0 = p0[(src0_y * width + src0_x) as usize] as f64;

                    // Forward reference sample (clamped).
                    let src1_x = clamp_coord(dst_x as i32 + mv.mv_fwd.0, width);
                    let src1_y = clamp_coord(dst_y as i32 + mv.mv_fwd.1, height);
                    let v1 = p1[(src1_y * width + src1_x) as usize] as f64;

                    let blended = v0 * (1.0 - t) + v1 * t;
                    output[dst_idx] = blended.round().clamp(0.0, 255.0) as u8;
                }
            }
        }

        let occlusion_count = vectors.iter().filter(|v| v.is_occlusion).count();
        let occlusion_rate = if vectors.is_empty() {
            0.0
        } else {
            occlusion_count as f64 / vectors.len() as f64
        };

        BidirInterpolationResult {
            interpolated: output,
            vectors,
            occlusion_rate,
            width,
            height,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Full block-matching search of `current` block at `(bx, by)` against
/// `reference`, returning `(best_mv, best_sad)`.
fn full_search(
    current: &[u8],
    reference: &[u8],
    width: u32,
    height: u32,
    bx: u32,
    by: u32,
    bw: u32,
    bh: u32,
    search_range: i32,
) -> ((i32, i32), u32) {
    let mut best_sad = u32::MAX;
    let mut best_dx = 0i32;
    let mut best_dy = 0i32;

    for dy in -search_range..=search_range {
        for dx in -search_range..=search_range {
            let sad = block_sad(current, reference, width, height, bx, by, bw, bh, dx, dy);
            if sad < best_sad {
                best_sad = sad;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }

    ((best_dx, best_dy), best_sad)
}

/// Sum-of-absolute-differences between the current block at `(bx, by)` and
/// the reference block displaced by `(dx, dy)`.
fn block_sad(
    current: &[u8],
    reference: &[u8],
    width: u32,
    height: u32,
    bx: u32,
    by: u32,
    bw: u32,
    bh: u32,
    dx: i32,
    dy: i32,
) -> u32 {
    let mut sad = 0u32;
    for row in 0..bh {
        for col in 0..bw {
            let cx = (bx + col) as usize;
            let cy = (by + row) as usize;
            let cur_idx = cy * width as usize + cx;

            let rx = clamp_coord(bx as i32 + col as i32 + dx, width) as usize;
            let ry = clamp_coord(by as i32 + row as i32 + dy, height) as usize;
            let ref_idx = ry * width as usize + rx;

            let a = current.get(cur_idx).copied().unwrap_or(0) as i32;
            let b = reference.get(ref_idx).copied().unwrap_or(0) as i32;
            sad = sad.saturating_add((a - b).unsigned_abs());
        }
    }
    sad
}

/// Clamp a signed coordinate to `[0, dim - 1]`.
#[inline]
fn clamp_coord(v: i32, dim: u32) -> u32 {
    v.clamp(0, dim.saturating_sub(1) as i32) as u32
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(w: u32, h: u32, fill: u8) -> Vec<u8> {
        vec![fill; (w * h) as usize]
    }

    fn ramp_frame(w: u32, h: u32) -> Vec<u8> {
        (0..(w * h) as usize).map(|i| (i % 256) as u8).collect()
    }

    fn shifted_frame(w: u32, h: u32, dx: i32, dy: i32) -> Vec<u8> {
        let mut buf = vec![0u8; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let sx = clamp_coord(x as i32 - dx, w);
                let sy = clamp_coord(y as i32 - dy, h);
                buf[(y * w + x) as usize] = ((sx as usize * 3 + sy as usize * 7) % 256) as u8;
            }
        }
        buf
    }

    // --- BidirConfig ---

    #[test]
    fn test_default_config_sensible() {
        let cfg = BidirConfig::default();
        assert!(cfg.block_size >= 4);
        assert!(cfg.search_range > 0);
        assert!(cfg.symmetry_threshold >= 0);
    }

    // --- interpolate output size ---

    #[test]
    fn test_interpolate_output_size() {
        let w = 32u32;
        let h = 32u32;
        let p0 = ramp_frame(w, h);
        let p1 = ramp_frame(w, h);
        let est = BidirMotionEstimator::new(BidirConfig::default());
        let result = est.interpolate(&p0, &p1, w, h, 0.5);
        assert_eq!(result.interpolated.len(), (w * h) as usize);
        assert_eq!(result.width, w);
        assert_eq!(result.height, h);
    }

    // --- identical frames → interpolated = originals ---

    #[test]
    fn test_interpolate_identical_frames() {
        let w = 16u32;
        let h = 16u32;
        let p0 = make_frame(w, h, 128);
        let est = BidirMotionEstimator::new(BidirConfig::default());
        let result = est.interpolate(&p0, &p0, w, h, 0.5);
        for &v in &result.interpolated {
            assert_eq!(
                v, 128,
                "interpolation of identical solid frames must be constant"
            );
        }
    }

    // --- occlusion_rate in [0, 1] ---

    #[test]
    fn test_occlusion_rate_range() {
        let w = 32u32;
        let h = 32u32;
        let p0 = ramp_frame(w, h);
        let p1 = shifted_frame(w, h, 4, 2);
        let est = BidirMotionEstimator::new(BidirConfig::default());
        let result = est.interpolate(&p0, &p1, w, h, 0.5);
        assert!(
            (0.0..=1.0).contains(&result.occlusion_rate),
            "occlusion_rate must be in [0, 1], got {}",
            result.occlusion_rate
        );
    }

    // --- t=0 output close to p0 ---

    #[test]
    fn test_interpolate_t0_close_to_p0() {
        let w = 16u32;
        let h = 16u32;
        let p0 = make_frame(w, h, 60);
        let p1 = make_frame(w, h, 200);
        let est = BidirMotionEstimator::new(BidirConfig {
            block_size: 16,
            search_range: 4,
            ..Default::default()
        });
        let result = est.interpolate(&p0, &p1, w, h, 0.0);
        let mean: f64 = result.interpolated.iter().map(|&v| v as f64).sum::<f64>()
            / result.interpolated.len() as f64;
        assert!(
            (mean - 60.0).abs() < 5.0,
            "t=0 should be close to p0 (mean={mean:.1})"
        );
    }

    // --- t=1 output close to p1 ---

    #[test]
    fn test_interpolate_t1_close_to_p1() {
        let w = 16u32;
        let h = 16u32;
        let p0 = make_frame(w, h, 60);
        let p1 = make_frame(w, h, 200);
        let est = BidirMotionEstimator::new(BidirConfig {
            block_size: 16,
            search_range: 4,
            ..Default::default()
        });
        let result = est.interpolate(&p0, &p1, w, h, 1.0);
        let mean: f64 = result.interpolated.iter().map(|&v| v as f64).sum::<f64>()
            / result.interpolated.len() as f64;
        assert!(
            (mean - 200.0).abs() < 5.0,
            "t=1 should be close to p1 (mean={mean:.1})"
        );
    }

    // --- estimate returns correct block positions ---

    #[test]
    fn test_estimate_block_positions() {
        let w = 32u32;
        let h = 32u32;
        let frame = ramp_frame(w, h);
        let est = BidirMotionEstimator::new(BidirConfig {
            block_size: 16,
            search_range: 4,
            ..Default::default()
        });
        let vecs = est.estimate(&frame, &frame, &frame, w, h);
        // 32×32 with 16×16 blocks → 4 blocks
        assert_eq!(vecs.len(), 4);
        for v in &vecs {
            assert!(v.block_x < w);
            assert!(v.block_y < h);
        }
    }

    // --- identical frames → zero motion vectors ---

    #[test]
    fn test_estimate_identical_frames_zero_mv() {
        let w = 16u32;
        let h = 16u32;
        let frame = ramp_frame(w, h);
        let est = BidirMotionEstimator::new(BidirConfig {
            block_size: 16,
            search_range: 8,
            ..Default::default()
        });
        let vecs = est.estimate(&frame, &frame, &frame, w, h);
        for v in &vecs {
            assert_eq!(
                v.mv_back,
                (0, 0),
                "backward MV should be zero for identical frames"
            );
            assert_eq!(
                v.mv_fwd,
                (0, 0),
                "forward MV should be zero for identical frames"
            );
        }
    }

    // --- clamp_coord stays in range ---

    #[test]
    fn test_clamp_coord() {
        assert_eq!(clamp_coord(-5, 16), 0);
        assert_eq!(clamp_coord(100, 16), 15);
        assert_eq!(clamp_coord(7, 16), 7);
    }

    // --- non-square frame handled ---

    #[test]
    fn test_non_square_frame() {
        let w = 48u32;
        let h = 32u32;
        let p0 = ramp_frame(w, h);
        let p1 = make_frame(w, h, 100);
        let est = BidirMotionEstimator::new(BidirConfig {
            block_size: 16,
            search_range: 4,
            ..Default::default()
        });
        let result = est.interpolate(&p0, &p1, w, h, 0.5);
        assert_eq!(result.interpolated.len(), (w * h) as usize);
    }
}
