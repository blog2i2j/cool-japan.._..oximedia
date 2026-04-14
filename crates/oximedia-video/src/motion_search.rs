//! Fast motion estimation search patterns: diamond and hexagonal.
//!
//! Provides block-based motion estimation using several search strategies:
//!
//! - [`SearchPattern::Full`]: Exhaustive search — highest quality, slowest.
//! - [`SearchPattern::Diamond`]: Large diamond + small diamond refinement (EPZS-style).
//! - [`SearchPattern::Hexagonal`]: Hexagonal pattern (HEXBS) + small diamond refinement.
//! - [`SearchPattern::SmallDiamond`]: 3-step small diamond — fastest, for refinement.
//!
//! The module also exposes [`block_sad`] for computing the Sum of Absolute
//! Differences between two blocks within frame buffers, and [`search_frame`]
//! for scanning all blocks in a frame at once.

#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]

/// A motion vector (dx, dy) in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MotionVector {
    /// Horizontal displacement (positive = right).
    pub dx: i32,
    /// Vertical displacement (positive = down).
    pub dy: i32,
}

impl MotionVector {
    /// Construct a new motion vector.
    #[must_use]
    pub fn new(dx: i32, dy: i32) -> Self {
        Self { dx, dy }
    }

    /// Squared Euclidean distance from the zero vector.
    #[must_use]
    pub fn dist_sq(&self) -> i64 {
        i64::from(self.dx) * i64::from(self.dx) + i64::from(self.dy) * i64::from(self.dy)
    }
}

/// Search pattern type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchPattern {
    /// Full search (exhaustive) — highest quality, slowest.
    Full,
    /// Diamond search — good speed/quality tradeoff.
    #[default]
    Diamond,
    /// Hexagonal search — faster than diamond, slightly lower quality.
    Hexagonal,
    /// Small diamond (3-step) — fastest, for refinement after coarse search.
    SmallDiamond,
}

/// Motion search configuration.
#[derive(Debug, Clone)]
pub struct MotionSearchConfig {
    /// Search pattern to use.
    pub pattern: SearchPattern,
    /// Block size for matching (e.g., 16).
    pub block_size: u32,
    /// Maximum search range in pixels (e.g., +-32).
    pub search_range: i32,
    /// Sub-pixel refinement (false = integer-pel only).
    pub subpixel: bool,
}

impl Default for MotionSearchConfig {
    fn default() -> Self {
        Self {
            pattern: SearchPattern::Diamond,
            block_size: 16,
            search_range: 32,
            subpixel: false,
        }
    }
}

// ── Search pattern constants ─────────────────────────────────────────────────

/// Large Diamond Search Pattern (LDSP): center + 4 cardinal (+-2,0)/(0,+-2) + 4 diagonal (+-1,+-1).
const LARGE_DIAMOND: [(i32, i32); 9] = [
    (0, 0),
    (-2, 0),
    (2, 0),
    (0, -2),
    (0, 2),
    (-1, -1),
    (1, -1),
    (-1, 1),
    (1, 1),
];

/// Small Diamond Search Pattern (SDSP): center + 4 adjacent (+-1,0)/(0,+-1).
const SMALL_DIAMOND: [(i32, i32); 5] = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)];

/// Hexagonal pattern: 6 points at approximate 60-degree intervals.
const HEX_PATTERN: [(i32, i32); 6] = [(-2, 0), (-1, -2), (1, -2), (2, 0), (1, 2), (-1, 2)];

// ── SAD computation ──────────────────────────────────────────────────────────

/// Compute SAD (Sum of Absolute Differences) between two blocks.
///
/// The reference block starts at (`ref_x`, `ref_y`) in the reference frame and
/// the current block starts at (`cur_x`, `cur_y`) in the current frame.  Both
/// blocks have dimensions `block_size x block_size`.
///
/// Out-of-bounds coordinates are clamped to the nearest edge pixel.
///
/// # Parameters
///
/// * `ref_frame` — reference frame pixel data (row-major, stride = `ref_stride`)
/// * `ref_stride` — row stride of the reference frame in bytes
/// * `cur_frame` — current frame pixel data (row-major, stride = `cur_stride`)
/// * `cur_stride` — row stride of the current frame in bytes
/// * `ref_x`, `ref_y` — top-left of the candidate block in the reference frame
/// * `cur_x`, `cur_y` — top-left of the current block in the current frame
/// * `block_size` — width and height of the block (square)
/// * `frame_width`, `frame_height` — dimensions of both frames
///
/// # Returns
///
/// The total SAD value as `u64`.
#[allow(clippy::too_many_arguments)]
pub fn block_sad(
    ref_frame: &[u8],
    ref_stride: u32,
    cur_frame: &[u8],
    cur_stride: u32,
    ref_x: i32,
    ref_y: i32,
    cur_x: u32,
    cur_y: u32,
    block_size: u32,
    frame_width: u32,
    frame_height: u32,
) -> u64 {
    let max_x = frame_width.saturating_sub(1) as i32;
    let max_y = frame_height.saturating_sub(1) as i32;
    let bs = block_size as i32;
    let cx = cur_x as i32;
    let cy = cur_y as i32;

    let mut total: u64 = 0;

    for row in 0..bs {
        for col in 0..bs {
            // Current frame pixel
            let c_px = (cx + col).clamp(0, max_x) as usize;
            let c_py = (cy + row).clamp(0, max_y) as usize;
            let cur_idx = c_py * (cur_stride as usize) + c_px;

            // Reference frame pixel
            let r_px = (ref_x + col).clamp(0, max_x) as usize;
            let r_py = (ref_y + row).clamp(0, max_y) as usize;
            let ref_idx = r_py * (ref_stride as usize) + r_px;

            let a = if cur_idx < cur_frame.len() {
                cur_frame[cur_idx]
            } else {
                0
            };
            let b = if ref_idx < ref_frame.len() {
                ref_frame[ref_idx]
            } else {
                0
            };
            total += u64::from(a.abs_diff(b));
        }
    }
    total
}

// ── Internal SAD helper (takes i32 coords for both blocks) ───────────────────

/// Internal SAD computation using i32 coordinates for both blocks.
#[allow(clippy::too_many_arguments)]
fn sad_at(
    ref_frame: &[u8],
    ref_stride: u32,
    cur_frame: &[u8],
    cur_stride: u32,
    ref_x: i32,
    ref_y: i32,
    cur_x: i32,
    cur_y: i32,
    block_size: u32,
    frame_width: u32,
    frame_height: u32,
) -> u64 {
    let max_x = frame_width.saturating_sub(1) as i32;
    let max_y = frame_height.saturating_sub(1) as i32;
    let bs = block_size as i32;

    let mut total: u64 = 0;

    for row in 0..bs {
        for col in 0..bs {
            let c_px = (cur_x + col).clamp(0, max_x) as usize;
            let c_py = (cur_y + row).clamp(0, max_y) as usize;
            let cur_idx = c_py * (cur_stride as usize) + c_px;

            let r_px = (ref_x + col).clamp(0, max_x) as usize;
            let r_py = (ref_y + row).clamp(0, max_y) as usize;
            let ref_idx = r_py * (ref_stride as usize) + r_px;

            let a = if cur_idx < cur_frame.len() {
                cur_frame[cur_idx]
            } else {
                0
            };
            let b = if ref_idx < ref_frame.len() {
                ref_frame[ref_idx]
            } else {
                0
            };
            total += u64::from(a.abs_diff(b));
        }
    }
    total
}

// ── Full (exhaustive) search ─────────────────────────────────────────────────

/// Exhaustive full search over the integer-pel grid.
#[allow(clippy::too_many_arguments)]
fn full_search(
    ref_frame: &[u8],
    ref_stride: u32,
    cur_frame: &[u8],
    cur_stride: u32,
    block_x: i32,
    block_y: i32,
    config: &MotionSearchConfig,
    frame_width: u32,
    frame_height: u32,
) -> (MotionVector, u64) {
    let sr = config.search_range;
    let mut best_mv = MotionVector::default();
    let mut best_sad = u64::MAX;

    for dy in -sr..=sr {
        for dx in -sr..=sr {
            let rx = block_x + dx;
            let ry = block_y + dy;
            let cost = sad_at(
                ref_frame,
                ref_stride,
                cur_frame,
                cur_stride,
                rx,
                ry,
                block_x,
                block_y,
                config.block_size,
                frame_width,
                frame_height,
            );
            if cost < best_sad {
                best_sad = cost;
                best_mv = MotionVector::new(dx, dy);
            }
        }
    }
    (best_mv, best_sad)
}

// ── Diamond search ───────────────────────────────────────────────────────────

/// Iterative diamond search: LDSP until center wins, then SDSP refinement.
#[allow(clippy::too_many_arguments)]
fn diamond_search(
    ref_frame: &[u8],
    ref_stride: u32,
    cur_frame: &[u8],
    cur_stride: u32,
    block_x: i32,
    block_y: i32,
    config: &MotionSearchConfig,
    frame_width: u32,
    frame_height: u32,
) -> (MotionVector, u64) {
    let sr = config.search_range;
    let bs = config.block_size;

    // Start at center (0,0)
    let mut best_mv = MotionVector::default();
    let mut best_sad = sad_at(
        ref_frame,
        ref_stride,
        cur_frame,
        cur_stride,
        block_x,
        block_y,
        block_x,
        block_y,
        bs,
        frame_width,
        frame_height,
    );

    // Large diamond iterations
    let max_iters = (sr * 2).max(8) as usize;
    let mut changed = true;
    let mut iter = 0usize;

    while changed && iter < max_iters {
        changed = false;
        iter += 1;

        for &(ox, oy) in &LARGE_DIAMOND[1..] {
            let candidate_dx = best_mv.dx + ox;
            let candidate_dy = best_mv.dy + oy;
            if candidate_dx.abs() > sr || candidate_dy.abs() > sr {
                continue;
            }
            let rx = block_x + candidate_dx;
            let ry = block_y + candidate_dy;
            let cost = sad_at(
                ref_frame,
                ref_stride,
                cur_frame,
                cur_stride,
                rx,
                ry,
                block_x,
                block_y,
                bs,
                frame_width,
                frame_height,
            );
            if cost < best_sad {
                best_sad = cost;
                best_mv = MotionVector::new(candidate_dx, candidate_dy);
                changed = true;
            }
        }
    }

    // Small diamond refinement
    small_diamond_refine(
        ref_frame,
        ref_stride,
        cur_frame,
        cur_stride,
        block_x,
        block_y,
        bs,
        sr,
        frame_width,
        frame_height,
        best_mv,
        best_sad,
    )
}

// ── Hexagonal search ─────────────────────────────────────────────────────────

/// Multi-hexagon search: hex iterations until center wins, then SDSP refinement.
#[allow(clippy::too_many_arguments)]
fn hexagonal_search(
    ref_frame: &[u8],
    ref_stride: u32,
    cur_frame: &[u8],
    cur_stride: u32,
    block_x: i32,
    block_y: i32,
    config: &MotionSearchConfig,
    frame_width: u32,
    frame_height: u32,
) -> (MotionVector, u64) {
    let sr = config.search_range;
    let bs = config.block_size;

    let mut best_mv = MotionVector::default();
    let mut best_sad = sad_at(
        ref_frame,
        ref_stride,
        cur_frame,
        cur_stride,
        block_x,
        block_y,
        block_x,
        block_y,
        bs,
        frame_width,
        frame_height,
    );

    let max_iters = (sr * 2).max(8) as usize;
    let mut changed = true;
    let mut iter = 0usize;

    while changed && iter < max_iters {
        changed = false;
        iter += 1;

        for &(ox, oy) in &HEX_PATTERN {
            let candidate_dx = best_mv.dx + ox;
            let candidate_dy = best_mv.dy + oy;
            if candidate_dx.abs() > sr || candidate_dy.abs() > sr {
                continue;
            }
            let rx = block_x + candidate_dx;
            let ry = block_y + candidate_dy;
            let cost = sad_at(
                ref_frame,
                ref_stride,
                cur_frame,
                cur_stride,
                rx,
                ry,
                block_x,
                block_y,
                bs,
                frame_width,
                frame_height,
            );
            if cost < best_sad {
                best_sad = cost;
                best_mv = MotionVector::new(candidate_dx, candidate_dy);
                changed = true;
            }
        }
    }

    // Final small diamond refinement
    small_diamond_refine(
        ref_frame,
        ref_stride,
        cur_frame,
        cur_stride,
        block_x,
        block_y,
        bs,
        sr,
        frame_width,
        frame_height,
        best_mv,
        best_sad,
    )
}

// ── Small diamond search (standalone 3-step) ─────────────────────────────────

/// 3-step small diamond search starting from (0,0).
#[allow(clippy::too_many_arguments)]
fn small_diamond_search(
    ref_frame: &[u8],
    ref_stride: u32,
    cur_frame: &[u8],
    cur_stride: u32,
    block_x: i32,
    block_y: i32,
    config: &MotionSearchConfig,
    frame_width: u32,
    frame_height: u32,
) -> (MotionVector, u64) {
    let sr = config.search_range;
    let bs = config.block_size;

    let mut best_mv = MotionVector::default();
    let mut best_sad = sad_at(
        ref_frame,
        ref_stride,
        cur_frame,
        cur_stride,
        block_x,
        block_y,
        block_x,
        block_y,
        bs,
        frame_width,
        frame_height,
    );

    // 3 refinement steps
    for _step in 0..3 {
        let (mv, cost) = small_diamond_refine(
            ref_frame,
            ref_stride,
            cur_frame,
            cur_stride,
            block_x,
            block_y,
            bs,
            sr,
            frame_width,
            frame_height,
            best_mv,
            best_sad,
        );
        if mv == best_mv {
            break;
        }
        best_mv = mv;
        best_sad = cost;
    }

    (best_mv, best_sad)
}

// ── Small diamond refinement helper ──────────────────────────────────────────

/// Single pass of small diamond refinement around the current best MV.
#[allow(clippy::too_many_arguments)]
fn small_diamond_refine(
    ref_frame: &[u8],
    ref_stride: u32,
    cur_frame: &[u8],
    cur_stride: u32,
    block_x: i32,
    block_y: i32,
    block_size: u32,
    search_range: i32,
    frame_width: u32,
    frame_height: u32,
    mut best_mv: MotionVector,
    mut best_sad: u64,
) -> (MotionVector, u64) {
    for &(ox, oy) in &SMALL_DIAMOND[1..] {
        let candidate_dx = best_mv.dx + ox;
        let candidate_dy = best_mv.dy + oy;
        if candidate_dx.abs() > search_range || candidate_dy.abs() > search_range {
            continue;
        }
        let rx = block_x + candidate_dx;
        let ry = block_y + candidate_dy;
        let cost = sad_at(
            ref_frame,
            ref_stride,
            cur_frame,
            cur_stride,
            rx,
            ry,
            block_x,
            block_y,
            block_size,
            frame_width,
            frame_height,
        );
        if cost < best_sad {
            best_sad = cost;
            best_mv = MotionVector::new(candidate_dx, candidate_dy);
        }
    }
    (best_mv, best_sad)
}

// ── Public search API ────────────────────────────────────────────────────────

/// Search for the best motion vector for a single block.
///
/// The block at position (`block_x`, `block_y`) in the current frame is matched
/// against the reference frame using the configured search pattern.
///
/// # Returns
///
/// `(best_mv, best_sad)` — the best motion vector and its SAD cost.
#[allow(clippy::too_many_arguments)]
pub fn search_block(
    ref_frame: &[u8],
    ref_stride: u32,
    cur_frame: &[u8],
    cur_stride: u32,
    block_x: u32,
    block_y: u32,
    config: &MotionSearchConfig,
    frame_width: u32,
    frame_height: u32,
) -> (MotionVector, u64) {
    let bx = block_x as i32;
    let by = block_y as i32;

    match config.pattern {
        SearchPattern::Full => full_search(
            ref_frame,
            ref_stride,
            cur_frame,
            cur_stride,
            bx,
            by,
            config,
            frame_width,
            frame_height,
        ),
        SearchPattern::Diamond => diamond_search(
            ref_frame,
            ref_stride,
            cur_frame,
            cur_stride,
            bx,
            by,
            config,
            frame_width,
            frame_height,
        ),
        SearchPattern::Hexagonal => hexagonal_search(
            ref_frame,
            ref_stride,
            cur_frame,
            cur_stride,
            bx,
            by,
            config,
            frame_width,
            frame_height,
        ),
        SearchPattern::SmallDiamond => small_diamond_search(
            ref_frame,
            ref_stride,
            cur_frame,
            cur_stride,
            bx,
            by,
            config,
            frame_width,
            frame_height,
        ),
    }
}

/// Search all blocks in a frame and return the motion field.
///
/// The frame is divided into `block_size x block_size` non-overlapping blocks.
/// Each block is searched independently using the configured pattern.
///
/// Both `ref_frame` and `cur_frame` are assumed to be row-major with stride =
/// `width`.
///
/// # Returns
///
/// A vector of `(MotionVector, sad)` tuples, one per block, in raster order
/// (left-to-right, top-to-bottom).
pub fn search_frame(
    ref_frame: &[u8],
    cur_frame: &[u8],
    width: u32,
    height: u32,
    config: &MotionSearchConfig,
) -> Vec<(MotionVector, u64)> {
    let bs = config.block_size;
    if bs == 0 {
        return Vec::new();
    }

    let blocks_x = width / bs;
    let blocks_y = height / bs;
    let total = (blocks_x * blocks_y) as usize;
    let mut results = Vec::with_capacity(total);

    for by_idx in 0..blocks_y {
        for bx_idx in 0..blocks_x {
            let block_x = bx_idx * bs;
            let block_y = by_idx * bs;
            let (mv, sad) = search_block(
                ref_frame, width, cur_frame, width, block_x, block_y, config, width, height,
            );
            results.push((mv, sad));
        }
    }

    results
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(width: usize, height: usize, fill: u8) -> Vec<u8> {
        vec![fill; width * height]
    }

    fn make_shifted_frame(
        width: usize,
        height: usize,
        shift_x: usize,
        shift_y: usize,
    ) -> (Vec<u8>, Vec<u8>) {
        // Create a current frame with a distinctive 16x16 block at (0,0)
        // and a reference frame with the same block at (shift_x, shift_y)
        let mut cur = vec![0u8; width * height];
        let mut reference = vec![0u8; width * height];

        // Paint a distinctive pattern in the current frame at (0,0)
        for r in 0..16 {
            for c in 0..16 {
                if r < height && c < width {
                    cur[r * width + c] = 200;
                }
            }
        }
        // Paint the same pattern shifted in the reference frame
        for r in 0..16 {
            for c in 0..16 {
                let ry = r + shift_y;
                let rx = c + shift_x;
                if ry < height && rx < width {
                    reference[ry * width + rx] = 200;
                }
            }
        }
        (cur, reference)
    }

    #[test]
    fn test_diamond_search_zero_motion() {
        let frame = make_frame(64, 64, 128);
        let config = MotionSearchConfig {
            pattern: SearchPattern::Diamond,
            block_size: 16,
            search_range: 16,
            subpixel: false,
        };
        let (mv, sad) = search_block(&frame, 64, &frame, 64, 0, 0, &config, 64, 64);
        assert_eq!(sad, 0, "identical frames should give SAD=0");
        assert_eq!(mv.dx, 0);
        assert_eq!(mv.dy, 0);
    }

    #[test]
    fn test_diamond_search_known_shift() {
        let (cur, reference) = make_shifted_frame(128, 128, 4, 4);
        let config = MotionSearchConfig {
            pattern: SearchPattern::Diamond,
            block_size: 16,
            search_range: 16,
            subpixel: false,
        };
        let (mv, sad) = search_block(&reference, 128, &cur, 128, 0, 0, &config, 128, 128);
        assert_eq!(sad, 0, "should find perfect match");
        assert_eq!(mv.dx, 4);
        assert_eq!(mv.dy, 4);
    }

    #[test]
    fn test_hexagonal_search_zero_motion() {
        let frame = make_frame(64, 64, 64);
        let config = MotionSearchConfig {
            pattern: SearchPattern::Hexagonal,
            block_size: 16,
            search_range: 16,
            subpixel: false,
        };
        let (mv, sad) = search_block(&frame, 64, &frame, 64, 0, 0, &config, 64, 64);
        assert_eq!(sad, 0);
        assert_eq!(mv.dx, 0);
        assert_eq!(mv.dy, 0);
    }

    #[test]
    fn test_full_search_optimal() {
        // Full search must find the global minimum within search_range
        let (cur, reference) = make_shifted_frame(128, 128, 6, 6);
        let config = MotionSearchConfig {
            pattern: SearchPattern::Full,
            block_size: 16,
            search_range: 16,
            subpixel: false,
        };
        let (mv, sad) = search_block(&reference, 128, &cur, 128, 0, 0, &config, 128, 128);
        assert_eq!(sad, 0, "full search should find perfect match");
        assert_eq!(mv.dx, 6);
        assert_eq!(mv.dy, 6);
    }

    #[test]
    fn test_block_sad_identical_zero() {
        let frame = make_frame(64, 64, 100);
        let sad = block_sad(&frame, 64, &frame, 64, 0, 0, 0, 0, 16, 64, 64);
        assert_eq!(sad, 0, "identical blocks should give SAD=0");
    }

    #[test]
    fn test_block_sad_opposite_max() {
        let black = make_frame(64, 64, 0);
        let white = make_frame(64, 64, 255);
        let sad = block_sad(&black, 64, &white, 64, 0, 0, 0, 0, 16, 64, 64);
        assert_eq!(sad, 16 * 16 * 255, "black vs white should give max SAD");
    }

    #[test]
    fn test_search_frame_produces_vectors() {
        let frame = make_frame(64, 64, 50);
        let config = MotionSearchConfig {
            pattern: SearchPattern::Diamond,
            block_size: 16,
            search_range: 8,
            subpixel: false,
        };
        let results = search_frame(&frame, &frame, 64, 64, &config);
        // 64/16 = 4 blocks in each dimension → 4*4 = 16
        assert_eq!(results.len(), 16, "should produce one vector per block");
        // All identical → all SADs should be 0
        for (mv, sad) in &results {
            assert_eq!(*sad, 0);
            assert_eq!(mv.dx, 0);
            assert_eq!(mv.dy, 0);
        }
    }

    #[test]
    fn test_small_diamond_refinement() {
        let frame = make_frame(64, 64, 128);
        let config = MotionSearchConfig {
            pattern: SearchPattern::SmallDiamond,
            block_size: 16,
            search_range: 8,
            subpixel: false,
        };
        let (mv, sad) = search_block(&frame, 64, &frame, 64, 0, 0, &config, 64, 64);
        assert_eq!(sad, 0);
        assert_eq!(mv.dx, 0);
        assert_eq!(mv.dy, 0);
    }

    #[test]
    fn test_search_range_respected() {
        // Place a matching block far outside search_range
        let (cur, reference) = make_shifted_frame(128, 128, 30, 30);
        let config = MotionSearchConfig {
            pattern: SearchPattern::Diamond,
            block_size: 16,
            search_range: 8,
            subpixel: false,
        };
        let (mv, _sad) = search_block(&reference, 128, &cur, 128, 0, 0, &config, 128, 128);
        assert!(
            mv.dx.abs() <= config.search_range,
            "dx={} exceeded search_range={}",
            mv.dx,
            config.search_range,
        );
        assert!(
            mv.dy.abs() <= config.search_range,
            "dy={} exceeded search_range={}",
            mv.dy,
            config.search_range,
        );
    }

    #[test]
    fn test_default_config() {
        let config = MotionSearchConfig::default();
        assert_eq!(config.pattern, SearchPattern::Diamond);
        assert_eq!(config.block_size, 16);
        assert_eq!(config.search_range, 32);
        assert!(!config.subpixel);
    }
}
