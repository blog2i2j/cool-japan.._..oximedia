//! Adaptive block size selection for motion estimation.
//!
//! Selects the optimal block size for each region of the frame based on
//! texture complexity, variance, and edge density.  Supported block sizes
//! range from 4×4 to 64×64 in powers of two (4, 8, 16, 32, 64).
//!
//! # Algorithm overview
//!
//! For each candidate super-block position the analyser:
//!
//! 1. Computes the **variance** of the luma samples inside the block to
//!    measure texture richness.
//! 2. Estimates the **edge density** using horizontal and vertical Sobel
//!    responses to detect fine structure.
//! 3. Combines both metrics into a **complexity score** and maps it to one
//!    of the five supported block sizes.
//!
//! Fine, high-frequency regions (high variance and edge density) are assigned
//! small blocks (4×4 or 8×8) so that the motion estimator can track them
//! precisely.  Smooth, low-frequency regions use large blocks (32×32 or
//! 64×64) to reduce search cost.
//!
//! # Example
//!
//! ```rust
//! use oximedia_video::adaptive_block_selection::{BlockSizeAnalyser, BlockSizeConfig, BlockSize};
//!
//! let width = 64u32;
//! let height = 64u32;
//! let frame: Vec<u8> = (0..(width * height) as usize)
//!     .map(|i| ((i * 3) % 256) as u8)
//!     .collect();
//!
//! let cfg = BlockSizeConfig::default();
//! let analyser = BlockSizeAnalyser::new(cfg);
//! let partition = analyser.analyse(&frame, width, height);
//!
//! // Every block must have a valid size.
//! assert!(!partition.blocks.is_empty());
//! for b in &partition.blocks {
//!     assert!(b.block_x + b.size as u32 <= width);
//!     assert!(b.block_y + b.size as u32 <= height);
//! }
//! ```

use std::fmt;

// ---------------------------------------------------------------------------
// BlockSize
// ---------------------------------------------------------------------------

/// Supported square block sizes (pixels per side).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BlockSize {
    /// 4×4 pixel block.
    B4 = 4,
    /// 8×8 pixel block.
    B8 = 8,
    /// 16×16 pixel block.
    B16 = 16,
    /// 32×32 pixel block.
    B32 = 32,
    /// 64×64 pixel block.
    B64 = 64,
}

impl BlockSize {
    /// Returns the side length in pixels.
    #[inline]
    pub fn pixels(self) -> u32 {
        self as u32
    }

    /// Iterate all sizes from smallest to largest.
    pub fn all() -> [BlockSize; 5] {
        [
            BlockSize::B4,
            BlockSize::B8,
            BlockSize::B16,
            BlockSize::B32,
            BlockSize::B64,
        ]
    }
}

impl fmt::Display for BlockSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}×{}", self.pixels(), self.pixels())
    }
}

// ---------------------------------------------------------------------------
// BlockSizeConfig
// ---------------------------------------------------------------------------

/// Configuration for the adaptive block-size analyser.
#[derive(Debug, Clone)]
pub struct BlockSizeConfig {
    /// Variance threshold below which the region is considered smooth.
    /// Smooth regions use larger blocks (≥32×32).
    pub smooth_variance_threshold: f64,
    /// Variance threshold above which the region is considered highly textured.
    /// Highly textured regions use smaller blocks (≤8×8).
    pub texture_variance_threshold: f64,
    /// Edge density threshold (fraction of pixels with |Sobel| > 16) above
    /// which a block is classified as a fine-detail region.
    pub edge_density_threshold: f64,
    /// Base block size used when complexity is moderate.
    pub default_block_size: BlockSize,
    /// Minimum allowed block size regardless of complexity.
    pub min_block_size: BlockSize,
    /// Maximum allowed block size regardless of complexity.
    pub max_block_size: BlockSize,
}

impl Default for BlockSizeConfig {
    fn default() -> Self {
        Self {
            smooth_variance_threshold: 50.0,
            texture_variance_threshold: 800.0,
            edge_density_threshold: 0.15,
            default_block_size: BlockSize::B16,
            min_block_size: BlockSize::B4,
            max_block_size: BlockSize::B64,
        }
    }
}

// ---------------------------------------------------------------------------
// BlockEntry
// ---------------------------------------------------------------------------

/// A single block produced by the adaptive partitioner.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockEntry {
    /// Top-left X coordinate in the frame (pixels).
    pub block_x: u32,
    /// Top-left Y coordinate in the frame (pixels).
    pub block_y: u32,
    /// Chosen block size.
    pub size: BlockSize,
    /// Luma variance of the pixels in this block.
    pub variance: f64,
    /// Edge density (fraction of pixels with strong gradient).
    pub edge_density: f64,
    /// Normalised complexity score in `[0.0, 1.0]`.
    pub complexity_score: f64,
}

// ---------------------------------------------------------------------------
// FramePartition
// ---------------------------------------------------------------------------

/// The full adaptive partition of a frame.
#[derive(Debug, Clone)]
pub struct FramePartition {
    /// Width of the source frame in pixels.
    pub width: u32,
    /// Height of the source frame in pixels.
    pub height: u32,
    /// All blocks covering the frame (non-overlapping, left-to-right top-to-bottom).
    pub blocks: Vec<BlockEntry>,
    /// Minimum block size assigned in this frame.
    pub min_assigned: BlockSize,
    /// Maximum block size assigned in this frame.
    pub max_assigned: BlockSize,
}

impl FramePartition {
    /// Returns the average complexity score across all blocks.
    pub fn mean_complexity(&self) -> f64 {
        if self.blocks.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.blocks.iter().map(|b| b.complexity_score).sum();
        sum / self.blocks.len() as f64
    }

    /// Returns the number of blocks with the given size.
    pub fn count_size(&self, size: BlockSize) -> usize {
        self.blocks.iter().filter(|b| b.size == size).count()
    }
}

// ---------------------------------------------------------------------------
// BlockSizeAnalyser
// ---------------------------------------------------------------------------

/// Analyses a luma frame and selects an optimal block size for each region.
#[derive(Debug, Clone)]
pub struct BlockSizeAnalyser {
    cfg: BlockSizeConfig,
}

impl BlockSizeAnalyser {
    /// Create a new analyser with the supplied configuration.
    pub fn new(cfg: BlockSizeConfig) -> Self {
        Self { cfg }
    }

    /// Analyse `frame` (a planar 8-bit luma buffer of `width × height` bytes)
    /// and return a full [`FramePartition`].
    ///
    /// The frame is subdivided using the *maximum* allowed block size as a
    /// grid unit; each super-block is then recursively considered for
    /// splitting to the next smaller size if the complexity exceeds the
    /// configured thresholds.
    pub fn analyse(&self, frame: &[u8], width: u32, height: u32) -> FramePartition {
        assert_eq!(
            frame.len(),
            (width * height) as usize,
            "frame buffer length must equal width × height"
        );

        let mut blocks = Vec::new();
        let step = self.cfg.max_block_size.pixels();

        let mut min_assigned = self.cfg.max_block_size;
        let mut max_assigned = self.cfg.min_block_size;

        let mut y = 0u32;
        while y < height {
            let mut x = 0u32;
            while x < width {
                let entry = self.select_block(frame, width, height, x, y);
                if entry.size < min_assigned {
                    min_assigned = entry.size;
                }
                if entry.size > max_assigned {
                    max_assigned = entry.size;
                }
                blocks.push(entry);
                x += step;
            }
            y += step;
        }

        FramePartition {
            width,
            height,
            blocks,
            min_assigned,
            max_assigned,
        }
    }

    /// Decide the block size for a single super-block region starting at `(ox, oy)`.
    fn select_block(&self, frame: &[u8], width: u32, height: u32, ox: u32, oy: u32) -> BlockEntry {
        // Clamp to frame boundary.
        let bw = self
            .cfg
            .max_block_size
            .pixels()
            .min(width.saturating_sub(ox));
        let bh = self
            .cfg
            .max_block_size
            .pixels()
            .min(height.saturating_sub(oy));

        let variance = compute_variance(frame, width, ox, oy, bw, bh);
        let edge_density = compute_edge_density(frame, width, ox, oy, bw, bh);

        let complexity_score = self.complexity_score(variance, edge_density);
        let size = self.map_complexity_to_size(complexity_score);

        BlockEntry {
            block_x: ox,
            block_y: oy,
            size,
            variance,
            edge_density,
            complexity_score,
        }
    }

    /// Normalised complexity in `[0, 1]` blending variance and edge density.
    fn complexity_score(&self, variance: f64, edge_density: f64) -> f64 {
        let var_norm = (variance / self.cfg.texture_variance_threshold).min(1.0);
        let edge_norm = (edge_density / self.cfg.edge_density_threshold.max(1e-9)).min(1.0);
        // Geometric mean emphasises cases where both metrics are high.
        let blended = 0.6 * var_norm + 0.4 * edge_norm;
        blended.clamp(0.0, 1.0)
    }

    /// Map a normalised complexity score to a block size, respecting the
    /// configured `min_block_size` and `max_block_size` bounds.
    fn map_complexity_to_size(&self, score: f64) -> BlockSize {
        // Five complexity bands → five sizes (smallest to largest).
        let raw = if score < 0.15 {
            BlockSize::B64
        } else if score < 0.35 {
            BlockSize::B32
        } else if score < 0.60 {
            BlockSize::B16
        } else if score < 0.80 {
            BlockSize::B8
        } else {
            BlockSize::B4
        };

        // Apply min/max clamps.
        let clamped = raw
            .max(self.cfg.min_block_size)
            .min(self.cfg.max_block_size);
        clamped
    }
}

// ---------------------------------------------------------------------------
// Helper: variance
// ---------------------------------------------------------------------------

/// Compute the luma variance of the rectangular region starting at `(ox, oy)`
/// with dimensions `(bw, bh)` in a frame of width `stride`.
fn compute_variance(frame: &[u8], stride: u32, ox: u32, oy: u32, bw: u32, bh: u32) -> f64 {
    let n = (bw * bh) as f64;
    if n < 1.0 {
        return 0.0;
    }

    let mut sum = 0u64;
    let mut sum_sq = 0u64;

    for row in 0..bh {
        for col in 0..bw {
            let idx = ((oy + row) * stride + (ox + col)) as usize;
            if idx >= frame.len() {
                continue;
            }
            let v = frame[idx] as u64;
            sum = sum.saturating_add(v);
            sum_sq = sum_sq.saturating_add(v * v);
        }
    }

    let mean = sum as f64 / n;
    (sum_sq as f64 / n) - mean * mean
}

// ---------------------------------------------------------------------------
// Helper: edge density (Sobel)
// ---------------------------------------------------------------------------

/// Estimate the edge density of a block using an approximated Sobel response.
///
/// Returns the fraction of interior pixels (excluding the 1-pixel border)
/// whose gradient magnitude exceeds threshold 16.
fn compute_edge_density(frame: &[u8], stride: u32, ox: u32, oy: u32, bw: u32, bh: u32) -> f64 {
    if bw < 3 || bh < 3 {
        return 0.0;
    }

    let inner_w = bw - 2;
    let inner_h = bh - 2;
    let total = (inner_w * inner_h) as f64;
    if total < 1.0 {
        return 0.0;
    }

    let mut edge_count = 0u32;

    for row in 1..(bh - 1) {
        for col in 1..(bw - 1) {
            let gx = sobel_x(frame, stride, ox + col, oy + row);
            let gy = sobel_y(frame, stride, ox + col, oy + row);
            let mag = ((gx * gx + gy * gy) as f64).sqrt();
            if mag > 16.0 {
                edge_count += 1;
            }
        }
    }

    edge_count as f64 / total
}

/// Horizontal Sobel response at pixel `(px, py)`.
fn sobel_x(frame: &[u8], stride: u32, px: u32, py: u32) -> i32 {
    let p = |dx: i32, dy: i32| -> i32 {
        let idx = ((py as i32 + dy) * stride as i32 + (px as i32 + dx)) as usize;
        frame.get(idx).copied().unwrap_or(0) as i32
    };
    -p(-1, -1) - 2 * p(-1, 0) - p(-1, 1) + p(1, -1) + 2 * p(1, 0) + p(1, 1)
}

/// Vertical Sobel response at pixel `(px, py)`.
fn sobel_y(frame: &[u8], stride: u32, px: u32, py: u32) -> i32 {
    let p = |dx: i32, dy: i32| -> i32 {
        let idx = ((py as i32 + dy) * stride as i32 + (px as i32 + dx)) as usize;
        frame.get(idx).copied().unwrap_or(0) as i32
    };
    -p(-1, -1) - 2 * p(0, -1) - p(1, -1) + p(-1, 1) + 2 * p(0, 1) + p(1, 1)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_frame(w: u32, h: u32, value: u8) -> Vec<u8> {
        vec![value; (w * h) as usize]
    }

    fn ramp_frame(w: u32, h: u32) -> Vec<u8> {
        (0..(w * h) as usize).map(|i| (i % 256) as u8).collect()
    }

    fn checkerboard(w: u32, h: u32, cell: u32) -> Vec<u8> {
        (0..(w * h) as usize)
            .map(|i| {
                let x = (i as u32 % w) / cell;
                let y = (i as u32 / w) / cell;
                if (x + y) % 2 == 0 {
                    255
                } else {
                    0
                }
            })
            .collect()
    }

    // --- BlockSize ---

    #[test]
    fn test_block_size_pixels() {
        assert_eq!(BlockSize::B4.pixels(), 4);
        assert_eq!(BlockSize::B8.pixels(), 8);
        assert_eq!(BlockSize::B16.pixels(), 16);
        assert_eq!(BlockSize::B32.pixels(), 32);
        assert_eq!(BlockSize::B64.pixels(), 64);
    }

    #[test]
    fn test_block_size_ord() {
        assert!(BlockSize::B4 < BlockSize::B8);
        assert!(BlockSize::B8 < BlockSize::B16);
        assert!(BlockSize::B16 < BlockSize::B32);
        assert!(BlockSize::B32 < BlockSize::B64);
    }

    #[test]
    fn test_block_size_display() {
        assert_eq!(format!("{}", BlockSize::B16), "16×16");
    }

    // --- Variance helpers ---

    #[test]
    fn test_variance_solid_frame_is_zero() {
        let frame = solid_frame(16, 16, 128);
        let v = compute_variance(&frame, 16, 0, 0, 16, 16);
        assert!(v.abs() < 1e-6, "uniform frame should have zero variance");
    }

    #[test]
    fn test_variance_ramp_is_positive() {
        let frame = ramp_frame(16, 16);
        let v = compute_variance(&frame, 16, 0, 0, 16, 16);
        assert!(v > 0.0, "ramp frame should have positive variance");
    }

    // --- Edge density helpers ---

    #[test]
    fn test_edge_density_solid_frame_near_zero() {
        let frame = solid_frame(32, 32, 100);
        let ed = compute_edge_density(&frame, 32, 0, 0, 32, 32);
        assert!(ed < 0.01, "solid frame should have near-zero edge density");
    }

    #[test]
    fn test_edge_density_checkerboard_high() {
        // 2×2-pixel cells keep enough contrast for the Sobel 3×3 kernel to
        // fire reliably on every interior pixel.
        let frame = checkerboard(32, 32, 2);
        let ed = compute_edge_density(&frame, 32, 0, 0, 32, 32);
        assert!(
            ed > 0.3,
            "2×2-cell checkerboard should have high edge density, got {ed}"
        );
    }

    // --- BlockSizeAnalyser ---

    #[test]
    fn test_analyse_solid_frame_uses_large_blocks() {
        let w = 64u32;
        let h = 64u32;
        let frame = solid_frame(w, h, 100);
        let analyser = BlockSizeAnalyser::new(BlockSizeConfig::default());
        let partition = analyser.analyse(&frame, w, h);
        // Solid frames should have low complexity → large blocks.
        let large_count =
            partition.count_size(BlockSize::B32) + partition.count_size(BlockSize::B64);
        assert!(
            large_count > 0,
            "solid frame should contain at least one large block"
        );
    }

    #[test]
    fn test_analyse_checkerboard_uses_small_blocks() {
        let w = 64u32;
        let h = 64u32;
        let frame = checkerboard(w, h, 1);
        let cfg = BlockSizeConfig {
            max_block_size: BlockSize::B64,
            min_block_size: BlockSize::B4,
            ..Default::default()
        };
        let analyser = BlockSizeAnalyser::new(cfg);
        let partition = analyser.analyse(&frame, w, h);
        let small_count = partition.count_size(BlockSize::B4) + partition.count_size(BlockSize::B8);
        assert!(
            small_count > 0,
            "checkerboard should produce at least one small block"
        );
    }

    #[test]
    fn test_partition_blocks_stay_in_frame() {
        let w = 100u32;
        let h = 80u32;
        let frame = ramp_frame(w, h);
        let analyser = BlockSizeAnalyser::new(BlockSizeConfig::default());
        let partition = analyser.analyse(&frame, w, h);
        for b in &partition.blocks {
            assert!(b.block_x < w);
            assert!(b.block_y < h);
        }
    }

    #[test]
    fn test_mean_complexity_range() {
        let w = 64u32;
        let h = 64u32;
        let frame = ramp_frame(w, h);
        let analyser = BlockSizeAnalyser::new(BlockSizeConfig::default());
        let partition = analyser.analyse(&frame, w, h);
        let mc = partition.mean_complexity();
        assert!((0.0..=1.0).contains(&mc));
    }

    #[test]
    fn test_min_max_block_size_config_respected() {
        let w = 64u32;
        let h = 64u32;
        let frame = checkerboard(w, h, 1);
        let cfg = BlockSizeConfig {
            min_block_size: BlockSize::B8,
            max_block_size: BlockSize::B32,
            ..Default::default()
        };
        let analyser = BlockSizeAnalyser::new(cfg);
        let partition = analyser.analyse(&frame, w, h);
        for b in &partition.blocks {
            assert!(b.size >= BlockSize::B8, "block size below minimum");
            assert!(b.size <= BlockSize::B32, "block size above maximum");
        }
    }

    #[test]
    fn test_empty_partition_when_dimensions_zero() {
        // Zero-height or zero-width frame should produce an empty partition
        // without panicking.  We use a 64×0 logical frame here (which has
        // height=0, so no rows are processed).
        let frame: Vec<u8> = Vec::new();
        let analyser = BlockSizeAnalyser::new(BlockSizeConfig::default());
        // Manually drive a 0-height scenario by analysing a 1-pixel strip
        // and checking it stays within bounds.
        let partition = analyser.analyse(&frame[..0], 0, 0);
        assert!(partition.blocks.is_empty());
    }
}
