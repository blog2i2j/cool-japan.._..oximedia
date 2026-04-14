//! Advanced blockiness detector with DCT block boundary analysis.
//!
//! Detects block compression artifacts by analysing energy differences at
//! DCT block boundaries vs. interior pixels, computing a normalised blockiness
//! score, and providing actionable deblocking recommendations.
//!
//! # Algorithm overview
//!
//! 1. Compute an edge-strength map using an absolute first-order difference.
//! 2. Separate boundary pixels (multiples of `block_size`) from interior pixels.
//! 3. `blockiness_ratio` = mean(boundary_edges) / max(mean(interior_edges), ε).
//! 4. A value > 1 indicates that block boundaries are more energetic than
//!    the interior — the hallmark of blocking artifacts.
//! 5. A `DeblockingRecommendation` is derived from the ratio.
//!
//! The module also exposes per-region (tile) analysis so callers can locate
//! which areas of a frame suffer most.

use crate::{Frame, MetricType, QualityScore};
use oximedia_core::{OxiError, OxiResult};
use serde::{Deserialize, Serialize};

// ── Constants ──────────────────────────────────────────────────────────────────

/// Default DCT block size (pixels).
const DEFAULT_BLOCK_SIZE: usize = 8;

/// Epsilon used to avoid division by zero.
const EPS: f64 = 1e-6;

// ── Types ─────────────────────────────────────────────────────────────────────

/// Deblocking strength recommendation derived from the blockiness ratio.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeblockingRecommendation {
    /// No deblocking required (ratio ≤ 1.05).
    None,
    /// Light deblocking filter (ratio 1.05–1.20).
    Light,
    /// Moderate deblocking filter (ratio 1.20–1.50).
    Moderate,
    /// Strong deblocking filter (ratio > 1.50).
    Strong,
}

impl DeblockingRecommendation {
    /// Derives a recommendation from a blockiness ratio.
    #[must_use]
    pub fn from_ratio(ratio: f64) -> Self {
        if ratio > 1.50 {
            Self::Strong
        } else if ratio > 1.20 {
            Self::Moderate
        } else if ratio > 1.05 {
            Self::Light
        } else {
            Self::None
        }
    }

    /// Returns a human-readable description.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::None => "No deblocking needed",
            Self::Light => "Apply a light deblocking filter (e.g. H.264 deblock strength 1)",
            Self::Moderate => "Apply moderate deblocking (e.g. H.264 deblock strength 2–3)",
            Self::Strong => "Apply strong deblocking (e.g. H.264 deblock strength 4–6)",
        }
    }
}

/// Per-tile (region) blockiness result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TileBlockiness {
    /// Top-left corner x of the tile (pixels).
    pub x: usize,
    /// Top-left corner y of the tile (pixels).
    pub y: usize,
    /// Tile width (pixels).
    pub width: usize,
    /// Tile height (pixels).
    pub height: usize,
    /// Normalised blockiness score for this tile.
    pub score: f64,
    /// Derived deblocking recommendation for this tile.
    pub recommendation: DeblockingRecommendation,
}

/// Full result from the advanced blockiness detector.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlockinessResult {
    /// Overall normalised blockiness score (0.0 = no blockiness; higher = worse).
    pub score: f64,
    /// Mean boundary edge energy.
    pub boundary_energy: f64,
    /// Mean interior edge energy.
    pub interior_energy: f64,
    /// Raw blockiness ratio (boundary / interior).
    pub blockiness_ratio: f64,
    /// Deblocking recommendation for the whole frame.
    pub recommendation: DeblockingRecommendation,
    /// Per-tile breakdown (populated when `tile_size` > 0).
    pub tiles: Vec<TileBlockiness>,
}

// ── Detector ──────────────────────────────────────────────────────────────────

/// Advanced blockiness detector.
///
/// Uses DCT block-boundary energy analysis to detect and quantify block-based
/// compression artifacts.
pub struct AdvancedBlockinessDetector {
    /// DCT block size in pixels (usually 8 for H.264/VP8, 4 or 8 for AV1 CFL).
    block_size: usize,
    /// Tile size for per-region analysis (0 = no tiling).
    tile_size: usize,
}

impl AdvancedBlockinessDetector {
    /// Creates a detector with the default 8-pixel DCT block size and no tiling.
    #[must_use]
    pub fn new() -> Self {
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
            tile_size: 0,
        }
    }

    /// Creates a detector with a custom block size.
    #[must_use]
    pub fn with_block_size(block_size: usize) -> Self {
        Self {
            block_size,
            tile_size: 0,
        }
    }

    /// Enables per-region (tile) analysis with the given tile size.
    #[must_use]
    pub fn with_tile_size(mut self, tile_size: usize) -> Self {
        self.tile_size = tile_size;
        self
    }

    /// Analyses blockiness in the luma plane of a frame.
    ///
    /// # Errors
    ///
    /// Returns `OxiError::InvalidData` if the frame is too small.
    pub fn analyze(&self, frame: &Frame) -> OxiResult<BlockinessResult> {
        let width = frame.width;
        let height = frame.height;
        let min_dim = self.block_size * 2;

        if width < min_dim || height < min_dim {
            return Err(OxiError::InvalidData(format!(
                "Frame ({width}×{height}) is too small for block size {}; need at least {min_dim}×{min_dim}",
                self.block_size
            )));
        }

        let plane = &frame.planes[0];

        let (boundary_energy, interior_energy) =
            self.boundary_vs_interior(plane, width, height, 0, 0, width, height);

        let blockiness_ratio = if interior_energy > EPS {
            boundary_energy / interior_energy
        } else if boundary_energy > EPS {
            2.0 // high boundary energy, no interior energy → definitely blocky
        } else {
            1.0 // flat frame — no artifacts
        };

        // Normalise to [0, ∞) where 0 = no blockiness.
        let score = (blockiness_ratio - 1.0).max(0.0) * 100.0;

        let recommendation = DeblockingRecommendation::from_ratio(blockiness_ratio);

        let tiles = if self.tile_size > 0 {
            self.analyze_tiles(plane, width, height)?
        } else {
            Vec::new()
        };

        Ok(BlockinessResult {
            score,
            boundary_energy,
            interior_energy,
            blockiness_ratio,
            recommendation,
            tiles,
        })
    }

    /// Returns a `QualityScore` compatible with the rest of the quality API.
    ///
    /// # Errors
    ///
    /// Returns `OxiError::InvalidData` if the frame is too small.
    pub fn detect(&self, frame: &Frame) -> OxiResult<QualityScore> {
        let result = self.analyze(frame)?;
        let mut score = QualityScore::new(MetricType::Blockiness, result.score);
        score.add_component("boundary_energy", result.boundary_energy);
        score.add_component("interior_energy", result.interior_energy);
        score.add_component("blockiness_ratio", result.blockiness_ratio);
        Ok(score)
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    /// Computes mean boundary and interior first-order edge energies in a region.
    ///
    /// Horizontal and vertical differences at block boundaries are accumulated
    /// separately from those in block interiors.
    fn boundary_vs_interior(
        &self,
        plane: &[u8],
        stride: usize,
        _height: usize,
        region_x: usize,
        region_y: usize,
        region_w: usize,
        region_h: usize,
    ) -> (f64, f64) {
        let mut boundary_sum = 0.0_f64;
        let mut boundary_count = 0_u64;
        let mut interior_sum = 0.0_f64;
        let mut interior_count = 0_u64;

        // Horizontal differences (x direction)
        for y in region_y..region_y + region_h {
            for x in region_x + 1..region_x + region_w {
                let diff = (i32::from(plane[y * stride + x]) - i32::from(plane[y * stride + x - 1]))
                    .unsigned_abs() as f64;

                if x % self.block_size == 0 {
                    boundary_sum += diff;
                    boundary_count += 1;
                } else {
                    interior_sum += diff;
                    interior_count += 1;
                }
            }
        }

        // Vertical differences (y direction)
        for y in region_y + 1..region_y + region_h {
            for x in region_x..region_x + region_w {
                let diff = (i32::from(plane[y * stride + x])
                    - i32::from(plane[(y - 1) * stride + x]))
                .unsigned_abs() as f64;

                if y % self.block_size == 0 {
                    boundary_sum += diff;
                    boundary_count += 1;
                } else {
                    interior_sum += diff;
                    interior_count += 1;
                }
            }
        }

        let boundary_energy = if boundary_count > 0 {
            boundary_sum / boundary_count as f64
        } else {
            0.0
        };

        let interior_energy = if interior_count > 0 {
            interior_sum / interior_count as f64
        } else {
            0.0
        };

        (boundary_energy, interior_energy)
    }

    /// Splits the luma plane into tiles and computes per-tile blockiness.
    fn analyze_tiles(
        &self,
        plane: &[u8],
        width: usize,
        height: usize,
    ) -> OxiResult<Vec<TileBlockiness>> {
        if self.tile_size < self.block_size * 2 {
            return Err(OxiError::InvalidData(format!(
                "tile_size ({}) must be at least 2× block_size ({})",
                self.tile_size,
                self.block_size * 2
            )));
        }

        let mut tiles = Vec::new();

        let mut ty = 0;
        while ty < height {
            let tile_h = self.tile_size.min(height - ty);
            let mut tx = 0;
            while tx < width {
                let tile_w = self.tile_size.min(width - tx);

                // Skip tiles that are too small.
                if tile_w < self.block_size * 2 || tile_h < self.block_size * 2 {
                    tx += tile_w;
                    continue;
                }

                let (boundary_energy, interior_energy) =
                    self.boundary_vs_interior(plane, width, height, tx, ty, tile_w, tile_h);

                let ratio = if interior_energy > EPS {
                    boundary_energy / interior_energy
                } else if boundary_energy > EPS {
                    2.0
                } else {
                    1.0
                };

                let score = (ratio - 1.0).max(0.0) * 100.0;

                tiles.push(TileBlockiness {
                    x: tx,
                    y: ty,
                    width: tile_w,
                    height: tile_h,
                    score,
                    recommendation: DeblockingRecommendation::from_ratio(ratio),
                });

                tx += tile_w;
            }
            ty += tile_h;
        }

        Ok(tiles)
    }
}

impl Default for AdvancedBlockinessDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    fn flat_frame(width: usize, height: usize, value: u8) -> Frame {
        let mut frame =
            Frame::new(width, height, PixelFormat::Yuv420p).expect("frame creation should succeed");
        frame.planes[0].fill(value);
        frame
    }

    fn checkerboard_frame(width: usize, height: usize, block_size: usize) -> Frame {
        let mut frame =
            Frame::new(width, height, PixelFormat::Yuv420p).expect("frame creation should succeed");
        for y in 0..height {
            for x in 0..width {
                let v: u8 = if (x / block_size + y / block_size) % 2 == 0 {
                    64
                } else {
                    192
                };
                frame.planes[0][y * width + x] = v;
            }
        }
        frame
    }

    /// Returns a frame where every DCT block boundary column/row has a hard
    /// discontinuity, but the interior is smooth — the worst-case blocking pattern.
    fn artificial_blocking_frame(width: usize, height: usize, block_size: usize) -> Frame {
        let mut frame =
            Frame::new(width, height, PixelFormat::Yuv420p).expect("frame creation should succeed");
        for y in 0..height {
            for x in 0..width {
                // Interior pixels get a smooth ramp; boundary columns/rows get 0.
                let v: u8 = if x % block_size == 0 || y % block_size == 0 {
                    0
                } else {
                    128
                };
                frame.planes[0][y * width + x] = v;
            }
        }
        frame
    }

    #[test]
    fn test_flat_frame_no_blockiness() {
        let detector = AdvancedBlockinessDetector::new();
        let frame = flat_frame(64, 64, 128);
        let result = detector.analyze(&frame).expect("analyze should succeed");
        // Flat frame: both energies are 0 → ratio = 1.0 → score = 0.0
        assert_eq!(result.score, 0.0);
        assert_eq!(result.recommendation, DeblockingRecommendation::None);
    }

    #[test]
    fn test_blocking_frame_higher_score() {
        let detector = AdvancedBlockinessDetector::new();
        let smooth = checkerboard_frame(64, 64, 1); // every pixel alternates — uniform interior
        let blocky = artificial_blocking_frame(64, 64, 8);

        let smooth_result = detector.analyze(&smooth).expect("analyze should succeed");
        let blocky_result = detector.analyze(&blocky).expect("analyze should succeed");

        // The artificially blocked frame should score higher.
        assert!(
            blocky_result.score >= smooth_result.score,
            "blocky={} smooth={}",
            blocky_result.score,
            smooth_result.score
        );
    }

    #[test]
    fn test_deblocking_recommendation_thresholds() {
        assert_eq!(
            DeblockingRecommendation::from_ratio(1.0),
            DeblockingRecommendation::None
        );
        assert_eq!(
            DeblockingRecommendation::from_ratio(1.1),
            DeblockingRecommendation::Light
        );
        assert_eq!(
            DeblockingRecommendation::from_ratio(1.35),
            DeblockingRecommendation::Moderate
        );
        assert_eq!(
            DeblockingRecommendation::from_ratio(2.0),
            DeblockingRecommendation::Strong
        );
    }

    #[test]
    fn test_detect_returns_quality_score() {
        let detector = AdvancedBlockinessDetector::new();
        let frame = checkerboard_frame(64, 64, 8);
        let score = detector.detect(&frame).expect("detect should succeed");
        assert!(score.score >= 0.0);
        assert!(score.components.contains_key("blockiness_ratio"));
        assert!(score.components.contains_key("boundary_energy"));
        assert!(score.components.contains_key("interior_energy"));
    }

    #[test]
    fn test_too_small_frame_returns_error() {
        let detector = AdvancedBlockinessDetector::new();
        let small_frame = flat_frame(8, 8, 64); // exactly block_size; need 2× → error
        let result = detector.analyze(&small_frame);
        assert!(
            result.is_err(),
            "Expected error for too-small frame, got Ok"
        );
    }

    #[test]
    fn test_custom_block_size() {
        let detector = AdvancedBlockinessDetector::with_block_size(16);
        let frame = artificial_blocking_frame(128, 128, 16);
        let result = detector.analyze(&frame).expect("analyze should succeed");
        assert!(result.score >= 0.0);
    }

    #[test]
    fn test_tile_analysis_populates_tiles() {
        let detector = AdvancedBlockinessDetector::new().with_tile_size(32);
        let frame = checkerboard_frame(128, 128, 8);
        let result = detector.analyze(&frame).expect("analyze should succeed");
        // 128/32 = 4 columns × 4 rows = 16 tiles
        assert_eq!(result.tiles.len(), 16);
        for tile in &result.tiles {
            assert!(tile.score >= 0.0);
        }
    }

    #[test]
    fn test_tile_size_too_small_returns_error() {
        // tile_size < block_size * 2 should return an error
        let detector = AdvancedBlockinessDetector::new().with_tile_size(4); // 4 < 8*2
        let frame = checkerboard_frame(128, 128, 8);
        let result = detector.analyze(&frame);
        // Will call analyze_tiles which should fail
        assert!(
            result.is_err(),
            "Expected error for tile_size < 2*block_size"
        );
    }

    #[test]
    fn test_recommendation_descriptions_are_non_empty() {
        for rec in &[
            DeblockingRecommendation::None,
            DeblockingRecommendation::Light,
            DeblockingRecommendation::Moderate,
            DeblockingRecommendation::Strong,
        ] {
            assert!(!rec.description().is_empty());
        }
    }
}
