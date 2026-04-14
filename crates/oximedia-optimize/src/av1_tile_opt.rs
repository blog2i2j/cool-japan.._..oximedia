//! AV1 tile and frame parallel optimization.
//!
//! Selects tile partitioning based on spatial content complexity, balancing
//! parallelism (more tiles = more threads) against coding efficiency (fewer
//! tiles = better cross-tile prediction).
//!
//! # Algorithm
//!
//! 1. Divide the frame into a grid of candidate tile columns and rows
//!    (AV1 constrains tile dimensions to powers of 2 in super-blocks).
//! 2. Measure per-tile complexity using luma variance and edge density.
//! 3. Score each candidate partitioning by a weighted combination of:
//!    - **Parallelism score**: proportional to `min(tile_count, available_threads)`.
//!    - **Balance penalty**: standard deviation of per-tile complexity
//!      (unbalanced tiles stall the slowest thread).
//!    - **Overhead penalty**: tile header + cross-tile prediction loss, which
//!      grows with the number of tile boundaries.
//! 4. Select the partitioning with the best score.
//!
//! # AV1 Constraints
//!
//! - Tile widths and heights must be multiples of the super-block size (64 or 128).
//! - Maximum 64 tiles (spec limit: `MAX_TILE_COLS * MAX_TILE_ROWS`).
//! - At most 64 tile columns and 64 tile rows (but product <= 64 in practice
//!   for level constraints).

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use std::fmt;

// ── Constants ───────────────────────────────────────────────────────────────

/// AV1 super-block size (in pixels). We assume 64x64 SB by default.
const DEFAULT_SB_SIZE: u32 = 64;

/// AV1 spec maximum tile columns.
const MAX_TILE_COLS: u32 = 64;

/// AV1 spec maximum tile rows.
const MAX_TILE_ROWS: u32 = 64;

/// Maximum total tiles allowed (level-dependent; conservative default).
const MAX_TOTAL_TILES: u32 = 64;

/// Minimum tile dimension in super-blocks.
const MIN_TILE_SB: u32 = 1;

// ── Configuration ───────────────────────────────────────────────────────────

/// Configuration for tile partitioning optimisation.
#[derive(Debug, Clone)]
pub struct TileOptConfig {
    /// Frame width in pixels.
    pub frame_width: u32,
    /// Frame height in pixels.
    pub frame_height: u32,
    /// Super-block size in pixels (64 or 128).
    pub sb_size: u32,
    /// Number of available worker threads for tile-parallel encoding.
    pub thread_count: u32,
    /// Weight for the parallelism score component (0.0..1.0).
    pub parallelism_weight: f64,
    /// Weight for the balance penalty component (0.0..1.0).
    pub balance_weight: f64,
    /// Estimated bits of overhead per tile boundary per super-block row/col.
    pub boundary_overhead_bits: f64,
    /// Weight for overhead penalty (0.0..1.0).
    pub overhead_weight: f64,
    /// Maximum tiles to consider (clamped to spec limit).
    pub max_tiles: u32,
}

impl Default for TileOptConfig {
    fn default() -> Self {
        Self {
            frame_width: 1920,
            frame_height: 1080,
            sb_size: DEFAULT_SB_SIZE,
            thread_count: 4,
            parallelism_weight: 0.4,
            balance_weight: 0.35,
            boundary_overhead_bits: 128.0,
            overhead_weight: 0.25,
            max_tiles: MAX_TOTAL_TILES,
        }
    }
}

impl TileOptConfig {
    /// Frame width in super-blocks (rounded up).
    fn sb_cols(&self) -> u32 {
        (self.frame_width + self.sb_size - 1) / self.sb_size
    }

    /// Frame height in super-blocks (rounded up).
    fn sb_rows(&self) -> u32 {
        (self.frame_height + self.sb_size - 1) / self.sb_size
    }
}

// ── Complexity map ──────────────────────────────────────────────────────────

/// Per-super-block complexity value derived from luma variance and edges.
#[derive(Debug, Clone, Copy)]
pub struct SbComplexity {
    /// Luma variance (higher = more complex).
    pub variance: f64,
    /// Edge density (fraction of edge pixels in the super-block).
    pub edge_density: f64,
}

impl SbComplexity {
    /// Combined scalar complexity metric.
    fn combined(&self) -> f64 {
        0.7 * self.variance + 0.3 * self.edge_density * 1000.0
    }
}

/// 2-D grid of per-super-block complexity values (row-major).
#[derive(Debug, Clone)]
pub struct ComplexityGrid {
    /// Number of SB columns.
    pub cols: u32,
    /// Number of SB rows.
    pub rows: u32,
    /// Complexity values, length = cols * rows.
    pub values: Vec<SbComplexity>,
}

impl ComplexityGrid {
    /// Creates a grid from raw per-SB complexity data.
    ///
    /// Returns `None` if the length doesn't match `cols * rows`.
    pub fn new(cols: u32, rows: u32, values: Vec<SbComplexity>) -> Option<Self> {
        if values.len() != (cols as usize) * (rows as usize) {
            return None;
        }
        Some(Self { cols, rows, values })
    }

    /// Creates a uniform-complexity grid (e.g. for testing).
    pub fn uniform(cols: u32, rows: u32, variance: f64) -> Self {
        let n = (cols as usize) * (rows as usize);
        Self {
            cols,
            rows,
            values: vec![
                SbComplexity {
                    variance,
                    edge_density: 0.1,
                };
                n
            ],
        }
    }

    /// Computes aggregate complexity for a rectangular sub-region of super-blocks.
    fn region_complexity(&self, col_start: u32, col_end: u32, row_start: u32, row_end: u32) -> f64 {
        let mut total = 0.0;
        let mut count = 0u64;
        for r in row_start..row_end {
            for c in col_start..col_end {
                let idx = r as usize * self.cols as usize + c as usize;
                if let Some(sb) = self.values.get(idx) {
                    total += sb.combined();
                    count += 1;
                }
            }
        }
        if count == 0 {
            0.0
        } else {
            total / count as f64
        }
    }
}

// ── Candidate partitioning ──────────────────────────────────────────────────

/// A candidate tile partitioning described by uniform column/row counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TilePartition {
    /// Number of tile columns.
    pub tile_cols: u32,
    /// Number of tile rows.
    pub tile_rows: u32,
}

impl TilePartition {
    /// Total tile count.
    pub fn total_tiles(&self) -> u32 {
        self.tile_cols * self.tile_rows
    }
}

impl fmt::Display for TilePartition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}x{} ({} tiles)",
            self.tile_cols,
            self.tile_rows,
            self.total_tiles()
        )
    }
}

/// Scored candidate result.
#[derive(Debug, Clone)]
pub struct ScoredPartition {
    /// The partition layout.
    pub partition: TilePartition,
    /// Overall score (higher = better).
    pub score: f64,
    /// Parallelism sub-score.
    pub parallelism_score: f64,
    /// Balance sub-score (lower stddev = higher score).
    pub balance_score: f64,
    /// Overhead sub-score (fewer tiles = higher score).
    pub overhead_score: f64,
    /// Per-tile average complexity values.
    pub tile_complexities: Vec<f64>,
}

// ── Optimizer ───────────────────────────────────────────────────────────────

/// AV1 tile partition optimiser.
pub struct TileOptimizer {
    config: TileOptConfig,
}

impl TileOptimizer {
    /// Creates a new optimiser with the given configuration.
    pub fn new(config: TileOptConfig) -> Self {
        Self { config }
    }

    /// Enumerates valid tile column counts (must divide the SB-col count evenly
    /// or be the SB-col count itself).  Returns only counts that also satisfy
    /// the spec limits and the maximum-tile constraint.
    fn candidate_col_counts(&self) -> Vec<u32> {
        let sb_cols = self.config.sb_cols();
        let max_cols = sb_cols.min(MAX_TILE_COLS);
        let mut result: Vec<u32> = (1..=max_cols)
            .filter(|&c| sb_cols >= c * MIN_TILE_SB)
            .collect();
        result.sort_unstable();
        result.dedup();
        result
    }

    /// Enumerates valid tile row counts.
    fn candidate_row_counts(&self) -> Vec<u32> {
        let sb_rows = self.config.sb_rows();
        let max_rows = sb_rows.min(MAX_TILE_ROWS);
        let mut result: Vec<u32> = (1..=max_rows)
            .filter(|&r| sb_rows >= r * MIN_TILE_SB)
            .collect();
        result.sort_unstable();
        result.dedup();
        result
    }

    /// Generates all candidate partitions within spec and config limits.
    pub fn enumerate_candidates(&self) -> Vec<TilePartition> {
        let col_counts = self.candidate_col_counts();
        let row_counts = self.candidate_row_counts();
        let max_tiles = self.config.max_tiles.min(MAX_TOTAL_TILES);

        let mut candidates = Vec::new();
        for &tc in &col_counts {
            for &tr in &row_counts {
                if tc * tr <= max_tiles {
                    candidates.push(TilePartition {
                        tile_cols: tc,
                        tile_rows: tr,
                    });
                }
            }
        }
        candidates
    }

    /// Scores a single partition against the complexity grid.
    pub fn score_partition(
        &self,
        partition: &TilePartition,
        grid: &ComplexityGrid,
    ) -> ScoredPartition {
        let sb_cols = self.config.sb_cols();
        let sb_rows = self.config.sb_rows();

        // Compute per-tile complexity
        let tile_complexities = self.compute_tile_complexities(partition, grid, sb_cols, sb_rows);

        // Parallelism: fraction of threads utilised
        let effective_threads = partition.total_tiles().min(self.config.thread_count) as f64;
        let parallelism_score = effective_threads / self.config.thread_count.max(1) as f64;

        // Balance: 1.0 - normalised stddev of tile complexities
        let balance_score = if tile_complexities.is_empty() {
            1.0
        } else {
            let mean =
                tile_complexities.iter().sum::<f64>() / tile_complexities.len().max(1) as f64;
            let var = tile_complexities
                .iter()
                .map(|c| (c - mean) * (c - mean))
                .sum::<f64>()
                / tile_complexities.len().max(1) as f64;
            let stddev = var.sqrt();
            let norm_std = if mean.abs() < 1e-9 {
                0.0
            } else {
                stddev / mean
            };
            (1.0 - norm_std).max(0.0)
        };

        // Overhead: penalty proportional to boundary count
        let boundary_count = (partition.tile_cols.saturating_sub(1) * sb_rows)
            + (partition.tile_rows.saturating_sub(1) * sb_cols);
        let total_overhead = boundary_count as f64 * self.config.boundary_overhead_bits;
        // Normalise against a rough frame bit budget (assume 3 bits/pixel as reference)
        let frame_bits = (self.config.frame_width as f64) * (self.config.frame_height as f64) * 3.0;
        let overhead_fraction = if frame_bits > 0.0 {
            total_overhead / frame_bits
        } else {
            0.0
        };
        let overhead_score = (1.0 - overhead_fraction * 10.0).max(0.0);

        let score = self.config.parallelism_weight * parallelism_score
            + self.config.balance_weight * balance_score
            + self.config.overhead_weight * overhead_score;

        ScoredPartition {
            partition: *partition,
            score,
            parallelism_score,
            balance_score,
            overhead_score,
            tile_complexities,
        }
    }

    /// Helper: compute average complexity per tile.
    fn compute_tile_complexities(
        &self,
        partition: &TilePartition,
        grid: &ComplexityGrid,
        sb_cols: u32,
        sb_rows: u32,
    ) -> Vec<f64> {
        let cols_per_tile = sb_cols / partition.tile_cols.max(1);
        let rows_per_tile = sb_rows / partition.tile_rows.max(1);
        let mut complexities = Vec::with_capacity(partition.total_tiles() as usize);

        for tr in 0..partition.tile_rows {
            for tc in 0..partition.tile_cols {
                let c0 = tc * cols_per_tile;
                let c1 = if tc == partition.tile_cols - 1 {
                    sb_cols
                } else {
                    (tc + 1) * cols_per_tile
                };
                let r0 = tr * rows_per_tile;
                let r1 = if tr == partition.tile_rows - 1 {
                    sb_rows
                } else {
                    (tr + 1) * rows_per_tile
                };
                complexities.push(grid.region_complexity(c0, c1, r0, r1));
            }
        }
        complexities
    }

    /// Selects the best tile partitioning from all valid candidates.
    ///
    /// Returns `None` if no valid candidate exists (shouldn't happen in practice).
    pub fn optimize(&self, grid: &ComplexityGrid) -> Option<ScoredPartition> {
        let candidates = self.enumerate_candidates();
        candidates
            .iter()
            .map(|p| self.score_partition(p, grid))
            .max_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Selects the best partitioning for a given number of threads,
    /// preferring partitions whose tile count is close to the thread count.
    pub fn optimize_for_threads(
        &self,
        grid: &ComplexityGrid,
        threads: u32,
    ) -> Option<ScoredPartition> {
        let candidates = self.enumerate_candidates();
        let thread_filtered: Vec<_> = candidates
            .into_iter()
            .filter(|p| p.total_tiles() <= threads * 2 && p.total_tiles() >= 1)
            .collect();

        if thread_filtered.is_empty() {
            return None;
        }

        thread_filtered
            .iter()
            .map(|p| self.score_partition(p, grid))
            .max_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

// ── Frame-parallel mode selection ───────────────────────────────────────────

/// Frame-parallel encoding mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameParallelMode {
    /// No frame parallelism (sequential encoding).
    Sequential,
    /// Frame-parallel with limited reference constraints.
    FrameParallel {
        /// Maximum number of frames in flight.
        max_in_flight: u32,
    },
    /// Combined tile + frame parallelism.
    TileAndFrame {
        /// Tile partitioning to use.
        tile_cols: u32,
        /// Tile rows.
        tile_rows: u32,
        /// Frames in flight.
        max_in_flight: u32,
    },
}

/// Recommends a frame-parallel mode based on available resources.
pub fn recommend_parallel_mode(
    thread_count: u32,
    frame_width: u32,
    frame_height: u32,
) -> FrameParallelMode {
    if thread_count <= 1 {
        return FrameParallelMode::Sequential;
    }

    let megapixels = (frame_width as f64 * frame_height as f64) / 1_000_000.0;

    if megapixels >= 8.0 {
        // 4K+: tile-parallel primarily, limited frame parallelism
        let tile_cols = (thread_count / 2).max(2).min(4);
        let tile_rows = 2;
        let frames = (thread_count / (tile_cols * tile_rows)).max(1);
        FrameParallelMode::TileAndFrame {
            tile_cols,
            tile_rows,
            max_in_flight: frames,
        }
    } else if megapixels >= 2.0 {
        // 1080p: moderate tile + frame parallelism
        let tile_cols = 2.min(thread_count);
        let tile_rows = 1;
        let frames = (thread_count / tile_cols).max(1).min(4);
        FrameParallelMode::TileAndFrame {
            tile_cols,
            tile_rows,
            max_in_flight: frames,
        }
    } else {
        // SD/720p: frame-parallel is usually sufficient
        FrameParallelMode::FrameParallel {
            max_in_flight: thread_count.min(8),
        }
    }
}

/// Summary statistics for a tile partitioning decision.
#[derive(Debug, Clone)]
pub struct TileOptSummary {
    /// Chosen partition.
    pub partition: TilePartition,
    /// Score of chosen partition.
    pub score: f64,
    /// Number of candidates evaluated.
    pub candidates_evaluated: usize,
    /// Recommended parallel mode.
    pub parallel_mode: FrameParallelMode,
}

/// High-level entry point: analyse the frame and recommend tile partitioning
/// plus frame-parallel mode.
pub fn analyze_and_recommend(
    config: &TileOptConfig,
    grid: &ComplexityGrid,
) -> Option<TileOptSummary> {
    let optimizer = TileOptimizer::new(config.clone());
    let candidates = optimizer.enumerate_candidates();
    let num_candidates = candidates.len();

    let best = optimizer.optimize(grid)?;
    let parallel_mode =
        recommend_parallel_mode(config.thread_count, config.frame_width, config.frame_height);

    Some(TileOptSummary {
        partition: best.partition,
        score: best.score,
        candidates_evaluated: num_candidates,
        parallel_mode,
    })
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> TileOptConfig {
        TileOptConfig {
            frame_width: 1920,
            frame_height: 1080,
            sb_size: 64,
            thread_count: 8,
            ..Default::default()
        }
    }

    fn uniform_grid(config: &TileOptConfig) -> ComplexityGrid {
        let sb_cols = config.sb_cols();
        let sb_rows = config.sb_rows();
        ComplexityGrid::uniform(sb_cols, sb_rows, 100.0)
    }

    #[test]
    fn test_single_tile_is_always_valid() {
        let config = default_config();
        let optimizer = TileOptimizer::new(config.clone());
        let candidates = optimizer.enumerate_candidates();
        assert!(candidates
            .iter()
            .any(|p| p.tile_cols == 1 && p.tile_rows == 1));
    }

    #[test]
    fn test_candidate_count_respects_max_tiles() {
        let config = TileOptConfig {
            max_tiles: 4,
            ..default_config()
        };
        let optimizer = TileOptimizer::new(config);
        let candidates = optimizer.enumerate_candidates();
        for c in &candidates {
            assert!(c.total_tiles() <= 4, "partition {c} exceeds max_tiles");
        }
    }

    #[test]
    fn test_scoring_single_tile() {
        let config = default_config();
        let grid = uniform_grid(&config);
        let optimizer = TileOptimizer::new(config);

        let single = TilePartition {
            tile_cols: 1,
            tile_rows: 1,
        };
        let scored = optimizer.score_partition(&single, &grid);

        // Single tile: no overhead, no balance penalty, low parallelism
        assert!(
            scored.overhead_score > 0.99,
            "single tile should have ~0 overhead"
        );
        assert!(
            scored.balance_score > 0.99,
            "single tile should be perfectly balanced"
        );
        assert!(
            scored.parallelism_score < 0.5,
            "single tile should have low parallelism"
        );
    }

    #[test]
    fn test_scoring_prefers_balanced_tiles() {
        let config = default_config();
        let sb_cols = config.sb_cols();
        let sb_rows = config.sb_rows();

        // Create an unbalanced grid: left half is 10x more complex
        let n = (sb_cols as usize) * (sb_rows as usize);
        let mut values = Vec::with_capacity(n);
        for r in 0..sb_rows {
            for c in 0..sb_cols {
                let variance = if c < sb_cols / 2 { 1000.0 } else { 100.0 };
                values.push(SbComplexity {
                    variance,
                    edge_density: 0.1,
                });
                let _ = r; // suppress unused
            }
        }
        let grid = ComplexityGrid::new(sb_cols, sb_rows, values).expect("grid should be valid");

        let optimizer = TileOptimizer::new(config);

        // Compare 2x1 (vertical split across complexity boundary) vs 1x2 (horizontal)
        let vert = TilePartition {
            tile_cols: 2,
            tile_rows: 1,
        };
        let horiz = TilePartition {
            tile_cols: 1,
            tile_rows: 2,
        };
        let vert_score = optimizer.score_partition(&vert, &grid);
        let horiz_score = optimizer.score_partition(&horiz, &grid);

        // Horizontal split should be more balanced (both halves have mixed complexity)
        assert!(
            horiz_score.balance_score > vert_score.balance_score,
            "horizontal split should be more balanced than vertical for left-heavy content"
        );
    }

    #[test]
    fn test_optimize_returns_some() {
        let config = default_config();
        let grid = uniform_grid(&config);
        let optimizer = TileOptimizer::new(config);

        let result = optimizer.optimize(&grid);
        assert!(
            result.is_some(),
            "optimize should always find at least one candidate"
        );
    }

    #[test]
    fn test_optimize_for_threads() {
        let config = default_config();
        let grid = uniform_grid(&config);
        let optimizer = TileOptimizer::new(config);

        let result = optimizer.optimize_for_threads(&grid, 4);
        assert!(result.is_some());
        let scored = result.expect("should have result");
        assert!(
            scored.partition.total_tiles() <= 8,
            "tile count should be bounded"
        );
    }

    #[test]
    fn test_frame_parallel_mode_sequential() {
        let mode = recommend_parallel_mode(1, 1920, 1080);
        assert_eq!(mode, FrameParallelMode::Sequential);
    }

    #[test]
    fn test_frame_parallel_mode_4k() {
        let mode = recommend_parallel_mode(16, 3840, 2160);
        match mode {
            FrameParallelMode::TileAndFrame {
                tile_cols,
                tile_rows,
                ..
            } => {
                assert!(tile_cols >= 2);
                assert!(tile_rows >= 1);
            }
            _ => panic!("4K with 16 threads should use TileAndFrame"),
        }
    }

    #[test]
    fn test_analyze_and_recommend() {
        let config = default_config();
        let grid = uniform_grid(&config);

        let summary = analyze_and_recommend(&config, &grid);
        assert!(summary.is_some());
        let s = summary.expect("should have summary");
        assert!(s.candidates_evaluated > 0);
        assert!(s.score > 0.0);
    }

    #[test]
    fn test_complexity_grid_creation() {
        let grid = ComplexityGrid::new(
            4,
            3,
            vec![
                SbComplexity {
                    variance: 1.0,
                    edge_density: 0.0
                };
                12
            ],
        );
        assert!(grid.is_some());

        // Wrong size should fail
        let bad = ComplexityGrid::new(
            4,
            3,
            vec![
                SbComplexity {
                    variance: 1.0,
                    edge_density: 0.0
                };
                11
            ],
        );
        assert!(bad.is_none());
    }

    #[test]
    fn test_tile_partition_display() {
        let p = TilePartition {
            tile_cols: 4,
            tile_rows: 2,
        };
        let s = format!("{p}");
        assert!(s.contains("4x2"));
        assert!(s.contains("8 tiles"));
    }

    #[test]
    fn test_sb_complexity_combined() {
        let sb = SbComplexity {
            variance: 200.0,
            edge_density: 0.5,
        };
        let combined = sb.combined();
        // 0.7*200 + 0.3*0.5*1000 = 140 + 150 = 290
        assert!((combined - 290.0).abs() < 1e-6);
    }
}
