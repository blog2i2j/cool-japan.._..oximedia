//! Motion search algorithms for video encoding.
//!
//! This module provides various motion estimation search algorithms including:
//! - Full search (exhaustive)
//! - Diamond search (SDSP/LDSP)
//! - Hexagon search
//! - UMH search (Unsymmetrical-cross Multi-Hexagon)
//!
//! All algorithms implement the [`MotionSearch`] trait for consistent interface.

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::unused_self)]
#![allow(clippy::items_after_statements)]
#![allow(unused_assignments)]

use super::types::{BlockMatch, BlockSize, MotionVector, MvCost, MvPrecision, SearchRange};

/// Early termination threshold for search.
pub const EARLY_TERMINATION_SAD: u32 = 256;

/// Minimum SAD improvement ratio for early termination.
pub const EARLY_TERMINATION_RATIO: f32 = 0.9;

/// Configuration for motion search algorithms.
#[derive(Clone, Debug)]
pub struct SearchConfig {
    /// Search range in full pixels.
    pub range: SearchRange,
    /// Motion vector precision.
    pub precision: MvPrecision,
    /// Enable early termination.
    pub early_termination: bool,
    /// Early termination threshold.
    pub early_threshold: u32,
    /// MV cost calculator for RD optimization.
    pub mv_cost: MvCost,
    /// Maximum iterations for iterative algorithms.
    pub max_iterations: u32,
    /// Enable sub-pixel refinement.
    pub subpel_refine: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            range: SearchRange::default(),
            precision: MvPrecision::QuarterPel,
            early_termination: true,
            early_threshold: EARLY_TERMINATION_SAD,
            mv_cost: MvCost::default(),
            max_iterations: 16,
            subpel_refine: true,
        }
    }
}

impl SearchConfig {
    /// Creates a new search config with the given range.
    #[must_use]
    pub fn with_range(range: SearchRange) -> Self {
        Self {
            range,
            ..Default::default()
        }
    }

    /// Sets the search range.
    #[must_use]
    pub const fn range(mut self, range: SearchRange) -> Self {
        self.range = range;
        self
    }

    /// Sets motion vector precision.
    #[must_use]
    pub const fn precision(mut self, precision: MvPrecision) -> Self {
        self.precision = precision;
        self
    }

    /// Enables or disables early termination.
    #[must_use]
    pub const fn early_termination(mut self, enable: bool) -> Self {
        self.early_termination = enable;
        self
    }

    /// Sets the reference motion vector for cost calculation.
    #[must_use]
    pub fn ref_mv(mut self, mv: MotionVector) -> Self {
        self.mv_cost.set_ref_mv(mv);
        self
    }

    /// Sets the lambda for rate-distortion optimization.
    #[must_use]
    pub fn lambda(mut self, lambda: f32) -> Self {
        self.mv_cost.lambda = lambda;
        self
    }
}

/// Search context containing frame data and configuration.
pub struct SearchContext<'a> {
    /// Source block data.
    pub src: &'a [u8],
    /// Source stride.
    pub src_stride: usize,
    /// Reference frame data.
    pub ref_frame: &'a [u8],
    /// Reference stride.
    pub ref_stride: usize,
    /// Block size.
    pub block_size: BlockSize,
    /// Block position X in source.
    pub block_x: usize,
    /// Block position Y in source.
    pub block_y: usize,
    /// Reference frame width.
    pub ref_width: usize,
    /// Reference frame height.
    pub ref_height: usize,
}

impl<'a> SearchContext<'a> {
    /// Creates a new search context.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        src: &'a [u8],
        src_stride: usize,
        ref_frame: &'a [u8],
        ref_stride: usize,
        block_size: BlockSize,
        block_x: usize,
        block_y: usize,
        ref_width: usize,
        ref_height: usize,
    ) -> Self {
        Self {
            src,
            src_stride,
            ref_frame,
            ref_stride,
            block_size,
            block_x,
            block_y,
            ref_width,
            ref_height,
        }
    }

    /// Returns the source block offset.
    #[must_use]
    pub const fn src_offset(&self) -> usize {
        self.block_y * self.src_stride + self.block_x
    }

    /// Calculates SAD for a given motion vector.
    #[must_use]
    #[allow(clippy::cast_sign_loss)]
    pub fn calculate_sad(&self, mv: &MotionVector) -> Option<u32> {
        let ref_x = self.block_x as i32 + mv.full_pel_x();
        let ref_y = self.block_y as i32 + mv.full_pel_y();

        // Check bounds
        if ref_x < 0 || ref_y < 0 {
            return None;
        }

        let ref_x = ref_x as usize;
        let ref_y = ref_y as usize;
        let width = self.block_size.width();
        let height = self.block_size.height();

        if ref_x + width > self.ref_width || ref_y + height > self.ref_height {
            return None;
        }

        let src_offset = self.src_offset();
        let ref_offset = ref_y * self.ref_stride + ref_x;

        // Check slice bounds
        if src_offset + (height - 1) * self.src_stride + width > self.src.len() {
            return None;
        }
        if ref_offset + (height - 1) * self.ref_stride + width > self.ref_frame.len() {
            return None;
        }

        let mut sad = 0u32;
        for row in 0..height {
            let src_row_offset = src_offset + row * self.src_stride;
            let ref_row_offset = ref_offset + row * self.ref_stride;

            for col in 0..width {
                let src_val = self.src[src_row_offset + col];
                let ref_val = self.ref_frame[ref_row_offset + col];
                let diff = i32::from(src_val) - i32::from(ref_val);
                sad += diff.unsigned_abs();
            }
        }

        Some(sad)
    }

    /// Checks if a motion vector is within valid bounds.
    #[must_use]
    pub fn is_valid_mv(&self, mv: &MotionVector, range: &SearchRange) -> bool {
        let ref_x = self.block_x as i32 + mv.full_pel_x();
        let ref_y = self.block_y as i32 + mv.full_pel_y();
        let width = self.block_size.width() as i32;
        let height = self.block_size.height() as i32;

        ref_x >= 0
            && ref_y >= 0
            && ref_x + width <= self.ref_width as i32
            && ref_y + height <= self.ref_height as i32
            && range.contains(mv.full_pel_x(), mv.full_pel_y())
    }
}

/// Trait for motion search algorithms.
pub trait MotionSearch {
    /// Performs motion search and returns the best match.
    fn search(&self, ctx: &SearchContext, config: &SearchConfig) -> BlockMatch;

    /// Performs motion search with a starting point prediction.
    fn search_with_predictor(
        &self,
        ctx: &SearchContext,
        config: &SearchConfig,
        predictor: MotionVector,
    ) -> BlockMatch;
}

/// Full exhaustive search algorithm.
///
/// Checks every position in the search range. Guaranteed to find the
/// global optimum but computationally expensive.
#[derive(Clone, Debug, Default)]
pub struct FullSearch;

impl FullSearch {
    /// Creates a new full search instance.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl MotionSearch for FullSearch {
    fn search(&self, ctx: &SearchContext, config: &SearchConfig) -> BlockMatch {
        self.search_with_predictor(ctx, config, MotionVector::zero())
    }

    fn search_with_predictor(
        &self,
        ctx: &SearchContext,
        config: &SearchConfig,
        _predictor: MotionVector,
    ) -> BlockMatch {
        let mut best = BlockMatch::worst();
        let range = &config.range;

        for dy in -range.vertical..=range.vertical {
            for dx in -range.horizontal..=range.horizontal {
                let mv = MotionVector::from_full_pel(dx, dy);

                if let Some(sad) = ctx.calculate_sad(&mv) {
                    let cost = config.mv_cost.rd_cost(&mv, sad);
                    let candidate = BlockMatch::new(mv, sad, cost);

                    best.update_if_better(&candidate);

                    // Early termination
                    if config.early_termination && sad < config.early_threshold {
                        return best;
                    }
                }
            }
        }

        best
    }
}

/// Diamond search algorithm.
///
/// Uses small diamond pattern (SDSP) for refinement and large diamond
/// pattern (LDSP) for initial coarse search.
#[derive(Clone, Debug, Default)]
pub struct DiamondSearch {
    /// Use large diamond for initial search.
    use_large_diamond: bool,
}

impl DiamondSearch {
    /// Small diamond pattern offsets (4 points).
    const SDSP: [(i32, i32); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];

    /// Large diamond pattern offsets (8 points).
    const LDSP: [(i32, i32); 8] = [
        (0, -2),
        (0, 2),
        (-2, 0),
        (2, 0),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ];

    /// Creates a new diamond search instance.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            use_large_diamond: true,
        }
    }

    /// Sets whether to use large diamond pattern.
    #[must_use]
    pub const fn use_large_diamond(mut self, enable: bool) -> Self {
        self.use_large_diamond = enable;
        self
    }

    /// Performs diamond pattern search around a center point.
    fn diamond_step(
        &self,
        ctx: &SearchContext,
        config: &SearchConfig,
        center: MotionVector,
        pattern: &[(i32, i32)],
    ) -> (MotionVector, u32) {
        let mut best_mv = center;
        let mut best_sad = ctx.calculate_sad(&center).unwrap_or(u32::MAX);

        for &(dx, dy) in pattern {
            let mv =
                MotionVector::from_full_pel(center.full_pel_x() + dx, center.full_pel_y() + dy);

            if !ctx.is_valid_mv(&mv, &config.range) {
                continue;
            }

            if let Some(sad) = ctx.calculate_sad(&mv) {
                if sad < best_sad {
                    best_sad = sad;
                    best_mv = mv;
                }
            }
        }

        (best_mv, best_sad)
    }
}

impl MotionSearch for DiamondSearch {
    fn search(&self, ctx: &SearchContext, config: &SearchConfig) -> BlockMatch {
        self.search_with_predictor(ctx, config, MotionVector::zero())
    }

    fn search_with_predictor(
        &self,
        ctx: &SearchContext,
        config: &SearchConfig,
        predictor: MotionVector,
    ) -> BlockMatch {
        let mut center = predictor.to_precision(MvPrecision::FullPel);
        let mut best_sad = ctx.calculate_sad(&center).unwrap_or(u32::MAX);

        // Initial large diamond search
        if self.use_large_diamond {
            for _ in 0..config.max_iterations {
                let (new_center, new_sad) = self.diamond_step(ctx, config, center, &Self::LDSP);

                if new_center == center {
                    break;
                }

                center = new_center;
                best_sad = new_sad;

                if config.early_termination && best_sad < config.early_threshold {
                    let cost = config.mv_cost.rd_cost(&center, best_sad);
                    return BlockMatch::new(center, best_sad, cost);
                }
            }
        }

        // Refinement with small diamond
        loop {
            let (new_center, new_sad) = self.diamond_step(ctx, config, center, &Self::SDSP);

            if new_center == center {
                break;
            }

            center = new_center;
            best_sad = new_sad;
        }

        let cost = config.mv_cost.rd_cost(&center, best_sad);
        BlockMatch::new(center, best_sad, cost)
    }
}

/// Hexagon search algorithm.
///
/// Uses a hexagonal pattern which provides better coverage than diamond
/// with fewer points to check.
#[derive(Clone, Debug, Default)]
pub struct HexagonSearch;

impl HexagonSearch {
    /// Hexagon pattern offsets (6 points).
    const HEXAGON: [(i32, i32); 6] = [(-2, 0), (-1, -2), (1, -2), (2, 0), (1, 2), (-1, 2)];

    /// Square pattern for final refinement (4 points).
    const SQUARE: [(i32, i32); 4] = [(-1, -1), (-1, 1), (1, -1), (1, 1)];

    /// Creates a new hexagon search instance.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Performs hexagon pattern search around a center point.
    fn hexagon_step(
        &self,
        ctx: &SearchContext,
        config: &SearchConfig,
        center: MotionVector,
    ) -> (MotionVector, u32) {
        let mut best_mv = center;
        let mut best_sad = ctx.calculate_sad(&center).unwrap_or(u32::MAX);

        for &(dx, dy) in &Self::HEXAGON {
            let mv =
                MotionVector::from_full_pel(center.full_pel_x() + dx, center.full_pel_y() + dy);

            if !ctx.is_valid_mv(&mv, &config.range) {
                continue;
            }

            if let Some(sad) = ctx.calculate_sad(&mv) {
                if sad < best_sad {
                    best_sad = sad;
                    best_mv = mv;
                }
            }
        }

        (best_mv, best_sad)
    }

    /// Final square refinement.
    fn square_refine(
        &self,
        ctx: &SearchContext,
        config: &SearchConfig,
        center: MotionVector,
    ) -> (MotionVector, u32) {
        let mut best_mv = center;
        let mut best_sad = ctx.calculate_sad(&center).unwrap_or(u32::MAX);

        for &(dx, dy) in &Self::SQUARE {
            let mv =
                MotionVector::from_full_pel(center.full_pel_x() + dx, center.full_pel_y() + dy);

            if !ctx.is_valid_mv(&mv, &config.range) {
                continue;
            }

            if let Some(sad) = ctx.calculate_sad(&mv) {
                if sad < best_sad {
                    best_sad = sad;
                    best_mv = mv;
                }
            }
        }

        (best_mv, best_sad)
    }
}

impl MotionSearch for HexagonSearch {
    fn search(&self, ctx: &SearchContext, config: &SearchConfig) -> BlockMatch {
        self.search_with_predictor(ctx, config, MotionVector::zero())
    }

    fn search_with_predictor(
        &self,
        ctx: &SearchContext,
        config: &SearchConfig,
        predictor: MotionVector,
    ) -> BlockMatch {
        let mut center = predictor.to_precision(MvPrecision::FullPel);
        let mut best_sad = ctx.calculate_sad(&center).unwrap_or(u32::MAX);

        // Hexagon search iterations
        for _ in 0..config.max_iterations {
            let (new_center, new_sad) = self.hexagon_step(ctx, config, center);

            if new_center == center {
                break;
            }

            center = new_center;
            best_sad = new_sad;

            if config.early_termination && best_sad < config.early_threshold {
                let cost = config.mv_cost.rd_cost(&center, best_sad);
                return BlockMatch::new(center, best_sad, cost);
            }
        }

        // Final square refinement
        let (final_center, final_sad) = self.square_refine(ctx, config, center);

        let cost = config.mv_cost.rd_cost(&final_center, final_sad);
        BlockMatch::new(final_center, final_sad, cost)
    }
}

/// Unsymmetrical-cross Multi-Hexagon search algorithm.
///
/// Combines multiple patterns for efficient search:
/// 1. Unsymmetrical cross
/// 2. Multi-hexagon grid
/// 3. Extended hexagon
/// 4. Small diamond refinement
#[derive(Clone, Debug, Default)]
pub struct UmhSearch {
    /// Cross search range multiplier.
    cross_range: i32,
}

impl UmhSearch {
    /// Creates a new UMH search instance.
    #[must_use]
    pub const fn new() -> Self {
        Self { cross_range: 2 }
    }

    /// Sets the cross search range multiplier.
    #[must_use]
    pub const fn cross_range(mut self, range: i32) -> Self {
        self.cross_range = range;
        self
    }

    /// Performs unsymmetrical cross search.
    fn cross_search(
        &self,
        ctx: &SearchContext,
        config: &SearchConfig,
        center: MotionVector,
    ) -> (MotionVector, u32) {
        let mut best_mv = center;
        let mut best_sad = ctx.calculate_sad(&center).unwrap_or(u32::MAX);
        let range = config.range.horizontal.min(config.range.vertical) * self.cross_range;

        // Horizontal cross (more points)
        for dx in (-range..=range).step_by(2) {
            if dx == 0 {
                continue;
            }
            let mv = MotionVector::from_full_pel(center.full_pel_x() + dx, center.full_pel_y());

            if !ctx.is_valid_mv(&mv, &config.range) {
                continue;
            }

            if let Some(sad) = ctx.calculate_sad(&mv) {
                if sad < best_sad {
                    best_sad = sad;
                    best_mv = mv;
                }
            }
        }

        // Vertical cross (fewer points - unsymmetrical)
        for dy in (-range / 2..=range / 2).step_by(2) {
            if dy == 0 {
                continue;
            }
            let mv = MotionVector::from_full_pel(center.full_pel_x(), center.full_pel_y() + dy);

            if !ctx.is_valid_mv(&mv, &config.range) {
                continue;
            }

            if let Some(sad) = ctx.calculate_sad(&mv) {
                if sad < best_sad {
                    best_sad = sad;
                    best_mv = mv;
                }
            }
        }

        (best_mv, best_sad)
    }

    /// Multi-hexagon grid search.
    fn multi_hexagon(
        &self,
        ctx: &SearchContext,
        config: &SearchConfig,
        center: MotionVector,
    ) -> (MotionVector, u32) {
        let mut best_mv = center;
        let mut best_sad = ctx.calculate_sad(&center).unwrap_or(u32::MAX);

        // 16 points in a multi-hexagon pattern
        const MULTI_HEX: [(i32, i32); 16] = [
            (-4, 0),
            (-2, -2),
            (0, -4),
            (2, -2),
            (4, 0),
            (2, 2),
            (0, 4),
            (-2, 2),
            (-2, -4),
            (2, -4),
            (4, -2),
            (4, 2),
            (2, 4),
            (-2, 4),
            (-4, 2),
            (-4, -2),
        ];

        for &(dx, dy) in &MULTI_HEX {
            let mv =
                MotionVector::from_full_pel(center.full_pel_x() + dx, center.full_pel_y() + dy);

            if !ctx.is_valid_mv(&mv, &config.range) {
                continue;
            }

            if let Some(sad) = ctx.calculate_sad(&mv) {
                if sad < best_sad {
                    best_sad = sad;
                    best_mv = mv;
                }
            }
        }

        (best_mv, best_sad)
    }
}

impl MotionSearch for UmhSearch {
    fn search(&self, ctx: &SearchContext, config: &SearchConfig) -> BlockMatch {
        self.search_with_predictor(ctx, config, MotionVector::zero())
    }

    fn search_with_predictor(
        &self,
        ctx: &SearchContext,
        config: &SearchConfig,
        predictor: MotionVector,
    ) -> BlockMatch {
        let mut center = predictor.to_precision(MvPrecision::FullPel);
        let mut best_sad = ctx.calculate_sad(&center).unwrap_or(u32::MAX);

        // Step 1: Unsymmetrical cross search
        let (cross_mv, cross_sad) = self.cross_search(ctx, config, center);
        if cross_sad < best_sad {
            center = cross_mv;
            best_sad = cross_sad;
        }

        // Early termination check
        if config.early_termination && best_sad < config.early_threshold {
            let cost = config.mv_cost.rd_cost(&center, best_sad);
            return BlockMatch::new(center, best_sad, cost);
        }

        // Step 2: Multi-hexagon grid
        let (hex_mv, hex_sad) = self.multi_hexagon(ctx, config, center);
        if hex_sad < best_sad {
            center = hex_mv;
            best_sad = hex_sad;
        }

        // Step 3: Extended hexagon search
        let hex_search = HexagonSearch::new();
        for _ in 0..config.max_iterations / 2 {
            let (new_center, new_sad) = hex_search.hexagon_step(ctx, config, center);
            if new_center == center {
                break;
            }
            center = new_center;
            best_sad = new_sad;
        }

        // Step 4: Small diamond refinement
        let diamond = DiamondSearch::new().use_large_diamond(false);
        let result = diamond.search_with_predictor(ctx, config, center);

        if result.sad < best_sad {
            result
        } else {
            let cost = config.mv_cost.rd_cost(&center, best_sad);
            BlockMatch::new(center, best_sad, cost)
        }
    }
}

/// Three-step search algorithm.
///
/// Classic fast search algorithm that reduces search space logarithmically.
#[derive(Clone, Debug, Default)]
pub struct ThreeStepSearch;

impl ThreeStepSearch {
    /// Creates a new three-step search instance.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Square pattern for each step.
    const SQUARE_8: [(i32, i32); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];
}

impl MotionSearch for ThreeStepSearch {
    fn search(&self, ctx: &SearchContext, config: &SearchConfig) -> BlockMatch {
        self.search_with_predictor(ctx, config, MotionVector::zero())
    }

    fn search_with_predictor(
        &self,
        ctx: &SearchContext,
        config: &SearchConfig,
        predictor: MotionVector,
    ) -> BlockMatch {
        let mut center = predictor.to_precision(MvPrecision::FullPel);
        let mut best_sad = ctx.calculate_sad(&center).unwrap_or(u32::MAX);

        // Initial step size (half of search range)
        let initial_step = config.range.horizontal.min(config.range.vertical) / 2;
        let mut step = initial_step.max(4);

        // Three steps with decreasing step size
        while step >= 1 {
            let mut moved = false;

            for &(dx, dy) in &Self::SQUARE_8 {
                let mv = MotionVector::from_full_pel(
                    center.full_pel_x() + dx * step,
                    center.full_pel_y() + dy * step,
                );

                if !ctx.is_valid_mv(&mv, &config.range) {
                    continue;
                }

                if let Some(sad) = ctx.calculate_sad(&mv) {
                    if sad < best_sad {
                        best_sad = sad;
                        center = mv;
                        moved = true;
                    }
                }
            }

            // If no movement, reduce step size
            if !moved {
                step /= 2;
            }

            // Early termination
            if config.early_termination && best_sad < config.early_threshold {
                break;
            }
        }

        let cost = config.mv_cost.rd_cost(&center, best_sad);
        BlockMatch::new(center, best_sad, cost)
    }
}

/// Adaptive search that selects algorithm based on complexity.
#[derive(Clone, Debug)]
pub struct AdaptiveSearch {
    /// Threshold for switching to simpler algorithm.
    complexity_threshold: u32,
}

impl Default for AdaptiveSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveSearch {
    /// Creates a new adaptive search instance.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            complexity_threshold: 1000,
        }
    }

    /// Sets the complexity threshold.
    #[must_use]
    pub const fn threshold(mut self, threshold: u32) -> Self {
        self.complexity_threshold = threshold;
        self
    }

    /// Estimates block complexity (variance-like measure).
    fn estimate_complexity(&self, ctx: &SearchContext) -> u32 {
        let src_offset = ctx.src_offset();
        let width = ctx.block_size.width();
        let height = ctx.block_size.height();

        let mut sum = 0u32;
        let mut sum_sq = 0u64;
        let mut count = 0u32;

        for row in 0..height {
            let row_offset = src_offset + row * ctx.src_stride;
            for col in 0..width {
                if row_offset + col < ctx.src.len() {
                    let val = u32::from(ctx.src[row_offset + col]);
                    sum += val;
                    sum_sq += u64::from(val * val);
                    count += 1;
                }
            }
        }

        if count == 0 {
            return 0;
        }

        let mean = sum / count;

        (sum_sq / u64::from(count)) as u32 - mean * mean
    }
}

impl MotionSearch for AdaptiveSearch {
    fn search(&self, ctx: &SearchContext, config: &SearchConfig) -> BlockMatch {
        self.search_with_predictor(ctx, config, MotionVector::zero())
    }

    fn search_with_predictor(
        &self,
        ctx: &SearchContext,
        config: &SearchConfig,
        predictor: MotionVector,
    ) -> BlockMatch {
        let complexity = self.estimate_complexity(ctx);

        if complexity < self.complexity_threshold {
            // Low complexity: use faster diamond search
            DiamondSearch::new().search_with_predictor(ctx, config, predictor)
        } else {
            // High complexity: use more thorough UMH search
            UmhSearch::new().search_with_predictor(ctx, config, predictor)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_context<'a>(
        src: &'a [u8],
        ref_frame: &'a [u8],
        width: usize,
        height: usize,
    ) -> SearchContext<'a> {
        SearchContext::new(
            src,
            width,
            ref_frame,
            width,
            BlockSize::Block8x8,
            0,
            0,
            width,
            height,
        )
    }

    #[test]
    fn test_search_config_default() {
        let config = SearchConfig::default();
        assert_eq!(config.precision, MvPrecision::QuarterPel);
        assert!(config.early_termination);
    }

    #[test]
    fn test_search_config_builder() {
        let config = SearchConfig::default()
            .range(SearchRange::symmetric(32))
            .precision(MvPrecision::HalfPel)
            .early_termination(false);

        assert_eq!(config.range.horizontal, 32);
        assert_eq!(config.precision, MvPrecision::HalfPel);
        assert!(!config.early_termination);
    }

    #[test]
    fn test_search_context_sad_calculation() {
        let src = vec![100u8; 64]; // 8x8 block
        let ref_frame = vec![110u8; 64]; // 8x8 with offset

        let ctx = create_test_context(&src, &ref_frame, 8, 8);
        let mv = MotionVector::zero();

        let sad = ctx.calculate_sad(&mv);
        assert!(sad.is_some());
        assert_eq!(sad.expect("should succeed"), 640); // 64 * 10
    }

    #[test]
    fn test_search_context_identical_blocks() {
        let data = vec![128u8; 64];
        let ctx = create_test_context(&data, &data, 8, 8);
        let mv = MotionVector::zero();

        let sad = ctx.calculate_sad(&mv);
        assert_eq!(sad, Some(0));
    }

    #[test]
    fn test_full_search() {
        let src = vec![100u8; 64];
        let mut ref_frame = vec![50u8; 256]; // 16x16

        // Place matching block at (4, 4)
        for row in 0..8 {
            for col in 0..8 {
                ref_frame[(row + 4) * 16 + col + 4] = 100;
            }
        }

        let ctx = SearchContext::new(&src, 8, &ref_frame, 16, BlockSize::Block8x8, 0, 0, 16, 16);
        let config = SearchConfig::with_range(SearchRange::symmetric(8));

        let searcher = FullSearch::new();
        let result = searcher.search(&ctx, &config);

        assert_eq!(result.mv.full_pel_x(), 4);
        assert_eq!(result.mv.full_pel_y(), 4);
        assert_eq!(result.sad, 0);
    }

    #[test]
    fn test_diamond_search() {
        let src = vec![100u8; 64];
        let mut ref_frame = vec![50u8; 256];

        // Place matching block at (4, 4)
        for row in 0..8 {
            for col in 0..8 {
                ref_frame[(row + 4) * 16 + col + 4] = 100;
            }
        }

        let ctx = SearchContext::new(&src, 8, &ref_frame, 16, BlockSize::Block8x8, 0, 0, 16, 16);
        let config = SearchConfig::with_range(SearchRange::symmetric(8));

        let searcher = DiamondSearch::new();
        let result = searcher.search(&ctx, &config);

        // Diamond search should find reasonably close match
        assert!(result.sad < 1000);
    }

    #[test]
    fn test_hexagon_search() {
        let src = vec![100u8; 64];
        let mut ref_frame = vec![50u8; 256];

        for row in 0..8 {
            for col in 0..8 {
                ref_frame[(row + 4) * 16 + col + 4] = 100;
            }
        }

        let ctx = SearchContext::new(&src, 8, &ref_frame, 16, BlockSize::Block8x8, 0, 0, 16, 16);
        let config = SearchConfig::with_range(SearchRange::symmetric(8));

        let searcher = HexagonSearch::new();
        let result = searcher.search(&ctx, &config);

        assert!(result.sad < 1000);
    }

    #[test]
    fn test_umh_search() {
        let src = vec![100u8; 64];
        let mut ref_frame = vec![50u8; 256];

        for row in 0..8 {
            for col in 0..8 {
                ref_frame[(row + 4) * 16 + col + 4] = 100;
            }
        }

        let ctx = SearchContext::new(&src, 8, &ref_frame, 16, BlockSize::Block8x8, 0, 0, 16, 16);
        let config = SearchConfig::with_range(SearchRange::symmetric(8));

        let searcher = UmhSearch::new();
        let result = searcher.search(&ctx, &config);

        assert!(result.sad < 1000);
    }

    #[test]
    fn test_three_step_search() {
        let src = vec![100u8; 64];
        let mut ref_frame = vec![50u8; 256];

        for row in 0..8 {
            for col in 0..8 {
                ref_frame[(row + 4) * 16 + col + 4] = 100;
            }
        }

        let ctx = SearchContext::new(&src, 8, &ref_frame, 16, BlockSize::Block8x8, 0, 0, 16, 16);
        let config = SearchConfig::with_range(SearchRange::symmetric(8));

        let searcher = ThreeStepSearch::new();
        let result = searcher.search(&ctx, &config);

        assert!(result.sad < 1000);
    }

    #[test]
    fn test_search_with_predictor() {
        let src = vec![100u8; 64];
        let mut ref_frame = vec![50u8; 256];

        for row in 0..8 {
            for col in 0..8 {
                ref_frame[(row + 4) * 16 + col + 4] = 100;
            }
        }

        let ctx = SearchContext::new(&src, 8, &ref_frame, 16, BlockSize::Block8x8, 0, 0, 16, 16);
        let config = SearchConfig::with_range(SearchRange::symmetric(8));

        // Good predictor should help find the match faster
        let predictor = MotionVector::from_full_pel(3, 3);
        let searcher = DiamondSearch::new();
        let result = searcher.search_with_predictor(&ctx, &config, predictor);

        assert!(result.sad < 500);
    }

    #[test]
    fn test_early_termination() {
        let data = vec![128u8; 64];
        let mut ref_frame = vec![128u8; 256];

        // Matching block at origin
        for row in 0..8 {
            for col in 0..8 {
                ref_frame[row * 16 + col] = 128;
            }
        }

        let ctx = SearchContext::new(&data, 8, &ref_frame, 16, BlockSize::Block8x8, 0, 0, 16, 16);
        let config = SearchConfig::default().early_termination(true);

        let searcher = FullSearch::new();
        let result = searcher.search(&ctx, &config);

        // Should find perfect match at origin
        assert_eq!(result.sad, 0);
        assert_eq!(result.mv.full_pel_x(), 0);
        assert_eq!(result.mv.full_pel_y(), 0);
    }

    #[test]
    fn test_adaptive_search() {
        let src = vec![100u8; 64];
        let ref_frame = vec![100u8; 256];

        let ctx = SearchContext::new(&src, 8, &ref_frame, 16, BlockSize::Block8x8, 0, 0, 16, 16);
        let config = SearchConfig::with_range(SearchRange::symmetric(8));

        let searcher = AdaptiveSearch::new();
        let result = searcher.search(&ctx, &config);

        // Should work regardless of which algorithm is chosen
        assert!(result.cost < u32::MAX);
    }

    #[test]
    fn test_out_of_bounds_mv() {
        let src = vec![100u8; 64];
        let ref_frame = vec![100u8; 64];

        let ctx = SearchContext::new(&src, 8, &ref_frame, 8, BlockSize::Block8x8, 0, 0, 8, 8);

        // MV that would go out of bounds
        let mv = MotionVector::from_full_pel(5, 5);
        let sad = ctx.calculate_sad(&mv);

        assert!(sad.is_none());
    }

    #[test]
    fn test_is_valid_mv() {
        let src = vec![100u8; 64];
        let ref_frame = vec![100u8; 256];

        let ctx = SearchContext::new(&src, 8, &ref_frame, 16, BlockSize::Block8x8, 0, 0, 16, 16);
        let range = SearchRange::symmetric(8);

        // Valid MV
        assert!(ctx.is_valid_mv(&MotionVector::from_full_pel(4, 4), &range));

        // Invalid: out of range
        assert!(!ctx.is_valid_mv(&MotionVector::from_full_pel(10, 0), &range));

        // Invalid: would go out of frame bounds
        assert!(!ctx.is_valid_mv(&MotionVector::from_full_pel(9, 9), &range));
    }
}
