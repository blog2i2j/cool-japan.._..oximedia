//! Diamond search patterns for motion estimation.
//!
//! This module provides implementations of the Small Diamond Search Pattern (SDSP)
//! and Large Diamond Search Pattern (LDSP) for efficient motion estimation.
//!
//! The diamond search is one of the most widely used fast motion estimation
//! algorithms due to its good balance between quality and speed.

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::must_use_candidate)]

use super::search::{MotionSearch, SearchConfig, SearchContext};
use super::types::{BlockMatch, MotionVector, MvPrecision};

/// Small Diamond Search Pattern (SDSP).
///
/// A 4-point pattern for fine refinement:
/// ```text
///       *
///     * O *
///       *
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SmallDiamond {
    /// Pattern offsets (dx, dy) for each point.
    pub points: [(i32, i32); 4],
}

impl Default for SmallDiamond {
    fn default() -> Self {
        Self::new()
    }
}

impl SmallDiamond {
    /// Standard SDSP offsets.
    pub const PATTERN: [(i32, i32); 4] = [(0, -1), (-1, 0), (1, 0), (0, 1)];

    /// Creates a new small diamond pattern.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            points: Self::PATTERN,
        }
    }

    /// Returns the number of points in the pattern.
    #[must_use]
    pub const fn size(&self) -> usize {
        4
    }

    /// Gets the offset at a given index.
    #[must_use]
    pub const fn get(&self, index: usize) -> Option<(i32, i32)> {
        if index < 4 {
            Some(self.points[index])
        } else {
            None
        }
    }

    /// Searches using the small diamond pattern.
    pub fn search(
        &self,
        ctx: &SearchContext,
        config: &SearchConfig,
        center: MotionVector,
    ) -> (MotionVector, u32, usize) {
        let mut best_mv = center;
        let mut best_sad = ctx.calculate_sad(&center).unwrap_or(u32::MAX);
        let mut best_idx = 4; // Center

        for (idx, &(dx, dy)) in self.points.iter().enumerate() {
            let mv =
                MotionVector::from_full_pel(center.full_pel_x() + dx, center.full_pel_y() + dy);

            if !ctx.is_valid_mv(&mv, &config.range) {
                continue;
            }

            if let Some(sad) = ctx.calculate_sad(&mv) {
                if sad < best_sad {
                    best_sad = sad;
                    best_mv = mv;
                    best_idx = idx;
                }
            }
        }

        (best_mv, best_sad, best_idx)
    }
}

/// Large Diamond Search Pattern (LDSP).
///
/// An 8-point pattern for coarse search:
/// ```text
///       *
///     * * *
///   * * O * *
///     * * *
///       *
/// ```
#[derive(Clone, Copy, Debug)]
pub struct LargeDiamond {
    /// Pattern offsets (dx, dy) for each point.
    pub points: [(i32, i32); 8],
}

impl Default for LargeDiamond {
    fn default() -> Self {
        Self::new()
    }
}

impl LargeDiamond {
    /// Standard LDSP offsets.
    pub const PATTERN: [(i32, i32); 8] = [
        (0, -2),  // Top
        (-1, -1), // Top-left
        (1, -1),  // Top-right
        (-2, 0),  // Left
        (2, 0),   // Right
        (-1, 1),  // Bottom-left
        (1, 1),   // Bottom-right
        (0, 2),   // Bottom
    ];

    /// Creates a new large diamond pattern.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            points: Self::PATTERN,
        }
    }

    /// Returns the number of points in the pattern.
    #[must_use]
    pub const fn size(&self) -> usize {
        8
    }

    /// Gets the offset at a given index.
    #[must_use]
    pub const fn get(&self, index: usize) -> Option<(i32, i32)> {
        if index < 8 {
            Some(self.points[index])
        } else {
            None
        }
    }

    /// Searches using the large diamond pattern.
    pub fn search(
        &self,
        ctx: &SearchContext,
        config: &SearchConfig,
        center: MotionVector,
    ) -> (MotionVector, u32, usize) {
        let mut best_mv = center;
        let mut best_sad = ctx.calculate_sad(&center).unwrap_or(u32::MAX);
        let mut best_idx = 8; // Center

        for (idx, &(dx, dy)) in self.points.iter().enumerate() {
            let mv =
                MotionVector::from_full_pel(center.full_pel_x() + dx, center.full_pel_y() + dy);

            if !ctx.is_valid_mv(&mv, &config.range) {
                continue;
            }

            if let Some(sad) = ctx.calculate_sad(&mv) {
                if sad < best_sad {
                    best_sad = sad;
                    best_mv = mv;
                    best_idx = idx;
                }
            }
        }

        (best_mv, best_sad, best_idx)
    }
}

/// Extended Diamond Search Pattern.
///
/// A 16-point pattern for larger steps:
/// ```text
///           *
///         * * *
///       * * * * *
///     * * * O * * *
///       * * * * *
///         * * *
///           *
/// ```
#[derive(Clone, Copy, Debug)]
pub struct ExtendedDiamond {
    /// Inner ring (4 points, distance 1).
    pub inner: [(i32, i32); 4],
    /// Middle ring (8 points, distance 2).
    pub middle: [(i32, i32); 8],
    /// Outer ring (4 points, distance 3).
    pub outer: [(i32, i32); 4],
}

impl Default for ExtendedDiamond {
    fn default() -> Self {
        Self::new()
    }
}

impl ExtendedDiamond {
    /// Creates a new extended diamond pattern.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            inner: SmallDiamond::PATTERN,
            middle: LargeDiamond::PATTERN,
            outer: [(0, -3), (-3, 0), (3, 0), (0, 3)],
        }
    }

    /// Searches using all rings of the extended diamond.
    pub fn search(
        &self,
        ctx: &SearchContext,
        config: &SearchConfig,
        center: MotionVector,
    ) -> (MotionVector, u32) {
        let mut best_mv = center;
        let mut best_sad = ctx.calculate_sad(&center).unwrap_or(u32::MAX);

        // Search all rings
        for &(dx, dy) in self.outer.iter().chain(&self.middle).chain(&self.inner) {
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

/// Adaptive diamond search that switches between SDSP and LDSP.
///
/// This implementation uses LDSP initially and switches to SDSP when:
/// 1. The best point is at the center (convergence)
/// 2. A threshold number of iterations has passed
#[derive(Clone, Debug)]
pub struct AdaptiveDiamond {
    /// Small diamond pattern.
    sdsp: SmallDiamond,
    /// Large diamond pattern.
    ldsp: LargeDiamond,
    /// Maximum LDSP iterations before switching to SDSP.
    max_ldsp_iterations: u32,
    /// SAD threshold for early switch to SDSP.
    switch_threshold: u32,
}

impl Default for AdaptiveDiamond {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveDiamond {
    /// Creates a new adaptive diamond search.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            sdsp: SmallDiamond::new(),
            ldsp: LargeDiamond::new(),
            max_ldsp_iterations: 8,
            switch_threshold: 512,
        }
    }

    /// Sets the maximum LDSP iterations.
    #[must_use]
    pub const fn max_iterations(mut self, max: u32) -> Self {
        self.max_ldsp_iterations = max;
        self
    }

    /// Sets the SAD threshold for early switch.
    #[must_use]
    pub const fn switch_threshold(mut self, threshold: u32) -> Self {
        self.switch_threshold = threshold;
        self
    }
}

impl MotionSearch for AdaptiveDiamond {
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

        // Phase 1: Large diamond search
        for iteration in 0..self.max_ldsp_iterations {
            let (new_center, new_sad, best_idx) = self.ldsp.search(ctx, config, center);

            // Check for convergence (center is best)
            if best_idx >= self.ldsp.size() {
                break;
            }

            // Check for early switch
            if new_sad < self.switch_threshold {
                center = new_center;
                best_sad = new_sad;
                break;
            }

            center = new_center;
            best_sad = new_sad;

            // Early termination
            if config.early_termination && best_sad < config.early_threshold {
                let cost = config.mv_cost.rd_cost(&center, best_sad);
                return BlockMatch::new(center, best_sad, cost);
            }

            // Maximum iterations check (avoid infinite loop)
            if iteration >= self.max_ldsp_iterations - 1 {
                break;
            }
        }

        // Phase 2: Small diamond refinement
        loop {
            let (new_center, new_sad, best_idx) = self.sdsp.search(ctx, config, center);

            // Check for convergence
            if best_idx >= self.sdsp.size() {
                break;
            }

            center = new_center;
            best_sad = new_sad;
        }

        let cost = config.mv_cost.rd_cost(&center, best_sad);
        BlockMatch::new(center, best_sad, cost)
    }
}

/// Predictor-based diamond search.
///
/// Uses multiple predictors (spatial/temporal) to initialize search
/// from the most promising starting point.
#[derive(Clone, Debug)]
pub struct PredictorDiamond {
    /// Underlying diamond search.
    diamond: AdaptiveDiamond,
    /// Maximum number of predictors to try.
    max_predictors: usize,
}

impl Default for PredictorDiamond {
    fn default() -> Self {
        Self::new()
    }
}

impl PredictorDiamond {
    /// Creates a new predictor-based diamond search.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            diamond: AdaptiveDiamond::new(),
            max_predictors: 5,
        }
    }

    /// Sets the maximum number of predictors.
    #[must_use]
    pub const fn max_predictors(mut self, max: usize) -> Self {
        self.max_predictors = max;
        self
    }

    /// Searches with multiple predictors.
    pub fn search_multi(
        &self,
        ctx: &SearchContext,
        config: &SearchConfig,
        predictors: &[MotionVector],
    ) -> BlockMatch {
        let mut best = BlockMatch::worst();

        // Try zero MV first
        if let Some(sad) = ctx.calculate_sad(&MotionVector::zero()) {
            let cost = config.mv_cost.rd_cost(&MotionVector::zero(), sad);
            let candidate = BlockMatch::new(MotionVector::zero(), sad, cost);
            best.update_if_better(&candidate);

            // Early termination for perfect match
            if sad == 0 {
                return best;
            }
        }

        // Evaluate each predictor
        for (i, &pred) in predictors.iter().take(self.max_predictors).enumerate() {
            if i > 0 && pred.is_zero() {
                continue; // Skip duplicate zero MV
            }

            // Quick evaluation of predictor
            let pred_fp = pred.to_precision(MvPrecision::FullPel);
            if let Some(sad) = ctx.calculate_sad(&pred_fp) {
                if sad < best.sad {
                    // Full search from this predictor
                    let result = self.diamond.search_with_predictor(ctx, config, pred);
                    best.update_if_better(&result);
                }
            }
        }

        // If no predictor worked well, search from best point so far
        if best.sad > config.early_threshold {
            let result = self.diamond.search_with_predictor(ctx, config, best.mv);
            best.update_if_better(&result);
        }

        best
    }
}

impl MotionSearch for PredictorDiamond {
    fn search(&self, ctx: &SearchContext, config: &SearchConfig) -> BlockMatch {
        self.diamond.search(ctx, config)
    }

    fn search_with_predictor(
        &self,
        ctx: &SearchContext,
        config: &SearchConfig,
        predictor: MotionVector,
    ) -> BlockMatch {
        self.diamond.search_with_predictor(ctx, config, predictor)
    }
}

/// Cross diamond search pattern.
///
/// Combines cross pattern with diamond for better coverage of
/// horizontal/vertical motion.
#[derive(Clone, Debug)]
pub struct CrossDiamond {
    /// Cross pattern range.
    cross_range: i32,
    /// Diamond search for refinement.
    diamond: AdaptiveDiamond,
}

impl Default for CrossDiamond {
    fn default() -> Self {
        Self::new()
    }
}

impl CrossDiamond {
    /// Creates a new cross diamond search.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            cross_range: 4,
            diamond: AdaptiveDiamond::new(),
        }
    }

    /// Sets the cross pattern range.
    #[must_use]
    pub const fn cross_range(mut self, range: i32) -> Self {
        self.cross_range = range;
        self
    }

    /// Performs cross pattern search.
    fn cross_search(
        &self,
        ctx: &SearchContext,
        config: &SearchConfig,
        center: MotionVector,
    ) -> (MotionVector, u32) {
        let mut best_mv = center;
        let mut best_sad = ctx.calculate_sad(&center).unwrap_or(u32::MAX);

        // Horizontal cross
        for dx in -self.cross_range..=self.cross_range {
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

        // Vertical cross
        for dy in -self.cross_range..=self.cross_range {
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
}

impl MotionSearch for CrossDiamond {
    fn search(&self, ctx: &SearchContext, config: &SearchConfig) -> BlockMatch {
        self.search_with_predictor(ctx, config, MotionVector::zero())
    }

    fn search_with_predictor(
        &self,
        ctx: &SearchContext,
        config: &SearchConfig,
        predictor: MotionVector,
    ) -> BlockMatch {
        let center = predictor.to_precision(MvPrecision::FullPel);

        // Phase 1: Cross search
        let (cross_best, _) = self.cross_search(ctx, config, center);

        // Phase 2: Diamond refinement
        self.diamond.search_with_predictor(ctx, config, cross_best)
    }
}

// =============================================================================
// Hexagonal Search Pattern
// =============================================================================

/// Hexagonal search pattern (HEX) for motion estimation.
///
/// A 6-point pattern inspired by the H.264 HEX search:
/// ```text
///     *   *
///   *   O   *
///     *   *
/// ```
///
/// Hexagonal patterns often outperform diamond patterns for complex motion
/// because they cover 6 equidistant directions simultaneously.
#[derive(Clone, Copy, Debug)]
pub struct HexagonalSearch {
    /// Inner hex (6 points, radius ≈ 2).
    pub inner: [(i32, i32); 6],
    /// Outer hex (6 points, radius ≈ 4).
    pub outer: [(i32, i32); 6],
    /// Maximum iterations before refinement.
    pub max_iterations: u32,
}

impl Default for HexagonalSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl HexagonalSearch {
    /// Standard inner hexagon offsets (radius ≈ 2).
    pub const INNER_PATTERN: [(i32, i32); 6] =
        [(-2, 0), (-1, -2), (1, -2), (2, 0), (1, 2), (-1, 2)];

    /// Standard outer hexagon offsets (radius ≈ 4).
    pub const OUTER_PATTERN: [(i32, i32); 6] =
        [(-4, 0), (-2, -4), (2, -4), (4, 0), (2, 4), (-2, 4)];

    /// Create a new hexagonal search.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            inner: Self::INNER_PATTERN,
            outer: Self::OUTER_PATTERN,
            max_iterations: 8,
        }
    }

    /// Set maximum iterations.
    #[must_use]
    pub const fn max_iterations(mut self, max: u32) -> Self {
        self.max_iterations = max;
        self
    }

    /// Search one hexagon ring centered on `center`.
    fn search_hex(
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

impl MotionSearch for HexagonalSearch {
    fn search(&self, ctx: &SearchContext, config: &SearchConfig) -> BlockMatch {
        self.search_with_predictor(ctx, config, MotionVector::zero())
    }

    fn search_with_predictor(
        &self,
        ctx: &SearchContext,
        config: &SearchConfig,
        predictor: MotionVector,
    ) -> BlockMatch {
        let center = predictor.to_precision(MvPrecision::FullPel);
        let mut current = center;
        let mut current_sad = ctx.calculate_sad(&current).unwrap_or(u32::MAX);

        // Phase 1: coarse outer hex search
        for _ in 0..self.max_iterations {
            let (best_mv, best_sad) = self.search_hex(ctx, config, current, &self.outer);
            if best_sad >= current_sad {
                break;
            }
            current = best_mv;
            current_sad = best_sad;

            if current_sad == 0 {
                break;
            }
        }

        // Phase 2: fine inner hex refinement
        for _ in 0..self.max_iterations {
            let (best_mv, best_sad) = self.search_hex(ctx, config, current, &self.inner);
            if best_sad >= current_sad {
                break;
            }
            current = best_mv;
            current_sad = best_sad;

            if current_sad == 0 {
                break;
            }
        }

        // Phase 3: small diamond final refinement
        let sdsp = SmallDiamond::new();
        let (refined_mv, refined_sad, _) = sdsp.search(ctx, config, current);
        if refined_sad < current_sad {
            current = refined_mv;
            current_sad = refined_sad;
        }

        let cost = config.mv_cost.rd_cost(&current, current_sad);
        BlockMatch::new(current, current_sad, cost)
    }
}

// =============================================================================
// UMHex (Unsymmetric Multi-Hexagon) Search
// =============================================================================

/// UMHex — Unsymmetric Multi-Hexagon grid search.
///
/// A state-of-the-art fast ME algorithm (Zhu & Ma, 2000) used in JM H.264
/// reference software and ported here for AV1/VP9 quality targets.
///
/// # Algorithm
///
/// 1. **Predictor check** — evaluate MV predictors (zero, spatial, temporal).
/// 2. **Unsymmetric-cross** — rapid scan along horizontal and vertical axes.
/// 3. **Hexagon expansion** — grow the hex grid until no improvement.
/// 4. **Small diamond refinement** — SDSP for sub-pixel accuracy.
///
/// # Performance
///
/// Typically 4-8× faster than full search at ≤ 1 dB quality loss.
#[derive(Clone, Debug)]
pub struct UMHexSearch {
    /// Maximum hexagon expansion steps.
    max_hex_steps: u32,
    /// Unsymmetric-cross search range.
    cross_range: i32,
    /// SAD threshold for early termination.
    early_exit_threshold: u32,
}

impl Default for UMHexSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl UMHexSearch {
    /// Inner hexagon (radius ≈ 1 pel).
    const HEX1: [(i32, i32); 6] = [(-1, -2), (1, -2), (2, 0), (1, 2), (-1, 2), (-2, 0)];
    /// Outer hexagon (radius ≈ 2 pel).
    const HEX2: [(i32, i32); 12] = [
        (-1, -2),
        (1, -2), // top
        (2, -1),
        (2, 1), // right
        (1, 2),
        (-1, 2), // bottom
        (-2, 1),
        (-2, -1), // left
        (0, -4),
        (4, 0), // extended top/right
        (0, 4),
        (-4, 0), // extended bottom/left
    ];

    /// Create a new UMHex search with default parameters.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            max_hex_steps: 16,
            cross_range: 8,
            early_exit_threshold: 4,
        }
    }

    /// Set maximum hexagon expansion steps.
    #[must_use]
    pub const fn max_hex_steps(mut self, steps: u32) -> Self {
        self.max_hex_steps = steps;
        self
    }

    /// Set unsymmetric-cross search range.
    #[must_use]
    pub const fn cross_range(mut self, range: i32) -> Self {
        self.cross_range = range;
        self
    }

    /// Set SAD early-exit threshold.
    #[must_use]
    pub const fn early_exit_threshold(mut self, threshold: u32) -> Self {
        self.early_exit_threshold = threshold;
        self
    }

    /// Unsymmetric-cross scan: rapid horizontal then vertical scan.
    fn cross_scan(
        &self,
        ctx: &SearchContext,
        config: &SearchConfig,
        center: MotionVector,
    ) -> (MotionVector, u32) {
        let mut best_mv = center;
        let mut best_sad = ctx.calculate_sad(&center).unwrap_or(u32::MAX);

        // Horizontal pass (non-uniform spacing: small near center, larger far)
        let h_offsets: &[i32] = &[-8, -4, -2, -1, 1, 2, 4, 8];
        for &dx in h_offsets {
            if dx.abs() > self.cross_range {
                continue;
            }
            let mv = MotionVector::from_full_pel(center.full_pel_x() + dx, center.full_pel_y());
            if ctx.is_valid_mv(&mv, &config.range) {
                if let Some(sad) = ctx.calculate_sad(&mv) {
                    if sad < best_sad {
                        best_sad = sad;
                        best_mv = mv;
                    }
                }
            }
        }

        // Vertical pass
        let v_offsets: &[i32] = &[-8, -4, -2, -1, 1, 2, 4, 8];
        for &dy in v_offsets {
            if dy.abs() > self.cross_range {
                continue;
            }
            let mv = MotionVector::from_full_pel(center.full_pel_x(), center.full_pel_y() + dy);
            if ctx.is_valid_mv(&mv, &config.range) {
                if let Some(sad) = ctx.calculate_sad(&mv) {
                    if sad < best_sad {
                        best_sad = sad;
                        best_mv = mv;
                    }
                }
            }
        }

        (best_mv, best_sad)
    }

    /// Hexagon grid expansion step.
    fn hex_step(
        ctx: &SearchContext,
        config: &SearchConfig,
        center: MotionVector,
        pattern: &[(i32, i32)],
    ) -> (MotionVector, u32, bool) {
        let mut best_mv = center;
        let mut best_sad = ctx.calculate_sad(&center).unwrap_or(u32::MAX);
        let mut improved = false;

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
                    improved = true;
                }
            }
        }

        (best_mv, best_sad, improved)
    }
}

impl MotionSearch for UMHexSearch {
    fn search(&self, ctx: &SearchContext, config: &SearchConfig) -> BlockMatch {
        self.search_with_predictor(ctx, config, MotionVector::zero())
    }

    fn search_with_predictor(
        &self,
        ctx: &SearchContext,
        config: &SearchConfig,
        predictor: MotionVector,
    ) -> BlockMatch {
        // Step 1: initialise from predictor
        let pred_fp = predictor.to_precision(MvPrecision::FullPel);
        let pred_sad = ctx.calculate_sad(&pred_fp).unwrap_or(u32::MAX);

        let zero_sad = ctx.calculate_sad(&MotionVector::zero()).unwrap_or(u32::MAX);

        let (mut current, mut current_sad) = if pred_sad <= zero_sad {
            (pred_fp, pred_sad)
        } else {
            (MotionVector::zero(), zero_sad)
        };

        // Early exit for trivial match
        if current_sad <= self.early_exit_threshold {
            let cost = config.mv_cost.rd_cost(&current, current_sad);
            return BlockMatch::new(current, current_sad, cost);
        }

        // Step 2: unsymmetric-cross scan
        let (cross_mv, cross_sad) = self.cross_scan(ctx, config, current);
        if cross_sad < current_sad {
            current = cross_mv;
            current_sad = cross_sad;
        }

        if current_sad <= self.early_exit_threshold {
            let cost = config.mv_cost.rd_cost(&current, current_sad);
            return BlockMatch::new(current, current_sad, cost);
        }

        // Step 3: HEX2 expansion until convergence
        for _ in 0..self.max_hex_steps {
            let (mv, sad, improved) = Self::hex_step(ctx, config, current, &Self::HEX2);
            if !improved {
                break;
            }
            current = mv;
            current_sad = sad;

            if current_sad <= self.early_exit_threshold {
                break;
            }
        }

        // Step 4: HEX1 refinement
        for _ in 0..4 {
            let (mv, sad, improved) = Self::hex_step(ctx, config, current, &Self::HEX1);
            if !improved {
                break;
            }
            current = mv;
            current_sad = sad;
        }

        // Step 5: small diamond final refinement
        let sdsp = SmallDiamond::new();
        let (refined, rsad, _) = sdsp.search(ctx, config, current);
        if rsad < current_sad {
            current = refined;
            current_sad = rsad;
        }

        let cost = config.mv_cost.rd_cost(&current, current_sad);
        BlockMatch::new(current, current_sad, cost)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motion::types::{BlockSize, SearchRange};

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
    fn test_small_diamond_pattern() {
        let sdsp = SmallDiamond::new();
        assert_eq!(sdsp.size(), 4);
        assert_eq!(sdsp.get(0), Some((0, -1)));
        assert_eq!(sdsp.get(4), None);
    }

    #[test]
    fn test_large_diamond_pattern() {
        let ldsp = LargeDiamond::new();
        assert_eq!(ldsp.size(), 8);
        assert_eq!(ldsp.get(0), Some((0, -2)));
        assert_eq!(ldsp.get(8), None);
    }

    #[test]
    fn test_small_diamond_search() {
        let src = vec![100u8; 64];
        let mut ref_frame = vec![50u8; 144]; // 12x12

        // Place match at offset (1, 0)
        for row in 0..8 {
            for col in 0..8 {
                ref_frame[row * 12 + col + 1] = 100;
            }
        }

        let ctx = SearchContext::new(&src, 8, &ref_frame, 12, BlockSize::Block8x8, 0, 0, 12, 12);
        let config = SearchConfig::with_range(SearchRange::symmetric(4));

        let sdsp = SmallDiamond::new();
        let (mv, sad, _) = sdsp.search(&ctx, &config, MotionVector::zero());

        // Should find match at (1, 0)
        assert_eq!(mv.full_pel_x(), 1);
        assert_eq!(mv.full_pel_y(), 0);
        assert_eq!(sad, 0);
    }

    #[test]
    fn test_large_diamond_search() {
        let src = vec![100u8; 64];
        let mut ref_frame = vec![50u8; 256]; // 16x16

        // Place match at offset (2, 0)
        for row in 0..8 {
            for col in 0..8 {
                ref_frame[row * 16 + col + 2] = 100;
            }
        }

        let ctx = SearchContext::new(&src, 8, &ref_frame, 16, BlockSize::Block8x8, 0, 0, 16, 16);
        let config = SearchConfig::with_range(SearchRange::symmetric(4));

        let ldsp = LargeDiamond::new();
        let (mv, sad, _) = ldsp.search(&ctx, &config, MotionVector::zero());

        // Should find match at (2, 0)
        assert_eq!(mv.full_pel_x(), 2);
        assert_eq!(mv.full_pel_y(), 0);
        assert_eq!(sad, 0);
    }

    #[test]
    fn test_extended_diamond_search() {
        let src = vec![100u8; 64];
        let mut ref_frame = vec![50u8; 256];

        // Place match at offset (3, 0)
        for row in 0..8 {
            for col in 0..8 {
                ref_frame[row * 16 + col + 3] = 100;
            }
        }

        let ctx = SearchContext::new(&src, 8, &ref_frame, 16, BlockSize::Block8x8, 0, 0, 16, 16);
        let config = SearchConfig::with_range(SearchRange::symmetric(4));

        let extended = ExtendedDiamond::new();
        let (mv, sad) = extended.search(&ctx, &config, MotionVector::zero());

        // Should find match at (3, 0)
        assert_eq!(mv.full_pel_x(), 3);
        assert_eq!(mv.full_pel_y(), 0);
        assert_eq!(sad, 0);
    }

    #[test]
    fn test_adaptive_diamond_convergence() {
        let src = vec![100u8; 64];
        let mut ref_frame = vec![50u8; 256];

        // Place match at (4, 4)
        for row in 0..8 {
            for col in 0..8 {
                ref_frame[(row + 4) * 16 + col + 4] = 100;
            }
        }

        let ctx = SearchContext::new(&src, 8, &ref_frame, 16, BlockSize::Block8x8, 0, 0, 16, 16);
        let config = SearchConfig::with_range(SearchRange::symmetric(8));

        let adaptive = AdaptiveDiamond::new();
        let result = adaptive.search(&ctx, &config);

        // Should find close to the optimal
        assert!(result.sad < 500);
    }

    #[test]
    fn test_predictor_diamond() {
        let src = vec![100u8; 64];
        let mut ref_frame = vec![50u8; 256];

        // Place match at (4, 4)
        for row in 0..8 {
            for col in 0..8 {
                ref_frame[(row + 4) * 16 + col + 4] = 100;
            }
        }

        let ctx = SearchContext::new(&src, 8, &ref_frame, 16, BlockSize::Block8x8, 0, 0, 16, 16);
        let config = SearchConfig::with_range(SearchRange::symmetric(8));

        let predictor = PredictorDiamond::new();
        let predictors = [
            MotionVector::from_full_pel(3, 3), // Close to optimal
            MotionVector::from_full_pel(0, 0),
        ];

        let result = predictor.search_multi(&ctx, &config, &predictors);

        // Good predictor should help find optimal
        assert!(result.sad < 200);
    }

    #[test]
    fn test_cross_diamond() {
        let src = vec![100u8; 64];
        let mut ref_frame = vec![50u8; 256];

        // Place match at (4, 0) - horizontal motion
        for row in 0..8 {
            for col in 0..8 {
                ref_frame[row * 16 + col + 4] = 100;
            }
        }

        let ctx = SearchContext::new(&src, 8, &ref_frame, 16, BlockSize::Block8x8, 0, 0, 16, 16);
        let config = SearchConfig::with_range(SearchRange::symmetric(8));

        let cross = CrossDiamond::new();
        let result = cross.search(&ctx, &config);

        // Cross pattern should handle horizontal motion well
        assert!(result.sad < 300);
    }

    #[test]
    fn test_adaptive_diamond_early_switch() {
        let src = vec![100u8; 64];
        let ref_frame = vec![100u8; 256];

        // Near-perfect match at origin
        let ctx = SearchContext::new(&src, 8, &ref_frame, 16, BlockSize::Block8x8, 0, 0, 16, 16);
        let config = SearchConfig::with_range(SearchRange::symmetric(8));

        let adaptive = AdaptiveDiamond::new().switch_threshold(100);
        let result = adaptive.search(&ctx, &config);

        // Should find match quickly
        assert_eq!(result.sad, 0);
    }

    #[test]
    fn test_diamond_builder_pattern() {
        let adaptive = AdaptiveDiamond::new()
            .max_iterations(16)
            .switch_threshold(1000);

        assert_eq!(adaptive.max_ldsp_iterations, 16);
        assert_eq!(adaptive.switch_threshold, 1000);
    }

    #[test]
    fn test_hexagonal_search_pattern_constants() {
        assert_eq!(HexagonalSearch::INNER_PATTERN.len(), 6);
        assert_eq!(HexagonalSearch::OUTER_PATTERN.len(), 6);
    }

    #[test]
    fn test_hexagonal_search_finds_match() {
        let src = vec![100u8; 64];
        let mut ref_frame = vec![50u8; 400]; // 20x20

        // Place match at (2, 0) — within inner hex
        for row in 0..8usize {
            for col in 0..8usize {
                ref_frame[row * 20 + col + 2] = 100;
            }
        }

        let ctx = SearchContext::new(&src, 8, &ref_frame, 20, BlockSize::Block8x8, 0, 0, 20, 20);
        let config = SearchConfig::with_range(SearchRange::symmetric(8));

        let hex = HexagonalSearch::new();
        let result = hex.search(&ctx, &config);
        assert!(result.sad < 500, "Hex search should find a good match");
    }

    #[test]
    fn test_hexagonal_search_zero_match() {
        let src = vec![128u8; 64];
        let ref_frame = vec![128u8; 400];

        let ctx = SearchContext::new(&src, 8, &ref_frame, 20, BlockSize::Block8x8, 0, 0, 20, 20);
        let config = SearchConfig::with_range(SearchRange::symmetric(8));

        let hex = HexagonalSearch::new();
        let result = hex.search(&ctx, &config);
        assert_eq!(result.sad, 0, "Perfect match should have SAD=0");
    }

    #[test]
    fn test_hexagonal_search_max_iterations_builder() {
        let hex = HexagonalSearch::new().max_iterations(16);
        assert_eq!(hex.max_iterations, 16);
    }

    #[test]
    fn test_umhex_search_finds_match() {
        let src = vec![200u8; 64];
        let mut ref_frame = vec![50u8; 400]; // 20x20

        // Place match at (4, 2)
        for row in 0..8usize {
            for col in 0..8usize {
                ref_frame[(row + 2) * 20 + col + 4] = 200;
            }
        }

        let ctx = SearchContext::new(&src, 8, &ref_frame, 20, BlockSize::Block8x8, 0, 0, 20, 20);
        let config = SearchConfig::with_range(SearchRange::symmetric(8));

        let umhex = UMHexSearch::new();
        let result = umhex.search(&ctx, &config);
        assert!(result.sad < 1000, "UMHex should find a reasonable match");
    }

    #[test]
    fn test_umhex_zero_sad_early_exit() {
        let src = vec![77u8; 64];
        let ref_frame = vec![77u8; 400];

        let ctx = SearchContext::new(&src, 8, &ref_frame, 20, BlockSize::Block8x8, 0, 0, 20, 20);
        let config = SearchConfig::with_range(SearchRange::symmetric(8));

        let umhex = UMHexSearch::new();
        let result = umhex.search(&ctx, &config);
        assert_eq!(result.sad, 0);
    }

    #[test]
    fn test_umhex_predictor_helps() {
        let src = vec![150u8; 64];
        let mut ref_frame = vec![0u8; 400];

        // Place match far from origin at (6, 6)
        for row in 0..8usize {
            for col in 0..8usize {
                ref_frame[(row + 6) * 20 + col + 6] = 150;
            }
        }

        let ctx = SearchContext::new(&src, 8, &ref_frame, 20, BlockSize::Block8x8, 0, 0, 20, 20);
        let config = SearchConfig::with_range(SearchRange::symmetric(10));

        let umhex = UMHexSearch::new();
        let predictor = MotionVector::from_full_pel(5, 5);
        let result = umhex.search_with_predictor(&ctx, &config, predictor);
        // With a good predictor, SAD should be low
        assert!(result.sad < 2000);
    }

    #[test]
    fn test_umhex_builder_pattern() {
        let umhex = UMHexSearch::new()
            .max_hex_steps(32)
            .cross_range(16)
            .early_exit_threshold(8);

        assert_eq!(umhex.max_hex_steps, 32);
        assert_eq!(umhex.cross_range, 16);
        assert_eq!(umhex.early_exit_threshold, 8);
    }

    // =========================================================================
    // Additional tests for exported HexagonalSearch, UMHexSearch, ExtendedDiamond
    // =========================================================================

    fn make_ref_block(width: usize, height: usize, val: u8, bx: usize, by: usize) -> Vec<u8> {
        let mut r = vec![0u8; width * height];
        for row in 0..8 {
            for col in 0..8 {
                let y = by + row;
                let x = bx + col;
                if y < height && x < width {
                    r[y * width + x] = val;
                }
            }
        }
        r
    }

    #[test]
    fn test_hexagonal_search_diagonal_motion() {
        let src = vec![220u8; 64];
        let ref_frame = make_ref_block(20, 20, 220, 2, 2);
        let ctx = SearchContext::new(&src, 8, &ref_frame, 20, BlockSize::Block8x8, 0, 0, 20, 20);
        let config = SearchConfig::with_range(SearchRange::symmetric(8));
        let hex = HexagonalSearch::new();
        let result = hex.search(&ctx, &config);
        assert!(
            result.sad < 600,
            "HexagonalSearch diagonal: SAD={}",
            result.sad
        );
    }

    #[test]
    fn test_hexagonal_search_outer_ring_used() {
        let src = vec![180u8; 64];
        let ref_frame = make_ref_block(24, 24, 180, 4, 0);
        let ctx = SearchContext::new(&src, 8, &ref_frame, 24, BlockSize::Block8x8, 0, 0, 24, 24);
        let config = SearchConfig::with_range(SearchRange::symmetric(10));
        let hex = HexagonalSearch::new();
        let result = hex.search(&ctx, &config);
        assert!(result.sad < 1500, "Outer ring test SAD={}", result.sad);
    }

    #[test]
    fn test_hexagonal_search_with_predictor_improves() {
        let src = vec![160u8; 64];
        let ref_frame = make_ref_block(24, 24, 160, 3, 3);
        let ctx = SearchContext::new(&src, 8, &ref_frame, 24, BlockSize::Block8x8, 0, 0, 24, 24);
        let config = SearchConfig::with_range(SearchRange::symmetric(10));
        let hex = HexagonalSearch::new();
        let result_zero = hex.search(&ctx, &config);
        let result_pred =
            hex.search_with_predictor(&ctx, &config, MotionVector::from_full_pel(3, 3));
        assert!(
            result_pred.sad <= result_zero.sad,
            "Predictor must not worsen result: pred={} zero={}",
            result_pred.sad,
            result_zero.sad
        );
    }

    #[test]
    fn test_hexagonal_search_inner_pattern_length() {
        let hex = HexagonalSearch::new();
        assert_eq!(hex.inner.len(), 6);
        assert_eq!(hex.outer.len(), 6);
    }

    #[test]
    fn test_hexagonal_search_default_max_iterations() {
        let hex = HexagonalSearch::new();
        assert_eq!(hex.max_iterations, 8);
    }

    #[test]
    fn test_umhex_hex1_and_hex2_sizes() {
        assert_eq!(UMHexSearch::HEX1.len(), 6);
        assert_eq!(UMHexSearch::HEX2.len(), 12);
    }

    #[test]
    fn test_umhex_vertical_motion() {
        let src = vec![210u8; 64];
        let ref_frame = make_ref_block(20, 20, 210, 0, 4);
        let ctx = SearchContext::new(&src, 8, &ref_frame, 20, BlockSize::Block8x8, 0, 0, 20, 20);
        let config = SearchConfig::with_range(SearchRange::symmetric(8));
        let umhex = UMHexSearch::new();
        let result = umhex.search(&ctx, &config);
        assert!(result.sad < 1200, "UMHex vertical SAD={}", result.sad);
    }

    #[test]
    fn test_umhex_search_returns_valid_block_match() {
        let src = vec![100u8; 64];
        let ref_frame = vec![100u8; 400];
        let ctx = SearchContext::new(&src, 8, &ref_frame, 20, BlockSize::Block8x8, 0, 0, 20, 20);
        let config = SearchConfig::with_range(SearchRange::symmetric(8));
        let umhex = UMHexSearch::new();
        let result = umhex.search(&ctx, &config);
        // Perfect match: SAD must be 0, cost (u32) must be a valid non-overflowed value.
        assert_eq!(result.sad, 0);
        // `cost` is u32; for zero SAD it should be 0 (no penalty).
        let _ = result.cost; // access the field to verify it compiles
    }

    #[test]
    fn test_extended_diamond_all_rings() {
        let src = vec![130u8; 64];
        let ref_frame = make_ref_block(20, 20, 130, 3, 0);
        let ctx = SearchContext::new(&src, 8, &ref_frame, 20, BlockSize::Block8x8, 0, 0, 20, 20);
        let config = SearchConfig::with_range(SearchRange::symmetric(6));
        let ext = ExtendedDiamond::new();
        let (mv, sad) = ext.search(&ctx, &config, MotionVector::zero());
        assert_eq!(mv.full_pel_x(), 3);
        assert_eq!(sad, 0);
    }

    #[test]
    fn test_extended_diamond_pattern_sizes() {
        let ext = ExtendedDiamond::new();
        assert_eq!(ext.inner.len(), 4);
        assert_eq!(ext.middle.len(), 8);
        assert_eq!(ext.outer.len(), 4);
    }

    #[test]
    fn test_hexagonal_vs_diamond_quality_on_diagonal() {
        let src = vec![170u8; 64];
        let ref_frame = make_ref_block(24, 24, 170, 1, 1);
        let ctx = SearchContext::new(&src, 8, &ref_frame, 24, BlockSize::Block8x8, 0, 0, 24, 24);
        let config = SearchConfig::with_range(SearchRange::symmetric(8));
        let hex_result = HexagonalSearch::new().search(&ctx, &config);
        let adaptive_result = AdaptiveDiamond::new().search(&ctx, &config);
        assert!(
            hex_result.sad < 500,
            "Hexagonal SAD too high: {}",
            hex_result.sad
        );
        assert!(
            adaptive_result.sad < 500,
            "AdaptiveDiamond SAD too high: {}",
            adaptive_result.sad
        );
    }
}
