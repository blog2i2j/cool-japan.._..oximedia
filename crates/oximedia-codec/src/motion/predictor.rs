//! Motion vector prediction for video encoding.
//!
//! This module provides:
//! - Spatial predictors (left, top, top-right, top-left)
//! - Temporal predictors (co-located block from reference)
//! - MVP (Motion Vector Predictor) selection
//! - MV cost calculation using lambda-based RD optimization
//!
//! Good MV prediction reduces the bits needed to encode motion vectors
//! and provides better starting points for motion search.

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::unused_self)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::trivially_copy_pass_by_ref)]

use super::types::{BlockSize, MotionVector, MvCost};

/// Maximum number of MV predictors.
pub const MAX_PREDICTORS: usize = 8;

/// Weight for spatial predictors.
pub const SPATIAL_WEIGHT: u32 = 2;

/// Weight for temporal predictors.
pub const TEMPORAL_WEIGHT: u32 = 1;

/// Position of a neighboring block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NeighborPosition {
    /// Left neighbor.
    Left,
    /// Top neighbor.
    Top,
    /// Top-right neighbor.
    TopRight,
    /// Top-left neighbor.
    TopLeft,
    /// Co-located block in reference frame.
    CoLocated,
    /// Below-left neighbor.
    BelowLeft,
    /// Median of neighbors.
    Median,
}

/// Information about a neighboring block.
#[derive(Clone, Copy, Debug, Default)]
pub struct NeighborInfo {
    /// Motion vector.
    pub mv: MotionVector,
    /// Reference frame index.
    pub ref_idx: i8,
    /// Is this neighbor available?
    pub available: bool,
    /// Is this an inter-predicted block?
    pub is_inter: bool,
}

impl NeighborInfo {
    /// Creates unavailable neighbor info.
    #[must_use]
    pub const fn unavailable() -> Self {
        Self {
            mv: MotionVector::zero(),
            ref_idx: -1,
            available: false,
            is_inter: false,
        }
    }

    /// Creates neighbor info with MV.
    #[must_use]
    pub const fn with_mv(mv: MotionVector, ref_idx: i8) -> Self {
        Self {
            mv,
            ref_idx,
            available: true,
            is_inter: true,
        }
    }

    /// Creates intra neighbor (no MV).
    #[must_use]
    pub const fn intra() -> Self {
        Self {
            mv: MotionVector::zero(),
            ref_idx: -1,
            available: true,
            is_inter: false,
        }
    }
}

/// Context for motion vector prediction.
#[derive(Clone, Debug, Default)]
pub struct MvPredContext {
    /// Left neighbor.
    pub left: NeighborInfo,
    /// Top neighbor.
    pub top: NeighborInfo,
    /// Top-right neighbor.
    pub top_right: NeighborInfo,
    /// Top-left neighbor.
    pub top_left: NeighborInfo,
    /// Co-located block in reference.
    pub co_located: NeighborInfo,
    /// Current reference frame index.
    pub ref_idx: i8,
    /// Block position (x in 4x4 units).
    pub mi_col: usize,
    /// Block position (y in 4x4 units).
    pub mi_row: usize,
    /// Block size.
    pub block_size: BlockSize,
}

impl MvPredContext {
    /// Creates a new prediction context.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            left: NeighborInfo::unavailable(),
            top: NeighborInfo::unavailable(),
            top_right: NeighborInfo::unavailable(),
            top_left: NeighborInfo::unavailable(),
            co_located: NeighborInfo::unavailable(),
            ref_idx: 0,
            mi_col: 0,
            mi_row: 0,
            block_size: BlockSize::Block8x8,
        }
    }

    /// Sets the block position.
    #[must_use]
    pub const fn at_position(mut self, mi_row: usize, mi_col: usize) -> Self {
        self.mi_row = mi_row;
        self.mi_col = mi_col;
        self
    }

    /// Sets the block size.
    #[must_use]
    pub const fn with_size(mut self, size: BlockSize) -> Self {
        self.block_size = size;
        self
    }

    /// Sets the reference frame index.
    #[must_use]
    pub const fn with_ref(mut self, ref_idx: i8) -> Self {
        self.ref_idx = ref_idx;
        self
    }

    /// Sets the left neighbor.
    #[must_use]
    pub const fn with_left(mut self, info: NeighborInfo) -> Self {
        self.left = info;
        self
    }

    /// Sets the top neighbor.
    #[must_use]
    pub const fn with_top(mut self, info: NeighborInfo) -> Self {
        self.top = info;
        self
    }

    /// Sets the top-right neighbor.
    #[must_use]
    pub const fn with_top_right(mut self, info: NeighborInfo) -> Self {
        self.top_right = info;
        self
    }

    /// Sets the top-left neighbor.
    #[must_use]
    pub const fn with_top_left(mut self, info: NeighborInfo) -> Self {
        self.top_left = info;
        self
    }

    /// Sets the co-located block.
    #[must_use]
    pub const fn with_co_located(mut self, info: NeighborInfo) -> Self {
        self.co_located = info;
        self
    }
}

/// MV predictor candidate.
#[derive(Clone, Copy, Debug)]
pub struct MvCandidate {
    /// Predicted motion vector.
    pub mv: MotionVector,
    /// Weight/priority of this candidate.
    pub weight: u32,
    /// Source position.
    pub source: NeighborPosition,
}

impl MvCandidate {
    /// Creates a new MV candidate.
    #[must_use]
    pub const fn new(mv: MotionVector, weight: u32, source: NeighborPosition) -> Self {
        Self { mv, weight, source }
    }

    /// Creates a zero MV candidate.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            mv: MotionVector::zero(),
            weight: 0,
            source: NeighborPosition::Median,
        }
    }
}

/// Motion vector predictor list.
#[derive(Clone, Debug)]
pub struct MvPredictorList {
    /// Candidate predictors.
    candidates: [MvCandidate; MAX_PREDICTORS],
    /// Number of valid candidates.
    count: usize,
}

impl Default for MvPredictorList {
    fn default() -> Self {
        Self::new()
    }
}

impl MvPredictorList {
    /// Creates a new empty predictor list.
    #[must_use]
    pub fn new() -> Self {
        Self {
            candidates: [MvCandidate::zero(); MAX_PREDICTORS],
            count: 0,
        }
    }

    /// Adds a candidate to the list.
    pub fn add(&mut self, candidate: MvCandidate) {
        if self.count < MAX_PREDICTORS {
            // Check for duplicates
            for i in 0..self.count {
                if self.candidates[i].mv == candidate.mv {
                    // Update weight if higher
                    if candidate.weight > self.candidates[i].weight {
                        self.candidates[i].weight = candidate.weight;
                    }
                    return;
                }
            }
            self.candidates[self.count] = candidate;
            self.count += 1;
        }
    }

    /// Adds a candidate from neighbor info.
    pub fn add_from_neighbor(&mut self, info: &NeighborInfo, source: NeighborPosition) {
        if info.available && info.is_inter {
            let weight = match source {
                NeighborPosition::Left | NeighborPosition::Top | NeighborPosition::TopRight => {
                    SPATIAL_WEIGHT
                }
                NeighborPosition::CoLocated => TEMPORAL_WEIGHT,
                _ => 1,
            };
            self.add(MvCandidate::new(info.mv, weight, source));
        }
    }

    /// Sorts candidates by weight (descending).
    pub fn sort_by_weight(&mut self) {
        // Simple insertion sort for small array
        for i in 1..self.count {
            let key = self.candidates[i];
            let mut j = i;
            while j > 0 && self.candidates[j - 1].weight < key.weight {
                self.candidates[j] = self.candidates[j - 1];
                j -= 1;
            }
            self.candidates[j] = key;
        }
    }

    /// Returns the number of candidates.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Returns true if empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Gets a candidate by index.
    #[must_use]
    pub const fn get(&self, index: usize) -> Option<&MvCandidate> {
        if index < self.count {
            Some(&self.candidates[index])
        } else {
            None
        }
    }

    /// Returns the best (first) predictor.
    #[must_use]
    pub fn best(&self) -> MotionVector {
        if self.count > 0 {
            self.candidates[0].mv
        } else {
            MotionVector::zero()
        }
    }

    /// Returns all predictors as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[MvCandidate] {
        &self.candidates[..self.count]
    }

    /// Extracts motion vectors only.
    pub fn motion_vectors(&self) -> Vec<MotionVector> {
        self.candidates[..self.count].iter().map(|c| c.mv).collect()
    }
}

/// Spatial MV predictor calculator.
#[derive(Clone, Copy, Debug, Default)]
pub struct SpatialPredictor;

impl SpatialPredictor {
    /// Creates a new spatial predictor.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Gets the left neighbor MV.
    #[must_use]
    pub fn get_left(ctx: &MvPredContext) -> Option<MotionVector> {
        if ctx.left.available && ctx.left.is_inter {
            Some(ctx.left.mv)
        } else {
            None
        }
    }

    /// Gets the top neighbor MV.
    #[must_use]
    pub fn get_top(ctx: &MvPredContext) -> Option<MotionVector> {
        if ctx.top.available && ctx.top.is_inter {
            Some(ctx.top.mv)
        } else {
            None
        }
    }

    /// Gets the top-right neighbor MV.
    #[must_use]
    pub fn get_top_right(ctx: &MvPredContext) -> Option<MotionVector> {
        if ctx.top_right.available && ctx.top_right.is_inter {
            Some(ctx.top_right.mv)
        } else {
            None
        }
    }

    /// Gets the top-left neighbor MV.
    #[must_use]
    pub fn get_top_left(ctx: &MvPredContext) -> Option<MotionVector> {
        if ctx.top_left.available && ctx.top_left.is_inter {
            Some(ctx.top_left.mv)
        } else {
            None
        }
    }

    /// Calculates the median predictor from three MVs.
    #[must_use]
    pub fn median(a: MotionVector, b: MotionVector, c: MotionVector) -> MotionVector {
        // Component-wise median
        let dx = Self::median_of_3(a.dx, b.dx, c.dx);
        let dy = Self::median_of_3(a.dy, b.dy, c.dy);
        MotionVector::new(dx, dy)
    }

    /// Median of three values.
    #[must_use]
    fn median_of_3(a: i32, b: i32, c: i32) -> i32 {
        a.max(b.min(c)).min(b.max(c))
    }

    /// Calculates the median MVP from spatial neighbors.
    #[must_use]
    pub fn calculate_median(ctx: &MvPredContext) -> MotionVector {
        let left = Self::get_left(ctx).unwrap_or_else(MotionVector::zero);
        let top = Self::get_top(ctx).unwrap_or_else(MotionVector::zero);
        let top_right = Self::get_top_right(ctx)
            .or_else(|| Self::get_top_left(ctx))
            .unwrap_or_else(MotionVector::zero);

        Self::median(left, top, top_right)
    }

    /// Builds spatial predictor list.
    pub fn build_predictors(ctx: &MvPredContext, list: &mut MvPredictorList) {
        list.add_from_neighbor(&ctx.left, NeighborPosition::Left);
        list.add_from_neighbor(&ctx.top, NeighborPosition::Top);
        list.add_from_neighbor(&ctx.top_right, NeighborPosition::TopRight);
        list.add_from_neighbor(&ctx.top_left, NeighborPosition::TopLeft);

        // Add median if we have multiple neighbors
        let mut neighbor_count = 0;
        if ctx.left.available && ctx.left.is_inter {
            neighbor_count += 1;
        }
        if ctx.top.available && ctx.top.is_inter {
            neighbor_count += 1;
        }
        if ctx.top_right.available && ctx.top_right.is_inter {
            neighbor_count += 1;
        }

        if neighbor_count >= 2 {
            let median = Self::calculate_median(ctx);
            list.add(MvCandidate::new(
                median,
                SPATIAL_WEIGHT + 1,
                NeighborPosition::Median,
            ));
        }
    }
}

/// Temporal MV predictor calculator.
#[derive(Clone, Copy, Debug, Default)]
pub struct TemporalPredictor;

impl TemporalPredictor {
    /// Creates a new temporal predictor.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Gets the co-located MV from reference frame.
    #[must_use]
    pub fn get_co_located(ctx: &MvPredContext) -> Option<MotionVector> {
        if ctx.co_located.available && ctx.co_located.is_inter {
            Some(ctx.co_located.mv)
        } else {
            None
        }
    }

    /// Scales MV for different temporal distances.
    ///
    /// If the co-located block references a frame at distance `src_dist`,
    /// and we want to predict for target at distance `dst_dist`,
    /// scale the MV proportionally.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn scale_mv(mv: MotionVector, src_dist: i32, dst_dist: i32) -> MotionVector {
        if src_dist == 0 || src_dist == dst_dist {
            return mv;
        }

        let scale_x = (i64::from(mv.dx) * i64::from(dst_dist)) / i64::from(src_dist);
        let scale_y = (i64::from(mv.dy) * i64::from(dst_dist)) / i64::from(src_dist);

        MotionVector::new(scale_x as i32, scale_y as i32)
    }

    /// Builds temporal predictor.
    pub fn build_predictors(ctx: &MvPredContext, list: &mut MvPredictorList) {
        list.add_from_neighbor(&ctx.co_located, NeighborPosition::CoLocated);
    }
}

/// Combined MV predictor that uses both spatial and temporal information.
#[derive(Clone, Debug, Default)]
pub struct MvPredictor {
    /// Spatial predictor.
    spatial: SpatialPredictor,
    /// Temporal predictor.
    temporal: TemporalPredictor,
    /// Predictor list.
    predictors: MvPredictorList,
}

impl MvPredictor {
    /// Creates a new MV predictor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            spatial: SpatialPredictor::new(),
            temporal: TemporalPredictor::new(),
            predictors: MvPredictorList::new(),
        }
    }

    /// Builds all predictors from context.
    pub fn build(&mut self, ctx: &MvPredContext) {
        self.predictors = MvPredictorList::new();

        // Always add zero MV as fallback
        self.predictors.add(MvCandidate::new(
            MotionVector::zero(),
            1,
            NeighborPosition::Median,
        ));

        // Add spatial predictors
        SpatialPredictor::build_predictors(ctx, &mut self.predictors);

        // Add temporal predictors
        TemporalPredictor::build_predictors(ctx, &mut self.predictors);

        // Sort by weight
        self.predictors.sort_by_weight();
    }

    /// Returns the best MVP.
    #[must_use]
    pub fn best_mvp(&self) -> MotionVector {
        self.predictors.best()
    }

    /// Returns all predictors.
    #[must_use]
    pub fn all_predictors(&self) -> &MvPredictorList {
        &self.predictors
    }

    /// Returns predictors as motion vectors.
    pub fn motion_vectors(&self) -> Vec<MotionVector> {
        self.predictors.motion_vectors()
    }
}

/// MV cost calculator for rate-distortion optimization.
#[derive(Clone, Debug)]
pub struct MvCostCalculator {
    /// Lambda for RD tradeoff.
    lambda: f32,
    /// MV cost tables (optional precomputed).
    cost_table: Option<Vec<u32>>,
}

impl Default for MvCostCalculator {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl MvCostCalculator {
    /// Creates a new cost calculator.
    #[must_use]
    pub const fn new(lambda: f32) -> Self {
        Self {
            lambda,
            cost_table: None,
        }
    }

    /// Builds cost table for fast lookup.
    pub fn build_cost_table(&mut self, max_mv: i32) {
        let size = (2 * max_mv + 1) as usize;
        let mut table = vec![0u32; size];

        for i in 0..size {
            let mv_component = i as i32 - max_mv;
            table[i] = self.component_cost(mv_component);
        }

        self.cost_table = Some(table);
    }

    /// Calculates cost for a single MV component.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn component_cost(&self, value: i32) -> u32 {
        if value == 0 {
            return (self.lambda * 1.0) as u32;
        }

        let abs_val = value.unsigned_abs();
        // Approximate bit cost: log2(abs) * 2 + constant
        let log2 = 32 - abs_val.leading_zeros();
        let bits = log2 * 2 + 2;
        ((bits as f32) * self.lambda) as u32
    }

    /// Calculates the bit cost of an MV differential.
    #[must_use]
    pub fn cost(&self, mv: &MotionVector, mvp: &MotionVector) -> u32 {
        let diff = *mv - *mvp;
        self.component_cost(diff.dx) + self.component_cost(diff.dy)
    }

    /// Calculates full RD cost (distortion + rate).
    #[must_use]
    pub fn rd_cost(&self, mv: &MotionVector, mvp: &MotionVector, distortion: u32) -> u32 {
        distortion.saturating_add(self.cost(mv, mvp))
    }

    /// Creates an MvCost instance from this calculator.
    #[must_use]
    pub fn to_mv_cost(&self, mvp: MotionVector) -> MvCost {
        MvCost::with_ref_mv(self.lambda, mvp)
    }
}

/// MVP selection modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MvpMode {
    /// Use spatial median.
    #[default]
    Median,
    /// Use left neighbor.
    Left,
    /// Use top neighbor.
    Top,
    /// Use co-located temporal.
    Temporal,
    /// Use zero MV.
    Zero,
}

impl MvpMode {
    /// Gets the MVP for this mode.
    #[must_use]
    pub fn get_mvp(&self, ctx: &MvPredContext) -> MotionVector {
        match self {
            Self::Median => SpatialPredictor::calculate_median(ctx),
            Self::Left => SpatialPredictor::get_left(ctx).unwrap_or_else(MotionVector::zero),
            Self::Top => SpatialPredictor::get_top(ctx).unwrap_or_else(MotionVector::zero),
            Self::Temporal => {
                TemporalPredictor::get_co_located(ctx).unwrap_or_else(MotionVector::zero)
            }
            Self::Zero => MotionVector::zero(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neighbor_info_unavailable() {
        let info = NeighborInfo::unavailable();
        assert!(!info.available);
        assert!(!info.is_inter);
    }

    #[test]
    fn test_neighbor_info_with_mv() {
        let mv = MotionVector::new(10, 20);
        let info = NeighborInfo::with_mv(mv, 0);
        assert!(info.available);
        assert!(info.is_inter);
        assert_eq!(info.mv.dx, 10);
        assert_eq!(info.mv.dy, 20);
    }

    #[test]
    fn test_mv_pred_context_builder() {
        let left = NeighborInfo::with_mv(MotionVector::new(5, 5), 0);
        let ctx = MvPredContext::new()
            .at_position(10, 20)
            .with_size(BlockSize::Block16x16)
            .with_ref(0)
            .with_left(left);

        assert_eq!(ctx.mi_row, 10);
        assert_eq!(ctx.mi_col, 20);
        assert_eq!(ctx.block_size, BlockSize::Block16x16);
        assert!(ctx.left.available);
    }

    #[test]
    fn test_mv_candidate() {
        let mv = MotionVector::new(10, 20);
        let candidate = MvCandidate::new(mv, 5, NeighborPosition::Left);

        assert_eq!(candidate.mv.dx, 10);
        assert_eq!(candidate.weight, 5);
        assert_eq!(candidate.source, NeighborPosition::Left);
    }

    #[test]
    fn test_mv_predictor_list_add() {
        let mut list = MvPredictorList::new();

        list.add(MvCandidate::new(
            MotionVector::new(10, 20),
            2,
            NeighborPosition::Left,
        ));
        list.add(MvCandidate::new(
            MotionVector::new(30, 40),
            3,
            NeighborPosition::Top,
        ));

        assert_eq!(list.len(), 2);
        assert_eq!(list.get(0).expect("get should return value").mv.dx, 10);
        assert_eq!(list.get(1).expect("get should return value").mv.dx, 30);
    }

    #[test]
    fn test_mv_predictor_list_dedup() {
        let mut list = MvPredictorList::new();

        list.add(MvCandidate::new(
            MotionVector::new(10, 20),
            2,
            NeighborPosition::Left,
        ));
        list.add(MvCandidate::new(
            MotionVector::new(10, 20),
            3,
            NeighborPosition::Top,
        ));

        // Should merge duplicates
        assert_eq!(list.len(), 1);
        assert_eq!(list.get(0).expect("get should return value").weight, 3); // Higher weight kept
    }

    #[test]
    fn test_mv_predictor_list_sort() {
        let mut list = MvPredictorList::new();

        list.add(MvCandidate::new(
            MotionVector::new(10, 20),
            1,
            NeighborPosition::Left,
        ));
        list.add(MvCandidate::new(
            MotionVector::new(30, 40),
            3,
            NeighborPosition::Top,
        ));
        list.add(MvCandidate::new(
            MotionVector::new(50, 60),
            2,
            NeighborPosition::TopRight,
        ));

        list.sort_by_weight();

        assert_eq!(list.get(0).expect("get should return value").weight, 3);
        assert_eq!(list.get(1).expect("get should return value").weight, 2);
        assert_eq!(list.get(2).expect("get should return value").weight, 1);
    }

    #[test]
    fn test_spatial_predictor_median() {
        let a = MotionVector::new(10, 30);
        let b = MotionVector::new(20, 20);
        let c = MotionVector::new(30, 10);

        let median = SpatialPredictor::median(a, b, c);

        assert_eq!(median.dx, 20); // Median of 10, 20, 30
        assert_eq!(median.dy, 20); // Median of 30, 20, 10
    }

    #[test]
    fn test_spatial_predictor_build() {
        let ctx = MvPredContext::new()
            .with_left(NeighborInfo::with_mv(MotionVector::new(10, 10), 0))
            .with_top(NeighborInfo::with_mv(MotionVector::new(20, 20), 0));

        let mut list = MvPredictorList::new();
        SpatialPredictor::build_predictors(&ctx, &mut list);

        assert!(list.len() >= 2);
    }

    #[test]
    fn test_temporal_predictor_scale() {
        let mv = MotionVector::new(100, 200);

        // Same distance - no change
        let same = TemporalPredictor::scale_mv(mv, 1, 1);
        assert_eq!(same.dx, 100);
        assert_eq!(same.dy, 200);

        // Double distance
        let doubled = TemporalPredictor::scale_mv(mv, 1, 2);
        assert_eq!(doubled.dx, 200);
        assert_eq!(doubled.dy, 400);

        // Half distance
        let halved = TemporalPredictor::scale_mv(mv, 2, 1);
        assert_eq!(halved.dx, 50);
        assert_eq!(halved.dy, 100);
    }

    #[test]
    fn test_mv_predictor_build() {
        let ctx = MvPredContext::new()
            .with_left(NeighborInfo::with_mv(MotionVector::new(10, 10), 0))
            .with_top(NeighborInfo::with_mv(MotionVector::new(20, 20), 0))
            .with_co_located(NeighborInfo::with_mv(MotionVector::new(15, 15), 0));

        let mut predictor = MvPredictor::new();
        predictor.build(&ctx);

        // Should have multiple predictors
        assert!(predictor.all_predictors().len() >= 3);

        // Best MVP should be calculated
        let mvp = predictor.best_mvp();
        assert!(mvp.dx != 0 || mvp.dy != 0 || predictor.all_predictors().len() == 1);
    }

    #[test]
    fn test_mv_cost_calculator() {
        let calc = MvCostCalculator::new(1.0);
        let mv = MotionVector::new(16, 16);
        let mvp = MotionVector::zero();

        let cost = calc.cost(&mv, &mvp);
        assert!(cost > 0);

        // Same MV as predictor should have lower cost
        let same_cost = calc.cost(&mvp, &mvp);
        assert!(same_cost < cost);
    }

    #[test]
    fn test_mv_cost_rd() {
        let calc = MvCostCalculator::new(1.0);
        let mv = MotionVector::new(16, 16);
        let mvp = MotionVector::zero();

        let rd = calc.rd_cost(&mv, &mvp, 100);
        assert!(rd > 100); // Should include MV cost
    }

    #[test]
    fn test_mvp_mode() {
        let ctx =
            MvPredContext::new().with_left(NeighborInfo::with_mv(MotionVector::new(10, 10), 0));

        assert_eq!(MvpMode::Left.get_mvp(&ctx).dx, 10);
        assert_eq!(MvpMode::Zero.get_mvp(&ctx).dx, 0);
    }

    #[test]
    fn test_motion_vectors_extraction() {
        let mut list = MvPredictorList::new();
        list.add(MvCandidate::new(
            MotionVector::new(10, 20),
            2,
            NeighborPosition::Left,
        ));
        list.add(MvCandidate::new(
            MotionVector::new(30, 40),
            1,
            NeighborPosition::Top,
        ));

        let mvs = list.motion_vectors();
        assert_eq!(mvs.len(), 2);
        assert_eq!(mvs[0].dx, 10);
        assert_eq!(mvs[1].dx, 30);
    }
}
