//! Block partitioning decisions for motion estimation.
//!
//! This module provides:
//! - Partition decision structures
//! - Split decisions (16x16 -> 8x8 -> 4x4)
//! - Skip detection for direct mode
//! - Merge candidate generation
//!
//! Efficient partitioning is crucial for balancing compression
//! efficiency with encoding complexity.

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::match_same_arms)]

use super::types::{BlockMatch, BlockSize, MotionVector, MvCost};

/// Threshold for skip mode SAD.
pub const SKIP_THRESHOLD: u32 = 64;

/// Threshold for considering a partition split.
pub const SPLIT_THRESHOLD_RATIO: f32 = 0.8;

/// Maximum merge candidates.
pub const MAX_MERGE_CANDIDATES: usize = 5;

/// Partition type for a block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum PartitionType {
    /// No partition (use current block size).
    #[default]
    None = 0,
    /// Horizontal split (top/bottom halves).
    HorizontalSplit = 1,
    /// Vertical split (left/right halves).
    VerticalSplit = 2,
    /// Quad split (4 equal quadrants).
    Split = 3,
    /// Horizontal split with top half smaller.
    HorizontalA = 4,
    /// Horizontal split with bottom half smaller.
    HorizontalB = 5,
    /// Vertical split with left half smaller.
    VerticalA = 6,
    /// Vertical split with right half smaller.
    VerticalB = 7,
    /// Horizontal 4-way split.
    Horizontal4 = 8,
    /// Vertical 4-way split.
    Vertical4 = 9,
}

impl PartitionType {
    /// Returns the number of sub-partitions.
    #[must_use]
    pub const fn num_parts(&self) -> usize {
        match self {
            Self::None => 1,
            Self::HorizontalSplit | Self::VerticalSplit => 2,
            Self::Split
            | Self::HorizontalA
            | Self::HorizontalB
            | Self::VerticalA
            | Self::VerticalB => 4,
            Self::Horizontal4 | Self::Vertical4 => 4,
        }
    }

    /// Returns true if this is a split partition.
    #[must_use]
    pub const fn is_split(&self) -> bool {
        !matches!(self, Self::None)
    }
}

/// Mode for inter prediction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum InterMode {
    /// Skip mode (use predicted MV directly).
    Skip = 0,
    /// Merge mode (copy MV from neighbor).
    Merge = 1,
    /// New MV mode (search for best MV).
    #[default]
    NewMv = 2,
    /// Nearest MV mode (use nearest neighbor MV).
    NearestMv = 3,
    /// Near MV mode (use second-nearest neighbor MV).
    NearMv = 4,
    /// Zero MV mode.
    ZeroMv = 5,
}

impl InterMode {
    /// Returns true if this mode requires MV search.
    #[must_use]
    pub const fn requires_search(&self) -> bool {
        matches!(self, Self::NewMv)
    }

    /// Returns true if this mode uses a predictor MV.
    #[must_use]
    pub const fn uses_predictor(&self) -> bool {
        matches!(self, Self::NearestMv | Self::NearMv | Self::ZeroMv)
    }
}

/// Decision for a single partition.
#[derive(Clone, Debug)]
pub struct PartitionDecision {
    /// Block size for this partition.
    pub block_size: BlockSize,
    /// Partition type.
    pub partition_type: PartitionType,
    /// Inter prediction mode.
    pub mode: InterMode,
    /// Motion vector.
    pub mv: MotionVector,
    /// Reference frame index.
    pub ref_idx: i8,
    /// Rate-distortion cost.
    pub cost: u32,
    /// Distortion (SAD/SATD).
    pub distortion: u32,
    /// Estimated bits for this partition.
    pub bits: u32,
    /// Is this a skip block?
    pub is_skip: bool,
    /// Merge candidate index (if merge mode).
    pub merge_idx: u8,
}

impl Default for PartitionDecision {
    fn default() -> Self {
        Self::new()
    }
}

impl PartitionDecision {
    /// Creates a new default decision.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            block_size: BlockSize::Block8x8,
            partition_type: PartitionType::None,
            mode: InterMode::NewMv,
            mv: MotionVector::zero(),
            ref_idx: 0,
            cost: u32::MAX,
            distortion: u32::MAX,
            bits: 0,
            is_skip: false,
            merge_idx: 0,
        }
    }

    /// Creates a skip decision.
    #[must_use]
    pub const fn skip(block_size: BlockSize, mv: MotionVector, distortion: u32) -> Self {
        Self {
            block_size,
            partition_type: PartitionType::None,
            mode: InterMode::Skip,
            mv,
            ref_idx: 0,
            cost: distortion,
            distortion,
            bits: 0,
            is_skip: true,
            merge_idx: 0,
        }
    }

    /// Creates a decision from block match result.
    #[must_use]
    pub const fn from_match(block_size: BlockSize, block_match: &BlockMatch) -> Self {
        Self {
            block_size,
            partition_type: PartitionType::None,
            mode: InterMode::NewMv,
            mv: block_match.mv,
            ref_idx: 0,
            cost: block_match.cost,
            distortion: block_match.sad,
            bits: 0,
            is_skip: false,
            merge_idx: 0,
        }
    }

    /// Checks if this decision is better than another.
    #[must_use]
    pub const fn is_better_than(&self, other: &Self) -> bool {
        self.cost < other.cost
    }

    /// Updates with a better decision.
    pub fn update_if_better(&mut self, other: &Self) {
        if other.is_better_than(self) {
            *self = other.clone();
        }
    }
}

/// Split decision result for recursive partitioning.
#[derive(Clone, Debug)]
pub struct SplitDecision {
    /// Should we split this block?
    pub should_split: bool,
    /// Cost of the unsplit block.
    pub unsplit_cost: u32,
    /// Cost of the split blocks (sum).
    pub split_cost: u32,
    /// Child decisions (if split).
    pub children: Vec<PartitionDecision>,
}

impl Default for SplitDecision {
    fn default() -> Self {
        Self::new()
    }
}

impl SplitDecision {
    /// Creates a new split decision.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            should_split: false,
            unsplit_cost: u32::MAX,
            split_cost: u32::MAX,
            children: Vec::new(),
        }
    }

    /// Creates a decision to not split.
    #[must_use]
    pub const fn no_split(cost: u32) -> Self {
        Self {
            should_split: false,
            unsplit_cost: cost,
            split_cost: u32::MAX,
            children: Vec::new(),
        }
    }

    /// Creates a decision to split.
    #[must_use]
    pub fn split(unsplit_cost: u32, split_cost: u32, children: Vec<PartitionDecision>) -> Self {
        Self {
            should_split: split_cost < unsplit_cost,
            unsplit_cost,
            split_cost,
            children,
        }
    }
}

/// Skip detection for inter prediction.
#[derive(Clone, Debug, Default)]
pub struct SkipDetector {
    /// SAD threshold for skip.
    threshold: u32,
    /// MV cost weight.
    mv_weight: f32,
}

impl SkipDetector {
    /// Creates a new skip detector.
    #[must_use]
    pub const fn new(threshold: u32) -> Self {
        Self {
            threshold,
            mv_weight: 1.0,
        }
    }

    /// Sets the MV cost weight.
    #[must_use]
    pub const fn with_mv_weight(mut self, weight: f32) -> Self {
        self.mv_weight = weight;
        self
    }

    /// Checks if a block can be skipped.
    #[must_use]
    pub fn can_skip(&self, block_match: &BlockMatch, predicted_mv: &MotionVector) -> bool {
        // Skip if SAD is low enough
        if block_match.sad > self.threshold {
            return false;
        }

        // Skip if MV matches predicted MV well
        let mv_diff = block_match.mv - *predicted_mv;
        let mv_dist = mv_diff.l1_norm();

        // Allow skip if MV is close to prediction
        mv_dist < 8 // Within 1 full pixel
    }

    /// Evaluates skip mode cost.
    #[must_use]
    pub fn evaluate_skip(
        &self,
        block_match: &BlockMatch,
        predicted_mv: &MotionVector,
        mv_cost: &MvCost,
    ) -> Option<PartitionDecision> {
        if !self.can_skip(block_match, predicted_mv) {
            return None;
        }

        // Calculate skip mode cost (no MV bits needed)
        let skip_cost = block_match.sad + 1; // +1 for skip flag

        // Compare with regular mode cost
        let regular_cost = mv_cost.rd_cost(&block_match.mv, block_match.sad);

        if skip_cost < regular_cost {
            Some(PartitionDecision::skip(
                BlockSize::Block8x8,
                *predicted_mv,
                block_match.sad,
            ))
        } else {
            None
        }
    }
}

/// Merge candidate for merge mode.
#[derive(Clone, Copy, Debug)]
pub struct MergeCandidate {
    /// Motion vector.
    pub mv: MotionVector,
    /// Reference frame index.
    pub ref_idx: i8,
    /// Source of this candidate.
    pub source: MergeSource,
}

impl MergeCandidate {
    /// Creates a new merge candidate.
    #[must_use]
    pub const fn new(mv: MotionVector, ref_idx: i8, source: MergeSource) -> Self {
        Self {
            mv,
            ref_idx,
            source,
        }
    }

    /// Creates a zero MV candidate.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            mv: MotionVector::zero(),
            ref_idx: 0,
            source: MergeSource::Zero,
        }
    }
}

/// Source of merge candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MergeSource {
    /// Left neighbor.
    Left,
    /// Top neighbor.
    Top,
    /// Top-right neighbor.
    TopRight,
    /// Top-left neighbor.
    TopLeft,
    /// Co-located temporal.
    CoLocated,
    /// Zero MV.
    Zero,
}

/// Merge candidate list.
#[derive(Clone, Debug)]
pub struct MergeCandidateList {
    /// Candidates.
    candidates: [MergeCandidate; MAX_MERGE_CANDIDATES],
    /// Number of valid candidates.
    count: usize,
}

impl Default for MergeCandidateList {
    fn default() -> Self {
        Self::new()
    }
}

impl MergeCandidateList {
    /// Creates a new empty list.
    #[must_use]
    pub fn new() -> Self {
        Self {
            candidates: [MergeCandidate::zero(); MAX_MERGE_CANDIDATES],
            count: 0,
        }
    }

    /// Adds a candidate.
    pub fn add(&mut self, candidate: MergeCandidate) {
        if self.count >= MAX_MERGE_CANDIDATES {
            return;
        }

        // Check for duplicates
        for i in 0..self.count {
            if self.candidates[i].mv == candidate.mv
                && self.candidates[i].ref_idx == candidate.ref_idx
            {
                return;
            }
        }

        self.candidates[self.count] = candidate;
        self.count += 1;
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
    pub const fn get(&self, index: usize) -> Option<&MergeCandidate> {
        if index < self.count {
            Some(&self.candidates[index])
        } else {
            None
        }
    }

    /// Returns candidates as slice.
    #[must_use]
    pub fn as_slice(&self) -> &[MergeCandidate] {
        &self.candidates[..self.count]
    }
}

/// Partition context for decision making.
#[derive(Clone, Debug, Default)]
pub struct PartitionContext {
    /// Block position (x in pixels).
    pub x: usize,
    /// Block position (y in pixels).
    pub y: usize,
    /// Frame width.
    pub frame_width: usize,
    /// Frame height.
    pub frame_height: usize,
    /// Maximum block size allowed.
    pub max_block_size: BlockSize,
    /// Minimum block size allowed.
    pub min_block_size: BlockSize,
    /// Lambda for RD.
    pub lambda: f32,
}

impl PartitionContext {
    /// Creates a new partition context.
    #[must_use]
    pub const fn new(frame_width: usize, frame_height: usize) -> Self {
        Self {
            x: 0,
            y: 0,
            frame_width,
            frame_height,
            max_block_size: BlockSize::Block64x64,
            min_block_size: BlockSize::Block4x4,
            lambda: 1.0,
        }
    }

    /// Sets the block position.
    #[must_use]
    pub const fn at(mut self, x: usize, y: usize) -> Self {
        self.x = x;
        self.y = y;
        self
    }

    /// Sets the block size limits.
    #[must_use]
    pub const fn with_size_range(mut self, min: BlockSize, max: BlockSize) -> Self {
        self.min_block_size = min;
        self.max_block_size = max;
        self
    }

    /// Checks if a block size fits within frame bounds.
    #[must_use]
    pub fn can_use_size(&self, size: BlockSize) -> bool {
        self.x + size.width() <= self.frame_width && self.y + size.height() <= self.frame_height
    }

    /// Returns the child block size for quad split.
    #[must_use]
    pub const fn child_size(size: BlockSize) -> Option<BlockSize> {
        match size {
            BlockSize::Block128x128 => Some(BlockSize::Block64x64),
            BlockSize::Block64x64 => Some(BlockSize::Block32x32),
            BlockSize::Block32x32 => Some(BlockSize::Block16x16),
            BlockSize::Block16x16 => Some(BlockSize::Block8x8),
            BlockSize::Block8x8 => Some(BlockSize::Block4x4),
            _ => None,
        }
    }
}

/// Partition decision maker.
#[derive(Clone, Debug, Default)]
pub struct PartitionDecider {
    /// Skip detector.
    skip_detector: SkipDetector,
    /// Cost ratio threshold for splitting.
    split_threshold: f32,
}

impl PartitionDecider {
    /// Creates a new partition decider.
    #[must_use]
    pub fn new() -> Self {
        Self {
            skip_detector: SkipDetector::new(SKIP_THRESHOLD),
            split_threshold: SPLIT_THRESHOLD_RATIO,
        }
    }

    /// Sets the split threshold.
    #[must_use]
    pub const fn with_split_threshold(mut self, threshold: f32) -> Self {
        self.split_threshold = threshold;
        self
    }

    /// Decides whether to split a block.
    pub fn decide_split(
        &self,
        unsplit_result: &PartitionDecision,
        split_results: &[PartitionDecision],
        ctx: &PartitionContext,
    ) -> SplitDecision {
        let unsplit_cost = unsplit_result.cost;

        // Calculate split cost (sum of children + overhead)
        let split_overhead = (ctx.lambda * 2.0) as u32; // Bits for split flag
        let split_cost: u32 = split_results
            .iter()
            .map(|r| r.cost)
            .fold(split_overhead, u32::saturating_add);

        // Apply threshold
        let effective_split_cost = (f64::from(split_cost) * f64::from(self.split_threshold)) as u32;

        if effective_split_cost < unsplit_cost {
            SplitDecision::split(unsplit_cost, split_cost, split_results.to_vec())
        } else {
            SplitDecision::no_split(unsplit_cost)
        }
    }

    /// Checks for early termination (no need to try smaller partitions).
    #[must_use]
    pub fn can_early_terminate(&self, result: &PartitionDecision) -> bool {
        // Skip blocks don't need further partitioning
        if result.is_skip {
            return true;
        }

        // Very low distortion blocks don't need splitting
        result.distortion < SKIP_THRESHOLD / 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_type_num_parts() {
        assert_eq!(PartitionType::None.num_parts(), 1);
        assert_eq!(PartitionType::HorizontalSplit.num_parts(), 2);
        assert_eq!(PartitionType::VerticalSplit.num_parts(), 2);
        assert_eq!(PartitionType::Split.num_parts(), 4);
    }

    #[test]
    fn test_partition_type_is_split() {
        assert!(!PartitionType::None.is_split());
        assert!(PartitionType::Split.is_split());
        assert!(PartitionType::HorizontalSplit.is_split());
    }

    #[test]
    fn test_inter_mode() {
        assert!(!InterMode::Skip.requires_search());
        assert!(InterMode::NewMv.requires_search());
        assert!(InterMode::NearestMv.uses_predictor());
        assert!(!InterMode::NewMv.uses_predictor());
    }

    #[test]
    fn test_partition_decision_default() {
        let decision = PartitionDecision::new();
        assert_eq!(decision.cost, u32::MAX);
        assert!(!decision.is_skip);
    }

    #[test]
    fn test_partition_decision_skip() {
        let mv = MotionVector::new(8, 16);
        let decision = PartitionDecision::skip(BlockSize::Block8x8, mv, 50);

        assert!(decision.is_skip);
        assert_eq!(decision.mode, InterMode::Skip);
        assert_eq!(decision.mv.dx, 8);
        assert_eq!(decision.distortion, 50);
    }

    #[test]
    fn test_partition_decision_from_match() {
        let block_match = BlockMatch::new(MotionVector::new(10, 20), 100, 150);
        let decision = PartitionDecision::from_match(BlockSize::Block16x16, &block_match);

        assert_eq!(decision.mv.dx, 10);
        assert_eq!(decision.distortion, 100);
        assert_eq!(decision.cost, 150);
    }

    #[test]
    fn test_partition_decision_comparison() {
        let better = PartitionDecision {
            cost: 100,
            ..PartitionDecision::new()
        };
        let worse = PartitionDecision {
            cost: 200,
            ..PartitionDecision::new()
        };

        assert!(better.is_better_than(&worse));
        assert!(!worse.is_better_than(&better));
    }

    #[test]
    fn test_split_decision_no_split() {
        let decision = SplitDecision::no_split(100);
        assert!(!decision.should_split);
        assert_eq!(decision.unsplit_cost, 100);
    }

    #[test]
    fn test_split_decision_split() {
        let children = vec![
            PartitionDecision {
                cost: 30,
                ..PartitionDecision::new()
            },
            PartitionDecision {
                cost: 30,
                ..PartitionDecision::new()
            },
        ];

        let decision = SplitDecision::split(100, 60, children);
        assert!(decision.should_split);
    }

    #[test]
    fn test_skip_detector() {
        let detector = SkipDetector::new(100);
        let block_match = BlockMatch::new(MotionVector::new(4, 4), 50, 60);
        let predicted_mv = MotionVector::new(4, 4);

        assert!(detector.can_skip(&block_match, &predicted_mv));

        let bad_match = BlockMatch::new(MotionVector::new(100, 100), 150, 200);
        assert!(!detector.can_skip(&bad_match, &predicted_mv));
    }

    #[test]
    fn test_merge_candidate() {
        let candidate = MergeCandidate::new(MotionVector::new(10, 20), 0, MergeSource::Left);
        assert_eq!(candidate.mv.dx, 10);
        assert_eq!(candidate.source, MergeSource::Left);
    }

    #[test]
    fn test_merge_candidate_list() {
        let mut list = MergeCandidateList::new();

        list.add(MergeCandidate::new(
            MotionVector::new(10, 20),
            0,
            MergeSource::Left,
        ));
        list.add(MergeCandidate::new(
            MotionVector::new(30, 40),
            0,
            MergeSource::Top,
        ));

        assert_eq!(list.len(), 2);
        assert_eq!(list.get(0).expect("get should return value").mv.dx, 10);
    }

    #[test]
    fn test_merge_candidate_list_dedup() {
        let mut list = MergeCandidateList::new();

        list.add(MergeCandidate::new(
            MotionVector::new(10, 20),
            0,
            MergeSource::Left,
        ));
        list.add(MergeCandidate::new(
            MotionVector::new(10, 20),
            0,
            MergeSource::Top,
        ));

        // Duplicate should not be added
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn test_partition_context() {
        let ctx = PartitionContext::new(1920, 1080)
            .at(100, 200)
            .with_size_range(BlockSize::Block4x4, BlockSize::Block64x64);

        assert_eq!(ctx.x, 100);
        assert_eq!(ctx.y, 200);
        assert!(ctx.can_use_size(BlockSize::Block64x64));
    }

    #[test]
    fn test_partition_context_child_size() {
        assert_eq!(
            PartitionContext::child_size(BlockSize::Block64x64),
            Some(BlockSize::Block32x32)
        );
        assert_eq!(
            PartitionContext::child_size(BlockSize::Block8x8),
            Some(BlockSize::Block4x4)
        );
        assert_eq!(PartitionContext::child_size(BlockSize::Block4x4), None);
    }

    #[test]
    fn test_partition_decider() {
        let decider = PartitionDecider::new();

        let unsplit = PartitionDecision {
            cost: 200,
            ..PartitionDecision::new()
        };
        let split_results = vec![
            PartitionDecision {
                cost: 40,
                ..PartitionDecision::new()
            },
            PartitionDecision {
                cost: 40,
                ..PartitionDecision::new()
            },
            PartitionDecision {
                cost: 40,
                ..PartitionDecision::new()
            },
            PartitionDecision {
                cost: 40,
                ..PartitionDecision::new()
            },
        ];

        let ctx = PartitionContext::new(1920, 1080);
        let decision = decider.decide_split(&unsplit, &split_results, &ctx);

        // Split should be better (4*40 = 160 + overhead < 200)
        assert!(decision.should_split || decision.split_cost < 200);
    }

    #[test]
    fn test_partition_decider_early_termination() {
        let decider = PartitionDecider::new();

        let skip_result = PartitionDecision {
            is_skip: true,
            distortion: 10,
            ..PartitionDecision::new()
        };

        assert!(decider.can_early_terminate(&skip_result));

        let low_distortion = PartitionDecision {
            distortion: 10,
            ..PartitionDecision::new()
        };

        assert!(decider.can_early_terminate(&low_distortion));
    }
}
