// Copyright 2024 The OxiMedia Project Developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Block coding mode decision for Theora encoding.
//!
//! Implements algorithms to choose the best coding mode for each block:
//! intra vs inter, motion vector selection, and subblock partitioning.

use crate::theora::intra_pred::{select_best_mode, IntraPredContext, IntraPredMode};
use crate::theora::motion::{motion_estimation_diamond, MotionVector};
use crate::theora::tables::CodingMode;
use crate::theora::transform::{copy_block, Block8x8};

/// Block decision result.
#[derive(Debug, Clone)]
pub struct BlockDecision {
    /// Chosen coding mode.
    pub mode: CodingMode,
    /// Motion vector (if applicable).
    pub mv: Option<MotionVector>,
    /// Intra prediction mode (if applicable).
    pub intra_mode: Option<IntraPredMode>,
    /// Rate-distortion cost.
    pub cost: f64,
}

/// Block decision engine.
pub struct BlockDecisionEngine {
    /// Lambda parameter for rate-distortion optimization.
    lambda: f32,
    /// Motion estimation search range.
    me_range: i16,
    /// Enable subpixel motion estimation.
    subpel_me: bool,
    /// Enable rate-distortion optimization.
    rdo_enabled: bool,
}

impl BlockDecisionEngine {
    /// Create a new block decision engine.
    ///
    /// # Arguments
    ///
    /// * `lambda` - Rate-distortion tradeoff parameter
    /// * `me_range` - Motion estimation search range (pixels)
    #[must_use]
    pub const fn new(lambda: f32, me_range: i16) -> Self {
        Self {
            lambda,
            me_range,
            subpel_me: true,
            rdo_enabled: true,
        }
    }

    /// Decide best mode for a block.
    ///
    /// # Arguments
    ///
    /// * `current` - Current block to encode
    /// * `reference` - Reference frame (for inter prediction)
    /// * `ref_stride` - Reference frame stride
    /// * `block_x` - Block X coordinate
    /// * `block_y` - Block Y coordinate
    /// * `intra_ctx` - Intra prediction context
    /// * `is_keyframe` - Whether this is a keyframe
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn decide_block_mode(
        &self,
        current: &[u8; 64],
        reference: Option<&[u8]>,
        ref_stride: usize,
        block_x: usize,
        block_y: usize,
        intra_ctx: &IntraPredContext,
        is_keyframe: bool,
    ) -> BlockDecision {
        if is_keyframe || reference.is_none() {
            // Keyframe: only intra modes
            return self.decide_intra_mode(current, intra_ctx);
        }

        let reference = reference.expect("reference is Some: checked is_none() above");

        // Try both intra and inter modes
        let intra_decision = self.decide_intra_mode(current, intra_ctx);
        let inter_decision =
            self.decide_inter_mode(current, reference, ref_stride, block_x, block_y);

        // Choose best based on cost
        if intra_decision.cost < inter_decision.cost {
            intra_decision
        } else {
            inter_decision
        }
    }

    /// Decide best intra prediction mode.
    fn decide_intra_mode(&self, current: &[u8; 64], ctx: &IntraPredContext) -> BlockDecision {
        let (best_mode, sad) = select_best_mode(current, ctx);

        BlockDecision {
            mode: CodingMode::Intra,
            mv: None,
            intra_mode: Some(best_mode),
            cost: f64::from(sad),
        }
    }

    /// Decide best inter prediction mode.
    fn decide_inter_mode(
        &self,
        current: &[u8; 64],
        reference: &[u8],
        ref_stride: usize,
        block_x: usize,
        block_y: usize,
    ) -> BlockDecision {
        // Motion estimation
        let (mv, sad) = motion_estimation_diamond(
            current,
            reference,
            ref_stride,
            block_x,
            block_y,
            self.me_range,
        );

        // Check skip mode (zero MV)
        let skip_cost = if mv.is_zero() {
            f64::from(sad)
        } else {
            f64::MAX
        };

        // Calculate inter cost with MV
        let mv_bits = self.estimate_mv_bits(&mv);
        let inter_cost = f64::from(sad) + f64::from(self.lambda * mv_bits);

        // Choose best
        if skip_cost < inter_cost && skip_cost < 100.0 {
            BlockDecision {
                mode: CodingMode::InterNoMv,
                mv: Some(MotionVector::new(0, 0)),
                intra_mode: None,
                cost: skip_cost,
            }
        } else {
            BlockDecision {
                mode: CodingMode::InterMv,
                mv: Some(mv),
                intra_mode: None,
                cost: inter_cost,
            }
        }
    }

    /// Estimate motion vector bits.
    fn estimate_mv_bits(&self, mv: &MotionVector) -> f32 {
        let x_bits = self.estimate_component_bits(mv.x);
        let y_bits = self.estimate_component_bits(mv.y);
        x_bits + y_bits
    }

    /// Estimate bits for a single MV component.
    fn estimate_component_bits(&self, value: i16) -> f32 {
        if value == 0 {
            1.0
        } else {
            let abs_val = value.abs();
            let magnitude_bits = 16 - abs_val.leading_zeros();
            (magnitude_bits * 2 + 1) as f32 // Sign + magnitude
        }
    }
}

/// Fast mode decision using heuristics.
///
/// Faster than full RDO but less optimal.
pub struct FastModeDecision {
    /// Intra bias (higher = prefer intra).
    intra_bias: f32,
    /// SAD threshold for skip mode.
    skip_threshold: u32,
}

impl FastModeDecision {
    /// Create a new fast mode decision engine.
    #[must_use]
    pub const fn new(intra_bias: f32, skip_threshold: u32) -> Self {
        Self {
            intra_bias,
            skip_threshold,
        }
    }

    /// Quickly decide block mode.
    ///
    /// # Arguments
    ///
    /// * `current` - Current block
    /// * `reference` - Reference frame (if available)
    /// * `ref_stride` - Reference stride
    /// * `block_x` - Block X coordinate
    /// * `block_y` - Block Y coordinate
    #[must_use]
    pub fn decide_fast(
        &self,
        current: &[u8; 64],
        reference: Option<&[u8]>,
        ref_stride: usize,
        block_x: usize,
        block_y: usize,
    ) -> CodingMode {
        if let Some(reference) = reference {
            // Check skip mode first
            let mut skip_block = [0u8; 64];
            copy_block(reference, ref_stride, block_x, block_y, &mut skip_block);
            let skip_sad = calculate_sad(current, &skip_block);

            if skip_sad < self.skip_threshold {
                return CodingMode::InterNoMv;
            }

            // Quick motion search (limited range)
            let (mv, inter_sad) =
                motion_estimation_diamond(current, reference, ref_stride, block_x, block_y, 4);

            // Compare inter vs intra with bias
            let biased_inter_sad = inter_sad as f32;
            let intra_sad_estimate = calculate_intra_sad_estimate(current);
            let biased_intra_sad = intra_sad_estimate as f32 * self.intra_bias;

            if biased_inter_sad < biased_intra_sad {
                if mv.is_zero() {
                    CodingMode::InterNoMv
                } else {
                    CodingMode::InterMv
                }
            } else {
                CodingMode::Intra
            }
        } else {
            CodingMode::Intra
        }
    }
}

/// Calculate SAD between two blocks.
fn calculate_sad(block1: &[u8; 64], block2: &[u8; 64]) -> u32 {
    let mut sad = 0u32;
    for i in 0..64 {
        sad += (i32::from(block1[i]) - i32::from(block2[i])).unsigned_abs();
    }
    sad
}

/// Estimate intra SAD quickly.
fn calculate_intra_sad_estimate(block: &[u8; 64]) -> u32 {
    // Use DC prediction as estimate
    let mut sum = 0u32;
    for &pixel in block.iter() {
        sum += u32::from(pixel);
    }
    let dc = sum / 64;

    let mut sad = 0u32;
    for &pixel in block.iter() {
        sad += (i32::from(pixel) - dc as i32).unsigned_abs();
    }
    sad
}

/// Subblock partitioning decision.
///
/// Decides whether to split a macroblock into smaller partitions.
pub struct SubblockDecision {
    /// Variance threshold for splitting.
    variance_threshold: f32,
    /// Enable adaptive partitioning.
    adaptive: bool,
}

impl SubblockDecision {
    /// Create a new subblock decision engine.
    #[must_use]
    pub const fn new(variance_threshold: f32, adaptive: bool) -> Self {
        Self {
            variance_threshold,
            adaptive,
        }
    }

    /// Decide whether to split a macroblock.
    ///
    /// # Arguments
    ///
    /// * `block` - 16x16 macroblock data
    #[must_use]
    pub fn should_split(&self, block: &[u8; 256]) -> bool {
        if !self.adaptive {
            return false; // Always use 16x16
        }

        // Calculate variance for each 8x8 subblock
        let variances = [
            self.calculate_subblock_variance(block, 0, 0),
            self.calculate_subblock_variance(block, 8, 0),
            self.calculate_subblock_variance(block, 0, 8),
            self.calculate_subblock_variance(block, 8, 8),
        ];

        // Check if variance differences are significant
        let max_var = variances.iter().copied().fold(f32::MIN, f32::max);
        let min_var = variances.iter().copied().fold(f32::MAX, f32::min);

        (max_var - min_var) > self.variance_threshold
    }

    /// Calculate variance for an 8x8 subblock.
    fn calculate_subblock_variance(&self, block: &[u8; 256], x: usize, y: usize) -> f32 {
        let mut sum = 0u32;
        let mut sum_sq = 0u32;

        for dy in 0..8 {
            for dx in 0..8 {
                let pixel = u32::from(block[(y + dy) * 16 + x + dx]);
                sum += pixel;
                sum_sq += pixel * pixel;
            }
        }

        let mean = sum / 64;
        let variance = (sum_sq / 64) - (mean * mean);
        variance as f32
    }
}

/// Merge decision for small partitions.
///
/// Decides whether adjacent small blocks should be merged.
pub struct MergeDecision {
    /// SAD threshold for merging.
    sad_threshold: u32,
}

impl MergeDecision {
    /// Create a new merge decision engine.
    #[must_use]
    pub const fn new(sad_threshold: u32) -> Self {
        Self { sad_threshold }
    }

    /// Check if two blocks should be merged.
    ///
    /// # Arguments
    ///
    /// * `block1` - First block
    /// * `block2` - Second block
    #[must_use]
    pub fn should_merge(&self, block1: &[u8; 64], block2: &[u8; 64]) -> bool {
        let sad = calculate_sad(block1, block2);
        sad < self.sad_threshold
    }
}

/// Early termination decision.
///
/// Decides whether to stop mode search early.
pub struct EarlyTermination {
    /// SAD threshold for early termination.
    threshold: u32,
    /// Enable early termination.
    enabled: bool,
}

impl EarlyTermination {
    /// Create a new early termination decision engine.
    #[must_use]
    pub const fn new(threshold: u32, enabled: bool) -> Self {
        Self { threshold, enabled }
    }

    /// Check if we should terminate mode search.
    ///
    /// # Arguments
    ///
    /// * `current_sad` - Current best SAD
    #[must_use]
    pub fn should_terminate(&self, current_sad: u32) -> bool {
        self.enabled && current_sad < self.threshold
    }
}

/// Block complexity analyzer.
///
/// Analyzes block characteristics to guide encoding decisions.
pub struct BlockComplexity {
    /// Spatial activity threshold.
    activity_threshold: f32,
}

impl BlockComplexity {
    /// Create a new block complexity analyzer.
    #[must_use]
    pub const fn new(activity_threshold: f32) -> Self {
        Self { activity_threshold }
    }

    /// Calculate block spatial activity.
    ///
    /// # Arguments
    ///
    /// * `block` - Block data
    #[must_use]
    pub fn spatial_activity(&self, block: &[u8; 64]) -> f32 {
        let mut activity = 0u32;

        // Horizontal gradients
        for y in 0..8 {
            for x in 0..7 {
                let diff = (i16::from(block[y * 8 + x + 1]) - i16::from(block[y * 8 + x])).abs();
                activity += diff as u32;
            }
        }

        // Vertical gradients
        for y in 0..7 {
            for x in 0..8 {
                let diff = (i16::from(block[(y + 1) * 8 + x]) - i16::from(block[y * 8 + x])).abs();
                activity += diff as u32;
            }
        }

        activity as f32 / 112.0 // Normalize
    }

    /// Check if block is homogeneous.
    #[must_use]
    pub fn is_homogeneous(&self, block: &[u8; 64]) -> bool {
        self.spatial_activity(block) < self.activity_threshold
    }

    /// Calculate temporal activity (difference from reference).
    ///
    /// # Arguments
    ///
    /// * `current` - Current block
    /// * `reference` - Reference block
    #[must_use]
    pub fn temporal_activity(&self, current: &[u8; 64], reference: &[u8; 64]) -> f32 {
        let sad = calculate_sad(current, reference);
        sad as f32 / 64.0
    }
}

/// Mode decision statistics collector.
#[derive(Debug, Clone, Default)]
pub struct ModeStats {
    /// Number of intra blocks.
    pub intra_count: u32,
    /// Number of inter blocks.
    pub inter_count: u32,
    /// Number of skip blocks.
    pub skip_count: u32,
    /// Total RD cost.
    pub total_cost: f64,
}

impl ModeStats {
    /// Create a new stats collector.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            intra_count: 0,
            inter_count: 0,
            skip_count: 0,
            total_cost: 0.0,
        }
    }

    /// Update statistics with a decision.
    pub fn update(&mut self, decision: &BlockDecision) {
        match decision.mode {
            CodingMode::Intra => self.intra_count += 1,
            CodingMode::InterMv | CodingMode::InterGoldenMv => self.inter_count += 1,
            CodingMode::InterNoMv | CodingMode::InterGoldenNoMv | CodingMode::NotCoded => {
                self.skip_count += 1
            }
            _ => {}
        }
        self.total_cost += decision.cost;
    }

    /// Get average cost per block.
    #[must_use]
    pub fn average_cost(&self) -> f64 {
        let total_blocks = self.intra_count + self.inter_count + self.skip_count;
        if total_blocks > 0 {
            self.total_cost / f64::from(total_blocks)
        } else {
            0.0
        }
    }

    /// Reset statistics.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_decision_engine() {
        let engine = BlockDecisionEngine::new(1.0, 16);
        let current = [128u8; 64];
        let ctx = IntraPredContext::default();

        let decision = engine.decide_intra_mode(&current, &ctx);
        assert_eq!(decision.mode, CodingMode::Intra);
    }

    #[test]
    fn test_fast_mode_decision() {
        let fast = FastModeDecision::new(1.2, 100);
        let current = [128u8; 64];
        let mode = fast.decide_fast(&current, None, 0, 0, 0);
        assert_eq!(mode, CodingMode::Intra);
    }

    #[test]
    fn test_sad_calculation() {
        let block1 = [100u8; 64];
        let block2 = [110u8; 64];
        let sad = calculate_sad(&block1, &block2);
        assert_eq!(sad, 64 * 10);
    }

    #[test]
    fn test_subblock_decision() {
        let decision = SubblockDecision::new(100.0, true);
        let block = [128u8; 256];
        assert!(!decision.should_split(&block)); // Uniform block shouldn't split
    }

    #[test]
    fn test_merge_decision() {
        // SAD of [128;64] vs [129;64] = 64 * 1 = 64, which is below threshold of 100
        let merge = MergeDecision::new(100);
        let block1 = [128u8; 64];
        let block2 = [129u8; 64];
        assert!(merge.should_merge(&block1, &block2));

        // SAD of [128;64] vs [130;64] = 64 * 2 = 128, which is above threshold of 100
        let block3 = [130u8; 64];
        assert!(!merge.should_merge(&block1, &block3));
    }

    #[test]
    fn test_early_termination() {
        let early = EarlyTermination::new(50, true);
        assert!(early.should_terminate(30));
        assert!(!early.should_terminate(100));
    }

    #[test]
    fn test_block_complexity() {
        let analyzer = BlockComplexity::new(10.0);
        let block = [128u8; 64];
        let activity = analyzer.spatial_activity(&block);
        assert_eq!(activity, 0.0); // Uniform block has no activity
        assert!(analyzer.is_homogeneous(&block));
    }

    #[test]
    fn test_mode_stats() {
        let mut stats = ModeStats::new();
        let decision = BlockDecision {
            mode: CodingMode::Intra,
            mv: None,
            intra_mode: None,
            cost: 100.0,
        };
        stats.update(&decision);
        assert_eq!(stats.intra_count, 1);
        assert_eq!(stats.average_cost(), 100.0);
    }
}
