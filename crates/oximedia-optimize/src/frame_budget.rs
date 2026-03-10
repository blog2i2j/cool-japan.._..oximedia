#![allow(dead_code)]
//! Frame-level bit budget allocation and tracking.
//!
//! This module provides tools for distributing a total bitrate budget across
//! individual frames, taking into account frame type, scene complexity, and
//! temporal position within a GOP. The budget tracker monitors actual vs.
//! predicted usage and dynamically adjusts future allocations to stay on target.

use std::collections::VecDeque;

/// Frame type for budget weighting purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BudgetFrameType {
    /// Intra-coded frame (I-frame) - typically largest.
    Intra,
    /// Predicted frame (P-frame) - medium size.
    Predicted,
    /// Bi-predicted frame (B-frame) - typically smallest.
    BiPredicted,
}

/// Weight multipliers for different frame types relative to a P-frame baseline.
#[derive(Debug, Clone)]
pub struct FrameTypeWeights {
    /// Weight for I-frames (typically > 1.0).
    pub intra_weight: f64,
    /// Weight for P-frames (baseline, typically 1.0).
    pub predicted_weight: f64,
    /// Weight for B-frames (typically < 1.0).
    pub bi_predicted_weight: f64,
}

impl Default for FrameTypeWeights {
    fn default() -> Self {
        Self {
            intra_weight: 5.0,
            predicted_weight: 1.0,
            bi_predicted_weight: 0.5,
        }
    }
}

impl FrameTypeWeights {
    /// Creates custom frame type weights.
    #[must_use]
    pub fn new(intra: f64, predicted: f64, bi_predicted: f64) -> Self {
        Self {
            intra_weight: intra.max(0.1),
            predicted_weight: predicted.max(0.1),
            bi_predicted_weight: bi_predicted.max(0.1),
        }
    }

    /// Returns the weight for the given frame type.
    #[must_use]
    pub fn weight_for(&self, frame_type: BudgetFrameType) -> f64 {
        match frame_type {
            BudgetFrameType::Intra => self.intra_weight,
            BudgetFrameType::Predicted => self.predicted_weight,
            BudgetFrameType::BiPredicted => self.bi_predicted_weight,
        }
    }
}

/// A single frame's budget allocation and actual usage.
#[derive(Debug, Clone)]
pub struct FrameBudgetEntry {
    /// Frame index in the stream.
    pub frame_index: u64,
    /// Frame type.
    pub frame_type: BudgetFrameType,
    /// Allocated bit budget in bits.
    pub allocated_bits: u64,
    /// Actual bits used (filled in after encoding).
    pub actual_bits: Option<u64>,
    /// Scene complexity factor (0.0 to 2.0, 1.0 = average).
    pub complexity: f64,
}

impl FrameBudgetEntry {
    /// Creates a new budget entry.
    #[must_use]
    pub fn new(frame_index: u64, frame_type: BudgetFrameType, allocated_bits: u64) -> Self {
        Self {
            frame_index,
            frame_type,
            allocated_bits,
            actual_bits: None,
            complexity: 1.0,
        }
    }

    /// Records the actual bits used by this frame.
    pub fn record_actual(&mut self, bits: u64) {
        self.actual_bits = Some(bits);
    }

    /// Returns the budget deviation (actual - allocated) if actual is known.
    #[must_use]
    pub fn deviation(&self) -> Option<i64> {
        self.actual_bits
            .map(|actual| actual as i64 - self.allocated_bits as i64)
    }

    /// Returns the usage ratio (actual / allocated) if actual is known.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn usage_ratio(&self) -> Option<f64> {
        if self.allocated_bits == 0 {
            return self.actual_bits.map(|_| 0.0);
        }
        self.actual_bits
            .map(|actual| actual as f64 / self.allocated_bits as f64)
    }
}

/// Configuration for the frame budget allocator.
#[derive(Debug, Clone)]
pub struct BudgetAllocatorConfig {
    /// Target bitrate in bits per second.
    pub target_bitrate_bps: u64,
    /// Frame rate.
    pub frame_rate: f64,
    /// Frame type weight multipliers.
    pub weights: FrameTypeWeights,
    /// Maximum allowed overshoot ratio (e.g., 1.5 = 150% of budget).
    pub max_overshoot_ratio: f64,
    /// Smoothing factor for budget corrections (0.0 to 1.0).
    pub smoothing_factor: f64,
    /// Number of recent frames to consider for running average.
    pub window_size: usize,
}

impl Default for BudgetAllocatorConfig {
    fn default() -> Self {
        Self {
            target_bitrate_bps: 5_000_000,
            frame_rate: 30.0,
            weights: FrameTypeWeights::default(),
            max_overshoot_ratio: 1.5,
            smoothing_factor: 0.3,
            window_size: 30,
        }
    }
}

/// Frame-level bit budget allocator.
///
/// Distributes a total bitrate budget across frames, adjusting allocations
/// based on frame type, complexity, and accumulated budget deviation.
#[derive(Debug)]
pub struct FrameBudgetAllocator {
    config: BudgetAllocatorConfig,
    /// Bits per frame at baseline (target_bps / fps).
    base_bits_per_frame: f64,
    /// Running deviation (positive = over budget, negative = under budget).
    accumulated_deviation: i64,
    /// Recent frame entries for windowed analysis.
    recent_entries: VecDeque<FrameBudgetEntry>,
    /// Total frames allocated so far.
    total_frames_allocated: u64,
    /// Total bits allocated so far.
    total_bits_allocated: u64,
    /// Total actual bits used so far.
    total_bits_used: u64,
}

impl FrameBudgetAllocator {
    /// Creates a new frame budget allocator.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(config: BudgetAllocatorConfig) -> Self {
        let base_bits_per_frame = if config.frame_rate > 0.0 {
            config.target_bitrate_bps as f64 / config.frame_rate
        } else {
            0.0
        };
        Self {
            config,
            base_bits_per_frame,
            accumulated_deviation: 0,
            recent_entries: VecDeque::new(),
            total_frames_allocated: 0,
            total_bits_allocated: 0,
            total_bits_used: 0,
        }
    }

    /// Allocates a bit budget for the next frame.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::cast_possible_truncation)]
    pub fn allocate(
        &mut self,
        frame_index: u64,
        frame_type: BudgetFrameType,
        complexity: f64,
    ) -> FrameBudgetEntry {
        let type_weight = self.config.weights.weight_for(frame_type);
        let complexity_clamped = complexity.clamp(0.1, 3.0);

        // Base allocation adjusted by frame type and complexity
        let raw_budget = self.base_bits_per_frame * type_weight * complexity_clamped;

        // Apply correction based on accumulated deviation
        let correction = if self.accumulated_deviation > 0 {
            // Over budget: reduce allocation
            let correction_amount =
                self.accumulated_deviation as f64 * self.config.smoothing_factor;
            (-correction_amount).max(-raw_budget * 0.5) // Don't cut more than 50%
        } else {
            // Under budget: slightly increase allocation
            let correction_amount =
                (-self.accumulated_deviation) as f64 * self.config.smoothing_factor * 0.5;
            correction_amount.min(raw_budget * 0.3) // Don't add more than 30%
        };

        let adjusted_budget = (raw_budget + correction).max(1.0);

        // Clamp to max overshoot
        let max_budget = self.base_bits_per_frame * self.config.max_overshoot_ratio * type_weight;
        let final_budget = adjusted_budget.min(max_budget).max(1.0) as u64;

        let mut entry = FrameBudgetEntry::new(frame_index, frame_type, final_budget);
        entry.complexity = complexity_clamped;

        self.total_frames_allocated += 1;
        self.total_bits_allocated += final_budget;

        entry
    }

    /// Records the actual bits used for a frame and updates the deviation tracker.
    pub fn record_usage(&mut self, entry: &mut FrameBudgetEntry, actual_bits: u64) {
        entry.record_actual(actual_bits);

        if let Some(dev) = entry.deviation() {
            self.accumulated_deviation += dev;
        }

        self.total_bits_used += actual_bits;

        // Maintain window
        self.recent_entries.push_back(entry.clone());
        if self.recent_entries.len() > self.config.window_size {
            self.recent_entries.pop_front();
        }
    }

    /// Returns the accumulated budget deviation in bits.
    #[must_use]
    pub fn accumulated_deviation(&self) -> i64 {
        self.accumulated_deviation
    }

    /// Returns the average usage ratio over the recent window.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn average_usage_ratio(&self) -> f64 {
        let entries_with_actual: Vec<f64> = self
            .recent_entries
            .iter()
            .filter_map(|e| e.usage_ratio())
            .collect();
        if entries_with_actual.is_empty() {
            return 1.0;
        }
        entries_with_actual.iter().sum::<f64>() / entries_with_actual.len() as f64
    }

    /// Returns total frames allocated.
    #[must_use]
    pub fn total_frames(&self) -> u64 {
        self.total_frames_allocated
    }

    /// Returns the total bits allocated.
    #[must_use]
    pub fn total_allocated(&self) -> u64 {
        self.total_bits_allocated
    }

    /// Returns the total bits actually used.
    #[must_use]
    pub fn total_used(&self) -> u64 {
        self.total_bits_used
    }

    /// Returns the overall budget efficiency (used / allocated).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn overall_efficiency(&self) -> f64 {
        if self.total_bits_allocated == 0 {
            return 0.0;
        }
        self.total_bits_used as f64 / self.total_bits_allocated as f64
    }

    /// Returns the base bits per frame at the target bitrate.
    #[must_use]
    pub fn base_bits_per_frame(&self) -> f64 {
        self.base_bits_per_frame
    }

    /// Resets the allocator state for a new segment.
    pub fn reset(&mut self) {
        self.accumulated_deviation = 0;
        self.recent_entries.clear();
        self.total_frames_allocated = 0;
        self.total_bits_allocated = 0;
        self.total_bits_used = 0;
    }
}

/// Summary of budget allocation over a range of frames.
#[derive(Debug, Clone)]
pub struct BudgetSummary {
    /// Number of frames.
    pub frame_count: u64,
    /// Total allocated bits.
    pub total_allocated: u64,
    /// Total actual bits used.
    pub total_used: u64,
    /// Average usage ratio.
    pub avg_usage_ratio: f64,
    /// Peak frame usage in bits.
    pub peak_frame_bits: u64,
    /// Minimum frame usage in bits.
    pub min_frame_bits: u64,
}

impl BudgetSummary {
    /// Creates a summary from a slice of budget entries.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn from_entries(entries: &[FrameBudgetEntry]) -> Self {
        let frame_count = entries.len() as u64;
        let total_allocated: u64 = entries.iter().map(|e| e.allocated_bits).sum();
        let total_used: u64 = entries.iter().filter_map(|e| e.actual_bits).sum();

        let ratios: Vec<f64> = entries.iter().filter_map(|e| e.usage_ratio()).collect();
        let avg_usage_ratio = if ratios.is_empty() {
            0.0
        } else {
            ratios.iter().sum::<f64>() / ratios.len() as f64
        };

        let peak_frame_bits = entries
            .iter()
            .filter_map(|e| e.actual_bits)
            .max()
            .unwrap_or(0);
        let min_frame_bits = entries
            .iter()
            .filter_map(|e| e.actual_bits)
            .min()
            .unwrap_or(0);

        Self {
            frame_count,
            total_allocated,
            total_used,
            avg_usage_ratio,
            peak_frame_bits,
            min_frame_bits,
        }
    }

    /// Returns the overall budget deviation as a percentage.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn deviation_percent(&self) -> f64 {
        if self.total_allocated == 0 {
            return 0.0;
        }
        ((self.total_used as f64 - self.total_allocated as f64) / self.total_allocated as f64)
            * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_type_weights_default() {
        let w = FrameTypeWeights::default();
        assert!((w.intra_weight - 5.0).abs() < f64::EPSILON);
        assert!((w.predicted_weight - 1.0).abs() < f64::EPSILON);
        assert!((w.bi_predicted_weight - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_frame_type_weights_custom() {
        let w = FrameTypeWeights::new(3.0, 1.0, 0.3);
        assert!((w.intra_weight - 3.0).abs() < f64::EPSILON);
        assert!((w.bi_predicted_weight - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_frame_type_weights_clamped() {
        let w = FrameTypeWeights::new(-1.0, 0.0, -5.0);
        assert!(w.intra_weight >= 0.1);
        assert!(w.predicted_weight >= 0.1);
        assert!(w.bi_predicted_weight >= 0.1);
    }

    #[test]
    fn test_weight_for() {
        let w = FrameTypeWeights::default();
        assert!((w.weight_for(BudgetFrameType::Intra) - 5.0).abs() < f64::EPSILON);
        assert!((w.weight_for(BudgetFrameType::Predicted) - 1.0).abs() < f64::EPSILON);
        assert!((w.weight_for(BudgetFrameType::BiPredicted) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_frame_budget_entry_new() {
        let entry = FrameBudgetEntry::new(0, BudgetFrameType::Intra, 10000);
        assert_eq!(entry.frame_index, 0);
        assert_eq!(entry.frame_type, BudgetFrameType::Intra);
        assert_eq!(entry.allocated_bits, 10000);
        assert!(entry.actual_bits.is_none());
    }

    #[test]
    fn test_frame_budget_entry_record_actual() {
        let mut entry = FrameBudgetEntry::new(0, BudgetFrameType::Predicted, 5000);
        entry.record_actual(4800);
        assert_eq!(entry.actual_bits, Some(4800));
    }

    #[test]
    fn test_frame_budget_entry_deviation() {
        let mut entry = FrameBudgetEntry::new(0, BudgetFrameType::Predicted, 5000);
        assert!(entry.deviation().is_none());
        entry.record_actual(5500);
        assert_eq!(entry.deviation(), Some(500));
        entry.record_actual(4000);
        assert_eq!(entry.deviation(), Some(-1000));
    }

    #[test]
    fn test_frame_budget_entry_usage_ratio() {
        let mut entry = FrameBudgetEntry::new(0, BudgetFrameType::Predicted, 5000);
        assert!(entry.usage_ratio().is_none());
        entry.record_actual(5000);
        assert!(
            (entry
                .usage_ratio()
                .expect("usage ratio should be computable")
                - 1.0)
                .abs()
                < f64::EPSILON
        );
        entry.record_actual(2500);
        assert!(
            (entry
                .usage_ratio()
                .expect("usage ratio should be computable")
                - 0.5)
                .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_allocator_base_bits() {
        let config = BudgetAllocatorConfig {
            target_bitrate_bps: 6_000_000,
            frame_rate: 30.0,
            ..Default::default()
        };
        let allocator = FrameBudgetAllocator::new(config);
        // 6_000_000 / 30 = 200_000
        assert!((allocator.base_bits_per_frame() - 200_000.0).abs() < 1.0);
    }

    #[test]
    fn test_allocator_basic_allocation() {
        let config = BudgetAllocatorConfig {
            target_bitrate_bps: 3_000_000,
            frame_rate: 30.0,
            ..Default::default()
        };
        let mut allocator = FrameBudgetAllocator::new(config);
        let entry = allocator.allocate(0, BudgetFrameType::Predicted, 1.0);
        assert!(entry.allocated_bits > 0);
        assert_eq!(allocator.total_frames(), 1);
    }

    #[test]
    fn test_allocator_intra_gets_more() {
        let config = BudgetAllocatorConfig::default();
        let mut allocator = FrameBudgetAllocator::new(config);
        let intra = allocator.allocate(0, BudgetFrameType::Intra, 1.0);
        let predicted = allocator.allocate(1, BudgetFrameType::Predicted, 1.0);
        assert!(intra.allocated_bits > predicted.allocated_bits);
    }

    #[test]
    fn test_allocator_record_usage() {
        let config = BudgetAllocatorConfig::default();
        let mut allocator = FrameBudgetAllocator::new(config);
        let mut entry = allocator.allocate(0, BudgetFrameType::Predicted, 1.0);
        let allocated = entry.allocated_bits;
        allocator.record_usage(&mut entry, allocated + 1000);
        assert_eq!(allocator.accumulated_deviation(), 1000);
        assert_eq!(allocator.total_used(), allocated + 1000);
    }

    #[test]
    fn test_allocator_reset() {
        let config = BudgetAllocatorConfig::default();
        let mut allocator = FrameBudgetAllocator::new(config);
        let mut entry = allocator.allocate(0, BudgetFrameType::Predicted, 1.0);
        allocator.record_usage(&mut entry, 5000);
        allocator.reset();
        assert_eq!(allocator.accumulated_deviation(), 0);
        assert_eq!(allocator.total_frames(), 0);
        assert_eq!(allocator.total_used(), 0);
    }

    #[test]
    fn test_budget_summary_from_entries() {
        let mut entries = vec![
            FrameBudgetEntry::new(0, BudgetFrameType::Intra, 10000),
            FrameBudgetEntry::new(1, BudgetFrameType::Predicted, 3000),
            FrameBudgetEntry::new(2, BudgetFrameType::BiPredicted, 1500),
        ];
        entries[0].record_actual(9500);
        entries[1].record_actual(3200);
        entries[2].record_actual(1400);

        let summary = BudgetSummary::from_entries(&entries);
        assert_eq!(summary.frame_count, 3);
        assert_eq!(summary.total_allocated, 14500);
        assert_eq!(summary.total_used, 14100);
        assert_eq!(summary.peak_frame_bits, 9500);
        assert_eq!(summary.min_frame_bits, 1400);
    }

    #[test]
    fn test_budget_summary_deviation_percent() {
        let mut entries = vec![FrameBudgetEntry::new(0, BudgetFrameType::Predicted, 10000)];
        entries[0].record_actual(11000);
        let summary = BudgetSummary::from_entries(&entries);
        assert!((summary.deviation_percent() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_overall_efficiency() {
        let config = BudgetAllocatorConfig::default();
        let mut allocator = FrameBudgetAllocator::new(config);
        let mut entry = allocator.allocate(0, BudgetFrameType::Predicted, 1.0);
        let allocated = entry.allocated_bits;
        allocator.record_usage(&mut entry, allocated / 2);
        assert!((allocator.overall_efficiency() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_average_usage_ratio_empty() {
        let config = BudgetAllocatorConfig::default();
        let allocator = FrameBudgetAllocator::new(config);
        assert!((allocator.average_usage_ratio() - 1.0).abs() < f64::EPSILON);
    }
}
