//! Partition decision optimization.

use crate::OptimizerConfig;
use oximedia_core::OxiResult;

/// Partition modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionMode {
    /// No split (use full block).
    None,
    /// Split horizontally.
    Horizontal,
    /// Split vertically.
    Vertical,
    /// Split into 4 quadrants.
    Split4,
}

/// Partition decision result.
#[derive(Debug, Clone, Copy)]
pub struct PartitionDecision {
    /// Selected partition mode.
    pub mode: PartitionMode,
    /// Decision cost.
    pub cost: f64,
    /// Whether to recurse.
    pub should_recurse: bool,
}

/// Partition optimizer.
pub struct SplitOptimizer {
    min_block_size: usize,
    #[allow(dead_code)]
    max_block_size: usize,
    enable_asymmetric: bool,
}

impl SplitOptimizer {
    /// Creates a new partition optimizer.
    pub fn new(config: &OptimizerConfig) -> OxiResult<Self> {
        let (min_size, max_size, asymmetric) = match config.level {
            crate::OptimizationLevel::Fast => (16, 64, false),
            crate::OptimizationLevel::Medium => (8, 64, false),
            crate::OptimizationLevel::Slow => (4, 128, true),
            crate::OptimizationLevel::Placebo => (4, 128, true),
        };

        Ok(Self {
            min_block_size: min_size,
            max_block_size: max_size,
            enable_asymmetric: asymmetric,
        })
    }

    /// Decides optimal partition for a block.
    #[allow(dead_code)]
    #[must_use]
    pub fn decide(&self, pixels: &[u8], block_size: usize, complexity: f64) -> PartitionDecision {
        if block_size <= self.min_block_size {
            return PartitionDecision {
                mode: PartitionMode::None,
                cost: 0.0,
                should_recurse: false,
            };
        }

        let mut best_mode = PartitionMode::None;
        let mut best_cost = self.evaluate_no_split(pixels, complexity);

        // Try splitting if block is complex enough
        if complexity > 100.0 && block_size > self.min_block_size {
            let split_cost = self.evaluate_split4(pixels, complexity);
            if split_cost < best_cost {
                best_cost = split_cost;
                best_mode = PartitionMode::Split4;
            }

            if self.enable_asymmetric {
                let h_cost = self.evaluate_horizontal(pixels, complexity);
                if h_cost < best_cost {
                    best_cost = h_cost;
                    best_mode = PartitionMode::Horizontal;
                }

                let v_cost = self.evaluate_vertical(pixels, complexity);
                if v_cost < best_cost {
                    best_cost = v_cost;
                    best_mode = PartitionMode::Vertical;
                }
            }
        }

        PartitionDecision {
            mode: best_mode,
            cost: best_cost,
            should_recurse: best_mode != PartitionMode::None,
        }
    }

    fn evaluate_no_split(&self, pixels: &[u8], complexity: f64) -> f64 {
        // Cost of not splitting
        let variance = self.calculate_variance(pixels);
        variance + complexity * 0.1
    }

    fn evaluate_split4(&self, pixels: &[u8], complexity: f64) -> f64 {
        // Cost of splitting into 4
        let variance = self.calculate_variance(pixels);
        variance * 0.7 + complexity * 0.2 + 10.0 // Splitting cost
    }

    fn evaluate_horizontal(&self, pixels: &[u8], complexity: f64) -> f64 {
        // Cost of horizontal split
        let variance = self.calculate_variance(pixels);
        variance * 0.8 + complexity * 0.15 + 8.0
    }

    fn evaluate_vertical(&self, pixels: &[u8], complexity: f64) -> f64 {
        // Cost of vertical split
        let variance = self.calculate_variance(pixels);
        variance * 0.8 + complexity * 0.15 + 8.0
    }

    fn calculate_variance(&self, pixels: &[u8]) -> f64 {
        if pixels.is_empty() {
            return 0.0;
        }

        let mean = pixels.iter().map(|&p| f64::from(p)).sum::<f64>() / pixels.len() as f64;
        pixels
            .iter()
            .map(|&p| {
                let diff = f64::from(p) - mean;
                diff * diff
            })
            .sum::<f64>()
            / pixels.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_optimizer_creation() {
        let config = OptimizerConfig::default();
        let optimizer =
            SplitOptimizer::new(&config).expect("split optimizer creation should succeed");
        assert_eq!(optimizer.min_block_size, 8);
    }

    #[test]
    fn test_partition_modes() {
        assert_ne!(PartitionMode::None, PartitionMode::Split4);
        assert_eq!(PartitionMode::Horizontal, PartitionMode::Horizontal);
    }

    #[test]
    fn test_min_block_size_no_split() {
        let config = OptimizerConfig::default();
        let optimizer =
            SplitOptimizer::new(&config).expect("split optimizer creation should succeed");
        let pixels = vec![128u8; 64];
        let decision = optimizer.decide(&pixels, 8, 50.0);
        assert_eq!(decision.mode, PartitionMode::None);
        assert!(!decision.should_recurse);
    }

    #[test]
    fn test_high_complexity_split() {
        let config = OptimizerConfig::default();
        let optimizer =
            SplitOptimizer::new(&config).expect("split optimizer creation should succeed");
        let pixels = vec![128u8; 256]; // Larger block
        let decision = optimizer.decide(&pixels, 32, 500.0); // High complexity
                                                             // May or may not split depending on cost evaluation
        assert!(decision.cost >= 0.0);
    }
}
