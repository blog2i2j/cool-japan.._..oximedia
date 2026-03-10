//! Intra mode selection optimization.

use crate::{
    rdo::{ModeCandidate, RdoEngine},
    OptimizerConfig,
};
use oximedia_core::OxiResult;

/// Intra prediction modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntraMode {
    /// DC (mean) prediction.
    Dc,
    /// Horizontal prediction.
    Horizontal,
    /// Vertical prediction.
    Vertical,
    /// Diagonal prediction.
    Diagonal,
    /// Paeth predictor.
    Paeth,
    /// Smooth prediction.
    Smooth,
    /// Directional mode with angle.
    Directional(u8),
}

/// Intra mode optimizer.
pub struct ModeOptimizer {
    rdo_engine: RdoEngine,
    full_rdo: bool,
}

impl ModeOptimizer {
    /// Creates a new intra mode optimizer.
    pub fn new(config: &OptimizerConfig) -> OxiResult<Self> {
        let rdo_engine = RdoEngine::new(config)?;
        let full_rdo = rdo_engine.should_perform_full_rdo();

        Ok(Self {
            rdo_engine,
            full_rdo,
        })
    }

    /// Selects the best intra mode for a block.
    #[allow(dead_code)]
    #[must_use]
    pub fn select_mode(&self, src: &[u8], neighbors: &[u8], qp: u8) -> IntraModeDecision {
        let modes = self.candidate_modes();
        let candidates: Vec<_> = modes
            .iter()
            .enumerate()
            .map(|(idx, _)| ModeCandidate {
                mode_idx: idx,
                qp,
                data: vec![],
            })
            .collect();

        let result = self.rdo_engine.evaluate_modes(&candidates, |candidate| {
            self.evaluate_mode(&modes[candidate.mode_idx], src, neighbors)
        });

        IntraModeDecision {
            mode: modes[result.best_mode_idx],
            cost: result.cost,
            distortion: result.distortion,
            rate: result.rate,
        }
    }

    fn candidate_modes(&self) -> Vec<IntraMode> {
        if self.full_rdo {
            // Full set of modes
            let mut modes = vec![
                IntraMode::Dc,
                IntraMode::Horizontal,
                IntraMode::Vertical,
                IntraMode::Diagonal,
                IntraMode::Paeth,
                IntraMode::Smooth,
            ];
            // Add directional modes
            for angle in (0..8).map(|i| i * 22) {
                modes.push(IntraMode::Directional(angle));
            }
            modes
        } else {
            // Reduced set for fast encoding
            vec![
                IntraMode::Dc,
                IntraMode::Horizontal,
                IntraMode::Vertical,
                IntraMode::Diagonal,
            ]
        }
    }

    fn evaluate_mode(&self, mode: &IntraMode, src: &[u8], _neighbors: &[u8]) -> (f64, f64) {
        // Simplified evaluation (would perform actual prediction in production)
        let distortion = self.calculate_distortion(src, mode);
        let rate = self.estimate_rate(mode);
        (distortion, rate)
    }

    fn calculate_distortion(&self, src: &[u8], _mode: &IntraMode) -> f64 {
        // Simplified (would compare with predicted block)
        src.iter().map(|&x| f64::from(x)).sum::<f64>() / 10.0
    }

    fn estimate_rate(&self, mode: &IntraMode) -> f64 {
        match mode {
            IntraMode::Dc => 2.0,
            IntraMode::Horizontal => 3.0,
            IntraMode::Vertical => 3.0,
            IntraMode::Diagonal => 4.0,
            IntraMode::Paeth => 5.0,
            IntraMode::Smooth => 5.0,
            IntraMode::Directional(_) => 6.0,
        }
    }
}

/// Intra mode decision result.
#[derive(Debug, Clone, Copy)]
pub struct IntraModeDecision {
    /// Selected mode.
    pub mode: IntraMode,
    /// RD cost.
    pub cost: f64,
    /// Distortion.
    pub distortion: f64,
    /// Rate in bits.
    pub rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mode_optimizer_creation() {
        let config = OptimizerConfig::default();
        let optimizer =
            ModeOptimizer::new(&config).expect("mode optimizer creation should succeed");
        assert!(!optimizer.full_rdo); // Medium level doesn't use full RDO
    }

    #[test]
    fn test_candidate_modes_fast() {
        let mut config = OptimizerConfig::default();
        config.level = crate::OptimizationLevel::Fast;
        let optimizer =
            ModeOptimizer::new(&config).expect("mode optimizer creation should succeed");
        let modes = optimizer.candidate_modes();
        assert!(modes.len() <= 6); // Reduced set
    }

    #[test]
    fn test_candidate_modes_slow() {
        let mut config = OptimizerConfig::default();
        config.level = crate::OptimizationLevel::Slow;
        let optimizer =
            ModeOptimizer::new(&config).expect("mode optimizer creation should succeed");
        let modes = optimizer.candidate_modes();
        assert!(modes.len() > 6); // Full set with directional modes
    }

    #[test]
    fn test_intra_mode_types() {
        let dc = IntraMode::Dc;
        let dir = IntraMode::Directional(45);
        assert_eq!(dc, IntraMode::Dc);
        assert_eq!(dir, IntraMode::Directional(45));
    }
}
