//! Adaptive quantization module.
//!
//! Variance-based and psychovisual adaptive quantization modes.

pub mod psycho;
pub mod variance;

pub use psycho::{PsychoAq, PsychoAqParams};
pub use variance::{VarianceAq, VarianceMap};

use crate::OptimizerConfig;
use oximedia_core::OxiResult;

/// Adaptive quantization modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AqMode {
    /// No AQ.
    None,
    /// Variance-based AQ.
    #[default]
    Variance,
    /// Psychovisual AQ.
    Psychovisual,
    /// Combined variance and psychovisual.
    Combined,
}

/// AQ result with QP offset.
#[derive(Debug, Clone, Copy)]
pub struct AqResult {
    /// QP offset to apply.
    pub qp_offset: i8,
    /// Variance metric.
    pub variance: f64,
    /// Psychovisual weight.
    pub psycho_weight: f64,
}

impl Default for AqResult {
    fn default() -> Self {
        Self {
            qp_offset: 0,
            variance: 0.0,
            psycho_weight: 1.0,
        }
    }
}

/// Adaptive quantization engine.
pub struct AqEngine {
    mode: AqMode,
    strength: f64,
    variance_aq: VarianceAq,
    psycho_aq: PsychoAq,
}

impl AqEngine {
    /// Creates a new AQ engine.
    pub fn new(config: &OptimizerConfig) -> OxiResult<Self> {
        let mode = if !config.enable_aq {
            AqMode::None
        } else if config.enable_psychovisual {
            AqMode::Combined
        } else {
            AqMode::Variance
        };

        Ok(Self {
            mode,
            strength: 1.0,
            variance_aq: VarianceAq::default(),
            psycho_aq: PsychoAq::default(),
        })
    }

    /// Calculates AQ for a block.
    #[allow(dead_code)]
    #[must_use]
    pub fn calculate_aq(&self, pixels: &[u8], width: usize) -> AqResult {
        match self.mode {
            AqMode::None => AqResult::default(),
            AqMode::Variance => {
                let variance = self.variance_aq.calculate_variance(pixels);
                let qp_offset = self
                    .variance_aq
                    .variance_to_qp_offset(variance, self.strength);
                AqResult {
                    qp_offset,
                    variance,
                    psycho_weight: 1.0,
                }
            }
            AqMode::Psychovisual => {
                let psycho_weight = self.psycho_aq.calculate_weight(pixels, width);
                let qp_offset = self
                    .psycho_aq
                    .weight_to_qp_offset(psycho_weight, self.strength);
                AqResult {
                    qp_offset,
                    variance: 0.0,
                    psycho_weight,
                }
            }
            AqMode::Combined => {
                let variance = self.variance_aq.calculate_variance(pixels);
                let psycho_weight = self.psycho_aq.calculate_weight(pixels, width);

                let var_offset = self
                    .variance_aq
                    .variance_to_qp_offset(variance, self.strength);
                let psycho_offset = self
                    .psycho_aq
                    .weight_to_qp_offset(psycho_weight, self.strength);

                // Average the offsets
                let qp_offset = ((i16::from(var_offset) + i16::from(psycho_offset)) / 2) as i8;

                AqResult {
                    qp_offset,
                    variance,
                    psycho_weight,
                }
            }
        }
    }

    /// Gets the AQ mode.
    #[must_use]
    pub fn mode(&self) -> AqMode {
        self.mode
    }

    /// Sets the AQ strength.
    pub fn set_strength(&mut self, strength: f64) {
        self.strength = strength.clamp(0.0, 2.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aq_engine_creation() {
        let config = OptimizerConfig::default();
        let engine = AqEngine::new(&config).expect("AQ engine creation should succeed");
        assert_eq!(engine.mode(), AqMode::Combined); // Default has psycho enabled
    }

    #[test]
    fn test_aq_modes() {
        assert_ne!(AqMode::None, AqMode::Variance);
        assert_eq!(AqMode::Psychovisual, AqMode::Psychovisual);
    }

    #[test]
    fn test_aq_result_default() {
        let result = AqResult::default();
        assert_eq!(result.qp_offset, 0);
        assert_eq!(result.variance, 0.0);
        assert_eq!(result.psycho_weight, 1.0);
    }

    #[test]
    fn test_set_strength() {
        let config = OptimizerConfig::default();
        let mut engine = AqEngine::new(&config).expect("AQ engine creation should succeed");
        engine.set_strength(1.5);
        assert_eq!(engine.strength, 1.5);

        engine.set_strength(3.0); // Should clamp to 2.0
        assert_eq!(engine.strength, 2.0);
    }
}
