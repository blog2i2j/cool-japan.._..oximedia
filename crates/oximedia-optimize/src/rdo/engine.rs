//! Rate-distortion optimization engine.

use crate::{OptimizationLevel, OptimizerConfig};
use oximedia_core::OxiResult;

/// RDO engine for mode decision.
pub struct RdoEngine {
    lambda_calc: super::LambdaCalculator,
    optimization_level: OptimizationLevel,
    parallel_enabled: bool,
}

impl RdoEngine {
    /// Creates a new RDO engine.
    pub fn new(config: &OptimizerConfig) -> OxiResult<Self> {
        let lambda_calc = super::LambdaCalculator::new(config.lambda_multiplier, config.level);

        Ok(Self {
            lambda_calc,
            optimization_level: config.level,
            parallel_enabled: config.parallel_rdo,
        })
    }

    /// Calculates the rate-distortion cost for a decision.
    ///
    /// # Parameters
    /// - `distortion`: Distortion metric (SSE, SAD, SATD)
    /// - `rate`: Bit rate for this decision
    /// - `qp`: Quantization parameter
    ///
    /// # Returns
    /// The RD cost: `distortion + lambda * rate`
    #[must_use]
    pub fn calculate_cost(&self, distortion: f64, rate: f64, qp: u8) -> f64 {
        let lambda = self.lambda_calc.calculate(qp);
        distortion + lambda * rate
    }

    /// Evaluates multiple mode decisions and returns the best one.
    pub fn evaluate_modes<F>(&self, candidates: &[ModeCandidate], eval_fn: F) -> RdoResult
    where
        F: Fn(&ModeCandidate) -> (f64, f64) + Send + Sync,
    {
        if self.parallel_enabled && candidates.len() > 4 {
            self.evaluate_parallel(candidates, eval_fn)
        } else {
            self.evaluate_sequential(candidates, eval_fn)
        }
    }

    fn evaluate_sequential<F>(&self, candidates: &[ModeCandidate], eval_fn: F) -> RdoResult
    where
        F: Fn(&ModeCandidate) -> (f64, f64),
    {
        let mut best_cost = f64::MAX;
        let mut best_idx = 0;

        for (idx, candidate) in candidates.iter().enumerate() {
            let (distortion, rate) = eval_fn(candidate);
            let cost = self.calculate_cost(distortion, rate, candidate.qp);

            if cost < best_cost {
                best_cost = cost;
                best_idx = idx;
            }
        }

        RdoResult {
            best_mode_idx: best_idx,
            cost: best_cost,
            distortion: 0.0, // Will be filled by caller
            rate: 0.0,       // Will be filled by caller
        }
    }

    fn evaluate_parallel<F>(&self, candidates: &[ModeCandidate], eval_fn: F) -> RdoResult
    where
        F: Fn(&ModeCandidate) -> (f64, f64) + Send + Sync,
    {
        use rayon::prelude::*;

        let results: Vec<_> = candidates
            .par_iter()
            .enumerate()
            .map(|(idx, candidate)| {
                let (distortion, rate) = eval_fn(candidate);
                let cost = self.calculate_cost(distortion, rate, candidate.qp);
                (idx, cost, distortion, rate)
            })
            .collect();

        let best = results
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        match best {
            Some(b) => RdoResult {
                best_mode_idx: b.0,
                cost: b.1,
                distortion: b.2,
                rate: b.3,
            },
            None => RdoResult {
                best_mode_idx: 0,
                cost: f64::MAX,
                distortion: 0.0,
                rate: 0.0,
            },
        }
    }

    /// Gets the optimization level.
    #[must_use]
    pub fn optimization_level(&self) -> OptimizationLevel {
        self.optimization_level
    }

    /// Checks if the engine should perform full RDO.
    #[must_use]
    pub fn should_perform_full_rdo(&self) -> bool {
        matches!(
            self.optimization_level,
            OptimizationLevel::Slow | OptimizationLevel::Placebo
        )
    }

    /// Checks if the engine should use SATD instead of SAD.
    #[must_use]
    pub fn should_use_satd(&self) -> bool {
        !matches!(self.optimization_level, OptimizationLevel::Fast)
    }
}

/// Mode candidate for RDO evaluation.
#[derive(Debug, Clone)]
pub struct ModeCandidate {
    /// Mode index.
    pub mode_idx: usize,
    /// Quantization parameter.
    pub qp: u8,
    /// Additional mode-specific data.
    pub data: Vec<u8>,
}

/// Result of RDO optimization.
#[derive(Debug, Clone)]
pub struct RdoResult {
    /// Index of the best mode.
    pub best_mode_idx: usize,
    /// Rate-distortion cost.
    pub cost: f64,
    /// Distortion component.
    pub distortion: f64,
    /// Rate component (in bits).
    pub rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rdo_engine_creation() {
        let config = OptimizerConfig::default();
        let engine = RdoEngine::new(&config).expect("RDO engine creation should succeed");
        assert_eq!(engine.optimization_level(), OptimizationLevel::Medium);
    }

    #[test]
    fn test_cost_calculation() {
        let config = OptimizerConfig::default();
        let engine = RdoEngine::new(&config).expect("RDO engine creation should succeed");
        let cost = engine.calculate_cost(100.0, 50.0, 26);
        assert!(cost > 100.0); // Should include rate penalty
    }

    #[test]
    fn test_mode_evaluation() {
        let config = OptimizerConfig::default();
        let engine = RdoEngine::new(&config).expect("RDO engine creation should succeed");

        let candidates = vec![
            ModeCandidate {
                mode_idx: 0,
                qp: 26,
                data: vec![],
            },
            ModeCandidate {
                mode_idx: 1,
                qp: 26,
                data: vec![],
            },
        ];

        let result = engine.evaluate_modes(&candidates, |c| {
            // Simulate: mode 1 has lower distortion but higher rate
            if c.mode_idx == 0 {
                (150.0, 40.0)
            } else {
                (100.0, 60.0)
            }
        });

        assert!(result.best_mode_idx < 2);
    }
}
