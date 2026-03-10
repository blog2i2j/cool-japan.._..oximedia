//! Advanced motion search algorithms.

use crate::{OptimizationLevel, OptimizerConfig};
use oximedia_core::OxiResult;

/// Motion vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MotionVector {
    /// Horizontal component (in quarter-pel units).
    pub x: i16,
    /// Vertical component (in quarter-pel units).
    pub y: i16,
}

impl MotionVector {
    /// Creates a new motion vector.
    #[must_use]
    pub const fn new(x: i16, y: i16) -> Self {
        Self { x, y }
    }

    /// Zero motion vector.
    #[must_use]
    pub const fn zero() -> Self {
        Self::new(0, 0)
    }
}

/// Motion search algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchAlgorithm {
    /// Full search (exhaustive).
    Full,
    /// Diamond search.
    Diamond,
    /// Hexagon search.
    Hexagon,
    /// Test Zone Search.
    TzSearch,
    /// Enhanced Predictive Zonal Search.
    Epzs,
    /// Uneven Multi-Hexagon.
    Umh,
}

/// Motion optimizer for advanced motion estimation.
pub struct MotionOptimizer {
    algorithm: SearchAlgorithm,
    search_range: i16,
    #[allow(dead_code)]
    subpel_enabled: bool,
}

impl MotionOptimizer {
    /// Creates a new motion optimizer.
    pub fn new(config: &OptimizerConfig) -> OxiResult<Self> {
        let (algorithm, search_range) = match config.level {
            OptimizationLevel::Fast => (SearchAlgorithm::Diamond, 16),
            OptimizationLevel::Medium => (SearchAlgorithm::Hexagon, 32),
            OptimizationLevel::Slow => (SearchAlgorithm::TzSearch, 64),
            OptimizationLevel::Placebo => (SearchAlgorithm::Umh, 128),
        };

        Ok(Self {
            algorithm,
            search_range,
            subpel_enabled: config.level != OptimizationLevel::Fast,
        })
    }

    /// Performs motion search.
    #[allow(dead_code)]
    #[must_use]
    pub fn search(
        &self,
        src: &[u8],
        reference: &[u8],
        width: usize,
        height: usize,
        predictor: MotionVector,
    ) -> MotionSearchResult {
        match self.algorithm {
            SearchAlgorithm::Full => self.full_search(src, reference, width, height),
            SearchAlgorithm::Diamond => {
                self.diamond_search(src, reference, width, height, predictor)
            }
            SearchAlgorithm::Hexagon => {
                self.hexagon_search(src, reference, width, height, predictor)
            }
            SearchAlgorithm::TzSearch => self.tz_search(src, reference, width, height, predictor),
            SearchAlgorithm::Epzs => self.epzs_search(src, reference, width, height, predictor),
            SearchAlgorithm::Umh => self.umh_search(src, reference, width, height, predictor),
        }
    }

    fn full_search(
        &self,
        src: &[u8],
        reference: &[u8],
        _width: usize,
        _height: usize,
    ) -> MotionSearchResult {
        // Simplified full search
        let mut best_mv = MotionVector::zero();
        let mut best_cost = self.calculate_cost(src, reference, best_mv);

        for y in -self.search_range..=self.search_range {
            for x in -self.search_range..=self.search_range {
                let mv = MotionVector::new(x * 4, y * 4); // Convert to qpel
                let cost = self.calculate_cost(src, reference, mv);
                if cost < best_cost {
                    best_cost = cost;
                    best_mv = mv;
                }
            }
        }

        MotionSearchResult {
            mv: best_mv,
            cost: best_cost,
            iterations: ((2 * self.search_range + 1) * (2 * self.search_range + 1)) as usize,
        }
    }

    fn diamond_search(
        &self,
        src: &[u8],
        reference: &[u8],
        _width: usize,
        _height: usize,
        predictor: MotionVector,
    ) -> MotionSearchResult {
        let mut best_mv = predictor;
        let mut best_cost = self.calculate_cost(src, reference, best_mv);
        let mut iterations = 0;

        // Large diamond pattern
        let large_diamond = [
            (0, -2),
            (-1, -1),
            (1, -1),
            (-2, 0),
            (2, 0),
            (-1, 1),
            (1, 1),
            (0, 2),
        ];

        loop {
            let mut improved = false;

            for &(dx, dy) in &large_diamond {
                let mv = MotionVector::new(best_mv.x + dx * 4, best_mv.y + dy * 4);
                let cost = self.calculate_cost(src, reference, mv);
                iterations += 1;

                if cost < best_cost {
                    best_cost = cost;
                    best_mv = mv;
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }

        MotionSearchResult {
            mv: best_mv,
            cost: best_cost,
            iterations,
        }
    }

    fn hexagon_search(
        &self,
        src: &[u8],
        reference: &[u8],
        _width: usize,
        _height: usize,
        predictor: MotionVector,
    ) -> MotionSearchResult {
        let mut best_mv = predictor;
        let mut best_cost = self.calculate_cost(src, reference, best_mv);
        let mut iterations = 0;

        // Hexagon pattern
        let hexagon = [(0, -2), (-2, -1), (2, -1), (-2, 1), (2, 1), (0, 2)];

        loop {
            let mut improved = false;

            for &(dx, dy) in &hexagon {
                let mv = MotionVector::new(best_mv.x + dx * 4, best_mv.y + dy * 4);
                let cost = self.calculate_cost(src, reference, mv);
                iterations += 1;

                if cost < best_cost {
                    best_cost = cost;
                    best_mv = mv;
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }

        MotionSearchResult {
            mv: best_mv,
            cost: best_cost,
            iterations,
        }
    }

    fn tz_search(
        &self,
        src: &[u8],
        reference: &[u8],
        width: usize,
        height: usize,
        predictor: MotionVector,
    ) -> MotionSearchResult {
        // Test Zone Search (simplified version)
        // Start with diamond search
        let mut result = self.diamond_search(src, reference, width, height, predictor);

        // Refine with smaller pattern
        let small_diamond = [(0, -1), (-1, 0), (1, 0), (0, 1)];
        let mut improved = true;

        while improved {
            improved = false;
            for &(dx, dy) in &small_diamond {
                let mv = MotionVector::new(result.mv.x + dx * 4, result.mv.y + dy * 4);
                let cost = self.calculate_cost(src, reference, mv);
                result.iterations += 1;

                if cost < result.cost {
                    result.cost = cost;
                    result.mv = mv;
                    improved = true;
                }
            }
        }

        result
    }

    fn epzs_search(
        &self,
        src: &[u8],
        reference: &[u8],
        width: usize,
        height: usize,
        predictor: MotionVector,
    ) -> MotionSearchResult {
        // Enhanced Predictive Zonal Search (simplified)
        // Use predictor-based early termination
        let best_mv = predictor;
        let best_cost = self.calculate_cost(src, reference, best_mv);
        let iterations = 1;

        // Early termination if predictor is good enough
        if best_cost < 100.0 {
            return MotionSearchResult {
                mv: best_mv,
                cost: best_cost,
                iterations,
            };
        }

        // Otherwise, use diamond search
        self.diamond_search(src, reference, width, height, predictor)
    }

    fn umh_search(
        &self,
        src: &[u8],
        reference: &[u8],
        width: usize,
        height: usize,
        predictor: MotionVector,
    ) -> MotionSearchResult {
        // Uneven Multi-Hexagon (simplified)
        // Start with hexagon search
        let mut result = self.hexagon_search(src, reference, width, height, predictor);

        // Add additional refinement passes
        for _ in 0..2 {
            let small_hex = [(0, -1), (-1, 0), (1, 0), (0, 1)];
            for &(dx, dy) in &small_hex {
                let mv = MotionVector::new(result.mv.x + dx * 2, result.mv.y + dy * 2);
                let cost = self.calculate_cost(src, reference, mv);
                result.iterations += 1;

                if cost < result.cost {
                    result.cost = cost;
                    result.mv = mv;
                }
            }
        }

        result
    }

    fn calculate_cost(&self, src: &[u8], _reference: &[u8], _mv: MotionVector) -> f64 {
        // Simplified cost calculation (would use actual SAD/SATD in production)
        // For now, return a dummy cost based on src
        src.iter().map(|&x| f64::from(x)).sum::<f64>() / src.len() as f64
    }
}

/// Motion search result.
#[derive(Debug, Clone, Copy)]
pub struct MotionSearchResult {
    /// Best motion vector.
    pub mv: MotionVector,
    /// Search cost.
    pub cost: f64,
    /// Number of iterations.
    pub iterations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_vector() {
        let mv = MotionVector::new(16, -8);
        assert_eq!(mv.x, 16);
        assert_eq!(mv.y, -8);

        let zero = MotionVector::zero();
        assert_eq!(zero.x, 0);
        assert_eq!(zero.y, 0);
    }

    #[test]
    fn test_motion_optimizer_creation() {
        let config = OptimizerConfig::default();
        let optimizer =
            MotionOptimizer::new(&config).expect("motion optimizer creation should succeed");
        assert_eq!(optimizer.algorithm, SearchAlgorithm::Hexagon);
    }

    #[test]
    fn test_search_algorithm_selection() {
        let mut config = OptimizerConfig::default();

        config.level = OptimizationLevel::Fast;
        let opt_fast =
            MotionOptimizer::new(&config).expect("motion optimizer creation should succeed");
        assert_eq!(opt_fast.algorithm, SearchAlgorithm::Diamond);

        config.level = OptimizationLevel::Placebo;
        let opt_placebo =
            MotionOptimizer::new(&config).expect("motion optimizer creation should succeed");
        assert_eq!(opt_placebo.algorithm, SearchAlgorithm::Umh);
    }
}
