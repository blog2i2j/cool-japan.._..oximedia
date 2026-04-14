//! Quality budget allocation across scenes or encode segments.
//!
//! [`QualityBudget`] distributes a fixed total quality budget across scenes
//! proportionally to each scene's complexity.  More complex scenes (fast
//! motion, high detail) receive a larger share of the budget; simpler scenes
//! (static shots, talking heads) receive less.
//!
//! # Budget Allocation Algorithm
//!
//! Given a total budget `T` and `n` scenes with raw complexity scores
//! `c_0 … c_{n-1}`:
//!
//! 1. Normalise: `w_i = c_i / Σ c_i`  (falls back to uniform when all
//!    complexities are zero or the list is empty)
//! 2. Allocate: `alloc_i = w_i * T`
//!
//! All allocated values are non-negative and their sum equals `total` (within
//! floating-point rounding error).

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// QualityBudget
// ---------------------------------------------------------------------------

/// Distributes a quality budget proportionally to scene complexity weights.
///
/// # Example
///
/// ```
/// use oximedia_quality::budget::QualityBudget;
///
/// let budget = QualityBudget::new(100.0);
/// let allocs = budget.allocate(&[1.0, 3.0]);
/// // Scene 0 gets 25, scene 1 gets 75
/// assert!((allocs[0] - 25.0).abs() < 1e-6);
/// assert!((allocs[1] - 75.0).abs() < 1e-6);
/// ```
pub struct QualityBudget {
    total: f64,
}

impl QualityBudget {
    /// Creates a new budget with the given total.
    ///
    /// If `total` is negative it is clamped to `0.0`.
    #[must_use]
    pub fn new(total: f64) -> Self {
        Self {
            total: total.max(0.0),
        }
    }

    /// Returns the configured total budget.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.total
    }

    /// Allocates the budget across scenes proportional to `scene_complexity`.
    ///
    /// * When `scene_complexity` is empty, returns an empty vector.
    /// * When all complexity values are zero (or negative), the budget is
    ///   distributed uniformly.
    /// * Negative complexity values are clamped to `0.0`.
    ///
    /// The returned vector has the same length as `scene_complexity`.
    #[must_use]
    pub fn allocate(&self, scene_complexity: &[f32]) -> Vec<f64> {
        if scene_complexity.is_empty() {
            return Vec::new();
        }

        let n = scene_complexity.len();

        // Clamp negatives; gather positive weights
        let weights: Vec<f64> = scene_complexity
            .iter()
            .map(|&c| c.max(0.0) as f64)
            .collect();

        let weight_sum: f64 = weights.iter().sum();

        if weight_sum < 1e-12 {
            // Uniform allocation when all complexities are zero
            let per_scene = self.total / n as f64;
            return vec![per_scene; n];
        }

        weights
            .iter()
            .map(|&w| w / weight_sum * self.total)
            .collect()
    }

    /// Allocates the budget and additionally enforces a per-scene minimum and
    /// maximum bound.
    ///
    /// After proportional allocation, any scene whose allocation falls below
    /// `min_per_scene` is raised to that floor, and any scene exceeding
    /// `max_per_scene` is capped.  The remaining budget (after enforcing
    /// floors) is redistributed proportionally among unclamped scenes in a
    /// single pass.
    ///
    /// This is a best-effort single-pass clamp; it does not iterate until
    /// convergence.
    #[must_use]
    pub fn allocate_bounded(
        &self,
        scene_complexity: &[f32],
        min_per_scene: f64,
        max_per_scene: f64,
    ) -> Vec<f64> {
        let mut allocs = self.allocate(scene_complexity);
        if allocs.is_empty() {
            return allocs;
        }

        // Apply floor and ceiling in a single pass.
        for v in &mut allocs {
            *v = v.clamp(min_per_scene, max_per_scene);
        }

        // Re-normalise so the total is preserved.
        let current_sum: f64 = allocs.iter().sum();
        if current_sum > 1e-12 {
            let scale = self.total / current_sum;
            for v in &mut allocs {
                *v *= scale;
                // Clamp again after rescaling to stay within bounds.
                *v = v.clamp(min_per_scene, max_per_scene);
            }
        }

        allocs
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    #[test]
    fn test_allocate_proportional_two_scenes() {
        let b = QualityBudget::new(100.0);
        let allocs = b.allocate(&[1.0, 3.0]);
        assert_eq!(allocs.len(), 2);
        assert!((allocs[0] - 25.0).abs() < EPS, "got {}", allocs[0]);
        assert!((allocs[1] - 75.0).abs() < EPS, "got {}", allocs[1]);
    }

    #[test]
    fn test_allocate_sums_to_total() {
        let b = QualityBudget::new(500.0);
        let complexities = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let allocs = b.allocate(&complexities);
        let sum: f64 = allocs.iter().sum();
        assert!((sum - 500.0).abs() < 1e-6, "sum should be 500, got {sum}");
    }

    #[test]
    fn test_allocate_empty_returns_empty() {
        let b = QualityBudget::new(100.0);
        assert!(b.allocate(&[]).is_empty());
    }

    #[test]
    fn test_allocate_all_zero_gives_uniform() {
        let b = QualityBudget::new(90.0);
        let allocs = b.allocate(&[0.0, 0.0, 0.0]);
        for a in &allocs {
            assert!((a - 30.0).abs() < EPS, "uniform: expected 30, got {a}");
        }
    }

    #[test]
    fn test_allocate_single_scene_gets_full_budget() {
        let b = QualityBudget::new(77.0);
        let allocs = b.allocate(&[5.0]);
        assert_eq!(allocs.len(), 1);
        assert!((allocs[0] - 77.0).abs() < EPS);
    }

    #[test]
    fn test_allocate_equal_complexity_uniform() {
        let b = QualityBudget::new(60.0);
        let allocs = b.allocate(&[2.0, 2.0, 2.0]);
        for a in &allocs {
            assert!(
                (a - 20.0).abs() < EPS,
                "equal weights: expected 20, got {a}"
            );
        }
    }

    #[test]
    fn test_allocate_negative_complexity_treated_as_zero() {
        let b = QualityBudget::new(100.0);
        // Negative scenes should be treated as zero-weight
        let allocs = b.allocate(&[-1.0, 0.0, 4.0]);
        // Only scene 2 has positive weight → gets all 100
        assert!((allocs[2] - 100.0).abs() < EPS, "got {}", allocs[2]);
        assert!(
            allocs[0].abs() < EPS,
            "negative scene: expected 0, got {}",
            allocs[0]
        );
    }

    #[test]
    fn test_allocate_negative_total_clamped() {
        let b = QualityBudget::new(-50.0);
        assert!((b.total()).abs() < EPS, "negative total should be 0");
        let allocs = b.allocate(&[1.0, 1.0]);
        for a in &allocs {
            assert!(a.abs() < EPS, "zero budget: expected 0, got {a}");
        }
    }

    #[test]
    fn test_allocate_bounded_no_underflow() {
        let b = QualityBudget::new(100.0);
        // Min floor of 10 — with uniform weights each scene gets 20, well above floor
        let allocs = b.allocate_bounded(&[1.0, 1.0, 1.0, 1.0, 1.0], 10.0, 50.0);
        assert_eq!(allocs.len(), 5);
        for a in &allocs {
            assert!(*a >= 10.0 - 1e-6, "floor violated: {a}");
            assert!(*a <= 50.0 + 1e-6, "ceiling violated: {a}");
        }
    }

    #[test]
    fn test_total_accessor() {
        let b = QualityBudget::new(250.0);
        assert!((b.total() - 250.0).abs() < EPS);
    }
}
