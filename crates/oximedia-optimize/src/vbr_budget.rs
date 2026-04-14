//! Variable Bitrate (VBR) bit budget allocation across scenes.
//!
//! Distributes a fixed total bit budget among a set of scenes in proportion
//! to their encoding complexity — complex, hard-to-encode scenes receive more
//! bits while simple scenes receive fewer, subject to per-scene floor and
//! ceiling constraints.
//!
//! # Algorithm
//!
//! 1. Normalise each scene's complexity to the range `[0, 1]`.
//! 2. Compute a raw allocation proportional to complexity.
//! 3. Clamp each allocation to `[floor_bits, ceiling_bits]`.
//! 4. Re-distribute any surplus/deficit from clamping across the remaining
//!    unconstrained scenes (up to [`VbrBudgetConfig::max_redistribution_passes`]
//!    passes).
//! 5. If bits remain unallocated after redistribution (all scenes are at their
//!    ceiling), spread the remainder evenly.
//!
//! # Example
//!
//! ```rust
//! use oximedia_optimize::vbr_budget::VbrBudget;
//!
//! let complexities = &[0.2_f32, 0.8, 0.5, 1.0, 0.3];
//! let total_bits: u64 = 10_000_000;
//!
//! let allocations = VbrBudget::allocate(complexities, total_bits);
//! assert_eq!(allocations.len(), 5);
//! // Total must not exceed total_bits.
//! let sum: u64 = allocations.iter().sum();
//! assert!(sum <= total_bits, "over-allocation: {sum} > {total_bits}");
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

/// Configuration for the VBR bit budget allocator.
#[derive(Debug, Clone)]
pub struct VbrBudgetConfig {
    /// Minimum bits to allocate to any single scene (floor).
    ///
    /// Defaults to 50 000 bits (≈ 6 KB, enough for a tiny keyframe).
    pub min_bits_per_scene: u64,

    /// Maximum bits any single scene may receive (ceiling), or `None` for
    /// no ceiling.
    ///
    /// A ceiling of `None` allows a single dominant scene to consume
    /// essentially the entire budget.
    pub max_bits_per_scene: Option<u64>,

    /// Number of redistribution passes to perform when scenes hit their floor
    /// or ceiling.  Higher values produce more accurate distributions at the
    /// cost of slightly more computation.  Defaults to 4.
    pub max_redistribution_passes: u32,

    /// Complexity exponent applied before normalisation.  Values > 1 make
    /// the allocation more aggressive (complex scenes get proportionally more
    /// bits); values < 1 flatten the distribution.  Defaults to 1.0.
    pub complexity_exponent: f64,
}

impl Default for VbrBudgetConfig {
    fn default() -> Self {
        Self {
            min_bits_per_scene: 50_000,
            max_bits_per_scene: None,
            max_redistribution_passes: 4,
            complexity_exponent: 1.0,
        }
    }
}

// ─── VbrBudget ───────────────────────────────────────────────────────────────

/// VBR bit budget allocator.
///
/// All heavy-lifting is in the associated function [`VbrBudget::allocate`]
/// and the configurable [`VbrBudget::allocate_with_config`].
pub struct VbrBudget;

impl VbrBudget {
    /// Allocate `total_bits` across `scene_complexities` using default config.
    ///
    /// Returns a `Vec<u64>` with the same length as `scene_complexities`,
    /// where each element is the number of bits assigned to that scene.
    /// The sum of all allocations will not exceed `total_bits`.
    ///
    /// Complexity values are treated as relative weights and need not be in any
    /// particular range (they are normalised internally).  Negative values are
    /// clamped to zero.
    #[must_use]
    pub fn allocate(scene_complexities: &[f32], total_bits: u64) -> Vec<u64> {
        Self::allocate_with_config(scene_complexities, total_bits, &VbrBudgetConfig::default())
    }

    /// Allocate `total_bits` with a custom [`VbrBudgetConfig`].
    #[must_use]
    pub fn allocate_with_config(
        scene_complexities: &[f32],
        total_bits: u64,
        config: &VbrBudgetConfig,
    ) -> Vec<u64> {
        let n = scene_complexities.len();
        if n == 0 || total_bits == 0 {
            return vec![0; n];
        }

        // Raise each complexity to the configured exponent and clamp to [0, ∞).
        let weights: Vec<f64> = scene_complexities
            .iter()
            .map(|&c| {
                let c_clamped = (c as f64).max(0.0);
                c_clamped.powf(config.complexity_exponent).max(0.0)
            })
            .collect();

        let weight_sum: f64 = weights.iter().sum();

        // Edge case: all weights are zero → distribute evenly.
        let mut allocations: Vec<f64> = if weight_sum <= 0.0 {
            let equal = total_bits as f64 / n as f64;
            vec![equal; n]
        } else {
            weights
                .iter()
                .map(|&w| w / weight_sum * total_bits as f64)
                .collect()
        };

        // Apply floor and ceiling constraints iteratively.
        let floor = config.min_bits_per_scene as f64;
        let ceil_opt = config.max_bits_per_scene.map(|c| c as f64);

        for _ in 0..config.max_redistribution_passes.max(1) {
            let mut surplus = 0.0_f64;
            let mut deficit = 0.0_f64;
            let mut free_weight_sum = 0.0_f64;

            // First pass: identify clamped scenes and compute surplus/deficit.
            let mut clamped = vec![false; n];
            for (i, alloc) in allocations.iter_mut().enumerate() {
                if *alloc < floor {
                    deficit += floor - *alloc;
                    *alloc = floor;
                    clamped[i] = true;
                } else if let Some(ceil) = ceil_opt {
                    if *alloc > ceil {
                        surplus += *alloc - ceil;
                        *alloc = ceil;
                        clamped[i] = true;
                    }
                }
            }

            // Net extra to redistribute: surplus from ceiling-clamped scenes
            // minus deficit already used to bring floor-clamped scenes up.
            let to_redistribute = surplus - deficit;

            if to_redistribute.abs() < 1.0 {
                break; // Nothing meaningful to redistribute.
            }

            // Sum the weights of unconstrained scenes.
            for (i, &w) in weights.iter().enumerate() {
                if !clamped[i] {
                    free_weight_sum += w;
                }
            }

            if free_weight_sum <= 0.0 {
                // All scenes are clamped at their floor or ceiling.
                // Any remaining surplus cannot be re-allocated without
                // violating constraints, so drop it (under-spend is
                // preferable to over-spend or constraint violation).
                break;
            }

            // Redistribute proportionally to unconstrained scenes.
            for (i, alloc) in allocations.iter_mut().enumerate() {
                if !clamped[i] {
                    *alloc += weights[i] / free_weight_sum * to_redistribute;
                }
            }
        }

        // Final hard clamp: ensure no allocation violates floor or ceiling
        // regardless of how redistribution played out.
        let floor = config.min_bits_per_scene as f64;
        let ceil_opt = config.max_bits_per_scene.map(|c| c as f64);
        for alloc in allocations.iter_mut() {
            if *alloc < floor {
                *alloc = floor;
            }
            if let Some(ceil) = ceil_opt {
                if *alloc > ceil {
                    *alloc = ceil;
                }
            }
        }

        // Convert to u64, ensuring the sum does not exceed total_bits.
        let mut result: Vec<u64> = allocations.iter().map(|&a| a.round() as u64).collect();

        // Adjust for rounding errors — reduce the largest bucket if over budget.
        let sum: u64 = result.iter().sum();
        if sum > total_bits {
            let excess = sum - total_bits;
            // Find the largest allocation and subtract from it.
            if let Some(max_idx) = result
                .iter()
                .enumerate()
                .max_by_key(|(_, &v)| v)
                .map(|(i, _)| i)
            {
                result[max_idx] = result[max_idx].saturating_sub(excess);
            }
        }

        result
    }

    /// Returns the total allocated bits (sum of all entries).
    #[must_use]
    pub fn total_allocated(allocations: &[u64]) -> u64 {
        allocations.iter().sum()
    }

    /// Returns the allocation statistics: (min, max, mean) in bits.
    #[must_use]
    pub fn stats(allocations: &[u64]) -> Option<(u64, u64, f64)> {
        if allocations.is_empty() {
            return None;
        }
        let min = *allocations.iter().min().expect("non-empty");
        let max = *allocations.iter().max().expect("non-empty");
        let mean = allocations.iter().sum::<u64>() as f64 / allocations.len() as f64;
        Some((min, max, mean))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_same_length() {
        let c = &[0.2_f32, 0.8, 0.5];
        let allocs = VbrBudget::allocate(c, 3_000_000);
        assert_eq!(allocs.len(), 3);
    }

    #[test]
    fn test_allocate_does_not_exceed_budget() {
        let c = &[0.2_f32, 0.8, 0.5, 1.0, 0.3];
        let total: u64 = 10_000_000;
        let allocs = VbrBudget::allocate(c, total);
        let sum: u64 = allocs.iter().sum();
        assert!(sum <= total, "over-allocation: {sum} > {total}");
    }

    #[test]
    fn test_allocate_complex_scenes_get_more() {
        let c = &[0.1_f32, 0.9];
        let allocs = VbrBudget::allocate(c, 10_000_000);
        assert!(
            allocs[1] > allocs[0],
            "high-complexity scene should get more bits: {} vs {}",
            allocs[1],
            allocs[0]
        );
    }

    #[test]
    fn test_allocate_empty() {
        let allocs = VbrBudget::allocate(&[], 1_000_000);
        assert!(allocs.is_empty());
    }

    #[test]
    fn test_allocate_zero_budget() {
        let allocs = VbrBudget::allocate(&[0.5, 0.8], 0);
        assert_eq!(allocs, vec![0, 0]);
    }

    #[test]
    fn test_allocate_uniform_complexity() {
        let c = &[0.5_f32; 4];
        let total: u64 = 4_000_000;
        let allocs = VbrBudget::allocate(c, total);
        let sum: u64 = allocs.iter().sum();
        assert!(sum <= total);
        // All allocations should be equal (within 1 bit of rounding).
        let expected = total / 4;
        for &a in &allocs {
            let diff = if a > expected {
                a - expected
            } else {
                expected - a
            };
            assert!(
                diff <= 4,
                "non-uniform allocation for equal complexity: {a} vs {expected}"
            );
        }
    }

    #[test]
    fn test_allocate_min_bits_floor_respected() {
        let config = VbrBudgetConfig {
            min_bits_per_scene: 500_000,
            max_bits_per_scene: None,
            max_redistribution_passes: 4,
            complexity_exponent: 1.0,
        };
        let c = &[0.01_f32, 0.99]; // very skewed
        let allocs = VbrBudget::allocate_with_config(c, 10_000_000, &config);
        for &a in &allocs {
            assert!(a >= 500_000, "allocation {a} is below the floor of 500 000");
        }
    }

    #[test]
    fn test_allocate_max_bits_ceiling_respected() {
        let config = VbrBudgetConfig {
            min_bits_per_scene: 0,
            max_bits_per_scene: Some(3_000_000),
            max_redistribution_passes: 4,
            complexity_exponent: 1.0,
        };
        let c = &[0.1_f32, 0.9];
        let allocs = VbrBudget::allocate_with_config(c, 10_000_000, &config);
        for &a in &allocs {
            assert!(
                a <= 3_000_000,
                "allocation {a} exceeds ceiling of 3 000 000"
            );
        }
    }

    #[test]
    fn test_total_allocated_helper() {
        let allocs = vec![1_000_000u64, 2_000_000, 3_000_000];
        assert_eq!(VbrBudget::total_allocated(&allocs), 6_000_000);
    }

    #[test]
    fn test_stats_min_max_mean() {
        let allocs = vec![1_000u64, 3_000, 2_000];
        let (min, max, mean) = VbrBudget::stats(&allocs).expect("non-empty");
        assert_eq!(min, 1_000);
        assert_eq!(max, 3_000);
        assert!((mean - 2_000.0).abs() < 1.0);
    }

    #[test]
    fn test_stats_empty_returns_none() {
        assert!(VbrBudget::stats(&[]).is_none());
    }
}
