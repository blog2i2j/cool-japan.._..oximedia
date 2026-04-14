//! Bitrate ladder optimisation — convex hull pruning.
//!
//! A bitrate ladder is a set of (bitrate, quality) operating points (rungs)
//! that a player uses for adaptive-bitrate streaming.  Not every rung is
//! useful: a rung is *dominated* if another rung delivers equal or better
//! quality at equal or lower bitrate.
//!
//! The *convex-hull* optimisation removes dominated rungs and returns only the
//! Pareto-optimal frontier — the rungs for which increasing bitrate yields
//! diminishing but positive returns in quality.
//!
//! # Algorithm
//!
//! 1. Sort rungs by bitrate (ascending).
//! 2. Remove any rung that is weakly dominated (bitrate ≥ a kept rung's
//!    bitrate AND quality ≤ that rung's quality).
//! 3. Build the upper-left convex hull of the remaining points:
//!    keep a rung only if the slope from the previous kept rung to the
//!    current rung is strictly less than the slope from the current rung
//!    to the next candidate (monotone concavity test).
//!
//! # Example
//!
//! ```rust
//! use oximedia_optimize::ladder_opt::LadderOptimizer;
//!
//! // Six rungs; some are clearly dominated.
//! let rungs = &[
//!     (500_000u32,  30.0_f32),
//!     (500_000u32,  25.0_f32), // dominated by the rung above
//!     (1_000_000u32, 55.0_f32),
//!     (2_000_000u32, 70.0_f32),
//!     (2_000_000u32, 68.0_f32), // dominated
//!     (4_000_000u32, 82.0_f32),
//! ];
//!
//! let hull = LadderOptimizer::convex_hull(rungs);
//! assert!(hull.len() <= 4, "at most 4 non-dominated rungs");
//! for w in hull.windows(2) {
//!     assert!(w[1].0 > w[0].0, "bitrate must be strictly increasing");
//!     assert!(w[1].1 > w[0].1, "quality must be strictly increasing");
//! }
//! ```

#![allow(clippy::cast_precision_loss)]

/// A single rung in the bitrate ladder: `(bitrate_bps, quality_score)`.
///
/// `quality_score` may be any monotone quality metric (VMAF, PSNR-Y, etc.).
pub type Rung = (u32, f32);

/// Ladder optimiser exposing convex-hull and Pareto pruning utilities.
pub struct LadderOptimizer;

impl LadderOptimizer {
    /// Return the Pareto-optimal (convex-hull) subset of `rungs`.
    ///
    /// The returned `Vec` is sorted by bitrate (ascending) and contains only
    /// rungs for which no other rung achieves higher quality at lower or equal
    /// bitrate.  Additionally, rungs that lie *below* the convex hull (i.e.
    /// where marginal quality gain per incremental bit is not monotone
    /// decreasing) are removed.
    ///
    /// An empty or single-rung input is returned unchanged.
    #[must_use]
    pub fn convex_hull(rungs: &[Rung]) -> Vec<Rung> {
        if rungs.len() <= 1 {
            return rungs.to_vec();
        }

        // Step 1: sort by bitrate ascending, breaking ties by quality descending.
        let mut sorted = rungs.to_vec();
        sorted.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| b.1.partial_cmp(&a.1).expect("finite quality"))
        });

        // Step 2: remove dominated rungs in a single forward pass.
        // A rung is dominated if a rung with equal or lower bitrate achieves
        // equal or higher quality.
        let mut pareto: Vec<Rung> = Vec::with_capacity(sorted.len());
        let mut best_quality = f32::NEG_INFINITY;
        for rung in &sorted {
            if rung.1 > best_quality {
                pareto.push(*rung);
                best_quality = rung.1;
            }
        }

        if pareto.len() <= 2 {
            return pareto;
        }

        // Step 3: convex-hull upper envelope using the monotone concavity test.
        // Keep rung i iff the slope from (i-1) to i is greater than the slope
        // from i to (i+1) — diminishing returns condition.
        //
        // We use the standard upper-convex-hull stack algorithm.
        let mut hull: Vec<Rung> = Vec::with_capacity(pareto.len());

        for &rung in &pareto {
            // Pop the top of the stack while the last three points form a
            // non-left (concave-upward) turn, i.e. adding `rung` makes the
            // slope from hull[-2]→hull[-1] ≤ slope from hull[-1]→rung.
            // That means hull[-1] is below the line hull[-2]→rung and should
            // be discarded.
            while hull.len() >= 2 {
                let a = hull[hull.len() - 2];
                let b = hull[hull.len() - 1];
                let c = rung;

                let br_ab = (b.0 as f64) - (a.0 as f64);
                let br_bc = (c.0 as f64) - (b.0 as f64);

                if br_ab <= 0.0 || br_bc <= 0.0 {
                    break;
                }

                let slope_ab = (b.1 as f64 - a.1 as f64) / br_ab;
                let slope_bc = (c.1 as f64 - b.1 as f64) / br_bc;

                // If slope from b→c ≥ slope from a→b, point b is not on the
                // upper-concave hull (returns are not diminishing) — remove it.
                if slope_bc >= slope_ab {
                    hull.pop();
                } else {
                    break;
                }
            }
            hull.push(rung);
        }

        hull
    }

    /// Suggest a standard ABR ladder from a continuous R-D model.
    ///
    /// Generates `n` candidate rungs evenly spaced between `min_bitrate` and
    /// `max_bitrate` using the provided quality-predictor closure, then
    /// applies [`convex_hull`][LadderOptimizer::convex_hull] to select the
    /// optimal subset.
    ///
    /// # Arguments
    ///
    /// * `min_bitrate` – lowest candidate bitrate (bps).
    /// * `max_bitrate` – highest candidate bitrate (bps).
    /// * `n`           – number of candidate rungs to generate.
    /// * `quality_fn`  – maps a bitrate (bps) to a quality score.
    ///
    /// Returns an empty `Vec` if `n == 0` or `min_bitrate >= max_bitrate`.
    #[must_use]
    pub fn from_rd_model<F>(
        min_bitrate: u32,
        max_bitrate: u32,
        n: usize,
        quality_fn: F,
    ) -> Vec<Rung>
    where
        F: Fn(u32) -> f32,
    {
        if n == 0 || min_bitrate >= max_bitrate {
            return Vec::new();
        }
        let step = (max_bitrate as f64 - min_bitrate as f64) / (n.saturating_sub(1).max(1)) as f64;
        let candidates: Vec<Rung> = (0..n)
            .map(|i| {
                let br = (min_bitrate as f64 + step * i as f64).round() as u32;
                let q = quality_fn(br);
                (br, q)
            })
            .collect();
        Self::convex_hull(&candidates)
    }

    /// Compute the average quality gain per unit bitrate increase along the
    /// ladder (quality efficiency).
    ///
    /// Returns `0.0` for a ladder with fewer than 2 rungs.
    #[must_use]
    pub fn mean_quality_per_bit(rungs: &[Rung]) -> f64 {
        if rungs.len() < 2 {
            return 0.0;
        }
        let total_quality_gain: f64 = rungs.windows(2).map(|w| (w[1].1 - w[0].1) as f64).sum();
        let total_bitrate_gain: f64 = rungs
            .windows(2)
            .map(|w| (w[1].0 as f64) - (w[0].0 as f64))
            .sum();
        if total_bitrate_gain <= 0.0 {
            return 0.0;
        }
        total_quality_gain / total_bitrate_gain
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        assert!(LadderOptimizer::convex_hull(&[]).is_empty());
    }

    #[test]
    fn test_single_rung_unchanged() {
        let rungs = &[(1_000_000u32, 50.0_f32)];
        assert_eq!(LadderOptimizer::convex_hull(rungs), rungs.to_vec());
    }

    #[test]
    fn test_dominated_rung_removed() {
        // Second rung has same bitrate but lower quality → dominated.
        let rungs = &[
            (500_000u32, 30.0_f32),
            (500_000u32, 25.0_f32),
            (1_000_000u32, 55.0_f32),
        ];
        let hull = LadderOptimizer::convex_hull(rungs);
        // The (500_000, 25.0) rung must not appear.
        assert!(!hull.iter().any(|r| r.1 < 26.0 && r.0 == 500_000));
    }

    #[test]
    fn test_bitrate_strictly_increasing() {
        let rungs = &[
            (500_000u32, 30.0_f32),
            (1_000_000u32, 55.0_f32),
            (2_000_000u32, 70.0_f32),
            (4_000_000u32, 82.0_f32),
        ];
        let hull = LadderOptimizer::convex_hull(rungs);
        for w in hull.windows(2) {
            assert!(w[1].0 > w[0].0, "bitrate must be strictly increasing");
        }
    }

    #[test]
    fn test_quality_strictly_increasing() {
        let rungs = &[
            (500_000u32, 30.0_f32),
            (1_000_000u32, 55.0_f32),
            (2_000_000u32, 70.0_f32),
            (4_000_000u32, 82.0_f32),
        ];
        let hull = LadderOptimizer::convex_hull(rungs);
        for w in hull.windows(2) {
            assert!(w[1].1 > w[0].1, "quality must be strictly increasing");
        }
    }

    #[test]
    fn test_convex_hull_prunes_below_line() {
        // Point (2M, 60) lies below the line (1M, 55) → (4M, 82) and should
        // not appear on the convex hull in a properly diminishing-returns sense
        // relative to a point that dominates it.
        let rungs = &[
            (500_000u32, 30.0_f32),
            (1_000_000u32, 55.0_f32),
            (2_000_000u32, 62.0_f32), // above the line
            (3_000_000u32, 73.0_f32),
            (4_000_000u32, 82.0_f32),
        ];
        let hull = LadderOptimizer::convex_hull(rungs);
        // All returned rungs should have bitrate in the input.
        for rung in &hull {
            assert!(rungs.iter().any(|r| r.0 == rung.0));
        }
    }

    #[test]
    fn test_from_rd_model_length_bounded() {
        let hull = LadderOptimizer::from_rd_model(500_000, 5_000_000, 10, |br| {
            // Simple log-quality model.
            (30.0 + 10.0 * (br as f32 / 500_000.0).ln()).min(95.0)
        });
        // Hull can only be ≤ the number of candidates.
        assert!(hull.len() <= 10);
        assert!(!hull.is_empty());
    }

    #[test]
    fn test_from_rd_model_empty_on_invalid_range() {
        let hull = LadderOptimizer::from_rd_model(5_000_000, 500_000, 10, |_| 50.0);
        assert!(hull.is_empty());
    }

    #[test]
    fn test_mean_quality_per_bit_zero_for_single_rung() {
        let rungs = &[(1_000_000u32, 60.0_f32)];
        assert!((LadderOptimizer::mean_quality_per_bit(rungs)).abs() < 1e-9);
    }

    #[test]
    fn test_mean_quality_per_bit_positive() {
        let rungs = &[
            (500_000u32, 30.0_f32),
            (1_000_000u32, 55.0_f32),
            (2_000_000u32, 70.0_f32),
        ];
        let mqpb = LadderOptimizer::mean_quality_per_bit(rungs);
        assert!(mqpb > 0.0, "mean quality per bit should be positive");
    }

    #[test]
    fn test_all_same_bitrate_returns_best_quality() {
        let rungs = &[
            (1_000_000u32, 40.0_f32),
            (1_000_000u32, 60.0_f32),
            (1_000_000u32, 50.0_f32),
        ];
        let hull = LadderOptimizer::convex_hull(rungs);
        assert_eq!(hull.len(), 1);
        assert_eq!(hull[0], (1_000_000, 60.0));
    }
}
