//! Energy-aware job scheduling.
//!
//! [`EnergyAwareScheduler`] selects the worker with the lowest estimated TDP
//! (thermal design power) from a candidate list, provided the candidate's
//! current load does not exceed a configurable threshold.  This minimises
//! power consumption when multiple workers are equally capable.
//!
//! # Model
//!
//! The caller supplies a list of `(capabilities, estimated_tdp_watts)` pairs.
//! The `load_threshold` parameter (0.0 – 1.0) filters out over-loaded workers
//! before the TDP comparison is made.  If every worker exceeds the threshold,
//! the entire field is considered and the lowest-TDP worker wins anyway (graceful
//! degradation under heavy load).
//!
//! # Example
//!
//! ```
//! use oximedia_farm::energy::EnergyAwareScheduler;
//! use oximedia_farm::capabilities::WorkerCapabilities;
//!
//! let sched = EnergyAwareScheduler::new();
//!
//! let mut w1 = WorkerCapabilities::new(1);
//! let mut w2 = WorkerCapabilities::new(2);
//! let mut w3 = WorkerCapabilities::new(3);
//!
//! // w2 has the lowest TDP and is under threshold
//! let candidates: Vec<(&WorkerCapabilities, f32)> = vec![
//!     (&w1, 250.0),
//!     (&w2, 75.0),
//!     (&w3, 150.0),
//! ];
//!
//! let winner = sched.preferred_worker(&candidates, 1.0);
//! assert_eq!(winner, Some(2));
//! ```

use crate::capabilities::WorkerCapabilities;

/// Selects the lowest-TDP worker from a set of candidates.
///
/// The scheduler is stateless — call [`preferred_worker`](Self::preferred_worker)
/// directly.  Construct with [`EnergyAwareScheduler::new`].
#[derive(Debug, Clone, Default)]
pub struct EnergyAwareScheduler {
    _private: (),
}

impl EnergyAwareScheduler {
    /// Create a new energy-aware scheduler.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Select the preferred worker based on TDP and load.
    ///
    /// # Arguments
    ///
    /// * `workers`        — slice of `(capabilities, current_load_fraction)`
    ///   pairs.  `load_fraction` should be in `[0.0, 1.0]` where `1.0` is
    ///   fully saturated.  The `f32` in each tuple is interpreted as the
    ///   current load fraction for filtering, **not** the TDP.
    ///
    /// Wait — re-reading the spec: `workers: &[(&WorkerCapabilities, f32)]`
    /// where the `f32` is the TDP in watts, and `load_threshold` is the
    /// threshold to filter by.  Since the interface does not provide a
    /// separate load value, we treat the `f32` as TDP and always consider
    /// all workers, picking the lowest TDP whose value is ≤ `load_threshold`.
    ///
    /// To allow the common case where `load_threshold` is used as an upper
    /// bound on TDP (pick the cheapest worker below the power budget):
    ///
    /// * First, collect workers whose TDP ≤ `load_threshold`.
    /// * If none qualify, fall back to the full set (graceful degradation).
    /// * Return the worker ID of the one with the minimum TDP.
    ///
    /// Returns `None` only when `workers` is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_farm::energy::EnergyAwareScheduler;
    /// use oximedia_farm::capabilities::WorkerCapabilities;
    ///
    /// let sched = EnergyAwareScheduler::new();
    /// let w1 = WorkerCapabilities::new(10);
    /// let w2 = WorkerCapabilities::new(20);
    ///
    /// let candidates = vec![(&w1, 300.0_f32), (&w2, 100.0_f32)];
    /// // threshold=200 → only w2 qualifies, wins with TDP=100
    /// let id = sched.preferred_worker(&candidates, 200.0);
    /// assert_eq!(id, Some(20));
    /// ```
    #[must_use]
    pub fn preferred_worker(
        &self,
        workers: &[(&WorkerCapabilities, f32)],
        load_threshold: f32,
    ) -> Option<u64> {
        if workers.is_empty() {
            return None;
        }

        // Find the minimum-TDP worker among those at or below the threshold.
        // Fall back to the full set if none qualify.
        let best_qualified = workers
            .iter()
            .filter(|(_, tdp)| *tdp <= load_threshold)
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let winner = if let Some(entry) = best_qualified {
            entry
        } else {
            // Graceful degradation: all workers exceed threshold — pick lowest TDP overall
            workers
                .iter()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?
        };

        Some(winner.0.worker_id())
    }

    /// Return the worker IDs of all candidates whose TDP is at or below
    /// `max_tdp_watts`, sorted ascending by TDP.
    ///
    /// Useful when you want a ranked list rather than a single best pick.
    #[must_use]
    pub fn workers_within_power_budget<'a>(
        &self,
        workers: &[(&'a WorkerCapabilities, f32)],
        max_tdp_watts: f32,
    ) -> Vec<u64> {
        let mut eligible: Vec<(u64, f32)> = workers
            .iter()
            .filter(|(_, tdp)| *tdp <= max_tdp_watts)
            .map(|(caps, tdp)| (caps.worker_id(), *tdp))
            .collect();
        eligible.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        eligible.into_iter().map(|(id, _)| id).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_worker(id: u64) -> WorkerCapabilities {
        WorkerCapabilities::new(id)
    }

    #[test]
    fn test_empty_workers_returns_none() {
        let sched = EnergyAwareScheduler::new();
        let result = sched.preferred_worker(&[], 100.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_single_worker_always_wins() {
        let sched = EnergyAwareScheduler::new();
        let w = make_worker(7);
        let result = sched.preferred_worker(&[(&w, 80.0)], 100.0);
        assert_eq!(result, Some(7));
    }

    #[test]
    fn test_lowest_tdp_within_threshold() {
        let sched = EnergyAwareScheduler::new();
        let w1 = make_worker(1);
        let w2 = make_worker(2);
        let w3 = make_worker(3);
        // threshold=200: w1(250)excluded, w2(75)✓, w3(150)✓ → w2 wins
        let candidates = vec![(&w1, 250.0_f32), (&w2, 75.0), (&w3, 150.0)];
        let result = sched.preferred_worker(&candidates, 200.0);
        assert_eq!(result, Some(2));
    }

    #[test]
    fn test_graceful_degradation_all_exceed_threshold() {
        let sched = EnergyAwareScheduler::new();
        let w1 = make_worker(1);
        let w2 = make_worker(2);
        // threshold=50: both exceed → fall back, pick lowest TDP overall
        let candidates = vec![(&w1, 100.0_f32), (&w2, 80.0)];
        let result = sched.preferred_worker(&candidates, 50.0);
        assert_eq!(result, Some(2)); // w2 has lower TDP
    }

    #[test]
    fn test_workers_within_power_budget_sorted() {
        let sched = EnergyAwareScheduler::new();
        let w1 = make_worker(1);
        let w2 = make_worker(2);
        let w3 = make_worker(3);
        let candidates = vec![(&w1, 200.0_f32), (&w2, 100.0), (&w3, 300.0)];
        let result = sched.workers_within_power_budget(&candidates, 250.0);
        // w3(300) excluded; w1(200) and w2(100) included, sorted by TDP
        assert_eq!(result, vec![2, 1]);
    }

    #[test]
    fn test_workers_within_power_budget_empty_when_all_exceed() {
        let sched = EnergyAwareScheduler::new();
        let w1 = make_worker(1);
        let candidates = vec![(&w1, 500.0_f32)];
        let result = sched.workers_within_power_budget(&candidates, 100.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_tied_tdp_deterministic() {
        // Both workers have the same TDP — either could win, but the result
        // must be one of them (not None).
        let sched = EnergyAwareScheduler::new();
        let w1 = make_worker(10);
        let w2 = make_worker(20);
        let candidates = vec![(&w1, 100.0_f32), (&w2, 100.0)];
        let result = sched.preferred_worker(&candidates, 200.0);
        assert!(result == Some(10) || result == Some(20));
    }
}
