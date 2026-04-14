//! Predictive cache warming — pre-populate edge caches before demand arrives.
//!
//! # Overview
//!
//! [`CacheWarmer`] maintains a list of [`WarmingCandidate`] entries ranked by
//! their `priority_score`.  Callers add candidates, then retrieve the top-N
//! (greedy budget selection, strategy filtering, etc.) to drive actual warming
//! requests against edge PoPs.
//!
//! Five warming strategies are modelled:
//!
//! - [`WarmingStrategy::MostPopular`] — push the most-viewed assets.
//! - [`WarmingStrategy::RecentlyPublished`] — warm fresh content before its
//!   first organic requests arrive.
//! - [`WarmingStrategy::ScheduledRelease`] — coordinate warming for content
//!   that has a known release date/time.
//! - [`WarmingStrategy::HighValueUsers`] — pre-warm for premium-tier subscribers.
//! - [`WarmingStrategy::Geographic`] — warm a specific regional PoP cluster.

use std::collections::HashMap;

// ─── WarmingStrategy ─────────────────────────────────────────────────────────

/// Reason / mechanism driving a particular warming decision.
#[derive(Debug, Clone, PartialEq)]
pub enum WarmingStrategy {
    /// Top-`n` assets by historical view-count.
    MostPopular {
        /// Number of top assets to consider.
        top_n: usize,
    },
    /// Assets published within the last `hours` hours.
    RecentlyPublished {
        /// Lookback window.
        hours: u64,
    },
    /// Assets with a hard-coded release event (e.g. sports broadcast).
    ScheduledRelease,
    /// Pre-warm for users in the given service tier (0 = free, 1 = standard,
    /// 2 = premium, …).
    HighValueUsers {
        /// Tier identifier.
        tier: u8,
    },
    /// Warm a named geographic region cluster.
    Geographic {
        /// Region label, e.g. `"ap-southeast-1"`.
        region: String,
    },
}

impl WarmingStrategy {
    /// Returns a short string tag used to compare strategy *types* without
    /// caring about the payload value.
    fn type_tag(&self) -> &'static str {
        match self {
            Self::MostPopular { .. } => "most_popular",
            Self::RecentlyPublished { .. } => "recently_published",
            Self::ScheduledRelease => "scheduled_release",
            Self::HighValueUsers { .. } => "high_value_users",
            Self::Geographic { .. } => "geographic",
        }
    }
}

// ─── WarmingCandidate ────────────────────────────────────────────────────────

/// A single asset that has been nominated for cache warming.
#[derive(Debug, Clone)]
pub struct WarmingCandidate {
    /// Unique asset identifier (URL path, CMS ID, etc.).
    pub asset_id: String,
    /// Score in [0, ∞) — higher means higher priority.
    pub priority_score: f32,
    /// Reason this candidate was nominated.
    pub reason: WarmingStrategy,
    /// Estimated binary size of the asset.
    pub estimated_size_bytes: u64,
}

impl WarmingCandidate {
    /// Construct a new candidate.
    pub fn new(
        asset_id: impl Into<String>,
        priority_score: f32,
        reason: WarmingStrategy,
        estimated_size_bytes: u64,
    ) -> Self {
        Self {
            asset_id: asset_id.into(),
            priority_score,
            reason,
            estimated_size_bytes,
        }
    }
}

// ─── WarmingStats ────────────────────────────────────────────────────────────

/// Aggregate statistics describing the current warming queue.
#[derive(Debug, Clone, Default)]
pub struct WarmingStats {
    /// Total candidates currently queued.
    pub candidates_queued: usize,
    /// Sum of `estimated_size_bytes` across all queued candidates.
    pub bytes_queued: u64,
    /// Rough hit-rate improvement estimate in [0, 1].
    ///
    /// Calculated as `1 − e^(−candidates_queued / 100)` — a simple
    /// saturation curve that plateaus as the queue grows.
    pub estimated_hit_rate_improvement: f32,
}

impl WarmingStats {
    fn compute(candidates_queued: usize, bytes_queued: u64) -> Self {
        // Saturation model: improvement ≈ 1 − e^(−n/100)
        let n = candidates_queued as f32;
        let improvement = 1.0_f32 - (-n / 100.0_f32).exp();
        Self {
            candidates_queued,
            bytes_queued,
            estimated_hit_rate_improvement: improvement,
        }
    }
}

// ─── CacheWarmer ─────────────────────────────────────────────────────────────

/// Maintains a collection of warming candidates and provides selection helpers.
#[derive(Debug, Default)]
pub struct CacheWarmer {
    candidates: Vec<WarmingCandidate>,
}

impl CacheWarmer {
    /// Create an empty warmer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enqueue a new candidate.
    ///
    /// Duplicates (same `asset_id`) are permitted; the caller is responsible
    /// for de-duplication if required.
    pub fn add_candidate(&mut self, candidate: WarmingCandidate) {
        self.candidates.push(candidate);
    }

    /// Return references to the top-`n` candidates, sorted by
    /// `priority_score` descending.
    ///
    /// If `n` is larger than the total number of candidates all of them are
    /// returned.
    pub fn top_candidates(&self, n: usize) -> Vec<&WarmingCandidate> {
        let mut sorted: Vec<&WarmingCandidate> = self.candidates.iter().collect();
        sorted.sort_by(|a, b| {
            b.priority_score
                .partial_cmp(&a.priority_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(n);
        sorted
    }

    /// Sum of `estimated_size_bytes` across all queued candidates.
    pub fn total_bytes_to_warm(&self) -> u64 {
        self.candidates
            .iter()
            .map(|c| c.estimated_size_bytes)
            .sum()
    }

    /// Return references to candidates whose strategy **type** matches `strategy`.
    ///
    /// Only the strategy *variant discriminant* is compared — payload values are
    /// ignored.  For example, `Geographic { region: "us-east-1" }` will match
    /// any `Geographic { .. }` candidate regardless of the region value.
    pub fn filter_by_strategy(&self, strategy: &WarmingStrategy) -> Vec<&WarmingCandidate> {
        let tag = strategy.type_tag();
        self.candidates
            .iter()
            .filter(|c| c.reason.type_tag() == tag)
            .collect()
    }

    /// Greedy knapsack: select candidates by descending priority until
    /// `max_bytes` is exhausted or all candidates are included.
    ///
    /// Returns references into the internal candidate list.
    pub fn warm_budget(&self, max_bytes: u64) -> Vec<&WarmingCandidate> {
        let mut sorted: Vec<&WarmingCandidate> = self.candidates.iter().collect();
        sorted.sort_by(|a, b| {
            b.priority_score
                .partial_cmp(&a.priority_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut remaining = max_bytes;
        let mut selected = Vec::new();
        for c in sorted {
            if c.estimated_size_bytes <= remaining {
                remaining -= c.estimated_size_bytes;
                selected.push(c);
            }
            // skip items that are individually too large
        }
        selected
    }

    /// Compute aggregate statistics for the current candidate queue.
    pub fn stats(&self) -> WarmingStats {
        WarmingStats::compute(self.candidates.len(), self.total_bytes_to_warm())
    }

    /// Number of candidates currently queued.
    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    /// Returns `true` if the warmer has no candidates.
    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    /// Remove all candidates.
    pub fn clear(&mut self) {
        self.candidates.clear();
    }

    /// Drain all candidates matching `strategy` type from the queue, returning
    /// them as an owned `Vec`.
    pub fn drain_by_strategy(&mut self, strategy: &WarmingStrategy) -> Vec<WarmingCandidate> {
        let tag = strategy.type_tag();
        let mut drained = Vec::new();
        let mut remaining = Vec::new();
        for c in self.candidates.drain(..) {
            if c.reason.type_tag() == tag {
                drained.push(c);
            } else {
                remaining.push(c);
            }
        }
        self.candidates = remaining;
        drained
    }

    /// Strategy-type breakdown: returns a map of strategy tag → count.
    pub fn strategy_breakdown(&self) -> HashMap<&'static str, usize> {
        let mut map: HashMap<&'static str, usize> = HashMap::new();
        for c in &self.candidates {
            *map.entry(c.reason.type_tag()).or_insert(0) += 1;
        }
        map
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn pop(id: &str, score: f32, size: u64) -> WarmingCandidate {
        WarmingCandidate::new(id, score, WarmingStrategy::MostPopular { top_n: 50 }, size)
    }

    fn geo(id: &str, score: f32, size: u64, region: &str) -> WarmingCandidate {
        WarmingCandidate::new(
            id,
            score,
            WarmingStrategy::Geographic {
                region: region.to_string(),
            },
            size,
        )
    }

    fn recent(id: &str, score: f32, size: u64) -> WarmingCandidate {
        WarmingCandidate::new(
            id,
            score,
            WarmingStrategy::RecentlyPublished { hours: 24 },
            size,
        )
    }

    // 1. top_candidates returns N items in descending order
    #[test]
    fn test_top_candidates_order() {
        let mut w = CacheWarmer::new();
        w.add_candidate(pop("a", 1.0, 100));
        w.add_candidate(pop("b", 5.0, 100));
        w.add_candidate(pop("c", 3.0, 100));
        w.add_candidate(pop("d", 9.0, 100));
        let top = w.top_candidates(3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].asset_id, "d");
        assert_eq!(top[1].asset_id, "b");
        assert_eq!(top[2].asset_id, "c");
    }

    // 2. top_candidates with n > len returns all
    #[test]
    fn test_top_candidates_clamped_to_len() {
        let mut w = CacheWarmer::new();
        w.add_candidate(pop("a", 1.0, 1));
        w.add_candidate(pop("b", 2.0, 1));
        assert_eq!(w.top_candidates(100).len(), 2);
    }

    // 3. total_bytes_to_warm sums correctly
    #[test]
    fn test_total_bytes_to_warm() {
        let mut w = CacheWarmer::new();
        w.add_candidate(pop("a", 1.0, 1_000));
        w.add_candidate(pop("b", 2.0, 2_000));
        w.add_candidate(geo("c", 3.0, 500, "eu-west-1"));
        assert_eq!(w.total_bytes_to_warm(), 3_500);
    }

    // 4. filter_by_strategy returns only matching type
    #[test]
    fn test_filter_by_strategy_type() {
        let mut w = CacheWarmer::new();
        w.add_candidate(pop("a", 2.0, 100));
        w.add_candidate(geo("b", 3.0, 200, "us-east-1"));
        w.add_candidate(geo("c", 1.0, 150, "ap-southeast-1"));
        w.add_candidate(recent("d", 4.0, 300));

        let geo_strat = WarmingStrategy::Geographic {
            region: "anything".into(),
        };
        let matches = w.filter_by_strategy(&geo_strat);
        assert_eq!(matches.len(), 2);
        for m in &matches {
            assert!(matches!(m.reason, WarmingStrategy::Geographic { .. }));
        }
    }

    // 5. warm_budget greedy selection within byte cap
    #[test]
    fn test_warm_budget_greedy() {
        let mut w = CacheWarmer::new();
        // Budget: 500 bytes
        // Candidates ordered by score: d(9.0,200), b(5.0,300), c(3.0,100), a(1.0,150)
        w.add_candidate(pop("a", 1.0, 150));
        w.add_candidate(pop("b", 5.0, 300));
        w.add_candidate(pop("c", 3.0, 100));
        w.add_candidate(pop("d", 9.0, 200));

        let selected = w.warm_budget(500);
        let ids: Vec<&str> = selected.iter().map(|c| c.asset_id.as_str()).collect();
        // d(200) fits → 300 remain; b(300) fits → 0 remain; c(100) does not fit
        assert!(ids.contains(&"d"), "highest priority 'd' must be selected");
        assert!(ids.contains(&"b"), "'b' should fit after 'd'");
        assert!(!ids.contains(&"c"), "'c' should not fit (0 budget left)");
        assert!(!ids.contains(&"a"), "'a' should not fit");
    }

    // 6. warm_budget with zero budget returns empty
    #[test]
    fn test_warm_budget_zero() {
        let mut w = CacheWarmer::new();
        w.add_candidate(pop("x", 9.9, 1));
        assert!(w.warm_budget(0).is_empty());
    }

    // 7. warm_budget with sufficient budget includes all
    #[test]
    fn test_warm_budget_all_fit() {
        let mut w = CacheWarmer::new();
        w.add_candidate(pop("a", 1.0, 100));
        w.add_candidate(pop("b", 2.0, 200));
        w.add_candidate(pop("c", 3.0, 300));
        let selected = w.warm_budget(1_000);
        assert_eq!(selected.len(), 3);
    }

    // 8. stats: candidates_queued and bytes_queued
    #[test]
    fn test_stats_counts() {
        let mut w = CacheWarmer::new();
        w.add_candidate(pop("a", 1.0, 1_024));
        w.add_candidate(pop("b", 2.0, 2_048));
        let s = w.stats();
        assert_eq!(s.candidates_queued, 2);
        assert_eq!(s.bytes_queued, 3_072);
    }

    // 9. stats: hit_rate_improvement saturates
    #[test]
    fn test_stats_hit_rate_improvement() {
        let mut w = CacheWarmer::new();
        let s0 = w.stats();
        assert!((s0.estimated_hit_rate_improvement - 0.0).abs() < 1e-6);

        for i in 0..50 {
            w.add_candidate(pop(&format!("a{i}"), i as f32, 1_000));
        }
        let s50 = w.stats();
        // With 50 candidates improvement should be meaningful but < 1
        assert!(s50.estimated_hit_rate_improvement > 0.3);
        assert!(s50.estimated_hit_rate_improvement < 1.0);
    }

    // 10. strategy_breakdown counts per type
    #[test]
    fn test_strategy_breakdown() {
        let mut w = CacheWarmer::new();
        w.add_candidate(pop("a", 1.0, 100));
        w.add_candidate(pop("b", 2.0, 100));
        w.add_candidate(geo("c", 3.0, 100, "eu"));
        w.add_candidate(recent("d", 4.0, 100));
        w.add_candidate(recent("e", 5.0, 100));
        w.add_candidate(WarmingCandidate::new(
            "f",
            6.0,
            WarmingStrategy::ScheduledRelease,
            100,
        ));
        let bd = w.strategy_breakdown();
        assert_eq!(bd.get("most_popular").copied().unwrap_or(0), 2);
        assert_eq!(bd.get("geographic").copied().unwrap_or(0), 1);
        assert_eq!(bd.get("recently_published").copied().unwrap_or(0), 2);
        assert_eq!(bd.get("scheduled_release").copied().unwrap_or(0), 1);
    }

    // 11. drain_by_strategy removes matching from queue
    #[test]
    fn test_drain_by_strategy() {
        let mut w = CacheWarmer::new();
        w.add_candidate(pop("a", 1.0, 100));
        w.add_candidate(geo("b", 2.0, 100, "us"));
        w.add_candidate(geo("c", 3.0, 100, "eu"));
        w.add_candidate(recent("d", 4.0, 100));

        let strat = WarmingStrategy::Geographic { region: "x".into() };
        let drained = w.drain_by_strategy(&strat);
        assert_eq!(drained.len(), 2);
        assert_eq!(w.len(), 2); // "a" and "d" remain
    }

    // 12. HighValueUsers strategy filtering
    #[test]
    fn test_high_value_users_filter() {
        let mut w = CacheWarmer::new();
        w.add_candidate(WarmingCandidate::new(
            "premium-stream",
            8.5,
            WarmingStrategy::HighValueUsers { tier: 2 },
            512_000,
        ));
        w.add_candidate(pop("regular", 7.0, 100_000));
        let strat = WarmingStrategy::HighValueUsers { tier: 99 }; // tier ignored in match
        let matches = w.filter_by_strategy(&strat);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].asset_id, "premium-stream");
    }

    // 13. clear empties the queue
    #[test]
    fn test_clear() {
        let mut w = CacheWarmer::new();
        w.add_candidate(pop("a", 1.0, 100));
        w.clear();
        assert!(w.is_empty());
        assert_eq!(w.total_bytes_to_warm(), 0);
    }

    // 14. warm_budget skips items larger than remaining budget individually
    #[test]
    fn test_warm_budget_skips_oversized_items() {
        let mut w = CacheWarmer::new();
        // highest priority is huge — won't fit; second should be chosen
        w.add_candidate(pop("huge", 10.0, 1_000_000));
        w.add_candidate(pop("small", 5.0, 500));
        let selected = w.warm_budget(1_000);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].asset_id, "small");
    }
}
