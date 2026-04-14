//! Click-through rate (CTR) tracking for thumbnails and content previews.
//!
//! Tracks impression and click events per item (thumbnail, preview card, etc.)
//! and computes click-through rates, confidence intervals, and performance
//! rankings across a catalogue of content items.
//!
//! ## Definitions
//!
//! - **Impression** — an item was displayed to a viewer.
//! - **Click** — a viewer interacted with (clicked/tapped) the item.
//! - **CTR** — `clicks / impressions` expressed as a fraction in `[0.0, 1.0]`.
//!
//! Wilson score confidence intervals are used for the CTR bounds because they
//! remain valid even for small sample sizes and extreme rates (near 0 or 1)
//! unlike the normal approximation.
//!
//! ## Example
//!
//! ```rust
//! use oximedia_analytics::ctr::{CtrTracker, CtrVariant};
//!
//! let mut tracker = CtrTracker::new();
//! tracker.record_impression("thumb_a");
//! tracker.record_impression("thumb_a");
//! tracker.record_click("thumb_a");
//!
//! let stats = tracker.stats("thumb_a").unwrap();
//! assert_eq!(stats.impressions, 2);
//! assert_eq!(stats.clicks, 1);
//! assert!((stats.ctr() - 0.5).abs() < 1e-6);
//! ```

use crate::error::AnalyticsError;
use std::collections::HashMap;

// ─── CtrStats ────────────────────────────────────────────────────────────────

/// Raw impression/click counts and derived metrics for a single item.
#[derive(Debug, Clone, PartialEq)]
pub struct CtrStats {
    /// The item identifier (e.g. thumbnail slug or content ID).
    pub item_id: String,
    /// Total number of times this item was shown to viewers.
    pub impressions: u64,
    /// Total number of times viewers clicked/tapped this item.
    pub clicks: u64,
}

impl CtrStats {
    /// Computes the raw click-through rate: `clicks / impressions`.
    ///
    /// Returns `0.0` when `impressions == 0`.
    #[must_use]
    pub fn ctr(&self) -> f64 {
        if self.impressions == 0 {
            0.0
        } else {
            self.clicks as f64 / self.impressions as f64
        }
    }

    /// Computes the Wilson score confidence interval for the CTR.
    ///
    /// Returns `(lower, upper)` bounds at the given `z` level.  Common values:
    /// - `z = 1.645` → 90 % confidence
    /// - `z = 1.960` → 95 % confidence  (default)
    /// - `z = 2.576` → 99 % confidence
    ///
    /// Returns `(0.0, 0.0)` when `impressions == 0`.
    ///
    /// # Reference
    ///
    /// Wilson, E. B. (1927). Probable inference, the law of succession, and
    /// statistical inference. *Journal of the American Statistical Association*,
    /// 22(158), 209-212.
    #[must_use]
    pub fn wilson_interval(&self, z: f64) -> (f64, f64) {
        let n = self.impressions as f64;
        if n == 0.0 {
            return (0.0, 0.0);
        }
        let p = self.clicks as f64 / n;
        let z2 = z * z;
        let denom = 1.0 + z2 / n;
        let centre = (p + z2 / (2.0 * n)) / denom;
        let margin = (z / denom) * ((p * (1.0 - p) / n) + z2 / (4.0 * n * n)).sqrt();
        ((centre - margin).max(0.0), (centre + margin).min(1.0))
    }

    /// Returns `true` when this item has never been shown.
    #[must_use]
    pub fn is_untracked(&self) -> bool {
        self.impressions == 0
    }
}

// ─── CtrVariant ──────────────────────────────────────────────────────────────

/// A summarised variant for ranking and A/B reporting.
#[derive(Debug, Clone, PartialEq)]
pub struct CtrVariant {
    /// Item identifier.
    pub item_id: String,
    /// Raw CTR in `[0.0, 1.0]`.
    pub ctr: f64,
    /// Wilson 95 % confidence lower bound.
    pub ci_lower: f64,
    /// Wilson 95 % confidence upper bound.
    pub ci_upper: f64,
    /// Total impressions recorded.
    pub impressions: u64,
    /// Total clicks recorded.
    pub clicks: u64,
}

// ─── CtrTracker ──────────────────────────────────────────────────────────────

/// Tracks CTR metrics for an arbitrary number of items.
///
/// Each item is identified by a `&str` key.  Items are created on first
/// impression or click — there is no need to pre-register them.
#[derive(Debug, Default, Clone)]
pub struct CtrTracker {
    data: HashMap<String, CtrStats>,
}

impl CtrTracker {
    /// Creates an empty `CtrTracker`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Records one impression for `item_id`.
    ///
    /// If the item is not yet tracked it is created with zero counts first.
    pub fn record_impression(&mut self, item_id: &str) {
        let entry = self
            .data
            .entry(item_id.to_owned())
            .or_insert_with(|| CtrStats {
                item_id: item_id.to_owned(),
                impressions: 0,
                clicks: 0,
            });
        entry.impressions = entry.impressions.saturating_add(1);
    }

    /// Records `count` impressions for `item_id` in a single call.
    ///
    /// Useful for bulk-ingesting impression counts from a reporting pipeline.
    pub fn record_impressions(&mut self, item_id: &str, count: u64) {
        let entry = self
            .data
            .entry(item_id.to_owned())
            .or_insert_with(|| CtrStats {
                item_id: item_id.to_owned(),
                impressions: 0,
                clicks: 0,
            });
        entry.impressions = entry.impressions.saturating_add(count);
    }

    /// Records one click for `item_id`.
    ///
    /// If the item is not yet tracked it is created with zero counts first.
    /// Clicks without a preceding impression are allowed (e.g. deep links).
    pub fn record_click(&mut self, item_id: &str) {
        let entry = self
            .data
            .entry(item_id.to_owned())
            .or_insert_with(|| CtrStats {
                item_id: item_id.to_owned(),
                impressions: 0,
                clicks: 0,
            });
        entry.clicks = entry.clicks.saturating_add(1);
    }

    /// Records `count` clicks for `item_id` in a single call.
    pub fn record_clicks(&mut self, item_id: &str, count: u64) {
        let entry = self
            .data
            .entry(item_id.to_owned())
            .or_insert_with(|| CtrStats {
                item_id: item_id.to_owned(),
                impressions: 0,
                clicks: 0,
            });
        entry.clicks = entry.clicks.saturating_add(count);
    }

    /// Returns the [`CtrStats`] for `item_id`, or `None` if it has never been
    /// seen.
    #[must_use]
    pub fn stats(&self, item_id: &str) -> Option<&CtrStats> {
        self.data.get(item_id)
    }

    /// Returns the raw CTR for `item_id`.
    ///
    /// # Errors
    ///
    /// Returns [`AnalyticsError::InvalidInput`] when `item_id` is not tracked.
    pub fn ctr(&self, item_id: &str) -> Result<f64, AnalyticsError> {
        self.data
            .get(item_id)
            .map(CtrStats::ctr)
            .ok_or_else(|| AnalyticsError::InvalidInput(format!("item '{item_id}' not tracked")))
    }

    /// Returns items ranked by CTR descending.
    ///
    /// Items with no impressions are placed at the bottom (CTR = 0).  Items
    /// with equal CTR are ordered by total impressions descending (more data
    /// first).
    #[must_use]
    pub fn ranked(&self) -> Vec<CtrVariant> {
        let z = 1.960_f64; // 95 % confidence
        let mut variants: Vec<CtrVariant> = self
            .data
            .values()
            .map(|s| {
                let (lo, hi) = s.wilson_interval(z);
                CtrVariant {
                    item_id: s.item_id.clone(),
                    ctr: s.ctr(),
                    ci_lower: lo,
                    ci_upper: hi,
                    impressions: s.impressions,
                    clicks: s.clicks,
                }
            })
            .collect();

        // Primary sort: CTR descending.  Secondary: impressions descending.
        variants.sort_by(|a, b| {
            b.ctr
                .partial_cmp(&a.ctr)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.impressions.cmp(&a.impressions))
        });

        variants
    }

    /// Returns the item with the highest CTR among those with at least
    /// `min_impressions` impressions.
    ///
    /// # Errors
    ///
    /// Returns [`AnalyticsError::InsufficientData`] when no item meets the
    /// `min_impressions` threshold.
    pub fn winner(&self, min_impressions: u64) -> Result<CtrVariant, AnalyticsError> {
        let z = 1.960_f64;
        self.data
            .values()
            .filter(|s| s.impressions >= min_impressions)
            .max_by(|a, b| {
                a.ctr()
                    .partial_cmp(&b.ctr())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|s| {
                let (lo, hi) = s.wilson_interval(z);
                CtrVariant {
                    item_id: s.item_id.clone(),
                    ctr: s.ctr(),
                    ci_lower: lo,
                    ci_upper: hi,
                    impressions: s.impressions,
                    clicks: s.clicks,
                }
            })
            .ok_or_else(|| {
                AnalyticsError::InsufficientData(format!(
                    "no item has ≥ {min_impressions} impressions"
                ))
            })
    }

    /// Returns the total number of items currently being tracked.
    #[must_use]
    pub fn item_count(&self) -> usize {
        self.data.len()
    }

    /// Clears all tracking data.
    pub fn reset(&mut self) {
        self.data.clear();
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CtrStats ──────────────────────────────────────────────────────────────

    #[test]
    fn ctr_zero_when_no_impressions() {
        let s = CtrStats {
            item_id: "x".into(),
            impressions: 0,
            clicks: 0,
        };
        assert_eq!(s.ctr(), 0.0);
        assert!(s.is_untracked());
    }

    #[test]
    fn ctr_computed_correctly() {
        let s = CtrStats {
            item_id: "a".into(),
            impressions: 4,
            clicks: 1,
        };
        assert!((s.ctr() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn wilson_interval_symmetry_at_50_percent() {
        // 50 % CTR with 100 impressions — interval should be symmetric around 0.5.
        let s = CtrStats {
            item_id: "sym".into(),
            impressions: 100,
            clicks: 50,
        };
        let (lo, hi) = s.wilson_interval(1.960);
        // Both bounds should exist and be roughly equidistant from 0.5.
        let dist_lo = 0.5 - lo;
        let dist_hi = hi - 0.5;
        assert!((dist_lo - dist_hi).abs() < 0.01, "lo={lo:.4}, hi={hi:.4}");
    }

    #[test]
    fn wilson_interval_empty_item() {
        let s = CtrStats {
            item_id: "empty".into(),
            impressions: 0,
            clicks: 0,
        };
        let (lo, hi) = s.wilson_interval(1.960);
        assert_eq!((lo, hi), (0.0, 0.0));
    }

    #[test]
    fn wilson_interval_bounds_in_range() {
        let s = CtrStats {
            item_id: "b".into(),
            impressions: 200,
            clicks: 30,
        };
        let (lo, hi) = s.wilson_interval(1.960);
        assert!(lo >= 0.0 && lo <= 1.0, "lower={lo}");
        assert!(hi >= 0.0 && hi <= 1.0, "upper={hi}");
        assert!(lo < hi, "lower must be < upper");
    }

    // ── CtrTracker ────────────────────────────────────────────────────────────

    #[test]
    fn record_impression_and_click() {
        let mut t = CtrTracker::new();
        t.record_impression("thumb_a");
        t.record_impression("thumb_a");
        t.record_click("thumb_a");

        let s = t.stats("thumb_a").expect("exists");
        assert_eq!(s.impressions, 2);
        assert_eq!(s.clicks, 1);
        assert!((s.ctr() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn stats_unknown_item_returns_none() {
        let t = CtrTracker::new();
        assert!(t.stats("unknown").is_none());
    }

    #[test]
    fn ctr_unknown_item_errors() {
        let t = CtrTracker::new();
        assert!(t.ctr("ghost").is_err());
    }

    #[test]
    fn ranked_orders_by_ctr_descending() {
        let mut t = CtrTracker::new();
        // "low": 5 clicks / 100 impressions = 5 %
        t.record_impressions("low", 100);
        t.record_clicks("low", 5);
        // "high": 20 clicks / 100 impressions = 20 %
        t.record_impressions("high", 100);
        t.record_clicks("high", 20);

        let ranked = t.ranked();
        assert_eq!(ranked[0].item_id, "high");
        assert_eq!(ranked[1].item_id, "low");
    }

    #[test]
    fn winner_selects_best_item() {
        let mut t = CtrTracker::new();
        t.record_impressions("a", 1000);
        t.record_clicks("a", 50);
        t.record_impressions("b", 1000);
        t.record_clicks("b", 200);

        let w = t.winner(100).expect("winner found");
        assert_eq!(w.item_id, "b");
        assert!((w.ctr - 0.2).abs() < 1e-9);
    }

    #[test]
    fn winner_errors_when_min_impressions_not_met() {
        let mut t = CtrTracker::new();
        t.record_impressions("tiny", 5);
        t.record_clicks("tiny", 1);

        assert!(t.winner(100).is_err());
    }

    #[test]
    fn item_count_and_reset() {
        let mut t = CtrTracker::new();
        t.record_impression("x");
        t.record_impression("y");
        assert_eq!(t.item_count(), 2);
        t.reset();
        assert_eq!(t.item_count(), 0);
    }

    #[test]
    fn bulk_impression_and_click_recording() {
        let mut t = CtrTracker::new();
        t.record_impressions("bulk", 500);
        t.record_clicks("bulk", 100);
        let s = t.stats("bulk").expect("exists");
        assert_eq!(s.impressions, 500);
        assert_eq!(s.clicks, 100);
        assert!((s.ctr() - 0.2).abs() < 1e-9);
    }

    #[test]
    fn click_without_impression_allowed() {
        let mut t = CtrTracker::new();
        t.record_click("deep_link"); // deep link — no prior impression
        let s = t.stats("deep_link").expect("exists");
        assert_eq!(s.impressions, 0);
        assert_eq!(s.clicks, 1);
        assert_eq!(s.ctr(), 0.0); // 0 impressions → ctr 0
    }

    #[test]
    fn ranked_ci_bounds_populated() {
        let mut t = CtrTracker::new();
        t.record_impressions("item", 200);
        t.record_clicks("item", 40);
        let ranked = t.ranked();
        assert_eq!(ranked.len(), 1);
        let v = &ranked[0];
        assert!(v.ci_lower < v.ctr);
        assert!(v.ci_upper > v.ctr);
    }
}
