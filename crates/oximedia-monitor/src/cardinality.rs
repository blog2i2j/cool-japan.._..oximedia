//! Metric cardinality enforcement with configurable overflow actions.
//!
//! High-cardinality label sets (e.g. per-user IDs, per-request trace labels)
//! can cause unbounded memory growth in metric stores.  This module provides a
//! [`CardinalityGuard`] that tracks the number of distinct label combinations
//! per metric name and enforces configurable limits.
//!
//! Three overflow strategies are supported:
//!
//! | [`OverflowAction`]  | Behaviour when limit is reached                              |
//! |---------------------|--------------------------------------------------------------|
//! | `Drop`              | Returns [`CardinalityResult::Overflow`] with the original   |
//! |                     | label combo — caller should skip recording.                  |
//! | `UseOverflow`       | Returns [`CardinalityResult::Overflow`] with the sentinel    |
//! |                     | string `"__overflow__"`.                                     |
//! | `Error`             | Returns [`CardinalityResult::Error`] so the caller can log  |
//! |                     | or propagate the problem.                                    |
//!
//! # Example
//!
//! ```rust
//! use oximedia_monitor::cardinality::{
//!     CardinalityConfig, CardinalityGuard, CardinalityResult, OverflowAction,
//! };
//!
//! let cfg = CardinalityConfig::default_safe();
//! let mut guard = CardinalityGuard::new(cfg);
//!
//! // First series is allowed.
//! let result = guard.check_and_record("cpu_usage", "host=node1");
//! assert!(matches!(result, CardinalityResult::Allowed(_)));
//!
//! // When limit is exceeded with Drop action the caller receives Overflow.
//! let small_cfg = CardinalityConfig {
//!     max_label_values_per_metric: 1,
//!     max_total_series: 10_000,
//!     overflow_action: OverflowAction::Drop,
//! };
//! let mut guard2 = CardinalityGuard::new(small_cfg);
//! guard2.check_and_record("cpu_usage", "host=node1");
//! let over = guard2.check_and_record("cpu_usage", "host=node2");
//! assert!(matches!(over, CardinalityResult::Overflow(_)));
//! ```

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Action taken when a cardinality limit is exceeded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OverflowAction {
    /// Silently ignore the new label combination (caller should skip storing).
    Drop,
    /// Map the new label combination to the `"__overflow__"` sentinel value.
    UseOverflow,
    /// Return an error so the caller can log or propagate.
    Error,
}

/// Cardinality limit configuration.
#[derive(Debug, Clone)]
pub struct CardinalityConfig {
    /// Maximum number of distinct label combinations allowed for a single
    /// metric name.
    pub max_label_values_per_metric: usize,
    /// Maximum number of distinct series across *all* metric names combined.
    pub max_total_series: usize,
    /// Action applied when either limit is exceeded.
    pub overflow_action: OverflowAction,
}

impl CardinalityConfig {
    /// Conservative default: 100 label values per metric, 10 000 total series,
    /// silently drop overflows.
    #[must_use]
    pub fn default_safe() -> Self {
        Self {
            max_label_values_per_metric: 100,
            max_total_series: 10_000,
            overflow_action: OverflowAction::Drop,
        }
    }

    /// Permissive configuration: 10 000 label values per metric, 1 000 000
    /// total series, silently drop overflows.
    #[must_use]
    pub fn permissive() -> Self {
        Self {
            max_label_values_per_metric: 10_000,
            max_total_series: 1_000_000,
            overflow_action: OverflowAction::Drop,
        }
    }
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Outcome of a [`CardinalityGuard::check_and_record`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CardinalityResult {
    /// The series was within limits and has been recorded.
    /// The inner `String` is the label combo that was stored (identical to the
    /// input for new series, or the existing value for already-known ones).
    Allowed(String),
    /// The series exceeds a limit.
    ///
    /// - With [`OverflowAction::Drop`] the inner string is the *original* label
    ///   combo (caller should discard the data point).
    /// - With [`OverflowAction::UseOverflow`] the inner string is the sentinel
    ///   `"__overflow__"` (caller may store under this canonical label).
    Overflow(String),
    /// The series exceeds a limit and the action was [`OverflowAction::Error`].
    /// The inner string contains a human-readable explanation.
    Error(String),
}

// ---------------------------------------------------------------------------
// Guard
// ---------------------------------------------------------------------------

/// Tracks distinct label combinations per metric and enforces cardinality
/// limits according to the supplied [`CardinalityConfig`].
///
/// All state is held in-process; there is no persistence.
#[derive(Debug)]
pub struct CardinalityGuard {
    config: CardinalityConfig,
    /// `metric_name` → set of known label-combo strings.
    series_count: HashMap<String, HashSet<String>>,
}

impl CardinalityGuard {
    /// Create a new guard with the given configuration.
    #[must_use]
    pub fn new(config: CardinalityConfig) -> Self {
        Self {
            config,
            series_count: HashMap::new(),
        }
    }

    /// Check whether `label_combo` can be recorded for `metric_name`, and if
    /// allowed, record it.
    ///
    /// Returns a [`CardinalityResult`] describing the outcome.  The caller
    /// should act on the result as follows:
    ///
    /// | Result | Recommended caller action |
    /// |--------|---------------------------|
    /// | `Allowed(combo)` | Store the data point under `combo`. |
    /// | `Overflow(combo)` | Either discard (`Drop`) or store under `"__overflow__"` (`UseOverflow`). |
    /// | `Error(msg)` | Log `msg` and discard. |
    pub fn check_and_record(&mut self, metric_name: &str, label_combo: &str) -> CardinalityResult {
        // If the series already exists it does not increase cardinality.
        if let Some(set) = self.series_count.get(metric_name) {
            if set.contains(label_combo) {
                return CardinalityResult::Allowed(label_combo.to_string());
            }
        }

        // Check per-metric limit.
        let per_metric_count = self
            .series_count
            .get(metric_name)
            .map_or(0, HashSet::len);
        if per_metric_count >= self.config.max_label_values_per_metric {
            return self.overflow(metric_name, label_combo);
        }

        // Check total series limit.
        if self.total_series() >= self.config.max_total_series {
            return self.overflow(metric_name, label_combo);
        }

        // Admit the new series.
        self.series_count
            .entry(metric_name.to_string())
            .or_default()
            .insert(label_combo.to_string());

        CardinalityResult::Allowed(label_combo.to_string())
    }

    /// Number of distinct label combinations tracked for `metric_name`.
    #[must_use]
    pub fn series_count_for(&self, metric_name: &str) -> usize {
        self.series_count.get(metric_name).map_or(0, HashSet::len)
    }

    /// Total number of distinct series across all metric names.
    #[must_use]
    pub fn total_series(&self) -> usize {
        self.series_count.values().map(HashSet::len).sum()
    }

    /// Remove all recorded state, resetting all counts to zero.
    pub fn reset(&mut self) {
        self.series_count.clear();
    }

    // ── Private ──────────────────────────────────────────────────────────────

    fn overflow(&self, metric_name: &str, label_combo: &str) -> CardinalityResult {
        match self.config.overflow_action {
            OverflowAction::Drop => CardinalityResult::Overflow(label_combo.to_string()),
            OverflowAction::UseOverflow => {
                CardinalityResult::Overflow("__overflow__".to_string())
            }
            OverflowAction::Error => CardinalityResult::Error(format!(
                "cardinality limit exceeded for metric '{metric_name}': \
                 label_combo='{label_combo}' \
                 (per_metric_limit={}, total_series_limit={})",
                self.config.max_label_values_per_metric, self.config.max_total_series,
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// TODO-specified API
// ---------------------------------------------------------------------------
//
// The TODO specification requires a distinct `CardinalityConfig` structure with
// three fields: `max_series`, `max_labels_per_metric`, and `max_label_values`,
// plus a free function `check_series` and methods `series_count` / `label_value_count`.
// These are provided below as a separate, parallel API that wraps the richer
// types above.

/// Cardinality limit configuration (TODO-specified API).
///
/// This is a simplified companion to the full [`CardinalityConfig`] above,
/// with field names as required by the TODO specification.
#[derive(Debug, Clone)]
pub struct CardinalityLimitConfig {
    /// Maximum number of distinct time-series across all metric names combined.
    pub max_series: usize,
    /// Maximum number of distinct label keys per metric name.
    pub max_labels_per_metric: usize,
    /// Maximum number of distinct values per label key per metric.
    pub max_label_values: usize,
}

impl Default for CardinalityLimitConfig {
    fn default() -> Self {
        Self {
            max_series: 10_000,
            max_labels_per_metric: 20,
            max_label_values: 100,
        }
    }
}

/// Cardinality guard (TODO-specified API).
///
/// Tracks per-metric label key cardinality as well as per-(metric, label-key)
/// value cardinality, enforcing the limits in [`CardinalityLimitConfig`].
///
/// Unlike the richer [`CardinalityGuard`] above, this type uses
/// `HashMap<String, String>` label maps rather than pre-serialized combo strings.
#[derive(Debug)]
pub struct CardinalityLimiter {
    config: CardinalityLimitConfig,
    /// metric_name → set of label fingerprints (one per unique label-set).
    series: HashMap<String, HashSet<String>>,
    /// (metric_name, label_key) → set of observed values.
    label_values: HashMap<(String, String), HashSet<String>>,
}

impl CardinalityLimiter {
    /// Create a new limiter with the given configuration.
    #[must_use]
    pub fn new(config: CardinalityLimitConfig) -> Self {
        Self {
            config,
            series: HashMap::new(),
            label_values: HashMap::new(),
        }
    }

    /// Total number of distinct series across all metrics.
    #[must_use]
    pub fn series_count(&self) -> usize {
        self.series.values().map(HashSet::len).sum()
    }

    /// Number of distinct values observed for `label` under `metric`.
    #[must_use]
    pub fn label_value_count(&self, metric: &str, label: &str) -> usize {
        self.label_values
            .get(&(metric.to_string(), label.to_string()))
            .map_or(0, HashSet::len)
    }

    /// Check whether a new data point with the given `metric_name` and `labels`
    /// can be admitted.
    ///
    /// Returns `true` if the point is within all cardinality limits and has
    /// been recorded; returns `false` if adding it would exceed any limit.
    pub fn check(&mut self, metric_name: &str, labels: &HashMap<String, String>) -> bool {
        // Build a canonical fingerprint for this label-set.
        let mut pairs: Vec<(&String, &String)> = labels.iter().collect();
        pairs.sort_by_key(|(k, _)| k.as_str());
        let fingerprint = pairs
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",");

        // Check per-metric label count.
        if labels.len() > self.config.max_labels_per_metric {
            return false;
        }

        // Check per-(metric, label-key) value count.
        for (label_key, label_val) in labels {
            let key = (metric_name.to_string(), label_key.clone());
            let existing_count = self.label_values.get(&key).map_or(0, HashSet::len);
            let already_known = self
                .label_values
                .get(&key)
                .map_or(false, |s| s.contains(label_val));
            if !already_known && existing_count >= self.config.max_label_values {
                return false;
            }
        }

        // Check total series limit.
        let current_total: usize = self.series.values().map(HashSet::len).sum();
        let series_set = self.series.entry(metric_name.to_string()).or_default();
        if !series_set.contains(&fingerprint) {
            if current_total >= self.config.max_series {
                return false;
            }
            series_set.insert(fingerprint);
        }

        // Admit: record per-label-key values.
        for (label_key, label_val) in labels {
            let key = (metric_name.to_string(), label_key.clone());
            self.label_values
                .entry(key)
                .or_default()
                .insert(label_val.clone());
        }

        true
    }
}

/// Check whether a data point with `metric_name` + `labels` can be admitted
/// by `guard`, recording it if allowed.
///
/// This is the free-function form required by the TODO specification.
/// Returns `true` if the point is within limits; `false` otherwise.
pub fn check_series(
    guard: &mut CardinalityLimiter,
    metric_name: &str,
    labels: &HashMap<String, String>,
) -> bool {
    guard.check(metric_name, labels)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── CardinalityConfig ────────────────────────────────────────────────────

    #[test]
    fn test_default_safe_limits() {
        let cfg = CardinalityConfig::default_safe();
        assert_eq!(cfg.max_label_values_per_metric, 100);
        assert_eq!(cfg.max_total_series, 10_000);
        assert_eq!(cfg.overflow_action, OverflowAction::Drop);
    }

    #[test]
    fn test_permissive_limits() {
        let cfg = CardinalityConfig::permissive();
        assert_eq!(cfg.max_label_values_per_metric, 10_000);
        assert_eq!(cfg.max_total_series, 1_000_000);
    }

    // ── CardinalityGuard — basic admission ───────────────────────────────────

    #[test]
    fn test_new_series_under_limit_is_allowed() {
        let mut guard = CardinalityGuard::new(CardinalityConfig::default_safe());
        let result = guard.check_and_record("cpu_usage", "host=node1");
        assert_eq!(
            result,
            CardinalityResult::Allowed("host=node1".to_string())
        );
        assert_eq!(guard.series_count_for("cpu_usage"), 1);
    }

    #[test]
    fn test_same_series_again_is_allowed() {
        let mut guard = CardinalityGuard::new(CardinalityConfig::default_safe());
        guard.check_and_record("cpu_usage", "host=node1");
        let result = guard.check_and_record("cpu_usage", "host=node1");
        assert_eq!(
            result,
            CardinalityResult::Allowed("host=node1".to_string()),
            "re-recording a known series must not trigger overflow"
        );
        // Cardinality does not grow.
        assert_eq!(guard.series_count_for("cpu_usage"), 1);
    }

    #[test]
    fn test_multiple_series_same_metric() {
        let mut guard = CardinalityGuard::new(CardinalityConfig::default_safe());
        for i in 0..10 {
            let combo = format!("host=node{i}");
            let r = guard.check_and_record("cpu_usage", &combo);
            assert!(matches!(r, CardinalityResult::Allowed(_)));
        }
        assert_eq!(guard.series_count_for("cpu_usage"), 10);
        assert_eq!(guard.total_series(), 10);
    }

    // ── Drop action ──────────────────────────────────────────────────────────

    #[test]
    fn test_over_limit_drop_returns_overflow_with_original_label() {
        let cfg = CardinalityConfig {
            max_label_values_per_metric: 2,
            max_total_series: 10_000,
            overflow_action: OverflowAction::Drop,
        };
        let mut guard = CardinalityGuard::new(cfg);
        guard.check_and_record("m", "a");
        guard.check_and_record("m", "b");
        let result = guard.check_and_record("m", "c");
        assert_eq!(result, CardinalityResult::Overflow("c".to_string()));
        // Cardinality must not grow.
        assert_eq!(guard.series_count_for("m"), 2);
    }

    // ── UseOverflow action ───────────────────────────────────────────────────

    #[test]
    fn test_over_limit_use_overflow_returns_sentinel() {
        let cfg = CardinalityConfig {
            max_label_values_per_metric: 1,
            max_total_series: 10_000,
            overflow_action: OverflowAction::UseOverflow,
        };
        let mut guard = CardinalityGuard::new(cfg);
        guard.check_and_record("m", "a");
        let result = guard.check_and_record("m", "b");
        assert_eq!(
            result,
            CardinalityResult::Overflow("__overflow__".to_string())
        );
    }

    // ── Error action ─────────────────────────────────────────────────────────

    #[test]
    fn test_over_limit_error_action() {
        let cfg = CardinalityConfig {
            max_label_values_per_metric: 1,
            max_total_series: 10_000,
            overflow_action: OverflowAction::Error,
        };
        let mut guard = CardinalityGuard::new(cfg);
        guard.check_and_record("m", "a");
        let result = guard.check_and_record("m", "b");
        match result {
            CardinalityResult::Error(msg) => {
                assert!(msg.contains("cardinality limit"), "error should describe the limit");
                assert!(msg.contains("'m'"), "error should include the metric name");
            }
            other => panic!("expected Error, got {other:?}"),
        }
    }

    // ── Total series limit ───────────────────────────────────────────────────

    #[test]
    fn test_total_series_limit() {
        let cfg = CardinalityConfig {
            max_label_values_per_metric: 1_000,
            max_total_series: 3,
            overflow_action: OverflowAction::Drop,
        };
        let mut guard = CardinalityGuard::new(cfg);
        // Different metric names, each with one series.
        guard.check_and_record("a", "x");
        guard.check_and_record("b", "y");
        guard.check_and_record("c", "z");
        assert_eq!(guard.total_series(), 3);

        // 4th series should overflow.
        let result = guard.check_and_record("d", "w");
        assert!(
            matches!(result, CardinalityResult::Overflow(_)),
            "total series limit should be enforced"
        );
    }

    // ── reset ────────────────────────────────────────────────────────────────

    #[test]
    fn test_reset_clears_state() {
        let mut guard = CardinalityGuard::new(CardinalityConfig::default_safe());
        guard.check_and_record("cpu", "host=a");
        guard.check_and_record("mem", "host=b");
        assert_eq!(guard.total_series(), 2);

        guard.reset();
        assert_eq!(guard.total_series(), 0);
        assert_eq!(guard.series_count_for("cpu"), 0);

        // After reset, same combos can be re-admitted.
        let r = guard.check_and_record("cpu", "host=a");
        assert!(matches!(r, CardinalityResult::Allowed(_)));
    }
}

// ---------------------------------------------------------------------------
// Tests for the TODO-specified types
// ---------------------------------------------------------------------------

#[cfg(test)]
mod todo_api_tests {
    use super::*;

    fn labels(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn test_cardinality_limit_config_defaults() {
        let cfg = CardinalityLimitConfig::default();
        assert_eq!(cfg.max_series, 10_000);
        assert_eq!(cfg.max_labels_per_metric, 20);
        assert_eq!(cfg.max_label_values, 100);
    }

    #[test]
    fn test_check_series_admits_new_series() {
        let mut guard = CardinalityLimiter::new(CardinalityLimitConfig::default());
        let lbls = labels(&[("host", "node1")]);
        assert!(check_series(&mut guard, "cpu_usage", &lbls));
        assert_eq!(guard.series_count(), 1);
    }

    #[test]
    fn test_check_series_idempotent_for_same_series() {
        let mut guard = CardinalityLimiter::new(CardinalityLimitConfig::default());
        let lbls = labels(&[("host", "node1")]);
        assert!(check_series(&mut guard, "cpu_usage", &lbls));
        assert!(check_series(&mut guard, "cpu_usage", &lbls));
        // Re-ingesting the same series must not increase the count.
        assert_eq!(guard.series_count(), 1);
    }

    #[test]
    fn test_check_series_blocks_when_max_series_reached() {
        let cfg = CardinalityLimitConfig {
            max_series: 2,
            max_labels_per_metric: 20,
            max_label_values: 100,
        };
        let mut guard = CardinalityLimiter::new(cfg);
        assert!(check_series(&mut guard, "m", &labels(&[("h", "a")])));
        assert!(check_series(&mut guard, "m", &labels(&[("h", "b")])));
        // Third unique series should be blocked.
        assert!(
            !check_series(&mut guard, "m", &labels(&[("h", "c")])),
            "must be blocked when max_series is reached"
        );
        assert_eq!(guard.series_count(), 2);
    }

    #[test]
    fn test_check_series_blocks_when_max_label_values_reached() {
        let cfg = CardinalityLimitConfig {
            max_series: 10_000,
            max_labels_per_metric: 20,
            max_label_values: 2,
        };
        let mut guard = CardinalityLimiter::new(cfg);
        assert!(check_series(&mut guard, "m", &labels(&[("host", "a")])));
        assert!(check_series(&mut guard, "m", &labels(&[("host", "b")])));
        // Third value for "host" should be blocked.
        assert!(
            !check_series(&mut guard, "m", &labels(&[("host", "c")])),
            "must be blocked when max_label_values is reached"
        );
    }

    #[test]
    fn test_label_value_count_after_ingestion() {
        let mut guard = CardinalityLimiter::new(CardinalityLimitConfig::default());
        check_series(&mut guard, "cpu", &labels(&[("host", "n1")]));
        check_series(&mut guard, "cpu", &labels(&[("host", "n2")]));
        check_series(&mut guard, "cpu", &labels(&[("host", "n3")]));
        assert_eq!(guard.label_value_count("cpu", "host"), 3);
    }

    #[test]
    fn test_series_count_across_metrics() {
        let mut guard = CardinalityLimiter::new(CardinalityLimitConfig::default());
        check_series(&mut guard, "cpu", &labels(&[("h", "a")]));
        check_series(&mut guard, "mem", &labels(&[("h", "a")]));
        check_series(&mut guard, "disk", &labels(&[("h", "a")]));
        assert_eq!(guard.series_count(), 3);
    }

    #[test]
    fn test_too_many_labels_per_metric_blocked() {
        let cfg = CardinalityLimitConfig {
            max_series: 10_000,
            max_labels_per_metric: 2,
            max_label_values: 100,
        };
        let mut guard = CardinalityLimiter::new(cfg);
        let too_many = labels(&[("a", "1"), ("b", "2"), ("c", "3")]);
        assert!(
            !check_series(&mut guard, "m", &too_many),
            "must be blocked when label count exceeds max_labels_per_metric"
        );
    }
}
