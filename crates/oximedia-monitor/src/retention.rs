//! Metric retention policies: downsampling, rolling windows, and archival rules.
//!
//! This module defines how long different categories of metrics are kept,
//! how high-resolution data is downsampled into coarser buckets for long-term
//! storage, and which metrics qualify for archival versus eviction.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::Duration;

/// Resolution tier for downsampled data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResolutionTier {
    /// Raw samples at full collection frequency (e.g. 1 s).
    Raw,
    /// 1-minute aggregates.
    OneMinute,
    /// 5-minute aggregates.
    FiveMinute,
    /// 1-hour aggregates.
    OneHour,
    /// 1-day aggregates.
    OneDay,
}

impl ResolutionTier {
    /// Nominal sample interval for this tier.
    #[must_use]
    pub const fn interval(&self) -> Duration {
        match self {
            ResolutionTier::Raw => Duration::from_secs(1),
            ResolutionTier::OneMinute => Duration::from_secs(60),
            ResolutionTier::FiveMinute => Duration::from_secs(300),
            ResolutionTier::OneHour => Duration::from_secs(3_600),
            ResolutionTier::OneDay => Duration::from_secs(86_400),
        }
    }

    /// Human-readable label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            ResolutionTier::Raw => "raw",
            ResolutionTier::OneMinute => "1m",
            ResolutionTier::FiveMinute => "5m",
            ResolutionTier::OneHour => "1h",
            ResolutionTier::OneDay => "1d",
        }
    }
}

/// Aggregation function applied when downsampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationFn {
    /// Arithmetic mean.
    Mean,
    /// Maximum value observed.
    Max,
    /// Minimum value observed.
    Min,
    /// Sum of all values.
    Sum,
    /// Last observed value.
    Last,
    /// 95th percentile.
    P95,
}

/// Downsampling rule: from a source tier produce a target tier using an aggregation.
#[derive(Debug, Clone)]
pub struct DownsampleRule {
    /// Source tier (higher resolution).
    pub source: ResolutionTier,
    /// Target tier (lower resolution).
    pub target: ResolutionTier,
    /// Aggregation function used to merge source samples.
    pub aggregation: AggregationFn,
}

impl DownsampleRule {
    /// Create a new downsample rule.
    #[must_use]
    pub const fn new(
        source: ResolutionTier,
        target: ResolutionTier,
        aggregation: AggregationFn,
    ) -> Self {
        Self {
            source,
            target,
            aggregation,
        }
    }
}

/// How long data at a given tier is retained before eviction.
#[derive(Debug, Clone)]
pub struct RetentionWindow {
    /// The resolution tier this window applies to.
    pub tier: ResolutionTier,
    /// How long data is kept.
    pub keep_for: Duration,
    /// Whether matching data should be archived (copied to cold storage) before eviction.
    pub archive_before_evict: bool,
}

impl RetentionWindow {
    /// Create a window that evicts without archiving.
    #[must_use]
    pub const fn evict_only(tier: ResolutionTier, keep_for: Duration) -> Self {
        Self {
            tier,
            keep_for,
            archive_before_evict: false,
        }
    }

    /// Create a window that archives before eviction.
    #[must_use]
    pub const fn archive_then_evict(tier: ResolutionTier, keep_for: Duration) -> Self {
        Self {
            tier,
            keep_for,
            archive_before_evict: true,
        }
    }
}

/// A complete retention policy for a named metric category.
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Identifier for this policy (e.g. `"system.cpu"`, `"encoding.*"`).
    pub name: String,
    /// Glob-style pattern matched against metric names.
    pub pattern: String,
    /// Per-tier retention windows, keyed by tier.
    pub windows: Vec<RetentionWindow>,
    /// Ordered downsampling rules applied in sequence.
    pub downsample_rules: Vec<DownsampleRule>,
}

impl RetentionPolicy {
    /// Create a new policy with no windows or rules.
    #[must_use]
    pub fn new(name: impl Into<String>, pattern: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            pattern: pattern.into(),
            windows: Vec::new(),
            downsample_rules: Vec::new(),
        }
    }

    /// Add a retention window.
    pub fn add_window(&mut self, window: RetentionWindow) {
        self.windows.push(window);
    }

    /// Add a downsampling rule.
    pub fn add_rule(&mut self, rule: DownsampleRule) {
        self.downsample_rules.push(rule);
    }

    /// Return the retention window for the given tier, if configured.
    #[must_use]
    pub fn window_for(&self, tier: ResolutionTier) -> Option<&RetentionWindow> {
        self.windows.iter().find(|w| w.tier == tier)
    }

    /// Simple glob match: supports only leading/trailing `*`.
    #[must_use]
    pub fn matches(&self, metric_name: &str) -> bool {
        let pat = &self.pattern;
        if pat == "*" {
            return true;
        }
        if let Some(suffix) = pat.strip_prefix('*') {
            return metric_name.ends_with(suffix);
        }
        if let Some(prefix) = pat.strip_suffix('*') {
            return metric_name.starts_with(prefix);
        }
        pat == metric_name
    }
}

/// Registry of all retention policies.
#[derive(Debug, Default)]
pub struct RetentionRegistry {
    policies: Vec<RetentionPolicy>,
    /// Cache: metric name → index of first matching policy.
    cache: HashMap<String, usize>,
}

impl RetentionRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a policy. Policies are evaluated in insertion order.
    pub fn register(&mut self, policy: RetentionPolicy) {
        self.cache.clear();
        self.policies.push(policy);
    }

    /// Return the first policy matching `metric_name`, if any.
    #[must_use]
    pub fn policy_for(&mut self, metric_name: &str) -> Option<&RetentionPolicy> {
        if let Some(&idx) = self.cache.get(metric_name) {
            return self.policies.get(idx);
        }
        let idx = self.policies.iter().position(|p| p.matches(metric_name))?;
        self.cache.insert(metric_name.to_owned(), idx);
        self.policies.get(idx)
    }

    /// Total number of registered policies.
    #[must_use]
    pub fn len(&self) -> usize {
        self.policies.len()
    }

    /// Return `true` if no policies are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.policies.is_empty()
    }
}

/// Build the default retention registry used by the monitor.
#[must_use]
pub fn default_registry() -> RetentionRegistry {
    let mut reg = RetentionRegistry::new();

    // System metrics: keep raw for 1 h, 1-min for 7 days, 5-min for 30 days.
    let mut sys = RetentionPolicy::new("system", "system.*");
    sys.add_window(RetentionWindow::evict_only(
        ResolutionTier::Raw,
        Duration::from_secs(3_600),
    ));
    sys.add_window(RetentionWindow::evict_only(
        ResolutionTier::OneMinute,
        Duration::from_secs(7 * 86_400),
    ));
    sys.add_window(RetentionWindow::archive_then_evict(
        ResolutionTier::FiveMinute,
        Duration::from_secs(30 * 86_400),
    ));
    sys.add_rule(DownsampleRule::new(
        ResolutionTier::Raw,
        ResolutionTier::OneMinute,
        AggregationFn::Mean,
    ));
    sys.add_rule(DownsampleRule::new(
        ResolutionTier::OneMinute,
        ResolutionTier::FiveMinute,
        AggregationFn::Mean,
    ));
    reg.register(sys);

    // Quality metrics: keep raw for 6 h, 5-min for 90 days.
    let mut qos = RetentionPolicy::new("quality", "quality.*");
    qos.add_window(RetentionWindow::evict_only(
        ResolutionTier::Raw,
        Duration::from_secs(6 * 3_600),
    ));
    qos.add_window(RetentionWindow::archive_then_evict(
        ResolutionTier::FiveMinute,
        Duration::from_secs(90 * 86_400),
    ));
    qos.add_rule(DownsampleRule::new(
        ResolutionTier::Raw,
        ResolutionTier::FiveMinute,
        AggregationFn::P95,
    ));
    reg.register(qos);

    // Catch-all.
    let mut fallback = RetentionPolicy::new("default", "*");
    fallback.add_window(RetentionWindow::evict_only(
        ResolutionTier::Raw,
        Duration::from_secs(86_400),
    ));
    reg.register(fallback);

    reg
}

// ── aggregation helpers ──────────────────────────────────────────────────────

/// Aggregate a slice of `f64` values according to `fn`.
/// Returns `None` for empty slices.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn aggregate(values: &[f64], func: AggregationFn) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    match func {
        AggregationFn::Mean => {
            let sum: f64 = values.iter().sum();
            Some(sum / values.len() as f64)
        }
        AggregationFn::Max => values.iter().copied().reduce(f64::max),
        AggregationFn::Min => values.iter().copied().reduce(f64::min),
        AggregationFn::Sum => Some(values.iter().sum()),
        AggregationFn::Last => values.last().copied(),
        AggregationFn::P95 => {
            let mut sorted = values.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = (((sorted.len() - 1) as f64 * 0.95) as usize).min(sorted.len() - 1);
            Some(sorted[idx])
        }
    }
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolution_tier_intervals() {
        assert_eq!(ResolutionTier::Raw.interval(), Duration::from_secs(1));
        assert_eq!(
            ResolutionTier::OneMinute.interval(),
            Duration::from_secs(60)
        );
        assert_eq!(
            ResolutionTier::OneHour.interval(),
            Duration::from_secs(3_600)
        );
        assert_eq!(
            ResolutionTier::OneDay.interval(),
            Duration::from_secs(86_400)
        );
    }

    #[test]
    fn test_resolution_tier_labels() {
        assert_eq!(ResolutionTier::Raw.label(), "raw");
        assert_eq!(ResolutionTier::OneMinute.label(), "1m");
        assert_eq!(ResolutionTier::FiveMinute.label(), "5m");
        assert_eq!(ResolutionTier::OneHour.label(), "1h");
        assert_eq!(ResolutionTier::OneDay.label(), "1d");
    }

    #[test]
    fn test_policy_exact_match() {
        let policy = RetentionPolicy::new("test", "system.cpu");
        assert!(policy.matches("system.cpu"));
        assert!(!policy.matches("system.mem"));
    }

    #[test]
    fn test_policy_prefix_glob() {
        let policy = RetentionPolicy::new("test", "system.*");
        assert!(policy.matches("system.cpu"));
        assert!(policy.matches("system.memory"));
        assert!(!policy.matches("quality.psnr"));
    }

    #[test]
    fn test_policy_suffix_glob() {
        let policy = RetentionPolicy::new("test", "*.cpu");
        assert!(policy.matches("system.cpu"));
        assert!(policy.matches("host.cpu"));
        assert!(!policy.matches("system.mem"));
    }

    #[test]
    fn test_policy_wildcard() {
        let policy = RetentionPolicy::new("test", "*");
        assert!(policy.matches("anything.at.all"));
    }

    #[test]
    fn test_policy_add_window() {
        let mut policy = RetentionPolicy::new("test", "system.*");
        policy.add_window(RetentionWindow::evict_only(
            ResolutionTier::Raw,
            Duration::from_secs(3600),
        ));
        assert_eq!(policy.windows.len(), 1);
        assert!(policy.window_for(ResolutionTier::Raw).is_some());
        assert!(policy.window_for(ResolutionTier::OneMinute).is_none());
    }

    #[test]
    fn test_policy_add_rule() {
        let mut policy = RetentionPolicy::new("test", "system.*");
        policy.add_rule(DownsampleRule::new(
            ResolutionTier::Raw,
            ResolutionTier::OneMinute,
            AggregationFn::Mean,
        ));
        assert_eq!(policy.downsample_rules.len(), 1);
    }

    #[test]
    fn test_registry_order_and_first_match() {
        let mut reg = RetentionRegistry::new();
        reg.register(RetentionPolicy::new("specific", "system.cpu"));
        reg.register(RetentionPolicy::new("broad", "system.*"));

        let p = reg
            .policy_for("system.cpu")
            .expect("policy_for should succeed");
        assert_eq!(p.name, "specific");
    }

    #[test]
    fn test_registry_fallback() {
        let mut reg = RetentionRegistry::new();
        reg.register(RetentionPolicy::new("specific", "system.*"));
        reg.register(RetentionPolicy::new("default", "*"));

        let p = reg
            .policy_for("quality.psnr")
            .expect("policy_for should succeed");
        assert_eq!(p.name, "default");
    }

    #[test]
    fn test_registry_empty() {
        let mut reg = RetentionRegistry::new();
        assert!(reg.is_empty());
        assert!(reg.policy_for("anything").is_none());
    }

    #[test]
    fn test_aggregate_mean() {
        let vals = vec![1.0, 2.0, 3.0, 4.0];
        assert!(
            (aggregate(&vals, AggregationFn::Mean).expect("operation should succeed") - 2.5).abs()
                < 1e-9
        );
    }

    #[test]
    fn test_aggregate_max_min() {
        let vals = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        assert!(
            (aggregate(&vals, AggregationFn::Max).expect("operation should succeed") - 5.0).abs()
                < 1e-9
        );
        assert!(
            (aggregate(&vals, AggregationFn::Min).expect("operation should succeed") - 1.0).abs()
                < 1e-9
        );
    }

    #[test]
    fn test_aggregate_sum() {
        let vals = vec![1.0, 2.0, 3.0];
        assert!(
            (aggregate(&vals, AggregationFn::Sum).expect("operation should succeed") - 6.0).abs()
                < 1e-9
        );
    }

    #[test]
    fn test_aggregate_last() {
        let vals = vec![10.0, 20.0, 30.0];
        assert!(
            (aggregate(&vals, AggregationFn::Last).expect("operation should succeed") - 30.0).abs()
                < 1e-9
        );
    }

    #[test]
    fn test_aggregate_p95() {
        let vals: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let p95 = aggregate(&vals, AggregationFn::P95).expect("operation should succeed");
        // 95th percentile of 1..=100 should be 95.
        assert!((p95 - 95.0).abs() < 1.0);
    }

    #[test]
    fn test_aggregate_empty() {
        assert!(aggregate(&[], AggregationFn::Mean).is_none());
        assert!(aggregate(&[], AggregationFn::Max).is_none());
    }

    #[test]
    fn test_default_registry_has_policies() {
        let mut reg = default_registry();
        assert!(!reg.is_empty());
        assert!(reg.policy_for("system.cpu").is_some());
        assert!(reg.policy_for("quality.psnr").is_some());
        assert!(reg.policy_for("some.unknown.metric").is_some()); // fallback
    }

    #[test]
    fn test_retention_window_archive_flag() {
        let w = RetentionWindow::archive_then_evict(ResolutionTier::Raw, Duration::from_secs(100));
        assert!(w.archive_before_evict);
        let w2 = RetentionWindow::evict_only(ResolutionTier::Raw, Duration::from_secs(100));
        assert!(!w2.archive_before_evict);
    }
}
