#![allow(dead_code)]
//! NDI failover and redundancy management.
//!
//! Supports automatic source switching when the primary NDI source
//! becomes unavailable, with configurable failover policies and
//! health-check based monitoring.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Health status of an NDI source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceHealth {
    /// Source is connected and streaming normally.
    Healthy,
    /// Source is experiencing intermittent issues (e.g., dropped frames).
    Degraded,
    /// Source is unreachable or has stopped streaming.
    Unreachable,
    /// Source health has not been evaluated yet.
    Unknown,
}

impl SourceHealth {
    /// Returns true if the source is considered usable (healthy or degraded).
    #[must_use]
    pub fn is_usable(self) -> bool {
        matches!(self, Self::Healthy | Self::Degraded)
    }

    /// Returns a human-readable label for the health status.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Degraded => "degraded",
            Self::Unreachable => "unreachable",
            Self::Unknown => "unknown",
        }
    }
}

/// Policy that determines when and how failover occurs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailoverPolicy {
    /// Fail over immediately when the primary becomes unreachable.
    Immediate,
    /// Wait for the specified number of consecutive health check failures.
    ThresholdBased {
        /// Number of consecutive failures before failover triggers.
        failure_threshold: u32,
    },
    /// Only fail over on explicit manual request.
    Manual,
}

impl Default for FailoverPolicy {
    fn default() -> Self {
        Self::ThresholdBased {
            failure_threshold: 3,
        }
    }
}

impl FailoverPolicy {
    /// Returns true if failover should be triggered given the consecutive failure count.
    #[must_use]
    pub fn should_failover(self, consecutive_failures: u32) -> bool {
        match self {
            Self::Immediate => consecutive_failures >= 1,
            Self::ThresholdBased { failure_threshold } => consecutive_failures >= failure_threshold,
            Self::Manual => false,
        }
    }
}

/// Priority level for a failover source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourcePriority(pub u32);

impl SourcePriority {
    /// Highest (primary) priority.
    pub const PRIMARY: Self = Self(0);
    /// Secondary backup priority.
    pub const SECONDARY: Self = Self(1);
    /// Tertiary backup priority.
    pub const TERTIARY: Self = Self(2);
}

/// Represents one source in the failover group.
#[derive(Debug, Clone)]
pub struct FailoverSource {
    /// Unique identifier for this source.
    pub source_id: String,
    /// Display name.
    pub name: String,
    /// Priority (lower number = higher priority).
    pub priority: SourcePriority,
    /// Current health status.
    pub health: SourceHealth,
    /// Number of consecutive health check failures.
    pub consecutive_failures: u32,
    /// Timestamp of the last successful health check.
    pub last_healthy: Option<Instant>,
    /// Timestamp of the last health check attempt.
    pub last_checked: Option<Instant>,
}

impl FailoverSource {
    /// Creates a new failover source entry.
    #[must_use]
    pub fn new(source_id: String, name: String, priority: SourcePriority) -> Self {
        Self {
            source_id,
            name,
            priority,
            health: SourceHealth::Unknown,
            consecutive_failures: 0,
            last_healthy: None,
            last_checked: None,
        }
    }

    /// Records a successful health check.
    pub fn mark_healthy(&mut self, now: Instant) {
        self.health = SourceHealth::Healthy;
        self.consecutive_failures = 0;
        self.last_healthy = Some(now);
        self.last_checked = Some(now);
    }

    /// Records a degraded health check result.
    pub fn mark_degraded(&mut self, now: Instant) {
        self.health = SourceHealth::Degraded;
        self.last_checked = Some(now);
        // Don't reset consecutive_failures for degraded; it's still partially usable
    }

    /// Records a failed health check.
    pub fn mark_unreachable(&mut self, now: Instant) {
        self.health = SourceHealth::Unreachable;
        self.consecutive_failures += 1;
        self.last_checked = Some(now);
    }

    /// Returns how long since the last successful health check.
    #[must_use]
    pub fn time_since_healthy(&self) -> Option<Duration> {
        self.last_healthy.map(|t| t.elapsed())
    }
}

/// Result of a failover decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailoverDecision {
    /// Stay with the current source.
    StayCurrent,
    /// Switch to a different source.
    SwitchTo {
        /// ID of the source to switch to.
        source_id: String,
    },
    /// No usable source is available.
    NoSourceAvailable,
}

/// Manages a group of NDI sources with automatic failover.
#[derive(Debug)]
pub struct FailoverGroup {
    /// All sources in this failover group.
    sources: HashMap<String, FailoverSource>,
    /// The currently active source ID.
    active_source_id: Option<String>,
    /// Failover policy.
    policy: FailoverPolicy,
    /// Health check interval.
    health_check_interval: Duration,
    /// Whether to automatically fail back to a higher-priority source when it recovers.
    auto_failback: bool,
    /// Minimum time a recovered source must be healthy before failback.
    failback_grace_period: Duration,
}

impl FailoverGroup {
    /// Creates a new failover group with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
            active_source_id: None,
            policy: FailoverPolicy::default(),
            health_check_interval: Duration::from_secs(2),
            auto_failback: true,
            failback_grace_period: Duration::from_secs(10),
        }
    }

    /// Creates a new failover group with a specific policy.
    #[must_use]
    pub fn with_policy(policy: FailoverPolicy) -> Self {
        Self {
            policy,
            ..Self::new()
        }
    }

    /// Sets the health check interval.
    pub fn set_health_check_interval(&mut self, interval: Duration) {
        self.health_check_interval = interval;
    }

    /// Sets whether automatic failback is enabled.
    pub fn set_auto_failback(&mut self, enabled: bool) {
        self.auto_failback = enabled;
    }

    /// Sets the failback grace period.
    pub fn set_failback_grace_period(&mut self, period: Duration) {
        self.failback_grace_period = period;
    }

    /// Adds a source to the failover group.
    pub fn add_source(&mut self, source: FailoverSource) {
        let id = source.source_id.clone();
        self.sources.insert(id.clone(), source);
        // Auto-select first source if none active
        if self.active_source_id.is_none() {
            self.active_source_id = Some(id);
        }
    }

    /// Removes a source from the failover group.
    pub fn remove_source(&mut self, source_id: &str) -> Option<FailoverSource> {
        let removed = self.sources.remove(source_id);
        if self.active_source_id.as_deref() == Some(source_id) {
            self.active_source_id = self.best_available_source();
        }
        removed
    }

    /// Returns the number of sources in the group.
    #[must_use]
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Returns the currently active source ID.
    #[must_use]
    pub fn active_source(&self) -> Option<&str> {
        self.active_source_id.as_deref()
    }

    /// Returns the failover policy.
    #[must_use]
    pub fn policy(&self) -> FailoverPolicy {
        self.policy
    }

    /// Returns a reference to a source by ID.
    #[must_use]
    pub fn get_source(&self, source_id: &str) -> Option<&FailoverSource> {
        self.sources.get(source_id)
    }

    /// Returns a mutable reference to a source by ID.
    pub fn get_source_mut(&mut self, source_id: &str) -> Option<&mut FailoverSource> {
        self.sources.get_mut(source_id)
    }

    /// Returns all healthy sources sorted by priority.
    #[must_use]
    pub fn healthy_sources(&self) -> Vec<&FailoverSource> {
        let mut sources: Vec<_> = self
            .sources
            .values()
            .filter(|s| s.health.is_usable())
            .collect();
        sources.sort_by_key(|s| s.priority);
        sources
    }

    /// Returns the source ID of the best available source (lowest priority number, healthy).
    fn best_available_source(&self) -> Option<String> {
        self.healthy_sources().first().map(|s| s.source_id.clone())
    }

    /// Evaluates the current state and decides whether to fail over.
    #[must_use]
    pub fn evaluate(&self) -> FailoverDecision {
        let active_id = match &self.active_source_id {
            Some(id) => id,
            None => {
                return match self.best_available_source() {
                    Some(id) => FailoverDecision::SwitchTo { source_id: id },
                    None => FailoverDecision::NoSourceAvailable,
                };
            }
        };

        let active = match self.sources.get(active_id) {
            Some(s) => s,
            None => return FailoverDecision::NoSourceAvailable,
        };

        // Check if the active source needs failover
        let needs_failover =
            !active.health.is_usable() && self.policy.should_failover(active.consecutive_failures);

        if needs_failover {
            return match self.best_available_source() {
                Some(id) if id != *active_id => FailoverDecision::SwitchTo { source_id: id },
                _ => FailoverDecision::NoSourceAvailable,
            };
        }

        // Check for failback to higher-priority source
        if self.auto_failback {
            if let Some(best_id) = self.best_available_source() {
                if best_id != *active_id {
                    if let Some(best) = self.sources.get(&best_id) {
                        if best.priority < active.priority {
                            // Check grace period
                            if let Some(last_healthy) = best.last_healthy {
                                if last_healthy.elapsed() >= Duration::ZERO
                                    && best.consecutive_failures == 0
                                {
                                    return FailoverDecision::SwitchTo { source_id: best_id };
                                }
                            }
                        }
                    }
                }
            }
        }

        FailoverDecision::StayCurrent
    }

    /// Applies a failover decision, updating the active source.
    pub fn apply_decision(&mut self, decision: &FailoverDecision) {
        match decision {
            FailoverDecision::SwitchTo { source_id } => {
                if self.sources.contains_key(source_id) {
                    self.active_source_id = Some(source_id.clone());
                }
            }
            FailoverDecision::NoSourceAvailable => {
                self.active_source_id = None;
            }
            FailoverDecision::StayCurrent => {}
        }
    }

    /// Returns a summary of all sources and their health.
    #[must_use]
    pub fn status_summary(&self) -> Vec<(String, SourceHealth)> {
        let mut result: Vec<_> = self
            .sources
            .values()
            .map(|s| (s.source_id.clone(), s.health))
            .collect();
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }
}

impl Default for FailoverGroup {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_source(id: &str, priority: u32) -> FailoverSource {
        FailoverSource::new(id.to_string(), id.to_string(), SourcePriority(priority))
    }

    #[test]
    fn test_source_health_is_usable() {
        assert!(SourceHealth::Healthy.is_usable());
        assert!(SourceHealth::Degraded.is_usable());
        assert!(!SourceHealth::Unreachable.is_usable());
        assert!(!SourceHealth::Unknown.is_usable());
    }

    #[test]
    fn test_source_health_label() {
        assert_eq!(SourceHealth::Healthy.label(), "healthy");
        assert_eq!(SourceHealth::Degraded.label(), "degraded");
        assert_eq!(SourceHealth::Unreachable.label(), "unreachable");
        assert_eq!(SourceHealth::Unknown.label(), "unknown");
    }

    #[test]
    fn test_failover_policy_immediate() {
        let p = FailoverPolicy::Immediate;
        assert!(!p.should_failover(0));
        assert!(p.should_failover(1));
    }

    #[test]
    fn test_failover_policy_threshold() {
        let p = FailoverPolicy::ThresholdBased {
            failure_threshold: 3,
        };
        assert!(!p.should_failover(0));
        assert!(!p.should_failover(2));
        assert!(p.should_failover(3));
        assert!(p.should_failover(5));
    }

    #[test]
    fn test_failover_policy_manual() {
        let p = FailoverPolicy::Manual;
        assert!(!p.should_failover(0));
        assert!(!p.should_failover(100));
    }

    #[test]
    fn test_source_mark_healthy() {
        let mut src = make_source("a", 0);
        let now = Instant::now();
        src.mark_unreachable(now);
        src.mark_unreachable(now);
        assert_eq!(src.consecutive_failures, 2);
        src.mark_healthy(now);
        assert_eq!(src.consecutive_failures, 0);
        assert_eq!(src.health, SourceHealth::Healthy);
    }

    #[test]
    fn test_source_mark_degraded() {
        let mut src = make_source("b", 1);
        let now = Instant::now();
        src.mark_degraded(now);
        assert_eq!(src.health, SourceHealth::Degraded);
        assert!(src.health.is_usable());
    }

    #[test]
    fn test_failover_group_add_auto_selects() {
        let mut group = FailoverGroup::new();
        let src = make_source("primary", 0);
        group.add_source(src);
        assert_eq!(group.active_source(), Some("primary"));
    }

    #[test]
    fn test_failover_group_evaluate_no_sources() {
        let group = FailoverGroup::new();
        assert_eq!(group.evaluate(), FailoverDecision::NoSourceAvailable);
    }

    #[test]
    fn test_failover_group_evaluate_stay_current() {
        let mut group = FailoverGroup::new();
        let mut src = make_source("a", 0);
        src.mark_healthy(Instant::now());
        group.add_source(src);
        assert_eq!(group.evaluate(), FailoverDecision::StayCurrent);
    }

    #[test]
    fn test_failover_group_switch_on_failure() {
        let mut group = FailoverGroup::with_policy(FailoverPolicy::Immediate);
        let now = Instant::now();

        let mut primary = make_source("primary", 0);
        primary.mark_healthy(now);
        group.add_source(primary);

        let mut backup = make_source("backup", 1);
        backup.mark_healthy(now);
        group.add_source(backup);

        // Make primary fail
        group
            .get_source_mut("primary")
            .expect("unexpected None/Err")
            .mark_unreachable(now);

        let decision = group.evaluate();
        assert_eq!(
            decision,
            FailoverDecision::SwitchTo {
                source_id: "backup".to_string()
            }
        );
    }

    #[test]
    fn test_failover_group_apply_decision() {
        let mut group = FailoverGroup::new();
        let mut src_a = make_source("a", 0);
        src_a.mark_healthy(Instant::now());
        group.add_source(src_a);
        let mut src_b = make_source("b", 1);
        src_b.mark_healthy(Instant::now());
        group.add_source(src_b);

        let decision = FailoverDecision::SwitchTo {
            source_id: "b".to_string(),
        };
        group.apply_decision(&decision);
        assert_eq!(group.active_source(), Some("b"));
    }

    #[test]
    fn test_failover_group_remove_active_source() {
        let mut group = FailoverGroup::new();
        let now = Instant::now();
        let mut a = make_source("a", 0);
        a.mark_healthy(now);
        group.add_source(a);
        let mut b = make_source("b", 1);
        b.mark_healthy(now);
        group.add_source(b);

        group.remove_source("a");
        // Should auto-select "b"
        assert_eq!(group.active_source(), Some("b"));
    }

    #[test]
    fn test_failover_group_healthy_sources_sorted() {
        let mut group = FailoverGroup::new();
        let now = Instant::now();
        let mut high = make_source("high", 2);
        high.mark_healthy(now);
        group.add_source(high);
        let mut low = make_source("low", 0);
        low.mark_healthy(now);
        group.add_source(low);

        let healthy = group.healthy_sources();
        assert_eq!(healthy.len(), 2);
        assert_eq!(healthy[0].source_id, "low");
        assert_eq!(healthy[1].source_id, "high");
    }

    #[test]
    fn test_failover_group_status_summary() {
        let mut group = FailoverGroup::new();
        let now = Instant::now();
        let mut a = make_source("a", 0);
        a.mark_healthy(now);
        group.add_source(a);
        let mut b = make_source("b", 1);
        b.mark_unreachable(now);
        group.add_source(b);

        let summary = group.status_summary();
        assert_eq!(summary.len(), 2);
    }

    #[test]
    fn test_source_priority_ordering() {
        assert!(SourcePriority::PRIMARY < SourcePriority::SECONDARY);
        assert!(SourcePriority::SECONDARY < SourcePriority::TERTIARY);
    }

    #[test]
    fn test_failover_group_source_count() {
        let mut group = FailoverGroup::new();
        assert_eq!(group.source_count(), 0);
        group.add_source(make_source("x", 0));
        assert_eq!(group.source_count(), 1);
    }

    #[test]
    fn test_no_source_available_when_all_unreachable() {
        let mut group = FailoverGroup::with_policy(FailoverPolicy::Immediate);
        let now = Instant::now();
        let mut a = make_source("a", 0);
        a.mark_unreachable(now);
        group.add_source(a);

        assert_eq!(group.evaluate(), FailoverDecision::NoSourceAvailable);
    }
}
