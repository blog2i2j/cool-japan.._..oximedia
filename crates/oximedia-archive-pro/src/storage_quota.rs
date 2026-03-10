#![allow(dead_code)]
//! Storage quota management for archive systems.
//!
//! Tracks and enforces storage limits across archive tiers, users, and projects.
//! Supports warnings, hard limits, and quota usage reporting.

use std::collections::HashMap;

/// Unit multipliers for storage sizes.
const KB: u64 = 1024;
const MB: u64 = KB * 1024;
const GB: u64 = MB * 1024;
const TB: u64 = GB * 1024;

/// Identifies a quota scope (who the quota applies to).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum QuotaScope {
    /// Global scope across the entire archive system.
    Global,
    /// Per-user quota identified by user name.
    User(String),
    /// Per-project quota identified by project name.
    Project(String),
    /// Per storage tier.
    Tier(String),
}

/// Policy for what happens when a quota is exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuotaPolicy {
    /// Only warn; do not block writes.
    WarnOnly,
    /// Block new writes when hard limit is reached.
    HardLimit,
    /// Automatically archive old data to free space.
    AutoArchive,
}

/// Configuration for a single quota.
#[derive(Debug, Clone)]
pub struct QuotaConfig {
    /// The scope this quota applies to.
    pub scope: QuotaScope,
    /// Soft limit in bytes (triggers warnings).
    pub soft_limit_bytes: u64,
    /// Hard limit in bytes (blocks writes if policy is `HardLimit`).
    pub hard_limit_bytes: u64,
    /// Enforcement policy.
    pub policy: QuotaPolicy,
}

impl QuotaConfig {
    /// Creates a new quota configuration.
    #[must_use]
    pub fn new(scope: QuotaScope, soft_limit_bytes: u64, hard_limit_bytes: u64) -> Self {
        Self {
            scope,
            soft_limit_bytes,
            hard_limit_bytes,
            policy: QuotaPolicy::HardLimit,
        }
    }

    /// Sets the enforcement policy.
    #[must_use]
    pub fn with_policy(mut self, policy: QuotaPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Creates a quota from gigabyte values.
    #[must_use]
    pub fn from_gb(scope: QuotaScope, soft_gb: u64, hard_gb: u64) -> Self {
        Self::new(scope, soft_gb * GB, hard_gb * GB)
    }

    /// Creates a quota from terabyte values.
    #[must_use]
    pub fn from_tb(scope: QuotaScope, soft_tb: u64, hard_tb: u64) -> Self {
        Self::new(scope, soft_tb * TB, hard_tb * TB)
    }
}

/// Current usage snapshot for a quota scope.
#[derive(Debug, Clone)]
pub struct QuotaUsage {
    /// Current used bytes.
    pub used_bytes: u64,
    /// Number of files.
    pub file_count: u64,
    /// Timestamp of last update (seconds since epoch).
    pub last_updated: u64,
}

impl QuotaUsage {
    /// Creates a new usage record.
    #[must_use]
    pub fn new(used_bytes: u64, file_count: u64) -> Self {
        Self {
            used_bytes,
            file_count,
            last_updated: 0,
        }
    }

    /// Returns the usage as a percentage of the given limit (0.0 to 100.0+).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn percentage_of(&self, limit_bytes: u64) -> f64 {
        if limit_bytes == 0 {
            return if self.used_bytes > 0 {
                f64::INFINITY
            } else {
                0.0
            };
        }
        (self.used_bytes as f64 / limit_bytes as f64) * 100.0
    }

    /// Returns human-readable used size.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn human_readable_used(&self) -> String {
        format_bytes(self.used_bytes)
    }
}

/// Formats byte count into human-readable string.
#[allow(clippy::cast_precision_loss)]
fn format_bytes(bytes: u64) -> String {
    if bytes >= TB {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Result of a quota check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuotaCheckResult {
    /// Usage is within soft limit.
    Ok,
    /// Usage exceeds soft limit but is under hard limit.
    SoftLimitExceeded,
    /// Usage exceeds hard limit.
    HardLimitExceeded,
    /// No quota configured for this scope.
    NoQuota,
}

/// Summary report for a single quota.
#[derive(Debug, Clone)]
pub struct QuotaReport {
    /// The scope of the quota.
    pub scope: QuotaScope,
    /// Used bytes.
    pub used_bytes: u64,
    /// Soft limit bytes.
    pub soft_limit_bytes: u64,
    /// Hard limit bytes.
    pub hard_limit_bytes: u64,
    /// Percentage of hard limit used.
    pub usage_percent: f64,
    /// Check result.
    pub status: QuotaCheckResult,
}

/// Manages storage quotas across the archive system.
#[derive(Debug, Default)]
pub struct QuotaManager {
    /// Configured quotas indexed by scope.
    configs: HashMap<QuotaScope, QuotaConfig>,
    /// Current usage indexed by scope.
    usage: HashMap<QuotaScope, QuotaUsage>,
}

impl QuotaManager {
    /// Creates a new empty quota manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a quota configuration.
    pub fn add_quota(&mut self, config: QuotaConfig) {
        self.configs.insert(config.scope.clone(), config);
    }

    /// Updates the current usage for a scope.
    pub fn update_usage(&mut self, scope: QuotaScope, usage: QuotaUsage) {
        self.usage.insert(scope, usage);
    }

    /// Records additional bytes used in a scope.
    pub fn add_bytes(&mut self, scope: &QuotaScope, bytes: u64) {
        let entry = self
            .usage
            .entry(scope.clone())
            .or_insert_with(|| QuotaUsage::new(0, 0));
        entry.used_bytes += bytes;
        entry.file_count += 1;
    }

    /// Removes bytes from a scope's usage.
    pub fn remove_bytes(&mut self, scope: &QuotaScope, bytes: u64) {
        if let Some(usage) = self.usage.get_mut(scope) {
            usage.used_bytes = usage.used_bytes.saturating_sub(bytes);
            usage.file_count = usage.file_count.saturating_sub(1);
        }
    }

    /// Checks whether a write of the given size is allowed.
    #[must_use]
    pub fn check(&self, scope: &QuotaScope, additional_bytes: u64) -> QuotaCheckResult {
        let config = match self.configs.get(scope) {
            Some(c) => c,
            None => return QuotaCheckResult::NoQuota,
        };
        let current = self.usage.get(scope).map_or(0, |u| u.used_bytes);
        let projected = current + additional_bytes;

        if projected > config.hard_limit_bytes {
            QuotaCheckResult::HardLimitExceeded
        } else if projected > config.soft_limit_bytes {
            QuotaCheckResult::SoftLimitExceeded
        } else {
            QuotaCheckResult::Ok
        }
    }

    /// Returns whether a write of given size should be blocked.
    #[must_use]
    pub fn is_blocked(&self, scope: &QuotaScope, additional_bytes: u64) -> bool {
        let config = match self.configs.get(scope) {
            Some(c) => c,
            None => return false,
        };
        if config.policy != QuotaPolicy::HardLimit {
            return false;
        }
        self.check(scope, additional_bytes) == QuotaCheckResult::HardLimitExceeded
    }

    /// Returns the remaining bytes before hard limit for a scope.
    #[must_use]
    pub fn remaining_bytes(&self, scope: &QuotaScope) -> Option<u64> {
        let config = self.configs.get(scope)?;
        let current = self.usage.get(scope).map_or(0, |u| u.used_bytes);
        Some(config.hard_limit_bytes.saturating_sub(current))
    }

    /// Generates a report for a specific scope.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn report(&self, scope: &QuotaScope) -> Option<QuotaReport> {
        let config = self.configs.get(scope)?;
        let used = self.usage.get(scope).map_or(0, |u| u.used_bytes);
        let percent = if config.hard_limit_bytes > 0 {
            (used as f64 / config.hard_limit_bytes as f64) * 100.0
        } else {
            0.0
        };
        Some(QuotaReport {
            scope: scope.clone(),
            used_bytes: used,
            soft_limit_bytes: config.soft_limit_bytes,
            hard_limit_bytes: config.hard_limit_bytes,
            usage_percent: percent,
            status: self.check(scope, 0),
        })
    }

    /// Returns the number of configured quotas.
    #[must_use]
    pub fn quota_count(&self) -> usize {
        self.configs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes_small() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(0), "0 B");
    }

    #[test]
    fn test_format_bytes_large() {
        let result = format_bytes(2 * GB + GB / 2);
        assert!(result.contains("GB"));
        let result_tb = format_bytes(3 * TB);
        assert!(result_tb.contains("TB"));
    }

    #[test]
    fn test_quota_config_from_gb() {
        let cfg = QuotaConfig::from_gb(QuotaScope::Global, 80, 100);
        assert_eq!(cfg.soft_limit_bytes, 80 * GB);
        assert_eq!(cfg.hard_limit_bytes, 100 * GB);
    }

    #[test]
    fn test_quota_config_from_tb() {
        let cfg = QuotaConfig::from_tb(QuotaScope::Global, 1, 2);
        assert_eq!(cfg.soft_limit_bytes, TB);
        assert_eq!(cfg.hard_limit_bytes, 2 * TB);
    }

    #[test]
    fn test_quota_config_with_policy() {
        let cfg = QuotaConfig::new(QuotaScope::Global, 100, 200).with_policy(QuotaPolicy::WarnOnly);
        assert_eq!(cfg.policy, QuotaPolicy::WarnOnly);
    }

    #[test]
    fn test_usage_percentage() {
        let usage = QuotaUsage::new(50 * GB, 10);
        let pct = usage.percentage_of(100 * GB);
        assert!((pct - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_usage_percentage_zero_limit() {
        let usage = QuotaUsage::new(100, 1);
        let pct = usage.percentage_of(0);
        assert!(pct.is_infinite());

        let empty = QuotaUsage::new(0, 0);
        let pct2 = empty.percentage_of(0);
        assert!((pct2 - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_manager_check_ok() {
        let mut mgr = QuotaManager::new();
        let scope = QuotaScope::User("alice".into());
        mgr.add_quota(QuotaConfig::from_gb(scope.clone(), 80, 100));
        mgr.update_usage(scope.clone(), QuotaUsage::new(50 * GB, 5));

        assert_eq!(mgr.check(&scope, 10 * GB), QuotaCheckResult::Ok);
    }

    #[test]
    fn test_manager_check_soft_exceeded() {
        let mut mgr = QuotaManager::new();
        let scope = QuotaScope::User("bob".into());
        mgr.add_quota(QuotaConfig::from_gb(scope.clone(), 80, 100));
        mgr.update_usage(scope.clone(), QuotaUsage::new(75 * GB, 5));

        assert_eq!(
            mgr.check(&scope, 10 * GB),
            QuotaCheckResult::SoftLimitExceeded
        );
    }

    #[test]
    fn test_manager_check_hard_exceeded() {
        let mut mgr = QuotaManager::new();
        let scope = QuotaScope::Project("big".into());
        mgr.add_quota(QuotaConfig::from_gb(scope.clone(), 80, 100));
        mgr.update_usage(scope.clone(), QuotaUsage::new(95 * GB, 50));

        assert_eq!(
            mgr.check(&scope, 10 * GB),
            QuotaCheckResult::HardLimitExceeded
        );
    }

    #[test]
    fn test_manager_no_quota() {
        let mgr = QuotaManager::new();
        let scope = QuotaScope::User("unknown".into());
        assert_eq!(mgr.check(&scope, 1), QuotaCheckResult::NoQuota);
    }

    #[test]
    fn test_manager_is_blocked() {
        let mut mgr = QuotaManager::new();
        let scope = QuotaScope::Global;
        mgr.add_quota(QuotaConfig::new(scope.clone(), 500, 1000));
        mgr.update_usage(scope.clone(), QuotaUsage::new(990, 10));

        assert!(mgr.is_blocked(&scope, 20));
        assert!(!mgr.is_blocked(&scope, 5));
    }

    #[test]
    fn test_manager_remaining_bytes() {
        let mut mgr = QuotaManager::new();
        let scope = QuotaScope::Tier("hot".into());
        mgr.add_quota(QuotaConfig::new(scope.clone(), 500, 1000));
        mgr.update_usage(scope.clone(), QuotaUsage::new(300, 3));

        assert_eq!(mgr.remaining_bytes(&scope), Some(700));
    }

    #[test]
    fn test_manager_add_remove_bytes() {
        let mut mgr = QuotaManager::new();
        let scope = QuotaScope::Global;
        mgr.add_quota(QuotaConfig::new(scope.clone(), 500, 1000));

        mgr.add_bytes(&scope, 200);
        mgr.add_bytes(&scope, 100);
        assert_eq!(mgr.remaining_bytes(&scope), Some(700));

        mgr.remove_bytes(&scope, 50);
        assert_eq!(mgr.remaining_bytes(&scope), Some(750));
    }

    #[test]
    fn test_manager_report() {
        let mut mgr = QuotaManager::new();
        let scope = QuotaScope::User("eve".into());
        mgr.add_quota(QuotaConfig::from_gb(scope.clone(), 80, 100));
        mgr.update_usage(scope.clone(), QuotaUsage::new(50 * GB, 25));

        let report = mgr.report(&scope).expect("operation should succeed");
        assert_eq!(report.used_bytes, 50 * GB);
        assert!((report.usage_percent - 50.0).abs() < 0.01);
        assert_eq!(report.status, QuotaCheckResult::Ok);
    }

    #[test]
    fn test_human_readable_used() {
        let usage = QuotaUsage::new(5 * GB + GB / 4, 10);
        let s = usage.human_readable_used();
        assert!(s.contains("GB"));
    }
}
