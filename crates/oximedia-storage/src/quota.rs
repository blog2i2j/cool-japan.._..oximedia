//! Storage quota management for users, projects, departments, and global limits.
//!
//! Provides policy definition, usage tracking, and enforcement of storage quotas
//! across multiple organisational scopes.

#![allow(dead_code)]

use std::collections::HashMap;

/// Policy that defines the limits for a quota scope
#[derive(Debug, Clone)]
pub struct QuotaPolicy {
    /// Maximum number of bytes allowed
    pub max_size_bytes: u64,
    /// Maximum number of files allowed (None = unlimited)
    pub max_files: Option<u64>,
    /// Fraction of quota used (0.0–1.0) at which to issue a warning
    pub warn_threshold_pct: f32,
}

impl QuotaPolicy {
    /// Create a new policy with no file limit
    pub fn new(max_size_bytes: u64, warn_threshold_pct: f32) -> Self {
        Self {
            max_size_bytes,
            max_files: None,
            warn_threshold_pct,
        }
    }

    /// Create a policy with a file count limit
    pub fn with_file_limit(mut self, max_files: u64) -> Self {
        self.max_files = Some(max_files);
        self
    }
}

impl Default for QuotaPolicy {
    fn default() -> Self {
        // 10 GiB default quota, warn at 80%
        Self::new(10 * 1024 * 1024 * 1024, 0.80)
    }
}

/// Current usage against a quota policy
#[derive(Debug, Clone)]
pub struct QuotaUsage {
    /// Bytes currently used
    pub used_bytes: u64,
    /// Number of files currently stored
    pub file_count: u64,
    /// The policy this usage is measured against
    pub quota: QuotaPolicy,
}

impl QuotaUsage {
    /// Create new usage starting at zero
    pub fn new(quota: QuotaPolicy) -> Self {
        Self {
            used_bytes: 0,
            file_count: 0,
            quota,
        }
    }

    /// Percentage of byte quota consumed (0.0–100.0)
    pub fn usage_pct(&self) -> f32 {
        if self.quota.max_size_bytes == 0 {
            return 100.0;
        }
        (self.used_bytes as f32 / self.quota.max_size_bytes as f32) * 100.0
    }

    /// Returns `true` if the byte quota has been exceeded
    pub fn is_exceeded(&self) -> bool {
        self.used_bytes > self.quota.max_size_bytes
            || self
                .quota
                .max_files
                .is_some_and(|max| self.file_count > max)
    }

    /// Returns `true` if usage is above the warning threshold but not yet exceeded
    pub fn is_near_limit(&self) -> bool {
        if self.is_exceeded() {
            return false;
        }
        let fraction = self.used_bytes as f32 / self.quota.max_size_bytes as f32;
        fraction >= self.quota.warn_threshold_pct
    }

    /// Available bytes remaining (0 if quota is exceeded)
    pub fn available_bytes(&self) -> u64 {
        self.quota.max_size_bytes.saturating_sub(self.used_bytes)
    }
}

/// The organisational scope a quota entry belongs to
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum QuotaScope {
    /// Individual user
    User,
    /// Project / production
    Project,
    /// Business department
    Department,
    /// Organisation-wide limit
    Global,
}

impl std::fmt::Display for QuotaScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::User => write!(f, "User"),
            Self::Project => write!(f, "Project"),
            Self::Department => write!(f, "Department"),
            Self::Global => write!(f, "Global"),
        }
    }
}

/// An entry binding a scope + ID pair to a policy and its usage
#[derive(Debug, Clone)]
pub struct QuotaEntry {
    /// Organisational scope
    pub scope: QuotaScope,
    /// Identifier within the scope (e.g. username, project ID)
    pub scope_id: String,
    /// The quota policy
    pub policy: QuotaPolicy,
    /// Current usage
    pub usage: QuotaUsage,
}

impl QuotaEntry {
    /// Create a new entry with zero usage
    pub fn new(scope: QuotaScope, scope_id: impl Into<String>, policy: QuotaPolicy) -> Self {
        let usage = QuotaUsage::new(policy.clone());
        Self {
            scope,
            scope_id: scope_id.into(),
            policy,
            usage,
        }
    }
}

/// Result of a quota check for a requested byte allocation
#[derive(Debug, Clone, PartialEq)]
pub enum QuotaCheckResult {
    /// Request is within quota
    Allowed,
    /// Request is within quota but above the warning threshold (contains usage %)
    Warning(f32),
    /// Request would exceed quota (contains a human-readable explanation)
    Denied(String),
}

impl QuotaCheckResult {
    /// Returns `true` if the operation is allowed (Allowed or Warning)
    pub fn is_permitted(&self) -> bool {
        !matches!(self, Self::Denied(_))
    }
}

/// Composite key for looking up quota entries
type QuotaKey = (QuotaScope, String);

/// Central manager for quota policies and usage tracking
pub struct QuotaManager {
    entries: HashMap<QuotaKey, QuotaEntry>,
}

impl QuotaManager {
    /// Create a new, empty quota manager
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Set (or replace) a quota policy for the given scope + id
    pub fn set_quota(&mut self, scope: QuotaScope, id: &str, policy: QuotaPolicy) {
        let key = (scope.clone(), id.to_string());
        let entry = QuotaEntry::new(scope, id, policy);
        self.entries.insert(key, entry);
    }

    /// Update usage for the given scope + id by the given byte and file deltas.
    ///
    /// Negative deltas reduce usage (clamped at zero).
    pub fn update_usage(
        &mut self,
        scope: QuotaScope,
        id: &str,
        bytes_delta: i64,
        files_delta: i64,
    ) {
        let key = (scope, id.to_string());
        if let Some(entry) = self.entries.get_mut(&key) {
            if bytes_delta >= 0 {
                entry.usage.used_bytes = entry.usage.used_bytes.saturating_add(bytes_delta as u64);
            } else {
                entry.usage.used_bytes =
                    entry.usage.used_bytes.saturating_sub((-bytes_delta) as u64);
            }

            if files_delta >= 0 {
                entry.usage.file_count = entry.usage.file_count.saturating_add(files_delta as u64);
            } else {
                entry.usage.file_count =
                    entry.usage.file_count.saturating_sub((-files_delta) as u64);
            }
        }
    }

    /// Check whether `bytes_needed` can be allocated for the given scope + id.
    ///
    /// Returns `Denied` if the scope has no quota configured.
    pub fn check_quota(&self, scope: QuotaScope, id: &str, bytes_needed: u64) -> QuotaCheckResult {
        let key = (scope, id.to_string());
        let entry = match self.entries.get(&key) {
            Some(e) => e,
            None => {
                return QuotaCheckResult::Denied(format!(
                    "No quota configured for {}/{}",
                    key.0, key.1
                ))
            }
        };

        let projected_bytes = entry.usage.used_bytes.saturating_add(bytes_needed);

        if projected_bytes > entry.policy.max_size_bytes {
            return QuotaCheckResult::Denied(format!(
                "Quota exceeded: {} + {} > {} bytes",
                entry.usage.used_bytes, bytes_needed, entry.policy.max_size_bytes
            ));
        }

        let projected_pct = projected_bytes as f32 / entry.policy.max_size_bytes as f32;

        if projected_pct >= entry.policy.warn_threshold_pct {
            return QuotaCheckResult::Warning(projected_pct * 100.0);
        }

        QuotaCheckResult::Allowed
    }

    /// Return the quota entry for the given scope + id, if any
    pub fn get_entry(&self, scope: QuotaScope, id: &str) -> Option<&QuotaEntry> {
        self.entries.get(&(scope, id.to_string()))
    }

    /// Return all entries whose usage exceeds their quota
    pub fn overquota_entries(&self) -> Vec<&QuotaEntry> {
        self.entries
            .values()
            .filter(|e| e.usage.is_exceeded())
            .collect()
    }
}

impl Default for QuotaManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1024 * 1024 * 1024;

    fn simple_policy(max_gb: u64) -> QuotaPolicy {
        QuotaPolicy::new(max_gb * GB, 0.80)
    }

    // --- QuotaUsage ---

    #[test]
    fn test_usage_pct_zero() {
        let usage = QuotaUsage::new(simple_policy(10));
        assert!((usage.usage_pct() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_usage_pct_halfway() {
        let mut usage = QuotaUsage::new(simple_policy(10));
        usage.used_bytes = 5 * GB;
        assert!((usage.usage_pct() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_is_exceeded_false() {
        let usage = QuotaUsage::new(simple_policy(10));
        assert!(!usage.is_exceeded());
    }

    #[test]
    fn test_is_exceeded_true() {
        let mut usage = QuotaUsage::new(simple_policy(10));
        usage.used_bytes = 11 * GB;
        assert!(usage.is_exceeded());
    }

    #[test]
    fn test_is_near_limit_true() {
        let mut usage = QuotaUsage::new(simple_policy(10));
        usage.used_bytes = 9 * GB; // 90%, threshold 80%
        assert!(usage.is_near_limit());
    }

    #[test]
    fn test_is_near_limit_false_when_low() {
        let mut usage = QuotaUsage::new(simple_policy(10));
        usage.used_bytes = 1 * GB; // 10%
        assert!(!usage.is_near_limit());
    }

    #[test]
    fn test_available_bytes() {
        let mut usage = QuotaUsage::new(simple_policy(10));
        usage.used_bytes = 3 * GB;
        assert_eq!(usage.available_bytes(), 7 * GB);
    }

    #[test]
    fn test_available_bytes_exceeded_clamps_to_zero() {
        let mut usage = QuotaUsage::new(simple_policy(10));
        usage.used_bytes = 15 * GB;
        assert_eq!(usage.available_bytes(), 0);
    }

    // --- QuotaCheckResult ---

    #[test]
    fn test_check_result_permitted() {
        assert!(QuotaCheckResult::Allowed.is_permitted());
        assert!(QuotaCheckResult::Warning(85.0).is_permitted());
        assert!(!QuotaCheckResult::Denied("over".into()).is_permitted());
    }

    // --- QuotaManager ---

    #[test]
    fn test_manager_allowed() {
        let mut mgr = QuotaManager::new();
        mgr.set_quota(QuotaScope::User, "alice", simple_policy(10));
        let result = mgr.check_quota(QuotaScope::User, "alice", 1 * GB);
        assert_eq!(result, QuotaCheckResult::Allowed);
    }

    #[test]
    fn test_manager_warning() {
        let mut mgr = QuotaManager::new();
        mgr.set_quota(QuotaScope::User, "bob", simple_policy(10));
        // Pre-fill 8 GB, then request 1 GB → 90% → warning
        mgr.update_usage(QuotaScope::User, "bob", (8 * GB) as i64, 100);
        let result = mgr.check_quota(QuotaScope::User, "bob", 1 * GB);
        assert!(matches!(result, QuotaCheckResult::Warning(_)));
    }

    #[test]
    fn test_manager_denied_over_quota() {
        let mut mgr = QuotaManager::new();
        mgr.set_quota(QuotaScope::Project, "proj-x", simple_policy(10));
        mgr.update_usage(QuotaScope::Project, "proj-x", (9 * GB) as i64, 0);
        let result = mgr.check_quota(QuotaScope::Project, "proj-x", 2 * GB);
        assert!(matches!(result, QuotaCheckResult::Denied(_)));
    }

    #[test]
    fn test_manager_denied_no_quota() {
        let mgr = QuotaManager::new();
        let result = mgr.check_quota(QuotaScope::User, "unknown", 1);
        assert!(matches!(result, QuotaCheckResult::Denied(_)));
    }

    #[test]
    fn test_manager_update_usage_negative_clamps() {
        let mut mgr = QuotaManager::new();
        mgr.set_quota(QuotaScope::Department, "eng", simple_policy(10));
        mgr.update_usage(QuotaScope::Department, "eng", -999_999_999, -999_999);
        let entry = mgr
            .get_entry(QuotaScope::Department, "eng")
            .expect("entry should exist");
        assert_eq!(entry.usage.used_bytes, 0);
        assert_eq!(entry.usage.file_count, 0);
    }

    #[test]
    fn test_manager_overquota_entries() {
        let mut mgr = QuotaManager::new();
        mgr.set_quota(QuotaScope::User, "carol", simple_policy(10));
        mgr.set_quota(QuotaScope::User, "dave", simple_policy(10));
        mgr.update_usage(QuotaScope::User, "carol", (12 * GB) as i64, 0);
        let over = mgr.overquota_entries();
        assert_eq!(over.len(), 1);
        assert_eq!(over[0].scope_id, "carol");
    }

    #[test]
    fn test_quota_scope_display() {
        assert_eq!(QuotaScope::User.to_string(), "User");
        assert_eq!(QuotaScope::Project.to_string(), "Project");
        assert_eq!(QuotaScope::Department.to_string(), "Department");
        assert_eq!(QuotaScope::Global.to_string(), "Global");
    }
}
