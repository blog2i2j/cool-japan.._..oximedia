//! Quota enforcement — per-user/tenant job limits, storage quotas,
//! CPU hour budgets, and violation handling.
//!
//! [`QuotaEnforcer`] tracks resource consumption per principal (user or
//! tenant) across four axes:
//!
//! | Axis | Unit | Description |
//! |------|------|-------------|
//! | `concurrent_jobs` | count | Maximum simultaneously running jobs |
//! | `daily_jobs` | count | Maximum job submissions per calendar day |
//! | `storage_bytes` | bytes | Maximum disk usage attributed to outputs |
//! | `cpu_hours` | hours (f64) | Maximum compute time charged to the principal |
//!
//! When a job is admitted or completed, the caller calls the appropriate
//! `charge_*` / `release_*` methods.  Before admission, `check_can_admit`
//! validates that no quota would be exceeded.
//!
//! Violations are recorded in an append-only `VecDeque` and can be queried
//! for alerting or dashboards.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::error::{BatchError, Result};

// ---------------------------------------------------------------------------
// Quota limits
// ---------------------------------------------------------------------------

/// Hard limits for a single principal.
///
/// `None` means "unlimited" for that axis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuotaLimits {
    /// Maximum number of concurrently running jobs.
    pub max_concurrent_jobs: Option<u32>,
    /// Maximum number of jobs submitted per calendar day (UTC).
    pub max_daily_jobs: Option<u32>,
    /// Maximum total bytes of output storage.
    pub max_storage_bytes: Option<u64>,
    /// Maximum cumulative CPU hours.
    pub max_cpu_hours: Option<f64>,
}

impl QuotaLimits {
    /// Create a fully permissive (unlimited) quota.
    #[must_use]
    pub fn unlimited() -> Self {
        Self {
            max_concurrent_jobs: None,
            max_daily_jobs: None,
            max_storage_bytes: None,
            max_cpu_hours: None,
        }
    }

    /// Create a quota with sensible defaults for a shared environment.
    ///
    /// * 4 concurrent jobs
    /// * 50 jobs per day
    /// * 10 GiB storage
    /// * 24 CPU hours
    #[must_use]
    pub fn default_shared() -> Self {
        Self {
            max_concurrent_jobs: Some(4),
            max_daily_jobs: Some(50),
            max_storage_bytes: Some(10 * 1024 * 1024 * 1024),
            max_cpu_hours: Some(24.0),
        }
    }
}

// ---------------------------------------------------------------------------
// Usage snapshot
// ---------------------------------------------------------------------------

/// Current resource consumption for a single principal.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuotaUsage {
    /// Currently running jobs.
    pub concurrent_jobs: u32,
    /// Jobs submitted today (UTC).
    pub daily_jobs_today: u32,
    /// UTC date (YYYY-MM-DD) for which `daily_jobs_today` was last reset.
    pub daily_window_date: String,
    /// Bytes of output storage currently attributed.
    pub storage_bytes: u64,
    /// Cumulative CPU hours attributed.
    pub cpu_hours: f64,
}

impl QuotaUsage {
    fn today_date() -> String {
        // Compute today's UTC date as "YYYY-MM-DD" without chrono dependency:
        // seconds since epoch → days → year/month/day via proleptic Gregorian.
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let days = secs / 86_400;
        // Algorithm: civil date from Julian Day Number.
        let z = days + 719_468;
        let era = z / 146_097;
        let doe = z - era * 146_097;
        let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        let y = if m <= 2 { y + 1 } else { y };
        format!("{y:04}-{m:02}-{d:02}")
    }

    /// Refresh the daily window if the calendar date has changed.
    fn refresh_daily_window(&mut self) {
        let today = Self::today_date();
        if self.daily_window_date != today {
            self.daily_jobs_today = 0;
            self.daily_window_date = today;
        }
    }
}

// ---------------------------------------------------------------------------
// Violation record
// ---------------------------------------------------------------------------

/// A recorded quota violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotaViolation {
    /// The principal who attempted to exceed their quota.
    pub principal: String,
    /// Which quota axis was violated.
    pub axis: QuotaAxis,
    /// The limit that was in effect.
    pub limit: f64,
    /// The current usage at the time of the violation.
    pub usage: f64,
    /// The additional amount that was requested.
    pub requested: f64,
    /// Unix timestamp (seconds).
    pub occurred_at: u64,
}

impl QuotaViolation {
    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}

/// Which resource axis triggered a quota violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuotaAxis {
    /// Concurrent job limit.
    ConcurrentJobs,
    /// Daily submission limit.
    DailyJobs,
    /// Storage byte limit.
    StorageBytes,
    /// CPU hour limit.
    CpuHours,
}

impl std::fmt::Display for QuotaAxis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConcurrentJobs => write!(f, "concurrent_jobs"),
            Self::DailyJobs => write!(f, "daily_jobs"),
            Self::StorageBytes => write!(f, "storage_bytes"),
            Self::CpuHours => write!(f, "cpu_hours"),
        }
    }
}

// ---------------------------------------------------------------------------
// Principal record
// ---------------------------------------------------------------------------

/// Per-principal data: limits + live usage.
#[derive(Debug)]
struct PrincipalRecord {
    limits: QuotaLimits,
    usage: QuotaUsage,
}

impl PrincipalRecord {
    fn new(limits: QuotaLimits) -> Self {
        Self {
            limits,
            usage: QuotaUsage::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// QuotaEnforcer
// ---------------------------------------------------------------------------

/// Thread-safe quota manager for multiple principals.
///
/// # Example
///
/// ```
/// use oximedia_batch::quota_enforcer::{QuotaEnforcer, QuotaLimits};
///
/// let enforcer = QuotaEnforcer::new();
/// enforcer.set_limits("alice", QuotaLimits {
///     max_concurrent_jobs: Some(2),
///     max_daily_jobs: Some(10),
///     max_storage_bytes: None,
///     max_cpu_hours: None,
/// });
///
/// // Admit a job.
/// assert!(enforcer.check_can_admit("alice").is_ok());
/// enforcer.charge_concurrent_job("alice");
///
/// // Release when done.
/// enforcer.release_concurrent_job("alice");
/// ```
pub struct QuotaEnforcer {
    /// Principal → record, protected by a single write-lock.
    principals: RwLock<HashMap<String, PrincipalRecord>>,
    /// Append-only violation log.
    violations: RwLock<std::collections::VecDeque<QuotaViolation>>,
    /// Maximum violations retained in memory.
    max_violation_history: usize,
    /// Default limits applied to principals with no explicit entry.
    default_limits: QuotaLimits,
}

impl QuotaEnforcer {
    /// Create a new enforcer with unlimited defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::with_defaults(QuotaLimits::unlimited())
    }

    /// Create a new enforcer with the supplied default limits.
    #[must_use]
    pub fn with_defaults(default_limits: QuotaLimits) -> Self {
        Self {
            principals: RwLock::new(HashMap::new()),
            violations: RwLock::new(std::collections::VecDeque::new()),
            max_violation_history: 10_000,
            default_limits,
        }
    }

    // -----------------------------------------------------------------------
    // Limit management
    // -----------------------------------------------------------------------

    /// Set (or replace) quota limits for `principal`.
    pub fn set_limits(&self, principal: impl Into<String>, limits: QuotaLimits) {
        let key = principal.into();
        let mut map = self.principals.write();
        map.entry(key)
            .and_modify(|r| r.limits = limits.clone())
            .or_insert_with(|| PrincipalRecord::new(limits));
    }

    /// Remove per-principal limits; the principal will fall back to defaults.
    pub fn remove_limits(&self, principal: &str) {
        self.principals.write().remove(principal);
    }

    /// Return the effective limits for `principal`.
    #[must_use]
    pub fn limits(&self, principal: &str) -> QuotaLimits {
        self.principals
            .read()
            .get(principal)
            .map(|r| r.limits.clone())
            .unwrap_or_else(|| self.default_limits.clone())
    }

    // -----------------------------------------------------------------------
    // Admission control
    // -----------------------------------------------------------------------

    /// Check whether `principal` may submit a new job.
    ///
    /// This validates `concurrent_jobs` and `daily_jobs` limits.  It does NOT
    /// automatically charge usage — call [`charge_concurrent_job`] and
    /// [`charge_daily_job`] after a successful check.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::ResourceError`] describing which quota would be
    /// exceeded.
    pub fn check_can_admit(&self, principal: &str) -> Result<()> {
        let mut map = self.principals.write();
        let record = map
            .entry(principal.to_owned())
            .or_insert_with(|| PrincipalRecord::new(self.default_limits.clone()));

        record.usage.refresh_daily_window();

        // Check concurrent jobs.
        if let Some(max) = record.limits.max_concurrent_jobs {
            if record.usage.concurrent_jobs >= max {
                let v = QuotaViolation {
                    principal: principal.to_owned(),
                    axis: QuotaAxis::ConcurrentJobs,
                    limit: f64::from(max),
                    usage: f64::from(record.usage.concurrent_jobs),
                    requested: 1.0,
                    occurred_at: QuotaViolation::now(),
                };
                drop(map);
                self.record_violation(v);
                return Err(BatchError::ResourceError(format!(
                    "principal '{principal}' has reached the concurrent job limit ({max})"
                )));
            }
        }

        // Check daily jobs.
        if let Some(max) = record.limits.max_daily_jobs {
            if record.usage.daily_jobs_today >= max {
                let v = QuotaViolation {
                    principal: principal.to_owned(),
                    axis: QuotaAxis::DailyJobs,
                    limit: f64::from(max),
                    usage: f64::from(record.usage.daily_jobs_today),
                    requested: 1.0,
                    occurred_at: QuotaViolation::now(),
                };
                drop(map);
                self.record_violation(v);
                return Err(BatchError::ResourceError(format!(
                    "principal '{principal}' has reached the daily job limit ({max})"
                )));
            }
        }

        Ok(())
    }

    /// Check whether `principal` may allocate `bytes` of additional storage.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::ResourceError`] if the storage quota would be exceeded.
    pub fn check_storage(&self, principal: &str, bytes: u64) -> Result<()> {
        let mut map = self.principals.write();
        let record = map
            .entry(principal.to_owned())
            .or_insert_with(|| PrincipalRecord::new(self.default_limits.clone()));

        if let Some(max) = record.limits.max_storage_bytes {
            let after = record.usage.storage_bytes.saturating_add(bytes);
            if after > max {
                let v = QuotaViolation {
                    principal: principal.to_owned(),
                    axis: QuotaAxis::StorageBytes,
                    limit: max as f64,
                    usage: record.usage.storage_bytes as f64,
                    requested: bytes as f64,
                    occurred_at: QuotaViolation::now(),
                };
                drop(map);
                self.record_violation(v);
                return Err(BatchError::ResourceError(format!(
                    "principal '{principal}' storage quota would be exceeded \
                     (limit={max} bytes, usage={after} bytes)",
                )));
            }
        }
        Ok(())
    }

    /// Check whether `principal` may spend `hours` of additional CPU time.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::ResourceError`] if the CPU hour budget would be exceeded.
    pub fn check_cpu_hours(&self, principal: &str, hours: f64) -> Result<()> {
        let mut map = self.principals.write();
        let record = map
            .entry(principal.to_owned())
            .or_insert_with(|| PrincipalRecord::new(self.default_limits.clone()));

        if let Some(max) = record.limits.max_cpu_hours {
            let after = record.usage.cpu_hours + hours;
            if after > max {
                let v = QuotaViolation {
                    principal: principal.to_owned(),
                    axis: QuotaAxis::CpuHours,
                    limit: max,
                    usage: record.usage.cpu_hours,
                    requested: hours,
                    occurred_at: QuotaViolation::now(),
                };
                drop(map);
                self.record_violation(v);
                return Err(BatchError::ResourceError(format!(
                    "principal '{principal}' CPU hour budget would be exceeded \
                     (limit={max:.2}h, after={after:.2}h)",
                )));
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Charge / release
    // -----------------------------------------------------------------------

    /// Increment the running-job counter for `principal`.
    ///
    /// Call this immediately after a job is admitted and begins execution.
    pub fn charge_concurrent_job(&self, principal: &str) {
        let mut map = self.principals.write();
        let record = map
            .entry(principal.to_owned())
            .or_insert_with(|| PrincipalRecord::new(self.default_limits.clone()));
        record.usage.concurrent_jobs = record.usage.concurrent_jobs.saturating_add(1);
    }

    /// Decrement the running-job counter for `principal`.
    ///
    /// Call this when a job completes, fails, or is cancelled.
    pub fn release_concurrent_job(&self, principal: &str) {
        let mut map = self.principals.write();
        if let Some(record) = map.get_mut(principal) {
            record.usage.concurrent_jobs = record.usage.concurrent_jobs.saturating_sub(1);
        }
    }

    /// Increment the daily job counter for `principal`.
    ///
    /// Call this once per job submission (after a successful `check_can_admit`).
    pub fn charge_daily_job(&self, principal: &str) {
        let mut map = self.principals.write();
        let record = map
            .entry(principal.to_owned())
            .or_insert_with(|| PrincipalRecord::new(self.default_limits.clone()));
        record.usage.refresh_daily_window();
        record.usage.daily_jobs_today = record.usage.daily_jobs_today.saturating_add(1);
    }

    /// Add `bytes` of storage usage for `principal`.
    pub fn charge_storage(&self, principal: &str, bytes: u64) {
        let mut map = self.principals.write();
        let record = map
            .entry(principal.to_owned())
            .or_insert_with(|| PrincipalRecord::new(self.default_limits.clone()));
        record.usage.storage_bytes = record.usage.storage_bytes.saturating_add(bytes);
    }

    /// Subtract `bytes` of storage usage for `principal` (e.g., on output deletion).
    pub fn release_storage(&self, principal: &str, bytes: u64) {
        let mut map = self.principals.write();
        if let Some(record) = map.get_mut(principal) {
            record.usage.storage_bytes = record.usage.storage_bytes.saturating_sub(bytes);
        }
    }

    /// Add `hours` of CPU time to `principal`'s cumulative usage.
    pub fn charge_cpu_hours(&self, principal: &str, hours: f64) {
        let mut map = self.principals.write();
        let record = map
            .entry(principal.to_owned())
            .or_insert_with(|| PrincipalRecord::new(self.default_limits.clone()));
        record.usage.cpu_hours += hours;
    }

    // -----------------------------------------------------------------------
    // Usage queries
    // -----------------------------------------------------------------------

    /// Return a snapshot of the current usage for `principal`.
    #[must_use]
    pub fn usage(&self, principal: &str) -> QuotaUsage {
        let mut map = self.principals.write();
        let record = map
            .entry(principal.to_owned())
            .or_insert_with(|| PrincipalRecord::new(self.default_limits.clone()));
        record.usage.refresh_daily_window();
        record.usage.clone()
    }

    /// Return the usage percentage (0.0 ..= 100.0) for each axis.
    ///
    /// Axes with unlimited quota are omitted from the map.
    #[must_use]
    pub fn usage_percent(&self, principal: &str) -> HashMap<String, f64> {
        let limits = self.limits(principal);
        let usage = self.usage(principal);
        let mut out = HashMap::new();

        if let Some(max) = limits.max_concurrent_jobs {
            if max > 0 {
                let pct = (f64::from(usage.concurrent_jobs) / f64::from(max)) * 100.0;
                out.insert("concurrent_jobs".into(), pct.min(100.0));
            }
        }
        if let Some(max) = limits.max_daily_jobs {
            if max > 0 {
                let pct = (f64::from(usage.daily_jobs_today) / f64::from(max)) * 100.0;
                out.insert("daily_jobs".into(), pct.min(100.0));
            }
        }
        if let Some(max) = limits.max_storage_bytes {
            if max > 0 {
                let pct = (usage.storage_bytes as f64 / max as f64) * 100.0;
                out.insert("storage_bytes".into(), pct.min(100.0));
            }
        }
        if let Some(max) = limits.max_cpu_hours {
            if max > 0.0 {
                let pct = (usage.cpu_hours / max) * 100.0;
                out.insert("cpu_hours".into(), pct.min(100.0));
            }
        }
        out
    }

    // -----------------------------------------------------------------------
    // Violation log
    // -----------------------------------------------------------------------

    fn record_violation(&self, v: QuotaViolation) {
        let mut log = self.violations.write();
        if log.len() >= self.max_violation_history {
            log.pop_front();
        }
        log.push_back(v);
    }

    /// Return up to `limit` most-recent violations (newest last).
    #[must_use]
    pub fn recent_violations(&self, limit: usize) -> Vec<QuotaViolation> {
        let log = self.violations.read();
        let skip = log.len().saturating_sub(limit);
        log.iter().skip(skip).cloned().collect()
    }

    /// Return all violations for a specific principal.
    #[must_use]
    pub fn violations_for(&self, principal: &str) -> Vec<QuotaViolation> {
        let log = self.violations.read();
        log.iter()
            .filter(|v| v.principal == principal)
            .cloned()
            .collect()
    }

    /// Total violation count since the enforcer was created.
    #[must_use]
    pub fn violation_count(&self) -> usize {
        self.violations.read().len()
    }

    // -----------------------------------------------------------------------
    // Principal listing
    // -----------------------------------------------------------------------

    /// Return the names of all principals currently registered.
    #[must_use]
    pub fn principals(&self) -> Vec<String> {
        self.principals.read().keys().cloned().collect()
    }

    /// Reset all usage counters for `principal` (does not change limits).
    ///
    /// Useful in tests and for end-of-period roll-overs managed externally.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::JobNotFound`] if `principal` is unknown.
    pub fn reset_usage(&self, principal: &str) -> Result<()> {
        let mut map = self.principals.write();
        match map.get_mut(principal) {
            Some(record) => {
                record.usage = QuotaUsage::default();
                Ok(())
            }
            None => Err(BatchError::JobNotFound(format!(
                "principal '{principal}' not found"
            ))),
        }
    }
}

impl Default for QuotaEnforcer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn limited_enforcer() -> QuotaEnforcer {
        let e = QuotaEnforcer::new();
        e.set_limits(
            "alice",
            QuotaLimits {
                max_concurrent_jobs: Some(2),
                max_daily_jobs: Some(5),
                max_storage_bytes: Some(1024 * 1024),   // 1 MiB
                max_cpu_hours: Some(10.0),
            },
        );
        e
    }

    #[test]
    fn test_admit_below_limit_succeeds() {
        let e = limited_enforcer();
        assert!(e.check_can_admit("alice").is_ok());
    }

    #[test]
    fn test_concurrent_job_limit_enforced() {
        let e = limited_enforcer();
        e.charge_concurrent_job("alice");
        e.charge_concurrent_job("alice");
        let result = e.check_can_admit("alice");
        assert!(result.is_err());
        let err = result.err().expect("should be err");
        assert!(err.to_string().contains("concurrent"));
    }

    #[test]
    fn test_release_concurrent_job_restores_capacity() {
        let e = limited_enforcer();
        e.charge_concurrent_job("alice");
        e.charge_concurrent_job("alice");
        e.release_concurrent_job("alice");
        assert!(e.check_can_admit("alice").is_ok());
    }

    #[test]
    fn test_daily_job_limit_enforced() {
        let e = limited_enforcer();
        for _ in 0..5 {
            assert!(e.check_can_admit("alice").is_ok());
            e.charge_daily_job("alice");
        }
        let result = e.check_can_admit("alice");
        assert!(result.is_err());
    }

    #[test]
    fn test_storage_quota_enforced() {
        let e = limited_enforcer();
        e.charge_storage("alice", 900_000);
        let result = e.check_storage("alice", 200_000); // 900k + 200k > 1 MiB
        assert!(result.is_err());
        let err = result.err().expect("should be err");
        assert!(err.to_string().contains("storage"));
    }

    #[test]
    fn test_storage_release_frees_quota() {
        let e = limited_enforcer();
        e.charge_storage("alice", 900_000);
        e.release_storage("alice", 200_000);
        assert!(e.check_storage("alice", 200_000).is_ok());
    }

    #[test]
    fn test_cpu_hours_quota_enforced() {
        let e = limited_enforcer();
        e.charge_cpu_hours("alice", 9.5);
        let result = e.check_cpu_hours("alice", 1.0); // 9.5 + 1.0 > 10.0
        assert!(result.is_err());
        let err = result.err().expect("should be err");
        assert!(err.to_string().contains("CPU"));
    }

    #[test]
    fn test_violation_recorded_on_concurrent_breach() {
        let e = limited_enforcer();
        e.charge_concurrent_job("alice");
        e.charge_concurrent_job("alice");
        let _ = e.check_can_admit("alice");
        assert_eq!(e.violation_count(), 1);
        let v = &e.recent_violations(1)[0];
        assert_eq!(v.principal, "alice");
        assert_eq!(v.axis, QuotaAxis::ConcurrentJobs);
    }

    #[test]
    fn test_usage_percent_calculation() {
        let e = limited_enforcer();
        e.charge_cpu_hours("alice", 5.0);
        let pct = e.usage_percent("alice");
        let cpu_pct = pct.get("cpu_hours").copied().unwrap_or(0.0);
        assert!((cpu_pct - 50.0).abs() < 1e-4, "expected 50%, got {cpu_pct}");
    }

    #[test]
    fn test_unlimited_principal_always_admitted() {
        let e = QuotaEnforcer::new();
        e.set_limits("bob", QuotaLimits::unlimited());
        for _ in 0..100 {
            assert!(e.check_can_admit("bob").is_ok());
        }
    }

    #[test]
    fn test_reset_usage_clears_counters() {
        let e = limited_enforcer();
        e.charge_concurrent_job("alice");
        e.charge_daily_job("alice");
        assert!(e.reset_usage("alice").is_ok());
        let usage = e.usage("alice");
        assert_eq!(usage.concurrent_jobs, 0);
        assert_eq!(usage.daily_jobs_today, 0);
    }

    #[test]
    fn test_reset_unknown_principal_returns_error() {
        let e = QuotaEnforcer::new();
        assert!(e.reset_usage("ghost").is_err());
    }

    #[test]
    fn test_violations_for_principal_filtered() {
        let e = limited_enforcer();
        e.set_limits(
            "carol",
            QuotaLimits {
                max_concurrent_jobs: Some(1),
                ..QuotaLimits::unlimited()
            },
        );
        e.charge_concurrent_job("carol");
        let _ = e.check_can_admit("carol"); // triggers violation for carol
        let _ = e.check_can_admit("carol"); // triggers another

        e.charge_concurrent_job("alice");
        e.charge_concurrent_job("alice");
        let _ = e.check_can_admit("alice"); // violation for alice

        let carol_v = e.violations_for("carol");
        assert_eq!(carol_v.len(), 2);
        let alice_v = e.violations_for("alice");
        assert_eq!(alice_v.len(), 1);
    }

    #[test]
    fn test_default_shared_limits() {
        let lim = QuotaLimits::default_shared();
        assert_eq!(lim.max_concurrent_jobs, Some(4));
        assert_eq!(lim.max_daily_jobs, Some(50));
        assert!(lim.max_storage_bytes.is_some());
        assert!(lim.max_cpu_hours.is_some());
    }
}
