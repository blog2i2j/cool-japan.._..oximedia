#![allow(dead_code)]
//! Multi-tenant support with per-tenant job quotas, resource isolation, and
//! cost estimation.
//!
//! ## Overview
//!
//! A [`TenantRegistry`] holds named [`Tenant`] records, each with:
//! - A configurable job quota (`max_concurrent_jobs`).
//! - A resource budget (`ResourceBudget`) tracking CPU, GPU, and memory
//!   in use across all active jobs.
//! - A [`CostEstimator`] that computes job cost from resource usage and
//!   elapsed time.
//!
//! The registry enforces isolation: one tenant's quota exhaustion does not
//! affect others.

use std::collections::HashMap;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Resource budget
// ---------------------------------------------------------------------------

/// Resource budget tracking for one tenant.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ResourceBudget {
    /// Maximum CPU-seconds the tenant may consume per billing period.
    pub max_cpu_seconds: f64,
    /// Maximum GPU-seconds allowed per billing period.
    pub max_gpu_seconds: f64,
    /// Maximum memory-hours (GB·h) per billing period.
    pub max_memory_gb_hours: f64,
}

impl ResourceBudget {
    /// Create a budget with explicit limits.
    #[must_use]
    pub fn new(max_cpu_secs: f64, max_gpu_secs: f64, max_mem_gb_h: f64) -> Self {
        Self {
            max_cpu_seconds: max_cpu_secs,
            max_gpu_seconds: max_gpu_secs,
            max_memory_gb_hours: max_mem_gb_h,
        }
    }
}

// ---------------------------------------------------------------------------
// Usage counters
// ---------------------------------------------------------------------------

/// Accumulated resource usage for one tenant in the current billing period.
#[derive(Debug, Clone, Default)]
pub struct TenantUsage {
    /// CPU-seconds consumed.
    pub cpu_seconds: f64,
    /// GPU-seconds consumed.
    pub gpu_seconds: f64,
    /// Memory-hours (GB·h) consumed.
    pub memory_gb_hours: f64,
    /// Number of jobs submitted.
    pub jobs_submitted: u64,
    /// Number of jobs successfully completed.
    pub jobs_completed: u64,
    /// Number of jobs that failed.
    pub jobs_failed: u64,
}

impl TenantUsage {
    /// Record the resource consumption of one completed job.
    pub fn record_job(
        &mut self,
        cpu_cores: f64,
        gpu_devices: f64,
        memory_gb: f64,
        duration: Duration,
        succeeded: bool,
    ) {
        let secs = duration.as_secs_f64();
        self.cpu_seconds += cpu_cores * secs;
        self.gpu_seconds += gpu_devices * secs;
        self.memory_gb_hours += memory_gb * (secs / 3600.0);
        self.jobs_submitted += 1;
        if succeeded {
            self.jobs_completed += 1;
        } else {
            self.jobs_failed += 1;
        }
    }

    /// Return `true` when all usage figures are within the `budget`.
    #[must_use]
    pub fn within_budget(&self, budget: &ResourceBudget) -> bool {
        self.cpu_seconds <= budget.max_cpu_seconds
            && self.gpu_seconds <= budget.max_gpu_seconds
            && self.memory_gb_hours <= budget.max_memory_gb_hours
    }

    /// Reset all counters (e.g., at the start of a new billing period).
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

// ---------------------------------------------------------------------------
// Cost estimation
// ---------------------------------------------------------------------------

/// Unit prices used by the cost estimator.
#[derive(Debug, Clone)]
pub struct CostRates {
    /// Price per CPU-second (e.g., in micro-USD).
    pub per_cpu_second: f64,
    /// Price per GPU-second.
    pub per_gpu_second: f64,
    /// Price per GB·h of memory.
    pub per_memory_gb_hour: f64,
    /// Flat per-job submission fee.
    pub per_job_fee: f64,
}

impl Default for CostRates {
    fn default() -> Self {
        // Reasonable defaults expressed in fractional USD.
        Self {
            per_cpu_second: 0.000_010, // $0.01 / 1 000 CPU-sec
            per_gpu_second: 0.000_100, // $0.10 / 1 000 GPU-sec
            per_memory_gb_hour: 0.010, // $0.01 / GB·h
            per_job_fee: 0.001,        // $0.001 flat fee
        }
    }
}

/// Estimates job cost from resource usage and duration.
#[derive(Debug, Clone)]
pub struct CostEstimator {
    rates: CostRates,
}

impl CostEstimator {
    /// Create an estimator with custom rates.
    #[must_use]
    pub fn new(rates: CostRates) -> Self {
        Self { rates }
    }

    /// Create an estimator with default rates.
    #[must_use]
    pub fn default_rates() -> Self {
        Self::new(CostRates::default())
    }

    /// Estimate the cost of a single job.
    ///
    /// # Arguments
    ///
    /// - `cpu_cores` – number of CPU cores used.
    /// - `gpu_devices` – number of GPU devices used (fractional allowed).
    /// - `memory_gb` – amount of RAM allocated in GB.
    /// - `duration` – wall-clock runtime.
    ///
    /// Returns the estimated cost in the same unit as `CostRates` (e.g., USD).
    #[must_use]
    pub fn estimate_job(
        &self,
        cpu_cores: f64,
        gpu_devices: f64,
        memory_gb: f64,
        duration: Duration,
    ) -> f64 {
        let secs = duration.as_secs_f64();
        let r = &self.rates;
        r.per_job_fee
            + cpu_cores * secs * r.per_cpu_second
            + gpu_devices * secs * r.per_gpu_second
            + memory_gb * (secs / 3600.0) * r.per_memory_gb_hour
    }

    /// Estimate the cost of accumulated usage statistics.
    #[must_use]
    pub fn estimate_usage(&self, usage: &TenantUsage) -> f64 {
        let r = &self.rates;
        r.per_job_fee * usage.jobs_submitted as f64
            + usage.cpu_seconds * r.per_cpu_second
            + usage.gpu_seconds * r.per_gpu_second
            + usage.memory_gb_hours * r.per_memory_gb_hour
    }
}

// ---------------------------------------------------------------------------
// Tenant
// ---------------------------------------------------------------------------

/// A single tenant with an isolated quota and usage counters.
#[derive(Debug, Clone)]
pub struct Tenant {
    /// Unique tenant identifier.
    pub id: String,
    /// Human-readable display name.
    pub display_name: String,
    /// Maximum concurrent jobs this tenant may have in-flight.
    pub max_concurrent_jobs: usize,
    /// Resource budget for the current billing period.
    pub budget: ResourceBudget,
    /// Current usage counters.
    pub usage: TenantUsage,
    /// Number of jobs currently in-flight for this tenant.
    pub active_jobs: usize,
    /// Whether the tenant is currently enabled.
    pub enabled: bool,
}

impl Tenant {
    /// Create a new tenant.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        display_name: impl Into<String>,
        max_concurrent_jobs: usize,
    ) -> Self {
        Self {
            id: id.into(),
            display_name: display_name.into(),
            max_concurrent_jobs,
            budget: ResourceBudget::default(),
            usage: TenantUsage::default(),
            active_jobs: 0,
            enabled: true,
        }
    }

    /// Set the resource budget for this tenant.
    #[must_use]
    pub fn with_budget(mut self, budget: ResourceBudget) -> Self {
        self.budget = budget;
        self
    }

    /// Check whether the tenant can accept another job submission.
    ///
    /// Returns `false` if the tenant is disabled, at capacity, or over budget.
    #[must_use]
    pub fn can_accept_job(&self) -> bool {
        self.enabled
            && self.active_jobs < self.max_concurrent_jobs
            && self.usage.within_budget(&self.budget)
    }

    /// Mark a job as started (increments `active_jobs`).
    ///
    /// Returns `false` if the quota would be exceeded.
    pub fn start_job(&mut self) -> bool {
        if !self.can_accept_job() {
            return false;
        }
        self.active_jobs += 1;
        true
    }

    /// Mark a job as finished and record resource consumption.
    pub fn finish_job(
        &mut self,
        cpu_cores: f64,
        gpu_devices: f64,
        memory_gb: f64,
        duration: Duration,
        succeeded: bool,
    ) {
        self.active_jobs = self.active_jobs.saturating_sub(1);
        self.usage
            .record_job(cpu_cores, gpu_devices, memory_gb, duration, succeeded);
    }
}

// ---------------------------------------------------------------------------
// Tenant registry
// ---------------------------------------------------------------------------

/// Error type for tenant registry operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TenantError {
    /// Tenant not found.
    NotFound(String),
    /// Tenant already registered.
    AlreadyExists(String),
    /// Tenant quota exhausted.
    QuotaExceeded(String),
    /// Tenant is disabled.
    Disabled(String),
}

impl std::fmt::Display for TenantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "tenant not found: {id}"),
            Self::AlreadyExists(id) => write!(f, "tenant already exists: {id}"),
            Self::QuotaExceeded(id) => write!(f, "tenant quota exceeded: {id}"),
            Self::Disabled(id) => write!(f, "tenant is disabled: {id}"),
        }
    }
}

impl std::error::Error for TenantError {}

/// Result type for tenant registry operations.
pub type TenantResult<T> = std::result::Result<T, TenantError>;

/// Registry that manages multiple tenants with isolated quotas.
#[derive(Debug, Default)]
pub struct TenantRegistry {
    tenants: HashMap<String, Tenant>,
    /// Cost estimator used for per-tenant cost calculations.
    estimator: Option<CostEstimator>,
}

impl TenantRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a registry with a custom cost estimator.
    #[must_use]
    pub fn with_estimator(estimator: CostEstimator) -> Self {
        Self {
            tenants: HashMap::new(),
            estimator: Some(estimator),
        }
    }

    /// Register a new tenant.
    ///
    /// # Errors
    ///
    /// Returns [`TenantError::AlreadyExists`] if a tenant with the same ID is
    /// already registered.
    pub fn register(&mut self, tenant: Tenant) -> TenantResult<()> {
        if self.tenants.contains_key(&tenant.id) {
            return Err(TenantError::AlreadyExists(tenant.id.clone()));
        }
        self.tenants.insert(tenant.id.clone(), tenant);
        Ok(())
    }

    /// Remove a tenant by ID.
    ///
    /// Returns the removed tenant, or `None` when not found.
    pub fn remove(&mut self, id: &str) -> Option<Tenant> {
        self.tenants.remove(id)
    }

    /// Get a reference to a tenant.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&Tenant> {
        self.tenants.get(id)
    }

    /// Get a mutable reference to a tenant.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut Tenant> {
        self.tenants.get_mut(id)
    }

    /// Attempt to start a job for `tenant_id`.
    ///
    /// # Errors
    ///
    /// - [`TenantError::NotFound`] – unknown tenant.
    /// - [`TenantError::Disabled`] – tenant is disabled.
    /// - [`TenantError::QuotaExceeded`] – concurrent job limit reached or
    ///   resource budget exceeded.
    pub fn start_job(&mut self, tenant_id: &str) -> TenantResult<()> {
        let tenant = self
            .tenants
            .get_mut(tenant_id)
            .ok_or_else(|| TenantError::NotFound(tenant_id.to_string()))?;

        if !tenant.enabled {
            return Err(TenantError::Disabled(tenant_id.to_string()));
        }

        if !tenant.start_job() {
            return Err(TenantError::QuotaExceeded(tenant_id.to_string()));
        }

        Ok(())
    }

    /// Finish a job for `tenant_id` and record resource consumption.
    ///
    /// # Errors
    ///
    /// Returns [`TenantError::NotFound`] when the tenant is not registered.
    pub fn finish_job(
        &mut self,
        tenant_id: &str,
        cpu_cores: f64,
        gpu_devices: f64,
        memory_gb: f64,
        duration: Duration,
        succeeded: bool,
    ) -> TenantResult<()> {
        let tenant = self
            .tenants
            .get_mut(tenant_id)
            .ok_or_else(|| TenantError::NotFound(tenant_id.to_string()))?;
        tenant.finish_job(cpu_cores, gpu_devices, memory_gb, duration, succeeded);
        Ok(())
    }

    /// Estimate the cost incurred by `tenant_id` for their accumulated usage.
    ///
    /// Returns `None` when no estimator is configured or when the tenant is
    /// not found.
    #[must_use]
    pub fn estimated_cost(&self, tenant_id: &str) -> Option<f64> {
        let estimator = self.estimator.as_ref()?;
        let tenant = self.tenants.get(tenant_id)?;
        Some(estimator.estimate_usage(&tenant.usage))
    }

    /// Number of registered tenants.
    #[must_use]
    pub fn tenant_count(&self) -> usize {
        self.tenants.len()
    }

    /// Reset usage counters for all tenants (call at the start of each billing
    /// period).
    pub fn reset_all_usage(&mut self) {
        for tenant in self.tenants.values_mut() {
            tenant.usage.reset();
        }
    }

    /// Return IDs of all tenants that have exceeded their budget.
    #[must_use]
    pub fn over_budget_tenants(&self) -> Vec<&str> {
        self.tenants
            .values()
            .filter(|t| !t.usage.within_budget(&t.budget))
            .map(|t| t.id.as_str())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tenant(id: &str, max_jobs: usize) -> Tenant {
        Tenant::new(id, format!("Tenant {id}"), max_jobs)
    }

    // ── ResourceBudget / TenantUsage ─────────────────────────────────────────

    #[test]
    fn test_usage_within_budget() {
        let budget = ResourceBudget::new(1_000.0, 100.0, 50.0);
        let mut usage = TenantUsage::default();
        assert!(usage.within_budget(&budget));

        usage.record_job(2.0, 0.0, 4.0, Duration::from_secs(100), true);
        // cpu_seconds = 200, gpu = 0, mem_gb_h ≈ 0.11
        assert!(usage.within_budget(&budget));
    }

    #[test]
    fn test_usage_exceeds_cpu_budget() {
        let budget = ResourceBudget::new(100.0, 100.0, 100.0);
        let mut usage = TenantUsage::default();
        usage.record_job(4.0, 0.0, 4.0, Duration::from_secs(200), true);
        // cpu_seconds = 800 > 100
        assert!(!usage.within_budget(&budget));
    }

    #[test]
    fn test_usage_reset() {
        let mut usage = TenantUsage::default();
        usage.record_job(2.0, 1.0, 8.0, Duration::from_secs(3600), true);
        usage.reset();
        assert_eq!(usage.cpu_seconds, 0.0);
        assert_eq!(usage.jobs_submitted, 0);
    }

    // ── CostEstimator ─────────────────────────────────────────────────────────

    #[test]
    fn test_estimate_job_zero_duration_is_flat_fee() {
        let estimator = CostEstimator::default_rates();
        let cost = estimator.estimate_job(4.0, 0.0, 8.0, Duration::ZERO);
        // Only flat fee should apply.
        assert!((cost - CostRates::default().per_job_fee).abs() < 1e-9);
    }

    #[test]
    fn test_estimate_job_positive() {
        let estimator = CostEstimator::default_rates();
        let cost = estimator.estimate_job(4.0, 1.0, 8.0, Duration::from_secs(3600));
        assert!(cost > 0.0);
    }

    #[test]
    fn test_estimate_usage_matches_manual() {
        let rates = CostRates {
            per_cpu_second: 1.0,
            per_gpu_second: 2.0,
            per_memory_gb_hour: 3.0,
            per_job_fee: 10.0,
        };
        let estimator = CostEstimator::new(rates);
        let mut usage = TenantUsage::default();
        usage.cpu_seconds = 100.0;
        usage.gpu_seconds = 50.0;
        usage.memory_gb_hours = 10.0;
        usage.jobs_submitted = 2;

        let expected = 10.0 * 2.0 + 100.0 * 1.0 + 50.0 * 2.0 + 10.0 * 3.0;
        let actual = estimator.estimate_usage(&usage);
        assert!((actual - expected).abs() < 1e-9);
    }

    // ── Tenant ────────────────────────────────────────────────────────────────

    #[test]
    fn test_tenant_can_accept_job_initial() {
        let tenant = make_tenant("t1", 3);
        assert!(tenant.can_accept_job());
    }

    #[test]
    fn test_tenant_quota_enforced() {
        let mut tenant = make_tenant("t1", 1);
        assert!(tenant.start_job());
        assert!(!tenant.start_job(), "quota of 1 should be exhausted");
    }

    #[test]
    fn test_tenant_finish_decrements_active() {
        let mut tenant = make_tenant("t1", 2);
        tenant.start_job();
        tenant.finish_job(2.0, 0.0, 4.0, Duration::from_secs(60), true);
        assert_eq!(tenant.active_jobs, 0);
        assert_eq!(tenant.usage.jobs_completed, 1);
    }

    #[test]
    fn test_tenant_disabled_rejects_jobs() {
        let mut tenant = make_tenant("t1", 10);
        tenant.enabled = false;
        assert!(!tenant.can_accept_job());
    }

    // ── TenantRegistry ────────────────────────────────────────────────────────

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = TenantRegistry::new();
        reg.register(make_tenant("t1", 5)).expect("register ok");
        assert_eq!(reg.tenant_count(), 1);
        assert!(reg.get("t1").is_some());
        assert!(reg.get("unknown").is_none());
    }

    #[test]
    fn test_registry_duplicate_returns_error() {
        let mut reg = TenantRegistry::new();
        reg.register(make_tenant("t1", 5)).expect("register ok");
        let err = reg.register(make_tenant("t1", 5)).unwrap_err();
        assert_eq!(err, TenantError::AlreadyExists("t1".to_string()));
    }

    #[test]
    fn test_registry_start_and_finish_job() {
        let mut reg = TenantRegistry::new();
        reg.register(make_tenant("t1", 5)).expect("register ok");
        reg.start_job("t1").expect("start ok");
        reg.finish_job("t1", 2.0, 0.0, 8.0, Duration::from_secs(120), true)
            .expect("finish ok");
        let t = reg.get("t1").expect("tenant");
        assert_eq!(t.active_jobs, 0);
        assert_eq!(t.usage.jobs_completed, 1);
    }

    #[test]
    fn test_registry_quota_exceeded_error() {
        let mut reg = TenantRegistry::new();
        reg.register(make_tenant("t1", 1)).expect("register ok");
        reg.start_job("t1").expect("first ok");
        let err = reg.start_job("t1").unwrap_err();
        assert_eq!(err, TenantError::QuotaExceeded("t1".to_string()));
    }

    #[test]
    fn test_registry_not_found_error() {
        let mut reg = TenantRegistry::new();
        let err = reg.start_job("ghost").unwrap_err();
        assert_eq!(err, TenantError::NotFound("ghost".to_string()));
    }

    #[test]
    fn test_registry_estimated_cost() {
        let estimator = CostEstimator::default_rates();
        let mut reg = TenantRegistry::with_estimator(estimator);
        reg.register(make_tenant("t1", 5)).expect("register ok");
        reg.start_job("t1").expect("start ok");
        reg.finish_job("t1", 4.0, 0.0, 8.0, Duration::from_secs(3600), true)
            .expect("finish ok");
        let cost = reg.estimated_cost("t1");
        assert!(cost.is_some());
        assert!(cost.expect("cost") > 0.0);
    }

    #[test]
    fn test_registry_no_estimator_returns_none() {
        let mut reg = TenantRegistry::new();
        reg.register(make_tenant("t1", 5)).expect("register ok");
        assert!(reg.estimated_cost("t1").is_none());
    }

    #[test]
    fn test_registry_over_budget_tenants() {
        let budget = ResourceBudget::new(1.0, 0.0, 0.0); // very low cpu budget
        let mut tenant = make_tenant("t1", 5);
        tenant.budget = budget;
        let mut reg = TenantRegistry::new();
        reg.register(tenant).expect("register ok");
        // Consume 200 CPU-seconds → over budget
        reg.finish_job("t1", 2.0, 0.0, 0.0, Duration::from_secs(100), true)
            .expect("finish ok");
        let over = reg.over_budget_tenants();
        assert_eq!(over.len(), 1);
        assert_eq!(over[0], "t1");
    }

    #[test]
    fn test_registry_reset_usage() {
        let mut reg = TenantRegistry::new();
        reg.register(make_tenant("t1", 5)).expect("register ok");
        reg.finish_job("t1", 4.0, 0.0, 8.0, Duration::from_secs(3600), true)
            .expect("finish ok");
        reg.reset_all_usage();
        let t = reg.get("t1").expect("tenant");
        assert_eq!(t.usage.cpu_seconds, 0.0);
    }

    #[test]
    fn test_tenant_error_display() {
        let err = TenantError::QuotaExceeded("acme".to_string());
        assert!(err.to_string().contains("quota"));
    }
}
