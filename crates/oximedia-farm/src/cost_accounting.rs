#![allow(dead_code)]
//! Farm cost accounting — per-job CPU/GPU hours, billing-period aggregation,
//! and cost anomaly detection.
//!
//! ## Model
//!
//! Every completed (or failed) job can have one or more [`ResourceUsage`]
//! records attached, describing how many CPU-seconds and GPU-seconds were
//! consumed on which worker.  A [`CostLedger`] accumulates these records,
//! applies configurable unit-cost rates, and exposes aggregated views:
//!
//! - Per-job totals
//! - Per-job-type totals
//! - Per-worker totals
//! - Billing-period summaries (daily / weekly / custom windows)
//! - Anomaly detection: jobs whose cost deviates more than `z_threshold`
//!   standard deviations from the rolling mean are flagged.
//!
//! ## Cost model
//!
//! ```text
//! job_cost = (cpu_seconds * cpu_rate_per_second)
//!          + (gpu_seconds * gpu_rate_per_second)
//!          + (memory_gb_seconds * mem_rate_per_gb_second)
//! ```
//!
//! All rates are user-configurable via [`CostRates`].

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::{FarmError, JobId, JobType, WorkerId};

// ---------------------------------------------------------------------------
// CostRates
// ---------------------------------------------------------------------------

/// Unit cost rates used to convert resource usage into a monetary cost.
///
/// All rates are expressed per **second** of resource consumption.
#[derive(Debug, Clone)]
pub struct CostRates {
    /// Cost per CPU-second (e.g. `0.000_028` USD/s ≈ $0.10/h).
    pub cpu_per_second: f64,
    /// Cost per GPU-second (typically much higher than CPU).
    pub gpu_per_second: f64,
    /// Cost per GB·s of memory usage.
    pub memory_gb_per_second: f64,
    /// Fixed overhead applied to every job regardless of duration.
    pub per_job_overhead: f64,
}

impl Default for CostRates {
    fn default() -> Self {
        Self {
            // ~$0.10/vCPU-hour
            cpu_per_second: 0.000_027_78,
            // ~$0.90/GPU-hour
            gpu_per_second: 0.000_25,
            // ~$0.01/GB-hour
            memory_gb_per_second: 0.000_002_78,
            per_job_overhead: 0.001,
        }
    }
}

impl CostRates {
    /// Compute the cost for a single [`ResourceUsage`] record.
    #[must_use]
    pub fn compute(&self, usage: &ResourceUsage) -> f64 {
        let cpu_cost = usage.cpu_seconds * self.cpu_per_second;
        let gpu_cost = usage.gpu_seconds * self.gpu_per_second;
        let mem_cost = usage.memory_gb_seconds * self.memory_gb_per_second;
        cpu_cost + gpu_cost + mem_cost + self.per_job_overhead
    }
}

// ---------------------------------------------------------------------------
// ResourceUsage
// ---------------------------------------------------------------------------

/// Resource consumption record for one job execution.
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// Job this record belongs to.
    pub job_id: JobId,
    /// Job type (for aggregation by job category).
    pub job_type: JobType,
    /// Worker that executed the job.
    pub worker_id: WorkerId,
    /// Wall-clock duration of the job (seconds).
    pub wall_seconds: f64,
    /// CPU time consumed (seconds; may exceed wall time on multi-core).
    pub cpu_seconds: f64,
    /// GPU time consumed (seconds; 0 for CPU-only jobs).
    pub gpu_seconds: f64,
    /// Integral of memory allocated over time (GB·s).
    pub memory_gb_seconds: f64,
    /// Unix timestamp (seconds since epoch) when the job finished.
    pub finished_at: u64,
}

impl ResourceUsage {
    /// Convenience constructor from wall-time and simple CPU/GPU/memory values.
    ///
    /// `finished_at` defaults to the current system time.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::InvalidConfig`] if any numeric field is negative.
    pub fn new(
        job_id: JobId,
        job_type: JobType,
        worker_id: WorkerId,
        wall_time: Duration,
        cpu_seconds: f64,
        gpu_seconds: f64,
        memory_gb_seconds: f64,
    ) -> crate::Result<Self> {
        if cpu_seconds < 0.0 || gpu_seconds < 0.0 || memory_gb_seconds < 0.0 {
            return Err(FarmError::InvalidConfig(
                "ResourceUsage: negative field value".into(),
            ));
        }
        let finished_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Ok(Self {
            job_id,
            job_type,
            worker_id,
            wall_seconds: wall_time.as_secs_f64(),
            cpu_seconds,
            gpu_seconds,
            memory_gb_seconds,
            finished_at,
        })
    }
}

// ---------------------------------------------------------------------------
// CostRecord  (internal)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct CostRecord {
    usage: ResourceUsage,
    total_cost: f64,
}

// ---------------------------------------------------------------------------
// BillingPeriodSummary
// ---------------------------------------------------------------------------

/// Aggregated cost summary for a time window.
#[derive(Debug, Clone, Default)]
pub struct BillingPeriodSummary {
    /// Start of the window (Unix seconds, inclusive).
    pub period_start: u64,
    /// End of the window (Unix seconds, exclusive).
    pub period_end: u64,
    /// Number of job records included.
    pub job_count: usize,
    /// Total cost across all jobs in the window.
    pub total_cost: f64,
    /// Total CPU seconds consumed.
    pub total_cpu_seconds: f64,
    /// Total GPU seconds consumed.
    pub total_gpu_seconds: f64,
    /// Breakdown of cost per [`JobType`].
    pub cost_by_job_type: HashMap<String, f64>,
    /// Breakdown of cost per worker.
    pub cost_by_worker: HashMap<String, f64>,
}

// ---------------------------------------------------------------------------
// AnomalyFlag
// ---------------------------------------------------------------------------

/// A job flagged as a cost anomaly.
#[derive(Debug, Clone)]
pub struct AnomalyFlag {
    /// The anomalous job.
    pub job_id: JobId,
    /// Computed cost for this job.
    pub cost: f64,
    /// Current rolling mean cost across all jobs.
    pub mean_cost: f64,
    /// Current rolling standard deviation.
    pub std_dev: f64,
    /// Z-score: `(cost - mean) / std_dev`.
    pub z_score: f64,
}

// ---------------------------------------------------------------------------
// CostLedger
// ---------------------------------------------------------------------------

/// Central ledger that accumulates resource-usage records, computes costs,
/// and provides analysis views.
#[derive(Debug)]
pub struct CostLedger {
    rates: CostRates,
    records: Vec<CostRecord>,
    /// Welford's online algorithm state for incremental mean/variance.
    welford_count: usize,
    welford_mean: f64,
    welford_m2: f64,
    /// Z-score threshold above which a job is flagged as anomalous.
    z_threshold: f64,
}

impl CostLedger {
    /// Create a new ledger with default cost rates and a z-score threshold of 3.
    #[must_use]
    pub fn new() -> Self {
        Self::with_rates(CostRates::default(), 3.0)
    }

    /// Create a ledger with custom rates and anomaly z-score threshold.
    ///
    /// `z_threshold` must be positive (typically 2.0–4.0).
    #[must_use]
    pub fn with_rates(rates: CostRates, z_threshold: f64) -> Self {
        Self {
            rates,
            records: Vec::new(),
            welford_count: 0,
            welford_mean: 0.0,
            welford_m2: 0.0,
            z_threshold: z_threshold.abs().max(0.01),
        }
    }

    // ------------------------------------------------------------------
    // Ingestion
    // ------------------------------------------------------------------

    /// Record a new job's resource usage.
    ///
    /// Returns the computed cost for this job and, if the job is a statistical
    /// outlier, an [`AnomalyFlag`].
    pub fn record(&mut self, usage: ResourceUsage) -> (f64, Option<AnomalyFlag>) {
        let cost = self.rates.compute(&usage);

        // Welford online mean/variance update.
        self.welford_count += 1;
        let delta = cost - self.welford_mean;
        self.welford_mean += delta / self.welford_count as f64;
        let delta2 = cost - self.welford_mean;
        self.welford_m2 += delta * delta2;

        let anomaly = if self.welford_count >= 10 {
            let variance = self.welford_m2 / (self.welford_count - 1) as f64;
            let std_dev = variance.sqrt();
            if std_dev > 1e-12 {
                let z = (cost - self.welford_mean) / std_dev;
                if z.abs() > self.z_threshold {
                    Some(AnomalyFlag {
                        job_id: usage.job_id,
                        cost,
                        mean_cost: self.welford_mean,
                        std_dev,
                        z_score: z,
                    })
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        self.records.push(CostRecord {
            usage,
            total_cost: cost,
        });

        (cost, anomaly)
    }

    // ------------------------------------------------------------------
    // Per-job queries
    // ------------------------------------------------------------------

    /// Total cost for all usage records attributed to `job_id`.
    #[must_use]
    pub fn cost_for_job(&self, job_id: JobId) -> f64 {
        self.records
            .iter()
            .filter(|r| r.usage.job_id == job_id)
            .map(|r| r.total_cost)
            .sum()
    }

    /// All usage records for a given job.
    pub fn records_for_job(&self, job_id: JobId) -> Vec<&ResourceUsage> {
        self.records
            .iter()
            .filter(|r| r.usage.job_id == job_id)
            .map(|r| &r.usage)
            .collect()
    }

    // ------------------------------------------------------------------
    // Per-job-type aggregation
    // ------------------------------------------------------------------

    /// Total cost aggregated by job type.
    #[must_use]
    pub fn cost_by_job_type(&self) -> HashMap<String, f64> {
        let mut map: HashMap<String, f64> = HashMap::new();
        for rec in &self.records {
            *map.entry(rec.usage.job_type.to_string()).or_insert(0.0) += rec.total_cost;
        }
        map
    }

    // ------------------------------------------------------------------
    // Per-worker aggregation
    // ------------------------------------------------------------------

    /// Total cost aggregated by worker id.
    #[must_use]
    pub fn cost_by_worker(&self) -> HashMap<String, f64> {
        let mut map: HashMap<String, f64> = HashMap::new();
        for rec in &self.records {
            *map.entry(rec.usage.worker_id.0.clone()).or_insert(0.0) += rec.total_cost;
        }
        map
    }

    // ------------------------------------------------------------------
    // Billing period
    // ------------------------------------------------------------------

    /// Summarize all records whose `finished_at` falls within
    /// `[period_start, period_end)`.
    #[must_use]
    pub fn billing_period_summary(
        &self,
        period_start: u64,
        period_end: u64,
    ) -> BillingPeriodSummary {
        let mut summary = BillingPeriodSummary {
            period_start,
            period_end,
            ..Default::default()
        };
        for rec in &self.records {
            let t = rec.usage.finished_at;
            if t >= period_start && t < period_end {
                summary.job_count += 1;
                summary.total_cost += rec.total_cost;
                summary.total_cpu_seconds += rec.usage.cpu_seconds;
                summary.total_gpu_seconds += rec.usage.gpu_seconds;
                *summary
                    .cost_by_job_type
                    .entry(rec.usage.job_type.to_string())
                    .or_insert(0.0) += rec.total_cost;
                *summary
                    .cost_by_worker
                    .entry(rec.usage.worker_id.0.clone())
                    .or_insert(0.0) += rec.total_cost;
            }
        }
        summary
    }

    /// Convenience: summarise the last `window` duration up to now.
    #[must_use]
    pub fn recent_summary(&self, window: Duration) -> BillingPeriodSummary {
        let end = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let start = end.saturating_sub(window.as_secs());
        self.billing_period_summary(start, end + 1)
    }

    // ------------------------------------------------------------------
    // Rolling statistics
    // ------------------------------------------------------------------

    /// Current rolling mean cost per job.
    #[must_use]
    pub fn mean_cost(&self) -> f64 {
        self.welford_mean
    }

    /// Current rolling standard deviation of cost per job.
    #[must_use]
    pub fn std_dev_cost(&self) -> f64 {
        if self.welford_count < 2 {
            return 0.0;
        }
        let variance = self.welford_m2 / (self.welford_count - 1) as f64;
        variance.sqrt()
    }

    /// Total number of recorded jobs.
    #[must_use]
    pub fn job_count(&self) -> usize {
        self.records.len()
    }

    /// Grand total cost across all records.
    #[must_use]
    pub fn grand_total(&self) -> f64 {
        self.records.iter().map(|r| r.total_cost).sum()
    }
}

impl Default for CostLedger {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// JobCostEstimator — pre-run cost prediction with budget enforcement
// ---------------------------------------------------------------------------

/// A pre-run resource profile describing the expected consumption of a job.
///
/// Used by [`JobCostEstimator`] to predict cost before the job executes.
#[derive(Debug, Clone)]
pub struct JobProfile {
    /// Expected number of CPU cores used.
    pub cpu_cores: f64,
    /// Expected number of GPU devices used (0.0 for CPU-only jobs).
    pub gpu_devices: f64,
    /// Expected memory in GB.
    pub memory_gb: f64,
    /// Expected wall-clock duration.
    pub estimated_duration: Duration,
    /// Job type identifier for per-type budget tracking.
    pub job_type: JobType,
}

impl JobProfile {
    /// Create a CPU-only profile.
    #[must_use]
    pub fn cpu_only(cpu_cores: f64, memory_gb: f64, estimated_duration: Duration, job_type: JobType) -> Self {
        Self {
            cpu_cores,
            gpu_devices: 0.0,
            memory_gb,
            estimated_duration,
            job_type,
        }
    }

    /// Create a GPU-accelerated profile.
    #[must_use]
    pub fn gpu(
        cpu_cores: f64,
        gpu_devices: f64,
        memory_gb: f64,
        estimated_duration: Duration,
        job_type: JobType,
    ) -> Self {
        Self {
            cpu_cores,
            gpu_devices,
            memory_gb,
            estimated_duration,
            job_type,
        }
    }
}

/// A per-job-type spending budget.
#[derive(Debug, Clone)]
pub struct JobTypeBudget {
    /// Maximum spend allowed per billing period, keyed by [`JobType`] display name.
    pub limits: HashMap<String, f64>,
}

impl JobTypeBudget {
    /// Create an empty budget with no limits.
    #[must_use]
    pub fn new() -> Self {
        Self {
            limits: HashMap::new(),
        }
    }

    /// Set the limit for a job type.
    pub fn set(&mut self, job_type: JobType, max_spend: f64) {
        self.limits.insert(job_type.to_string(), max_spend);
    }

    /// Get the limit for a job type (returns `f64::MAX` if unconstrained).
    #[must_use]
    pub fn limit_for(&self, job_type: JobType) -> f64 {
        self.limits
            .get(&job_type.to_string())
            .copied()
            .unwrap_or(f64::MAX)
    }
}

impl Default for JobTypeBudget {
    fn default() -> Self {
        Self::new()
    }
}

/// Budget alert returned when estimated cost would breach a limit.
#[derive(Debug, Clone)]
pub struct BudgetAlert {
    /// Job type that would breach its budget.
    pub job_type: String,
    /// Estimated cost of the proposed job.
    pub estimated_cost: f64,
    /// Current accumulated spend for this job type.
    pub current_spend: f64,
    /// Configured limit.
    pub limit: f64,
    /// Overage (estimated_cost + current_spend - limit).
    pub overage: f64,
}

/// A cost estimation and budget-enforcement engine.
///
/// Works alongside [`CostLedger`] for post-hoc accounting. The estimator
/// answers the pre-flight question: _"how much will this job cost, and does
/// it fit within the current budget?"_
///
/// ## Cost model
///
/// ```text
/// estimated_cost = overhead
///                + (cpu_cores × duration_s × cpu_per_second)
///                + (gpu_devices × duration_s × gpu_per_second)
///                + (memory_gb × duration_s × memory_gb_per_second)
/// ```
#[derive(Debug)]
pub struct JobCostEstimator {
    rates: CostRates,
    budget: JobTypeBudget,
    /// Accumulated spend per job-type name (updated when `record_actual` is called).
    spent: HashMap<String, f64>,
    /// Total spend cap across all job types (0.0 = unlimited).
    total_cap: f64,
}

impl JobCostEstimator {
    /// Create an estimator with default rates, no budgets, and no total cap.
    #[must_use]
    pub fn new() -> Self {
        Self::with_rates(CostRates::default())
    }

    /// Create an estimator with custom rates.
    #[must_use]
    pub fn with_rates(rates: CostRates) -> Self {
        Self {
            rates,
            budget: JobTypeBudget::new(),
            spent: HashMap::new(),
            total_cap: 0.0,
        }
    }

    /// Set per-job-type budget limits.
    pub fn set_budget(&mut self, budget: JobTypeBudget) {
        self.budget = budget;
    }

    /// Set a hard total spend cap across all job types.
    pub fn set_total_cap(&mut self, cap: f64) {
        self.total_cap = cap;
    }

    /// Estimate the cost of running a job described by `profile`.
    ///
    /// This does **not** check budgets — use [`Self::check_budget`] for that.
    #[must_use]
    pub fn estimate(&self, profile: &JobProfile) -> f64 {
        let secs = profile.estimated_duration.as_secs_f64();
        let r = &self.rates;
        r.per_job_overhead
            + profile.cpu_cores * secs * r.cpu_per_second
            + profile.gpu_devices * secs * r.gpu_per_second
            + profile.memory_gb * secs * r.memory_gb_per_second
    }

    /// Check whether submitting `profile` would breach any configured budget.
    ///
    /// Returns `Some(alert)` when a budget would be breached, `None` when the
    /// job fits within all limits.
    #[must_use]
    pub fn check_budget(&self, profile: &JobProfile) -> Option<BudgetAlert> {
        let estimated = self.estimate(profile);
        let type_key = profile.job_type.to_string();

        // Per-type check.
        let type_limit = self.budget.limit_for(profile.job_type);
        let type_spent = self.spent.get(&type_key).copied().unwrap_or(0.0);
        if type_spent + estimated > type_limit {
            return Some(BudgetAlert {
                job_type: type_key,
                estimated_cost: estimated,
                current_spend: type_spent,
                limit: type_limit,
                overage: (type_spent + estimated) - type_limit,
            });
        }

        // Total cap check.
        if self.total_cap > 0.0 {
            let total_spent: f64 = self.spent.values().sum();
            if total_spent + estimated > self.total_cap {
                return Some(BudgetAlert {
                    job_type: type_key,
                    estimated_cost: estimated,
                    current_spend: total_spent,
                    limit: self.total_cap,
                    overage: (total_spent + estimated) - self.total_cap,
                });
            }
        }

        None
    }

    /// Record the actual cost of a completed job to update accumulated spend.
    pub fn record_actual(&mut self, job_type: JobType, actual_cost: f64) {
        *self.spent.entry(job_type.to_string()).or_insert(0.0) += actual_cost;
    }

    /// Total accumulated spend across all job types.
    #[must_use]
    pub fn total_spend(&self) -> f64 {
        self.spent.values().sum()
    }

    /// Accumulated spend for one job type.
    #[must_use]
    pub fn spend_for_type(&self, job_type: JobType) -> f64 {
        self.spent
            .get(&job_type.to_string())
            .copied()
            .unwrap_or(0.0)
    }

    /// Reset accumulated spend counters (e.g., at the start of a new billing period).
    pub fn reset_spend(&mut self) {
        self.spent.clear();
    }

    /// Generate a `CostReport` summarising current estimated spend vs. budgets.
    #[must_use]
    pub fn cost_report(&self) -> CostReport {
        let total_spend = self.total_spend();
        let mut per_type: Vec<CostReportEntry> = self
            .spent
            .iter()
            .map(|(k, &spend)| {
                let type_limit = self.budget.limits.get(k).copied().unwrap_or(f64::MAX);
                CostReportEntry {
                    job_type: k.clone(),
                    spend,
                    limit: if type_limit == f64::MAX {
                        None
                    } else {
                        Some(type_limit)
                    },
                    utilisation: if type_limit > 0.0 && type_limit != f64::MAX {
                        (spend / type_limit).min(1.0)
                    } else {
                        0.0
                    },
                }
            })
            .collect();
        per_type.sort_by(|a, b| b.spend.partial_cmp(&a.spend).unwrap_or(std::cmp::Ordering::Equal));

        CostReport {
            total_spend,
            total_cap: if self.total_cap > 0.0 {
                Some(self.total_cap)
            } else {
                None
            },
            per_type,
        }
    }
}

impl Default for JobCostEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// A summary of current spending vs budgets.
#[derive(Debug, Clone)]
pub struct CostReport {
    /// Total spend across all job types.
    pub total_spend: f64,
    /// Total cap (if configured).
    pub total_cap: Option<f64>,
    /// Per-job-type breakdown, sorted by spend (descending).
    pub per_type: Vec<CostReportEntry>,
}

/// A single row in a [`CostReport`].
#[derive(Debug, Clone)]
pub struct CostReportEntry {
    /// Job type name.
    pub job_type: String,
    /// Accumulated spend.
    pub spend: f64,
    /// Budget limit (None if uncapped).
    pub limit: Option<f64>,
    /// `spend / limit` clamped to 0–1 (0.0 when uncapped).
    pub utilisation: f64,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::JobType;

    fn make_usage(job_id: JobId, job_type: JobType, worker: &str, cpu_s: f64) -> ResourceUsage {
        ResourceUsage::new(
            job_id,
            job_type,
            WorkerId::new(worker),
            Duration::from_secs(cpu_s as u64),
            cpu_s,
            0.0,
            1.0,
        )
        .expect("usage creation should succeed")
    }

    #[test]
    fn test_basic_cost_recording() {
        let mut ledger = CostLedger::new();
        let jid = JobId::new();
        let usage = make_usage(jid, JobType::VideoTranscode, "w1", 100.0);
        let (cost, anomaly) = ledger.record(usage);
        assert!(cost > 0.0);
        assert!(anomaly.is_none()); // too few samples for anomaly detection
        assert!((ledger.cost_for_job(jid) - cost).abs() < 1e-9);
    }

    #[test]
    fn test_cost_rates_custom() {
        let rates = CostRates {
            cpu_per_second: 0.01,
            gpu_per_second: 0.0,
            memory_gb_per_second: 0.0,
            per_job_overhead: 0.0,
        };
        let mut ledger = CostLedger::with_rates(rates, 3.0);
        let usage = make_usage(JobId::new(), JobType::AudioTranscode, "w1", 10.0);
        let (cost, _) = ledger.record(usage);
        assert!((cost - 0.10).abs() < 1e-9);
    }

    #[test]
    fn test_aggregation_by_job_type() {
        let mut ledger = CostLedger::new();
        for _ in 0..3 {
            ledger.record(make_usage(JobId::new(), JobType::VideoTranscode, "w1", 60.0));
        }
        for _ in 0..2 {
            ledger.record(make_usage(JobId::new(), JobType::AudioTranscode, "w1", 30.0));
        }
        let by_type = ledger.cost_by_job_type();
        assert!(by_type.contains_key("VideoTranscode"));
        assert!(by_type.contains_key("AudioTranscode"));
        assert!(by_type["VideoTranscode"] > by_type["AudioTranscode"]);
    }

    #[test]
    fn test_aggregation_by_worker() {
        let mut ledger = CostLedger::new();
        for i in 0..4u64 {
            let worker = if i % 2 == 0 { "w1" } else { "w2" };
            ledger.record(make_usage(JobId::new(), JobType::VideoTranscode, worker, 60.0));
        }
        let by_worker = ledger.cost_by_worker();
        assert_eq!(by_worker.len(), 2);
        assert!((by_worker["w1"] - by_worker["w2"]).abs() < 1e-9);
    }

    #[test]
    fn test_billing_period_summary() {
        let mut ledger = CostLedger::new();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // All jobs land within the window.
        for _ in 0..5 {
            let usage = make_usage(JobId::new(), JobType::VideoTranscode, "w1", 60.0);
            ledger.record(usage);
        }
        let summary = ledger.billing_period_summary(now - 10, now + 10);
        assert_eq!(summary.job_count, 5);
        assert!(summary.total_cost > 0.0);
        assert!(summary.total_cpu_seconds > 0.0);
    }

    #[test]
    fn test_billing_period_excludes_outside() {
        let mut ledger = CostLedger::new();
        for _ in 0..3 {
            ledger.record(make_usage(JobId::new(), JobType::VideoTranscode, "w1", 60.0));
        }
        // Use a window far in the past — nothing should match.
        let summary = ledger.billing_period_summary(0, 1);
        assert_eq!(summary.job_count, 0);
        assert_eq!(summary.total_cost, 0.0);
    }

    #[test]
    fn test_anomaly_detection() {
        let mut ledger = CostLedger::with_rates(CostRates::default(), 2.0);
        // Record 15 cheap jobs.
        for _ in 0..15 {
            ledger.record(make_usage(JobId::new(), JobType::VideoTranscode, "w1", 10.0));
        }
        // Record one extremely expensive job.
        let (_, anomaly) =
            ledger.record(make_usage(JobId::new(), JobType::VideoTranscode, "w1", 100_000.0));
        assert!(
            anomaly.is_some(),
            "expensive job should be flagged as anomaly"
        );
        let flag = anomaly.expect("anomaly should be present");
        assert!(flag.z_score > 2.0);
    }

    #[test]
    fn test_rolling_stats_update() {
        let mut ledger = CostLedger::new();
        for i in 1..=10u32 {
            ledger.record(make_usage(
                JobId::new(),
                JobType::VideoTranscode,
                "w1",
                i as f64 * 10.0,
            ));
        }
        assert!(ledger.mean_cost() > 0.0);
        assert!(ledger.std_dev_cost() > 0.0);
    }

    #[test]
    fn test_negative_usage_error() {
        let result = ResourceUsage::new(
            JobId::new(),
            JobType::VideoTranscode,
            WorkerId::new("w1"),
            Duration::from_secs(10),
            -1.0, // negative CPU seconds
            0.0,
            0.0,
        );
        assert!(matches!(result, Err(FarmError::InvalidConfig(_))));
    }

    #[test]
    fn test_grand_total() {
        let rates = CostRates {
            cpu_per_second: 1.0,
            gpu_per_second: 0.0,
            memory_gb_per_second: 0.0,
            per_job_overhead: 0.0,
        };
        let mut ledger = CostLedger::with_rates(rates, 3.0);
        ledger.record(make_usage(JobId::new(), JobType::AudioTranscode, "w1", 5.0));
        ledger.record(make_usage(JobId::new(), JobType::AudioTranscode, "w1", 3.0));
        // total CPU seconds = 8, rate=1.0, overhead=0 → total=8.0
        assert!((ledger.grand_total() - 8.0).abs() < 1e-9);
    }

    // -- JobCostEstimator tests ----------------------------------------------

    fn simple_estimator() -> JobCostEstimator {
        let rates = CostRates {
            cpu_per_second: 1.0,
            gpu_per_second: 2.0,
            memory_gb_per_second: 0.0,
            per_job_overhead: 0.0,
        };
        JobCostEstimator::with_rates(rates)
    }

    fn cpu_profile(secs: u64) -> JobProfile {
        JobProfile::cpu_only(1.0, 4.0, Duration::from_secs(secs), JobType::VideoTranscode)
    }

    fn gpu_profile(secs: u64) -> JobProfile {
        JobProfile::gpu(1.0, 1.0, 4.0, Duration::from_secs(secs), JobType::VideoTranscode)
    }

    #[test]
    fn test_estimate_cpu_only() {
        let est = simple_estimator();
        let profile = cpu_profile(10);
        // cpu_cores=1 * 10s * rate=1.0 = 10.0
        assert!((est.estimate(&profile) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_estimate_gpu_adds_gpu_cost() {
        let est = simple_estimator();
        let profile = gpu_profile(10);
        // cpu=1*10*1 + gpu=1*10*2 = 30.0
        assert!((est.estimate(&profile) - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_check_budget_no_limit_returns_none() {
        let est = simple_estimator();
        let profile = cpu_profile(10_000);
        assert!(est.check_budget(&profile).is_none(), "no budget configured → no alert");
    }

    #[test]
    fn test_check_budget_within_limit_returns_none() {
        let mut est = simple_estimator();
        let mut budget = JobTypeBudget::new();
        budget.set(JobType::VideoTranscode, 1_000.0);
        est.set_budget(budget);
        let profile = cpu_profile(10); // cost = 10
        assert!(est.check_budget(&profile).is_none());
    }

    #[test]
    fn test_check_budget_exceeds_limit_returns_alert() {
        let mut est = simple_estimator();
        let mut budget = JobTypeBudget::new();
        budget.set(JobType::VideoTranscode, 5.0); // low limit
        est.set_budget(budget);
        let profile = cpu_profile(10); // cost = 10 > 5
        let alert = est.check_budget(&profile);
        assert!(alert.is_some());
        let a = alert.expect("alert");
        assert!(a.overage > 0.0);
        assert_eq!(a.limit, 5.0);
    }

    #[test]
    fn test_check_budget_accumulated_spend_triggers_alert() {
        let mut est = simple_estimator();
        let mut budget = JobTypeBudget::new();
        budget.set(JobType::VideoTranscode, 15.0);
        est.set_budget(budget);
        // Record 10 of spend.
        est.record_actual(JobType::VideoTranscode, 10.0);
        // Remaining budget = 5; new job costs 10 → should alert.
        let profile = cpu_profile(10); // cost = 10
        let alert = est.check_budget(&profile);
        assert!(alert.is_some());
    }

    #[test]
    fn test_total_cap_triggers_alert() {
        let mut est = simple_estimator();
        est.set_total_cap(8.0);
        est.record_actual(JobType::VideoTranscode, 5.0);
        // Total spent = 5; new job costs 10 → 15 > 8 cap.
        let profile = cpu_profile(10);
        assert!(est.check_budget(&profile).is_some());
    }

    #[test]
    fn test_record_actual_updates_spend() {
        let mut est = simple_estimator();
        est.record_actual(JobType::VideoTranscode, 25.0);
        est.record_actual(JobType::AudioTranscode, 10.0);
        assert!((est.total_spend() - 35.0).abs() < 1e-9);
        assert!((est.spend_for_type(JobType::VideoTranscode) - 25.0).abs() < 1e-9);
        assert!((est.spend_for_type(JobType::AudioTranscode) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_reset_spend() {
        let mut est = simple_estimator();
        est.record_actual(JobType::VideoTranscode, 50.0);
        est.reset_spend();
        assert!((est.total_spend() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_cost_report_structure() {
        let mut est = simple_estimator();
        let mut budget = JobTypeBudget::new();
        budget.set(JobType::VideoTranscode, 100.0);
        est.set_budget(budget);
        est.set_total_cap(200.0);
        est.record_actual(JobType::VideoTranscode, 40.0);
        est.record_actual(JobType::AudioTranscode, 15.0);

        let report = est.cost_report();
        assert!((report.total_spend - 55.0).abs() < 1e-9);
        assert_eq!(report.total_cap, Some(200.0));
        // Sorted descending by spend → VideoTranscode (40) before AudioTranscode (15).
        assert!(!report.per_type.is_empty());
        let video_entry = report
            .per_type
            .iter()
            .find(|e| e.job_type == "VideoTranscode")
            .expect("VideoTranscode entry missing");
        assert!((video_entry.spend - 40.0).abs() < 1e-9);
        assert_eq!(video_entry.limit, Some(100.0));
        assert!((video_entry.utilisation - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_job_profile_cpu_only_has_no_gpu() {
        let profile = JobProfile::cpu_only(4.0, 8.0, Duration::from_secs(60), JobType::AudioTranscode);
        assert_eq!(profile.gpu_devices, 0.0);
        assert_eq!(profile.cpu_cores, 4.0);
    }

    #[test]
    fn test_job_type_budget_unconstrained_returns_max() {
        let budget = JobTypeBudget::new(); // no limits set
        assert_eq!(budget.limit_for(JobType::VideoTranscode), f64::MAX);
    }
}
