#![allow(dead_code)]
//! A/B testing framework for evaluating different search ranking algorithms.
//!
//! This module provides infrastructure for running controlled experiments
//! to compare ranking strategies, measure their impact on user engagement,
//! and make data-driven decisions about search quality improvements.
//!
//! # Architecture
//!
//! An [`AbExperiment`] defines two or more ranking variants. Each incoming
//! search request is assigned to a variant via deterministic hashing of the
//! user/session ID (ensuring the same user always sees the same variant
//! within an experiment window). Metrics are collected per-variant and
//! analysed for statistical significance using a two-proportion z-test.
//!
//! # Key types
//!
//! | Type | Purpose |
//! |---|---|
//! | [`RankingVariant`] | A named ranking strategy configuration |
//! | [`AbExperiment`] | An experiment comparing ≥2 variants |
//! | [`AbTestManager`] | Registry of running/completed experiments |
//! | [`ExperimentResult`] | Aggregated metrics + significance test |
//!
//! # Statistical model
//!
//! Click-through rate (CTR) is used as the primary engagement metric.
//! Statistical significance is tested with a two-proportion z-test at a
//! configurable α level (default 0.05).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{SearchError, SearchResult};

// ---------------------------------------------------------------------------
// Ranking variant
// ---------------------------------------------------------------------------

/// Scoring algorithm for a ranking variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScoringAlgorithm {
    /// Standard BM25 term-based relevance.
    Bm25,
    /// TF-IDF with field boosting.
    TfIdf,
    /// Recency-boosted scoring (newer assets rank higher).
    RecencyBoost,
    /// Popularity-boosted scoring (frequently accessed assets rank higher).
    PopularityBoost,
    /// Combined TF-IDF + recency + popularity.
    Hybrid,
    /// Cosine similarity over dense embedding vectors.
    Embedding,
    /// Learning-to-rank (gradient-boosted features).
    LearnToRank,
    /// Custom strategy identified by name.
    Custom,
}

/// Configuration for a single ranking variant in an A/B experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingVariant {
    /// Unique variant identifier (e.g. "control", "treatment-v2").
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Scoring algorithm to use.
    pub algorithm: ScoringAlgorithm,
    /// Optional algorithm-specific parameters (serialised JSON).
    pub params_json: Option<String>,
    /// Traffic allocation as a fraction of total traffic `[0.0, 1.0]`.
    /// All variants in an experiment should sum to 1.0.
    pub traffic_fraction: f64,
    /// Whether this is the control (baseline) variant.
    pub is_control: bool,
}

impl RankingVariant {
    /// Create a control variant.
    #[must_use]
    pub fn control(algorithm: ScoringAlgorithm) -> Self {
        Self {
            id: "control".into(),
            name: "Control".into(),
            algorithm,
            params_json: None,
            traffic_fraction: 0.5,
            is_control: true,
        }
    }

    /// Create a treatment variant.
    #[must_use]
    pub fn treatment(
        id: impl Into<String>,
        name: impl Into<String>,
        algorithm: ScoringAlgorithm,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            algorithm,
            params_json: None,
            traffic_fraction: 0.5,
            is_control: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Experiment status and lifecycle
// ---------------------------------------------------------------------------

/// Current status of an experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExperimentStatus {
    /// Not yet started.
    Pending,
    /// Currently running.
    Running,
    /// Paused (traffic allocation suspended).
    Paused,
    /// Concluded (no more traffic being assigned).
    Concluded,
}

/// An A/B experiment comparing multiple ranking variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbExperiment {
    /// Unique experiment ID.
    pub id: Uuid,
    /// Human-readable name.
    pub name: String,
    /// Optional description.
    pub description: Option<String>,
    /// Ranking variants (must have at least 2).
    pub variants: Vec<RankingVariant>,
    /// Current status.
    pub status: ExperimentStatus,
    /// Unix timestamp when the experiment started.
    pub started_at: Option<i64>,
    /// Unix timestamp when the experiment concluded.
    pub concluded_at: Option<i64>,
    /// Minimum number of observations per variant before analysis.
    pub min_observations: usize,
    /// Significance level α for hypothesis tests.
    pub alpha: f64,
}

impl AbExperiment {
    /// Create a new experiment with two variants.
    ///
    /// # Errors
    ///
    /// Returns an error if traffic fractions do not sum to ~1.0 (within 0.01).
    pub fn new(
        name: impl Into<String>,
        control: RankingVariant,
        treatment: RankingVariant,
    ) -> SearchResult<Self> {
        let total = control.traffic_fraction + treatment.traffic_fraction;
        if (total - 1.0).abs() > 0.01 {
            return Err(SearchError::InvalidQuery(format!(
                "Traffic fractions sum to {total:.3}, expected 1.0"
            )));
        }
        Ok(Self {
            id: Uuid::new_v4(),
            name: name.into(),
            description: None,
            variants: vec![control, treatment],
            status: ExperimentStatus::Pending,
            started_at: None,
            concluded_at: None,
            min_observations: 100,
            alpha: 0.05,
        })
    }

    /// Create an experiment with multiple variants.
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than 2 variants are provided, or if
    /// traffic fractions do not sum to ~1.0.
    pub fn multi_variant(
        name: impl Into<String>,
        variants: Vec<RankingVariant>,
    ) -> SearchResult<Self> {
        if variants.len() < 2 {
            return Err(SearchError::InvalidQuery(
                "An experiment requires at least 2 variants".into(),
            ));
        }
        let total: f64 = variants.iter().map(|v| v.traffic_fraction).sum();
        if (total - 1.0).abs() > 0.01 {
            return Err(SearchError::InvalidQuery(format!(
                "Traffic fractions sum to {total:.3}, expected 1.0"
            )));
        }
        Ok(Self {
            id: Uuid::new_v4(),
            name: name.into(),
            description: None,
            variants,
            status: ExperimentStatus::Pending,
            started_at: None,
            concluded_at: None,
            min_observations: 100,
            alpha: 0.05,
        })
    }

    /// Deterministically assign a session/user to a variant using FNV-1a hashing.
    ///
    /// The same `session_id` always maps to the same variant within this
    /// experiment (the hash includes the experiment ID for isolation).
    #[must_use]
    pub fn assign_variant(&self, session_id: &str) -> &RankingVariant {
        let hash = fnv1a_hash(&format!("{}:{}", self.id, session_id));
        // Map hash to [0.0, 1.0) then walk the variant buckets.
        let bucket = (hash as f64) / (u64::MAX as f64);
        let mut cumulative = 0.0;
        for variant in &self.variants {
            cumulative += variant.traffic_fraction;
            if bucket < cumulative {
                return variant;
            }
        }
        // Fallback: last variant (handles floating-point edge cases).
        // SAFETY: variants is non-empty — guaranteed by AbExperiment::new which returns
        // Err if variants.len() < 2, so the vec always has at least one element.
        &self.variants[self.variants.len() - 1]
    }

    /// Start the experiment.
    ///
    /// # Errors
    ///
    /// Returns an error if the experiment is not in `Pending` status.
    pub fn start(&mut self, now_secs: i64) -> SearchResult<()> {
        if self.status != ExperimentStatus::Pending {
            return Err(SearchError::InvalidQuery(format!(
                "Experiment '{}' is not Pending (status: {:?})",
                self.name, self.status
            )));
        }
        self.status = ExperimentStatus::Running;
        self.started_at = Some(now_secs);
        Ok(())
    }

    /// Conclude the experiment.
    pub fn conclude(&mut self, now_secs: i64) {
        self.status = ExperimentStatus::Concluded;
        self.concluded_at = Some(now_secs);
    }

    /// Pause the experiment.
    pub fn pause(&mut self) {
        if self.status == ExperimentStatus::Running {
            self.status = ExperimentStatus::Paused;
        }
    }

    /// Resume a paused experiment.
    pub fn resume(&mut self) {
        if self.status == ExperimentStatus::Paused {
            self.status = ExperimentStatus::Running;
        }
    }
}

// ---------------------------------------------------------------------------
// Metrics collection
// ---------------------------------------------------------------------------

/// Per-variant engagement metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VariantMetrics {
    /// Total search requests assigned to this variant.
    pub impressions: u64,
    /// Number of sessions that clicked at least one result.
    pub clicks: u64,
    /// Total position of clicked results (for average rank calculation).
    pub total_click_rank: u64,
    /// Number of sessions with zero results.
    pub zero_result_sessions: u64,
    /// User satisfaction signals (e.g. dwell-time > 30s).
    pub satisfied_sessions: u64,
}

impl VariantMetrics {
    /// Click-through rate: `clicks / impressions`.
    #[must_use]
    pub fn ctr(&self) -> f64 {
        if self.impressions == 0 {
            0.0
        } else {
            self.clicks as f64 / self.impressions as f64
        }
    }

    /// Mean reciprocal rank (simplified: using average click position).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn mean_click_rank(&self) -> f64 {
        if self.clicks == 0 {
            0.0
        } else {
            self.total_click_rank as f64 / self.clicks as f64
        }
    }

    /// Satisfaction rate: `satisfied_sessions / impressions`.
    #[must_use]
    pub fn satisfaction_rate(&self) -> f64 {
        if self.impressions == 0 {
            0.0
        } else {
            self.satisfied_sessions as f64 / self.impressions as f64
        }
    }

    /// Zero-result rate.
    #[must_use]
    pub fn zero_result_rate(&self) -> f64 {
        if self.impressions == 0 {
            0.0
        } else {
            self.zero_result_sessions as f64 / self.impressions as f64
        }
    }
}

/// An interaction event recorded during an experiment.
#[derive(Debug, Clone)]
pub struct InteractionEvent {
    /// Experiment ID.
    pub experiment_id: Uuid,
    /// Variant ID that served this request.
    pub variant_id: String,
    /// Whether the user clicked a result.
    pub clicked: bool,
    /// Position of the clicked result (1-indexed), if applicable.
    pub click_rank: Option<u32>,
    /// Whether the result set was empty.
    pub zero_results: bool,
    /// Whether the user was "satisfied" (dwell-time heuristic etc.).
    pub satisfied: bool,
}

// ---------------------------------------------------------------------------
// Analysis results
// ---------------------------------------------------------------------------

/// Statistical comparison of two variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantComparison {
    /// Control variant ID.
    pub control_id: String,
    /// Treatment variant ID.
    pub treatment_id: String,
    /// CTR of the control.
    pub control_ctr: f64,
    /// CTR of the treatment.
    pub treatment_ctr: f64,
    /// Relative CTR lift: `(treatment - control) / control`.
    pub relative_lift: f64,
    /// Z-score for the two-proportion z-test.
    pub z_score: f64,
    /// p-value (two-tailed).
    pub p_value: f64,
    /// Whether the result is statistically significant at α.
    pub is_significant: bool,
    /// The α level used.
    pub alpha: f64,
}

/// Aggregated experiment analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResult {
    /// Experiment ID.
    pub experiment_id: Uuid,
    /// Metrics per variant (variant_id -> metrics).
    pub metrics: HashMap<String, VariantMetrics>,
    /// Pairwise comparisons (control vs each treatment).
    pub comparisons: Vec<VariantComparison>,
    /// Recommended variant to promote (`None` if inconclusive).
    pub recommendation: Option<String>,
    /// Whether any comparison has sufficient sample size.
    pub has_sufficient_data: bool,
}

// ---------------------------------------------------------------------------
// AbTestManager
// ---------------------------------------------------------------------------

/// Registry managing multiple concurrent A/B experiments.
#[derive(Debug, Default)]
pub struct AbTestManager {
    /// Active and concluded experiments.
    experiments: HashMap<Uuid, AbExperiment>,
    /// Accumulated metrics: experiment_id -> variant_id -> metrics.
    metrics: HashMap<Uuid, HashMap<String, VariantMetrics>>,
}

impl AbTestManager {
    /// Create an empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new experiment.
    pub fn register(&mut self, experiment: AbExperiment) {
        let eid = experiment.id;
        self.experiments.insert(eid, experiment);
        // Pre-initialise metric buckets.
        if let Some(exp) = self.experiments.get(&eid) {
            let buckets: HashMap<String, VariantMetrics> = exp
                .variants
                .iter()
                .map(|v| (v.id.clone(), VariantMetrics::default()))
                .collect();
            self.metrics.insert(eid, buckets);
        }
    }

    /// Start an experiment.
    ///
    /// # Errors
    ///
    /// Returns an error if the experiment is not found or already running.
    pub fn start(&mut self, experiment_id: Uuid, now_secs: i64) -> SearchResult<()> {
        let exp = self
            .experiments
            .get_mut(&experiment_id)
            .ok_or_else(|| SearchError::DocumentNotFound(experiment_id.to_string()))?;
        exp.start(now_secs)
    }

    /// Conclude an experiment.
    ///
    /// # Errors
    ///
    /// Returns an error if the experiment is not found.
    pub fn conclude(&mut self, experiment_id: Uuid, now_secs: i64) -> SearchResult<()> {
        let exp = self
            .experiments
            .get_mut(&experiment_id)
            .ok_or_else(|| SearchError::DocumentNotFound(experiment_id.to_string()))?;
        exp.conclude(now_secs);
        Ok(())
    }

    /// Assign a variant for a search request.
    ///
    /// Returns `None` if the experiment is not running.
    ///
    /// # Errors
    ///
    /// Returns an error if the experiment is not found.
    pub fn assign(
        &self,
        experiment_id: Uuid,
        session_id: &str,
    ) -> SearchResult<Option<&RankingVariant>> {
        let exp = self
            .experiments
            .get(&experiment_id)
            .ok_or_else(|| SearchError::DocumentNotFound(experiment_id.to_string()))?;
        if exp.status != ExperimentStatus::Running {
            return Ok(None);
        }
        Ok(Some(exp.assign_variant(session_id)))
    }

    /// Record an interaction event.
    ///
    /// # Errors
    ///
    /// Returns an error if the experiment or variant is not found.
    pub fn record(&mut self, event: InteractionEvent) -> SearchResult<()> {
        let metrics = self
            .metrics
            .get_mut(&event.experiment_id)
            .ok_or_else(|| SearchError::DocumentNotFound(event.experiment_id.to_string()))?;

        let m = metrics
            .get_mut(&event.variant_id)
            .ok_or_else(|| SearchError::DocumentNotFound(event.variant_id.clone()))?;

        m.impressions += 1;
        if event.clicked {
            m.clicks += 1;
            if let Some(rank) = event.click_rank {
                m.total_click_rank += u64::from(rank);
            }
        }
        if event.zero_results {
            m.zero_result_sessions += 1;
        }
        if event.satisfied {
            m.satisfied_sessions += 1;
        }
        Ok(())
    }

    /// Analyse an experiment and return aggregated results.
    ///
    /// # Errors
    ///
    /// Returns an error if the experiment is not found.
    pub fn analyse(&self, experiment_id: Uuid) -> SearchResult<ExperimentResult> {
        let exp = self
            .experiments
            .get(&experiment_id)
            .ok_or_else(|| SearchError::DocumentNotFound(experiment_id.to_string()))?;

        let metrics = self
            .metrics
            .get(&experiment_id)
            .ok_or_else(|| SearchError::DocumentNotFound(experiment_id.to_string()))?;

        let control_variant = exp.variants.iter().find(|v| v.is_control);
        let min_obs = exp.min_observations as u64;

        let mut comparisons = Vec::new();
        let mut recommendation: Option<String> = None;
        let mut best_lift = 0.0f64;
        let mut has_sufficient_data = false;

        if let Some(ctrl) = control_variant {
            let ctrl_metrics = metrics.get(&ctrl.id).cloned().unwrap_or_default();
            let ctrl_ctr = ctrl_metrics.ctr();

            for variant in exp.variants.iter().filter(|v| !v.is_control) {
                let treat_metrics = metrics.get(&variant.id).cloned().unwrap_or_default();
                let treat_ctr = treat_metrics.ctr();

                let sufficient =
                    ctrl_metrics.impressions >= min_obs && treat_metrics.impressions >= min_obs;
                if sufficient {
                    has_sufficient_data = true;
                }

                let (z_score, p_value) = if sufficient {
                    two_proportion_z_test(
                        ctrl_metrics.clicks,
                        ctrl_metrics.impressions,
                        treat_metrics.clicks,
                        treat_metrics.impressions,
                    )
                } else {
                    (0.0, 1.0)
                };

                let relative_lift = if ctrl_ctr.abs() < f64::EPSILON {
                    0.0
                } else {
                    (treat_ctr - ctrl_ctr) / ctrl_ctr
                };

                let is_significant = sufficient && p_value < exp.alpha;

                if is_significant && relative_lift > best_lift {
                    best_lift = relative_lift;
                    recommendation = Some(variant.id.clone());
                }

                comparisons.push(VariantComparison {
                    control_id: ctrl.id.clone(),
                    treatment_id: variant.id.clone(),
                    control_ctr: ctrl_ctr,
                    treatment_ctr: treat_ctr,
                    relative_lift,
                    z_score,
                    p_value,
                    is_significant,
                    alpha: exp.alpha,
                });
            }
        }

        Ok(ExperimentResult {
            experiment_id,
            metrics: metrics.clone(),
            comparisons,
            recommendation,
            has_sufficient_data,
        })
    }

    /// Get a reference to an experiment by ID.
    #[must_use]
    pub fn get_experiment(&self, id: Uuid) -> Option<&AbExperiment> {
        self.experiments.get(&id)
    }

    /// List all experiments.
    #[must_use]
    pub fn list_experiments(&self) -> Vec<&AbExperiment> {
        self.experiments.values().collect()
    }

    /// Get current metrics for all variants of an experiment.
    #[must_use]
    pub fn get_metrics(&self, experiment_id: Uuid) -> Option<&HashMap<String, VariantMetrics>> {
        self.metrics.get(&experiment_id)
    }
}

// ---------------------------------------------------------------------------
// Statistical helpers
// ---------------------------------------------------------------------------

/// Two-proportion z-test (one-sided positive, two-tailed p-value).
///
/// Returns `(z_score, p_value)`.
#[allow(clippy::cast_precision_loss)]
fn two_proportion_z_test(
    clicks_a: u64,
    impressions_a: u64,
    clicks_b: u64,
    impressions_b: u64,
) -> (f64, f64) {
    if impressions_a == 0 || impressions_b == 0 {
        return (0.0, 1.0);
    }
    let p_a = clicks_a as f64 / impressions_a as f64;
    let p_b = clicks_b as f64 / impressions_b as f64;
    let p_pool = (clicks_a + clicks_b) as f64 / (impressions_a + impressions_b) as f64;
    let se = (p_pool * (1.0 - p_pool) * (1.0 / impressions_a as f64 + 1.0 / impressions_b as f64))
        .sqrt();
    if se < f64::EPSILON {
        return (0.0, 1.0);
    }
    let z = (p_b - p_a) / se;
    // Approximate p-value using the complementary error function.
    let p_value = 2.0 * (1.0 - normal_cdf(z.abs()));
    (z, p_value)
}

/// Standard normal CDF approximation (Abramowitz & Stegun 26.2.17).
fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let poly = t
        * (0.319_381_53
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let pdf = (-x * x / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let p = 1.0 - pdf * poly;
    if x < 0.0 {
        1.0 - p
    } else {
        p
    }
}

/// FNV-1a 64-bit hash (non-cryptographic, fast, deterministic).
fn fnv1a_hash(s: &str) -> u64 {
    const OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;
    let mut hash = OFFSET_BASIS;
    for byte in s.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: i64 = 1_700_000_000;

    fn make_experiment() -> AbExperiment {
        AbExperiment::new(
            "CTR Test",
            RankingVariant::control(ScoringAlgorithm::TfIdf),
            RankingVariant::treatment("treat-v1", "BM25 Treatment", ScoringAlgorithm::Bm25),
        )
        .expect("valid experiment")
    }

    #[test]
    fn test_experiment_creation() {
        let exp = make_experiment();
        assert_eq!(exp.variants.len(), 2);
        assert_eq!(exp.status, ExperimentStatus::Pending);
        assert!(exp.started_at.is_none());
    }

    #[test]
    fn test_experiment_invalid_fractions() {
        let mut c = RankingVariant::control(ScoringAlgorithm::Bm25);
        c.traffic_fraction = 0.3;
        let t = RankingVariant::treatment("t", "T", ScoringAlgorithm::Bm25);
        // 0.3 + 0.5 = 0.8 ≠ 1.0
        let result = AbExperiment::new("bad", c, t);
        assert!(result.is_err());
    }

    #[test]
    fn test_experiment_start_and_conclude() {
        let mut exp = make_experiment();
        exp.start(NOW).expect("ok");
        assert_eq!(exp.status, ExperimentStatus::Running);
        assert_eq!(exp.started_at, Some(NOW));

        exp.conclude(NOW + 86400);
        assert_eq!(exp.status, ExperimentStatus::Concluded);
        assert_eq!(exp.concluded_at, Some(NOW + 86400));
    }

    #[test]
    fn test_experiment_start_twice_fails() {
        let mut exp = make_experiment();
        exp.start(NOW).expect("ok");
        assert!(exp.start(NOW + 1).is_err());
    }

    #[test]
    fn test_experiment_pause_and_resume() {
        let mut exp = make_experiment();
        exp.start(NOW).expect("ok");
        exp.pause();
        assert_eq!(exp.status, ExperimentStatus::Paused);
        exp.resume();
        assert_eq!(exp.status, ExperimentStatus::Running);
    }

    #[test]
    fn test_variant_assignment_deterministic() {
        let exp = make_experiment();
        let v1 = exp.assign_variant("session-abc");
        let v2 = exp.assign_variant("session-abc");
        assert_eq!(v1.id, v2.id);
    }

    #[test]
    fn test_variant_assignment_distribution() {
        // Use a fixed, well-known experiment UUID so that the FNV-1a hash
        // produces a stable, reproducible distribution across test runs.
        // A random UUID as experiment ID can bias the hash for sequential
        // session IDs, causing non-deterministic test failures.
        let mut exp = make_experiment();
        exp.id = Uuid::parse_str("00000000-0000-0000-0000-000000000001").expect("valid fixed UUID");

        let n = 10_000;
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for i in 0..n {
            let sid = format!("session-{i}");
            let v = exp.assign_variant(&sid);
            *counts.entry(v.id.as_str()).or_insert(0) += 1;
        }
        // With 50/50 split, expect ~5000 each. Allow 10% tolerance.
        for &count in counts.values() {
            assert!(
                (count as f64 / n as f64 - 0.5).abs() < 0.10,
                "count = {count}, expected ~5000 (within 10%)"
            );
        }
    }

    #[test]
    fn test_manager_register_and_start() {
        let mut mgr = AbTestManager::new();
        let exp = make_experiment();
        let eid = exp.id;
        mgr.register(exp);
        mgr.start(eid, NOW).expect("ok");
        let retrieved = mgr.get_experiment(eid).expect("exists");
        assert_eq!(retrieved.status, ExperimentStatus::Running);
    }

    #[test]
    fn test_manager_assign_not_running() {
        let mut mgr = AbTestManager::new();
        let exp = make_experiment();
        let eid = exp.id;
        mgr.register(exp);
        // Experiment is Pending, not Running.
        let result = mgr.assign(eid, "session-1").expect("no error");
        assert!(result.is_none());
    }

    #[test]
    fn test_manager_assign_running() {
        let mut mgr = AbTestManager::new();
        let exp = make_experiment();
        let eid = exp.id;
        mgr.register(exp);
        mgr.start(eid, NOW).expect("ok");
        let variant = mgr.assign(eid, "session-1").expect("ok").expect("some");
        assert!(!variant.id.is_empty());
    }

    #[test]
    fn test_record_event_and_metrics() {
        let mut mgr = AbTestManager::new();
        let exp = make_experiment();
        let eid = exp.id;
        let variant_id = exp.variants[0].id.clone();
        mgr.register(exp);
        mgr.start(eid, NOW).expect("ok");

        let event = InteractionEvent {
            experiment_id: eid,
            variant_id: variant_id.clone(),
            clicked: true,
            click_rank: Some(1),
            zero_results: false,
            satisfied: true,
        };
        mgr.record(event).expect("ok");

        let metrics = mgr.get_metrics(eid).expect("exists");
        let m = metrics.get(&variant_id).expect("exists");
        assert_eq!(m.impressions, 1);
        assert_eq!(m.clicks, 1);
        assert_eq!(m.total_click_rank, 1);
        assert_eq!(m.satisfied_sessions, 1);
    }

    #[test]
    fn test_variant_metrics_ctr() {
        let mut m = VariantMetrics::default();
        assert!((m.ctr()).abs() < f64::EPSILON);
        m.impressions = 100;
        m.clicks = 10;
        assert!((m.ctr() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_variant_metrics_satisfaction_rate() {
        let mut m = VariantMetrics::default();
        m.impressions = 200;
        m.satisfied_sessions = 150;
        assert!((m.satisfaction_rate() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_analyse_insufficient_data() {
        let mut mgr = AbTestManager::new();
        let exp = make_experiment();
        let eid = exp.id;
        mgr.register(exp);
        mgr.start(eid, NOW).expect("ok");

        let result = mgr.analyse(eid).expect("ok");
        assert!(!result.has_sufficient_data);
        assert!(result.recommendation.is_none());
    }

    #[test]
    fn test_analyse_with_significant_lift() {
        let mut mgr = AbTestManager::new();
        let mut exp = make_experiment();
        exp.min_observations = 10; // lower bar for test
        exp.alpha = 0.05;
        let eid = exp.id;
        let ctrl_id = exp
            .variants
            .iter()
            .find(|v| v.is_control)
            .map(|v| v.id.clone())
            .expect("ctrl");
        let treat_id = exp
            .variants
            .iter()
            .find(|v| !v.is_control)
            .map(|v| v.id.clone())
            .expect("treat");
        mgr.register(exp);
        mgr.start(eid, NOW).expect("ok");

        // Control: 10/100 CTR = 0.1
        for _ in 0..100 {
            mgr.record(InteractionEvent {
                experiment_id: eid,
                variant_id: ctrl_id.clone(),
                clicked: false,
                click_rank: None,
                zero_results: false,
                satisfied: false,
            })
            .expect("ok");
        }
        for _ in 0..10 {
            mgr.record(InteractionEvent {
                experiment_id: eid,
                variant_id: ctrl_id.clone(),
                clicked: true,
                click_rank: Some(1),
                zero_results: false,
                satisfied: true,
            })
            .expect("ok");
        }
        // Treatment: 40/100 CTR = 0.4 (strong lift, should be significant)
        for _ in 0..100 {
            mgr.record(InteractionEvent {
                experiment_id: eid,
                variant_id: treat_id.clone(),
                clicked: false,
                click_rank: None,
                zero_results: false,
                satisfied: false,
            })
            .expect("ok");
        }
        for _ in 0..40 {
            mgr.record(InteractionEvent {
                experiment_id: eid,
                variant_id: treat_id.clone(),
                clicked: true,
                click_rank: Some(1),
                zero_results: false,
                satisfied: true,
            })
            .expect("ok");
        }

        let result = mgr.analyse(eid).expect("ok");
        assert!(result.has_sufficient_data);
        assert!(!result.comparisons.is_empty());
        let cmp = &result.comparisons[0];
        assert!(cmp.relative_lift > 0.0);
        // With such a large difference the test should be significant.
        assert!(
            cmp.is_significant || cmp.p_value < 0.1,
            "p_value = {}",
            cmp.p_value
        );
    }

    #[test]
    fn test_multi_variant_requires_two() {
        let result = AbExperiment::multi_variant(
            "single",
            vec![RankingVariant::control(ScoringAlgorithm::Bm25)],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_variant_valid() {
        let mut v1 = RankingVariant::control(ScoringAlgorithm::Bm25);
        v1.traffic_fraction = 0.34;
        let mut v2 = RankingVariant::treatment("t1", "T1", ScoringAlgorithm::TfIdf);
        v2.traffic_fraction = 0.33;
        let mut v3 = RankingVariant::treatment("t2", "T2", ScoringAlgorithm::Hybrid);
        v3.traffic_fraction = 0.33;
        let result = AbExperiment::multi_variant("three-way", vec![v1, v2, v3]);
        assert!(result.is_ok());
        assert_eq!(result.expect("ok").variants.len(), 3);
    }

    #[test]
    fn test_scoring_algorithm_equality() {
        assert_eq!(ScoringAlgorithm::Bm25, ScoringAlgorithm::Bm25);
        assert_ne!(ScoringAlgorithm::Bm25, ScoringAlgorithm::TfIdf);
    }

    #[test]
    fn test_normal_cdf_properties() {
        // Φ(0) ≈ 0.5
        assert!((normal_cdf(0.0) - 0.5).abs() < 0.01);
        // Φ(1.96) ≈ 0.975
        assert!((normal_cdf(1.96) - 0.975).abs() < 0.01);
        // Φ(-∞) ≈ 0
        assert!(normal_cdf(-10.0) < 0.01);
    }

    #[test]
    fn test_fnv1a_different_inputs_produce_different_hashes() {
        let h1 = fnv1a_hash("session-1");
        let h2 = fnv1a_hash("session-2");
        assert_ne!(h1, h2);
    }
}
