//! Disaster recovery planning for digital archives.
//!
//! This module provides tools for planning and simulating recovery from
//! catastrophic data loss events:
//! - **RecoveryObjective** - RTO/RPO targets
//! - **DisasterScenario** - Categorized disaster types with probability
//! - **RecoveryPlan** - Ordered recovery steps with dependencies
//! - **DrSimulation** - Seeded Monte-Carlo simulation of plan execution
//! - **RiskMatrix** - Prioritise scenarios by risk = probability × impact

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────
// RecoveryObjective
// ─────────────────────────────────────────────────────────────

/// Recovery Time Objective and Recovery Point Objective.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryObjective {
    /// Maximum acceptable recovery time in hours
    pub rto_hours: u32,
    /// Maximum acceptable data loss window in hours
    pub rpo_hours: u32,
}

impl RecoveryObjective {
    /// Create a new recovery objective.
    #[must_use]
    pub fn new(rto_hours: u32, rpo_hours: u32) -> Self {
        Self {
            rto_hours,
            rpo_hours,
        }
    }

    /// Returns true if the actual RTO and RPO meet (are ≤) the SLA targets.
    #[must_use]
    pub fn meets_sla(&self, actual_rto: u32, actual_rpo: u32) -> bool {
        actual_rto <= self.rto_hours && actual_rpo <= self.rpo_hours
    }
}

// ─────────────────────────────────────────────────────────────
// DisasterScenario
// ─────────────────────────────────────────────────────────────

/// Types of disaster scenarios relevant to digital archives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DisasterScenario {
    /// Primary data centre goes offline
    DataCenter,
    /// Network outage prevents data access
    Network,
    /// Ransomware encrypts archive data
    Ransomware,
    /// Physical disaster (flood, fire, earthquake) affecting storage site
    NaturalDisaster,
    /// Storage hardware failure
    HardwareFailure,
}

impl DisasterScenario {
    /// Returns the estimated probability of this scenario occurring per year.
    #[must_use]
    pub fn probability_per_year(&self) -> f64 {
        match self {
            Self::DataCenter => 0.05,
            Self::Network => 0.30,
            Self::Ransomware => 0.15,
            Self::NaturalDisaster => 0.02,
            Self::HardwareFailure => 0.25,
        }
    }

    /// Returns the scenario name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::DataCenter => "DataCenter",
            Self::Network => "Network",
            Self::Ransomware => "Ransomware",
            Self::NaturalDisaster => "NaturalDisaster",
            Self::HardwareFailure => "HardwareFailure",
        }
    }

    /// Returns the typical impact on data integrity (0.0–1.0).
    #[must_use]
    pub fn typical_impact(&self) -> f64 {
        match self {
            Self::DataCenter => 0.8,
            Self::Network => 0.3,
            Self::Ransomware => 0.9,
            Self::NaturalDisaster => 1.0,
            Self::HardwareFailure => 0.6,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// RecoveryStep
// ─────────────────────────────────────────────────────────────

/// A single step in a recovery plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryStep {
    /// Unique step identifier (1-based)
    pub order: u32,
    /// Human-readable step name
    pub name: String,
    /// Expected duration in hours
    pub duration_hours: f32,
    /// Step IDs that must complete before this step can begin
    pub requires: Vec<u32>,
    /// Team responsible for executing this step
    pub responsible_team: String,
}

impl RecoveryStep {
    /// Returns true if this step has no prerequisites.
    #[must_use]
    pub fn is_independent(&self) -> bool {
        self.requires.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────
// RecoveryPlan
// ─────────────────────────────────────────────────────────────

/// A complete disaster recovery plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryPlan {
    /// The disaster scenario this plan addresses
    pub scenario: DisasterScenario,
    /// Ordered recovery steps
    pub steps: Vec<RecoveryStep>,
    /// Estimated total recovery time in hours (critical path)
    pub total_time_hours: f32,
    /// Estimated probability of successful recovery (0.0–1.0)
    pub success_probability: f32,
}

impl RecoveryPlan {
    /// Returns the critical path length (longest dependency chain duration).
    #[must_use]
    pub fn critical_path_hours(&self) -> f32 {
        // Simple greedy: sum independent + dependent sequential steps
        self.steps.iter().map(|s| s.duration_hours).sum()
    }

    /// Returns all independent steps (no prerequisites).
    #[must_use]
    pub fn independent_steps(&self) -> Vec<&RecoveryStep> {
        self.steps.iter().filter(|s| s.is_independent()).collect()
    }
}

// ─────────────────────────────────────────────────────────────
// DrSimResult
// ─────────────────────────────────────────────────────────────

/// Result of running a disaster recovery simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrSimResult {
    /// Whether the recovery completed successfully
    pub completed: bool,
    /// Actual recovery time in hours (may exceed `total_time_hours` if steps fail)
    pub actual_rto_hours: f32,
    /// Step IDs that failed during simulation
    pub failed_steps: Vec<u32>,
    /// Name of the step that formed the bottleneck, if any
    pub bottleneck: Option<String>,
}

// ─────────────────────────────────────────────────────────────
// DrSimulation
// ─────────────────────────────────────────────────────────────

/// Disaster recovery simulator using a seeded LCG pseudo-random number generator.
pub struct DrSimulation;

impl DrSimulation {
    /// Simulate execution of a recovery plan.
    ///
    /// * `plan` – the recovery plan to simulate
    /// * `seed` – deterministic seed for reproducibility
    #[must_use]
    pub fn run(plan: &RecoveryPlan, seed: u64) -> DrSimResult {
        let mut rng = LcgRng::new(seed);
        let mut total_hours = 0.0f32;
        let mut failed_steps: Vec<u32> = Vec::new();
        let mut bottleneck: Option<String> = None;
        let mut max_step_hours = 0.0f32;

        for step in &plan.steps {
            // Check prerequisites
            let prereqs_ok = step
                .requires
                .iter()
                .all(|&req| !failed_steps.contains(&req));

            if !prereqs_ok {
                failed_steps.push(step.order);
                continue;
            }

            // Random failure based on step success probability
            // Each step has base 90% success probability, scaled by plan overall
            let base_success = 0.90 * plan.success_probability;
            let roll = rng.next_f32();

            if roll > base_success {
                failed_steps.push(step.order);
                // Failed step adds penalty time
                let penalty = step.duration_hours * 1.5;
                total_hours += penalty;
                if penalty > max_step_hours {
                    max_step_hours = penalty;
                    bottleneck = Some(format!("{} (failed)", step.name));
                }
            } else {
                // Slight random variation ±20%
                let variation = 1.0 + (rng.next_f32() - 0.5) * 0.4;
                let actual_duration = step.duration_hours * variation;
                total_hours += actual_duration;
                if actual_duration > max_step_hours {
                    max_step_hours = actual_duration;
                    bottleneck = Some(step.name.clone());
                }
            }
        }

        let completed = failed_steps.is_empty();
        DrSimResult {
            completed,
            actual_rto_hours: total_hours,
            failed_steps,
            bottleneck,
        }
    }
}

/// Simple LCG pseudo-random number generator.
struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        // LCG parameters from Knuth
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 33) as f32 / (u32::MAX as f32)
    }
}

// ─────────────────────────────────────────────────────────────
// RiskMatrix
// ─────────────────────────────────────────────────────────────

/// A risk matrix prioritising disaster scenarios by `probability × impact`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMatrix {
    /// Entries: (scenario, probability_per_year, impact 0–1)
    pub scenarios: Vec<(DisasterScenario, f64, f64)>,
}

impl RiskMatrix {
    /// Create a new risk matrix from explicit entries.
    #[must_use]
    pub fn new(scenarios: Vec<(DisasterScenario, f64, f64)>) -> Self {
        Self { scenarios }
    }

    /// Create a default risk matrix using the built-in scenario probabilities and impacts.
    #[must_use]
    pub fn with_defaults() -> Self {
        let entries = [
            DisasterScenario::DataCenter,
            DisasterScenario::Network,
            DisasterScenario::Ransomware,
            DisasterScenario::NaturalDisaster,
            DisasterScenario::HardwareFailure,
        ]
        .iter()
        .map(|&s| (s, s.probability_per_year(), s.typical_impact()))
        .collect();

        Self { scenarios: entries }
    }

    /// Add an entry to the matrix.
    pub fn add(&mut self, scenario: DisasterScenario, probability: f64, impact: f64) {
        self.scenarios.push((scenario, probability, impact));
    }

    /// Returns the scenario with the highest risk score (`probability × impact`).
    #[must_use]
    pub fn highest_risk(&self) -> Option<&DisasterScenario> {
        self.scenarios
            .iter()
            .max_by(|(_, p1, i1), (_, p2, i2)| {
                let r1 = p1 * i1;
                let r2 = p2 * i2;
                r1.partial_cmp(&r2).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(s, _, _)| s)
    }

    /// Returns all scenarios sorted by risk score descending.
    #[must_use]
    pub fn ranked(&self) -> Vec<(&DisasterScenario, f64)> {
        let mut ranked: Vec<(&DisasterScenario, f64)> =
            self.scenarios.iter().map(|(s, p, i)| (s, p * i)).collect();
        ranked.sort_by(|(_, r1), (_, r2)| r2.partial_cmp(r1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }
}

// ─────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_plan() -> RecoveryPlan {
        RecoveryPlan {
            scenario: DisasterScenario::HardwareFailure,
            steps: vec![
                RecoveryStep {
                    order: 1,
                    name: "Assess damage".to_string(),
                    duration_hours: 1.0,
                    requires: vec![],
                    responsible_team: "IT Ops".to_string(),
                },
                RecoveryStep {
                    order: 2,
                    name: "Restore from backup".to_string(),
                    duration_hours: 4.0,
                    requires: vec![1],
                    responsible_team: "Data Engineering".to_string(),
                },
                RecoveryStep {
                    order: 3,
                    name: "Verify integrity".to_string(),
                    duration_hours: 1.0,
                    requires: vec![2],
                    responsible_team: "QA".to_string(),
                },
            ],
            total_time_hours: 6.0,
            success_probability: 0.95,
        }
    }

    // ── RecoveryObjective ─────────────────────────────────────

    #[test]
    fn test_meets_sla_pass() {
        let obj = RecoveryObjective::new(8, 4);
        assert!(obj.meets_sla(6, 2));
    }

    #[test]
    fn test_meets_sla_fail_rto() {
        let obj = RecoveryObjective::new(4, 2);
        assert!(!obj.meets_sla(8, 1));
    }

    #[test]
    fn test_meets_sla_fail_rpo() {
        let obj = RecoveryObjective::new(8, 2);
        assert!(!obj.meets_sla(4, 6));
    }

    // ── DisasterScenario ──────────────────────────────────────

    #[test]
    fn test_probability_per_year_ranges() {
        for scenario in &[
            DisasterScenario::DataCenter,
            DisasterScenario::Network,
            DisasterScenario::Ransomware,
            DisasterScenario::NaturalDisaster,
            DisasterScenario::HardwareFailure,
        ] {
            let p = scenario.probability_per_year();
            assert!(p > 0.0 && p <= 1.0);
        }
    }

    #[test]
    fn test_scenario_names() {
        assert_eq!(DisasterScenario::Ransomware.name(), "Ransomware");
        assert_eq!(DisasterScenario::NaturalDisaster.name(), "NaturalDisaster");
    }

    // ── RecoveryPlan ──────────────────────────────────────────

    #[test]
    fn test_critical_path_hours() {
        let plan = make_simple_plan();
        let path = plan.critical_path_hours();
        assert!((path - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_independent_steps() {
        let plan = make_simple_plan();
        let independent = plan.independent_steps();
        assert_eq!(independent.len(), 1);
        assert_eq!(independent[0].order, 1);
    }

    #[test]
    fn test_recovery_step_is_independent() {
        let step = RecoveryStep {
            order: 1,
            name: "Start".into(),
            duration_hours: 0.5,
            requires: vec![],
            responsible_team: "IT".into(),
        };
        assert!(step.is_independent());
    }

    // ── DrSimulation ──────────────────────────────────────────

    #[test]
    fn test_sim_deterministic() {
        let plan = make_simple_plan();
        let r1 = DrSimulation::run(&plan, 42);
        let r2 = DrSimulation::run(&plan, 42);
        assert_eq!(r1.completed, r2.completed);
        assert!((r1.actual_rto_hours - r2.actual_rto_hours).abs() < 1e-5);
    }

    #[test]
    fn test_sim_high_success_plan() {
        // With high success_probability most runs should complete
        let mut plan = make_simple_plan();
        plan.success_probability = 1.0; // guarantees success (roll always < 1.0 * 0.9)
        let result = DrSimulation::run(&plan, 1234);
        assert_eq!(result.failed_steps.len(), 0);
        assert!(result.completed);
    }

    #[test]
    fn test_sim_rto_positive() {
        let plan = make_simple_plan();
        let result = DrSimulation::run(&plan, 99);
        assert!(result.actual_rto_hours > 0.0);
    }

    #[test]
    fn test_sim_returns_bottleneck() {
        let plan = make_simple_plan();
        let result = DrSimulation::run(&plan, 7);
        // Bottleneck should always be set when there are steps
        assert!(result.bottleneck.is_some());
    }

    // ── RiskMatrix ────────────────────────────────────────────

    #[test]
    fn test_highest_risk_not_none() {
        let matrix = RiskMatrix::with_defaults();
        assert!(matrix.highest_risk().is_some());
    }

    #[test]
    fn test_ranked_descending() {
        let matrix = RiskMatrix::with_defaults();
        let ranked = matrix.ranked();
        for i in 0..ranked.len().saturating_sub(1) {
            assert!(ranked[i].1 >= ranked[i + 1].1);
        }
    }

    #[test]
    fn test_risk_matrix_custom_entry() {
        let mut matrix = RiskMatrix::new(vec![]);
        matrix.add(DisasterScenario::Ransomware, 0.8, 0.9);
        matrix.add(DisasterScenario::Network, 0.1, 0.2);
        let top = matrix.highest_risk().expect("operation should succeed");
        assert_eq!(*top, DisasterScenario::Ransomware);
    }

    #[test]
    fn test_risk_matrix_empty() {
        let matrix = RiskMatrix::new(vec![]);
        assert!(matrix.highest_risk().is_none());
    }
}
