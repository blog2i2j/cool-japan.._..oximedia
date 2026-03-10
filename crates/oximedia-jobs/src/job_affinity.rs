#![allow(dead_code)]
//! Worker affinity and anti-affinity rules for job placement.
//!
//! Controls which workers a job prefers (affinity) or avoids (anti-affinity),
//! based on labels, resource capabilities, and historical performance.

use std::collections::{HashMap, HashSet};

/// Kind of affinity rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AffinityKind {
    /// Job should prefer workers matching this rule (soft).
    Preferred,
    /// Job must run on workers matching this rule (hard).
    Required,
    /// Job should avoid workers matching this rule (soft).
    PreferredAntiAffinity,
    /// Job must not run on workers matching this rule (hard).
    RequiredAntiAffinity,
}

impl std::fmt::Display for AffinityKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Preferred => write!(f, "Preferred"),
            Self::Required => write!(f, "Required"),
            Self::PreferredAntiAffinity => write!(f, "PreferredAntiAffinity"),
            Self::RequiredAntiAffinity => write!(f, "RequiredAntiAffinity"),
        }
    }
}

/// A single affinity rule matching on worker labels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffinityRule {
    /// The kind of affinity.
    pub kind: AffinityKind,
    /// Label key to match.
    pub label_key: String,
    /// Acceptable values for the label (any match is a hit).
    pub label_values: HashSet<String>,
    /// Relative weight for soft rules (higher = stronger preference).
    pub weight: u32,
}

impl AffinityRule {
    /// Create a new affinity rule.
    pub fn new(kind: AffinityKind, label_key: &str) -> Self {
        Self {
            kind,
            label_key: label_key.to_string(),
            label_values: HashSet::new(),
            weight: 1,
        }
    }

    /// Add an acceptable label value.
    pub fn with_value(mut self, value: &str) -> Self {
        self.label_values.insert(value.to_string());
        self
    }

    /// Set the weight.
    pub fn with_weight(mut self, weight: u32) -> Self {
        self.weight = weight;
        self
    }

    /// Check if a set of worker labels matches this rule.
    pub fn matches(&self, worker_labels: &HashMap<String, String>) -> bool {
        if let Some(val) = worker_labels.get(&self.label_key) {
            if self.label_values.is_empty() {
                // Key presence is sufficient
                true
            } else {
                self.label_values.contains(val)
            }
        } else {
            false
        }
    }

    /// Whether this rule is a hard constraint (Required / RequiredAntiAffinity).
    pub fn is_hard(&self) -> bool {
        matches!(
            self.kind,
            AffinityKind::Required | AffinityKind::RequiredAntiAffinity
        )
    }
}

/// A worker description used for matching.
#[derive(Debug, Clone)]
pub struct WorkerDescriptor {
    /// Worker identifier.
    pub id: String,
    /// Labels assigned to this worker.
    pub labels: HashMap<String, String>,
    /// Current load (0.0 = idle, 1.0 = fully loaded).
    pub load: f64,
}

impl WorkerDescriptor {
    /// Create a new worker descriptor.
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            labels: HashMap::new(),
            load: 0.0,
        }
    }

    /// Add a label.
    pub fn with_label(mut self, key: &str, value: &str) -> Self {
        self.labels.insert(key.to_string(), value.to_string());
        self
    }

    /// Set the load.
    pub fn with_load(mut self, load: f64) -> Self {
        self.load = load;
        self
    }
}

/// Affinity specification for a job.
#[derive(Debug, Clone, Default)]
pub struct AffinitySpec {
    /// Rules applied to this job.
    pub rules: Vec<AffinityRule>,
}

impl AffinitySpec {
    /// Create an empty affinity spec.
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Add a rule.
    pub fn add_rule(mut self, rule: AffinityRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Check whether a worker satisfies all hard constraints.
    pub fn satisfies_hard_constraints(&self, worker: &WorkerDescriptor) -> bool {
        for rule in &self.rules {
            match rule.kind {
                AffinityKind::Required => {
                    if !rule.matches(&worker.labels) {
                        return false;
                    }
                }
                AffinityKind::RequiredAntiAffinity => {
                    if rule.matches(&worker.labels) {
                        return false;
                    }
                }
                _ => {}
            }
        }
        true
    }

    /// Compute a preference score for a worker (higher = better).
    #[allow(clippy::cast_precision_loss)]
    pub fn preference_score(&self, worker: &WorkerDescriptor) -> f64 {
        let mut score = 0.0_f64;
        for rule in &self.rules {
            let matched = rule.matches(&worker.labels);
            match rule.kind {
                AffinityKind::Preferred => {
                    if matched {
                        score += rule.weight as f64;
                    }
                }
                AffinityKind::PreferredAntiAffinity => {
                    if matched {
                        score -= rule.weight as f64;
                    }
                }
                _ => {}
            }
        }
        score
    }
}

/// Matcher that selects the best worker for a job from a pool.
#[derive(Debug)]
pub struct AffinityMatcher {
    /// Available workers.
    pub workers: Vec<WorkerDescriptor>,
}

impl AffinityMatcher {
    /// Create a new matcher with a pool of workers.
    pub fn new(workers: Vec<WorkerDescriptor>) -> Self {
        Self { workers }
    }

    /// Filter workers that satisfy all hard constraints.
    pub fn eligible_workers(&self, spec: &AffinitySpec) -> Vec<&WorkerDescriptor> {
        self.workers
            .iter()
            .filter(|w| spec.satisfies_hard_constraints(w))
            .collect()
    }

    /// Select the best worker based on affinity scoring.
    pub fn select_best(&self, spec: &AffinitySpec) -> Option<&WorkerDescriptor> {
        self.eligible_workers(spec).into_iter().max_by(|a, b| {
            spec.preference_score(a)
                .partial_cmp(&spec.preference_score(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Rank all eligible workers by preference score (descending).
    pub fn rank_workers(&self, spec: &AffinitySpec) -> Vec<(&WorkerDescriptor, f64)> {
        let mut scored: Vec<(&WorkerDescriptor, f64)> = self
            .eligible_workers(spec)
            .into_iter()
            .map(|w| (w, spec.preference_score(w)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    /// Count how many workers are eligible.
    pub fn eligible_count(&self, spec: &AffinitySpec) -> usize {
        self.eligible_workers(spec).len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu_worker() -> WorkerDescriptor {
        WorkerDescriptor::new("gpu-1")
            .with_label("gpu", "nvidia")
            .with_label("region", "us-east")
            .with_load(0.5)
    }

    fn cpu_worker() -> WorkerDescriptor {
        WorkerDescriptor::new("cpu-1")
            .with_label("gpu", "none")
            .with_label("region", "us-west")
            .with_load(0.2)
    }

    fn eu_worker() -> WorkerDescriptor {
        WorkerDescriptor::new("eu-1")
            .with_label("gpu", "nvidia")
            .with_label("region", "eu-west")
            .with_load(0.8)
    }

    #[test]
    fn test_affinity_rule_matches() {
        let rule = AffinityRule::new(AffinityKind::Required, "gpu").with_value("nvidia");
        let worker = gpu_worker();
        assert!(rule.matches(&worker.labels));
    }

    #[test]
    fn test_affinity_rule_no_match() {
        let rule = AffinityRule::new(AffinityKind::Required, "gpu").with_value("nvidia");
        let worker = cpu_worker();
        assert!(!rule.matches(&worker.labels));
    }

    #[test]
    fn test_affinity_rule_key_presence_only() {
        let rule = AffinityRule::new(AffinityKind::Required, "gpu");
        let worker = gpu_worker();
        assert!(rule.matches(&worker.labels));
        let worker2 = cpu_worker();
        assert!(rule.matches(&worker2.labels)); // "gpu" key exists with value "none"
    }

    #[test]
    fn test_hard_required_constraint() {
        let spec = AffinitySpec::new()
            .add_rule(AffinityRule::new(AffinityKind::Required, "gpu").with_value("nvidia"));
        assert!(spec.satisfies_hard_constraints(&gpu_worker()));
        assert!(!spec.satisfies_hard_constraints(&cpu_worker()));
    }

    #[test]
    fn test_hard_anti_affinity_constraint() {
        let spec = AffinitySpec::new().add_rule(
            AffinityRule::new(AffinityKind::RequiredAntiAffinity, "region").with_value("eu-west"),
        );
        assert!(spec.satisfies_hard_constraints(&gpu_worker()));
        assert!(!spec.satisfies_hard_constraints(&eu_worker()));
    }

    #[test]
    fn test_preference_score_positive() {
        let spec = AffinitySpec::new().add_rule(
            AffinityRule::new(AffinityKind::Preferred, "gpu")
                .with_value("nvidia")
                .with_weight(10),
        );
        let score = spec.preference_score(&gpu_worker());
        assert!((score - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_preference_score_anti_affinity() {
        let spec = AffinitySpec::new().add_rule(
            AffinityRule::new(AffinityKind::PreferredAntiAffinity, "region")
                .with_value("eu-west")
                .with_weight(5),
        );
        let score_eu = spec.preference_score(&eu_worker());
        let score_us = spec.preference_score(&gpu_worker());
        assert!(score_eu < score_us);
    }

    #[test]
    fn test_select_best_worker() {
        let workers = vec![gpu_worker(), cpu_worker(), eu_worker()];
        let matcher = AffinityMatcher::new(workers);
        let spec = AffinitySpec::new()
            .add_rule(AffinityRule::new(AffinityKind::Required, "gpu").with_value("nvidia"))
            .add_rule(
                AffinityRule::new(AffinityKind::Preferred, "region")
                    .with_value("us-east")
                    .with_weight(10),
            );
        let best = matcher.select_best(&spec).expect("best should be valid");
        assert_eq!(best.id, "gpu-1");
    }

    #[test]
    fn test_eligible_count() {
        let workers = vec![gpu_worker(), cpu_worker(), eu_worker()];
        let matcher = AffinityMatcher::new(workers);
        let spec = AffinitySpec::new()
            .add_rule(AffinityRule::new(AffinityKind::Required, "gpu").with_value("nvidia"));
        assert_eq!(matcher.eligible_count(&spec), 2); // gpu-1 and eu-1
    }

    #[test]
    fn test_rank_workers() {
        let workers = vec![gpu_worker(), cpu_worker(), eu_worker()];
        let matcher = AffinityMatcher::new(workers);
        let spec = AffinitySpec::new()
            .add_rule(AffinityRule::new(AffinityKind::Required, "gpu").with_value("nvidia"))
            .add_rule(
                AffinityRule::new(AffinityKind::Preferred, "region")
                    .with_value("us-east")
                    .with_weight(5),
            );
        let ranked = matcher.rank_workers(&spec);
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].0.id, "gpu-1");
    }

    #[test]
    fn test_empty_spec_all_eligible() {
        let workers = vec![gpu_worker(), cpu_worker()];
        let matcher = AffinityMatcher::new(workers);
        let spec = AffinitySpec::new();
        assert_eq!(matcher.eligible_count(&spec), 2);
    }

    #[test]
    fn test_affinity_kind_display() {
        assert_eq!(AffinityKind::Preferred.to_string(), "Preferred");
        assert_eq!(AffinityKind::Required.to_string(), "Required");
        assert_eq!(
            AffinityKind::PreferredAntiAffinity.to_string(),
            "PreferredAntiAffinity"
        );
        assert_eq!(
            AffinityKind::RequiredAntiAffinity.to_string(),
            "RequiredAntiAffinity"
        );
    }

    #[test]
    fn test_is_hard() {
        assert!(AffinityRule::new(AffinityKind::Required, "x").is_hard());
        assert!(AffinityRule::new(AffinityKind::RequiredAntiAffinity, "x").is_hard());
        assert!(!AffinityRule::new(AffinityKind::Preferred, "x").is_hard());
        assert!(!AffinityRule::new(AffinityKind::PreferredAntiAffinity, "x").is_hard());
    }
}
