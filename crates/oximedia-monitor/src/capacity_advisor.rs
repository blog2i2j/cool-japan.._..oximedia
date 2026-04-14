//! Multi-dimensional capacity advisor with headroom calculation, scale-up
//! recommendation engine, and composite resource scoring.
//!
//! This module extends the lower-level [`crate::capacity_planner`] and
//! [`crate::resource_forecast`] by providing a higher-level decision layer:
//!
//! - **`ResourceHeadroom`** — per-resource headroom analysis: current usage,
//!   margin to the configured ceiling, and a derived headroom tier
//!   (Ample / Moderate / Tight / Critical).
//! - **`ScaleRecommendation`** — a typed scale-up or scale-out action with an
//!   urgency level and the triggering resource that caused it.
//! - **`CapacityScore`** — a single composite 0–100 score across all tracked
//!   resources, allowing quick at-a-glance health assessment.
//! - **`CapacityAdvisor`** — the central struct that holds per-resource
//!   utilisation samples, computes headroom, and emits [`ScaleRecommendation`]
//!   values when thresholds are crossed.
//!
//! # Design rationale
//!
//! Unlike `CapacityPlanner` (which performs single-resource OLS forecasting),
//! the advisor works on **current utilisation snapshots** and applies
//! configurable thresholds that map directly to operational playbooks.  A
//! composite score lets ops dashboards surface overall infrastructure health
//! without requiring operators to inspect every resource dimension individually.

#![allow(dead_code)]

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Resource Kind
// ---------------------------------------------------------------------------

/// The resource dimension being evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AdvisedResource {
    /// CPU compute.
    Cpu,
    /// System memory.
    Memory,
    /// Persistent disk / storage I/O.
    Disk,
    /// Network bandwidth.
    Network,
    /// GPU compute (encoding / decoding acceleration).
    Gpu,
    /// Encoding worker slots (logical concurrency).
    EncoderWorkers,
    /// Ingest stream capacity.
    IngestCapacity,
}

impl AdvisedResource {
    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Memory => "Memory",
            Self::Disk => "Disk",
            Self::Network => "Network",
            Self::Gpu => "GPU",
            Self::EncoderWorkers => "Encoder Workers",
            Self::IngestCapacity => "Ingest Capacity",
        }
    }
}

impl std::fmt::Display for AdvisedResource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ---------------------------------------------------------------------------
// Headroom Tier
// ---------------------------------------------------------------------------

/// Qualitative headroom tier derived from percentage margin to ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HeadroomTier {
    /// > 40 % margin — comfortable, no action needed.
    Ample,
    /// 20–40 % margin — monitor closely.
    Moderate,
    /// 5–20 % margin — plan capacity addition soon.
    Tight,
    /// < 5 % margin — immediate action required.
    Critical,
}

impl HeadroomTier {
    /// Derive a headroom tier from a percentage margin (0–100).
    #[must_use]
    pub fn from_margin_pct(margin_pct: f64) -> Self {
        if margin_pct > 40.0 {
            Self::Ample
        } else if margin_pct > 20.0 {
            Self::Moderate
        } else if margin_pct > 5.0 {
            Self::Tight
        } else {
            Self::Critical
        }
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Ample => "Ample",
            Self::Moderate => "Moderate",
            Self::Tight => "Tight",
            Self::Critical => "Critical",
        }
    }
}

impl std::fmt::Display for HeadroomTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ---------------------------------------------------------------------------
// ResourceHeadroom
// ---------------------------------------------------------------------------

/// Headroom analysis for a single resource dimension.
#[derive(Debug, Clone)]
pub struct ResourceHeadroom {
    /// The resource this analysis covers.
    pub resource: AdvisedResource,
    /// Current utilisation in the same unit as the ceiling (e.g. percentage).
    pub current_utilisation: f64,
    /// Configured ceiling (e.g. 90.0 for 90 %).
    pub ceiling: f64,
    /// Remaining capacity before the ceiling is reached.
    pub margin: f64,
    /// Margin expressed as a percentage of the ceiling.
    pub margin_pct: f64,
    /// Qualitative tier.
    pub tier: HeadroomTier,
}

impl ResourceHeadroom {
    /// Compute headroom given current utilisation and a ceiling value.
    #[must_use]
    pub fn compute(resource: AdvisedResource, current_utilisation: f64, ceiling: f64) -> Self {
        let ceiling = ceiling.max(1e-12); // avoid div-by-zero
        let current = current_utilisation.clamp(0.0, ceiling * 2.0); // allow over-ceiling
        let margin = (ceiling - current).max(0.0);
        let margin_pct = (margin / ceiling * 100.0).clamp(0.0, 100.0);
        let tier = HeadroomTier::from_margin_pct(margin_pct);
        Self {
            resource,
            current_utilisation: current,
            ceiling,
            margin,
            margin_pct,
            tier,
        }
    }

    /// Returns `true` if this resource is at or above the ceiling.
    #[must_use]
    pub fn is_at_ceiling(&self) -> bool {
        self.current_utilisation >= self.ceiling
    }
}

// ---------------------------------------------------------------------------
// Scale Action
// ---------------------------------------------------------------------------

/// The kind of capacity action recommended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScaleAction {
    /// Add more compute units of the same type (vertical scale-up).
    ScaleUp,
    /// Add more instances / nodes (horizontal scale-out).
    ScaleOut,
    /// Redistribute load across existing capacity.
    Rebalance,
    /// Reduce or defer non-critical workloads to free headroom.
    ShedLoad,
}

impl std::fmt::Display for ScaleAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ScaleUp => write!(f, "scale-up"),
            Self::ScaleOut => write!(f, "scale-out"),
            Self::Rebalance => write!(f, "rebalance"),
            Self::ShedLoad => write!(f, "shed-load"),
        }
    }
}

// ---------------------------------------------------------------------------
// Urgency
// ---------------------------------------------------------------------------

/// Urgency level attached to a scale recommendation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Urgency {
    /// Plan within the next maintenance window.
    Planned,
    /// Execute within the next 24 hours.
    Soon,
    /// Execute immediately.
    Immediate,
}

impl std::fmt::Display for Urgency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Planned => write!(f, "planned"),
            Self::Soon => write!(f, "soon"),
            Self::Immediate => write!(f, "immediate"),
        }
    }
}

// ---------------------------------------------------------------------------
// ScaleRecommendation
// ---------------------------------------------------------------------------

/// A concrete scaling recommendation produced by the advisor.
#[derive(Debug, Clone)]
pub struct ScaleRecommendation {
    /// Which resource triggered this recommendation.
    pub resource: AdvisedResource,
    /// Recommended action.
    pub action: ScaleAction,
    /// How urgently the action should be taken.
    pub urgency: Urgency,
    /// Human-readable rationale.
    pub rationale: String,
    /// Current utilisation at the time of recommendation.
    pub current_utilisation: f64,
    /// Headroom tier that triggered the recommendation.
    pub headroom_tier: HeadroomTier,
}

impl ScaleRecommendation {
    fn new(
        resource: AdvisedResource,
        action: ScaleAction,
        urgency: Urgency,
        headroom: &ResourceHeadroom,
    ) -> Self {
        let rationale = format!(
            "{resource} utilisation at {:.1} % (ceiling {:.1} %, margin {:.1} % — {headroom_tier}). \
             Recommend {action} with {urgency} urgency.",
            headroom.current_utilisation,
            headroom.ceiling,
            headroom.margin_pct,
            headroom_tier = headroom.tier,
        );
        Self {
            resource,
            action,
            urgency,
            rationale,
            current_utilisation: headroom.current_utilisation,
            headroom_tier: headroom.tier,
        }
    }
}

// ---------------------------------------------------------------------------
// Capacity Score
// ---------------------------------------------------------------------------

/// Composite capacity health score (0 = fully saturated, 100 = fully available).
#[derive(Debug, Clone)]
pub struct CapacityScore {
    /// Overall composite score (0–100).
    pub overall: f64,
    /// Per-resource scores contributing to the composite.
    pub per_resource: Vec<(AdvisedResource, f64)>,
    /// Number of resources in Critical headroom tier.
    pub critical_count: usize,
    /// Number of resources in Tight headroom tier.
    pub tight_count: usize,
}

impl CapacityScore {
    /// Returns the qualitative health label for the overall score.
    #[must_use]
    pub fn health_label(&self) -> &'static str {
        if self.overall >= 80.0 {
            "Healthy"
        } else if self.overall >= 60.0 {
            "Warning"
        } else if self.overall >= 30.0 {
            "Degraded"
        } else {
            "Critical"
        }
    }
}

// ---------------------------------------------------------------------------
// Resource Config
// ---------------------------------------------------------------------------

/// Configuration for a single resource dimension tracked by the advisor.
#[derive(Debug, Clone)]
pub struct ResourceConfig {
    /// Utilisation ceiling that triggers `Tight` warnings.
    pub ceiling: f64,
    /// Preferred scale action when this resource is constrained.
    pub preferred_action: ScaleAction,
    /// Weight applied when computing the composite score (0.0–1.0).
    pub weight: f64,
}

impl ResourceConfig {
    /// Create a config with default weight of 1.0.
    #[must_use]
    pub fn new(ceiling: f64, preferred_action: ScaleAction) -> Self {
        Self {
            ceiling: ceiling.clamp(1.0, f64::MAX),
            preferred_action,
            weight: 1.0,
        }
    }

    /// Override the weight.
    #[must_use]
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight.clamp(0.0, 1.0);
        self
    }
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self::new(90.0, ScaleAction::ScaleUp)
    }
}

// ---------------------------------------------------------------------------
// CapacityAdvisor
// ---------------------------------------------------------------------------

/// Multi-dimensional capacity advisor.
///
/// # Usage
///
/// ```
/// use oximedia_monitor::capacity_advisor::{
///     CapacityAdvisor, AdvisedResource, ResourceConfig, ScaleAction,
/// };
///
/// let mut advisor = CapacityAdvisor::new();
/// advisor.configure(
///     AdvisedResource::Cpu,
///     ResourceConfig::new(90.0, ScaleAction::ScaleOut),
/// );
/// advisor.update(AdvisedResource::Cpu, 85.0);
///
/// let recs = advisor.recommendations();
/// for rec in &recs {
///     println!("{}: {} ({})", rec.resource, rec.action, rec.urgency);
/// }
/// ```
#[derive(Debug, Default)]
pub struct CapacityAdvisor {
    /// Per-resource configuration.
    configs: HashMap<AdvisedResource, ResourceConfig>,
    /// Latest utilisation reading per resource.
    utilisation: HashMap<AdvisedResource, f64>,
}

impl CapacityAdvisor {
    /// Create a new advisor with no resources configured.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an advisor pre-configured with sensible defaults for all
    /// `AdvisedResource` variants (ceiling = 90 %, scale-up preferred).
    #[must_use]
    pub fn with_defaults() -> Self {
        let resources = [
            AdvisedResource::Cpu,
            AdvisedResource::Memory,
            AdvisedResource::Disk,
            AdvisedResource::Network,
            AdvisedResource::Gpu,
            AdvisedResource::EncoderWorkers,
            AdvisedResource::IngestCapacity,
        ];
        let mut advisor = Self::new();
        for r in resources {
            advisor.configure(r, ResourceConfig::default());
        }
        advisor
    }

    /// Register or update configuration for a resource.
    pub fn configure(&mut self, resource: AdvisedResource, config: ResourceConfig) {
        self.configs.insert(resource, config);
    }

    /// Record or update the current utilisation for a resource.
    ///
    /// The resource must have been registered via [`Self::configure`] first;
    /// if it hasn't, the observation is stored but no recommendation will be
    /// generated without a config.
    pub fn update(&mut self, resource: AdvisedResource, utilisation: f64) {
        self.utilisation.insert(resource, utilisation);
    }

    /// Compute headroom for a single resource.
    ///
    /// Returns `None` if no configuration is registered for the resource.
    #[must_use]
    pub fn headroom(&self, resource: AdvisedResource) -> Option<ResourceHeadroom> {
        let config = self.configs.get(&resource)?;
        let current = self.utilisation.get(&resource).copied().unwrap_or(0.0);
        Some(ResourceHeadroom::compute(resource, current, config.ceiling))
    }

    /// Compute headroom for all configured resources.
    #[must_use]
    pub fn all_headroom(&self) -> Vec<ResourceHeadroom> {
        let mut resources: Vec<AdvisedResource> = self.configs.keys().copied().collect();
        resources.sort(); // deterministic ordering
        resources
            .into_iter()
            .filter_map(|r| self.headroom(r))
            .collect()
    }

    /// Generate scale recommendations for all resources that are in `Tight` or
    /// `Critical` headroom tier.
    #[must_use]
    pub fn recommendations(&self) -> Vec<ScaleRecommendation> {
        let mut recs = Vec::new();
        let mut resources: Vec<AdvisedResource> = self.configs.keys().copied().collect();
        resources.sort();

        for resource in resources {
            let Some(config) = self.configs.get(&resource) else {
                continue;
            };
            let Some(headroom) = self.headroom(resource) else {
                continue;
            };

            let urgency = match headroom.tier {
                HeadroomTier::Critical => Urgency::Immediate,
                HeadroomTier::Tight => Urgency::Soon,
                HeadroomTier::Moderate | HeadroomTier::Ample => continue, // no action needed
            };

            recs.push(ScaleRecommendation::new(
                resource,
                config.preferred_action,
                urgency,
                &headroom,
            ));
        }
        recs
    }

    /// Compute the composite capacity health score (0–100).
    ///
    /// Each configured resource contributes a per-resource score equal to
    /// `margin_pct` (i.e. 100 when fully available, 0 when at ceiling).
    /// Resources are weighted by their configured weight; the final score is
    /// the weighted average.
    ///
    /// Returns `100.0` if no resources are configured.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn composite_score(&self) -> CapacityScore {
        let headrooms = self.all_headroom();
        if headrooms.is_empty() {
            return CapacityScore {
                overall: 100.0,
                per_resource: Vec::new(),
                critical_count: 0,
                tight_count: 0,
            };
        }

        let mut weighted_sum = 0.0_f64;
        let mut total_weight = 0.0_f64;
        let mut per_resource = Vec::new();
        let mut critical_count = 0usize;
        let mut tight_count = 0usize;

        for h in &headrooms {
            let weight = self
                .configs
                .get(&h.resource)
                .map_or(1.0, |c| c.weight);
            let score = h.margin_pct; // 0–100
            weighted_sum += score * weight;
            total_weight += weight;
            per_resource.push((h.resource, score));
            match h.tier {
                HeadroomTier::Critical => critical_count += 1,
                HeadroomTier::Tight => tight_count += 1,
                _ => {}
            }
        }

        let overall = if total_weight < 1e-12 {
            100.0
        } else {
            (weighted_sum / total_weight).clamp(0.0, 100.0)
        };

        CapacityScore {
            overall,
            per_resource,
            critical_count,
            tight_count,
        }
    }

    /// Number of configured resources.
    #[must_use]
    pub fn configured_resource_count(&self) -> usize {
        self.configs.len()
    }

    /// Returns `true` if any configured resource is in the `Critical` headroom tier.
    #[must_use]
    pub fn has_critical_resource(&self) -> bool {
        self.all_headroom()
            .iter()
            .any(|h| h.tier == HeadroomTier::Critical)
    }

    /// Remove a resource from tracking.
    pub fn remove_resource(&mut self, resource: AdvisedResource) {
        self.configs.remove(&resource);
        self.utilisation.remove(&resource);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn advisor_with_cpu(utilisation: f64) -> CapacityAdvisor {
        let mut a = CapacityAdvisor::new();
        a.configure(
            AdvisedResource::Cpu,
            ResourceConfig::new(90.0, ScaleAction::ScaleOut),
        );
        a.update(AdvisedResource::Cpu, utilisation);
        a
    }

    // --- HeadroomTier tests ---

    #[test]
    fn test_headroom_tier_ample() {
        assert_eq!(HeadroomTier::from_margin_pct(50.0), HeadroomTier::Ample);
    }

    #[test]
    fn test_headroom_tier_moderate() {
        assert_eq!(HeadroomTier::from_margin_pct(30.0), HeadroomTier::Moderate);
    }

    #[test]
    fn test_headroom_tier_tight() {
        assert_eq!(HeadroomTier::from_margin_pct(10.0), HeadroomTier::Tight);
    }

    #[test]
    fn test_headroom_tier_critical() {
        assert_eq!(HeadroomTier::from_margin_pct(2.0), HeadroomTier::Critical);
    }

    // --- ResourceHeadroom tests ---

    #[test]
    fn test_headroom_compute_healthy() {
        let h = ResourceHeadroom::compute(AdvisedResource::Cpu, 50.0, 90.0);
        assert!((h.margin - 40.0).abs() < 1e-9);
        assert!((h.margin_pct - (40.0 / 90.0 * 100.0)).abs() < 1e-6);
        assert_eq!(h.tier, HeadroomTier::Ample);
        assert!(!h.is_at_ceiling());
    }

    #[test]
    fn test_headroom_compute_at_ceiling() {
        let h = ResourceHeadroom::compute(AdvisedResource::Memory, 90.0, 90.0);
        assert_eq!(h.margin, 0.0);
        assert!(h.is_at_ceiling());
        assert_eq!(h.tier, HeadroomTier::Critical);
    }

    #[test]
    fn test_headroom_compute_tight() {
        // 85 / 90 = 5.56 % margin → Tight
        let h = ResourceHeadroom::compute(AdvisedResource::Disk, 85.0, 90.0);
        assert_eq!(h.tier, HeadroomTier::Tight);
    }

    // --- CapacityAdvisor basic operations ---

    #[test]
    fn test_advisor_no_resources() {
        let a = CapacityAdvisor::new();
        assert_eq!(a.configured_resource_count(), 0);
        let score = a.composite_score();
        assert!((score.overall - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_advisor_with_defaults_has_all_resources() {
        let a = CapacityAdvisor::with_defaults();
        assert_eq!(a.configured_resource_count(), 7);
    }

    #[test]
    fn test_advisor_update_and_headroom() {
        let a = advisor_with_cpu(70.0);
        let h = a.headroom(AdvisedResource::Cpu).expect("headroom should exist");
        assert!((h.current_utilisation - 70.0).abs() < f64::EPSILON);
        assert_eq!(h.tier, HeadroomTier::Moderate);
    }

    #[test]
    fn test_advisor_no_recommendations_when_healthy() {
        let a = advisor_with_cpu(40.0); // lots of headroom
        let recs = a.recommendations();
        assert!(recs.is_empty());
    }

    #[test]
    fn test_advisor_tight_generates_soon_recommendation() {
        // ceiling=90, util=80 → margin=10, margin_pct=11.1% → Tight
        let a = advisor_with_cpu(80.0);
        let recs = a.recommendations();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].urgency, Urgency::Soon);
        assert_eq!(recs[0].resource, AdvisedResource::Cpu);
        assert_eq!(recs[0].action, ScaleAction::ScaleOut);
    }

    #[test]
    fn test_advisor_critical_generates_immediate_recommendation() {
        let a = advisor_with_cpu(89.0); // ~1.1 % margin → Critical
        let recs = a.recommendations();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].urgency, Urgency::Immediate);
        assert_eq!(recs[0].headroom_tier, HeadroomTier::Critical);
    }

    #[test]
    fn test_advisor_composite_score_healthy() {
        let a = advisor_with_cpu(40.0); // ~55 % margin → score ~55
        let score = a.composite_score();
        assert!(score.overall > 40.0, "score should be > 40, got {}", score.overall);
        assert_eq!(score.critical_count, 0);
        assert_eq!(score.tight_count, 0);
    }

    #[test]
    fn test_advisor_composite_score_critical() {
        let a = advisor_with_cpu(89.5); // nearly at ceiling
        let score = a.composite_score();
        assert!(score.overall < 10.0, "score should be < 10, got {}", score.overall);
        assert_eq!(score.critical_count, 1);
    }

    #[test]
    fn test_advisor_composite_weighted_average() {
        let mut a = CapacityAdvisor::new();
        // CPU: ceiling 100, util 60 → margin 40 %, weight 2.0
        a.configure(
            AdvisedResource::Cpu,
            ResourceConfig::new(100.0, ScaleAction::ScaleUp).with_weight(0.8),
        );
        // Memory: ceiling 100, util 90 → margin 10 %, weight 1.0
        a.configure(
            AdvisedResource::Memory,
            ResourceConfig::new(100.0, ScaleAction::ScaleUp).with_weight(0.4),
        );
        a.update(AdvisedResource::Cpu, 60.0);
        a.update(AdvisedResource::Memory, 90.0);

        let score = a.composite_score();
        // weighted: (40*0.8 + 10*0.4) / (0.8+0.4) = (32+4)/1.2 = 36/1.2 = 30
        assert!((score.overall - 30.0).abs() < 1e-6, "expected 30, got {}", score.overall);
    }

    #[test]
    fn test_advisor_has_critical_resource() {
        let a = advisor_with_cpu(89.9);
        assert!(a.has_critical_resource());
    }

    #[test]
    fn test_advisor_remove_resource() {
        let mut a = advisor_with_cpu(80.0);
        a.remove_resource(AdvisedResource::Cpu);
        assert_eq!(a.configured_resource_count(), 0);
        assert!(a.headroom(AdvisedResource::Cpu).is_none());
    }

    #[test]
    fn test_advisor_all_headroom_sorted() {
        let mut a = CapacityAdvisor::new();
        a.configure(AdvisedResource::Network, ResourceConfig::default());
        a.configure(AdvisedResource::Cpu, ResourceConfig::default());
        a.configure(AdvisedResource::Memory, ResourceConfig::default());
        a.update(AdvisedResource::Cpu, 50.0);
        a.update(AdvisedResource::Memory, 70.0);
        a.update(AdvisedResource::Network, 30.0);
        let headrooms = a.all_headroom();
        assert_eq!(headrooms.len(), 3);
        // Sorted by AdvisedResource order: Cpu < Memory < Network
        assert_eq!(headrooms[0].resource, AdvisedResource::Cpu);
        assert_eq!(headrooms[1].resource, AdvisedResource::Memory);
        assert_eq!(headrooms[2].resource, AdvisedResource::Network);
    }

    #[test]
    fn test_health_label_healthy() {
        let score = CapacityScore {
            overall: 85.0,
            per_resource: vec![],
            critical_count: 0,
            tight_count: 0,
        };
        assert_eq!(score.health_label(), "Healthy");
    }

    #[test]
    fn test_health_label_critical() {
        let score = CapacityScore {
            overall: 15.0,
            per_resource: vec![],
            critical_count: 3,
            tight_count: 1,
        };
        assert_eq!(score.health_label(), "Critical");
    }

    #[test]
    fn test_resource_config_weight_clamped() {
        let cfg = ResourceConfig::new(90.0, ScaleAction::ScaleUp).with_weight(5.0);
        assert!((cfg.weight - 1.0).abs() < f64::EPSILON);
    }
}
