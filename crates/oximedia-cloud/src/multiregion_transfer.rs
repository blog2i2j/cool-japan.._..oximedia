// multiregion_transfer.rs — Cross-region transfer optimisation.
//
// Provides:
//   - `RegionLatency` — latency + cost matrix between region pairs
//   - `TransferOptimizer` — selects the optimal source region
//   - `TransferPlan` — full routing plan with cost estimate
//   - `TransferRoute` — one hop in a multi-hop plan
#![allow(dead_code)]

use std::collections::HashMap;

use crate::error::{CloudError, Result};

// ---------------------------------------------------------------------------
// RegionNode
// ---------------------------------------------------------------------------

/// A single named cloud region participating in cross-region transfers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RegionNode {
    /// Unique region identifier (e.g. `"us-east-1"`, `"eu-west-1"`).
    pub id: String,
    /// Human-readable display name.
    pub display_name: String,
    /// Cloud provider tag (e.g. `"aws"`, `"gcp"`, `"azure"`).
    pub provider: String,
}

impl RegionNode {
    /// Create a new region node.
    pub fn new(
        id: impl Into<String>,
        display_name: impl Into<String>,
        provider: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            display_name: display_name.into(),
            provider: provider.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// EdgeMetrics
// ---------------------------------------------------------------------------

/// Network and cost metrics for a directed transfer edge (source → destination).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeMetrics {
    /// Observed round-trip latency in milliseconds.
    pub latency_ms: u32,
    /// Estimated egress cost in USD per GiB transferred.
    pub cost_per_gib_usd: f64,
    /// Achievable bandwidth in Mbps (used for transfer time estimation).
    pub bandwidth_mbps: f64,
    /// Reliability score in [0.0, 1.0] where 1.0 = fully reliable.
    pub reliability: f64,
}

impl EdgeMetrics {
    /// Create a new `EdgeMetrics`.
    ///
    /// Returns an error when any value is out of the valid range.
    pub fn new(
        latency_ms: u32,
        cost_per_gib_usd: f64,
        bandwidth_mbps: f64,
        reliability: f64,
    ) -> Result<Self> {
        if cost_per_gib_usd < 0.0 {
            return Err(CloudError::InvalidParameter(
                "cost_per_gib_usd must be non-negative".into(),
            ));
        }
        if bandwidth_mbps <= 0.0 {
            return Err(CloudError::InvalidParameter(
                "bandwidth_mbps must be positive".into(),
            ));
        }
        if !(0.0..=1.0).contains(&reliability) {
            return Err(CloudError::InvalidParameter(
                "reliability must be in [0.0, 1.0]".into(),
            ));
        }
        Ok(Self {
            latency_ms,
            cost_per_gib_usd,
            bandwidth_mbps,
            reliability,
        })
    }

    /// Estimate transfer time in seconds for `size_bytes` over this edge.
    pub fn estimated_transfer_secs(&self, size_bytes: u64) -> f64 {
        let size_megabits = (size_bytes as f64 * 8.0) / 1_000_000.0;
        (size_megabits / self.bandwidth_mbps) + (self.latency_ms as f64 / 1000.0)
    }

    /// Estimate cost in USD to transfer `size_bytes` over this edge.
    pub fn estimated_cost_usd(&self, size_bytes: u64) -> f64 {
        let size_gib = size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        size_gib * self.cost_per_gib_usd
    }
}

// ---------------------------------------------------------------------------
// RegionLatency  (the latency/cost matrix)
// ---------------------------------------------------------------------------

/// A directed graph of inter-region edges, each annotated with `EdgeMetrics`.
///
/// Internally keyed by `(source_id, destination_id)`.
#[derive(Debug, Default)]
pub struct RegionLatency {
    edges: HashMap<(String, String), EdgeMetrics>,
    nodes: HashMap<String, RegionNode>,
}

impl RegionLatency {
    /// Create an empty matrix.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a cloud region.
    pub fn add_region(&mut self, node: RegionNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    /// Add or update the metrics for a directed edge.
    pub fn set_edge(
        &mut self,
        source_id: impl Into<String>,
        destination_id: impl Into<String>,
        metrics: EdgeMetrics,
    ) {
        let src = source_id.into();
        let dst = destination_id.into();
        self.edges.insert((src, dst), metrics);
    }

    /// Retrieve the metrics for a directed edge, if present.
    pub fn edge(&self, source_id: &str, destination_id: &str) -> Option<&EdgeMetrics> {
        self.edges.get(&(source_id.to_string(), destination_id.to_string()))
    }

    /// Return all regions registered in this matrix.
    pub fn regions(&self) -> impl Iterator<Item = &RegionNode> {
        self.nodes.values()
    }

    /// Return all source regions that have an edge to `destination_id`.
    pub fn sources_for(&self, destination_id: &str) -> Vec<(&RegionNode, &EdgeMetrics)> {
        self.edges
            .iter()
            .filter(|((_, dst), _)| dst == destination_id)
            .filter_map(|((src, _), metrics)| {
                self.nodes.get(src).map(|node| (node, metrics))
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// OptimisationObjective
// ---------------------------------------------------------------------------

/// The primary dimension to optimise when selecting a transfer route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimisationObjective {
    /// Minimise estimated transfer latency (lowest latency wins).
    MinimiseLatency,
    /// Minimise estimated egress cost (cheapest edge wins).
    MinimiseCost,
    /// Balanced score: weighted combination of latency + cost + reliability.
    Balanced,
}

impl Default for OptimisationObjective {
    fn default() -> Self {
        Self::Balanced
    }
}

// ---------------------------------------------------------------------------
// TransferRoute
// ---------------------------------------------------------------------------

/// One hop in a (potentially multi-hop) transfer route.
#[derive(Debug, Clone)]
pub struct TransferRoute {
    pub source_region: String,
    pub destination_region: String,
    pub metrics: EdgeMetrics,
}

impl TransferRoute {
    /// Estimated cost for `size_bytes` over this hop.
    pub fn cost_usd(&self, size_bytes: u64) -> f64 {
        self.metrics.estimated_cost_usd(size_bytes)
    }

    /// Estimated transfer time for `size_bytes` over this hop.
    pub fn transfer_secs(&self, size_bytes: u64) -> f64 {
        self.metrics.estimated_transfer_secs(size_bytes)
    }
}

// ---------------------------------------------------------------------------
// TransferPlan
// ---------------------------------------------------------------------------

/// A complete cross-region transfer plan produced by `TransferOptimizer`.
#[derive(Debug, Clone)]
pub struct TransferPlan {
    /// Ordered list of hops from the chosen source to the final destination.
    pub routes: Vec<TransferRoute>,
    /// Payload size used to produce cost/time estimates.
    pub payload_bytes: u64,
    /// Total estimated cost in USD.
    pub estimated_cost_usd: f64,
    /// Total estimated transfer time in seconds (sum of hop latencies + propagation).
    pub estimated_transfer_secs: f64,
    /// Objective used when this plan was generated.
    pub objective: OptimisationObjective,
}

impl TransferPlan {
    /// Returns the region id of the chosen source.
    pub fn source_region(&self) -> Option<&str> {
        self.routes.first().map(|r| r.source_region.as_str())
    }

    /// Returns the final destination region id.
    pub fn destination_region(&self) -> Option<&str> {
        self.routes.last().map(|r| r.destination_region.as_str())
    }

    /// Number of hops.
    pub fn hop_count(&self) -> usize {
        self.routes.len()
    }
}

// ---------------------------------------------------------------------------
// TransferOptimizer
// ---------------------------------------------------------------------------

/// Selects the optimal source region for a cross-region transfer given a
/// [`RegionLatency`] matrix and an [`OptimisationObjective`].
pub struct TransferOptimizer {
    matrix: RegionLatency,
    objective: OptimisationObjective,
    /// Weight applied to the latency component in Balanced scoring.
    latency_weight: f64,
    /// Weight applied to the cost component in Balanced scoring.
    cost_weight: f64,
    /// Weight applied to the reliability component in Balanced scoring.
    reliability_weight: f64,
    /// Minimum acceptable reliability (edges below this are excluded).
    min_reliability: f64,
    /// Maximum acceptable latency in milliseconds (edges above are excluded).
    max_latency_ms: u32,
}

impl TransferOptimizer {
    /// Create a new optimizer with the given matrix and objective.
    pub fn new(matrix: RegionLatency, objective: OptimisationObjective) -> Self {
        Self {
            matrix,
            objective,
            latency_weight: 0.4,
            cost_weight: 0.4,
            reliability_weight: 0.2,
            min_reliability: 0.0,
            max_latency_ms: u32::MAX,
        }
    }

    /// Builder: set the minimum reliability threshold (edges below are skipped).
    pub fn with_min_reliability(mut self, min: f64) -> Self {
        self.min_reliability = min.clamp(0.0, 1.0);
        self
    }

    /// Builder: set the maximum acceptable latency in milliseconds.
    pub fn with_max_latency_ms(mut self, max: u32) -> Self {
        self.max_latency_ms = max;
        self
    }

    /// Builder: customise balanced scoring weights (values are normalised to sum=1).
    pub fn with_weights(
        mut self,
        latency_weight: f64,
        cost_weight: f64,
        reliability_weight: f64,
    ) -> Result<Self> {
        let total = latency_weight + cost_weight + reliability_weight;
        if total <= 0.0 {
            return Err(CloudError::InvalidParameter(
                "sum of weights must be positive".into(),
            ));
        }
        self.latency_weight = latency_weight / total;
        self.cost_weight = cost_weight / total;
        self.reliability_weight = reliability_weight / total;
        Ok(self)
    }

    /// Generate the optimal `TransferPlan` from any registered source to
    /// `destination_id` for a payload of `payload_bytes`.
    ///
    /// Returns an error when no viable route exists.
    pub fn plan(
        &self,
        destination_id: &str,
        payload_bytes: u64,
    ) -> Result<TransferPlan> {
        let candidates = self.matrix.sources_for(destination_id);
        if candidates.is_empty() {
            return Err(CloudError::NotFound(format!(
                "no routes to destination '{}'",
                destination_id
            )));
        }

        // Filter by hard constraints.
        let viable: Vec<_> = candidates
            .into_iter()
            .filter(|(_, m)| {
                m.reliability >= self.min_reliability && m.latency_ms <= self.max_latency_ms
            })
            .collect();

        if viable.is_empty() {
            return Err(CloudError::ServiceUnavailable(format!(
                "all routes to '{}' fail reliability/latency constraints",
                destination_id
            )));
        }

        // Score candidates.
        let (best_node, best_metrics) = self.select_best(&viable, payload_bytes)?;

        let route = TransferRoute {
            source_region: best_node.id.clone(),
            destination_region: destination_id.to_string(),
            metrics: *best_metrics,
        };

        let estimated_cost_usd = route.cost_usd(payload_bytes);
        let estimated_transfer_secs = route.transfer_secs(payload_bytes);

        Ok(TransferPlan {
            routes: vec![route],
            payload_bytes,
            estimated_cost_usd,
            estimated_transfer_secs,
            objective: self.objective,
        })
    }

    /// Generate a plan from a specific source region to a destination.
    pub fn plan_from(
        &self,
        source_id: &str,
        destination_id: &str,
        payload_bytes: u64,
    ) -> Result<TransferPlan> {
        let metrics = self
            .matrix
            .edge(source_id, destination_id)
            .ok_or_else(|| {
                CloudError::NotFound(format!(
                    "no edge from '{}' to '{}'",
                    source_id, destination_id
                ))
            })?;

        if metrics.reliability < self.min_reliability {
            return Err(CloudError::ServiceUnavailable(format!(
                "edge reliability {:.2} below minimum {:.2}",
                metrics.reliability, self.min_reliability
            )));
        }
        if metrics.latency_ms > self.max_latency_ms {
            return Err(CloudError::ServiceUnavailable(format!(
                "edge latency {} ms exceeds maximum {} ms",
                metrics.latency_ms, self.max_latency_ms
            )));
        }

        let route = TransferRoute {
            source_region: source_id.to_string(),
            destination_region: destination_id.to_string(),
            metrics: *metrics,
        };
        let estimated_cost_usd = route.cost_usd(payload_bytes);
        let estimated_transfer_secs = route.transfer_secs(payload_bytes);
        Ok(TransferPlan {
            routes: vec![route],
            payload_bytes,
            estimated_cost_usd,
            estimated_transfer_secs,
            objective: self.objective,
        })
    }

    // ------------------------------------------------------------------
    // Private scoring
    // ------------------------------------------------------------------

    fn select_best<'a>(
        &self,
        candidates: &[(&'a RegionNode, &'a EdgeMetrics)],
        payload_bytes: u64,
    ) -> Result<(&'a RegionNode, &'a EdgeMetrics)> {
        match self.objective {
            OptimisationObjective::MinimiseLatency => candidates
                .iter()
                .min_by_key(|(_, m)| m.latency_ms)
                .copied()
                .ok_or_else(|| CloudError::Other("empty candidate list".into())),

            OptimisationObjective::MinimiseCost => candidates
                .iter()
                .min_by(|(_, a), (_, b)| {
                    let ca = a.estimated_cost_usd(payload_bytes);
                    let cb = b.estimated_cost_usd(payload_bytes);
                    ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied()
                .ok_or_else(|| CloudError::Other("empty candidate list".into())),

            OptimisationObjective::Balanced => {
                // Compute normalisation ranges.
                let max_lat = candidates.iter().map(|(_, m)| m.latency_ms).max().unwrap_or(1);
                let max_cost = candidates
                    .iter()
                    .map(|(_, m)| m.estimated_cost_usd(payload_bytes))
                    .fold(f64::NEG_INFINITY, f64::max);
                let max_lat_f = max_lat.max(1) as f64;
                let max_cost_f = if max_cost <= 0.0 { 1.0 } else { max_cost };

                candidates
                    .iter()
                    .min_by(|(_, a), (_, b)| {
                        let score_a = self.balanced_score(*a, payload_bytes, max_lat_f, max_cost_f);
                        let score_b = self.balanced_score(*b, payload_bytes, max_lat_f, max_cost_f);
                        score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .copied()
                    .ok_or_else(|| CloudError::Other("empty candidate list".into()))
            }
        }
    }

    /// Balanced score in [0.0, 1.0] — lower is better.
    fn balanced_score(
        &self,
        m: &EdgeMetrics,
        payload_bytes: u64,
        max_lat_f: f64,
        max_cost_f: f64,
    ) -> f64 {
        let lat_norm = m.latency_ms as f64 / max_lat_f;
        let cost_norm = m.estimated_cost_usd(payload_bytes) / max_cost_f;
        // Reliability: higher reliability → lower penalty (1 - reliability).
        let rel_penalty = 1.0 - m.reliability;

        self.latency_weight * lat_norm
            + self.cost_weight * cost_norm
            + self.reliability_weight * rel_penalty
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_metrics(latency_ms: u32, cost_per_gib: f64, bw_mbps: f64, rel: f64) -> EdgeMetrics {
        EdgeMetrics::new(latency_ms, cost_per_gib, bw_mbps, rel).expect("valid metrics")
    }

    fn make_matrix() -> RegionLatency {
        let mut m = RegionLatency::new();
        m.add_region(RegionNode::new("us-east-1", "US East (N. Virginia)", "aws"));
        m.add_region(RegionNode::new("eu-west-1", "EU (Ireland)", "aws"));
        m.add_region(RegionNode::new("ap-southeast-1", "Asia Pacific (Singapore)", "aws"));
        // Edges to "eu-west-1"
        m.set_edge("us-east-1", "eu-west-1", make_metrics(85, 0.09, 500.0, 0.99));
        m.set_edge("ap-southeast-1", "eu-west-1", make_metrics(200, 0.06, 300.0, 0.95));
        // Edge to "ap-southeast-1"
        m.set_edge("us-east-1", "ap-southeast-1", make_metrics(160, 0.09, 400.0, 0.98));
        m
    }

    #[test]
    fn test_edge_metrics_validation_negative_cost() {
        let result = EdgeMetrics::new(10, -1.0, 100.0, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_edge_metrics_validation_zero_bandwidth() {
        let result = EdgeMetrics::new(10, 0.05, 0.0, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_edge_metrics_validation_reliability_out_of_range() {
        let result = EdgeMetrics::new(10, 0.05, 100.0, 1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_edge_metrics_cost_estimation() {
        let m = make_metrics(50, 0.09, 500.0, 1.0);
        // 1 GiB = 1024^3 bytes
        let one_gib = 1024u64 * 1024 * 1024;
        let cost = m.estimated_cost_usd(one_gib);
        assert!((cost - 0.09).abs() < 1e-9, "cost for 1 GiB should be ~0.09, got {}", cost);
    }

    #[test]
    fn test_edge_metrics_transfer_time_estimation() {
        // 1 Gbps = 1000 Mbps; 1 GiB = 8192 Mbits ≈ 8.192 s at 1000 Mbps
        let m = make_metrics(0, 0.0, 1000.0, 1.0);
        let one_gib = 1024u64 * 1024 * 1024;
        let secs = m.estimated_transfer_secs(one_gib);
        assert!(secs > 8.0 && secs < 9.0, "transfer time ~8.59 s, got {}", secs);
    }

    #[test]
    fn test_region_latency_sources_for() {
        let matrix = make_matrix();
        let sources = matrix.sources_for("eu-west-1");
        assert_eq!(sources.len(), 2, "two sources to eu-west-1");
    }

    #[test]
    fn test_optimizer_minimise_latency() {
        let optimizer = TransferOptimizer::new(make_matrix(), OptimisationObjective::MinimiseLatency);
        let plan = optimizer.plan("eu-west-1", 1024 * 1024 * 1024).expect("plan");
        // us-east-1 has lower latency (85 ms) vs ap-southeast-1 (200 ms).
        assert_eq!(plan.source_region(), Some("us-east-1"));
    }

    #[test]
    fn test_optimizer_minimise_cost() {
        let optimizer = TransferOptimizer::new(make_matrix(), OptimisationObjective::MinimiseCost);
        let plan = optimizer.plan("eu-west-1", 10 * 1024 * 1024 * 1024).expect("plan");
        // ap-southeast-1 has lower cost (0.06/GiB) vs us-east-1 (0.09/GiB).
        assert_eq!(plan.source_region(), Some("ap-southeast-1"));
    }

    #[test]
    fn test_optimizer_balanced() {
        let optimizer = TransferOptimizer::new(make_matrix(), OptimisationObjective::Balanced);
        let plan = optimizer.plan("eu-west-1", 1024 * 1024 * 1024).expect("plan");
        // Just verify a plan is produced and contains a valid source.
        assert!(plan.source_region().is_some());
        assert_eq!(plan.destination_region(), Some("eu-west-1"));
    }

    #[test]
    fn test_optimizer_no_routes_error() {
        let optimizer = TransferOptimizer::new(make_matrix(), OptimisationObjective::MinimiseLatency);
        let result = optimizer.plan("nonexistent-region", 1000);
        assert!(result.is_err());
        match result.unwrap_err() {
            CloudError::NotFound(_) => {}
            other => panic!("expected NotFound, got {:?}", other),
        }
    }

    #[test]
    fn test_optimizer_min_reliability_filter() {
        let optimizer = TransferOptimizer::new(make_matrix(), OptimisationObjective::MinimiseLatency)
            .with_min_reliability(0.97); // ap-southeast-1 has 0.95 → excluded
        let plan = optimizer.plan("eu-west-1", 1000).expect("plan");
        // Only us-east-1 (reliability 0.99) should survive.
        assert_eq!(plan.source_region(), Some("us-east-1"));
    }

    #[test]
    fn test_optimizer_max_latency_filter_excludes_all() {
        let optimizer = TransferOptimizer::new(make_matrix(), OptimisationObjective::MinimiseLatency)
            .with_max_latency_ms(10); // both edges exceed 10 ms
        let result = optimizer.plan("eu-west-1", 1000);
        assert!(result.is_err());
        match result.unwrap_err() {
            CloudError::ServiceUnavailable(_) => {}
            other => panic!("expected ServiceUnavailable, got {:?}", other),
        }
    }

    #[test]
    fn test_optimizer_plan_from_specific_source() {
        let optimizer = TransferOptimizer::new(make_matrix(), OptimisationObjective::Balanced);
        let plan = optimizer
            .plan_from("us-east-1", "eu-west-1", 1024 * 1024)
            .expect("plan_from");
        assert_eq!(plan.source_region(), Some("us-east-1"));
        assert_eq!(plan.hop_count(), 1);
        assert!(plan.estimated_cost_usd > 0.0);
    }

    #[test]
    fn test_plan_cost_and_time_non_negative() {
        let optimizer = TransferOptimizer::new(make_matrix(), OptimisationObjective::MinimiseCost);
        let size = 5 * 1024 * 1024 * 1024u64; // 5 GiB
        let plan = optimizer.plan("eu-west-1", size).expect("plan");
        assert!(plan.estimated_cost_usd >= 0.0);
        assert!(plan.estimated_transfer_secs >= 0.0);
    }
}
