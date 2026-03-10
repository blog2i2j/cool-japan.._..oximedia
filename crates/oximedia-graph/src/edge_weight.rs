//! Edge weight types for the filter graph pipeline.
//!
//! Models bandwidth, latency, and other costs associated with edges
//! (connections) in a processing graph.

#![allow(dead_code)]

use std::collections::HashMap;

/// The dimension in which an edge weight is expressed.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WeightType {
    /// Data throughput in bits per second.
    Bandwidth,
    /// Propagation or processing latency in microseconds.
    LatencyUs,
    /// Relative cost (dimensionless score, lower is cheaper).
    Cost,
    /// Packet-loss probability in the range `[0, 1]`.
    LossProbability,
}

impl WeightType {
    /// Returns the measurement unit string for this weight type.
    pub fn unit(&self) -> &'static str {
        match self {
            WeightType::Bandwidth => "bps",
            WeightType::LatencyUs => "µs",
            WeightType::Cost => "",
            WeightType::LossProbability => "",
        }
    }
}

/// A numeric weight attached to a graph edge.
#[derive(Debug, Clone)]
pub struct EdgeWeight {
    /// The kind of measurement this weight represents.
    weight_type: WeightType,
    /// The numeric value.
    value: f64,
    /// Threshold above which the edge is considered a bottleneck.
    bottleneck_threshold: f64,
}

impl EdgeWeight {
    /// Creates a new `EdgeWeight`.
    pub fn new(weight_type: WeightType, value: f64, bottleneck_threshold: f64) -> Self {
        Self {
            weight_type,
            value,
            bottleneck_threshold,
        }
    }

    /// Returns the weight type.
    pub fn weight_type(&self) -> &WeightType {
        &self.weight_type
    }

    /// Returns the raw numeric value.
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Returns `true` if `value` exceeds `bottleneck_threshold`.
    pub fn is_bottleneck(&self) -> bool {
        self.value > self.bottleneck_threshold
    }

    /// Sets a new value.
    pub fn set_value(&mut self, value: f64) {
        self.value = value;
    }
}

/// An edge in the graph that carries a weight between two node IDs.
#[derive(Debug, Clone)]
pub struct WeightedEdge {
    /// Source node ID.
    from: u64,
    /// Destination node ID.
    to: u64,
    /// Bandwidth weight for this edge in bps.
    bandwidth_bps: f64,
    /// Additional named weights.
    weights: Vec<EdgeWeight>,
}

impl WeightedEdge {
    /// Creates a new `WeightedEdge` between `from` and `to` with the given
    /// bandwidth in bps.
    pub fn new(from: u64, to: u64, bandwidth_bps: f64) -> Self {
        Self {
            from,
            to,
            bandwidth_bps,
            weights: Vec::new(),
        }
    }

    /// Returns the source node ID.
    pub fn from(&self) -> u64 {
        self.from
    }

    /// Returns the destination node ID.
    pub fn to(&self) -> u64 {
        self.to
    }

    /// Returns the bandwidth in bps for this edge.
    pub fn bandwidth_bps(&self) -> f64 {
        self.bandwidth_bps
    }

    /// Computes the ratio of this edge's bandwidth to a reference `total_bps`.
    /// Returns `0.0` if `total_bps` is zero.
    pub fn bandwidth_ratio(&self, total_bps: f64) -> f64 {
        if total_bps <= 0.0 {
            return 0.0;
        }
        self.bandwidth_bps / total_bps
    }

    /// Attaches an additional weight to this edge.
    pub fn add_weight(&mut self, weight: EdgeWeight) {
        self.weights.push(weight);
    }

    /// Returns a slice of all attached weights.
    pub fn weights(&self) -> &[EdgeWeight] {
        &self.weights
    }

    /// Returns `true` if any attached weight is a bottleneck.
    pub fn has_bottleneck(&self) -> bool {
        self.weights.iter().any(|w| w.is_bottleneck())
    }
}

/// A map of `WeightedEdge` instances keyed by `(from, to)` tuple.
#[derive(Debug, Clone, Default)]
pub struct EdgeWeightMap {
    edges: HashMap<(u64, u64), WeightedEdge>,
}

impl EdgeWeightMap {
    /// Creates an empty `EdgeWeightMap`.
    pub fn new() -> Self {
        Self {
            edges: HashMap::new(),
        }
    }

    /// Inserts a `WeightedEdge`. Overwrites any existing edge for `(from, to)`.
    pub fn insert(&mut self, edge: WeightedEdge) {
        self.edges.insert((edge.from(), edge.to()), edge);
    }

    /// Returns the edge between `from` and `to` if it exists.
    pub fn get(&self, from: u64, to: u64) -> Option<&WeightedEdge> {
        self.edges.get(&(from, to))
    }

    /// Returns the edge with the smallest `bandwidth_bps`, or `None` if empty.
    pub fn min_weight(&self) -> Option<&WeightedEdge> {
        self.edges.values().min_by(|a, b| {
            a.bandwidth_bps()
                .partial_cmp(&b.bandwidth_bps())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Returns the number of edges.
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    /// Returns `true` if no edges are registered.
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Returns all edges that are considered bottlenecks.
    pub fn bottleneck_edges(&self) -> Vec<&WeightedEdge> {
        self.edges.values().filter(|e| e.has_bottleneck()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_type_unit_bandwidth() {
        assert_eq!(WeightType::Bandwidth.unit(), "bps");
    }

    #[test]
    fn test_weight_type_unit_latency() {
        assert_eq!(WeightType::LatencyUs.unit(), "µs");
    }

    #[test]
    fn test_weight_type_unit_cost_empty() {
        assert_eq!(WeightType::Cost.unit(), "");
    }

    #[test]
    fn test_edge_weight_not_bottleneck() {
        let w = EdgeWeight::new(WeightType::Bandwidth, 100.0, 1000.0);
        assert!(!w.is_bottleneck());
    }

    #[test]
    fn test_edge_weight_is_bottleneck() {
        let w = EdgeWeight::new(WeightType::LatencyUs, 2000.0, 1000.0);
        assert!(w.is_bottleneck());
    }

    #[test]
    fn test_edge_weight_set_value() {
        let mut w = EdgeWeight::new(WeightType::Cost, 5.0, 10.0);
        w.set_value(15.0);
        assert!(w.is_bottleneck());
    }

    #[test]
    fn test_weighted_edge_bandwidth_ratio() {
        let e = WeightedEdge::new(0, 1, 500_000.0);
        assert!((e.bandwidth_ratio(1_000_000.0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_weighted_edge_bandwidth_ratio_zero_total() {
        let e = WeightedEdge::new(0, 1, 500_000.0);
        assert_eq!(e.bandwidth_ratio(0.0), 0.0);
    }

    #[test]
    fn test_weighted_edge_has_bottleneck_false() {
        let e = WeightedEdge::new(0, 1, 1_000_000.0);
        assert!(!e.has_bottleneck());
    }

    #[test]
    fn test_weighted_edge_has_bottleneck_true() {
        let mut e = WeightedEdge::new(0, 1, 1_000_000.0);
        e.add_weight(EdgeWeight::new(WeightType::LatencyUs, 5000.0, 1000.0));
        assert!(e.has_bottleneck());
    }

    #[test]
    fn test_edge_weight_map_insert_and_get() {
        let mut map = EdgeWeightMap::new();
        map.insert(WeightedEdge::new(0, 1, 100_000.0));
        assert!(map.get(0, 1).is_some());
        assert!(map.get(1, 0).is_none());
    }

    #[test]
    fn test_edge_weight_map_min_weight() {
        let mut map = EdgeWeightMap::new();
        map.insert(WeightedEdge::new(0, 1, 200_000.0));
        map.insert(WeightedEdge::new(1, 2, 50_000.0));
        let min = map.min_weight().expect("min_weight should succeed");
        assert!((min.bandwidth_bps() - 50_000.0).abs() < 1.0);
    }

    #[test]
    fn test_edge_weight_map_empty() {
        let map = EdgeWeightMap::new();
        assert!(map.is_empty());
        assert!(map.min_weight().is_none());
    }

    #[test]
    fn test_edge_weight_map_bottleneck_edges() {
        let mut map = EdgeWeightMap::new();
        let mut e = WeightedEdge::new(0, 1, 1_000_000.0);
        e.add_weight(EdgeWeight::new(WeightType::Cost, 999.0, 100.0));
        map.insert(e);
        map.insert(WeightedEdge::new(2, 3, 500_000.0));
        assert_eq!(map.bottleneck_edges().len(), 1);
    }
}
