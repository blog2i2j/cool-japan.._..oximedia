#![allow(dead_code)]

//! Route optimization for media signal paths.
//!
//! Evaluates candidate routes and selects the best one based on
//! configurable optimization goals such as minimal latency,
//! maximum reliability, or lowest cost.

/// Optimization goal for route selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationGoal {
    /// Minimize end-to-end latency.
    MinLatency,
    /// Maximize path reliability (fewest hops / highest uptime).
    MaxReliability,
    /// Minimize monetary cost.
    MinCost,
    /// Balance latency and reliability equally.
    Balanced,
}

/// A candidate route to evaluate.
#[derive(Debug, Clone)]
pub struct CandidateRoute {
    /// Route identifier.
    pub id: String,
    /// Estimated latency in microseconds.
    pub latency_us: f64,
    /// Reliability score 0.0..=1.0.
    pub reliability: f64,
    /// Cost per hour of use.
    pub cost_per_hour: f64,
    /// Number of hops.
    pub hop_count: u32,
}

impl CandidateRoute {
    /// Create a new candidate route.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            latency_us: 0.0,
            reliability: 1.0,
            cost_per_hour: 0.0,
            hop_count: 1,
        }
    }

    /// Set latency.
    pub fn with_latency_us(mut self, us: f64) -> Self {
        self.latency_us = us;
        self
    }

    /// Set reliability.
    pub fn with_reliability(mut self, r: f64) -> Self {
        self.reliability = r.clamp(0.0, 1.0);
        self
    }

    /// Set cost.
    pub fn with_cost(mut self, c: f64) -> Self {
        self.cost_per_hour = c;
        self
    }

    /// Set hop count.
    pub fn with_hops(mut self, h: u32) -> Self {
        self.hop_count = h;
        self
    }
}

/// Result of an optimization run.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// The chosen route id.
    pub chosen_id: String,
    /// Composite score (higher is better).
    pub score: f64,
    /// All scored candidates sorted best-first.
    pub ranked: Vec<(String, f64)>,
}

/// Route optimizer engine.
#[derive(Debug, Clone)]
pub struct RouteOptimizer {
    goal: OptimizationGoal,
    /// Weight for latency component in balanced mode (0..=1).
    latency_weight: f64,
    /// Weight for reliability component in balanced mode (0..=1).
    reliability_weight: f64,
    /// Weight for cost component in balanced mode (0..=1).
    cost_weight: f64,
    /// Maximum acceptable latency (0 = no limit).
    max_latency_us: f64,
    /// Minimum acceptable reliability (0 = no limit).
    min_reliability: f64,
}

impl RouteOptimizer {
    /// Create a new optimizer with the given goal.
    pub fn new(goal: OptimizationGoal) -> Self {
        Self {
            goal,
            latency_weight: 0.4,
            reliability_weight: 0.4,
            cost_weight: 0.2,
            max_latency_us: 0.0,
            min_reliability: 0.0,
        }
    }

    /// Set balanced-mode weights. They need not sum to 1; they are normalized.
    #[allow(clippy::cast_precision_loss)]
    pub fn with_weights(mut self, latency: f64, reliability: f64, cost: f64) -> Self {
        let total = latency + reliability + cost;
        if total > 0.0 {
            self.latency_weight = latency / total;
            self.reliability_weight = reliability / total;
            self.cost_weight = cost / total;
        }
        self
    }

    /// Set a hard latency constraint.
    pub fn with_max_latency(mut self, us: f64) -> Self {
        self.max_latency_us = us;
        self
    }

    /// Set a hard reliability constraint.
    pub fn with_min_reliability(mut self, r: f64) -> Self {
        self.min_reliability = r;
        self
    }

    /// Score a single candidate. Higher is better.
    #[allow(clippy::cast_precision_loss)]
    fn score(&self, route: &CandidateRoute) -> f64 {
        match self.goal {
            OptimizationGoal::MinLatency => {
                if route.latency_us <= 0.0 {
                    return f64::MAX;
                }
                1_000_000.0 / route.latency_us
            }
            OptimizationGoal::MaxReliability => route.reliability * 1000.0,
            OptimizationGoal::MinCost => {
                if route.cost_per_hour <= 0.0 {
                    return f64::MAX;
                }
                1000.0 / route.cost_per_hour
            }
            OptimizationGoal::Balanced => {
                let lat_score = if route.latency_us > 0.0 {
                    1_000_000.0 / route.latency_us
                } else {
                    1_000_000.0
                };
                let rel_score = route.reliability * 1000.0;
                let cost_score = if route.cost_per_hour > 0.0 {
                    1000.0 / route.cost_per_hour
                } else {
                    1000.0
                };
                self.latency_weight * lat_score
                    + self.reliability_weight * rel_score
                    + self.cost_weight * cost_score
            }
        }
    }

    /// Filter candidates that violate hard constraints.
    fn filter<'a>(&self, routes: &'a [CandidateRoute]) -> Vec<&'a CandidateRoute> {
        routes
            .iter()
            .filter(|r| {
                if self.max_latency_us > 0.0 && r.latency_us > self.max_latency_us {
                    return false;
                }
                if self.min_reliability > 0.0 && r.reliability < self.min_reliability {
                    return false;
                }
                true
            })
            .collect()
    }

    /// Optimize: pick the best route from candidates.
    /// Returns `None` if no candidates survive filtering.
    pub fn optimize(&self, candidates: &[CandidateRoute]) -> Option<OptimizationResult> {
        let filtered = self.filter(candidates);
        if filtered.is_empty() {
            return None;
        }

        let mut scored: Vec<(String, f64)> = filtered
            .iter()
            .map(|r| (r.id.clone(), self.score(r)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let best = scored[0].clone();
        Some(OptimizationResult {
            chosen_id: best.0,
            score: best.1,
            ranked: scored,
        })
    }

    /// Current optimization goal.
    pub fn goal(&self) -> OptimizationGoal {
        self.goal
    }
}

// ============================================================================
// Graph-based shortest path routing
// ============================================================================

/// A node in a routing graph (switch, router, endpoint).
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Node identifier.
    pub id: String,
    /// Human-readable label.
    pub label: String,
    /// Current load as a fraction (0.0 = idle, 1.0 = fully loaded).
    pub load: f64,
    /// Maximum throughput capacity (arbitrary units).
    pub capacity: f64,
}

impl GraphNode {
    /// Create a new graph node.
    pub fn new(id: impl Into<String>, label: impl Into<String>, capacity: f64) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            load: 0.0,
            capacity,
        }
    }

    /// Available capacity (capacity * (1 - load)).
    #[must_use]
    pub fn available_capacity(&self) -> f64 {
        self.capacity * (1.0 - self.load)
    }
}

/// A directed edge in the routing graph.
#[derive(Debug, Clone)]
pub struct GraphEdge {
    /// Source node ID.
    pub from: String,
    /// Destination node ID.
    pub to: String,
    /// Latency cost of this edge in microseconds.
    pub latency_us: f64,
    /// Bandwidth capacity of this link.
    pub bandwidth: f64,
    /// Current utilization (0.0 to 1.0).
    pub utilization: f64,
}

impl GraphEdge {
    /// Create a new edge.
    pub fn new(
        from: impl Into<String>,
        to: impl Into<String>,
        latency_us: f64,
        bandwidth: f64,
    ) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            latency_us,
            bandwidth,
            utilization: 0.0,
        }
    }

    /// Effective cost considering utilization (higher utilization = higher cost).
    /// Uses an exponential penalty to discourage near-capacity links.
    #[must_use]
    pub fn effective_cost(&self) -> f64 {
        // Base cost is latency; add congestion penalty
        let congestion_factor = 1.0 / (1.0 - self.utilization.min(0.99));
        self.latency_us * congestion_factor
    }
}

/// A routing graph for shortest-path and load-balancing computations.
#[derive(Debug, Default)]
pub struct RoutingGraph {
    /// Nodes indexed by ID.
    nodes: std::collections::HashMap<String, GraphNode>,
    /// Edges (adjacency list keyed by source node ID).
    edges: std::collections::HashMap<String, Vec<GraphEdge>>,
}

impl RoutingGraph {
    /// Create an empty routing graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: GraphNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    /// Add a directed edge to the graph.
    pub fn add_edge(&mut self, edge: GraphEdge) {
        self.edges.entry(edge.from.clone()).or_default().push(edge);
    }

    /// Add a bidirectional edge (two directed edges with same properties).
    pub fn add_bidirectional_edge(
        &mut self,
        node_a: impl Into<String>,
        node_b: impl Into<String>,
        latency_us: f64,
        bandwidth: f64,
    ) {
        let a = node_a.into();
        let b = node_b.into();
        self.add_edge(GraphEdge::new(&a, &b, latency_us, bandwidth));
        self.add_edge(GraphEdge::new(&b, &a, latency_us, bandwidth));
    }

    /// Get the number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the total number of directed edges.
    pub fn edge_count(&self) -> usize {
        self.edges.values().map(|v| v.len()).sum()
    }

    /// Dijkstra's shortest path algorithm using effective cost.
    ///
    /// Returns the shortest path from `source` to `destination` as a list of
    /// node IDs, along with the total cost. Returns `None` if no path exists.
    pub fn shortest_path(&self, source: &str, destination: &str) -> Option<(Vec<String>, f64)> {
        use std::cmp::Ordering;
        use std::collections::{BinaryHeap, HashMap};

        // Ensure both endpoints exist
        if !self.nodes.contains_key(source) || !self.nodes.contains_key(destination) {
            return None;
        }

        #[derive(Debug)]
        struct State {
            cost: f64,
            node: String,
        }

        impl PartialEq for State {
            fn eq(&self, other: &Self) -> bool {
                self.cost == other.cost
            }
        }

        impl Eq for State {}

        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for State {
            fn cmp(&self, other: &Self) -> Ordering {
                // Reverse ordering for min-heap behavior
                other
                    .cost
                    .partial_cmp(&self.cost)
                    .unwrap_or(Ordering::Equal)
            }
        }

        let mut dist: HashMap<String, f64> = HashMap::new();
        let mut prev: HashMap<String, String> = HashMap::new();
        let mut heap = BinaryHeap::new();

        dist.insert(source.to_string(), 0.0);
        heap.push(State {
            cost: 0.0,
            node: source.to_string(),
        });

        while let Some(State { cost, node }) = heap.pop() {
            if node == destination {
                // Reconstruct path
                let mut path = Vec::new();
                let mut current = destination.to_string();
                path.push(current.clone());

                while let Some(p) = prev.get(&current) {
                    path.push(p.clone());
                    current = p.clone();
                }

                path.reverse();
                return Some((path, cost));
            }

            // Skip if we've already found a better path
            if let Some(&best) = dist.get(&node) {
                if cost > best {
                    continue;
                }
            }

            // Explore neighbors
            if let Some(edges) = self.edges.get(&node) {
                for edge in edges {
                    let next_cost = cost + edge.effective_cost();

                    let is_better = dist
                        .get(&edge.to)
                        .map_or(true, |&current_best| next_cost < current_best);

                    if is_better {
                        dist.insert(edge.to.clone(), next_cost);
                        prev.insert(edge.to.clone(), node.clone());
                        heap.push(State {
                            cost: next_cost,
                            node: edge.to.clone(),
                        });
                    }
                }
            }
        }

        None // No path found
    }

    /// Find multiple disjoint paths for load balancing.
    ///
    /// Uses iterative shortest-path with edge removal to find up to `k`
    /// edge-disjoint paths. Returns paths sorted by cost (cheapest first).
    pub fn k_shortest_paths(
        &self,
        source: &str,
        destination: &str,
        k: usize,
    ) -> Vec<(Vec<String>, f64)> {
        let mut results = Vec::new();
        let mut used_edges: std::collections::HashSet<(String, String)> =
            std::collections::HashSet::new();

        for _ in 0..k {
            // Build a temporary graph excluding used edges
            let path = self.shortest_path_excluding(source, destination, &used_edges);

            if let Some((path, cost)) = path {
                // Mark edges in this path as used
                for window in path.windows(2) {
                    used_edges.insert((window[0].clone(), window[1].clone()));
                }
                results.push((path, cost));
            } else {
                break; // No more paths available
            }
        }

        results
    }

    /// Shortest path excluding certain edges.
    fn shortest_path_excluding(
        &self,
        source: &str,
        destination: &str,
        excluded: &std::collections::HashSet<(String, String)>,
    ) -> Option<(Vec<String>, f64)> {
        use std::cmp::Ordering;
        use std::collections::{BinaryHeap, HashMap};

        if !self.nodes.contains_key(source) || !self.nodes.contains_key(destination) {
            return None;
        }

        #[derive(Debug)]
        struct State {
            cost: f64,
            node: String,
        }

        impl PartialEq for State {
            fn eq(&self, other: &Self) -> bool {
                self.cost == other.cost
            }
        }

        impl Eq for State {}

        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for State {
            fn cmp(&self, other: &Self) -> Ordering {
                other
                    .cost
                    .partial_cmp(&self.cost)
                    .unwrap_or(Ordering::Equal)
            }
        }

        let mut dist: HashMap<String, f64> = HashMap::new();
        let mut prev: HashMap<String, String> = HashMap::new();
        let mut heap = BinaryHeap::new();

        dist.insert(source.to_string(), 0.0);
        heap.push(State {
            cost: 0.0,
            node: source.to_string(),
        });

        while let Some(State { cost, node }) = heap.pop() {
            if node == destination {
                let mut path = Vec::new();
                let mut current = destination.to_string();
                path.push(current.clone());

                while let Some(p) = prev.get(&current) {
                    path.push(p.clone());
                    current = p.clone();
                }

                path.reverse();
                return Some((path, cost));
            }

            if let Some(&best) = dist.get(&node) {
                if cost > best {
                    continue;
                }
            }

            if let Some(edges) = self.edges.get(&node) {
                for edge in edges {
                    // Skip excluded edges
                    if excluded.contains(&(edge.from.clone(), edge.to.clone())) {
                        continue;
                    }

                    let next_cost = cost + edge.effective_cost();

                    let is_better = dist
                        .get(&edge.to)
                        .map_or(true, |&current_best| next_cost < current_best);

                    if is_better {
                        dist.insert(edge.to.clone(), next_cost);
                        prev.insert(edge.to.clone(), node.clone());
                        heap.push(State {
                            cost: next_cost,
                            node: edge.to.clone(),
                        });
                    }
                }
            }
        }

        None
    }

    /// Load-balanced route selection.
    ///
    /// Finds the path with the least congested links (lowest max utilization
    /// along the path). This is the "widest path" variant that maximizes
    /// minimum available bandwidth.
    pub fn least_congested_path(
        &self,
        source: &str,
        destination: &str,
    ) -> Option<(Vec<String>, f64)> {
        // Find multiple candidate paths and pick the one with lowest peak utilization
        let candidates = self.k_shortest_paths(source, destination, 5);

        candidates
            .into_iter()
            .map(|(path, _cost)| {
                let max_util = self.path_max_utilization(&path);
                (path, max_util)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get the maximum edge utilization along a path.
    fn path_max_utilization(&self, path: &[String]) -> f64 {
        let mut max_util = 0.0f64;
        for window in path.windows(2) {
            if let Some(edges) = self.edges.get(&window[0]) {
                for edge in edges {
                    if edge.to == window[1] {
                        max_util = max_util.max(edge.utilization);
                    }
                }
            }
        }
        max_util
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn routes() -> Vec<CandidateRoute> {
        vec![
            CandidateRoute::new("fast")
                .with_latency_us(100.0)
                .with_reliability(0.95)
                .with_cost(10.0)
                .with_hops(2),
            CandidateRoute::new("reliable")
                .with_latency_us(500.0)
                .with_reliability(0.999)
                .with_cost(20.0)
                .with_hops(3),
            CandidateRoute::new("cheap")
                .with_latency_us(1000.0)
                .with_reliability(0.90)
                .with_cost(2.0)
                .with_hops(5),
        ]
    }

    #[test]
    fn test_min_latency_picks_fastest() {
        let opt = RouteOptimizer::new(OptimizationGoal::MinLatency);
        let result = opt.optimize(&routes()).expect("should succeed in test");
        assert_eq!(result.chosen_id, "fast");
    }

    #[test]
    fn test_max_reliability_picks_most_reliable() {
        let opt = RouteOptimizer::new(OptimizationGoal::MaxReliability);
        let result = opt.optimize(&routes()).expect("should succeed in test");
        assert_eq!(result.chosen_id, "reliable");
    }

    #[test]
    fn test_min_cost_picks_cheapest() {
        let opt = RouteOptimizer::new(OptimizationGoal::MinCost);
        let result = opt.optimize(&routes()).expect("should succeed in test");
        assert_eq!(result.chosen_id, "cheap");
    }

    #[test]
    fn test_balanced_returns_result() {
        let opt = RouteOptimizer::new(OptimizationGoal::Balanced);
        let result = opt.optimize(&routes());
        assert!(result.is_some());
    }

    #[test]
    fn test_empty_candidates() {
        let opt = RouteOptimizer::new(OptimizationGoal::MinLatency);
        assert!(opt.optimize(&[]).is_none());
    }

    #[test]
    fn test_latency_constraint_filters() {
        let opt = RouteOptimizer::new(OptimizationGoal::MinCost).with_max_latency(200.0);
        let result = opt.optimize(&routes()).expect("should succeed in test");
        // Only "fast" has latency <= 200
        assert_eq!(result.chosen_id, "fast");
    }

    #[test]
    fn test_reliability_constraint_filters() {
        let opt = RouteOptimizer::new(OptimizationGoal::MinCost).with_min_reliability(0.99);
        let result = opt.optimize(&routes()).expect("should succeed in test");
        // Only "reliable" has reliability >= 0.99
        assert_eq!(result.chosen_id, "reliable");
    }

    #[test]
    fn test_all_filtered_returns_none() {
        let opt = RouteOptimizer::new(OptimizationGoal::MinLatency).with_max_latency(1.0); // nothing is under 1us
        assert!(opt.optimize(&routes()).is_none());
    }

    #[test]
    fn test_ranked_ordering() {
        let opt = RouteOptimizer::new(OptimizationGoal::MinLatency);
        let result = opt.optimize(&routes()).expect("should succeed in test");
        // Scores should be descending
        for w in result.ranked.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    #[test]
    fn test_custom_weights() {
        let opt = RouteOptimizer::new(OptimizationGoal::Balanced).with_weights(0.0, 0.0, 1.0); // only cost matters
        let result = opt.optimize(&routes()).expect("should succeed in test");
        assert_eq!(result.chosen_id, "cheap");
    }

    #[test]
    fn test_goal_accessor() {
        let opt = RouteOptimizer::new(OptimizationGoal::MaxReliability);
        assert_eq!(opt.goal(), OptimizationGoal::MaxReliability);
    }

    #[test]
    fn test_candidate_builder() {
        let r = CandidateRoute::new("test")
            .with_latency_us(42.0)
            .with_reliability(0.99)
            .with_cost(5.0)
            .with_hops(3);
        assert_eq!(r.id, "test");
        assert!((r.latency_us - 42.0).abs() < f64::EPSILON);
        assert_eq!(r.hop_count, 3);
    }

    #[test]
    fn test_reliability_clamped() {
        let r = CandidateRoute::new("over").with_reliability(1.5);
        assert!((r.reliability - 1.0).abs() < f64::EPSILON);
        let r2 = CandidateRoute::new("under").with_reliability(-0.5);
        assert!(r2.reliability.abs() < f64::EPSILON);
    }

    #[test]
    fn test_single_candidate_always_chosen() {
        let opt = RouteOptimizer::new(OptimizationGoal::Balanced);
        let cands = vec![CandidateRoute::new("only").with_latency_us(500.0)];
        let result = opt.optimize(&cands).expect("should succeed in test");
        assert_eq!(result.chosen_id, "only");
    }

    #[test]
    fn test_zero_latency_route_scores_high() {
        let opt = RouteOptimizer::new(OptimizationGoal::MinLatency);
        let cands = vec![
            CandidateRoute::new("zero_lat").with_latency_us(0.0),
            CandidateRoute::new("some_lat").with_latency_us(100.0),
        ];
        let result = opt.optimize(&cands).expect("should succeed in test");
        assert_eq!(result.chosen_id, "zero_lat");
    }

    #[test]
    fn test_optimization_result_score_positive() {
        let opt = RouteOptimizer::new(OptimizationGoal::MaxReliability);
        let result = opt.optimize(&routes()).expect("should succeed in test");
        assert!(result.score > 0.0);
    }

    // Graph-based routing tests

    fn build_test_graph() -> RoutingGraph {
        let mut graph = RoutingGraph::new();
        graph.add_node(GraphNode::new("A", "Source", 1000.0));
        graph.add_node(GraphNode::new("B", "Switch-1", 1000.0));
        graph.add_node(GraphNode::new("C", "Switch-2", 1000.0));
        graph.add_node(GraphNode::new("D", "Destination", 1000.0));

        // A -> B (fast, direct)
        graph.add_edge(GraphEdge::new("A", "B", 100.0, 1000.0));
        // B -> D (fast)
        graph.add_edge(GraphEdge::new("B", "D", 100.0, 1000.0));
        // A -> C (slower)
        graph.add_edge(GraphEdge::new("A", "C", 200.0, 1000.0));
        // C -> D (slower)
        graph.add_edge(GraphEdge::new("C", "D", 200.0, 1000.0));
        // B -> C (cross link)
        graph.add_edge(GraphEdge::new("B", "C", 50.0, 500.0));

        graph
    }

    #[test]
    fn test_graph_node_available_capacity() {
        let mut node = GraphNode::new("n1", "Node", 1000.0);
        node.load = 0.3;
        assert!((node.available_capacity() - 700.0).abs() < 1e-6);
    }

    #[test]
    fn test_graph_edge_effective_cost() {
        let edge = GraphEdge::new("a", "b", 100.0, 1000.0);
        // Zero utilization: cost = latency * 1/(1-0) = 100
        assert!((edge.effective_cost() - 100.0).abs() < 1e-6);

        let mut congested = GraphEdge::new("a", "b", 100.0, 1000.0);
        congested.utilization = 0.5;
        // cost = 100 * 1/(1-0.5) = 200
        assert!((congested.effective_cost() - 200.0).abs() < 1e-6);
    }

    #[test]
    fn test_routing_graph_counts() {
        let graph = build_test_graph();
        assert_eq!(graph.node_count(), 4);
        assert_eq!(graph.edge_count(), 5);
    }

    #[test]
    fn test_shortest_path_direct() {
        let graph = build_test_graph();
        let result = graph.shortest_path("A", "D");
        assert!(result.is_some());
        let (path, cost) = result.expect("should have path");
        // Should prefer A -> B -> D (200us) over A -> C -> D (400us)
        assert_eq!(path, vec!["A", "B", "D"]);
        assert!((cost - 200.0).abs() < 1e-6);
    }

    #[test]
    fn test_shortest_path_no_path() {
        let mut graph = RoutingGraph::new();
        graph.add_node(GraphNode::new("A", "Start", 100.0));
        graph.add_node(GraphNode::new("B", "End", 100.0));
        // No edges
        assert!(graph.shortest_path("A", "B").is_none());
    }

    #[test]
    fn test_shortest_path_unknown_node() {
        let graph = build_test_graph();
        assert!(graph.shortest_path("X", "D").is_none());
        assert!(graph.shortest_path("A", "Z").is_none());
    }

    #[test]
    fn test_k_shortest_paths() {
        let graph = build_test_graph();
        let paths = graph.k_shortest_paths("A", "D", 3);
        assert!(paths.len() >= 2, "Should find at least 2 paths");
    }

    #[test]
    fn test_least_congested_path() {
        let mut graph = RoutingGraph::new();
        graph.add_node(GraphNode::new("A", "Src", 1000.0));
        graph.add_node(GraphNode::new("B", "SW1", 1000.0));
        graph.add_node(GraphNode::new("C", "SW2", 1000.0));
        graph.add_node(GraphNode::new("D", "Dst", 1000.0));

        // Path A-B-D: fast but congested
        let mut e1 = GraphEdge::new("A", "B", 100.0, 1000.0);
        e1.utilization = 0.9; // Very congested
        graph.add_edge(e1);
        let mut e2 = GraphEdge::new("B", "D", 100.0, 1000.0);
        e2.utilization = 0.8;
        graph.add_edge(e2);

        // Path A-C-D: slower but less congested
        let mut e3 = GraphEdge::new("A", "C", 200.0, 1000.0);
        e3.utilization = 0.1;
        graph.add_edge(e3);
        let mut e4 = GraphEdge::new("C", "D", 200.0, 1000.0);
        e4.utilization = 0.2;
        graph.add_edge(e4);

        let result = graph.least_congested_path("A", "D");
        assert!(result.is_some());
        let (path, max_util) = result.expect("should find path");
        // Should prefer the less congested path A-C-D
        assert_eq!(path, vec!["A", "C", "D"]);
        assert!(max_util < 0.5, "Max utilization should be low: {max_util}");
    }

    #[test]
    fn test_bidirectional_edge() {
        let mut graph = RoutingGraph::new();
        graph.add_node(GraphNode::new("A", "A", 100.0));
        graph.add_node(GraphNode::new("B", "B", 100.0));
        graph.add_bidirectional_edge("A", "B", 50.0, 500.0);

        assert_eq!(graph.edge_count(), 2);
        assert!(graph.shortest_path("A", "B").is_some());
        assert!(graph.shortest_path("B", "A").is_some());
    }

    #[test]
    fn test_same_source_destination() {
        let graph = build_test_graph();
        let result = graph.shortest_path("A", "A");
        assert!(result.is_some());
        let (path, cost) = result.expect("should succeed");
        assert_eq!(path, vec!["A"]);
        assert!((cost - 0.0).abs() < 1e-6);
    }
}
