//! Graph metrics: diameter, clustering coefficient, and betweenness centrality approximation.

use std::collections::{HashMap, HashSet, VecDeque};

/// A simple adjacency-list graph for metric computation.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct MetricsGraph {
    /// Adjacency list: node_id -> set of neighbor ids.
    adjacency: HashMap<usize, HashSet<usize>>,
    /// Whether the graph is directed.
    directed: bool,
}

impl MetricsGraph {
    /// Create a new directed metrics graph.
    #[allow(dead_code)]
    pub fn new_directed() -> Self {
        Self {
            adjacency: HashMap::new(),
            directed: true,
        }
    }

    /// Create a new undirected metrics graph.
    #[allow(dead_code)]
    pub fn new_undirected() -> Self {
        Self {
            adjacency: HashMap::new(),
            directed: false,
        }
    }

    /// Add a node.
    #[allow(dead_code)]
    pub fn add_node(&mut self, id: usize) {
        self.adjacency.entry(id).or_default();
    }

    /// Add an edge between two nodes. In undirected mode, adds both directions.
    #[allow(dead_code)]
    pub fn add_edge(&mut self, from: usize, to: usize) {
        self.adjacency.entry(from).or_default().insert(to);
        self.adjacency.entry(to).or_default(); // Ensure target node exists
        if !self.directed {
            self.adjacency.entry(to).or_default().insert(from);
        }
    }

    /// Returns the number of nodes.
    #[allow(dead_code)]
    pub fn node_count(&self) -> usize {
        self.adjacency.len()
    }

    /// Returns the number of edges.
    #[allow(dead_code)]
    pub fn edge_count(&self) -> usize {
        let total: usize = self.adjacency.values().map(|n| n.len()).sum();
        if self.directed {
            total
        } else {
            total / 2
        }
    }

    /// BFS shortest path distances from a source node.
    #[allow(dead_code)]
    fn bfs_distances(&self, source: usize) -> HashMap<usize, usize> {
        let mut dist = HashMap::new();
        let mut queue = VecDeque::new();
        dist.insert(source, 0usize);
        queue.push_back(source);

        while let Some(u) = queue.pop_front() {
            let d = dist[&u];
            if let Some(neighbors) = self.adjacency.get(&u) {
                for &v in neighbors {
                    if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(v) {
                        e.insert(d + 1);
                        queue.push_back(v);
                    }
                }
            }
        }
        dist
    }

    /// Compute the graph diameter (longest shortest path between any two nodes).
    /// Returns `None` if the graph is empty or disconnected (between the connected components).
    #[allow(dead_code)]
    pub fn diameter(&self) -> Option<usize> {
        if self.adjacency.is_empty() {
            return None;
        }

        let nodes: Vec<usize> = self.adjacency.keys().copied().collect();
        let mut max_dist = 0usize;
        let mut found_any = false;

        for &source in &nodes {
            let distances = self.bfs_distances(source);
            for (&target, &d) in &distances {
                if target != source {
                    found_any = true;
                    if d > max_dist {
                        max_dist = d;
                    }
                }
            }
        }

        if found_any {
            Some(max_dist)
        } else {
            None
        }
    }

    /// Compute the local clustering coefficient for a given node.
    ///
    /// For an undirected graph: C(v) = 2 * edges_between_neighbors / (k * (k-1))
    /// where k = degree of v.
    #[allow(dead_code)]
    pub fn local_clustering_coefficient(&self, node: usize) -> f64 {
        let neighbors = match self.adjacency.get(&node) {
            Some(n) => n,
            None => return 0.0,
        };

        let k = neighbors.len();
        if k < 2 {
            return 0.0;
        }

        let neighbors_vec: Vec<usize> = neighbors.iter().copied().collect();
        let mut triangle_edges = 0usize;

        for i in 0..k {
            for j in (i + 1)..k {
                let u = neighbors_vec[i];
                let v = neighbors_vec[j];
                // Check if u-v edge exists
                if let Some(u_neighbors) = self.adjacency.get(&u) {
                    if u_neighbors.contains(&v) {
                        triangle_edges += 1;
                    }
                }
            }
        }

        2.0 * triangle_edges as f64 / (k * (k - 1)) as f64
    }

    /// Compute the average clustering coefficient of the graph.
    #[allow(dead_code)]
    pub fn average_clustering_coefficient(&self) -> f64 {
        if self.adjacency.is_empty() {
            return 0.0;
        }
        let nodes: Vec<usize> = self.adjacency.keys().copied().collect();
        let sum: f64 = nodes
            .iter()
            .map(|&n| self.local_clustering_coefficient(n))
            .sum();
        sum / nodes.len() as f64
    }

    /// Approximate betweenness centrality using Brandes' algorithm (exact for small graphs).
    ///
    /// Returns a map of node_id -> centrality score (unnormalized).
    #[allow(dead_code)]
    pub fn betweenness_centrality(&self) -> HashMap<usize, f64> {
        let nodes: Vec<usize> = self.adjacency.keys().copied().collect();
        let mut centrality: HashMap<usize, f64> = nodes.iter().map(|&n| (n, 0.0)).collect();

        for &source in &nodes {
            // BFS to compute sigma (number of shortest paths) and dist
            let mut stack: Vec<usize> = Vec::new();
            let mut pred: HashMap<usize, Vec<usize>> =
                nodes.iter().map(|&n| (n, Vec::new())).collect();
            let mut sigma: HashMap<usize, f64> = nodes.iter().map(|&n| (n, 0.0)).collect();
            let mut dist: HashMap<usize, i64> = nodes.iter().map(|&n| (n, -1)).collect();

            sigma.insert(source, 1.0);
            dist.insert(source, 0);

            let mut queue = VecDeque::new();
            queue.push_back(source);

            while let Some(v) = queue.pop_front() {
                stack.push(v);
                let dv = dist[&v];

                if let Some(neighbors) = self.adjacency.get(&v) {
                    for &w in neighbors {
                        let dw = dist[&w];
                        if dw < 0 {
                            queue.push_back(w);
                            dist.insert(w, dv + 1);
                        }
                        if dist[&w] == dv + 1 {
                            let sw = sigma[&v];
                            // SAFETY: sigma and pred are pre-populated with all node ids
                            if let Some(sw_entry) = sigma.get_mut(&w) {
                                *sw_entry += sw;
                            }
                            if let Some(pred_entry) = pred.get_mut(&w) {
                                pred_entry.push(v);
                            }
                        }
                    }
                }
            }

            // Accumulation
            let mut delta: HashMap<usize, f64> = nodes.iter().map(|&n| (n, 0.0)).collect();
            while let Some(w) = stack.pop() {
                let sw = sigma[&w];
                let dw = delta[&w];
                for &v in &pred[&w] {
                    let sv = sigma[&v];
                    let contribution = (sv / sw) * (1.0 + dw);
                    // SAFETY: delta is pre-populated with all node ids
                    if let Some(dv_entry) = delta.get_mut(&v) {
                        *dv_entry += contribution;
                    }
                }
                if w != source {
                    let dw_val = delta[&w];
                    // SAFETY: centrality is pre-populated with all node ids
                    if let Some(c) = centrality.get_mut(&w) {
                        *c += dw_val;
                    }
                }
            }
        }

        centrality
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_path_graph(n: usize) -> MetricsGraph {
        let mut g = MetricsGraph::new_undirected();
        for i in 0..n {
            g.add_node(i);
        }
        for i in 0..n.saturating_sub(1) {
            g.add_edge(i, i + 1);
        }
        g
    }

    fn make_complete_graph(n: usize) -> MetricsGraph {
        let mut g = MetricsGraph::new_undirected();
        for i in 0..n {
            g.add_node(i);
        }
        for i in 0..n {
            for j in (i + 1)..n {
                g.add_edge(i, j);
            }
        }
        g
    }

    #[test]
    fn test_node_count() {
        let g = make_path_graph(4);
        assert_eq!(g.node_count(), 4);
    }

    #[test]
    fn test_edge_count_undirected() {
        let g = make_path_graph(4);
        assert_eq!(g.edge_count(), 3);
    }

    #[test]
    fn test_diameter_path_graph() {
        let g = make_path_graph(5);
        assert_eq!(g.diameter(), Some(4));
    }

    #[test]
    fn test_diameter_single_node() {
        let mut g = MetricsGraph::new_undirected();
        g.add_node(0);
        assert_eq!(g.diameter(), None);
    }

    #[test]
    fn test_diameter_empty() {
        let g = MetricsGraph::new_undirected();
        assert_eq!(g.diameter(), None);
    }

    #[test]
    fn test_diameter_complete_graph() {
        let g = make_complete_graph(4);
        assert_eq!(g.diameter(), Some(1));
    }

    #[test]
    fn test_clustering_coefficient_complete_graph() {
        let g = make_complete_graph(4);
        // Complete graph: every node's neighborhood is fully connected
        for i in 0..4 {
            let cc = g.local_clustering_coefficient(i);
            assert!((cc - 1.0).abs() < 1e-10, "node {i}: cc={cc}");
        }
    }

    #[test]
    fn test_clustering_coefficient_path_graph() {
        // Path graph has no triangles, so CC = 0 for all degree-2 nodes
        let g = make_path_graph(5);
        let cc = g.local_clustering_coefficient(2);
        assert_eq!(cc, 0.0);
    }

    #[test]
    fn test_clustering_coefficient_low_degree() {
        let g = make_path_graph(4);
        // Endpoint has degree 1, CC = 0
        let cc = g.local_clustering_coefficient(0);
        assert_eq!(cc, 0.0);
    }

    #[test]
    fn test_average_clustering_complete() {
        let g = make_complete_graph(4);
        let avg = g.average_clustering_coefficient();
        assert!((avg - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_average_clustering_empty() {
        let g = MetricsGraph::new_undirected();
        assert_eq!(g.average_clustering_coefficient(), 0.0);
    }

    #[test]
    fn test_betweenness_centrality_path() {
        // In a path 0-1-2-3-4, node 2 (middle) should have highest centrality
        let g = make_path_graph(5);
        let bc = g.betweenness_centrality();
        let mid = bc[&2];
        let end = bc[&0];
        assert!(
            mid > end,
            "middle node should have higher betweenness than endpoint"
        );
    }

    #[test]
    fn test_betweenness_centrality_complete_symmetric() {
        let g = make_complete_graph(4);
        let bc = g.betweenness_centrality();
        // In a complete graph all centralities should be equal
        let vals: Vec<f64> = bc.values().copied().collect();
        let first = vals[0];
        for v in &vals {
            assert!(
                (v - first).abs() < 1e-9,
                "values should be equal: {first} vs {v}"
            );
        }
    }

    #[test]
    fn test_directed_graph_edge_count() {
        let mut g = MetricsGraph::new_directed();
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        assert_eq!(g.edge_count(), 2);
    }
}
