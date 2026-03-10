//! Graph execution scheduler.
//!
//! Provides topological scheduling and resource-aware execution planning for
//! filter graphs.

#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};

// ─────────────────────────────────────────────────────────────────────────────
// GraphTopology trait
// ─────────────────────────────────────────────────────────────────────────────

/// Abstract view of a graph's topology needed for scheduling.
pub trait GraphTopology {
    /// Return all node IDs in the graph.
    fn nodes(&self) -> Vec<String>;
    /// Return IDs of nodes that `node_id` depends on (must run before it).
    fn dependencies(&self, node_id: &str) -> Vec<String>;
}

// ─────────────────────────────────────────────────────────────────────────────
// ExecutionOrder
// ─────────────────────────────────────────────────────────────────────────────

/// The result of topological scheduling.  Contains a flat execution order and
/// groups of nodes that can be executed in parallel.
#[derive(Debug, Clone)]
pub struct ExecutionOrder {
    /// Flat topological order of all node IDs.
    pub node_ids: Vec<String>,
    /// Groups of nodes with no mutual dependency that may run in parallel.
    pub parallelizable_groups: Vec<Vec<String>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// TopologicalScheduler
// ─────────────────────────────────────────────────────────────────────────────

/// Produces a topological execution order using Kahn's algorithm and groups
/// independent nodes into parallel execution groups.
pub struct TopologicalScheduler;

impl TopologicalScheduler {
    /// Schedule all nodes in the graph.
    ///
    /// Returns an [`ExecutionOrder`] on success, or an error string if the
    /// graph contains a cycle.
    ///
    /// # Errors
    /// Returns `Err` when a cycle is detected.
    pub fn schedule(graph: &dyn GraphTopology) -> Result<ExecutionOrder, String> {
        let all_nodes = graph.nodes();
        if all_nodes.is_empty() {
            return Ok(ExecutionOrder {
                node_ids: vec![],
                parallelizable_groups: vec![],
            });
        }

        // Build in-degree and adjacency list.
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut successors: HashMap<&str, Vec<&str>> = HashMap::new();

        for node in &all_nodes {
            in_degree.entry(node.as_str()).or_insert(0);
            successors.entry(node.as_str()).or_default();
        }

        for node in &all_nodes {
            for dep in graph.dependencies(node.as_str()) {
                // dep must run before node.
                if let Some(succ) = successors.get_mut(dep.as_str()) {
                    succ.push(node.as_str());
                }
                *in_degree.entry(node.as_str()).or_insert(0) += 1;
            }
        }

        // Kahn's algorithm – process in waves (each wave = parallel group).
        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut node_ids: Vec<String> = Vec::new();
        let mut parallelizable_groups: Vec<Vec<String>> = Vec::new();
        let mut remaining = all_nodes.len();

        // We process level by level to build parallel groups.
        let mut current_wave: Vec<&str> = queue.drain(..).collect();

        while !current_wave.is_empty() {
            // Sort for deterministic output.
            current_wave.sort_unstable();
            node_ids.extend(current_wave.iter().map(|s| s.to_string()));
            parallelizable_groups.push(current_wave.iter().map(|s| s.to_string()).collect());
            remaining -= current_wave.len();

            let mut next_wave: Vec<&str> = Vec::new();
            for &node in &current_wave {
                if let Some(succs) = successors.get(node) {
                    for &succ in succs {
                        // succ was inserted into in_degree during graph construction above,
                        // so this entry is guaranteed to exist.
                        let deg = in_degree.entry(succ).or_insert(0);
                        *deg -= 1;
                        if *deg == 0 {
                            next_wave.push(succ);
                        }
                    }
                }
            }
            current_wave = next_wave;
        }

        if remaining > 0 {
            return Err("Cycle detected in graph".to_string());
        }

        Ok(ExecutionOrder {
            node_ids,
            parallelizable_groups,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Resource-aware scheduling
// ─────────────────────────────────────────────────────────────────────────────

/// Hardware resource constraints for execution planning.
#[derive(Debug, Clone, Copy)]
pub struct ResourceConstraint {
    /// Maximum number of CPU threads that may be used simultaneously.
    pub max_cpu_threads: u32,
    /// Maximum memory that may be allocated simultaneously (in MB).
    pub max_memory_mb: u64,
    /// Whether a GPU is available.
    pub gpu_available: bool,
}

impl Default for ResourceConstraint {
    fn default() -> Self {
        Self {
            max_cpu_threads: 4,
            max_memory_mb: 2048,
            gpu_available: false,
        }
    }
}

/// A single stage of a resource-aware execution plan.
#[derive(Debug, Clone)]
pub struct ExecutionStage {
    /// Node IDs to execute in this stage.
    pub nodes: Vec<String>,
    /// Estimated CPU threads required.
    pub estimated_cpu_threads: u32,
    /// Estimated memory required (MB).
    pub estimated_memory_mb: u64,
}

/// A full resource-aware execution plan composed of ordered stages.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Ordered execution stages.
    pub stages: Vec<ExecutionStage>,
}

/// Converts a topological [`ExecutionOrder`] into an [`ExecutionPlan`] that
/// respects the given [`ResourceConstraint`].
///
/// Each parallelizable group becomes one or more stages, splitting groups that
/// would exceed the thread or memory limits.
pub struct ResourceAwareScheduler;

impl ResourceAwareScheduler {
    /// Build an execution plan from an execution order and resource constraints.
    ///
    /// Uses simple heuristics:
    /// - 1 CPU thread per node.
    /// - 64 MB estimated memory per node.
    #[must_use]
    pub fn schedule(order: &ExecutionOrder, constraints: &ResourceConstraint) -> ExecutionPlan {
        const MEM_PER_NODE_MB: u64 = 64;

        let mut stages: Vec<ExecutionStage> = Vec::new();

        for group in &order.parallelizable_groups {
            if group.is_empty() {
                continue;
            }

            // Split the group into chunks that fit within constraints.
            let max_nodes_by_threads = constraints.max_cpu_threads.max(1) as usize;
            let max_nodes_by_mem = (constraints.max_memory_mb / MEM_PER_NODE_MB).max(1) as usize;
            let chunk_size = max_nodes_by_threads.min(max_nodes_by_mem);

            for chunk in group.chunks(chunk_size) {
                let n = chunk.len() as u32;
                stages.push(ExecutionStage {
                    nodes: chunk.to_vec(),
                    estimated_cpu_threads: n,
                    estimated_memory_mb: n as u64 * MEM_PER_NODE_MB,
                });
            }
        }

        ExecutionPlan { stages }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Simple mock graph for testing.
    struct MockGraph {
        /// Adjacency list: node_id → list of dependency node_ids.
        deps: HashMap<String, Vec<String>>,
    }

    impl MockGraph {
        fn new() -> Self {
            Self {
                deps: HashMap::new(),
            }
        }

        fn add_node(&mut self, id: &str) {
            self.deps.entry(id.to_string()).or_default();
        }

        fn add_dep(&mut self, node: &str, dep: &str) {
            self.deps
                .entry(node.to_string())
                .or_default()
                .push(dep.to_string());
        }
    }

    impl GraphTopology for MockGraph {
        fn nodes(&self) -> Vec<String> {
            let mut ids: Vec<String> = self.deps.keys().cloned().collect();
            ids.sort(); // deterministic
            ids
        }

        fn dependencies(&self, node_id: &str) -> Vec<String> {
            self.deps.get(node_id).cloned().unwrap_or_default()
        }
    }

    // ── TopologicalScheduler ─────────────────────────────────────────────────

    #[test]
    fn test_schedule_empty_graph() {
        let graph = MockGraph::new();
        let order = TopologicalScheduler::schedule(&graph).expect("schedule should succeed");
        assert!(order.node_ids.is_empty());
        assert!(order.parallelizable_groups.is_empty());
    }

    #[test]
    fn test_schedule_single_node() {
        let mut graph = MockGraph::new();
        graph.add_node("a");
        let order = TopologicalScheduler::schedule(&graph).expect("schedule should succeed");
        assert_eq!(order.node_ids, vec!["a"]);
        assert_eq!(order.parallelizable_groups.len(), 1);
    }

    #[test]
    fn test_schedule_linear_chain() {
        // a → b → c
        let mut graph = MockGraph::new();
        graph.add_node("a");
        graph.add_node("b");
        graph.add_node("c");
        graph.add_dep("b", "a");
        graph.add_dep("c", "b");
        let order = TopologicalScheduler::schedule(&graph).expect("schedule should succeed");
        let pos = |id: &str| {
            order
                .node_ids
                .iter()
                .position(|x| x == id)
                .expect("iter should succeed")
        };
        assert!(pos("a") < pos("b"));
        assert!(pos("b") < pos("c"));
        // Each wave should be a single node.
        assert_eq!(order.parallelizable_groups.len(), 3);
    }

    #[test]
    fn test_schedule_parallel_nodes() {
        // a and b have no dependency → should be in the same group.
        let mut graph = MockGraph::new();
        graph.add_node("a");
        graph.add_node("b");
        let order = TopologicalScheduler::schedule(&graph).expect("schedule should succeed");
        assert_eq!(order.parallelizable_groups.len(), 1);
        assert_eq!(order.parallelizable_groups[0].len(), 2);
    }

    #[test]
    fn test_schedule_diamond() {
        // root → left, root → right, left → sink, right → sink
        let mut graph = MockGraph::new();
        graph.add_node("root");
        graph.add_node("left");
        graph.add_node("right");
        graph.add_node("sink");
        graph.add_dep("left", "root");
        graph.add_dep("right", "root");
        graph.add_dep("sink", "left");
        graph.add_dep("sink", "right");
        let order = TopologicalScheduler::schedule(&graph).expect("schedule should succeed");
        let pos = |id: &str| {
            order
                .node_ids
                .iter()
                .position(|x| x == id)
                .expect("iter should succeed")
        };
        assert!(pos("root") < pos("left"));
        assert!(pos("root") < pos("right"));
        assert!(pos("left") < pos("sink"));
        assert!(pos("right") < pos("sink"));
    }

    #[test]
    fn test_schedule_all_nodes_included() {
        let mut graph = MockGraph::new();
        for id in ["a", "b", "c", "d"] {
            graph.add_node(id);
        }
        graph.add_dep("b", "a");
        graph.add_dep("d", "c");
        let order = TopologicalScheduler::schedule(&graph).expect("schedule should succeed");
        let mut sorted = order.node_ids.clone();
        sorted.sort();
        assert_eq!(sorted, vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn test_schedule_cycle_detection() {
        // a → b → c → a (cycle)
        let mut graph = MockGraph::new();
        graph.add_node("a");
        graph.add_node("b");
        graph.add_node("c");
        graph.add_dep("b", "a");
        graph.add_dep("c", "b");
        graph.add_dep("a", "c"); // creates cycle
        let result = TopologicalScheduler::schedule(&graph);
        assert!(result.is_err(), "cycle should produce an error");
    }

    // ── ResourceAwareScheduler ────────────────────────────────────────────────

    #[test]
    fn test_resource_schedule_empty_order() {
        let order = ExecutionOrder {
            node_ids: vec![],
            parallelizable_groups: vec![],
        };
        let constraints = ResourceConstraint::default();
        let plan = ResourceAwareScheduler::schedule(&order, &constraints);
        assert!(plan.stages.is_empty());
    }

    #[test]
    fn test_resource_schedule_respects_thread_limit() {
        // 6 independent nodes with a 2-thread limit → at least 3 stages.
        let nodes: Vec<String> = (0..6).map(|i| format!("n{i}")).collect();
        let order = ExecutionOrder {
            node_ids: nodes.clone(),
            parallelizable_groups: vec![nodes],
        };
        let constraints = ResourceConstraint {
            max_cpu_threads: 2,
            max_memory_mb: 2048,
            gpu_available: false,
        };
        let plan = ResourceAwareScheduler::schedule(&order, &constraints);
        assert!(plan.stages.len() >= 3);
        for stage in &plan.stages {
            assert!(stage.estimated_cpu_threads <= 2);
        }
    }

    #[test]
    fn test_resource_schedule_stage_fields() {
        let order = ExecutionOrder {
            node_ids: vec!["a".to_string()],
            parallelizable_groups: vec![vec!["a".to_string()]],
        };
        let constraints = ResourceConstraint::default();
        let plan = ResourceAwareScheduler::schedule(&order, &constraints);
        assert_eq!(plan.stages.len(), 1);
        let stage = &plan.stages[0];
        assert_eq!(stage.nodes, vec!["a"]);
        assert!(stage.estimated_cpu_threads >= 1);
        assert!(stage.estimated_memory_mb > 0);
    }

    #[test]
    fn test_resource_schedule_all_nodes_covered() {
        let mut graph = MockGraph::new();
        for id in ["a", "b", "c"] {
            graph.add_node(id);
        }
        graph.add_dep("b", "a");
        let order = TopologicalScheduler::schedule(&graph).expect("schedule should succeed");
        let constraints = ResourceConstraint::default();
        let plan = ResourceAwareScheduler::schedule(&order, &constraints);
        let covered: HashSet<String> = plan.stages.iter().flat_map(|s| s.nodes.clone()).collect();
        assert!(covered.contains("a"));
        assert!(covered.contains("b"));
        assert!(covered.contains("c"));
    }
}
