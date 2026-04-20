//! Workflow simulation (dry-run) engine.
//!
//! Traces the execution path of a workflow DAG without actually running any
//! tasks. Useful for pre-flight validation, capacity planning, cost estimation,
//! and debugging complex conditional workflows.
//!
//! The simulator walks the DAG in topological order, evaluates edge conditions
//! against simulated outputs, and produces a detailed execution trace showing
//! which nodes would run, be skipped, or potentially fail.

use crate::dag::{DagError, NodeId, WorkflowDag};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Duration;

/// Configuration for a simulation run.
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Simulated outputs per node task type. When a node of this task type
    /// runs, these outputs are written to its output map.
    pub simulated_outputs: HashMap<String, HashMap<String, serde_json::Value>>,
    /// Task types that should be marked as failed during simulation.
    pub failing_task_types: HashSet<String>,
    /// Estimated duration per task type (for cost/time estimation).
    pub estimated_durations: HashMap<String, Duration>,
    /// Whether to continue tracing after a simulated failure.
    pub continue_on_failure: bool,
    /// Maximum number of nodes to trace (safety limit).
    pub max_nodes: usize,
    /// Condition evaluator: maps condition string to boolean result.
    /// If not provided, all conditions default to true.
    pub condition_overrides: HashMap<String, bool>,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            simulated_outputs: HashMap::new(),
            failing_task_types: HashSet::new(),
            estimated_durations: HashMap::new(),
            continue_on_failure: true,
            max_nodes: 10_000,
            condition_overrides: HashMap::new(),
        }
    }
}

impl SimulationConfig {
    /// Create a new empty simulation config.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add simulated outputs for a task type.
    #[must_use]
    pub fn with_outputs(
        mut self,
        task_type: impl Into<String>,
        outputs: HashMap<String, serde_json::Value>,
    ) -> Self {
        self.simulated_outputs.insert(task_type.into(), outputs);
        self
    }

    /// Mark a task type as failing.
    #[must_use]
    pub fn with_failing_type(mut self, task_type: impl Into<String>) -> Self {
        self.failing_task_types.insert(task_type.into());
        self
    }

    /// Set estimated duration for a task type.
    #[must_use]
    pub fn with_duration(mut self, task_type: impl Into<String>, duration: Duration) -> Self {
        self.estimated_durations.insert(task_type.into(), duration);
        self
    }

    /// Set whether to continue after failure.
    #[must_use]
    pub fn with_continue_on_failure(mut self, cont: bool) -> Self {
        self.continue_on_failure = cont;
        self
    }

    /// Override a condition expression to return a specific boolean.
    #[must_use]
    pub fn with_condition_override(mut self, condition: impl Into<String>, value: bool) -> Self {
        self.condition_overrides.insert(condition.into(), value);
        self
    }
}

/// A single entry in the simulation trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationTraceEntry {
    /// Node that was evaluated.
    pub node_id: NodeId,
    /// Task type of the node.
    pub task_type: String,
    /// Outcome of the simulated execution.
    pub outcome: SimulatedOutcome,
    /// Estimated duration (if configured).
    pub estimated_duration: Option<Duration>,
    /// Dependencies that were checked.
    pub dependencies: Vec<NodeId>,
    /// Edge conditions that were evaluated.
    pub evaluated_conditions: Vec<EvaluatedCondition>,
    /// Order in which this node was visited (0-based).
    pub visit_order: usize,
}

/// Outcome of a simulated node execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimulatedOutcome {
    /// Node would execute successfully.
    WouldSucceed,
    /// Node would fail.
    WouldFail {
        /// Reason for failure.
        reason: String,
    },
    /// Node would be skipped due to unsatisfied conditions.
    WouldSkip {
        /// Reason for skipping.
        reason: String,
    },
    /// Node not reachable from root nodes.
    Unreachable,
    /// Node blocked because a dependency failed and continue_on_failure is false.
    Blocked {
        /// The failed dependency.
        failed_dependency: NodeId,
    },
}

/// Result of evaluating an edge condition during simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluatedCondition {
    /// The condition expression.
    pub condition: String,
    /// The result of evaluation.
    pub result: bool,
    /// Source of the result (override or default).
    pub source: ConditionSource,
}

/// How a condition was resolved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConditionSource {
    /// Resolved from a config override.
    Override,
    /// Defaulted to true (no override provided).
    DefaultTrue,
}

/// Complete result of a simulation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    /// Ordered trace of all visited nodes.
    pub trace: Vec<SimulationTraceEntry>,
    /// Summary statistics.
    pub summary: SimulationSummary,
    /// Final node statuses.
    pub node_statuses: HashMap<NodeId, SimulatedOutcome>,
}

/// Summary statistics from a simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationSummary {
    /// Total nodes in the DAG.
    pub total_nodes: usize,
    /// Nodes that would succeed.
    pub would_succeed: usize,
    /// Nodes that would fail.
    pub would_fail: usize,
    /// Nodes that would be skipped.
    pub would_skip: usize,
    /// Nodes that are unreachable.
    pub unreachable: usize,
    /// Nodes blocked by failures.
    pub blocked: usize,
    /// Estimated total duration (sum of all would-succeed node durations).
    pub estimated_total_duration: Duration,
    /// Estimated critical path duration (max path sum).
    pub estimated_critical_path_duration: Duration,
    /// Whether the workflow would complete successfully.
    pub would_complete: bool,
}

/// The workflow simulation engine.
#[derive(Debug)]
pub struct WorkflowSimulator {
    config: SimulationConfig,
}

impl WorkflowSimulator {
    /// Create a new simulator with the given config.
    #[must_use]
    pub fn new(config: SimulationConfig) -> Self {
        Self { config }
    }

    /// Create a simulator with default config (all nodes succeed).
    #[must_use]
    pub fn default_simulator() -> Self {
        Self::new(SimulationConfig::default())
    }

    /// Run the simulation on a DAG without modifying it.
    ///
    /// # Errors
    ///
    /// Returns `DagError` if the DAG contains a cycle.
    pub fn simulate(&self, dag: &WorkflowDag) -> Result<SimulationResult, DagError> {
        let order = dag.topological_sort()?;

        let mut trace = Vec::new();
        let mut node_statuses: HashMap<NodeId, SimulatedOutcome> = HashMap::new();
        let mut failed_nodes: HashSet<NodeId> = HashSet::new();
        let mut node_outputs: HashMap<NodeId, HashMap<String, serde_json::Value>> = HashMap::new();
        let mut visit_order = 0usize;

        for &node_id in &order {
            let node = match dag.nodes.get(&node_id) {
                Some(n) => n,
                None => continue,
            };

            // Check if any dependency failed
            let deps = dag.predecessors(node_id);
            let blocked_by = if !self.config.continue_on_failure {
                deps.iter().find(|d| failed_nodes.contains(d)).copied()
            } else {
                None
            };

            // Evaluate incoming edge conditions
            let evaluated_conditions = self.evaluate_incoming_conditions(dag, node_id);

            // Check if any required condition is false
            let conditions_met = evaluated_conditions.iter().all(|ec| ec.result);

            let outcome = if let Some(failed_dep) = blocked_by {
                SimulatedOutcome::Blocked {
                    failed_dependency: failed_dep,
                }
            } else if !conditions_met {
                let failed_conds: Vec<&str> = evaluated_conditions
                    .iter()
                    .filter(|ec| !ec.result)
                    .map(|ec| ec.condition.as_str())
                    .collect();
                SimulatedOutcome::WouldSkip {
                    reason: format!("conditions not met: {}", failed_conds.join(", ")),
                }
            } else if self.config.failing_task_types.contains(&node.task_type) {
                failed_nodes.insert(node_id);
                SimulatedOutcome::WouldFail {
                    reason: format!("task type '{}' configured to fail", node.task_type),
                }
            } else {
                // Would succeed: store simulated outputs
                if let Some(outputs) = self.config.simulated_outputs.get(&node.task_type) {
                    node_outputs.insert(node_id, outputs.clone());
                }
                SimulatedOutcome::WouldSucceed
            };

            let estimated_duration = self
                .config
                .estimated_durations
                .get(&node.task_type)
                .copied();

            let entry = SimulationTraceEntry {
                node_id,
                task_type: node.task_type.clone(),
                outcome: outcome.clone(),
                estimated_duration,
                dependencies: deps,
                evaluated_conditions,
                visit_order,
            };

            node_statuses.insert(node_id, outcome);
            trace.push(entry);
            visit_order += 1;
        }

        // Identify unreachable nodes
        let visited: HashSet<NodeId> = node_statuses.keys().copied().collect();
        for &node_id in dag.nodes.keys() {
            if !visited.contains(&node_id) {
                node_statuses.insert(node_id, SimulatedOutcome::Unreachable);
            }
        }

        let summary = self.compute_summary(dag, &node_statuses, &trace);

        Ok(SimulationResult {
            trace,
            summary,
            node_statuses,
        })
    }

    /// Evaluate incoming edge conditions for a node.
    fn evaluate_incoming_conditions(
        &self,
        dag: &WorkflowDag,
        node_id: NodeId,
    ) -> Vec<EvaluatedCondition> {
        let mut results = Vec::new();

        for edge in &dag.edges {
            if edge.to_node != node_id {
                continue;
            }
            if let Some(ref condition) = edge.condition {
                let (result, source) =
                    if let Some(&override_val) = self.config.condition_overrides.get(condition) {
                        (override_val, ConditionSource::Override)
                    } else {
                        (true, ConditionSource::DefaultTrue)
                    };

                results.push(EvaluatedCondition {
                    condition: condition.clone(),
                    result,
                    source,
                });
            }
        }

        results
    }

    /// Compute summary statistics from the simulation results.
    fn compute_summary(
        &self,
        dag: &WorkflowDag,
        statuses: &HashMap<NodeId, SimulatedOutcome>,
        trace: &[SimulationTraceEntry],
    ) -> SimulationSummary {
        let total_nodes = dag.nodes.len();
        let mut would_succeed = 0usize;
        let mut would_fail = 0usize;
        let mut would_skip = 0usize;
        let mut unreachable = 0usize;
        let mut blocked = 0usize;
        let mut total_duration = Duration::ZERO;

        for outcome in statuses.values() {
            match outcome {
                SimulatedOutcome::WouldSucceed => would_succeed += 1,
                SimulatedOutcome::WouldFail { .. } => would_fail += 1,
                SimulatedOutcome::WouldSkip { .. } => would_skip += 1,
                SimulatedOutcome::Unreachable => unreachable += 1,
                SimulatedOutcome::Blocked { .. } => blocked += 1,
            }
        }

        for entry in trace {
            if entry.outcome == SimulatedOutcome::WouldSucceed {
                if let Some(d) = entry.estimated_duration {
                    total_duration += d;
                }
            }
        }

        let critical_path = self.compute_critical_path_duration(dag, trace);

        SimulationSummary {
            total_nodes,
            would_succeed,
            would_fail,
            would_skip,
            unreachable,
            blocked,
            estimated_total_duration: total_duration,
            estimated_critical_path_duration: critical_path,
            would_complete: would_fail == 0 && blocked == 0,
        }
    }

    /// Compute the critical path duration using longest-path in DAG.
    fn compute_critical_path_duration(
        &self,
        dag: &WorkflowDag,
        trace: &[SimulationTraceEntry],
    ) -> Duration {
        // Build duration map from trace
        let duration_map: HashMap<NodeId, Duration> = trace
            .iter()
            .filter(|e| e.outcome == SimulatedOutcome::WouldSucceed)
            .filter_map(|e| e.estimated_duration.map(|d| (e.node_id, d)))
            .collect();

        if duration_map.is_empty() {
            return Duration::ZERO;
        }

        // Longest path via topological order
        let order = match dag.topological_sort() {
            Ok(o) => o,
            Err(_) => return Duration::ZERO,
        };

        let mut dist: HashMap<NodeId, Duration> = HashMap::new();

        for &node_id in &order {
            let self_dur = duration_map
                .get(&node_id)
                .copied()
                .unwrap_or(Duration::ZERO);
            let max_pred = dag
                .predecessors(node_id)
                .iter()
                .filter_map(|p| dist.get(p))
                .max()
                .copied()
                .unwrap_or(Duration::ZERO);

            dist.insert(node_id, max_pred + self_dur);
        }

        dist.values().max().copied().unwrap_or(Duration::ZERO)
    }
}

/// Helper to create a simple simulation for a DAG.
///
/// Returns a result indicating whether the workflow would complete successfully.
///
/// # Errors
///
/// Returns `DagError` if the DAG contains a cycle.
pub fn quick_simulate(dag: &WorkflowDag) -> Result<bool, DagError> {
    let sim = WorkflowSimulator::default_simulator();
    let result = sim.simulate(dag)?;
    Ok(result.summary.would_complete)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::{WorkflowEdge, WorkflowNode};

    fn make_node(task_type: &str) -> WorkflowNode {
        WorkflowNode::new(task_type)
    }

    fn make_linear_dag() -> (WorkflowDag, NodeId, NodeId, NodeId) {
        let mut dag = WorkflowDag::new();
        let a = dag.add_node(make_node("ingest")).expect("add node");
        let b = dag.add_node(make_node("transcode")).expect("add node");
        let c = dag.add_node(make_node("deliver")).expect("add node");
        dag.add_edge(WorkflowEdge::new(a, b, "raw"))
            .expect("add edge");
        dag.add_edge(WorkflowEdge::new(b, c, "encoded"))
            .expect("add edge");
        (dag, a, b, c)
    }

    // --- Basic simulation ---

    #[test]
    fn test_simulate_all_succeed() {
        let (dag, _, _, _) = make_linear_dag();
        let sim = WorkflowSimulator::default_simulator();
        let result = sim.simulate(&dag).expect("simulate");

        assert!(result.summary.would_complete);
        assert_eq!(result.summary.would_succeed, 3);
        assert_eq!(result.summary.would_fail, 0);
        assert_eq!(result.trace.len(), 3);
    }

    #[test]
    fn test_simulate_with_failing_task() {
        let (dag, _, _, _) = make_linear_dag();
        let config = SimulationConfig::new().with_failing_type("transcode");
        let sim = WorkflowSimulator::new(config);
        let result = sim.simulate(&dag).expect("simulate");

        assert!(!result.summary.would_complete);
        assert_eq!(result.summary.would_fail, 1);
        assert_eq!(result.summary.would_succeed, 2); // ingest + deliver still succeed
    }

    #[test]
    fn test_simulate_blocked_on_failure() {
        let (dag, _, _, _) = make_linear_dag();
        let config = SimulationConfig::new()
            .with_failing_type("transcode")
            .with_continue_on_failure(false);
        let sim = WorkflowSimulator::new(config);
        let result = sim.simulate(&dag).expect("simulate");

        assert!(!result.summary.would_complete);
        assert_eq!(result.summary.blocked, 1); // deliver is blocked
        assert_eq!(result.summary.would_fail, 1);
        assert_eq!(result.summary.would_succeed, 1); // only ingest
    }

    // --- Condition evaluation ---

    #[test]
    fn test_simulate_with_conditions() {
        let mut dag = WorkflowDag::new();
        let a = dag.add_node(make_node("check")).expect("add node");
        let b = dag.add_node(make_node("high_res")).expect("add node");
        let c = dag.add_node(make_node("low_res")).expect("add node");

        dag.add_edge(WorkflowEdge::with_condition(
            a,
            b,
            "video",
            "resolution == 4k",
        ))
        .expect("add edge");
        dag.add_edge(WorkflowEdge::with_condition(
            a,
            c,
            "video",
            "resolution != 4k",
        ))
        .expect("add edge");

        // Simulate with 4k resolution
        let config = SimulationConfig::new()
            .with_condition_override("resolution == 4k", true)
            .with_condition_override("resolution != 4k", false);
        let sim = WorkflowSimulator::new(config);
        let result = sim.simulate(&dag).expect("simulate");

        assert_eq!(result.summary.would_succeed, 2); // check + high_res
        assert_eq!(result.summary.would_skip, 1); // low_res
    }

    #[test]
    fn test_conditions_default_true() {
        let mut dag = WorkflowDag::new();
        let a = dag.add_node(make_node("source")).expect("add node");
        let b = dag.add_node(make_node("sink")).expect("add node");
        dag.add_edge(WorkflowEdge::with_condition(a, b, "data", "some_condition"))
            .expect("add edge");

        let sim = WorkflowSimulator::default_simulator();
        let result = sim.simulate(&dag).expect("simulate");

        // No override -> defaults to true
        assert_eq!(result.summary.would_succeed, 2);
        assert!(result.trace[1]
            .evaluated_conditions
            .iter()
            .all(|ec| ec.source == ConditionSource::DefaultTrue));
    }

    // --- Duration estimation ---

    #[test]
    fn test_estimated_durations() {
        let (dag, _, _, _) = make_linear_dag();
        let config = SimulationConfig::new()
            .with_duration("ingest", Duration::from_secs(10))
            .with_duration("transcode", Duration::from_secs(60))
            .with_duration("deliver", Duration::from_secs(5));
        let sim = WorkflowSimulator::new(config);
        let result = sim.simulate(&dag).expect("simulate");

        assert_eq!(
            result.summary.estimated_total_duration,
            Duration::from_secs(75)
        );
        // Critical path = sequential chain = 75s
        assert_eq!(
            result.summary.estimated_critical_path_duration,
            Duration::from_secs(75)
        );
    }

    #[test]
    fn test_critical_path_parallel() {
        let mut dag = WorkflowDag::new();
        let root = dag.add_node(make_node("start")).expect("add node");
        let branch_a = dag.add_node(make_node("fast")).expect("add node");
        let branch_b = dag.add_node(make_node("slow")).expect("add node");
        let join = dag.add_node(make_node("finish")).expect("add node");

        dag.add_edge(WorkflowEdge::new(root, branch_a, "x"))
            .expect("add edge");
        dag.add_edge(WorkflowEdge::new(root, branch_b, "x"))
            .expect("add edge");
        dag.add_edge(WorkflowEdge::new(branch_a, join, "x"))
            .expect("add edge");
        dag.add_edge(WorkflowEdge::new(branch_b, join, "x"))
            .expect("add edge");

        let config = SimulationConfig::new()
            .with_duration("start", Duration::from_secs(5))
            .with_duration("fast", Duration::from_secs(10))
            .with_duration("slow", Duration::from_secs(30))
            .with_duration("finish", Duration::from_secs(5));
        let sim = WorkflowSimulator::new(config);
        let result = sim.simulate(&dag).expect("simulate");

        // Total = 5 + 10 + 30 + 5 = 50
        assert_eq!(
            result.summary.estimated_total_duration,
            Duration::from_secs(50)
        );
        // Critical path = start(5) -> slow(30) -> finish(5) = 40
        assert_eq!(
            result.summary.estimated_critical_path_duration,
            Duration::from_secs(40)
        );
    }

    // --- Trace details ---

    #[test]
    fn test_trace_visit_order() {
        let (dag, _, _, _) = make_linear_dag();
        let sim = WorkflowSimulator::default_simulator();
        let result = sim.simulate(&dag).expect("simulate");

        for (i, entry) in result.trace.iter().enumerate() {
            assert_eq!(entry.visit_order, i);
        }
    }

    #[test]
    fn test_trace_dependencies_recorded() {
        let (dag, a, _, _) = make_linear_dag();
        let sim = WorkflowSimulator::default_simulator();
        let result = sim.simulate(&dag).expect("simulate");

        // Second node (transcode) should have ingest as dependency
        let transcode_entry = result
            .trace
            .iter()
            .find(|e| e.task_type == "transcode")
            .expect("find transcode");
        assert_eq!(transcode_entry.dependencies.len(), 1);
        assert_eq!(transcode_entry.dependencies[0], a);
    }

    // --- Edge cases ---

    #[test]
    fn test_simulate_empty_dag() {
        let dag = WorkflowDag::new();
        let sim = WorkflowSimulator::default_simulator();
        let result = sim.simulate(&dag).expect("simulate");

        assert!(result.summary.would_complete);
        assert_eq!(result.summary.total_nodes, 0);
        assert!(result.trace.is_empty());
    }

    #[test]
    fn test_simulate_single_node() {
        let mut dag = WorkflowDag::new();
        dag.add_node(make_node("solo")).expect("add node");

        let sim = WorkflowSimulator::default_simulator();
        let result = sim.simulate(&dag).expect("simulate");

        assert!(result.summary.would_complete);
        assert_eq!(result.summary.would_succeed, 1);
    }

    #[test]
    fn test_quick_simulate_success() {
        let (dag, _, _, _) = make_linear_dag();
        assert!(quick_simulate(&dag).expect("simulate"));
    }

    #[test]
    fn test_simulate_with_outputs() {
        let (dag, _, _, _) = make_linear_dag();
        let mut outputs = HashMap::new();
        outputs.insert(
            "path".to_string(),
            serde_json::json!(std::env::temp_dir()
                .join("oximedia-workflow-sim-out.mp4")
                .to_string_lossy()),
        );
        let config = SimulationConfig::new().with_outputs("ingest", outputs);
        let sim = WorkflowSimulator::new(config);
        let result = sim.simulate(&dag).expect("simulate");

        assert!(result.summary.would_complete);
    }

    #[test]
    fn test_simulate_diamond_dag() {
        // root -> A, root -> B, A -> join, B -> join
        let mut dag = WorkflowDag::new();
        let root = dag.add_node(make_node("root")).expect("add node");
        let node_a = dag.add_node(make_node("branch_a")).expect("add node");
        let node_b = dag.add_node(make_node("branch_b")).expect("add node");
        let join = dag.add_node(make_node("join")).expect("add node");

        dag.add_edge(WorkflowEdge::new(root, node_a, "x"))
            .expect("add edge");
        dag.add_edge(WorkflowEdge::new(root, node_b, "x"))
            .expect("add edge");
        dag.add_edge(WorkflowEdge::new(node_a, join, "x"))
            .expect("add edge");
        dag.add_edge(WorkflowEdge::new(node_b, join, "x"))
            .expect("add edge");

        let sim = WorkflowSimulator::default_simulator();
        let result = sim.simulate(&dag).expect("simulate");

        assert!(result.summary.would_complete);
        assert_eq!(result.summary.would_succeed, 4);
    }

    #[test]
    fn test_node_statuses_map() {
        let (dag, a, b, c) = make_linear_dag();
        let sim = WorkflowSimulator::default_simulator();
        let result = sim.simulate(&dag).expect("simulate");

        assert_eq!(result.node_statuses.len(), 3);
        assert_eq!(
            result.node_statuses.get(&a),
            Some(&SimulatedOutcome::WouldSucceed)
        );
        assert_eq!(
            result.node_statuses.get(&b),
            Some(&SimulatedOutcome::WouldSucceed)
        );
        assert_eq!(
            result.node_statuses.get(&c),
            Some(&SimulatedOutcome::WouldSucceed)
        );
    }
}

// =============================================================================
// WorkflowDryRun — high-level dry-run simulator for the Workflow type
// =============================================================================

/// A simple simulation report produced by dry-running a [`crate::workflow::Workflow`].
#[derive(Debug, Clone)]
pub struct SimulationReport {
    /// Total number of tasks (steps) in the workflow.
    pub step_count: usize,
    /// Estimated total duration in milliseconds (sum of per-task estimates).
    pub estimated_duration_ms: u64,
    /// Descriptions of dependency relationships found
    /// (e.g. `"task_b [<id>] depends on task_a [<id>]"`).
    pub dependencies: Vec<String>,
    /// Warnings detected during simulation (e.g. empty workflow, no timeout set).
    pub warnings: Vec<String>,
}

impl SimulationReport {
    /// Return `true` if no warnings were generated.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.warnings.is_empty()
    }
}

/// Dry-run simulator for [`crate::workflow::Workflow`].
///
/// Analyses a workflow statically — no tasks are executed.
pub struct WorkflowDryRun;

impl WorkflowDryRun {
    /// Simulate a workflow and return a [`SimulationReport`].
    ///
    /// Analyses:
    /// - `step_count`: number of tasks in the workflow.
    /// - `estimated_duration_ms`: sum of per-task duration estimates.
    /// - `dependencies`: textual descriptions of every `Task::dependencies` relationship.
    /// - `warnings`: issues detected (empty workflow, missing task refs, etc.).
    #[must_use]
    pub fn simulate(workflow: &crate::workflow::Workflow) -> SimulationReport {
        let mut warnings = Vec::new();
        let mut dependencies = Vec::new();

        let tasks: Vec<&crate::task::Task> = workflow.tasks().collect();
        let step_count = tasks.len();

        if step_count == 0 {
            warnings.push("Workflow has no tasks".to_string());
            return SimulationReport {
                step_count: 0,
                estimated_duration_ms: 0,
                dependencies,
                warnings,
            };
        }

        // Compute estimated duration per task and build dep descriptions.
        let mut estimated_duration_ms: u64 = 0;

        for task in &tasks {
            let dur = Self::estimate_task_duration_ms(task);
            estimated_duration_ms = estimated_duration_ms.saturating_add(dur);

            // Record dependency descriptions from task.dependencies
            for dep_id in &task.dependencies {
                // Look up the dependency task name if available.
                let dep_name = workflow
                    .get_task(dep_id)
                    .map(|t| t.name.as_str())
                    .unwrap_or("<unknown>");
                dependencies.push(format!(
                    "'{}' [{}] depends on '{}' [{}]",
                    task.name, task.id, dep_name, dep_id
                ));
            }
        }

        // Also record edge-based dependencies from workflow.edges.
        for edge in &workflow.edges {
            let from_name = workflow
                .get_task(&edge.from)
                .map(|t| t.name.as_str())
                .unwrap_or("<unknown>");
            let to_name = workflow
                .get_task(&edge.to)
                .map(|t| t.name.as_str())
                .unwrap_or("<unknown>");
            let dep_str = if let Some(ref cond) = edge.condition {
                format!(
                    "edge: '{}' [{}] -> '{}' [{}] (condition: {cond})",
                    from_name, edge.from, to_name, edge.to
                )
            } else {
                format!(
                    "edge: '{}' [{}] -> '{}' [{}]",
                    from_name, edge.from, to_name, edge.to
                )
            };
            // Avoid duplicate entries (task.dependencies vs edges may overlap)
            if !dependencies.contains(&dep_str) {
                dependencies.push(dep_str);
            }
        }

        // Cycle detection warnings.
        let cycle_warnings = Self::check_cycles(workflow);
        warnings.extend(cycle_warnings);

        // Warn about tasks with no timeout set (uses the default 1-hour timeout).
        for task in &tasks {
            if task.timeout == std::time::Duration::from_secs(3600) {
                // Default timeout — not necessarily a problem but worth noting.
            }
            // Warn about tasks with conditions that reference unknown variables.
            for cond in &task.conditions {
                if cond.trim().is_empty() {
                    warnings.push(format!(
                        "Task '{}' has an empty condition string",
                        task.name
                    ));
                }
            }
        }

        SimulationReport {
            step_count,
            estimated_duration_ms,
            dependencies,
            warnings,
        }
    }

    /// Check for circular dependencies in task.dependencies lists.
    fn check_cycles(workflow: &crate::workflow::Workflow) -> Vec<String> {
        // Build adjacency: task_id -> [dep_task_ids]
        let mut adj: std::collections::HashMap<crate::task::TaskId, Vec<crate::task::TaskId>> =
            std::collections::HashMap::new();

        for task in workflow.tasks() {
            adj.insert(task.id, task.dependencies.clone());
        }

        // Also add edges from workflow.edges.
        for edge in &workflow.edges {
            adj.entry(edge.to).or_default().push(edge.from);
        }

        let mut warnings = Vec::new();
        let mut color: std::collections::HashMap<crate::task::TaskId, u8> =
            std::collections::HashMap::new(); // 0=white,1=gray,2=black

        for &start in adj.keys() {
            if color.get(&start).copied().unwrap_or(0) == 0 {
                let mut path = Vec::new();
                if Self::dfs_cycle(start, &adj, &mut color, &mut path) {
                    warnings.push(format!("Potential cycle detected near task [{}]", start));
                }
            }
        }

        warnings
    }

    fn dfs_cycle(
        node: crate::task::TaskId,
        adj: &std::collections::HashMap<crate::task::TaskId, Vec<crate::task::TaskId>>,
        color: &mut std::collections::HashMap<crate::task::TaskId, u8>,
        path: &mut Vec<crate::task::TaskId>,
    ) -> bool {
        color.insert(node, 1); // gray
        path.push(node);

        if let Some(neighbors) = adj.get(&node) {
            for &next in neighbors {
                let c = color.get(&next).copied().unwrap_or(0);
                if c == 1 {
                    return true; // back edge → cycle
                }
                if c == 0 && Self::dfs_cycle(next, adj, color, path) {
                    return true;
                }
            }
        }

        path.pop();
        color.insert(node, 2); // black
        false
    }

    /// Estimate the execution duration of a single task in milliseconds.
    ///
    /// Uses task-type-specific heuristics:
    /// - `Wait { duration }` → uses the actual wait duration.
    /// - `Transcode` → 60 000 ms (1 minute default estimate).
    /// - `QualityControl` → 30 000 ms.
    /// - `Transfer` → 20 000 ms.
    /// - `Analysis` → 15 000 ms.
    /// - `Notification` / `HttpRequest` → 1 000 ms.
    /// - `CustomScript` / `Conditional` → 5 000 ms.
    #[must_use]
    fn estimate_task_duration_ms(task: &crate::task::Task) -> u64 {
        match &task.task_type {
            crate::task::TaskType::Wait { duration } => {
                // Convert Duration to ms, capping at u64::MAX.
                duration.as_millis().try_into().unwrap_or(u64::MAX)
            }
            crate::task::TaskType::Transcode { .. } => 60_000,
            crate::task::TaskType::QualityControl { .. } => 30_000,
            crate::task::TaskType::Transfer { .. } => 20_000,
            crate::task::TaskType::Analysis { .. } => 15_000,
            crate::task::TaskType::Notification { .. } => 1_000,
            crate::task::TaskType::HttpRequest { .. } => 1_000,
            crate::task::TaskType::CustomScript { .. } => 5_000,
            crate::task::TaskType::Conditional { .. } => 5_000,
        }
    }
}

// =============================================================================
// WorkflowDryRun tests
// =============================================================================

#[cfg(test)]
mod dry_run_tests {
    use super::*;
    use crate::task::{Task, TaskType};
    use crate::workflow::Workflow;
    use std::path::PathBuf;
    use std::time::Duration;

    fn make_transcode_task(name: &str) -> Task {
        Task::new(
            name,
            TaskType::Transcode {
                input: PathBuf::from("/in/video.mp4"),
                output: PathBuf::from("/out/video.mp4"),
                preset: "h264".to_string(),
                params: std::collections::HashMap::new(),
            },
        )
    }

    fn make_wait_task(name: &str, secs: u64) -> Task {
        Task::new(
            name,
            TaskType::Wait {
                duration: Duration::from_secs(secs),
            },
        )
    }

    fn make_notify_task(name: &str) -> Task {
        Task::new(
            name,
            TaskType::Notification {
                channel: crate::task::NotificationChannel::Webhook {
                    url: "https://example.com/hook".to_string(),
                },
                message: "done".to_string(),
                metadata: std::collections::HashMap::new(),
            },
        )
    }

    // ── empty workflow ────────────────────────────────────────────────────────

    #[test]
    fn test_dry_run_empty_workflow() {
        let wf = Workflow::new("empty");
        let report = WorkflowDryRun::simulate(&wf);
        assert_eq!(report.step_count, 0);
        assert_eq!(report.estimated_duration_ms, 0);
        assert!(!report.warnings.is_empty(), "should warn about no tasks");
        assert!(
            report.warnings.iter().any(|w| w.contains("no tasks")),
            "warning: {:?}",
            report.warnings
        );
    }

    // ── single task ───────────────────────────────────────────────────────────

    #[test]
    fn test_dry_run_single_task_step_count() {
        let mut wf = Workflow::new("single");
        wf.add_task(make_transcode_task("encode"));
        let report = WorkflowDryRun::simulate(&wf);
        assert_eq!(report.step_count, 1);
    }

    #[test]
    fn test_dry_run_single_task_no_deps() {
        let mut wf = Workflow::new("single");
        wf.add_task(make_transcode_task("encode"));
        let report = WorkflowDryRun::simulate(&wf);
        assert!(report.dependencies.is_empty());
    }

    // ── multiple tasks ────────────────────────────────────────────────────────

    #[test]
    fn test_dry_run_multiple_tasks_step_count() {
        let mut wf = Workflow::new("multi");
        wf.add_task(make_transcode_task("encode"));
        wf.add_task(make_notify_task("notify"));
        wf.add_task(make_wait_task("pause", 5));
        let report = WorkflowDryRun::simulate(&wf);
        assert_eq!(report.step_count, 3);
    }

    // ── Wait task duration ────────────────────────────────────────────────────

    #[test]
    fn test_dry_run_wait_task_uses_actual_duration() {
        let mut wf = Workflow::new("wait-wf");
        wf.add_task(make_wait_task("sleep", 10)); // 10 s = 10_000 ms
        let report = WorkflowDryRun::simulate(&wf);
        assert_eq!(report.estimated_duration_ms, 10_000);
    }

    // ── Duration sum ──────────────────────────────────────────────────────────

    #[test]
    fn test_dry_run_estimated_duration_sum() {
        let mut wf = Workflow::new("sum-wf");
        wf.add_task(make_transcode_task("encode")); // 60_000 ms
        wf.add_task(make_notify_task("notify")); // 1_000 ms
        let report = WorkflowDryRun::simulate(&wf);
        assert_eq!(report.estimated_duration_ms, 61_000);
    }

    // ── Dependencies via task.dependencies ───────────────────────────────────

    #[test]
    fn test_dry_run_dependencies_listed_via_task_deps() {
        let mut wf = Workflow::new("dep-wf");
        let encode_id = wf.add_task(make_transcode_task("encode"));
        let mut notify = make_notify_task("notify");
        notify.add_dependency(encode_id);
        wf.add_task(notify);

        let report = WorkflowDryRun::simulate(&wf);
        assert!(!report.dependencies.is_empty(), "should record dependency");
        assert!(
            report
                .dependencies
                .iter()
                .any(|d| d.contains("notify") && d.contains("encode")),
            "deps: {:?}",
            report.dependencies
        );
    }

    // ── Dependencies via workflow.edges ───────────────────────────────────────

    #[test]
    fn test_dry_run_dependencies_listed_via_edges() {
        let mut wf = Workflow::new("edge-wf");
        let encode_id = wf.add_task(make_transcode_task("encode"));
        let notify_id = wf.add_task(make_notify_task("notify"));
        wf.add_edge(encode_id, notify_id).expect("add edge");

        let report = WorkflowDryRun::simulate(&wf);
        assert!(
            report
                .dependencies
                .iter()
                .any(|d| d.contains("encode") && d.contains("notify")),
            "deps: {:?}",
            report.dependencies
        );
    }

    // ── Warnings ─────────────────────────────────────────────────────────────

    #[test]
    fn test_dry_run_warnings_for_empty_workflow() {
        let wf = Workflow::new("empty");
        let report = WorkflowDryRun::simulate(&wf);
        assert!(!report.warnings.is_empty());
    }

    #[test]
    fn test_dry_run_no_warnings_for_simple_valid_workflow() {
        let mut wf = Workflow::new("clean");
        wf.add_task(make_transcode_task("encode"));
        let report = WorkflowDryRun::simulate(&wf);
        // Simple valid workflow should have no warnings.
        assert!(
            report.warnings.is_empty(),
            "unexpected warnings: {:?}",
            report.warnings
        );
    }

    // ── SimulationReport fields ───────────────────────────────────────────────

    #[test]
    fn test_dry_run_simulation_report_all_fields_accessible() {
        let mut wf = Workflow::new("fields-test");
        wf.add_task(make_transcode_task("encode"));
        let report = WorkflowDryRun::simulate(&wf);
        // Just verify all fields are accessible.
        let _ = report.step_count;
        let _ = report.estimated_duration_ms;
        let _ = report.dependencies;
        let _ = report.warnings;
        assert!(report.is_clean());
    }

    // ── Multiple dependencies ─────────────────────────────────────────────────

    #[test]
    fn test_dry_run_multiple_dependencies() {
        let mut wf = Workflow::new("multi-dep");
        let a = wf.add_task(make_transcode_task("encode-a"));
        let b = wf.add_task(make_transcode_task("encode-b"));
        let notify_id = wf.add_task(make_notify_task("notify"));

        wf.add_edge(a, notify_id).expect("edge a->notify");
        wf.add_edge(b, notify_id).expect("edge b->notify");

        let report = WorkflowDryRun::simulate(&wf);
        // Should have at least 2 dependency entries (one per edge).
        let edge_deps: Vec<_> = report
            .dependencies
            .iter()
            .filter(|d| d.contains("notify"))
            .collect();
        assert!(
            edge_deps.len() >= 2,
            "expected 2 deps to notify, got: {:?}",
            report.dependencies
        );
    }
}
