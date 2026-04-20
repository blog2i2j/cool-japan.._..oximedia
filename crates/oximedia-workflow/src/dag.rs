//! DAG-based workflow definition with `WorkflowNode`, `WorkflowEdge`, `WorkflowDag`,
//! `WorkflowEngine` with node-level status tracking, and `WorkflowTemplate`.

#![allow(dead_code)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::type_complexity)]

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// WorkflowNode
// ---------------------------------------------------------------------------

/// Unique identifier for a workflow node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(Uuid);

impl NodeId {
    /// Create a new random node ID.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Get underlying UUID.
    #[must_use]
    pub const fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for NodeId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Execution status of a single workflow node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeStatus {
    /// Not yet started.
    Pending,
    /// Waiting for dependencies.
    Waiting,
    /// Currently executing.
    Running,
    /// Completed successfully.
    Completed,
    /// Execution failed.
    Failed(String),
    /// Skipped (e.g., due to conditional edge).
    Skipped,
}

impl NodeStatus {
    /// Returns `true` if the node finished (completed, failed, or skipped).
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed(_) | Self::Skipped)
    }

    /// Returns `true` if the node completed successfully.
    #[must_use]
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Completed)
    }
}

/// A node inside a `WorkflowDag`.
///
/// Each node represents a single processing step with typed inputs, typed
/// outputs, an opaque parameter bag, and an optional task-type tag.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowNode {
    /// Unique node identifier.
    pub node_id: NodeId,
    /// Human-readable task type / name.
    pub task_type: String,
    /// Named input ports and their current values (optional).
    pub inputs: HashMap<String, serde_json::Value>,
    /// Named output ports and their produced values (optional).
    pub outputs: HashMap<String, serde_json::Value>,
    /// Opaque parameter bag for task configuration.
    pub parameters: HashMap<String, serde_json::Value>,
    /// Current execution status.
    pub status: NodeStatus,
}

impl WorkflowNode {
    /// Create a new node with the given task type.
    #[must_use]
    pub fn new(task_type: impl Into<String>) -> Self {
        Self {
            node_id: NodeId::new(),
            task_type: task_type.into(),
            inputs: HashMap::new(),
            outputs: HashMap::new(),
            parameters: HashMap::new(),
            status: NodeStatus::Pending,
        }
    }

    /// Attach an input value.
    #[must_use]
    pub fn with_input(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.inputs.insert(key.into(), value);
        self
    }

    /// Attach a parameter.
    #[must_use]
    pub fn with_parameter(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.parameters.insert(key.into(), value);
        self
    }

    /// Record an output produced by this node.
    pub fn set_output(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.outputs.insert(key.into(), value);
    }
}

// ---------------------------------------------------------------------------
// WorkflowEdge
// ---------------------------------------------------------------------------

/// An edge connecting two nodes in a `WorkflowDag`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowEdge {
    /// Source node.
    pub from_node: NodeId,
    /// Destination node.
    pub to_node: NodeId,
    /// Data type flowing along this edge (e.g. `"video/mp4"`, `"audio/pcm"`).
    pub data_type: String,
    /// Optional condition expression.
    pub condition: Option<String>,
}

impl WorkflowEdge {
    /// Create an edge without a condition.
    #[must_use]
    pub fn new(from_node: NodeId, to_node: NodeId, data_type: impl Into<String>) -> Self {
        Self {
            from_node,
            to_node,
            data_type: data_type.into(),
            condition: None,
        }
    }

    /// Create a conditional edge.
    #[must_use]
    pub fn with_condition(
        from_node: NodeId,
        to_node: NodeId,
        data_type: impl Into<String>,
        condition: impl Into<String>,
    ) -> Self {
        Self {
            from_node,
            to_node,
            data_type: data_type.into(),
            condition: Some(condition.into()),
        }
    }
}

// ---------------------------------------------------------------------------
// WorkflowDag
// ---------------------------------------------------------------------------

/// Error type for DAG operations.
#[derive(Debug, thiserror::Error)]
pub enum DagError {
    /// Node not found in the DAG.
    #[error("Node not found: {0}")]
    NodeNotFound(NodeId),

    /// A cycle was detected in the DAG.
    #[error("Cycle detected in DAG")]
    CycleDetected,

    /// Duplicate node.
    #[error("Node already exists: {0}")]
    DuplicateNode(NodeId),
}

/// A directed acyclic graph of `WorkflowNode`s connected by `WorkflowEdge`s.
///
/// Provides cycle detection and Kahn's-algorithm topological sort.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WorkflowDag {
    /// Nodes, keyed by their ID.
    pub nodes: HashMap<NodeId, WorkflowNode>,
    /// Edges in insertion order.
    pub edges: Vec<WorkflowEdge>,
}

impl WorkflowDag {
    /// Create an empty DAG.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node. Returns an error if the node ID already exists.
    pub fn add_node(&mut self, node: WorkflowNode) -> Result<NodeId, DagError> {
        let id = node.node_id;
        if self.nodes.contains_key(&id) {
            return Err(DagError::DuplicateNode(id));
        }
        self.nodes.insert(id, node);
        Ok(id)
    }

    /// Add an edge. Both nodes must already exist.
    pub fn add_edge(&mut self, edge: WorkflowEdge) -> Result<(), DagError> {
        if !self.nodes.contains_key(&edge.from_node) {
            return Err(DagError::NodeNotFound(edge.from_node));
        }
        if !self.nodes.contains_key(&edge.to_node) {
            return Err(DagError::NodeNotFound(edge.to_node));
        }
        self.edges.push(edge);

        // Eagerly detect cycles.
        if self.has_cycle() {
            self.edges.pop();
            return Err(DagError::CycleDetected);
        }
        Ok(())
    }

    /// Returns `true` if the DAG contains a cycle (DFS-based).
    #[must_use]
    pub fn has_cycle(&self) -> bool {
        let mut visited = HashSet::new();
        let mut stack = HashSet::new();
        for &id in self.nodes.keys() {
            if self.dfs_cycle(id, &mut visited, &mut stack) {
                return true;
            }
        }
        false
    }

    fn dfs_cycle(
        &self,
        id: NodeId,
        visited: &mut HashSet<NodeId>,
        stack: &mut HashSet<NodeId>,
    ) -> bool {
        if stack.contains(&id) {
            return true;
        }
        if visited.contains(&id) {
            return false;
        }
        visited.insert(id);
        stack.insert(id);
        for edge in &self.edges {
            if edge.from_node == id && self.dfs_cycle(edge.to_node, visited, stack) {
                return true;
            }
        }
        stack.remove(&id);
        false
    }

    /// Topological sort using Kahn's algorithm.
    ///
    /// Returns nodes in an order where every dependency appears before its
    /// dependents.
    pub fn topological_sort(&self) -> Result<Vec<NodeId>, DagError> {
        if self.has_cycle() {
            return Err(DagError::CycleDetected);
        }

        // Build in-degree map.
        let mut in_degree: HashMap<NodeId, usize> = self.nodes.keys().map(|&k| (k, 0)).collect();
        for edge in &self.edges {
            *in_degree.entry(edge.to_node).or_insert(0) += 1;
        }

        let mut queue: VecDeque<NodeId> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut result = Vec::with_capacity(self.nodes.len());
        while let Some(id) = queue.pop_front() {
            result.push(id);
            for edge in &self.edges {
                if edge.from_node == id {
                    let deg = in_degree.entry(edge.to_node).or_insert(0);
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        queue.push_back(edge.to_node);
                    }
                }
            }
        }

        Ok(result)
    }

    /// Return immediate predecessors of `node_id`.
    #[must_use]
    pub fn predecessors(&self, node_id: NodeId) -> Vec<NodeId> {
        self.edges
            .iter()
            .filter(|e| e.to_node == node_id)
            .map(|e| e.from_node)
            .collect()
    }

    /// Return immediate successors of `node_id`.
    #[must_use]
    pub fn successors(&self, node_id: NodeId) -> Vec<NodeId> {
        self.edges
            .iter()
            .filter(|e| e.from_node == node_id)
            .map(|e| e.to_node)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// DagWorkflowEngine
// ---------------------------------------------------------------------------

/// Per-run node status snapshot produced by `DagWorkflowEngine::execute`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagRunStatus {
    /// Node statuses at the end of the run.
    pub node_statuses: HashMap<NodeId, NodeStatus>,
    /// Whether the overall run succeeded.
    pub succeeded: bool,
    /// Total number of nodes executed.
    pub nodes_executed: usize,
    /// Total number of nodes that failed.
    pub nodes_failed: usize,
}

/// Lightweight DAG workflow engine.
///
/// Executes nodes in topological order.  For each node it calls the registered
/// `executor` closure (if any) and tracks per-node status.
pub struct DagWorkflowEngine {
    /// Node executor: receives the mutable node and returns Ok or error string.
    executor: Option<Arc<dyn Fn(&mut WorkflowNode) -> Result<(), String> + Send + Sync>>,
    /// Status registry shared across callers.
    statuses: Arc<Mutex<HashMap<NodeId, NodeStatus>>>,
}

impl DagWorkflowEngine {
    /// Create an engine without an executor (nodes are marked Completed immediately).
    #[must_use]
    pub fn new() -> Self {
        Self {
            executor: None,
            statuses: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Attach a node executor closure.
    #[must_use]
    pub fn with_executor<F>(mut self, f: F) -> Self
    where
        F: Fn(&mut WorkflowNode) -> Result<(), String> + Send + Sync + 'static,
    {
        self.executor = Some(Arc::new(f));
        self
    }

    /// Execute a DAG and return a status snapshot.
    ///
    /// # Errors
    ///
    /// Returns a `DagError` if the DAG contains a cycle.
    pub fn execute(&self, dag: &mut WorkflowDag) -> Result<DagRunStatus, DagError> {
        let order = dag.topological_sort()?;

        let mut statuses: HashMap<NodeId, NodeStatus> = HashMap::new();
        let mut nodes_executed = 0usize;
        let mut nodes_failed = 0usize;

        for node_id in &order {
            let Some(node) = dag.nodes.get_mut(node_id) else {
                continue;
            };

            node.status = NodeStatus::Running;
            statuses.insert(*node_id, NodeStatus::Running);

            let result = if let Some(ref exec) = self.executor {
                exec(node)
            } else {
                Ok(())
            };

            match result {
                Ok(()) => {
                    node.status = NodeStatus::Completed;
                    statuses.insert(*node_id, NodeStatus::Completed);
                    nodes_executed += 1;
                }
                Err(msg) => {
                    node.status = NodeStatus::Failed(msg.clone());
                    statuses.insert(*node_id, NodeStatus::Failed(msg));
                    nodes_failed += 1;
                }
            }
        }

        // Persist in shared registry.
        if let Ok(mut guard) = self.statuses.lock() {
            guard.extend(statuses.clone());
        }

        let succeeded = nodes_failed == 0;

        Ok(DagRunStatus {
            node_statuses: statuses,
            succeeded,
            nodes_executed,
            nodes_failed,
        })
    }

    /// Get the current status of a node (across all runs).
    #[must_use]
    pub fn node_status(&self, node_id: NodeId) -> Option<NodeStatus> {
        self.statuses.lock().ok()?.get(&node_id).cloned()
    }
}

impl Default for DagWorkflowEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// WorkflowTemplate
// ---------------------------------------------------------------------------

/// A named, reusable workflow configuration that can be instantiated into a
/// `WorkflowDag`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowTemplate {
    /// Unique template name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Default parameters shared by all nodes.
    pub default_parameters: HashMap<String, serde_json::Value>,
    /// Pre-built node definitions (without IDs – IDs are assigned on instantiation).
    node_specs: Vec<NodeSpec>,
    /// Edge definitions as `(from-index, to-index, data_type)`.
    edge_specs: Vec<(usize, usize, String)>,
}

/// Lightweight spec used inside a `WorkflowTemplate`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct NodeSpec {
    task_type: String,
    parameters: HashMap<String, serde_json::Value>,
}

impl WorkflowTemplate {
    /// Create a new empty template.
    #[must_use]
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            default_parameters: HashMap::new(),
            node_specs: Vec::new(),
            edge_specs: Vec::new(),
        }
    }

    /// Add a default parameter.
    #[must_use]
    pub fn with_default_parameter(
        mut self,
        key: impl Into<String>,
        value: serde_json::Value,
    ) -> Self {
        self.default_parameters.insert(key.into(), value);
        self
    }

    /// Append a node spec; returns the spec index for use in `add_edge`.
    pub fn add_node(
        &mut self,
        task_type: impl Into<String>,
        parameters: HashMap<String, serde_json::Value>,
    ) -> usize {
        self.node_specs.push(NodeSpec {
            task_type: task_type.into(),
            parameters,
        });
        self.node_specs.len() - 1
    }

    /// Append an edge spec between two node indices.
    pub fn add_edge(&mut self, from_idx: usize, to_idx: usize, data_type: impl Into<String>) {
        self.edge_specs.push((from_idx, to_idx, data_type.into()));
    }

    /// Instantiate the template into a fresh `WorkflowDag`.
    ///
    /// `overrides` can supply parameter overrides (merged with defaults).
    pub fn instantiate(
        &self,
        overrides: &HashMap<String, serde_json::Value>,
    ) -> Result<WorkflowDag, DagError> {
        let mut dag = WorkflowDag::new();
        let mut ids: Vec<NodeId> = Vec::with_capacity(self.node_specs.len());

        for spec in &self.node_specs {
            let mut params = self.default_parameters.clone();
            params.extend(spec.parameters.clone());
            params.extend(overrides.clone());

            let node = WorkflowNode {
                node_id: NodeId::new(),
                task_type: spec.task_type.clone(),
                inputs: HashMap::new(),
                outputs: HashMap::new(),
                parameters: params,
                status: NodeStatus::Pending,
            };
            let id = dag.add_node(node)?;
            ids.push(id);
        }

        for &(from_idx, to_idx, ref dt) in &self.edge_specs {
            let edge = WorkflowEdge::new(ids[from_idx], ids[to_idx], dt.clone());
            dag.add_edge(edge)?;
        }

        Ok(dag)
    }

    /// Return the number of node specs.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.node_specs.len()
    }
}

// ---------------------------------------------------------------------------
// Conditional branching
// ---------------------------------------------------------------------------

/// Type of branch node for conditional execution paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BranchType {
    /// If-else: evaluate condition on predecessor output, choose `then_branch`
    /// or `else_branch` node.
    IfElse {
        /// Condition expression evaluated against the predecessor's outputs.
        condition: String,
        /// Node to execute when condition is true.
        then_branch: NodeId,
        /// Node to execute when condition is false.
        else_branch: NodeId,
    },
    /// Switch-case: match a key against multiple values, each mapping to a
    /// different successor node.
    Switch {
        /// The output key whose value is inspected.
        key: String,
        /// Mapping from expected string value to target node.
        cases: HashMap<String, NodeId>,
        /// Fallback node if no case matches.
        default: Option<NodeId>,
    },
}

/// A branch node that selects which successor(s) to execute.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchNode {
    /// The node ID of this branch decision point.
    pub node_id: NodeId,
    /// Which predecessor's output to evaluate.
    pub predecessor: NodeId,
    /// Branch logic.
    pub branch_type: BranchType,
}

/// Evaluator for branch conditions.
pub struct BranchEvaluator;

impl BranchEvaluator {
    /// Evaluate a simple condition expression against a set of outputs.
    ///
    /// Supported expressions:
    /// - `key == value`  (string equality)
    /// - `key != value`
    /// - `key > number`  (numeric comparison)
    /// - `key >= number`
    /// - `key < number`
    /// - `key <= number`
    /// - `key exists`    (key is present)
    /// - `key not_exists` (key is absent)
    ///
    /// Returns `None` if the expression cannot be parsed.
    #[must_use]
    pub fn evaluate_condition(
        condition: &str,
        outputs: &HashMap<String, serde_json::Value>,
    ) -> Option<bool> {
        let parts: Vec<&str> = condition.trim().splitn(3, ' ').collect();
        if parts.len() < 2 {
            return None;
        }

        let key = parts[0];

        // Unary operators
        if parts.len() == 2 {
            return match parts[1] {
                "exists" => Some(outputs.contains_key(key)),
                "not_exists" => Some(!outputs.contains_key(key)),
                _ => None,
            };
        }

        let op = parts[1];
        let rhs = parts[2];
        let lhs_val = outputs.get(key)?;

        match op {
            "==" => {
                let rhs_trimmed = rhs.trim_matches('"');
                if let Some(s) = lhs_val.as_str() {
                    Some(s == rhs_trimmed)
                } else if let Some(n) = lhs_val.as_f64() {
                    rhs.parse::<f64>()
                        .ok()
                        .map(|r| (n - r).abs() < f64::EPSILON)
                } else if let Some(b) = lhs_val.as_bool() {
                    rhs.parse::<bool>().ok().map(|r| b == r)
                } else {
                    None
                }
            }
            "!=" => {
                let eq = Self::evaluate_condition(&format!("{key} == {rhs}"), outputs)?;
                Some(!eq)
            }
            ">" | ">=" | "<" | "<=" => {
                let lhs_num = lhs_val.as_f64()?;
                let rhs_num = rhs.parse::<f64>().ok()?;
                Some(match op {
                    ">" => lhs_num > rhs_num,
                    ">=" => lhs_num >= rhs_num,
                    "<" => lhs_num < rhs_num,
                    "<=" => lhs_num <= rhs_num,
                    _ => return None,
                })
            }
            _ => None,
        }
    }

    /// Resolve which node(s) should be activated based on a branch node
    /// and the predecessor's outputs.
    #[must_use]
    pub fn resolve_branch(
        branch: &BranchNode,
        predecessor_outputs: &HashMap<String, serde_json::Value>,
    ) -> Vec<NodeId> {
        match &branch.branch_type {
            BranchType::IfElse {
                condition,
                then_branch,
                else_branch,
            } => {
                let result =
                    Self::evaluate_condition(condition, predecessor_outputs).unwrap_or(false);
                if result {
                    vec![*then_branch]
                } else {
                    vec![*else_branch]
                }
            }
            BranchType::Switch {
                key,
                cases,
                default,
            } => {
                if let Some(val) = predecessor_outputs.get(key) {
                    let val_str = if let Some(s) = val.as_str() {
                        s.to_string()
                    } else {
                        val.to_string()
                    };
                    if let Some(&target) = cases.get(&val_str) {
                        return vec![target];
                    }
                }
                default.map_or_else(Vec::new, |d| vec![d])
            }
        }
    }
}

impl WorkflowDag {
    /// Execute the DAG with branch support.
    ///
    /// For each branch node found (identified by node IDs in `branches`),
    /// only the selected successor path is activated; the other paths'
    /// nodes are marked `Skipped`.
    ///
    /// # Errors
    ///
    /// Returns `DagError` if the DAG contains a cycle.
    pub fn execute_with_branches(
        &mut self,
        branches: &HashMap<NodeId, BranchNode>,
        executor: Option<&dyn Fn(&mut WorkflowNode) -> Result<(), String>>,
    ) -> Result<DagRunStatus, DagError> {
        let order = self.topological_sort()?;

        let mut statuses: HashMap<NodeId, NodeStatus> = HashMap::new();
        let mut nodes_executed = 0usize;
        let mut nodes_failed = 0usize;
        let mut skipped_nodes: HashSet<NodeId> = HashSet::new();

        for &node_id in &order {
            // Skip nodes that have been excluded by branch decisions
            if skipped_nodes.contains(&node_id) {
                if let Some(node) = self.nodes.get_mut(&node_id) {
                    node.status = NodeStatus::Skipped;
                }
                statuses.insert(node_id, NodeStatus::Skipped);
                continue;
            }

            let Some(node) = self.nodes.get_mut(&node_id) else {
                continue;
            };

            node.status = NodeStatus::Running;
            statuses.insert(node_id, NodeStatus::Running);

            let result = if let Some(exec) = &executor {
                exec(node)
            } else {
                Ok(())
            };

            match result {
                Ok(()) => {
                    node.status = NodeStatus::Completed;
                    statuses.insert(node_id, NodeStatus::Completed);
                    nodes_executed += 1;

                    // Check if this node is a branch decision point
                    if let Some(branch) = branches.get(&node_id) {
                        let predecessor_outputs = self
                            .nodes
                            .get(&branch.predecessor)
                            .map(|n| n.outputs.clone())
                            .unwrap_or_default();
                        let selected =
                            BranchEvaluator::resolve_branch(branch, &predecessor_outputs);

                        // Mark non-selected successors for skipping
                        let all_successors = self.successors(node_id);
                        for succ_id in &all_successors {
                            if !selected.contains(succ_id) {
                                Self::collect_descendants_static(
                                    &self.edges,
                                    *succ_id,
                                    &mut skipped_nodes,
                                );
                                skipped_nodes.insert(*succ_id);
                            }
                        }
                    }
                }
                Err(msg) => {
                    node.status = NodeStatus::Failed(msg.clone());
                    statuses.insert(node_id, NodeStatus::Failed(msg));
                    nodes_failed += 1;
                }
            }
        }

        let succeeded = nodes_failed == 0;
        Ok(DagRunStatus {
            node_statuses: statuses,
            succeeded,
            nodes_executed,
            nodes_failed,
        })
    }

    /// Collect all transitive descendants of a node.
    fn collect_descendants_static(
        edges: &[WorkflowEdge],
        node_id: NodeId,
        result: &mut HashSet<NodeId>,
    ) {
        for edge in edges {
            if edge.from_node == node_id && !result.contains(&edge.to_node) {
                result.insert(edge.to_node);
                Self::collect_descendants_static(edges, edge.to_node, result);
            }
        }
    }

    /// Return all transitive descendants of the given node.
    #[must_use]
    pub fn descendants(&self, node_id: NodeId) -> HashSet<NodeId> {
        let mut result = HashSet::new();
        Self::collect_descendants_static(&self.edges, node_id, &mut result);
        result
    }
}

// ---------------------------------------------------------------------------
// Pre-built templates
// ---------------------------------------------------------------------------

/// Pre-built template: ingest and transcode.
///
/// Nodes: ingest → probe → transcode → package
#[must_use]
pub fn ingest_transcode() -> WorkflowTemplate {
    let mut tmpl = WorkflowTemplate::new(
        "ingest_transcode",
        "Ingest a source file, probe it, transcode, then package for delivery",
    )
    .with_default_parameter("output_format", serde_json::json!("mp4"))
    .with_default_parameter("preset", serde_json::json!("medium"));

    let i0 = tmpl.add_node("ingest", HashMap::new());
    let i1 = tmpl.add_node("probe", HashMap::new());
    let i2 = tmpl.add_node("transcode", HashMap::new());
    let i3 = tmpl.add_node("package", HashMap::new());

    tmpl.add_edge(i0, i1, "raw_media");
    tmpl.add_edge(i1, i2, "media_info");
    tmpl.add_edge(i2, i3, "encoded_video");

    tmpl
}

/// Pre-built template: burn subtitles into video.
///
/// Nodes: ingest → `subtitle_parse` → `burn_subtitles` → encode → deliver
#[must_use]
pub fn subtitle_burn() -> WorkflowTemplate {
    let mut tmpl = WorkflowTemplate::new(
        "subtitle_burn",
        "Parse subtitle file and burn it into the video stream",
    )
    .with_default_parameter("subtitle_format", serde_json::json!("srt"))
    .with_default_parameter("font_size", serde_json::json!(24));

    let i0 = tmpl.add_node("ingest", HashMap::new());
    let i1 = tmpl.add_node("subtitle_parse", HashMap::new());
    let i2 = tmpl.add_node("burn_subtitles", HashMap::new());
    let i3 = tmpl.add_node("encode", HashMap::new());
    let i4 = tmpl.add_node("deliver", HashMap::new());

    tmpl.add_edge(i0, i1, "raw_media");
    tmpl.add_edge(i1, i2, "subtitle_events");
    tmpl.add_edge(i2, i3, "filtered_video");
    tmpl.add_edge(i3, i4, "encoded_video");

    tmpl
}

/// Pre-built template: audio normalization.
///
/// Nodes: ingest → `audio_analyze` → normalize → encode → deliver
#[must_use]
pub fn audio_normalize() -> WorkflowTemplate {
    let mut tmpl = WorkflowTemplate::new(
        "audio_normalize",
        "Analyze audio loudness and normalize to a target LUFS level",
    )
    .with_default_parameter("target_lufs", serde_json::json!(-23.0))
    .with_default_parameter("true_peak_dbtp", serde_json::json!(-1.0));

    let i0 = tmpl.add_node("ingest", HashMap::new());
    let i1 = tmpl.add_node("audio_analyze", HashMap::new());
    let i2 = tmpl.add_node("normalize", HashMap::new());
    let i3 = tmpl.add_node("encode", HashMap::new());
    let i4 = tmpl.add_node("deliver", HashMap::new());

    tmpl.add_edge(i0, i1, "raw_audio");
    tmpl.add_edge(i1, i2, "loudness_stats");
    tmpl.add_edge(i2, i3, "normalized_audio");
    tmpl.add_edge(i3, i4, "encoded_audio");

    tmpl
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(task_type: &str) -> WorkflowNode {
        WorkflowNode::new(task_type)
    }

    // --- WorkflowNode ---

    #[test]
    fn test_node_creation() {
        let node = make_node("transcode");
        assert_eq!(node.task_type, "transcode");
        assert_eq!(node.status, NodeStatus::Pending);
        assert!(node.inputs.is_empty());
        assert!(node.outputs.is_empty());
        assert!(node.parameters.is_empty());
    }

    #[test]
    fn test_node_with_input_and_parameter() {
        let in_path = std::env::temp_dir()
            .join("oximedia-workflow-dag-in.mp4")
            .to_string_lossy()
            .into_owned();
        let node = make_node("encode")
            .with_input("src", serde_json::json!(in_path))
            .with_parameter("preset", serde_json::json!("slow"));
        assert_eq!(node.inputs["src"], serde_json::json!(in_path));
        assert_eq!(node.parameters["preset"], serde_json::json!("slow"));
    }

    #[test]
    fn test_node_set_output() {
        let mut node = make_node("transcode");
        let out_path = std::env::temp_dir()
            .join("oximedia-workflow-dag-out.mp4")
            .to_string_lossy()
            .into_owned();
        node.set_output("dst", serde_json::json!(out_path));
        assert!(node.outputs.contains_key("dst"));
    }

    #[test]
    fn test_node_status_terminal() {
        assert!(!NodeStatus::Pending.is_terminal());
        assert!(!NodeStatus::Running.is_terminal());
        assert!(NodeStatus::Completed.is_terminal());
        assert!(NodeStatus::Failed("err".to_string()).is_terminal());
        assert!(NodeStatus::Skipped.is_terminal());
    }

    // --- WorkflowEdge ---

    #[test]
    fn test_edge_creation() {
        let a = NodeId::new();
        let b = NodeId::new();
        let edge = WorkflowEdge::new(a, b, "video/mp4");
        assert_eq!(edge.from_node, a);
        assert_eq!(edge.to_node, b);
        assert_eq!(edge.data_type, "video/mp4");
        assert!(edge.condition.is_none());
    }

    #[test]
    fn test_edge_with_condition() {
        let a = NodeId::new();
        let b = NodeId::new();
        let edge = WorkflowEdge::with_condition(a, b, "audio/pcm", "bitrate > 128");
        assert_eq!(edge.condition, Some("bitrate > 128".to_string()));
    }

    // --- WorkflowDag ---

    #[test]
    fn test_dag_add_node_and_edge() {
        let mut dag = WorkflowDag::new();
        let a = dag
            .add_node(make_node("ingest"))
            .expect("should succeed in test");
        let b = dag
            .add_node(make_node("transcode"))
            .expect("should succeed in test");
        dag.add_edge(WorkflowEdge::new(a, b, "raw_media"))
            .expect("should succeed in test");

        assert_eq!(dag.nodes.len(), 2);
        assert_eq!(dag.edges.len(), 1);
        assert!(!dag.has_cycle());
    }

    #[test]
    fn test_dag_cycle_detection() {
        let mut dag = WorkflowDag::new();
        let a = dag
            .add_node(make_node("a"))
            .expect("should succeed in test");
        let b = dag
            .add_node(make_node("b"))
            .expect("should succeed in test");
        dag.add_edge(WorkflowEdge::new(a, b, "x"))
            .expect("should succeed in test");
        let result = dag.add_edge(WorkflowEdge::new(b, a, "x"));
        assert!(matches!(result, Err(DagError::CycleDetected)));
    }

    #[test]
    fn test_dag_topological_sort() {
        let mut dag = WorkflowDag::new();
        let a = dag
            .add_node(make_node("a"))
            .expect("should succeed in test");
        let b = dag
            .add_node(make_node("b"))
            .expect("should succeed in test");
        let c = dag
            .add_node(make_node("c"))
            .expect("should succeed in test");
        dag.add_edge(WorkflowEdge::new(a, b, "x"))
            .expect("should succeed in test");
        dag.add_edge(WorkflowEdge::new(b, c, "x"))
            .expect("should succeed in test");

        let order = dag.topological_sort().expect("should succeed in test");
        let pos_a = order
            .iter()
            .position(|&x| x == a)
            .expect("should succeed in test");
        let pos_b = order
            .iter()
            .position(|&x| x == b)
            .expect("should succeed in test");
        let pos_c = order
            .iter()
            .position(|&x| x == c)
            .expect("should succeed in test");
        assert!(pos_a < pos_b && pos_b < pos_c);
    }

    #[test]
    fn test_dag_predecessors_successors() {
        let mut dag = WorkflowDag::new();
        let a = dag
            .add_node(make_node("a"))
            .expect("should succeed in test");
        let b = dag
            .add_node(make_node("b"))
            .expect("should succeed in test");
        let c = dag
            .add_node(make_node("c"))
            .expect("should succeed in test");
        dag.add_edge(WorkflowEdge::new(a, c, "x"))
            .expect("should succeed in test");
        dag.add_edge(WorkflowEdge::new(b, c, "x"))
            .expect("should succeed in test");

        let preds = dag.predecessors(c);
        assert_eq!(preds.len(), 2);
        assert!(preds.contains(&a));
        assert!(preds.contains(&b));

        let succs = dag.successors(a);
        assert_eq!(succs.len(), 1);
        assert_eq!(succs[0], c);
    }

    // --- DagWorkflowEngine ---

    #[test]
    fn test_engine_execute_no_executor() {
        let mut dag = WorkflowDag::new();
        let a = dag
            .add_node(make_node("a"))
            .expect("should succeed in test");
        let b = dag
            .add_node(make_node("b"))
            .expect("should succeed in test");
        dag.add_edge(WorkflowEdge::new(a, b, "x"))
            .expect("should succeed in test");

        let engine = DagWorkflowEngine::new();
        let result = engine.execute(&mut dag).expect("should succeed in test");
        assert!(result.succeeded);
        assert_eq!(result.nodes_executed, 2);
        assert_eq!(result.nodes_failed, 0);
    }

    #[test]
    fn test_engine_execute_with_executor() {
        let mut dag = WorkflowDag::new();
        let a = dag
            .add_node(make_node("a"))
            .expect("should succeed in test");
        let b = dag
            .add_node(make_node("b"))
            .expect("should succeed in test");
        dag.add_edge(WorkflowEdge::new(a, b, "x"))
            .expect("should succeed in test");

        let engine = DagWorkflowEngine::new().with_executor(|node| {
            node.set_output("done", serde_json::json!(true));
            Ok(())
        });

        let result = engine.execute(&mut dag).expect("should succeed in test");
        assert!(result.succeeded);
        assert_eq!(result.nodes_executed, 2);
        // Outputs were set.
        assert!(dag.nodes[&a].outputs.contains_key("done"));
    }

    #[test]
    fn test_engine_node_failure() {
        let mut dag = WorkflowDag::new();
        let a = dag
            .add_node(make_node("failing"))
            .expect("should succeed in test");
        dag.add_node(make_node("b"))
            .expect("should succeed in test"); // isolated node

        let engine = DagWorkflowEngine::new().with_executor(|node| {
            if node.task_type == "failing" {
                Err("intentional failure".to_string())
            } else {
                Ok(())
            }
        });

        let result = engine.execute(&mut dag).expect("should succeed in test");
        assert!(!result.succeeded);
        assert!(result.nodes_failed > 0);
        assert!(matches!(result.node_statuses[&a], NodeStatus::Failed(_)));
    }

    #[test]
    fn test_engine_node_status_accessor() {
        let mut dag = WorkflowDag::new();
        let a = dag
            .add_node(make_node("a"))
            .expect("should succeed in test");

        let engine = DagWorkflowEngine::new();
        engine.execute(&mut dag).expect("should succeed in test");

        assert_eq!(engine.node_status(a), Some(NodeStatus::Completed));
        assert_eq!(engine.node_status(NodeId::new()), None);
    }

    // --- WorkflowTemplate ---

    #[test]
    fn test_template_instantiate_ingest_transcode() {
        let tmpl = ingest_transcode();
        assert_eq!(tmpl.name, "ingest_transcode");
        assert_eq!(tmpl.node_count(), 4);

        let dag = tmpl
            .instantiate(&HashMap::new())
            .expect("should succeed in test");
        assert_eq!(dag.nodes.len(), 4);
        assert_eq!(dag.edges.len(), 3);
        assert!(!dag.has_cycle());
    }

    #[test]
    fn test_template_instantiate_subtitle_burn() {
        let tmpl = subtitle_burn();
        let dag = tmpl
            .instantiate(&HashMap::new())
            .expect("should succeed in test");
        assert_eq!(dag.nodes.len(), 5);
        assert_eq!(dag.edges.len(), 4);
        assert!(!dag.has_cycle());
    }

    #[test]
    fn test_template_instantiate_audio_normalize() {
        let tmpl = audio_normalize();
        let dag = tmpl
            .instantiate(&HashMap::new())
            .expect("should succeed in test");
        assert_eq!(dag.nodes.len(), 5);
        assert_eq!(dag.edges.len(), 4);
        assert!(!dag.has_cycle());
    }

    #[test]
    fn test_template_parameter_override() {
        let tmpl = ingest_transcode();
        let mut overrides = HashMap::new();
        overrides.insert("preset".to_string(), serde_json::json!("ultrafast"));

        let dag = tmpl
            .instantiate(&overrides)
            .expect("should succeed in test");
        // Every node should have the overridden preset.
        for node in dag.nodes.values() {
            assert_eq!(node.parameters["preset"], serde_json::json!("ultrafast"));
        }
    }

    #[test]
    fn test_template_default_parameters() {
        let tmpl = audio_normalize();
        assert_eq!(
            tmpl.default_parameters["target_lufs"],
            serde_json::json!(-23.0)
        );
        assert_eq!(
            tmpl.default_parameters["true_peak_dbtp"],
            serde_json::json!(-1.0)
        );
    }

    #[test]
    fn test_dag_error_node_not_found() {
        let mut dag = WorkflowDag::new();
        let a = dag
            .add_node(make_node("a"))
            .expect("should succeed in test");
        let ghost = NodeId::new();
        let result = dag.add_edge(WorkflowEdge::new(a, ghost, "x"));
        assert!(matches!(result, Err(DagError::NodeNotFound(_))));
    }

    // --- BranchEvaluator ---

    #[test]
    fn test_branch_evaluator_equality() {
        let mut outputs = HashMap::new();
        outputs.insert("codec".to_string(), serde_json::json!("h264"));

        assert_eq!(
            BranchEvaluator::evaluate_condition("codec == h264", &outputs),
            Some(true)
        );
        assert_eq!(
            BranchEvaluator::evaluate_condition("codec == vp9", &outputs),
            Some(false)
        );
        assert_eq!(
            BranchEvaluator::evaluate_condition("codec != vp9", &outputs),
            Some(true)
        );
    }

    #[test]
    fn test_branch_evaluator_numeric_comparison() {
        let mut outputs = HashMap::new();
        outputs.insert("bitrate".to_string(), serde_json::json!(5000.0));

        assert_eq!(
            BranchEvaluator::evaluate_condition("bitrate > 3000", &outputs),
            Some(true)
        );
        assert_eq!(
            BranchEvaluator::evaluate_condition("bitrate < 3000", &outputs),
            Some(false)
        );
        assert_eq!(
            BranchEvaluator::evaluate_condition("bitrate >= 5000", &outputs),
            Some(true)
        );
        assert_eq!(
            BranchEvaluator::evaluate_condition("bitrate <= 5000", &outputs),
            Some(true)
        );
    }

    #[test]
    fn test_branch_evaluator_exists() {
        let mut outputs = HashMap::new();
        outputs.insert("result".to_string(), serde_json::json!(42));

        assert_eq!(
            BranchEvaluator::evaluate_condition("result exists", &outputs),
            Some(true)
        );
        assert_eq!(
            BranchEvaluator::evaluate_condition("missing not_exists", &outputs),
            Some(true)
        );
        assert_eq!(
            BranchEvaluator::evaluate_condition("result not_exists", &outputs),
            Some(false)
        );
    }

    #[test]
    fn test_branch_evaluator_boolean() {
        let mut outputs = HashMap::new();
        outputs.insert("success".to_string(), serde_json::json!(true));

        assert_eq!(
            BranchEvaluator::evaluate_condition("success == true", &outputs),
            Some(true)
        );
        assert_eq!(
            BranchEvaluator::evaluate_condition("success == false", &outputs),
            Some(false)
        );
    }

    #[test]
    fn test_branch_evaluator_numeric_equality() {
        let mut outputs = HashMap::new();
        outputs.insert("count".to_string(), serde_json::json!(42.0));

        assert_eq!(
            BranchEvaluator::evaluate_condition("count == 42", &outputs),
            Some(true)
        );
    }

    #[test]
    fn test_branch_evaluator_invalid_expression() {
        let outputs = HashMap::new();
        assert_eq!(BranchEvaluator::evaluate_condition("", &outputs), None);
        assert_eq!(
            BranchEvaluator::evaluate_condition("single", &outputs),
            None
        );
    }

    #[test]
    fn test_resolve_branch_if_else_true() {
        let then_id = NodeId::new();
        let else_id = NodeId::new();
        let pred_id = NodeId::new();
        let branch_id = NodeId::new();

        let branch = BranchNode {
            node_id: branch_id,
            predecessor: pred_id,
            branch_type: BranchType::IfElse {
                condition: "quality > 90".to_string(),
                then_branch: then_id,
                else_branch: else_id,
            },
        };

        let mut outputs = HashMap::new();
        outputs.insert("quality".to_string(), serde_json::json!(95.0));

        let result = BranchEvaluator::resolve_branch(&branch, &outputs);
        assert_eq!(result, vec![then_id]);
    }

    #[test]
    fn test_resolve_branch_if_else_false() {
        let then_id = NodeId::new();
        let else_id = NodeId::new();
        let pred_id = NodeId::new();
        let branch_id = NodeId::new();

        let branch = BranchNode {
            node_id: branch_id,
            predecessor: pred_id,
            branch_type: BranchType::IfElse {
                condition: "quality > 90".to_string(),
                then_branch: then_id,
                else_branch: else_id,
            },
        };

        let mut outputs = HashMap::new();
        outputs.insert("quality".to_string(), serde_json::json!(50.0));

        let result = BranchEvaluator::resolve_branch(&branch, &outputs);
        assert_eq!(result, vec![else_id]);
    }

    #[test]
    fn test_resolve_branch_switch() {
        let av1_id = NodeId::new();
        let h264_id = NodeId::new();
        let default_id = NodeId::new();
        let pred_id = NodeId::new();
        let branch_id = NodeId::new();

        let mut cases = HashMap::new();
        cases.insert("av1".to_string(), av1_id);
        cases.insert("h264".to_string(), h264_id);

        let branch = BranchNode {
            node_id: branch_id,
            predecessor: pred_id,
            branch_type: BranchType::Switch {
                key: "codec".to_string(),
                cases,
                default: Some(default_id),
            },
        };

        let mut outputs = HashMap::new();
        outputs.insert("codec".to_string(), serde_json::json!("av1"));
        assert_eq!(
            BranchEvaluator::resolve_branch(&branch, &outputs),
            vec![av1_id]
        );

        outputs.insert("codec".to_string(), serde_json::json!("h264"));
        assert_eq!(
            BranchEvaluator::resolve_branch(&branch, &outputs),
            vec![h264_id]
        );

        outputs.insert("codec".to_string(), serde_json::json!("vp9"));
        assert_eq!(
            BranchEvaluator::resolve_branch(&branch, &outputs),
            vec![default_id]
        );
    }

    #[test]
    fn test_resolve_branch_switch_no_default() {
        let av1_id = NodeId::new();
        let pred_id = NodeId::new();
        let branch_id = NodeId::new();

        let mut cases = HashMap::new();
        cases.insert("av1".to_string(), av1_id);

        let branch = BranchNode {
            node_id: branch_id,
            predecessor: pred_id,
            branch_type: BranchType::Switch {
                key: "codec".to_string(),
                cases,
                default: None,
            },
        };

        let mut outputs = HashMap::new();
        outputs.insert("codec".to_string(), serde_json::json!("vp9"));
        assert!(BranchEvaluator::resolve_branch(&branch, &outputs).is_empty());
    }

    #[test]
    fn test_execute_with_branches_if_else() {
        // Build DAG: probe -> branch_decision
        //   branch_decision -> high_res_encode
        //   branch_decision -> low_res_encode
        let mut dag = WorkflowDag::new();
        let probe = dag.add_node(make_node("probe")).expect("add node");
        let decision = dag.add_node(make_node("branch")).expect("add node");
        let high = dag.add_node(make_node("high_res")).expect("add node");
        let low = dag.add_node(make_node("low_res")).expect("add node");

        dag.add_edge(WorkflowEdge::new(probe, decision, "media_info"))
            .expect("add edge");
        dag.add_edge(WorkflowEdge::new(decision, high, "video"))
            .expect("add edge");
        dag.add_edge(WorkflowEdge::new(decision, low, "video"))
            .expect("add edge");

        let branch = BranchNode {
            node_id: decision,
            predecessor: probe,
            branch_type: BranchType::IfElse {
                condition: "resolution > 1080".to_string(),
                then_branch: high,
                else_branch: low,
            },
        };

        let mut branches = HashMap::new();
        branches.insert(decision, branch);

        // Executor that sets probe output to resolution=4000
        let result = dag
            .execute_with_branches(
                &branches,
                Some(&|node: &mut WorkflowNode| {
                    if node.task_type == "probe" {
                        node.set_output("resolution", serde_json::json!(4000.0));
                    }
                    Ok(())
                }),
            )
            .expect("execute");

        assert!(result.succeeded);
        assert_eq!(
            result.node_statuses.get(&high),
            Some(&NodeStatus::Completed)
        );
        assert_eq!(result.node_statuses.get(&low), Some(&NodeStatus::Skipped));
    }

    #[test]
    fn test_descendants() {
        let mut dag = WorkflowDag::new();
        let a = dag.add_node(make_node("a")).expect("add node");
        let b = dag.add_node(make_node("b")).expect("add node");
        let c = dag.add_node(make_node("c")).expect("add node");
        let d = dag.add_node(make_node("d")).expect("add node");

        dag.add_edge(WorkflowEdge::new(a, b, "x")).expect("edge");
        dag.add_edge(WorkflowEdge::new(b, c, "x")).expect("edge");
        dag.add_edge(WorkflowEdge::new(b, d, "x")).expect("edge");

        let desc = dag.descendants(a);
        assert_eq!(desc.len(), 3);
        assert!(desc.contains(&b));
        assert!(desc.contains(&c));
        assert!(desc.contains(&d));

        let desc_b = dag.descendants(b);
        assert_eq!(desc_b.len(), 2);
        assert!(!desc_b.contains(&a));
    }

    #[test]
    fn test_dag_error_duplicate_node() {
        let mut dag = WorkflowDag::new();
        let node = make_node("a");
        let id = node.node_id;
        dag.add_node(node).expect("should succeed in test");
        let dup = WorkflowNode {
            node_id: id,
            task_type: "b".to_string(),
            inputs: HashMap::new(),
            outputs: HashMap::new(),
            parameters: HashMap::new(),
            status: NodeStatus::Pending,
        };
        assert!(matches!(dag.add_node(dup), Err(DagError::DuplicateNode(_))));
    }
}
