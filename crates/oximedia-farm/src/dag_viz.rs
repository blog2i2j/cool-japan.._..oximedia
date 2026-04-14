//! Job dependency DAG visualization for complex encoding pipelines.
//!
//! This module provides a rich visualization layer on top of the job dependency
//! graph, producing both textual and structured output formats suitable for
//! rendering in terminals, web UIs, or dedicated graph tools.
//!
//! ## Supported output formats
//!
//! | Format | Description |
//! |--------|-------------|
//! | DOT    | Graphviz DOT language — suitable for `dot -Tpng` or online viewers |
//! | ASCII  | Lightweight text tree for terminal display |
//! | JSON   | Machine-readable adjacency list for custom frontends |
//! | Mermaid | Mermaid.js flowchart syntax for embedding in Markdown |
//!
//! ## Node styling
//!
//! Nodes are styled by their [`JobState`]-equivalent status:
//! - `Completed` → solid green border
//! - `Running`   → dashed blue border
//! - `Failed`    → solid red border
//! - `Pending` / others → default grey
//!
//! ## Example
//!
//! ```rust
//! use oximedia_farm::dag_viz::{DagViz, NodeState, VizFormat};
//!
//! let mut viz = DagViz::new("live-production");
//! viz.add_node("ingest", "Ingest", NodeState::Completed);
//! viz.add_node("transcode-hd", "Transcode HD", NodeState::Running);
//! viz.add_node("transcode-sd", "Transcode SD", NodeState::Pending);
//! viz.add_node("package", "Package HLS", NodeState::Pending);
//! viz.add_edge("transcode-hd", "ingest");
//! viz.add_edge("transcode-sd", "ingest");
//! viz.add_edge("package", "transcode-hd");
//! viz.add_edge("package", "transcode-sd");
//!
//! let dot = viz.render(VizFormat::Dot).unwrap();
//! assert!(dot.contains("digraph"));
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Node state
// ---------------------------------------------------------------------------

/// Status of a node in the visualization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum NodeState {
    /// Job has not yet started.
    Pending,
    /// Job is actively running.
    Running,
    /// Job finished successfully.
    Completed,
    /// Job encountered a fatal error.
    Failed,
    /// Job is queued but blocked on dependencies.
    Blocked,
    /// Job has been cancelled.
    Cancelled,
}

impl NodeState {
    /// Return a DOT colour attribute for the node.
    #[must_use]
    fn dot_color(self) -> &'static str {
        match self {
            Self::Pending => "grey",
            Self::Running => "blue",
            Self::Completed => "green",
            Self::Failed => "red",
            Self::Blocked => "orange",
            Self::Cancelled => "purple",
        }
    }

    /// Return a DOT style attribute for the node.
    #[must_use]
    fn dot_style(self) -> &'static str {
        match self {
            Self::Running => "dashed",
            _ => "solid",
        }
    }

    /// Return the Mermaid state class for the node.
    #[must_use]
    fn mermaid_class(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Blocked => "blocked",
            Self::Cancelled => "cancelled",
        }
    }
}

impl std::fmt::Display for NodeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Pending => "Pending",
            Self::Running => "Running",
            Self::Completed => "Completed",
            Self::Failed => "Failed",
            Self::Blocked => "Blocked",
            Self::Cancelled => "Cancelled",
        };
        write!(f, "{s}")
    }
}

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

/// A node in the DAG, representing a single farm job.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VizNode {
    /// Unique identifier (used for edge references).
    pub id: String,
    /// Human-readable label shown in the visualization.
    pub label: String,
    /// Current execution state.
    pub state: NodeState,
    /// Optional metadata key-value pairs (e.g., worker, duration, codec).
    pub metadata: HashMap<String, String>,
}

impl VizNode {
    /// Create a new node with no metadata.
    #[must_use]
    pub fn new(id: impl Into<String>, label: impl Into<String>, state: NodeState) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            state,
            metadata: HashMap::new(),
        }
    }

    /// Add a metadata key-value pair.
    #[must_use]
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during DAG visualization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VizError {
    /// A referenced node ID was not found.
    UnknownNode(String),
    /// Adding an edge would create a cycle.
    CycleDetected {
        /// The source node of the offending edge.
        from: String,
        /// The target node of the offending edge.
        to: String,
    },
    /// A node with the same ID was already registered.
    DuplicateNode(String),
}

impl std::fmt::Display for VizError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownNode(id) => write!(f, "unknown node id: {id}"),
            Self::CycleDetected { from, to } => {
                write!(f, "adding edge {from} -> {to} would create a cycle")
            }
            Self::DuplicateNode(id) => write!(f, "duplicate node id: {id}"),
        }
    }
}

impl std::error::Error for VizError {}

/// Result type for visualization operations.
pub type Result<T> = std::result::Result<T, VizError>;

// ---------------------------------------------------------------------------
// Output format
// ---------------------------------------------------------------------------

/// Supported output formats for DAG rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VizFormat {
    /// Graphviz DOT language.
    Dot,
    /// ASCII tree printed to a string.
    Ascii,
    /// JSON adjacency list.
    Json,
    /// Mermaid.js flowchart.
    Mermaid,
}

// ---------------------------------------------------------------------------
// DagViz
// ---------------------------------------------------------------------------

/// A visualization of a job dependency DAG.
///
/// Nodes and edges are stored in insertion order so that the output is
/// deterministic for the same sequence of calls.
#[derive(Debug, Default)]
pub struct DagViz {
    /// Name of the pipeline or farm (used as graph title).
    name: String,
    /// Ordered list of node IDs.
    node_order: Vec<String>,
    /// Node data keyed by ID.
    nodes: HashMap<String, VizNode>,
    /// Directed edges: `(from, to)` means `from` depends on `to`.
    edges: Vec<(String, String)>,
}

impl DagViz {
    /// Create a new empty DAG visualization with the given name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            node_order: Vec::new(),
            nodes: HashMap::new(),
            edges: Vec::new(),
        }
    }

    /// Add a node.
    ///
    /// # Errors
    ///
    /// Returns `VizError::DuplicateNode` if a node with the same `id` is
    /// already registered.
    pub fn add_node_checked(
        &mut self,
        id: impl Into<String>,
        label: impl Into<String>,
        state: NodeState,
    ) -> Result<()> {
        let id = id.into();
        if self.nodes.contains_key(&id) {
            return Err(VizError::DuplicateNode(id));
        }
        self.node_order.push(id.clone());
        self.nodes.insert(id.clone(), VizNode::new(id, label, state));
        Ok(())
    }

    /// Add a node, silently replacing any existing node with the same ID.
    pub fn add_node(&mut self, id: impl Into<String>, label: impl Into<String>, state: NodeState) {
        let id = id.into();
        if !self.nodes.contains_key(&id) {
            self.node_order.push(id.clone());
        }
        self.nodes.insert(id.clone(), VizNode::new(id, label, state));
    }

    /// Add a node with metadata.
    pub fn add_node_with_meta(&mut self, node: VizNode) {
        let id = node.id.clone();
        if !self.nodes.contains_key(&id) {
            self.node_order.push(id.clone());
        }
        self.nodes.insert(id, node);
    }

    /// Update the state of an existing node.
    ///
    /// # Errors
    ///
    /// Returns `VizError::UnknownNode` if `id` is not registered.
    pub fn set_state(&mut self, id: &str, state: NodeState) -> Result<()> {
        self.nodes
            .get_mut(id)
            .ok_or_else(|| VizError::UnknownNode(id.to_string()))
            .map(|n| n.state = state)
    }

    /// Add a directed edge: `from` depends on `to`.
    ///
    /// Both nodes must be registered.  Adding an edge that would create a
    /// cycle is rejected.  Duplicate edges are silently ignored.
    ///
    /// # Errors
    ///
    /// Returns `VizError::UnknownNode` or `VizError::CycleDetected`.
    pub fn add_edge(&mut self, from: impl Into<String>, to: impl Into<String>) -> Result<()> {
        let from = from.into();
        let to = to.into();

        if !self.nodes.contains_key(&from) {
            return Err(VizError::UnknownNode(from));
        }
        if !self.nodes.contains_key(&to) {
            return Err(VizError::UnknownNode(to));
        }

        // Duplicate check
        if self.edges.iter().any(|(f, t)| f == &from && t == &to) {
            return Ok(());
        }

        // Cycle check: would adding `to → from` (the reverse) create a path
        // from `to` back to `from` through existing edges?  Equivalently, does
        // a path already exist from `to` to `from`?
        if self.has_path(&to, &from) {
            return Err(VizError::CycleDetected {
                from: from.clone(),
                to: to.clone(),
            });
        }

        self.edges.push((from, to));
        Ok(())
    }

    /// Return `true` if a directed path exists from `start` to `goal`.
    fn has_path(&self, start: &str, goal: &str) -> bool {
        let mut visited: HashSet<&str> = HashSet::new();
        let mut queue: VecDeque<&str> = VecDeque::new();
        queue.push_back(start);
        while let Some(current) = queue.pop_front() {
            if current == goal {
                return true;
            }
            if !visited.insert(current) {
                continue;
            }
            for (f, t) in &self.edges {
                if f == current {
                    queue.push_back(t);
                }
            }
        }
        false
    }

    /// Return the number of nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return the number of edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Render the DAG to the requested format.
    ///
    /// # Errors
    ///
    /// Currently infallible, but returns `Result` for forward compatibility.
    pub fn render(&self, format: VizFormat) -> Result<String> {
        match format {
            VizFormat::Dot => Ok(self.render_dot()),
            VizFormat::Ascii => Ok(self.render_ascii()),
            VizFormat::Json => Ok(self.render_json()),
            VizFormat::Mermaid => Ok(self.render_mermaid()),
        }
    }

    // -----------------------------------------------------------------------
    // DOT renderer
    // -----------------------------------------------------------------------

    fn render_dot(&self) -> String {
        let mut out = format!("digraph \"{}\" {{\n", self.name);
        out.push_str("  rankdir=LR;\n");
        out.push_str("  node [shape=box fontname=\"Helvetica\"];\n\n");

        for id in &self.node_order {
            if let Some(node) = self.nodes.get(id) {
                let safe_id = sanitize_dot_id(id);
                let color = node.state.dot_color();
                let style = node.state.dot_style();
                let tooltip: String = node
                    .metadata
                    .iter()
                    .map(|(k, v)| format!("{k}={v}"))
                    .collect::<Vec<_>>()
                    .join("\\n");
                out.push_str(&format!(
                    "  {safe_id} [label=\"{label}\\n({state})\" color={color} style={style} tooltip=\"{tooltip}\"];\n",
                    label = escape_dot_label(&node.label),
                    state = node.state,
                ));
            }
        }

        out.push('\n');
        for (from, to) in &self.edges {
            let safe_from = sanitize_dot_id(from);
            let safe_to = sanitize_dot_id(to);
            out.push_str(&format!("  {safe_from} -> {safe_to};\n"));
        }

        out.push_str("}\n");
        out
    }

    // -----------------------------------------------------------------------
    // ASCII renderer
    // -----------------------------------------------------------------------

    fn render_ascii(&self) -> String {
        // Find root nodes (no incoming edges) in insertion order
        let has_incoming: HashSet<&str> = self.edges.iter().map(|(_, t)| t.as_str()).collect();
        let roots: Vec<&str> = self
            .node_order
            .iter()
            .map(String::as_str)
            .filter(|id| !has_incoming.contains(id))
            .collect();

        let mut out = format!("Pipeline: {}\n", self.name);
        for root in roots {
            self.ascii_subtree(root, "", true, &mut out, &mut HashSet::new());
        }
        out
    }

    fn ascii_subtree<'a>(
        &'a self,
        node_id: &'a str,
        prefix: &str,
        is_last: bool,
        out: &mut String,
        visited: &mut HashSet<&'a str>,
    ) {
        let connector = if is_last { "└─ " } else { "├─ " };
        let label = self
            .nodes
            .get(node_id)
            .map_or(node_id, |n| n.label.as_str());
        let state = self
            .nodes
            .get(node_id)
            .map_or(NodeState::Pending, |n| n.state);
        out.push_str(&format!("{prefix}{connector}[{state}] {label}\n"));

        if visited.contains(node_id) {
            return;
        }
        visited.insert(node_id);

        let children: Vec<&str> = self
            .edges
            .iter()
            .filter(|(f, _)| f == node_id)
            .map(|(_, t)| t.as_str())
            .collect();

        let child_prefix = format!("{}{}", prefix, if is_last { "   " } else { "│  " });
        for (i, child) in children.iter().enumerate() {
            let last = i == children.len() - 1;
            self.ascii_subtree(child, &child_prefix, last, out, visited);
        }
    }

    // -----------------------------------------------------------------------
    // JSON renderer
    // -----------------------------------------------------------------------

    fn render_json(&self) -> String {
        #[derive(serde::Serialize)]
        struct JsonGraph<'a> {
            name: &'a str,
            nodes: Vec<&'a VizNode>,
            edges: Vec<JsonEdge<'a>>,
        }

        #[derive(serde::Serialize)]
        struct JsonEdge<'a> {
            from: &'a str,
            to: &'a str,
        }

        let graph = JsonGraph {
            name: &self.name,
            nodes: self.node_order.iter().filter_map(|id| self.nodes.get(id)).collect(),
            edges: self
                .edges
                .iter()
                .map(|(f, t)| JsonEdge {
                    from: f.as_str(),
                    to: t.as_str(),
                })
                .collect(),
        };

        serde_json::to_string_pretty(&graph).unwrap_or_else(|_| "{}".to_string())
    }

    // -----------------------------------------------------------------------
    // Mermaid renderer
    // -----------------------------------------------------------------------

    fn render_mermaid(&self) -> String {
        let mut out = String::from("flowchart LR\n");

        // Class definitions
        out.push_str("  classDef pending fill:#ccc,stroke:#999;\n");
        out.push_str("  classDef running fill:#cce5ff,stroke:#0066cc,stroke-dasharray:5 5;\n");
        out.push_str("  classDef completed fill:#d4edda,stroke:#28a745;\n");
        out.push_str("  classDef failed fill:#f8d7da,stroke:#dc3545;\n");
        out.push_str("  classDef blocked fill:#fff3cd,stroke:#fd7e14;\n");
        out.push_str("  classDef cancelled fill:#e2d9f3,stroke:#6f42c1;\n\n");

        for id in &self.node_order {
            if let Some(node) = self.nodes.get(id) {
                let safe_id = sanitize_mermaid_id(id);
                out.push_str(&format!(
                    "  {}[\"{}: {}\"]\n",
                    safe_id, node.label, node.state
                ));
            }
        }

        out.push('\n');
        for (from, to) in &self.edges {
            out.push_str(&format!(
                "  {} --> {}\n",
                sanitize_mermaid_id(from),
                sanitize_mermaid_id(to)
            ));
        }

        // Class assignments
        out.push('\n');
        for id in &self.node_order {
            if let Some(node) = self.nodes.get(id) {
                let safe_id = sanitize_mermaid_id(id);
                out.push_str(&format!(
                    "  class {} {};\n",
                    safe_id,
                    node.state.mermaid_class()
                ));
            }
        }

        out
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Sanitize a node ID for use as a DOT identifier.
fn sanitize_dot_id(id: &str) -> String {
    let s: String = id
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
        .collect();
    // DOT identifiers must not start with a digit
    if s.starts_with(|c: char| c.is_ascii_digit()) {
        format!("n_{s}")
    } else {
        s
    }
}

/// Escape a label string for embedding inside a DOT `label="…"` attribute.
fn escape_dot_label(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// Sanitize a node ID for use as a Mermaid identifier.
fn sanitize_mermaid_id(id: &str) -> String {
    id.chars()
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn build_simple_dag() -> DagViz {
        let mut viz = DagViz::new("test-pipeline");
        viz.add_node("ingest", "Ingest", NodeState::Completed);
        viz.add_node("transcode", "Transcode", NodeState::Running);
        viz.add_node("package", "Package", NodeState::Pending);
        viz.add_edge("transcode", "ingest").expect("edge 1");
        viz.add_edge("package", "transcode").expect("edge 2");
        viz
    }

    #[test]
    fn test_node_and_edge_count() {
        let viz = build_simple_dag();
        assert_eq!(viz.node_count(), 3);
        assert_eq!(viz.edge_count(), 2);
    }

    #[test]
    fn test_dot_output_contains_digraph() {
        let viz = build_simple_dag();
        let dot = viz.render(VizFormat::Dot).expect("dot render");
        assert!(dot.contains("digraph"), "DOT should contain 'digraph'");
        assert!(dot.contains("test-pipeline"));
    }

    #[test]
    fn test_dot_output_contains_all_nodes() {
        let viz = build_simple_dag();
        let dot = viz.render(VizFormat::Dot).expect("dot render");
        assert!(dot.contains("ingest"), "should contain ingest node");
        assert!(dot.contains("transcode"), "should contain transcode node");
        assert!(dot.contains("package"), "should contain package node");
    }

    #[test]
    fn test_ascii_output_contains_pipeline_name() {
        let viz = build_simple_dag();
        let ascii = viz.render(VizFormat::Ascii).expect("ascii render");
        assert!(ascii.contains("test-pipeline"), "ASCII should contain pipeline name");
    }

    #[test]
    fn test_json_output_is_valid() {
        let viz = build_simple_dag();
        let json = viz.render(VizFormat::Json).expect("json render");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert!(parsed["nodes"].is_array());
        assert!(parsed["edges"].is_array());
        assert_eq!(parsed["nodes"].as_array().map(|a| a.len()), Some(3));
        assert_eq!(parsed["edges"].as_array().map(|a| a.len()), Some(2));
    }

    #[test]
    fn test_mermaid_output_contains_flowchart() {
        let viz = build_simple_dag();
        let mermaid = viz.render(VizFormat::Mermaid).expect("mermaid render");
        assert!(mermaid.contains("flowchart"), "Mermaid should start with flowchart");
        assert!(mermaid.contains("-->"), "should contain edge arrows");
    }

    #[test]
    fn test_cycle_detection_prevents_cycle() {
        let mut viz = DagViz::new("cycle-test");
        viz.add_node("a", "A", NodeState::Pending);
        viz.add_node("b", "B", NodeState::Pending);
        viz.add_node("c", "C", NodeState::Pending);
        viz.add_edge("b", "a").expect("b depends on a");
        viz.add_edge("c", "b").expect("c depends on b");
        // This would create a cycle: a -> c -> b -> a
        let err = viz.add_edge("a", "c").unwrap_err();
        assert!(matches!(err, VizError::CycleDetected { .. }));
    }

    #[test]
    fn test_unknown_node_edge_returns_error() {
        let mut viz = DagViz::new("x");
        viz.add_node("a", "A", NodeState::Pending);
        let err = viz.add_edge("a", "nonexistent").unwrap_err();
        assert!(matches!(err, VizError::UnknownNode(_)));
    }

    #[test]
    fn test_set_state_updates_node() {
        let mut viz = DagViz::new("x");
        viz.add_node("job1", "Job 1", NodeState::Pending);
        viz.set_state("job1", NodeState::Completed).expect("set state");
        let snap = viz.render(VizFormat::Dot).expect("dot");
        assert!(snap.contains("Completed") || snap.contains("green"));
    }

    #[test]
    fn test_set_state_unknown_node_returns_error() {
        let mut viz = DagViz::new("x");
        let err = viz.set_state("ghost", NodeState::Running).unwrap_err();
        assert!(matches!(err, VizError::UnknownNode(_)));
    }

    #[test]
    fn test_duplicate_edge_ignored() {
        let mut viz = DagViz::new("dup-test");
        viz.add_node("a", "A", NodeState::Pending);
        viz.add_node("b", "B", NodeState::Pending);
        viz.add_edge("b", "a").expect("first");
        viz.add_edge("b", "a").expect("duplicate should be silently ignored");
        assert_eq!(viz.edge_count(), 1);
    }

    #[test]
    fn test_node_state_display() {
        assert_eq!(NodeState::Completed.to_string(), "Completed");
        assert_eq!(NodeState::Failed.to_string(), "Failed");
    }

    #[test]
    fn test_viz_error_display() {
        let err = VizError::UnknownNode("ghost".to_string());
        assert!(err.to_string().contains("ghost"));
        let err2 = VizError::CycleDetected {
            from: "a".to_string(),
            to: "b".to_string(),
        };
        assert!(err2.to_string().contains("cycle"));
    }
}
