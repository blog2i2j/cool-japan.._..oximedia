//! Standalone DOT/Graphviz export utilities.
//!
//! This module provides a lightweight, graph-agnostic API for generating
//! Graphviz DOT output from any set of nodes and edges.  Unlike the
//! [`crate::dot`] module — which is tightly coupled to [`crate::graph::PipelineGraph`] —
//! the types here accept plain `String` identifiers and simple attribute
//! structs, making them easy to use from tests, CLI tooling, or external
//! integrations.
//!
//! # Quick start
//!
//! ```rust
//! use oximedia_pipeline::dot_export::{DotConfig, DotDirection, DotNode, DotEdge, export_to_dot};
//!
//! let nodes = vec![
//!     DotNode { id: "A".into(), label: "Source".into(), shape: None,
//!               color: None, fillcolor: None, is_critical: false },
//!     DotNode { id: "B".into(), label: "Sink".into(), shape: None,
//!               color: None, fillcolor: Some("#f7c59f".into()), is_critical: false },
//! ];
//! let edges = vec![
//!     DotEdge { from_id: "A".into(), to_id: "B".into(), label: None, is_critical: false },
//! ];
//! let config = DotConfig::new("my_graph").with_direction(DotDirection::LeftRight);
//! let dot = export_to_dot(&nodes, &edges, &config);
//! assert!(dot.contains("digraph"));
//! assert!(dot.contains("A"));
//! assert!(dot.contains("->"));
//! ```

use std::fmt::Write as FmtWrite;

// ── DotDirection ─────────────────────────────────────────────────────────────

/// Rank direction used by the Graphviz layout engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DotDirection {
    /// Left-to-right layout (`rankdir=LR`).
    LeftRight,
    /// Top-to-bottom layout (`rankdir=TD`).
    TopDown,
    /// Bottom-to-top layout (`rankdir=BU`).
    BottomUp,
}

impl DotDirection {
    fn as_rankdir(&self) -> &'static str {
        match self {
            DotDirection::LeftRight => "LR",
            DotDirection::TopDown => "TD",
            DotDirection::BottomUp => "BU",
        }
    }
}

// ── DotConfig ────────────────────────────────────────────────────────────────

/// Configuration for a DOT graph export.
#[derive(Debug, Clone)]
pub struct DotConfig {
    /// Name placed after the `digraph` keyword.
    pub graph_name: String,
    /// Layout direction.
    pub direction: DotDirection,
    /// Default node shape (`"box"`, `"ellipse"`, `"diamond"`, …).
    pub node_shape: String,
    /// Highlight critical-path nodes with a distinct style.
    pub highlight_critical_path: bool,
    /// Include per-edge label strings in the output.
    pub include_edge_labels: bool,
}

impl DotConfig {
    /// Create a new config with sensible defaults.
    ///
    /// Defaults: `LR` direction, `"box"` shape, no critical-path
    /// highlighting, no edge labels.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            graph_name: name.into(),
            direction: DotDirection::LeftRight,
            node_shape: "box".to_string(),
            highlight_critical_path: false,
            include_edge_labels: false,
        }
    }

    /// Set the layout direction.
    pub fn with_direction(mut self, dir: DotDirection) -> Self {
        self.direction = dir;
        self
    }

    /// Set the default node shape.
    pub fn with_node_shape(mut self, shape: impl Into<String>) -> Self {
        self.node_shape = shape.into();
        self
    }

    /// Enable per-edge label output.
    pub fn with_edge_labels(mut self) -> Self {
        self.include_edge_labels = true;
        self
    }
}

// ── DotNode ──────────────────────────────────────────────────────────────────

/// A node to be rendered in the DOT graph.
#[derive(Debug, Clone)]
pub struct DotNode {
    /// Unique identifier used to reference this node in edges.
    pub id: String,
    /// Human-readable label displayed inside the node.
    pub label: String,
    /// Override the default node shape (e.g. `"diamond"`).
    pub shape: Option<String>,
    /// Border colour (e.g. `"#CC0000"`).
    pub color: Option<String>,
    /// Fill colour (e.g. `"#a8d8a8"`).
    pub fillcolor: Option<String>,
    /// Whether this node lies on the critical execution path.
    pub is_critical: bool,
}

// ── DotEdge ──────────────────────────────────────────────────────────────────

/// A directed edge between two nodes.
#[derive(Debug, Clone)]
pub struct DotEdge {
    /// Identifier of the source node.
    pub from_id: String,
    /// Identifier of the destination node.
    pub to_id: String,
    /// Optional label placed on the edge.
    pub label: Option<String>,
    /// Whether this edge is part of the critical path.
    pub is_critical: bool,
}

// ── export_to_dot ─────────────────────────────────────────────────────────────

/// Generate a complete Graphviz DOT string from nodes, edges, and config.
///
/// The resulting string is valid DOT syntax and can be piped to `dot -Tpng`.
///
/// Critical nodes (when `config.highlight_critical_path` is `true`) are
/// rendered with a red (`"#cc0000"`) border and bold style; critical edges are
/// rendered in red.
pub fn export_to_dot(nodes: &[DotNode], edges: &[DotEdge], config: &DotConfig) -> String {
    let mut out = String::with_capacity(256 + nodes.len() * 80 + edges.len() * 60);

    let safe_name = dot_escape(&config.graph_name);
    let _ = writeln!(out, "digraph \"{safe_name}\" {{");
    let _ = writeln!(out, "  rankdir={};", config.direction.as_rankdir());
    let _ = writeln!(out, "  node [shape={}];", config.node_shape);

    // Node declarations.
    for node in nodes {
        let label = dot_escape(&node.label);
        let shape_str = match &node.shape {
            Some(s) => format!(", shape={s}"),
            None => String::new(),
        };

        let color_str = if config.highlight_critical_path && node.is_critical {
            ", color=\"#cc0000\", style=\"bold,filled\"".to_string()
        } else {
            match (&node.color, &node.fillcolor) {
                (Some(c), Some(f)) => {
                    format!(", color=\"{c}\", fillcolor=\"{f}\", style=filled")
                }
                (Some(c), None) => format!(", color=\"{c}\""),
                (None, Some(f)) => format!(", fillcolor=\"{f}\", style=filled"),
                (None, None) => String::new(),
            }
        };

        let id = dot_escape(&node.id);
        let _ = writeln!(
            out,
            "  \"{id}\" [label=\"{label}\"{shape_str}{color_str}];"
        );
    }

    // Edge declarations.
    for edge in edges {
        let from = dot_escape(&edge.from_id);
        let to = dot_escape(&edge.to_id);

        let label_str = if config.include_edge_labels {
            match &edge.label {
                Some(l) => format!(" [label=\"{}\"]", dot_escape(l)),
                None => String::new(),
            }
        } else {
            String::new()
        };

        let color_str = if config.highlight_critical_path && edge.is_critical {
            if label_str.is_empty() {
                " [color=\"#cc0000\"]".to_string()
            } else {
                // Merge critical colour into existing label attr block.
                let trimmed = label_str.trim_end_matches(']');
                format!("{trimmed}, color=\"#cc0000\"]")
            }
        } else {
            label_str
        };

        let _ = writeln!(out, "  \"{from}\" -> \"{to}\"{color_str};");
    }

    let _ = writeln!(out, "}}");
    out
}

// ── dot_escape ────────────────────────────────────────────────────────────────

/// Escape a string for safe embedding inside a DOT double-quoted label.
///
/// Replaces `"` with `\"`, backslash with `\\`, and newlines with `\\n`.
pub fn dot_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_node(id: &str, label: &str) -> DotNode {
        DotNode {
            id: id.to_string(),
            label: label.to_string(),
            shape: None,
            color: None,
            fillcolor: None,
            is_critical: false,
        }
    }

    fn simple_edge(from: &str, to: &str) -> DotEdge {
        DotEdge {
            from_id: from.to_string(),
            to_id: to.to_string(),
            label: None,
            is_critical: false,
        }
    }

    #[test]
    fn export_contains_digraph_keyword() {
        let nodes = vec![simple_node("A", "Node A")];
        let edges: Vec<DotEdge> = vec![];
        let config = DotConfig::new("test");
        let dot = export_to_dot(&nodes, &edges, &config);
        assert!(dot.contains("digraph"), "output must start with digraph");
    }

    #[test]
    fn correct_node_count_in_output() {
        let nodes = vec![
            simple_node("A", "A"),
            simple_node("B", "B"),
            simple_node("C", "C"),
        ];
        let edges: Vec<DotEdge> = vec![];
        let config = DotConfig::new("g");
        let dot = export_to_dot(&nodes, &edges, &config);
        // Each node produces one line containing `[label=`
        let count = dot.matches("[label=").count();
        assert_eq!(count, 3, "should produce 3 node declarations");
    }

    #[test]
    fn edge_arrows_present() {
        let nodes = vec![simple_node("src", "Source"), simple_node("sink", "Sink")];
        let edges = vec![simple_edge("src", "sink")];
        let config = DotConfig::new("g");
        let dot = export_to_dot(&nodes, &edges, &config);
        assert!(dot.contains("->"), "output must contain edge arrows");
    }

    #[test]
    fn dot_escape_handles_quotes() {
        let result = dot_escape("say \"hello\"");
        assert_eq!(result, r#"say \"hello\""#);
    }

    #[test]
    fn dot_escape_handles_backslash_and_newline() {
        let result = dot_escape("line1\nline2\\end");
        assert_eq!(result, r"line1\nline2\\end");
    }

    #[test]
    fn direction_lr_appears_in_output() {
        let config = DotConfig::new("g").with_direction(DotDirection::LeftRight);
        let dot = export_to_dot(&[], &[], &config);
        assert!(dot.contains("rankdir=LR"), "should use LR direction");
    }

    #[test]
    fn critical_node_has_different_color() {
        let mut node = simple_node("slow", "Slow Node");
        node.is_critical = true;
        let config = DotConfig::new("g").with_direction(DotDirection::TopDown);
        // Need highlight_critical_path enabled.
        let config = DotConfig {
            highlight_critical_path: true,
            ..config
        };
        let dot = export_to_dot(&[node], &[], &config);
        assert!(
            dot.contains("#cc0000"),
            "critical node should have red color"
        );
    }

    #[test]
    fn empty_graph_produces_valid_dot() {
        let config = DotConfig::new("empty_pipeline");
        let dot = export_to_dot(&[], &[], &config);
        assert!(dot.contains("digraph"), "must have digraph");
        assert!(dot.contains('}'), "must have closing brace");
        // Should not crash or panic on empty input.
    }

    #[test]
    fn edge_labels_included_when_option_set() {
        let nodes = vec![simple_node("A", "A"), simple_node("B", "B")];
        let mut edge = simple_edge("A", "B");
        edge.label = Some("my_label".to_string());
        let config = DotConfig::new("g").with_edge_labels();
        let dot = export_to_dot(&nodes, &[edge], &config);
        assert!(dot.contains("my_label"), "edge label should appear in output");
    }
}
