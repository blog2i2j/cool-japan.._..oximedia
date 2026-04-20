//! Signal flow visualization (DOT export for Graphviz).

use super::graph::{NodeType, SignalFlowGraph};
use std::fmt::Write;

/// Options for visualization export
#[derive(Debug, Clone)]
pub struct VisualizeOptions {
    /// Include channel counts in labels
    pub show_channels: bool,
    /// Include gain values on edges
    pub show_gain: bool,
    /// Highlight inactive connections
    pub show_inactive: bool,
    /// Graph direction (LR for left-to-right, TB for top-to-bottom)
    pub direction: GraphDirection,
}

/// Graph layout direction
#[derive(Debug, Clone, Copy)]
pub enum GraphDirection {
    /// Left to right
    LeftToRight,
    /// Top to bottom
    TopToBottom,
}

impl Default for VisualizeOptions {
    fn default() -> Self {
        Self {
            show_channels: true,
            show_gain: true,
            show_inactive: false,
            direction: GraphDirection::LeftToRight,
        }
    }
}

impl SignalFlowGraph {
    /// Export the graph to DOT format for Graphviz
    #[must_use]
    pub fn to_dot(&self, options: &VisualizeOptions) -> String {
        let mut dot = String::new();

        // Graph header
        let _ = writeln!(&mut dot, "digraph SignalFlow {{");

        // Graph attributes
        match options.direction {
            GraphDirection::LeftToRight => {
                let _ = writeln!(&mut dot, "  rankdir=LR;");
            }
            GraphDirection::TopToBottom => {
                let _ = writeln!(&mut dot, "  rankdir=TB;");
            }
        }

        let _ = writeln!(&mut dot, "  node [shape=box, style=rounded];");

        // Add nodes
        for (idx, node_type) in self.get_all_nodes() {
            let (label, color, shape) = match node_type {
                NodeType::Input { label, channels } => {
                    let lbl = if options.show_channels {
                        format!("{label}\\n({channels} ch)")
                    } else {
                        label.clone()
                    };
                    (lbl, "lightblue", "box")
                }
                NodeType::Output { label, channels } => {
                    let lbl = if options.show_channels {
                        format!("{label}\\n({channels} ch)")
                    } else {
                        label.clone()
                    };
                    (lbl, "lightgreen", "box")
                }
                NodeType::Processor {
                    label,
                    processor_type,
                } => {
                    let lbl = format!("{label}\\n[{processor_type}]");
                    (lbl, "lightyellow", "ellipse")
                }
                NodeType::Bus { label, channels } => {
                    let lbl = if options.show_channels {
                        format!("{label}\\n({channels} ch)")
                    } else {
                        label.clone()
                    };
                    (lbl, "lightgray", "box")
                }
            };

            let _ = writeln!(
                &mut dot,
                "  n{} [label=\"{}\", fillcolor={}, style=filled, shape={}];",
                idx.index(),
                label,
                color,
                shape
            );
        }

        // Add edges
        for node in self.graph().node_indices() {
            for (target, edge) in self.get_outputs(node) {
                if !options.show_inactive && !edge.active {
                    continue;
                }

                let mut edge_attrs = Vec::new();

                if options.show_gain && edge.gain_db.abs() > f32::EPSILON {
                    edge_attrs.push(format!("label=\"{:.1} dB\"", edge.gain_db));
                }

                if !edge.active {
                    edge_attrs.push("style=dashed".to_string());
                    edge_attrs.push("color=gray".to_string());
                }

                let attrs = if edge_attrs.is_empty() {
                    String::new()
                } else {
                    format!(" [{}]", edge_attrs.join(", "))
                };

                let _ = writeln!(
                    &mut dot,
                    "  n{} -> n{}{};",
                    node.index(),
                    target.index(),
                    attrs
                );
            }
        }

        let _ = writeln!(&mut dot, "}}");
        dot
    }

    /// Export simplified DOT format
    #[must_use]
    pub fn to_dot_simple(&self) -> String {
        let options = VisualizeOptions {
            show_channels: false,
            show_gain: false,
            show_inactive: false,
            direction: GraphDirection::LeftToRight,
        };
        self.to_dot(&options)
    }

    /// Export detailed DOT format
    #[must_use]
    pub fn to_dot_detailed(&self) -> String {
        let options = VisualizeOptions {
            show_channels: true,
            show_gain: true,
            show_inactive: true,
            direction: GraphDirection::LeftToRight,
        };
        self.to_dot(&options)
    }

    /// Generate a text-based ASCII representation
    #[must_use]
    pub fn to_ascii(&self) -> String {
        let mut output = String::new();

        let _ = writeln!(&mut output, "Signal Flow Graph:");
        let _ = writeln!(&mut output, "==================");
        let _ = writeln!(&mut output);

        // List all nodes
        let _ = writeln!(&mut output, "Nodes:");
        for (idx, node_type) in self.get_all_nodes() {
            let node_str = match node_type {
                NodeType::Input { label, channels } => {
                    format!("[INPUT] {label} ({channels} ch)")
                }
                NodeType::Output { label, channels } => {
                    format!("[OUTPUT] {label} ({channels} ch)")
                }
                NodeType::Processor {
                    label,
                    processor_type,
                } => {
                    format!("[PROCESSOR] {label} [{processor_type}]")
                }
                NodeType::Bus { label, channels } => {
                    format!("[BUS] {label} ({channels} ch)")
                }
            };
            let _ = writeln!(&mut output, "  {}: {}", idx.index(), node_str);
        }

        let _ = writeln!(&mut output);
        let _ = writeln!(&mut output, "Connections:");

        // List all connections
        for node in self.graph().node_indices() {
            for (target, edge) in self.get_outputs(node) {
                let status = if edge.active { "active" } else { "inactive" };
                let gain_str = if edge.gain_db.abs() > f32::EPSILON {
                    format!(" ({:.1} dB)", edge.gain_db)
                } else {
                    String::new()
                };

                let _ = writeln!(
                    &mut output,
                    "  {} -> {} [{}{}]",
                    node.index(),
                    target.index(),
                    status,
                    gain_str
                );
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::graph::SignalFlowGraph;

    use crate::flow::FlowEdge;
    #[test]
    fn test_dot_export_simple() {
        let mut graph = SignalFlowGraph::new();

        let input = graph.add_input("Mic".to_string(), 1);
        let output = graph.add_output("Speaker".to_string(), 2);

        graph
            .connect(input, output, FlowEdge::default())
            .expect("should succeed in test");

        let dot = graph.to_dot_simple();
        assert!(dot.contains("digraph SignalFlow"));
        assert!(dot.contains("Mic"));
        assert!(dot.contains("Speaker"));
    }

    #[test]
    fn test_dot_export_with_gain() {
        let mut graph = SignalFlowGraph::new();

        let input = graph.add_input("Source".to_string(), 2);
        let output = graph.add_output("Dest".to_string(), 2);

        let edge = FlowEdge {
            gain_db: -6.0,
            channels: 2,
            active: true,
        };

        graph
            .connect(input, output, edge)
            .expect("should succeed in test");

        let options = VisualizeOptions::default();
        let dot = graph.to_dot(&options);

        assert!(dot.contains("-6.0 dB"));
    }

    #[test]
    fn test_dot_export_inactive() {
        let mut graph = SignalFlowGraph::new();

        let input = graph.add_input("Source".to_string(), 2);
        let output = graph.add_output("Dest".to_string(), 2);

        let edge = FlowEdge {
            gain_db: 0.0,
            channels: 2,
            active: false,
        };

        graph
            .connect(input, output, edge)
            .expect("should succeed in test");

        let options = VisualizeOptions {
            show_inactive: true,
            ..Default::default()
        };

        let dot = graph.to_dot(&options);
        assert!(dot.contains("dashed"));
    }

    #[test]
    fn test_ascii_export() {
        let mut graph = SignalFlowGraph::new();

        let input = graph.add_input("Mic".to_string(), 1);
        let bus = graph.add_bus("Mix".to_string(), 2);
        let output = graph.add_output("Monitor".to_string(), 2);

        graph
            .connect(input, bus, FlowEdge::default())
            .expect("should succeed in test");
        graph
            .connect(bus, output, FlowEdge::default())
            .expect("should succeed in test");

        let ascii = graph.to_ascii();

        assert!(ascii.contains("Signal Flow Graph"));
        assert!(ascii.contains("INPUT"));
        assert!(ascii.contains("BUS"));
        assert!(ascii.contains("OUTPUT"));
        assert!(ascii.contains("Connections"));
    }

    #[test]
    fn test_processor_in_visualization() {
        let mut graph = SignalFlowGraph::new();

        let proc = graph.add_processor("EQ".to_string(), "Equalizer".to_string());
        let output = graph.add_output("Out".to_string(), 2);

        graph
            .connect(proc, output, FlowEdge::default())
            .expect("should succeed in test");

        let ascii = graph.to_ascii();
        assert!(ascii.contains("PROCESSOR"));
        assert!(ascii.contains("Equalizer"));
    }

    #[test]
    fn test_direction_options() {
        let mut graph = SignalFlowGraph::new();
        graph.add_input("In".to_string(), 1);

        let lr_options = VisualizeOptions {
            direction: GraphDirection::LeftToRight,
            ..Default::default()
        };
        let lr_dot = graph.to_dot(&lr_options);
        assert!(lr_dot.contains("rankdir=LR"));

        let tb_options = VisualizeOptions {
            direction: GraphDirection::TopToBottom,
            ..Default::default()
        };
        let tb_dot = graph.to_dot(&tb_options);
        assert!(tb_dot.contains("rankdir=TB"));
    }
}
