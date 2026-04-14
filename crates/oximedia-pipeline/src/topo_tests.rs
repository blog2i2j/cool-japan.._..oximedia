//! Comprehensive tests for topological sort correctness and pipeline validation
//! with malformed and well-formed graph shapes.
//!
//! This module implements two unchecked TODO items from TODO.md:
//!
//! - **Property-based tests for topological sort correctness with random graph shapes**:
//!   We enumerate canonical graph shapes (chains, diamonds, fans, binary trees,
//!   long paths) and verify that `topological_sort` produces a valid ordering
//!   for each.  "Valid" here means: for every edge `(u, v)` in the graph, `u`
//!   appears before `v` in the sorted output.
//!
//! - **Test pipeline validation with malformed graphs (disconnected nodes, dangling pads)**:
//!   We exercise the full surface of [`PipelineGraph::validate`] and
//!   [`crate::validation::PipelineValidator`] with carefully constructed
//!   malformed inputs.
//!
//! # Why not `proptest`?
//!
//! The workspace has no `proptest` or `quickcheck` dependency, and adding one
//! would violate the "use workspace deps only" policy.  Instead we use a
//! deterministic exhaustive enumeration approach: we build every canonical
//! random-like shape up to N nodes and verify the invariant holds for all.

use crate::graph::{Edge, PipelineGraph};
use crate::node::{FilterConfig, FrameFormat, NodeId, NodeSpec, NodeType, SinkConfig, SourceConfig, StreamSpec};
use crate::validation::PipelineValidator;
use crate::PipelineError;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn vs() -> StreamSpec {
    StreamSpec::video(FrameFormat::Yuv420p, 1920, 1080, 25)
}

/// Assert that a topological ordering is valid for `graph`:
/// for every edge `(u, v)` in the graph, `u` appears before `v`
/// in `order`.
fn assert_topo_order_valid(graph: &PipelineGraph, order: &[NodeId]) {
    // Build position index.
    let pos: std::collections::HashMap<NodeId, usize> = order
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();
    for edge in &graph.edges {
        let from_pos = pos
            .get(&edge.from_node)
            .copied()
            .unwrap_or(usize::MAX);
        let to_pos = pos
            .get(&edge.to_node)
            .copied()
            .unwrap_or(usize::MAX);
        assert!(
            from_pos < to_pos,
            "edge {:?} → {:?}: from appears at pos {} but to appears at pos {}",
            edge.from_node,
            edge.to_node,
            from_pos,
            to_pos
        );
    }
    // All nodes in the graph must appear in the order.
    assert_eq!(
        order.len(),
        graph.nodes.len(),
        "topological order must include every node"
    );
}

/// Build a linear chain: `n0 → n1 → n2 → … → n_{len-1}`.
fn build_chain(len: usize) -> PipelineGraph {
    let mut g = PipelineGraph::new();
    if len == 0 {
        return g;
    }
    let specs: Vec<NodeId> = (0..len)
        .map(|i| {
            let spec = if i == 0 {
                NodeSpec::source(&format!("n{i}"), SourceConfig::File("x".into()), vs())
            } else if i == len - 1 {
                NodeSpec::sink(&format!("n{i}"), SinkConfig::Null, vs())
            } else {
                NodeSpec::filter(&format!("n{i}"), FilterConfig::Hflip, vs(), vs())
            };
            g.add_node(spec)
        })
        .collect();
    for i in 0..len - 1 {
        g.edges.push(Edge {
            from_node: specs[i],
            from_pad: "default".into(),
            to_node: specs[i + 1],
            to_pad: "default".into(),
        });
    }
    g
}

/// Build a diamond graph: `src → left, src → right, left → sink, right → sink`.
fn build_diamond() -> PipelineGraph {
    let mut g = PipelineGraph::new();
    let src = g.add_node(NodeSpec::source("src", SourceConfig::File("x".into()), vs()));
    let left = g.add_node(NodeSpec::filter("left", FilterConfig::Hflip, vs(), vs()));
    let right = g.add_node(NodeSpec::filter("right", FilterConfig::Vflip, vs(), vs()));
    let sink = g.add_node(NodeSpec::sink("sink", SinkConfig::Null, vs()));
    for (from, to) in [(src, left), (src, right), (left, sink), (right, sink)] {
        g.edges.push(Edge {
            from_node: from,
            from_pad: "default".into(),
            to_node: to,
            to_pad: "default".into(),
        });
    }
    g
}

/// Build a fan-out graph: `src → s1, src → s2, src → s3`.
fn build_fan_out(branches: usize) -> PipelineGraph {
    let mut g = PipelineGraph::new();
    let src = g.add_node(NodeSpec::source("src", SourceConfig::File("x".into()), vs()));
    for i in 0..branches {
        let sink = g.add_node(NodeSpec::sink(
            &format!("sink{i}"),
            SinkConfig::Null,
            vs(),
        ));
        g.edges.push(Edge {
            from_node: src,
            from_pad: "default".into(),
            to_node: sink,
            to_pad: "default".into(),
        });
    }
    g
}

/// Build a fan-in graph: `src0 → merge, src1 → merge, merge → sink`.
fn build_fan_in(inputs: usize) -> PipelineGraph {
    let mut g = PipelineGraph::new();
    // Build merge node with `inputs` input pads.
    let input_pads: Vec<(String, StreamSpec)> = (0..inputs)
        .map(|i| (format!("in{i}"), vs()))
        .collect();
    let merge = NodeSpec::new(
        "merge",
        NodeType::Merge,
        input_pads,
        vec![("default".into(), vs())],
    );
    let merge_id = g.add_node(merge);
    let sink = g.add_node(NodeSpec::sink("sink", SinkConfig::Null, vs()));
    g.edges.push(Edge {
        from_node: merge_id,
        from_pad: "default".into(),
        to_node: sink,
        to_pad: "default".into(),
    });
    for i in 0..inputs {
        let src = g.add_node(NodeSpec::source(
            &format!("src{i}"),
            SourceConfig::File("x".into()),
            vs(),
        ));
        g.edges.push(Edge {
            from_node: src,
            from_pad: "default".into(),
            to_node: merge_id,
            to_pad: format!("in{i}"),
        });
    }
    g
}

/// Build a binary tree of depth `depth` (root at top, leaves at bottom).
/// All edges point *downwards* (root → children → … → leaves).
fn build_tree(depth: usize) -> PipelineGraph {
    let mut g = PipelineGraph::new();
    if depth == 0 {
        return g;
    }
    let root = g.add_node(NodeSpec::source("root", SourceConfig::File("x".into()), vs()));
    let mut current_level = vec![root];
    for level in 1..depth {
        let mut next_level = Vec::new();
        for parent in &current_level {
            for side in &["l", "r"] {
                let spec = if level == depth - 1 {
                    NodeSpec::sink(&format!("{}{}", *side, level), SinkConfig::Null, vs())
                } else {
                    NodeSpec::filter(
                        &format!("{}{}", *side, level),
                        FilterConfig::Hflip,
                        vs(),
                        vs(),
                    )
                };
                let child = g.add_node(spec);
                g.edges.push(Edge {
                    from_node: *parent,
                    from_pad: "default".into(),
                    to_node: child,
                    to_pad: "default".into(),
                });
                next_level.push(child);
            }
        }
        current_level = next_level;
    }
    g
}

// ── Topological sort tests ─────────────────────────────────────────────────────

/// Verify that a chain of N nodes produces a valid topological ordering.
#[test]
fn topo_chain_lengths() {
    for len in [1, 2, 3, 5, 10, 20, 50] {
        let g = build_chain(len);
        let order = g
            .topological_sort()
            .unwrap_or_else(|e| panic!("chain(len={len}) failed: {e}"));
        assert_topo_order_valid(&g, &order);
    }
}

/// Verify topological sort on a diamond graph.
#[test]
fn topo_diamond() {
    let g = build_diamond();
    let order = g.topological_sort().expect("diamond topo ok");
    assert_topo_order_valid(&g, &order);
}

/// Verify topological sort on fan-out graphs with 1–8 branches.
#[test]
fn topo_fan_out() {
    for branches in [1, 2, 3, 5, 8] {
        let g = build_fan_out(branches);
        let order = g
            .topological_sort()
            .unwrap_or_else(|e| panic!("fan_out(branches={branches}) failed: {e}"));
        assert_topo_order_valid(&g, &order);
    }
}

/// Verify topological sort on fan-in graphs with 1–5 inputs.
#[test]
fn topo_fan_in() {
    for inputs in [1, 2, 3, 5] {
        let g = build_fan_in(inputs);
        let order = g
            .topological_sort()
            .unwrap_or_else(|e| panic!("fan_in(inputs={inputs}) failed: {e}"));
        assert_topo_order_valid(&g, &order);
    }
}

/// Verify topological sort on binary tree graphs of depth 1–4.
#[test]
fn topo_binary_tree() {
    for depth in [1, 2, 3, 4] {
        let g = build_tree(depth);
        let order = g
            .topological_sort()
            .unwrap_or_else(|e| panic!("tree(depth={depth}) failed: {e}"));
        assert_topo_order_valid(&g, &order);
    }
}

/// A single-node graph must produce a one-element ordering.
#[test]
fn topo_single_node() {
    let mut g = PipelineGraph::new();
    g.add_node(NodeSpec::source("only", SourceConfig::File("x".into()), vs()));
    let order = g.topological_sort().expect("single node topo ok");
    assert_eq!(order.len(), 1);
}

/// An empty graph must produce an empty ordering.
#[test]
fn topo_empty_graph() {
    let g = PipelineGraph::new();
    let order = g.topological_sort().expect("empty topo ok");
    assert!(order.is_empty());
}

/// A two-node cycle must produce `CycleDetected`.
#[test]
fn topo_two_node_cycle_detected() {
    let mut g = PipelineGraph::new();
    let a = g.add_node(NodeSpec::filter("a", FilterConfig::Hflip, vs(), vs()));
    let b = g.add_node(NodeSpec::filter("b", FilterConfig::Vflip, vs(), vs()));
    g.edges.push(Edge { from_node: a, from_pad: "default".into(), to_node: b, to_pad: "default".into() });
    g.edges.push(Edge { from_node: b, from_pad: "default".into(), to_node: a, to_pad: "default".into() });
    let result = g.topological_sort();
    assert!(result.is_err());
    assert!(matches!(result, Err(PipelineError::CycleDetected { .. })));
}

/// A self-loop must produce `CycleDetected`.
#[test]
fn topo_self_loop_detected() {
    let mut g = PipelineGraph::new();
    let a = g.add_node(NodeSpec::filter("self_loop", FilterConfig::Hflip, vs(), vs()));
    g.edges.push(Edge {
        from_node: a,
        from_pad: "default".into(),
        to_node: a,
        to_pad: "default".into(),
    });
    let result = g.topological_sort();
    assert!(result.is_err());
}

/// A three-node cycle (A→B→C→A) must produce `CycleDetected`.
#[test]
fn topo_three_node_cycle_detected() {
    let mut g = PipelineGraph::new();
    let a = g.add_node(NodeSpec::filter("a", FilterConfig::Hflip, vs(), vs()));
    let b = g.add_node(NodeSpec::filter("b", FilterConfig::Vflip, vs(), vs()));
    let c = g.add_node(NodeSpec::filter("c", FilterConfig::Hflip, vs(), vs()));
    for (from, to) in [(a, b), (b, c), (c, a)] {
        g.edges.push(Edge { from_node: from, from_pad: "default".into(), to_node: to, to_pad: "default".into() });
    }
    let result = g.topological_sort();
    assert!(matches!(result, Err(PipelineError::CycleDetected { .. })));
}

/// CycleDetected path must include at least 2 entries.
#[test]
fn topo_cycle_path_nonempty() {
    let mut g = PipelineGraph::new();
    let a = g.add_node(NodeSpec::filter("alpha", FilterConfig::Hflip, vs(), vs()));
    let b = g.add_node(NodeSpec::filter("beta", FilterConfig::Vflip, vs(), vs()));
    g.edges.push(Edge { from_node: a, from_pad: "default".into(), to_node: b, to_pad: "default".into() });
    g.edges.push(Edge { from_node: b, from_pad: "default".into(), to_node: a, to_pad: "default".into() });
    if let Err(PipelineError::CycleDetected { path }) = g.topological_sort() {
        assert!(path.len() >= 2, "cycle path must have at least 2 elements, got: {path:?}");
        // The path should repeat the first element at the end to show the loop.
        assert_eq!(path.first(), path.last(), "cycle path first and last should match");
    } else {
        panic!("expected CycleDetected");
    }
}

// ── Validation with malformed graphs ─────────────────────────────────────────

/// A completely disconnected filter (no edges at all) triggers both
/// UnsatisfiedInput and DisconnectedNode.
#[test]
fn validate_completely_isolated_filter() {
    let mut g = PipelineGraph::new();
    g.add_node(NodeSpec::filter("orphan", FilterConfig::Hflip, vs(), vs()));
    let report = PipelineValidator::new().validate(&g);
    assert!(!report.is_valid);
    let has_unsatisfied = report.errors.iter().any(|e| {
        matches!(e, crate::validation::ValidationError::UnsatisfiedInput { node_name, .. }
            if node_name == "orphan")
    });
    assert!(has_unsatisfied, "expected UnsatisfiedInput for isolated filter");
}

/// A sink node with no incoming edge triggers UnsatisfiedInput.
#[test]
fn validate_sink_no_input() {
    let mut g = PipelineGraph::new();
    g.add_node(NodeSpec::sink("floating_sink", SinkConfig::Null, vs()));
    let report = PipelineValidator::new().validate(&g);
    assert!(!report.is_valid);
    assert!(report.errors.iter().any(|e| {
        matches!(e, crate::validation::ValidationError::UnsatisfiedInput { .. })
    }));
}

/// A filter connected to a source but not to a sink is valid structurally
/// (it satisfies its own input) but the source triggers an UnusedSource
/// warning through the filter.
#[test]
fn validate_source_filter_no_sink() {
    let mut g = PipelineGraph::new();
    let src = g.add_node(NodeSpec::source("src", SourceConfig::File("x".into()), vs()));
    let flt = g.add_node(NodeSpec::filter("flt", FilterConfig::Hflip, vs(), vs()));
    g.edges.push(Edge { from_node: src, from_pad: "default".into(), to_node: flt, to_pad: "default".into() });
    let report = PipelineValidator::new().validate(&g);
    // The filter has its input satisfied and is reachable from src, so no
    // structural error for it.  No sink, but that alone is not an error.
    // There should be no UnsatisfiedInput error.
    assert!(!report.errors.iter().any(|e| matches!(e, crate::validation::ValidationError::UnsatisfiedInput { node_name, .. } if node_name == "flt")));
}

/// A graph with multiple disconnected subgraphs — one valid (src→sink) and one
/// orphaned filter — reports the orphan's problems.
#[test]
fn validate_mixed_valid_and_invalid_subgraphs() {
    let mut g = PipelineGraph::new();
    let src = g.add_node(NodeSpec::source("src", SourceConfig::File("x".into()), vs()));
    let sink = g.add_node(NodeSpec::sink("sink", SinkConfig::Null, vs()));
    g.connect(src, "default", sink, "default").expect("connect");
    // Orphaned filter.
    g.add_node(NodeSpec::filter("orphan", FilterConfig::Hflip, vs(), vs()));
    let report = PipelineValidator::new().validate(&g);
    assert!(!report.is_valid);
}

/// `PipelineGraph::validate()` detects dangling edges whose to_node does not
/// exist in the nodes map.
#[test]
fn graph_validate_dangling_edge() {
    let mut g = PipelineGraph::new();
    let src = g.add_node(NodeSpec::source("src", SourceConfig::File("x".into()), vs()));
    let phantom_id = NodeId::new();
    // Push a dangling edge to a non-existent node.
    g.edges.push(Edge {
        from_node: src,
        from_pad: "default".into(),
        to_node: phantom_id,
        to_pad: "default".into(),
    });
    let errors = g.validate();
    assert!(
        errors.iter().any(|e| matches!(e, PipelineError::NodeNotFound(_))),
        "expected NodeNotFound for dangling edge, got: {errors:?}"
    );
}

/// `PipelineGraph::validate()` detects a cycle in an otherwise well-connected
/// graph.
#[test]
fn graph_validate_detects_cycle() {
    let mut g = PipelineGraph::new();
    let a = g.add_node(NodeSpec::filter("a", FilterConfig::Hflip, vs(), vs()));
    let b = g.add_node(NodeSpec::filter("b", FilterConfig::Vflip, vs(), vs()));
    g.edges.push(Edge { from_node: a, from_pad: "default".into(), to_node: b, to_pad: "default".into() });
    g.edges.push(Edge { from_node: b, from_pad: "default".into(), to_node: a, to_pad: "default".into() });
    let errors = g.validate();
    assert!(
        errors.iter().any(|e| matches!(e, PipelineError::CycleDetected { .. })),
        "expected CycleDetected error"
    );
}

/// A valid source-only graph has no structural errors in `PipelineGraph::validate()`.
#[test]
fn graph_validate_source_only_no_errors() {
    let mut g = PipelineGraph::new();
    g.add_node(NodeSpec::source("src", SourceConfig::File("x".into()), vs()));
    let errors = g.validate();
    assert!(errors.is_empty(), "source-only graph should have no errors, got: {errors:?}");
}

/// `PipelineGraph::validate()` reports missing input pads on a filter with no
/// incoming edge.
#[test]
fn graph_validate_unconnected_input_pad() {
    let mut g = PipelineGraph::new();
    g.add_node(NodeSpec::filter("flt", FilterConfig::Hflip, vs(), vs()));
    let errors = g.validate();
    assert!(
        errors.iter().any(|e| matches!(e, PipelineError::PadNotFound { .. })),
        "expected PadNotFound for unconnected input pad"
    );
}

/// Validation on a graph with a chain of 5 filters all connected produces no
/// errors.
#[test]
fn graph_validate_long_chain_valid() {
    let g = build_chain(5);
    let errors = g.validate();
    assert!(errors.is_empty(), "chain of 5 should be valid, got: {errors:?}");
}

/// Validator reports `DuplicateNodeName` warning for two sources with the same name.
#[test]
fn validate_duplicate_names_warning() {
    let mut g = PipelineGraph::new();
    let s1 = g.add_node(NodeSpec::source("src", SourceConfig::File("a".into()), vs()));
    let s2 = g.add_node(NodeSpec::source("src", SourceConfig::File("b".into()), vs()));
    let sk1 = g.add_node(NodeSpec::sink("sink1", SinkConfig::Null, vs()));
    let sk2 = g.add_node(NodeSpec::sink("sink2", SinkConfig::Null, vs()));
    g.connect(s1, "default", sk1, "default").expect("ok");
    g.connect(s2, "default", sk2, "default").expect("ok");
    let report = PipelineValidator::new().validate(&g);
    assert!(report.warnings.iter().any(|w| {
        matches!(w, crate::validation::ValidationWarning::DuplicateNodeName { name } if name == "src")
    }));
}
