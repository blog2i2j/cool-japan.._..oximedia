//! Flame graph data structure generation from profiling spans.
//!
//! This module provides a `FlameGraphBuilder` that accumulates per-stack-trace
//! samples (each identified by a slice of frame names and a duration in
//! microseconds) and then aggregates them into a `FlameGraph` tree.
//!
//! The resulting tree can be queried for the hottest call path, the top-N
//! nodes by self time, and serialised to the folded format consumed by
//! `flamegraph.pl` and `inferno`.
//!
//! # Example
//!
//! ```
//! use oximedia_profiler::flame_graph::FlameGraphBuilder;
//!
//! let mut builder = FlameGraphBuilder::new();
//! builder.record(&["main", "render", "draw_frame"], 1_200).unwrap();
//! builder.record(&["main", "render", "composite"], 800).unwrap();
//! builder.record(&["main", "audio_mix"], 300).unwrap();
//! let fg = builder.build().unwrap();
//! println!("total nodes: {}", fg.total_nodes());
//! ```

use std::collections::HashMap;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can arise during flame graph construction.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FlameError {
    /// A sample was submitted with an empty call stack.
    #[error("Call stack must contain at least one frame")]
    EmptyStack,

    /// The builder could not produce a valid graph (e.g. zero samples).
    #[error("Build failed: {0}")]
    BuildFailed(String),
}

// ---------------------------------------------------------------------------
// FlameNode
// ---------------------------------------------------------------------------

/// A single node in the flame graph call tree.
#[derive(Debug, Clone)]
pub struct FlameNode {
    /// Name of the function / pipeline stage represented by this node.
    pub name: String,
    /// Total microseconds attributed to this node and all its descendants.
    pub total_us: u64,
    /// Microseconds attributed exclusively to this node (not children).
    pub self_us: u64,
    /// How many times this node appeared as the leaf of a recorded stack.
    pub call_count: u64,
    /// Child nodes, ordered by insertion (stable for determinism in tests).
    pub children: Vec<FlameNode>,
}

impl FlameNode {
    /// Create a new node with the given name and zero counters.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            total_us: 0,
            self_us: 0,
            call_count: 0,
            children: Vec::new(),
        }
    }

    /// Total count of nodes in this subtree (including self).
    pub fn subtree_node_count(&self) -> usize {
        1 + self
            .children
            .iter()
            .map(|c| c.subtree_node_count())
            .sum::<usize>()
    }

    /// Walk the path `frames[1..]` into this node's children, creating nodes
    /// as needed, and accumulate `duration_us` into every node on the path.
    fn insert_path(&mut self, frames: &[&str], duration_us: u64) {
        if frames.is_empty() {
            // This node is the leaf — attribute self time here.
            self.self_us = self.self_us.saturating_add(duration_us);
            self.call_count += 1;
            return;
        }

        let child_name = frames[0];
        // Find or create the child with this name.
        let pos = self
            .children
            .iter()
            .position(|c| c.name == child_name)
            .unwrap_or_else(|| {
                self.children.push(FlameNode::new(child_name));
                self.children.len() - 1
            });

        let child = &mut self.children[pos];
        child.total_us = child.total_us.saturating_add(duration_us);
        child.insert_path(&frames[1..], duration_us);
    }

    /// Emit this node (and its subtree) in folded format into `out`.
    fn collect_folded<'a>(&'a self, path: &mut Vec<&'a str>, out: &mut Vec<String>) {
        path.push(&self.name);
        if self.children.is_empty() {
            out.push(format!("{} {}", path.join(";"), self.total_us));
        } else {
            for child in &self.children {
                child.collect_folded(path, out);
            }
        }
        path.pop();
    }
}

// ---------------------------------------------------------------------------
// FlameGraph
// ---------------------------------------------------------------------------

/// A built, immutable flame graph ready for querying and serialisation.
#[derive(Debug)]
pub struct FlameGraph {
    /// Synthetic root node whose children are the top-level stack frames.
    pub root: FlameNode,
    /// Total duration (µs) of all recorded samples.
    pub total_duration_us: u64,
}

impl FlameGraph {
    /// Total number of nodes in the entire tree (excluding the synthetic root).
    pub fn total_nodes(&self) -> usize {
        self.root
            .children
            .iter()
            .map(|c| c.subtree_node_count())
            .sum()
    }

    /// DFS traversal returning the chain of node names with the maximum
    /// `total_us` at each level (greedy hottest-child descent).
    ///
    /// The returned `Vec` starts at a top-level frame and ends at a leaf.
    /// Returns an empty `Vec` if the graph has no nodes.
    pub fn hottest_path(&self) -> Vec<String> {
        let mut path = Vec::new();
        let mut current_children = &self.root.children;

        loop {
            let hottest = current_children
                .iter()
                .max_by_key(|n| n.total_us);
            match hottest {
                None => break,
                Some(node) => {
                    path.push(node.name.clone());
                    current_children = &node.children;
                }
            }
        }

        path
    }

    /// Returns the top `n` nodes across the entire tree sorted by `self_us`
    /// descending.  Each entry is `(name, self_us)`.
    ///
    /// Nodes with `self_us == 0` are excluded.
    pub fn self_time_top_n(&self, n: usize) -> Vec<(&str, u64)> {
        let mut all: Vec<(&str, u64)> = Vec::new();
        collect_self_times(&self.root, &mut all);
        all.sort_by(|a, b| b.1.cmp(&a.1));
        all.into_iter().take(n).collect()
    }

    /// Serialise this flame graph to the folded text format.
    ///
    /// Each line has the form `frame1;frame2;...;leaf total_us\n`,
    /// compatible with `flamegraph.pl` and `inferno-flamegraph`.
    pub fn to_folded(&self) -> String {
        let mut lines: Vec<String> = Vec::new();
        for child in &self.root.children {
            child.collect_folded(&mut Vec::new(), &mut lines);
        }
        lines.sort();
        let mut out = lines.join("\n");
        if !out.is_empty() {
            out.push('\n');
        }
        out
    }
}

/// Recursively collect `(name, self_us)` for every node with `self_us > 0`.
fn collect_self_times<'a>(node: &'a FlameNode, out: &mut Vec<(&'a str, u64)>) {
    if node.self_us > 0 {
        out.push((&node.name, node.self_us));
    }
    for child in &node.children {
        collect_self_times(child, out);
    }
}

// ---------------------------------------------------------------------------
// FlameGraphBuilder
// ---------------------------------------------------------------------------

/// Incrementally accumulates call-stack samples and builds a `FlameGraph`.
#[derive(Debug, Default)]
pub struct FlameGraphBuilder {
    /// Per-root-frame subtrees indexed by root-frame name.
    ///
    /// We store the index into `roots_vec` in the map for O(1) lookup while
    /// preserving insertion order via the vec.
    roots_index: HashMap<String, usize>,
    roots_vec: Vec<FlameNode>,
    total_duration_us: u64,
    sample_count: u64,
}

impl FlameGraphBuilder {
    /// Create a new, empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a single sample.
    ///
    /// `stack` must contain at least one frame name.  The first element is
    /// the outermost (root) frame; the last element is the leaf that
    /// receives the self-time attribution.
    ///
    /// Returns `Err(FlameError::EmptyStack)` if `stack` is empty.
    pub fn record(&mut self, stack: &[&str], duration_us: u64) -> Result<(), FlameError> {
        if stack.is_empty() {
            return Err(FlameError::EmptyStack);
        }

        self.total_duration_us = self.total_duration_us.saturating_add(duration_us);
        self.sample_count += 1;

        let root_name = stack[0];
        let idx = if let Some(&i) = self.roots_index.get(root_name) {
            i
        } else {
            let i = self.roots_vec.len();
            self.roots_vec.push(FlameNode::new(root_name));
            self.roots_index.insert(root_name.to_owned(), i);
            i
        };

        let root_node = &mut self.roots_vec[idx];
        root_node.total_us = root_node.total_us.saturating_add(duration_us);
        root_node.insert_path(&stack[1..], duration_us);

        Ok(())
    }

    /// Consume the builder and produce a `FlameGraph`.
    ///
    /// Returns `Err(FlameError::BuildFailed)` if no samples have been
    /// recorded.
    pub fn build(self) -> Result<FlameGraph, FlameError> {
        if self.sample_count == 0 {
            return Err(FlameError::BuildFailed("no samples recorded".to_owned()));
        }

        // Wrap all roots under a single synthetic root for uniform traversal.
        let mut root = FlameNode::new("<root>");
        root.total_us = self.total_duration_us;
        root.children = self.roots_vec;

        Ok(FlameGraph {
            root,
            total_duration_us: self.total_duration_us,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a graph from a list of (stack, duration_us) pairs.
    fn build(samples: &[(&[&str], u64)]) -> FlameGraph {
        let mut b = FlameGraphBuilder::new();
        for (stack, dur) in samples {
            b.record(stack, *dur).expect("record should not fail");
        }
        b.build().expect("build should not fail")
    }

    // ------------------------------------------------------------------
    // FlameGraphBuilder record / error cases
    // ------------------------------------------------------------------

    #[test]
    fn test_empty_stack_returns_error() {
        let mut b = FlameGraphBuilder::new();
        let err = b.record(&[], 100);
        assert!(matches!(err, Err(FlameError::EmptyStack)));
    }

    #[test]
    fn test_build_with_no_samples_fails() {
        let b = FlameGraphBuilder::new();
        let err = b.build();
        assert!(matches!(err, Err(FlameError::BuildFailed(_))));
    }

    // ------------------------------------------------------------------
    // Single stack
    // ------------------------------------------------------------------

    #[test]
    fn test_single_stack_total_duration() {
        let fg = build(&[(&["main", "render"], 500)]);
        assert_eq!(fg.total_duration_us, 500);
    }

    #[test]
    fn test_single_stack_self_time_on_leaf() {
        let fg = build(&[(&["main", "render"], 300)]);
        // "render" is the leaf; it should own the self time.
        let top = fg.self_time_top_n(10);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].0, "render");
        assert_eq!(top[0].1, 300);
    }

    #[test]
    fn test_single_stack_total_nodes() {
        // stack depth 3 → 3 nodes
        let fg = build(&[(&["a", "b", "c"], 100)]);
        assert_eq!(fg.total_nodes(), 3);
    }

    // ------------------------------------------------------------------
    // Nested / overlapping stacks
    // ------------------------------------------------------------------

    #[test]
    fn test_nested_stacks_accumulate_total_us() {
        let fg = build(&[
            (&["main", "render"], 400),
            (&["main", "audio"], 200),
        ]);
        // "main" total_us should be 600
        let main_node = fg
            .root
            .children
            .iter()
            .find(|c| c.name == "main")
            .expect("main node must exist");
        assert_eq!(main_node.total_us, 600);
    }

    #[test]
    fn test_self_time_correct_for_shared_parent() {
        // "main" is never a leaf — self_us should be 0.
        let fg = build(&[
            (&["main", "render"], 100),
            (&["main", "audio"], 50),
        ]);
        let self_times = fg.self_time_top_n(10);
        // Only "render" and "audio" should appear.
        let names: Vec<&str> = self_times.iter().map(|(n, _)| *n).collect();
        assert!(!names.contains(&"main"),
            "main should have self_us == 0, names: {:?}", names);
    }

    // ------------------------------------------------------------------
    // Hottest path
    // ------------------------------------------------------------------

    #[test]
    fn test_hottest_path_follows_max_total_us() {
        let fg = build(&[
            (&["main", "slow", "decode"], 900),
            (&["main", "fast"], 100),
        ]);
        let path = fg.hottest_path();
        assert_eq!(path, vec!["main", "slow", "decode"]);
    }

    #[test]
    fn test_hottest_path_empty_graph_is_impossible_but_single_leaf() {
        let fg = build(&[(&["leaf"], 42)]);
        let path = fg.hottest_path();
        assert_eq!(path, vec!["leaf"]);
    }

    // ------------------------------------------------------------------
    // Folded format
    // ------------------------------------------------------------------

    #[test]
    fn test_folded_format_basic() {
        let fg = build(&[(&["main", "render"], 500)]);
        let folded = fg.to_folded();
        // Should contain "main;render 500\n"
        assert!(folded.contains("main;render 500"), "folded was: {folded}");
    }

    #[test]
    fn test_folded_format_multiple_stacks() {
        let fg = build(&[
            (&["main", "a"], 300),
            (&["main", "b"], 200),
        ]);
        let folded = fg.to_folded();
        assert!(folded.contains("main;a 300"), "folded was: {folded}");
        assert!(folded.contains("main;b 200"), "folded was: {folded}");
    }

    // ------------------------------------------------------------------
    // total_nodes count
    // ------------------------------------------------------------------

    #[test]
    fn test_total_nodes_two_roots() {
        let fg = build(&[
            (&["root_a", "child"], 100),
            (&["root_b"], 50),
        ]);
        // root_a + child + root_b = 3
        assert_eq!(fg.total_nodes(), 3);
    }

    // ------------------------------------------------------------------
    // self_time_top_n ordering
    // ------------------------------------------------------------------

    #[test]
    fn test_self_time_top_n_ordering() {
        let fg = build(&[
            (&["main", "slow_leaf"], 800),
            (&["main", "fast_leaf"], 200),
        ]);
        let top2 = fg.self_time_top_n(2);
        assert_eq!(top2[0].0, "slow_leaf");
        assert_eq!(top2[1].0, "fast_leaf");
    }
}
