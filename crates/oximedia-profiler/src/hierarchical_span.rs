//! Hierarchical span tracking for nested function timing.
//!
//! Builds on `crate::span::SpanTracker` to provide post-hoc analysis
//! structures: `SpanTree` (depth, self-time, total-time, child spans) and
//! `SpanReport` (formatted tree output).
//!
//! The thread-local span stack in `crate::span` provides zero-contention
//! recording; this module focuses on the *analysis* side.
//!
//! # Example
//!
//! ```
//! use oximedia_profiler::span::SpanTracker;
//! use oximedia_profiler::hierarchical_span::{SpanTree, SpanReport};
//!
//! let tracker = SpanTracker::new();
//! {
//!     let _outer = tracker.enter("render");
//!     {
//!         let _inner = tracker.enter("decode");
//!         std::thread::sleep(std::time::Duration::from_millis(1));
//!     }
//! }
//!
//! let trees = SpanTree::from_tracker(&tracker);
//! let report = SpanReport::new(&trees);
//! println!("{}", report.format());
//! ```

#![allow(dead_code)]

use crate::span::{Span, SpanId, SpanTracker};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

// ---------------------------------------------------------------------------
// SpanTree
// ---------------------------------------------------------------------------

/// A single node in a hierarchical span tree.
///
/// `SpanTree` mirrors the parent/child structure stored in `Span` but
/// pre-computes derived metrics that are useful for analysis:
///
/// * **total_time** --- wall-clock duration from span open to span close.
/// * **self_time** --- `total_time` minus the sum of direct children's
///   `total_time` (i.e. time spent in *this* span, not descendants).
/// * **depth** --- nesting level (root = 0).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanTree {
    /// The span id.
    pub id: SpanId,
    /// Human-readable name.
    pub name: String,
    /// Nesting depth (0 = root).
    pub depth: u32,
    /// Total wall-clock duration of this span.
    pub total_time: Duration,
    /// Time attributable exclusively to this span (excluding children).
    pub self_time: Duration,
    /// Direct child trees.
    pub children: Vec<SpanTree>,
}

impl SpanTree {
    // -- constructors -------------------------------------------------------

    /// Builds all root-level `SpanTree`s from the given `SpanTracker`.
    ///
    /// Only closed spans are included; open spans are silently skipped.
    #[must_use]
    pub fn from_tracker(tracker: &SpanTracker) -> Vec<SpanTree> {
        let all_spans = tracker.all_spans();
        let lookup: HashMap<SpanId, &Span> = all_spans.iter().map(|s| (s.id, s)).collect();
        let root_ids = tracker.root_span_ids();

        let mut trees = Vec::new();
        for rid in &root_ids {
            if let Some(tree) = Self::build_tree(*rid, &lookup, 0) {
                trees.push(tree);
            }
        }
        trees
    }

    /// Builds all root-level `SpanTree`s from a raw slice of spans.
    #[must_use]
    pub fn from_spans(spans: &[Span]) -> Vec<SpanTree> {
        let lookup: HashMap<SpanId, &Span> = spans.iter().map(|s| (s.id, s)).collect();

        // Find roots (spans with no parent).
        let roots: Vec<SpanId> = spans
            .iter()
            .filter(|s| s.parent_id.is_none())
            .map(|s| s.id)
            .collect();

        let mut trees = Vec::new();
        for rid in &roots {
            if let Some(tree) = Self::build_tree(*rid, &lookup, 0) {
                trees.push(tree);
            }
        }
        trees
    }

    fn build_tree(id: SpanId, lookup: &HashMap<SpanId, &Span>, depth: u32) -> Option<SpanTree> {
        let span = lookup.get(&id)?;
        let total_time = span.duration()?;

        let mut child_trees = Vec::new();
        let mut children_total = Duration::ZERO;

        for &child_id in &span.children {
            if let Some(child_tree) = Self::build_tree(child_id, lookup, depth + 1) {
                children_total += child_tree.total_time;
                child_trees.push(child_tree);
            }
        }

        let self_time = total_time.saturating_sub(children_total);

        Some(SpanTree {
            id,
            name: span.name.clone(),
            depth,
            total_time,
            self_time,
            children: child_trees,
        })
    }

    // -- queries ------------------------------------------------------------

    /// Returns the total number of nodes in this tree (including `self`).
    #[must_use]
    pub fn node_count(&self) -> usize {
        1 + self.children.iter().map(|c| c.node_count()).sum::<usize>()
    }

    /// Returns the maximum depth of this tree.
    #[must_use]
    pub fn max_depth(&self) -> u32 {
        self.children
            .iter()
            .map(|c| c.max_depth())
            .max()
            .unwrap_or(self.depth)
    }

    /// Collects all nodes in pre-order traversal.
    #[must_use]
    pub fn flatten(&self) -> Vec<&SpanTree> {
        let mut out = vec![self];
        for child in &self.children {
            out.extend(child.flatten());
        }
        out
    }

    /// Returns the node with the highest self-time in this tree.
    #[must_use]
    pub fn hottest_self_time(&self) -> &SpanTree {
        let mut hottest = self;
        for child in &self.children {
            let candidate = child.hottest_self_time();
            if candidate.self_time > hottest.self_time {
                hottest = candidate;
            }
        }
        hottest
    }

    /// Returns `self_time` as a fraction of `total_time` (0.0 to 1.0).
    #[must_use]
    pub fn self_time_fraction(&self) -> f64 {
        let total_ns = self.total_time.as_nanos();
        if total_ns == 0 {
            return 0.0;
        }
        self.self_time.as_nanos() as f64 / total_ns as f64
    }
}

// ---------------------------------------------------------------------------
// SpanReport
// ---------------------------------------------------------------------------

/// Formatted report of a span tree hierarchy.
///
/// Produces a human-readable, indented text representation of the span tree
/// with timing information.
#[derive(Debug, Clone)]
pub struct SpanReport {
    lines: Vec<String>,
}

impl SpanReport {
    /// Creates a report from a slice of root `SpanTree` nodes.
    #[must_use]
    pub fn new(trees: &[SpanTree]) -> Self {
        let mut lines = Vec::new();
        lines.push("=== Hierarchical Span Report ===".to_owned());
        lines.push(String::new());
        for tree in trees {
            Self::format_tree(tree, &mut lines);
        }
        Self { lines }
    }

    fn format_tree(tree: &SpanTree, lines: &mut Vec<String>) {
        let indent = "  ".repeat(tree.depth as usize);
        let total_us = tree.total_time.as_micros();
        let self_us = tree.self_time.as_micros();
        let pct = tree.self_time_fraction() * 100.0;
        lines.push(format!(
            "{}{} [total: {}us, self: {}us ({:.1}%)]",
            indent, tree.name, total_us, self_us, pct,
        ));
        for child in &tree.children {
            Self::format_tree(child, lines);
        }
    }

    /// Returns the formatted report as a single string.
    #[must_use]
    pub fn format(&self) -> String {
        self.lines.join("\n")
    }

    /// Returns the individual report lines.
    #[must_use]
    pub fn lines(&self) -> &[String] {
        &self.lines
    }

    /// Returns the number of lines in the report.
    #[must_use]
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }
}

impl std::fmt::Display for SpanReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format())
    }
}

// ---------------------------------------------------------------------------
// SpanSummaryEntry — flat summary sorted by self-time
// ---------------------------------------------------------------------------

/// A flat summary entry for one span, used for ranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanSummaryEntry {
    /// Span name.
    pub name: String,
    /// Nesting depth.
    pub depth: u32,
    /// Total wall-clock time.
    pub total_time: Duration,
    /// Self (exclusive) time.
    pub self_time: Duration,
    /// Number of child spans.
    pub child_count: usize,
}

/// Collects a flat list of entries from a slice of `SpanTree` roots, sorted
/// by self-time descending (hottest first).
#[must_use]
pub fn ranked_by_self_time(trees: &[SpanTree]) -> Vec<SpanSummaryEntry> {
    let mut entries = Vec::new();
    for tree in trees {
        collect_entries(tree, &mut entries);
    }
    entries.sort_by(|a, b| b.self_time.cmp(&a.self_time));
    entries
}

fn collect_entries(tree: &SpanTree, out: &mut Vec<SpanSummaryEntry>) {
    out.push(SpanSummaryEntry {
        name: tree.name.clone(),
        depth: tree.depth,
        total_time: tree.total_time,
        self_time: tree.self_time,
        child_count: tree.children.len(),
    });
    for child in &tree.children {
        collect_entries(child, out);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::span::SpanTracker;
    use std::thread;
    use std::time::Duration;

    fn make_tracker_with_nested_spans() -> SpanTracker {
        let tracker = SpanTracker::new();
        {
            let _root = tracker.enter("root");
            thread::sleep(Duration::from_millis(5));
            {
                let _child_a = tracker.enter("child_a");
                thread::sleep(Duration::from_millis(5));
                {
                    let _grandchild = tracker.enter("grandchild");
                    thread::sleep(Duration::from_millis(5));
                }
            }
            {
                let _child_b = tracker.enter("child_b");
                thread::sleep(Duration::from_millis(5));
            }
        }
        tracker
    }

    #[test]
    fn test_span_tree_from_tracker_basic() {
        let tracker = make_tracker_with_nested_spans();
        let trees = SpanTree::from_tracker(&tracker);
        assert_eq!(trees.len(), 1);
        assert_eq!(trees[0].name, "root");
        assert_eq!(trees[0].depth, 0);
    }

    #[test]
    fn test_span_tree_children_present() {
        let tracker = make_tracker_with_nested_spans();
        let trees = SpanTree::from_tracker(&tracker);
        let root = &trees[0];
        assert_eq!(root.children.len(), 2);
        let names: Vec<&str> = root.children.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"child_a"));
        assert!(names.contains(&"child_b"));
    }

    #[test]
    fn test_span_tree_depth_computation() {
        let tracker = make_tracker_with_nested_spans();
        let trees = SpanTree::from_tracker(&tracker);
        let root = &trees[0];
        assert_eq!(root.depth, 0);

        let child_a = root.children.iter().find(|c| c.name == "child_a");
        assert!(child_a.is_some());
        let child_a = child_a.expect("checked above");
        assert_eq!(child_a.depth, 1);

        assert_eq!(child_a.children.len(), 1);
        assert_eq!(child_a.children[0].depth, 2);
        assert_eq!(child_a.children[0].name, "grandchild");
    }

    #[test]
    fn test_self_time_less_than_total_time() {
        let tracker = make_tracker_with_nested_spans();
        let trees = SpanTree::from_tracker(&tracker);
        let root = &trees[0];
        // Root has children, so self_time < total_time
        assert!(root.self_time < root.total_time);
    }

    #[test]
    fn test_leaf_self_time_equals_total_time() {
        let tracker = make_tracker_with_nested_spans();
        let trees = SpanTree::from_tracker(&tracker);
        let root = &trees[0];
        let child_b = root
            .children
            .iter()
            .find(|c| c.name == "child_b")
            .expect("child_b exists");
        // child_b is a leaf node
        assert!(child_b.children.is_empty());
        assert_eq!(child_b.self_time, child_b.total_time);
    }

    #[test]
    fn test_node_count() {
        let tracker = make_tracker_with_nested_spans();
        let trees = SpanTree::from_tracker(&tracker);
        // root -> child_a -> grandchild, child_b = 4 nodes
        assert_eq!(trees[0].node_count(), 4);
    }

    #[test]
    fn test_flatten_returns_preorder() {
        let tracker = make_tracker_with_nested_spans();
        let trees = SpanTree::from_tracker(&tracker);
        let flat = trees[0].flatten();
        assert_eq!(flat.len(), 4);
        assert_eq!(flat[0].name, "root");
    }

    #[test]
    fn test_span_report_format() {
        let tracker = make_tracker_with_nested_spans();
        let trees = SpanTree::from_tracker(&tracker);
        let report = SpanReport::new(&trees);
        let text = report.format();
        assert!(text.contains("Hierarchical Span Report"));
        assert!(text.contains("root"));
        assert!(text.contains("child_a"));
        assert!(text.contains("grandchild"));
        assert!(text.contains("child_b"));
    }

    #[test]
    fn test_span_report_display_trait() {
        let tracker = make_tracker_with_nested_spans();
        let trees = SpanTree::from_tracker(&tracker);
        let report = SpanReport::new(&trees);
        let s = format!("{}", report);
        assert!(s.contains("Hierarchical Span Report"));
    }

    #[test]
    fn test_ranked_by_self_time_ordering() {
        let tracker = make_tracker_with_nested_spans();
        let trees = SpanTree::from_tracker(&tracker);
        let ranked = ranked_by_self_time(&trees);
        assert_eq!(ranked.len(), 4);
        // Verify descending self-time order
        for window in ranked.windows(2) {
            assert!(window[0].self_time >= window[1].self_time);
        }
    }

    #[test]
    fn test_self_time_fraction() {
        let tracker = SpanTracker::new();
        {
            let _root = tracker.enter("only");
            thread::sleep(Duration::from_millis(5));
        }
        let trees = SpanTree::from_tracker(&tracker);
        let frac = trees[0].self_time_fraction();
        // Leaf: self_time == total_time, so fraction ~= 1.0
        assert!((frac - 1.0).abs() < 0.01, "fraction was {}", frac);
    }

    #[test]
    fn test_from_spans_matches_from_tracker() {
        let tracker = make_tracker_with_nested_spans();
        let all_spans = tracker.all_spans();
        let trees_from_tracker = SpanTree::from_tracker(&tracker);
        let trees_from_spans = SpanTree::from_spans(&all_spans);
        assert_eq!(trees_from_tracker.len(), trees_from_spans.len());
        assert_eq!(
            trees_from_tracker[0].node_count(),
            trees_from_spans[0].node_count()
        );
    }

    #[test]
    fn test_hottest_self_time() {
        let tracker = make_tracker_with_nested_spans();
        let trees = SpanTree::from_tracker(&tracker);
        let hottest = trees[0].hottest_self_time();
        // The hottest should have self_time >= any other node
        let flat = trees[0].flatten();
        for node in &flat {
            assert!(hottest.self_time >= node.self_time);
        }
    }

    #[test]
    fn test_empty_tracker_produces_empty_trees() {
        let tracker = SpanTracker::new();
        let trees = SpanTree::from_tracker(&tracker);
        assert!(trees.is_empty());
    }

    #[test]
    fn test_report_line_count() {
        let tracker = make_tracker_with_nested_spans();
        let trees = SpanTree::from_tracker(&tracker);
        let report = SpanReport::new(&trees);
        // Header line + blank line + 4 span lines = 6
        assert_eq!(report.line_count(), 6);
    }
}
