//! High-performance DAG index for operation-log traversal.
//!
//! For large edit histories (>10 K operations), the naïve linear-scan approach
//! used in [`crate::operation_log`] becomes prohibitive.  This module builds an
//! auxiliary adjacency structure — a **DAG index** — on top of an
//! `OperationLog` so that ancestry queries, causal-order traversals, and
//! topological sorts can run in amortised O(1) / O(V + E) time rather than
//! O(N²).
//!
//! The main entry point is [`DagIndex`].  After construction it can be kept
//! incrementally up-to-date as new operations arrive via [`DagIndex::add`].

use crate::operation_log::Operation;
use std::collections::{HashMap, HashSet, VecDeque};

// ─────────────────────────────────────────────────────────────────────────────
// DagIndex
// ─────────────────────────────────────────────────────────────────────────────

/// A DAG index that accelerates ancestor and descendant queries on a set of
/// [`Operation`] nodes.
///
/// Internally the index stores two adjacency maps:
/// * `children`: op_id → set of direct child ids (reverse edge).
/// * `parents`:  op_id → direct parent id (there is at most one because the
///   operation log is a singly-rooted chain with optional branches).
///
/// Additionally it maintains a **depth** map (`depth[id]` = number of edges
/// from the root to `id`) to allow O(1) fast-path checks such as
/// "is A an ancestor of B?".
#[derive(Debug, Default)]
pub struct DagIndex {
    /// Maps each operation id to its direct parent id (if any).
    parents: HashMap<u64, Option<u64>>,
    /// Maps each operation id to its set of direct children.
    children: HashMap<u64, HashSet<u64>>,
    /// Topological depth of each node (root = 0).
    depth: HashMap<u64, u64>,
}

impl DagIndex {
    /// Create an empty index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a fresh `DagIndex` from a slice of operations.
    ///
    /// Operations may be supplied in any order; the index handles out-of-order
    /// insertion gracefully.
    #[must_use]
    pub fn build(ops: &[Operation]) -> Self {
        let mut idx = Self::new();
        for op in ops {
            idx.add(op);
        }
        idx
    }

    /// Incrementally add a single operation to the index.
    ///
    /// Idempotent: re-adding an already-known operation is a no-op.
    pub fn add(&mut self, op: &Operation) {
        if self.parents.contains_key(&op.id) {
            // Already indexed.
            return;
        }

        // Register the parent link.
        self.parents.insert(op.id, op.parent_id);

        // Ensure child set exists for this node.
        self.children.entry(op.id).or_default();

        // Register reverse edge parent → child.
        if let Some(parent) = op.parent_id {
            self.children.entry(parent).or_default().insert(op.id);
        }

        // Compute depth.
        let depth = match op.parent_id {
            None => 0,
            Some(pid) => {
                // Parent may not be indexed yet when ops arrive out of order.
                self.depth.get(&pid).copied().unwrap_or(0) + 1
            }
        };
        self.depth.insert(op.id, depth);

        // Fix up depths of already-known children whose depth was computed
        // with a missing parent (they got depth 0 + 1 = 1).  We propagate
        // the correct depth downward via BFS.
        self.propagate_depths(op.id, depth);
    }

    /// Fix up depth values for all descendants of `root_id` starting from
    /// `root_depth`.  Called after adding a node whose children may already
    /// have been inserted with an approximated depth.
    fn propagate_depths(&mut self, root_id: u64, root_depth: u64) {
        let mut queue: VecDeque<(u64, u64)> = VecDeque::new();
        queue.push_back((root_id, root_depth));

        while let Some((cur, depth)) = queue.pop_front() {
            // Collect children to avoid holding a borrow.
            let children: Vec<u64> = self
                .children
                .get(&cur)
                .map(|s| s.iter().copied().collect())
                .unwrap_or_default();

            for child in children {
                let expected = depth + 1;
                let current = self.depth.entry(child).or_insert(expected);
                if *current != expected {
                    *current = expected;
                    queue.push_back((child, expected));
                }
            }
        }
    }

    /// Return the depth (distance from the root) of `op_id`, or `None` if
    /// the operation is unknown to the index.
    #[must_use]
    pub fn depth_of(&self, op_id: u64) -> Option<u64> {
        self.depth.get(&op_id).copied()
    }

    /// Return the direct parent id of `op_id`, or `None` if it is a root
    /// operation or unknown.
    #[must_use]
    pub fn parent_of(&self, op_id: u64) -> Option<u64> {
        self.parents.get(&op_id).copied().flatten()
    }

    /// Return the direct children of `op_id`.
    #[must_use]
    pub fn children_of(&self, op_id: u64) -> Vec<u64> {
        self.children
            .get(&op_id)
            .map(|s| {
                let mut v: Vec<u64> = s.iter().copied().collect();
                v.sort_unstable();
                v
            })
            .unwrap_or_default()
    }

    /// Check whether `ancestor_id` is an ancestor of (or equal to)
    /// `descendant_id`.
    ///
    /// Uses depth to short-circuit: if `ancestor`'s depth ≥ `descendant`'s
    /// depth it cannot be an ancestor (assuming no cycles).  Then walks the
    /// parent chain from `descendant` upward.
    #[must_use]
    pub fn is_ancestor(&self, ancestor_id: u64, descendant_id: u64) -> bool {
        if ancestor_id == descendant_id {
            return true;
        }

        // Short-circuit by depth: an ancestor must have strictly smaller depth.
        let a_depth = self.depth.get(&ancestor_id).copied().unwrap_or(u64::MAX);
        let d_depth = self.depth.get(&descendant_id).copied().unwrap_or(0);
        if a_depth >= d_depth {
            return false;
        }

        // Walk the parent chain from descendant toward root.
        let mut cur = descendant_id;
        while let Some(Some(parent)) = self.parents.get(&cur) {
            if *parent == ancestor_id {
                return true;
            }
            cur = *parent;
        }
        false
    }

    /// Return the set of **all** ancestors of `op_id` (inclusive of `op_id`
    /// itself) as a `HashSet`.  Uses BFS up the parent chain.
    #[must_use]
    pub fn ancestors(&self, op_id: u64) -> HashSet<u64> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(op_id);
        visited.insert(op_id);

        while let Some(cur) = queue.pop_front() {
            if let Some(Some(parent)) = self.parents.get(&cur) {
                if visited.insert(*parent) {
                    queue.push_back(*parent);
                }
            }
        }
        visited
    }

    /// Return the set of **all** descendants of `op_id` (inclusive) using
    /// BFS over the children map.
    #[must_use]
    pub fn descendants(&self, op_id: u64) -> HashSet<u64> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(op_id);
        visited.insert(op_id);

        while let Some(cur) = queue.pop_front() {
            if let Some(children) = self.children.get(&cur) {
                for &child in children {
                    if visited.insert(child) {
                        queue.push_back(child);
                    }
                }
            }
        }
        visited
    }

    /// Find the most-recent common ancestor (MRCA) of `id1` and `id2`.
    ///
    /// Returns `None` if the two nodes have no common ancestor (disconnected
    /// graph).
    ///
    /// Algorithm: collect ancestor-sets of both nodes and pick the common
    /// element with the greatest depth.
    #[must_use]
    pub fn mrca(&self, id1: u64, id2: u64) -> Option<u64> {
        let ancestors1 = self.ancestors(id1);
        let ancestors2 = self.ancestors(id2);

        ancestors1
            .intersection(&ancestors2)
            .max_by_key(|&&id| self.depth.get(&id).copied().unwrap_or(0))
            .copied()
    }

    /// Produce a topological ordering of all indexed operations (Kahn's
    /// algorithm).  Operations with no parent come first; within a level,
    /// ordering is by id for determinism.
    #[must_use]
    pub fn topological_order(&self) -> Vec<u64> {
        // Build in-degree map.
        let mut in_degree: HashMap<u64, usize> = HashMap::new();
        for &id in self.parents.keys() {
            in_degree.entry(id).or_insert(0);
            if let Some(Some(parent)) = self.parents.get(&id) {
                *in_degree.entry(*parent).or_insert(0) += 0; // ensure parent is in map
                *in_degree.entry(id).or_insert(0) += 1;
            }
        }

        // Collect roots (nodes with in-degree 0).
        let mut queue: std::collections::BinaryHeap<std::cmp::Reverse<u64>> =
            std::collections::BinaryHeap::new();
        for (&id, &deg) in &in_degree {
            if deg == 0 {
                queue.push(std::cmp::Reverse(id));
            }
        }

        let mut result = Vec::with_capacity(in_degree.len());
        while let Some(std::cmp::Reverse(cur)) = queue.pop() {
            result.push(cur);
            let children: Vec<u64> = self
                .children
                .get(&cur)
                .map(|s| {
                    let mut v: Vec<u64> = s.iter().copied().collect();
                    v.sort_unstable();
                    v
                })
                .unwrap_or_default();
            for child in children {
                let deg = in_degree.entry(child).or_insert(1);
                if *deg <= 1 {
                    queue.push(std::cmp::Reverse(child));
                } else {
                    *deg -= 1;
                }
            }
        }
        result
    }

    /// Total number of operations tracked by the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.parents.len()
    }

    /// Return `true` if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.parents.is_empty()
    }

    /// Remove all data from the index.
    pub fn clear(&mut self) {
        self.parents.clear();
        self.children.clear();
        self.depth.clear();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operation_log::{OpType, Operation};

    fn make_op(id: u64, parent: Option<u64>) -> Operation {
        let mut op = Operation::new(id, 1, 0, "p", OpType::Delete { index: 0 });
        op.parent_id = parent;
        op
    }

    /// Build a linear chain: 1 → 2 → 3 → 4 → 5
    fn linear_chain() -> DagIndex {
        let ops: Vec<Operation> = (1u64..=5)
            .map(|i| make_op(i, if i == 1 { None } else { Some(i - 1) }))
            .collect();
        DagIndex::build(&ops)
    }

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn test_build_and_len() {
        let idx = linear_chain();
        assert_eq!(idx.len(), 5);
        assert!(!idx.is_empty());
    }

    #[test]
    fn test_empty_index() {
        let idx = DagIndex::new();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
    }

    // ── Depth ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_depth_linear_chain() {
        let idx = linear_chain();
        for i in 1u64..=5 {
            assert_eq!(
                idx.depth_of(i),
                Some(i - 1),
                "node {} should be at depth {}",
                i,
                i - 1
            );
        }
    }

    #[test]
    fn test_depth_unknown_node() {
        let idx = linear_chain();
        assert_eq!(idx.depth_of(99), None);
    }

    // ── Parent / child ───────────────────────────────────────────────────────

    #[test]
    fn test_parent_of() {
        let idx = linear_chain();
        assert_eq!(idx.parent_of(1), None); // root
        assert_eq!(idx.parent_of(3), Some(2));
        assert_eq!(idx.parent_of(5), Some(4));
    }

    #[test]
    fn test_children_of() {
        let idx = linear_chain();
        assert_eq!(idx.children_of(1), vec![2]);
        assert_eq!(idx.children_of(5), vec![]); // leaf
    }

    // ── Ancestry ─────────────────────────────────────────────────────────────

    #[test]
    fn test_is_ancestor_direct() {
        let idx = linear_chain();
        assert!(idx.is_ancestor(1, 5)); // 1 is ancestor of 5
        assert!(idx.is_ancestor(3, 5)); // 3 is ancestor of 5
        assert!(!idx.is_ancestor(5, 1)); // leaf is not ancestor of root
        assert!(!idx.is_ancestor(3, 2)); // 3 is not ancestor of 2 (reverse)
    }

    #[test]
    fn test_is_ancestor_self() {
        let idx = linear_chain();
        assert!(idx.is_ancestor(3, 3));
    }

    #[test]
    fn test_ancestors_of_root() {
        let idx = linear_chain();
        let anc = idx.ancestors(1);
        // Root has only itself as ancestor.
        assert_eq!(anc.len(), 1);
        assert!(anc.contains(&1));
    }

    #[test]
    fn test_ancestors_of_leaf() {
        let idx = linear_chain();
        let anc = idx.ancestors(5);
        assert_eq!(anc.len(), 5); // includes itself + 1..4
        for i in 1u64..=5 {
            assert!(anc.contains(&i));
        }
    }

    // ── Descendants ──────────────────────────────────────────────────────────

    #[test]
    fn test_descendants_of_root() {
        let idx = linear_chain();
        let desc = idx.descendants(1);
        assert_eq!(desc.len(), 5); // all nodes
    }

    #[test]
    fn test_descendants_of_leaf() {
        let idx = linear_chain();
        let desc = idx.descendants(5);
        assert_eq!(desc.len(), 1); // leaf has only itself
    }

    // ── MRCA ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_mrca_linear_chain() {
        let idx = linear_chain();
        // mrca(3, 5) should be 3 (3 is an ancestor of 5)
        assert_eq!(idx.mrca(3, 5), Some(3));
    }

    #[test]
    fn test_mrca_self() {
        let idx = linear_chain();
        assert_eq!(idx.mrca(2, 2), Some(2));
    }

    #[test]
    fn test_mrca_branching() {
        // Topology: 1 → 2 → 3, 2 → 4
        let mut idx = DagIndex::new();
        idx.add(&make_op(1, None));
        idx.add(&make_op(2, Some(1)));
        idx.add(&make_op(3, Some(2)));
        idx.add(&make_op(4, Some(2)));

        // mrca(3, 4) should be 2 (their common parent)
        assert_eq!(idx.mrca(3, 4), Some(2));
        // mrca(1, 4) should be 1
        assert_eq!(idx.mrca(1, 4), Some(1));
    }

    // ── Topological order ────────────────────────────────────────────────────

    #[test]
    fn test_topological_order_linear() {
        let idx = linear_chain();
        let order = idx.topological_order();
        // The linear chain must appear in order 1 2 3 4 5.
        assert_eq!(order, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_topological_order_branching() {
        // 1 → 2 → 3, 2 → 4.  Valid topo orders include [1,2,3,4] and [1,2,4,3].
        let mut idx = DagIndex::new();
        idx.add(&make_op(1, None));
        idx.add(&make_op(2, Some(1)));
        idx.add(&make_op(3, Some(2)));
        idx.add(&make_op(4, Some(2)));
        let order = idx.topological_order();
        // 1 must come before 2; 2 must come before 3 and 4.
        let pos: HashMap<u64, usize> = order.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        assert!(pos[&1] < pos[&2]);
        assert!(pos[&2] < pos[&3]);
        assert!(pos[&2] < pos[&4]);
    }

    // ── Incremental add ───────────────────────────────────────────────────────

    #[test]
    fn test_incremental_add_then_query() {
        let mut idx = DagIndex::new();
        idx.add(&make_op(10, None));
        idx.add(&make_op(20, Some(10)));
        assert_eq!(idx.depth_of(10), Some(0));
        assert_eq!(idx.depth_of(20), Some(1));
        assert!(idx.is_ancestor(10, 20));
        assert!(!idx.is_ancestor(20, 10));
    }

    #[test]
    fn test_idempotent_add() {
        let mut idx = DagIndex::new();
        let op = make_op(5, None);
        idx.add(&op);
        idx.add(&op); // re-add should be a no-op
        assert_eq!(idx.len(), 1);
    }

    // ── Clear ────────────────────────────────────────────────────────────────

    #[test]
    fn test_clear() {
        let mut idx = linear_chain();
        idx.clear();
        assert!(idx.is_empty());
        assert_eq!(idx.depth_of(1), None);
    }
}
