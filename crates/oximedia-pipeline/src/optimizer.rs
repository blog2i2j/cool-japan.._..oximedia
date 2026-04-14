//! Pipeline graph optimizer — node fusion and filter-chain optimisations.
//!
//! [`PipelineOptimizer`] applies a series of rewrite passes over a
//! [`PipelineGraph`] to reduce the number of nodes that need to be executed,
//! lower per-frame computational cost, and enable zero-copy data paths.
//!
//! # Passes available
//!
//! | Pass | Description |
//! |------|-------------|
//! | [`PipelineOptimizer::fuse_scale_crop`] | Merge a `Scale` node followed immediately by a `Crop` into a single `FusedScaleCrop` node. |
//! | [`PipelineOptimizer::eliminate_noops`] | Remove filter nodes whose [`FilterConfig::is_noop`] returns `true` (e.g. `Volume { gain_db: 0.0 }`). |
//! | [`PipelineOptimizer::reorder_by_cost`] | Re-sort linear filter chains so cheaper filters run before expensive ones. |
//! | [`PipelineOptimizer::run_all`] | Apply all passes in a sensible order. |
//!
//! # Example
//!
//! ```rust
//! use oximedia_pipeline::builder::PipelineBuilder;
//! use oximedia_pipeline::node::{SourceConfig, SinkConfig};
//! use oximedia_pipeline::optimizer::PipelineOptimizer;
//!
//! let graph = PipelineBuilder::new()
//!     .source("in", SourceConfig::File("input.mp4".into()))
//!     .scale(1920, 1080)
//!     .crop(0, 0, 1280, 720)
//!     .sink("out", SinkConfig::Null)
//!     .build()
//!     .expect("build ok");
//!
//! let original_nodes = graph.node_count();
//! let (optimized, _report) = PipelineOptimizer::new().fuse_scale_crop(graph).expect("fuse ok");
//! // One fewer node: Scale+Crop fused into a single FusedScaleCrop node.
//! assert!(optimized.node_count() < original_nodes);
//! ```

use std::collections::HashSet;

use crate::graph::{Edge, PipelineGraph};
use crate::node::{FilterConfig, NodeId, NodeSpec, NodeType};
use crate::PipelineError;

// ── FusedFilter ───────────────────────────────────────────────────────────────

/// A compound filter created by the optimizer when adjacent filters can be
/// executed in a single pass.
#[derive(Debug, Clone, PartialEq)]
pub enum FusedFilter {
    /// A `Scale` followed immediately by a `Crop` merged into one operation:
    /// scale the source to `(scale_w, scale_h)` then crop to
    /// `(x, y, crop_w, crop_h)`.
    ScaleCrop {
        /// Intermediate scaled width in pixels.
        scale_w: u32,
        /// Intermediate scaled height in pixels.
        scale_h: u32,
        /// Crop origin X in the scaled coordinate space.
        x: u32,
        /// Crop origin Y in the scaled coordinate space.
        y: u32,
        /// Final output width after cropping.
        crop_w: u32,
        /// Final output height after cropping.
        crop_h: u32,
    },
    /// A `Scale` followed by `Pad` — scale first, then pad to the target canvas.
    ScalePad {
        /// Target scale size.
        scale_w: u32,
        /// Target scale size.
        scale_h: u32,
        /// Final padded canvas width.
        pad_w: u32,
        /// Final padded canvas height.
        pad_h: u32,
    },
    /// A chain of `Hflip` and/or `Vflip` operations collapsed to a single
    /// combined flip (net flip state after cancellation).
    CombinedFlip {
        /// Net horizontal flip: `true` if an odd number of `Hflip`s remain.
        hflip: bool,
        /// Net vertical flip: `true` if an odd number of `Vflip`s remain.
        vflip: bool,
    },
}

// ── OptimizationReport ────────────────────────────────────────────────────────

/// Summary of what the optimizer changed in a single run.
#[derive(Debug, Clone, Default)]
pub struct OptimizationReport {
    /// Number of node pairs fused into single compound nodes.
    pub fusions_applied: u32,
    /// Number of no-op filter nodes removed.
    pub noops_eliminated: u32,
    /// Number of filter nodes whose order was swapped.
    pub reorders_applied: u32,
    /// Total number of nodes removed from the graph.
    pub nodes_removed: u32,
}

impl OptimizationReport {
    /// Returns `true` if the optimizer made any change.
    pub fn any_changes(&self) -> bool {
        self.fusions_applied > 0
            || self.noops_eliminated > 0
            || self.reorders_applied > 0
            || self.nodes_removed > 0
    }
}

// ── PipelineOptimizer ─────────────────────────────────────────────────────────

/// Applies structural rewrite passes to a [`PipelineGraph`].
///
/// `PipelineOptimizer` is stateless: it takes ownership of a graph (or clones
/// it), transforms it, and returns the optimised version together with an
/// [`OptimizationReport`].
#[derive(Debug, Clone, Default)]
pub struct PipelineOptimizer {
    /// When `true`, no-op elimination is active.
    pub eliminate_noops: bool,
    /// When `true`, linear chains are reordered by computational cost.
    pub reorder_by_cost: bool,
    /// When `true`, adjacent Scale+Crop / Scale+Pad nodes are fused.
    pub fuse_adjacent: bool,
    /// When `true`, consecutive flips are collapsed.
    pub collapse_flips: bool,
}

impl PipelineOptimizer {
    /// Create a default optimizer with all passes enabled.
    pub fn new() -> Self {
        Self {
            eliminate_noops: true,
            reorder_by_cost: true,
            fuse_adjacent: true,
            collapse_flips: true,
        }
    }

    /// Create an optimizer with all passes **disabled**.  Use builder methods
    /// to enable individual passes.
    pub fn none() -> Self {
        Self {
            eliminate_noops: false,
            reorder_by_cost: false,
            fuse_adjacent: false,
            collapse_flips: false,
        }
    }

    // ── Pass: fuse_scale_crop ─────────────────────────────────────────────────

    /// Fuse all `Scale → Crop` node pairs found in the graph.
    ///
    /// For each linear chain `A → Scale → Crop → B`, the two filter nodes are
    /// replaced with a single `FusedScaleCrop` node and the edges `A→Scale`,
    /// `Scale→Crop`, `Crop→B` are replaced with `A→Fused`, `Fused→B`.
    ///
    /// Returns the modified graph and an [`OptimizationReport`].
    pub fn fuse_scale_crop(
        &self,
        graph: PipelineGraph,
    ) -> Result<(PipelineGraph, OptimizationReport), PipelineError> {
        let mut report = OptimizationReport::default();
        let mut g = graph;

        loop {
            let fuse_candidate = find_scale_crop_pair(&g);
            match fuse_candidate {
                None => break,
                Some((scale_id, crop_id, fused_filter)) => {
                    g = apply_node_fusion(g, scale_id, crop_id, fused_filter)?;
                    report.fusions_applied += 1;
                    report.nodes_removed += 1;
                }
            }
        }

        Ok((g, report))
    }

    /// Fuse all `Scale → Pad` node pairs found in the graph.
    pub fn fuse_scale_pad(
        &self,
        graph: PipelineGraph,
    ) -> Result<(PipelineGraph, OptimizationReport), PipelineError> {
        let mut report = OptimizationReport::default();
        let mut g = graph;

        loop {
            let candidate = find_scale_pad_pair(&g);
            match candidate {
                None => break,
                Some((scale_id, pad_id, fused_filter)) => {
                    g = apply_node_fusion(g, scale_id, pad_id, fused_filter)?;
                    report.fusions_applied += 1;
                    report.nodes_removed += 1;
                }
            }
        }

        Ok((g, report))
    }

    // ── Pass: eliminate_noops ─────────────────────────────────────────────────

    /// Remove all filter nodes whose configuration is a no-operation.
    ///
    /// A no-op node is one where [`FilterConfig::is_noop`] returns `true`.
    /// Removing such a node re-wires each upstream node directly to each
    /// downstream node.
    pub fn eliminate_noops_pass(
        &self,
        graph: PipelineGraph,
    ) -> Result<(PipelineGraph, OptimizationReport), PipelineError> {
        let mut report = OptimizationReport::default();
        let mut g = graph;

        loop {
            let noop_candidate = find_noop_filter(&g);
            match noop_candidate {
                None => break,
                Some(noop_id) => {
                    g = remove_noop_node(g, noop_id)?;
                    report.noops_eliminated += 1;
                    report.nodes_removed += 1;
                }
            }
        }

        Ok((g, report))
    }

    // ── Pass: reorder_by_cost ─────────────────────────────────────────────────

    /// Reorder adjacent filter nodes in linear chains so cheaper operations
    /// precede more expensive ones.
    ///
    /// Only *pure* single-input, single-output filter nodes are considered for
    /// reordering.  Source, Sink, Split, Merge, and conditional nodes are never
    /// moved.
    pub fn reorder_by_cost_pass(
        &self,
        graph: PipelineGraph,
    ) -> Result<(PipelineGraph, OptimizationReport), PipelineError> {
        let mut report = OptimizationReport::default();
        let mut g = graph;

        // Collect all linear filter chains (topological order)
        let chains = collect_linear_chains(&g);

        for chain in &chains {
            if chain.len() < 2 {
                continue;
            }
            // Bubble-sort the chain by cost_estimate (stable, preserves semantics
            // for equal-cost nodes)
            let swaps = bubble_sort_chain(&mut g, chain)?;
            report.reorders_applied += swaps;
        }

        Ok((g, report))
    }

    // ── Pass: collapse_flips ──────────────────────────────────────────────────

    /// Collapse consecutive `Hflip` / `Vflip` filter nodes in a chain.
    ///
    /// An even number of identical flips cancel out (net = no-op).  An odd
    /// number reduces to a single flip.  The entire flip sub-chain is replaced
    /// by zero or one `CombinedFlip` node (represented as a `Custom` filter
    /// for compatibility with downstream code).
    pub fn collapse_flips_pass(
        &self,
        graph: PipelineGraph,
    ) -> Result<(PipelineGraph, OptimizationReport), PipelineError> {
        let mut report = OptimizationReport::default();
        let mut g = graph;

        loop {
            let flip_run = find_flip_run(&g);
            match flip_run {
                None => break,
                Some((run, hflip_count, vflip_count)) if run.len() >= 2 => {
                    let net_h = hflip_count % 2 != 0;
                    let net_v = vflip_count % 2 != 0;
                    g = collapse_flip_run(g, &run, net_h, net_v)?;
                    report.fusions_applied += 1;
                    report.nodes_removed += (run.len() as u32).saturating_sub(1);
                }
                _ => break,
            }
        }

        Ok((g, report))
    }

    // ── run_all ───────────────────────────────────────────────────────────────

    /// Apply all enabled passes in sequence and return the optimised graph with
    /// a combined [`OptimizationReport`].
    ///
    /// Pass order:
    /// 1. `collapse_flips`   (structural simplification first)
    /// 2. `eliminate_noops`  (removes identity nodes)
    /// 3. `fuse_scale_crop`  (compound node creation)
    /// 4. `fuse_scale_pad`
    /// 5. `reorder_by_cost`  (last, after topology is stabilised)
    pub fn run_all(
        &self,
        mut graph: PipelineGraph,
    ) -> Result<(PipelineGraph, OptimizationReport), PipelineError> {
        let mut combined = OptimizationReport::default();

        if self.collapse_flips {
            let (g, r) = self.collapse_flips_pass(graph)?;
            graph = g;
            combined.fusions_applied += r.fusions_applied;
            combined.nodes_removed += r.nodes_removed;
        }
        if self.eliminate_noops {
            let (g, r) = self.eliminate_noops_pass(graph)?;
            graph = g;
            combined.noops_eliminated += r.noops_eliminated;
            combined.nodes_removed += r.nodes_removed;
        }
        if self.fuse_adjacent {
            let (g, r) = self.fuse_scale_crop(graph)?;
            graph = g;
            combined.fusions_applied += r.fusions_applied;
            combined.nodes_removed += r.nodes_removed;

            let (g, r) = self.fuse_scale_pad(graph)?;
            graph = g;
            combined.fusions_applied += r.fusions_applied;
            combined.nodes_removed += r.nodes_removed;
        }
        if self.reorder_by_cost {
            let (g, r) = self.reorder_by_cost_pass(graph)?;
            graph = g;
            combined.reorders_applied += r.reorders_applied;
        }

        Ok((graph, combined))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Find the first `Scale → Crop` pair in the graph (linear chain only).
fn find_scale_crop_pair(
    g: &PipelineGraph,
) -> Option<(NodeId, NodeId, FusedFilter)> {
    for (id, spec) in &g.nodes {
        if let NodeType::Filter(FilterConfig::Scale { width: sw, height: sh }) = &spec.node_type {
            let sw = *sw;
            let sh = *sh;
            // Find single downstream neighbour
            if let Some(next_id) = single_downstream(g, *id) {
                if let Some(next_spec) = g.nodes.get(&next_id) {
                    if let NodeType::Filter(FilterConfig::Crop { x, y, w, h }) =
                        &next_spec.node_type
                    {
                        let fused = FusedFilter::ScaleCrop {
                            scale_w: sw,
                            scale_h: sh,
                            x: *x,
                            y: *y,
                            crop_w: *w,
                            crop_h: *h,
                        };
                        return Some((*id, next_id, fused));
                    }
                }
            }
        }
    }
    None
}

/// Find the first `Scale → Pad` pair in the graph (linear chain only).
fn find_scale_pad_pair(g: &PipelineGraph) -> Option<(NodeId, NodeId, FusedFilter)> {
    for (id, spec) in &g.nodes {
        if let NodeType::Filter(FilterConfig::Scale { width: sw, height: sh }) = &spec.node_type {
            let sw = *sw;
            let sh = *sh;
            if let Some(next_id) = single_downstream(g, *id) {
                if let Some(next_spec) = g.nodes.get(&next_id) {
                    if let NodeType::Filter(FilterConfig::Pad {
                        width: pw,
                        height: ph,
                    }) = &next_spec.node_type
                    {
                        let fused = FusedFilter::ScalePad {
                            scale_w: sw,
                            scale_h: sh,
                            pad_w: *pw,
                            pad_h: *ph,
                        };
                        return Some((*id, next_id, fused));
                    }
                }
            }
        }
    }
    None
}

/// Apply the actual fusion: replace `first_id` with a fused node and remove
/// `second_id`, rewiring edges accordingly.
fn apply_node_fusion(
    mut g: PipelineGraph,
    first_id: NodeId,
    second_id: NodeId,
    fused: FusedFilter,
) -> Result<PipelineGraph, PipelineError> {
    // Determine output stream spec from the second (downstream) node
    let output_spec = {
        let second = g
            .nodes
            .get(&second_id)
            .ok_or_else(|| PipelineError::NodeNotFound(second_id.to_string()))?;
        second
            .output_pads
            .first()
            .map(|(_, s)| s.clone())
            .unwrap_or_default()
    };

    // Determine input stream spec from the first node
    let input_spec = {
        let first = g
            .nodes
            .get(&first_id)
            .ok_or_else(|| PipelineError::NodeNotFound(first_id.to_string()))?;
        first
            .input_pads
            .first()
            .map(|(_, s)| s.clone())
            .unwrap_or_default()
    };

    let first_name = g
        .nodes
        .get(&first_id)
        .map(|s| s.name.clone())
        .unwrap_or_default();

    // Build the fused FilterConfig using Custom as a carrier
    let fused_config = fused_to_filter_config(&fused);

    // Build replacement node
    let fused_spec = NodeSpec {
        id: first_id,
        name: format!("fused_{first_name}"),
        node_type: NodeType::Filter(fused_config),
        input_pads: vec![("default".to_string(), input_spec)],
        output_pads: vec![("default".to_string(), output_spec)],
    };

    // Replace the first node with the fused node
    g.nodes.insert(first_id, fused_spec);

    // Collect edges that need to be updated:
    // - Edges from first → second become no longer needed (internal edge)
    // - Edges from second → X become first → X
    let mut new_edges: Vec<Edge> = Vec::new();
    for edge in &g.edges {
        if edge.from_node == second_id {
            // Re-wire: was second → X, now first → X
            new_edges.push(Edge {
                from_node: first_id,
                from_pad: edge.from_pad.clone(),
                to_node: edge.to_node,
                to_pad: edge.to_pad.clone(),
            });
        } else if edge.to_node == second_id || edge.from_node == second_id {
            // Skip the internal edge (first→second) and any other edge touching second
            // (already handled above for from_node == second_id)
        } else {
            new_edges.push(edge.clone());
        }
    }
    g.edges = new_edges;

    // Remove the second node
    g.nodes.remove(&second_id);

    Ok(g)
}

/// Convert a [`FusedFilter`] to a `FilterConfig::Custom` for storage.
fn fused_to_filter_config(fused: &FusedFilter) -> FilterConfig {
    match fused {
        FusedFilter::ScaleCrop {
            scale_w,
            scale_h,
            x,
            y,
            crop_w,
            crop_h,
        } => FilterConfig::Custom {
            name: "fused_scale_crop".to_string(),
            params: vec![
                ("scale_w".to_string(), scale_w.to_string()),
                ("scale_h".to_string(), scale_h.to_string()),
                ("x".to_string(), x.to_string()),
                ("y".to_string(), y.to_string()),
                ("crop_w".to_string(), crop_w.to_string()),
                ("crop_h".to_string(), crop_h.to_string()),
            ],
        },
        FusedFilter::ScalePad {
            scale_w,
            scale_h,
            pad_w,
            pad_h,
        } => FilterConfig::Custom {
            name: "fused_scale_pad".to_string(),
            params: vec![
                ("scale_w".to_string(), scale_w.to_string()),
                ("scale_h".to_string(), scale_h.to_string()),
                ("pad_w".to_string(), pad_w.to_string()),
                ("pad_h".to_string(), pad_h.to_string()),
            ],
        },
        FusedFilter::CombinedFlip { hflip, vflip } => FilterConfig::Custom {
            name: "combined_flip".to_string(),
            params: vec![
                ("hflip".to_string(), hflip.to_string()),
                ("vflip".to_string(), vflip.to_string()),
            ],
        },
    }
}

/// Find the single downstream neighbour of `node_id` if there is exactly one.
fn single_downstream(g: &PipelineGraph, node_id: NodeId) -> Option<NodeId> {
    let successors: Vec<NodeId> = g
        .edges
        .iter()
        .filter(|e| e.from_node == node_id)
        .map(|e| e.to_node)
        .collect();
    if successors.len() == 1 {
        Some(successors[0])
    } else {
        None
    }
}

/// Find the single upstream neighbour of `node_id` if there is exactly one.
fn single_upstream(g: &PipelineGraph, node_id: NodeId) -> Option<NodeId> {
    let predecessors: Vec<NodeId> = g
        .edges
        .iter()
        .filter(|e| e.to_node == node_id)
        .map(|e| e.from_node)
        .collect();
    if predecessors.len() == 1 {
        Some(predecessors[0])
    } else {
        None
    }
}

/// Find the first no-op filter node.
fn find_noop_filter(g: &PipelineGraph) -> Option<NodeId> {
    g.nodes.iter().find_map(|(id, spec)| {
        if let NodeType::Filter(cfg) = &spec.node_type {
            if cfg.is_noop() {
                return Some(*id);
            }
        }
        None
    })
}

/// Remove a no-op filter node, re-wiring its upstream to its downstream.
fn remove_noop_node(
    mut g: PipelineGraph,
    noop_id: NodeId,
) -> Result<PipelineGraph, PipelineError> {
    // Collect all upstream and downstream edges
    let in_edges: Vec<Edge> = g
        .edges
        .iter()
        .filter(|e| e.to_node == noop_id)
        .cloned()
        .collect();
    let out_edges: Vec<Edge> = g
        .edges
        .iter()
        .filter(|e| e.from_node == noop_id)
        .cloned()
        .collect();

    // Cross-product: each upstream directly to each downstream
    let mut bypass_edges: Vec<Edge> = Vec::new();
    for ie in &in_edges {
        for oe in &out_edges {
            bypass_edges.push(Edge {
                from_node: ie.from_node,
                from_pad: ie.from_pad.clone(),
                to_node: oe.to_node,
                to_pad: oe.to_pad.clone(),
            });
        }
    }

    // Rebuild edge list: remove edges touching noop_id, add bypass edges
    g.edges.retain(|e| e.to_node != noop_id && e.from_node != noop_id);
    g.edges.extend(bypass_edges);
    g.nodes.remove(&noop_id);

    Ok(g)
}

/// Collect linear filter-only chains (nodes with exactly one in-edge and one
/// out-edge that are `Filter` nodes).  Returns a `Vec<Vec<NodeId>>` where each
/// inner `Vec` is a maximal linear chain.
fn collect_linear_chains(g: &PipelineGraph) -> Vec<Vec<NodeId>> {
    let mut chains: Vec<Vec<NodeId>> = Vec::new();
    let mut visited: HashSet<NodeId> = HashSet::new();

    // Find all filter nodes that are the *start* of a linear chain (their
    // single predecessor is not a filter or has multiple successors)
    for (id, spec) in &g.nodes {
        if !matches!(spec.node_type, NodeType::Filter(_)) {
            continue;
        }
        if visited.contains(id) {
            continue;
        }
        // Check if this is the head of a chain
        let upstream = single_upstream(g, *id);
        let is_chain_head = match upstream {
            None => true,
            Some(prev_id) => {
                // Head if previous has multiple successors or is not a filter
                let prev_spec = g.nodes.get(&prev_id);
                match prev_spec {
                    None => true,
                    Some(ps) => {
                        !matches!(ps.node_type, NodeType::Filter(_))
                            || single_downstream(g, prev_id).is_none()
                    }
                }
            }
        };

        if !is_chain_head {
            continue;
        }

        // Walk forward through the chain
        let mut chain = vec![*id];
        visited.insert(*id);
        let mut cur = *id;
        loop {
            match single_downstream(g, cur) {
                None => break,
                Some(next) => {
                    if visited.contains(&next) {
                        break;
                    }
                    let next_spec = g.nodes.get(&next);
                    match next_spec {
                        None => break,
                        Some(ns) => {
                            if !matches!(ns.node_type, NodeType::Filter(_)) {
                                break;
                            }
                            // Make sure next has exactly one upstream (linear)
                            if single_upstream(g, next).is_none() {
                                break;
                            }
                            chain.push(next);
                            visited.insert(next);
                            cur = next;
                        }
                    }
                }
            }
        }

        if chain.len() >= 2 {
            chains.push(chain);
        }
    }

    chains
}

/// Bubble-sort a linear chain in-place by swapping adjacent node types.
/// Returns number of swaps performed.
fn bubble_sort_chain(
    g: &mut PipelineGraph,
    chain: &[NodeId],
) -> Result<u32, PipelineError> {
    let n = chain.len();
    let mut swaps = 0u32;

    for _ in 0..n {
        for i in 0..n.saturating_sub(1) {
            let a_id = chain[i];
            let b_id = chain[i + 1];
            let cost_a = node_cost(g, a_id);
            let cost_b = node_cost(g, b_id);
            if cost_a > cost_b {
                // Swap the node_type (and name) between the two nodes
                let node_a_type = g
                    .nodes
                    .get(&a_id)
                    .map(|s| s.node_type.clone())
                    .ok_or_else(|| PipelineError::NodeNotFound(a_id.to_string()))?;
                let node_b_type = g
                    .nodes
                    .get(&b_id)
                    .map(|s| s.node_type.clone())
                    .ok_or_else(|| PipelineError::NodeNotFound(b_id.to_string()))?;

                if let Some(a) = g.nodes.get_mut(&a_id) {
                    a.node_type = node_b_type;
                }
                if let Some(b) = g.nodes.get_mut(&b_id) {
                    b.node_type = node_a_type;
                }
                swaps += 1;
            }
        }
    }

    Ok(swaps)
}

/// Return the cost estimate for a node (0 for non-filter nodes).
fn node_cost(g: &PipelineGraph, id: NodeId) -> u32 {
    g.nodes.get(&id).map_or(0, |spec| {
        match &spec.node_type {
            NodeType::Filter(cfg) => cfg.cost_estimate(),
            _ => 0,
        }
    })
}

/// Find the first run of consecutive `Hflip`/`Vflip` filter nodes of length >= 2.
fn find_flip_run(g: &PipelineGraph) -> Option<(Vec<NodeId>, usize, usize)> {
    let chains = collect_linear_chains(g);
    for chain in chains {
        // Find the longest contiguous run of flips within this chain
        let mut run_start = 0;
        while run_start < chain.len() {
            if !is_flip_node(g, chain[run_start]) {
                run_start += 1;
                continue;
            }
            let mut run_end = run_start + 1;
            while run_end < chain.len() && is_flip_node(g, chain[run_end]) {
                run_end += 1;
            }
            if run_end - run_start >= 2 {
                let run: Vec<NodeId> = chain[run_start..run_end].to_vec();
                let mut hflips = 0;
                let mut vflips = 0;
                for &nid in &run {
                    if let Some(spec) = g.nodes.get(&nid) {
                        match &spec.node_type {
                            NodeType::Filter(FilterConfig::Hflip) => hflips += 1,
                            NodeType::Filter(FilterConfig::Vflip) => vflips += 1,
                            _ => {}
                        }
                    }
                }
                return Some((run, hflips, vflips));
            }
            run_start = run_end;
        }
    }
    None
}

fn is_flip_node(g: &PipelineGraph, id: NodeId) -> bool {
    g.nodes.get(&id).map_or(false, |spec| {
        matches!(
            spec.node_type,
            NodeType::Filter(FilterConfig::Hflip) | NodeType::Filter(FilterConfig::Vflip)
        )
    })
}

/// Collapse a flip run into zero or one combined flip node.
fn collapse_flip_run(
    mut g: PipelineGraph,
    run: &[NodeId],
    net_h: bool,
    net_v: bool,
) -> Result<PipelineGraph, PipelineError> {
    if run.is_empty() {
        return Ok(g);
    }

    let keep_id = run[0];
    let remove_ids = &run[1..];

    if net_h || net_v {
        // Update the first node to the combined flip
        let combined_cfg = FilterConfig::Custom {
            name: "combined_flip".to_string(),
            params: vec![
                ("hflip".to_string(), net_h.to_string()),
                ("vflip".to_string(), net_v.to_string()),
            ],
        };
        if let Some(spec) = g.nodes.get_mut(&keep_id) {
            spec.node_type = NodeType::Filter(combined_cfg);
            spec.name = "combined_flip".to_string();
        }
    } else {
        // Net result is identity — remove ALL nodes in the run (mark keep as noop)
        if let Some(spec) = g.nodes.get_mut(&keep_id) {
            spec.node_type = NodeType::Filter(FilterConfig::Volume { gain_db: 0.0 });
        }
    }

    // Remove all but the first node in the run
    for &rid in remove_ids {
        g = remove_noop_node_direct(g, rid)?;
    }

    Ok(g)
}

/// Remove a node, directly bypassing its edges (like `remove_noop_node` but
/// does not check `is_noop`).
fn remove_noop_node_direct(
    mut g: PipelineGraph,
    node_id: NodeId,
) -> Result<PipelineGraph, PipelineError> {
    let in_edges: Vec<Edge> = g
        .edges
        .iter()
        .filter(|e| e.to_node == node_id)
        .cloned()
        .collect();
    let out_edges: Vec<Edge> = g
        .edges
        .iter()
        .filter(|e| e.from_node == node_id)
        .cloned()
        .collect();

    let mut bypass_edges: Vec<Edge> = Vec::new();
    for ie in &in_edges {
        for oe in &out_edges {
            bypass_edges.push(Edge {
                from_node: ie.from_node,
                from_pad: ie.from_pad.clone(),
                to_node: oe.to_node,
                to_pad: oe.to_pad.clone(),
            });
        }
    }

    g.edges.retain(|e| e.to_node != node_id && e.from_node != node_id);
    g.edges.extend(bypass_edges);
    g.nodes.remove(&node_id);

    Ok(g)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::PipelineBuilder;
    use crate::node::{SinkConfig, SourceConfig};

    fn simple_pipeline_with_scale_crop() -> PipelineGraph {
        PipelineBuilder::new()
            .source("in", SourceConfig::File("in.mp4".into()))
            .scale(1920, 1080)
            .crop(0, 0, 1280, 720)
            .sink("out", SinkConfig::Null)
            .build()
            .unwrap()
    }

    fn simple_pipeline_with_noop() -> PipelineGraph {
        // Trim{0, 0} is a no-op and keeps stream kind = Video, compatible
        // with the default video source/sink pads.
        PipelineBuilder::new()
            .source("in", SourceConfig::File("in.mp4".into()))
            .trim(0, 0) // start == end → noop
            .sink("out", SinkConfig::Null)
            .build()
            .unwrap()
    }

    #[test]
    fn fuse_scale_crop_reduces_node_count() {
        let g = simple_pipeline_with_scale_crop();
        let before = g.node_count();
        let opt = PipelineOptimizer::new();
        let (optimized, report) = opt.fuse_scale_crop(g).unwrap();
        assert!(optimized.node_count() < before, "expected fewer nodes after fusion");
        assert_eq!(report.fusions_applied, 1);
        assert_eq!(report.nodes_removed, 1);
    }

    #[test]
    fn fused_node_has_custom_filter_name() {
        let g = simple_pipeline_with_scale_crop();
        let opt = PipelineOptimizer::new();
        let (optimized, _) = opt.fuse_scale_crop(g).unwrap();
        let has_fused = optimized.nodes.values().any(|spec| {
            matches!(&spec.node_type, NodeType::Filter(FilterConfig::Custom { name, .. }) if name == "fused_scale_crop")
        });
        assert!(has_fused, "expected a fused_scale_crop node");
    }

    #[test]
    fn eliminate_noops_removes_unity_volume() {
        let g = simple_pipeline_with_noop();
        let before = g.node_count();
        let opt = PipelineOptimizer::new();
        let (optimized, report) = opt.eliminate_noops_pass(g).unwrap();
        assert!(optimized.node_count() < before);
        assert_eq!(report.noops_eliminated, 1);
    }

    #[test]
    fn graph_without_noops_unchanged() {
        let g = PipelineBuilder::new()
            .source("in", SourceConfig::File("in.mp4".into()))
            .scale(1280, 720)
            .sink("out", SinkConfig::Null)
            .build()
            .unwrap();
        let before = g.node_count();
        let opt = PipelineOptimizer::new();
        let (optimized, report) = opt.eliminate_noops_pass(g).unwrap();
        assert_eq!(optimized.node_count(), before);
        assert_eq!(report.noops_eliminated, 0);
    }

    #[test]
    fn run_all_applies_multiple_passes() {
        let g = PipelineBuilder::new()
            .source("in", SourceConfig::File("in.mp4".into()))
            .trim(0, 0) // noop (start == end, video filter)
            .scale(1920, 1080)
            .crop(0, 0, 1280, 720)
            .sink("out", SinkConfig::Null)
            .build()
            .unwrap();
        let before = g.node_count();
        let opt = PipelineOptimizer::new();
        let (optimized, report) = opt.run_all(g).unwrap();
        // Should remove noop Volume and fuse Scale+Crop  →  2 nodes gone
        assert!(optimized.node_count() < before);
        assert!(report.any_changes());
    }

    #[test]
    fn optimizer_none_makes_no_changes() {
        let g = simple_pipeline_with_scale_crop();
        let before = g.node_count();
        let opt = PipelineOptimizer::none();
        let (optimized, report) = opt.run_all(g).unwrap();
        assert_eq!(optimized.node_count(), before);
        assert!(!report.any_changes());
    }

    #[test]
    fn fuse_scale_pad_reduces_node_count() {
        let g = PipelineBuilder::new()
            .source("in", SourceConfig::File("in.mp4".into()))
            .scale(1280, 720)
            .filter("pad", FilterConfig::Pad { width: 1920, height: 1080 })
            .sink("out", SinkConfig::Null)
            .build()
            .unwrap();
        let before = g.node_count();
        let opt = PipelineOptimizer::new();
        let (optimized, report) = opt.fuse_scale_pad(g).unwrap();
        assert!(optimized.node_count() < before);
        assert_eq!(report.fusions_applied, 1);
    }

    #[test]
    fn reorder_by_cost_swaps_expensive_before_cheap() {
        // Scale (cost 8) before Hflip (cost 1) — reorder should swap them.
        // The optimizer swaps *node_type in-place*, so after optimisation:
        //   - the first filter node (named "scale") will hold Hflip
        //   - the second filter node (named "hflip") will hold Scale
        let g = PipelineBuilder::new()
            .source("in", SourceConfig::File("in.mp4".into()))
            .scale(1920, 1080)
            .hflip()
            .sink("out", SinkConfig::Null)
            .build()
            .unwrap();
        let opt = PipelineOptimizer { reorder_by_cost: true, ..PipelineOptimizer::none() };
        let (optimized, report) = opt.reorder_by_cost_pass(g).unwrap();
        // At least one swap should occur
        assert!(report.reorders_applied > 0, "expected at least one reorder swap");
        // After swap: the node with the LOWER cost_estimate should appear first in
        // topological order. We check filter types by walking the topo order.
        let topo_ids = topological_order_ids(&optimized);
        let filter_ids: Vec<NodeId> = topo_ids
            .into_iter()
            .filter(|id| matches!(optimized.nodes.get(id), Some(s) if matches!(s.node_type, NodeType::Filter(_))))
            .collect();
        // There should be exactly 2 filter nodes
        assert_eq!(filter_ids.len(), 2, "expected 2 filter nodes");
        let first_cost = node_cost(&optimized, filter_ids[0]);
        let second_cost = node_cost(&optimized, filter_ids[1]);
        assert!(
            first_cost <= second_cost,
            "first filter (cost {first_cost}) should be cheaper than or equal to second (cost {second_cost})"
        );
    }

    /// Walk the graph in topological order and collect node IDs.
    fn topological_order_ids(g: &PipelineGraph) -> Vec<NodeId> {
        use std::collections::{HashMap, VecDeque};
        let mut in_degree: HashMap<NodeId, usize> = g.nodes.keys().map(|k| (*k, 0)).collect();
        for e in &g.edges {
            *in_degree.entry(e.to_node).or_insert(0) += 1;
        }
        let mut queue: VecDeque<NodeId> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&id, _)| id)
            .collect();
        let mut result = Vec::new();
        while let Some(id) = queue.pop_front() {
            result.push(id);
            for e in g.edges.iter().filter(|e| e.from_node == id) {
                let deg = in_degree.entry(e.to_node).or_insert(0);
                *deg = deg.saturating_sub(1);
                if *deg == 0 {
                    queue.push_back(e.to_node);
                }
            }
        }
        result
    }
}
