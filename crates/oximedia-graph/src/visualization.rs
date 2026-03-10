//! Graph visualization: layout computation and ASCII art rendering.
//!
//! Provides a BFS-based left-to-right layer layout algorithm and a simple
//! ASCII art renderer for quick textual inspection of a processing graph.

// ── NodePosition ─────────────────────────────────────────────────────────────

/// A 2-D position for a node on the visualization canvas.
#[derive(Debug, Clone, PartialEq)]
pub struct NodePosition {
    /// Horizontal coordinate (pixels or arbitrary units).
    pub x: f32,
    /// Vertical coordinate (pixels or arbitrary units).
    pub y: f32,
}

impl NodePosition {
    /// Creates a new position.
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Returns the Euclidean distance from `self` to `other`.
    pub fn distance(&self, other: &Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

// ── BoundingBox ──────────────────────────────────────────────────────────────

/// An axis-aligned bounding rectangle.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundingBox {
    /// Left edge.
    pub min_x: f32,
    /// Top edge.
    pub min_y: f32,
    /// Right edge.
    pub max_x: f32,
    /// Bottom edge.
    pub max_y: f32,
}

impl BoundingBox {
    /// Creates a new bounding box.
    pub fn new(min_x: f32, min_y: f32, max_x: f32, max_y: f32) -> Self {
        Self {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    /// Width of the bounding box.
    pub fn width(&self) -> f32 {
        self.max_x - self.min_x
    }

    /// Height of the bounding box.
    pub fn height(&self) -> f32 {
        self.max_y - self.min_y
    }

    /// Centre point of the bounding box.
    pub fn center(&self) -> NodePosition {
        NodePosition::new(
            (self.min_x + self.max_x) * 0.5,
            (self.min_y + self.max_y) * 0.5,
        )
    }

    /// Returns `true` if `pos` lies within (or on the boundary of) this box.
    pub fn contains(&self, pos: &NodePosition) -> bool {
        pos.x >= self.min_x && pos.x <= self.max_x && pos.y >= self.min_y && pos.y <= self.max_y
    }
}

// ── GraphLayout ──────────────────────────────────────────────────────────────

/// A mapping of node IDs to 2-D canvas positions.
#[derive(Debug, Default)]
pub struct GraphLayout {
    /// (node_id, position) pairs.
    pub positions: Vec<(u64, NodePosition)>,
}

impl GraphLayout {
    /// Creates an empty layout.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the position of `node_id`, or `None` if not yet placed.
    pub fn get_position(&self, node_id: u64) -> Option<&NodePosition> {
        self.positions
            .iter()
            .find(|(id, _)| *id == node_id)
            .map(|(_, pos)| pos)
    }

    /// Sets or updates the position for `node_id`.
    pub fn set_position(&mut self, node_id: u64, pos: NodePosition) {
        if let Some(entry) = self.positions.iter_mut().find(|(id, _)| *id == node_id) {
            entry.1 = pos;
        } else {
            self.positions.push((node_id, pos));
        }
    }

    /// Computes a column-based left-to-right layout using BFS layering.
    ///
    /// `nodes` is the ordered list of all node IDs; `edges` is a list of
    /// `(from_id, to_id)` pairs.  Nodes in the same BFS depth layer are
    /// stacked vertically, spaced 100 units apart.  Layers are spaced 200
    /// units horizontally.
    pub fn auto_layout_left_right(nodes: &[u64], edges: &[(u64, u64)]) -> Self {
        use std::collections::{HashMap, VecDeque};

        // Build adjacency and in-degree maps.
        let mut in_degree: HashMap<u64, usize> = nodes.iter().map(|&id| (id, 0)).collect();
        let mut successors: HashMap<u64, Vec<u64>> =
            nodes.iter().map(|&id| (id, Vec::new())).collect();

        for &(from, to) in edges {
            if in_degree.contains_key(&from) && in_degree.contains_key(&to) {
                *in_degree.entry(to).or_insert(0) += 1;
                successors.entry(from).or_default().push(to);
            }
        }

        // BFS from zero-in-degree nodes, assigning each node a column (layer).
        let mut layer: HashMap<u64, usize> = HashMap::new();
        let mut queue: VecDeque<u64> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&id, _)| id)
            .collect();

        // Use BFS order (deterministic via sort).
        let mut seed: Vec<u64> = queue.drain(..).collect();
        seed.sort_unstable();
        queue.extend(seed);

        // Restart properly.
        let mut visited: std::collections::HashSet<u64> = std::collections::HashSet::new();
        while let Some(id) = queue.pop_front() {
            if visited.contains(&id) {
                continue;
            }
            visited.insert(id);
            let current_layer = *layer.entry(id).or_insert(0);
            let mut nexts: Vec<u64> = successors.get(&id).cloned().unwrap_or_default();
            nexts.sort_unstable();
            for next in nexts {
                let next_layer = layer.entry(next).or_insert(0);
                if *next_layer <= current_layer {
                    *next_layer = current_layer + 1;
                }
                queue.push_back(next);
            }
        }

        // Assign positions: group nodes by layer, stack vertically.
        let mut layer_counts: HashMap<usize, usize> = HashMap::new();
        let mut positions: Vec<(u64, NodePosition)> = nodes
            .iter()
            .map(|&id| {
                let col = *layer.get(&id).unwrap_or(&0);
                let row = {
                    let cnt = layer_counts.entry(col).or_insert(0);
                    let r = *cnt;
                    *cnt += 1;
                    r
                };
                let x = col as f32 * 200.0;
                let y = row as f32 * 100.0;
                (id, NodePosition::new(x, y))
            })
            .collect();

        // Sort for determinism.
        positions.sort_by(|a, b| a.0.cmp(&b.0));

        Self { positions }
    }
}

// ── GraphRenderer ─────────────────────────────────────────────────────────────

/// A simple canvas-based renderer that produces ASCII art.
pub struct GraphRenderer {
    /// Canvas width in character columns.
    pub canvas_width: u32,
    /// Canvas height in character rows.
    pub canvas_height: u32,
}

impl GraphRenderer {
    /// Creates a new renderer with the given canvas dimensions.
    pub fn new(canvas_width: u32, canvas_height: u32) -> Self {
        Self {
            canvas_width,
            canvas_height,
        }
    }

    /// Renders the layout as an ASCII art string.
    ///
    /// Each node is represented as `[name]` placed at its grid position
    /// (scaled to fit the canvas).  The returned string contains
    /// `canvas_height` lines separated by `'\n'`.
    pub fn render_ascii(&self, layout: &GraphLayout, nodes: &[(u64, String)]) -> String {
        if nodes.is_empty() || layout.positions.is_empty() {
            return String::new();
        }

        // Determine coordinate ranges for normalisation.
        let (min_x, max_x, min_y, max_y) = layout.positions.iter().fold(
            (f32::MAX, f32::MIN, f32::MAX, f32::MIN),
            |(ax, bx, ay, by), (_, p)| (ax.min(p.x), bx.max(p.x), ay.min(p.y), by.max(p.y)),
        );

        let range_x = (max_x - min_x).max(1.0);
        let range_y = (max_y - min_y).max(1.0);

        let cols = self.canvas_width as usize;
        let rows = self.canvas_height as usize;

        // Build a grid of spaces.
        let mut grid: Vec<Vec<char>> = vec![vec![' '; cols]; rows];

        for (id, name) in nodes {
            if let Some(pos) = layout.get_position(*id) {
                let col = ((pos.x - min_x) / range_x * (cols.saturating_sub(10)) as f32) as usize;
                let row = ((pos.y - min_y) / range_y * (rows.saturating_sub(3)) as f32) as usize;
                let label = format!("[{name}]");
                for (i, ch) in label.chars().enumerate() {
                    let c = col + i;
                    if row < rows && c < cols {
                        grid[row][c] = ch;
                    }
                }
            }
        }

        grid.iter()
            .map(|row| row.iter().collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // ── NodePosition ──────────────────────────────────────────────────────────

    #[test]
    fn distance_to_same_point_is_zero() {
        let p = NodePosition::new(3.0, 4.0);
        assert!((p.distance(&p)).abs() < 1e-6);
    }

    #[test]
    fn distance_3_4_5_triangle() {
        let a = NodePosition::new(0.0, 0.0);
        let b = NodePosition::new(3.0, 4.0);
        assert!((a.distance(&b) - 5.0).abs() < 1e-5);
    }

    // ── BoundingBox ───────────────────────────────────────────────────────────

    #[test]
    fn bounding_box_width_and_height() {
        let bb = BoundingBox::new(0.0, 0.0, 100.0, 50.0);
        assert!((bb.width() - 100.0).abs() < 1e-6);
        assert!((bb.height() - 50.0).abs() < 1e-6);
    }

    #[test]
    fn bounding_box_center() {
        let bb = BoundingBox::new(0.0, 0.0, 200.0, 100.0);
        let c = bb.center();
        assert!((c.x - 100.0).abs() < 1e-6);
        assert!((c.y - 50.0).abs() < 1e-6);
    }

    #[test]
    fn bounding_box_contains_interior_point() {
        let bb = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        assert!(bb.contains(&NodePosition::new(5.0, 5.0)));
    }

    #[test]
    fn bounding_box_excludes_exterior_point() {
        let bb = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        assert!(!bb.contains(&NodePosition::new(11.0, 5.0)));
    }

    #[test]
    fn bounding_box_contains_boundary_point() {
        let bb = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        assert!(bb.contains(&NodePosition::new(10.0, 10.0)));
    }

    // ── GraphLayout ───────────────────────────────────────────────────────────

    #[test]
    fn layout_set_and_get_position() {
        let mut layout = GraphLayout::new();
        layout.set_position(1, NodePosition::new(0.0, 0.0));
        let pos = layout.get_position(1).expect("get_position should succeed");
        assert!((pos.x - 0.0).abs() < 1e-6);
    }

    #[test]
    fn layout_update_position() {
        let mut layout = GraphLayout::new();
        layout.set_position(2, NodePosition::new(0.0, 0.0));
        layout.set_position(2, NodePosition::new(100.0, 200.0));
        let pos = layout.get_position(2).expect("get_position should succeed");
        assert!((pos.x - 100.0).abs() < 1e-6);
        // Only one entry for the node.
        assert_eq!(
            layout.positions.iter().filter(|(id, _)| *id == 2).count(),
            1
        );
    }

    #[test]
    fn layout_get_missing_returns_none() {
        let layout = GraphLayout::new();
        assert!(layout.get_position(99).is_none());
    }

    #[test]
    fn auto_layout_places_all_nodes() {
        let nodes = vec![1u64, 2, 3];
        let edges = vec![(1u64, 2u64), (2, 3)];
        let layout = GraphLayout::auto_layout_left_right(&nodes, &edges);
        assert_eq!(layout.positions.len(), 3);
        for &id in &nodes {
            assert!(layout.get_position(id).is_some(), "missing node {id}");
        }
    }

    #[test]
    fn auto_layout_source_is_leftmost() {
        // node 1 (source) -> node 2 -> node 3 (sink)
        let nodes = vec![1u64, 2, 3];
        let edges = vec![(1u64, 2u64), (2, 3)];
        let layout = GraphLayout::auto_layout_left_right(&nodes, &edges);
        let x1 = layout
            .get_position(1)
            .expect("get_position should succeed")
            .x;
        let x2 = layout
            .get_position(2)
            .expect("get_position should succeed")
            .x;
        let x3 = layout
            .get_position(3)
            .expect("get_position should succeed")
            .x;
        assert!(x1 <= x2, "source should be left of middle");
        assert!(x2 <= x3, "middle should be left of sink");
    }

    // ── GraphRenderer ─────────────────────────────────────────────────────────

    #[test]
    fn render_ascii_empty_graph_returns_empty() {
        let renderer = GraphRenderer::new(80, 24);
        let layout = GraphLayout::new();
        let result = renderer.render_ascii(&layout, &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn render_ascii_contains_node_name() {
        let mut layout = GraphLayout::new();
        layout.set_position(1, NodePosition::new(0.0, 0.0));
        let renderer = GraphRenderer::new(80, 24);
        let nodes = vec![(1u64, "source".to_string())];
        let output = renderer.render_ascii(&layout, &nodes);
        assert!(output.contains("source"), "expected 'source' in:\n{output}");
    }

    #[test]
    fn render_ascii_has_correct_line_count() {
        let mut layout = GraphLayout::new();
        layout.set_position(1, NodePosition::new(0.0, 0.0));
        let renderer = GraphRenderer::new(40, 10);
        let nodes = vec![(1u64, "n".to_string())];
        let output = renderer.render_ascii(&layout, &nodes);
        // canvas_height lines joined by '\n' gives canvas_height - 1 newlines,
        // so split gives canvas_height parts.
        assert_eq!(output.lines().count(), 10);
    }
}
