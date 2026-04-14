//! Hierarchical scene graph with transform inheritance, z-ordering, visibility,
//! and group opacity.
//!
//! A scene graph organises drawable nodes into a tree. Each node carries its own
//! local [`Transform2D`], opacity, visibility flag, z-order, and [`NodeContent`].
//! During rendering the transforms and opacities are accumulated from the root
//! downward so that every node is rendered in world-space.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// NodeId
// ─────────────────────────────────────────────────────────────────────────────

/// Opaque node identifier — guaranteed unique within a single [`SceneGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(u64);

// ─────────────────────────────────────────────────────────────────────────────
// Transform2D
// ─────────────────────────────────────────────────────────────────────────────

/// 2-D affine transform stored in column-major order as `[a, b, c, d, tx, ty]`.
///
/// The transform maps a point `(x, y)` to:
/// ```text
/// x' = a*x + c*y + tx
/// y' = b*x + d*y + ty
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Transform2D {
    /// `[a, b, c, d, tx, ty]` — column-major affine matrix.
    pub matrix: [f32; 6],
}

impl Transform2D {
    /// Identity transform.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            matrix: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        }
    }

    /// Pure translation.
    #[must_use]
    pub fn translate(tx: f32, ty: f32) -> Self {
        Self {
            matrix: [1.0, 0.0, 0.0, 1.0, tx, ty],
        }
    }

    /// Non-uniform scale around the origin.
    #[must_use]
    pub fn scale(sx: f32, sy: f32) -> Self {
        Self {
            matrix: [sx, 0.0, 0.0, sy, 0.0, 0.0],
        }
    }

    /// Counter-clockwise rotation (radians).
    #[must_use]
    pub fn rotate(angle_rad: f32) -> Self {
        let (s, c) = angle_rad.sin_cos();
        Self {
            matrix: [c, s, -s, c, 0.0, 0.0],
        }
    }

    /// Compose two transforms: returns `self * other` (i.e. `other` is applied first).
    #[must_use]
    pub fn compose(&self, other: &Transform2D) -> Transform2D {
        // self = [a0,b0,c0,d0,tx0,ty0]  other = [a1,b1,c1,d1,tx1,ty1]
        let [a0, b0, c0, d0, tx0, ty0] = self.matrix;
        let [a1, b1, c1, d1, tx1, ty1] = other.matrix;
        Transform2D {
            matrix: [
                a0 * a1 + c0 * b1,
                b0 * a1 + d0 * b1,
                a0 * c1 + c0 * d1,
                b0 * c1 + d0 * d1,
                a0 * tx1 + c0 * ty1 + tx0,
                b0 * tx1 + d0 * ty1 + ty0,
            ],
        }
    }

    /// Apply this transform to the point `(x, y)`, returning the world-space result.
    #[must_use]
    pub fn transform_point(&self, x: f32, y: f32) -> (f32, f32) {
        let [a, b, c, d, tx, ty] = self.matrix;
        (a * x + c * y + tx, b * x + d * y + ty)
    }
}

impl Default for Transform2D {
    fn default() -> Self {
        Self::identity()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rect
// ─────────────────────────────────────────────────────────────────────────────

/// Axis-aligned bounding rectangle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    /// Left edge X.
    pub x: f32,
    /// Top edge Y.
    pub y: f32,
    /// Width (non-negative).
    pub width: f32,
    /// Height (non-negative).
    pub height: f32,
}

impl Rect {
    /// Construct from components.
    #[must_use]
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NodeContent
// ─────────────────────────────────────────────────────────────────────────────

/// Describes what a single node draws.
#[derive(Debug, Clone)]
pub enum NodeContent {
    /// Container node — nothing is drawn for the node itself.
    Empty,
    /// A solid-colour filled rectangle at the local origin.
    ColorRect {
        /// Rectangle width in local units.
        width: f32,
        /// Rectangle height in local units.
        height: f32,
        /// RGBA fill colour.
        color: [u8; 4],
    },
    /// A text label rendered as coloured glyph blocks.
    Text {
        /// String to display.
        content: String,
        /// RGBA text colour.
        color: [u8; 4],
        /// Approximate font size in pixels.
        font_size: f32,
    },
    /// A solid-colour filled circle centred at the local origin.
    Circle {
        /// Circle radius in local units.
        radius: f32,
        /// RGBA fill colour.
        color: [u8; 4],
    },
    /// A line from the local origin `(0,0)` to `(x2, y2)`.
    Line {
        /// Line end X in local units.
        x2: f32,
        /// Line end Y in local units.
        y2: f32,
        /// RGBA stroke colour.
        color: [u8; 4],
        /// Stroke width in pixels.
        width: f32,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// SceneNode
// ─────────────────────────────────────────────────────────────────────────────

/// A single node in the scene graph.
#[derive(Debug, Clone)]
pub struct SceneNode {
    /// Unique node identifier.
    pub id: NodeId,
    /// Human-readable debug name.
    pub name: String,
    /// Local transform relative to the parent.
    pub transform: Transform2D,
    /// Opacity multiplier `[0.0, 1.0]`; multiplied with the parent's opacity.
    pub opacity: f32,
    /// When `false` the node and all its descendants are skipped during render.
    pub visible: bool,
    /// Z-order among siblings (lower value = rendered first = further behind).
    pub z_order: i32,
    /// Ordered list of child node IDs (order within the list is stable storage
    /// order; actual render order is determined by `z_order`).
    pub children: Vec<NodeId>,
    /// The drawable content of this node.
    pub content: NodeContent,
}

// ─────────────────────────────────────────────────────────────────────────────
// SceneGraph
// ─────────────────────────────────────────────────────────────────────────────

/// Hierarchical scene graph.
///
/// Nodes are stored in a flat `HashMap`; child/parent relationships are tracked
/// via the `children` list on each `SceneNode`.  A separate reverse-parent map
/// enables O(1) reparenting.
pub struct SceneGraph {
    nodes: HashMap<NodeId, SceneNode>,
    /// `node -> parent` reverse map (the root has no parent).
    parent: HashMap<NodeId, NodeId>,
    root: NodeId,
    next_id: u64,
}

impl SceneGraph {
    /// Create a new scene graph containing a single empty root node.
    #[must_use]
    pub fn new() -> Self {
        let root_id = NodeId(0);
        let root_node = SceneNode {
            id: root_id,
            name: "root".to_string(),
            transform: Transform2D::identity(),
            opacity: 1.0,
            visible: true,
            z_order: 0,
            children: Vec::new(),
            content: NodeContent::Empty,
        };
        let mut nodes = HashMap::new();
        nodes.insert(root_id, root_node);
        Self {
            nodes,
            parent: HashMap::new(),
            root: root_id,
            next_id: 1,
        }
    }

    /// Return the root node ID.
    #[must_use]
    pub fn root(&self) -> NodeId {
        self.root
    }

    /// Add a child node under `parent` with the given content.
    ///
    /// Returns `None` if `parent` does not exist in the graph.
    pub fn add_node(&mut self, parent: NodeId, content: NodeContent) -> Option<NodeId> {
        if !self.nodes.contains_key(&parent) {
            return None;
        }
        let id = NodeId(self.next_id);
        self.next_id += 1;
        let node = SceneNode {
            id,
            name: format!("node_{}", id.0),
            transform: Transform2D::identity(),
            opacity: 1.0,
            visible: true,
            z_order: 0,
            children: Vec::new(),
            content,
        };
        self.nodes.insert(id, node);
        self.parent.insert(id, parent);
        // Register as child of parent — we know the parent exists.
        if let Some(p) = self.nodes.get_mut(&parent) {
            p.children.push(id);
        }
        Some(id)
    }

    /// Remove a node and all of its descendants from the graph.
    ///
    /// Returns `true` if the node was found and removed.  The root node cannot
    /// be removed; attempting to do so returns `false`.
    pub fn remove_node(&mut self, id: NodeId) -> bool {
        if id == self.root {
            return false;
        }
        if !self.nodes.contains_key(&id) {
            return false;
        }
        // Collect all descendants (including `id`) via iterative DFS.
        let to_remove = self.collect_subtree(id);
        for node_id in &to_remove {
            self.nodes.remove(node_id);
            self.parent.remove(node_id);
        }
        // Remove `id` from its parent's children list.
        if let Some(par_id) = self.parent.get(&id).copied() {
            // parent entry was already removed above for `id` — but we stored a
            // copy before removal; now patch the parent node's children list.
            if let Some(par_node) = self.nodes.get_mut(&par_id) {
                par_node.children.retain(|c| *c != id);
            }
        }
        true
    }

    /// Move `id` from its current parent to `new_parent`.
    ///
    /// Returns `true` on success.  Fails if either node does not exist, if
    /// `id == root`, or if `new_parent` is a descendant of `id` (which would
    /// create a cycle).
    pub fn reparent(&mut self, id: NodeId, new_parent: NodeId) -> bool {
        if id == self.root {
            return false;
        }
        if !self.nodes.contains_key(&id) || !self.nodes.contains_key(&new_parent) {
            return false;
        }
        // Guard against cycles — new_parent must not be in the subtree of id.
        let subtree = self.collect_subtree(id);
        if subtree.contains(&new_parent) {
            return false;
        }
        // Detach from old parent.
        if let Some(old_parent) = self.parent.get(&id).copied() {
            if let Some(old_p_node) = self.nodes.get_mut(&old_parent) {
                old_p_node.children.retain(|c| *c != id);
            }
        }
        // Attach to new parent.
        self.parent.insert(id, new_parent);
        if let Some(np_node) = self.nodes.get_mut(&new_parent) {
            np_node.children.push(id);
        }
        true
    }

    /// Get an immutable reference to the node with the given ID.
    #[must_use]
    pub fn get(&self, id: NodeId) -> Option<&SceneNode> {
        self.nodes.get(&id)
    }

    /// Get a mutable reference to the node with the given ID.
    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut SceneNode> {
        self.nodes.get_mut(&id)
    }

    /// Total number of nodes in the graph (including the root).
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Render the scene graph into an RGBA pixel buffer.
    ///
    /// `pixels` must have length `width * height * 4`.  Pixels with coordinates
    /// outside `[0, width) × [0, height)` are silently clipped.  The buffer is
    /// not cleared before rendering; it is assumed to already contain the
    /// background (or all-zeroes for a transparent canvas).
    pub fn render(&self, pixels: &mut [u8], width: u32, height: u32) {
        if width == 0 || height == 0 || pixels.is_empty() {
            return;
        }
        let expected = (width as usize) * (height as usize) * 4;
        if pixels.len() < expected {
            return;
        }
        self.render_node(pixels, width, height, self.root, &Transform2D::identity(), 1.0);
    }

    // ── Private helpers ────────────────────────────────────────────────────

    /// Iterative DFS collecting the full subtree rooted at `id` (inclusive).
    fn collect_subtree(&self, id: NodeId) -> Vec<NodeId> {
        let mut result = Vec::new();
        let mut stack = vec![id];
        while let Some(current) = stack.pop() {
            result.push(current);
            if let Some(node) = self.nodes.get(&current) {
                for &child in &node.children {
                    stack.push(child);
                }
            }
        }
        result
    }

    /// Recursive depth-first render of a single node.
    fn render_node(
        &self,
        pixels: &mut [u8],
        width: u32,
        height: u32,
        id: NodeId,
        parent_transform: &Transform2D,
        parent_opacity: f32,
    ) {
        let node = match self.nodes.get(&id) {
            Some(n) => n,
            None => return,
        };

        if !node.visible {
            return;
        }

        let world_transform = parent_transform.compose(&node.transform);
        let eff_opacity = (parent_opacity * node.opacity).clamp(0.0, 1.0);

        // Render this node's content.
        match &node.content {
            NodeContent::Empty => {}
            NodeContent::ColorRect {
                width: w,
                height: h,
                color,
            } => {
                draw_rect(pixels, width, height, &world_transform, 0.0, 0.0, *w, *h, *color, eff_opacity);
            }
            NodeContent::Text {
                content,
                color,
                font_size,
            } => {
                draw_text(pixels, width, height, &world_transform, content, *color, *font_size, eff_opacity);
            }
            NodeContent::Circle { radius, color } => {
                draw_circle(pixels, width, height, &world_transform, *radius, *color, eff_opacity);
            }
            NodeContent::Line {
                x2,
                y2,
                color,
                width: lw,
            } => {
                let (x0, y0) = world_transform.transform_point(0.0, 0.0);
                let (ex, ey) = world_transform.transform_point(*x2, *y2);
                draw_line(pixels, width, height, x0, y0, ex, ey, *color, *lw, eff_opacity);
            }
        }

        // Collect children sorted by z_order (stable sort for equal z values).
        let mut children: Vec<NodeId> = node.children.clone();
        children.sort_by(|a, b| {
            let za = self.nodes.get(a).map_or(0, |n| n.z_order);
            let zb = self.nodes.get(b).map_or(0, |n| n.z_order);
            za.cmp(&zb)
        });

        for child_id in children {
            self.render_node(pixels, width, height, child_id, &world_transform, eff_opacity);
        }
    }
}

impl Default for SceneGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Low-level pixel helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Alpha-composite `src` over the existing pixel at `(px, py)` in `pixels`.
///
/// Uses the standard Porter–Duff "src-over" formula with pre-multiplied alpha.
#[inline]
fn composite_pixel(
    pixels: &mut [u8],
    buf_width: u32,
    buf_height: u32,
    px: i32,
    py: i32,
    color: [u8; 4],
    opacity: f32,
) {
    if px < 0 || py < 0 || px >= buf_width as i32 || py >= buf_height as i32 {
        return;
    }
    let idx = (py as usize * buf_width as usize + px as usize) * 4;
    if idx + 3 >= pixels.len() {
        return;
    }

    let src_a = (f32::from(color[3]) / 255.0 * opacity).clamp(0.0, 1.0);
    if src_a <= 0.0 {
        return;
    }
    let inv_a = 1.0 - src_a;

    let dst_r = f32::from(pixels[idx]);
    let dst_g = f32::from(pixels[idx + 1]);
    let dst_b = f32::from(pixels[idx + 2]);
    let dst_a = f32::from(pixels[idx + 3]) / 255.0;

    let out_a = src_a + dst_a * inv_a;
    if out_a <= 0.0 {
        return;
    }
    let src_r = f32::from(color[0]);
    let src_g = f32::from(color[1]);
    let src_b = f32::from(color[2]);

    pixels[idx]     = ((src_r * src_a + dst_r * dst_a * inv_a) / out_a).clamp(0.0, 255.0) as u8;
    pixels[idx + 1] = ((src_g * src_a + dst_g * dst_a * inv_a) / out_a).clamp(0.0, 255.0) as u8;
    pixels[idx + 2] = ((src_b * src_a + dst_b * dst_a * inv_a) / out_a).clamp(0.0, 255.0) as u8;
    pixels[idx + 3] = (out_a * 255.0).clamp(0.0, 255.0) as u8;
}

/// Draw a filled axis-aligned rectangle in local space `(lx, ly, lw, lh)`,
/// transformed to world space via `transform`.
fn draw_rect(
    pixels: &mut [u8],
    buf_width: u32,
    buf_height: u32,
    transform: &Transform2D,
    lx: f32,
    ly: f32,
    lw: f32,
    lh: f32,
    color: [u8; 4],
    opacity: f32,
) {
    // Transform all four corners to world space and find the AABB.
    let corners = [
        transform.transform_point(lx, ly),
        transform.transform_point(lx + lw, ly),
        transform.transform_point(lx, ly + lh),
        transform.transform_point(lx + lw, ly + lh),
    ];
    let (min_x, min_y, max_x, max_y) = aabb_of_corners(&corners);

    let x0 = min_x.floor() as i32;
    let y0 = min_y.floor() as i32;
    let x1 = max_x.ceil() as i32;
    let y1 = max_y.ceil() as i32;

    for py in y0..=y1 {
        for px in x0..=x1 {
            composite_pixel(pixels, buf_width, buf_height, px, py, color, opacity);
        }
    }
}

/// Draw a filled circle centred at the world-space transform of `(0,0)`.
fn draw_circle(
    pixels: &mut [u8],
    buf_width: u32,
    buf_height: u32,
    transform: &Transform2D,
    radius: f32,
    color: [u8; 4],
    opacity: f32,
) {
    let (cx, cy) = transform.transform_point(0.0, 0.0);

    // Estimate a scale factor from the transform matrix to handle scaled circles.
    let [a, b, c, d, _, _] = transform.matrix;
    let scale = ((a * a + b * b).sqrt() + (c * c + d * d).sqrt()) / 2.0;
    let world_radius = radius * scale;

    let x0 = (cx - world_radius).floor() as i32;
    let y0 = (cy - world_radius).floor() as i32;
    let x1 = (cx + world_radius).ceil() as i32;
    let y1 = (cy + world_radius).ceil() as i32;

    let r2 = world_radius * world_radius;

    for py in y0..=y1 {
        for px in x0..=x1 {
            let dx = px as f32 + 0.5 - cx;
            let dy = py as f32 + 0.5 - cy;
            if dx * dx + dy * dy <= r2 {
                composite_pixel(pixels, buf_width, buf_height, px, py, color, opacity);
            }
        }
    }
}

/// Bresenham-style line with a given stroke width.
fn draw_line(
    pixels: &mut [u8],
    buf_width: u32,
    buf_height: u32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    color: [u8; 4],
    stroke_width: f32,
    opacity: f32,
) {
    // For each sample point along the line we draw a small square brush of
    // the requested width.  This gives a uniform-width stroke without requiring
    // a full anti-aliased renderer.
    let dx = x1 - x0;
    let dy = y1 - y0;
    let length = (dx * dx + dy * dy).sqrt();
    let steps = if length < 1.0 { 1 } else { length.ceil() as u32 * 2 };
    let half_w = (stroke_width / 2.0).max(0.5);

    for i in 0..=steps {
        let t = if steps == 0 { 0.0 } else { i as f32 / steps as f32 };
        let sx = x0 + dx * t;
        let sy = y0 + dy * t;

        let bx0 = (sx - half_w).floor() as i32;
        let by0 = (sy - half_w).floor() as i32;
        let bx1 = (sx + half_w).ceil() as i32;
        let by1 = (sy + half_w).ceil() as i32;

        for py in by0..=by1 {
            for px in bx0..=bx1 {
                composite_pixel(pixels, buf_width, buf_height, px, py, color, opacity);
            }
        }
    }
}

/// Draw text as simple coloured blocks (one 5×7 pixel block per character).
fn draw_text(
    pixels: &mut [u8],
    buf_width: u32,
    buf_height: u32,
    transform: &Transform2D,
    text: &str,
    color: [u8; 4],
    font_size: f32,
    opacity: f32,
) {
    let char_w = (font_size * 0.6).max(4.0);
    let char_h = font_size.max(6.0);

    let mut cursor_x = 0.0_f32;
    for ch in text.chars() {
        if ch == ' ' {
            cursor_x += char_w;
            continue;
        }
        // Compute world-space AABB of this glyph block.
        draw_rect(
            pixels,
            buf_width,
            buf_height,
            transform,
            cursor_x,
            0.0,
            char_w * 0.85,
            char_h,
            color,
            opacity,
        );
        cursor_x += char_w;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AABB helper
// ─────────────────────────────────────────────────────────────────────────────

fn aabb_of_corners(corners: &[(f32, f32)]) -> (f32, f32, f32, f32) {
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    for &(x, y) in corners {
        if x < min_x { min_x = x; }
        if y < min_y { min_y = y; }
        if x > max_x { max_x = x; }
        if y > max_y { max_y = y; }
    }
    (min_x, min_y, max_x, max_y)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn blank_buffer(w: u32, h: u32) -> Vec<u8> {
        vec![0u8; (w * h * 4) as usize]
    }

    /// Count the number of pixels whose colour is (r,g,b) regardless of alpha.
    fn count_color(buf: &[u8], w: u32, h: u32, r: u8, g: u8, b: u8) -> usize {
        let total = (w * h) as usize;
        (0..total)
            .filter(|&i| buf[i * 4] == r && buf[i * 4 + 1] == g && buf[i * 4 + 2] == b)
            .count()
    }

    fn any_nonzero_alpha(buf: &[u8]) -> bool {
        buf.chunks_exact(4).any(|p| p[3] > 0)
    }

    // ── 1. new() ─────────────────────────────────────────────────────────────

    #[test]
    fn test_scene_graph_new_has_root() {
        let sg = SceneGraph::new();
        assert_eq!(sg.node_count(), 1, "fresh graph must have exactly 1 node");
        let root = sg.get(sg.root());
        assert!(root.is_some(), "root must exist");
    }

    // ── 2. add_node increases count ───────────────────────────────────────────

    #[test]
    fn test_add_node_increases_count() {
        let mut sg = SceneGraph::new();
        let root = sg.root();
        sg.add_node(root, NodeContent::Empty);
        assert_eq!(sg.node_count(), 2);
        sg.add_node(root, NodeContent::Empty);
        assert_eq!(sg.node_count(), 3);
    }

    // ── 3. add_node returns valid id ─────────────────────────────────────────

    #[test]
    fn test_add_node_returns_valid_id() {
        let mut sg = SceneGraph::new();
        let root = sg.root();
        let id = sg.add_node(root, NodeContent::Empty).expect("must return Some");
        assert!(sg.get(id).is_some(), "returned id must be retrievable");
    }

    // ── 4. remove_node decreases count ───────────────────────────────────────

    #[test]
    fn test_remove_node_decreases_count() {
        let mut sg = SceneGraph::new();
        let root = sg.root();
        let id = sg.add_node(root, NodeContent::Empty).expect("add");
        assert_eq!(sg.node_count(), 2);
        let removed = sg.remove_node(id);
        assert!(removed, "remove must succeed");
        assert_eq!(sg.node_count(), 1);
    }

    // ── 5. remove_node removes descendants ───────────────────────────────────

    #[test]
    fn test_remove_node_removes_descendants() {
        let mut sg = SceneGraph::new();
        let root = sg.root();
        let parent = sg.add_node(root, NodeContent::Empty).expect("parent");
        let child1 = sg.add_node(parent, NodeContent::Empty).expect("child1");
        let child2 = sg.add_node(parent, NodeContent::Empty).expect("child2");
        let grand  = sg.add_node(child1, NodeContent::Empty).expect("grand");

        // 1 root + 4 nodes
        assert_eq!(sg.node_count(), 5);
        sg.remove_node(parent);
        assert_eq!(sg.node_count(), 1, "only root should remain");
        assert!(sg.get(child1).is_none());
        assert!(sg.get(child2).is_none());
        assert!(sg.get(grand).is_none());
    }

    // ── 6. reparent moves node ────────────────────────────────────────────────

    #[test]
    fn test_reparent_moves_node() {
        let mut sg = SceneGraph::new();
        let root = sg.root();
        let a = sg.add_node(root, NodeContent::Empty).expect("a");
        let b = sg.add_node(root, NodeContent::Empty).expect("b");
        let child = sg.add_node(a, NodeContent::Empty).expect("child");

        let ok = sg.reparent(child, b);
        assert!(ok, "reparent should succeed");
        let b_node = sg.get(b).expect("b must exist");
        assert!(b_node.children.contains(&child), "child must now be under b");
    }

    // ── 7. reparent removes from old parent ───────────────────────────────────

    #[test]
    fn test_reparent_removes_from_old_parent() {
        let mut sg = SceneGraph::new();
        let root = sg.root();
        let a = sg.add_node(root, NodeContent::Empty).expect("a");
        let b = sg.add_node(root, NodeContent::Empty).expect("b");
        let child = sg.add_node(a, NodeContent::Empty).expect("child");

        sg.reparent(child, b);
        let a_node = sg.get(a).expect("a must exist");
        assert!(!a_node.children.contains(&child), "child must no longer be under a");
    }

    // ── 8. invisible node produces no colored pixels ──────────────────────────

    #[test]
    fn test_node_visibility_skips_render() {
        let mut sg = SceneGraph::new();
        let root = sg.root();
        let id = sg.add_node(root, NodeContent::ColorRect {
            width: 50.0,
            height: 50.0,
            color: [255, 0, 0, 255],
        }).expect("rect");
        sg.get_mut(id).expect("node").visible = false;

        let (w, h) = (64, 64);
        let mut buf = blank_buffer(w, h);
        sg.render(&mut buf, w, h);

        assert!(!any_nonzero_alpha(&buf), "invisible node must not paint anything");
    }

    // ── 9. opacity affects alpha ──────────────────────────────────────────────

    #[test]
    fn test_node_opacity_affects_alpha() {
        let mut sg_full  = SceneGraph::new();
        let mut sg_half  = SceneGraph::new();
        let root_f = sg_full.root();
        let root_h = sg_half.root();

        let content = NodeContent::ColorRect {
            width: 20.0, height: 20.0, color: [200, 100, 50, 255],
        };
        let id_f = sg_full.add_node(root_f, content.clone()).expect("f");
        let id_h = sg_half.add_node(root_h, content).expect("h");

        sg_full.get_mut(id_f).expect("nf").opacity = 1.0;
        sg_half.get_mut(id_h).expect("nh").opacity = 0.5;

        let (w, h) = (32, 32);
        let mut buf_full = blank_buffer(w, h);
        let mut buf_half = blank_buffer(w, h);
        sg_full.render(&mut buf_full, w, h);
        sg_half.render(&mut buf_half, w, h);

        // Find any painted pixel and compare alpha values.
        let alpha_full: Vec<u8> = buf_full.chunks_exact(4).filter(|p| p[3] > 0).map(|p| p[3]).collect();
        let alpha_half: Vec<u8> = buf_half.chunks_exact(4).filter(|p| p[3] > 0).map(|p| p[3]).collect();

        assert!(!alpha_full.is_empty(), "full opacity must paint pixels");
        assert!(!alpha_half.is_empty(), "half opacity must paint pixels");

        let avg_full: f32 = alpha_full.iter().map(|&a| a as f32).sum::<f32>() / alpha_full.len() as f32;
        let avg_half: f32 = alpha_half.iter().map(|&a| a as f32).sum::<f32>() / alpha_half.len() as f32;
        assert!(avg_full > avg_half, "full-opacity alpha ({avg_full}) must exceed half-opacity alpha ({avg_half})");
    }

    // ── 10. translate positions content ──────────────────────────────────────

    #[test]
    fn test_transform_translate() {
        let mut sg = SceneGraph::new();
        let root = sg.root();
        let id = sg.add_node(root, NodeContent::ColorRect {
            width: 4.0, height: 4.0, color: [0, 255, 0, 255],
        }).expect("rect");
        sg.get_mut(id).expect("n").transform = Transform2D::translate(20.0, 20.0);

        let (w, h) = (64, 64);
        let mut buf = blank_buffer(w, h);
        sg.render(&mut buf, w, h);

        // Pixel at (0,0) should be untouched.
        assert_eq!(buf[3], 0, "pixel at (0,0) should have alpha=0 (not painted)");

        // Pixel around (20,20) should be painted.
        let idx = (20 * w as usize + 20) * 4;
        assert!(buf[idx + 3] > 0, "pixel at (20,20) should be painted");
    }

    // ── 11. child inherits parent's translation ───────────────────────────────

    #[test]
    fn test_transform_compose() {
        let mut sg = SceneGraph::new();
        let root = sg.root();

        let parent = sg.add_node(root, NodeContent::Empty).expect("parent");
        sg.get_mut(parent).expect("p").transform = Transform2D::translate(10.0, 10.0);

        let child = sg.add_node(parent, NodeContent::ColorRect {
            width: 4.0, height: 4.0, color: [0, 0, 255, 255],
        }).expect("child");
        sg.get_mut(child).expect("c").transform = Transform2D::translate(5.0, 5.0);

        let (w, h) = (32, 32);
        let mut buf = blank_buffer(w, h);
        sg.render(&mut buf, w, h);

        // Combined offset = (15, 15).
        let idx = (15 * w as usize + 15) * 4;
        assert!(buf[idx + 3] > 0, "combined translate must place pixel at (15,15)");
    }

    // ── 12. z_order — higher renders on top ──────────────────────────────────

    #[test]
    fn test_z_order_matters() {
        // Two overlapping rects: red (z=1) under blue (z=2).
        // At the overlapping region the final colour should be blue.
        let mut sg = SceneGraph::new();
        let root = sg.root();

        let red  = sg.add_node(root, NodeContent::ColorRect {
            width: 16.0, height: 16.0, color: [255, 0, 0, 255],
        }).expect("red");
        let blue = sg.add_node(root, NodeContent::ColorRect {
            width: 16.0, height: 16.0, color: [0, 0, 255, 255],
        }).expect("blue");

        sg.get_mut(red).expect("r").z_order  = 1;
        sg.get_mut(blue).expect("b").z_order = 2;

        let (w, h) = (32, 32);
        let mut buf = blank_buffer(w, h);
        sg.render(&mut buf, w, h);

        // Centre pixel should be blue (the higher z_order).
        let idx = (8 * w as usize + 8) * 4;
        assert_eq!(buf[idx], 0,   "R channel should be 0 (blue wins)");
        assert_eq!(buf[idx + 2], 255, "B channel should be 255 (blue wins)");
    }

    // ── 13. ColorRect renders expected pixels ─────────────────────────────────

    #[test]
    fn test_render_color_rect() {
        let mut sg = SceneGraph::new();
        let root = sg.root();
        sg.add_node(root, NodeContent::ColorRect {
            width: 10.0, height: 10.0, color: [128, 64, 32, 255],
        }).expect("rect");

        let (w, h) = (16, 16);
        let mut buf = blank_buffer(w, h);
        sg.render(&mut buf, w, h);

        let painted = count_color(&buf, w, h, 128, 64, 32);
        assert!(painted > 0, "ColorRect must paint some pixels");
    }

    // ── 14. Circle renders pixels inside radius ───────────────────────────────

    #[test]
    fn test_render_circle() {
        let mut sg = SceneGraph::new();
        let root = sg.root();
        let id = sg.add_node(root, NodeContent::Circle {
            radius: 10.0,
            color: [0, 200, 100, 255],
        }).expect("circle");
        // Centre the circle.
        sg.get_mut(id).expect("n").transform = Transform2D::translate(16.0, 16.0);

        let (w, h) = (32, 32);
        let mut buf = blank_buffer(w, h);
        sg.render(&mut buf, w, h);

        // Pixel at centre must be painted.
        let centre_idx = (16 * w as usize + 16) * 4;
        assert!(buf[centre_idx + 3] > 0, "pixel at circle centre must be painted");

        // Pixel far outside the circle must be unpainted.
        let far_idx = (0 * w as usize + 0) * 4;
        assert_eq!(buf[far_idx + 3], 0, "pixel far outside circle must not be painted");
    }

    // ── 15. render to empty / zero-size buffer doesn't panic ─────────────────

    #[test]
    fn test_render_empty_buffer_no_panic() {
        let mut sg = SceneGraph::new();
        let root = sg.root();
        sg.add_node(root, NodeContent::ColorRect {
            width: 100.0, height: 100.0, color: [255, 255, 255, 255],
        });

        let mut empty: Vec<u8> = Vec::new();
        sg.render(&mut empty, 0, 0);   // zero dimensions

        let mut tiny = vec![0u8; 4];
        sg.render(&mut tiny, 1, 1);    // 1×1 buffer — must not panic
    }

    // ── 16. identity transform doesn't move content ───────────────────────────

    #[test]
    fn test_identity_transform() {
        let mut sg = SceneGraph::new();
        let root = sg.root();
        let id = sg.add_node(root, NodeContent::ColorRect {
            width: 5.0, height: 5.0, color: [10, 20, 30, 255],
        }).expect("rect");
        sg.get_mut(id).expect("n").transform = Transform2D::identity();

        let (w, h) = (16, 16);
        let mut buf = blank_buffer(w, h);
        sg.render(&mut buf, w, h);

        // Should be painted at (0,0) with identity transform.
        let idx = 0;
        assert!(buf[idx + 3] > 0, "identity transform must paint at origin");
    }

    // ── 17. default opacity is 1.0 ───────────────────────────────────────────

    #[test]
    fn test_node_default_opacity_one() {
        let mut sg = SceneGraph::new();
        let root = sg.root();
        let id = sg.add_node(root, NodeContent::Empty).expect("node");
        let node = sg.get(id).expect("must exist");
        assert!((node.opacity - 1.0).abs() < f32::EPSILON,
            "default opacity must be 1.0, got {}", node.opacity);
    }
}
