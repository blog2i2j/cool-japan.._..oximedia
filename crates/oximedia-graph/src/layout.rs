//! Graph layout algorithms.
//!
//! Provides force-directed, hierarchical, and grid layout algorithms
//! for positioning graph nodes in 2D space.

use std::collections::HashMap;

/// A 2D position.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position {
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
}

impl Position {
    /// Create a new position.
    #[allow(dead_code)]
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Euclidean distance to another position.
    #[allow(dead_code)]
    pub fn distance(&self, other: &Position) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Node identifier for layout purposes.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LayoutNodeId(pub usize);

/// An edge between two layout nodes.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct LayoutEdge {
    /// Source node.
    pub from: LayoutNodeId,
    /// Target node.
    pub to: LayoutNodeId,
}

/// Layout algorithm selection.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayoutAlgorithm {
    /// Force-directed (Fruchterman-Reingold approximation).
    ForceDirected {
        /// Number of iterations.
        iterations: usize,
        /// Ideal spring length.
        ideal_length: f32,
    },
    /// Hierarchical (Sugiyama-style layering).
    Hierarchical {
        /// Vertical spacing between layers.
        layer_spacing: f32,
        /// Horizontal spacing between nodes in same layer.
        node_spacing: f32,
    },
    /// Grid layout.
    Grid {
        /// Number of columns.
        columns: usize,
        /// Horizontal cell size.
        cell_width: f32,
        /// Vertical cell size.
        cell_height: f32,
    },
}

impl Default for LayoutAlgorithm {
    fn default() -> Self {
        LayoutAlgorithm::Grid {
            columns: 4,
            cell_width: 150.0,
            cell_height: 100.0,
        }
    }
}

/// Computes node positions using the selected layout algorithm.
#[allow(dead_code)]
pub struct GraphLayout {
    algorithm: LayoutAlgorithm,
}

impl GraphLayout {
    /// Create a new graph layout with the given algorithm.
    #[allow(dead_code)]
    pub fn new(algorithm: LayoutAlgorithm) -> Self {
        Self { algorithm }
    }

    /// Compute positions for nodes given the edge list.
    #[allow(dead_code)]
    pub fn compute(
        &self,
        node_ids: &[LayoutNodeId],
        edges: &[LayoutEdge],
    ) -> HashMap<LayoutNodeId, Position> {
        match self.algorithm {
            LayoutAlgorithm::Grid {
                columns,
                cell_width,
                cell_height,
            } => self.grid_layout(node_ids, columns, cell_width, cell_height),
            LayoutAlgorithm::Hierarchical {
                layer_spacing,
                node_spacing,
            } => self.hierarchical_layout(node_ids, edges, layer_spacing, node_spacing),
            LayoutAlgorithm::ForceDirected {
                iterations,
                ideal_length,
            } => self.force_directed_layout(node_ids, edges, iterations, ideal_length),
        }
    }

    /// Grid layout: place nodes in a grid.
    #[allow(dead_code)]
    fn grid_layout(
        &self,
        node_ids: &[LayoutNodeId],
        columns: usize,
        cell_width: f32,
        cell_height: f32,
    ) -> HashMap<LayoutNodeId, Position> {
        let cols = columns.max(1);
        let mut positions = HashMap::new();
        for (i, &node_id) in node_ids.iter().enumerate() {
            let col = i % cols;
            let row = i / cols;
            positions.insert(
                node_id,
                Position::new(col as f32 * cell_width, row as f32 * cell_height),
            );
        }
        positions
    }

    /// Simple hierarchical layout based on in-degree (topological levels).
    #[allow(dead_code)]
    fn hierarchical_layout(
        &self,
        node_ids: &[LayoutNodeId],
        edges: &[LayoutEdge],
        layer_spacing: f32,
        node_spacing: f32,
    ) -> HashMap<LayoutNodeId, Position> {
        // Compute in-degree for each node
        let mut in_degree: HashMap<LayoutNodeId, usize> = HashMap::new();
        for &id in node_ids {
            in_degree.insert(id, 0);
        }
        for edge in edges {
            *in_degree.entry(edge.to).or_insert(0) += 1;
        }

        // Group by in-degree as proxy for layer
        let mut layers: HashMap<usize, Vec<LayoutNodeId>> = HashMap::new();
        for (&node, &deg) in &in_degree {
            layers.entry(deg).or_default().push(node);
        }

        let mut positions = HashMap::new();
        for (layer_idx, (_, layer_nodes)) in layers.iter().enumerate() {
            for (node_idx, &node_id) in layer_nodes.iter().enumerate() {
                positions.insert(
                    node_id,
                    Position::new(
                        node_idx as f32 * node_spacing,
                        layer_idx as f32 * layer_spacing,
                    ),
                );
            }
        }
        positions
    }

    /// Simplified force-directed layout (Fruchterman-Reingold approximation).
    #[allow(dead_code)]
    fn force_directed_layout(
        &self,
        node_ids: &[LayoutNodeId],
        edges: &[LayoutEdge],
        iterations: usize,
        ideal_length: f32,
    ) -> HashMap<LayoutNodeId, Position> {
        if node_ids.is_empty() {
            return HashMap::new();
        }

        // Initialize positions in a circle
        let n = node_ids.len();
        let mut positions: HashMap<LayoutNodeId, Position> = HashMap::new();
        for (i, &id) in node_ids.iter().enumerate() {
            let angle = 2.0 * std::f32::consts::PI * i as f32 / n as f32;
            positions.insert(
                id,
                Position::new(angle.cos() * ideal_length, angle.sin() * ideal_length),
            );
        }

        let k = ideal_length;
        let k2 = k * k;

        for _ in 0..iterations {
            let mut displacements: HashMap<LayoutNodeId, (f32, f32)> = HashMap::new();
            for &id in node_ids {
                displacements.insert(id, (0.0, 0.0));
            }

            // Repulsive forces
            for i in 0..n {
                for j in (i + 1)..n {
                    let u = node_ids[i];
                    let v = node_ids[j];
                    // SAFETY: positions is populated for every node in node_ids above
                    let pu = match positions.get(&u) {
                        Some(p) => *p,
                        None => continue,
                    };
                    let pv = match positions.get(&v) {
                        Some(p) => *p,
                        None => continue,
                    };
                    let dx = pu.x - pv.x;
                    let dy = pu.y - pv.y;
                    let dist2 = (dx * dx + dy * dy).max(0.001);
                    let dist = dist2.sqrt();
                    let force = k2 / dist;
                    let fx = (dx / dist) * force;
                    let fy = (dy / dist) * force;
                    if let Some(d) = displacements.get_mut(&u) {
                        d.0 += fx;
                        d.1 += fy;
                    }
                    if let Some(d) = displacements.get_mut(&v) {
                        d.0 -= fx;
                        d.1 -= fy;
                    }
                }
            }

            // Attractive forces along edges
            for edge in edges {
                let pu = match positions.get(&edge.from) {
                    Some(p) => *p,
                    None => continue,
                };
                let pv = match positions.get(&edge.to) {
                    Some(p) => *p,
                    None => continue,
                };
                let dx = pu.x - pv.x;
                let dy = pu.y - pv.y;
                let dist = (dx * dx + dy * dy).sqrt().max(0.001);
                let force = dist * dist / k;
                let fx = (dx / dist) * force;
                let fy = (dy / dist) * force;
                if let Some(d) = displacements.get_mut(&edge.from) {
                    d.0 -= fx;
                    d.1 -= fy;
                }
                if let Some(d) = displacements.get_mut(&edge.to) {
                    d.0 += fx;
                    d.1 += fy;
                }
            }

            // Apply displacements with cooling (temperature = k / 10)
            let temp = k / 10.0;
            for &id in node_ids {
                // SAFETY: displacements is populated for every node in node_ids above
                let (dx, dy) = match displacements.get(&id) {
                    Some(&d) => d,
                    None => continue,
                };
                let disp_len = (dx * dx + dy * dy).sqrt().max(0.001);
                let clamped = disp_len.min(temp);
                if let Some(pos) = positions.get_mut(&id) {
                    pos.x += (dx / disp_len) * clamped;
                    pos.y += (dy / disp_len) * clamped;
                }
            }
        }

        positions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node_ids(n: usize) -> Vec<LayoutNodeId> {
        (0..n).map(LayoutNodeId).collect()
    }

    #[test]
    fn test_position_creation() {
        let p = Position::new(1.0, 2.0);
        assert_eq!(p.x, 1.0);
        assert_eq!(p.y, 2.0);
    }

    #[test]
    fn test_position_distance_zero() {
        let p = Position::new(3.0, 4.0);
        assert_eq!(p.distance(&p), 0.0);
    }

    #[test]
    fn test_position_distance_nonzero() {
        let a = Position::new(0.0, 0.0);
        let b = Position::new(3.0, 4.0);
        assert!((a.distance(&b) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_grid_layout_empty() {
        let layout = GraphLayout::new(LayoutAlgorithm::Grid {
            columns: 4,
            cell_width: 100.0,
            cell_height: 100.0,
        });
        let positions = layout.compute(&[], &[]);
        assert!(positions.is_empty());
    }

    #[test]
    fn test_grid_layout_count() {
        let layout = GraphLayout::new(LayoutAlgorithm::Grid {
            columns: 3,
            cell_width: 100.0,
            cell_height: 100.0,
        });
        let ids = node_ids(6);
        let positions = layout.compute(&ids, &[]);
        assert_eq!(positions.len(), 6);
    }

    #[test]
    fn test_grid_layout_first_node_origin() {
        let layout = GraphLayout::new(LayoutAlgorithm::Grid {
            columns: 4,
            cell_width: 100.0,
            cell_height: 100.0,
        });
        let ids = node_ids(4);
        let positions = layout.compute(&ids, &[]);
        let p = positions[&LayoutNodeId(0)];
        assert_eq!(p.x, 0.0);
        assert_eq!(p.y, 0.0);
    }

    #[test]
    fn test_grid_layout_second_row() {
        let layout = GraphLayout::new(LayoutAlgorithm::Grid {
            columns: 2,
            cell_width: 100.0,
            cell_height: 80.0,
        });
        let ids = node_ids(4);
        let positions = layout.compute(&ids, &[]);
        let p = positions[&LayoutNodeId(2)]; // row 1, col 0
        assert_eq!(p.x, 0.0);
        assert_eq!(p.y, 80.0);
    }

    #[test]
    fn test_hierarchical_layout_count() {
        let layout = GraphLayout::new(LayoutAlgorithm::Hierarchical {
            layer_spacing: 80.0,
            node_spacing: 120.0,
        });
        let ids = node_ids(4);
        let edges = vec![LayoutEdge {
            from: LayoutNodeId(0),
            to: LayoutNodeId(1),
        }];
        let positions = layout.compute(&ids, &edges);
        assert_eq!(positions.len(), 4);
    }

    #[test]
    fn test_hierarchical_positions_finite() {
        let layout = GraphLayout::new(LayoutAlgorithm::Hierarchical {
            layer_spacing: 80.0,
            node_spacing: 120.0,
        });
        let ids = node_ids(5);
        let edges = vec![
            LayoutEdge {
                from: LayoutNodeId(0),
                to: LayoutNodeId(2),
            },
            LayoutEdge {
                from: LayoutNodeId(1),
                to: LayoutNodeId(3),
            },
        ];
        let positions = layout.compute(&ids, &edges);
        for (_, pos) in &positions {
            assert!(pos.x.is_finite());
            assert!(pos.y.is_finite());
        }
    }

    #[test]
    fn test_force_directed_empty() {
        let layout = GraphLayout::new(LayoutAlgorithm::ForceDirected {
            iterations: 10,
            ideal_length: 100.0,
        });
        let positions = layout.compute(&[], &[]);
        assert!(positions.is_empty());
    }

    #[test]
    fn test_force_directed_count() {
        let layout = GraphLayout::new(LayoutAlgorithm::ForceDirected {
            iterations: 5,
            ideal_length: 100.0,
        });
        let ids = node_ids(4);
        let edges = vec![LayoutEdge {
            from: LayoutNodeId(0),
            to: LayoutNodeId(1),
        }];
        let positions = layout.compute(&ids, &edges);
        assert_eq!(positions.len(), 4);
    }

    #[test]
    fn test_force_directed_positions_finite() {
        let layout = GraphLayout::new(LayoutAlgorithm::ForceDirected {
            iterations: 20,
            ideal_length: 80.0,
        });
        let ids = node_ids(6);
        let edges = vec![
            LayoutEdge {
                from: LayoutNodeId(0),
                to: LayoutNodeId(1),
            },
            LayoutEdge {
                from: LayoutNodeId(1),
                to: LayoutNodeId(2),
            },
            LayoutEdge {
                from: LayoutNodeId(2),
                to: LayoutNodeId(3),
            },
        ];
        let positions = layout.compute(&ids, &edges);
        for (_, pos) in &positions {
            assert!(pos.x.is_finite());
            assert!(pos.y.is_finite());
        }
    }

    #[test]
    fn test_default_algorithm_is_grid() {
        let algo = LayoutAlgorithm::default();
        assert!(matches!(algo, LayoutAlgorithm::Grid { .. }));
    }
}
