//! Uniform spatial grid for O(1) average-case particle neighbourhood queries.
//!
//! Partitions 2D space into a regular grid of cells.  Each cell holds the
//! indices of particles whose position falls within it.  Neighbourhood queries
//! (all particles within radius `r` of a point) only need to inspect the
//! `⌈r/cell_size⌉` surrounding cells rather than all particles.
//!
//! # Usage
//!
//! ```
//! use oximedia_vfx::particle::spatial_grid::SpatialGrid;
//!
//! let mut grid = SpatialGrid::new(10.0, 100.0, 100.0);
//! // Insert particle 0 at (25, 35)
//! grid.insert(0, 25.0, 35.0);
//! // Query neighbours within 15 pixels of (30, 30)
//! let neighbours = grid.query_radius(30.0, 30.0, 15.0);
//! assert!(neighbours.contains(&0));
//! ```

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// SpatialGrid
// ─────────────────────────────────────────────────────────────────────────────

/// Uniform-cell spatial hash grid for 2D particle queries.
///
/// Particle indices are stored in a `HashMap` keyed by `(cell_x, cell_y)`.
/// The grid does **not** own the particles; it only stores particle *indices*.
/// Callers are responsible for keeping the grid consistent with particle
/// positions by calling [`clear`](Self::clear) and [`insert`](Self::insert)
/// each simulation step.
#[derive(Debug, Clone)]
pub struct SpatialGrid {
    /// Width of a single grid cell in world-space units.
    pub cell_size: f32,
    /// World-space width (used only for overflow clamping, not required).
    pub world_w: f32,
    /// World-space height.
    pub world_h: f32,
    /// Cell storage: `(cell_x, cell_y) → [particle_index, …]`.
    cells: HashMap<(i32, i32), Vec<usize>>,
}

impl SpatialGrid {
    /// Create a new grid.
    ///
    /// # Arguments
    ///
    /// * `cell_size` - Size of each cell in world-space units.  Should be
    ///   roughly equal to the largest interaction radius you intend to query.
    /// * `world_w` / `world_h` - World dimensions (used for documentation;
    ///   the grid handles any coordinates without bounds checking).
    #[must_use]
    pub fn new(cell_size: f32, world_w: f32, world_h: f32) -> Self {
        let cell_size = cell_size.max(0.001);
        Self {
            cell_size,
            world_w,
            world_h,
            cells: HashMap::new(),
        }
    }

    /// Remove all particles from the grid.
    pub fn clear(&mut self) {
        self.cells.clear();
    }

    /// Insert particle `index` at world position `(x, y)`.
    pub fn insert(&mut self, index: usize, x: f32, y: f32) {
        let key = self.cell_key(x, y);
        self.cells.entry(key).or_default().push(index);
    }

    /// Return all particle indices within the `cell_size`-aligned cell
    /// containing `(x, y)`.  Cheaper than [`query_radius`](Self::query_radius)
    /// when exact-cell lookup is sufficient.
    #[must_use]
    pub fn query_cell(&self, x: f32, y: f32) -> &[usize] {
        let key = self.cell_key(x, y);
        self.cells.get(&key).map_or(&[], Vec::as_slice)
    }

    /// Return all particle indices within radius `r` of `(x, y)`.
    ///
    /// Inspects all cells that overlap the bounding square `[x±r, y±r]`.
    /// The returned slice may contain particles slightly outside the circular
    /// radius; callers should post-filter by distance if exact results are
    /// required.
    #[must_use]
    pub fn query_radius(&self, x: f32, y: f32, r: f32) -> Vec<usize> {
        if r <= 0.0 {
            return Vec::new();
        }
        let min_cx = self.world_to_cell(x - r);
        let max_cx = self.world_to_cell(x + r);
        let min_cy = self.world_to_cell(y - r);
        let max_cy = self.world_to_cell(y + r);

        let r2 = r * r;
        let mut result = Vec::new();

        for cy in min_cy..=max_cy {
            for cx in min_cx..=max_cx {
                if let Some(bucket) = self.cells.get(&(cx, cy)) {
                    // We return the indices without filtering — the caller
                    // can do exact distance checks.  We do a loose AABB check
                    // here to avoid returning far-away cells' particles:
                    // the cell centre distance from query centre.
                    let cell_cx = (cx as f32 + 0.5) * self.cell_size;
                    let cell_cy = (cy as f32 + 0.5) * self.cell_size;
                    let dx = cell_cx - x;
                    let dy = cell_cy - y;
                    // Cell half-diagonal = cell_size * sqrt(2) / 2
                    let half_diag = self.cell_size * 0.7072;
                    if dx * dx + dy * dy <= (r + half_diag) * (r + half_diag) {
                        result.extend_from_slice(bucket);
                    } else {
                        // Still include — the entire cell AABB may intersect
                        // the query circle even if the centre is far
                        result.extend_from_slice(bucket);
                    }
                }
            }
        }

        // Deduplicate (a particle can only be in one cell, but multiple cells
        // may map to the same bucket via hash collisions in theory — dedup
        // to be safe).
        result.sort_unstable();
        result.dedup();
        result
    }

    /// Return all particle indices within radius `r`, **filtered** to only
    /// those whose stored position is within the exact circular radius.
    ///
    /// Requires the caller to pass a slice of `(x, y)` positions indexed by
    /// particle index so that distances can be computed.
    #[must_use]
    pub fn query_radius_exact(
        &self,
        x: f32,
        y: f32,
        r: f32,
        positions: &[(f32, f32)],
    ) -> Vec<usize> {
        let r2 = r * r;
        self.query_radius(x, y, r)
            .into_iter()
            .filter(|&idx| {
                if let Some(&(px, py)) = positions.get(idx) {
                    let dx = px - x;
                    let dy = py - y;
                    dx * dx + dy * dy <= r2
                } else {
                    false
                }
            })
            .collect()
    }

    /// Total number of particle entries currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cells.values().map(Vec::len).sum()
    }

    /// Returns `true` when the grid contains no particles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    fn world_to_cell(&self, v: f32) -> i32 {
        (v / self.cell_size).floor() as i32
    }

    fn cell_key(&self, x: f32, y: f32) -> (i32, i32) {
        (self.world_to_cell(x), self.world_to_cell(y))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_query_cell() {
        let mut g = SpatialGrid::new(10.0, 100.0, 100.0);
        g.insert(0, 5.0, 5.0);
        g.insert(1, 15.0, 5.0); // different cell
        let bucket = g.query_cell(5.0, 5.0);
        assert!(bucket.contains(&0));
        assert!(!bucket.contains(&1));
    }

    #[test]
    fn test_query_radius_finds_nearby() {
        let mut g = SpatialGrid::new(10.0, 100.0, 100.0);
        g.insert(0, 50.0, 50.0);
        g.insert(1, 55.0, 50.0);
        g.insert(2, 90.0, 90.0); // far away
        let near = g.query_radius(50.0, 50.0, 10.0);
        assert!(near.contains(&0));
        assert!(near.contains(&1));
        assert!(!near.contains(&2));
    }

    #[test]
    fn test_query_radius_exact_filters_correctly() {
        let mut g = SpatialGrid::new(10.0, 100.0, 100.0);
        g.insert(0, 50.0, 50.0);
        g.insert(1, 59.0, 50.0); // within 10-unit radius
        g.insert(2, 62.0, 50.0); // just outside 10-unit radius

        let positions = vec![(50.0, 50.0), (59.0, 50.0), (62.0, 50.0)];
        let result = g.query_radius_exact(50.0, 50.0, 10.0, &positions);
        assert!(result.contains(&0));
        assert!(result.contains(&1));
        assert!(!result.contains(&2), "62 should be outside radius 10: {result:?}");
    }

    #[test]
    fn test_clear_removes_all_particles() {
        let mut g = SpatialGrid::new(10.0, 100.0, 100.0);
        g.insert(0, 5.0, 5.0);
        g.insert(1, 15.0, 15.0);
        assert_eq!(g.len(), 2);
        g.clear();
        assert_eq!(g.len(), 0);
        assert!(g.is_empty());
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut g = SpatialGrid::new(10.0, 100.0, 100.0);
        assert!(g.is_empty());
        g.insert(0, 5.0, 5.0);
        assert_eq!(g.len(), 1);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_query_radius_zero_returns_empty() {
        let mut g = SpatialGrid::new(10.0, 100.0, 100.0);
        g.insert(0, 50.0, 50.0);
        assert!(g.query_radius(50.0, 50.0, 0.0).is_empty());
    }

    #[test]
    fn test_multiple_particles_same_cell() {
        let mut g = SpatialGrid::new(10.0, 100.0, 100.0);
        for i in 0..5 {
            g.insert(i, 3.0, 3.0 + i as f32 * 0.5);
        }
        assert_eq!(g.len(), 5);
        let bucket = g.query_cell(3.0, 3.0);
        assert_eq!(bucket.len(), 5);
    }

    #[test]
    fn test_negative_coordinates() {
        let mut g = SpatialGrid::new(10.0, 200.0, 200.0);
        g.insert(0, -5.0, -3.0);
        let found = g.query_radius(-5.0, -3.0, 3.0);
        assert!(found.contains(&0));
    }
}
