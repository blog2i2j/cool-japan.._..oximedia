//! Grid-based 2-D heatmap accumulation and density querying.
//!
//! Divides the frame (or any 2-D canvas) into uniform cells and counts
//! how many event points fall inside each cell.  Density is returned as
//! a fraction of the maximum cell count so that the caller can map it to
//! a colour gradient without knowing the absolute event volume.

use crate::error::AnalyticsError;

// ─── Heatmap ─────────────────────────────────────────────────────────────────

/// A uniform-cell 2-D heatmap.
///
/// # Layout
///
/// Cells are stored in row-major order.  Cell `(cx, cy)` maps to index
/// `cy * cols + cx`, where `cols = ceil(width / cell_size)`.
#[derive(Debug, Clone)]
pub struct Heatmap {
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
    /// Edge length of each square cell in pixels.
    pub cell_size: u32,
    /// Number of columns (x-axis cells).
    cols: u32,
    /// Number of rows (y-axis cells).
    rows: u32,
    /// Hit counts for each cell.
    counts: Vec<u64>,
    /// Cumulative total of all added points.
    total: u64,
}

impl Heatmap {
    /// Creates a new, empty heatmap.
    ///
    /// # Errors
    ///
    /// Returns [`AnalyticsError::InvalidInput`] when any dimension is zero.
    pub fn new(width: u32, height: u32, cell_size: u32) -> Result<Self, AnalyticsError> {
        if width == 0 {
            return Err(AnalyticsError::InvalidInput("width must be > 0".into()));
        }
        if height == 0 {
            return Err(AnalyticsError::InvalidInput("height must be > 0".into()));
        }
        if cell_size == 0 {
            return Err(AnalyticsError::InvalidInput("cell_size must be > 0".into()));
        }

        let cols = width.div_ceil(cell_size);
        let rows = height.div_ceil(cell_size);
        let cells = (cols as usize)
            .checked_mul(rows as usize)
            .ok_or_else(|| AnalyticsError::InvalidInput("heatmap too large".into()))?;

        Ok(Self {
            width,
            height,
            cell_size,
            cols,
            rows,
            counts: vec![0u64; cells],
            total: 0,
        })
    }

    /// Records a point at canvas coordinates `(x, y)`.
    ///
    /// Points outside the canvas `[0, width) × [0, height)` are silently
    /// discarded.
    pub fn add_point(&mut self, x: f32, y: f32) {
        if x < 0.0 || y < 0.0 || x >= self.width as f32 || y >= self.height as f32 {
            return;
        }
        let cx = (x / self.cell_size as f32) as u32;
        let cy = (y / self.cell_size as f32) as u32;
        // Clamp to valid cell range (handles floating-point edge cases).
        let cx = cx.min(self.cols - 1);
        let cy = cy.min(self.rows - 1);
        let idx = (cy * self.cols + cx) as usize;
        self.counts[idx] = self.counts[idx].saturating_add(1);
        self.total = self.total.saturating_add(1);
    }

    /// Returns the normalised density `[0.0, 1.0]` for the cell identified by
    /// its zero-based column/row index `(cx, cy)`.
    ///
    /// Returns `0.0` for out-of-range cells or when the heatmap is empty.
    #[must_use]
    pub fn get_density(&self, cx: u32, cy: u32) -> f32 {
        if cx >= self.cols || cy >= self.rows {
            return 0.0;
        }
        let idx = (cy * self.cols + cx) as usize;
        let max_count = self.counts.iter().copied().max().unwrap_or(0);
        if max_count == 0 {
            return 0.0;
        }
        self.counts[idx] as f32 / max_count as f32
    }

    /// Returns the raw hit count for cell `(cx, cy)`.
    ///
    /// Returns `0` for out-of-range cells.
    #[must_use]
    pub fn get_count(&self, cx: u32, cy: u32) -> u64 {
        if cx >= self.cols || cy >= self.rows {
            return 0;
        }
        self.counts[(cy * self.cols + cx) as usize]
    }

    /// Total number of valid points added so far.
    #[must_use]
    pub fn total_points(&self) -> u64 {
        self.total
    }

    /// Returns the number of columns (x-axis cells).
    #[must_use]
    pub fn cols(&self) -> u32 {
        self.cols
    }

    /// Returns the number of rows (y-axis cells).
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// Resets all counts to zero.
    pub fn clear(&mut self) {
        self.counts.fill(0);
        self.total = 0;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_zero_width() {
        assert!(Heatmap::new(0, 100, 10).is_err());
    }

    #[test]
    fn new_rejects_zero_height() {
        assert!(Heatmap::new(100, 0, 10).is_err());
    }

    #[test]
    fn new_rejects_zero_cell_size() {
        assert!(Heatmap::new(100, 100, 0).is_err());
    }

    #[test]
    fn empty_heatmap_density_is_zero() {
        let h = Heatmap::new(100, 100, 10).expect("valid");
        assert_eq!(h.get_density(0, 0), 0.0);
    }

    #[test]
    fn single_point_max_density() {
        let mut h = Heatmap::new(100, 100, 10).expect("valid");
        h.add_point(5.0, 5.0); // cell (0, 0)
        assert!((h.get_density(0, 0) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn out_of_bounds_point_ignored() {
        let mut h = Heatmap::new(100, 100, 10).expect("valid");
        h.add_point(-1.0, 50.0);
        h.add_point(100.0, 50.0);
        h.add_point(50.0, -1.0);
        h.add_point(50.0, 100.0);
        assert_eq!(h.total_points(), 0);
    }

    #[test]
    fn density_relative_to_max_cell() {
        let mut h = Heatmap::new(100, 100, 50).expect("valid"); // 2×2 cells
                                                                // cell (0,0) gets 3 hits, cell (1,0) gets 1 hit
        for _ in 0..3 {
            h.add_point(10.0, 10.0);
        }
        h.add_point(60.0, 10.0);
        assert!((h.get_density(0, 0) - 1.0).abs() < f32::EPSILON);
        assert!((h.get_density(1, 0) - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn out_of_range_cell_returns_zero() {
        let h = Heatmap::new(100, 100, 10).expect("valid");
        assert_eq!(h.get_density(999, 999), 0.0);
        assert_eq!(h.get_count(999, 999), 0);
    }

    #[test]
    fn clear_resets_counts() {
        let mut h = Heatmap::new(100, 100, 10).expect("valid");
        h.add_point(5.0, 5.0);
        h.clear();
        assert_eq!(h.total_points(), 0);
        assert_eq!(h.get_density(0, 0), 0.0);
    }

    #[test]
    fn cols_rows_computed_correctly() {
        let h = Heatmap::new(105, 95, 10).expect("valid");
        assert_eq!(h.cols(), 11); // ceil(105/10)
        assert_eq!(h.rows(), 10); // ceil(95/10)
    }
}
