//! Grid-based composition for multi-camera production.

use super::{Compositor, Layout};
use crate::{AngleId, Result};

/// Grid compositor
#[derive(Debug)]
pub struct GridCompositor {
    /// Output dimensions
    dimensions: (u32, u32),
    /// Current layout
    layout: Layout,
    /// Grid cell spacing (pixels)
    spacing: u32,
    /// Cell aspect ratio (width/height)
    cell_aspect: f32,
}

impl GridCompositor {
    /// Create a new grid compositor
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            dimensions: (width, height),
            layout: Layout::Single { angle: 0 },
            spacing: 2,
            cell_aspect: 16.0 / 9.0,
        }
    }

    /// Set cell spacing
    pub fn set_spacing(&mut self, spacing: u32) {
        self.spacing = spacing;
    }

    /// Get cell spacing
    #[must_use]
    pub fn spacing(&self) -> u32 {
        self.spacing
    }

    /// Set cell aspect ratio
    pub fn set_cell_aspect(&mut self, aspect: f32) {
        self.cell_aspect = aspect;
    }

    /// Calculate grid cell dimensions
    #[must_use]
    pub fn calculate_cell_size(&self, rows: usize, cols: usize) -> (u32, u32) {
        let (width, height) = self.dimensions;
        let total_spacing_x = (cols as u32 + 1) * self.spacing;
        let total_spacing_y = (rows as u32 + 1) * self.spacing;

        let available_width = width.saturating_sub(total_spacing_x);
        let available_height = height.saturating_sub(total_spacing_y);

        let cell_width = available_width / cols as u32;
        let cell_height = available_height / rows as u32;

        (cell_width, cell_height)
    }

    /// Calculate grid layout positions
    #[must_use]
    pub fn calculate_grid(&self, rows: usize, cols: usize) -> Vec<(u32, u32, u32, u32)> {
        let (cell_width, cell_height) = self.calculate_cell_size(rows, cols);
        let mut cells = Vec::new();

        for row in 0..rows {
            for col in 0..cols {
                let x = self.spacing + col as u32 * (cell_width + self.spacing);
                let y = self.spacing + row as u32 * (cell_height + self.spacing);
                cells.push((x, y, cell_width, cell_height));
            }
        }

        cells
    }

    /// Create 2x2 grid layout
    #[must_use]
    pub fn grid_2x2(&self) -> Vec<(u32, u32, u32, u32)> {
        self.calculate_grid(2, 2)
    }

    /// Create 3x3 grid layout
    #[must_use]
    pub fn grid_3x3(&self) -> Vec<(u32, u32, u32, u32)> {
        self.calculate_grid(3, 3)
    }

    /// Create 4x4 grid layout
    #[must_use]
    pub fn grid_4x4(&self) -> Vec<(u32, u32, u32, u32)> {
        self.calculate_grid(4, 4)
    }

    /// Create custom grid layout
    #[must_use]
    pub fn grid_custom(&self, rows: usize, cols: usize) -> Vec<(u32, u32, u32, u32)> {
        self.calculate_grid(rows, cols)
    }

    /// Get optimal grid dimensions for angle count
    #[must_use]
    pub fn optimal_grid_for_angles(angle_count: usize) -> (usize, usize) {
        match angle_count {
            0..=1 => (1, 1),
            2 => (1, 2),
            3..=4 => (2, 2),
            5..=6 => (2, 3),
            7..=9 => (3, 3),
            10..=12 => (3, 4),
            13..=16 => (4, 4),
            17..=20 => (4, 5),
            21..=25 => (5, 5),
            _ => {
                let side = (angle_count as f64).sqrt().ceil() as usize;
                (side, side)
            }
        }
    }

    /// Create layout from angles
    pub fn create_layout_from_angles(&mut self, angles: &[AngleId]) -> Result<()> {
        let (rows, cols) = Self::optimal_grid_for_angles(angles.len());
        self.layout = Layout::Grid {
            dimensions: (rows, cols),
            angles: angles.to_vec(),
        };
        Ok(())
    }

    /// Get angle at grid position
    #[must_use]
    pub fn angle_at_position(&self, row: usize, col: usize) -> Option<AngleId> {
        if let Layout::Grid {
            dimensions,
            ref angles,
        } = self.layout
        {
            let (rows, cols) = dimensions;
            if row < rows && col < cols {
                let index = row * cols + col;
                return angles.get(index).copied();
            }
        }
        None
    }

    /// Set angle at grid position
    pub fn set_angle_at_position(&mut self, row: usize, col: usize, angle: AngleId) -> Result<()> {
        if let Layout::Grid {
            dimensions,
            ref mut angles,
        } = self.layout
        {
            let (rows, cols) = dimensions;
            if row < rows && col < cols {
                let index = row * cols + col;
                if index < angles.len() {
                    angles[index] = angle;
                    return Ok(());
                }
            }
        }
        Err(crate::MultiCamError::LayoutError(
            "Invalid grid position".to_string(),
        ))
    }
}

impl Compositor for GridCompositor {
    fn set_dimensions(&mut self, width: u32, height: u32) {
        self.dimensions = (width, height);
    }

    fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    fn set_layout(&mut self, layout: Layout) -> Result<()> {
        match layout {
            Layout::Grid { .. } | Layout::QuadSplit { .. } | Layout::Single { .. } => {
                self.layout = layout;
                Ok(())
            }
            _ => Err(crate::MultiCamError::LayoutError(
                "Grid compositor only supports Grid, QuadSplit, and Single layouts".to_string(),
            )),
        }
    }

    fn layout(&self) -> &Layout {
        &self.layout
    }
}

/// Spotlight grid compositor (highlight one cell)
#[derive(Debug)]
pub struct SpotlightGrid {
    /// Base grid compositor
    grid: GridCompositor,
    /// Spotlight angle
    spotlight_angle: Option<AngleId>,
    /// Spotlight scale factor
    spotlight_scale: f32,
}

impl SpotlightGrid {
    /// Create a new spotlight grid
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            grid: GridCompositor::new(width, height),
            spotlight_angle: None,
            spotlight_scale: 1.5,
        }
    }

    /// Set spotlight angle
    pub fn set_spotlight(&mut self, angle: Option<AngleId>) {
        self.spotlight_angle = angle;
    }

    /// Get spotlight angle
    #[must_use]
    pub fn spotlight(&self) -> Option<AngleId> {
        self.spotlight_angle
    }

    /// Set spotlight scale
    pub fn set_spotlight_scale(&mut self, scale: f32) {
        self.spotlight_scale = scale.max(1.0);
    }

    /// Calculate spotlight layout
    #[must_use]
    pub fn calculate_spotlight_layout(
        &self,
        rows: usize,
        cols: usize,
    ) -> Vec<(u32, u32, u32, u32)> {
        let mut cells = self.grid.calculate_grid(rows, cols);

        if let Some(spotlight) = self.spotlight_angle {
            if let Layout::Grid { ref angles, .. } = self.grid.layout {
                if let Some(index) = angles.iter().position(|&a| a == spotlight) {
                    if index < cells.len() {
                        let cell = cells[index];
                        let new_width = (cell.2 as f32 * self.spotlight_scale) as u32;
                        let new_height = (cell.3 as f32 * self.spotlight_scale) as u32;

                        cells[index] = (
                            cell.0.saturating_sub((new_width - cell.2) / 2),
                            cell.1.saturating_sub((new_height - cell.3) / 2),
                            new_width,
                            new_height,
                        );
                    }
                }
            }
        }

        cells
    }

    /// Get base grid
    #[must_use]
    pub fn grid(&self) -> &GridCompositor {
        &self.grid
    }

    /// Get mutable base grid
    pub fn grid_mut(&mut self) -> &mut GridCompositor {
        &mut self.grid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_creation() {
        let grid = GridCompositor::new(1920, 1080);
        assert_eq!(grid.dimensions(), (1920, 1080));
        assert_eq!(grid.spacing(), 2);
    }

    #[test]
    fn test_grid_2x2() {
        let grid = GridCompositor::new(1920, 1080);
        let cells = grid.grid_2x2();
        assert_eq!(cells.len(), 4);

        // Check that cells cover the grid
        let total_width = cells.iter().map(|(_, _, w, _)| *w).sum::<u32>();
        let total_height = cells.iter().map(|(_, _, _, h)| *h).sum::<u32>();
        assert!(total_width > 0);
        assert!(total_height > 0);
    }

    #[test]
    fn test_grid_3x3() {
        let grid = GridCompositor::new(1920, 1080);
        let cells = grid.grid_3x3();
        assert_eq!(cells.len(), 9);
    }

    #[test]
    fn test_optimal_grid() {
        assert_eq!(GridCompositor::optimal_grid_for_angles(1), (1, 1));
        assert_eq!(GridCompositor::optimal_grid_for_angles(4), (2, 2));
        assert_eq!(GridCompositor::optimal_grid_for_angles(9), (3, 3));
        assert_eq!(GridCompositor::optimal_grid_for_angles(16), (4, 4));
    }

    #[test]
    fn test_angle_at_position() {
        let mut grid = GridCompositor::new(1920, 1080);
        let angles = vec![0, 1, 2, 3];
        grid.create_layout_from_angles(&angles)
            .expect("multicam test operation should succeed");

        assert_eq!(grid.angle_at_position(0, 0), Some(0));
        assert_eq!(grid.angle_at_position(0, 1), Some(1));
        assert_eq!(grid.angle_at_position(1, 0), Some(2));
        assert_eq!(grid.angle_at_position(1, 1), Some(3));
    }

    #[test]
    fn test_set_angle_at_position() {
        let mut grid = GridCompositor::new(1920, 1080);
        let angles = vec![0, 1, 2, 3];
        grid.create_layout_from_angles(&angles)
            .expect("multicam test operation should succeed");

        assert!(grid.set_angle_at_position(0, 0, 5).is_ok());
        assert_eq!(grid.angle_at_position(0, 0), Some(5));
    }

    #[test]
    fn test_spotlight_grid() {
        let mut spotlight = SpotlightGrid::new(1920, 1080);
        spotlight.set_spotlight(Some(1));
        assert_eq!(spotlight.spotlight(), Some(1));

        spotlight.set_spotlight_scale(2.0);
        assert_eq!(spotlight.spotlight_scale, 2.0);
    }

    #[test]
    fn test_spacing() {
        let mut grid = GridCompositor::new(1920, 1080);
        grid.set_spacing(10);
        assert_eq!(grid.spacing(), 10);

        let cells = grid.grid_2x2();
        // With spacing, cells should be smaller
        assert!(cells[0].2 < 1920 / 2);
    }
}
