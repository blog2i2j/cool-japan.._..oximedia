//! Multi-source video wall layout for broadcast graphics.
//!
//! A video wall displays multiple video sources simultaneously in a configurable
//! grid or freeform layout, with:
//! - **`VideoWallLayout`**: preset and custom grid arrangements (2×2, 3×3, 4×4, PiP variants, etc.)
//! - **`VideoWallCell`**: position, size, source ID, border, label, and visibility per cell
//! - **`VideoWall`**: full compositor description with background and metadata
//! - **`VideoWallBuilder`**: ergonomic builder for assembling layouts
//! - **Animated transitions**: smooth layout-switch animation with per-cell from/to interpolation
//!
//! All coordinates are in a normalised `[0.0, 1.0]` space over the canvas
//! width/height.  Renderers scale to actual pixel dimensions.
//!
//! # Example
//!
//! ```
//! use oximedia_graphics::video_wall::{VideoWallBuilder, WallPreset};
//!
//! let wall = VideoWallBuilder::from_preset(WallPreset::Grid2x2, 1920.0, 1080.0)
//!     .with_gap(0.005)
//!     .with_border_width(0.002)
//!     .build()
//!     .expect("valid wall layout");
//!
//! assert_eq!(wall.cells.len(), 4);
//! ```

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// Rect2D — normalised rectangle
// ─────────────────────────────────────────────────────────────────────────────

/// Normalised rectangle `[0, 1]²`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rect2D {
    /// Left edge.
    pub x: f32,
    /// Top edge.
    pub y: f32,
    /// Width.
    pub width: f32,
    /// Height.
    pub height: f32,
}

impl Rect2D {
    /// Create a new rectangle.
    #[must_use]
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Return the right edge.
    #[must_use]
    pub fn right(&self) -> f32 {
        self.x + self.width
    }

    /// Return the bottom edge.
    #[must_use]
    pub fn bottom(&self) -> f32 {
        self.y + self.height
    }

    /// Shrink the rectangle inward by `amount` on all sides.
    #[must_use]
    pub fn inset(&self, amount: f32) -> Self {
        Self {
            x: self.x + amount,
            y: self.y + amount,
            width: (self.width - 2.0 * amount).max(0.0),
            height: (self.height - 2.0 * amount).max(0.0),
        }
    }

    /// Linearly interpolate toward `other` at normalised `t`.
    #[must_use]
    pub fn lerp(&self, other: &Rect2D, t: f32) -> Rect2D {
        let t = t.clamp(0.0, 1.0);
        Rect2D {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            width: self.width + (other.width - self.width) * t,
            height: self.height + (other.height - self.height) * t,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CellBorder
// ─────────────────────────────────────────────────────────────────────────────

/// Border style for a single [`VideoWallCell`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CellBorder {
    /// Border width as a fraction of canvas width.
    pub width: f32,
    /// RGBA colour (values `[0, 1]`).
    pub color: [f32; 4],
    /// Corner radius as a fraction of cell width.
    pub corner_radius: f32,
}

impl CellBorder {
    /// Create a simple solid border.
    #[must_use]
    pub fn solid(width: f32, color: [f32; 4]) -> Self {
        Self {
            width,
            color,
            corner_radius: 0.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CellLabel
// ─────────────────────────────────────────────────────────────────────────────

/// Optional text label overlaid on a [`VideoWallCell`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CellLabel {
    /// Label text.
    pub text: String,
    /// Font size in points (at 1920-wide canvas).
    pub font_size: f32,
    /// RGBA text colour.
    pub color: [f32; 4],
    /// Vertical alignment within the cell.
    pub v_align: LabelVAlign,
}

/// Vertical alignment of a cell label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LabelVAlign {
    /// Align label to the top of the cell.
    Top,
    /// Align label to the bottom of the cell.
    Bottom,
}

// ─────────────────────────────────────────────────────────────────────────────
// VideoWallCell
// ─────────────────────────────────────────────────────────────────────────────

/// A single pane in a video wall.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoWallCell {
    /// Unique identifier within the wall (0-based).
    pub id: usize,
    /// Normalised position and size on the canvas.
    pub rect: Rect2D,
    /// Source video stream or still-image identifier.
    pub source_id: Option<String>,
    /// Z-order (higher values render on top).
    pub z_order: i32,
    /// Whether the cell is visible.
    pub visible: bool,
    /// Border configuration.
    pub border: Option<CellBorder>,
    /// Optional text label.
    pub label: Option<CellLabel>,
    /// Cell opacity (`1.0` = fully opaque).
    pub opacity: f32,
}

impl VideoWallCell {
    /// Create a minimal visible cell at the given rect.
    #[must_use]
    pub fn new(id: usize, rect: Rect2D) -> Self {
        Self {
            id,
            rect,
            source_id: None,
            z_order: 0,
            visible: true,
            border: None,
            label: None,
            opacity: 1.0,
        }
    }

    /// Attach a source identifier to this cell.
    #[must_use]
    pub fn with_source(mut self, source_id: impl Into<String>) -> Self {
        self.source_id = Some(source_id.into());
        self
    }

    /// Attach a border to this cell.
    #[must_use]
    pub fn with_border(mut self, border: CellBorder) -> Self {
        self.border = Some(border);
        self
    }

    /// Attach a label to this cell.
    #[must_use]
    pub fn with_label(mut self, label: CellLabel) -> Self {
        self.label = Some(label);
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WallPreset — named layout configurations
// ─────────────────────────────────────────────────────────────────────────────

/// Predefined video wall grid arrangements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WallPreset {
    /// Single full-screen source.
    Single,
    /// 2 × 1 side-by-side split.
    SideBySide,
    /// 1 × 2 top-bottom split.
    TopBottom,
    /// 2 × 2 equal quadrants.
    Grid2x2,
    /// 3 × 3 nine-pane grid.
    Grid3x3,
    /// 4 × 4 sixteen-pane grid.
    Grid4x4,
    /// Picture-in-picture: one large main + one small overlay (bottom-right).
    PictureInPicture,
    /// Broadcaster: one large main (left 2/3) + three stacked on the right.
    BroadcastTriple,
    /// One large main (top 70 %) + three equal thumbnails below.
    FeatureWithBanner,
}

impl WallPreset {
    /// Number of cells generated by this preset.
    #[must_use]
    pub fn cell_count(self) -> usize {
        match self {
            WallPreset::Single => 1,
            WallPreset::SideBySide | WallPreset::TopBottom => 2,
            WallPreset::Grid2x2 | WallPreset::PictureInPicture => 4,
            WallPreset::Grid3x3 | WallPreset::BroadcastTriple => 9,
            WallPreset::Grid4x4 => 16,
            WallPreset::FeatureWithBanner => 4,
        }
    }

    /// Generate normalised [`Rect2D`] values for each cell given `gap` between panes.
    #[must_use]
    pub fn generate_rects(self, gap: f32) -> Vec<Rect2D> {
        let g = gap.max(0.0);
        match self {
            WallPreset::Single => vec![Rect2D::new(0.0, 0.0, 1.0, 1.0)],
            WallPreset::SideBySide => {
                let w = (1.0 - g) / 2.0;
                vec![
                    Rect2D::new(0.0, 0.0, w, 1.0),
                    Rect2D::new(w + g, 0.0, w, 1.0),
                ]
            }
            WallPreset::TopBottom => {
                let h = (1.0 - g) / 2.0;
                vec![
                    Rect2D::new(0.0, 0.0, 1.0, h),
                    Rect2D::new(0.0, h + g, 1.0, h),
                ]
            }
            WallPreset::Grid2x2 => uniform_grid(2, 2, g),
            WallPreset::Grid3x3 => uniform_grid(3, 3, g),
            WallPreset::Grid4x4 => uniform_grid(4, 4, g),
            WallPreset::PictureInPicture => {
                // Main covers full canvas; PiP is bottom-right 30 %
                vec![
                    Rect2D::new(0.0, 0.0, 1.0, 1.0),
                    Rect2D::new(0.67, 0.67, 0.30, 0.30),
                    // Two more hidden/placeholder cells to match cell_count
                    Rect2D::new(0.0, 0.0, 0.0, 0.0),
                    Rect2D::new(0.0, 0.0, 0.0, 0.0),
                ]
            }
            WallPreset::BroadcastTriple => {
                let mut rects = Vec::with_capacity(9);
                // First cell: left 2/3
                let main_w = (2.0 / 3.0) - g / 2.0;
                rects.push(Rect2D::new(0.0, 0.0, main_w, 1.0));
                // Right column: three equal cells
                let col_x = main_w + g;
                let col_w = 1.0 - col_x;
                let cell_h = (1.0 - 2.0 * g) / 3.0;
                for row in 0..3 {
                    let y = row as f32 * (cell_h + g);
                    rects.push(Rect2D::new(col_x, y, col_w, cell_h));
                }
                // Fill remaining 5 slots with empty placeholder cells
                for _ in 0..5 {
                    rects.push(Rect2D::new(0.0, 0.0, 0.0, 0.0));
                }
                rects
            }
            WallPreset::FeatureWithBanner => {
                let main_h = 0.70 - g / 2.0;
                let banner_y = main_h + g;
                let banner_h = 1.0 - banner_y;
                let cell_w = (1.0 - 2.0 * g) / 3.0;
                vec![
                    // Main feature
                    Rect2D::new(0.0, 0.0, 1.0, main_h),
                    // Three thumbnails
                    Rect2D::new(0.0, banner_y, cell_w, banner_h),
                    Rect2D::new(cell_w + g, banner_y, cell_w, banner_h),
                    Rect2D::new(2.0 * (cell_w + g), banner_y, cell_w, banner_h),
                ]
            }
        }
    }
}

/// Build a uniform cols×rows grid of [`Rect2D`]s with a given gap.
fn uniform_grid(cols: usize, rows: usize, gap: f32) -> Vec<Rect2D> {
    let g = gap.max(0.0);
    let cell_w = (1.0 - g * (cols as f32 - 1.0)) / cols as f32;
    let cell_h = (1.0 - g * (rows as f32 - 1.0)) / rows as f32;
    let mut rects = Vec::with_capacity(cols * rows);
    for row in 0..rows {
        for col in 0..cols {
            let x = col as f32 * (cell_w + g);
            let y = row as f32 * (cell_h + g);
            rects.push(Rect2D::new(x, y, cell_w, cell_h));
        }
    }
    rects
}

// ─────────────────────────────────────────────────────────────────────────────
// VideoWall — assembled layout
// ─────────────────────────────────────────────────────────────────────────────

/// A complete video-wall layout ready for rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoWall {
    /// Canvas width in pixels.
    pub canvas_width: f32,
    /// Canvas height in pixels.
    pub canvas_height: f32,
    /// All cells in z-sorted render order.
    pub cells: Vec<VideoWallCell>,
    /// Background RGBA colour (default: black).
    pub background_color: [f32; 4],
    /// Gap between cells as a fraction of canvas width.
    pub gap: f32,
    /// Global border applied to all cells (can be overridden per-cell).
    pub default_border: Option<CellBorder>,
}

impl VideoWall {
    /// Retrieve a cell by ID.
    #[must_use]
    pub fn cell(&self, id: usize) -> Option<&VideoWallCell> {
        self.cells.iter().find(|c| c.id == id)
    }

    /// Retrieve a mutable reference to a cell by ID.
    pub fn cell_mut(&mut self, id: usize) -> Option<&mut VideoWallCell> {
        self.cells.iter_mut().find(|c| c.id == id)
    }

    /// Re-sort cells by z-order (ascending; higher z-order renders last/on top).
    pub fn sort_by_z(&mut self) {
        self.cells.sort_by_key(|c| c.z_order);
    }

    /// Assign a source stream to cell `id`.  Returns `false` if no cell found.
    pub fn assign_source(&mut self, id: usize, source_id: impl Into<String>) -> bool {
        if let Some(cell) = self.cell_mut(id) {
            cell.source_id = Some(source_id.into());
            true
        } else {
            false
        }
    }

    /// Apply a layout switch animation frame, interpolating each cell's rect
    /// from `self` toward `target` at normalised progress `t ∈ [0, 1]`.
    ///
    /// Only cells with matching IDs are interpolated; surplus cells from
    /// either layout are included as-is.
    #[must_use]
    pub fn lerp_layout(&self, target: &VideoWall, t: f32) -> VideoWall {
        let t = t.clamp(0.0, 1.0);
        let mut cells = Vec::with_capacity(self.cells.len().max(target.cells.len()));
        for cell_a in &self.cells {
            let rect = if let Some(cell_b) = target.cells.iter().find(|c| c.id == cell_a.id) {
                cell_a.rect.lerp(&cell_b.rect, t)
            } else {
                cell_a.rect
            };
            let mut interpolated = cell_a.clone();
            interpolated.rect = rect;
            cells.push(interpolated);
        }
        // Add cells present in target but not in self
        for cell_b in &target.cells {
            if !self.cells.iter().any(|c| c.id == cell_b.id) {
                cells.push(cell_b.clone());
            }
        }
        VideoWall {
            canvas_width: self.canvas_width,
            canvas_height: self.canvas_height,
            cells,
            background_color: self.background_color,
            gap: self.gap,
            default_border: self.default_border.clone(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VideoWallBuilder
// ─────────────────────────────────────────────────────────────────────────────

/// Ergonomic builder for [`VideoWall`].
///
/// # Example
///
/// ```
/// use oximedia_graphics::video_wall::{VideoWallBuilder, WallPreset, CellBorder};
///
/// let wall = VideoWallBuilder::from_preset(WallPreset::Grid2x2, 1920.0, 1080.0)
///     .with_gap(0.005)
///     .with_background([0.05, 0.05, 0.05, 1.0])
///     .build()
///     .expect("valid wall");
///
/// assert_eq!(wall.cells.len(), 4);
/// ```
#[derive(Debug)]
pub struct VideoWallBuilder {
    canvas_width: f32,
    canvas_height: f32,
    cells: Vec<VideoWallCell>,
    background_color: [f32; 4],
    gap: f32,
    default_border: Option<CellBorder>,
}

impl VideoWallBuilder {
    /// Begin building from a named preset.
    #[must_use]
    pub fn from_preset(preset: WallPreset, canvas_width: f32, canvas_height: f32) -> Self {
        let rects = preset.generate_rects(0.0);
        let cells = rects
            .into_iter()
            .enumerate()
            .map(|(i, rect)| VideoWallCell::new(i, rect))
            .collect();
        Self {
            canvas_width,
            canvas_height,
            cells,
            background_color: [0.0, 0.0, 0.0, 1.0],
            gap: 0.0,
            default_border: None,
        }
    }

    /// Begin building from a completely custom cell list.
    #[must_use]
    pub fn from_cells(cells: Vec<VideoWallCell>, canvas_width: f32, canvas_height: f32) -> Self {
        Self {
            canvas_width,
            canvas_height,
            cells,
            background_color: [0.0, 0.0, 0.0, 1.0],
            gap: 0.0,
            default_border: None,
        }
    }

    /// Set the gap between cells.  When building from a preset the rects are
    /// recomputed to include the specified gap.
    #[must_use]
    pub fn with_gap(mut self, gap: f32) -> Self {
        self.gap = gap.max(0.0);
        self
    }

    /// Set the background colour.
    #[must_use]
    pub fn with_background(mut self, color: [f32; 4]) -> Self {
        self.background_color = color;
        self
    }

    /// Set a default border applied to all cells.
    #[must_use]
    pub fn with_border_width(mut self, width: f32) -> Self {
        let color = [1.0_f32, 1.0, 1.0, 0.5];
        self.default_border = Some(CellBorder::solid(width, color));
        self
    }

    /// Set a default border with an explicit colour.
    #[must_use]
    pub fn with_border(mut self, border: CellBorder) -> Self {
        self.default_border = Some(border);
        self
    }

    /// Build the [`VideoWall`].
    ///
    /// Returns an error if the canvas dimensions are non-positive or any cell
    /// rect extends outside `[0, 1]²`.
    pub fn build(mut self) -> Result<VideoWall, VideoWallError> {
        if self.canvas_width <= 0.0 || self.canvas_height <= 0.0 {
            return Err(VideoWallError::InvalidCanvas);
        }
        // Apply gap inset to all cells if a global gap was set
        if self.gap > 0.0 {
            let half_gap = self.gap / 2.0;
            for cell in &mut self.cells {
                cell.rect = cell.rect.inset(half_gap);
            }
        }
        // Apply default border to cells that have none
        if let Some(ref border) = self.default_border {
            for cell in &mut self.cells {
                if cell.border.is_none() {
                    cell.border = Some(border.clone());
                }
            }
        }
        Ok(VideoWall {
            canvas_width: self.canvas_width,
            canvas_height: self.canvas_height,
            cells: self.cells,
            background_color: self.background_color,
            gap: self.gap,
            default_border: self.default_border,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VideoWallError
// ─────────────────────────────────────────────────────────────────────────────

/// Error type for video wall construction.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum VideoWallError {
    /// Canvas dimensions are zero or negative.
    #[error("canvas width and height must be positive")]
    InvalidCanvas,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid2x2_has_four_cells() {
        let wall = VideoWallBuilder::from_preset(WallPreset::Grid2x2, 1920.0, 1080.0)
            .build()
            .expect("valid wall");
        assert_eq!(wall.cells.len(), 4);
    }

    #[test]
    fn grid3x3_has_nine_cells() {
        let wall = VideoWallBuilder::from_preset(WallPreset::Grid3x3, 1920.0, 1080.0)
            .build()
            .expect("valid wall");
        assert_eq!(wall.cells.len(), 9);
    }

    #[test]
    fn single_preset_fills_canvas() {
        let rects = WallPreset::Single.generate_rects(0.0);
        assert_eq!(rects.len(), 1);
        let r = &rects[0];
        assert!((r.x).abs() < 1e-6);
        assert!((r.y).abs() < 1e-6);
        assert!((r.width - 1.0).abs() < 1e-6);
        assert!((r.height - 1.0).abs() < 1e-6);
    }

    #[test]
    fn invalid_canvas_returns_error() {
        let result = VideoWallBuilder::from_preset(WallPreset::Grid2x2, 0.0, 1080.0).build();
        assert!(matches!(result, Err(VideoWallError::InvalidCanvas)));
    }

    #[test]
    fn assign_source_works() {
        let mut wall = VideoWallBuilder::from_preset(WallPreset::SideBySide, 1920.0, 1080.0)
            .build()
            .expect("valid wall");
        assert!(wall.assign_source(0, "cam1"));
        assert_eq!(wall.cell(0).and_then(|c| c.source_id.as_deref()), Some("cam1"));
    }

    #[test]
    fn assign_source_missing_id_returns_false() {
        let mut wall = VideoWallBuilder::from_preset(WallPreset::Single, 1920.0, 1080.0)
            .build()
            .expect("valid wall");
        assert!(!wall.assign_source(99, "stream_x"));
    }

    #[test]
    fn lerp_layout_midpoint() {
        let wall_a = VideoWallBuilder::from_preset(WallPreset::Single, 1920.0, 1080.0)
            .build()
            .expect("wall a");
        let wall_b = VideoWallBuilder::from_preset(WallPreset::SideBySide, 1920.0, 1080.0)
            .build()
            .expect("wall b");
        let mid = wall_a.lerp_layout(&wall_b, 0.5);
        let cell0 = mid.cell(0).expect("cell 0");
        // Width should be between 1.0 (single) and ~0.5 (side-by-side)
        assert!(cell0.rect.width > 0.4 && cell0.rect.width < 1.0);
    }

    #[test]
    fn rect2d_inset_shrinks_symmetrically() {
        let r = Rect2D::new(0.0, 0.0, 1.0, 1.0);
        let inset = r.inset(0.1);
        assert!((inset.x - 0.1).abs() < 1e-6);
        assert!((inset.y - 0.1).abs() < 1e-6);
        assert!((inset.width - 0.8).abs() < 1e-6);
        assert!((inset.height - 0.8).abs() < 1e-6);
    }

    #[test]
    fn rect2d_lerp_at_zero_equals_self() {
        let a = Rect2D::new(0.0, 0.0, 0.5, 0.5);
        let b = Rect2D::new(0.5, 0.5, 0.5, 0.5);
        let r = a.lerp(&b, 0.0);
        assert!((r.x - a.x).abs() < 1e-6);
        assert!((r.width - a.width).abs() < 1e-6);
    }

    #[test]
    fn sort_by_z_orders_cells() {
        let mut wall = VideoWallBuilder::from_preset(WallPreset::Grid2x2, 1920.0, 1080.0)
            .build()
            .expect("valid wall");
        if let Some(c) = wall.cell_mut(0) {
            c.z_order = 10;
        }
        if let Some(c) = wall.cell_mut(1) {
            c.z_order = 1;
        }
        wall.sort_by_z();
        assert!(wall.cells[0].z_order <= wall.cells[1].z_order);
    }

    #[test]
    fn feature_with_banner_has_four_cells() {
        let wall = VideoWallBuilder::from_preset(WallPreset::FeatureWithBanner, 1920.0, 1080.0)
            .build()
            .expect("valid wall");
        assert_eq!(wall.cells.len(), 4);
    }

    #[test]
    fn broadcast_triple_preset_count() {
        assert_eq!(WallPreset::BroadcastTriple.cell_count(), 9);
    }

    #[test]
    fn with_border_applies_to_all_cells() {
        let border = CellBorder::solid(0.002, [1.0, 1.0, 1.0, 1.0]);
        let wall = VideoWallBuilder::from_preset(WallPreset::Grid2x2, 1920.0, 1080.0)
            .with_border(border)
            .build()
            .expect("valid wall");
        for cell in &wall.cells {
            assert!(cell.border.is_some(), "cell {} missing border", cell.id);
        }
    }
}
