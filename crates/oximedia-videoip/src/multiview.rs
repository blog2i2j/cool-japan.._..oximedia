//! Multi-view compositor: combines multiple receiver streams into a mosaic.
//!
//! The `Multiview` compositor arranges N input video streams on a canvas using
//! configurable grid or custom layouts.  Each cell specifies the source stream,
//! its position/size on the canvas, and optional label.
//!
//! Compositing is done in RGBA8 colour space; callers supply raw pixel buffers
//! and receive a combined RGBA8 output canvas.

#![allow(dead_code)]

/// Error type for multiview operations.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MultiviewError {
    /// Layout is invalid (e.g., cells overlap the canvas boundary).
    #[error("invalid layout: {0}")]
    InvalidLayout(String),
    /// Source stream not registered.
    #[error("source '{0}' not registered")]
    SourceNotFound(String),
    /// Frame dimensions mismatch the cell size.
    #[error("frame dimension mismatch for source '{src}': expected {ew}x{eh}, got {fw}x{fh}")]
    DimensionMismatch {
        /// Source ID.
        src: String,
        /// Expected width.
        ew: u32,
        /// Expected height.
        eh: u32,
        /// Actual frame width.
        fw: u32,
        /// Actual frame height.
        fh: u32,
    },
}

/// Result type for multiview operations.
pub type MultiviewResult<T> = Result<T, MultiviewError>;

/// A cell in the multiview layout, describing where a source is placed on the
/// output canvas.
#[derive(Debug, Clone)]
pub struct MultiviewCell {
    /// Unique identifier for the cell.
    pub id: String,
    /// Horizontal offset on the output canvas (pixels).
    pub x: u32,
    /// Vertical offset on the output canvas (pixels).
    pub y: u32,
    /// Cell width on the output canvas (pixels).
    pub width: u32,
    /// Cell height on the output canvas (pixels).
    pub height: u32,
    /// Optional label rendered in the cell (for on-screen display).
    pub label: Option<String>,
    /// ID of the source stream assigned to this cell (`None` = blank/muted).
    pub source_id: Option<String>,
}

impl MultiviewCell {
    /// Creates a new cell.
    #[must_use]
    pub fn new(id: impl Into<String>, x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            id: id.into(),
            x,
            y,
            width,
            height,
            label: None,
            source_id: None,
        }
    }

    /// Assigns a source stream to this cell.
    pub fn with_source(mut self, source_id: impl Into<String>) -> Self {
        self.source_id = Some(source_id.into());
        self
    }

    /// Attaches a label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Returns `true` if the cell extends beyond the canvas boundary.
    #[must_use]
    pub fn fits_in_canvas(&self, canvas_w: u32, canvas_h: u32) -> bool {
        self.x.saturating_add(self.width) <= canvas_w
            && self.y.saturating_add(self.height) <= canvas_h
    }
}

/// Pre-built mosaic layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MosaicLayout {
    /// 1 source, full canvas.
    Single,
    /// 2 sources side by side.
    Side2,
    /// 4 sources in a 2×2 grid.
    Grid2x2,
    /// 9 sources in a 3×3 grid.
    Grid3x3,
    /// 16 sources in a 4×4 grid.
    Grid4x4,
    /// Picture-in-picture: one large source + one small overlay.
    Pip,
}

impl MosaicLayout {
    /// Generates `MultiviewCell` list for the given canvas dimensions.
    #[must_use]
    pub fn build_cells(&self, canvas_w: u32, canvas_h: u32) -> Vec<MultiviewCell> {
        match self {
            Self::Single => vec![
                MultiviewCell::new("0", 0, 0, canvas_w, canvas_h),
            ],
            Self::Side2 => {
                let half = canvas_w / 2;
                vec![
                    MultiviewCell::new("0", 0, 0, half, canvas_h),
                    MultiviewCell::new("1", half, 0, half, canvas_h),
                ]
            }
            Self::Grid2x2 => {
                let hw = canvas_w / 2;
                let hh = canvas_h / 2;
                vec![
                    MultiviewCell::new("0", 0, 0, hw, hh),
                    MultiviewCell::new("1", hw, 0, hw, hh),
                    MultiviewCell::new("2", 0, hh, hw, hh),
                    MultiviewCell::new("3", hw, hh, hw, hh),
                ]
            }
            Self::Grid3x3 => {
                let cw = canvas_w / 3;
                let ch = canvas_h / 3;
                (0..9u32)
                    .map(|i| {
                        let row = i / 3;
                        let col = i % 3;
                        MultiviewCell::new(
                            format!("{i}"),
                            col * cw,
                            row * ch,
                            cw,
                            ch,
                        )
                    })
                    .collect()
            }
            Self::Grid4x4 => {
                let cw = canvas_w / 4;
                let ch = canvas_h / 4;
                (0..16u32)
                    .map(|i| {
                        let row = i / 4;
                        let col = i % 4;
                        MultiviewCell::new(
                            format!("{i}"),
                            col * cw,
                            row * ch,
                            cw,
                            ch,
                        )
                    })
                    .collect()
            }
            Self::Pip => {
                // Large background (full canvas) + small overlay (top-right, 25% width/height)
                let pip_w = canvas_w / 4;
                let pip_h = canvas_h / 4;
                let pip_x = canvas_w - pip_w;
                let pip_y = 0;
                vec![
                    MultiviewCell::new("main", 0, 0, canvas_w, canvas_h),
                    MultiviewCell::new("pip", pip_x, pip_y, pip_w, pip_h),
                ]
            }
        }
    }
}

/// A registered video source for the compositor.
#[derive(Debug, Clone)]
struct RegisteredSource {
    /// Most-recently received frame (RGBA8, `width * height * 4` bytes).
    frame: Vec<u8>,
    /// Frame width.
    width: u32,
    /// Frame height.
    height: u32,
    /// Number of frame updates received.
    frame_count: u64,
}

/// Multi-view compositor that blits source frames onto a single output canvas.
#[derive(Debug)]
pub struct MultiviewCompositor {
    /// Canvas width in pixels.
    canvas_width: u32,
    /// Canvas height in pixels.
    canvas_height: u32,
    /// Layout cells (define placement of each source on the canvas).
    cells: Vec<MultiviewCell>,
    /// Registered source frames by source ID.
    sources: std::collections::HashMap<String, RegisteredSource>,
    /// Background colour (RGBA8) for unoccupied canvas areas.
    background: [u8; 4],
}

impl MultiviewCompositor {
    /// Creates a new compositor with a custom cell layout.
    pub fn new(
        canvas_width: u32,
        canvas_height: u32,
        cells: Vec<MultiviewCell>,
    ) -> MultiviewResult<Self> {
        for cell in &cells {
            if !cell.fits_in_canvas(canvas_width, canvas_height) {
                return Err(MultiviewError::InvalidLayout(format!(
                    "cell '{}' at ({},{}) {}x{} exceeds canvas {}x{}",
                    cell.id, cell.x, cell.y, cell.width, cell.height,
                    canvas_width, canvas_height
                )));
            }
        }
        Ok(Self {
            canvas_width,
            canvas_height,
            cells,
            sources: std::collections::HashMap::new(),
            background: [0, 0, 0, 255],
        })
    }

    /// Creates a compositor with one of the built-in mosaic layouts.
    pub fn with_layout(
        canvas_width: u32,
        canvas_height: u32,
        layout: MosaicLayout,
    ) -> MultiviewResult<Self> {
        let cells = layout.build_cells(canvas_width, canvas_height);
        Self::new(canvas_width, canvas_height, cells)
    }

    /// Sets the background colour (RGBA8) for blank areas.
    pub fn set_background(&mut self, rgba: [u8; 4]) {
        self.background = rgba;
    }

    /// Registers a new source stream.  The expected frame dimensions must be
    /// supplied so the compositor can validate incoming frames.
    pub fn register_source(
        &mut self,
        source_id: impl Into<String>,
        width: u32,
        height: u32,
    ) {
        let id = source_id.into();
        self.sources.insert(id, RegisteredSource {
            frame: vec![0u8; (width * height * 4) as usize],
            width,
            height,
            frame_count: 0,
        });
    }

    /// Removes a source stream.
    pub fn unregister_source(&mut self, source_id: &str) {
        self.sources.remove(source_id);
    }

    /// Updates the frame buffer for a source.
    ///
    /// `rgba` must be `width * height * 4` bytes in RGBA8 format.
    pub fn update_source_frame(
        &mut self,
        source_id: &str,
        rgba: &[u8],
    ) -> MultiviewResult<()> {
        let src = self
            .sources
            .get_mut(source_id)
            .ok_or_else(|| MultiviewError::SourceNotFound(source_id.to_owned()))?;

        let expected = (src.width * src.height * 4) as usize;
        if rgba.len() != expected {
            let fw = rgba.len() as u32 / (src.height * 4).max(1);
            let fh = rgba.len() as u32 / (src.width * 4).max(1);
            return Err(MultiviewError::DimensionMismatch {
                src: source_id.to_owned(),
                ew: src.width,
                eh: src.height,
                fw,
                fh,
            });
        }
        src.frame.copy_from_slice(rgba);
        src.frame_count += 1;
        Ok(())
    }

    /// Assigns a source to a named cell.
    pub fn assign_source_to_cell(
        &mut self,
        cell_id: &str,
        source_id: Option<String>,
    ) -> MultiviewResult<()> {
        let cell = self
            .cells
            .iter_mut()
            .find(|c| c.id == cell_id)
            .ok_or_else(|| MultiviewError::SourceNotFound(cell_id.to_owned()))?;
        cell.source_id = source_id;
        Ok(())
    }

    /// Composites all source frames into the output canvas and returns the
    /// result as a flat RGBA8 buffer of `canvas_width * canvas_height * 4` bytes.
    #[must_use]
    pub fn composite(&self) -> Vec<u8> {
        let canvas_size = (self.canvas_width * self.canvas_height * 4) as usize;
        let mut canvas = vec![
            self.background[0],
            self.background[1],
            self.background[2],
            self.background[3],
        ]
        .into_iter()
        .cycle()
        .take(canvas_size)
        .collect::<Vec<u8>>();

        for cell in &self.cells {
            let src_frame = cell
                .source_id
                .as_ref()
                .and_then(|id| self.sources.get(id.as_str()));

            let src = match src_frame {
                Some(s) => s,
                None => continue, // blank cell
            };

            // Simple nearest-neighbour scale blit: src -> cell rectangle.
            let src_w = src.width as f64;
            let src_h = src.height as f64;
            let dst_w = cell.width as f64;
            let dst_h = cell.height as f64;

            for dy in 0..cell.height {
                for dx in 0..cell.width {
                    let sx = ((dx as f64 + 0.5) / dst_w * src_w) as u32;
                    let sy = ((dy as f64 + 0.5) / dst_h * src_h) as u32;
                    let sx = sx.min(src.width.saturating_sub(1));
                    let sy = sy.min(src.height.saturating_sub(1));

                    let src_off = ((sy * src.width + sx) * 4) as usize;
                    let dst_x = cell.x + dx;
                    let dst_y = cell.y + dy;
                    let dst_off = ((dst_y * self.canvas_width + dst_x) * 4) as usize;

                    if dst_off + 3 < canvas_size && src_off + 3 < src.frame.len() {
                        canvas[dst_off..dst_off + 4]
                            .copy_from_slice(&src.frame[src_off..src_off + 4]);
                    }
                }
            }
        }
        canvas
    }

    /// Returns the number of registered sources.
    #[must_use]
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Returns the number of cells in the layout.
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Returns canvas dimensions `(width, height)`.
    #[must_use]
    pub fn canvas_size(&self) -> (u32, u32) {
        (self.canvas_width, self.canvas_height)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_layout_full_canvas() {
        let cells = MosaicLayout::Single.build_cells(1920, 1080);
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].width, 1920);
        assert_eq!(cells[0].height, 1080);
    }

    #[test]
    fn test_grid2x2_four_cells() {
        let cells = MosaicLayout::Grid2x2.build_cells(1920, 1080);
        assert_eq!(cells.len(), 4);
    }

    #[test]
    fn test_grid3x3_nine_cells() {
        let cells = MosaicLayout::Grid3x3.build_cells(1920, 1080);
        assert_eq!(cells.len(), 9);
    }

    #[test]
    fn test_grid4x4_sixteen_cells() {
        let cells = MosaicLayout::Grid4x4.build_cells(1920, 1080);
        assert_eq!(cells.len(), 16);
    }

    #[test]
    fn test_pip_two_cells() {
        let cells = MosaicLayout::Pip.build_cells(1920, 1080);
        assert_eq!(cells.len(), 2);
    }

    #[test]
    fn test_compositor_creation() {
        let comp = MultiviewCompositor::with_layout(1920, 1080, MosaicLayout::Grid2x2).unwrap();
        assert_eq!(comp.cell_count(), 4);
        assert_eq!(comp.canvas_size(), (1920, 1080));
    }

    #[test]
    fn test_invalid_layout_rejected() {
        let bad_cell = MultiviewCell::new("bad", 1900, 1000, 100, 100); // extends beyond 1920x1080
        let result = MultiviewCompositor::new(1920, 1080, vec![bad_cell]);
        assert!(result.is_err());
    }

    #[test]
    fn test_register_and_update_source() {
        let mut comp = MultiviewCompositor::with_layout(4, 4, MosaicLayout::Single).unwrap();
        comp.register_source("cam1", 4, 4);
        let frame = vec![255u8; 4 * 4 * 4];
        comp.update_source_frame("cam1", &frame).unwrap();
        assert_eq!(comp.source_count(), 1);
    }

    #[test]
    fn test_update_unknown_source_error() {
        let mut comp = MultiviewCompositor::with_layout(4, 4, MosaicLayout::Single).unwrap();
        let frame = vec![0u8; 16 * 4];
        let result = comp.update_source_frame("unknown", &frame);
        assert!(matches!(result, Err(MultiviewError::SourceNotFound(_))));
    }

    #[test]
    fn test_composite_all_white() {
        let mut comp = MultiviewCompositor::with_layout(4, 4, MosaicLayout::Single).unwrap();
        comp.register_source("cam1", 4, 4);
        let mut cells = MosaicLayout::Single.build_cells(4, 4);
        cells[0].source_id = Some("cam1".to_owned());
        let mut comp2 = MultiviewCompositor::new(4, 4, cells).unwrap();
        comp2.register_source("cam1", 4, 4);
        let frame = vec![255u8; 4 * 4 * 4];
        comp2.update_source_frame("cam1", &frame).unwrap();
        let canvas = comp2.composite();
        assert!(canvas.iter().all(|&v| v == 255));
    }

    #[test]
    fn test_composite_blank_cell_uses_background() {
        let comp = MultiviewCompositor::with_layout(4, 4, MosaicLayout::Single).unwrap();
        // No source assigned => background colour
        let canvas = comp.composite();
        assert_eq!(canvas.len(), 4 * 4 * 4);
        // Default background is black with alpha 255
        assert_eq!(&canvas[0..4], &[0, 0, 0, 255]);
    }

    #[test]
    fn test_side2_layout() {
        let cells = MosaicLayout::Side2.build_cells(100, 50);
        assert_eq!(cells.len(), 2);
        assert_eq!(cells[0].width, 50);
        assert_eq!(cells[1].x, 50);
    }

    #[test]
    fn test_cell_fits_in_canvas() {
        let cell = MultiviewCell::new("c", 0, 0, 100, 100);
        assert!(cell.fits_in_canvas(100, 100));
        assert!(!cell.fits_in_canvas(99, 100));
    }

    #[test]
    fn test_cell_with_source_and_label() {
        let cell = MultiviewCell::new("c", 0, 0, 100, 100)
            .with_source("cam1")
            .with_label("Camera 1");
        assert_eq!(cell.source_id.as_deref(), Some("cam1"));
        assert_eq!(cell.label.as_deref(), Some("Camera 1"));
    }

    #[test]
    fn test_assign_source_to_cell() {
        let mut comp = MultiviewCompositor::with_layout(100, 100, MosaicLayout::Side2).unwrap();
        comp.assign_source_to_cell("0", Some("cam1".to_owned())).unwrap();
        // verify via source count (source is not registered but assignment works)
        assert_eq!(comp.cell_count(), 2);
    }

    #[test]
    fn test_assign_unknown_cell_error() {
        let mut comp = MultiviewCompositor::with_layout(100, 100, MosaicLayout::Single).unwrap();
        let res = comp.assign_source_to_cell("nonexistent", Some("cam1".to_owned()));
        assert!(matches!(res, Err(MultiviewError::SourceNotFound(_))));
    }

    #[test]
    fn test_unregister_source() {
        let mut comp = MultiviewCompositor::with_layout(4, 4, MosaicLayout::Single).unwrap();
        comp.register_source("cam1", 4, 4);
        assert_eq!(comp.source_count(), 1);
        comp.unregister_source("cam1");
        assert_eq!(comp.source_count(), 0);
    }

    #[test]
    fn test_set_background_colour() {
        let mut comp = MultiviewCompositor::with_layout(4, 4, MosaicLayout::Single).unwrap();
        comp.set_background([255, 0, 0, 255]); // red
        let canvas = comp.composite();
        // Canvas should now be red (no source assigned).
        assert_eq!(&canvas[0..4], &[255, 0, 0, 255]);
    }

    #[test]
    fn test_pip_layout_aspect_ratios() {
        let cells = MosaicLayout::Pip.build_cells(1920, 1080);
        // Main cell covers full canvas.
        assert_eq!(cells[0].width, 1920);
        assert_eq!(cells[0].height, 1080);
        // PIP cell is 1/4 of the canvas dimensions.
        assert_eq!(cells[1].width, 480);
        assert_eq!(cells[1].height, 270);
    }

    #[test]
    fn test_compositor_canvas_size() {
        let comp = MultiviewCompositor::with_layout(640, 360, MosaicLayout::Grid2x2).unwrap();
        assert_eq!(comp.canvas_size(), (640, 360));
    }
}
