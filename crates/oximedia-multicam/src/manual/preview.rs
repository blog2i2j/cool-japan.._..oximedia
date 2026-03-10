//! Preview management for multi-camera control.

use crate::{AngleId, Result};

/// Preview manager
#[derive(Debug)]
pub struct PreviewManager {
    /// Number of angles
    angle_count: usize,
    /// Active preview windows
    preview_windows: Vec<PreviewWindow>,
    /// Selected preview
    selected_preview: Option<usize>,
}

/// Preview window
#[derive(Debug, Clone)]
pub struct PreviewWindow {
    /// Angle being previewed
    pub angle: AngleId,
    /// Window position (x, y)
    pub position: (u32, u32),
    /// Window size (width, height)
    pub size: (u32, u32),
    /// Visible flag
    pub visible: bool,
    /// Label
    pub label: String,
}

impl PreviewWindow {
    /// Create a new preview window
    #[must_use]
    pub fn new(angle: AngleId, position: (u32, u32), size: (u32, u32)) -> Self {
        Self {
            angle,
            position,
            size,
            visible: true,
            label: format!("Angle {}", angle + 1),
        }
    }

    /// Set label
    pub fn set_label(&mut self, label: String) {
        self.label = label;
    }

    /// Set visibility
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    /// Get aspect ratio
    #[must_use]
    pub fn aspect_ratio(&self) -> f32 {
        self.size.0 as f32 / self.size.1 as f32
    }
}

impl PreviewManager {
    /// Create a new preview manager
    #[must_use]
    pub fn new(angle_count: usize) -> Self {
        Self {
            angle_count,
            preview_windows: Vec::new(),
            selected_preview: None,
        }
    }

    /// Add preview window
    pub fn add_preview(&mut self, window: PreviewWindow) {
        self.preview_windows.push(window);
    }

    /// Create grid layout
    pub fn create_grid_layout(&mut self, rows: u32, cols: u32, width: u32, height: u32) {
        self.preview_windows.clear();

        let cell_width = width / cols;
        let cell_height = height / rows;

        let mut angle = 0;
        for row in 0..rows {
            for col in 0..cols {
                if angle < self.angle_count {
                    let window = PreviewWindow::new(
                        angle,
                        (col * cell_width, row * cell_height),
                        (cell_width, cell_height),
                    );
                    self.preview_windows.push(window);
                    angle += 1;
                }
            }
        }
    }

    /// Get preview window by index
    #[must_use]
    pub fn get_preview(&self, index: usize) -> Option<&PreviewWindow> {
        self.preview_windows.get(index)
    }

    /// Get mutable preview window by index
    pub fn get_preview_mut(&mut self, index: usize) -> Option<&mut PreviewWindow> {
        self.preview_windows.get_mut(index)
    }

    /// Get preview window by angle
    #[must_use]
    pub fn get_preview_by_angle(&self, angle: AngleId) -> Option<&PreviewWindow> {
        self.preview_windows.iter().find(|w| w.angle == angle)
    }

    /// Select preview
    ///
    /// # Errors
    ///
    /// Returns an error if index is invalid
    pub fn select_preview(&mut self, index: usize) -> Result<()> {
        if index >= self.preview_windows.len() {
            return Err(crate::MultiCamError::InvalidOperation(
                "Invalid preview index".to_string(),
            ));
        }

        self.selected_preview = Some(index);
        Ok(())
    }

    /// Get selected preview
    #[must_use]
    pub fn selected_preview(&self) -> Option<&PreviewWindow> {
        self.selected_preview
            .and_then(|i| self.preview_windows.get(i))
    }

    /// Clear selection
    pub fn clear_selection(&mut self) {
        self.selected_preview = None;
    }

    /// Get all visible previews
    #[must_use]
    pub fn visible_previews(&self) -> Vec<&PreviewWindow> {
        self.preview_windows.iter().filter(|w| w.visible).collect()
    }

    /// Show all previews
    pub fn show_all(&mut self) {
        for window in &mut self.preview_windows {
            window.visible = true;
        }
    }

    /// Hide all previews
    pub fn hide_all(&mut self) {
        for window in &mut self.preview_windows {
            window.visible = false;
        }
    }

    /// Get preview count
    #[must_use]
    pub fn preview_count(&self) -> usize {
        self.preview_windows.len()
    }
}

/// Multi-view layout preset
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiViewLayout {
    /// 2x2 grid (4 angles)
    Grid2x2,
    /// 3x3 grid (9 angles)
    Grid3x3,
    /// 4x4 grid (16 angles)
    Grid4x4,
    /// 1+5 (main + 5 small)
    MainPlus5,
    /// 1+7 (main + 7 small)
    MainPlus7,
    /// Custom layout
    Custom,
}

impl MultiViewLayout {
    /// Get grid dimensions (rows, cols)
    #[must_use]
    pub fn dimensions(&self) -> (u32, u32) {
        match self {
            MultiViewLayout::Grid2x2 => (2, 2),
            MultiViewLayout::Grid3x3 => (3, 3),
            MultiViewLayout::Grid4x4 => (4, 4),
            MultiViewLayout::MainPlus5 => (2, 3),
            MultiViewLayout::MainPlus7 => (2, 4),
            MultiViewLayout::Custom => (1, 1),
        }
    }

    /// Get maximum angle count
    #[must_use]
    pub fn max_angles(&self) -> usize {
        let (rows, cols) = self.dimensions();
        (rows * cols) as usize
    }
}

/// Multi-view builder
#[derive(Debug)]
pub struct MultiViewBuilder {
    /// Layout preset
    layout: MultiViewLayout,
    /// Total width
    width: u32,
    /// Total height
    height: u32,
}

impl MultiViewBuilder {
    /// Create a new multi-view builder
    #[must_use]
    pub fn new(layout: MultiViewLayout, width: u32, height: u32) -> Self {
        Self {
            layout,
            width,
            height,
        }
    }

    /// Build preview manager with layout
    #[must_use]
    pub fn build(&self, angle_count: usize) -> PreviewManager {
        let mut manager = PreviewManager::new(angle_count);

        match self.layout {
            MultiViewLayout::Grid2x2 | MultiViewLayout::Grid3x3 | MultiViewLayout::Grid4x4 => {
                let (rows, cols) = self.layout.dimensions();
                manager.create_grid_layout(rows, cols, self.width, self.height);
            }
            MultiViewLayout::MainPlus5 => {
                self.build_main_plus(&mut manager, 5);
            }
            MultiViewLayout::MainPlus7 => {
                self.build_main_plus(&mut manager, 7);
            }
            MultiViewLayout::Custom => {
                // Leave empty for custom layout
            }
        }

        manager
    }

    /// Build main + small windows layout
    fn build_main_plus(&self, manager: &mut PreviewManager, small_count: usize) {
        // Main window takes 2/3 of width
        let main_width = (self.width * 2) / 3;
        let small_width = self.width - main_width;
        let small_height = self.height / small_count as u32;

        // Add main window
        manager.add_preview(PreviewWindow::new(0, (0, 0), (main_width, self.height)));

        // Add small windows
        for i in 0..small_count.min(manager.angle_count - 1) {
            let y = i as u32 * small_height;
            manager.add_preview(PreviewWindow::new(
                i + 1,
                (main_width, y),
                (small_width, small_height),
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preview_window_creation() {
        let window = PreviewWindow::new(0, (0, 0), (1920, 1080));
        assert_eq!(window.angle, 0);
        assert_eq!(window.size, (1920, 1080));
        assert!(window.visible);
    }

    #[test]
    fn test_aspect_ratio() {
        let window = PreviewWindow::new(0, (0, 0), (1920, 1080));
        let ratio = window.aspect_ratio();
        assert!((ratio - 16.0 / 9.0).abs() < 0.01);
    }

    #[test]
    fn test_preview_manager_creation() {
        let manager = PreviewManager::new(4);
        assert_eq!(manager.angle_count, 4);
        assert_eq!(manager.preview_count(), 0);
    }

    #[test]
    fn test_create_grid_layout() {
        let mut manager = PreviewManager::new(4);
        manager.create_grid_layout(2, 2, 1920, 1080);
        assert_eq!(manager.preview_count(), 4);
    }

    #[test]
    fn test_select_preview() {
        let mut manager = PreviewManager::new(4);
        manager.create_grid_layout(2, 2, 1920, 1080);

        assert!(manager.select_preview(0).is_ok());
        assert!(manager.selected_preview().is_some());
        assert_eq!(
            manager
                .selected_preview()
                .expect("multicam test operation should succeed")
                .angle,
            0
        );
    }

    #[test]
    fn test_visibility() {
        let mut manager = PreviewManager::new(4);
        manager.create_grid_layout(2, 2, 1920, 1080);

        manager.hide_all();
        assert_eq!(manager.visible_previews().len(), 0);

        manager.show_all();
        assert_eq!(manager.visible_previews().len(), 4);
    }

    #[test]
    fn test_multiview_layout() {
        assert_eq!(MultiViewLayout::Grid2x2.dimensions(), (2, 2));
        assert_eq!(MultiViewLayout::Grid3x3.max_angles(), 9);
    }

    #[test]
    fn test_multiview_builder() {
        let builder = MultiViewBuilder::new(MultiViewLayout::Grid2x2, 1920, 1080);
        let manager = builder.build(4);
        assert_eq!(manager.preview_count(), 4);
    }

    #[test]
    fn test_main_plus_5() {
        let builder = MultiViewBuilder::new(MultiViewLayout::MainPlus5, 1920, 1080);
        let manager = builder.build(6);
        assert_eq!(manager.preview_count(), 6);
    }
}
