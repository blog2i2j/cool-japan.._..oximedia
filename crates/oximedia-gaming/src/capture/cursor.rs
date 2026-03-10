//! Cursor capture and overlay.
//!
//! Provides cursor position tracking and overlay rendering.

use crate::GamingResult;

/// Cursor capture and overlay.
pub struct CursorCapture {
    enabled: bool,
    last_position: (i32, i32),
}

/// Cursor information.
#[derive(Debug, Clone, Copy)]
pub struct CursorInfo {
    /// Cursor position (x, y)
    pub position: (i32, i32),
    /// Cursor visibility
    pub visible: bool,
    /// Cursor type
    pub cursor_type: CursorType,
}

/// Cursor type/shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CursorType {
    /// Default arrow cursor
    Arrow,
    /// Text input cursor
    IBeam,
    /// Hand/pointer cursor
    Hand,
    /// Crosshair cursor
    Crosshair,
    /// Resize cursor
    Resize,
    /// Wait/busy cursor
    Wait,
    /// Custom cursor
    Custom,
}

impl CursorCapture {
    /// Create a new cursor capture.
    #[must_use]
    pub fn new() -> Self {
        Self {
            enabled: true,
            last_position: (0, 0),
        }
    }

    /// Enable cursor capture.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable cursor capture.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Get current cursor information.
    ///
    /// # Errors
    ///
    /// Returns error if cursor info retrieval fails.
    pub fn get_cursor_info(&self) -> GamingResult<CursorInfo> {
        Ok(CursorInfo {
            position: self.last_position,
            visible: self.enabled,
            cursor_type: CursorType::Arrow,
        })
    }

    /// Update cursor position.
    pub fn update_position(&mut self, x: i32, y: i32) {
        self.last_position = (x, y);
    }

    /// Check if cursor capture is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl Default for CursorCapture {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cursor_capture_creation() {
        let capture = CursorCapture::new();
        assert!(capture.is_enabled());
    }

    #[test]
    fn test_enable_disable() {
        let mut capture = CursorCapture::new();

        capture.disable();
        assert!(!capture.is_enabled());

        capture.enable();
        assert!(capture.is_enabled());
    }

    #[test]
    fn test_position_update() {
        let mut capture = CursorCapture::new();
        capture.update_position(100, 200);

        let info = capture
            .get_cursor_info()
            .expect("cursor info should succeed");
        assert_eq!(info.position, (100, 200));
    }

    #[test]
    fn test_cursor_info() {
        let capture = CursorCapture::new();
        let info = capture
            .get_cursor_info()
            .expect("cursor info should succeed");

        assert_eq!(info.cursor_type, CursorType::Arrow);
        assert!(info.visible);
    }
}
