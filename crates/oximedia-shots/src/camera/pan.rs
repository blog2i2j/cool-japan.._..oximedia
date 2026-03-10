//! Pan (horizontal camera movement) detection.

use crate::error::ShotResult;
use crate::frame_buffer::FrameBuffer;

/// Pan detector (left/right horizontal movement).
pub struct PanDetector {
    /// Threshold for pan detection.
    threshold: f32,
}

impl PanDetector {
    /// Create a new pan detector.
    #[must_use]
    pub const fn new() -> Self {
        Self { threshold: 5.0 }
    }

    /// Detect pan between two frames.
    ///
    /// # Errors
    ///
    /// Returns error if frames are invalid.
    pub fn detect_pan(
        &self,
        _frame1: &FrameBuffer,
        _frame2: &FrameBuffer,
    ) -> ShotResult<(bool, f32)> {
        // Simplified implementation
        Ok((false, 0.0))
    }
}

impl Default for PanDetector {
    fn default() -> Self {
        Self::new()
    }
}
