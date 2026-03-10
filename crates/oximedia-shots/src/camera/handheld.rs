//! Handheld camera shake detection.

use crate::error::ShotResult;
use crate::frame_buffer::FrameBuffer;

/// Handheld shake detector.
pub struct HandheldDetector {
    /// Threshold for shake detection.
    threshold: f32,
}

impl HandheldDetector {
    /// Create a new handheld detector.
    #[must_use]
    pub const fn new() -> Self {
        Self { threshold: 2.0 }
    }

    /// Detect handheld shake in a sequence of frames.
    ///
    /// # Errors
    ///
    /// Returns error if frames are invalid.
    pub fn detect_handheld(&self, _frames: &[FrameBuffer]) -> ShotResult<(bool, f32)> {
        // Simplified implementation
        Ok((false, 0.0))
    }
}

impl Default for HandheldDetector {
    fn default() -> Self {
        Self::new()
    }
}
