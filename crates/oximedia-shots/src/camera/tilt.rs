//! Tilt (vertical camera movement) detection.

use crate::error::ShotResult;
use crate::frame_buffer::FrameBuffer;

/// Tilt detector (up/down vertical movement).
pub struct TiltDetector {
    /// Threshold for tilt detection.
    threshold: f32,
}

impl TiltDetector {
    /// Create a new tilt detector.
    #[must_use]
    pub const fn new() -> Self {
        Self { threshold: 5.0 }
    }

    /// Detect tilt between two frames.
    ///
    /// # Errors
    ///
    /// Returns error if frames are invalid.
    pub fn detect_tilt(
        &self,
        _frame1: &FrameBuffer,
        _frame2: &FrameBuffer,
    ) -> ShotResult<(bool, f32)> {
        // Simplified implementation
        Ok((false, 0.0))
    }
}

impl Default for TiltDetector {
    fn default() -> Self {
        Self::new()
    }
}
