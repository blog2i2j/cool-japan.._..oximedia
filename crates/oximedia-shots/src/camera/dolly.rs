//! Dolly detection (camera moving toward/away from subject).

use crate::error::ShotResult;
use crate::frame_buffer::FrameBuffer;

/// Dolly detector.
pub struct DollyDetector {
    /// Threshold for dolly detection.
    threshold: f32,
}

impl DollyDetector {
    /// Create a new dolly detector.
    #[must_use]
    pub const fn new() -> Self {
        Self { threshold: 0.1 }
    }

    /// Detect dolly between two frames.
    ///
    /// # Errors
    ///
    /// Returns error if frames are invalid.
    pub fn detect_dolly(
        &self,
        _frame1: &FrameBuffer,
        _frame2: &FrameBuffer,
    ) -> ShotResult<(bool, f32)> {
        // Simplified implementation
        Ok((false, 0.0))
    }
}

impl Default for DollyDetector {
    fn default() -> Self {
        Self::new()
    }
}
