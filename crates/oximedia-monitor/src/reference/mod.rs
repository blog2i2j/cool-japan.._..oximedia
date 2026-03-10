//! Reference signal comparison.

pub mod compare;
pub mod diff;

use crate::{MonitorError, MonitorResult};
use serde::{Deserialize, Serialize};

pub use compare::SignalComparator;
pub use diff::DifferenceCalculator;

/// Reference difference metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReferenceDiff {
    /// Mean squared error.
    pub mse: f32,

    /// Peak signal-to-noise ratio (dB).
    pub psnr: f32,

    /// Structural similarity index.
    pub ssim: f32,

    /// Difference percentage.
    pub diff_percentage: f32,
}

/// Reference comparator.
pub struct ReferenceComparator {
    reference_frame: Option<Vec<u8>>,
    reference_width: u32,
    reference_height: u32,
}

impl ReferenceComparator {
    /// Create a new reference comparator.
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    pub fn new() -> MonitorResult<Self> {
        Ok(Self {
            reference_frame: None,
            reference_width: 0,
            reference_height: 0,
        })
    }

    /// Set reference frame.
    pub fn set_reference(&mut self, frame: Vec<u8>, width: u32, height: u32) {
        self.reference_frame = Some(frame);
        self.reference_width = width;
        self.reference_height = height;
    }

    /// Compare current frame to reference.
    ///
    /// # Errors
    ///
    /// Returns an error if comparison fails.
    pub fn compare(&self, frame: &[u8], width: u32, height: u32) -> MonitorResult<ReferenceDiff> {
        if let Some(ref reference) = self.reference_frame {
            if width != self.reference_width || height != self.reference_height {
                return Err(MonitorError::ProcessingError(
                    "Frame dimensions do not match reference".to_string(),
                ));
            }

            let diff = self.calculate_difference(reference, frame, width, height);
            Ok(diff)
        } else {
            Err(MonitorError::ProcessingError(
                "No reference frame set".to_string(),
            ))
        }
    }

    fn calculate_difference(&self, ref_frame: &[u8], frame: &[u8], width: u32, height: u32) -> ReferenceDiff {
        let mut mse = 0.0f32;
        let pixel_count = (width * height * 3) as usize;

        for i in 0..pixel_count.min(ref_frame.len()).min(frame.len()) {
            let diff = f32::from(ref_frame[i]) - f32::from(frame[i]);
            mse += diff * diff;
        }

        mse /= pixel_count as f32;

        let psnr = if mse > 0.0 {
            20.0 * (255.0f32).log10() - 10.0 * mse.log10()
        } else {
            f32::INFINITY
        };

        ReferenceDiff {
            mse,
            psnr,
            ssim: 0.0, // Simplified, would need full SSIM calculation
            diff_percentage: (mse.sqrt() / 255.0) * 100.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reference_comparator() {
        let result = ReferenceComparator::new();
        assert!(result.is_ok());
    }

    #[test]
    fn test_reference_compare() {
        let mut comparator = ReferenceComparator::new().expect("failed to create");

        let reference = vec![128u8; 100 * 100 * 3];
        comparator.set_reference(reference, 100, 100);

        let frame = vec![128u8; 100 * 100 * 3];
        let result = comparator.compare(&frame, 100, 100);
        assert!(result.is_ok());

        let diff = result.expect("result should be valid");
        assert_eq!(diff.mse, 0.0);
        assert_eq!(diff.psnr, f32::INFINITY);
    }
}
