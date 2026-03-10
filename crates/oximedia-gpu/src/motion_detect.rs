//! GPU-accelerated motion detection.
//!
//! This module provides CPU-fallback simulations of GPU compute operations
//! for inter-frame motion detection. In production, the pixel difference
//! computation and reduction would be executed via WGPU compute shaders.

use crate::{GpuError, Result};

// ---------------------------------------------------------------------------
// Sensitivity
// ---------------------------------------------------------------------------

/// Motion detection sensitivity level.
///
/// Controls the per-pixel difference threshold above which a pixel is
/// considered to have changed between frames.
#[derive(Debug, Clone, Copy)]
pub enum Sensitivity {
    /// Low sensitivity — threshold = 30.
    Low,
    /// Medium sensitivity — threshold = 15.
    Medium,
    /// High sensitivity — threshold = 8.
    High,
}

impl Sensitivity {
    /// Return the pixel-difference threshold for this sensitivity level.
    #[must_use]
    pub fn threshold(&self) -> u8 {
        match self {
            Self::Low => 30,
            Self::Medium => 15,
            Self::High => 8,
        }
    }
}

// ---------------------------------------------------------------------------
// MotionRegion
// ---------------------------------------------------------------------------

/// Per-region motion information.
#[derive(Debug, Clone)]
pub struct MotionRegion {
    /// X coordinate of the region's top-left corner (in pixels).
    pub x: u32,
    /// Y coordinate of the region's top-left corner (in pixels).
    pub y: u32,
    /// Width of the region in pixels.
    pub width: u32,
    /// Height of the region in pixels.
    pub height: u32,
    /// Normalized motion magnitude in `[0.0, 1.0]`; 0 = no motion, 1 = maximum.
    pub magnitude: f32,
    /// Number of pixels that exceeded the motion threshold.
    pub changed_pixels: u32,
}

// ---------------------------------------------------------------------------
// MotionAnalysis
// ---------------------------------------------------------------------------

/// Frame-level motion analysis result.
#[derive(Debug, Clone)]
pub struct MotionAnalysis {
    /// Normalized global motion magnitude in `[0.0, 1.0]`.
    pub global_magnitude: f32,
    /// Fraction of pixels that changed (0.0–1.0).
    pub changed_pixel_ratio: f32,
    /// Per-region motion information.
    pub regions: Vec<MotionRegion>,
    /// `true` if any motion was detected at the current sensitivity level.
    pub motion_detected: bool,
}

// ---------------------------------------------------------------------------
// MotionDetector
// ---------------------------------------------------------------------------

/// GPU-accelerated motion detector.
///
/// Compares successive frames to detect and quantify pixel-level motion.
/// The first call to [`analyze`] returns a zero-motion result because
/// there is no previous frame to compare against.
///
/// [`analyze`]: MotionDetector::analyze
pub struct MotionDetector {
    sensitivity: Sensitivity,
    /// Number of analysis regions in the horizontal direction.
    region_count_x: u32,
    /// Number of analysis regions in the vertical direction.
    region_count_y: u32,
    prev_frame: Option<Vec<u8>>,
}

impl MotionDetector {
    /// Create a new `MotionDetector`.
    ///
    /// # Arguments
    ///
    /// * `sensitivity` - Detection sensitivity level.
    /// * `region_count_x` - Number of sub-regions horizontally (minimum 1).
    /// * `region_count_y` - Number of sub-regions vertically (minimum 1).
    #[must_use]
    pub fn new(sensitivity: Sensitivity, region_count_x: u32, region_count_y: u32) -> Self {
        Self {
            sensitivity,
            region_count_x: region_count_x.max(1),
            region_count_y: region_count_y.max(1),
            prev_frame: None,
        }
    }

    /// Analyze motion between the previous frame and the current frame.
    ///
    /// On the first call (no previous frame), returns a `MotionAnalysis`
    /// with all metrics at zero. On subsequent calls the current frame is
    /// compared against the previously stored frame.
    ///
    /// # Arguments
    ///
    /// * `frame` - Raw pixel data (single channel / luma expected for best
    ///   results; multi-channel frames work but all bytes are compared).
    /// * `width` - Frame width in pixels.
    /// * `height` - Frame height in pixels.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::InvalidDimensions`] if `width * height == 0`.
    /// Returns [`GpuError::InvalidBufferSize`] if a previous frame was stored
    /// with a different size.
    pub fn analyze(&mut self, frame: &[u8], width: u32, height: u32) -> Result<MotionAnalysis> {
        if width == 0 || height == 0 {
            return Err(GpuError::InvalidDimensions { width, height });
        }

        // First frame — no motion data available yet.
        let Some(prev) = self.prev_frame.take() else {
            self.prev_frame = Some(frame.to_vec());
            return Ok(MotionAnalysis {
                global_magnitude: 0.0,
                changed_pixel_ratio: 0.0,
                regions: vec![],
                motion_detected: false,
            });
        };

        if prev.len() != frame.len() {
            // Replace stored frame before returning the error.
            self.prev_frame = Some(frame.to_vec());
            return Err(GpuError::InvalidBufferSize {
                expected: prev.len(),
                actual: frame.len(),
            });
        }

        let threshold = self.sensitivity.threshold();
        let total_pixels = frame.len() as u32;

        // Global metrics
        let (total_changed, total_diff_sum) =
            prev.iter()
                .zip(frame.iter())
                .fold((0u32, 0u64), |(changed, sum), (&p, &c)| {
                    let diff = p.abs_diff(c);
                    if diff >= threshold {
                        (changed + 1, sum + u64::from(diff))
                    } else {
                        (changed, sum)
                    }
                });

        let global_magnitude = if total_pixels > 0 {
            (total_diff_sum as f64 / (f64::from(total_pixels) * 255.0)) as f32
        } else {
            0.0
        };
        let changed_pixel_ratio = total_changed as f32 / total_pixels as f32;
        let motion_detected = total_changed > 0;

        // Per-region metrics
        let regions = self.compute_regions(frame, &prev, width, height, threshold, total_pixels);

        // Store current frame for next call.
        self.prev_frame = Some(frame.to_vec());

        Ok(MotionAnalysis {
            global_magnitude,
            changed_pixel_ratio,
            regions,
            motion_detected,
        })
    }

    /// Compute per-region motion statistics.
    #[allow(clippy::too_many_arguments)]
    fn compute_regions(
        &self,
        frame: &[u8],
        prev: &[u8],
        width: u32,
        height: u32,
        threshold: u8,
        total_pixels: u32,
    ) -> Vec<MotionRegion> {
        let rx = self.region_count_x;
        let ry = self.region_count_y;

        // We index pixels by byte position (works for any channel count).
        let bytes_per_row = frame.len() / height.max(1) as usize;

        let mut regions = Vec::with_capacity((rx * ry) as usize);

        for ry_idx in 0..ry {
            for rx_idx in 0..rx {
                let region_x = rx_idx * width / rx;
                let region_y = ry_idx * height / ry;
                let region_w = (rx_idx + 1) * width / rx - region_x;
                let region_h = (ry_idx + 1) * height / ry - region_y;

                let mut changed = 0u32;
                let mut diff_sum = 0u64;
                let mut region_pixels = 0u32;

                for row in region_y..(region_y + region_h) {
                    let row_start = row as usize * bytes_per_row;
                    // Byte range within row for this region's columns.
                    let col_byte_start =
                        row_start + (region_x as usize * bytes_per_row / width.max(1) as usize);
                    let col_bytes = region_w as usize * bytes_per_row / width.max(1) as usize;

                    let end = (col_byte_start + col_bytes).min(frame.len());
                    for i in col_byte_start..end {
                        let diff = frame[i].abs_diff(prev[i]);
                        region_pixels += 1;
                        if diff >= threshold {
                            changed += 1;
                            diff_sum += u64::from(diff);
                        }
                    }
                }

                let magnitude = if total_pixels > 0 && region_pixels > 0 {
                    (diff_sum as f64 / (f64::from(region_pixels) * 255.0)) as f32
                } else {
                    0.0
                };

                regions.push(MotionRegion {
                    x: region_x,
                    y: region_y,
                    width: region_w,
                    height: region_h,
                    magnitude,
                    changed_pixels: changed,
                });
            }
        }

        regions
    }

    /// Reset the motion detector, clearing the stored previous frame.
    pub fn reset(&mut self) {
        self.prev_frame = None;
    }

    /// Return the total number of motion analysis regions (x * y).
    #[must_use]
    pub fn region_count(&self) -> u32 {
        self.region_count_x * self.region_count_y
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensitivity_thresholds() {
        assert_eq!(Sensitivity::Low.threshold(), 30);
        assert_eq!(Sensitivity::Medium.threshold(), 15);
        assert_eq!(Sensitivity::High.threshold(), 8);
    }

    #[test]
    fn test_first_frame_returns_no_motion() {
        let mut detector = MotionDetector::new(Sensitivity::Medium, 2, 2);
        let frame = vec![100u8; 16]; // 4x4 grayscale
        let result = detector
            .analyze(&frame, 4, 4)
            .expect("motion analysis should succeed");

        assert_eq!(result.global_magnitude, 0.0);
        assert_eq!(result.changed_pixel_ratio, 0.0);
        assert!(!result.motion_detected);
        assert!(result.regions.is_empty());
    }

    #[test]
    fn test_identical_frames_returns_no_motion() {
        let mut detector = MotionDetector::new(Sensitivity::Medium, 2, 2);
        let frame = vec![100u8; 16];

        // First call stores the frame.
        detector
            .analyze(&frame, 4, 4)
            .expect("motion analysis should succeed");

        // Second call with identical frame.
        let result = detector
            .analyze(&frame, 4, 4)
            .expect("motion analysis should succeed");
        assert_eq!(result.global_magnitude, 0.0);
        assert_eq!(result.changed_pixel_ratio, 0.0);
        assert!(!result.motion_detected);
    }

    #[test]
    fn test_different_frames_returns_motion() {
        let mut detector = MotionDetector::new(Sensitivity::High, 1, 1);
        let frame_a = vec![0u8; 16];
        let frame_b = vec![255u8; 16];

        detector
            .analyze(&frame_a, 4, 4)
            .expect("motion analysis should succeed");
        let result = detector
            .analyze(&frame_b, 4, 4)
            .expect("motion analysis should succeed");

        assert!(result.motion_detected);
        assert!(result.global_magnitude > 0.0);
        assert!(result.changed_pixel_ratio > 0.0);
        assert_eq!(result.changed_pixel_ratio, 1.0);
    }

    #[test]
    fn test_region_count() {
        let detector = MotionDetector::new(Sensitivity::Low, 4, 3);
        assert_eq!(detector.region_count(), 12);
    }

    #[test]
    fn test_reset_clears_previous_frame() {
        let mut detector = MotionDetector::new(Sensitivity::Medium, 1, 1);
        let frame_a = vec![0u8; 16];
        let frame_b = vec![255u8; 16];

        detector
            .analyze(&frame_a, 4, 4)
            .expect("motion analysis should succeed");
        detector.reset();

        // After reset, the next call should behave as the first call.
        let result = detector
            .analyze(&frame_b, 4, 4)
            .expect("motion analysis should succeed");
        assert!(!result.motion_detected);
    }

    #[test]
    fn test_below_threshold_not_detected() {
        // Low sensitivity threshold = 30; diff of 10 should not trigger motion.
        let mut detector = MotionDetector::new(Sensitivity::Low, 1, 1);
        let frame_a = vec![100u8; 16];
        let frame_b = vec![110u8; 16]; // diff = 10, below 30

        detector
            .analyze(&frame_a, 4, 4)
            .expect("motion analysis should succeed");
        let result = detector
            .analyze(&frame_b, 4, 4)
            .expect("motion analysis should succeed");

        assert!(!result.motion_detected);
    }

    #[test]
    fn test_invalid_dimensions() {
        let mut detector = MotionDetector::new(Sensitivity::Medium, 1, 1);
        let frame = vec![0u8; 16];
        assert!(detector.analyze(&frame, 0, 4).is_err());
        assert!(detector.analyze(&frame, 4, 0).is_err());
    }
}
