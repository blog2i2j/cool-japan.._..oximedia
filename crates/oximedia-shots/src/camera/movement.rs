//! Camera movement detection and analysis.

use crate::error::{ShotError, ShotResult};
use crate::frame_buffer::{FrameBuffer, GrayImage};
use crate::types::{CameraMovement, MovementType};

/// Camera movement detector.
pub struct MovementDetector {
    /// Threshold for movement detection.
    threshold: f32,
    /// Minimum duration for valid movement (frames).
    min_duration: usize,
}

impl MovementDetector {
    /// Create a new movement detector.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            threshold: 0.1,
            min_duration: 5,
        }
    }

    /// Detect camera movements in a sequence of frames.
    ///
    /// # Errors
    ///
    /// Returns error if frames are invalid or have mismatched dimensions.
    pub fn detect_movements(&self, frames: &[FrameBuffer]) -> ShotResult<Vec<CameraMovement>> {
        if frames.len() < 2 {
            return Ok(Vec::new());
        }

        let mut movements = Vec::new();

        // Calculate optical flow between consecutive frames
        let mut flow_vectors = Vec::new();
        for i in 1..frames.len() {
            let flow = self.calculate_optical_flow(&frames[i - 1], &frames[i])?;
            flow_vectors.push(flow);
        }

        // Analyze flow patterns to detect movements
        let mut i = 0;
        while i < flow_vectors.len() {
            if let Some((movement_type, duration, speed)) =
                self.analyze_flow_pattern(&flow_vectors[i..])
            {
                let start = i as f64 / 30.0; // Assuming 30 fps
                let end = start + (duration as f64 / 30.0);

                movements.push(CameraMovement {
                    movement_type,
                    start,
                    end,
                    confidence: 0.8,
                    speed,
                });

                i += duration;
            } else {
                i += 1;
            }
        }

        Ok(movements)
    }

    /// Calculate optical flow between two frames (simplified Lucas-Kanade).
    fn calculate_optical_flow(
        &self,
        frame1: &FrameBuffer,
        frame2: &FrameBuffer,
    ) -> ShotResult<(f32, f32)> {
        if frame1.dim() != frame2.dim() {
            return Err(ShotError::InvalidFrame(
                "Frame dimensions do not match".to_string(),
            ));
        }

        let shape = frame1.dim();
        let gray1 = self.to_grayscale(frame1);
        let gray2 = self.to_grayscale(frame2);

        let mut dx_sum = 0.0;
        let mut dy_sum = 0.0;
        let mut count = 0;

        // Sample grid points
        for y in (5..shape.0.saturating_sub(5)).step_by(10) {
            for x in (5..shape.1.saturating_sub(5)).step_by(10) {
                if let Some((dx, dy)) = self.compute_local_flow(&gray1, &gray2, x, y) {
                    dx_sum += dx;
                    dy_sum += dy;
                    count += 1;
                }
            }
        }

        if count == 0 {
            return Ok((0.0, 0.0));
        }

        Ok((dx_sum / count as f32, dy_sum / count as f32))
    }

    /// Compute local optical flow at a point.
    fn compute_local_flow(
        &self,
        gray1: &GrayImage,
        gray2: &GrayImage,
        x: usize,
        y: usize,
    ) -> Option<(f32, f32)> {
        let window_size = 5;
        let shape = gray1.dim();

        if y < window_size
            || y >= shape.0 - window_size
            || x < window_size
            || x >= shape.1 - window_size
        {
            return None;
        }

        // Compute image gradients
        let ix = (f32::from(gray1.get(y, x + 1)) - f32::from(gray1.get(y, x - 1))) / 2.0;
        let iy = (f32::from(gray1.get(y + 1, x)) - f32::from(gray1.get(y - 1, x))) / 2.0;
        let it = f32::from(gray2.get(y, x)) - f32::from(gray1.get(y, x));

        // Solve for flow using least squares
        let denom = ix * ix + iy * iy;
        if denom < 1.0 {
            return None;
        }

        let vx = -(ix * it) / denom;
        let vy = -(iy * it) / denom;

        Some((vx, vy))
    }

    /// Analyze flow pattern to determine movement type.
    fn analyze_flow_pattern(&self, flows: &[(f32, f32)]) -> Option<(MovementType, usize, f32)> {
        if flows.len() < self.min_duration {
            return None;
        }

        // Calculate average flow
        let mut avg_dx = 0.0;
        let mut avg_dy = 0.0;

        for (dx, dy) in flows.iter().take(self.min_duration) {
            avg_dx += dx;
            avg_dy += dy;
        }

        avg_dx /= self.min_duration as f32;
        avg_dy /= self.min_duration as f32;

        let magnitude = (avg_dx * avg_dx + avg_dy * avg_dy).sqrt();

        if magnitude < self.threshold {
            return Some((MovementType::Static, self.min_duration, 0.0));
        }

        // Determine movement type based on flow direction
        let movement_type = if avg_dx.abs() > avg_dy.abs() * 2.0 {
            if avg_dx > 0.0 {
                MovementType::PanRight
            } else {
                MovementType::PanLeft
            }
        } else if avg_dy.abs() > avg_dx.abs() * 2.0 {
            if avg_dy > 0.0 {
                MovementType::TiltDown
            } else {
                MovementType::TiltUp
            }
        } else {
            // Check for zoom or dolly
            if self.is_zoom_pattern(flows) {
                if magnitude > 0.0 {
                    MovementType::ZoomIn
                } else {
                    MovementType::ZoomOut
                }
            } else {
                MovementType::Handheld
            }
        };

        Some((movement_type, self.min_duration, magnitude))
    }

    /// Check if flow pattern indicates zoom.
    fn is_zoom_pattern(&self, flows: &[(f32, f32)]) -> bool {
        if flows.len() < 3 {
            return false;
        }

        // Zoom creates radial flow from center
        let center_x = 0.0; // Assuming normalized coordinates
        let center_y: f32 = 0.0;

        let mut radial_consistency = 0.0;

        for (dx, dy) in flows.iter().take(self.min_duration.min(flows.len())) {
            let angle = dy.atan2(*dx);
            let expected_angle = center_y.atan2(center_x);
            let diff = (angle - expected_angle).abs();

            if diff < 0.5 {
                radial_consistency += 1.0;
            }
        }

        radial_consistency / self.min_duration as f32 > 0.6
    }

    /// Convert RGB to grayscale.
    fn to_grayscale(&self, frame: &FrameBuffer) -> GrayImage {
        let shape = frame.dim();
        let mut gray = GrayImage::zeros(shape.0, shape.1);

        for y in 0..shape.0 {
            for x in 0..shape.1 {
                let r = f32::from(frame.get(y, x, 0));
                let g = f32::from(frame.get(y, x, 1));
                let b = f32::from(frame.get(y, x, 2));
                gray.set(y, x, ((r * 0.299) + (g * 0.587) + (b * 0.114)) as u8);
            }
        }

        gray
    }
}

impl Default for MovementDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_movement_detector_creation() {
        let detector = MovementDetector::new();
        assert!((detector.threshold - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_no_movement_single_frame() {
        let detector = MovementDetector::new();
        let frames = vec![FrameBuffer::zeros(100, 100, 3)];
        let result = detector.detect_movements(&frames);
        assert!(result.is_ok());
        if let Ok(movements) = result {
            assert!(movements.is_empty());
        }
    }

    #[test]
    fn test_static_frames() {
        let detector = MovementDetector::new();
        let frames = vec![FrameBuffer::zeros(100, 100, 3); 10];
        let result = detector.detect_movements(&frames);
        assert!(result.is_ok());
    }
}
