//! Frame warping application.

use crate::error::{StabilizeError, StabilizeResult};
use crate::transform::calculate::StabilizationTransform;
use crate::warp::boundary::BoundaryMode;
use crate::warp::interpolation::InterpolationMethod;
use crate::Frame;
use scirs2_core::ndarray::Array2;

/// Frame warper that applies stabilization transforms to frames.
#[derive(Debug)]
pub struct FrameWarper {
    interpolation: InterpolationMethod,
    boundary: BoundaryMode,
}

impl FrameWarper {
    /// Create a new frame warper.
    #[must_use]
    pub fn new() -> Self {
        Self {
            interpolation: InterpolationMethod::Bilinear,
            boundary: BoundaryMode::Replicate,
        }
    }

    /// Set interpolation method.
    pub fn set_interpolation(&mut self, method: InterpolationMethod) {
        self.interpolation = method;
    }

    /// Set boundary mode.
    pub fn set_boundary(&mut self, mode: BoundaryMode) {
        self.boundary = mode;
    }

    /// Warp a sequence of frames.
    ///
    /// # Errors
    ///
    /// Returns an error if frames and transforms lengths don't match.
    pub fn warp(
        &self,
        frames: &[Frame],
        transforms: &[StabilizationTransform],
    ) -> StabilizeResult<Vec<Frame>> {
        if frames.len() != transforms.len() {
            return Err(StabilizeError::dimension_mismatch(
                format!("{}", frames.len()),
                format!("{}", transforms.len()),
            ));
        }

        frames
            .iter()
            .zip(transforms.iter())
            .map(|(frame, transform)| self.warp_frame(frame, transform))
            .collect()
    }

    /// Warp a single frame.
    fn warp_frame(
        &self,
        frame: &Frame,
        transform: &StabilizationTransform,
    ) -> StabilizeResult<Frame> {
        let mut warped_data = Array2::zeros((frame.height, frame.width));

        for y in 0..frame.height {
            for x in 0..frame.width {
                let (src_x, src_y) = self.inverse_transform(x as f64, y as f64, transform);

                let pixel_value =
                    self.interpolation
                        .interpolate(&frame.data, src_x, src_y, self.boundary);

                warped_data[[y, x]] = pixel_value;
            }
        }

        Ok(Frame {
            width: frame.width,
            height: frame.height,
            timestamp: frame.timestamp,
            data: warped_data,
        })
    }

    /// Compute inverse transform to find source pixel.
    fn inverse_transform(&self, x: f64, y: f64, transform: &StabilizationTransform) -> (f64, f64) {
        let cos_a = transform.angle.cos();
        let sin_a = transform.angle.sin();
        let s = transform.scale;

        // Center coordinates
        let cx = x - transform.dx;
        let cy = y - transform.dy;

        // Inverse rotation and scale
        let src_x = (cos_a * cx + sin_a * cy) / s;
        let src_y = (-sin_a * cx + cos_a * cy) / s;

        (src_x, src_y)
    }
}

impl Default for FrameWarper {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warper_creation() {
        let warper = FrameWarper::new();
        assert!(matches!(
            warper.interpolation,
            InterpolationMethod::Bilinear
        ));
    }
}
