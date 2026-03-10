//! Motion estimation using block matching.
//!
//! Block matching divides frames into blocks and searches for the best
//! matching block in a reference frame, producing motion vectors.

use crate::{DenoiseError, DenoiseResult};
use oximedia_codec::VideoFrame;
use rayon::prelude::*;

/// Motion estimator configuration and state.
pub struct MotionEstimator {
    /// Block size for motion estimation (typically 8 or 16).
    pub block_size: usize,
    /// Search range in pixels (typically 8-32).
    pub search_range: i32,
    /// Matching criterion threshold.
    pub threshold: u32,
}

impl MotionEstimator {
    /// Create a new motion estimator.
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size,
            search_range: 16,
            threshold: 1000,
        }
    }

    /// Estimate motion between current and reference frames.
    ///
    /// Returns a vector of motion vectors (dx, dy) for each block.
    pub fn estimate(
        &self,
        current: &VideoFrame,
        reference: &VideoFrame,
    ) -> DenoiseResult<Vec<(i16, i16)>> {
        if current.planes.is_empty() || reference.planes.is_empty() {
            return Err(DenoiseError::MotionEstimationError(
                "Frame has no planes".to_string(),
            ));
        }

        // Use luma plane for motion estimation
        let current_plane = &current.planes[0];
        let reference_plane = &reference.planes[0];
        let (width, height) = current.plane_dimensions(0);

        let num_blocks_x = (width as usize).div_ceil(self.block_size);
        let num_blocks_y = (height as usize).div_ceil(self.block_size);
        let num_blocks = num_blocks_x * num_blocks_y;

        // Estimate motion for each block in parallel
        let motion_vectors: Vec<(i16, i16)> = (0..num_blocks)
            .into_par_iter()
            .map(|block_idx| {
                let bx = (block_idx % num_blocks_x) * self.block_size;
                let by = (block_idx / num_blocks_x) * self.block_size;

                self.estimate_block_motion(
                    current_plane.data.as_ref(),
                    reference_plane.data.as_ref(),
                    width as usize,
                    height as usize,
                    current_plane.stride,
                    reference_plane.stride,
                    bx,
                    by,
                )
            })
            .collect();

        Ok(motion_vectors)
    }

    /// Estimate motion for a single block using full search.
    #[allow(clippy::too_many_arguments)]
    fn estimate_block_motion(
        &self,
        current: &[u8],
        reference: &[u8],
        width: usize,
        height: usize,
        current_stride: usize,
        reference_stride: usize,
        block_x: usize,
        block_y: usize,
    ) -> (i16, i16) {
        let mut best_mv = (0i16, 0i16);
        let mut best_sad = u32::MAX;

        // Full search within search range
        for dy in -self.search_range..=self.search_range {
            for dx in -self.search_range..=self.search_range {
                let ref_x = block_x as i32 + dx;
                let ref_y = block_y as i32 + dy;

                // Check bounds
                if ref_x < 0
                    || ref_y < 0
                    || ref_x + self.block_size as i32 > width as i32
                    || ref_y + self.block_size as i32 > height as i32
                {
                    continue;
                }

                // Compute SAD (Sum of Absolute Differences)
                let sad = self.compute_sad(
                    current,
                    reference,
                    current_stride,
                    reference_stride,
                    block_x,
                    block_y,
                    ref_x as usize,
                    ref_y as usize,
                );

                if sad < best_sad {
                    best_sad = sad;
                    best_mv = (dx as i16, dy as i16);

                    // Early termination if very good match
                    if sad < self.threshold {
                        break;
                    }
                }
            }

            if best_sad < self.threshold {
                break;
            }
        }

        best_mv
    }

    /// Compute Sum of Absolute Differences between two blocks.
    #[allow(clippy::too_many_arguments)]
    fn compute_sad(
        &self,
        current: &[u8],
        reference: &[u8],
        current_stride: usize,
        reference_stride: usize,
        current_x: usize,
        current_y: usize,
        reference_x: usize,
        reference_y: usize,
    ) -> u32 {
        let mut sad = 0u32;

        for y in 0..self.block_size {
            for x in 0..self.block_size {
                let curr_idx = (current_y + y) * current_stride + current_x + x;
                let ref_idx = (reference_y + y) * reference_stride + reference_x + x;

                let diff = (i32::from(current[curr_idx]) - i32::from(reference[ref_idx])).abs();
                sad += diff as u32;
            }
        }

        sad
    }
}

/// Fast motion estimation using diamond search.
pub fn diamond_search(
    current: &VideoFrame,
    reference: &VideoFrame,
    block_size: usize,
) -> DenoiseResult<Vec<(i16, i16)>> {
    let estimator = MotionEstimator {
        block_size,
        search_range: 16,
        threshold: 1000,
    };

    // Diamond search pattern (simplified - uses full search for now)
    estimator.estimate(current, reference)
}

/// Hierarchical motion estimation using multi-resolution.
pub fn hierarchical_motion_estimation(
    current: &VideoFrame,
    reference: &VideoFrame,
    block_size: usize,
) -> DenoiseResult<Vec<(i16, i16)>> {
    let estimator = MotionEstimator {
        block_size,
        search_range: 32,
        threshold: 500,
    };

    // Simplified hierarchical search
    estimator.estimate(current, reference)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    #[test]
    fn test_motion_estimator_creation() {
        let estimator = MotionEstimator::new(16);
        assert_eq!(estimator.block_size, 16);
        assert_eq!(estimator.search_range, 16);
    }

    #[test]
    fn test_motion_estimation() {
        let mut current = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        current.allocate();

        let mut reference = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        reference.allocate();

        let estimator = MotionEstimator::new(16);
        let result = estimator.estimate(&current, &reference);

        assert!(result.is_ok());
        let motion_vectors = result.expect("motion_vectors should be valid");
        assert!(!motion_vectors.is_empty());
    }

    #[test]
    fn test_diamond_search() {
        let mut current = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        current.allocate();

        let mut reference = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        reference.allocate();

        let result = diamond_search(&current, &reference, 8);
        assert!(result.is_ok());
    }

    #[test]
    fn test_hierarchical_motion_estimation() {
        let mut current = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        current.allocate();

        let mut reference = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        reference.allocate();

        let result = hierarchical_motion_estimation(&current, &reference, 16);
        assert!(result.is_ok());
    }
}
