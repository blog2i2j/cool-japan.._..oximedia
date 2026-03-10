//! Video super-resolution: VideoFrame, TemporalFilter, MotionEstimator, VideoSuperResolution.

use super::model::SuperResolutionModel;
use crate::error::CvResult;
use std::collections::VecDeque;

/// Frame data for video processing.
#[derive(Clone)]
pub struct VideoFrame {
    /// Frame data in RGB format.
    pub data: Vec<u8>,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Frame timestamp (optional).
    pub timestamp: Option<f64>,
}

impl VideoFrame {
    /// Create a new video frame.
    #[must_use]
    pub fn new(data: Vec<u8>, width: u32, height: u32) -> Self {
        Self {
            data,
            width,
            height,
            timestamp: None,
        }
    }

    /// Create a new video frame with timestamp.
    #[must_use]
    pub fn with_timestamp(data: Vec<u8>, width: u32, height: u32, timestamp: f64) -> Self {
        Self {
            data,
            width,
            height,
            timestamp: Some(timestamp),
        }
    }

    /// Get the frame size in pixels.
    #[must_use]
    pub const fn pixel_count(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }
}

/// Temporal consistency filter for video.
///
/// Applies temporal filtering to reduce flickering and maintain consistency
/// across frames while preserving motion.
pub struct TemporalFilter {
    /// Temporal weight (0.0 = no filtering, 1.0 = maximum filtering).
    pub(super) temporal_weight: f32,
    /// Previous frame data.
    previous_frame: Option<Vec<u8>>,
    /// Motion threshold for adaptive filtering.
    pub(super) motion_threshold: f32,
}

impl TemporalFilter {
    /// Create a new temporal filter.
    ///
    /// # Arguments
    ///
    /// * `temporal_weight` - Temporal filtering strength (0.0 - 1.0)
    /// * `motion_threshold` - Motion detection threshold (higher = less sensitive)
    #[must_use]
    pub fn new(temporal_weight: f32, motion_threshold: f32) -> Self {
        Self {
            temporal_weight: temporal_weight.clamp(0.0, 1.0),
            previous_frame: None,
            motion_threshold,
        }
    }

    /// Apply temporal filtering to a frame.
    ///
    /// # Arguments
    ///
    /// * `current` - Current frame data
    /// * `width` - Frame width
    /// * `height` - Frame height
    ///
    /// # Returns
    ///
    /// Filtered frame data
    pub fn filter(&mut self, current: &[u8], width: u32, height: u32) -> Vec<u8> {
        let size = (width * height * 3) as usize;

        if current.len() != size {
            return current.to_vec();
        }

        let filtered = if let Some(ref prev) = self.previous_frame {
            if prev.len() == size {
                self.apply_temporal_blend(current, prev, width, height)
            } else {
                current.to_vec()
            }
        } else {
            current.to_vec()
        };

        self.previous_frame = Some(filtered.clone());
        filtered
    }

    /// Apply temporal blending between current and previous frames.
    fn apply_temporal_blend(
        &self,
        current: &[u8],
        previous: &[u8],
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let mut output = vec![0u8; current.len()];
        let w = width as usize;
        let h = height as usize;

        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;

                // Calculate motion magnitude
                let mut motion = 0.0f32;
                for c in 0..3 {
                    let diff = current[idx + c] as f32 - previous[idx + c] as f32;
                    motion += diff * diff;
                }
                motion = motion.sqrt();

                // Adaptive temporal weight based on motion
                let adaptive_weight = if motion > self.motion_threshold {
                    self.temporal_weight * (self.motion_threshold / motion).min(1.0)
                } else {
                    self.temporal_weight
                };

                // Blend current and previous frames
                for c in 0..3 {
                    let curr_val = current[idx + c] as f32;
                    let prev_val = previous[idx + c] as f32;
                    let blended = curr_val * (1.0 - adaptive_weight) + prev_val * adaptive_weight;
                    output[idx + c] = blended.round() as u8;
                }
            }
        }

        output
    }

    /// Reset the filter state.
    pub fn reset(&mut self) {
        self.previous_frame = None;
    }

    /// Check if the filter has previous frame data.
    #[must_use]
    pub fn has_previous(&self) -> bool {
        self.previous_frame.is_some()
    }
}

impl Default for TemporalFilter {
    fn default() -> Self {
        Self::new(0.3, 10.0)
    }
}

/// Motion estimator for video frames.
///
/// Estimates motion between frames for motion-aware processing.
pub struct MotionEstimator {
    /// Block size for motion estimation.
    pub(super) block_size: u32,
    /// Search range for motion vectors.
    pub(super) search_range: i32,
}

impl MotionEstimator {
    /// Create a new motion estimator.
    ///
    /// # Arguments
    ///
    /// * `block_size` - Size of blocks for motion estimation
    /// * `search_range` - Maximum search distance for motion vectors
    #[must_use]
    pub fn new(block_size: u32, search_range: i32) -> Self {
        Self {
            block_size,
            search_range,
        }
    }

    /// Estimate motion between two frames.
    ///
    /// # Arguments
    ///
    /// * `current` - Current frame
    /// * `reference` - Reference frame
    /// * `width` - Frame width
    /// * `height` - Frame height
    ///
    /// # Returns
    ///
    /// Motion vectors as (dx, dy) pairs for each block
    pub fn estimate(
        &self,
        current: &[u8],
        reference: &[u8],
        width: u32,
        height: u32,
    ) -> Vec<(i32, i32)> {
        let blocks_x = width.div_ceil(self.block_size);
        let blocks_y = height.div_ceil(self.block_size);
        let mut motion_vectors = Vec::with_capacity((blocks_x * blocks_y) as usize);

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let block_x = bx * self.block_size;
                let block_y = by * self.block_size;

                let (dx, dy) =
                    self.estimate_block_motion(current, reference, width, height, block_x, block_y);

                motion_vectors.push((dx, dy));
            }
        }

        motion_vectors
    }

    /// Estimate motion for a single block using block matching.
    #[allow(clippy::too_many_arguments)]
    fn estimate_block_motion(
        &self,
        current: &[u8],
        reference: &[u8],
        width: u32,
        height: u32,
        block_x: u32,
        block_y: u32,
    ) -> (i32, i32) {
        let mut best_dx = 0;
        let mut best_dy = 0;
        let mut best_sad = f32::MAX;

        for dy in -self.search_range..=self.search_range {
            for dx in -self.search_range..=self.search_range {
                let ref_x = block_x as i32 + dx;
                let ref_y = block_y as i32 + dy;

                if ref_x < 0
                    || ref_y < 0
                    || ref_x + self.block_size as i32 > width as i32
                    || ref_y + self.block_size as i32 > height as i32
                {
                    continue;
                }

                let sad = self.calculate_sad(
                    current,
                    reference,
                    width,
                    block_x,
                    block_y,
                    ref_x as u32,
                    ref_y as u32,
                );

                if sad < best_sad {
                    best_sad = sad;
                    best_dx = dx;
                    best_dy = dy;
                }
            }
        }

        (best_dx, best_dy)
    }

    /// Calculate Sum of Absolute Differences (SAD) between two blocks.
    #[allow(clippy::too_many_arguments)]
    fn calculate_sad(
        &self,
        current: &[u8],
        reference: &[u8],
        width: u32,
        curr_x: u32,
        curr_y: u32,
        ref_x: u32,
        ref_y: u32,
    ) -> f32 {
        let mut sad = 0.0f32;
        let w = width as usize;

        for y in 0..self.block_size {
            for x in 0..self.block_size {
                let curr_idx = ((curr_y + y) as usize * w + (curr_x + x) as usize) * 3;
                let ref_idx = ((ref_y + y) as usize * w + (ref_x + x) as usize) * 3;

                for c in 0..3 {
                    sad += (current[curr_idx + c] as f32 - reference[ref_idx + c] as f32).abs();
                }
            }
        }

        sad
    }

    /// Calculate average motion magnitude from motion vectors.
    #[must_use]
    pub fn calculate_average_motion(motion_vectors: &[(i32, i32)]) -> f32 {
        if motion_vectors.is_empty() {
            return 0.0;
        }

        let sum: f32 = motion_vectors
            .iter()
            .map(|(dx, dy)| ((*dx * *dx + *dy * *dy) as f32).sqrt())
            .sum();

        sum / motion_vectors.len() as f32
    }
}

impl Default for MotionEstimator {
    fn default() -> Self {
        Self::new(16, 8)
    }
}

/// Video super-resolution processor.
///
/// Handles video-specific super-resolution with temporal consistency,
/// frame buffering, and motion-aware processing.
pub struct VideoSuperResolution {
    /// Underlying super-resolution model.
    model: SuperResolutionModel,
    /// Frame buffer for temporal processing.
    frame_buffer: VecDeque<VideoFrame>,
    /// Temporal filter for consistency.
    temporal_filter: TemporalFilter,
    /// Motion estimator.
    motion_estimator: MotionEstimator,
    /// Buffer size (number of frames to keep).
    buffer_size: usize,
    /// Enable temporal filtering.
    enable_temporal_filtering: bool,
    /// Enable motion-aware processing.
    enable_motion_aware: bool,
}

impl VideoSuperResolution {
    /// Create a new video super-resolution processor.
    ///
    /// # Arguments
    ///
    /// * `model` - Super-resolution model to use
    /// * `buffer_size` - Number of frames to buffer for temporal processing
    pub fn new(model: SuperResolutionModel, buffer_size: usize) -> Self {
        Self {
            model,
            frame_buffer: VecDeque::with_capacity(buffer_size),
            temporal_filter: TemporalFilter::default(),
            motion_estimator: MotionEstimator::default(),
            buffer_size,
            enable_temporal_filtering: true,
            enable_motion_aware: true,
        }
    }

    /// Create a video processor with custom settings.
    ///
    /// # Arguments
    ///
    /// * `model` - Super-resolution model
    /// * `buffer_size` - Frame buffer size
    /// * `temporal_weight` - Temporal filtering strength (0.0 - 1.0)
    /// * `motion_threshold` - Motion detection threshold
    pub fn with_settings(
        model: SuperResolutionModel,
        buffer_size: usize,
        temporal_weight: f32,
        motion_threshold: f32,
    ) -> Self {
        Self {
            model,
            frame_buffer: VecDeque::with_capacity(buffer_size),
            temporal_filter: TemporalFilter::new(temporal_weight, motion_threshold),
            motion_estimator: MotionEstimator::default(),
            buffer_size,
            enable_temporal_filtering: true,
            enable_motion_aware: true,
        }
    }

    /// Enable or disable temporal filtering.
    pub fn set_temporal_filtering(&mut self, enable: bool) {
        self.enable_temporal_filtering = enable;
    }

    /// Enable or disable motion-aware processing.
    pub fn set_motion_aware(&mut self, enable: bool) {
        self.enable_motion_aware = enable;
    }

    /// Set temporal filter parameters.
    pub fn set_temporal_params(&mut self, weight: f32, motion_threshold: f32) {
        self.temporal_filter = TemporalFilter::new(weight, motion_threshold);
    }

    /// Process a single video frame.
    ///
    /// # Arguments
    ///
    /// * `frame` - Input frame
    ///
    /// # Returns
    ///
    /// Upscaled frame
    ///
    /// # Errors
    ///
    /// Returns an error if processing fails.
    pub fn process_frame(&mut self, mut frame: VideoFrame) -> CvResult<VideoFrame> {
        // Add frame to buffer
        if self.frame_buffer.len() >= self.buffer_size {
            self.frame_buffer.pop_front();
        }

        // Apply temporal filtering if enabled
        if self.enable_temporal_filtering && self.temporal_filter.has_previous() {
            frame.data = self
                .temporal_filter
                .filter(&frame.data, frame.width, frame.height);
        }

        // Analyze motion if enabled
        let motion_magnitude = if self.enable_motion_aware && !self.frame_buffer.is_empty() {
            let prev_frame = &self.frame_buffer[self.frame_buffer.len() - 1];
            let motion_vectors = self.motion_estimator.estimate(
                &frame.data,
                &prev_frame.data,
                frame.width,
                frame.height,
            );
            MotionEstimator::calculate_average_motion(&motion_vectors)
        } else {
            0.0
        };

        // Adjust processing based on motion
        if self.enable_motion_aware && motion_magnitude > 5.0 {
            // High motion: reduce temporal filtering
            let original_weight = self.temporal_filter.temporal_weight;
            self.temporal_filter.temporal_weight = (original_weight * 0.5).max(0.1);

            // Process frame
            let upscaled_data = self.model.upscale(&frame.data, frame.width, frame.height)?;

            // Restore original weight
            self.temporal_filter.temporal_weight = original_weight;

            let scale = self.model.scale_factor().scale();
            let output_frame = VideoFrame {
                data: upscaled_data,
                width: frame.width * scale,
                height: frame.height * scale,
                timestamp: frame.timestamp,
            };

            self.frame_buffer.push_back(frame);
            Ok(output_frame)
        } else {
            // Normal processing
            let upscaled_data = self.model.upscale(&frame.data, frame.width, frame.height)?;

            let scale = self.model.scale_factor().scale();
            let output_frame = VideoFrame {
                data: upscaled_data,
                width: frame.width * scale,
                height: frame.height * scale,
                timestamp: frame.timestamp,
            };

            self.frame_buffer.push_back(frame);
            Ok(output_frame)
        }
    }

    /// Process multiple video frames in sequence.
    ///
    /// # Arguments
    ///
    /// * `frames` - Input frames
    ///
    /// # Returns
    ///
    /// Vector of upscaled frames
    ///
    /// # Errors
    ///
    /// Returns an error if processing fails.
    pub fn process_frames(&mut self, frames: Vec<VideoFrame>) -> CvResult<Vec<VideoFrame>> {
        let mut output_frames = Vec::with_capacity(frames.len());

        for frame in frames {
            let output = self.process_frame(frame)?;
            output_frames.push(output);
        }

        Ok(output_frames)
    }

    /// Reset the video processor state.
    pub fn reset(&mut self) {
        self.frame_buffer.clear();
        self.temporal_filter.reset();
    }

    /// Get the current buffer size.
    #[must_use]
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Get the number of buffered frames.
    #[must_use]
    pub fn buffered_frames(&self) -> usize {
        self.frame_buffer.len()
    }

    /// Get a reference to the underlying model.
    #[must_use]
    pub const fn model(&self) -> &SuperResolutionModel {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_frame() {
        let data = vec![0u8; 64 * 64 * 3];
        let frame = VideoFrame::new(data.clone(), 64, 64);
        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 64);
        assert_eq!(frame.pixel_count(), 64 * 64);
        assert!(frame.timestamp.is_none());

        let frame_with_ts = VideoFrame::with_timestamp(data, 64, 64, 1.5);
        assert_eq!(frame_with_ts.timestamp, Some(1.5));
    }

    #[test]
    fn test_temporal_filter() {
        let mut filter = TemporalFilter::new(0.5, 10.0);
        assert!(!filter.has_previous());

        let frame1 = vec![100u8; 64 * 64 * 3];
        let frame2 = vec![110u8; 64 * 64 * 3];

        let filtered1 = filter.filter(&frame1, 64, 64);
        assert_eq!(filtered1.len(), frame1.len());
        assert!(filter.has_previous());

        let filtered2 = filter.filter(&frame2, 64, 64);
        assert_eq!(filtered2.len(), frame2.len());

        // Filtered value should be between original frame values
        assert!(filtered2[0] >= 100 && filtered2[0] <= 110);

        filter.reset();
        assert!(!filter.has_previous());
    }

    #[test]
    fn test_motion_estimator() {
        let estimator = MotionEstimator::new(16, 4);
        let frame1 = vec![50u8; 128 * 128 * 3];
        let frame2 = vec![60u8; 128 * 128 * 3];

        let motion_vectors = estimator.estimate(&frame1, &frame2, 128, 128);
        assert!(!motion_vectors.is_empty());

        let avg_motion = MotionEstimator::calculate_average_motion(&motion_vectors);
        assert!(avg_motion >= 0.0);
    }

    #[test]
    fn test_motion_estimator_default() {
        let estimator = MotionEstimator::default();
        assert_eq!(estimator.block_size, 16);
        assert_eq!(estimator.search_range, 8);
    }

    #[test]
    fn test_temporal_filter_default() {
        let filter = TemporalFilter::default();
        assert_eq!(filter.temporal_weight, 0.3);
        assert_eq!(filter.motion_threshold, 10.0);
    }
}
