//! Optical flow estimation for frame interpolation.
//!
//! This module provides optical flow algorithms optimized for frame interpolation,
//! including bidirectional flow estimation and multi-scale pyramid processing.

use crate::error::{CvError, CvResult};
use crate::tracking;
use oximedia_codec::VideoFrame;

/// Optical flow computation method for interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FlowMethod {
    /// Block matching for fast computation.
    BlockMatching,
    /// Lucas-Kanade sparse optical flow.
    #[default]
    LucasKanade,
    /// Farneback dense optical flow.
    Farneback,
}

/// Flow field representing motion vectors.
#[derive(Debug, Clone)]
pub struct FlowField {
    /// Horizontal flow components.
    pub flow_x: Vec<f32>,
    /// Vertical flow components.
    pub flow_y: Vec<f32>,
    /// Field width.
    pub width: u32,
    /// Field height.
    pub height: u32,
}

impl FlowField {
    /// Create a new flow field.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let size = width as usize * height as usize;
        Self {
            flow_x: vec![0.0; size],
            flow_y: vec![0.0; size],
            width,
            height,
        }
    }

    /// Get flow vector at a specific position.
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> (f32, f32) {
        if x >= self.width || y >= self.height {
            return (0.0, 0.0);
        }

        let idx = (y * self.width + x) as usize;
        if idx < self.flow_x.len() {
            (self.flow_x[idx], self.flow_y[idx])
        } else {
            (0.0, 0.0)
        }
    }

    /// Set flow vector at a specific position.
    pub fn set(&mut self, x: u32, y: u32, dx: f32, dy: f32) {
        if x >= self.width || y >= self.height {
            return;
        }

        let idx = (y * self.width + x) as usize;
        if idx < self.flow_x.len() {
            self.flow_x[idx] = dx;
            self.flow_y[idx] = dy;
        }
    }

    /// Get flow magnitude at a position.
    #[must_use]
    pub fn magnitude(&self, x: u32, y: u32) -> f32 {
        let (dx, dy) = self.get(x, y);
        (dx * dx + dy * dy).sqrt()
    }

    /// Get average flow magnitude.
    #[must_use]
    pub fn average_magnitude(&self) -> f32 {
        if self.flow_x.is_empty() {
            return 0.0;
        }

        let mut sum = 0.0;
        for i in 0..self.flow_x.len() {
            sum += (self.flow_x[i] * self.flow_x[i] + self.flow_y[i] * self.flow_y[i]).sqrt();
        }
        sum / self.flow_x.len() as f32
    }

    /// Get maximum flow magnitude.
    #[must_use]
    pub fn max_magnitude(&self) -> f32 {
        let mut max_mag: f32 = 0.0;
        for i in 0..self.flow_x.len() {
            let mag = (self.flow_x[i] * self.flow_x[i] + self.flow_y[i] * self.flow_y[i]).sqrt();
            max_mag = max_mag.max(mag);
        }
        max_mag
    }

    /// Scale flow field by a factor.
    pub fn scale(&mut self, factor: f32) {
        for i in 0..self.flow_x.len() {
            self.flow_x[i] *= factor;
            self.flow_y[i] *= factor;
        }
    }

    /// Resize flow field to new dimensions.
    #[must_use]
    pub fn resize(&self, new_width: u32, new_height: u32) -> Self {
        let mut result = Self::new(new_width, new_height);

        let scale_x = new_width as f32 / self.width as f32;
        let scale_y = new_height as f32 / self.height as f32;

        for y in 0..new_height {
            for x in 0..new_width {
                let src_x = (x as f32 / scale_x) as u32;
                let src_y = (y as f32 / scale_y) as u32;

                let (dx, dy) = self.get(src_x.min(self.width - 1), src_y.min(self.height - 1));

                result.set(x, y, dx * scale_x, dy * scale_y);
            }
        }

        result
    }
}

/// Multi-scale pyramid for optical flow.
#[derive(Debug, Clone)]
pub struct FlowPyramid {
    /// Pyramid levels (from coarse to fine).
    pub levels: Vec<FlowField>,
}

impl FlowPyramid {
    /// Create a new flow pyramid.
    #[must_use]
    pub fn new() -> Self {
        Self { levels: Vec::new() }
    }

    /// Add a level to the pyramid.
    pub fn add_level(&mut self, flow: FlowField) {
        self.levels.push(flow);
    }

    /// Get the finest level.
    #[must_use]
    pub fn finest(&self) -> Option<&FlowField> {
        self.levels.last()
    }

    /// Get the coarsest level.
    #[must_use]
    pub fn coarsest(&self) -> Option<&FlowField> {
        self.levels.first()
    }
}

impl Default for FlowPyramid {
    fn default() -> Self {
        Self::new()
    }
}

/// Optical flow estimator for frame interpolation.
pub struct FlowEstimator {
    /// Flow computation method.
    method: FlowMethod,
    /// Use pyramid for multi-scale processing.
    use_pyramid: bool,
    /// Maximum pyramid levels.
    pyramid_levels: u32,
    /// Window size for local computation.
    window_size: u32,
    /// Block size for block matching.
    block_size: u32,
    /// Search range for block matching.
    search_range: i32,
}

impl FlowEstimator {
    /// Create a new flow estimator.
    #[must_use]
    pub fn new(method: FlowMethod) -> Self {
        Self {
            method,
            use_pyramid: true,
            pyramid_levels: 3,
            window_size: 21,
            block_size: 16,
            search_range: 16,
        }
    }

    /// Set whether to use pyramid processing.
    #[must_use]
    pub const fn with_pyramid_levels(mut self, levels: u32) -> Self {
        self.pyramid_levels = levels;
        self.use_pyramid = levels > 1;
        self
    }

    /// Set window size.
    #[must_use]
    pub const fn with_window_size(mut self, size: u32) -> Self {
        self.window_size = size;
        self
    }

    /// Set block size for block matching.
    #[must_use]
    pub const fn with_block_size(mut self, size: u32) -> Self {
        self.block_size = size;
        self
    }

    /// Set search range for block matching.
    #[must_use]
    pub const fn with_search_range(mut self, range: i32) -> Self {
        self.search_range = range;
        self
    }

    /// Estimate bidirectional optical flow between two frames.
    ///
    /// Returns (forward_flow, backward_flow) where:
    /// - forward_flow: motion from frame1 to frame2
    /// - backward_flow: motion from frame2 to frame1
    pub fn estimate_bidirectional(
        &self,
        frame1: &VideoFrame,
        frame2: &VideoFrame,
    ) -> CvResult<(FlowField, FlowField)> {
        // Extract grayscale data from frames
        let gray1 = self.extract_grayscale(frame1)?;
        let gray2 = self.extract_grayscale(frame2)?;

        // Estimate forward and backward flow
        let forward = self.estimate_flow(&gray1, &gray2, frame1.width, frame1.height)?;
        let backward = self.estimate_flow(&gray2, &gray1, frame2.width, frame2.height)?;

        Ok((forward, backward))
    }

    /// Estimate optical flow from one frame to another.
    fn estimate_flow(
        &self,
        prev: &[u8],
        curr: &[u8],
        width: u32,
        height: u32,
    ) -> CvResult<FlowField> {
        match self.method {
            FlowMethod::BlockMatching => self.estimate_block_matching(prev, curr, width, height),
            FlowMethod::LucasKanade => self.estimate_lucas_kanade(prev, curr, width, height),
            FlowMethod::Farneback => self.estimate_farneback(prev, curr, width, height),
        }
    }

    /// Block matching optical flow.
    fn estimate_block_matching(
        &self,
        prev: &[u8],
        curr: &[u8],
        width: u32,
        height: u32,
    ) -> CvResult<FlowField> {
        let mut flow = FlowField::new(width, height);
        let block_size = self.block_size as i32;
        let search_range = self.search_range;

        let wi = width as i32;
        let hi = height as i32;

        for y in (0..hi).step_by(block_size as usize) {
            for x in (0..wi).step_by(block_size as usize) {
                let (best_dx, best_dy) =
                    find_best_match(prev, curr, width, height, x, y, block_size, search_range);

                // Fill the block with the flow vector
                for by in 0..block_size {
                    for bx in 0..block_size {
                        let px = x + bx;
                        let py = y + by;
                        if px < wi && py < hi {
                            flow.set(px as u32, py as u32, best_dx as f32, best_dy as f32);
                        }
                    }
                }
            }
        }

        Ok(flow)
    }

    /// Lucas-Kanade optical flow using the tracking module.
    fn estimate_lucas_kanade(
        &self,
        prev: &[u8],
        curr: &[u8],
        width: u32,
        height: u32,
    ) -> CvResult<FlowField> {
        let optical_flow = tracking::OpticalFlow::new(tracking::FlowMethod::LucasKanade)
            .with_window_size(self.window_size)
            .with_max_level(if self.use_pyramid {
                self.pyramid_levels
            } else {
                1
            });

        let tracking_flow = optical_flow.compute(prev, curr, width, height)?;

        // Convert from tracking::FlowField to our FlowField
        Ok(FlowField {
            flow_x: tracking_flow.flow_x,
            flow_y: tracking_flow.flow_y,
            width: tracking_flow.width,
            height: tracking_flow.height,
        })
    }

    /// Farneback optical flow using the tracking module.
    fn estimate_farneback(
        &self,
        prev: &[u8],
        curr: &[u8],
        width: u32,
        height: u32,
    ) -> CvResult<FlowField> {
        let optical_flow = tracking::OpticalFlow::new(tracking::FlowMethod::Farneback)
            .with_window_size(self.window_size)
            .with_max_level(if self.use_pyramid {
                self.pyramid_levels
            } else {
                1
            });

        let tracking_flow = optical_flow.compute(prev, curr, width, height)?;

        // Convert from tracking::FlowField to our FlowField
        Ok(FlowField {
            flow_x: tracking_flow.flow_x,
            flow_y: tracking_flow.flow_y,
            width: tracking_flow.width,
            height: tracking_flow.height,
        })
    }

    /// Extract grayscale data from a video frame.
    ///
    /// For YUV formats, this uses the Y plane directly.
    /// For RGB formats, this converts to grayscale.
    fn extract_grayscale(&self, frame: &VideoFrame) -> CvResult<Vec<u8>> {
        if frame.planes.is_empty() {
            return Err(CvError::insufficient_data(1, frame.planes.len()));
        }

        // For YUV formats, the first plane is the luma (Y) channel
        // which is already grayscale
        let plane = &frame.planes[0];
        Ok(plane.data.clone())
    }
}

impl Default for FlowEstimator {
    fn default() -> Self {
        Self::new(FlowMethod::LucasKanade)
    }
}

/// Find the best matching block using sum of absolute differences.
#[allow(clippy::too_many_arguments)]
fn find_best_match(
    prev: &[u8],
    curr: &[u8],
    width: u32,
    height: u32,
    x: i32,
    y: i32,
    block_size: i32,
    search_range: i32,
) -> (i32, i32) {
    let wi = width as i32;
    let hi = height as i32;

    let mut best_dx = 0;
    let mut best_dy = 0;
    let mut best_sad = u32::MAX;

    for dy in -search_range..=search_range {
        for dx in -search_range..=search_range {
            let mut sad = 0u32;
            let mut count = 0u32;

            for by in 0..block_size {
                for bx in 0..block_size {
                    let px = x + bx;
                    let py = y + by;

                    let qx = px + dx;
                    let qy = py + dy;

                    if px >= 0
                        && px < wi
                        && py >= 0
                        && py < hi
                        && qx >= 0
                        && qx < wi
                        && qy >= 0
                        && qy < hi
                    {
                        let pidx = (py * wi + px) as usize;
                        let qidx = (qy * wi + qx) as usize;

                        if pidx < prev.len() && qidx < curr.len() {
                            let diff = (prev[pidx] as i32 - curr[qidx] as i32).unsigned_abs();
                            sad += diff;
                            count += 1;
                        }
                    }
                }
            }

            if count > 0 {
                let avg_sad = sad / count;
                if avg_sad < best_sad {
                    best_sad = avg_sad;
                    best_dx = dx;
                    best_dy = dy;
                }
            }
        }
    }

    (best_dx, best_dy)
}

/// Build image pyramid for multi-scale processing.
#[allow(dead_code)]
fn build_pyramid(img: &[u8], width: u32, height: u32, levels: u32) -> Vec<(Vec<u8>, u32, u32)> {
    let mut pyramid = Vec::with_capacity(levels as usize);
    pyramid.push((img.to_vec(), width, height));

    for _ in 1..levels {
        let (prev_img, prev_w, prev_h) = pyramid
            .last()
            .expect("pyramid is non-empty by construction");
        let new_w = prev_w / 2;
        let new_h = prev_h / 2;

        if new_w < 8 || new_h < 8 {
            break;
        }

        let downsampled = downsample(prev_img, *prev_w, *prev_h);
        pyramid.push((downsampled, new_w, new_h));
    }

    pyramid
}

/// Downsample image by factor of 2.
fn downsample(img: &[u8], width: u32, height: u32) -> Vec<u8> {
    let new_w = width / 2;
    let new_h = height / 2;
    let mut result = vec![0u8; (new_w * new_h) as usize];

    for y in 0..new_h {
        for x in 0..new_w {
            let sx = (x * 2) as usize;
            let sy = (y * 2) as usize;

            let mut sum = 0u32;
            let mut count = 0u32;

            for dy in 0..2 {
                for dx in 0..2 {
                    let px = sx + dx;
                    let py = sy + dy;
                    if px < width as usize && py < height as usize {
                        sum += img[py * width as usize + px] as u32;
                        count += 1;
                    }
                }
            }

            result[(y * new_w + x) as usize] = if count > 0 { (sum / count) as u8 } else { 0 };
        }
    }

    result
}
