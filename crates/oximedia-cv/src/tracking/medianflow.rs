//! MedianFlow tracker.
//!
//! Tracks a sparse set of points using forward-backward error and
//! uses the median displacement for robust motion estimation.
//!
//! # Example
//!
//! ```
//! use oximedia_cv::tracking::medianflow::MedianFlowTracker;
//! use oximedia_cv::detect::BoundingBox;
//!
//! let bbox = BoundingBox::new(50.0, 50.0, 100.0, 100.0);
//! let tracker = MedianFlowTracker::new(bbox);
//! ```

use crate::detect::BoundingBox;
use crate::error::{CvError, CvResult};

/// MedianFlow tracker configuration.
#[derive(Debug, Clone)]
pub struct MedianFlowTracker {
    /// Current bounding box.
    bbox: BoundingBox,
    /// Grid size for point sampling.
    grid_size: usize,
    /// Tracking points.
    points: Vec<(f32, f32)>,
    /// Previous frame (for forward-backward check).
    prev_frame: Vec<u8>,
    /// Previous frame dimensions.
    prev_dims: (u32, u32),
    /// Forward-backward error threshold.
    fb_threshold: f64,
    /// Current confidence.
    confidence: f64,
    /// Template size for point tracking.
    template_size: usize,
    /// Search radius.
    search_radius: usize,
}

impl MedianFlowTracker {
    /// Create a new MedianFlow tracker.
    ///
    /// # Arguments
    ///
    /// * `bbox` - Initial bounding box
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::tracking::medianflow::MedianFlowTracker;
    /// use oximedia_cv::detect::BoundingBox;
    ///
    /// let bbox = BoundingBox::new(100.0, 100.0, 50.0, 50.0);
    /// let tracker = MedianFlowTracker::new(bbox);
    /// ```
    #[must_use]
    pub fn new(bbox: BoundingBox) -> Self {
        Self {
            bbox,
            grid_size: 10,
            points: Vec::new(),
            prev_frame: Vec::new(),
            prev_dims: (0, 0),
            fb_threshold: 1.0,
            confidence: 1.0,
            template_size: 5,
            search_radius: 10,
        }
    }

    /// Set grid size for point sampling.
    #[must_use]
    pub const fn with_grid_size(mut self, size: usize) -> Self {
        self.grid_size = size;
        self
    }

    /// Set forward-backward error threshold.
    #[must_use]
    pub const fn with_fb_threshold(mut self, threshold: f64) -> Self {
        self.fb_threshold = threshold;
        self
    }

    /// Initialize the tracker with the first frame.
    ///
    /// # Errors
    ///
    /// Returns an error if frame dimensions are invalid.
    pub fn initialize(&mut self, frame: &[u8], width: u32, height: u32) -> CvResult<()> {
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        // Sample points in a grid within the bounding box
        self.points = self.sample_grid_points();

        // Store previous frame
        self.prev_frame = frame.to_vec();
        self.prev_dims = (width, height);

        Ok(())
    }

    /// Update tracker with a new frame.
    ///
    /// # Errors
    ///
    /// Returns an error if tracking fails or dimensions are invalid.
    pub fn update(&mut self, frame: &[u8], width: u32, height: u32) -> CvResult<BoundingBox> {
        if self.prev_frame.is_empty() {
            return Err(CvError::tracking_error("Tracker not initialized"));
        }

        // Track points forward
        let mut tracked_forward = Vec::new();
        let mut forward_success = Vec::new();

        for &(x, y) in &self.points {
            if let Some((new_x, new_y)) = track_point(
                &self.prev_frame,
                frame,
                self.prev_dims.0,
                self.prev_dims.1,
                x,
                y,
                self.template_size,
                self.search_radius,
            ) {
                tracked_forward.push((new_x, new_y));
                forward_success.push(true);
            } else {
                tracked_forward.push((x, y));
                forward_success.push(false);
            }
        }

        // Track points backward (for error checking)
        let mut fb_errors = Vec::new();

        for (i, &(x, y)) in tracked_forward.iter().enumerate() {
            if forward_success[i] {
                if let Some((back_x, back_y)) = track_point(
                    frame,
                    &self.prev_frame,
                    width,
                    height,
                    x,
                    y,
                    self.template_size,
                    self.search_radius,
                ) {
                    let dx = self.points[i].0 - back_x;
                    let dy = self.points[i].1 - back_y;
                    let error = ((dx * dx + dy * dy) as f64).sqrt();
                    fb_errors.push(error);
                } else {
                    fb_errors.push(f64::INFINITY);
                }
            } else {
                fb_errors.push(f64::INFINITY);
            }
        }

        // Filter points by forward-backward error
        let mut valid_displacements_x = Vec::new();
        let mut valid_displacements_y = Vec::new();

        for i in 0..self.points.len() {
            if forward_success[i] && fb_errors[i] < self.fb_threshold {
                let dx = tracked_forward[i].0 - self.points[i].0;
                let dy = tracked_forward[i].1 - self.points[i].1;
                valid_displacements_x.push(dx);
                valid_displacements_y.push(dy);
            }
        }

        if valid_displacements_x.is_empty() {
            self.confidence = 0.0;
            return Err(CvError::tracking_error("No valid points tracked"));
        }

        // Compute median displacement
        valid_displacements_x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        valid_displacements_y.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median_dx = valid_displacements_x[valid_displacements_x.len() / 2];
        let median_dy = valid_displacements_y[valid_displacements_y.len() / 2];

        // Estimate scale change
        let scale_change = self.estimate_scale_change(&tracked_forward, &fb_errors)?;

        // Update bounding box
        self.bbox.x += median_dx;
        self.bbox.y += median_dy;
        self.bbox.width *= scale_change;
        self.bbox.height *= scale_change;

        // Clamp to image bounds
        self.bbox = self.bbox.clamp(width as f32, height as f32);

        // Update confidence based on tracking success rate
        let success_rate = valid_displacements_x.len() as f64 / self.points.len() as f64;
        self.confidence = success_rate;

        // Update points for next frame
        self.points = self.sample_grid_points();

        // Store current frame for next iteration
        self.prev_frame = frame.to_vec();
        self.prev_dims = (width, height);

        Ok(self.bbox)
    }

    /// Get current bounding box.
    #[must_use]
    pub const fn bbox(&self) -> &BoundingBox {
        &self.bbox
    }

    /// Get current confidence.
    #[must_use]
    pub const fn confidence(&self) -> f64 {
        self.confidence
    }

    /// Reset tracker with new bounding box.
    pub fn reset(&mut self, bbox: BoundingBox) {
        self.bbox = bbox;
        self.points.clear();
        self.prev_frame.clear();
        self.confidence = 1.0;
    }

    /// Sample points in a grid within the bounding box.
    fn sample_grid_points(&self) -> Vec<(f32, f32)> {
        let mut points = Vec::new();

        for i in 0..self.grid_size {
            for j in 0..self.grid_size {
                let x = self.bbox.x + (i as f32 + 0.5) * self.bbox.width / self.grid_size as f32;
                let y = self.bbox.y + (j as f32 + 0.5) * self.bbox.height / self.grid_size as f32;
                points.push((x, y));
            }
        }

        points
    }

    /// Estimate scale change from point correspondences.
    fn estimate_scale_change(&self, tracked: &[(f32, f32)], errors: &[f64]) -> CvResult<f32> {
        // Compute pairwise distances before and after
        let mut scale_ratios = Vec::new();

        for i in 0..self.points.len() {
            if errors[i] >= self.fb_threshold {
                continue;
            }

            for j in (i + 1)..self.points.len() {
                if errors[j] >= self.fb_threshold {
                    continue;
                }

                // Distance before
                let dx1 = self.points[j].0 - self.points[i].0;
                let dy1 = self.points[j].1 - self.points[i].1;
                let dist1 = (dx1 * dx1 + dy1 * dy1).sqrt();

                // Distance after
                let dx2 = tracked[j].0 - tracked[i].0;
                let dy2 = tracked[j].1 - tracked[i].1;
                let dist2 = (dx2 * dx2 + dy2 * dy2).sqrt();

                if dist1 > 1.0 && dist2 > 1.0 {
                    scale_ratios.push(dist2 / dist1);
                }
            }
        }

        if scale_ratios.is_empty() {
            return Ok(1.0);
        }

        // Return median scale ratio
        scale_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scale_ratios[scale_ratios.len() / 2])
    }
}

/// Track a single point using template matching.
#[allow(clippy::too_many_arguments)]
fn track_point(
    prev_frame: &[u8],
    curr_frame: &[u8],
    width: u32,
    height: u32,
    x: f32,
    y: f32,
    template_size: usize,
    search_radius: usize,
) -> Option<(f32, f32)> {
    let xi = x as i32;
    let yi = y as i32;

    let half_temp = template_size as i32 / 2;
    let search_rad = search_radius as i32;

    // Check if template is within bounds
    if xi < half_temp
        || xi >= width as i32 - half_temp
        || yi < half_temp
        || yi >= height as i32 - half_temp
    {
        return None;
    }

    // Extract template from previous frame
    let mut template = Vec::new();

    for dy in -half_temp..=half_temp {
        for dx in -half_temp..=half_temp {
            let idx = ((yi + dy) * width as i32 + (xi + dx)) as usize;
            if idx < prev_frame.len() {
                template.push(prev_frame[idx]);
            } else {
                return None;
            }
        }
    }

    // Search in current frame
    let mut best_score = f64::MIN;
    let mut best_pos = (x, y);

    for dy in -search_rad..=search_rad {
        for dx in -search_rad..=search_rad {
            let test_x = xi + dx;
            let test_y = yi + dy;

            // Check bounds
            if test_x < half_temp
                || test_x >= width as i32 - half_temp
                || test_y < half_temp
                || test_y >= height as i32 - half_temp
            {
                continue;
            }

            // Compute NCC (Normalized Cross-Correlation)
            let mut sum_template = 0.0;
            let mut sum_image = 0.0;
            let mut sum_template_sq = 0.0;
            let mut sum_image_sq = 0.0;
            let mut sum_product = 0.0;
            let mut count = 0.0;

            for ty in -half_temp..=half_temp {
                for tx in -half_temp..=half_temp {
                    let tidx =
                        ((ty + half_temp) * template_size as i32 + (tx + half_temp)) as usize;
                    let iidx = ((test_y + ty) * width as i32 + (test_x + tx)) as usize;

                    if tidx < template.len() && iidx < curr_frame.len() {
                        let t_val = template[tidx] as f64;
                        let i_val = curr_frame[iidx] as f64;

                        sum_template += t_val;
                        sum_image += i_val;
                        sum_template_sq += t_val * t_val;
                        sum_image_sq += i_val * i_val;
                        sum_product += t_val * i_val;
                        count += 1.0;
                    }
                }
            }

            if count > 0.0 {
                let mean_t = sum_template / count;
                let mean_i = sum_image / count;

                let num = sum_product - count * mean_t * mean_i;
                let denom_t = (sum_template_sq - count * mean_t * mean_t).sqrt();
                let denom_i = (sum_image_sq - count * mean_i * mean_i).sqrt();

                let ncc = if denom_t > 1e-6 && denom_i > 1e-6 {
                    num / (denom_t * denom_i)
                } else {
                    0.0
                };

                if ncc > best_score {
                    best_score = ncc;
                    best_pos = (test_x as f32, test_y as f32);
                }
            }
        }
    }

    // Only accept if score is good enough
    if best_score > 0.7 {
        Some(best_pos)
    } else {
        None
    }
}
