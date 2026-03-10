//! Object tracking algorithms.
//!
//! This module provides single-object tracking with various algorithms:
//!
//! - KCF (Kernelized Correlation Filter)
//! - MOSSE (Minimum Output Sum of Squared Error)
//! - MedianFlow
//! - CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability)
//!
//! # Example
//!
//! ```
//! use oximedia_cv::tracking::{ObjectTracker, TrackerType};
//! use oximedia_cv::detect::BoundingBox;
//!
//! let bbox = BoundingBox::new(100.0, 100.0, 50.0, 50.0);
//! let tracker = ObjectTracker::new(TrackerType::Kcf, bbox);
//! ```

use crate::detect::BoundingBox;
use crate::error::{CvError, CvResult};

/// Object tracker type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackerType {
    /// Kernelized Correlation Filter tracker.
    Kcf,
    /// MOSSE (Minimum Output Sum of Squared Error) tracker.
    Mosse,
    /// MedianFlow tracker.
    MedianFlow,
    /// CSRT (Channel and Spatial Reliability) tracker.
    Csrt,
}

/// Single-object tracker.
///
/// Tracks a single object across frames using correlation-based methods.
///
/// # Examples
///
/// ```
/// use oximedia_cv::tracking::{ObjectTracker, TrackerType};
/// use oximedia_cv::detect::BoundingBox;
///
/// let bbox = BoundingBox::new(50.0, 50.0, 100.0, 100.0);
/// let tracker = ObjectTracker::new(TrackerType::Mosse, bbox);
/// ```
#[derive(Debug, Clone)]
pub struct ObjectTracker {
    /// Tracker type.
    tracker_type: TrackerType,
    /// Current bounding box.
    bbox: BoundingBox,
    /// Tracking confidence.
    confidence: f32,
    /// Tracking history.
    history: Vec<BoundingBox>,
    /// Object template.
    template: Option<Vec<f64>>,
    /// Template size.
    template_size: (usize, usize),
    /// Learning rate.
    learning_rate: f64,
    /// Peak-to-sidelobe ratio threshold.
    psr_threshold: f64,
}

impl ObjectTracker {
    /// Create a new object tracker.
    ///
    /// # Arguments
    ///
    /// * `tracker_type` - Type of tracker to use
    /// * `initial_bbox` - Initial bounding box of the object
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::tracking::{ObjectTracker, TrackerType};
    /// use oximedia_cv::detect::BoundingBox;
    ///
    /// let bbox = BoundingBox::new(100.0, 100.0, 50.0, 50.0);
    /// let tracker = ObjectTracker::new(TrackerType::Kcf, bbox);
    /// ```
    #[must_use]
    pub fn new(tracker_type: TrackerType, initial_bbox: BoundingBox) -> Self {
        Self {
            tracker_type,
            bbox: initial_bbox,
            confidence: 1.0,
            history: vec![initial_bbox],
            template: None,
            template_size: (32, 32),
            learning_rate: 0.02,
            psr_threshold: 8.0,
        }
    }

    /// Set learning rate for template update.
    #[must_use]
    pub const fn with_learning_rate(mut self, rate: f64) -> Self {
        self.learning_rate = rate;
        self
    }

    /// Set PSR threshold for tracking quality.
    #[must_use]
    pub const fn with_psr_threshold(mut self, threshold: f64) -> Self {
        self.psr_threshold = threshold;
        self
    }

    /// Update tracker with a new frame.
    ///
    /// # Arguments
    ///
    /// * `frame` - Grayscale frame data
    /// * `w` - Frame width
    /// * `h` - Frame height
    ///
    /// # Returns
    ///
    /// Updated bounding box.
    ///
    /// # Errors
    ///
    /// Returns an error if tracking fails or dimensions are invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::tracking::{ObjectTracker, TrackerType};
    /// use oximedia_cv::detect::BoundingBox;
    ///
    /// let bbox = BoundingBox::new(50.0, 50.0, 30.0, 30.0);
    /// let mut tracker = ObjectTracker::new(TrackerType::Mosse, bbox);
    ///
    /// let frame = vec![100u8; 10000];
    /// let new_bbox = tracker.update(&frame, 100, 100)?;
    /// ```
    pub fn update(&mut self, frame: &[u8], w: u32, h: u32) -> CvResult<BoundingBox> {
        if w == 0 || h == 0 {
            return Err(CvError::invalid_dimensions(w, h));
        }

        let size = w as usize * h as usize;
        if frame.len() < size {
            return Err(CvError::insufficient_data(size, frame.len()));
        }

        // Initialize template on first frame
        if self.template.is_none() {
            self.initialize_template(frame, w, h)?;
            return Ok(self.bbox);
        }

        // Track based on method
        let new_bbox = match self.tracker_type {
            TrackerType::Kcf => self.update_kcf(frame, w, h)?,
            TrackerType::Mosse => self.update_mosse(frame, w, h)?,
            TrackerType::MedianFlow => self.update_medianflow(frame, w, h)?,
            TrackerType::Csrt => self.update_csrt(frame, w, h)?,
        };

        // Update state
        self.bbox = new_bbox;
        self.history.push(new_bbox);

        // Limit history size
        if self.history.len() > 100 {
            self.history.remove(0);
        }

        Ok(new_bbox)
    }

    /// Reset tracker with a new bounding box.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::tracking::{ObjectTracker, TrackerType};
    /// use oximedia_cv::detect::BoundingBox;
    ///
    /// let bbox1 = BoundingBox::new(50.0, 50.0, 30.0, 30.0);
    /// let mut tracker = ObjectTracker::new(TrackerType::Mosse, bbox1);
    ///
    /// let bbox2 = BoundingBox::new(100.0, 100.0, 40.0, 40.0);
    /// tracker.reset(bbox2);
    /// ```
    pub fn reset(&mut self, bbox: BoundingBox) {
        self.bbox = bbox;
        self.confidence = 1.0;
        self.history.clear();
        self.history.push(bbox);
        self.template = None;
    }

    /// Get current tracking confidence.
    #[must_use]
    pub const fn confidence(&self) -> f32 {
        self.confidence
    }

    /// Get current bounding box.
    #[must_use]
    pub const fn bbox(&self) -> &BoundingBox {
        &self.bbox
    }

    /// Initialize template from first frame.
    fn initialize_template(&mut self, frame: &[u8], w: u32, h: u32) -> CvResult<()> {
        let patch = extract_patch(frame, w, h, &self.bbox, self.template_size)?;
        self.template = Some(patch);
        Ok(())
    }

    /// Update using KCF tracker.
    fn update_kcf(&mut self, frame: &[u8], w: u32, h: u32) -> CvResult<BoundingBox> {
        // Search region (1.5x the bbox size)
        let search_bbox = self.bbox.expand(self.bbox.width * 0.25);

        // Extract search region
        let search_patch = extract_patch(frame, w, h, &search_bbox, (64, 64))?;

        // Correlate with template
        let template = self
            .template
            .as_ref()
            .expect("template initialized before tracking");
        let response = compute_correlation(&search_patch, template, (64, 64), self.template_size);

        // Find peak
        let (max_idx, _max_val, psr) = find_peak_with_psr(&response, (64, 64));

        // Update confidence based on PSR
        self.confidence = if psr > self.psr_threshold {
            (psr / 20.0).min(1.0) as f32
        } else {
            0.5
        };

        // Convert peak location to bbox
        let peak_x = (max_idx % 64) as f32;
        let peak_y = (max_idx / 64) as f32;

        let scale_x = search_bbox.width / 64.0;
        let scale_y = search_bbox.height / 64.0;

        let new_center_x = search_bbox.x + peak_x * scale_x;
        let new_center_y = search_bbox.y + peak_y * scale_y;

        let new_bbox = BoundingBox::new(
            new_center_x - self.bbox.width / 2.0,
            new_center_y - self.bbox.height / 2.0,
            self.bbox.width,
            self.bbox.height,
        );

        // Update template
        if psr > self.psr_threshold {
            let new_patch = extract_patch(frame, w, h, &new_bbox, self.template_size)?;
            self.update_template(&new_patch);
        }

        Ok(new_bbox)
    }

    /// Update using MOSSE tracker.
    fn update_mosse(&mut self, frame: &[u8], w: u32, h: u32) -> CvResult<BoundingBox> {
        // Similar to KCF but simpler correlation
        let search_bbox = self.bbox.expand(self.bbox.width * 0.2);
        let search_patch = extract_patch(frame, w, h, &search_bbox, (48, 48))?;

        let template = self
            .template
            .as_ref()
            .expect("template initialized before tracking");
        let response = compute_correlation(&search_patch, template, (48, 48), self.template_size);

        let (max_idx, _max_val, psr) = find_peak_with_psr(&response, (48, 48));

        self.confidence = if psr > self.psr_threshold { 0.9 } else { 0.5 };

        let peak_x = (max_idx % 48) as f32;
        let peak_y = (max_idx / 48) as f32;

        let scale_x = search_bbox.width / 48.0;
        let scale_y = search_bbox.height / 48.0;

        let new_center_x = search_bbox.x + peak_x * scale_x;
        let new_center_y = search_bbox.y + peak_y * scale_y;

        let new_bbox = BoundingBox::new(
            new_center_x - self.bbox.width / 2.0,
            new_center_y - self.bbox.height / 2.0,
            self.bbox.width,
            self.bbox.height,
        );

        // Update template with high learning rate for MOSSE
        if psr > self.psr_threshold {
            let new_patch = extract_patch(frame, w, h, &new_bbox, self.template_size)?;
            self.update_template(&new_patch);
        }

        Ok(new_bbox)
    }

    /// Update using MedianFlow tracker.
    fn update_medianflow(&mut self, frame: &[u8], w: u32, h: u32) -> CvResult<BoundingBox> {
        // For MedianFlow, we use a grid of points and track them
        // Then use median displacement
        let grid_size = 10;
        let mut displacements_x = Vec::new();
        let mut displacements_y = Vec::new();

        // Sample points in bbox
        for i in 0..grid_size {
            for j in 0..grid_size {
                let x = self.bbox.x + (i as f32 + 0.5) * self.bbox.width / grid_size as f32;
                let y = self.bbox.y + (j as f32 + 0.5) * self.bbox.height / grid_size as f32;

                // Simple local search
                if let Some((dx, dy)) = track_point_simple(frame, w, h, x, y, 7) {
                    displacements_x.push(dx);
                    displacements_y.push(dy);
                }
            }
        }

        if displacements_x.is_empty() {
            self.confidence = 0.3;
            return Ok(self.bbox);
        }

        // Compute median displacement
        displacements_x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        displacements_y.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median_dx = displacements_x[displacements_x.len() / 2];
        let median_dy = displacements_y[displacements_y.len() / 2];

        let new_bbox = BoundingBox::new(
            self.bbox.x + median_dx,
            self.bbox.y + median_dy,
            self.bbox.width,
            self.bbox.height,
        );

        self.confidence = 0.8;

        Ok(new_bbox)
    }

    /// Update using CSRT tracker.
    fn update_csrt(&mut self, frame: &[u8], w: u32, h: u32) -> CvResult<BoundingBox> {
        // CSRT uses channel features and spatial reliability
        // Simplified version here
        let search_bbox = self.bbox.expand(self.bbox.width * 0.3);
        let search_patch = extract_patch(frame, w, h, &search_bbox, (64, 64))?;

        let template = self
            .template
            .as_ref()
            .expect("template initialized before tracking");
        let response = compute_correlation(&search_patch, template, (64, 64), self.template_size);

        let (max_idx, _max_val, psr) = find_peak_with_psr(&response, (64, 64));

        self.confidence = if psr > self.psr_threshold { 0.95 } else { 0.6 };

        let peak_x = (max_idx % 64) as f32;
        let peak_y = (max_idx / 64) as f32;

        let scale_x = search_bbox.width / 64.0;
        let scale_y = search_bbox.height / 64.0;

        let new_center_x = search_bbox.x + peak_x * scale_x;
        let new_center_y = search_bbox.y + peak_y * scale_y;

        let new_bbox = BoundingBox::new(
            new_center_x - self.bbox.width / 2.0,
            new_center_y - self.bbox.height / 2.0,
            self.bbox.width,
            self.bbox.height,
        );

        // Update template with low learning rate for CSRT
        if psr > self.psr_threshold {
            let new_patch = extract_patch(frame, w, h, &new_bbox, self.template_size)?;
            self.update_template(&new_patch);
        }

        Ok(new_bbox)
    }

    /// Update template with exponential moving average.
    fn update_template(&mut self, new_patch: &[f64]) {
        if let Some(template) = &mut self.template {
            let alpha = self.learning_rate;
            for i in 0..template.len().min(new_patch.len()) {
                template[i] = alpha * new_patch[i] + (1.0 - alpha) * template[i];
            }
        }
    }
}

/// Extract and normalize image patch.
fn extract_patch(
    frame: &[u8],
    w: u32,
    h: u32,
    bbox: &BoundingBox,
    target_size: (usize, usize),
) -> CvResult<Vec<f64>> {
    let (tw, th) = target_size;

    // Clamp bbox to image bounds
    let clamped = bbox.clamp(w as f32, h as f32);

    let x0 = clamped.x as usize;
    let y0 = clamped.y as usize;
    let x1 = (clamped.x + clamped.width).min(w as f32) as usize;
    let y1 = (clamped.y + clamped.height).min(h as f32) as usize;

    if x1 <= x0 || y1 <= y0 {
        return Err(CvError::invalid_roi(
            x0 as u32,
            y0 as u32,
            (x1 - x0) as u32,
            (y1 - y0) as u32,
        ));
    }

    let mut patch = vec![0.0; tw * th];

    // Simple nearest-neighbor resizing
    for ty in 0..th {
        for tx in 0..tw {
            let sx = x0 + (tx * (x1 - x0)) / tw;
            let sy = y0 + (ty * (y1 - y0)) / th;

            if sx < w as usize && sy < h as usize {
                let idx = sy * w as usize + sx;
                if idx < frame.len() {
                    patch[ty * tw + tx] = frame[idx] as f64;
                }
            }
        }
    }

    // Normalize patch
    normalize_patch(&mut patch);

    Ok(patch)
}

/// Normalize patch to zero mean and unit variance.
fn normalize_patch(patch: &mut [f64]) {
    let n = patch.len() as f64;
    let mean = patch.iter().sum::<f64>() / n;
    let variance = patch.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
    let std = variance.sqrt();

    if std > 1.0 {
        for val in patch {
            *val = (*val - mean) / std;
        }
    }
}

/// Compute correlation response map.
fn compute_correlation(
    search: &[f64],
    template: &[f64],
    search_size: (usize, usize),
    template_size: (usize, usize),
) -> Vec<f64> {
    let (sw, sh) = search_size;
    let (tw, th) = template_size;

    let mut response = vec![0.0; sw * sh];

    for sy in 0..=(sh.saturating_sub(th)) {
        for sx in 0..=(sw.saturating_sub(tw)) {
            let mut sum = 0.0;

            for ty in 0..th {
                for tx in 0..tw {
                    let search_idx = (sy + ty) * sw + (sx + tx);
                    let template_idx = ty * tw + tx;

                    if search_idx < search.len() && template_idx < template.len() {
                        sum += search[search_idx] * template[template_idx];
                    }
                }
            }

            let response_idx = sy * sw + sx;
            if response_idx < response.len() {
                response[response_idx] = sum;
            }
        }
    }

    response
}

/// Find peak in response map and compute PSR (Peak-to-Sidelobe Ratio).
fn find_peak_with_psr(response: &[f64], size: (usize, usize)) -> (usize, f64, f64) {
    let (w, _h) = size;

    // Find maximum
    let mut max_idx = 0;
    let mut max_val = f64::MIN;

    for (i, &val) in response.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    // Compute PSR
    let peak_x = max_idx % w;
    let peak_y = max_idx / w;

    // Compute mean and std of sidelobes (excluding 11x11 window around peak)
    let mut sidelobe_sum = 0.0;
    let mut sidelobe_sum_sq = 0.0;
    let mut sidelobe_count = 0.0;

    for (i, &val) in response.iter().enumerate() {
        let x = i % w;
        let y = i / w;

        let dx = (x as i32 - peak_x as i32).abs();
        let dy = (y as i32 - peak_y as i32).abs();

        if dx > 5 || dy > 5 {
            sidelobe_sum += val;
            sidelobe_sum_sq += val * val;
            sidelobe_count += 1.0;
        }
    }

    let psr = if sidelobe_count > 0.0 {
        let sidelobe_mean = sidelobe_sum / sidelobe_count;
        let sidelobe_var = (sidelobe_sum_sq / sidelobe_count) - (sidelobe_mean * sidelobe_mean);
        let sidelobe_std = if sidelobe_var > 0.0 {
            sidelobe_var.sqrt()
        } else {
            0.0
        };

        if sidelobe_std > 1e-6 {
            (max_val - sidelobe_mean) / sidelobe_std
        } else if max_val > sidelobe_mean + 1e-6 {
            // Peak is clearly dominant but sidelobes are flat; return normalized peak
            (max_val - sidelobe_mean).max(0.0)
        } else {
            0.0
        }
    } else if max_val > 0.0 {
        // No sidelobes region found (small response map); if peak is positive, PSR is positive
        max_val
    } else {
        0.0
    };

    (max_idx, max_val, psr)
}

/// Track a single point using template matching.
#[allow(dead_code)]
fn track_point_simple(
    frame: &[u8],
    w: u32,
    h: u32,
    x: f32,
    y: f32,
    search_radius: i32,
) -> Option<(f32, f32)> {
    let xi = x as i32;
    let yi = y as i32;

    if xi < search_radius
        || xi >= w as i32 - search_radius
        || yi < search_radius
        || yi >= h as i32 - search_radius
    {
        return None;
    }

    // Extract template
    let template_size = 5;
    let mut template = Vec::new();

    for dy in -template_size..=template_size {
        for dx in -template_size..=template_size {
            let idx = ((yi + dy) * w as i32 + (xi + dx)) as usize;
            if idx < frame.len() {
                template.push(frame[idx] as f64);
            }
        }
    }

    // Search
    let mut best_match = (0.0f32, 0.0f32);
    let mut best_score = f64::MIN;

    for dy in -search_radius..=search_radius {
        for dx in -search_radius..=search_radius {
            let mut score = 0.0;

            for ty in -template_size..=template_size {
                for tx in -template_size..=template_size {
                    let idx1 = ((yi + ty) * w as i32 + (xi + tx)) as usize;
                    let idx2 = ((yi + dy + ty) * w as i32 + (xi + dx + tx)) as usize;

                    if idx1 < frame.len() && idx2 < frame.len() {
                        let tidx = ((ty + template_size) * (2 * template_size + 1)
                            + (tx + template_size)) as usize;
                        if tidx < template.len() {
                            score += template[tidx] * frame[idx2] as f64;
                        }
                    }
                }
            }

            if score > best_score {
                best_score = score;
                best_match = (dx as f32, dy as f32);
            }
        }
    }

    Some(best_match)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_tracker_new() {
        let bbox = BoundingBox::new(50.0, 50.0, 100.0, 100.0);
        let tracker = ObjectTracker::new(TrackerType::Kcf, bbox);

        assert_eq!(tracker.tracker_type, TrackerType::Kcf);
        assert_eq!(tracker.confidence(), 1.0);
    }

    #[test]
    fn test_object_tracker_with_params() {
        let bbox = BoundingBox::new(50.0, 50.0, 100.0, 100.0);
        let tracker = ObjectTracker::new(TrackerType::Mosse, bbox)
            .with_learning_rate(0.05)
            .with_psr_threshold(10.0);

        assert!((tracker.learning_rate - 0.05).abs() < f64::EPSILON);
        assert!((tracker.psr_threshold - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_object_tracker_update() {
        let bbox = BoundingBox::new(40.0, 40.0, 20.0, 20.0);
        let mut tracker = ObjectTracker::new(TrackerType::Mosse, bbox);

        let frame = vec![100u8; 10000];
        let result = tracker.update(&frame, 100, 100);
        assert!(result.is_ok());
    }

    #[test]
    fn test_object_tracker_reset() {
        let bbox1 = BoundingBox::new(50.0, 50.0, 30.0, 30.0);
        let mut tracker = ObjectTracker::new(TrackerType::Kcf, bbox1);

        let bbox2 = BoundingBox::new(100.0, 100.0, 40.0, 40.0);
        tracker.reset(bbox2);

        assert_eq!(tracker.bbox().x, 100.0);
        assert_eq!(tracker.confidence(), 1.0);
    }

    #[test]
    fn test_extract_patch() {
        let frame = vec![100u8; 10000];
        let bbox = BoundingBox::new(25.0, 25.0, 50.0, 50.0);

        let patch = extract_patch(&frame, 100, 100, &bbox, (32, 32));
        assert!(patch.is_ok());

        let p = patch.expect("p should be valid");
        assert_eq!(p.len(), 32 * 32);
    }

    #[test]
    fn test_normalize_patch() {
        let mut patch = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        normalize_patch(&mut patch);

        // Check mean is ~0
        let mean = patch.iter().sum::<f64>() / patch.len() as f64;
        assert!(mean.abs() < 1e-10);
    }

    #[test]
    fn test_compute_correlation() {
        let search = vec![1.0; 64];
        let template = vec![1.0; 16];

        let response = compute_correlation(&search, &template, (8, 8), (4, 4));
        assert!(!response.is_empty());
    }

    #[test]
    fn test_find_peak_with_psr() {
        let mut response = vec![0.0; 100];
        response[55] = 10.0; // Peak at (5, 5) in 10x10 grid

        let (max_idx, max_val, psr) = find_peak_with_psr(&response, (10, 10));
        assert_eq!(max_idx, 55);
        assert!((max_val - 10.0).abs() < f64::EPSILON);
        assert!(psr > 0.0);
    }

    #[test]
    fn test_tracker_types() {
        assert_eq!(TrackerType::Kcf, TrackerType::Kcf);
        assert_ne!(TrackerType::Kcf, TrackerType::Mosse);
    }
}
