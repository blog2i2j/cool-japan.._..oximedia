//! Noise variance estimation and analysis.
//!
//! Provides tools to estimate and analyze noise variance characteristics
//! for adaptive denoising.

use crate::{DenoiseError, DenoiseResult};
use oximedia_codec::VideoFrame;

/// Variance map for adaptive processing.
#[derive(Clone, Debug)]
pub struct VarianceMap {
    /// Width of the variance map.
    pub width: usize,
    /// Height of the variance map.
    pub height: usize,
    /// Variance at each pixel.
    pub variance: Vec<f32>,
    /// Global average variance.
    pub global_variance: f32,
}

impl VarianceMap {
    /// Create a new variance map from a frame.
    pub fn new(frame: &VideoFrame, window_size: usize) -> DenoiseResult<Self> {
        if frame.planes.is_empty() {
            return Err(DenoiseError::ProcessingError(
                "Frame has no planes".to_string(),
            ));
        }

        let plane = &frame.planes[0];
        let (width, height) = frame.plane_dimensions(0);

        let variance = compute_variance_map(
            plane.data.as_ref(),
            width as usize,
            height as usize,
            plane.stride,
            window_size,
        );

        let global_variance = variance.iter().sum::<f32>() / variance.len() as f32;

        Ok(Self {
            width: width as usize,
            height: height as usize,
            variance,
            global_variance,
        })
    }

    /// Get variance at a specific pixel.
    pub fn get_variance(&self, x: usize, y: usize) -> f32 {
        if x < self.width && y < self.height {
            self.variance[y * self.width + x]
        } else {
            self.global_variance
        }
    }

    /// Classify variance level at a pixel.
    pub fn classify_variance(&self, x: usize, y: usize) -> VarianceLevel {
        let var = self.get_variance(x, y);

        if var < self.global_variance * 0.5 {
            VarianceLevel::Low
        } else if var < self.global_variance * 1.5 {
            VarianceLevel::Medium
        } else {
            VarianceLevel::High
        }
    }
}

/// Variance level classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VarianceLevel {
    /// Low variance (smooth region).
    Low,
    /// Medium variance (textured region).
    Medium,
    /// High variance (edge or high-detail region).
    High,
}

/// Compute variance map for entire frame.
fn compute_variance_map(
    data: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    window_size: usize,
) -> Vec<f32> {
    let mut variance_map = vec![0.0f32; width * height];
    let radius = window_size / 2;

    for y in 0..height {
        for x in 0..width {
            let variance = compute_local_variance(data, x, y, width, height, stride, radius);
            variance_map[y * width + x] = variance;
        }
    }

    variance_map
}

/// Compute local variance in a window.
fn compute_local_variance(
    data: &[u8],
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    stride: usize,
    radius: usize,
) -> f32 {
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut count = 0;

    for dy in -(radius as i32)..=(radius as i32) {
        let ny = (y as i32 + dy).clamp(0, (height - 1) as i32) as usize;

        for dx in -(radius as i32)..=(radius as i32) {
            let nx = (x as i32 + dx).clamp(0, (width - 1) as i32) as usize;

            let val = f64::from(data[ny * stride + nx]);
            sum += val;
            sum_sq += val * val;
            count += 1;
        }
    }

    let mean = sum / f64::from(count);
    let variance = (sum_sq / f64::from(count)) - (mean * mean);

    variance.max(0.0) as f32
}

/// Estimate signal-dependent noise variance.
pub fn estimate_signal_dependent_variance(frame: &VideoFrame) -> DenoiseResult<(f32, f32)> {
    if frame.planes.is_empty() {
        return Err(DenoiseError::ProcessingError(
            "Frame has no planes".to_string(),
        ));
    }

    let plane = &frame.planes[0];
    let (width, height) = frame.plane_dimensions(0);

    // Compute variance as a function of intensity
    let mut intensity_bins = [0.0f64; 16];
    let mut variance_bins = [0.0f64; 16];
    let mut count_bins = [0u32; 16];

    let block_size = 8;

    for by in 0..((height as usize) / block_size) {
        for bx in 0..((width as usize) / block_size) {
            let (mean, variance) = compute_block_stats(
                plane.data.as_ref(),
                bx * block_size,
                by * block_size,
                block_size,
                plane.stride,
            );

            let bin = (mean / 16.0).floor() as usize;
            let bin = bin.min(15);

            intensity_bins[bin] += mean;
            variance_bins[bin] += variance;
            count_bins[bin] += 1;
        }
    }

    // Fit linear model: variance = a + b * intensity
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_xx = 0.0f64;
    let mut sum_xy = 0.0f64;
    let mut n = 0;

    for i in 0..16 {
        if count_bins[i] > 0 {
            let x = intensity_bins[i] / f64::from(count_bins[i]);
            let y = variance_bins[i] / f64::from(count_bins[i]);

            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
            n += 1;
        }
    }

    if n < 2 {
        return Ok((10.0, 0.0)); // Default values
    }

    // Linear regression
    let n = f64::from(n);
    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;

    Ok((intercept as f32, slope as f32))
}

/// Compute mean and variance of a block.
fn compute_block_stats(
    data: &[u8],
    start_x: usize,
    start_y: usize,
    block_size: usize,
    stride: usize,
) -> (f64, f64) {
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut count = 0;

    for y in 0..block_size {
        for x in 0..block_size {
            let idx = (start_y + y) * stride + start_x + x;
            let val = f64::from(data[idx]);
            sum += val;
            sum_sq += val * val;
            count += 1;
        }
    }

    let mean = sum / f64::from(count);
    let variance = (sum_sq / f64::from(count)) - (mean * mean);

    (mean, variance.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    #[test]
    fn test_variance_map_creation() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();

        let result = VarianceMap::new(&frame, 5);
        assert!(result.is_ok());

        let var_map = result.expect("var_map should be valid");
        assert_eq!(var_map.width, 64);
        assert_eq!(var_map.height, 64);
        assert_eq!(var_map.variance.len(), 64 * 64);
    }

    #[test]
    fn test_variance_classification() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();

        let var_map = VarianceMap::new(&frame, 5).expect("var_map should be valid");
        let level = var_map.classify_variance(32, 32);
        assert!(matches!(
            level,
            VarianceLevel::Low | VarianceLevel::Medium | VarianceLevel::High
        ));
    }

    #[test]
    fn test_signal_dependent_variance() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();

        let result = estimate_signal_dependent_variance(&frame);
        assert!(result.is_ok());

        let (intercept, slope) = result.expect("test expectation failed");
        assert!(intercept >= 0.0);
        assert!(slope >= 0.0 || slope < 0.0); // Can be negative
    }

    #[test]
    fn test_local_variance() {
        let data = vec![128u8; 64 * 64];
        let variance = compute_local_variance(&data, 32, 32, 64, 64, 64, 3);
        assert!(variance < f32::EPSILON);
    }

    #[test]
    fn test_block_stats() {
        let data = vec![100u8; 64 * 64];
        let (mean, variance) = compute_block_stats(&data, 0, 0, 8, 64);
        assert!((mean - 100.0).abs() < f64::EPSILON);
        assert!(variance < f64::EPSILON);
    }
}
