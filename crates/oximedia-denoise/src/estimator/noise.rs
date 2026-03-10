//! Noise level estimation.
//!
//! Estimates the noise level in video frames using various statistical
//! and signal processing techniques.

use crate::{DenoiseError, DenoiseResult};
use oximedia_codec::VideoFrame;

/// Noise estimator state.
pub struct NoiseEstimator {
    /// Estimated noise level (standard deviation).
    noise_level: Option<f32>,
    /// Noise estimation method.
    method: NoiseEstimationMethod,
}

/// Method used for noise estimation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoiseEstimationMethod {
    /// Median Absolute Deviation (MAD) of high-frequency components.
    MedianAbsoluteDeviation,
    /// Block-based variance estimation.
    BlockVariance,
    /// Laplacian variance method.
    LaplacianVariance,
}

impl NoiseEstimator {
    /// Create a new noise estimator.
    pub fn new() -> Self {
        Self {
            noise_level: None,
            method: NoiseEstimationMethod::MedianAbsoluteDeviation,
        }
    }

    /// Create estimator with specific method.
    pub fn with_method(method: NoiseEstimationMethod) -> Self {
        Self {
            noise_level: None,
            method,
        }
    }

    /// Estimate noise level from a frame.
    pub fn estimate(&mut self, frame: &VideoFrame) -> DenoiseResult<f32> {
        let noise = match self.method {
            NoiseEstimationMethod::MedianAbsoluteDeviation => estimate_noise_mad(frame)?,
            NoiseEstimationMethod::BlockVariance => estimate_noise_block_variance(frame)?,
            NoiseEstimationMethod::LaplacianVariance => estimate_noise_laplacian(frame)?,
        };

        self.noise_level = Some(noise);
        Ok(noise)
    }

    /// Get the estimated noise level.
    pub fn noise_level(&self) -> Option<f32> {
        self.noise_level
    }

    /// Reset the estimator.
    pub fn reset(&mut self) {
        self.noise_level = None;
    }
}

impl Default for NoiseEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Estimate noise using Median Absolute Deviation method.
fn estimate_noise_mad(frame: &VideoFrame) -> DenoiseResult<f32> {
    if frame.planes.is_empty() {
        return Err(DenoiseError::ProcessingError(
            "Frame has no planes".to_string(),
        ));
    }

    let plane = &frame.planes[0]; // Use luma plane
    let (width, height) = frame.plane_dimensions(0);

    // Compute high-frequency component using horizontal differences
    let mut diffs = Vec::new();

    for y in 0..(height as usize) {
        for x in 0..(width as usize - 1) {
            let idx = y * plane.stride + x;
            let diff = (i32::from(plane.data[idx + 1]) - i32::from(plane.data[idx])).abs();
            diffs.push(diff as f32);
        }
    }

    if diffs.is_empty() {
        return Ok(0.0);
    }

    // Compute median
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = diffs[diffs.len() / 2];

    // MAD estimator: sigma = median / 0.6745
    let noise = median / 0.6745;

    Ok(noise)
}

/// Estimate noise using block variance method.
fn estimate_noise_block_variance(frame: &VideoFrame) -> DenoiseResult<f32> {
    if frame.planes.is_empty() {
        return Err(DenoiseError::ProcessingError(
            "Frame has no planes".to_string(),
        ));
    }

    let plane = &frame.planes[0];
    let (width, height) = frame.plane_dimensions(0);
    let block_size = 8;

    let mut variances = Vec::new();

    // Compute variance for each block
    for by in 0..((height as usize) / block_size) {
        for bx in 0..((width as usize) / block_size) {
            let variance = compute_block_variance(
                plane.data.as_ref(),
                bx * block_size,
                by * block_size,
                block_size,
                plane.stride,
            );
            variances.push(variance);
        }
    }

    if variances.is_empty() {
        return Ok(0.0);
    }

    // Use minimum variance blocks (assumed to be smooth regions with mostly noise)
    variances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let percentile_10 = variances[variances.len() / 10];

    Ok(percentile_10.sqrt())
}

/// Compute variance of a block.
fn compute_block_variance(
    data: &[u8],
    start_x: usize,
    start_y: usize,
    block_size: usize,
    stride: usize,
) -> f32 {
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

    variance.max(0.0) as f32
}

/// Estimate noise using Laplacian variance method.
fn estimate_noise_laplacian(frame: &VideoFrame) -> DenoiseResult<f32> {
    if frame.planes.is_empty() {
        return Err(DenoiseError::ProcessingError(
            "Frame has no planes".to_string(),
        ));
    }

    let plane = &frame.planes[0];
    let (width, height) = frame.plane_dimensions(0);

    let mut sum = 0.0f64;
    let mut count = 0;

    // Apply Laplacian filter
    for y in 1..(height as usize - 1) {
        for x in 1..(width as usize - 1) {
            let idx = y * plane.stride + x;
            let center = f64::from(plane.data[idx]);

            let laplacian = 4.0 * center
                - f64::from(plane.data[idx - 1])
                - f64::from(plane.data[idx + 1])
                - f64::from(plane.data[idx - plane.stride])
                - f64::from(plane.data[idx + plane.stride]);

            sum += laplacian * laplacian;
            count += 1;
        }
    }

    let variance = if count > 0 {
        sum / f64::from(count)
    } else {
        0.0
    };

    // Noise sigma = sqrt(variance) / sqrt(2)
    let noise = (variance.sqrt() / std::f64::consts::SQRT_2) as f32;

    Ok(noise)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    #[test]
    fn test_noise_estimator_creation() {
        let estimator = NoiseEstimator::new();
        assert_eq!(
            estimator.method,
            NoiseEstimationMethod::MedianAbsoluteDeviation
        );
        assert!(estimator.noise_level().is_none());
    }

    #[test]
    fn test_noise_estimation_mad() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();

        let result = estimate_noise_mad(&frame);
        assert!(result.is_ok());

        let noise = result.expect("noise should be valid");
        assert!(noise >= 0.0);
    }

    #[test]
    fn test_noise_estimation_block_variance() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();

        let result = estimate_noise_block_variance(&frame);
        assert!(result.is_ok());
    }

    #[test]
    fn test_noise_estimation_laplacian() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();

        let result = estimate_noise_laplacian(&frame);
        assert!(result.is_ok());
    }

    #[test]
    fn test_estimator_with_frame() {
        let mut estimator = NoiseEstimator::new();
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();

        let result = estimator.estimate(&frame);
        assert!(result.is_ok());
        assert!(estimator.noise_level().is_some());
    }

    #[test]
    fn test_estimator_reset() {
        let mut estimator = NoiseEstimator::new();
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();

        let _ = estimator.estimate(&frame);
        assert!(estimator.noise_level().is_some());

        estimator.reset();
        assert!(estimator.noise_level().is_none());
    }

    #[test]
    fn test_block_variance() {
        let data = vec![128u8; 64 * 64];
        let variance = compute_block_variance(&data, 0, 0, 8, 64);
        assert!(variance < f32::EPSILON);
    }
}
