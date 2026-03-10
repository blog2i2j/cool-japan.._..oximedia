//! Wiener filter for frequency-domain denoising.
//!
//! The Wiener filter is an optimal linear filter that minimizes the mean
//! square error between the estimated and true signal in the frequency domain.

use crate::{DenoiseError, DenoiseResult};
use oximedia_codec::VideoFrame;
use rayon::prelude::*;

/// Apply Wiener filter to a video frame.
///
/// The Wiener filter operates in the frequency domain and is particularly
/// effective for signals with known or estimated power spectral density.
///
/// # Arguments
/// * `frame` - Input video frame
/// * `strength` - Denoising strength (0.0 - 1.0)
///
/// # Returns
/// Filtered video frame
pub fn wiener_filter(frame: &VideoFrame, strength: f32) -> DenoiseResult<VideoFrame> {
    if frame.planes.is_empty() {
        return Err(DenoiseError::ProcessingError(
            "Frame has no planes".to_string(),
        ));
    }

    let mut output = frame.clone();

    // Process each plane in parallel
    output
        .planes
        .par_iter_mut()
        .enumerate()
        .try_for_each(|(plane_idx, plane)| {
            let input_plane = &frame.planes[plane_idx];
            let (width, height) = frame.plane_dimensions(plane_idx);

            // Estimate noise variance based on strength
            let noise_variance = (strength * 100.0).powi(2);

            wiener_filter_plane(
                input_plane.data.as_ref(),
                &mut plane.data.clone(),
                width as usize,
                height as usize,
                plane.stride,
                noise_variance,
            )
        })?;

    Ok(output)
}

/// Apply Wiener filter to a single plane using local statistics.
fn wiener_filter_plane(
    input: &[u8],
    output: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
    noise_variance: f32,
) -> DenoiseResult<()> {
    let window_size = 5;
    let window_radius = window_size / 2;

    for y in 0..height {
        for x in 0..width {
            // Compute local mean and variance
            let (local_mean, local_variance) =
                compute_local_statistics(input, width, height, stride, x, y, window_radius);

            let center_val = f32::from(input[y * stride + x]);

            // Wiener filter formula
            let signal_variance = (local_variance - noise_variance).max(0.0);
            let wiener_gain = if local_variance > 0.0 {
                signal_variance / local_variance
            } else {
                0.0
            };

            let filtered = local_mean + wiener_gain * (center_val - local_mean);
            output[y * stride + x] = filtered.round().clamp(0.0, 255.0) as u8;
        }
    }

    Ok(())
}

/// Compute local mean and variance in a window.
fn compute_local_statistics(
    data: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    x: usize,
    y: usize,
    radius: usize,
) -> (f32, f32) {
    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;
    let mut count = 0;

    for dy in -(radius as i32)..=(radius as i32) {
        let ny = (y as i32 + dy).clamp(0, (height - 1) as i32) as usize;

        for dx in -(radius as i32)..=(radius as i32) {
            let nx = (x as i32 + dx).clamp(0, (width - 1) as i32) as usize;

            let val = f32::from(data[ny * stride + nx]);
            sum += val;
            sum_sq += val * val;
            count += 1;
        }
    }

    let mean = sum / count as f32;
    let variance = (sum_sq / count as f32) - (mean * mean);

    (mean, variance.max(0.0))
}

/// Adaptive Wiener filter with automatic noise estimation.
pub fn adaptive_wiener_filter(frame: &VideoFrame, strength: f32) -> DenoiseResult<VideoFrame> {
    if frame.planes.is_empty() {
        return Err(DenoiseError::ProcessingError(
            "Frame has no planes".to_string(),
        ));
    }

    let mut output = frame.clone();

    output
        .planes
        .par_iter_mut()
        .enumerate()
        .try_for_each(|(plane_idx, plane)| {
            let input_plane = &frame.planes[plane_idx];
            let (width, height) = frame.plane_dimensions(plane_idx);

            // Estimate noise from high-frequency components
            let estimated_noise = estimate_noise_from_plane(
                input_plane.data.as_ref(),
                width as usize,
                height as usize,
                plane.stride,
            );

            let noise_variance = (estimated_noise * strength).powi(2);

            wiener_filter_plane(
                input_plane.data.as_ref(),
                &mut plane.data.clone(),
                width as usize,
                height as usize,
                plane.stride,
                noise_variance,
            )
        })?;

    Ok(output)
}

/// Estimate noise level from high-frequency components.
fn estimate_noise_from_plane(data: &[u8], width: usize, height: usize, stride: usize) -> f32 {
    let mut sum_abs_diff = 0.0f32;
    let mut count = 0;

    // Use horizontal differences as noise proxy
    for y in 0..height {
        for x in 0..(width - 1) {
            let diff = f32::from(data[y * stride + x + 1]) - f32::from(data[y * stride + x]);
            sum_abs_diff += diff.abs();
            count += 1;
        }
    }

    if count > 0 {
        sum_abs_diff / count as f32
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    #[test]
    fn test_wiener_filter() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();

        let result = wiener_filter(&frame, 0.5);
        assert!(result.is_ok());

        let filtered = result.expect("filtered should be valid");
        assert_eq!(filtered.width, 64);
        assert_eq!(filtered.height, 64);
    }

    #[test]
    fn test_adaptive_wiener_filter() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();

        let result = adaptive_wiener_filter(&frame, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_local_statistics() {
        let data = vec![128u8; 100 * 100];
        let (mean, variance) = compute_local_statistics(&data, 100, 100, 100, 50, 50, 2);
        assert!((mean - 128.0).abs() < f32::EPSILON);
        assert!(variance < f32::EPSILON);
    }

    #[test]
    fn test_noise_estimation() {
        let data = vec![100u8; 64 * 64];
        let noise = estimate_noise_from_plane(&data, 64, 64, 64);
        assert!(noise < f32::EPSILON);
    }
}
