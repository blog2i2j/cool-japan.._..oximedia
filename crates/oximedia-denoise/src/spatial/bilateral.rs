//! Bilateral filter for edge-preserving spatial denoising.
//!
//! The bilateral filter is a non-linear, edge-preserving smoothing filter
//! that combines spatial and range (intensity) Gaussian kernels.

use crate::{DenoiseError, DenoiseResult};
use oximedia_codec::VideoFrame;
use rayon::prelude::*;

/// Apply bilateral filter to a video frame.
///
/// The bilateral filter preserves edges while smoothing uniform regions.
/// It uses both spatial proximity and intensity similarity to determine weights.
///
/// # Arguments
/// * `frame` - Input video frame
/// * `strength` - Denoising strength (0.0 - 1.0)
///
/// # Returns
/// Filtered video frame
pub fn bilateral_filter(frame: &VideoFrame, strength: f32) -> DenoiseResult<VideoFrame> {
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

            // Convert strength to filter parameters
            let sigma_space = 3.0 * strength;
            let sigma_range = 25.0 * strength;

            bilateral_filter_plane(
                input_plane.data.as_ref(),
                &mut plane.data.clone(),
                width as usize,
                height as usize,
                plane.stride,
                sigma_space,
                sigma_range,
            )
        })?;

    Ok(output)
}

/// Apply bilateral filter to a single plane.
#[allow(clippy::too_many_arguments)]
fn bilateral_filter_plane(
    input: &[u8],
    output: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
    sigma_space: f32,
    sigma_range: f32,
) -> DenoiseResult<()> {
    let kernel_radius = (3.0 * sigma_space).ceil() as i32;
    let space_coeff = -0.5 / (sigma_space * sigma_space);
    let range_coeff = -0.5 / (sigma_range * sigma_range);

    // Precompute spatial weights
    let mut spatial_weights = vec![0.0f32; (2 * kernel_radius + 1) as usize];
    for i in 0..spatial_weights.len() {
        let d = (i as i32 - kernel_radius) as f32;
        spatial_weights[i] = (d * d * space_coeff).exp();
    }

    // Process each pixel
    for y in 0..height {
        for x in 0..width {
            let center_val = f32::from(input[y * stride + x]);
            let mut sum = 0.0f32;
            let mut weight_sum = 0.0f32;

            // Apply kernel
            for ky in -kernel_radius..=kernel_radius {
                let ny = (y as i32 + ky).clamp(0, (height - 1) as i32) as usize;

                for kx in -kernel_radius..=kernel_radius {
                    let nx = (x as i32 + kx).clamp(0, (width - 1) as i32) as usize;

                    let neighbor_val = f32::from(input[ny * stride + nx]);
                    let value_diff = neighbor_val - center_val;

                    // Combine spatial and range weights
                    let spatial_weight = spatial_weights[(ky + kernel_radius) as usize]
                        * spatial_weights[(kx + kernel_radius) as usize];
                    let range_weight = (value_diff * value_diff * range_coeff).exp();
                    let weight = spatial_weight * range_weight;

                    sum += neighbor_val * weight;
                    weight_sum += weight;
                }
            }

            output[y * stride + x] = if weight_sum > 0.0 {
                (sum / weight_sum).round().clamp(0.0, 255.0) as u8
            } else {
                input[y * stride + x]
            };
        }
    }

    Ok(())
}

/// Fast bilateral filter approximation using separable kernels.
pub fn fast_bilateral_filter(frame: &VideoFrame, strength: f32) -> DenoiseResult<VideoFrame> {
    if frame.planes.is_empty() {
        return Err(DenoiseError::ProcessingError(
            "Frame has no planes".to_string(),
        ));
    }

    let mut output = frame.clone();

    // Process each plane with fast approximation
    output
        .planes
        .par_iter_mut()
        .enumerate()
        .try_for_each(|(plane_idx, plane)| {
            let input_plane = &frame.planes[plane_idx];
            let (width, height) = frame.plane_dimensions(plane_idx);

            let sigma_space = 2.0 * strength;
            let sigma_range = 20.0 * strength;

            fast_bilateral_filter_plane(
                input_plane.data.as_ref(),
                &mut plane.data.clone(),
                width as usize,
                height as usize,
                plane.stride,
                sigma_space,
                sigma_range,
            )
        })?;

    Ok(output)
}

fn fast_bilateral_filter_plane(
    input: &[u8],
    output: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
    sigma_space: f32,
    sigma_range: f32,
) -> DenoiseResult<()> {
    let kernel_radius = (2.0 * sigma_space).ceil() as i32;
    let range_coeff = -0.5 / (sigma_range * sigma_range);

    // Simplified box filter approximation
    for y in 0..height {
        for x in 0..width {
            let center_val = f32::from(input[y * stride + x]);
            let mut sum = 0.0f32;
            let mut weight_sum = 0.0f32;

            for ky in -kernel_radius..=kernel_radius {
                let ny = (y as i32 + ky).clamp(0, (height - 1) as i32) as usize;

                for kx in -kernel_radius..=kernel_radius {
                    let nx = (x as i32 + kx).clamp(0, (width - 1) as i32) as usize;

                    let neighbor_val = f32::from(input[ny * stride + nx]);
                    let value_diff = neighbor_val - center_val;
                    let weight = (value_diff * value_diff * range_coeff).exp();

                    sum += neighbor_val * weight;
                    weight_sum += weight;
                }
            }

            output[y * stride + x] = if weight_sum > 0.0 {
                (sum / weight_sum).round().clamp(0.0, 255.0) as u8
            } else {
                input[y * stride + x]
            };
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    #[test]
    fn test_bilateral_filter() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();

        let result = bilateral_filter(&frame, 0.5);
        assert!(result.is_ok());

        let filtered = result.expect("filtered should be valid");
        assert_eq!(filtered.width, 64);
        assert_eq!(filtered.height, 64);
    }

    #[test]
    fn test_fast_bilateral_filter() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();

        let result = fast_bilateral_filter(&frame, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_bilateral_zero_strength() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 32, 32);
        frame.allocate();

        let result = bilateral_filter(&frame, 0.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_bilateral_max_strength() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 32, 32);
        frame.allocate();

        let result = bilateral_filter(&frame, 1.0);
        assert!(result.is_ok());
    }
}
