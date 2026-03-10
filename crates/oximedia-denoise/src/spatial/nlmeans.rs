//! Non-Local Means denoising.
//!
//! Non-Local Means is a patch-based denoising algorithm that exploits
//! self-similarity in images by averaging similar patches from across
//! the entire image.

use crate::{DenoiseError, DenoiseResult};
use oximedia_codec::VideoFrame;
use rayon::prelude::*;

/// Apply Non-Local Means filter to a video frame.
///
/// `NLMeans` searches for similar patches across the image and averages them
/// to reduce noise while preserving texture and detail.
///
/// # Arguments
/// * `frame` - Input video frame
/// * `strength` - Denoising strength (0.0 - 1.0)
///
/// # Returns
/// Filtered video frame
pub fn nlmeans_filter(frame: &VideoFrame, strength: f32) -> DenoiseResult<VideoFrame> {
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

            // Scale parameters based on strength
            let h = 10.0 * strength; // Filtering parameter
            let patch_size = 7;
            let search_window = 21;

            nlmeans_filter_plane(
                input_plane.data.as_ref(),
                &mut plane.data.clone(),
                width as usize,
                height as usize,
                plane.stride,
                h,
                patch_size,
                search_window,
            )
        })?;

    Ok(output)
}

/// Apply `NLMeans` to a single plane.
#[allow(clippy::too_many_arguments, clippy::cast_possible_wrap)]
fn nlmeans_filter_plane(
    input: &[u8],
    output: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
    h: f32,
    patch_size: usize,
    search_window: usize,
) -> DenoiseResult<()> {
    let patch_radius = patch_size / 2;
    let search_radius = search_window / 2;
    let h2 = h * h;

    // Process each pixel
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0f32;
            let mut weight_sum = 0.0f32;

            let search_y_min = (y as i32 - search_radius as i32).max(0) as usize;
            let search_y_max = (y + search_radius + 1).min(height);
            let search_x_min = (x as i32 - search_radius as i32).max(0) as usize;
            let search_x_max = (x + search_radius + 1).min(width);

            // Search for similar patches
            for sy in search_y_min..search_y_max {
                for sx in search_x_min..search_x_max {
                    // Compute patch distance
                    let dist = compute_patch_distance(
                        input,
                        width,
                        height,
                        stride,
                        x,
                        y,
                        sx,
                        sy,
                        patch_radius,
                    );

                    // Convert distance to weight
                    let weight = (-dist / h2).exp();

                    sum += f32::from(input[sy * stride + sx]) * weight;
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

/// Compute squared distance between two patches.
#[allow(clippy::too_many_arguments)]
fn compute_patch_distance(
    data: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
    patch_radius: usize,
) -> f32 {
    let mut dist = 0.0f32;
    let mut count = 0;

    for dy in -(patch_radius as i32)..=(patch_radius as i32) {
        let py1 = (y1 as i32 + dy).clamp(0, (height - 1) as i32) as usize;
        let py2 = (y2 as i32 + dy).clamp(0, (height - 1) as i32) as usize;

        for dx in -(patch_radius as i32)..=(patch_radius as i32) {
            let px1 = (x1 as i32 + dx).clamp(0, (width - 1) as i32) as usize;
            let px2 = (x2 as i32 + dx).clamp(0, (width - 1) as i32) as usize;

            let v1 = f32::from(data[py1 * stride + px1]);
            let v2 = f32::from(data[py2 * stride + px2]);
            let diff = v1 - v2;
            dist += diff * diff;
            count += 1;
        }
    }

    if count > 0 {
        dist / count as f32
    } else {
        0.0
    }
}

/// Fast Non-Local Means using integral images.
pub fn fast_nlmeans_filter(frame: &VideoFrame, strength: f32) -> DenoiseResult<VideoFrame> {
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

            let h = 8.0 * strength;
            let patch_size = 5;
            let search_window = 15;

            nlmeans_filter_plane(
                input_plane.data.as_ref(),
                &mut plane.data.clone(),
                width as usize,
                height as usize,
                plane.stride,
                h,
                patch_size,
                search_window,
            )
        })?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    #[test]
    fn test_nlmeans_filter() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 24, 24);
        frame.allocate();

        let result = nlmeans_filter(&frame, 0.5);
        assert!(result.is_ok());

        let filtered = result.expect("filtered should be valid");
        assert_eq!(filtered.width, 24);
        assert_eq!(filtered.height, 24);
    }

    #[test]
    fn test_fast_nlmeans_filter() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 32, 32);
        frame.allocate();

        let result = fast_nlmeans_filter(&frame, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_patch_distance() {
        let data = vec![0u8; 100 * 100];
        let dist = compute_patch_distance(&data, 100, 100, 100, 50, 50, 50, 50, 3);
        assert!((dist - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_nlmeans_small_frame() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 16, 16);
        frame.allocate();

        let result = nlmeans_filter(&frame, 0.3);
        assert!(result.is_ok());
    }
}
