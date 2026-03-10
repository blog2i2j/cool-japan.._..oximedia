//! Laplacian pyramid denoising.
//!
//! Multi-scale denoising using Laplacian pyramids to process different
//! frequency bands separately.

use crate::{DenoiseError, DenoiseResult};
use oximedia_codec::VideoFrame;
use rayon::prelude::*;

/// Laplacian pyramid representation.
pub struct LaplacianPyramid {
    /// Pyramid levels (from fine to coarse).
    pub levels: Vec<PyramidLevel>,
    /// Number of levels in pyramid.
    pub num_levels: usize,
}

/// Single level of the pyramid.
#[derive(Clone)]
pub struct PyramidLevel {
    /// Image data at this level.
    pub data: Vec<f32>,
    /// Width at this level.
    pub width: usize,
    /// Height at this level.
    pub height: usize,
}

impl LaplacianPyramid {
    /// Create Laplacian pyramid from image data.
    pub fn new(data: &[u8], width: usize, height: usize, num_levels: usize) -> Self {
        let mut gaussian_pyramid = vec![convert_to_f32(data, width, height)];

        // Build Gaussian pyramid
        for _ in 1..num_levels {
            let Some(prev) = gaussian_pyramid.last() else {
                break;
            };
            let downsampled = downsample(prev);
            gaussian_pyramid.push(downsampled);
        }

        // Build Laplacian pyramid
        let mut laplacian_levels = Vec::new();

        for i in 0..(num_levels - 1) {
            let upsampled = upsample(
                &gaussian_pyramid[i + 1],
                gaussian_pyramid[i].width,
                gaussian_pyramid[i].height,
            );
            let laplacian = subtract_levels(&gaussian_pyramid[i], &upsampled);
            laplacian_levels.push(laplacian);
        }

        // Add the coarsest Gaussian level
        laplacian_levels.push(gaussian_pyramid[num_levels - 1].clone());

        Self {
            levels: laplacian_levels,
            num_levels,
        }
    }

    /// Reconstruct image from Laplacian pyramid.
    pub fn reconstruct(&self) -> Vec<u8> {
        if self.levels.is_empty() {
            return Vec::new();
        }

        // Start with coarsest level
        let mut result = self.levels[self.num_levels - 1].clone();

        // Reconstruct from coarse to fine
        for i in (0..(self.num_levels - 1)).rev() {
            let upsampled = upsample(&result, self.levels[i].width, self.levels[i].height);
            result = add_levels(&upsampled, &self.levels[i]);
        }

        convert_to_u8(&result)
    }

    /// Apply denoising to each pyramid level.
    pub fn denoise(&mut self, strength: f32) {
        self.levels
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, level)| {
                // Apply stronger denoising to finer levels
                let level_strength = strength * (1.0 - i as f32 / self.num_levels as f32);
                denoise_level(level, level_strength);
            });
    }
}

/// Convert u8 data to f32 pyramid level.
fn convert_to_f32(data: &[u8], width: usize, height: usize) -> PyramidLevel {
    PyramidLevel {
        data: data.iter().map(|&x| f32::from(x)).collect(),
        width,
        height,
    }
}

/// Convert f32 pyramid level to u8 data.
fn convert_to_u8(level: &PyramidLevel) -> Vec<u8> {
    level
        .data
        .iter()
        .map(|&x| x.round().clamp(0.0, 255.0) as u8)
        .collect()
}

/// Downsample an image by 2x using Gaussian filtering.
fn downsample(level: &PyramidLevel) -> PyramidLevel {
    let new_width = level.width / 2;
    let new_height = level.height / 2;
    let mut data = vec![0.0f32; new_width * new_height];

    // Simple 2x2 averaging
    for y in 0..new_height {
        for x in 0..new_width {
            let src_x = x * 2;
            let src_y = y * 2;

            let sum = level.data[src_y * level.width + src_x]
                + level.data[src_y * level.width + src_x + 1]
                + level.data[(src_y + 1) * level.width + src_x]
                + level.data[(src_y + 1) * level.width + src_x + 1];

            data[y * new_width + x] = sum / 4.0;
        }
    }

    PyramidLevel {
        data,
        width: new_width,
        height: new_height,
    }
}

/// Upsample an image by 2x with interpolation.
fn upsample(level: &PyramidLevel, target_width: usize, target_height: usize) -> PyramidLevel {
    let mut data = vec![0.0f32; target_width * target_height];

    for y in 0..target_height {
        for x in 0..target_width {
            let src_x = (x as f32 / 2.0).floor() as usize;
            let src_y = (y as f32 / 2.0).floor() as usize;

            let src_x = src_x.min(level.width - 1);
            let src_y = src_y.min(level.height - 1);

            data[y * target_width + x] = level.data[src_y * level.width + src_x];
        }
    }

    PyramidLevel {
        data,
        width: target_width,
        height: target_height,
    }
}

/// Subtract two pyramid levels.
fn subtract_levels(a: &PyramidLevel, b: &PyramidLevel) -> PyramidLevel {
    assert_eq!(a.width, b.width);
    assert_eq!(a.height, b.height);

    let data: Vec<f32> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&x, &y)| x - y)
        .collect();

    PyramidLevel {
        data,
        width: a.width,
        height: a.height,
    }
}

/// Add two pyramid levels.
fn add_levels(a: &PyramidLevel, b: &PyramidLevel) -> PyramidLevel {
    assert_eq!(a.width, b.width);
    assert_eq!(a.height, b.height);

    let data: Vec<f32> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&x, &y)| x + y)
        .collect();

    PyramidLevel {
        data,
        width: a.width,
        height: a.height,
    }
}

/// Denoise a single pyramid level using soft thresholding.
fn denoise_level(level: &mut PyramidLevel, strength: f32) {
    let threshold = strength * 10.0;

    for val in &mut level.data {
        if val.abs() < threshold {
            *val *= 1.0 - (threshold - val.abs()) / threshold;
        }
    }
}

/// Apply Laplacian pyramid denoising to a video frame.
pub fn pyramid_denoise(
    frame: &VideoFrame,
    strength: f32,
    num_levels: usize,
) -> DenoiseResult<VideoFrame> {
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

            // Build pyramid
            let mut pyramid = LaplacianPyramid::new(
                input_plane.data.as_ref(),
                width as usize,
                height as usize,
                num_levels,
            );

            // Denoise each level
            pyramid.denoise(strength);

            // Reconstruct
            let denoised = pyramid.reconstruct();

            // Update plane data
            plane.data = denoised;
            Ok::<(), DenoiseError>(())
        })?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    #[test]
    fn test_pyramid_construction() {
        let data = vec![128u8; 64 * 64];
        let pyramid = LaplacianPyramid::new(&data, 64, 64, 3);
        assert_eq!(pyramid.num_levels, 3);
        assert_eq!(pyramid.levels.len(), 3);
    }

    #[test]
    fn test_pyramid_reconstruction() {
        let data = vec![128u8; 64 * 64];
        let pyramid = LaplacianPyramid::new(&data, 64, 64, 3);
        let reconstructed = pyramid.reconstruct();
        assert_eq!(reconstructed.len(), 64 * 64);
    }

    #[test]
    fn test_pyramid_denoise() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();

        let result = pyramid_denoise(&frame, 0.5, 3);
        assert!(result.is_ok());
    }

    #[test]
    fn test_downsample_upsample() {
        let level = PyramidLevel {
            data: vec![128.0; 64 * 64],
            width: 64,
            height: 64,
        };

        let down = downsample(&level);
        assert_eq!(down.width, 32);
        assert_eq!(down.height, 32);

        let up = upsample(&down, 64, 64);
        assert_eq!(up.width, 64);
        assert_eq!(up.height, 64);
    }
}
