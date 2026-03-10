//! Utility functions for super-resolution.

use super::types::UpscaleFactor;

/// Calculate optimal tile size based on image dimensions and available memory.
///
/// # Arguments
///
/// * `width` - Image width
/// * `height` - Image height
/// * `available_memory_mb` - Available memory in megabytes
///
/// # Returns
///
/// Recommended tile size
#[must_use]
pub fn calculate_optimal_tile_size(width: u32, height: u32, available_memory_mb: usize) -> u32 {
    // Rough estimation: float32 tensor needs 4 bytes per value
    // For a tile of size NxN with 3 channels and scale factor 4:
    // Input: N * N * 3 * 4 bytes
    // Output: (N*4) * (N*4) * 3 * 4 bytes
    // Total ≈ N^2 * 12 + N^2 * 64 * 12 ≈ N^2 * 780 bytes

    let bytes_per_pixel_approx = 780;
    let available_bytes = available_memory_mb * 1024 * 1024;
    let max_tile_pixels = available_bytes / bytes_per_pixel_approx;
    let max_tile_size = (max_tile_pixels as f32).sqrt() as u32;

    // Clamp to reasonable range
    let tile_size = max_tile_size.clamp(128, 1024);

    // Round down to nearest power of 2 for efficiency
    let tile_size = tile_size.next_power_of_two() / 2;

    tile_size.min(width.max(height))
}

/// Estimate memory requirement for upscaling an image.
///
/// # Arguments
///
/// * `width` - Image width
/// * `height` - Image height
/// * `scale_factor` - Upscale factor
///
/// # Returns
///
/// Estimated memory in megabytes
#[must_use]
pub fn estimate_memory_requirement(width: u32, height: u32, scale_factor: UpscaleFactor) -> usize {
    let scale = scale_factor.scale();
    let input_pixels = width as usize * height as usize;
    let output_pixels = (width * scale) as usize * (height * scale) as usize;

    // Input tensor (float32): pixels * 3 * 4
    let input_bytes = input_pixels * 3 * 4;
    // Output tensor (float32): pixels * 3 * 4
    let output_bytes = output_pixels * 3 * 4;
    // Output RGB u8: pixels * 3
    let result_bytes = output_pixels * 3;

    let total_bytes = input_bytes + output_bytes + result_bytes;
    total_bytes.div_ceil(1024 * 1024) // Round up to MB
}

/// Create a feathering weight map for blending.
///
/// # Arguments
///
/// * `width` - Map width
/// * `height` - Map height
/// * `feather_width` - Feathering width in pixels
///
/// # Returns
///
/// Weight map where edges fade from 0 to 1
#[must_use]
pub fn create_feather_weights(width: u32, height: u32, feather_width: u32) -> Vec<f32> {
    let mut weights = vec![1.0f32; (width * height) as usize];

    for y in 0..height {
        for x in 0..width {
            let dist_left = x;
            let dist_right = width - x - 1;
            let dist_top = y;
            let dist_bottom = height - y - 1;

            let min_dist = dist_left.min(dist_right).min(dist_top).min(dist_bottom);

            let weight = if min_dist >= feather_width {
                1.0
            } else {
                (min_dist as f32 + 1.0) / (feather_width as f32 + 1.0)
            };

            weights[(y * width + x) as usize] = weight;
        }
    }

    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_optimal_tile_size() {
        let tile_size = calculate_optimal_tile_size(2048, 2048, 512);
        assert!(tile_size >= 128);
        assert!(tile_size <= 1024);
    }

    #[test]
    fn test_estimate_memory_requirement() {
        let mem = estimate_memory_requirement(1920, 1080, UpscaleFactor::X4);
        assert!(mem > 0);
        // For 1920x1080 -> 7680x4320, should be around 300-400 MB
        assert!(mem > 100);
        assert!(mem < 1000);
    }

    #[test]
    fn test_create_feather_weights() {
        let weights = create_feather_weights(10, 10, 2);
        assert_eq!(weights.len(), 100);

        // Center should be 1.0
        assert_eq!(weights[5 * 10 + 5], 1.0);

        // Corner should be less than 1.0
        assert!(weights[0] < 1.0);
    }
}
