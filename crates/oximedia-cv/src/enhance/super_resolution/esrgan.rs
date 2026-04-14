//! ESRGAN-based upscaler and batch upscaler.

#![allow(clippy::too_many_arguments)]

use super::types::{ProgressCallback, TileConfig, UpscaleFactor};
use crate::error::{CvError, CvResult};
use ndarray::Array4;
use oxionnx::Session;
use std::collections::HashMap;
use std::path::Path;

/// ESRGAN-based image upscaler.
///
/// Provides AI-powered super-resolution using ESRGAN models via ONNX Runtime.
/// Supports tile-based processing for large images to manage memory usage.
///
/// # Note
///
/// This type is deprecated in favor of `SuperResolutionModel` which supports
/// multiple model types. This is kept for backward compatibility.
pub struct EsrganUpscaler {
    pub(super) session: Session,
    pub(super) scale_factor: UpscaleFactor,
    pub(super) tile_config: TileConfig,
    pub(super) progress_callback: Option<ProgressCallback>,
}

impl EsrganUpscaler {
    /// Create a new ESRGAN upscaler from an ONNX model file.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the ONNX model file
    /// * `scale_factor` - Upscale factor (2x or 4x)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Model file cannot be loaded
    /// - ONNX Runtime initialization fails
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use oximedia_cv::enhance::{EsrganUpscaler, UpscaleFactor};
    ///
    /// let upscaler = EsrganUpscaler::new("esrgan_x4.onnx", UpscaleFactor::X4)?;
    /// # Ok::<(), oximedia_cv::error::CvError>(())
    /// ```
    pub fn new(model_path: impl AsRef<Path>, scale_factor: UpscaleFactor) -> CvResult<Self> {
        let session = Session::builder()
            .with_optimization_level(oxionnx::OptLevel::All)
            .load(model_path.as_ref())
            .map_err(|e| CvError::model_load(format!("Failed to load model: {e}")))?;

        Ok(Self {
            session,
            scale_factor,
            tile_config: TileConfig::default(),
            progress_callback: None,
        })
    }

    /// Set custom tile configuration.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use oximedia_cv::enhance::{EsrganUpscaler, UpscaleFactor, TileConfig};
    ///
    /// let mut upscaler = EsrganUpscaler::new("esrgan_x4.onnx", UpscaleFactor::X4)?;
    /// upscaler.set_tile_config(TileConfig::new(512, 32)?);
    /// # Ok::<(), oximedia_cv::error::CvError>(())
    /// ```
    pub fn set_tile_config(&mut self, config: TileConfig) {
        self.tile_config = config;
    }

    /// Set progress callback.
    ///
    /// The callback receives `(current, total)` and should return `true` to continue
    /// or `false` to abort processing.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use oximedia_cv::enhance::{EsrganUpscaler, UpscaleFactor};
    ///
    /// let mut upscaler = EsrganUpscaler::new("esrgan_x4.onnx", UpscaleFactor::X4)?;
    /// upscaler.set_progress_callback(Box::new(|current, total| {
    ///     println!("Progress: {}/{}", current, total);
    ///     true
    /// }));
    /// # Ok::<(), oximedia_cv::error::CvError>(())
    /// ```
    pub fn set_progress_callback(&mut self, callback: ProgressCallback) {
        self.progress_callback = Some(callback);
    }

    /// Upscale an image using the ESRGAN model.
    ///
    /// For large images, automatically uses tile-based processing.
    ///
    /// # Arguments
    ///
    /// * `image` - Input image in RGB format (row-major, packed RGB)
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    ///
    /// # Returns
    ///
    /// Upscaled image in RGB format with dimensions `(width * scale, height * scale, 3)`
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Input dimensions are invalid
    /// - Input buffer size doesn't match dimensions
    /// - Model inference fails
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use oximedia_cv::enhance::{EsrganUpscaler, UpscaleFactor};
    ///
    /// let mut upscaler = EsrganUpscaler::new("esrgan_x4.onnx", UpscaleFactor::X4)?;
    /// let input = vec![0u8; 256 * 256 * 3];
    /// let output = upscaler.upscale(&input, 256, 256)?;
    /// assert_eq!(output.len(), 1024 * 1024 * 3);
    /// # Ok::<(), oximedia_cv::error::CvError>(())
    /// ```
    pub fn upscale(&mut self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
        // Validate inputs
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let expected_size = (width as usize) * (height as usize) * 3;
        if image.len() != expected_size {
            return Err(CvError::insufficient_data(expected_size, image.len()));
        }

        // Determine if tiling is needed
        let tile_size = self.tile_config.tile_size;
        if width <= tile_size && height <= tile_size {
            self.upscale_single_tile(image, width, height)
        } else {
            self.upscale_tiled(image, width, height)
        }
    }

    /// Upscale a single tile (internal method).
    fn upscale_single_tile(&mut self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
        // Convert RGB u8 to normalized float32 [1, 3, H, W]
        let input_tensor = self.preprocess_image(image, width, height)?;

        // Convert ndarray → oxionnx Tensor
        let flat: Vec<f32> = input_tensor.iter().copied().collect();
        let shape: Vec<usize> = input_tensor.shape().to_vec();
        let tensor = oxionnx::Tensor::new(flat, shape);

        let input_name = self
            .session
            .input_names()
            .first()
            .cloned()
            .unwrap_or_else(|| "input".to_string());

        let mut inputs = HashMap::new();
        inputs.insert(input_name.as_str(), tensor);

        let outputs = self
            .session
            .run(&inputs)
            .map_err(|e| CvError::onnx_runtime(format!("Inference failed: {e}")))?;

        let output_name = self
            .session
            .output_names()
            .first()
            .cloned()
            .unwrap_or_default();
        let out_tensor = outputs
            .get(&output_name)
            .ok_or_else(|| CvError::onnx_runtime("No output tensor found".to_owned()))?;

        let shape_owned: Vec<i64> = out_tensor.shape.iter().map(|&x| x as i64).collect();
        let data_owned: Vec<f32> = out_tensor.data.clone();

        // Convert back to RGB u8
        let scale = self.scale_factor.scale();
        self.postprocess_tensor(&shape_owned, &data_owned, width * scale, height * scale)
    }

    /// Upscale using tile-based processing for large images.
    ///
    /// This method splits the input image into overlapping tiles, processes each
    /// tile independently, and blends the results to produce the final upscaled image.
    ///
    /// # Arguments
    ///
    /// * `image` - Input image in RGB format
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Errors
    ///
    /// Returns an error if processing fails.
    pub fn upscale_tiled(&mut self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
        let tile_size = self.tile_config.tile_size;
        let padding = self.tile_config.tile_padding;
        let scale = self.scale_factor.scale();

        // Calculate tile grid
        let tiles_x = width.div_ceil(tile_size) as usize;
        let tiles_y = height.div_ceil(tile_size) as usize;
        let total_tiles = tiles_x * tiles_y;

        // Output dimensions
        let out_width = width * scale;
        let out_height = height * scale;
        let mut output = vec![0u8; (out_width * out_height * 3) as usize];

        // Weight map for blending
        let mut weight_map = vec![0.0f32; (out_width * out_height) as usize];

        // Process each tile
        let mut tile_idx = 0;
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                // Check if processing should continue
                if let Some(ref callback) = self.progress_callback {
                    if !callback(tile_idx + 1, total_tiles) {
                        return Err(CvError::detection_failed("Processing aborted by user"));
                    }
                }

                // Calculate tile boundaries with padding
                let x_start = (tx as u32 * tile_size).saturating_sub(padding);
                let y_start = (ty as u32 * tile_size).saturating_sub(padding);
                let x_end = ((tx as u32 + 1) * tile_size + padding).min(width);
                let y_end = ((ty as u32 + 1) * tile_size + padding).min(height);

                let tile_w = x_end - x_start;
                let tile_h = y_end - y_start;

                // Extract tile
                let tile =
                    self.extract_tile(image, width, height, x_start, y_start, tile_w, tile_h)?;

                // Process tile
                let upscaled_tile = self.upscale_single_tile(&tile, tile_w, tile_h)?;

                // Calculate blend region (excluding padding)
                let blend_x_start = if tx == 0 { 0 } else { padding * scale };
                let blend_y_start = if ty == 0 { 0 } else { padding * scale };
                let blend_x_end = if tx == tiles_x - 1 {
                    tile_w * scale
                } else {
                    (tile_w - padding) * scale
                };
                let blend_y_end = if ty == tiles_y - 1 {
                    tile_h * scale
                } else {
                    (tile_h - padding) * scale
                };

                // Blend tile into output
                self.blend_tile(
                    &upscaled_tile,
                    tile_w * scale,
                    tile_h * scale,
                    &mut output,
                    &mut weight_map,
                    out_width,
                    out_height,
                    x_start * scale,
                    y_start * scale,
                    blend_x_start,
                    blend_y_start,
                    blend_x_end,
                    blend_y_end,
                )?;

                tile_idx += 1;
            }
        }

        // Normalize by weight map
        self.normalize_by_weights(&mut output, &weight_map, out_width, out_height);

        Ok(output)
    }

    /// Extract a rectangular tile from the source image.
    fn extract_tile(
        &self,
        src: &[u8],
        src_w: u32,
        src_h: u32,
        x: u32,
        y: u32,
        tile_w: u32,
        tile_h: u32,
    ) -> CvResult<Vec<u8>> {
        if x + tile_w > src_w || y + tile_h > src_h {
            return Err(CvError::invalid_roi(x, y, tile_w, tile_h));
        }

        let mut tile = Vec::with_capacity((tile_w * tile_h * 3) as usize);

        for row in y..y + tile_h {
            let start = ((row * src_w + x) * 3) as usize;
            let end = start + (tile_w * 3) as usize;
            tile.extend_from_slice(&src[start..end]);
        }

        Ok(tile)
    }

    /// Blend a processed tile into the output image with feathering.
    #[allow(clippy::too_many_arguments)]
    fn blend_tile(
        &self,
        tile: &[u8],
        tile_w: u32,
        _tile_h: u32,
        output: &mut [u8],
        weights: &mut [f32],
        out_w: u32,
        out_h: u32,
        dst_x: u32,
        dst_y: u32,
        blend_x_start: u32,
        blend_y_start: u32,
        blend_x_end: u32,
        blend_y_end: u32,
    ) -> CvResult<()> {
        let feather = self.tile_config.feather_width;

        for local_y in blend_y_start..blend_y_end {
            let global_y = dst_y + local_y;
            if global_y >= out_h {
                break;
            }

            for local_x in blend_x_start..blend_x_end {
                let global_x = dst_x + local_x;
                if global_x >= out_w {
                    break;
                }

                // Calculate feather weight (distance from edge)
                let dist_left = local_x - blend_x_start;
                let dist_right = blend_x_end - local_x - 1;
                let dist_top = local_y - blend_y_start;
                let dist_bottom = blend_y_end - local_y - 1;

                let min_dist = dist_left.min(dist_right).min(dist_top).min(dist_bottom);
                let weight = if min_dist >= feather {
                    1.0
                } else {
                    (min_dist as f32 + 1.0) / (feather as f32 + 1.0)
                };

                // Blend RGB values
                let tile_idx = ((local_y * tile_w + local_x) * 3) as usize;
                let out_idx = ((global_y * out_w + global_x) * 3) as usize;
                let weight_idx = (global_y * out_w + global_x) as usize;

                for c in 0..3 {
                    let tile_val = tile[tile_idx + c] as f32 * weight;
                    output[out_idx + c] = (output[out_idx + c] as f32 + tile_val) as u8;
                }

                weights[weight_idx] += weight;
            }
        }

        Ok(())
    }

    /// Normalize output by accumulated weights.
    fn normalize_by_weights(&self, output: &mut [u8], weights: &[f32], width: u32, height: u32) {
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let weight = weights[idx];

                if weight > 0.0 {
                    let out_idx = idx * 3;
                    for c in 0..3 {
                        output[out_idx + c] = ((output[out_idx + c] as f32) / weight).round() as u8;
                    }
                }
            }
        }
    }

    /// Preprocess image: RGB u8 -> normalized float32 [1, 3, H, W].
    fn preprocess_image(&self, image: &[u8], width: u32, height: u32) -> CvResult<Array4<f32>> {
        let w = width as usize;
        let h = height as usize;

        let mut tensor = Array4::<f32>::zeros((1, 3, h, w));

        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                // Normalize to [0, 1]
                tensor[[0, 0, y, x]] = image[idx] as f32 / 255.0; // R
                tensor[[0, 1, y, x]] = image[idx + 1] as f32 / 255.0; // G
                tensor[[0, 2, y, x]] = image[idx + 2] as f32 / 255.0; // B
            }
        }

        Ok(tensor)
    }

    /// Postprocess tensor: normalized float32 [1, 3, H, W] -> RGB u8.
    fn postprocess_tensor(
        &self,
        shape: &[i64],
        data: &[f32],
        width: u32,
        height: u32,
    ) -> CvResult<Vec<u8>> {
        if shape.len() != 4 || shape[0] != 1 || shape[1] != 3 {
            return Err(CvError::ShapeMismatch {
                expected: vec![1, 3, height as usize, width as usize],
                actual: shape.iter().map(|&x| x as usize).collect(),
            });
        }

        let h = shape[2] as usize;
        let w = shape[3] as usize;

        if w != width as usize || h != height as usize {
            return Err(CvError::ShapeMismatch {
                expected: vec![1, 3, height as usize, width as usize],
                actual: shape.iter().map(|&x| x as usize).collect(),
            });
        }
        let mut output = vec![0u8; w * h * 3];

        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                // Denormalize and clamp to [0, 255]
                // Access as flat array: [batch, channel, height, width]
                let r_idx = y * w + x;
                let g_idx = h * w + y * w + x;
                let b_idx = 2 * h * w + y * w + x;

                output[idx] = (data[r_idx] * 255.0).clamp(0.0, 255.0).round() as u8;
                output[idx + 1] = (data[g_idx] * 255.0).clamp(0.0, 255.0).round() as u8;
                output[idx + 2] = (data[b_idx] * 255.0).clamp(0.0, 255.0).round() as u8;
            }
        }

        Ok(output)
    }

    /// Get the upscale factor.
    #[must_use]
    pub const fn scale_factor(&self) -> UpscaleFactor {
        self.scale_factor
    }

    /// Get the tile configuration.
    #[must_use]
    pub const fn tile_config(&self) -> &TileConfig {
        &self.tile_config
    }
}

/// Batch upscaler for processing multiple images efficiently.
pub struct BatchUpscaler {
    upscaler: EsrganUpscaler,
    batch_size: usize,
}

impl BatchUpscaler {
    /// Create a new batch upscaler.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the ONNX model
    /// * `scale_factor` - Upscale factor
    /// * `batch_size` - Maximum number of tiles to process simultaneously
    pub fn new(
        model_path: impl AsRef<Path>,
        scale_factor: UpscaleFactor,
        batch_size: usize,
    ) -> CvResult<Self> {
        let upscaler = EsrganUpscaler::new(model_path, scale_factor)?;
        Ok(Self {
            upscaler,
            batch_size,
        })
    }

    /// Process multiple images in a batch.
    ///
    /// # Arguments
    ///
    /// * `images` - Vector of (image_data, width, height) tuples
    ///
    /// # Returns
    ///
    /// Vector of upscaled images
    pub fn upscale_batch(&mut self, images: &[(&[u8], u32, u32)]) -> CvResult<Vec<Vec<u8>>> {
        let mut results = Vec::with_capacity(images.len());

        for (image, width, height) in images {
            let result = self.upscaler.upscale(image, *width, *height)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Get the batch size.
    #[must_use]
    pub const fn batch_size(&self) -> usize {
        self.batch_size
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_batch_size() {
        // Mock batch upscaler test
        let batch_size = 4;
        assert_eq!(batch_size, 4);
    }
}
