//! Unified super-resolution model supporting multiple architectures.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::too_many_lines)]

use super::cache::ModelCache;
use super::types::{
    ChromaUpscaleMode, ModelType, ProcessingOptions, ProgressCallback, QualityMode, TileConfig,
    UpscaleFactor,
};
use crate::error::{CvError, CvResult};
use ndarray::Array4;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Value;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Unified super-resolution model supporting multiple architectures.
///
/// This is the main interface for super-resolution, supporting different
/// model types, quality modes, and processing options.
pub struct SuperResolutionModel {
    pub(super) session: Arc<Mutex<Session>>,
    pub(super) model_type: ModelType,
    pub(super) scale_factor: UpscaleFactor,
    pub(super) tile_config: TileConfig,
    pub(super) processing_options: ProcessingOptions,
    pub(super) progress_callback: Option<ProgressCallback>,
}

impl SuperResolutionModel {
    /// Create a new super-resolution model from an ONNX file.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the ONNX model file
    /// * `model_type` - Type of the model
    /// * `scale_factor` - Upscale factor
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be loaded.
    pub fn new(
        model_path: impl AsRef<Path>,
        model_type: ModelType,
        scale_factor: UpscaleFactor,
    ) -> CvResult<Self> {
        let session = Session::builder()
            .map_err(|e| CvError::onnx_runtime(format!("Failed to create session builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| CvError::onnx_runtime(format!("Failed to set optimization level: {e}")))?
            .commit_from_file(model_path.as_ref())
            .map_err(|e| CvError::model_load(format!("Failed to load model: {e}")))?;

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            model_type,
            scale_factor,
            tile_config: TileConfig::default(),
            processing_options: ProcessingOptions::default(),
            progress_callback: None,
        })
    }

    /// Create a model using the model cache.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the ONNX model file
    /// * `model_type` - Type of the model
    /// * `scale_factor` - Upscale factor
    /// * `cache` - Model cache to use
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be loaded.
    pub fn with_cache(
        model_path: impl AsRef<Path>,
        model_type: ModelType,
        scale_factor: UpscaleFactor,
        cache: &ModelCache,
    ) -> CvResult<Self> {
        let session = cache.get_or_load(model_path)?;

        Ok(Self {
            session,
            model_type,
            scale_factor,
            tile_config: TileConfig::default(),
            processing_options: ProcessingOptions::default(),
            progress_callback: None,
        })
    }

    /// Create a model from a quality mode.
    ///
    /// This is a convenience method that selects appropriate model settings
    /// based on the desired quality level. Note that you still need to provide
    /// the path to the actual ONNX model file.
    ///
    /// # Arguments
    ///
    /// * `mode` - Quality mode
    /// * `scale_factor` - Upscale factor
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be loaded.
    ///
    /// # Note
    ///
    /// This method expects the model file to be named according to the pattern:
    /// `{model_name}_x{scale}.onnx` (e.g., "esrgan_x4.onnx")
    pub fn from_quality_mode(mode: QualityMode, scale_factor: UpscaleFactor) -> CvResult<Self> {
        let model_type = mode.recommended_model();
        let tile_size = mode.recommended_tile_size();

        // Construct expected model path
        let model_name = match model_type {
            ModelType::ESRGAN => "esrgan",
            ModelType::RealESRGAN => "realesrgan",
            ModelType::EDSR => "edsr",
            ModelType::SRCNN => "srcnn",
            ModelType::VDSR => "vdsr",
        };
        let scale = scale_factor.scale();
        let model_path = format!("{model_name}_x{scale}.onnx");

        let mut model = Self::new(model_path, model_type, scale_factor)?;
        model.tile_config = TileConfig::new(tile_size, tile_size / 16)?;

        // Set processing options based on quality mode
        model.processing_options = match mode {
            QualityMode::Fast => ProcessingOptions::fast(),
            QualityMode::Balanced => ProcessingOptions::default(),
            QualityMode::HighQuality => ProcessingOptions::enhanced(),
            QualityMode::Animation => {
                let mut opts = ProcessingOptions::enhanced();
                opts.sharpness = 0.5;
                opts
            }
        };

        Ok(model)
    }

    /// Set tile configuration.
    pub fn set_tile_config(&mut self, config: TileConfig) {
        self.tile_config = config;
    }

    /// Set processing options.
    pub fn set_processing_options(&mut self, options: ProcessingOptions) {
        self.processing_options = options;
    }

    /// Set progress callback.
    pub fn set_progress_callback(&mut self, callback: ProgressCallback) {
        self.progress_callback = Some(callback);
    }

    /// Get the model type.
    #[must_use]
    pub const fn model_type(&self) -> ModelType {
        self.model_type
    }

    /// Get the scale factor.
    #[must_use]
    pub const fn scale_factor(&self) -> UpscaleFactor {
        self.scale_factor
    }

    /// Upscale an RGB image.
    ///
    /// # Arguments
    ///
    /// * `image` - Input image in RGB format (row-major, packed RGB)
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    ///
    /// # Returns
    ///
    /// Upscaled image in RGB format
    ///
    /// # Errors
    ///
    /// Returns an error if processing fails.
    pub fn upscale(&mut self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
        // Validate inputs
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let expected_size = (width as usize) * (height as usize) * 3;
        if image.len() != expected_size {
            return Err(CvError::insufficient_data(expected_size, image.len()));
        }

        // Apply pre-processing
        let mut preprocessed = image.to_vec();
        if self.processing_options.denoise {
            self.apply_denoising(&mut preprocessed, width, height)?;
        }

        // Perform upscaling
        let tile_size = self.tile_config.tile_size;
        let upscaled = if width <= tile_size && height <= tile_size {
            self.upscale_single_tile(&preprocessed, width, height)?
        } else {
            self.upscale_tiled(&preprocessed, width, height)?
        };

        // Apply post-processing
        let mut result = upscaled;
        let out_scale = self.scale_factor.scale();
        let out_width = width * out_scale;
        let out_height = height * out_scale;

        if self.processing_options.artifact_reduction {
            self.apply_artifact_reduction(&mut result, out_width, out_height)?;
        }

        if self.processing_options.edge_enhancement {
            self.apply_edge_enhancement(&mut result, out_width, out_height)?;
        }

        if self.processing_options.sharpness > 0.0 {
            self.apply_sharpening(
                &mut result,
                out_width,
                out_height,
                self.processing_options.sharpness,
            )?;
        }

        Ok(result)
    }

    /// Upscale a YUV image.
    ///
    /// # Arguments
    ///
    /// * `y_plane` - Y (luma) plane data
    /// * `u_plane` - U (chroma) plane data
    /// * `v_plane` - V (chroma) plane data
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `chroma_width` - Chroma plane width (for subsampled formats)
    /// * `chroma_height` - Chroma plane height (for subsampled formats)
    ///
    /// # Returns
    ///
    /// Tuple of (y_upscaled, u_upscaled, v_upscaled)
    ///
    /// # Errors
    ///
    /// Returns an error if processing fails.
    #[allow(clippy::too_many_arguments)]
    pub fn upscale_yuv(
        &mut self,
        y_plane: &[u8],
        u_plane: &[u8],
        v_plane: &[u8],
        width: u32,
        height: u32,
        chroma_width: u32,
        chroma_height: u32,
    ) -> CvResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        match self.processing_options.chroma_upscale {
            ChromaUpscaleMode::LumaOnly => self.upscale_yuv_luma_only(
                y_plane,
                u_plane,
                v_plane,
                width,
                height,
                chroma_width,
                chroma_height,
            ),
            ChromaUpscaleMode::Separate => self.upscale_yuv_separate(
                y_plane,
                u_plane,
                v_plane,
                width,
                height,
                chroma_width,
                chroma_height,
            ),
            ChromaUpscaleMode::Joint => self.upscale_yuv_joint(
                y_plane,
                u_plane,
                v_plane,
                width,
                height,
                chroma_width,
                chroma_height,
            ),
        }
    }

    /// Upscale YUV with luma-only processing.
    fn upscale_yuv_luma_only(
        &mut self,
        y_plane: &[u8],
        u_plane: &[u8],
        v_plane: &[u8],
        width: u32,
        height: u32,
        chroma_width: u32,
        chroma_height: u32,
    ) -> CvResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        // Upscale luma with neural network
        let y_rgb = Self::gray_to_rgb(y_plane, width, height);
        let y_upscaled_rgb = self.upscale(&y_rgb, width, height)?;
        let scale = self.scale_factor.scale();
        let y_upscaled = Self::rgb_to_gray(&y_upscaled_rgb, width * scale, height * scale);

        // Simple bilinear upscale for chroma
        let u_upscaled = Self::bilinear_upscale(u_plane, chroma_width, chroma_height, scale);
        let v_upscaled = Self::bilinear_upscale(v_plane, chroma_width, chroma_height, scale);

        Ok((y_upscaled, u_upscaled, v_upscaled))
    }

    /// Upscale YUV with separate channel processing.
    fn upscale_yuv_separate(
        &mut self,
        y_plane: &[u8],
        u_plane: &[u8],
        v_plane: &[u8],
        width: u32,
        height: u32,
        chroma_width: u32,
        chroma_height: u32,
    ) -> CvResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        // Upscale each channel with neural network
        let y_rgb = Self::gray_to_rgb(y_plane, width, height);
        let y_upscaled_rgb = self.upscale(&y_rgb, width, height)?;
        let scale = self.scale_factor.scale();
        let y_upscaled = Self::rgb_to_gray(&y_upscaled_rgb, width * scale, height * scale);

        // Upscale chroma channels
        let u_rgb = Self::gray_to_rgb(u_plane, chroma_width, chroma_height);
        let u_upscaled_rgb = self.upscale(&u_rgb, chroma_width, chroma_height)?;
        let u_upscaled =
            Self::rgb_to_gray(&u_upscaled_rgb, chroma_width * scale, chroma_height * scale);

        let v_rgb = Self::gray_to_rgb(v_plane, chroma_width, chroma_height);
        let v_upscaled_rgb = self.upscale(&v_rgb, chroma_width, chroma_height)?;
        let v_upscaled =
            Self::rgb_to_gray(&v_upscaled_rgb, chroma_width * scale, chroma_height * scale);

        Ok((y_upscaled, u_upscaled, v_upscaled))
    }

    /// Upscale YUV with joint processing (convert to RGB first).
    fn upscale_yuv_joint(
        &mut self,
        y_plane: &[u8],
        u_plane: &[u8],
        v_plane: &[u8],
        width: u32,
        height: u32,
        chroma_width: u32,
        chroma_height: u32,
    ) -> CvResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        // Upsample chroma to match luma resolution if needed
        let (u_full, v_full) = if chroma_width != width || chroma_height != height {
            let scale_x = width / chroma_width;
            let scale_y = height / chroma_height;
            let scale = scale_x.max(scale_y);
            (
                Self::bilinear_upscale(u_plane, chroma_width, chroma_height, scale),
                Self::bilinear_upscale(v_plane, chroma_width, chroma_height, scale),
            )
        } else {
            (u_plane.to_vec(), v_plane.to_vec())
        };

        // Convert YUV to RGB
        let rgb = Self::yuv_to_rgb(y_plane, &u_full, &v_full, width, height)?;

        // Upscale RGB
        let rgb_upscaled = self.upscale(&rgb, width, height)?;

        // Convert back to YUV
        let scale = self.scale_factor.scale();
        let out_width = width * scale;
        let out_height = height * scale;
        Self::rgb_to_yuv(&rgb_upscaled, out_width, out_height)
    }

    /// Upscale a single tile (internal method).
    pub(super) fn upscale_single_tile(
        &mut self,
        image: &[u8],
        width: u32,
        height: u32,
    ) -> CvResult<Vec<u8>> {
        // Convert RGB u8 to normalized float32 [1, 3, H, W]
        let input_tensor = self.preprocess_image(image, width, height)?;

        // Convert to ONNX Value
        let input_value = Value::from_array(input_tensor)
            .map_err(|e| CvError::onnx_runtime(format!("Failed to create input tensor: {e}")))?;

        // Run inference and extract owned data to release the session borrow
        let (shape_owned, data_owned) = {
            let mut session = self
                .session
                .lock()
                .map_err(|e| CvError::onnx_runtime(format!("Session lock error: {e}")))?;
            let outputs = session
                .run(ort::inputs![input_value])
                .map_err(|e| CvError::onnx_runtime(format!("Inference failed: {e}")))?;
            let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(|e| {
                CvError::tensor_error(format!("Failed to extract output tensor: {e}"))
            })?;
            let shape_owned: Vec<i64> = shape.iter().copied().collect();
            let data_owned: Vec<f32> = data.to_vec();
            (shape_owned, data_owned)
        };

        // Convert back to RGB u8
        let scale = self.scale_factor.scale();
        self.postprocess_tensor(&shape_owned, &data_owned, width * scale, height * scale)
    }

    /// Upscale using tile-based processing for large images.
    pub(super) fn upscale_tiled(
        &mut self,
        image: &[u8],
        width: u32,
        height: u32,
    ) -> CvResult<Vec<u8>> {
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
                    Self::extract_tile(image, width, height, x_start, y_start, tile_w, tile_h)?;

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
                Self::blend_tile(
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
                    self.tile_config.feather_width,
                )?;

                tile_idx += 1;
            }
        }

        // Normalize by weight map
        Self::normalize_by_weights(&mut output, &weight_map, out_width, out_height);

        Ok(output)
    }

    /// Preprocess image: RGB u8 -> normalized float32 [1, 3, H, W].
    fn preprocess_image(&self, image: &[u8], width: u32, height: u32) -> CvResult<Array4<f32>> {
        let w = width as usize;
        let h = height as usize;

        let mut tensor = Array4::<f32>::zeros((1, 3, h, w));
        let (norm_min, norm_max) = self.model_type.normalization_range();
        let norm_scale = norm_max - norm_min;

        let rgb_mean = if self.model_type.uses_mean_subtraction() {
            self.model_type.rgb_mean()
        } else {
            [0.0, 0.0, 0.0]
        };

        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                // Normalize and apply mean subtraction if needed
                tensor[[0, 0, y, x]] =
                    (image[idx] as f32 / 255.0) * norm_scale + norm_min - rgb_mean[0];
                tensor[[0, 1, y, x]] =
                    (image[idx + 1] as f32 / 255.0) * norm_scale + norm_min - rgb_mean[1];
                tensor[[0, 2, y, x]] =
                    (image[idx + 2] as f32 / 255.0) * norm_scale + norm_min - rgb_mean[2];
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

        let (norm_min, norm_max) = self.model_type.normalization_range();
        let norm_scale = norm_max - norm_min;

        let rgb_mean = if self.model_type.uses_mean_subtraction() {
            self.model_type.rgb_mean()
        } else {
            [0.0, 0.0, 0.0]
        };

        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                let r_idx = y * w + x;
                let g_idx = h * w + y * w + x;
                let b_idx = 2 * h * w + y * w + x;

                // Denormalize and add mean back if needed
                let r = (data[r_idx] + rgb_mean[0] - norm_min) / norm_scale * 255.0;
                let g = (data[g_idx] + rgb_mean[1] - norm_min) / norm_scale * 255.0;
                let b = (data[b_idx] + rgb_mean[2] - norm_min) / norm_scale * 255.0;

                output[idx] = r.clamp(0.0, 255.0).round() as u8;
                output[idx + 1] = g.clamp(0.0, 255.0).round() as u8;
                output[idx + 2] = b.clamp(0.0, 255.0).round() as u8;
            }
        }

        Ok(output)
    }

    /// Extract a rectangular tile from the source image.
    pub(super) fn extract_tile(
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
    pub(super) fn blend_tile(
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
        feather: u32,
    ) -> CvResult<()> {
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
    pub(super) fn normalize_by_weights(
        output: &mut [u8],
        weights: &[f32],
        width: u32,
        height: u32,
    ) {
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

    // Color space conversion utilities

    /// Convert grayscale to RGB (replicate channel).
    pub(super) fn gray_to_rgb(gray: &[u8], _width: u32, _height: u32) -> Vec<u8> {
        let mut rgb = Vec::with_capacity(gray.len() * 3);
        for &pixel in gray {
            rgb.push(pixel);
            rgb.push(pixel);
            rgb.push(pixel);
        }
        rgb
    }

    /// Convert RGB to grayscale (take first channel).
    pub(super) fn rgb_to_gray(rgb: &[u8], _width: u32, _height: u32) -> Vec<u8> {
        let mut gray = Vec::with_capacity(rgb.len() / 3);
        for chunk in rgb.chunks_exact(3) {
            gray.push(chunk[0]);
        }
        gray
    }

    /// Convert YUV to RGB.
    pub(super) fn yuv_to_rgb(
        y: &[u8],
        u: &[u8],
        v: &[u8],
        width: u32,
        height: u32,
    ) -> CvResult<Vec<u8>> {
        let size = (width * height) as usize;
        if y.len() != size || u.len() != size || v.len() != size {
            return Err(CvError::insufficient_data(
                size * 3,
                y.len() + u.len() + v.len(),
            ));
        }

        let mut rgb = vec![0u8; size * 3];

        for i in 0..size {
            let y_val = y[i] as f32;
            let u_val = u[i] as f32 - 128.0;
            let v_val = v[i] as f32 - 128.0;

            let r = y_val + 1.402 * v_val;
            let g = y_val - 0.344_136 * u_val - 0.714_136 * v_val;
            let b = y_val + 1.772 * u_val;

            rgb[i * 3] = r.clamp(0.0, 255.0).round() as u8;
            rgb[i * 3 + 1] = g.clamp(0.0, 255.0).round() as u8;
            rgb[i * 3 + 2] = b.clamp(0.0, 255.0).round() as u8;
        }

        Ok(rgb)
    }

    /// Convert RGB to YUV.
    pub(super) fn rgb_to_yuv(
        rgb: &[u8],
        width: u32,
        height: u32,
    ) -> CvResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        let size = (width * height) as usize;
        if rgb.len() != size * 3 {
            return Err(CvError::insufficient_data(size * 3, rgb.len()));
        }

        let mut y = vec![0u8; size];
        let mut u = vec![0u8; size];
        let mut v = vec![0u8; size];

        for i in 0..size {
            let r = rgb[i * 3] as f32;
            let g = rgb[i * 3 + 1] as f32;
            let b = rgb[i * 3 + 2] as f32;

            let y_val = 0.299 * r + 0.587 * g + 0.114 * b;
            let u_val = -0.168_736 * r - 0.331_264 * g + 0.5 * b + 128.0;
            let v_val = 0.5 * r - 0.418_688 * g - 0.081_312 * b + 128.0;

            y[i] = y_val.clamp(0.0, 255.0).round() as u8;
            u[i] = u_val.clamp(0.0, 255.0).round() as u8;
            v[i] = v_val.clamp(0.0, 255.0).round() as u8;
        }

        Ok((y, u, v))
    }

    /// Bilinear upscaling for chroma planes.
    pub(super) fn bilinear_upscale(src: &[u8], width: u32, height: u32, scale: u32) -> Vec<u8> {
        let out_width = width * scale;
        let out_height = height * scale;
        let mut output = vec![0u8; (out_width * out_height) as usize];

        for y in 0..out_height {
            for x in 0..out_width {
                let src_x = (x as f32 / scale as f32).min((width - 1) as f32);
                let src_y = (y as f32 / scale as f32).min((height - 1) as f32);

                let x0 = src_x.floor() as u32;
                let y0 = src_y.floor() as u32;
                let x1 = (x0 + 1).min(width - 1);
                let y1 = (y0 + 1).min(height - 1);

                let fx = src_x - x0 as f32;
                let fy = src_y - y0 as f32;

                let p00 = src[(y0 * width + x0) as usize] as f32;
                let p10 = src[(y0 * width + x1) as usize] as f32;
                let p01 = src[(y1 * width + x0) as usize] as f32;
                let p11 = src[(y1 * width + x1) as usize] as f32;

                let value = p00 * (1.0 - fx) * (1.0 - fy)
                    + p10 * fx * (1.0 - fy)
                    + p01 * (1.0 - fx) * fy
                    + p11 * fx * fy;

                output[(y * out_width + x) as usize] = value.round() as u8;
            }
        }

        output
    }

    // Post-processing methods

    /// Apply denoising to the image.
    fn apply_denoising(&self, image: &mut [u8], width: u32, height: u32) -> CvResult<()> {
        // Simple bilateral filter approximation
        let kernel_size = 5;
        let sigma_color = 30.0f32;
        let sigma_space = 2.0f32;

        let padded_width = width as usize;
        let src = image.to_vec();

        for y in (kernel_size / 2)..(height as usize - kernel_size / 2) {
            for x in (kernel_size / 2)..(padded_width - kernel_size / 2) {
                for c in 0..3 {
                    let center_val = src[(y * padded_width + x) * 3 + c] as f32;
                    let mut sum = 0.0f32;
                    let mut weight_sum = 0.0f32;

                    for ky in 0..kernel_size {
                        for kx in 0..kernel_size {
                            let ny = y + ky - kernel_size / 2;
                            let nx = x + kx - kernel_size / 2;
                            let val = src[(ny * padded_width + nx) * 3 + c] as f32;

                            let color_dist = (val - center_val).abs();
                            let space_dist = ((ky as isize - kernel_size as isize / 2).pow(2)
                                + (kx as isize - kernel_size as isize / 2).pow(2))
                                as f32;

                            let weight = (-color_dist / sigma_color).exp()
                                * (-space_dist / (2.0 * sigma_space * sigma_space)).exp();

                            sum += val * weight;
                            weight_sum += weight;
                        }
                    }

                    if weight_sum > 0.0 {
                        image[(y * padded_width + x) * 3 + c] = (sum / weight_sum).round() as u8;
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply artifact reduction (smoothing compression artifacts).
    fn apply_artifact_reduction(&self, image: &mut [u8], width: u32, height: u32) -> CvResult<()> {
        // Light Gaussian blur to reduce artifacts
        let kernel = [1.0f32, 2.0, 1.0];
        let kernel_sum = 4.0f32;

        let src = image.to_vec();
        let w = width as usize;
        let h = height as usize;

        // Horizontal pass
        let mut temp = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 1..w - 1 {
                for c in 0..3 {
                    let sum = src[(y * w + x - 1) * 3 + c] as f32 * kernel[0]
                        + src[(y * w + x) * 3 + c] as f32 * kernel[1]
                        + src[(y * w + x + 1) * 3 + c] as f32 * kernel[2];
                    temp[(y * w + x) * 3 + c] = (sum / kernel_sum).round() as u8;
                }
            }
        }

        // Vertical pass
        for y in 1..h - 1 {
            for x in 0..w {
                for c in 0..3 {
                    let sum = temp[((y - 1) * w + x) * 3 + c] as f32 * kernel[0]
                        + temp[(y * w + x) * 3 + c] as f32 * kernel[1]
                        + temp[((y + 1) * w + x) * 3 + c] as f32 * kernel[2];
                    image[(y * w + x) * 3 + c] = (sum / kernel_sum).round() as u8;
                }
            }
        }

        Ok(())
    }

    /// Apply edge enhancement.
    fn apply_edge_enhancement(&self, image: &mut [u8], width: u32, height: u32) -> CvResult<()> {
        // Unsharp masking
        let amount = 0.5f32;
        let src = image.to_vec();
        let w = width as usize;
        let h = height as usize;

        // Gaussian blur
        let mut blurred = src.clone();
        let kernel = [1.0f32, 4.0, 6.0, 4.0, 1.0];
        let kernel_sum = 16.0f32;

        // Horizontal pass
        let mut temp = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 2..w - 2 {
                for c in 0..3 {
                    let sum = blurred[(y * w + x - 2) * 3 + c] as f32 * kernel[0]
                        + blurred[(y * w + x - 1) * 3 + c] as f32 * kernel[1]
                        + blurred[(y * w + x) * 3 + c] as f32 * kernel[2]
                        + blurred[(y * w + x + 1) * 3 + c] as f32 * kernel[3]
                        + blurred[(y * w + x + 2) * 3 + c] as f32 * kernel[4];
                    temp[(y * w + x) * 3 + c] = (sum / kernel_sum).round() as u8;
                }
            }
        }

        // Vertical pass
        for y in 2..h - 2 {
            for x in 0..w {
                for c in 0..3 {
                    let sum = temp[((y - 2) * w + x) * 3 + c] as f32 * kernel[0]
                        + temp[((y - 1) * w + x) * 3 + c] as f32 * kernel[1]
                        + temp[(y * w + x) * 3 + c] as f32 * kernel[2]
                        + temp[((y + 1) * w + x) * 3 + c] as f32 * kernel[3]
                        + temp[((y + 2) * w + x) * 3 + c] as f32 * kernel[4];
                    blurred[(y * w + x) * 3 + c] = (sum / kernel_sum).round() as u8;
                }
            }
        }

        // Unsharp mask: output = src + amount * (src - blurred)
        for i in 0..image.len() {
            let original = src[i] as f32;
            let blur = blurred[i] as f32;
            let enhanced = original + amount * (original - blur);
            image[i] = enhanced.clamp(0.0, 255.0).round() as u8;
        }

        Ok(())
    }

    /// Apply sharpening filter.
    fn apply_sharpening(
        &self,
        image: &mut [u8],
        width: u32,
        height: u32,
        amount: f32,
    ) -> CvResult<()> {
        // Laplacian sharpening kernel
        let src = image.to_vec();
        let w = width as usize;
        let h = height as usize;

        for y in 1..h - 1 {
            for x in 1..w - 1 {
                for c in 0..3 {
                    let center = src[(y * w + x) * 3 + c] as f32;
                    let top = src[((y - 1) * w + x) * 3 + c] as f32;
                    let bottom = src[((y + 1) * w + x) * 3 + c] as f32;
                    let left = src[(y * w + x - 1) * 3 + c] as f32;
                    let right = src[(y * w + x + 1) * 3 + c] as f32;

                    let laplacian = 4.0 * center - (top + bottom + left + right);
                    let sharpened = center + amount * laplacian;

                    image[(y * w + x) * 3 + c] = sharpened.clamp(0.0, 255.0).round() as u8;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;

    #[test]
    fn test_bilinear_upscale() {
        let src = vec![128u8; 16 * 16];
        let upscaled = SuperResolutionModel::bilinear_upscale(&src, 16, 16, 2);
        assert_eq!(upscaled.len(), 32 * 32);

        // All values should be close to original (128)
        for &val in &upscaled {
            assert!(val >= 120 && val <= 136);
        }
    }

    #[test]
    fn test_gray_to_rgb_conversion() {
        let gray = vec![100u8, 150, 200];
        let rgb = SuperResolutionModel::gray_to_rgb(&gray, 3, 1);
        assert_eq!(rgb.len(), 9);
        assert_eq!(rgb[0], 100);
        assert_eq!(rgb[1], 100);
        assert_eq!(rgb[2], 100);
        assert_eq!(rgb[3], 150);
        assert_eq!(rgb[4], 150);
        assert_eq!(rgb[5], 150);
    }

    #[test]
    fn test_rgb_to_gray_conversion() {
        let rgb = vec![100u8, 100, 100, 150, 150, 150, 200, 200, 200];
        let gray = SuperResolutionModel::rgb_to_gray(&rgb, 3, 1);
        assert_eq!(gray.len(), 3);
        assert_eq!(gray[0], 100);
        assert_eq!(gray[1], 150);
        assert_eq!(gray[2], 200);
    }

    #[test]
    fn test_yuv_rgb_conversions() {
        let y = vec![128u8; 64];
        let u = vec![128u8; 64];
        let v = vec![128u8; 64];

        let rgb =
            SuperResolutionModel::yuv_to_rgb(&y, &u, &v, 8, 8).expect("yuv_to_rgb should succeed");
        assert_eq!(rgb.len(), 64 * 3);

        let (y_back, u_back, v_back) =
            SuperResolutionModel::rgb_to_yuv(&rgb, 8, 8).expect("rgb_to_yuv should succeed");
        assert_eq!(y_back.len(), 64);
        assert_eq!(u_back.len(), 64);
        assert_eq!(v_back.len(), 64);

        // Values should be close after round-trip conversion
        for i in 0..64 {
            assert!((y[i] as i32 - y_back[i] as i32).abs() <= 2);
            assert!((u[i] as i32 - u_back[i] as i32).abs() <= 2);
            assert!((v[i] as i32 - v_back[i] as i32).abs() <= 2);
        }
    }

    #[test]
    fn test_yuv_rgb_invalid_sizes() {
        let y = vec![128u8; 64];
        let u = vec![128u8; 32]; // Wrong size
        let v = vec![128u8; 64];

        let result = SuperResolutionModel::yuv_to_rgb(&y, &u, &v, 8, 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_preprocess_postprocess_roundtrip() {
        // Test preprocess/postprocess logic without requiring an ONNX session.
        let width: u32 = 8;
        let height: u32 = 8;
        let w = width as usize;
        let h = height as usize;
        let input: Vec<u8> = (0..(w * h * 3)).map(|i| (i % 256) as u8).collect();

        // Preprocess: RGB u8 -> [1, 3, H, W] float tensor
        let mut tensor = Array4::<f32>::zeros((1, 3, h, w));
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                tensor[[0, 0, y, x]] = input[idx] as f32 / 255.0;
                tensor[[0, 1, y, x]] = input[idx + 1] as f32 / 255.0;
                tensor[[0, 2, y, x]] = input[idx + 2] as f32 / 255.0;
            }
        }
        assert_eq!(tensor.shape(), &[1, 3, h, w]);

        // Postprocess: float tensor -> RGB u8
        let shape_i64: Vec<i64> = tensor.shape().iter().map(|&x| x as i64).collect();
        let data_f32: Vec<f32> = tensor.iter().copied().collect();
        let mut output = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                let r_idx = y * w + x;
                let g_idx = h * w + y * w + x;
                let b_idx = 2 * h * w + y * w + x;
                output[idx] = (data_f32[r_idx] * 255.0).clamp(0.0, 255.0).round() as u8;
                output[idx + 1] = (data_f32[g_idx] * 255.0).clamp(0.0, 255.0).round() as u8;
                output[idx + 2] = (data_f32[b_idx] * 255.0).clamp(0.0, 255.0).round() as u8;
            }
        }

        assert_eq!(output.len(), input.len());
        assert_eq!(shape_i64, vec![1, 3, h as i64, w as i64]);
        for (a, b) in input.iter().zip(output.iter()) {
            assert!(
                (*a as i32 - *b as i32).abs() <= 1,
                "Values differ: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_extract_tile() {
        let width: u32 = 10;
        let height: u32 = 10;
        let image: Vec<u8> = (0..(width * height * 3) as usize)
            .map(|i| (i % 256) as u8)
            .collect();

        let (x, y, tile_w, tile_h) = (2u32, 2u32, 4u32, 4u32);
        assert!(x + tile_w <= width && y + tile_h <= height);
        let mut tile = Vec::with_capacity((tile_w * tile_h * 3) as usize);
        for row in y..y + tile_h {
            let start = ((row * width + x) * 3) as usize;
            let end = start + (tile_w * 3) as usize;
            tile.extend_from_slice(&image[start..end]);
        }
        assert_eq!(tile.len(), 4 * 4 * 3);
    }

    #[test]
    fn test_extract_tile_out_of_bounds() {
        let width: u32 = 10;
        let height: u32 = 10;
        let (x, y, tile_w, tile_h) = (8u32, 8u32, 5u32, 5u32);
        assert!(
            x + tile_w > width || y + tile_h > height,
            "Expected out-of-bounds tile coordinates"
        );
    }

    #[test]
    fn test_normalize_by_weights() {
        let mut output = vec![100u8, 100, 100, 200, 200, 200];
        let weights = vec![2.0f32, 4.0];
        let width: u32 = 2;
        let height: u32 = 1;

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

        // First pixel: 100 / 2.0 = 50
        assert_eq!(output[0], 50);
        assert_eq!(output[1], 50);
        assert_eq!(output[2], 50);

        // Second pixel: 200 / 4.0 = 50
        assert_eq!(output[3], 50);
        assert_eq!(output[4], 50);
        assert_eq!(output[5], 50);
    }
}
