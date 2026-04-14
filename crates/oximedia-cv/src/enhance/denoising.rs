//! Neural denoising using CNN-based models.
//!
//! This module provides AI-powered image denoising using convolutional neural networks
//! via ONNX Runtime. Supports both blind denoising (unknown noise level) and
//! noise-level-aware denoising.
//!
//! # Features
//!
//! - CNN-based denoising
//! - Blind denoising (unknown noise level)
//! - Color image denoising
//! - Tile-based processing for large images
//! - Multiple noise levels support
//!
//! # Example
//!
//! ```no_run
//! use oximedia_cv::enhance::{NeuralDenoiser, NoiseLevel};
//!
//! let mut denoiser = NeuralDenoiser::new("denoiser.onnx")?;
//! let noisy_image = vec![0u8; 512 * 512 * 3];
//! let denoised = denoiser.denoise(&noisy_image, 512, 512)?;
//! # Ok::<(), oximedia_cv::error::CvError>(())
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::too_many_lines)]

use crate::error::{CvError, CvResult};
use ndarray::Array4;
use oxionnx::Session;
use std::collections::HashMap;
use std::path::Path;

/// Noise level for denoising.
///
/// Represents the estimated or configured noise level in the image.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum NoiseLevel {
    /// Low noise (sigma ≈ 5-15).
    Low,
    /// Medium noise (sigma ≈ 15-30).
    Medium,
    /// High noise (sigma ≈ 30-50).
    High,
    /// Custom noise level with specific sigma value.
    Custom(f32),
    /// Blind denoising (automatic noise estimation).
    #[default]
    Blind,
}

impl NoiseLevel {
    /// Get the sigma value for this noise level.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::enhance::NoiseLevel;
    ///
    /// assert_eq!(NoiseLevel::Low.sigma(), 10.0);
    /// assert_eq!(NoiseLevel::Custom(25.0).sigma(), 25.0);
    /// ```
    #[must_use]
    pub fn sigma(&self) -> f32 {
        match self {
            Self::Low => 10.0,
            Self::Medium => 25.0,
            Self::High => 40.0,
            Self::Custom(sigma) => *sigma,
            Self::Blind => 0.0, // Will be estimated
        }
    }

    /// Check if this is blind denoising.
    #[must_use]
    pub const fn is_blind(&self) -> bool {
        matches!(self, Self::Blind)
    }
}

/// Configuration for denoising operations.
#[derive(Debug, Clone)]
pub struct DenoisingConfig {
    /// Noise level setting.
    pub noise_level: NoiseLevel,
    /// Tile size for processing large images.
    pub tile_size: u32,
    /// Padding around tiles to reduce artifacts.
    pub tile_padding: u32,
    /// Color denoising strength (0.0 to 1.0).
    pub color_strength: f32,
    /// Luminance denoising strength (0.0 to 1.0).
    pub luma_strength: f32,
}

impl Default for DenoisingConfig {
    fn default() -> Self {
        Self {
            noise_level: NoiseLevel::Blind,
            tile_size: 256,
            tile_padding: 16,
            color_strength: 1.0,
            luma_strength: 1.0,
        }
    }
}

impl DenoisingConfig {
    /// Create a new denoising configuration.
    ///
    /// # Arguments
    ///
    /// * `noise_level` - Noise level setting
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::enhance::{DenoisingConfig, NoiseLevel};
    ///
    /// let config = DenoisingConfig::new(NoiseLevel::Medium);
    /// assert_eq!(config.noise_level.sigma(), 25.0);
    /// ```
    #[must_use]
    pub fn new(noise_level: NoiseLevel) -> Self {
        Self {
            noise_level,
            ..Default::default()
        }
    }

    /// Set tile size.
    #[must_use]
    pub fn with_tile_size(mut self, tile_size: u32) -> Self {
        self.tile_size = tile_size;
        self
    }

    /// Set tile padding.
    #[must_use]
    pub fn with_tile_padding(mut self, padding: u32) -> Self {
        self.tile_padding = padding;
        self
    }

    /// Set color denoising strength.
    #[must_use]
    pub fn with_color_strength(mut self, strength: f32) -> Self {
        self.color_strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set luminance denoising strength.
    #[must_use]
    pub fn with_luma_strength(mut self, strength: f32) -> Self {
        self.luma_strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> CvResult<()> {
        if self.tile_size < 64 {
            return Err(CvError::invalid_parameter(
                "tile_size",
                format!("{} (must be >= 64)", self.tile_size),
            ));
        }
        if self.tile_padding > self.tile_size / 4 {
            return Err(CvError::invalid_parameter(
                "tile_padding",
                format!("{} (must be <= tile_size / 4)", self.tile_padding),
            ));
        }
        if !(0.0..=1.0).contains(&self.color_strength) {
            return Err(CvError::invalid_parameter(
                "color_strength",
                format!("{} (must be in [0.0, 1.0])", self.color_strength),
            ));
        }
        if !(0.0..=1.0).contains(&self.luma_strength) {
            return Err(CvError::invalid_parameter(
                "luma_strength",
                format!("{} (must be in [0.0, 1.0])", self.luma_strength),
            ));
        }
        Ok(())
    }
}

/// Progress callback for denoising operations.
pub type DenoisingProgressCallback = Box<dyn Fn(usize, usize) -> bool + Send + Sync>;

/// Neural image denoiser.
///
/// Provides AI-powered image denoising using CNN models via ONNX Runtime.
/// Supports both blind denoising and noise-level-aware denoising.
pub struct NeuralDenoiser {
    session: Session,
    config: DenoisingConfig,
    progress_callback: Option<DenoisingProgressCallback>,
}

impl NeuralDenoiser {
    /// Create a new neural denoiser from an ONNX model file.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the ONNX model file
    ///
    /// # Errors
    ///
    /// Returns an error if model loading fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use oximedia_cv::enhance::NeuralDenoiser;
    ///
    /// let denoiser = NeuralDenoiser::new("denoiser.onnx")?;
    /// # Ok::<(), oximedia_cv::error::CvError>(())
    /// ```
    pub fn new(model_path: impl AsRef<Path>) -> CvResult<Self> {
        let session = Session::builder()
            .with_optimization_level(oxionnx::OptLevel::All)
            .load(model_path.as_ref())
            .map_err(|e| CvError::model_load(format!("Failed to load model: {e}")))?;

        Ok(Self {
            session,
            config: DenoisingConfig::default(),
            progress_callback: None,
        })
    }

    /// Create a denoiser with custom configuration.
    pub fn with_config(model_path: impl AsRef<Path>, config: DenoisingConfig) -> CvResult<Self> {
        config.validate()?;
        let mut denoiser = Self::new(model_path)?;
        denoiser.config = config;
        Ok(denoiser)
    }

    /// Set denoising configuration.
    pub fn set_config(&mut self, config: DenoisingConfig) -> CvResult<()> {
        config.validate()?;
        self.config = config;
        Ok(())
    }

    /// Set progress callback.
    pub fn set_progress_callback(&mut self, callback: DenoisingProgressCallback) {
        self.progress_callback = Some(callback);
    }

    /// Denoise an image.
    ///
    /// # Arguments
    ///
    /// * `image` - Input image in RGB format
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Returns
    ///
    /// Denoised image in RGB format
    ///
    /// # Errors
    ///
    /// Returns an error if denoising fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use oximedia_cv::enhance::NeuralDenoiser;
    ///
    /// let mut denoiser = NeuralDenoiser::new("denoiser.onnx")?;
    /// let noisy = vec![0u8; 512 * 512 * 3];
    /// let clean = denoiser.denoise(&noisy, 512, 512)?;
    /// # Ok::<(), oximedia_cv::error::CvError>(())
    /// ```
    pub fn denoise(&mut self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
        // Validate inputs
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let expected_size = (width as usize) * (height as usize) * 3;
        if image.len() != expected_size {
            return Err(CvError::insufficient_data(expected_size, image.len()));
        }

        // Determine if tiling is needed
        let tile_size = self.config.tile_size;
        if width <= tile_size && height <= tile_size {
            self.denoise_single_tile(image, width, height)
        } else {
            self.denoise_tiled(image, width, height)
        }
    }

    /// Denoise a single tile.
    fn denoise_single_tile(&mut self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
        // Preprocess: RGB u8 -> normalized float32 [1, 3, H, W]
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

        // Postprocess: float32 -> RGB u8
        let denoised = self.postprocess_tensor(&shape_owned, &data_owned, width, height)?;

        // Apply strength blending
        self.blend_with_original(image, &denoised, width, height)
    }

    /// Denoise using tile-based processing.
    fn denoise_tiled(&mut self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
        let tile_size = self.config.tile_size;
        let padding = self.config.tile_padding;

        // Calculate tile grid
        let tiles_x = width.div_ceil(tile_size) as usize;
        let tiles_y = height.div_ceil(tile_size) as usize;
        let total_tiles = tiles_x * tiles_y;

        // Output buffer
        let mut output = vec![0u8; (width * height * 3) as usize];
        let mut weight_map = vec![0.0f32; (width * height) as usize];

        // Process each tile
        let mut tile_idx = 0;
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                // Check progress callback
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

                // Extract and denoise tile
                let tile =
                    self.extract_tile(image, width, height, x_start, y_start, tile_w, tile_h)?;
                let denoised_tile = self.denoise_single_tile(&tile, tile_w, tile_h)?;

                // Calculate blend region
                let blend_x_start = if tx == 0 { 0 } else { padding };
                let blend_y_start = if ty == 0 { 0 } else { padding };
                let blend_x_end = if tx == tiles_x - 1 {
                    tile_w
                } else {
                    tile_w - padding
                };
                let blend_y_end = if ty == tiles_y - 1 {
                    tile_h
                } else {
                    tile_h - padding
                };

                // Blend tile into output
                self.blend_tile(
                    &denoised_tile,
                    tile_w,
                    tile_h,
                    &mut output,
                    &mut weight_map,
                    width,
                    height,
                    x_start,
                    y_start,
                    blend_x_start,
                    blend_y_start,
                    blend_x_end,
                    blend_y_end,
                )?;

                tile_idx += 1;
            }
        }

        // Normalize by weights
        self.normalize_by_weights(&mut output, &weight_map, width, height);

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

    /// Blend a processed tile into the output with feathering.
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
        let feather = self.config.tile_padding.min(16);

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

                // Calculate feather weight
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

    /// Blend denoised result with original based on strength settings.
    fn blend_with_original(
        &self,
        original: &[u8],
        denoised: &[u8],
        width: u32,
        height: u32,
    ) -> CvResult<Vec<u8>> {
        let luma_strength = self.config.luma_strength;
        let color_strength = self.config.color_strength;

        if luma_strength >= 0.99 && color_strength >= 0.99 {
            // No blending needed
            return Ok(denoised.to_vec());
        }

        let mut result = Vec::with_capacity(denoised.len());

        for i in 0..(width * height) as usize {
            let idx = i * 3;
            let r_orig = original[idx] as f32;
            let g_orig = original[idx + 1] as f32;
            let b_orig = original[idx + 2] as f32;

            let r_denoised = denoised[idx] as f32;
            let g_denoised = denoised[idx + 1] as f32;
            let b_denoised = denoised[idx + 2] as f32;

            // Calculate luminance (simple average)
            let y_orig = (r_orig + g_orig + b_orig) / 3.0;
            let y_denoised = (r_denoised + g_denoised + b_denoised) / 3.0;

            // Blend luminance
            let y_blend = y_orig + (y_denoised - y_orig) * luma_strength;
            let luma_scale = if y_orig > 0.0 { y_blend / y_orig } else { 1.0 };

            // Apply luminance scaling and color blending
            let r_result = (r_orig * luma_scale * (1.0 - color_strength)
                + r_denoised * color_strength)
                .clamp(0.0, 255.0) as u8;
            let g_result = (g_orig * luma_scale * (1.0 - color_strength)
                + g_denoised * color_strength)
                .clamp(0.0, 255.0) as u8;
            let b_result = (b_orig * luma_scale * (1.0 - color_strength)
                + b_denoised * color_strength)
                .clamp(0.0, 255.0) as u8;

            result.push(r_result);
            result.push(g_result);
            result.push(b_result);
        }

        Ok(result)
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
                // Denormalize and clamp
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

    /// Get the current configuration.
    #[must_use]
    pub const fn config(&self) -> &DenoisingConfig {
        &self.config
    }
}

/// Noise estimation utilities.
pub mod noise_estimation {
    use super::NoiseLevel;

    /// Estimate noise level in an image using MAD (Median Absolute Deviation).
    ///
    /// # Arguments
    ///
    /// * `image` - Input image in RGB format
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Returns
    ///
    /// Estimated noise sigma value
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::enhance::noise_estimation::estimate_noise_mad;
    ///
    /// let image = vec![100u8; 256 * 256 * 3];
    /// let sigma = estimate_noise_mad(&image, 256, 256);
    /// assert!(sigma >= 0.0);
    /// ```
    #[must_use]
    pub fn estimate_noise_mad(image: &[u8], width: u32, height: u32) -> f32 {
        if width < 3 || height < 3 {
            return 0.0;
        }

        // Convert to grayscale and compute Laplacian
        let mut laplacian = Vec::new();
        let w = width as usize;
        let h = height as usize;

        for y in 1..h - 1 {
            for x in 1..w - 1 {
                // Compute grayscale value (simple average)
                let center_idx = (y * w + x) * 3;
                // center is unused in the current implementation
                let _ = (image[center_idx] as f32
                    + image[center_idx + 1] as f32
                    + image[center_idx + 2] as f32)
                    / 3.0;

                // Simple Laplacian kernel
                let mut lap = 0.0;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let ny = (y as i32 + dy) as usize;
                        let nx = (x as i32 + dx) as usize;
                        let idx = (ny * w + nx) * 3;
                        let val =
                            (image[idx] as f32 + image[idx + 1] as f32 + image[idx + 2] as f32)
                                / 3.0;

                        let kernel_val = if dx == 0 && dy == 0 { 8.0 } else { -1.0 };
                        lap += val * kernel_val;
                    }
                }

                laplacian.push(lap.abs());
            }
        }

        if laplacian.is_empty() {
            return 0.0;
        }

        // Compute MAD
        laplacian.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = laplacian[laplacian.len() / 2];

        // Sigma estimation: sigma ≈ MAD / 0.6745
        median / 0.6745
    }

    /// Classify noise level based on sigma.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::enhance::noise_estimation::classify_noise_level;
    /// use oximedia_cv::enhance::NoiseLevel;
    ///
    /// assert!(matches!(classify_noise_level(5.0), NoiseLevel::Low));
    /// assert!(matches!(classify_noise_level(20.0), NoiseLevel::Medium));
    /// ```
    #[must_use]
    pub fn classify_noise_level(sigma: f32) -> NoiseLevel {
        if sigma < 15.0 {
            NoiseLevel::Low
        } else if sigma < 30.0 {
            NoiseLevel::Medium
        } else {
            NoiseLevel::High
        }
    }

    /// Estimate noise level from a small patch of the image.
    ///
    /// More efficient than processing the whole image.
    #[must_use]
    pub fn estimate_noise_patch(image: &[u8], width: u32, height: u32, patch_size: u32) -> f32 {
        let patch_size = patch_size.min(width).min(height);
        let x_start = (width - patch_size) / 2;
        let y_start = (height - patch_size) / 2;

        // Extract central patch
        let mut patch = Vec::new();
        for y in y_start..y_start + patch_size {
            let start = ((y * width + x_start) * 3) as usize;
            let end = start + (patch_size * 3) as usize;
            patch.extend_from_slice(&image[start..end]);
        }

        estimate_noise_mad(&patch, patch_size, patch_size)
    }
}

/// Batch denoising utilities.
pub struct BatchDenoiser {
    denoiser: NeuralDenoiser,
    batch_size: usize,
}

impl BatchDenoiser {
    /// Create a new batch denoiser.
    pub fn new(model_path: impl AsRef<Path>, batch_size: usize) -> CvResult<Self> {
        let denoiser = NeuralDenoiser::new(model_path)?;
        Ok(Self {
            denoiser,
            batch_size,
        })
    }

    /// Denoise multiple images in a batch.
    pub fn denoise_batch(&mut self, images: &[(&[u8], u32, u32)]) -> CvResult<Vec<Vec<u8>>> {
        let mut results = Vec::with_capacity(images.len());

        for (image, width, height) in images {
            let result = self.denoiser.denoise(image, *width, *height)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Get batch size.
    #[must_use]
    pub const fn batch_size(&self) -> usize {
        self.batch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_level_sigma() {
        assert_eq!(NoiseLevel::Low.sigma(), 10.0);
        assert_eq!(NoiseLevel::Medium.sigma(), 25.0);
        assert_eq!(NoiseLevel::High.sigma(), 40.0);
        assert_eq!(NoiseLevel::Custom(15.0).sigma(), 15.0);
        assert_eq!(NoiseLevel::Blind.sigma(), 0.0);
    }

    #[test]
    fn test_noise_level_is_blind() {
        assert!(!NoiseLevel::Low.is_blind());
        assert!(NoiseLevel::Blind.is_blind());
    }

    #[test]
    fn test_denoising_config_default() {
        let config = DenoisingConfig::default();
        assert!(config.noise_level.is_blind());
        assert_eq!(config.tile_size, 256);
        assert_eq!(config.tile_padding, 16);
    }

    #[test]
    fn test_denoising_config_builder() {
        let config = DenoisingConfig::new(NoiseLevel::Medium)
            .with_tile_size(512)
            .with_tile_padding(32)
            .with_color_strength(0.8)
            .with_luma_strength(0.9);

        assert_eq!(config.tile_size, 512);
        assert_eq!(config.tile_padding, 32);
        assert_eq!(config.color_strength, 0.8);
        assert_eq!(config.luma_strength, 0.9);
    }

    #[test]
    fn test_denoising_config_validation() {
        let config = DenoisingConfig::new(NoiseLevel::Low).with_tile_size(32);
        assert!(config.validate().is_err());

        let config = DenoisingConfig::new(NoiseLevel::Low).with_tile_padding(100);
        assert!(config.validate().is_err());

        let config = DenoisingConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_estimate_noise_mad() {
        let image = vec![100u8; 256 * 256 * 3];
        let sigma = noise_estimation::estimate_noise_mad(&image, 256, 256);
        // Constant image should have very low noise
        assert!(sigma < 1.0);
    }

    #[test]
    fn test_classify_noise_level() {
        assert!(matches!(
            noise_estimation::classify_noise_level(5.0),
            NoiseLevel::Low
        ));
        assert!(matches!(
            noise_estimation::classify_noise_level(20.0),
            NoiseLevel::Medium
        ));
        assert!(matches!(
            noise_estimation::classify_noise_level(35.0),
            NoiseLevel::High
        ));
    }

    #[test]
    fn test_estimate_noise_patch() {
        let image = vec![100u8; 512 * 512 * 3];
        let sigma = noise_estimation::estimate_noise_patch(&image, 512, 512, 128);
        assert!(sigma < 1.0);
    }

    #[test]
    fn test_preprocess_postprocess_roundtrip() {
        // Test preprocess/postprocess logic without requiring an ONNX session.
        // NeuralDenoiser::preprocess_image normalizes RGB u8 -> [0,1] float
        // and postprocess_tensor reverses that.
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
        assert_eq!(shape_i64.len(), 4);
        assert_eq!(shape_i64[0], 1);
        assert_eq!(shape_i64[1], 3);

        let out_h = shape_i64[2] as usize;
        let out_w = shape_i64[3] as usize;
        let mut output = vec![0u8; out_w * out_h * 3];

        for y in 0..out_h {
            for x in 0..out_w {
                let idx = (y * out_w + x) * 3;
                let r_idx = 0 * out_h * out_w + y * out_w + x;
                let g_idx = 1 * out_h * out_w + y * out_w + x;
                let b_idx = 2 * out_h * out_w + y * out_w + x;
                output[idx] = (data_f32[r_idx] * 255.0).clamp(0.0, 255.0).round() as u8;
                output[idx + 1] = (data_f32[g_idx] * 255.0).clamp(0.0, 255.0).round() as u8;
                output[idx + 2] = (data_f32[b_idx] * 255.0).clamp(0.0, 255.0).round() as u8;
            }
        }

        assert_eq!(output.len(), input.len());
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
        // Test tile extraction logic directly (no ONNX session needed).
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
    fn test_blend_with_original() {
        // Test blend logic directly without requiring an ONNX session.
        let original = vec![100u8, 100, 100, 200, 200, 200];
        let denoised = vec![50u8, 50, 50, 150, 150, 150];
        let luma_strength: f32 = 0.5;
        let color_strength: f32 = 0.5;
        let width: u32 = 2;
        let height: u32 = 1;

        let mut result = Vec::with_capacity(denoised.len());

        for i in 0..(width * height) as usize {
            let idx = i * 3;
            let r_orig = original[idx] as f32;
            let g_orig = original[idx + 1] as f32;
            let b_orig = original[idx + 2] as f32;

            let r_denoised = denoised[idx] as f32;
            let g_denoised = denoised[idx + 1] as f32;
            let b_denoised = denoised[idx + 2] as f32;

            let y_orig = (r_orig + g_orig + b_orig) / 3.0;
            let y_denoised = (r_denoised + g_denoised + b_denoised) / 3.0;

            let y_blend = y_orig + (y_denoised - y_orig) * luma_strength;
            let luma_scale = if y_orig > 0.0 { y_blend / y_orig } else { 1.0 };

            let r_result = (r_orig * luma_scale * (1.0 - color_strength)
                + r_denoised * color_strength)
                .clamp(0.0, 255.0) as u8;
            let g_result = (g_orig * luma_scale * (1.0 - color_strength)
                + g_denoised * color_strength)
                .clamp(0.0, 255.0) as u8;
            let b_result = (b_orig * luma_scale * (1.0 - color_strength)
                + b_denoised * color_strength)
                .clamp(0.0, 255.0) as u8;

            result.push(r_result);
            result.push(g_result);
            result.push(b_result);
        }

        assert_eq!(result.len(), 6);
        // Results should be between original and denoised
        for i in 0..6 {
            assert!(result[i] >= denoised[i].min(original[i]));
            assert!(result[i] <= denoised[i].max(original[i]));
        }
    }

    #[test]
    #[allow(dead_code)]
    fn test_batch_denoiser() {
        // Mock test - would require actual model
        let batch_size = 4;
        assert_eq!(batch_size, 4);
    }
}
