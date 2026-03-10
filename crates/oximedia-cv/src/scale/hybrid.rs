//! Hybrid scaling approaches combining multiple techniques.
//!
//! This module provides hybrid scaling methods that combine seam carving
//! with traditional scaling, cropping, and other techniques for better
//! visual results.

use super::energy::EnergyFunction;
use super::seam_carving::SeamCarver;
use crate::error::{CvError, CvResult};
use crate::image::resize::{resize_image, ResizeConfig, ResizeMethod};

/// Hybrid scaling strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridStrategy {
    /// Use only seam carving.
    SeamCarvingOnly,
    /// Use only traditional scaling.
    TraditionalOnly,
    /// Seam carve small changes, traditional scale large changes.
    Adaptive,
    /// Crop to target aspect ratio, then traditional scale.
    CropAndScale,
    /// Seam carve to intermediate size, then traditional scale.
    SeamThenScale,
    /// Traditional scale, then seam carve for final adjustment.
    ScaleThenSeam,
    /// Multi-operator approach (tries multiple strategies and picks best).
    MultiOperator,
}

/// Configuration for hybrid scaling.
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Scaling strategy to use.
    pub strategy: HybridStrategy,
    /// Energy function for seam carving.
    pub energy_function: EnergyFunction,
    /// Traditional scaling method.
    pub resize_method: ResizeMethod,
    /// Threshold for adaptive strategy (fraction of dimension).
    pub adaptive_threshold: f64,
    /// Whether to preserve aspect ratio.
    pub preserve_aspect_ratio: bool,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            strategy: HybridStrategy::Adaptive,
            energy_function: EnergyFunction::Gradient,
            resize_method: ResizeMethod::Bilinear,
            adaptive_threshold: 0.2,
            preserve_aspect_ratio: false,
        }
    }
}

impl HybridConfig {
    /// Create a new hybrid configuration.
    #[must_use]
    pub fn new(strategy: HybridStrategy) -> Self {
        Self {
            strategy,
            ..Default::default()
        }
    }

    /// Set energy function.
    #[must_use]
    pub const fn with_energy_function(mut self, energy_function: EnergyFunction) -> Self {
        self.energy_function = energy_function;
        self
    }

    /// Set resize method.
    #[must_use]
    pub const fn with_resize_method(mut self, resize_method: ResizeMethod) -> Self {
        self.resize_method = resize_method;
        self
    }

    /// Set adaptive threshold.
    #[must_use]
    pub const fn with_adaptive_threshold(mut self, threshold: f64) -> Self {
        self.adaptive_threshold = threshold;
        self
    }

    /// Set whether to preserve aspect ratio.
    #[must_use]
    pub const fn with_preserve_aspect_ratio(mut self, preserve: bool) -> Self {
        self.preserve_aspect_ratio = preserve;
        self
    }
}

/// Hybrid scaler combining multiple techniques.
#[derive(Debug, Clone)]
pub struct HybridScaler {
    config: HybridConfig,
    seam_carver: SeamCarver,
}

impl HybridScaler {
    /// Create a new hybrid scaler.
    #[must_use]
    pub fn new(config: HybridConfig) -> Self {
        let seam_carver = SeamCarver::new(config.energy_function);
        Self {
            config,
            seam_carver,
        }
    }

    /// Resize a grayscale image.
    ///
    /// # Arguments
    ///
    /// * `image` - Input grayscale image
    /// * `src_width` - Source width
    /// * `src_height` - Source height
    /// * `dst_width` - Target width
    /// * `dst_height` - Target height
    ///
    /// # Returns
    ///
    /// Resized image.
    pub fn resize(
        &self,
        image: &[u8],
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
    ) -> CvResult<Vec<u8>> {
        if src_width == 0 || src_height == 0 {
            return Err(CvError::invalid_dimensions(src_width, src_height));
        }
        if dst_width == 0 || dst_height == 0 {
            return Err(CvError::invalid_dimensions(dst_width, dst_height));
        }

        // Handle no-op case
        if src_width == dst_width && src_height == dst_height {
            return Ok(image.to_vec());
        }

        match self.config.strategy {
            HybridStrategy::SeamCarvingOnly => {
                self.resize_seam_only(image, src_width, src_height, dst_width, dst_height)
            }
            HybridStrategy::TraditionalOnly => {
                self.resize_traditional(image, src_width, src_height, dst_width, dst_height)
            }
            HybridStrategy::Adaptive => {
                self.resize_adaptive(image, src_width, src_height, dst_width, dst_height)
            }
            HybridStrategy::CropAndScale => {
                self.resize_crop_and_scale(image, src_width, src_height, dst_width, dst_height)
            }
            HybridStrategy::SeamThenScale => {
                self.resize_seam_then_scale(image, src_width, src_height, dst_width, dst_height)
            }
            HybridStrategy::ScaleThenSeam => {
                self.resize_scale_then_seam(image, src_width, src_height, dst_width, dst_height)
            }
            HybridStrategy::MultiOperator => {
                self.resize_multi_operator(image, src_width, src_height, dst_width, dst_height)
            }
        }
    }

    /// Resize using only seam carving.
    fn resize_seam_only(
        &self,
        image: &[u8],
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
    ) -> CvResult<Vec<u8>> {
        let mut result = image.to_vec();
        let mut current_width = src_width;
        let mut current_height = src_height;

        // Resize width
        if dst_width < src_width {
            result =
                self.seam_carver
                    .reduce_width(&result, current_width, current_height, dst_width)?;
            current_width = dst_width;
        } else if dst_width > src_width {
            result = self.seam_carver.enlarge_width(
                &result,
                current_width,
                current_height,
                dst_width,
            )?;
            current_width = dst_width;
        }

        // Resize height
        if dst_height < current_height {
            result = self.seam_carver.reduce_height(
                &result,
                current_width,
                current_height,
                dst_height,
            )?;
        } else if dst_height > current_height {
            // For height enlargement, we need to implement it
            // For now, fall back to traditional scaling
            let config = ResizeConfig::new(current_width, dst_height, self.config.resize_method, 1);
            result = resize_image(&result, current_width, current_height, &config)?;
        }

        Ok(result)
    }

    /// Resize using only traditional scaling.
    fn resize_traditional(
        &self,
        image: &[u8],
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
    ) -> CvResult<Vec<u8>> {
        let config = ResizeConfig::new(dst_width, dst_height, self.config.resize_method, 1);
        resize_image(image, src_width, src_height, &config)
    }

    /// Adaptive strategy: choose method based on size change.
    fn resize_adaptive(
        &self,
        image: &[u8],
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
    ) -> CvResult<Vec<u8>> {
        let width_change = (dst_width as f64 - src_width as f64).abs() / src_width as f64;
        let height_change = (dst_height as f64 - src_height as f64).abs() / src_height as f64;

        // If change is small, use seam carving
        if width_change < self.config.adaptive_threshold
            && height_change < self.config.adaptive_threshold
        {
            self.resize_seam_only(image, src_width, src_height, dst_width, dst_height)
        } else if width_change < self.config.adaptive_threshold {
            // Small width change, use seam + scale
            self.resize_seam_then_scale(image, src_width, src_height, dst_width, dst_height)
        } else if height_change < self.config.adaptive_threshold {
            // Small height change, use seam + scale
            self.resize_seam_then_scale(image, src_width, src_height, dst_width, dst_height)
        } else {
            // Large change, use traditional
            self.resize_traditional(image, src_width, src_height, dst_width, dst_height)
        }
    }

    /// Crop to aspect ratio, then scale.
    fn resize_crop_and_scale(
        &self,
        image: &[u8],
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
    ) -> CvResult<Vec<u8>> {
        let src_aspect = src_width as f64 / src_height as f64;
        let dst_aspect = dst_width as f64 / dst_height as f64;

        let (crop_width, crop_height) = if src_aspect > dst_aspect {
            // Source is wider, crop width
            let new_width = (src_height as f64 * dst_aspect).round() as u32;
            (new_width, src_height)
        } else {
            // Source is taller, crop height
            let new_height = (src_width as f64 / dst_aspect).round() as u32;
            (src_width, new_height)
        };

        // Crop from center
        let crop_x = (src_width - crop_width) / 2;
        let crop_y = (src_height - crop_height) / 2;

        let cropped = crop_image(
            image,
            src_width,
            src_height,
            crop_x,
            crop_y,
            crop_width,
            crop_height,
        )?;

        // Scale to final size
        let config = ResizeConfig::new(dst_width, dst_height, self.config.resize_method, 1);
        resize_image(&cropped, crop_width, crop_height, &config)
    }

    /// Seam carve to intermediate size, then traditional scale.
    fn resize_seam_then_scale(
        &self,
        image: &[u8],
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
    ) -> CvResult<Vec<u8>> {
        // Calculate intermediate size (50% of the change via seam carving)
        let width_diff = dst_width as i32 - src_width as i32;
        let height_diff = dst_height as i32 - src_height as i32;

        let mid_width = if width_diff != 0 {
            (src_width as i32 + width_diff / 2).max(1) as u32
        } else {
            src_width
        };

        let mid_height = if height_diff != 0 {
            (src_height as i32 + height_diff / 2).max(1) as u32
        } else {
            src_height
        };

        // Seam carve to intermediate size
        let intermediate =
            self.resize_seam_only(image, src_width, src_height, mid_width, mid_height)?;

        // Traditional scale to final size
        let config = ResizeConfig::new(dst_width, dst_height, self.config.resize_method, 1);
        resize_image(&intermediate, mid_width, mid_height, &config)
    }

    /// Traditional scale first, then seam carve for adjustment.
    fn resize_scale_then_seam(
        &self,
        image: &[u8],
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
    ) -> CvResult<Vec<u8>> {
        // Calculate intermediate size (traditional scale 80% of the way)
        let width_diff = dst_width as i32 - src_width as i32;
        let height_diff = dst_height as i32 - src_height as i32;

        let mid_width = (src_width as i32 + (width_diff * 8 / 10)).max(1) as u32;
        let mid_height = (src_height as i32 + (height_diff * 8 / 10)).max(1) as u32;

        // Traditional scale to intermediate
        let config = ResizeConfig::new(mid_width, mid_height, self.config.resize_method, 1);
        let intermediate = resize_image(image, src_width, src_height, &config)?;

        // Seam carve for final adjustment
        self.resize_seam_only(&intermediate, mid_width, mid_height, dst_width, dst_height)
    }

    /// Multi-operator approach: try multiple strategies and pick best.
    fn resize_multi_operator(
        &self,
        image: &[u8],
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
    ) -> CvResult<Vec<u8>> {
        // For now, use adaptive strategy
        // In a full implementation, we would try multiple strategies
        // and use a quality metric to pick the best result
        self.resize_adaptive(image, src_width, src_height, dst_width, dst_height)
    }

    /// Set protection mask for seam carving.
    pub fn set_protection_mask(&mut self, mask: Vec<u8>) {
        self.seam_carver.set_protection_mask(mask);
    }
}

/// Crop an image.
fn crop_image(
    image: &[u8],
    width: u32,
    height: u32,
    x: u32,
    y: u32,
    crop_width: u32,
    crop_height: u32,
) -> CvResult<Vec<u8>> {
    if x + crop_width > width || y + crop_height > height {
        return Err(CvError::invalid_roi(x, y, crop_width, crop_height));
    }

    let mut result = vec![0u8; crop_width as usize * crop_height as usize];

    for cy in 0..crop_height {
        let src_y = y + cy;
        let src_offset = src_y as usize * width as usize + x as usize;
        let dst_offset = cy as usize * crop_width as usize;

        result[dst_offset..dst_offset + crop_width as usize]
            .copy_from_slice(&image[src_offset..src_offset + crop_width as usize]);
    }

    Ok(result)
}

/// Multi-scale approach for large size changes.
///
/// Performs resizing in multiple steps to preserve quality.
#[derive(Debug)]
pub struct MultiScaleResizer {
    config: HybridConfig,
    max_scale_factor: f64,
}

impl MultiScaleResizer {
    /// Create a new multi-scale resizer.
    ///
    /// # Arguments
    ///
    /// * `config` - Hybrid configuration
    /// * `max_scale_factor` - Maximum scale factor per step
    #[must_use]
    pub const fn new(config: HybridConfig, max_scale_factor: f64) -> Self {
        Self {
            config,
            max_scale_factor,
        }
    }

    /// Resize image using multi-scale approach.
    pub fn resize(
        &self,
        image: &[u8],
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
    ) -> CvResult<Vec<u8>> {
        let width_ratio = dst_width as f64 / src_width as f64;
        let height_ratio = dst_height as f64 / src_height as f64;

        // Determine if we need multi-scale
        let needs_multiscale = width_ratio > self.max_scale_factor
            || width_ratio < 1.0 / self.max_scale_factor
            || height_ratio > self.max_scale_factor
            || height_ratio < 1.0 / self.max_scale_factor;

        if !needs_multiscale {
            // Single step is fine
            let scaler = HybridScaler::new(self.config.clone());
            return scaler.resize(image, src_width, src_height, dst_width, dst_height);
        }

        // Calculate steps
        let width_steps = calculate_steps(width_ratio, self.max_scale_factor);
        let height_steps = calculate_steps(height_ratio, self.max_scale_factor);
        let num_steps = width_steps.len().max(height_steps.len());

        let mut current_image = image.to_vec();
        let mut current_width = src_width;
        let mut current_height = src_height;

        for i in 0..num_steps {
            let target_width = if i < width_steps.len() {
                (src_width as f64 * width_steps[i]).round() as u32
            } else {
                dst_width
            };

            let target_height = if i < height_steps.len() {
                (src_height as f64 * height_steps[i]).round() as u32
            } else {
                dst_height
            };

            let scaler = HybridScaler::new(self.config.clone());
            current_image = scaler.resize(
                &current_image,
                current_width,
                current_height,
                target_width,
                target_height,
            )?;
            current_width = target_width;
            current_height = target_height;
        }

        Ok(current_image)
    }
}

/// Calculate intermediate steps for multi-scale resizing.
fn calculate_steps(ratio: f64, max_factor: f64) -> Vec<f64> {
    let mut steps = Vec::new();

    if ratio > 1.0 {
        // Upscaling
        let mut current = 1.0;
        while current * max_factor < ratio {
            current *= max_factor;
            steps.push(current);
        }
        steps.push(ratio);
    } else if ratio < 1.0 {
        // Downscaling
        let mut current = 1.0;
        while current / max_factor > ratio {
            current /= max_factor;
            steps.push(current);
        }
        steps.push(ratio);
    }

    steps
}

/// Aspect ratio preservation modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AspectMode {
    /// Stretch to fit (ignore aspect ratio).
    Stretch,
    /// Fit inside target (may have letterboxing).
    Fit,
    /// Fill target (may crop).
    Fill,
    /// Use content-aware scaling to adjust aspect ratio.
    ContentAware,
}

/// Resize with aspect ratio preservation.
pub fn resize_with_aspect_ratio(
    image: &[u8],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    mode: AspectMode,
    config: &HybridConfig,
) -> CvResult<Vec<u8>> {
    match mode {
        AspectMode::Stretch | AspectMode::ContentAware => {
            let scaler = HybridScaler::new(config.clone());
            scaler.resize(image, src_width, src_height, dst_width, dst_height)
        }
        AspectMode::Fit => resize_fit(image, src_width, src_height, dst_width, dst_height, config),
        AspectMode::Fill => {
            resize_fill(image, src_width, src_height, dst_width, dst_height, config)
        }
    }
}

/// Resize to fit inside target dimensions.
fn resize_fit(
    image: &[u8],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    config: &HybridConfig,
) -> CvResult<Vec<u8>> {
    let src_aspect = src_width as f64 / src_height as f64;
    let dst_aspect = dst_width as f64 / dst_height as f64;

    let (scale_width, scale_height) = if src_aspect > dst_aspect {
        // Fit to width
        (dst_width, (dst_width as f64 / src_aspect).round() as u32)
    } else {
        // Fit to height
        ((dst_height as f64 * src_aspect).round() as u32, dst_height)
    };

    let scaler = HybridScaler::new(config.clone());
    scaler.resize(image, src_width, src_height, scale_width, scale_height)
}

/// Resize to fill target dimensions.
fn resize_fill(
    image: &[u8],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    config: &HybridConfig,
) -> CvResult<Vec<u8>> {
    let src_aspect = src_width as f64 / src_height as f64;
    let dst_aspect = dst_width as f64 / dst_height as f64;

    let (scale_width, scale_height) = if src_aspect < dst_aspect {
        // Scale to width
        (dst_width, (dst_width as f64 / src_aspect).round() as u32)
    } else {
        // Scale to height
        ((dst_height as f64 * src_aspect).round() as u32, dst_height)
    };

    let scaler = HybridScaler::new(config.clone());
    let scaled = scaler.resize(image, src_width, src_height, scale_width, scale_height)?;

    // Crop to final size
    let crop_x = (scale_width - dst_width) / 2;
    let crop_y = (scale_height - dst_height) / 2;
    crop_image(
        &scaled,
        scale_width,
        scale_height,
        crop_x,
        crop_y,
        dst_width,
        dst_height,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_config_default() {
        let config = HybridConfig::default();
        assert_eq!(config.strategy, HybridStrategy::Adaptive);
    }

    #[test]
    fn test_hybrid_scaler_traditional() {
        let image = vec![128u8; 100];
        let config = HybridConfig::new(HybridStrategy::TraditionalOnly);
        let scaler = HybridScaler::new(config);
        let result = scaler
            .resize(&image, 10, 10, 8, 8)
            .expect("resize should succeed");
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_crop_image() {
        let image = vec![128u8; 100];
        let cropped = crop_image(&image, 10, 10, 2, 2, 4, 4).expect("crop_image should succeed");
        assert_eq!(cropped.len(), 16);
    }

    #[test]
    fn test_calculate_steps() {
        let steps = calculate_steps(4.0, 2.0);
        assert!(!steps.is_empty());
        assert_eq!(*steps.last().expect("last should succeed"), 4.0);
    }

    #[test]
    fn test_multiscale_resizer() {
        let image = vec![128u8; 100];
        let config = HybridConfig::default();
        let resizer = MultiScaleResizer::new(config, 2.0);
        let result = resizer
            .resize(&image, 10, 10, 20, 20)
            .expect("resize should succeed");
        assert_eq!(result.len(), 400);
    }

    #[test]
    fn test_resize_with_aspect_fit() {
        let image = vec![128u8; 200]; // 10x20
        let config = HybridConfig::default();
        let result = resize_with_aspect_ratio(&image, 10, 20, 10, 10, AspectMode::Fit, &config)
            .expect("resize_with_aspect_ratio should succeed");
        // Should fit height, so width will be 5
        assert_eq!(result.len(), 50); // 5x10
    }
}
