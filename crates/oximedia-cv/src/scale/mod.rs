//! Content-aware video scaling for OxiMedia.
//!
//! This module provides content-aware image and video resizing using seam carving
//! and related techniques. Content-aware scaling preserves important regions
//! (faces, objects, high-detail areas) while removing or adding less important
//! pixels, resulting in better visual quality than traditional scaling.
//!
//! # Features
//!
//! - **Seam carving**: Remove or insert optimal seams based on image energy
//! - **Energy maps**: Multiple energy functions (gradient, entropy, forward/backward)
//! - **Saliency detection**: Identify important regions automatically
//! - **Protection masks**: Prevent important regions from being removed
//! - **Hybrid scaling**: Combine seam carving with traditional scaling
//! - **Face protection**: Integrate with face detection to preserve faces
//!
//! # Example
//!
//! ```
//! use oximedia_cv::scale::{ContentAwareScaler, EnergyFunction};
//!
//! let scaler = ContentAwareScaler::builder()
//!     .energy_function(EnergyFunction::Gradient)
//!     .preserve_faces(true)
//!     .build();
//! ```

pub mod energy;
pub mod hybrid;
pub mod protection;
pub mod saliency;
pub mod seam_carving;

use crate::error::{CvError, CvResult};
use energy::compute_rgb_energy;
use oximedia_codec::VideoFrame;

// Re-export commonly used types
pub use energy::{EnergyFunction, EnergyMap};
pub use hybrid::{
    AspectMode, AspectMode as ScaleAspectMode, HybridConfig, HybridScaler, HybridStrategy,
    HybridStrategy as ScaleStrategy,
};
pub use protection::{ProtectionMask, ProtectionMaskBuilder};
pub use saliency::{SaliencyMap, SaliencyMethod};
pub use seam_carving::SeamCarver;

/// Content-aware scaler for images and video frames.
///
/// Provides intelligent resizing that preserves important content
/// using seam carving and related techniques.
#[derive(Debug, Clone)]
pub struct ContentAwareScaler {
    /// Whether to detect and preserve faces.
    preserve_faces: bool,
    /// Energy function for seam carving.
    energy_function: energy::EnergyFunction,
    /// Saliency detection method.
    saliency_method: saliency::SaliencyMethod,
    /// Hybrid scaling strategy.
    hybrid_strategy: hybrid::HybridStrategy,
    /// Protection mask (optional).
    protection_mask: Option<protection::ProtectionMask>,
    /// Saliency threshold for protection (0-255).
    saliency_threshold: u8,
    /// Whether to use saliency-based protection.
    use_saliency_protection: bool,
    /// Padding around protected regions.
    protection_padding: u32,
}

impl Default for ContentAwareScaler {
    fn default() -> Self {
        Self {
            preserve_faces: false,
            energy_function: energy::EnergyFunction::Gradient,
            saliency_method: saliency::SaliencyMethod::FrequencyTuned,
            hybrid_strategy: hybrid::HybridStrategy::Adaptive,
            protection_mask: None,
            saliency_threshold: 128,
            use_saliency_protection: true,
            protection_padding: 5,
        }
    }
}

impl ContentAwareScaler {
    /// Create a new content-aware scaler with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder for configuring the scaler.
    #[must_use]
    pub fn builder() -> ContentAwareScalerBuilder {
        ContentAwareScalerBuilder::default()
    }

    /// Set whether to preserve faces.
    pub fn set_preserve_faces(&mut self, preserve: bool) {
        self.preserve_faces = preserve;
    }

    /// Set energy function.
    pub fn set_energy_function(&mut self, energy_function: energy::EnergyFunction) {
        self.energy_function = energy_function;
    }

    /// Set saliency method.
    pub fn set_saliency_method(&mut self, method: saliency::SaliencyMethod) {
        self.saliency_method = method;
    }

    /// Set hybrid strategy.
    pub fn set_hybrid_strategy(&mut self, strategy: hybrid::HybridStrategy) {
        self.hybrid_strategy = strategy;
    }

    /// Set protection mask.
    pub fn set_protection_mask(&mut self, mask: protection::ProtectionMask) {
        self.protection_mask = Some(mask);
    }

    /// Resize a video frame.
    ///
    /// # Arguments
    ///
    /// * `input` - Input video frame
    /// * `target_width` - Target width
    /// * `target_height` - Target height
    ///
    /// # Returns
    ///
    /// Resized video frame.
    ///
    /// # Errors
    ///
    /// Returns an error if the frame format is unsupported or resizing fails.
    pub fn resize(
        &self,
        input: &VideoFrame,
        target_width: u32,
        target_height: u32,
    ) -> CvResult<VideoFrame> {
        // Create output frame
        let mut output = VideoFrame::new(input.format, target_width, target_height);
        output.timestamp = input.timestamp;
        output.frame_type = input.frame_type;
        output.color_info = input.color_info;
        output.allocate();

        // Process each plane
        let plane_count = input.format.plane_count();

        for plane_idx in 0..plane_count {
            let plane_idx_usize = plane_idx as usize;
            let (src_width, src_height) = input.plane_dimensions(plane_idx_usize);
            let (dst_width, dst_height) = output.plane_dimensions(plane_idx_usize);

            // Get source plane data
            let src_plane = &input.planes[plane_idx_usize];
            let src_data = &src_plane.data;

            // Resize plane
            let resized =
                self.resize_grayscale(src_data, src_width, src_height, dst_width, dst_height)?;

            // Copy to output plane
            let dst_plane = &mut output.planes[plane_idx_usize];
            let dst_data = dst_plane.data.clone();
            let mut dst_vec = dst_data;
            dst_vec.clear();
            dst_vec.extend_from_slice(&resized);

            // Update plane with new data
            output.planes[plane_idx_usize] = oximedia_codec::frame::Plane::with_dimensions(
                dst_vec,
                dst_width as usize,
                dst_width,
                dst_height,
            );
        }

        Ok(output)
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
    pub fn resize_grayscale(
        &self,
        image: &[u8],
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
    ) -> CvResult<Vec<u8>> {
        // Build protection mask
        let protection_mask = self.build_protection_mask(image, src_width, src_height)?;

        // Create hybrid config
        let config = hybrid::HybridConfig {
            strategy: self.hybrid_strategy,
            energy_function: self.energy_function,
            resize_method: crate::image::ResizeMethod::Bilinear,
            adaptive_threshold: 0.2,
            preserve_aspect_ratio: false,
        };

        // Create scaler and set protection mask
        let mut scaler = hybrid::HybridScaler::new(config);
        if let Some(mask) = protection_mask {
            scaler.set_protection_mask(mask.data);
        }

        // Resize
        scaler.resize(image, src_width, src_height, dst_width, dst_height)
    }

    /// Resize an RGB image.
    ///
    /// # Arguments
    ///
    /// * `image` - Input RGB image (interleaved)
    /// * `src_width` - Source width
    /// * `src_height` - Source height
    /// * `dst_width` - Target width
    /// * `dst_height` - Target height
    ///
    /// # Returns
    ///
    /// Resized RGB image.
    pub fn resize_rgb(
        &self,
        image: &[u8],
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
    ) -> CvResult<Vec<u8>> {
        // Split into channels
        let (r_channel, g_channel, b_channel) = split_rgb_channels(image, src_width, src_height)?;

        // Resize each channel
        let r_resized =
            self.resize_grayscale(&r_channel, src_width, src_height, dst_width, dst_height)?;
        let g_resized =
            self.resize_grayscale(&g_channel, src_width, src_height, dst_width, dst_height)?;
        let b_resized =
            self.resize_grayscale(&b_channel, src_width, src_height, dst_width, dst_height)?;

        // Merge channels
        Ok(merge_rgb_channels(&r_resized, &g_resized, &b_resized))
    }

    /// Resize with aspect ratio preservation.
    ///
    /// # Arguments
    ///
    /// * `image` - Input grayscale image
    /// * `src_width` - Source width
    /// * `src_height` - Source height
    /// * `dst_width` - Target width
    /// * `dst_height` - Target height
    /// * `mode` - Aspect ratio mode
    ///
    /// # Returns
    ///
    /// Resized image.
    pub fn resize_with_aspect(
        &self,
        image: &[u8],
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
        mode: hybrid::AspectMode,
    ) -> CvResult<Vec<u8>> {
        let config = hybrid::HybridConfig {
            strategy: self.hybrid_strategy,
            energy_function: self.energy_function,
            resize_method: crate::image::ResizeMethod::Bilinear,
            adaptive_threshold: 0.2,
            preserve_aspect_ratio: true,
        };

        hybrid::resize_with_aspect_ratio(
            image, src_width, src_height, dst_width, dst_height, mode, &config,
        )
    }

    /// Build protection mask based on configuration.
    fn build_protection_mask(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
    ) -> CvResult<Option<protection::ProtectionMask>> {
        // If user provided mask, use it
        if let Some(ref mask) = self.protection_mask {
            return Ok(Some(mask.clone()));
        }

        // Build mask based on saliency if enabled
        if self.use_saliency_protection {
            let saliency_map = self.compute_saliency(image, width, height)?;
            let regions = saliency_map.find_regions(self.saliency_threshold);

            if !regions.is_empty() {
                let mask = protection::ProtectionMask::from_regions(
                    width,
                    height,
                    &regions,
                    self.protection_padding,
                );
                return Ok(Some(mask));
            }
        }

        Ok(None)
    }

    /// Compute saliency map for an image.
    fn compute_saliency(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
    ) -> CvResult<saliency::SaliencyMap> {
        let data = self.saliency_method.compute(image, width, height)?;
        saliency::SaliencyMap::from_data(data, width, height)
    }
}

/// Builder for `ContentAwareScaler`.
#[derive(Debug, Default)]
pub struct ContentAwareScalerBuilder {
    preserve_faces: bool,
    energy_function: Option<energy::EnergyFunction>,
    saliency_method: Option<saliency::SaliencyMethod>,
    hybrid_strategy: Option<hybrid::HybridStrategy>,
    saliency_threshold: Option<u8>,
    use_saliency_protection: Option<bool>,
    protection_padding: Option<u32>,
}

impl ContentAwareScalerBuilder {
    /// Set whether to preserve faces.
    #[must_use]
    pub const fn preserve_faces(mut self, preserve: bool) -> Self {
        self.preserve_faces = preserve;
        self
    }

    /// Set energy function.
    #[must_use]
    pub const fn energy_function(mut self, energy_function: energy::EnergyFunction) -> Self {
        self.energy_function = Some(energy_function);
        self
    }

    /// Set saliency method.
    #[must_use]
    pub const fn saliency_method(mut self, method: saliency::SaliencyMethod) -> Self {
        self.saliency_method = Some(method);
        self
    }

    /// Set hybrid strategy.
    #[must_use]
    pub const fn hybrid_strategy(mut self, strategy: hybrid::HybridStrategy) -> Self {
        self.hybrid_strategy = Some(strategy);
        self
    }

    /// Set saliency threshold.
    #[must_use]
    pub const fn saliency_threshold(mut self, threshold: u8) -> Self {
        self.saliency_threshold = Some(threshold);
        self
    }

    /// Set whether to use saliency-based protection.
    #[must_use]
    pub const fn use_saliency_protection(mut self, use_saliency: bool) -> Self {
        self.use_saliency_protection = Some(use_saliency);
        self
    }

    /// Set protection padding.
    #[must_use]
    pub const fn protection_padding(mut self, padding: u32) -> Self {
        self.protection_padding = Some(padding);
        self
    }

    /// Build the scaler.
    #[must_use]
    pub fn build(self) -> ContentAwareScaler {
        ContentAwareScaler {
            preserve_faces: self.preserve_faces,
            energy_function: self
                .energy_function
                .unwrap_or(energy::EnergyFunction::Gradient),
            saliency_method: self
                .saliency_method
                .unwrap_or(saliency::SaliencyMethod::FrequencyTuned),
            hybrid_strategy: self
                .hybrid_strategy
                .unwrap_or(hybrid::HybridStrategy::Adaptive),
            protection_mask: None,
            saliency_threshold: self.saliency_threshold.unwrap_or(128),
            use_saliency_protection: self.use_saliency_protection.unwrap_or(true),
            protection_padding: self.protection_padding.unwrap_or(5),
        }
    }
}

/// Split RGB image into separate channels.
fn split_rgb_channels(
    image: &[u8],
    width: u32,
    height: u32,
) -> CvResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let size = width as usize * height as usize;
    let expected = size * 3;

    if image.len() < expected {
        return Err(CvError::insufficient_data(expected, image.len()));
    }

    let mut r_channel = vec![0u8; size];
    let mut g_channel = vec![0u8; size];
    let mut b_channel = vec![0u8; size];

    for i in 0..size {
        r_channel[i] = image[i * 3];
        g_channel[i] = image[i * 3 + 1];
        b_channel[i] = image[i * 3 + 2];
    }

    Ok((r_channel, g_channel, b_channel))
}

/// Merge RGB channels into interleaved image.
fn merge_rgb_channels(r: &[u8], g: &[u8], b: &[u8]) -> Vec<u8> {
    let size = r.len();
    let mut result = vec![0u8; size * 3];

    for i in 0..size {
        result[i * 3] = r[i];
        result[i * 3 + 1] = g[i];
        result[i * 3 + 2] = b[i];
    }

    result
}

/// Batch resize multiple frames with consistent settings.
///
/// This is useful for video processing where you want to maintain
/// consistent quality across frames.
#[derive(Debug)]
pub struct BatchScaler {
    scaler: ContentAwareScaler,
    target_width: u32,
    target_height: u32,
}

impl BatchScaler {
    /// Create a new batch scaler.
    #[must_use]
    pub const fn new(scaler: ContentAwareScaler, target_width: u32, target_height: u32) -> Self {
        Self {
            scaler,
            target_width,
            target_height,
        }
    }

    /// Resize a frame.
    pub fn resize_frame(&self, frame: &VideoFrame) -> CvResult<VideoFrame> {
        self.scaler
            .resize(frame, self.target_width, self.target_height)
    }

    /// Resize multiple frames.
    pub fn resize_frames(&self, frames: &[VideoFrame]) -> CvResult<Vec<VideoFrame>> {
        frames.iter().map(|f| self.resize_frame(f)).collect()
    }
}

/// Quality metrics for evaluating resize results.
#[derive(Debug, Clone, Copy)]
pub struct ResizeQuality {
    /// Mean squared error.
    pub mse: f64,
    /// Peak signal-to-noise ratio.
    pub psnr: f64,
    /// Structural similarity index.
    pub ssim: f64,
}

impl ResizeQuality {
    /// Compute quality metrics comparing two images.
    ///
    /// # Arguments
    ///
    /// * `original` - Original image
    /// * `resized` - Resized image
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Returns
    ///
    /// Quality metrics.
    pub fn compute(original: &[u8], resized: &[u8], width: u32, height: u32) -> CvResult<Self> {
        let size = width as usize * height as usize;

        if original.len() < size || resized.len() < size {
            return Err(CvError::insufficient_data(
                size,
                original.len().min(resized.len()),
            ));
        }

        // Compute MSE
        let mut mse = 0.0;
        for i in 0..size {
            let diff = original[i] as f64 - resized[i] as f64;
            mse += diff * diff;
        }
        mse /= size as f64;

        // Compute PSNR
        let psnr = if mse > f64::EPSILON {
            20.0 * (255.0f64).log10() - 10.0 * mse.log10()
        } else {
            f64::INFINITY
        };

        // Simplified SSIM (full implementation would be more complex)
        let ssim = compute_simple_ssim(original, resized, width, height);

        Ok(Self { mse, psnr, ssim })
    }
}

/// Compute simplified SSIM.
fn compute_simple_ssim(img1: &[u8], img2: &[u8], width: u32, height: u32) -> f64 {
    let size = width as usize * height as usize;

    // Compute means
    let mean1: f64 = img1[..size].iter().map(|&x| x as f64).sum::<f64>() / size as f64;
    let mean2: f64 = img2[..size].iter().map(|&x| x as f64).sum::<f64>() / size as f64;

    // Compute variances and covariance
    let mut var1 = 0.0;
    let mut var2 = 0.0;
    let mut covar = 0.0;

    for i in 0..size {
        let diff1 = img1[i] as f64 - mean1;
        let diff2 = img2[i] as f64 - mean2;
        var1 += diff1 * diff1;
        var2 += diff2 * diff2;
        covar += diff1 * diff2;
    }

    var1 /= size as f64;
    var2 /= size as f64;
    covar /= size as f64;

    // SSIM constants
    const C1: f64 = 6.5025; // (0.01 * 255)^2
    const C2: f64 = 58.5225; // (0.03 * 255)^2

    let numerator = (2.0 * mean1 * mean2 + C1) * (2.0 * covar + C2);
    let denominator = (mean1 * mean1 + mean2 * mean2 + C1) * (var1 + var2 + C2);

    numerator / denominator
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    #[test]
    fn test_content_aware_scaler_new() {
        let scaler = ContentAwareScaler::new();
        assert!(!scaler.preserve_faces);
    }

    #[test]
    fn test_builder() {
        let scaler = ContentAwareScaler::builder()
            .preserve_faces(true)
            .energy_function(energy::EnergyFunction::Forward)
            .saliency_threshold(150)
            .build();

        assert!(scaler.preserve_faces);
        assert_eq!(scaler.saliency_threshold, 150);
    }

    #[test]
    fn test_resize_grayscale() {
        let image = vec![128u8; 100];
        let scaler = ContentAwareScaler::new();
        let result = scaler
            .resize_grayscale(&image, 10, 10, 8, 8)
            .expect("resize_grayscale should succeed");
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_split_rgb_channels() {
        let image = vec![100u8, 150u8, 200u8, 100u8, 150u8, 200u8];
        let (r, g, b) =
            split_rgb_channels(&image, 2, 1).expect("split_rgb_channels should succeed");
        assert_eq!(r, vec![100, 100]);
        assert_eq!(g, vec![150, 150]);
        assert_eq!(b, vec![200, 200]);
    }

    #[test]
    fn test_merge_rgb_channels() {
        let r = vec![100u8, 100u8];
        let g = vec![150u8, 150u8];
        let b = vec![200u8, 200u8];
        let merged = merge_rgb_channels(&r, &g, &b);
        assert_eq!(merged, vec![100, 150, 200, 100, 150, 200]);
    }

    #[test]
    fn test_resize_quality_compute() {
        let img1 = vec![128u8; 100];
        let img2 = vec![128u8; 100];
        let quality = ResizeQuality::compute(&img1, &img2, 10, 10).expect("compute should succeed");
        assert_eq!(quality.mse, 0.0);
        assert_eq!(quality.psnr, f64::INFINITY);
    }

    #[test]
    fn test_batch_scaler() {
        let scaler = ContentAwareScaler::new();
        let batch = BatchScaler::new(scaler, 8, 8);
        assert_eq!(batch.target_width, 8);
        assert_eq!(batch.target_height, 8);
    }

    #[test]
    fn test_resize_video_frame() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 10, 10);
        frame.allocate();

        let scaler = ContentAwareScaler::new();
        let resized = scaler.resize(&frame, 8, 8).expect("resize should succeed");
        assert_eq!(resized.width, 8);
        assert_eq!(resized.height, 8);
    }
}
