//! Perceptual noise shaping based on Just-Noticeable Difference (JND) thresholds.
//!
//! This module implements perceptual noise shaping that ensures denoising
//! artifacts remain below the JND threshold, producing visually transparent
//! results. The JND model accounts for:
//!
//! - **Luminance adaptation**: The eye is less sensitive to noise in very
//!   bright or very dark regions.
//! - **Contrast masking**: High-contrast edges mask nearby noise.
//! - **Texture masking**: Textured regions can tolerate more noise without
//!   visible artifacts.
//!
//! The shaped denoiser applies spatially varying strength so that smooth
//! regions are cleaned aggressively while textured/edge regions are preserved,
//! always staying below the perceptual visibility threshold.

use crate::{DenoiseError, DenoiseResult};
use oximedia_codec::VideoFrame;
use rayon::prelude::*;

/// Configuration for perceptual noise shaping.
#[derive(Clone, Debug)]
pub struct PerceptualConfig {
    /// Base denoising strength (0.0 = none, 1.0 = maximum).
    pub base_strength: f32,
    /// JND visibility threshold multiplier (lower = more conservative).
    /// A value of 1.0 means denoise up to exactly the JND boundary.
    /// Values below 1.0 leave a safety margin.
    pub jnd_multiplier: f32,
    /// Luminance adaptation weight (0.0 to disable, 1.0 for full effect).
    pub luminance_weight: f32,
    /// Contrast masking weight (0.0 to disable, 1.0 for full effect).
    pub contrast_weight: f32,
    /// Texture masking weight (0.0 to disable, 1.0 for full effect).
    pub texture_weight: f32,
    /// Window radius for local statistics computation.
    pub analysis_radius: usize,
}

impl Default for PerceptualConfig {
    fn default() -> Self {
        Self {
            base_strength: 0.5,
            jnd_multiplier: 0.8,
            luminance_weight: 1.0,
            contrast_weight: 1.0,
            texture_weight: 1.0,
            analysis_radius: 3,
        }
    }
}

impl PerceptualConfig {
    /// Create a conservative configuration with wide safety margin.
    #[must_use]
    pub fn conservative() -> Self {
        Self {
            base_strength: 0.4,
            jnd_multiplier: 0.5,
            ..Default::default()
        }
    }

    /// Create an aggressive configuration that denoises up to the JND limit.
    #[must_use]
    pub fn aggressive() -> Self {
        Self {
            base_strength: 0.8,
            jnd_multiplier: 1.0,
            ..Default::default()
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> DenoiseResult<()> {
        if !(0.0..=1.0).contains(&self.base_strength) {
            return Err(DenoiseError::InvalidConfig(
                "base_strength must be between 0.0 and 1.0".to_string(),
            ));
        }
        if !(0.0..=2.0).contains(&self.jnd_multiplier) {
            return Err(DenoiseError::InvalidConfig(
                "jnd_multiplier must be between 0.0 and 2.0".to_string(),
            ));
        }
        if self.analysis_radius == 0 || self.analysis_radius > 15 {
            return Err(DenoiseError::InvalidConfig(
                "analysis_radius must be between 1 and 15".to_string(),
            ));
        }
        Ok(())
    }
}

/// JND (Just-Noticeable Difference) map for a frame.
///
/// Each pixel in the map represents the maximum noise amplitude that can
/// be added/removed at that location without being perceptible.
#[derive(Clone, Debug)]
pub struct JndMap {
    /// JND threshold values per pixel (in luminance units, 0-255 scale).
    pub data: Vec<f32>,
    /// Width of the map.
    pub width: usize,
    /// Height of the map.
    pub height: usize,
}

impl JndMap {
    /// Create a new JND map with given dimensions.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            data: vec![0.0; width * height],
            width,
            height,
        }
    }

    /// Get JND threshold at a specific pixel location.
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        if x < self.width && y < self.height {
            self.data[y * self.width + x]
        } else {
            0.0
        }
    }

    /// Set JND threshold at a specific pixel location.
    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        if x < self.width && y < self.height {
            self.data[y * self.width + x] = value;
        }
    }

    /// Get the average JND threshold across the map.
    #[must_use]
    pub fn average_threshold(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.data.iter().map(|&v| f64::from(v)).sum();
        (sum / self.data.len() as f64) as f32
    }

    /// Get the minimum JND threshold (most sensitive region).
    #[must_use]
    pub fn min_threshold(&self) -> f32 {
        self.data
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min)
    }

    /// Get the maximum JND threshold (least sensitive region).
    #[must_use]
    pub fn max_threshold(&self) -> f32 {
        self.data
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
    }
}

/// Compute the luminance adaptation component of JND.
///
/// The human visual system (HVS) is less sensitive to noise in very dark
/// or very bright regions. The Weber-Fechner law models this as a
/// logarithmic sensitivity curve.
///
/// Based on Yang et al. (2005) JND model:
/// - For bg < 127: T_lum = 17 * (1 - sqrt(bg/127)) + 3
/// - For bg >= 127: T_lum = 3/128 * (bg - 127) + 3
fn luminance_adaptation(bg_luminance: f32) -> f32 {
    if bg_luminance < 127.0 {
        let normalized = (bg_luminance / 127.0).max(0.0);
        17.0 * (1.0 - normalized.sqrt()) + 3.0
    } else {
        let excess = (bg_luminance - 127.0).min(128.0);
        (3.0 / 128.0) * excess + 3.0
    }
}

/// Compute the contrast masking component of JND.
///
/// High-contrast edges mask nearby noise. The masking effect is
/// proportional to the local maximum gradient magnitude.
fn contrast_masking(max_gradient: f32) -> f32 {
    // Legge & Foley masking model (simplified)
    // Masking increases with contrast raised to 0.7 power
    if max_gradient < 1.0 {
        0.0
    } else {
        let masking = max_gradient.powf(0.7) * 0.5;
        masking.min(20.0) // Cap at reasonable maximum
    }
}

/// Compute the texture masking component of JND.
///
/// Textured regions can hide more noise than smooth regions.
/// Uses local variance as a proxy for texture complexity.
fn texture_masking(local_variance: f32) -> f32 {
    // Texture masking grows with variance but saturates
    let masking = local_variance.sqrt() * 0.3;
    masking.min(15.0)
}

/// Compute the JND map for a video frame.
///
/// Analyzes the luma plane to produce a spatially varying JND threshold
/// map. The map combines luminance adaptation, contrast masking, and
/// texture masking components.
///
/// # Arguments
/// * `frame` - Input video frame
/// * `config` - Perceptual configuration
///
/// # Returns
/// JND map with per-pixel visibility thresholds
pub fn compute_jnd_map(frame: &VideoFrame, config: &PerceptualConfig) -> DenoiseResult<JndMap> {
    if frame.planes.is_empty() {
        return Err(DenoiseError::ProcessingError(
            "Frame has no planes".to_string(),
        ));
    }

    let plane = &frame.planes[0]; // Luma plane
    let (width, height) = frame.plane_dimensions(0);
    let w = width as usize;
    let h = height as usize;
    let stride = plane.stride;
    let data: &[u8] = plane.data.as_ref();

    let radius = config.analysis_radius;
    let mut jnd_map = JndMap::new(w, h);

    // Compute per-pixel JND thresholds
    for y in 0..h {
        for x in 0..w {
            let bg_lum = f32::from(data[y * stride + x]);

            // 1. Luminance adaptation
            let t_lum = luminance_adaptation(bg_lum) * config.luminance_weight;

            // 2. Contrast masking (local gradient magnitude)
            let max_grad = compute_max_gradient(data, x, y, w, h, stride);
            let t_contrast = contrast_masking(max_grad) * config.contrast_weight;

            // 3. Texture masking (local variance)
            let local_var = compute_local_variance(data, x, y, w, h, stride, radius);
            let t_texture = texture_masking(local_var) * config.texture_weight;

            // Combined JND threshold (components add in quadrature)
            // JND = sqrt(T_lum^2 + max(T_contrast, T_texture)^2)
            let masking = t_contrast.max(t_texture);
            let jnd = (t_lum * t_lum + masking * masking).sqrt();

            jnd_map.set(x, y, jnd * config.jnd_multiplier);
        }
    }

    Ok(jnd_map)
}

/// Compute the maximum gradient magnitude at a pixel using Sobel operators.
fn compute_max_gradient(
    data: &[u8],
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    stride: usize,
) -> f32 {
    if x == 0 || x >= width - 1 || y == 0 || y >= height - 1 {
        return 0.0;
    }

    // Sobel kernels
    let get = |dx: i32, dy: i32| -> f32 {
        let nx = (x as i32 + dx).clamp(0, (width - 1) as i32) as usize;
        let ny = (y as i32 + dy).clamp(0, (height - 1) as i32) as usize;
        f32::from(data[ny * stride + nx])
    };

    let gx = -get(-1, -1) - 2.0 * get(-1, 0) - get(-1, 1)
           + get(1, -1) + 2.0 * get(1, 0) + get(1, 1);

    let gy = -get(-1, -1) - 2.0 * get(0, -1) - get(1, -1)
           + get(-1, 1) + 2.0 * get(0, 1) + get(1, 1);

    (gx * gx + gy * gy).sqrt()
}

/// Compute local variance in a window around a pixel.
fn compute_local_variance(
    data: &[u8],
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    stride: usize,
    radius: usize,
) -> f32 {
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut count = 0u32;

    for dy in -(radius as i32)..=(radius as i32) {
        let ny = (y as i32 + dy).clamp(0, (height - 1) as i32) as usize;
        for dx in -(radius as i32)..=(radius as i32) {
            let nx = (x as i32 + dx).clamp(0, (width - 1) as i32) as usize;
            let val = f64::from(data[ny * stride + nx]);
            sum += val;
            sum_sq += val * val;
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }

    let mean = sum / f64::from(count);
    let variance = (sum_sq / f64::from(count)) - (mean * mean);
    variance.max(0.0) as f32
}

/// Apply perceptual noise shaping to a video frame.
///
/// Denoises the frame using spatially varying strength derived from the
/// JND map, ensuring that artifacts remain below the visibility threshold.
///
/// # Arguments
/// * `frame` - Input video frame
/// * `config` - Perceptual configuration
///
/// # Returns
/// Perceptually shaped denoised frame
pub fn perceptual_denoise(
    frame: &VideoFrame,
    config: &PerceptualConfig,
) -> DenoiseResult<VideoFrame> {
    config.validate()?;

    if frame.planes.is_empty() {
        return Err(DenoiseError::ProcessingError(
            "Frame has no planes".to_string(),
        ));
    }

    let jnd_map = compute_jnd_map(frame, config)?;
    let mut output = frame.clone();

    // Process each plane with JND-guided strength
    output
        .planes
        .par_iter_mut()
        .enumerate()
        .try_for_each(|(plane_idx, out_plane)| {
            let input_plane = &frame.planes[plane_idx];
            let (width, height) = frame.plane_dimensions(plane_idx);
            let w = width as usize;
            let h = height as usize;
            let in_stride = input_plane.stride;
            let out_stride = out_plane.stride;
            let in_data: &[u8] = input_plane.data.as_ref();

            // Scale factor for chroma planes (JND map is at luma resolution)
            let scale_x = if w > 0 { jnd_map.width as f32 / w as f32 } else { 1.0 };
            let scale_y = if h > 0 { jnd_map.height as f32 / h as f32 } else { 1.0 };

            let filter_radius: usize = 2;

            for y in 0..h {
                for x in 0..w {
                    // Map to JND coordinates
                    let jnd_x = ((x as f32) * scale_x).min((jnd_map.width - 1) as f32) as usize;
                    let jnd_y = ((y as f32) * scale_y).min((jnd_map.height - 1) as f32) as usize;
                    let jnd_threshold = jnd_map.get(jnd_x, jnd_y);

                    // Convert JND threshold to local filter strength
                    // Higher JND = region can tolerate more change = stronger filtering
                    let local_strength = (jnd_threshold / 20.0).clamp(0.0, 1.0) * config.base_strength;

                    if local_strength < 0.01 {
                        // Very sensitive region: skip filtering
                        continue;
                    }

                    // Apply JND-guided bilateral-style filter
                    let center_val = f32::from(in_data[y * in_stride + x]);
                    let sigma_range = 15.0 * local_strength;
                    let range_coeff = if sigma_range > 0.001 {
                        -0.5 / (sigma_range * sigma_range)
                    } else {
                        0.0
                    };

                    let mut weighted_sum = 0.0f32;
                    let mut weight_sum = 0.0f32;

                    for dy in -(filter_radius as i32)..=(filter_radius as i32) {
                        let ny = (y as i32 + dy).clamp(0, (h - 1) as i32) as usize;
                        for dx in -(filter_radius as i32)..=(filter_radius as i32) {
                            let nx = (x as i32 + dx).clamp(0, (w - 1) as i32) as usize;

                            let neighbor_val = f32::from(in_data[ny * in_stride + nx]);
                            let val_diff = neighbor_val - center_val;

                            // Only allow changes within JND threshold
                            if val_diff.abs() > jnd_threshold {
                                // This neighbor is perceptually different; low weight
                                let weight = 0.01;
                                weighted_sum += neighbor_val * weight;
                                weight_sum += weight;
                            } else {
                                let weight = (val_diff * val_diff * range_coeff).exp();
                                weighted_sum += neighbor_val * weight;
                                weight_sum += weight;
                            }
                        }
                    }

                    let filtered = if weight_sum > 0.0 {
                        weighted_sum / weight_sum
                    } else {
                        center_val
                    };

                    // Clamp the change to be within JND threshold
                    let change = filtered - center_val;
                    let clamped_change = change.clamp(-jnd_threshold, jnd_threshold);
                    let result = (center_val + clamped_change).clamp(0.0, 255.0);

                    out_plane.data[y * out_stride + x] = result.round() as u8;
                }
            }

            Ok::<(), DenoiseError>(())
        })?;

    Ok(output)
}

/// Compute the perceptual error between original and denoised frames.
///
/// Returns the ratio of actual change to JND threshold at each pixel,
/// summarized as a single metric. Values below 1.0 mean all changes
/// are below JND (visually transparent).
pub fn perceptual_error_metric(
    original: &VideoFrame,
    denoised: &VideoFrame,
    config: &PerceptualConfig,
) -> DenoiseResult<PerceptualErrorReport> {
    if original.planes.is_empty() || denoised.planes.is_empty() {
        return Err(DenoiseError::ProcessingError(
            "Frame has no planes".to_string(),
        ));
    }

    let jnd_map = compute_jnd_map(original, config)?;
    let (width, height) = original.plane_dimensions(0);
    let w = width as usize;
    let h = height as usize;
    let orig_plane = &original.planes[0];
    let den_plane = &denoised.planes[0];

    let mut max_ratio = 0.0f32;
    let mut sum_ratio = 0.0f64;
    let mut above_jnd_count = 0u64;
    let total_pixels = (w * h) as u64;

    for y in 0..h {
        for x in 0..w {
            let orig_val = f32::from(orig_plane.data[y * orig_plane.stride + x]);
            let den_val = f32::from(den_plane.data[y * den_plane.stride + x]);
            let change = (den_val - orig_val).abs();
            let jnd = jnd_map.get(x, y).max(0.001); // Avoid division by zero
            let ratio = change / jnd;

            sum_ratio += f64::from(ratio);
            if ratio > max_ratio {
                max_ratio = ratio;
            }
            if ratio > 1.0 {
                above_jnd_count += 1;
            }
        }
    }

    let mean_ratio = if total_pixels > 0 {
        (sum_ratio / total_pixels as f64) as f32
    } else {
        0.0
    };

    let above_jnd_fraction = if total_pixels > 0 {
        above_jnd_count as f32 / total_pixels as f32
    } else {
        0.0
    };

    Ok(PerceptualErrorReport {
        mean_jnd_ratio: mean_ratio,
        max_jnd_ratio: max_ratio,
        above_jnd_fraction,
        total_pixels,
    })
}

/// Report from perceptual error analysis.
#[derive(Clone, Debug)]
pub struct PerceptualErrorReport {
    /// Mean ratio of actual change to JND threshold (< 1.0 = transparent).
    pub mean_jnd_ratio: f32,
    /// Maximum ratio (worst-case pixel).
    pub max_jnd_ratio: f32,
    /// Fraction of pixels where change exceeds JND threshold.
    pub above_jnd_fraction: f32,
    /// Total number of pixels analyzed.
    pub total_pixels: u64,
}

impl PerceptualErrorReport {
    /// Whether all changes are below JND (visually transparent).
    #[must_use]
    pub fn is_transparent(&self) -> bool {
        self.max_jnd_ratio <= 1.0
    }

    /// Whether the denoising is perceptually safe (< 5% pixels above JND).
    #[must_use]
    pub fn is_safe(&self) -> bool {
        self.above_jnd_fraction < 0.05
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    fn make_test_frame(width: u32, height: u32) -> VideoFrame {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, width, height);
        frame.allocate();
        frame
    }

    fn make_patterned_frame(width: u32, height: u32) -> VideoFrame {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, width, height);
        frame.allocate();
        let stride = frame.planes[0].stride;
        for y in 0..height as usize {
            for x in 0..width as usize {
                // Gradient pattern with some texture
                let val = ((x * 4 + y * 2) % 256) as u8;
                frame.planes[0].data[y * stride + x] = val;
            }
        }
        frame
    }

    #[test]
    fn test_perceptual_config_default() {
        let config = PerceptualConfig::default();
        assert!((config.base_strength - 0.5).abs() < f32::EPSILON);
        assert!((config.jnd_multiplier - 0.8).abs() < f32::EPSILON);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_perceptual_config_conservative() {
        let config = PerceptualConfig::conservative();
        assert!((config.jnd_multiplier - 0.5).abs() < f32::EPSILON);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_perceptual_config_aggressive() {
        let config = PerceptualConfig::aggressive();
        assert!((config.jnd_multiplier - 1.0).abs() < f32::EPSILON);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_bad_strength() {
        let config = PerceptualConfig {
            base_strength: 1.5,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_bad_radius() {
        let config = PerceptualConfig {
            analysis_radius: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_luminance_adaptation_dark() {
        // Dark regions should have higher threshold (less sensitive)
        let dark = luminance_adaptation(10.0);
        let mid = luminance_adaptation(127.0);
        assert!(dark > mid, "Dark regions should have higher JND threshold");
    }

    #[test]
    fn test_luminance_adaptation_bright() {
        // Bright regions should have moderately higher threshold
        let bright = luminance_adaptation(240.0);
        let mid = luminance_adaptation(127.0);
        assert!(
            bright > mid,
            "Bright regions should have higher JND threshold than mid"
        );
    }

    #[test]
    fn test_luminance_adaptation_midtone() {
        // Midtone should have the lowest threshold (most sensitive)
        let val = luminance_adaptation(127.0);
        assert!(val > 0.0, "Midtone JND should be positive");
        assert!(val < 10.0, "Midtone JND should be modest");
    }

    #[test]
    fn test_contrast_masking_low() {
        let m = contrast_masking(0.5);
        assert!((m - 0.0).abs() < f32::EPSILON, "Low contrast should give zero masking");
    }

    #[test]
    fn test_contrast_masking_high() {
        let m = contrast_masking(100.0);
        assert!(m > 0.0, "High contrast should produce masking");
        assert!(m <= 20.0, "Masking should be capped");
    }

    #[test]
    fn test_texture_masking_smooth() {
        let m = texture_masking(0.0);
        assert!((m - 0.0).abs() < f32::EPSILON, "Smooth region: no texture masking");
    }

    #[test]
    fn test_texture_masking_complex() {
        let m = texture_masking(100.0);
        assert!(m > 0.0, "Textured region should produce masking");
        assert!(m <= 15.0, "Texture masking should be capped");
    }

    #[test]
    fn test_jnd_map_creation() {
        let map = JndMap::new(64, 64);
        assert_eq!(map.width, 64);
        assert_eq!(map.height, 64);
        assert_eq!(map.data.len(), 64 * 64);
    }

    #[test]
    fn test_jnd_map_get_set() {
        let mut map = JndMap::new(32, 32);
        map.set(10, 15, 5.5);
        assert!((map.get(10, 15) - 5.5).abs() < f32::EPSILON);
        // Out of bounds should return 0
        assert!((map.get(100, 100) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_jnd_map_statistics() {
        let mut map = JndMap::new(4, 4);
        for i in 0..16 {
            map.data[i] = (i + 1) as f32;
        }
        assert!((map.min_threshold() - 1.0).abs() < f32::EPSILON);
        assert!((map.max_threshold() - 16.0).abs() < f32::EPSILON);
        // Average of 1..=16 is 8.5
        assert!((map.average_threshold() - 8.5).abs() < 0.01);
    }

    #[test]
    fn test_compute_jnd_map() {
        let frame = make_test_frame(64, 64);
        let config = PerceptualConfig::default();
        let result = compute_jnd_map(&frame, &config);
        assert!(result.is_ok());
        let map = result.expect("jnd map should be valid");
        assert_eq!(map.width, 64);
        assert_eq!(map.height, 64);
        // All thresholds should be positive
        assert!(
            map.data.iter().all(|&v| v >= 0.0),
            "All JND thresholds should be non-negative"
        );
    }

    #[test]
    fn test_compute_jnd_map_patterned() {
        let frame = make_patterned_frame(64, 64);
        let config = PerceptualConfig::default();
        let result = compute_jnd_map(&frame, &config);
        assert!(result.is_ok());
        let map = result.expect("jnd map should be valid");
        // Patterned frame should have variation in JND thresholds
        let min = map.min_threshold();
        let max = map.max_threshold();
        assert!(max > min, "Patterned frame should have varying JND thresholds");
    }

    #[test]
    fn test_perceptual_denoise_basic() {
        let frame = make_test_frame(64, 64);
        let config = PerceptualConfig::default();
        let result = perceptual_denoise(&frame, &config);
        assert!(result.is_ok());
        let output = result.expect("perceptual denoise should succeed");
        assert_eq!(output.width, 64);
        assert_eq!(output.height, 64);
    }

    #[test]
    fn test_perceptual_denoise_patterned() {
        let frame = make_patterned_frame(64, 64);
        let config = PerceptualConfig::default();
        let result = perceptual_denoise(&frame, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_perceptual_denoise_empty_frame() {
        let frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        // Not allocated
        let config = PerceptualConfig::default();
        let result = perceptual_denoise(&frame, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_perceptual_denoise_conservative_preserves_more() {
        let frame = make_patterned_frame(32, 32);
        let conservative = PerceptualConfig::conservative();
        let aggressive = PerceptualConfig::aggressive();

        let cons_result = perceptual_denoise(&frame, &conservative)
            .expect("conservative should succeed");
        let aggr_result = perceptual_denoise(&frame, &aggressive)
            .expect("aggressive should succeed");

        // Conservative should change less than aggressive
        let cons_diff = compute_total_change(&frame, &cons_result);
        let aggr_diff = compute_total_change(&frame, &aggr_result);
        assert!(
            cons_diff <= aggr_diff,
            "Conservative should change less: cons={cons_diff}, aggr={aggr_diff}"
        );
    }

    #[test]
    fn test_perceptual_error_metric() {
        let original = make_patterned_frame(64, 64);
        let config = PerceptualConfig::default();
        let denoised = perceptual_denoise(&original, &config)
            .expect("denoise should succeed");
        let report = perceptual_error_metric(&original, &denoised, &config)
            .expect("error metric should succeed");
        assert!(report.mean_jnd_ratio >= 0.0);
        assert!(report.total_pixels > 0);
    }

    #[test]
    fn test_perceptual_error_identical_frames() {
        let frame = make_patterned_frame(32, 32);
        let config = PerceptualConfig::default();
        let report = perceptual_error_metric(&frame, &frame, &config)
            .expect("error metric should succeed");
        assert!(
            (report.mean_jnd_ratio - 0.0).abs() < f32::EPSILON,
            "Identical frames should have zero error"
        );
        assert!(report.is_transparent());
        assert!(report.is_safe());
    }

    #[test]
    fn test_perceptual_error_report_safety() {
        let report = PerceptualErrorReport {
            mean_jnd_ratio: 0.3,
            max_jnd_ratio: 0.9,
            above_jnd_fraction: 0.0,
            total_pixels: 1000,
        };
        assert!(report.is_transparent());
        assert!(report.is_safe());

        let bad_report = PerceptualErrorReport {
            mean_jnd_ratio: 1.5,
            max_jnd_ratio: 3.0,
            above_jnd_fraction: 0.2,
            total_pixels: 1000,
        };
        assert!(!bad_report.is_transparent());
        assert!(!bad_report.is_safe());
    }

    /// Helper: compute total absolute pixel change between two frames (luma only).
    fn compute_total_change(a: &VideoFrame, b: &VideoFrame) -> f64 {
        let (w, h) = a.plane_dimensions(0);
        let a_plane = &a.planes[0];
        let b_plane = &b.planes[0];
        let mut total = 0.0f64;
        for y in 0..h as usize {
            for x in 0..w as usize {
                let va = f64::from(a_plane.data[y * a_plane.stride + x]);
                let vb = f64::from(b_plane.data[y * b_plane.stride + x]);
                total += (va - vb).abs();
            }
        }
        total
    }
}
