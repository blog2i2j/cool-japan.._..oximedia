//! Video Multi-method Assessment Fusion (VMAF) implementation.
//!
//! VMAF is a perceptual video quality metric developed by Netflix that
//! combines multiple elementary metrics using machine learning (SVM).
//!
//! # Features
//!
//! VMAF combines:
//! - **VIF** (Visual Information Fidelity): Measures information loss
//! - **DLM** (Detail Loss Metric): Measures detail/texture loss
//! - **Motion**: Temporal difference information
//!
//! These features are fed into an SVM model to predict subjective quality scores.
//!
//! # Examples
//!
//! ```
//! use oximedia_cv::quality::vmaf::calculate_vmaf;
//! use oximedia_codec::VideoFrame;
//! use oximedia_core::PixelFormat;
//!
//! let mut reference = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
//! reference.allocate();
//! let mut distorted = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
//! distorted.allocate();
//!
//! let result = calculate_vmaf(&reference, &distorted)?;
//! println!("VMAF: {:.2}", result.score);
//! ```

use crate::error::{CvError, CvResult};
use oximedia_codec::VideoFrame;
use oximedia_core::PixelFormat;

/// Result of VMAF calculation.
#[derive(Debug, Clone, PartialEq)]
pub struct VmafResult {
    /// Overall VMAF score (0-100, higher is better).
    pub score: f64,

    /// Individual feature scores.
    pub features: VmafFeatures,

    /// Per-frame scores (if multiple frames analyzed).
    pub per_frame: Vec<f64>,

    /// Harmonic mean of scores (more sensitive to low scores).
    pub harmonic_mean: f64,
}

impl VmafResult {
    /// Create a new VMAF result.
    #[must_use]
    pub fn new(score: f64, features: VmafFeatures) -> Self {
        Self {
            score,
            features,
            per_frame: vec![score],
            harmonic_mean: score,
        }
    }

    /// Check if VMAF indicates acceptable quality (> 70).
    #[must_use]
    pub fn is_acceptable(&self) -> bool {
        self.score > 70.0
    }

    /// Check if VMAF indicates high quality (> 85).
    #[must_use]
    pub fn is_high_quality(&self) -> bool {
        self.score > 85.0
    }

    /// Get quality category.
    #[must_use]
    pub fn quality_category(&self) -> VmafQuality {
        VmafQuality::from_score(self.score)
    }
}

/// VMAF feature scores.
#[derive(Debug, Clone, PartialEq)]
pub struct VmafFeatures {
    /// Visual Information Fidelity (0-1, higher is better).
    pub vif: f64,

    /// Detail Loss Metric (0-1, higher is better).
    pub dlm: f64,

    /// Motion score (0-1, higher indicates more motion).
    pub motion: f64,

    /// Additional VIF scales.
    pub vif_scales: [f64; 4],
}

impl VmafFeatures {
    /// Create default features.
    #[must_use]
    pub fn new() -> Self {
        Self {
            vif: 0.0,
            dlm: 0.0,
            motion: 0.0,
            vif_scales: [0.0; 4],
        }
    }
}

impl Default for VmafFeatures {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate VMAF score between reference and distorted frames.
///
/// # Errors
///
/// Returns an error if frames are incompatible or computation fails.
///
/// # Examples
///
/// ```
/// use oximedia_cv::quality::vmaf::calculate_vmaf;
/// use oximedia_codec::VideoFrame;
/// use oximedia_core::PixelFormat;
///
/// let mut reference = VideoFrame::new(PixelFormat::Yuv420p, 1280, 720);
/// reference.allocate();
/// let mut distorted = VideoFrame::new(PixelFormat::Yuv420p, 1280, 720);
/// distorted.allocate();
///
/// let result = calculate_vmaf(&reference, &distorted)?;
/// assert!(result.score >= 0.0 && result.score <= 100.0);
/// ```
pub fn calculate_vmaf(reference: &VideoFrame, distorted: &VideoFrame) -> CvResult<VmafResult> {
    validate_frames(reference, distorted)?;

    // Extract features
    let features = extract_features(reference, distorted)?;

    // Apply SVM model to predict VMAF score
    let score = predict_vmaf_score(&features);

    Ok(VmafResult::new(score, features))
}

/// Extract VMAF features from frame pair.
fn extract_features(reference: &VideoFrame, distorted: &VideoFrame) -> CvResult<VmafFeatures> {
    // Work on luma plane
    if reference.planes.is_empty() || distorted.planes.is_empty() {
        return Err(CvError::insufficient_data(1, 0));
    }

    let ref_plane = &reference.planes[0];
    let dist_plane = &distorted.planes[0];
    let width = reference.width as usize;
    let height = reference.height as usize;
    let stride = ref_plane.stride;

    // Calculate VIF (Visual Information Fidelity)
    let vif_scales =
        calculate_vif_scales(&ref_plane.data, &dist_plane.data, width, height, stride)?;
    let vif = vif_scales.iter().sum::<f64>() / vif_scales.len() as f64;

    // Calculate DLM (Detail Loss Metric)
    let dlm = calculate_dlm(&ref_plane.data, &dist_plane.data, width, height, stride)?;

    // Calculate motion (temporal difference indicator)
    let motion = calculate_motion_score(reference)?;

    Ok(VmafFeatures {
        vif,
        dlm,
        motion,
        vif_scales,
    })
}

/// Calculate Visual Information Fidelity at multiple scales.
#[allow(clippy::too_many_arguments)]
fn calculate_vif_scales(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> CvResult<[f64; 4]> {
    let mut vif_scales = [0.0; 4];

    // Calculate VIF at 4 scales (full, 1/2, 1/4, 1/8)
    let mut current_ref = reference.to_vec();
    let mut current_dist = distorted.to_vec();
    let mut current_width = width;
    let mut current_height = height;
    let mut current_stride = stride;

    for scale in 0..4 {
        let vif = calculate_vif_single_scale(
            &current_ref,
            &current_dist,
            current_width,
            current_height,
            current_stride,
        )?;

        vif_scales[scale] = vif;

        // Downsample for next scale (except last iteration)
        if scale < 3 {
            let downsampled =
                downsample_by_2(&current_ref, current_width, current_height, current_stride);
            current_ref = downsampled;

            let downsampled =
                downsample_by_2(&current_dist, current_width, current_height, current_stride);
            current_dist = downsampled;

            current_width /= 2;
            current_height /= 2;
            current_stride = current_width;
        }
    }

    Ok(vif_scales)
}

/// Calculate VIF at a single scale.
fn calculate_vif_single_scale(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> CvResult<f64> {
    const BLOCK_SIZE: usize = 8;
    const SIGMA_NSQ: f64 = 2.0; // Noise variance

    let mut num_sum = 0.0;
    let mut den_sum = 0.0;

    // Process in blocks
    for block_y in (0..height).step_by(BLOCK_SIZE) {
        for block_x in (0..width).step_by(BLOCK_SIZE) {
            let block_height = (height - block_y).min(BLOCK_SIZE);
            let block_width = (width - block_x).min(BLOCK_SIZE);

            // Calculate block statistics
            let (mean_ref, var_ref) = calculate_block_stats(
                reference,
                block_x,
                block_y,
                block_width,
                block_height,
                stride,
            );

            let (mean_dist, var_dist) = calculate_block_stats(
                distorted,
                block_x,
                block_y,
                block_width,
                block_height,
                stride,
            );

            let covariance = calculate_block_covariance(
                reference,
                distorted,
                block_x,
                block_y,
                block_width,
                block_height,
                stride,
                mean_ref,
                mean_dist,
            );

            // VIF calculation
            let g = covariance / (var_ref + 1e-10);
            let sigma_v_sq = var_dist - (g * covariance);

            let num = (var_ref + SIGMA_NSQ).ln();
            let den =
                (var_ref * (sigma_v_sq + SIGMA_NSQ) / (g * g * var_ref + SIGMA_NSQ) + 1e-10).ln();

            if den > 0.0 {
                num_sum += num;
                den_sum += den;
            }
        }
    }

    if den_sum < 1e-10 {
        Ok(1.0)
    } else {
        Ok((num_sum / den_sum).clamp(0.0, 1.0))
    }
}

/// Calculate block statistics (mean and variance).
#[allow(clippy::too_many_arguments)]
fn calculate_block_stats(
    data: &[u8],
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    stride: usize,
) -> (f64, f64) {
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut count = 0.0;

    for dy in 0..height {
        for dx in 0..width {
            let idx = (y + dy) * stride + (x + dx);
            if idx < data.len() {
                let val = f64::from(data[idx]);
                sum += val;
                sum_sq += val * val;
                count += 1.0;
            }
        }
    }

    let mean = sum / count;
    let variance = (sum_sq / count) - (mean * mean);

    (mean, variance.max(0.0))
}

/// Calculate block covariance between two buffers.
#[allow(clippy::too_many_arguments)]
fn calculate_block_covariance(
    ref_data: &[u8],
    dist_data: &[u8],
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    stride: usize,
    mean_ref: f64,
    mean_dist: f64,
) -> f64 {
    let mut sum = 0.0;
    let mut count = 0.0;

    for dy in 0..height {
        for dx in 0..width {
            let idx = (y + dy) * stride + (x + dx);
            if idx < ref_data.len() && idx < dist_data.len() {
                let ref_val = f64::from(ref_data[idx]) - mean_ref;
                let dist_val = f64::from(dist_data[idx]) - mean_dist;
                sum += ref_val * dist_val;
                count += 1.0;
            }
        }
    }

    sum / count
}

/// Calculate Detail Loss Metric.
fn calculate_dlm(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> CvResult<f64> {
    // DLM measures loss of detail/texture using gradient magnitude
    let mut ref_gradients = Vec::with_capacity(width * height);
    let mut dist_gradients = Vec::with_capacity(width * height);

    // Calculate gradients using Sobel operators
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let ref_grad = calculate_gradient_magnitude(reference, x, y, stride);
            let dist_grad = calculate_gradient_magnitude(distorted, x, y, stride);

            ref_gradients.push(ref_grad);
            dist_gradients.push(dist_grad);
        }
    }

    // Compare gradient magnitudes
    let mut detail_loss = 0.0;
    for (ref_grad, dist_grad) in ref_gradients.iter().zip(dist_gradients.iter()) {
        let loss = (ref_grad - dist_grad).max(0.0);
        detail_loss += loss;
    }

    let avg_ref_grad = ref_gradients.iter().sum::<f64>() / ref_gradients.len() as f64;

    if avg_ref_grad < 1e-10 {
        Ok(1.0)
    } else {
        let dlm = 1.0 - (detail_loss / (ref_gradients.len() as f64 * avg_ref_grad));
        Ok(dlm.clamp(0.0, 1.0))
    }
}

/// Calculate gradient magnitude at a pixel using Sobel operator.
fn calculate_gradient_magnitude(data: &[u8], x: usize, y: usize, stride: usize) -> f64 {
    // Sobel kernels
    let get_pixel = |dx: isize, dy: isize| -> f64 {
        let px = (x as isize + dx) as usize;
        let py = (y as isize + dy) as usize;
        let idx = py * stride + px;
        if idx < data.len() {
            f64::from(data[idx])
        } else {
            0.0
        }
    };

    // Horizontal gradient (Gx)
    let gx = -get_pixel(-1, -1)
        + 1.0 * get_pixel(1, -1)
        + -2.0 * get_pixel(-1, 0)
        + 2.0 * get_pixel(1, 0)
        + -get_pixel(-1, 1)
        + 1.0 * get_pixel(1, 1);

    // Vertical gradient (Gy)
    let gy = -get_pixel(-1, -1)
        + -2.0 * get_pixel(0, -1)
        + -get_pixel(1, -1)
        + 1.0 * get_pixel(-1, 1)
        + 2.0 * get_pixel(0, 1)
        + 1.0 * get_pixel(1, 1);

    // Magnitude
    (gx * gx + gy * gy).sqrt()
}

/// Calculate motion score for a frame.
fn calculate_motion_score(frame: &VideoFrame) -> CvResult<f64> {
    // Motion is typically calculated between consecutive frames
    // For a single frame, we estimate motion by analyzing temporal gradients
    // This is a simplified version - full VMAF uses frame history

    if frame.planes.is_empty() {
        return Ok(0.0);
    }

    let plane = &frame.planes[0];
    let width = frame.width as usize;
    let height = frame.height as usize;
    let stride = plane.stride;

    // Calculate temporal activity as variance of pixel differences
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut count = 0;

    for y in 1..height {
        for x in 1..width {
            let idx = y * stride + x;
            let prev_idx = (y - 1) * stride + x;

            if idx < plane.data.len() && prev_idx < plane.data.len() {
                let diff = f64::from(plane.data[idx]) - f64::from(plane.data[prev_idx]);
                sum += diff;
                sum_sq += diff * diff;
                count += 1;
            }
        }
    }

    if count == 0 {
        return Ok(0.0);
    }

    let mean = sum / count as f64;
    let variance = (sum_sq / count as f64) - (mean * mean);

    // Normalize motion score to 0-1 range
    let motion = (variance.sqrt() / 128.0).clamp(0.0, 1.0);

    Ok(motion)
}

/// Predict VMAF score from features using SVM model.
///
/// This is a simplified SVM model based on Netflix's published parameters.
/// The full VMAF model is more complex and requires trained weights.
fn predict_vmaf_score(features: &VmafFeatures) -> f64 {
    // Simplified SVM model weights (approximation)
    // Real VMAF uses a trained libsvm model
    const VIF_WEIGHT: f64 = 0.6;
    const DLM_WEIGHT: f64 = 0.3;
    const MOTION_WEIGHT: f64 = 0.1;

    // Linear combination of features
    let linear_score = VIF_WEIGHT * features.vif
        + DLM_WEIGHT * features.dlm
        + MOTION_WEIGHT * (1.0 - features.motion); // Lower motion = higher quality perception

    // Apply non-linear transformation to match VMAF 0-100 scale
    let vmaf = 100.0 / (1.0 + (-5.0 * (linear_score - 0.5)).exp());

    // Additional adjustments based on VIF scales
    let vif_variance = calculate_variance(&features.vif_scales);
    let scale_penalty = if vif_variance > 0.1 { 5.0 } else { 0.0 };

    (vmaf - scale_penalty).clamp(0.0, 100.0)
}

/// Calculate variance of a slice.
fn calculate_variance(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance =
        values.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / values.len() as f64;
    variance
}

/// Downsample image by factor of 2.
fn downsample_by_2(data: &[u8], width: usize, height: usize, stride: usize) -> Vec<u8> {
    let new_width = width / 2;
    let new_height = height / 2;
    let mut result = vec![0u8; new_width * new_height];

    for y in 0..new_height {
        for x in 0..new_width {
            let src_y = y * 2;
            let src_x = x * 2;

            // Average 2x2 block
            let mut sum = 0u32;
            let mut count = 0u32;

            for dy in 0..2 {
                for dx in 0..2 {
                    let idx = (src_y + dy) * stride + (src_x + dx);
                    if idx < data.len() {
                        sum += u32::from(data[idx]);
                        count += 1;
                    }
                }
            }

            result[y * new_width + x] = (sum / count.max(1)) as u8;
        }
    }

    result
}

/// Validate frames for VMAF calculation.
fn validate_frames(reference: &VideoFrame, distorted: &VideoFrame) -> CvResult<()> {
    if reference.width != distorted.width || reference.height != distorted.height {
        return Err(CvError::invalid_parameter(
            "dimensions",
            format!(
                "{}x{} vs {}x{}",
                reference.width, reference.height, distorted.width, distorted.height
            ),
        ));
    }

    if reference.format != distorted.format {
        return Err(CvError::invalid_parameter(
            "pixel_format",
            "Frames must have the same pixel format",
        ));
    }

    if reference.planes.is_empty() || distorted.planes.is_empty() {
        return Err(CvError::insufficient_data(1, 0));
    }

    Ok(())
}

/// VMAF quality categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VmafQuality {
    /// Excellent quality (> 90).
    Excellent,
    /// Very good quality (80-90).
    VeryGood,
    /// Good quality (70-80).
    Good,
    /// Acceptable quality (60-70).
    Acceptable,
    /// Poor quality (50-60).
    Poor,
    /// Very poor quality (< 50).
    VeryPoor,
}

impl VmafQuality {
    /// Categorize VMAF score into quality level.
    #[must_use]
    pub fn from_score(score: f64) -> Self {
        if score >= 90.0 {
            Self::Excellent
        } else if score >= 80.0 {
            Self::VeryGood
        } else if score >= 70.0 {
            Self::Good
        } else if score >= 60.0 {
            Self::Acceptable
        } else if score >= 50.0 {
            Self::Poor
        } else {
            Self::VeryPoor
        }
    }

    /// Get descriptive string for quality level.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::Excellent => "Excellent - near transparent quality",
            Self::VeryGood => "Very Good - high quality",
            Self::Good => "Good - visible but not distracting",
            Self::Acceptable => "Acceptable - quality is adequate",
            Self::Poor => "Poor - artifacts are distracting",
            Self::VeryPoor => "Very Poor - severe quality issues",
        }
    }
}

/// Calculate VMAF for a sequence of frames.
///
/// Returns aggregate VMAF score and per-frame scores.
///
/// # Errors
///
/// Returns an error if frame sequences are incompatible.
pub fn calculate_vmaf_sequence(
    reference_frames: &[VideoFrame],
    distorted_frames: &[VideoFrame],
) -> CvResult<VmafResult> {
    if reference_frames.len() != distorted_frames.len() {
        return Err(CvError::invalid_parameter(
            "frame_count",
            format!("{} vs {}", reference_frames.len(), distorted_frames.len()),
        ));
    }

    if reference_frames.is_empty() {
        return Err(CvError::insufficient_data(1, 0));
    }

    let mut per_frame_scores = Vec::with_capacity(reference_frames.len());
    let mut all_features = Vec::with_capacity(reference_frames.len());

    // Calculate VMAF for each frame
    for (ref_frame, dist_frame) in reference_frames.iter().zip(distorted_frames.iter()) {
        let result = calculate_vmaf(ref_frame, dist_frame)?;
        per_frame_scores.push(result.score);
        all_features.push(result.features);
    }

    // Calculate aggregate metrics
    let mean_score = per_frame_scores.iter().sum::<f64>() / per_frame_scores.len() as f64;

    // Harmonic mean (more sensitive to low scores)
    let harmonic_mean = if per_frame_scores.iter().all(|&s| s > 0.0) {
        let reciprocal_sum: f64 = per_frame_scores.iter().map(|&s| 1.0 / s).sum();
        per_frame_scores.len() as f64 / reciprocal_sum
    } else {
        mean_score
    };

    // Average features
    let avg_features = average_features(&all_features);

    Ok(VmafResult {
        score: mean_score,
        features: avg_features,
        per_frame: per_frame_scores,
        harmonic_mean,
    })
}

/// Average features across multiple frames.
fn average_features(features: &[VmafFeatures]) -> VmafFeatures {
    let count = features.len() as f64;

    let vif = features.iter().map(|f| f.vif).sum::<f64>() / count;
    let dlm = features.iter().map(|f| f.dlm).sum::<f64>() / count;
    let motion = features.iter().map(|f| f.motion).sum::<f64>() / count;

    let mut vif_scales = [0.0; 4];
    for i in 0..4 {
        vif_scales[i] = features.iter().map(|f| f.vif_scales[i]).sum::<f64>() / count;
    }

    VmafFeatures {
        vif,
        dlm,
        motion,
        vif_scales,
    }
}
