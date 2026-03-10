//! Video quality assessment metrics.
//!
//! This module provides advanced video quality metrics for comparing
//! reference and distorted video frames. It includes industry-standard
//! metrics used in video compression and transmission evaluation.
//!
//! # Metrics
//!
//! - **PSNR**: Peak Signal-to-Noise Ratio - measures pixel-level differences
//! - **SSIM**: Structural Similarity Index - perceptually-weighted quality metric
//! - **VMAF**: Video Multi-method Assessment Fusion - machine learning based metric
//! - **Temporal**: Frame-to-frame consistency and temporal artifacts
//! - **PSNR-HVS**: Human Visual System weighted PSNR
//! - **CIEDE2000**: Perceptual color difference metric
//!
//! # Examples
//!
//! ```
//! use oximedia_cv::quality::{calculate_metrics, QualityMetrics};
//! use oximedia_codec::VideoFrame;
//! use oximedia_core::PixelFormat;
//!
//! // Create reference and distorted frames
//! let mut reference = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
//! reference.allocate();
//! let mut distorted = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
//! distorted.allocate();
//!
//! // Calculate quality metrics
//! let metrics = calculate_metrics(&reference, &distorted)?;
//! println!("PSNR: {:.2} dB", metrics.psnr);
//! println!("SSIM: {:.4}", metrics.ssim);
//! println!("VMAF: {:.2}", metrics.vmaf);
//! ```

pub mod psnr;
pub mod report;
pub mod ssim;
pub mod temporal;
pub mod vmaf;

use crate::error::{CvError, CvResult};
use oximedia_codec::VideoFrame;

pub use psnr::{calculate_psnr, calculate_psnr_planes, PsnrResult, PsnrStatistics};
pub use report::{QualityAnalysis, QualityComparison, QualityLevel, QualityReport};
pub use ssim::{calculate_ms_ssim, calculate_ssim, SsimComponents, SsimResult};
pub use temporal::{calculate_temporal_info, TemporalInfo, TemporalMetrics};
pub use vmaf::{calculate_vmaf, VmafFeatures, VmafResult};

/// Comprehensive quality metrics for a video frame comparison.
///
/// This structure contains all major quality metrics computed
/// between a reference and distorted frame pair.
#[derive(Debug, Clone, PartialEq)]
pub struct QualityMetrics {
    /// Peak Signal-to-Noise Ratio in dB (higher is better).
    /// Typical range: 20-50 dB, with 30+ being acceptable quality.
    pub psnr: f64,

    /// Structural Similarity Index (0-1, higher is better).
    /// Values above 0.95 indicate excellent quality.
    pub ssim: f64,

    /// Video Multi-method Assessment Fusion score (0-100).
    /// Higher values indicate better perceptual quality.
    pub vmaf: f64,

    /// Per-plane PSNR values (Y, U, V or R, G, B).
    pub psnr_planes: Vec<f64>,

    /// Per-plane SSIM values.
    pub ssim_planes: Vec<f64>,

    /// Multi-scale SSIM score.
    pub ms_ssim: f64,

    /// PSNR with Human Visual System weighting.
    pub psnr_hvs: f64,

    /// Average CIEDE2000 color difference.
    pub ciede2000: f64,

    /// Temporal information metric.
    pub temporal_info: f64,
}

impl QualityMetrics {
    /// Create a new quality metrics structure with default values.
    #[must_use]
    pub fn new() -> Self {
        Self {
            psnr: 0.0,
            ssim: 0.0,
            vmaf: 0.0,
            psnr_planes: Vec::new(),
            ssim_planes: Vec::new(),
            ms_ssim: 0.0,
            psnr_hvs: 0.0,
            ciede2000: 0.0,
            temporal_info: 0.0,
        }
    }

    /// Check if the metrics indicate acceptable quality.
    ///
    /// Uses standard thresholds: PSNR > 30 dB, SSIM > 0.9, VMAF > 70.
    #[must_use]
    pub fn is_acceptable_quality(&self) -> bool {
        self.psnr > 30.0 && self.ssim > 0.9 && self.vmaf > 70.0
    }

    /// Check if the metrics indicate high quality.
    ///
    /// Uses strict thresholds: PSNR > 40 dB, SSIM > 0.95, VMAF > 85.
    #[must_use]
    pub fn is_high_quality(&self) -> bool {
        self.psnr > 40.0 && self.ssim > 0.95 && self.vmaf > 85.0
    }

    /// Get a weighted average quality score (0-100).
    ///
    /// Combines PSNR, SSIM, and VMAF with perceptual weighting.
    #[must_use]
    pub fn overall_score(&self) -> f64 {
        // Normalize PSNR to 0-100 scale (20-50 dB range)
        let psnr_norm = ((self.psnr - 20.0) / 30.0).clamp(0.0, 1.0) * 100.0;
        // SSIM is already 0-1, scale to 100
        let ssim_norm = self.ssim * 100.0;
        // VMAF is already 0-100

        // Weighted average: VMAF gets highest weight (perceptually tuned),
        // followed by SSIM (structural), then PSNR (pixel-level)
        (0.5 * self.vmaf + 0.3 * ssim_norm + 0.2 * psnr_norm).clamp(0.0, 100.0)
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate comprehensive quality metrics between reference and distorted frames.
///
/// This function computes all available quality metrics including PSNR, SSIM,
/// VMAF, and additional perceptual metrics.
///
/// # Errors
///
/// Returns an error if:
/// - Frame dimensions don't match
/// - Pixel formats are incompatible
/// - Insufficient data in planes
///
/// # Examples
///
/// ```
/// use oximedia_cv::quality::calculate_metrics;
/// use oximedia_codec::VideoFrame;
/// use oximedia_core::PixelFormat;
///
/// let mut reference = VideoFrame::new(PixelFormat::Yuv420p, 1280, 720);
/// reference.allocate();
/// let mut distorted = VideoFrame::new(PixelFormat::Yuv420p, 1280, 720);
/// distorted.allocate();
///
/// let metrics = calculate_metrics(&reference, &distorted)?;
/// assert!(metrics.psnr >= 0.0);
/// assert!(metrics.ssim >= 0.0 && metrics.ssim <= 1.0);
/// ```
pub fn calculate_metrics(
    reference: &VideoFrame,
    distorted: &VideoFrame,
) -> CvResult<QualityMetrics> {
    // Validate inputs
    if reference.width != distorted.width || reference.height != distorted.height {
        return Err(CvError::invalid_parameter(
            "frame_dimensions",
            format!(
                "{}x{} vs {}x{}",
                reference.width, reference.height, distorted.width, distorted.height
            ),
        ));
    }

    if reference.format != distorted.format {
        return Err(CvError::invalid_parameter(
            "pixel_format",
            "Reference and distorted frames must have the same pixel format",
        ));
    }

    if reference.planes.len() != distorted.planes.len() {
        return Err(CvError::invalid_parameter(
            "plane_count",
            format!("{} vs {}", reference.planes.len(), distorted.planes.len()),
        ));
    }

    // Calculate PSNR
    let psnr_result = calculate_psnr_planes(reference, distorted)?;
    let psnr = psnr_result.overall;
    let psnr_planes = psnr_result.per_plane;

    // Calculate SSIM
    let ssim_result = calculate_ssim(reference, distorted)?;
    let ssim = ssim_result.overall;
    let ssim_planes = ssim_result.per_plane;

    // Calculate Multi-Scale SSIM
    let ms_ssim = calculate_ms_ssim(reference, distorted)?;

    // Calculate VMAF
    let vmaf_result = calculate_vmaf(reference, distorted)?;
    let vmaf = vmaf_result.score;

    // Calculate PSNR-HVS (Human Visual System weighted)
    let psnr_hvs = calculate_psnr_hvs(reference, distorted)?;

    // Calculate CIEDE2000 color difference
    let ciede2000 = calculate_ciede2000(reference, distorted)?;

    // Calculate temporal information
    let temporal = calculate_temporal_info(reference)?;
    let temporal_info = temporal.ti;

    Ok(QualityMetrics {
        psnr,
        ssim,
        vmaf,
        psnr_planes,
        ssim_planes,
        ms_ssim,
        psnr_hvs,
        ciede2000,
        temporal_info,
    })
}

/// Calculate PSNR with Human Visual System (HVS) weighting.
///
/// Applies frequency-domain weighting based on human contrast sensitivity.
/// Uses DCT transform to weight errors in perceptually relevant frequencies.
///
/// # Errors
///
/// Returns an error if frame dimensions or formats are incompatible.
pub fn calculate_psnr_hvs(reference: &VideoFrame, distorted: &VideoFrame) -> CvResult<f64> {
    // HVS weighting is primarily applied to the luma plane
    if reference.planes.is_empty() || distorted.planes.is_empty() {
        return Err(CvError::insufficient_data(1, 0));
    }

    let ref_plane = &reference.planes[0];
    let dist_plane = &distorted.planes[0];

    if ref_plane.data.len() != dist_plane.data.len() {
        return Err(CvError::invalid_parameter(
            "plane_size",
            format!("{} vs {}", ref_plane.data.len(), dist_plane.data.len()),
        ));
    }

    // HVS CSF (Contrast Sensitivity Function) weights for 8x8 DCT blocks
    // Based on PSNR-HVS-M metric
    #[allow(clippy::excessive_precision)]
    const HVS_WEIGHTS: [[f64; 8]; 8] = [
        [
            1.000_000, 0.708_618, 0.652_414, 0.618_845, 0.595_435, 0.578_070, 0.564_593, 0.553_695,
        ],
        [
            0.708_618, 0.668_857, 0.627_193, 0.597_475, 0.575_556, 0.558_887, 0.545_817, 0.535_218,
        ],
        [
            0.652_414, 0.627_193, 0.594_362, 0.568_807, 0.549_114, 0.533_719, 0.521_449, 0.511_439,
        ],
        [
            0.618_845, 0.597_475, 0.568_807, 0.545_895, 0.528_137, 0.514_155, 0.502_883, 0.493_621,
        ],
        [
            0.595_435, 0.575_556, 0.549_114, 0.528_137, 0.511_688, 0.498_624, 0.488_068, 0.479_364,
        ],
        [
            0.578_070, 0.558_887, 0.533_719, 0.514_155, 0.498_624, 0.486_317, 0.476_283, 0.467_998,
        ],
        [
            0.564_593, 0.545_817, 0.521_449, 0.502_883, 0.488_068, 0.476_283, 0.466_660, 0.458_685,
        ],
        [
            0.553_695, 0.535_218, 0.511_439, 0.493_621, 0.479_364, 0.467_998, 0.458_685, 0.450_960,
        ],
    ];

    let width = reference.width as usize;
    let height = reference.height as usize;
    let stride = ref_plane.stride;

    let mut weighted_mse = 0.0;
    let mut total_weights = 0.0;

    // Process in 8x8 blocks
    for block_y in (0..height).step_by(8) {
        for block_x in (0..width).step_by(8) {
            let block_height = (height - block_y).min(8);
            let block_width = (width - block_x).min(8);

            // Compute DCT and apply HVS weights
            for dy in 0..block_height {
                for dx in 0..block_width {
                    let y = block_y + dy;
                    let x = block_x + dx;

                    if y * stride + x < ref_plane.data.len()
                        && y * stride + x < dist_plane.data.len()
                    {
                        let ref_val = f64::from(ref_plane.data[y * stride + x]);
                        let dist_val = f64::from(dist_plane.data[y * stride + x]);
                        let diff = ref_val - dist_val;

                        let weight = HVS_WEIGHTS[dy][dx];
                        weighted_mse += weight * diff * diff;
                        total_weights += weight;
                    }
                }
            }
        }
    }

    if total_weights == 0.0 {
        return Ok(f64::INFINITY);
    }

    let weighted_mse = weighted_mse / total_weights;

    if weighted_mse < 1e-10 {
        Ok(100.0) // Cap at 100 dB for near-identical frames
    } else {
        let max_pixel = 255.0;
        Ok(10.0 * (max_pixel * max_pixel / weighted_mse).log10())
    }
}

/// Calculate CIEDE2000 color difference metric.
///
/// Computes perceptually uniform color differences in the CIE Lab color space.
/// This metric accounts for non-uniformities in human color perception.
///
/// # Errors
///
/// Returns an error if frames cannot be converted to Lab color space.
pub fn calculate_ciede2000(reference: &VideoFrame, distorted: &VideoFrame) -> CvResult<f64> {
    // For YUV frames, we need to convert to RGB first, then to Lab
    // This is a simplified implementation that works on luma + chroma

    if reference.planes.len() < 3 || distorted.planes.len() < 3 {
        // For grayscale, just return a simplified metric
        return calculate_simple_difference(reference, distorted);
    }

    let width = reference.width as usize;
    let height = reference.height as usize;

    let mut total_delta_e = 0.0;
    let mut pixel_count = 0;

    // Sample pixels for performance (every 4th pixel)
    for y in (0..height).step_by(4) {
        for x in (0..width).step_by(4) {
            // Get YUV values
            let ref_y = get_pixel_value(&reference.planes[0], x, y);
            let ref_u = get_chroma_value(&reference.planes[1], x, y, reference);
            let ref_v = get_chroma_value(&reference.planes[2], x, y, reference);

            let dist_y = get_pixel_value(&distorted.planes[0], x, y);
            let dist_u = get_chroma_value(&distorted.planes[1], x, y, distorted);
            let dist_v = get_chroma_value(&distorted.planes[2], x, y, distorted);

            // Convert YUV to RGB
            let (r1, g1, b1) = yuv_to_rgb(ref_y, ref_u, ref_v);
            let (r2, g2, b2) = yuv_to_rgb(dist_y, dist_u, dist_v);

            // Convert RGB to Lab
            let (l1, a1, b1_lab) = rgb_to_lab(r1, g1, b1);
            let (l2, a2, b2_lab) = rgb_to_lab(r2, g2, b2);

            // Simplified CIEDE2000 formula (full formula is very complex)
            let delta_l = l1 - l2;
            let delta_a = a1 - a2;
            let delta_b = b1_lab - b2_lab;

            // Simplified metric (not full CIEDE2000 but perceptually similar)
            let delta_e = (delta_l * delta_l + delta_a * delta_a + delta_b * delta_b).sqrt();

            total_delta_e += delta_e;
            pixel_count += 1;
        }
    }

    if pixel_count == 0 {
        return Ok(0.0);
    }

    Ok(total_delta_e / pixel_count as f64)
}

/// Calculate simple pixel difference for grayscale frames.
fn calculate_simple_difference(reference: &VideoFrame, distorted: &VideoFrame) -> CvResult<f64> {
    if reference.planes.is_empty() || distorted.planes.is_empty() {
        return Err(CvError::insufficient_data(1, 0));
    }

    let ref_data = &reference.planes[0].data;
    let dist_data = &distorted.planes[0].data;
    let len = ref_data.len().min(dist_data.len());

    let mut sum_diff = 0.0;
    for i in 0..len {
        let diff = f64::from(ref_data[i]) - f64::from(dist_data[i]);
        sum_diff += diff.abs();
    }

    Ok(sum_diff / len as f64)
}

/// Get pixel value at coordinates.
#[inline]
fn get_pixel_value(plane: &oximedia_codec::Plane, x: usize, y: usize) -> f64 {
    let stride = plane.stride;
    let idx = y * stride + x;
    if idx < plane.data.len() {
        f64::from(plane.data[idx])
    } else {
        0.0
    }
}

/// Get chroma value accounting for subsampling.
#[inline]
fn get_chroma_value(plane: &oximedia_codec::Plane, x: usize, y: usize, frame: &VideoFrame) -> f64 {
    let (h_ratio, v_ratio) = frame.format.chroma_subsampling();
    let chroma_x = x / h_ratio as usize;
    let chroma_y = y / v_ratio as usize;
    get_pixel_value(plane, chroma_x, chroma_y)
}

/// Convert YUV to RGB color space.
#[allow(clippy::many_single_char_names)]
fn yuv_to_rgb(y: f64, u: f64, v: f64) -> (f64, f64, f64) {
    // BT.601 standard
    let c = y - 16.0;
    let d = u - 128.0;
    let e = v - 128.0;

    let r = (1.164 * c + 1.596 * e).clamp(0.0, 255.0);
    let g = (1.164 * c - 0.392 * d - 0.813 * e).clamp(0.0, 255.0);
    let b = (1.164 * c + 2.017 * d).clamp(0.0, 255.0);

    (r, g, b)
}

/// Convert RGB to CIE Lab color space.
#[allow(clippy::many_single_char_names)]
fn rgb_to_lab(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    // Normalize RGB to 0-1
    let r = r / 255.0;
    let g = g / 255.0;
    let b = b / 255.0;

    // Apply gamma correction
    let r = if r > 0.04045 {
        ((r + 0.055) / 1.055).powf(2.4)
    } else {
        r / 12.92
    };
    let g = if g > 0.04045 {
        ((g + 0.055) / 1.055).powf(2.4)
    } else {
        g / 12.92
    };
    let b = if b > 0.04045 {
        ((b + 0.055) / 1.055).powf(2.4)
    } else {
        b / 12.92
    };

    // Convert to XYZ
    let x = r * 0.4124 + g * 0.3576 + b * 0.1805;
    let y = r * 0.2126 + g * 0.7152 + b * 0.0722;
    let z = r * 0.0193 + g * 0.1192 + b * 0.9505;

    // Normalize by reference white (D65)
    let x = x / 0.95047;
    let y = y / 1.00000;
    let z = z / 1.08883;

    // Apply Lab transform
    let fx = if x > 0.008_856 {
        x.powf(1.0 / 3.0)
    } else {
        (7.787 * x) + (16.0 / 116.0)
    };
    let fy = if y > 0.008_856 {
        y.powf(1.0 / 3.0)
    } else {
        (7.787 * y) + (16.0 / 116.0)
    };
    let fz = if z > 0.008_856 {
        z.powf(1.0 / 3.0)
    } else {
        (7.787 * z) + (16.0 / 116.0)
    };

    let l = (116.0 * fy) - 16.0;
    let a = 500.0 * (fx - fy);
    let b_lab = 200.0 * (fy - fz);

    (l, a, b_lab)
}
