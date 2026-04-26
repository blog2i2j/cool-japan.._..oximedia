//! Quality metrics calculations for video codec benchmarking.
//!
//! This module provides implementations of various quality metrics used to evaluate
//! codec performance:
//!
//! - **PSNR** (Peak Signal-to-Noise Ratio): Measures pixel-level differences
//! - **SSIM** (Structural Similarity Index): Measures perceptual quality
//! - **VMAF** (Video Multimethod Assessment Fusion): Netflix's perceptual quality metric
//!
//! # Example
//!
//! ```
//! use oximedia_bench::metrics::{MetricsCalculator, QualityMetrics};
//! use oximedia_codec::VideoFrame;
//!
//! # fn example(original: &VideoFrame, encoded: &VideoFrame) -> oximedia_bench::BenchResult<()> {
//! let calculator = MetricsCalculator::new(true, true, false);
//! let metrics = calculator.calculate(original, encoded)?;
//!
//! println!("PSNR: {:?} dB", metrics.psnr);
//! println!("SSIM: {:?}", metrics.ssim);
//! # Ok(())
//! # }
//! ```

use crate::{BenchError, BenchResult};
use oximedia_codec::VideoFrame;
use oximedia_core::PixelFormat;
use oximedia_quality::{Frame as QualityFrame, VmafCalculator};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Per-frame quality tracking
// ---------------------------------------------------------------------------

/// Per-frame quality metrics recorded during a benchmark run.
///
/// Enables temporal quality analysis such as detecting quality drops at scene
/// cuts or high-motion segments.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FrameQualityRecord {
    /// Frame index (0-based).
    pub frame_index: usize,
    /// PSNR in dB, if computed.
    pub psnr: Option<f64>,
    /// SSIM (0–1), if computed.
    pub ssim: Option<f64>,
    /// VMAF (0–100), if computed.
    pub vmaf: Option<f64>,
}

/// Temporal quality tracker accumulating per-frame [`FrameQualityRecord`]s.
///
/// After pushing all frames call [`TemporalQualityTracker::summary`] to obtain
/// aggregate statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TemporalQualityTracker {
    /// Per-frame records in presentation order.
    pub records: Vec<FrameQualityRecord>,
}

impl TemporalQualityTracker {
    /// Create an empty tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a per-frame quality measurement.
    pub fn push(&mut self, frame_index: usize, metrics: &QualityMetrics) {
        self.records.push(FrameQualityRecord {
            frame_index,
            psnr: metrics.psnr,
            ssim: metrics.ssim,
            vmaf: metrics.vmaf,
        });
    }

    /// Return the number of frames tracked.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.records.len()
    }

    /// Compute aggregate summary over all tracked frames.
    ///
    /// Returns `None` when no frames have been recorded.
    #[must_use]
    pub fn summary(&self) -> Option<TemporalQualitySummary> {
        if self.records.is_empty() {
            return None;
        }

        let psnr_vals: Vec<f64> = self.records.iter().filter_map(|r| r.psnr).collect();
        let ssim_vals: Vec<f64> = self.records.iter().filter_map(|r| r.ssim).collect();
        let vmaf_vals: Vec<f64> = self.records.iter().filter_map(|r| r.vmaf).collect();

        let mean_psnr = if psnr_vals.is_empty() {
            None
        } else {
            Some(psnr_vals.iter().sum::<f64>() / psnr_vals.len() as f64)
        };
        let min_psnr = psnr_vals.iter().copied().reduce(f64::min);
        let max_psnr = psnr_vals.iter().copied().reduce(f64::max);

        let mean_ssim = if ssim_vals.is_empty() {
            None
        } else {
            Some(ssim_vals.iter().sum::<f64>() / ssim_vals.len() as f64)
        };
        let min_ssim = ssim_vals.iter().copied().reduce(f64::min);
        let max_ssim = ssim_vals.iter().copied().reduce(f64::max);

        let mean_vmaf = if vmaf_vals.is_empty() {
            None
        } else {
            Some(vmaf_vals.iter().sum::<f64>() / vmaf_vals.len() as f64)
        };
        let min_vmaf = vmaf_vals.iter().copied().reduce(f64::min);

        // Detect the worst frame (lowest PSNR, if available).
        let worst_frame_index = if psnr_vals.is_empty() {
            None
        } else {
            self.records
                .iter()
                .filter_map(|r| r.psnr.map(|p| (r.frame_index, p)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
        };

        Some(TemporalQualitySummary {
            frame_count: self.records.len(),
            mean_psnr,
            min_psnr,
            max_psnr,
            mean_ssim,
            min_ssim,
            max_ssim,
            mean_vmaf,
            min_vmaf,
            worst_frame_index,
        })
    }
}

/// Aggregate summary produced by [`TemporalQualityTracker::summary`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalQualitySummary {
    /// Total number of frames measured.
    pub frame_count: usize,
    /// Mean PSNR across all frames.
    pub mean_psnr: Option<f64>,
    /// Minimum (worst) PSNR across all frames.
    pub min_psnr: Option<f64>,
    /// Maximum (best) PSNR across all frames.
    pub max_psnr: Option<f64>,
    /// Mean SSIM across all frames.
    pub mean_ssim: Option<f64>,
    /// Minimum SSIM across all frames.
    pub min_ssim: Option<f64>,
    /// Maximum SSIM across all frames.
    pub max_ssim: Option<f64>,
    /// Mean VMAF across all frames.
    pub mean_vmaf: Option<f64>,
    /// Minimum VMAF across all frames.
    pub min_vmaf: Option<f64>,
    /// Index of the frame with the lowest PSNR.
    pub worst_frame_index: Option<usize>,
}

/// Quality metrics for a video frame or sequence.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Peak Signal-to-Noise Ratio in dB
    pub psnr: Option<f64>,

    /// Structural Similarity Index (0.0 to 1.0)
    pub ssim: Option<f64>,

    /// Video Multimethod Assessment Fusion score (0.0 to 100.0)
    pub vmaf: Option<f64>,

    /// Mean Squared Error
    pub mse: Option<f64>,

    /// Per-component PSNR (Y, U, V)
    pub psnr_y: Option<f64>,
    /// U component PSNR
    pub psnr_u: Option<f64>,
    /// V component PSNR
    pub psnr_v: Option<f64>,

    /// Per-component SSIM (Y, U, V)
    pub ssim_y: Option<f64>,
    /// U component SSIM
    pub ssim_u: Option<f64>,
    /// V component SSIM
    pub ssim_v: Option<f64>,
}

impl QualityMetrics {
    /// Create empty metrics.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Check if any metrics are available.
    #[must_use]
    pub fn has_any(&self) -> bool {
        self.psnr.is_some() || self.ssim.is_some() || self.vmaf.is_some()
    }

    /// Get the overall quality score (normalized 0-100).
    #[must_use]
    pub fn overall_score(&self) -> Option<f64> {
        if let Some(vmaf) = self.vmaf {
            return Some(vmaf);
        }

        if let Some(ssim) = self.ssim {
            return Some(ssim * 100.0);
        }

        if let Some(psnr) = self.psnr {
            // Normalize PSNR (typically 20-50 dB) to 0-100 scale
            return Some(((psnr - 20.0) / 30.0 * 100.0).clamp(0.0, 100.0));
        }

        None
    }
}

/// Calculator for video quality metrics.
pub struct MetricsCalculator {
    enable_psnr: bool,
    enable_ssim: bool,
    enable_vmaf: bool,
    vmaf_model_path: Option<String>,
}

impl MetricsCalculator {
    /// Create a new metrics calculator.
    #[must_use]
    pub fn new(enable_psnr: bool, enable_ssim: bool, enable_vmaf: bool) -> Self {
        Self {
            enable_psnr,
            enable_ssim,
            enable_vmaf,
            vmaf_model_path: None,
        }
    }

    /// Set the VMAF model path.
    #[must_use]
    pub fn with_vmaf_model(mut self, path: impl Into<String>) -> Self {
        self.vmaf_model_path = Some(path.into());
        self
    }

    /// Calculate quality metrics between original and encoded frames.
    ///
    /// # Errors
    ///
    /// Returns an error if metric calculation fails.
    pub fn calculate(
        &self,
        original: &VideoFrame,
        encoded: &VideoFrame,
    ) -> BenchResult<QualityMetrics> {
        let mut metrics = QualityMetrics::default();

        // Validate frame dimensions match
        if original.width != encoded.width || original.height != encoded.height {
            return Err(BenchError::MetricFailed(
                "Frame dimensions do not match".to_string(),
            ));
        }

        if self.enable_psnr {
            let (psnr, psnr_y, psnr_u, psnr_v) = calculate_psnr(original, encoded)?;
            metrics.psnr = Some(psnr);
            metrics.psnr_y = Some(psnr_y);
            metrics.psnr_u = Some(psnr_u);
            metrics.psnr_v = Some(psnr_v);
            metrics.mse = Some(psnr_to_mse(psnr));
        }

        if self.enable_ssim {
            let (ssim, ssim_y, ssim_u, ssim_v) = calculate_ssim(original, encoded)?;
            metrics.ssim = Some(ssim);
            metrics.ssim_y = Some(ssim_y);
            metrics.ssim_u = Some(ssim_u);
            metrics.ssim_v = Some(ssim_v);
        }

        if self.enable_vmaf {
            metrics.vmaf = Some(calculate_vmaf(
                original,
                encoded,
                self.vmaf_model_path.as_deref(),
            )?);
        }

        Ok(metrics)
    }

    /// Calculate metrics for a sequence of frames.
    ///
    /// # Errors
    ///
    /// Returns an error if metric calculation fails.
    pub fn calculate_sequence(
        &self,
        original_frames: &[VideoFrame],
        encoded_frames: &[VideoFrame],
    ) -> BenchResult<QualityMetrics> {
        if original_frames.len() != encoded_frames.len() {
            return Err(BenchError::MetricFailed("Frame count mismatch".to_string()));
        }

        if original_frames.is_empty() {
            return Ok(QualityMetrics::empty());
        }

        // Calculate metrics for each frame and average
        let mut total_metrics = QualityMetrics::default();
        let mut count = 0;

        for (orig, enc) in original_frames.iter().zip(encoded_frames) {
            let frame_metrics = self.calculate(orig, enc)?;
            add_metrics(&mut total_metrics, &frame_metrics);
            count += 1;
        }

        Ok(average_metrics(&total_metrics, count))
    }
}

/// Calculate PSNR (Peak Signal-to-Noise Ratio) between two frames.
///
/// Returns overall PSNR and per-component PSNR (Y, U, V).
fn calculate_psnr(
    original: &VideoFrame,
    encoded: &VideoFrame,
) -> BenchResult<(f64, f64, f64, f64)> {
    let width = original.width as usize;
    let height = original.height as usize;

    // Get plane data
    let orig_y = original
        .planes
        .first()
        .ok_or_else(|| BenchError::MetricFailed("Missing Y plane in original".to_string()))?;
    let enc_y = encoded
        .planes
        .first()
        .ok_or_else(|| BenchError::MetricFailed("Missing Y plane in encoded".to_string()))?;

    let orig_u = original
        .planes
        .get(1)
        .ok_or_else(|| BenchError::MetricFailed("Missing U plane in original".to_string()))?;
    let enc_u = encoded
        .planes
        .get(1)
        .ok_or_else(|| BenchError::MetricFailed("Missing U plane in encoded".to_string()))?;

    let orig_v = original
        .planes
        .get(2)
        .ok_or_else(|| BenchError::MetricFailed("Missing V plane in original".to_string()))?;
    let enc_v = encoded
        .planes
        .get(2)
        .ok_or_else(|| BenchError::MetricFailed("Missing V plane in encoded".to_string()))?;

    // Calculate MSE for each plane
    let mse_y = calculate_mse(&orig_y.data, &enc_y.data, width, height, orig_y.stride);
    let mse_u = calculate_mse(
        &orig_u.data,
        &enc_u.data,
        width / 2,
        height / 2,
        orig_u.stride,
    );
    let mse_v = calculate_mse(
        &orig_v.data,
        &enc_v.data,
        width / 2,
        height / 2,
        orig_v.stride,
    );

    // Convert MSE to PSNR
    let psnr_y = mse_to_psnr(mse_y);
    let psnr_u = mse_to_psnr(mse_u);
    let psnr_v = mse_to_psnr(mse_v);

    // Calculate overall PSNR (weighted average favoring luma)
    let psnr = (6.0 * psnr_y + psnr_u + psnr_v) / 8.0;

    Ok((psnr, psnr_y, psnr_u, psnr_v))
}

/// Calculate Mean Squared Error between two planes.
fn calculate_mse(orig: &[u8], enc: &[u8], width: usize, height: usize, stride: usize) -> f64 {
    let mut sum_squared_diff: u64 = 0;

    for y in 0..height {
        for x in 0..width {
            let idx = y * stride + x;
            let diff = i32::from(orig[idx]) - i32::from(enc[idx]);
            sum_squared_diff += u64::try_from(diff * diff).unwrap_or(0);
        }
    }

    sum_squared_diff as f64 / (width * height) as f64
}

/// Convert MSE to PSNR.
fn mse_to_psnr(mse: f64) -> f64 {
    if mse < f64::EPSILON {
        return 100.0; // Perfect match
    }
    10.0 * (255.0 * 255.0 / mse).log10()
}

/// Convert PSNR to MSE.
fn psnr_to_mse(psnr: f64) -> f64 {
    if psnr >= 100.0 {
        return 0.0;
    }
    255.0 * 255.0 / 10.0_f64.powf(psnr / 10.0)
}

/// Calculate SSIM (Structural Similarity Index) between two frames.
///
/// Returns overall SSIM and per-component SSIM (Y, U, V).
fn calculate_ssim(
    original: &VideoFrame,
    encoded: &VideoFrame,
) -> BenchResult<(f64, f64, f64, f64)> {
    let width = original.width as usize;
    let height = original.height as usize;

    // Get plane data
    let orig_y = original
        .planes
        .first()
        .ok_or_else(|| BenchError::MetricFailed("Missing Y plane in original".to_string()))?;
    let enc_y = encoded
        .planes
        .first()
        .ok_or_else(|| BenchError::MetricFailed("Missing Y plane in encoded".to_string()))?;

    let orig_u = original
        .planes
        .get(1)
        .ok_or_else(|| BenchError::MetricFailed("Missing U plane in original".to_string()))?;
    let enc_u = encoded
        .planes
        .get(1)
        .ok_or_else(|| BenchError::MetricFailed("Missing U plane in encoded".to_string()))?;

    let orig_v = original
        .planes
        .get(2)
        .ok_or_else(|| BenchError::MetricFailed("Missing V plane in original".to_string()))?;
    let enc_v = encoded
        .planes
        .get(2)
        .ok_or_else(|| BenchError::MetricFailed("Missing V plane in encoded".to_string()))?;

    // Calculate SSIM for each plane
    let ssim_y = calculate_ssim_plane(&orig_y.data, &enc_y.data, width, height, orig_y.stride);
    let ssim_u = calculate_ssim_plane(
        &orig_u.data,
        &enc_u.data,
        width / 2,
        height / 2,
        orig_u.stride,
    );
    let ssim_v = calculate_ssim_plane(
        &orig_v.data,
        &enc_v.data,
        width / 2,
        height / 2,
        orig_v.stride,
    );

    // Calculate overall SSIM (weighted average favoring luma)
    let ssim = (6.0 * ssim_y + ssim_u + ssim_v) / 8.0;

    Ok((ssim, ssim_y, ssim_u, ssim_v))
}

/// Calculate SSIM for a single plane using a sliding window approach.
fn calculate_ssim_plane(
    orig: &[u8],
    enc: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> f64 {
    const WINDOW_SIZE: usize = 8;
    const C1: f64 = (0.01 * 255.0) * (0.01 * 255.0);
    const C2: f64 = (0.03 * 255.0) * (0.03 * 255.0);

    let mut ssim_sum = 0.0;
    let mut window_count = 0;

    // Slide window across the image
    for y in 0..=(height.saturating_sub(WINDOW_SIZE)) {
        for x in 0..=(width.saturating_sub(WINDOW_SIZE)) {
            let ssim = calculate_ssim_window(orig, enc, x, y, WINDOW_SIZE, stride, C1, C2);
            ssim_sum += ssim;
            window_count += 1;
        }
    }

    if window_count == 0 {
        return 1.0; // Default to perfect similarity for very small images
    }

    ssim_sum / window_count as f64
}

/// Calculate SSIM for a single window.
#[allow(clippy::too_many_arguments)]
fn calculate_ssim_window(
    orig: &[u8],
    enc: &[u8],
    x: usize,
    y: usize,
    window_size: usize,
    stride: usize,
    c1: f64,
    c2: f64,
) -> f64 {
    let mut sum_orig: u32 = 0;
    let mut sum_enc: u32 = 0;
    let mut sum_orig_sq: u64 = 0;
    let mut sum_enc_sq: u64 = 0;
    let mut sum_orig_enc: u64 = 0;

    // Calculate statistics for the window
    for dy in 0..window_size {
        for dx in 0..window_size {
            let idx = (y + dy) * stride + (x + dx);
            let orig_val = u32::from(orig[idx]);
            let enc_val = u32::from(enc[idx]);

            sum_orig += orig_val;
            sum_enc += enc_val;
            sum_orig_sq += u64::from(orig_val * orig_val);
            sum_enc_sq += u64::from(enc_val * enc_val);
            sum_orig_enc += u64::from(orig_val * enc_val);
        }
    }

    let n = (window_size * window_size) as f64;
    let mean_orig = sum_orig as f64 / n;
    let mean_enc = sum_enc as f64 / n;

    let var_orig = sum_orig_sq as f64 / n - mean_orig * mean_orig;
    let var_enc = sum_enc_sq as f64 / n - mean_enc * mean_enc;
    let covar = sum_orig_enc as f64 / n - mean_orig * mean_enc;

    // SSIM formula
    let numerator = (2.0 * mean_orig * mean_enc + c1) * (2.0 * covar + c2);
    let denominator =
        (mean_orig * mean_orig + mean_enc * mean_enc + c1) * (var_orig + var_enc + c2);

    if denominator.abs() < f64::EPSILON {
        1.0
    } else {
        numerator / denominator
    }
}

/// Convert a [`VideoFrame`] Y plane into a [`QualityFrame`] for VMAF assessment.
///
/// Only the luma (Y) plane is extracted, which is sufficient for the
/// pure-Rust `VmafCalculator` implementation.
fn video_frame_to_quality_frame(vf: &VideoFrame) -> BenchResult<QualityFrame> {
    let y_plane = vf
        .planes
        .first()
        .ok_or_else(|| BenchError::MetricFailed("Missing Y plane".to_string()))?;

    let width = vf.width as usize;
    let height = vf.height as usize;

    // Build a quality Frame with only the luma plane; VMAF reads planes[0].
    let mut qf = QualityFrame::new(width, height, PixelFormat::Yuv420p)
        .map_err(|e| BenchError::MetricFailed(e.to_string()))?;

    // Copy Y data row-by-row to handle differing strides.
    let stride = y_plane.stride;
    let y_dst = qf
        .planes
        .first_mut()
        .ok_or_else(|| BenchError::MetricFailed("QualityFrame has no Y plane".to_string()))?;
    for row in 0..height {
        let src_start = row * stride;
        let src_end = src_start + width;
        if src_end <= y_plane.data.len() {
            let dst_start = row * width;
            let dst_end = dst_start + width;
            y_dst[dst_start..dst_end].copy_from_slice(&y_plane.data[src_start..src_end]);
        }
    }

    Ok(qf)
}

/// Calculate VMAF (Video Multimethod Assessment Fusion) score using the
/// pure-Rust [`VmafCalculator`] from `oximedia-quality`.
///
/// Returns a score in the range \[0, 100\].
fn calculate_vmaf(
    original: &VideoFrame,
    encoded: &VideoFrame,
    _model_path: Option<&str>,
) -> BenchResult<f64> {
    let ref_frame = video_frame_to_quality_frame(original)?;
    let dis_frame = video_frame_to_quality_frame(encoded)?;

    let calc = VmafCalculator::new();
    let score = calc
        .calculate(&ref_frame, &dis_frame)
        .map_err(|e| BenchError::MetricFailed(e.to_string()))?;

    Ok(score.score)
}

// ---------------------------------------------------------------------------
// MS-SSIM (Multi-Scale SSIM)
// ---------------------------------------------------------------------------

/// Standard MS-SSIM scale weights from Wang et al. (2003).
const MS_SSIM_WEIGHTS: [f64; 5] = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333];

/// Compute Multi-Scale SSIM (MS-SSIM) between two grayscale pixel slices.
///
/// Both slices must have exactly `width * height` elements.  The function
/// performs 5-scale decomposition using 2× downsampling (local averaging) and
/// combines the per-scale SSIM contrast-structure terms with the luminance
/// term from the finest scale using the standard weights
/// `[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]`.
///
/// Returns a value in `[0.0, 1.0]` where `1.0` denotes perfect structural
/// similarity.  Returns `1.0` for identical inputs and falls back to a
/// single-scale SSIM when the image is too small for all 5 scales.
///
/// # Panics
///
/// Does not panic; inputs smaller than 8×8 return `1.0` as degenerate images
/// are assumed identical.
#[must_use]
pub fn compute_ms_ssim(ref_pixels: &[f32], test_pixels: &[f32], width: u32, height: u32) -> f64 {
    let w = width as usize;
    let h = height as usize;

    if ref_pixels.len() != w * h || test_pixels.len() != w * h {
        return 1.0; // mismatched dimensions — treat as degenerate
    }

    if w < 8 || h < 8 {
        return 1.0;
    }

    const NUM_SCALES: usize = 5;
    let mut luminance_product = 1.0_f64;
    let mut cs_products = [1.0_f64; NUM_SCALES];

    // Iterate over scales, downsampling by 2× at each stage.
    let mut cur_ref: Vec<f32> = ref_pixels.to_vec();
    let mut cur_tst: Vec<f32> = test_pixels.to_vec();
    let mut cur_w = w;
    let mut cur_h = h;

    for scale in 0..NUM_SCALES {
        // Compute SSIM statistics at this scale.
        let (lum, cs) = ssim_luminance_cs(&cur_ref, &cur_tst, cur_w, cur_h);

        cs_products[scale] = cs;
        if scale == NUM_SCALES - 1 {
            luminance_product = lum;
        }

        // Downsample by 2× for next scale (unless this is the last scale).
        if scale + 1 < NUM_SCALES {
            let new_w = cur_w / 2;
            let new_h = cur_h / 2;
            if new_w < 4 || new_h < 4 {
                // Image too small to continue; stop here.
                break;
            }
            cur_ref = downsample_2x(&cur_ref, cur_w, cur_h);
            cur_tst = downsample_2x(&cur_tst, cur_w, cur_h);
            cur_w = new_w;
            cur_h = new_h;
        }
    }

    // Combine: MS-SSIM = lum^w_M * prod_j(cs_j^w_j)
    // Guard against NaN/negative by clamping all intermediate values.
    let lum_clamped = luminance_product.clamp(0.0, 1.0);
    let mut result = lum_clamped.powf(MS_SSIM_WEIGHTS[NUM_SCALES - 1]);
    for (j, &cs) in cs_products.iter().enumerate() {
        let cs_safe = cs.clamp(0.0, 1.0);
        result *= cs_safe.powf(MS_SSIM_WEIGHTS[j]);
    }

    if result.is_nan() {
        1.0
    } else {
        result.clamp(0.0, 1.0)
    }
}

/// Compute the luminance and contrast-structure components of SSIM
/// for two f32 grayscale planes.
///
/// Returns `(luminance, cs)` each in `[0.0, 1.0]`.  Degenerate (constant)
/// images return `(1.0, 1.0)` to avoid division-by-zero / NaN.
fn ssim_luminance_cs(ref_px: &[f32], tst_px: &[f32], width: usize, height: usize) -> (f64, f64) {
    // SSIM stability constants for 0–255 range.
    const C1: f64 = (0.01 * 255.0) * (0.01 * 255.0); // ≈ 6.5025
    const C2: f64 = (0.03 * 255.0) * (0.03 * 255.0); // ≈ 58.5225

    let n = (width * height) as f64;
    if n == 0.0 {
        return (1.0, 1.0);
    }

    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut sum_xx = 0.0_f64;
    let mut sum_yy = 0.0_f64;
    let mut sum_xy = 0.0_f64;

    let len = (width * height).min(ref_px.len()).min(tst_px.len());
    for i in 0..len {
        let x = f64::from(ref_px[i]);
        let y = f64::from(tst_px[i]);
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_yy += y * y;
        sum_xy += x * y;
    }

    let mu_x = sum_x / n;
    let mu_y = sum_y / n;
    // Clamp numerical noise to avoid tiny negatives for constant images.
    let sigma_x2 = ((sum_xx / n) - mu_x * mu_x).max(0.0);
    let sigma_y2 = ((sum_yy / n) - mu_y * mu_y).max(0.0);
    let sigma_xy = (sum_xy / n) - mu_x * mu_y;

    let lum_denom = mu_x * mu_x + mu_y * mu_y + C1;
    let luminance = if lum_denom.abs() < f64::EPSILON {
        1.0
    } else {
        ((2.0 * mu_x * mu_y + C1) / lum_denom).clamp(0.0, 1.0)
    };

    let cs_denom = sigma_x2 + sigma_y2 + C2;
    let cs = if cs_denom.abs() < f64::EPSILON {
        1.0
    } else {
        ((2.0 * sigma_xy + C2) / cs_denom).clamp(0.0, 1.0)
    };

    (luminance, cs)
}

/// Downsample a grayscale f32 image by 2× using 2×2 box averaging.
fn downsample_2x(pixels: &[f32], width: usize, height: usize) -> Vec<f32> {
    let new_w = width / 2;
    let new_h = height / 2;
    let mut out = Vec::with_capacity(new_w * new_h);

    for row in 0..new_h {
        for col in 0..new_w {
            let r0 = row * 2;
            let c0 = col * 2;
            let idx00 = r0 * width + c0;
            let idx01 = r0 * width + (c0 + 1).min(width - 1);
            let idx10 = (r0 + 1).min(height - 1) * width + c0;
            let idx11 = (r0 + 1).min(height - 1) * width + (c0 + 1).min(width - 1);

            let v = (f64::from(pixels[idx00])
                + f64::from(pixels[idx01])
                + f64::from(pixels[idx10])
                + f64::from(pixels[idx11]))
                / 4.0;
            out.push(v as f32);
        }
    }
    out
}

/// Add metrics together for averaging.
fn add_metrics(total: &mut QualityMetrics, metrics: &QualityMetrics) {
    if let Some(psnr) = metrics.psnr {
        *total.psnr.get_or_insert(0.0) += psnr;
    }
    if let Some(ssim) = metrics.ssim {
        *total.ssim.get_or_insert(0.0) += ssim;
    }
    if let Some(vmaf) = metrics.vmaf {
        *total.vmaf.get_or_insert(0.0) += vmaf;
    }
    if let Some(mse) = metrics.mse {
        *total.mse.get_or_insert(0.0) += mse;
    }
    if let Some(psnr_y) = metrics.psnr_y {
        *total.psnr_y.get_or_insert(0.0) += psnr_y;
    }
    if let Some(psnr_u) = metrics.psnr_u {
        *total.psnr_u.get_or_insert(0.0) += psnr_u;
    }
    if let Some(psnr_v) = metrics.psnr_v {
        *total.psnr_v.get_or_insert(0.0) += psnr_v;
    }
    if let Some(ssim_y) = metrics.ssim_y {
        *total.ssim_y.get_or_insert(0.0) += ssim_y;
    }
    if let Some(ssim_u) = metrics.ssim_u {
        *total.ssim_u.get_or_insert(0.0) += ssim_u;
    }
    if let Some(ssim_v) = metrics.ssim_v {
        *total.ssim_v.get_or_insert(0.0) += ssim_v;
    }
}

/// Average metrics by dividing by count.
fn average_metrics(total: &QualityMetrics, count: usize) -> QualityMetrics {
    let count_f = count as f64;
    QualityMetrics {
        psnr: total.psnr.map(|v| v / count_f),
        ssim: total.ssim.map(|v| v / count_f),
        vmaf: total.vmaf.map(|v| v / count_f),
        mse: total.mse.map(|v| v / count_f),
        psnr_y: total.psnr_y.map(|v| v / count_f),
        psnr_u: total.psnr_u.map(|v| v / count_f),
        psnr_v: total.psnr_v.map(|v| v / count_f),
        ssim_y: total.ssim_y.map(|v| v / count_f),
        ssim_u: total.ssim_u.map(|v| v / count_f),
        ssim_v: total.ssim_v.map(|v| v / count_f),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_to_psnr() {
        let psnr = mse_to_psnr(100.0);
        assert!(psnr > 0.0);
        assert!(psnr < 100.0);
    }

    #[test]
    fn test_mse_to_psnr_zero() {
        let psnr = mse_to_psnr(0.0);
        assert_eq!(psnr, 100.0);
    }

    #[test]
    fn test_psnr_to_mse() {
        let mse = psnr_to_mse(40.0);
        assert!(mse > 0.0);
    }

    #[test]
    fn test_psnr_mse_roundtrip() {
        let original_mse = 50.0;
        let psnr = mse_to_psnr(original_mse);
        let recovered_mse = psnr_to_mse(psnr);
        assert!((original_mse - recovered_mse).abs() < 0.001);
    }

    #[test]
    fn test_quality_metrics_overall_score() {
        let metrics = QualityMetrics {
            psnr: Some(35.0),
            ssim: Some(0.95),
            vmaf: Some(85.0),
            ..Default::default()
        };

        let score = metrics.overall_score();
        assert_eq!(score, Some(85.0)); // VMAF takes precedence
    }

    #[test]
    fn test_quality_metrics_overall_score_ssim_only() {
        let metrics = QualityMetrics {
            psnr: None,
            ssim: Some(0.95),
            vmaf: None,
            ..Default::default()
        };

        let score = metrics.overall_score();
        assert_eq!(score, Some(95.0)); // SSIM * 100
    }

    #[test]
    fn test_quality_metrics_has_any() {
        let metrics = QualityMetrics::empty();
        assert!(!metrics.has_any());

        let metrics = QualityMetrics {
            psnr: Some(35.0),
            ..Default::default()
        };
        assert!(metrics.has_any());
    }

    #[test]
    fn test_metrics_calculator_new() {
        let calc = MetricsCalculator::new(true, true, false);
        assert!(calc.enable_psnr);
        assert!(calc.enable_ssim);
        assert!(!calc.enable_vmaf);
    }

    #[test]
    fn test_calculate_mse_identical() {
        let data = vec![128u8; 64 * 64];
        let mse = calculate_mse(&data, &data, 64, 64, 64);
        assert_eq!(mse, 0.0);
    }

    #[test]
    fn test_calculate_mse_different() {
        let orig = vec![128u8; 64 * 64];
        let enc = vec![130u8; 64 * 64];
        let mse = calculate_mse(&orig, &enc, 64, 64, 64);
        assert_eq!(mse, 4.0); // (128 - 130)^2 = 4
    }

    #[test]
    fn test_ssim_window_identical() {
        let data = vec![128u8; 64 * 64];
        let ssim = calculate_ssim_window(&data, &data, 0, 0, 8, 64, 6.5025, 58.5225);
        assert!((ssim - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_add_metrics() {
        let mut total = QualityMetrics::empty();
        let metrics = QualityMetrics {
            psnr: Some(35.0),
            ssim: Some(0.95),
            ..Default::default()
        };

        add_metrics(&mut total, &metrics);
        assert_eq!(total.psnr, Some(35.0));
        assert_eq!(total.ssim, Some(0.95));

        add_metrics(&mut total, &metrics);
        assert_eq!(total.psnr, Some(70.0));
        assert_eq!(total.ssim, Some(1.90));
    }

    #[test]
    fn test_average_metrics() {
        let total = QualityMetrics {
            psnr: Some(70.0),
            ssim: Some(1.90),
            ..Default::default()
        };

        let avg = average_metrics(&total, 2);
        assert_eq!(avg.psnr, Some(35.0));
        assert_eq!(avg.ssim, Some(0.95));
    }

    // ---- MS-SSIM tests ----

    #[test]
    fn test_ms_ssim_identical() {
        // Identical images must yield MS-SSIM = 1.0.
        let width: u32 = 64;
        let height: u32 = 64;
        let pixels: Vec<f32> = (0..(width * height)).map(|i| (i % 256) as f32).collect();
        let ms_ssim = compute_ms_ssim(&pixels, &pixels, width, height);
        assert!(
            (ms_ssim - 1.0).abs() < 1e-9,
            "MS-SSIM of identical images should be 1.0, got {ms_ssim:.12}"
        );
    }

    #[test]
    fn test_ms_ssim_different() {
        // Very different images should have MS-SSIM < 1.0.
        let width: u32 = 64;
        let height: u32 = 64;
        let all_black: Vec<f32> = vec![0.0; (width * height) as usize];
        let all_white: Vec<f32> = vec![255.0; (width * height) as usize];
        let ms_ssim = compute_ms_ssim(&all_black, &all_white, width, height);
        assert!(
            ms_ssim < 1.0,
            "MS-SSIM of maximally different images should be < 1.0, got {ms_ssim:.6}"
        );
    }

    #[test]
    fn test_ms_ssim_symmetric() {
        let width: u32 = 32;
        let height: u32 = 32;
        let a: Vec<f32> = (0..(width * height)).map(|i| (i % 200) as f32).collect();
        let b: Vec<f32> = (0..(width * height))
            .map(|i| (255 - i % 200) as f32)
            .collect();
        let ms_ab = compute_ms_ssim(&a, &b, width, height);
        let ms_ba = compute_ms_ssim(&b, &a, width, height);
        assert!(
            (ms_ab - ms_ba).abs() < 1e-10,
            "MS-SSIM should be symmetric: {ms_ab:.12} vs {ms_ba:.12}"
        );
    }

    #[test]
    fn test_ms_ssim_mismatched_dimensions_returns_one() {
        // Mismatched slice length → degenerate → return 1.0.
        let pixels: Vec<f32> = vec![128.0; 100];
        let short: Vec<f32> = vec![200.0; 50]; // wrong length
        let ms_ssim = compute_ms_ssim(&pixels, &short, 10, 10);
        assert_eq!(ms_ssim, 1.0);
    }
}
