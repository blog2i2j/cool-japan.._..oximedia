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
//! # fn example(original: &VideoFrame, encoded: &VideoFrame) {
//! let calculator = MetricsCalculator::new(true, true, false);
//! let metrics = calculator.calculate(original, encoded)?;
//!
//! println!("PSNR: {:?} dB", metrics.psnr);
//! println!("SSIM: {:?}", metrics.ssim);
//! # }
//! ```

use crate::{BenchError, BenchResult};
use oximedia_codec::VideoFrame;
use serde::{Deserialize, Serialize};

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

/// Calculate VMAF (Video Multimethod Assessment Fusion) score.
///
/// This is a placeholder implementation. In production, you would integrate
/// with the actual VMAF library (libvmaf).
fn calculate_vmaf(
    _original: &VideoFrame,
    _encoded: &VideoFrame,
    _model_path: Option<&str>,
) -> BenchResult<f64> {
    // Placeholder: In a real implementation, this would call libvmaf
    // For now, we'll return an estimated VMAF based on SSIM and PSNR

    #[cfg(feature = "vmaf")]
    {
        // Real VMAF implementation would go here
        Err(BenchError::MetricFailed(
            "VMAF support not yet implemented".to_string(),
        ))
    }

    #[cfg(not(feature = "vmaf"))]
    {
        Err(BenchError::MetricFailed(
            "VMAF feature not enabled".to_string(),
        ))
    }
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
}
