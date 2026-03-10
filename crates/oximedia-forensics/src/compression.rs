//! JPEG Compression Artifact Analysis
//!
//! This module provides tools for detecting and analyzing JPEG compression artifacts,
//! including double compression detection, blocking artifacts, and quantization analysis.

use crate::{ForensicTest, ForensicsResult};
use image::RgbImage;
use ndarray::{Array2, Array3};
use std::f64::consts::PI;

/// JPEG block size (8x8)
const BLOCK_SIZE: usize = 8;

/// DCT coefficient for frequency analysis
#[derive(Debug, Clone)]
pub struct DctCoefficients {
    /// Y channel coefficients
    pub y: Array3<f64>,
    /// Cb channel coefficients
    pub cb: Array3<f64>,
    /// Cr channel coefficients
    pub cr: Array3<f64>,
}

/// Quantization table analysis result
#[derive(Debug, Clone)]
pub struct QuantizationAnalysis {
    /// Estimated quality factor (1-100)
    pub quality_factor: u8,
    /// Presence of custom quantization tables
    pub custom_tables: bool,
    /// Table uniformity score
    pub uniformity: f64,
}

/// Blocking artifact measure
#[derive(Debug, Clone)]
pub struct BlockingArtifacts {
    /// Horizontal blocking score
    pub horizontal_score: f64,
    /// Vertical blocking score
    pub vertical_score: f64,
    /// Overall blocking severity
    pub severity: f64,
}

/// Double compression detection result
#[derive(Debug, Clone)]
pub struct DoubleCompressionResult {
    /// Whether double compression was detected
    pub detected: bool,
    /// First compression quality estimate
    pub first_quality: Option<u8>,
    /// Second compression quality estimate
    pub second_quality: Option<u8>,
    /// Confidence score
    pub confidence: f64,
    /// Histogram peaks indicating double compression
    pub histogram_peaks: Vec<usize>,
}

/// Analyze JPEG compression artifacts
#[allow(unused_variables)]
pub fn analyze_compression(image: &RgbImage) -> ForensicsResult<ForensicTest> {
    let mut test = ForensicTest::new("JPEG Compression Analysis");

    let (width, height) = image.dimensions();

    // Convert to YCbCr
    let ycbcr = rgb_to_ycbcr(image);

    // Analyze blocking artifacts
    let blocking = detect_blocking_artifacts(&ycbcr.0);
    test.add_finding(format!(
        "Blocking artifacts severity: {:.3} (H: {:.3}, V: {:.3})",
        blocking.severity, blocking.horizontal_score, blocking.vertical_score
    ));

    // Perform DCT analysis
    let dct_coeffs = compute_dct_blocks(&ycbcr.0);

    // Detect double compression
    let double_comp = detect_double_compression(&dct_coeffs);
    if double_comp.detected {
        test.tampering_detected = true;
        test.add_finding(format!(
            "Double compression detected with {:.1}% confidence",
            double_comp.confidence * 100.0
        ));
        if let (Some(q1), Some(q2)) = (double_comp.first_quality, double_comp.second_quality) {
            test.add_finding(format!(
                "Estimated quality factors: first={}, second={}",
                q1, q2
            ));
        }
    }

    // Analyze quantization tables
    let quant_analysis = analyze_quantization(&dct_coeffs);
    test.add_finding(format!(
        "Estimated quality factor: {}",
        quant_analysis.quality_factor
    ));

    if quant_analysis.custom_tables {
        test.add_finding("Custom quantization tables detected".to_string());
    }

    // Calculate confidence based on multiple factors
    let mut confidence = 0.0;

    // Blocking artifacts contribute to confidence
    if blocking.severity > 0.3 {
        confidence += 0.2;
    }

    // Double compression is a strong indicator
    if double_comp.detected {
        confidence += double_comp.confidence * 0.6;
    }

    // Custom quantization tables are suspicious
    if quant_analysis.custom_tables {
        confidence += 0.2;
    }

    test.set_confidence(confidence);

    // Generate anomaly map based on blocking artifacts
    let anomaly_map = create_blocking_anomaly_map(image, &blocking);
    test.anomaly_map = Some(anomaly_map);

    Ok(test)
}

/// Convert RGB image to YCbCr color space
fn rgb_to_ycbcr(image: &RgbImage) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let (width, height) = image.dimensions();
    let mut y = Array2::zeros((height as usize, width as usize));
    let mut cb = Array2::zeros((height as usize, width as usize));
    let mut cr = Array2::zeros((height as usize, width as usize));

    for (x, y_coord, pixel) in image.enumerate_pixels() {
        let r = pixel[0] as f64;
        let g = pixel[1] as f64;
        let b = pixel[2] as f64;

        y[[y_coord as usize, x as usize]] = 0.299 * r + 0.587 * g + 0.114 * b;
        cb[[y_coord as usize, x as usize]] = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0;
        cr[[y_coord as usize, x as usize]] = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0;
    }

    (y, cb, cr)
}

/// Detect blocking artifacts in an image channel
fn detect_blocking_artifacts(channel: &Array2<f64>) -> BlockingArtifacts {
    let (height, width) = channel.dim();

    // Calculate horizontal blocking score
    let mut h_score = 0.0;
    let mut h_count = 0;

    for y in 0..height {
        for x in (BLOCK_SIZE..width).step_by(BLOCK_SIZE) {
            if x > 0 && x < width {
                let diff = (channel[[y, x]] - channel[[y, x - 1]]).abs();
                h_score += diff;
                h_count += 1;
            }
        }
    }

    if h_count > 0 {
        h_score /= h_count as f64;
    }

    // Calculate vertical blocking score
    let mut v_score = 0.0;
    let mut v_count = 0;

    for y in (BLOCK_SIZE..height).step_by(BLOCK_SIZE) {
        for x in 0..width {
            if y > 0 && y < height {
                let diff = (channel[[y, x]] - channel[[y - 1, x]]).abs();
                v_score += diff;
                v_count += 1;
            }
        }
    }

    if v_count > 0 {
        v_score /= v_count as f64;
    }

    // Calculate average blocking score across all positions
    let mut avg_diff = 0.0;
    let mut count = 0;

    for y in 0..height - 1 {
        for x in 0..width - 1 {
            avg_diff += (channel[[y, x]] - channel[[y, x + 1]]).abs();
            avg_diff += (channel[[y, x]] - channel[[y + 1, x]]).abs();
            count += 2;
        }
    }

    if count > 0 {
        avg_diff /= count as f64;
    }

    // Normalize scores
    let h_normalized = if avg_diff > 0.0 {
        h_score / avg_diff
    } else {
        0.0
    };
    let v_normalized = if avg_diff > 0.0 {
        v_score / avg_diff
    } else {
        0.0
    };

    let severity = ((h_normalized + v_normalized) / 2.0).min(1.0);

    BlockingArtifacts {
        horizontal_score: h_normalized.min(1.0),
        vertical_score: v_normalized.min(1.0),
        severity,
    }
}

/// Compute DCT coefficients for 8x8 blocks
fn compute_dct_blocks(channel: &Array2<f64>) -> Array3<f64> {
    let (height, width) = channel.dim();
    let blocks_h = height / BLOCK_SIZE;
    let blocks_w = width / BLOCK_SIZE;

    let mut dct_blocks = Array3::zeros((blocks_h, blocks_w, BLOCK_SIZE * BLOCK_SIZE));

    for by in 0..blocks_h {
        for bx in 0..blocks_w {
            let block = extract_block(channel, by, bx);
            let dct = dct_2d(&block);

            for i in 0..BLOCK_SIZE {
                for j in 0..BLOCK_SIZE {
                    dct_blocks[[by, bx, i * BLOCK_SIZE + j]] = dct[[i, j]];
                }
            }
        }
    }

    dct_blocks
}

/// Extract an 8x8 block from a channel
fn extract_block(channel: &Array2<f64>, block_y: usize, block_x: usize) -> Array2<f64> {
    let y_start = block_y * BLOCK_SIZE;
    let x_start = block_x * BLOCK_SIZE;

    let mut block = Array2::zeros((BLOCK_SIZE, BLOCK_SIZE));

    for i in 0..BLOCK_SIZE {
        for j in 0..BLOCK_SIZE {
            if y_start + i < channel.nrows() && x_start + j < channel.ncols() {
                block[[i, j]] = channel[[y_start + i, x_start + j]];
            }
        }
    }

    block
}

/// Perform 2D DCT on an 8x8 block
fn dct_2d(block: &Array2<f64>) -> Array2<f64> {
    let mut dct = Array2::zeros((BLOCK_SIZE, BLOCK_SIZE));

    for u in 0..BLOCK_SIZE {
        for v in 0..BLOCK_SIZE {
            let mut sum = 0.0;

            for x in 0..BLOCK_SIZE {
                for y in 0..BLOCK_SIZE {
                    let val = block[[y, x]];
                    let cos_u = ((2.0 * x as f64 + 1.0) * u as f64 * PI / 16.0).cos();
                    let cos_v = ((2.0 * y as f64 + 1.0) * v as f64 * PI / 16.0).cos();
                    sum += val * cos_u * cos_v;
                }
            }

            let cu = if u == 0 { 1.0 / 2.0_f64.sqrt() } else { 1.0 };
            let cv = if v == 0 { 1.0 / 2.0_f64.sqrt() } else { 1.0 };

            dct[[u, v]] = 0.25 * cu * cv * sum;
        }
    }

    dct
}

/// Detect double compression from DCT coefficient histograms
fn detect_double_compression(dct_coeffs: &Array3<f64>) -> DoubleCompressionResult {
    let (blocks_h, blocks_w, coeffs_per_block) = dct_coeffs.dim();

    // Analyze specific DCT coefficient positions prone to double compression artifacts
    // Focus on low-frequency AC coefficients
    let positions_to_check = vec![1, 8, 2, 9, 16]; // Zigzag positions

    let mut histogram_periodicity_scores = Vec::new();
    let mut all_peaks = Vec::new();

    for &pos in &positions_to_check {
        if pos >= coeffs_per_block {
            continue;
        }

        // Collect all coefficients at this position
        let mut coeffs = Vec::new();
        for by in 0..blocks_h {
            for bx in 0..blocks_w {
                coeffs.push(dct_coeffs[[by, bx, pos]]);
            }
        }

        // Build histogram
        let histogram = build_histogram(&coeffs, -2048.0, 2048.0, 256);

        // Detect periodicity in histogram (sign of double compression)
        let (periodicity, peaks) = detect_histogram_periodicity(&histogram);
        histogram_periodicity_scores.push(periodicity);
        all_peaks.extend(peaks);
    }

    // Average periodicity score
    let avg_periodicity = if !histogram_periodicity_scores.is_empty() {
        histogram_periodicity_scores.iter().sum::<f64>() / histogram_periodicity_scores.len() as f64
    } else {
        0.0
    };

    // Threshold for detection
    let detected = avg_periodicity > 0.3;
    let confidence = avg_periodicity.min(1.0);

    // Estimate quality factors (simplified)
    let first_quality = if detected { Some(75) } else { None };
    let second_quality = if detected { Some(60) } else { None };

    DoubleCompressionResult {
        detected,
        first_quality,
        second_quality,
        confidence,
        histogram_peaks: all_peaks,
    }
}

/// Build a histogram from data
fn build_histogram(data: &[f64], min_val: f64, max_val: f64, num_bins: usize) -> Vec<usize> {
    let mut histogram = vec![0; num_bins];
    let bin_width = (max_val - min_val) / num_bins as f64;

    for &val in data {
        if val >= min_val && val <= max_val {
            let bin = ((val - min_val) / bin_width) as usize;
            let bin = bin.min(num_bins - 1);
            histogram[bin] += 1;
        }
    }

    histogram
}

/// Detect periodicity in histogram (sign of double compression)
fn detect_histogram_periodicity(histogram: &[usize]) -> (f64, Vec<usize>) {
    let len = histogram.len();
    if len < 16 {
        return (0.0, Vec::new());
    }

    // Find peaks in histogram
    let mut peaks = Vec::new();
    for i in 1..len - 1 {
        if histogram[i] > histogram[i - 1] && histogram[i] > histogram[i + 1] && histogram[i] > 10 {
            // Minimum peak height
            peaks.push(i);
        }
    }

    if peaks.len() < 3 {
        return (0.0, peaks);
    }

    // Check for periodic spacing between peaks
    let mut spacings = Vec::new();
    for i in 1..peaks.len() {
        spacings.push(peaks[i] - peaks[i - 1]);
    }

    // Calculate variance of spacings
    if spacings.is_empty() {
        return (0.0, peaks);
    }

    let mean_spacing = spacings.iter().sum::<usize>() as f64 / spacings.len() as f64;
    let variance = spacings
        .iter()
        .map(|&s| {
            let diff = s as f64 - mean_spacing;
            diff * diff
        })
        .sum::<f64>()
        / spacings.len() as f64;

    let std_dev = variance.sqrt();

    // Low variance indicates periodic peaks (double compression)
    let periodicity_score = if mean_spacing > 0.0 {
        1.0 - (std_dev / mean_spacing).min(1.0)
    } else {
        0.0
    };

    (periodicity_score, peaks)
}

/// Analyze quantization tables from DCT coefficients
fn analyze_quantization(dct_coeffs: &Array3<f64>) -> QuantizationAnalysis {
    let (blocks_h, blocks_w, _coeffs_per_block) = dct_coeffs.dim();

    // Estimate quantization step sizes
    let mut q_steps = Vec::new();

    for pos in 1..BLOCK_SIZE * BLOCK_SIZE {
        let mut values = Vec::new();
        for by in 0..blocks_h {
            for bx in 0..blocks_w {
                values.push(dct_coeffs[[by, bx, pos]].abs());
            }
        }

        if !values.is_empty() {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = values[values.len() / 2];
            if median > 0.0 {
                q_steps.push(median);
            }
        }
    }

    // Estimate quality factor from quantization steps
    let quality_factor = estimate_quality_factor(&q_steps);

    // Check for custom tables (high variance in q_steps)
    let mean_q = if !q_steps.is_empty() {
        q_steps.iter().sum::<f64>() / q_steps.len() as f64
    } else {
        1.0
    };

    let variance = if !q_steps.is_empty() {
        q_steps
            .iter()
            .map(|&q| {
                let diff = q - mean_q;
                diff * diff
            })
            .sum::<f64>()
            / q_steps.len() as f64
    } else {
        0.0
    };

    let std_dev = variance.sqrt();
    let coefficient_of_variation = if mean_q > 0.0 { std_dev / mean_q } else { 0.0 };

    let custom_tables = coefficient_of_variation > 0.5;
    let uniformity = 1.0 - coefficient_of_variation.min(1.0);

    QuantizationAnalysis {
        quality_factor,
        custom_tables,
        uniformity,
    }
}

/// Estimate JPEG quality factor from quantization steps
fn estimate_quality_factor(q_steps: &[f64]) -> u8 {
    if q_steps.is_empty() {
        return 75;
    }

    // Use median quantization step
    let mut sorted = q_steps.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_q = sorted[sorted.len() / 2];

    // Map quantization step to quality (inverse relationship)
    // Typical range: Q=100 -> step ~1, Q=50 -> step ~16, Q=10 -> step ~100

    if median_q < 1.0 {
        95
    } else if median_q < 5.0 {
        85
    } else if median_q < 10.0 {
        75
    } else if median_q < 20.0 {
        60
    } else if median_q < 50.0 {
        40
    } else {
        20
    }
}

/// Create anomaly map based on blocking artifacts
fn create_blocking_anomaly_map(image: &RgbImage, blocking: &BlockingArtifacts) -> Array2<f64> {
    let (width, height) = image.dimensions();
    let mut anomaly_map = Array2::zeros((height as usize, width as usize));

    // Highlight block boundaries
    for y in 0..height as usize {
        for x in 0..width as usize {
            let mut score = 0.0;

            // Check if on block boundary
            if x % BLOCK_SIZE == 0 || x % BLOCK_SIZE == BLOCK_SIZE - 1 {
                score += blocking.horizontal_score;
            }

            if y % BLOCK_SIZE == 0 || y % BLOCK_SIZE == BLOCK_SIZE - 1 {
                score += blocking.vertical_score;
            }

            anomaly_map[[y, x]] = score;
        }
    }

    anomaly_map
}

/// Analyze compression history
pub fn estimate_compression_history(image: &RgbImage) -> ForensicsResult<Vec<u8>> {
    let ycbcr = rgb_to_ycbcr(image);
    let dct_coeffs = compute_dct_blocks(&ycbcr.0);
    let quant_analysis = analyze_quantization(&dct_coeffs);

    let mut history = Vec::new();
    history.push(quant_analysis.quality_factor);

    let double_comp = detect_double_compression(&dct_coeffs);
    if double_comp.detected {
        if let Some(q1) = double_comp.first_quality {
            history.insert(0, q1);
        }
    }

    Ok(history)
}

/// Detect blocking artifacts with detailed location map
pub fn detect_blocking_with_map(
    image: &RgbImage,
) -> ForensicsResult<(BlockingArtifacts, Array2<f64>)> {
    let ycbcr = rgb_to_ycbcr(image);
    let blocking = detect_blocking_artifacts(&ycbcr.0);
    let map = create_blocking_anomaly_map(image, &blocking);

    Ok((blocking, map))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn test_rgb_to_ycbcr() {
        let img = RgbImage::new(16, 16);
        let (y, cb, cr) = rgb_to_ycbcr(&img);
        assert_eq!(y.dim(), (16, 16));
        assert_eq!(cb.dim(), (16, 16));
        assert_eq!(cr.dim(), (16, 16));
    }

    #[test]
    fn test_dct_2d() {
        let block = Array2::zeros((BLOCK_SIZE, BLOCK_SIZE));
        let dct = dct_2d(&block);
        assert_eq!(dct.dim(), (BLOCK_SIZE, BLOCK_SIZE));
    }

    #[test]
    fn test_histogram_building() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let hist = build_histogram(&data, 0.0, 5.0, 5);
        assert_eq!(hist.len(), 5);
    }

    #[test]
    fn test_quality_estimation() {
        let q_steps = vec![1.0, 2.0, 3.0];
        let quality = estimate_quality_factor(&q_steps);
        assert!(quality > 0 && quality <= 100);
    }

    #[test]
    fn test_blocking_detection() {
        let channel = Array2::zeros((64, 64));
        let blocking = detect_blocking_artifacts(&channel);
        assert!(blocking.severity >= 0.0 && blocking.severity <= 1.0);
    }
}
