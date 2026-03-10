#![allow(dead_code)]
//! Advanced noise pattern analysis for forensic image examination.
//!
//! This module provides tools for analyzing sensor noise patterns,
//! including Photo Response Non-Uniformity (PRNU) extraction,
//! noise level estimation, and spatial noise consistency checking.
//! Inconsistent noise patterns are strong indicators of tampering.

use std::collections::HashMap;

/// Type of noise pattern being analyzed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NoiseType {
    /// Gaussian (read) noise from the sensor.
    Gaussian,
    /// Shot (Poisson) noise from photon counting.
    Shot,
    /// Fixed pattern noise (dark current).
    FixedPattern,
    /// Photo Response Non-Uniformity.
    Prnu,
    /// Quantization noise from A/D conversion.
    Quantization,
}

impl NoiseType {
    /// Return a human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Gaussian => "Gaussian",
            Self::Shot => "Shot",
            Self::FixedPattern => "Fixed Pattern",
            Self::Prnu => "PRNU",
            Self::Quantization => "Quantization",
        }
    }
}

/// Noise level estimate for a region of an image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseLevel {
    /// Estimated noise standard deviation.
    pub sigma: f64,
    /// Mean intensity of the region.
    pub mean_intensity: f64,
    /// Signal-to-noise ratio in dB.
    pub snr_db: f64,
}

impl NoiseLevel {
    /// Create a new noise level estimate.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn new(sigma: f64, mean_intensity: f64) -> Self {
        let snr_db = if sigma > 1e-10 {
            20.0 * (mean_intensity / sigma).log10()
        } else {
            f64::INFINITY
        };
        Self {
            sigma,
            mean_intensity,
            snr_db,
        }
    }

    /// Check if the noise level is within a plausible range for natural images.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        self.sigma > 0.1 && self.sigma < 100.0
    }
}

/// Result of local noise analysis on a grid of blocks.
#[derive(Debug, Clone)]
pub struct NoiseMap {
    /// Width of the grid (number of blocks horizontally).
    pub grid_width: usize,
    /// Height of the grid (number of blocks vertically).
    pub grid_height: usize,
    /// Noise levels per block (row-major order).
    pub blocks: Vec<NoiseLevel>,
}

impl NoiseMap {
    /// Create a new noise map with the given grid dimensions.
    #[must_use]
    pub fn new(grid_width: usize, grid_height: usize) -> Self {
        Self {
            grid_width,
            grid_height,
            blocks: Vec::with_capacity(grid_width * grid_height),
        }
    }

    /// Add a block noise level to the map.
    pub fn push(&mut self, level: NoiseLevel) {
        self.blocks.push(level);
    }

    /// Get the noise level at a specific grid position.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> Option<&NoiseLevel> {
        if col < self.grid_width && row < self.grid_height {
            self.blocks.get(row * self.grid_width + col)
        } else {
            None
        }
    }

    /// Compute the mean noise sigma across all blocks.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn mean_sigma(&self) -> f64 {
        if self.blocks.is_empty() {
            return 0.0;
        }
        let total: f64 = self.blocks.iter().map(|b| b.sigma).sum();
        total / self.blocks.len() as f64
    }

    /// Compute the standard deviation of noise sigmas.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn sigma_std_dev(&self) -> f64 {
        if self.blocks.is_empty() {
            return 0.0;
        }
        let mean = self.mean_sigma();
        let variance: f64 = self
            .blocks
            .iter()
            .map(|b| (b.sigma - mean).powi(2))
            .sum::<f64>()
            / self.blocks.len() as f64;
        variance.sqrt()
    }

    /// Detect outlier blocks where noise is significantly different from the median.
    ///
    /// Uses the median sigma as a robust centre estimate so that extreme
    /// outliers do not inflate the reference and mask themselves.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn detect_outliers(&self, threshold_sigmas: f64) -> Vec<(usize, usize)> {
        if self.blocks.is_empty() {
            return Vec::new();
        }
        let mut sorted_sigmas: Vec<f64> = self.blocks.iter().map(|b| b.sigma).collect();
        sorted_sigmas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted_sigmas[sorted_sigmas.len() / 2];

        // Compute MAD (Median Absolute Deviation)
        let mut abs_devs: Vec<f64> = sorted_sigmas.iter().map(|&s| (s - median).abs()).collect();
        abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad = abs_devs[abs_devs.len() / 2];

        // Convert MAD to standard deviation estimate (Gaussian assumption)
        let sigma_est = mad * 1.4826;
        if sigma_est < 1e-10 {
            // MAD is zero: the majority of values are identical.
            // Any value that differs from the median at all is an outlier.
            let mut outliers = Vec::new();
            for (idx, block) in self.blocks.iter().enumerate() {
                if (block.sigma - median).abs() > 1e-10 {
                    let row = idx / self.grid_width;
                    let col = idx % self.grid_width;
                    outliers.push((row, col));
                }
            }
            return outliers;
        }

        let mut outliers = Vec::new();
        for (idx, block) in self.blocks.iter().enumerate() {
            let z_score = (block.sigma - median).abs() / sigma_est;
            if z_score > threshold_sigmas {
                let row = idx / self.grid_width;
                let col = idx % self.grid_width;
                outliers.push((row, col));
            }
        }
        outliers
    }
}

/// Estimate noise level from pixel data using the Median Absolute Deviation.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn estimate_noise_mad(pixels: &[f64]) -> f64 {
    if pixels.len() < 2 {
        return 0.0;
    }

    // Compute high-pass filtered values (differences)
    let diffs: Vec<f64> = pixels.windows(2).map(|w| (w[1] - w[0]).abs()).collect();

    // Compute median of absolute differences
    let mut sorted = diffs.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    // MAD to sigma conversion (assuming Gaussian noise)
    median * 1.4826
}

/// Estimate noise level using the standard deviation method.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn estimate_noise_std(pixels: &[f64]) -> f64 {
    if pixels.len() < 2 {
        return 0.0;
    }
    let n = pixels.len() as f64;
    let mean = pixels.iter().sum::<f64>() / n;
    let variance = pixels.iter().map(|&p| (p - mean).powi(2)).sum::<f64>() / (n - 1.0);
    variance.sqrt()
}

/// Analyze noise consistency across blocks of an image.
///
/// Returns a noise map and an inconsistency score (0.0 = consistent, 1.0 = very inconsistent).
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn analyze_noise_consistency(pixel_rows: &[Vec<f64>], block_size: usize) -> (NoiseMap, f64) {
    if pixel_rows.is_empty() || block_size == 0 {
        return (NoiseMap::new(0, 0), 0.0);
    }

    let height = pixel_rows.len();
    let width = pixel_rows[0].len();
    let grid_h = height / block_size;
    let grid_w = width / block_size;

    let mut map = NoiseMap::new(grid_w, grid_h);

    for br in 0..grid_h {
        for bc in 0..grid_w {
            let mut block_pixels = Vec::new();
            for r in 0..block_size {
                let row_idx = br * block_size + r;
                if row_idx < pixel_rows.len() {
                    for c in 0..block_size {
                        let col_idx = bc * block_size + c;
                        if col_idx < pixel_rows[row_idx].len() {
                            block_pixels.push(pixel_rows[row_idx][col_idx]);
                        }
                    }
                }
            }

            let sigma = estimate_noise_mad(&block_pixels);
            let mean_intensity = if block_pixels.is_empty() {
                0.0
            } else {
                block_pixels.iter().sum::<f64>() / block_pixels.len() as f64
            };
            map.push(NoiseLevel::new(sigma, mean_intensity));
        }
    }

    // Inconsistency score based on coefficient of variation of sigmas
    let mean_sig = map.mean_sigma();
    let inconsistency = if mean_sig > 1e-10 {
        (map.sigma_std_dev() / mean_sig).min(1.0)
    } else {
        0.0
    };

    (map, inconsistency)
}

/// PRNU fingerprint for camera identification.
#[derive(Debug, Clone)]
pub struct PrnuFingerprint {
    /// Width of the fingerprint in pixels.
    pub width: usize,
    /// Height of the fingerprint in pixels.
    pub height: usize,
    /// PRNU pattern values (row-major).
    pub pattern: Vec<f64>,
    /// Camera identifier (if known).
    pub camera_id: Option<String>,
}

impl PrnuFingerprint {
    /// Create a new PRNU fingerprint.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pattern: vec![0.0; width * height],
            camera_id: None,
        }
    }

    /// Set the camera identifier.
    pub fn set_camera_id(&mut self, id: &str) {
        self.camera_id = Some(id.to_string());
    }

    /// Compute the normalized cross-correlation with another fingerprint.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn correlate(&self, other: &Self) -> f64 {
        if self.pattern.len() != other.pattern.len() || self.pattern.is_empty() {
            return 0.0;
        }

        let n = self.pattern.len() as f64;
        let mean_a: f64 = self.pattern.iter().sum::<f64>() / n;
        let mean_b: f64 = other.pattern.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var_a = 0.0;
        let mut var_b = 0.0;

        for (a, b) in self.pattern.iter().zip(other.pattern.iter()) {
            let da = a - mean_a;
            let db = b - mean_b;
            cov += da * db;
            var_a += da * da;
            var_b += db * db;
        }

        let denom = (var_a * var_b).sqrt();
        if denom < 1e-15 {
            0.0
        } else {
            cov / denom
        }
    }

    /// Get the number of pixels in the fingerprint.
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }
}

/// Summary of noise analysis across different noise types.
#[derive(Debug, Clone)]
pub struct NoiseAnalysisSummary {
    /// Per-type noise level estimates.
    pub levels: HashMap<NoiseType, f64>,
    /// Overall noise inconsistency score (0.0..1.0).
    pub inconsistency_score: f64,
    /// Whether tampering is suspected based on noise analysis.
    pub tampering_suspected: bool,
    /// Textual findings.
    pub findings: Vec<String>,
}

impl NoiseAnalysisSummary {
    /// Create a new empty summary.
    #[must_use]
    pub fn new() -> Self {
        Self {
            levels: HashMap::new(),
            inconsistency_score: 0.0,
            tampering_suspected: false,
            findings: Vec::new(),
        }
    }

    /// Add a noise level measurement for a specific type.
    pub fn add_level(&mut self, noise_type: NoiseType, sigma: f64) {
        self.levels.insert(noise_type, sigma);
    }

    /// Add a textual finding.
    pub fn add_finding(&mut self, finding: &str) {
        self.findings.push(finding.to_string());
    }

    /// Evaluate whether noise patterns suggest tampering.
    pub fn evaluate(&mut self, threshold: f64) {
        self.tampering_suspected = self.inconsistency_score > threshold;
        if self.tampering_suspected {
            self.add_finding(&format!(
                "Noise inconsistency ({:.3}) exceeds threshold ({:.3})",
                self.inconsistency_score, threshold
            ));
        }
    }
}

impl Default for NoiseAnalysisSummary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_type_labels() {
        assert_eq!(NoiseType::Gaussian.label(), "Gaussian");
        assert_eq!(NoiseType::Prnu.label(), "PRNU");
        assert_eq!(NoiseType::Quantization.label(), "Quantization");
    }

    #[test]
    fn test_noise_level_snr() {
        let nl = NoiseLevel::new(1.0, 100.0);
        assert!((nl.snr_db - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_noise_level_zero_sigma() {
        let nl = NoiseLevel::new(0.0, 100.0);
        assert!(nl.snr_db.is_infinite());
    }

    #[test]
    fn test_noise_level_plausible() {
        assert!(NoiseLevel::new(5.0, 128.0).is_plausible());
        assert!(!NoiseLevel::new(0.001, 128.0).is_plausible());
        assert!(!NoiseLevel::new(200.0, 128.0).is_plausible());
    }

    #[test]
    fn test_noise_map_get() {
        let mut map = NoiseMap::new(2, 2);
        map.push(NoiseLevel::new(1.0, 50.0));
        map.push(NoiseLevel::new(2.0, 60.0));
        map.push(NoiseLevel::new(3.0, 70.0));
        map.push(NoiseLevel::new(4.0, 80.0));

        assert!((map.get(0, 0).expect("get should succeed").sigma - 1.0).abs() < f64::EPSILON);
        assert!((map.get(1, 1).expect("get should succeed").sigma - 4.0).abs() < f64::EPSILON);
        assert!(map.get(2, 0).is_none());
    }

    #[test]
    fn test_noise_map_mean_sigma() {
        let mut map = NoiseMap::new(2, 1);
        map.push(NoiseLevel::new(2.0, 50.0));
        map.push(NoiseLevel::new(4.0, 60.0));
        assert!((map.mean_sigma() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_noise_map_empty() {
        let map = NoiseMap::new(0, 0);
        assert!((map.mean_sigma()).abs() < f64::EPSILON);
        assert!((map.sigma_std_dev()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_noise_map_outliers() {
        let mut map = NoiseMap::new(4, 1);
        map.push(NoiseLevel::new(5.0, 100.0));
        map.push(NoiseLevel::new(5.0, 100.0));
        map.push(NoiseLevel::new(5.0, 100.0));
        map.push(NoiseLevel::new(50.0, 100.0)); // outlier
        let outliers = map.detect_outliers(2.0);
        assert!(!outliers.is_empty());
        assert_eq!(outliers[0], (0, 3));
    }

    #[test]
    fn test_estimate_noise_mad() {
        // Constant signal = 0 noise
        let constant = vec![5.0; 100];
        assert!((estimate_noise_mad(&constant)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_estimate_noise_mad_empty() {
        assert!((estimate_noise_mad(&[])).abs() < f64::EPSILON);
    }

    #[test]
    fn test_estimate_noise_std() {
        let constant = vec![5.0; 100];
        assert!((estimate_noise_std(&constant)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_analyze_noise_consistency_uniform() {
        let rows: Vec<Vec<f64>> = (0..8).map(|_| vec![128.0; 8]).collect();
        let (map, inconsistency) = analyze_noise_consistency(&rows, 4);
        assert_eq!(map.grid_width, 2);
        assert_eq!(map.grid_height, 2);
        // Uniform image should have low inconsistency
        assert!(inconsistency < 0.01);
    }

    #[test]
    fn test_analyze_noise_consistency_empty() {
        let (map, score) = analyze_noise_consistency(&[], 4);
        assert_eq!(map.grid_width, 0);
        assert!((score).abs() < f64::EPSILON);
    }

    #[test]
    fn test_prnu_fingerprint_self_correlation() {
        let mut fp = PrnuFingerprint::new(4, 4);
        for (i, v) in fp.pattern.iter_mut().enumerate() {
            *v = (i as f64) * 0.1;
        }
        let corr = fp.correlate(&fp);
        assert!((corr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_prnu_fingerprint_zero_correlation() {
        let fp1 = PrnuFingerprint::new(4, 4);
        let fp2 = PrnuFingerprint::new(4, 4);
        // Both all zeros => correlation undefined, returns 0
        assert!((fp1.correlate(&fp2)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_prnu_fingerprint_size_mismatch() {
        let fp1 = PrnuFingerprint::new(4, 4);
        let fp2 = PrnuFingerprint::new(3, 3);
        assert!((fp1.correlate(&fp2)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_prnu_camera_id() {
        let mut fp = PrnuFingerprint::new(2, 2);
        assert!(fp.camera_id.is_none());
        fp.set_camera_id("Canon-5D-001");
        assert_eq!(fp.camera_id.as_deref(), Some("Canon-5D-001"));
    }

    #[test]
    fn test_noise_summary_evaluate() {
        let mut summary = NoiseAnalysisSummary::new();
        summary.inconsistency_score = 0.6;
        summary.evaluate(0.5);
        assert!(summary.tampering_suspected);
        assert!(!summary.findings.is_empty());
    }

    #[test]
    fn test_noise_summary_evaluate_below_threshold() {
        let mut summary = NoiseAnalysisSummary::new();
        summary.inconsistency_score = 0.2;
        summary.evaluate(0.5);
        assert!(!summary.tampering_suspected);
    }
}
