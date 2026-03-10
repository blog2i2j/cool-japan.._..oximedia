//! Color matching and calibration for multi-camera alignment.
//!
//! This module provides tools for matching colors across different cameras:
//!
//! - Color transfer algorithms
//! - Histogram matching
//! - White balance estimation
//! - `ColorChecker` calibration
//! - Color space conversions

use crate::{AlignError, AlignResult};
// PI removed - unused

/// RGB color (0.0 - 1.0 range)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorRgb {
    /// Red channel
    pub r: f32,
    /// Green channel
    pub g: f32,
    /// Blue channel
    pub b: f32,
}

impl ColorRgb {
    /// Create new RGB color
    #[must_use]
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    /// Create from u8 values
    #[must_use]
    pub fn from_u8(r: u8, g: u8, b: u8) -> Self {
        Self {
            r: f32::from(r) / 255.0,
            g: f32::from(g) / 255.0,
            b: f32::from(b) / 255.0,
        }
    }

    /// Convert to u8 values
    #[must_use]
    pub fn to_u8(&self) -> (u8, u8, u8) {
        (
            (self.r * 255.0).clamp(0.0, 255.0) as u8,
            (self.g * 255.0).clamp(0.0, 255.0) as u8,
            (self.b * 255.0).clamp(0.0, 255.0) as u8,
        )
    }

    /// Convert to LAB color space
    #[must_use]
    pub fn to_lab(&self) -> ColorLab {
        // RGB -> XYZ (D65 illuminant)
        let r = Self::gamma_to_linear(self.r);
        let g = Self::gamma_to_linear(self.g);
        let b = Self::gamma_to_linear(self.b);

        let x = r * 0.4124 + g * 0.3576 + b * 0.1805;
        let y = r * 0.2126 + g * 0.7152 + b * 0.0722;
        let z = r * 0.0193 + g * 0.1192 + b * 0.9505;

        // XYZ -> LAB
        let xn = 0.95047; // D65 white point
        let yn = 1.0;
        let zn = 1.08883;

        let fx = Self::lab_f(x / xn);
        let fy = Self::lab_f(y / yn);
        let fz = Self::lab_f(z / zn);

        let l = 116.0 * fy - 16.0;
        let a = 500.0 * (fx - fy);
        let b_lab = 200.0 * (fy - fz);

        ColorLab::new(l, a, b_lab)
    }

    fn gamma_to_linear(v: f32) -> f32 {
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    }

    fn lab_f(t: f32) -> f32 {
        let delta = 6.0 / 29.0;
        if t > delta * delta * delta {
            t.powf(1.0 / 3.0)
        } else {
            t / (3.0 * delta * delta) + 4.0 / 29.0
        }
    }
}

/// LAB color space
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorLab {
    /// L (lightness)
    pub l: f32,
    /// A (green-red)
    pub a: f32,
    /// B (blue-yellow)
    pub b: f32,
}

impl ColorLab {
    /// Create new LAB color
    #[must_use]
    pub fn new(l: f32, a: f32, b: f32) -> Self {
        Self { l, a, b }
    }

    /// Convert to RGB
    #[must_use]
    pub fn to_rgb(&self) -> ColorRgb {
        // LAB -> XYZ
        let fy = (self.l + 16.0) / 116.0;
        let fx = self.a / 500.0 + fy;
        let fz = fy - self.b / 200.0;

        let xn = 0.95047;
        let yn = 1.0;
        let zn = 1.08883;

        let x = xn * Self::lab_f_inv(fx);
        let y = yn * Self::lab_f_inv(fy);
        let z = zn * Self::lab_f_inv(fz);

        // XYZ -> RGB
        let r = x * 3.2406 + y * -1.5372 + z * -0.4986;
        let g = x * -0.9689 + y * 1.8758 + z * 0.0415;
        let b = x * 0.0557 + y * -0.2040 + z * 1.0570;

        ColorRgb::new(
            Self::linear_to_gamma(r),
            Self::linear_to_gamma(g),
            Self::linear_to_gamma(b),
        )
    }

    fn lab_f_inv(t: f32) -> f32 {
        let delta = 6.0 / 29.0;
        if t > delta {
            t * t * t
        } else {
            3.0 * delta * delta * (t - 4.0 / 29.0)
        }
    }

    fn linear_to_gamma(v: f32) -> f32 {
        if v <= 0.0031308 {
            12.92 * v
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }
    }
}

/// Color statistics for an image
#[derive(Debug, Clone)]
pub struct ColorStatistics {
    /// Mean RGB values
    pub mean: ColorRgb,
    /// Standard deviation RGB values
    pub std_dev: ColorRgb,
    /// Mean LAB values
    pub mean_lab: ColorLab,
    /// Standard deviation LAB values
    pub std_dev_lab: ColorLab,
}

impl ColorStatistics {
    /// Compute statistics from RGB image
    #[must_use]
    pub fn from_image(rgb: &[u8], width: usize, height: usize) -> Self {
        let n = (width * height) as f32;
        let mut sum_r = 0.0f32;
        let mut sum_g = 0.0f32;
        let mut sum_b = 0.0f32;

        // Compute means
        for pixel in rgb.chunks_exact(3) {
            sum_r += f32::from(pixel[0]);
            sum_g += f32::from(pixel[1]);
            sum_b += f32::from(pixel[2]);
        }

        let mean = ColorRgb::new(
            sum_r / (n * 255.0),
            sum_g / (n * 255.0),
            sum_b / (n * 255.0),
        );

        // Compute standard deviations
        let mut var_r = 0.0f32;
        let mut var_g = 0.0f32;
        let mut var_b = 0.0f32;

        for pixel in rgb.chunks_exact(3) {
            let r = f32::from(pixel[0]) / 255.0 - mean.r;
            let g = f32::from(pixel[1]) / 255.0 - mean.g;
            let b = f32::from(pixel[2]) / 255.0 - mean.b;

            var_r += r * r;
            var_g += g * g;
            var_b += b * b;
        }

        let std_dev = ColorRgb::new((var_r / n).sqrt(), (var_g / n).sqrt(), (var_b / n).sqrt());

        // Convert to LAB
        let _mean_lab = mean.to_lab();

        // Compute LAB statistics
        let mut sum_l = 0.0f32;
        let mut sum_a = 0.0f32;
        let mut sum_b_lab = 0.0f32;

        for pixel in rgb.chunks_exact(3) {
            let color = ColorRgb::from_u8(pixel[0], pixel[1], pixel[2]);
            let lab = color.to_lab();
            sum_l += lab.l;
            sum_a += lab.a;
            sum_b_lab += lab.b;
        }

        let mean_lab_actual = ColorLab::new(sum_l / n, sum_a / n, sum_b_lab / n);

        let mut var_l = 0.0f32;
        let mut var_a = 0.0f32;
        let mut var_b_lab = 0.0f32;

        for pixel in rgb.chunks_exact(3) {
            let color = ColorRgb::from_u8(pixel[0], pixel[1], pixel[2]);
            let lab = color.to_lab();
            let dl = lab.l - mean_lab_actual.l;
            let da = lab.a - mean_lab_actual.a;
            let db = lab.b - mean_lab_actual.b;

            var_l += dl * dl;
            var_a += da * da;
            var_b_lab += db * db;
        }

        let std_dev_lab = ColorLab::new(
            (var_l / n).sqrt(),
            (var_a / n).sqrt(),
            (var_b_lab / n).sqrt(),
        );

        Self {
            mean,
            std_dev,
            mean_lab: mean_lab_actual,
            std_dev_lab,
        }
    }
}

/// Color transfer using LAB statistics
pub struct ColorTransfer;

impl ColorTransfer {
    /// Transfer color from source to target image
    ///
    /// # Errors
    /// Returns error if image dimensions are invalid
    pub fn transfer(
        source_rgb: &[u8],
        target_rgb: &[u8],
        width: usize,
        height: usize,
    ) -> AlignResult<Vec<u8>> {
        let expected_size = width * height * 3;
        if source_rgb.len() != expected_size || target_rgb.len() != expected_size {
            return Err(AlignError::InvalidConfig("Image size mismatch".to_string()));
        }

        let source_stats = ColorStatistics::from_image(source_rgb, width, height);
        let target_stats = ColorStatistics::from_image(target_rgb, width, height);

        let mut output = vec![0u8; expected_size];

        for (i, pixel) in target_rgb.chunks_exact(3).enumerate() {
            let color = ColorRgb::from_u8(pixel[0], pixel[1], pixel[2]);
            let lab = color.to_lab();

            // Transfer statistics in LAB space
            let l = (lab.l - target_stats.mean_lab.l)
                * (source_stats.std_dev_lab.l / target_stats.std_dev_lab.l.max(1e-6))
                + source_stats.mean_lab.l;
            let a = (lab.a - target_stats.mean_lab.a)
                * (source_stats.std_dev_lab.a / target_stats.std_dev_lab.a.max(1e-6))
                + source_stats.mean_lab.a;
            let b = (lab.b - target_stats.mean_lab.b)
                * (source_stats.std_dev_lab.b / target_stats.std_dev_lab.b.max(1e-6))
                + source_stats.mean_lab.b;

            let transferred_lab = ColorLab::new(l, a, b);
            let transferred_rgb = transferred_lab.to_rgb();
            let (r, g, b_val) = transferred_rgb.to_u8();

            output[i * 3] = r;
            output[i * 3 + 1] = g;
            output[i * 3 + 2] = b_val;
        }

        Ok(output)
    }
}

/// Histogram matching
pub struct HistogramMatcher;

impl HistogramMatcher {
    /// Match histogram of target to source
    ///
    /// # Errors
    /// Returns error if image dimensions are invalid
    pub fn match_histogram(
        source: &[u8],
        target: &[u8],
        width: usize,
        height: usize,
    ) -> AlignResult<Vec<u8>> {
        let expected_size = width * height * 3;
        if source.len() != expected_size || target.len() != expected_size {
            return Err(AlignError::InvalidConfig("Image size mismatch".to_string()));
        }

        let mut output = vec![0u8; expected_size];

        // Process each channel independently
        for channel in 0..3 {
            let source_channel: Vec<u8> = source.iter().skip(channel).step_by(3).copied().collect();
            let target_channel: Vec<u8> = target.iter().skip(channel).step_by(3).copied().collect();

            let matched = Self::match_channel(&source_channel, &target_channel);

            for (i, &value) in matched.iter().enumerate() {
                output[i * 3 + channel] = value;
            }
        }

        Ok(output)
    }

    /// Match single channel histogram
    fn match_channel(source: &[u8], target: &[u8]) -> Vec<u8> {
        // Compute histograms
        let source_hist = Self::compute_histogram(source);
        let target_hist = Self::compute_histogram(target);

        // Compute CDFs
        let source_cdf = Self::compute_cdf(&source_hist);
        let target_cdf = Self::compute_cdf(&target_hist);

        // Build lookup table
        let lut = Self::build_lut(&source_cdf, &target_cdf);

        // Apply lookup table
        target.iter().map(|&v| lut[v as usize]).collect()
    }

    /// Compute histogram
    fn compute_histogram(data: &[u8]) -> [u32; 256] {
        let mut hist = [0u32; 256];
        for &value in data {
            hist[value as usize] += 1;
        }
        hist
    }

    /// Compute cumulative distribution function
    fn compute_cdf(hist: &[u32; 256]) -> [f32; 256] {
        let mut cdf = [0.0f32; 256];
        let total: u32 = hist.iter().sum();

        if total == 0 {
            return cdf;
        }

        let mut cumsum = 0u32;
        for (i, &count) in hist.iter().enumerate() {
            cumsum += count;
            cdf[i] = cumsum as f32 / total as f32;
        }

        cdf
    }

    /// Build lookup table
    fn build_lut(source_cdf: &[f32; 256], target_cdf: &[f32; 256]) -> [u8; 256] {
        let mut lut = [0u8; 256];

        for (target_val, &target_prob) in target_cdf.iter().enumerate() {
            // Find closest source value
            let mut best_idx = 0;
            let mut best_diff = f32::INFINITY;

            for (source_val, &source_prob) in source_cdf.iter().enumerate() {
                let diff = (source_prob - target_prob).abs();
                if diff < best_diff {
                    best_diff = diff;
                    best_idx = source_val;
                }
            }

            lut[target_val] = best_idx as u8;
        }

        lut
    }
}

/// White balance estimator
pub struct WhiteBalanceEstimator;

impl WhiteBalanceEstimator {
    /// Estimate white balance using gray world assumption
    #[must_use]
    pub fn estimate_gray_world(rgb: &[u8], width: usize, height: usize) -> ColorRgb {
        let stats = ColorStatistics::from_image(rgb, width, height);

        // Gray world: average should be neutral gray
        let avg = (stats.mean.r + stats.mean.g + stats.mean.b) / 3.0;

        ColorRgb::new(
            avg / stats.mean.r.max(1e-6),
            avg / stats.mean.g.max(1e-6),
            avg / stats.mean.b.max(1e-6),
        )
    }

    /// Estimate white balance using white patch assumption
    #[must_use]
    pub fn estimate_white_patch(rgb: &[u8], _width: usize, _height: usize) -> ColorRgb {
        let mut max_r = 0u8;
        let mut max_g = 0u8;
        let mut max_b = 0u8;

        for pixel in rgb.chunks_exact(3) {
            max_r = max_r.max(pixel[0]);
            max_g = max_g.max(pixel[1]);
            max_b = max_b.max(pixel[2]);
        }

        let max_val = max_r.max(max_g).max(max_b);

        ColorRgb::new(
            f32::from(max_val) / f32::from(max_r).max(1.0),
            f32::from(max_val) / f32::from(max_g).max(1.0),
            f32::from(max_val) / f32::from(max_b).max(1.0),
        )
    }

    /// Apply white balance gains
    #[must_use]
    pub fn apply_white_balance(rgb: &[u8], gains: &ColorRgb) -> Vec<u8> {
        rgb.chunks_exact(3)
            .flat_map(|pixel| {
                let r = (f32::from(pixel[0]) * gains.r).clamp(0.0, 255.0) as u8;
                let g = (f32::from(pixel[1]) * gains.g).clamp(0.0, 255.0) as u8;
                let b = (f32::from(pixel[2]) * gains.b).clamp(0.0, 255.0) as u8;
                [r, g, b]
            })
            .collect()
    }
}

/// `ColorChecker` calibration
pub struct ColorCheckerCalibrator {
    /// Reference `ColorChecker` values (24 patches)
    reference: Vec<ColorRgb>,
}

impl Default for ColorCheckerCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl ColorCheckerCalibrator {
    /// Create new `ColorChecker` calibrator
    #[must_use]
    pub fn new() -> Self {
        // X-Rite ColorChecker Classic reference values (approximate sRGB)
        let reference = vec![
            ColorRgb::new(0.451, 0.315, 0.242), // Dark skin
            ColorRgb::new(0.773, 0.586, 0.502), // Light skin
            ColorRgb::new(0.350, 0.439, 0.594), // Blue sky
            ColorRgb::new(0.329, 0.400, 0.241), // Foliage
            ColorRgb::new(0.541, 0.548, 0.742), // Blue flower
            ColorRgb::new(0.492, 0.729, 0.636), // Bluish green
            ColorRgb::new(0.871, 0.482, 0.145), // Orange
            ColorRgb::new(0.299, 0.359, 0.635), // Purplish blue
            ColorRgb::new(0.789, 0.347, 0.376), // Moderate red
            ColorRgb::new(0.353, 0.241, 0.410), // Purple
            ColorRgb::new(0.596, 0.730, 0.247), // Yellow green
            ColorRgb::new(0.914, 0.620, 0.145), // Orange yellow
            ColorRgb::new(0.196, 0.263, 0.557), // Blue
            ColorRgb::new(0.329, 0.565, 0.298), // Green
            ColorRgb::new(0.757, 0.243, 0.224), // Red
            ColorRgb::new(0.918, 0.765, 0.094), // Yellow
            ColorRgb::new(0.773, 0.310, 0.557), // Magenta
            ColorRgb::new(0.149, 0.490, 0.624), // Cyan
            ColorRgb::new(0.957, 0.957, 0.957), // White
            ColorRgb::new(0.788, 0.788, 0.788), // Neutral 8
            ColorRgb::new(0.635, 0.635, 0.635), // Neutral 6.5
            ColorRgb::new(0.478, 0.478, 0.478), // Neutral 5
            ColorRgb::new(0.318, 0.318, 0.318), // Neutral 3.5
            ColorRgb::new(0.200, 0.200, 0.200), // Black
        ];

        Self { reference }
    }

    /// Compute color correction matrix
    ///
    /// # Errors
    /// Returns error if measurements are invalid
    #[allow(dead_code)]
    pub fn compute_correction_matrix(&self, measured: &[ColorRgb]) -> AlignResult<[[f32; 3]; 3]> {
        if measured.len() != 24 {
            return Err(AlignError::InvalidConfig(
                "Need exactly 24 ColorChecker measurements".to_string(),
            ));
        }

        // Simplified: return identity matrix
        // In production, this would use least squares to fit a 3x3 color matrix
        Ok([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    }

    /// Get reference patch color
    #[must_use]
    pub fn get_reference(&self, patch_index: usize) -> Option<ColorRgb> {
        self.reference.get(patch_index).copied()
    }
}

/// Gamma correction
pub struct GammaCorrection;

impl GammaCorrection {
    /// Apply gamma correction
    #[must_use]
    pub fn apply(rgb: &[u8], gamma: f32) -> Vec<u8> {
        let lut: Vec<u8> = (0..256)
            .map(|i| {
                let normalized = i as f32 / 255.0;
                let corrected = normalized.powf(gamma);
                (corrected * 255.0).clamp(0.0, 255.0) as u8
            })
            .collect();

        rgb.iter().map(|&v| lut[v as usize]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_rgb_creation() {
        let color = ColorRgb::new(0.5, 0.6, 0.7);
        assert_eq!(color.r, 0.5);
        assert_eq!(color.g, 0.6);
        assert_eq!(color.b, 0.7);
    }

    #[test]
    fn test_color_rgb_u8_conversion() {
        let color = ColorRgb::from_u8(128, 192, 255);
        let (r, g, b) = color.to_u8();
        assert_eq!(r, 128);
        assert_eq!(g, 192);
        assert_eq!(b, 255);
    }

    #[test]
    fn test_rgb_lab_roundtrip() {
        let rgb = ColorRgb::new(0.5, 0.6, 0.7);
        let lab = rgb.to_lab();
        let rgb2 = lab.to_rgb();

        assert!((rgb.r - rgb2.r).abs() < 0.01);
        assert!((rgb.g - rgb2.g).abs() < 0.01);
        assert!((rgb.b - rgb2.b).abs() < 0.01);
    }

    #[test]
    fn test_color_statistics() {
        let rgb = vec![128u8; 300]; // 10x10 gray image
        let stats = ColorStatistics::from_image(&rgb, 10, 10);

        assert!((stats.mean.r - 0.5).abs() < 0.01);
        assert!((stats.mean.g - 0.5).abs() < 0.01);
        assert!((stats.mean.b - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_histogram_computation() {
        let data = vec![0u8, 0, 128, 128, 255];
        let hist = HistogramMatcher::compute_histogram(&data);
        assert_eq!(hist[0], 2);
        assert_eq!(hist[128], 2);
        assert_eq!(hist[255], 1);
    }

    #[test]
    fn test_white_balance_gray_world() {
        let mut rgb = vec![0u8; 300];
        // Create reddish image
        for i in 0..100 {
            rgb[i * 3] = 200;
            rgb[i * 3 + 1] = 100;
            rgb[i * 3 + 2] = 100;
        }

        let gains = WhiteBalanceEstimator::estimate_gray_world(&rgb, 10, 10);
        assert!(gains.r < 1.0); // Should reduce red
        assert!(gains.g > 1.0); // Should boost green
        assert!(gains.b > 1.0); // Should boost blue
    }

    #[test]
    fn test_gamma_correction() {
        let rgb = vec![128u8; 30];
        let corrected = GammaCorrection::apply(&rgb, 2.2);
        assert_eq!(corrected.len(), 30);
    }

    #[test]
    fn test_colorchecker_calibrator() {
        let calibrator = ColorCheckerCalibrator::new();
        assert_eq!(calibrator.reference.len(), 24);

        let white = calibrator.get_reference(18).expect("white should be valid");
        assert!(white.r > 0.95);
        assert!(white.g > 0.95);
        assert!(white.b > 0.95);
    }
}
