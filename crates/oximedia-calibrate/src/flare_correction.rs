#![allow(dead_code)]
//! Lens flare correction for calibration accuracy.
//!
//! Lens flare (veiling glare) reduces the effective dynamic range and
//! introduces color shifts, which can compromise the accuracy of color
//! calibration measurements. This module provides tools to estimate and
//! correct for flare effects in calibration images.
//!
//! # Features
//!
//! - **Flare estimation**: Estimate the flare level from black patch measurements
//! - **Flare subtraction**: Remove estimated flare from pixel values
//! - **Flare map generation**: Create a spatially-varying flare map
//! - **Contrast ratio correction**: Correct measured contrast ratios for flare

/// Configuration for flare correction.
#[derive(Debug, Clone)]
pub struct FlareConfig {
    /// Number of radial zones for flare map estimation.
    pub num_zones: usize,
    /// Smoothing kernel size for flare map.
    pub smooth_kernel: usize,
    /// Minimum pixel value to consider as valid measurement.
    pub min_valid_value: f64,
    /// Whether to apply spatial variation correction.
    pub spatial_correction: bool,
    /// Flare estimation method.
    pub method: FlareMethod,
}

/// Method used to estimate lens flare.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlareMethod {
    /// Use black patch measurements to estimate uniform flare.
    BlackPatch,
    /// Use surround analysis to estimate spatially varying flare.
    SurroundAnalysis,
    /// Use a known target with black and white patches.
    ContrastTarget,
}

impl Default for FlareConfig {
    fn default() -> Self {
        Self {
            num_zones: 5,
            smooth_kernel: 3,
            min_valid_value: 0.0,
            spatial_correction: true,
            method: FlareMethod::BlackPatch,
        }
    }
}

impl FlareConfig {
    /// Create a new flare correction configuration with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of radial zones.
    #[must_use]
    pub const fn with_num_zones(mut self, zones: usize) -> Self {
        self.num_zones = zones;
        self
    }

    /// Set the smoothing kernel size.
    #[must_use]
    pub const fn with_smooth_kernel(mut self, size: usize) -> Self {
        self.smooth_kernel = size;
        self
    }

    /// Set the flare estimation method.
    #[must_use]
    pub const fn with_method(mut self, method: FlareMethod) -> Self {
        self.method = method;
        self
    }

    /// Enable or disable spatial correction.
    #[must_use]
    pub const fn with_spatial_correction(mut self, enable: bool) -> Self {
        self.spatial_correction = enable;
        self
    }
}

/// RGB color value for flare operations.
#[derive(Debug, Clone, Copy)]
pub struct FlareRgb {
    /// Red channel (0.0..=1.0).
    pub r: f64,
    /// Green channel (0.0..=1.0).
    pub g: f64,
    /// Blue channel (0.0..=1.0).
    pub b: f64,
}

impl FlareRgb {
    /// Create a new RGB value.
    #[must_use]
    pub const fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }

    /// Create a zero (black) RGB value.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            r: 0.0,
            g: 0.0,
            b: 0.0,
        }
    }

    /// Create a uniform gray value.
    #[must_use]
    pub const fn gray(v: f64) -> Self {
        Self { r: v, g: v, b: v }
    }

    /// Compute the luminance (Rec. 709).
    #[must_use]
    pub fn luminance(&self) -> f64 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }

    /// Subtract another RGB value, clamping to zero.
    #[must_use]
    pub fn subtract(&self, other: &Self) -> Self {
        Self {
            r: (self.r - other.r).max(0.0),
            g: (self.g - other.g).max(0.0),
            b: (self.b - other.b).max(0.0),
        }
    }

    /// Scale all channels by a factor.
    #[must_use]
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            r: self.r * factor,
            g: self.g * factor,
            b: self.b * factor,
        }
    }

    /// Per-channel average of two colors.
    #[must_use]
    pub fn average(&self, other: &Self) -> Self {
        Self {
            r: (self.r + other.r) * 0.5,
            g: (self.g + other.g) * 0.5,
            b: (self.b + other.b) * 0.5,
        }
    }
}

/// A measured flare estimate.
#[derive(Debug, Clone)]
pub struct FlareEstimate {
    /// Estimated uniform flare level per channel.
    pub flare_level: FlareRgb,
    /// Flare as a percentage of peak white.
    pub flare_percentage: f64,
    /// Measured contrast ratio before correction.
    pub measured_contrast_ratio: f64,
    /// Corrected contrast ratio after flare removal.
    pub corrected_contrast_ratio: f64,
}

/// A spatially-varying flare map.
#[derive(Debug, Clone)]
pub struct FlareMap {
    /// Width of the flare map.
    pub width: usize,
    /// Height of the flare map.
    pub height: usize,
    /// Per-pixel flare values (RGB, row-major, interleaved).
    pub data: Vec<f64>,
}

impl FlareMap {
    /// Create a uniform flare map.
    #[must_use]
    pub fn uniform(width: usize, height: usize, flare: &FlareRgb) -> Self {
        let mut data = Vec::with_capacity(width * height * 3);
        for _y in 0..height {
            for _x in 0..width {
                data.push(flare.r);
                data.push(flare.g);
                data.push(flare.b);
            }
        }
        Self {
            width,
            height,
            data,
        }
    }

    /// Get the flare value at a pixel coordinate.
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> Option<FlareRgb> {
        if x < self.width && y < self.height {
            let idx = (y * self.width + x) * 3;
            Some(FlareRgb::new(
                self.data[idx],
                self.data[idx + 1],
                self.data[idx + 2],
            ))
        } else {
            None
        }
    }

    /// Set the flare value at a pixel coordinate.
    pub fn set(&mut self, x: usize, y: usize, rgb: &FlareRgb) {
        if x < self.width && y < self.height {
            let idx = (y * self.width + x) * 3;
            self.data[idx] = rgb.r;
            self.data[idx + 1] = rgb.g;
            self.data[idx + 2] = rgb.b;
        }
    }
}

/// Flare corrector for calibration images.
#[derive(Debug)]
pub struct FlareCorrector {
    /// Configuration.
    config: FlareConfig,
}

impl FlareCorrector {
    /// Create a new flare corrector.
    #[must_use]
    pub fn new(config: FlareConfig) -> Self {
        Self { config }
    }

    /// Create a corrector with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(FlareConfig::default())
    }

    /// Estimate flare from black patch measurements.
    ///
    /// The black patches should ideally read as zero; any non-zero reading
    /// is attributed to lens flare (veiling glare).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn estimate_from_black_patches(
        &self,
        black_patches: &[FlareRgb],
        white_reference: &FlareRgb,
    ) -> FlareEstimate {
        if black_patches.is_empty() {
            return FlareEstimate {
                flare_level: FlareRgb::zero(),
                flare_percentage: 0.0,
                measured_contrast_ratio: f64::INFINITY,
                corrected_contrast_ratio: f64::INFINITY,
            };
        }

        let n = black_patches.len() as f64;
        let avg_r: f64 = black_patches.iter().map(|p| p.r).sum::<f64>() / n;
        let avg_g: f64 = black_patches.iter().map(|p| p.g).sum::<f64>() / n;
        let avg_b: f64 = black_patches.iter().map(|p| p.b).sum::<f64>() / n;
        let flare = FlareRgb::new(avg_r, avg_g, avg_b);

        let flare_lum = flare.luminance();
        let white_lum = white_reference.luminance().max(1e-10);
        let flare_pct = (flare_lum / white_lum) * 100.0;

        let measured_cr = if flare_lum > 1e-10 {
            white_lum / flare_lum
        } else {
            f64::INFINITY
        };

        let corrected_black = flare_lum; // After subtracting flare from black, it's ~0
        let corrected_white = (white_lum - flare_lum).max(1e-10);
        let corrected_cr = if corrected_black > 1e-10 {
            corrected_white / corrected_black
        } else {
            // True black after correction
            corrected_white / 1e-10
        };

        FlareEstimate {
            flare_level: flare,
            flare_percentage: flare_pct,
            measured_contrast_ratio: measured_cr,
            corrected_contrast_ratio: corrected_cr,
        }
    }

    /// Estimate flare from a contrast target with known black and white patches.
    #[must_use]
    pub fn estimate_from_contrast_target(
        &self,
        black_measured: &FlareRgb,
        white_measured: &FlareRgb,
        target_contrast_ratio: f64,
    ) -> FlareEstimate {
        // True black = white / target_CR
        let true_black_lum = white_measured.luminance() / target_contrast_ratio.max(1.0);
        let measured_black_lum = black_measured.luminance();

        // Flare = measured_black - true_black
        let flare_lum = (measured_black_lum - true_black_lum).max(0.0);
        let scale = if measured_black_lum > 1e-10 {
            flare_lum / measured_black_lum
        } else {
            0.0
        };

        let flare = FlareRgb::new(
            black_measured.r * scale,
            black_measured.g * scale,
            black_measured.b * scale,
        );

        let white_lum = white_measured.luminance().max(1e-10);
        let flare_pct = (flare_lum / white_lum) * 100.0;
        let measured_cr = if measured_black_lum > 1e-10 {
            white_lum / measured_black_lum
        } else {
            f64::INFINITY
        };

        FlareEstimate {
            flare_level: flare,
            flare_percentage: flare_pct,
            measured_contrast_ratio: measured_cr,
            corrected_contrast_ratio: target_contrast_ratio,
        }
    }

    /// Subtract flare from a set of color measurements.
    #[must_use]
    pub fn correct_measurements(measurements: &[FlareRgb], flare: &FlareRgb) -> Vec<FlareRgb> {
        measurements.iter().map(|m| m.subtract(flare)).collect()
    }

    /// Generate a radial flare map based on distance from center.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn generate_radial_flare_map(
        width: usize,
        height: usize,
        center_flare: &FlareRgb,
        edge_flare: &FlareRgb,
    ) -> FlareMap {
        let cx = width as f64 / 2.0;
        let cy = height as f64 / 2.0;
        let max_r = (cx * cx + cy * cy).sqrt();
        let mut map = FlareMap::uniform(width, height, center_flare);
        for y in 0..height {
            for x in 0..width {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                let r = (dx * dx + dy * dy).sqrt();
                let t = (r / max_r).min(1.0);
                let flare = FlareRgb::new(
                    center_flare.r * (1.0 - t) + edge_flare.r * t,
                    center_flare.g * (1.0 - t) + edge_flare.g * t,
                    center_flare.b * (1.0 - t) + edge_flare.b * t,
                );
                map.set(x, y, &flare);
            }
        }
        map
    }

    /// Get a reference to the configuration.
    #[must_use]
    pub const fn config(&self) -> &FlareConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = FlareConfig::default();
        assert_eq!(cfg.num_zones, 5);
        assert_eq!(cfg.method, FlareMethod::BlackPatch);
        assert!(cfg.spatial_correction);
    }

    #[test]
    fn test_config_builder() {
        let cfg = FlareConfig::new()
            .with_num_zones(8)
            .with_smooth_kernel(5)
            .with_method(FlareMethod::ContrastTarget)
            .with_spatial_correction(false);
        assert_eq!(cfg.num_zones, 8);
        assert_eq!(cfg.smooth_kernel, 5);
        assert_eq!(cfg.method, FlareMethod::ContrastTarget);
        assert!(!cfg.spatial_correction);
    }

    #[test]
    fn test_flare_rgb_luminance() {
        let white = FlareRgb::new(1.0, 1.0, 1.0);
        let lum = white.luminance();
        assert!((lum - 1.0).abs() < 1e-4);

        let black = FlareRgb::zero();
        assert!((black.luminance() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_flare_rgb_subtract() {
        let a = FlareRgb::new(0.5, 0.5, 0.5);
        let b = FlareRgb::new(0.1, 0.1, 0.1);
        let result = a.subtract(&b);
        assert!((result.r - 0.4).abs() < 1e-10);
        assert!((result.g - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_flare_rgb_subtract_clamp() {
        let a = FlareRgb::new(0.05, 0.05, 0.05);
        let b = FlareRgb::new(0.1, 0.1, 0.1);
        let result = a.subtract(&b);
        assert!((result.r - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_flare_rgb_scale() {
        let c = FlareRgb::new(0.5, 0.4, 0.3);
        let scaled = c.scale(2.0);
        assert!((scaled.r - 1.0).abs() < 1e-10);
        assert!((scaled.g - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_flare_rgb_average() {
        let a = FlareRgb::new(0.0, 0.0, 0.0);
        let b = FlareRgb::new(1.0, 1.0, 1.0);
        let avg = a.average(&b);
        assert!((avg.r - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_from_black_patches() {
        let corrector = FlareCorrector::with_defaults();
        let blacks = vec![
            FlareRgb::new(0.02, 0.02, 0.02),
            FlareRgb::new(0.03, 0.03, 0.03),
        ];
        let white = FlareRgb::new(1.0, 1.0, 1.0);
        let est = corrector.estimate_from_black_patches(&blacks, &white);
        assert!(est.flare_percentage > 0.0);
        assert!(est.flare_level.r > 0.0);
        assert!(est.measured_contrast_ratio > 1.0);
    }

    #[test]
    fn test_estimate_from_empty_patches() {
        let corrector = FlareCorrector::with_defaults();
        let est = corrector.estimate_from_black_patches(&[], &FlareRgb::new(1.0, 1.0, 1.0));
        assert!((est.flare_percentage - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_from_contrast_target() {
        let corrector = FlareCorrector::with_defaults();
        let black = FlareRgb::new(0.01, 0.01, 0.01);
        let white = FlareRgb::new(1.0, 1.0, 1.0);
        let est = corrector.estimate_from_contrast_target(&black, &white, 1000.0);
        assert!(est.flare_percentage > 0.0);
        assert!((est.corrected_contrast_ratio - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_correct_measurements() {
        let measurements = vec![FlareRgb::new(0.5, 0.5, 0.5), FlareRgb::new(0.3, 0.3, 0.3)];
        let flare = FlareRgb::new(0.02, 0.02, 0.02);
        let corrected = FlareCorrector::correct_measurements(&measurements, &flare);
        assert_eq!(corrected.len(), 2);
        assert!((corrected[0].r - 0.48).abs() < 1e-10);
        assert!((corrected[1].r - 0.28).abs() < 1e-10);
    }

    #[test]
    fn test_flare_map_uniform() {
        let flare = FlareRgb::new(0.01, 0.01, 0.01);
        let map = FlareMap::uniform(10, 10, &flare);
        let val = map.get(5, 5).expect("expected key to exist");
        assert!((val.r - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_flare_map_set_get() {
        let mut map = FlareMap::uniform(10, 10, &FlareRgb::zero());
        map.set(3, 4, &FlareRgb::new(0.5, 0.6, 0.7));
        let val = map.get(3, 4).expect("expected key to exist");
        assert!((val.r - 0.5).abs() < 1e-10);
        assert!((val.g - 0.6).abs() < 1e-10);
        assert!(map.get(20, 20).is_none());
    }

    #[test]
    fn test_radial_flare_map() {
        let center = FlareRgb::new(0.01, 0.01, 0.01);
        let edge = FlareRgb::new(0.05, 0.05, 0.05);
        let map = FlareCorrector::generate_radial_flare_map(20, 20, &center, &edge);
        // Center should be close to center_flare
        let c = map.get(10, 10).expect("expected key to exist");
        assert!(c.r < 0.03);
        // Corner should be close to edge_flare
        let corner = map.get(0, 0).expect("expected key to exist");
        assert!(corner.r > c.r);
    }
}
