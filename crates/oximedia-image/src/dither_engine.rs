#![allow(dead_code)]
//! Advanced dithering algorithms for image bit-depth reduction.
//!
//! Provides multiple dithering methods for converting high bit-depth images
//! to lower bit-depth representations while preserving perceptual quality.
//! Supports Floyd-Steinberg error diffusion, ordered (Bayer) dithering,
//! and blue-noise threshold dithering.

use std::collections::HashMap;

/// Dithering method selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DitherMethod {
    /// No dithering — simple truncation/rounding.
    None,
    /// Floyd-Steinberg error diffusion.
    FloydSteinberg,
    /// Jarvis-Judice-Ninke error diffusion (larger kernel).
    JarvisJudiceNinke,
    /// Stucki error diffusion.
    Stucki,
    /// Ordered Bayer dithering with configurable matrix size.
    OrderedBayer,
    /// Blue-noise threshold dithering.
    BlueNoise,
}

impl std::fmt::Display for DitherMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::FloydSteinberg => write!(f, "Floyd-Steinberg"),
            Self::JarvisJudiceNinke => write!(f, "Jarvis-Judice-Ninke"),
            Self::Stucki => write!(f, "Stucki"),
            Self::OrderedBayer => write!(f, "Ordered Bayer"),
            Self::BlueNoise => write!(f, "Blue Noise"),
        }
    }
}

/// Configuration for the dither engine.
#[derive(Clone, Debug)]
pub struct DitherConfig {
    /// The dithering method to use.
    pub method: DitherMethod,
    /// Target bit depth (1..=16).
    pub target_bits: u8,
    /// Source bit depth (1..=32).
    pub source_bits: u8,
    /// Error diffusion strength (0.0..=1.0), applies to error diffusion methods.
    pub strength: f64,
    /// Bayer matrix order for ordered dithering (2, 4, 8, 16).
    pub bayer_order: u32,
}

impl Default for DitherConfig {
    fn default() -> Self {
        Self {
            method: DitherMethod::FloydSteinberg,
            target_bits: 8,
            source_bits: 16,
            strength: 1.0,
            bayer_order: 4,
        }
    }
}

impl DitherConfig {
    /// Creates a new dither configuration with the given method and target bits.
    pub fn new(method: DitherMethod, target_bits: u8) -> Self {
        Self {
            method,
            target_bits,
            ..Default::default()
        }
    }

    /// Sets the source bit depth.
    pub fn with_source_bits(mut self, bits: u8) -> Self {
        self.source_bits = bits;
        self
    }

    /// Sets the error diffusion strength.
    pub fn with_strength(mut self, strength: f64) -> Self {
        self.strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Sets the Bayer matrix order.
    pub fn with_bayer_order(mut self, order: u32) -> Self {
        self.bayer_order = order.next_power_of_two().max(2);
        self
    }

    /// Validates the configuration and returns an error message if invalid.
    pub fn validate(&self) -> Result<(), String> {
        if self.target_bits == 0 || self.target_bits > 16 {
            return Err(format!(
                "target_bits must be 1..=16, got {}",
                self.target_bits
            ));
        }
        if self.source_bits == 0 || self.source_bits > 32 {
            return Err(format!(
                "source_bits must be 1..=32, got {}",
                self.source_bits
            ));
        }
        if self.target_bits >= self.source_bits {
            return Err(format!(
                "target_bits ({}) must be < source_bits ({})",
                self.target_bits, self.source_bits
            ));
        }
        Ok(())
    }
}

/// Generates a Bayer threshold matrix of the given order.
///
/// The order must be a power of 2. Returns a square matrix of size `order x order`
/// with threshold values normalized to 0.0..1.0.
#[allow(clippy::cast_precision_loss)]
pub fn generate_bayer_matrix(order: u32) -> Vec<Vec<f64>> {
    let n = order.next_power_of_two().max(2) as usize;
    let mut matrix = vec![vec![0.0_f64; n]; n];

    for (y, row) in matrix.iter_mut().enumerate() {
        for (x, cell) in row.iter_mut().enumerate() {
            *cell = bayer_value(x, y, n) / (n * n) as f64;
        }
    }

    matrix
}

/// Computes the Bayer matrix value at position (x, y) for matrix size n.
#[allow(clippy::cast_precision_loss)]
fn bayer_value(x: usize, y: usize, n: usize) -> f64 {
    if n == 2 {
        let table = [[0, 2], [3, 1]];
        return table[y][x] as f64;
    }

    let half = n / 2;
    let qx = x % half;
    let qy = y % half;
    let quadrant = 2 * (y / half) + (x / half);

    let sub_val = bayer_value(qx, qy, half);
    let multiplier = match quadrant {
        0 => 0,
        1 => 2,
        2 => 3,
        _ => 1,
    };

    #[allow(clippy::cast_precision_loss)]
    {
        4.0 * sub_val + multiplier as f64
    }
}

/// The dither engine applies dithering to pixel buffers.
#[derive(Debug)]
pub struct DitherEngine {
    /// Current configuration.
    config: DitherConfig,
    /// Cached Bayer matrix (if using ordered dithering).
    bayer_cache: Option<Vec<Vec<f64>>>,
}

impl DitherEngine {
    /// Creates a new dither engine with the given configuration.
    pub fn new(config: DitherConfig) -> Self {
        let bayer_cache = if config.method == DitherMethod::OrderedBayer {
            Some(generate_bayer_matrix(config.bayer_order))
        } else {
            None
        };
        Self {
            config,
            bayer_cache,
        }
    }

    /// Returns a reference to the current configuration.
    pub fn config(&self) -> &DitherConfig {
        &self.config
    }

    /// Applies dithering to a single-channel buffer in-place.
    ///
    /// `width` and `height` describe the image dimensions. The buffer must
    /// contain exactly `width * height` samples as `f64` values in 0.0..1.0.
    #[allow(clippy::cast_precision_loss)]
    pub fn apply_f64(&self, buf: &mut [f64], width: usize, height: usize) {
        assert_eq!(buf.len(), width * height, "buffer size mismatch");

        match self.config.method {
            DitherMethod::None => self.apply_none(buf),
            DitherMethod::FloydSteinberg => self.apply_floyd_steinberg(buf, width, height),
            DitherMethod::JarvisJudiceNinke => self.apply_jjn(buf, width, height),
            DitherMethod::Stucki => self.apply_stucki(buf, width, height),
            DitherMethod::OrderedBayer => self.apply_ordered(buf, width, height),
            DitherMethod::BlueNoise => self.apply_blue_noise(buf, width, height),
        }
    }

    /// Quantizes a value to the target bit-depth.
    #[allow(clippy::cast_precision_loss)]
    fn quantize(&self, val: f64) -> f64 {
        let levels = ((1_u32 << self.config.target_bits) - 1) as f64;
        (val * levels).round() / levels
    }

    /// Simple quantization with no dithering.
    fn apply_none(&self, buf: &mut [f64]) {
        for v in buf.iter_mut() {
            *v = self.quantize(v.clamp(0.0, 1.0));
        }
    }

    /// Floyd-Steinberg error diffusion.
    #[allow(clippy::cast_precision_loss)]
    fn apply_floyd_steinberg(&self, buf: &mut [f64], width: usize, height: usize) {
        let strength = self.config.strength;
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let old = buf[idx].clamp(0.0, 1.0);
                let new = self.quantize(old);
                let err = (old - new) * strength;
                buf[idx] = new;

                if x + 1 < width {
                    buf[idx + 1] += err * 7.0 / 16.0;
                }
                if y + 1 < height {
                    if x > 0 {
                        buf[(y + 1) * width + (x - 1)] += err * 3.0 / 16.0;
                    }
                    buf[(y + 1) * width + x] += err * 5.0 / 16.0;
                    if x + 1 < width {
                        buf[(y + 1) * width + (x + 1)] += err * 1.0 / 16.0;
                    }
                }
            }
        }
    }

    /// Jarvis-Judice-Ninke error diffusion.
    #[allow(clippy::cast_precision_loss)]
    fn apply_jjn(&self, buf: &mut [f64], width: usize, height: usize) {
        let strength = self.config.strength;
        let kernel: &[(i32, i32, f64)] = &[
            (1, 0, 7.0),
            (2, 0, 5.0),
            (-2, 1, 3.0),
            (-1, 1, 5.0),
            (0, 1, 7.0),
            (1, 1, 5.0),
            (2, 1, 3.0),
            (-2, 2, 1.0),
            (-1, 2, 3.0),
            (0, 2, 5.0),
            (1, 2, 3.0),
            (2, 2, 1.0),
        ];
        let divisor = 48.0;

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let old = buf[idx].clamp(0.0, 1.0);
                let new = self.quantize(old);
                let err = (old - new) * strength;
                buf[idx] = new;

                for &(dx, dy, weight) in kernel {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                        buf[ny as usize * width + nx as usize] += err * weight / divisor;
                    }
                }
            }
        }
    }

    /// Stucki error diffusion.
    #[allow(clippy::cast_precision_loss)]
    fn apply_stucki(&self, buf: &mut [f64], width: usize, height: usize) {
        let strength = self.config.strength;
        let kernel: &[(i32, i32, f64)] = &[
            (1, 0, 8.0),
            (2, 0, 4.0),
            (-2, 1, 2.0),
            (-1, 1, 4.0),
            (0, 1, 8.0),
            (1, 1, 4.0),
            (2, 1, 2.0),
            (-2, 2, 1.0),
            (-1, 2, 2.0),
            (0, 2, 4.0),
            (1, 2, 2.0),
            (2, 2, 1.0),
        ];
        let divisor = 42.0;

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let old = buf[idx].clamp(0.0, 1.0);
                let new = self.quantize(old);
                let err = (old - new) * strength;
                buf[idx] = new;

                for &(dx, dy, weight) in kernel {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                        buf[ny as usize * width + nx as usize] += err * weight / divisor;
                    }
                }
            }
        }
    }

    /// Ordered Bayer dithering.
    #[allow(clippy::cast_precision_loss)]
    fn apply_ordered(&self, buf: &mut [f64], width: usize, height: usize) {
        let Some(matrix) = self.bayer_cache.as_ref() else {
            return;
        };
        let n = matrix.len();
        let levels = ((1_u32 << self.config.target_bits) - 1) as f64;

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let threshold = matrix[y % n][x % n] - 0.5;
                let val = buf[idx].clamp(0.0, 1.0);
                let dithered = val + threshold / levels;
                buf[idx] = (dithered * levels).round() / levels;
                buf[idx] = buf[idx].clamp(0.0, 1.0);
            }
        }
    }

    /// Blue-noise dithering using a deterministic hash-based approach.
    #[allow(clippy::cast_precision_loss)]
    fn apply_blue_noise(&self, buf: &mut [f64], width: usize, height: usize) {
        let levels = ((1_u32 << self.config.target_bits) - 1) as f64;
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let noise = blue_noise_hash(x as u32, y as u32);
                let val = buf[idx].clamp(0.0, 1.0);
                let dithered = val + (noise - 0.5) / levels;
                buf[idx] = (dithered * levels).round() / levels;
                buf[idx] = buf[idx].clamp(0.0, 1.0);
            }
        }
    }
}

/// Deterministic blue-noise-like hash for dither threshold.
///
/// Uses a simple hash function to produce a pseudo-random value in 0.0..1.0
/// with blue-noise-like spectral properties.
#[allow(clippy::cast_precision_loss)]
fn blue_noise_hash(x: u32, y: u32) -> f64 {
    let mut h = x
        .wrapping_mul(374_761_393)
        .wrapping_add(y.wrapping_mul(668_265_263));
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    (h & 0x00FF_FFFF) as f64 / 16_777_215.0
}

/// Statistics about the dithering result.
#[derive(Clone, Debug, Default)]
pub struct DitherStats {
    /// Number of pixels processed.
    pub pixel_count: usize,
    /// Mean quantization error.
    pub mean_error: f64,
    /// Maximum absolute quantization error.
    pub max_error: f64,
    /// Histogram of output quantization levels (level -> count).
    pub level_histogram: HashMap<u32, usize>,
}

impl DitherStats {
    /// Computes dithering statistics by comparing original and dithered buffers.
    #[allow(clippy::cast_precision_loss)]
    pub fn compute(original: &[f64], dithered: &[f64], target_bits: u8) -> Self {
        assert_eq!(original.len(), dithered.len());
        let n = original.len();
        let levels = ((1_u32 << target_bits) - 1) as f64;

        let mut total_error = 0.0;
        let mut max_error = 0.0_f64;
        let mut histogram = HashMap::new();

        for i in 0..n {
            let err = (original[i] - dithered[i]).abs();
            total_error += err;
            max_error = max_error.max(err);

            let level = (dithered[i] * levels).round() as u32;
            *histogram.entry(level).or_insert(0) += 1;
        }

        Self {
            pixel_count: n,
            mean_error: if n > 0 { total_error / n as f64 } else { 0.0 },
            max_error,
            level_histogram: histogram,
        }
    }
}

/// Converts a buffer of u16 samples to f64 (0.0..1.0) given source bit depth.
#[allow(clippy::cast_precision_loss)]
pub fn u16_to_f64_buffer(src: &[u16], source_bits: u8) -> Vec<f64> {
    let max_val = ((1_u32 << source_bits) - 1) as f64;
    src.iter().map(|&v| f64::from(v) / max_val).collect()
}

/// Converts a buffer of f64 (0.0..1.0) to u16 values at the given target bit depth.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn f64_to_u16_buffer(src: &[f64], target_bits: u8) -> Vec<u16> {
    let max_val = ((1_u32 << target_bits) - 1) as f64;
    src.iter()
        .map(|&v| (v.clamp(0.0, 1.0) * max_val).round() as u16)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dither_config_default() {
        let cfg = DitherConfig::default();
        assert_eq!(cfg.method, DitherMethod::FloydSteinberg);
        assert_eq!(cfg.target_bits, 8);
        assert_eq!(cfg.source_bits, 16);
        assert!((cfg.strength - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dither_config_validation() {
        let cfg = DitherConfig::new(DitherMethod::None, 8).with_source_bits(16);
        assert!(cfg.validate().is_ok());

        let bad = DitherConfig::new(DitherMethod::None, 0);
        assert!(bad.validate().is_err());

        let bad2 = DitherConfig::new(DitherMethod::None, 16).with_source_bits(8);
        assert!(bad2.validate().is_err());
    }

    #[test]
    fn test_bayer_matrix_2x2() {
        let m = generate_bayer_matrix(2);
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].len(), 2);
        // Values should be in 0.0..1.0
        for row in &m {
            for &v in row {
                assert!(v >= 0.0 && v < 1.0, "bayer value out of range: {v}");
            }
        }
    }

    #[test]
    fn test_bayer_matrix_4x4() {
        let m = generate_bayer_matrix(4);
        assert_eq!(m.len(), 4);
        // All 16 values should be unique fractions of 1/16
        let mut vals: Vec<f64> = m.iter().flatten().copied().collect();
        vals.sort_by(|a, b| a.partial_cmp(b).expect("should succeed in test"));
        for i in 1..vals.len() {
            assert!(
                (vals[i] - vals[i - 1]).abs() > 1e-10,
                "duplicate bayer values"
            );
        }
    }

    #[test]
    fn test_dither_none() {
        let cfg = DitherConfig::new(DitherMethod::None, 1).with_source_bits(8);
        let engine = DitherEngine::new(cfg);
        let mut buf = vec![0.0, 0.25, 0.5, 0.75, 1.0, 0.3];
        let w = 3;
        let h = 2;
        engine.apply_f64(&mut buf, w, h);
        // With 1-bit target, values should be 0.0 or 1.0
        for &v in &buf {
            assert!(v == 0.0 || v == 1.0, "expected 0 or 1, got {v}");
        }
    }

    #[test]
    fn test_floyd_steinberg_basic() {
        let cfg = DitherConfig::new(DitherMethod::FloydSteinberg, 1).with_source_bits(8);
        let engine = DitherEngine::new(cfg);
        let mut buf = vec![0.5; 16];
        engine.apply_f64(&mut buf, 4, 4);
        // After FS dithering of 0.5 gray, output should be mix of 0 and 1
        let zeros = buf.iter().filter(|&&v| v == 0.0).count();
        let ones = buf.iter().filter(|&&v| v == 1.0).count();
        assert_eq!(zeros + ones, 16, "all values should be 0 or 1");
        // Roughly half should be each
        assert!(zeros > 2 && ones > 2, "should have a mix of 0 and 1");
    }

    #[test]
    fn test_jjn_dither() {
        let cfg = DitherConfig::new(DitherMethod::JarvisJudiceNinke, 1).with_source_bits(8);
        let engine = DitherEngine::new(cfg);
        let mut buf = vec![0.5; 36];
        engine.apply_f64(&mut buf, 6, 6);
        for &v in &buf {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_stucki_dither() {
        let cfg = DitherConfig::new(DitherMethod::Stucki, 1).with_source_bits(8);
        let engine = DitherEngine::new(cfg);
        let mut buf = vec![0.5; 36];
        engine.apply_f64(&mut buf, 6, 6);
        for &v in &buf {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_ordered_bayer_dither() {
        let cfg = DitherConfig::new(DitherMethod::OrderedBayer, 4)
            .with_source_bits(8)
            .with_bayer_order(4);
        let engine = DitherEngine::new(cfg);
        let mut buf = vec![0.5; 16];
        engine.apply_f64(&mut buf, 4, 4);
        // 4-bit target has 16 levels, values should be quantized
        for &v in &buf {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_blue_noise_dither() {
        let cfg = DitherConfig::new(DitherMethod::BlueNoise, 4).with_source_bits(8);
        let engine = DitherEngine::new(cfg);
        let mut buf = vec![0.5; 16];
        engine.apply_f64(&mut buf, 4, 4);
        for &v in &buf {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_blue_noise_hash_range() {
        for y in 0..100 {
            for x in 0..100 {
                let h = blue_noise_hash(x, y);
                assert!(h >= 0.0 && h <= 1.0, "hash out of range at ({x},{y}): {h}");
            }
        }
    }

    #[test]
    fn test_dither_stats() {
        let original = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let dithered = vec![0.0, 0.2, 0.5, 0.8, 1.0];
        let stats = DitherStats::compute(&original, &dithered, 8);
        assert_eq!(stats.pixel_count, 5);
        assert!(stats.mean_error >= 0.0);
        assert!(stats.max_error >= 0.0);
        assert!(!stats.level_histogram.is_empty());
    }

    #[test]
    fn test_u16_f64_roundtrip() {
        let src: Vec<u16> = vec![0, 128, 255, 512, 1023];
        let f64_buf = u16_to_f64_buffer(&src, 10);
        for &v in &f64_buf {
            assert!(v >= 0.0 && v <= 1.0);
        }
        let back = f64_to_u16_buffer(&f64_buf, 10);
        assert_eq!(src, back);
    }

    #[test]
    fn test_dither_method_display() {
        assert_eq!(DitherMethod::FloydSteinberg.to_string(), "Floyd-Steinberg");
        assert_eq!(DitherMethod::OrderedBayer.to_string(), "Ordered Bayer");
        assert_eq!(DitherMethod::BlueNoise.to_string(), "Blue Noise");
        assert_eq!(DitherMethod::None.to_string(), "None");
    }

    #[test]
    fn test_strength_clamping() {
        let cfg = DitherConfig::default().with_strength(2.0);
        assert!((cfg.strength - 1.0).abs() < f64::EPSILON);
        let cfg2 = DitherConfig::default().with_strength(-0.5);
        assert!((cfg2.strength).abs() < f64::EPSILON);
    }
}
