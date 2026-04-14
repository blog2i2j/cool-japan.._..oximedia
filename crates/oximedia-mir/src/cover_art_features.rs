//! Cover art visual feature extraction for Music Information Retrieval.
//!
//! Extracts perceptually relevant visual features from album cover art pixel
//! buffers, enabling downstream tasks such as mood-based playlist sorting,
//! visual genre hints, or artwork quality scoring.
//!
//! # Features extracted
//!
//! - **Dominant colors** via median-cut quantization
//! - **Perceptual brightness** (Rec. 709 luma) and **RMS contrast**
//! - **Visual complexity** from edge density, color entropy, and DCT-based
//!   spatial frequency analysis
//!
//! # Example
//!
//! ```
//! use oximedia_mir::cover_art_features::{CoverArtAnalyzer, CoverArtConfig};
//!
//! // Create a 4×4 RGB checkerboard
//! let mut pixels = vec![0u8; 4 * 4 * 3];
//! for y in 0..4 {
//!     for x in 0..4 {
//!         let idx = (y * 4 + x) * 3;
//!         let v = if (x + y) % 2 == 0 { 255 } else { 0 };
//!         pixels[idx] = v;
//!         pixels[idx + 1] = v;
//!         pixels[idx + 2] = v;
//!     }
//! }
//! let config = CoverArtConfig::default();
//! let analyzer = CoverArtAnalyzer::new(config);
//! let result = analyzer.analyze(&pixels, 4, 4, 3).unwrap();
//! println!("Brightness: {:.3}", result.brightness_contrast.mean_brightness);
//! println!("Complexity: {:.3}", result.complexity.overall_score);
//! ```

#![allow(dead_code)]

use std::f64::consts::LN_2;

// ── Errors ────────────────────────────────────────────────────────────────────

/// Errors that can occur during cover art feature extraction.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum CoverArtError {
    /// Buffer length does not match `width * height * channels`.
    BufferSizeMismatch {
        expected: usize,
        actual: usize,
    },
    /// Unsupported number of channels (only 3 = RGB and 4 = RGBA are supported).
    UnsupportedChannels(u8),
    /// Image is too small for meaningful analysis.
    ImageTooSmall {
        min_width: u32,
        min_height: u32,
        actual_width: u32,
        actual_height: u32,
    },
    /// Image has zero pixels.
    EmptyImage,
}

impl std::fmt::Display for CoverArtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BufferSizeMismatch { expected, actual } => {
                write!(f, "buffer size mismatch: expected {expected}, got {actual}")
            }
            Self::UnsupportedChannels(c) => {
                write!(f, "unsupported channel count {c} (only 3 or 4 supported)")
            }
            Self::ImageTooSmall {
                min_width,
                min_height,
                actual_width,
                actual_height,
            } => write!(
                f,
                "image {actual_width}×{actual_height} too small, minimum {min_width}×{min_height}"
            ),
            Self::EmptyImage => write!(f, "image has zero pixels"),
        }
    }
}

impl std::error::Error for CoverArtError {}

/// Alias for `Result<T, CoverArtError>`.
pub type CoverArtResult<T> = Result<T, CoverArtError>;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for [`CoverArtAnalyzer`].
#[derive(Debug, Clone)]
pub struct CoverArtConfig {
    /// Number of dominant colors to extract (default: 8).
    pub num_dominant_colors: usize,

    /// Number of hue histogram bins for color entropy (default: 36 → 10°/bin).
    pub hue_histogram_bins: usize,

    /// Minimum HSV saturation (0–255) for a pixel to contribute to the hue
    /// histogram.  Low-saturation pixels are mostly grey. Default: 30.
    pub min_saturation_for_hue: u8,

    /// Block size for DCT spatial frequency analysis (default: 8).
    pub dct_block_size: usize,
}

impl Default for CoverArtConfig {
    fn default() -> Self {
        Self {
            num_dominant_colors: 8,
            hue_histogram_bins: 36,
            min_saturation_for_hue: 30,
            dct_block_size: 8,
        }
    }
}

// ── Color types ───────────────────────────────────────────────────────────────

/// An RGB color with 8-bit channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RgbColor {
    /// Red channel (0–255).
    pub r: u8,
    /// Green channel (0–255).
    pub g: u8,
    /// Blue channel (0–255).
    pub b: u8,
}

impl RgbColor {
    /// Rec. 709 luma (perceptual brightness), normalized to [0, 1].
    #[must_use]
    pub fn luma(self) -> f64 {
        0.2126 * f64::from(self.r) / 255.0
            + 0.7152 * f64::from(self.g) / 255.0
            + 0.0722 * f64::from(self.b) / 255.0
    }
}

/// A dominant color together with its fraction of pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct DominantColor {
    /// The representative color.
    pub color: RgbColor,
    /// Fraction of image pixels represented by this color (0.0–1.0).
    pub fraction: f64,
}

// ── Result types ──────────────────────────────────────────────────────────────

/// Dominant color analysis result.
#[derive(Debug, Clone, PartialEq)]
pub struct DominantColorResult {
    /// Dominant colors sorted by descending population fraction.
    pub colors: Vec<DominantColor>,
    /// Color diversity index: effective number of distinct colors
    /// (exponential of Shannon entropy of the fractions).
    pub diversity_index: f64,
}

/// Brightness and contrast analysis result.
#[derive(Debug, Clone, PartialEq)]
pub struct BrightnessContrastResult {
    /// Mean Rec. 709 luma of all pixels, in [0, 1].
    pub mean_brightness: f64,
    /// RMS contrast: `sqrt(mean((luma - mean_luma)^2))`, in [0, 1].
    pub rms_contrast: f64,
    /// Minimum luma found in the image.
    pub min_brightness: f64,
    /// Maximum luma found in the image.
    pub max_brightness: f64,
    /// Dynamic range: `max - min`.
    pub dynamic_range: f64,
}

/// Visual complexity analysis result.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexityResult {
    /// Edge density: mean Sobel gradient magnitude normalized to [0, 1].
    pub edge_density: f64,
    /// Color entropy: Shannon entropy of the hue histogram (nats), in
    /// [0, ln(num_bins)].
    pub color_entropy_nats: f64,
    /// Spatial frequency score from 8×8 DCT blocks (AC energy / DC energy),
    /// clamped to [0, 1].
    pub spatial_frequency_score: f64,
    /// Overall weighted complexity score in [0, 1].
    pub overall_score: f64,
}

/// Aggregated cover art features.
#[derive(Debug, Clone, PartialEq)]
pub struct CoverArtFeatures {
    /// Dominant color extraction result.
    pub dominant_colors: DominantColorResult,
    /// Brightness and contrast result.
    pub brightness_contrast: BrightnessContrastResult,
    /// Visual complexity result.
    pub complexity: ComplexityResult,
    /// Image dimensions.
    pub width: u32,
    /// Image dimensions.
    pub height: u32,
}

// ── Internal: pixel extraction ────────────────────────────────────────────────

/// Extract RGB triples from a raw byte buffer.
fn extract_rgb(pixels: &[u8], channels: u8) -> Vec<RgbColor> {
    let step = channels as usize;
    let count = pixels.len() / step;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let base = i * step;
        out.push(RgbColor {
            r: pixels[base],
            g: pixels[base + 1],
            b: pixels[base + 2],
        });
    }
    out
}

// ── Internal: median-cut quantization ────────────────────────────────────────

/// A bucket of colors for median-cut.
struct ColorBucket {
    colors: Vec<RgbColor>,
}

impl ColorBucket {
    fn new(colors: Vec<RgbColor>) -> Self {
        Self { colors }
    }

    /// Range of the bucket along the R, G, B axes.
    fn ranges(&self) -> (u8, u8, u8) {
        let mut rmin = 255u8;
        let mut rmax = 0u8;
        let mut gmin = 255u8;
        let mut gmax = 0u8;
        let mut bmin = 255u8;
        let mut bmax = 0u8;
        for c in &self.colors {
            rmin = rmin.min(c.r);
            rmax = rmax.max(c.r);
            gmin = gmin.min(c.g);
            gmax = gmax.max(c.g);
            bmin = bmin.min(c.b);
            bmax = bmax.max(c.b);
        }
        (rmax - rmin, gmax - gmin, bmax - bmin)
    }

    /// Mean color of the bucket.
    fn mean_color(&self) -> RgbColor {
        if self.colors.is_empty() {
            return RgbColor { r: 0, g: 0, b: 0 };
        }
        let n = self.colors.len() as u64;
        let r = self.colors.iter().map(|c| u64::from(c.r)).sum::<u64>() / n;
        let g = self.colors.iter().map(|c| u64::from(c.g)).sum::<u64>() / n;
        let b = self.colors.iter().map(|c| u64::from(c.b)).sum::<u64>() / n;
        RgbColor {
            r: r as u8,
            g: g as u8,
            b: b as u8,
        }
    }

    /// Split the bucket along the axis with the largest range.
    /// Returns (lower_half, upper_half).
    fn split(mut self) -> (ColorBucket, ColorBucket) {
        let (dr, dg, db) = self.ranges();
        // Sort along the widest axis
        if dr >= dg && dr >= db {
            self.colors.sort_unstable_by_key(|c| c.r);
        } else if dg >= db {
            self.colors.sort_unstable_by_key(|c| c.g);
        } else {
            self.colors.sort_unstable_by_key(|c| c.b);
        }
        let mid = self.colors.len() / 2;
        let upper = self.colors.split_off(mid);
        (ColorBucket::new(self.colors), ColorBucket::new(upper))
    }
}

/// Median-cut quantization. Returns up to `num_colors` representative colors
/// with their population fractions.
fn median_cut(colors: &[RgbColor], num_colors: usize) -> Vec<DominantColor> {
    if colors.is_empty() || num_colors == 0 {
        return Vec::new();
    }

    let mut buckets: Vec<ColorBucket> = vec![ColorBucket::new(colors.to_vec())];

    // Keep splitting until we have enough buckets or can't split further
    let target = num_colors.min(colors.len());
    while buckets.len() < target {
        // Find the bucket with the largest range
        let best = buckets
            .iter()
            .enumerate()
            .max_by_key(|(_, b)| {
                let (dr, dg, db) = b.ranges();
                u16::from(dr) + u16::from(dg) + u16::from(db)
            })
            .map(|(i, _)| i);

        let idx = match best {
            Some(i) => i,
            None => break,
        };

        let bucket = buckets.remove(idx);
        if bucket.colors.len() < 2 {
            buckets.push(bucket);
            break;
        }
        let (lo, hi) = bucket.split();
        buckets.push(lo);
        buckets.push(hi);
    }

    let total = colors.len() as f64;
    let mut result: Vec<DominantColor> = buckets
        .into_iter()
        .map(|b| {
            let fraction = b.colors.len() as f64 / total;
            DominantColor {
                color: b.mean_color(),
                fraction,
            }
        })
        .collect();

    // Sort by descending fraction
    result.sort_by(|a, b| b.fraction.partial_cmp(&a.fraction).unwrap_or(std::cmp::Ordering::Equal));
    result
}

// ── Internal: HSV conversion ──────────────────────────────────────────────────

/// Convert an RGB color to HSV. Returns (H [0,360), S [0,255], V [0,255]).
fn rgb_to_hsv(c: RgbColor) -> (f64, u8, u8) {
    let r = f64::from(c.r) / 255.0;
    let g = f64::from(c.g) / 255.0;
    let b = f64::from(c.b) / 255.0;
    let cmax = r.max(g).max(b);
    let cmin = r.min(g).min(b);
    let delta = cmax - cmin;

    let h = if delta < 1e-10 {
        0.0
    } else if (cmax - r).abs() < 1e-10 {
        60.0 * (((g - b) / delta) % 6.0)
    } else if (cmax - g).abs() < 1e-10 {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };
    let h = ((h % 360.0) + 360.0) % 360.0;

    let s = if cmax < 1e-10 { 0.0 } else { delta / cmax };
    let v = cmax;

    ((h), (s * 255.0).round() as u8, (v * 255.0).round() as u8)
}

// ── Internal: Sobel edge detection ────────────────────────────────────────────

/// Compute mean Sobel gradient magnitude (normalized to [0, 1]) over a
/// grayscale image stored as a flat Vec of u8 values (row-major).
fn sobel_edge_density(gray: &[f64], width: usize, height: usize) -> f64 {
    if width < 3 || height < 3 {
        return 0.0;
    }
    let mut total_grad = 0.0_f64;
    let count = (width - 2) * (height - 2);

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let get = |dy: isize, dx: isize| -> f64 {
                let ry = (y as isize + dy) as usize;
                let rx = (x as isize + dx) as usize;
                gray[ry * width + rx]
            };
            // Sobel kernels
            let gx = -get(-1, -1) + get(-1, 1) - 2.0 * get(0, -1) + 2.0 * get(0, 1)
                - get(1, -1)
                + get(1, 1);
            let gy = -get(-1, -1) - 2.0 * get(-1, 0) - get(-1, 1) + get(1, -1)
                + 2.0 * get(1, 0)
                + get(1, 1);
            total_grad += (gx * gx + gy * gy).sqrt();
        }
    }
    // Maximum possible gradient per pixel is approximately 4*sqrt(2) ≈ 5.66
    let max_per_pixel = 4.0 * 2.0_f64.sqrt();
    (total_grad / (count as f64 * max_per_pixel)).clamp(0.0, 1.0)
}

// ── Internal: 2D DCT-II for spatial frequency ─────────────────────────────────

/// Compute the 2D DCT-II of a square block stored in row-major order.
/// Returns the DCT coefficients.
fn dct2_block(block: &[f64], size: usize) -> Vec<f64> {
    let n = size;
    let mut row_dct = vec![0.0_f64; n * n];

    // 1D DCT-II on each row
    for row in 0..n {
        for k in 0..n {
            let mut sum = 0.0_f64;
            for i in 0..n {
                let angle = std::f64::consts::PI * k as f64 * (2.0 * i as f64 + 1.0) / (2.0 * n as f64);
                sum += block[row * n + i] * angle.cos();
            }
            let scale = if k == 0 {
                (1.0 / n as f64).sqrt()
            } else {
                (2.0 / n as f64).sqrt()
            };
            row_dct[row * n + k] = sum * scale;
        }
    }

    // 1D DCT-II on each column of row_dct
    let mut out = vec![0.0_f64; n * n];
    for col in 0..n {
        for k in 0..n {
            let mut sum = 0.0_f64;
            for i in 0..n {
                let angle = std::f64::consts::PI * k as f64 * (2.0 * i as f64 + 1.0) / (2.0 * n as f64);
                sum += row_dct[i * n + col] * angle.cos();
            }
            let scale = if k == 0 {
                (1.0 / n as f64).sqrt()
            } else {
                (2.0 / n as f64).sqrt()
            };
            out[k * n + col] = sum * scale;
        }
    }
    out
}

/// Compute the spatial frequency score from 8×8 DCT blocks.
/// Returns AC energy / (DC energy + AC energy), clamped to [0, 1].
fn spatial_frequency_score(gray: &[f64], width: usize, height: usize, block_size: usize) -> f64 {
    if width < block_size || height < block_size {
        return 0.0;
    }
    let mut total_dc = 0.0_f64;
    let mut total_ac = 0.0_f64;
    let mut block_count = 0usize;

    let y_blocks = height / block_size;
    let x_blocks = width / block_size;

    let mut block = vec![0.0_f64; block_size * block_size];

    for by in 0..y_blocks {
        for bx in 0..x_blocks {
            for r in 0..block_size {
                for c in 0..block_size {
                    block[r * block_size + c] =
                        gray[(by * block_size + r) * width + (bx * block_size + c)];
                }
            }
            let dct = dct2_block(&block, block_size);
            // DC component is dct[0]
            let dc = dct[0] * dct[0];
            let ac: f64 = dct[1..].iter().map(|&v| v * v).sum();
            total_dc += dc;
            total_ac += ac;
            block_count += 1;
        }
    }

    if block_count == 0 {
        return 0.0;
    }
    let denom = total_dc + total_ac;
    if denom < 1e-15 {
        return 0.0;
    }
    (total_ac / denom).clamp(0.0, 1.0)
}

// ── Main analyzer ─────────────────────────────────────────────────────────────

/// Main entry point for cover art feature analysis.
pub struct CoverArtAnalyzer {
    config: CoverArtConfig,
}

impl CoverArtAnalyzer {
    /// Create a new analyzer with the given configuration.
    #[must_use]
    pub fn new(config: CoverArtConfig) -> Self {
        Self { config }
    }

    /// Analyze a raw pixel buffer.
    ///
    /// # Arguments
    ///
    /// * `pixels` — raw image data in row-major order
    /// * `width` — image width in pixels
    /// * `height` — image height in pixels
    /// * `channels` — bytes per pixel: `3` (RGB) or `4` (RGBA)
    ///
    /// # Errors
    ///
    /// - [`CoverArtError::BufferSizeMismatch`] if buffer length ≠ `width * height * channels`
    /// - [`CoverArtError::UnsupportedChannels`] if `channels` is not 3 or 4
    /// - [`CoverArtError::ImageTooSmall`] if image is smaller than 4×4
    /// - [`CoverArtError::EmptyImage`] if `width == 0` or `height == 0`
    pub fn analyze(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: u8,
    ) -> CoverArtResult<CoverArtFeatures> {
        // ── Validation ─────────────────────────────────────────────────────
        if width == 0 || height == 0 {
            return Err(CoverArtError::EmptyImage);
        }
        if channels != 3 && channels != 4 {
            return Err(CoverArtError::UnsupportedChannels(channels));
        }
        let expected_len = width as usize * height as usize * channels as usize;
        if pixels.len() != expected_len {
            return Err(CoverArtError::BufferSizeMismatch {
                expected: expected_len,
                actual: pixels.len(),
            });
        }
        if width < 4 || height < 4 {
            return Err(CoverArtError::ImageTooSmall {
                min_width: 4,
                min_height: 4,
                actual_width: width,
                actual_height: height,
            });
        }

        // ── Extract RGB pixels ─────────────────────────────────────────────
        let rgb_pixels = extract_rgb(pixels, channels);

        // ── Dominant colors ────────────────────────────────────────────────
        let dc_result = self.analyze_dominant_colors(&rgb_pixels);

        // ── Brightness and contrast ────────────────────────────────────────
        let bc_result = Self::analyze_brightness_contrast(&rgb_pixels);

        // ── Complexity ────────────────────────────────────────────────────
        let complexity =
            self.analyze_complexity(&rgb_pixels, width as usize, height as usize);

        Ok(CoverArtFeatures {
            dominant_colors: dc_result,
            brightness_contrast: bc_result,
            complexity,
            width,
            height,
        })
    }

    // ── Private: dominant colors ───────────────────────────────────────────────

    fn analyze_dominant_colors(&self, rgb_pixels: &[RgbColor]) -> DominantColorResult {
        let colors = median_cut(rgb_pixels, self.config.num_dominant_colors);

        // Diversity index: exp(Shannon entropy of fractions)
        let entropy: f64 = colors
            .iter()
            .map(|dc| {
                if dc.fraction < 1e-15 {
                    0.0
                } else {
                    -dc.fraction * dc.fraction.log2()
                }
            })
            .sum();
        let diversity_index = (entropy * LN_2).exp(); // convert to nats then exp

        DominantColorResult {
            colors,
            diversity_index,
        }
    }

    // ── Private: brightness & contrast ────────────────────────────────────────

    fn analyze_brightness_contrast(rgb_pixels: &[RgbColor]) -> BrightnessContrastResult {
        let lumas: Vec<f64> = rgb_pixels.iter().map(|c| c.luma()).collect();
        let n = lumas.len() as f64;

        let mean_brightness = lumas.iter().sum::<f64>() / n;
        let variance = lumas.iter().map(|&l| (l - mean_brightness).powi(2)).sum::<f64>() / n;
        let rms_contrast = variance.sqrt();

        let min_brightness = lumas.iter().copied().fold(f64::INFINITY, f64::min);
        let max_brightness = lumas.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let dynamic_range = max_brightness - min_brightness;

        BrightnessContrastResult {
            mean_brightness,
            rms_contrast,
            min_brightness: min_brightness.max(0.0),
            max_brightness: max_brightness.min(1.0),
            dynamic_range,
        }
    }

    // ── Private: complexity ────────────────────────────────────────────────────

    fn analyze_complexity(
        &self,
        rgb_pixels: &[RgbColor],
        width: usize,
        height: usize,
    ) -> ComplexityResult {
        // Grayscale luma values for spatial analysis
        let gray: Vec<f64> = rgb_pixels.iter().map(|c| c.luma()).collect();

        // Edge density via Sobel
        let edge_density = sobel_edge_density(&gray, width, height);

        // Color entropy via HSV hue histogram
        let color_entropy_nats = self.compute_color_entropy(rgb_pixels);

        // Spatial frequency via DCT blocks
        let sf_score =
            spatial_frequency_score(&gray, width, height, self.config.dct_block_size);

        // Weighted combination: edge 40%, color entropy 30%, spatial freq 30%
        let max_entropy = (self.config.hue_histogram_bins as f64).ln();
        let entropy_normalized = if max_entropy > 0.0 {
            (color_entropy_nats / max_entropy).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let overall_score =
            (0.4 * edge_density + 0.3 * entropy_normalized + 0.3 * sf_score).clamp(0.0, 1.0);

        ComplexityResult {
            edge_density,
            color_entropy_nats,
            spatial_frequency_score: sf_score,
            overall_score,
        }
    }

    fn compute_color_entropy(&self, rgb_pixels: &[RgbColor]) -> f64 {
        let bins = self.config.hue_histogram_bins;
        let mut histogram = vec![0u64; bins];

        for &c in rgb_pixels {
            let (h, s, _v) = rgb_to_hsv(c);
            if s < self.config.min_saturation_for_hue {
                continue;
            }
            let bin = ((h / 360.0) * bins as f64).floor() as usize % bins;
            histogram[bin] += 1;
        }

        let total: u64 = histogram.iter().sum();
        if total == 0 {
            return 0.0;
        }
        let total_f = total as f64;
        histogram
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / total_f;
                -p * p.ln()
            })
            .sum()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgb(r: u8, g: u8, b: u8, w: u32, h: u32) -> Vec<u8> {
        let n = (w * h) as usize;
        let mut buf = Vec::with_capacity(n * 3);
        for _ in 0..n {
            buf.push(r);
            buf.push(g);
            buf.push(b);
        }
        buf
    }

    fn checkerboard(w: u32, h: u32) -> Vec<u8> {
        let mut buf = Vec::with_capacity((w * h) as usize * 3);
        for y in 0..h {
            for x in 0..w {
                let v: u8 = if (x + y) % 2 == 0 { 255 } else { 0 };
                buf.push(v);
                buf.push(v);
                buf.push(v);
            }
        }
        buf
    }

    #[test]
    fn test_solid_white_brightness_one() {
        let pixels = solid_rgb(255, 255, 255, 16, 16);
        let config = CoverArtConfig::default();
        let analyzer = CoverArtAnalyzer::new(config);
        let result = analyzer.analyze(&pixels, 16, 16, 3).expect("should succeed");
        assert!(
            (result.brightness_contrast.mean_brightness - 1.0).abs() < 1e-6,
            "expected brightness=1.0, got {:.6}",
            result.brightness_contrast.mean_brightness
        );
    }

    #[test]
    fn test_solid_black_brightness_zero() {
        let pixels = solid_rgb(0, 0, 0, 16, 16);
        let config = CoverArtConfig::default();
        let analyzer = CoverArtAnalyzer::new(config);
        let result = analyzer.analyze(&pixels, 16, 16, 3).expect("should succeed");
        assert!(
            result.brightness_contrast.mean_brightness < 1e-6,
            "expected brightness≈0, got {:.6}",
            result.brightness_contrast.mean_brightness
        );
    }

    #[test]
    fn test_solid_image_zero_contrast() {
        let pixels = solid_rgb(128, 128, 128, 16, 16);
        let config = CoverArtConfig::default();
        let analyzer = CoverArtAnalyzer::new(config);
        let result = analyzer.analyze(&pixels, 16, 16, 3).expect("should succeed");
        assert!(
            result.brightness_contrast.rms_contrast < 1e-6,
            "expected zero contrast for solid image, got {:.6}",
            result.brightness_contrast.rms_contrast
        );
    }

    #[test]
    fn test_banded_image_has_edges() {
        // Create a 32x32 image with alternating 8-pixel-wide vertical black/white bands.
        // The transitions are wide enough for Sobel to detect.
        let w = 32u32;
        let h = 32u32;
        let mut pixels = Vec::with_capacity((w * h) as usize * 3);
        for _y in 0..h {
            for x in 0..w {
                let v: u8 = if (x / 8) % 2 == 0 { 0 } else { 255 };
                pixels.push(v);
                pixels.push(v);
                pixels.push(v);
            }
        }
        let config = CoverArtConfig::default();
        let analyzer = CoverArtAnalyzer::new(config);
        let result = analyzer.analyze(&pixels, w, h, 3).expect("should succeed");
        assert!(
            result.complexity.edge_density > 0.0,
            "banded image should have edges, got {:.4}",
            result.complexity.edge_density
        );
    }

    #[test]
    fn test_solid_image_no_edges() {
        let pixels = solid_rgb(100, 150, 200, 16, 16);
        let config = CoverArtConfig::default();
        let analyzer = CoverArtAnalyzer::new(config);
        let result = analyzer.analyze(&pixels, 16, 16, 3).expect("should succeed");
        assert!(
            result.complexity.edge_density < 1e-6,
            "solid image should have no edges, got {:.6}",
            result.complexity.edge_density
        );
    }

    #[test]
    fn test_dominant_colors_nonempty() {
        let pixels = solid_rgb(200, 50, 50, 16, 16);
        let config = CoverArtConfig::default();
        let analyzer = CoverArtAnalyzer::new(config);
        let result = analyzer.analyze(&pixels, 16, 16, 3).expect("should succeed");
        assert!(
            !result.dominant_colors.colors.is_empty(),
            "should have at least one dominant color"
        );
    }

    #[test]
    fn test_rgba_channels_accepted() {
        let w = 8u32;
        let h = 8u32;
        let mut pixels = Vec::with_capacity((w * h) as usize * 4);
        for _ in 0..(w * h) {
            pixels.push(100u8);
            pixels.push(150u8);
            pixels.push(200u8);
            pixels.push(255u8); // alpha
        }
        let config = CoverArtConfig::default();
        let analyzer = CoverArtAnalyzer::new(config);
        let result = analyzer.analyze(&pixels, w, h, 4);
        assert!(result.is_ok(), "RGBA should be accepted");
    }

    #[test]
    fn test_error_unsupported_channels() {
        let pixels = vec![0u8; 64 * 2]; // 2-channel
        let config = CoverArtConfig::default();
        let analyzer = CoverArtAnalyzer::new(config);
        let err = analyzer.analyze(&pixels, 8, 8, 2).unwrap_err();
        assert!(matches!(err, CoverArtError::UnsupportedChannels(2)));
    }

    #[test]
    fn test_error_buffer_size_mismatch() {
        let pixels = vec![0u8; 10]; // wrong size
        let config = CoverArtConfig::default();
        let analyzer = CoverArtAnalyzer::new(config);
        let err = analyzer.analyze(&pixels, 4, 4, 3).unwrap_err();
        assert!(matches!(err, CoverArtError::BufferSizeMismatch { .. }));
    }

    #[test]
    fn test_error_image_too_small() {
        let pixels = vec![128u8; 3 * 3 * 3];
        let config = CoverArtConfig::default();
        let analyzer = CoverArtAnalyzer::new(config);
        let err = analyzer.analyze(&pixels, 3, 3, 3).unwrap_err();
        assert!(matches!(err, CoverArtError::ImageTooSmall { .. }));
    }

    #[test]
    fn test_complexity_score_in_range() {
        // Use a banded image (wide black/white bands) so edge detection fires
        let w = 32u32;
        let h = 32u32;
        let mut pixels = Vec::with_capacity((w * h) as usize * 3);
        for _y in 0..h {
            for x in 0..w {
                let v: u8 = if (x / 8) % 2 == 0 { 0 } else { 255 };
                pixels.push(v);
                pixels.push(v);
                pixels.push(v);
            }
        }
        let config = CoverArtConfig::default();
        let analyzer = CoverArtAnalyzer::new(config);
        let result = analyzer.analyze(&pixels, w, h, 3).expect("should succeed");
        let score = result.complexity.overall_score;
        assert!(
            (0.0..=1.0).contains(&score),
            "complexity score should be in [0,1], got {score:.4}"
        );
    }
}
