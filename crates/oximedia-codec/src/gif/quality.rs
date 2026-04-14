//! Constant-quality mode for GIF encoding.
//!
//! This module implements a perceptual quality-driven GIF encoding pipeline that
//! automatically selects:
//!
//! - **Color count** (2–256) — more colors → sharper palette, larger file.
//! - **Dithering strategy** — none / Floyd-Steinberg / Bayer ordered.
//! - **Quantization algorithm** — median-cut vs octree based on image complexity.
//!
//! Quality is expressed as a value in `[0.0, 1.0]` where `1.0` is best visual
//! fidelity (256 colors, full dithering, octree quantization) and `0.0` is
//! minimum file size (2 colors, no dithering, median-cut).
//!
//! # Design
//!
//! The quality parameter drives a three-tier decision:
//!
//! | Quality range | Colors  | Dithering        | Quantizer   |
//! |---------------|---------|------------------|-------------|
//! | 0.00 – 0.33   | 2–64    | None             | Median Cut  |
//! | 0.33 – 0.66   | 64–128  | Floyd-Steinberg  | Median Cut  |
//! | 0.66 – 1.00   | 128–256 | Floyd-Steinberg  | Octree      |
//!
//! # Example
//!
//! ```
//! use oximedia_codec::gif::quality::{ConstantQualityConfig, ConstantQualityGifEncoder};
//!
//! let cfg = ConstantQualityConfig::new(0.75); // high quality
//! let encoder = ConstantQualityGifEncoder::new(cfg).expect("valid quality");
//!
//! // Resolve the concrete GIF encoder parameters
//! let gif_params = encoder.resolve();
//! assert!(gif_params.colors() >= 64);
//! ```

#![forbid(unsafe_code)]

use crate::error::{CodecError, CodecResult};
use crate::gif::encoder::{DitheringMethod, GifEncoderConfig, QuantizationMethod};

// ============================================================================
// Quality Level Enum
// ============================================================================

/// Coarse quality tier, derived from the floating-point quality parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum QualityTier {
    /// Low quality — minimum file size, reduced palette.
    Low,
    /// Medium quality — balanced palette and dithering.
    Medium,
    /// High quality — maximum palette depth and full dithering.
    High,
}

impl QualityTier {
    /// Derive a quality tier from a `[0.0, 1.0]` quality value.
    ///
    /// # Errors
    ///
    /// Returns `CodecError::InvalidParameter` when `quality` is outside `[0.0, 1.0]`.
    pub fn from_quality(quality: f32) -> CodecResult<Self> {
        if !(0.0..=1.0).contains(&quality) {
            return Err(CodecError::InvalidParameter(format!(
                "quality must be in [0.0, 1.0], got {quality}"
            )));
        }
        Ok(if quality < 0.334 {
            QualityTier::Low
        } else if quality < 0.667 {
            QualityTier::Medium
        } else {
            QualityTier::High
        })
    }

    /// Returns a human-readable description of the tier.
    pub fn description(self) -> &'static str {
        match self {
            QualityTier::Low => "low (2–64 colors, no dithering)",
            QualityTier::Medium => "medium (64–128 colors, Floyd-Steinberg)",
            QualityTier::High => "high (128–256 colors, Floyd-Steinberg + Octree)",
        }
    }
}

// ============================================================================
// Perceptual Metrics
// ============================================================================

/// Perceptual complexity metrics computed from raw RGBA pixel data.
///
/// These metrics guide the automatic selection of quantizer and dithering
/// strategies within a given quality tier.
#[derive(Clone, Debug)]
pub struct PerceptualMetrics {
    /// Estimate of unique hue regions (0.0 = monochrome, 1.0 = highly chromatic).
    pub chromatic_complexity: f32,
    /// Spatial edge density (0.0 = flat, 1.0 = highly detailed).
    pub edge_density: f32,
    /// Brightness variance normalised to [0.0, 1.0].
    pub luma_variance: f32,
    /// Number of distinct colours sampled (normalised to [0.0, 1.0] against 256).
    pub colour_richness: f32,
}

impl PerceptualMetrics {
    /// Analyse an RGBA pixel buffer.
    ///
    /// `data` must be a flat `[R, G, B, A, ...]` slice with
    /// exactly `width * height * 4` bytes.
    ///
    /// # Errors
    ///
    /// Returns `CodecError::InvalidParameter` when the slice length does not match
    /// `width * height * 4`.
    pub fn analyse(data: &[u8], width: usize, height: usize) -> CodecResult<Self> {
        let expected = width * height * 4;
        if data.len() != expected {
            return Err(CodecError::InvalidParameter(format!(
                "expected {expected} bytes for {width}x{height} RGBA, got {}",
                data.len()
            )));
        }

        let pixels = width * height;
        if pixels == 0 {
            return Ok(Self {
                chromatic_complexity: 0.0,
                edge_density: 0.0,
                luma_variance: 0.0,
                colour_richness: 0.0,
            });
        }

        // ── Luma statistics ──────────────────────────────────────────────────
        let mut luma_sum: f64 = 0.0;
        let mut luma_sq_sum: f64 = 0.0;
        let mut chroma_sum: f64 = 0.0;

        // Use a fixed-size histogram over 256 luma buckets for colour richness.
        let mut luma_hist = [0u32; 256];

        for chunk in data.chunks_exact(4) {
            let r = chunk[0] as f64;
            let g = chunk[1] as f64;
            let b = chunk[2] as f64;

            // BT.601 luma
            let y = 0.299 * r + 0.587 * g + 0.114 * b;
            luma_sum += y;
            luma_sq_sum += y * y;
            luma_hist[y as usize] += 1;

            // Chroma saturation estimate: max(r,g,b) – min(r,g,b)
            let cmax = r.max(g).max(b);
            let cmin = r.min(g).min(b);
            chroma_sum += cmax - cmin;
        }

        let n = pixels as f64;
        let mean_luma = luma_sum / n;
        let variance = (luma_sq_sum / n) - mean_luma * mean_luma;
        let luma_variance = (variance / (255.0 * 255.0 / 4.0)) as f32; // normalise

        let chromatic_complexity = (chroma_sum / (n * 255.0)) as f32;

        // Colour richness: number of non-empty luma buckets / 256
        let non_empty = luma_hist.iter().filter(|&&c| c > 0).count();
        let colour_richness = non_empty as f32 / 256.0;

        // ── Edge density via 3×3 Sobel on luma ──────────────────────────────
        let edge_density = compute_edge_density(data, width, height);

        Ok(Self {
            chromatic_complexity: chromatic_complexity.clamp(0.0, 1.0),
            edge_density: edge_density.clamp(0.0, 1.0),
            luma_variance: luma_variance.clamp(0.0, 1.0),
            colour_richness: colour_richness.clamp(0.0, 1.0),
        })
    }

    /// Aggregate complexity score in `[0.0, 1.0]`.
    ///
    /// Weights: 35% chromatic + 30% edge + 20% luma variance + 15% richness.
    pub fn complexity_score(&self) -> f32 {
        (0.35 * self.chromatic_complexity
            + 0.30 * self.edge_density
            + 0.20 * self.luma_variance
            + 0.15 * self.colour_richness)
            .clamp(0.0, 1.0)
    }
}

/// Compute a normalised edge density using a simplified Sobel operator on luma.
fn compute_edge_density(rgba: &[u8], width: usize, height: usize) -> f32 {
    if width < 3 || height < 3 {
        return 0.0;
    }

    // Extract luma plane
    let luma: Vec<f32> = rgba
        .chunks_exact(4)
        .map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32)
        .collect();

    let mut total_gradient: f64 = 0.0;
    let checked_pixels = (width - 2) * (height - 2);

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let gx = (-luma[(y - 1) * width + (x - 1)] + luma[(y - 1) * width + (x + 1)]
                - 2.0 * luma[y * width + (x - 1)]
                + 2.0 * luma[y * width + (x + 1)]
                - luma[(y + 1) * width + (x - 1)]
                + luma[(y + 1) * width + (x + 1)]) as f64;

            let gy = (-luma[(y - 1) * width + (x - 1)]
                - 2.0 * luma[(y - 1) * width + x]
                - luma[(y - 1) * width + (x + 1)]
                + luma[(y + 1) * width + (x - 1)]
                + 2.0 * luma[(y + 1) * width + x]
                + luma[(y + 1) * width + (x + 1)]) as f64;

            total_gradient += (gx * gx + gy * gy).sqrt();
        }
    }

    if checked_pixels == 0 {
        return 0.0;
    }

    // Maximum possible gradient per pixel is 4 * 255 * sqrt(2) ≈ 1442.2
    let max_gradient = 4.0 * 255.0 * std::f64::consts::SQRT_2;
    (total_gradient / (checked_pixels as f64 * max_gradient)) as f32
}

// ============================================================================
// ConstantQualityConfig
// ============================================================================

/// Configuration for constant-quality GIF encoding.
///
/// The quality parameter drives all encoder decisions: colour depth, dithering,
/// and quantisation algorithm.  Content-adaptive overrides can be applied by
/// calling [`ConstantQualityGifEncoder::resolve_with_metrics`].
#[derive(Clone, Debug)]
pub struct ConstantQualityConfig {
    /// Target quality in `[0.0, 1.0]` (0 = smallest file, 1 = best fidelity).
    pub quality: f32,
    /// When `true`, run perceptual analysis and adapt parameters to image content.
    pub content_adaptive: bool,
    /// Minimum palette size (overrides quality-derived minimum).
    pub min_colors: usize,
    /// Maximum palette size (overrides quality-derived maximum).
    pub max_colors: usize,
    /// Animation loop count (0 = infinite).
    pub loop_count: u16,
}

impl ConstantQualityConfig {
    /// Create a new constant-quality configuration.
    ///
    /// # Panics
    ///
    /// Does not panic; invalid quality values produce a validation error in
    /// [`ConstantQualityGifEncoder::new`].
    pub fn new(quality: f32) -> Self {
        Self {
            quality,
            content_adaptive: true,
            min_colors: 2,
            max_colors: 256,
            loop_count: 0,
        }
    }

    /// Disable content-adaptive analysis (fixed parameters from quality alone).
    pub fn non_adaptive(mut self) -> Self {
        self.content_adaptive = false;
        self
    }

    /// Override minimum colour count.
    pub fn with_min_colors(mut self, min: usize) -> Self {
        self.min_colors = min.clamp(2, 256);
        self
    }

    /// Override maximum colour count.
    pub fn with_max_colors(mut self, max: usize) -> Self {
        self.max_colors = max.clamp(2, 256);
        self
    }
}

// ============================================================================
// ResolvedGifParams — the encoder parameter set
// ============================================================================

/// Fully resolved GIF encoder parameters derived from a [`ConstantQualityConfig`].
///
/// This is a thin wrapper over [`GifEncoderConfig`] that exposes the quality
/// tier and perceptual metrics used to derive the parameters.
#[derive(Clone, Debug)]
pub struct ResolvedGifParams {
    /// The GIF encoder configuration ready for use.
    pub config: GifEncoderConfig,
    /// Quality tier that produced this configuration.
    pub tier: QualityTier,
    /// Content-complexity score (0 = flat, 1 = highly complex), if analysis ran.
    pub complexity_score: Option<f32>,
}

impl ResolvedGifParams {
    /// Returns the number of colours in the resolved palette.
    pub fn colors(&self) -> usize {
        self.config.colors
    }

    /// Returns the dithering method.
    pub fn dithering(&self) -> DitheringMethod {
        self.config.dithering
    }

    /// Returns the quantisation algorithm.
    pub fn quantization(&self) -> QuantizationMethod {
        self.config.quantization
    }
}

// ============================================================================
// ConstantQualityGifEncoder
// ============================================================================

/// Constant-quality GIF encoder.
///
/// Translates a floating-point quality value into concrete [`GifEncoderConfig`]
/// parameters using a combination of quality tiers and, optionally, content-
/// adaptive perceptual analysis.
///
/// # Example
///
/// ```
/// use oximedia_codec::gif::quality::{ConstantQualityConfig, ConstantQualityGifEncoder};
///
/// let cfg = ConstantQualityConfig::new(0.8).non_adaptive();
/// let enc = ConstantQualityGifEncoder::new(cfg).expect("valid quality");
/// let params = enc.resolve();
/// assert!(params.colors() >= 128);
/// ```
#[derive(Clone, Debug)]
pub struct ConstantQualityGifEncoder {
    config: ConstantQualityConfig,
    tier: QualityTier,
}

impl ConstantQualityGifEncoder {
    /// Create a new constant-quality GIF encoder.
    ///
    /// # Errors
    ///
    /// Returns `CodecError::InvalidParameter` when `quality` is outside `[0.0, 1.0]`
    /// or `min_colors` > `max_colors`.
    pub fn new(config: ConstantQualityConfig) -> CodecResult<Self> {
        let tier = QualityTier::from_quality(config.quality)?;
        if config.min_colors > config.max_colors {
            return Err(CodecError::InvalidParameter(format!(
                "min_colors ({}) > max_colors ({})",
                config.min_colors, config.max_colors
            )));
        }
        Ok(Self { config, tier })
    }

    /// Resolve encoder parameters from quality alone (no content analysis).
    pub fn resolve(&self) -> ResolvedGifParams {
        let (colors, dithering, quantization) = self.derive_params(self.config.quality, None);
        ResolvedGifParams {
            config: GifEncoderConfig {
                colors,
                dithering,
                quantization,
                transparent_index: None,
                loop_count: self.config.loop_count,
            },
            tier: self.tier,
            complexity_score: None,
        }
    }

    /// Resolve encoder parameters using pre-computed [`PerceptualMetrics`].
    ///
    /// When `config.content_adaptive` is `false` this is identical to [`Self::resolve`].
    pub fn resolve_with_metrics(&self, metrics: &PerceptualMetrics) -> ResolvedGifParams {
        if !self.config.content_adaptive {
            return self.resolve();
        }

        let score = metrics.complexity_score();
        // Blend base quality with content complexity
        let effective_quality = (self.config.quality * 0.7 + score * 0.3).clamp(0.0, 1.0);
        let (colors, dithering, quantization) =
            self.derive_params(effective_quality, Some(metrics));

        ResolvedGifParams {
            config: GifEncoderConfig {
                colors,
                dithering,
                quantization,
                transparent_index: None,
                loop_count: self.config.loop_count,
            },
            tier: self.tier,
            complexity_score: Some(score),
        }
    }

    /// Analyse an RGBA pixel buffer and resolve parameters content-adaptively.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`PerceptualMetrics::analyse`].
    pub fn analyse_and_resolve(
        &self,
        rgba: &[u8],
        width: usize,
        height: usize,
    ) -> CodecResult<ResolvedGifParams> {
        if !self.config.content_adaptive {
            return Ok(self.resolve());
        }
        let metrics = PerceptualMetrics::analyse(rgba, width, height)?;
        Ok(self.resolve_with_metrics(&metrics))
    }

    /// Quality tier this encoder was created with.
    pub fn tier(&self) -> QualityTier {
        self.tier
    }

    // ── Internal helpers ────────────────────────────────────────────────────

    /// Derive (colors, dithering, quantization) from an effective quality value.
    fn derive_params(
        &self,
        quality: f32,
        metrics: Option<&PerceptualMetrics>,
    ) -> (usize, DitheringMethod, QuantizationMethod) {
        // ── Color count ──────────────────────────────────────────────────────
        // Map quality linearly into [min_colors, max_colors], rounded to nearest
        // power of two in [2, 256] to comply with GIF colour-table restrictions.
        let raw_colors = self.config.min_colors as f32
            + quality * (self.config.max_colors - self.config.min_colors) as f32;
        let colors = clamp_to_valid_color_count(raw_colors as usize);

        // ── Dithering ────────────────────────────────────────────────────────
        let dithering = if quality < 0.25 {
            DitheringMethod::None
        } else if quality < 0.60 {
            // Prefer ordered dithering for medium quality (less banding)
            // but switch to Floyd-Steinberg when content is highly chromatic.
            match metrics {
                Some(m) if m.chromatic_complexity > 0.5 => DitheringMethod::FloydSteinberg,
                _ => DitheringMethod::Ordered,
            }
        } else {
            DitheringMethod::FloydSteinberg
        };

        // ── Quantization ─────────────────────────────────────────────────────
        // High quality (≥ 0.667): octree preserves local colour clusters better.
        // Medium/low quality: median-cut is faster and sufficient.
        let quantization = if quality >= 0.667 {
            QuantizationMethod::Octree
        } else {
            QuantizationMethod::MedianCut
        };

        (colors, dithering, quantization)
    }
}

/// Round a colour count to the nearest valid GIF power-of-two palette size.
///
/// GIF colour tables must have `2^(n+1)` entries where `n ∈ [0, 7]`, giving
/// sizes: 2, 4, 8, 16, 32, 64, 128, 256.
///
/// Values are rounded DOWN to the nearest valid size (conservative — prefer
/// fewer colours over more at any given quality level).
fn clamp_to_valid_color_count(n: usize) -> usize {
    const VALID: [usize; 8] = [2, 4, 8, 16, 32, 64, 128, 256];
    let clamped = n.clamp(2, 256);
    // Find the largest valid size that is <= clamped (round down).
    let mut best = 2usize;
    for &v in &VALID {
        if v <= clamped {
            best = v;
        } else {
            break;
        }
    }
    best
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── QualityTier ──────────────────────────────────────────────────────────

    #[test]
    fn test_quality_tier_low() {
        assert_eq!(QualityTier::from_quality(0.0).unwrap(), QualityTier::Low);
        assert_eq!(QualityTier::from_quality(0.1).unwrap(), QualityTier::Low);
        assert_eq!(QualityTier::from_quality(0.333).unwrap(), QualityTier::Low);
    }

    #[test]
    fn test_quality_tier_medium() {
        assert_eq!(QualityTier::from_quality(0.5).unwrap(), QualityTier::Medium);
        assert_eq!(
            QualityTier::from_quality(0.334).unwrap(),
            QualityTier::Medium
        );
    }

    #[test]
    fn test_quality_tier_high() {
        assert_eq!(QualityTier::from_quality(0.8).unwrap(), QualityTier::High);
        assert_eq!(QualityTier::from_quality(1.0).unwrap(), QualityTier::High);
    }

    #[test]
    fn test_quality_tier_invalid() {
        assert!(QualityTier::from_quality(-0.1).is_err());
        assert!(QualityTier::from_quality(1.1).is_err());
        assert!(QualityTier::from_quality(f32::NAN).is_err());
    }

    #[test]
    fn test_quality_tier_description() {
        assert!(!QualityTier::Low.description().is_empty());
        assert!(!QualityTier::Medium.description().is_empty());
        assert!(!QualityTier::High.description().is_empty());
    }

    // ── clamp_to_valid_color_count ───────────────────────────────────────────

    #[test]
    fn test_clamp_to_valid_color_count() {
        // Values below minimum clamp to 2
        assert_eq!(clamp_to_valid_color_count(1), 2);
        // Round DOWN to nearest valid size
        assert_eq!(clamp_to_valid_color_count(3), 2);
        assert_eq!(clamp_to_valid_color_count(9), 8);
        assert_eq!(clamp_to_valid_color_count(64), 64);
        assert_eq!(clamp_to_valid_color_count(200), 128);
        assert_eq!(clamp_to_valid_color_count(256), 256);
        // Exact matches
        assert_eq!(clamp_to_valid_color_count(128), 128);
        assert_eq!(clamp_to_valid_color_count(16), 16);
    }

    // ── ConstantQualityGifEncoder::resolve ───────────────────────────────────

    #[test]
    fn test_resolve_low_quality() {
        let cfg = ConstantQualityConfig::new(0.1).non_adaptive();
        let enc = ConstantQualityGifEncoder::new(cfg).unwrap();
        let params = enc.resolve();

        assert!(params.colors() <= 16, "low quality should have few colors");
        assert_eq!(params.dithering(), DitheringMethod::None);
        assert_eq!(params.tier, QualityTier::Low);
        assert!(params.complexity_score.is_none());
    }

    #[test]
    fn test_resolve_high_quality() {
        let cfg = ConstantQualityConfig::new(0.9).non_adaptive();
        let enc = ConstantQualityGifEncoder::new(cfg).unwrap();
        let params = enc.resolve();

        assert!(
            params.colors() >= 128,
            "high quality should have many colors"
        );
        assert_eq!(params.dithering(), DitheringMethod::FloydSteinberg);
        assert_eq!(params.quantization(), QuantizationMethod::Octree);
        assert_eq!(params.tier, QualityTier::High);
    }

    #[test]
    fn test_resolve_medium_quality() {
        let cfg = ConstantQualityConfig::new(0.5).non_adaptive();
        let enc = ConstantQualityGifEncoder::new(cfg).unwrap();
        let params = enc.resolve();

        assert!(params.colors() >= 64 && params.colors() <= 128);
        assert_eq!(params.tier, QualityTier::Medium);
    }

    #[test]
    fn test_invalid_quality_rejected() {
        assert!(ConstantQualityGifEncoder::new(ConstantQualityConfig::new(1.5)).is_err());
        assert!(ConstantQualityGifEncoder::new(ConstantQualityConfig::new(-0.1)).is_err());
    }

    #[test]
    fn test_min_colors_gt_max_colors_rejected() {
        let cfg = ConstantQualityConfig::new(0.5)
            .with_min_colors(200)
            .with_max_colors(64);
        assert!(ConstantQualityGifEncoder::new(cfg).is_err());
    }

    // ── PerceptualMetrics ────────────────────────────────────────────────────

    #[test]
    fn test_perceptual_metrics_flat_image() {
        // All pixels identical → zero complexity
        let rgba: Vec<u8> = vec![128u8, 64, 32, 255].repeat(8 * 8);
        let m = PerceptualMetrics::analyse(&rgba, 8, 8).unwrap();

        assert!(
            m.edge_density < 0.01,
            "flat image should have near-zero edge density"
        );
        assert!(m.complexity_score() < 0.4);
    }

    #[test]
    fn test_perceptual_metrics_wrong_size() {
        let rgba = vec![0u8; 10]; // wrong size for 2×2
        assert!(PerceptualMetrics::analyse(&rgba, 2, 2).is_err());
    }

    #[test]
    fn test_perceptual_metrics_zero_size() {
        let m = PerceptualMetrics::analyse(&[], 0, 0).unwrap();
        assert_eq!(m.complexity_score(), 0.0);
    }

    // ── Content-adaptive path ────────────────────────────────────────────────

    #[test]
    fn test_analyse_and_resolve_returns_ok() {
        // 4×4 solid blue image — simple content
        let rgba: Vec<u8> = vec![0u8, 0, 255, 255].repeat(4 * 4);
        let cfg = ConstantQualityConfig::new(0.7);
        let enc = ConstantQualityGifEncoder::new(cfg).unwrap();
        let params = enc.analyse_and_resolve(&rgba, 4, 4).unwrap();

        assert!(params.complexity_score.is_some());
        assert!(params.colors() >= 2);
    }

    #[test]
    fn test_analyse_and_resolve_nonadaptive_ignores_content() {
        let rgba: Vec<u8> = vec![255u8, 0, 0, 255].repeat(4 * 4);
        let cfg = ConstantQualityConfig::new(0.6).non_adaptive();
        let enc = ConstantQualityGifEncoder::new(cfg).unwrap();
        let params = enc.analyse_and_resolve(&rgba, 4, 4).unwrap();

        assert!(params.complexity_score.is_none());
    }

    #[test]
    fn test_loop_count_propagated() {
        let mut cfg = ConstantQualityConfig::new(0.5);
        cfg.loop_count = 3;
        let enc = ConstantQualityGifEncoder::new(cfg).unwrap();
        let params = enc.resolve();
        assert_eq!(params.config.loop_count, 3);
    }
}
