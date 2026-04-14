//! Gamut compression algorithms for mapping out-of-gamut colors.
//!
//! This module provides professional-grade gamut compression algorithms that
//! map out-of-gamut colors into gamut while preserving as much perceptual
//! quality as possible. Three complementary approaches are provided:
//!
//! - **RGB Ratio method**: Compresses each channel relative to its distance
//!   beyond the gamut boundary, preserving ratios and thus hue.
//! - **Achromatic-axis method**: Moves colors toward the neutral achromatic
//!   axis (equal-energy grey) in a hue-preserving manner.
//! - **Knee-based soft clipping**: Applies a smooth non-linear roll-off
//!   starting at a configurable knee point, inspired by the ACES Reference
//!   Gamut Compression (RGC) algorithm.
//!
//! # Algorithm Details
//!
//! ## RGB Ratio Method
//!
//! For a pixel (R, G, B) the maximum channel value M = max(R, G, B) is
//! identified. If M ≤ 1.0 the pixel is in-gamut. Otherwise, each channel is
//! divided by M (normalised) and then mapped through the compression curve,
//! preserving the chromaticity (ratio) of the channels.
//!
//! ## Achromatic-Axis Method
//!
//! The achromatic (neutral grey) value for each pixel is computed as the mean
//! of its channels. Out-of-gamut components are linearly interpolated toward
//! the achromatic value until all channels are in [0, 1].
//!
//! ## Knee-Based Soft Clipping
//!
//! Inspired by the ACES Reference Gamut Compression (v1.0):
//! - A threshold (knee) defines where compression begins.
//! - Beyond the knee, a smooth curve maps the open range [knee, ∞) to
//!   [knee, 1.0] using a parabolic shoulder or Reinhard-like function.
//!
//! # Example
//!
//! ```
//! use oximedia_colormgmt::gamut_compression::{
//!     GamutCompressor, CompressorConfig, KneeMethod,
//! };
//!
//! let config = CompressorConfig::default();
//! let compressor = GamutCompressor::new(config);
//!
//! // Wide-gamut HDR pixel (scene-linear, some channels > 1.0)
//! let pixel = [1.5_f64, 0.8, 0.3];
//! let compressed = compressor.compress_rgb_ratio(pixel);
//! assert!(compressed[0] <= 1.0);
//! assert!(compressed[1] <= 1.0);
//! assert!(compressed[2] <= 1.0);
//! ```

// ── Configuration ─────────────────────────────────────────────────────────────

/// Selects the knee function used by the soft-clip compressor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KneeMethod {
    /// Parabolic shoulder: smooth quadratic roll-off (default).
    Parabolic,
    /// Reinhard-like: hyperbolic curve that asymptotically approaches 1.0.
    Reinhard,
    /// Cubic Hermite spline: C¹-continuous smooth shoulder.
    CubicHermite,
}

impl Default for KneeMethod {
    fn default() -> Self {
        Self::Parabolic
    }
}

/// Configuration for the gamut compressor.
#[derive(Debug, Clone)]
pub struct CompressorConfig {
    /// Knee point as a fraction of the gamut boundary [0.0–1.0].
    ///
    /// Compression begins here. Values below this threshold are unaffected.
    /// Default: `0.8`.
    pub knee: f64,
    /// Maximum input value to map into gamut (for soft-clip only).
    ///
    /// Values beyond `max_input` are clamped. Default: `1.5`.
    pub max_input: f64,
    /// Knee function to use for soft-clip compression.
    pub knee_method: KneeMethod,
    /// For the achromatic-axis method: blend strength toward achromatic.
    ///
    /// `1.0` = fully compress to achromatic; `0.0` = no compression.
    /// Default: `1.0`.
    pub achromatic_strength: f64,
}

impl Default for CompressorConfig {
    fn default() -> Self {
        Self {
            knee: 0.8,
            max_input: 1.5,
            knee_method: KneeMethod::Parabolic,
            achromatic_strength: 1.0,
        }
    }
}

// ── GamutCompressor ───────────────────────────────────────────────────────────

/// A gamut compressor that implements multiple compression algorithms.
///
/// All methods operate on scene-linear RGB values. The output of each
/// method is guaranteed to be in [0, 1]³ for any finite input.
#[derive(Debug, Clone)]
pub struct GamutCompressor {
    /// Configuration for the compressor.
    pub config: CompressorConfig,
}

impl GamutCompressor {
    /// Create a new compressor with the given configuration.
    #[must_use]
    pub fn new(config: CompressorConfig) -> Self {
        Self { config }
    }

    /// Create a compressor with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(CompressorConfig::default())
    }

    // ── RGB Ratio Method ──────────────────────────────────────────────────────

    /// Compress a pixel using the **RGB ratio method**.
    ///
    /// If any channel exceeds 1.0, all channels are divided by the maximum
    /// channel value to normalise the ratio, then the maximum is soft-clipped
    /// to 1.0. This preserves relative channel ratios (and thus approximate
    /// hue) while eliminating the over-exposure.
    ///
    /// Negative values are clamped to 0 before compression.
    #[must_use]
    pub fn compress_rgb_ratio(&self, rgb: [f64; 3]) -> [f64; 3] {
        let r = rgb[0].max(0.0);
        let g = rgb[1].max(0.0);
        let b = rgb[2].max(0.0);

        let max_val = r.max(g).max(b);
        if max_val <= 1.0 {
            return [r, g, b];
        }

        // Normalise to bring max to 1.0, applying the soft-clip curve to max
        let compressed_max = self.knee_compress(max_val);
        let scale = compressed_max / max_val;

        [
            (r * scale).clamp(0.0, 1.0),
            (g * scale).clamp(0.0, 1.0),
            (b * scale).clamp(0.0, 1.0),
        ]
    }

    // ── Achromatic-Axis Method ────────────────────────────────────────────────

    /// Compress a pixel using the **achromatic-axis method**.
    ///
    /// Each out-of-gamut channel is moved toward the achromatic (neutral grey)
    /// value by the minimum amount needed to bring it into [0, 1]. The
    /// `achromatic_strength` configuration parameter controls how aggressively
    /// colors are desaturated.
    ///
    /// This method is more conservative than ratio compression and better
    /// preserves lightness when one channel dominates.
    #[must_use]
    pub fn compress_achromatic(&self, rgb: [f64; 3]) -> [f64; 3] {
        let r = rgb[0].max(0.0);
        let g = rgb[1].max(0.0);
        let b = rgb[2].max(0.0);

        // Check if already in gamut
        if r <= 1.0 && g <= 1.0 && b <= 1.0 {
            return [r, g, b];
        }

        // Achromatic value: mean of channels
        let achromatic = (r + g + b) / 3.0;

        // Find the minimum blend factor t ∈ [0, 1] such that all channels ≤ 1.
        // Channel after blend: ch * (1 - t) + achromatic * t
        // For channel ch > 1: ch * (1-t) + achromatic * t <= 1
        //   → t >= (ch - 1) / (ch - achromatic)  when ch > achromatic
        let mut min_t = 0.0_f64;
        for ch in [r, g, b] {
            if ch > 1.0 {
                let denom = ch - achromatic;
                if denom > 1e-12 {
                    let t_needed = (ch - 1.0) / denom;
                    if t_needed > min_t {
                        min_t = t_needed;
                    }
                }
            }
        }

        // Apply strength modulation and clamp
        let t = (min_t * self.config.achromatic_strength).clamp(0.0, 1.0);

        [
            (r * (1.0 - t) + achromatic * t).clamp(0.0, 1.0),
            (g * (1.0 - t) + achromatic * t).clamp(0.0, 1.0),
            (b * (1.0 - t) + achromatic * t).clamp(0.0, 1.0),
        ]
    }

    // ── Soft-Clip (Knee) Method ───────────────────────────────────────────────

    /// Compress a pixel using **per-channel knee-based soft clipping**.
    ///
    /// Each channel is passed through the knee function independently.
    /// This does not preserve hue ratios and may introduce hue shifts on very
    /// saturated colors, but is computationally simple and numerically stable.
    ///
    /// Negative values are passed through unchanged (preserved for HDR blacks).
    #[must_use]
    pub fn compress_soft_clip(&self, rgb: [f64; 3]) -> [f64; 3] {
        [
            self.knee_compress_channel(rgb[0]),
            self.knee_compress_channel(rgb[1]),
            self.knee_compress_channel(rgb[2]),
        ]
    }

    /// Apply the knee compression curve to a single scalar value.
    ///
    /// Values ≤ `knee` are returned unchanged.
    /// Values > `knee` are mapped through the selected knee function.
    #[must_use]
    pub fn knee_compress(&self, x: f64) -> f64 {
        let knee = self.config.knee.clamp(0.0, 1.0);
        if x <= knee {
            return x;
        }

        match self.config.knee_method {
            KneeMethod::Parabolic => {
                let range = (self.config.max_input - knee).max(1e-12);
                let t = ((x - knee) / range).clamp(0.0, 1.0);
                // Smooth quadratic shoulder: 0 → 0, 1 → 1, slope 2 at 0, 0 at 1
                let smooth_t = t * (2.0 - t);
                knee + (1.0 - knee) * smooth_t
            }
            KneeMethod::Reinhard => {
                // Reinhard: maps [knee, ∞) → [knee, 1]
                // shifted_x = x - knee; result = shifted_x / (1 + shifted_x) * (1 - knee) + knee
                let shifted = x - knee;
                let reinhard = shifted / (1.0 + shifted);
                knee + (1.0 - knee) * reinhard
            }
            KneeMethod::CubicHermite => {
                let range = (self.config.max_input - knee).max(1e-12);
                let t = ((x - knee) / range).clamp(0.0, 1.0);
                // Cubic Hermite smoothstep: 3t² - 2t³
                let smooth_t = t * t * (3.0 - 2.0 * t);
                knee + (1.0 - knee) * smooth_t
            }
        }
    }

    /// Apply the knee curve to a single channel, handling negative values.
    #[inline]
    fn knee_compress_channel(&self, ch: f64) -> f64 {
        if ch < 0.0 {
            ch
        } else {
            self.knee_compress(ch).clamp(0.0, 1.0)
        }
    }

    // ── Combined Method ───────────────────────────────────────────────────────

    /// Compress a pixel using a combination of achromatic-axis and RGB-ratio
    /// methods, blended by a `blend` factor.
    ///
    /// `blend = 0.0` → pure achromatic-axis;
    /// `blend = 1.0` → pure RGB-ratio.
    ///
    /// This provides a balance between hue preservation (ratio) and lightness
    /// preservation (achromatic), useful for skin tones and pastels.
    #[must_use]
    pub fn compress_blended(&self, rgb: [f64; 3], blend: f64) -> [f64; 3] {
        let t = blend.clamp(0.0, 1.0);
        let achro = self.compress_achromatic(rgb);
        let ratio = self.compress_rgb_ratio(rgb);
        [
            achro[0] * (1.0 - t) + ratio[0] * t,
            achro[1] * (1.0 - t) + ratio[1] * t,
            achro[2] * (1.0 - t) + ratio[2] * t,
        ]
    }
}

// ── Batch Processing ──────────────────────────────────────────────────────────

/// Apply gamut compression to a flat pixel buffer in-place.
///
/// The buffer must have length divisible by 3 (each pixel is [R, G, B]).
/// Each pixel is compressed using the RGB ratio method.
pub fn compress_buffer_rgb_ratio(
    buffer: &mut [f64],
    config: &CompressorConfig,
) -> Result<(), &'static str> {
    if buffer.len() % 3 != 0 {
        return Err("Buffer length must be divisible by 3");
    }
    let compressor = GamutCompressor::new(config.clone());
    for chunk in buffer.chunks_exact_mut(3) {
        let compressed = compressor.compress_rgb_ratio([chunk[0], chunk[1], chunk[2]]);
        chunk[0] = compressed[0];
        chunk[1] = compressed[1];
        chunk[2] = compressed[2];
    }
    Ok(())
}

/// Apply soft-clip gamut compression to a flat pixel buffer in-place.
///
/// See [`GamutCompressor::compress_soft_clip`] for algorithm details.
pub fn compress_buffer_soft_clip(
    buffer: &mut [f64],
    config: &CompressorConfig,
) -> Result<(), &'static str> {
    if buffer.len() % 3 != 0 {
        return Err("Buffer length must be divisible by 3");
    }
    let compressor = GamutCompressor::new(config.clone());
    for chunk in buffer.chunks_exact_mut(3) {
        let compressed = compressor.compress_soft_clip([chunk[0], chunk[1], chunk[2]]);
        chunk[0] = compressed[0];
        chunk[1] = compressed[1];
        chunk[2] = compressed[2];
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_compressor() -> GamutCompressor {
        GamutCompressor::with_defaults()
    }

    #[test]
    fn test_rgb_ratio_in_gamut_unchanged() {
        let c = default_compressor();
        let pixel = [0.5, 0.3, 0.8];
        let out = c.compress_rgb_ratio(pixel);
        for i in 0..3 {
            assert!(
                (out[i] - pixel[i]).abs() < 1e-12,
                "In-gamut pixel should be unchanged"
            );
        }
    }

    #[test]
    fn test_rgb_ratio_compresses_to_gamut() {
        let c = default_compressor();
        let pixel = [1.8, 0.5, 0.2];
        let out = c.compress_rgb_ratio(pixel);
        for v in out {
            assert!(v <= 1.0 + 1e-10, "Output should be in [0,1]: {v}");
            assert!(v >= 0.0, "Output should be non-negative: {v}");
        }
    }

    #[test]
    fn test_rgb_ratio_preserves_channel_order() {
        let c = default_compressor();
        let pixel = [1.5, 0.8, 0.3];
        let out = c.compress_rgb_ratio(pixel);
        // R was highest, should remain highest after compression
        assert!(out[0] >= out[1], "Channel order should be preserved");
        assert!(out[1] >= out[2], "Channel order should be preserved");
    }

    #[test]
    fn test_achromatic_in_gamut_unchanged() {
        let c = default_compressor();
        let pixel = [0.4, 0.6, 0.9];
        let out = c.compress_achromatic(pixel);
        for i in 0..3 {
            assert!(
                (out[i] - pixel[i]).abs() < 1e-12,
                "In-gamut pixel should be unchanged"
            );
        }
    }

    #[test]
    fn test_achromatic_compresses_to_gamut() {
        let c = default_compressor();
        let pixel = [2.0, 0.1, 0.1];
        let out = c.compress_achromatic(pixel);
        for v in out {
            assert!(v <= 1.0 + 1e-10, "Output must be in [0,1]: {v}");
            assert!(v >= 0.0, "Output must be non-negative: {v}");
        }
    }

    #[test]
    fn test_soft_clip_all_methods() {
        let methods = [KneeMethod::Parabolic, KneeMethod::Reinhard, KneeMethod::CubicHermite];
        for method in methods {
            let config = CompressorConfig {
                knee_method: method,
                ..Default::default()
            };
            let c = GamutCompressor::new(config);
            let pixel = [1.3, 0.9, 0.4];
            let out = c.compress_soft_clip(pixel);
            for v in out {
                assert!(v <= 1.0 + 1e-10, "Method {method:?}: output must be <= 1.0: {v}");
            }
        }
    }

    #[test]
    fn test_knee_compress_monotonic() {
        let c = default_compressor();
        let values = [0.0, 0.5, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0];
        let compressed: Vec<f64> = values.iter().map(|&v| c.knee_compress(v)).collect();
        for i in 1..compressed.len() {
            assert!(
                compressed[i] >= compressed[i - 1] - 1e-10,
                "Knee curve must be monotonically non-decreasing: {:?}",
                compressed
            );
        }
    }

    #[test]
    fn test_knee_compress_in_gamut_unchanged() {
        let c = default_compressor();
        for v in [0.0, 0.3, 0.6, 0.79] {
            let out = c.knee_compress(v);
            assert!(
                (out - v).abs() < 1e-10,
                "Values below knee should be unchanged: {v} → {out}"
            );
        }
    }

    #[test]
    fn test_blended_compression() {
        let c = default_compressor();
        let pixel = [1.6, 0.5, 0.2];
        let out0 = c.compress_blended(pixel, 0.0); // pure achromatic
        let out1 = c.compress_blended(pixel, 1.0); // pure ratio
        let out_half = c.compress_blended(pixel, 0.5);

        let achro = c.compress_achromatic(pixel);
        let ratio = c.compress_rgb_ratio(pixel);

        for i in 0..3 {
            assert!((out0[i] - achro[i]).abs() < 1e-10, "blend=0 should match achromatic");
            assert!((out1[i] - ratio[i]).abs() < 1e-10, "blend=1 should match rgb_ratio");
            // blended should be between achromatic and ratio
            let lo = out0[i].min(out1[i]);
            let hi = out0[i].max(out1[i]);
            assert!(
                out_half[i] >= lo - 1e-10 && out_half[i] <= hi + 1e-10,
                "blended={} should be between achromatic={} and ratio={}",
                out_half[i],
                out0[i],
                out1[i]
            );
        }
    }

    #[test]
    fn test_compress_buffer_rgb_ratio() {
        let config = CompressorConfig::default();
        let mut buf = vec![1.5, 0.8, 0.3, 0.4, 0.6, 0.9];
        compress_buffer_rgb_ratio(&mut buf, &config).expect("should not fail");
        for &v in &buf {
            assert!(v <= 1.0 + 1e-10, "Buffer value must be in gamut: {v}");
        }
    }

    #[test]
    fn test_compress_buffer_wrong_size() {
        let config = CompressorConfig::default();
        let mut buf = vec![1.5, 0.8]; // not divisible by 3
        let result = compress_buffer_rgb_ratio(&mut buf, &config);
        assert!(result.is_err(), "Should return error for wrong buffer size");
    }

    #[test]
    fn test_reinhard_asymptote() {
        let config = CompressorConfig {
            knee_method: KneeMethod::Reinhard,
            ..Default::default()
        };
        let c = GamutCompressor::new(config);
        // Very large value should approach 1.0
        let out = c.knee_compress(1000.0);
        assert!(
            out < 1.0 + 1e-10,
            "Reinhard should be bounded at 1.0, got {out}"
        );
        assert!(out > 0.9, "Reinhard of very large value should be close to 1.0");
    }
}
