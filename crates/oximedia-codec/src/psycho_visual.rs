//! Psycho-visual masking model for perceptual rate control.
//!
//! This module implements a visibility threshold model for per-block
//! quantization decisions. The human visual system (HVS) is less sensitive
//! to distortion in:
//!
//! - **High-texture regions** (spatial masking)
//! - **Bright regions** (luminance masking — Weber's law)
//! - **High temporal activity** (temporal masking)
//! - **Near strong edges** (edge masking)
//!
//! The combined Just-Noticeable Difference (JND) threshold drives a QP delta
//! that can be added to the base quantisation parameter. A positive delta means
//! the block can tolerate more distortion (higher QP); negative means it needs
//! finer coding (lower QP).
//!
//! # Reference
//!
//! Watson, A.B. (1993). *DCTune: A technique for visual optimization of DCT
//! quantization matrices.* Society for Information Display Digest of Technical
//! Papers, 24, 946–949.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the psycho-visual masking model.
#[derive(Debug, Clone)]
pub struct PsychoVisualConfig {
    /// Enable spatial (texture) masking.  Default: `true`.
    pub spatial_masking: bool,
    /// Enable luminance masking.  Default: `true`.
    pub luminance_masking: bool,
    /// Enable temporal masking (requires SAD from motion estimation).  Default: `true`.
    pub temporal_masking: bool,
    /// Enable edge masking.  Default: `true`.
    pub edge_masking: bool,
    /// Overall strength multiplier applied to the JND-derived QP delta.
    /// Range [0.0, 2.0]; default `1.0`.
    pub strength: f32,
    /// Maximum allowed positive QP delta (soften a block at most this much).
    pub max_qp_delta_pos: i32,
    /// Maximum allowed negative QP delta (sharpen a block at most this much).
    pub max_qp_delta_neg: i32,
}

impl Default for PsychoVisualConfig {
    fn default() -> Self {
        Self {
            spatial_masking: true,
            luminance_masking: true,
            temporal_masking: true,
            edge_masking: true,
            strength: 1.0,
            max_qp_delta_pos: 6,
            max_qp_delta_neg: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Block statistics fed into the masking model
// ---------------------------------------------------------------------------

/// Per-block statistics gathered during analysis (before encoding).
#[derive(Debug, Clone, Default)]
pub struct BlockMaskStats {
    /// Mean luma value of the block (0–255).
    pub mean_luma: f32,
    /// Spatial variance of luma values (proxy for texture energy).
    pub luma_variance: f32,
    /// Sum of absolute differences from motion estimation (temporal activity).
    pub motion_sad: f32,
    /// Edge energy (e.g. Sobel gradient magnitude sum over the block).
    pub edge_energy: f32,
    /// Number of pixels in the block.
    pub block_area: u32,
}

impl BlockMaskStats {
    /// Create stats from a raw luma block slice.
    ///
    /// `pixels` should be a row-major luma block (e.g. 8×8 = 64 elements, 0–255).
    #[must_use]
    pub fn from_luma_block(pixels: &[u8]) -> Self {
        if pixels.is_empty() {
            return Self::default();
        }
        let n = pixels.len() as f32;
        let mean = pixels.iter().map(|&p| p as f32).sum::<f32>() / n;
        let variance = pixels
            .iter()
            .map(|&p| {
                let d = p as f32 - mean;
                d * d
            })
            .sum::<f32>()
            / n;

        // Approximate edge energy: sum of horizontal and vertical absolute differences
        let side = (n.sqrt().round() as usize).max(1);
        let mut edge_energy = 0.0f32;
        for row in 0..side {
            for col in 0..side {
                let idx = row * side + col;
                let right = if col + 1 < side {
                    pixels[idx + 1] as f32
                } else {
                    pixels[idx] as f32
                };
                let down = if row + 1 < side {
                    pixels[idx + side] as f32
                } else {
                    pixels[idx] as f32
                };
                edge_energy +=
                    (pixels[idx] as f32 - right).abs() + (pixels[idx] as f32 - down).abs();
            }
        }

        Self {
            mean_luma: mean,
            luma_variance: variance,
            motion_sad: 0.0,
            edge_energy,
            block_area: pixels.len() as u32,
        }
    }
}

// ---------------------------------------------------------------------------
// Masking model
// ---------------------------------------------------------------------------

/// Psycho-visual masking model that computes per-block QP deltas.
#[derive(Debug, Clone)]
pub struct PsychoVisualModel {
    cfg: PsychoVisualConfig,
}

impl PsychoVisualModel {
    /// Create a new masking model with the given configuration.
    #[must_use]
    pub fn new(cfg: PsychoVisualConfig) -> Self {
        Self { cfg }
    }

    /// Create a masking model with default configuration.
    #[must_use]
    pub fn default_model() -> Self {
        Self::new(PsychoVisualConfig::default())
    }

    /// Compute a QP delta for a single block.
    ///
    /// Positive values allow higher QP (more compression); negative values
    /// force lower QP (higher quality).
    ///
    /// The block statistics should be gathered before calling this function.
    #[must_use]
    pub fn compute_qp_delta(&self, stats: &BlockMaskStats) -> i32 {
        if self.cfg.strength < 1e-6 {
            return 0;
        }

        let mut delta = 0.0f32;

        // --- Spatial / texture masking ---
        if self.cfg.spatial_masking {
            // Blocks with high variance can hide more distortion.
            // Log-scaling keeps the range reasonable.
            let texture_delta = (stats.luma_variance.max(0.0).ln_1p() / 5.0).min(3.0);
            delta += texture_delta;
        }

        // --- Luminance masking (Weber's law) ---
        if self.cfg.luminance_masking {
            // Very bright (>200) and very dark (<32) regions have reduced sensitivity.
            let luma = stats.mean_luma;
            let luma_delta = if luma > 200.0 {
                (luma - 200.0) / 55.0 * 2.0 // bright: up to +2
            } else if luma < 32.0 {
                (32.0 - luma) / 32.0 * 1.5 // dark: up to +1.5
            } else {
                0.0
            };
            delta += luma_delta;
        }

        // --- Temporal masking ---
        if self.cfg.temporal_masking {
            // High SAD areas are in motion; viewers track moving objects, so
            // perceptual sensitivity is *lower* → allow higher QP.
            let area = stats.block_area.max(1) as f32;
            let sad_per_pixel = stats.motion_sad / area;
            let temporal_delta = (sad_per_pixel / 16.0).min(2.0);
            delta += temporal_delta;
        }

        // --- Edge masking ---
        if self.cfg.edge_masking {
            // Near strong edges, texture masking is already captured;
            // but very flat blocks near edges need protection.
            let area = stats.block_area.max(1) as f32;
            let edge_density = stats.edge_energy / area;
            let edge_delta = if edge_density < 2.0 && stats.luma_variance < 10.0 {
                // Flat block near background: protect detail
                -1.5
            } else if edge_density > 20.0 {
                // Dense edge: masking applies
                1.0
            } else {
                0.0
            };
            delta += edge_delta;
        }

        // Apply strength multiplier and clamp
        let scaled = (delta * self.cfg.strength).round() as i32;
        scaled
            .max(-self.cfg.max_qp_delta_neg)
            .min(self.cfg.max_qp_delta_pos)
    }

    /// Build a QP delta map for an entire frame.
    ///
    /// `luma_plane` is the raw luma data (row-major, 8-bit), `width` and `height`
    /// are the frame dimensions.  Blocks are 8×8 pixels.  The returned vector
    /// has one entry per block in raster-scan order.
    #[must_use]
    pub fn compute_frame_qp_map(
        &self,
        luma_plane: &[u8],
        width: usize,
        height: usize,
        motion_sad_map: Option<&[f32]>,
    ) -> Vec<i32> {
        const BLOCK: usize = 8;
        let blocks_x = (width + BLOCK - 1) / BLOCK;
        let blocks_y = (height + BLOCK - 1) / BLOCK;
        let mut map = Vec::with_capacity(blocks_x * blocks_y);

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                // Gather block pixels
                let mut block_pixels: Vec<u8> = Vec::with_capacity(BLOCK * BLOCK);
                for row in 0..BLOCK {
                    let y = by * BLOCK + row;
                    if y >= height {
                        break;
                    }
                    for col in 0..BLOCK {
                        let x = bx * BLOCK + col;
                        if x < width {
                            block_pixels.push(luma_plane[y * width + x]);
                        }
                    }
                }

                let mut stats = BlockMaskStats::from_luma_block(&block_pixels);

                // Optionally fill motion SAD
                if let Some(sad_map) = motion_sad_map {
                    let block_idx = by * blocks_x + bx;
                    if block_idx < sad_map.len() {
                        stats.motion_sad = sad_map[block_idx];
                    }
                }

                map.push(self.compute_qp_delta(&stats));
            }
        }

        map
    }

    /// Return the current configuration.
    #[must_use]
    pub fn config(&self) -> &PsychoVisualConfig {
        &self.cfg
    }
}

// ---------------------------------------------------------------------------
// Visibility threshold lookup table (DCT-frequency domain)
// ---------------------------------------------------------------------------

/// Luminance visibility thresholds per DCT frequency (8×8 block, row-major).
///
/// Based on Watson (1993) Table 1 — luma quantization matrix scaled to
/// produce a minimum JND threshold.  Values are in the same scale as standard
/// JPEG quantization matrices; lower value = more visible frequency.
pub const LUMA_VISIBILITY_THRESHOLDS: [f32; 64] = [
    16.0, 11.0, 10.0, 16.0, 24.0, 40.0, 51.0, 61.0, 12.0, 12.0, 14.0, 19.0, 26.0, 58.0, 60.0, 55.0,
    14.0, 13.0, 16.0, 24.0, 40.0, 57.0, 69.0, 56.0, 14.0, 17.0, 22.0, 29.0, 51.0, 87.0, 80.0, 62.0,
    18.0, 22.0, 37.0, 56.0, 68.0, 109.0, 103.0, 77.0, 24.0, 35.0, 55.0, 64.0, 81.0, 104.0, 113.0,
    92.0, 49.0, 64.0, 78.0, 87.0, 103.0, 121.0, 120.0, 101.0, 72.0, 92.0, 95.0, 98.0, 112.0, 100.0,
    103.0, 99.0,
];

/// Chrominance visibility thresholds per DCT frequency (8×8 block).
pub const CHROMA_VISIBILITY_THRESHOLDS: [f32; 64] = [
    17.0, 18.0, 24.0, 47.0, 99.0, 99.0, 99.0, 99.0, 18.0, 21.0, 26.0, 66.0, 99.0, 99.0, 99.0, 99.0,
    24.0, 26.0, 56.0, 99.0, 99.0, 99.0, 99.0, 99.0, 47.0, 66.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
    99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
    99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
];

/// Scale the standard visibility threshold table by a quality factor.
///
/// `quality` is in [1, 100] as in JPEG; higher quality → smaller threshold
/// divisors → more bits used.
#[must_use]
pub fn scale_thresholds(base: &[f32; 64], quality: u8) -> [f32; 64] {
    let q = quality.clamp(1, 100) as f32;
    let scale = if q < 50.0 {
        5000.0 / q
    } else {
        200.0 - 2.0 * q
    };
    let mut out = [0.0f32; 64];
    for (i, (&b, o)) in base.iter().zip(out.iter_mut()).enumerate() {
        let _ = i; // suppress unused warning
        *o = ((b * scale / 100.0 + 0.5).floor()).clamp(1.0, 255.0);
    }
    out
}

// ---------------------------------------------------------------------------
// PvsModel — simplified psycho-visual model with visibility threshold
// ---------------------------------------------------------------------------

/// Simplified psycho-visual sensitivity (PVS) model.
///
/// Provides a single `visibility_threshold` method that returns the minimum
/// distortion energy that becomes just-noticeable to the human visual system
/// (HVS) for a given luma level and spatial frequency.
///
/// The model is derived from the Watson (1993) DCTune formulation:
/// - Base threshold from the luma quantization table (frequency-dependent).
/// - Luminance masking multiplier: brighter or darker regions tolerate more
///   distortion (Weber-Fechner / power-law masking).
///
/// # Reference
/// Watson, A.B. (1993). *DCTune*.
#[derive(Debug, Clone, Default)]
pub struct PvsModel;

impl PvsModel {
    /// Create a new `PvsModel`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Compute the visibility threshold (JND) for the given luma and frequency.
    ///
    /// # Parameters
    /// - `luma`  – mean luminance of the block, normalised [0.0, 1.0] (0 = black,
    ///             1 = white).
    /// - `freq`  – spatial frequency index in [0.0, 1.0] representing low (0) to
    ///             high (1) frequency DCT coefficients.  Linearly interpolated
    ///             between the DC threshold and the highest AC threshold in the
    ///             JPEG luma quantization matrix.
    ///
    /// # Returns
    /// A positive threshold value.  Distortions below this level are perceptually
    /// invisible; distortions above may be visible.
    #[must_use]
    pub fn visibility_threshold(luma: f32, freq: f32) -> f32 {
        // --- Frequency-dependent base threshold ---
        // Clamp freq to [0, 1] and linearly interpolate across the JPEG luma QT
        // (DC coefficient threshold = 16, highest AC = 121)
        let freq_t = freq.clamp(0.0, 1.0);
        let dc_thresh = LUMA_VISIBILITY_THRESHOLDS[0]; // 16.0
        let ac_max_thresh = 121.0_f32; // max in LUMA_VISIBILITY_THRESHOLDS
        let base = dc_thresh + (ac_max_thresh - dc_thresh) * freq_t;

        // --- Luminance masking multiplier ---
        // Power-law: threshold rises for very dark (<0.1) and very bright (>0.9)
        let luma_c = luma.clamp(0.0, 1.0);
        let masking_mult = if luma_c < 0.1 {
            // Dark region: HVS less sensitive to distortion
            1.0 + (0.1 - luma_c) * 5.0
        } else if luma_c > 0.9 {
            // Bright region: also less sensitive
            1.0 + (luma_c - 0.9) * 5.0
        } else {
            // Mid-range: most sensitive (threshold close to base)
            1.0
        };

        (base * masking_mult).max(1.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_strength() {
        let model = PsychoVisualModel::default_model();
        assert!((model.cfg.strength - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_qp_delta_zero_strength() {
        let mut cfg = PsychoVisualConfig::default();
        cfg.strength = 0.0;
        let model = PsychoVisualModel::new(cfg);
        let stats = BlockMaskStats {
            mean_luma: 200.0,
            luma_variance: 500.0,
            motion_sad: 1000.0,
            edge_energy: 0.0,
            block_area: 64,
        };
        assert_eq!(model.compute_qp_delta(&stats), 0);
    }

    #[test]
    fn test_qp_delta_clamped_positive() {
        let model = PsychoVisualModel::default_model();
        // Very high variance + bright + high motion → positive delta, capped at max
        let stats = BlockMaskStats {
            mean_luma: 255.0,
            luma_variance: 50000.0,
            motion_sad: 10000.0,
            edge_energy: 5000.0,
            block_area: 64,
        };
        let delta = model.compute_qp_delta(&stats);
        assert!(delta <= model.cfg.max_qp_delta_pos);
    }

    #[test]
    fn test_qp_delta_clamped_negative() {
        let model = PsychoVisualModel::default_model();
        // Flat dark block near edge → negative delta (protect quality), capped at -max_neg
        let stats = BlockMaskStats {
            mean_luma: 30.0,
            luma_variance: 0.5,
            motion_sad: 0.0,
            edge_energy: 0.5,
            block_area: 64,
        };
        let delta = model.compute_qp_delta(&stats);
        assert!(delta >= -model.cfg.max_qp_delta_neg);
    }

    #[test]
    fn test_block_stats_from_luma_uniform() {
        let pixels = vec![128u8; 64];
        let stats = BlockMaskStats::from_luma_block(&pixels);
        assert!((stats.mean_luma - 128.0).abs() < 1e-3);
        assert!(stats.luma_variance < 1e-3);
    }

    #[test]
    fn test_block_stats_from_luma_gradient() {
        let pixels: Vec<u8> = (0u8..64).collect();
        let stats = BlockMaskStats::from_luma_block(&pixels);
        // Mean of 0..64 = 31.5
        assert!((stats.mean_luma - 31.5).abs() < 0.5);
        assert!(stats.luma_variance > 100.0);
    }

    #[test]
    fn test_compute_frame_qp_map_dimensions() {
        let model = PsychoVisualModel::default_model();
        let luma = vec![128u8; 64 * 48];
        let map = model.compute_frame_qp_map(&luma, 64, 48, None);
        // 64/8 = 8 blocks wide, 48/8 = 6 blocks tall
        assert_eq!(map.len(), 8 * 6);
    }

    #[test]
    fn test_scale_thresholds_high_quality() {
        // At quality 100 the scale should reduce the thresholds significantly
        let hi = scale_thresholds(&LUMA_VISIBILITY_THRESHOLDS, 100);
        let lo = scale_thresholds(&LUMA_VISIBILITY_THRESHOLDS, 10);
        // All high-quality values should be <= low-quality values
        for (h, l) in hi.iter().zip(lo.iter()) {
            assert!(h <= l, "hi={h} lo={l}");
        }
    }

    #[test]
    fn test_scale_thresholds_min_clamp() {
        // Even at quality 1 the minimum threshold must be 1
        let out = scale_thresholds(&LUMA_VISIBILITY_THRESHOLDS, 1);
        for &v in &out {
            assert!(v >= 1.0);
        }
    }

    #[test]
    fn test_visibility_threshold_tables_length() {
        assert_eq!(LUMA_VISIBILITY_THRESHOLDS.len(), 64);
        assert_eq!(CHROMA_VISIBILITY_THRESHOLDS.len(), 64);
    }

    #[test]
    fn test_qp_map_with_sad_map() {
        let model = PsychoVisualModel::default_model();
        let luma = vec![100u8; 16 * 16];
        let sad_map = vec![200.0f32; 4]; // 2×2 blocks
        let map = model.compute_frame_qp_map(&luma, 16, 16, Some(&sad_map));
        assert_eq!(map.len(), 4);
    }

    // PvsModel tests
    #[test]
    fn pvs_model_threshold_positive() {
        let t = PvsModel::visibility_threshold(0.5, 0.0);
        assert!(t > 0.0, "threshold must be positive, got {t}");
    }

    #[test]
    fn pvs_model_low_freq_below_high_freq() {
        let t_low = PvsModel::visibility_threshold(0.5, 0.0);
        let t_high = PvsModel::visibility_threshold(0.5, 1.0);
        assert!(
            t_low <= t_high,
            "low-freq threshold {t_low} should be <= high-freq {t_high}"
        );
    }

    #[test]
    fn pvs_model_dark_region_higher_threshold() {
        let t_mid = PvsModel::visibility_threshold(0.5, 0.5);
        let t_dark = PvsModel::visibility_threshold(0.0, 0.5);
        assert!(
            t_dark >= t_mid,
            "dark region threshold {t_dark} should be >= mid-tone {t_mid}"
        );
    }

    #[test]
    fn pvs_model_bright_region_higher_threshold() {
        let t_mid = PvsModel::visibility_threshold(0.5, 0.5);
        let t_bright = PvsModel::visibility_threshold(1.0, 0.5);
        assert!(
            t_bright >= t_mid,
            "bright region threshold {t_bright} should be >= mid-tone {t_mid}"
        );
    }

    #[test]
    fn pvs_model_clamped_inputs_safe() {
        // Should not panic with out-of-range inputs
        let t1 = PvsModel::visibility_threshold(-1.0, -0.5);
        let t2 = PvsModel::visibility_threshold(2.0, 5.0);
        assert!(t1 > 0.0);
        assert!(t2 > 0.0);
    }
}
