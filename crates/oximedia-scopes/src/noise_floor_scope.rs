//! Noise floor scope for measuring and displaying signal-to-noise ratio per channel.
//!
//! This module provides per-channel SNR analysis for video frames, computing
//! the noise floor as the standard deviation of pixel intensities in near-black
//! regions, and rendering a vertical bar display showing noise level and SNR
//! for each RGB channel.
//!
//! # Algorithm
//!
//! 1. Identify "near-black" pixels (luma below a configurable threshold).
//! 2. Compute the standard deviation of intensity values in that region per channel.
//! 3. Express the result as dB SNR relative to the full-scale signal (255).
//!
//! # Display
//!
//! The scope renders three vertical bars (R, G, B) plus a composite luma bar.
//! Each bar's height shows the noise floor level; a horizontal graticule marks
//! common broadcast thresholds (−60 dB, −50 dB, −40 dB, −30 dB).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use oximedia_core::{OxiError, OxiResult};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the noise floor scope.
#[derive(Debug, Clone)]
pub struct NoiseFloorConfig {
    /// Output image width in pixels.
    pub width: u32,
    /// Output image height in pixels.
    pub height: u32,
    /// Luma threshold (0–255) below which a pixel is considered "near-black".
    /// Pixels with luma ≤ this value contribute to the noise measurement.
    pub dark_threshold: u8,
    /// Whether to show dB graticule lines.
    pub show_graticule: bool,
    /// dB reference range to display (bottom of scope = 0 dB noise, top = max_db_range dB SNR).
    pub max_snr_db: f32,
}

impl Default for NoiseFloorConfig {
    fn default() -> Self {
        Self {
            width: 256,
            height: 200,
            dark_threshold: 32,
            show_graticule: true,
            max_snr_db: 70.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Analysis result
// ─────────────────────────────────────────────────────────────────────────────

/// Noise floor analysis results for a single RGB24 frame.
#[derive(Debug, Clone)]
pub struct NoiseFloorAnalysis {
    /// Number of "near-black" pixels used for the noise measurement.
    pub dark_pixel_count: u64,
    /// Total pixels in the frame.
    pub total_pixels: u64,
    /// Noise floor (standard deviation) for the red channel [0.0, 255.0].
    pub noise_r: f64,
    /// Noise floor (standard deviation) for the green channel [0.0, 255.0].
    pub noise_g: f64,
    /// Noise floor (standard deviation) for the blue channel [0.0, 255.0].
    pub noise_b: f64,
    /// Noise floor (standard deviation) for composite BT.709 luma [0.0, 255.0].
    pub noise_luma: f64,
    /// SNR for the red channel in dB (relative to full scale = 255).
    pub snr_r_db: f64,
    /// SNR for the green channel in dB.
    pub snr_g_db: f64,
    /// SNR for the blue channel in dB.
    pub snr_b_db: f64,
    /// SNR for composite luma in dB.
    pub snr_luma_db: f64,
}

impl NoiseFloorAnalysis {
    /// Returns the best (highest) SNR across all channels in dB.
    #[must_use]
    pub fn best_snr_db(&self) -> f64 {
        self.snr_r_db
            .max(self.snr_g_db)
            .max(self.snr_b_db)
            .max(self.snr_luma_db)
    }

    /// Returns the worst (lowest) SNR across all channels in dB.
    #[must_use]
    pub fn worst_snr_db(&self) -> f64 {
        self.snr_r_db
            .min(self.snr_g_db)
            .min(self.snr_b_db)
            .min(self.snr_luma_db)
    }

    /// Returns `true` when the noise floor is below `threshold_db` SNR on every channel.
    ///
    /// A *higher* SNR means the noise floor is lower (better). This returns
    /// `true` if all channels are *above* the given SNR threshold.
    #[must_use]
    pub fn is_clean(&self, threshold_db: f64) -> bool {
        self.snr_r_db >= threshold_db
            && self.snr_g_db >= threshold_db
            && self.snr_b_db >= threshold_db
            && self.snr_luma_db >= threshold_db
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Analysis
// ─────────────────────────────────────────────────────────────────────────────

/// Analyse the noise floor of an RGB24 frame.
///
/// Near-black pixels (BT.709 luma ≤ `config.dark_threshold`) are collected and
/// their per-channel standard deviation is computed as the noise floor.  SNR is
/// expressed as `20 · log₁₀(255 / σ)` where `σ` is the standard deviation.
///
/// # Errors
///
/// Returns an error if the frame buffer is smaller than `width * height * 3`.
pub fn analyze_noise_floor(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &NoiseFloorConfig,
) -> OxiResult<NoiseFloorAnalysis> {
    let total_pixels = (width as usize) * (height as usize);
    let expected = total_pixels * 3;

    if frame.len() < expected {
        return Err(OxiError::InvalidData(format!(
            "Frame too small: need {expected}, got {}",
            frame.len()
        )));
    }

    if total_pixels == 0 {
        return Ok(NoiseFloorAnalysis {
            dark_pixel_count: 0,
            total_pixels: 0,
            noise_r: 0.0,
            noise_g: 0.0,
            noise_b: 0.0,
            noise_luma: 0.0,
            snr_r_db: 96.0,
            snr_g_db: 96.0,
            snr_b_db: 96.0,
            snr_luma_db: 96.0,
        });
    }

    let threshold = config.dark_threshold;

    // Collect near-black pixels
    let mut dark_r: Vec<f64> = Vec::new();
    let mut dark_g: Vec<f64> = Vec::new();
    let mut dark_b: Vec<f64> = Vec::new();
    let mut dark_luma: Vec<f64> = Vec::new();

    for i in 0..total_pixels {
        let base = i * 3;
        let r = frame[base];
        let g = frame[base + 1];
        let b = frame[base + 2];

        // BT.709 luma
        let luma = 0.2126 * f64::from(r) + 0.7152 * f64::from(g) + 0.0722 * f64::from(b);

        if luma <= f64::from(threshold) {
            dark_r.push(f64::from(r));
            dark_g.push(f64::from(g));
            dark_b.push(f64::from(b));
            dark_luma.push(luma);
        }
    }

    let dark_pixel_count = dark_r.len() as u64;

    let noise_r = stddev(&dark_r);
    let noise_g = stddev(&dark_g);
    let noise_b = stddev(&dark_b);
    let noise_luma = stddev(&dark_luma);

    let snr_r_db = sigma_to_snr_db(noise_r);
    let snr_g_db = sigma_to_snr_db(noise_g);
    let snr_b_db = sigma_to_snr_db(noise_b);
    let snr_luma_db = sigma_to_snr_db(noise_luma);

    Ok(NoiseFloorAnalysis {
        dark_pixel_count,
        total_pixels: total_pixels as u64,
        noise_r,
        noise_g,
        noise_b,
        noise_luma,
        snr_r_db,
        snr_g_db,
        snr_b_db,
        snr_luma_db,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Rendering
// ─────────────────────────────────────────────────────────────────────────────

/// Render the noise floor scope as an RGBA image.
///
/// Four vertical bars are drawn side-by-side: R (red), G (green), B (blue),
/// and composite luma (white).  Each bar's height represents the noise floor
/// level; a higher bar means *more* noise (lower SNR).  Horizontal graticule
/// lines are drawn at standard broadcast thresholds if `config.show_graticule`
/// is enabled.
///
/// # Errors
///
/// Returns an error if the frame buffer is too small.
pub fn render_noise_floor_scope(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &NoiseFloorConfig,
) -> OxiResult<Vec<u8>> {
    let analysis = analyze_noise_floor(frame, width, height, config)?;
    Ok(render_analysis(&analysis, config))
}

/// Render a pre-computed [`NoiseFloorAnalysis`] to an RGBA image.
#[must_use]
pub fn render_analysis(analysis: &NoiseFloorAnalysis, config: &NoiseFloorConfig) -> Vec<u8> {
    let out_w = config.width as usize;
    let out_h = config.height as usize;
    let mut buf = vec![0u8; out_w * out_h * 4];

    // Background: very dark grey
    for chunk in buf.chunks_exact_mut(4) {
        chunk[0] = 12;
        chunk[1] = 12;
        chunk[2] = 14;
        chunk[3] = 255;
    }

    let channels: &[(&NoiseFloorAnalysis, [u8; 4])] = &[
        (analysis, [220, 60, 60, 230]),  // R
        (analysis, [60, 220, 60, 230]),  // G
        (analysis, [60, 60, 220, 230]),  // B
        (analysis, [200, 200, 200, 220]), // Luma
    ];

    let snr_values = [
        analysis.snr_r_db,
        analysis.snr_g_db,
        analysis.snr_b_db,
        analysis.snr_luma_db,
    ];
    let colors = [
        [220u8, 60, 60, 230],
        [60, 220, 60, 230],
        [60, 60, 220, 230],
        [200, 200, 200, 220],
    ];

    let num_bars = channels.len();
    let gutter = 4usize;
    let total_gutters = (num_bars + 1) * gutter;
    let bar_w = if out_w > total_gutters {
        (out_w - total_gutters) / num_bars
    } else {
        4
    };

    let max_snr = config.max_snr_db as f64;

    for (bar_idx, (&snr, color)) in snr_values.iter().zip(colors.iter()).enumerate() {
        let x_start = gutter + bar_idx * (bar_w + gutter);

        // Height proportion: higher SNR = taller bar (less noise visible)
        // We display noise level as inverse of SNR: a low SNR (= high noise) fills more.
        // Noise level: 0 dB SNR = maximum noise, max_snr dB = minimum noise.
        let noise_fraction = (1.0 - snr / max_snr).clamp(0.0, 1.0);
        let bar_h = (noise_fraction * (out_h as f64 - 1.0)).round() as usize;
        let bar_h = bar_h.clamp(1, out_h);

        // Draw bar from top (noise level fills from top downward)
        for row in 0..bar_h {
            for col in 0..bar_w {
                let px = x_start + col;
                if px >= out_w {
                    break;
                }
                let idx = (row * out_w + px) * 4;
                if idx + 3 < buf.len() {
                    buf[idx] = color[0];
                    buf[idx + 1] = color[1];
                    buf[idx + 2] = color[2];
                    buf[idx + 3] = color[3];
                }
            }
        }

        // Draw a bright top line for readability
        if bar_h > 0 {
            let top_row = bar_h.saturating_sub(1);
            for col in 0..bar_w {
                let px = x_start + col;
                if px >= out_w {
                    break;
                }
                let idx = (top_row * out_w + px) * 4;
                if idx + 3 < buf.len() {
                    buf[idx] = 255;
                    buf[idx + 1] = 255;
                    buf[idx + 2] = 255;
                    buf[idx + 3] = 255;
                }
            }
        }
    }

    // Graticule: horizontal dashed lines at -30, -40, -50, -60 dB SNR
    if config.show_graticule {
        let graticule_snr_db = [30.0f64, 40.0, 50.0, 60.0];
        for &snr_mark in &graticule_snr_db {
            let fraction = (1.0 - snr_mark / max_snr).clamp(0.0, 1.0);
            let row = (fraction * (out_h as f64 - 1.0)).round() as usize;
            if row < out_h {
                for px in 0..out_w {
                    // Dashed: every 4 pixels
                    if (px / 4) % 2 == 0 {
                        let idx = (row * out_w + px) * 4;
                        if idx + 3 < buf.len() {
                            buf[idx] = 100;
                            buf[idx + 1] = 100;
                            buf[idx + 2] = 80;
                            buf[idx + 3] = 200;
                        }
                    }
                }
            }
        }
    }

    buf
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute population standard deviation of a slice.
fn stddev(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let variance = values.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
    variance.sqrt()
}

/// Convert a standard deviation σ to SNR in dB: `20 · log₁₀(255 / σ)`.
///
/// A σ of 0 (perfectly silent / no noise) returns a sentinel of 96 dB (≈ 8-bit limit).
fn sigma_to_snr_db(sigma: f64) -> f64 {
    if sigma < 1e-10 {
        96.0 // ~8-bit theoretical maximum
    } else {
        20.0 * (255.0 / sigma).log10()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgb(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        vec![[r, g, b]; (w * h) as usize]
            .into_iter()
            .flatten()
            .collect()
    }

    #[test]
    fn test_frame_too_small_returns_error() {
        let frame = vec![0u8; 5];
        let config = NoiseFloorConfig::default();
        let result = analyze_noise_floor(&frame, 10, 10, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_dimensions_returns_ok() {
        let frame = vec![];
        let config = NoiseFloorConfig::default();
        let result = analyze_noise_floor(&frame, 0, 0, &config);
        assert!(result.is_ok());
        let a = result.expect("ok");
        assert_eq!(a.dark_pixel_count, 0);
    }

    #[test]
    fn test_all_black_frame_high_snr() {
        // A perfectly flat black frame has σ = 0 ⟹ SNR = 96 dB
        let frame = solid_rgb(8, 8, 0, 0, 0);
        let config = NoiseFloorConfig {
            dark_threshold: 10,
            ..Default::default()
        };
        let result = analyze_noise_floor(&frame, 8, 8, &config).expect("ok");
        assert!((result.snr_r_db - 96.0).abs() < 0.01, "SNR={}", result.snr_r_db);
    }

    #[test]
    fn test_bright_frame_not_in_dark_sample() {
        // A fully bright frame → no pixels below threshold → dark_pixel_count = 0
        let frame = solid_rgb(4, 4, 200, 200, 200);
        let config = NoiseFloorConfig {
            dark_threshold: 32,
            ..Default::default()
        };
        let result = analyze_noise_floor(&frame, 4, 4, &config).expect("ok");
        assert_eq!(result.dark_pixel_count, 0);
    }

    #[test]
    fn test_is_clean_threshold() {
        let analysis = NoiseFloorAnalysis {
            dark_pixel_count: 16,
            total_pixels: 16,
            noise_r: 0.5,
            noise_g: 0.5,
            noise_b: 0.5,
            noise_luma: 0.5,
            snr_r_db: 54.0,
            snr_g_db: 54.0,
            snr_b_db: 54.0,
            snr_luma_db: 54.0,
        };
        assert!(analysis.is_clean(50.0));
        assert!(!analysis.is_clean(60.0));
    }

    #[test]
    fn test_best_and_worst_snr() {
        let analysis = NoiseFloorAnalysis {
            dark_pixel_count: 4,
            total_pixels: 4,
            noise_r: 1.0,
            noise_g: 2.0,
            noise_b: 3.0,
            noise_luma: 1.5,
            snr_r_db: 48.1,
            snr_g_db: 42.1,
            snr_b_db: 38.6,
            snr_luma_db: 44.6,
        };
        assert!((analysis.best_snr_db() - 48.1).abs() < 0.01);
        assert!((analysis.worst_snr_db() - 38.6).abs() < 0.01);
    }

    #[test]
    fn test_render_output_dimensions() {
        let frame = solid_rgb(16, 16, 0, 0, 0);
        let config = NoiseFloorConfig {
            width: 128,
            height: 100,
            ..Default::default()
        };
        let result = render_noise_floor_scope(&frame, 16, 16, &config).expect("ok");
        assert_eq!(result.len(), 128 * 100 * 4);
    }

    #[test]
    fn test_render_output_is_rgba() {
        let frame = solid_rgb(4, 4, 5, 5, 5);
        let config = NoiseFloorConfig {
            width: 64,
            height: 50,
            ..Default::default()
        };
        let result = render_noise_floor_scope(&frame, 4, 4, &config).expect("ok");
        // Every 4th byte (alpha) should be 255 in our background
        assert!(result.chunks_exact(4).all(|c| c[3] > 0));
    }

    #[test]
    fn test_sigma_to_snr_zero_sigma() {
        assert!((sigma_to_snr_db(0.0) - 96.0).abs() < 0.01);
    }

    #[test]
    fn test_sigma_to_snr_known_value() {
        // σ = 1.0 → 20 * log10(255/1) = 20 * log10(255) ≈ 48.13 dB
        let snr = sigma_to_snr_db(1.0);
        assert!((snr - 48.13).abs() < 0.1, "snr={snr}");
    }

    #[test]
    fn test_stddev_uniform() {
        // All same value → σ = 0
        let vals = vec![128.0f64; 20];
        assert!(stddev(&vals) < 1e-10);
    }

    #[test]
    fn test_config_default() {
        let cfg = NoiseFloorConfig::default();
        assert_eq!(cfg.dark_threshold, 32);
        assert!(cfg.show_graticule);
        assert!((cfg.max_snr_db - 70.0).abs() < 0.01);
    }
}
