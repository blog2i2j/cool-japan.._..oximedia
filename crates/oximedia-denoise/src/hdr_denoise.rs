//! HDR-aware denoising with PQ/HLG transfer function awareness.
//!
//! Standard denoising algorithms operate in gamma-encoded space, which
//! causes non-uniform treatment of luminance levels in HDR content.
//! This module provides denoising that is aware of HDR transfer functions
//! (Perceptual Quantizer / SMPTE ST 2084 and Hybrid Log-Gamma / ARIB STD-B67),
//! ensuring:
//!
//! - **Correct noise estimation** in linear-light domain
//! - **Perceptually uniform filtering** across the wide luminance range
//! - **Shadow preservation** (HDR dark regions are more sensitive to noise)
//! - **Highlight protection** (avoid clipping specular highlights)
//!
//! The approach linearizes the signal before noise estimation, applies
//! denoising in a perceptually uniform domain, then converts back to
//! the original transfer function.

use crate::{DenoiseError, DenoiseResult};
use oximedia_codec::VideoFrame;
use rayon::prelude::*;

/// HDR transfer function type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HdrTransferFunction {
    /// Perceptual Quantizer (SMPTE ST 2084 / PQ).
    /// Used by HDR10, HDR10+, Dolby Vision.
    Pq,
    /// Hybrid Log-Gamma (ARIB STD-B67 / HLG).
    /// Used by broadcast HDR (BBC/NHK).
    Hlg,
    /// Standard Dynamic Range (BT.1886 gamma).
    /// Included for completeness / fallback.
    Sdr,
}

impl HdrTransferFunction {
    /// Get a human-readable name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Pq => "PQ (ST 2084)",
            Self::Hlg => "HLG (ARIB STD-B67)",
            Self::Sdr => "SDR (BT.1886)",
        }
    }
}

/// Configuration for HDR-aware denoising.
#[derive(Clone, Debug)]
pub struct HdrDenoiseConfig {
    /// Transfer function of the input content.
    pub transfer_function: HdrTransferFunction,
    /// Base denoising strength (0.0 - 1.0).
    pub strength: f32,
    /// Peak luminance in cd/m^2 (nits). Typical: 1000, 4000, or 10000.
    pub peak_luminance: f32,
    /// Shadow boost factor: additional denoising in dark regions (1.0 = none).
    pub shadow_boost: f32,
    /// Highlight protection factor: reduce denoising near peak (0.0 = none).
    pub highlight_protection: f32,
    /// Bilateral filter radius.
    pub filter_radius: usize,
    /// Whether to process in linear-light domain (recommended).
    pub linearize: bool,
}

impl Default for HdrDenoiseConfig {
    fn default() -> Self {
        Self {
            transfer_function: HdrTransferFunction::Pq,
            strength: 0.5,
            peak_luminance: 1000.0,
            shadow_boost: 1.5,
            highlight_protection: 0.3,
            filter_radius: 3,
            linearize: true,
        }
    }
}

impl HdrDenoiseConfig {
    /// Create configuration for PQ (HDR10) content.
    #[must_use]
    pub fn pq(peak_nits: f32) -> Self {
        Self {
            transfer_function: HdrTransferFunction::Pq,
            peak_luminance: peak_nits,
            ..Default::default()
        }
    }

    /// Create configuration for HLG content.
    #[must_use]
    pub fn hlg() -> Self {
        Self {
            transfer_function: HdrTransferFunction::Hlg,
            peak_luminance: 1000.0,
            ..Default::default()
        }
    }

    /// Create configuration for SDR content (fallback).
    #[must_use]
    pub fn sdr() -> Self {
        Self {
            transfer_function: HdrTransferFunction::Sdr,
            peak_luminance: 100.0,
            shadow_boost: 1.0,
            highlight_protection: 0.0,
            ..Default::default()
        }
    }

    /// Validate configuration.
    pub fn validate(&self) -> DenoiseResult<()> {
        if !(0.0..=1.0).contains(&self.strength) {
            return Err(DenoiseError::InvalidConfig(
                "strength must be between 0.0 and 1.0".to_string(),
            ));
        }
        if self.peak_luminance <= 0.0 {
            return Err(DenoiseError::InvalidConfig(
                "peak_luminance must be positive".to_string(),
            ));
        }
        if self.shadow_boost < 0.0 {
            return Err(DenoiseError::InvalidConfig(
                "shadow_boost must be non-negative".to_string(),
            ));
        }
        if self.filter_radius == 0 || self.filter_radius > 15 {
            return Err(DenoiseError::InvalidConfig(
                "filter_radius must be between 1 and 15".to_string(),
            ));
        }
        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────
// PQ (ST 2084) EOTF / inverse EOTF
// ────────────────────────────────────────────────────────────────────

/// PQ constants from SMPTE ST 2084.
const PQ_M1: f64 = 2610.0 / 16384.0;
const PQ_M2: f64 = 2523.0 / 4096.0 * 128.0;
const PQ_C1: f64 = 3424.0 / 4096.0;
const PQ_C2: f64 = 2413.0 / 4096.0 * 32.0;
const PQ_C3: f64 = 2392.0 / 4096.0 * 32.0;

/// PQ EOTF: Convert PQ code value (0-1) to linear light (0-1, relative to 10000 nits).
fn pq_eotf(code: f64) -> f64 {
    if code <= 0.0 {
        return 0.0;
    }
    let code_m2 = code.powf(1.0 / PQ_M2);
    let numerator = (code_m2 - PQ_C1).max(0.0);
    let denominator = PQ_C2 - PQ_C3 * code_m2;
    if denominator <= 0.0 {
        return 0.0;
    }
    (numerator / denominator).powf(1.0 / PQ_M1)
}

/// PQ inverse EOTF: Convert linear light (0-1, relative to 10000 nits) to PQ code value (0-1).
fn pq_inverse_eotf(linear: f64) -> f64 {
    if linear <= 0.0 {
        return 0.0;
    }
    let y_m1 = linear.powf(PQ_M1);
    let numerator = PQ_C1 + PQ_C2 * y_m1;
    let denominator = 1.0 + PQ_C3 * y_m1;
    (numerator / denominator).powf(PQ_M2)
}

// ────────────────────────────────────────────────────────────────────
// HLG (ARIB STD-B67) OETF / inverse OETF
// ────────────────────────────────────────────────────────────────────

const HLG_A: f64 = 0.178_832_77;
const HLG_B: f64 = 0.284_668_92;
const HLG_C: f64 = 0.559_910_73;

/// HLG inverse OETF: Convert HLG signal (0-1) to scene-linear (0-1).
fn hlg_inverse_oetf(signal: f64) -> f64 {
    if signal <= 0.0 {
        return 0.0;
    }
    if signal <= 0.5 {
        (signal * signal) / 3.0
    } else {
        (((signal - HLG_C) / HLG_A).exp() + HLG_B) / 12.0
    }
}

/// HLG OETF: Convert scene-linear (0-1) to HLG signal (0-1).
fn hlg_oetf(linear: f64) -> f64 {
    if linear <= 0.0 {
        return 0.0;
    }
    if linear <= 1.0 / 12.0 {
        (3.0 * linear).sqrt()
    } else {
        HLG_A * (12.0 * linear - HLG_B).max(1e-15).ln() + HLG_C
    }
}

// ────────────────────────────────────────────────────────────────────
// SDR gamma
// ────────────────────────────────────────────────────────────────────

/// SDR gamma linearization (BT.1886 simplified: gamma 2.4).
fn sdr_linearize(code: f64) -> f64 {
    if code <= 0.0 {
        return 0.0;
    }
    code.powf(2.4)
}

/// SDR gamma encoding.
fn sdr_encode(linear: f64) -> f64 {
    if linear <= 0.0 {
        return 0.0;
    }
    linear.powf(1.0 / 2.4)
}

// ────────────────────────────────────────────────────────────────────
// Unified linearization helpers
// ────────────────────────────────────────────────────────────────────

/// Convert a code value (0-1 normalized) to linear light using the
/// specified transfer function.
fn linearize(code: f64, tf: HdrTransferFunction) -> f64 {
    match tf {
        HdrTransferFunction::Pq => pq_eotf(code),
        HdrTransferFunction::Hlg => hlg_inverse_oetf(code),
        HdrTransferFunction::Sdr => sdr_linearize(code),
    }
}

/// Convert linear light (0-1 normalized) back to code value using
/// the specified transfer function.
fn encode(linear: f64, tf: HdrTransferFunction) -> f64 {
    match tf {
        HdrTransferFunction::Pq => pq_inverse_eotf(linear),
        HdrTransferFunction::Hlg => hlg_oetf(linear),
        HdrTransferFunction::Sdr => sdr_encode(linear),
    }
}

/// Compute HDR-aware local denoising strength.
///
/// Adapts the filter strength based on the luminance level in the
/// HDR transfer function domain:
/// - Dark regions: boost strength (noise is more visible)
/// - Midtones: use base strength
/// - Highlights: reduce strength (protect specular detail)
fn hdr_local_strength(
    linear_luminance: f64,
    config: &HdrDenoiseConfig,
) -> f32 {
    let base = config.strength;

    // Normalize to 0-1 range relative to peak
    let normalized = (linear_luminance * 10000.0 / f64::from(config.peak_luminance))
        .clamp(0.0, 1.0);

    // Shadow region: below 1% of peak
    let shadow_factor = if normalized < 0.01 {
        config.shadow_boost
    } else if normalized < 0.05 {
        // Smooth transition from shadow boost to normal
        let t = (normalized - 0.01) / 0.04;
        config.shadow_boost * (1.0 - t as f32) + 1.0 * t as f32
    } else {
        1.0
    };

    // Highlight region: above 90% of peak
    let highlight_factor = if normalized > 0.9 {
        let t = ((normalized - 0.9) / 0.1) as f32;
        1.0 - config.highlight_protection * t
    } else {
        1.0
    };

    (base * shadow_factor * highlight_factor).clamp(0.0, 1.0)
}

/// Apply HDR-aware denoising to a video frame.
///
/// The pipeline:
/// 1. Convert input from transfer function domain to linear light
/// 2. Estimate noise in linear domain
/// 3. Apply adaptive bilateral filter with HDR-aware strength
/// 4. Convert back to original transfer function domain
///
/// # Arguments
/// * `frame` - Input video frame (8-bit per channel)
/// * `config` - HDR denoising configuration
///
/// # Returns
/// HDR-aware denoised frame
pub fn hdr_denoise(
    frame: &VideoFrame,
    config: &HdrDenoiseConfig,
) -> DenoiseResult<VideoFrame> {
    config.validate()?;

    if frame.planes.is_empty() {
        return Err(DenoiseError::ProcessingError(
            "Frame has no planes".to_string(),
        ));
    }

    let mut output = frame.clone();

    output
        .planes
        .par_iter_mut()
        .enumerate()
        .try_for_each(|(plane_idx, out_plane)| {
            let input_plane = &frame.planes[plane_idx];
            let (width, height) = frame.plane_dimensions(plane_idx);
            let w = width as usize;
            let h = height as usize;
            let in_stride = input_plane.stride;
            let out_stride = out_plane.stride;
            let in_data: &[u8] = input_plane.data.as_ref();
            let radius = config.filter_radius;

            // For chroma planes in YCbCr, we still apply denoising but
            // without HDR luminance adaptation (use uniform strength).
            let is_luma = plane_idx == 0;

            for y in 0..h {
                for x in 0..w {
                    let center_code = f64::from(in_data[y * in_stride + x]) / 255.0;

                    // Determine local strength
                    let local_strength = if is_luma && config.linearize {
                        let linear = linearize(center_code, config.transfer_function);
                        hdr_local_strength(linear, config)
                    } else {
                        config.strength
                    };

                    if local_strength < 0.001 {
                        continue;
                    }

                    // Apply bilateral filter in the appropriate domain
                    let sigma_range = 20.0 * local_strength;
                    let range_coeff = if sigma_range > 0.001 {
                        -0.5 / (sigma_range * sigma_range)
                    } else {
                        0.0
                    };

                    let center_val = if config.linearize && is_luma {
                        linearize(center_code, config.transfer_function) as f32 * 255.0
                    } else {
                        f32::from(in_data[y * in_stride + x])
                    };

                    let mut weighted_sum = 0.0f32;
                    let mut weight_sum = 0.0f32;

                    for dy in -(radius as i32)..=(radius as i32) {
                        let ny = (y as i32 + dy).clamp(0, (h - 1) as i32) as usize;
                        for dx in -(radius as i32)..=(radius as i32) {
                            let nx = (x as i32 + dx).clamp(0, (w - 1) as i32) as usize;

                            let neighbor_raw = f32::from(in_data[ny * in_stride + nx]);
                            let neighbor_val = if config.linearize && is_luma {
                                let nc = f64::from(neighbor_raw) / 255.0;
                                linearize(nc, config.transfer_function) as f32 * 255.0
                            } else {
                                neighbor_raw
                            };

                            let val_diff = neighbor_val - center_val;
                            let weight = (val_diff * val_diff * range_coeff).exp();

                            weighted_sum += neighbor_val * weight;
                            weight_sum += weight;
                        }
                    }

                    let filtered_linear = if weight_sum > 0.0 {
                        weighted_sum / weight_sum
                    } else {
                        center_val
                    };

                    // Convert back to code value domain
                    let result = if config.linearize && is_luma {
                        let linear_norm = (f64::from(filtered_linear) / 255.0).clamp(0.0, 1.0);
                        let encoded = encode(linear_norm, config.transfer_function);
                        (encoded * 255.0).clamp(0.0, 255.0) as u8
                    } else {
                        filtered_linear.clamp(0.0, 255.0).round() as u8
                    };

                    out_plane.data[y * out_stride + x] = result;
                }
            }

            Ok::<(), DenoiseError>(())
        })?;

    Ok(output)
}

/// Estimate noise level in the linear-light domain of an HDR frame.
///
/// This gives a more accurate noise estimate than working in the
/// gamma/PQ/HLG-encoded domain because noise variance is uniform
/// in linear light for sensor noise (Poisson + read noise).
pub fn estimate_hdr_noise(
    frame: &VideoFrame,
    tf: HdrTransferFunction,
) -> DenoiseResult<HdrNoiseEstimate> {
    if frame.planes.is_empty() {
        return Err(DenoiseError::ProcessingError(
            "Frame has no planes".to_string(),
        ));
    }

    let plane = &frame.planes[0];
    let (width, height) = frame.plane_dimensions(0);
    let w = width as usize;
    let h = height as usize;
    let stride = plane.stride;
    let data: &[u8] = plane.data.as_ref();

    // Compute noise using Median Absolute Deviation (MAD) on
    // high-pass filtered signal in linear domain
    let radius = 1;
    let mut residuals = Vec::with_capacity(w * h);

    for y in radius..h.saturating_sub(radius) {
        for x in radius..w.saturating_sub(radius) {
            let center_code = f64::from(data[y * stride + x]) / 255.0;
            let center_linear = linearize(center_code, tf);

            // Laplacian high-pass
            let mut neighbor_sum = 0.0f64;
            let mut count = 0u32;
            for dy in [-1i32, 0, 1] {
                for dx in [-1i32, 0, 1] {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let ny = (y as i32 + dy) as usize;
                    let nx = (x as i32 + dx) as usize;
                    let nc = f64::from(data[ny * stride + nx]) / 255.0;
                    neighbor_sum += linearize(nc, tf);
                    count += 1;
                }
            }

            let avg_neighbor = neighbor_sum / f64::from(count);
            let residual = (center_linear - avg_neighbor).abs();
            residuals.push(residual);
        }
    }

    if residuals.is_empty() {
        return Ok(HdrNoiseEstimate {
            linear_noise_std: 0.0,
            code_noise_std: 0.0,
            snr_db: f64::INFINITY,
            transfer_function: tf,
        });
    }

    // MAD estimator: sigma = 1.4826 * median(|residuals|)
    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = residuals[residuals.len() / 2];
    let linear_noise_std = median * 1.4826;

    // Estimate corresponding code-domain noise
    let mid_code = 0.5;
    let mid_linear = linearize(mid_code, tf);
    let perturbed = encode(mid_linear + linear_noise_std, tf);
    let code_noise_std = (perturbed - mid_code).abs();

    // Estimate SNR
    let signal_power: f64 = residuals.iter().map(|&r| r * r).sum::<f64>()
        / residuals.len() as f64;
    let noise_power = linear_noise_std * linear_noise_std;
    let snr_db = if noise_power > 1e-15 {
        10.0 * (signal_power / noise_power).log10()
    } else {
        f64::INFINITY
    };

    Ok(HdrNoiseEstimate {
        linear_noise_std,
        code_noise_std,
        snr_db,
        transfer_function: tf,
    })
}

/// Result of HDR noise estimation.
#[derive(Clone, Debug)]
pub struct HdrNoiseEstimate {
    /// Estimated noise standard deviation in linear-light domain (0-1 scale).
    pub linear_noise_std: f64,
    /// Estimated noise standard deviation in code-value domain (0-1 scale).
    pub code_noise_std: f64,
    /// Signal-to-noise ratio in decibels.
    pub snr_db: f64,
    /// Transfer function used for the estimate.
    pub transfer_function: HdrTransferFunction,
}

impl HdrNoiseEstimate {
    /// Whether the frame is considered clean (noise below threshold).
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.linear_noise_std < 0.001
    }

    /// Suggested denoising strength based on noise level.
    #[must_use]
    pub fn suggested_strength(&self) -> f32 {
        let noise_level = (self.linear_noise_std * 100.0) as f32;
        noise_level.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    fn make_test_frame(width: u32, height: u32, fill: u8) -> VideoFrame {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, width, height);
        frame.allocate();
        let stride = frame.planes[0].stride;
        for y in 0..height as usize {
            for x in 0..width as usize {
                frame.planes[0].data[y * stride + x] = fill;
            }
        }
        frame
    }

    fn make_gradient_frame(width: u32, height: u32) -> VideoFrame {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, width, height);
        frame.allocate();
        let stride = frame.planes[0].stride;
        for y in 0..height as usize {
            for x in 0..width as usize {
                let val = ((x * 255) / width.max(1) as usize) as u8;
                frame.planes[0].data[y * stride + x] = val;
            }
        }
        frame
    }

    // ── PQ transfer function tests ──────────────────────────────────

    #[test]
    fn test_pq_roundtrip() {
        // PQ EOTF and inverse should be inverses of each other
        for code_u8 in [0, 32, 64, 128, 192, 255] {
            let code = f64::from(code_u8) / 255.0;
            let linear = pq_eotf(code);
            let reconstructed = pq_inverse_eotf(linear);
            assert!(
                (code - reconstructed).abs() < 0.001,
                "PQ roundtrip failed for code={code}: got {reconstructed}"
            );
        }
    }

    #[test]
    fn test_pq_monotonic() {
        // PQ should be monotonically increasing
        let mut prev = 0.0;
        for i in 0..=255 {
            let code = f64::from(i) / 255.0;
            let linear = pq_eotf(code);
            assert!(
                linear >= prev,
                "PQ should be monotonic: code={code}, prev_linear={prev}, linear={linear}"
            );
            prev = linear;
        }
    }

    #[test]
    fn test_pq_zero() {
        assert!((pq_eotf(0.0) - 0.0).abs() < 1e-10);
        assert!((pq_inverse_eotf(0.0) - 0.0).abs() < 1e-10);
    }

    // ── HLG transfer function tests ─────────────────────────────────

    #[test]
    fn test_hlg_roundtrip() {
        for code_u8 in [0, 32, 64, 128, 192, 255] {
            let code = f64::from(code_u8) / 255.0;
            let linear = hlg_inverse_oetf(code);
            let reconstructed = hlg_oetf(linear);
            assert!(
                (code - reconstructed).abs() < 0.01,
                "HLG roundtrip failed for code={code}: got {reconstructed}"
            );
        }
    }

    #[test]
    fn test_hlg_monotonic() {
        let mut prev = 0.0;
        for i in 0..=255 {
            let code = f64::from(i) / 255.0;
            let linear = hlg_inverse_oetf(code);
            assert!(
                linear >= prev - 1e-10,
                "HLG should be monotonic: code={code}, linear={linear}"
            );
            prev = linear;
        }
    }

    #[test]
    fn test_hlg_zero() {
        assert!((hlg_inverse_oetf(0.0) - 0.0).abs() < 1e-10);
        assert!((hlg_oetf(0.0) - 0.0).abs() < 1e-10);
    }

    // ── SDR transfer function tests ─────────────────────────────────

    #[test]
    fn test_sdr_roundtrip() {
        for code_u8 in [0, 32, 64, 128, 192, 255] {
            let code = f64::from(code_u8) / 255.0;
            let linear = sdr_linearize(code);
            let reconstructed = sdr_encode(linear);
            assert!(
                (code - reconstructed).abs() < 0.001,
                "SDR roundtrip failed for code={code}: got {reconstructed}"
            );
        }
    }

    // ── Config tests ────────────────────────────────────────────────

    #[test]
    fn test_config_default_valid() {
        let config = HdrDenoiseConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.transfer_function, HdrTransferFunction::Pq);
    }

    #[test]
    fn test_config_pq() {
        let config = HdrDenoiseConfig::pq(4000.0);
        assert!(config.validate().is_ok());
        assert!((config.peak_luminance - 4000.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_hlg() {
        let config = HdrDenoiseConfig::hlg();
        assert!(config.validate().is_ok());
        assert_eq!(config.transfer_function, HdrTransferFunction::Hlg);
    }

    #[test]
    fn test_config_sdr() {
        let config = HdrDenoiseConfig::sdr();
        assert!(config.validate().is_ok());
        assert_eq!(config.transfer_function, HdrTransferFunction::Sdr);
    }

    #[test]
    fn test_config_invalid_strength() {
        let config = HdrDenoiseConfig {
            strength: 1.5,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_invalid_peak() {
        let config = HdrDenoiseConfig {
            peak_luminance: -100.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    // ── HDR local strength tests ────────────────────────────────────

    #[test]
    fn test_hdr_shadow_boost() {
        let config = HdrDenoiseConfig {
            shadow_boost: 2.0,
            ..Default::default()
        };
        // Very dark region (near 0 linear light)
        let shadow_strength = hdr_local_strength(0.0001, &config);
        // Midtone region
        let mid_strength = hdr_local_strength(0.05, &config);
        assert!(
            shadow_strength > mid_strength,
            "Shadows should get stronger filtering: shadow={shadow_strength}, mid={mid_strength}"
        );
    }

    #[test]
    fn test_hdr_highlight_protection() {
        let config = HdrDenoiseConfig {
            highlight_protection: 0.5,
            peak_luminance: 1000.0,
            ..Default::default()
        };
        // Highlight region (near peak)
        let highlight_strength = hdr_local_strength(0.095, &config);
        // Midtone
        let mid_strength = hdr_local_strength(0.03, &config);
        assert!(
            highlight_strength <= mid_strength,
            "Highlights should get less filtering: highlight={highlight_strength}, mid={mid_strength}"
        );
    }

    // ── HDR denoise integration tests ───────────────────────────────

    #[test]
    fn test_hdr_denoise_pq_basic() {
        let frame = make_test_frame(64, 64, 128);
        let config = HdrDenoiseConfig::pq(1000.0);
        let result = hdr_denoise(&frame, &config);
        assert!(result.is_ok());
        let output = result.expect("HDR denoise should succeed");
        assert_eq!(output.width, 64);
        assert_eq!(output.height, 64);
    }

    #[test]
    fn test_hdr_denoise_hlg_basic() {
        let frame = make_test_frame(64, 64, 128);
        let config = HdrDenoiseConfig::hlg();
        let result = hdr_denoise(&frame, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_hdr_denoise_sdr_basic() {
        let frame = make_test_frame(64, 64, 128);
        let config = HdrDenoiseConfig::sdr();
        let result = hdr_denoise(&frame, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_hdr_denoise_gradient() {
        let frame = make_gradient_frame(64, 64);
        let config = HdrDenoiseConfig::pq(1000.0);
        let result = hdr_denoise(&frame, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_hdr_denoise_empty_frame_error() {
        let frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        let config = HdrDenoiseConfig::default();
        let result = hdr_denoise(&frame, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_hdr_denoise_non_linearize() {
        let frame = make_test_frame(64, 64, 128);
        let config = HdrDenoiseConfig {
            linearize: false,
            ..Default::default()
        };
        let result = hdr_denoise(&frame, &config);
        assert!(result.is_ok());
    }

    // ── HDR noise estimation tests ──────────────────────────────────

    #[test]
    fn test_estimate_hdr_noise_uniform() {
        let frame = make_test_frame(64, 64, 128);
        let result = estimate_hdr_noise(&frame, HdrTransferFunction::Pq);
        assert!(result.is_ok());
        let estimate = result.expect("noise estimate should succeed");
        // Uniform frame should have very low noise
        assert!(
            estimate.linear_noise_std < 0.01,
            "Uniform frame should have low noise, got {}",
            estimate.linear_noise_std
        );
        assert!(estimate.is_clean());
    }

    #[test]
    fn test_estimate_hdr_noise_hlg() {
        let frame = make_test_frame(64, 64, 128);
        let result = estimate_hdr_noise(&frame, HdrTransferFunction::Hlg);
        assert!(result.is_ok());
    }

    #[test]
    fn test_estimate_hdr_noise_sdr() {
        let frame = make_test_frame(64, 64, 128);
        let result = estimate_hdr_noise(&frame, HdrTransferFunction::Sdr);
        assert!(result.is_ok());
    }

    #[test]
    fn test_estimate_hdr_noise_empty_error() {
        let frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        let result = estimate_hdr_noise(&frame, HdrTransferFunction::Pq);
        assert!(result.is_err());
    }

    #[test]
    fn test_hdr_noise_estimate_suggested_strength() {
        let estimate = HdrNoiseEstimate {
            linear_noise_std: 0.005,
            code_noise_std: 0.01,
            snr_db: 30.0,
            transfer_function: HdrTransferFunction::Pq,
        };
        let strength = estimate.suggested_strength();
        assert!(strength >= 0.0 && strength <= 1.0);
    }

    #[test]
    fn test_transfer_function_names() {
        assert!(!HdrTransferFunction::Pq.name().is_empty());
        assert!(!HdrTransferFunction::Hlg.name().is_empty());
        assert!(!HdrTransferFunction::Sdr.name().is_empty());
    }

    #[test]
    fn test_linearize_encode_consistency() {
        // All transfer functions should have linearize/encode as inverses
        for tf in [
            HdrTransferFunction::Pq,
            HdrTransferFunction::Hlg,
            HdrTransferFunction::Sdr,
        ] {
            for i in [0, 32, 64, 128, 192, 255] {
                let code = f64::from(i) / 255.0;
                let linear = linearize(code, tf);
                let reconstructed = encode(linear, tf);
                assert!(
                    (code - reconstructed).abs() < 0.01,
                    "Roundtrip failed for {tf:?} code={code}: got {reconstructed}"
                );
            }
        }
    }
}
