//! Denoising-aware encoding optimization.
//!
//! Coordinates with a pre-encode denoiser to avoid wasting bits on noise.
//! The key insight is that noise consumes encoding bits without contributing
//! to perceived quality, so the encoder should:
//!
//! 1. **Detect** how much noise is present in each block.
//! 2. **Adjust QP** upward in noisy regions (spend fewer bits on noise).
//! 3. **Coordinate** denoiser strength with encoder QP so that the denoiser
//!    removes exactly the noise the encoder would waste bits on.
//! 4. **Preserve** film grain and intentional texture by distinguishing it
//!    from sensor/compression noise.
//!
//! # Noise estimation
//!
//! Noise level is estimated from the high-frequency residual after a
//! separable median filter.  The standard deviation of this residual in
//! flat (low-gradient) regions gives a robust noise estimate.
//!
//! # QP adjustment model
//!
//! For a block with noise sigma `s`, the QP offset is:
//!
//! ```text
//! delta_qp = strength * log2(1 + s / threshold)
//! ```
//!
//! This gives zero offset when noise is below `threshold` and a
//! logarithmically increasing offset for higher noise.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use std::fmt;

// ── Noise estimation ────────────────────────────────────────────────────────

/// Result of per-block noise estimation.
#[derive(Debug, Clone)]
pub struct NoiseEstimate {
    /// Estimated noise standard deviation (luma, 0..255 scale).
    pub sigma: f64,
    /// Fraction of pixels classified as "flat" (low gradient).
    pub flat_fraction: f64,
    /// Whether the noise appears to be structured (e.g. film grain).
    pub is_structured: bool,
}

impl NoiseEstimate {
    /// A "clean" estimate with zero noise.
    pub fn clean() -> Self {
        Self {
            sigma: 0.0,
            flat_fraction: 1.0,
            is_structured: false,
        }
    }

    /// Creates a noise estimate with given sigma.
    pub fn with_sigma(sigma: f64) -> Self {
        Self {
            sigma: sigma.max(0.0),
            flat_fraction: 0.5,
            is_structured: false,
        }
    }

    /// Creates a structured-noise estimate (e.g. film grain).
    pub fn film_grain(sigma: f64) -> Self {
        Self {
            sigma: sigma.max(0.0),
            flat_fraction: 0.3,
            is_structured: true,
        }
    }
}

/// Estimates noise sigma from a block of luma samples.
///
/// Uses the Median Absolute Deviation (MAD) of the Laplacian as a
/// robust noise estimator.  For a block of pixels `p`:
///
/// 1. Compute Laplacian `L = |4*p(x,y) - p(x-1,y) - p(x+1,y) - p(x,y-1) - p(x,y+1)|`
/// 2. `sigma = 1.4826 * median(|L - median(L)|)`
///
/// The 1.4826 factor converts MAD to standard deviation for Gaussian noise.
pub fn estimate_noise_block(
    pixels: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> NoiseEstimate {
    if width < 3 || height < 3 {
        return NoiseEstimate::clean();
    }

    let mut laplacian_values = Vec::with_capacity((width - 2) * (height - 2));
    let mut gradient_magnitudes = Vec::with_capacity((width - 2) * (height - 2));

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let center = pixels[y * stride + x] as f64;
            let left = pixels[y * stride + x - 1] as f64;
            let right = pixels[y * stride + x + 1] as f64;
            let top = pixels[(y - 1) * stride + x] as f64;
            let bottom = pixels[(y + 1) * stride + x] as f64;

            let lap = (4.0 * center - left - right - top - bottom).abs();
            laplacian_values.push(lap);

            // Gradient magnitude for flat-region classification
            let gx = (right - left).abs();
            let gy = (bottom - top).abs();
            gradient_magnitudes.push(gx + gy);
        }
    }

    if laplacian_values.is_empty() {
        return NoiseEstimate::clean();
    }

    // Median of Laplacian values
    laplacian_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_lap = laplacian_values[laplacian_values.len() / 2];

    // MAD
    let mut abs_devs: Vec<f64> = laplacian_values
        .iter()
        .map(|v| (v - median_lap).abs())
        .collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = abs_devs[abs_devs.len() / 2];

    // sigma = 1.4826 * MAD / sqrt(20) for 5-tap Laplacian normalisation
    let sigma = 1.4826 * mad / 20.0_f64.sqrt();

    // Flat fraction: pixels with gradient below threshold
    let gradient_threshold = 10.0;
    let flat_count = gradient_magnitudes
        .iter()
        .filter(|&&g| g < gradient_threshold)
        .count();
    let flat_fraction = flat_count as f64 / gradient_magnitudes.len().max(1) as f64;

    // Structured noise detection: high noise but low flat fraction
    // suggests grain/texture rather than random sensor noise
    let is_structured = sigma > 3.0 && flat_fraction < 0.3;

    NoiseEstimate {
        sigma,
        flat_fraction,
        is_structured,
    }
}

/// Estimates noise for a full frame, returning per-block estimates.
pub fn estimate_noise_frame(
    pixels: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    block_size: usize,
) -> Vec<NoiseEstimate> {
    let mut estimates = Vec::new();
    let block_rows = height / block_size.max(1);
    let block_cols = width / block_size.max(1);

    for by in 0..block_rows {
        for bx in 0..block_cols {
            let x0 = bx * block_size;
            let y0 = by * block_size;
            let bw = block_size.min(width - x0);
            let bh = block_size.min(height - y0);

            let offset = y0 * stride + x0;
            if offset < pixels.len() {
                let est = estimate_noise_block(&pixels[offset..], bw, bh, stride);
                estimates.push(est);
            }
        }
    }

    estimates
}

// ── QP adjustment ───────────────────────────────────────────────────────────

/// Configuration for denoising-aware QP adjustment.
#[derive(Debug, Clone)]
pub struct DenoiseAwareConfig {
    /// QP adjustment strength.  1.0 = standard, >1.0 = more aggressive.
    pub strength: f64,
    /// Noise threshold below which no QP adjustment is applied.
    pub noise_threshold: f64,
    /// Maximum QP offset that can be applied.
    pub max_qp_delta: i8,
    /// Minimum QP offset (can be negative to boost quality in clean regions).
    pub min_qp_delta: i8,
    /// Film grain preservation factor (0.0 = denoise fully, 1.0 = preserve fully).
    pub grain_preservation: f64,
    /// Block size for noise analysis (in pixels).
    pub analysis_block_size: u32,
}

impl Default for DenoiseAwareConfig {
    fn default() -> Self {
        Self {
            noise_threshold: 2.0,
            max_qp_delta: 6,
            min_qp_delta: -2,
            strength: 1.0,
            grain_preservation: 0.5,
            analysis_block_size: 64,
        }
    }
}

impl fmt::Display for DenoiseAwareConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DenoiseAware(strength={:.1}, threshold={:.1}, grain_pres={:.1})",
            self.strength, self.noise_threshold, self.grain_preservation
        )
    }
}

/// Per-block QP adjustment result.
#[derive(Debug, Clone)]
pub struct DenoiseQpResult {
    /// QP delta to apply (positive = increase QP / reduce quality).
    pub qp_delta: i8,
    /// Recommended denoiser strength for this block (0.0 = no denoising).
    pub denoise_strength: f64,
    /// The noise estimate that drove the decision.
    pub noise_estimate: NoiseEstimate,
}

/// Computes the QP delta for a single block given its noise estimate.
pub fn compute_qp_delta(noise: &NoiseEstimate, config: &DenoiseAwareConfig) -> DenoiseQpResult {
    let sigma = noise.sigma;

    // No adjustment for low noise
    if sigma < config.noise_threshold {
        return DenoiseQpResult {
            qp_delta: 0,
            denoise_strength: 0.0,
            noise_estimate: noise.clone(),
        };
    }

    // Base QP offset from noise level
    let raw_delta = config.strength * (1.0 + sigma / config.noise_threshold).log2();

    // Reduce QP offset for structured noise (film grain) to preserve it
    let grain_factor = if noise.is_structured {
        1.0 - config.grain_preservation
    } else {
        1.0
    };

    let adjusted_delta = raw_delta * grain_factor;

    // Clamp
    let clamped = adjusted_delta
        .round()
        .max(config.min_qp_delta as f64)
        .min(config.max_qp_delta as f64) as i8;

    // Recommended denoiser strength: complement of QP adjustment
    // If we increase QP a lot, the denoiser can be lighter (encoder handles it).
    // If grain preservation is high, denoiser should be weaker.
    let denoise_strength = if noise.is_structured {
        (sigma / 255.0 * (1.0 - config.grain_preservation)).max(0.0)
    } else {
        (sigma / 255.0 * config.strength).min(1.0).max(0.0)
    };

    DenoiseQpResult {
        qp_delta: clamped,
        denoise_strength,
        noise_estimate: noise.clone(),
    }
}

// ── Frame-level coordinator ─────────────────────────────────────────────────

/// Frame-level denoising-aware optimization result.
#[derive(Debug, Clone)]
pub struct FrameDenoiseResult {
    /// Per-block QP adjustments (row-major order).
    pub block_results: Vec<DenoiseQpResult>,
    /// Block grid columns.
    pub block_cols: usize,
    /// Block grid rows.
    pub block_rows: usize,
    /// Average noise sigma across the frame.
    pub avg_sigma: f64,
    /// Fraction of blocks with structured noise.
    pub grain_fraction: f64,
    /// Overall recommended denoiser strength for the frame.
    pub frame_denoise_strength: f64,
}

/// Analyses a full frame and produces per-block denoising-aware QP deltas.
pub fn analyze_frame(
    pixels: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    config: &DenoiseAwareConfig,
) -> FrameDenoiseResult {
    let bs = config.analysis_block_size as usize;
    let block_cols = if bs > 0 { width / bs.max(1) } else { 0 };
    let block_rows = if bs > 0 { height / bs.max(1) } else { 0 };

    if block_cols == 0 || block_rows == 0 || pixels.is_empty() {
        return FrameDenoiseResult {
            block_results: Vec::new(),
            block_cols: 0,
            block_rows: 0,
            avg_sigma: 0.0,
            grain_fraction: 0.0,
            frame_denoise_strength: 0.0,
        };
    }

    let estimates = estimate_noise_frame(pixels, width, height, stride, bs);
    let block_results: Vec<DenoiseQpResult> = estimates
        .iter()
        .map(|est| compute_qp_delta(est, config))
        .collect();

    let total_sigma: f64 = block_results.iter().map(|r| r.noise_estimate.sigma).sum();
    let grain_count = block_results
        .iter()
        .filter(|r| r.noise_estimate.is_structured)
        .count();
    let n = block_results.len().max(1) as f64;

    let avg_sigma = total_sigma / n;
    let grain_fraction = grain_count as f64 / n;

    let frame_denoise_strength = block_results
        .iter()
        .map(|r| r.denoise_strength)
        .sum::<f64>()
        / n;

    FrameDenoiseResult {
        block_results,
        block_cols,
        block_rows,
        avg_sigma,
        grain_fraction,
        frame_denoise_strength,
    }
}

/// Returns the recommended denoiser filter strength for a given noise sigma
/// and grain preservation setting.
pub fn recommended_filter_sigma(noise_sigma: f64, grain_preservation: f64) -> f64 {
    let effective = noise_sigma * (1.0 - grain_preservation.clamp(0.0, 1.0));
    effective.max(0.0)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_flat_block(width: usize, height: usize, value: u8) -> Vec<u8> {
        vec![value; width * height]
    }

    fn make_noisy_block(width: usize, height: usize, base: u8, noise_amp: u8) -> Vec<u8> {
        let mut pixels = Vec::with_capacity(width * height);
        for i in 0..width * height {
            // Simple deterministic "noise" pattern
            let noise = ((i * 7 + 13) % (noise_amp as usize * 2 + 1)) as u8;
            pixels.push(base.saturating_add(noise).saturating_sub(noise_amp));
        }
        pixels
    }

    #[test]
    fn test_clean_block_no_noise() {
        let block = make_flat_block(16, 16, 128);
        let est = estimate_noise_block(&block, 16, 16, 16);
        assert!(
            est.sigma < 1.0,
            "flat block should have very low noise sigma, got {}",
            est.sigma
        );
    }

    #[test]
    fn test_noisy_block_detection() {
        let block = make_noisy_block(32, 32, 128, 30);
        let est = estimate_noise_block(&block, 32, 32, 32);
        assert!(
            est.sigma > 0.5,
            "noisy block should have measurable sigma, got {}",
            est.sigma
        );
    }

    #[test]
    fn test_qp_delta_below_threshold() {
        let noise = NoiseEstimate::with_sigma(0.5);
        let config = DenoiseAwareConfig {
            noise_threshold: 2.0,
            ..Default::default()
        };
        let result = compute_qp_delta(&noise, &config);
        assert_eq!(result.qp_delta, 0, "below threshold should give zero delta");
        assert!(
            result.denoise_strength < 0.01,
            "below threshold should not denoise"
        );
    }

    #[test]
    fn test_qp_delta_above_threshold() {
        let noise = NoiseEstimate::with_sigma(10.0);
        let config = DenoiseAwareConfig::default();
        let result = compute_qp_delta(&noise, &config);
        assert!(
            result.qp_delta > 0,
            "noisy block should get positive QP delta, got {}",
            result.qp_delta
        );
    }

    #[test]
    fn test_grain_preservation() {
        let grain = NoiseEstimate::film_grain(10.0);
        let random = NoiseEstimate::with_sigma(10.0);

        let config = DenoiseAwareConfig {
            grain_preservation: 0.8,
            ..Default::default()
        };

        let grain_result = compute_qp_delta(&grain, &config);
        let random_result = compute_qp_delta(&random, &config);

        assert!(
            grain_result.qp_delta <= random_result.qp_delta,
            "grain should get less QP boost than random noise"
        );
        assert!(
            grain_result.denoise_strength < random_result.denoise_strength,
            "grain should get less denoising"
        );
    }

    #[test]
    fn test_qp_delta_clamping() {
        let extreme_noise = NoiseEstimate::with_sigma(100.0);
        let config = DenoiseAwareConfig {
            max_qp_delta: 4,
            min_qp_delta: -1,
            ..Default::default()
        };
        let result = compute_qp_delta(&extreme_noise, &config);
        assert!(result.qp_delta <= 4, "should not exceed max_qp_delta");
        assert!(result.qp_delta >= -1, "should not go below min_qp_delta");
    }

    #[test]
    fn test_frame_analysis_empty() {
        let config = DenoiseAwareConfig::default();
        let result = analyze_frame(&[], 0, 0, 0, &config);
        assert!(result.block_results.is_empty());
        assert_eq!(result.avg_sigma, 0.0);
    }

    #[test]
    fn test_frame_analysis_flat() {
        let width = 128;
        let height = 128;
        let pixels = make_flat_block(width, height, 128);
        let config = DenoiseAwareConfig {
            analysis_block_size: 64,
            ..Default::default()
        };
        let result = analyze_frame(&pixels, width, height, width, &config);
        assert_eq!(result.block_cols, 2);
        assert_eq!(result.block_rows, 2);
        assert!(result.avg_sigma < 1.0);
    }

    #[test]
    fn test_recommended_filter_sigma() {
        assert!((recommended_filter_sigma(10.0, 0.0) - 10.0).abs() < 1e-9);
        assert!((recommended_filter_sigma(10.0, 1.0) - 0.0).abs() < 1e-9);
        assert!((recommended_filter_sigma(10.0, 0.5) - 5.0).abs() < 1e-9);
        assert!((recommended_filter_sigma(0.0, 0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_noise_estimate_constructors() {
        let clean = NoiseEstimate::clean();
        assert_eq!(clean.sigma, 0.0);
        assert!(!clean.is_structured);

        let noisy = NoiseEstimate::with_sigma(5.0);
        assert!((noisy.sigma - 5.0).abs() < 1e-9);

        let grain = NoiseEstimate::film_grain(8.0);
        assert!(grain.is_structured);
    }

    #[test]
    fn test_higher_noise_higher_delta() {
        let config = DenoiseAwareConfig::default();
        let low = compute_qp_delta(&NoiseEstimate::with_sigma(3.0), &config);
        let high = compute_qp_delta(&NoiseEstimate::with_sigma(20.0), &config);
        assert!(
            high.qp_delta >= low.qp_delta,
            "higher noise should give >= QP delta"
        );
    }

    #[test]
    fn test_small_block_returns_clean() {
        let block = vec![128u8; 4]; // 2x2 block, too small
        let est = estimate_noise_block(&block, 2, 2, 2);
        assert_eq!(est.sigma, 0.0, "too-small block should return clean");
    }

    #[test]
    fn test_denoise_config_display() {
        let config = DenoiseAwareConfig::default();
        let s = format!("{config}");
        assert!(s.contains("DenoiseAware"));
        assert!(s.contains("strength="));
    }
}
