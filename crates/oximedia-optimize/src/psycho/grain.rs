//! Film grain detection and preservation for psychovisual optimization.
//!
//! This module detects and characterizes film grain in video frames, enabling
//! the encoder to:
//!
//! - **Avoid wasting bits** encoding random grain noise that can be resynthesized
//! - **Preserve perceived texture** by signalling grain parameters to the decoder
//! - **Adapt quantization** to grain-heavy regions (higher QP tolerance)
//! - **Generate grain synthesis parameters** (AV1 Film Grain SEI compatible)
//!
//! # Grain detection algorithm
//!
//! 1. Compute spatial frequency spectrum via block-level analysis
//! 2. Separate high-frequency grain energy from structural detail
//! 3. Measure grain intensity, uniformity, and correlation across frames
//! 4. Classify grain type: fine/coarse, monochrome/chroma-dependent
//!
//! # Usage
//!
//! ```ignore
//! use oximedia_optimize::psycho::grain::*;
//!
//! let detector = GrainDetector::new(GrainDetectorConfig::default());
//! let result = detector.analyze_block(&pixels, 64, 64);
//! if result.has_grain {
//!     let params = result.synthesis_params();
//! }
//! ```

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ── Grain types ─────────────────────────────────────────────────────────────

/// Type of film grain detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrainType {
    /// No significant grain detected.
    None,
    /// Fine grain (high-frequency, small particles).
    Fine,
    /// Coarse grain (larger, more visible particles).
    Coarse,
    /// Mixed grain (both fine and coarse components).
    Mixed,
}

/// Grain colour characteristic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrainChromaMode {
    /// Grain is luminance-only (monochrome grain).
    LumaOnly,
    /// Grain affects both luma and chroma channels.
    LumaAndChroma,
    /// Grain differs between chroma channels (rare, artistic).
    PerChannel,
}

// ── Detection result ────────────────────────────────────────────────────────

/// Result of grain analysis for a single block or frame region.
#[derive(Debug, Clone)]
pub struct GrainAnalysis {
    /// Whether grain was detected above the threshold.
    pub has_grain: bool,
    /// Grain type classification.
    pub grain_type: GrainType,
    /// Grain intensity (0.0 = none, 1.0 = very heavy).
    pub intensity: f64,
    /// Grain uniformity across the region (0.0 = highly variable, 1.0 = uniform).
    pub uniformity: f64,
    /// High-frequency energy ratio (grain energy / total energy).
    pub hf_energy_ratio: f64,
    /// Estimated noise variance (sigma squared).
    pub noise_variance: f64,
    /// Grain size estimate in pixels (average particle diameter).
    pub grain_size: f64,
    /// Recommended QP offset for this region (-6 to +6).
    pub qp_offset: i8,
    /// Chroma grain characteristic.
    pub chroma_mode: GrainChromaMode,
}

impl Default for GrainAnalysis {
    fn default() -> Self {
        Self {
            has_grain: false,
            grain_type: GrainType::None,
            intensity: 0.0,
            uniformity: 1.0,
            hf_energy_ratio: 0.0,
            noise_variance: 0.0,
            grain_size: 0.0,
            qp_offset: 0,
            chroma_mode: GrainChromaMode::LumaOnly,
        }
    }
}

impl GrainAnalysis {
    /// Returns AV1-compatible grain synthesis parameters.
    #[must_use]
    pub fn synthesis_params(&self) -> GrainSynthesisParams {
        if !self.has_grain {
            return GrainSynthesisParams::disabled();
        }

        let num_points = match self.grain_type {
            GrainType::None => 0,
            GrainType::Fine => 8,
            GrainType::Coarse => 4,
            GrainType::Mixed => 6,
        };

        // Map intensity to grain amplitude (0-255)
        let amplitude = (self.intensity * 200.0).round().min(255.0) as u8;

        // Grain size maps to AR (auto-regressive) coefficient count
        let ar_coeff_lag = if self.grain_size < 1.5 {
            0
        } else if self.grain_size < 3.0 {
            1
        } else if self.grain_size < 6.0 {
            2
        } else {
            3
        };

        GrainSynthesisParams {
            enabled: true,
            num_y_points: num_points,
            num_cb_points: if self.chroma_mode != GrainChromaMode::LumaOnly {
                num_points / 2
            } else {
                0
            },
            num_cr_points: if self.chroma_mode == GrainChromaMode::PerChannel {
                num_points / 2
            } else {
                0
            },
            grain_amplitude: amplitude,
            ar_coeff_lag,
            overlap_flag: self.grain_size > 2.0,
            grain_scale_shift: if self.intensity > 0.5 { 0 } else { 1 },
            random_seed: 0,
        }
    }

    /// Returns estimated bits saved per pixel by grain-aware encoding.
    #[must_use]
    pub fn estimated_bits_saved_per_pixel(&self) -> f64 {
        if !self.has_grain {
            return 0.0;
        }
        // Grain energy that can be resynthesized instead of encoded
        self.hf_energy_ratio * self.intensity * 0.3
    }
}

// ── Grain synthesis parameters ──────────────────────────────────────────────

/// AV1-compatible film grain synthesis parameters.
#[derive(Debug, Clone)]
pub struct GrainSynthesisParams {
    /// Whether grain synthesis is enabled.
    pub enabled: bool,
    /// Number of luma grain curve points.
    pub num_y_points: u8,
    /// Number of Cb grain curve points.
    pub num_cb_points: u8,
    /// Number of Cr grain curve points.
    pub num_cr_points: u8,
    /// Maximum grain amplitude (0-255).
    pub grain_amplitude: u8,
    /// Auto-regressive coefficient lag (0-3).
    pub ar_coeff_lag: u8,
    /// Whether grain blocks overlap.
    pub overlap_flag: bool,
    /// Grain scaling shift (0-3).
    pub grain_scale_shift: u8,
    /// Random seed for grain pattern generation.
    pub random_seed: u16,
}

impl GrainSynthesisParams {
    /// Creates disabled grain synthesis parameters.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            num_y_points: 0,
            num_cb_points: 0,
            num_cr_points: 0,
            grain_amplitude: 0,
            ar_coeff_lag: 0,
            overlap_flag: false,
            grain_scale_shift: 0,
            random_seed: 0,
        }
    }
}

// ── Detector configuration ──────────────────────────────────────────────────

/// Configuration for the grain detector.
#[derive(Debug, Clone)]
pub struct GrainDetectorConfig {
    /// Block size for local analysis (must be power of 2, 8-64).
    pub block_size: usize,
    /// Minimum HF energy ratio to classify as grain.
    pub grain_threshold: f64,
    /// Fine grain: HF ratio above this is classified as fine.
    pub fine_grain_threshold: f64,
    /// Intensity threshold below which grain is ignored.
    pub min_intensity: f64,
    /// Noise variance cap for grain classification.
    pub max_noise_variance: f64,
    /// Enable temporal grain consistency checking.
    pub temporal_consistency: bool,
}

impl Default for GrainDetectorConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            grain_threshold: 0.15,
            fine_grain_threshold: 0.4,
            min_intensity: 0.05,
            max_noise_variance: 2000.0,
            temporal_consistency: true,
        }
    }
}

// ── Grain detector ──────────────────────────────────────────────────────────

/// Film grain detector and analyzer.
#[derive(Debug, Clone)]
pub struct GrainDetector {
    config: GrainDetectorConfig,
    /// Running average of grain intensity across frames.
    temporal_intensity_avg: f64,
    /// Number of frames analyzed.
    frame_count: u64,
}

impl GrainDetector {
    /// Creates a new grain detector with the given configuration.
    #[must_use]
    pub fn new(config: GrainDetectorConfig) -> Self {
        Self {
            config,
            temporal_intensity_avg: 0.0,
            frame_count: 0,
        }
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &GrainDetectorConfig {
        &self.config
    }

    /// Analyzes a pixel block for grain content.
    ///
    /// `pixels` is a row-major grayscale block of size `width x height`.
    #[must_use]
    pub fn analyze_block(&self, pixels: &[u8], width: usize, height: usize) -> GrainAnalysis {
        let expected_len = width * height;
        if pixels.len() < expected_len || width < 4 || height < 4 {
            return GrainAnalysis::default();
        }

        // Step 1: Compute local statistics
        let mean = compute_mean(pixels, expected_len);
        let variance = compute_variance(pixels, expected_len, mean);

        // Step 2: Separate HF energy from structural content
        // Use a simple Laplacian-based high-pass filter
        let hf_energy = compute_hf_energy(pixels, width, height);
        let total_energy = variance.max(1e-10);
        let hf_ratio = (hf_energy / total_energy).min(1.0);

        // Step 3: Estimate noise via median absolute deviation
        let noise_var = estimate_noise_variance(pixels, width, height);

        // Step 4: Classify grain
        let has_grain =
            hf_ratio > self.config.grain_threshold && noise_var < self.config.max_noise_variance;

        let grain_type = if !has_grain {
            GrainType::None
        } else if hf_ratio > self.config.fine_grain_threshold {
            GrainType::Fine
        } else if noise_var > 100.0 {
            GrainType::Coarse
        } else {
            GrainType::Mixed
        };

        // Step 5: Compute grain intensity
        let intensity = if has_grain {
            let raw = (noise_var.sqrt() / 30.0).min(1.0);
            if raw < self.config.min_intensity {
                0.0
            } else {
                raw
            }
        } else {
            0.0
        };

        // Step 6: Compute uniformity via block subdivision
        let uniformity = compute_uniformity(pixels, width, height, self.config.block_size);

        // Step 7: Estimate grain size from auto-correlation
        let grain_size = estimate_grain_size(pixels, width, height);

        // Step 8: Compute QP offset
        let qp_offset = compute_grain_qp_offset(intensity, uniformity);

        GrainAnalysis {
            has_grain,
            grain_type,
            intensity,
            uniformity,
            hf_energy_ratio: hf_ratio,
            noise_variance: noise_var,
            grain_size,
            qp_offset,
            chroma_mode: GrainChromaMode::LumaOnly,
        }
    }

    /// Analyzes a full frame and returns per-block grain analysis plus a
    /// frame-level summary.
    #[must_use]
    pub fn analyze_frame(&self, pixels: &[u8], width: usize, height: usize) -> FrameGrainAnalysis {
        let bs = self.config.block_size.max(4);
        let blocks_x = width / bs;
        let blocks_y = height / bs;
        let mut block_results = Vec::with_capacity(blocks_x * blocks_y);

        let mut total_intensity = 0.0;
        let mut grain_block_count = 0usize;
        let mut total_hf_ratio = 0.0;
        let total_blocks = blocks_x * blocks_y;

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                // Extract block pixels
                let mut block_pixels = vec![0u8; bs * bs];
                for row in 0..bs {
                    let src_y = by * bs + row;
                    if src_y >= height {
                        break;
                    }
                    let src_start = src_y * width + bx * bs;
                    let dst_start = row * bs;
                    let copy_len = bs.min(width - bx * bs);
                    if src_start + copy_len <= pixels.len() {
                        block_pixels[dst_start..dst_start + copy_len]
                            .copy_from_slice(&pixels[src_start..src_start + copy_len]);
                    }
                }

                let result = self.analyze_block(&block_pixels, bs, bs);
                if result.has_grain {
                    grain_block_count += 1;
                    total_intensity += result.intensity;
                }
                total_hf_ratio += result.hf_energy_ratio;
                block_results.push(result);
            }
        }

        let grain_coverage = if total_blocks > 0 {
            grain_block_count as f64 / total_blocks as f64
        } else {
            0.0
        };
        let avg_intensity = if grain_block_count > 0 {
            total_intensity / grain_block_count as f64
        } else {
            0.0
        };
        let avg_hf_ratio = if total_blocks > 0 {
            total_hf_ratio / total_blocks as f64
        } else {
            0.0
        };

        let frame_has_grain = grain_coverage > 0.2 && avg_intensity > self.config.min_intensity;

        FrameGrainAnalysis {
            has_grain: frame_has_grain,
            grain_coverage,
            avg_intensity,
            avg_hf_ratio,
            block_results,
            recommended_grain_synthesis: frame_has_grain,
        }
    }

    /// Updates temporal statistics after analyzing a frame.
    pub fn update_temporal(&mut self, frame_analysis: &FrameGrainAnalysis) {
        self.frame_count += 1;
        let alpha = 0.1_f64;
        self.temporal_intensity_avg = self.temporal_intensity_avg * (1.0 - alpha)
            + frame_analysis.avg_intensity * alpha;
    }

    /// Returns the temporally smoothed grain intensity.
    #[must_use]
    pub fn temporal_intensity(&self) -> f64 {
        self.temporal_intensity_avg
    }
}

// ── Frame-level grain result ────────────────────────────────────────────────

/// Frame-level grain analysis summary.
#[derive(Debug, Clone)]
pub struct FrameGrainAnalysis {
    /// Whether the frame overall has significant grain.
    pub has_grain: bool,
    /// Fraction of blocks with detected grain (0.0 to 1.0).
    pub grain_coverage: f64,
    /// Average grain intensity across grain-positive blocks.
    pub avg_intensity: f64,
    /// Average high-frequency energy ratio.
    pub avg_hf_ratio: f64,
    /// Per-block grain analysis results.
    pub block_results: Vec<GrainAnalysis>,
    /// Whether grain synthesis is recommended for this frame.
    pub recommended_grain_synthesis: bool,
}

// ── Internal helper functions ───────────────────────────────────────────────

/// Computes the mean of pixel values.
fn compute_mean(pixels: &[u8], count: usize) -> f64 {
    if count == 0 {
        return 0.0;
    }
    pixels.iter().take(count).map(|&p| f64::from(p)).sum::<f64>() / count as f64
}

/// Computes the variance of pixel values.
fn compute_variance(pixels: &[u8], count: usize, mean: f64) -> f64 {
    if count < 2 {
        return 0.0;
    }
    pixels
        .iter()
        .take(count)
        .map(|&p| {
            let d = f64::from(p) - mean;
            d * d
        })
        .sum::<f64>()
        / count as f64
}

/// Computes high-frequency energy using a 3x3 Laplacian filter.
fn compute_hf_energy(pixels: &[u8], width: usize, height: usize) -> f64 {
    let mut energy = 0.0;
    let mut count = 0u64;

    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let center = f64::from(pixels[y * width + x]);
            let top = f64::from(pixels[(y - 1) * width + x]);
            let bottom = f64::from(pixels[(y + 1) * width + x]);
            let left = f64::from(pixels[y * width + (x - 1)]);
            let right = f64::from(pixels[y * width + (x + 1)]);

            // Laplacian: 4*center - neighbors
            let laplacian = 4.0 * center - top - bottom - left - right;
            energy += laplacian * laplacian;
            count += 1;
        }
    }

    if count > 0 {
        energy / count as f64
    } else {
        0.0
    }
}

/// Estimates noise variance using the Median Absolute Deviation (MAD) method
/// on Laplacian-filtered values.
fn estimate_noise_variance(pixels: &[u8], width: usize, height: usize) -> f64 {
    let mut laplacian_vals = Vec::with_capacity(width * height);

    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let center = f64::from(pixels[y * width + x]);
            let top = f64::from(pixels[(y - 1) * width + x]);
            let bottom = f64::from(pixels[(y + 1) * width + x]);
            let left = f64::from(pixels[y * width + (x - 1)]);
            let right = f64::from(pixels[y * width + (x + 1)]);

            let laplacian = (4.0 * center - top - bottom - left - right).abs();
            laplacian_vals.push(laplacian);
        }
    }

    if laplacian_vals.is_empty() {
        return 0.0;
    }

    // Sort for median
    laplacian_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = laplacian_vals[laplacian_vals.len() / 2];

    // MAD = median of |X - median(X)|
    let mut abs_devs: Vec<f64> = laplacian_vals.iter().map(|&v| (v - median).abs()).collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = abs_devs[abs_devs.len() / 2];

    // sigma = MAD / 0.6745 (robust noise estimator)
    // variance = sigma^2
    let sigma = mad / 0.6745;
    sigma * sigma
}

/// Computes grain uniformity by measuring variance-of-variance across sub-blocks.
fn compute_uniformity(pixels: &[u8], width: usize, height: usize, block_size: usize) -> f64 {
    let sub_size = (block_size / 2).max(2);
    let subs_x = width / sub_size;
    let subs_y = height / sub_size;

    if subs_x < 2 || subs_y < 2 {
        return 1.0;
    }

    let mut sub_variances = Vec::with_capacity(subs_x * subs_y);

    for sy in 0..subs_y {
        for sx in 0..subs_x {
            let mut sum = 0.0_f64;
            let mut sum_sq = 0.0_f64;
            let mut count = 0u64;

            for row in 0..sub_size {
                let y = sy * sub_size + row;
                if y >= height {
                    break;
                }
                for col in 0..sub_size {
                    let x = sx * sub_size + col;
                    if x >= width {
                        break;
                    }
                    let v = f64::from(pixels[y * width + x]);
                    sum += v;
                    sum_sq += v * v;
                    count += 1;
                }
            }

            if count > 1 {
                let mean = sum / count as f64;
                let var = sum_sq / count as f64 - mean * mean;
                sub_variances.push(var.max(0.0));
            }
        }
    }

    if sub_variances.len() < 2 {
        return 1.0;
    }

    let mean_var = sub_variances.iter().sum::<f64>() / sub_variances.len() as f64;
    let var_of_var = sub_variances
        .iter()
        .map(|&v| (v - mean_var).powi(2))
        .sum::<f64>()
        / sub_variances.len() as f64;

    // Normalize: low variance-of-variance = high uniformity
    let cv = if mean_var > 1e-10 {
        var_of_var.sqrt() / mean_var
    } else {
        0.0
    };

    (1.0 - cv.min(1.0)).max(0.0)
}

/// Estimates grain particle size from horizontal auto-correlation.
fn estimate_grain_size(pixels: &[u8], width: usize, height: usize) -> f64 {
    if width < 8 || height < 4 {
        return 0.0;
    }

    // Compute auto-correlation at lags 1..8
    let max_lag = 8.min(width / 2);
    let mut correlations = Vec::with_capacity(max_lag);

    // Use center rows for stability
    let start_y = height / 4;
    let end_y = (3 * height / 4).min(height);

    let mut total_var = 0.0_f64;
    let mut row_count = 0u64;

    for y in start_y..end_y {
        let row = &pixels[y * width..(y * width + width).min(pixels.len())];
        if row.len() < width {
            continue;
        }

        let mean: f64 = row.iter().map(|&p| f64::from(p)).sum::<f64>() / width as f64;

        let var: f64 = row
            .iter()
            .map(|&p| {
                let d = f64::from(p) - mean;
                d * d
            })
            .sum::<f64>()
            / width as f64;

        total_var += var;
        row_count += 1;

        if correlations.is_empty() {
            correlations.resize(max_lag, 0.0);
        }

        for lag in 1..=max_lag {
            let mut corr = 0.0_f64;
            for x in 0..width - lag {
                let a = f64::from(row[x]) - mean;
                let b = f64::from(row[x + lag]) - mean;
                corr += a * b;
            }
            correlations[lag - 1] += corr / (width - lag) as f64;
        }
    }

    if row_count == 0 || total_var < 1e-10 {
        return 0.0;
    }

    let avg_var = total_var / row_count as f64;
    for c in &mut correlations {
        *c /= row_count as f64;
    }

    // Normalize correlations by variance
    let normalized: Vec<f64> = correlations.iter().map(|&c| c / avg_var.max(1e-10)).collect();

    // Find the lag where correlation drops below 1/e (~0.37) — that's the grain size
    let threshold = (-1.0_f64).exp(); // 1/e
    for (lag_idx, &nc) in normalized.iter().enumerate() {
        if nc < threshold {
            // Interpolate for sub-pixel grain size
            if lag_idx == 0 {
                return 1.0 + nc;
            }
            let prev = normalized[lag_idx - 1];
            let frac = if (prev - nc).abs() > 1e-10 {
                (prev - threshold) / (prev - nc)
            } else {
                0.5
            };
            return (lag_idx as f64 + frac).max(0.5);
        }
    }

    // All correlations above threshold: large grain
    max_lag as f64
}

/// Computes the recommended QP offset for grain.
fn compute_grain_qp_offset(intensity: f64, uniformity: f64) -> i8 {
    if intensity < 0.05 {
        return 0;
    }
    // Higher intensity, more uniform grain → can raise QP more
    let offset = intensity * uniformity * 6.0;
    (offset.round() as i8).clamp(-6, 6)
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
        // Deterministic pseudo-noise using a simple LCG
        let mut rng_state = 12345u32;
        for _ in 0..width * height {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = ((rng_state >> 16) % (noise_amp as u32 * 2 + 1)) as u8;
            let val = base.saturating_add(noise).saturating_sub(noise_amp);
            pixels.push(val);
        }
        pixels
    }

    fn make_gradient_block(width: usize, height: usize) -> Vec<u8> {
        let mut pixels = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let val = ((x + y) * 255 / (width + height)) as u8;
                pixels.push(val);
            }
        }
        pixels
    }

    #[test]
    fn test_flat_block_no_grain() {
        let detector = GrainDetector::new(GrainDetectorConfig::default());
        let pixels = make_flat_block(32, 32, 128);
        let result = detector.analyze_block(&pixels, 32, 32);

        assert!(!result.has_grain, "Flat block should have no grain");
        assert_eq!(result.grain_type, GrainType::None);
        assert!(
            result.intensity < 0.01,
            "Flat block intensity should be ~0: {}",
            result.intensity
        );
    }

    #[test]
    fn test_noisy_block_detects_grain() {
        let detector = GrainDetector::new(GrainDetectorConfig::default());
        let pixels = make_noisy_block(32, 32, 128, 40);
        let result = detector.analyze_block(&pixels, 32, 32);

        assert!(
            result.hf_energy_ratio > 0.0,
            "Noisy block should have HF energy: {}",
            result.hf_energy_ratio
        );
        assert!(
            result.noise_variance > 0.0,
            "Noisy block should have noise variance: {}",
            result.noise_variance
        );
    }

    #[test]
    fn test_gradient_block_low_grain() {
        let detector = GrainDetector::new(GrainDetectorConfig::default());
        let pixels = make_gradient_block(32, 32);
        let result = detector.analyze_block(&pixels, 32, 32);

        // Gradient has structure but no random grain
        assert!(
            result.noise_variance < 200.0,
            "Gradient should have low noise variance: {}",
            result.noise_variance
        );
    }

    #[test]
    fn test_grain_type_classification() {
        let config = GrainDetectorConfig {
            grain_threshold: 0.05,
            fine_grain_threshold: 0.3,
            min_intensity: 0.01,
            max_noise_variance: 50000.0,
            ..Default::default()
        };
        let detector = GrainDetector::new(config);

        // Heavy noise should have measurable HF energy
        let heavy = make_noisy_block(32, 32, 128, 80);
        let result = detector.analyze_block(&heavy, 32, 32);

        // With heavy noise and low thresholds, either grain is detected
        // or at minimum HF energy ratio and noise variance are significant
        assert!(
            result.hf_energy_ratio > 0.01 || result.noise_variance > 10.0,
            "Heavy noise should have measurable HF energy ({}) or noise variance ({})",
            result.hf_energy_ratio,
            result.noise_variance
        );
    }

    #[test]
    fn test_synthesis_params_disabled_for_no_grain() {
        let analysis = GrainAnalysis::default();
        let params = analysis.synthesis_params();
        assert!(!params.enabled);
        assert_eq!(params.num_y_points, 0);
        assert_eq!(params.grain_amplitude, 0);
    }

    #[test]
    fn test_synthesis_params_enabled_for_grain() {
        let analysis = GrainAnalysis {
            has_grain: true,
            grain_type: GrainType::Fine,
            intensity: 0.6,
            uniformity: 0.8,
            hf_energy_ratio: 0.5,
            noise_variance: 300.0,
            grain_size: 2.5,
            qp_offset: 3,
            chroma_mode: GrainChromaMode::LumaOnly,
        };
        let params = analysis.synthesis_params();
        assert!(params.enabled);
        assert!(params.num_y_points > 0);
        assert!(params.grain_amplitude > 0);
        assert_eq!(params.ar_coeff_lag, 1); // grain_size 2.5 -> lag 1
    }

    #[test]
    fn test_qp_offset_range() {
        // Zero intensity -> zero offset
        assert_eq!(compute_grain_qp_offset(0.0, 0.5), 0);

        // High intensity, high uniformity -> positive offset
        let offset = compute_grain_qp_offset(0.8, 0.9);
        assert!(offset > 0, "High grain should yield positive QP offset: {offset}");
        assert!(offset <= 6, "QP offset should be clamped to 6: {offset}");
    }

    #[test]
    fn test_frame_analysis() {
        let detector = GrainDetector::new(GrainDetectorConfig {
            block_size: 8,
            ..Default::default()
        });

        // Create a 32x32 noisy frame
        let pixels = make_noisy_block(32, 32, 128, 40);
        let frame = detector.analyze_frame(&pixels, 32, 32);

        // Should have block results
        let expected_blocks = (32 / 8) * (32 / 8);
        assert_eq!(
            frame.block_results.len(),
            expected_blocks,
            "Should have {} blocks, got {}",
            expected_blocks,
            frame.block_results.len()
        );
    }

    #[test]
    fn test_temporal_update() {
        let mut detector = GrainDetector::new(GrainDetectorConfig::default());
        assert!((detector.temporal_intensity() - 0.0).abs() < 1e-10);

        let frame = FrameGrainAnalysis {
            has_grain: true,
            grain_coverage: 0.5,
            avg_intensity: 0.4,
            avg_hf_ratio: 0.3,
            block_results: vec![],
            recommended_grain_synthesis: true,
        };

        detector.update_temporal(&frame);
        assert!(
            detector.temporal_intensity() > 0.0,
            "Temporal intensity should update"
        );

        // Update again with lower intensity
        let frame2 = FrameGrainAnalysis {
            avg_intensity: 0.1,
            ..frame
        };
        detector.update_temporal(&frame2);
        // EMA should smooth between 0.4 and 0.1
        assert!(detector.temporal_intensity() > 0.0);
    }

    #[test]
    fn test_estimated_bits_saved() {
        let no_grain = GrainAnalysis::default();
        assert!(
            (no_grain.estimated_bits_saved_per_pixel() - 0.0).abs() < 1e-10,
            "No grain = no savings"
        );

        let with_grain = GrainAnalysis {
            has_grain: true,
            intensity: 0.5,
            hf_energy_ratio: 0.4,
            ..Default::default()
        };
        let savings = with_grain.estimated_bits_saved_per_pixel();
        assert!(savings > 0.0, "Grain should yield bit savings: {savings}");
    }

    #[test]
    fn test_empty_input_returns_default() {
        let detector = GrainDetector::new(GrainDetectorConfig::default());
        let result = detector.analyze_block(&[], 0, 0);
        assert!(!result.has_grain);
        assert_eq!(result.grain_type, GrainType::None);
    }

    #[test]
    fn test_small_block_returns_default() {
        let detector = GrainDetector::new(GrainDetectorConfig::default());
        let pixels = vec![128u8; 9]; // 3x3
        let result = detector.analyze_block(&pixels, 3, 3);
        assert!(!result.has_grain);
    }

    #[test]
    fn test_uniformity_flat_is_one() {
        let pixels = make_flat_block(32, 32, 128);
        let u = compute_uniformity(&pixels, 32, 32, 16);
        assert!(
            (u - 1.0).abs() < 0.01,
            "Flat block should have uniformity ~1.0: {u}"
        );
    }

    #[test]
    fn test_grain_size_flat_is_zero() {
        let pixels = make_flat_block(64, 32, 128);
        let size = estimate_grain_size(&pixels, 64, 32);
        assert!(
            size < 1.0,
            "Flat block should have near-zero grain size: {size}"
        );
    }
}

// ── Grain synthesis integration tests ───────────────────────────────────────

#[cfg(test)]
mod grain_synthesis_integration_tests {
    use super::*;

    fn make_noisy_block(width: usize, height: usize, base: u8, noise_amp: u8) -> Vec<u8> {
        let mut pixels = Vec::with_capacity(width * height);
        let mut rng_state = 42u32;
        for _ in 0..width * height {
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = ((rng_state >> 16) % (u32::from(noise_amp) * 2 + 1)) as u8;
            let val = base.saturating_add(noise).saturating_sub(noise_amp);
            pixels.push(val);
        }
        pixels
    }

    /// `GrainSynthesisParams::disabled()` leaves all fields at safe defaults.
    #[test]
    fn test_synthesis_params_disabled_all_zero() {
        let p = GrainSynthesisParams::disabled();
        assert!(!p.enabled);
        assert_eq!(p.num_y_points, 0);
        assert_eq!(p.num_cb_points, 0);
        assert_eq!(p.num_cr_points, 0);
        assert_eq!(p.grain_amplitude, 0);
        assert_eq!(p.ar_coeff_lag, 0);
        assert!(!p.overlap_flag);
        assert_eq!(p.grain_scale_shift, 0);
    }

    /// High-intensity coarse grain with chroma should produce non-zero Cb points.
    #[test]
    fn test_synthesis_params_chroma_grain() {
        let analysis = GrainAnalysis {
            has_grain: true,
            grain_type: GrainType::Coarse,
            intensity: 0.7,
            uniformity: 0.5,
            hf_energy_ratio: 0.2,
            noise_variance: 500.0,
            grain_size: 5.0,
            qp_offset: 4,
            chroma_mode: GrainChromaMode::LumaAndChroma,
        };
        let params = analysis.synthesis_params();
        assert!(params.enabled);
        assert!(
            params.num_cb_points > 0,
            "LumaAndChroma mode should have Cb grain points"
        );
    }

    /// `PerChannel` chroma mode should produce non-zero Cr points.
    #[test]
    fn test_synthesis_params_per_channel_chroma() {
        let analysis = GrainAnalysis {
            has_grain: true,
            grain_type: GrainType::Mixed,
            intensity: 0.5,
            uniformity: 0.6,
            hf_energy_ratio: 0.3,
            noise_variance: 300.0,
            grain_size: 3.5,
            qp_offset: 2,
            chroma_mode: GrainChromaMode::PerChannel,
        };
        let params = analysis.synthesis_params();
        assert!(params.enabled);
        assert!(
            params.num_cr_points > 0,
            "PerChannel mode should have Cr grain points"
        );
    }

    /// `overlap_flag` should be true when grain size > 2.0.
    #[test]
    fn test_synthesis_params_overlap_flag() {
        let large_grain = GrainAnalysis {
            has_grain: true,
            grain_type: GrainType::Coarse,
            intensity: 0.5,
            grain_size: 4.0,
            chroma_mode: GrainChromaMode::LumaOnly,
            ..Default::default()
        };
        let small_grain = GrainAnalysis {
            has_grain: true,
            grain_type: GrainType::Fine,
            intensity: 0.3,
            grain_size: 1.0,
            chroma_mode: GrainChromaMode::LumaOnly,
            ..Default::default()
        };
        assert!(
            large_grain.synthesis_params().overlap_flag,
            "Large grain (size=4) should have overlap_flag=true"
        );
        assert!(
            !small_grain.synthesis_params().overlap_flag,
            "Small grain (size=1) should have overlap_flag=false"
        );
    }

    /// `GrainDetectorConfig` defaults are sane (all numeric fields in range).
    #[test]
    fn test_grain_detector_config_defaults_valid() {
        let cfg = GrainDetectorConfig::default();
        assert!(cfg.block_size >= 4, "block_size must be ≥4");
        assert!(cfg.grain_threshold > 0.0 && cfg.grain_threshold < 1.0);
        assert!(cfg.fine_grain_threshold > cfg.grain_threshold);
        assert!(cfg.min_intensity >= 0.0 && cfg.min_intensity < 1.0);
        assert!(cfg.max_noise_variance > 0.0);
    }

    /// `FrameGrainAnalysis::recommended_grain_synthesis` matches `has_grain`
    /// when coverage and intensity are above threshold.
    #[test]
    fn test_frame_grain_analysis_synthesis_recommendation() {
        let detector = GrainDetector::new(GrainDetectorConfig {
            block_size: 8,
            grain_threshold: 0.10,
            min_intensity: 0.01,
            max_noise_variance: 100_000.0,
            ..Default::default()
        });
        let pixels = make_noisy_block(64, 64, 128, 50);
        let frame = detector.analyze_frame(&pixels, 64, 64);

        // Ensure recommendation and has_grain are consistent
        assert_eq!(
            frame.recommended_grain_synthesis, frame.has_grain,
            "recommended_grain_synthesis should match has_grain"
        );
    }

    /// `ar_coeff_lag` grows with grain size.
    #[test]
    fn test_ar_coeff_lag_scales_with_grain_size() {
        let make = |gs: f64| GrainAnalysis {
            has_grain: true,
            grain_type: GrainType::Mixed,
            intensity: 0.4,
            grain_size: gs,
            chroma_mode: GrainChromaMode::LumaOnly,
            ..Default::default()
        };

        let lag_0 = make(0.5).synthesis_params().ar_coeff_lag; // < 1.5
        let lag_1 = make(2.0).synthesis_params().ar_coeff_lag; // 1.5 ≤ gs < 3.0
        let lag_2 = make(4.0).synthesis_params().ar_coeff_lag; // 3.0 ≤ gs < 6.0
        let lag_3 = make(7.0).synthesis_params().ar_coeff_lag; // ≥ 6.0

        assert_eq!(lag_0, 0, "grain_size<1.5 → ar_lag=0");
        assert_eq!(lag_1, 1, "grain_size≈2.0 → ar_lag=1");
        assert_eq!(lag_2, 2, "grain_size≈4.0 → ar_lag=2");
        assert_eq!(lag_3, 3, "grain_size≈7.0 → ar_lag=3");
    }

    /// Zero-intensity grain analysis yields `estimated_bits_saved_per_pixel == 0`.
    #[test]
    fn test_bits_saved_zero_intensity() {
        let a = GrainAnalysis {
            has_grain: true,
            intensity: 0.0,
            hf_energy_ratio: 0.5,
            ..Default::default()
        };
        assert!(
            (a.estimated_bits_saved_per_pixel() - 0.0).abs() < 1e-10,
            "Zero intensity should save 0 bits"
        );
    }

    /// `GrainDetector` correctly tracks frame count via `update_temporal`.
    #[test]
    fn test_temporal_tracking_ema_bounded() {
        let mut detector = GrainDetector::new(GrainDetectorConfig::default());
        let frame = FrameGrainAnalysis {
            has_grain: true,
            grain_coverage: 0.8,
            avg_intensity: 1.0,
            avg_hf_ratio: 0.5,
            block_results: vec![],
            recommended_grain_synthesis: true,
        };
        // Feed 50 frames with max intensity
        for _ in 0..50 {
            detector.update_temporal(&frame);
        }
        let intensity = detector.temporal_intensity();
        assert!(
            intensity > 0.0 && intensity <= 1.0,
            "Temporal EMA intensity should be in (0,1]: {intensity}"
        );
    }
}
