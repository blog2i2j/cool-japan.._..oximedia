#![allow(dead_code)]
//! Noise profiling and estimation for video frames.
//!
//! This module provides tools for estimating noise levels in video content, which
//! is critical for:
//!
//! - **Denoising parameter tuning** - Automatically select filter strength
//! - **Quality assessment** - Detect noisy source material vs compression noise
//! - **Encoding optimization** - Pre-filter noisy content for better compression
//! - **Sensor characterization** - Profile camera noise at different ISO/gain levels
//!
//! The module implements two noise estimation approaches:
//! 1. **Laplacian-based** - Uses the median absolute deviation of high-pass filtered pixels
//! 2. **Block variance** - Finds the quietest blocks and uses their variance as the noise floor

use std::collections::VecDeque;

/// Configuration for noise profiling.
#[derive(Debug, Clone)]
pub struct NoiseProfileConfig {
    /// Block size for block-variance estimation.
    pub block_size: usize,
    /// Percentile of block variances to use as noise estimate (0.0-1.0).
    /// Lower percentile picks quieter blocks (more conservative).
    pub variance_percentile: f64,
    /// Temporal window size for tracking noise over time.
    pub temporal_window: usize,
    /// Whether to compute per-channel noise estimates (requires UV planes).
    pub per_channel: bool,
}

impl Default for NoiseProfileConfig {
    fn default() -> Self {
        Self {
            block_size: 8,
            variance_percentile: 0.10,
            temporal_window: 30,
            per_channel: false,
        }
    }
}

/// Noise estimate for a single frame.
#[derive(Debug, Clone)]
pub struct NoiseEstimate {
    /// Frame index.
    pub frame_index: usize,
    /// Laplacian-based noise standard deviation (luma).
    pub laplacian_sigma: f64,
    /// Block-variance-based noise standard deviation (luma).
    pub block_sigma: f64,
    /// Combined noise estimate (weighted average of both methods).
    pub combined_sigma: f64,
    /// Signal-to-noise ratio estimate in dB.
    pub snr_db: f64,
    /// Per-channel noise sigmas [Y, U, V] if `per_channel` is enabled.
    pub channel_sigmas: Option<[f64; 3]>,
}

/// Classification of noise level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseLevel {
    /// Very clean signal (sigma < 2.0).
    VeryClean,
    /// Clean signal (sigma < 5.0).
    Clean,
    /// Moderate noise (sigma < 12.0).
    Moderate,
    /// Noisy signal (sigma < 25.0).
    Noisy,
    /// Very noisy (sigma >= 25.0).
    VeryNoisy,
}

impl std::fmt::Display for NoiseLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            Self::VeryClean => "very_clean",
            Self::Clean => "clean",
            Self::Moderate => "moderate",
            Self::Noisy => "noisy",
            Self::VeryNoisy => "very_noisy",
        };
        write!(f, "{label}")
    }
}

/// Temporal noise statistics.
#[derive(Debug, Clone)]
pub struct NoiseStats {
    /// Mean noise sigma over the window.
    pub mean_sigma: f64,
    /// Standard deviation of noise sigma over the window.
    pub std_sigma: f64,
    /// Minimum noise sigma in the window.
    pub min_sigma: f64,
    /// Maximum noise sigma in the window.
    pub max_sigma: f64,
    /// Noise classification based on mean sigma.
    pub level: NoiseLevel,
    /// Number of frames analyzed.
    pub frame_count: usize,
}

/// Noise profiler that analyzes video frames.
pub struct NoiseProfiler {
    /// Configuration.
    config: NoiseProfileConfig,
    /// History of combined sigmas for temporal analysis.
    sigma_history: VecDeque<f64>,
    /// Total frames processed.
    frame_count: usize,
}

impl NoiseProfiler {
    /// Create a new noise profiler with default configuration.
    pub fn new() -> Self {
        Self::with_config(NoiseProfileConfig::default())
    }

    /// Create a new noise profiler with custom configuration.
    pub fn with_config(config: NoiseProfileConfig) -> Self {
        Self {
            sigma_history: VecDeque::with_capacity(config.temporal_window),
            config,
            frame_count: 0,
        }
    }

    /// Estimate noise using the Laplacian method.
    ///
    /// Applies a 3x3 Laplacian kernel and computes the median absolute deviation
    /// of the response. The robust median-based approach is less sensitive to
    /// actual edges in the image.
    #[allow(clippy::cast_precision_loss)]
    fn laplacian_noise_estimate(y_plane: &[u8], width: usize, height: usize) -> f64 {
        if width < 3 || height < 3 {
            return 0.0;
        }

        // Laplacian kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        let mut responses: Vec<f64> = Vec::with_capacity((width - 2) * (height - 2));

        for py in 1..height - 1 {
            for px in 1..width - 1 {
                let center = f64::from(y_plane[py * width + px]);
                let top = f64::from(y_plane[(py - 1) * width + px]);
                let bottom = f64::from(y_plane[(py + 1) * width + px]);
                let left = f64::from(y_plane[py * width + px - 1]);
                let right = f64::from(y_plane[py * width + px + 1]);

                let laplacian = top + bottom + left + right - 4.0 * center;
                responses.push(laplacian.abs());
            }
        }

        if responses.is_empty() {
            return 0.0;
        }

        // Median absolute deviation
        responses.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = responses[responses.len() / 2];

        // The MAD-based noise estimate for Laplacian:
        // sigma = median / (sqrt(2) * erfinv(0.5)) ≈ median * 1.4826 / sqrt(20)
        // Simplified: sigma ≈ median * 1.4826 / 4.4721 ≈ median * 0.3314
        // But for Laplacian specifically, the scaling is: sigma ≈ median * sqrt(pi/2) / 6
        // A practical constant is ~0.37
        median * 0.37
    }

    /// Estimate noise using the block variance method.
    ///
    /// Divides the frame into blocks, computes variance of each, and uses the
    /// low-percentile variance as the noise floor (assuming quietest blocks are
    /// flat regions dominated by noise).
    #[allow(clippy::cast_precision_loss)]
    fn block_variance_estimate(
        y_plane: &[u8],
        width: usize,
        height: usize,
        block_size: usize,
        percentile: f64,
    ) -> f64 {
        if block_size == 0 || width < block_size || height < block_size {
            return 0.0;
        }

        let blocks_x = width / block_size;
        let blocks_y = height / block_size;
        if blocks_x == 0 || blocks_y == 0 {
            return 0.0;
        }

        let mut variances: Vec<f64> = Vec::with_capacity(blocks_x * blocks_y);

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let mut sum = 0.0_f64;
                let mut sum_sq = 0.0_f64;
                let count = (block_size * block_size) as f64;

                for dy in 0..block_size {
                    for dx in 0..block_size {
                        let val = f64::from(
                            y_plane[(by * block_size + dy) * width + bx * block_size + dx],
                        );
                        sum += val;
                        sum_sq += val * val;
                    }
                }

                let mean = sum / count;
                let variance = (sum_sq / count - mean * mean).max(0.0);
                variances.push(variance);
            }
        }

        if variances.is_empty() {
            return 0.0;
        }

        variances.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((variances.len() as f64 * percentile) as usize).min(variances.len() - 1);
        variances[idx].sqrt()
    }

    /// Estimate noise for a single channel plane.
    #[allow(clippy::cast_precision_loss)]
    fn estimate_channel(&self, plane: &[u8], width: usize, height: usize) -> f64 {
        let lap = Self::laplacian_noise_estimate(plane, width, height);
        let blk = Self::block_variance_estimate(
            plane,
            width,
            height,
            self.config.block_size,
            self.config.variance_percentile,
        );
        0.5 * lap + 0.5 * blk
    }

    /// Process a video frame and return noise estimates.
    ///
    /// Provide `u_plane` and `v_plane` only if `per_channel` is enabled and
    /// chroma planes are available. Otherwise pass empty slices.
    #[allow(clippy::cast_precision_loss)]
    pub fn process_frame(
        &mut self,
        y_plane: &[u8],
        u_plane: &[u8],
        v_plane: &[u8],
        width: usize,
        height: usize,
        frame_index: usize,
    ) -> NoiseEstimate {
        let laplacian_sigma = Self::laplacian_noise_estimate(y_plane, width, height);
        let block_sigma = Self::block_variance_estimate(
            y_plane,
            width,
            height,
            self.config.block_size,
            self.config.variance_percentile,
        );

        let combined_sigma = 0.5 * laplacian_sigma + 0.5 * block_sigma;

        // Compute SNR: signal power / noise power
        let signal_mean = if y_plane.is_empty() {
            0.0
        } else {
            let sum: u64 = y_plane.iter().map(|&v| u64::from(v)).sum();
            sum as f64 / y_plane.len() as f64
        };
        let snr_db = if combined_sigma > 0.001 {
            20.0 * (signal_mean / combined_sigma).log10()
        } else {
            100.0 // effectively infinite SNR
        };

        let channel_sigmas = if self.config.per_channel {
            let chroma_w = width / 2;
            let chroma_h = height / 2;
            let u_sigma = if u_plane.len() >= chroma_w * chroma_h && chroma_w > 0 && chroma_h > 0 {
                self.estimate_channel(u_plane, chroma_w, chroma_h)
            } else {
                0.0
            };
            let v_sigma = if v_plane.len() >= chroma_w * chroma_h && chroma_w > 0 && chroma_h > 0 {
                self.estimate_channel(v_plane, chroma_w, chroma_h)
            } else {
                0.0
            };
            Some([combined_sigma, u_sigma, v_sigma])
        } else {
            None
        };

        // Update temporal buffer
        if self.sigma_history.len() >= self.config.temporal_window {
            self.sigma_history.pop_front();
        }
        self.sigma_history.push_back(combined_sigma);
        self.frame_count += 1;

        NoiseEstimate {
            frame_index,
            laplacian_sigma,
            block_sigma,
            combined_sigma,
            snr_db,
            channel_sigmas,
        }
    }

    /// Get temporal noise statistics.
    #[allow(clippy::cast_precision_loss)]
    pub fn get_stats(&self) -> NoiseStats {
        let n = self.sigma_history.len();
        if n == 0 {
            return NoiseStats {
                mean_sigma: 0.0,
                std_sigma: 0.0,
                min_sigma: 0.0,
                max_sigma: 0.0,
                level: NoiseLevel::VeryClean,
                frame_count: 0,
            };
        }

        let sum: f64 = self.sigma_history.iter().sum();
        let mean = sum / n as f64;
        let variance: f64 = self
            .sigma_history
            .iter()
            .map(|&s| (s - mean) * (s - mean))
            .sum::<f64>()
            / n as f64;
        let std = variance.sqrt();

        let min = self
            .sigma_history
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let max = self
            .sigma_history
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        let level = classify_noise(mean);

        NoiseStats {
            mean_sigma: mean,
            std_sigma: std,
            min_sigma: min,
            max_sigma: max,
            level,
            frame_count: self.frame_count,
        }
    }

    /// Reset the profiler state.
    pub fn reset(&mut self) {
        self.sigma_history.clear();
        self.frame_count = 0;
    }
}

/// Classify noise level from a sigma value.
pub fn classify_noise(sigma: f64) -> NoiseLevel {
    if sigma < 2.0 {
        NoiseLevel::VeryClean
    } else if sigma < 5.0 {
        NoiseLevel::Clean
    } else if sigma < 12.0 {
        NoiseLevel::Moderate
    } else if sigma < 25.0 {
        NoiseLevel::Noisy
    } else {
        NoiseLevel::VeryNoisy
    }
}

/// Suggest a denoising strength based on the noise estimate.
///
/// Returns a value from 0.0 (no denoising) to 1.0 (maximum denoising).
pub fn suggest_denoise_strength(sigma: f64) -> f64 {
    // Sigmoid-like mapping: tanh(sigma / 20)
    (sigma / 20.0).tanh()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_flat_frame(width: usize, height: usize, value: u8) -> Vec<u8> {
        vec![value; width * height]
    }

    fn make_noisy_frame(width: usize, height: usize, base: u8, noise_amplitude: u8) -> Vec<u8> {
        // Deterministic pseudo-noise using a simple LCG
        let mut data = vec![0u8; width * height];
        let mut seed: u32 = 12345;
        for pixel in &mut data {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = ((seed >> 16) & 0xFF) as u8 % (noise_amplitude.saturating_mul(2).max(1));
            let val = base as i32 + noise as i32 - noise_amplitude as i32;
            *pixel = val.clamp(0, 255) as u8;
        }
        data
    }

    #[test]
    fn test_flat_frame_low_noise() {
        let mut profiler = NoiseProfiler::new();
        let frame = make_flat_frame(64, 64, 128);
        let est = profiler.process_frame(&frame, &[], &[], 64, 64, 0);
        assert!(
            est.combined_sigma < 1.0,
            "flat frame should have very low noise: {}",
            est.combined_sigma
        );
    }

    #[test]
    fn test_noisy_frame_higher_sigma() {
        let mut profiler = NoiseProfiler::new();
        let frame = make_noisy_frame(64, 64, 128, 30);
        let est = profiler.process_frame(&frame, &[], &[], 64, 64, 0);
        assert!(
            est.combined_sigma > 1.0,
            "noisy frame should have significant sigma: {}",
            est.combined_sigma
        );
    }

    #[test]
    fn test_laplacian_flat() {
        let frame = make_flat_frame(32, 32, 100);
        let sigma = NoiseProfiler::laplacian_noise_estimate(&frame, 32, 32);
        assert!(
            sigma < 0.01,
            "flat frame laplacian should be near zero: {}",
            sigma
        );
    }

    #[test]
    fn test_block_variance_flat() {
        let frame = make_flat_frame(64, 64, 100);
        let sigma = NoiseProfiler::block_variance_estimate(&frame, 64, 64, 8, 0.1);
        assert!(
            sigma < 0.01,
            "flat frame block variance should be near zero: {}",
            sigma
        );
    }

    #[test]
    fn test_block_variance_noisy() {
        let frame = make_noisy_frame(64, 64, 128, 20);
        let sigma = NoiseProfiler::block_variance_estimate(&frame, 64, 64, 8, 0.1);
        assert!(
            sigma > 0.5,
            "noisy frame should produce measurable sigma: {}",
            sigma
        );
    }

    #[test]
    fn test_classify_noise_levels() {
        assert_eq!(classify_noise(0.5), NoiseLevel::VeryClean);
        assert_eq!(classify_noise(3.0), NoiseLevel::Clean);
        assert_eq!(classify_noise(8.0), NoiseLevel::Moderate);
        assert_eq!(classify_noise(18.0), NoiseLevel::Noisy);
        assert_eq!(classify_noise(30.0), NoiseLevel::VeryNoisy);
    }

    #[test]
    fn test_noise_level_display() {
        assert_eq!(format!("{}", NoiseLevel::VeryClean), "very_clean");
        assert_eq!(format!("{}", NoiseLevel::Moderate), "moderate");
    }

    #[test]
    fn test_suggest_denoise_strength() {
        let low = suggest_denoise_strength(0.0);
        let mid = suggest_denoise_strength(10.0);
        let high = suggest_denoise_strength(50.0);
        assert!(low < mid);
        assert!(mid < high);
        assert!(low >= 0.0);
        assert!(high <= 1.0);
    }

    #[test]
    fn test_snr_db_clean_signal() {
        let mut profiler = NoiseProfiler::new();
        let frame = make_flat_frame(64, 64, 128);
        let est = profiler.process_frame(&frame, &[], &[], 64, 64, 0);
        // Very clean signal => very high SNR
        assert!(
            est.snr_db > 40.0,
            "clean signal SNR should be high: {}",
            est.snr_db
        );
    }

    #[test]
    fn test_temporal_stats() {
        let mut profiler = NoiseProfiler::new();
        for i in 0..10 {
            let frame = make_flat_frame(32, 32, 128);
            profiler.process_frame(&frame, &[], &[], 32, 32, i);
        }
        let stats = profiler.get_stats();
        assert_eq!(stats.frame_count, 10);
        assert!(stats.mean_sigma < 1.0);
        assert_eq!(stats.level, NoiseLevel::VeryClean);
    }

    #[test]
    fn test_per_channel_noise() {
        let config = NoiseProfileConfig {
            per_channel: true,
            ..Default::default()
        };
        let mut profiler = NoiseProfiler::with_config(config);
        let y = make_flat_frame(64, 64, 128);
        let u = make_flat_frame(32, 32, 128);
        let v = make_flat_frame(32, 32, 128);
        let est = profiler.process_frame(&y, &u, &v, 64, 64, 0);
        assert!(est.channel_sigmas.is_some());
        let ch = est
            .channel_sigmas
            .expect("expected channel_sigmas to be Some/Ok");
        assert!(ch[1] < 1.0, "clean U plane should have low sigma");
        assert!(ch[2] < 1.0, "clean V plane should have low sigma");
    }

    #[test]
    fn test_reset_profiler() {
        let mut profiler = NoiseProfiler::new();
        let frame = make_flat_frame(32, 32, 128);
        profiler.process_frame(&frame, &[], &[], 32, 32, 0);
        assert_eq!(profiler.frame_count, 1);
        profiler.reset();
        assert_eq!(profiler.frame_count, 0);
        let stats = profiler.get_stats();
        assert_eq!(stats.frame_count, 0);
    }

    #[test]
    fn test_small_frame_edge_case() {
        let mut profiler = NoiseProfiler::new();
        let frame = vec![128u8; 4]; // 2x2 frame
        let est = profiler.process_frame(&frame, &[], &[], 2, 2, 0);
        // Should not crash, returns zero
        assert_eq!(est.laplacian_sigma, 0.0);
    }
}
