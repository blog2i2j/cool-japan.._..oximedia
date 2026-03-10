//! Noise estimation from raw video frames.
//!
//! Provides methods to estimate noise characteristics directly from pixel data,
//! including variance-based, median-absolute-deviation, and high-frequency energy
//! approaches.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::VecDeque;

/// Method used to estimate noise from a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseEstimateMethod {
    /// Estimate from local variance in flat regions.
    LocalVariance,
    /// Median Absolute Deviation — robust to outliers.
    Mad,
    /// High-frequency energy in a Laplacian decomposition.
    LaplacianEnergy,
    /// Difference between two consecutive frames in a static scene.
    FrameDiff,
}

impl std::fmt::Display for NoiseEstimateMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LocalVariance => write!(f, "LocalVariance"),
            Self::Mad => write!(f, "MAD"),
            Self::LaplacianEnergy => write!(f, "LaplacianEnergy"),
            Self::FrameDiff => write!(f, "FrameDiff"),
        }
    }
}

/// A single noise estimate result for one frame or region.
#[derive(Debug, Clone)]
pub struct NoiseEstimate {
    /// Estimated noise standard deviation (in pixel value units 0-255).
    pub sigma: f32,
    /// Method that produced this estimate.
    pub method: NoiseEstimateMethod,
    /// Optional per-channel breakdown `[Y, Cb, Cr]`.
    pub channel_sigma: Option<[f32; 3]>,
    /// Confidence in the estimate (0.0 – 1.0).
    pub confidence: f32,
}

impl NoiseEstimate {
    /// Create a new noise estimate.
    #[must_use]
    pub fn new(sigma: f32, method: NoiseEstimateMethod, confidence: f32) -> Self {
        Self {
            sigma,
            method,
            channel_sigma: None,
            confidence,
        }
    }

    /// Attach per-channel sigma values.
    #[must_use]
    pub fn with_channels(mut self, channel_sigma: [f32; 3]) -> Self {
        self.channel_sigma = Some(channel_sigma);
        self
    }

    /// Returns `true` if this estimate indicates a noisy frame (sigma > threshold).
    pub fn is_noisy(&self, threshold: f32) -> bool {
        self.sigma > threshold
    }

    /// Signal-to-noise ratio in dB given a signal range of 0–255.
    pub fn snr_db(&self) -> f32 {
        if self.sigma <= 0.0 {
            return f32::INFINITY;
        }
        20.0 * (255.0_f32 / self.sigma).log10()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute the variance of a slice of pixel values.
fn slice_variance(pixels: &[u8]) -> f32 {
    if pixels.is_empty() {
        return 0.0;
    }
    let n = pixels.len() as f32;
    let mean: f32 = pixels.iter().map(|&p| p as f32).sum::<f32>() / n;
    pixels
        .iter()
        .map(|&p| {
            let diff = p as f32 - mean;
            diff * diff
        })
        .sum::<f32>()
        / n
}

/// Compute MAD (median absolute deviation) for a pixel slice.
fn slice_mad(pixels: &[u8]) -> f32 {
    if pixels.is_empty() {
        return 0.0;
    }
    let mut vals: Vec<f32> = pixels.iter().map(|&p| p as f32).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = vals[vals.len() / 2];
    let mut deviations: Vec<f32> = vals.iter().map(|&v| (v - median).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // σ ≈ MAD / 0.6745 for Gaussian noise
    deviations[deviations.len() / 2] / 0.6745
}

// ---------------------------------------------------------------------------
// NoiseEstimator
// ---------------------------------------------------------------------------

/// Stateful noise estimator that can track estimates over a sliding window of
/// frames.
///
/// # Example
///
/// ```rust
/// use oximedia_denoise::noise_estimate::{NoiseEstimator, NoiseEstimateMethod};
///
/// let mut est = NoiseEstimator::new(NoiseEstimateMethod::LocalVariance);
/// let pixels = vec![128u8; 64 * 64];
/// let result = est.estimate_from_frame(&pixels, 64, 64);
/// assert!(result.sigma >= 0.0);
/// ```
pub struct NoiseEstimator {
    method: NoiseEstimateMethod,
    history: VecDeque<NoiseEstimate>,
    window: usize,
}

impl NoiseEstimator {
    /// Create a new estimator using the specified method and a history window of
    /// 8 frames.
    #[must_use]
    pub fn new(method: NoiseEstimateMethod) -> Self {
        Self {
            method,
            history: VecDeque::new(),
            window: 8,
        }
    }

    /// Create an estimator with a custom history window size.
    #[must_use]
    pub fn with_window(method: NoiseEstimateMethod, window: usize) -> Self {
        Self {
            method,
            history: VecDeque::new(),
            window: window.max(1),
        }
    }

    /// Estimate noise from a raw luma pixel buffer.
    ///
    /// * `pixels` – row-major luma (Y) plane, values 0–255.
    /// * `width` / `height` – frame dimensions.
    pub fn estimate_from_frame(
        &mut self,
        pixels: &[u8],
        width: usize,
        height: usize,
    ) -> NoiseEstimate {
        let estimate = match self.method {
            NoiseEstimateMethod::LocalVariance => {
                self.estimate_local_variance(pixels, width, height)
            }
            NoiseEstimateMethod::Mad => self.estimate_mad(pixels),
            NoiseEstimateMethod::LaplacianEnergy => self.estimate_laplacian(pixels, width, height),
            NoiseEstimateMethod::FrameDiff => self.estimate_frame_diff(pixels),
        };

        if self.history.len() >= self.window {
            self.history.pop_front();
        }
        self.history.push_back(estimate.clone());
        estimate
    }

    /// Return the smoothed (averaged) sigma over the history window.
    pub fn smoothed_sigma(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history.iter().map(|e| e.sigma).sum::<f32>() / self.history.len() as f32
    }

    /// Clear the history buffer.
    pub fn reset(&mut self) {
        self.history.clear();
    }

    // -----------------------------------------------------------------------
    // Private estimation strategies
    // -----------------------------------------------------------------------

    fn estimate_local_variance(&self, pixels: &[u8], width: usize, height: usize) -> NoiseEstimate {
        if width < 4 || height < 4 || pixels.len() < width * height {
            return NoiseEstimate::new(0.0, self.method, 0.0);
        }

        let patch = 4usize;
        let step = patch;
        let mut variances: Vec<f32> = Vec::new();

        let rows = height / step;
        let cols = width / step;
        for row in 0..rows {
            for col in 0..cols {
                let block: Vec<u8> = (0..patch)
                    .flat_map(|r| {
                        let base = (row * step + r) * width + col * step;
                        pixels.get(base..base + patch).unwrap_or(&[]).to_vec()
                    })
                    .collect();
                let v = slice_variance(&block);
                variances.push(v);
            }
        }

        if variances.is_empty() {
            return NoiseEstimate::new(0.0, self.method, 0.0);
        }

        // Use the 10th percentile of local variances as noise estimate
        variances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = (variances.len() as f32 * 0.1) as usize;
        let sigma = variances[idx].sqrt();
        let confidence = (variances.len() as f32 / 100.0).min(1.0);

        NoiseEstimate::new(sigma, self.method, confidence)
    }

    fn estimate_mad(&self, pixels: &[u8]) -> NoiseEstimate {
        if pixels.is_empty() {
            return NoiseEstimate::new(0.0, self.method, 0.0);
        }
        let sigma = slice_mad(pixels);
        NoiseEstimate::new(sigma, self.method, 0.9)
    }

    fn estimate_laplacian(&self, pixels: &[u8], width: usize, height: usize) -> NoiseEstimate {
        if width < 3 || height < 3 || pixels.len() < width * height {
            return NoiseEstimate::new(0.0, self.method, 0.0);
        }

        // Apply discrete Laplacian kernel and accumulate energy
        let mut energy: f64 = 0.0;
        let mut count = 0usize;

        for row in 1..height - 1 {
            for col in 1..width - 1 {
                let c = pixels[row * width + col] as f64;
                let n = pixels[(row - 1) * width + col] as f64;
                let s = pixels[(row + 1) * width + col] as f64;
                let w = pixels[row * width + col - 1] as f64;
                let e = pixels[row * width + col + 1] as f64;
                let lap = (4.0 * c - n - s - w - e).abs();
                energy += lap * lap;
                count += 1;
            }
        }

        if count == 0 {
            return NoiseEstimate::new(0.0, self.method, 0.0);
        }

        let sigma = ((energy / count as f64).sqrt() / 4.0) as f32;
        NoiseEstimate::new(sigma, self.method, 0.85)
    }

    fn estimate_frame_diff(&self, pixels: &[u8]) -> NoiseEstimate {
        // Without a previous frame we use overall MAD as a proxy
        if let Some(prev) = self.history.back() {
            return NoiseEstimate::new(prev.sigma, self.method, prev.confidence * 0.9);
        }
        self.estimate_mad(pixels)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_frame(value: u8, size: usize) -> Vec<u8> {
        vec![value; size]
    }

    fn noisy_frame(size: usize) -> Vec<u8> {
        (0..size).map(|i| ((i * 37 + 13) % 256) as u8).collect()
    }

    #[test]
    fn test_method_display() {
        assert_eq!(
            NoiseEstimateMethod::LocalVariance.to_string(),
            "LocalVariance"
        );
        assert_eq!(NoiseEstimateMethod::Mad.to_string(), "MAD");
        assert_eq!(
            NoiseEstimateMethod::LaplacianEnergy.to_string(),
            "LaplacianEnergy"
        );
        assert_eq!(NoiseEstimateMethod::FrameDiff.to_string(), "FrameDiff");
    }

    #[test]
    fn test_noise_estimate_new() {
        let e = NoiseEstimate::new(5.0, NoiseEstimateMethod::Mad, 0.9);
        assert!((e.sigma - 5.0).abs() < 1e-5);
        assert!((e.confidence - 0.9).abs() < 1e-5);
        assert!(e.channel_sigma.is_none());
    }

    #[test]
    fn test_noise_estimate_with_channels() {
        let e = NoiseEstimate::new(3.0, NoiseEstimateMethod::LocalVariance, 0.8)
            .with_channels([3.0, 1.5, 1.5]);
        assert!(e.channel_sigma.is_some());
        let ch = e.channel_sigma.expect("ch should be valid");
        assert!((ch[0] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_is_noisy() {
        let e = NoiseEstimate::new(10.0, NoiseEstimateMethod::Mad, 0.9);
        assert!(e.is_noisy(5.0));
        assert!(!e.is_noisy(15.0));
    }

    #[test]
    fn test_snr_db_zero_sigma() {
        let e = NoiseEstimate::new(0.0, NoiseEstimateMethod::Mad, 1.0);
        assert!(e.snr_db().is_infinite());
    }

    #[test]
    fn test_snr_db_nonzero() {
        let e = NoiseEstimate::new(1.0, NoiseEstimateMethod::Mad, 1.0);
        // SNR = 20*log10(255/1) ≈ 48.1
        let snr = e.snr_db();
        assert!((snr - 48.13).abs() < 0.1, "snr={snr}");
    }

    #[test]
    fn test_slice_variance_flat() {
        let pixels = flat_frame(100, 256);
        let v = slice_variance(&pixels);
        assert!(v.abs() < 1e-4);
    }

    #[test]
    fn test_slice_mad_flat() {
        let pixels = flat_frame(128, 256);
        let m = slice_mad(&pixels);
        assert!(m.abs() < 1e-4);
    }

    #[test]
    fn test_estimator_local_variance_flat() {
        let mut est = NoiseEstimator::new(NoiseEstimateMethod::LocalVariance);
        let pixels = flat_frame(100, 64 * 64);
        let result = est.estimate_from_frame(&pixels, 64, 64);
        assert!(
            result.sigma < 1.0,
            "Expected low sigma on flat frame, got {}",
            result.sigma
        );
    }

    #[test]
    fn test_estimator_mad_noisy() {
        let mut est = NoiseEstimator::new(NoiseEstimateMethod::Mad);
        let pixels = noisy_frame(128 * 128);
        let result = est.estimate_from_frame(&pixels, 128, 128);
        assert!(result.sigma > 0.0);
    }

    #[test]
    fn test_estimator_laplacian() {
        let mut est = NoiseEstimator::new(NoiseEstimateMethod::LaplacianEnergy);
        let pixels = noisy_frame(64 * 64);
        let result = est.estimate_from_frame(&pixels, 64, 64);
        assert!(result.sigma >= 0.0);
    }

    #[test]
    fn test_estimator_frame_diff_first_frame() {
        let mut est = NoiseEstimator::new(NoiseEstimateMethod::FrameDiff);
        let pixels = noisy_frame(32 * 32);
        let result = est.estimate_from_frame(&pixels, 32, 32);
        // First frame falls back to MAD
        assert!(result.sigma >= 0.0);
    }

    #[test]
    fn test_smoothed_sigma_empty() {
        let est = NoiseEstimator::new(NoiseEstimateMethod::Mad);
        assert!((est.smoothed_sigma() - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_smoothed_sigma_after_frames() {
        let mut est = NoiseEstimator::new(NoiseEstimateMethod::Mad);
        for _ in 0..4 {
            let px = noisy_frame(32 * 32);
            est.estimate_from_frame(&px, 32, 32);
        }
        assert!(est.smoothed_sigma() > 0.0);
    }

    #[test]
    fn test_estimator_reset() {
        let mut est = NoiseEstimator::new(NoiseEstimateMethod::Mad);
        let px = noisy_frame(32 * 32);
        est.estimate_from_frame(&px, 32, 32);
        est.reset();
        assert!((est.smoothed_sigma() - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_with_window() {
        let mut est = NoiseEstimator::with_window(NoiseEstimateMethod::Mad, 2);
        for _ in 0..5 {
            let px = noisy_frame(32 * 32);
            est.estimate_from_frame(&px, 32, 32);
        }
        // history should be capped at window size
        assert!(est.history.len() <= 2);
    }

    #[test]
    fn test_empty_frame() {
        let mut est = NoiseEstimator::new(NoiseEstimateMethod::LocalVariance);
        let result = est.estimate_from_frame(&[], 0, 0);
        assert!((result.sigma - 0.0).abs() < 1e-5);
    }
}
