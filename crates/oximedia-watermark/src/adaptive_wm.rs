//! Adaptive watermark strength computation based on local signal statistics.
//!
//! The human auditory (and visual) system tolerates larger modifications in
//! high-energy regions than in near-silence, allowing watermarks to be
//! embedded at a higher strength where the host signal already masks them.
//! This module computes a locally-adapted strength that maximises robustness
//! while staying below the perceptual threshold.
//!
//! ## Model
//!
//! Given the local variance σ² of a signal segment and a global base strength
//! `s₀`, the adaptive strength is:
//!
//! ```text
//!   s_adaptive = s₀ × (1 + α × log₁₀(1 + σ / σ_ref))
//! ```
//!
//! where:
//! - `α = 2.0`   — sensitivity of adaptation to energy.
//! - `σ_ref = 0.01` — normalisation constant (≈ −40 dB reference).
//!
//! The result is clamped to `[MIN_STRENGTH, MAX_STRENGTH]`.

/// Minimum allowed watermark strength (prevents near-zero embedding).
const MIN_STRENGTH: f32 = 0.001;
/// Maximum allowed watermark strength (prevents perceptible artefacts).
const MAX_STRENGTH: f32 = 0.5;
/// Sensitivity scaling factor for the logarithmic adaptation curve.
const ALPHA: f32 = 2.0;
/// Reference signal standard deviation for normalisation (≈ −40 dBFS).
const SIGMA_REF: f32 = 0.01;

/// Adaptive watermark strength controller.
///
/// # Example
///
/// ```rust
/// use oximedia_watermark::adaptive_wm::AdaptiveWatermark;
///
/// // A quiet region should use strength close to the base.
/// let quiet = AdaptiveWatermark::compute_strength(1e-6, 0.1);
/// // A loud region should use higher strength.
/// let loud  = AdaptiveWatermark::compute_strength(0.25, 0.1);
/// assert!(loud > quiet, "loud region should tolerate stronger watermark");
/// ```
pub struct AdaptiveWatermark;

impl AdaptiveWatermark {
    /// Compute the adaptive watermark strength for a signal segment.
    ///
    /// # Parameters
    ///
    /// - `local_variance` : variance of the host signal in this segment (≥ 0).
    /// - `base_strength`  : global base embedding strength in `[0, 1]`.
    ///
    /// # Returns
    ///
    /// An adapted strength in `[MIN_STRENGTH, MAX_STRENGTH]`.
    #[must_use]
    pub fn compute_strength(local_variance: f32, base_strength: f32) -> f32 {
        let sigma = local_variance.max(0.0).sqrt();
        let gain = 1.0 + ALPHA * (1.0 + sigma / SIGMA_REF).log10();
        let strength = base_strength.clamp(0.0, 1.0) * gain;
        strength.clamp(MIN_STRENGTH, MAX_STRENGTH)
    }

    /// Compute local variance for a slice of f32 audio samples.
    ///
    /// Returns the variance σ² = E[(x − μ)²].
    #[must_use]
    pub fn local_variance(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let n = samples.len() as f32;
        let mean = samples.iter().sum::<f32>() / n;
        samples.iter().map(|&s| (s - mean).powi(2)).sum::<f32>() / n
    }

    /// Compute local variance for a slice of f32 image pixels.
    ///
    /// Pixels are expected in the range [0, 1] or [0, 255] — only the
    /// relative dispersion matters for the strength model.
    #[inline]
    #[must_use]
    pub fn local_variance_image(pixels: &[f32]) -> f32 {
        Self::local_variance(pixels)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strength_increases_with_energy() {
        let low = AdaptiveWatermark::compute_strength(1e-8, 0.1);
        let high = AdaptiveWatermark::compute_strength(0.5, 0.1);
        assert!(high > low, "higher variance should produce higher strength");
    }

    #[test]
    fn test_strength_clamped_to_max() {
        let s = AdaptiveWatermark::compute_strength(100.0, 1.0);
        assert!(s <= MAX_STRENGTH, "should not exceed MAX_STRENGTH");
    }

    #[test]
    fn test_strength_clamped_to_min() {
        let s = AdaptiveWatermark::compute_strength(0.0, 0.0);
        assert!(s >= MIN_STRENGTH, "should not fall below MIN_STRENGTH");
    }

    #[test]
    fn test_local_variance_zero_signal() {
        let samples = vec![0.5f32; 1024];
        let v = AdaptiveWatermark::local_variance(&samples);
        assert!(v < 1e-10, "constant signal should have near-zero variance");
    }

    #[test]
    fn test_local_variance_unit_signal() {
        // Alternating ±1 → variance ≈ 1.0
        let samples: Vec<f32> = (0..1024)
            .map(|i| if i % 2 == 0 { 1.0f32 } else { -1.0f32 })
            .collect();
        let v = AdaptiveWatermark::local_variance(&samples);
        assert!((v - 1.0).abs() < 0.01, "alternating ±1 should have variance ≈ 1");
    }

    #[test]
    fn test_local_variance_empty() {
        let v = AdaptiveWatermark::local_variance(&[]);
        assert_eq!(v, 0.0);
    }
}
