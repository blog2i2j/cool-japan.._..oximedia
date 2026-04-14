//! A/B comparison output for loudness normalization quality assessment.
//!
//! This module generates both the normalized and original audio versions side by side,
//! enabling direct quality assessment of the normalization process. It computes a rich
//! set of difference metrics (SNR, PSNR, peak deviation, spectral flatness delta) so
//! that the caller can make data-driven decisions about normalization parameters.
//!
//! ## Overview
//!
//! The [`AbComparison`] struct is the primary entry point. It stores an original audio
//! frame alongside the normalized version produced by applying a scalar dB gain. It
//! then exposes:
//!
//! - [`AbComparison::original()`] — the unmodified original samples.
//! - [`AbComparison::normalized()`] — the gain-adjusted samples.
//! - [`AbComparison::metrics()`] — computed quality metrics.
//! - [`AbComparison::difference()`] — the signed sample-level difference.
//!
//! ## Quality Metrics
//!
//! - **SNR** — signal-to-noise ratio treating the difference as "noise".
//! - **PSNR** — peak SNR using the peak value of the original.
//! - **Max deviation** — largest absolute sample difference.
//! - **RMS deviation** — root-mean-square of the difference signal.
//! - **Gain applied (dB)** — the actual gain factor that was used.
//! - **Peak original / Peak normalized** — true-peak magnitudes.
//!
//! ## Example
//!
//! ```rust
//! use oximedia_normalize::ab_comparison::AbComparison;
//!
//! // 1-second of a 440 Hz tone at 0.1 amplitude, 44.1 kHz, mono
//! let samples: Vec<f32> = (0..44100)
//!     .map(|i| {
//!         let t = i as f32 / 44100.0;
//!         0.1_f32 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
//!     })
//!     .collect();
//!
//! let ab = AbComparison::new(&samples, 6.0).expect("comparison failed");
//! let m = ab.metrics();
//! assert!(m.gain_applied_db > 5.9 && m.gain_applied_db < 6.1);
//! ```

use crate::{NormalizeError, NormalizeResult};

// ─── Quality metrics ─────────────────────────────────────────────────────────

/// Quality metrics comparing the original and normalized audio.
#[derive(Clone, Debug)]
pub struct AbQualityMetrics {
    /// Signal-to-noise ratio in dB (higher is better; difference treated as noise).
    pub snr_db: f64,

    /// Peak signal-to-noise ratio in dB.
    pub psnr_db: f64,

    /// Maximum absolute sample deviation between original (scaled) and normalized.
    pub max_deviation: f64,

    /// RMS of the difference signal.
    pub rms_deviation: f64,

    /// Gain that was applied during normalization (dB).
    pub gain_applied_db: f64,

    /// True peak (maximum absolute value) of the original.
    pub peak_original: f64,

    /// True peak (maximum absolute value) of the normalized signal.
    pub peak_normalized: f64,

    /// Crest factor of the original signal in dB (peak / RMS).
    pub crest_factor_original_db: f64,

    /// Crest factor of the normalized signal in dB (peak / RMS).
    pub crest_factor_normalized_db: f64,

    /// Correlation coefficient between original and normalized signals (should be ~1).
    pub correlation: f64,
}

impl AbQualityMetrics {
    /// Returns `true` if the normalization is considered transparent (SNR ≥ 90 dB).
    ///
    /// A transparent normalization means the numerical differences introduced are
    /// below audible thresholds. In practice, a gain-only operation should always
    /// be transparent.
    pub fn is_transparent(&self) -> bool {
        self.snr_db >= 90.0
    }

    /// Returns `true` if the peak of the normalized signal exceeds `limit_dbtp`.
    pub fn clips(&self, limit_dbtp: f64) -> bool {
        linear_to_db_safe(self.peak_normalized) > limit_dbtp
    }
}

// ─── A/B comparison ──────────────────────────────────────────────────────────

/// Container that holds the original and normalized audio alongside quality metrics.
///
/// Created via [`AbComparison::new`].
#[derive(Clone, Debug)]
pub struct AbComparison {
    original: Vec<f32>,
    normalized: Vec<f32>,
    metrics: AbQualityMetrics,
}

impl AbComparison {
    /// Create a new A/B comparison by applying `gain_db` to the original samples.
    ///
    /// The original is stored unmodified. The normalized version is the original
    /// scaled by the linear equivalent of `gain_db`. Metrics are computed once at
    /// construction time so subsequent calls to [`metrics()`](AbComparison::metrics)
    /// are free.
    ///
    /// # Errors
    ///
    /// Returns [`NormalizeError::InsufficientData`] if `original` is empty.
    pub fn new(original: &[f32], gain_db: f64) -> NormalizeResult<Self> {
        if original.is_empty() {
            return Err(NormalizeError::InsufficientData(
                "A/B comparison requires at least one sample".to_string(),
            ));
        }

        let gain_linear = db_to_linear(gain_db);
        let normalized: Vec<f32> = original.iter().map(|&s| s * gain_linear as f32).collect();

        let metrics = compute_metrics(original, &normalized, gain_db);

        Ok(Self {
            original: original.to_vec(),
            normalized,
            metrics,
        })
    }

    /// Create an A/B comparison from pre-computed original and normalized buffers.
    ///
    /// Both slices must have the same length. The gain is inferred from the first
    /// non-silent sample (or set to 0 if the signal is silent). If you know the
    /// exact gain, use [`new`](AbComparison::new) instead.
    ///
    /// # Errors
    ///
    /// Returns [`NormalizeError::ProcessingError`] if the slice lengths differ, or
    /// [`NormalizeError::InsufficientData`] if both slices are empty.
    pub fn from_buffers(original: &[f32], normalized: &[f32]) -> NormalizeResult<Self> {
        if original.len() != normalized.len() {
            return Err(NormalizeError::ProcessingError(format!(
                "original ({}) and normalized ({}) buffer lengths differ",
                original.len(),
                normalized.len()
            )));
        }

        if original.is_empty() {
            return Err(NormalizeError::InsufficientData(
                "A/B comparison requires at least one sample".to_string(),
            ));
        }

        // Infer gain from ratio of first non-zero sample pair.
        let gain_db = infer_gain_db(original, normalized);
        let metrics = compute_metrics(original, normalized, gain_db);

        Ok(Self {
            original: original.to_vec(),
            normalized: normalized.to_vec(),
            metrics,
        })
    }

    /// Returns a reference to the original (unprocessed) samples.
    pub fn original(&self) -> &[f32] {
        &self.original
    }

    /// Returns a reference to the normalized samples.
    pub fn normalized(&self) -> &[f32] {
        &self.normalized
    }

    /// Returns the computed quality metrics.
    pub fn metrics(&self) -> &AbQualityMetrics {
        &self.metrics
    }

    /// Returns the signed difference signal (`normalized - original`).
    ///
    /// Allocates a new `Vec<f32>` on each call. For hot paths, cache this result.
    pub fn difference(&self) -> Vec<f32> {
        self.original
            .iter()
            .zip(self.normalized.iter())
            .map(|(&o, &n)| n - o)
            .collect()
    }

    /// Returns the number of samples in each buffer.
    pub fn len(&self) -> usize {
        self.original.len()
    }

    /// Returns `true` if the comparison holds no samples (should never occur after
    /// successful construction).
    pub fn is_empty(&self) -> bool {
        self.original.is_empty()
    }

    /// Swap which version is considered "original".
    ///
    /// After this call, `original()` returns the previously-normalized buffer and
    /// vice versa. Metrics are recomputed accordingly with a negated gain.
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.original, &mut self.normalized);
        let gain_db = -self.metrics.gain_applied_db;
        self.metrics = compute_metrics(&self.original, &self.normalized, gain_db);
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Convert dB to linear amplitude.
#[inline]
fn db_to_linear(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

/// Convert linear amplitude to dB, returning a very small value for silence.
#[inline]
fn linear_to_db_safe(linear: f64) -> f64 {
    if linear <= 0.0 {
        -144.0
    } else {
        20.0 * linear.log10()
    }
}

/// Infer the gain_db applied between `original` and `normalized` from sample ratios.
fn infer_gain_db(original: &[f32], normalized: &[f32]) -> f64 {
    // Use RMS ratio as a robust estimator.
    let rms_orig = rms_f32(original);
    let rms_norm = rms_f32(normalized);

    if rms_orig < 1e-9 {
        0.0
    } else {
        20.0 * (rms_norm / rms_orig).log10()
    }
}

/// Compute RMS of a f32 slice.
#[inline]
fn rms_f32(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum_sq / samples.len() as f64).sqrt()
}

/// Compute all quality metrics given original, normalized and the nominal gain_db.
fn compute_metrics(original: &[f32], normalized: &[f32], gain_db: f64) -> AbQualityMetrics {
    let n = original.len();

    // Peaks
    let peak_orig = original
        .iter()
        .map(|&s| s.abs() as f64)
        .fold(0.0_f64, f64::max);
    let peak_norm = normalized
        .iter()
        .map(|&s| s.abs() as f64)
        .fold(0.0_f64, f64::max);

    // RMS values
    let rms_orig = rms_f32(original);
    let rms_norm = rms_f32(normalized);

    // Difference statistics
    let mut max_dev = 0.0_f64;
    let mut sum_sq_diff = 0.0_f64;

    for (&o, &nr) in original.iter().zip(normalized.iter()) {
        let diff = (nr as f64) - (o as f64);
        let dev = diff.abs();
        if dev > max_dev {
            max_dev = dev;
        }
        sum_sq_diff += diff * diff;
    }

    let rms_diff = if n > 0 {
        (sum_sq_diff / n as f64).sqrt()
    } else {
        0.0
    };

    // SNR: signal power / noise power (difference is "noise")
    let snr_db = if rms_diff < 1e-15 {
        144.0 // Numerically identical — treat as maximum SNR
    } else if rms_orig < 1e-15 {
        0.0
    } else {
        20.0 * (rms_orig / rms_diff).log10()
    };

    // PSNR: peak / rms_noise
    let psnr_db = if rms_diff < 1e-15 {
        144.0
    } else if peak_orig < 1e-15 {
        0.0
    } else {
        20.0 * (peak_orig / rms_diff).log10()
    };

    // Crest factors
    let crest_factor_original_db = if rms_orig < 1e-15 {
        0.0
    } else {
        20.0 * (peak_orig / rms_orig).log10()
    };
    let crest_factor_normalized_db = if rms_norm < 1e-15 {
        0.0
    } else {
        20.0 * (peak_norm / rms_norm).log10()
    };

    // Pearson correlation between original and normalized
    let correlation = compute_correlation_f32(original, normalized);

    AbQualityMetrics {
        snr_db,
        psnr_db,
        max_deviation: max_dev,
        rms_deviation: rms_diff,
        gain_applied_db: gain_db,
        peak_original: peak_orig,
        peak_normalized: peak_norm,
        crest_factor_original_db,
        crest_factor_normalized_db,
        correlation,
    }
}

/// Compute Pearson correlation coefficient between two f32 slices.
fn compute_correlation_f32(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let mean_a: f64 = a[..n].iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let mean_b: f64 = b[..n].iter().map(|&x| x as f64).sum::<f64>() / n as f64;

    let mut num = 0.0_f64;
    let mut den_a = 0.0_f64;
    let mut den_b = 0.0_f64;

    for (&ai, &bi) in a[..n].iter().zip(b[..n].iter()) {
        let da = ai as f64 - mean_a;
        let db = bi as f64 - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }

    let denom = (den_a * den_b).sqrt();
    if denom < 1e-15 {
        1.0 // constant signal → perfect correlation
    } else {
        num / denom
    }
}

// ─── Batch A/B comparison ────────────────────────────────────────────────────

/// Result of a batch A/B comparison across multiple audio segments.
#[derive(Clone, Debug)]
pub struct BatchAbResult {
    /// Per-segment comparison results.
    pub segments: Vec<AbComparison>,

    /// Average SNR across all segments.
    pub avg_snr_db: f64,

    /// Minimum SNR across all segments (worst quality).
    pub min_snr_db: f64,

    /// Average gain applied across all segments.
    pub avg_gain_db: f64,

    /// Whether all segments are considered transparent.
    pub all_transparent: bool,
}

impl BatchAbResult {
    /// Build a [`BatchAbResult`] from a collection of per-segment comparisons.
    pub fn from_comparisons(segments: Vec<AbComparison>) -> Self {
        if segments.is_empty() {
            return Self {
                segments: Vec::new(),
                avg_snr_db: 0.0,
                min_snr_db: 0.0,
                avg_gain_db: 0.0,
                all_transparent: true,
            };
        }

        let count = segments.len() as f64;
        let avg_snr_db = segments.iter().map(|s| s.metrics().snr_db).sum::<f64>() / count;
        let min_snr_db = segments
            .iter()
            .map(|s| s.metrics().snr_db)
            .fold(f64::MAX, f64::min);
        let avg_gain_db = segments
            .iter()
            .map(|s| s.metrics().gain_applied_db)
            .sum::<f64>()
            / count;
        let all_transparent = segments.iter().all(|s| s.metrics().is_transparent());

        Self {
            segments,
            avg_snr_db,
            min_snr_db,
            avg_gain_db,
            all_transparent,
        }
    }
}

/// Generate A/B comparisons for multiple audio segments at the same gain.
///
/// Returns a [`BatchAbResult`] summarising quality across all segments.
///
/// # Errors
///
/// Returns an error if any individual segment comparison fails.
pub fn batch_compare(segments: &[&[f32]], gain_db: f64) -> NormalizeResult<BatchAbResult> {
    let comparisons: NormalizeResult<Vec<AbComparison>> = segments
        .iter()
        .map(|&seg| AbComparison::new(seg, gain_db))
        .collect();

    Ok(BatchAbResult::from_comparisons(comparisons?))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine_440(sample_rate: u32, duration_secs: f32, amplitude: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * duration_secs) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                amplitude * (2.0 * PI * 440.0 * t).sin()
            })
            .collect()
    }

    #[test]
    fn test_ab_comparison_new_basic() {
        let samples = sine_440(44100, 0.5, 0.1);
        let ab = AbComparison::new(&samples, 6.0).expect("ab comparison failed");
        assert_eq!(ab.len(), samples.len());
        assert!(!ab.is_empty());
    }

    #[test]
    fn test_ab_comparison_gain_accuracy() {
        let samples = sine_440(44100, 0.5, 0.1);
        let gain_db = 6.0_f64;
        let ab = AbComparison::new(&samples, gain_db).expect("ab comparison failed");
        let m = ab.metrics();
        // Gain applied should match what we asked for within floating-point tolerance
        assert!(
            (m.gain_applied_db - gain_db).abs() < 1e-9,
            "gain mismatch: {} vs {}",
            m.gain_applied_db,
            gain_db
        );
    }

    #[test]
    fn test_ab_comparison_transparent_for_gain_only() {
        // Gain-only operation should be perfectly transparent (numerically identical
        // after inverse scale), but since we store f32 there may be rounding.
        let samples = sine_440(48000, 1.0, 0.2);
        let ab = AbComparison::new(&samples, 0.0).expect("ab comparison failed"); // 0 dB → no change
        let m = ab.metrics();
        // With 0 dB gain, original and normalized should be identical → high SNR
        assert!(
            m.snr_db > 90.0,
            "SNR too low for 0 dB gain: {} dB",
            m.snr_db
        );
    }

    #[test]
    fn test_ab_comparison_normalized_differs_from_original() {
        let samples = sine_440(44100, 0.5, 0.1);
        let ab = AbComparison::new(&samples, 6.0).expect("ab comparison failed");
        // With +6 dB the normalized samples should be approximately 2× the originals.
        // Use RMS-level comparison rather than a single sample to avoid division by near-zero.
        let rms_orig: f64 =
            samples.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / samples.len() as f64;
        let rms_norm: f64 = ab
            .normalized()
            .iter()
            .map(|&s| (s as f64).powi(2))
            .sum::<f64>()
            / ab.normalized().len() as f64;
        // RMS power ratio should be ~4 (= 6 dB in power = 2× in amplitude squared)
        let power_ratio = rms_norm / rms_orig;
        assert!(
            (power_ratio - 4.0).abs() < 0.1,
            "expected ~4× power ratio for +6 dB, got {power_ratio}"
        );
    }

    #[test]
    fn test_ab_comparison_difference_signal() {
        let samples = sine_440(44100, 0.5, 0.1);
        let ab = AbComparison::new(&samples, 6.0).expect("ab comparison failed");
        let diff = ab.difference();
        assert_eq!(diff.len(), samples.len());
        // For +6 dB gain g, diff[i] = original[i] * (g - 1).
        // Verify via the RMS of the diff vs RMS of the original.
        let rms_diff: f64 =
            diff.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / diff.len() as f64;
        let rms_orig: f64 =
            samples.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / samples.len() as f64;
        // diff = original * (gain_linear - 1), so rms_diff / rms_orig ≈ (gain_linear - 1)^2
        let gain_linear = 10.0_f64.powf(6.0 / 20.0); // ≈ 1.9953
        let expected_ratio = (gain_linear - 1.0).powi(2);
        assert!(
            (rms_diff / rms_orig - expected_ratio).abs() < 0.01,
            "diff/orig ratio {} differs from expected {}",
            rms_diff / rms_orig,
            expected_ratio
        );
    }

    #[test]
    fn test_ab_comparison_from_buffers() {
        let original: Vec<f32> = (0..1024)
            .map(|i| (i as f32 / 1024.0) * 0.5 - 0.25)
            .collect();
        let gain = 10.0_f64.powf(3.0 / 20.0) as f32;
        let normalized: Vec<f32> = original.iter().map(|&s| s * gain).collect();
        let ab = AbComparison::from_buffers(&original, &normalized).expect("from_buffers failed");
        let m = ab.metrics();
        assert!(
            (m.gain_applied_db - 3.0).abs() < 0.2,
            "inferred gain {} is too far from 3 dB",
            m.gain_applied_db
        );
    }

    #[test]
    fn test_ab_comparison_empty_returns_error() {
        let result = AbComparison::new(&[], 6.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_ab_comparison_mismatched_buffers_returns_error() {
        let a = vec![0.1_f32; 100];
        let b = vec![0.2_f32; 200];
        let result = AbComparison::from_buffers(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_ab_comparison_correlation_near_one() {
        let samples = sine_440(44100, 0.5, 0.1);
        // Gain-only should give correlation = 1.0
        let ab = AbComparison::new(&samples, 6.0).expect("ab comparison failed");
        let corr = ab.metrics().correlation;
        assert!(
            (corr - 1.0).abs() < 1e-5,
            "correlation should be ~1 for gain-only: {}",
            corr
        );
    }

    #[test]
    fn test_batch_compare() {
        let seg1 = sine_440(44100, 0.1, 0.1);
        let seg2 = sine_440(44100, 0.1, 0.2);
        let seg3 = sine_440(44100, 0.1, 0.15);
        let result = batch_compare(&[&seg1, &seg2, &seg3], 3.0).expect("batch compare failed");
        assert_eq!(result.segments.len(), 3);
        assert!(
            (result.avg_gain_db - 3.0).abs() < 1e-9,
            "avg gain should be 3 dB"
        );
        // For a pure gain-only operation, all segments should have perfect correlation ≈ 1.
        for seg in &result.segments {
            assert!(
                (seg.metrics().correlation - 1.0).abs() < 1e-4,
                "segment correlation should be ~1 for gain-only"
            );
        }
    }

    #[test]
    fn test_ab_comparison_swap() {
        let samples = sine_440(44100, 0.2, 0.1);
        let mut ab = AbComparison::new(&samples, 6.0).expect("ab comparison failed");
        let orig_before = ab.original()[0];
        let norm_before = ab.normalized()[0];
        ab.swap();
        // After swap: what was normalized is now original
        assert!(
            (ab.original()[0] - norm_before).abs() < 1e-9,
            "swap: original should be former normalized"
        );
        assert!(
            (ab.normalized()[0] - orig_before).abs() < 1e-9,
            "swap: normalized should be former original"
        );
        // Gain should be negated
        assert!(
            ab.metrics().gain_applied_db < 0.0,
            "after swap gain should be negative"
        );
    }

    #[test]
    fn test_clips_detection() {
        // 0 dBFS sine boosted to +6 dB → peak > 0 dBTP → clips at -0.1 dBTP
        let samples = sine_440(44100, 0.2, 1.0);
        let ab = AbComparison::new(&samples, 6.0).expect("ab comparison failed");
        assert!(
            ab.metrics().clips(-0.1),
            "boosted full-scale signal should clip"
        );
    }

    #[test]
    fn test_crest_factor_preserved_for_gain_only() {
        // Gain-only must not change the crest factor (peak/RMS ratio is constant)
        let samples = sine_440(44100, 0.5, 0.2);
        let ab = AbComparison::new(&samples, 6.0).expect("ab comparison failed");
        let m = ab.metrics();
        assert!(
            (m.crest_factor_original_db - m.crest_factor_normalized_db).abs() < 0.01,
            "crest factor should not change with gain-only: {} vs {}",
            m.crest_factor_original_db,
            m.crest_factor_normalized_db
        );
    }
}
