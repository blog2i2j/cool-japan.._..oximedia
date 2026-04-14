//! HDR-specific quality assessment metrics.
//!
//! Provides tools for analyzing HDR (High Dynamic Range) video content quality,
//! including luminance analysis, dynamic range assessment, and tone-mapping fidelity.
//!
//! # Transfer Functions Supported
//! - **PQ** (ST.2084 / SMPTE 2084) — used in HDR10 and Dolby Vision
//! - **HLG** (Hybrid Log-Gamma / ARIB STD-B67) — used in broadcast HDR
//! - **Linear** — raw linear light values without transfer encoding

use serde::{Deserialize, Serialize};
use std::fmt;

/// Errors produced by HDR quality assessment.
#[derive(Debug, Clone, PartialEq)]
pub enum HdrQualityError {
    /// The pixel buffer was empty.
    EmptyFrame,
    /// Reference and distorted frames have different sizes.
    DimensionMismatch {
        /// Expected total pixel count.
        expected: usize,
        /// Actual total pixel count.
        actual: usize,
    },
    /// One or more pixel values are NaN or infinite.
    InvalidPixelValues,
}

impl fmt::Display for HdrQualityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyFrame => write!(f, "HDR frame pixel buffer is empty"),
            Self::DimensionMismatch { expected, actual } => write!(
                f,
                "dimension mismatch: expected {expected} pixels, got {actual}"
            ),
            Self::InvalidPixelValues => {
                write!(f, "pixel buffer contains NaN or infinite values")
            }
        }
    }
}

impl std::error::Error for HdrQualityError {}

/// HDR transfer function (electro-optical transfer function).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HdrTransfer {
    /// SMPTE ST 2084 (PQ) — perceptual quantizer, max 10 000 nit reference.
    Pq,
    /// Hybrid Log-Gamma (HLG / ARIB STD-B67).
    Hlg,
    /// Linear light — no transfer encoding applied.
    Linear,
}

impl HdrTransfer {
    /// Approximate scene-referred peak luminance in nits for this transfer function.
    ///
    /// Returns the reference peak white level in nits used when deriving
    /// absolute luminance from normalised linear values.
    #[must_use]
    pub const fn peak_nits(self) -> f32 {
        match self {
            Self::Pq => 10_000.0,
            Self::Hlg => 1_000.0,
            Self::Linear => 100.0,
        }
    }
}

/// Aggregate HDR quality metrics for a single frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdrMetrics {
    /// Peak luminance measured in the frame (nits).
    pub peak_luminance_nits: f32,
    /// Minimum (black-level) luminance measured in the frame (nits).
    pub black_level_nits: f32,
    /// Dynamic range expressed as photographic stops.
    pub dynamic_range_stops: f32,
    /// Fraction of pixels at or above the transfer-function clip ceiling [0, 1].
    pub highlight_clipping_fraction: f32,
    /// Fraction of pixels at or below the minimum representable value [0, 1].
    pub shadow_clipping_fraction: f32,
}

/// Tone-mapping fidelity comparison between an HDR reference and a distorted frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToneMappingQuality {
    /// How well highlights (top 5 % of luminance) are preserved [0, 1].
    pub preserved_highlights: f32,
    /// How well shadows (bottom 5 % of luminance) are preserved [0, 1].
    pub preserved_shadows: f32,
    /// Fidelity of mid-tones (middle 90 % of luminance) [0, 1].
    pub midtone_fidelity: f32,
}

/// HDR quality assessor.
///
/// Analyses linear-light float pixel buffers (one value per pixel, representing
/// scene-referred luminance in relative units where 1.0 equals the transfer
/// function's reference white).
pub struct HdrQualityAssessor {
    /// Fraction of max value considered "highlight clipping" threshold.
    highlight_clip_threshold: f32,
    /// Absolute minimum value considered non-zero (shadow clip).
    shadow_clip_threshold: f32,
}

impl HdrQualityAssessor {
    /// Creates a new `HdrQualityAssessor` with default thresholds.
    #[must_use]
    pub fn new() -> Self {
        Self {
            highlight_clip_threshold: 0.999,
            shadow_clip_threshold: 1e-4,
        }
    }

    /// Creates an assessor with custom clipping thresholds.
    ///
    /// * `highlight_clip_threshold` — normalised value above which a pixel is
    ///   considered clipped in the highlights (range 0 – 1, relative to peak).
    /// * `shadow_clip_threshold` — absolute linear value below which a pixel is
    ///   considered clipped in the shadows.
    #[must_use]
    pub fn with_thresholds(highlight_clip_threshold: f32, shadow_clip_threshold: f32) -> Self {
        Self {
            highlight_clip_threshold,
            shadow_clip_threshold,
        }
    }

    /// Analyses an HDR frame and returns aggregate quality metrics.
    ///
    /// # Parameters
    /// * `pixels` — linear light values, one `f32` per pixel (single channel,
    ///   e.g. luminance Y).  Values should be ≥ 0.
    /// * `width` / `height` — frame dimensions.
    /// * `transfer` — the electro-optical transfer function in use; used to
    ///   convert normalised values to absolute nits.
    ///
    /// # Errors
    /// Returns [`HdrQualityError::EmptyFrame`] if `pixels` is empty,
    /// [`HdrQualityError::DimensionMismatch`] if `pixels.len() != width * height`,
    /// or [`HdrQualityError::InvalidPixelValues`] if any value is NaN / infinite.
    pub fn analyze_frame(
        &self,
        pixels: &[f32],
        width: u32,
        height: u32,
        transfer: HdrTransfer,
    ) -> Result<HdrMetrics, HdrQualityError> {
        analyze_frame(pixels, width, height, transfer, self)
    }

    /// Compares an HDR reference frame to a distorted frame, returning
    /// tone-mapping fidelity scores.
    ///
    /// # Errors
    /// Returns errors for empty buffers, mismatched sizes, or invalid values.
    pub fn compare_hdr_frames(
        &self,
        reference: &[f32],
        distorted: &[f32],
        width: u32,
        height: u32,
    ) -> Result<ToneMappingQuality, HdrQualityError> {
        compare_hdr_frames(reference, distorted, width, height)
    }
}

impl Default for HdrQualityAssessor {
    fn default() -> Self {
        Self::new()
    }
}

// ─── free-standing analysis functions ─────────────────────────────────────────

/// Analyses an HDR frame buffer and returns aggregate quality metrics.
///
/// This is the functional (non-method) entry point — useful when a full
/// [`HdrQualityAssessor`] object is not needed.
///
/// # Errors
/// See [`HdrQualityAssessor::analyze_frame`].
pub fn analyze_frame(
    pixels: &[f32],
    width: u32,
    height: u32,
    transfer: HdrTransfer,
    assessor: &HdrQualityAssessor,
) -> Result<HdrMetrics, HdrQualityError> {
    let expected = (width as usize) * (height as usize);
    if pixels.is_empty() || expected == 0 {
        return Err(HdrQualityError::EmptyFrame);
    }
    if pixels.len() != expected {
        return Err(HdrQualityError::DimensionMismatch {
            expected,
            actual: pixels.len(),
        });
    }
    for &v in pixels {
        if !v.is_finite() {
            return Err(HdrQualityError::InvalidPixelValues);
        }
    }

    let peak_nits = transfer.peak_nits();

    // Find statistical extremes in a single pass.
    let mut max_val: f32 = 0.0;
    let mut min_val: f32 = f32::MAX;
    let mut highlight_clipped: usize = 0;
    let mut shadow_clipped: usize = 0;

    for &v in pixels {
        let v = v.max(0.0); // clamp negative artefacts
        if v > max_val {
            max_val = v;
        }
        if v < min_val {
            min_val = v;
        }
        if v >= assessor.highlight_clip_threshold {
            highlight_clipped += 1;
        }
        if v <= assessor.shadow_clip_threshold {
            shadow_clipped += 1;
        }
    }

    let n = pixels.len() as f32;
    let peak_luminance_nits = max_val * peak_nits;
    let black_level_nits = min_val * peak_nits;

    // Dynamic range in stops: log2(peak / black).  Guard against division by zero.
    let dynamic_range_stops = if black_level_nits > 1e-6 {
        (peak_luminance_nits / black_level_nits).log2()
    } else if peak_luminance_nits > 1e-6 {
        // Black level is effectively zero → use a sentinel indicating very high DR.
        (peak_luminance_nits / 1e-6_f32).log2()
    } else {
        0.0
    };

    Ok(HdrMetrics {
        peak_luminance_nits,
        black_level_nits,
        dynamic_range_stops,
        highlight_clipping_fraction: highlight_clipped as f32 / n,
        shadow_clipping_fraction: shadow_clipped as f32 / n,
    })
}

/// Compares a reference HDR frame to a distorted frame and returns
/// tone-mapping fidelity scores.
///
/// # Errors
/// See [`HdrQualityAssessor::compare_hdr_frames`].
pub fn compare_hdr_frames(
    reference: &[f32],
    distorted: &[f32],
    width: u32,
    height: u32,
) -> Result<ToneMappingQuality, HdrQualityError> {
    let expected = (width as usize) * (height as usize);
    if reference.is_empty() || expected == 0 {
        return Err(HdrQualityError::EmptyFrame);
    }
    if reference.len() != expected {
        return Err(HdrQualityError::DimensionMismatch {
            expected,
            actual: reference.len(),
        });
    }
    if distorted.len() != expected {
        return Err(HdrQualityError::DimensionMismatch {
            expected,
            actual: distorted.len(),
        });
    }
    for &v in reference.iter().chain(distorted.iter()) {
        if !v.is_finite() {
            return Err(HdrQualityError::InvalidPixelValues);
        }
    }

    // Sort reference values to find luminance percentile thresholds.
    let mut ref_sorted: Vec<f32> = reference.iter().map(|&v| v.max(0.0)).collect();
    ref_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = ref_sorted.len();
    let shadow_end = (n as f32 * 0.05) as usize;
    let highlight_start = n - (n as f32 * 0.05) as usize;

    // Build index-sorted pairs so we can evaluate distorted at the same positions.
    // We need to compare reference to distorted at corresponding pixel positions.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        reference[a]
            .partial_cmp(&reference[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let shadow_indices = &indices[..shadow_end.max(1)];
    let highlight_indices = &indices[highlight_start.min(n - 1)..];
    let midtone_indices = &indices[shadow_end.max(1)..highlight_start.min(n - 1)];

    let fidelity = |idx_slice: &[usize]| -> f32 {
        if idx_slice.is_empty() {
            return 1.0;
        }
        // Measure normalised absolute error relative to the reference band range.
        let band_ref: Vec<f32> = idx_slice.iter().map(|&i| reference[i].max(0.0)).collect();
        let band_dist: Vec<f32> = idx_slice.iter().map(|&i| distorted[i].max(0.0)).collect();

        let ref_mean = band_ref.iter().sum::<f32>() / band_ref.len() as f32;
        let scale = ref_mean.max(1e-6);

        let mae = band_ref
            .iter()
            .zip(band_dist.iter())
            .map(|(&r, &d)| (r - d).abs())
            .sum::<f32>()
            / band_ref.len() as f32;

        // Fidelity is 1 minus normalised MAE, clamped to [0, 1].
        (1.0 - mae / scale).clamp(0.0, 1.0)
    };

    Ok(ToneMappingQuality {
        preserved_highlights: fidelity(highlight_indices),
        preserved_shadows: fidelity(shadow_indices),
        midtone_fidelity: fidelity(midtone_indices),
    })
}

// ─── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(value: f32, n: usize) -> Vec<f32> {
        vec![value; n]
    }

    // ── analyze_frame ──────────────────────────────────────────────────────────

    #[test]
    fn empty_frame_returns_error() {
        let assessor = HdrQualityAssessor::new();
        let result = assessor.analyze_frame(&[], 0, 0, HdrTransfer::Pq);
        assert!(matches!(result, Err(HdrQualityError::EmptyFrame)));
    }

    #[test]
    fn dimension_mismatch_returns_error() {
        let assessor = HdrQualityAssessor::new();
        let pixels = vec![0.5_f32; 10];
        let result = assessor.analyze_frame(&pixels, 4, 4, HdrTransfer::Pq);
        assert!(matches!(
            result,
            Err(HdrQualityError::DimensionMismatch {
                expected: 16,
                actual: 10
            })
        ));
    }

    #[test]
    fn invalid_pixel_nan_returns_error() {
        let assessor = HdrQualityAssessor::new();
        let mut pixels = vec![0.5_f32; 4];
        pixels[2] = f32::NAN;
        let result = assessor.analyze_frame(&pixels, 2, 2, HdrTransfer::Pq);
        assert!(matches!(result, Err(HdrQualityError::InvalidPixelValues)));
    }

    #[test]
    fn all_black_frame_pq() {
        let assessor = HdrQualityAssessor::new();
        // Completely black frame (all zeros).
        let pixels = make_frame(0.0, 4);
        let metrics = assessor
            .analyze_frame(&pixels, 2, 2, HdrTransfer::Pq)
            .expect("should succeed");

        assert_eq!(metrics.peak_luminance_nits, 0.0);
        assert_eq!(metrics.dynamic_range_stops, 0.0);
        // All pixels are at or below shadow threshold → 100 % shadow clipping.
        assert_eq!(metrics.shadow_clipping_fraction, 1.0);
        assert_eq!(metrics.highlight_clipping_fraction, 0.0);
    }

    #[test]
    fn all_white_frame_pq() {
        let assessor = HdrQualityAssessor::new();
        // All pixels at reference peak (1.0 in normalised PQ).
        let pixels = make_frame(1.0, 9);
        let metrics = assessor
            .analyze_frame(&pixels, 3, 3, HdrTransfer::Pq)
            .expect("should succeed");

        // Peak should be PQ peak × 1.0.
        assert!((metrics.peak_luminance_nits - HdrTransfer::Pq.peak_nits()).abs() < 1.0);
        // All pixels are at or above highlight threshold → 100 % highlight clipping.
        assert_eq!(metrics.highlight_clipping_fraction, 1.0);
    }

    #[test]
    fn highlight_clipping_detection() {
        let assessor = HdrQualityAssessor::new();
        // 75 % bright (clipped), 25 % mid.
        let mut pixels = vec![1.0_f32; 6];
        pixels.extend_from_slice(&[0.5_f32; 2]);
        let metrics = assessor
            .analyze_frame(&pixels, 4, 2, HdrTransfer::Hlg)
            .expect("should succeed");

        assert!((metrics.highlight_clipping_fraction - 0.75).abs() < 0.01);
    }

    #[test]
    fn shadow_clipping_detection() {
        let assessor = HdrQualityAssessor::new();
        // Half the pixels are black (clipped shadows).
        let mut pixels = vec![0.0_f32; 4];
        pixels.extend_from_slice(&[0.5_f32; 4]);
        let metrics = assessor
            .analyze_frame(&pixels, 4, 2, HdrTransfer::Linear)
            .expect("should succeed");

        assert!((metrics.shadow_clipping_fraction - 0.5).abs() < 0.01);
    }

    #[test]
    fn dynamic_range_stops_plausible() {
        let assessor = HdrQualityAssessor::new();
        // Black = 1/1024, white = 1.0 → should give ~10 stops.
        let mut pixels = vec![1.0_f32; 1];
        pixels.extend(vec![1.0 / 1024.0_f32; 1023]);
        let metrics = assessor
            .analyze_frame(&pixels, 32, 32, HdrTransfer::Pq)
            .expect("should succeed");

        // log2(1.0 / (1/1024)) = log2(1024) = 10
        assert!(
            metrics.dynamic_range_stops > 9.0,
            "expected >9 stops, got {}",
            metrics.dynamic_range_stops
        );
    }

    #[test]
    fn hlg_peak_nits_used() {
        let assessor = HdrQualityAssessor::new();
        let pixels = make_frame(1.0, 1);
        let metrics = assessor
            .analyze_frame(&pixels, 1, 1, HdrTransfer::Hlg)
            .expect("should succeed");
        assert!((metrics.peak_luminance_nits - 1000.0).abs() < 0.1);
    }

    // ── compare_hdr_frames ────────────────────────────────────────────────────

    #[test]
    fn identical_frames_perfect_fidelity() {
        let pixels = vec![0.1_f32, 0.3, 0.5, 0.7, 0.9, 0.05, 0.15, 0.8, 0.95, 1.0];
        let result = compare_hdr_frames(&pixels, &pixels, 2, 5).expect("should succeed");

        assert!(
            result.preserved_highlights > 0.99,
            "highlights: {}",
            result.preserved_highlights
        );
        assert!(
            result.preserved_shadows > 0.99,
            "shadows: {}",
            result.preserved_shadows
        );
        assert!(
            result.midtone_fidelity > 0.99,
            "midtones: {}",
            result.midtone_fidelity
        );
    }

    #[test]
    fn zeroed_distorted_poor_fidelity() {
        let reference: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
        let distorted = vec![0.0_f32; 16];
        let result = compare_hdr_frames(&reference, &distorted, 4, 4).expect("should succeed");

        // All fidelity scores should be much less than 1.
        assert!(
            result.midtone_fidelity < 0.5,
            "midtone_fidelity too high: {}",
            result.midtone_fidelity
        );
    }

    #[test]
    fn compare_mismatch_returns_error() {
        let reference = vec![0.5_f32; 4];
        let distorted = vec![0.5_f32; 9];
        let result = compare_hdr_frames(&reference, &distorted, 2, 2);
        assert!(matches!(
            result,
            Err(HdrQualityError::DimensionMismatch { .. })
        ));
    }
}
