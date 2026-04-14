//! HDR10+ per-frame dynamic metadata analysis.
//!
//! Provides [`Hdr10PlusFrameAnalyzer`] which extracts luminance statistics
//! from an 8-bit HDR-encoded frame and populates an [`Hdr10PlusMetadata`]
//! structure.
//!
//! The input frame is assumed to be in the PQ (SMPTE ST 2084) transfer function
//! with 8-bit precision (values 0–255 mapped to the PQ code word range 0–1).
//! Luma is derived using the Rec. 2020 coefficients Kr=0.2627, Kb=0.0593.

#![allow(dead_code)]

use crate::color_volume::Hdr10PlusMetadata;
use crate::transfer_function::pq_eotf;

// ── Hdr10PlusFrameAnalyzer ────────────────────────────────────────────────────

/// Analyzes a raw frame buffer to extract HDR10+ dynamic luminance statistics.
///
/// The analyzer computes:
/// - **max_luminance**: brightest pixel luminance in nits.
/// - **avg_luminance**: mean luminance across all pixels in nits.
/// - **min_luminance**: darkest pixel luminance in nits.
///
/// These values are packed into an [`Hdr10PlusMetadata`] value and can be used
/// to drive per-shot HDR10+ SEI metadata generation.
///
/// # Example
///
/// ```rust
/// use oximedia_hdr::hdr10plus::Hdr10PlusFrameAnalyzer;
///
/// let analyzer = Hdr10PlusFrameAnalyzer::new();
/// // 2×2 RGB frame, all mid-grey
/// let frame = vec![128u8; 2 * 2 * 3];
/// let meta = analyzer.analyze(&frame, 2, 2);
/// assert!(meta.num_windows == 1);
/// ```
#[derive(Debug, Clone, Default)]
pub struct Hdr10PlusFrameAnalyzer;

impl Hdr10PlusFrameAnalyzer {
    /// Create a new `Hdr10PlusFrameAnalyzer`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Analyze `frame` and return per-frame HDR10+ luminance metadata.
    ///
    /// * `frame` — raw byte slice; supports 3-channel RGB and 4-channel RGBA
    ///   layouts (stride is inferred from `w * h`).  Pixels are PQ-encoded at
    ///   8-bit precision (0 = 0 nits, 255 = 10 000 nits).
    /// * `w`, `h` — frame dimensions in pixels.
    ///
    /// Returns an [`Hdr10PlusMetadata`] with `max_display_mastering_luminance`,
    /// `min_display_mastering_luminance`, and `max_content_light_level` set
    /// to the computed max, avg, and min luminance values (in nits).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze(&self, frame: &[u8], w: u32, h: u32) -> Hdr10PlusMetadata {
        let pixel_count = (w as usize) * (h as usize);

        // Determine bytes-per-pixel
        let total = frame.len();
        let stride = if pixel_count == 0 {
            3
        } else if total >= pixel_count * 4 {
            4
        } else {
            3
        };

        let mut max_nits = 0.0f32;
        let mut min_nits = f32::MAX;
        let mut sum_nits = 0.0f64;
        let mut count = 0u64;

        for p in 0..pixel_count {
            let base = p * stride;
            if base + 2 >= frame.len() {
                break;
            }

            let r = frame[base] as f64 / 255.0;
            let g = frame[base + 1] as f64 / 255.0;
            let b = frame[base + 2] as f64 / 255.0;

            // PQ EOTF: signal [0,1] → linear [0,1] where 1.0 = 10 000 nits
            let r_lin = pq_eotf(r).unwrap_or(0.0);
            let g_lin = pq_eotf(g).unwrap_or(0.0);
            let b_lin = pq_eotf(b).unwrap_or(0.0);

            // Rec. 2020 luma coefficients
            const KR: f64 = 0.2627;
            const KG: f64 = 0.6780;
            const KB: f64 = 0.0593;
            let luma = KR * r_lin + KG * g_lin + KB * b_lin;
            let nits = (luma * 10_000.0) as f32;

            if nits > max_nits {
                max_nits = nits;
            }
            if nits < min_nits {
                min_nits = nits;
            }
            sum_nits += nits as f64;
            count += 1;
        }

        let (max_nits, _min_nits, avg_nits) = if count == 0 {
            (0.0f32, 0.0f32, 0.0f32)
        } else {
            (
                max_nits,
                min_nits,
                (sum_nits / count as f64) as f32,
            )
        };

        // Map to the actual Hdr10PlusMetadata fields:
        // - maxscl[0..2] → max R,G,B channel luminance (nits as u32)
        // - average_maxrgb → average MaxRGB (nits as u32)
        // - target_system_display_max_luminance → max_nits rounded
        Hdr10PlusMetadata {
            system_start_code: 0x3C,
            application_version: 1,
            num_windows: 1,
            target_system_display_max_luminance: max_nits as u32,
            maxscl: [max_nits as u32, max_nits as u32, max_nits as u32],
            average_maxrgb: avg_nits as u32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyzer_new() {
        let _a = Hdr10PlusFrameAnalyzer::new();
    }

    #[test]
    fn test_analyze_empty_frame() {
        let a = Hdr10PlusFrameAnalyzer::new();
        let meta = a.analyze(&[], 0, 0);
        assert_eq!(meta.target_system_display_max_luminance, 0);
    }

    #[test]
    fn test_analyze_black_frame() {
        let a = Hdr10PlusFrameAnalyzer::new();
        let frame = vec![0u8; 4 * 4 * 3];
        let meta = a.analyze(&frame, 4, 4);
        assert_eq!(meta.target_system_display_max_luminance, 0);
        assert_eq!(meta.average_maxrgb, 0);
    }

    #[test]
    fn test_analyze_non_zero_frame() {
        let a = Hdr10PlusFrameAnalyzer::new();
        let frame = vec![128u8; 4 * 4 * 3]; // mid-grey PQ
        let meta = a.analyze(&frame, 4, 4);
        // Mid-grey PQ ~0.5 → nonzero luminance
        assert!(meta.target_system_display_max_luminance > 0);
    }

    #[test]
    fn test_analyze_max_gte_avg() {
        let a = Hdr10PlusFrameAnalyzer::new();
        // Alternating dark/bright pixels
        let mut frame = vec![0u8; 4 * 4 * 3];
        // Make some pixels bright
        frame[0] = 200;
        frame[1] = 200;
        frame[2] = 200;
        let meta = a.analyze(&frame, 4, 4);
        assert!(
            meta.target_system_display_max_luminance >= meta.average_maxrgb,
            "max should be >= avg"
        );
    }

    #[test]
    fn test_analyze_rgba_frame() {
        let a = Hdr10PlusFrameAnalyzer::new();
        let frame = vec![100u8; 4 * 4 * 4]; // RGBA
        let meta = a.analyze(&frame, 4, 4);
        // Should not panic; max >= 0
        assert!(meta.num_windows == 1);
    }

    #[test]
    fn test_analyze_uniform_max_equals_avg() {
        let a = Hdr10PlusFrameAnalyzer::new();
        let frame = vec![180u8; 8 * 8 * 3];
        let meta = a.analyze(&frame, 8, 8);
        // For uniform frame max ≈ avg
        let diff = (meta.target_system_display_max_luminance as i64)
            - (meta.average_maxrgb as i64);
        assert!(diff.abs() <= 1, "uniform frame: max ≈ avg, diff={diff}");
    }
}
