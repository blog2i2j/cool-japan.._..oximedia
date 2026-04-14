//! Bandwidth estimation and adaptive streaming bitrate recommendations.
//!
//! Provides a [`BitrateRecommender`] that suggests a target bitrate based on
//! resolution, frame rate, and content complexity.  The recommendation formula
//! mirrors common empirical rules used by adaptive-bitrate (ABR) ladder builders.

#![allow(dead_code)]

/// Bitrate recommendation for a given resolution / frame rate / complexity.
#[derive(Debug, Clone, Copy)]
pub struct BitrateRecommendation {
    /// Recommended bitrate in bits per second.
    pub bps: u32,
    /// Recommended bitrate in kilobits per second.
    pub kbps: f32,
    /// Recommended bitrate in megabits per second.
    pub mbps: f32,
}

impl BitrateRecommendation {
    /// Create from a raw bits-per-second value.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn from_bps(bps: u32) -> Self {
        let kbps = bps as f32 / 1000.0;
        let mbps = bps as f32 / 1_000_000.0;
        Self { bps, kbps, mbps }
    }
}

/// Adaptive bitrate recommender.
///
/// Computes a target bitrate using the empirical formula:
///
/// ```text
/// bitrate_bps = width × height × fps × complexity × 0.07
/// ```
///
/// The factor 0.07 encodes approximate bits-per-pixel-per-second for typical
/// H.264/H.265 content at moderate quality.
pub struct BitrateRecommender {
    /// Multiplier applied to the base formula (default 0.07).
    pub bits_per_pixel_per_second: f64,
}

impl BitrateRecommender {
    /// Create a recommender with the default BPP factor (0.07).
    #[must_use]
    pub fn new() -> Self {
        Self {
            bits_per_pixel_per_second: 0.07,
        }
    }

    /// Create a recommender with a custom BPP factor.
    #[must_use]
    pub fn with_factor(bits_per_pixel_per_second: f64) -> Self {
        Self {
            bits_per_pixel_per_second: bits_per_pixel_per_second.max(0.0),
        }
    }

    /// Recommend a bitrate for the given parameters.
    ///
    /// # Arguments
    ///
    /// * `width` – video width in pixels
    /// * `height` – video height in pixels
    /// * `fps` – frame rate (frames per second)
    /// * `complexity` – content complexity in [0.0, 1.0]; 0.5 is typical live content
    ///
    /// # Returns
    ///
    /// A [`BitrateRecommendation`] with bps, kbps, and mbps fields.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn recommend(
        &self,
        width: u32,
        height: u32,
        fps: f32,
        complexity: f32,
    ) -> BitrateRecommendation {
        let bps = recommend_impl(width, height, fps, complexity, self.bits_per_pixel_per_second);
        BitrateRecommendation::from_bps(bps)
    }
}

impl Default for BitrateRecommender {
    fn default() -> Self {
        Self::new()
    }
}

/// Free-function variant for convenience.
///
/// Uses the default BPP factor (0.07).
///
/// ```
/// use oximedia_analysis::bitrate_recommender::recommend;
/// let bps = recommend(1920, 1080, 30.0, 0.5);
/// assert!(bps > 0);
/// ```
#[must_use]
pub fn recommend(width: u32, height: u32, fps: f32, complexity: f32) -> u32 {
    recommend_impl(width, height, fps, complexity, 0.07)
}

#[allow(clippy::cast_precision_loss)]
fn recommend_impl(
    width: u32,
    height: u32,
    fps: f32,
    complexity: f32,
    bpp: f64,
) -> u32 {
    let w = width as f64;
    let h = height as f64;
    let f = fps as f64;
    let c = complexity.clamp(0.0, 1.0) as f64;
    (w * h * f * c * bpp) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recommend_nonzero_for_valid_params() {
        let bps = recommend(1920, 1080, 30.0, 0.5);
        assert!(bps > 0, "Expected nonzero bitrate, got {bps}");
    }

    #[test]
    fn test_recommend_formula_1080p_30fps_0_5_complexity() {
        // 1920 * 1080 * 30 * 0.5 * 0.07 = 2_177_280 bps ≈ 2.18 Mbps
        let bps = recommend(1920, 1080, 30.0, 0.5);
        assert!(
            bps > 2_000_000 && bps < 3_000_000,
            "1080p30 recommendation={bps}"
        );
    }

    #[test]
    fn test_recommend_zero_for_zero_width() {
        let bps = recommend(0, 1080, 30.0, 0.5);
        assert_eq!(bps, 0);
    }

    #[test]
    fn test_recommend_zero_complexity() {
        let bps = recommend(1920, 1080, 30.0, 0.0);
        assert_eq!(bps, 0);
    }

    #[test]
    fn test_recommend_higher_res_higher_bitrate() {
        let bps_720 = recommend(1280, 720, 30.0, 0.5);
        let bps_1080 = recommend(1920, 1080, 30.0, 0.5);
        assert!(bps_1080 > bps_720, "1080p should need more bits than 720p");
    }

    #[test]
    fn test_recommender_struct_default() {
        let rec = BitrateRecommender::new();
        let r = rec.recommend(1280, 720, 25.0, 0.6);
        // 1280 * 720 * 25 * 0.6 * 0.07 = 967_680 bps ≈ 0.97 Mbps
        assert!(r.bps > 0);
        assert!(r.kbps > 0.0);
        assert!(r.mbps > 0.0);
    }

    #[test]
    fn test_recommender_fields_consistent() {
        let r = BitrateRecommendation::from_bps(2_000_000);
        assert!((r.kbps - 2000.0).abs() < 0.1);
        assert!((r.mbps - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_recommender_complexity_clamps_at_one() {
        let bps_capped = recommend(1920, 1080, 30.0, 1.0);
        let bps_over = recommend(1920, 1080, 30.0, 2.0);
        assert_eq!(bps_capped, bps_over, "complexity should be clamped at 1.0");
    }
}
