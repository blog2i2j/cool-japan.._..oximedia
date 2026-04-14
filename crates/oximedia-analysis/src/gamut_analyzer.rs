//! Color gamut analysis: detect out-of-gamut pixels for SDR/HDR content.
//!
//! Provides [`GamutAnalyzer`] which counts pixels whose saturation exceeds the
//! maximum allowed by a target [`Gamut`] descriptor.

/// Color gamut descriptor.
///
/// Defines the maximum chroma saturation (0.0–1.0) that is considered
/// in-gamut for a particular color space standard.
#[derive(Debug, Clone)]
pub struct Gamut {
    /// Human-readable name (e.g. "Rec.709", "DCI-P3").
    pub name: &'static str,
    /// Maximum saturation value in [0.0, 1.0] that is considered in-gamut.
    /// Pixels with saturation > this value are out of gamut.
    pub max_saturation: f32,
}

impl Gamut {
    /// Create a custom gamut descriptor.
    #[must_use]
    pub fn custom(name: &'static str, max_saturation: f32) -> Self {
        Self {
            name,
            max_saturation: max_saturation.clamp(0.0, 1.0),
        }
    }

    /// Rec.709 gamut (standard SDR broadcast — saturation cap ≈ 0.78).
    #[must_use]
    pub fn rec709() -> Self {
        Self {
            name: "Rec.709",
            max_saturation: 0.78,
        }
    }

    /// DCI-P3 gamut (cinema — wider than Rec.709 but narrower than Rec.2020).
    #[must_use]
    pub fn dci_p3() -> Self {
        Self {
            name: "DCI-P3",
            max_saturation: 0.90,
        }
    }

    /// Rec.2020 gamut (HDR wide gamut — effectively allows all saturations ≤ 1.0).
    #[must_use]
    pub fn rec2020() -> Self {
        Self {
            name: "Rec.2020",
            max_saturation: 1.0,
        }
    }

    /// sRGB gamut — same gamut as Rec.709 but with a different transfer function.
    #[must_use]
    pub fn srgb() -> Self {
        Self {
            name: "sRGB",
            max_saturation: 0.78,
        }
    }
}

/// Result of a gamut analysis pass.
#[derive(Debug, Clone)]
pub struct GamutAnalysisResult {
    /// Fraction of pixels (0.0–1.0) that were out of gamut.
    pub out_of_gamut_ratio: f32,
    /// Total number of pixels analysed.
    pub total_pixels: u64,
    /// Number of pixels that were out of gamut.
    pub out_of_gamut_pixels: u64,
    /// Gamut used for the analysis.
    pub gamut_name: &'static str,
}

impl GamutAnalysisResult {
    /// Returns `true` if more than `threshold` of pixels are out of gamut.
    #[must_use]
    pub fn exceeds(&self, threshold: f32) -> bool {
        self.out_of_gamut_ratio > threshold
    }
}

/// Colour gamut analyser.
///
/// Accepts packed RGB24 frames and counts pixels whose saturation exceeds the
/// gamut's `max_saturation`.
///
/// # Saturation formula
///
/// Saturation is computed as:
/// ```text
/// max_rgb = max(R, G, B) / 255.0
/// min_rgb = min(R, G, B) / 255.0
/// saturation = if max_rgb > 0.0 { (max_rgb - min_rgb) / max_rgb } else { 0.0 }
/// ```
pub struct GamutAnalyzer;

impl GamutAnalyzer {
    /// Compute the fraction of out-of-gamut pixels in an RGB24 frame.
    ///
    /// # Arguments
    ///
    /// * `frame` – packed RGB24 bytes (`w * h * 3`)
    /// * `w` – frame width in pixels
    /// * `h` – frame height in pixels
    /// * `gamut` – the target gamut to check against
    ///
    /// # Returns
    ///
    /// Fraction in [0.0, 1.0] of pixels that are out of gamut, or 0.0 if the
    /// input dimensions are invalid.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn out_of_gamut_ratio(frame: &[u8], w: u32, h: u32, gamut: &Gamut) -> f32 {
        let total = (w as usize) * (h as usize);
        if frame.len() < total * 3 || total == 0 {
            return 0.0;
        }

        let mut out_of_gamut = 0u64;

        for chunk in frame[..total * 3].chunks_exact(3) {
            let r = chunk[0] as f32 / 255.0;
            let g = chunk[1] as f32 / 255.0;
            let b = chunk[2] as f32 / 255.0;

            let max_c = r.max(g).max(b);
            let min_c = r.min(g).min(b);

            let saturation = if max_c > f32::EPSILON {
                (max_c - min_c) / max_c
            } else {
                0.0
            };

            if saturation > gamut.max_saturation {
                out_of_gamut += 1;
            }
        }

        out_of_gamut as f32 / total as f32
    }

    /// Full analysis result with pixel counts.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze(frame: &[u8], w: u32, h: u32, gamut: &Gamut) -> GamutAnalysisResult {
        let total = (w as usize) * (h as usize);
        if frame.len() < total * 3 || total == 0 {
            return GamutAnalysisResult {
                out_of_gamut_ratio: 0.0,
                total_pixels: 0,
                out_of_gamut_pixels: 0,
                gamut_name: gamut.name,
            };
        }

        let mut out_of_gamut = 0u64;
        for chunk in frame[..total * 3].chunks_exact(3) {
            let r = chunk[0] as f32 / 255.0;
            let g = chunk[1] as f32 / 255.0;
            let b = chunk[2] as f32 / 255.0;
            let max_c = r.max(g).max(b);
            let min_c = r.min(g).min(b);
            let sat = if max_c > f32::EPSILON { (max_c - min_c) / max_c } else { 0.0 };
            if sat > gamut.max_saturation {
                out_of_gamut += 1;
            }
        }

        let ratio = out_of_gamut as f32 / total as f32;
        GamutAnalysisResult {
            out_of_gamut_ratio: ratio,
            total_pixels: total as u64,
            out_of_gamut_pixels: out_of_gamut,
            gamut_name: gamut.name,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_grey(n: usize) -> Vec<u8> {
        vec![128u8; n * 3]
    }

    #[test]
    fn test_grey_frame_no_out_of_gamut() {
        // Grey = equal R/G/B → saturation 0 → always in gamut
        let frame = all_grey(64 * 64);
        let ratio = GamutAnalyzer::out_of_gamut_ratio(&frame, 64, 64, &Gamut::rec709());
        assert!((ratio - 0.0).abs() < f32::EPSILON, "ratio={ratio}");
    }

    #[test]
    fn test_pure_red_exceeds_rec709() {
        // Pure red (255, 0, 0) → saturation = 1.0 → out of Rec.709 gamut (max 0.78)
        let frame = vec![255u8, 0, 0, 255, 0, 0];
        let ratio = GamutAnalyzer::out_of_gamut_ratio(&frame, 2, 1, &Gamut::rec709());
        assert!((ratio - 1.0).abs() < f32::EPSILON, "ratio={ratio}");
    }

    #[test]
    fn test_pure_red_within_rec2020() {
        // Rec.2020 allows saturation up to 1.0 → pure red is in gamut
        let frame = vec![255u8, 0, 0, 255, 0, 0];
        let ratio = GamutAnalyzer::out_of_gamut_ratio(&frame, 2, 1, &Gamut::rec2020());
        assert!((ratio - 0.0).abs() < f32::EPSILON, "rec2020 should accept pure red");
    }

    #[test]
    fn test_empty_frame_returns_zero() {
        let ratio = GamutAnalyzer::out_of_gamut_ratio(&[], 0, 0, &Gamut::rec709());
        assert!((ratio).abs() < f32::EPSILON);
    }

    #[test]
    fn test_analyze_returns_correct_pixel_count() {
        // 4 pure-red pixels (12 bytes total)
        let frame: Vec<u8> = std::iter::repeat([255u8, 0, 0]).take(4).flatten().collect();
        let r = GamutAnalyzer::analyze(&frame, 2, 2, &Gamut::rec709());
        assert_eq!(r.total_pixels, 4);
        assert_eq!(r.out_of_gamut_pixels, 4);
        assert!((r.out_of_gamut_ratio - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_gamut_names() {
        assert_eq!(Gamut::rec709().name, "Rec.709");
        assert_eq!(Gamut::dci_p3().name, "DCI-P3");
        assert_eq!(Gamut::rec2020().name, "Rec.2020");
    }

    #[test]
    fn test_result_exceeds() {
        let r = GamutAnalysisResult {
            out_of_gamut_ratio: 0.15,
            total_pixels: 100,
            out_of_gamut_pixels: 15,
            gamut_name: "test",
        };
        assert!(r.exceeds(0.10));
        assert!(!r.exceeds(0.20));
    }

    #[test]
    fn test_custom_gamut() {
        // Custom gamut with very low saturation cap (0.1)
        let custom = Gamut::custom("Custom-Tight", 0.1);
        // A pixel with moderate saturation (e.g. R=200, G=128, B=128):
        // max=200/255≈0.784, min=128/255≈0.502 → sat≈(0.784-0.502)/0.784≈0.36 > 0.1 → out of gamut
        let frame = vec![200u8, 128, 128];
        let ratio = GamutAnalyzer::out_of_gamut_ratio(&frame, 1, 1, &custom);
        assert!((ratio - 1.0).abs() < f32::EPSILON, "pixel should be out of tight custom gamut, ratio={ratio}");
    }

    #[test]
    fn test_half_out_of_gamut() {
        // 2 pixels: one grey (in gamut), one pure red (out of gamut for Rec.709)
        let frame = vec![
            128u8, 128, 128, // grey → sat=0 → in gamut
            255, 0, 0,       // pure red → sat=1.0 → out of Rec.709 gamut
        ];
        let r = GamutAnalyzer::analyze(&frame, 2, 1, &Gamut::rec709());
        assert_eq!(r.total_pixels, 2);
        assert_eq!(r.out_of_gamut_pixels, 1);
        assert!((r.out_of_gamut_ratio - 0.5).abs() < f32::EPSILON, "ratio={}", r.out_of_gamut_ratio);
    }
}
