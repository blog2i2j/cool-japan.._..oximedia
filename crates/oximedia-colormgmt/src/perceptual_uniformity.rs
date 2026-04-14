//! Perceptual uniformity analysis for color spaces.
//!
//! This module provides tools for analyzing and measuring the perceptual
//! uniformity of color spaces, including:
//!
//! - **MacAdam ellipses**: Standard just-noticeable difference (JND) regions
//!   in the CIE 1931 xy chromaticity diagram.
//! - **JND measurement**: Just-Noticeable Difference thresholds in Lab and
//!   JzCzhz color spaces.
//! - **Uniformity scoring**: Quantitative assessment of color space uniformity
//!   relative to MacAdam ellipses.
//! - **JzCzhz analysis**: Perceptual uniformity testing using the HDR-capable
//!   Jzazbz / JzCzhz color space.
//!
//! # Background
//!
//! A perfectly perceptually uniform color space would have equal Euclidean
//! distances corresponding to equal perceived color differences across the
//! entire color solid. In practice, all color spaces exhibit some deviation
//! from this ideal.
//!
//! MacAdam (1942) empirically determined regions of just-noticeable difference
//! around 25 reference chromaticities. These ellipses remain the standard
//! benchmark for evaluating color space uniformity.
//!
//! # Example
//!
//! ```
//! use oximedia_colormgmt::perceptual_uniformity::{
//!     MacAdamEllipse, JndAnalyzer, UniformityScore,
//! };
//!
//! // Get a MacAdam ellipse and check a nearby point
//! let ellipse = MacAdamEllipse::standard_set()[0];
//! let nearby = [ellipse.x + 0.001, ellipse.y + 0.001];
//! let steps = ellipse.steps_to_boundary(nearby[0], nearby[1]);
//! println!("JND steps from center: {:.2}", steps);
//!
//! // Analyze JND in Lab space
//! let analyzer = JndAnalyzer::new();
//! let lab = [50.0_f64, 10.0, 10.0];
//! let jnd = analyzer.lab_jnd_threshold(lab);
//! println!("Lab JND threshold: {:.4}", jnd);
//! ```

use std::f64::consts::PI;

// ── MacAdam Ellipse Data ─────────────────────────────────────────────────────

/// A MacAdam ellipse: a region of just-noticeable difference (JND = 1 step)
/// in the CIE 1931 xy chromaticity diagram.
///
/// The ellipse is defined by its centre (x, y), semi-axes (a, b in units of
/// 10⁻⁴), and tilt angle θ (radians, measured from the positive x-axis).
///
/// Reference: MacAdam, D.L. (1942). "Visual sensitivities to color differences
/// in daylight." JOSA 32(5):247-274.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MacAdamEllipse {
    /// CIE x chromaticity of the ellipse centre.
    pub x: f64,
    /// CIE y chromaticity of the ellipse centre.
    pub y: f64,
    /// Semi-major axis length (×10⁻⁴ chromaticity units).
    pub semi_major: f64,
    /// Semi-minor axis length (×10⁻⁴ chromaticity units).
    pub semi_minor: f64,
    /// Tilt angle of the semi-major axis (radians from positive x-axis).
    pub angle_rad: f64,
}

impl MacAdamEllipse {
    /// Create a new MacAdam ellipse.
    #[must_use]
    pub fn new(x: f64, y: f64, semi_major: f64, semi_minor: f64, angle_deg: f64) -> Self {
        Self {
            x,
            y,
            semi_major,
            semi_minor,
            angle_rad: angle_deg * PI / 180.0,
        }
    }

    /// Return the 25 standard MacAdam ellipses (from the 1942 paper).
    ///
    /// Semi-axes are given in units of 10⁻⁴ chromaticity (as tabulated).
    /// Angles are in degrees from the positive x-axis.
    ///
    /// Data from Wyszecki & Stiles, "Color Science" (2nd ed.), Table 8(3.3).
    #[must_use]
    pub fn standard_set() -> Vec<Self> {
        // (x, y, a×10^4, b×10^4, θ°)
        let raw: &[(f64, f64, f64, f64, f64)] = &[
            (0.160, 0.057, 1.500, 0.570, 162.0),
            (0.187, 0.118, 1.500, 0.750, 172.0),
            (0.253, 0.125, 2.000, 1.300, 103.0),
            (0.150, 0.680, 2.100, 0.950, 28.0),
            (0.131, 0.521, 2.300, 1.000, 3.0),
            (0.212, 0.550, 2.400, 1.200, 20.0),
            (0.258, 0.450, 2.600, 1.400, 15.0),
            (0.152, 0.365, 1.700, 1.000, 57.0),
            (0.280, 0.385, 2.800, 1.200, 60.0),
            (0.380, 0.498, 2.900, 1.000, 48.0),
            (0.160, 0.200, 1.400, 0.700, 163.0),
            (0.228, 0.250, 1.600, 1.200, 27.0),
            (0.305, 0.323, 3.300, 1.300, 31.0),
            (0.385, 0.393, 4.200, 1.600, 29.0),
            (0.472, 0.399, 4.100, 2.000, 40.0),
            (0.527, 0.350, 3.800, 2.100, 47.0),
            (0.475, 0.300, 3.100, 1.500, 55.0),
            (0.510, 0.236, 2.700, 1.200, 50.0),
            (0.596, 0.283, 4.100, 2.000, 25.0),
            (0.344, 0.284, 2.800, 1.200, 30.0),
            (0.390, 0.237, 3.300, 1.100, 34.0),
            (0.441, 0.198, 3.100, 1.300, 47.0),
            (0.278, 0.223, 2.100, 1.100, 17.0),
            (0.300, 0.163, 2.600, 0.900, 25.0),
            (0.365, 0.153, 3.400, 1.400, 30.0),
        ];

        raw.iter()
            .map(|&(x, y, a, b, theta)| Self::new(x, y, a * 1e-4, b * 1e-4, theta))
            .collect()
    }

    /// Compute the number of JND steps from the ellipse centre to a query
    /// chromaticity point (x_q, y_q).
    ///
    /// The computation projects the displacement vector onto the ellipse axes
    /// and computes the normalised Mahalanobis-like distance. A value ≤ 1.0
    /// means the point is within one JND of the centre.
    #[must_use]
    pub fn steps_to_boundary(&self, x_q: f64, y_q: f64) -> f64 {
        let dx = x_q - self.x;
        let dy = y_q - self.y;
        let cos_t = self.angle_rad.cos();
        let sin_t = self.angle_rad.sin();
        // Rotate into ellipse-aligned frame
        let u = dx * cos_t + dy * sin_t;
        let v = -dx * sin_t + dy * cos_t;
        let a = self.semi_major.max(1e-12);
        let b = self.semi_minor.max(1e-12);
        ((u / a).powi(2) + (v / b).powi(2)).sqrt()
    }

    /// Return `true` if the point (x_q, y_q) lies within one JND step of the
    /// ellipse centre.
    #[must_use]
    pub fn contains(&self, x_q: f64, y_q: f64) -> bool {
        self.steps_to_boundary(x_q, y_q) <= 1.0
    }

    /// Area of the ellipse in chromaticity units².
    #[must_use]
    pub fn area(&self) -> f64 {
        PI * self.semi_major * self.semi_minor
    }

    /// Aspect ratio (semi-major / semi-minor). A value of 1.0 means circular.
    #[must_use]
    pub fn aspect_ratio(&self) -> f64 {
        let b = self.semi_minor.max(1e-12);
        self.semi_major / b
    }
}

// ── JND Analyzer ─────────────────────────────────────────────────────────────

/// Configuration for JND threshold analysis.
#[derive(Debug, Clone)]
pub struct JndConfig {
    /// JND threshold in Lab ΔE (default 1.0).
    pub lab_threshold: f64,
    /// JND threshold in JzCzhz ΔJz (default 0.002).
    pub jzczhz_threshold: f64,
    /// Scale factor applied to chromaticity-based metrics (default 1.0).
    pub scale: f64,
}

impl Default for JndConfig {
    fn default() -> Self {
        Self {
            lab_threshold: 1.0,
            jzczhz_threshold: 0.002,
            scale: 1.0,
        }
    }
}

/// Analyzer for just-noticeable difference (JND) thresholds in multiple color
/// spaces.
///
/// The analyzer uses empirical models derived from the CIE 1976 (Lab) and
/// Safdar et al. Jzazbz color spaces to estimate how many JNDs separate two
/// colors.
#[derive(Debug, Clone)]
pub struct JndAnalyzer {
    /// Configuration for JND thresholds.
    pub config: JndConfig,
}

impl JndAnalyzer {
    /// Create an analyzer with default thresholds.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: JndConfig::default(),
        }
    }

    /// Create an analyzer with custom configuration.
    #[must_use]
    pub fn with_config(config: JndConfig) -> Self {
        Self { config }
    }

    /// Compute the perceptual distance in Lab space between two colors.
    ///
    /// Returns the CIE76 ΔE* (Euclidean distance in Lab).
    #[must_use]
    pub fn lab_distance(&self, lab1: [f64; 3], lab2: [f64; 3]) -> f64 {
        let dl = lab1[0] - lab2[0];
        let da = lab1[1] - lab2[1];
        let db = lab1[2] - lab2[2];
        (dl * dl + da * da + db * db).sqrt()
    }

    /// Estimate the JND threshold (minimum perceptible ΔE) at a given Lab
    /// location.
    ///
    /// The model accounts for the well-known non-uniformity of Lab space:
    /// - Threshold increases slightly with chroma (chromatic adaptation).
    /// - Threshold is higher in the blue-violet region (hue corrections).
    /// - Lightness modulation follows a parabolic model from psychophysics.
    ///
    /// This is an empirical approximation suitable for quality assessment.
    #[must_use]
    pub fn lab_jnd_threshold(&self, lab: [f64; 3]) -> f64 {
        let l = lab[0];
        let a = lab[1];
        let b = lab[2];
        let chroma = (a * a + b * b).sqrt();
        let hue_rad = b.atan2(a);

        // Lightness modulation: higher tolerance at mid-greys
        let l_factor = 1.0 + 0.015 * ((l - 50.0).powi(2) / (20.0 + (l - 50.0).powi(2)));

        // Chroma modulation: slightly more tolerant at higher chroma
        let c_factor = 1.0 + 0.045 * chroma;

        // Hue-specific correction: blue-violet region has higher threshold
        let hue_deg = hue_rad.to_degrees().rem_euclid(360.0);
        let h_correction = if (240.0..=300.0).contains(&hue_deg) {
            1.2
        } else {
            1.0
        };

        self.config.lab_threshold * l_factor * c_factor * h_correction * self.config.scale
    }

    /// Compute the number of JND steps in Lab space between two colors.
    ///
    /// Uses the local JND threshold at the midpoint of the two colors.
    #[must_use]
    pub fn lab_jnd_steps(&self, lab1: [f64; 3], lab2: [f64; 3]) -> f64 {
        let mid = [
            (lab1[0] + lab2[0]) / 2.0,
            (lab1[1] + lab2[1]) / 2.0,
            (lab1[2] + lab2[2]) / 2.0,
        ];
        let dist = self.lab_distance(lab1, lab2);
        let threshold = self.lab_jnd_threshold(mid);
        dist / threshold.max(1e-12)
    }

    /// Compute the perceptual distance in JzCzhz space between two Jzazbz
    /// color values.
    ///
    /// The JzCzhz distance accounts for the cylindrical geometry:
    /// ΔE = √(ΔJz² + ΔCz² + 2·Cz1·Cz2·(1 − cos(Δhz)))
    #[must_use]
    pub fn jzczhz_distance(&self, jab1: [f64; 3], jab2: [f64; 3]) -> f64 {
        // jab = [Jz, az, bz]
        let jz1 = jab1[0];
        let az1 = jab1[1];
        let bz1 = jab1[2];
        let jz2 = jab2[0];
        let az2 = jab2[1];
        let bz2 = jab2[2];

        let cz1 = (az1 * az1 + bz1 * bz1).sqrt();
        let cz2 = (az2 * az2 + bz2 * bz2).sqrt();
        let hz1 = bz1.atan2(az1);
        let hz2 = bz2.atan2(az2);

        let d_jz = jz2 - jz1;
        let d_cz = cz2 - cz1;
        let d_hz_cos = (hz2 - hz1).cos();
        let hue_term = 2.0 * cz1 * cz2 * (1.0 - d_hz_cos);

        (d_jz * d_jz + d_cz * d_cz + hue_term).sqrt()
    }

    /// Compute the number of JND steps in JzCzhz space.
    #[must_use]
    pub fn jzczhz_jnd_steps(&self, jab1: [f64; 3], jab2: [f64; 3]) -> f64 {
        let dist = self.jzczhz_distance(jab1, jab2);
        dist / self.config.jzczhz_threshold.max(1e-12)
    }
}

impl Default for JndAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Uniformity Score ─────────────────────────────────────────────────────────

/// A quantitative score of color space perceptual uniformity based on the
/// spread of MacAdam ellipse aspect ratios and areas.
///
/// A perfectly uniform color space would yield circular MacAdam ellipses
/// (aspect ratio 1.0) of equal area everywhere.
#[derive(Debug, Clone)]
pub struct UniformityScore {
    /// Mean aspect ratio across all ellipses (closer to 1 = more uniform).
    pub mean_aspect_ratio: f64,
    /// Standard deviation of aspect ratios (lower = more uniform).
    pub std_aspect_ratio: f64,
    /// Coefficient of variation of ellipse areas (lower = more uniform).
    pub area_cv: f64,
    /// Overall uniformity score in [0, 1] (1 = perfectly uniform).
    pub score: f64,
}

impl UniformityScore {
    /// Compute a uniformity score for a set of MacAdam ellipses.
    ///
    /// The score is derived from:
    /// 1. The mean and spread of aspect ratios.
    /// 2. The coefficient of variation of ellipse areas.
    ///
    /// Higher scores indicate more perceptually uniform spacing.
    #[must_use]
    pub fn from_ellipses(ellipses: &[MacAdamEllipse]) -> Self {
        if ellipses.is_empty() {
            return Self {
                mean_aspect_ratio: 1.0,
                std_aspect_ratio: 0.0,
                area_cv: 0.0,
                score: 1.0,
            };
        }

        let n = ellipses.len() as f64;
        let aspects: Vec<f64> = ellipses.iter().map(|e| e.aspect_ratio()).collect();
        let areas: Vec<f64> = ellipses.iter().map(|e| e.area()).collect();

        let mean_aspect = aspects.iter().sum::<f64>() / n;
        let var_aspect =
            aspects.iter().map(|&a| (a - mean_aspect).powi(2)).sum::<f64>() / n.max(1.0);
        let std_aspect = var_aspect.sqrt();

        let mean_area = areas.iter().sum::<f64>() / n;
        let var_area = areas
            .iter()
            .map(|&a| (a - mean_area).powi(2))
            .sum::<f64>()
            / n.max(1.0);
        let std_area = var_area.sqrt();
        let area_cv = if mean_area > 1e-20 {
            std_area / mean_area
        } else {
            0.0
        };

        // Score: penalise deviation from ideal aspect ratio (1.0) and area
        // non-uniformity.  Both penalties are in [0, ∞); we map to [0, 1].
        let aspect_penalty = (mean_aspect - 1.0).abs() + std_aspect;
        let area_penalty = area_cv;
        let raw_penalty = aspect_penalty + area_penalty;
        let score = (-raw_penalty).exp(); // ∈ (0, 1]

        Self {
            mean_aspect_ratio: mean_aspect,
            std_aspect_ratio: std_aspect,
            area_cv,
            score,
        }
    }
}

// ── JzCzhz Uniformity Test ───────────────────────────────────────────────────

/// Test the perceptual uniformity of the JzCzhz color space along the hue
/// circle at a fixed lightness and chroma.
///
/// Generates `n_samples` evenly-spaced hue angles and computes the adjacent
/// JzCzhz distances. For a perfectly uniform space all distances should be
/// equal. Returns the coefficient of variation (std / mean) of these
/// distances; a value close to 0 indicates high uniformity.
#[must_use]
pub fn jzczhz_hue_circle_uniformity(jz: f64, cz: f64, n_samples: usize) -> f64 {
    let n = n_samples.max(2);
    let step = 2.0 * PI / n as f64;

    let mut distances = Vec::with_capacity(n);
    let analyzer = JndAnalyzer::new();

    for i in 0..n {
        let h1 = i as f64 * step;
        let h2 = (i + 1) as f64 * step;
        let jab1 = [jz, cz * h1.cos(), cz * h1.sin()];
        let jab2 = [jz, cz * h2.cos(), cz * h2.sin()];
        distances.push(analyzer.jzczhz_distance(jab1, jab2));
    }

    let mean = distances.iter().sum::<f64>() / distances.len() as f64;
    if mean < 1e-20 {
        return 0.0;
    }
    let variance = distances
        .iter()
        .map(|&d| (d - mean).powi(2))
        .sum::<f64>()
        / distances.len() as f64;
    variance.sqrt() / mean
}

// ── MacAdam ellipse nearest-centre lookup ────────────────────────────────────

/// Find the MacAdam ellipse whose centre is nearest to the given chromaticity
/// (x, y) and return the number of JND steps to that centre.
///
/// Returns `None` if the ellipse set is empty.
#[must_use]
pub fn nearest_macadam_steps(
    x: f64,
    y: f64,
    ellipses: &[MacAdamEllipse],
) -> Option<(usize, f64)> {
    if ellipses.is_empty() {
        return None;
    }
    let (idx, _) = ellipses
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let dx = x - e.x;
            let dy = y - e.y;
            (i, dx * dx + dy * dy)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))?;

    let steps = ellipses[idx].steps_to_boundary(x, y);
    Some((idx, steps))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macadam_standard_set_count() {
        let ellipses = MacAdamEllipse::standard_set();
        assert_eq!(ellipses.len(), 25, "Should have 25 standard MacAdam ellipses");
    }

    #[test]
    fn test_macadam_contains_centre() {
        let ellipses = MacAdamEllipse::standard_set();
        for e in &ellipses {
            let steps = e.steps_to_boundary(e.x, e.y);
            assert!(
                steps < 1e-10,
                "Centre should have 0 JND steps, got {steps}"
            );
            assert!(e.contains(e.x, e.y), "Ellipse should contain its own centre");
        }
    }

    #[test]
    fn test_macadam_outside_point() {
        let e = MacAdamEllipse::standard_set()[12]; // centre at ~(0.305, 0.323)
        // Move 10× the semi-major axis away — definitely outside
        let far_x = e.x + e.semi_major * 10.0 * e.angle_rad.cos();
        let far_y = e.y + e.semi_major * 10.0 * e.angle_rad.sin();
        assert!(
            !e.contains(far_x, far_y),
            "Point 10× semi-major away should be outside"
        );
        let steps = e.steps_to_boundary(far_x, far_y);
        assert!(steps > 1.0, "Steps should exceed 1.0 outside ellipse");
    }

    #[test]
    fn test_macadam_area_positive() {
        for e in MacAdamEllipse::standard_set() {
            assert!(e.area() > 0.0, "Area must be positive");
        }
    }

    #[test]
    fn test_macadam_aspect_ratio_gte_one() {
        for e in MacAdamEllipse::standard_set() {
            assert!(
                e.aspect_ratio() >= 1.0 - 1e-10,
                "Aspect ratio must be >= 1 (semi_major >= semi_minor)"
            );
        }
    }

    #[test]
    fn test_jnd_analyzer_lab_distance_zero() {
        let a = JndAnalyzer::new();
        let lab = [50.0, 10.0, -10.0];
        assert!(
            a.lab_distance(lab, lab) < 1e-10,
            "Identical colors should have zero distance"
        );
    }

    #[test]
    fn test_jnd_analyzer_lab_steps() {
        let a = JndAnalyzer::new();
        let lab1 = [50.0, 0.0, 0.0];
        let lab2 = [51.0, 0.0, 0.0]; // 1 L* unit difference
        let steps = a.lab_jnd_steps(lab1, lab2);
        // Should be roughly 1 step (threshold is ~1.0 at grey)
        assert!(steps > 0.0, "Steps must be positive");
        assert!(steps < 5.0, "Steps should be reasonable for small shift");
    }

    #[test]
    fn test_jnd_analyzer_jzczhz_distance_zero() {
        let a = JndAnalyzer::new();
        let jab = [0.5, 0.1, 0.1];
        assert!(
            a.jzczhz_distance(jab, jab) < 1e-10,
            "Identical Jzazbz values should have zero distance"
        );
    }

    #[test]
    fn test_jnd_analyzer_jzczhz_steps_positive() {
        let a = JndAnalyzer::new();
        let jab1 = [0.5, 0.1, 0.0];
        let jab2 = [0.5, 0.0, 0.1];
        let steps = a.jzczhz_jnd_steps(jab1, jab2);
        assert!(steps > 0.0, "Steps must be positive for different colors");
    }

    #[test]
    fn test_uniformity_score_perfect_circles() {
        // All ellipses equal semi-axes → aspect ratio = 1 → high score
        let ellipses: Vec<MacAdamEllipse> = (0..8)
            .map(|i| MacAdamEllipse::new(0.1 + i as f64 * 0.05, 0.3, 0.01, 0.01, 0.0))
            .collect();
        let score = UniformityScore::from_ellipses(&ellipses);
        assert!(
            score.score > 0.5,
            "Score for perfect circles should be high, got {}",
            score.score
        );
    }

    #[test]
    fn test_uniformity_score_empty() {
        let score = UniformityScore::from_ellipses(&[]);
        assert_eq!(score.score, 1.0, "Empty set should yield score 1.0");
    }

    #[test]
    fn test_jzczhz_hue_circle_uniformity() {
        // At a fixed Jz and Cz, the hue circle should be fairly uniform
        // (coefficient of variation should be well below 1.0)
        let cv = jzczhz_hue_circle_uniformity(0.5, 0.1, 36);
        assert!(
            cv < 0.5,
            "Hue circle CV should indicate reasonable uniformity, got {cv}"
        );
    }

    #[test]
    fn test_nearest_macadam_steps_found() {
        let ellipses = MacAdamEllipse::standard_set();
        let result = nearest_macadam_steps(0.305, 0.323, &ellipses);
        assert!(result.is_some(), "Should find nearest ellipse");
        let (idx, steps) = result.unwrap();
        // Ellipse 12 is (0.305, 0.323) — should be the nearest and have ~0 steps
        assert_eq!(idx, 12, "Nearest ellipse index should be 12");
        assert!(steps < 0.1, "Steps to centre should be near zero");
    }

    #[test]
    fn test_nearest_macadam_steps_empty() {
        let result = nearest_macadam_steps(0.3, 0.3, &[]);
        assert!(result.is_none(), "Empty set should return None");
    }
}
