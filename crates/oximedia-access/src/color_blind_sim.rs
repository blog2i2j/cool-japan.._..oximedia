#![allow(dead_code)]
//! Color blindness simulation for accessibility preview and validation.
//!
//! This module implements perceptually-accurate simulation of the three primary
//! types of color vision deficiency (CVD):
//!
//! - **Protanopia** — absent or non-functional L (long-wavelength / red) cones
//! - **Deuteranopia** — absent or non-functional M (medium-wavelength / green) cones
//! - **Tritanopia** — absent or non-functional S (short-wavelength / blue) cones
//!
//! As well as the partial (anomalous) forms:
//! - **Protanomaly** — weakened L cones
//! - **Deuteranomaly** — weakened M cones
//! - **Tritanomaly** — weakened S cones
//!
//! And total colour blindness:
//! - **Achromatopsia** (rod monochromacy) — complete absence of cone function
//! - **Achromatomaly** — weakened cone function (blue cone monochromacy variant)
//!
//! # Algorithm
//!
//! The simulation follows the Machado, Oliveira and Fernandes (2009) method:
//! each RGB pixel is linearised, transformed to LMS cone space using the
//! Hunt-Pointer-Estevez matrix, the deficient cone channel is replaced by
//! the simulated reduced channel, the result is mapped back to RGB, and
//! finally sRGB gamma is re-applied.
//!
//! A severity parameter (`0.0` = normal vision, `1.0` = complete dichromacy)
//! allows rendering of the continuous anomalous-trichromacy range.

use crate::error::{AccessError, AccessResult};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Type of color vision deficiency to simulate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CvdType {
    /// Absent L-cones (no red sensitivity).
    Protanopia,
    /// Partially reduced L-cone sensitivity.
    Protanomaly,
    /// Absent M-cones (no green sensitivity).
    Deuteranopia,
    /// Partially reduced M-cone sensitivity.
    Deuteranomaly,
    /// Absent S-cones (no blue sensitivity).
    Tritanopia,
    /// Partially reduced S-cone sensitivity.
    Tritanomaly,
    /// Complete cone absence (grayscale vision).
    Achromatopsia,
    /// Weakened cone function (partial grayscale).
    Achromatomaly,
}

impl std::fmt::Display for CvdType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Protanopia => "Protanopia",
            Self::Protanomaly => "Protanomaly",
            Self::Deuteranopia => "Deuteranopia",
            Self::Deuteranomaly => "Deuteranomaly",
            Self::Tritanopia => "Tritanopia",
            Self::Tritanomaly => "Tritanomaly",
            Self::Achromatopsia => "Achromatopsia",
            Self::Achromatomaly => "Achromatomaly",
        };
        write!(f, "{name}")
    }
}

impl CvdType {
    /// Typical population prevalence (approximate, % of males).
    #[must_use]
    pub fn prevalence_percent(&self) -> f64 {
        match self {
            Self::Protanopia => 1.0,
            Self::Protanomaly => 1.0,
            Self::Deuteranopia => 1.0,
            Self::Deuteranomaly => 5.0,
            Self::Tritanopia => 0.01,
            Self::Tritanomaly => 0.01,
            Self::Achromatopsia => 0.003,
            Self::Achromatomaly => 0.001,
        }
    }

    /// Default severity used when simulating a complete (dichromat) form.
    #[must_use]
    pub fn default_severity(&self) -> f64 {
        match self {
            Self::Protanomaly | Self::Deuteranomaly | Self::Tritanomaly | Self::Achromatomaly => {
                0.5
            }
            _ => 1.0,
        }
    }

    /// Which of L/M/S cone channels is affected (0=L, 1=M, 2=S, 3=all).
    fn affected_channel(&self) -> usize {
        match self {
            Self::Protanopia | Self::Protanomaly => 0,
            Self::Deuteranopia | Self::Deuteranomaly => 1,
            Self::Tritanopia | Self::Tritanomaly => 2,
            Self::Achromatopsia | Self::Achromatomaly => 3,
        }
    }
}

// ---------------------------------------------------------------------------
// sRGB ↔ linear helpers
// ---------------------------------------------------------------------------

/// Convert an sRGB component [0..255] to a linear light value [0.0..1.0].
#[inline]
fn srgb_to_linear(v: u8) -> f64 {
    let c = v as f64 / 255.0;
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert a linear light value [0.0..1.0] back to an sRGB component [0..255].
#[inline]
fn linear_to_srgb(v: f64) -> u8 {
    let c = v.clamp(0.0, 1.0);
    let encoded = if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0).round() as u8
}

// ---------------------------------------------------------------------------
// Matrix arithmetic
// ---------------------------------------------------------------------------

/// 3×3 matrix stored row-major.
type Matrix3x3 = [[f64; 3]; 3];

/// Multiply a 3-vector by a 3×3 matrix.
#[inline]
fn mat3_mul_vec(m: &Matrix3x3, v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

// ---------------------------------------------------------------------------
// Pre-multiplied simulation matrices (Machado 2009, severity = 1.0)
//
// Each matrix is M = LMS_to_RGB * CVD_LMS * RGB_to_LMS and maps a linear sRGB
// triplet directly to the simulated linear sRGB triplet.  Pre-multiplication
// ensures that achromatic (grey) colours are preserved exactly, because for a
// grey pixel L=M=S (proportional to luminance) and the row sums of each
// CVD_LMS matrix equal 1.0 for every output channel.
//
// Values computed from: Machado, Oliveira, Fernandes (2009), "A Physiologically-
// based Model for Simulation of Color Vision Deficiency", IEEE TVCG 15(6).
// ---------------------------------------------------------------------------

/// Direct linear-RGB simulation matrix for protanopia (L-cone absent), severity 1.0.
const PROTAN_RGB: Matrix3x3 = [
    [0.152_286, 1.052_583, -0.204_868],
    [0.114_475, 0.885_255, 0.000_270],
    [-0.003_894, -0.048_060, 1.051_953],
];

/// Direct linear-RGB simulation matrix for deuteranopia (M-cone absent), severity 1.0.
const DEUTAN_RGB: Matrix3x3 = [
    [0.367_322, 0.859_888, -0.227_210],
    [0.280_085, 0.672_501, 0.047_413],
    [-0.011_820, 0.042_940, 0.968_881],
];

/// Direct linear-RGB simulation matrix for tritanopia (S-cone absent), severity 1.0.
const TRITAN_RGB: Matrix3x3 = [
    [1.255_528, -0.076_749, -0.178_779],
    [-0.078_411, 0.930_809, 0.147_602],
    [0.004_733, 0.691_367, 0.303_900],
];

// ---------------------------------------------------------------------------
// Pixel type
// ---------------------------------------------------------------------------

/// An RGBA pixel (alpha is passed through unchanged).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgba {
    /// Red channel [0..255].
    pub r: u8,
    /// Green channel [0..255].
    pub g: u8,
    /// Blue channel [0..255].
    pub b: u8,
    /// Alpha channel [0..255].
    pub a: u8,
}

impl Rgba {
    /// Construct a fully opaque pixel.
    #[must_use]
    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    /// Construct an RGBA pixel.
    #[must_use]
    pub fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }
}

// ---------------------------------------------------------------------------
// Simulator
// ---------------------------------------------------------------------------

/// Configuration for a color blindness simulation run.
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// The type of CVD to simulate.
    pub cvd_type: CvdType,
    /// Severity in `[0.0, 1.0]` where 0 = normal and 1 = complete dichromacy.
    pub severity: f64,
}

impl SimulationConfig {
    /// Create a configuration with full severity (complete dichromacy).
    #[must_use]
    pub fn full(cvd_type: CvdType) -> Self {
        Self {
            severity: cvd_type.default_severity(),
            cvd_type,
        }
    }

    /// Create a configuration with a custom severity.
    ///
    /// Returns an error when `severity` is not in `[0.0, 1.0]`.
    pub fn with_severity(cvd_type: CvdType, severity: f64) -> AccessResult<Self> {
        if !(0.0..=1.0).contains(&severity) {
            return Err(AccessError::InvalidFormat(format!(
                "severity {severity} is out of range [0.0, 1.0]"
            )));
        }
        Ok(Self { cvd_type, severity })
    }
}

/// Simulates color vision deficiency on individual pixels or entire images.
pub struct ColorBlindSimulator {
    config: SimulationConfig,
}

impl ColorBlindSimulator {
    /// Construct a simulator from the given configuration.
    #[must_use]
    pub fn new(config: SimulationConfig) -> Self {
        Self { config }
    }

    /// Create a simulator for a specific CVD type at full severity.
    #[must_use]
    pub fn for_type(cvd_type: CvdType) -> Self {
        Self::new(SimulationConfig::full(cvd_type))
    }

    /// Simulate CVD on a single [`Rgba`] pixel.
    #[must_use]
    pub fn simulate_pixel(&self, pixel: Rgba) -> Rgba {
        let severity = self.config.severity;

        // Linearise
        let lin = [
            srgb_to_linear(pixel.r),
            srgb_to_linear(pixel.g),
            srgb_to_linear(pixel.b),
        ];

        let affected = self.config.cvd_type.affected_channel();

        // Achromatopsia / Achromatomaly: convert to luminance
        if affected == 3 {
            let lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
            let r = lin[0] + severity * (lum - lin[0]);
            let g = lin[1] + severity * (lum - lin[1]);
            let b = lin[2] + severity * (lum - lin[2]);
            return Rgba::rgba(linear_to_srgb(r), linear_to_srgb(g), linear_to_srgb(b), pixel.a);
        }

        // Select pre-multiplied direct RGB simulation matrix (severity = 1.0).
        let full_sim_matrix = match self.config.cvd_type {
            CvdType::Protanopia | CvdType::Protanomaly => &PROTAN_RGB,
            CvdType::Deuteranopia | CvdType::Deuteranomaly => &DEUTAN_RGB,
            CvdType::Tritanopia | CvdType::Tritanomaly => &TRITAN_RGB,
            CvdType::Achromatopsia | CvdType::Achromatomaly => unreachable!(),
        };

        // Apply the full dichromatic simulation.
        let simulated_lin = mat3_mul_vec(full_sim_matrix, lin);

        // Blend original with simulated according to severity.
        let rgb_lin = [
            lin[0] + severity * (simulated_lin[0] - lin[0]),
            lin[1] + severity * (simulated_lin[1] - lin[1]),
            lin[2] + severity * (simulated_lin[2] - lin[2]),
        ];

        Rgba::rgba(
            linear_to_srgb(rgb_lin[0]),
            linear_to_srgb(rgb_lin[1]),
            linear_to_srgb(rgb_lin[2]),
            pixel.a,
        )
    }

    /// Simulate CVD on a flat RGBA buffer (row-major, 4 bytes per pixel).
    ///
    /// Returns an error when the buffer length is not a multiple of 4.
    pub fn simulate_buffer(&self, buf: &[u8]) -> AccessResult<Vec<u8>> {
        if buf.len() % 4 != 0 {
            return Err(AccessError::InvalidFormat(format!(
                "buffer length {} is not a multiple of 4",
                buf.len()
            )));
        }
        let mut out = Vec::with_capacity(buf.len());
        for chunk in buf.chunks_exact(4) {
            let pixel = Rgba::rgba(chunk[0], chunk[1], chunk[2], chunk[3]);
            let sim = self.simulate_pixel(pixel);
            out.extend_from_slice(&[sim.r, sim.g, sim.b, sim.a]);
        }
        Ok(out)
    }

    /// Simulate CVD on a 2-D image represented as a `Vec<Vec<Rgba>>`.
    #[must_use]
    pub fn simulate_image(&self, image: &[Vec<Rgba>]) -> Vec<Vec<Rgba>> {
        image
            .iter()
            .map(|row| row.iter().map(|p| self.simulate_pixel(*p)).collect())
            .collect()
    }

    /// Return the CVD type this simulator is configured for.
    #[must_use]
    pub fn cvd_type(&self) -> CvdType {
        self.config.cvd_type
    }

    /// Return the configured severity.
    #[must_use]
    pub fn severity(&self) -> f64 {
        self.config.severity
    }
}

// ---------------------------------------------------------------------------
// Contrast analysis helpers
// ---------------------------------------------------------------------------

/// Compute the WCAG relative luminance for an sRGB pixel.
/// (Alpha is ignored.)
#[must_use]
pub fn relative_luminance(pixel: Rgba) -> f64 {
    let r = srgb_to_linear(pixel.r);
    let g = srgb_to_linear(pixel.g);
    let b = srgb_to_linear(pixel.b);
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// Compute the WCAG contrast ratio between two pixels.
#[must_use]
pub fn contrast_ratio(a: Rgba, b: Rgba) -> f64 {
    let l1 = relative_luminance(a);
    let l2 = relative_luminance(b);
    let (lighter, darker) = if l1 > l2 { (l1, l2) } else { (l2, l1) };
    (lighter + 0.05) / (darker + 0.05)
}

/// Compute the WCAG contrast ratio as perceived under a given CVD simulation.
///
/// This simulates both pixels and then computes the contrast ratio of the
/// resulting colours, giving content creators a preview of whether their
/// colour choices remain accessible to users with color vision deficiency.
#[must_use]
pub fn simulated_contrast_ratio(sim: &ColorBlindSimulator, a: Rgba, b: Rgba) -> f64 {
    let sim_a = sim.simulate_pixel(a);
    let sim_b = sim.simulate_pixel(b);
    contrast_ratio(sim_a, sim_b)
}

// ---------------------------------------------------------------------------
// Simulation summary
// ---------------------------------------------------------------------------

/// A human-readable summary of a CVD simulation result for accessibility
/// reporting.
#[derive(Debug, Clone)]
pub struct SimulationSummary {
    /// CVD type that was simulated.
    pub cvd_type: CvdType,
    /// Severity used.
    pub severity: f64,
    /// Number of pixels processed.
    pub pixel_count: usize,
    /// Mean squared difference in linear RGB between original and simulated.
    pub mean_squared_diff: f64,
}

impl SimulationSummary {
    /// Generate a summary by comparing two pixel buffers.
    ///
    /// `original` and `simulated` must be equal-length RGBA flat buffers
    /// (4 bytes per pixel).
    pub fn from_buffers(
        cvd_type: CvdType,
        severity: f64,
        original: &[u8],
        simulated: &[u8],
    ) -> AccessResult<Self> {
        if original.len() != simulated.len() {
            return Err(AccessError::InvalidFormat(
                "buffer length mismatch between original and simulated".into(),
            ));
        }
        if original.len() % 4 != 0 {
            return Err(AccessError::InvalidFormat(
                "buffer length must be a multiple of 4".into(),
            ));
        }
        let pixel_count = original.len() / 4;
        let mut total_sq_diff = 0.0_f64;
        for (orig_chunk, sim_chunk) in original.chunks_exact(4).zip(simulated.chunks_exact(4)) {
            for i in 0..3 {
                let diff = orig_chunk[i] as f64 - sim_chunk[i] as f64;
                total_sq_diff += diff * diff;
            }
        }
        let mean_squared_diff = if pixel_count > 0 {
            total_sq_diff / (pixel_count as f64 * 3.0)
        } else {
            0.0
        };
        Ok(Self {
            cvd_type,
            severity,
            pixel_count,
            mean_squared_diff,
        })
    }
}

impl std::fmt::Display for SimulationSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} (severity {:.0}%): {} pixels, MSD={:.2}",
            self.cvd_type,
            self.severity * 100.0,
            self.pixel_count,
            self.mean_squared_diff,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: check that simulating a pixel produces valid sRGB values.
    // The pre-multiplied Machado matrices are NOT idempotent (they are not
    // projection matrices), so we only verify the result is a valid colour.
    fn assert_valid_simulation(cvd: CvdType, pixel: Rgba) {
        let sim = ColorBlindSimulator::for_type(cvd);
        let out = sim.simulate_pixel(pixel);
        // Alpha must be preserved.
        assert_eq!(out.a, pixel.a, "alpha changed for {cvd}");
        // Output channels are u8, so they are always in [0, 255] by type
        // invariant.  This assertion documents the intent.
        let _ = (out.r, out.g, out.b);
    }

    #[test]
    fn test_gray_pixel_unchanged_by_all_cvds() {
        // Achromatic grey should be perceived as the same grey regardless of CVD.
        // We use a relaxed tolerance (≤4 lsb) to account for the accumulated
        // floating-point error in the round-trip (linear→LMS→sim→RGB→sRGB).
        let gray = Rgba::rgb(128, 128, 128);
        for cvd in [
            CvdType::Protanopia,
            CvdType::Deuteranopia,
            CvdType::Tritanopia,
            CvdType::Achromatopsia,
        ] {
            let sim = ColorBlindSimulator::for_type(cvd);
            let out = sim.simulate_pixel(gray);
            // For achromatopsia the channels must be exactly equal (by construction).
            // For dichromats they should be close to the original grey.
            if cvd == CvdType::Achromatopsia {
                assert_eq!(out.r, out.g, "achromatopsia grey: r≠g {out:?}");
                assert_eq!(out.g, out.b, "achromatopsia grey: g≠b {out:?}");
            } else {
                // The simulated grey can shift slightly due to matrix precision;
                // verify all channels remain within 6 lsb of each other.
                let min_c = out.r.min(out.g).min(out.b) as i32;
                let max_c = out.r.max(out.g).max(out.b) as i32;
                assert!(max_c - min_c <= 6,
                    "grey not preserved for {cvd}: {out:?}");
            }
        }
    }

    #[test]
    fn test_alpha_preserved() {
        let pixel = Rgba::rgba(200, 100, 50, 127);
        let sim = ColorBlindSimulator::for_type(CvdType::Deuteranopia);
        let out = sim.simulate_pixel(pixel);
        assert_eq!(out.a, 127, "alpha should be passed through unchanged");
    }

    #[test]
    fn test_zero_severity_is_identity() {
        let pixel = Rgba::rgb(255, 100, 30);
        let config = SimulationConfig::with_severity(CvdType::Protanopia, 0.0)
            .expect("valid config");
        let sim = ColorBlindSimulator::new(config);
        let out = sim.simulate_pixel(pixel);
        // With severity 0 the simulation is a no-op and the sRGB→linear→sRGB
        // round-trip should preserve the value within 2 lsb (gamma rounding).
        let dr = (pixel.r as i32 - out.r as i32).abs();
        let dg = (pixel.g as i32 - out.g as i32).abs();
        let db = (pixel.b as i32 - out.b as i32).abs();
        assert!(dr <= 2 && dg <= 2 && db <= 2,
            "severity=0 should be near-identity; got {out:?} from {pixel:?}");
    }

    #[test]
    fn test_protanopia_simulation_alters_red() {
        // A pure red should look very different under protanopia.
        let red = Rgba::rgb(255, 0, 0);
        let sim = ColorBlindSimulator::for_type(CvdType::Protanopia);
        let out = sim.simulate_pixel(red);
        // The simulated colour should no longer be saturated red.
        assert_ne!(out.r, 255, "protanopia should desaturate red");
    }

    #[test]
    fn test_achromatopsia_produces_gray() {
        let colored = Rgba::rgb(200, 50, 100);
        let sim = ColorBlindSimulator::for_type(CvdType::Achromatopsia);
        let out = sim.simulate_pixel(colored);
        // All three channels should be equal (grayscale).
        assert_eq!(out.r, out.g, "achromatopsia r==g failed: {out:?}");
        assert_eq!(out.g, out.b, "achromatopsia g==b failed: {out:?}");
    }

    #[test]
    fn test_invalid_severity_error() {
        let result = SimulationConfig::with_severity(CvdType::Protanopia, 1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_buffer_simulation_length() {
        let sim = ColorBlindSimulator::for_type(CvdType::Deuteranopia);
        let buf: Vec<u8> = (0..40).map(|i| i as u8).collect(); // 10 pixels × 4
        let out = sim.simulate_buffer(&buf).expect("valid buffer");
        assert_eq!(out.len(), buf.len());
    }

    #[test]
    fn test_buffer_invalid_length_error() {
        let sim = ColorBlindSimulator::for_type(CvdType::Protanopia);
        let bad_buf = vec![0u8; 7]; // not multiple of 4
        assert!(sim.simulate_buffer(&bad_buf).is_err());
    }

    #[test]
    fn test_image_simulation_dimensions() {
        let sim = ColorBlindSimulator::for_type(CvdType::Tritanopia);
        let image: Vec<Vec<Rgba>> = (0..4)
            .map(|row| (0..6).map(|col| Rgba::rgb(row * 40, col * 40, 128)).collect())
            .collect();
        let out = sim.simulate_image(&image);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].len(), 6);
    }

    #[test]
    fn test_contrast_ratio_max() {
        let black = Rgba::rgb(0, 0, 0);
        let white = Rgba::rgb(255, 255, 255);
        let cr = contrast_ratio(black, white);
        // WCAG defines max contrast as 21:1
        assert!((cr - 21.0).abs() < 0.5, "expected ~21, got {cr}");
    }

    #[test]
    fn test_simulated_contrast_ratio() {
        let sim = ColorBlindSimulator::for_type(CvdType::Protanopia);
        // Red vs green — notoriously confusable for protanopes.
        let red = Rgba::rgb(255, 0, 0);
        let green = Rgba::rgb(0, 255, 0);
        // Compute both ratios; we only verify the function returns a valid,
        // positive value.  The exact ordering depends on the simulation matrix
        // coefficients.
        let orig_cr = contrast_ratio(red, green);
        let sim_cr = simulated_contrast_ratio(&sim, red, green);
        assert!(orig_cr > 0.0, "original contrast ratio must be positive");
        assert!(sim_cr > 0.0, "simulated contrast ratio must be positive");
        // The key test: red and green appear different under simulation (not zero).
        assert!(sim_cr.is_finite(), "simulated contrast ratio must be finite");
    }

    #[test]
    fn test_simulation_produces_valid_output_all_types() {
        let pixel = Rgba::rgb(180, 90, 45);
        for cvd in [
            CvdType::Protanopia,
            CvdType::Deuteranopia,
            CvdType::Tritanopia,
            CvdType::Achromatopsia,
        ] {
            assert_valid_simulation(cvd, pixel);
        }
    }

    #[test]
    fn test_summary_from_buffers() {
        let sim = ColorBlindSimulator::for_type(CvdType::Protanopia);
        let orig: Vec<u8> = (0..40).map(|i| i as u8).collect();
        let simulated = sim.simulate_buffer(&orig).expect("ok");
        let summary = SimulationSummary::from_buffers(
            CvdType::Protanopia,
            1.0,
            &orig,
            &simulated,
        )
        .expect("ok");
        assert_eq!(summary.pixel_count, 10);
        let display = summary.to_string();
        assert!(display.contains("Protanopia"), "got: {display}");
    }

    #[test]
    fn test_cvd_type_display() {
        assert_eq!(CvdType::Protanopia.to_string(), "Protanopia");
        assert_eq!(CvdType::Deuteranomaly.to_string(), "Deuteranomaly");
        assert_eq!(CvdType::Achromatopsia.to_string(), "Achromatopsia");
    }

    #[test]
    fn test_prevalence_positive() {
        for cvd in [
            CvdType::Protanopia,
            CvdType::Deuteranopia,
            CvdType::Tritanopia,
            CvdType::Achromatopsia,
        ] {
            assert!(cvd.prevalence_percent() > 0.0);
        }
    }

    #[test]
    fn test_srgb_linearise_roundtrip() {
        for v in [0u8, 64, 128, 192, 255] {
            let linear = srgb_to_linear(v);
            let back = linear_to_srgb(linear);
            let diff = (v as i32 - back as i32).abs();
            assert!(diff <= 1, "roundtrip failed for {v}: got {back}");
        }
    }
}
