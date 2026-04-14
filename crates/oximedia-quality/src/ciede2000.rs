#![allow(dead_code)]
//! CIEDE2000 color difference metric for color fidelity assessment.
//!
//! Implements the CIE Delta E 2000 (CIEDE2000) color difference formula,
//! which is the most perceptually uniform color difference metric standardized
//! by the International Commission on Illumination (CIE).
//!
//! The metric operates in CIELAB color space and accounts for:
//! - Lightness weighting
//! - Chroma weighting
//! - Hue weighting
//! - Chroma-hue interaction (rotation term)
//!
//! # Example
//!
//! ```
//! use oximedia_quality::ciede2000::{Ciede2000Calculator, Lab};
//!
//! let calc = Ciede2000Calculator::default();
//! let lab1 = Lab { l: 50.0, a: 2.6772, b: -79.7751 };
//! let lab2 = Lab { l: 50.0, a: 0.0, b: -82.7485 };
//! let de = calc.delta_e(&lab1, &lab2);
//! assert!(de > 0.0);
//! ```

use serde::{Deserialize, Serialize};

/// A color in CIELAB color space.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Lab {
    /// Lightness (L*), typically 0..100
    pub l: f64,
    /// Green-red axis (a*)
    pub a: f64,
    /// Blue-yellow axis (b*)
    pub b: f64,
}

impl Lab {
    /// Creates a new Lab color.
    #[must_use]
    pub fn new(l: f64, a: f64, b: f64) -> Self {
        Self { l, a, b }
    }

    /// Converts from sRGB (0..255 per channel) to CIELAB via XYZ.
    /// Uses D65 illuminant reference white.
    #[must_use]
    pub fn from_srgb(r: u8, g: u8, b: u8) -> Self {
        let (x, y, z) = srgb_to_xyz(r, g, b);
        xyz_to_lab(x, y, z)
    }

    /// Computes chroma C* = sqrt(a*^2 + b*^2).
    #[must_use]
    pub fn chroma(&self) -> f64 {
        (self.a * self.a + self.b * self.b).sqrt()
    }

    /// Computes hue angle h in degrees [0, 360).
    #[must_use]
    pub fn hue_degrees(&self) -> f64 {
        let h = self.b.atan2(self.a).to_degrees();
        if h < 0.0 {
            h + 360.0
        } else {
            h
        }
    }
}

/// Parametric weighting factors for CIEDE2000.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Ciede2000Params {
    /// Lightness weight (k_L). Default 1.0; use 2.0 for textiles.
    pub k_l: f64,
    /// Chroma weight (k_C). Default 1.0.
    pub k_c: f64,
    /// Hue weight (k_H). Default 1.0.
    pub k_h: f64,
}

impl Default for Ciede2000Params {
    fn default() -> Self {
        Self {
            k_l: 1.0,
            k_c: 1.0,
            k_h: 1.0,
        }
    }
}

impl Ciede2000Params {
    /// Creates parameters for textile applications (k_L = 2).
    #[must_use]
    pub fn textile() -> Self {
        Self {
            k_l: 2.0,
            k_c: 1.0,
            k_h: 1.0,
        }
    }
}

/// Calculator for CIEDE2000 color differences.
///
/// Computes perceptually uniform color differences between pairs of colors
/// in CIELAB space using the CIEDE2000 formula (CIE Technical Report 142-2001).
#[derive(Debug, Clone)]
pub struct Ciede2000Calculator {
    /// Parametric weighting factors.
    pub params: Ciede2000Params,
}

impl Default for Ciede2000Calculator {
    fn default() -> Self {
        Self {
            params: Ciede2000Params::default(),
        }
    }
}

impl Ciede2000Calculator {
    /// Creates a calculator with custom parameters.
    #[must_use]
    pub fn with_params(params: Ciede2000Params) -> Self {
        Self { params }
    }

    /// Computes CIEDE2000 Delta E between two Lab colors.
    ///
    /// Returns a non-negative value where 0 means identical colors.
    /// Typical thresholds:
    /// - < 1.0: not perceptible by human eye
    /// - 1-2: perceptible through close observation
    /// - 2-10: perceptible at a glance
    /// - 11-49: colors are more similar than opposite
    /// - 100: exact opposite colors
    #[must_use]
    pub fn delta_e(&self, lab1: &Lab, lab2: &Lab) -> f64 {
        let k_l = self.params.k_l;
        let k_c = self.params.k_c;
        let k_h = self.params.k_h;

        // Step 1: Calculate C'_i and h'_i
        let c_star_1 = lab1.chroma();
        let c_star_2 = lab2.chroma();
        let c_star_mean = (c_star_1 + c_star_2) / 2.0;

        let c_star_mean_7 = c_star_mean.powi(7);
        let g = 0.5 * (1.0 - (c_star_mean_7 / (c_star_mean_7 + 6_103_515_625.0_f64)).sqrt());
        // 6103515625 = 25^7

        let a_prime_1 = lab1.a * (1.0 + g);
        let a_prime_2 = lab2.a * (1.0 + g);

        let c_prime_1 = (a_prime_1 * a_prime_1 + lab1.b * lab1.b).sqrt();
        let c_prime_2 = (a_prime_2 * a_prime_2 + lab2.b * lab2.b).sqrt();

        let h_prime_1 = compute_hue_prime(lab1.b, a_prime_1);
        let h_prime_2 = compute_hue_prime(lab2.b, a_prime_2);

        // Step 2: Calculate Delta L', Delta C', Delta H'
        let delta_l_prime = lab2.l - lab1.l;
        let delta_c_prime = c_prime_2 - c_prime_1;

        let delta_h_prime_rad = if c_prime_1 * c_prime_2 == 0.0 {
            0.0
        } else {
            let diff = h_prime_2 - h_prime_1;
            if diff.abs() <= 180.0 {
                diff
            } else if diff > 180.0 {
                diff - 360.0
            } else {
                diff + 360.0
            }
        };

        let delta_big_h_prime =
            2.0 * (c_prime_1 * c_prime_2).sqrt() * (delta_h_prime_rad.to_radians() / 2.0).sin();

        // Step 3: Calculate CIEDE2000
        let l_prime_mean = (lab1.l + lab2.l) / 2.0;
        let c_prime_mean = (c_prime_1 + c_prime_2) / 2.0;

        let h_prime_mean = if c_prime_1 * c_prime_2 == 0.0 {
            h_prime_1 + h_prime_2
        } else if (h_prime_1 - h_prime_2).abs() <= 180.0 {
            (h_prime_1 + h_prime_2) / 2.0
        } else if h_prime_1 + h_prime_2 < 360.0 {
            (h_prime_1 + h_prime_2 + 360.0) / 2.0
        } else {
            (h_prime_1 + h_prime_2 - 360.0) / 2.0
        };

        let t = 1.0 - 0.17 * (h_prime_mean - 30.0).to_radians().cos()
            + 0.24 * (2.0 * h_prime_mean).to_radians().cos()
            + 0.32 * (3.0 * h_prime_mean + 6.0).to_radians().cos()
            - 0.20 * (4.0 * h_prime_mean - 63.0).to_radians().cos();

        let l_prime_mean_50_sq = (l_prime_mean - 50.0).powi(2);
        let s_l = 1.0 + 0.015 * l_prime_mean_50_sq / (20.0 + l_prime_mean_50_sq).sqrt();
        let s_c = 1.0 + 0.045 * c_prime_mean;
        let s_h = 1.0 + 0.015 * c_prime_mean * t;

        let c_prime_mean_7 = c_prime_mean.powi(7);
        let r_c = 2.0 * (c_prime_mean_7 / (c_prime_mean_7 + 6_103_515_625.0_f64)).sqrt();
        let delta_theta = 30.0 * (-((h_prime_mean - 275.0) / 25.0).powi(2)).exp();
        let r_t = -(2.0 * delta_theta).to_radians().sin() * r_c;

        let term_l = delta_l_prime / (k_l * s_l);
        let term_c = delta_c_prime / (k_c * s_c);
        let term_h = delta_big_h_prime / (k_h * s_h);

        (term_l * term_l + term_c * term_c + term_h * term_h + r_t * term_c * term_h)
            .max(0.0)
            .sqrt()
    }

    /// Computes mean CIEDE2000 over a batch of Lab color pairs.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the two slices have different lengths.
    pub fn mean_delta_e(&self, colors1: &[Lab], colors2: &[Lab]) -> Result<f64, Ciede2000Error> {
        if colors1.len() != colors2.len() {
            return Err(Ciede2000Error::LengthMismatch {
                a: colors1.len(),
                b: colors2.len(),
            });
        }
        if colors1.is_empty() {
            return Err(Ciede2000Error::EmptyInput);
        }
        let sum: f64 = colors1
            .iter()
            .zip(colors2.iter())
            .map(|(c1, c2)| self.delta_e(c1, c2))
            .sum();
        Ok(sum / colors1.len() as f64)
    }

    /// Computes CIEDE2000 for each pixel in two frames (Y plane treated as L*,
    /// Cb/Cr treated as a*/b* approximation).
    ///
    /// Returns a per-pixel Delta E map (row-major, width x height).
    ///
    /// # Errors
    ///
    /// Returns `Err` if the frames have different dimensions or insufficient planes.
    pub fn frame_delta_e_map(
        &self,
        ref_frame: &crate::Frame,
        dist_frame: &crate::Frame,
    ) -> Result<Vec<f64>, Ciede2000Error> {
        if ref_frame.width != dist_frame.width || ref_frame.height != dist_frame.height {
            return Err(Ciede2000Error::DimensionMismatch);
        }
        let w = ref_frame.width;
        let h = ref_frame.height;
        let count = w * h;

        // We need at least luma plane; for chroma we use planes 1,2 if available
        let ref_y = &ref_frame.planes[0];
        let dist_y = &dist_frame.planes[0];

        let (ref_a_data, ref_b_data) = extract_chroma_planes(ref_frame, count);
        let (dist_a_data, dist_b_data) = extract_chroma_planes(dist_frame, count);

        let mut map = Vec::with_capacity(count);
        for i in 0..count {
            let y_idx = i.min(ref_y.len().saturating_sub(1));
            let lab1 = Lab {
                l: f64::from(ref_y.get(y_idx).copied().unwrap_or(0)) * 100.0 / 255.0,
                a: ref_a_data[i.min(ref_a_data.len().saturating_sub(1))],
                b: ref_b_data[i.min(ref_b_data.len().saturating_sub(1))],
            };
            let lab2 = Lab {
                l: f64::from(dist_y.get(y_idx).copied().unwrap_or(0)) * 100.0 / 255.0,
                a: dist_a_data[i.min(dist_a_data.len().saturating_sub(1))],
                b: dist_b_data[i.min(dist_b_data.len().saturating_sub(1))],
            };
            map.push(self.delta_e(&lab1, &lab2));
        }
        Ok(map)
    }

    /// Computes the mean frame-level CIEDE2000 between two frames.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the frames have different dimensions.
    pub fn frame_mean_delta_e(
        &self,
        ref_frame: &crate::Frame,
        dist_frame: &crate::Frame,
    ) -> Result<f64, Ciede2000Error> {
        let map = self.frame_delta_e_map(ref_frame, dist_frame)?;
        if map.is_empty() {
            return Err(Ciede2000Error::EmptyInput);
        }
        Ok(map.iter().sum::<f64>() / map.len() as f64)
    }

    /// Classifies the perceptual difference.
    #[must_use]
    pub fn classify(delta_e: f64) -> DeltaEClassification {
        if delta_e < 1.0 {
            DeltaEClassification::NotPerceptible
        } else if delta_e < 2.0 {
            DeltaEClassification::CloseObservation
        } else if delta_e < 3.5 {
            DeltaEClassification::PerceptibleAtGlance
        } else if delta_e < 5.0 {
            DeltaEClassification::Noticeable
        } else {
            DeltaEClassification::SignificantDifference
        }
    }
}

/// Classification of perceptual color difference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeltaEClassification {
    /// Delta E < 1: not perceptible by human eye
    NotPerceptible,
    /// Delta E 1-2: perceptible through close observation
    CloseObservation,
    /// Delta E 2-3.5: perceptible at a glance
    PerceptibleAtGlance,
    /// Delta E 3.5-5: clearly noticeable
    Noticeable,
    /// Delta E >= 5: significant color difference
    SignificantDifference,
}

/// Errors specific to CIEDE2000 computation.
#[derive(Debug, Clone, thiserror::Error)]
pub enum Ciede2000Error {
    /// Two color arrays have different lengths.
    #[error("color arrays have different lengths: {a} vs {b}")]
    LengthMismatch {
        /// Length of first array
        a: usize,
        /// Length of second array
        b: usize,
    },
    /// Input is empty.
    #[error("empty input")]
    EmptyInput,
    /// Frame dimensions do not match.
    #[error("frame dimensions do not match")]
    DimensionMismatch,
}

// ─── Internal helpers ────────────────────────────────────────────────

fn compute_hue_prime(b: f64, a_prime: f64) -> f64 {
    if a_prime.abs() < 1e-14 && b.abs() < 1e-14 {
        0.0
    } else {
        let h = b.atan2(a_prime).to_degrees();
        if h < 0.0 {
            h + 360.0
        } else {
            h
        }
    }
}

/// Converts sRGB (0-255) to XYZ (D65).
fn srgb_to_xyz(r: u8, g: u8, b: u8) -> (f64, f64, f64) {
    let linearize = |c: f64| -> f64 {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };

    let rl = linearize(f64::from(r) / 255.0);
    let gl = linearize(f64::from(g) / 255.0);
    let bl = linearize(f64::from(b) / 255.0);

    // sRGB -> XYZ (D65) matrix
    let x = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl;
    let y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl;
    let z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl;
    (x, y, z)
}

/// Converts XYZ (D65) to CIELAB.
fn xyz_to_lab(x: f64, y: f64, z: f64) -> Lab {
    // D65 reference white
    let xn = 0.95047;
    let yn = 1.00000;
    let zn = 1.08883;

    let f = |t: f64| -> f64 {
        let delta = 6.0 / 29.0;
        if t > delta * delta * delta {
            t.cbrt()
        } else {
            t / (3.0 * delta * delta) + 4.0 / 29.0
        }
    };

    let fx = f(x / xn);
    let fy = f(y / yn);
    let fz = f(z / zn);

    Lab {
        l: 116.0 * fy - 16.0,
        a: 500.0 * (fx - fy),
        b: 200.0 * (fy - fz),
    }
}

/// Extracts chroma-like planes from a frame, returning (a*, b*) arrays.
/// For YUV frames, Cb/Cr are mapped to a rough [-128,127] scale.
/// For non-YUV or single-plane frames, returns zeros.
fn extract_chroma_planes(frame: &crate::Frame, count: usize) -> (Vec<f64>, Vec<f64>) {
    if frame.planes.len() >= 3 {
        let cb = &frame.planes[1];
        let cr = &frame.planes[2];
        let a_data: Vec<f64> = (0..count)
            .map(|i| {
                f64::from(
                    cr.get(i.min(cr.len().saturating_sub(1)))
                        .copied()
                        .unwrap_or(128),
                ) - 128.0
            })
            .collect();
        let b_data: Vec<f64> = (0..count)
            .map(|i| {
                f64::from(
                    cb.get(i.min(cb.len().saturating_sub(1)))
                        .copied()
                        .unwrap_or(128),
                ) - 128.0
            })
            .collect();
        (a_data, b_data)
    } else {
        (vec![0.0; count], vec![0.0; count])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calc() -> Ciede2000Calculator {
        Ciede2000Calculator::default()
    }

    #[test]
    fn test_identical_colors() {
        let lab = Lab::new(50.0, 25.0, -10.0);
        let de = calc().delta_e(&lab, &lab);
        assert!(
            de.abs() < 1e-10,
            "identical colors should have delta_e ~0, got {de}"
        );
    }

    #[test]
    fn test_known_pair_from_sharma_2005() {
        // Test pair 1 from Sharma et al. 2005 "The CIEDE2000 Color-Difference Formula"
        let lab1 = Lab::new(50.0, 2.6772, -79.7751);
        let lab2 = Lab::new(50.0, 0.0, -82.7485);
        let de = calc().delta_e(&lab1, &lab2);
        // Expected: ~2.0425
        assert!((de - 2.0425).abs() < 0.05, "expected ~2.0425, got {de}");
    }

    #[test]
    fn test_symmetric() {
        let lab1 = Lab::new(60.0, 30.0, -20.0);
        let lab2 = Lab::new(70.0, -10.0, 40.0);
        let de_12 = calc().delta_e(&lab1, &lab2);
        let de_21 = calc().delta_e(&lab2, &lab1);
        assert!(
            (de_12 - de_21).abs() < 1e-10,
            "delta_e should be symmetric: {de_12} vs {de_21}"
        );
    }

    #[test]
    fn test_black_vs_white() {
        let black = Lab::new(0.0, 0.0, 0.0);
        let white = Lab::new(100.0, 0.0, 0.0);
        let de = calc().delta_e(&black, &white);
        assert!(
            de > 30.0,
            "black-white difference should be large, got {de}"
        );
    }

    #[test]
    fn test_textile_params() {
        let params = Ciede2000Params::textile();
        assert!((params.k_l - 2.0).abs() < 1e-10);
        let calc_textile = Ciede2000Calculator::with_params(params);
        let lab1 = Lab::new(50.0, 2.6772, -79.7751);
        let lab2 = Lab::new(50.0, 0.0, -82.7485);
        let de_default = calc().delta_e(&lab1, &lab2);
        let de_textile = calc_textile.delta_e(&lab1, &lab2);
        // Textile params increase k_L so lightness differences are weighted less
        // For this pair lightness is the same, so difference should be similar
        assert!(de_textile > 0.0);
        assert!(de_default > 0.0);
    }

    #[test]
    fn test_mean_delta_e() {
        let c = calc();
        let a = vec![Lab::new(50.0, 0.0, 0.0), Lab::new(60.0, 10.0, -10.0)];
        let b = vec![Lab::new(50.0, 0.0, 0.0), Lab::new(65.0, 15.0, -5.0)];
        let mean = c.mean_delta_e(&a, &b).expect("should succeed");
        // First pair identical (0.0), second pair different => mean > 0
        assert!(mean > 0.0);
    }

    #[test]
    fn test_mean_delta_e_length_mismatch() {
        let c = calc();
        let a = vec![Lab::new(50.0, 0.0, 0.0)];
        let b = vec![Lab::new(50.0, 0.0, 0.0), Lab::new(60.0, 0.0, 0.0)];
        let result = c.mean_delta_e(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_mean_delta_e_empty() {
        let result = calc().mean_delta_e(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_classification() {
        assert_eq!(
            Ciede2000Calculator::classify(0.5),
            DeltaEClassification::NotPerceptible
        );
        assert_eq!(
            Ciede2000Calculator::classify(1.5),
            DeltaEClassification::CloseObservation
        );
        assert_eq!(
            Ciede2000Calculator::classify(2.5),
            DeltaEClassification::PerceptibleAtGlance
        );
        assert_eq!(
            Ciede2000Calculator::classify(4.0),
            DeltaEClassification::Noticeable
        );
        assert_eq!(
            Ciede2000Calculator::classify(10.0),
            DeltaEClassification::SignificantDifference
        );
    }

    #[test]
    fn test_srgb_to_lab() {
        // Pure white in sRGB should map to L*=100, a*=0, b*=0 approximately
        let lab = Lab::from_srgb(255, 255, 255);
        assert!(
            (lab.l - 100.0).abs() < 0.5,
            "white L should be ~100, got {}",
            lab.l
        );
        assert!(lab.a.abs() < 1.0, "white a* should be ~0, got {}", lab.a);
        assert!(lab.b.abs() < 1.0, "white b* should be ~0, got {}", lab.b);

        // Pure black
        let lab_black = Lab::from_srgb(0, 0, 0);
        assert!(
            lab_black.l.abs() < 1.0,
            "black L should be ~0, got {}",
            lab_black.l
        );
    }

    #[test]
    fn test_lab_chroma_and_hue() {
        let lab = Lab::new(50.0, 30.0, 40.0);
        let c = lab.chroma();
        assert!((c - 50.0).abs() < 1e-10, "chroma should be 50, got {c}");

        let h = lab.hue_degrees();
        assert!(
            h > 0.0 && h < 90.0,
            "hue should be in first quadrant, got {h}"
        );
    }

    #[test]
    fn test_frame_mean_delta_e_identical() {
        use oximedia_core::PixelFormat;
        let frame = crate::Frame::new(8, 8, PixelFormat::Yuv420p).expect("should create frame");
        let de = calc()
            .frame_mean_delta_e(&frame, &frame)
            .expect("should succeed");
        assert!(
            de.abs() < 1e-10,
            "identical frames should have mean delta_e ~0, got {de}"
        );
    }

    #[test]
    fn test_frame_delta_e_map_dimension_mismatch() {
        use oximedia_core::PixelFormat;
        let f1 = crate::Frame::new(8, 8, PixelFormat::Yuv420p).expect("frame");
        let f2 = crate::Frame::new(16, 16, PixelFormat::Yuv420p).expect("frame");
        let result = calc().frame_delta_e_map(&f1, &f2);
        assert!(result.is_err());
    }
}
