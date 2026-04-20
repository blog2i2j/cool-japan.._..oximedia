#![allow(dead_code)]
//! Tone curve and transfer function operations for image grading.
//!
//! Provides tools for constructing and applying tone curves used in
//! color grading, HDR tone-mapping, and display calibration:
//!
//! - **Bezier curves** - Smooth spline-based tone curves
//! - **Gamma curves** - Power-law transfer functions
//! - **S-curves** - Contrast enhancement curves
//! - **Log curves** - Logarithmic encoding (Cineon, LogC, etc.)
//! - **LUT-backed curves** - Pre-computed lookup tables for fast evaluation

use std::f64::consts::E;

/// The type of tone curve.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CurveType {
    /// Linear (identity) transfer.
    Linear,
    /// Power-law gamma curve.
    Gamma,
    /// Sigmoid / S-curve for contrast enhancement.
    Sigmoid,
    /// Logarithmic encoding curve.
    Logarithmic,
    /// sRGB transfer function.
    Srgb,
    /// Rec. 709 transfer function.
    Rec709,
    /// PQ (Perceptual Quantizer) for HDR.
    Pq,
    /// HLG (Hybrid Log-Gamma) for HDR.
    Hlg,
    /// Custom spline curve.
    Spline,
}

/// A control point on a tone curve.
#[derive(Clone, Copy, Debug)]
pub struct ControlPoint {
    /// Input value (0.0-1.0).
    pub input: f64,
    /// Output value (0.0-1.0).
    pub output: f64,
}

impl ControlPoint {
    /// Create a new control point.
    pub fn new(input: f64, output: f64) -> Self {
        Self { input, output }
    }
}

/// A tone curve that maps input values to output values.
#[derive(Clone, Debug)]
pub struct ToneCurve {
    /// The type of this curve.
    pub curve_type: CurveType,
    /// Gamma exponent (for gamma curves).
    pub gamma: f64,
    /// Control points for spline curves (sorted by input).
    pub control_points: Vec<ControlPoint>,
    /// Pre-computed LUT for fast evaluation (1024 entries).
    lut: Option<Vec<f64>>,
    /// Sigmoid midpoint parameter.
    pub sigmoid_mid: f64,
    /// Sigmoid contrast parameter.
    pub sigmoid_contrast: f64,
}

/// LUT resolution for pre-computed curves.
const LUT_SIZE: usize = 1024;

impl ToneCurve {
    /// Create a linear (identity) tone curve.
    pub fn linear() -> Self {
        Self {
            curve_type: CurveType::Linear,
            gamma: 1.0,
            control_points: vec![ControlPoint::new(0.0, 0.0), ControlPoint::new(1.0, 1.0)],
            lut: None,
            sigmoid_mid: 0.5,
            sigmoid_contrast: 1.0,
        }
    }

    /// Create a gamma curve with the given exponent.
    pub fn gamma(exponent: f64) -> Self {
        let mut curve = Self {
            curve_type: CurveType::Gamma,
            gamma: exponent,
            control_points: Vec::new(),
            lut: None,
            sigmoid_mid: 0.5,
            sigmoid_contrast: 1.0,
        };
        curve.build_lut();
        curve
    }

    /// Create an sRGB transfer function curve.
    pub fn srgb() -> Self {
        let mut curve = Self {
            curve_type: CurveType::Srgb,
            gamma: 2.4,
            control_points: Vec::new(),
            lut: None,
            sigmoid_mid: 0.5,
            sigmoid_contrast: 1.0,
        };
        curve.build_lut();
        curve
    }

    /// Create a Rec. 709 transfer function curve.
    pub fn rec709() -> Self {
        let mut curve = Self {
            curve_type: CurveType::Rec709,
            gamma: 1.0 / 0.45,
            control_points: Vec::new(),
            lut: None,
            sigmoid_mid: 0.5,
            sigmoid_contrast: 1.0,
        };
        curve.build_lut();
        curve
    }

    /// Create a sigmoid (S-curve) for contrast enhancement.
    pub fn sigmoid(midpoint: f64, contrast: f64) -> Self {
        let mut curve = Self {
            curve_type: CurveType::Sigmoid,
            gamma: 1.0,
            control_points: Vec::new(),
            lut: None,
            sigmoid_mid: midpoint.clamp(0.01, 0.99),
            sigmoid_contrast: contrast.clamp(0.1, 20.0),
        };
        curve.build_lut();
        curve
    }

    /// Create a logarithmic encoding curve.
    pub fn logarithmic() -> Self {
        let mut curve = Self {
            curve_type: CurveType::Logarithmic,
            gamma: 1.0,
            control_points: Vec::new(),
            lut: None,
            sigmoid_mid: 0.5,
            sigmoid_contrast: 1.0,
        };
        curve.build_lut();
        curve
    }

    /// Create a PQ (Perceptual Quantizer) curve for HDR.
    pub fn pq() -> Self {
        let mut curve = Self {
            curve_type: CurveType::Pq,
            gamma: 1.0,
            control_points: Vec::new(),
            lut: None,
            sigmoid_mid: 0.5,
            sigmoid_contrast: 1.0,
        };
        curve.build_lut();
        curve
    }

    /// Create an HLG (Hybrid Log-Gamma) curve for HDR.
    pub fn hlg() -> Self {
        let mut curve = Self {
            curve_type: CurveType::Hlg,
            gamma: 1.0,
            control_points: Vec::new(),
            lut: None,
            sigmoid_mid: 0.5,
            sigmoid_contrast: 1.0,
        };
        curve.build_lut();
        curve
    }

    /// Create a spline-based curve from control points.
    pub fn spline(mut points: Vec<ControlPoint>) -> Self {
        points.sort_by(|a, b| {
            a.input
                .partial_cmp(&b.input)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut curve = Self {
            curve_type: CurveType::Spline,
            gamma: 1.0,
            control_points: points,
            lut: None,
            sigmoid_mid: 0.5,
            sigmoid_contrast: 1.0,
        };
        curve.build_lut();
        curve
    }

    /// Evaluate the curve at an input value without using the LUT.
    #[allow(clippy::cast_precision_loss)]
    pub fn evaluate(&self, input: f64) -> f64 {
        let x = input.clamp(0.0, 1.0);
        match self.curve_type {
            CurveType::Linear => x,
            CurveType::Gamma => x.powf(self.gamma),
            CurveType::Srgb => srgb_oetf(x),
            CurveType::Rec709 => rec709_oetf(x),
            CurveType::Sigmoid => sigmoid_fn(x, self.sigmoid_mid, self.sigmoid_contrast),
            CurveType::Logarithmic => log_encode(x),
            CurveType::Pq => pq_oetf(x),
            CurveType::Hlg => hlg_oetf(x),
            CurveType::Spline => self.evaluate_spline(x),
        }
    }

    /// Evaluate the curve using the pre-computed LUT (fast path).
    #[allow(clippy::cast_precision_loss)]
    pub fn evaluate_lut(&self, input: f64) -> f64 {
        if let Some(ref lut) = self.lut {
            let x = input.clamp(0.0, 1.0);
            let idx_f = x * (LUT_SIZE - 1) as f64;
            let idx_lo = idx_f.floor() as usize;
            let idx_hi = (idx_lo + 1).min(LUT_SIZE - 1);
            let frac = idx_f - idx_lo as f64;
            lut[idx_lo] * (1.0 - frac) + lut[idx_hi] * frac
        } else {
            self.evaluate(input)
        }
    }

    /// Apply the curve to a buffer of f64 values in-place.
    pub fn apply_to_buffer(&self, buffer: &mut [f64]) {
        if self.lut.is_some() {
            for v in buffer.iter_mut() {
                *v = self.evaluate_lut(*v);
            }
        } else {
            for v in buffer.iter_mut() {
                *v = self.evaluate(*v);
            }
        }
    }

    /// Build the internal LUT for fast evaluation.
    #[allow(clippy::cast_precision_loss)]
    fn build_lut(&mut self) {
        let mut lut = Vec::with_capacity(LUT_SIZE);
        for i in 0..LUT_SIZE {
            let x = i as f64 / (LUT_SIZE - 1) as f64;
            lut.push(self.evaluate(x));
        }
        self.lut = Some(lut);
    }

    /// Evaluate a spline curve using linear interpolation between control points.
    fn evaluate_spline(&self, x: f64) -> f64 {
        if self.control_points.is_empty() {
            return x;
        }
        if self.control_points.len() == 1 {
            return self.control_points[0].output;
        }

        // Clamp to range of control points. len >= 2 is guaranteed by the early returns above.
        let last = &self.control_points[self.control_points.len() - 1];
        if x <= self.control_points[0].input {
            return self.control_points[0].output;
        }
        if x >= last.input {
            return last.output;
        }

        // Find the segment
        for window in self.control_points.windows(2) {
            let p0 = &window[0];
            let p1 = &window[1];
            if x >= p0.input && x <= p1.input {
                let range = p1.input - p0.input;
                if range < f64::EPSILON {
                    return p0.output;
                }
                let t = (x - p0.input) / range;
                // Hermite interpolation for smoothness
                let t2 = t * t;
                let t3 = t2 * t;
                let h1 = 2.0 * t3 - 3.0 * t2 + 1.0;
                let h2 = -2.0 * t3 + 3.0 * t2;
                return h1 * p0.output + h2 * p1.output;
            }
        }

        x
    }

    /// Return the inverse of this curve (approximate via LUT).
    #[allow(clippy::cast_precision_loss)]
    pub fn inverse(&self) -> Self {
        let mut inv_points = Vec::with_capacity(LUT_SIZE / 4);
        for i in 0..=(LUT_SIZE / 4) {
            let x = i as f64 / (LUT_SIZE / 4) as f64;
            let y = self.evaluate(x);
            inv_points.push(ControlPoint::new(y, x));
        }
        Self::spline(inv_points)
    }

    /// Check if this is an identity (linear) curve.
    pub fn is_identity(&self) -> bool {
        matches!(self.curve_type, CurveType::Linear)
    }

    /// Get the number of control points.
    pub fn control_point_count(&self) -> usize {
        self.control_points.len()
    }
}

/// sRGB OETF (Opto-Electronic Transfer Function).
fn srgb_oetf(x: f64) -> f64 {
    if x <= 0.003_130_8 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

/// sRGB EOTF (Electro-Optical Transfer Function) - inverse of OETF.
fn srgb_eotf(x: f64) -> f64 {
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

/// Rec. 709 OETF.
fn rec709_oetf(x: f64) -> f64 {
    if x < 0.018 {
        4.5 * x
    } else {
        1.099 * x.powf(0.45) - 0.099
    }
}

/// Sigmoid function for S-curve.
fn sigmoid_fn(x: f64, mid: f64, contrast: f64) -> f64 {
    let t = (x - mid) * contrast;
    1.0 / (1.0 + E.powf(-t))
}

/// Logarithmic encoding (simplified Cineon-style).
fn log_encode(x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        let log_val = (x * 9.0 + 1.0).log10(); // log10(1) = 0, log10(10) = 1
        log_val.clamp(0.0, 1.0)
    }
}

/// PQ (Perceptual Quantizer) OETF - simplified SMPTE ST 2084.
fn pq_oetf(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let m1 = 0.159_301_758_125;
    let m2 = 78.84375;
    let c1 = 0.835_937_5;
    let c2 = 18.851_562_5;
    let c3 = 18.6875;

    let xm1 = x.powf(m1);
    let num = c1 + c2 * xm1;
    let den = 1.0 + c3 * xm1;
    (num / den).powf(m2).clamp(0.0, 1.0)
}

/// HLG (Hybrid Log-Gamma) OETF - simplified ARIB STD-B67.
fn hlg_oetf(x: f64) -> f64 {
    let a = 0.178_832_77;
    let b = 0.284_668_92;
    let c = 0.559_910_73;
    if x <= 1.0 / 12.0 {
        (3.0 * x).sqrt()
    } else {
        a * (12.0 * x - b).ln() + c
    }
    .clamp(0.0, 1.0)
}

/// Apply a tone curve to convert linear values to display-referred.
pub fn linearize(values: &mut [f64], curve_type: CurveType) {
    match curve_type {
        CurveType::Srgb => {
            for v in values.iter_mut() {
                *v = srgb_eotf(*v);
            }
        }
        CurveType::Gamma => {
            for v in values.iter_mut() {
                *v = v.powf(2.2);
            }
        }
        _ => {
            // For other curves, use the inverse approach
            let curve = match curve_type {
                CurveType::Linear => ToneCurve::linear(),
                CurveType::Rec709 => ToneCurve::rec709(),
                CurveType::Pq => ToneCurve::pq(),
                CurveType::Hlg => ToneCurve::hlg(),
                _ => ToneCurve::linear(),
            };
            let inv = curve.inverse();
            for v in values.iter_mut() {
                *v = inv.evaluate_lut(*v);
            }
        }
    }
}

/// Compute the average slope of a tone curve over a range.
#[allow(clippy::cast_precision_loss)]
pub fn average_slope(curve: &ToneCurve, start: f64, end: f64, steps: usize) -> f64 {
    if steps < 2 || (end - start).abs() < f64::EPSILON {
        return 0.0;
    }
    let step = (end - start) / steps as f64;
    let mut total_slope = 0.0;
    let mut prev_y = curve.evaluate(start);
    for i in 1..=steps {
        let x = start + step * i as f64;
        let y = curve.evaluate(x);
        total_slope += (y - prev_y) / step;
        prev_y = y;
    }
    total_slope / steps as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_curve() {
        let curve = ToneCurve::linear();
        assert!(curve.is_identity());
        assert!((curve.evaluate(0.0) - 0.0).abs() < f64::EPSILON);
        assert!((curve.evaluate(0.5) - 0.5).abs() < f64::EPSILON);
        assert!((curve.evaluate(1.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gamma_curve() {
        let curve = ToneCurve::gamma(2.2);
        assert!(!curve.is_identity());
        // Gamma 2.2: 0.5^2.2 ~ 0.2176
        let val = curve.evaluate(0.5);
        assert!((val - 0.5_f64.powf(2.2)).abs() < 0.001);
        // Endpoints
        assert!((curve.evaluate(0.0) - 0.0).abs() < f64::EPSILON);
        assert!((curve.evaluate(1.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_srgb_curve() {
        let curve = ToneCurve::srgb();
        // sRGB: 0 maps to 0, 1 maps to 1
        assert!((curve.evaluate(0.0) - 0.0).abs() < 0.001);
        assert!((curve.evaluate(1.0) - 1.0).abs() < 0.001);
        // Mid should be lifted (sRGB brightens mid-tones)
        assert!(curve.evaluate(0.5) > 0.5);
    }

    #[test]
    fn test_rec709_curve() {
        let curve = ToneCurve::rec709();
        assert!((curve.evaluate(0.0) - 0.0).abs() < 0.001);
        assert!((curve.evaluate(1.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_sigmoid_curve() {
        let curve = ToneCurve::sigmoid(0.5, 5.0);
        // Sigmoid endpoints approach but don't reach 0/1
        // Mid-point should be ~0.5
        let mid = curve.evaluate(0.5);
        assert!((mid - 0.5).abs() < 0.01, "Sigmoid midpoint: {mid}");
        // Low values should be compressed
        assert!(curve.evaluate(0.1) < 0.3);
        // High values should be expanded
        assert!(curve.evaluate(0.9) > 0.7);
    }

    #[test]
    fn test_log_curve() {
        let curve = ToneCurve::logarithmic();
        assert!((curve.evaluate(0.0) - 0.0).abs() < 0.01);
        assert!((curve.evaluate(1.0) - 1.0).abs() < 0.01);
        // Log should lift shadows
        assert!(curve.evaluate(0.1) > 0.1);
    }

    #[test]
    fn test_pq_curve() {
        let curve = ToneCurve::pq();
        assert!((curve.evaluate(0.0) - 0.0).abs() < 0.01);
        // PQ maps linear to perceptual
        let val = curve.evaluate(0.5);
        assert!(val > 0.0 && val <= 1.0);
    }

    #[test]
    fn test_hlg_curve() {
        let curve = ToneCurve::hlg();
        let val = curve.evaluate(0.5);
        assert!(val > 0.0 && val <= 1.0);
        // HLG should be monotonically increasing
        assert!(curve.evaluate(0.3) < curve.evaluate(0.7));
    }

    #[test]
    fn test_spline_curve() {
        let points = vec![
            ControlPoint::new(0.0, 0.0),
            ControlPoint::new(0.25, 0.15),
            ControlPoint::new(0.5, 0.5),
            ControlPoint::new(0.75, 0.85),
            ControlPoint::new(1.0, 1.0),
        ];
        let curve = ToneCurve::spline(points);
        assert_eq!(curve.control_point_count(), 5);
        assert!((curve.evaluate(0.0) - 0.0).abs() < f64::EPSILON);
        assert!((curve.evaluate(1.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_lut_evaluation() {
        let curve = ToneCurve::gamma(2.2);
        let direct = curve.evaluate(0.5);
        let lut = curve.evaluate_lut(0.5);
        // LUT should closely match direct evaluation
        assert!(
            (direct - lut).abs() < 0.002,
            "LUT error: direct={direct}, lut={lut}"
        );
    }

    #[test]
    fn test_apply_to_buffer() {
        let curve = ToneCurve::gamma(1.0);
        let mut buffer = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        curve.apply_to_buffer(&mut buffer);
        // Gamma 1.0 is identity
        assert!((buffer[2] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_average_slope_linear() {
        let curve = ToneCurve::linear();
        let slope = average_slope(&curve, 0.0, 1.0, 100);
        assert!(
            (slope - 1.0).abs() < 0.01,
            "Linear slope should be ~1.0, got {slope}"
        );
    }

    #[test]
    fn test_curve_monotonicity() {
        // All standard curves should be monotonically increasing
        let curves = vec![
            ToneCurve::linear(),
            ToneCurve::gamma(2.2),
            ToneCurve::srgb(),
            ToneCurve::rec709(),
            ToneCurve::logarithmic(),
        ];
        for curve in &curves {
            let mut prev = curve.evaluate(0.0);
            for i in 1..=100 {
                #[allow(clippy::cast_precision_loss)]
                let x = i as f64 / 100.0;
                let y = curve.evaluate(x);
                assert!(
                    y >= prev - f64::EPSILON,
                    "Curve {:?} not monotonic at x={x}: prev={prev}, y={y}",
                    curve.curve_type
                );
                prev = y;
            }
        }
    }

    #[test]
    fn test_srgb_roundtrip() {
        // sRGB OETF then EOTF should give identity
        let original = 0.5;
        let encoded = srgb_oetf(original);
        let decoded = srgb_eotf(encoded);
        assert!(
            (original - decoded).abs() < 0.001,
            "sRGB roundtrip failed: {original} -> {encoded} -> {decoded}"
        );
    }

    #[test]
    fn test_linearize_srgb() {
        let mut values = vec![0.0, 0.5, 1.0];
        linearize(&mut values, CurveType::Srgb);
        assert!((values[0] - 0.0).abs() < 0.001);
        assert!(values[1] < 0.5); // linearized mid should be darker
        assert!((values[2] - 1.0).abs() < 0.001);
    }
}
