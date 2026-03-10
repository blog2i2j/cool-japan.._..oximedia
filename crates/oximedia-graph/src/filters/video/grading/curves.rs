//! Color correction curves: RGB curves, hue vs. sat, hue vs. lum.

use super::types::{HslColor, RgbColor};

/// A point on a curve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurvePoint {
    /// X coordinate (input value)
    pub x: f64,
    /// Y coordinate (output value)
    pub y: f64,
}

impl CurvePoint {
    /// Create a new curve point.
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// Type of curve interpolation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CurveInterpolation {
    /// Linear interpolation.
    Linear,
    /// Cubic spline interpolation.
    CubicSpline,
    /// Bezier curve interpolation.
    Bezier,
}

/// A color correction curve.
#[derive(Clone, Debug, PartialEq)]
pub struct Curve {
    /// Control points defining the curve.
    points: Vec<CurvePoint>,
    /// Interpolation method.
    interpolation: CurveInterpolation,
    /// Cached lookup table for performance.
    lut: Vec<f64>,
}

impl Curve {
    /// Create a new linear (identity) curve.
    #[must_use]
    pub fn linear() -> Self {
        Self {
            points: vec![CurvePoint::new(0.0, 0.0), CurvePoint::new(1.0, 1.0)],
            interpolation: CurveInterpolation::Linear,
            lut: Vec::new(),
        }
    }

    /// Create a new curve with specified points.
    #[must_use]
    pub fn with_points(mut points: Vec<CurvePoint>) -> Self {
        // Sort points by x coordinate
        points.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));

        let mut curve = Self {
            points,
            interpolation: CurveInterpolation::CubicSpline,
            lut: Vec::new(),
        };

        curve.rebuild_lut();
        curve
    }

    /// Set interpolation method.
    #[must_use]
    pub fn with_interpolation(mut self, interpolation: CurveInterpolation) -> Self {
        self.interpolation = interpolation;
        self.rebuild_lut();
        self
    }

    /// Add a control point to the curve.
    pub fn add_point(&mut self, x: f64, y: f64) {
        let point = CurvePoint::new(x.clamp(0.0, 1.0), y.clamp(0.0, 1.0));
        self.points.push(point);
        self.points
            .sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));
        self.rebuild_lut();
    }

    /// Remove a control point.
    pub fn remove_point(&mut self, index: usize) {
        if index < self.points.len() && self.points.len() > 2 {
            self.points.remove(index);
            self.rebuild_lut();
        }
    }

    /// Move a control point.
    pub fn move_point(&mut self, index: usize, x: f64, y: f64) {
        if index < self.points.len() {
            self.points[index] = CurvePoint::new(x.clamp(0.0, 1.0), y.clamp(0.0, 1.0));
            self.points
                .sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));
            self.rebuild_lut();
        }
    }

    /// Evaluate the curve at a given x position.
    #[must_use]
    pub fn evaluate(&self, x: f64) -> f64 {
        if !self.lut.is_empty() {
            // Use cached LUT
            let index = (x * (self.lut.len() - 1) as f64).clamp(0.0, (self.lut.len() - 1) as f64);
            let i0 = index as usize;
            let i1 = (i0 + 1).min(self.lut.len() - 1);
            let frac = index - i0 as f64;
            return self.lut[i0] + (self.lut[i1] - self.lut[i0]) * frac;
        }

        // Fallback to direct evaluation
        self.evaluate_direct(x)
    }

    /// Evaluate the curve directly without using LUT.
    fn evaluate_direct(&self, x: f64) -> f64 {
        let x = x.clamp(0.0, 1.0);

        match self.interpolation {
            CurveInterpolation::Linear => self.evaluate_linear(x),
            CurveInterpolation::CubicSpline => self.evaluate_cubic_spline(x),
            CurveInterpolation::Bezier => self.evaluate_bezier(x),
        }
    }

    /// Linear interpolation between points.
    fn evaluate_linear(&self, x: f64) -> f64 {
        if self.points.is_empty() {
            return x;
        }

        if x <= self.points[0].x {
            return self.points[0].y;
        }

        for i in 0..self.points.len() - 1 {
            let p0 = self.points[i];
            let p1 = self.points[i + 1];

            if x >= p0.x && x <= p1.x {
                let t = (x - p0.x) / (p1.x - p0.x);
                return p0.y + (p1.y - p0.y) * t;
            }
        }

        self.points.last().map(|p| p.y).unwrap_or(x)
    }

    /// Cubic spline interpolation using Catmull-Rom splines.
    fn evaluate_cubic_spline(&self, x: f64) -> f64 {
        if self.points.len() < 2 {
            return x;
        }

        if x <= self.points[0].x {
            return self.points[0].y;
        }

        if x >= self.points.last().map(|p| p.x).unwrap_or(1.0) {
            return self.points.last().map(|p| p.y).unwrap_or(x);
        }

        // Find the segment containing x
        for i in 0..self.points.len() - 1 {
            let p1 = self.points[i];
            let p2 = self.points[i + 1];

            if x >= p1.x && x <= p2.x {
                // Get neighboring points for Catmull-Rom
                let p0 = if i > 0 {
                    self.points[i - 1]
                } else {
                    // Extrapolate before first point
                    CurvePoint::new(p1.x - (p2.x - p1.x), p1.y - (p2.y - p1.y))
                };

                let p3 = if i + 2 < self.points.len() {
                    self.points[i + 2]
                } else {
                    // Extrapolate after last point
                    CurvePoint::new(p2.x + (p2.x - p1.x), p2.y + (p2.y - p1.y))
                };

                // Normalize t to [0, 1] within segment
                let t = (x - p1.x) / (p2.x - p1.x);

                // Catmull-Rom interpolation
                let t2 = t * t;
                let t3 = t2 * t;

                let y = 0.5
                    * ((2.0 * p1.y)
                        + (-p0.y + p2.y) * t
                        + (2.0 * p0.y - 5.0 * p1.y + 4.0 * p2.y - p3.y) * t2
                        + (-p0.y + 3.0 * p1.y - 3.0 * p2.y + p3.y) * t3);

                return y.clamp(0.0, 1.0);
            }
        }

        x
    }

    /// Bezier curve interpolation using cubic Bezier.
    fn evaluate_bezier(&self, x: f64) -> f64 {
        if self.points.len() < 2 {
            return x;
        }

        // Use cubic Bezier segments between control points
        for i in 0..self.points.len() - 1 {
            let p0 = self.points[i];
            let p3 = self.points[i + 1];

            if x >= p0.x && x <= p3.x {
                // Compute control points for smooth curve
                let dx = p3.x - p0.x;
                let dy = p3.y - p0.y;

                // Smooth tangent (1/3 rule)
                let p1 = CurvePoint::new(p0.x + dx / 3.0, p0.y + dy / 3.0);
                let p2 = CurvePoint::new(p3.x - dx / 3.0, p3.y - dy / 3.0);

                // Binary search for t parameter
                let mut t_min = 0.0;
                let mut t_max = 1.0;
                let mut t = 0.5;

                for _ in 0..20 {
                    // Cubic Bezier x(t)
                    let x_t = (1.0_f64 - t).powi(3) * p0.x
                        + 3.0 * (1.0_f64 - t).powi(2) * t * p1.x
                        + 3.0 * (1.0_f64 - t) * t.powi(2) * p2.x
                        + t.powi(3) * p3.x;

                    if (x_t - x).abs() < 1e-6 {
                        break;
                    }

                    if x_t < x {
                        t_min = t;
                    } else {
                        t_max = t;
                    }

                    t = (t_min + t_max) / 2.0;
                }

                // Compute y(t)
                let y = (1.0 - t).powi(3) * p0.y
                    + 3.0 * (1.0 - t).powi(2) * t * p1.y
                    + 3.0 * (1.0 - t) * t.powi(2) * p2.y
                    + t.powi(3) * p3.y;

                return y.clamp(0.0, 1.0);
            }
        }

        x
    }

    /// Rebuild the lookup table.
    fn rebuild_lut(&mut self) {
        const LUT_SIZE: usize = 1024;
        self.lut.clear();
        self.lut.reserve(LUT_SIZE);

        for i in 0..LUT_SIZE {
            let x = i as f64 / (LUT_SIZE - 1) as f64;
            self.lut.push(self.evaluate_direct(x));
        }
    }

    /// Create an S-curve for contrast adjustment.
    #[must_use]
    pub fn s_curve(strength: f64) -> Self {
        let mid = 0.5;
        let offset = strength * 0.25;

        Self::with_points(vec![
            CurvePoint::new(0.0, 0.0),
            CurvePoint::new(mid - 0.25, mid - offset),
            CurvePoint::new(mid, mid),
            CurvePoint::new(mid + 0.25, mid + offset),
            CurvePoint::new(1.0, 1.0),
        ])
    }

    /// Create a contrast curve.
    #[must_use]
    pub fn contrast(contrast: f64) -> Self {
        let mid = 0.5;
        let y_low = mid - (mid * contrast);
        let y_high = mid + ((1.0 - mid) * contrast);

        Self::with_points(vec![
            CurvePoint::new(0.0, y_low.max(0.0)),
            CurvePoint::new(mid, mid),
            CurvePoint::new(1.0, y_high.min(1.0)),
        ])
    }

    /// Create a brightness curve.
    #[must_use]
    pub fn brightness(amount: f64) -> Self {
        Self::with_points(vec![
            CurvePoint::new(0.0, amount.max(0.0)),
            CurvePoint::new(1.0, (1.0 + amount).min(1.0)),
        ])
    }

    /// Create an exposure curve (logarithmic).
    #[must_use]
    pub fn exposure(stops: f64) -> Self {
        let factor = 2_f64.powf(stops);
        Self::with_points(vec![
            CurvePoint::new(0.0, 0.0),
            CurvePoint::new(0.5, (0.5 * factor).min(1.0)),
            CurvePoint::new(1.0, (1.0 * factor).min(1.0)),
        ])
    }

    /// Create a highlights recovery curve.
    #[must_use]
    pub fn highlights(amount: f64) -> Self {
        Self::with_points(vec![
            CurvePoint::new(0.0, 0.0),
            CurvePoint::new(0.7, 0.7),
            CurvePoint::new(0.85, 0.85 - amount * 0.15),
            CurvePoint::new(1.0, (1.0 - amount * 0.3).max(0.7)),
        ])
    }

    /// Create a shadows lift curve.
    #[must_use]
    pub fn shadows(amount: f64) -> Self {
        Self::with_points(vec![
            CurvePoint::new(0.0, (amount * 0.3).max(0.0)),
            CurvePoint::new(0.15, 0.15 + amount * 0.15),
            CurvePoint::new(0.3, 0.3),
            CurvePoint::new(1.0, 1.0),
        ])
    }

    /// Create an inverted curve.
    #[must_use]
    pub fn invert() -> Self {
        Self::with_points(vec![CurvePoint::new(0.0, 1.0), CurvePoint::new(1.0, 0.0)])
    }

    /// Create a film-like response curve.
    #[must_use]
    pub fn film_response() -> Self {
        Self::with_points(vec![
            CurvePoint::new(0.0, 0.0),
            CurvePoint::new(0.1, 0.15),
            CurvePoint::new(0.3, 0.35),
            CurvePoint::new(0.5, 0.55),
            CurvePoint::new(0.7, 0.73),
            CurvePoint::new(0.9, 0.88),
            CurvePoint::new(1.0, 0.95),
        ])
    }

    /// Create a cross-process effect curve.
    #[must_use]
    pub fn cross_process(strength: f64) -> Self {
        Self::with_points(vec![
            CurvePoint::new(0.0, 0.05 * strength),
            CurvePoint::new(0.25, 0.2 + 0.1 * strength),
            CurvePoint::new(0.5, 0.5),
            CurvePoint::new(0.75, 0.8 - 0.1 * strength),
            CurvePoint::new(1.0, 0.95 - 0.05 * strength),
        ])
    }

    /// Get all control points.
    #[must_use]
    pub fn points(&self) -> &[CurvePoint] {
        &self.points
    }

    /// Get the interpolation method.
    #[must_use]
    pub const fn interpolation(&self) -> CurveInterpolation {
        self.interpolation
    }
}

/// RGB curves for per-channel color correction.
#[derive(Clone, Debug)]
pub struct RgbCurves {
    /// Red channel curve
    pub red: Curve,
    /// Green channel curve
    pub green: Curve,
    /// Blue channel curve
    pub blue: Curve,
    /// Master curve applied to all channels
    pub master: Curve,
}

impl Default for RgbCurves {
    fn default() -> Self {
        Self {
            red: Curve::linear(),
            green: Curve::linear(),
            blue: Curve::linear(),
            master: Curve::linear(),
        }
    }
}

impl RgbCurves {
    /// Create new RGB curves.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply curves to a color.
    #[must_use]
    pub fn apply(&self, color: RgbColor) -> RgbColor {
        // Apply master curve first
        let r = self.master.evaluate(color.r);
        let g = self.master.evaluate(color.g);
        let b = self.master.evaluate(color.b);

        // Apply per-channel curves
        RgbColor::new(
            self.red.evaluate(r),
            self.green.evaluate(g),
            self.blue.evaluate(b),
        )
    }
}

/// Hue vs. Saturation curve.
#[derive(Clone, Debug)]
pub struct HueVsSatCurve {
    curve: Curve,
}

impl Default for HueVsSatCurve {
    fn default() -> Self {
        Self {
            curve: Curve::linear(),
        }
    }
}

impl HueVsSatCurve {
    /// Create a new hue vs. saturation curve.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply the curve to a color.
    #[must_use]
    pub fn apply(&self, color: RgbColor) -> RgbColor {
        let hsl = color.to_hsl();

        // Map hue (0-360) to curve input (0-1)
        let hue_norm = hsl.h / 360.0;
        let sat_mult = self.curve.evaluate(hue_norm);

        // Apply saturation adjustment
        let new_sat = (hsl.s * sat_mult).clamp(0.0, 1.0);

        HslColor::new(hsl.h, new_sat, hsl.l).to_rgb()
    }
}

/// Hue vs. Luminance curve.
#[derive(Clone, Debug)]
pub struct HueVsLumCurve {
    curve: Curve,
}

impl Default for HueVsLumCurve {
    fn default() -> Self {
        Self {
            curve: Curve::linear(),
        }
    }
}

impl HueVsLumCurve {
    /// Create a new hue vs. luminance curve.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply the curve to a color.
    #[must_use]
    pub fn apply(&self, color: RgbColor) -> RgbColor {
        let hsl = color.to_hsl();

        // Map hue (0-360) to curve input (0-1)
        let hue_norm = hsl.h / 360.0;
        let lum_mult = self.curve.evaluate(hue_norm);

        // Apply luminance adjustment
        let new_lum = (hsl.l * lum_mult).clamp(0.0, 1.0);

        HslColor::new(hsl.h, hsl.s, new_lum).to_rgb()
    }
}
