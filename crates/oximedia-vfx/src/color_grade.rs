//! Color grading primitives: color wheels, tone curves, and lift-gamma-gain.
//!
//! Complements [`color_grading`](super::color_grading) with a lightweight,
//! self-contained grading model suitable for real-time preview.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

// ── ColorWheel ────────────────────────────────────────────────────────────────

/// Three-way color wheel: per-channel RGB offsets for shadows, midtones, and
/// highlights.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorWheel {
    /// Additive RGB offset applied to dark areas (lift).
    pub shadows: [f32; 3],
    /// Additive RGB offset applied to mid-grey areas.
    pub midtones: [f32; 3],
    /// Additive RGB offset applied to bright areas (gain).
    pub highlights: [f32; 3],
}

impl ColorWheel {
    /// Identity wheel – all offsets zero.
    #[must_use]
    pub fn reset() -> Self {
        Self {
            shadows: [0.0; 3],
            midtones: [0.0; 3],
            highlights: [0.0; 3],
        }
    }

    /// Returns `true` if any offset is non-zero.
    #[must_use]
    pub fn has_adjustment(&self) -> bool {
        let any_nonzero = |arr: &[f32; 3]| arr.iter().any(|&v| v != 0.0);
        any_nonzero(&self.shadows) || any_nonzero(&self.midtones) || any_nonzero(&self.highlights)
    }
}

// ── ColorCurvePoint ───────────────────────────────────────────────────────────

/// A single (input, output) control point on a tone curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorCurvePoint {
    /// Input value in [0.0, 1.0].
    pub input: f32,
    /// Output value in [0.0, 1.0].
    pub output: f32,
}

impl ColorCurvePoint {
    /// Returns `true` when `input == output` (no change at this point).
    #[must_use]
    pub fn is_identity(&self) -> bool {
        (self.input - self.output).abs() < f32::EPSILON
    }
}

// ── ColorCurve ────────────────────────────────────────────────────────────────

/// A piecewise-linear tone curve defined by ordered control points.
#[derive(Debug, Clone, Default)]
pub struct ColorCurve {
    /// Sorted control points.
    pub points: Vec<ColorCurvePoint>,
}

impl ColorCurve {
    /// Add a control point; maintains ascending-input order.
    pub fn add_point(&mut self, input: f32, output: f32) {
        let pt = ColorCurvePoint { input, output };
        // Insert in sorted order by input
        let pos = self.points.partition_point(|p| p.input < input);
        if pos < self.points.len() && (self.points[pos].input - input).abs() < f32::EPSILON {
            self.points[pos] = pt; // replace existing
        } else {
            self.points.insert(pos, pt);
        }
    }

    /// Linearly interpolate the curve at `x`.
    ///
    /// Returns `x` unchanged when the curve is empty.
    #[must_use]
    pub fn evaluate(&self, x: f32) -> f32 {
        if self.points.is_empty() {
            return x;
        }
        if self.points.len() == 1 {
            return self.points[0].output;
        }
        // Find the segment
        let x = x.clamp(0.0, 1.0);
        if x <= self.points[0].input {
            return self.points[0].output;
        }
        let Some(last) = self.points.last() else {
            return x;
        };
        if x >= last.input {
            return last.output;
        }
        let idx = self.points.partition_point(|p| p.input < x);
        let p0 = &self.points[idx - 1];
        let p1 = &self.points[idx];
        let t = (x - p0.input) / (p1.input - p0.input);
        p0.output + (p1.output - p0.output) * t
    }

    /// Returns `true` when every control point is an identity point.
    #[must_use]
    pub fn is_linear(&self) -> bool {
        self.points.iter().all(ColorCurvePoint::is_identity)
    }
}

// ── ColorGradeOp ──────────────────────────────────────────────────────────────

/// A set of standard grading parameters applied per-channel.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorGradeOp {
    /// Lift (black-point shift). Default 0.0.
    pub lift: f32,
    /// Gamma (mid-tone power). Default 1.0.
    pub gamma: f32,
    /// Gain (white-point scale). Default 1.0.
    pub gain: f32,
    /// Contrast around 0.5. Default 1.0.
    pub contrast: f32,
    /// Saturation multiplier. Default 1.0.
    pub saturation: f32,
    /// Colour temperature shift in Kelvin (positive = warm). Default 0.0.
    pub temperature: f32,
}

impl Default for ColorGradeOp {
    fn default() -> Self {
        Self {
            lift: 0.0,
            gamma: 1.0,
            gain: 1.0,
            contrast: 1.0,
            saturation: 1.0,
            temperature: 0.0,
        }
    }
}

impl ColorGradeOp {
    /// Returns `true` when the op has no effect.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.lift == 0.0
            && (self.gamma - 1.0).abs() < f32::EPSILON
            && (self.gain - 1.0).abs() < f32::EPSILON
            && (self.contrast - 1.0).abs() < f32::EPSILON
            && (self.saturation - 1.0).abs() < f32::EPSILON
            && self.temperature == 0.0
    }

    /// Apply lift-gamma-gain to a linear [0.0, 1.0] value.
    ///
    /// Formula: `gain * (linear + lift) ^ (1 / gamma)`.
    #[must_use]
    pub fn apply_lgg(&self, linear: f32) -> f32 {
        let lifted = (linear + self.lift).clamp(0.0, 1.0);
        let gamma_inv = if self.gamma.abs() > f32::EPSILON {
            1.0 / self.gamma
        } else {
            1.0
        };
        let gamma_corrected = lifted.powf(gamma_inv);
        (self.gain * gamma_corrected).clamp(0.0, 1.0)
    }
}

// ── apply_color_grade ─────────────────────────────────────────────────────────

/// Apply `op` to every RGB pixel in `pixels` (packed R,G,B triples) and write
/// results into `dst`.  Both slices must have the same length (multiple of 3).
pub fn apply_color_grade(pixels: &[u8], dst: &mut [u8], op: &ColorGradeOp) {
    assert_eq!(pixels.len(), dst.len());
    assert_eq!(pixels.len() % 3, 0);

    for (src_chunk, dst_chunk) in pixels.chunks_exact(3).zip(dst.chunks_exact_mut(3)) {
        let r = src_chunk[0] as f32 / 255.0;
        let g = src_chunk[1] as f32 / 255.0;
        let b = src_chunk[2] as f32 / 255.0;

        // Luminance for saturation
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;

        let r_sat = luma + op.saturation * (r - luma);
        let g_sat = luma + op.saturation * (g - luma);
        let b_sat = luma + op.saturation * (b - luma);

        // Contrast around 0.5
        let r_con = 0.5 + op.contrast * (r_sat - 0.5);
        let g_con = 0.5 + op.contrast * (g_sat - 0.5);
        let b_con = 0.5 + op.contrast * (b_sat - 0.5);

        dst_chunk[0] = (op.apply_lgg(r_con) * 255.0).round().clamp(0.0, 255.0) as u8;
        dst_chunk[1] = (op.apply_lgg(g_con) * 255.0).round().clamp(0.0, 255.0) as u8;
        dst_chunk[2] = (op.apply_lgg(b_con) * 255.0).round().clamp(0.0, 255.0) as u8;
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ColorWheel ───────────────────────────────────────────────────────────────

    #[test]
    fn test_reset_has_no_adjustment() {
        let w = ColorWheel::reset();
        assert!(!w.has_adjustment());
    }

    #[test]
    fn test_has_adjustment_shadow() {
        let mut w = ColorWheel::reset();
        w.shadows[0] = 0.1;
        assert!(w.has_adjustment());
    }

    #[test]
    fn test_has_adjustment_midtone() {
        let mut w = ColorWheel::reset();
        w.midtones[2] = -0.05;
        assert!(w.has_adjustment());
    }

    #[test]
    fn test_has_adjustment_highlight() {
        let mut w = ColorWheel::reset();
        w.highlights[1] = 0.2;
        assert!(w.has_adjustment());
    }

    // ColorCurvePoint ──────────────────────────────────────────────────────────

    #[test]
    fn test_curve_point_is_identity_true() {
        let p = ColorCurvePoint {
            input: 0.5,
            output: 0.5,
        };
        assert!(p.is_identity());
    }

    #[test]
    fn test_curve_point_is_identity_false() {
        let p = ColorCurvePoint {
            input: 0.5,
            output: 0.7,
        };
        assert!(!p.is_identity());
    }

    // ColorCurve ───────────────────────────────────────────────────────────────

    #[test]
    fn test_curve_empty_evaluate_passthrough() {
        let c = ColorCurve::default();
        assert_eq!(c.evaluate(0.4), 0.4);
    }

    #[test]
    fn test_curve_is_linear_identity_points() {
        let mut c = ColorCurve::default();
        c.add_point(0.0, 0.0);
        c.add_point(1.0, 1.0);
        assert!(c.is_linear());
    }

    #[test]
    fn test_curve_evaluate_midpoint() {
        let mut c = ColorCurve::default();
        c.add_point(0.0, 0.0);
        c.add_point(1.0, 1.0);
        assert!((c.evaluate(0.5) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_curve_evaluate_s_curve() {
        let mut c = ColorCurve::default();
        c.add_point(0.0, 0.0);
        c.add_point(0.5, 0.6); // lifted mids
        c.add_point(1.0, 1.0);
        // Between 0 and 0.5 the output should be higher than linear
        let v = c.evaluate(0.25);
        assert!(v > 0.25);
    }

    #[test]
    fn test_curve_is_not_linear_with_boost() {
        let mut c = ColorCurve::default();
        c.add_point(0.5, 0.7);
        assert!(!c.is_linear());
    }

    // ColorGradeOp ─────────────────────────────────────────────────────────────

    #[test]
    fn test_grade_op_default_is_identity() {
        let op = ColorGradeOp::default();
        assert!(op.is_identity());
    }

    #[test]
    fn test_grade_op_lgg_identity() {
        let op = ColorGradeOp::default();
        assert!((op.apply_lgg(0.5) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_grade_op_lgg_gain_doubles() {
        let op = ColorGradeOp {
            gain: 2.0,
            ..Default::default()
        };
        // 0.4 * 2.0 = 0.8
        assert!((op.apply_lgg(0.4) - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_grade_op_not_identity_with_lift() {
        let op = ColorGradeOp {
            lift: 0.1,
            ..Default::default()
        };
        assert!(!op.is_identity());
    }

    // apply_color_grade ────────────────────────────────────────────────────────

    #[test]
    fn test_apply_grade_identity_preserves_pixels() {
        let src: Vec<u8> = (0u8..30).collect();
        let mut dst = vec![0u8; 30];
        let op = ColorGradeOp::default();
        apply_color_grade(&src, &mut dst, &op);
        // With identity op, output should be very close to input
        for (s, d) in src.iter().zip(dst.iter()) {
            let diff = (*s as i32 - *d as i32).abs();
            assert!(diff <= 1, "pixel diff too large: {diff}");
        }
    }

    #[test]
    fn test_apply_grade_output_length_matches() {
        let src = vec![100u8; 30];
        let mut dst = vec![0u8; 30];
        apply_color_grade(&src, &mut dst, &ColorGradeOp::default());
        assert_eq!(dst.len(), 30);
    }
}
