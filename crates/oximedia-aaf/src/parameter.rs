//! AAF parameter definitions
//!
//! Control points, interpolation definitions, and parameter sets for
//! AAF effects parameters (SMPTE ST 377-1 Section 14).

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

/// Interpolation type for parameter curves
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    /// Constant (step) interpolation
    Constant,
    /// Linear interpolation
    Linear,
    /// Bezier curve interpolation
    BSpline,
    /// Logarithmic interpolation
    Log,
    /// Power curve interpolation
    Power,
}

/// A typed value for an AAF parameter
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterValue {
    /// Integer value
    Int(i64),
    /// Floating-point value
    Float(f64),
    /// Rational value (numerator/denominator)
    Rational(i32, i32),
    /// String value
    Str(String),
    /// Boolean value
    Bool(bool),
}

impl ParameterValue {
    /// Convert to f64 if numeric
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Int(v) => Some(*v as f64),
            Self::Float(v) => Some(*v),
            Self::Rational(n, d) => {
                if *d == 0 {
                    None
                } else {
                    Some(*n as f64 / *d as f64)
                }
            }
            _ => None,
        }
    }

    /// Convert to i64 if integer or truncatable
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            Self::Float(v) => Some(*v as i64),
            _ => None,
        }
    }
}

/// A control point on a parameter curve
#[derive(Debug, Clone)]
pub struct ControlPoint {
    /// Time position (in edit units)
    pub time: i64,
    /// Value at this point
    pub value: ParameterValue,
    /// Tangent for Bezier in [edit-unit offsets and value-offsets]
    pub tangent_in: Option<(f64, f64)>,
    pub tangent_out: Option<(f64, f64)>,
}

impl ControlPoint {
    /// Create a new control point with no tangents
    #[must_use]
    pub fn new(time: i64, value: ParameterValue) -> Self {
        Self {
            time,
            value,
            tangent_in: None,
            tangent_out: None,
        }
    }

    /// Set tangent handles
    pub fn with_tangents(mut self, tan_in: (f64, f64), tan_out: (f64, f64)) -> Self {
        self.tangent_in = Some(tan_in);
        self.tangent_out = Some(tan_out);
        self
    }
}

/// An animated parameter defined by control points and interpolation
#[derive(Debug, Clone)]
pub struct VaryingValue {
    pub interpolation: Interpolation,
    pub control_points: Vec<ControlPoint>,
}

impl VaryingValue {
    /// Create a new varying value
    #[must_use]
    pub fn new(interpolation: Interpolation) -> Self {
        Self {
            interpolation,
            control_points: Vec::new(),
        }
    }

    /// Add a control point (kept sorted by time)
    pub fn add_control_point(&mut self, cp: ControlPoint) {
        self.control_points.push(cp);
        self.control_points.sort_by_key(|c| c.time);
    }

    /// Evaluate the parameter at the given time using the interpolation method.
    ///
    /// # Interpolation modes
    ///
    /// * **Constant** — step function: returns the value of the left-hand
    ///   control point unchanged.
    /// * **Linear** — lerp between the two surrounding control points.
    /// * **BSpline** — Catmull-Rom cubic spline.  When the surrounding
    ///   segment has tangent data stored in [`ControlPoint::tangent_out`] /
    ///   [`ControlPoint::tangent_in`] those tangents are used directly (scaled
    ///   by the segment duration); otherwise the tangents are estimated from
    ///   the neighbouring control points (or mirrored at the endpoints).
    /// * **Log** — geometric (exponential) interpolation:
    ///   `v0 * (v1/v0)^frac`.  Falls back to linear if `v0 == 0`.
    /// * **Power** — ease-in quadratic: `v0 + (v1 − v0) * frac²`.
    #[must_use]
    pub fn evaluate(&self, time: i64) -> Option<f64> {
        if self.control_points.is_empty() {
            return None;
        }
        if self.control_points.len() == 1 {
            return self.control_points[0].value.as_f64();
        }
        // Find surrounding control points
        let pos = self.control_points.partition_point(|cp| cp.time <= time);
        if pos == 0 {
            return self.control_points[0].value.as_f64();
        }
        if pos >= self.control_points.len() {
            return self.control_points.last().and_then(|cp| cp.value.as_f64());
        }
        let cp0 = &self.control_points[pos - 1];
        let cp1 = &self.control_points[pos];
        let v0 = cp0.value.as_f64()?;
        let v1 = cp1.value.as_f64()?;
        let t0 = cp0.time as f64;
        let t1 = cp1.time as f64;
        let t = time as f64;
        // Normalised parameter in [0, 1]
        let frac = (t - t0) / (t1 - t0);
        match self.interpolation {
            Interpolation::Constant => Some(v0),
            Interpolation::Linear => Some(v0 + frac * (v1 - v0)),
            Interpolation::BSpline => Some(self.eval_catmull_rom(pos, v0, v1, t0, t1, t)),
            Interpolation::Log => {
                // Geometric interpolation: v0 * (v1/v0)^frac
                // Falls back to linear when v0 == 0 to avoid division by zero.
                if v0.abs() < f64::EPSILON {
                    Some(v0 + frac * (v1 - v0))
                } else {
                    Some(v0 * (v1 / v0).powf(frac))
                }
            }
            Interpolation::Power => {
                // Ease-in quadratic (power-of-2 curve)
                Some(v0 + (v1 - v0) * frac * frac)
            }
        }
    }

    /// Catmull-Rom cubic spline evaluation between control points at `pos-1`
    /// and `pos`.
    ///
    /// Uses tangents stored on the control points when available; otherwise
    /// estimates them from neighbouring points (finite differences).
    fn eval_catmull_rom(&self, pos: usize, v0: f64, v1: f64, t0: f64, t1: f64, t: f64) -> f64 {
        let dt = t1 - t0;
        if dt.abs() < f64::EPSILON {
            return v0;
        }
        let frac = (t - t0) / dt;

        // Estimate tangents from the stored tangent_out/tangent_in, or from
        // finite differences of neighbouring values.
        let m0 = if let Some((_, dy)) = self.control_points[pos - 1].tangent_out {
            // Stored tangent is in (dt_units, dv_units); scale to value/unit
            dy / dt
        } else {
            // Finite difference: average of [p-2..p-1] and [p-1..p]
            if pos >= 2 {
                let vp = self.control_points[pos - 2].value.as_f64().unwrap_or(v0);
                let tp = self.control_points[pos - 2].time as f64;
                let d_left = if (t0 - tp).abs() > f64::EPSILON {
                    (v0 - vp) / (t0 - tp)
                } else {
                    0.0
                };
                let d_right = (v1 - v0) / dt;
                0.5 * (d_left + d_right)
            } else {
                // At the first segment mirror the right derivative
                (v1 - v0) / dt
            }
        };

        let m1 = if let Some((_, dy)) = self.control_points[pos].tangent_in {
            dy / dt
        } else {
            if pos + 1 < self.control_points.len() {
                let vn = self.control_points[pos + 1].value.as_f64().unwrap_or(v1);
                let tn = self.control_points[pos + 1].time as f64;
                let d_right = if (tn - t1).abs() > f64::EPSILON {
                    (vn - v1) / (tn - t1)
                } else {
                    0.0
                };
                let d_left = (v1 - v0) / dt;
                0.5 * (d_left + d_right)
            } else {
                // At the last segment mirror the left derivative
                (v1 - v0) / dt
            }
        };

        // Hermite basis: h00, h10, h01, h11
        let h00 = 2.0 * frac.powi(3) - 3.0 * frac.powi(2) + 1.0;
        let h10 = frac.powi(3) - 2.0 * frac.powi(2) + frac;
        let h01 = -2.0 * frac.powi(3) + 3.0 * frac.powi(2);
        let h11 = frac.powi(3) - frac.powi(2);

        h00 * v0 + h10 * dt * m0 + h01 * v1 + h11 * dt * m1
    }
}

/// A named parameter in a parameter set
#[derive(Debug, Clone)]
pub struct ParameterDef {
    pub name: String,
    pub auid: String,
    pub description: String,
    pub default_value: Option<ParameterValue>,
    pub min_value: Option<ParameterValue>,
    pub max_value: Option<ParameterValue>,
}

impl ParameterDef {
    /// Create a new parameter definition
    #[must_use]
    pub fn new(name: impl Into<String>, auid: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            auid: auid.into(),
            description: String::new(),
            default_value: None,
            min_value: None,
            max_value: None,
        }
    }

    /// Set the description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set the default value
    pub fn with_default(mut self, v: ParameterValue) -> Self {
        self.default_value = Some(v);
        self
    }

    /// Set range
    pub fn with_range(mut self, min: ParameterValue, max: ParameterValue) -> Self {
        self.min_value = Some(min);
        self.max_value = Some(max);
        self
    }
}

/// A collection of parameter definitions forming a parameter set
#[derive(Debug, Clone, Default)]
pub struct ParameterSet {
    params: HashMap<String, ParameterDef>,
}

impl ParameterSet {
    /// Create a new empty parameter set
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a parameter definition
    pub fn register(&mut self, def: ParameterDef) {
        self.params.insert(def.name.clone(), def);
    }

    /// Look up a parameter by name
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&ParameterDef> {
        self.params.get(name)
    }

    /// Number of registered parameters
    #[must_use]
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Whether the set is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// All parameter names
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.params.keys().map(String::as_str).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_value_as_f64() {
        assert!(
            (ParameterValue::Int(42)
                .as_f64()
                .expect("as_f64 should succeed")
                - 42.0)
                .abs()
                < 1e-9
        );
        assert!(
            (ParameterValue::Float(3.14)
                .as_f64()
                .expect("as_f64 should succeed")
                - 3.14)
                .abs()
                < 1e-9
        );
        let r = ParameterValue::Rational(1, 4)
            .as_f64()
            .expect("r should be valid");
        assert!((r - 0.25).abs() < 1e-9);
        assert!(ParameterValue::Str("x".into()).as_f64().is_none());
    }

    #[test]
    fn test_parameter_value_rational_zero_denominator() {
        assert!(ParameterValue::Rational(1, 0).as_f64().is_none());
    }

    #[test]
    fn test_parameter_value_as_i64() {
        assert_eq!(ParameterValue::Int(99).as_i64(), Some(99));
        assert_eq!(ParameterValue::Float(3.7).as_i64(), Some(3));
        assert!(ParameterValue::Bool(true).as_i64().is_none());
    }

    #[test]
    fn test_control_point_new() {
        let cp = ControlPoint::new(100, ParameterValue::Float(0.5));
        assert_eq!(cp.time, 100);
        assert!(cp.tangent_in.is_none());
    }

    #[test]
    fn test_control_point_with_tangents() {
        let cp =
            ControlPoint::new(0, ParameterValue::Float(0.0)).with_tangents((1.0, 0.5), (-1.0, 0.5));
        assert!(cp.tangent_in.is_some());
        assert!(cp.tangent_out.is_some());
    }

    #[test]
    fn test_varying_value_empty() {
        let vv = VaryingValue::new(Interpolation::Linear);
        assert!(vv.evaluate(0).is_none());
    }

    #[test]
    fn test_varying_value_single_point() {
        let mut vv = VaryingValue::new(Interpolation::Linear);
        vv.add_control_point(ControlPoint::new(0, ParameterValue::Float(1.0)));
        assert!((vv.evaluate(100).expect("evaluate should succeed") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_varying_value_linear_interpolation() {
        let mut vv = VaryingValue::new(Interpolation::Linear);
        vv.add_control_point(ControlPoint::new(0, ParameterValue::Float(0.0)));
        vv.add_control_point(ControlPoint::new(100, ParameterValue::Float(1.0)));
        let mid = vv.evaluate(50).expect("mid should be valid");
        assert!((mid - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_varying_value_constant_interpolation() {
        let mut vv = VaryingValue::new(Interpolation::Constant);
        vv.add_control_point(ControlPoint::new(0, ParameterValue::Float(0.0)));
        vv.add_control_point(ControlPoint::new(100, ParameterValue::Float(1.0)));
        let val = vv.evaluate(50).expect("val should be valid");
        // Constant interpolation returns left value
        assert!((val - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_varying_value_before_first_point() {
        let mut vv = VaryingValue::new(Interpolation::Linear);
        vv.add_control_point(ControlPoint::new(50, ParameterValue::Float(2.0)));
        vv.add_control_point(ControlPoint::new(100, ParameterValue::Float(4.0)));
        // time=10 is before first point -> return first value
        let val = vv.evaluate(10).expect("val should be valid");
        assert!((val - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_varying_value_after_last_point() {
        let mut vv = VaryingValue::new(Interpolation::Linear);
        vv.add_control_point(ControlPoint::new(0, ParameterValue::Float(0.0)));
        vv.add_control_point(ControlPoint::new(100, ParameterValue::Float(5.0)));
        let val = vv.evaluate(200).expect("val should be valid");
        assert!((val - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_parameter_def_builder() {
        let def = ParameterDef::new("Opacity", "auid-opacity")
            .with_description("Controls opacity")
            .with_default(ParameterValue::Float(1.0))
            .with_range(ParameterValue::Float(0.0), ParameterValue::Float(1.0));
        assert_eq!(def.name, "Opacity");
        assert!(def.default_value.is_some());
        assert!(def.min_value.is_some());
    }

    #[test]
    fn test_parameter_set_register_and_lookup() {
        let mut ps = ParameterSet::new();
        let def = ParameterDef::new("Gain", "auid-gain");
        ps.register(def);
        assert_eq!(ps.len(), 1);
        assert!(ps.get("Gain").is_some());
        assert!(ps.get("Missing").is_none());
    }

    #[test]
    fn test_parameter_set_is_empty() {
        let ps = ParameterSet::new();
        assert!(ps.is_empty());
    }

    #[test]
    fn test_parameter_set_names() {
        let mut ps = ParameterSet::new();
        ps.register(ParameterDef::new("A", "a1"));
        ps.register(ParameterDef::new("B", "b1"));
        let names = ps.names();
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn test_varying_value_sorted_on_insert() {
        let mut vv = VaryingValue::new(Interpolation::Linear);
        vv.add_control_point(ControlPoint::new(100, ParameterValue::Float(1.0)));
        vv.add_control_point(ControlPoint::new(0, ParameterValue::Float(0.0)));
        // Should be sorted: first point at time=0
        assert_eq!(vv.control_points[0].time, 0);
        assert_eq!(vv.control_points[1].time, 100);
    }

    // ── BSpline (Catmull-Rom) ────────────────────────────────────────────────

    #[test]
    fn test_bspline_at_endpoints_returns_endpoint_values() {
        let mut vv = VaryingValue::new(Interpolation::BSpline);
        vv.add_control_point(ControlPoint::new(0, ParameterValue::Float(0.0)));
        vv.add_control_point(ControlPoint::new(100, ParameterValue::Float(10.0)));
        // Exactly at t0
        let at_start = vv.evaluate(0).expect("at start");
        assert!((at_start - 0.0).abs() < 1e-6, "expected 0 got {at_start}");
        // Just before t1 (last point is returned by the ≥len guard)
        let at_end = vv.evaluate(100).expect("at end");
        assert!((at_end - 10.0).abs() < 1e-6, "expected 10 got {at_end}");
    }

    #[test]
    fn test_bspline_midpoint_between_two_linear_points() {
        // With only two control points the Catmull-Rom tangents are mirrors of
        // (v1-v0)/dt, so the spline degenerates to linear.
        let mut vv = VaryingValue::new(Interpolation::BSpline);
        vv.add_control_point(ControlPoint::new(0, ParameterValue::Float(0.0)));
        vv.add_control_point(ControlPoint::new(100, ParameterValue::Float(100.0)));
        let mid = vv.evaluate(50).expect("midpoint");
        // For a degenerate (2-point) Catmull-Rom the midpoint is exactly linear
        assert!((mid - 50.0).abs() < 1e-4, "expected ~50.0 got {mid}");
    }

    #[test]
    fn test_bspline_three_points_smooth() {
        let mut vv = VaryingValue::new(Interpolation::BSpline);
        vv.add_control_point(ControlPoint::new(0, ParameterValue::Float(0.0)));
        vv.add_control_point(ControlPoint::new(50, ParameterValue::Float(5.0)));
        vv.add_control_point(ControlPoint::new(100, ParameterValue::Float(0.0)));
        // The curve must be at 5.0 exactly at t=50 (control point)
        let at_peak = vv.evaluate(50).expect("peak");
        assert!((at_peak - 5.0).abs() < 1e-4, "expected 5.0 got {at_peak}");
        // Mid of first segment should be > 0 and <= 5
        let q1 = vv.evaluate(25).expect("q1");
        assert!(q1 > 0.0 && q1 <= 5.0, "q1={q1} out of range");
    }

    #[test]
    fn test_bspline_with_stored_tangents() {
        let mut vv = VaryingValue::new(Interpolation::BSpline);
        // tangent_out = (dt=100, dv=0) means flat exit tangent
        let cp0 = ControlPoint::new(0, ParameterValue::Float(0.0))
            .with_tangents((0.0, 0.0), (100.0, 0.0));
        // tangent_in = (dt=100, dv=0) means flat entry tangent
        let cp1 = ControlPoint::new(100, ParameterValue::Float(10.0))
            .with_tangents((100.0, 0.0), (0.0, 0.0));
        vv.add_control_point(cp0);
        vv.add_control_point(cp1);
        // With m0=0, m1=0 the Hermite polynomial is a smooth S-curve from 0→10
        let at_zero = vv.evaluate(0).expect("t=0");
        let at_end = vv.evaluate(100).expect("t=100");
        assert!((at_zero - 0.0).abs() < 1e-4);
        assert!((at_end - 10.0).abs() < 1e-4);
    }

    // ── Log interpolation ────────────────────────────────────────────────────

    #[test]
    fn test_log_interpolation_midpoint() {
        let mut vv = VaryingValue::new(Interpolation::Log);
        vv.add_control_point(ControlPoint::new(0, ParameterValue::Float(1.0)));
        vv.add_control_point(ControlPoint::new(100, ParameterValue::Float(100.0)));
        // Geometric midpoint: sqrt(1 * 100) = 10.0
        let mid = vv.evaluate(50).expect("log mid");
        assert!((mid - 10.0).abs() < 1e-4, "expected 10.0 got {mid}");
    }

    #[test]
    fn test_log_interpolation_v0_zero_falls_back_to_linear() {
        let mut vv = VaryingValue::new(Interpolation::Log);
        vv.add_control_point(ControlPoint::new(0, ParameterValue::Float(0.0)));
        vv.add_control_point(ControlPoint::new(100, ParameterValue::Float(10.0)));
        // v0==0 → linear fallback
        let mid = vv.evaluate(50).expect("fallback mid");
        assert!((mid - 5.0).abs() < 1e-6, "expected 5.0 got {mid}");
    }

    #[test]
    fn test_log_interpolation_at_start_and_end() {
        let mut vv = VaryingValue::new(Interpolation::Log);
        vv.add_control_point(ControlPoint::new(0, ParameterValue::Float(2.0)));
        vv.add_control_point(ControlPoint::new(100, ParameterValue::Float(8.0)));
        // At t=0 (exactly on first point) partition_point returns 1, frac=0
        let at_start = vv.evaluate(0).expect("start");
        assert!((at_start - 2.0).abs() < 1e-6, "expected 2.0 got {at_start}");
    }

    // ── Power interpolation ──────────────────────────────────────────────────

    #[test]
    fn test_power_interpolation_ease_in() {
        let mut vv = VaryingValue::new(Interpolation::Power);
        vv.add_control_point(ControlPoint::new(0, ParameterValue::Float(0.0)));
        vv.add_control_point(ControlPoint::new(100, ParameterValue::Float(100.0)));
        // At t=50, frac=0.5, result = 0 + 100 * 0.25 = 25.0
        let mid = vv.evaluate(50).expect("power mid");
        assert!((mid - 25.0).abs() < 1e-6, "expected 25.0 got {mid}");
    }

    #[test]
    fn test_power_interpolation_at_endpoints() {
        let mut vv = VaryingValue::new(Interpolation::Power);
        vv.add_control_point(ControlPoint::new(0, ParameterValue::Float(5.0)));
        vv.add_control_point(ControlPoint::new(200, ParameterValue::Float(25.0)));
        // At exact start (frac=0) → v0
        let start = vv.evaluate(0).expect("start");
        assert!((start - 5.0).abs() < 1e-6, "expected 5.0 got {start}");
        // At exact end → v1
        let end = vv.evaluate(200).expect("end");
        assert!((end - 25.0).abs() < 1e-6, "expected 25.0 got {end}");
    }

    #[test]
    fn test_power_curve_is_slower_than_linear_at_midpoint() {
        // Ease-in: at the midpoint the power curve lags behind linear
        let mut vv_pow = VaryingValue::new(Interpolation::Power);
        let mut vv_lin = VaryingValue::new(Interpolation::Linear);
        for vv in [&mut vv_pow, &mut vv_lin] {
            vv.add_control_point(ControlPoint::new(0, ParameterValue::Float(0.0)));
            vv.add_control_point(ControlPoint::new(100, ParameterValue::Float(100.0)));
        }
        let pow_mid = vv_pow.evaluate(50).expect("power");
        let lin_mid = vv_lin.evaluate(50).expect("linear");
        assert!(
            pow_mid < lin_mid,
            "power ({pow_mid}) should be < linear ({lin_mid}) at midpoint"
        );
    }
}
