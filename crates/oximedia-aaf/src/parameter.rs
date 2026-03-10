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

    /// Evaluate the parameter at the given time using the interpolation method
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
        match self.interpolation {
            Interpolation::Constant => Some(v0),
            Interpolation::Linear
            | Interpolation::BSpline
            | Interpolation::Log
            | Interpolation::Power => {
                // Linear fallback for all non-constant modes
                let frac = (t - t0) / (t1 - t0);
                Some(v0 + frac * (v1 - v0))
            }
        }
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
}
