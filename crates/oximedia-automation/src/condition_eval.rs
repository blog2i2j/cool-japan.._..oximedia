//! Condition evaluation for automation rules.
//!
//! Provides `ConditionType`, `ConditionValue`, `Condition`, and
//! `ConditionSet` for building and evaluating Boolean rules that
//! drive automation decisions.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// ConditionType
// ---------------------------------------------------------------------------

/// How a condition compares two values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConditionType {
    /// Left equals right.
    Equal,
    /// Left does not equal right.
    NotEqual,
    /// Left is strictly less than right.
    LessThan,
    /// Left is less than or equal to right.
    LessOrEqual,
    /// Left is strictly greater than right.
    GreaterThan,
    /// Left is greater than or equal to right.
    GreaterOrEqual,
    /// String/list membership test.
    Contains,
    /// Logical negation of another condition.
    Not,
}

impl ConditionType {
    /// Returns `true` for types that perform a numeric or string comparison
    /// (excludes `Not`).
    #[must_use]
    pub fn is_comparison(&self) -> bool {
        !matches!(self, Self::Not)
    }

    /// Human-readable operator symbol.
    #[must_use]
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::Equal => "==",
            Self::NotEqual => "!=",
            Self::LessThan => "<",
            Self::LessOrEqual => "<=",
            Self::GreaterThan => ">",
            Self::GreaterOrEqual => ">=",
            Self::Contains => "contains",
            Self::Not => "not",
        }
    }
}

// ---------------------------------------------------------------------------
// ConditionValue
// ---------------------------------------------------------------------------

/// A typed value used in condition evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum ConditionValue {
    /// Boolean value.
    Bool(bool),
    /// 64-bit integer.
    Int(i64),
    /// 64-bit float.
    Float(f64),
    /// UTF-8 string.
    Str(String),
    /// Absence of a value.
    Null,
}

impl ConditionValue {
    /// Cast to `f64` for numeric comparisons; returns `None` for non-numeric.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Int(i) => Some(*i as f64),
            Self::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Returns `true` if this value is falsy (false, 0, empty string, null).
    #[must_use]
    pub fn is_falsy(&self) -> bool {
        match self {
            Self::Bool(b) => !b,
            Self::Int(i) => *i == 0,
            Self::Float(f) => *f == 0.0,
            Self::Str(s) => s.is_empty(),
            Self::Null => true,
        }
    }
}

// ---------------------------------------------------------------------------
// Condition
// ---------------------------------------------------------------------------

/// A single predicate comparing a named variable to a constant value.
#[derive(Debug, Clone)]
pub struct Condition {
    /// Name of the variable (looked up at evaluation time).
    pub variable: String,
    /// Type of comparison.
    pub condition_type: ConditionType,
    /// Expected value to compare against.
    pub expected: ConditionValue,
}

impl Condition {
    /// Construct a new condition.
    #[must_use]
    pub fn new(
        variable: impl Into<String>,
        condition_type: ConditionType,
        expected: ConditionValue,
    ) -> Self {
        Self {
            variable: variable.into(),
            condition_type,
            expected,
        }
    }

    /// Evaluate the condition against a resolved actual value.
    ///
    /// Numeric comparisons use `f64` coercion; string equality is exact;
    /// `Contains` checks whether the actual value's string form is a
    /// substring of the expected string (or vice-versa for lists).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn evaluate(&self, actual: &ConditionValue) -> bool {
        match &self.condition_type {
            ConditionType::Equal => actual == &self.expected,
            ConditionType::NotEqual => actual != &self.expected,
            ConditionType::Not => actual.is_falsy(),
            ConditionType::LessThan => match (actual.as_f64(), self.expected.as_f64()) {
                (Some(a), Some(e)) => a < e,
                _ => false,
            },
            ConditionType::LessOrEqual => match (actual.as_f64(), self.expected.as_f64()) {
                (Some(a), Some(e)) => a <= e,
                _ => false,
            },
            ConditionType::GreaterThan => match (actual.as_f64(), self.expected.as_f64()) {
                (Some(a), Some(e)) => a > e,
                _ => false,
            },
            ConditionType::GreaterOrEqual => match (actual.as_f64(), self.expected.as_f64()) {
                (Some(a), Some(e)) => a >= e,
                _ => false,
            },
            ConditionType::Contains => {
                if let (ConditionValue::Str(a), ConditionValue::Str(e)) = (actual, &self.expected) {
                    a.contains(e.as_str())
                } else {
                    false
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ConditionSet
// ---------------------------------------------------------------------------

/// A collection of conditions evaluated together.
#[derive(Debug, Default)]
pub struct ConditionSet {
    conditions: Vec<Condition>,
}

impl ConditionSet {
    /// Create an empty set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a condition.
    pub fn add(&mut self, condition: Condition) {
        self.conditions.push(condition);
    }

    /// Returns `true` if every condition passes when evaluated against the
    /// provided resolver function.
    #[must_use]
    pub fn all_pass<F>(&self, mut resolver: F) -> bool
    where
        F: FnMut(&str) -> ConditionValue,
    {
        self.conditions
            .iter()
            .all(|c| c.evaluate(&resolver(&c.variable)))
    }

    /// Returns `true` if at least one condition passes.
    #[must_use]
    pub fn any_pass<F>(&self, mut resolver: F) -> bool
    where
        F: FnMut(&str) -> ConditionValue,
    {
        self.conditions
            .iter()
            .any(|c| c.evaluate(&resolver(&c.variable)))
    }

    /// Number of conditions in this set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.conditions.len()
    }

    /// Returns `true` if the set has no conditions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.conditions.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn val_int(i: i64) -> ConditionValue {
        ConditionValue::Int(i)
    }
    fn val_str(s: &str) -> ConditionValue {
        ConditionValue::Str(s.to_string())
    }

    #[test]
    fn test_condition_type_is_comparison_true() {
        assert!(ConditionType::Equal.is_comparison());
        assert!(ConditionType::LessThan.is_comparison());
        assert!(ConditionType::Contains.is_comparison());
    }

    #[test]
    fn test_condition_type_is_comparison_false_for_not() {
        assert!(!ConditionType::Not.is_comparison());
    }

    #[test]
    fn test_condition_type_symbols() {
        assert_eq!(ConditionType::Equal.symbol(), "==");
        assert_eq!(ConditionType::GreaterThan.symbol(), ">");
        assert_eq!(ConditionType::Contains.symbol(), "contains");
    }

    #[test]
    fn test_condition_value_as_f64_int() {
        assert!(
            (ConditionValue::Int(5)
                .as_f64()
                .expect("as_f64 should succeed")
                - 5.0)
                .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_condition_value_as_f64_float() {
        let test_value = std::f64::consts::PI;
        assert!(
            (ConditionValue::Float(test_value)
                .as_f64()
                .expect("as_f64 should succeed")
                - test_value)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn test_condition_value_as_f64_str_none() {
        assert!(val_str("hello").as_f64().is_none());
    }

    #[test]
    fn test_condition_value_is_falsy() {
        assert!(ConditionValue::Bool(false).is_falsy());
        assert!(ConditionValue::Int(0).is_falsy());
        assert!(ConditionValue::Str(String::new()).is_falsy());
        assert!(ConditionValue::Null.is_falsy());
        assert!(!ConditionValue::Bool(true).is_falsy());
        assert!(!ConditionValue::Int(1).is_falsy());
    }

    #[test]
    fn test_condition_equal() {
        let c = Condition::new("x", ConditionType::Equal, val_int(42));
        assert!(c.evaluate(&val_int(42)));
        assert!(!c.evaluate(&val_int(43)));
    }

    #[test]
    fn test_condition_less_than() {
        let c = Condition::new("x", ConditionType::LessThan, val_int(10));
        assert!(c.evaluate(&val_int(5)));
        assert!(!c.evaluate(&val_int(10)));
        assert!(!c.evaluate(&val_int(11)));
    }

    #[test]
    fn test_condition_greater_or_equal() {
        let c = Condition::new("x", ConditionType::GreaterOrEqual, val_int(5));
        assert!(c.evaluate(&val_int(5)));
        assert!(c.evaluate(&val_int(6)));
        assert!(!c.evaluate(&val_int(4)));
    }

    #[test]
    fn test_condition_contains() {
        let c = Condition::new("msg", ConditionType::Contains, val_str("error"));
        assert!(c.evaluate(&val_str("critical error occurred")));
        assert!(!c.evaluate(&val_str("all good")));
    }

    #[test]
    fn test_condition_not() {
        let c = Condition::new("flag", ConditionType::Not, ConditionValue::Null);
        assert!(c.evaluate(&ConditionValue::Bool(false)));
        assert!(!c.evaluate(&ConditionValue::Bool(true)));
    }

    #[test]
    fn test_condition_set_all_pass() {
        let mut set = ConditionSet::new();
        set.add(Condition::new("a", ConditionType::Equal, val_int(1)));
        set.add(Condition::new("b", ConditionType::Equal, val_int(2)));
        let result = set.all_pass(|name| match name {
            "a" => val_int(1),
            "b" => val_int(2),
            _ => ConditionValue::Null,
        });
        assert!(result);
    }

    #[test]
    fn test_condition_set_all_pass_fails() {
        let mut set = ConditionSet::new();
        set.add(Condition::new("a", ConditionType::Equal, val_int(1)));
        set.add(Condition::new("b", ConditionType::Equal, val_int(2)));
        let result = set.all_pass(|name| match name {
            "a" => val_int(1),
            _ => val_int(99),
        });
        assert!(!result);
    }

    #[test]
    fn test_condition_set_any_pass() {
        let mut set = ConditionSet::new();
        set.add(Condition::new("a", ConditionType::Equal, val_int(1)));
        set.add(Condition::new("b", ConditionType::Equal, val_int(2)));
        let result = set.any_pass(|name| match name {
            "a" => val_int(99), // fails
            "b" => val_int(2),  // passes
            _ => ConditionValue::Null,
        });
        assert!(result);
    }

    #[test]
    fn test_condition_set_empty_all_pass() {
        let set = ConditionSet::new();
        assert!(set.all_pass(|_| ConditionValue::Null));
    }

    #[test]
    fn test_condition_set_len_and_is_empty() {
        let mut set = ConditionSet::new();
        assert!(set.is_empty());
        set.add(Condition::new("x", ConditionType::Equal, val_int(0)));
        assert_eq!(set.len(), 1);
        assert!(!set.is_empty());
    }
}
