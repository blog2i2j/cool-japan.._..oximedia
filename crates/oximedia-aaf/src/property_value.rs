//! AAF property type and value model.
//!
//! Provides `PropertyType`, `PropertyValue`, and `PropertyBag` for
//! representing and storing typed AAF object properties.

#![allow(dead_code)]

use std::collections::HashMap;

/// The type tag of an AAF property.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PropertyType {
    /// Signed 32-bit integer.
    Int32,
    /// Signed 64-bit integer.
    Int64,
    /// 64-bit floating-point number.
    Float64,
    /// Boolean flag.
    Boolean,
    /// UTF-8 string.
    String,
    /// Raw byte blob.
    Bytes,
    /// A UUID / AUID reference.
    Auid,
}

impl PropertyType {
    /// Returns `true` if the type is numeric (integer or float).
    #[must_use]
    pub fn is_numeric(self) -> bool {
        matches!(self, Self::Int32 | Self::Int64 | Self::Float64)
    }

    /// Returns `true` if the type carries text.
    #[must_use]
    pub fn is_textual(self) -> bool {
        matches!(self, Self::String)
    }
}

/// A typed value that can be stored in an AAF property bag.
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyValue {
    /// 32-bit integer value.
    Int32(i32),
    /// 64-bit integer value.
    Int64(i64),
    /// 64-bit float value.
    Float64(f64),
    /// Boolean value.
    Boolean(bool),
    /// String value.
    String(String),
    /// Byte blob value.
    Bytes(Vec<u8>),
    /// UUID string value.
    Auid(String),
}

impl PropertyValue {
    /// Convert to `i64` if possible.
    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int32(v) => Some(i64::from(*v)),
            Self::Int64(v) => Some(*v),
            _ => None,
        }
    }

    /// Convert to `f64` if possible.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float64(v) => Some(*v),
            Self::Int32(v) => Some(f64::from(*v)),
            Self::Int64(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Return a string reference if the value is a `String` variant.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            Self::Auid(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Return the discriminant `PropertyType` for this value.
    #[must_use]
    pub fn property_type(&self) -> PropertyType {
        match self {
            Self::Int32(_) => PropertyType::Int32,
            Self::Int64(_) => PropertyType::Int64,
            Self::Float64(_) => PropertyType::Float64,
            Self::Boolean(_) => PropertyType::Boolean,
            Self::String(_) => PropertyType::String,
            Self::Bytes(_) => PropertyType::Bytes,
            Self::Auid(_) => PropertyType::Auid,
        }
    }
}

/// A keyed collection of `PropertyValue` entries, analogous to an AAF
/// property set on an object.
#[derive(Debug, Clone, Default)]
pub struct PropertyBag {
    entries: HashMap<String, PropertyValue>,
}

impl PropertyBag {
    /// Create an empty `PropertyBag`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or overwrite a property by name.
    pub fn set(&mut self, name: impl Into<String>, value: PropertyValue) {
        self.entries.insert(name.into(), value);
    }

    /// Retrieve a property by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&PropertyValue> {
        self.entries.get(name)
    }

    /// Returns `true` if the bag contains the named property.
    #[must_use]
    pub fn has(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Return a sorted list of all property names.
    #[must_use]
    pub fn keys(&self) -> Vec<&str> {
        let mut k: Vec<&str> = self.entries.keys().map(String::as_str).collect();
        k.sort_unstable();
        k
    }

    /// Return the number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the bag is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- PropertyType tests ---

    #[test]
    fn test_property_type_is_numeric_int32() {
        assert!(PropertyType::Int32.is_numeric());
    }

    #[test]
    fn test_property_type_is_numeric_int64() {
        assert!(PropertyType::Int64.is_numeric());
    }

    #[test]
    fn test_property_type_is_numeric_float64() {
        assert!(PropertyType::Float64.is_numeric());
    }

    #[test]
    fn test_property_type_is_not_numeric_string() {
        assert!(!PropertyType::String.is_numeric());
    }

    #[test]
    fn test_property_type_is_not_numeric_boolean() {
        assert!(!PropertyType::Boolean.is_numeric());
    }

    // --- PropertyValue tests ---

    #[test]
    fn test_value_as_i64_int32() {
        let v = PropertyValue::Int32(42);
        assert_eq!(v.as_i64(), Some(42i64));
    }

    #[test]
    fn test_value_as_i64_int64() {
        let v = PropertyValue::Int64(i64::MAX);
        assert_eq!(v.as_i64(), Some(i64::MAX));
    }

    #[test]
    fn test_value_as_i64_non_numeric() {
        let v = PropertyValue::Boolean(true);
        assert_eq!(v.as_i64(), None);
    }

    #[test]
    fn test_value_as_f64_float() {
        let v = PropertyValue::Float64(3.14);
        assert!((v.as_f64().expect("as_f64 should succeed") - 3.14).abs() < 1e-9);
    }

    #[test]
    fn test_value_as_f64_from_int32() {
        let v = PropertyValue::Int32(7);
        assert!((v.as_f64().expect("as_f64 should succeed") - 7.0).abs() < 1e-9);
    }

    #[test]
    fn test_value_as_str_string() {
        let v = PropertyValue::String("hello".to_string());
        assert_eq!(v.as_str(), Some("hello"));
    }

    #[test]
    fn test_value_as_str_non_string() {
        let v = PropertyValue::Int32(1);
        assert_eq!(v.as_str(), None);
    }

    #[test]
    fn test_value_property_type() {
        assert_eq!(
            PropertyValue::Float64(0.0).property_type(),
            PropertyType::Float64
        );
    }

    // --- PropertyBag tests ---

    #[test]
    fn test_bag_set_and_get() {
        let mut bag = PropertyBag::new();
        bag.set("width", PropertyValue::Int32(1920));
        assert_eq!(bag.get("width"), Some(&PropertyValue::Int32(1920)));
    }

    #[test]
    fn test_bag_has() {
        let mut bag = PropertyBag::new();
        assert!(!bag.has("x"));
        bag.set("x", PropertyValue::Boolean(false));
        assert!(bag.has("x"));
    }

    #[test]
    fn test_bag_keys_sorted() {
        let mut bag = PropertyBag::new();
        bag.set("z_key", PropertyValue::Int32(0));
        bag.set("a_key", PropertyValue::Int32(1));
        bag.set("m_key", PropertyValue::Int32(2));
        let keys = bag.keys();
        assert_eq!(keys, vec!["a_key", "m_key", "z_key"]);
    }

    #[test]
    fn test_bag_len_and_is_empty() {
        let mut bag = PropertyBag::new();
        assert!(bag.is_empty());
        bag.set("k", PropertyValue::Boolean(true));
        assert_eq!(bag.len(), 1);
        assert!(!bag.is_empty());
    }

    #[test]
    fn test_bag_overwrite() {
        let mut bag = PropertyBag::new();
        bag.set("v", PropertyValue::Int32(1));
        bag.set("v", PropertyValue::Int32(99));
        assert_eq!(bag.get("v"), Some(&PropertyValue::Int32(99)));
        assert_eq!(bag.len(), 1);
    }
}
