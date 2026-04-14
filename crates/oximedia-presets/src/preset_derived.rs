//! Derived preset system — inherit from a base preset with selective param overrides.
//!
//! This module provides a higher-level, value-oriented alternative to the
//! field-level [`crate::preset_inheritance`] registry.  Instead of working
//! with opaque [`InheritableField`] variants you register a
//! [`DerivedPreset`] that carries typed [`PresetParamValue`] overrides and
//! call [`DerivedPresetRegistry::resolve`] to obtain a flat
//! [`ResolvedPreset`] ready for use.
//!
//! # Design constraints
//!
//! * **Single-level only** — a derived preset's `base_name` must refer to a
//!   *base* preset, not another derived preset.  This keeps the resolution
//!   logic O(1) and avoids recursive chains that are hard to reason about.
//! * **Fail-fast** — attempting to resolve a derived preset whose base is
//!   not registered returns an error rather than silently producing a
//!   half-baked result.
//!
//! # Quick start
//!
//! ```rust
//! use oximedia_presets::preset_derived::{
//!     DerivedPreset, DerivedPresetRegistry, PresetParamValue,
//! };
//! use std::collections::HashMap;
//!
//! let mut registry = DerivedPresetRegistry::new();
//!
//! // Register base params
//! let mut base_params = HashMap::new();
//! base_params.insert("video_bitrate".to_string(), PresetParamValue::Integer(5_000_000));
//! base_params.insert("audio_codec".to_string(), PresetParamValue::Text("opus".to_string()));
//! registry.register_base("my-base", "My Base Preset", "A base preset", base_params);
//!
//! // Register a derived preset that bumps the bitrate
//! let mut overrides = HashMap::new();
//! overrides.insert("video_bitrate".to_string(), PresetParamValue::Integer(8_000_000));
//! let derived = DerivedPreset {
//!     name: "my-derived".to_string(),
//!     description: "Derived with higher bitrate".to_string(),
//!     base_name: "my-base".to_string(),
//!     overrides,
//! };
//! registry.register_derived(derived).expect("register should succeed");
//!
//! let resolved = registry.resolve("my-derived").expect("resolve should succeed");
//! assert_eq!(resolved.get_integer("video_bitrate"), Some(8_000_000));
//! assert_eq!(resolved.get_text("audio_codec").as_deref(), Some("opus"));
//! ```

#![allow(dead_code)]

use std::collections::HashMap;
use thiserror::Error;

// ── Errors ──────────────────────────────────────────────────────────────────

/// Errors produced by the derived-preset system.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum DerivedPresetError {
    /// The named base preset is not registered.
    #[error("Base preset not found: {0}")]
    BaseNotFound(String),

    /// Attempted to register a derived preset that references another derived
    /// preset as its base (chained inheritance is not supported).
    #[error("Derived-from-derived inheritance is not supported (base '{0}' is itself derived)")]
    ChainedInheritanceNotSupported(String),

    /// A preset with this name is already registered.
    #[error("Preset already registered: {0}")]
    AlreadyRegistered(String),
}

// ── PresetParamValue ─────────────────────────────────────────────────────────

/// A typed parameter value for a preset parameter.
///
/// All four variants are value types that can be compared for equality and
/// cloned freely.
#[derive(Debug, Clone, PartialEq)]
pub enum PresetParamValue {
    /// Signed integer (bitrates, dimensions, sample rates …).
    Integer(i64),
    /// IEEE-754 double (CRF, quality factor, speed preset …).
    Float(f64),
    /// Free-form text (codec name, container, profile …).
    Text(String),
    /// Boolean flag (two-pass, HDR, fast-decode …).
    Bool(bool),
}

impl PresetParamValue {
    /// Return the integer value if this is the `Integer` variant.
    #[must_use]
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Self::Integer(v) => Some(*v),
            _ => None,
        }
    }

    /// Return the float value if this is the `Float` variant.
    #[must_use]
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Return a reference to the text value if this is the `Text` variant.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Return the boolean value if this is the `Bool` variant.
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

// ── DerivedPreset ────────────────────────────────────────────────────────────

/// A preset that derives its default parameters from a named *base* preset
/// and overrides a selective subset of them.
///
/// Only single-level inheritance is supported: `base_name` must refer to a
/// base preset registered via [`DerivedPresetRegistry::register_base`].
#[derive(Debug, Clone)]
pub struct DerivedPreset {
    /// Name of the base preset to inherit from.
    pub base_name: String,
    /// Parameter values that override the corresponding base values.
    /// Keys not present in `overrides` are inherited verbatim from the base.
    pub overrides: HashMap<String, PresetParamValue>,
    /// Unique name for this derived preset.
    pub name: String,
    /// Human-readable description.
    pub description: String,
}

// ── ResolvedPreset ───────────────────────────────────────────────────────────

/// A fully-resolved (flat) preset produced by merging a base preset's
/// parameters with the derived preset's overrides.
///
/// Override values win over base values; base values that are not overridden
/// are preserved unchanged.
#[derive(Debug, Clone)]
pub struct ResolvedPreset {
    /// Name of the resolved preset.
    pub name: String,
    /// Description of the resolved preset.
    pub description: String,
    /// Name of the base preset this was derived from.
    pub base_name: String,
    /// Effective parameters after applying overrides.
    pub params: HashMap<String, PresetParamValue>,
}

impl ResolvedPreset {
    /// Retrieve a parameter by key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&PresetParamValue> {
        self.params.get(key)
    }

    /// Convenience: get an integer parameter.
    #[must_use]
    pub fn get_integer(&self, key: &str) -> Option<i64> {
        self.params.get(key)?.as_integer()
    }

    /// Convenience: get a float parameter.
    #[must_use]
    pub fn get_float(&self, key: &str) -> Option<f64> {
        self.params.get(key)?.as_float()
    }

    /// Convenience: get a text parameter.
    #[must_use]
    pub fn get_text(&self, key: &str) -> Option<String> {
        self.params.get(key)?.as_text().map(str::to_string)
    }

    /// Convenience: get a boolean parameter.
    #[must_use]
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.params.get(key)?.as_bool()
    }

    /// Number of effective parameters.
    #[must_use]
    pub fn param_count(&self) -> usize {
        self.params.len()
    }
}

// ── Internal base record ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct BaseRecord {
    name: String,
    description: String,
    params: HashMap<String, PresetParamValue>,
}

// ── DerivedPresetRegistry ────────────────────────────────────────────────────

/// Registry for base presets and their single-level derived variants.
///
/// Registration order does not matter — derived presets may be registered
/// before their base if the base is subsequently registered before any
/// [`resolve`] call.
///
/// [`resolve`]: DerivedPresetRegistry::resolve
#[derive(Debug, Default)]
pub struct DerivedPresetRegistry {
    bases: HashMap<String, BaseRecord>,
    derived: HashMap<String, DerivedPreset>,
}

impl DerivedPresetRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a base preset with its complete parameter map.
    ///
    /// Silently replaces any previously-registered base with the same name.
    pub fn register_base(
        &mut self,
        name: &str,
        human_name: &str,
        description: &str,
        params: HashMap<String, PresetParamValue>,
    ) {
        self.bases.insert(
            name.to_string(),
            BaseRecord {
                name: human_name.to_string(),
                description: description.to_string(),
                params,
            },
        );
    }

    /// Register a derived preset.
    ///
    /// # Errors
    ///
    /// * [`DerivedPresetError::ChainedInheritanceNotSupported`] — if
    ///   `derived.base_name` itself names a *derived* preset (depth > 1).
    pub fn register_derived(
        &mut self,
        derived: DerivedPreset,
    ) -> Result<(), DerivedPresetError> {
        // Enforce single-level: base_name must NOT be a derived preset.
        if self.derived.contains_key(&derived.base_name) {
            return Err(DerivedPresetError::ChainedInheritanceNotSupported(
                derived.base_name.clone(),
            ));
        }
        self.derived.insert(derived.name.clone(), derived);
        Ok(())
    }

    /// Resolve a derived preset by name, producing a [`ResolvedPreset`] with
    /// base params merged with overrides (override values win on conflict).
    ///
    /// # Errors
    ///
    /// * [`DerivedPresetError::BaseNotFound`] — if the base preset referenced
    ///   by the derived preset has not been registered.
    pub fn resolve(&self, name: &str) -> Result<ResolvedPreset, DerivedPresetError> {
        // `name` could be either a derived preset or a base preset.
        if let Some(d) = self.derived.get(name) {
            // Locate the base record.
            let base = self
                .bases
                .get(&d.base_name)
                .ok_or_else(|| DerivedPresetError::BaseNotFound(d.base_name.clone()))?;

            // Start from the base's params, then apply overrides.
            let mut params = base.params.clone();
            for (k, v) in &d.overrides {
                params.insert(k.clone(), v.clone());
            }

            return Ok(ResolvedPreset {
                name: d.name.clone(),
                description: d.description.clone(),
                base_name: d.base_name.clone(),
                params,
            });
        }

        // Fall back: maybe the caller is resolving a base preset directly.
        if let Some(base) = self.bases.get(name) {
            return Ok(ResolvedPreset {
                name: base.name.clone(),
                description: base.description.clone(),
                base_name: name.to_string(),
                params: base.params.clone(),
            });
        }

        Err(DerivedPresetError::BaseNotFound(name.to_string()))
    }

    /// Return whether a name is registered as a base preset.
    #[must_use]
    pub fn has_base(&self, name: &str) -> bool {
        self.bases.contains_key(name)
    }

    /// Return whether a name is registered as a derived preset.
    #[must_use]
    pub fn has_derived(&self, name: &str) -> bool {
        self.derived.contains_key(name)
    }

    /// Total number of registered base presets.
    #[must_use]
    pub fn base_count(&self) -> usize {
        self.bases.len()
    }

    /// Total number of registered derived presets.
    #[must_use]
    pub fn derived_count(&self) -> usize {
        self.derived.len()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a base param map for a typical video preset.
    fn base_video_params() -> HashMap<String, PresetParamValue> {
        let mut m = HashMap::new();
        m.insert(
            "video_codec".to_string(),
            PresetParamValue::Text("vp9".to_string()),
        );
        m.insert(
            "audio_codec".to_string(),
            PresetParamValue::Text("opus".to_string()),
        );
        m.insert(
            "video_bitrate".to_string(),
            PresetParamValue::Integer(4_000_000),
        );
        m.insert(
            "audio_bitrate".to_string(),
            PresetParamValue::Integer(128_000),
        );
        m.insert("width".to_string(), PresetParamValue::Integer(1280));
        m.insert("height".to_string(), PresetParamValue::Integer(720));
        m.insert("two_pass".to_string(), PresetParamValue::Bool(false));
        m.insert("crf".to_string(), PresetParamValue::Float(28.0));
        m
    }

    fn make_registry_with_base() -> DerivedPresetRegistry {
        let mut reg = DerivedPresetRegistry::new();
        reg.register_base(
            "base-720p",
            "Base 720p",
            "Base 720p VP9/Opus preset",
            base_video_params(),
        );
        reg
    }

    // ── Test 1: Register and resolve a derived preset ─────────────────────

    #[test]
    fn test_register_and_resolve_derived() {
        let mut reg = make_registry_with_base();

        let mut overrides = HashMap::new();
        overrides.insert("width".to_string(), PresetParamValue::Integer(1920));
        overrides.insert("height".to_string(), PresetParamValue::Integer(1080));
        overrides.insert(
            "video_bitrate".to_string(),
            PresetParamValue::Integer(8_000_000),
        );

        let derived = DerivedPreset {
            name: "derived-1080p".to_string(),
            description: "Derived 1080p from base 720p".to_string(),
            base_name: "base-720p".to_string(),
            overrides,
        };
        reg.register_derived(derived).expect("registration should succeed");

        let resolved = reg
            .resolve("derived-1080p")
            .expect("resolution should succeed");

        assert_eq!(resolved.name, "derived-1080p");
        assert_eq!(resolved.base_name, "base-720p");
        // Overridden values
        assert_eq!(resolved.get_integer("width"), Some(1920));
        assert_eq!(resolved.get_integer("height"), Some(1080));
        assert_eq!(resolved.get_integer("video_bitrate"), Some(8_000_000));
        // Inherited values
        assert_eq!(
            resolved.get_text("video_codec").as_deref(),
            Some("vp9")
        );
        assert_eq!(
            resolved.get_text("audio_codec").as_deref(),
            Some("opus")
        );
        assert_eq!(resolved.get_integer("audio_bitrate"), Some(128_000));
    }

    // ── Test 2: Override single param ────────────────────────────────────

    #[test]
    fn test_override_single_param() {
        let mut reg = make_registry_with_base();

        let mut overrides = HashMap::new();
        overrides.insert(
            "video_bitrate".to_string(),
            PresetParamValue::Integer(6_000_000),
        );

        let derived = DerivedPreset {
            name: "high-bitrate".to_string(),
            description: "Same 720p but higher bitrate".to_string(),
            base_name: "base-720p".to_string(),
            overrides,
        };
        reg.register_derived(derived).expect("registration should succeed");

        let resolved = reg.resolve("high-bitrate").expect("resolve should succeed");

        // Only video_bitrate was overridden
        assert_eq!(resolved.get_integer("video_bitrate"), Some(6_000_000));
        // Everything else is inherited from base
        assert_eq!(resolved.get_integer("width"), Some(1280));
        assert_eq!(resolved.get_integer("height"), Some(720));
        assert_eq!(
            resolved.get_text("video_codec").as_deref(),
            Some("vp9")
        );
    }

    // ── Test 3: Override multiple params ─────────────────────────────────

    #[test]
    fn test_override_multiple_params() {
        let mut reg = make_registry_with_base();

        let mut overrides = HashMap::new();
        overrides.insert(
            "video_bitrate".to_string(),
            PresetParamValue::Integer(12_000_000),
        );
        overrides.insert("width".to_string(), PresetParamValue::Integer(3840));
        overrides.insert("height".to_string(), PresetParamValue::Integer(2160));
        overrides.insert("two_pass".to_string(), PresetParamValue::Bool(true));
        overrides.insert("crf".to_string(), PresetParamValue::Float(24.0));

        let derived = DerivedPreset {
            name: "derived-4k".to_string(),
            description: "4K two-pass variant".to_string(),
            base_name: "base-720p".to_string(),
            overrides,
        };
        reg.register_derived(derived).expect("registration should succeed");

        let resolved = reg.resolve("derived-4k").expect("resolve should succeed");

        assert_eq!(resolved.get_integer("video_bitrate"), Some(12_000_000));
        assert_eq!(resolved.get_integer("width"), Some(3840));
        assert_eq!(resolved.get_integer("height"), Some(2160));
        assert_eq!(resolved.get_bool("two_pass"), Some(true));
        assert_eq!(resolved.get_float("crf"), Some(24.0));
        // Inherited
        assert_eq!(
            resolved.get_text("video_codec").as_deref(),
            Some("vp9")
        );
    }

    // ── Test 4: Unknown base → error ─────────────────────────────────────

    #[test]
    fn test_unknown_base_returns_error() {
        let mut reg = DerivedPresetRegistry::new();
        // No base registered

        let derived = DerivedPreset {
            name: "orphan".to_string(),
            description: "Derived with no registered base".to_string(),
            base_name: "nonexistent-base".to_string(),
            overrides: HashMap::new(),
        };
        reg.register_derived(derived).expect("register step should succeed");

        // Resolution must fail because the base does not exist
        let result = reg.resolve("orphan");
        assert!(
            matches!(result, Err(DerivedPresetError::BaseNotFound(ref n)) if n == "nonexistent-base"),
            "Expected BaseNotFound, got {:?}",
            result
        );
    }

    // ── Test 5: Derived-from-derived → error (depth 1 only) ─────────────

    #[test]
    fn test_derived_from_derived_is_rejected() {
        let mut reg = make_registry_with_base();

        // Register first derived
        let derived1 = DerivedPreset {
            name: "derived-1".to_string(),
            description: "First level derived".to_string(),
            base_name: "base-720p".to_string(),
            overrides: HashMap::new(),
        };
        reg.register_derived(derived1).expect("first derived should succeed");

        // Attempt to register a second derived that references the first derived
        let derived2 = DerivedPreset {
            name: "derived-2".to_string(),
            description: "Second level derived — not allowed".to_string(),
            base_name: "derived-1".to_string(), // references another derived
            overrides: HashMap::new(),
        };
        let result = reg.register_derived(derived2);
        assert!(
            matches!(
                result,
                Err(DerivedPresetError::ChainedInheritanceNotSupported(_))
            ),
            "Expected ChainedInheritanceNotSupported, got {:?}",
            result
        );
    }

    // ── Test 6: Empty overrides → same as base ───────────────────────────

    #[test]
    fn test_empty_overrides_inherits_all_base_params() {
        let mut reg = make_registry_with_base();

        let derived = DerivedPreset {
            name: "clone-of-base".to_string(),
            description: "Identical to base — no overrides".to_string(),
            base_name: "base-720p".to_string(),
            overrides: HashMap::new(), // empty
        };
        reg.register_derived(derived).expect("registration should succeed");

        let base_params = base_video_params();
        let resolved = reg
            .resolve("clone-of-base")
            .expect("resolve should succeed");

        // Every base param must appear with the same value
        assert_eq!(resolved.param_count(), base_params.len());
        for (key, val) in &base_params {
            assert_eq!(
                resolved.get(key),
                Some(val),
                "param '{}' should equal base value",
                key
            );
        }
    }

    // ── Additional: PresetParamValue accessor round-trips ────────────────

    #[test]
    fn test_param_value_accessors() {
        assert_eq!(PresetParamValue::Integer(42).as_integer(), Some(42));
        assert_eq!(PresetParamValue::Float(3.14).as_float(), Some(3.14));
        assert_eq!(
            PresetParamValue::Text("av1".to_string()).as_text(),
            Some("av1")
        );
        assert_eq!(PresetParamValue::Bool(true).as_bool(), Some(true));
        // Wrong variant
        assert_eq!(PresetParamValue::Integer(1).as_float(), None);
        assert_eq!(PresetParamValue::Float(1.0).as_bool(), None);
        assert_eq!(
            PresetParamValue::Text("x".to_string()).as_integer(),
            None
        );
    }

    // ── Additional: has_base / has_derived / counts ───────────────────────

    #[test]
    fn test_registry_membership_queries() {
        let mut reg = make_registry_with_base();
        assert!(reg.has_base("base-720p"));
        assert!(!reg.has_derived("base-720p"));
        assert_eq!(reg.base_count(), 1);
        assert_eq!(reg.derived_count(), 0);

        let derived = DerivedPreset {
            name: "d1".to_string(),
            description: String::new(),
            base_name: "base-720p".to_string(),
            overrides: HashMap::new(),
        };
        reg.register_derived(derived).expect("should succeed");
        assert!(reg.has_derived("d1"));
        assert!(!reg.has_base("d1"));
        assert_eq!(reg.derived_count(), 1);
    }
}
