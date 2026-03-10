#![allow(dead_code)]
//! # Python Configuration Bindings
//!
//! Structured configuration for OxiMedia's Python API surface. Provides
//! named configuration sections, typed values, a fluent builder, and
//! serialisation helpers for round-tripping through Python dicts.

use std::collections::HashMap;

/// Top-level configuration section.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConfigSection {
    /// Codec-related settings (bitrate, CRF, preset …).
    Codec,
    /// Container / muxer settings.
    Container,
    /// Network transport settings.
    Network,
    /// Hardware acceleration settings.
    HardwareAccel,
    /// Logging and diagnostics.
    Logging,
    /// Python module initialisation.
    Module,
}

impl ConfigSection {
    /// Return the canonical section name used as a key prefix.
    pub fn name(&self) -> &'static str {
        match self {
            ConfigSection::Codec => "codec",
            ConfigSection::Container => "container",
            ConfigSection::Network => "network",
            ConfigSection::HardwareAccel => "hw_accel",
            ConfigSection::Logging => "logging",
            ConfigSection::Module => "module",
        }
    }
}

/// A typed configuration value.
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigValue {
    /// String literal.
    Str(String),
    /// 64-bit signed integer.
    Int(i64),
    /// 64-bit float.
    Float(f64),
    /// Boolean flag.
    Bool(bool),
    /// Ordered list of string values.
    List(Vec<String>),
}

impl ConfigValue {
    /// Render the value as a human-readable string.
    pub fn as_str_repr(&self) -> String {
        match self {
            ConfigValue::Str(s) => s.clone(),
            ConfigValue::Int(i) => i.to_string(),
            ConfigValue::Float(f) => format!("{f:.6}"),
            ConfigValue::Bool(b) => b.to_string(),
            ConfigValue::List(l) => l.join(","),
        }
    }

    /// Try to extract the integer value.
    pub fn as_int(&self) -> Option<i64> {
        if let ConfigValue::Int(i) = self {
            Some(*i)
        } else {
            None
        }
    }

    /// Try to extract the bool value.
    pub fn as_bool(&self) -> Option<bool> {
        if let ConfigValue::Bool(b) = self {
            Some(*b)
        } else {
            None
        }
    }

    /// Try to extract the float value.
    pub fn as_float(&self) -> Option<f64> {
        if let ConfigValue::Float(f) = self {
            Some(*f)
        } else {
            None
        }
    }
}

/// Structured configuration container.
#[derive(Debug, Clone, Default)]
pub struct PyConfig {
    /// Nested map: section → (key → value).
    data: HashMap<ConfigSection, HashMap<String, ConfigValue>>,
}

impl PyConfig {
    /// Create an empty configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or overwrite a key within a section.
    pub fn set(&mut self, section: ConfigSection, key: impl Into<String>, value: ConfigValue) {
        self.data
            .entry(section)
            .or_default()
            .insert(key.into(), value);
    }

    /// Retrieve a value from a section.
    pub fn get(&self, section: ConfigSection, key: &str) -> Option<&ConfigValue> {
        self.data.get(&section)?.get(key)
    }

    /// Remove a key from a section.
    pub fn remove(&mut self, section: ConfigSection, key: &str) -> Option<ConfigValue> {
        self.data.get_mut(&section)?.remove(key)
    }

    /// Return `true` if the key exists in the section.
    pub fn contains(&self, section: ConfigSection, key: &str) -> bool {
        self.data
            .get(&section)
            .map_or(false, |m| m.contains_key(key))
    }

    /// Return all keys within a section.
    pub fn keys(&self, section: ConfigSection) -> Vec<&str> {
        self.data
            .get(&section)
            .map(|m| m.keys().map(String::as_str).collect())
            .unwrap_or_default()
    }

    /// Total number of key-value entries across all sections.
    pub fn total_entries(&self) -> usize {
        self.data.values().map(HashMap::len).sum()
    }

    /// Flatten to `HashMap<String, String>` with `"section.key"` keys.
    pub fn to_flat(&self) -> HashMap<String, String> {
        let mut out = HashMap::new();
        for (section, map) in &self.data {
            for (key, val) in map {
                out.insert(format!("{}.{}", section.name(), key), val.as_str_repr());
            }
        }
        out
    }
}

/// Fluent builder for [`PyConfig`].
pub struct ConfigBuilder {
    config: PyConfig,
}

impl ConfigBuilder {
    /// Start building a new configuration.
    pub fn new() -> Self {
        Self {
            config: PyConfig::new(),
        }
    }

    /// Set a string key.
    pub fn str(
        mut self,
        section: ConfigSection,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.config
            .set(section, key, ConfigValue::Str(value.into()));
        self
    }

    /// Set an integer key.
    pub fn int(mut self, section: ConfigSection, key: impl Into<String>, value: i64) -> Self {
        self.config.set(section, key, ConfigValue::Int(value));
        self
    }

    /// Set a float key.
    pub fn float(mut self, section: ConfigSection, key: impl Into<String>, value: f64) -> Self {
        self.config.set(section, key, ConfigValue::Float(value));
        self
    }

    /// Set a boolean key.
    pub fn bool(mut self, section: ConfigSection, key: impl Into<String>, value: bool) -> Self {
        self.config.set(section, key, ConfigValue::Bool(value));
        self
    }

    /// Set a list key.
    pub fn list(
        mut self,
        section: ConfigSection,
        key: impl Into<String>,
        values: Vec<String>,
    ) -> Self {
        self.config.set(section, key, ConfigValue::List(values));
        self
    }

    /// Consume the builder and return the finished [`PyConfig`].
    pub fn build(self) -> PyConfig {
        self.config
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_section_name() {
        assert_eq!(ConfigSection::Codec.name(), "codec");
        assert_eq!(ConfigSection::HardwareAccel.name(), "hw_accel");
    }

    #[test]
    fn test_value_str_repr() {
        assert_eq!(ConfigValue::Str("av1".into()).as_str_repr(), "av1");
        assert_eq!(ConfigValue::Int(42).as_str_repr(), "42");
        assert_eq!(ConfigValue::Bool(false).as_str_repr(), "false");
    }

    #[test]
    fn test_value_list_repr() {
        let v = ConfigValue::List(vec!["a".into(), "b".into()]);
        assert_eq!(v.as_str_repr(), "a,b");
    }

    #[test]
    fn test_value_as_int() {
        assert_eq!(ConfigValue::Int(7).as_int(), Some(7));
        assert_eq!(ConfigValue::Str("7".into()).as_int(), None);
    }

    #[test]
    fn test_value_as_bool() {
        assert_eq!(ConfigValue::Bool(true).as_bool(), Some(true));
        assert_eq!(ConfigValue::Int(1).as_bool(), None);
    }

    #[test]
    fn test_config_set_get() {
        let mut c = PyConfig::new();
        c.set(ConfigSection::Codec, "crf", ConfigValue::Int(28));
        assert_eq!(
            c.get(ConfigSection::Codec, "crf"),
            Some(&ConfigValue::Int(28))
        );
    }

    #[test]
    fn test_config_overwrite() {
        let mut c = PyConfig::new();
        c.set(ConfigSection::Codec, "crf", ConfigValue::Int(28));
        c.set(ConfigSection::Codec, "crf", ConfigValue::Int(32));
        assert_eq!(
            c.get(ConfigSection::Codec, "crf"),
            Some(&ConfigValue::Int(32))
        );
    }

    #[test]
    fn test_config_remove() {
        let mut c = PyConfig::new();
        c.set(
            ConfigSection::Network,
            "host",
            ConfigValue::Str("localhost".into()),
        );
        let removed = c
            .remove(ConfigSection::Network, "host")
            .expect("removed should be valid");
        assert_eq!(removed, ConfigValue::Str("localhost".into()));
        assert!(!c.contains(ConfigSection::Network, "host"));
    }

    #[test]
    fn test_config_contains() {
        let mut c = PyConfig::new();
        c.set(
            ConfigSection::Logging,
            "level",
            ConfigValue::Str("info".into()),
        );
        assert!(c.contains(ConfigSection::Logging, "level"));
        assert!(!c.contains(ConfigSection::Logging, "file"));
    }

    #[test]
    fn test_total_entries() {
        let mut c = PyConfig::new();
        c.set(ConfigSection::Codec, "a", ConfigValue::Int(1));
        c.set(ConfigSection::Network, "b", ConfigValue::Int(2));
        assert_eq!(c.total_entries(), 2);
    }

    #[test]
    fn test_to_flat_keys() {
        let mut c = PyConfig::new();
        c.set(
            ConfigSection::Codec,
            "preset",
            ConfigValue::Str("slow".into()),
        );
        let flat = c.to_flat();
        assert!(flat.contains_key("codec.preset"));
        assert_eq!(flat["codec.preset"], "slow");
    }

    #[test]
    fn test_builder_str() {
        let c = ConfigBuilder::new()
            .str(ConfigSection::Codec, "preset", "medium")
            .build();
        assert_eq!(
            c.get(ConfigSection::Codec, "preset"),
            Some(&ConfigValue::Str("medium".into()))
        );
    }

    #[test]
    fn test_builder_int() {
        let c = ConfigBuilder::new()
            .int(ConfigSection::Codec, "crf", 28)
            .build();
        assert_eq!(
            c.get(ConfigSection::Codec, "crf"),
            Some(&ConfigValue::Int(28))
        );
    }

    #[test]
    fn test_builder_float() {
        let c = ConfigBuilder::new()
            .float(ConfigSection::Codec, "speed", 1.5)
            .build();
        assert!(matches!(
            c.get(ConfigSection::Codec, "speed"),
            Some(ConfigValue::Float(_))
        ));
    }

    #[test]
    fn test_builder_bool() {
        let c = ConfigBuilder::new()
            .bool(ConfigSection::HardwareAccel, "enabled", true)
            .build();
        assert_eq!(
            c.get(ConfigSection::HardwareAccel, "enabled"),
            Some(&ConfigValue::Bool(true))
        );
    }

    #[test]
    fn test_builder_list() {
        let c = ConfigBuilder::new()
            .list(
                ConfigSection::Module,
                "plugins",
                vec!["a".into(), "b".into()],
            )
            .build();
        assert_eq!(
            c.get(ConfigSection::Module, "plugins"),
            Some(&ConfigValue::List(vec!["a".into(), "b".into()]))
        );
    }
}
