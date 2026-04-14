//! Plugin configuration persistence — save and load plugin settings between sessions.
//!
//! This module provides a structured, JSON-backed key-value store for per-plugin
//! configuration.  Each plugin is identified by its name (the same name returned
//! by [`CodecPluginInfo::name`](crate::traits::CodecPluginInfo)).  Settings are
//! stored in a [`PluginConfig`] map and can be flushed to disk as a single JSON
//! file, or loaded back on startup.
//!
//! # Design
//!
//! - Values are typed via [`ConfigValue`] (bool, int, float, string, list, map).
//! - The on-disk format is a single JSON object:
//!   ```json
//!   {
//!     "plugin-name": {
//!       "key1": { "type": "string", "value": "hello" },
//!       "key2": { "type": "int",    "value": 42 }
//!     }
//!   }
//!   ```
//! - [`PluginConfigStore`] is the main entry point; it manages configs for all
//!   registered plugins and handles serialisation / deserialisation.
//!
//! # Example
//!
//! ```rust
//! use oximedia_plugin::plugin_config::{PluginConfigStore, ConfigValue};
//!
//! let mut store = PluginConfigStore::new();
//! store.set("my-plugin", "quality", ConfigValue::Int(80));
//! store.set("my-plugin", "enable-hdr", ConfigValue::Bool(true));
//!
//! assert_eq!(store.get("my-plugin", "quality"), Some(&ConfigValue::Int(80)));
//!
//! // Persist to a temp file
//! let dir = std::env::temp_dir();
//! let path = dir.join("oximedia_plugin_config_example.json");
//! store.save_to_file(&path).expect("save");
//!
//! let loaded = PluginConfigStore::load_from_file(&path).expect("load");
//! assert_eq!(loaded.get("my-plugin", "quality"), Some(&ConfigValue::Int(80)));
//! ```

use crate::error::PluginResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// ── ConfigValue ───────────────────────────────────────────────────────────────

/// A typed configuration value that can be stored and retrieved for a plugin.
///
/// All variants are serialisable to JSON so that the configuration can be
/// persisted transparently.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "lowercase")]
pub enum ConfigValue {
    /// A boolean flag.
    Bool(bool),
    /// A 64-bit signed integer.
    Int(i64),
    /// A 64-bit floating-point number.
    Float(f64),
    /// A UTF-8 string.
    String(String),
    /// An ordered list of config values.
    List(Vec<ConfigValue>),
    /// A nested key-value map.
    Map(HashMap<String, ConfigValue>),
}

impl ConfigValue {
    /// Try to extract the inner `bool`.
    pub fn as_bool(&self) -> Option<bool> {
        if let ConfigValue::Bool(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Try to extract the inner `i64`.
    pub fn as_int(&self) -> Option<i64> {
        if let ConfigValue::Int(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Try to extract the inner `f64`.
    pub fn as_float(&self) -> Option<f64> {
        if let ConfigValue::Float(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Try to extract the inner `&str`.
    pub fn as_str(&self) -> Option<&str> {
        if let ConfigValue::String(v) = self {
            Some(v.as_str())
        } else {
            None
        }
    }

    /// Try to extract the inner list slice.
    pub fn as_list(&self) -> Option<&[ConfigValue]> {
        if let ConfigValue::List(v) = self {
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Try to extract the inner map reference.
    pub fn as_map(&self) -> Option<&HashMap<String, ConfigValue>> {
        if let ConfigValue::Map(v) = self {
            Some(v)
        } else {
            None
        }
    }
}

impl std::fmt::Display for ConfigValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigValue::Bool(v) => write!(f, "{v}"),
            ConfigValue::Int(v) => write!(f, "{v}"),
            ConfigValue::Float(v) => write!(f, "{v}"),
            ConfigValue::String(v) => write!(f, "{v}"),
            ConfigValue::List(v) => {
                write!(f, "[")?;
                for (i, item) in v.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{item}")?;
                }
                write!(f, "]")
            }
            ConfigValue::Map(v) => {
                write!(f, "{{")?;
                for (i, (k, val)) in v.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{k}: {val}")?;
                }
                write!(f, "}}")
            }
        }
    }
}

// ── PluginConfig ──────────────────────────────────────────────────────────────

/// Configuration namespace for a single plugin: a keyed map of [`ConfigValue`]s.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PluginConfig {
    entries: HashMap<String, ConfigValue>,
}

impl PluginConfig {
    /// Create a new, empty plugin config.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Insert or replace a configuration entry.
    pub fn set(&mut self, key: impl Into<String>, value: ConfigValue) {
        self.entries.insert(key.into(), value);
    }

    /// Retrieve a configuration value by key.
    pub fn get(&self, key: &str) -> Option<&ConfigValue> {
        self.entries.get(key)
    }

    /// Remove a configuration entry.  Returns the removed value, if any.
    pub fn remove(&mut self, key: &str) -> Option<ConfigValue> {
        self.entries.remove(key)
    }

    /// Return `true` if no entries are stored.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return the number of stored entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Iterate over all (key, value) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ConfigValue)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Merge another `PluginConfig` into this one; existing keys are overwritten.
    pub fn merge(&mut self, other: &PluginConfig) {
        for (k, v) in &other.entries {
            self.entries.insert(k.clone(), v.clone());
        }
    }

    /// Return the inner map of raw entries.
    pub fn entries(&self) -> &HashMap<String, ConfigValue> {
        &self.entries
    }
}

// ── PluginConfigStore ─────────────────────────────────────────────────────────

/// Central storage for per-plugin configuration, backed by a JSON file on disk.
///
/// The store maps plugin names to their [`PluginConfig`] namespaces.  Callers
/// can read and write individual keys via [`get`](Self::get) / [`set`](Self::set),
/// and flush the entire store to disk with [`save_to_file`](Self::save_to_file).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PluginConfigStore {
    /// Plugin name → its configuration namespace.
    #[serde(flatten)]
    configs: HashMap<String, PluginConfig>,
}

impl PluginConfigStore {
    /// Create a new, empty store.
    pub fn new() -> Self {
        Self {
            configs: HashMap::new(),
        }
    }

    /// Load a store from a JSON file.
    ///
    /// If the file does not exist, an empty store is returned (not an error).
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::Io`] on read failure, or [`PluginError::Json`] on
    /// parse failure.
    pub fn load_from_file(path: &Path) -> PluginResult<Self> {
        if !path.exists() {
            return Ok(Self::new());
        }
        let data = std::fs::read_to_string(path)?;
        let store: Self = serde_json::from_str(&data)?;
        Ok(store)
    }

    /// Persist the store to a JSON file.
    ///
    /// The parent directory must already exist.  The file is written atomically
    /// by first writing to a sibling `.tmp` file then renaming.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::Io`] or [`PluginError::Json`] on failure.
    pub fn save_to_file(&self, path: &Path) -> PluginResult<()> {
        let json = serde_json::to_string_pretty(self)?;

        // Write to a temp file alongside the target, then rename for atomicity.
        let tmp_path = path.with_extension("json.tmp");
        std::fs::write(&tmp_path, &json)?;
        std::fs::rename(&tmp_path, path)?;
        Ok(())
    }

    /// Set a configuration value for the given plugin and key.
    pub fn set(&mut self, plugin_name: &str, key: &str, value: ConfigValue) {
        self.configs
            .entry(plugin_name.to_string())
            .or_insert_with(PluginConfig::new)
            .set(key, value);
    }

    /// Get a configuration value for the given plugin and key.
    pub fn get(&self, plugin_name: &str, key: &str) -> Option<&ConfigValue> {
        self.configs.get(plugin_name)?.get(key)
    }

    /// Remove a single key for a plugin.  Returns the removed value if present.
    pub fn remove_key(&mut self, plugin_name: &str, key: &str) -> Option<ConfigValue> {
        self.configs.get_mut(plugin_name)?.remove(key)
    }

    /// Remove all configuration entries for a specific plugin.
    pub fn remove_plugin(&mut self, plugin_name: &str) -> Option<PluginConfig> {
        self.configs.remove(plugin_name)
    }

    /// Get the entire [`PluginConfig`] namespace for a plugin (read-only).
    pub fn plugin_config(&self, plugin_name: &str) -> Option<&PluginConfig> {
        self.configs.get(plugin_name)
    }

    /// Get the entire [`PluginConfig`] namespace for a plugin (mutable).
    pub fn plugin_config_mut(&mut self, plugin_name: &str) -> &mut PluginConfig {
        self.configs
            .entry(plugin_name.to_string())
            .or_insert_with(PluginConfig::new)
    }

    /// Return the number of plugins that have configuration stored.
    pub fn plugin_count(&self) -> usize {
        self.configs.len()
    }

    /// Return `true` if no configuration is stored for any plugin.
    pub fn is_empty(&self) -> bool {
        self.configs.is_empty()
    }

    /// Merge another store into this one; values from `other` overwrite existing ones.
    pub fn merge(&mut self, other: &PluginConfigStore) {
        for (name, cfg) in &other.configs {
            self.configs
                .entry(name.clone())
                .or_insert_with(PluginConfig::new)
                .merge(cfg);
        }
    }

    /// List all plugin names that have stored configuration.
    pub fn plugin_names(&self) -> Vec<&str> {
        self.configs.keys().map(|s| s.as_str()).collect()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. ConfigValue typed accessors
    #[test]
    fn test_config_value_accessors() {
        let b = ConfigValue::Bool(true);
        assert_eq!(b.as_bool(), Some(true));
        assert_eq!(b.as_int(), None);

        let i = ConfigValue::Int(42);
        assert_eq!(i.as_int(), Some(42));
        assert_eq!(i.as_bool(), None);

        let f = ConfigValue::Float(3.14);
        assert!((f.as_float().unwrap() - 3.14).abs() < 1e-10);

        let s = ConfigValue::String("hello".to_string());
        assert_eq!(s.as_str(), Some("hello"));

        let l = ConfigValue::List(vec![ConfigValue::Int(1), ConfigValue::Int(2)]);
        assert_eq!(l.as_list().map(|s| s.len()), Some(2));

        let mut inner = HashMap::new();
        inner.insert("k".to_string(), ConfigValue::Bool(false));
        let m = ConfigValue::Map(inner);
        assert_eq!(m.as_map().map(|h| h.len()), Some(1));
    }

    // 2. ConfigValue Display
    #[test]
    fn test_config_value_display() {
        assert_eq!(ConfigValue::Bool(true).to_string(), "true");
        assert_eq!(ConfigValue::Int(-7).to_string(), "-7");
        assert_eq!(ConfigValue::String("abc".to_string()).to_string(), "abc");
        let list = ConfigValue::List(vec![ConfigValue::Int(1), ConfigValue::Int(2)]);
        assert_eq!(list.to_string(), "[1, 2]");
    }

    // 3. PluginConfig basic set/get/remove
    #[test]
    fn test_plugin_config_basic() {
        let mut cfg = PluginConfig::new();
        cfg.set("quality", ConfigValue::Int(85));
        assert_eq!(cfg.get("quality"), Some(&ConfigValue::Int(85)));
        assert_eq!(cfg.len(), 1);

        let removed = cfg.remove("quality");
        assert_eq!(removed, Some(ConfigValue::Int(85)));
        assert!(cfg.is_empty());
    }

    // 4. PluginConfig merge
    #[test]
    fn test_plugin_config_merge() {
        let mut a = PluginConfig::new();
        a.set("key1", ConfigValue::Int(1));
        a.set("key2", ConfigValue::Bool(false));

        let mut b = PluginConfig::new();
        b.set("key2", ConfigValue::Bool(true)); // overwrites
        b.set("key3", ConfigValue::String("new".to_string()));

        a.merge(&b);
        assert_eq!(a.get("key1"), Some(&ConfigValue::Int(1)));
        assert_eq!(a.get("key2"), Some(&ConfigValue::Bool(true)));
        assert_eq!(
            a.get("key3"),
            Some(&ConfigValue::String("new".to_string()))
        );
    }

    // 5. PluginConfigStore set/get
    #[test]
    fn test_store_set_get() {
        let mut store = PluginConfigStore::new();
        store.set("my-plugin", "bitrate", ConfigValue::Int(5000));
        store.set("my-plugin", "preset", ConfigValue::String("fast".to_string()));

        assert_eq!(
            store.get("my-plugin", "bitrate"),
            Some(&ConfigValue::Int(5000))
        );
        assert_eq!(
            store.get("my-plugin", "preset"),
            Some(&ConfigValue::String("fast".to_string()))
        );
        assert_eq!(store.get("other-plugin", "bitrate"), None);
        assert_eq!(store.get("my-plugin", "nonexistent"), None);
    }

    // 6. PluginConfigStore remove operations
    #[test]
    fn test_store_remove() {
        let mut store = PluginConfigStore::new();
        store.set("plug-a", "x", ConfigValue::Bool(true));
        store.set("plug-b", "y", ConfigValue::Int(99));

        let removed = store.remove_key("plug-a", "x");
        assert_eq!(removed, Some(ConfigValue::Bool(true)));
        assert_eq!(store.get("plug-a", "x"), None);

        let removed_plugin = store.remove_plugin("plug-b");
        assert!(removed_plugin.is_some());
        assert_eq!(store.plugin_count(), 1); // plug-a still present (empty)
    }

    // 7. PluginConfigStore save and load round-trip
    #[test]
    fn test_store_save_load_roundtrip() {
        let mut store = PluginConfigStore::new();
        store.set("codec-x", "threads", ConfigValue::Int(4));
        store.set("codec-x", "lossless", ConfigValue::Bool(false));
        store.set("codec-y", "level", ConfigValue::Float(4.2));

        let tmp_dir = std::env::temp_dir();
        let path = tmp_dir.join("oximedia_plugin_config_test.json");

        store.save_to_file(&path).expect("save");
        let loaded = PluginConfigStore::load_from_file(&path).expect("load");

        assert_eq!(
            loaded.get("codec-x", "threads"),
            Some(&ConfigValue::Int(4))
        );
        assert_eq!(
            loaded.get("codec-x", "lossless"),
            Some(&ConfigValue::Bool(false))
        );
        // Float comparison with tolerance
        if let Some(ConfigValue::Float(v)) = loaded.get("codec-y", "level") {
            assert!((*v - 4.2).abs() < 1e-10);
        } else {
            panic!("expected float");
        }

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    // 8. Load from nonexistent file returns empty store
    #[test]
    fn test_load_nonexistent_file() {
        let path = std::env::temp_dir().join("oximedia_plugin_config_nonexistent_xyz.json");
        // Ensure it doesn't exist
        let _ = std::fs::remove_file(&path);
        let store = PluginConfigStore::load_from_file(&path).expect("should return empty");
        assert!(store.is_empty());
    }

    // 9. Store merge
    #[test]
    fn test_store_merge() {
        let mut a = PluginConfigStore::new();
        a.set("plug", "x", ConfigValue::Int(1));

        let mut b = PluginConfigStore::new();
        b.set("plug", "x", ConfigValue::Int(99)); // overwrites
        b.set("plug", "y", ConfigValue::Bool(true));
        b.set("other", "z", ConfigValue::String("hi".to_string()));

        a.merge(&b);
        assert_eq!(a.get("plug", "x"), Some(&ConfigValue::Int(99)));
        assert_eq!(a.get("plug", "y"), Some(&ConfigValue::Bool(true)));
        assert_eq!(
            a.get("other", "z"),
            Some(&ConfigValue::String("hi".to_string()))
        );
    }

    // 10. PluginConfig iter
    #[test]
    fn test_plugin_config_iter() {
        let mut cfg = PluginConfig::new();
        cfg.set("a", ConfigValue::Int(1));
        cfg.set("b", ConfigValue::Int(2));
        let keys: Vec<&str> = cfg.iter().map(|(k, _)| k).collect();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"a"));
        assert!(keys.contains(&"b"));
    }

    // 11. ConfigValue nested list serialise/deserialise
    #[test]
    fn test_config_value_nested_list_roundtrip() {
        let val = ConfigValue::List(vec![
            ConfigValue::Bool(true),
            ConfigValue::Int(7),
            ConfigValue::String("item".to_string()),
        ]);
        let json = serde_json::to_string(&val).expect("serialize");
        let back: ConfigValue = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(val, back);
    }

    // 12. plugin_names lists all plugins
    #[test]
    fn test_plugin_names() {
        let mut store = PluginConfigStore::new();
        store.set("alpha", "k", ConfigValue::Bool(true));
        store.set("beta", "k", ConfigValue::Bool(false));
        let mut names = store.plugin_names();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
    }
}
