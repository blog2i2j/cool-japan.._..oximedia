//! Plugin configuration persistence — save and restore plugin settings between sessions.
//!
//! This module provides a [`PluginConfigStore`] that serialises plugin configuration
//! key-value maps to JSON files and reloads them on startup.  Each plugin's configuration
//! is stored in a separate namespace (its plugin name) within a single backing file.
//!
//! # Design
//!
//! - [`PluginConfig`] represents one plugin's settings as a `HashMap<String, ConfigValue>`.
//! - [`PluginConfigStore`] manages all plugin configs in memory and persists them to a
//!   directory on disk.
//! - [`ConfigValue`] is a typed value that covers the common configuration primitives:
//!   `bool`, `i64`, `f64`, and `String`.
//!
//! # Example
//!
//! ```rust
//! use oximedia_plugin::config_persistence::{PluginConfigStore, ConfigValue};
//!
//! let mut store = PluginConfigStore::new_in_memory();
//! store.set("my-plugin", "threads", ConfigValue::Int(4));
//! store.set("my-plugin", "quality", ConfigValue::Float(0.85));
//! store.set("my-plugin", "preset", ConfigValue::Str("fast".to_string()));
//!
//! assert_eq!(store.get("my-plugin", "threads"), Some(&ConfigValue::Int(4)));
//! assert_eq!(store.get("my-plugin", "missing"), None);
//! ```

use crate::error::{PluginError, PluginResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ── ConfigValue ───────────────────────────────────────────────────────────────

/// A typed plugin configuration value.
///
/// Covers the four primitive types most commonly needed in plugin settings.
/// Complex structures should be serialised to a JSON string and stored as
/// [`ConfigValue::Str`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase", tag = "type", content = "value")]
pub enum ConfigValue {
    /// Boolean flag.
    Bool(bool),
    /// Signed 64-bit integer.
    Int(i64),
    /// 64-bit floating-point number.
    Float(f64),
    /// UTF-8 string.
    Str(String),
}

impl ConfigValue {
    /// Return the value as a `bool` if the variant is [`ConfigValue::Bool`].
    pub fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Return the value as an `i64` if the variant is [`ConfigValue::Int`].
    pub fn as_int(&self) -> Option<i64> {
        if let Self::Int(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Return the value as an `f64` if the variant is [`ConfigValue::Float`].
    pub fn as_float(&self) -> Option<f64> {
        if let Self::Float(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Return a reference to the string if the variant is [`ConfigValue::Str`].
    pub fn as_str(&self) -> Option<&str> {
        if let Self::Str(v) = self {
            Some(v.as_str())
        } else {
            None
        }
    }
}

impl std::fmt::Display for ConfigValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bool(v) => write!(f, "{v}"),
            Self::Int(v) => write!(f, "{v}"),
            Self::Float(v) => write!(f, "{v}"),
            Self::Str(v) => write!(f, "{v}"),
        }
    }
}

// ── PluginConfig ──────────────────────────────────────────────────────────────

/// The configuration map for a single plugin.
///
/// Keyed by arbitrary string keys.  Order is not guaranteed.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PluginConfig {
    /// Key → value settings for the plugin.
    pub settings: HashMap<String, ConfigValue>,
}

impl PluginConfig {
    /// Create an empty `PluginConfig`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or update a setting.
    pub fn set(&mut self, key: impl Into<String>, value: ConfigValue) {
        self.settings.insert(key.into(), value);
    }

    /// Retrieve a setting by key.
    pub fn get(&self, key: &str) -> Option<&ConfigValue> {
        self.settings.get(key)
    }

    /// Remove a setting and return its previous value, if any.
    pub fn remove(&mut self, key: &str) -> Option<ConfigValue> {
        self.settings.remove(key)
    }

    /// Return `true` if the config contains no settings.
    pub fn is_empty(&self) -> bool {
        self.settings.is_empty()
    }

    /// Return the number of settings in this config.
    pub fn len(&self) -> usize {
        self.settings.len()
    }

    /// Merge settings from `other` into `self`.  Settings in `other` overwrite
    /// matching keys in `self`.
    pub fn merge_from(&mut self, other: &PluginConfig) {
        for (k, v) in &other.settings {
            self.settings.insert(k.clone(), v.clone());
        }
    }
}

// ── StoredConfigs ─────────────────────────────────────────────────────────────

/// On-disk serialisation format: a map from plugin name to its config.
#[derive(Debug, Default, Serialize, Deserialize)]
struct StoredConfigs {
    /// Top-level map: plugin name → [`PluginConfig`].
    plugins: HashMap<String, PluginConfig>,
}

// ── PluginConfigStore ─────────────────────────────────────────────────────────

/// Manages plugin configurations for all registered plugins.
///
/// Configurations are stored in memory and can optionally be persisted to and
/// loaded from a JSON file.  The store is **not** thread-safe by itself; wrap
/// in an `Arc<Mutex<PluginConfigStore>>` if shared access is needed.
///
/// # Persistence
///
/// When a `storage_path` is provided, [`save`](Self::save) writes the entire
/// store to `<storage_path>/plugin_configs.json`.  [`load`](Self::load) reads
/// that file and merges the stored settings into the in-memory store (existing
/// settings are replaced, unknown keys are added).
///
/// An in-memory-only store (no persistence) is created via
/// [`new_in_memory`](Self::new_in_memory).
pub struct PluginConfigStore {
    configs: HashMap<String, PluginConfig>,
    storage_path: Option<PathBuf>,
}

impl PluginConfigStore {
    /// Create a new store that never writes to disk.
    pub fn new_in_memory() -> Self {
        Self {
            configs: HashMap::new(),
            storage_path: None,
        }
    }

    /// Create a new store backed by the given directory.
    ///
    /// The directory does **not** need to exist; it will be created on the
    /// first call to [`save`](Self::save).
    pub fn new(storage_dir: impl Into<PathBuf>) -> Self {
        Self {
            configs: HashMap::new(),
            storage_path: Some(storage_dir.into()),
        }
    }

    /// Return the path to the backing JSON file, if this store has one.
    pub fn config_file_path(&self) -> Option<PathBuf> {
        self.storage_path
            .as_deref()
            .map(|dir| dir.join("plugin_configs.json"))
    }

    // ── Read helpers ──────────────────────────────────────────────────────────

    /// Get the [`PluginConfig`] for a specific plugin.
    ///
    /// Returns `None` if no configuration has been stored for that plugin.
    pub fn plugin_config(&self, plugin_name: &str) -> Option<&PluginConfig> {
        self.configs.get(plugin_name)
    }

    /// Get a mutable reference to the [`PluginConfig`] for a specific plugin,
    /// inserting an empty one if it did not exist.
    pub fn plugin_config_mut(&mut self, plugin_name: &str) -> &mut PluginConfig {
        self.configs
            .entry(plugin_name.to_string())
            .or_insert_with(PluginConfig::new)
    }

    /// Look up a single setting value.
    pub fn get(&self, plugin_name: &str, key: &str) -> Option<&ConfigValue> {
        self.configs.get(plugin_name)?.get(key)
    }

    // ── Write helpers ─────────────────────────────────────────────────────────

    /// Insert or overwrite a single setting for a plugin.
    pub fn set(&mut self, plugin_name: &str, key: &str, value: ConfigValue) {
        self.configs
            .entry(plugin_name.to_string())
            .or_insert_with(PluginConfig::new)
            .set(key, value);
    }

    /// Remove a single setting for a plugin.
    ///
    /// Returns the old value, or `None` if the key was not present.
    pub fn remove_setting(&mut self, plugin_name: &str, key: &str) -> Option<ConfigValue> {
        self.configs.get_mut(plugin_name)?.remove(key)
    }

    /// Remove all configuration for a plugin.
    pub fn remove_plugin(&mut self, plugin_name: &str) {
        self.configs.remove(plugin_name);
    }

    /// Return the names of all plugins that have stored configuration.
    pub fn plugin_names(&self) -> Vec<&str> {
        self.configs.keys().map(String::as_str).collect()
    }

    // ── Persistence ───────────────────────────────────────────────────────────

    /// Persist all configurations to disk.
    ///
    /// If this store was created via [`new_in_memory`](Self::new_in_memory)
    /// this call is a no-op and returns `Ok(())`.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created, the file cannot
    /// be written, or JSON serialisation fails.
    pub fn save(&self) -> PluginResult<()> {
        let Some(dir) = &self.storage_path else {
            return Ok(());
        };

        std::fs::create_dir_all(dir)?;

        let file_path = dir.join("plugin_configs.json");
        let stored = StoredConfigs {
            plugins: self.configs.clone(),
        };
        let json = serde_json::to_string_pretty(&stored)?;
        std::fs::write(&file_path, json.as_bytes())?;

        tracing::debug!("Saved plugin configs to {}", file_path.display());
        Ok(())
    }

    /// Load configurations from disk, merging them into the current store.
    ///
    /// Settings in the file overwrite in-memory values for matching keys.
    /// Unknown plugins / keys are added.
    ///
    /// If this store was created via [`new_in_memory`](Self::new_in_memory)
    /// or the backing file does not yet exist, this call is a no-op.
    ///
    /// # Errors
    ///
    /// Returns an error if the file exists but cannot be read, or if JSON
    /// deserialisation fails.
    pub fn load(&mut self) -> PluginResult<()> {
        let file_path = match self.config_file_path() {
            Some(p) => p,
            None => return Ok(()),
        };

        if !file_path.exists() {
            tracing::debug!(
                "No plugin config file found at {}; starting fresh.",
                file_path.display()
            );
            return Ok(());
        }

        let bytes = std::fs::read(&file_path)?;
        let stored: StoredConfigs = serde_json::from_slice(&bytes)?;

        for (plugin_name, config) in stored.plugins {
            let entry = self
                .configs
                .entry(plugin_name)
                .or_insert_with(PluginConfig::new);
            entry.merge_from(&config);
        }

        tracing::debug!("Loaded plugin configs from {}", file_path.display());
        Ok(())
    }

    /// Load from an explicit path, merging into the current store.
    ///
    /// Useful when the config file lives in a non-standard location.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or JSON is malformed.
    pub fn load_from(&mut self, path: &Path) -> PluginResult<()> {
        if !path.exists() {
            return Err(PluginError::NotFound(format!(
                "Config file not found: {}",
                path.display()
            )));
        }

        let bytes = std::fs::read(path)?;
        let stored: StoredConfigs = serde_json::from_slice(&bytes)?;

        for (plugin_name, config) in stored.plugins {
            let entry = self
                .configs
                .entry(plugin_name)
                .or_insert_with(PluginConfig::new);
            entry.merge_from(&config);
        }
        Ok(())
    }

    /// Save to an explicit path (e.g., for export / backup).
    ///
    /// # Errors
    ///
    /// Returns an error if the parent directory cannot be created, the file
    /// cannot be written, or serialisation fails.
    pub fn save_to(&self, path: &Path) -> PluginResult<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let stored = StoredConfigs {
            plugins: self.configs.clone(),
        };
        let json = serde_json::to_string_pretty(&stored)?;
        std::fs::write(path, json.as_bytes())?;
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    fn temp_config_dir(suffix: &str) -> PathBuf {
        temp_dir().join(format!("oximedia_plugin_config_test_{suffix}"))
    }

    // 1. new_in_memory creates an empty, non-persisting store
    #[test]
    fn test_new_in_memory() {
        let store = PluginConfigStore::new_in_memory();
        assert!(store.plugin_names().is_empty());
        assert!(store.config_file_path().is_none());
    }

    // 2. set and get a value
    #[test]
    fn test_set_get() {
        let mut store = PluginConfigStore::new_in_memory();
        store.set("my-plugin", "threads", ConfigValue::Int(4));
        assert_eq!(store.get("my-plugin", "threads"), Some(&ConfigValue::Int(4)));
        assert_eq!(store.get("my-plugin", "missing"), None);
        assert_eq!(store.get("other-plugin", "threads"), None);
    }

    // 3. set multiple types
    #[test]
    fn test_set_multiple_types() {
        let mut store = PluginConfigStore::new_in_memory();
        store.set("p", "flag", ConfigValue::Bool(true));
        store.set("p", "count", ConfigValue::Int(42));
        store.set("p", "ratio", ConfigValue::Float(0.5));
        store.set("p", "name", ConfigValue::Str("fast".to_string()));

        assert_eq!(store.get("p", "flag"), Some(&ConfigValue::Bool(true)));
        assert_eq!(store.get("p", "count"), Some(&ConfigValue::Int(42)));
        assert_eq!(store.get("p", "ratio"), Some(&ConfigValue::Float(0.5)));
        assert_eq!(
            store.get("p", "name"),
            Some(&ConfigValue::Str("fast".to_string()))
        );
    }

    // 4. remove_setting removes a key
    #[test]
    fn test_remove_setting() {
        let mut store = PluginConfigStore::new_in_memory();
        store.set("p", "key", ConfigValue::Int(1));
        let old = store.remove_setting("p", "key");
        assert_eq!(old, Some(ConfigValue::Int(1)));
        assert_eq!(store.get("p", "key"), None);
    }

    // 5. remove_plugin removes all settings for a plugin
    #[test]
    fn test_remove_plugin() {
        let mut store = PluginConfigStore::new_in_memory();
        store.set("p", "a", ConfigValue::Bool(false));
        store.set("p", "b", ConfigValue::Int(7));
        store.remove_plugin("p");
        assert!(store.plugin_config("p").is_none());
    }

    // 6. plugin_names returns all stored plugin names
    #[test]
    fn test_plugin_names() {
        let mut store = PluginConfigStore::new_in_memory();
        store.set("alpha", "x", ConfigValue::Bool(true));
        store.set("beta", "y", ConfigValue::Int(2));
        let mut names = store.plugin_names();
        names.sort_unstable();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    // 7. save_to / load_from round-trip (disk)
    #[test]
    fn test_save_load_roundtrip() {
        let dir = temp_config_dir("roundtrip");
        std::fs::create_dir_all(&dir).ok();
        let file = dir.join("test_config.json");

        let mut store = PluginConfigStore::new_in_memory();
        store.set("codec-x", "bitrate", ConfigValue::Int(1_000_000));
        store.set("codec-x", "hw_accel", ConfigValue::Bool(false));
        store.set("filter-y", "intensity", ConfigValue::Float(0.75));

        store.save_to(&file).expect("save_to should succeed");
        assert!(file.exists());

        let mut loaded = PluginConfigStore::new_in_memory();
        loaded.load_from(&file).expect("load_from should succeed");

        assert_eq!(
            loaded.get("codec-x", "bitrate"),
            Some(&ConfigValue::Int(1_000_000))
        );
        assert_eq!(
            loaded.get("codec-x", "hw_accel"),
            Some(&ConfigValue::Bool(false))
        );
        assert_eq!(
            loaded.get("filter-y", "intensity"),
            Some(&ConfigValue::Float(0.75))
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    // 8. save / load using storage_path
    #[test]
    fn test_save_load_via_storage_path() {
        let dir = temp_config_dir("storage_path");

        let mut store = PluginConfigStore::new(dir.clone());
        store.set("my-plugin", "quality", ConfigValue::Int(8));
        store.save().expect("save should succeed");

        let file_path = store.config_file_path().expect("should have file path");
        assert!(file_path.exists());

        let mut store2 = PluginConfigStore::new(dir.clone());
        store2.load().expect("load should succeed");
        assert_eq!(
            store2.get("my-plugin", "quality"),
            Some(&ConfigValue::Int(8))
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    // 9. in-memory save is a no-op (returns Ok)
    #[test]
    fn test_in_memory_save_is_noop() {
        let store = PluginConfigStore::new_in_memory();
        assert!(store.save().is_ok());
    }

    // 10. in-memory load is a no-op (returns Ok)
    #[test]
    fn test_in_memory_load_is_noop() {
        let mut store = PluginConfigStore::new_in_memory();
        assert!(store.load().is_ok());
    }

    // 11. load_from missing file returns an error
    #[test]
    fn test_load_from_missing_file() {
        let mut store = PluginConfigStore::new_in_memory();
        let err = store.load_from(Path::new("/nonexistent/path/config.json"));
        assert!(err.is_err());
    }

    // 12. merge_from overwrites matching keys, adds new ones
    #[test]
    fn test_merge_from() {
        let mut base = PluginConfig::new();
        base.set("a", ConfigValue::Int(1));
        base.set("b", ConfigValue::Int(2));

        let mut overlay = PluginConfig::new();
        overlay.set("b", ConfigValue::Int(99)); // overwrite
        overlay.set("c", ConfigValue::Bool(true)); // new key

        base.merge_from(&overlay);

        assert_eq!(base.get("a"), Some(&ConfigValue::Int(1)));
        assert_eq!(base.get("b"), Some(&ConfigValue::Int(99)));
        assert_eq!(base.get("c"), Some(&ConfigValue::Bool(true)));
    }

    // 13. ConfigValue display
    #[test]
    fn test_config_value_display() {
        assert_eq!(ConfigValue::Bool(true).to_string(), "true");
        assert_eq!(ConfigValue::Int(-7).to_string(), "-7");
        assert_eq!(ConfigValue::Float(1.5).to_string(), "1.5");
        assert_eq!(ConfigValue::Str("hi".to_string()).to_string(), "hi");
    }

    // 14. ConfigValue typed accessors
    #[test]
    fn test_config_value_typed_accessors() {
        assert_eq!(ConfigValue::Bool(false).as_bool(), Some(false));
        assert_eq!(ConfigValue::Bool(false).as_int(), None);

        assert_eq!(ConfigValue::Int(42).as_int(), Some(42));
        assert_eq!(ConfigValue::Int(42).as_float(), None);

        assert!((ConfigValue::Float(3.14).as_float().unwrap() - 3.14).abs() < 1e-10);
        assert_eq!(ConfigValue::Float(3.14).as_str(), None);

        let s = ConfigValue::Str("hello".to_string());
        assert_eq!(s.as_str(), Some("hello"));
        assert_eq!(s.as_bool(), None);
    }

    // 15. PluginConfig len and is_empty
    #[test]
    fn test_plugin_config_len_empty() {
        let mut cfg = PluginConfig::new();
        assert!(cfg.is_empty());
        assert_eq!(cfg.len(), 0);
        cfg.set("k", ConfigValue::Int(1));
        assert!(!cfg.is_empty());
        assert_eq!(cfg.len(), 1);
    }

    // 16. overwriting a setting in the store
    #[test]
    fn test_overwrite_setting() {
        let mut store = PluginConfigStore::new_in_memory();
        store.set("p", "key", ConfigValue::Int(1));
        store.set("p", "key", ConfigValue::Int(2));
        assert_eq!(store.get("p", "key"), Some(&ConfigValue::Int(2)));
    }
}
