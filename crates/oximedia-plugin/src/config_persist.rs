//! Plugin configuration persistence.
//!
//! Provides save/load of plugin settings between sessions using JSON.
//! Each plugin has an isolated key-value configuration namespace stored
//! as a JSON object in a per-plugin file.
//!
//! # Design
//!
//! - [`PluginConfig`] holds in-memory key-value pairs for a single plugin.
//! - [`ConfigStore`] manages a directory of per-plugin config files and
//!   handles serialisation / deserialisation.
//! - All values are stored as [`serde_json::Value`] to allow arbitrary types.
//! - File names use the pattern `<plugin-name>.config.json`.

use crate::error::{PluginError, PluginResult};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ── PluginConfig ─────────────────────────────────────────────────────────────

/// In-memory configuration for a single plugin.
///
/// Keys are arbitrary strings; values are JSON values which can represent
/// strings, numbers, booleans, arrays, or objects.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PluginConfig {
    /// Plugin identifier (used as the file-name stem).
    pub plugin_name: String,
    /// Key-value settings.
    #[serde(default)]
    pub settings: HashMap<String, Value>,
}

impl PluginConfig {
    /// Create an empty configuration for `plugin_name`.
    pub fn new(plugin_name: impl Into<String>) -> Self {
        Self {
            plugin_name: plugin_name.into(),
            settings: HashMap::new(),
        }
    }

    /// Insert or replace a string value.
    pub fn set_str(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.settings
            .insert(key.into(), Value::String(value.into()));
    }

    /// Insert or replace a numeric (i64) value.
    pub fn set_int(&mut self, key: impl Into<String>, value: i64) {
        self.settings.insert(key.into(), Value::from(value));
    }

    /// Insert or replace a boolean value.
    pub fn set_bool(&mut self, key: impl Into<String>, value: bool) {
        self.settings.insert(key.into(), Value::Bool(value));
    }

    /// Insert or replace a floating-point value.
    ///
    /// If `value` is not finite the key is removed.
    pub fn set_float(&mut self, key: impl Into<String>, value: f64) {
        if let Some(v) = Value::from(value).as_f64() {
            self.settings.insert(
                key.into(),
                serde_json::Number::from_f64(v).map_or(Value::Null, Value::Number),
            );
        } else {
            self.settings.remove(&key.into());
        }
    }

    /// Insert an arbitrary JSON value.
    pub fn set_raw(&mut self, key: impl Into<String>, value: Value) {
        self.settings.insert(key.into(), value);
    }

    /// Retrieve a string value.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.settings.get(key)?.as_str()
    }

    /// Retrieve an integer value.
    pub fn get_int(&self, key: &str) -> Option<i64> {
        self.settings.get(key)?.as_i64()
    }

    /// Retrieve a boolean value.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.settings.get(key)?.as_bool()
    }

    /// Retrieve a float value.
    pub fn get_float(&self, key: &str) -> Option<f64> {
        self.settings.get(key)?.as_f64()
    }

    /// Retrieve a raw JSON value.
    pub fn get_raw(&self, key: &str) -> Option<&Value> {
        self.settings.get(key)
    }

    /// Remove a key, returning the old value if present.
    pub fn remove(&mut self, key: &str) -> Option<Value> {
        self.settings.remove(key)
    }

    /// Return `true` if the configuration has no settings.
    pub fn is_empty(&self) -> bool {
        self.settings.is_empty()
    }

    /// Return the number of settings.
    pub fn len(&self) -> usize {
        self.settings.len()
    }

    /// Serialise to a JSON string.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::Json`] on serialisation failure.
    pub fn to_json(&self) -> PluginResult<String> {
        serde_json::to_string_pretty(self).map_err(PluginError::Json)
    }

    /// Deserialise from a JSON string.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::InvalidManifest`] on parse failure.
    pub fn from_json(json: &str) -> PluginResult<Self> {
        serde_json::from_str(json).map_err(|e| PluginError::InvalidManifest(e.to_string()))
    }
}

// ── ConfigStore ──────────────────────────────────────────────────────────────

/// Manages a directory of per-plugin configuration files.
///
/// Configuration files are stored as `<config_dir>/<plugin_name>.config.json`.
pub struct ConfigStore {
    /// Root directory where config files are stored.
    config_dir: PathBuf,
}

impl ConfigStore {
    /// Create a `ConfigStore` rooted at `config_dir`.
    ///
    /// The directory is created if it does not already exist.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::Io`] if the directory cannot be created.
    pub fn new(config_dir: impl Into<PathBuf>) -> PluginResult<Self> {
        let dir = config_dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self { config_dir: dir })
    }

    /// Compute the file path for `plugin_name`.
    fn path_for(&self, plugin_name: &str) -> PathBuf {
        // Sanitise the plugin name: replace path separators.
        let safe_name = plugin_name.replace(['/', '\\', ':'], "_");
        self.config_dir.join(format!("{safe_name}.config.json"))
    }

    /// Save `config` to disk.
    ///
    /// Overwrites any existing file for the same plugin.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::Json`] or [`PluginError::Io`] on failure.
    pub fn save(&self, config: &PluginConfig) -> PluginResult<()> {
        let json = config.to_json()?;
        let path = self.path_for(&config.plugin_name);
        std::fs::write(&path, json)?;
        Ok(())
    }

    /// Load configuration for `plugin_name` from disk.
    ///
    /// Returns a default (empty) config if no file exists yet, so callers
    /// do not need to handle the "first run" case specially.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::Io`] on read failure (other than file-not-found),
    /// or [`PluginError::InvalidManifest`] if the JSON is malformed.
    pub fn load(&self, plugin_name: &str) -> PluginResult<PluginConfig> {
        let path = self.path_for(plugin_name);
        if !path.exists() {
            return Ok(PluginConfig::new(plugin_name));
        }
        let json = std::fs::read_to_string(&path)?;
        PluginConfig::from_json(&json)
    }

    /// Delete the configuration file for `plugin_name`.
    ///
    /// Returns `Ok(())` even if the file does not exist.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::Io`] if the file exists but cannot be deleted.
    pub fn delete(&self, plugin_name: &str) -> PluginResult<()> {
        let path = self.path_for(plugin_name);
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        Ok(())
    }

    /// List the plugin names that have a saved configuration file.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::Io`] if the directory cannot be read.
    pub fn list_stored(&self) -> PluginResult<Vec<String>> {
        let mut names = Vec::new();
        for entry in std::fs::read_dir(&self.config_dir)? {
            let entry = entry?;
            let file_name = entry.file_name();
            let fname = file_name.to_string_lossy();
            if let Some(stem) = fname.strip_suffix(".config.json") {
                names.push(stem.to_string());
            }
        }
        names.sort();
        Ok(names)
    }

    /// Return the config directory path.
    pub fn config_dir(&self) -> &Path {
        &self.config_dir
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temp_store() -> (ConfigStore, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "oximedia-plugin-config-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        let store = ConfigStore::new(&dir).expect("store");
        (store, dir)
    }

    // 1. PluginConfig starts empty.
    #[test]
    fn test_config_empty() {
        let cfg = PluginConfig::new("my-plugin");
        assert!(cfg.is_empty());
        assert_eq!(cfg.len(), 0);
    }

    // 2. set/get string round-trips.
    #[test]
    fn test_set_get_str() {
        let mut cfg = PluginConfig::new("p");
        cfg.set_str("codec", "h264");
        assert_eq!(cfg.get_str("codec"), Some("h264"));
        assert_eq!(cfg.get_str("missing"), None);
    }

    // 3. set/get int round-trips.
    #[test]
    fn test_set_get_int() {
        let mut cfg = PluginConfig::new("p");
        cfg.set_int("bitrate", 5_000_000);
        assert_eq!(cfg.get_int("bitrate"), Some(5_000_000));
    }

    // 4. set/get bool round-trips.
    #[test]
    fn test_set_get_bool() {
        let mut cfg = PluginConfig::new("p");
        cfg.set_bool("hw_accel", true);
        assert_eq!(cfg.get_bool("hw_accel"), Some(true));
    }

    // 5. set_raw / get_raw.
    #[test]
    fn test_set_get_raw() {
        let mut cfg = PluginConfig::new("p");
        cfg.set_raw("formats", json!(["yuv420p", "nv12"]));
        assert_eq!(cfg.get_raw("formats"), Some(&json!(["yuv420p", "nv12"])));
    }

    // 6. remove deletes key.
    #[test]
    fn test_remove() {
        let mut cfg = PluginConfig::new("p");
        cfg.set_str("key", "val");
        let old = cfg.remove("key");
        assert_eq!(old, Some(Value::String("val".to_string())));
        assert!(cfg.is_empty());
    }

    // 7. JSON serialisation round-trip.
    #[test]
    fn test_json_roundtrip() {
        let data_path = std::env::temp_dir()
            .join("oximedia-plugin-cfg-data")
            .to_string_lossy()
            .into_owned();
        let mut cfg = PluginConfig::new("round-trip");
        cfg.set_str("path", data_path.clone());
        cfg.set_int("threads", 4);
        let json = cfg.to_json().expect("to_json");
        let parsed = PluginConfig::from_json(&json).expect("from_json");
        assert_eq!(parsed.plugin_name, "round-trip");
        assert_eq!(parsed.get_str("path"), Some(data_path.as_str()));
        assert_eq!(parsed.get_int("threads"), Some(4));
    }

    // 8. ConfigStore save/load round-trip.
    #[test]
    fn test_store_save_load() {
        let (store, dir) = temp_store();
        let mut cfg = PluginConfig::new("my-codec-plugin");
        cfg.set_str("profile", "high");
        cfg.set_int("crf", 23);
        store.save(&cfg).expect("save");

        let loaded = store.load("my-codec-plugin").expect("load");
        assert_eq!(loaded.get_str("profile"), Some("high"));
        assert_eq!(loaded.get_int("crf"), Some(23));

        let _ = std::fs::remove_dir_all(&dir);
    }

    // 9. Load non-existent returns empty default.
    #[test]
    fn test_load_nonexistent_returns_default() {
        let (store, dir) = temp_store();
        let cfg = store.load("never-saved").expect("load default");
        assert!(cfg.is_empty());
        assert_eq!(cfg.plugin_name, "never-saved");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 10. Delete removes the file.
    #[test]
    fn test_delete() {
        let (store, dir) = temp_store();
        let mut cfg = PluginConfig::new("del-plugin");
        cfg.set_bool("enabled", true);
        store.save(&cfg).expect("save");
        store.delete("del-plugin").expect("delete");
        // Loading after delete returns default.
        let after = store.load("del-plugin").expect("load after delete");
        assert!(after.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 11. list_stored returns saved plugin names.
    #[test]
    fn test_list_stored() {
        let (store, dir) = temp_store();
        store.save(&PluginConfig::new("alpha")).expect("save alpha");
        store.save(&PluginConfig::new("beta")).expect("save beta");
        let names = store.list_stored().expect("list");
        assert!(names.contains(&"alpha".to_string()));
        assert!(names.contains(&"beta".to_string()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 12. Plugin name with special chars is sanitised.
    #[test]
    fn test_sanitised_name() {
        let (store, dir) = temp_store();
        let mut cfg = PluginConfig::new("org/codec:v1");
        cfg.set_str("x", "y");
        store.save(&cfg).expect("save");
        // Should not panic or produce invalid paths.
        let loaded = store.load("org/codec:v1").expect("load");
        assert_eq!(loaded.get_str("x"), Some("y"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 13. ConfigStore::config_dir returns the root path.
    #[test]
    fn test_config_dir_accessor() {
        let (store, dir) = temp_store();
        assert_eq!(store.config_dir(), dir.as_path());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
