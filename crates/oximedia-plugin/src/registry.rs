//! Central plugin registry.
//!
//! The [`PluginRegistry`] is the main entry point for managing plugins.
//! It handles registration, discovery, and codec lookup across all
//! loaded plugins.

use crate::error::{PluginError, PluginResult};
use crate::traits::{CodecPlugin, CodecPluginInfo, PluginCapability, PLUGIN_API_VERSION};
use oximedia_codec::{CodecResult, EncoderConfig, VideoDecoder, VideoEncoder};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// Central registry for all loaded codec plugins.
///
/// The registry maintains a list of registered plugins and provides
/// methods to discover codecs, create decoders/encoders, and manage
/// the plugin lifecycle.
///
/// # Thread Safety
///
/// The registry uses interior mutability (`RwLock`) so it can be
/// shared across threads. Multiple readers can query the registry
/// concurrently; writes (registration) acquire an exclusive lock.
///
/// # Example
///
/// ```rust
/// use oximedia_plugin::{PluginRegistry, StaticPlugin, CodecPluginInfo, PluginCapability};
/// use std::sync::Arc;
/// use std::collections::HashMap;
///
/// let registry = PluginRegistry::new();
///
/// let info = CodecPluginInfo {
///     name: "example".to_string(),
///     version: "0.1.0".to_string(),
///     author: "Test".to_string(),
///     description: "Example plugin".to_string(),
///     api_version: oximedia_plugin::PLUGIN_API_VERSION,
///     license: "MIT".to_string(),
///     patent_encumbered: false,
/// };
///
/// let plugin = StaticPlugin::new(info)
///     .add_capability(PluginCapability {
///         codec_name: "test".to_string(),
///         can_decode: true,
///         can_encode: false,
///         pixel_formats: vec![],
///         properties: HashMap::new(),
///     });
///
/// registry.register(Arc::new(plugin)).expect("should register");
/// assert!(registry.has_codec("test"));
/// ```
pub struct PluginRegistry {
    plugins: RwLock<Vec<Arc<dyn CodecPlugin>>>,
    search_paths: Vec<PathBuf>,
}

impl PluginRegistry {
    /// Create a new empty plugin registry with default search paths.
    #[must_use]
    pub fn new() -> Self {
        Self {
            plugins: RwLock::new(Vec::new()),
            search_paths: Self::default_search_paths(),
        }
    }

    /// Create a registry with no search paths (for testing).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            plugins: RwLock::new(Vec::new()),
            search_paths: Vec::new(),
        }
    }

    /// Add a directory to search for plugins.
    pub fn add_search_path(&mut self, path: PathBuf) {
        if !self.search_paths.contains(&path) {
            self.search_paths.push(path);
        }
    }

    /// Get the list of search paths.
    pub fn search_paths(&self) -> &[PathBuf] {
        &self.search_paths
    }

    /// Compute default search paths for plugin discovery.
    ///
    /// The following paths are checked (in order):
    /// 1. `$OXIMEDIA_PLUGIN_PATH` (colon-separated on Unix, semicolon on Windows)
    /// 2. `~/.oximedia/plugins/`
    /// 3. `/usr/lib/oximedia/plugins/` (Unix only)
    /// 4. `/usr/local/lib/oximedia/plugins/` (Unix only)
    #[must_use]
    pub fn default_search_paths() -> Vec<PathBuf> {
        let mut paths = Vec::new();

        // Check environment variable
        if let Ok(env_paths) = std::env::var("OXIMEDIA_PLUGIN_PATH") {
            let separator = if cfg!(windows) { ';' } else { ':' };
            for p in env_paths.split(separator) {
                let path = PathBuf::from(p);
                if !paths.contains(&path) {
                    paths.push(path);
                }
            }
        }

        // User home directory
        if let Some(home) = home_dir() {
            let user_plugins = home.join(".oximedia").join("plugins");
            if !paths.contains(&user_plugins) {
                paths.push(user_plugins);
            }
        }

        // System paths (Unix)
        #[cfg(unix)]
        {
            let sys_paths = [
                PathBuf::from("/usr/lib/oximedia/plugins"),
                PathBuf::from("/usr/local/lib/oximedia/plugins"),
            ];
            for p in sys_paths {
                if !paths.contains(&p) {
                    paths.push(p);
                }
            }
        }

        paths
    }

    /// Register a static plugin instance.
    ///
    /// The plugin is validated (API version check, duplicate name check)
    /// before being added to the registry.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::ApiIncompatible`] if the API version does
    /// not match, or [`PluginError::AlreadyRegistered`] if a plugin
    /// with the same name is already loaded.
    pub fn register(&self, plugin: Arc<dyn CodecPlugin>) -> PluginResult<()> {
        let info = plugin.info();

        // Validate API version
        if info.api_version != PLUGIN_API_VERSION {
            return Err(PluginError::ApiIncompatible(format!(
                "Plugin '{}' has API v{}, host expects v{PLUGIN_API_VERSION}",
                info.name, info.api_version
            )));
        }

        let mut plugins = self
            .plugins
            .write()
            .map_err(|e| PluginError::InitFailed(format!("Lock poisoned: {e}")))?;

        // Check for duplicates
        for existing in plugins.iter() {
            if existing.info().name == info.name {
                return Err(PluginError::AlreadyRegistered(info.name));
            }
        }

        tracing::info!(
            "Registered plugin: {} v{} ({} codec(s))",
            info.name,
            info.version,
            plugin.capabilities().len()
        );

        if info.patent_encumbered {
            tracing::warn!(
                "Plugin '{}' contains patent-encumbered codecs. Use at your own risk.",
                info.name
            );
        }

        plugins.push(plugin);
        Ok(())
    }

    /// Load a plugin from a shared library file.
    ///
    /// Requires the `dynamic-loading` feature.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::DynamicLoadingDisabled`] if the feature
    /// is not enabled, or propagates loading errors.
    ///
    /// # Safety
    ///
    /// Loading a shared library executes arbitrary code. Only load
    /// plugins from trusted sources.
    #[cfg(feature = "dynamic-loading")]
    pub fn load_plugin(&self, path: &Path) -> PluginResult<()> {
        let loaded = crate::loader::LoadedPlugin::load(path)?;
        self.register(loaded.into_plugin())
    }

    /// Load a plugin from a shared library file.
    ///
    /// This is a stub that returns an error when the `dynamic-loading`
    /// feature is not enabled.
    #[cfg(not(feature = "dynamic-loading"))]
    pub fn load_plugin(&self, _path: &Path) -> PluginResult<()> {
        Err(PluginError::DynamicLoadingDisabled)
    }

    /// Discover and load all plugins from search paths.
    ///
    /// Scans each search path for `plugin.json` manifest files,
    /// validates them, and loads the corresponding shared libraries.
    ///
    /// Returns information about all successfully loaded plugins.
    /// Errors for individual plugins are logged but do not prevent
    /// other plugins from loading.
    ///
    /// Requires the `dynamic-loading` feature.
    #[cfg(feature = "dynamic-loading")]
    pub fn discover_plugins(&self) -> PluginResult<Vec<CodecPluginInfo>> {
        let mut loaded = Vec::new();

        for search_path in &self.search_paths {
            if !search_path.is_dir() {
                tracing::debug!(
                    "Plugin search path does not exist: {}",
                    search_path.display()
                );
                continue;
            }

            let entries = std::fs::read_dir(search_path)?;
            for entry in entries {
                let entry = entry?;
                let path = entry.path();

                // Look for plugin.json in subdirectories
                let manifest_path = if path.is_dir() {
                    path.join("plugin.json")
                } else if path.extension().and_then(|e| e.to_str()) == Some("json") {
                    path.clone()
                } else {
                    continue;
                };

                if !manifest_path.exists() {
                    continue;
                }

                match self.load_from_manifest(&manifest_path) {
                    Ok(info) => loaded.push(info),
                    Err(e) => {
                        tracing::warn!(
                            "Failed to load plugin from {}: {e}",
                            manifest_path.display()
                        );
                    }
                }
            }
        }

        Ok(loaded)
    }

    /// Discover plugins stub when dynamic loading is disabled.
    #[cfg(not(feature = "dynamic-loading"))]
    pub fn discover_plugins(&self) -> PluginResult<Vec<CodecPluginInfo>> {
        Err(PluginError::DynamicLoadingDisabled)
    }

    /// Load a plugin from its manifest file.
    #[cfg(feature = "dynamic-loading")]
    fn load_from_manifest(&self, manifest_path: &Path) -> PluginResult<CodecPluginInfo> {
        let manifest = crate::manifest::PluginManifest::from_file(manifest_path)?;
        manifest.validate()?;

        let lib_path = manifest.library_path(manifest_path).ok_or_else(|| {
            PluginError::LoadFailed("Cannot determine library path from manifest".to_string())
        })?;

        self.load_plugin(&lib_path)?;

        // Return the info from the last registered plugin
        let plugins = self
            .plugins
            .read()
            .map_err(|e| PluginError::InitFailed(format!("Lock poisoned: {e}")))?;

        plugins.last().map(|p| p.info()).ok_or_else(|| {
            PluginError::InitFailed("Plugin was not added after loading".to_string())
        })
    }

    /// List all registered plugins.
    pub fn list_plugins(&self) -> Vec<CodecPluginInfo> {
        let plugins = match self.plugins.read() {
            Ok(p) => p,
            Err(_) => return Vec::new(),
        };
        plugins.iter().map(|p| p.info()).collect()
    }

    /// List all available codecs across all plugins.
    pub fn list_codecs(&self) -> Vec<PluginCapability> {
        let plugins = match self.plugins.read() {
            Ok(p) => p,
            Err(_) => return Vec::new(),
        };
        plugins.iter().flat_map(|p| p.capabilities()).collect()
    }

    /// Find and create a decoder for a given codec name.
    ///
    /// Searches all registered plugins for one that can decode the
    /// requested codec, and creates a new decoder instance.
    ///
    /// # Errors
    ///
    /// Returns error if no plugin supports decoding the given codec.
    pub fn find_decoder(&self, codec_name: &str) -> CodecResult<Box<dyn VideoDecoder>> {
        let plugins = self
            .plugins
            .read()
            .map_err(|e| oximedia_codec::CodecError::Internal(format!("Lock poisoned: {e}")))?;

        for plugin in plugins.iter() {
            if plugin.can_decode(codec_name) {
                return plugin.create_decoder(codec_name);
            }
        }

        Err(oximedia_codec::CodecError::UnsupportedFeature(format!(
            "No plugin provides decoder for '{codec_name}'"
        )))
    }

    /// Find and create an encoder for a given codec name.
    ///
    /// Searches all registered plugins for one that can encode the
    /// requested codec, and creates a new encoder instance.
    ///
    /// # Errors
    ///
    /// Returns error if no plugin supports encoding the given codec.
    pub fn find_encoder(
        &self,
        codec_name: &str,
        config: EncoderConfig,
    ) -> CodecResult<Box<dyn VideoEncoder>> {
        let plugins = self
            .plugins
            .read()
            .map_err(|e| oximedia_codec::CodecError::Internal(format!("Lock poisoned: {e}")))?;

        for plugin in plugins.iter() {
            if plugin.can_encode(codec_name) {
                return plugin.create_encoder(codec_name, config);
            }
        }

        Err(oximedia_codec::CodecError::UnsupportedFeature(format!(
            "No plugin provides encoder for '{codec_name}'"
        )))
    }

    /// Check if any plugin provides a given codec (decode or encode).
    pub fn has_codec(&self, codec_name: &str) -> bool {
        let plugins = match self.plugins.read() {
            Ok(p) => p,
            Err(_) => return false,
        };
        plugins.iter().any(|p| p.supports_codec(codec_name))
    }

    /// Check if any plugin can decode a given codec.
    pub fn has_decoder(&self, codec_name: &str) -> bool {
        let plugins = match self.plugins.read() {
            Ok(p) => p,
            Err(_) => return false,
        };
        plugins.iter().any(|p| p.can_decode(codec_name))
    }

    /// Check if any plugin can encode a given codec.
    pub fn has_encoder(&self, codec_name: &str) -> bool {
        let plugins = match self.plugins.read() {
            Ok(p) => p,
            Err(_) => return false,
        };
        plugins.iter().any(|p| p.can_encode(codec_name))
    }

    /// Get the number of registered plugins.
    pub fn plugin_count(&self) -> usize {
        match self.plugins.read() {
            Ok(p) => p.len(),
            Err(_) => 0,
        }
    }

    /// Unload all registered plugins.
    pub fn clear(&self) {
        if let Ok(mut plugins) = self.plugins.write() {
            plugins.clear();
        }
    }

    /// Find the plugin that provides a given codec.
    pub fn find_plugin_for_codec(&self, codec_name: &str) -> Option<CodecPluginInfo> {
        let plugins = self.plugins.read().ok()?;
        plugins
            .iter()
            .find(|p| p.supports_codec(codec_name))
            .map(|p| p.info())
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Get the user's home directory in a cross-platform way.
fn home_dir() -> Option<PathBuf> {
    #[cfg(unix)]
    {
        std::env::var("HOME").ok().map(PathBuf::from)
    }
    #[cfg(windows)]
    {
        std::env::var("USERPROFILE").ok().map(PathBuf::from)
    }
    #[cfg(not(any(unix, windows)))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::static_plugin::StaticPlugin;
    use std::collections::HashMap;

    fn make_test_plugin(name: &str, codecs: &[(&str, bool, bool)]) -> Arc<dyn CodecPlugin> {
        let info = CodecPluginInfo {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            author: "Test".to_string(),
            description: format!("Test plugin: {name}"),
            api_version: PLUGIN_API_VERSION,
            license: "MIT".to_string(),
            patent_encumbered: false,
        };

        let mut plugin = StaticPlugin::new(info);
        for (codec_name, decode, encode) in codecs {
            plugin = plugin.add_capability(PluginCapability {
                codec_name: (*codec_name).to_string(),
                can_decode: *decode,
                can_encode: *encode,
                pixel_formats: vec!["yuv420p".to_string()],
                properties: HashMap::new(),
            });
        }
        Arc::new(plugin)
    }

    #[test]
    fn test_registry_new() {
        let registry = PluginRegistry::empty();
        assert_eq!(registry.plugin_count(), 0);
        assert!(registry.list_plugins().is_empty());
        assert!(registry.list_codecs().is_empty());
    }

    #[test]
    fn test_register_plugin() {
        let registry = PluginRegistry::empty();
        let plugin = make_test_plugin("test-1", &[("h264", true, true)]);
        registry.register(plugin).expect("should register");
        assert_eq!(registry.plugin_count(), 1);
    }

    #[test]
    fn test_register_duplicate_rejected() {
        let registry = PluginRegistry::empty();
        let p1 = make_test_plugin("same-name", &[("h264", true, false)]);
        let p2 = make_test_plugin("same-name", &[("h265", true, false)]);
        registry.register(p1).expect("first should succeed");
        let err = registry.register(p2).expect_err("second should fail");
        assert!(err.to_string().contains("already registered"));
    }

    #[test]
    fn test_register_wrong_api_version() {
        let registry = PluginRegistry::empty();
        let info = CodecPluginInfo {
            name: "bad-api".to_string(),
            version: "1.0.0".to_string(),
            author: "Test".to_string(),
            description: "Bad API plugin".to_string(),
            api_version: 999,
            license: "MIT".to_string(),
            patent_encumbered: false,
        };
        let plugin = Arc::new(StaticPlugin::new(info));
        let err = registry.register(plugin).expect_err("should fail");
        assert!(err.to_string().contains("API"));
    }

    #[test]
    fn test_has_codec() {
        let registry = PluginRegistry::empty();
        let plugin = make_test_plugin("test", &[("h264", true, true), ("h265", true, false)]);
        registry.register(plugin).expect("should register");

        assert!(registry.has_codec("h264"));
        assert!(registry.has_codec("h265"));
        assert!(!registry.has_codec("vp9"));
    }

    #[test]
    fn test_has_decoder_encoder() {
        let registry = PluginRegistry::empty();
        let plugin = make_test_plugin("test", &[("h264", true, true), ("h265", true, false)]);
        registry.register(plugin).expect("should register");

        assert!(registry.has_decoder("h264"));
        assert!(registry.has_encoder("h264"));
        assert!(registry.has_decoder("h265"));
        assert!(!registry.has_encoder("h265"));
        assert!(!registry.has_decoder("nonexistent"));
    }

    #[test]
    fn test_list_plugins() {
        let registry = PluginRegistry::empty();
        let p1 = make_test_plugin("alpha", &[("h264", true, false)]);
        let p2 = make_test_plugin("beta", &[("h265", true, false)]);
        registry.register(p1).expect("should register alpha");
        registry.register(p2).expect("should register beta");

        let list = registry.list_plugins();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].name, "alpha");
        assert_eq!(list[1].name, "beta");
    }

    #[test]
    fn test_list_codecs() {
        let registry = PluginRegistry::empty();
        let p1 = make_test_plugin("p1", &[("h264", true, true)]);
        let p2 = make_test_plugin("p2", &[("h265", true, false), ("aac", false, true)]);
        registry.register(p1).expect("should register");
        registry.register(p2).expect("should register");

        let codecs = registry.list_codecs();
        assert_eq!(codecs.len(), 3);
        let names: Vec<&str> = codecs.iter().map(|c| c.codec_name.as_str()).collect();
        assert!(names.contains(&"h264"));
        assert!(names.contains(&"h265"));
        assert!(names.contains(&"aac"));
    }

    #[test]
    fn test_find_decoder_not_found() {
        let registry = PluginRegistry::empty();
        let result = registry.find_decoder("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_find_encoder_not_found() {
        let registry = PluginRegistry::empty();
        let config = EncoderConfig::default();
        let result = registry.find_encoder("nonexistent", config);
        assert!(result.is_err());
    }

    #[test]
    fn test_clear() {
        let registry = PluginRegistry::empty();
        let plugin = make_test_plugin("test", &[("h264", true, false)]);
        registry.register(plugin).expect("should register");
        assert_eq!(registry.plugin_count(), 1);
        registry.clear();
        assert_eq!(registry.plugin_count(), 0);
    }

    #[test]
    fn test_find_plugin_for_codec() {
        let registry = PluginRegistry::empty();
        let plugin = make_test_plugin("h264-provider", &[("h264", true, true)]);
        registry.register(plugin).expect("should register");

        let found = registry.find_plugin_for_codec("h264");
        assert!(found.is_some());
        assert_eq!(
            found.as_ref().map(|i| i.name.as_str()),
            Some("h264-provider")
        );

        let not_found = registry.find_plugin_for_codec("aac");
        assert!(not_found.is_none());
    }

    #[test]
    fn test_multiple_plugins_first_wins() {
        let registry = PluginRegistry::empty();
        // Both provide h264 decode, but first registered wins
        let p1 = make_test_plugin("first", &[("h264", true, false)]);
        let p2 = make_test_plugin("second", &[("h264", true, true)]);
        registry.register(p1).expect("should register first");
        registry.register(p2).expect("should register second");

        let found = registry.find_plugin_for_codec("h264");
        assert_eq!(found.as_ref().map(|i| i.name.as_str()), Some("first"));
    }

    #[test]
    fn test_default_search_paths() {
        let paths = PluginRegistry::default_search_paths();
        // Should at least have the home directory path
        assert!(!paths.is_empty());
    }

    #[test]
    fn test_add_search_path() {
        let mut registry = PluginRegistry::empty();
        let path = PathBuf::from("/tmp/test-plugins");
        registry.add_search_path(path.clone());
        assert!(registry.search_paths().contains(&path));

        // Adding same path again should not duplicate
        registry.add_search_path(path.clone());
        assert_eq!(
            registry
                .search_paths()
                .iter()
                .filter(|p| **p == path)
                .count(),
            1
        );
    }

    #[test]
    fn test_load_plugin_without_dynamic_loading() {
        let registry = PluginRegistry::empty();
        let result = registry.load_plugin(Path::new("/nonexistent.so"));

        #[cfg(not(feature = "dynamic-loading"))]
        {
            assert!(result.is_err());
            assert!(result
                .unwrap_err()
                .to_string()
                .contains("Dynamic loading not enabled"));
        }

        #[cfg(feature = "dynamic-loading")]
        {
            // With dynamic loading, it will fail trying to load the file
            assert!(result.is_err());
        }
    }
}
