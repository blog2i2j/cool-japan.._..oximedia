//! Central plugin registry with priority ordering and capability caching.
//!
//! The [`PluginRegistry`] is the main entry point for managing plugins.
//! It handles registration, discovery, and codec lookup across all
//! loaded plugins.
//!
//! # Priority
//!
//! Each plugin can be assigned a numeric priority (higher = preferred).
//! When multiple plugins provide the same codec, the one with the highest
//! priority wins.  Plugins with equal priority are ranked by registration
//! order (first-registered wins among equals).
//!
//! # Capability Cache
//!
//! The registry maintains an internal cache of codec → plugin index
//! mappings to avoid O(n) scans on every lookup.  The cache is invalidated
//! atomically on every `register` or `unregister` operation.

use crate::error::{PluginError, PluginResult};
use crate::traits::{CodecPlugin, CodecPluginInfo, PluginCapability, PLUGIN_API_VERSION};
use oximedia_codec::{CodecResult, EncoderConfig, VideoDecoder, VideoEncoder};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

// ── PluginEntry ───────────────────────────────────────────────────────────────

/// An entry in the plugin registry, bundling the plugin with its priority.
struct PluginEntry {
    plugin: Arc<dyn CodecPlugin>,
    /// Numeric priority: higher values are preferred over lower values.
    /// When two plugins have the same priority, the one registered first wins.
    priority: i32,
}

// ── CapabilityCache ───────────────────────────────────────────────────────────

/// Cached mapping from codec name to the index of the best plugin for that codec.
///
/// Two separate caches for decode and encode allow plugins that only support
/// one direction to serve correctly without confusion.
#[derive(Default)]
struct CapabilityCache {
    /// codec_name → index into the sorted plugins Vec for decode.
    decoder_index: HashMap<String, usize>,
    /// codec_name → index into the sorted plugins Vec for encode.
    encoder_index: HashMap<String, usize>,
}

impl CapabilityCache {
    fn new() -> Self {
        Self {
            decoder_index: HashMap::new(),
            encoder_index: HashMap::new(),
        }
    }

    fn invalidate(&mut self) {
        self.decoder_index.clear();
        self.encoder_index.clear();
    }

    /// Rebuild the cache from the current (already priority-sorted) plugin list.
    fn rebuild(&mut self, plugins: &[PluginEntry]) {
        self.invalidate();
        for (idx, entry) in plugins.iter().enumerate() {
            for cap in entry.plugin.capabilities() {
                if cap.can_decode {
                    self.decoder_index
                        .entry(cap.codec_name.clone())
                        .or_insert(idx);
                }
                if cap.can_encode {
                    self.encoder_index
                        .entry(cap.codec_name.clone())
                        .or_insert(idx);
                }
            }
        }
    }
}

// ── PluginRegistry ────────────────────────────────────────────────────────────

/// Central registry for all loaded codec plugins.
///
/// The registry maintains a list of registered plugins ordered by priority
/// (descending) and provides methods to discover codecs, create
/// decoders/encoders, and manage the plugin lifecycle.
///
/// # Thread Safety
///
/// The registry uses interior mutability (`RwLock`) so it can be
/// shared across threads. Multiple readers can query the registry
/// concurrently; writes (registration) acquire an exclusive lock.
///
/// # Priority
///
/// Higher `priority` values take precedence when multiple plugins provide
/// the same codec.  Use [`register_with_priority`](Self::register_with_priority)
/// to supply a custom priority (default is `0`).
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
    /// Plugins sorted by descending priority.
    plugins: RwLock<Vec<PluginEntry>>,
    /// Capability lookup cache — invalidated on every mutation.
    cache: RwLock<CapabilityCache>,
    search_paths: Vec<PathBuf>,
}

impl PluginRegistry {
    /// Create a new empty plugin registry with default search paths.
    #[must_use]
    pub fn new() -> Self {
        Self {
            plugins: RwLock::new(Vec::new()),
            cache: RwLock::new(CapabilityCache::new()),
            search_paths: Self::default_search_paths(),
        }
    }

    /// Create a registry with no search paths (for testing).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            plugins: RwLock::new(Vec::new()),
            cache: RwLock::new(CapabilityCache::new()),
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

    /// Register a static plugin instance with default priority (0).
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
        self.register_with_priority(plugin, 0)
    }

    /// Register a plugin with an explicit priority.
    ///
    /// Plugins with a higher `priority` value are preferred when multiple
    /// plugins provide the same codec.  Negative priorities are allowed.
    ///
    /// # Errors
    ///
    /// Same as [`register`](Self::register).
    pub fn register_with_priority(
        &self,
        plugin: Arc<dyn CodecPlugin>,
        priority: i32,
    ) -> PluginResult<()> {
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
            if existing.plugin.info().name == info.name {
                return Err(PluginError::AlreadyRegistered(info.name));
            }
        }

        tracing::info!(
            "Registered plugin: {} v{} (priority={}, {} codec(s))",
            info.name,
            info.version,
            priority,
            plugin.capabilities().len()
        );

        if info.patent_encumbered {
            tracing::warn!(
                "Plugin '{}' contains patent-encumbered codecs. Use at your own risk.",
                info.name
            );
        }

        plugins.push(PluginEntry { plugin, priority });

        // Re-sort by descending priority (stable sort preserves FIFO for equal priorities).
        plugins.sort_by(|a, b| b.priority.cmp(&a.priority));

        // Invalidate and rebuild the cache.
        drop(plugins);
        self.rebuild_cache()?;
        Ok(())
    }

    /// Unregister a plugin by name.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::NotFound`] if no plugin with that name is registered.
    pub fn unregister(&self, name: &str) -> PluginResult<()> {
        let mut plugins = self
            .plugins
            .write()
            .map_err(|e| PluginError::InitFailed(format!("Lock poisoned: {e}")))?;

        let before = plugins.len();
        plugins.retain(|e| e.plugin.info().name != name);

        if plugins.len() == before {
            return Err(PluginError::NotFound(name.to_string()));
        }

        drop(plugins);
        self.rebuild_cache()?;
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

        plugins.last().map(|e| e.plugin.info()).ok_or_else(|| {
            PluginError::InitFailed("Plugin was not added after loading".to_string())
        })
    }

    /// List all registered plugins (in priority order, highest first).
    pub fn list_plugins(&self) -> Vec<CodecPluginInfo> {
        let plugins = match self.plugins.read() {
            Ok(p) => p,
            Err(_) => return Vec::new(),
        };
        plugins.iter().map(|e| e.plugin.info()).collect()
    }

    /// List all available codecs across all plugins.
    pub fn list_codecs(&self) -> Vec<PluginCapability> {
        let plugins = match self.plugins.read() {
            Ok(p) => p,
            Err(_) => return Vec::new(),
        };
        plugins
            .iter()
            .flat_map(|e| e.plugin.capabilities())
            .collect()
    }

    /// Find and create a decoder for a given codec name.
    ///
    /// Searches all registered plugins (in priority order) for one that
    /// can decode the requested codec, and creates a new decoder instance.
    ///
    /// Uses the capability cache for O(1) plugin lookup.
    ///
    /// # Errors
    ///
    /// Returns error if no plugin supports decoding the given codec.
    pub fn find_decoder(&self, codec_name: &str) -> CodecResult<Box<dyn VideoDecoder>> {
        // Try the fast cache path first.
        if let Some(plugin) = self.cached_decoder_plugin(codec_name) {
            return plugin.create_decoder(codec_name);
        }

        // Fall back to a linear scan (cache miss or rebuild needed).
        let plugins = self
            .plugins
            .read()
            .map_err(|e| oximedia_codec::CodecError::Internal(format!("Lock poisoned: {e}")))?;

        for entry in plugins.iter() {
            if entry.plugin.can_decode(codec_name) {
                return entry.plugin.create_decoder(codec_name);
            }
        }

        Err(oximedia_codec::CodecError::UnsupportedFeature(format!(
            "No plugin provides decoder for '{codec_name}'"
        )))
    }

    /// Find and create an encoder for a given codec name.
    ///
    /// Searches all registered plugins (in priority order) for one that
    /// can encode the requested codec, and creates a new encoder instance.
    ///
    /// Uses the capability cache for O(1) plugin lookup.
    ///
    /// # Errors
    ///
    /// Returns error if no plugin supports encoding the given codec.
    pub fn find_encoder(
        &self,
        codec_name: &str,
        config: EncoderConfig,
    ) -> CodecResult<Box<dyn VideoEncoder>> {
        // Try the fast cache path first.
        if let Some(plugin) = self.cached_encoder_plugin(codec_name) {
            return plugin.create_encoder(codec_name, config);
        }

        // Fall back to a linear scan.
        let plugins = self
            .plugins
            .read()
            .map_err(|e| oximedia_codec::CodecError::Internal(format!("Lock poisoned: {e}")))?;

        for entry in plugins.iter() {
            if entry.plugin.can_encode(codec_name) {
                return entry.plugin.create_encoder(codec_name, config);
            }
        }

        Err(oximedia_codec::CodecError::UnsupportedFeature(format!(
            "No plugin provides encoder for '{codec_name}'"
        )))
    }

    /// Check if any plugin provides a given codec (decode or encode).
    pub fn has_codec(&self, codec_name: &str) -> bool {
        // Check cache first.
        if let Ok(cache) = self.cache.read() {
            if cache.decoder_index.contains_key(codec_name)
                || cache.encoder_index.contains_key(codec_name)
            {
                return true;
            }
        }
        let plugins = match self.plugins.read() {
            Ok(p) => p,
            Err(_) => return false,
        };
        plugins.iter().any(|e| e.plugin.supports_codec(codec_name))
    }

    /// Check if any plugin can decode a given codec.
    pub fn has_decoder(&self, codec_name: &str) -> bool {
        if let Ok(cache) = self.cache.read() {
            if cache.decoder_index.contains_key(codec_name) {
                return true;
            }
        }
        let plugins = match self.plugins.read() {
            Ok(p) => p,
            Err(_) => return false,
        };
        plugins.iter().any(|e| e.plugin.can_decode(codec_name))
    }

    /// Check if any plugin can encode a given codec.
    pub fn has_encoder(&self, codec_name: &str) -> bool {
        if let Ok(cache) = self.cache.read() {
            if cache.encoder_index.contains_key(codec_name) {
                return true;
            }
        }
        let plugins = match self.plugins.read() {
            Ok(p) => p,
            Err(_) => return false,
        };
        plugins.iter().any(|e| e.plugin.can_encode(codec_name))
    }

    /// Get the number of registered plugins.
    pub fn plugin_count(&self) -> usize {
        match self.plugins.read() {
            Ok(p) => p.len(),
            Err(_) => 0,
        }
    }

    /// Unload all registered plugins and clear the cache.
    pub fn clear(&self) {
        if let Ok(mut plugins) = self.plugins.write() {
            plugins.clear();
        }
        if let Ok(mut cache) = self.cache.write() {
            cache.invalidate();
        }
    }

    /// Find the plugin that provides a given codec (respects priority ordering).
    pub fn find_plugin_for_codec(&self, codec_name: &str) -> Option<CodecPluginInfo> {
        // Check cache for decoder first, then encoder.
        if let Some(plugin) = self.cached_decoder_plugin(codec_name) {
            return Some(plugin.info());
        }
        if let Some(plugin) = self.cached_encoder_plugin(codec_name) {
            return Some(plugin.info());
        }
        let plugins = self.plugins.read().ok()?;
        plugins
            .iter()
            .find(|e| e.plugin.supports_codec(codec_name))
            .map(|e| e.plugin.info())
    }

    /// Get the priority of a registered plugin by name.
    ///
    /// Returns `None` if no plugin with that name is registered.
    pub fn plugin_priority(&self, name: &str) -> Option<i32> {
        let plugins = self.plugins.read().ok()?;
        plugins
            .iter()
            .find(|e| e.plugin.info().name == name)
            .map(|e| e.priority)
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Rebuild the capability cache from the current sorted plugin list.
    fn rebuild_cache(&self) -> PluginResult<()> {
        let plugins = self
            .plugins
            .read()
            .map_err(|e| PluginError::InitFailed(format!("Lock poisoned: {e}")))?;
        let mut cache = self
            .cache
            .write()
            .map_err(|e| PluginError::InitFailed(format!("Cache lock poisoned: {e}")))?;
        cache.rebuild(&plugins);
        Ok(())
    }

    /// Return the plugin for the best decoder of `codec_name` using the cache.
    fn cached_decoder_plugin(&self, codec_name: &str) -> Option<Arc<dyn CodecPlugin>> {
        let cache = self.cache.read().ok()?;
        let idx = *cache.decoder_index.get(codec_name)?;
        drop(cache);
        let plugins = self.plugins.read().ok()?;
        plugins.get(idx).map(|e| Arc::clone(&e.plugin))
    }

    /// Return the plugin for the best encoder of `codec_name` using the cache.
    fn cached_encoder_plugin(&self, codec_name: &str) -> Option<Arc<dyn CodecPlugin>> {
        let cache = self.cache.read().ok()?;
        let idx = *cache.encoder_index.get(codec_name)?;
        drop(cache);
        let plugins = self.plugins.read().ok()?;
        plugins.get(idx).map(|e| Arc::clone(&e.plugin))
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
        // Both default priority 0: FIFO preserved (alpha first)
        let names: Vec<&str> = list.iter().map(|i| i.name.as_str()).collect();
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"beta"));
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
        // Cache should also be cleared.
        assert!(!registry.has_codec("h264"));
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
    fn test_multiple_plugins_first_wins_same_priority() {
        let registry = PluginRegistry::empty();
        // Both provide h264 decode, but first registered wins at equal priority
        let p1 = make_test_plugin("first", &[("h264", true, false)]);
        let p2 = make_test_plugin("second", &[("h264", true, true)]);
        registry.register(p1).expect("should register first");
        registry.register(p2).expect("should register second");

        let found = registry.find_plugin_for_codec("h264");
        assert_eq!(found.as_ref().map(|i| i.name.as_str()), Some("first"));
    }

    #[test]
    fn test_priority_ordering() {
        let registry = PluginRegistry::empty();
        // Register "low" first at priority 0, then "high" at priority 10.
        let low = make_test_plugin("low-priority", &[("h264", true, false)]);
        let high = make_test_plugin("high-priority", &[("h264", true, true)]);

        registry
            .register_with_priority(low, 0)
            .expect("register low");
        registry
            .register_with_priority(high, 10)
            .expect("register high");

        // High-priority plugin should win for h264.
        let found = registry.find_plugin_for_codec("h264");
        assert_eq!(
            found.as_ref().map(|i| i.name.as_str()),
            Some("high-priority")
        );
    }

    #[test]
    fn test_priority_negative() {
        let registry = PluginRegistry::empty();
        let normal = make_test_plugin("normal", &[("vp9", true, true)]);
        let fallback = make_test_plugin("fallback", &[("vp9", true, false)]);

        registry
            .register_with_priority(normal, 0)
            .expect("register normal");
        registry
            .register_with_priority(fallback, -5)
            .expect("register fallback");

        // Normal should win (higher priority).
        let found = registry.find_plugin_for_codec("vp9");
        assert_eq!(found.as_ref().map(|i| i.name.as_str()), Some("normal"));
    }

    #[test]
    fn test_priority_accessor() {
        let registry = PluginRegistry::empty();
        let p = make_test_plugin("prio-test", &[]);
        registry.register_with_priority(p, 42).expect("register");
        assert_eq!(registry.plugin_priority("prio-test"), Some(42));
        assert_eq!(registry.plugin_priority("nonexistent"), None);
    }

    #[test]
    fn test_unregister() {
        let registry = PluginRegistry::empty();
        let p = make_test_plugin("to-remove", &[("aac", true, false)]);
        registry.register(p).expect("register");
        assert_eq!(registry.plugin_count(), 1);
        assert!(registry.has_codec("aac"));

        registry.unregister("to-remove").expect("unregister");
        assert_eq!(registry.plugin_count(), 0);
        assert!(!registry.has_codec("aac"));
    }

    #[test]
    fn test_unregister_not_found() {
        let registry = PluginRegistry::empty();
        assert!(matches!(
            registry.unregister("ghost"),
            Err(PluginError::NotFound(_))
        ));
    }

    #[test]
    fn test_capability_cache_after_clear() {
        let registry = PluginRegistry::empty();
        let p = make_test_plugin("cached", &[("opus", true, true)]);
        registry.register(p).expect("register");
        assert!(registry.has_decoder("opus"));
        registry.clear();
        assert!(!registry.has_decoder("opus"));
        assert!(!registry.has_encoder("opus"));
    }

    #[test]
    fn test_cache_invalidated_on_unregister() {
        let registry = PluginRegistry::empty();
        let p1 = make_test_plugin("provider-a", &[("vorbis", true, false)]);
        let p2 = make_test_plugin("provider-b", &[("flac", true, true)]);
        registry.register(p1).expect("register a");
        registry.register(p2).expect("register b");

        assert!(registry.has_decoder("vorbis"));
        registry.unregister("provider-a").expect("unregister");
        assert!(!registry.has_decoder("vorbis"));
        assert!(registry.has_codec("flac")); // b still present
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
        let path = std::env::temp_dir().join("oximedia-plugin-registry-test-plugins");
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
