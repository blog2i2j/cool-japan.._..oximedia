//! Dynamic plugin loader.
//!
//! This module is only available when the `dynamic-loading` feature
//! is enabled. It uses `libloading` to load shared libraries and
//! extract the plugin entry points.
//!
//! # Security Considerations
//!
//! Loading a shared library executes arbitrary code. Only load
//! plugins from trusted sources. The loader validates the API
//! version before calling any other plugin functions, but this
//! provides compatibility checking, not security.
// Dynamic plugin loading fundamentally requires unsafe code to call into
// foreign shared libraries. The safety invariants are documented inline.
#![allow(unsafe_code)]

use crate::error::{PluginError, PluginResult};
use crate::traits::{CodecPlugin, PluginApiVersionFn, PluginCreateFn, PLUGIN_API_VERSION};
use libloading::{Library, Symbol};
use std::path::Path;
use std::sync::Arc;

/// A plugin loaded from a shared library.
///
/// This struct keeps the `Library` handle alive for as long as the
/// plugin instance exists. Dropping this struct will unload the
/// shared library (after the plugin Arc's refcount reaches zero).
pub struct LoadedPlugin {
    /// The loaded library handle. Must be kept alive as long as
    /// the plugin is in use.
    _library: Library,
    /// The plugin instance created from the library.
    plugin: Arc<dyn CodecPlugin>,
}

impl std::fmt::Debug for LoadedPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedPlugin")
            .field("plugin", &"<dyn CodecPlugin>")
            .finish_non_exhaustive()
    }
}

impl LoadedPlugin {
    /// Load a plugin from a shared library file.
    ///
    /// The loader performs the following steps:
    /// 1. Opens the shared library
    /// 2. Looks up `oximedia_plugin_api_version` and validates it
    /// 3. Looks up `oximedia_plugin_create` and calls it
    /// 4. Takes ownership of the returned plugin pointer
    ///
    /// # Errors
    ///
    /// - [`PluginError::LoadFailed`] if the library cannot be opened
    ///   or required symbols are missing
    /// - [`PluginError::ApiIncompatible`] if the API version does not match
    ///
    /// # Safety
    ///
    /// This function loads and executes code from an external shared library.
    /// The caller must ensure the library is from a trusted source.
    pub fn load(path: &Path) -> PluginResult<Self> {
        // Safety: Loading external code is inherently unsafe.
        // We validate API version before calling any other functions.
        let library = unsafe {
            Library::new(path).map_err(|e| {
                PluginError::LoadFailed(format!("Failed to open '{}': {e}", path.display()))
            })?
        };

        // Check API version first (least risky call)
        let api_version = {
            let api_version_fn: Symbol<PluginApiVersionFn> = unsafe {
                library.get(b"oximedia_plugin_api_version").map_err(|e| {
                    PluginError::LoadFailed(format!(
                        "Missing 'oximedia_plugin_api_version' symbol in '{}': {e}",
                        path.display()
                    ))
                })?
            };
            unsafe { api_version_fn() }
        };

        if api_version != PLUGIN_API_VERSION {
            return Err(PluginError::ApiIncompatible(format!(
                "Plugin '{}' has API v{api_version}, host expects v{PLUGIN_API_VERSION}",
                path.display()
            )));
        }

        // Create the plugin instance
        let plugin = {
            let create_fn: Symbol<PluginCreateFn> = unsafe {
                library.get(b"oximedia_plugin_create").map_err(|e| {
                    PluginError::LoadFailed(format!(
                        "Missing 'oximedia_plugin_create' symbol in '{}': {e}",
                        path.display()
                    ))
                })?
            };

            let raw_plugin = unsafe { create_fn() };
            if raw_plugin.is_null() {
                return Err(PluginError::InitFailed(format!(
                    "Plugin create function returned null for '{}'",
                    path.display()
                )));
            }

            // Safety: The plugin was created by Box::into_raw in the shared library.
            // We take ownership here. The raw pointer came from PluginCreateFn which
            // is documented to return a Box::into_raw'd pointer.
            unsafe { Arc::from_raw(raw_plugin) }
        };

        // Log plugin info
        let info = plugin.info();
        tracing::info!(
            "Loaded plugin from '{}': {} v{} [{}]{}",
            path.display(),
            info.name,
            info.version,
            info.license,
            if info.patent_encumbered {
                " (patent-encumbered)"
            } else {
                ""
            }
        );

        if info.patent_encumbered {
            tracing::warn!(
                "Plugin '{}' contains patent-encumbered codecs. \
                 Ensure you have appropriate licenses before use.",
                info.name
            );
        }

        let caps = plugin.capabilities();
        tracing::debug!(
            "Plugin '{}' provides {} codec(s): {}",
            info.name,
            caps.len(),
            caps.iter()
                .map(|c| c.codec_name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );

        Ok(Self {
            _library: library,
            plugin,
        })
    }

    /// Get a reference to the loaded plugin.
    pub fn plugin(&self) -> &Arc<dyn CodecPlugin> {
        &self.plugin
    }

    /// Consume this `LoadedPlugin` and return the plugin Arc.
    ///
    /// Note: The caller must ensure the `Library` is kept alive
    /// for as long as the plugin is used. In practice, this is
    /// typically called only by the registry which manages lifetimes.
    pub fn into_plugin(self) -> Arc<dyn CodecPlugin> {
        // We intentionally leak the library handle here because the
        // plugin code references it. The registry owns the plugin Arc
        // and when it drops, the library symbols may still be needed
        // during drop. Leaking is safe and prevents use-after-free.
        let library = self._library;
        std::mem::forget(library);
        self.plugin
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_nonexistent_library() {
        let result = LoadedPlugin::load(Path::new("/nonexistent/plugin.so"));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, PluginError::LoadFailed(_)));
    }

    #[test]
    fn test_load_invalid_library() {
        // Create a temp file that is not a valid shared library
        let dir = std::env::temp_dir().join("oximedia-plugin-test-loader");
        std::fs::create_dir_all(&dir).expect("dir creation should succeed");
        let fake_lib = dir.join("fake_plugin.so");
        std::fs::write(&fake_lib, b"not a real library").expect("write should succeed");

        let result = LoadedPlugin::load(&fake_lib);
        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
