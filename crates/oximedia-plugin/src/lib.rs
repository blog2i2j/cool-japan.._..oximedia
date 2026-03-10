//! Plugin system for OxiMedia.
//!
//! Enables dynamic loading of external codec implementations,
//! allowing third-party or patent-encumbered codecs to be used
//! without bundling them in the core library.
//!
//! # Architecture
//!
//! Plugins implement the [`CodecPlugin`] trait and are loaded from
//! shared libraries (.so/.dylib/.dll) at runtime. Each plugin
//! declares its capabilities (which codecs it provides, whether
//! it can decode/encode), and the central [`PluginRegistry`] manages
//! discovery, loading, and codec lookup.
//!
//! # Feature Gates
//!
//! - `dynamic-loading`: Enables loading plugins from shared libraries
//!   (requires libloading). Without this feature, only static plugin
//!   registration is available.
//!
//! # Static Plugins
//!
//! Even without dynamic loading, you can register plugins statically
//! using [`StaticPlugin`] and the builder pattern:
//!
//! ```rust
//! use oximedia_plugin::{StaticPlugin, CodecPluginInfo, PluginCapability, PluginRegistry};
//! use std::sync::Arc;
//! use std::collections::HashMap;
//!
//! let info = CodecPluginInfo {
//!     name: "my-plugin".to_string(),
//!     version: "1.0.0".to_string(),
//!     author: "Test".to_string(),
//!     description: "A test plugin".to_string(),
//!     api_version: oximedia_plugin::PLUGIN_API_VERSION,
//!     license: "MIT".to_string(),
//!     patent_encumbered: false,
//! };
//!
//! let plugin = StaticPlugin::new(info)
//!     .add_capability(PluginCapability {
//!         codec_name: "test-codec".to_string(),
//!         can_decode: true,
//!         can_encode: false,
//!         pixel_formats: vec!["yuv420p".to_string()],
//!         properties: HashMap::new(),
//!     });
//!
//! let registry = PluginRegistry::new();
//! registry.register(Arc::new(plugin)).expect("registration should succeed");
//! assert_eq!(registry.plugin_count(), 1);
//! ```
//!
//! # Dynamic Plugins (feature = "dynamic-loading")
//!
//! With the `dynamic-loading` feature, plugins can be loaded from
//! shared libraries. The shared library must export two symbols:
//!
//! - `oximedia_plugin_api_version() -> u32`
//! - `oximedia_plugin_create() -> *mut dyn CodecPlugin`
//!
//! Use the [`declare_plugin!`] macro to generate these exports.

pub mod error;
pub mod manifest;
pub mod registry;
pub mod static_plugin;
pub mod traits;

#[cfg(feature = "dynamic-loading")]
pub mod loader;

pub use error::{PluginError, PluginResult};
pub use manifest::{
    resolve_dependencies, DependencyResolution, ManifestCodec, PluginManifest, SemVer, SemVerOp,
    SemVerReq,
};
pub use registry::PluginRegistry;
pub use static_plugin::StaticPlugin;
pub use traits::{CodecPlugin, CodecPluginInfo, PluginCapability, PLUGIN_API_VERSION};
