//! Lazy (deferred) plugin initialization.
//!
//! [`LazyPlugin`] wraps a factory closure and defers actual plugin construction
//! until the first call to [`get_or_init`](LazyPlugin::get_or_init).  This
//! avoids paying the initialization cost for plugins that may never be used.
//!
//! # Example
//!
//! ```rust
//! use oximedia_plugin::lazy::LazyPlugin;
//! use oximedia_plugin::{StaticPlugin, CodecPluginInfo, PLUGIN_API_VERSION};
//!
//! let lazy = LazyPlugin::new("my-plugin".to_string(), || {
//!     let info = CodecPluginInfo {
//!         name: "my-plugin".to_string(),
//!         version: "1.0.0".to_string(),
//!         author: "Test".to_string(),
//!         description: "Lazy plugin".to_string(),
//!         api_version: PLUGIN_API_VERSION,
//!         license: "MIT".to_string(),
//!         patent_encumbered: false,
//!     };
//!     Box::new(StaticPlugin::new(info)) as Box<dyn oximedia_plugin::CodecPlugin>
//! });
//!
//! assert!(!lazy.is_initialized());
//! let _plugin = lazy.get_or_init();
//! assert!(lazy.is_initialized());
//! ```

use crate::traits::CodecPlugin;
use oximedia_codec::{CodecResult, EncoderConfig, VideoDecoder, VideoEncoder};
use std::sync::OnceLock;

// ── LazyPlugin ────────────────────────────────────────────────────────────────

/// A plugin entry that defers construction until the first codec access.
///
/// `LazyPlugin` implements [`CodecPlugin`] by delegating all calls to the
/// inner plugin after initializing it on first use.  Initialization is
/// thread-safe: concurrent calls to [`get_or_init`](Self::get_or_init)
/// are guaranteed to produce exactly one initialization.
///
/// # Type Parameters
///
/// - `F`: A factory function `Fn() -> Box<dyn CodecPlugin>`.  It must be
///   `Send + Sync` so the `LazyPlugin` itself satisfies those bounds.
pub struct LazyPlugin<F>
where
    F: Fn() -> Box<dyn CodecPlugin> + Send + Sync,
{
    /// The logical name of this plugin (used for diagnostics and lookup).
    name: String,
    /// Factory that produces the underlying plugin on first access.
    factory: F,
    /// Holds the initialized plugin after the first call to `get_or_init`.
    inner: OnceLock<Box<dyn CodecPlugin>>,
}

impl<F> LazyPlugin<F>
where
    F: Fn() -> Box<dyn CodecPlugin> + Send + Sync,
{
    /// Create a new lazy plugin with the given name and factory.
    ///
    /// The factory is not called until [`get_or_init`](Self::get_or_init) is
    /// first invoked.
    pub fn new(name: String, factory: F) -> Self {
        Self {
            name,
            factory,
            inner: OnceLock::new(),
        }
    }

    /// Return a reference to the inner plugin, initializing it if needed.
    ///
    /// Thread-safe: the factory is called at most once, even under concurrent
    /// access.
    pub fn get_or_init(&self) -> &dyn CodecPlugin {
        self.inner
            .get_or_init(|| (self.factory)())
            .as_ref()
    }

    /// Return `true` if the inner plugin has already been initialized.
    pub fn is_initialized(&self) -> bool {
        self.inner.get().is_some()
    }

    /// Return the name of this lazy plugin entry.
    pub fn name(&self) -> &str {
        &self.name
    }
}

// ── CodecPlugin impl ──────────────────────────────────────────────────────────

impl<F> CodecPlugin for LazyPlugin<F>
where
    F: Fn() -> Box<dyn CodecPlugin> + Send + Sync,
{
    fn info(&self) -> crate::traits::CodecPluginInfo {
        self.get_or_init().info()
    }

    fn capabilities(&self) -> Vec<crate::traits::PluginCapability> {
        self.get_or_init().capabilities()
    }

    fn create_decoder(&self, codec_name: &str) -> CodecResult<Box<dyn VideoDecoder>> {
        self.get_or_init().create_decoder(codec_name)
    }

    fn create_encoder(
        &self,
        codec_name: &str,
        config: EncoderConfig,
    ) -> CodecResult<Box<dyn VideoEncoder>> {
        self.get_or_init().create_encoder(codec_name, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::static_plugin::StaticPlugin;
    use crate::traits::{CodecPluginInfo, PluginCapability, PLUGIN_API_VERSION};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn make_factory_counted(
        name: &str,
        codec: &str,
        counter: Arc<AtomicUsize>,
    ) -> impl Fn() -> Box<dyn CodecPlugin> + Send + Sync {
        let name = name.to_string();
        let codec = codec.to_string();
        move || {
            counter.fetch_add(1, Ordering::SeqCst);
            let info = CodecPluginInfo {
                name: name.clone(),
                version: "1.0.0".to_string(),
                author: "Test".to_string(),
                description: "Counted lazy plugin".to_string(),
                api_version: PLUGIN_API_VERSION,
                license: "MIT".to_string(),
                patent_encumbered: false,
            };
            let plugin = StaticPlugin::new(info).add_capability(PluginCapability {
                codec_name: codec.clone(),
                can_decode: true,
                can_encode: false,
                pixel_formats: vec![],
                properties: HashMap::new(),
            });
            Box::new(plugin)
        }
    }

    // 1. Not initialized before first access
    #[test]
    fn test_lazy_not_initialized_before_access() {
        let counter = Arc::new(AtomicUsize::new(0));
        let lazy = LazyPlugin::new(
            "test".to_string(),
            make_factory_counted("test", "h264", Arc::clone(&counter)),
        );
        assert!(!lazy.is_initialized());
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    // 2. Initialized after get_or_init
    #[test]
    fn test_lazy_initialized_after_get() {
        let counter = Arc::new(AtomicUsize::new(0));
        let lazy = LazyPlugin::new(
            "test".to_string(),
            make_factory_counted("test", "h264", Arc::clone(&counter)),
        );
        let _plugin = lazy.get_or_init();
        assert!(lazy.is_initialized());
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    // 3. Multiple get_or_init calls return same instance (factory called once)
    #[test]
    fn test_lazy_factory_called_once() {
        let counter = Arc::new(AtomicUsize::new(0));
        let lazy = LazyPlugin::new(
            "test".to_string(),
            make_factory_counted("test", "h264", Arc::clone(&counter)),
        );
        let _p1 = lazy.get_or_init();
        let _p2 = lazy.get_or_init();
        let _p3 = lazy.get_or_init();
        assert_eq!(counter.load(Ordering::SeqCst), 1, "factory must be called exactly once");
    }

    // 4. CodecPlugin trait methods delegate to inner plugin
    #[test]
    fn test_lazy_codec_plugin_delegation() {
        let counter = Arc::new(AtomicUsize::new(0));
        let lazy = LazyPlugin::new(
            "delegating".to_string(),
            make_factory_counted("delegating", "opus", Arc::clone(&counter)),
        );
        // capabilities triggers initialization
        let caps = lazy.capabilities();
        assert_eq!(caps.len(), 1);
        assert_eq!(caps[0].codec_name, "opus");
        assert!(lazy.is_initialized());
    }

    // 5. name() returns the correct name
    #[test]
    fn test_lazy_name() {
        let counter = Arc::new(AtomicUsize::new(0));
        let lazy = LazyPlugin::new(
            "my-lazy-plugin".to_string(),
            make_factory_counted("my-lazy-plugin", "vp9", counter),
        );
        assert_eq!(lazy.name(), "my-lazy-plugin");
    }
}
