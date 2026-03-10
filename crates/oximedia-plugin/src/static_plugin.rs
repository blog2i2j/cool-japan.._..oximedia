//! Static plugin implementation.
//!
//! Provides [`StaticPlugin`], a convenience type for creating plugins
//! without shared library loading. This is useful for:
//!
//! - Testing and development
//! - Embedding plugins directly in the application binary
//! - Creating adapter plugins for existing codec implementations
//!
//! Also provides the [`declare_plugin!`] macro for creating shared
//! library entry points.

use crate::traits::{CodecPlugin, CodecPluginInfo, PluginCapability};
use oximedia_codec::{CodecError, CodecResult, EncoderConfig, VideoDecoder, VideoEncoder};

/// A static plugin that wraps decoder/encoder factory functions.
///
/// Use the builder pattern to construct a plugin with custom factories:
///
/// ```rust
/// use oximedia_plugin::{StaticPlugin, CodecPluginInfo, PluginCapability, PLUGIN_API_VERSION};
/// use std::collections::HashMap;
///
/// let info = CodecPluginInfo {
///     name: "my-plugin".to_string(),
///     version: "1.0.0".to_string(),
///     author: "Me".to_string(),
///     description: "My custom plugin".to_string(),
///     api_version: PLUGIN_API_VERSION,
///     license: "MIT".to_string(),
///     patent_encumbered: false,
/// };
///
/// let plugin = StaticPlugin::new(info)
///     .add_capability(PluginCapability {
///         codec_name: "custom-codec".to_string(),
///         can_decode: true,
///         can_encode: false,
///         pixel_formats: vec!["yuv420p".to_string()],
///         properties: HashMap::new(),
///     });
/// ```
pub struct StaticPlugin {
    info: CodecPluginInfo,
    capabilities: Vec<PluginCapability>,
    decoder_factory: Option<Box<dyn Fn(&str) -> CodecResult<Box<dyn VideoDecoder>> + Send + Sync>>,
    encoder_factory: Option<
        Box<dyn Fn(&str, EncoderConfig) -> CodecResult<Box<dyn VideoEncoder>> + Send + Sync>,
    >,
}

impl StaticPlugin {
    /// Create a new static plugin with the given metadata.
    ///
    /// The plugin starts with no capabilities and no factories.
    /// Use [`add_capability`](Self::add_capability),
    /// [`with_decoder`](Self::with_decoder), and
    /// [`with_encoder`](Self::with_encoder) to configure it.
    #[must_use]
    pub fn new(info: CodecPluginInfo) -> Self {
        Self {
            info,
            capabilities: Vec::new(),
            decoder_factory: None,
            encoder_factory: None,
        }
    }

    /// Register a decoder factory function.
    ///
    /// The factory receives the codec name and should return a new
    /// decoder instance or an error if the codec is not supported.
    #[must_use]
    pub fn with_decoder<F>(mut self, factory: F) -> Self
    where
        F: Fn(&str) -> CodecResult<Box<dyn VideoDecoder>> + Send + Sync + 'static,
    {
        self.decoder_factory = Some(Box::new(factory));
        self
    }

    /// Register an encoder factory function.
    ///
    /// The factory receives the codec name and encoder configuration,
    /// and should return a new encoder instance or an error.
    #[must_use]
    pub fn with_encoder<F>(mut self, factory: F) -> Self
    where
        F: Fn(&str, EncoderConfig) -> CodecResult<Box<dyn VideoEncoder>> + Send + Sync + 'static,
    {
        self.encoder_factory = Some(Box::new(factory));
        self
    }

    /// Add a codec capability to this plugin.
    #[must_use]
    pub fn add_capability(mut self, cap: PluginCapability) -> Self {
        self.capabilities.push(cap);
        self
    }
}

impl CodecPlugin for StaticPlugin {
    fn info(&self) -> CodecPluginInfo {
        self.info.clone()
    }

    fn capabilities(&self) -> Vec<PluginCapability> {
        self.capabilities.clone()
    }

    fn create_decoder(&self, codec_name: &str) -> CodecResult<Box<dyn VideoDecoder>> {
        match &self.decoder_factory {
            Some(factory) => factory(codec_name),
            None => Err(CodecError::UnsupportedFeature(format!(
                "No decoder factory registered for '{codec_name}'"
            ))),
        }
    }

    fn create_encoder(
        &self,
        codec_name: &str,
        config: EncoderConfig,
    ) -> CodecResult<Box<dyn VideoEncoder>> {
        match &self.encoder_factory {
            Some(factory) => factory(codec_name, config),
            None => Err(CodecError::UnsupportedFeature(format!(
                "No encoder factory registered for '{codec_name}'"
            ))),
        }
    }
}

/// Macro for defining a plugin entry point in a shared library.
///
/// This macro generates the two required `extern "C"` functions
/// that the host uses to load the plugin:
///
/// - `oximedia_plugin_api_version() -> u32` - returns the API version
/// - `oximedia_plugin_create() -> *mut dyn CodecPlugin` - creates the plugin
///
/// # Usage
///
/// In your plugin crate's `lib.rs`:
///
/// ```rust,ignore
/// use oximedia_plugin::{CodecPlugin, CodecPluginInfo};
///
/// struct MyPlugin;
///
/// impl CodecPlugin for MyPlugin {
///     // ... implement trait methods
/// }
///
/// fn create_my_plugin() -> MyPlugin {
///     MyPlugin
/// }
///
/// oximedia_plugin::declare_plugin!(MyPlugin, create_my_plugin);
/// ```
///
/// # Safety
///
/// The generated functions use `unsafe extern "C"` ABI. The create
/// function allocates the plugin on the heap and returns a raw pointer.
/// The host is responsible for taking ownership (via `Arc::from_raw`).
#[macro_export]
macro_rules! declare_plugin {
    ($plugin_type:ty, $create_fn:ident) => {
        /// Return the plugin API version for compatibility checking.
        ///
        /// # Safety
        ///
        /// This function is called by the host through FFI.
        #[no_mangle]
        pub unsafe extern "C" fn oximedia_plugin_api_version() -> u32 {
            $crate::PLUGIN_API_VERSION
        }

        /// Create a new plugin instance.
        ///
        /// # Safety
        ///
        /// This function is called by the host through FFI.
        /// The returned pointer must be passed to `Arc::from_raw` by the caller.
        #[no_mangle]
        pub unsafe extern "C" fn oximedia_plugin_create() -> *mut dyn $crate::CodecPlugin {
            let plugin = $create_fn();
            let boxed: Box<dyn $crate::CodecPlugin> = Box::new(plugin);
            Box::into_raw(boxed)
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::PLUGIN_API_VERSION;
    use std::collections::HashMap;

    fn make_test_info(name: &str) -> CodecPluginInfo {
        CodecPluginInfo {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            author: "Test".to_string(),
            description: "Test plugin".to_string(),
            api_version: PLUGIN_API_VERSION,
            license: "MIT".to_string(),
            patent_encumbered: false,
        }
    }

    #[test]
    fn test_static_plugin_info() {
        let plugin = StaticPlugin::new(make_test_info("my-plugin"));
        let info = plugin.info();
        assert_eq!(info.name, "my-plugin");
        assert_eq!(info.version, "1.0.0");
        assert_eq!(info.api_version, PLUGIN_API_VERSION);
    }

    #[test]
    fn test_static_plugin_no_capabilities() {
        let plugin = StaticPlugin::new(make_test_info("empty"));
        assert!(plugin.capabilities().is_empty());
        assert!(!plugin.supports_codec("h264"));
    }

    #[test]
    fn test_static_plugin_add_capability() {
        let plugin = StaticPlugin::new(make_test_info("cap-test"))
            .add_capability(PluginCapability {
                codec_name: "h264".to_string(),
                can_decode: true,
                can_encode: false,
                pixel_formats: vec!["yuv420p".to_string()],
                properties: HashMap::new(),
            })
            .add_capability(PluginCapability {
                codec_name: "h265".to_string(),
                can_decode: true,
                can_encode: true,
                pixel_formats: vec!["yuv420p".to_string(), "nv12".to_string()],
                properties: HashMap::new(),
            });

        assert_eq!(plugin.capabilities().len(), 2);
        assert!(plugin.supports_codec("h264"));
        assert!(plugin.supports_codec("h265"));
        assert!(plugin.can_decode("h264"));
        assert!(!plugin.can_encode("h264"));
        assert!(plugin.can_decode("h265"));
        assert!(plugin.can_encode("h265"));
    }

    #[test]
    fn test_static_plugin_no_decoder_factory() {
        let plugin = StaticPlugin::new(make_test_info("no-factory"));
        let result = plugin.create_decoder("h264");
        assert!(result.is_err());
    }

    #[test]
    fn test_static_plugin_no_encoder_factory() {
        let plugin = StaticPlugin::new(make_test_info("no-factory"));
        let config = EncoderConfig::default();
        let result = plugin.create_encoder("h264", config);
        assert!(result.is_err());
    }

    #[test]
    fn test_static_plugin_with_decoder_factory() {
        let plugin = StaticPlugin::new(make_test_info("factory-test"))
            .add_capability(PluginCapability {
                codec_name: "test".to_string(),
                can_decode: true,
                can_encode: false,
                pixel_formats: vec![],
                properties: HashMap::new(),
            })
            .with_decoder(|codec_name| {
                // Return an error to test the factory is called correctly
                Err(CodecError::UnsupportedFeature(format!(
                    "Mock decoder for '{codec_name}' - factory was called"
                )))
            });

        let result = plugin.create_decoder("test");
        match result {
            Err(e) => {
                let err_msg = e.to_string();
                assert!(err_msg.contains("Mock decoder for 'test'"));
                assert!(err_msg.contains("factory was called"));
            }
            Ok(_) => panic!("Expected error from mock decoder factory"),
        }
    }

    #[test]
    fn test_static_plugin_with_encoder_factory() {
        let plugin = StaticPlugin::new(make_test_info("enc-factory"))
            .add_capability(PluginCapability {
                codec_name: "test".to_string(),
                can_decode: false,
                can_encode: true,
                pixel_formats: vec![],
                properties: HashMap::new(),
            })
            .with_encoder(|codec_name, _config| {
                Err(CodecError::UnsupportedFeature(format!(
                    "Mock encoder for '{codec_name}'"
                )))
            });

        let config = EncoderConfig::default();
        let result = plugin.create_encoder("test", config);
        match result {
            Err(e) => {
                assert!(e.to_string().contains("Mock encoder"));
            }
            Ok(_) => panic!("Expected error from mock encoder factory"),
        }
    }

    #[test]
    fn test_codec_plugin_trait_default_methods() {
        let plugin = StaticPlugin::new(make_test_info("defaults"))
            .add_capability(PluginCapability {
                codec_name: "codec-a".to_string(),
                can_decode: true,
                can_encode: true,
                pixel_formats: vec![],
                properties: HashMap::new(),
            })
            .add_capability(PluginCapability {
                codec_name: "codec-b".to_string(),
                can_decode: true,
                can_encode: false,
                pixel_formats: vec![],
                properties: HashMap::new(),
            });

        // supports_codec
        assert!(plugin.supports_codec("codec-a"));
        assert!(plugin.supports_codec("codec-b"));
        assert!(!plugin.supports_codec("codec-c"));

        // can_decode
        assert!(plugin.can_decode("codec-a"));
        assert!(plugin.can_decode("codec-b"));
        assert!(!plugin.can_decode("codec-c"));

        // can_encode
        assert!(plugin.can_encode("codec-a"));
        assert!(!plugin.can_encode("codec-b"));
        assert!(!plugin.can_encode("codec-c"));
    }
}
