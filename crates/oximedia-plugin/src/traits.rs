//! Core plugin traits and types.
//!
//! This module defines the [`CodecPlugin`] trait that all plugins must
//! implement, along with supporting types for plugin metadata and
//! capability declaration.

use oximedia_codec::{CodecResult, EncoderConfig, VideoDecoder, VideoEncoder};
use std::collections::HashMap;

/// API version for plugin compatibility checking.
///
/// This value is incremented whenever the plugin ABI changes in a
/// backward-incompatible way. Plugins built against a different
/// API version will be rejected at load time.
pub const PLUGIN_API_VERSION: u32 = 1;

/// Metadata about a codec plugin.
///
/// This struct is returned by [`CodecPlugin::info`] and provides
/// identification, versioning, and licensing information.
#[derive(Debug, Clone)]
pub struct CodecPluginInfo {
    /// Plugin name (e.g., "oximedia-plugin-h264").
    pub name: String,
    /// Plugin version (semver string, e.g., "1.0.0").
    pub version: String,
    /// Plugin author or organization.
    pub author: String,
    /// Human-readable description of the plugin.
    pub description: String,
    /// API version this plugin was built against.
    pub api_version: u32,
    /// License identifier (e.g., "MIT", "GPL-2.0", "proprietary").
    pub license: String,
    /// Whether this plugin contains patent-encumbered codecs.
    ///
    /// When true, a warning is emitted at load time. Users must
    /// acknowledge the patent implications of using such plugins.
    pub patent_encumbered: bool,
}

impl std::fmt::Display for CodecPluginInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} v{} [{}]{}",
            self.name,
            self.version,
            self.license,
            if self.patent_encumbered {
                " (patent-encumbered)"
            } else {
                ""
            }
        )
    }
}

/// A single capability (codec) provided by a plugin.
///
/// Each plugin can provide multiple capabilities, one per codec
/// it supports. The capability declares whether decoding and/or
/// encoding is available, along with supported formats and properties.
#[derive(Debug, Clone)]
pub struct PluginCapability {
    /// Codec identifier string (e.g., "h264", "h265", "aac").
    pub codec_name: String,
    /// Whether decoding is supported for this codec.
    pub can_decode: bool,
    /// Whether encoding is supported for this codec.
    pub can_encode: bool,
    /// Supported pixel formats for video (e.g., "yuv420p", "nv12").
    pub pixel_formats: Vec<String>,
    /// Additional codec-specific properties (key-value pairs).
    pub properties: HashMap<String, String>,
}

impl std::fmt::Display for PluginCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mode = match (self.can_decode, self.can_encode) {
            (true, true) => "decode+encode",
            (true, false) => "decode",
            (false, true) => "encode",
            (false, false) => "none",
        };
        write!(f, "{} ({})", self.codec_name, mode)
    }
}

/// The main plugin trait that external codec libraries implement.
///
/// A plugin provides one or more codecs, each with optional decode
/// and encode support. The host application uses the registry to
/// discover plugins and create decoder/encoder instances on demand.
///
/// # Thread Safety
///
/// Plugins must be `Send + Sync` because the registry may be shared
/// across threads. Individual decoder/encoder instances returned by
/// the factory methods need only be `Send`.
///
/// # Implementing a Plugin
///
/// For simple cases, use [`StaticPlugin`](crate::StaticPlugin) with
/// the builder pattern. For shared libraries, implement this trait
/// on your own type and use the [`declare_plugin!`] macro.
pub trait CodecPlugin: Send + Sync {
    /// Get plugin metadata and identification.
    fn info(&self) -> CodecPluginInfo;

    /// List all capabilities (codecs) provided by this plugin.
    fn capabilities(&self) -> Vec<PluginCapability>;

    /// Create a decoder instance for the given codec name.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError`](oximedia_codec::CodecError) if the codec
    /// is not supported or decoder creation fails.
    fn create_decoder(&self, codec_name: &str) -> CodecResult<Box<dyn VideoDecoder>>;

    /// Create an encoder instance for the given codec name with configuration.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError`](oximedia_codec::CodecError) if the codec
    /// is not supported or encoder creation fails.
    fn create_encoder(
        &self,
        codec_name: &str,
        config: EncoderConfig,
    ) -> CodecResult<Box<dyn VideoEncoder>>;

    /// Check if this plugin supports a specific codec (decode or encode).
    fn supports_codec(&self, codec_name: &str) -> bool {
        self.capabilities()
            .iter()
            .any(|c| c.codec_name == codec_name)
    }

    /// Check if this plugin can decode a specific codec.
    fn can_decode(&self, codec_name: &str) -> bool {
        self.capabilities()
            .iter()
            .any(|c| c.codec_name == codec_name && c.can_decode)
    }

    /// Check if this plugin can encode a specific codec.
    fn can_encode(&self, codec_name: &str) -> bool {
        self.capabilities()
            .iter()
            .any(|c| c.codec_name == codec_name && c.can_encode)
    }
}

/// Function signature for the plugin API version query.
///
/// The shared library must export this as:
/// `extern "C" fn oximedia_plugin_api_version() -> u32`
pub type PluginApiVersionFn = unsafe extern "C" fn() -> u32;

/// Function signature for the plugin factory function.
///
/// The shared library must export this as:
/// `extern "C" fn oximedia_plugin_create() -> *mut dyn CodecPlugin`
///
/// The returned pointer must have been created via `Box::into_raw(Box::new(...))`.
/// The host takes ownership and will drop it when the plugin is unloaded.
///
/// Note: trait objects are not FFI-safe per Rust's type system, but this is
/// the standard pattern for Rust-to-Rust plugin loading where both sides
/// share the same trait definition and ABI. This is safe as long as both
/// the host and plugin are compiled with the same Rust compiler version.
#[allow(improper_ctypes_definitions)]
pub type PluginCreateFn = unsafe extern "C" fn() -> *mut dyn CodecPlugin;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_info_display() {
        let info = CodecPluginInfo {
            name: "test-plugin".to_string(),
            version: "1.0.0".to_string(),
            author: "Test Author".to_string(),
            description: "A test plugin".to_string(),
            api_version: PLUGIN_API_VERSION,
            license: "MIT".to_string(),
            patent_encumbered: false,
        };
        assert_eq!(format!("{info}"), "test-plugin v1.0.0 [MIT]");
    }

    #[test]
    fn test_plugin_info_display_patent() {
        let info = CodecPluginInfo {
            name: "h264-plugin".to_string(),
            version: "2.0.0".to_string(),
            author: "Corp".to_string(),
            description: "H.264 decoder".to_string(),
            api_version: PLUGIN_API_VERSION,
            license: "proprietary".to_string(),
            patent_encumbered: true,
        };
        assert_eq!(
            format!("{info}"),
            "h264-plugin v2.0.0 [proprietary] (patent-encumbered)"
        );
    }

    #[test]
    fn test_capability_display() {
        let cap = PluginCapability {
            codec_name: "h264".to_string(),
            can_decode: true,
            can_encode: true,
            pixel_formats: vec!["yuv420p".to_string()],
            properties: HashMap::new(),
        };
        assert_eq!(format!("{cap}"), "h264 (decode+encode)");

        let decode_only = PluginCapability {
            codec_name: "hevc".to_string(),
            can_decode: true,
            can_encode: false,
            pixel_formats: vec![],
            properties: HashMap::new(),
        };
        assert_eq!(format!("{decode_only}"), "hevc (decode)");

        let encode_only = PluginCapability {
            codec_name: "aac".to_string(),
            can_decode: false,
            can_encode: true,
            pixel_formats: vec![],
            properties: HashMap::new(),
        };
        assert_eq!(format!("{encode_only}"), "aac (encode)");
    }

    #[test]
    fn test_api_version_is_nonzero() {
        assert!(PLUGIN_API_VERSION > 0);
    }
}
