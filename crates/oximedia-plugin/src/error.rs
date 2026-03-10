//! Plugin-specific error types.

use thiserror::Error;

/// Errors that can occur during plugin operations.
#[derive(Error, Debug)]
pub enum PluginError {
    /// The requested plugin was not found in the registry.
    #[error("Plugin not found: {0}")]
    NotFound(String),

    /// Failed to load a plugin from a shared library.
    #[error("Plugin load failed: {0}")]
    LoadFailed(String),

    /// Plugin API version does not match the host.
    #[error("Plugin version mismatch: expected {expected}, got {actual}")]
    VersionMismatch {
        /// The expected version string.
        expected: String,
        /// The actual version string found.
        actual: String,
    },

    /// Plugin initialization failed after loading.
    #[error("Plugin initialization failed: {0}")]
    InitFailed(String),

    /// The requested codec is not provided by any loaded plugin.
    #[error("Codec not provided by plugin: {0}")]
    CodecNotAvailable(String),

    /// The plugin's API version is incompatible with the host.
    #[error("Plugin API version incompatible: {0}")]
    ApiIncompatible(String),

    /// Dynamic loading was requested but the feature is not enabled.
    #[error("Dynamic loading not enabled (compile with 'dynamic-loading' feature)")]
    DynamicLoadingDisabled,

    /// An I/O error occurred during plugin operations.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A codec-level error propagated from the plugin.
    #[error("Codec error: {0}")]
    Codec(#[from] oximedia_codec::CodecError),

    /// The plugin manifest file is invalid or malformed.
    #[error("Invalid manifest: {0}")]
    InvalidManifest(String),

    /// A plugin with the same name is already registered.
    #[error("Plugin already registered: {0}")]
    AlreadyRegistered(String),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Result type alias for plugin operations.
pub type PluginResult<T> = Result<T, PluginError>;
