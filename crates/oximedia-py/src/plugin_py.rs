//! Python bindings for the `oximedia-plugin` system.
//!
//! Exposes plugin registry, plugin info, and codec capabilities to Python.

use pyo3::prelude::*;
use std::collections::HashMap;

use oximedia_plugin::{CodecPluginInfo, PluginCapability, PluginRegistry, PLUGIN_API_VERSION};

// ---------------------------------------------------------------------------
// PyPluginInfo
// ---------------------------------------------------------------------------

/// Metadata about a registered plugin.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyPluginInfo {
    /// Plugin name.
    #[pyo3(get)]
    pub name: String,
    /// Plugin version string (semver).
    #[pyo3(get)]
    pub version: String,
    /// Plugin author or organisation.
    #[pyo3(get)]
    pub author: String,
    /// Human-readable description.
    #[pyo3(get)]
    pub description: String,
    /// License identifier (e.g. "MIT", "GPL-2.0").
    #[pyo3(get)]
    pub license: String,
    /// Whether the plugin includes patent-encumbered codecs.
    #[pyo3(get)]
    pub patent_encumbered: bool,
    /// Plugin API version.
    #[pyo3(get)]
    pub api_version: u32,
}

#[pymethods]
impl PyPluginInfo {
    fn __repr__(&self) -> String {
        let patent_tag = if self.patent_encumbered {
            " [PATENT]"
        } else {
            ""
        };
        format!(
            "PyPluginInfo(name='{}', version='{}', license='{}'{patent_tag})",
            self.name, self.version, self.license
        )
    }

    /// Convert to a Python dict (as JSON-compatible HashMap).
    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("name".to_string(), self.name.clone());
        m.insert("version".to_string(), self.version.clone());
        m.insert("author".to_string(), self.author.clone());
        m.insert("description".to_string(), self.description.clone());
        m.insert("license".to_string(), self.license.clone());
        m.insert(
            "patent_encumbered".to_string(),
            self.patent_encumbered.to_string(),
        );
        m.insert("api_version".to_string(), self.api_version.to_string());
        m
    }

    /// Whether this plugin contains patent-encumbered codecs.
    fn is_patent_encumbered(&self) -> bool {
        self.patent_encumbered
    }
}

impl From<CodecPluginInfo> for PyPluginInfo {
    fn from(info: CodecPluginInfo) -> Self {
        Self {
            name: info.name,
            version: info.version,
            author: info.author,
            description: info.description,
            license: info.license,
            patent_encumbered: info.patent_encumbered,
            api_version: info.api_version,
        }
    }
}

// ---------------------------------------------------------------------------
// PyPluginCapability
// ---------------------------------------------------------------------------

/// A single codec capability provided by a plugin.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyPluginCapability {
    /// Codec identifier string (e.g. "h264", "aac").
    #[pyo3(get)]
    pub codec_name: String,
    /// Whether decoding is supported.
    #[pyo3(get)]
    pub can_decode: bool,
    /// Whether encoding is supported.
    #[pyo3(get)]
    pub can_encode: bool,
    /// Supported pixel formats for video.
    #[pyo3(get)]
    pub pixel_formats: Vec<String>,
}

#[pymethods]
impl PyPluginCapability {
    fn __repr__(&self) -> String {
        let mode = match (self.can_decode, self.can_encode) {
            (true, true) => "decode+encode",
            (true, false) => "decode",
            (false, true) => "encode",
            (false, false) => "none",
        };
        format!(
            "PyPluginCapability(codec='{}', mode='{}')",
            self.codec_name, mode
        )
    }

    /// Convert to a Python dict (as JSON-compatible HashMap).
    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("codec_name".to_string(), self.codec_name.clone());
        m.insert("can_decode".to_string(), self.can_decode.to_string());
        m.insert("can_encode".to_string(), self.can_encode.to_string());
        m.insert("pixel_formats".to_string(), self.pixel_formats.join(","));
        m
    }
}

impl From<PluginCapability> for PyPluginCapability {
    fn from(cap: PluginCapability) -> Self {
        Self {
            codec_name: cap.codec_name,
            can_decode: cap.can_decode,
            can_encode: cap.can_encode,
            pixel_formats: cap.pixel_formats,
        }
    }
}

// ---------------------------------------------------------------------------
// PyPluginRegistry
// ---------------------------------------------------------------------------

/// Central registry for managing OxiMedia plugins.
///
/// Provides methods to list plugins, query codecs, and manage search paths.
#[pyclass]
pub struct PyPluginRegistry {
    inner: PluginRegistry,
}

#[pymethods]
impl PyPluginRegistry {
    /// Create a new plugin registry with default search paths.
    #[new]
    fn new() -> Self {
        Self {
            inner: PluginRegistry::new(),
        }
    }

    /// Add a directory to the plugin search path.
    fn add_search_path(&mut self, path: String) {
        self.inner.add_search_path(std::path::PathBuf::from(path));
    }

    /// List all registered plugins.
    fn list_plugins(&self) -> Vec<PyPluginInfo> {
        self.inner
            .list_plugins()
            .into_iter()
            .map(PyPluginInfo::from)
            .collect()
    }

    /// List all available codecs across all registered plugins.
    fn list_codecs(&self) -> Vec<PyPluginCapability> {
        self.inner
            .list_codecs()
            .into_iter()
            .map(PyPluginCapability::from)
            .collect()
    }

    /// Check if any plugin provides the given codec.
    fn has_codec(&self, name: &str) -> bool {
        self.inner.has_codec(name)
    }

    /// Get the number of registered plugins.
    fn plugin_count(&self) -> usize {
        self.inner.plugin_count()
    }

    /// Get the list of search paths as strings.
    fn search_paths(&self) -> Vec<String> {
        self.inner
            .search_paths()
            .iter()
            .map(|p| p.display().to_string())
            .collect()
    }

    /// Check if any plugin can decode the given codec.
    fn has_decoder(&self, name: &str) -> bool {
        self.inner.has_decoder(name)
    }

    /// Check if any plugin can encode the given codec.
    fn has_encoder(&self, name: &str) -> bool {
        self.inner.has_encoder(name)
    }

    /// Find which plugin provides a given codec.
    fn find_plugin_for_codec(&self, codec_name: &str) -> Option<PyPluginInfo> {
        self.inner
            .find_plugin_for_codec(codec_name)
            .map(PyPluginInfo::from)
    }

    /// Clear all registered plugins.
    fn clear(&self) {
        self.inner.clear();
    }

    fn __repr__(&self) -> String {
        format!(
            "PyPluginRegistry(plugins={}, search_paths={})",
            self.inner.plugin_count(),
            self.inner.search_paths().len(),
        )
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Get the default plugin search paths.
///
/// Returns a list of directory paths where OxiMedia looks for plugins.
#[pyfunction]
pub fn default_plugin_paths() -> Vec<String> {
    PluginRegistry::default_search_paths()
        .iter()
        .map(|p| p.display().to_string())
        .collect()
}

/// Get the current plugin API version.
///
/// Plugins must be built against the same API version to be loaded.
#[pyfunction]
pub fn plugin_api_version() -> u32 {
    PLUGIN_API_VERSION
}

// ---------------------------------------------------------------------------
// Registration helper
// ---------------------------------------------------------------------------

/// Register all plugin bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPluginInfo>()?;
    m.add_class::<PyPluginCapability>()?;
    m.add_class::<PyPluginRegistry>()?;
    m.add_function(wrap_pyfunction!(default_plugin_paths, m)?)?;
    m.add_function(wrap_pyfunction!(plugin_api_version, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_info_from_codec_plugin_info() {
        let info = CodecPluginInfo {
            name: "test-plugin".to_string(),
            version: "1.0.0".to_string(),
            author: "Test Author".to_string(),
            description: "A test plugin".to_string(),
            api_version: PLUGIN_API_VERSION,
            license: "MIT".to_string(),
            patent_encumbered: false,
        };
        let py_info = PyPluginInfo::from(info);
        assert_eq!(py_info.name, "test-plugin");
        assert_eq!(py_info.version, "1.0.0");
        assert!(!py_info.patent_encumbered);
    }

    #[test]
    fn test_plugin_info_repr() {
        let info = PyPluginInfo {
            name: "my-plugin".to_string(),
            version: "2.0.0".to_string(),
            author: "Author".to_string(),
            description: "Desc".to_string(),
            license: "MIT".to_string(),
            patent_encumbered: false,
            api_version: 1,
        };
        let repr = info.__repr__();
        assert!(repr.contains("my-plugin"));
        assert!(repr.contains("2.0.0"));
        assert!(!repr.contains("PATENT"));
    }

    #[test]
    fn test_plugin_info_repr_patent() {
        let info = PyPluginInfo {
            name: "h264-plugin".to_string(),
            version: "1.0.0".to_string(),
            author: "Corp".to_string(),
            description: "H.264".to_string(),
            license: "proprietary".to_string(),
            patent_encumbered: true,
            api_version: 1,
        };
        let repr = info.__repr__();
        assert!(repr.contains("PATENT"));
    }

    #[test]
    fn test_capability_from() {
        let cap = PluginCapability {
            codec_name: "h264".to_string(),
            can_decode: true,
            can_encode: false,
            pixel_formats: vec!["yuv420p".to_string()],
            properties: std::collections::HashMap::new(),
        };
        let py_cap = PyPluginCapability::from(cap);
        assert_eq!(py_cap.codec_name, "h264");
        assert!(py_cap.can_decode);
        assert!(!py_cap.can_encode);
        assert_eq!(py_cap.pixel_formats, vec!["yuv420p".to_string()]);
    }

    #[test]
    fn test_capability_repr() {
        let cap = PyPluginCapability {
            codec_name: "aac".to_string(),
            can_decode: false,
            can_encode: true,
            pixel_formats: vec![],
        };
        let repr = cap.__repr__();
        assert!(repr.contains("aac"));
        assert!(repr.contains("encode"));
    }

    #[test]
    fn test_plugin_api_version_nonzero() {
        assert!(plugin_api_version() > 0);
    }

    #[test]
    fn test_default_paths_not_empty() {
        let paths = default_plugin_paths();
        assert!(!paths.is_empty());
    }

    #[test]
    fn test_registry_empty() {
        let reg = PyPluginRegistry::new();
        // New registry has no plugins loaded by default (just search paths)
        assert_eq!(reg.plugin_count(), 0);
        assert!(!reg.search_paths().is_empty());
    }
}
