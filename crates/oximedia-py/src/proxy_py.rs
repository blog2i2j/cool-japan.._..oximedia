//! Python bindings for proxy media generation and management.
//!
//! Provides `PyProxyGenerator`, `PyProxyConfig`, `PyProxyFile` and standalone
//! functions for generating and managing proxy media files.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn now_timestamp() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", dur.as_secs())
}

fn gen_id() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("proxy-{:016x}", dur.as_nanos())
}

// ---------------------------------------------------------------------------
// PyProxyConfig
// ---------------------------------------------------------------------------

/// Configuration for proxy generation.
#[pyclass]
#[derive(Clone)]
pub struct PyProxyConfig {
    /// Resolution preset: quarter, half, full.
    #[pyo3(get, set)]
    pub resolution: String,
    /// Quality preset: low, medium, high.
    #[pyo3(get, set)]
    pub quality: String,
    /// Codec: vp9, av1.
    #[pyo3(get, set)]
    pub codec: String,
    /// Target bitrate in bps (0 = auto).
    #[pyo3(get, set)]
    pub bitrate: u64,
}

#[pymethods]
impl PyProxyConfig {
    #[new]
    #[pyo3(signature = (resolution="quarter", quality="medium", codec="vp9", bitrate=0))]
    fn new(resolution: &str, quality: &str, codec: &str, bitrate: u64) -> Self {
        Self {
            resolution: resolution.to_string(),
            quality: quality.to_string(),
            codec: codec.to_string(),
            bitrate,
        }
    }

    /// Get the resolution scale factor (0.0-1.0).
    fn scale_factor(&self) -> f64 {
        match self.resolution.as_str() {
            "quarter" => 0.25,
            "half" => 0.5,
            "full" => 1.0,
            _ => 0.25,
        }
    }

    /// Estimate output bitrate for a given resolution.
    fn estimated_bitrate(&self, width: u32, height: u32) -> u64 {
        if self.bitrate > 0 {
            return self.bitrate;
        }
        let scale = self.scale_factor();
        let pixels = (width as f64 * scale) * (height as f64 * scale);
        let quality_mult = match self.quality.as_str() {
            "low" => 0.5,
            "medium" => 1.0,
            "high" => 2.0,
            _ => 1.0,
        };
        (pixels * 2.0 * quality_mult) as u64
    }

    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("resolution".to_string(), self.resolution.clone());
        m.insert("quality".to_string(), self.quality.clone());
        m.insert("codec".to_string(), self.codec.clone());
        m.insert("bitrate".to_string(), self.bitrate.to_string());
        m
    }

    fn __repr__(&self) -> String {
        format!(
            "PyProxyConfig(resolution='{}', quality='{}', codec='{}')",
            self.resolution, self.quality, self.codec
        )
    }
}

// ---------------------------------------------------------------------------
// PyProxyFile
// ---------------------------------------------------------------------------

/// A proxy file entry linking proxy to original.
#[pyclass]
#[derive(Clone)]
pub struct PyProxyFile {
    /// Proxy identifier.
    #[pyo3(get)]
    pub id: String,
    /// Original file path.
    #[pyo3(get)]
    pub original_path: String,
    /// Proxy file path.
    #[pyo3(get)]
    pub proxy_path: String,
    /// Resolution preset used.
    #[pyo3(get)]
    pub resolution: String,
    /// Quality preset used.
    #[pyo3(get)]
    pub quality: String,
    /// Codec used.
    #[pyo3(get)]
    pub codec: String,
    /// Original file size in bytes.
    #[pyo3(get)]
    pub original_size: u64,
    /// Proxy file size in bytes.
    #[pyo3(get)]
    pub proxy_size: u64,
    /// Creation timestamp.
    #[pyo3(get)]
    pub created_at: String,
}

#[pymethods]
impl PyProxyFile {
    /// Compression ratio (original / proxy).
    fn compression_ratio(&self) -> f64 {
        if self.proxy_size == 0 {
            return 0.0;
        }
        self.original_size as f64 / self.proxy_size as f64
    }

    /// Space savings as percentage.
    fn space_savings_pct(&self) -> f64 {
        if self.original_size == 0 {
            return 0.0;
        }
        (1.0 - self.proxy_size as f64 / self.original_size as f64) * 100.0
    }

    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("id".to_string(), self.id.clone());
        m.insert("original_path".to_string(), self.original_path.clone());
        m.insert("proxy_path".to_string(), self.proxy_path.clone());
        m.insert("resolution".to_string(), self.resolution.clone());
        m.insert("quality".to_string(), self.quality.clone());
        m.insert("codec".to_string(), self.codec.clone());
        m.insert("original_size".to_string(), self.original_size.to_string());
        m.insert("proxy_size".to_string(), self.proxy_size.to_string());
        m
    }

    fn __repr__(&self) -> String {
        format!(
            "PyProxyFile(original='{}', proxy='{}', resolution='{}')",
            self.original_path, self.proxy_path, self.resolution
        )
    }
}

// ---------------------------------------------------------------------------
// PyProxyGenerator
// ---------------------------------------------------------------------------

/// Proxy media generator and manager.
#[pyclass]
pub struct PyProxyGenerator {
    config: PyProxyConfig,
    proxies: Vec<PyProxyFile>,
}

#[pymethods]
impl PyProxyGenerator {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyProxyConfig>) -> Self {
        Self {
            config: config.unwrap_or_else(|| PyProxyConfig::new("quarter", "medium", "vp9", 0)),
            proxies: Vec::new(),
        }
    }

    /// Generate a proxy for a source file.
    fn generate(&mut self, original_path: &str, proxy_path: &str) -> PyResult<PyProxyFile> {
        let orig = std::path::Path::new(original_path);
        if !orig.exists() {
            return Err(PyValueError::new_err(format!(
                "Original file not found: {original_path}"
            )));
        }

        let orig_meta = std::fs::metadata(orig)
            .map_err(|e| PyRuntimeError::new_err(format!("Metadata error: {e}")))?;

        let scale = self.config.scale_factor();
        let quality_factor = match self.config.quality.as_str() {
            "low" => 0.1,
            "medium" => 0.25,
            "high" => 0.5,
            _ => 0.25,
        };
        let estimated_size = (orig_meta.len() as f64 * scale * scale * quality_factor) as u64;

        // Write placeholder proxy
        let content = format!(
            "PROXY:original={},resolution={},quality={},codec={}",
            original_path, self.config.resolution, self.config.quality, self.config.codec
        );
        if let Some(parent) = std::path::Path::new(proxy_path).parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to create proxy dir: {e}"))
                })?;
            }
        }
        std::fs::write(proxy_path, &content)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to write proxy: {e}")))?;

        let proxy_file = PyProxyFile {
            id: gen_id(),
            original_path: original_path.to_string(),
            proxy_path: proxy_path.to_string(),
            resolution: self.config.resolution.clone(),
            quality: self.config.quality.clone(),
            codec: self.config.codec.clone(),
            original_size: orig_meta.len(),
            proxy_size: estimated_size,
            created_at: now_timestamp(),
        };

        let result = proxy_file.clone();
        self.proxies.push(proxy_file);
        Ok(result)
    }

    /// List all generated proxies.
    fn list_proxies(&self) -> Vec<PyProxyFile> {
        self.proxies.clone()
    }

    /// Get proxy count.
    fn proxy_count(&self) -> usize {
        self.proxies.len()
    }

    /// Get current config.
    fn config(&self) -> PyProxyConfig {
        self.config.clone()
    }

    /// Total space saved across all proxies.
    fn total_space_saved(&self) -> u64 {
        self.proxies
            .iter()
            .map(|p| p.original_size.saturating_sub(p.proxy_size))
            .sum()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyProxyGenerator(proxies={}, config={})",
            self.proxies.len(),
            self.config.__repr__()
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Generate a proxy for a single file with default settings.
#[pyfunction]
#[pyo3(signature = (original_path, proxy_path, resolution="quarter", quality="medium"))]
pub fn generate_proxy(
    original_path: &str,
    proxy_path: &str,
    resolution: &str,
    quality: &str,
) -> PyResult<PyProxyFile> {
    let config = PyProxyConfig::new(resolution, quality, "vp9", 0);
    let mut gen = PyProxyGenerator::new(Some(config));
    gen.generate(original_path, proxy_path)
}

/// List supported proxy formats.
#[pyfunction]
pub fn list_proxy_formats() -> Vec<HashMap<String, String>> {
    let formats = vec![
        ("vp9", "WebM/VP9", "Good balance of quality and size"),
        ("av1", "WebM/AV1", "Best compression, slower encoding"),
    ];
    formats
        .into_iter()
        .map(|(codec, name, desc)| {
            let mut m = HashMap::new();
            m.insert("codec".to_string(), codec.to_string());
            m.insert("name".to_string(), name.to_string());
            m.insert("description".to_string(), desc.to_string());
            m
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all proxy bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProxyConfig>()?;
    m.add_class::<PyProxyFile>()?;
    m.add_class::<PyProxyGenerator>()?;
    m.add_function(wrap_pyfunction!(generate_proxy, m)?)?;
    m.add_function(wrap_pyfunction!(list_proxy_formats, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_str(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("oximedia-py-proxy-{name}"))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn test_proxy_config_scale() {
        let cfg = PyProxyConfig::new("quarter", "medium", "vp9", 0);
        assert!((cfg.scale_factor() - 0.25).abs() < f64::EPSILON);

        let cfg = PyProxyConfig::new("half", "medium", "vp9", 0);
        assert!((cfg.scale_factor() - 0.5).abs() < f64::EPSILON);

        let cfg = PyProxyConfig::new("full", "medium", "vp9", 0);
        assert!((cfg.scale_factor() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_proxy_config_estimated_bitrate() {
        let cfg = PyProxyConfig::new("quarter", "medium", "vp9", 5_000_000);
        assert_eq!(cfg.estimated_bitrate(1920, 1080), 5_000_000);

        let cfg = PyProxyConfig::new("quarter", "medium", "vp9", 0);
        let br = cfg.estimated_bitrate(1920, 1080);
        assert!(br > 0);
    }

    #[test]
    fn test_proxy_file_compression() {
        let pf = PyProxyFile {
            id: "proxy-001".to_string(),
            original_path: tmp_str("orig.mov"),
            proxy_path: tmp_str("proxy.webm"),
            resolution: "quarter".to_string(),
            quality: "medium".to_string(),
            codec: "vp9".to_string(),
            original_size: 1_000_000,
            proxy_size: 100_000,
            created_at: "0".to_string(),
        };
        assert!((pf.compression_ratio() - 10.0).abs() < f64::EPSILON);
        assert!((pf.space_savings_pct() - 90.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_proxy_file_zero_size() {
        let pf = PyProxyFile {
            id: "proxy-002".to_string(),
            original_path: tmp_str("orig.mov"),
            proxy_path: tmp_str("proxy.webm"),
            resolution: "quarter".to_string(),
            quality: "medium".to_string(),
            codec: "vp9".to_string(),
            original_size: 0,
            proxy_size: 0,
            created_at: "0".to_string(),
        };
        assert!((pf.compression_ratio() - 0.0).abs() < f64::EPSILON);
        assert!((pf.space_savings_pct() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_list_proxy_formats() {
        let formats = list_proxy_formats();
        assert_eq!(formats.len(), 2);
        assert!(formats[0].contains_key("codec"));
    }
}
