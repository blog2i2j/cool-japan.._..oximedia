//! Python bindings for HLS/DASH adaptive streaming packaging via `oximedia-packager`.
//!
//! Exposes `PackagingConfig` and `PyPackager` classes that map directly to the
//! `Packager` and `PackagerConfig` Rust types.

use std::time::Duration;

use pyo3::prelude::*;

use oximedia_packager::{
    config::{PackagingFormat, SegmentConfig},
    Packager, PackagerConfig,
};

// ---------------------------------------------------------------------------
// Python-visible types
// ---------------------------------------------------------------------------

/// Configuration for HLS/DASH adaptive streaming packaging.
///
/// Parameters
/// ----------
/// format : str, optional
///     Packaging format: ``"hls"`` (fMP4 segments), ``"hls_ts"`` (TS
///     segments), ``"dash"``, or ``"both"``.  Default: ``"hls"``.
/// segment_duration : int, optional
///     Target segment duration in seconds.  Default: 6.
/// output_dir : str, optional
///     Output directory path.  Default: ``"output"``.
/// low_latency : bool, optional
///     Enable low-latency mode.  Default: ``False``.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PackagingConfig {
    /// Packaging format string.
    #[pyo3(get, set)]
    pub format: String,
    /// Target segment duration in seconds.
    #[pyo3(get, set)]
    pub segment_duration: u64,
    /// Output directory.
    #[pyo3(get, set)]
    pub output_dir: String,
    /// Low-latency mode flag.
    #[pyo3(get, set)]
    pub low_latency: bool,
}

#[pymethods]
impl PackagingConfig {
    #[new]
    #[pyo3(signature = (format="hls", segment_duration=6, output_dir="output", low_latency=false))]
    pub fn new(format: &str, segment_duration: u64, output_dir: &str, low_latency: bool) -> Self {
        Self {
            format: format.to_string(),
            segment_duration,
            output_dir: output_dir.to_string(),
            low_latency,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PackagingConfig(format='{}', segment_duration={}, output_dir='{}', low_latency={})",
            self.format, self.segment_duration, self.output_dir, self.low_latency
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map a format string to the `PackagingFormat` enum.
fn parse_format(s: &str) -> PyResult<PackagingFormat> {
    match s.to_lowercase().as_str() {
        "hls" | "hls_fmp4" | "hlsfmp4" => Ok(PackagingFormat::HlsFmp4),
        "hls_ts" | "hlsts" => Ok(PackagingFormat::HlsTs),
        "dash" => Ok(PackagingFormat::Dash),
        "both" => Ok(PackagingFormat::Both),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown packaging format '{other}'. \
             Valid values: 'hls', 'hls_ts', 'dash', 'both'."
        ))),
    }
}

/// Build a `PackagerConfig` from a `PackagingConfig`.
fn build_packager_config(cfg: &PackagingConfig) -> PyResult<PackagerConfig> {
    let format = parse_format(&cfg.format)?;
    let segment = SegmentConfig {
        duration: Duration::from_secs(cfg.segment_duration),
        ..SegmentConfig::default()
    };

    let output = oximedia_packager::config::OutputConfig {
        directory: std::path::PathBuf::from(&cfg.output_dir),
        ..oximedia_packager::config::OutputConfig::default()
    };

    Ok(PackagerConfig {
        format,
        segment,
        output,
        low_latency: cfg.low_latency,
        ..PackagerConfig::default()
    })
}

// ---------------------------------------------------------------------------
// Packager class
// ---------------------------------------------------------------------------

/// HLS/DASH adaptive streaming packager.
///
/// Parameters
/// ----------
/// config : PackagingConfig
///     Packaging configuration.
///
/// Example
/// -------
/// .. code-block:: python
///
///     config = oximedia.PackagingConfig(format="hls", segment_duration=6, output_dir="/tmp/out")
///     packager = oximedia.PyPackager(config)
///     packager.package("/path/to/input.mkv")
#[pyclass]
pub struct PyPackager {
    config: PackagingConfig,
}

#[pymethods]
impl PyPackager {
    #[new]
    #[pyo3(signature = (config))]
    pub fn new(config: PackagingConfig) -> PyResult<Self> {
        // Validate the config early so errors surface at construction time
        let packager_config = build_packager_config(&config)?;
        packager_config
            .validate()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { config })
    }

    /// Package the input media file into the configured streaming format.
    ///
    /// Parameters
    /// ----------
    /// input_path : str
    ///     Path to the input media file (e.g. ``"/path/to/video.mkv"``).
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If packaging fails.
    #[pyo3(signature = (input_path))]
    pub fn package(&self, input_path: &str) -> PyResult<()> {
        let packager_config = build_packager_config(&self.config)?;
        let packager = Packager::new(packager_config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Run packaging on a tokio runtime — oximedia-packager uses async I/O
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        match self.config.format.to_lowercase().as_str() {
            "dash" => {
                rt.block_on(packager.package_dash(input_path, &self.config.output_dir))
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            }
            "both" => {
                rt.block_on(packager.package_both(input_path, &self.config.output_dir))
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            }
            _ => {
                // HLS variants
                rt.block_on(packager.package_hls(input_path, &self.config.output_dir))
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            }
        }

        Ok(())
    }

    /// Return the packaging configuration.
    #[getter]
    pub fn config(&self) -> PackagingConfig {
        self.config.clone()
    }

    fn __repr__(&self) -> String {
        format!("PyPackager(config={})", self.config.__repr__())
    }
}

// ---------------------------------------------------------------------------
// Module registration helper
// ---------------------------------------------------------------------------

/// Register all streaming packaging classes into the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PackagingConfig>()?;
    m.add_class::<PyPackager>()?;
    Ok(())
}
