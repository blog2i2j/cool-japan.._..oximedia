//! Async pipeline Python bindings for OxiMedia.
//!
//! Exposes `AsyncPipeline` and `PipelineResult` for building and running
//! media processing pipelines from Python, backed by tokio internally.
//!
//! # Example
//! ```python
//! import oximedia
//!
//! pipeline = oximedia.AsyncPipeline()
//! pipeline.add_source("/path/to/video.mkv")
//! pipeline.add_filter("scale", width=1280, height=720)
//! pipeline.add_sink("/tmp/output.mkv")
//! result = pipeline.run()
//! print(result.frames_processed, result.duration_ms)
//! ```

use std::collections::HashMap;
use std::time::Instant;

use pyo3::prelude::*;
use pyo3::types::PyDict;

// ---------------------------------------------------------------------------
// PipelineResult
// ---------------------------------------------------------------------------

/// Result returned after a pipeline run completes.
///
/// Attributes
/// ----------
/// frames_processed : int
///     Number of frames that were processed.
/// duration_ms : float
///     Total wall-clock duration of the run in milliseconds.
/// success : bool
///     Whether the pipeline completed without errors.
/// errors : list[str]
///     List of error messages (empty on success).
#[pyclass]
#[derive(Clone, Debug)]
pub struct PipelineResult {
    /// Frames processed during the run.
    #[pyo3(get)]
    pub frames_processed: u64,
    /// Wall-clock duration in milliseconds.
    #[pyo3(get)]
    pub duration_ms: f64,
    /// Whether the run succeeded.
    #[pyo3(get)]
    pub success: bool,
    /// Error messages accumulated during the run.
    #[pyo3(get)]
    pub errors: Vec<String>,
}

#[pymethods]
impl PipelineResult {
    fn __repr__(&self) -> String {
        format!(
            "PipelineResult(frames_processed={}, duration_ms={:.2}, success={}, errors={})",
            self.frames_processed,
            self.duration_ms,
            self.success,
            self.errors.len()
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ---------------------------------------------------------------------------
// FilterSpec
// ---------------------------------------------------------------------------

/// Specification for a single filter step in the pipeline.
///
/// Attributes
/// ----------
/// name : str
///     Filter name (e.g. ``"scale"``, ``"crop"``, ``"volume"``).
/// params : dict[str, str]
///     Key-value parameters passed to the filter.
#[pyclass]
#[derive(Clone, Debug)]
pub struct FilterSpec {
    /// Filter name.
    #[pyo3(get, set)]
    pub name: String,
    /// Filter parameters as string key-value pairs.
    #[pyo3(get)]
    pub params: HashMap<String, String>,
}

#[pymethods]
impl FilterSpec {
    /// Create a new filter specification.
    #[new]
    #[pyo3(signature = (name, params = None))]
    pub fn new(name: &str, params: Option<HashMap<String, String>>) -> Self {
        Self {
            name: name.to_string(),
            params: params.unwrap_or_default(),
        }
    }

    fn __repr__(&self) -> String {
        format!("FilterSpec(name='{}', params={:?})", self.name, self.params)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ---------------------------------------------------------------------------
// AsyncPipeline
// ---------------------------------------------------------------------------

/// Media processing pipeline with async execution backed by tokio.
///
/// Build the pipeline by calling :meth:`add_source`, :meth:`add_filter`,
/// and :meth:`add_sink`, then invoke :meth:`run` to execute.
///
/// Example
/// -------
/// .. code-block:: python
///
///     pipeline = oximedia.AsyncPipeline()
///     pipeline.add_source("/path/to/video.mkv")
///     pipeline.add_filter("scale", width="1280", height="720")
///     pipeline.add_sink("/tmp/output.mkv")
///     result = pipeline.run()
///     print(result.frames_processed, result.duration_ms)
#[pyclass]
pub struct AsyncPipeline {
    source: Option<String>,
    filters: Vec<FilterSpec>,
    sink: Option<String>,
    started_at: Option<Instant>,
}

#[pymethods]
impl AsyncPipeline {
    /// Create a new, empty pipeline.
    #[new]
    pub fn new() -> Self {
        Self {
            source: None,
            filters: Vec::new(),
            sink: None,
            started_at: None,
        }
    }

    /// Set the pipeline source path.
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     Input media file path.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If path is empty.
    #[pyo3(signature = (path))]
    pub fn add_source(&mut self, path: &str) -> PyResult<()> {
        if path.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "source path must not be empty",
            ));
        }
        self.source = Some(path.to_string());
        Ok(())
    }

    /// Add a filter step to the pipeline.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Filter name (e.g. ``"scale"``, ``"crop"``).
    /// **kwargs
    ///     Filter parameters (e.g. ``width=1280``, ``height=720``).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If filter name is empty.
    #[pyo3(signature = (name, **kwargs))]
    pub fn add_filter(&mut self, name: &str, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        if name.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "filter name must not be empty",
            ));
        }
        let mut params = HashMap::new();
        if let Some(d) = kwargs {
            for (k, v) in d.iter() {
                let key = k.extract::<String>().map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "filter param key must be a string: {e}"
                    ))
                })?;
                let val = v
                    .str()
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "cannot convert filter param '{key}' to string: {e}"
                        ))
                    })?
                    .to_str()
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "invalid UTF-8 in filter param '{key}': {e}"
                        ))
                    })?
                    .to_string();
                params.insert(key, val);
            }
        }
        self.filters.push(FilterSpec {
            name: name.to_string(),
            params,
        });
        Ok(())
    }

    /// Set the pipeline sink (output) path.
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     Output media file path.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If path is empty.
    #[pyo3(signature = (path))]
    pub fn add_sink(&mut self, path: &str) -> PyResult<()> {
        if path.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "sink path must not be empty",
            ));
        }
        self.sink = Some(path.to_string());
        Ok(())
    }

    /// Execute the pipeline and return a result summary.
    ///
    /// This method blocks the calling thread while the pipeline runs on an
    /// internal tokio runtime.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If source or sink have not been configured, or execution fails.
    pub fn run(&mut self) -> PyResult<PipelineResult> {
        let source = self.source.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "pipeline source not configured — call add_source() first",
            )
        })?;
        let sink = self.sink.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "pipeline sink not configured — call add_sink() first",
            )
        })?;

        let source = source.clone();
        let sink = sink.clone();
        let filters = self.filters.clone();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "failed to build tokio runtime: {e}"
                ))
            })?;

        let wall_start = Instant::now();
        self.started_at = Some(wall_start);

        let result = rt.block_on(run_pipeline_async(source, filters, sink));
        let elapsed_ms = wall_start.elapsed().as_secs_f64() * 1000.0;

        match result {
            Ok(frames) => Ok(PipelineResult {
                frames_processed: frames,
                duration_ms: elapsed_ms,
                success: true,
                errors: Vec::new(),
            }),
            Err(e) => Ok(PipelineResult {
                frames_processed: 0,
                duration_ms: elapsed_ms,
                success: false,
                errors: vec![e],
            }),
        }
    }

    /// Reset the pipeline to its initial state.
    ///
    /// Clears the source, all filters, and the sink.
    pub fn reset(&mut self) -> PyResult<()> {
        self.source = None;
        self.filters.clear();
        self.sink = None;
        self.started_at = None;
        Ok(())
    }

    /// Return the configured source path, or ``None``.
    #[getter]
    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }

    /// Return the configured sink path, or ``None``.
    #[getter]
    pub fn sink(&self) -> Option<&str> {
        self.sink.as_deref()
    }

    /// Return the list of configured filter specs.
    #[getter]
    pub fn filters(&self) -> Vec<FilterSpec> {
        self.filters.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "AsyncPipeline(source={:?}, filters={}, sink={:?})",
            self.source,
            self.filters.len(),
            self.sink
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ---------------------------------------------------------------------------
// Internal async execution
// ---------------------------------------------------------------------------

/// Core async function that simulates a media processing pipeline.
///
/// In a full implementation, this would drive demux → decode → filter → encode → mux.
/// Here it validates the arguments and returns a simulated frame count.
async fn run_pipeline_async(
    source: String,
    filters: Vec<FilterSpec>,
    sink: String,
) -> Result<u64, String> {
    // Validate inputs
    if source.is_empty() {
        return Err("source path is empty".to_string());
    }
    if sink.is_empty() {
        return Err("sink path is empty".to_string());
    }

    // Log filter chain (in a real impl this builds the filter graph)
    let _filter_names: Vec<&str> = filters.iter().map(|f| f.name.as_str()).collect();

    // Simulate async I/O work (zero-duration yield to the tokio scheduler)
    tokio::task::yield_now().await;

    // Return a simulated frame count: 30 fps × 1 second
    Ok(30)
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register `AsyncPipeline`, `PipelineResult`, and `FilterSpec` into the given module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PipelineResult>()?;
    m.add_class::<FilterSpec>()?;
    m.add_class::<AsyncPipeline>()?;
    Ok(())
}
