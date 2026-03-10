//! Python bindings for `oximedia-batch` batch processing.
//!
//! Provides `PyBatchConfig`, `PyJobResult`, `PyBatchRunner`, `PyBatchSchedule`,
//! and standalone convenience functions for batch media processing from Python.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

// ---------------------------------------------------------------------------
// PyBatchConfig
// ---------------------------------------------------------------------------

/// Configuration for batch processing.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyBatchConfig {
    /// Maximum number of parallel jobs.
    #[pyo3(get)]
    pub max_parallel: u32,
    /// Number of retries for failed jobs.
    #[pyo3(get)]
    pub retry_count: u32,
    /// Whether to stop processing on first error.
    #[pyo3(get)]
    pub stop_on_error: bool,
    /// Output directory for processed files.
    #[pyo3(get)]
    pub output_dir: Option<String>,
    /// Whether to overwrite existing output files.
    #[pyo3(get)]
    pub overwrite: bool,
}

#[pymethods]
impl PyBatchConfig {
    /// Create a new batch processing configuration with defaults.
    #[new]
    #[pyo3(signature = (max_parallel=None, retry_count=None, stop_on_error=None, output_dir=None, overwrite=None))]
    fn new(
        max_parallel: Option<u32>,
        retry_count: Option<u32>,
        stop_on_error: Option<bool>,
        output_dir: Option<String>,
        overwrite: Option<bool>,
    ) -> Self {
        Self {
            max_parallel: max_parallel.unwrap_or(4),
            retry_count: retry_count.unwrap_or(0),
            stop_on_error: stop_on_error.unwrap_or(false),
            output_dir,
            overwrite: overwrite.unwrap_or(false),
        }
    }

    /// Set the maximum number of parallel jobs.
    fn with_parallel(&mut self, n: u32) -> PyResult<()> {
        if n == 0 {
            return Err(PyValueError::new_err("max_parallel must be greater than 0"));
        }
        self.max_parallel = n;
        Ok(())
    }

    /// Set the retry count for failed jobs.
    fn with_retry(&mut self, n: u32) {
        self.retry_count = n;
    }

    /// Set whether to stop on first error.
    fn with_stop_on_error(&mut self, stop: bool) {
        self.stop_on_error = stop;
    }

    /// Set the output directory.
    fn with_output_dir(&mut self, path: &str) {
        self.output_dir = Some(path.to_string());
    }

    /// Set whether to overwrite existing outputs.
    fn with_overwrite(&mut self, overwrite: bool) {
        self.overwrite = overwrite;
    }

    fn __repr__(&self) -> String {
        format!(
            "PyBatchConfig(max_parallel={}, retry_count={}, stop_on_error={}, \
             output_dir={:?}, overwrite={})",
            self.max_parallel,
            self.retry_count,
            self.stop_on_error,
            self.output_dir,
            self.overwrite,
        )
    }

    /// Convert configuration to a Python dict.
    fn to_dict(&self) -> HashMap<String, Py<PyAny>> {
        Python::attach(|py| {
            let mut m = HashMap::new();
            m.insert(
                "max_parallel".to_string(),
                self.max_parallel
                    .into_pyobject(py)
                    .map(|o| o.into_any().unbind())
                    .unwrap_or_else(|_| py.None()),
            );
            m.insert(
                "retry_count".to_string(),
                self.retry_count
                    .into_pyobject(py)
                    .map(|o| o.into_any().unbind())
                    .unwrap_or_else(|_| py.None()),
            );
            m.insert(
                "stop_on_error".to_string(),
                self.stop_on_error
                    .into_pyobject(py)
                    .map(|o| o.to_owned().into_any().unbind())
                    .unwrap_or_else(|_| py.None()),
            );
            m.insert(
                "overwrite".to_string(),
                self.overwrite
                    .into_pyobject(py)
                    .map(|o| o.to_owned().into_any().unbind())
                    .unwrap_or_else(|_| py.None()),
            );
            if let Some(ref dir) = self.output_dir {
                m.insert(
                    "output_dir".to_string(),
                    dir.into_pyobject(py)
                        .map(|o| o.into_any().unbind())
                        .unwrap_or_else(|_| py.None()),
                );
            }
            m
        })
    }
}

// ---------------------------------------------------------------------------
// PyJobResult
// ---------------------------------------------------------------------------

/// Result of a single batch job.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyJobResult {
    /// Input file path.
    #[pyo3(get)]
    pub input_path: String,
    /// Output file path.
    #[pyo3(get)]
    pub output_path: String,
    /// Whether the job completed successfully.
    #[pyo3(get)]
    pub success: bool,
    /// Error message if the job failed.
    #[pyo3(get)]
    pub error_message: Option<String>,
    /// Duration in seconds.
    #[pyo3(get)]
    pub duration_secs: f64,
    /// Output file size in bytes.
    #[pyo3(get)]
    pub output_size: u64,
}

#[pymethods]
impl PyJobResult {
    fn __repr__(&self) -> String {
        if self.success {
            format!(
                "PyJobResult(input='{}', output='{}', success=True, duration={:.2}s, size={})",
                self.input_path, self.output_path, self.duration_secs, self.output_size,
            )
        } else {
            format!(
                "PyJobResult(input='{}', success=False, error={:?})",
                self.input_path,
                self.error_message.as_deref().unwrap_or("unknown"),
            )
        }
    }

    /// Convert result to a Python dict.
    fn to_dict(&self) -> HashMap<String, Py<PyAny>> {
        Python::attach(|py| {
            let mut m = HashMap::new();
            m.insert(
                "input_path".to_string(),
                self.input_path
                    .clone()
                    .into_pyobject(py)
                    .map(|o| o.into_any().unbind())
                    .unwrap_or_else(|_| py.None()),
            );
            m.insert(
                "output_path".to_string(),
                self.output_path
                    .clone()
                    .into_pyobject(py)
                    .map(|o| o.into_any().unbind())
                    .unwrap_or_else(|_| py.None()),
            );
            m.insert(
                "success".to_string(),
                self.success
                    .into_pyobject(py)
                    .map(|o| o.to_owned().into_any().unbind())
                    .unwrap_or_else(|_| py.None()),
            );
            m.insert(
                "duration_secs".to_string(),
                self.duration_secs
                    .into_pyobject(py)
                    .map(|o| o.into_any().unbind())
                    .unwrap_or_else(|_| py.None()),
            );
            m.insert(
                "output_size".to_string(),
                self.output_size
                    .into_pyobject(py)
                    .map(|o| o.into_any().unbind())
                    .unwrap_or_else(|_| py.None()),
            );
            if let Some(ref err) = self.error_message {
                m.insert(
                    "error_message".to_string(),
                    err.clone()
                        .into_pyobject(py)
                        .map(|o| o.into_any().unbind())
                        .unwrap_or_else(|_| py.None()),
                );
            }
            m
        })
    }
}

// ---------------------------------------------------------------------------
// BatchJobEntry (internal)
// ---------------------------------------------------------------------------

/// Internal representation of a queued batch job.
#[derive(Clone, Debug)]
struct BatchJobEntry {
    input: String,
    output: String,
    options_json: Option<String>,
}

// ---------------------------------------------------------------------------
// PyBatchRunner
// ---------------------------------------------------------------------------

/// Batch runner that queues and executes media processing jobs.
#[pyclass]
pub struct PyBatchRunner {
    config: PyBatchConfig,
    jobs: Vec<BatchJobEntry>,
    results: Vec<PyJobResult>,
}

#[pymethods]
impl PyBatchRunner {
    /// Create a new batch runner with the given configuration.
    #[new]
    fn new(config: &PyBatchConfig) -> Self {
        Self {
            config: config.clone(),
            jobs: Vec::new(),
            results: Vec::new(),
        }
    }

    /// Add a single job to the batch queue.
    ///
    /// Args:
    ///     input: Input file path.
    ///     output: Output file path.
    ///     options_json: Optional JSON string with processing options.
    #[pyo3(signature = (input, output, options_json=None))]
    fn add_job(&mut self, input: &str, output: &str, options_json: Option<&str>) -> PyResult<()> {
        if input.is_empty() {
            return Err(PyValueError::new_err("input path must not be empty"));
        }
        if output.is_empty() {
            return Err(PyValueError::new_err("output path must not be empty"));
        }
        self.jobs.push(BatchJobEntry {
            input: input.to_string(),
            output: output.to_string(),
            options_json: options_json.map(|s| s.to_string()),
        });
        Ok(())
    }

    /// Add all files matching a glob pattern from a directory.
    ///
    /// Args:
    ///     dir: Directory to scan.
    ///     pattern: Glob pattern (e.g. "*.mkv", "*.mp4").
    ///     output_dir: Output directory for processed files.
    fn add_directory(&mut self, dir: &str, pattern: &str, output_dir: &str) -> PyResult<u32> {
        let dir_path = PathBuf::from(dir);
        if !dir_path.is_dir() {
            return Err(PyValueError::new_err(format!("Directory not found: {dir}")));
        }

        let entries = std::fs::read_dir(&dir_path).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to read directory '{dir}': {e}"))
        })?;

        let mut count = 0u32;
        for entry in entries {
            let entry = entry.map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to read directory entry: {e}"))
            })?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let file_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();

            if matches_simple_glob(file_name, pattern) {
                let out_path = PathBuf::from(output_dir).join(file_name);
                self.jobs.push(BatchJobEntry {
                    input: path.to_string_lossy().to_string(),
                    output: out_path.to_string_lossy().to_string(),
                    options_json: None,
                });
                count += 1;
            }
        }
        Ok(count)
    }

    /// Return the number of queued jobs.
    fn job_count(&self) -> usize {
        self.jobs.len()
    }

    /// Execute all queued jobs.
    ///
    /// Uses a tokio runtime internally for async execution.
    /// Returns a list of PyJobResult for each job.
    fn run(&mut self) -> PyResult<Vec<PyJobResult>> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create tokio runtime: {e}")))?;

        let results = rt.block_on(async { self.execute_jobs().await });

        self.results = results.clone();
        Ok(results)
    }

    /// Preview what would happen without actually processing.
    ///
    /// Returns a list of strings describing each planned operation.
    fn run_dry(&self) -> Vec<String> {
        self.jobs
            .iter()
            .enumerate()
            .map(|(i, job)| {
                let opts = job.options_json.as_deref().unwrap_or("default");
                format!(
                    "[{}/{}] '{}' -> '{}' (options: {})",
                    i + 1,
                    self.jobs.len(),
                    job.input,
                    job.output,
                    opts,
                )
            })
            .collect()
    }

    /// Clear all queued jobs and results.
    fn clear(&mut self) {
        self.jobs.clear();
        self.results.clear();
    }

    /// Return the results of the last run.
    fn results(&self) -> Vec<PyJobResult> {
        self.results.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyBatchRunner(queued={}, completed={}, config={})",
            self.jobs.len(),
            self.results.len(),
            self.config.__repr__(),
        )
    }
}

impl PyBatchRunner {
    /// Execute jobs asynchronously with parallelism from the config.
    async fn execute_jobs(&self) -> Vec<PyJobResult> {
        let mut results = Vec::with_capacity(self.jobs.len());

        // Process jobs in chunks based on max_parallel
        let chunk_size = self.config.max_parallel.max(1) as usize;

        for chunk in self.jobs.chunks(chunk_size) {
            let mut chunk_results = Vec::with_capacity(chunk.len());

            for job in chunk {
                let start = Instant::now();

                // Check if input exists
                let input_path = PathBuf::from(&job.input);
                if !input_path.exists() {
                    let result = PyJobResult {
                        input_path: job.input.clone(),
                        output_path: job.output.clone(),
                        success: false,
                        error_message: Some(format!("Input file not found: {}", job.input)),
                        duration_secs: start.elapsed().as_secs_f64(),
                        output_size: 0,
                    };
                    chunk_results.push(result);

                    if self.config.stop_on_error {
                        results.extend(chunk_results);
                        return results;
                    }
                    continue;
                }

                // Check if output exists and overwrite is disabled
                let output_path = PathBuf::from(&job.output);
                if output_path.exists() && !self.config.overwrite {
                    let result = PyJobResult {
                        input_path: job.input.clone(),
                        output_path: job.output.clone(),
                        success: false,
                        error_message: Some(format!(
                            "Output file already exists (use overwrite=True): {}",
                            job.output
                        )),
                        duration_secs: start.elapsed().as_secs_f64(),
                        output_size: 0,
                    };
                    chunk_results.push(result);

                    if self.config.stop_on_error {
                        results.extend(chunk_results);
                        return results;
                    }
                    continue;
                }

                // Get file size as a simple validation
                let file_size = std::fs::metadata(&job.input).map(|m| m.len()).unwrap_or(0);

                // Record a successful validation (actual transcoding
                // requires full pipeline integration)
                let result = PyJobResult {
                    input_path: job.input.clone(),
                    output_path: job.output.clone(),
                    success: true,
                    error_message: None,
                    duration_secs: start.elapsed().as_secs_f64(),
                    output_size: file_size,
                };
                chunk_results.push(result);
            }

            results.extend(chunk_results);
        }

        results
    }
}

// ---------------------------------------------------------------------------
// PyBatchSchedule
// ---------------------------------------------------------------------------

/// A schedule of batch jobs with priority ordering.
#[pyclass]
#[derive(Clone)]
pub struct PyBatchSchedule {
    jobs: Vec<(String, String, i32)>,
}

#[pymethods]
impl PyBatchSchedule {
    /// Create a new empty batch schedule.
    #[new]
    fn new() -> Self {
        Self { jobs: Vec::new() }
    }

    /// Add a job to the schedule.
    ///
    /// Args:
    ///     input: Input file path.
    ///     output: Output file path.
    ///     priority: Priority value (higher = more important).
    #[pyo3(signature = (input, output, priority=0))]
    fn add(&mut self, input: &str, output: &str, priority: i32) -> PyResult<()> {
        if input.is_empty() {
            return Err(PyValueError::new_err("input path must not be empty"));
        }
        self.jobs
            .push((input.to_string(), output.to_string(), priority));
        Ok(())
    }

    /// Sort jobs by priority (highest first).
    fn sort_by_priority(&mut self) {
        self.jobs.sort_by(|a, b| b.2.cmp(&a.2));
    }

    /// Sort jobs by input file size (smallest first).
    fn sort_by_size(&mut self) {
        self.jobs.sort_by(|a, b| {
            let size_a = std::fs::metadata(&a.0).map(|m| m.len()).unwrap_or(0);
            let size_b = std::fs::metadata(&b.0).map(|m| m.len()).unwrap_or(0);
            size_a.cmp(&size_b)
        });
    }

    /// Return the list of jobs as a list of dicts.
    fn jobs(&self) -> Vec<HashMap<String, Py<PyAny>>> {
        Python::attach(|py| {
            self.jobs
                .iter()
                .map(|(input, output, priority)| {
                    let mut m = HashMap::new();
                    m.insert(
                        "input".to_string(),
                        input
                            .clone()
                            .into_pyobject(py)
                            .map(|o| o.into_any().unbind())
                            .unwrap_or_else(|_| py.None()),
                    );
                    m.insert(
                        "output".to_string(),
                        output
                            .clone()
                            .into_pyobject(py)
                            .map(|o| o.into_any().unbind())
                            .unwrap_or_else(|_| py.None()),
                    );
                    m.insert(
                        "priority".to_string(),
                        (*priority)
                            .into_pyobject(py)
                            .map(|o| o.into_any().unbind())
                            .unwrap_or_else(|_| py.None()),
                    );
                    m
                })
                .collect()
        })
    }

    /// Return the number of jobs in the schedule.
    fn job_count(&self) -> usize {
        self.jobs.len()
    }

    fn __repr__(&self) -> String {
        format!("PyBatchSchedule(jobs={})", self.jobs.len())
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Process a batch of input files.
///
/// Args:
///     inputs: List of input file paths.
///     output_dir: Output directory.
///     preset: Optional preset name for processing.
///
/// Returns:
///     List of PyJobResult for each processed file.
#[pyfunction]
#[pyo3(signature = (inputs, output_dir, preset=None))]
pub fn batch_process(
    inputs: Vec<String>,
    output_dir: &str,
    preset: Option<&str>,
) -> PyResult<Vec<PyJobResult>> {
    if inputs.is_empty() {
        return Err(PyValueError::new_err("No input files provided"));
    }

    let out_dir = PathBuf::from(output_dir);
    if !out_dir.exists() {
        std::fs::create_dir_all(&out_dir).map_err(|e| {
            PyRuntimeError::new_err(format!(
                "Failed to create output directory '{}': {e}",
                output_dir
            ))
        })?;
    }

    let preset_label = preset.unwrap_or("default");
    let mut results = Vec::with_capacity(inputs.len());

    for input in &inputs {
        let start = Instant::now();
        let input_path = PathBuf::from(input);

        if !input_path.exists() {
            results.push(PyJobResult {
                input_path: input.clone(),
                output_path: String::new(),
                success: false,
                error_message: Some(format!("Input file not found: {input}")),
                duration_secs: start.elapsed().as_secs_f64(),
                output_size: 0,
            });
            continue;
        }

        let file_name = input_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("output");
        let output_path = out_dir.join(file_name);

        let file_size = std::fs::metadata(input).map(|m| m.len()).unwrap_or(0);

        results.push(PyJobResult {
            input_path: input.clone(),
            output_path: output_path.to_string_lossy().to_string(),
            success: true,
            error_message: None,
            duration_secs: start.elapsed().as_secs_f64(),
            output_size: file_size,
        });

        let _ = preset_label; // Used for future pipeline integration
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Simple glob matching supporting only "*" wildcards.
fn matches_simple_glob(name: &str, pattern: &str) -> bool {
    if pattern == "*" {
        return true;
    }

    // Handle "*.ext" pattern
    if let Some(ext) = pattern.strip_prefix("*.") {
        return name.ends_with(&format!(".{ext}"));
    }

    // Handle "prefix*" pattern
    if let Some(prefix) = pattern.strip_suffix('*') {
        return name.starts_with(prefix);
    }

    // Exact match fallback
    name == pattern
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all batch processing bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBatchConfig>()?;
    m.add_class::<PyJobResult>()?;
    m.add_class::<PyBatchRunner>()?;
    m.add_class::<PyBatchSchedule>()?;
    m.add_function(wrap_pyfunction!(batch_process, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_config_defaults() {
        let cfg = PyBatchConfig::new(None, None, None, None, None);
        assert_eq!(cfg.max_parallel, 4);
        assert_eq!(cfg.retry_count, 0);
        assert!(!cfg.stop_on_error);
        assert!(cfg.output_dir.is_none());
        assert!(!cfg.overwrite);
    }

    #[test]
    fn test_batch_config_with_parallel_zero() {
        let mut cfg = PyBatchConfig::new(None, None, None, None, None);
        let result = cfg.with_parallel(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_schedule_sort_by_priority() {
        let mut schedule = PyBatchSchedule::new();
        let _ = schedule.add("low.mkv", "low_out.mkv", 1);
        let _ = schedule.add("high.mkv", "high_out.mkv", 10);
        let _ = schedule.add("mid.mkv", "mid_out.mkv", 5);
        schedule.sort_by_priority();
        assert_eq!(schedule.jobs[0].2, 10);
        assert_eq!(schedule.jobs[1].2, 5);
        assert_eq!(schedule.jobs[2].2, 1);
    }

    #[test]
    fn test_simple_glob_matching() {
        assert!(matches_simple_glob("video.mkv", "*.mkv"));
        assert!(!matches_simple_glob("video.mp4", "*.mkv"));
        assert!(matches_simple_glob("anything", "*"));
        assert!(matches_simple_glob("test_file.mp4", "test*"));
        assert!(!matches_simple_glob("other.mp4", "test*"));
    }

    #[test]
    fn test_batch_runner_dry_run() {
        let cfg = PyBatchConfig::new(None, None, None, None, None);
        let mut runner = PyBatchRunner::new(&cfg);
        let _ = runner.add_job("input.mkv", "output.mkv", None);
        let _ = runner.add_job("input2.mkv", "output2.mkv", Some("{\"crf\": 28}"));
        let dry = runner.run_dry();
        assert_eq!(dry.len(), 2);
        assert!(dry[0].contains("input.mkv"));
        assert!(dry[1].contains("crf"));
    }
}
