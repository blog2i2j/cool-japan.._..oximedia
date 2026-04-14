//! `oximedia.benchmark` submodule — Python-accessible performance profiling.
//!
//! Exposes timer, throughput counter, and benchmark-suite types so Python code
//! can measure and report the performance of OxiMedia operations.
//!
//! # Example
//! ```python
//! import oximedia
//! timer = oximedia.benchmark.Timer("encode_av1")
//! timer.start()
//! # ... encoding work ...
//! elapsed = timer.stop()
//! print(f"Elapsed: {elapsed:.3f} ms")
//!
//! suite = oximedia.benchmark.BenchmarkSuite("codec_comparison")
//! suite.add_result("av1_crf28", elapsed_ms=250.5, frames=300)
//! suite.add_result("vp9_crf28", elapsed_ms=180.2, frames=300)
//! suite.report()
//! ```

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Timer
// ---------------------------------------------------------------------------

/// High-resolution wall-clock timer for benchmarking Rust/Python operations.
///
/// Attributes
/// ----------
/// name : str
///     Human-readable label for this timer.
/// elapsed_ms : float
///     Total elapsed milliseconds after calling :meth:`stop`.
#[pyclass]
pub struct Timer {
    /// Timer name.
    #[pyo3(get)]
    pub name: String,
    started_at: Option<Instant>,
    /// Total elapsed time in milliseconds (populated after stop()).
    #[pyo3(get)]
    pub elapsed_ms: f64,
    lap_times: Vec<f64>,
    running: bool,
}

#[pymethods]
impl Timer {
    /// Create a new timer with the given name.
    #[new]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            started_at: None,
            elapsed_ms: 0.0,
            lap_times: Vec::new(),
            running: false,
        }
    }

    /// Start (or restart) the timer.
    ///
    /// Returns
    /// -------
    /// self
    ///     For method chaining.
    pub fn start(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.started_at = Some(Instant::now());
        slf.running = true;
        slf.elapsed_ms = 0.0;
        slf.lap_times.clear();
        slf
    }

    /// Stop the timer and return elapsed milliseconds.
    ///
    /// Returns
    /// -------
    /// float
    ///     Elapsed time in milliseconds.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the timer was never started.
    pub fn stop(&mut self) -> PyResult<f64> {
        let start = self.started_at.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Timer has not been started — call start() first",
            )
        })?;
        self.elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.running = false;
        Ok(self.elapsed_ms)
    }

    /// Record a lap time without stopping the timer.
    ///
    /// Returns
    /// -------
    /// float
    ///     Milliseconds since start (or last reset).
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the timer is not running.
    pub fn lap(&mut self) -> PyResult<f64> {
        let start = self.started_at.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Timer is not running — call start() first",
            )
        })?;
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        self.lap_times.push(ms);
        Ok(ms)
    }

    /// Return all recorded lap times in milliseconds.
    pub fn laps<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.lap_times.clone())
    }

    /// Whether the timer is currently running.
    #[getter]
    pub fn running(&self) -> bool {
        self.running
    }

    fn __repr__(&self) -> String {
        format!(
            "Timer(name={:?}, elapsed_ms={:.3}, running={})",
            self.name, self.elapsed_ms, self.running
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ---------------------------------------------------------------------------
// ThroughputCounter
// ---------------------------------------------------------------------------

/// Measures throughput (units per second) over a timed window.
///
/// Example
/// -------
/// ```python
/// counter = oximedia.benchmark.ThroughputCounter("frames_per_sec")
/// counter.start()
/// for frame in frames:
///     encode(frame)
///     counter.tick()
/// fps = counter.stop()
/// ```
#[pyclass]
pub struct ThroughputCounter {
    /// Counter name.
    #[pyo3(get)]
    pub name: String,
    ticks: u64,
    started_at: Option<Instant>,
}

#[pymethods]
impl ThroughputCounter {
    /// Create a new throughput counter.
    #[new]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ticks: 0,
            started_at: None,
        }
    }

    /// Start the counter (resets tick count).
    pub fn start(&mut self) {
        self.started_at = Some(Instant::now());
        self.ticks = 0;
    }

    /// Increment the tick count by `n` (default: 1).
    #[pyo3(signature = (n = 1))]
    pub fn tick(&mut self, n: u64) {
        self.ticks += n;
    }

    /// Stop the counter and return throughput in units per second.
    ///
    /// Returns
    /// -------
    /// float
    ///     Units per second.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the counter was never started.
    pub fn stop(&self) -> PyResult<f64> {
        let start = self.started_at.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Counter has not been started — call start() first",
            )
        })?;
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed <= 0.0 {
            return Ok(0.0);
        }
        Ok(self.ticks as f64 / elapsed)
    }

    /// Total ticks recorded since last start.
    #[getter]
    pub fn ticks(&self) -> u64 {
        self.ticks
    }

    /// Elapsed seconds since start (0 if not started).
    #[getter]
    pub fn elapsed_seconds(&self) -> f64 {
        self.started_at
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0)
    }

    fn __repr__(&self) -> String {
        format!(
            "ThroughputCounter(name={:?}, ticks={}, elapsed_s={:.3})",
            self.name,
            self.ticks,
            self.elapsed_seconds()
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ---------------------------------------------------------------------------
// BenchmarkResult
// ---------------------------------------------------------------------------

/// A single benchmark measurement entry.
///
/// Attributes
/// ----------
/// name : str
///     Run label.
/// elapsed_ms : float
///     Wall-clock time in milliseconds.
/// frames : int
///     Number of frames processed.
/// fps : float
///     Derived frames-per-second.
/// bytes_per_frame : float
///     Average bytes per frame (if size_bytes was provided, else 0).
#[pyclass]
#[derive(Clone, Debug)]
pub struct BenchmarkResult {
    /// Run name.
    #[pyo3(get)]
    pub name: String,
    /// Elapsed time in milliseconds.
    #[pyo3(get)]
    pub elapsed_ms: f64,
    /// Number of frames processed.
    #[pyo3(get)]
    pub frames: u64,
    /// Derived frames per second.
    #[pyo3(get)]
    pub fps: f64,
    /// Average bytes per frame.
    #[pyo3(get)]
    pub bytes_per_frame: f64,
}

#[pymethods]
impl BenchmarkResult {
    fn __repr__(&self) -> String {
        format!(
            "BenchmarkResult(name={:?}, elapsed_ms={:.1}, frames={}, fps={:.1})",
            self.name, self.elapsed_ms, self.frames, self.fps
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Convert to a Python dict.
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("name", &self.name)?;
        d.set_item("elapsed_ms", self.elapsed_ms)?;
        d.set_item("frames", self.frames)?;
        d.set_item("fps", self.fps)?;
        d.set_item("bytes_per_frame", self.bytes_per_frame)?;
        Ok(d)
    }
}

// ---------------------------------------------------------------------------
// BenchmarkSuite
// ---------------------------------------------------------------------------

/// A collection of benchmark results with summary statistics.
///
/// Example
/// -------
/// ```python
/// suite = oximedia.benchmark.BenchmarkSuite("codec_comparison")
/// suite.add_result("av1_crf28", elapsed_ms=250.5, frames=300)
/// suite.add_result("vp9_crf28", elapsed_ms=180.2, frames=300)
/// for r in suite.results():
///     print(r.name, r.fps)
/// print(suite.fastest().name)
/// ```
#[pyclass]
pub struct BenchmarkSuite {
    /// Suite name.
    #[pyo3(get)]
    pub name: String,
    results: Vec<BenchmarkResult>,
}

#[pymethods]
impl BenchmarkSuite {
    /// Create a new benchmark suite.
    #[new]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            results: Vec::new(),
        }
    }

    /// Add a benchmark result to the suite.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Run label.
    /// elapsed_ms : float
    ///     Wall-clock time in milliseconds.
    /// frames : int
    ///     Number of frames processed.
    /// size_bytes : int, optional
    ///     Total output size in bytes (for bytes_per_frame calculation).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If name is empty or elapsed_ms ≤ 0.
    #[pyo3(signature = (name, elapsed_ms, frames, size_bytes = 0))]
    pub fn add_result(
        &mut self,
        name: &str,
        elapsed_ms: f64,
        frames: u64,
        size_bytes: u64,
    ) -> PyResult<()> {
        if name.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "name must not be empty",
            ));
        }
        if elapsed_ms <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "elapsed_ms must be > 0",
            ));
        }
        let fps = if elapsed_ms > 0.0 {
            frames as f64 / (elapsed_ms / 1000.0)
        } else {
            0.0
        };
        let bytes_per_frame = if frames > 0 {
            size_bytes as f64 / frames as f64
        } else {
            0.0
        };
        self.results.push(BenchmarkResult {
            name: name.to_string(),
            elapsed_ms,
            frames,
            fps,
            bytes_per_frame,
        });
        Ok(())
    }

    /// Return all results.
    pub fn results(&self) -> Vec<BenchmarkResult> {
        self.results.clone()
    }

    /// Return the result with the highest FPS (fastest).
    ///
    /// Returns
    /// -------
    /// BenchmarkResult | None
    pub fn fastest(&self) -> Option<BenchmarkResult> {
        self.results
            .iter()
            .max_by(|a, b| {
                a.fps
                    .partial_cmp(&b.fps)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
    }

    /// Return the result with the lowest FPS (slowest).
    ///
    /// Returns
    /// -------
    /// BenchmarkResult | None
    pub fn slowest(&self) -> Option<BenchmarkResult> {
        self.results
            .iter()
            .min_by(|a, b| {
                a.fps
                    .partial_cmp(&b.fps)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
    }

    /// Mean FPS across all results.
    #[getter]
    pub fn mean_fps(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.results.iter().map(|r| r.fps).sum();
        sum / self.results.len() as f64
    }

    /// Number of results recorded.
    #[getter]
    pub fn count(&self) -> usize {
        self.results.len()
    }

    /// Print a human-readable report to stdout.
    pub fn report(&self) {
        println!("=== BenchmarkSuite: {} ===", self.name);
        for r in &self.results {
            println!(
                "  {:30} | {:8.1} ms | {:6} frames | {:7.1} fps",
                r.name, r.elapsed_ms, r.frames, r.fps
            );
        }
        if let Some(f) = self.fastest() {
            println!("  Fastest: {}", f.name);
        }
        if let Some(s) = self.slowest() {
            println!("  Slowest: {}", s.name);
        }
        println!("  Mean FPS: {:.1}", self.mean_fps());
    }

    /// Export all results as a list of dicts.
    pub fn to_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items: Vec<Bound<'py, PyDict>> = self
            .results
            .iter()
            .map(|r| r.to_dict(py))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, items)
    }

    fn __repr__(&self) -> String {
        format!(
            "BenchmarkSuite(name={:?}, results={})",
            self.name,
            self.results.len()
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ---------------------------------------------------------------------------
// time_call  (convenience function)
// ---------------------------------------------------------------------------

/// Time a callable and return ``(result, elapsed_ms)``.
///
/// Parameters
/// ----------
/// callable : Callable[[], T]
///     Any Python callable to benchmark.
///
/// Returns
/// -------
/// tuple[object, float]
///     ``(return_value, elapsed_ms)``
#[pyfunction]
pub fn time_call<'py>(
    _py: Python<'py>,
    callable: &Bound<'py, PyAny>,
) -> PyResult<(Py<PyAny>, f64)> {
    let start = Instant::now();
    let result = callable.call0()?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok((result.into(), elapsed_ms))
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register the `oximedia.benchmark` submodule into the parent module.
pub fn register_submodule(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "benchmark")?;
    m.add_class::<Timer>()?;
    m.add_class::<ThroughputCounter>()?;
    m.add_class::<BenchmarkResult>()?;
    m.add_class::<BenchmarkSuite>()?;
    m.add_function(wrap_pyfunction!(time_call, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_timer_new() {
        let t = Timer::new("test");
        assert_eq!(t.name, "test");
        assert!(!t.running());
        assert_eq!(t.elapsed_ms, 0.0);
    }

    #[test]
    fn test_timer_stop_without_start() {
        let mut t = Timer::new("x");
        assert!(t.stop().is_err());
    }

    #[test]
    fn test_timer_lap_without_start() {
        let mut t = Timer::new("x");
        assert!(t.lap().is_err());
    }

    #[test]
    fn test_timer_repr() {
        let t = Timer::new("encode");
        let r = t.__repr__();
        assert!(r.contains("encode"));
    }

    #[test]
    fn test_throughput_counter_stop_without_start() {
        let c = ThroughputCounter::new("x");
        assert!(c.stop().is_err());
    }

    #[test]
    fn test_throughput_counter_tick_and_ticks() {
        let mut c = ThroughputCounter::new("frames");
        c.start();
        c.tick(10);
        c.tick(5);
        assert_eq!(c.ticks(), 15);
    }

    #[test]
    fn test_throughput_counter_elapsed() {
        let mut c = ThroughputCounter::new("x");
        c.start();
        thread::sleep(Duration::from_millis(5));
        assert!(c.elapsed_seconds() > 0.0);
    }

    #[test]
    fn test_throughput_counter_repr() {
        let c = ThroughputCounter::new("fps_counter");
        let r = c.__repr__();
        assert!(r.contains("fps_counter"));
    }

    #[test]
    fn test_benchmark_suite_add_result() {
        let mut suite = BenchmarkSuite::new("test_suite");
        suite
            .add_result("run1", 100.0, 300, 0)
            .expect("add valid result");
        assert_eq!(suite.count(), 1);
    }

    #[test]
    fn test_benchmark_suite_empty_name() {
        let mut suite = BenchmarkSuite::new("s");
        assert!(suite.add_result("", 100.0, 300, 0).is_err());
    }

    #[test]
    fn test_benchmark_suite_zero_elapsed() {
        let mut suite = BenchmarkSuite::new("s");
        assert!(suite.add_result("run", 0.0, 300, 0).is_err());
    }

    #[test]
    fn test_benchmark_suite_fastest_slowest() {
        let mut suite = BenchmarkSuite::new("comparison");
        suite
            .add_result("fast", 50.0, 300, 0)
            .expect("add fast result");
        suite
            .add_result("slow", 200.0, 300, 0)
            .expect("add slow result");
        let fastest = suite
            .fastest()
            .expect("fastest should exist with 2 results");
        let slowest = suite
            .slowest()
            .expect("slowest should exist with 2 results");
        assert_eq!(fastest.name, "fast");
        assert_eq!(slowest.name, "slow");
    }

    #[test]
    fn test_benchmark_suite_mean_fps() {
        let mut suite = BenchmarkSuite::new("s");
        // 300 frames / 1s = 300 fps; 300 frames / 2s = 150 fps; mean = 225
        suite.add_result("a", 1000.0, 300, 0).expect("add result a");
        suite.add_result("b", 2000.0, 300, 0).expect("add result b");
        let mean = suite.mean_fps();
        assert!((mean - 225.0).abs() < 0.1);
    }

    #[test]
    fn test_benchmark_suite_empty_fastest() {
        let suite = BenchmarkSuite::new("empty");
        assert!(suite.fastest().is_none());
        assert_eq!(suite.mean_fps(), 0.0);
    }

    #[test]
    fn test_benchmark_result_repr() {
        let r = BenchmarkResult {
            name: "test".into(),
            elapsed_ms: 100.0,
            frames: 300,
            fps: 3000.0,
            bytes_per_frame: 0.0,
        };
        assert!(r.__repr__().contains("test"));
    }

    #[test]
    fn test_benchmark_suite_repr() {
        let s = BenchmarkSuite::new("my_suite");
        assert!(s.__repr__().contains("my_suite"));
    }
}
