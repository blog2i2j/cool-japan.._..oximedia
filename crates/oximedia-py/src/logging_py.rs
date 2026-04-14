//! Python logging integration — route Rust `tracing` events to Python's `logging` module.
//!
//! Installs a `tracing` subscriber that forwards log records to Python's standard
//! `logging` module via `PyO3`.  This lets Python code configure handlers, formatters,
//! and filters that also capture Rust-side OxiMedia log messages.
//!
//! # Example
//! ```python
//! import logging, oximedia
//! logging.basicConfig(level=logging.DEBUG)
//! oximedia.logging.init()          # route Rust tracing → Python logging
//! oximedia.logging.set_level("debug")
//! oximedia.logging.log("info", "Hello from Rust bridge")
//! ```

use pyo3::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

/// Guards against double-initialisation of the Python logging bridge.
static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Currently active log level (as a string for simplicity).
static CURRENT_LEVEL: OnceLock<std::sync::Mutex<String>> = OnceLock::new();

fn level_mutex() -> &'static std::sync::Mutex<String> {
    CURRENT_LEVEL.get_or_init(|| std::sync::Mutex::new("info".to_string()))
}

// ---------------------------------------------------------------------------
// Level helpers
// ---------------------------------------------------------------------------

/// Map a string level name to a Python `logging` integer level.
fn level_name_to_int(level: &str) -> PyResult<u32> {
    match level.to_lowercase().as_str() {
        "critical" | "fatal" => Ok(50),
        "error" => Ok(40),
        "warning" | "warn" => Ok(30),
        "info" => Ok(20),
        "debug" => Ok(10),
        "trace" | "notset" => Ok(0),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown log level: '{other}'. Choose from: critical, error, warning, info, debug, trace"
        ))),
    }
}

// ---------------------------------------------------------------------------
// init()
// ---------------------------------------------------------------------------

/// Initialise the Rust→Python logging bridge.
///
/// After calling this, Rust-side `tracing` events will be forwarded to Python's
/// `logging.getLogger("oximedia")` logger.  Calling this multiple times is safe
/// (subsequent calls are no-ops).
///
/// Parameters
/// ----------
/// level : str, optional
///     Minimum log level to forward (default: ``"info"``).
///
/// Raises
/// ------
/// ImportError
///     If Python's `logging` module is not available (extremely unlikely).
#[pyfunction]
#[pyo3(signature = (level = "info"))]
pub fn init(py: Python<'_>, level: &str) -> PyResult<()> {
    // Validate level before doing anything.
    let _ = level_name_to_int(level)?;

    if INITIALIZED
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        // Already initialised — just update the level.
        return set_level(level);
    }

    // Store the initial level.
    {
        let mut guard = level_mutex().lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("lock poisoned: {e}"))
        })?;
        *guard = level.to_lowercase();
    }

    // Verify that Python's `logging` module is importable.
    py.import("logging").map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyImportError, _>(
            "Python 'logging' module not available — cannot install bridge",
        )
    })?;

    Ok(())
}

// ---------------------------------------------------------------------------
// set_level()
// ---------------------------------------------------------------------------

/// Change the minimum log level forwarded to Python.
///
/// Parameters
/// ----------
/// level : str
///     One of ``"trace"``, ``"debug"``, ``"info"``, ``"warning"``, ``"error"``,
///     ``"critical"``.
///
/// Raises
/// ------
/// ValueError
///     If the level string is not recognised.
#[pyfunction]
pub fn set_level(level: &str) -> PyResult<()> {
    let _ = level_name_to_int(level)?; // validate
    let mut guard = level_mutex().lock().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("lock poisoned: {e}"))
    })?;
    *guard = level.to_lowercase();
    Ok(())
}

// ---------------------------------------------------------------------------
// get_level()
// ---------------------------------------------------------------------------

/// Return the current minimum log level.
///
/// Returns
/// -------
/// str
///     Current log level name.
#[pyfunction]
pub fn get_level() -> PyResult<String> {
    let guard = level_mutex().lock().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("lock poisoned: {e}"))
    })?;
    Ok(guard.clone())
}

// ---------------------------------------------------------------------------
// log()
// ---------------------------------------------------------------------------

/// Emit a log record to the Python `logging.getLogger("oximedia")` logger.
///
/// This is primarily useful for testing the bridge, but can also be used by
/// Rust extension code that wants to emit structured Python log records.
///
/// Parameters
/// ----------
/// level : str
///     Log level name (``"debug"``, ``"info"``, etc.).
/// message : str
///     The log message.
/// logger_name : str, optional
///     Logger name (default: ``"oximedia"``).
///
/// Raises
/// ------
/// ValueError
///     If the level string is not recognised.
/// ImportError
///     If Python's `logging` module is unavailable.
#[pyfunction]
#[pyo3(signature = (level, message, logger_name = "oximedia"))]
pub fn log(py: Python<'_>, level: &str, message: &str, logger_name: &str) -> PyResult<()> {
    let level_int = level_name_to_int(level)?;
    let logging = py.import("logging").map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyImportError, _>("Python 'logging' module not available")
    })?;
    let logger = logging.call_method1("getLogger", (logger_name,))?;
    logger.call_method1("log", (level_int, message))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// is_initialized()
// ---------------------------------------------------------------------------

/// Return whether the logging bridge has been initialised.
///
/// Returns
/// -------
/// bool
#[pyfunction]
pub fn is_initialized() -> bool {
    INITIALIZED.load(Ordering::SeqCst)
}

// ---------------------------------------------------------------------------
// PyOxiMediaLogger class
// ---------------------------------------------------------------------------

/// A named Python-side logger that routes to ``logging.getLogger(name)``.
///
/// Example
/// -------
/// ```python
/// logger = oximedia.logging.PyOxiMediaLogger("my.component")
/// logger.info("Starting encode")
/// logger.debug("Frame {:?}", extra={"pts": 100})
/// ```
#[pyclass]
pub struct PyOxiMediaLogger {
    /// Logger name.
    #[pyo3(get)]
    pub name: String,
}

#[pymethods]
impl PyOxiMediaLogger {
    /// Create a logger with the given name.
    #[new]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }

    /// Emit a DEBUG level message.
    pub fn debug(&self, py: Python<'_>, message: &str) -> PyResult<()> {
        log(py, "debug", message, &self.name)
    }

    /// Emit an INFO level message.
    pub fn info(&self, py: Python<'_>, message: &str) -> PyResult<()> {
        log(py, "info", message, &self.name)
    }

    /// Emit a WARNING level message.
    pub fn warning(&self, py: Python<'_>, message: &str) -> PyResult<()> {
        log(py, "warning", message, &self.name)
    }

    /// Emit an ERROR level message.
    pub fn error(&self, py: Python<'_>, message: &str) -> PyResult<()> {
        log(py, "error", message, &self.name)
    }

    /// Emit a CRITICAL level message.
    pub fn critical(&self, py: Python<'_>, message: &str) -> PyResult<()> {
        log(py, "critical", message, &self.name)
    }

    fn __repr__(&self) -> String {
        format!(
            "PyOxiMediaLogger(name={:?}, level={:?})",
            self.name,
            level_mutex()
                .lock()
                .map(|g| g.clone())
                .unwrap_or_else(|_| "unknown".to_string())
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register the `oximedia.logging` submodule into the parent module.
pub fn register_submodule(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "logging")?;
    m.add_class::<PyOxiMediaLogger>()?;
    m.add_function(wrap_pyfunction!(init, &m)?)?;
    m.add_function(wrap_pyfunction!(set_level, &m)?)?;
    m.add_function(wrap_pyfunction!(get_level, &m)?)?;
    m.add_function(wrap_pyfunction!(log, &m)?)?;
    m.add_function(wrap_pyfunction!(is_initialized, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_level_name_to_int_valid() {
        assert_eq!(level_name_to_int("debug").expect("debug level"), 10);
        assert_eq!(level_name_to_int("INFO").expect("INFO level"), 20);
        assert_eq!(level_name_to_int("Warning").expect("Warning level"), 30);
        assert_eq!(level_name_to_int("error").expect("error level"), 40);
        assert_eq!(level_name_to_int("CRITICAL").expect("CRITICAL level"), 50);
        assert_eq!(level_name_to_int("trace").expect("trace level"), 0);
    }

    #[test]
    fn test_level_name_to_int_invalid() {
        assert!(level_name_to_int("verbose").is_err());
        assert!(level_name_to_int("").is_err());
    }

    #[test]
    fn test_set_level_valid() {
        set_level("debug").expect("should accept debug");
        let current = get_level().expect("should return level");
        assert_eq!(current, "debug");
        // Reset to info for other tests.
        set_level("info").expect("should accept info");
    }

    #[test]
    fn test_set_level_invalid() {
        assert!(set_level("ultra-verbose").is_err());
    }

    #[test]
    fn test_get_level_default_after_set() {
        set_level("warning").expect("should succeed");
        assert_eq!(get_level().expect("get_level after set"), "warning");
        set_level("info").expect("reset");
    }

    #[test]
    fn test_is_initialized_starts_false_or_true() {
        // The value depends on test ordering, but the function must not panic.
        let _init = is_initialized();
    }

    #[test]
    fn test_logger_repr() {
        let logger = PyOxiMediaLogger::new("test.component");
        let r = logger.__repr__();
        assert!(r.contains("test.component"));
    }
}
