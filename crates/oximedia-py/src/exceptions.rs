//! Custom Python exception classes for the OxiMedia Python bindings.
//!
//! This module registers a hierarchy of typed exception classes into the
//! `oximedia` Python module so callers can catch them with precise `except`
//! clauses.
//!
//! # Exception hierarchy
//!
//! ```text
//! Exception
//! └── OxiError                   — base for all OxiMedia errors
//!     ├── OxiCodecError          — codec encode/decode failures
//!     ├── OxiContainerError      — container mux/demux failures
//!     ├── OxiIoError             — I/O and network errors
//!     ├── OxiQualityError        — quality assessment errors
//!     ├── OxiRightsError         — DRM / rights violations
//!     ├── OxiTimeoutError        — operation time-out
//!     └── OxiInvalidArgError     — invalid Python argument
//! ```

use pyo3::{create_exception, exceptions::PyException, prelude::*, types::PyModule, Bound};

// ── Exception declarations ────────────────────────────────────────────────

create_exception!(oximedia, OxiError, PyException,
    "Base exception for all OxiMedia errors.");

create_exception!(oximedia, OxiCodecError, OxiError,
    "Raised when a codec encode or decode operation fails.");

create_exception!(oximedia, OxiContainerError, OxiError,
    "Raised when a container mux or demux operation fails.");

create_exception!(oximedia, OxiIoError, OxiError,
    "Raised on I/O or network errors during media operations.");

create_exception!(oximedia, OxiQualityError, OxiError,
    "Raised when a quality assessment operation fails.");

create_exception!(oximedia, OxiRightsError, OxiError,
    "Raised on DRM or rights management violations.");

create_exception!(oximedia, OxiTimeoutError, OxiError,
    "Raised when an OxiMedia operation exceeds its time limit.");

create_exception!(oximedia, OxiInvalidArgError, OxiError,
    "Raised when an invalid argument is passed to an OxiMedia function.");

// ── Registration helper ───────────────────────────────────────────────────

/// Register all custom exception classes into `m`.
///
/// Call this from the top-level `#[pymodule]` initialiser:
/// ```rust,ignore
/// use oximedia_py::exceptions::register_exceptions;
/// register_exceptions(m)?;
/// ```
pub fn register_exceptions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("OxiError", m.py().get_type::<OxiError>())?;
    m.add("OxiCodecError", m.py().get_type::<OxiCodecError>())?;
    m.add("OxiContainerError", m.py().get_type::<OxiContainerError>())?;
    m.add("OxiIoError", m.py().get_type::<OxiIoError>())?;
    m.add("OxiQualityError", m.py().get_type::<OxiQualityError>())?;
    m.add("OxiRightsError", m.py().get_type::<OxiRightsError>())?;
    m.add("OxiTimeoutError", m.py().get_type::<OxiTimeoutError>())?;
    m.add("OxiInvalidArgError", m.py().get_type::<OxiInvalidArgError>())?;
    Ok(())
}

/// Convenience: wrap an error message as an [`OxiCodecError`].
pub fn codec_err(msg: impl Into<String>) -> PyErr {
    OxiCodecError::new_err(msg.into())
}

/// Convenience: wrap an error message as an [`OxiContainerError`].
pub fn container_err(msg: impl Into<String>) -> PyErr {
    OxiContainerError::new_err(msg.into())
}

/// Convenience: wrap an error message as an [`OxiIoError`].
pub fn io_err(msg: impl Into<String>) -> PyErr {
    OxiIoError::new_err(msg.into())
}

/// Convenience: wrap an error message as an [`OxiRightsError`].
pub fn rights_err(msg: impl Into<String>) -> PyErr {
    OxiRightsError::new_err(msg.into())
}

/// Convenience: wrap an error message as an [`OxiInvalidArgError`].
pub fn invalid_arg_err(msg: impl Into<String>) -> PyErr {
    OxiInvalidArgError::new_err(msg.into())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_error_constructors_exist() {
        // Verify that the convenience constructors compile and are callable.
        // Full PyErr assertions require an embedded Python interpreter,
        // so we just confirm the function signatures are correct.
        let _: fn(String) -> pyo3::PyErr = super::codec_err;
        let _: fn(String) -> pyo3::PyErr = super::container_err;
        let _: fn(String) -> pyo3::PyErr = super::io_err;
        let _: fn(String) -> pyo3::PyErr = super::rights_err;
        let _: fn(String) -> pyo3::PyErr = super::invalid_arg_err;
    }

    #[test]
    fn test_exception_module_compiles() {
        // Verify the exception types and registration fn exist at compile time.
        let _ = super::register_exceptions;
    }
}
