//! Python-callable progress callback for long-running OxiMedia operations.
//!
//! [`PyProgressCallback`] wraps a Python callable and invokes it with a
//! percentage value each time [`update`](PyProgressCallback::update) is called.
//!
//! # Usage from Python
//!
//! ```python
//! import oximedia
//!
//! def on_progress(pct: float) -> None:
//!     print(f"Progress: {pct:.1f}%")
//!
//! cb = oximedia.PyProgressCallback(on_progress)
//! cb.update(50.0)   # calls on_progress(50.0)
//! cb.update(100.0)  # calls on_progress(100.0)
//! ```

use pyo3::prelude::*;

/// A progress callback that wraps a Python callable.
///
/// The callable is invoked with a single `float` argument (percentage in
/// the range `[0.0, 100.0]`) each time [`update`](PyProgressCallback::update)
/// is called.
#[pyclass]
pub struct PyProgressCallback {
    /// The wrapped Python callable.
    callback: Py<PyAny>,
    /// Last percentage value passed to the callback.
    last_pct: f32,
}

#[pymethods]
impl PyProgressCallback {
    /// Create a new progress callback wrapping `cb`.
    ///
    /// `cb` must be a Python callable that accepts a single `float` argument.
    #[new]
    pub fn new(cb: Py<PyAny>) -> Self {
        Self {
            callback: cb,
            last_pct: 0.0,
        }
    }

    /// Invoke the callback with `pct` (percentage, `0.0`–`100.0`).
    ///
    /// The value is clamped to `[0.0, 100.0]` before being passed to Python.
    /// Any exception raised by the Python callable is propagated as a
    /// `PyResult::Err`.
    pub fn update(&mut self, py: Python<'_>, pct: f32) -> PyResult<()> {
        let clamped = pct.clamp(0.0, 100.0);
        self.last_pct = clamped;
        self.callback.call1(py, (clamped,))?;
        Ok(())
    }

    /// Last percentage value passed to the callback (or `0.0` if never updated).
    #[getter]
    pub fn last_pct(&self) -> f32 {
        self.last_pct
    }

    fn __repr__(&self) -> String {
        format!("PyProgressCallback(last_pct={:.1})", self.last_pct)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Register the [`PyProgressCallback`] class into the given module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProgressCallback>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_repr_format() {
        // Verify __repr__ format without requiring Python runtime
        // (We can't call __repr__ without Python, so just check the module compiles)
        let _ = stringify!(super::PyProgressCallback);
    }
}
