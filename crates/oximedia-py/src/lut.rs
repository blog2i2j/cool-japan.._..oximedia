//! Python bindings for LUT (Look-Up Table) color grading operations.
//!
//! Wraps `oximedia-lut` to provide loading, generation, and application of
//! 3D LUTs from Python.  All pixel operations use tetrahedral interpolation
//! for accuracy.

use pyo3::prelude::*;

use oximedia_lut::{Lut3d, LutInterpolation, LutSize};

// ---------------------------------------------------------------------------
// PyLut3d class
// ---------------------------------------------------------------------------

/// A 3D color LUT (Look-Up Table) for color grading.
///
/// Encapsulates an oximedia `Lut3d` and exposes per-pixel and per-frame
/// apply operations to Python.
#[pyclass]
pub struct PyLut3d {
    inner: Lut3d,
}

#[pymethods]
impl PyLut3d {
    /// Apply the LUT to a single normalised RGB triplet.
    ///
    /// # Arguments
    /// * `r`, `g`, `b` - Input colour components in [0.0, 1.0]
    ///
    /// # Returns
    /// `(r, g, b)` tuple after LUT transform, values in [0.0, 1.0].
    pub fn apply_rgb(&self, r: f32, g: f32, b: f32) -> PyResult<(f32, f32, f32)> {
        let input = [r as f64, g as f64, b as f64];
        let out = self.inner.apply(&input, LutInterpolation::Tetrahedral);
        Ok((out[0] as f32, out[1] as f32, out[2] as f32))
    }

    /// Apply the LUT to raw RGB24 frame bytes.
    ///
    /// # Arguments
    /// * `data` - RGB24 bytes (any length that is a multiple of 3)
    ///
    /// # Returns
    /// RGB24 bytes with LUT applied.
    pub fn apply_to_frame(&self, data: Vec<u8>) -> PyResult<Vec<u8>> {
        apply_lut_to_bytes(&self.inner, data)
    }

    /// LUT size per dimension (e.g., 33 for a 33³ LUT).
    #[getter]
    pub fn size(&self) -> u32 {
        self.inner.size() as u32
    }

    fn __repr__(&self) -> String {
        format!("PyLut3d(size={})", self.inner.size())
    }
}

// ---------------------------------------------------------------------------
// Module-level functions
// ---------------------------------------------------------------------------

/// Load a 3D LUT from a `.cube` file.
///
/// # Arguments
/// * `path` - File-system path to a `.cube` LUT file
#[pyfunction]
pub fn load_lut(path: &str) -> PyResult<PyLut3d> {
    let lut = Lut3d::from_file(path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Failed to load LUT from '{}': {}", path, e))
    })?;
    Ok(PyLut3d { inner: lut })
}

/// Apply a `PyLut3d` to raw RGB24 frame bytes.
///
/// # Arguments
/// * `lut`  - A `PyLut3d` instance
/// * `data` - RGB24 bytes (length must be a multiple of 3)
///
/// # Returns
/// RGB24 bytes with the LUT applied.
#[pyfunction]
pub fn apply_lut(lut: &PyLut3d, data: Vec<u8>) -> PyResult<Vec<u8>> {
    apply_lut_to_bytes(&lut.inner, data)
}

/// Generate an identity 3D LUT that produces no colour change.
///
/// Useful as a starting point for custom LUT construction or testing.
///
/// # Arguments
/// * `size` - LUT size per dimension: 17, 33 (default), or 65
#[pyfunction]
#[pyo3(signature = (size=33))]
pub fn generate_identity_lut(size: u32) -> PyResult<PyLut3d> {
    let lut_size = LutSize::from(size as usize);
    let lut = Lut3d::identity(lut_size);
    Ok(PyLut3d { inner: lut })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Apply a `Lut3d` to a raw RGB24 byte buffer using tetrahedral interpolation.
fn apply_lut_to_bytes(lut: &Lut3d, data: Vec<u8>) -> PyResult<Vec<u8>> {
    if data.len() % 3 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "RGB24 data length {} is not a multiple of 3",
            data.len()
        )));
    }

    let pixel_count = data.len() / 3;
    let mut out = vec![0u8; data.len()];

    for i in 0..pixel_count {
        let r = data[i * 3] as f64 / 255.0;
        let g = data[i * 3 + 1] as f64 / 255.0;
        let b = data[i * 3 + 2] as f64 / 255.0;

        let result = lut.apply(&[r, g, b], LutInterpolation::Tetrahedral);

        out[i * 3] = (result[0].clamp(0.0, 1.0) * 255.0) as u8;
        out[i * 3 + 1] = (result[1].clamp(0.0, 1.0) * 255.0) as u8;
        out[i * 3 + 2] = (result[2].clamp(0.0, 1.0) * 255.0) as u8;
    }

    Ok(out)
}
