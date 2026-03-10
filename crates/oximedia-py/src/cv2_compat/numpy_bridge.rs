//! Internal image buffer utilities for the cv2_compat layer.
//!
//! Images are passed as raw byte buffers (via numpy's buffer protocol or bytes objects)
//! with separate shape parameters. This avoids requiring pyo3-numpy and works with
//! standard numpy arrays via the buffer protocol.

use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Representation of an image buffer extracted from Python.
#[allow(dead_code)]
pub struct ImageBuf {
    pub data: Vec<u8>,
    pub height: usize,
    pub width: usize,
    pub channels: usize,
}

#[allow(dead_code)]
impl ImageBuf {
    pub fn new(data: Vec<u8>, height: usize, width: usize, channels: usize) -> Self {
        Self {
            data,
            height,
            width,
            channels,
        }
    }

    /// Total byte size of the image.
    pub fn byte_len(&self) -> usize {
        self.height * self.width * self.channels
    }

    /// Get pixel at (row, col) — returns slice of `channels` bytes.
    pub fn pixel(&self, row: usize, col: usize) -> &[u8] {
        let off = (row * self.width + col) * self.channels;
        &self.data[off..off + self.channels]
    }

    /// Set pixel at (row, col).
    pub fn set_pixel(&mut self, row: usize, col: usize, val: &[u8]) {
        let off = (row * self.width + col) * self.channels;
        self.data[off..off + self.channels].copy_from_slice(val);
    }

    /// Convert to PyBytes.
    pub fn to_pybytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.data)
    }
}

/// Extract image data from a Python object.
///
/// Accepts:
/// - `bytes` / `bytearray` objects
/// - numpy arrays (via buffer protocol — calls `.tobytes()` internally)
/// - Python lists of ints
#[allow(dead_code)]
pub fn extract_image_bytes(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    // Try bytes directly
    if let Ok(b) = obj.cast::<PyBytes>() {
        return Ok(b.as_bytes().to_vec());
    }
    // Try buffer protocol (numpy arrays implement this)
    if let Ok(buf) = obj.call_method0("tobytes") {
        if let Ok(b) = buf.cast::<PyBytes>() {
            return Ok(b.as_bytes().to_vec());
        }
        // Try extracting as bytes-like via extract
        if let Ok(v) = buf.extract::<Vec<u8>>() {
            return Ok(v);
        }
    }
    // Try direct extract as Vec<u8>
    if let Ok(v) = obj.extract::<Vec<u8>>() {
        return Ok(v);
    }
    let _ = py;
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected image data as bytes, bytearray, or numpy array",
    ))
}

/// Extract shape tuple (H, W, C) or (H, W) from a Python object.
#[allow(dead_code)]
pub fn extract_shape(obj: &Bound<'_, PyAny>) -> PyResult<(usize, usize, usize)> {
    // Try (H, W, C) first
    if let Ok((h, w, c)) = obj.extract::<(usize, usize, usize)>() {
        return Ok((h, w, c));
    }
    // Try (H, W) — grayscale
    if let Ok((h, w)) = obj.extract::<(usize, usize)>() {
        return Ok((h, w, 1));
    }
    Err(pyo3::exceptions::PyValueError::new_err(
        "Expected shape as (H, W, C) or (H, W)",
    ))
}

/// Clamp a value to [0, 255] and cast to u8.
#[inline]
pub fn clamp_u8(v: f32) -> u8 {
    v.clamp(0.0, 255.0) as u8
}

/// Linear interpolation between two u8 values.
#[inline]
pub fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    let av = a as f32;
    let bv = b as f32;
    clamp_u8(av + (bv - av) * t)
}
