//! Python context manager protocol (`__enter__` / `__exit__`) for resource types.
//!
//! Provides a reusable `PyContextManager` trait that resource-holding types can
//! implement, plus concrete wrappers for `PyDecoder` and `PyEncoder` stub types
//! demonstrating the pattern.
//!
//! # Example (Python)
//!
//! ```python
//! import oximedia
//!
//! with oximedia.ManagedDecoder("file.mkv") as dec:
//!     info = dec.probe()
//!
//! with oximedia.ManagedEncoder(config) as enc:
//!     enc.send_frame(frame)
//!     pkt = enc.receive_packet()
//! ```

use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// PyContextManager trait
// ---------------------------------------------------------------------------

/// Trait implemented by resource types that support the Python context-manager
/// protocol.  Types implementing this trait get `__enter__` / `__exit__`
/// forwarding for free via the blanket methods below.
///
/// The associated `Resource` type is the object returned by `__enter__`
/// (usually `self` cloned or a reference wrapper).
pub trait PyContextManager: Sized {
    /// Called on `__enter__`.  Returns a reference to (or clone of) the
    /// resource that will be bound to the `as` variable.
    fn on_enter(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>>;

    /// Called on `__exit__`.  The three optional exception arguments mirror
    /// the Python `__exit__` signature.  Return `true` to suppress the
    /// exception.
    fn on_exit(
        &mut self,
        py: Python<'_>,
        exc_type: Option<&Bound<'_, PyAny>>,
        exc_val: Option<&Bound<'_, PyAny>>,
        exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool>;
}

// ---------------------------------------------------------------------------
// ManagedDecoder — example resource wrapper
// ---------------------------------------------------------------------------

/// A decoder wrapper that supports the Python context-manager protocol.
///
/// Opening a media file for decoding with `with oximedia.ManagedDecoder(path)
/// as dec:` ensures `close()` is called automatically even when an exception
/// occurs.
#[pyclass]
#[derive(Clone)]
pub struct ManagedDecoder {
    /// File path.
    #[pyo3(get)]
    pub path: String,
    /// Whether the decoder is currently open.
    #[pyo3(get)]
    pub is_open: bool,
}

#[pymethods]
impl ManagedDecoder {
    /// Create a new managed decoder.
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     Path to the media file to decode.
    #[new]
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
            is_open: false,
        }
    }

    /// Open the media file.  Called automatically by `__enter__`.
    pub fn open(&mut self) -> PyResult<()> {
        self.is_open = true;
        Ok(())
    }

    /// Close the media file.  Called automatically by `__exit__`.
    pub fn close(&mut self) -> PyResult<()> {
        self.is_open = false;
        Ok(())
    }

    /// Probe the media file and return basic information.
    ///
    /// Returns
    /// -------
    /// str
    ///     Human-readable media info summary (stub).
    pub fn probe(&self) -> PyResult<String> {
        if !self.is_open {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Decoder is not open. Use as a context manager or call open() first.",
            ));
        }
        Ok(format!("MediaInfo(path={:?}, format=stub)", self.path))
    }

    /// Context manager entry: opens the decoder and returns self.
    fn __enter__(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        slf.open()?;
        Ok(slf)
    }

    /// Context manager exit: closes the decoder.
    ///
    /// The `exc_type`, `exc_val`, `exc_tb` arguments correspond to the Python
    /// exception context.  This implementation always returns `False` (does not
    /// suppress exceptions).
    #[pyo3(signature = (exc_type=None, exc_val=None, exc_tb=None))]
    fn __exit__(
        &mut self,
        _py: Python<'_>,
        exc_type: Option<&Bound<'_, PyAny>>,
        exc_val: Option<&Bound<'_, PyAny>>,
        exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        self.close()?;
        // Return False → do not suppress exceptions.
        let _ = (exc_type, exc_val, exc_tb);
        Ok(false)
    }

    fn __repr__(&self) -> String {
        format!(
            "ManagedDecoder(path={:?}, is_open={})",
            self.path, self.is_open
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ---------------------------------------------------------------------------
// ManagedEncoder — example resource wrapper
// ---------------------------------------------------------------------------

/// An encoder wrapper that supports the Python context-manager protocol.
///
/// Using `with oximedia.ManagedEncoder(config) as enc:` ensures that the
/// encoder is flushed and released even when an exception occurs.
#[pyclass]
#[derive(Clone)]
pub struct ManagedEncoder {
    /// Width of the encoded video.
    #[pyo3(get)]
    pub width: u32,
    /// Height of the encoded video.
    #[pyo3(get)]
    pub height: u32,
    /// Whether the encoder is currently initialised.
    #[pyo3(get)]
    pub is_open: bool,
    /// Number of frames encoded so far.
    #[pyo3(get)]
    pub frames_encoded: u64,
}

#[pymethods]
impl ManagedEncoder {
    /// Create a new managed encoder.
    ///
    /// Parameters
    /// ----------
    /// width : int
    ///     Output frame width in pixels.
    /// height : int
    ///     Output frame height in pixels.
    #[new]
    #[pyo3(signature = (width = 1920, height = 1080))]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            is_open: false,
            frames_encoded: 0,
        }
    }

    /// Initialise the encoder.  Called automatically by `__enter__`.
    pub fn open(&mut self) -> PyResult<()> {
        self.is_open = true;
        Ok(())
    }

    /// Flush pending frames and release resources.  Called automatically by
    /// `__exit__`.
    pub fn close(&mut self) -> PyResult<()> {
        self.is_open = false;
        Ok(())
    }

    /// Encode a frame (stub).
    ///
    /// Returns
    /// -------
    /// bytes
    ///     Encoded packet data (stub — returns empty bytes).
    pub fn encode_frame(&mut self) -> PyResult<Vec<u8>> {
        if !self.is_open {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Encoder is not open. Use as a context manager or call open() first.",
            ));
        }
        self.frames_encoded += 1;
        Ok(Vec::new())
    }

    /// Context manager entry: opens the encoder and returns self.
    fn __enter__(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        slf.open()?;
        Ok(slf)
    }

    /// Context manager exit: closes the encoder.
    ///
    /// Always returns `False` (does not suppress exceptions).
    #[pyo3(signature = (exc_type=None, exc_val=None, exc_tb=None))]
    fn __exit__(
        &mut self,
        _py: Python<'_>,
        exc_type: Option<&Bound<'_, PyAny>>,
        exc_val: Option<&Bound<'_, PyAny>>,
        exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        self.close()?;
        let _ = (exc_type, exc_val, exc_tb);
        Ok(false)
    }

    fn __repr__(&self) -> String {
        format!(
            "ManagedEncoder({}x{}, is_open={}, frames_encoded={})",
            self.width, self.height, self.is_open, self.frames_encoded
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register context-manager types into the parent module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ManagedDecoder>()?;
    m.add_class::<ManagedEncoder>()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_managed_decoder_open_close() {
        let mut dec = ManagedDecoder::new("test.mkv");
        assert!(!dec.is_open);
        dec.open().expect("open should succeed");
        assert!(dec.is_open);
        dec.close().expect("close should succeed");
        assert!(!dec.is_open);
    }

    #[test]
    fn test_managed_decoder_probe_when_closed() {
        let dec = ManagedDecoder::new("test.mkv");
        assert!(dec.probe().is_err());
    }

    #[test]
    fn test_managed_decoder_probe_when_open() {
        let mut dec = ManagedDecoder::new("test.mkv");
        dec.open().expect("open should succeed");
        assert!(dec.probe().is_ok());
    }

    #[test]
    fn test_managed_encoder_open_close() {
        let mut enc = ManagedEncoder::new(1920, 1080);
        assert!(!enc.is_open);
        enc.open().expect("open should succeed");
        assert!(enc.is_open);
        enc.close().expect("close should succeed");
        assert!(!enc.is_open);
    }

    #[test]
    fn test_managed_encoder_encode_when_closed() {
        let mut enc = ManagedEncoder::new(1920, 1080);
        assert!(enc.encode_frame().is_err());
    }

    #[test]
    fn test_managed_encoder_frame_count() {
        let mut enc = ManagedEncoder::new(1920, 1080);
        enc.open().expect("open should succeed");
        enc.encode_frame().expect("encode should succeed");
        enc.encode_frame().expect("encode should succeed");
        assert_eq!(enc.frames_encoded, 2);
    }
}
