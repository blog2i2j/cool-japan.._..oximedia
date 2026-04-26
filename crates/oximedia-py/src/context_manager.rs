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
//! with oximedia.ManagedEncoder(1920, 1080) as enc:
//!     packet_bytes = enc.encode_frame(yuv_bytes, pts=0)
//! ```

use pyo3::prelude::*;

use oximedia_codec::{
    frame::{Plane, VideoFrame as RustVideoFrame},
    Av1Encoder, BitrateMode, EncoderConfig, EncoderPreset, VideoEncoder,
};
use oximedia_core::{CodecId, PixelFormat, Rational};

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

    /// Probe the media file and return basic information as a `key=value`
    /// string.
    ///
    /// The implementation derives what it can from the file path without
    /// performing actual I/O:
    ///
    /// - `path`   — the file path supplied at construction time.
    /// - `format` — the lower-cased file extension (e.g. `mkv`, `mp4`).
    ///              Falls back to `unknown` when no extension is present.
    /// - `is_open` — whether the decoder is currently open.
    ///
    /// Returns
    /// -------
    /// str
    ///     Comma-separated `key=value` pairs describing the media file.
    pub fn probe(&self) -> PyResult<String> {
        if !self.is_open {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Decoder is not open. Use as a context manager or call open() first.",
            ));
        }

        // Derive the container format from the file extension.
        let format = std::path::Path::new(&self.path)
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase())
            .unwrap_or_else(|| "unknown".to_string());

        Ok(format!(
            "path={path},format={format},is_open={is_open}",
            path = self.path,
            format = format,
            is_open = self.is_open,
        ))
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

/// Computes the expected YUV 4:2:0 planar byte length for a frame of the
/// given dimensions.  Returns `None` if the arithmetic would overflow.
///
/// Layout: Y plane (`width × height`) + U plane (`⌈w/2⌉ × ⌈h/2⌉`) +
/// V plane (`⌈w/2⌉ × ⌈h/2⌉`).
fn yuv420p_byte_len(width: u32, height: u32) -> Option<usize> {
    let y_len = (width as usize).checked_mul(height as usize)?;
    let chroma_w = ((width as usize) + 1) / 2;
    let chroma_h = ((height as usize) + 1) / 2;
    let uv_len = chroma_w.checked_mul(chroma_h)?;
    y_len.checked_add(uv_len)?.checked_add(uv_len)
}

/// Build a [`RustVideoFrame`] from a flat YUV 4:2:0 planar byte slice.
///
/// The caller must supply exactly [`yuv420p_byte_len`] bytes.
fn frame_from_yuv420p(
    data: &[u8],
    width: u32,
    height: u32,
    pts: i64,
) -> Result<RustVideoFrame, String> {
    let expected = yuv420p_byte_len(width, height)
        .ok_or_else(|| "Frame dimensions overflow usize".to_string())?;

    if data.len() != expected {
        return Err(format!(
            "YUV420p frame data length mismatch: expected {expected} bytes \
             for {width}×{height}, got {}",
            data.len()
        ));
    }

    let y_len = (width as usize) * (height as usize);
    let chroma_w = ((width as usize) + 1) / 2;
    let chroma_h = ((height as usize) + 1) / 2;
    let uv_len = chroma_w * chroma_h;

    let y_plane = Plane::with_dimensions(data[..y_len].to_vec(), width as usize, width, height);
    let u_plane = Plane::with_dimensions(
        data[y_len..y_len + uv_len].to_vec(),
        chroma_w,
        chroma_w as u32,
        chroma_h as u32,
    );
    let v_plane = Plane::with_dimensions(
        data[y_len + uv_len..].to_vec(),
        chroma_w,
        chroma_w as u32,
        chroma_h as u32,
    );

    let mut frame = RustVideoFrame::new(PixelFormat::Yuv420p, width, height);
    frame.planes = vec![y_plane, u_plane, v_plane];
    frame.timestamp = oximedia_core::Timestamp::new(pts, Rational::new(1, 90_000));

    Ok(frame)
}

/// An encoder wrapper that supports the Python context-manager protocol.
///
/// Using `with oximedia.ManagedEncoder(width, height) as enc:` ensures that
/// the encoder is flushed and released even when an exception occurs.
///
/// Once opened, call [`encode_frame`][ManagedEncoder::encode_frame] with a
/// flat YUV 4:2:0 planar byte buffer to receive the concatenated AV1 packet
/// bytes for that frame.
///
/// Note: this type does not implement `Clone` because the underlying
/// [`Av1Encoder`] is not cloneable.
#[pyclass]
pub struct ManagedEncoder {
    /// Width of the encoded video in pixels.
    #[pyo3(get)]
    pub width: u32,
    /// Height of the encoded video in pixels.
    #[pyo3(get)]
    pub height: u32,
    /// Whether the encoder is currently initialised.
    #[pyo3(get)]
    pub is_open: bool,
    /// Number of frames encoded so far.
    #[pyo3(get)]
    pub frames_encoded: u64,
    /// Underlying AV1 encoder; `None` while the encoder is closed.
    encoder: Option<Av1Encoder>,
}

#[pymethods]
impl ManagedEncoder {
    /// Create a new managed encoder.
    ///
    /// Parameters
    /// ----------
    /// width : int
    ///     Output frame width in pixels (default 1920).
    /// height : int
    ///     Output frame height in pixels (default 1080).
    #[new]
    #[pyo3(signature = (width = 1920, height = 1080))]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            is_open: false,
            frames_encoded: 0,
            encoder: None,
        }
    }

    /// Initialise the underlying AV1 encoder.  Called automatically by
    /// `__enter__`.
    ///
    /// The encoder uses CRF 28, medium preset, and a 90 kHz timebase.
    pub fn open(&mut self) -> PyResult<()> {
        let config = EncoderConfig {
            codec: CodecId::Av1,
            width: self.width,
            height: self.height,
            pixel_format: PixelFormat::Yuv420p,
            framerate: Rational::new(30, 1),
            bitrate: BitrateMode::Crf(28.0),
            preset: EncoderPreset::Medium,
            profile: None,
            keyint: 250,
            threads: 0,
            timebase: Rational::new(1, 90_000),
        };

        let enc = Av1Encoder::new(config).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to initialise AV1 encoder: {e}"
            ))
        })?;

        self.encoder = Some(enc);
        self.is_open = true;
        Ok(())
    }

    /// Flush pending frames and release the encoder.  Called automatically by
    /// `__exit__`.
    pub fn close(&mut self) -> PyResult<()> {
        if let Some(ref mut enc) = self.encoder {
            // Best-effort flush; errors are ignored on close.
            let _ = enc.flush();
        }
        self.encoder = None;
        self.is_open = false;
        Ok(())
    }

    /// Encode one video frame of YUV 4:2:0 planar data.
    ///
    /// Parameters
    /// ----------
    /// frame_data : bytes
    ///     Flat YUV 4:2:0 planar byte buffer.  Must contain exactly
    ///     `width × height + 2 × ⌈width/2⌉ × ⌈height/2⌉` bytes (luma
    ///     plane followed by Cb and Cr half-size planes).
    /// pts : int
    ///     Presentation timestamp in 90 kHz ticks (default 0).
    ///
    /// Returns
    /// -------
    /// bytes
    ///     Concatenated AV1 packet data produced by the encoder for this
    ///     frame.  An empty byte string is possible when the encoder buffers
    ///     frames, though the pure-Rust AV1 implementation typically emits
    ///     one packet per frame.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the encoder is not open or if encoding fails.
    /// ValueError
    ///     If `frame_data` has the wrong length.
    #[pyo3(signature = (frame_data, pts = 0))]
    pub fn encode_frame(&mut self, frame_data: &[u8], pts: i64) -> PyResult<Vec<u8>> {
        if !self.is_open {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Encoder is not open. Use as a context manager or call open() first.",
            ));
        }

        let enc = self.encoder.as_mut().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Internal error: encoder slot is None while is_open is true.",
            )
        })?;

        // Build a VideoFrame from the raw planar bytes.
        let frame = frame_from_yuv420p(frame_data, self.width, self.height, pts)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

        // Push the frame into the encoder.
        enc.send_frame(&frame).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Encoder send_frame failed: {e}"
            ))
        })?;

        // Drain all available encoded packets and concatenate their data.
        let mut output = Vec::new();
        loop {
            match enc.receive_packet() {
                Ok(Some(pkt)) => output.extend_from_slice(&pkt.data),
                Ok(None) => break,
                Err(e) => {
                    return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Encoder receive_packet failed: {e}"
                    )));
                }
            }
        }

        self.frames_encoded += 1;
        Ok(output)
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
    fn test_managed_decoder_probe_contains_format() {
        let mut dec = ManagedDecoder::new("movie.mp4");
        dec.open().expect("open should succeed");
        let info = dec.probe().expect("probe should succeed");
        assert!(
            info.contains("format=mp4"),
            "probe should include format=mp4, got: {info}"
        );
        assert!(
            info.contains("path=movie.mp4"),
            "probe should include path, got: {info}"
        );
    }

    #[test]
    fn test_managed_decoder_probe_no_extension() {
        let mut dec = ManagedDecoder::new("media_file_no_ext");
        dec.open().expect("open should succeed");
        let info = dec.probe().expect("probe should succeed");
        assert!(
            info.contains("format=unknown"),
            "probe should fall back to format=unknown, got: {info}"
        );
    }

    #[test]
    fn test_managed_decoder_probe_uppercase_extension_normalised() {
        let mut dec = ManagedDecoder::new("clip.MKV");
        dec.open().expect("open should succeed");
        let info = dec.probe().expect("probe should succeed");
        assert!(
            info.contains("format=mkv"),
            "extension should be lower-cased in probe output, got: {info}"
        );
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
        let mut enc = ManagedEncoder::new(64, 64);
        let dummy = vec![0u8; yuv420p_byte_len(64, 64).unwrap()];
        assert!(enc.encode_frame(&dummy, 0).is_err());
    }

    /// Encoding two frames must increment `frames_encoded` to 2 and produce
    /// non-empty AV1 packet bytes for each frame.
    #[test]
    fn test_managed_encoder_frame_count_and_output() {
        let w = 64u32;
        let h = 64u32;
        let mut enc = ManagedEncoder::new(w, h);
        enc.open().expect("open should succeed");

        let frame_len = yuv420p_byte_len(w, h).unwrap();
        // Mid-grey YUV: Y=128, U=128, V=128.
        let frame_data = vec![128u8; frame_len];

        let pkt0 = enc
            .encode_frame(&frame_data, 0)
            .expect("first encode should succeed");
        let pkt1 = enc
            .encode_frame(&frame_data, 1)
            .expect("second encode should succeed");

        assert_eq!(enc.frames_encoded, 2);

        // The first frame is a keyframe and must carry a sequence-header OBU.
        assert!(!pkt0.is_empty(), "keyframe packet must be non-empty");
        assert!(!pkt1.is_empty(), "inter-frame packet must be non-empty");
    }

    /// Supplying a buffer with the wrong length must return an error.
    #[test]
    fn test_managed_encoder_bad_frame_size() {
        let mut enc = ManagedEncoder::new(64, 64);
        enc.open().expect("open should succeed");

        let bad = vec![0u8; 10]; // Far too short.
        assert!(
            enc.encode_frame(&bad, 0).is_err(),
            "encoding wrong-sized buffer must fail"
        );
    }

    // ----- yuv420p_byte_len unit tests -----

    #[test]
    fn test_yuv420p_byte_len_hd() {
        // 1920×1080: Y=2_073_600, U=960×540=518_400, V=518_400
        assert_eq!(
            yuv420p_byte_len(1920, 1080),
            Some(1920 * 1080 + 960 * 540 + 960 * 540)
        );
    }

    #[test]
    fn test_yuv420p_byte_len_small() {
        // 64×64: Y=4096, U=1024, V=1024 → 6144
        assert_eq!(yuv420p_byte_len(64, 64), Some(6144));
    }

    #[test]
    fn test_yuv420p_byte_len_odd_dims() {
        // 7×5: Y=35, U=⌈7/2⌉×⌈5/2⌉=4×3=12, V=12 → 59
        assert_eq!(yuv420p_byte_len(7, 5), Some(35 + 12 + 12));
    }

    // ----- frame_from_yuv420p unit tests -----

    #[test]
    fn test_frame_from_yuv420p_rejects_bad_length() {
        assert!(frame_from_yuv420p(&[], 64, 64, 0).is_err());
        let one_short = vec![0u8; yuv420p_byte_len(64, 64).unwrap() - 1];
        assert!(frame_from_yuv420p(&one_short, 64, 64, 0).is_err());
    }

    #[test]
    fn test_frame_from_yuv420p_accepts_correct_length() {
        let data = vec![0u8; yuv420p_byte_len(64, 64).unwrap()];
        assert!(frame_from_yuv420p(&data, 64, 64, 0).is_ok());
    }

    #[test]
    fn test_frame_from_yuv420p_pts_propagated() {
        let data = vec![0u8; yuv420p_byte_len(16, 16).unwrap()];
        let frame = frame_from_yuv420p(&data, 16, 16, 42).expect("should succeed");
        assert_eq!(frame.timestamp.pts, 42);
    }

    #[test]
    fn test_frame_from_yuv420p_plane_count() {
        let data = vec![0u8; yuv420p_byte_len(32, 32).unwrap()];
        let frame = frame_from_yuv420p(&data, 32, 32, 0).expect("should succeed");
        // YUV420p has 3 planes: Y, U, V.
        assert_eq!(frame.planes.len(), 3);
        // Y plane: 32×32 = 1024 bytes.
        assert_eq!(frame.planes[0].data.len(), 32 * 32);
        // U/V planes: 16×16 = 256 bytes each.
        assert_eq!(frame.planes[1].data.len(), 16 * 16);
        assert_eq!(frame.planes[2].data.len(), 16 * 16);
    }
}
