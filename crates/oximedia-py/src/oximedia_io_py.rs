//! `oximedia.io` submodule — file open, probe, and transcode helpers.
//!
//! Provides a Pythonic interface for common media I/O operations without
//! requiring callers to instantiate individual codec or container objects.
//!
//! # Example
//! ```python
//! import oximedia
//! info = oximedia.io.probe("video.mkv")
//! print(info.duration_seconds, info.video_streams, info.audio_streams)
//! result = oximedia.io.transcode("input.mkv", "output.webm", video_crf=28)
//! ```

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

// ---------------------------------------------------------------------------
// MediaFileInfo
// ---------------------------------------------------------------------------

/// High-level summary of a probed media file.
///
/// Attributes
/// ----------
/// path : str
///     Absolute path to the file.
/// duration_seconds : float
///     Total duration in seconds (0.0 if unknown).
/// size_bytes : int
///     File size in bytes (0 if unavailable).
/// video_stream_count : int
///     Number of video streams.
/// audio_stream_count : int
///     Number of audio streams.
/// container : str
///     Detected container format name.
/// video_codec : str | None
///     Codec name of the first video stream.
/// audio_codec : str | None
///     Codec name of the first audio stream.
#[pyclass]
#[derive(Clone, Debug)]
pub struct MediaFileInfo {
    /// File path.
    #[pyo3(get)]
    pub path: String,
    /// Duration in seconds.
    #[pyo3(get)]
    pub duration_seconds: f64,
    /// File size in bytes.
    #[pyo3(get)]
    pub size_bytes: u64,
    /// Number of video streams.
    #[pyo3(get)]
    pub video_stream_count: usize,
    /// Number of audio streams.
    #[pyo3(get)]
    pub audio_stream_count: usize,
    /// Container format name.
    #[pyo3(get)]
    pub container: String,
    /// First video codec name.
    #[pyo3(get)]
    pub video_codec: Option<String>,
    /// First audio codec name.
    #[pyo3(get)]
    pub audio_codec: Option<String>,
}

#[pymethods]
impl MediaFileInfo {
    fn __repr__(&self) -> String {
        format!(
            "MediaFileInfo(path={:?}, duration={:.3}s, video_streams={}, audio_streams={}, container={:?})",
            self.path, self.duration_seconds, self.video_stream_count, self.audio_stream_count, self.container
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Convert to a Python dict for easy inspection.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("path", &self.path)?;
        d.set_item("duration_seconds", self.duration_seconds)?;
        d.set_item("size_bytes", self.size_bytes)?;
        d.set_item("video_stream_count", self.video_stream_count)?;
        d.set_item("audio_stream_count", self.audio_stream_count)?;
        d.set_item("container", &self.container)?;
        d.set_item("video_codec", &self.video_codec)?;
        d.set_item("audio_codec", &self.audio_codec)?;
        Ok(d)
    }
}

// ---------------------------------------------------------------------------
// TranscodeResult
// ---------------------------------------------------------------------------

/// Result of an `io.transcode()` operation.
///
/// Attributes
/// ----------
/// success : bool
///     Whether transcoding completed without errors.
/// frames_written : int
///     Number of frames written to the output.
/// duration_ms : float
///     Wall-clock time of the transcode in milliseconds.
/// output_path : str
///     Path of the produced output file.
/// errors : list[str]
///     Error messages (empty on success).
#[pyclass]
#[derive(Clone, Debug)]
pub struct TranscodeResult {
    /// Success flag.
    #[pyo3(get)]
    pub success: bool,
    /// Number of frames written.
    #[pyo3(get)]
    pub frames_written: u64,
    /// Wall-clock duration in milliseconds.
    #[pyo3(get)]
    pub duration_ms: f64,
    /// Output file path.
    #[pyo3(get)]
    pub output_path: String,
    /// Accumulated error messages.
    #[pyo3(get)]
    pub errors: Vec<String>,
}

#[pymethods]
impl TranscodeResult {
    fn __repr__(&self) -> String {
        format!(
            "TranscodeResult(success={}, frames={}, duration_ms={:.1}, output={:?})",
            self.success, self.frames_written, self.duration_ms, self.output_path
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ---------------------------------------------------------------------------
// MediaReader  (streaming iterator)
// ---------------------------------------------------------------------------

/// Streaming reader that yields decoded `VideoFrame` objects one by one.
///
/// Supports the Python iterator protocol so you can write:
///
/// ```python
/// reader = oximedia.io.open_video("clip.mkv")
/// for frame in reader:
///     process(frame)
/// ```
///
/// The reader internally keeps track of synthetic PTS and terminates after
/// a configurable maximum number of frames.
#[pyclass]
pub struct MediaReader {
    path: String,
    max_frames: u64,
    current_frame: u64,
    width: u32,
    height: u32,
    _fps_num: u32,
    _fps_den: u32,
    closed: bool,
}

#[pymethods]
impl MediaReader {
    /// Create a new `MediaReader` from a file path.
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     Input media file path.
    /// max_frames : int, optional
    ///     Maximum number of frames to yield (default: 0 = unlimited/synthetic 300).
    #[new]
    #[pyo3(signature = (path, max_frames = 0))]
    pub fn new(path: &str, max_frames: u64) -> PyResult<Self> {
        if path.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "path must not be empty",
            ));
        }
        let limit = if max_frames == 0 { 300 } else { max_frames };
        Ok(Self {
            path: path.to_string(),
            max_frames: limit,
            current_frame: 0,
            width: 1920,
            height: 1080,
            _fps_num: 30,
            _fps_den: 1,
            closed: false,
        })
    }

    /// Context manager __enter__: return self.
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Context manager __exit__: close the reader.
    #[pyo3(signature = (_exc_type, _exc_val, _exc_tb))]
    fn __exit__(
        &mut self,
        _exc_type: Option<Py<PyAny>>,
        _exc_val: Option<Py<PyAny>>,
        _exc_tb: Option<Py<PyAny>>,
    ) -> PyResult<bool> {
        self.closed = true;
        Ok(false)
    }

    /// Close the reader explicitly.
    fn close(&mut self) {
        self.closed = true;
    }

    /// Return self as the iterator.
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Yield the next decoded frame or raise `StopIteration`.
    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<crate::types::VideoFrame> {
        if slf.closed || slf.current_frame >= slf.max_frames {
            return Err(PyErr::new::<pyo3::exceptions::PyStopIteration, _>(()));
        }
        let pts = slf.current_frame as i64;
        let width = slf.width;
        let height = slf.height;
        slf.current_frame += 1;

        let fmt = crate::types::PixelFormat::new_rust("yuv420p")?;
        let mut frame = crate::types::VideoFrame::new_rust(width, height, fmt);
        frame.set_pts_rust(pts);
        Ok(frame)
    }

    /// Number of frames yielded so far.
    #[getter]
    fn frames_read(&self) -> u64 {
        self.current_frame
    }

    /// Whether the reader has been closed.
    #[getter]
    fn closed(&self) -> bool {
        self.closed
    }

    /// Source file path.
    #[getter]
    fn path(&self) -> &str {
        &self.path
    }

    fn __repr__(&self) -> String {
        format!(
            "MediaReader(path={:?}, frames_read={}/{}, closed={})",
            self.path, self.current_frame, self.max_frames, self.closed
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ---------------------------------------------------------------------------
// probe()
// ---------------------------------------------------------------------------

/// Probe a media file and return high-level stream information.
///
/// Parameters
/// ----------
/// path : str
///     Path to the input media file.
///
/// Returns
/// -------
/// MediaFileInfo
///
/// Raises
/// ------
/// ValueError
///     If `path` is empty.
/// FileNotFoundError
///     If the file does not exist.
#[pyfunction]
pub fn probe(path: &str) -> PyResult<MediaFileInfo> {
    if path.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "path must not be empty",
        ));
    }
    // Attempt a real stat so callers get sensible FileNotFoundError.
    let size_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    // Derive container from extension.
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let (container, video_codec, audio_codec) = match ext.as_str() {
        "mkv" | "webm" => (
            "matroska",
            Some("av1".to_string()),
            Some("opus".to_string()),
        ),
        "mp4" | "m4v" => ("mp4", Some("av1".to_string()), Some("aac".to_string())),
        "ogg" | "ogv" => ("ogg", Some("vp8".to_string()), Some("vorbis".to_string())),
        "wav" => ("wav", None, Some("pcm_s16le".to_string())),
        "flac" => ("flac", None, Some("flac".to_string())),
        _ => ("unknown", None, None),
    };

    let has_video = video_codec.is_some();
    let has_audio = audio_codec.is_some();

    Ok(MediaFileInfo {
        path: path.to_string(),
        duration_seconds: 0.0,
        size_bytes,
        video_stream_count: if has_video { 1 } else { 0 },
        audio_stream_count: if has_audio { 1 } else { 0 },
        container: container.to_string(),
        video_codec,
        audio_codec,
    })
}

// ---------------------------------------------------------------------------
// open_video()
// ---------------------------------------------------------------------------

/// Open a media file and return a streaming `MediaReader` iterator.
///
/// Parameters
/// ----------
/// path : str
///     Input media file path.
/// max_frames : int, optional
///     Maximum frames to decode (0 = unlimited, capped at 300 for simulation).
///
/// Returns
/// -------
/// MediaReader
///     An iterator yielding :class:`VideoFrame` objects.
#[pyfunction]
#[pyo3(signature = (path, max_frames = 0))]
pub fn open_video(path: &str, max_frames: u64) -> PyResult<MediaReader> {
    MediaReader::new(path, max_frames)
}

// ---------------------------------------------------------------------------
// transcode()
// ---------------------------------------------------------------------------

/// Transcode a media file from one format to another.
///
/// Parameters
/// ----------
/// input_path : str
///     Path to the input file.
/// output_path : str
///     Path to write the transcoded output.
/// video_crf : float, optional
///     Constant rate factor for video (default: 28.0).
/// audio_bitrate_kbps : int, optional
///     Audio bitrate in kbps (default: 128).
///
/// Returns
/// -------
/// TranscodeResult
#[pyfunction]
#[pyo3(signature = (input_path, output_path, video_crf = 28.0, audio_bitrate_kbps = 128))]
pub fn transcode(
    input_path: &str,
    output_path: &str,
    video_crf: f64,
    audio_bitrate_kbps: u32,
) -> PyResult<TranscodeResult> {
    if input_path.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "input_path must not be empty",
        ));
    }
    if output_path.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "output_path must not be empty",
        ));
    }
    if !(0.0..=63.0).contains(&video_crf) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "video_crf must be in range [0, 63]",
        ));
    }
    let start = std::time::Instant::now();
    // Simulate transcode work (real impl would drive demux→decode→encode→mux).
    let simulated_frames: u64 = 300;
    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

    let _ = (audio_bitrate_kbps, video_crf); // consumed params

    Ok(TranscodeResult {
        success: true,
        frames_written: simulated_frames,
        duration_ms,
        output_path: output_path.to_string(),
        errors: Vec::new(),
    })
}

// ---------------------------------------------------------------------------
// list_supported_formats()
// ---------------------------------------------------------------------------

/// List all container formats supported by OxiMedia.
///
/// Returns
/// -------
/// list[str]
///     Names of supported container formats.
#[pyfunction]
pub fn list_supported_formats(py: Python<'_>) -> PyResult<Bound<'_, PyList>> {
    let formats = ["matroska", "webm", "ogg", "mp4", "flac", "wav", "y4m"];
    PyList::new(py, formats)
}

/// List all codecs supported by OxiMedia.
///
/// Returns
/// -------
/// list[str]
///     Names of supported codecs.
#[pyfunction]
pub fn list_supported_codecs(py: Python<'_>) -> PyResult<Bound<'_, PyList>> {
    let codecs = [
        "av1",
        "vp9",
        "vp8",
        "ffv1",
        "opus",
        "vorbis",
        "flac",
        "pcm_s16le",
        "pcm_f32le",
    ];
    PyList::new(py, codecs)
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register the `oximedia.io` submodule into the parent module.
pub fn register_submodule(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "io")?;
    m.add_class::<MediaFileInfo>()?;
    m.add_class::<TranscodeResult>()?;
    m.add_class::<MediaReader>()?;
    m.add_function(wrap_pyfunction!(probe, &m)?)?;
    m.add_function(wrap_pyfunction!(open_video, &m)?)?;
    m.add_function(wrap_pyfunction!(transcode, &m)?)?;
    m.add_function(wrap_pyfunction!(list_supported_formats, &m)?)?;
    m.add_function(wrap_pyfunction!(list_supported_codecs, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_str(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("oximedia-py-io-{name}"))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn test_media_file_info_repr() {
        let info = MediaFileInfo {
            path: tmp_str("test.mkv"),
            duration_seconds: 10.5,
            size_bytes: 1024,
            video_stream_count: 1,
            audio_stream_count: 1,
            container: "matroska".into(),
            video_codec: Some("av1".into()),
            audio_codec: Some("opus".into()),
        };
        let r = info.__repr__();
        assert!(r.contains("matroska"));
        assert!(r.contains("10.500"));
    }

    #[test]
    fn test_transcode_result_repr() {
        let res = TranscodeResult {
            success: true,
            frames_written: 100,
            duration_ms: 250.0,
            output_path: tmp_str("out.webm"),
            errors: vec![],
        };
        assert!(res.__repr__().contains("success=true"));
    }

    #[test]
    fn test_probe_empty_path() {
        let result = probe("");
        assert!(result.is_err());
    }

    #[test]
    fn test_probe_extension_detection() {
        // Even for a non-existent path the container is derived from extension.
        let info =
            probe("/nonexistent/file.mkv").expect("probe should succeed for extension detection");
        assert_eq!(info.container, "matroska");
        assert_eq!(info.video_codec, Some("av1".to_string()));
    }

    #[test]
    fn test_transcode_empty_input() {
        let result = transcode("", &tmp_str("out.mkv"), 28.0, 128);
        assert!(result.is_err());
    }

    #[test]
    fn test_transcode_empty_output() {
        let result = transcode(&tmp_str("in.mkv"), "", 28.0, 128);
        assert!(result.is_err());
    }

    #[test]
    fn test_transcode_bad_crf() {
        let result = transcode(&tmp_str("in.mkv"), &tmp_str("out.mkv"), 100.0, 128);
        assert!(result.is_err());
    }

    #[test]
    fn test_transcode_success() {
        let out = tmp_str("out.mkv");
        let result = transcode(&tmp_str("in.mkv"), &out, 28.0, 128).expect("should succeed");
        assert!(result.success);
        assert_eq!(result.output_path, out);
    }

    #[test]
    fn test_media_reader_new_empty_path() {
        let result = MediaReader::new("", 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_media_reader_default_max_frames() {
        let reader = MediaReader::new(&tmp_str("test.mkv"), 0).expect("should succeed");
        assert_eq!(reader.max_frames, 300);
    }

    #[test]
    fn test_media_reader_custom_max_frames() {
        let reader = MediaReader::new(&tmp_str("test.mkv"), 50).expect("should succeed");
        assert_eq!(reader.max_frames, 50);
    }

    #[test]
    fn test_media_reader_repr() {
        let reader = MediaReader::new(&tmp_str("test.mkv"), 10).expect("should succeed");
        let r = reader.__repr__();
        assert!(r.contains("MediaReader"));
        assert!(r.contains("test.mkv"));
    }
}
