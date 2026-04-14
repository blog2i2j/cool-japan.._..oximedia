#![allow(dead_code)]
//! Pickle serialization support for OxiMedia Python types.
//!
//! Provides `__getstate__` / `__setstate__` implementations for serializable
//! types so that Python's `pickle` module can serialize and deserialize them.
//!
//! # Supported types
//!
//! - [`PyPickleEncoderConfig`] — encoder configuration (codec, resolution, CRF, etc.)
//! - [`PyPickleQualityScore`] — quality assessment result (PSNR, SSIM, overall)
//! - [`PyPickleMediaMetadata`] — basic media file metadata
//!
//! # Example (Python)
//!
//! ```python
//! import pickle, oximedia
//!
//! cfg = oximedia.PyPickleEncoderConfig(codec="av1", width=1920, height=1080, crf=28.0)
//! data = pickle.dumps(cfg)
//! cfg2 = pickle.loads(data)
//! assert cfg2.width == 1920
//! ```

use pyo3::prelude::*;
use pyo3::types::PyDict;

// ---------------------------------------------------------------------------
// PyPickleEncoderConfig
// ---------------------------------------------------------------------------

/// A picklable encoder configuration.
///
/// This type supports Python `pickle.dumps()` / `pickle.loads()` via the
/// `__getstate__` / `__setstate__` protocol.
#[pyclass]
#[derive(Debug, Clone, PartialEq)]
pub struct PyPickleEncoderConfig {
    /// Video codec name (e.g. "av1", "vp9").
    #[pyo3(get, set)]
    pub codec: String,
    /// Frame width in pixels.
    #[pyo3(get, set)]
    pub width: u32,
    /// Frame height in pixels.
    #[pyo3(get, set)]
    pub height: u32,
    /// Constant rate factor (quality).
    #[pyo3(get, set)]
    pub crf: f64,
    /// Framerate numerator.
    #[pyo3(get, set)]
    pub fps_num: u32,
    /// Framerate denominator.
    #[pyo3(get, set)]
    pub fps_den: u32,
    /// Encoding preset name.
    #[pyo3(get, set)]
    pub preset: String,
}

#[pymethods]
impl PyPickleEncoderConfig {
    #[new]
    #[pyo3(signature = (codec="av1", width=1920, height=1080, crf=28.0, fps_num=30, fps_den=1, preset="medium"))]
    pub fn new(
        codec: &str,
        width: u32,
        height: u32,
        crf: f64,
        fps_num: u32,
        fps_den: u32,
        preset: &str,
    ) -> Self {
        Self {
            codec: codec.to_string(),
            width,
            height,
            crf,
            fps_num,
            fps_den,
            preset: preset.to_string(),
        }
    }

    /// Serialize state for pickle.
    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("codec", &self.codec)?;
        dict.set_item("width", self.width)?;
        dict.set_item("height", self.height)?;
        dict.set_item("crf", self.crf)?;
        dict.set_item("fps_num", self.fps_num)?;
        dict.set_item("fps_den", self.fps_den)?;
        dict.set_item("preset", &self.preset)?;
        Ok(dict)
    }

    /// Restore state from pickle.
    fn __setstate__(&mut self, state: &Bound<'_, PyDict>) -> PyResult<()> {
        self.codec = state
            .get_item("codec")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("codec"))?
            .extract()?;
        self.width = state
            .get_item("width")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("width"))?
            .extract()?;
        self.height = state
            .get_item("height")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("height"))?
            .extract()?;
        self.crf = state
            .get_item("crf")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("crf"))?
            .extract()?;
        self.fps_num = state
            .get_item("fps_num")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("fps_num"))?
            .extract()?;
        self.fps_den = state
            .get_item("fps_den")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("fps_den"))?
            .extract()?;
        self.preset = state
            .get_item("preset")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("preset"))?
            .extract()?;
        Ok(())
    }

    /// Pickle protocol: return (class, args, state).
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String, u32, u32, f64, u32, u32, String),)> {
        let cls = py.get_type::<Self>().into_any().unbind();
        Ok((
            cls,
            (
                self.codec.clone(),
                self.width,
                self.height,
                self.crf,
                self.fps_num,
                self.fps_den,
                self.preset.clone(),
            ),
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "PyPickleEncoderConfig(codec={:?}, {}x{}, crf={:.1}, {}/{} fps, preset={:?})",
            self.codec, self.width, self.height, self.crf, self.fps_num, self.fps_den, self.preset
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Total number of pixels per frame.
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Framerate as a float.
    pub fn fps(&self) -> f64 {
        if self.fps_den == 0 {
            0.0
        } else {
            f64::from(self.fps_num) / f64::from(self.fps_den)
        }
    }
}

// ---------------------------------------------------------------------------
// PyPickleQualityScore
// ---------------------------------------------------------------------------

/// A picklable quality assessment result.
#[pyclass]
#[derive(Debug, Clone, PartialEq)]
pub struct PyPickleQualityScore {
    /// PSNR value in dB.
    #[pyo3(get, set)]
    pub psnr: f64,
    /// SSIM value (0.0 to 1.0).
    #[pyo3(get, set)]
    pub ssim: f64,
    /// Overall quality score (0.0 to 100.0).
    #[pyo3(get, set)]
    pub overall: f64,
    /// Quality grade label.
    #[pyo3(get, set)]
    pub grade: String,
}

#[pymethods]
impl PyPickleQualityScore {
    #[new]
    #[pyo3(signature = (psnr=0.0, ssim=0.0, overall=0.0, grade="unknown"))]
    pub fn new(psnr: f64, ssim: f64, overall: f64, grade: &str) -> Self {
        Self {
            psnr,
            ssim,
            overall,
            grade: grade.to_string(),
        }
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("psnr", self.psnr)?;
        dict.set_item("ssim", self.ssim)?;
        dict.set_item("overall", self.overall)?;
        dict.set_item("grade", &self.grade)?;
        Ok(dict)
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyDict>) -> PyResult<()> {
        self.psnr = state
            .get_item("psnr")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("psnr"))?
            .extract()?;
        self.ssim = state
            .get_item("ssim")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("ssim"))?
            .extract()?;
        self.overall = state
            .get_item("overall")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("overall"))?
            .extract()?;
        self.grade = state
            .get_item("grade")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("grade"))?
            .extract()?;
        Ok(())
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (f64, f64, f64, String))> {
        let cls = py.get_type::<Self>().into_any().unbind();
        Ok((cls, (self.psnr, self.ssim, self.overall, self.grade.clone())))
    }

    fn __repr__(&self) -> String {
        format!(
            "PyPickleQualityScore(psnr={:.2}, ssim={:.4}, overall={:.1}, grade={:?})",
            self.psnr, self.ssim, self.overall, self.grade
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Whether PSNR exceeds a given threshold.
    pub fn psnr_above(&self, threshold: f64) -> bool {
        self.psnr > threshold
    }

    /// Whether SSIM exceeds a given threshold.
    pub fn ssim_above(&self, threshold: f64) -> bool {
        self.ssim > threshold
    }
}

// ---------------------------------------------------------------------------
// PyPickleMediaMetadata
// ---------------------------------------------------------------------------

/// Picklable media file metadata.
#[pyclass]
#[derive(Debug, Clone, PartialEq)]
pub struct PyPickleMediaMetadata {
    /// File path.
    #[pyo3(get, set)]
    pub path: String,
    /// Duration in seconds.
    #[pyo3(get, set)]
    pub duration_secs: f64,
    /// Container format.
    #[pyo3(get, set)]
    pub container: String,
    /// Video codec.
    #[pyo3(get, set)]
    pub video_codec: String,
    /// Audio codec.
    #[pyo3(get, set)]
    pub audio_codec: String,
    /// Frame width.
    #[pyo3(get, set)]
    pub width: u32,
    /// Frame height.
    #[pyo3(get, set)]
    pub height: u32,
    /// Sample rate.
    #[pyo3(get, set)]
    pub sample_rate: u32,
    /// Audio channels.
    #[pyo3(get, set)]
    pub channels: u32,
}

#[pymethods]
impl PyPickleMediaMetadata {
    #[new]
    #[pyo3(signature = (
        path="",
        duration_secs=0.0,
        container="",
        video_codec="",
        audio_codec="",
        width=0,
        height=0,
        sample_rate=0,
        channels=0
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        path: &str,
        duration_secs: f64,
        container: &str,
        video_codec: &str,
        audio_codec: &str,
        width: u32,
        height: u32,
        sample_rate: u32,
        channels: u32,
    ) -> Self {
        Self {
            path: path.to_string(),
            duration_secs,
            container: container.to_string(),
            video_codec: video_codec.to_string(),
            audio_codec: audio_codec.to_string(),
            width,
            height,
            sample_rate,
            channels,
        }
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("path", &self.path)?;
        dict.set_item("duration_secs", self.duration_secs)?;
        dict.set_item("container", &self.container)?;
        dict.set_item("video_codec", &self.video_codec)?;
        dict.set_item("audio_codec", &self.audio_codec)?;
        dict.set_item("width", self.width)?;
        dict.set_item("height", self.height)?;
        dict.set_item("sample_rate", self.sample_rate)?;
        dict.set_item("channels", self.channels)?;
        Ok(dict)
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyDict>) -> PyResult<()> {
        self.path = state
            .get_item("path")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("path"))?
            .extract()?;
        self.duration_secs = state
            .get_item("duration_secs")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("duration_secs"))?
            .extract()?;
        self.container = state
            .get_item("container")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("container"))?
            .extract()?;
        self.video_codec = state
            .get_item("video_codec")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("video_codec"))?
            .extract()?;
        self.audio_codec = state
            .get_item("audio_codec")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("audio_codec"))?
            .extract()?;
        self.width = state
            .get_item("width")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("width"))?
            .extract()?;
        self.height = state
            .get_item("height")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("height"))?
            .extract()?;
        self.sample_rate = state
            .get_item("sample_rate")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("sample_rate"))?
            .extract()?;
        self.channels = state
            .get_item("channels")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("channels"))?
            .extract()?;
        Ok(())
    }

    fn __reduce__(
        &self,
        py: Python<'_>,
    ) -> PyResult<(Py<PyAny>, (String, f64, String, String, String, u32, u32, u32, u32))> {
        let cls = py.get_type::<Self>().into_any().unbind();
        Ok((
            cls,
            (
                self.path.clone(),
                self.duration_secs,
                self.container.clone(),
                self.video_codec.clone(),
                self.audio_codec.clone(),
                self.width,
                self.height,
                self.sample_rate,
                self.channels,
            ),
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "PyPickleMediaMetadata(path={:?}, {:.1}s, {}x{}, {} + {})",
            self.path, self.duration_secs, self.width, self.height,
            self.video_codec, self.audio_codec
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Whether this metadata describes a video stream.
    pub fn has_video(&self) -> bool {
        self.width > 0 && self.height > 0 && !self.video_codec.is_empty()
    }

    /// Whether this metadata describes an audio stream.
    pub fn has_audio(&self) -> bool {
        self.sample_rate > 0 && self.channels > 0 && !self.audio_codec.is_empty()
    }

    /// Pixel count.
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register pickle-support types into the parent module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPickleEncoderConfig>()?;
    m.add_class::<PyPickleQualityScore>()?;
    m.add_class::<PyPickleMediaMetadata>()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_config_new_defaults() {
        let cfg = PyPickleEncoderConfig::new("av1", 1920, 1080, 28.0, 30, 1, "medium");
        assert_eq!(cfg.codec, "av1");
        assert_eq!(cfg.width, 1920);
        assert_eq!(cfg.height, 1080);
        assert!((cfg.crf - 28.0).abs() < f64::EPSILON);
        assert_eq!(cfg.fps_num, 30);
        assert_eq!(cfg.fps_den, 1);
        assert_eq!(cfg.preset, "medium");
    }

    #[test]
    fn test_encoder_config_pixel_count() {
        let cfg = PyPickleEncoderConfig::new("vp9", 3840, 2160, 30.0, 60, 1, "fast");
        assert_eq!(cfg.pixel_count(), 3840 * 2160);
    }

    #[test]
    fn test_encoder_config_fps() {
        let cfg = PyPickleEncoderConfig::new("av1", 1920, 1080, 28.0, 30000, 1001, "medium");
        let fps = cfg.fps();
        assert!((fps - 29.97).abs() < 0.01);
    }

    #[test]
    fn test_encoder_config_fps_zero_den() {
        let cfg = PyPickleEncoderConfig::new("av1", 1920, 1080, 28.0, 30, 0, "medium");
        assert!((cfg.fps()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_encoder_config_repr() {
        let cfg = PyPickleEncoderConfig::new("av1", 1920, 1080, 28.0, 30, 1, "medium");
        let repr = cfg.__repr__();
        assert!(repr.contains("av1"));
        assert!(repr.contains("1920"));
        assert!(repr.contains("1080"));
    }

    #[test]
    fn test_encoder_config_clone_eq() {
        let cfg = PyPickleEncoderConfig::new("av1", 1920, 1080, 28.0, 30, 1, "medium");
        let cfg2 = cfg.clone();
        assert_eq!(cfg, cfg2);
    }

    #[test]
    fn test_quality_score_new() {
        let qs = PyPickleQualityScore::new(40.5, 0.95, 85.0, "good");
        assert!((qs.psnr - 40.5).abs() < f64::EPSILON);
        assert!((qs.ssim - 0.95).abs() < f64::EPSILON);
        assert!((qs.overall - 85.0).abs() < f64::EPSILON);
        assert_eq!(qs.grade, "good");
    }

    #[test]
    fn test_quality_score_thresholds() {
        let qs = PyPickleQualityScore::new(40.0, 0.95, 80.0, "good");
        assert!(qs.psnr_above(35.0));
        assert!(!qs.psnr_above(45.0));
        assert!(qs.ssim_above(0.9));
        assert!(!qs.ssim_above(0.99));
    }

    #[test]
    fn test_quality_score_repr() {
        let qs = PyPickleQualityScore::new(40.5, 0.95, 85.0, "good");
        let repr = qs.__repr__();
        assert!(repr.contains("40.50"));
        assert!(repr.contains("0.9500"));
        assert!(repr.contains("good"));
    }

    #[test]
    fn test_quality_score_clone_eq() {
        let qs = PyPickleQualityScore::new(40.5, 0.95, 85.0, "good");
        let qs2 = qs.clone();
        assert_eq!(qs, qs2);
    }

    #[test]
    fn test_media_metadata_new() {
        let md = PyPickleMediaMetadata::new(
            "test.mkv", 120.0, "matroska", "av1", "opus",
            1920, 1080, 48000, 2,
        );
        assert_eq!(md.path, "test.mkv");
        assert!((md.duration_secs - 120.0).abs() < f64::EPSILON);
        assert_eq!(md.container, "matroska");
    }

    #[test]
    fn test_media_metadata_has_video() {
        let md = PyPickleMediaMetadata::new(
            "test.mkv", 10.0, "matroska", "av1", "",
            1920, 1080, 0, 0,
        );
        assert!(md.has_video());
        assert!(!md.has_audio());
    }

    #[test]
    fn test_media_metadata_has_audio() {
        let md = PyPickleMediaMetadata::new(
            "test.ogg", 10.0, "ogg", "", "opus",
            0, 0, 48000, 2,
        );
        assert!(!md.has_video());
        assert!(md.has_audio());
    }

    #[test]
    fn test_media_metadata_pixel_count() {
        let md = PyPickleMediaMetadata::new(
            "test.mkv", 10.0, "matroska", "av1", "opus",
            3840, 2160, 48000, 2,
        );
        assert_eq!(md.pixel_count(), 3840 * 2160);
    }

    #[test]
    fn test_media_metadata_repr() {
        let md = PyPickleMediaMetadata::new(
            "test.mkv", 120.5, "matroska", "av1", "opus",
            1920, 1080, 48000, 2,
        );
        let repr = md.__repr__();
        assert!(repr.contains("test.mkv"));
        assert!(repr.contains("120.5"));
        assert!(repr.contains("1920"));
    }

    #[test]
    fn test_media_metadata_clone_eq() {
        let md = PyPickleMediaMetadata::new(
            "test.mkv", 10.0, "matroska", "av1", "opus",
            1920, 1080, 48000, 2,
        );
        let md2 = md.clone();
        assert_eq!(md, md2);
    }

    #[test]
    fn test_media_metadata_no_streams() {
        let md = PyPickleMediaMetadata::new("", 0.0, "", "", "", 0, 0, 0, 0);
        assert!(!md.has_video());
        assert!(!md.has_audio());
        assert_eq!(md.pixel_count(), 0);
    }
}
