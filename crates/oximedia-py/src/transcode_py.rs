//! Python bindings for transcoding operations from `oximedia-transcode`.
//!
//! Provides a fluent transcoder API, ABR ladder management, preset listing,
//! codec enumeration, and configuration validation.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;

use oximedia_transcode::{QualityMode, TranscodeOutput, Transcoder};

// ---------------------------------------------------------------------------
// Helper: convert TranscodeError -> PyRuntimeError
// ---------------------------------------------------------------------------

fn tc_err(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(format!("{e}"))
}

fn val_err(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

// ---------------------------------------------------------------------------
// PyTranscodeResult
// ---------------------------------------------------------------------------

/// Result returned after a successful transcode operation.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyTranscodeResult {
    /// Output file size in bytes.
    #[pyo3(get)]
    pub file_size: u64,
    /// Media duration in seconds.
    #[pyo3(get)]
    pub duration_secs: f64,
    /// Effective bitrate in kilobits per second.
    #[pyo3(get)]
    pub bitrate_kbps: f64,
    /// Wall-clock encoding time in seconds.
    #[pyo3(get)]
    pub encoding_time_secs: f64,
    /// Ratio of input size to output size.
    #[pyo3(get)]
    pub compression_ratio: f64,
}

#[pymethods]
impl PyTranscodeResult {
    fn __repr__(&self) -> String {
        format!(
            "PyTranscodeResult(size={}, duration={:.2}s, bitrate={:.1}kbps, \
             encode_time={:.2}s, ratio={:.2})",
            self.file_size,
            self.duration_secs,
            self.bitrate_kbps,
            self.encoding_time_secs,
            self.compression_ratio,
        )
    }

    /// Return a plain dict representation.
    fn to_dict(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("file_size".to_string(), self.file_size as f64);
        m.insert("duration_secs".to_string(), self.duration_secs);
        m.insert("bitrate_kbps".to_string(), self.bitrate_kbps);
        m.insert("encoding_time_secs".to_string(), self.encoding_time_secs);
        m.insert("compression_ratio".to_string(), self.compression_ratio);
        m
    }
}

impl From<&TranscodeOutput> for PyTranscodeResult {
    fn from(out: &TranscodeOutput) -> Self {
        let bitrate_kbps = out.video_bitrate as f64 / 1000.0;
        let compression_ratio = if out.duration > 0.0 {
            (out.file_size as f64 * 8.0) / (out.video_bitrate as f64 * out.duration)
        } else {
            0.0
        };
        Self {
            file_size: out.file_size,
            duration_secs: out.duration,
            bitrate_kbps,
            encoding_time_secs: out.encoding_time,
            compression_ratio,
        }
    }
}

// ---------------------------------------------------------------------------
// PyAbrRung
// ---------------------------------------------------------------------------

/// A single rung (rendition) in an ABR ladder.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyAbrRung {
    /// Video width in pixels.
    #[pyo3(get)]
    pub width: u32,
    /// Video height in pixels.
    #[pyo3(get)]
    pub height: u32,
    /// Target bitrate in kbps.
    #[pyo3(get)]
    pub bitrate_kbps: u32,
    /// Video codec identifier.
    #[pyo3(get)]
    pub codec: String,
    /// Constant Rate Factor (optional).
    #[pyo3(get)]
    pub crf: Option<u32>,
}

#[pymethods]
impl PyAbrRung {
    /// Create a new ABR rung.
    #[new]
    #[pyo3(signature = (width, height, bitrate_kbps, codec = "av1", crf = None))]
    fn new(width: u32, height: u32, bitrate_kbps: u32, codec: &str, crf: Option<u32>) -> Self {
        Self {
            width,
            height,
            bitrate_kbps,
            codec: codec.to_string(),
            crf,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PyAbrRung({}x{}, {}kbps, codec='{}', crf={:?})",
            self.width, self.height, self.bitrate_kbps, self.codec, self.crf,
        )
    }

    /// Return a dict representation.
    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("width".to_string(), self.width.to_string());
        m.insert("height".to_string(), self.height.to_string());
        m.insert("bitrate_kbps".to_string(), self.bitrate_kbps.to_string());
        m.insert("codec".to_string(), self.codec.clone());
        if let Some(crf) = self.crf {
            m.insert("crf".to_string(), crf.to_string());
        }
        m
    }
}

// ---------------------------------------------------------------------------
// PyAbrLadder
// ---------------------------------------------------------------------------

/// Adaptive Bitrate ladder for multi-rendition encoding.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyAbrLadder {
    rungs: Vec<PyAbrRung>,
}

#[pymethods]
impl PyAbrLadder {
    /// Create an empty ABR ladder.
    #[new]
    fn new() -> Self {
        Self { rungs: Vec::new() }
    }

    /// Add a rung to the ladder.
    fn add_rung(&mut self, rung: PyAbrRung) {
        self.rungs.push(rung);
    }

    /// Return the list of rungs.
    fn rungs(&self) -> Vec<PyAbrRung> {
        self.rungs.clone()
    }

    /// Number of rungs in the ladder.
    fn rung_count(&self) -> usize {
        self.rungs.len()
    }

    /// Return a list-of-dicts representation suitable for JSON serialization.
    fn to_dict(&self) -> Vec<HashMap<String, String>> {
        self.rungs.iter().map(|r| r.to_dict()).collect()
    }

    /// Build a standard HLS ladder with sensible defaults.
    #[staticmethod]
    fn hls_default() -> Self {
        let entries: &[(u32, u32, u32, Option<u32>)] = &[
            (426, 240, 400, Some(36)),
            (640, 360, 800, Some(33)),
            (854, 480, 1400, Some(31)),
            (1280, 720, 2800, Some(28)),
            (1920, 1080, 5000, Some(25)),
        ];
        let rungs = entries
            .iter()
            .map(|&(w, h, br, crf)| PyAbrRung {
                width: w,
                height: h,
                bitrate_kbps: br,
                codec: "av1".to_string(),
                crf,
            })
            .collect();
        Self { rungs }
    }

    fn __repr__(&self) -> String {
        format!("PyAbrLadder(rungs={})", self.rungs.len())
    }
}

// ---------------------------------------------------------------------------
// PyPresetInfo
// ---------------------------------------------------------------------------

/// Metadata describing a transcoding preset.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyPresetInfo {
    /// Preset identifier.
    #[pyo3(get)]
    pub name: String,
    /// Human-readable description.
    #[pyo3(get)]
    pub description: String,
    /// Video codec used by this preset.
    #[pyo3(get)]
    pub video_codec: String,
    /// Audio codec used by this preset.
    #[pyo3(get)]
    pub audio_codec: String,
    /// Container format.
    #[pyo3(get)]
    pub container: String,
    /// Quality mode label.
    #[pyo3(get)]
    pub quality_mode: String,
}

#[pymethods]
impl PyPresetInfo {
    fn __repr__(&self) -> String {
        format!(
            "PyPresetInfo(name='{}', video='{}', audio='{}', container='{}')",
            self.name, self.video_codec, self.audio_codec, self.container,
        )
    }

    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("name".to_string(), self.name.clone());
        m.insert("description".to_string(), self.description.clone());
        m.insert("video_codec".to_string(), self.video_codec.clone());
        m.insert("audio_codec".to_string(), self.audio_codec.clone());
        m.insert("container".to_string(), self.container.clone());
        m.insert("quality_mode".to_string(), self.quality_mode.clone());
        m
    }
}

// ---------------------------------------------------------------------------
// PyTranscoder
// ---------------------------------------------------------------------------

/// Fluent transcoder interface mirroring `oximedia_transcode::Transcoder`.
#[pyclass]
pub struct PyTranscoder {
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    preset: Option<String>,
    video_codec: Option<String>,
    audio_codec: Option<String>,
    crf: Option<u32>,
    bitrate: Option<u32>,
    width: Option<u32>,
    height: Option<u32>,
    frame_rate: Option<f64>,
    audio_bitrate: Option<u32>,
    sample_rate: Option<u32>,
    channels: Option<u32>,
    two_pass: bool,
    quality_mode: String,
}

#[pymethods]
impl PyTranscoder {
    /// Create a new transcoder with default settings.
    #[new]
    fn new() -> Self {
        Self {
            input: None,
            output: None,
            preset: None,
            video_codec: None,
            audio_codec: None,
            crf: None,
            bitrate: None,
            width: None,
            height: None,
            frame_rate: None,
            audio_bitrate: None,
            sample_rate: None,
            channels: None,
            two_pass: false,
            quality_mode: "balanced".to_string(),
        }
    }

    /// Set the input file path.
    fn input(&mut self, path: &str) -> PyResult<()> {
        if path.is_empty() {
            return Err(val_err("Input path must not be empty"));
        }
        self.input = Some(PathBuf::from(path));
        Ok(())
    }

    /// Set the output file path.
    fn output(&mut self, path: &str) -> PyResult<()> {
        if path.is_empty() {
            return Err(val_err("Output path must not be empty"));
        }
        self.output = Some(PathBuf::from(path));
        Ok(())
    }

    /// Set a named preset (e.g. ``web_optimized``, ``archive_quality``).
    fn preset(&mut self, name: &str) -> PyResult<()> {
        let valid = [
            "web_optimized",
            "archive_quality",
            "fast_preview",
            "broadcast_hd",
            "social_media",
            "youtube_1080p",
            "vimeo_hd",
        ];
        if !valid.contains(&name) {
            return Err(val_err(format!(
                "Unknown preset '{}'. Valid presets: {}",
                name,
                valid.join(", ")
            )));
        }
        self.preset = Some(name.to_string());
        Ok(())
    }

    /// Set the video codec (e.g. ``av1``, ``vp9``, ``vp8``).
    fn video_codec(&mut self, codec: &str) -> PyResult<()> {
        let valid = ["av1", "vp9", "vp8"];
        if !valid.contains(&codec) {
            return Err(val_err(format!(
                "Unsupported video codec '{}'. Supported: {}",
                codec,
                valid.join(", ")
            )));
        }
        self.video_codec = Some(codec.to_string());
        Ok(())
    }

    /// Set the audio codec (e.g. ``opus``, ``vorbis``, ``flac``).
    fn audio_codec(&mut self, codec: &str) -> PyResult<()> {
        let valid = ["opus", "vorbis", "flac", "pcm"];
        if !valid.contains(&codec) {
            return Err(val_err(format!(
                "Unsupported audio codec '{}'. Supported: {}",
                codec,
                valid.join(", ")
            )));
        }
        self.audio_codec = Some(codec.to_string());
        Ok(())
    }

    /// Set the Constant Rate Factor for quality-based encoding.
    fn crf(&mut self, value: u32) -> PyResult<()> {
        if value > 63 {
            return Err(val_err("CRF must be in range 0..63"));
        }
        self.crf = Some(value);
        Ok(())
    }

    /// Set the target video bitrate in kbps.
    fn bitrate(&mut self, kbps: u32) -> PyResult<()> {
        if kbps == 0 {
            return Err(val_err("Bitrate must be > 0"));
        }
        self.bitrate = Some(kbps);
        Ok(())
    }

    /// Set the output resolution.
    fn scale(&mut self, width: u32, height: u32) -> PyResult<()> {
        if width == 0 || height == 0 {
            return Err(val_err("Width and height must be > 0"));
        }
        self.width = Some(width);
        self.height = Some(height);
        Ok(())
    }

    /// Set the output frame rate.
    fn frame_rate(&mut self, fps: f64) -> PyResult<()> {
        if fps <= 0.0 || fps > 240.0 {
            return Err(val_err("Frame rate must be in range (0, 240]"));
        }
        self.frame_rate = Some(fps);
        Ok(())
    }

    /// Set the audio bitrate in kbps.
    fn audio_bitrate(&mut self, kbps: u32) -> PyResult<()> {
        if kbps == 0 {
            return Err(val_err("Audio bitrate must be > 0"));
        }
        self.audio_bitrate = Some(kbps);
        Ok(())
    }

    /// Set the audio sample rate.
    fn sample_rate(&mut self, rate: u32) -> PyResult<()> {
        let valid = [8000, 16000, 22050, 44100, 48000, 96000, 192000];
        if !valid.contains(&rate) {
            return Err(val_err(format!(
                "Unusual sample rate {}. Common rates: {}",
                rate,
                valid
                    .iter()
                    .map(|r| r.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        self.sample_rate = Some(rate);
        Ok(())
    }

    /// Set the number of audio channels.
    fn channels(&mut self, ch: u32) -> PyResult<()> {
        if ch == 0 || ch > 8 {
            return Err(val_err("Channels must be in range 1..8"));
        }
        self.channels = Some(ch);
        Ok(())
    }

    /// Set the quality mode: ``fast``, ``balanced``, ``high``.
    fn quality_mode(&mut self, mode: &str) -> PyResult<()> {
        let valid = ["fast", "balanced", "high"];
        if !valid.contains(&mode) {
            return Err(val_err(format!(
                "Unknown quality mode '{}'. Valid: {}",
                mode,
                valid.join(", ")
            )));
        }
        self.quality_mode = mode.to_string();
        Ok(())
    }

    /// Enable or disable two-pass encoding.
    fn two_pass(&mut self, enable: bool) -> PyResult<()> {
        self.two_pass = enable;
        Ok(())
    }

    /// Execute the transcode operation.
    ///
    /// Blocks until the transcode completes and returns a ``PyTranscodeResult``.
    fn transcode(&self, py: Python<'_>) -> PyResult<PyTranscodeResult> {
        let input_str = self
            .input
            .as_ref()
            .ok_or_else(|| val_err("Input path not set"))?
            .to_string_lossy()
            .to_string();
        let output_str = self
            .output
            .as_ref()
            .ok_or_else(|| val_err("Output path not set"))?
            .to_string_lossy()
            .to_string();

        // Build the internal Transcoder
        let mut tc = Transcoder::new().input(&input_str).output(&output_str);

        if let Some(ref codec) = self.video_codec {
            tc = tc.video_codec(codec.as_str());
        }
        if let Some(ref codec) = self.audio_codec {
            tc = tc.audio_codec(codec.as_str());
        }
        if let Some(br) = self.bitrate {
            tc = tc.video_bitrate(u64::from(br) * 1000);
        }
        if let Some(abr) = self.audio_bitrate {
            tc = tc.audio_bitrate(u64::from(abr) * 1000);
        }
        if let Some(w) = self.width {
            let h = self.height.unwrap_or(w * 9 / 16);
            tc = tc.resolution(w, h);
        }
        if let Some(fps) = self.frame_rate {
            let num = (fps * 1000.0) as u32;
            tc = tc.frame_rate(num, 1000);
        }

        let qm = match self.quality_mode.as_str() {
            "fast" => QualityMode::Low,
            "high" => QualityMode::High,
            _ => QualityMode::Medium,
        };
        tc = tc.quality(qm);

        if self.two_pass {
            tc = tc.multi_pass(oximedia_transcode::MultiPassMode::TwoPass);
        }

        // Release the GIL and run the async transcode
        py.detach(|| {
            let rt = tokio::runtime::Runtime::new().map_err(tc_err)?;
            let output: TranscodeOutput = rt.block_on(tc.transcode()).map_err(tc_err)?;
            Ok(PyTranscodeResult::from(&output))
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PyTranscoder(input={:?}, output={:?}, video={:?}, audio={:?}, quality='{}')",
            self.input, self.output, self.video_codec, self.audio_codec, self.quality_mode,
        )
    }
}

// ---------------------------------------------------------------------------
// Module-level functions
// ---------------------------------------------------------------------------

/// Perform a simple transcode with minimal configuration.
///
/// Arguments:
///     input: Source file path.
///     output: Destination file path.
///     preset: Optional preset name.
///     crf: Optional CRF value.
#[pyfunction]
#[pyo3(signature = (input, output, preset = None, crf = None))]
pub fn transcode_simple(
    input: &str,
    output: &str,
    preset: Option<&str>,
    crf: Option<u32>,
) -> PyResult<PyTranscodeResult> {
    if input.is_empty() || output.is_empty() {
        return Err(val_err("Input and output paths must not be empty"));
    }

    let mut tc = Transcoder::new().input(input).output(output);

    // Apply preset defaults if given
    if let Some(p) = preset {
        let (vc, ac, container) = preset_defaults(p)?;
        tc = tc.video_codec(vc).audio_codec(ac);
        let _ = container; // container inferred from output extension
    }

    if let Some(c) = crf {
        // Use CRF via quality mode; for now map to quality level
        if c <= 20 {
            tc = tc.quality(QualityMode::VeryHigh);
        } else if c <= 30 {
            tc = tc.quality(QualityMode::High);
        } else if c <= 40 {
            tc = tc.quality(QualityMode::Medium);
        } else {
            tc = tc.quality(QualityMode::Low);
        }
    }

    let rt = tokio::runtime::Runtime::new().map_err(tc_err)?;
    let output_result = rt.block_on(tc.transcode()).map_err(tc_err)?;
    Ok(PyTranscodeResult::from(&output_result))
}

/// List all available transcoding presets.
#[pyfunction]
pub fn list_presets() -> PyResult<Vec<PyPresetInfo>> {
    Ok(built_in_presets())
}

/// List supported codecs with metadata.
#[pyfunction]
pub fn list_codecs() -> PyResult<Vec<HashMap<String, String>>> {
    Ok(built_in_codecs())
}

/// Validate a transcode configuration and return a list of warnings.
///
/// An empty list means the configuration is valid.
#[pyfunction]
#[pyo3(signature = (input, output, video_codec = None, audio_codec = None))]
pub fn validate_transcode_config(
    input: &str,
    output: &str,
    video_codec: Option<&str>,
    audio_codec: Option<&str>,
) -> PyResult<Vec<String>> {
    let mut warnings: Vec<String> = Vec::new();

    if input.is_empty() {
        warnings.push("Input path is empty".to_string());
    }
    if output.is_empty() {
        warnings.push("Output path is empty".to_string());
    }

    let supported_video = ["av1", "vp9", "vp8"];
    if let Some(vc) = video_codec {
        if !supported_video.contains(&vc) {
            warnings.push(format!(
                "Video codec '{}' is not a patent-free codec supported by OxiMedia",
                vc
            ));
        }
    }

    let supported_audio = ["opus", "vorbis", "flac", "pcm"];
    if let Some(ac) = audio_codec {
        if !supported_audio.contains(&ac) {
            warnings.push(format!(
                "Audio codec '{}' is not a patent-free codec supported by OxiMedia",
                ac
            ));
        }
    }

    // Check output extension matches expected containers
    let ext = output.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    let valid_exts = ["webm", "mkv", "ogg", "opus", "flac", "wav"];
    if !ext.is_empty() && !valid_exts.contains(&ext.as_str()) {
        warnings.push(format!(
            "Output extension '.{}' may not be a supported container. Consider: {}",
            ext,
            valid_exts.join(", ")
        ));
    }

    Ok(warnings)
}

// ---------------------------------------------------------------------------
// Registration helper
// ---------------------------------------------------------------------------

/// Register transcode bindings on the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTranscodeResult>()?;
    m.add_class::<PyAbrRung>()?;
    m.add_class::<PyAbrLadder>()?;
    m.add_class::<PyPresetInfo>()?;
    m.add_class::<PyTranscoder>()?;
    m.add_function(wrap_pyfunction!(transcode_simple, m)?)?;
    m.add_function(wrap_pyfunction!(list_presets, m)?)?;
    m.add_function(wrap_pyfunction!(list_codecs, m)?)?;
    m.add_function(wrap_pyfunction!(validate_transcode_config, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn preset_defaults(name: &str) -> PyResult<(&'static str, &'static str, &'static str)> {
    match name {
        "web_optimized" => Ok(("av1", "opus", "webm")),
        "archive_quality" => Ok(("av1", "flac", "mkv")),
        "fast_preview" => Ok(("vp8", "vorbis", "webm")),
        "broadcast_hd" => Ok(("av1", "opus", "mkv")),
        "social_media" => Ok(("vp9", "opus", "webm")),
        "youtube_1080p" => Ok(("vp9", "opus", "webm")),
        "vimeo_hd" => Ok(("av1", "opus", "webm")),
        _ => Err(val_err(format!("Unknown preset: {name}"))),
    }
}

fn built_in_presets() -> Vec<PyPresetInfo> {
    vec![
        PyPresetInfo {
            name: "web_optimized".to_string(),
            description: "Web-optimized AV1/Opus encoding for low-latency streaming".to_string(),
            video_codec: "av1".to_string(),
            audio_codec: "opus".to_string(),
            container: "webm".to_string(),
            quality_mode: "balanced".to_string(),
        },
        PyPresetInfo {
            name: "archive_quality".to_string(),
            description: "High-fidelity archival encoding with lossless audio".to_string(),
            video_codec: "av1".to_string(),
            audio_codec: "flac".to_string(),
            container: "mkv".to_string(),
            quality_mode: "high".to_string(),
        },
        PyPresetInfo {
            name: "fast_preview".to_string(),
            description: "Quick preview transcode with minimal processing".to_string(),
            video_codec: "vp8".to_string(),
            audio_codec: "vorbis".to_string(),
            container: "webm".to_string(),
            quality_mode: "fast".to_string(),
        },
        PyPresetInfo {
            name: "broadcast_hd".to_string(),
            description: "Broadcast-grade 1080p encoding".to_string(),
            video_codec: "av1".to_string(),
            audio_codec: "opus".to_string(),
            container: "mkv".to_string(),
            quality_mode: "high".to_string(),
        },
        PyPresetInfo {
            name: "social_media".to_string(),
            description: "Optimized for social media platforms with small file size".to_string(),
            video_codec: "vp9".to_string(),
            audio_codec: "opus".to_string(),
            container: "webm".to_string(),
            quality_mode: "balanced".to_string(),
        },
        PyPresetInfo {
            name: "youtube_1080p".to_string(),
            description: "YouTube-optimized 1080p VP9/Opus".to_string(),
            video_codec: "vp9".to_string(),
            audio_codec: "opus".to_string(),
            container: "webm".to_string(),
            quality_mode: "balanced".to_string(),
        },
        PyPresetInfo {
            name: "vimeo_hd".to_string(),
            description: "Vimeo-quality AV1 encoding".to_string(),
            video_codec: "av1".to_string(),
            audio_codec: "opus".to_string(),
            container: "webm".to_string(),
            quality_mode: "high".to_string(),
        },
    ]
}

fn built_in_codecs() -> Vec<HashMap<String, String>> {
    let entries: &[(&str, &str, &str)] = &[
        (
            "av1",
            "video",
            "AV1 (Alliance for Open Media) - patent-free next-gen codec",
        ),
        (
            "vp9",
            "video",
            "VP9 (Google) - patent-free 4K-capable codec",
        ),
        ("vp8", "video", "VP8 (Google) - patent-free web video codec"),
        (
            "opus",
            "audio",
            "Opus - patent-free low-latency audio codec",
        ),
        ("vorbis", "audio", "Vorbis - patent-free Ogg audio codec"),
        ("flac", "audio", "FLAC - patent-free lossless audio codec"),
        ("pcm", "audio", "PCM - uncompressed audio"),
    ];
    entries
        .iter()
        .map(|(name, kind, desc)| {
            let mut m = HashMap::new();
            m.insert("name".to_string(), (*name).to_string());
            m.insert("type".to_string(), (*kind).to_string());
            m.insert("description".to_string(), (*desc).to_string());
            m
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preset_defaults_known() {
        let (vc, ac, ct) = preset_defaults("web_optimized").expect("should resolve");
        assert_eq!(vc, "av1");
        assert_eq!(ac, "opus");
        assert_eq!(ct, "webm");
    }

    #[test]
    fn test_preset_defaults_unknown() {
        assert!(preset_defaults("nonexistent").is_err());
    }

    #[test]
    fn test_built_in_presets_not_empty() {
        let presets = built_in_presets();
        assert!(presets.len() >= 5);
        for p in &presets {
            assert!(!p.name.is_empty());
            assert!(!p.video_codec.is_empty());
        }
    }

    #[test]
    fn test_built_in_codecs() {
        let codecs = built_in_codecs();
        assert!(codecs.len() >= 5);
        let names: Vec<&str> = codecs
            .iter()
            .filter_map(|c| c.get("name").map(|s| s.as_str()))
            .collect();
        assert!(names.contains(&"av1"));
        assert!(names.contains(&"opus"));
    }

    #[test]
    fn test_abr_ladder_hls_default() {
        let ladder = PyAbrLadder::hls_default();
        assert_eq!(ladder.rung_count(), 5);
        let rungs = ladder.rungs();
        assert_eq!(rungs[0].width, 426);
        assert_eq!(rungs[4].width, 1920);
    }

    #[test]
    fn test_validate_empty_paths() {
        let warnings = validate_transcode_config("", "output.webm", None, None)
            .expect("validation should not fail");
        assert!(warnings.iter().any(|w| w.contains("Input path is empty")));
    }

    #[test]
    fn test_validate_bad_codec() {
        let warnings = validate_transcode_config("in.mkv", "out.webm", Some("h264"), None)
            .expect("validation should not fail");
        assert!(warnings.iter().any(|w| w.contains("patent-free")));
    }

    #[test]
    fn test_validate_bad_extension() {
        let warnings = validate_transcode_config("in.mkv", "out.mp4", None, None)
            .expect("validation should not fail");
        assert!(warnings.iter().any(|w| w.contains("container")));
    }

    #[test]
    fn test_transcode_result_repr() {
        let r = PyTranscodeResult {
            file_size: 1024,
            duration_secs: 10.0,
            bitrate_kbps: 500.0,
            encoding_time_secs: 2.5,
            compression_ratio: 0.8,
        };
        let repr = r.__repr__();
        assert!(repr.contains("1024"));
        assert!(repr.contains("500.0"));
    }
}
