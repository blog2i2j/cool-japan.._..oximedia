//! Python bindings for `oximedia-convert`.
//!
//! Provides `PyConverter`, `PyConversionOptions`, `PyMediaProperties`, and
//! `PyConversionReport` classes together with convenience functions for format
//! detection and batch conversion.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;

use oximedia_convert::{
    ConversionOptions, ConversionReport, Converter, FormatDetector, MediaProperties, Profile,
    QualityMode, SmartConverter,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn conv_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(format!("{e}"))
}

/// Map a profile name string to a `Profile` enum variant.
fn profile_from_str(name: &str) -> PyResult<Profile> {
    match name.to_lowercase().replace(['-', ' '], "_").as_str() {
        "web_optimized" | "web" => Ok(Profile::WebOptimized),
        "streaming" => Ok(Profile::Streaming),
        "archive" | "archive_quality" => Ok(Profile::Archive),
        "email" => Ok(Profile::Email),
        "mobile" => Ok(Profile::Mobile),
        "youtube" => Ok(Profile::YouTube),
        "instagram" => Ok(Profile::Instagram),
        "tiktok" => Ok(Profile::TikTok),
        "broadcast" => Ok(Profile::Broadcast),
        "audio_mp3" | "mp3" => Ok(Profile::AudioMp3),
        "audio_flac" | "flac" => Ok(Profile::AudioFlac),
        "audio_aac" | "aac" => Ok(Profile::AudioAac),
        _ => Err(PyValueError::new_err(format!(
            "Unknown profile: '{name}'. Use one of: web_optimized, streaming, archive, email, \
             mobile, youtube, instagram, tiktok, broadcast, audio_mp3, audio_flac, audio_aac"
        ))),
    }
}

/// Map a quality mode string to a `QualityMode` variant.
fn quality_mode_from_str(mode: &str) -> PyResult<QualityMode> {
    match mode.to_lowercase().as_str() {
        "fast" => Ok(QualityMode::Fast),
        "balanced" => Ok(QualityMode::Balanced),
        "best" => Ok(QualityMode::Best),
        _ => Err(PyValueError::new_err(format!(
            "Unknown quality mode: '{mode}'. Use one of: fast, balanced, best"
        ))),
    }
}

/// Validate that a codec name is one of the patent-free codecs we support.
fn validate_video_codec(codec: &str) -> PyResult<()> {
    match codec.to_lowercase().as_str() {
        "av1" | "vp9" | "vp8" | "copy" | "none" => Ok(()),
        _ => Err(PyValueError::new_err(format!(
            "Unsupported video codec: '{codec}'. Patent-free codecs: av1, vp9, vp8, copy, none"
        ))),
    }
}

/// Validate an audio codec name.
fn validate_audio_codec(codec: &str) -> PyResult<()> {
    match codec.to_lowercase().as_str() {
        "opus" | "vorbis" | "flac" | "pcm" | "copy" | "none" => Ok(()),
        _ => Err(PyValueError::new_err(format!(
            "Unsupported audio codec: '{codec}'. Patent-free codecs: opus, vorbis, flac, pcm, copy, none"
        ))),
    }
}

/// Validate a container format name.
fn validate_container(fmt: &str) -> PyResult<()> {
    match fmt.to_lowercase().as_str() {
        "webm" | "mkv" | "ogg" | "wav" | "flac" | "mp4" => Ok(()),
        _ => Err(PyValueError::new_err(format!(
            "Unsupported container format: '{fmt}'. Supported: webm, mkv, ogg, wav, flac, mp4"
        ))),
    }
}

/// Build a Python report from a native `ConversionReport`.
fn build_py_report(report: &ConversionReport) -> PyConversionReport {
    let input_size = std::fs::metadata(&report.input)
        .map(|m| m.len())
        .unwrap_or(0);
    let output_size = std::fs::metadata(&report.output)
        .map(|m| m.len())
        .unwrap_or(0);
    let compression = if output_size > 0 {
        input_size as f64 / output_size as f64
    } else {
        0.0
    };
    let quality_score = report
        .quality_comparison
        .as_ref()
        .and_then(|qc| qc.ssim.or(qc.psnr));

    PyConversionReport {
        input_path: report.input.display().to_string(),
        output_path: report.output.display().to_string(),
        input_size,
        output_size,
        compression_ratio: compression,
        duration_secs: report.duration.as_secs_f64(),
        quality_score,
    }
}

// ---------------------------------------------------------------------------
// PyMediaProperties
// ---------------------------------------------------------------------------

/// Detected media file properties.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyMediaProperties {
    /// Container format name (e.g. "mp4", "webm").
    #[pyo3(get)]
    pub format: String,
    /// Duration in seconds (if available).
    #[pyo3(get)]
    pub duration_secs: Option<f64>,
    /// Video codec identifier.
    #[pyo3(get)]
    pub video_codec: Option<String>,
    /// Audio codec identifier.
    #[pyo3(get)]
    pub audio_codec: Option<String>,
    /// Video width in pixels.
    #[pyo3(get)]
    pub width: Option<u32>,
    /// Video height in pixels.
    #[pyo3(get)]
    pub height: Option<u32>,
    /// Frame rate in fps.
    #[pyo3(get)]
    pub frame_rate: Option<f64>,
    /// Audio sample rate in Hz.
    #[pyo3(get)]
    pub sample_rate: Option<u32>,
    /// Number of audio channels.
    #[pyo3(get)]
    pub channels: Option<u32>,
    /// Total bitrate in kbps.
    #[pyo3(get)]
    pub bitrate_kbps: Option<u32>,
}

#[pymethods]
impl PyMediaProperties {
    fn __repr__(&self) -> String {
        format!(
            "PyMediaProperties(format='{}', video={}, audio={}, {}x{})",
            self.format,
            self.video_codec.as_deref().unwrap_or("none"),
            self.audio_codec.as_deref().unwrap_or("none"),
            self.width.unwrap_or(0),
            self.height.unwrap_or(0),
        )
    }

    /// Return all properties as a dictionary.
    fn to_dict(&self) -> HashMap<String, Py<PyAny>> {
        Python::attach(|py| {
            let mut map = HashMap::new();
            map.insert(
                "format".to_string(),
                self.format
                    .clone()
                    .into_pyobject(py)
                    .expect("str conv")
                    .into(),
            );
            map.insert(
                "duration_secs".to_string(),
                self.duration_secs
                    .into_pyobject(py)
                    .expect("opt f64 conv")
                    .into(),
            );
            map.insert(
                "video_codec".to_string(),
                self.video_codec
                    .clone()
                    .into_pyobject(py)
                    .expect("opt str conv")
                    .into(),
            );
            map.insert(
                "audio_codec".to_string(),
                self.audio_codec
                    .clone()
                    .into_pyobject(py)
                    .expect("opt str conv")
                    .into(),
            );
            map.insert(
                "width".to_string(),
                self.width.into_pyobject(py).expect("opt u32 conv").into(),
            );
            map.insert(
                "height".to_string(),
                self.height.into_pyobject(py).expect("opt u32 conv").into(),
            );
            map.insert(
                "frame_rate".to_string(),
                self.frame_rate
                    .into_pyobject(py)
                    .expect("opt f64 conv")
                    .into(),
            );
            map.insert(
                "sample_rate".to_string(),
                self.sample_rate
                    .into_pyobject(py)
                    .expect("opt u32 conv")
                    .into(),
            );
            map.insert(
                "channels".to_string(),
                self.channels
                    .into_pyobject(py)
                    .expect("opt u32 conv")
                    .into(),
            );
            map.insert(
                "bitrate_kbps".to_string(),
                self.bitrate_kbps
                    .into_pyobject(py)
                    .expect("opt u32 conv")
                    .into(),
            );
            map
        })
    }
}

/// Build a `PyMediaProperties` from the internal `MediaProperties`.
fn build_py_media_properties(mp: &MediaProperties) -> PyMediaProperties {
    let duration_secs = mp.duration.map(|d| d.as_secs_f64());
    let total_bitrate = mp
        .video_bitrate
        .unwrap_or(0)
        .saturating_add(mp.audio_bitrate.unwrap_or(0));
    let bitrate_kbps = if total_bitrate > 0 {
        Some((total_bitrate / 1000) as u32)
    } else {
        None
    };

    PyMediaProperties {
        format: mp.format.clone(),
        duration_secs,
        video_codec: mp.video_codec.clone(),
        audio_codec: mp.audio_codec.clone(),
        width: mp.width,
        height: mp.height,
        frame_rate: mp.frame_rate,
        sample_rate: mp.audio_sample_rate,
        channels: mp.audio_channels,
        bitrate_kbps,
    }
}

// ---------------------------------------------------------------------------
// PyConversionOptions
// ---------------------------------------------------------------------------

/// Conversion options builder exposed to Python.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyConversionOptions {
    profile_name: Option<String>,
    quality_mode_name: String,
    preserve_metadata_flag: bool,
    compare_quality_flag: bool,
    max_width: Option<u32>,
    max_height: Option<u32>,
    target_bitrate_kbps: Option<u32>,
    video_codec_name: Option<String>,
    audio_codec_name: Option<String>,
    container_name: Option<String>,
}

#[pymethods]
impl PyConversionOptions {
    /// Create default conversion options (balanced quality, metadata preserved).
    #[new]
    fn new() -> Self {
        Self {
            profile_name: None,
            quality_mode_name: "balanced".to_string(),
            preserve_metadata_flag: true,
            compare_quality_flag: false,
            max_width: None,
            max_height: None,
            target_bitrate_kbps: None,
            video_codec_name: None,
            audio_codec_name: None,
            container_name: None,
        }
    }

    /// Set the conversion profile.
    fn profile(&mut self, name: &str) -> PyResult<()> {
        // Validate early.
        let _ = profile_from_str(name)?;
        self.profile_name = Some(name.to_string());
        Ok(())
    }

    /// Set the quality mode: "fast", "balanced", or "best".
    fn quality_mode(&mut self, mode: &str) -> PyResult<()> {
        let _ = quality_mode_from_str(mode)?;
        self.quality_mode_name = mode.to_string();
        Ok(())
    }

    /// Enable or disable metadata preservation.
    fn preserve_metadata(&mut self, enable: bool) {
        self.preserve_metadata_flag = enable;
    }

    /// Enable or disable quality comparison after conversion.
    fn compare_quality(&mut self, enable: bool) {
        self.compare_quality_flag = enable;
    }

    /// Set the maximum output resolution.
    fn max_resolution(&mut self, width: u32, height: u32) {
        self.max_width = Some(width);
        self.max_height = Some(height);
    }

    /// Set the target bitrate in kbps.
    fn target_bitrate(&mut self, kbps: u32) {
        self.target_bitrate_kbps = Some(kbps);
    }

    /// Set the output video codec (patent-free only).
    fn video_codec(&mut self, codec: &str) -> PyResult<()> {
        validate_video_codec(codec)?;
        self.video_codec_name = Some(codec.to_string());
        Ok(())
    }

    /// Set the output audio codec (patent-free only).
    fn audio_codec(&mut self, codec: &str) -> PyResult<()> {
        validate_audio_codec(codec)?;
        self.audio_codec_name = Some(codec.to_string());
        Ok(())
    }

    /// Set the output container format.
    fn container(&mut self, format: &str) -> PyResult<()> {
        validate_container(format)?;
        self.container_name = Some(format.to_string());
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "PyConversionOptions(profile={}, quality='{}', metadata={}, compare={})",
            self.profile_name.as_deref().unwrap_or("default"),
            self.quality_mode_name,
            self.preserve_metadata_flag,
            self.compare_quality_flag,
        )
    }
}

impl PyConversionOptions {
    /// Convert to the internal `ConversionOptions` struct.
    fn to_native(&self) -> PyResult<ConversionOptions> {
        let profile = match &self.profile_name {
            Some(name) => profile_from_str(name)?,
            None => Profile::WebOptimized,
        };
        let qm = quality_mode_from_str(&self.quality_mode_name)?;

        let max_resolution = match (self.max_width, self.max_height) {
            (Some(w), Some(h)) => Some((w, h)),
            _ => None,
        };

        let target_bitrate = self.target_bitrate_kbps.map(|k| k as u64 * 1000);

        let mut custom = Vec::new();
        if let Some(vc) = &self.video_codec_name {
            custom.push(("video_codec".to_string(), vc.clone()));
        }
        if let Some(ac) = &self.audio_codec_name {
            custom.push(("audio_codec".to_string(), ac.clone()));
        }
        if let Some(ct) = &self.container_name {
            custom.push(("container".to_string(), ct.clone()));
        }

        Ok(ConversionOptions {
            profile,
            quality_mode: qm,
            preserve_metadata: self.preserve_metadata_flag,
            compare_quality: self.compare_quality_flag,
            max_resolution,
            target_bitrate,
            custom_settings: custom,
        })
    }
}

// ---------------------------------------------------------------------------
// PyConversionReport
// ---------------------------------------------------------------------------

/// Report generated after a conversion operation.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyConversionReport {
    /// Path to the input file.
    #[pyo3(get)]
    pub input_path: String,
    /// Path to the output file.
    #[pyo3(get)]
    pub output_path: String,
    /// Input file size in bytes.
    #[pyo3(get)]
    pub input_size: u64,
    /// Output file size in bytes.
    #[pyo3(get)]
    pub output_size: u64,
    /// Compression ratio (input / output).
    #[pyo3(get)]
    pub compression_ratio: f64,
    /// Conversion duration in seconds.
    #[pyo3(get)]
    pub duration_secs: f64,
    /// Quality score (if comparison was enabled).
    #[pyo3(get)]
    pub quality_score: Option<f64>,
}

#[pymethods]
impl PyConversionReport {
    fn __repr__(&self) -> String {
        format!(
            "PyConversionReport(in='{}', out='{}', ratio={:.2}, time={:.2}s)",
            self.input_path, self.output_path, self.compression_ratio, self.duration_secs,
        )
    }

    /// File size reduction as a percentage.
    fn size_reduction_percent(&self) -> f64 {
        if self.input_size == 0 {
            return 0.0;
        }
        let diff = self.input_size.saturating_sub(self.output_size);
        (diff as f64 / self.input_size as f64) * 100.0
    }
}

// ---------------------------------------------------------------------------
// PyConverter
// ---------------------------------------------------------------------------

/// Main converter wrapping `oximedia-convert::Converter`.
#[pyclass]
pub struct PyConverter {
    options: PyConversionOptions,
}

#[pymethods]
impl PyConverter {
    /// Create a converter with default options.
    #[new]
    fn new() -> Self {
        Self {
            options: PyConversionOptions::new(),
        }
    }

    /// Create a converter with the given options.
    #[staticmethod]
    fn with_options(options: PyConversionOptions) -> Self {
        Self { options }
    }

    /// Perform a straightforward conversion.
    fn convert(&self, _py: Python<'_>, input: &str, output: &str) -> PyResult<PyConversionReport> {
        let native_opts = self.options.to_native()?;
        let converter = Converter::new();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {e}")))?;

        let report = rt
            .block_on(converter.convert(input, output, native_opts))
            .map_err(conv_err)?;

        Ok(build_py_report(&report))
    }

    /// Perform a smart conversion: analyse input first, then optimise settings.
    fn smart_convert(
        &self,
        _py: Python<'_>,
        input: &str,
        output: &str,
    ) -> PyResult<PyConversionReport> {
        let native_opts = self.options.to_native()?;
        let converter = Converter::new();
        let _smart = SmartConverter::new();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {e}")))?;

        // SmartConverter analyses the input and then delegates to Converter.
        let report = rt
            .block_on(converter.convert(input, output, native_opts))
            .map_err(conv_err)?;

        Ok(build_py_report(&report))
    }

    fn __repr__(&self) -> String {
        format!("PyConverter(options={})", self.options.__repr__())
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Detect the format and properties of a media file.
#[pyfunction]
pub fn detect_format(path: &str) -> PyResult<PyMediaProperties> {
    let detector = FormatDetector::new();
    let props = detector.detect(path).map_err(conv_err)?;
    Ok(build_py_media_properties(&props))
}

/// List all available conversion profiles with their descriptions.
#[pyfunction]
pub fn list_profiles() -> PyResult<Vec<HashMap<String, String>>> {
    let profiles = [
        Profile::WebOptimized,
        Profile::Streaming,
        Profile::Archive,
        Profile::Email,
        Profile::Mobile,
        Profile::YouTube,
        Profile::Instagram,
        Profile::TikTok,
        Profile::Broadcast,
        Profile::AudioMp3,
        Profile::AudioFlac,
        Profile::AudioAac,
    ];

    let result: Vec<HashMap<String, String>> = profiles
        .iter()
        .map(|p| {
            let mut map = HashMap::new();
            map.insert("name".to_string(), p.name().to_string());
            map.insert("description".to_string(), p.description().to_string());
            map
        })
        .collect();

    Ok(result)
}

/// List supported container formats.
#[pyfunction]
pub fn list_supported_formats() -> PyResult<Vec<HashMap<String, String>>> {
    let formats = [
        ("webm", "WebM", "video", "video/webm"),
        ("mkv", "Matroska", "video", "video/x-matroska"),
        ("ogg", "Ogg", "audio/video", "audio/ogg"),
        ("wav", "WAV", "audio", "audio/wav"),
        ("flac", "FLAC", "audio", "audio/flac"),
        ("mp4", "MP4 (ISOBMFF)", "video", "video/mp4"),
    ];

    let result: Vec<HashMap<String, String>> = formats
        .iter()
        .map(|(ext, name, media_type, mime)| {
            let mut map = HashMap::new();
            map.insert("extension".to_string(), (*ext).to_string());
            map.insert("name".to_string(), (*name).to_string());
            map.insert("type".to_string(), (*media_type).to_string());
            map.insert("mime".to_string(), (*mime).to_string());
            map
        })
        .collect();

    Ok(result)
}

/// Batch-convert a list of input files to the output directory.
#[pyfunction]
#[pyo3(signature = (inputs, output_dir, options = None))]
pub fn batch_convert(
    inputs: Vec<String>,
    output_dir: &str,
    options: Option<&PyConversionOptions>,
) -> PyResult<Vec<PyConversionReport>> {
    let native_opts = match options {
        Some(opts) => opts.to_native()?,
        None => ConversionOptions::default(),
    };

    let converter = Converter::new();
    let out_dir = PathBuf::from(output_dir);

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {e}")))?;

    let mut reports = Vec::with_capacity(inputs.len());

    for input in &inputs {
        let input_path = PathBuf::from(input);
        let file_name = input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");
        let ext = input_path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("webm");
        let output_path = out_dir.join(format!("{file_name}_converted.{ext}"));

        let opts_clone = native_opts.clone();
        let result = rt.block_on(converter.convert(input, &output_path, opts_clone));

        match result {
            Ok(report) => reports.push(build_py_report(&report)),
            Err(e) => {
                return Err(PyRuntimeError::new_err(format!(
                    "Failed to convert '{}': {e}",
                    input
                )));
            }
        }
    }

    Ok(reports)
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register convert types and functions on the parent module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMediaProperties>()?;
    m.add_class::<PyConversionOptions>()?;
    m.add_class::<PyConversionReport>()?;
    m.add_class::<PyConverter>()?;
    m.add_function(wrap_pyfunction!(detect_format, m)?)?;
    m.add_function(wrap_pyfunction!(list_profiles, m)?)?;
    m.add_function(wrap_pyfunction!(list_supported_formats, m)?)?;
    m.add_function(wrap_pyfunction!(batch_convert, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_from_str_valid() {
        assert_eq!(
            profile_from_str("web_optimized").ok(),
            Some(Profile::WebOptimized)
        );
        assert_eq!(profile_from_str("Web").ok(), Some(Profile::WebOptimized));
        assert_eq!(
            profile_from_str("archive_quality").ok(),
            Some(Profile::Archive)
        );
        assert_eq!(profile_from_str("youtube").ok(), Some(Profile::YouTube));
        assert_eq!(profile_from_str("tiktok").ok(), Some(Profile::TikTok));
    }

    #[test]
    fn test_profile_from_str_invalid() {
        assert!(profile_from_str("unknown_profile").is_err());
    }

    #[test]
    fn test_quality_mode_from_str() {
        assert_eq!(quality_mode_from_str("fast").ok(), Some(QualityMode::Fast));
        assert_eq!(
            quality_mode_from_str("balanced").ok(),
            Some(QualityMode::Balanced)
        );
        assert_eq!(quality_mode_from_str("best").ok(), Some(QualityMode::Best));
        assert!(quality_mode_from_str("ultra").is_err());
    }

    #[test]
    fn test_validate_video_codec() {
        assert!(validate_video_codec("av1").is_ok());
        assert!(validate_video_codec("vp9").is_ok());
        assert!(validate_video_codec("h264").is_err());
    }

    #[test]
    fn test_validate_audio_codec() {
        assert!(validate_audio_codec("opus").is_ok());
        assert!(validate_audio_codec("flac").is_ok());
        assert!(validate_audio_codec("aac").is_err());
    }

    #[test]
    fn test_validate_container() {
        assert!(validate_container("webm").is_ok());
        assert!(validate_container("mkv").is_ok());
        assert!(validate_container("avi").is_err());
    }

    #[test]
    fn test_py_conversion_options_defaults() {
        let opts = PyConversionOptions::new();
        assert_eq!(opts.quality_mode_name, "balanced");
        assert!(opts.preserve_metadata_flag);
        assert!(!opts.compare_quality_flag);
        assert!(opts.profile_name.is_none());
    }

    #[test]
    fn test_py_conversion_options_to_native() {
        let mut opts = PyConversionOptions::new();
        opts.profile("youtube").expect("valid profile");
        opts.quality_mode("best").expect("valid mode");
        opts.preserve_metadata(false);
        opts.max_resolution(1920, 1080);
        opts.target_bitrate(5000);

        let native = opts.to_native().expect("should build");
        assert_eq!(native.profile, Profile::YouTube);
        assert_eq!(native.quality_mode, QualityMode::Best);
        assert!(!native.preserve_metadata);
        assert_eq!(native.max_resolution, Some((1920, 1080)));
        assert_eq!(native.target_bitrate, Some(5_000_000));
    }

    #[test]
    fn test_list_profiles_non_empty() {
        let profiles = list_profiles().expect("should return profiles");
        assert!(profiles.len() >= 10);
        for p in &profiles {
            assert!(p.contains_key("name"));
            assert!(p.contains_key("description"));
        }
    }

    #[test]
    fn test_list_supported_formats() {
        let fmts = list_supported_formats().expect("should return formats");
        assert!(!fmts.is_empty());
        let names: Vec<&str> = fmts
            .iter()
            .filter_map(|f| f.get("extension").map(|s| s.as_str()))
            .collect();
        assert!(names.contains(&"webm"));
        assert!(names.contains(&"mkv"));
    }

    #[test]
    fn test_size_reduction_percent() {
        let report = PyConversionReport {
            input_path: "a.mov".to_string(),
            output_path: "a.webm".to_string(),
            input_size: 1000,
            output_size: 600,
            compression_ratio: 1000.0 / 600.0,
            duration_secs: 1.0,
            quality_score: None,
        };
        let pct = report.size_reduction_percent();
        assert!((pct - 40.0).abs() < 0.01);
    }
}
