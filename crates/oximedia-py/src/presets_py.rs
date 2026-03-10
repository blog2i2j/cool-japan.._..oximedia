//! Python bindings for `oximedia-transcode` encoding presets.
//!
//! Provides `PyPreset` and `PyPresetManager` for managing encoding presets
//! from Python, plus standalone convenience functions.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::collections::HashMap;

use oximedia_transcode::{PresetConfig, QualityMode};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn quality_mode_to_string(mode: &QualityMode) -> String {
    match mode {
        QualityMode::Low => "low".to_string(),
        QualityMode::Medium => "medium".to_string(),
        QualityMode::High => "high".to_string(),
        QualityMode::VeryHigh => "very_high".to_string(),
        QualityMode::Custom => "custom".to_string(),
    }
}

fn parse_quality_mode(s: &str) -> PyResult<QualityMode> {
    match s {
        "low" => Ok(QualityMode::Low),
        "medium" => Ok(QualityMode::Medium),
        "high" => Ok(QualityMode::High),
        "very_high" => Ok(QualityMode::VeryHigh),
        "custom" => Ok(QualityMode::Custom),
        other => Err(PyValueError::new_err(format!(
            "Unknown quality mode '{}'. Expected: low, medium, high, very_high, custom",
            other
        ))),
    }
}

fn preset_config_to_py(name: &str, description: &str, config: &PresetConfig) -> PyPreset {
    PyPreset {
        name: name.to_string(),
        description: description.to_string(),
        video_codec: config.video_codec.clone().unwrap_or_default(),
        audio_codec: config.audio_codec.clone().unwrap_or_default(),
        container: config.container.clone().unwrap_or_default(),
        video_bitrate: config.video_bitrate.map(|v| v as u32),
        audio_bitrate: config.audio_bitrate.map(|v| v as u32),
        crf: None,
        width: config.width,
        height: config.height,
        frame_rate: config.frame_rate.map(|(n, d)| {
            if d == 0 {
                0.0
            } else {
                f64::from(n) / f64::from(d)
            }
        }),
        sample_rate: None,
        quality_mode: config
            .quality_mode
            .as_ref()
            .map(quality_mode_to_string)
            .unwrap_or_else(|| "medium".to_string()),
        custom_params: HashMap::new(),
    }
}

#[allow(dead_code)]
fn py_to_preset_config(py: &PyPreset) -> PresetConfig {
    PresetConfig {
        video_codec: if py.video_codec.is_empty() {
            None
        } else {
            Some(py.video_codec.clone())
        },
        audio_codec: if py.audio_codec.is_empty() {
            None
        } else {
            Some(py.audio_codec.clone())
        },
        video_bitrate: py.video_bitrate.map(u64::from),
        audio_bitrate: py.audio_bitrate.map(u64::from),
        width: py.width,
        height: py.height,
        frame_rate: py.frame_rate.map(|fps| {
            if fps <= 0.0 {
                (0, 1)
            } else {
                // Approximate as integer ratio
                let num = (fps * 1000.0).round() as u32;
                (num, 1000)
            }
        }),
        quality_mode: parse_quality_mode(&py.quality_mode).ok(),
        container: if py.container.is_empty() {
            None
        } else {
            Some(py.container.clone())
        },
    }
}

// ---------------------------------------------------------------------------
// PyPreset
// ---------------------------------------------------------------------------

/// An encoding preset with video/audio codec, container, and quality settings.
#[pyclass]
#[derive(Clone)]
pub struct PyPreset {
    /// Preset name.
    #[pyo3(get)]
    pub name: String,
    /// Human-readable description.
    #[pyo3(get)]
    pub description: String,
    /// Video codec (e.g. "av1", "vp9").
    #[pyo3(get)]
    pub video_codec: String,
    /// Audio codec (e.g. "opus", "flac").
    #[pyo3(get)]
    pub audio_codec: String,
    /// Container format (e.g. "webm", "mp4").
    #[pyo3(get)]
    pub container: String,
    /// Video bitrate in bits per second.
    #[pyo3(get)]
    pub video_bitrate: Option<u32>,
    /// Audio bitrate in bits per second.
    #[pyo3(get)]
    pub audio_bitrate: Option<u32>,
    /// Constant Rate Factor (CRF).
    #[pyo3(get)]
    pub crf: Option<u32>,
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
    /// Quality mode: "low", "medium", "high", "very_high", "custom".
    #[pyo3(get)]
    pub quality_mode: String,
    /// Custom parameters map.
    custom_params: HashMap<String, String>,
}

#[pymethods]
impl PyPreset {
    /// Create a web-optimized preset (AV1 + Opus, 720p).
    #[classmethod]
    fn web_optimized(_cls: &Bound<'_, PyType>) -> Self {
        let config = oximedia_transcode::presets::av1_opus(1280, 720, 2_500_000, 128_000);
        preset_config_to_py("web_optimized", "Web-optimized AV1/Opus 720p", &config)
    }

    /// Create an archive-quality preset (AV1 + Opus, 1080p, high quality).
    #[classmethod]
    fn archive_quality(_cls: &Bound<'_, PyType>) -> Self {
        let config = oximedia_transcode::presets::av1_opus(1920, 1080, 8_000_000, 256_000);
        let mut p =
            preset_config_to_py("archive_quality", "Archive-quality AV1/Opus 1080p", &config);
        p.quality_mode = "very_high".to_string();
        p
    }

    /// Create a broadcast HD preset (VP9 + Opus, 1080p).
    #[classmethod]
    fn broadcast_hd(_cls: &Bound<'_, PyType>) -> Self {
        let config = oximedia_transcode::presets::vp9_opus(1920, 1080, 8_000_000, 256_000);
        preset_config_to_py("broadcast_hd", "Broadcast HD VP9/Opus 1080p", &config)
    }

    /// Create a social media preset (VP9 + Opus, 720p).
    #[classmethod]
    fn social_media(_cls: &Bound<'_, PyType>) -> Self {
        let config = oximedia_transcode::presets::vp9_opus(1280, 720, 4_000_000, 128_000);
        preset_config_to_py("social_media", "Social media VP9/Opus 720p", &config)
    }

    /// Create a YouTube 1080p preset (AV1 + Opus).
    #[classmethod]
    fn youtube_1080p(_cls: &Bound<'_, PyType>) -> Self {
        let config = oximedia_transcode::presets::av1_opus(1920, 1080, 5_000_000, 192_000);
        let mut p = preset_config_to_py("youtube_1080p", "YouTube 1080p AV1/Opus", &config);
        p.frame_rate = Some(30.0);
        p
    }

    /// Create a fast preview preset (VP9 + Opus, 480p, low quality).
    #[classmethod]
    fn fast_preview(_cls: &Bound<'_, PyType>) -> Self {
        let config = oximedia_transcode::presets::vp9_opus(854, 480, 1_000_000, 64_000);
        let mut p = preset_config_to_py("fast_preview", "Fast preview VP9/Opus 480p", &config);
        p.quality_mode = "low".to_string();
        p
    }

    /// Create a custom preset with the given codecs and container.
    #[classmethod]
    fn custom(
        _cls: &Bound<'_, PyType>,
        name: &str,
        video_codec: &str,
        audio_codec: &str,
        container: &str,
    ) -> Self {
        Self {
            name: name.to_string(),
            description: format!("Custom preset: {name}"),
            video_codec: video_codec.to_string(),
            audio_codec: audio_codec.to_string(),
            container: container.to_string(),
            video_bitrate: None,
            audio_bitrate: None,
            crf: None,
            width: None,
            height: None,
            frame_rate: None,
            sample_rate: None,
            quality_mode: "medium".to_string(),
            custom_params: HashMap::new(),
        }
    }

    /// Set the video bitrate in bits per second.
    fn with_video_bitrate(&mut self, bps: u32) {
        self.video_bitrate = Some(bps);
    }

    /// Set the audio bitrate in bits per second.
    fn with_audio_bitrate(&mut self, bps: u32) {
        self.audio_bitrate = Some(bps);
    }

    /// Set the CRF value.
    fn with_crf(&mut self, crf: u32) -> PyResult<()> {
        if crf > 63 {
            return Err(PyValueError::new_err(format!(
                "CRF must be 0-63, got {crf}"
            )));
        }
        self.crf = Some(crf);
        Ok(())
    }

    /// Set the output resolution.
    fn with_resolution(&mut self, width: u32, height: u32) -> PyResult<()> {
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err("Width and height must be > 0"));
        }
        self.width = Some(width);
        self.height = Some(height);
        Ok(())
    }

    /// Set the frame rate in fps.
    fn with_frame_rate(&mut self, fps: f64) -> PyResult<()> {
        if fps <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "Frame rate must be > 0, got {fps}"
            )));
        }
        self.frame_rate = Some(fps);
        Ok(())
    }

    /// Set the quality mode.
    fn with_quality(&mut self, mode: &str) -> PyResult<()> {
        let _ = parse_quality_mode(mode)?;
        self.quality_mode = mode.to_string();
        Ok(())
    }

    /// Set a custom parameter key-value pair.
    fn set_param(&mut self, key: &str, value: &str) {
        self.custom_params
            .insert(key.to_string(), value.to_string());
    }

    /// Get a custom parameter by key.
    fn get_param(&self, key: &str) -> Option<String> {
        self.custom_params.get(key).cloned()
    }

    /// Convert preset to a Python dict.
    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("name".to_string(), self.name.clone());
        m.insert("description".to_string(), self.description.clone());
        m.insert("video_codec".to_string(), self.video_codec.clone());
        m.insert("audio_codec".to_string(), self.audio_codec.clone());
        m.insert("container".to_string(), self.container.clone());
        m.insert("quality_mode".to_string(), self.quality_mode.clone());
        if let Some(vb) = self.video_bitrate {
            m.insert("video_bitrate".to_string(), vb.to_string());
        }
        if let Some(ab) = self.audio_bitrate {
            m.insert("audio_bitrate".to_string(), ab.to_string());
        }
        if let Some(crf) = self.crf {
            m.insert("crf".to_string(), crf.to_string());
        }
        if let Some(w) = self.width {
            m.insert("width".to_string(), w.to_string());
        }
        if let Some(h) = self.height {
            m.insert("height".to_string(), h.to_string());
        }
        if let Some(fps) = self.frame_rate {
            m.insert("frame_rate".to_string(), format!("{fps:.3}"));
        }
        if let Some(sr) = self.sample_rate {
            m.insert("sample_rate".to_string(), sr.to_string());
        }
        for (k, v) in &self.custom_params {
            m.insert(k.clone(), v.clone());
        }
        m
    }

    /// Serialize preset to JSON string.
    fn to_json(&self) -> PyResult<String> {
        let d = self.to_dict();
        serde_json::to_string_pretty(&d)
            .map_err(|e| PyRuntimeError::new_err(format!("JSON serialization failed: {e}")))
    }

    fn __repr__(&self) -> String {
        format!(
            "PyPreset(name='{}', video='{}', audio='{}', container='{}', quality='{}')",
            self.name, self.video_codec, self.audio_codec, self.container, self.quality_mode,
        )
    }
}

// ---------------------------------------------------------------------------
// PyPresetManager
// ---------------------------------------------------------------------------

/// Manages a collection of encoding presets.
#[pyclass]
pub struct PyPresetManager {
    presets: HashMap<String, PyPreset>,
}

#[pymethods]
impl PyPresetManager {
    /// Create a new preset manager pre-loaded with built-in presets.
    #[new]
    fn new() -> Self {
        let mut presets = HashMap::new();
        let builtin = builtin_preset_list();
        for p in builtin {
            presets.insert(p.name.clone(), p);
        }
        Self { presets }
    }

    /// Get a preset by name.
    fn get(&self, name: &str) -> Option<PyPreset> {
        self.presets.get(name).cloned()
    }

    /// Add a preset to the manager.
    fn add(&mut self, preset: PyPreset) -> PyResult<()> {
        if preset.name.is_empty() {
            return Err(PyValueError::new_err("Preset name must not be empty"));
        }
        self.presets.insert(preset.name.clone(), preset);
        Ok(())
    }

    /// Remove a preset by name.
    fn remove(&mut self, name: &str) -> PyResult<()> {
        if self.presets.remove(name).is_none() {
            return Err(PyValueError::new_err(format!(
                "Preset '{}' not found",
                name
            )));
        }
        Ok(())
    }

    /// List all preset names.
    fn list_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.presets.keys().cloned().collect();
        names.sort();
        names
    }

    /// List all presets.
    fn list_all(&self) -> Vec<PyPreset> {
        let mut items: Vec<PyPreset> = self.presets.values().cloned().collect();
        items.sort_by(|a, b| a.name.cmp(&b.name));
        items
    }

    /// Save presets to a JSON file.
    fn save_to_file(&self, path: &str) -> PyResult<()> {
        let data: Vec<HashMap<String, String>> =
            self.presets.values().map(|p| p.to_dict()).collect();
        let json = serde_json::to_string_pretty(&data)
            .map_err(|e| PyRuntimeError::new_err(format!("JSON serialization failed: {e}")))?;
        std::fs::write(path, json)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to write file: {e}")))?;
        Ok(())
    }

    /// Load presets from a JSON file.
    #[classmethod]
    fn load_from_file(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to read file: {e}")))?;
        let data: Vec<HashMap<String, String>> = serde_json::from_str(&json)
            .map_err(|e| PyRuntimeError::new_err(format!("JSON parse failed: {e}")))?;

        let mut presets = HashMap::new();
        for d in data {
            let name = d.get("name").cloned().unwrap_or_default();
            if name.is_empty() {
                continue;
            }
            let p = PyPreset {
                name: name.clone(),
                description: d.get("description").cloned().unwrap_or_default(),
                video_codec: d.get("video_codec").cloned().unwrap_or_default(),
                audio_codec: d.get("audio_codec").cloned().unwrap_or_default(),
                container: d.get("container").cloned().unwrap_or_default(),
                video_bitrate: d.get("video_bitrate").and_then(|v| v.parse().ok()),
                audio_bitrate: d.get("audio_bitrate").and_then(|v| v.parse().ok()),
                crf: d.get("crf").and_then(|v| v.parse().ok()),
                width: d.get("width").and_then(|v| v.parse().ok()),
                height: d.get("height").and_then(|v| v.parse().ok()),
                frame_rate: d.get("frame_rate").and_then(|v| v.parse().ok()),
                sample_rate: d.get("sample_rate").and_then(|v| v.parse().ok()),
                quality_mode: d
                    .get("quality_mode")
                    .cloned()
                    .unwrap_or_else(|| "medium".to_string()),
                custom_params: HashMap::new(),
            };
            presets.insert(name, p);
        }

        Ok(Self { presets })
    }

    /// List names of all built-in presets.
    #[staticmethod]
    fn builtin_presets() -> Vec<String> {
        builtin_preset_list().into_iter().map(|p| p.name).collect()
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// List all built-in presets.
#[pyfunction]
pub fn list_presets() -> Vec<PyPreset> {
    builtin_preset_list()
}

/// Get a built-in preset by name.
#[pyfunction]
pub fn get_preset(name: &str) -> PyResult<PyPreset> {
    let presets = builtin_preset_list();
    presets.into_iter().find(|p| p.name == name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "Preset '{}' not found. Use list_presets() to see available presets.",
            name
        ))
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn builtin_preset_list() -> Vec<PyPreset> {
    vec![
        preset_config_to_py(
            "web_optimized",
            "Web-optimized AV1/Opus 720p",
            &oximedia_transcode::presets::av1_opus(1280, 720, 2_500_000, 128_000),
        ),
        {
            let mut p = preset_config_to_py(
                "archive_quality",
                "Archive-quality AV1/Opus 1080p",
                &oximedia_transcode::presets::av1_opus(1920, 1080, 8_000_000, 256_000),
            );
            p.quality_mode = "very_high".to_string();
            p
        },
        preset_config_to_py(
            "broadcast_hd",
            "Broadcast HD VP9/Opus 1080p",
            &oximedia_transcode::presets::vp9_opus(1920, 1080, 8_000_000, 256_000),
        ),
        preset_config_to_py(
            "social_media",
            "Social media VP9/Opus 720p",
            &oximedia_transcode::presets::vp9_opus(1280, 720, 4_000_000, 128_000),
        ),
        {
            let mut p = preset_config_to_py(
                "youtube_1080p",
                "YouTube 1080p AV1/Opus",
                &oximedia_transcode::presets::av1_opus(1920, 1080, 5_000_000, 192_000),
            );
            p.frame_rate = Some(30.0);
            p
        },
        {
            let mut p = preset_config_to_py(
                "fast_preview",
                "Fast preview VP9/Opus 480p",
                &oximedia_transcode::presets::vp9_opus(854, 480, 1_000_000, 64_000),
            );
            p.quality_mode = "low".to_string();
            p
        },
    ]
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all preset bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPreset>()?;
    m.add_class::<PyPresetManager>()?;
    m.add_function(wrap_pyfunction!(list_presets, m)?)?;
    m.add_function(wrap_pyfunction!(get_preset, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_preset_count() {
        let presets = builtin_preset_list();
        assert_eq!(presets.len(), 6);
    }

    #[test]
    fn test_preset_to_dict() {
        let presets = builtin_preset_list();
        let first = &presets[0];
        let d = first.to_dict();
        assert_eq!(d.get("name").map(|s| s.as_str()), Some("web_optimized"));
        assert!(d.contains_key("video_codec"));
    }

    #[test]
    fn test_preset_to_json() {
        let presets = builtin_preset_list();
        let json = presets[0].to_json();
        assert!(json.is_ok());
        let s = json.expect("should serialize");
        assert!(s.contains("web_optimized"));
    }

    #[test]
    fn test_preset_manager_lifecycle() {
        let mut mgr = PyPresetManager::new();
        assert!(mgr.list_names().len() >= 6);

        let custom = PyPreset {
            name: "test_custom".to_string(),
            description: "Test".to_string(),
            video_codec: "av1".to_string(),
            audio_codec: "opus".to_string(),
            container: "webm".to_string(),
            video_bitrate: Some(1_000_000),
            audio_bitrate: Some(64_000),
            crf: None,
            width: Some(640),
            height: Some(480),
            frame_rate: Some(24.0),
            sample_rate: None,
            quality_mode: "low".to_string(),
            custom_params: HashMap::new(),
        };

        assert!(mgr.add(custom).is_ok());
        assert!(mgr.get("test_custom").is_some());
        assert!(mgr.remove("test_custom").is_ok());
        assert!(mgr.get("test_custom").is_none());
    }

    #[test]
    fn test_get_preset_found() {
        let p = get_preset("web_optimized");
        assert!(p.is_ok());
        let p = p.expect("should find preset");
        assert_eq!(p.video_codec, "av1");
    }

    #[test]
    fn test_get_preset_not_found() {
        let p = get_preset("nonexistent");
        assert!(p.is_err());
    }
}
