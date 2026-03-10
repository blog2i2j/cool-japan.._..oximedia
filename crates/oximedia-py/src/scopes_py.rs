//! Python bindings for video scopes from `oximedia-scopes`.
//!
//! Provides `PyVideoScopes`, `PyScopeConfig`, `PyScopeData`, plus standalone
//! utility functions for exposure analysis and scope type enumeration.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

use oximedia_scopes::{
    false_color, GamutColorspace, HistogramMode, ScopeConfig, ScopeData, ScopeType,
    VectorscopeMode, VideoScopes, WaveformMode,
};

// ---------------------------------------------------------------------------
// PyScopeConfig
// ---------------------------------------------------------------------------

/// Configuration for video scope rendering.
#[pyclass]
#[derive(Clone)]
pub struct PyScopeConfig {
    /// Width of the scope display in pixels.
    #[pyo3(get)]
    pub width: u32,

    /// Height of the scope display in pixels.
    #[pyo3(get)]
    pub height: u32,

    /// Whether to show graticule overlay.
    #[pyo3(get)]
    pub show_graticule: bool,

    /// Whether to show text labels.
    #[pyo3(get)]
    pub show_labels: bool,

    /// Whether to enable anti-aliasing.
    #[pyo3(get)]
    pub anti_alias: bool,
}

#[pymethods]
impl PyScopeConfig {
    /// Create a new scope configuration.
    ///
    /// Defaults to 512x512 with graticule, labels, and anti-alias enabled.
    #[new]
    #[pyo3(signature = (width=None, height=None))]
    fn new(width: Option<u32>, height: Option<u32>) -> Self {
        Self {
            width: width.unwrap_or(512),
            height: height.unwrap_or(512),
            show_graticule: true,
            show_labels: true,
            anti_alias: true,
        }
    }

    /// Enable or disable graticule overlay.
    fn with_graticule(&mut self, enable: bool) {
        self.show_graticule = enable;
    }

    /// Enable or disable text labels.
    fn with_labels(&mut self, enable: bool) {
        self.show_labels = enable;
    }

    /// Enable or disable anti-aliasing.
    fn with_anti_alias(&mut self, enable: bool) {
        self.anti_alias = enable;
    }

    fn __repr__(&self) -> String {
        format!(
            "PyScopeConfig(width={}, height={}, graticule={}, labels={}, anti_alias={})",
            self.width, self.height, self.show_graticule, self.show_labels, self.anti_alias
        )
    }
}

impl PyScopeConfig {
    /// Convert to the internal `ScopeConfig`.
    fn to_internal(&self) -> ScopeConfig {
        ScopeConfig {
            width: self.width,
            height: self.height,
            show_graticule: self.show_graticule,
            show_labels: self.show_labels,
            anti_alias: self.anti_alias,
            waveform_mode: WaveformMode::Overlay,
            vectorscope_mode: VectorscopeMode::Circular,
            histogram_mode: HistogramMode::Overlay,
            vectorscope_gain: 1.0,
            highlight_gamut: false,
            gamut_colorspace: GamutColorspace::Rec709,
        }
    }
}

// ---------------------------------------------------------------------------
// PyScopeData
// ---------------------------------------------------------------------------

/// Scope rendering result containing RGBA pixel data.
#[pyclass]
#[derive(Clone)]
pub struct PyScopeData {
    /// Width of the scope image.
    #[pyo3(get)]
    pub width: u32,

    /// Height of the scope image.
    #[pyo3(get)]
    pub height: u32,

    /// Type of scope that generated this data.
    #[pyo3(get)]
    pub scope_type: String,

    /// Internal RGBA pixel data.
    data: Vec<u8>,
}

#[pymethods]
impl PyScopeData {
    /// Return the raw RGBA pixel data as bytes.
    fn data_as_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        pyo3::types::PyBytes::new(py, &self.data)
    }

    /// Return a copy of the RGBA pixel data as a list of integers.
    fn to_rgba(&self) -> Vec<u8> {
        self.data.clone()
    }

    /// Total number of bytes in the RGBA buffer.
    fn byte_count(&self) -> usize {
        self.data.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyScopeData(type='{}', width={}, height={}, bytes={})",
            self.scope_type,
            self.width,
            self.height,
            self.data.len()
        )
    }
}

impl PyScopeData {
    fn from_internal(scope: &ScopeData) -> Self {
        let type_name = match scope.scope_type {
            ScopeType::WaveformLuma => "waveform_luma",
            ScopeType::WaveformRgbParade => "waveform_rgb_parade",
            ScopeType::WaveformRgbOverlay => "waveform_rgb_overlay",
            ScopeType::WaveformYcbcr => "waveform_ycbcr",
            ScopeType::Vectorscope => "vectorscope",
            ScopeType::HistogramRgb => "histogram_rgb",
            ScopeType::HistogramLuma => "histogram_luma",
            ScopeType::ParadeRgb => "parade_rgb",
            ScopeType::ParadeYcbcr => "parade_ycbcr",
            ScopeType::FalseColor => "false_color",
            ScopeType::CieDiagram => "cie_diagram",
            ScopeType::FocusAssist => "focus_assist",
            ScopeType::HdrWaveform => "hdr_waveform",
        };
        Self {
            width: scope.width,
            height: scope.height,
            scope_type: type_name.to_string(),
            data: scope.data.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// PyVideoScopes
// ---------------------------------------------------------------------------

/// Main video scopes analyser.
///
/// Wraps `oximedia_scopes::VideoScopes` and provides waveform, vectorscope,
/// histogram, parade, and false-color analysis from raw RGB24 frame data.
#[pyclass]
pub struct PyVideoScopes {
    inner: VideoScopes,
}

#[pymethods]
impl PyVideoScopes {
    /// Create a new video scopes analyser.
    ///
    /// If no config is provided, sensible defaults are used (512x512, graticule on).
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<&PyScopeConfig>) -> PyResult<Self> {
        let internal_config = config.map(PyScopeConfig::to_internal).unwrap_or_default();
        Ok(Self {
            inner: VideoScopes::new(internal_config),
        })
    }

    /// Generate a waveform scope from RGB24 frame data.
    ///
    /// `mode` can be: `luma`, `rgb_parade`, `rgb_overlay`, `ycbcr`.
    #[pyo3(signature = (frame_data, width, height, mode=None))]
    fn waveform(
        &self,
        frame_data: Vec<u8>,
        width: u32,
        height: u32,
        mode: Option<&str>,
    ) -> PyResult<PyScopeData> {
        let scope_type = parse_waveform_mode(mode.unwrap_or("luma"))?;
        let result = self
            .inner
            .analyze(&frame_data, width, height, scope_type)
            .map_err(|e| PyRuntimeError::new_err(format!("Waveform analysis failed: {e}")))?;
        Ok(PyScopeData::from_internal(&result))
    }

    /// Generate a vectorscope from RGB24 frame data.
    ///
    /// `mode` can be: `circular`, `rectangular`.
    #[pyo3(signature = (frame_data, width, height, mode=None, gain=None))]
    fn vectorscope(
        &self,
        frame_data: Vec<u8>,
        width: u32,
        height: u32,
        mode: Option<&str>,
        gain: Option<f64>,
    ) -> PyResult<PyScopeData> {
        let vs_mode = match mode.unwrap_or("circular") {
            "circular" => VectorscopeMode::Circular,
            "rectangular" => VectorscopeMode::Rectangular,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown vectorscope mode '{other}'. Use: circular, rectangular"
                )))
            }
        };

        // Build config with vectorscope-specific settings
        let mut config = self.inner.config().clone();
        config.vectorscope_mode = vs_mode;
        if let Some(g) = gain {
            config.vectorscope_gain = g as f32;
        }

        let scopes = VideoScopes::new(config);
        let result = scopes
            .analyze(&frame_data, width, height, ScopeType::Vectorscope)
            .map_err(|e| PyRuntimeError::new_err(format!("Vectorscope analysis failed: {e}")))?;
        Ok(PyScopeData::from_internal(&result))
    }

    /// Generate a histogram from RGB24 frame data.
    ///
    /// `mode` can be: `rgb`, `luma`, `overlay`, `stacked`, `logarithmic`.
    #[pyo3(signature = (frame_data, width, height, mode=None))]
    fn histogram(
        &self,
        frame_data: Vec<u8>,
        width: u32,
        height: u32,
        mode: Option<&str>,
    ) -> PyResult<PyScopeData> {
        let (scope_type, hist_mode) = parse_histogram_mode(mode.unwrap_or("rgb"))?;
        let mut config = self.inner.config().clone();
        config.histogram_mode = hist_mode;

        let scopes = VideoScopes::new(config);
        let result = scopes
            .analyze(&frame_data, width, height, scope_type)
            .map_err(|e| PyRuntimeError::new_err(format!("Histogram analysis failed: {e}")))?;
        Ok(PyScopeData::from_internal(&result))
    }

    /// Generate a parade display from RGB24 frame data.
    ///
    /// `mode` can be: `rgb`, `ycbcr`.
    #[pyo3(signature = (frame_data, width, height, mode=None))]
    fn parade(
        &self,
        frame_data: Vec<u8>,
        width: u32,
        height: u32,
        mode: Option<&str>,
    ) -> PyResult<PyScopeData> {
        let scope_type = match mode.unwrap_or("rgb") {
            "rgb" => ScopeType::ParadeRgb,
            "ycbcr" => ScopeType::ParadeYcbcr,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown parade mode '{other}'. Use: rgb, ycbcr"
                )))
            }
        };

        let result = self
            .inner
            .analyze(&frame_data, width, height, scope_type)
            .map_err(|e| PyRuntimeError::new_err(format!("Parade analysis failed: {e}")))?;
        Ok(PyScopeData::from_internal(&result))
    }

    /// Generate a false color exposure visualization from RGB24 frame data.
    #[pyo3(signature = (frame_data, width, height))]
    fn false_color(&self, frame_data: Vec<u8>, width: u32, height: u32) -> PyResult<PyScopeData> {
        let result = self
            .inner
            .analyze(&frame_data, width, height, ScopeType::FalseColor)
            .map_err(|e| PyRuntimeError::new_err(format!("False color analysis failed: {e}")))?;
        Ok(PyScopeData::from_internal(&result))
    }

    /// Update the configuration of this scope analyser.
    fn set_config(&mut self, config: &PyScopeConfig) -> PyResult<()> {
        self.inner.set_config(config.to_internal());
        Ok(())
    }

    fn __repr__(&self) -> String {
        let cfg = self.inner.config();
        format!(
            "PyVideoScopes(width={}, height={}, graticule={})",
            cfg.width, cfg.height, cfg.show_graticule
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// List all available scope types.
#[pyfunction]
pub fn list_scope_types() -> Vec<String> {
    vec![
        "waveform_luma".to_string(),
        "waveform_rgb_parade".to_string(),
        "waveform_rgb_overlay".to_string(),
        "waveform_ycbcr".to_string(),
        "vectorscope".to_string(),
        "histogram_rgb".to_string(),
        "histogram_luma".to_string(),
        "parade_rgb".to_string(),
        "parade_ycbcr".to_string(),
        "false_color".to_string(),
        "cie_diagram".to_string(),
        "focus_assist".to_string(),
        "hdr_waveform".to_string(),
    ]
}

/// Analyse exposure statistics from an RGB24 frame.
///
/// Returns a dictionary with keys: `min`, `max`, `mean`, `median`, `std_dev`,
/// `clipping_low_pct`, `clipping_high_pct`, `good_exposure_pct`.
#[pyfunction]
pub fn analyze_exposure(
    frame_data: Vec<u8>,
    width: u32,
    height: u32,
) -> PyResult<HashMap<String, f64>> {
    let expected = (width * height * 3) as usize;
    if frame_data.len() < expected {
        return Err(PyValueError::new_err(format!(
            "Frame data too small: need {expected} bytes for {width}x{height} RGB24, got {}",
            frame_data.len()
        )));
    }

    // Compute luma for every pixel
    let pixel_count = (width * height) as usize;
    let mut luma_values = Vec::with_capacity(pixel_count);

    for i in 0..pixel_count {
        let idx = i * 3;
        let r = f64::from(frame_data[idx]);
        let g = f64::from(frame_data[idx + 1]);
        let b = f64::from(frame_data[idx + 2]);
        // BT.709 luma
        let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        luma_values.push(y);
    }

    // Statistics
    let n = luma_values.len() as f64;
    let sum: f64 = luma_values.iter().sum();
    let mean = sum / n;

    let mut sorted = luma_values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min_val = sorted.first().copied().unwrap_or(0.0);
    let max_val = sorted.last().copied().unwrap_or(0.0);
    let median = if sorted.len() % 2 == 0 {
        let mid = sorted.len() / 2;
        (sorted[mid.saturating_sub(1)] + sorted[mid]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    let variance: f64 = luma_values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    // Clipping thresholds (broadcast legal: 16-235)
    let low_clip_count = luma_values.iter().filter(|&&v| v < 16.0).count() as f64;
    let high_clip_count = luma_values.iter().filter(|&&v| v > 235.0).count() as f64;
    let good_count = luma_values
        .iter()
        .filter(|&&v| (20.0..=230.0).contains(&v))
        .count() as f64;

    let mut result = HashMap::new();
    result.insert("min".to_string(), min_val);
    result.insert("max".to_string(), max_val);
    result.insert("mean".to_string(), mean);
    result.insert("median".to_string(), median);
    result.insert("std_dev".to_string(), std_dev);
    result.insert("clipping_low_pct".to_string(), (low_clip_count / n) * 100.0);
    result.insert(
        "clipping_high_pct".to_string(),
        (high_clip_count / n) * 100.0,
    );
    result.insert("good_exposure_pct".to_string(), (good_count / n) * 100.0);

    Ok(result)
}

/// Compute false-color exposure statistics from an RGB24 frame.
///
/// Returns a dictionary with zone distribution and clipping percentages.
#[pyfunction]
pub fn false_color_stats(
    frame_data: Vec<u8>,
    width: u32,
    height: u32,
) -> PyResult<HashMap<String, f64>> {
    let expected = (width * height * 3) as usize;
    if frame_data.len() < expected {
        return Err(PyValueError::new_err(format!(
            "Frame data too small: need {expected} bytes, got {}",
            frame_data.len()
        )));
    }

    let stats = false_color::compute_false_color_stats(&frame_data, width, height);
    let mut result = HashMap::new();
    result.insert(
        "highlight_clip_pct".to_string(),
        f64::from(stats.highlight_clip_percent),
    );
    result.insert(
        "shadow_clip_pct".to_string(),
        f64::from(stats.shadow_clip_percent),
    );
    result.insert(
        "good_exposure_pct".to_string(),
        f64::from(stats.good_exposure_percent),
    );
    for (name, pct) in &stats.zone_distribution {
        result.insert(name.clone(), f64::from(*pct));
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all scope bindings on the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScopeConfig>()?;
    m.add_class::<PyScopeData>()?;
    m.add_class::<PyVideoScopes>()?;
    m.add_function(wrap_pyfunction!(list_scope_types, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_exposure, m)?)?;
    m.add_function(wrap_pyfunction!(false_color_stats, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn parse_waveform_mode(mode: &str) -> PyResult<ScopeType> {
    match mode.to_lowercase().as_str() {
        "luma" => Ok(ScopeType::WaveformLuma),
        "rgb_parade" | "rgb-parade" => Ok(ScopeType::WaveformRgbParade),
        "rgb_overlay" | "rgb-overlay" => Ok(ScopeType::WaveformRgbOverlay),
        "ycbcr" => Ok(ScopeType::WaveformYcbcr),
        other => Err(PyValueError::new_err(format!(
            "Unknown waveform mode '{other}'. Use: luma, rgb_parade, rgb_overlay, ycbcr"
        ))),
    }
}

fn parse_histogram_mode(mode: &str) -> PyResult<(ScopeType, HistogramMode)> {
    match mode.to_lowercase().as_str() {
        "rgb" => Ok((ScopeType::HistogramRgb, HistogramMode::Overlay)),
        "luma" => Ok((ScopeType::HistogramLuma, HistogramMode::Overlay)),
        "overlay" => Ok((ScopeType::HistogramRgb, HistogramMode::Overlay)),
        "stacked" => Ok((ScopeType::HistogramRgb, HistogramMode::Stacked)),
        "logarithmic" | "log" => Ok((ScopeType::HistogramRgb, HistogramMode::Logarithmic)),
        other => Err(PyValueError::new_err(format!(
            "Unknown histogram mode '{other}'. Use: rgb, luma, overlay, stacked, logarithmic"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_frame(width: u32, height: u32) -> Vec<u8> {
        let mut data = vec![0u8; (width * height * 3) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                data[idx] = (x & 0xFF) as u8;
                data[idx + 1] = (y & 0xFF) as u8;
                data[idx + 2] = 128;
            }
        }
        data
    }

    #[test]
    fn test_scope_config_defaults() {
        let cfg = PyScopeConfig::new(None, None);
        assert_eq!(cfg.width, 512);
        assert_eq!(cfg.height, 512);
        assert!(cfg.show_graticule);
    }

    #[test]
    fn test_scope_config_custom() {
        let cfg = PyScopeConfig::new(Some(256), Some(128));
        assert_eq!(cfg.width, 256);
        assert_eq!(cfg.height, 128);
    }

    #[test]
    fn test_scope_config_to_internal() {
        let cfg = PyScopeConfig::new(Some(64), Some(64));
        let internal = cfg.to_internal();
        assert_eq!(internal.width, 64);
        assert_eq!(internal.height, 64);
    }

    #[test]
    fn test_list_scope_types() {
        let types = list_scope_types();
        assert!(types.len() >= 10);
        assert!(types.contains(&"waveform_luma".to_string()));
        assert!(types.contains(&"vectorscope".to_string()));
    }

    #[test]
    fn test_parse_waveform_mode() {
        assert!(parse_waveform_mode("luma").is_ok());
        assert!(parse_waveform_mode("rgb_parade").is_ok());
        assert!(parse_waveform_mode("rgb-overlay").is_ok());
        assert!(parse_waveform_mode("ycbcr").is_ok());
        assert!(parse_waveform_mode("bad").is_err());
    }

    #[test]
    fn test_parse_histogram_mode() {
        assert!(parse_histogram_mode("rgb").is_ok());
        assert!(parse_histogram_mode("luma").is_ok());
        assert!(parse_histogram_mode("stacked").is_ok());
        assert!(parse_histogram_mode("logarithmic").is_ok());
        assert!(parse_histogram_mode("bad").is_err());
    }

    #[test]
    fn test_analyze_exposure_basic() {
        let frame = test_frame(32, 32);
        let result = analyze_exposure(frame, 32, 32);
        assert!(result.is_ok());
        let stats = result.expect("analysis should succeed");
        assert!(stats.contains_key("mean"));
        assert!(stats.contains_key("std_dev"));
        assert!(stats.contains_key("clipping_low_pct"));
        assert!(stats.contains_key("clipping_high_pct"));
    }

    #[test]
    fn test_analyze_exposure_bad_size() {
        let result = analyze_exposure(vec![0u8; 10], 100, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_false_color_stats_basic() {
        let frame = test_frame(32, 32);
        let result = false_color_stats(frame, 32, 32);
        assert!(result.is_ok());
        let stats = result.expect("stats should succeed");
        assert!(stats.contains_key("good_exposure_pct"));
    }

    #[test]
    fn test_scope_data_from_internal() {
        let internal = ScopeData {
            width: 64,
            height: 64,
            data: vec![0u8; 64 * 64 * 4],
            scope_type: ScopeType::WaveformLuma,
        };
        let py_data = PyScopeData::from_internal(&internal);
        assert_eq!(py_data.width, 64);
        assert_eq!(py_data.scope_type, "waveform_luma");
        assert_eq!(py_data.byte_count(), 64 * 64 * 4);
    }

    #[test]
    fn test_scope_data_repr() {
        let data = PyScopeData {
            width: 100,
            height: 100,
            scope_type: "histogram_rgb".to_string(),
            data: vec![0u8; 40000],
        };
        let repr = data.__repr__();
        assert!(repr.contains("histogram_rgb"));
        assert!(repr.contains("100"));
    }
}
