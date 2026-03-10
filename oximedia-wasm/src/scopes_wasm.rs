//! WebAssembly bindings for video scopes from `oximedia-scopes`.
//!
//! Provides `WasmVideoScopes` for generating waveform, vectorscope, and histogram
//! displays from raw RGB24 frame data in the browser, plus standalone functions
//! for exposure analysis and false-color mapping.

use oximedia_scopes::{
    false_color, GamutColorspace, HistogramMode, ScopeConfig, ScopeType, VectorscopeMode,
    VideoScopes, WaveformMode,
};
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// WasmVideoScopes
// ---------------------------------------------------------------------------

/// Video scopes analyser for browser-side frame analysis.
///
/// All frame data parameters are raw RGB24 (3 bytes per pixel, row-major).
/// All scope rendering output is RGBA (4 bytes per pixel, row-major).
#[wasm_bindgen]
pub struct WasmVideoScopes {
    width: u32,
    height: u32,
    show_graticule: bool,
}

#[wasm_bindgen]
impl WasmVideoScopes {
    /// Create a new scopes analyser with the given display dimensions.
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            show_graticule: true,
        }
    }

    /// Enable or disable graticule overlay on rendered scopes.
    pub fn set_graticule(&mut self, enable: bool) {
        self.show_graticule = enable;
    }

    /// Render a waveform scope from an RGB24 frame.
    ///
    /// `mode` is one of: `"luma"`, `"rgb_parade"`, `"rgb_overlay"`, `"ycbcr"`.
    ///
    /// Returns RGBA pixel data of dimensions `self.width * self.height * 4`.
    ///
    /// # Errors
    ///
    /// Returns an error if the mode is unknown or if frame data is invalid.
    pub fn render_waveform(
        &self,
        frame_data: &[u8],
        frame_width: u32,
        frame_height: u32,
        mode: &str,
    ) -> Result<Vec<u8>, JsValue> {
        let scope_type = parse_waveform_mode(mode)?;
        let config = self.build_config();

        let scopes = VideoScopes::new(config);
        let result = scopes
            .analyze(frame_data, frame_width, frame_height, scope_type)
            .map_err(|e| crate::utils::js_err(&format!("Waveform analysis failed: {e}")))?;
        Ok(result.data)
    }

    /// Render a vectorscope from an RGB24 frame.
    ///
    /// `gain` controls the zoom level (1.0 = normal, 2.0 = 2x zoom).
    ///
    /// Returns RGBA pixel data.
    ///
    /// # Errors
    ///
    /// Returns an error if frame data is invalid.
    pub fn render_vectorscope(
        &self,
        frame_data: &[u8],
        frame_width: u32,
        frame_height: u32,
        gain: f64,
    ) -> Result<Vec<u8>, JsValue> {
        let mut config = self.build_config();
        config.vectorscope_gain = gain as f32;

        let scopes = VideoScopes::new(config);
        let result = scopes
            .analyze(
                frame_data,
                frame_width,
                frame_height,
                ScopeType::Vectorscope,
            )
            .map_err(|e| crate::utils::js_err(&format!("Vectorscope analysis failed: {e}")))?;
        Ok(result.data)
    }

    /// Render a histogram from an RGB24 frame.
    ///
    /// `mode` is one of: `"rgb"`, `"luma"`, `"overlay"`, `"stacked"`, `"logarithmic"`.
    ///
    /// Returns RGBA pixel data.
    ///
    /// # Errors
    ///
    /// Returns an error if the mode is unknown or if frame data is invalid.
    pub fn render_histogram(
        &self,
        frame_data: &[u8],
        frame_width: u32,
        frame_height: u32,
        mode: &str,
    ) -> Result<Vec<u8>, JsValue> {
        let (scope_type, hist_mode) = parse_histogram_mode(mode)?;
        let mut config = self.build_config();
        config.histogram_mode = hist_mode;

        let scopes = VideoScopes::new(config);
        let result = scopes
            .analyze(frame_data, frame_width, frame_height, scope_type)
            .map_err(|e| crate::utils::js_err(&format!("Histogram analysis failed: {e}")))?;
        Ok(result.data)
    }

    /// Render a parade display from an RGB24 frame.
    ///
    /// `mode` is one of: `"rgb"`, `"ycbcr"`.
    ///
    /// Returns RGBA pixel data.
    ///
    /// # Errors
    ///
    /// Returns an error if the mode is unknown or if frame data is invalid.
    pub fn render_parade(
        &self,
        frame_data: &[u8],
        frame_width: u32,
        frame_height: u32,
        mode: &str,
    ) -> Result<Vec<u8>, JsValue> {
        let scope_type = match mode {
            "rgb" => ScopeType::ParadeRgb,
            "ycbcr" => ScopeType::ParadeYcbcr,
            other => {
                return Err(crate::utils::js_err(&format!(
                    "Unknown parade mode '{other}'. Use: rgb, ycbcr"
                )))
            }
        };
        let config = self.build_config();

        let scopes = VideoScopes::new(config);
        let result = scopes
            .analyze(frame_data, frame_width, frame_height, scope_type)
            .map_err(|e| crate::utils::js_err(&format!("Parade analysis failed: {e}")))?;
        Ok(result.data)
    }
}

impl WasmVideoScopes {
    fn build_config(&self) -> ScopeConfig {
        ScopeConfig {
            width: self.width,
            height: self.height,
            show_graticule: self.show_graticule,
            show_labels: self.show_graticule,
            anti_alias: true,
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
// Standalone functions
// ---------------------------------------------------------------------------

/// Analyse frame exposure and return a JSON string with statistics.
///
/// Input is RGB24 data (`width * height * 3` bytes).
///
/// Returned JSON keys: `min`, `max`, `mean`, `median`, `std_dev`,
/// `clipping_low_pct`, `clipping_high_pct`, `good_exposure_pct`.
///
/// # Errors
///
/// Returns an error if frame data is too small.
#[wasm_bindgen]
pub fn wasm_analyze_exposure(
    frame_data: &[u8],
    width: u32,
    height: u32,
) -> Result<String, JsValue> {
    let expected = (width * height * 3) as usize;
    if frame_data.len() < expected {
        return Err(crate::utils::js_err(&format!(
            "Frame data too small: need {expected} bytes for {width}x{height} RGB24, got {}",
            frame_data.len()
        )));
    }

    let pixel_count = (width * height) as usize;
    let mut luma_values = Vec::with_capacity(pixel_count);

    for i in 0..pixel_count {
        let idx = i * 3;
        let r = f64::from(frame_data[idx]);
        let g = f64::from(frame_data[idx + 1]);
        let b = f64::from(frame_data[idx + 2]);
        let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        luma_values.push(y);
    }

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

    let low_clip = luma_values.iter().filter(|&&v| v < 16.0).count() as f64;
    let high_clip = luma_values.iter().filter(|&&v| v > 235.0).count() as f64;
    let good = luma_values
        .iter()
        .filter(|&&v| (20.0..=230.0).contains(&v))
        .count() as f64;

    let json = serde_json::json!({
        "min": min_val,
        "max": max_val,
        "mean": mean,
        "median": median,
        "std_dev": std_dev,
        "clipping_low_pct": (low_clip / n) * 100.0,
        "clipping_high_pct": (high_clip / n) * 100.0,
        "good_exposure_pct": (good / n) * 100.0,
    });

    serde_json::to_string(&json)
        .map_err(|e| crate::utils::js_err(&format!("JSON serialisation failed: {e}")))
}

/// Apply false-color exposure mapping to an RGB24 frame.
///
/// Returns RGBA pixel data (`width * height * 4` bytes) with false-color
/// overlay applied using the default IRE zones.
///
/// # Errors
///
/// Returns an error if frame data is too small.
#[wasm_bindgen]
pub fn wasm_false_color(frame_data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, JsValue> {
    let expected = (width * height * 3) as usize;
    if frame_data.len() < expected {
        return Err(crate::utils::js_err(&format!(
            "Frame data too small: need {expected} bytes, got {}",
            frame_data.len()
        )));
    }

    let scale = false_color::FalseColorScale::default();
    let result = false_color::generate_false_color(
        frame_data,
        width,
        height,
        false_color::FalseColorMode::Ire,
        &scale,
    )
    .map_err(|e| crate::utils::js_err(&format!("False color generation failed: {e}")))?;

    Ok(result.data)
}

/// List available scope types as a JSON array of strings.
///
/// # Errors
///
/// Returns an error if JSON serialisation fails.
#[wasm_bindgen]
pub fn wasm_scope_types() -> Result<String, JsValue> {
    let types = vec![
        "waveform_luma",
        "waveform_rgb_parade",
        "waveform_rgb_overlay",
        "waveform_ycbcr",
        "vectorscope",
        "histogram_rgb",
        "histogram_luma",
        "parade_rgb",
        "parade_ycbcr",
        "false_color",
        "cie_diagram",
        "focus_assist",
        "hdr_waveform",
    ];
    serde_json::to_string(&types)
        .map_err(|e| crate::utils::js_err(&format!("JSON serialisation failed: {e}")))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn resolve_waveform_mode(mode: &str) -> Result<ScopeType, String> {
    match mode {
        "luma" => Ok(ScopeType::WaveformLuma),
        "rgb_parade" | "rgb-parade" => Ok(ScopeType::WaveformRgbParade),
        "rgb_overlay" | "rgb-overlay" => Ok(ScopeType::WaveformRgbOverlay),
        "ycbcr" => Ok(ScopeType::WaveformYcbcr),
        other => Err(format!(
            "Unknown waveform mode '{other}'. Use: luma, rgb_parade, rgb_overlay, ycbcr"
        )),
    }
}

fn parse_waveform_mode(mode: &str) -> Result<ScopeType, JsValue> {
    resolve_waveform_mode(mode).map_err(|e| crate::utils::js_err(&e))
}

fn resolve_histogram_mode(mode: &str) -> Result<(ScopeType, HistogramMode), String> {
    match mode {
        "rgb" => Ok((ScopeType::HistogramRgb, HistogramMode::Overlay)),
        "luma" => Ok((ScopeType::HistogramLuma, HistogramMode::Overlay)),
        "overlay" => Ok((ScopeType::HistogramRgb, HistogramMode::Overlay)),
        "stacked" => Ok((ScopeType::HistogramRgb, HistogramMode::Stacked)),
        "logarithmic" | "log" => Ok((ScopeType::HistogramRgb, HistogramMode::Logarithmic)),
        other => Err(format!(
            "Unknown histogram mode '{other}'. Use: rgb, luma, overlay, stacked, logarithmic"
        )),
    }
}

fn parse_histogram_mode(mode: &str) -> Result<(ScopeType, HistogramMode), JsValue> {
    resolve_histogram_mode(mode).map_err(|e| crate::utils::js_err(&e))
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

    // --- Internal helper tests (run on any target) ---

    #[test]
    fn test_resolve_waveform_mode_valid() {
        assert!(resolve_waveform_mode("luma").is_ok());
        assert!(resolve_waveform_mode("rgb_parade").is_ok());
        assert!(resolve_waveform_mode("rgb-parade").is_ok());
        assert!(resolve_waveform_mode("rgb_overlay").is_ok());
        assert!(resolve_waveform_mode("rgb-overlay").is_ok());
        assert!(resolve_waveform_mode("ycbcr").is_ok());
    }

    #[test]
    fn test_resolve_waveform_mode_invalid() {
        assert!(resolve_waveform_mode("invalid").is_err());
    }

    #[test]
    fn test_resolve_histogram_mode_valid() {
        for mode in &["rgb", "luma", "overlay", "stacked", "logarithmic", "log"] {
            assert!(
                resolve_histogram_mode(mode).is_ok(),
                "Failed for mode: {mode}"
            );
        }
    }

    #[test]
    fn test_resolve_histogram_mode_invalid() {
        assert!(resolve_histogram_mode("bad").is_err());
    }

    #[test]
    fn test_build_config_defaults() {
        let scopes = WasmVideoScopes {
            width: 256,
            height: 128,
            show_graticule: true,
        };
        let config = scopes.build_config();
        assert_eq!(config.width, 256);
        assert_eq!(config.height, 128);
        assert!(config.show_graticule);
        assert!(config.anti_alias);
    }

    #[test]
    fn test_set_graticule() {
        let mut scopes = WasmVideoScopes {
            width: 64,
            height: 64,
            show_graticule: true,
        };
        scopes.show_graticule = false;
        assert!(!scopes.show_graticule);
        scopes.show_graticule = true;
        assert!(scopes.show_graticule);
    }

    #[test]
    fn test_scope_analysis_via_internal_api() {
        let frame = test_frame(32, 32);
        let config = ScopeConfig {
            width: 64,
            height: 64,
            show_graticule: false,
            show_labels: false,
            anti_alias: false,
            waveform_mode: WaveformMode::Overlay,
            vectorscope_mode: VectorscopeMode::Circular,
            histogram_mode: HistogramMode::Overlay,
            vectorscope_gain: 1.0,
            highlight_gamut: false,
            gamut_colorspace: GamutColorspace::Rec709,
        };
        let scopes = VideoScopes::new(config);
        let result = scopes.analyze(&frame, 32, 32, ScopeType::WaveformLuma);
        assert!(result.is_ok());
        let data = result.expect("waveform analysis should succeed");
        assert_eq!(data.data.len(), (64 * 64 * 4) as usize);
    }

    #[test]
    fn test_histogram_analysis_via_internal_api() {
        let frame = test_frame(32, 32);
        let config = ScopeConfig {
            width: 64,
            height: 64,
            show_graticule: false,
            show_labels: false,
            anti_alias: false,
            waveform_mode: WaveformMode::Overlay,
            vectorscope_mode: VectorscopeMode::Circular,
            histogram_mode: HistogramMode::Overlay,
            vectorscope_gain: 1.0,
            highlight_gamut: false,
            gamut_colorspace: GamutColorspace::Rec709,
        };
        let scopes = VideoScopes::new(config);
        for scope_type in &[ScopeType::HistogramRgb, ScopeType::HistogramLuma] {
            let result = scopes.analyze(&frame, 32, 32, *scope_type);
            assert!(result.is_ok(), "Failed for scope type: {scope_type:?}");
        }
    }

    #[test]
    fn test_vectorscope_analysis_via_internal_api() {
        let frame = test_frame(32, 32);
        let config = ScopeConfig {
            width: 64,
            height: 64,
            show_graticule: false,
            show_labels: false,
            anti_alias: false,
            waveform_mode: WaveformMode::Overlay,
            vectorscope_mode: VectorscopeMode::Circular,
            histogram_mode: HistogramMode::Overlay,
            vectorscope_gain: 1.0,
            highlight_gamut: false,
            gamut_colorspace: GamutColorspace::Rec709,
        };
        let scopes = VideoScopes::new(config);
        let result = scopes.analyze(&frame, 32, 32, ScopeType::Vectorscope);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parade_analysis_via_internal_api() {
        let frame = test_frame(32, 32);
        let config = ScopeConfig {
            width: 96,
            height: 64,
            show_graticule: false,
            show_labels: false,
            anti_alias: false,
            waveform_mode: WaveformMode::Overlay,
            vectorscope_mode: VectorscopeMode::Circular,
            histogram_mode: HistogramMode::Overlay,
            vectorscope_gain: 1.0,
            highlight_gamut: false,
            gamut_colorspace: GamutColorspace::Rec709,
        };
        let scopes = VideoScopes::new(config);
        let result = scopes.analyze(&frame, 32, 32, ScopeType::ParadeRgb);
        assert!(result.is_ok());
    }

    #[test]
    fn test_false_color_via_internal_api() {
        let frame = test_frame(32, 32);
        let scale = false_color::FalseColorScale::default();
        let result = false_color::generate_false_color(
            &frame,
            32,
            32,
            false_color::FalseColorMode::Ire,
            &scale,
        );
        assert!(result.is_ok());
        let data = result.expect("false color should succeed");
        assert_eq!(data.data.len(), (32 * 32 * 4) as usize);
    }

    #[test]
    fn test_exposure_stats_computation() {
        // Test the exposure analysis logic directly
        let frame = test_frame(16, 16);
        let pixel_count = 16 * 16;
        let mut luma_values = Vec::with_capacity(pixel_count);
        for i in 0..pixel_count {
            let idx = i * 3;
            let r = f64::from(frame[idx]);
            let g = f64::from(frame[idx + 1]);
            let b = f64::from(frame[idx + 2]);
            let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            luma_values.push(y);
        }
        let sum: f64 = luma_values.iter().sum();
        let mean = sum / luma_values.len() as f64;
        assert!(mean > 0.0);
        assert!(mean < 255.0);
    }
}
