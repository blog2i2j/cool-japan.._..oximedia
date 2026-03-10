//! Python bindings for `oximedia-graphics` broadcast graphics engine.
//!
//! Provides `PyGraphicsEngine`, `PyTemplate`, `PyLowerThird`, and standalone
//! functions for rendering broadcast graphics from Python.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a hex color string into an RGBA array.
fn parse_hex_color_py(hex: &str) -> PyResult<[u8; 4]> {
    let hex = hex.trim_start_matches('#');
    match hex.len() {
        6 => {
            let r = u8::from_str_radix(&hex[0..2], 16)
                .map_err(|_| PyValueError::new_err(format!("Invalid hex color: {hex}")))?;
            let g = u8::from_str_radix(&hex[2..4], 16)
                .map_err(|_| PyValueError::new_err(format!("Invalid hex color: {hex}")))?;
            let b = u8::from_str_radix(&hex[4..6], 16)
                .map_err(|_| PyValueError::new_err(format!("Invalid hex color: {hex}")))?;
            Ok([r, g, b, 255])
        }
        8 => {
            let r = u8::from_str_radix(&hex[0..2], 16)
                .map_err(|_| PyValueError::new_err(format!("Invalid hex color: {hex}")))?;
            let g = u8::from_str_radix(&hex[2..4], 16)
                .map_err(|_| PyValueError::new_err(format!("Invalid hex color: {hex}")))?;
            let b = u8::from_str_radix(&hex[4..6], 16)
                .map_err(|_| PyValueError::new_err(format!("Invalid hex color: {hex}")))?;
            let a = u8::from_str_radix(&hex[6..8], 16)
                .map_err(|_| PyValueError::new_err(format!("Invalid hex color: {hex}")))?;
            Ok([r, g, b, a])
        }
        _ => Err(PyValueError::new_err(format!(
            "Invalid hex color '{}': expected 6 or 8 hex characters",
            hex
        ))),
    }
}

/// Convert RGBA tuple to array.
fn rgba_tuple_to_array(r: u8, g: u8, b: u8, a: u8) -> [u8; 4] {
    [r, g, b, a]
}

// ---------------------------------------------------------------------------
// PyGraphicsEngine
// ---------------------------------------------------------------------------

/// High-level graphics rendering engine.
#[pyclass]
pub struct PyGraphicsEngine {
    #[pyo3(get)]
    width: u32,
    #[pyo3(get)]
    height: u32,
    background_color: [u8; 4],
}

#[pymethods]
impl PyGraphicsEngine {
    /// Create a new graphics engine with the given resolution.
    #[new]
    #[pyo3(signature = (width=1920, height=1080))]
    fn new(width: u32, height: u32) -> PyResult<Self> {
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err("Width and height must be > 0"));
        }
        Ok(Self {
            width,
            height,
            background_color: [0, 0, 0, 0],
        })
    }

    /// Set the background color as RGBA.
    fn set_background(&mut self, r: u8, g: u8, b: u8, a: u8) {
        self.background_color = rgba_tuple_to_array(r, g, b, a);
    }

    /// Render a lower-third graphic.
    ///
    /// Args:
    ///     title: Primary text.
    ///     subtitle: Secondary text (optional).
    ///     config: Optional dict with keys: bg_color, text_color, style, duration.
    ///
    /// Returns:
    ///     RGBA pixel data as bytes.
    #[pyo3(signature = (title, subtitle=None, config=None))]
    fn render_lower_third(
        &self,
        title: &str,
        subtitle: Option<&str>,
        config: Option<HashMap<String, String>>,
    ) -> PyResult<Vec<u8>> {
        let conf = config.unwrap_or_default();

        let mut lt_config = oximedia_graphics::lower_third::LowerThirdConfig {
            name: title.to_string(),
            title: subtitle.unwrap_or("").to_string(),
            subtitle: subtitle.map(String::from),
            ..oximedia_graphics::lower_third::LowerThirdConfig::default()
        };

        if let Some(bg) = conf.get("bg_color") {
            lt_config.background_color = parse_hex_color_py(bg)?;
        }
        if let Some(tc) = conf.get("text_color") {
            lt_config.text_color = parse_hex_color_py(tc)?;
        }
        if let Some(style_str) = conf.get("style") {
            lt_config.style = match style_str.as_str() {
                "classic" => oximedia_graphics::lower_third::LowerThirdStyle::Classic,
                "modern" => oximedia_graphics::lower_third::LowerThirdStyle::Modern,
                "minimal" => oximedia_graphics::lower_third::LowerThirdStyle::Minimal,
                "news" => oximedia_graphics::lower_third::LowerThirdStyle::News,
                "sports" => oximedia_graphics::lower_third::LowerThirdStyle::Sports,
                "corporate" => oximedia_graphics::lower_third::LowerThirdStyle::Corporate,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "Unknown style '{}'. Expected: classic, modern, minimal, news, sports, corporate",
                        other
                    )));
                }
            };
        }

        let duration = conf
            .get("duration")
            .and_then(|d| d.parse::<f64>().ok())
            .unwrap_or(3.0);
        let total_frames = (duration * 30.0) as u32;
        let frame_idx = total_frames / 2;

        Ok(oximedia_graphics::lower_third::LowerThirdRenderer::render(
            &lt_config,
            frame_idx,
            total_frames,
            self.width,
            self.height,
        ))
    }

    /// Render a ticker strip.
    ///
    /// Args:
    ///     text: Ticker text content.
    ///     speed: Scroll speed in pixels per second (optional).
    ///     config: Optional dict with keys: bg_color, text_color.
    ///
    /// Returns:
    ///     RGBA pixel data as bytes.
    #[pyo3(signature = (text, speed=None, config=None))]
    fn render_ticker(
        &self,
        text: &str,
        speed: Option<f64>,
        config: Option<HashMap<String, String>>,
    ) -> PyResult<Vec<u8>> {
        let conf = config.unwrap_or_default();

        let mut ticker_config = oximedia_graphics::ticker::TickerConfig::default();
        ticker_config.height_px = (self.height as f32 * 0.05).max(40.0) as u32;

        if let Some(s) = speed {
            ticker_config.scroll_speed_pps = s as f32;
        }
        if let Some(bg) = conf.get("bg_color") {
            ticker_config.bg_color = parse_hex_color_py(bg)?;
        }
        if let Some(tc) = conf.get("text_color") {
            ticker_config.text_color = parse_hex_color_py(tc)?;
        }

        let item = oximedia_graphics::ticker::TickerItem::new(text, None, 128);
        let state = oximedia_graphics::ticker::TickerState::new(vec![item]);

        Ok(oximedia_graphics::ticker::TickerRenderer::render(
            &state,
            &ticker_config,
            self.width,
        ))
    }

    /// Alpha-blend an overlay onto a base frame.
    ///
    /// Args:
    ///     base: Base RGBA pixel data.
    ///     overlay: Overlay RGBA pixel data.
    ///     x: Overlay X offset.
    ///     y: Overlay Y offset.
    ///     opacity: Blend opacity (0.0-1.0).
    ///     overlay_width: Width of overlay image.
    ///     overlay_height: Height of overlay image.
    ///
    /// Returns:
    ///     Composited RGBA pixel data.
    #[pyo3(signature = (base, overlay, x, y, opacity, overlay_width, overlay_height))]
    fn render_overlay(
        &self,
        base: Vec<u8>,
        overlay: Vec<u8>,
        x: i32,
        y: i32,
        opacity: f64,
        overlay_width: u32,
        overlay_height: u32,
    ) -> PyResult<Vec<u8>> {
        let expected_base = (self.width as usize) * (self.height as usize) * 4;
        if base.len() < expected_base {
            return Err(PyValueError::new_err(format!(
                "Base data too small: need {} bytes for {}x{} RGBA, got {}",
                expected_base,
                self.width,
                self.height,
                base.len()
            )));
        }

        let expected_overlay = (overlay_width as usize) * (overlay_height as usize) * 4;
        if overlay.len() < expected_overlay {
            return Err(PyValueError::new_err(format!(
                "Overlay data too small: need {} bytes for {}x{} RGBA, got {}",
                expected_overlay,
                overlay_width,
                overlay_height,
                overlay.len()
            )));
        }

        let mut result = base;
        let alpha = opacity.clamp(0.0, 1.0) as f32;

        for oy in 0..overlay_height {
            for ox in 0..overlay_width {
                let dst_x = ox as i32 + x;
                let dst_y = oy as i32 + y;

                if dst_x < 0
                    || dst_x >= self.width as i32
                    || dst_y < 0
                    || dst_y >= self.height as i32
                {
                    continue;
                }

                let src_idx = ((oy * overlay_width + ox) * 4) as usize;
                let dst_idx = ((dst_y as u32 * self.width + dst_x as u32) * 4) as usize;

                if src_idx + 3 >= overlay.len() || dst_idx + 3 >= result.len() {
                    continue;
                }

                let oa = (f32::from(overlay[src_idx + 3]) / 255.0) * alpha;
                let inv_a = 1.0 - oa;

                result[dst_idx] =
                    (f32::from(overlay[src_idx]) * oa + f32::from(result[dst_idx]) * inv_a) as u8;
                result[dst_idx + 1] = (f32::from(overlay[src_idx + 1]) * oa
                    + f32::from(result[dst_idx + 1]) * inv_a)
                    as u8;
                result[dst_idx + 2] = (f32::from(overlay[src_idx + 2]) * oa
                    + f32::from(result[dst_idx + 2]) * inv_a)
                    as u8;
                result[dst_idx + 3] = 255;
            }
        }

        Ok(result)
    }

    /// Render a named template.
    ///
    /// Args:
    ///     template_name: Template name (lower_third, full_screen_title, bug, watermark, color_bars).
    ///     params: Optional dict of template parameters.
    ///
    /// Returns:
    ///     RGBA pixel data as bytes.
    #[pyo3(signature = (template_name, params=None))]
    fn render_template(
        &self,
        template_name: &str,
        params: Option<HashMap<String, String>>,
    ) -> PyResult<Vec<u8>> {
        let p = params.unwrap_or_default();
        match template_name {
            "lower_third" => {
                let title = p.get("title").cloned().unwrap_or_else(|| "Title".to_string());
                let subtitle = p.get("subtitle").cloned();
                self.render_lower_third(&title, subtitle.as_deref(), Some(p))
            }
            "color_bars" => Ok(render_color_bars_impl(self.width, self.height)),
            "full_screen_title" => Ok(render_full_screen_title_impl(self.width, self.height, &p)),
            "bug" => Ok(render_bug_impl(self.width, self.height, &p)),
            "watermark" => Ok(render_watermark_impl(self.width, self.height, &p)),
            other => Err(PyValueError::new_err(format!(
                "Unknown template '{}'. Available: lower_third, full_screen_title, bug, watermark, color_bars",
                other
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PyGraphicsEngine(width={}, height={}, bg=[{},{},{},{}])",
            self.width,
            self.height,
            self.background_color[0],
            self.background_color[1],
            self.background_color[2],
            self.background_color[3],
        )
    }
}

// ---------------------------------------------------------------------------
// PyTemplate
// ---------------------------------------------------------------------------

/// A configurable graphics template.
#[pyclass]
#[derive(Clone)]
pub struct PyTemplate {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    description: String,
    params: HashMap<String, String>,
}

#[pymethods]
impl PyTemplate {
    /// Create a new template by name.
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        let description = match name {
            "lower_third" => "Broadcast lower-third with title and subtitle",
            "full_screen_title" => "Full-screen title card",
            "bug" => "Channel bug (corner logo placeholder)",
            "watermark" => "Semi-transparent watermark pattern",
            "color_bars" => "SMPTE-style color bars test pattern",
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown template '{}'. Available: lower_third, full_screen_title, bug, watermark, color_bars",
                    other
                )));
            }
        };

        Ok(Self {
            name: name.to_string(),
            description: description.to_string(),
            params: HashMap::new(),
        })
    }

    /// Set a template parameter.
    fn set_param(&mut self, key: &str, value: &str) {
        self.params.insert(key.to_string(), value.to_string());
    }

    /// Get a template parameter value.
    fn get_param(&self, key: &str) -> Option<String> {
        self.params.get(key).cloned()
    }

    /// Get all parameters as a dict.
    fn params(&self) -> HashMap<String, String> {
        self.params.clone()
    }

    /// Render the template at the given resolution.
    ///
    /// Returns RGBA pixel data as bytes.
    fn render(&self, width: u32, height: u32) -> PyResult<Vec<u8>> {
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err("Width and height must be > 0"));
        }

        match self.name.as_str() {
            "lower_third" => {
                let title = self.params.get("title").map_or("Title", |s| s.as_str());
                let mut config = oximedia_graphics::lower_third::LowerThirdConfig {
                    name: title.to_string(),
                    title: self
                        .params
                        .get("subtitle")
                        .map_or("", |s| s.as_str())
                        .to_string(),
                    subtitle: self.params.get("subtitle").cloned(),
                    ..oximedia_graphics::lower_third::LowerThirdConfig::default()
                };
                if let Some(bg) = self.params.get("bg_color") {
                    config.background_color = parse_hex_color_py(bg)?;
                }
                Ok(oximedia_graphics::lower_third::LowerThirdRenderer::render(
                    &config, 45, 90, width, height,
                ))
            }
            "color_bars" => Ok(render_color_bars_impl(width, height)),
            "full_screen_title" => Ok(render_full_screen_title_impl(width, height, &self.params)),
            "bug" => Ok(render_bug_impl(width, height, &self.params)),
            "watermark" => Ok(render_watermark_impl(width, height, &self.params)),
            _ => Err(PyRuntimeError::new_err("Template not implemented")),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PyTemplate(name='{}', params={})",
            self.name,
            self.params.len()
        )
    }
}

// ---------------------------------------------------------------------------
// PyLowerThird
// ---------------------------------------------------------------------------

/// Specialized lower-third builder.
#[pyclass]
#[derive(Clone)]
pub struct PyLowerThird {
    #[pyo3(get)]
    title: String,
    #[pyo3(get)]
    subtitle: Option<String>,
    bg_color: [u8; 4],
    text_color: [u8; 4],
    #[pyo3(get)]
    width: u32,
    #[pyo3(get)]
    height: u32,
    #[pyo3(get)]
    margin: u32,
}

#[pymethods]
impl PyLowerThird {
    /// Create a new lower-third with title and optional subtitle.
    #[new]
    #[pyo3(signature = (title, subtitle=None))]
    fn new(title: &str, subtitle: Option<&str>) -> Self {
        Self {
            title: title.to_string(),
            subtitle: subtitle.map(String::from),
            bg_color: [0, 0, 0, 200],
            text_color: [255, 255, 255, 255],
            width: 1920,
            height: 1080,
            margin: 40,
        }
    }

    /// Set background and text colors as hex strings.
    fn with_colors(&mut self, bg: &str, text: &str) -> PyResult<()> {
        self.bg_color = parse_hex_color_py(bg)?;
        self.text_color = parse_hex_color_py(text)?;
        Ok(())
    }

    /// Set output dimensions.
    fn with_size(&mut self, width: u32, height: u32) -> PyResult<()> {
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err("Width and height must be > 0"));
        }
        self.width = width;
        self.height = height;
        Ok(())
    }

    /// Set margin in pixels.
    fn with_margin(&mut self, margin: u32) {
        self.margin = margin;
    }

    /// Render the lower-third as RGBA bytes.
    fn render(&self) -> Vec<u8> {
        let config = oximedia_graphics::lower_third::LowerThirdConfig {
            name: self.title.clone(),
            title: self.subtitle.clone().unwrap_or_default(),
            subtitle: self.subtitle.clone(),
            background_color: self.bg_color,
            text_color: self.text_color,
            ..oximedia_graphics::lower_third::LowerThirdConfig::default()
        };

        oximedia_graphics::lower_third::LowerThirdRenderer::render(
            &config,
            45,
            90,
            self.width,
            self.height,
        )
    }

    /// Get background color as a tuple (r, g, b, a).
    fn bg_color(&self) -> (u8, u8, u8, u8) {
        (
            self.bg_color[0],
            self.bg_color[1],
            self.bg_color[2],
            self.bg_color[3],
        )
    }

    /// Get text color as a tuple (r, g, b, a).
    fn text_color(&self) -> (u8, u8, u8, u8) {
        (
            self.text_color[0],
            self.text_color[1],
            self.text_color[2],
            self.text_color[3],
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "PyLowerThird(title='{}', subtitle={:?}, size={}x{})",
            self.title, self.subtitle, self.width, self.height,
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// List available graphics templates.
///
/// Returns a list of template info dicts with name and description.
#[pyfunction]
pub fn list_templates() -> Vec<HashMap<String, String>> {
    let templates = vec![
        (
            "lower_third",
            "Broadcast lower-third with title and subtitle",
        ),
        ("full_screen_title", "Full-screen title card"),
        ("bug", "Channel bug (corner logo placeholder)"),
        ("watermark", "Semi-transparent watermark pattern"),
        ("color_bars", "SMPTE-style color bars test pattern"),
    ];

    templates
        .into_iter()
        .map(|(name, desc)| {
            let mut m = HashMap::new();
            m.insert("name".to_string(), name.to_string());
            m.insert("description".to_string(), desc.to_string());
            m
        })
        .collect()
}

/// Render SMPTE-style color bars test pattern.
///
/// Args:
///     width: Image width in pixels.
///     height: Image height in pixels.
///
/// Returns:
///     RGBA pixel data as bytes.
#[pyfunction]
pub fn render_color_bars(width: u32, height: u32) -> PyResult<Vec<u8>> {
    if width == 0 || height == 0 {
        return Err(PyValueError::new_err("Width and height must be > 0"));
    }
    Ok(render_color_bars_impl(width, height))
}

// ---------------------------------------------------------------------------
// Internal rendering helpers
// ---------------------------------------------------------------------------

fn render_color_bars_impl(width: u32, height: u32) -> Vec<u8> {
    let size = (width as usize) * (height as usize) * 4;
    let mut data = vec![0u8; size];

    let bars: [[u8; 3]; 7] = [
        [235, 235, 235],
        [235, 235, 16],
        [16, 235, 235],
        [16, 235, 16],
        [235, 16, 235],
        [235, 16, 16],
        [16, 16, 235],
    ];

    let bar_width = width / 7;

    for y in 0..height {
        for x in 0..width {
            let bar_idx = ((x / bar_width) as usize).min(bars.len() - 1);
            let color = bars[bar_idx];
            let idx = ((y * width + x) * 4) as usize;
            if idx + 3 < data.len() {
                data[idx] = color[0];
                data[idx + 1] = color[1];
                data[idx + 2] = color[2];
                data[idx + 3] = 255;
            }
        }
    }

    data
}

fn render_full_screen_title_impl(
    width: u32,
    height: u32,
    params: &HashMap<String, String>,
) -> Vec<u8> {
    let size = (width as usize) * (height as usize) * 4;
    let mut data = vec![0u8; size];

    let bg = params
        .get("bg_color")
        .and_then(|c| parse_hex_color_py(c).ok())
        .unwrap_or([20, 20, 30, 255]);

    for chunk in data.chunks_exact_mut(4) {
        chunk[0] = bg[0];
        chunk[1] = bg[1];
        chunk[2] = bg[2];
        chunk[3] = bg[3];
    }

    // Centered horizontal accent rule
    let rule_y = height / 2;
    let margin = width / 8;
    let accent = params
        .get("accent_color")
        .and_then(|c| parse_hex_color_py(c).ok())
        .unwrap_or([255, 165, 0, 255]);

    for dy in 0..4u32 {
        let y = rule_y + dy;
        if y >= height {
            break;
        }
        for x in margin..(width - margin) {
            let idx = ((y * width + x) * 4) as usize;
            if idx + 3 < data.len() {
                data[idx] = accent[0];
                data[idx + 1] = accent[1];
                data[idx + 2] = accent[2];
                data[idx + 3] = accent[3];
            }
        }
    }

    data
}

fn render_bug_impl(width: u32, height: u32, params: &HashMap<String, String>) -> Vec<u8> {
    let size = (width as usize) * (height as usize) * 4;
    let mut data = vec![0u8; size];

    let bug_size = params
        .get("size")
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(80);
    let margin = 40u32;
    let bug_color = params
        .get("color")
        .and_then(|c| parse_hex_color_py(c).ok())
        .unwrap_or([255, 255, 255, 180]);

    let start_x = width.saturating_sub(bug_size + margin);
    let start_y = margin;

    for dy in 0..bug_size {
        for dx in 0..bug_size {
            let x = start_x + dx;
            let y = start_y + dy;
            if x >= width || y >= height {
                continue;
            }
            let cx = bug_size as f32 / 2.0;
            let cy = bug_size as f32 / 2.0;
            let dist = ((dx as f32 - cx).powi(2) + (dy as f32 - cy).powi(2)).sqrt();
            if dist <= cx {
                let idx = ((y * width + x) * 4) as usize;
                if idx + 3 < data.len() {
                    data[idx] = bug_color[0];
                    data[idx + 1] = bug_color[1];
                    data[idx + 2] = bug_color[2];
                    data[idx + 3] = bug_color[3];
                }
            }
        }
    }

    data
}

fn render_watermark_impl(width: u32, height: u32, params: &HashMap<String, String>) -> Vec<u8> {
    let size = (width as usize) * (height as usize) * 4;
    let mut data = vec![0u8; size];

    let watermark_alpha = params
        .get("alpha")
        .and_then(|s| s.parse::<u8>().ok())
        .unwrap_or(40);

    let spacing = 80i32;
    let line_color = [200u8, 200, 200, watermark_alpha];

    for y in 0..height {
        for x in 0..width {
            let diag = (x as i32 + y as i32) % spacing;
            if diag == 0 || diag == 1 {
                let idx = ((y * width + x) * 4) as usize;
                if idx + 3 < data.len() {
                    data[idx] = line_color[0];
                    data[idx + 1] = line_color[1];
                    data[idx + 2] = line_color[2];
                    data[idx + 3] = line_color[3];
                }
            }
        }
    }

    data
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all graphics bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraphicsEngine>()?;
    m.add_class::<PyTemplate>()?;
    m.add_class::<PyLowerThird>()?;
    m.add_function(wrap_pyfunction!(list_templates, m)?)?;
    m.add_function(wrap_pyfunction!(render_color_bars, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hex_color_6_digit() {
        let color = parse_hex_color_py("FF8800").expect("valid");
        assert_eq!(color, [255, 136, 0, 255]);
    }

    #[test]
    fn test_parse_hex_color_8_digit() {
        let color = parse_hex_color_py("#AABBCCDD").expect("valid");
        assert_eq!(color, [170, 187, 204, 221]);
    }

    #[test]
    fn test_parse_hex_color_invalid() {
        assert!(parse_hex_color_py("XYZ").is_err());
    }

    #[test]
    fn test_graphics_engine_creation() {
        let engine = PyGraphicsEngine::new(1920, 1080);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_graphics_engine_zero_size() {
        assert!(PyGraphicsEngine::new(0, 1080).is_err());
    }

    #[test]
    fn test_lower_third_creation() {
        let lt = PyLowerThird::new("Test Title", Some("Subtitle"));
        assert_eq!(lt.title, "Test Title");
        assert_eq!(lt.subtitle, Some("Subtitle".to_string()));
    }

    #[test]
    fn test_lower_third_render() {
        let lt = PyLowerThird::new("Hello", None);
        let data = lt.render();
        assert_eq!(data.len(), 1920 * 1080 * 4);
    }

    #[test]
    fn test_template_creation() {
        let t = PyTemplate::new("color_bars");
        assert!(t.is_ok());
    }

    #[test]
    fn test_template_unknown() {
        assert!(PyTemplate::new("nonexistent").is_err());
    }

    #[test]
    fn test_list_templates_count() {
        let templates = list_templates();
        assert_eq!(templates.len(), 5);
    }

    #[test]
    fn test_render_color_bars_fn() {
        let data = render_color_bars(100, 50);
        assert!(data.is_ok());
        assert_eq!(data.expect("ok").len(), 100 * 50 * 4);
    }

    #[test]
    fn test_render_color_bars_zero() {
        assert!(render_color_bars(0, 100).is_err());
    }
}
