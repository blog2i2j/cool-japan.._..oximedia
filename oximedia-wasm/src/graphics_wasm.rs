//! WebAssembly bindings for `oximedia-graphics` broadcast graphics engine.
//!
//! Provides `WasmGraphicsRenderer` for rendering lower-thirds, tickers,
//! overlays, and templates in the browser.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// WasmGraphicsRenderer
// ---------------------------------------------------------------------------

/// Browser-side broadcast graphics renderer.
#[wasm_bindgen]
pub struct WasmGraphicsRenderer {
    width: u32,
    height: u32,
}

#[wasm_bindgen]
impl WasmGraphicsRenderer {
    /// Create a new graphics renderer with the given resolution.
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32) -> Result<WasmGraphicsRenderer, JsValue> {
        if width == 0 || height == 0 {
            return Err(crate::utils::js_err("Width and height must be > 0"));
        }
        Ok(Self { width, height })
    }

    /// Render a lower-third graphic.
    ///
    /// `config_json` is an optional JSON string with keys:
    /// - `subtitle`: subtitle text
    /// - `bg_color`: hex color (e.g., "000000C8")
    /// - `text_color`: hex color
    /// - `style`: "classic", "modern", "minimal", "news", "sports", "corporate"
    /// - `duration`: duration in seconds
    ///
    /// Returns RGBA pixel data.
    pub fn render_lower_third(
        &self,
        title: &str,
        subtitle: &str,
        config_json: &str,
    ) -> Result<Vec<u8>, JsValue> {
        let mut lt_config = oximedia_graphics::lower_third::LowerThirdConfig {
            name: title.to_string(),
            title: subtitle.to_string(),
            subtitle: if subtitle.is_empty() {
                None
            } else {
                Some(subtitle.to_string())
            },
            ..oximedia_graphics::lower_third::LowerThirdConfig::default()
        };

        // Parse optional config
        if !config_json.is_empty() {
            if let Some(bg) = extract_json_string(config_json, "bg_color") {
                lt_config.background_color = parse_hex_color_wasm(&bg)?;
            }
            if let Some(tc) = extract_json_string(config_json, "text_color") {
                lt_config.text_color = parse_hex_color_wasm(&tc)?;
            }
        }

        let duration = extract_json_f64(config_json, "duration").unwrap_or(3.0);
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
    /// `config_json` is an optional JSON string with keys:
    /// - `speed`: scroll speed in pixels per second
    /// - `bg_color`: hex color
    /// - `text_color`: hex color
    /// - `height`: ticker height in pixels
    ///
    /// Returns RGBA pixel data.
    pub fn render_ticker(&self, text: &str, config_json: &str) -> Result<Vec<u8>, JsValue> {
        let mut ticker_config = oximedia_graphics::ticker::TickerConfig::default();

        if !config_json.is_empty() {
            if let Some(speed) = extract_json_f64(config_json, "speed") {
                ticker_config.scroll_speed_pps = speed as f32;
            }
            if let Some(bg) = extract_json_string(config_json, "bg_color") {
                ticker_config.bg_color = parse_hex_color_wasm(&bg)?;
            }
            if let Some(tc) = extract_json_string(config_json, "text_color") {
                ticker_config.text_color = parse_hex_color_wasm(&tc)?;
            }
            if let Some(h) = extract_json_f64(config_json, "height") {
                ticker_config.height_px = h as u32;
            }
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
    /// Both `base` and `overlay` must be RGBA data.
    pub fn render_overlay(
        &self,
        base: &[u8],
        overlay: &[u8],
        x: i32,
        y: i32,
        overlay_w: u32,
        overlay_h: u32,
        opacity: f64,
    ) -> Result<Vec<u8>, JsValue> {
        let expected_base = (self.width as usize) * (self.height as usize) * 4;
        if base.len() < expected_base {
            return Err(crate::utils::js_err(&format!(
                "Base data too small: need {} bytes, got {}",
                expected_base,
                base.len()
            )));
        }

        let expected_overlay = (overlay_w as usize) * (overlay_h as usize) * 4;
        if overlay.len() < expected_overlay {
            return Err(crate::utils::js_err(&format!(
                "Overlay data too small: need {} bytes, got {}",
                expected_overlay,
                overlay.len()
            )));
        }

        let mut result = base.to_vec();
        let alpha = opacity.clamp(0.0, 1.0) as f32;

        for oy in 0..overlay_h {
            for ox in 0..overlay_w {
                let dst_x = ox as i32 + x;
                let dst_y = oy as i32 + y;

                if dst_x < 0
                    || dst_x >= self.width as i32
                    || dst_y < 0
                    || dst_y >= self.height as i32
                {
                    continue;
                }

                let src_idx = ((oy * overlay_w + ox) * 4) as usize;
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

    /// Get the configured width.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get the configured height.
    pub fn height(&self) -> u32 {
        self.height
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// List available graphics templates as a JSON array.
#[wasm_bindgen]
pub fn wasm_list_templates() -> String {
    "[{\"name\":\"lower_third\",\"description\":\"Broadcast lower-third with title and subtitle\"},\
     {\"name\":\"full_screen_title\",\"description\":\"Full-screen title card\"},\
     {\"name\":\"bug\",\"description\":\"Channel bug (corner logo placeholder)\"},\
     {\"name\":\"watermark\",\"description\":\"Semi-transparent watermark pattern\"},\
     {\"name\":\"color_bars\",\"description\":\"SMPTE-style color bars test pattern\"}]"
        .to_string()
}

/// Render a named template.
///
/// Returns RGBA pixel data.
#[wasm_bindgen]
pub fn wasm_render_template(
    name: &str,
    params_json: &str,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, JsValue> {
    if width == 0 || height == 0 {
        return Err(crate::utils::js_err("Width and height must be > 0"));
    }

    match name {
        "lower_third" => {
            let title = extract_json_string(params_json, "title")
                .unwrap_or_else(|| "Title".to_string());
            let subtitle = extract_json_string(params_json, "subtitle").unwrap_or_default();

            let config = oximedia_graphics::lower_third::LowerThirdConfig {
                name: title,
                title: subtitle.clone(),
                subtitle: if subtitle.is_empty() {
                    None
                } else {
                    Some(subtitle)
                },
                ..oximedia_graphics::lower_third::LowerThirdConfig::default()
            };
            Ok(oximedia_graphics::lower_third::LowerThirdRenderer::render(
                &config, 45, 90, width, height,
            ))
        }
        "color_bars" => Ok(render_color_bars_impl(width, height)),
        "full_screen_title" => Ok(render_full_screen_title_impl(width, height)),
        "bug" => Ok(render_bug_impl(width, height)),
        "watermark" => Ok(render_watermark_impl(width, height)),
        _ => Err(crate::utils::js_err(&format!(
            "Unknown template '{}'. Available: lower_third, full_screen_title, bug, watermark, color_bars",
            name
        ))),
    }
}

/// Render SMPTE-style color bars test pattern.
///
/// Returns RGBA pixel data.
#[wasm_bindgen]
pub fn wasm_render_color_bars(width: u32, height: u32) -> Result<Vec<u8>, JsValue> {
    if width == 0 || height == 0 {
        return Err(crate::utils::js_err("Width and height must be > 0"));
    }
    Ok(render_color_bars_impl(width, height))
}

// ---------------------------------------------------------------------------
// Internal helpers
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

fn render_full_screen_title_impl(width: u32, height: u32) -> Vec<u8> {
    let size = (width as usize) * (height as usize) * 4;
    let mut data = vec![0u8; size];

    // Dark background
    for chunk in data.chunks_exact_mut(4) {
        chunk[0] = 20;
        chunk[1] = 20;
        chunk[2] = 30;
        chunk[3] = 255;
    }

    // Centered accent rule
    let rule_y = height / 2;
    let margin = width / 8;
    for dy in 0..4u32 {
        let y = rule_y + dy;
        if y >= height {
            break;
        }
        for x in margin..(width - margin) {
            let idx = ((y * width + x) * 4) as usize;
            if idx + 3 < data.len() {
                data[idx] = 255;
                data[idx + 1] = 165;
                data[idx + 2] = 0;
                data[idx + 3] = 255;
            }
        }
    }

    data
}

fn render_bug_impl(width: u32, height: u32) -> Vec<u8> {
    let size = (width as usize) * (height as usize) * 4;
    let mut data = vec![0u8; size];

    let bug_size = 80u32;
    let margin = 40u32;
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
                    data[idx] = 255;
                    data[idx + 1] = 255;
                    data[idx + 2] = 255;
                    data[idx + 3] = 180;
                }
            }
        }
    }

    data
}

fn render_watermark_impl(width: u32, height: u32) -> Vec<u8> {
    let size = (width as usize) * (height as usize) * 4;
    let mut data = vec![0u8; size];

    let spacing = 80i32;
    for y in 0..height {
        for x in 0..width {
            let diag = (x as i32 + y as i32) % spacing;
            if diag == 0 || diag == 1 {
                let idx = ((y * width + x) * 4) as usize;
                if idx + 3 < data.len() {
                    data[idx] = 200;
                    data[idx + 1] = 200;
                    data[idx + 2] = 200;
                    data[idx + 3] = 40;
                }
            }
        }
    }

    data
}

fn parse_hex_color_wasm(hex: &str) -> Result<[u8; 4], JsValue> {
    let hex = hex.trim_start_matches('#');
    match hex.len() {
        6 => {
            let r = u8::from_str_radix(&hex[0..2], 16)
                .map_err(|_| crate::utils::js_err(&format!("Invalid hex color: {hex}")))?;
            let g = u8::from_str_radix(&hex[2..4], 16)
                .map_err(|_| crate::utils::js_err(&format!("Invalid hex color: {hex}")))?;
            let b = u8::from_str_radix(&hex[4..6], 16)
                .map_err(|_| crate::utils::js_err(&format!("Invalid hex color: {hex}")))?;
            Ok([r, g, b, 255])
        }
        8 => {
            let r = u8::from_str_radix(&hex[0..2], 16)
                .map_err(|_| crate::utils::js_err(&format!("Invalid hex color: {hex}")))?;
            let g = u8::from_str_radix(&hex[2..4], 16)
                .map_err(|_| crate::utils::js_err(&format!("Invalid hex color: {hex}")))?;
            let b = u8::from_str_radix(&hex[4..6], 16)
                .map_err(|_| crate::utils::js_err(&format!("Invalid hex color: {hex}")))?;
            let a = u8::from_str_radix(&hex[6..8], 16)
                .map_err(|_| crate::utils::js_err(&format!("Invalid hex color: {hex}")))?;
            Ok([r, g, b, a])
        }
        _ => Err(crate::utils::js_err(&format!(
            "Invalid hex color '{}': expected 6 or 8 hex characters",
            hex
        ))),
    }
}

/// Extract a string value from JSON by key.
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\":\"", key);
    if let Some(start) = json.find(&pattern) {
        let rest = &json[start + pattern.len()..];
        if let Some(end) = rest.find('"') {
            return Some(rest[..end].to_string());
        }
    }
    None
}

/// Extract a float value from JSON by key.
fn extract_json_f64(json: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{}\":", key);
    if let Some(start) = json.find(&pattern) {
        let rest = &json[start + pattern.len()..];
        let end = rest.find([',', '}', ' ']).unwrap_or(rest.len());
        return rest[..end].trim().parse::<f64>().ok();
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_renderer_creation() {
        let r = WasmGraphicsRenderer::new(1920, 1080);
        assert!(r.is_ok());
    }

    #[test]
    fn test_renderer_zero_size() {
        assert!(WasmGraphicsRenderer::new(0, 100).is_err());
    }

    #[test]
    fn test_render_lower_third() {
        let r = WasmGraphicsRenderer::new(100, 50).expect("ok");
        let result = r.render_lower_third("Test", "", "{}");
        assert!(result.is_ok());
        assert_eq!(result.expect("ok").len(), 100 * 50 * 4);
    }

    #[test]
    fn test_render_ticker() {
        let r = WasmGraphicsRenderer::new(100, 50).expect("ok");
        let result = r.render_ticker("Breaking News", "{}");
        assert!(result.is_ok());
    }

    #[test]
    fn test_wasm_list_templates() {
        let json = wasm_list_templates();
        assert!(json.contains("lower_third"));
        assert!(json.contains("color_bars"));
    }

    #[test]
    fn test_wasm_render_color_bars() {
        let result = wasm_render_color_bars(100, 50);
        assert!(result.is_ok());
        assert_eq!(result.expect("ok").len(), 100 * 50 * 4);
    }

    #[test]
    fn test_wasm_render_color_bars_zero() {
        assert!(wasm_render_color_bars(0, 50).is_err());
    }

    #[test]
    fn test_wasm_render_template() {
        let result = wasm_render_template("color_bars", "{}", 100, 50);
        assert!(result.is_ok());
    }

    #[test]
    fn test_wasm_render_template_unknown() {
        assert!(wasm_render_template("nonexistent", "{}", 100, 50).is_err());
    }

    #[test]
    fn test_parse_hex_color() {
        let c = parse_hex_color_wasm("FF8800");
        assert!(c.is_ok());
        assert_eq!(c.expect("ok"), [255, 136, 0, 255]);
    }

    #[test]
    fn test_extract_json_string() {
        let json = r#"{"title":"Hello","subtitle":"World"}"#;
        assert_eq!(
            extract_json_string(json, "title"),
            Some("Hello".to_string())
        );
        assert_eq!(extract_json_string(json, "missing"), None);
    }

    #[test]
    fn test_extract_json_f64() {
        let json = r#"{"duration":3.5,"speed":120}"#;
        let dur = extract_json_f64(json, "duration");
        assert!(dur.is_some());
        assert!((dur.expect("ok") - 3.5).abs() < 0.01);
    }
}
