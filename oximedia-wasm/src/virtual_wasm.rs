//! WebAssembly bindings for virtual production utilities.
//!
//! Provides `WasmVirtualSource` and standalone functions for
//! test source generation and configuration in the browser.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// WasmVirtualSource
// ---------------------------------------------------------------------------

/// A virtual production source for browser-side management.
#[wasm_bindgen]
pub struct WasmVirtualSource {
    name: String,
    source_type: String,
    active: bool,
    fps: f64,
}

#[wasm_bindgen]
impl WasmVirtualSource {
    /// Create a new virtual source.
    ///
    /// # Arguments
    /// * `name` - Source name.
    /// * `source_type` - Type: camera, led-wall, compositor, genlock.
    /// * `fps` - Target FPS.
    ///
    /// # Errors
    /// Returns an error if parameters are invalid.
    #[wasm_bindgen(constructor)]
    pub fn new(name: &str, source_type: &str, fps: f64) -> Result<WasmVirtualSource, JsValue> {
        let valid_types = ["camera", "led-wall", "compositor", "genlock"];
        if !valid_types.contains(&source_type) {
            return Err(crate::utils::js_err(&format!(
                "Unknown source type '{}'. Supported: camera, led-wall, compositor, genlock",
                source_type
            )));
        }
        if fps <= 0.0 || fps > 240.0 {
            return Err(crate::utils::js_err("FPS must be between 0 and 240"));
        }

        Ok(Self {
            name: name.to_string(),
            source_type: source_type.to_string(),
            active: false,
            fps,
        })
    }

    /// Activate the source.
    pub fn activate(&mut self) {
        self.active = true;
    }

    /// Deactivate the source.
    pub fn deactivate(&mut self) {
        self.active = false;
    }

    /// Check if active.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Get source info as JSON.
    pub fn info(&self) -> String {
        format!(
            "{{\"name\":\"{}\",\"type\":\"{}\",\"active\":{},\"fps\":{}}}",
            self.name, self.source_type, self.active, self.fps
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Generate a test source pattern frame (solid color).
///
/// # Arguments
/// * `width` - Frame width.
/// * `height` - Frame height.
/// * `pattern` - Pattern type: color-bars, gradient, checkerboard, solid.
///
/// # Returns
/// RGB byte array of the generated frame.
#[wasm_bindgen]
pub fn wasm_generate_test_source(
    width: u32,
    height: u32,
    pattern: &str,
) -> Result<Vec<u8>, JsValue> {
    if width == 0 || height == 0 {
        return Err(crate::utils::js_err("Width and height must be > 0"));
    }
    if width > 7680 || height > 4320 {
        return Err(crate::utils::js_err("Dimensions exceed 7680x4320"));
    }

    let w = width as usize;
    let h = height as usize;
    let mut frame = vec![0u8; w * h * 3];

    match pattern {
        "color-bars" => {
            let bar_colors: [(u8, u8, u8); 7] = [
                (255, 255, 255), // White
                (255, 255, 0),   // Yellow
                (0, 255, 255),   // Cyan
                (0, 255, 0),     // Green
                (255, 0, 255),   // Magenta
                (255, 0, 0),     // Red
                (0, 0, 255),     // Blue
            ];
            let bar_width = w / 7;
            for y in 0..h {
                for x in 0..w {
                    let bar_idx = (x / bar_width.max(1)).min(6);
                    let (r, g, b) = bar_colors[bar_idx];
                    let idx = (y * w + x) * 3;
                    frame[idx] = r;
                    frame[idx + 1] = g;
                    frame[idx + 2] = b;
                }
            }
        }
        "gradient" => {
            for y in 0..h {
                for x in 0..w {
                    let idx = (y * w + x) * 3;
                    let luma = ((x as f64 / w as f64) * 255.0) as u8;
                    frame[idx] = luma;
                    frame[idx + 1] = luma;
                    frame[idx + 2] = luma;
                }
            }
        }
        "checkerboard" => {
            let block = 32;
            for y in 0..h {
                for x in 0..w {
                    let idx = (y * w + x) * 3;
                    let white = ((x / block) + (y / block)) % 2 == 0;
                    let val = if white { 255 } else { 0 };
                    frame[idx] = val;
                    frame[idx + 1] = val;
                    frame[idx + 2] = val;
                }
            }
        }
        _ => {
            // Solid gray
            for pixel in frame.chunks_exact_mut(3) {
                pixel[0] = 128;
                pixel[1] = 128;
                pixel[2] = 128;
            }
        }
    }

    Ok(frame)
}

/// Get a list of supported virtual source types as JSON.
#[wasm_bindgen]
pub fn wasm_virtual_source_types() -> String {
    "[\"camera\",\"led-wall\",\"compositor\",\"genlock\"]".to_string()
}

/// Get supported workflows as JSON.
#[wasm_bindgen]
pub fn wasm_virtual_workflows() -> String {
    "[\"led-wall\",\"hybrid\",\"green-screen\",\"ar\"]".to_string()
}

/// Get recommended virtual production settings for a given workflow.
#[wasm_bindgen]
pub fn wasm_virtual_settings(workflow: &str, fps: f64) -> Result<String, JsValue> {
    if fps <= 0.0 || fps > 240.0 {
        return Err(crate::utils::js_err("FPS must be between 0 and 240"));
    }

    let quality = if fps >= 120.0 {
        "draft"
    } else if fps >= 60.0 {
        "preview"
    } else {
        "final"
    };

    let sync_ms = if fps >= 120.0 { 0.25 } else { 0.5 };

    Ok(format!(
        "{{\"workflow\":\"{workflow}\",\"fps\":{fps},\"quality\":\"{quality}\",\"sync_accuracy_ms\":{sync_ms}}}"
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_creation() {
        let src = WasmVirtualSource::new("cam1", "camera", 60.0);
        assert!(src.is_ok());
    }

    #[test]
    fn test_source_invalid_type() {
        let src = WasmVirtualSource::new("x", "invalid", 60.0);
        assert!(src.is_err());
    }

    #[test]
    fn test_test_source_color_bars() {
        let frame = wasm_generate_test_source(64, 48, "color-bars");
        assert!(frame.is_ok());
        let data = frame.expect("should succeed");
        assert_eq!(data.len(), 64 * 48 * 3);
    }

    #[test]
    fn test_source_types_json() {
        let json = wasm_virtual_source_types();
        assert!(json.contains("camera"));
        assert!(json.contains("genlock"));
    }

    #[test]
    fn test_virtual_settings() {
        let result = wasm_virtual_settings("led-wall", 60.0);
        assert!(result.is_ok());
        let json = result.expect("should succeed");
        assert!(json.contains("led-wall"));
    }
}
