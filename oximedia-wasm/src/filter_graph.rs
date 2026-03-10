//! WASM filter graph for in-browser media processing.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Filter operation specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterSpec {
    /// Filter name.
    pub name: String,
    /// Filter parameters.
    pub params: HashMap<String, String>,
}

/// WASM filter graph for in-browser media processing.
///
/// Builds and executes simple filter chains on video or audio data.
#[wasm_bindgen]
pub struct WasmFilterGraph {
    filters: Vec<FilterSpec>,
    width: u32,
    height: u32,
}

#[wasm_bindgen]
impl WasmFilterGraph {
    /// Create a new filter graph.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
            width: 0,
            height: 0,
        }
    }

    /// Set video dimensions for the graph.
    pub fn set_dimensions(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    /// Add a scale filter.
    ///
    /// A width or height of 0 indicates maintain aspect ratio.
    pub fn add_scale(&mut self, width: u32, height: u32) {
        let mut params = HashMap::new();
        params.insert("width".to_string(), width.to_string());
        params.insert("height".to_string(), height.to_string());
        self.filters.push(FilterSpec {
            name: "scale".to_string(),
            params,
        });
    }

    /// Add a crop filter.
    pub fn add_crop(&mut self, x: u32, y: u32, width: u32, height: u32) {
        let mut params = HashMap::new();
        params.insert("x".to_string(), x.to_string());
        params.insert("y".to_string(), y.to_string());
        params.insert("width".to_string(), width.to_string());
        params.insert("height".to_string(), height.to_string());
        self.filters.push(FilterSpec {
            name: "crop".to_string(),
            params,
        });
    }

    /// Add a volume filter for audio.
    ///
    /// `gain_db` is the gain in decibels (positive = louder, negative = quieter).
    pub fn add_volume(&mut self, gain_db: f32) {
        let mut params = HashMap::new();
        params.insert("gain_db".to_string(), gain_db.to_string());
        self.filters.push(FilterSpec {
            name: "volume".to_string(),
            params,
        });
    }

    /// Add a colorspace conversion filter.
    ///
    /// Supported conversions: "rgb" -> "gray"
    pub fn add_colorspace(&mut self, from: &str, to: &str) {
        let mut params = HashMap::new();
        params.insert("from".to_string(), from.to_string());
        params.insert("to".to_string(), to.to_string());
        self.filters.push(FilterSpec {
            name: "colorspace".to_string(),
            params,
        });
    }

    /// Add a custom filter by name and JSON params string.
    ///
    /// `params_json` must be a valid JSON object string, e.g. `{"key":"value"}`.
    ///
    /// # Errors
    ///
    /// Returns an error if `params_json` is not valid JSON.
    pub fn add_filter(&mut self, name: &str, params_json: &str) -> Result<(), JsValue> {
        let params: HashMap<String, String> = serde_json::from_str(params_json)
            .map_err(|e| crate::utils::js_err(&format!("Invalid params JSON: {e}")))?;
        self.filters.push(FilterSpec {
            name: name.to_string(),
            params,
        });
        Ok(())
    }

    /// Process a video frame (RGB24 `Uint8Array`).
    ///
    /// Returns the processed frame as a `Uint8Array`.
    ///
    /// # Errors
    ///
    /// Returns an error if processing fails (e.g., dimension mismatch).
    pub fn process_video(
        &self,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<js_sys::Uint8Array, JsValue> {
        let expected = (width * height * 3) as usize;
        if data.len() < expected {
            return Err(crate::utils::js_err(&format!(
                "Input data too small: expected {expected} bytes for {}x{} RGB24, got {}",
                width,
                height,
                data.len()
            )));
        }

        let mut current_data = data[..expected].to_vec();
        let mut current_width = width;
        let mut current_height = height;

        for filter in &self.filters {
            match filter.name.as_str() {
                "scale" => {
                    let out_w = filter
                        .params
                        .get("width")
                        .and_then(|v| v.parse::<u32>().ok())
                        .unwrap_or(0);
                    let out_h = filter
                        .params
                        .get("height")
                        .and_then(|v| v.parse::<u32>().ok())
                        .unwrap_or(0);
                    let result = apply_scale_rgb24(
                        &current_data,
                        current_width,
                        current_height,
                        out_w,
                        out_h,
                    )?;
                    current_width = result.1;
                    current_height = result.2;
                    current_data = result.0;
                }
                "crop" => {
                    let x = filter
                        .params
                        .get("x")
                        .and_then(|v| v.parse::<u32>().ok())
                        .unwrap_or(0);
                    let y = filter
                        .params
                        .get("y")
                        .and_then(|v| v.parse::<u32>().ok())
                        .unwrap_or(0);
                    let cw = filter
                        .params
                        .get("width")
                        .and_then(|v| v.parse::<u32>().ok())
                        .unwrap_or(current_width);
                    let ch = filter
                        .params
                        .get("height")
                        .and_then(|v| v.parse::<u32>().ok())
                        .unwrap_or(current_height);
                    let result = apply_crop_rgb24(
                        &current_data,
                        current_width,
                        current_height,
                        x,
                        y,
                        cw,
                        ch,
                    )?;
                    current_width = result.1;
                    current_height = result.2;
                    current_data = result.0;
                }
                "colorspace" => {
                    let from = filter
                        .params
                        .get("from")
                        .map(String::as_str)
                        .unwrap_or("rgb");
                    let to = filter.params.get("to").map(String::as_str).unwrap_or("rgb");
                    let result = apply_colorspace_rgb24(
                        &current_data,
                        current_width,
                        current_height,
                        from,
                        to,
                    )?;
                    current_width = result.1;
                    current_height = result.2;
                    current_data = result.0;
                }
                // Unknown video filters are silently passed through.
                _ => {}
            }
        }

        let out = js_sys::Uint8Array::new_with_length(current_data.len() as u32);
        out.copy_from(&current_data);
        Ok(out)
    }

    /// Process audio samples (interleaved f32).
    ///
    /// Returns the processed samples as a `Float32Array`.
    ///
    /// # Errors
    ///
    /// Returns an error if processing fails.
    pub fn process_audio(&self, samples: &[f32]) -> Result<js_sys::Float32Array, JsValue> {
        let mut current = samples.to_vec();

        for filter in &self.filters {
            if filter.name == "volume" {
                let gain_db = filter
                    .params
                    .get("gain_db")
                    .and_then(|v| v.parse::<f32>().ok())
                    .unwrap_or(0.0);
                let linear_gain = 10.0_f32.powf(gain_db / 20.0);
                for s in &mut current {
                    *s *= linear_gain;
                }
            }
            // Unknown audio filters are silently passed through.
        }

        let out = js_sys::Float32Array::new_with_length(current.len() as u32);
        out.copy_from(&current);
        Ok(out)
    }

    /// Get number of filters in the chain.
    pub fn filter_count(&self) -> u32 {
        self.filters.len() as u32
    }

    /// Clear all filters.
    pub fn clear(&mut self) {
        self.filters.clear();
    }

    /// Get filter chain as a JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error if serialisation fails.
    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.filters)
            .map_err(|e| crate::utils::js_err(&format!("Serialisation error: {e}")))
    }
}

// ─── Internal pixel-manipulation helpers ──────────────────────────────────────

/// Bilinear-interpolation scale for RGB24 data.
///
/// If either output dimension is 0 it is computed to preserve aspect ratio.
/// Returns `(data, out_width, out_height)`.
fn apply_scale_rgb24(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    mut dst_w: u32,
    mut dst_h: u32,
) -> Result<(Vec<u8>, u32, u32), JsValue> {
    // Resolve aspect-ratio cases.
    match (dst_w, dst_h) {
        (0, 0) => {
            dst_w = src_w;
            dst_h = src_h;
        }
        (0, h) => {
            dst_w = (src_w * h + src_h / 2) / src_h.max(1);
            dst_h = h;
        }
        (w, 0) => {
            dst_w = w;
            dst_h = (src_h * w + src_w / 2) / src_w.max(1);
        }
        _ => {}
    }

    if dst_w == 0 || dst_h == 0 {
        return Err(crate::utils::js_err("Scale: output dimensions must be > 0"));
    }

    let out_size = (dst_w * dst_h * 3) as usize;
    let mut out = vec![0u8; out_size];

    let scale_x = src_w as f32 / dst_w as f32;
    let scale_y = src_h as f32 / dst_h as f32;

    for dy in 0..dst_h {
        let src_y_f = (dy as f32 + 0.5) * scale_y - 0.5;
        let y0 = src_y_f.floor() as i32;
        let y1 = y0 + 1;
        let fy = src_y_f - y0 as f32;

        let y0c = y0.clamp(0, src_h as i32 - 1) as u32;
        let y1c = y1.clamp(0, src_h as i32 - 1) as u32;

        for dx in 0..dst_w {
            let src_x_f = (dx as f32 + 0.5) * scale_x - 0.5;
            let x0 = src_x_f.floor() as i32;
            let x1 = x0 + 1;
            let fx = src_x_f - x0 as f32;

            let x0c = x0.clamp(0, src_w as i32 - 1) as u32;
            let x1c = x1.clamp(0, src_w as i32 - 1) as u32;

            for c in 0..3usize {
                let p00 = src[(y0c * src_w * 3 + x0c * 3 + c as u32) as usize] as f32;
                let p10 = src[(y0c * src_w * 3 + x1c * 3 + c as u32) as usize] as f32;
                let p01 = src[(y1c * src_w * 3 + x0c * 3 + c as u32) as usize] as f32;
                let p11 = src[(y1c * src_w * 3 + x1c * 3 + c as u32) as usize] as f32;

                let value = p00 * (1.0 - fx) * (1.0 - fy)
                    + p10 * fx * (1.0 - fy)
                    + p01 * (1.0 - fx) * fy
                    + p11 * fx * fy;

                out[(dy * dst_w * 3 + dx * 3 + c as u32) as usize] = value.round() as u8;
            }
        }
    }

    Ok((out, dst_w, dst_h))
}

/// Crop RGB24 frame.  Returns `(data, out_width, out_height)`.
fn apply_crop_rgb24(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    x: u32,
    y: u32,
    crop_w: u32,
    crop_h: u32,
) -> Result<(Vec<u8>, u32, u32), JsValue> {
    if x + crop_w > src_w || y + crop_h > src_h {
        return Err(crate::utils::js_err(&format!(
            "Crop region ({x},{y}) {crop_w}x{crop_h} exceeds frame {src_w}x{src_h}"
        )));
    }
    let mut out = Vec::with_capacity((crop_w * crop_h * 3) as usize);
    for row in y..(y + crop_h) {
        let row_start = (row * src_w * 3 + x * 3) as usize;
        let row_end = row_start + (crop_w * 3) as usize;
        out.extend_from_slice(&src[row_start..row_end]);
    }
    Ok((out, crop_w, crop_h))
}

/// Colorspace conversion for RGB24 data.  Returns `(data, out_width, out_height)`.
///
/// Supported: `rgb` → `gray` (BT.601 luma, output is gray stored as RGB24 triples).
fn apply_colorspace_rgb24(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    from: &str,
    to: &str,
) -> Result<(Vec<u8>, u32, u32), JsValue> {
    let canonical = format!("{from}-to-{to}").to_lowercase();
    match canonical.as_str() {
        "rgb-to-gray" | "rgb24-to-gray" => {
            let pixel_count = (src_w * src_h) as usize;
            let mut out = Vec::with_capacity(pixel_count * 3);
            for i in 0..pixel_count {
                let r = src[i * 3] as f32;
                let g = src[i * 3 + 1] as f32;
                let b = src[i * 3 + 2] as f32;
                let luma = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
                out.push(luma);
                out.push(luma);
                out.push(luma);
            }
            Ok((out, src_w, src_h))
        }
        "rgb-to-rgb" | "rgb24-to-rgb24" => Ok((src.to_vec(), src_w, src_h)),
        other => Err(crate::utils::js_err(&format!(
            "Unsupported colorspace conversion: {other}"
        ))),
    }
}

// ─── Pure-Rust pixel-manipulation helpers for testing (no JsValue) ───────────
//
// These duplicate the core logic of the `apply_*` functions but return
// `Result<_, String>` so they are callable from native `#[test]` blocks
// without triggering the `wasm_bindgen` `JsValue::from_str` non-wasm panic.

/// Scale RGB24 — pure Rust, no `JsValue`.
#[cfg(test)]
fn scale_rgb24_native(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    mut dst_w: u32,
    mut dst_h: u32,
) -> Result<(Vec<u8>, u32, u32), String> {
    match (dst_w, dst_h) {
        (0, 0) => {
            dst_w = src_w;
            dst_h = src_h;
        }
        (0, h) => {
            dst_w = (src_w * h + src_h / 2) / src_h.max(1);
            dst_h = h;
        }
        (w, 0) => {
            dst_w = w;
            dst_h = (src_h * w + src_w / 2) / src_w.max(1);
        }
        _ => {}
    }
    if dst_w == 0 || dst_h == 0 {
        return Err("Scale: output dimensions must be > 0".to_string());
    }
    let out_size = (dst_w * dst_h * 3) as usize;
    let mut out = vec![0u8; out_size];
    let scale_x = src_w as f32 / dst_w as f32;
    let scale_y = src_h as f32 / dst_h as f32;
    for dy in 0..dst_h {
        let src_y_f = (dy as f32 + 0.5) * scale_y - 0.5;
        let y0 = src_y_f.floor() as i32;
        let y1 = y0 + 1;
        let fy = src_y_f - y0 as f32;
        let y0c = y0.clamp(0, src_h as i32 - 1) as u32;
        let y1c = y1.clamp(0, src_h as i32 - 1) as u32;
        for dx in 0..dst_w {
            let src_x_f = (dx as f32 + 0.5) * scale_x - 0.5;
            let x0 = src_x_f.floor() as i32;
            let x1 = x0 + 1;
            let fx = src_x_f - x0 as f32;
            let x0c = x0.clamp(0, src_w as i32 - 1) as u32;
            let x1c = x1.clamp(0, src_w as i32 - 1) as u32;
            for c in 0..3usize {
                let p00 = src[(y0c * src_w * 3 + x0c * 3 + c as u32) as usize] as f32;
                let p10 = src[(y0c * src_w * 3 + x1c * 3 + c as u32) as usize] as f32;
                let p01 = src[(y1c * src_w * 3 + x0c * 3 + c as u32) as usize] as f32;
                let p11 = src[(y1c * src_w * 3 + x1c * 3 + c as u32) as usize] as f32;
                let value = p00 * (1.0 - fx) * (1.0 - fy)
                    + p10 * fx * (1.0 - fy)
                    + p01 * (1.0 - fx) * fy
                    + p11 * fx * fy;
                out[(dy * dst_w * 3 + dx * 3 + c as u32) as usize] = value.round() as u8;
            }
        }
    }
    Ok((out, dst_w, dst_h))
}

/// Crop RGB24 — pure Rust, no `JsValue`.
#[cfg(test)]
fn crop_rgb24_native(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    x: u32,
    y: u32,
    crop_w: u32,
    crop_h: u32,
) -> Result<(Vec<u8>, u32, u32), String> {
    if x + crop_w > src_w || y + crop_h > src_h {
        return Err(format!(
            "Crop region ({x},{y}) {crop_w}x{crop_h} exceeds frame {src_w}x{src_h}"
        ));
    }
    let mut out = Vec::with_capacity((crop_w * crop_h * 3) as usize);
    for row in y..(y + crop_h) {
        let row_start = (row * src_w * 3 + x * 3) as usize;
        let row_end = row_start + (crop_w * 3) as usize;
        out.extend_from_slice(&src[row_start..row_end]);
    }
    Ok((out, crop_w, crop_h))
}

/// Colorspace conversion — pure Rust, no `JsValue`.
#[cfg(test)]
fn colorspace_native(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    from: &str,
    to: &str,
) -> Result<(Vec<u8>, u32, u32), String> {
    let canonical = format!("{from}-to-{to}").to_lowercase();
    match canonical.as_str() {
        "rgb-to-gray" | "rgb24-to-gray" => {
            let pixel_count = (src_w * src_h) as usize;
            let mut out = Vec::with_capacity(pixel_count * 3);
            for i in 0..pixel_count {
                let r = src[i * 3] as f32;
                let g = src[i * 3 + 1] as f32;
                let b = src[i * 3 + 2] as f32;
                let luma = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
                out.push(luma);
                out.push(luma);
                out.push(luma);
            }
            Ok((out, src_w, src_h))
        }
        "rgb-to-rgb" | "rgb24-to-rgb24" => Ok((src.to_vec(), src_w, src_h)),
        other => Err(format!("Unsupported colorspace conversion: {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── apply_scale_rgb24 (via scale_rgb24_native) ───────────────────────

    #[test]
    fn test_scale_identity() {
        // A 2×2 red frame scaled to 2×2 must be identical.
        let src: Vec<u8> = vec![255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0];
        let (out, w, h) =
            scale_rgb24_native(&src, 2, 2, 2, 2).expect("identity scale should succeed");
        assert_eq!(w, 2);
        assert_eq!(h, 2);
        assert_eq!(out.len(), 12);
        for chunk in out.chunks(3) {
            assert_eq!(chunk[0], 255, "R must be 255");
            assert_eq!(chunk[1], 0, "G must be 0");
            assert_eq!(chunk[2], 0, "B must be 0");
        }
    }

    #[test]
    fn test_scale_2x2_to_1x1_averages_white() {
        let src = vec![255u8; 4 * 3]; // 2×2 white
        let (out, w, h) =
            scale_rgb24_native(&src, 2, 2, 1, 1).expect("scale 2x2 to 1x1 should succeed");
        assert_eq!((w, h), (1, 1));
        assert_eq!(out.len(), 3);
        for &v in &out {
            assert!(v > 250, "expected white pixel, got {v}");
        }
    }

    #[test]
    fn test_scale_preserves_output_byte_count() {
        let src = vec![128u8; 4 * 4 * 3]; // 4×4 grey
        let (out, w, h) =
            scale_rgb24_native(&src, 4, 4, 8, 8).expect("scale 4x4 to 8x8 should succeed");
        assert_eq!(w, 8);
        assert_eq!(h, 8);
        assert_eq!(out.len(), 8 * 8 * 3);
    }

    #[test]
    fn test_scale_zero_dimensions_passes_through_src() {
        let src = vec![1u8; 3 * 3 * 3];
        let (_, w, h) =
            scale_rgb24_native(&src, 3, 3, 0, 0).expect("zero-dim passthrough should succeed");
        assert_eq!((w, h), (3, 3));
    }

    #[test]
    fn test_scale_aspect_ratio_height_zero() {
        let src = vec![0u8; 4 * 2 * 3]; // 4×2
        let (_, w, h) =
            scale_rgb24_native(&src, 4, 2, 8, 0).expect("aspect ratio scale should succeed");
        assert_eq!(w, 8);
        assert_eq!(h, 4);
    }

    // ─── apply_crop_rgb24 ─────────────────────────────────────────────────

    #[test]
    fn test_crop_full_frame() {
        let src: Vec<u8> = (0u8..9).flat_map(|i| [i, i, i]).collect();
        let (out, w, h) =
            crop_rgb24_native(&src, 3, 3, 0, 0, 3, 3).expect("full frame crop should succeed");
        assert_eq!((w, h), (3, 3));
        assert_eq!(out, src);
    }

    #[test]
    fn test_crop_single_pixel() {
        let mut src = vec![0u8; 3 * 3 * 3];
        let idx = (1 * 3 + 1) * 3;
        src[idx] = 99;
        src[idx + 1] = 88;
        src[idx + 2] = 77;
        let (out, w, h) =
            crop_rgb24_native(&src, 3, 3, 1, 1, 1, 1).expect("single pixel crop should succeed");
        assert_eq!((w, h), (1, 1));
        assert_eq!(out, vec![99, 88, 77]);
    }

    #[test]
    fn test_crop_out_of_bounds_errors() {
        let src = vec![0u8; 4 * 4 * 3];
        // x=3, width=2 → x+w = 5 > 4 → error.
        let result = crop_rgb24_native(&src, 4, 4, 3, 0, 2, 2);
        assert!(result.is_err(), "out-of-bounds crop must fail");
    }

    // ─── apply_colorspace_rgb24 ───────────────────────────────────────────

    #[test]
    fn test_colorspace_rgb_to_gray_white() {
        let src = vec![255u8, 255, 255];
        let (out, w, h) =
            colorspace_native(&src, 1, 1, "rgb", "gray").expect("rgb to gray white should succeed");
        assert_eq!((w, h), (1, 1));
        assert_eq!(out.len(), 3);
        for &v in &out {
            assert!(v >= 254, "expected luma ~255, got {v}");
        }
    }

    #[test]
    fn test_colorspace_rgb_to_gray_black() {
        let src = vec![0u8, 0, 0];
        let (out, _, _) =
            colorspace_native(&src, 1, 1, "rgb", "gray").expect("rgb to gray black should succeed");
        for &v in &out {
            assert_eq!(v, 0, "expected luma 0");
        }
    }

    #[test]
    fn test_colorspace_rgb_to_rgb_passthrough() {
        let src: Vec<u8> = (0u8..12).collect();
        let (out, w, h) = colorspace_native(&src, 2, 2, "rgb", "rgb")
            .expect("rgb to rgb passthrough should succeed");
        assert_eq!((w, h), (2, 2));
        assert_eq!(out, src);
    }

    #[test]
    fn test_colorspace_unknown_conversion_errors() {
        let src = vec![0u8; 3];
        let result = colorspace_native(&src, 1, 1, "rgb", "hsv");
        assert!(result.is_err(), "unknown conversion must fail");
    }

    // ─── WasmFilterGraph builder / state ──────────────────────────────────

    #[test]
    fn test_filter_graph_count_and_clear() {
        let mut fg = WasmFilterGraph::new();
        assert_eq!(fg.filter_count(), 0);
        fg.add_scale(640, 480);
        fg.add_volume(6.0);
        assert_eq!(fg.filter_count(), 2);
        fg.clear();
        assert_eq!(fg.filter_count(), 0);
    }

    #[test]
    fn test_filter_graph_add_crop_stored() {
        let mut fg = WasmFilterGraph::new();
        fg.add_crop(10, 20, 100, 200);
        assert_eq!(fg.filter_count(), 1);
        assert_eq!(fg.filters[0].name, "crop");
        assert_eq!(fg.filters[0].params["x"], "10");
        assert_eq!(fg.filters[0].params["y"], "20");
        assert_eq!(fg.filters[0].params["width"], "100");
        assert_eq!(fg.filters[0].params["height"], "200");
    }

    #[test]
    fn test_filter_graph_add_colorspace_stored() {
        let mut fg = WasmFilterGraph::new();
        fg.add_colorspace("rgb", "gray");
        assert_eq!(fg.filters[0].params["from"], "rgb");
        assert_eq!(fg.filters[0].params["to"], "gray");
    }

    #[test]
    fn test_filter_graph_to_json_non_empty() {
        let mut fg = WasmFilterGraph::new();
        fg.add_scale(320, 240);
        let json = fg.to_json().expect("to_json should succeed");
        assert!(json.contains("scale"), "expected 'scale' in JSON: {json}");
        assert!(json.contains("320"), "expected '320' in JSON: {json}");
    }

    #[test]
    fn test_filter_graph_dimensions_stored() {
        let mut fg = WasmFilterGraph::new();
        fg.set_dimensions(1920, 1080);
        assert_eq!(fg.width, 1920);
        assert_eq!(fg.height, 1080);
    }

    #[test]
    fn test_filter_graph_multiple_filters_ordered() {
        let mut fg = WasmFilterGraph::new();
        fg.add_scale(640, 480);
        fg.add_crop(0, 0, 320, 240);
        fg.add_colorspace("rgb", "gray");
        fg.add_volume(-6.0);
        assert_eq!(fg.filter_count(), 4);
        assert_eq!(fg.filters[0].name, "scale");
        assert_eq!(fg.filters[1].name, "crop");
        assert_eq!(fg.filters[2].name, "colorspace");
        assert_eq!(fg.filters[3].name, "volume");
    }

    // ─── volume gain formula (pure arithmetic) ────────────────────────────

    #[test]
    fn test_volume_gain_0db_is_unity() {
        let linear = 10.0_f32.powf(0.0_f32 / 20.0);
        assert!((linear - 1.0).abs() < 1e-6, "0 dB → gain 1.0, got {linear}");
    }

    #[test]
    fn test_volume_gain_20db_is_ten() {
        let linear = 10.0_f32.powf(20.0_f32 / 20.0);
        assert!(
            (linear - 10.0).abs() < 1e-4,
            "20 dB → gain 10.0, got {linear}"
        );
    }

    #[test]
    fn test_volume_gain_minus_20db_is_tenth() {
        let linear = 10.0_f32.powf(-20.0_f32 / 20.0);
        assert!(
            (linear - 0.1).abs() < 1e-6,
            "-20 dB → gain 0.1, got {linear}"
        );
    }

    #[test]
    fn test_volume_gain_6db_doubles() {
        // 6.02 dB ≈ ×2
        let linear = 10.0_f32.powf(6.02_f32 / 20.0);
        assert!((linear - 2.0).abs() < 0.01, "~6 dB → ~×2, got {linear}");
    }
}
