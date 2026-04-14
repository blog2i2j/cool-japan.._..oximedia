//! WebAssembly bindings for LUT (Look-Up Table) processing from `oximedia-lut`.
//!
//! Provides 3-D LUT application, identity LUT generation, photographic presets,
//! simple .cube parsing, and pixel-level colour grading — all operating in-memory
//! without file-system access, for browser-based colour grading workflows.

use wasm_bindgen::prelude::*;

use oximedia_lut::hald_clut::Lut3DData;
use oximedia_lut::photographic_luts::PhotoLutPreset;
use oximedia_lut::{Lut3d, LutSize};

// ---------------------------------------------------------------------------
// Error helper
// ---------------------------------------------------------------------------

fn js_err(msg: impl std::fmt::Display) -> JsValue {
    crate::utils::js_err(&format!("{msg}"))
}

// ---------------------------------------------------------------------------
// Photographic preset helpers
// ---------------------------------------------------------------------------

/// Return a JSON array of available photographic/cinematic LUT preset names.
///
/// Each entry has `name` (identifier) and `label` (human-readable) fields.
#[wasm_bindgen]
pub fn wasm_lut_preset_names() -> Result<String, JsValue> {
    let presets = [
        ("film_noir", "Film Noir"),
        ("kodachrome", "Kodachrome"),
        ("fuji_chrome", "FujiChrome"),
        ("cinematic_teal", "Cinematic Teal/Orange"),
        ("bleach", "Bleach Bypass"),
        ("vintage", "Vintage"),
        ("moonlight", "Moonlight"),
        ("sunrise", "Sunrise"),
        ("commercial", "Commercial"),
        ("bw_high", "B&W High Contrast"),
        ("bw_low", "B&W Low Contrast"),
        ("log_to_rec709", "Log to Rec.709"),
    ];
    let list: Vec<serde_json::Value> = presets
        .iter()
        .map(|(name, label)| serde_json::json!({ "name": name, "label": label }))
        .collect();
    serde_json::to_string(&list).map_err(|e| js_err(format!("JSON error: {e}")))
}

/// Apply a named photographic LUT preset to a single RGB pixel.
///
/// `r`, `g`, `b` must be in [0, 1].  Returns a JSON object `{r, g, b}`.
///
/// # Errors
/// Returns an error for unrecognised preset names.
#[wasm_bindgen]
pub fn wasm_apply_lut_preset_pixel(
    preset_name: &str,
    r: f32,
    g: f32,
    b: f32,
) -> Result<String, JsValue> {
    let preset = resolve_photo_preset(preset_name)?;
    let (ro, go, bo) = PhotoLutPreset::apply_to_pixel(r, g, b, &preset);
    serde_json::to_string(&serde_json::json!({ "r": ro, "g": go, "b": bo }))
        .map_err(|e| js_err(format!("JSON error: {e}")))
}

/// Apply a named photographic LUT preset to a flat RGB frame buffer.
///
/// `data` is `[r, g, b, r, g, b, ...]` with values in [0, 1].
/// Returns the transformed buffer with the same length.
///
/// # Errors
/// Returns an error for unrecognised preset names or non-multiple-of-3 lengths.
#[wasm_bindgen]
pub fn wasm_apply_lut_preset_frame(
    data: &[f32],
    preset_name: &str,
) -> Result<Vec<f32>, JsValue> {
    if data.len() % 3 != 0 {
        return Err(js_err(format!(
            "data length {} is not divisible by 3",
            data.len()
        )));
    }
    let preset = resolve_photo_preset(preset_name)?;
    let out: Vec<f32> = data
        .chunks_exact(3)
        .flat_map(|px| {
            let (ro, go, bo) = PhotoLutPreset::apply_to_pixel(px[0], px[1], px[2], &preset);
            [ro, go, bo]
        })
        .collect();
    Ok(out)
}

// ---------------------------------------------------------------------------
// Identity LUT
// ---------------------------------------------------------------------------

/// Generate an identity 3-D LUT as a flat JSON-serialisable array.
///
/// `size` must be one of 17, 33, or 65 (defaults to 33 for other values).
/// Returns the LUT data as a JSON object:
/// `{ "size": N, "data": [[r, g, b], ...] }`.
#[wasm_bindgen]
pub fn wasm_lut_identity(size: u32) -> Result<String, JsValue> {
    let lut_size = LutSize::from(size as usize);
    let lut = Lut3d::identity(lut_size);
    let s = lut.size();
    // Collect data as nested [r, g, b] triples for easy JS consumption.
    let entries: Vec<serde_json::Value> = (0..s)
        .flat_map(|r| {
            (0..s).flat_map(move |g| {
                (0..s).map(move |b| {
                    let rgb = [r as f64 / (s - 1) as f64, g as f64 / (s - 1) as f64, b as f64 / (s - 1) as f64];
                    serde_json::json!([rgb[0], rgb[1], rgb[2]])
                })
            })
        })
        .collect();

    let result = serde_json::json!({
        "size": s,
        "total_entries": lut.entry_count(),
        "data": entries,
    });
    serde_json::to_string(&result).map_err(|e| js_err(format!("JSON error: {e}")))
}

// ---------------------------------------------------------------------------
// WasmLut3d — stateful 3-D LUT for multi-pixel workflows
// ---------------------------------------------------------------------------

/// Stateful 3-D LUT that can be applied to multiple pixels or frames.
#[wasm_bindgen]
pub struct WasmLut3d {
    inner: Lut3DData,
}

#[wasm_bindgen]
impl WasmLut3d {
    /// Create a new identity 3-D LUT with the given number of divisions per axis.
    ///
    /// `size` is typically 17, 33, or 65; other values are clamped to 33.
    #[wasm_bindgen(constructor)]
    pub fn new(size: u32) -> WasmLut3d {
        let s = match size {
            2..=65 => size as usize,
            _ => 33,
        };
        WasmLut3d {
            inner: Lut3DData::identity(s),
        }
    }

    /// Load a named photographic preset into this LUT.
    ///
    /// # Errors
    /// Returns an error for unrecognised preset names.
    pub fn load_preset(&mut self, preset_name: &str) -> Result<(), JsValue> {
        let preset = resolve_photo_preset(preset_name)?;
        self.inner = preset.to_lut3d();
        Ok(())
    }

    /// Apply this LUT to a single `(r, g, b)` pixel.
    ///
    /// Returns `[r, g, b]` clamped to [0, 1].
    pub fn apply_pixel(&self, r: f32, g: f32, b: f32) -> Vec<f32> {
        let out = self.inner.lookup(r, g, b);
        vec![out[0], out[1], out[2]]
    }

    /// Apply this LUT to a flat RGB frame buffer.
    ///
    /// `data` is `[r, g, b, r, g, b, ...]` with values in [0, 1].
    ///
    /// # Errors
    /// Returns an error if `data.len()` is not divisible by 3.
    pub fn apply_frame(&self, data: &[f32]) -> Result<Vec<f32>, JsValue> {
        if data.len() % 3 != 0 {
            return Err(js_err(format!(
                "data length {} is not divisible by 3",
                data.len()
            )));
        }
        let out: Vec<f32> = data
            .chunks_exact(3)
            .flat_map(|px| {
                let rgb = self.inner.lookup(px[0], px[1], px[2]);
                [rgb[0], rgb[1], rgb[2]]
            })
            .collect();
        Ok(out)
    }

    /// Return the number of lattice divisions per axis.
    pub fn size(&self) -> usize {
        self.inner.size
    }

    /// Return the total number of LUT entries (size³).
    pub fn total_entries(&self) -> usize {
        self.inner.data.len()
    }
}

// ---------------------------------------------------------------------------
// Simple .cube format parser
// ---------------------------------------------------------------------------

/// Parse a minimal .cube LUT text and apply it to a flat RGB frame buffer.
///
/// The parser supports `TITLE`, `LUT_3D_SIZE`, and bare float triplets.
/// `data` is `[r, g, b, r, g, b, ...]` in [0, 1].
/// Returns the transformed buffer.
///
/// # Errors
/// Returns an error for malformed .cube content or non-multiple-of-3 data.
#[wasm_bindgen]
pub fn wasm_apply_cube_lut(cube_text: &str, data: &[f32]) -> Result<Vec<f32>, JsValue> {
    if data.len() % 3 != 0 {
        return Err(js_err(format!(
            "data length {} is not divisible by 3",
            data.len()
        )));
    }
    let lut = parse_cube(cube_text)?;
    let out: Vec<f32> = data
        .chunks_exact(3)
        .flat_map(|px| {
            let rgb = lut.lookup(px[0], px[1], px[2]);
            [rgb[0], rgb[1], rgb[2]]
        })
        .collect();
    Ok(out)
}

/// Parse a .cube LUT and return metadata as a JSON object.
///
/// Returns `{ "size": N, "title": "..." }`.
///
/// # Errors
/// Returns an error for malformed .cube content.
#[wasm_bindgen]
pub fn wasm_inspect_cube_lut(cube_text: &str) -> Result<String, JsValue> {
    let lut = parse_cube(cube_text)?;
    let result = serde_json::json!({
        "size": lut.size,
        "total_entries": lut.data.len(),
    });
    serde_json::to_string(&result).map_err(|e| js_err(format!("JSON error: {e}")))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn resolve_photo_preset(name: &str) -> Result<PhotoLutPreset, JsValue> {
    match name.to_ascii_lowercase().replace('-', "_").as_str() {
        "film_noir" | "filmnoir" => Ok(PhotoLutPreset::FilmNoir),
        "kodachrome" => Ok(PhotoLutPreset::Kodachrome),
        "fuji_chrome" | "fujichrome" => Ok(PhotoLutPreset::FujiChrome),
        "cinematic_teal" | "cinematicteal" | "teal_orange" => Ok(PhotoLutPreset::CinematicTeal),
        "bleach" | "bleach_bypass" => Ok(PhotoLutPreset::Bleach),
        "vintage" => Ok(PhotoLutPreset::Vintage),
        "moonlight" => Ok(PhotoLutPreset::Moonlight),
        "sunrise" => Ok(PhotoLutPreset::Sunrise),
        "commercial" => Ok(PhotoLutPreset::Commercial),
        "bw_high" | "bwhigh" => Ok(PhotoLutPreset::BwHigh),
        "bw_low" | "bwlow" => Ok(PhotoLutPreset::BwLow),
        "log_to_rec709" | "logtorec709" => Ok(PhotoLutPreset::LogToRec709),
        other => Err(js_err(format!(
            "Unknown LUT preset '{other}'. \
             Use wasm_lut_preset_names() to list available presets."
        ))),
    }
}

/// Minimal .cube parser that produces a `Lut3DData`.
fn parse_cube(cube_text: &str) -> Result<Lut3DData, JsValue> {
    let mut size: Option<usize> = None;
    let mut entries: Vec<[f32; 3]> = Vec::new();

    for line in cube_text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("LUT_3D_SIZE") {
            let s = rest
                .trim()
                .parse::<usize>()
                .map_err(|_| js_err(format!("Invalid LUT_3D_SIZE: '{}'", rest.trim())))?;
            if s < 2 || s > 256 {
                return Err(js_err(format!("LUT_3D_SIZE {s} is out of range [2, 256]")));
            }
            size = Some(s);
            continue;
        }
        if trimmed.starts_with("TITLE")
            || trimmed.starts_with("DOMAIN_MIN")
            || trimmed.starts_with("DOMAIN_MAX")
            || trimmed.starts_with("LUT_1D_SIZE")
        {
            continue;
        }
        // Try to parse as an RGB triplet.
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() == 3 {
            let r = parts[0]
                .parse::<f32>()
                .map_err(|_| js_err(format!("Parse error in LUT data: '{trimmed}'")))?;
            let g = parts[1]
                .parse::<f32>()
                .map_err(|_| js_err(format!("Parse error in LUT data: '{trimmed}'")))?;
            let b = parts[2]
                .parse::<f32>()
                .map_err(|_| js_err(format!("Parse error in LUT data: '{trimmed}'")))?;
            entries.push([r, g, b]);
        }
    }

    let s = size.ok_or_else(|| js_err("Missing LUT_3D_SIZE directive in .cube file"))?;
    let expected = s * s * s;
    if entries.len() != expected {
        return Err(js_err(format!(
            "Expected {expected} LUT entries ({}³) but found {}",
            s,
            entries.len()
        )));
    }
    Ok(Lut3DData { size: s, data: entries })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lut_preset_names_json() {
        let json = wasm_lut_preset_names().expect("ok");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert!(parsed.as_array().expect("array").len() >= 10);
    }

    #[test]
    fn test_apply_lut_preset_pixel_identity_like() {
        // Vintage shouldn't map black to black exactly, but output should stay in [0,1]
        let out_json = wasm_apply_lut_preset_pixel("film_noir", 0.5, 0.5, 0.5).expect("ok");
        let v: serde_json::Value = serde_json::from_str(&out_json).expect("valid json");
        let r = v["r"].as_f64().expect("r");
        let g = v["g"].as_f64().expect("g");
        let b = v["b"].as_f64().expect("b");
        assert!((0.0..=1.0).contains(&r), "r out of range: {r}");
        assert!((0.0..=1.0).contains(&g), "g out of range: {g}");
        assert!((0.0..=1.0).contains(&b), "b out of range: {b}");
    }

    #[test]
    fn test_apply_lut_preset_unknown_error() {
        assert!(wasm_apply_lut_preset_pixel("nonexistent_preset", 0.5, 0.5, 0.5).is_err());
    }

    #[test]
    fn test_apply_lut_preset_frame_length() {
        let data: Vec<f32> = (0..30).map(|i| i as f32 / 30.0).collect();
        let out = wasm_apply_lut_preset_frame(&data, "kodachrome").expect("ok");
        assert_eq!(out.len(), 30);
    }

    #[test]
    fn test_apply_lut_preset_frame_wrong_length() {
        let data = vec![0.5_f32, 0.3]; // not divisible by 3
        assert!(wasm_apply_lut_preset_frame(&data, "kodachrome").is_err());
    }

    #[test]
    fn test_lut_identity_json() {
        let json = wasm_lut_identity(17).expect("ok");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(parsed["size"].as_u64().expect("size"), 17);
        assert_eq!(parsed["total_entries"].as_u64().expect("entries"), 17 * 17 * 17);
    }

    #[test]
    fn test_wasm_lut3d_identity_apply_pixel() {
        let lut = WasmLut3d::new(17);
        // Identity LUT should map each pixel to itself (within floating-point precision).
        let out = lut.apply_pixel(0.5, 0.3, 0.7);
        assert!((out[0] - 0.5).abs() < 0.05, "r: {}", out[0]);
        assert!((out[1] - 0.3).abs() < 0.05, "g: {}", out[1]);
        assert!((out[2] - 0.7).abs() < 0.05, "b: {}", out[2]);
    }

    #[test]
    fn test_wasm_lut3d_size() {
        let lut = WasmLut3d::new(33);
        assert_eq!(lut.size(), 33);
        assert_eq!(lut.total_entries(), 33 * 33 * 33);
    }

    #[test]
    fn test_wasm_lut3d_load_preset() {
        let mut lut = WasmLut3d::new(33);
        lut.load_preset("film_noir").expect("preset loaded");
        // After loading, black input should still map within [0,1]
        let out = lut.apply_pixel(0.0, 0.0, 0.0);
        assert!(out.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn test_parse_cube_identity() {
        // Build a 2-cube identity (8 entries).
        let cube = "LUT_3D_SIZE 2\n\
                    0.0 0.0 0.0\n\
                    1.0 0.0 0.0\n\
                    0.0 1.0 0.0\n\
                    1.0 1.0 0.0\n\
                    0.0 0.0 1.0\n\
                    1.0 0.0 1.0\n\
                    0.0 1.0 1.0\n\
                    1.0 1.0 1.0\n";
        let data = vec![0.5_f32, 0.5, 0.5];
        let out = wasm_apply_cube_lut(cube, &data).expect("ok");
        assert_eq!(out.len(), 3);
        // Trilinear interpolation on identity should give ~(0.5, 0.5, 0.5)
        assert!((out[0] - 0.5).abs() < 0.05, "r: {}", out[0]);
    }

    #[test]
    fn test_parse_cube_missing_size_error() {
        let cube = "0.0 0.0 0.0\n1.0 1.0 1.0\n";
        assert!(wasm_apply_cube_lut(cube, &[0.5_f32, 0.5, 0.5]).is_err());
    }

    #[test]
    fn test_inspect_cube_lut() {
        let cube = "TITLE \"Test LUT\"\nLUT_3D_SIZE 2\n\
                    0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n1.0 1.0 0.0\n\
                    0.0 0.0 1.0\n1.0 0.0 1.0\n0.0 1.0 1.0\n1.0 1.0 1.0\n";
        let json = wasm_inspect_cube_lut(cube).expect("ok");
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(v["size"].as_u64().expect("size"), 2);
        assert_eq!(v["total_entries"].as_u64().expect("entries"), 8);
    }
}
