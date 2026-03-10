//! WebAssembly bindings for color management from `oximedia-colormgmt`.
//!
//! Provides color-space conversion, tone mapping, gamut checking, delta-E
//! calculations, and a chainable color pipeline -- all operating in-memory
//! without file-system access.

use wasm_bindgen::prelude::*;

use oximedia_colormgmt::colorspaces::ColorSpace;
use oximedia_colormgmt::delta_e;
use oximedia_colormgmt::gamut::{GamutMapper, GamutMappingAlgorithm};
use oximedia_colormgmt::hdr::{ToneMapper, ToneMappingOperator};
use oximedia_colormgmt::pipeline::{ColorPipeline, ColorTransform};
use oximedia_colormgmt::xyz::Lab;

// ---------------------------------------------------------------------------
// Error helper
// ---------------------------------------------------------------------------

fn js_err(msg: impl std::fmt::Display) -> JsValue {
    crate::utils::js_err(&format!("{msg}"))
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Convert an image buffer between color spaces.
///
/// `data` is a flat `[r,g,b, r,g,b, ...]` array with values in [0, 1].
/// Returns the converted buffer.
#[wasm_bindgen]
pub fn wasm_convert_colorspace(
    data: &[f64],
    width: u32,
    height: u32,
    from_space: &str,
    to_space: &str,
) -> Result<Vec<f64>, JsValue> {
    let expected = (width as usize) * (height as usize) * 3;
    if data.len() != expected {
        return Err(js_err(format!(
            "Data length {} != {}x{}x3 = {}",
            data.len(),
            width,
            height,
            expected,
        )));
    }
    let src = resolve_cs(from_space)?;
    let dst = resolve_cs(to_space)?;

    let mut pipeline = ColorPipeline::new();
    pipeline.add_transform(ColorTransform::ColorSpaceConversion { from: src, to: dst });

    let mut buf = data.to_vec();
    pipeline.transform_image(&mut buf);
    Ok(buf)
}

/// Apply tone mapping to an HDR image buffer.
///
/// `data` is flat `[r,g,b,...]` in linear HDR space.
/// `operator`: `reinhard`, `reinhard_extended`, `hable`, `aces`, `linear`.
/// `peak_luminance`: input peak luminance in nits.
#[wasm_bindgen]
pub fn wasm_apply_tone_map(
    data: &[f64],
    width: u32,
    height: u32,
    operator: &str,
    peak_luminance: f64,
) -> Result<Vec<f64>, JsValue> {
    let expected = (width as usize) * (height as usize) * 3;
    if data.len() != expected {
        return Err(js_err(format!(
            "Data length {} != {}x{}x3 = {}",
            data.len(),
            width,
            height,
            expected,
        )));
    }
    let op = parse_tone_op(operator)?;
    let mapper = ToneMapper::new(op, peak_luminance, 100.0);

    let out: Vec<f64> = data
        .chunks_exact(3)
        .flat_map(|px| {
            let mapped = mapper.apply([px[0], px[1], px[2]]);
            mapped.into_iter()
        })
        .collect();
    Ok(out)
}

/// Calculate delta-E 1976 between two Lab colors.
#[wasm_bindgen]
pub fn wasm_delta_e(l1: f64, a1: f64, b1: f64, l2: f64, a2: f64, b2: f64) -> Result<f64, JsValue> {
    let lab1 = Lab::new(l1, a1, b1);
    let lab2 = Lab::new(l2, a2, b2);
    Ok(delta_e::delta_e_1976(&lab1, &lab2))
}

/// Calculate delta-E 2000 (CIEDE2000) between two Lab colors.
#[wasm_bindgen]
pub fn wasm_delta_e_2000(
    l1: f64,
    a1: f64,
    b1: f64,
    l2: f64,
    a2: f64,
    b2: f64,
) -> Result<f64, JsValue> {
    let lab1 = Lab::new(l1, a1, b1);
    let lab2 = Lab::new(l2, a2, b2);
    Ok(delta_e::delta_e_2000(&lab1, &lab2))
}

/// Return a JSON array of supported color spaces.
#[wasm_bindgen]
pub fn wasm_list_colorspaces() -> Result<String, JsValue> {
    let list = serde_json::json!([
        { "name": "srgb",         "gamut": "BT.709",   "is_hdr": false, "is_linear": false },
        { "name": "rec709",       "gamut": "BT.709",   "is_hdr": false, "is_linear": false },
        { "name": "rec2020",      "gamut": "BT.2020",  "is_hdr": true,  "is_linear": true  },
        { "name": "rec2020_pq",   "gamut": "BT.2020",  "is_hdr": true,  "is_linear": false },
        { "name": "rec2020_hlg",  "gamut": "BT.2020",  "is_hdr": true,  "is_linear": false },
        { "name": "display_p3",   "gamut": "P3",       "is_hdr": false, "is_linear": false },
        { "name": "dci_p3",       "gamut": "P3",       "is_hdr": false, "is_linear": false },
        { "name": "adobe_rgb",    "gamut": "Adobe RGB", "is_hdr": false, "is_linear": false },
        { "name": "prophoto_rgb", "gamut": "ProPhoto",  "is_hdr": false, "is_linear": false },
        { "name": "linear_rec709","gamut": "BT.709",   "is_hdr": false, "is_linear": true  }
    ]);
    serde_json::to_string(&list).map_err(|e| js_err(format!("JSON error: {e}")))
}

/// Return a JSON array of available tone mapping operators.
#[wasm_bindgen]
pub fn wasm_list_tone_map_operators() -> Result<String, JsValue> {
    let ops = serde_json::json!([
        { "name": "reinhard",          "description": "Reinhard global operator" },
        { "name": "reinhard_extended", "description": "Reinhard with white point" },
        { "name": "hable",            "description": "Hable (Uncharted 2) filmic" },
        { "name": "aces",             "description": "ACES filmic tone mapping" },
        { "name": "linear",           "description": "Simple linear clamp" }
    ]);
    serde_json::to_string(&ops).map_err(|e| js_err(format!("JSON error: {e}")))
}

/// Check whether an RGB triplet is inside a named color-space gamut ([0,1]).
#[wasm_bindgen]
pub fn wasm_gamut_check(r: f64, g: f64, b: f64, colorspace: &str) -> Result<bool, JsValue> {
    let _cs = resolve_cs(colorspace)?;
    Ok((0.0..=1.0).contains(&r) && (0.0..=1.0).contains(&g) && (0.0..=1.0).contains(&b))
}

// ---------------------------------------------------------------------------
// WasmColorPipeline
// ---------------------------------------------------------------------------

/// Chainable color transformation pipeline for in-browser processing.
#[wasm_bindgen]
pub struct WasmColorPipeline {
    descriptions: Vec<String>,
    inner: ColorPipeline,
}

#[wasm_bindgen]
impl WasmColorPipeline {
    /// Create an empty pipeline.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            descriptions: Vec::new(),
            inner: ColorPipeline::new(),
        }
    }

    /// Append a color-space conversion.
    pub fn add_colorspace_conversion(&mut self, from: &str, to: &str) -> Result<(), JsValue> {
        let src = resolve_cs(from)?;
        let dst = resolve_cs(to)?;
        self.inner
            .add_transform(ColorTransform::ColorSpaceConversion { from: src, to: dst });
        self.descriptions
            .push(format!("ColorSpaceConversion({from} -> {to})"));
        Ok(())
    }

    /// Append a tone-mapping step.
    pub fn add_tone_map(&mut self, operator: &str, peak_luminance: f64) -> Result<(), JsValue> {
        let op = parse_tone_op(operator)?;
        let mapper = ToneMapper::new(op, peak_luminance, 100.0);
        self.inner.add_transform(ColorTransform::ToneMap(mapper));
        self.descriptions
            .push(format!("ToneMap({operator}, peak={peak_luminance})"));
        Ok(())
    }

    /// Append a gamut-mapping step.
    pub fn add_gamut_map(&mut self, algorithm: &str) -> Result<(), JsValue> {
        let algo = parse_gamut_algo(algorithm)?;
        let mapper = GamutMapper::new(algo);
        self.inner.add_transform(ColorTransform::GamutMap(mapper));
        self.descriptions.push(format!("GamutMap({algorithm})"));
        Ok(())
    }

    /// Append an exposure adjustment in photographic stops.
    pub fn add_exposure(&mut self, stops: f64) -> Result<(), JsValue> {
        self.inner.add_transform(ColorTransform::Exposure(stops));
        self.descriptions.push(format!("Exposure({stops})"));
        Ok(())
    }

    /// Append a contrast adjustment.
    pub fn add_contrast(&mut self, amount: f64) -> Result<(), JsValue> {
        self.inner.add_transform(ColorTransform::Contrast(amount));
        self.descriptions.push(format!("Contrast({amount})"));
        Ok(())
    }

    /// Append a saturation adjustment.
    pub fn add_saturation(&mut self, amount: f64) -> Result<(), JsValue> {
        self.inner.add_transform(ColorTransform::Saturation(amount));
        self.descriptions.push(format!("Saturation({amount})"));
        Ok(())
    }

    /// Transform a single pixel, returning `[r, g, b]`.
    pub fn transform_pixel(&self, r: f64, g: f64, b: f64) -> Result<Vec<f64>, JsValue> {
        let out = self.inner.transform_pixel([r, g, b]);
        Ok(vec![out[0], out[1], out[2]])
    }

    /// Transform an image buffer (`[r,g,b, r,g,b, ...]`).
    pub fn transform_image(
        &self,
        data: &[f64],
        width: u32,
        height: u32,
    ) -> Result<Vec<f64>, JsValue> {
        let expected = (width as usize) * (height as usize) * 3;
        if data.len() != expected {
            return Err(js_err(format!(
                "Data length {} != {}x{}x3 = {}",
                data.len(),
                width,
                height,
                expected,
            )));
        }
        let mut buf = data.to_vec();
        self.inner.transform_image(&mut buf);
        Ok(buf)
    }

    /// Number of transform steps.
    pub fn step_count(&self) -> usize {
        self.inner.len()
    }

    /// Export the pipeline configuration as a JSON array of step descriptions.
    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.descriptions).map_err(|e| js_err(format!("JSON error: {e}")))
    }
}

impl Default for WasmColorPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn resolve_cs(name: &str) -> Result<ColorSpace, JsValue> {
    match name.to_ascii_lowercase().as_str() {
        "srgb" => ColorSpace::srgb().map_err(|e| js_err(e)),
        "rec709" | "bt709" | "rec.709" => ColorSpace::rec709().map_err(|e| js_err(e)),
        "rec2020" | "bt2020" | "rec.2020" => ColorSpace::rec2020().map_err(|e| js_err(e)),
        "rec2020_pq" | "rec2020pq" => ColorSpace::rec2020_pq().map_err(|e| js_err(e)),
        "rec2020_hlg" | "rec2020hlg" => ColorSpace::rec2020_hlg().map_err(|e| js_err(e)),
        "display_p3" | "displayp3" | "p3" => ColorSpace::display_p3().map_err(|e| js_err(e)),
        "dci_p3" | "dcip3" => ColorSpace::dci_p3().map_err(|e| js_err(e)),
        "adobe_rgb" | "adobergb" => ColorSpace::adobe_rgb().map_err(|e| js_err(e)),
        "prophoto_rgb" | "prophotorgb" | "prophoto" => {
            ColorSpace::prophoto_rgb().map_err(|e| js_err(e))
        }
        "linear_rec709" | "linear_bt709" | "linear" => {
            ColorSpace::linear_rec709().map_err(|e| js_err(e))
        }
        other => Err(js_err(format!(
            "Unknown color space '{other}'. Use wasm_list_colorspaces() to see available names."
        ))),
    }
}

fn parse_tone_op(name: &str) -> Result<ToneMappingOperator, JsValue> {
    match name.to_ascii_lowercase().as_str() {
        "reinhard" => Ok(ToneMappingOperator::Reinhard),
        "reinhard_extended" | "reinhardextended" => Ok(ToneMappingOperator::ReinhardExtended),
        "hable" | "uncharted2" => Ok(ToneMappingOperator::Hable),
        "aces" | "aces_filmic" => Ok(ToneMappingOperator::Aces),
        "linear" | "clamp" => Ok(ToneMappingOperator::Linear),
        other => Err(js_err(format!(
            "Unknown tone map operator '{other}'. \
             Valid: reinhard, reinhard_extended, hable, aces, linear"
        ))),
    }
}

fn parse_gamut_algo(name: &str) -> Result<GamutMappingAlgorithm, JsValue> {
    match name.to_ascii_lowercase().as_str() {
        "clip" => Ok(GamutMappingAlgorithm::Clip),
        "compress" => Ok(GamutMappingAlgorithm::Compress),
        "desaturate" => Ok(GamutMappingAlgorithm::Desaturate),
        "perceptual" => Ok(GamutMappingAlgorithm::Perceptual),
        other => Err(js_err(format!(
            "Unknown gamut algorithm '{other}'. Valid: clip, compress, desaturate, perceptual"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_srgb() {
        let cs = resolve_cs("srgb").expect("should resolve srgb");
        assert_eq!(cs.name, "sRGB");
    }

    #[test]
    fn test_resolve_unknown() {
        assert!(resolve_cs("bogus").is_err());
    }

    #[test]
    fn test_delta_e_same() {
        let de = wasm_delta_e(50.0, 0.0, 0.0, 50.0, 0.0, 0.0).expect("ok");
        assert!((de - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_delta_e_2000_same() {
        let de = wasm_delta_e_2000(50.0, 0.0, 0.0, 50.0, 0.0, 0.0).expect("ok");
        assert!((de - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_gamut_check_in() {
        assert!(wasm_gamut_check(0.5, 0.5, 0.5, "srgb").expect("ok"));
    }

    #[test]
    fn test_gamut_check_out() {
        assert!(!wasm_gamut_check(1.5, 0.5, 0.5, "srgb").expect("ok"));
    }

    #[test]
    fn test_list_colorspaces_json() {
        let json = wasm_list_colorspaces().expect("ok");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert!(parsed.as_array().expect("array").len() >= 8);
    }

    #[test]
    fn test_list_tone_map_operators_json() {
        let json = wasm_list_tone_map_operators().expect("ok");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert!(parsed.as_array().expect("array").len() >= 4);
    }

    #[test]
    fn test_convert_colorspace_identity() {
        // srgb -> srgb should be roughly identity
        let data = vec![0.5, 0.3, 0.7];
        let out = wasm_convert_colorspace(&data, 1, 1, "srgb", "srgb").expect("ok");
        assert!((out[0] - 0.5).abs() < 0.01);
        assert!((out[1] - 0.3).abs() < 0.01);
        assert!((out[2] - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_pipeline_step_count() {
        let mut p = WasmColorPipeline::new();
        assert_eq!(p.step_count(), 0);
        p.add_exposure(1.0).expect("ok");
        p.add_contrast(1.2).expect("ok");
        assert_eq!(p.step_count(), 2);
    }
}
