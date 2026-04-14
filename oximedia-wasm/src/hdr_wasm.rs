//! WebAssembly bindings for HDR processing from `oximedia-hdr`.
//!
//! Provides PQ/HLG transfer function conversions, tone mapping, HDR format
//! detection, and metadata querying — all operating in-memory without
//! file-system access, suitable for browser-based HDR video workflows.

use wasm_bindgen::prelude::*;

use oximedia_hdr::transfer_function::{hlg_eotf, hlg_oetf, pq_eotf, pq_oetf, sdr_gamma, sdr_gamma_inv, TransferFunction};
use oximedia_hdr::tone_mapping::{ToneMappingConfig, ToneMappingOperator, ToneMapper};

// ---------------------------------------------------------------------------
// Error helper
// ---------------------------------------------------------------------------

fn js_err(msg: impl std::fmt::Display) -> JsValue {
    crate::utils::js_err(&format!("{msg}"))
}

// ---------------------------------------------------------------------------
// Transfer function conversions
// ---------------------------------------------------------------------------

/// Apply the PQ OETF: scene-linear (normalised to 10 000 nits) → PQ signal [0, 1].
///
/// `linear_value` must be ≥ 0.0.
///
/// # Errors
/// Returns an error if `linear_value` is negative.
#[wasm_bindgen]
pub fn wasm_pq_oetf(linear_value: f64) -> Result<f64, JsValue> {
    pq_oetf(linear_value).map_err(|e| js_err(e))
}

/// Apply the PQ EOTF: PQ signal [0, 1] → scene-linear (normalised to 10 000 nits).
///
/// Multiply the result by 10 000 to obtain luminance in nits.
///
/// # Errors
/// Returns an error if `pq_value` is outside [0, 1].
#[wasm_bindgen]
pub fn wasm_pq_eotf(pq_value: f64) -> Result<f64, JsValue> {
    pq_eotf(pq_value).map_err(|e| js_err(e))
}

/// Apply the HLG OETF: scene-linear [0, 1] → HLG signal [0, 1].
///
/// # Errors
/// Returns an error if `linear_value` is negative.
#[wasm_bindgen]
pub fn wasm_hlg_oetf(linear_value: f64) -> Result<f64, JsValue> {
    hlg_oetf(linear_value).map_err(|e| js_err(e))
}

/// Apply the HLG EOTF: HLG signal [0, 1] → scene-linear [0, 1].
///
/// # Errors
/// Returns an error if `hlg_value` is outside [0, 1].
#[wasm_bindgen]
pub fn wasm_hlg_eotf(hlg_value: f64) -> Result<f64, JsValue> {
    hlg_eotf(hlg_value).map_err(|e| js_err(e))
}

/// Apply SDR gamma EOTF (gamma 2.2): encoded signal → scene-linear.
#[wasm_bindgen]
pub fn wasm_sdr_gamma(encoded_value: f64) -> f64 {
    sdr_gamma(encoded_value)
}

/// Apply SDR gamma OETF (gamma 1/2.2): scene-linear → encoded signal.
#[wasm_bindgen]
pub fn wasm_sdr_gamma_inv(linear_value: f64) -> f64 {
    sdr_gamma_inv(linear_value)
}

/// Convert an entire PQ-encoded frame buffer to scene-linear.
///
/// `data` is a flat array of PQ signal values in [0, 1] (any number of channels).
/// Returns the corresponding linear values (normalised to 10 000 nits).
///
/// # Errors
/// Returns an error if any value is outside [0, 1].
#[wasm_bindgen]
pub fn wasm_pq_eotf_frame(data: &[f64]) -> Result<Vec<f64>, JsValue> {
    data.iter()
        .map(|&v| pq_eotf(v).map_err(|e| js_err(e)))
        .collect()
}

/// Convert an entire scene-linear frame buffer to PQ-encoded signals.
///
/// `data` is a flat array of linear values normalised to 10 000 nits (≥ 0).
/// Returns PQ signal values in [0, 1].
///
/// # Errors
/// Returns an error if any value is negative.
#[wasm_bindgen]
pub fn wasm_pq_oetf_frame(data: &[f64]) -> Result<Vec<f64>, JsValue> {
    data.iter()
        .map(|&v| pq_oetf(v).map_err(|e| js_err(e)))
        .collect()
}

/// Convert an entire HLG-encoded frame buffer to scene-linear.
///
/// # Errors
/// Returns an error if any value is outside [0, 1].
#[wasm_bindgen]
pub fn wasm_hlg_eotf_frame(data: &[f64]) -> Result<Vec<f64>, JsValue> {
    data.iter()
        .map(|&v| hlg_eotf(v).map_err(|e| js_err(e)))
        .collect()
}

/// Convert a scene-linear frame buffer to HLG-encoded signals.
///
/// # Errors
/// Returns an error if any value is negative.
#[wasm_bindgen]
pub fn wasm_hlg_oetf_frame(data: &[f64]) -> Result<Vec<f64>, JsValue> {
    data.iter()
        .map(|&v| hlg_oetf(v).map_err(|e| js_err(e)))
        .collect()
}

// ---------------------------------------------------------------------------
// Transfer function string API
// ---------------------------------------------------------------------------

/// Apply a named transfer function OETF to a single linear value.
///
/// Supported names: `pq`, `hlg`, `sdr_gamma`.
///
/// # Errors
/// Returns an error for unknown names or out-of-range inputs.
#[wasm_bindgen]
pub fn wasm_apply_oetf(transfer_function: &str, linear_value: f64) -> Result<f64, JsValue> {
    match transfer_function.to_ascii_lowercase().as_str() {
        "pq" | "st2084" | "hdr10" => pq_oetf(linear_value).map_err(|e| js_err(e)),
        "hlg" | "arib_b67" => hlg_oetf(linear_value).map_err(|e| js_err(e)),
        "sdr_gamma" | "sdr" | "gamma22" => Ok(sdr_gamma_inv(linear_value)),
        other => Err(js_err(format!(
            "Unknown transfer function '{other}'. Supported: pq, hlg, sdr_gamma"
        ))),
    }
}

/// Apply a named transfer function EOTF to a single encoded value.
///
/// Supported names: `pq`, `hlg`, `sdr_gamma`.
///
/// # Errors
/// Returns an error for unknown names or out-of-range inputs.
#[wasm_bindgen]
pub fn wasm_apply_eotf(transfer_function: &str, encoded_value: f64) -> Result<f64, JsValue> {
    match transfer_function.to_ascii_lowercase().as_str() {
        "pq" | "st2084" | "hdr10" => pq_eotf(encoded_value).map_err(|e| js_err(e)),
        "hlg" | "arib_b67" => hlg_eotf(encoded_value).map_err(|e| js_err(e)),
        "sdr_gamma" | "sdr" | "gamma22" => Ok(sdr_gamma(encoded_value)),
        other => Err(js_err(format!(
            "Unknown transfer function '{other}'. Supported: pq, hlg, sdr_gamma"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Tone mapping
// ---------------------------------------------------------------------------

/// Apply HDR-to-SDR tone mapping to a flat RGB image buffer.
///
/// `data` is `[r, g, b, r, g, b, ...]` in linear HDR light, normalised so
/// 1.0 = `input_peak_nits`.  `operator` selects the tone curve:
/// `reinhard`, `hable`, `aces`, `clamp`.  Returns the tone-mapped SDR buffer
/// (linear, before display gamma).
///
/// # Errors
/// Returns an error for unrecognised operators or mismatched buffer sizes.
#[wasm_bindgen]
pub fn wasm_hdr_tone_map(
    data: &[f32],
    operator: &str,
    input_peak_nits: f32,
    output_peak_nits: f32,
) -> Result<Vec<f32>, JsValue> {
    if data.len() % 3 != 0 {
        return Err(js_err(format!(
            "data length {} is not divisible by 3",
            data.len()
        )));
    }
    let op = parse_tone_operator(operator)?;
    let config = ToneMappingConfig {
        operator: op,
        input_peak_nits,
        output_peak_nits,
        exposure: 1.0,
        saturation: 1.0,
        gamma_out: 1.0, // caller applies gamma separately
    };
    let mapper = ToneMapper::new(config);

    let out: Vec<f32> = data
        .chunks_exact(3)
        .flat_map(|px| {
            // Use the maximum component as the proxy luminance (simple per-channel scaling).
            let lum = 0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2];
            let mapped_lum = mapper.map_luminance(lum);
            // Preserve hue/ratio: scale each channel by the luminance ratio.
            let scale = if lum > 1e-7 { mapped_lum / lum } else { mapped_lum };
            [px[0] * scale, px[1] * scale, px[2] * scale]
        })
        .collect();
    Ok(out)
}

/// Return a JSON array describing available tone mapping operators.
///
/// Each entry has `name`, `description`, and `has_parameter` fields.
#[wasm_bindgen]
pub fn wasm_list_tone_operators() -> Result<String, JsValue> {
    let ops = serde_json::json!([
        { "name": "reinhard",  "description": "Classic Reinhard global operator",    "has_parameter": false },
        { "name": "hable",     "description": "Uncharted 2 / Hable filmic curve",    "has_parameter": false },
        { "name": "aces",      "description": "ACES fitted approximation (Narkowicz)", "has_parameter": false },
        { "name": "clamp",     "description": "Hard clamp to [0, 1]",                 "has_parameter": false },
    ]);
    serde_json::to_string(&ops).map_err(|e| js_err(format!("JSON error: {e}")))
}

/// Return a JSON array describing available HDR transfer functions.
#[wasm_bindgen]
pub fn wasm_list_transfer_functions() -> Result<String, JsValue> {
    let tfs = serde_json::json!([
        { "name": "pq",        "full_name": "SMPTE ST 2084 (PQ/HDR10)",    "peak_nits": 10000 },
        { "name": "hlg",       "full_name": "ARIB STD-B67 (HLG)",           "peak_nits": 1000  },
        { "name": "sdr_gamma", "full_name": "Power-law SDR gamma 2.2",      "peak_nits": 100   },
    ]);
    serde_json::to_string(&tfs).map_err(|e| js_err(format!("JSON error: {e}")))
}

// ---------------------------------------------------------------------------
// WasmHdrConverter — stateful per-frame converter
// ---------------------------------------------------------------------------

/// Stateful HDR converter that can be reused across multiple frames.
#[wasm_bindgen]
pub struct WasmHdrConverter {
    transfer: TransferFunction,
    description: String,
}

#[wasm_bindgen]
impl WasmHdrConverter {
    /// Create a converter for the named transfer function.
    ///
    /// Supported names: `pq`, `hlg`, `sdr_gamma`, `linear`.
    ///
    /// # Errors
    /// Returns an error for unrecognised transfer function names.
    #[wasm_bindgen(constructor)]
    pub fn new(transfer_function: &str) -> Result<WasmHdrConverter, JsValue> {
        let (transfer, description) = match transfer_function.to_ascii_lowercase().as_str() {
            "pq" | "st2084" | "hdr10" => (TransferFunction::Pq, "PQ (SMPTE ST 2084)".to_string()),
            "hlg" | "arib_b67" => (TransferFunction::Hlg, "HLG (ARIB STD-B67)".to_string()),
            "sdr_gamma" | "sdr" | "gamma22" => (
                TransferFunction::SdrGamma(2.2),
                "SDR Gamma 2.2".to_string(),
            ),
            "linear" => (TransferFunction::Linear, "Linear (no transfer)".to_string()),
            other => {
                return Err(js_err(format!(
                    "Unknown transfer function '{other}'. Supported: pq, hlg, sdr_gamma, linear"
                )));
            }
        };
        Ok(WasmHdrConverter {
            transfer,
            description,
        })
    }

    /// Return the name of the active transfer function.
    pub fn transfer_function_name(&self) -> String {
        self.description.clone()
    }

    /// Apply the EOTF (encoded → linear) to a buffer of signal values.
    ///
    /// # Errors
    /// Returns an error if any signal value is out of the valid range.
    pub fn apply_eotf(&self, data: &[f64]) -> Result<Vec<f64>, JsValue> {
        data.iter()
            .map(|&v| {
                self.transfer
                    .to_linear(v)
                    .map_err(|e| js_err(format!("{e}")))
            })
            .collect()
    }

    /// Apply the OETF (linear → encoded) to a buffer of linear values.
    ///
    /// # Errors
    /// Returns an error if any linear value is out of the valid range.
    pub fn apply_oetf(&self, data: &[f64]) -> Result<Vec<f64>, JsValue> {
        data.iter()
            .map(|&v| {
                self.transfer
                    .from_linear(v)
                    .map_err(|e| js_err(format!("{e}")))
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn parse_tone_operator(name: &str) -> Result<ToneMappingOperator, JsValue> {
    match name.to_ascii_lowercase().as_str() {
        "reinhard" => Ok(ToneMappingOperator::Reinhard),
        "hable" | "uncharted2" | "filmic" => Ok(ToneMappingOperator::Hable),
        "aces" | "aces_filmic" => Ok(ToneMappingOperator::Aces),
        "clamp" | "clip" => Ok(ToneMappingOperator::Clamp),
        other => Err(js_err(format!(
            "Unknown tone mapping operator '{other}'. \
             Valid: reinhard, hable, aces, clamp"
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
    fn test_pq_roundtrip() {
        // encode 0.5 linear, then decode back
        let linear = 0.5_f64;
        let encoded = wasm_pq_oetf(linear).expect("pq oetf ok");
        let decoded = wasm_pq_eotf(encoded).expect("pq eotf ok");
        assert!((decoded - linear).abs() < 1e-9, "PQ roundtrip: {decoded} != {linear}");
    }

    #[test]
    fn test_hlg_roundtrip() {
        let linear = 0.3_f64;
        let encoded = wasm_hlg_oetf(linear).expect("hlg oetf ok");
        let decoded = wasm_hlg_eotf(encoded).expect("hlg eotf ok");
        assert!((decoded - linear).abs() < 1e-9, "HLG roundtrip: {decoded} != {linear}");
    }

    #[test]
    fn test_sdr_gamma_roundtrip() {
        let linear = 0.4_f64;
        let encoded = wasm_sdr_gamma_inv(linear);
        let decoded = wasm_sdr_gamma(encoded);
        assert!((decoded - linear).abs() < 1e-9, "SDR gamma roundtrip: {decoded} != {linear}");
    }

    #[test]
    fn test_pq_eotf_negative_error() {
        assert!(wasm_pq_eotf(-0.1).is_err());
    }

    #[test]
    fn test_pq_oetf_negative_error() {
        assert!(wasm_pq_oetf(-0.1).is_err());
    }

    #[test]
    fn test_pq_eotf_frame_identity() {
        let data = vec![0.0, 0.5, 1.0];
        let out = wasm_pq_eotf_frame(&data).expect("frame ok");
        assert_eq!(out.len(), 3);
        // 0.0 → 0.0, 1.0 → 1.0 (normalised)
        assert!((out[0] - 0.0).abs() < 1e-9);
        assert!((out[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hlg_eotf_frame_length() {
        let data: Vec<f64> = (0..10).map(|i| i as f64 / 10.0).collect();
        let out = wasm_hlg_eotf_frame(&data).expect("frame ok");
        assert_eq!(out.len(), 10);
    }

    #[test]
    fn test_list_transfer_functions_json() {
        let json = wasm_list_transfer_functions().expect("ok");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(parsed.as_array().expect("array").len(), 3);
    }

    #[test]
    fn test_list_tone_operators_json() {
        let json = wasm_list_tone_operators().expect("ok");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert!(parsed.as_array().expect("array").len() >= 4);
    }

    #[test]
    fn test_apply_oetf_pq() {
        let val = wasm_apply_oetf("pq", 0.5).expect("ok");
        let expected = wasm_pq_oetf(0.5).expect("ok");
        assert!((val - expected).abs() < 1e-12);
    }

    #[test]
    fn test_apply_eotf_hlg() {
        let val = wasm_apply_eotf("hlg", 0.5).expect("ok");
        let expected = wasm_hlg_eotf(0.5).expect("ok");
        assert!((val - expected).abs() < 1e-12);
    }

    #[test]
    fn test_apply_unknown_tf_error() {
        assert!(wasm_apply_oetf("bogus", 0.5).is_err());
        assert!(wasm_apply_eotf("unknown_tf", 0.5).is_err());
    }

    #[test]
    fn test_tone_map_reinhard_dark_input() {
        // Pure black should stay black
        let data = vec![0.0_f32, 0.0, 0.0];
        let out = wasm_hdr_tone_map(&data, "reinhard", 1000.0, 100.0).expect("ok");
        assert_eq!(out.len(), 3);
        assert!((out[0]).abs() < 1e-7);
    }

    #[test]
    fn test_tone_map_unknown_operator_error() {
        let data = vec![0.5_f32, 0.3, 0.1];
        assert!(wasm_hdr_tone_map(&data, "bogus_op", 1000.0, 100.0).is_err());
    }

    #[test]
    fn test_tone_map_mismatched_buffer() {
        let data = vec![0.5_f32, 0.3]; // length 2, not divisible by 3
        assert!(wasm_hdr_tone_map(&data, "reinhard", 1000.0, 100.0).is_err());
    }

    #[test]
    fn test_wasm_hdr_converter_pq_roundtrip() {
        let converter = WasmHdrConverter::new("pq").expect("ok");
        let linear = vec![0.0, 0.1, 0.5, 1.0];
        let encoded = converter.apply_oetf(&linear).expect("oetf ok");
        let decoded = converter.apply_eotf(&encoded).expect("eotf ok");
        for (orig, dec) in linear.iter().zip(decoded.iter()) {
            assert!((dec - orig).abs() < 1e-8, "mismatch: {dec} != {orig}");
        }
    }

    #[test]
    fn test_wasm_hdr_converter_unknown_error() {
        assert!(WasmHdrConverter::new("xyz_transfer").is_err());
    }

    #[test]
    fn test_wasm_hdr_converter_name() {
        let c = WasmHdrConverter::new("hlg").expect("ok");
        assert!(c.transfer_function_name().contains("HLG"));
    }
}
