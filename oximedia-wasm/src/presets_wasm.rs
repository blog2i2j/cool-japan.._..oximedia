//! WebAssembly bindings for `oximedia-transcode` encoding presets.
//!
//! Provides WASM-accessible functions for listing, retrieving, validating,
//! and merging encoding presets. All data is exchanged as JSON strings.

use wasm_bindgen::prelude::*;

use oximedia_transcode::{PresetConfig, QualityMode};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn quality_mode_to_str(mode: &QualityMode) -> &'static str {
    match mode {
        QualityMode::Low => "low",
        QualityMode::Medium => "medium",
        QualityMode::High => "high",
        QualityMode::VeryHigh => "very_high",
        QualityMode::Custom => "custom",
    }
}

struct NamedPreset {
    name: &'static str,
    description: &'static str,
    config: PresetConfig,
}

fn builtin_presets() -> Vec<NamedPreset> {
    vec![
        NamedPreset {
            name: "web_optimized",
            description: "Web-optimized AV1/Opus 720p",
            config: oximedia_transcode::presets::av1_opus(1280, 720, 2_500_000, 128_000),
        },
        NamedPreset {
            name: "archive_quality",
            description: "Archive-quality AV1/Opus 1080p",
            config: oximedia_transcode::presets::av1_opus(1920, 1080, 8_000_000, 256_000),
        },
        NamedPreset {
            name: "broadcast_hd",
            description: "Broadcast HD VP9/Opus 1080p",
            config: oximedia_transcode::presets::vp9_opus(1920, 1080, 8_000_000, 256_000),
        },
        NamedPreset {
            name: "social_media",
            description: "Social media VP9/Opus 720p",
            config: oximedia_transcode::presets::vp9_opus(1280, 720, 4_000_000, 128_000),
        },
        NamedPreset {
            name: "youtube_1080p",
            description: "YouTube 1080p AV1/Opus",
            config: oximedia_transcode::presets::av1_opus(1920, 1080, 5_000_000, 192_000),
        },
        NamedPreset {
            name: "fast_preview",
            description: "Fast preview VP9/Opus 480p",
            config: oximedia_transcode::presets::vp9_opus(854, 480, 1_000_000, 64_000),
        },
    ]
}

fn preset_to_json(name: &str, desc: &str, config: &PresetConfig) -> String {
    let vc = config.video_codec.as_deref().unwrap_or("");
    let ac = config.audio_codec.as_deref().unwrap_or("");
    let ct = config.container.as_deref().unwrap_or("");
    let qm = config
        .quality_mode
        .as_ref()
        .map(quality_mode_to_str)
        .unwrap_or("medium");
    let vb = config.video_bitrate.unwrap_or(0);
    let ab = config.audio_bitrate.unwrap_or(0);
    let w = config.width.unwrap_or(0);
    let h = config.height.unwrap_or(0);
    let fps = config
        .frame_rate
        .map(|(n, d)| {
            if d == 0 {
                0.0
            } else {
                f64::from(n) / f64::from(d)
            }
        })
        .unwrap_or(0.0);

    format!(
        "{{\"name\":\"{name}\",\"description\":\"{desc}\",\
         \"video_codec\":\"{vc}\",\"audio_codec\":\"{ac}\",\
         \"container\":\"{ct}\",\"quality_mode\":\"{qm}\",\
         \"video_bitrate\":{vb},\"audio_bitrate\":{ab},\
         \"width\":{w},\"height\":{h},\"frame_rate\":{fps:.3}}}"
    )
}

// ---------------------------------------------------------------------------
// WASM functions
// ---------------------------------------------------------------------------

/// List all built-in encoding presets as a JSON array.
///
/// Returns a JSON array of preset objects with name, codec, resolution, etc.
#[wasm_bindgen]
pub fn wasm_list_encoding_presets() -> String {
    let presets = builtin_presets();
    let items: Vec<String> = presets
        .iter()
        .map(|p| preset_to_json(p.name, p.description, &p.config))
        .collect();
    format!("[{}]", items.join(","))
}

/// Get a specific encoding preset by name as a JSON object.
///
/// Returns an error if the preset is not found.
#[wasm_bindgen]
pub fn wasm_get_preset(name: &str) -> Result<String, JsValue> {
    let presets = builtin_presets();
    for p in &presets {
        if p.name == name {
            return Ok(preset_to_json(p.name, p.description, &p.config));
        }
    }
    Err(crate::utils::js_err(&format!(
        "Preset '{}' not found. Available: {}",
        name,
        presets
            .iter()
            .map(|p| p.name)
            .collect::<Vec<_>>()
            .join(", ")
    )))
}

/// List all preset names as a JSON array of strings.
#[wasm_bindgen]
pub fn wasm_preset_names() -> String {
    let presets = builtin_presets();
    let names: Vec<String> = presets.iter().map(|p| format!("\"{}\"", p.name)).collect();
    format!("[{}]", names.join(","))
}

/// Validate a custom preset JSON string.
///
/// Checks that required fields are present and values are reasonable.
/// Returns a JSON object with "valid" (bool) and "errors" (array of strings).
#[wasm_bindgen]
pub fn wasm_validate_preset(preset_json: &str) -> Result<String, JsValue> {
    let parsed: HashMap<String, serde_json::Value> = serde_json::from_str(preset_json)
        .map_err(|e| crate::utils::js_err(&format!("Invalid JSON: {e}")))?;

    let mut errors: Vec<String> = Vec::new();

    if !parsed.contains_key("video_codec") {
        errors.push("Missing 'video_codec' field".to_string());
    }
    if !parsed.contains_key("audio_codec") {
        errors.push("Missing 'audio_codec' field".to_string());
    }

    if let Some(serde_json::Value::Number(w)) = parsed.get("width") {
        if let Some(wv) = w.as_u64() {
            if wv == 0 || wv > 16384 {
                errors.push(format!("Invalid width: {wv} (must be 1-16384)"));
            }
        }
    }
    if let Some(serde_json::Value::Number(h)) = parsed.get("height") {
        if let Some(hv) = h.as_u64() {
            if hv == 0 || hv > 16384 {
                errors.push(format!("Invalid height: {hv} (must be 1-16384)"));
            }
        }
    }

    if let Some(serde_json::Value::String(qm)) = parsed.get("quality_mode") {
        let valid_modes = ["draft", "low", "medium", "high", "maximum"];
        if !valid_modes.contains(&qm.as_str()) {
            errors.push(format!("Invalid quality_mode: '{qm}'"));
        }
    }

    let valid = errors.is_empty();
    let err_json: Vec<String> = errors.iter().map(|e| format!("\"{e}\"")).collect();
    Ok(format!(
        "{{\"valid\":{valid},\"errors\":[{}]}}",
        err_json.join(",")
    ))
}

/// Merge two preset JSON objects, with the override taking precedence.
///
/// Returns the merged preset as a JSON string.
#[wasm_bindgen]
pub fn wasm_merge_presets(base: &str, override_json: &str) -> Result<String, JsValue> {
    let mut base_map: HashMap<String, serde_json::Value> = serde_json::from_str(base)
        .map_err(|e| crate::utils::js_err(&format!("Invalid base JSON: {e}")))?;
    let override_map: HashMap<String, serde_json::Value> = serde_json::from_str(override_json)
        .map_err(|e| crate::utils::js_err(&format!("Invalid override JSON: {e}")))?;

    for (k, v) in override_map {
        base_map.insert(k, v);
    }

    serde_json::to_string(&base_map)
        .map_err(|e| crate::utils::js_err(&format!("JSON serialization failed: {e}")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_encoding_presets() {
        let json = wasm_list_encoding_presets();
        assert!(json.starts_with('['));
        assert!(json.ends_with(']'));
        assert!(json.contains("web_optimized"));
        assert!(json.contains("youtube_1080p"));
    }

    #[test]
    fn test_get_preset_found() {
        let result = wasm_get_preset("web_optimized");
        assert!(result.is_ok());
        let json = result.expect("should find preset");
        assert!(json.contains("av1"));
    }

    #[test]
    fn test_get_preset_not_found() {
        let result = wasm_get_preset("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_preset_names() {
        let json = wasm_preset_names();
        assert!(json.contains("\"web_optimized\""));
        assert!(json.contains("\"fast_preview\""));
    }

    #[test]
    fn test_validate_preset_valid() {
        let result = wasm_validate_preset(
            r#"{"video_codec":"av1","audio_codec":"opus","width":1920,"height":1080}"#,
        );
        assert!(result.is_ok());
        let json = result.expect("should validate");
        assert!(json.contains("\"valid\":true"));
    }

    #[test]
    fn test_validate_preset_missing_fields() {
        let result = wasm_validate_preset(r#"{"width":1920}"#);
        assert!(result.is_ok());
        let json = result.expect("should validate");
        assert!(json.contains("\"valid\":false"));
    }

    #[test]
    fn test_merge_presets() {
        let base = r#"{"video_codec":"vp9","width":1280}"#;
        let over = r#"{"video_codec":"av1","height":720}"#;
        let result = wasm_merge_presets(base, over);
        assert!(result.is_ok());
        let json = result.expect("should merge");
        assert!(json.contains("av1"));
        assert!(json.contains("1280"));
        assert!(json.contains("720"));
    }
}
