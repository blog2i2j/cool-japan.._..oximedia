//! WebAssembly bindings for automated video editing utilities.
//!
//! Provides WASM-accessible functions for listing auto tasks,
//! validating automation configurations, and listing templates.

use wasm_bindgen::prelude::*;

/// List available auto editing tasks and their status.
///
/// Returns a JSON object with task list and summary info.
/// In a browser environment, this returns an empty task list
/// with the available use cases and configuration options.
#[wasm_bindgen]
pub fn wasm_list_auto_tasks() -> String {
    let result = serde_json::json!({
        "tasks": [],
        "total_count": 0,
        "available_use_cases": [
            "trailer", "highlights", "social", "documentary", "music_video"
        ],
        "available_pacing": [
            "slow", "medium", "fast", "dynamic"
        ],
        "available_aspect_ratios": [
            "16x9", "9x16", "4x3", "1x1", "21x9"
        ],
        "music_sync_modes": [
            "none", "beats", "bars", "downbeats"
        ],
    });
    serde_json::to_string(&result).unwrap_or_else(|_| "{}".to_string())
}

/// Validate an automation configuration.
///
/// `config_json`: JSON object with keys:
///   - `use_case` (string): trailer, highlights, social, documentary, music_video
///   - `target_duration` (f64, optional): target duration in seconds
///   - `pacing` (string, optional): slow, medium, fast, dynamic
///   - `aspect_ratio` (string, optional): 16x9, 9x16, 4x3, 1x1, 21x9
///   - `dramatic_arc` (bool, optional): enable dramatic arc shaping
///   - `music_sync` (string, optional): none, beats, bars, downbeats
///
/// Returns a JSON validation result.
#[wasm_bindgen]
pub fn wasm_validate_automation(config_json: &str) -> Result<String, JsValue> {
    let config: serde_json::Value = serde_json::from_str(config_json)
        .map_err(|e| crate::utils::js_err(&format!("Invalid config JSON: {e}")))?;

    let mut errors: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    // Validate use case
    let use_case = config
        .get("use_case")
        .and_then(|v| v.as_str())
        .unwrap_or("highlights");
    let valid_use_cases = [
        "trailer",
        "highlights",
        "social",
        "documentary",
        "music_video",
    ];
    if !valid_use_cases.contains(&use_case) {
        errors.push(format!(
            "Invalid use_case '{}'. Expected: {}",
            use_case,
            valid_use_cases.join(", ")
        ));
    }

    // Validate pacing
    if let Some(pacing) = config.get("pacing").and_then(|v| v.as_str()) {
        let valid_pacing = ["slow", "medium", "fast", "dynamic"];
        if !valid_pacing.contains(&pacing) {
            errors.push(format!(
                "Invalid pacing '{}'. Expected: {}",
                pacing,
                valid_pacing.join(", ")
            ));
        }
    }

    // Validate aspect ratio
    if let Some(ar) = config.get("aspect_ratio").and_then(|v| v.as_str()) {
        let valid_ar = ["16x9", "9x16", "4x3", "1x1", "21x9"];
        if !valid_ar.contains(&ar) {
            errors.push(format!(
                "Invalid aspect_ratio '{}'. Expected: {}",
                ar,
                valid_ar.join(", ")
            ));
        }
    }

    // Validate music sync
    if let Some(ms) = config.get("music_sync").and_then(|v| v.as_str()) {
        let valid_ms = ["none", "beats", "bars", "downbeats"];
        if !valid_ms.contains(&ms) {
            errors.push(format!(
                "Invalid music_sync '{}'. Expected: {}",
                ms,
                valid_ms.join(", ")
            ));
        }
    }

    // Validate target duration
    if let Some(dur) = config.get("target_duration").and_then(|v| v.as_f64()) {
        if dur <= 0.0 {
            errors.push("target_duration must be positive".to_string());
        } else if dur < 5.0 {
            warnings.push("target_duration < 5s may produce poor results".to_string());
        } else if dur > 3600.0 {
            warnings.push("target_duration > 1hr is unusual for auto editing".to_string());
        }
    }

    // Check recommended combinations
    if use_case == "social" {
        if let Some(dur) = config.get("target_duration").and_then(|v| v.as_f64()) {
            if dur > 60.0 {
                warnings.push("Social clips are typically under 60s".to_string());
            }
        }
    }

    let valid = errors.is_empty();

    let result = serde_json::json!({
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "config": {
            "use_case": use_case,
            "pacing": config.get("pacing").and_then(|v| v.as_str()).unwrap_or("medium"),
            "aspect_ratio": config.get("aspect_ratio").and_then(|v| v.as_str()),
            "dramatic_arc": config.get("dramatic_arc").and_then(|v| v.as_bool()).unwrap_or(false),
            "music_sync": config.get("music_sync").and_then(|v| v.as_str()).unwrap_or("none"),
            "target_duration": config.get("target_duration").and_then(|v| v.as_f64()),
        },
    });

    serde_json::to_string(&result)
        .map_err(|e| crate::utils::js_err(&format!("Serialization error: {e}")))
}

/// List available automation workflow templates.
///
/// Returns a JSON array of template descriptors with configuration defaults.
#[wasm_bindgen]
pub fn wasm_auto_templates() -> String {
    let templates = serde_json::json!([
        {
            "name": "highlight-reel",
            "description": "Generate a highlight reel from longer content",
            "default_use_case": "highlights",
            "default_pacing": "fast",
            "default_duration_seconds": 120,
            "best_for": ["sports", "events", "concerts", "conferences"],
        },
        {
            "name": "social-clips",
            "description": "Generate short clips optimized for social media platforms",
            "default_use_case": "social",
            "default_pacing": "fast",
            "default_duration_seconds": 30,
            "best_for": ["tiktok", "reels", "shorts", "stories"],
            "supported_ratios": ["9x16", "1x1", "16x9"],
        },
        {
            "name": "trailer",
            "description": "Generate a compelling trailer with dramatic arc",
            "default_use_case": "trailer",
            "default_pacing": "dynamic",
            "default_duration_seconds": 90,
            "best_for": ["films", "documentaries", "series", "courses"],
        },
        {
            "name": "batch-transcode",
            "description": "Automated batch transcoding with quality optimization",
            "default_use_case": "highlights",
            "default_pacing": "medium",
            "default_duration_seconds": null,
            "best_for": ["archive", "migration", "platform_delivery"],
        },
        {
            "name": "quality-check",
            "description": "Automated quality analysis and validation pipeline",
            "default_use_case": "highlights",
            "default_pacing": "medium",
            "default_duration_seconds": null,
            "best_for": ["qc_workflow", "ingest_validation", "delivery_check"],
        },
    ]);
    serde_json::to_string(&templates).unwrap_or_else(|_| "[]".to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_auto_tasks() {
        let tasks = wasm_list_auto_tasks();
        assert!(tasks.contains("highlights"));
        assert!(tasks.contains("trailer"));
        assert!(tasks.contains("available_pacing"));
    }

    #[test]
    fn test_validate_automation_valid() {
        let config = r#"{"use_case":"highlights","pacing":"fast","target_duration":60.0}"#;
        let result = wasm_validate_automation(config);
        assert!(result.is_ok());
        let json = result.expect("should validate");
        assert!(json.contains("\"valid\":true"));
    }

    #[test]
    fn test_validate_automation_invalid_use_case() {
        let config = r#"{"use_case":"invalid_type"}"#;
        let result = wasm_validate_automation(config);
        assert!(result.is_ok());
        let json = result.expect("should validate");
        assert!(json.contains("\"valid\":false"));
    }

    #[test]
    fn test_validate_automation_warnings() {
        let config = r#"{"use_case":"social","target_duration":120.0}"#;
        let result = wasm_validate_automation(config);
        assert!(result.is_ok());
        let json = result.expect("should validate");
        assert!(json.contains("typically under 60s"));
    }

    #[test]
    fn test_auto_templates() {
        let templates = wasm_auto_templates();
        assert!(templates.contains("highlight-reel"));
        assert!(templates.contains("social-clips"));
        assert!(templates.contains("trailer"));
        assert!(templates.contains("batch-transcode"));
        assert!(templates.contains("quality-check"));
    }
}
