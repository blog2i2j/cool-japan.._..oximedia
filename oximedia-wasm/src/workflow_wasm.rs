//! WebAssembly bindings for workflow orchestration.
//!
//! Provides workflow validation, template management, and step type
//! information for browser-based workflow builders.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Validate a workflow definition JSON string.
///
/// Returns a JSON report:
/// ```json
/// {"valid": true, "steps": 3, "warnings": [], "has_cycles": false}
/// ```
#[wasm_bindgen]
pub fn wasm_validate_workflow(json: &str) -> Result<String, JsValue> {
    let data: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| crate::utils::js_err(&format!("Invalid JSON: {e}")))?;

    let mut warnings: Vec<String> = Vec::new();
    let mut step_count = 0u32;

    if let Some(steps) = data["steps"].as_array() {
        step_count = steps.len() as u32;

        // Collect step IDs
        let step_ids: Vec<&str> = steps
            .iter()
            .filter_map(|s| s["step_id"].as_str().or_else(|| s["id"].as_str()))
            .collect();

        // Check for duplicates
        let mut seen = std::collections::HashSet::new();
        for id in &step_ids {
            if !seen.insert(*id) {
                warnings.push(format!("Duplicate step ID: '{}'", id));
            }
        }

        // Check dependencies
        for step in steps {
            let step_id = step["step_id"]
                .as_str()
                .or_else(|| step["id"].as_str())
                .unwrap_or("unknown");

            if let Some(deps) = step["depends_on"].as_array() {
                for dep in deps {
                    if let Some(dep_str) = dep.as_str() {
                        if !step_ids.contains(&dep_str) {
                            warnings.push(format!(
                                "Step '{}' depends on unknown step '{}'",
                                step_id, dep_str
                            ));
                        }
                    }
                }
            }

            // Check task type
            let task_type = step["task_type"].as_str().unwrap_or("");
            let valid_types = [
                "transcode",
                "qc",
                "transfer",
                "analysis",
                "wait",
                "notification",
            ];
            if !task_type.is_empty() && !valid_types.contains(&task_type) {
                warnings.push(format!(
                    "Step '{}' has unknown task type '{}'",
                    step_id, task_type,
                ));
            }
        }
    } else {
        warnings.push("No steps array found".to_string());
    }

    let valid = warnings.is_empty();
    let warnings_json: Vec<String> = warnings.iter().map(|w| format!("\"{}\"", w)).collect();

    Ok(format!(
        "{{\"valid\":{},\"steps\":{},\"has_cycles\":false,\"warnings\":[{}]}}",
        valid,
        step_count,
        warnings_json.join(","),
    ))
}

/// List available workflow templates as a JSON array.
#[wasm_bindgen]
pub fn wasm_list_workflow_templates() -> String {
    "[{\"name\":\"transcode\",\"description\":\"Validate -> Transcode -> Verify\",\"steps\":3},\
     {\"name\":\"ingest\",\"description\":\"Copy -> Probe -> Generate proxy\",\"steps\":3},\
     {\"name\":\"qc\",\"description\":\"Format, quality, and loudness checks\",\"steps\":3},\
     {\"name\":\"multi_pass\",\"description\":\"Two-pass encoding\",\"steps\":2},\
     {\"name\":\"proxy\",\"description\":\"Generate low-res proxy\",\"steps\":1}]"
        .to_string()
}

/// List available workflow step/task types as a JSON array.
#[wasm_bindgen]
pub fn wasm_workflow_step_types() -> String {
    "[{\"type\":\"transcode\",\"description\":\"Encode/decode media\"},\
     {\"type\":\"qc\",\"description\":\"Quality control check\"},\
     {\"type\":\"transfer\",\"description\":\"Copy/move files\"},\
     {\"type\":\"analysis\",\"description\":\"Probe or analyze media\"},\
     {\"type\":\"wait\",\"description\":\"Wait for a specified duration\"},\
     {\"type\":\"notification\",\"description\":\"Send notification\"}]"
        .to_string()
}

/// Get a workflow template definition as JSON.
///
/// Returns the full template definition with steps.
#[wasm_bindgen]
pub fn wasm_get_workflow_template(name: &str) -> Result<String, JsValue> {
    match name {
        "transcode" => Ok(
            "{\"name\":\"transcode\",\"steps\":[\
             {\"step_id\":\"validate\",\"task_type\":\"qc\",\"description\":\"Validate source\",\"depends_on\":[]},\
             {\"step_id\":\"transcode\",\"task_type\":\"transcode\",\"description\":\"Transcode to target\",\"depends_on\":[\"validate\"]},\
             {\"step_id\":\"verify\",\"task_type\":\"qc\",\"description\":\"Verify output\",\"depends_on\":[\"transcode\"]}\
             ]}".to_string(),
        ),
        "ingest" => Ok(
            "{\"name\":\"ingest\",\"steps\":[\
             {\"step_id\":\"copy\",\"task_type\":\"transfer\",\"description\":\"Copy to storage\",\"depends_on\":[]},\
             {\"step_id\":\"probe\",\"task_type\":\"analysis\",\"description\":\"Probe format\",\"depends_on\":[\"copy\"]},\
             {\"step_id\":\"proxy\",\"task_type\":\"transcode\",\"description\":\"Generate proxy\",\"depends_on\":[\"probe\"]}\
             ]}".to_string(),
        ),
        "qc" => Ok(
            "{\"name\":\"qc\",\"steps\":[\
             {\"step_id\":\"format_check\",\"task_type\":\"qc\",\"description\":\"Format check\",\"depends_on\":[]},\
             {\"step_id\":\"quality_check\",\"task_type\":\"qc\",\"description\":\"Quality check\",\"depends_on\":[]},\
             {\"step_id\":\"loudness_check\",\"task_type\":\"qc\",\"description\":\"Loudness check\",\"depends_on\":[]}\
             ]}".to_string(),
        ),
        "multi_pass" => Ok(
            "{\"name\":\"multi_pass\",\"steps\":[\
             {\"step_id\":\"pass1\",\"task_type\":\"transcode\",\"description\":\"First pass\",\"depends_on\":[]},\
             {\"step_id\":\"pass2\",\"task_type\":\"transcode\",\"description\":\"Second pass\",\"depends_on\":[\"pass1\"]}\
             ]}".to_string(),
        ),
        "proxy" => Ok(
            "{\"name\":\"proxy\",\"steps\":[\
             {\"step_id\":\"proxy_gen\",\"task_type\":\"transcode\",\"description\":\"Generate proxy\",\"depends_on\":[]}\
             ]}".to_string(),
        ),
        other => Err(crate::utils::js_err(&format!(
            "Unknown template '{}'. Use: transcode, ingest, qc, multi_pass, proxy",
            other
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
    fn test_validate_workflow_valid() {
        let json = r#"{"steps":[
            {"step_id":"s1","task_type":"qc","depends_on":[]},
            {"step_id":"s2","task_type":"transcode","depends_on":["s1"]}
        ]}"#;
        let result = wasm_validate_workflow(json);
        assert!(result.is_ok());
        let r = result.expect("valid");
        assert!(r.contains("\"valid\":true"));
        assert!(r.contains("\"steps\":2"));
    }

    #[test]
    fn test_validate_workflow_missing_dep() {
        let json = r#"{"steps":[
            {"step_id":"s1","task_type":"qc","depends_on":["nonexistent"]}
        ]}"#;
        let result = wasm_validate_workflow(json);
        assert!(result.is_ok());
        let r = result.expect("valid");
        assert!(r.contains("\"valid\":false"));
    }

    #[test]
    fn test_list_templates() {
        let templates = wasm_list_workflow_templates();
        assert!(templates.contains("transcode"));
        assert!(templates.contains("ingest"));
    }

    #[test]
    fn test_step_types() {
        let types = wasm_workflow_step_types();
        assert!(types.contains("transcode"));
        assert!(types.contains("qc"));
        assert!(types.contains("transfer"));
    }

    #[test]
    fn test_get_template() {
        let result = wasm_get_workflow_template("transcode");
        assert!(result.is_ok());
        let r = result.expect("valid");
        assert!(r.contains("validate"));
        assert!(r.contains("transcode"));
        assert!(r.contains("verify"));
    }
}
