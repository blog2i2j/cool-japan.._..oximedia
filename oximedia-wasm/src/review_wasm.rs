//! WebAssembly bindings for review and approval workflow utilities.
//!
//! Provides functions for creating annotations, exporting review data,
//! and querying supported review formats in the browser.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Create a frame-accurate annotation as JSON.
///
/// # Arguments
/// * `author` - Author name.
/// * `message` - Annotation text.
/// * `annotation_type` - One of: "general", "issue", "suggestion", "question".
/// * `frame` - Frame number (0 = no specific frame).
///
/// # Returns
/// JSON object with annotation data.
///
/// # Errors
/// Returns an error if annotation_type is invalid.
#[wasm_bindgen]
pub fn wasm_create_annotation(
    author: &str,
    message: &str,
    annotation_type: &str,
    frame: u64,
) -> Result<String, JsValue> {
    let valid_type = match annotation_type {
        "general" | "issue" | "suggestion" | "question" | "approval" | "rejection" => {
            annotation_type
        }
        _ => {
            return Err(crate::utils::js_err(
                "Invalid annotation type. Use general, issue, suggestion, question, approval, or rejection",
            ))
        }
    };

    let frame_str = if frame > 0 {
        format!(",\"frame\":{frame}")
    } else {
        String::new()
    };

    Ok(format!(
        "{{\"author\":\"{author}\",\"message\":\"{}\",\"type\":\"{valid_type}\",\
         \"resolved\":false{frame_str}}}",
        message.replace('"', "\\\"")
    ))
}

/// Export a set of annotations as a formatted report.
///
/// # Arguments
/// * `annotations_json` - JSON array of annotation objects.
/// * `format` - Export format: "json", "csv".
///
/// # Returns
/// Exported data as string.
///
/// # Errors
/// Returns an error if parsing fails.
#[wasm_bindgen]
pub fn wasm_export_annotations(annotations_json: &str, format: &str) -> Result<String, JsValue> {
    let annotations: Vec<serde_json::Value> = serde_json::from_str(annotations_json)
        .map_err(|e| crate::utils::js_err(&format!("Failed to parse annotations: {e}")))?;

    match format {
        "json" => {
            // Re-serialize pretty
            serde_json::to_string_pretty(&annotations)
                .map_err(|e| crate::utils::js_err(&format!("Serialize error: {e}")))
        }
        "csv" => {
            let mut csv = String::from("author,type,message,frame,resolved\n");
            for ann in &annotations {
                let author = ann.get("author").and_then(|a| a.as_str()).unwrap_or("");
                let ann_type = ann
                    .get("type")
                    .and_then(|t| t.as_str())
                    .unwrap_or("general");
                let message = ann
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("")
                    .replace(',', ";");
                let frame = ann
                    .get("frame")
                    .and_then(|f| f.as_u64())
                    .map_or(String::new(), |f| f.to_string());
                let resolved = ann
                    .get("resolved")
                    .and_then(|r| r.as_bool())
                    .unwrap_or(false);
                csv.push_str(&format!(
                    "{author},{ann_type},{message},{frame},{resolved}\n"
                ));
            }
            Ok(csv)
        }
        _ => Err(crate::utils::js_err(&format!(
            "Unsupported export format: {format}. Use json or csv"
        ))),
    }
}

/// List supported review formats and capabilities as JSON.
#[wasm_bindgen]
pub fn wasm_review_formats() -> String {
    "[{\"format\":\"json\",\"description\":\"Full annotation data with metadata\"},\
      {\"format\":\"csv\",\"description\":\"Tabular format for spreadsheet import\"},\
      {\"format\":\"pdf\",\"description\":\"Formatted report for sharing\"}]"
        .to_string()
}

/// List supported annotation types as JSON array.
#[wasm_bindgen]
pub fn wasm_annotation_types() -> String {
    "[{\"type\":\"general\",\"label\":\"General Feedback\",\"color\":\"#4ECDC4\"},\
      {\"type\":\"issue\",\"label\":\"Issue\",\"color\":\"#FF6B6B\"},\
      {\"type\":\"suggestion\",\"label\":\"Suggestion\",\"color\":\"#45B7D1\"},\
      {\"type\":\"question\",\"label\":\"Question\",\"color\":\"#FFA07A\"},\
      {\"type\":\"approval\",\"label\":\"Approval\",\"color\":\"#98D8C8\"},\
      {\"type\":\"rejection\",\"label\":\"Rejection\",\"color\":\"#BB8FCE\"}]"
        .to_string()
}

/// List supported workflow types as JSON array.
#[wasm_bindgen]
pub fn wasm_review_workflows() -> String {
    "[{\"type\":\"simple\",\"description\":\"Creator -> Reviewer -> Approved\"},\
      {\"type\":\"multi-stage\",\"description\":\"Multiple sequential review stages\"},\
      {\"type\":\"parallel\",\"description\":\"Multiple reviewers simultaneously\"},\
      {\"type\":\"sequential\",\"description\":\"One reviewer after another\"}]"
        .to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_annotation() {
        let result = wasm_create_annotation("bob", "Fix color", "issue", 1500);
        assert!(result.is_ok());
        let json = result.expect("should create");
        assert!(json.contains("\"author\":\"bob\""));
        assert!(json.contains("\"type\":\"issue\""));
        assert!(json.contains("\"frame\":1500"));
    }

    #[test]
    fn test_create_annotation_no_frame() {
        let result = wasm_create_annotation("alice", "Looks good", "general", 0);
        assert!(result.is_ok());
        let json = result.expect("should create");
        assert!(!json.contains("\"frame\""));
    }

    #[test]
    fn test_create_annotation_invalid_type() {
        let result = wasm_create_annotation("bob", "Test", "invalid", 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_export_annotations_json() {
        let annotations = r#"[{"author":"bob","type":"issue","message":"Fix this","frame":100,"resolved":false}]"#;
        let result = wasm_export_annotations(annotations, "json");
        assert!(result.is_ok());
        let json = result.expect("should export");
        assert!(json.contains("bob"));
    }

    #[test]
    fn test_export_annotations_csv() {
        let annotations =
            r#"[{"author":"alice","type":"general","message":"Nice work","resolved":true}]"#;
        let result = wasm_export_annotations(annotations, "csv");
        assert!(result.is_ok());
        let csv = result.expect("should export");
        assert!(csv.contains("alice,general,Nice work"));
    }

    #[test]
    fn test_review_formats() {
        let json = wasm_review_formats();
        assert!(json.contains("json"));
        assert!(json.contains("csv"));
    }

    #[test]
    fn test_annotation_types() {
        let json = wasm_annotation_types();
        assert!(json.contains("issue"));
        assert!(json.contains("suggestion"));
    }
}
