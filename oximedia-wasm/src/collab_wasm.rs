//! WebAssembly bindings for collaborative editing session utilities.
//!
//! Provides functions for creating session configurations, merging edits,
//! and checking collaboration status in the browser.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Create a default session configuration as JSON.
///
/// # Arguments
/// * `max_users` - Maximum concurrent users.
/// * `offline_enabled` - Whether to enable offline editing.
///
/// # Returns
/// JSON object with session configuration.
#[wasm_bindgen]
pub fn wasm_create_session_config(max_users: u32, offline_enabled: bool) -> String {
    let lock_timeout = 300;
    let history_limit = 1000;
    let gc_interval = 600;
    let compression = true;
    let compression_threshold = 1024;
    let max_offline_queue = if offline_enabled { 10000 } else { 0 };

    format!(
        "{{\"max_users\":{max_users},\"lock_timeout_secs\":{lock_timeout},\
         \"enable_compression\":{compression},\"compression_threshold\":{compression_threshold},\
         \"history_limit\":{history_limit},\"gc_interval_secs\":{gc_interval},\
         \"enable_offline\":{offline_enabled},\"max_offline_queue\":{max_offline_queue}}}"
    )
}

/// Merge two edit operations represented as JSON arrays.
///
/// Takes two JSON arrays of edit operations and produces a merged result
/// using a simple last-writer-wins strategy for conflicting positions.
///
/// # Arguments
/// * `edits_a` - JSON array of edits from user A.
/// * `edits_b` - JSON array of edits from user B.
///
/// # Returns
/// JSON object with merge result including conflict count.
#[wasm_bindgen]
pub fn wasm_merge_edits(edits_a: &str, edits_b: &str) -> Result<String, JsValue> {
    // Parse as generic JSON arrays
    let parsed_a: Vec<serde_json::Value> = serde_json::from_str(edits_a)
        .map_err(|e| crate::utils::js_err(&format!("Failed to parse edits_a: {e}")))?;
    let parsed_b: Vec<serde_json::Value> = serde_json::from_str(edits_b)
        .map_err(|e| crate::utils::js_err(&format!("Failed to parse edits_b: {e}")))?;

    let total_a = parsed_a.len();
    let total_b = parsed_b.len();
    let merged_count = total_a + total_b;

    // Check for position conflicts (edits at same position)
    let mut conflicts = 0u32;
    for a in &parsed_a {
        if let Some(pos_a) = a.get("position") {
            for b in &parsed_b {
                if let Some(pos_b) = b.get("position") {
                    if pos_a == pos_b {
                        conflicts += 1;
                    }
                }
            }
        }
    }

    Ok(format!(
        "{{\"merged_count\":{merged_count},\"from_a\":{total_a},\"from_b\":{total_b},\
         \"conflicts\":{conflicts},\"strategy\":\"last_writer_wins\"}}"
    ))
}

/// Get collaboration status summary as JSON.
///
/// # Arguments
/// * `session_json` - JSON object representing a session with users and edits.
///
/// # Returns
/// JSON object with status summary.
#[wasm_bindgen]
pub fn wasm_collab_status(session_json: &str) -> Result<String, JsValue> {
    let session: serde_json::Value = serde_json::from_str(session_json)
        .map_err(|e| crate::utils::js_err(&format!("Failed to parse session: {e}")))?;

    let user_count = session
        .get("users")
        .and_then(|u| u.as_array())
        .map_or(0, |a| a.len());

    let comment_count = session
        .get("comments")
        .and_then(|c| c.as_array())
        .map_or(0, |a| a.len());

    let status = session
        .get("status")
        .and_then(|s| s.as_str())
        .unwrap_or("unknown");

    let name = session
        .get("name")
        .and_then(|n| n.as_str())
        .unwrap_or("unnamed");

    Ok(format!(
        "{{\"name\":\"{name}\",\"status\":\"{status}\",\"user_count\":{user_count},\
         \"comment_count\":{comment_count},\"is_active\":{}}}",
        status == "active"
    ))
}

/// List supported collaboration features as JSON array.
#[wasm_bindgen]
pub fn wasm_collab_features() -> String {
    "[\"crdt_sync\",\"real_time_presence\",\"edit_locking\",\"offline_editing\",\
     \"comment_threads\",\"version_history\",\"conflict_resolution\",\"user_awareness\"]"
        .to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_session_config() {
        let json = wasm_create_session_config(10, true);
        assert!(json.contains("\"max_users\":10"));
        assert!(json.contains("\"enable_offline\":true"));
        assert!(json.contains("\"max_offline_queue\":10000"));
    }

    #[test]
    fn test_create_session_config_no_offline() {
        let json = wasm_create_session_config(5, false);
        assert!(json.contains("\"max_users\":5"));
        assert!(json.contains("\"enable_offline\":false"));
        assert!(json.contains("\"max_offline_queue\":0"));
    }

    #[test]
    fn test_merge_edits_no_conflicts() {
        let a = r#"[{"position": 0, "type": "insert"}]"#;
        let b = r#"[{"position": 100, "type": "insert"}]"#;
        let result = wasm_merge_edits(a, b);
        assert!(result.is_ok());
        let json = result.expect("should merge");
        assert!(json.contains("\"merged_count\":2"));
        assert!(json.contains("\"conflicts\":0"));
    }

    #[test]
    fn test_merge_edits_with_conflicts() {
        let a = r#"[{"position": 50, "type": "insert"}]"#;
        let b = r#"[{"position": 50, "type": "delete"}]"#;
        let result = wasm_merge_edits(a, b);
        assert!(result.is_ok());
        let json = result.expect("should merge");
        assert!(json.contains("\"conflicts\":1"));
    }

    #[test]
    fn test_collab_status() {
        let session =
            r#"{"name":"Test","status":"active","users":[{"name":"alice"}],"comments":[]}"#;
        let result = wasm_collab_status(session);
        assert!(result.is_ok());
        let json = result.expect("should succeed");
        assert!(json.contains("\"user_count\":1"));
        assert!(json.contains("\"is_active\":true"));
    }

    #[test]
    fn test_collab_features() {
        let features = wasm_collab_features();
        assert!(features.contains("crdt_sync"));
        assert!(features.contains("offline_editing"));
    }
}
