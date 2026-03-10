//! WebAssembly bindings for playout schedule management.
//!
//! Provides `WasmPlayoutScheduler` for building and validating playout
//! schedules in the browser, plus standalone utility functions.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// WasmPlayoutScheduler
// ---------------------------------------------------------------------------

/// Browser-side playout schedule builder and validator.
#[wasm_bindgen]
pub struct WasmPlayoutScheduler {
    channel: String,
    video_format: String,
    items: Vec<PlayoutItemData>,
    next_id: u64,
}

struct PlayoutItemData {
    id: u64,
    title: String,
    source_path: String,
    scheduled_time: String,
    duration_secs: f64,
    item_type: String,
}

#[wasm_bindgen]
impl WasmPlayoutScheduler {
    /// Create a new playout scheduler.
    #[wasm_bindgen(constructor)]
    pub fn new(channel: &str, video_format: &str) -> Self {
        Self {
            channel: channel.to_string(),
            video_format: video_format.to_string(),
            items: Vec::new(),
            next_id: 1,
        }
    }

    /// Add an item from a JSON string. Returns the item ID.
    ///
    /// JSON format:
    /// ```json
    /// {"title": "News", "source_path": "news.mxf", "duration_secs": 1800, "scheduled_time": "18:00:00", "item_type": "programme"}
    /// ```
    pub fn add_item(&mut self, item_json: &str) -> Result<u64, JsValue> {
        let parsed: serde_json::Value = serde_json::from_str(item_json)
            .map_err(|e| crate::utils::js_err(&format!("Invalid JSON: {e}")))?;

        let title = parsed["title"].as_str().unwrap_or("Untitled").to_string();
        let source_path = parsed["source_path"].as_str().unwrap_or("").to_string();
        let duration_secs = parsed["duration_secs"].as_f64().unwrap_or(0.0);
        let scheduled_time = parsed["scheduled_time"]
            .as_str()
            .unwrap_or("00:00:00")
            .to_string();
        let item_type = parsed["item_type"]
            .as_str()
            .unwrap_or("programme")
            .to_string();

        if duration_secs <= 0.0 {
            return Err(crate::utils::js_err("duration_secs must be > 0"));
        }

        let id = self.next_id;
        self.next_id += 1;

        self.items.push(PlayoutItemData {
            id,
            title,
            source_path,
            scheduled_time,
            duration_secs,
            item_type,
        });

        Ok(id)
    }

    /// Remove an item by ID.
    pub fn remove_item(&mut self, item_id: u64) -> Result<(), JsValue> {
        let pos = self
            .items
            .iter()
            .position(|i| i.id == item_id)
            .ok_or_else(|| crate::utils::js_err(&format!("Item {} not found", item_id)))?;
        self.items.remove(pos);
        Ok(())
    }

    /// Get the number of items.
    pub fn item_count(&self) -> u32 {
        self.items.len() as u32
    }

    /// Get total duration in seconds.
    pub fn total_duration(&self) -> f64 {
        self.items.iter().map(|i| i.duration_secs).sum()
    }

    /// Get schedule info as JSON.
    pub fn info(&self) -> String {
        format!(
            "{{\"channel\":\"{}\",\"format\":\"{}\",\"items\":{},\"total_duration\":{:.2}}}",
            self.channel,
            self.video_format,
            self.items.len(),
            self.total_duration(),
        )
    }

    /// Serialize the entire schedule to JSON.
    pub fn to_json(&self) -> String {
        let items_json: Vec<String> = self
            .items
            .iter()
            .map(|i| {
                format!(
                    "{{\"id\":{},\"title\":\"{}\",\"source_path\":\"{}\",\
                     \"scheduled_time\":\"{}\",\"duration_secs\":{:.2},\"item_type\":\"{}\"}}",
                    i.id, i.title, i.source_path, i.scheduled_time, i.duration_secs, i.item_type,
                )
            })
            .collect();

        format!(
            "{{\"channel\":\"{}\",\"format\":\"{}\",\"items\":[{}]}}",
            self.channel,
            self.video_format,
            items_json.join(","),
        )
    }

    /// Clear all items.
    pub fn clear(&mut self) {
        self.items.clear();
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Validate a playout schedule JSON string.
///
/// Returns a JSON report:
/// ```json
/// {"valid": true, "items": 5, "warnings": [], "total_duration": 7200.0}
/// ```
#[wasm_bindgen]
pub fn wasm_validate_schedule(json: &str) -> Result<String, JsValue> {
    let data: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| crate::utils::js_err(&format!("Invalid JSON: {e}")))?;

    let mut warnings: Vec<String> = Vec::new();
    let mut item_count = 0u32;
    let mut total_duration = 0.0_f64;

    if let Some(items) = data["items"].as_array() {
        item_count = items.len() as u32;
        for (i, item) in items.iter().enumerate() {
            let dur = item["duration_secs"].as_f64().unwrap_or(0.0);
            if dur <= 0.0 {
                warnings.push(format!("Item {} has non-positive duration", i));
            }
            total_duration += dur;

            let source = item["source_path"].as_str().unwrap_or("");
            if source.is_empty() {
                warnings.push(format!("Item {} has empty source path", i));
            }
        }
    } else {
        warnings.push("No items array found".to_string());
    }

    let valid = warnings.is_empty();
    let warnings_json: Vec<String> = warnings.iter().map(|w| format!("\"{}\"", w)).collect();

    Ok(format!(
        "{{\"valid\":{},\"items\":{},\"total_duration\":{:.2},\"warnings\":[{}]}}",
        valid,
        item_count,
        total_duration,
        warnings_json.join(","),
    ))
}

/// Get the current playout status (stub for browser use).
///
/// Returns a JSON string with status information.
#[wasm_bindgen]
pub fn wasm_playout_status() -> String {
    "{\"state\":\"stopped\",\"current_item\":null,\"next_item\":null,\
     \"frames_played\":0,\"frames_dropped\":0}"
        .to_string()
}

/// List supported video formats for playout.
#[wasm_bindgen]
pub fn wasm_playout_formats() -> String {
    "[\"hd1080p25\",\"hd1080p30\",\"hd1080p50\",\"hd1080p60\",\
     \"hd1080i50\",\"uhd2160p25\",\"uhd2160p50\"]"
        .to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_creation() {
        let sched = WasmPlayoutScheduler::new("Ch1", "hd1080p25");
        assert_eq!(sched.item_count(), 0);
        assert!((sched.total_duration() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scheduler_add_item() {
        let mut sched = WasmPlayoutScheduler::new("Ch1", "hd1080p25");
        let json = r#"{"title":"News","source_path":"news.mxf","duration_secs":1800}"#;
        let id = sched.add_item(json);
        assert!(id.is_ok());
        assert_eq!(sched.item_count(), 1);
        assert!((sched.total_duration() - 1800.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_scheduler_remove_item() {
        let mut sched = WasmPlayoutScheduler::new("Ch1", "hd1080p25");
        let json = r#"{"title":"Test","source_path":"t.mxf","duration_secs":30}"#;
        let id = sched.add_item(json).expect("valid");
        assert_eq!(sched.item_count(), 1);

        sched.remove_item(id).expect("valid");
        assert_eq!(sched.item_count(), 0);
    }

    #[test]
    fn test_validate_schedule_valid() {
        let json = r#"{"items":[{"title":"A","source_path":"a.mxf","duration_secs":60}]}"#;
        let result = wasm_validate_schedule(json);
        assert!(result.is_ok());
        let r = result.expect("valid");
        assert!(r.contains("\"valid\":true"));
    }

    #[test]
    fn test_validate_schedule_invalid() {
        let json = r#"{"items":[{"title":"Bad","source_path":"","duration_secs":0}]}"#;
        let result = wasm_validate_schedule(json);
        assert!(result.is_ok());
        let r = result.expect("valid");
        assert!(r.contains("\"valid\":false"));
    }
}
