//! WebAssembly bindings for clip management utilities.
//!
//! Provides `WasmClipManager` for browser-side clip management and standalone
//! functions for creating and merging clips.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// WasmClipManager
// ---------------------------------------------------------------------------

/// Browser-side clip manager for organizing video clips.
#[wasm_bindgen]
pub struct WasmClipManager {
    clips: Vec<ClipEntry>,
    next_id: u32,
}

struct ClipEntry {
    id: u32,
    name: String,
    source: String,
    tc_in: String,
    tc_out: String,
    rating: u8,
    keywords: Vec<String>,
}

#[wasm_bindgen]
impl WasmClipManager {
    /// Create a new clip manager.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            clips: Vec::new(),
            next_id: 1,
        }
    }

    /// Add a clip to the manager.
    ///
    /// # Arguments
    /// * `name` - Clip name.
    /// * `source` - Source file path or URL.
    /// * `tc_in` - In-point timecode.
    /// * `tc_out` - Out-point timecode.
    /// * `rating` - Rating (0-5).
    ///
    /// # Returns
    /// The clip ID.
    pub fn add_clip(
        &mut self,
        name: &str,
        source: &str,
        tc_in: &str,
        tc_out: &str,
        rating: u8,
    ) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.clips.push(ClipEntry {
            id,
            name: name.to_string(),
            source: source.to_string(),
            tc_in: tc_in.to_string(),
            tc_out: tc_out.to_string(),
            rating: rating.min(5),
            keywords: Vec::new(),
        });
        id
    }

    /// Add a keyword to a clip.
    pub fn add_keyword(&mut self, clip_id: u32, keyword: &str) -> bool {
        if let Some(clip) = self.clips.iter_mut().find(|c| c.id == clip_id) {
            let kw = keyword.to_string();
            if !clip.keywords.contains(&kw) {
                clip.keywords.push(kw);
            }
            true
        } else {
            false
        }
    }

    /// Remove a clip by ID.
    pub fn remove_clip(&mut self, clip_id: u32) -> bool {
        let before = self.clips.len();
        self.clips.retain(|c| c.id != clip_id);
        self.clips.len() < before
    }

    /// Get all clips as JSON array.
    pub fn list_clips(&self) -> String {
        let entries: Vec<String> = self
            .clips
            .iter()
            .map(|c| {
                let kw_json: Vec<String> = c.keywords.iter().map(|k| format!("\"{k}\"")).collect();
                format!(
                    "{{\"id\":{},\"name\":\"{}\",\"source\":\"{}\",\"tc_in\":\"{}\",\
                     \"tc_out\":\"{}\",\"rating\":{},\"keywords\":[{}]}}",
                    c.id,
                    c.name,
                    c.source,
                    c.tc_in,
                    c.tc_out,
                    c.rating,
                    kw_json.join(",")
                )
            })
            .collect();
        format!("[{}]", entries.join(","))
    }

    /// Search clips by keyword or name.
    pub fn search(&self, query: &str) -> String {
        let query_lower = query.to_lowercase();
        let entries: Vec<String> = self
            .clips
            .iter()
            .filter(|c| {
                c.name.to_lowercase().contains(&query_lower)
                    || c.keywords
                        .iter()
                        .any(|k| k.to_lowercase().contains(&query_lower))
            })
            .map(|c| {
                format!(
                    "{{\"id\":{},\"name\":\"{}\",\"rating\":{}}}",
                    c.id, c.name, c.rating
                )
            })
            .collect();
        format!("[{}]", entries.join(","))
    }

    /// Get the clip count.
    pub fn clip_count(&self) -> u32 {
        self.clips.len() as u32
    }

    /// Reset the manager, clearing all clips.
    pub fn reset(&mut self) {
        self.clips.clear();
        self.next_id = 1;
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Create a clip descriptor as JSON.
///
/// # Arguments
/// * `name` - Clip name.
/// * `source` - Source file.
/// * `tc_in` - In timecode.
/// * `tc_out` - Out timecode.
/// * `rating` - Rating 0-5.
///
/// # Returns
/// JSON object with clip info.
#[wasm_bindgen]
pub fn wasm_create_clip(name: &str, source: &str, tc_in: &str, tc_out: &str, rating: u8) -> String {
    let r = rating.min(5);
    format!(
        "{{\"name\":\"{name}\",\"source\":\"{source}\",\"tc_in\":\"{tc_in}\",\
         \"tc_out\":\"{tc_out}\",\"rating\":{r}}}"
    )
}

/// Merge multiple clip descriptors into a sequence.
///
/// # Arguments
/// * `clips_json` - JSON array of clip objects.
/// * `sequence_name` - Name for the merged sequence.
///
/// # Returns
/// JSON object with merged sequence info.
#[wasm_bindgen]
pub fn wasm_merge_clips(clips_json: &str, sequence_name: &str) -> Result<String, JsValue> {
    let clips: Vec<serde_json::Value> = serde_json::from_str(clips_json)
        .map_err(|e| crate::utils::js_err(&format!("Failed to parse clips: {e}")))?;

    let clip_count = clips.len();

    // Collect names for the merged sequence
    let names: Vec<String> = clips
        .iter()
        .filter_map(|c| c.get("name").and_then(|n| n.as_str()).map(String::from))
        .collect();

    Ok(format!(
        "{{\"sequence_name\":\"{sequence_name}\",\"clip_count\":{clip_count},\
         \"clips\":[{}]}}",
        names
            .iter()
            .map(|n| format!("\"{n}\""))
            .collect::<Vec<_>>()
            .join(",")
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_manager_add_list() {
        let mut mgr = WasmClipManager::new();
        let id = mgr.add_clip("Take 1", "video.mov", "01:00:00:00", "01:00:30:00", 4);
        assert_eq!(id, 1);
        assert_eq!(mgr.clip_count(), 1);

        let json = mgr.list_clips();
        assert!(json.contains("Take 1"));
        assert!(json.contains("video.mov"));
    }

    #[test]
    fn test_clip_manager_remove() {
        let mut mgr = WasmClipManager::new();
        let id = mgr.add_clip("Test", "a.mov", "00:00:00:00", "00:00:10:00", 3);
        assert!(mgr.remove_clip(id));
        assert_eq!(mgr.clip_count(), 0);
        assert!(!mgr.remove_clip(999));
    }

    #[test]
    fn test_clip_manager_search() {
        let mut mgr = WasmClipManager::new();
        let id = mgr.add_clip("Interview A", "a.mov", "00:00:00:00", "00:01:00:00", 5);
        mgr.add_keyword(id, "interview");
        mgr.add_clip("B-Roll Sunset", "b.mov", "00:00:00:00", "00:00:30:00", 3);

        let results = mgr.search("interview");
        assert!(results.contains("Interview A"));
        assert!(!results.contains("B-Roll"));
    }

    #[test]
    fn test_create_clip() {
        let json = wasm_create_clip("Take 1", "video.mov", "01:00:00:00", "01:00:30:00", 4);
        assert!(json.contains("\"name\":\"Take 1\""));
        assert!(json.contains("\"rating\":4"));
    }

    #[test]
    fn test_merge_clips() {
        let clips = r#"[{"name":"Clip A"},{"name":"Clip B"}]"#;
        let result = wasm_merge_clips(clips, "Merged");
        assert!(result.is_ok());
        let json = result.expect("should merge");
        assert!(json.contains("\"sequence_name\":\"Merged\""));
        assert!(json.contains("\"clip_count\":2"));
    }
}
