//! WebAssembly bindings for timeline editing.
//!
//! Provides `WasmTimeline` for creating and managing multi-track timelines
//! in the browser, plus standalone utility functions.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// WasmTimeline
// ---------------------------------------------------------------------------

/// Browser-side multi-track timeline editor.
///
/// Usage:
/// 1. Create a timeline with resolution and frame rate.
/// 2. Add tracks (video/audio/subtitle).
/// 3. Add clips to tracks with source info and timing.
/// 4. Query timeline info or export to JSON.
#[wasm_bindgen]
pub struct WasmTimeline {
    name: String,
    fps: f64,
    width: u32,
    height: u32,
    tracks: Vec<TrackData>,
    next_clip_id: u64,
}

struct TrackData {
    name: String,
    track_type: String,
    muted: bool,
    locked: bool,
    clips: Vec<ClipData>,
}

struct ClipData {
    id: u64,
    source: String,
    start_time: f64,
    duration: f64,
    in_point: f64,
    out_point: f64,
    speed: f64,
}

#[wasm_bindgen]
impl WasmTimeline {
    /// Create a new timeline.
    ///
    /// # Arguments
    /// * `fps` - Frame rate (e.g., 24.0, 30.0, 60.0).
    /// * `width` - Video width in pixels.
    /// * `height` - Video height in pixels.
    ///
    /// # Errors
    /// Returns an error if parameters are invalid.
    #[wasm_bindgen(constructor)]
    pub fn new(fps: f64, width: u32, height: u32) -> Result<WasmTimeline, JsValue> {
        if fps <= 0.0 {
            return Err(crate::utils::js_err("fps must be > 0"));
        }
        if width == 0 || height == 0 {
            return Err(crate::utils::js_err("width and height must be > 0"));
        }

        Ok(Self {
            name: "Untitled".to_string(),
            fps,
            width,
            height,
            tracks: Vec::new(),
            next_clip_id: 1,
        })
    }

    /// Set the timeline name.
    pub fn set_name(&mut self, name: &str) {
        self.name = name.to_string();
    }

    /// Get the timeline name.
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// Add a new track. Returns the track index.
    ///
    /// `track_type` must be "video", "audio", or "subtitle".
    pub fn add_track(&mut self, name: &str, track_type: &str) -> Result<u32, JsValue> {
        match track_type {
            "video" | "audio" | "subtitle" => {}
            _ => {
                return Err(crate::utils::js_err(&format!(
                    "Invalid track type '{}'. Use: video, audio, subtitle",
                    track_type
                )));
            }
        }

        let index = self.tracks.len() as u32;
        self.tracks.push(TrackData {
            name: name.to_string(),
            track_type: track_type.to_string(),
            muted: false,
            locked: false,
            clips: Vec::new(),
        });
        Ok(index)
    }

    /// Add a clip to a track.
    ///
    /// `source_json` must be a JSON string:
    /// ```json
    /// {"source": "file.mp4", "start_time": 0.0, "duration": 5.0, "in_point": 0.0, "out_point": 5.0, "speed": 1.0}
    /// ```
    ///
    /// Returns the clip ID.
    pub fn add_clip(&mut self, track_index: u32, source_json: &str) -> Result<u64, JsValue> {
        let idx = track_index as usize;
        if idx >= self.tracks.len() {
            return Err(crate::utils::js_err(&format!(
                "Track index {} out of range (0..{})",
                track_index,
                self.tracks.len()
            )));
        }

        if self.tracks[idx].locked {
            return Err(crate::utils::js_err("Track is locked"));
        }

        let parsed: serde_json::Value = serde_json::from_str(source_json)
            .map_err(|e| crate::utils::js_err(&format!("Invalid JSON: {e}")))?;

        let source = parsed["source"].as_str().unwrap_or("").to_string();
        let start_time = parsed["start_time"].as_f64().unwrap_or(0.0);
        let duration = parsed["duration"].as_f64().unwrap_or(10.0);
        let in_point = parsed["in_point"].as_f64().unwrap_or(0.0);
        let out_point = parsed["out_point"].as_f64().unwrap_or(duration);
        let speed = parsed["speed"].as_f64().unwrap_or(1.0);

        if duration <= 0.0 {
            return Err(crate::utils::js_err("duration must be > 0"));
        }
        if speed <= 0.0 {
            return Err(crate::utils::js_err("speed must be > 0"));
        }

        let clip_id = self.next_clip_id;
        self.next_clip_id += 1;

        self.tracks[idx].clips.push(ClipData {
            id: clip_id,
            source,
            start_time,
            duration,
            in_point,
            out_point,
            speed,
        });

        // Sort by start time
        self.tracks[idx].clips.sort_by(|a, b| {
            a.start_time
                .partial_cmp(&b.start_time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(clip_id)
    }

    /// Remove a clip from a track by clip index.
    pub fn remove_clip(&mut self, track_index: u32, clip_index: u32) -> Result<(), JsValue> {
        let tidx = track_index as usize;
        let cidx = clip_index as usize;

        if tidx >= self.tracks.len() {
            return Err(crate::utils::js_err(&format!(
                "Track index {} out of range",
                track_index
            )));
        }

        if self.tracks[tidx].locked {
            return Err(crate::utils::js_err("Track is locked"));
        }

        if cidx >= self.tracks[tidx].clips.len() {
            return Err(crate::utils::js_err(&format!(
                "Clip index {} out of range",
                clip_index
            )));
        }

        self.tracks[tidx].clips.remove(cidx);
        Ok(())
    }

    /// Get total timeline duration in seconds.
    pub fn duration(&self) -> f64 {
        let mut max_end = 0.0_f64;
        for track in &self.tracks {
            for clip in &track.clips {
                let end = clip.start_time + clip.duration;
                if end > max_end {
                    max_end = end;
                }
            }
        }
        max_end
    }

    /// Get the total number of tracks.
    pub fn track_count(&self) -> u32 {
        self.tracks.len() as u32
    }

    /// Get the total number of clips across all tracks.
    pub fn clip_count(&self) -> u32 {
        self.tracks.iter().map(|t| t.clips.len() as u32).sum()
    }

    /// Get timeline info as a JSON string.
    pub fn info(&self) -> String {
        format!(
            "{{\"name\":\"{}\",\"fps\":{},\"width\":{},\"height\":{},\
             \"duration\":{:.4},\"tracks\":{},\"clips\":{}}}",
            self.name,
            self.fps,
            self.width,
            self.height,
            self.duration(),
            self.tracks.len(),
            self.clip_count(),
        )
    }

    /// Serialize the entire timeline to a JSON string.
    pub fn to_json(&self) -> String {
        let tracks_json: Vec<String> = self
            .tracks
            .iter()
            .map(|t| {
                let clips_json: Vec<String> = t
                    .clips
                    .iter()
                    .map(|c| {
                        format!(
                            "{{\"id\":{},\"source\":\"{}\",\"start_time\":{:.4},\
                             \"duration\":{:.4},\"in_point\":{:.4},\"out_point\":{:.4},\
                             \"speed\":{}}}",
                            c.id,
                            c.source,
                            c.start_time,
                            c.duration,
                            c.in_point,
                            c.out_point,
                            c.speed,
                        )
                    })
                    .collect();
                format!(
                    "{{\"name\":\"{}\",\"track_type\":\"{}\",\"muted\":{},\"locked\":{},\
                     \"clips\":[{}]}}",
                    t.name,
                    t.track_type,
                    t.muted,
                    t.locked,
                    clips_json.join(","),
                )
            })
            .collect();

        format!(
            "{{\"name\":\"{}\",\"fps\":{},\"width\":{},\"height\":{},\"tracks\":[{}]}}",
            self.name,
            self.fps,
            self.width,
            self.height,
            tracks_json.join(","),
        )
    }

    /// Create a timeline from a JSON string.
    pub fn from_json(json: &str) -> Result<WasmTimeline, JsValue> {
        let data: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| crate::utils::js_err(&format!("Invalid JSON: {e}")))?;

        let fps = data["fps"].as_f64().unwrap_or(24.0);
        let width = data["width"].as_u64().unwrap_or(1920) as u32;
        let height = data["height"].as_u64().unwrap_or(1080) as u32;

        let mut tl = WasmTimeline::new(fps, width, height)?;
        tl.name = data["name"].as_str().unwrap_or("Untitled").to_string();

        if let Some(tracks) = data["tracks"].as_array() {
            for track_data in tracks {
                let tt = track_data["track_type"].as_str().unwrap_or("video");
                let tn = track_data["name"].as_str().unwrap_or("Track");
                let tidx = tl.add_track(tn, tt)?;

                if let Some(muted) = track_data["muted"].as_bool() {
                    tl.tracks[tidx as usize].muted = muted;
                }
                if let Some(locked) = track_data["locked"].as_bool() {
                    tl.tracks[tidx as usize].locked = locked;
                }

                if let Some(clips) = track_data["clips"].as_array() {
                    for clip_data in clips {
                        let clip_id = tl.next_clip_id;
                        tl.next_clip_id += 1;
                        tl.tracks[tidx as usize].clips.push(ClipData {
                            id: clip_id,
                            source: clip_data["source"].as_str().unwrap_or("").to_string(),
                            start_time: clip_data["start_time"].as_f64().unwrap_or(0.0),
                            duration: clip_data["duration"].as_f64().unwrap_or(0.0),
                            in_point: clip_data["in_point"].as_f64().unwrap_or(0.0),
                            out_point: clip_data["out_point"].as_f64().unwrap_or(0.0),
                            speed: clip_data["speed"].as_f64().unwrap_or(1.0),
                        });
                    }
                }
            }
        }

        Ok(tl)
    }

    /// Mute or unmute a track.
    pub fn set_track_muted(&mut self, track_index: u32, muted: bool) -> Result<(), JsValue> {
        let idx = track_index as usize;
        if idx >= self.tracks.len() {
            return Err(crate::utils::js_err("Track index out of range"));
        }
        self.tracks[idx].muted = muted;
        Ok(())
    }

    /// Lock or unlock a track.
    pub fn set_track_locked(&mut self, track_index: u32, locked: bool) -> Result<(), JsValue> {
        let idx = track_index as usize;
        if idx >= self.tracks.len() {
            return Err(crate::utils::js_err("Track index out of range"));
        }
        self.tracks[idx].locked = locked;
        Ok(())
    }

    /// Reset the timeline, clearing all tracks and clips.
    pub fn reset(&mut self) {
        self.tracks.clear();
        self.next_clip_id = 1;
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Create a new timeline instance.
#[wasm_bindgen]
pub fn wasm_create_timeline(fps: f64, width: u32, height: u32) -> Result<WasmTimeline, JsValue> {
    WasmTimeline::new(fps, width, height)
}

/// Validate a timeline JSON string and return a report.
///
/// Returns a JSON string with validation results:
/// ```json
/// {"valid": true, "tracks": 2, "clips": 5, "warnings": []}
/// ```
#[wasm_bindgen]
pub fn wasm_validate_timeline(json: &str) -> Result<String, JsValue> {
    let data: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| crate::utils::js_err(&format!("Invalid JSON: {e}")))?;

    let mut warnings: Vec<String> = Vec::new();

    let fps = data["fps"].as_f64().unwrap_or(0.0);
    if fps <= 0.0 {
        warnings.push("fps is missing or <= 0".to_string());
    }

    let width = data["width"].as_u64().unwrap_or(0);
    let height = data["height"].as_u64().unwrap_or(0);
    if width == 0 || height == 0 {
        warnings.push("width or height is missing or 0".to_string());
    }

    let mut track_count = 0u32;
    let mut clip_count = 0u32;

    if let Some(tracks) = data["tracks"].as_array() {
        track_count = tracks.len() as u32;
        for (i, track) in tracks.iter().enumerate() {
            let tt = track["track_type"].as_str().unwrap_or("");
            if !matches!(tt, "video" | "audio" | "subtitle") {
                warnings.push(format!("Track {} has invalid type '{}'", i, tt));
            }
            if let Some(clips) = track["clips"].as_array() {
                clip_count += clips.len() as u32;
                for (j, clip) in clips.iter().enumerate() {
                    let dur = clip["duration"].as_f64().unwrap_or(0.0);
                    if dur <= 0.0 {
                        warnings.push(format!("Track {} clip {} has invalid duration", i, j));
                    }
                }
            }
        }
    } else {
        warnings.push("No tracks array found".to_string());
    }

    let valid = warnings.is_empty();
    let warnings_json: Vec<String> = warnings.iter().map(|w| format!("\"{}\"", w)).collect();

    Ok(format!(
        "{{\"valid\":{},\"tracks\":{},\"clips\":{},\"warnings\":[{}]}}",
        valid,
        track_count,
        clip_count,
        warnings_json.join(","),
    ))
}

/// List supported timeline export/import formats as a JSON array.
#[wasm_bindgen]
pub fn wasm_timeline_formats() -> String {
    "[\"json\",\"edl\",\"fcpxml\",\"otio\"]".to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeline_creation() {
        let tl = WasmTimeline::new(24.0, 1920, 1080);
        assert!(tl.is_ok());
        let tl = tl.expect("should succeed");
        assert_eq!(tl.track_count(), 0);
        assert!((tl.duration() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_timeline_invalid_params() {
        assert!(WasmTimeline::new(0.0, 1920, 1080).is_err());
        assert!(WasmTimeline::new(24.0, 0, 1080).is_err());
        assert!(WasmTimeline::new(24.0, 1920, 0).is_err());
    }

    #[test]
    fn test_add_track() {
        let mut tl = WasmTimeline::new(30.0, 1920, 1080).expect("valid");
        let idx = tl.add_track("V1", "video");
        assert!(idx.is_ok());
        assert_eq!(idx.expect("valid"), 0);
        assert_eq!(tl.track_count(), 1);
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_add_track_invalid_type() {
        let mut tl = WasmTimeline::new(30.0, 1920, 1080).expect("valid");
        assert!(tl.add_track("X", "invalid").is_err());
    }

    #[test]
    fn test_add_and_remove_clip() {
        let mut tl = WasmTimeline::new(30.0, 1920, 1080).expect("valid");
        tl.add_track("V1", "video").expect("valid");

        let clip_json = r#"{"source":"test.mp4","start_time":0,"duration":5,"speed":1}"#;
        let clip_id = tl.add_clip(0, clip_json);
        assert!(clip_id.is_ok());
        assert_eq!(tl.clip_count(), 1);
        assert!((tl.duration() - 5.0).abs() < f64::EPSILON);

        let remove = tl.remove_clip(0, 0);
        assert!(remove.is_ok());
        assert_eq!(tl.clip_count(), 0);
    }

    #[test]
    fn test_to_json_and_from_json() {
        let mut tl = WasmTimeline::new(24.0, 1920, 1080).expect("valid");
        tl.set_name("Test");
        tl.add_track("V1", "video").expect("valid");
        let clip_json = r#"{"source":"clip.mp4","start_time":0,"duration":3,"speed":1}"#;
        tl.add_clip(0, clip_json).expect("valid");

        let json = tl.to_json();
        let tl2 = WasmTimeline::from_json(&json);
        assert!(tl2.is_ok());
        let tl2 = tl2.expect("valid");
        assert_eq!(tl2.name(), "Test");
        assert_eq!(tl2.track_count(), 1);
        assert_eq!(tl2.clip_count(), 1);
    }

    #[test]
    fn test_validate_timeline_valid() {
        let json = r#"{"fps":24,"width":1920,"height":1080,"tracks":[{"track_type":"video","clips":[{"duration":5}]}]}"#;
        let result = wasm_validate_timeline(json);
        assert!(result.is_ok());
        let r = result.expect("valid");
        assert!(r.contains("\"valid\":true"));
    }

    #[test]
    fn test_validate_timeline_invalid() {
        let json = r#"{"fps":0,"width":0,"height":0}"#;
        let result = wasm_validate_timeline(json);
        assert!(result.is_ok());
        let r = result.expect("valid");
        assert!(r.contains("\"valid\":false"));
    }

    #[test]
    fn test_wasm_create_timeline() {
        let tl = wasm_create_timeline(30.0, 1920, 1080);
        assert!(tl.is_ok());
    }

    #[test]
    fn test_timeline_formats() {
        let formats = wasm_timeline_formats();
        assert!(formats.contains("json"));
        assert!(formats.contains("edl"));
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_track_mute_lock() {
        let mut tl = WasmTimeline::new(24.0, 1920, 1080).expect("valid");
        tl.add_track("V1", "video").expect("valid");
        tl.set_track_muted(0, true).expect("valid");
        tl.set_track_locked(0, true).expect("valid");

        // Locked track should reject clips
        let clip_json = r#"{"source":"test.mp4","duration":5,"speed":1}"#;
        assert!(tl.add_clip(0, clip_json).is_err());
    }
}
