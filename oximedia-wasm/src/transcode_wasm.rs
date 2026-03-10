//! WebAssembly bindings for transcoding configuration and validation.
//!
//! Actual file-based transcoding cannot run inside a WASM sandbox, so this
//! module focuses on configuration building, validation, preset/codec
//! enumeration, output-size estimation, and recommended-settings generation.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Preset & codec listing
// ---------------------------------------------------------------------------

/// Return a JSON array describing all built-in transcoding presets.
///
/// Each element: `{ name, description, video_codec, audio_codec, container, quality_mode }`.
#[wasm_bindgen]
pub fn wasm_list_presets() -> Result<String, JsValue> {
    let presets = serde_json::json!([
        {
            "name": "web_optimized",
            "description": "Web-optimized AV1/Opus encoding for low-latency streaming",
            "video_codec": "av1",
            "audio_codec": "opus",
            "container": "webm",
            "quality_mode": "balanced"
        },
        {
            "name": "archive_quality",
            "description": "High-fidelity archival encoding with lossless audio",
            "video_codec": "av1",
            "audio_codec": "flac",
            "container": "mkv",
            "quality_mode": "high"
        },
        {
            "name": "fast_preview",
            "description": "Quick preview transcode with minimal processing",
            "video_codec": "vp8",
            "audio_codec": "vorbis",
            "container": "webm",
            "quality_mode": "fast"
        },
        {
            "name": "broadcast_hd",
            "description": "Broadcast-grade 1080p encoding",
            "video_codec": "av1",
            "audio_codec": "opus",
            "container": "mkv",
            "quality_mode": "high"
        },
        {
            "name": "social_media",
            "description": "Optimized for social media with small file size",
            "video_codec": "vp9",
            "audio_codec": "opus",
            "container": "webm",
            "quality_mode": "balanced"
        },
        {
            "name": "youtube_1080p",
            "description": "YouTube-optimized 1080p VP9/Opus",
            "video_codec": "vp9",
            "audio_codec": "opus",
            "container": "webm",
            "quality_mode": "balanced"
        },
        {
            "name": "vimeo_hd",
            "description": "Vimeo-quality AV1 encoding",
            "video_codec": "av1",
            "audio_codec": "opus",
            "container": "webm",
            "quality_mode": "high"
        }
    ]);
    serde_json::to_string(&presets)
        .map_err(|e| crate::utils::js_err(&format!("JSON serialization error: {e}")))
}

/// Return a JSON array of supported codec metadata.
///
/// Each element: `{ name, type, description }`.
#[wasm_bindgen]
pub fn wasm_list_codecs() -> Result<String, JsValue> {
    let codecs = serde_json::json!([
        { "name": "av1",    "type": "video", "description": "AV1 (Alliance for Open Media) - patent-free next-gen codec" },
        { "name": "vp9",    "type": "video", "description": "VP9 (Google) - patent-free 4K-capable codec" },
        { "name": "vp8",    "type": "video", "description": "VP8 (Google) - patent-free web video codec" },
        { "name": "opus",   "type": "audio", "description": "Opus - patent-free low-latency audio codec" },
        { "name": "vorbis", "type": "audio", "description": "Vorbis - patent-free Ogg audio codec" },
        { "name": "flac",   "type": "audio", "description": "FLAC - patent-free lossless audio codec" },
        { "name": "pcm",    "type": "audio", "description": "PCM - uncompressed audio" }
    ]);
    serde_json::to_string(&codecs)
        .map_err(|e| crate::utils::js_err(&format!("JSON serialization error: {e}")))
}

// ---------------------------------------------------------------------------
// Config validation
// ---------------------------------------------------------------------------

/// Validate a JSON transcode configuration.
///
/// Input JSON should have optional fields: `video_codec`, `audio_codec`,
/// `width`, `height`, `bitrate_kbps`, `crf`, `output_format`.
///
/// Returns a JSON object `{ valid: bool, warnings: [...], errors: [...] }`.
#[wasm_bindgen]
pub fn wasm_validate_transcode_config(config_json: &str) -> Result<String, JsValue> {
    let parsed: serde_json::Value = serde_json::from_str(config_json)
        .map_err(|e| crate::utils::js_err(&format!("Invalid JSON: {e}")))?;

    let mut warnings: Vec<String> = Vec::new();
    let mut errors: Vec<String> = Vec::new();

    let supported_video = ["av1", "vp9", "vp8"];
    let supported_audio = ["opus", "vorbis", "flac", "pcm"];

    if let Some(vc) = parsed.get("video_codec").and_then(|v| v.as_str()) {
        if !supported_video.contains(&vc) {
            errors.push(format!("Unsupported video codec '{vc}'"));
        }
    }

    if let Some(ac) = parsed.get("audio_codec").and_then(|v| v.as_str()) {
        if !supported_audio.contains(&ac) {
            errors.push(format!("Unsupported audio codec '{ac}'"));
        }
    }

    if let Some(w) = parsed.get("width").and_then(|v| v.as_u64()) {
        if w == 0 || w > 7680 {
            warnings.push(format!("Unusual width {w} (expected 1-7680)"));
        }
    }

    if let Some(h) = parsed.get("height").and_then(|v| v.as_u64()) {
        if h == 0 || h > 4320 {
            warnings.push(format!("Unusual height {h} (expected 1-4320)"));
        }
    }

    if let Some(crf) = parsed.get("crf").and_then(|v| v.as_u64()) {
        if crf > 63 {
            errors.push(format!("CRF {crf} out of range (0-63)"));
        }
    }

    if let Some(br) = parsed.get("bitrate_kbps").and_then(|v| v.as_u64()) {
        if br == 0 {
            errors.push("Bitrate must be > 0".to_string());
        } else if br > 100_000 {
            warnings.push(format!("Very high bitrate {br} kbps"));
        }
    }

    let valid = errors.is_empty();
    let result = serde_json::json!({
        "valid": valid,
        "warnings": warnings,
        "errors": errors,
    });
    serde_json::to_string(&result)
        .map_err(|e| crate::utils::js_err(&format!("JSON serialization error: {e}")))
}

// ---------------------------------------------------------------------------
// WasmTranscodeWorker
// ---------------------------------------------------------------------------

/// Client-side transcode configuration builder and validator.
///
/// Actual encoding cannot run in WASM; this worker is for building and
/// validating configurations before sending them to a server or service worker.
#[wasm_bindgen]
pub struct WasmTranscodeWorker {
    video_codec: Option<String>,
    audio_codec: Option<String>,
    crf: Option<u32>,
    bitrate_kbps: Option<u32>,
    width: Option<u32>,
    height: Option<u32>,
    frame_rate: Option<f64>,
    audio_bitrate_kbps: Option<u32>,
    preset: Option<String>,
}

#[wasm_bindgen]
impl WasmTranscodeWorker {
    /// Create a new worker with empty configuration.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            video_codec: None,
            audio_codec: None,
            crf: None,
            bitrate_kbps: None,
            width: None,
            height: None,
            frame_rate: None,
            audio_bitrate_kbps: None,
            preset: None,
        }
    }

    /// Set the video codec.
    pub fn set_video_codec(&mut self, codec: &str) -> Result<(), JsValue> {
        let valid = ["av1", "vp9", "vp8"];
        if !valid.contains(&codec) {
            return Err(crate::utils::js_err(&format!(
                "Unsupported video codec '{codec}'. Supported: {}",
                valid.join(", ")
            )));
        }
        self.video_codec = Some(codec.to_string());
        Ok(())
    }

    /// Set the audio codec.
    pub fn set_audio_codec(&mut self, codec: &str) -> Result<(), JsValue> {
        let valid = ["opus", "vorbis", "flac", "pcm"];
        if !valid.contains(&codec) {
            return Err(crate::utils::js_err(&format!(
                "Unsupported audio codec '{codec}'. Supported: {}",
                valid.join(", ")
            )));
        }
        self.audio_codec = Some(codec.to_string());
        Ok(())
    }

    /// Set the CRF value (0-63).
    pub fn set_crf(&mut self, crf: u32) -> Result<(), JsValue> {
        if crf > 63 {
            return Err(crate::utils::js_err("CRF must be in range 0-63"));
        }
        self.crf = Some(crf);
        Ok(())
    }

    /// Set the target video bitrate in kbps.
    pub fn set_bitrate(&mut self, kbps: u32) -> Result<(), JsValue> {
        if kbps == 0 {
            return Err(crate::utils::js_err("Bitrate must be > 0"));
        }
        self.bitrate_kbps = Some(kbps);
        Ok(())
    }

    /// Set the output resolution.
    pub fn set_resolution(&mut self, width: u32, height: u32) -> Result<(), JsValue> {
        if width == 0 || height == 0 {
            return Err(crate::utils::js_err("Width and height must be > 0"));
        }
        self.width = Some(width);
        self.height = Some(height);
        Ok(())
    }

    /// Set the output frame rate.
    pub fn set_frame_rate(&mut self, fps: f64) -> Result<(), JsValue> {
        if fps <= 0.0 || fps > 240.0 {
            return Err(crate::utils::js_err("Frame rate must be in range (0, 240]"));
        }
        self.frame_rate = Some(fps);
        Ok(())
    }

    /// Set the audio bitrate in kbps.
    pub fn set_audio_bitrate(&mut self, kbps: u32) -> Result<(), JsValue> {
        if kbps == 0 {
            return Err(crate::utils::js_err("Audio bitrate must be > 0"));
        }
        self.audio_bitrate_kbps = Some(kbps);
        Ok(())
    }

    /// Set a named preset.
    pub fn set_preset(&mut self, name: &str) -> Result<(), JsValue> {
        let valid = [
            "web_optimized",
            "archive_quality",
            "fast_preview",
            "broadcast_hd",
            "social_media",
            "youtube_1080p",
            "vimeo_hd",
        ];
        if !valid.contains(&name) {
            return Err(crate::utils::js_err(&format!(
                "Unknown preset '{name}'. Valid: {}",
                valid.join(", ")
            )));
        }
        self.preset = Some(name.to_string());
        Ok(())
    }

    /// Return the current configuration as a JSON string.
    pub fn config_json(&self) -> String {
        let config = serde_json::json!({
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "crf": self.crf,
            "bitrate_kbps": self.bitrate_kbps,
            "width": self.width,
            "height": self.height,
            "frame_rate": self.frame_rate,
            "audio_bitrate_kbps": self.audio_bitrate_kbps,
            "preset": self.preset,
        });
        // This serialization is infallible for simple JSON values
        serde_json::to_string(&config).unwrap_or_else(|_| "{}".to_string())
    }

    /// Validate the current configuration, returning a JSON result.
    pub fn validate(&self) -> Result<String, JsValue> {
        wasm_validate_transcode_config(&self.config_json())
    }
}

impl Default for WasmTranscodeWorker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Estimation & recommendation
// ---------------------------------------------------------------------------

/// Estimate the output file size in bytes.
///
/// Returns a JSON object `{ estimated_bytes, estimated_mb }`.
#[wasm_bindgen]
pub fn wasm_estimate_output_size(
    duration_secs: f64,
    video_bitrate_kbps: u32,
    audio_bitrate_kbps: u32,
) -> Result<String, JsValue> {
    if duration_secs <= 0.0 {
        return Err(crate::utils::js_err("Duration must be > 0"));
    }
    let total_kbps = f64::from(video_bitrate_kbps) + f64::from(audio_bitrate_kbps);
    let bytes = (total_kbps * 1000.0 / 8.0) * duration_secs;
    let mb = bytes / (1024.0 * 1024.0);

    let result = serde_json::json!({
        "estimated_bytes": bytes as u64,
        "estimated_mb": (mb * 100.0).round() / 100.0,
        "duration_secs": duration_secs,
        "video_bitrate_kbps": video_bitrate_kbps,
        "audio_bitrate_kbps": audio_bitrate_kbps,
    });
    serde_json::to_string(&result)
        .map_err(|e| crate::utils::js_err(&format!("JSON serialization error: {e}")))
}

/// Get recommended encoding settings for a target format and quality level.
///
/// Arguments:
///     target_format: ``webm``, ``mkv``, ``ogg``
///     quality: ``low``, ``medium``, ``high``, ``very_high``
///
/// Returns a JSON object with recommended codec, bitrate, CRF, etc.
#[wasm_bindgen]
pub fn wasm_recommend_settings(target_format: &str, quality: &str) -> Result<String, JsValue> {
    let (video_codec, audio_codec) = match target_format.to_ascii_lowercase().as_str() {
        "webm" => ("av1", "opus"),
        "mkv" => ("av1", "opus"),
        "ogg" => ("vp8", "vorbis"),
        other => {
            return Err(crate::utils::js_err(&format!(
                "Unknown format '{other}'. Supported: webm, mkv, ogg"
            )));
        }
    };

    let (crf, video_bitrate_kbps, audio_bitrate_kbps, quality_mode) =
        match quality.to_ascii_lowercase().as_str() {
            "low" => (40, 800, 64, "fast"),
            "medium" => (30, 2500, 128, "balanced"),
            "high" => (23, 5000, 192, "high"),
            "very_high" => (18, 8000, 256, "high"),
            other => {
                return Err(crate::utils::js_err(&format!(
                    "Unknown quality '{other}'. Use: low, medium, high, very_high"
                )));
            }
        };

    let result = serde_json::json!({
        "target_format": target_format,
        "video_codec": video_codec,
        "audio_codec": audio_codec,
        "crf": crf,
        "video_bitrate_kbps": video_bitrate_kbps,
        "audio_bitrate_kbps": audio_bitrate_kbps,
        "quality_mode": quality_mode,
        "container": target_format,
    });
    serde_json::to_string(&result)
        .map_err(|e| crate::utils::js_err(&format!("JSON serialization error: {e}")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_presets_json() {
        let json = wasm_list_presets().expect("should produce JSON");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        let arr = parsed.as_array().expect("should be array");
        assert!(arr.len() >= 5);
    }

    #[test]
    fn test_list_codecs_json() {
        let json = wasm_list_codecs().expect("should produce JSON");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        let arr = parsed.as_array().expect("should be array");
        assert!(arr.len() >= 5);
    }

    #[test]
    fn test_validate_good_config() {
        let json = wasm_validate_transcode_config(r#"{"video_codec":"av1","crf":28}"#)
            .expect("should validate");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(parsed["valid"], true);
        assert!(parsed["errors"].as_array().expect("array").is_empty());
    }

    #[test]
    fn test_validate_bad_codec() {
        let json =
            wasm_validate_transcode_config(r#"{"video_codec":"h264"}"#).expect("should validate");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(parsed["valid"], false);
    }

    #[test]
    fn test_estimate_output_size() {
        let json = wasm_estimate_output_size(60.0, 5000, 128).expect("should estimate");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        let bytes = parsed["estimated_bytes"].as_u64().expect("u64");
        // 5128 kbps * 60s = 307680 kbits = 38460000 bytes
        assert!(bytes > 30_000_000);
    }

    #[test]
    fn test_recommend_settings() {
        let json = wasm_recommend_settings("webm", "high").expect("should recommend");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(parsed["video_codec"], "av1");
        assert_eq!(parsed["audio_codec"], "opus");
    }

    #[test]
    fn test_worker_config_round_trip() {
        let mut w = WasmTranscodeWorker::new();
        w.set_video_codec("vp9").expect("ok");
        w.set_crf(25).expect("ok");
        w.set_resolution(1920, 1080).expect("ok");
        let json = w.config_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(parsed["video_codec"], "vp9");
        assert_eq!(parsed["crf"], 25);
        assert_eq!(parsed["width"], 1920);
    }
}
