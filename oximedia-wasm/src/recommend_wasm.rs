//! WebAssembly bindings for content recommendation utilities.
//!
//! Provides functions for codec recommendation, encoding settings,
//! and content analysis in the browser.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Recommend a codec for a given use case.
///
/// # Arguments
/// * `use_case` - Target use case: streaming, archival, editing, broadcast.
///
/// # Returns
/// JSON object with video and audio codec recommendations.
#[wasm_bindgen]
pub fn wasm_recommend_codec(use_case: &str) -> Result<String, JsValue> {
    let valid = ["streaming", "archival", "editing", "broadcast"];
    if !valid.contains(&use_case) {
        return Err(crate::utils::js_err(&format!(
            "Unknown use case '{}'. Supported: streaming, archival, editing, broadcast",
            use_case
        )));
    }

    let (video, audio, confidence, reason) = match use_case {
        "streaming" => ("AV1", "Opus", 0.95, "Best compression for streaming"),
        "archival" => ("AV1", "FLAC", 0.90, "Efficient long-term storage"),
        "editing" => ("VP9", "FLAC", 0.85, "Fast decode for editing"),
        "broadcast" => ("AV1", "Opus", 0.92, "High quality broadcast"),
        _ => ("AV1", "Opus", 0.80, "Default"),
    };

    Ok(format!(
        "{{\"use_case\":\"{use_case}\",\"video_codec\":\"{video}\",\"audio_codec\":\"{audio}\",\"confidence\":{confidence:.2},\"reason\":\"{reason}\"}}"
    ))
}

/// Recommend encoding settings for a codec and optimization target.
///
/// # Arguments
/// * `codec` - Target codec (av1, vp9, vp8, opus, vorbis, flac).
/// * `target` - Optimization target: quality, speed, size, balanced.
///
/// # Returns
/// JSON object with recommended settings.
#[wasm_bindgen]
pub fn wasm_recommend_settings(codec: &str, target: &str) -> Result<String, JsValue> {
    let valid_codecs = ["av1", "vp9", "vp8", "opus", "vorbis", "flac"];
    if !valid_codecs.contains(&codec) {
        return Err(crate::utils::js_err(&format!(
            "Unsupported codec '{}'. Patent-free only.",
            codec
        )));
    }

    let valid_targets = ["quality", "speed", "size", "balanced"];
    if !valid_targets.contains(&target) {
        return Err(crate::utils::js_err(&format!(
            "Unknown target '{}'. Supported: quality, speed, size, balanced",
            target
        )));
    }

    let (preset, crf, threads) = match target {
        "quality" => ("slow", 22, 0),
        "speed" => ("ultrafast", 28, 0),
        "size" => ("medium", 32, 0),
        _ => ("medium", 26, 0),
    };

    Ok(format!(
        "{{\"codec\":\"{codec}\",\"target\":\"{target}\",\"preset\":\"{preset}\",\"crf\":{crf},\"threads\":{threads},\"pixel_format\":\"yuv420p\"}}"
    ))
}

/// Analyze content characteristics for recommendation.
///
/// # Arguments
/// * `width` - Video width.
/// * `height` - Video height.
/// * `fps` - Frames per second.
/// * `duration_secs` - Duration in seconds.
///
/// # Returns
/// JSON object with content analysis and recommendations.
#[wasm_bindgen]
pub fn wasm_analyze_content(
    width: u32,
    height: u32,
    fps: f64,
    duration_secs: f64,
) -> Result<String, JsValue> {
    if width == 0 || height == 0 {
        return Err(crate::utils::js_err("Width and height must be > 0"));
    }
    if fps <= 0.0 {
        return Err(crate::utils::js_err("FPS must be positive"));
    }
    if duration_secs <= 0.0 {
        return Err(crate::utils::js_err("Duration must be positive"));
    }

    let total_frames = (fps * duration_secs) as u64;
    let pixels_per_frame = width as u64 * height as u64;
    let total_pixels = pixels_per_frame * total_frames;

    // Recommend bitrate based on resolution
    let recommended_bitrate = match pixels_per_frame {
        0..=921_600 => 2500,
        921_601..=2_073_600 => 6000,
        2_073_601..=3_686_400 => 12000,
        _ => 20000,
    };

    let complexity = if fps > 60.0 {
        "high"
    } else if fps > 30.0 {
        "medium"
    } else {
        "low"
    };

    Ok(format!(
        "{{\"resolution\":\"{}x{}\",\"fps\":{fps},\"duration_secs\":{duration_secs},\"total_frames\":{total_frames},\"total_pixels\":{total_pixels},\"recommended_bitrate_kbps\":{recommended_bitrate},\"complexity\":\"{complexity}\",\"recommended_codec\":\"AV1\"}}",
        width, height
    ))
}

/// Get supported recommendation strategies as JSON.
#[wasm_bindgen]
pub fn wasm_recommendation_strategies() -> String {
    "[\"content-based\",\"collaborative\",\"hybrid\",\"personalized\",\"trending\"]".to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recommend_codec() {
        let result = wasm_recommend_codec("streaming");
        assert!(result.is_ok());
        let json = result.expect("should succeed");
        assert!(json.contains("AV1"));
        assert!(json.contains("Opus"));
    }

    #[test]
    fn test_recommend_codec_invalid() {
        let result = wasm_recommend_codec("gaming");
        assert!(result.is_err());
    }

    #[test]
    fn test_recommend_settings() {
        let result = wasm_recommend_settings("av1", "quality");
        assert!(result.is_ok());
        let json = result.expect("should succeed");
        assert!(json.contains("slow"));
    }

    #[test]
    fn test_analyze_content() {
        let result = wasm_analyze_content(1920, 1080, 30.0, 60.0);
        assert!(result.is_ok());
        let json = result.expect("should succeed");
        assert!(json.contains("1920x1080"));
        assert!(json.contains("AV1"));
    }

    #[test]
    fn test_strategies() {
        let json = wasm_recommendation_strategies();
        assert!(json.contains("hybrid"));
    }
}
