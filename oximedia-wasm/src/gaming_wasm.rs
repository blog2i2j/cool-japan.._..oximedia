//! WebAssembly bindings for gaming highlight detection and capture utilities.
//!
//! Provides `WasmHighlightDetector` for incremental highlight detection,
//! plus standalone functions for motion analysis, audio peaks, and capture settings.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// WasmHighlightDetector
// ---------------------------------------------------------------------------

/// Browser-side highlight detector for gaming content.
///
/// Usage:
/// 1. Create a detector with threshold and minimum duration.
/// 2. Feed per-frame scores via `add_frame_score`.
/// 3. Call `detect()` to get JSON array of highlight regions.
#[wasm_bindgen]
pub struct WasmHighlightDetector {
    threshold: f64,
    min_duration: f64,
    fps: f64,
    scores: Vec<f64>,
}

#[wasm_bindgen]
impl WasmHighlightDetector {
    /// Create a new highlight detector.
    ///
    /// # Arguments
    /// * `threshold` - Minimum score to count as a highlight (0.0-1.0).
    /// * `min_duration` - Minimum highlight duration in seconds.
    /// * `fps` - Frames per second for time calculations.
    ///
    /// # Errors
    /// Returns an error if parameters are invalid.
    #[wasm_bindgen(constructor)]
    pub fn new(
        threshold: f64,
        min_duration: f64,
        fps: f64,
    ) -> Result<WasmHighlightDetector, JsValue> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(crate::utils::js_err(
                "Threshold must be between 0.0 and 1.0",
            ));
        }
        if min_duration <= 0.0 {
            return Err(crate::utils::js_err("Minimum duration must be positive"));
        }
        if fps <= 0.0 {
            return Err(crate::utils::js_err("FPS must be positive"));
        }

        Ok(Self {
            threshold,
            min_duration,
            fps,
            scores: Vec::new(),
        })
    }

    /// Add a per-frame excitement/motion score (0.0-1.0).
    pub fn add_frame_score(&mut self, score: f64) {
        self.scores.push(score.clamp(0.0, 1.0));
    }

    /// Detect highlights from accumulated scores.
    ///
    /// Returns a JSON array of highlight objects:
    /// ```json
    /// [{"start_time": 5.2, "end_time": 8.5, "score": 0.87, "type": "motion"}]
    /// ```
    pub fn detect(&self) -> Result<String, JsValue> {
        if self.scores.is_empty() {
            return Ok("[]".to_string());
        }

        let total_duration = self.scores.len() as f64 / self.fps;
        let mut highlights = Vec::new();
        let mut region_start: Option<usize> = None;
        let mut region_max_score: f64 = 0.0;

        for (i, &score) in self.scores.iter().enumerate() {
            if score >= self.threshold {
                if region_start.is_none() {
                    region_start = Some(i);
                    region_max_score = score;
                } else if score > region_max_score {
                    region_max_score = score;
                }
            } else if let Some(start) = region_start {
                let start_time = start as f64 / self.fps;
                let end_time = i as f64 / self.fps;
                let duration = end_time - start_time;

                if duration >= self.min_duration {
                    highlights.push(format!(
                        "{{\"start_time\":{start_time:.4},\"end_time\":{end_time:.4},\"score\":{region_max_score:.4},\"type\":\"motion\"}}"
                    ));
                }
                region_start = None;
                region_max_score = 0.0;
            }
        }

        // Handle region at end
        if let Some(start) = region_start {
            let start_time = start as f64 / self.fps;
            let end_time = total_duration;
            let duration = end_time - start_time;

            if duration >= self.min_duration {
                highlights.push(format!(
                    "{{\"start_time\":{start_time:.4},\"end_time\":{end_time:.4},\"score\":{region_max_score:.4},\"type\":\"motion\"}}"
                ));
            }
        }

        Ok(format!("[{}]", highlights.join(",")))
    }

    /// Reset the detector, clearing all accumulated scores.
    pub fn reset(&mut self) {
        self.scores.clear();
    }

    /// Get the number of frames processed.
    pub fn frame_count(&self) -> u32 {
        self.scores.len() as u32
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Calculate motion intensity between two RGB frames.
///
/// Both frames must contain `width * height * 3` bytes of RGB data.
/// Returns a value in [0.0, 1.0] representing motion intensity.
#[wasm_bindgen]
pub fn wasm_detect_motion_intensity(
    frame1: &[u8],
    frame2: &[u8],
    width: u32,
    height: u32,
) -> Result<f64, JsValue> {
    let expected = (width as usize) * (height as usize) * 3;

    if frame1.len() < expected {
        return Err(crate::utils::js_err(&format!(
            "frame1 too small: need {} bytes, got {}",
            expected,
            frame1.len()
        )));
    }
    if frame2.len() < expected {
        return Err(crate::utils::js_err(&format!(
            "frame2 too small: need {} bytes, got {}",
            expected,
            frame2.len()
        )));
    }

    let len = expected;
    let mut sum_diff: u64 = 0;
    for i in 0..len {
        let diff = (frame1[i] as i32 - frame2[i] as i32).unsigned_abs();
        sum_diff += diff as u64;
    }

    let avg_diff = sum_diff as f64 / len as f64;
    Ok((avg_diff / 128.0).min(1.0))
}

/// Generate a preview frame with text overlay.
///
/// Takes an RGB frame and overlays simple text (rendered as bright pixels in a
/// small region at the top). Returns a new RGB frame.
#[wasm_bindgen]
pub fn wasm_generate_clip_preview(
    frame_data: &[u8],
    width: u32,
    height: u32,
    overlay_text: &str,
) -> Result<Vec<u8>, JsValue> {
    let w = width as usize;
    let h = height as usize;
    let expected = w * h * 3;

    if frame_data.len() < expected {
        return Err(crate::utils::js_err(&format!(
            "Frame data too small: need {} bytes, got {}",
            expected,
            frame_data.len()
        )));
    }

    let mut output = frame_data[..expected].to_vec();

    // Draw a semi-transparent bar at the top for text overlay
    let bar_height = 40.min(h);
    for y in 0..bar_height {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            // Darken the background for contrast
            output[idx] /= 3;
            output[idx + 1] /= 3;
            output[idx + 2] /= 3;
        }
    }

    // Simple pixel-level text representation (each char = 6px wide block)
    let char_width = 6;
    let char_height = 10.min(bar_height);
    let y_offset = (bar_height.saturating_sub(char_height)) / 2;
    let x_start = 10;

    for (ci, _ch) in overlay_text.chars().enumerate() {
        let cx = x_start + ci * char_width;
        if cx + char_width >= w {
            break;
        }
        // Draw a bright block for each character position
        for dy in 0..char_height {
            for dx in 0..char_width.saturating_sub(1) {
                let y = y_offset + dy;
                let x = cx + dx;
                if y < bar_height && x < w {
                    let idx = (y * w + x) * 3;
                    output[idx] = 255;
                    output[idx + 1] = 255;
                    output[idx + 2] = 255;
                }
            }
        }
    }

    Ok(output)
}

/// Detect audio peaks for highlight moments.
///
/// Analyzes audio samples in sliding windows and returns a JSON array of
/// peak objects with time and level.
///
/// # Arguments
/// * `samples` - Audio samples as f32 array.
/// * `window_size` - Window size in samples.
///
/// # Returns
/// JSON array: `[{"time": 1.5, "level": 0.85}, ...]`
#[wasm_bindgen]
pub fn wasm_detect_audio_peak(
    samples: &[f32],
    sample_rate: u32,
    window_size: u32,
) -> Result<String, JsValue> {
    if samples.is_empty() {
        return Ok("[]".to_string());
    }
    if sample_rate == 0 {
        return Err(crate::utils::js_err("Sample rate must be > 0"));
    }
    if window_size == 0 {
        return Err(crate::utils::js_err("Window size must be > 0"));
    }

    let win = window_size as usize;
    let hop = win / 2;
    if hop == 0 {
        return Err(crate::utils::js_err("Window size must be at least 2"));
    }

    // Compute RMS per window
    let mut windows = Vec::new();
    let mut offset = 0;

    while offset + win <= samples.len() {
        let window = &samples[offset..offset + win];
        let sum_sq: f64 = window.iter().map(|&s| (s as f64) * (s as f64)).sum();
        let rms = (sum_sq / win as f64).sqrt();
        let time = (offset as f64 + win as f64 / 2.0) / sample_rate as f64;
        windows.push((time, rms));
        offset += hop;
    }

    // Find max RMS for normalization
    let max_rms = windows.iter().map(|(_, r)| *r).fold(0.0_f64, f64::max);

    if max_rms < 1e-10 {
        return Ok("[]".to_string());
    }

    // Return peaks above 0.7 of max
    let peak_threshold = 0.7 * max_rms;
    let mut peaks = Vec::new();

    for (time, rms) in &windows {
        if *rms >= peak_threshold {
            let level = *rms / max_rms;
            peaks.push(format!("{{\"time\":{time:.4},\"level\":{level:.4}}}"));
        }
    }

    Ok(format!("[{}]", peaks.join(",")))
}

/// Return recommended capture settings as JSON for common gaming resolutions.
///
/// # Arguments
/// * `resolution` - One of: "720p", "1080p", "1440p", "4k".
/// * `fps` - Target framerate.
///
/// # Returns
/// JSON object with recommended settings.
#[wasm_bindgen]
pub fn wasm_gaming_capture_settings(resolution: &str, fps: u32) -> String {
    let (width, height, base_bitrate) = match resolution {
        "720p" => (1280, 720, 2500),
        "1080p" => (1920, 1080, 6000),
        "1440p" => (2560, 1440, 12000),
        "4k" | "2160p" => (3840, 2160, 20000),
        _ => (1920, 1080, 6000),
    };

    let bitrate = if fps > 60 {
        base_bitrate * 3 / 2
    } else {
        base_bitrate
    };

    let preset = if fps >= 120 {
        "ultra_low_latency"
    } else if fps >= 60 {
        "low_latency"
    } else {
        "balanced"
    };

    let keyframe_interval = fps * 2;

    format!(
        "{{\"width\":{width},\"height\":{height},\"fps\":{fps},\"bitrate_kbps\":{bitrate},\
         \"codec\":\"av1\",\"format\":\"webm\",\"preset\":\"{preset}\",\
         \"keyframe_interval\":{keyframe_interval},\"pixel_format\":\"yuv420p\"}}"
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detector_creation() {
        let d = WasmHighlightDetector::new(0.5, 2.0, 30.0);
        assert!(d.is_ok());
    }

    #[test]
    fn test_detector_invalid_threshold() {
        let d = WasmHighlightDetector::new(1.5, 2.0, 30.0);
        assert!(d.is_err());
    }

    #[test]
    fn test_detect_empty() {
        let d = WasmHighlightDetector::new(0.5, 1.0, 30.0).expect("creation should succeed");
        let result = d.detect();
        assert!(result.is_ok());
        assert_eq!(result.expect("detect should succeed"), "[]");
    }

    #[test]
    fn test_motion_intensity_identical() {
        let frame = vec![128u8; 16 * 16 * 3];
        let result = wasm_detect_motion_intensity(&frame, &frame, 16, 16);
        assert!(result.is_ok());
        let intensity = result.expect("should succeed");
        assert!(intensity < 0.01);
    }

    #[test]
    fn test_gaming_capture_settings() {
        let json = wasm_gaming_capture_settings("1080p", 60);
        assert!(json.contains("\"width\":1920"));
        assert!(json.contains("\"height\":1080"));
        assert!(json.contains("\"fps\":60"));
    }

    #[test]
    fn test_audio_peak_empty() {
        let result = wasm_detect_audio_peak(&[], 48000, 1024);
        assert!(result.is_ok());
        assert_eq!(result.expect("should succeed"), "[]");
    }

    #[test]
    fn test_clip_preview() {
        let frame = vec![100u8; 32 * 32 * 3];
        let result = wasm_generate_clip_preview(&frame, 32, 32, "HI");
        assert!(result.is_ok());
        let output = result.expect("should succeed");
        assert_eq!(output.len(), 32 * 32 * 3);
    }
}
