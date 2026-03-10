//! Scene detection WASM bindings.
//!
//! Exposes a pure-Rust histogram-difference scene-cut detector that works
//! entirely on raw RGB frame data supplied from JavaScript.  No external
//! dependencies beyond `wasm_bindgen`, `serde_json`, and the standard
//! library are required — the algorithm is intentionally self-contained so
//! it compiles to `wasm32-unknown-unknown` without issue.
//!
//! # Algorithm
//!
//! For each consecutive pair of frames the function computes the **mean
//! absolute difference of luminance** (approximated as
//! `0.299·R + 0.587·G + 0.114·B`) across all pixels.  When this metric
//! exceeds `threshold` a scene boundary is recorded.
//!
//! # JavaScript Example
//!
//! ```javascript
//! import * as oximedia from 'oximedia-wasm';
//!
//! // Build a flat Uint8Array of width * height * 3 bytes per frame,
//! // frame_count frames concatenated.
//! const scenes = JSON.parse(
//!     oximedia.wasm_detect_scenes(frames, width, height, frame_count, 0.35)
//! );
//! // scenes: [{start_frame, end_frame, score}, ...]
//! console.log(`Detected ${scenes.length} scene(s)`);
//! ```

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Public data types

/// Describes a single detected scene segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneSegment {
    /// Index of the first frame in this scene (inclusive, 0-based).
    pub start_frame: u32,
    /// Index of the last frame in this scene (inclusive).
    pub end_frame: u32,
    /// Maximum inter-frame difference score that triggered the boundary.
    /// Range: 0.0 – 1.0.
    pub score: f32,
}

// ---------------------------------------------------------------------------
// Wasm entry point

/// Detect scene cuts in a flat buffer of RGB video frames.
///
/// # Arguments
///
/// * `frames` - Flat `Uint8Array` containing all frames concatenated in
///   display order.  Each frame occupies exactly `width * height * 3` bytes
///   in packed RGB format (R, G, B, R, G, B, …).
/// * `width` - Frame width in pixels.
/// * `height` - Frame height in pixels.
/// * `frame_count` - Total number of frames in `frames`.
/// * `threshold` - Inter-frame difference threshold in the range `[0.0, 1.0]`.
///   Lower values produce more cuts; a value around `0.3`–`0.5` is typical
///   for hard cuts with little compression noise.
///
/// # Returns
///
/// A JSON string encoding an array of scene segment objects:
/// `[{"start_frame":0,"end_frame":42,"score":0.0},{"start_frame":43,...},...]`
///
/// The first segment always starts at frame 0.
///
/// # Errors
///
/// Returns a JavaScript exception if:
/// - `width` or `height` is zero
/// - `frame_count` is zero
/// - `frames` length does not match `width * height * 3 * frame_count`
/// - JSON serialisation fails (should never occur)
#[wasm_bindgen]
pub fn wasm_detect_scenes(
    frames: &[u8],
    width: u32,
    height: u32,
    frame_count: u32,
    threshold: f32,
) -> Result<String, JsValue> {
    detect_scenes_inner(frames, width, height, frame_count, threshold)
        .map_err(|e| crate::utils::js_err(&e))
}

/// Pure-Rust implementation of scene detection — callable from native tests.
///
/// Returns a JSON string on success, or an error string on failure.
pub(crate) fn detect_scenes_inner(
    frames: &[u8],
    width: u32,
    height: u32,
    frame_count: u32,
    threshold: f32,
) -> Result<String, String> {
    // --- Validate inputs ---------------------------------------------------
    if width == 0 {
        return Err("wasm_detect_scenes: width must be > 0".to_string());
    }
    if height == 0 {
        return Err("wasm_detect_scenes: height must be > 0".to_string());
    }
    if frame_count == 0 {
        return Err("wasm_detect_scenes: frame_count must be > 0".to_string());
    }
    let frame_bytes = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| "wasm_detect_scenes: frame dimensions overflow usize".to_string())?;

    let expected_total = frame_bytes
        .checked_mul(frame_count as usize)
        .ok_or_else(|| "wasm_detect_scenes: total byte count overflows usize".to_string())?;

    if frames.len() != expected_total {
        return Err(format!(
            "wasm_detect_scenes: frames buffer length {} does not match \
             expected {} (width={} height={} frame_count={} bytes_per_frame={})",
            frames.len(),
            expected_total,
            width,
            height,
            frame_count,
            frame_bytes
        ));
    }

    let threshold_clamped = threshold.clamp(0.0, 1.0);

    // --- Compute inter-frame differences and emit segments ----------------
    let mut segments: Vec<SceneSegment> = Vec::new();
    let mut current_scene_start: u32 = 0;
    let mut max_score_in_scene: f32 = 0.0;

    // We compare frame[i-1] with frame[i] for i in 1..frame_count.
    for i in 1..(frame_count as usize) {
        let prev_start = (i - 1) * frame_bytes;
        let curr_start = i * frame_bytes;

        let diff = mean_luma_difference(
            &frames[prev_start..prev_start + frame_bytes],
            &frames[curr_start..curr_start + frame_bytes],
        );

        // Track the highest difference seen within the current segment.
        if diff > max_score_in_scene {
            max_score_in_scene = diff;
        }

        if diff >= threshold_clamped {
            // Scene cut detected before frame `i`.
            segments.push(SceneSegment {
                start_frame: current_scene_start,
                end_frame: (i as u32).saturating_sub(1),
                score: max_score_in_scene,
            });
            current_scene_start = i as u32;
            max_score_in_scene = 0.0;
        }
    }

    // Always emit the final (possibly only) segment.
    segments.push(SceneSegment {
        start_frame: current_scene_start,
        end_frame: frame_count.saturating_sub(1),
        score: max_score_in_scene,
    });

    serde_json::to_string(&segments).map_err(|e| format!("wasm_detect_scenes: JSON error: {e}"))
}

// ---------------------------------------------------------------------------
// Internal helpers

/// Compute the mean absolute difference of approximate luminance between two
/// same-sized RGB byte slices.
///
/// Luminance approximation: `Y ≈ 0.299·R + 0.587·G + 0.114·B`
///
/// The result is normalised to `[0.0, 1.0]` where `1.0` represents the
/// maximum possible difference (all pixels flipped from black to white or
/// vice-versa).
fn mean_luma_difference(a: &[u8], b: &[u8]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len() % 3, 0);

    if a.is_empty() {
        return 0.0;
    }

    let pixel_count = a.len() / 3;
    let mut sum_diff: f64 = 0.0;

    // Iterate over triples (R, G, B).
    let mut idx = 0;
    while idx + 3 <= a.len() {
        let luma_a = luma_approx(a[idx], a[idx + 1], a[idx + 2]);
        let luma_b = luma_approx(b[idx], b[idx + 1], b[idx + 2]);
        sum_diff += (luma_a as f64 - luma_b as f64).abs();
        idx += 3;
    }

    // Normalise: max possible sum_diff = pixel_count * 255.0
    let mean = sum_diff / (pixel_count as f64 * 255.0);
    mean as f32
}

/// Fast integer approximation of luminance.
///
/// Uses the Rec. 601 coefficients scaled by 1024 to avoid floating point:
/// `Y = (306*R + 601*G + 117*B) >> 10`
///
/// The result is in the range `[0, 255]`.
#[inline(always)]
fn luma_approx(r: u8, g: u8, b: u8) -> u16 {
    let r = r as u32;
    let g = g as u32;
    let b = b as u32;
    // Coefficients: 0.299 ≈ 306/1024, 0.587 ≈ 601/1024, 0.114 ≈ 117/1024
    ((306 * r + 601 * g + 117 * b) >> 10) as u16
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_frame(r: u8, g: u8, b: u8, pixels: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(pixels * 3);
        for _ in 0..pixels {
            v.push(r);
            v.push(g);
            v.push(b);
        }
        v
    }

    #[test]
    fn test_luma_approx_black() {
        assert_eq!(luma_approx(0, 0, 0), 0);
    }

    #[test]
    fn test_luma_approx_white() {
        // Maximum value should be close to 255.
        let y = luma_approx(255, 255, 255);
        assert!(y >= 254 && y <= 255, "expected ~255, got {y}");
    }

    #[test]
    fn test_mean_luma_difference_identical() {
        let frame = solid_frame(100, 150, 200, 16);
        let diff = mean_luma_difference(&frame, &frame);
        assert!(
            diff < 1e-6,
            "identical frames should have diff ≈ 0, got {diff}"
        );
    }

    #[test]
    fn test_mean_luma_difference_black_white() {
        let black = solid_frame(0, 0, 0, 16);
        let white = solid_frame(255, 255, 255, 16);
        let diff = mean_luma_difference(&black, &white);
        assert!(diff > 0.99, "black vs white should be near 1.0, got {diff}");
    }

    #[test]
    fn test_detect_scenes_single_frame() {
        // A single frame: no inter-frame differences, one segment covering frame 0.
        let frame = solid_frame(128, 128, 128, 4); // 2x2 RGB
        let json =
            detect_scenes_inner(&frame, 2, 2, 1, 0.3).expect("scene detection should succeed");
        let segments: Vec<SceneSegment> =
            serde_json::from_str(&json).expect("serde_json::from_str should succeed");
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start_frame, 0);
        assert_eq!(segments[0].end_frame, 0);
    }

    #[test]
    fn test_detect_scenes_no_cuts() {
        // 5 identical frames → no cuts, one segment.
        let frame = solid_frame(100, 100, 100, 4);
        let mut all: Vec<u8> = Vec::new();
        for _ in 0..5 {
            all.extend_from_slice(&frame);
        }
        let json = detect_scenes_inner(&all, 2, 2, 5, 0.3).expect("scene detection should succeed");
        let segments: Vec<SceneSegment> =
            serde_json::from_str(&json).expect("serde_json::from_str should succeed");
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start_frame, 0);
        assert_eq!(segments[0].end_frame, 4);
    }

    #[test]
    fn test_detect_scenes_hard_cut_in_middle() {
        // Frames 0-2: black.  Frame 3 onwards: white.
        let black = solid_frame(0, 0, 0, 4);
        let white = solid_frame(255, 255, 255, 4);
        let mut all: Vec<u8> = Vec::new();
        for _ in 0..3 {
            all.extend_from_slice(&black);
        }
        for _ in 0..3 {
            all.extend_from_slice(&white);
        }
        let json = detect_scenes_inner(&all, 2, 2, 6, 0.3).expect("scene detection should succeed");
        let segments: Vec<SceneSegment> =
            serde_json::from_str(&json).expect("serde_json::from_str should succeed");
        assert_eq!(segments.len(), 2, "expected exactly one cut: {json}");
        assert_eq!(segments[0].start_frame, 0);
        assert_eq!(segments[0].end_frame, 2);
        assert_eq!(segments[1].start_frame, 3);
        assert_eq!(segments[1].end_frame, 5);
    }

    #[test]
    fn test_detect_scenes_wrong_buffer_size() {
        let result = detect_scenes_inner(&[0u8; 10], 4, 4, 2, 0.3);
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_scenes_zero_width_fails() {
        let result = detect_scenes_inner(&[], 0, 4, 1, 0.3);
        assert!(result.is_err());
    }
}
