//! WebAssembly bindings for media deduplication utilities.
//!
//! Provides browser-side content hashing, frame comparison,
//! and dedup strategy enumeration.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Compute a content hash for the given media data bytes.
///
/// Uses a fast non-cryptographic hash suitable for duplicate detection.
///
/// # Arguments
/// * `data` - Raw media bytes.
///
/// # Returns
/// Hex-encoded hash string.
#[wasm_bindgen]
pub fn wasm_compute_media_hash(data: &[u8]) -> String {
    let mut hasher: u64 = 0x6295c58d62b82175;
    for &byte in data {
        hasher ^= u64::from(byte);
        hasher = hasher.wrapping_mul(0x517cc1b727220a95);
        hasher = hasher.rotate_left(31);
    }
    format!("{:016x}", hasher)
}

/// Compare two RGB frames and return a similarity score [0.0, 1.0].
///
/// Both frames must have the same dimensions (`width * height * 3` bytes).
/// A score of 1.0 means identical frames.
///
/// # Arguments
/// * `frame1` - First RGB frame data.
/// * `frame2` - Second RGB frame data.
/// * `width` - Frame width in pixels.
/// * `height` - Frame height in pixels.
///
/// # Errors
/// Returns an error if frame sizes don't match expected dimensions.
#[wasm_bindgen]
pub fn wasm_compare_frames(
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

    let mut sum_diff: u64 = 0;
    for i in 0..expected {
        let diff = (frame1[i] as i32 - frame2[i] as i32).unsigned_abs();
        sum_diff += diff as u64;
    }

    let avg_diff = sum_diff as f64 / expected as f64;
    // Normalize: 0 difference = 1.0 similarity, max diff (255) = 0.0
    let similarity = 1.0 - (avg_diff / 255.0).min(1.0);
    Ok(similarity)
}

/// Return a JSON array of available dedup strategies with descriptions.
#[wasm_bindgen]
pub fn wasm_dedup_strategies() -> String {
    let strategies = r#"[
  {"name": "exact_hash", "description": "Cryptographic hash comparison for exact duplicates", "speed": "fast"},
  {"name": "perceptual_hash", "description": "Perceptual hashing for visually similar images/frames", "speed": "medium"},
  {"name": "ssim", "description": "Structural Similarity Index for visual comparison", "speed": "slow"},
  {"name": "histogram", "description": "Color histogram comparison", "speed": "medium"},
  {"name": "feature_match", "description": "Feature point matching for similar content", "speed": "slow"},
  {"name": "audio_fingerprint", "description": "Audio fingerprint comparison", "speed": "medium"},
  {"name": "metadata", "description": "Metadata-based fuzzy matching", "speed": "fast"},
  {"name": "fast", "description": "Combination of fast methods (hash + perceptual + metadata)", "speed": "fast"},
  {"name": "all", "description": "All detection methods combined", "speed": "slow"}
]"#;
    strategies.to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_hash_deterministic() {
        let data = b"test data for dedup wasm";
        let h1 = wasm_compute_media_hash(data);
        let h2 = wasm_compute_media_hash(data);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 16);
    }

    #[test]
    fn test_compute_hash_different_inputs() {
        let h1 = wasm_compute_media_hash(b"input one");
        let h2 = wasm_compute_media_hash(b"input two");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_compare_frames_identical() {
        let frame = vec![128u8; 8 * 8 * 3];
        let result = wasm_compare_frames(&frame, &frame, 8, 8);
        assert!(result.is_ok());
        let similarity = result.expect("compare");
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compare_frames_different() {
        let frame1 = vec![0u8; 8 * 8 * 3];
        let frame2 = vec![255u8; 8 * 8 * 3];
        let result = wasm_compare_frames(&frame1, &frame2, 8, 8);
        assert!(result.is_ok());
        let similarity = result.expect("compare");
        assert!(similarity < 0.01);
    }

    #[test]
    fn test_dedup_strategies_valid_json() {
        let s = wasm_dedup_strategies();
        assert!(s.contains("exact_hash"));
        assert!(s.contains("perceptual_hash"));
    }
}
