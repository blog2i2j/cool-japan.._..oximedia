//! WebAssembly bindings for proxy media utilities.
//!
//! Provides functions for proxy settings, size estimation, and format queries
//! in the browser.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Get proxy settings for a given resolution and quality preset.
///
/// # Arguments
/// * `resolution` - One of: "quarter", "half", "full".
/// * `quality` - One of: "low", "medium", "high".
///
/// # Returns
/// JSON object with recommended proxy settings.
#[wasm_bindgen]
pub fn wasm_proxy_settings(resolution: &str, quality: &str) -> String {
    let scale = match resolution {
        "quarter" => 0.25,
        "half" => 0.5,
        "full" => 1.0,
        _ => 0.25,
    };

    let (bitrate_1080p, quality_label) = match quality {
        "low" => (2_000_000, "low"),
        "medium" => (5_000_000, "medium"),
        "high" => (10_000_000, "high"),
        _ => (5_000_000, "medium"),
    };

    let target_width = (1920.0 * scale) as u32;
    let target_height = (1080.0 * scale) as u32;
    let target_bitrate = (bitrate_1080p as f64 * scale * scale) as u64;

    format!(
        "{{\"resolution\":\"{resolution}\",\"quality\":\"{quality_label}\",\
         \"scale\":{scale},\"target_width\":{target_width},\"target_height\":{target_height},\
         \"bitrate_bps\":{target_bitrate},\"codec\":\"vp9\",\"format\":\"webm\",\
         \"keyframe_interval\":60,\"pixel_format\":\"yuv420p\"}}"
    )
}

/// Estimate the proxy file size for a given original.
///
/// # Arguments
/// * `original_size_bytes` - Size of the original file in bytes.
/// * `resolution` - Resolution preset.
/// * `quality` - Quality preset.
///
/// # Returns
/// JSON object with estimated sizes and compression ratio.
#[wasm_bindgen]
pub fn wasm_estimate_proxy_size(
    original_size_bytes: u64,
    resolution: &str,
    quality: &str,
) -> String {
    let scale = match resolution {
        "quarter" => 0.25,
        "half" => 0.5,
        "full" => 1.0,
        _ => 0.25,
    };

    let quality_factor = match quality {
        "low" => 0.1,
        "medium" => 0.25,
        "high" => 0.5,
        _ => 0.25,
    };

    let estimated_size = (original_size_bytes as f64 * scale * scale * quality_factor) as u64;
    let savings_bytes = original_size_bytes.saturating_sub(estimated_size);
    let savings_pct = if original_size_bytes > 0 {
        (savings_bytes as f64 / original_size_bytes as f64 * 100.0) as u32
    } else {
        0
    };
    let ratio = if estimated_size > 0 {
        format!("{:.1}", original_size_bytes as f64 / estimated_size as f64)
    } else {
        "0".to_string()
    };

    format!(
        "{{\"original_bytes\":{original_size_bytes},\"estimated_proxy_bytes\":{estimated_size},\
         \"savings_bytes\":{savings_bytes},\"savings_pct\":{savings_pct},\
         \"compression_ratio\":{ratio}}}"
    )
}

/// List supported proxy formats as JSON array.
///
/// # Returns
/// JSON array of format objects with codec, container, and description.
#[wasm_bindgen]
pub fn wasm_proxy_formats() -> String {
    "[{\"codec\":\"vp9\",\"container\":\"webm\",\"description\":\"VP9 in WebM - good balance\"},\
      {\"codec\":\"av1\",\"container\":\"webm\",\"description\":\"AV1 in WebM - best compression\"}]"
        .to_string()
}

/// List supported resolution presets as JSON array.
#[wasm_bindgen]
pub fn wasm_proxy_resolutions() -> String {
    "[{\"name\":\"quarter\",\"scale\":0.25,\"label\":\"Quarter Resolution\"},\
      {\"name\":\"half\",\"scale\":0.5,\"label\":\"Half Resolution\"},\
      {\"name\":\"full\",\"scale\":1.0,\"label\":\"Full Resolution\"}]"
        .to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proxy_settings_quarter() {
        let json = wasm_proxy_settings("quarter", "medium");
        assert!(json.contains("\"scale\":0.25"));
        assert!(json.contains("\"target_width\":480"));
        assert!(json.contains("\"target_height\":270"));
        assert!(json.contains("\"codec\":\"vp9\""));
    }

    #[test]
    fn test_proxy_settings_full() {
        let json = wasm_proxy_settings("full", "high");
        assert!(json.contains("\"scale\":1"));
        assert!(json.contains("\"target_width\":1920"));
    }

    #[test]
    fn test_estimate_proxy_size() {
        let json = wasm_estimate_proxy_size(1_000_000, "quarter", "medium");
        assert!(json.contains("\"original_bytes\":1000000"));
        // quarter * quarter * medium = 0.25 * 0.25 * 0.25 = 0.015625 => ~15625
        assert!(json.contains("\"estimated_proxy_bytes\":15625"));
    }

    #[test]
    fn test_estimate_proxy_size_zero() {
        let json = wasm_estimate_proxy_size(0, "quarter", "medium");
        assert!(json.contains("\"original_bytes\":0"));
        assert!(json.contains("\"estimated_proxy_bytes\":0"));
        assert!(json.contains("\"savings_pct\":0"));
    }

    #[test]
    fn test_proxy_formats() {
        let json = wasm_proxy_formats();
        assert!(json.contains("vp9"));
        assert!(json.contains("av1"));
    }

    #[test]
    fn test_proxy_resolutions() {
        let json = wasm_proxy_resolutions();
        assert!(json.contains("quarter"));
        assert!(json.contains("half"));
        assert!(json.contains("full"));
    }
}
