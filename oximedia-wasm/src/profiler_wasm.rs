//! WebAssembly bindings for performance profiling utilities.
//!
//! Provides functions for profiling operations, benchmarking codecs,
//! and generating bottleneck reports in the browser.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Profile a media operation in the browser.
///
/// # Arguments
/// * `operation` - Operation type: encode, decode, filter, pipeline.
/// * `codec` - Codec being profiled.
/// * `width` - Frame width.
/// * `height` - Frame height.
/// * `frames` - Number of frames.
///
/// # Returns
/// JSON string with profiling results.
#[wasm_bindgen]
pub fn wasm_profile_operation(
    operation: &str,
    codec: &str,
    width: u32,
    height: u32,
    frames: u32,
) -> Result<String, JsValue> {
    let valid_ops = ["encode", "decode", "filter", "pipeline"];
    if !valid_ops.contains(&operation) {
        return Err(crate::utils::js_err(&format!(
            "Unknown operation '{}'. Supported: encode, decode, filter, pipeline",
            operation
        )));
    }

    let valid_codecs = ["av1", "vp9", "vp8", "opus", "vorbis", "flac"];
    if !valid_codecs.contains(&codec) {
        return Err(crate::utils::js_err(&format!(
            "Unsupported codec '{}'. Patent-free only: av1, vp9, vp8, opus, vorbis, flac",
            codec
        )));
    }

    if width == 0 || height == 0 {
        return Err(crate::utils::js_err("Width and height must be > 0"));
    }

    let pixels_per_frame = width as u64 * height as u64;
    let total_pixels = pixels_per_frame * frames as u64;

    // Estimated throughput based on codec complexity
    let est_fps = match codec {
        "av1" => 15.0,
        "vp9" => 30.0,
        "vp8" => 60.0,
        _ => 100.0, // audio codecs
    };

    let est_time_ms = if est_fps > 0.0 {
        (frames as f64 / est_fps) * 1000.0
    } else {
        0.0
    };

    Ok(format!(
        "{{\"operation\":\"{operation}\",\"codec\":\"{codec}\",\"resolution\":\"{}x{}\",\"frames\":{frames},\"total_pixels\":{total_pixels},\"estimated_fps\":{est_fps:.1},\"estimated_time_ms\":{est_time_ms:.1}}}",
        width, height
    ))
}

/// Benchmark a codec with multiple configuration sets.
///
/// # Arguments
/// * `codec` - Codec to benchmark.
/// * `presets` - Comma-separated presets to test.
///
/// # Returns
/// JSON array of benchmark results.
#[wasm_bindgen]
pub fn wasm_benchmark_codec(codec: &str, presets: &str) -> Result<String, JsValue> {
    let valid_codecs = ["av1", "vp9", "vp8", "opus", "vorbis", "flac"];
    if !valid_codecs.contains(&codec) {
        return Err(crate::utils::js_err(&format!(
            "Unsupported codec '{}'. Patent-free only.",
            codec
        )));
    }

    let preset_list: Vec<&str> = presets.split(',').map(|s| s.trim()).collect();
    let mut results = Vec::new();

    for (i, preset) in preset_list.iter().enumerate() {
        // Simulated benchmark scores
        let speed_score = 1.0 - (i as f64 * 0.15).min(0.9);
        let quality_score = 0.5 + (i as f64 * 0.1).min(0.45);
        results.push(format!(
            "{{\"preset\":\"{preset}\",\"speed_score\":{speed_score:.3},\"quality_score\":{quality_score:.3}}}"
        ));
    }

    Ok(format!("[{}]", results.join(",")))
}

/// Generate a bottleneck report for a pipeline configuration.
///
/// # Arguments
/// * `pipeline_json` - JSON describing the pipeline (or simple description).
///
/// # Returns
/// JSON report of detected bottlenecks.
#[wasm_bindgen]
pub fn wasm_bottleneck_report(pipeline_json: &str) -> String {
    // Parse pipeline description or use as-is
    let has_encode = pipeline_json.contains("encode");
    let has_decode = pipeline_json.contains("decode");
    let has_filter = pipeline_json.contains("filter");

    let mut bottlenecks = Vec::new();

    if has_encode {
        bottlenecks.push(
            "{\"component\":\"encoder\",\"severity\":0.7,\"suggestion\":\"Use faster preset or reduce CRF\"}".to_string()
        );
    }
    if has_decode {
        bottlenecks.push(
            "{\"component\":\"decoder\",\"severity\":0.3,\"suggestion\":\"Enable threading for parallel decode\"}".to_string()
        );
    }
    if has_filter {
        bottlenecks.push(
            "{\"component\":\"filter\",\"severity\":0.5,\"suggestion\":\"Reduce filter chain complexity\"}".to_string()
        );
    }
    if bottlenecks.is_empty() {
        bottlenecks.push(
            "{\"component\":\"none\",\"severity\":0.0,\"suggestion\":\"No bottlenecks detected\"}"
                .to_string(),
        );
    }

    format!("{{\"bottlenecks\":[{}]}}", bottlenecks.join(","))
}

/// Get supported profiling modes as JSON.
#[wasm_bindgen]
pub fn wasm_profiling_modes() -> String {
    "[\"sampling\",\"instrumentation\",\"event-based\",\"continuous\"]".to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_operation() {
        let result = wasm_profile_operation("encode", "av1", 1920, 1080, 100);
        assert!(result.is_ok());
        let json = result.expect("should succeed");
        assert!(json.contains("encode"));
        assert!(json.contains("av1"));
    }

    #[test]
    fn test_profile_invalid_op() {
        let result = wasm_profile_operation("invalid", "av1", 1920, 1080, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_benchmark_codec() {
        let result = wasm_benchmark_codec("av1", "fast,medium,slow");
        assert!(result.is_ok());
        let json = result.expect("should succeed");
        assert!(json.contains("fast"));
        assert!(json.contains("slow"));
    }

    #[test]
    fn test_bottleneck_report() {
        let report = wasm_bottleneck_report("decode -> filter -> encode");
        assert!(report.contains("encoder"));
        assert!(report.contains("decoder"));
        assert!(report.contains("filter"));
    }

    #[test]
    fn test_profiling_modes() {
        let modes = wasm_profiling_modes();
        assert!(modes.contains("sampling"));
    }
}
