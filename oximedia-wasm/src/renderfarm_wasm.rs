//! WebAssembly bindings for render farm cluster status and estimation.
//!
//! Provides WASM-accessible functions for querying farm status,
//! estimating render times, and listing node types.

use wasm_bindgen::prelude::*;

/// Get the current farm status as a JSON object.
///
/// Returns cluster overview including node counts and job states.
#[wasm_bindgen]
pub fn wasm_farm_status() -> String {
    let status = serde_json::json!({
        "cluster_status": "ready",
        "total_nodes": 0,
        "idle_nodes": 0,
        "busy_nodes": 0,
        "offline_nodes": 0,
        "job_states": {
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
        },
        "cluster_utilization": 0.0,
        "schedulers": ["round-robin", "least-loaded", "priority", "affinity"],
    });
    serde_json::to_string(&status).unwrap_or_else(|_| "{}".to_string())
}

/// Estimate render time for a given configuration.
///
/// `config_json`: JSON object with keys:
///   - `frame_count` (u32): number of frames to render
///   - `frame_complexity` (f64): complexity factor (0.0-1.0, default 0.5)
///   - `node_count` (u32): available render nodes
///   - `node_cores` (u32): cores per node (default 8)
///   - `tile_render` (bool): whether to use tile-based rendering (default false)
///
/// Returns a JSON object with estimated time in seconds and breakdown.
#[wasm_bindgen]
pub fn wasm_estimate_render_time(config_json: &str) -> Result<String, JsValue> {
    let config: serde_json::Value = serde_json::from_str(config_json)
        .map_err(|e| crate::utils::js_err(&format!("Invalid config JSON: {e}")))?;

    let frame_count = config
        .get("frame_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(100) as f64;
    let complexity = config
        .get("frame_complexity")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.5);
    let node_count = config
        .get("node_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as f64;
    let node_cores = config
        .get("node_cores")
        .and_then(|v| v.as_u64())
        .unwrap_or(8) as f64;
    let tile_render = config
        .get("tile_render")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if node_count <= 0.0 {
        return Err(crate::utils::js_err("node_count must be > 0"));
    }

    // Base time per frame: ~0.5s for simple, up to 5s for complex
    let base_time_per_frame = 0.5 + 4.5 * complexity;

    // Tile rendering reduces time by ~30% with overhead
    let tile_factor = if tile_render { 0.7 } else { 1.0 };

    // Parallelism factor: diminishing returns after ~8 nodes
    let effective_parallelism =
        node_count.min(frame_count) * (1.0 - 0.02 * (node_count - 1.0).max(0.0).min(20.0));

    // Core utilization factor
    let core_factor = 1.0 / (1.0 + (node_cores - 1.0).max(0.0) * 0.1);

    let total_render_time = (frame_count * base_time_per_frame * tile_factor * core_factor)
        / effective_parallelism.max(1.0);

    // Overhead: distribution, collection, and post-processing
    let overhead = frame_count * 0.01 + node_count * 0.5;

    let total_time = total_render_time + overhead;

    let result = serde_json::json!({
        "frame_count": frame_count as u32,
        "node_count": node_count as u32,
        "complexity": complexity,
        "tile_render": tile_render,
        "estimated_render_seconds": (total_render_time * 100.0).round() / 100.0,
        "estimated_overhead_seconds": (overhead * 100.0).round() / 100.0,
        "estimated_total_seconds": (total_time * 100.0).round() / 100.0,
        "estimated_total_minutes": ((total_time / 60.0) * 100.0).round() / 100.0,
    });

    serde_json::to_string(&result)
        .map_err(|e| crate::utils::js_err(&format!("Serialization error: {e}")))
}

/// List available farm node types and their capabilities.
///
/// Returns a JSON array of node type descriptors.
#[wasm_bindgen]
pub fn wasm_farm_node_types() -> String {
    let types = serde_json::json!([
        {
            "type": "cpu-general",
            "description": "General-purpose CPU render node",
            "typical_cores": 16,
            "typical_memory_gb": 64,
            "gpu": false,
            "best_for": ["video_transcode", "audio_processing", "thumbnail_generation"],
        },
        {
            "type": "cpu-high-mem",
            "description": "High-memory CPU render node",
            "typical_cores": 32,
            "typical_memory_gb": 256,
            "gpu": false,
            "best_for": ["8k_processing", "multi_layer_compositing", "batch_analysis"],
        },
        {
            "type": "gpu-render",
            "description": "GPU-accelerated render node",
            "typical_cores": 16,
            "typical_memory_gb": 64,
            "gpu": true,
            "best_for": ["av1_encode", "video_effects", "ml_inference"],
        },
        {
            "type": "gpu-encode",
            "description": "Hardware encoder node",
            "typical_cores": 8,
            "typical_memory_gb": 32,
            "gpu": true,
            "best_for": ["realtime_transcode", "streaming_encode", "proxy_generation"],
        },
        {
            "type": "storage",
            "description": "Storage-optimized node for asset distribution",
            "typical_cores": 8,
            "typical_memory_gb": 32,
            "gpu": false,
            "best_for": ["asset_distribution", "cache_management", "archival"],
        },
    ]);
    serde_json::to_string(&types).unwrap_or_else(|_| "[]".to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_farm_status() {
        let status = wasm_farm_status();
        assert!(status.contains("cluster_status"));
        assert!(status.contains("ready"));
    }

    #[test]
    fn test_estimate_render_time() {
        let config = r#"{"frame_count":1000,"node_count":4,"frame_complexity":0.5}"#;
        let result = wasm_estimate_render_time(config);
        assert!(result.is_ok());
        let json = result.expect("should estimate");
        assert!(json.contains("estimated_total_seconds"));
    }

    #[test]
    fn test_estimate_render_time_tile() {
        let config = r#"{"frame_count":500,"node_count":2,"tile_render":true}"#;
        let result = wasm_estimate_render_time(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_estimate_render_time_invalid() {
        let result = wasm_estimate_render_time("not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_farm_node_types() {
        let types = wasm_farm_node_types();
        assert!(types.contains("cpu-general"));
        assert!(types.contains("gpu-render"));
        assert!(types.contains("storage"));
    }
}
