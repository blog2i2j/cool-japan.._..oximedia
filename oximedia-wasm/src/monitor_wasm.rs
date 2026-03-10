//! WebAssembly bindings for `oximedia-monitor` system monitoring.
//!
//! Provides stream health checking, signal quality analysis, and
//! a stateful stream monitor for browser-based monitoring dashboards.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn escape_json_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

// ---------------------------------------------------------------------------
// Health check
// ---------------------------------------------------------------------------

/// Check stream health by analyzing the given data buffer.
///
/// Returns JSON:
/// ```json
/// {
///   "status": "healthy",
///   "data_size": 1024,
///   "has_data": true,
///   "entropy": 7.5,
///   "zero_ratio": 0.01
/// }
/// ```
///
/// # Errors
///
/// Returns an error if the data is empty.
#[wasm_bindgen]
pub fn wasm_check_stream_health(data: &[u8]) -> Result<String, JsValue> {
    if data.is_empty() {
        return Err(crate::utils::js_err("Empty data buffer"));
    }

    let has_data = !data.is_empty();
    let data_size = data.len();

    // Compute entropy
    let mut histogram = [0u64; 256];
    for &b in data {
        histogram[b as usize] += 1;
    }
    let total = data.len() as f64;
    let entropy: f64 = histogram
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total;
            -p * p.log2()
        })
        .sum();

    // Zero ratio
    #[allow(clippy::naive_bytecount)]
    let zero_count = data.iter().filter(|&&b| b == 0).count();
    let zero_ratio = zero_count as f64 / total;

    let status = if entropy > 1.0 && zero_ratio < 0.95 {
        "healthy"
    } else {
        "degraded"
    };

    Ok(format!(
        "{{\"status\":\"{status}\",\"data_size\":{data_size},\"has_data\":{has_data},\
         \"entropy\":{entropy:.4},\"zero_ratio\":{zero_ratio:.6}}}",
    ))
}

// ---------------------------------------------------------------------------
// Signal quality analysis
// ---------------------------------------------------------------------------

/// Analyze signal quality of a video frame (raw RGB bytes).
///
/// `data` is expected to be raw pixel data (width * height * 3 bytes for RGB).
/// `width` and `height` specify the frame dimensions.
///
/// Returns JSON:
/// ```json
/// {
///   "width": 1920,
///   "height": 1080,
///   "avg_brightness": 128.5,
///   "contrast": 45.2,
///   "noise_estimate": 3.1,
///   "quality_score": 85.0
/// }
/// ```
///
/// # Errors
///
/// Returns an error if dimensions do not match data size.
#[wasm_bindgen]
pub fn wasm_analyze_signal_quality(
    data: &[u8],
    width: u32,
    height: u32,
) -> Result<String, JsValue> {
    let expected = (width as usize) * (height as usize) * 3;
    if data.len() < expected && expected > 0 {
        return Err(crate::utils::js_err(&format!(
            "Data size {} < expected {} ({}x{}x3)",
            data.len(),
            expected,
            width,
            height,
        )));
    }

    if data.is_empty() {
        return Err(crate::utils::js_err("Empty frame data"));
    }

    // Calculate per-pixel luminance (simplified BT.601)
    let pixel_count = width as usize * height as usize;
    let mut luma_sum: f64 = 0.0;
    let mut luma_sq_sum: f64 = 0.0;
    let mut min_luma: f64 = 255.0;
    let mut max_luma: f64 = 0.0;

    for i in 0..pixel_count.min(data.len() / 3) {
        let r = data[i * 3] as f64;
        let g = data[i * 3 + 1] as f64;
        let b = data[i * 3 + 2] as f64;
        let luma = 0.299 * r + 0.587 * g + 0.114 * b;
        luma_sum += luma;
        luma_sq_sum += luma * luma;
        if luma < min_luma {
            min_luma = luma;
        }
        if luma > max_luma {
            max_luma = luma;
        }
    }

    let count = pixel_count.max(1) as f64;
    let avg_brightness = luma_sum / count;
    let variance = (luma_sq_sum / count) - (avg_brightness * avg_brightness);
    let contrast = max_luma - min_luma;

    // Noise estimate: use local variance proxy
    let noise_estimate = variance.sqrt().min(50.0);

    // Quality score: heuristic combining brightness, contrast, noise
    let brightness_score = 1.0 - ((avg_brightness - 128.0).abs() / 128.0).min(1.0);
    let contrast_score = (contrast / 255.0).min(1.0);
    let noise_score = 1.0 - (noise_estimate / 50.0).min(1.0);
    let quality_score =
        ((brightness_score * 30.0) + (contrast_score * 40.0) + (noise_score * 30.0))
            .clamp(0.0, 100.0);

    Ok(format!(
        "{{\"width\":{width},\"height\":{height},\"avg_brightness\":{avg_brightness:.1},\
         \"contrast\":{contrast:.1},\"noise_estimate\":{noise_estimate:.1},\
         \"quality_score\":{quality_score:.1}}}",
    ))
}

// ---------------------------------------------------------------------------
// WasmStreamMonitor
// ---------------------------------------------------------------------------

/// Stateful stream monitor that accumulates samples and tracks health.
#[wasm_bindgen]
pub struct WasmStreamMonitor {
    sample_count: u64,
    total_bytes: u64,
    error_count: u64,
    last_entropy: f64,
    health_status: String,
}

#[wasm_bindgen]
impl WasmStreamMonitor {
    /// Create a new stream monitor.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            sample_count: 0,
            total_bytes: 0,
            error_count: 0,
            last_entropy: 0.0,
            health_status: "unknown".to_string(),
        }
    }

    /// Add a data sample to the monitor.
    ///
    /// Updates health metrics based on the new data.
    pub fn add_sample(&mut self, data: &[u8]) {
        self.sample_count += 1;
        self.total_bytes += data.len() as u64;

        if data.is_empty() {
            self.error_count += 1;
            self.health_status = "error".to_string();
            return;
        }

        // Compute entropy
        let mut histogram = [0u64; 256];
        for &b in data {
            histogram[b as usize] += 1;
        }
        let total = data.len() as f64;
        let entropy: f64 = histogram
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / total;
                -p * p.log2()
            })
            .sum();

        self.last_entropy = entropy;
        self.health_status = if entropy > 1.0 {
            "healthy".to_string()
        } else {
            "degraded".to_string()
        };
    }

    /// Get the current status as a JSON string.
    pub fn get_status(&self) -> String {
        format!(
            "{{\"health\":\"{}\",\"samples\":{},\"total_bytes\":{},\
             \"errors\":{},\"last_entropy\":{:.4}}}",
            escape_json_string(&self.health_status),
            self.sample_count,
            self.total_bytes,
            self.error_count,
            self.last_entropy,
        )
    }

    /// Get recent alerts as a JSON array.
    ///
    /// Returns alerts based on current health status.
    pub fn alerts(&self) -> String {
        let mut alerts = Vec::new();

        if self.health_status == "error" {
            alerts.push(format!(
                "{{\"severity\":\"error\",\"message\":\"Stream error: empty data received\"}}",
            ));
        }

        if self.health_status == "degraded" {
            alerts.push(format!(
                "{{\"severity\":\"warning\",\"message\":\"Low entropy ({:.2}): possible signal degradation\"}}",
                self.last_entropy,
            ));
        }

        if self.error_count > 0 {
            alerts.push(format!(
                "{{\"severity\":\"info\",\"message\":\"Total errors: {}\"}}",
                self.error_count,
            ));
        }

        format!("[{}]", alerts.join(","))
    }
}

impl Default for WasmStreamMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_stream_health_healthy() {
        let data: Vec<u8> = (0..=255).collect();
        let result = wasm_check_stream_health(&data);
        assert!(result.is_ok());
        let json = result.expect("should succeed");
        assert!(json.contains("\"status\":\"healthy\""));
    }

    #[test]
    fn test_check_stream_health_empty() {
        let result = wasm_check_stream_health(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_analyze_signal_quality() {
        // 2x2 RGB frame
        let data = vec![128u8; 2 * 2 * 3];
        let result = wasm_analyze_signal_quality(&data, 2, 2);
        assert!(result.is_ok());
        let json = result.expect("should succeed");
        assert!(json.contains("\"width\":2"));
        assert!(json.contains("\"height\":2"));
    }

    #[test]
    fn test_stream_monitor_lifecycle() {
        let mut monitor = WasmStreamMonitor::new();
        assert!(monitor.get_status().contains("\"health\":\"unknown\""));

        monitor.add_sample(&(0..=255).collect::<Vec<u8>>());
        assert!(monitor.get_status().contains("\"health\":\"healthy\""));

        monitor.add_sample(&[]);
        assert!(monitor.get_status().contains("\"health\":\"error\""));
        assert!(monitor.alerts().contains("error"));
    }

    #[test]
    fn test_stream_monitor_alerts_empty() {
        let monitor = WasmStreamMonitor::new();
        let alerts = monitor.alerts();
        assert_eq!(alerts, "[]");
    }
}
