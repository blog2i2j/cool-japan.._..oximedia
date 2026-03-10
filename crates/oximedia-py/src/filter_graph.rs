//! Python bindings for the OxiMedia filter graph.
//!
//! Provides a high-level Python API for building and applying filter chains
//! to video frames and audio samples.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// A filter node specification with name and key/value parameters.
#[pyclass]
#[derive(Clone)]
pub struct PyFilterNode {
    /// Filter name (e.g., "scale", "crop", "volume").
    #[pyo3(get)]
    pub name: String,
    /// Filter parameters as string key/value pairs.
    #[pyo3(get)]
    pub params: HashMap<String, String>,
}

#[pymethods]
impl PyFilterNode {
    /// Create a new filter node.
    ///
    /// # Arguments
    /// * `name` - Filter name
    /// * `params` - Optional keyword parameters
    #[new]
    #[pyo3(signature = (name, **params))]
    fn new(name: String, params: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut map = HashMap::new();
        if let Some(dict) = params {
            for (k, v) in dict.iter() {
                let key: String = k.extract()?;
                let val: String = v.str()?.to_string();
                map.insert(key, val);
            }
        }
        Ok(Self { name, params: map })
    }

    fn __repr__(&self) -> String {
        if self.params.is_empty() {
            format!("PyFilterNode({})", self.name)
        } else {
            let params_str = self
                .params
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join(", ");
            format!("PyFilterNode({}, {})", self.name, params_str)
        }
    }
}

/// Filter graph for composing media processing pipelines.
///
/// Supports video operations (scale, crop) and audio operations (volume).
///
/// # Example (Python)
/// ```python
/// import oximedia
/// fg = oximedia.FilterGraph()
/// fg.scale(1280, 720)
/// fg.volume(-6.0)
/// processed = fg.process_video(frame_bytes, 1920, 1080)
/// ```
#[pyclass]
pub struct FilterGraph {
    nodes: Vec<PyFilterNode>,
}

#[pymethods]
impl FilterGraph {
    /// Create an empty filter graph.
    #[new]
    fn new() -> Self {
        FilterGraph { nodes: Vec::new() }
    }

    /// Add a generic filter node by name and optional parameters.
    #[pyo3(signature = (name, params=None))]
    fn add(&mut self, name: &str, params: Option<HashMap<String, String>>) -> PyResult<()> {
        let node = PyFilterNode {
            name: name.to_string(),
            params: params.unwrap_or_default(),
        };
        self.nodes.push(node);
        Ok(())
    }

    /// Add a scale filter to resize video frames.
    ///
    /// # Arguments
    /// * `width` - Target width in pixels
    /// * `height` - Target height in pixels
    fn scale(&mut self, width: u32, height: u32) -> PyResult<()> {
        let mut params = HashMap::new();
        params.insert("width".to_string(), width.to_string());
        params.insert("height".to_string(), height.to_string());
        self.nodes.push(PyFilterNode {
            name: "scale".to_string(),
            params,
        });
        Ok(())
    }

    /// Add a crop filter to extract a region from video frames.
    ///
    /// # Arguments
    /// * `x` - Left offset in pixels
    /// * `y` - Top offset in pixels
    /// * `width` - Crop width in pixels
    /// * `height` - Crop height in pixels
    fn crop(&mut self, x: u32, y: u32, width: u32, height: u32) -> PyResult<()> {
        let mut params = HashMap::new();
        params.insert("x".to_string(), x.to_string());
        params.insert("y".to_string(), y.to_string());
        params.insert("width".to_string(), width.to_string());
        params.insert("height".to_string(), height.to_string());
        self.nodes.push(PyFilterNode {
            name: "crop".to_string(),
            params,
        });
        Ok(())
    }

    /// Add a volume filter for audio gain adjustment.
    ///
    /// # Arguments
    /// * `gain_db` - Gain in decibels (negative = attenuate, positive = amplify)
    fn volume(&mut self, gain_db: f32) -> PyResult<()> {
        let mut params = HashMap::new();
        params.insert("gain_db".to_string(), gain_db.to_string());
        self.nodes.push(PyFilterNode {
            name: "volume".to_string(),
            params,
        });
        Ok(())
    }

    /// Process an RGB24 video frame through the filter chain.
    ///
    /// Input must be raw RGB24 bytes: `width * height * 3` bytes.
    /// Returns processed RGB24 bytes.
    ///
    /// Supported filters: scale, crop (passthrough for unsupported).
    fn process_video(&self, data: Vec<u8>, width: u32, height: u32) -> PyResult<Vec<u8>> {
        let mut current_data = data;
        let mut current_width = width;
        let mut current_height = height;

        for node in &self.nodes {
            match node.name.as_str() {
                "scale" => {
                    let target_w = node
                        .params
                        .get("width")
                        .and_then(|v| v.parse::<u32>().ok())
                        .unwrap_or(current_width);
                    let target_h = node
                        .params
                        .get("height")
                        .and_then(|v| v.parse::<u32>().ok())
                        .unwrap_or(current_height);

                    current_data = bilinear_scale_rgb(
                        &current_data,
                        current_width,
                        current_height,
                        target_w,
                        target_h,
                    )?;
                    current_width = target_w;
                    current_height = target_h;
                }
                "crop" => {
                    let x = node
                        .params
                        .get("x")
                        .and_then(|v| v.parse::<u32>().ok())
                        .unwrap_or(0);
                    let y = node
                        .params
                        .get("y")
                        .and_then(|v| v.parse::<u32>().ok())
                        .unwrap_or(0);
                    let crop_w = node
                        .params
                        .get("width")
                        .and_then(|v| v.parse::<u32>().ok())
                        .unwrap_or(current_width);
                    let crop_h = node
                        .params
                        .get("height")
                        .and_then(|v| v.parse::<u32>().ok())
                        .unwrap_or(current_height);

                    current_data = crop_rgb(
                        &current_data,
                        current_width,
                        current_height,
                        x,
                        y,
                        crop_w,
                        crop_h,
                    )?;
                    current_width = crop_w;
                    current_height = crop_h;
                }
                // Non-video filters and unknown filters are passthrough
                _ => {}
            }
        }

        let _ = (current_width, current_height); // suppress unused warnings
        Ok(current_data)
    }

    /// Process audio samples through the filter chain.
    ///
    /// Input: interleaved float32 samples.
    /// Returns processed float32 samples.
    ///
    /// Supported filters: volume (passthrough for unsupported).
    fn process_audio(&self, samples: Vec<f32>) -> PyResult<Vec<f32>> {
        let mut current = samples;

        for node in &self.nodes {
            if node.name == "volume" {
                let gain_db = node
                    .params
                    .get("gain_db")
                    .and_then(|v| v.parse::<f32>().ok())
                    .unwrap_or(0.0);

                // Convert dB to linear gain factor
                let linear_gain = 10.0_f32.powf(gain_db / 20.0);
                current = current.into_iter().map(|s| s * linear_gain).collect();
            }
            // Non-audio filters are passthrough
        }

        Ok(current)
    }

    /// Get the number of filter nodes in the graph.
    fn __len__(&self) -> usize {
        self.nodes.len()
    }

    /// Get a string representation of the filter chain.
    fn __repr__(&self) -> String {
        let chain = self
            .nodes
            .iter()
            .map(|n| n.name.as_str())
            .collect::<Vec<_>>()
            .join(" -> ");
        format!("FilterGraph({})", chain)
    }
}

/// Bilinear scaling for RGB24 frames.
///
/// Produces output of `target_w * target_h * 3` bytes.
fn bilinear_scale_rgb(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> PyResult<Vec<u8>> {
    if src_w == 0 || src_h == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Source dimensions must be non-zero",
        ));
    }
    if dst_w == 0 || dst_h == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Target dimensions must be non-zero",
        ));
    }

    let expected_src = (src_w * src_h * 3) as usize;
    if src.len() != expected_src {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Expected {} bytes for {}x{} RGB24, got {}",
            expected_src,
            src_w,
            src_h,
            src.len()
        )));
    }

    let src_w_f = src_w as f32;
    let src_h_f = src_h as f32;
    let dst_w_usize = dst_w as usize;
    let dst_h_usize = dst_h as usize;
    let src_w_usize = src_w as usize;

    let mut dst = vec![0u8; dst_w_usize * dst_h_usize * 3];

    for dy in 0..dst_h_usize {
        for dx in 0..dst_w_usize {
            // Map destination pixel to source coordinate
            let sx = (dx as f32 + 0.5) * src_w_f / dst_w as f32 - 0.5;
            let sy = (dy as f32 + 0.5) * src_h_f / dst_h as f32 - 0.5;

            let sx0 = sx.floor() as i32;
            let sy0 = sy.floor() as i32;
            let sx1 = sx0 + 1;
            let sy1 = sy0 + 1;

            let fx = sx - sx0 as f32;
            let fy = sy - sy0 as f32;

            // Clamp to valid source range
            let sx0c = sx0.clamp(0, src_w as i32 - 1) as usize;
            let sy0c = sy0.clamp(0, src_h as i32 - 1) as usize;
            let sx1c = sx1.clamp(0, src_w as i32 - 1) as usize;
            let sy1c = sy1.clamp(0, src_h as i32 - 1) as usize;

            for c in 0..3usize {
                let p00 = src[(sy0c * src_w_usize + sx0c) * 3 + c] as f32;
                let p10 = src[(sy0c * src_w_usize + sx1c) * 3 + c] as f32;
                let p01 = src[(sy1c * src_w_usize + sx0c) * 3 + c] as f32;
                let p11 = src[(sy1c * src_w_usize + sx1c) * 3 + c] as f32;

                let val = p00 * (1.0 - fx) * (1.0 - fy)
                    + p10 * fx * (1.0 - fy)
                    + p01 * (1.0 - fx) * fy
                    + p11 * fx * fy;

                dst[(dy * dst_w_usize + dx) * 3 + c] = val.clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(dst)
}

/// Crop a region from an RGB24 frame.
fn crop_rgb(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    x: u32,
    y: u32,
    crop_w: u32,
    crop_h: u32,
) -> PyResult<Vec<u8>> {
    let expected_src = (src_w * src_h * 3) as usize;
    if src.len() != expected_src {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Expected {} bytes for {}x{} RGB24, got {}",
            expected_src,
            src_w,
            src_h,
            src.len()
        )));
    }

    if x + crop_w > src_w || y + crop_h > src_h {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Crop region {}x{}+{}+{} exceeds source dimensions {}x{}",
            crop_w, crop_h, x, y, src_w, src_h
        )));
    }

    let src_w_usize = src_w as usize;
    let crop_w_usize = crop_w as usize;
    let crop_h_usize = crop_h as usize;
    let x_usize = x as usize;
    let y_usize = y as usize;

    let mut dst = vec![0u8; crop_w_usize * crop_h_usize * 3];

    for dy in 0..crop_h_usize {
        let sy = y_usize + dy;
        let src_row_start = (sy * src_w_usize + x_usize) * 3;
        let dst_row_start = dy * crop_w_usize * 3;
        let row_bytes = crop_w_usize * 3;
        dst[dst_row_start..dst_row_start + row_bytes]
            .copy_from_slice(&src[src_row_start..src_row_start + row_bytes]);
    }

    Ok(dst)
}
