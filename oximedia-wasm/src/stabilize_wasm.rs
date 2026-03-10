//! WebAssembly bindings for `oximedia-stabilize` video stabilization.
//!
//! Provides `WasmStabilizer` for incremental frame analysis and stabilization,
//! plus standalone functions for motion estimation.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// WasmStabilizer
// ---------------------------------------------------------------------------

/// Browser-side video stabilizer.
///
/// Usage:
/// 1. Create a stabilizer with frame dimensions and desired mode.
/// 2. Feed frames via `add_frame` — each returns JSON motion data.
/// 3. After all frames are added, retrieve smoothed transforms and apply them.
#[wasm_bindgen]
pub struct WasmStabilizer {
    width: u32,
    height: u32,
    mode: String,
    strength: f64,
    frames_analyzed: u32,
    // Raw grayscale pixels for previous frame (for motion estimation)
    prev_gray: Option<Vec<u8>>,
    // Accumulated per-frame motion vectors
    motion_dx: Vec<f64>,
    motion_dy: Vec<f64>,
    motion_rotation: Vec<f64>,
    motion_scale: Vec<f64>,
}

#[wasm_bindgen]
impl WasmStabilizer {
    /// Create a new WASM stabilizer.
    ///
    /// # Arguments
    /// * `width` - Frame width in pixels.
    /// * `height` - Frame height in pixels.
    /// * `mode` - Stabilization mode: "translation", "affine", "perspective", or "3d".
    /// * `strength` - Smoothing strength (0.0 to 1.0).
    ///
    /// # Errors
    /// Returns an error if parameters are invalid.
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: u32,
        height: u32,
        mode: &str,
        strength: f64,
    ) -> Result<WasmStabilizer, JsValue> {
        if width == 0 || height == 0 {
            return Err(crate::utils::js_err("Width and height must be > 0"));
        }
        if !(0.0..=1.0).contains(&strength) {
            return Err(crate::utils::js_err("Strength must be between 0.0 and 1.0"));
        }
        match mode {
            "translation" | "affine" | "perspective" | "3d" => {}
            _ => {
                return Err(crate::utils::js_err(&format!(
                    "Unknown mode '{}'. Expected: translation, affine, perspective, 3d",
                    mode
                )));
            }
        }

        Ok(Self {
            width,
            height,
            mode: mode.to_string(),
            strength,
            frames_analyzed: 0,
            prev_gray: None,
            motion_dx: Vec::new(),
            motion_dy: Vec::new(),
            motion_rotation: Vec::new(),
            motion_scale: Vec::new(),
        })
    }

    /// Add a frame for analysis.
    ///
    /// `frame_data` must contain `width * height * 3` bytes of RGB data.
    ///
    /// Returns a JSON string with motion data for this frame:
    /// ```json
    /// {"dx": 1.2, "dy": -0.5, "rotation": 0.002, "scale": 1.0, "confidence": 0.85}
    /// ```
    pub fn add_frame(&mut self, frame_data: &[u8]) -> Result<String, JsValue> {
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if frame_data.len() < expected {
            return Err(crate::utils::js_err(&format!(
                "Frame data too small: need {} bytes, got {}",
                expected,
                frame_data.len()
            )));
        }

        // Convert to grayscale
        let gray = rgb_to_gray(frame_data, self.width as usize, self.height as usize);

        let (dx, dy, rotation, scale, confidence) = if let Some(prev) = &self.prev_gray {
            estimate_motion_gray(prev, &gray, self.width as usize, self.height as usize)
        } else {
            (0.0, 0.0, 0.0, 1.0, 1.0)
        };

        self.motion_dx.push(dx);
        self.motion_dy.push(dy);
        self.motion_rotation.push(rotation);
        self.motion_scale.push(scale);
        self.prev_gray = Some(gray);
        self.frames_analyzed += 1;

        Ok(format!(
            "{{\"dx\":{dx},\"dy\":{dy},\"rotation\":{rotation},\"scale\":{scale},\"confidence\":{confidence}}}"
        ))
    }

    /// Get the smoothed stabilization transform for a frame.
    ///
    /// Returns JSON with the transform to apply:
    /// ```json
    /// {"dx": -1.0, "dy": 0.3, "rotation": -0.001, "scale": 1.0}
    /// ```
    pub fn get_transform(&self, frame_index: u32) -> Result<String, JsValue> {
        let idx = frame_index as usize;
        if idx >= self.motion_dx.len() {
            return Err(crate::utils::js_err(&format!(
                "Frame index {} out of range (0..{})",
                frame_index,
                self.motion_dx.len()
            )));
        }

        // Compute cumulative trajectory
        let n = self.motion_dx.len();
        let mut cum_dx = vec![0.0_f64; n];
        let mut cum_dy = vec![0.0_f64; n];
        let mut cum_rot = vec![0.0_f64; n];

        for i in 1..n {
            cum_dx[i] = cum_dx[i - 1] + self.motion_dx[i];
            cum_dy[i] = cum_dy[i - 1] + self.motion_dy[i];
            cum_rot[i] = cum_rot[i - 1] + self.motion_rotation[i];
        }

        // Apply Gaussian smoothing to the trajectory
        let smooth_dx = gaussian_smooth(&cum_dx, self.strength);
        let smooth_dy = gaussian_smooth(&cum_dy, self.strength);
        let smooth_rot = gaussian_smooth(&cum_rot, self.strength);

        // Transform = smoothed - original trajectory
        let tdx = smooth_dx[idx] - cum_dx[idx];
        let tdy = smooth_dy[idx] - cum_dy[idx];
        let trot = smooth_rot[idx] - cum_rot[idx];

        Ok(format!(
            "{{\"dx\":{tdx},\"dy\":{tdy},\"rotation\":{trot},\"scale\":1.0}}"
        ))
    }

    /// Apply the stabilization transform to a frame.
    ///
    /// `frame_data` must contain `width * height * 3` RGB bytes.
    /// Returns the stabilized frame as RGB bytes.
    pub fn apply_transform(&self, frame_data: &[u8], frame_index: u32) -> Result<Vec<u8>, JsValue> {
        let w = self.width as usize;
        let h = self.height as usize;
        let expected = w * h * 3;
        if frame_data.len() < expected {
            return Err(crate::utils::js_err(&format!(
                "Frame data too small: need {} bytes, got {}",
                expected,
                frame_data.len()
            )));
        }

        let idx = frame_index as usize;
        if idx >= self.motion_dx.len() {
            return Err(crate::utils::js_err(&format!(
                "Frame index {} out of range (0..{})",
                frame_index,
                self.motion_dx.len()
            )));
        }

        // Get the transform for this frame
        let transform_json = self.get_transform(frame_index)?;
        let (tdx, tdy) = parse_transform_dx_dy(&transform_json);

        // Apply translation transform to the frame
        let mut output = vec![0u8; expected];
        let dx_i = tdx.round() as isize;
        let dy_i = tdy.round() as isize;

        for y in 0..h {
            for x in 0..w {
                let src_x = x as isize - dx_i;
                let src_y = y as isize - dy_i;

                if src_x >= 0 && src_x < w as isize && src_y >= 0 && src_y < h as isize {
                    let src_idx = ((src_y as usize) * w + src_x as usize) * 3;
                    let dst_idx = (y * w + x) * 3;
                    output[dst_idx] = frame_data[src_idx];
                    output[dst_idx + 1] = frame_data[src_idx + 1];
                    output[dst_idx + 2] = frame_data[src_idx + 2];
                }
            }
        }

        Ok(output)
    }

    /// Get the number of frames that have been analyzed.
    pub fn frames_analyzed(&self) -> u32 {
        self.frames_analyzed
    }

    /// Reset the stabilizer, clearing all accumulated data.
    pub fn reset(&mut self) {
        self.frames_analyzed = 0;
        self.prev_gray = None;
        self.motion_dx.clear();
        self.motion_dy.clear();
        self.motion_rotation.clear();
        self.motion_scale.clear();
    }

    /// Get a JSON summary of the accumulated motion statistics.
    ///
    /// Returns:
    /// ```json
    /// {
    ///   "frames_analyzed": 120,
    ///   "avg_dx": 0.5,
    ///   "avg_dy": -0.2,
    ///   "max_dx": 5.1,
    ///   "max_dy": 3.2,
    ///   "avg_rotation": 0.001,
    ///   "mode": "affine",
    ///   "strength": 0.8
    /// }
    /// ```
    pub fn motion_summary(&self) -> Result<String, JsValue> {
        let n = self.motion_dx.len();
        if n == 0 {
            return Ok("{\"frames_analyzed\":0}".to_string());
        }

        let nf = n as f64;
        let avg_dx: f64 = self.motion_dx.iter().sum::<f64>() / nf;
        let avg_dy: f64 = self.motion_dy.iter().sum::<f64>() / nf;
        let max_dx = self
            .motion_dx
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        let max_dy = self
            .motion_dy
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        let avg_rot: f64 = self.motion_rotation.iter().sum::<f64>() / nf;

        Ok(format!(
            "{{\"frames_analyzed\":{},\"avg_dx\":{avg_dx:.4},\"avg_dy\":{avg_dy:.4},\
             \"max_dx\":{max_dx:.4},\"max_dy\":{max_dy:.4},\
             \"avg_rotation\":{avg_rot:.6},\"mode\":\"{}\",\"strength\":{}}}",
            self.frames_analyzed, self.mode, self.strength
        ))
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Estimate motion between two RGB frames.
///
/// Both frames must be `width * height * 3` bytes.
/// Returns a JSON string with dx, dy, rotation, scale, and confidence.
#[wasm_bindgen]
pub fn wasm_estimate_motion(
    frame1: &[u8],
    frame2: &[u8],
    width: u32,
    height: u32,
) -> Result<String, JsValue> {
    let w = width as usize;
    let h = height as usize;
    let expected = w * h * 3;

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

    let g1 = rgb_to_gray(frame1, w, h);
    let g2 = rgb_to_gray(frame2, w, h);
    let (dx, dy, rotation, scale, confidence) = estimate_motion_gray(&g1, &g2, w, h);

    Ok(format!(
        "{{\"dx\":{dx},\"dy\":{dy},\"rotation\":{rotation},\"scale\":{scale},\"confidence\":{confidence}}}"
    ))
}

/// List available stabilization modes as a JSON array.
#[wasm_bindgen]
pub fn wasm_stabilization_modes() -> Result<String, JsValue> {
    Ok("[\"translation\",\"affine\",\"perspective\",\"3d\"]".to_string())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Convert RGB bytes to grayscale using BT.601 luma coefficients.
fn rgb_to_gray(rgb: &[u8], width: usize, height: usize) -> Vec<u8> {
    let n = width * height;
    let mut gray = Vec::with_capacity(n);
    for pixel in rgb[..n * 3].chunks_exact(3) {
        let r = pixel[0] as f64;
        let g = pixel[1] as f64;
        let b = pixel[2] as f64;
        let luma = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
        gray.push(luma);
    }
    gray
}

/// Block-matching motion estimation on grayscale frames.
///
/// Returns (dx, dy, rotation, scale, confidence).
fn estimate_motion_gray(
    prev: &[u8],
    curr: &[u8],
    width: usize,
    height: usize,
) -> (f64, f64, f64, f64, f64) {
    let block_size: usize = 16;
    let search_range: isize = 8;

    let mut sum_dx: f64 = 0.0;
    let mut sum_dy: f64 = 0.0;
    let mut count: f64 = 0.0;

    for by in (0..height).step_by(block_size) {
        for bx in (0..width).step_by(block_size) {
            let bw = block_size.min(width - bx);
            let bh = block_size.min(height - by);

            let mut best_dx: isize = 0;
            let mut best_dy: isize = 0;
            let mut best_sad = u64::MAX;

            for sy in -search_range..=search_range {
                for sx in -search_range..=search_range {
                    let tx = bx as isize + sx;
                    let ty = by as isize + sy;
                    if tx < 0 || ty < 0 {
                        continue;
                    }
                    let tx_u = tx as usize;
                    let ty_u = ty as usize;
                    if tx_u + bw > width || ty_u + bh > height {
                        continue;
                    }

                    let mut sad: u64 = 0;
                    for yy in 0..bh {
                        for xx in 0..bw {
                            let p_idx = (by + yy) * width + (bx + xx);
                            let c_idx = (ty_u + yy) * width + (tx_u + xx);
                            let p = prev[p_idx] as i32;
                            let c = curr[c_idx] as i32;
                            sad += (p - c).unsigned_abs() as u64;
                        }
                    }

                    if sad < best_sad {
                        best_sad = sad;
                        best_dx = sx;
                        best_dy = sy;
                    }
                }
            }

            sum_dx += best_dx as f64;
            sum_dy += best_dy as f64;
            count += 1.0;
        }
    }

    let dx = if count > 0.0 { sum_dx / count } else { 0.0 };
    let dy = if count > 0.0 { sum_dy / count } else { 0.0 };

    let motion_mag = (dx * dx + dy * dy).sqrt();
    let confidence = if motion_mag < 0.5 {
        0.95
    } else if motion_mag < 5.0 {
        0.8
    } else {
        0.6
    };

    (dx, dy, 0.0, 1.0, confidence)
}

/// Simple 1-D Gaussian smoothing on a trajectory.
///
/// `strength` controls the smoothing kernel radius:
/// radius = (strength * 30).max(1).
fn gaussian_smooth(values: &[f64], strength: f64) -> Vec<f64> {
    let n = values.len();
    if n <= 1 {
        return values.to_vec();
    }

    let radius = ((strength * 30.0) as usize).max(1).min(n);
    let sigma = radius as f64 / 3.0;
    let sigma2 = 2.0 * sigma * sigma;

    let mut kernel = Vec::with_capacity(radius * 2 + 1);
    let mut kernel_sum = 0.0_f64;
    for i in 0..=(radius * 2) {
        let x = i as f64 - radius as f64;
        let w = (-x * x / sigma2).exp();
        kernel.push(w);
        kernel_sum += w;
    }
    // Normalize
    for w in &mut kernel {
        *w /= kernel_sum;
    }

    let mut smoothed = Vec::with_capacity(n);
    for i in 0..n {
        let mut val = 0.0_f64;
        for (k_idx, &kw) in kernel.iter().enumerate() {
            let j = i as isize + k_idx as isize - radius as isize;
            let j_clamped = j.max(0).min(n as isize - 1) as usize;
            val += values[j_clamped] * kw;
        }
        smoothed.push(val);
    }

    smoothed
}

/// Parse dx and dy from a transform JSON string.
fn parse_transform_dx_dy(json: &str) -> (f64, f64) {
    // Simple manual parse for our known format
    let dx = extract_json_f64(json, "\"dx\":");
    let dy = extract_json_f64(json, "\"dy\":");
    (dx, dy)
}

fn extract_json_f64(json: &str, key: &str) -> f64 {
    if let Some(start) = json.find(key) {
        let rest = &json[start + key.len()..];
        let end = rest.find([',', '}']).unwrap_or(rest.len());
        rest[..end].trim().parse::<f64>().unwrap_or(0.0)
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rgb(w: u32, h: u32, val: u8) -> Vec<u8> {
        vec![val; (w as usize) * (h as usize) * 3]
    }

    #[test]
    fn test_stabilizer_creation() {
        let s = WasmStabilizer::new(64, 64, "affine", 0.8);
        assert!(s.is_ok());
    }

    #[test]
    fn test_stabilizer_invalid_mode() {
        let s = WasmStabilizer::new(64, 64, "bad", 0.8);
        assert!(s.is_err());
    }

    #[test]
    fn test_stabilizer_invalid_strength() {
        let s = WasmStabilizer::new(64, 64, "affine", 2.0);
        assert!(s.is_err());
    }

    #[test]
    fn test_add_frame_and_summary() {
        let mut s = WasmStabilizer::new(8, 8, "translation", 0.5).expect("creation should succeed");
        let frame = make_rgb(8, 8, 128);
        let r1 = s.add_frame(&frame);
        assert!(r1.is_ok());
        let r2 = s.add_frame(&frame);
        assert!(r2.is_ok());
        assert_eq!(s.frames_analyzed(), 2);

        let summary = s.motion_summary();
        assert!(summary.is_ok());
        let json = summary.expect("summary should succeed");
        assert!(json.contains("\"frames_analyzed\":2"));
    }

    #[test]
    fn test_reset() {
        let mut s = WasmStabilizer::new(8, 8, "affine", 0.8).expect("creation should succeed");
        let frame = make_rgb(8, 8, 100);
        let _ = s.add_frame(&frame);
        s.reset();
        assert_eq!(s.frames_analyzed(), 0);
    }

    #[test]
    fn test_wasm_estimate_motion_identical() {
        let frame = make_rgb(16, 16, 80);
        let result = wasm_estimate_motion(&frame, &frame, 16, 16);
        assert!(result.is_ok());
    }

    #[test]
    fn test_wasm_stabilization_modes() {
        let modes = wasm_stabilization_modes();
        assert!(modes.is_ok());
        let json = modes.expect("modes should succeed");
        assert!(json.contains("affine"));
        assert!(json.contains("translation"));
    }

    #[test]
    fn test_gaussian_smooth_identity() {
        let values = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let smoothed = gaussian_smooth(&values, 0.5);
        for v in &smoothed {
            assert!((*v - 1.0).abs() < 0.01);
        }
    }
}
