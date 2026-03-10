//! Python bindings for `oximedia-stabilize` video stabilization.
//!
//! Provides `PyStabilizeConfig`, `PyMotionData`, `PyStabilizer`, and
//! standalone convenience functions for stabilizing video frames from Python.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::collections::HashMap;

use ndarray::Array2;
use oximedia_stabilize::{Frame, QualityPreset, StabilizationMode, StabilizeConfig, Stabilizer};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_mode(mode: &str) -> PyResult<StabilizationMode> {
    match mode {
        "translation" => Ok(StabilizationMode::Translation),
        "affine" => Ok(StabilizationMode::Affine),
        "perspective" => Ok(StabilizationMode::Perspective),
        "3d" | "three_d" => Ok(StabilizationMode::ThreeD),
        other => Err(PyValueError::new_err(format!(
            "Unknown stabilization mode '{}'. Expected: translation, affine, perspective, 3d",
            other
        ))),
    }
}

fn mode_to_string(mode: StabilizationMode) -> String {
    match mode {
        StabilizationMode::Translation => "translation".to_string(),
        StabilizationMode::Affine => "affine".to_string(),
        StabilizationMode::Perspective => "perspective".to_string(),
        StabilizationMode::ThreeD => "3d".to_string(),
    }
}

fn parse_quality(preset: &str) -> PyResult<QualityPreset> {
    match preset {
        "fast" => Ok(QualityPreset::Fast),
        "balanced" => Ok(QualityPreset::Balanced),
        "maximum" => Ok(QualityPreset::Maximum),
        other => Err(PyValueError::new_err(format!(
            "Unknown quality preset '{}'. Expected: fast, balanced, maximum",
            other
        ))),
    }
}

fn quality_to_string(q: QualityPreset) -> String {
    match q {
        QualityPreset::Fast => "fast".to_string(),
        QualityPreset::Balanced => "balanced".to_string(),
        QualityPreset::Maximum => "maximum".to_string(),
    }
}

fn build_internal_config(py_cfg: &PyStabilizeConfig) -> PyResult<StabilizeConfig> {
    let mode = parse_mode(&py_cfg.mode)?;
    let quality = parse_quality(&py_cfg.quality_preset)?;

    let config = StabilizeConfig::new()
        .with_mode(mode)
        .with_quality(quality)
        .with_smoothing_strength(py_cfg.strength)
        .with_zoom_optimization(py_cfg.enable_zoom_optimization)
        .with_horizon_leveling(py_cfg.enable_horizon_leveling);

    Ok(config)
}

/// Convert raw RGB bytes into a grayscale stabilizer `Frame`.
fn rgb_bytes_to_frame(data: &[u8], width: u32, height: u32, timestamp: f64) -> PyResult<Frame> {
    let w = width as usize;
    let h = height as usize;
    let expected = w * h * 3;
    if data.len() < expected {
        return Err(PyValueError::new_err(format!(
            "Frame data too small: need {} bytes for {}x{} RGB, got {}",
            expected,
            width,
            height,
            data.len()
        )));
    }

    // Convert RGB to grayscale using BT.601 luma
    let mut gray = Vec::with_capacity(w * h);
    for pixel in data[..expected].chunks_exact(3) {
        let r = pixel[0] as f64;
        let g = pixel[1] as f64;
        let b = pixel[2] as f64;
        let luma = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
        gray.push(luma);
    }

    let arr = Array2::from_shape_vec((h, w), gray)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create frame array: {e}")))?;

    Ok(Frame::new(w, h, timestamp, arr))
}

/// Convert a stabilizer `Frame` back to RGB bytes (grayscale -> RGB triplicate).
fn frame_to_rgb_bytes(frame: &Frame) -> Vec<u8> {
    let mut out = Vec::with_capacity(frame.width * frame.height * 3);
    for &val in frame.data.iter() {
        out.push(val);
        out.push(val);
        out.push(val);
    }
    out
}

// ---------------------------------------------------------------------------
// PyStabilizeConfig
// ---------------------------------------------------------------------------

/// Configuration for video stabilization.
#[pyclass]
#[derive(Clone)]
pub struct PyStabilizeConfig {
    /// Stabilization mode: "translation", "affine", "perspective", or "3d".
    #[pyo3(get)]
    pub mode: String,
    /// Smoothing strength (0.0 to 1.0).
    #[pyo3(get)]
    pub strength: f64,
    /// Temporal smoothing window size in frames.
    #[pyo3(get)]
    pub temporal_window: u32,
    /// Quality preset: "fast", "balanced", or "maximum".
    #[pyo3(get)]
    pub quality_preset: String,
    /// Whether zoom optimization is enabled.
    #[pyo3(get)]
    pub enable_zoom_optimization: bool,
    /// Whether horizon leveling is enabled.
    #[pyo3(get)]
    pub enable_horizon_leveling: bool,
}

#[pymethods]
impl PyStabilizeConfig {
    /// Create a new stabilization configuration.
    ///
    /// Args:
    ///     mode: Stabilization mode (default: "affine").
    ///     strength: Smoothing strength 0.0-1.0 (default: 0.8).
    ///     quality_preset: Quality preset (default: "balanced").
    #[new]
    #[pyo3(signature = (mode=None, strength=None, quality_preset=None))]
    fn new(
        mode: Option<&str>,
        strength: Option<f64>,
        quality_preset: Option<&str>,
    ) -> PyResult<Self> {
        let m = mode.unwrap_or("affine");
        let s = strength.unwrap_or(0.8);
        let q = quality_preset.unwrap_or("balanced");

        // Validate
        let _ = parse_mode(m)?;
        let _ = parse_quality(q)?;

        if !(0.0..=1.0).contains(&s) {
            return Err(PyValueError::new_err(format!(
                "strength must be between 0.0 and 1.0, got {s}"
            )));
        }

        let qp = parse_quality(q)?;
        Ok(Self {
            mode: m.to_string(),
            strength: s,
            temporal_window: qp.smoothing_window() as u32,
            quality_preset: q.to_string(),
            enable_zoom_optimization: true,
            enable_horizon_leveling: false,
        })
    }

    /// Create a translation-only stabilization config.
    #[classmethod]
    fn translation(_cls: &Bound<'_, PyType>) -> PyResult<Self> {
        Self::new(Some("translation"), None, None)
    }

    /// Create an affine stabilization config.
    #[classmethod]
    fn affine(_cls: &Bound<'_, PyType>) -> PyResult<Self> {
        Self::new(Some("affine"), None, None)
    }

    /// Create a perspective stabilization config.
    #[classmethod]
    fn perspective(_cls: &Bound<'_, PyType>) -> PyResult<Self> {
        Self::new(Some("perspective"), None, None)
    }

    /// Set smoothing strength (0.0 to 1.0).
    fn with_strength(&mut self, strength: f64) -> PyResult<()> {
        if !(0.0..=1.0).contains(&strength) {
            return Err(PyValueError::new_err(format!(
                "strength must be between 0.0 and 1.0, got {strength}"
            )));
        }
        self.strength = strength;
        Ok(())
    }

    /// Set temporal smoothing window in frames.
    fn with_temporal_window(&mut self, frames: u32) -> PyResult<()> {
        if frames == 0 {
            return Err(PyValueError::new_err(
                "temporal_window must be greater than 0",
            ));
        }
        self.temporal_window = frames;
        Ok(())
    }

    /// Set quality preset: "fast", "balanced", or "maximum".
    fn with_quality(&mut self, preset: &str) -> PyResult<()> {
        let qp = parse_quality(preset)?;
        self.quality_preset = preset.to_string();
        self.temporal_window = qp.smoothing_window() as u32;
        Ok(())
    }

    /// Enable or disable zoom optimization.
    fn with_zoom_optimization(&mut self, enable: bool) {
        self.enable_zoom_optimization = enable;
    }

    /// Enable or disable horizon leveling.
    fn with_horizon_leveling(&mut self, enable: bool) {
        self.enable_horizon_leveling = enable;
    }

    fn __repr__(&self) -> String {
        format!(
            "PyStabilizeConfig(mode='{}', strength={:.2}, quality='{}', \
             zoom_opt={}, horizon={})",
            self.mode,
            self.strength,
            self.quality_preset,
            self.enable_zoom_optimization,
            self.enable_horizon_leveling,
        )
    }
}

// ---------------------------------------------------------------------------
// PyMotionData
// ---------------------------------------------------------------------------

/// Motion data for a single frame pair.
#[pyclass]
#[derive(Clone)]
pub struct PyMotionData {
    /// Horizontal translation (pixels).
    #[pyo3(get)]
    pub dx: f64,
    /// Vertical translation (pixels).
    #[pyo3(get)]
    pub dy: f64,
    /// Rotation angle (radians).
    #[pyo3(get)]
    pub rotation: f64,
    /// Scale factor.
    #[pyo3(get)]
    pub scale: f64,
    /// Confidence of the motion estimate (0.0-1.0).
    #[pyo3(get)]
    pub confidence: f64,
}

#[pymethods]
impl PyMotionData {
    fn __repr__(&self) -> String {
        format!(
            "PyMotionData(dx={:.2}, dy={:.2}, rot={:.4}, scale={:.4}, conf={:.3})",
            self.dx, self.dy, self.rotation, self.scale, self.confidence
        )
    }

    /// Convert to a Python dict.
    fn to_dict(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("dx".to_string(), self.dx);
        m.insert("dy".to_string(), self.dy);
        m.insert("rotation".to_string(), self.rotation);
        m.insert("scale".to_string(), self.scale);
        m.insert("confidence".to_string(), self.confidence);
        m
    }
}

// ---------------------------------------------------------------------------
// PyStabilizer
// ---------------------------------------------------------------------------

/// Video stabilizer that processes frames incrementally.
#[pyclass]
pub struct PyStabilizer {
    config: StabilizeConfig,
    motion_data: Vec<PyMotionData>,
    frames_buffer: Vec<Frame>,
}

#[pymethods]
impl PyStabilizer {
    /// Create a new stabilizer with the given configuration.
    #[new]
    fn new(config: &PyStabilizeConfig) -> PyResult<Self> {
        let internal_config = build_internal_config(config)?;
        Ok(Self {
            config: internal_config,
            motion_data: Vec::new(),
            frames_buffer: Vec::new(),
        })
    }

    /// Analyze a frame and accumulate motion data.
    ///
    /// Args:
    ///     frame_data: Raw RGB bytes (width * height * 3).
    ///     width: Frame width in pixels.
    ///     height: Frame height in pixels.
    ///
    /// Returns:
    ///     Motion data estimated between this frame and the previous one.
    fn analyze_frame(
        &mut self,
        frame_data: Vec<u8>,
        width: u32,
        height: u32,
    ) -> PyResult<PyMotionData> {
        let timestamp = self.frames_buffer.len() as f64 / 30.0;
        let frame = rgb_bytes_to_frame(&frame_data, width, height, timestamp)?;

        let motion = if self.frames_buffer.is_empty() {
            // First frame has no motion reference
            PyMotionData {
                dx: 0.0,
                dy: 0.0,
                rotation: 0.0,
                scale: 1.0,
                confidence: 1.0,
            }
        } else {
            // Estimate motion from previous frame using simple block matching
            let prev = self
                .frames_buffer
                .last()
                .ok_or_else(|| PyRuntimeError::new_err("No previous frame available"))?;
            estimate_simple_motion(prev, &frame)
        };

        self.motion_data.push(motion.clone());
        self.frames_buffer.push(frame);
        Ok(motion)
    }

    /// Stabilize a single frame using accumulated motion data.
    ///
    /// Note: All frames should be analyzed first via `analyze_frame` before
    /// calling `process_frame` for best results.
    ///
    /// Args:
    ///     frame_data: Raw RGB bytes (width * height * 3).
    ///     width: Frame width in pixels.
    ///     height: Frame height in pixels.
    ///
    /// Returns:
    ///     Stabilized frame as RGB bytes.
    fn process_frame(&mut self, frame_data: Vec<u8>, width: u32, height: u32) -> PyResult<Vec<u8>> {
        let timestamp = self.frames_buffer.len() as f64 / 30.0;
        let frame = rgb_bytes_to_frame(&frame_data, width, height, timestamp)?;

        // For single-frame processing, accumulate and then stabilize the batch
        self.frames_buffer.push(frame);

        let mut stabilizer = Stabilizer::new(self.config.clone())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create stabilizer: {e}")))?;

        let stabilized = stabilizer
            .stabilize(&self.frames_buffer)
            .map_err(|e| PyRuntimeError::new_err(format!("Stabilization failed: {e}")))?;

        let last = stabilized
            .last()
            .ok_or_else(|| PyRuntimeError::new_err("No stabilized frames produced"))?;

        Ok(frame_to_rgb_bytes(last))
    }

    /// Return all accumulated motion data.
    fn motion_data(&self) -> Vec<PyMotionData> {
        self.motion_data.clone()
    }

    /// Reset the stabilizer state, clearing all accumulated data.
    fn reset(&mut self) {
        self.motion_data.clear();
        self.frames_buffer.clear();
    }

    /// Get the current configuration as a PyStabilizeConfig.
    fn config(&self) -> PyStabilizeConfig {
        PyStabilizeConfig {
            mode: mode_to_string(self.config.mode),
            strength: self.config.smoothing_strength,
            temporal_window: self.config.quality.smoothing_window() as u32,
            quality_preset: quality_to_string(self.config.quality),
            enable_zoom_optimization: self.config.enable_zoom_optimization,
            enable_horizon_leveling: self.config.enable_horizon_leveling,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PyStabilizer(mode='{}', frames_buffered={}, motion_samples={})",
            mode_to_string(self.config.mode),
            self.frames_buffer.len(),
            self.motion_data.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// Simple motion estimation between two frames
// ---------------------------------------------------------------------------

fn estimate_simple_motion(prev: &Frame, curr: &Frame) -> PyMotionData {
    let w = prev.width.min(curr.width);
    let h = prev.height.min(curr.height);

    if w == 0 || h == 0 {
        return PyMotionData {
            dx: 0.0,
            dy: 0.0,
            rotation: 0.0,
            scale: 1.0,
            confidence: 0.0,
        };
    }

    // Compute centroid difference of intensity-weighted positions
    // as a coarse motion estimate
    let block_size = 16;
    let mut sum_dx: f64 = 0.0;
    let mut sum_dy: f64 = 0.0;
    let mut count: f64 = 0.0;

    let search_range: isize = 8;

    for by in (0..h).step_by(block_size) {
        for bx in (0..w).step_by(block_size) {
            let bw = block_size.min(w - bx);
            let bh = block_size.min(h - by);

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
                    let tx = tx as usize;
                    let ty = ty as usize;
                    if tx + bw > w || ty + bh > h {
                        continue;
                    }

                    let mut sad: u64 = 0;
                    for yy in 0..bh {
                        for xx in 0..bw {
                            let p = prev.data[[by + yy, bx + xx]] as i32;
                            let c = curr.data[[ty + yy, tx + xx]] as i32;
                            sad += (p - c).unsigned_abs() as u64;
                        }
                    }

                    // When SADs are equal prefer the candidate closest to zero
                    // displacement (L1 distance), so that identical frames
                    // correctly produce (dx=0, dy=0) regardless of search order.
                    let better_sad = sad < best_sad;
                    let tied_but_closer =
                        sad == best_sad && sx.abs() + sy.abs() < best_dx.abs() + best_dy.abs();
                    if better_sad || tied_but_closer {
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

    // Confidence based on how consistent the block motions are
    let motion_mag = (dx * dx + dy * dy).sqrt();
    let confidence = if motion_mag < 0.5 {
        0.95
    } else if motion_mag < 5.0 {
        0.8
    } else {
        0.6
    };

    PyMotionData {
        dx,
        dy,
        rotation: 0.0,
        scale: 1.0,
        confidence,
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Stabilize a batch of video frames.
///
/// Args:
///     frames: List of raw RGB byte arrays (each width * height * 3).
///     width: Frame width in pixels.
///     height: Frame height in pixels.
///     mode: Stabilization mode (default: "affine").
///     strength: Smoothing strength 0.0-1.0 (default: 0.8).
///
/// Returns:
///     List of stabilized RGB byte arrays.
#[pyfunction]
#[pyo3(signature = (frames, width, height, mode=None, strength=None))]
pub fn stabilize_video_frames(
    frames: Vec<Vec<u8>>,
    width: u32,
    height: u32,
    mode: Option<&str>,
    strength: Option<f64>,
) -> PyResult<Vec<Vec<u8>>> {
    if frames.is_empty() {
        return Err(PyValueError::new_err("No frames provided"));
    }

    let stab_mode = parse_mode(mode.unwrap_or("affine"))?;
    let stab_strength = strength.unwrap_or(0.8);

    if !(0.0..=1.0).contains(&stab_strength) {
        return Err(PyValueError::new_err(format!(
            "strength must be between 0.0 and 1.0, got {stab_strength}"
        )));
    }

    let config = StabilizeConfig::new()
        .with_mode(stab_mode)
        .with_smoothing_strength(stab_strength);

    // Convert all frames
    let internal_frames: Vec<Frame> = frames
        .iter()
        .enumerate()
        .map(|(i, f)| rgb_bytes_to_frame(f, width, height, i as f64 / 30.0))
        .collect::<PyResult<Vec<_>>>()?;

    let mut stabilizer = Stabilizer::new(config)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create stabilizer: {e}")))?;

    let stabilized = stabilizer
        .stabilize(&internal_frames)
        .map_err(|e| PyRuntimeError::new_err(format!("Stabilization failed: {e}")))?;

    Ok(stabilized.iter().map(frame_to_rgb_bytes).collect())
}

/// Estimate motion between two consecutive frames.
///
/// Args:
///     frame1: First frame as raw RGB bytes.
///     frame2: Second frame as raw RGB bytes.
///     width: Frame width in pixels.
///     height: Frame height in pixels.
///
/// Returns:
///     Motion data between the two frames.
#[pyfunction]
pub fn estimate_motion(
    frame1: Vec<u8>,
    frame2: Vec<u8>,
    width: u32,
    height: u32,
) -> PyResult<PyMotionData> {
    let f1 = rgb_bytes_to_frame(&frame1, width, height, 0.0)?;
    let f2 = rgb_bytes_to_frame(&frame2, width, height, 1.0 / 30.0)?;
    Ok(estimate_simple_motion(&f1, &f2))
}

/// List available stabilization modes.
///
/// Returns:
///     List of mode name strings.
#[pyfunction]
pub fn list_stabilization_modes() -> Vec<String> {
    vec![
        "translation".to_string(),
        "affine".to_string(),
        "perspective".to_string(),
        "3d".to_string(),
    ]
}

// ---------------------------------------------------------------------------
// Registration helper
// ---------------------------------------------------------------------------

/// Register all stabilize bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStabilizeConfig>()?;
    m.add_class::<PyMotionData>()?;
    m.add_class::<PyStabilizer>()?;
    m.add_function(wrap_pyfunction!(stabilize_video_frames, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_motion, m)?)?;
    m.add_function(wrap_pyfunction!(list_stabilization_modes, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_rgb(width: u32, height: u32, fill: u8) -> Vec<u8> {
        vec![fill; (width as usize) * (height as usize) * 3]
    }

    #[test]
    fn test_config_default() {
        let cfg = PyStabilizeConfig::new(None, None, None);
        assert!(cfg.is_ok());
        let cfg = cfg.expect("config should be valid");
        assert_eq!(cfg.mode, "affine");
        assert!((cfg.strength - 0.8).abs() < f64::EPSILON);
        assert_eq!(cfg.quality_preset, "balanced");
    }

    #[test]
    fn test_config_invalid_strength() {
        let result = PyStabilizeConfig::new(None, Some(1.5), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_invalid_mode() {
        let result = PyStabilizeConfig::new(Some("invalid"), None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_rgb_to_frame_conversion() {
        let data = make_test_rgb(4, 4, 128);
        let frame = rgb_bytes_to_frame(&data, 4, 4, 0.0);
        assert!(frame.is_ok());
        let f = frame.expect("frame should be valid");
        assert_eq!(f.width, 4);
        assert_eq!(f.height, 4);
    }

    #[test]
    fn test_motion_data_to_dict() {
        let md = PyMotionData {
            dx: 1.0,
            dy: 2.0,
            rotation: 0.1,
            scale: 1.0,
            confidence: 0.9,
        };
        let d = md.to_dict();
        assert_eq!(d.len(), 5);
        assert!((d["dx"] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_estimate_motion_identical_frames() {
        let data = make_test_rgb(32, 32, 100);
        let result = estimate_motion(data.clone(), data, 32, 32);
        assert!(result.is_ok());
        let m = result.expect("motion should succeed");
        assert!(m.dx.abs() < 1.0);
        assert!(m.dy.abs() < 1.0);
    }

    #[test]
    fn test_list_modes() {
        let modes = list_stabilization_modes();
        assert_eq!(modes.len(), 4);
        assert!(modes.contains(&"affine".to_string()));
    }
}
