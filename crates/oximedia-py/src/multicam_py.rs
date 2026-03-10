//! Python bindings for `oximedia-multicam` multi-camera production.
//!
//! Provides `PyMultiCamTimeline`, `PyAutoSwitcher`, `PyCompositor`, and
//! standalone compositing functions for multi-camera workflows from Python.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_layout(layout: &str) -> PyResult<()> {
    match layout {
        "grid" | "pip" | "side_by_side" | "stack" => Ok(()),
        other => Err(PyValueError::new_err(format!(
            "Unknown layout '{}'. Expected: grid, pip, side_by_side, stack",
            other
        ))),
    }
}

fn validate_sync_method(method: &str) -> PyResult<()> {
    match method {
        "audio" | "timecode" | "marker" => Ok(()),
        other => Err(PyValueError::new_err(format!(
            "Unknown sync method '{}'. Expected: audio, timecode, marker",
            other
        ))),
    }
}

// ---------------------------------------------------------------------------
// SwitchPoint
// ---------------------------------------------------------------------------

/// A single switch point in a multi-camera timeline.
#[derive(Clone, Debug)]
struct SwitchPoint {
    time: f64,
    camera_index: usize,
}

// ---------------------------------------------------------------------------
// PyMultiCamTimeline
// ---------------------------------------------------------------------------

/// Multi-camera timeline for synchronizing and switching between camera angles.
#[pyclass]
pub struct PyMultiCamTimeline {
    cameras: Vec<String>,
    sync_method: String,
    switch_points: Vec<SwitchPoint>,
    #[pyo3(get)]
    fps: f64,
    #[pyo3(get)]
    duration: f64,
    config: oximedia_multicam::MultiCamConfig,
}

#[pymethods]
impl PyMultiCamTimeline {
    /// Create a new multi-camera timeline.
    ///
    /// Args:
    ///     fps: Frame rate (default: 25.0).
    #[new]
    #[pyo3(signature = (fps=25.0))]
    fn new(fps: f64) -> PyResult<Self> {
        if fps <= 0.0 {
            return Err(PyValueError::new_err("FPS must be positive"));
        }
        Ok(Self {
            cameras: Vec::new(),
            sync_method: "audio".to_string(),
            switch_points: Vec::new(),
            fps,
            duration: 0.0,
            config: oximedia_multicam::MultiCamConfig {
                frame_rate: fps,
                output_frame_rate: fps,
                ..oximedia_multicam::MultiCamConfig::default()
            },
        })
    }

    /// Add a camera angle by path.
    fn add_camera(&mut self, path: &str) {
        self.cameras.push(path.to_string());
        self.config.angle_count = self.cameras.len();
    }

    /// Synchronize cameras using the given method.
    ///
    /// Args:
    ///     method: Sync method - "audio", "timecode", or "marker".
    fn sync_cameras(&mut self, method: &str) -> PyResult<()> {
        validate_sync_method(method)?;
        self.sync_method = method.to_string();
        self.config.enable_audio_sync = method == "audio";
        self.config.enable_timecode_sync = method == "timecode";
        self.config.enable_visual_sync = method == "marker";
        Ok(())
    }

    /// Add a switch point at the given time.
    ///
    /// Args:
    ///     time: Time in seconds.
    ///     camera_index: Camera angle index to switch to.
    fn add_switch_point(&mut self, time: f64, camera_index: usize) -> PyResult<()> {
        if camera_index >= self.cameras.len() && !self.cameras.is_empty() {
            return Err(PyValueError::new_err(format!(
                "Camera index {} out of range (0..{})",
                camera_index,
                self.cameras.len()
            )));
        }
        if time < 0.0 {
            return Err(PyValueError::new_err("Switch time must be >= 0"));
        }
        self.switch_points.push(SwitchPoint { time, camera_index });
        // Keep sorted by time
        self.switch_points.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // Update duration to at least the latest switch point
        if time > self.duration {
            self.duration = time;
        }
        Ok(())
    }

    /// Generate automatic switch points based on shot variety strategy.
    ///
    /// Distributes switches evenly across cameras at the given interval.
    ///
    /// Args:
    ///     interval_secs: Time between switches (default: 3.0).
    ///     duration_secs: Total timeline duration.
    #[pyo3(signature = (interval_secs=3.0, duration_secs=30.0))]
    fn auto_switch(&mut self, interval_secs: f64, duration_secs: f64) -> PyResult<()> {
        if self.cameras.is_empty() {
            return Err(PyRuntimeError::new_err("No cameras added to timeline"));
        }
        if interval_secs <= 0.0 {
            return Err(PyValueError::new_err("Interval must be positive"));
        }

        self.switch_points.clear();
        self.duration = duration_secs;

        let mut t = 0.0;
        let mut cam_idx = 0;
        while t < duration_secs {
            self.switch_points.push(SwitchPoint {
                time: t,
                camera_index: cam_idx,
            });
            t += interval_secs;
            cam_idx = (cam_idx + 1) % self.cameras.len();
        }

        Ok(())
    }

    /// Get the number of cameras.
    fn camera_count(&self) -> usize {
        self.cameras.len()
    }

    /// Get the number of switch points.
    fn switch_count(&self) -> usize {
        self.switch_points.len()
    }

    /// Get the camera at a specific time.
    fn camera_at_time(&self, time: f64) -> usize {
        let mut active_camera = 0;
        for sp in &self.switch_points {
            if sp.time <= time {
                active_camera = sp.camera_index;
            } else {
                break;
            }
        }
        active_camera
    }

    /// Export timeline to JSON.
    fn to_json(&self) -> PyResult<String> {
        let switch_list: Vec<HashMap<String, serde_json::Value>> = self
            .switch_points
            .iter()
            .map(|sp| {
                let mut m = HashMap::new();
                m.insert("time".to_string(), serde_json::Value::from(sp.time));
                m.insert(
                    "camera".to_string(),
                    serde_json::Value::from(sp.camera_index),
                );
                m
            })
            .collect();

        let result = serde_json::json!({
            "cameras": self.cameras,
            "sync_method": self.sync_method,
            "fps": self.fps,
            "duration": self.duration,
            "switch_points": switch_list,
        });

        serde_json::to_string_pretty(&result)
            .map_err(|e| PyRuntimeError::new_err(format!("JSON serialization failed: {e}")))
    }

    fn __repr__(&self) -> String {
        format!(
            "PyMultiCamTimeline(cameras={}, fps={:.1}, switches={}, duration={:.1}s)",
            self.cameras.len(),
            self.fps,
            self.switch_points.len(),
            self.duration,
        )
    }
}

// ---------------------------------------------------------------------------
// PyAutoSwitcher
// ---------------------------------------------------------------------------

/// Automatic camera angle switcher based on content analysis strategies.
#[pyclass]
pub struct PyAutoSwitcher {
    #[pyo3(get)]
    strategy: String,
    #[pyo3(get)]
    min_shot_duration: f64,
    #[pyo3(get)]
    transition_type: String,
}

#[pymethods]
impl PyAutoSwitcher {
    /// Create a new auto-switcher.
    ///
    /// Args:
    ///     strategy: Switching strategy - "round_robin", "random", "shot_variety", "content_aware".
    #[new]
    #[pyo3(signature = (strategy="round_robin"))]
    fn new(strategy: &str) -> PyResult<Self> {
        match strategy {
            "round_robin" | "random" | "shot_variety" | "content_aware" => {}
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown strategy '{}'. Expected: round_robin, random, shot_variety, content_aware",
                    other
                )));
            }
        }

        Ok(Self {
            strategy: strategy.to_string(),
            min_shot_duration: 2.0,
            transition_type: "cut".to_string(),
        })
    }

    /// Set minimum shot duration in seconds.
    fn with_min_duration(&mut self, secs: f64) -> PyResult<()> {
        if secs <= 0.0 {
            return Err(PyValueError::new_err("Duration must be positive"));
        }
        self.min_shot_duration = secs;
        Ok(())
    }

    /// Set transition type: "cut", "dissolve", "wipe".
    fn with_transition(&mut self, transition: &str) -> PyResult<()> {
        match transition {
            "cut" | "dissolve" | "wipe" => {
                self.transition_type = transition.to_string();
                Ok(())
            }
            other => Err(PyValueError::new_err(format!(
                "Unknown transition '{}'. Expected: cut, dissolve, wipe",
                other
            ))),
        }
    }

    /// Generate switch points for the given number of cameras and duration.
    ///
    /// Args:
    ///     camera_count: Number of camera angles.
    ///     duration_secs: Total duration in seconds.
    ///
    /// Returns:
    ///     List of (time, camera_index) tuples.
    fn switch(&self, camera_count: usize, duration_secs: f64) -> PyResult<Vec<(f64, usize)>> {
        if camera_count == 0 {
            return Err(PyValueError::new_err("Camera count must be > 0"));
        }
        if duration_secs <= 0.0 {
            return Err(PyValueError::new_err("Duration must be positive"));
        }

        let mut points = Vec::new();
        let mut t = 0.0;
        let mut cam_idx = 0;

        match self.strategy.as_str() {
            "round_robin" => {
                while t < duration_secs {
                    points.push((t, cam_idx));
                    t += self.min_shot_duration;
                    cam_idx = (cam_idx + 1) % camera_count;
                }
            }
            "random" | "shot_variety" | "content_aware" => {
                // Pseudo-random based on a simple hash-like distribution
                let mut seed = 42u64;
                while t < duration_secs {
                    cam_idx = (seed as usize) % camera_count;
                    points.push((t, cam_idx));
                    t += self.min_shot_duration;
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                }
            }
            _ => {
                // Fallback: round robin
                while t < duration_secs {
                    points.push((t, cam_idx % camera_count));
                    t += self.min_shot_duration;
                    cam_idx += 1;
                }
            }
        }

        Ok(points)
    }

    fn __repr__(&self) -> String {
        format!(
            "PyAutoSwitcher(strategy='{}', min_duration={:.1}s, transition='{}')",
            self.strategy, self.min_shot_duration, self.transition_type,
        )
    }
}

// ---------------------------------------------------------------------------
// PyCompositor
// ---------------------------------------------------------------------------

/// Multi-camera frame compositor for creating multi-view layouts.
#[pyclass]
pub struct PyCompositor {
    #[pyo3(get)]
    layout: String,
    #[pyo3(get)]
    width: u32,
    #[pyo3(get)]
    height: u32,
    spacing: u32,
}

#[pymethods]
impl PyCompositor {
    /// Create a new compositor.
    ///
    /// Args:
    ///     layout: Layout type (grid, pip, side_by_side, stack).
    ///     width: Output width.
    ///     height: Output height.
    #[new]
    #[pyo3(signature = (layout="grid", width=1920, height=1080))]
    fn new(layout: &str, width: u32, height: u32) -> PyResult<Self> {
        validate_layout(layout)?;
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err("Width and height must be > 0"));
        }
        Ok(Self {
            layout: layout.to_string(),
            width,
            height,
            spacing: 4,
        })
    }

    /// Set grid spacing in pixels.
    fn set_spacing(&mut self, spacing: u32) {
        self.spacing = spacing;
    }

    /// Composite frames into a grid layout.
    ///
    /// Args:
    ///     frames: List of RGBA frame byte arrays (all same size).
    ///     frame_w: Width of each input frame.
    ///     frame_h: Height of each input frame.
    ///
    /// Returns:
    ///     Composited RGBA pixel data at output resolution.
    fn grid(&self, frames: Vec<Vec<u8>>, frame_w: u32, frame_h: u32) -> PyResult<Vec<u8>> {
        if frames.is_empty() {
            return Err(PyValueError::new_err("No frames provided"));
        }
        let expected = (frame_w as usize) * (frame_h as usize) * 4;
        for (i, f) in frames.iter().enumerate() {
            if f.len() < expected {
                return Err(PyValueError::new_err(format!(
                    "Frame {} too small: need {} bytes, got {}",
                    i,
                    expected,
                    f.len()
                )));
            }
        }

        let (rows, cols) =
            oximedia_multicam::composite::grid::GridCompositor::optimal_grid_for_angles(
                frames.len(),
            );
        let mut grid =
            oximedia_multicam::composite::grid::GridCompositor::new(self.width, self.height);
        grid.set_spacing(self.spacing);
        let cells = grid.calculate_grid(rows, cols);

        let out_size = (self.width as usize) * (self.height as usize) * 4;
        let mut output = vec![0u8; out_size];

        for (i, frame) in frames.iter().enumerate() {
            if i >= cells.len() {
                break;
            }
            let (cx, cy, cw, ch) = cells[i];
            blit_scaled(
                frame,
                frame_w,
                frame_h,
                &mut output,
                self.width,
                self.height,
                cx,
                cy,
                cw,
                ch,
            );
        }

        Ok(output)
    }

    /// Composite two frames in picture-in-picture layout.
    ///
    /// Args:
    ///     main_frame: Main view RGBA bytes.
    ///     overlay_frame: PIP view RGBA bytes.
    ///     main_w: Main frame width.
    ///     main_h: Main frame height.
    ///     pip_x: PIP X position in output.
    ///     pip_y: PIP Y position in output.
    ///     pip_w: PIP width in output.
    ///     pip_h: PIP height in output.
    ///
    /// Returns:
    ///     Composited RGBA pixel data.
    #[pyo3(signature = (main_frame, overlay_frame, main_w, main_h, pip_x=None, pip_y=None, pip_w=None, pip_h=None))]
    fn pip(
        &self,
        main_frame: Vec<u8>,
        overlay_frame: Vec<u8>,
        main_w: u32,
        main_h: u32,
        pip_x: Option<u32>,
        pip_y: Option<u32>,
        pip_w: Option<u32>,
        pip_h: Option<u32>,
    ) -> PyResult<Vec<u8>> {
        let expected_main = (main_w as usize) * (main_h as usize) * 4;
        if main_frame.len() < expected_main {
            return Err(PyValueError::new_err("Main frame data too small"));
        }

        let out_size = (self.width as usize) * (self.height as usize) * 4;
        let mut output = vec![0u8; out_size];

        // Blit main frame scaled to output
        blit_scaled(
            &main_frame,
            main_w,
            main_h,
            &mut output,
            self.width,
            self.height,
            0,
            0,
            self.width,
            self.height,
        );

        // PIP defaults: bottom-right, 1/4 size
        let pw = pip_w.unwrap_or(self.width / 4);
        let ph = pip_h.unwrap_or(self.height / 4);
        let px = pip_x.unwrap_or(self.width - pw - 20);
        let py = pip_y.unwrap_or(self.height - ph - 20);

        let ov_pixel_count = overlay_frame.len() / 4;
        let ov_w = if ov_pixel_count > 0 {
            let sqrt = (ov_pixel_count as f64).sqrt() as u32;
            if sqrt > 0 {
                sqrt
            } else {
                1
            }
        } else {
            return Err(PyValueError::new_err("Overlay frame is empty"));
        };
        let ov_h = (ov_pixel_count as u32).checked_div(ov_w).unwrap_or(1);

        blit_scaled(
            &overlay_frame,
            ov_w,
            ov_h,
            &mut output,
            self.width,
            self.height,
            px,
            py,
            pw,
            ph,
        );

        Ok(output)
    }

    /// Composite two frames side by side.
    fn side_by_side(
        &self,
        left: Vec<u8>,
        right: Vec<u8>,
        frame_w: u32,
        frame_h: u32,
    ) -> PyResult<Vec<u8>> {
        let expected = (frame_w as usize) * (frame_h as usize) * 4;
        if left.len() < expected {
            return Err(PyValueError::new_err("Left frame data too small"));
        }
        if right.len() < expected {
            return Err(PyValueError::new_err("Right frame data too small"));
        }

        let out_size = (self.width as usize) * (self.height as usize) * 4;
        let mut output = vec![0u8; out_size];

        let half_w = self.width / 2;

        blit_scaled(
            &left,
            frame_w,
            frame_h,
            &mut output,
            self.width,
            self.height,
            0,
            0,
            half_w,
            self.height,
        );
        blit_scaled(
            &right,
            frame_w,
            frame_h,
            &mut output,
            self.width,
            self.height,
            half_w,
            0,
            self.width - half_w,
            self.height,
        );

        Ok(output)
    }

    fn __repr__(&self) -> String {
        format!(
            "PyCompositor(layout='{}', size={}x{}, spacing={})",
            self.layout, self.width, self.height, self.spacing,
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Composite multiple frames into a grid layout.
///
/// Args:
///     frames: List of RGBA frame byte arrays (all same size).
///     frame_w: Width of each input frame.
///     frame_h: Height of each input frame.
///     cols: Number of grid columns (optional, auto if 0).
///
/// Returns:
///     Composited RGBA data at a resolution derived from inputs.
#[pyfunction]
#[pyo3(signature = (frames, frame_w, frame_h, cols=0))]
pub fn composite_grid(
    frames: Vec<Vec<u8>>,
    frame_w: u32,
    frame_h: u32,
    cols: u32,
) -> PyResult<Vec<u8>> {
    if frames.is_empty() {
        return Err(PyValueError::new_err("No frames provided"));
    }

    let (rows, grid_cols) = if cols == 0 {
        oximedia_multicam::composite::grid::GridCompositor::optimal_grid_for_angles(frames.len())
    } else {
        let r = (frames.len() as f64 / cols as f64).ceil() as usize;
        (r, cols as usize)
    };

    let out_w = frame_w * grid_cols as u32;
    let out_h = frame_h * rows as u32;
    let out_size = (out_w as usize) * (out_h as usize) * 4;
    let mut output = vec![0u8; out_size];

    let expected = (frame_w as usize) * (frame_h as usize) * 4;

    for (i, frame) in frames.iter().enumerate() {
        if frame.len() < expected {
            return Err(PyValueError::new_err(format!(
                "Frame {} too small: need {} bytes, got {}",
                i,
                expected,
                frame.len()
            )));
        }

        let col = i % grid_cols;
        let row = i / grid_cols;
        let cx = (col as u32) * frame_w;
        let cy = (row as u32) * frame_h;

        blit_scaled(
            frame,
            frame_w,
            frame_h,
            &mut output,
            out_w,
            out_h,
            cx,
            cy,
            frame_w,
            frame_h,
        );
    }

    Ok(output)
}

/// Composite a picture-in-picture layout.
///
/// Args:
///     main_frame: Main view RGBA bytes.
///     overlay_frame: PIP RGBA bytes.
///     width: Main frame width.
///     height: Main frame height.
///     position: PIP position: "bottom_right", "bottom_left", "top_right", "top_left".
///
/// Returns:
///     Composited RGBA pixel data.
#[pyfunction]
#[pyo3(signature = (main_frame, overlay_frame, width, height, position="bottom_right"))]
pub fn composite_pip(
    main_frame: Vec<u8>,
    overlay_frame: Vec<u8>,
    width: u32,
    height: u32,
    position: &str,
) -> PyResult<Vec<u8>> {
    let expected = (width as usize) * (height as usize) * 4;
    if main_frame.len() < expected {
        return Err(PyValueError::new_err("Main frame data too small"));
    }

    let pip_w = width / 4;
    let pip_h = height / 4;
    let pad = 20u32;

    let (px, py) = match position {
        "bottom_right" => (width - pip_w - pad, height - pip_h - pad),
        "bottom_left" => (pad, height - pip_h - pad),
        "top_right" => (width - pip_w - pad, pad),
        "top_left" => (pad, pad),
        other => {
            return Err(PyValueError::new_err(format!(
                "Unknown position '{}'. Expected: bottom_right, bottom_left, top_right, top_left",
                other
            )));
        }
    };

    let mut output = main_frame;

    let ov_pixel_count = overlay_frame.len() / 4;
    let ov_w = if ov_pixel_count > 0 {
        let sqrt = (ov_pixel_count as f64).sqrt() as u32;
        if sqrt > 0 {
            sqrt
        } else {
            1
        }
    } else {
        return Err(PyValueError::new_err("Overlay frame is empty"));
    };
    let ov_h = (ov_pixel_count as u32).checked_div(ov_w).unwrap_or(1);

    blit_scaled(
        &overlay_frame,
        ov_w,
        ov_h,
        &mut output,
        width,
        height,
        px,
        py,
        pip_w,
        pip_h,
    );

    Ok(output)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Nearest-neighbor scaled blit from source into destination region.
fn blit_scaled(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst: &mut [u8],
    dst_total_w: u32,
    dst_total_h: u32,
    dst_x: u32,
    dst_y: u32,
    region_w: u32,
    region_h: u32,
) {
    if src_w == 0 || src_h == 0 || region_w == 0 || region_h == 0 {
        return;
    }

    for ry in 0..region_h {
        for rx in 0..region_w {
            let px = dst_x + rx;
            let py = dst_y + ry;
            if px >= dst_total_w || py >= dst_total_h {
                continue;
            }

            // Nearest neighbor sampling
            let sx = (rx as u64 * src_w as u64 / region_w as u64) as u32;
            let sy = (ry as u64 * src_h as u64 / region_h as u64) as u32;
            let sx = sx.min(src_w - 1);
            let sy = sy.min(src_h - 1);

            let src_idx = ((sy * src_w + sx) * 4) as usize;
            let dst_idx = ((py * dst_total_w + px) * 4) as usize;

            if src_idx + 3 < src.len() && dst_idx + 3 < dst.len() {
                dst[dst_idx] = src[src_idx];
                dst[dst_idx + 1] = src[src_idx + 1];
                dst[dst_idx + 2] = src[src_idx + 2];
                dst[dst_idx + 3] = src[src_idx + 3];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all multicam bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMultiCamTimeline>()?;
    m.add_class::<PyAutoSwitcher>()?;
    m.add_class::<PyCompositor>()?;
    m.add_function(wrap_pyfunction!(composite_grid, m)?)?;
    m.add_function(wrap_pyfunction!(composite_pip, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rgba_frame(w: u32, h: u32, val: u8) -> Vec<u8> {
        vec![val; (w as usize) * (h as usize) * 4]
    }

    #[test]
    fn test_timeline_creation() {
        let t = PyMultiCamTimeline::new(30.0);
        assert!(t.is_ok());
        let t = t.expect("ok");
        assert_eq!(t.camera_count(), 0);
    }

    #[test]
    fn test_timeline_add_camera() {
        let mut t = PyMultiCamTimeline::new(25.0).expect("ok");
        t.add_camera("/path/cam1.mkv");
        t.add_camera("/path/cam2.mkv");
        assert_eq!(t.camera_count(), 2);
    }

    #[test]
    fn test_timeline_switch_points() {
        let mut t = PyMultiCamTimeline::new(25.0).expect("ok");
        t.add_camera("cam1");
        t.add_camera("cam2");
        assert!(t.add_switch_point(0.0, 0).is_ok());
        assert!(t.add_switch_point(3.0, 1).is_ok());
        assert_eq!(t.switch_count(), 2);
        assert_eq!(t.camera_at_time(1.0), 0);
        assert_eq!(t.camera_at_time(4.0), 1);
    }

    #[test]
    fn test_timeline_auto_switch() {
        let mut t = PyMultiCamTimeline::new(25.0).expect("ok");
        t.add_camera("cam1");
        t.add_camera("cam2");
        assert!(t.auto_switch(2.0, 10.0).is_ok());
        assert!(t.switch_count() >= 5);
    }

    #[test]
    fn test_auto_switcher_creation() {
        let s = PyAutoSwitcher::new("round_robin");
        assert!(s.is_ok());
    }

    #[test]
    fn test_auto_switcher_invalid_strategy() {
        assert!(PyAutoSwitcher::new("invalid").is_err());
    }

    #[test]
    fn test_auto_switcher_switch() {
        let s = PyAutoSwitcher::new("round_robin").expect("ok");
        let points = s.switch(3, 9.0).expect("ok");
        assert!(!points.is_empty());
        // Each point should have camera in range 0..3
        for (_, cam) in &points {
            assert!(*cam < 3);
        }
    }

    #[test]
    fn test_compositor_creation() {
        let c = PyCompositor::new("grid", 1920, 1080);
        assert!(c.is_ok());
    }

    #[test]
    fn test_compositor_invalid_layout() {
        assert!(PyCompositor::new("invalid", 1920, 1080).is_err());
    }

    #[test]
    fn test_compositor_grid() {
        let c = PyCompositor::new("grid", 100, 100).expect("ok");
        let frames = vec![make_rgba_frame(50, 50, 100), make_rgba_frame(50, 50, 200)];
        let result = c.grid(frames, 50, 50);
        assert!(result.is_ok());
        assert_eq!(result.expect("ok").len(), 100 * 100 * 4);
    }

    #[test]
    fn test_composite_grid_fn() {
        let frames = vec![
            make_rgba_frame(10, 10, 50),
            make_rgba_frame(10, 10, 100),
            make_rgba_frame(10, 10, 150),
            make_rgba_frame(10, 10, 200),
        ];
        let result = composite_grid(frames, 10, 10, 2);
        assert!(result.is_ok());
        // 2 cols x 2 rows -> 20x20
        assert_eq!(result.expect("ok").len(), 20 * 20 * 4);
    }

    #[test]
    fn test_blit_scaled_basic() {
        let src = vec![255u8; 4 * 4 * 4]; // 4x4 RGBA
        let mut dst = vec![0u8; 8 * 8 * 4]; // 8x8 RGBA
        blit_scaled(&src, 4, 4, &mut dst, 8, 8, 0, 0, 8, 8);
        // Top-left pixel should be filled
        assert_eq!(dst[0], 255);
    }

    #[test]
    fn test_timeline_to_json() {
        let mut t = PyMultiCamTimeline::new(25.0).expect("ok");
        t.add_camera("cam1");
        t.add_camera("cam2");
        let _ = t.add_switch_point(0.0, 0);
        let json = t.to_json();
        assert!(json.is_ok());
        let json_str = json.expect("ok");
        assert!(json_str.contains("cam1"));
        assert!(json_str.contains("switch_points"));
    }
}
