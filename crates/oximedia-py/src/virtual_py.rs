//! Python bindings for `oximedia-virtual` virtual production.
//!
//! Provides `PyVirtualSource`, `PyVirtualConfig`, and standalone functions
//! for creating and managing virtual production sessions.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PyVirtualConfig
// ---------------------------------------------------------------------------

/// Configuration for a virtual production session.
#[pyclass]
#[derive(Clone)]
pub struct PyVirtualConfig {
    /// Workflow type string.
    #[pyo3(get)]
    pub workflow: String,
    /// Target frames per second.
    #[pyo3(get)]
    pub target_fps: f64,
    /// Synchronization accuracy in milliseconds.
    #[pyo3(get)]
    pub sync_accuracy_ms: f64,
    /// Quality mode string.
    #[pyo3(get)]
    pub quality: String,
    /// Number of tracked cameras.
    #[pyo3(get)]
    pub num_cameras: usize,
    /// Enable color calibration.
    #[pyo3(get)]
    pub color_calibration: bool,
    /// Enable lens distortion correction.
    #[pyo3(get)]
    pub lens_correction: bool,
    /// Enable motion capture integration.
    #[pyo3(get)]
    pub motion_capture: bool,
}

#[pymethods]
impl PyVirtualConfig {
    /// Create a new virtual production configuration.
    ///
    /// Args:
    ///     workflow: Workflow type (led-wall, hybrid, green-screen, ar). Default: led-wall.
    ///     target_fps: Target FPS. Default: 60.0.
    ///     quality: Quality mode (draft, preview, final). Default: preview.
    #[new]
    #[pyo3(signature = (workflow=None, target_fps=None, quality=None))]
    fn new(
        workflow: Option<&str>,
        target_fps: Option<f64>,
        quality: Option<&str>,
    ) -> PyResult<Self> {
        let wf = workflow.unwrap_or("led-wall");
        let valid_wf = ["led-wall", "hybrid", "green-screen", "ar"];
        if !valid_wf.contains(&wf) {
            return Err(PyValueError::new_err(format!(
                "Unknown workflow '{}'. Supported: {}",
                wf,
                valid_wf.join(", ")
            )));
        }

        let q = quality.unwrap_or("preview");
        let valid_q = ["draft", "preview", "final"];
        if !valid_q.contains(&q) {
            return Err(PyValueError::new_err(format!(
                "Unknown quality '{}'. Supported: {}",
                q,
                valid_q.join(", ")
            )));
        }

        let fps = target_fps.unwrap_or(60.0);
        if fps <= 0.0 || fps > 240.0 {
            return Err(PyValueError::new_err(format!(
                "Target FPS must be between 0 and 240, got {fps}"
            )));
        }

        Ok(Self {
            workflow: wf.to_string(),
            target_fps: fps,
            sync_accuracy_ms: 0.5,
            quality: q.to_string(),
            num_cameras: 1,
            color_calibration: true,
            lens_correction: true,
            motion_capture: false,
        })
    }

    /// Set the number of cameras.
    fn with_cameras(&mut self, count: usize) -> PyResult<()> {
        if count == 0 || count > 64 {
            return Err(PyValueError::new_err(format!(
                "Camera count must be 1-64, got {count}"
            )));
        }
        self.num_cameras = count;
        Ok(())
    }

    /// Set sync accuracy in milliseconds.
    fn with_sync_accuracy(&mut self, ms: f64) -> PyResult<()> {
        if ms <= 0.0 {
            return Err(PyValueError::new_err("Sync accuracy must be positive"));
        }
        self.sync_accuracy_ms = ms;
        Ok(())
    }

    /// Enable or disable motion capture.
    fn with_motion_capture(&mut self, enabled: bool) {
        self.motion_capture = enabled;
    }

    fn __repr__(&self) -> String {
        format!(
            "PyVirtualConfig(workflow='{}', fps={}, quality='{}', cameras={})",
            self.workflow, self.target_fps, self.quality, self.num_cameras,
        )
    }
}

// ---------------------------------------------------------------------------
// PyVirtualSource
// ---------------------------------------------------------------------------

/// A virtual production source (camera, LED wall, compositor).
#[pyclass]
#[derive(Clone)]
pub struct PyVirtualSource {
    /// Source name.
    #[pyo3(get)]
    pub name: String,
    /// Source type (camera, led-wall, compositor, genlock).
    #[pyo3(get)]
    pub source_type: String,
    /// Whether the source is active.
    #[pyo3(get)]
    pub active: bool,
    /// Source configuration as JSON.
    #[pyo3(get)]
    pub config_json: String,
}

#[pymethods]
impl PyVirtualSource {
    /// Create a new virtual source.
    #[new]
    fn new(name: &str, source_type: &str) -> PyResult<Self> {
        let valid_types = ["camera", "led-wall", "compositor", "genlock"];
        if !valid_types.contains(&source_type) {
            return Err(PyValueError::new_err(format!(
                "Unknown source type '{}'. Supported: {}",
                source_type,
                valid_types.join(", ")
            )));
        }
        Ok(Self {
            name: name.to_string(),
            source_type: source_type.to_string(),
            active: false,
            config_json: "{}".to_string(),
        })
    }

    /// Activate the source.
    fn activate(&mut self) {
        self.active = true;
    }

    /// Deactivate the source.
    fn deactivate(&mut self) {
        self.active = false;
    }

    /// Get source info as dict.
    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("name".to_string(), self.name.clone());
        m.insert("source_type".to_string(), self.source_type.clone());
        m.insert("active".to_string(), self.active.to_string());
        m
    }

    fn __repr__(&self) -> String {
        format!(
            "PyVirtualSource(name='{}', type='{}', active={})",
            self.name, self.source_type, self.active,
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Create a virtual camera source with default configuration.
///
/// Args:
///     name: Camera name.
///     fps: Target FPS (default: 60.0).
///
/// Returns:
///     JSON string with camera configuration.
#[pyfunction]
#[pyo3(signature = (name, fps=None))]
pub fn create_virtual_camera(name: &str, fps: Option<f64>) -> PyResult<String> {
    let f = fps.unwrap_or(60.0);
    if f <= 0.0 || f > 240.0 {
        return Err(PyValueError::new_err(format!(
            "FPS must be between 0 and 240, got {f}"
        )));
    }

    // Validate by constructing a real config
    let _config = oximedia_virtual::VirtualProductionConfig::default()
        .with_target_fps(f)
        .with_num_cameras(1);

    Ok(format!(
        "{{\"name\":\"{name}\",\"type\":\"camera\",\"fps\":{f},\"status\":\"created\"}}"
    ))
}

/// List supported virtual source types.
///
/// Returns:
///     List of supported source type strings.
#[pyfunction]
pub fn list_virtual_sources() -> Vec<String> {
    vec![
        "camera".to_string(),
        "led-wall".to_string(),
        "compositor".to_string(),
        "genlock".to_string(),
    ]
}

/// Get supported workflow types.
///
/// Returns:
///     List of workflow type strings.
#[pyfunction]
pub fn list_virtual_workflows() -> Vec<String> {
    vec![
        "led-wall".to_string(),
        "hybrid".to_string(),
        "green-screen".to_string(),
        "ar".to_string(),
    ]
}

/// Create a virtual production session with the given config.
///
/// Args:
///     config: PyVirtualConfig instance.
///
/// Returns:
///     JSON string with session details.
#[pyfunction]
pub fn create_virtual_session(config: &PyVirtualConfig) -> PyResult<String> {
    let wf = match config.workflow.as_str() {
        "led-wall" => oximedia_virtual::WorkflowType::LedWall,
        "hybrid" => oximedia_virtual::WorkflowType::Hybrid,
        "green-screen" => oximedia_virtual::WorkflowType::GreenScreen,
        "ar" => oximedia_virtual::WorkflowType::AugmentedReality,
        other => {
            return Err(PyValueError::new_err(format!("Unknown workflow '{other}'")));
        }
    };

    let vp_config = oximedia_virtual::VirtualProductionConfig::default()
        .with_workflow(wf)
        .with_target_fps(config.target_fps)
        .with_quality(match config.quality.as_str() {
            "draft" => oximedia_virtual::QualityMode::Draft,
            "final" => oximedia_virtual::QualityMode::Final,
            _ => oximedia_virtual::QualityMode::Preview,
        })
        .with_num_cameras(config.num_cameras)
        .with_sync_accuracy_ms(config.sync_accuracy_ms);

    let _vp = oximedia_virtual::VirtualProduction::new(vp_config)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create session: {e}")))?;

    Ok(format!(
        "{{\"workflow\":\"{}\",\"fps\":{},\"cameras\":{},\"quality\":\"{}\",\"status\":\"created\"}}",
        config.workflow, config.target_fps, config.num_cameras, config.quality,
    ))
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all virtual production bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVirtualConfig>()?;
    m.add_class::<PyVirtualSource>()?;
    m.add_function(wrap_pyfunction!(create_virtual_camera, m)?)?;
    m.add_function(wrap_pyfunction!(list_virtual_sources, m)?)?;
    m.add_function(wrap_pyfunction!(list_virtual_workflows, m)?)?;
    m.add_function(wrap_pyfunction!(create_virtual_session, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let cfg = PyVirtualConfig::new(None, None, None);
        assert!(cfg.is_ok());
        let cfg = cfg.expect("config should be valid");
        assert_eq!(cfg.workflow, "led-wall");
        assert_eq!(cfg.target_fps, 60.0);
        assert_eq!(cfg.quality, "preview");
    }

    #[test]
    fn test_config_invalid_workflow() {
        let result = PyVirtualConfig::new(Some("invalid"), None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_invalid_quality() {
        let result = PyVirtualConfig::new(None, None, Some("ultra"));
        assert!(result.is_err());
    }

    #[test]
    fn test_source_creation() {
        let src = PyVirtualSource::new("cam1", "camera");
        assert!(src.is_ok());
        let src = src.expect("should succeed");
        assert_eq!(src.name, "cam1");
        assert!(!src.active);
    }

    #[test]
    fn test_source_invalid_type() {
        let result = PyVirtualSource::new("x", "invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_list_sources() {
        let sources = list_virtual_sources();
        assert_eq!(sources.len(), 4);
        assert!(sources.contains(&"camera".to_string()));
    }
}
