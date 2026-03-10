//! Python bindings for `oximedia-scaling` video/image scaling.
//!
//! Provides `PyScaler`, `PyScaleConfig`, `PyScaleResult`, and standalone
//! functions for upscale, downscale, and quality comparison.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PyScaleConfig
// ---------------------------------------------------------------------------

/// Configuration for a scaling operation.
#[pyclass]
#[derive(Clone)]
pub struct PyScaleConfig {
    /// Target width.
    #[pyo3(get)]
    pub width: u32,
    /// Target height.
    #[pyo3(get)]
    pub height: u32,
    /// Scaling algorithm (bilinear, bicubic, lanczos).
    #[pyo3(get)]
    pub algorithm: String,
    /// Aspect ratio mode (stretch, letterbox, crop).
    #[pyo3(get)]
    pub aspect_mode: String,
}

#[pymethods]
impl PyScaleConfig {
    /// Create a new scale configuration.
    ///
    /// Args:
    ///     width: Target width in pixels.
    ///     height: Target height in pixels.
    ///     algorithm: Scaling algorithm (default: lanczos).
    ///     aspect_mode: Aspect ratio handling (default: letterbox).
    #[new]
    #[pyo3(signature = (width, height, algorithm=None, aspect_mode=None))]
    fn new(
        width: u32,
        height: u32,
        algorithm: Option<&str>,
        aspect_mode: Option<&str>,
    ) -> PyResult<Self> {
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err("Width and height must be > 0"));
        }
        if width > 7680 || height > 4320 {
            return Err(PyValueError::new_err(format!(
                "Dimensions exceed 7680x4320: {}x{}",
                width, height
            )));
        }

        let alg = algorithm.unwrap_or("lanczos");
        let valid_alg = ["bilinear", "bicubic", "lanczos"];
        if !valid_alg.contains(&alg) {
            return Err(PyValueError::new_err(format!(
                "Unknown algorithm '{}'. Supported: {}",
                alg,
                valid_alg.join(", ")
            )));
        }

        let asp = aspect_mode.unwrap_or("letterbox");
        let valid_asp = ["stretch", "letterbox", "crop"];
        if !valid_asp.contains(&asp) {
            return Err(PyValueError::new_err(format!(
                "Unknown aspect mode '{}'. Supported: {}",
                asp,
                valid_asp.join(", ")
            )));
        }

        Ok(Self {
            width,
            height,
            algorithm: alg.to_string(),
            aspect_mode: asp.to_string(),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PyScaleConfig({}x{}, alg='{}', aspect='{}')",
            self.width, self.height, self.algorithm, self.aspect_mode,
        )
    }
}

// ---------------------------------------------------------------------------
// PyScaleResult
// ---------------------------------------------------------------------------

/// Result of a scaling operation.
#[pyclass]
#[derive(Clone)]
pub struct PyScaleResult {
    /// Source width.
    #[pyo3(get)]
    pub src_width: u32,
    /// Source height.
    #[pyo3(get)]
    pub src_height: u32,
    /// Output width.
    #[pyo3(get)]
    pub dst_width: u32,
    /// Output height.
    #[pyo3(get)]
    pub dst_height: u32,
    /// Algorithm used.
    #[pyo3(get)]
    pub algorithm: String,
    /// Scale factor X.
    #[pyo3(get)]
    pub scale_factor_x: f64,
    /// Scale factor Y.
    #[pyo3(get)]
    pub scale_factor_y: f64,
}

#[pymethods]
impl PyScaleResult {
    fn __repr__(&self) -> String {
        format!(
            "PyScaleResult({}x{} -> {}x{}, scale={:.3}x/{:.3}x)",
            self.src_width,
            self.src_height,
            self.dst_width,
            self.dst_height,
            self.scale_factor_x,
            self.scale_factor_y,
        )
    }

    /// Convert to dict.
    fn to_dict(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("src_width".to_string(), self.src_width as f64);
        m.insert("src_height".to_string(), self.src_height as f64);
        m.insert("dst_width".to_string(), self.dst_width as f64);
        m.insert("dst_height".to_string(), self.dst_height as f64);
        m.insert("scale_factor_x".to_string(), self.scale_factor_x);
        m.insert("scale_factor_y".to_string(), self.scale_factor_y);
        m
    }

    /// Check if this is an upscale operation.
    fn is_upscale(&self) -> bool {
        self.scale_factor_x > 1.0 || self.scale_factor_y > 1.0
    }
}

// ---------------------------------------------------------------------------
// PyScaler
// ---------------------------------------------------------------------------

/// Video/image scaler.
#[pyclass]
pub struct PyScaler {
    config: PyScaleConfig,
}

#[pymethods]
impl PyScaler {
    /// Create a new scaler.
    #[new]
    fn new(config: PyScaleConfig) -> Self {
        Self { config }
    }

    /// Calculate output dimensions for a given source.
    ///
    /// Args:
    ///     src_width: Source width.
    ///     src_height: Source height.
    ///
    /// Returns:
    ///     PyScaleResult with calculated dimensions.
    fn calculate(&self, src_width: u32, src_height: u32) -> PyResult<PyScaleResult> {
        if src_width == 0 || src_height == 0 {
            return Err(PyValueError::new_err("Source dimensions must be > 0"));
        }

        let mode = match self.config.algorithm.as_str() {
            "bilinear" => oximedia_scaling::ScalingMode::Bilinear,
            "bicubic" => oximedia_scaling::ScalingMode::Bicubic,
            _ => oximedia_scaling::ScalingMode::Lanczos,
        };

        let aspect = match self.config.aspect_mode.as_str() {
            "stretch" => oximedia_scaling::AspectRatioMode::Stretch,
            "crop" => oximedia_scaling::AspectRatioMode::Crop,
            _ => oximedia_scaling::AspectRatioMode::Letterbox,
        };

        let params = oximedia_scaling::ScalingParams::new(self.config.width, self.config.height)
            .with_mode(mode)
            .with_aspect_ratio(aspect);

        let scaler = oximedia_scaling::VideoScaler::new(params);
        let (dst_w, dst_h) = scaler.calculate_dimensions(src_width, src_height);

        let scale_x = dst_w as f64 / src_width as f64;
        let scale_y = dst_h as f64 / src_height as f64;

        Ok(PyScaleResult {
            src_width,
            src_height,
            dst_width: dst_w,
            dst_height: dst_h,
            algorithm: self.config.algorithm.clone(),
            scale_factor_x: scale_x,
            scale_factor_y: scale_y,
        })
    }

    /// Get the current configuration.
    fn get_config(&self) -> PyScaleConfig {
        self.config.clone()
    }

    fn __repr__(&self) -> String {
        format!("PyScaler(config={})", self.config.__repr__())
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Upscale dimensions calculator.
///
/// Args:
///     src_width: Source width.
///     src_height: Source height.
///     dst_width: Target width.
///     dst_height: Target height.
///     algorithm: Scaling algorithm (default: lanczos).
///
/// Returns:
///     PyScaleResult.
#[pyfunction]
#[pyo3(signature = (src_width, src_height, dst_width, dst_height, algorithm=None))]
pub fn upscale(
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    algorithm: Option<&str>,
) -> PyResult<PyScaleResult> {
    let config = PyScaleConfig::new(dst_width, dst_height, algorithm, None)?;
    let scaler = PyScaler::new(config);
    scaler.calculate(src_width, src_height)
}

/// Downscale dimensions calculator.
///
/// Args:
///     src_width: Source width.
///     src_height: Source height.
///     dst_width: Target width.
///     dst_height: Target height.
///     algorithm: Scaling algorithm (default: lanczos).
///
/// Returns:
///     PyScaleResult.
#[pyfunction]
#[pyo3(signature = (src_width, src_height, dst_width, dst_height, algorithm=None))]
pub fn downscale(
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    algorithm: Option<&str>,
) -> PyResult<PyScaleResult> {
    let config = PyScaleConfig::new(dst_width, dst_height, algorithm, None)?;
    let scaler = PyScaler::new(config);
    scaler.calculate(src_width, src_height)
}

/// Compare quality between scaling algorithms.
///
/// Args:
///     src_width: Source width.
///     src_height: Source height.
///     dst_width: Target width.
///     dst_height: Target height.
///
/// Returns:
///     JSON string comparing all algorithms.
#[pyfunction]
pub fn compare_quality(
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> PyResult<String> {
    if src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0 {
        return Err(PyValueError::new_err("All dimensions must be > 0"));
    }

    let algorithms = ["bilinear", "bicubic", "lanczos"];
    let mut parts = Vec::new();

    for alg in &algorithms {
        let config = PyScaleConfig::new(dst_width, dst_height, Some(alg), None)?;
        let scaler = PyScaler::new(config);
        let result = scaler.calculate(src_width, src_height)?;
        parts.push(format!(
            "{{\"algorithm\":\"{alg}\",\"output\":\"{}x{}\",\"scale_x\":{:.4},\"scale_y\":{:.4}}}",
            result.dst_width, result.dst_height, result.scale_factor_x, result.scale_factor_y
        ));
    }

    Ok(format!("[{}]", parts.join(",")))
}

/// List available scaling algorithms.
#[pyfunction]
pub fn list_scaling_algorithms() -> Vec<String> {
    vec![
        "bilinear".to_string(),
        "bicubic".to_string(),
        "lanczos".to_string(),
    ]
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all scaling bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScaler>()?;
    m.add_class::<PyScaleConfig>()?;
    m.add_class::<PyScaleResult>()?;
    m.add_function(wrap_pyfunction!(upscale, m)?)?;
    m.add_function(wrap_pyfunction!(downscale, m)?)?;
    m.add_function(wrap_pyfunction!(compare_quality, m)?)?;
    m.add_function(wrap_pyfunction!(list_scaling_algorithms, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_config_defaults() {
        let cfg = PyScaleConfig::new(1920, 1080, None, None);
        assert!(cfg.is_ok());
        let cfg = cfg.expect("should succeed");
        assert_eq!(cfg.algorithm, "lanczos");
        assert_eq!(cfg.aspect_mode, "letterbox");
    }

    #[test]
    fn test_scale_config_invalid_dims() {
        assert!(PyScaleConfig::new(0, 100, None, None).is_err());
        assert!(PyScaleConfig::new(100, 0, None, None).is_err());
    }

    #[test]
    fn test_scaler_calculate() {
        let cfg = PyScaleConfig::new(1920, 1080, None, Some("stretch")).expect("should succeed");
        let scaler = PyScaler::new(cfg);
        let result = scaler.calculate(3840, 2160).expect("should succeed");
        assert_eq!(result.dst_width, 1920);
        assert_eq!(result.dst_height, 1080);
    }

    #[test]
    fn test_upscale_function() {
        let result = upscale(1920, 1080, 3840, 2160, None);
        assert!(result.is_ok());
        let r = result.expect("should succeed");
        assert!(r.is_upscale());
    }

    #[test]
    fn test_compare_quality() {
        let result = compare_quality(1920, 1080, 3840, 2160);
        assert!(result.is_ok());
        let json = result.expect("should succeed");
        assert!(json.contains("bilinear"));
        assert!(json.contains("bicubic"));
        assert!(json.contains("lanczos"));
    }
}
