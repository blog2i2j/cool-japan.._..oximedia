//! Python bindings for `oximedia-gpu` GPU device management and acceleration.
//!
//! Provides `PyGpuInfo`, `PyGpuDevice`, `PyGpuAccelerator`, and standalone
//! functions for GPU enumeration and benchmarking.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PyGpuInfo
// ---------------------------------------------------------------------------

/// Information about a GPU device.
#[pyclass]
#[derive(Clone)]
pub struct PyGpuInfo {
    /// Device name.
    #[pyo3(get)]
    pub name: String,
    /// Vendor name or ID.
    #[pyo3(get)]
    pub vendor: String,
    /// Device type (discrete, integrated, virtual, cpu, unknown).
    #[pyo3(get)]
    pub device_type: String,
    /// Estimated VRAM in megabytes (0 if unknown).
    #[pyo3(get)]
    pub memory_mb: u64,
    /// Estimated compute units (0 if unknown).
    #[pyo3(get)]
    pub compute_units: u32,
    /// Whether the device supports hardware video encoding.
    #[pyo3(get)]
    pub supports_encode: bool,
    /// Whether the device supports hardware video decoding.
    #[pyo3(get)]
    pub supports_decode: bool,
    /// Backend API (vulkan, metal, dx12, cpu).
    #[pyo3(get)]
    pub api: String,
}

#[pymethods]
impl PyGpuInfo {
    fn __repr__(&self) -> String {
        format!(
            "PyGpuInfo(name='{}', vendor='{}', type='{}', api='{}', mem={}MB)",
            self.name, self.vendor, self.device_type, self.api, self.memory_mb,
        )
    }

    /// Convert to a JSON string representation.
    fn to_dict(&self) -> String {
        format!(
            "{{\"name\":\"{}\",\"vendor\":\"{}\",\"device_type\":\"{}\",\
             \"memory_mb\":{},\"compute_units\":{},\"supports_encode\":{},\
             \"supports_decode\":{},\"api\":\"{}\"}}",
            self.name,
            self.vendor,
            self.device_type,
            self.memory_mb,
            self.compute_units,
            self.supports_encode,
            self.supports_decode,
            self.api,
        )
    }
}

/// Convert a `GpuDeviceInfo` from the GPU crate into a `PyGpuInfo`.
fn device_info_to_py(info: &oximedia_gpu::GpuDeviceInfo) -> PyGpuInfo {
    let vendor_name = match info.vendor {
        0x10DE => "NVIDIA".to_string(),
        0x1002 => "AMD".to_string(),
        0x8086 => "Intel".to_string(),
        0x106B => "Apple".to_string(),
        other => format!("0x{:04X}", other),
    };

    PyGpuInfo {
        name: info.name.clone(),
        vendor: vendor_name,
        device_type: info.device_type.clone(),
        memory_mb: 0, // WGPU does not expose VRAM size directly
        compute_units: 0,
        supports_encode: false, // GPU encode/decode is feature-gated
        supports_decode: false,
        api: info.backend.clone(),
    }
}

// ---------------------------------------------------------------------------
// PyGpuDevice
// ---------------------------------------------------------------------------

/// A selected GPU device handle.
#[pyclass]
pub struct PyGpuDevice {
    info: PyGpuInfo,
    device_index: Option<usize>,
}

#[pymethods]
impl PyGpuDevice {
    /// Select a GPU device by index.
    ///
    /// Args:
    ///     index: Device index (from list_gpu_devices).
    #[new]
    fn new(index: usize) -> PyResult<Self> {
        let devices = oximedia_gpu::GpuDevice::list_devices()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to list GPU devices: {e}")))?;

        if index >= devices.len() {
            return Err(PyValueError::new_err(format!(
                "Device index {} out of range (0..{})",
                index,
                devices.len()
            )));
        }

        let info = device_info_to_py(&devices[index]);
        Ok(Self {
            info,
            device_index: Some(index),
        })
    }

    /// Get the default (primary) GPU device.
    #[staticmethod]
    fn default_device() -> PyResult<Self> {
        let devices = oximedia_gpu::GpuDevice::list_devices()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to list GPU devices: {e}")))?;

        if devices.is_empty() {
            return Err(PyRuntimeError::new_err("No GPU devices available"));
        }

        let info = device_info_to_py(&devices[0]);
        Ok(Self {
            info,
            device_index: Some(0),
        })
    }

    /// Get device information.
    fn info(&self) -> PyGpuInfo {
        self.info.clone()
    }

    /// Check if the GPU device is available and operational.
    fn is_available(&self) -> bool {
        if let Some(idx) = self.device_index {
            oximedia_gpu::GpuContext::with_device(idx).is_ok()
        } else {
            oximedia_gpu::GpuContext::new().is_ok()
        }
    }

    /// Get estimated available VRAM in megabytes.
    ///
    /// Note: WGPU does not expose VRAM directly; returns 0 as a placeholder.
    fn memory_available_mb(&self) -> u64 {
        0
    }

    fn __repr__(&self) -> String {
        format!(
            "PyGpuDevice(index={}, name='{}', api='{}')",
            self.device_index.unwrap_or(0),
            self.info.name,
            self.info.api,
        )
    }
}

// ---------------------------------------------------------------------------
// PyGpuAccelerator
// ---------------------------------------------------------------------------

/// GPU-accelerated frame processing.
#[pyclass]
pub struct PyGpuAccelerator {
    device_index: Option<usize>,
    enabled: bool,
}

#[pymethods]
impl PyGpuAccelerator {
    /// Create a GPU accelerator for a specific device.
    ///
    /// Args:
    ///     device: PyGpuDevice to use for acceleration.
    #[new]
    fn new(device: &PyGpuDevice) -> Self {
        Self {
            device_index: device.device_index,
            enabled: true,
        }
    }

    /// Enable GPU acceleration.
    fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable GPU acceleration (fall back to CPU).
    fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if GPU acceleration is enabled.
    fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Process a frame using GPU-accelerated operations.
    ///
    /// Args:
    ///     data: Raw RGBA pixel data (width * height * 4 bytes).
    ///     width: Frame width.
    ///     height: Frame height.
    ///     operation: Operation name (blur, sharpen, edge_detect, rgb_to_yuv, yuv_to_rgb).
    ///     params: Operation parameters as a dict (e.g. {"sigma": 2.0} for blur).
    ///
    /// Returns:
    ///     Processed RGBA pixel data.
    #[pyo3(signature = (data, width, height, operation, params=None))]
    fn process_frame(
        &self,
        data: Vec<u8>,
        width: u32,
        height: u32,
        operation: &str,
        params: Option<HashMap<String, f64>>,
    ) -> PyResult<Vec<u8>> {
        if !self.enabled {
            return Err(PyRuntimeError::new_err(
                "GPU acceleration is disabled. Call enable() first",
            ));
        }

        let expected = (width as usize) * (height as usize) * 4;
        if data.len() < expected {
            return Err(PyValueError::new_err(format!(
                "Frame data too small: need {} bytes for {}x{} RGBA, got {}",
                expected,
                width,
                height,
                data.len()
            )));
        }

        let ctx = if let Some(idx) = self.device_index {
            oximedia_gpu::GpuContext::with_device(idx)
        } else {
            oximedia_gpu::GpuContext::new()
        }
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create GPU context: {e}")))?;

        let input = &data[..expected];
        let mut output = vec![0u8; expected];
        let params_map = params.unwrap_or_default();

        match operation {
            "blur" => {
                let sigma = params_map.get("sigma").copied().unwrap_or(2.0) as f32;
                ctx.gaussian_blur(input, &mut output, width, height, sigma)
                    .map_err(|e| PyRuntimeError::new_err(format!("Blur failed: {e}")))?;
            }
            "sharpen" => {
                let amount = params_map.get("amount").copied().unwrap_or(1.0) as f32;
                ctx.sharpen(input, &mut output, width, height, amount)
                    .map_err(|e| PyRuntimeError::new_err(format!("Sharpen failed: {e}")))?;
            }
            "edge_detect" => {
                ctx.edge_detect(input, &mut output, width, height)
                    .map_err(|e| PyRuntimeError::new_err(format!("Edge detect failed: {e}")))?;
            }
            "rgb_to_yuv" => {
                ctx.rgb_to_yuv(input, &mut output)
                    .map_err(|e| PyRuntimeError::new_err(format!("RGB to YUV failed: {e}")))?;
            }
            "yuv_to_rgb" => {
                ctx.yuv_to_rgb(input, &mut output)
                    .map_err(|e| PyRuntimeError::new_err(format!("YUV to RGB failed: {e}")))?;
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown operation '{}'. Use available_operations() to list supported operations",
                    other
                )));
            }
        }

        Ok(output)
    }

    /// List available GPU-accelerated operations.
    fn available_operations(&self) -> Vec<String> {
        vec![
            "blur".to_string(),
            "sharpen".to_string(),
            "edge_detect".to_string(),
            "rgb_to_yuv".to_string(),
            "yuv_to_rgb".to_string(),
        ]
    }

    /// Benchmark a GPU operation for the given frame dimensions.
    ///
    /// Args:
    ///     width: Frame width.
    ///     height: Frame height.
    ///     operation: Operation to benchmark.
    ///
    /// Returns:
    ///     Estimated milliseconds per frame.
    fn benchmark(&self, width: u32, height: u32, operation: &str) -> PyResult<f64> {
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err("Width and height must be > 0"));
        }

        // Validate operation
        let valid_ops = ["blur", "sharpen", "edge_detect", "rgb_to_yuv", "yuv_to_rgb"];
        if !valid_ops.contains(&operation) {
            return Err(PyValueError::new_err(format!(
                "Unknown operation '{}'. Supported: {}",
                operation,
                valid_ops.join(", ")
            )));
        }

        let ctx = if let Some(idx) = self.device_index {
            oximedia_gpu::GpuContext::with_device(idx)
        } else {
            oximedia_gpu::GpuContext::new()
        }
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create GPU context: {e}")))?;

        let size = (width as usize) * (height as usize) * 4;
        let input = vec![128u8; size];
        let mut output = vec![0u8; size];

        let start = std::time::Instant::now();
        let iterations = 5;

        for _ in 0..iterations {
            match operation {
                "blur" => {
                    let _ = ctx.gaussian_blur(&input, &mut output, width, height, 2.0);
                }
                "sharpen" => {
                    let _ = ctx.sharpen(&input, &mut output, width, height, 1.0);
                }
                "edge_detect" => {
                    let _ = ctx.edge_detect(&input, &mut output, width, height);
                }
                "rgb_to_yuv" => {
                    let _ = ctx.rgb_to_yuv(&input, &mut output);
                }
                "yuv_to_rgb" => {
                    let _ = ctx.yuv_to_rgb(&input, &mut output);
                }
                _ => {}
            }
            ctx.wait();
        }

        let elapsed = start.elapsed();
        let ms_per_frame = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
        Ok(ms_per_frame)
    }

    fn __repr__(&self) -> String {
        format!(
            "PyGpuAccelerator(device={}, enabled={})",
            self.device_index.unwrap_or(0),
            self.enabled,
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// List all available GPU devices.
///
/// Returns:
///     List of PyGpuInfo for each available GPU.
#[pyfunction]
pub fn list_gpu_devices() -> PyResult<Vec<PyGpuInfo>> {
    let devices = oximedia_gpu::GpuDevice::list_devices()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to list GPU devices: {e}")))?;

    Ok(devices.iter().map(device_info_to_py).collect())
}

/// Check if any GPU device is available.
///
/// Returns:
///     True if at least one GPU device is available.
#[pyfunction]
pub fn is_gpu_available() -> bool {
    oximedia_gpu::GpuContext::new().is_ok()
}

/// Benchmark GPU operations at the given resolution.
///
/// Args:
///     width: Frame width.
///     height: Frame height.
///
/// Returns:
///     Dict mapping operation name to milliseconds per frame.
#[pyfunction]
pub fn gpu_benchmark(width: u32, height: u32) -> PyResult<HashMap<String, f64>> {
    if width == 0 || height == 0 {
        return Err(PyValueError::new_err("Width and height must be > 0"));
    }

    let ctx = oximedia_gpu::GpuContext::new()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create GPU context: {e}")))?;

    let size = (width as usize) * (height as usize) * 4;
    let input = vec![128u8; size];
    let mut output = vec![0u8; size];
    let iterations = 3;

    let mut results = HashMap::new();

    // Benchmark blur
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = ctx.gaussian_blur(&input, &mut output, width, height, 2.0);
        ctx.wait();
    }
    results.insert(
        "blur".to_string(),
        start.elapsed().as_secs_f64() * 1000.0 / iterations as f64,
    );

    // Benchmark sharpen
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = ctx.sharpen(&input, &mut output, width, height, 1.0);
        ctx.wait();
    }
    results.insert(
        "sharpen".to_string(),
        start.elapsed().as_secs_f64() * 1000.0 / iterations as f64,
    );

    // Benchmark edge_detect
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = ctx.edge_detect(&input, &mut output, width, height);
        ctx.wait();
    }
    results.insert(
        "edge_detect".to_string(),
        start.elapsed().as_secs_f64() * 1000.0 / iterations as f64,
    );

    // Benchmark rgb_to_yuv
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = ctx.rgb_to_yuv(&input, &mut output);
        ctx.wait();
    }
    results.insert(
        "rgb_to_yuv".to_string(),
        start.elapsed().as_secs_f64() * 1000.0 / iterations as f64,
    );

    Ok(results)
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all GPU bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGpuInfo>()?;
    m.add_class::<PyGpuDevice>()?;
    m.add_class::<PyGpuAccelerator>()?;
    m.add_function(wrap_pyfunction!(list_gpu_devices, m)?)?;
    m.add_function(wrap_pyfunction!(is_gpu_available, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_benchmark, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_info_to_py_nvidia() {
        let info = oximedia_gpu::GpuDeviceInfo {
            name: "GeForce RTX 4090".to_string(),
            vendor: 0x10DE,
            device: 0x2684,
            device_type: "discrete".to_string(),
            backend: "vulkan".to_string(),
        };
        let py_info = device_info_to_py(&info);
        assert_eq!(py_info.vendor, "NVIDIA");
        assert_eq!(py_info.api, "vulkan");
    }

    #[test]
    fn test_device_info_to_py_amd() {
        let info = oximedia_gpu::GpuDeviceInfo {
            name: "Radeon RX 7900".to_string(),
            vendor: 0x1002,
            device: 0x0001,
            device_type: "discrete".to_string(),
            backend: "vulkan".to_string(),
        };
        let py_info = device_info_to_py(&info);
        assert_eq!(py_info.vendor, "AMD");
    }

    #[test]
    fn test_device_info_to_py_apple() {
        let info = oximedia_gpu::GpuDeviceInfo {
            name: "Apple M2 Max".to_string(),
            vendor: 0x106B,
            device: 0x0001,
            device_type: "integrated".to_string(),
            backend: "metal".to_string(),
        };
        let py_info = device_info_to_py(&info);
        assert_eq!(py_info.vendor, "Apple");
        assert_eq!(py_info.api, "metal");
    }

    #[test]
    fn test_available_operations() {
        let device = PyGpuDevice {
            info: PyGpuInfo {
                name: "test".to_string(),
                vendor: "test".to_string(),
                device_type: "cpu".to_string(),
                memory_mb: 0,
                compute_units: 0,
                supports_encode: false,
                supports_decode: false,
                api: "cpu".to_string(),
            },
            device_index: None,
        };
        let accel = PyGpuAccelerator::new(&device);
        let ops = accel.available_operations();
        assert!(ops.contains(&"blur".to_string()));
        assert!(ops.contains(&"sharpen".to_string()));
        assert_eq!(ops.len(), 5);
    }

    #[test]
    fn test_is_gpu_available_runs() {
        // Just ensure it does not panic
        let _ = is_gpu_available();
    }
}
