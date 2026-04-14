//! ONNX Runtime wrapper for model inference.
//!
//! This module provides a safe wrapper around ONNX Runtime, managing
//! sessions, device selection, and model loading.

use crate::error::{CvError, CvResult};
use crate::ml::tensor::Tensor;
use std::path::Path;
use std::sync::Arc;

#[cfg(feature = "onnx")]
use oxionnx::Session as OxiSession;

/// Device type for model execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceType {
    /// CPU execution.
    #[default]
    Cpu,
    /// NVIDIA CUDA GPU.
    Cuda,
    /// AMD ROCm GPU.
    Rocm,
    /// NVIDIA TensorRT.
    TensorRt,
    /// DirectML (Windows).
    DirectMl,
    /// Apple CoreML.
    CoreMl,
}

/// ONNX Runtime wrapper.
///
/// Manages ONNX Runtime environment and provides methods
/// to create inference sessions.
///
/// # Example
///
/// ```no_run
/// use oximedia_cv::ml::{OnnxRuntime, DeviceType};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let runtime = OnnxRuntime::new()?;
/// let session = runtime.load_model("model.onnx")?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct OnnxRuntime {
    device: DeviceType,
}

impl OnnxRuntime {
    /// Create a new ONNX Runtime instance with CPU device.
    ///
    /// # Errors
    ///
    /// Returns an error if ONNX Runtime initialization fails.
    pub fn new() -> CvResult<Self> {
        Self::with_device(DeviceType::Cpu)
    }

    /// Create a new ONNX Runtime instance with specified device.
    ///
    /// # Arguments
    ///
    /// * `device` - Target device for execution
    ///
    /// # Errors
    ///
    /// Returns an error if ONNX Runtime initialization fails or
    /// the specified device is not available.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use oximedia_cv::ml::{OnnxRuntime, DeviceType};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let runtime = OnnxRuntime::with_device(DeviceType::Cuda)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_device(device: DeviceType) -> CvResult<Self> {
        // Initialize ort (this is done automatically on first use)
        Ok(Self { device })
    }

    /// Load an ONNX model from a file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the ONNX model file
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - File cannot be read
    /// - Model format is invalid
    /// - Session creation fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// use oximedia_cv::ml::OnnxRuntime;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let runtime = OnnxRuntime::new()?;
    /// let session = runtime.load_model("model.onnx")?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "onnx")]
    pub fn load_model(&self, path: impl AsRef<Path>) -> CvResult<Session> {
        let path = path.as_ref();

        // Build session with device configuration
        let session = self.build_session(path)?;

        Ok(Session {
            inner: Arc::new(std::sync::Mutex::new(session)),
        })
    }

    /// Load an ONNX model from bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Model data as bytes
    ///
    /// # Errors
    ///
    /// Returns an error if model format is invalid or session creation fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use oximedia_cv::ml::OnnxRuntime;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let model_bytes = std::fs::read("model.onnx")?;
    /// let runtime = OnnxRuntime::new()?;
    /// let session = runtime.load_model_from_bytes(&model_bytes)?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "onnx")]
    pub fn load_model_from_bytes(&self, bytes: &[u8]) -> CvResult<Session> {
        let session = self.build_session_from_bytes(bytes)?;

        Ok(Session {
            inner: Arc::new(std::sync::Mutex::new(session)),
        })
    }

    /// Get the configured device type.
    #[must_use]
    pub const fn device(&self) -> DeviceType {
        self.device
    }

    /// Build an oxionnx session from file.
    #[cfg(feature = "onnx")]
    fn build_session(&self, path: &Path) -> CvResult<OxiSession> {
        // For GPU device types, ort-backend is required
        self.check_device_support()?;

        oxionnx::Session::builder()
            .with_optimization_level(oxionnx::OptLevel::All)
            .load(path)
            .map_err(|e| CvError::model_load(format!("Failed to load model from file: {e}")))
    }

    /// Build an oxionnx session from bytes.
    #[cfg(feature = "onnx")]
    fn build_session_from_bytes(&self, bytes: &[u8]) -> CvResult<OxiSession> {
        // For GPU device types, ort-backend is required
        self.check_device_support()?;

        oxionnx::Session::builder()
            .with_optimization_level(oxionnx::OptLevel::All)
            .load_from_bytes(bytes)
            .map_err(|e| CvError::model_load(format!("Failed to load model from bytes: {e}")))
    }

    /// Validate that the requested device is supported.
    #[cfg(feature = "onnx")]
    fn check_device_support(&self) -> CvResult<()> {
        match self.device {
            DeviceType::Cpu => Ok(()),
            // CUDA: Pure Rust path via oxionnx/cuda (OxiCUDA backend).
            // Gracefully falls back to CPU at runtime when no GPU is present.
            #[cfg(feature = "cuda")]
            DeviceType::Cuda => Ok(()),
            #[cfg(not(feature = "cuda"))]
            DeviceType::Cuda => Err(CvError::onnx_runtime(
                "CUDA support requires the 'cuda' feature".to_owned(),
            )),
            // ROCm: Pure Rust CPU inference (GPU backend planned).
            #[cfg(feature = "rocm")]
            DeviceType::Rocm => Ok(()),
            #[cfg(not(feature = "rocm"))]
            DeviceType::Rocm => Err(CvError::onnx_runtime(
                "ROCm support requires the 'rocm' feature".to_owned(),
            )),
            // TensorRT: Pure Rust CPU inference (GPU backend planned).
            #[cfg(feature = "tensorrt")]
            DeviceType::TensorRt => Ok(()),
            #[cfg(not(feature = "tensorrt"))]
            DeviceType::TensorRt => Err(CvError::onnx_runtime(
                "TensorRT support requires the 'tensorrt' feature".to_owned(),
            )),
            // DirectML: Windows-only (no current Pure Rust backend).
            #[cfg(target_os = "windows")]
            DeviceType::DirectMl => Ok(()),
            #[cfg(not(target_os = "windows"))]
            DeviceType::DirectMl => Err(CvError::onnx_runtime(
                "DirectML is only available on Windows".to_owned(),
            )),
            // CoreML: macOS-only (no current Pure Rust backend).
            #[cfg(target_os = "macos")]
            DeviceType::CoreMl => Ok(()),
            #[cfg(not(target_os = "macos"))]
            DeviceType::CoreMl => Err(CvError::onnx_runtime(
                "CoreML is only available on macOS".to_owned(),
            )),
        }
    }
}

impl Default for OnnxRuntime {
    fn default() -> Self {
        Self {
            device: DeviceType::Cpu,
        }
    }
}

/// ONNX inference session.
///
/// Represents a loaded ONNX model ready for inference.
/// Sessions are thread-safe and can be cloned cheaply.
#[cfg(feature = "onnx")]
#[derive(Clone)]
pub struct Session {
    inner: Arc<std::sync::Mutex<OxiSession>>,
}

#[cfg(feature = "onnx")]
impl Session {
    /// Run inference with input tensors.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Input tensors for the model
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Input shape/type doesn't match model requirements
    /// - Inference fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// use oximedia_cv::ml::{OnnxRuntime, Tensor};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let runtime = OnnxRuntime::new()?;
    /// let session = runtime.load_model("model.onnx")?;
    ///
    /// let input = Tensor::zeros(&[1, 3, 224, 224]);
    /// let outputs = session.run(&[input])?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn run(&self, inputs: &[Tensor]) -> CvResult<Vec<Tensor>> {
        // Convert our tensors to oxionnx tensors
        let oxi_inputs: Vec<oxionnx_core::Tensor> = inputs
            .iter()
            .map(super::tensor::Tensor::to_oxionnx_tensor)
            .collect::<CvResult<Vec<_>>>()?;

        // Lock the session for access
        let session = self
            .inner
            .lock()
            .map_err(|e| CvError::onnx_runtime(format!("Session lock error: {e}")))?;

        // Get input names from the session to build named inputs
        let input_names: Vec<String> = session.input_names().to_vec();

        if oxi_inputs.len() > input_names.len() {
            return Err(CvError::onnx_runtime(format!(
                "Too many inputs: got {}, model expects {}",
                oxi_inputs.len(),
                input_names.len()
            )));
        }

        // Build named input HashMap and run inference
        let mut named_inputs = std::collections::HashMap::new();
        for (name, tensor) in input_names.iter().zip(oxi_inputs) {
            named_inputs.insert(name.as_str(), tensor);
        }

        let outputs = session
            .run(&named_inputs)
            .map_err(|e| CvError::onnx_runtime(format!("Inference failed: {e}")))?;

        // Collect outputs in order of output_names to preserve Vec ordering
        let output_names = session.output_names().to_vec();
        output_names
            .iter()
            .filter_map(|name| outputs.get(name))
            .map(super::tensor::Tensor::from_oxionnx_tensor)
            .collect()
    }

    /// Get input names for the model.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use oximedia_cv::ml::OnnxRuntime;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let runtime = OnnxRuntime::new()?;
    /// let session = runtime.load_model("model.onnx")?;
    /// let input_names = session.input_names();
    /// println!("Model inputs: {:?}", input_names);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn input_names(&self) -> Vec<String> {
        self.inner
            .lock()
            .map_or_else(|_| Vec::new(), |s| s.input_names().to_vec())
    }

    /// Get output names for the model.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use oximedia_cv::ml::OnnxRuntime;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let runtime = OnnxRuntime::new()?;
    /// let session = runtime.load_model("model.onnx")?;
    /// let output_names = session.output_names();
    /// println!("Model outputs: {:?}", output_names);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn output_names(&self) -> Vec<String> {
        self.inner
            .lock()
            .map_or_else(|_| Vec::new(), |s| s.output_names().to_vec())
    }

    /// Get the number of inputs.
    #[must_use]
    pub fn input_count(&self) -> usize {
        self.inner.lock().map_or(0, |s| s.input_names().len())
    }

    /// Get the number of outputs.
    #[must_use]
    pub fn output_count(&self) -> usize {
        self.inner.lock().map_or(0, |s| s.output_names().len())
    }
}

/// Stub Session for when the onnx feature is not enabled.
#[cfg(not(feature = "onnx"))]
#[derive(Clone)]
pub struct Session {
    _private: (),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_type_default() {
        assert_eq!(DeviceType::default(), DeviceType::Cpu);
    }

    #[test]
    fn test_onnx_runtime_new() {
        let runtime = OnnxRuntime::new();
        assert!(runtime.is_ok());
        assert_eq!(
            runtime.expect("value should be valid").device(),
            DeviceType::Cpu
        );
    }

    #[test]
    fn test_onnx_runtime_with_device() {
        let runtime = OnnxRuntime::with_device(DeviceType::Cuda);
        assert!(runtime.is_ok());
        assert_eq!(
            runtime.expect("value should be valid").device(),
            DeviceType::Cuda
        );
    }

    #[test]
    fn test_onnx_runtime_default() {
        let runtime = OnnxRuntime::default();
        assert_eq!(runtime.device(), DeviceType::Cpu);
    }
}
