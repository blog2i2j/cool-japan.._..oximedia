//! ONNX Runtime wrapper for model inference.
//!
//! This module provides a safe wrapper around ONNX Runtime, managing
//! sessions, device selection, and model loading.

use crate::error::{CvError, CvResult};
use crate::ml::tensor::Tensor;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::Session as OrtSession;
use std::path::Path;
use std::sync::Arc;

// Execution providers are optional features
#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;
#[cfg(target_os = "macos")]
use ort::execution_providers::CoreMLExecutionProvider;
#[cfg(target_os = "windows")]
use ort::execution_providers::DirectMLExecutionProvider;
#[cfg(feature = "rocm")]
use ort::execution_providers::ROCmExecutionProvider;
#[cfg(feature = "tensorrt")]
use ort::execution_providers::TensorRTExecutionProvider;

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

    /// Build an ONNX Runtime session from file.
    fn build_session(&self, path: &Path) -> CvResult<OrtSession> {
        let mut builder = OrtSession::builder()
            .map_err(|e| CvError::onnx_runtime(format!("Failed to create session builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| CvError::onnx_runtime(format!("Failed to set optimization level: {e}")))?;

        // Configure execution provider based on device
        builder = self.configure_execution_provider(builder)?;

        // Load model
        builder
            .commit_from_file(path)
            .map_err(|e| CvError::model_load(format!("Failed to load model from file: {e}")))
    }

    /// Build an ONNX Runtime session from bytes.
    fn build_session_from_bytes(&self, bytes: &[u8]) -> CvResult<OrtSession> {
        let mut builder = OrtSession::builder()
            .map_err(|e| CvError::onnx_runtime(format!("Failed to create session builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| CvError::onnx_runtime(format!("Failed to set optimization level: {e}")))?;

        // Configure execution provider based on device
        builder = self.configure_execution_provider(builder)?;

        // Load model from bytes
        builder
            .commit_from_memory(bytes)
            .map_err(|e| CvError::model_load(format!("Failed to load model from bytes: {e}")))
    }

    /// Configure execution provider for the session builder.
    #[allow(unused_variables)]
    fn configure_execution_provider(&self, builder: SessionBuilder) -> CvResult<SessionBuilder> {
        let builder = match self.device {
            DeviceType::Cpu => builder,
            #[cfg(feature = "cuda")]
            DeviceType::Cuda => builder
                .with_execution_providers([CUDAExecutionProvider::default().build()])
                .map_err(|e| {
                    CvError::onnx_runtime(format!("Failed to configure CUDA provider: {e}"))
                })?,
            #[cfg(not(feature = "cuda"))]
            DeviceType::Cuda => {
                return Err(CvError::onnx_runtime(
                    "CUDA support not compiled in".to_owned(),
                ))
            }
            #[cfg(feature = "rocm")]
            DeviceType::Rocm => builder
                .with_execution_providers([ROCmExecutionProvider::default().build()])
                .map_err(|e| {
                    CvError::onnx_runtime(format!("Failed to configure ROCm provider: {e}"))
                })?,
            #[cfg(not(feature = "rocm"))]
            DeviceType::Rocm => {
                return Err(CvError::onnx_runtime(
                    "ROCm support not compiled in".to_owned(),
                ))
            }
            #[cfg(feature = "tensorrt")]
            DeviceType::TensorRt => builder
                .with_execution_providers([TensorRTExecutionProvider::default().build()])
                .map_err(|e| {
                    CvError::onnx_runtime(format!("Failed to configure TensorRT provider: {e}"))
                })?,
            #[cfg(not(feature = "tensorrt"))]
            DeviceType::TensorRt => {
                return Err(CvError::onnx_runtime(
                    "TensorRT support not compiled in".to_owned(),
                ))
            }
            #[cfg(target_os = "windows")]
            DeviceType::DirectMl => builder
                .with_execution_providers([DirectMLExecutionProvider::default().build()])
                .map_err(|e| {
                    CvError::onnx_runtime(format!("Failed to configure DirectML provider: {e}"))
                })?,
            #[cfg(not(target_os = "windows"))]
            DeviceType::DirectMl => {
                return Err(CvError::onnx_runtime(
                    "DirectML is only available on Windows".to_owned(),
                ))
            }
            #[cfg(target_os = "macos")]
            DeviceType::CoreMl => builder
                .with_execution_providers([CoreMLExecutionProvider::default().build()])
                .map_err(|e| {
                    CvError::onnx_runtime(format!("Failed to configure CoreML provider: {e}"))
                })?,
            #[cfg(not(target_os = "macos"))]
            DeviceType::CoreMl => {
                return Err(CvError::onnx_runtime(
                    "CoreML is only available on macOS".to_owned(),
                ))
            }
        };

        Ok(builder)
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
#[derive(Clone)]
pub struct Session {
    inner: Arc<std::sync::Mutex<OrtSession>>,
}

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
        // Convert our tensors to ort values
        let ort_inputs: Vec<ort::value::DynValue> = inputs
            .iter()
            .map(super::tensor::Tensor::to_ort_value)
            .collect::<CvResult<Vec<_>>>()?;

        // Lock the session for mutable access
        let mut session = self
            .inner
            .lock()
            .map_err(|e| CvError::onnx_runtime(format!("Session lock error: {e}")))?;

        // Get input names from the session to build named inputs
        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();

        if ort_inputs.len() > input_names.len() {
            return Err(CvError::onnx_runtime(format!(
                "Too many inputs: got {}, model expects {}",
                ort_inputs.len(),
                input_names.len()
            )));
        }

        // Build named input pairs and run inference
        let named_inputs: Vec<(String, ort::value::DynValue)> =
            input_names.into_iter().zip(ort_inputs).collect();

        let outputs = session
            .run(named_inputs)
            .map_err(|e| CvError::onnx_runtime(format!("Inference failed: {e}")))?;

        // Convert outputs back to our tensors
        outputs
            .values()
            .map(|value| Tensor::from_ort_value(&value))
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
        self.inner.lock().map_or_else(
            |_| Vec::new(),
            |s| s.inputs().iter().map(|i| i.name().to_string()).collect(),
        )
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
        self.inner.lock().map_or_else(
            |_| Vec::new(),
            |s| s.outputs().iter().map(|o| o.name().to_string()).collect(),
        )
    }

    /// Get the number of inputs.
    #[must_use]
    pub fn input_count(&self) -> usize {
        self.inner.lock().map_or(0, |s| s.inputs().len())
    }

    /// Get the number of outputs.
    #[must_use]
    pub fn output_count(&self) -> usize {
        self.inner.lock().map_or(0, |s| s.outputs().len())
    }
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
