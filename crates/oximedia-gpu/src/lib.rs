//! GPU compute pipeline using WGPU for `OxiMedia`
//!
//! This crate provides a cross-platform GPU acceleration layer using WGPU,
//! supporting Vulkan, Metal, DirectX 12, and WebGPU backends.
//!
//! # Features
//!
//! - Color space conversions (RGB ↔ YUV with BT.601, BT.709, BT.2020)
//! - Image scaling (nearest, bilinear, bicubic)
//! - Convolution filters (blur, sharpen, edge detection)
//! - Transform operations (DCT, FFT)
//! - Automatic CPU fallback
//! - Multi-GPU support
//!
//! # Example
//!
//! ```no_run
//! use oximedia_gpu::GpuContext;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let ctx = GpuContext::new()?;
//!
//! let input = vec![0u8; 1920 * 1080 * 4];
//! let mut output = vec![0u8; 1920 * 1080 * 4];
//!
//! ctx.rgb_to_yuv(&input, &mut output)?;
//! # Ok(())
//! # }
//! ```

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

// Core modules
pub mod buffer;
pub mod device;
pub mod ops;
pub mod shader;

// New comprehensive modules
pub mod accelerator;
pub mod backend;
pub mod cache;
pub mod compiler;
pub mod compute;
pub mod kernels;
pub mod memory;
pub mod queue;
pub mod sync;

// GPU compute operation modules
pub mod histogram;
pub mod motion_detect;
pub mod pipeline;
pub mod texture;
pub mod video_process;

// New kernel / pass / shader-param modules
pub mod compute_pass;
pub mod kernel;
pub mod shader_params;

// Wave-8 new modules
pub mod compute_dispatch;
pub mod memory_pool;
pub mod shader_cache;

// Wave-9 new modules
pub mod gpu_buffer;
pub mod gpu_fence;
pub mod render_pass;

// Wave-10 new modules
pub mod command_buffer;
pub mod resource_manager;
pub mod sync_primitive;

// Wave-11 new modules
pub mod descriptor_set;
pub mod gpu_stats;
pub mod viewport;

// Wave-12 new modules
pub mod gpu_profiler;
pub mod sampler;
pub mod vertex_buffer;

// Wave-13 new modules
pub mod fence_pool;
pub mod gpu_timer;
pub mod upload_queue;

// Wave-14 new modules
pub mod buffer_copy;
pub mod occupancy;
pub mod workgroup;

// Wave-15 new modules
pub mod buffer_pool;
pub mod compute_kernels;
pub mod pipeline_stages;

// Wave-16 new modules (0.1.2 enhancements)
pub mod motion_estimation;
pub mod multi_gpu;

// Wave-17 new modules
pub mod compute_shader;
pub mod histogram_equalization;

// Previously undeclared modules (discovered in src/ inventory)
pub mod async_compute;
pub mod barrier_manager;
pub mod blend_kernel;
pub mod color_convert_kernel;
pub mod compute_graph;
pub mod double_buffer;
pub mod film_grain;
pub mod gpu_cpu_verify;
pub mod indirect_dispatch;
pub mod kernel_scheduler;
pub mod mipmap_gen;
pub mod optical_flow;
pub mod perspective_transform;
pub mod pipeline_cache;
pub mod readback;
pub mod scale_kernel;
pub mod texture_atlas;
pub mod texture_cache;
pub mod tone_curve;

use std::sync::Arc;
use thiserror::Error;

// Accelerator exports
pub use accelerator::{AcceleratorBuilder, CpuAccelerator, GpuAccelerator, WgpuAccelerator};

// Core exports
pub use buffer::{BufferType, GpuBuffer};
pub use device::{GpuDevice, GpuDeviceInfo};
pub use ops::quality_metrics::{
    compute_ms_ssim, compute_psnr, compute_ssim, MsSsimResult, PsnrResult, SsimResult,
};
pub use ops::{
    ChromaOps, ChromaSubsampling, ColorSpaceConversion, FilterOperation, ScaleOperation,
    TransformOperation, YcbcrCoefficients,
};

// Backend exports
pub use backend::{Backend, BackendCapabilities, BackendType, CpuBackend, VulkanBackend};

// Cache exports
pub use cache::{CacheStats, PipelineCache, ShaderCache};

// Compiler exports
pub use compiler::{
    CompilationError, CompilationOptions, OptimizationLevel, ShaderCompiler, ShaderPreprocessor,
};

// Compute exports
pub use compute::{
    ComputeExecutor, ComputePassBuilder, ComputePipelineHandle, ComputePipelineManager,
    DispatchHelper,
};

// Kernels exports
pub use kernels::{
    ColorConversionKernel, ConvolutionKernel, FilterKernel, ReduceKernel, ReduceOp, ResizeFilter,
    ResizeKernel, TransformKernel, TransformType,
};

// Memory exports
pub use memory::{ManagedBuffer, MemoryAllocator, MemoryPool, MemoryStats};

// Queue exports
pub use queue::{
    AsyncSubmission, BatchSubmitter, CommandBufferBuilder, CommandQueue, QueueManager, QueueType,
};

// Sync exports
pub use sync::{Barrier, Event, Fence, Semaphore};

// Workgroup auto-tuner exports
pub use workgroup::{DeviceLimits, WorkgroupAutoTuner};

// Memory pool defragmentation exports
pub use memory_pool::{CompactionPlan, DefragResult, MigrationEntry};

// Video processing exports
pub use buffer_pool::SubAllocator;
pub use compute_pass::{BatchedComputePass, DispatchCommand};
pub use histogram::{ChannelHistogram, ImageHistogram};
pub use motion_detect::{MotionAnalysis, MotionDetector, MotionRegion, Sensitivity};
pub use pipeline::{
    BarrierBatcher, BarrierKind, BarrierStrategy, BufferBarrier, FlushRecord, GpuPipeline,
    PipelineMetrics, PipelineNode, PipelineStage,
};
pub use texture::{TextureDescriptor, TextureFormat, TexturePool};
pub use video_process::{FrameProcessConfig, FrameProcessResult, VideoFrameProcessor};

// Wave-17 exports
pub use compute_shader::{ComputeShaderSimulator, ShaderKernel, ThreadGroupContext};
pub use histogram_equalization::{ClaheConfig, EqualizationStats, HistogramEqualizer};

/// Error types for GPU operations
#[derive(Debug, Error)]
pub enum GpuError {
    /// Device initialization failed
    #[error("Failed to initialize GPU device: {0}")]
    DeviceInit(String),

    /// Adapter selection failed
    #[error("No suitable GPU adapter found")]
    NoAdapter,

    /// Device request failed
    #[error("Failed to request GPU device: {0}")]
    DeviceRequest(String),

    /// Buffer creation failed
    #[error("Failed to create GPU buffer: {0}")]
    BufferCreation(String),

    /// Shader compilation failed
    #[error("Failed to compile shader: {0}")]
    ShaderCompilation(String),

    /// Pipeline creation failed
    #[error("Failed to create compute pipeline: {0}")]
    PipelineCreation(String),

    /// Command submission failed
    #[error("Failed to submit GPU commands: {0}")]
    CommandSubmission(String),

    /// Buffer mapping failed
    #[error("Failed to map GPU buffer: {0}")]
    BufferMapping(String),

    /// Invalid dimensions
    #[error("Invalid image dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },

    /// Invalid buffer size
    #[error("Invalid buffer size: expected {expected}, got {actual}")]
    InvalidBufferSize { expected: usize, actual: usize },

    /// Operation not supported
    #[error("Operation not supported: {0}")]
    NotSupported(String),

    /// Internal error
    #[error("Internal GPU error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, GpuError>;

/// GPU context for compute operations
///
/// This is the main entry point for GPU-accelerated operations.
/// It manages device selection, resource allocation, and command submission.
pub struct GpuContext {
    device: Arc<GpuDevice>,
}

impl GpuContext {
    /// Create a new GPU context with automatic device selection
    ///
    /// This will select the most suitable GPU device available on the system.
    /// If no GPU is available, an error is returned.
    ///
    /// # Errors
    ///
    /// Returns an error if no suitable GPU device is found or if device
    /// initialization fails.
    pub fn new() -> Result<Self> {
        let device = GpuDevice::new(None)?;
        Ok(Self {
            device: Arc::new(device),
        })
    }

    /// Create a new GPU context with a specific device
    ///
    /// # Arguments
    ///
    /// * `device_index` - Index of the device to use (from `list_devices`)
    ///
    /// # Errors
    ///
    /// Returns an error if the device index is invalid or if device
    /// initialization fails.
    pub fn with_device(device_index: usize) -> Result<Self> {
        let device = GpuDevice::new(Some(device_index))?;
        Ok(Self {
            device: Arc::new(device),
        })
    }

    /// List available GPU devices
    ///
    /// Returns information about all GPU devices available on the system.
    pub fn list_devices() -> Result<Vec<GpuDeviceInfo>> {
        GpuDevice::list_devices()
    }

    /// Get information about the current device
    #[must_use]
    pub fn device_info(&self) -> &GpuDeviceInfo {
        self.device.info()
    }

    /// Convert RGB to YUV (BT.601)
    ///
    /// # Arguments
    ///
    /// * `input` - Input RGB buffer (packed RGBA format)
    /// * `output` - Output YUV buffer (packed YUVA format)
    ///
    /// # Errors
    ///
    /// Returns an error if buffer sizes are invalid or if the GPU operation fails.
    pub fn rgb_to_yuv(&self, input: &[u8], output: &mut [u8]) -> Result<()> {
        if input.len() != output.len() {
            return Err(GpuError::InvalidBufferSize {
                expected: input.len(),
                actual: output.len(),
            });
        }

        if input.len() % 4 != 0 {
            return Err(GpuError::InvalidBufferSize {
                expected: (input.len() / 4) * 4,
                actual: input.len(),
            });
        }

        let width = ((input.len() / 4) as f32).sqrt() as u32;
        let height = width;

        ops::ColorSpaceConversion::rgb_to_yuv(
            &self.device,
            input,
            output,
            width,
            height,
            ops::ColorSpace::BT601,
        )
    }

    /// Convert YUV to RGB (BT.601)
    ///
    /// # Arguments
    ///
    /// * `input` - Input YUV buffer (packed YUVA format)
    /// * `output` - Output RGB buffer (packed RGBA format)
    ///
    /// # Errors
    ///
    /// Returns an error if buffer sizes are invalid or if the GPU operation fails.
    pub fn yuv_to_rgb(&self, input: &[u8], output: &mut [u8]) -> Result<()> {
        if input.len() != output.len() {
            return Err(GpuError::InvalidBufferSize {
                expected: input.len(),
                actual: output.len(),
            });
        }

        if input.len() % 4 != 0 {
            return Err(GpuError::InvalidBufferSize {
                expected: (input.len() / 4) * 4,
                actual: input.len(),
            });
        }

        let width = ((input.len() / 4) as f32).sqrt() as u32;
        let height = width;

        ops::ColorSpaceConversion::yuv_to_rgb(
            &self.device,
            input,
            output,
            width,
            height,
            ops::ColorSpace::BT601,
        )
    }

    /// Scale an image using bilinear interpolation
    ///
    /// # Arguments
    ///
    /// * `input` - Input image buffer (packed RGBA format)
    /// * `src_width` - Source image width
    /// * `src_height` - Source image height
    /// * `output` - Output image buffer (packed RGBA format)
    /// * `dst_width` - Destination image width
    /// * `dst_height` - Destination image height
    ///
    /// # Errors
    ///
    /// Returns an error if buffer sizes are invalid or if the GPU operation fails.
    pub fn scale_bilinear(
        &self,
        input: &[u8],
        src_width: u32,
        src_height: u32,
        output: &mut [u8],
        dst_width: u32,
        dst_height: u32,
    ) -> Result<()> {
        ops::ScaleOperation::scale(
            &self.device,
            input,
            src_width,
            src_height,
            output,
            dst_width,
            dst_height,
            ops::ScaleFilter::Bilinear,
        )
    }

    /// Scale an image using bicubic interpolation
    ///
    /// # Arguments
    ///
    /// * `input` - Input image buffer (packed RGBA format)
    /// * `src_width` - Source image width
    /// * `src_height` - Source image height
    /// * `output` - Output image buffer (packed RGBA format)
    /// * `dst_width` - Destination image width
    /// * `dst_height` - Destination image height
    ///
    /// # Errors
    ///
    /// Returns an error if buffer sizes are invalid or if the GPU operation fails.
    pub fn scale_bicubic(
        &self,
        input: &[u8],
        src_width: u32,
        src_height: u32,
        output: &mut [u8],
        dst_width: u32,
        dst_height: u32,
    ) -> Result<()> {
        ops::ScaleOperation::scale(
            &self.device,
            input,
            src_width,
            src_height,
            output,
            dst_width,
            dst_height,
            ops::ScaleFilter::Bicubic,
        )
    }

    /// Scale an image using Lanczos-3 interpolation (highest quality)
    ///
    /// # Arguments
    ///
    /// * `input` - Input image buffer (packed RGBA format)
    /// * `src_width` - Source image width
    /// * `src_height` - Source image height
    /// * `output` - Output image buffer (packed RGBA format)
    /// * `dst_width` - Destination image width
    /// * `dst_height` - Destination image height
    ///
    /// # Errors
    ///
    /// Returns an error if buffer sizes are invalid or if the GPU operation fails.
    pub fn scale_lanczos(
        &self,
        input: &[u8],
        src_width: u32,
        src_height: u32,
        output: &mut [u8],
        dst_width: u32,
        dst_height: u32,
    ) -> Result<()> {
        ops::ScaleOperation::scale(
            &self.device,
            input,
            src_width,
            src_height,
            output,
            dst_width,
            dst_height,
            ops::ScaleFilter::Lanczos3,
        )
    }

    /// Apply Gaussian blur
    ///
    /// # Arguments
    ///
    /// * `input` - Input image buffer (packed RGBA format)
    /// * `output` - Output image buffer (packed RGBA format)
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `sigma` - Blur radius (standard deviation)
    ///
    /// # Errors
    ///
    /// Returns an error if buffer sizes are invalid or if the GPU operation fails.
    #[allow(clippy::too_many_arguments)]
    pub fn gaussian_blur(
        &self,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
        sigma: f32,
    ) -> Result<()> {
        ops::FilterOperation::gaussian_blur(&self.device, input, output, width, height, sigma)
    }

    /// Apply sharpening filter
    ///
    /// # Arguments
    ///
    /// * `input` - Input image buffer (packed RGBA format)
    /// * `output` - Output image buffer (packed RGBA format)
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `amount` - Sharpening strength
    ///
    /// # Errors
    ///
    /// Returns an error if buffer sizes are invalid or if the GPU operation fails.
    #[allow(clippy::too_many_arguments)]
    pub fn sharpen(
        &self,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
        amount: f32,
    ) -> Result<()> {
        ops::FilterOperation::sharpen(&self.device, input, output, width, height, amount)
    }

    /// Detect edges using Sobel operator
    ///
    /// # Arguments
    ///
    /// * `input` - Input image buffer (packed RGBA format)
    /// * `output` - Output image buffer (packed RGBA format)
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Errors
    ///
    /// Returns an error if buffer sizes are invalid or if the GPU operation fails.
    pub fn edge_detect(
        &self,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
    ) -> Result<()> {
        ops::FilterOperation::edge_detect(&self.device, input, output, width, height)
    }

    /// Compute 2D DCT (Discrete Cosine Transform)
    ///
    /// # Arguments
    ///
    /// * `input` - Input data (f32 values)
    /// * `output` - Output DCT coefficients
    /// * `width` - Data width (must be multiple of 8)
    /// * `height` - Data height (must be multiple of 8)
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid or if the GPU operation fails.
    pub fn dct_2d(&self, input: &[f32], output: &mut [f32], width: u32, height: u32) -> Result<()> {
        ops::TransformOperation::dct_2d(&self.device, input, output, width, height)
    }

    /// Compute 2D IDCT (Inverse Discrete Cosine Transform)
    ///
    /// # Arguments
    ///
    /// * `input` - Input DCT coefficients
    /// * `output` - Output reconstructed data
    /// * `width` - Data width (must be multiple of 8)
    /// * `height` - Data height (must be multiple of 8)
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid or if the GPU operation fails.
    pub fn idct_2d(
        &self,
        input: &[f32],
        output: &mut [f32],
        width: u32,
        height: u32,
    ) -> Result<()> {
        ops::TransformOperation::idct_2d(&self.device, input, output, width, height)
    }

    /// Wait for all GPU operations to complete
    ///
    /// This is useful for synchronization and benchmarking.
    pub fn wait(&self) {
        self.device.wait();
    }
}

// GpuContext intentionally does not implement Default.
//
// GPU context creation is inherently fallible (no adapter, driver error, etc.).
// Callers must use GpuContext::new() or GpuContext::with_device() and handle
// the returned Result explicitly.  A silent Default impl that can either panic
// or silently return a non-functional context would be misleading.
//
// If a best-effort fallback context is needed, use:
//   GpuContext::new().or_else(|_| GpuContext::with_device(0))
