//! Hardware acceleration abstraction layer for `OxiMedia`.
//!
//! `oximedia-accel` provides GPU-accelerated computation for video processing
//! operations using Vulkan compute shaders. It includes CPU fallback paths
//! for systems without GPU support.
//!
//! # Features
//!
//! - **Device Management**: Automatic GPU device enumeration and selection
//! - **Buffer Management**: Efficient GPU memory allocation and transfer
//! - **Compute Kernels**: Image scaling, color conversion, motion estimation
//! - **CPU Fallback**: Automatic fallback to CPU implementations
//! - **Safe Vulkan**: Uses vulkano for safe Vulkan API access
//!
//! # Architecture
//!
//! The acceleration layer is designed around the [`HardwareAccel`] trait,
//! which provides a unified interface for both GPU and CPU implementations.
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │         HardwareAccel Trait             │
//! └─────────────────────────────────────────┘
//!          │                      │
//!          ▼                      ▼
//!   ┌────────────┐         ┌────────────┐
//!   │ VulkanAccel│         │ CpuFallback│
//!   └────────────┘         └────────────┘
//! ```
//!
//! # Example
//!
//! ```no_run
//! use oximedia_accel::{AccelContext, HardwareAccel, ScaleFilter};
//! use oximedia_core::types::PixelFormat;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create acceleration context (automatically selects GPU or CPU)
//! let accel = AccelContext::new()?;
//!
//! // Perform image scaling
//! let input = vec![0u8; 1920 * 1080 * 3];
//! let output = accel.scale_image(
//!     &input,
//!     1920, 1080,
//!     1280, 720,
//!     PixelFormat::Rgb24,
//!     ScaleFilter::Bilinear,
//! )?;
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::too_many_arguments)]

pub mod accel_profile;
pub mod accel_stats;
pub mod buffer;
pub mod cache;
pub mod cpu_fallback;
pub mod cpu_simd;
pub mod device;
pub mod device_caps;
pub mod dispatch;
pub mod error;
pub mod fence_timeline;
pub mod kernels;
pub mod memory_arena;
pub mod memory_bandwidth;
pub mod ops;
pub mod pipeline_accel;
pub mod pool;
pub mod prefetch;
pub mod shaders;
mod stress_tests;
pub mod subgroup;
pub mod task_graph;
pub mod task_scheduler;
pub mod traits;
pub mod vulkan;
pub mod webgpu_backend;
pub mod workgroup;

// Re-export commonly used items
pub use error::{AccelError, AccelResult};
pub use traits::{HardwareAccel, ScaleFilter};

// Re-export new feature types
pub use accel_stats::{AccelProfiler, ProfileEntry};
pub use memory_arena::{MemoryPressureMonitor, MemoryPressurePolicy, PressureLevel};
pub use ops::convolution::{ConvolutionConfig, ConvolutionFilter, EdgeMode};
pub use ops::deinterlace::{DeinterlaceConfig, DeinterlaceMethod, FieldOrder};

// Re-export colour / HDR operations
pub use ops::color::{
    hlg_to_sdr_tonemap, pq_to_sdr_tonemap, rgb_to_yuv420, yuv420_to_rgb, YuvRange, YuvStandard,
};
pub use ops::{alpha_blend, alpha_blend_rgba};

// Re-export GPU compute operations
pub use ops::affine_gpu::{apply_affine, AffineTransform};
pub use ops::dct_gpu::{
    forward_dct_8x8, forward_dct_batch, inverse_dct_8x8, inverse_dct_batch, DctBlock,
};
pub use ops::histogram_gpu::{compute_histogram, GpuHistogram, HISTOGRAM_BINS};

// Re-export WebGPU backend
pub use webgpu_backend::{WebGpuAccelBackend, WebGpuAdapterInfo, WebGpuBackendState};

// Re-export workgroup auto-tuning
pub use workgroup::{compute_optimal_workgroup, OpType};

// Re-export descriptor pool
pub mod descriptor_pool;

use device::DeviceSelector;
use std::sync::Arc;
use vulkan::VulkanAccel;

/// Main acceleration context that automatically selects GPU or CPU backend.
///
/// This is the primary entry point for hardware-accelerated operations.
/// It will attempt to initialize Vulkan compute, falling back to CPU
/// if GPU acceleration is unavailable.
pub struct AccelContext {
    backend: AccelBackend,
}

enum AccelBackend {
    Vulkan(Arc<VulkanAccel>),
    Cpu(Arc<cpu_fallback::CpuAccel>),
}

impl AccelContext {
    /// Creates a new acceleration context.
    ///
    /// Attempts to initialize GPU acceleration first, falling back to CPU
    /// if Vulkan is unavailable or device selection fails.
    ///
    /// # Errors
    ///
    /// Returns an error only if both GPU and CPU initialization fail,
    /// which should be extremely rare.
    pub fn new() -> AccelResult<Self> {
        Self::with_device_selector(&DeviceSelector::default())
    }

    /// Creates a new acceleration context with a custom device selector.
    ///
    /// # Errors
    ///
    /// Returns an error if both GPU and CPU initialization fail.
    pub fn with_device_selector(selector: &DeviceSelector) -> AccelResult<Self> {
        match VulkanAccel::new(selector) {
            Ok(vulkan) => {
                tracing::info!("Hardware acceleration: Vulkan GPU");
                Ok(Self {
                    backend: AccelBackend::Vulkan(Arc::new(vulkan)),
                })
            }
            Err(e) => {
                tracing::warn!("Vulkan initialization failed: {}, using CPU fallback", e);
                Ok(Self {
                    backend: AccelBackend::Cpu(Arc::new(cpu_fallback::CpuAccel::new())),
                })
            }
        }
    }

    /// Forces CPU-only acceleration (no GPU).
    ///
    /// Useful for testing or when GPU acceleration is explicitly unwanted.
    #[must_use]
    pub fn cpu_only() -> Self {
        tracing::info!("Hardware acceleration: CPU only (forced)");
        Self {
            backend: AccelBackend::Cpu(Arc::new(cpu_fallback::CpuAccel::new())),
        }
    }

    /// Returns `true` if using GPU acceleration.
    #[must_use]
    pub fn is_gpu_accelerated(&self) -> bool {
        matches!(self.backend, AccelBackend::Vulkan(_))
    }

    /// Returns the name of the current backend.
    #[must_use]
    pub fn backend_name(&self) -> &str {
        match &self.backend {
            AccelBackend::Vulkan(v) => v.device_name(),
            AccelBackend::Cpu(_) => "CPU",
        }
    }
}

impl HardwareAccel for AccelContext {
    fn scale_image(
        &self,
        input: &[u8],
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
        format: oximedia_core::PixelFormat,
        filter: ScaleFilter,
    ) -> AccelResult<Vec<u8>> {
        match &self.backend {
            AccelBackend::Vulkan(v) => v.scale_image(
                input, src_width, src_height, dst_width, dst_height, format, filter,
            ),
            AccelBackend::Cpu(c) => c.scale_image(
                input, src_width, src_height, dst_width, dst_height, format, filter,
            ),
        }
    }

    fn convert_color(
        &self,
        input: &[u8],
        width: u32,
        height: u32,
        src_format: oximedia_core::PixelFormat,
        dst_format: oximedia_core::PixelFormat,
    ) -> AccelResult<Vec<u8>> {
        match &self.backend {
            AccelBackend::Vulkan(v) => {
                v.convert_color(input, width, height, src_format, dst_format)
            }
            AccelBackend::Cpu(c) => c.convert_color(input, width, height, src_format, dst_format),
        }
    }

    fn motion_estimation(
        &self,
        reference: &[u8],
        current: &[u8],
        width: u32,
        height: u32,
        block_size: u32,
    ) -> AccelResult<Vec<(i16, i16)>> {
        match &self.backend {
            AccelBackend::Vulkan(v) => {
                v.motion_estimation(reference, current, width, height, block_size)
            }
            AccelBackend::Cpu(c) => {
                c.motion_estimation(reference, current, width, height, block_size)
            }
        }
    }
}
