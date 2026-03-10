/// GPU acceleration hints for denoise operations.
use super::DenoiseMethod;

/// GPU acceleration availability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuSupport {
    /// No GPU support.
    None,
    /// CUDA support.
    Cuda,
    /// OpenCL support.
    OpenCl,
    /// Vulkan support.
    Vulkan,
    /// Metal support (macOS/iOS).
    Metal,
}

/// GPU compute kernel hint.
#[derive(Debug, Clone)]
pub struct GpuKernelHint {
    /// Method being accelerated.
    pub method: DenoiseMethod,
    /// Recommended workgroup size.
    pub workgroup_size: (u32, u32),
    /// Memory requirements (bytes).
    pub memory_requirements: usize,
    /// Expected speedup factor.
    pub speedup_factor: f32,
}

impl GpuKernelHint {
    /// Get GPU kernel hint for a method.
    #[must_use]
    pub fn for_method(method: DenoiseMethod) -> Self {
        match method {
            DenoiseMethod::Bilateral => Self {
                method,
                workgroup_size: (16, 16),
                memory_requirements: 1024 * 1024,
                speedup_factor: 8.0,
            },
            DenoiseMethod::NonLocalMeans => Self {
                method,
                workgroup_size: (8, 8),
                memory_requirements: 4 * 1024 * 1024,
                speedup_factor: 15.0,
            },
            DenoiseMethod::Gaussian => Self {
                method,
                workgroup_size: (32, 8),
                memory_requirements: 512 * 1024,
                speedup_factor: 5.0,
            },
            DenoiseMethod::Median => Self {
                method,
                workgroup_size: (16, 16),
                memory_requirements: 2 * 1024 * 1024,
                speedup_factor: 6.0,
            },
            DenoiseMethod::Adaptive => Self {
                method,
                workgroup_size: (16, 16),
                memory_requirements: 2 * 1024 * 1024,
                speedup_factor: 7.0,
            },
            DenoiseMethod::MotionCompensated => Self {
                method,
                workgroup_size: (8, 8),
                memory_requirements: 8 * 1024 * 1024,
                speedup_factor: 20.0,
            },
            DenoiseMethod::BlockMatching3D => Self {
                method,
                workgroup_size: (8, 8),
                memory_requirements: 16 * 1024 * 1024,
                speedup_factor: 25.0,
            },
            _ => Self {
                method,
                workgroup_size: (16, 16),
                memory_requirements: 1024 * 1024,
                speedup_factor: 3.0,
            },
        }
    }

    /// Check if GPU is worth using for given resolution.
    #[must_use]
    pub fn is_worthwhile(&self, width: u32, height: u32) -> bool {
        let pixels = width * height;
        pixels > 1920 * 1080 / 4
    }
}

/// Detect available GPU support.
#[must_use]
#[allow(unexpected_cfgs)]
pub fn detect_gpu_support() -> Vec<GpuSupport> {
    let mut support = Vec::new();

    if cfg!(feature = "cuda") {
        support.push(GpuSupport::Cuda);
    }

    if cfg!(feature = "opencl") {
        support.push(GpuSupport::OpenCl);
    }

    if cfg!(feature = "vulkan") {
        support.push(GpuSupport::Vulkan);
    }

    if cfg!(target_os = "macos") || cfg!(target_os = "ios") {
        support.push(GpuSupport::Metal);
    }

    if support.is_empty() {
        support.push(GpuSupport::None);
    }

    support
}
