//! CPU-fallback GPU backend trait for VFX effects.
//!
//! Effects that declare `supports_gpu() = true` in their [`VideoEffect`]
//! implementation can optionally implement [`GpuEffect`] to gain access to
//! the GPU dispatch path.  When no GPU device is available the system
//! automatically falls back to the CPU path.
//!
//! This module defines the backend trait and a [`CpuGpuDispatcher`] that
//! selects between the two paths at runtime.
//!
//! # Design notes
//!
//! Because `oximedia-vfx` is a pure-Rust, no-unsafe crate with no GPU
//! runtime dependency by default, the GPU path here is a *trait-object*
//! abstraction.  A real GPU implementation (wgpu/vulkan/metal) would live in
//! a separate optional crate and inject an implementation via
//! [`CpuGpuDispatcher::with_gpu`].

use crate::{EffectParams, Frame, VfxResult, VideoEffect};

// ─────────────────────────────────────────────────────────────────────────────
// GpuEffect trait
// ─────────────────────────────────────────────────────────────────────────────

/// Extension trait for effects that provide a GPU compute-shader implementation.
///
/// Implementors receive the same `input`/`output`/`params` as [`VideoEffect::apply`]
/// but are allowed to keep GPU-side buffer state across calls.
pub trait GpuEffect: Send + Sync {
    /// Name of this GPU effect variant (for diagnostics).
    fn name(&self) -> &str;

    /// Upload `input` to the GPU, run the compute pass, and download results
    /// into `output`.
    ///
    /// # Errors
    ///
    /// Returns a [`VfxError`](crate::VfxError) if the GPU pass fails.
    fn apply_gpu(
        &mut self,
        input: &Frame,
        output: &mut Frame,
        params: &EffectParams,
    ) -> VfxResult<()>;

    /// Returns `true` if the GPU backend is ready (device initialised, buffers
    /// allocated).
    fn is_ready(&self) -> bool;
}

// ─────────────────────────────────────────────────────────────────────────────
// GpuAvailability
// ─────────────────────────────────────────────────────────────────────────────

/// Runtime GPU availability status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuAvailability {
    /// A GPU backend is available and ready.
    Available,
    /// No GPU backend was injected; will use CPU path.
    NotAvailable,
    /// A GPU backend was present but initialisation failed.
    InitFailed,
}

// ─────────────────────────────────────────────────────────────────────────────
// CpuGpuDispatcher
// ─────────────────────────────────────────────────────────────────────────────

/// Dispatches between a GPU and a CPU [`VideoEffect`] implementation.
///
/// If a [`GpuEffect`] is installed and [`GpuEffect::is_ready`] returns `true`,
/// [`CpuGpuDispatcher::apply`] forwards to the GPU path; otherwise it falls
/// through to the CPU effect.
///
/// # Example
///
/// ```ignore
/// use oximedia_vfx::gpu_backend::CpuGpuDispatcher;
/// use oximedia_vfx::distortion::barrel::BarrelDistortion;
///
/// let cpu_effect = BarrelDistortion::default();
/// let dispatcher = CpuGpuDispatcher::cpu_only(cpu_effect);
/// assert!(!dispatcher.has_gpu());
/// ```
pub struct CpuGpuDispatcher {
    cpu: Box<dyn VideoEffect>,
    gpu: Option<Box<dyn GpuEffect>>,
}

impl CpuGpuDispatcher {
    /// Create a dispatcher that only uses the CPU path.
    #[must_use]
    pub fn cpu_only(cpu: impl VideoEffect + 'static) -> Self {
        Self {
            cpu: Box::new(cpu),
            gpu: None,
        }
    }

    /// Create a dispatcher with both a CPU fallback and a GPU path.
    #[must_use]
    pub fn with_gpu(cpu: impl VideoEffect + 'static, gpu: impl GpuEffect + 'static) -> Self {
        Self {
            cpu: Box::new(cpu),
            gpu: Some(Box::new(gpu)),
        }
    }

    /// Returns `true` if a GPU backend has been installed.
    #[must_use]
    pub fn has_gpu(&self) -> bool {
        self.gpu.is_some()
    }

    /// Returns the current GPU availability status.
    #[must_use]
    pub fn gpu_availability(&self) -> GpuAvailability {
        match &self.gpu {
            None => GpuAvailability::NotAvailable,
            Some(g) if g.is_ready() => GpuAvailability::Available,
            Some(_) => GpuAvailability::InitFailed,
        }
    }

    /// Apply the effect, preferring the GPU path when available and ready.
    ///
    /// Falls back to the CPU path automatically.
    ///
    /// # Errors
    ///
    /// Returns the CPU path's error if the GPU path is unavailable and the
    /// CPU path fails.  GPU errors are silently demoted to CPU fallback.
    pub fn apply(
        &mut self,
        input: &Frame,
        output: &mut Frame,
        params: &EffectParams,
    ) -> VfxResult<()> {
        if let Some(ref mut g) = self.gpu {
            if g.is_ready() {
                // Try GPU; on failure fall through to CPU
                if g.apply_gpu(input, output, params).is_ok() {
                    return Ok(());
                }
            }
        }
        // CPU fallback
        self.cpu.apply(input, output, params)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Frame, QualityMode};

    /// Minimal CPU effect for testing: copies input to output unchanged.
    struct PassThroughCpu;
    impl VideoEffect for PassThroughCpu {
        fn name(&self) -> &str {
            "PassThrough"
        }
        fn apply(
            &mut self,
            input: &Frame,
            output: &mut Frame,
            _params: &EffectParams,
        ) -> VfxResult<()> {
            output.data.copy_from_slice(&input.data);
            Ok(())
        }
    }

    /// GPU effect that always claims to be ready and simply inverts colours.
    struct InvertGpu;
    impl GpuEffect for InvertGpu {
        fn name(&self) -> &str {
            "InvertGpu"
        }
        fn apply_gpu(
            &mut self,
            input: &Frame,
            output: &mut Frame,
            _params: &EffectParams,
        ) -> VfxResult<()> {
            for (i, (&s, d)) in input.data.iter().zip(output.data.iter_mut()).enumerate() {
                *d = if i % 4 == 3 { s } else { 255 - s };
            }
            Ok(())
        }
        fn is_ready(&self) -> bool {
            true
        }
    }

    /// GPU effect that always reports not-ready (simulates failed init).
    struct NotReadyGpu;
    impl GpuEffect for NotReadyGpu {
        fn name(&self) -> &str {
            "NotReady"
        }
        fn apply_gpu(
            &mut self,
            _input: &Frame,
            _output: &mut Frame,
            _params: &EffectParams,
        ) -> VfxResult<()> {
            Err(crate::VfxError::GpuError("not ready".into()))
        }
        fn is_ready(&self) -> bool {
            false
        }
    }

    fn params() -> EffectParams {
        EffectParams {
            progress: 0.0,
            quality: QualityMode::Preview,
            time: 0.0,
            use_gpu: true,
            motion_blur: 0.0,
        }
    }

    #[test]
    fn test_cpu_only_dispatcher_no_gpu() {
        let d = CpuGpuDispatcher::cpu_only(PassThroughCpu);
        assert!(!d.has_gpu());
        assert_eq!(d.gpu_availability(), GpuAvailability::NotAvailable);
    }

    #[test]
    fn test_with_gpu_has_gpu() {
        let d = CpuGpuDispatcher::with_gpu(PassThroughCpu, InvertGpu);
        assert!(d.has_gpu());
        assert_eq!(d.gpu_availability(), GpuAvailability::Available);
    }

    #[test]
    fn test_not_ready_gpu_reports_init_failed() {
        let d = CpuGpuDispatcher::with_gpu(PassThroughCpu, NotReadyGpu);
        assert_eq!(d.gpu_availability(), GpuAvailability::InitFailed);
    }

    #[test]
    fn test_cpu_fallback_when_no_gpu() {
        let mut d = CpuGpuDispatcher::cpu_only(PassThroughCpu);
        let input = {
            let mut f = Frame::new(4, 4).expect("frame");
            f.clear([100, 150, 200, 255]);
            f
        };
        let mut output = Frame::new(4, 4).expect("output");
        d.apply(&input, &mut output, &params()).expect("apply");
        assert_eq!(output.get_pixel(0, 0), Some([100, 150, 200, 255]));
    }

    #[test]
    fn test_gpu_path_used_when_ready() {
        let mut d = CpuGpuDispatcher::with_gpu(PassThroughCpu, InvertGpu);
        let input = {
            let mut f = Frame::new(4, 4).expect("frame");
            f.clear([100, 150, 200, 255]);
            f
        };
        let mut output = Frame::new(4, 4).expect("output");
        d.apply(&input, &mut output, &params()).expect("apply");
        // InvertGpu should have inverted R/G/B channels
        let px = output.get_pixel(0, 0).expect("px");
        assert_eq!(px[0], 155); // 255 - 100
        assert_eq!(px[1], 105); // 255 - 150
        assert_eq!(px[2], 55);  // 255 - 200
        assert_eq!(px[3], 255); // alpha preserved
    }

    #[test]
    fn test_not_ready_gpu_falls_back_to_cpu() {
        let mut d = CpuGpuDispatcher::with_gpu(PassThroughCpu, NotReadyGpu);
        let input = {
            let mut f = Frame::new(4, 4).expect("frame");
            f.clear([42, 43, 44, 255]);
            f
        };
        let mut output = Frame::new(4, 4).expect("output");
        d.apply(&input, &mut output, &params()).expect("apply");
        // CPU PassThrough: output == input
        assert_eq!(output.get_pixel(0, 0), Some([42, 43, 44, 255]));
    }
}
