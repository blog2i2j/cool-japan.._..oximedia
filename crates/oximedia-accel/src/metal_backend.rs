//! Metal backend for macOS/iOS GPU acceleration.
//!
//! ## Feature gate
//!
//! All Metal-specific code is gated behind **both** `target_os = "macos"` (or
//! `target_os = "ios"`) **and** the `metal-backend` Cargo feature.  This
//! satisfies the COOLJAPAN Pure Rust Policy: the `metal` crate introduces Obj-C
//! FFI, so it must remain opt-in.
//!
//! When the feature is disabled (or the platform is not Apple), every call
//! immediately returns `Err(AccelError::Unsupported(...))` and the existing CPU
//! fallback takes over transparently.
//!
//! ## Usage
//!
//! ```toml
//! [dependencies]
//! oximedia-accel = { version = "*", features = ["metal-backend"] }
//! ```
//!
//! ## MSL shaders
//!
//! YUV ↔ RGB conversion and bilinear scale kernels are embedded as `const` MSL
//! source strings and compiled at runtime via `MTLDevice.newLibraryWithSource`.
//! No external `.metallib` files are required.

#![allow(dead_code)]

use crate::cpu_fallback::CpuAccel;
use crate::error::{AccelError, AccelResult};
use crate::traits::{HardwareAccel, ScaleFilter};
use oximedia_core::PixelFormat;

// ─── MSL shader sources (embedded) ──────────────────────────────────────────

/// MSL source for YUV ↔ RGB color conversion and bilinear image scaling.
///
/// Entry points:
/// - `yuv_to_rgb_kernel`  – convert packed RGBA/YUVA (BT.601) on-device
/// - `rgb_to_yuv_kernel`  – convert packed RGBA → YUVA (BT.601) on-device
/// - `bilinear_scale_kernel` – bilinear rescale of an RGBA image
const METAL_COLOR_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ── Pixel helpers ──────────────────────────────────────────────────────────

inline float4 unpack_rgba(uint packed) {
    float r = float((packed >> 24) & 0xFF) / 255.0f;
    float g = float((packed >> 16) & 0xFF) / 255.0f;
    float b = float((packed >>  8) & 0xFF) / 255.0f;
    float a = float( packed        & 0xFF) / 255.0f;
    return float4(r, g, b, a);
}

inline uint pack_rgba(float4 v) {
    uint r = uint(clamp(v.r * 255.0f, 0.0f, 255.0f));
    uint g = uint(clamp(v.g * 255.0f, 0.0f, 255.0f));
    uint b = uint(clamp(v.b * 255.0f, 0.0f, 255.0f));
    uint a = uint(clamp(v.a * 255.0f, 0.0f, 255.0f));
    return (r << 24) | (g << 16) | (b << 8) | a;
}

// ── BT.601 YUV ↔ RGB ─────────────────────────────────────────────────────

struct ConvertParams {
    uint width;
    uint height;
};

kernel void yuv_to_rgb_kernel(
    const device uint*  input  [[ buffer(0) ]],
          device uint*  output [[ buffer(1) ]],
    constant ConvertParams& p  [[ buffer(2) ]],
    uint2 gid [[ thread_position_in_grid ]])
{
    if (gid.x >= p.width || gid.y >= p.height) return;
    uint idx     = gid.y * p.width + gid.x;
    float4 yuva  = unpack_rgba(input[idx]);
    float  Y     = yuva.r;
    float  Cb    = yuva.g - 0.5f;
    float  Cr    = yuva.b - 0.5f;
    float  r     = clamp(Y + 1.402f   * Cr,            0.0f, 1.0f);
    float  g     = clamp(Y - 0.344136f * Cb - 0.714136f * Cr, 0.0f, 1.0f);
    float  b     = clamp(Y + 1.772f   * Cb,            0.0f, 1.0f);
    output[idx]  = pack_rgba(float4(r, g, b, yuva.a));
}

kernel void rgb_to_yuv_kernel(
    const device uint*  input  [[ buffer(0) ]],
          device uint*  output [[ buffer(1) ]],
    constant ConvertParams& p  [[ buffer(2) ]],
    uint2 gid [[ thread_position_in_grid ]])
{
    if (gid.x >= p.width || gid.y >= p.height) return;
    uint idx    = gid.y * p.width + gid.x;
    float4 rgba = unpack_rgba(input[idx]);
    float  r    = rgba.r;
    float  g    = rgba.g;
    float  b    = rgba.b;
    float  Y    = clamp( 0.299f   * r + 0.587f * g + 0.114f * b,      0.0f, 1.0f);
    float  Cb   = clamp(-0.168736f* r - 0.331264f*g + 0.5f  * b + 0.5f, 0.0f, 1.0f);
    float  Cr   = clamp( 0.5f    * r - 0.418688f*g - 0.081312f*b + 0.5f, 0.0f, 1.0f);
    output[idx] = pack_rgba(float4(Y, Cb, Cr, rgba.a));
}

// ── Bilinear scale ────────────────────────────────────────────────────────

struct ScaleParams {
    uint src_width;
    uint src_height;
    uint dst_width;
    uint dst_height;
};

kernel void bilinear_scale_kernel(
    const device uint*  input  [[ buffer(0) ]],
          device uint*  output [[ buffer(1) ]],
    constant ScaleParams& p    [[ buffer(2) ]],
    uint2 gid [[ thread_position_in_grid ]])
{
    if (gid.x >= p.dst_width || gid.y >= p.dst_height) return;

    float fx = (float(gid.x) + 0.5f) * float(p.src_width)  / float(p.dst_width)  - 0.5f;
    float fy = (float(gid.y) + 0.5f) * float(p.src_height) / float(p.dst_height) - 0.5f;

    int x0 = max(0,           int(floor(fx)));
    int y0 = max(0,           int(floor(fy)));
    int x1 = min(int(p.src_width)  - 1, x0 + 1);
    int y1 = min(int(p.src_height) - 1, y0 + 1);
    float wx = fx - float(x0);
    float wy = fy - float(y0);

    float4 c00 = unpack_rgba(input[y0 * p.src_width + x0]);
    float4 c10 = unpack_rgba(input[y0 * p.src_width + x1]);
    float4 c01 = unpack_rgba(input[y1 * p.src_width + x0]);
    float4 c11 = unpack_rgba(input[y1 * p.src_width + x1]);

    float4 top    = mix(c00, c10, wx);
    float4 bottom = mix(c01, c11, wx);
    float4 result = mix(top, bottom, wy);

    output[gid.y * p.dst_width + gid.x] = pack_rgba(result);
}
"#;

// ─── Platform-specific implementation ────────────────────────────────────────

/// Device information reported by Metal.
#[derive(Debug, Clone)]
pub struct MetalDeviceInfo {
    /// Device name (e.g. "Apple M3 Pro").
    pub name: String,
    /// GPU family tier (1–9, approximate).
    pub gpu_family: u32,
    /// Total unified memory in bytes (Apple Silicon).
    pub unified_memory_bytes: u64,
    /// Recommended max working-set size in bytes.
    pub recommended_max_working_set_bytes: u64,
    /// Whether GPU and CPU share memory (Apple Silicon UMA).
    pub has_unified_memory: bool,
    /// Whether this is a low-power (integrated) GPU.
    pub is_low_power: bool,
    /// Whether the device is headless (no display).
    pub is_headless: bool,
}

impl MetalDeviceInfo {
    /// Create a synthetic stub info for CI / non-Metal platforms.
    #[must_use]
    pub fn stub() -> Self {
        Self {
            name: "Metal Stub (no device)".to_string(),
            gpu_family: 0,
            unified_memory_bytes: 0,
            recommended_max_working_set_bytes: 0,
            has_unified_memory: false,
            is_low_power: false,
            is_headless: true,
        }
    }

    /// Total unified memory in gigabytes.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn unified_memory_gb(&self) -> f64 {
        self.unified_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// State of the Metal backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalBackendState {
    /// Metal is not available on this platform or was not compiled in.
    Unavailable,
    /// Metal is available but device enumeration failed.
    NoDevice,
    /// Metal is fully initialised and ready.
    Ready,
}

// ── Real Metal implementation (macOS + feature gate) ─────────────────────────

#[cfg(all(target_os = "macos", feature = "metal-backend"))]
mod metal_impl {
    use super::*;
    use metal::{
        Buffer as MetalBuffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef,
        ComputePipelineState, Device, MTLResourceOptions, MTLSize,
    };
    use std::ffi::c_void;

    /// Wrapper around an MTLBuffer that tracks its byte length.
    pub(super) struct MtlBuf {
        pub buf: MetalBuffer,
        pub len: usize,
    }

    impl MtlBuf {
        pub(super) fn new(device: &Device, len: usize, options: MTLResourceOptions) -> Self {
            let buf = device.new_buffer(len as u64, options);
            Self { buf, len }
        }

        pub(super) fn write(&self, data: &[u8]) {
            let ptr = self.buf.contents() as *mut u8;
            // SAFETY: Metal guarantees the buffer pointer is valid and
            //         the length matches the allocation we requested.
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len()) };
        }

        pub(super) fn read(&self, out: &mut [u8]) {
            let ptr = self.buf.contents() as *const u8;
            // SAFETY: same guarantee as write().
            unsafe { std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), out.len()) };
        }
    }

    /// Compiled pipeline and the command queue used to dispatch it.
    pub(super) struct MtlKernels {
        pub queue: CommandQueue,
        pub yuv_to_rgb: ComputePipelineState,
        pub rgb_to_yuv: ComputePipelineState,
        pub bilinear_scale: ComputePipelineState,
    }

    /// Compile all kernels from MSL source.
    pub(super) fn compile_kernels(device: &Device) -> AccelResult<MtlKernels> {
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(super::METAL_COLOR_MSL, &options)
            .map_err(|e| AccelError::ShaderCompilation(e.to_string()))?;

        let make_pipeline = |name: &str| -> AccelResult<ComputePipelineState> {
            let func = library
                .get_function(name, None)
                .map_err(|e| AccelError::ShaderCompilation(format!("{name}: {e}")))?;
            device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| AccelError::PipelineCreation(e.to_string()))
        };

        Ok(MtlKernels {
            queue: device.new_command_queue(),
            yuv_to_rgb: make_pipeline("yuv_to_rgb_kernel")?,
            rgb_to_yuv: make_pipeline("rgb_to_yuv_kernel")?,
            bilinear_scale: make_pipeline("bilinear_scale_kernel")?,
        })
    }

    /// Execute a compute kernel synchronously.
    ///
    /// `set_buffers` is called to bind buffers/uniforms before dispatch.
    /// `grid` is the total number of threads (width × height for 2D), split
    /// across the GPU's threadgroup size automatically.
    pub(super) fn dispatch_1d<F>(
        kernels: &MtlKernels,
        pipeline: &ComputePipelineState,
        width: u32,
        height: u32,
        set_buffers: F,
    ) -> AccelResult<()>
    where
        F: Fn(&ComputeCommandEncoderRef),
    {
        let cmd_buf = kernels
            .queue
            .new_command_buffer_with_unretained_references();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        set_buffers(encoder);

        let thread_w = pipeline.thread_execution_width();
        let thread_h = pipeline.max_total_threads_per_threadgroup() / thread_w;
        let threadgroup = MTLSize {
            width: thread_w,
            height: thread_h,
            depth: 1,
        };
        let grid = MTLSize {
            width: u64::from(width),
            height: u64::from(height),
            depth: 1,
        };
        encoder.dispatch_thread_groups_indirect(grid, threadgroup);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        Ok(())
    }

    /// Allocate a shared (CPU+GPU) Metal buffer pre-filled with `data`.
    pub(super) fn upload_buf(device: &Device, data: &[u8]) -> MtlBuf {
        let buf = MtlBuf::new(device, data.len(), MTLResourceOptions::StorageModeShared);
        buf.write(data);
        buf
    }

    /// Allocate a shared Metal buffer large enough to hold `len` bytes.
    pub(super) fn output_buf(device: &Device, len: usize) -> MtlBuf {
        MtlBuf::new(device, len, MTLResourceOptions::StorageModeShared)
    }

    /// Convenience: write a `u32` as 4 little-endian bytes into a `[u8; 8]`.
    fn u32_le(v: u32) -> [u8; 4] {
        v.to_le_bytes()
    }

    pub(super) fn make_convert_params(width: u32, height: u32) -> [u8; 8] {
        let mut out = [0u8; 8];
        out[0..4].copy_from_slice(&u32_le(width));
        out[4..8].copy_from_slice(&u32_le(height));
        out
    }

    pub(super) fn make_scale_params(sw: u32, sh: u32, dw: u32, dh: u32) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[0..4].copy_from_slice(&u32_le(sw));
        out[4..8].copy_from_slice(&u32_le(sh));
        out[8..12].copy_from_slice(&u32_le(dw));
        out[12..16].copy_from_slice(&u32_le(dh));
        out
    }
}

// ─── MetalAccel struct ────────────────────────────────────────────────────────

/// Metal acceleration backend.
///
/// On macOS with the `metal-backend` feature, this dispatches `scale_image`,
/// `convert_color`, and `motion_estimation` to Metal compute pipelines.
///
/// On non-Apple platforms or without the feature, every operation is delegated
/// to the CPU fallback transparently.
pub struct MetalAccel {
    state: MetalBackendState,
    device_info: Option<MetalDeviceInfo>,
    cpu_fallback: CpuAccel,
    /// Live Metal context — only populated when `state == Ready` and the
    /// `metal-backend` feature is enabled.
    #[cfg(all(target_os = "macos", feature = "metal-backend"))]
    metal_ctx: Option<MetalCtx>,
}

/// Holds the Metal device and compiled compute kernels.
#[cfg(all(target_os = "macos", feature = "metal-backend"))]
struct MetalCtx {
    device: metal::Device,
    kernels: metal_impl::MtlKernels,
}

impl MetalAccel {
    /// Attempt to initialise the Metal backend.
    ///
    /// On platforms without Metal support, returns a valid `MetalAccel` in
    /// the `Unavailable` state (never returns an `Err`).
    #[must_use]
    pub fn new() -> Self {
        #[cfg(all(target_os = "macos", feature = "metal-backend"))]
        {
            Self::try_init_metal()
        }
        #[cfg(not(all(target_os = "macos", feature = "metal-backend")))]
        {
            Self {
                state: MetalBackendState::Unavailable,
                device_info: None,
                cpu_fallback: CpuAccel::new(),
            }
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal-backend"))]
    fn try_init_metal() -> Self {
        // Request the system-default Metal device (equivalent to MTLCreateSystemDefaultDevice).
        let device = match metal::Device::system_default() {
            Some(d) => d,
            None => {
                return Self {
                    state: MetalBackendState::NoDevice,
                    device_info: None,
                    cpu_fallback: CpuAccel::new(),
                    metal_ctx: None,
                }
            }
        };

        // Compile kernels from embedded MSL.
        let kernels = match metal_impl::compile_kernels(&device) {
            Ok(k) => k,
            Err(e) => {
                tracing::warn!("Metal kernel compilation failed: {e}");
                return Self {
                    state: MetalBackendState::NoDevice,
                    device_info: None,
                    cpu_fallback: CpuAccel::new(),
                    metal_ctx: None,
                };
            }
        };

        let info = MetalDeviceInfo {
            name: device.name().to_string(),
            gpu_family: 0, // family tier detection requires GPU-family enum queries
            unified_memory_bytes: 0, // not directly exposed without IOKit
            recommended_max_working_set_bytes: device.recommended_max_working_set_size(),
            has_unified_memory: device.has_unified_memory(),
            is_low_power: device.is_low_power(),
            is_headless: device.is_headless(),
        };

        Self {
            state: MetalBackendState::Ready,
            device_info: Some(info),
            cpu_fallback: CpuAccel::new(),
            metal_ctx: Some(MetalCtx { device, kernels }),
        }
    }

    /// Returns the current backend state.
    #[must_use]
    pub fn state(&self) -> MetalBackendState {
        self.state
    }

    /// Returns device information if the backend is ready.
    #[must_use]
    pub fn device_info(&self) -> Option<&MetalDeviceInfo> {
        self.device_info.as_ref()
    }

    /// Returns `true` if Metal acceleration is actually active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.state == MetalBackendState::Ready
    }

    /// Returns the backend name suitable for display.
    #[must_use]
    pub fn backend_name(&self) -> &str {
        match self.state {
            MetalBackendState::Ready => self
                .device_info
                .as_ref()
                .map(|d| d.name.as_str())
                .unwrap_or("Metal"),
            MetalBackendState::NoDevice => "Metal (no device)",
            MetalBackendState::Unavailable => "Metal (unavailable)",
        }
    }
}

impl Default for MetalAccel {
    fn default() -> Self {
        Self::new()
    }
}

impl HardwareAccel for MetalAccel {
    fn scale_image(
        &self,
        input: &[u8],
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
        format: PixelFormat,
        filter: ScaleFilter,
    ) -> AccelResult<Vec<u8>> {
        #[cfg(all(target_os = "macos", feature = "metal-backend"))]
        if self.state == MetalBackendState::Ready {
            if let Some(ctx) = &self.metal_ctx {
                return self.scale_image_metal(
                    ctx, input, src_width, src_height, dst_width, dst_height, format, filter,
                );
            }
        }

        // CPU fallback.
        self.cpu_fallback
            .scale_image(input, src_width, src_height, dst_width, dst_height, format, filter)
    }

    fn convert_color(
        &self,
        input: &[u8],
        width: u32,
        height: u32,
        src_format: PixelFormat,
        dst_format: PixelFormat,
    ) -> AccelResult<Vec<u8>> {
        #[cfg(all(target_os = "macos", feature = "metal-backend"))]
        if self.state == MetalBackendState::Ready {
            if let Some(ctx) = &self.metal_ctx {
                return self.convert_color_metal(ctx, input, width, height, src_format, dst_format);
            }
        }

        self.cpu_fallback
            .convert_color(input, width, height, src_format, dst_format)
    }

    fn motion_estimation(
        &self,
        reference: &[u8],
        current: &[u8],
        width: u32,
        height: u32,
        block_size: u32,
    ) -> AccelResult<Vec<(i16, i16)>> {
        // Metal motion estimation requires a multi-pass SAD reduction that is
        // substantially more complex than color/scale; delegate to CPU for now.
        // (Planned for a future release as GPU NLM is planned for oximedia-gpu.)
        self.cpu_fallback
            .motion_estimation(reference, current, width, height, block_size)
    }
}

// ── Metal-specific operation implementations ─────────────────────────────────

#[cfg(all(target_os = "macos", feature = "metal-backend"))]
impl MetalAccel {
    /// Scale an RGBA image using the Metal bilinear scale kernel.
    #[allow(clippy::too_many_arguments)]
    fn scale_image_metal(
        &self,
        ctx: &MetalCtx,
        input: &[u8],
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
        format: PixelFormat,
        filter: ScaleFilter,
    ) -> AccelResult<Vec<u8>> {
        // Metal bilinear kernel works on packed-RGBA u32 pixels.
        // For non-RGBA formats or non-bilinear filters fall back to CPU.
        if format != PixelFormat::Rgba32 || filter == ScaleFilter::Nearest {
            return self.cpu_fallback.scale_image(
                input, src_width, src_height, dst_width, dst_height, format, filter,
            );
        }

        let expected = (src_width * src_height * 4) as usize;
        if input.len() != expected {
            return Err(AccelError::BufferSizeMismatch {
                expected,
                actual: input.len(),
            });
        }

        // Pack u8 RGBA bytes into u32 words (R<<24 | G<<16 | B<<8 | A).
        let mut input_u32: Vec<u32> = Vec::with_capacity((src_width * src_height) as usize);
        for chunk in input.chunks_exact(4) {
            let packed = ((chunk[0] as u32) << 24)
                | ((chunk[1] as u32) << 16)
                | ((chunk[2] as u32) << 8)
                | (chunk[3] as u32);
            input_u32.push(packed);
        }
        let input_bytes: &[u8] = bytemuck::cast_slice(&input_u32);
        let output_pixel_count = (dst_width * dst_height) as usize;
        let output_byte_len = output_pixel_count * 4;

        let input_buf = metal_impl::upload_buf(&ctx.device, input_bytes);
        let output_buf = metal_impl::output_buf(&ctx.device, output_byte_len);
        let params_raw = metal_impl::make_scale_params(src_width, src_height, dst_width, dst_height);
        let params_buf = metal_impl::upload_buf(&ctx.device, &params_raw);

        metal_impl::dispatch_1d(
            &ctx.kernels,
            &ctx.kernels.bilinear_scale,
            dst_width,
            dst_height,
            |enc| {
                enc.set_buffer(0, Some(&input_buf.buf), 0);
                enc.set_buffer(1, Some(&output_buf.buf), 0);
                enc.set_buffer(2, Some(&params_buf.buf), 0);
            },
        )?;

        // Unpack u32 → RGBA bytes.
        let raw_out_len = output_byte_len; // same as output_buf len
        let mut raw = vec![0u8; raw_out_len];
        output_buf.read(&mut raw);
        let result_u32: &[u32] = bytemuck::cast_slice(&raw);
        let mut out = vec![0u8; output_byte_len];
        for (i, &packed) in result_u32.iter().enumerate() {
            out[i * 4]     = ((packed >> 24) & 0xFF) as u8;
            out[i * 4 + 1] = ((packed >> 16) & 0xFF) as u8;
            out[i * 4 + 2] = ((packed >>  8) & 0xFF) as u8;
            out[i * 4 + 3] = ( packed        & 0xFF) as u8;
        }
        Ok(out)
    }

    /// Convert an image between YUV and RGB using the Metal kernel (BT.601).
    fn convert_color_metal(
        &self,
        ctx: &MetalCtx,
        input: &[u8],
        width: u32,
        height: u32,
        src_format: PixelFormat,
        dst_format: PixelFormat,
    ) -> AccelResult<Vec<u8>> {
        // Metal kernels only handle Rgba32 ↔ Yuv444p (packed YUVA) for now.
        let is_rgb_to_yuv =
            src_format == PixelFormat::Rgba32 && dst_format == PixelFormat::Yuv444p;
        let is_yuv_to_rgb =
            src_format == PixelFormat::Yuv444p && dst_format == PixelFormat::Rgba32;

        if !is_rgb_to_yuv && !is_yuv_to_rgb {
            return self
                .cpu_fallback
                .convert_color(input, width, height, src_format, dst_format);
        }

        let num_pixels = (width * height) as usize;
        let expected = num_pixels * 4;
        if input.len() != expected {
            return Err(AccelError::BufferSizeMismatch {
                expected,
                actual: input.len(),
            });
        }

        // Pack input RGBA bytes as u32 words.
        let mut input_u32: Vec<u32> = Vec::with_capacity(num_pixels);
        for chunk in input.chunks_exact(4) {
            let packed = ((chunk[0] as u32) << 24)
                | ((chunk[1] as u32) << 16)
                | ((chunk[2] as u32) << 8)
                | (chunk[3] as u32);
            input_u32.push(packed);
        }
        let input_bytes: &[u8] = bytemuck::cast_slice(&input_u32);
        let output_byte_len = num_pixels * 4;

        let input_buf = metal_impl::upload_buf(&ctx.device, input_bytes);
        let output_buf = metal_impl::output_buf(&ctx.device, output_byte_len);
        let params_raw = metal_impl::make_convert_params(width, height);
        let params_buf = metal_impl::upload_buf(&ctx.device, &params_raw);

        let pipeline = if is_rgb_to_yuv {
            &ctx.kernels.rgb_to_yuv
        } else {
            &ctx.kernels.yuv_to_rgb
        };

        metal_impl::dispatch_1d(&ctx.kernels, pipeline, width, height, |enc| {
            enc.set_buffer(0, Some(&input_buf.buf), 0);
            enc.set_buffer(1, Some(&output_buf.buf), 0);
            enc.set_buffer(2, Some(&params_buf.buf), 0);
        })?;

        let mut raw = vec![0u8; output_byte_len];
        output_buf.read(&mut raw);
        let result_u32: &[u32] = bytemuck::cast_slice(&raw);
        let mut out = vec![0u8; output_byte_len];
        for (i, &packed) in result_u32.iter().enumerate() {
            out[i * 4]     = ((packed >> 24) & 0xFF) as u8;
            out[i * 4 + 1] = ((packed >> 16) & 0xFF) as u8;
            out[i * 4 + 2] = ((packed >>  8) & 0xFF) as u8;
            out[i * 4 + 3] = ( packed        & 0xFF) as u8;
        }
        Ok(out)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_accel_new_does_not_panic() {
        let accel = MetalAccel::new();
        // On non-macOS platforms it must always be Unavailable.
        #[cfg(not(target_os = "macos"))]
        assert_eq!(accel.state(), MetalBackendState::Unavailable);

        let _ = accel.backend_name();
        let _ = accel.is_active();
    }

    #[test]
    fn test_metal_accel_default() {
        let accel = MetalAccel::default();
        // is_active can only be true when state == Ready.
        assert!(!accel.is_active() || accel.state() == MetalBackendState::Ready);
    }

    #[test]
    fn test_metal_device_info_stub() {
        let info = MetalDeviceInfo::stub();
        assert_eq!(info.unified_memory_bytes, 0);
        assert!((info.unified_memory_gb() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_metal_device_info_memory_gb() {
        let info = MetalDeviceInfo {
            unified_memory_bytes: 16 * 1024 * 1024 * 1024,
            ..MetalDeviceInfo::stub()
        };
        assert!((info.unified_memory_gb() - 16.0).abs() < 0.01);
    }

    #[test]
    fn test_metal_scale_image_fallback() {
        let accel = MetalAccel::new();
        let input = vec![128u8; 4 * 4 * 3];
        let result =
            accel.scale_image(&input, 4, 4, 2, 2, PixelFormat::Rgb24, ScaleFilter::Nearest);
        // When unavailable, falls back to CPU which succeeds.
        assert!(result.is_ok(), "fallback should succeed: {:?}", result.err());
    }

    #[test]
    fn test_metal_convert_color_fallback() {
        let accel = MetalAccel::new();
        let input = vec![128u8; 4 * 4 * 3];
        let result =
            accel.convert_color(&input, 4, 4, PixelFormat::Rgb24, PixelFormat::Yuv420p);
        assert!(result.is_ok(), "fallback should succeed: {:?}", result.err());
    }

    #[test]
    fn test_metal_motion_estimation_fallback() {
        let accel = MetalAccel::new();
        let frame = vec![0u8; 8 * 8];
        let result = accel.motion_estimation(&frame, &frame, 8, 8, 4);
        assert!(result.is_ok());
    }

    #[test]
    fn test_metal_backend_state_unavailable_has_no_device_info() {
        let accel = MetalAccel::new();
        if accel.state() == MetalBackendState::Unavailable {
            assert!(accel.device_info().is_none());
        }
    }

    #[test]
    fn test_metal_backend_name_not_empty() {
        let accel = MetalAccel::new();
        assert!(!accel.backend_name().is_empty());
    }

    // ── Metal-specific tests (macOS + feature) ───────────────────────────────

    #[cfg(all(target_os = "macos", feature = "metal-backend"))]
    #[test]
    fn test_metal_scale_image_rgba_bilinear() {
        let accel = MetalAccel::new();
        if !accel.is_active() {
            return; // no Metal GPU in this environment
        }
        // 4×4 solid orange RGBA image scaled to 2×2.
        let input: Vec<u8> = (0..16).flat_map(|_| [255u8, 128, 0, 255]).collect();
        let result = accel.scale_image(
            &input,
            4,
            4,
            2,
            2,
            PixelFormat::Rgba32,
            ScaleFilter::Bilinear,
        );
        assert!(result.is_ok(), "Metal scale failed: {:?}", result.err());
        let out = result.unwrap();
        assert_eq!(out.len(), 2 * 2 * 4);
        // Each output pixel should be close to the solid colour.
        for chunk in out.chunks_exact(4) {
            assert!((chunk[0] as i32 - 255).abs() < 2, "R channel mismatch");
            assert!((chunk[1] as i32 - 128).abs() < 2, "G channel mismatch");
            assert!((chunk[2] as i32 -   0).abs() < 2, "B channel mismatch");
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal-backend"))]
    #[test]
    fn test_metal_convert_color_rgba_yuv_roundtrip() {
        let accel = MetalAccel::new();
        if !accel.is_active() {
            return;
        }
        // 4×4 solid grey image.
        let input: Vec<u8> = (0..16).flat_map(|_| [128u8, 128, 128, 255]).collect();
        let yuv = accel.convert_color(&input, 4, 4, PixelFormat::Rgba32, PixelFormat::Yuv444p);
        assert!(yuv.is_ok(), "rgb→yuv failed: {:?}", yuv.err());
        let yuv_data = yuv.unwrap();
        let rgb = accel.convert_color(&yuv_data, 4, 4, PixelFormat::Yuv444p, PixelFormat::Rgba32);
        assert!(rgb.is_ok(), "yuv→rgb failed: {:?}", rgb.err());
        // After round-trip, channels should be within ±5 of the original.
        let rgb_data = rgb.unwrap();
        for (orig, rt) in input.iter().zip(rgb_data.iter()) {
            assert!(
                (*orig as i32 - *rt as i32).abs() <= 5,
                "round-trip error: orig={orig}, rt={rt}"
            );
        }
    }
}
