//! WebGPU acceleration backend for WASM / browser targets.
//!
//! When the `webgpu` feature is enabled a real `wgpu`-backed `WebGpuContext`
//! is provided. Without the feature the entire file falls back to the CPU
//! stub so that the rest of the codebase continues to compile on every target.
//!
//! # Feature gate
//!
//! ```toml
//! oximedia-accel = { features = ["webgpu"] }
//! ```

#![allow(dead_code)]

use crate::cpu_fallback::CpuAccel;
use crate::error::AccelResult;
use crate::traits::{HardwareAccel, ScaleFilter};
use oximedia_core::PixelFormat;

// ── Shared types (always compiled) ──────────────────────────────────────────

/// State of the WebGPU backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebGpuBackendState {
    /// WebGPU is not available in this environment (non-WASM, or browser lacks
    /// the `navigator.gpu` API), or the `webgpu` feature is disabled.
    Unavailable,
    /// WebGPU adapter could not be obtained.
    NoAdapter,
    /// WebGPU is fully initialised and ready.
    Ready,
}

/// Information about the WebGPU adapter.
#[derive(Debug, Clone)]
pub struct WebGpuAdapterInfo {
    /// Adapter name (e.g. "Intel HD Graphics 620 (SwiftShader)").
    pub name: String,
    /// Vendor identifier string.
    pub vendor: String,
    /// Device identifier string.
    pub device: String,
    /// Backend type (e.g. "vulkan", "metal", "dx12", "opengl").
    pub backend: String,
}

impl WebGpuAdapterInfo {
    /// Create a synthetic stub info for non-WASM platforms.
    #[must_use]
    pub fn stub() -> Self {
        Self {
            name: "WebGPU Stub (no adapter)".to_string(),
            vendor: "unknown".to_string(),
            device: "stub".to_string(),
            backend: "none".to_string(),
        }
    }
}

// ── Real WebGPU implementation (feature = "webgpu") ─────────────────────────

#[cfg(feature = "webgpu")]
mod real {
    use super::*;
    use crate::error::AccelError;
    use wgpu::{
        Adapter, Backends, Device, DeviceDescriptor, Features, Instance, InstanceDescriptor,
        Limits, Queue, RequestAdapterOptions,
    };

    // WGSL kernel: bilinear image scaling.
    // Reads from a flat RGBA u8 storage buffer, writes scaled RGBA to output.
    const SCALE_SHADER: &str = r#"
struct ScaleParams {
    src_width:  u32,
    src_height: u32,
    dst_width:  u32,
    dst_height: u32,
}

@group(0) @binding(0) var<storage, read>       src:    array<u32>;
@group(0) @binding(1) var<storage, read_write> dst:    array<u32>;
@group(0) @binding(2) var<uniform>             params: ScaleParams;

fn unpack_rgba(v: u32) -> vec4<f32> {
    return vec4<f32>(
        f32((v >> 24u) & 0xFFu) / 255.0,
        f32((v >> 16u) & 0xFFu) / 255.0,
        f32((v >>  8u) & 0xFFu) / 255.0,
        f32( v         & 0xFFu) / 255.0,
    );
}

fn pack_rgba(c: vec4<f32>) -> u32 {
    let r = u32(clamp(c.r * 255.0, 0.0, 255.0));
    let g = u32(clamp(c.g * 255.0, 0.0, 255.0));
    let b = u32(clamp(c.b * 255.0, 0.0, 255.0));
    let a = u32(clamp(c.a * 255.0, 0.0, 255.0));
    return (r << 24u) | (g << 16u) | (b << 8u) | a;
}

fn fetch(x: u32, y: u32) -> vec4<f32> {
    let cx = clamp(x, 0u, params.src_width  - 1u);
    let cy = clamp(y, 0u, params.src_height - 1u);
    return unpack_rgba(src[cy * params.src_width + cx]);
}

@compute @workgroup_size(16, 16, 1)
fn scale_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dx = gid.x;
    let dy = gid.y;
    if (dx >= params.dst_width || dy >= params.dst_height) { return; }

    // Map destination pixel centre to source space
    let u = (f32(dx) + 0.5) * f32(params.src_width)  / f32(params.dst_width)  - 0.5;
    let v = (f32(dy) + 0.5) * f32(params.src_height) / f32(params.dst_height) - 0.5;

    let x0 = u32(max(floor(u), 0.0));
    let y0 = u32(max(floor(v), 0.0));
    let x1 = min(x0 + 1u, params.src_width  - 1u);
    let y1 = min(y0 + 1u, params.src_height - 1u);

    let fx = fract(max(u, 0.0));
    let fy = fract(max(v, 0.0));

    let c00 = fetch(x0, y0);
    let c10 = fetch(x1, y0);
    let c01 = fetch(x0, y1);
    let c11 = fetch(x1, y1);

    let col = mix(mix(c00, c10, fx), mix(c01, c11, fx), fy);

    dst[dy * params.dst_width + dx] = pack_rgba(col);
}
"#;

    // WGSL kernel: YUV↔RGB colour conversion (BT.601 coefficients).
    // Packed RGBA storage for both input and output, alpha pass-through.
    const COLOR_SHADER: &str = r#"
struct ColorParams {
    width:     u32,
    height:    u32,
    direction: u32, // 0 = RGB→YUV, 1 = YUV→RGB
    _pad:      u32,
}

@group(0) @binding(0) var<storage, read>       src:    array<u32>;
@group(0) @binding(1) var<storage, read_write> dst:    array<u32>;
@group(0) @binding(2) var<uniform>             params: ColorParams;

fn unpack_rgba(v: u32) -> vec4<f32> {
    return vec4<f32>(
        f32((v >> 24u) & 0xFFu) / 255.0,
        f32((v >> 16u) & 0xFFu) / 255.0,
        f32((v >>  8u) & 0xFFu) / 255.0,
        f32( v         & 0xFFu) / 255.0,
    );
}

fn pack_rgba(c: vec4<f32>) -> u32 {
    let r = u32(clamp(c.r * 255.0, 0.0, 255.0));
    let g = u32(clamp(c.g * 255.0, 0.0, 255.0));
    let b = u32(clamp(c.b * 255.0, 0.0, 255.0));
    let a = u32(clamp(c.a * 255.0, 0.0, 255.0));
    return (r << 24u) | (g << 16u) | (b << 8u) | a;
}

// BT.601 full-range RGB→YCbCr
fn rgb_to_yuv(rgb: vec3<f32>) -> vec3<f32> {
    let y  =  0.299    * rgb.r + 0.587    * rgb.g + 0.114    * rgb.b;
    let cb = -0.168736 * rgb.r - 0.331264 * rgb.g + 0.5      * rgb.b + 0.5;
    let cr =  0.5      * rgb.r - 0.418688 * rgb.g - 0.081312 * rgb.b + 0.5;
    return vec3<f32>(y, cb, cr);
}

// BT.601 full-range YCbCr→RGB
fn yuv_to_rgb(yuv: vec3<f32>) -> vec3<f32> {
    let y  = yuv.x;
    let cb = yuv.y - 0.5;
    let cr = yuv.z - 0.5;
    let r = clamp(y                  + 1.402    * cr, 0.0, 1.0);
    let g = clamp(y - 0.344136 * cb - 0.714136 * cr, 0.0, 1.0);
    let b = clamp(y + 1.772    * cb,                  0.0, 1.0);
    return vec3<f32>(r, g, b);
}

@compute @workgroup_size(16, 16, 1)
fn color_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) { return; }

    let idx  = y * params.width + x;
    let rgba = unpack_rgba(src[idx]);

    var out: vec4<f32>;
    if (params.direction == 0u) {
        let yuv = rgb_to_yuv(rgba.rgb);
        out = vec4<f32>(yuv, rgba.a);
    } else {
        let rgb = yuv_to_rgb(rgba.rgb);
        out = vec4<f32>(rgb, rgba.a);
    }

    dst[idx] = pack_rgba(out);
}
"#;

    /// A live `wgpu` context (device + queue + adapter).
    pub struct WebGpuContext {
        pub adapter: Adapter,
        pub device: Device,
        pub queue: Queue,
    }

    impl WebGpuContext {
        /// Request a wgpu adapter and device asynchronously.
        pub async fn new_async() -> Result<Self, AccelError> {
            let instance = Instance::new(InstanceDescriptor {
                backends: Backends::all(),
                flags: wgpu::InstanceFlags::default(),
                memory_budget_thresholds: Default::default(),
                backend_options: Default::default(),
                display: None,
            });

            let adapter = instance
                .request_adapter(&RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .map_err(|e| {
                    AccelError::Unsupported(format!(
                        "No WebGPU adapter available on this platform: {e}"
                    ))
                })?;

            let (device, queue) = adapter
                .request_device(&DeviceDescriptor {
                    label: Some("oximedia-accel WebGPU"),
                    required_features: Features::empty(),
                    required_limits: Limits::downlevel_defaults(),
                    experimental_features: Default::default(),
                    memory_hints: Default::default(),
                    trace: Default::default(),
                })
                .await
                .map_err(|e| {
                    AccelError::Unsupported(format!("WebGPU device request failed: {e}"))
                })?;

            Ok(Self {
                adapter,
                device,
                queue,
            })
        }

        // ── Internal helpers ────────────────────────────────────────────────

        fn create_storage_buffer(&self, size: u64) -> wgpu::Buffer {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        }

        fn create_uniform_buffer(&self, data: &[u8]) -> wgpu::Buffer {
            use wgpu::util::DeviceExt;
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: data,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                })
        }

        fn upload_and_run(
            &self,
            shader_src: &str,
            entry_point: &str,
            uniform_bytes: &[u8],
            input_bytes: &[u8],
            output_len: u64,
            dispatch_x: u32,
            dispatch_y: u32,
        ) -> AccelResult<Vec<u8>> {
            use wgpu::util::DeviceExt;

            // Compile shader
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(entry_point),
                    source: wgpu::ShaderSource::Wgsl(shader_src.into()),
                });

            // Create buffers
            let src_buf = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("webgpu-src"),
                    contents: input_bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
            let dst_buf = self.create_storage_buffer(output_len);
            let uniform_buf = self.create_uniform_buffer(uniform_bytes);

            // Build bind group layout and pipeline
            let bgl = self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("webgpu-bgl"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("webgpu-pl"),
                        bind_group_layouts: &[Some(&bgl)],
                        immediate_size: 0,
                    });

            let pipeline = self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry_point),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    cache: None,
                });

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("webgpu-bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform_buf.as_entire_binding(),
                    },
                ],
            });

            // Dispatch
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }

            // Readback
            let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("webgpu-readback"),
                size: output_len,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(&dst_buf, 0, &readback, 0, output_len);
            self.queue.submit(Some(encoder.finish()));

            // Map + read
            let buf_slice = readback.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            buf_slice.map_async(wgpu::MapMode::Read, move |v| {
                let _ = tx.send(v);
            });
            // Poll until all submitted work completes
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            rx.recv()
                .map_err(|_| AccelError::Synchronization("WebGPU channel recv error".to_string()))?
                .map_err(|e| {
                    AccelError::Synchronization(format!("WebGPU map_async failed: {e:?}"))
                })?;

            let data = buf_slice.get_mapped_range();
            let result = data.to_vec();
            drop(data);
            readback.unmap();
            Ok(result)
        }

        // ── Public operations ───────────────────────────────────────────────

        /// GPU bilinear image scaling.
        ///
        /// Input/output are packed RGBA u8 buffers (each pixel = 4 bytes).
        pub fn scale_image_gpu(
            &self,
            input: &[u8],
            src_width: u32,
            src_height: u32,
            dst_width: u32,
            dst_height: u32,
        ) -> AccelResult<Vec<u8>> {
            // Uniform: 4 × u32
            let uniform = [src_width, src_height, dst_width, dst_height];
            let uniform_bytes = bytemuck::cast_slice::<u32, u8>(&uniform);

            let output_len = (dst_width * dst_height * 4) as u64;
            let dispatch_x = dst_width.div_ceil(16);
            let dispatch_y = dst_height.div_ceil(16);

            self.upload_and_run(
                SCALE_SHADER,
                "scale_main",
                uniform_bytes,
                input,
                output_len,
                dispatch_x,
                dispatch_y,
            )
        }

        /// GPU RGB↔YUV (BT.601) colour conversion.
        ///
        /// `direction = 0` → RGB→YUV, `direction = 1` → YUV→RGB.
        pub fn convert_color_gpu(
            &self,
            input: &[u8],
            width: u32,
            height: u32,
            direction: u32,
        ) -> AccelResult<Vec<u8>> {
            // Uniform: width, height, direction, _pad
            let uniform = [width, height, direction, 0u32];
            let uniform_bytes = bytemuck::cast_slice::<u32, u8>(&uniform);

            let output_len = (width * height * 4) as u64;
            let dispatch_x = width.div_ceil(16);
            let dispatch_y = height.div_ceil(16);

            self.upload_and_run(
                COLOR_SHADER,
                "color_main",
                uniform_bytes,
                input,
                output_len,
                dispatch_x,
                dispatch_y,
            )
        }
    }
}

// ── Public facade: WebGpuAccelBackend ────────────────────────────────────────

/// WebGPU acceleration backend.
///
/// When compiled with `features = ["webgpu"]` this will attempt to initialise
/// a real `wgpu` device for GPU-accelerated image scaling and colour
/// conversion.  Without the feature (or when no adapter is found) all
/// operations fall through to the CPU fallback.
pub struct WebGpuAccelBackend {
    state: WebGpuBackendState,
    adapter_info: Option<WebGpuAdapterInfo>,
    cpu_fallback: CpuAccel,
    #[cfg(feature = "webgpu")]
    gpu_ctx: Option<real::WebGpuContext>,
}

impl WebGpuAccelBackend {
    /// Create a new WebGPU backend, auto-detecting availability.
    ///
    /// When the `webgpu` feature is enabled this calls `pollster::block_on`
    /// to initialise the wgpu adapter synchronously; on WASM targets callers
    /// should prefer the async constructor.  Falls back silently to CPU when
    /// no adapter is found.
    pub fn new() -> Self {
        #[cfg(feature = "webgpu")]
        {
            match pollster::block_on(real::WebGpuContext::new_async()) {
                Ok(ctx) => {
                    let info = build_adapter_info(&ctx.adapter);
                    return Self {
                        state: WebGpuBackendState::Ready,
                        adapter_info: Some(info),
                        cpu_fallback: CpuAccel::new(),
                        gpu_ctx: Some(ctx),
                    };
                }
                Err(e) => {
                    tracing::warn!("WebGPU init failed, using CPU fallback: {e}");
                    return Self {
                        state: WebGpuBackendState::NoAdapter,
                        adapter_info: None,
                        cpu_fallback: CpuAccel::new(),
                        gpu_ctx: None,
                    };
                }
            }
        }
        #[cfg(not(feature = "webgpu"))]
        Self {
            state: WebGpuBackendState::Unavailable,
            adapter_info: None,
            cpu_fallback: CpuAccel::new(),
        }
    }

    /// Return the current backend state.
    #[must_use]
    pub fn state(&self) -> WebGpuBackendState {
        self.state
    }

    /// Return adapter information if the backend is ready.
    #[must_use]
    pub fn adapter_info(&self) -> Option<&WebGpuAdapterInfo> {
        self.adapter_info.as_ref()
    }

    /// Returns `true` if WebGPU acceleration is actually active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.state == WebGpuBackendState::Ready
    }

    /// Return `true` when the `webgpu` feature is compiled in.
    #[must_use]
    pub fn is_available() -> bool {
        cfg!(feature = "webgpu")
    }

    /// Return the backend name suitable for display.
    #[must_use]
    pub fn backend_name(&self) -> &str {
        match self.state {
            WebGpuBackendState::Ready => self
                .adapter_info
                .as_ref()
                .map(|a| a.name.as_str())
                .unwrap_or("WebGPU"),
            WebGpuBackendState::NoAdapter => "WebGPU (no adapter)",
            WebGpuBackendState::Unavailable => "WebGPU (unavailable)",
        }
    }
}

impl Default for WebGpuAccelBackend {
    fn default() -> Self {
        Self::new()
    }
}

// Build adapter info from a live wgpu adapter.
#[cfg(feature = "webgpu")]
fn build_adapter_info(adapter: &wgpu::Adapter) -> WebGpuAdapterInfo {
    let info = adapter.get_info();
    WebGpuAdapterInfo {
        name: info.name.clone(),
        vendor: format!("{:04x}", info.vendor),
        device: format!("{:04x}", info.device),
        backend: format!("{:?}", info.backend),
    }
}

impl HardwareAccel for WebGpuAccelBackend {
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
        #[cfg(feature = "webgpu")]
        if let Some(ctx) = &self.gpu_ctx {
            if format == PixelFormat::Rgba32 {
                return ctx.scale_image_gpu(input, src_width, src_height, dst_width, dst_height);
            }
        }
        self.cpu_fallback.scale_image(
            input, src_width, src_height, dst_width, dst_height, format, filter,
        )
    }

    fn convert_color(
        &self,
        input: &[u8],
        width: u32,
        height: u32,
        src_format: PixelFormat,
        dst_format: PixelFormat,
    ) -> AccelResult<Vec<u8>> {
        #[cfg(feature = "webgpu")]
        if let Some(ctx) = &self.gpu_ctx {
            match (src_format, dst_format) {
                (PixelFormat::Rgba32, PixelFormat::Yuv420p) => {
                    return ctx.convert_color_gpu(input, width, height, 0);
                }
                (PixelFormat::Yuv420p, PixelFormat::Rgba32) => {
                    return ctx.convert_color_gpu(input, width, height, 1);
                }
                _ => {}
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
        // Motion estimation does not have a GPU path yet; always use CPU.
        self.cpu_fallback
            .motion_estimation(reference, current, width, height, block_size)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webgpu_backend_new_does_not_panic() {
        let backend = WebGpuAccelBackend::new();
        let _ = backend.state();
        let _ = backend.backend_name();
        let _ = backend.is_active();
    }

    #[test]
    fn test_webgpu_backend_default() {
        let backend = WebGpuAccelBackend::default();
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Without a real GPU driver in CI the backend may be Ready (software
            // adapter) or NoAdapter/Unavailable — but it must not panic.
            let _state = backend.state();
        }
        let _ = backend;
    }

    #[test]
    fn test_webgpu_is_available_matches_feature() {
        #[cfg(feature = "webgpu")]
        assert!(WebGpuAccelBackend::is_available());
        #[cfg(not(feature = "webgpu"))]
        assert!(!WebGpuAccelBackend::is_available());
    }

    #[test]
    fn test_webgpu_scale_image_fallback() {
        let backend = WebGpuAccelBackend::new();
        let input = vec![128u8; 4 * 4 * 3];
        let result =
            backend.scale_image(&input, 4, 4, 2, 2, PixelFormat::Rgb24, ScaleFilter::Nearest);
        assert!(
            result.is_ok(),
            "fallback should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_webgpu_convert_color_fallback() {
        let backend = WebGpuAccelBackend::new();
        let input = vec![128u8; 4 * 4 * 3];
        let result = backend.convert_color(&input, 4, 4, PixelFormat::Rgb24, PixelFormat::Yuv420p);
        assert!(
            result.is_ok(),
            "fallback should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_webgpu_motion_estimation_fallback() {
        let backend = WebGpuAccelBackend::new();
        let frame = vec![0u8; 8 * 8];
        let result = backend.motion_estimation(&frame, &frame, 8, 8, 4);
        assert!(result.is_ok());
    }

    #[test]
    fn test_webgpu_adapter_info_none_when_unavailable() {
        let backend = WebGpuAccelBackend::new();
        if backend.state() == WebGpuBackendState::Unavailable {
            assert!(backend.adapter_info().is_none());
        }
    }

    #[test]
    fn test_webgpu_backend_name_not_empty() {
        let backend = WebGpuAccelBackend::new();
        assert!(!backend.backend_name().is_empty());
    }

    #[test]
    fn test_webgpu_adapter_info_stub() {
        let info = WebGpuAdapterInfo::stub();
        assert!(!info.name.is_empty());
        assert!(!info.vendor.is_empty());
    }

    /// When the webgpu feature is active and a GPU context was obtained,
    /// scale_image for RGBA32 should use the GPU path (or fall through to CPU
    /// if no adapter is present — both paths must return Ok).
    #[test]
    #[cfg(feature = "webgpu")]
    fn test_webgpu_rgba_scale_succeeds() {
        let backend = WebGpuAccelBackend::new();
        // 4×4 RGBA32 filled with mid-grey
        let input: Vec<u8> = vec![128u8, 64, 32, 255]
            .into_iter()
            .cycle()
            .take(4 * 4 * 4)
            .collect();
        let result = backend.scale_image(
            &input,
            4,
            4,
            2,
            2,
            PixelFormat::Rgba32,
            ScaleFilter::Bilinear,
        );
        assert!(
            result.is_ok(),
            "RGBA32 scale should succeed: {:?}",
            result.err()
        );
        if let Ok(out) = result {
            assert_eq!(out.len(), 2 * 2 * 4, "output must be 2×2×4 bytes");
        }
    }
}
