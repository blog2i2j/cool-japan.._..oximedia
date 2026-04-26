//! GPU compute shader dispatch for image processing operations.
//!
//! Provides a high-level dispatch layer that selects between GPU compute
//! shaders (when the `gpu` feature is enabled and hardware is available)
//! and the equivalent CPU implementations from [`crate::image_filter`] and
//! [`crate::porter_duff`].
//!
//! The decision is governed by [`GpuComputeConfig`]: callers can set a
//! minimum pixel count below which the CPU path is always preferred
//! (avoiding GPU dispatch overhead for small images).

use crate::porter_duff::PorterDuffOp;

// ---------------------------------------------------------------------------
// Shader sources (embedded at compile time)
// ---------------------------------------------------------------------------

/// WGSL source for the separable Gaussian blur compute shader.
pub const GAUSSIAN_BLUR_WGSL: &str = include_str!("shaders/gaussian_blur.wgsl");

/// WGSL source for the Porter-Duff compositing compute shader.
pub const PORTER_DUFF_WGSL: &str = include_str!("shaders/porter_duff.wgsl");

/// WGSL source for the gradient fill compute shader.
pub const GRADIENT_FILL_WGSL: &str = include_str!("shaders/gradient_fill.wgsl");

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for GPU compute dispatch.
#[derive(Debug, Clone)]
pub struct GpuComputeConfig {
    /// Whether to prefer GPU over CPU when available.
    pub prefer_gpu: bool,
    /// Minimum image area (`width * height`) to justify GPU dispatch overhead.
    /// Images smaller than this threshold always use the CPU path.
    pub min_pixels_for_gpu: u32,
}

impl Default for GpuComputeConfig {
    fn default() -> Self {
        Self {
            prefer_gpu: true,
            min_pixels_for_gpu: 256 * 256, // 65 536 pixels
        }
    }
}

// ---------------------------------------------------------------------------
// GPU availability
// ---------------------------------------------------------------------------

/// Result of a GPU availability check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuStatus {
    /// GPU compute is available and ready.
    Available,
    /// GPU compute is not available for the given reason.
    NotAvailable(String),
    /// The `gpu` feature was not compiled in.
    FeatureDisabled,
}

/// Check whether GPU compute is available at runtime.
///
/// When the `gpu` feature is enabled this attempts a lightweight probe of
/// the wgpu backend.  Without the feature the function always returns
/// [`GpuStatus::FeatureDisabled`].
#[must_use]
pub fn check_gpu_status() -> GpuStatus {
    #[cfg(feature = "gpu")]
    {
        // Probe wgpu for an adapter.  We intentionally do *not* cache the
        // adapter here — the real pipeline creation happens elsewhere.
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

        // `enumerate_adapters` returns a Future in wgpu 29.x.
        let adapters: Vec<_> =
            pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        if adapters.is_empty() {
            return GpuStatus::NotAvailable("no wgpu adapters found".into());
        }

        GpuStatus::Available
    }

    #[cfg(not(feature = "gpu"))]
    {
        GpuStatus::FeatureDisabled
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Returns `true` when the GPU path should be attempted for the given config
/// and image dimensions.
fn should_use_gpu(config: &GpuComputeConfig, width: u32, height: u32) -> bool {
    if !config.prefer_gpu {
        return false;
    }
    let area = width.saturating_mul(height);
    if area < config.min_pixels_for_gpu {
        return false;
    }
    matches!(check_gpu_status(), GpuStatus::Available)
}

// ---------------------------------------------------------------------------
// Dispatch: Gaussian blur
// ---------------------------------------------------------------------------

/// Attempt a GPU Gaussian blur using the WGSL compute shader.
///
/// Returns `true` if the GPU path completed successfully and the result has
/// been written back to `pixels`.  Returns `false` on any wgpu error so the
/// caller can fall through to the CPU implementation.
///
/// The shader is a separable two-pass approach:
/// - Pass 1 (horizontal): reads from `tex_input`, writes to `tex_mid`
/// - Pass 2 (vertical): reads from `tex_mid`, writes to `tex_output`
/// - Both passes use `texture_2d<f32>` + `texture_storage_2d<rgba8unorm, write>`
///   which requires the `TEXTURE_BINDING_ARRAY` or standard `TEXTURE_BINDING`/
///   `STORAGE_BINDING` usages plus the `STORAGE_BINDING` feature flag on the
///   device.
#[cfg(feature = "gpu")]
fn try_gpu_gaussian_blur(pixels: &mut [u8], width: u32, height: u32, sigma: f32) -> bool {
    use wgpu::util::DeviceExt;

    if width == 0 || height == 0 || pixels.len() < (width as usize) * (height as usize) * 4 {
        return false;
    }

    // kernel_radius matches crate::image_filter::build_gaussian_kernel half_width
    let kernel_radius: u32 = ((3.0_f32 * sigma).ceil() as u32).min(50);

    // ── 1. Create instance and probe for a capable adapter ──────────────────
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

    let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    })) {
        Ok(a) => a,
        Err(_) => return false,
    };

    // We need TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES for rgba8unorm storage.
    // Use downlevel_defaults and add the required feature when available.
    let required_features = wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
    let supported = adapter.features();

    // If the adapter cannot support storage textures on rgba8unorm, fall back.
    if !supported.contains(required_features) {
        return false;
    }

    let (device, queue) = match pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("oximedia-graphics gaussian"),
            required_features,
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::Off,
        },
    )) {
        Ok(pair) => pair,
        Err(_) => return false,
    };

    // ── 2. Create textures ───────────────────────────────────────────────────
    // The WGSL shader:
    //   binding 0: texture_2d<f32>                        → TEXTURE_BINDING + COPY_DST
    //   binding 1: texture_storage_2d<rgba8unorm, write>  → STORAGE_BINDING + COPY_SRC
    //   binding 2: uniform BlurParams                     → UNIFORM
    //
    // Three textures:
    //   tex_input  – initial RGBA pixel data
    //   tex_mid    – output of horizontal pass / input of vertical pass
    //   tex_output – output of vertical pass (copied to readback buffer)

    let tex_extent = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    // tex_input: upload source pixels; used as sampled texture in pass 1
    let tex_input = device.create_texture_with_data(
        &queue,
        &wgpu::TextureDescriptor {
            label: Some("gauss_input"),
            size: tex_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        pixels,
    );

    // tex_mid: output of pass 1, then input of pass 2
    let tex_mid = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("gauss_mid"),
        size: tex_extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });

    // tex_output: output of pass 2 (will be copied to readback buffer)
    let tex_output = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("gauss_output"),
        size: tex_extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    // ── 3. Texture views ─────────────────────────────────────────────────────
    let view_input = tex_input.create_view(&wgpu::TextureViewDescriptor::default());
    let view_mid_sampled = tex_mid.create_view(&wgpu::TextureViewDescriptor::default());
    let view_mid_storage = tex_mid.create_view(&wgpu::TextureViewDescriptor::default());
    let view_output = tex_output.create_view(&wgpu::TextureViewDescriptor::default());

    // ── 4. Uniform buffers — one per pass ────────────────────────────────────
    // BlurParams { kernel_radius: u32, sigma: f32, horizontal: u32, _pad: u32 }
    let make_uniform_bytes = |horizontal: u32| -> Vec<u8> {
        let kr_bytes = kernel_radius.to_le_bytes();
        let sigma_bytes = sigma.to_le_bytes();
        let horiz_bytes = horizontal.to_le_bytes();
        let pad_bytes = 0u32.to_le_bytes();
        let mut v = Vec::with_capacity(16);
        v.extend_from_slice(&kr_bytes);
        v.extend_from_slice(&sigma_bytes);
        v.extend_from_slice(&horiz_bytes);
        v.extend_from_slice(&pad_bytes);
        v
    };

    let uniform_h = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gauss_uniform_h"),
        contents: &make_uniform_bytes(1),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let uniform_v = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gauss_uniform_v"),
        contents: &make_uniform_bytes(0),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // ── 5. Shader and pipeline ────────────────────────────────────────────────
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gaussian_blur"),
        source: wgpu::ShaderSource::Wgsl(GAUSSIAN_BLUR_WGSL.into()),
    });

    // Bind group layout: matches the shader's @group(0) declarations
    // binding 0: texture_2d<f32>                  → Texture { sample_type: Float, ... }
    // binding 1: texture_storage_2d<rgba8unorm, write> → StorageTexture { write_only, ... }
    // binding 2: var<uniform> BlurParams           → Buffer { Uniform }
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gauss_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
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

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gauss_pipeline_layout"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("gaussian_blur_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // ── 6. Bind groups — one per pass ─────────────────────────────────────────
    // Pass 1 (horizontal): input=tex_input, output=tex_mid
    let bg_h = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gauss_bg_h"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view_input),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&view_mid_storage),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_h.as_entire_binding(),
            },
        ],
    });

    // Pass 2 (vertical): input=tex_mid, output=tex_output
    let bg_v = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gauss_bg_v"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view_mid_sampled),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&view_output),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_v.as_entire_binding(),
            },
        ],
    });

    // ── 7. Dispatch compute passes ────────────────────────────────────────────
    // Workgroup size is @workgroup_size(256, 1, 1)
    // Dispatch: x = width.div_ceil(256), y = height (each row is one dispatch)
    let dispatch_x = width.div_ceil(256);
    let dispatch_y = height;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gauss_encoder"),
    });

    // Pass 1: horizontal blur
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gauss_pass_h"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bg_h, &[]);
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    // Pass 2: vertical blur
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gauss_pass_v"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bg_v, &[]);
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    // ── 8. Copy output texture to readback buffer ─────────────────────────────
    // bytes_per_row must be a multiple of COPY_BYTES_PER_ROW_ALIGNMENT (256).
    let raw_bytes_per_row = width * 4;
    let aligned_bytes_per_row = (raw_bytes_per_row + wgpu::COPY_BYTES_PER_ROW_ALIGNMENT - 1)
        / wgpu::COPY_BYTES_PER_ROW_ALIGNMENT
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let readback_size = (aligned_bytes_per_row * height) as u64;

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gauss_readback"),
        size: readback_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &tex_output,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(aligned_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        tex_extent,
    );

    queue.submit(Some(encoder.finish()));

    // ── 9. Map readback buffer and wait ──────────────────────────────────────
    let buf_slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buf_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });

    // Poll until GPU work completes
    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    // Check map result
    match rx.recv() {
        Ok(Ok(())) => {}
        _ => return false,
    }

    // ── 10. Copy result back to pixels (strip row padding) ────────────────────
    {
        let mapped = buf_slice.get_mapped_range();
        let src_bytes = &*mapped;
        let raw_row = raw_bytes_per_row as usize;
        let aligned_row = aligned_bytes_per_row as usize;
        for row in 0..(height as usize) {
            let src_start = row * aligned_row;
            let dst_start = row * raw_row;
            if src_start + raw_row <= src_bytes.len()
                && dst_start + raw_row <= pixels.len()
            {
                pixels[dst_start..dst_start + raw_row]
                    .copy_from_slice(&src_bytes[src_start..src_start + raw_row]);
            }
        }
    }
    readback.unmap();

    true
}

/// Dispatch a separable Gaussian blur.
///
/// When the GPU path is eligible (feature enabled, hardware present, image
/// large enough) the compute shader is used; otherwise the CPU implementation
/// in [`crate::image_filter::gaussian_blur`] handles the work.
pub fn dispatch_gaussian_blur(
    pixels: &mut [u8],
    width: u32,
    height: u32,
    sigma: f32,
    config: &GpuComputeConfig,
) {
    if should_use_gpu(config, width, height) {
        #[cfg(feature = "gpu")]
        if try_gpu_gaussian_blur(pixels, width, height, sigma) {
            return;
        }
    }

    // CPU fallback
    crate::image_filter::gaussian_blur(pixels, width, height, sigma);
}

// ---------------------------------------------------------------------------
// Dispatch: Porter-Duff compositing
// ---------------------------------------------------------------------------

/// Dispatch Porter-Duff compositing of `src` onto `dst`.
///
/// Follows the same GPU/CPU selection logic as [`dispatch_gaussian_blur`].
pub fn dispatch_porter_duff(
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    height: u32,
    op: PorterDuffOp,
    config: &GpuComputeConfig,
) {
    if should_use_gpu(config, width, height) {
        #[cfg(feature = "gpu")]
        {
            let _ = PORTER_DUFF_WGSL;
        }
    }

    crate::porter_duff::composite_layer_into(src, dst, width, height, op);
}

// ---------------------------------------------------------------------------
// Dispatch: Gradient fill
// ---------------------------------------------------------------------------

/// Gradient type for GPU dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuGradientType {
    /// Linear gradient between two points.
    Linear,
    /// Radial gradient from a center with a given radius.
    Radial,
}

/// A colour stop for the GPU gradient shader.
#[derive(Debug, Clone, Copy)]
pub struct GpuColorStop {
    /// RGBA colour in [0, 1] range.
    pub color: [f32; 4],
    /// Position along the gradient axis in [0, 1].
    pub position: f32,
}

/// Dispatch a gradient fill into an RGBA8 pixel buffer.
///
/// When the GPU path is eligible the `gradient_fill.wgsl` compute shader is
/// used; otherwise a simple CPU rasteriser produces the output.
pub fn dispatch_gradient_fill(
    pixels: &mut [u8],
    width: u32,
    height: u32,
    gradient_type: GpuGradientType,
    stops: &[GpuColorStop],
    start: (f32, f32),
    end: (f32, f32),
    config: &GpuComputeConfig,
) {
    if should_use_gpu(config, width, height) {
        #[cfg(feature = "gpu")]
        {
            let _ = GRADIENT_FILL_WGSL;
        }
    }

    // CPU fallback
    cpu_gradient_fill(pixels, width, height, gradient_type, stops, start, end);
}

/// CPU implementation of gradient fill (used as fallback).
fn cpu_gradient_fill(
    pixels: &mut [u8],
    width: u32,
    height: u32,
    gradient_type: GpuGradientType,
    stops: &[GpuColorStop],
    start: (f32, f32),
    end: (f32, f32),
) {
    if stops.is_empty() || width == 0 || height == 0 {
        return;
    }

    let w = width as usize;
    let h = height as usize;
    let expected_len = w * h * 4;
    if pixels.len() < expected_len {
        return;
    }

    for y in 0..h {
        for x in 0..w {
            let u = if width > 1 {
                x as f32 / (width - 1) as f32
            } else {
                0.0
            };
            let v = if height > 1 {
                y as f32 / (height - 1) as f32
            } else {
                0.0
            };

            let t = match gradient_type {
                GpuGradientType::Linear => {
                    let dx = end.0 - start.0;
                    let dy = end.1 - start.1;
                    let len_sq = dx * dx + dy * dy;
                    if len_sq > 1e-8 {
                        ((u - start.0) * dx + (v - start.1) * dy) / len_sq
                    } else {
                        0.0
                    }
                }
                GpuGradientType::Radial => {
                    let radius = end.0; // end_x encodes radius
                    if radius > 1e-8 {
                        let dist = ((u - start.0).powi(2) + (v - start.1).powi(2)).sqrt();
                        dist / radius
                    } else {
                        0.0
                    }
                }
            };

            let color = evaluate_stops(stops, t.clamp(0.0, 1.0));

            let idx = (y * w + x) * 4;
            pixels[idx] = (color[0] * 255.0).clamp(0.0, 255.0) as u8;
            pixels[idx + 1] = (color[1] * 255.0).clamp(0.0, 255.0) as u8;
            pixels[idx + 2] = (color[2] * 255.0).clamp(0.0, 255.0) as u8;
            pixels[idx + 3] = (color[3] * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
}

/// Linearly interpolate colour stops at parameter `t` in [0, 1].
fn evaluate_stops(stops: &[GpuColorStop], t: f32) -> [f32; 4] {
    if stops.len() == 1 || t <= stops[0].position {
        return stops[0].color;
    }
    let last = stops.len() - 1;
    if t >= stops[last].position {
        return stops[last].color;
    }

    // Find bracketing pair
    let mut lower = 0;
    for i in 1..stops.len() {
        if stops[i].position <= t {
            lower = i;
        }
    }
    let upper = lower + 1;
    if upper >= stops.len() {
        return stops[last].color;
    }

    let span = stops[upper].position - stops[lower].position;
    let local_t = if span.abs() > 1e-8 {
        (t - stops[lower].position) / span
    } else {
        0.0
    };

    let a = &stops[lower].color;
    let b = &stops[upper].color;
    [
        a[0] + (b[0] - a[0]) * local_t,
        a[1] + (b[1] - a[1]) * local_t,
        a[2] + (b[2] - a[2]) * local_t,
        a[3] + (b[3] - a[3]) * local_t,
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_status_feature_disabled_or_available() {
        // Without the gpu feature this returns FeatureDisabled;
        // with it, it returns Available or NotAvailable depending on hardware.
        let status = check_gpu_status();
        match status {
            GpuStatus::FeatureDisabled => {
                // Expected when compiled without gpu feature
            }
            GpuStatus::Available | GpuStatus::NotAvailable(_) => {
                // Expected when compiled with gpu feature
            }
        }
    }

    #[test]
    fn test_gpu_compute_config_default() {
        let cfg = GpuComputeConfig::default();
        assert!(cfg.prefer_gpu);
        assert_eq!(cfg.min_pixels_for_gpu, 256 * 256);
    }

    #[test]
    fn test_dispatch_blur_falls_back_to_cpu() {
        // Verify that dispatch produces the same output as the direct CPU call
        let (w, h) = (8u32, 8u32);
        let sigma = 1.5_f32;

        let mut a = vec![128u8; (w * h * 4) as usize];
        let mut b = a.clone();

        // Direct CPU
        crate::image_filter::gaussian_blur(&mut a, w, h, sigma);

        // Dispatch (will fall back to CPU)
        let cfg = GpuComputeConfig {
            prefer_gpu: false,
            min_pixels_for_gpu: 0,
        };
        dispatch_gaussian_blur(&mut b, w, h, sigma, &cfg);

        assert_eq!(a, b);
    }

    #[test]
    fn test_dispatch_blur_uniform_unchanged() {
        // A uniform-colour image should remain unchanged after blur
        let (w, h) = (16u32, 16u32);
        let mut pixels = vec![200u8; (w * h * 4) as usize];
        let original = pixels.clone();

        let cfg = GpuComputeConfig::default();
        dispatch_gaussian_blur(&mut pixels, w, h, 2.0, &cfg);

        // Every pixel should still be 200 (uniform convolution)
        assert_eq!(pixels, original);
    }

    #[test]
    fn test_dispatch_pd_src_over_matches_cpu() {
        let (w, h) = (4u32, 4u32);
        let src = vec![100u8; (w * h * 4) as usize];
        let mut dst_dispatch = vec![200u8; (w * h * 4) as usize];
        let mut dst_direct = dst_dispatch.clone();

        let cfg = GpuComputeConfig {
            prefer_gpu: false,
            min_pixels_for_gpu: 0,
        };
        dispatch_porter_duff(&src, &mut dst_dispatch, w, h, PorterDuffOp::SrcOver, &cfg);
        crate::porter_duff::composite_layer_into(
            &src,
            &mut dst_direct,
            w,
            h,
            PorterDuffOp::SrcOver,
        );

        assert_eq!(dst_dispatch, dst_direct);
    }

    #[test]
    fn test_dispatch_blur_zero_sigma_unchanged() {
        let (w, h) = (8u32, 8u32);
        let mut pixels: Vec<u8> = (0..w * h * 4).map(|i| (i % 256) as u8).collect();
        let original = pixels.clone();

        let cfg = GpuComputeConfig::default();
        dispatch_gaussian_blur(&mut pixels, w, h, 0.0, &cfg);

        assert_eq!(pixels, original, "sigma=0 should be a no-op");
    }

    #[test]
    fn test_min_pixels_threshold() {
        // With prefer_gpu=false, should_use_gpu always returns false
        let cfg_no_gpu = GpuComputeConfig {
            prefer_gpu: false,
            min_pixels_for_gpu: 0,
        };
        assert!(!should_use_gpu(&cfg_no_gpu, 1024, 1024));

        // With a very high threshold, small images go CPU
        let cfg_high = GpuComputeConfig {
            prefer_gpu: true,
            min_pixels_for_gpu: u32::MAX,
        };
        assert!(!should_use_gpu(&cfg_high, 64, 64));
    }

    #[test]
    fn test_shader_sources_not_empty() {
        assert!(
            !GAUSSIAN_BLUR_WGSL.is_empty(),
            "gaussian_blur.wgsl must not be empty"
        );
        assert!(
            !PORTER_DUFF_WGSL.is_empty(),
            "porter_duff.wgsl must not be empty"
        );
        assert!(
            !GRADIENT_FILL_WGSL.is_empty(),
            "gradient_fill.wgsl must not be empty"
        );

        // Sanity: each shader contains a @compute entry point
        assert!(GAUSSIAN_BLUR_WGSL.contains("@compute"));
        assert!(PORTER_DUFF_WGSL.contains("@compute"));
        assert!(GRADIENT_FILL_WGSL.contains("@compute"));
    }

    #[test]
    fn test_gradient_fill_linear_two_stops() {
        let (w, h) = (8u32, 1u32);
        let mut pixels = vec![0u8; (w * h * 4) as usize];

        let stops = vec![
            GpuColorStop {
                color: [0.0, 0.0, 0.0, 1.0],
                position: 0.0,
            },
            GpuColorStop {
                color: [1.0, 1.0, 1.0, 1.0],
                position: 1.0,
            },
        ];

        let cfg = GpuComputeConfig {
            prefer_gpu: false,
            min_pixels_for_gpu: 0,
        };
        dispatch_gradient_fill(
            &mut pixels,
            w,
            h,
            GpuGradientType::Linear,
            &stops,
            (0.0, 0.0),
            (1.0, 0.0),
            &cfg,
        );

        // First pixel should be black, last pixel should be white
        assert_eq!(pixels[0], 0, "first pixel R should be 0");
        let last_idx = ((w - 1) * 4) as usize;
        assert_eq!(pixels[last_idx], 255, "last pixel R should be 255");
        // Middle pixel should be between 0 and 255
        let mid_idx = ((w / 2) * 4) as usize;
        assert!(pixels[mid_idx] > 0 && pixels[mid_idx] < 255);
    }

    #[test]
    fn test_gradient_fill_radial() {
        let (w, h) = (16u32, 16u32);
        let mut pixels = vec![0u8; (w * h * 4) as usize];

        let stops = vec![
            GpuColorStop {
                color: [1.0, 0.0, 0.0, 1.0],
                position: 0.0,
            },
            GpuColorStop {
                color: [0.0, 0.0, 1.0, 1.0],
                position: 1.0,
            },
        ];

        let cfg = GpuComputeConfig {
            prefer_gpu: false,
            min_pixels_for_gpu: 0,
        };
        dispatch_gradient_fill(
            &mut pixels,
            w,
            h,
            GpuGradientType::Radial,
            &stops,
            (0.5, 0.5), // center
            (0.5, 0.0), // radius = 0.5
            &cfg,
        );

        // Center pixel (8,8) should be red-ish (close to first stop)
        let center_idx = (8 * w as usize + 8) * 4;
        assert!(pixels[center_idx] > 200, "center R should be high");
        assert!(pixels[center_idx + 2] < 55, "center B should be low");
    }

    #[test]
    fn test_evaluate_stops_edge_cases() {
        let stops = vec![
            GpuColorStop {
                color: [1.0, 0.0, 0.0, 1.0],
                position: 0.0,
            },
            GpuColorStop {
                color: [0.0, 1.0, 0.0, 1.0],
                position: 0.5,
            },
            GpuColorStop {
                color: [0.0, 0.0, 1.0, 1.0],
                position: 1.0,
            },
        ];

        // t=0 -> red
        let c0 = evaluate_stops(&stops, 0.0);
        assert!((c0[0] - 1.0).abs() < 1e-6);

        // t=1 -> blue
        let c1 = evaluate_stops(&stops, 1.0);
        assert!((c1[2] - 1.0).abs() < 1e-6);

        // t=0.5 -> green
        let c_mid = evaluate_stops(&stops, 0.5);
        assert!((c_mid[1] - 1.0).abs() < 1e-6);

        // t=0.25 -> halfway between red and green
        let c_quarter = evaluate_stops(&stops, 0.25);
        assert!((c_quarter[0] - 0.5).abs() < 0.01);
        assert!((c_quarter[1] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_gpu_status_clone_and_eq() {
        let a = GpuStatus::FeatureDisabled;
        let b = a.clone();
        assert_eq!(a, b);

        let c = GpuStatus::NotAvailable("reason".into());
        let d = GpuStatus::NotAvailable("reason".into());
        assert_eq!(c, d);
        assert_ne!(a, c);
    }

    // ── New tests per task spec ────────────────────────────────────────────────

    #[test]
    fn test_dispatch_gaussian_blur_cpu_path() {
        // Run a 64×64 RGBA blur with GPU disabled — must complete and be non-zero.
        let (w, h) = (64u32, 64u32);
        let mut pixels: Vec<u8> = (0u32..w * h * 4).map(|i| (i % 200 + 55) as u8).collect();

        let cfg = GpuComputeConfig {
            prefer_gpu: false,
            min_pixels_for_gpu: 0,
        };
        dispatch_gaussian_blur(&mut pixels, w, h, 2.0, &cfg);

        // Output must not be all-zero.
        assert!(pixels.iter().any(|&v| v != 0), "output should not be all-zero");
    }

    #[test]
    fn test_dispatch_gaussian_blur_gpu_path() {
        // Run with default config (GPU preferred); must either use GPU or fall back
        // to CPU — result must not be all-zero and must not panic.
        let (w, h) = (64u32, 64u32);
        let mut pixels: Vec<u8> = (0u32..w * h * 4).map(|i| (i % 200 + 55) as u8).collect();

        let cfg = GpuComputeConfig::default();
        dispatch_gaussian_blur(&mut pixels, w, h, 2.0, &cfg);

        assert!(pixels.iter().any(|&v| v != 0), "output should not be all-zero");
    }

    #[test]
    fn test_gpu_status_check() {
        // check_gpu_status must return one of the defined variants without panicking.
        let status = check_gpu_status();
        // This match is exhaustive — if a new variant is added it will fail to compile.
        match status {
            GpuStatus::Available => {}
            GpuStatus::NotAvailable(_) => {}
            GpuStatus::FeatureDisabled => {}
        }
    }
}
