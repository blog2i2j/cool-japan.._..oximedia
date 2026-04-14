# oximedia-gpu

![Status: Stable](https://img.shields.io/badge/status-stable-green)
![Version: 0.1.3](https://img.shields.io/badge/version-0.1.3-blue)

Cross-platform GPU compute pipeline for OxiMedia using WGPU, supporting Vulkan, Metal, DirectX 12, and WebGPU backends.

Part of the [oximedia](https://github.com/cool-japan/oximedia) workspace — a comprehensive pure-Rust media processing framework.

## Features

- **Color Space Conversions** — RGB ↔ YUV with BT.601, BT.709, BT.2020 matrices
- **Image Scaling** — Nearest, bilinear, and bicubic interpolation on GPU
- **Convolution Filters** — Blur, sharpen, edge detection kernels
- **Transform Operations** — DCT and FFT on GPU
- **Automatic CPU Fallback** — Graceful degradation when GPU unavailable
- **Multi-GPU Support** — Select and use multiple GPU devices
- **Cross-platform** — Vulkan, Metal, DirectX 12, WebGPU via WGPU
- **Shader Cache** — Compiled shader caching for faster startup
- **Memory Pool** — Efficient GPU buffer pool management
- **Command Buffer** — Batched GPU command recording
- **Compute Pass** — Structured compute pass dispatch
- **Descriptor Sets** — Resource binding management
- **Render Pass** — GPU render pass management
- **Fence Pool** — GPU fence lifecycle management
- **Vertex Buffer** — Vertex data management
- **Sampler** — Texture sampler configuration
- **Profiling** — GPU timer, stats, and profiler
- **Video Processing** — Frame processing pipelines
- **Histogram** — GPU-accelerated histogram computation
- **Motion Detection** — GPU-accelerated motion detection
- **Texture** — Texture management and operations
- **Occupancy** — Compute occupancy analysis
- **Workgroup** — Workgroup sizing and dispatch

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oximedia-gpu = "0.1.3"
```

```rust
use oximedia_gpu::GpuContext;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = GpuContext::new()?;

    let input = vec![0u8; 1920 * 1080 * 4];
    let mut output = vec![0u8; 1920 * 1080 * 4];

    ctx.rgb_to_yuv(&input, &mut output)?;
    Ok(())
}
```

## API Overview

**Core types:**
- `GpuContext` — Main GPU context and entry point
- `GpuBuffer`, `GpuFence` — GPU resource types

**Device and backend:**
- `device` — GPU device enumeration and selection
- `backend` — Backend initialization (Vulkan/Metal/DX12/WebGPU)
- `accelerator` — High-level acceleration interface

**Buffer and memory:**
- `buffer`, `gpu_buffer` — Buffer allocation and management
- `memory`, `memory_pool` — GPU memory pool
- `vertex_buffer` — Vertex buffer management
- `buffer_copy` — Buffer copy operations
- `upload_queue` — Staging buffer upload queue

**Shader management:**
- `shader`, `shader_cache`, `shader_params` — Shader compilation and caching

**Compute pipeline:**
- `compute`, `compute_pass`, `compute_dispatch` — Compute operations
- `pipeline` — Compute pipeline configuration
- `kernels`, `kernel` — Compute kernel definitions
- `descriptor_set` — Resource descriptor binding
- `workgroup` — Workgroup configuration

**Rendering:**
- `render_pass` — GPU render pass
- `texture` — Texture management
- `sampler` — Sampler configuration
- `viewport` — Viewport configuration

**Synchronization:**
- `queue` — Command queue management
- `sync`, `sync_primitive` — Fence and semaphore synchronization
- `fence_pool` — Fence lifecycle management

**High-level operations:**
- `ops` — High-level GPU operations (color conversion, scaling)
- `video_process` — Video frame processing pipeline
- `histogram` — Histogram computation kernel
- `motion_detect` — Motion detection kernel
- `cache` — Operation result caching

**Profiling:**
- `gpu_profiler` — GPU profiling
- `gpu_timer` — GPU timing queries
- `gpu_stats` — GPU statistics collection
- `resource_manager` — GPU resource lifecycle tracking
- `occupancy` — Compute occupancy analysis

## License

Apache-2.0 — Copyright 2024-2026 COOLJAPAN OU (Team Kitasan)
