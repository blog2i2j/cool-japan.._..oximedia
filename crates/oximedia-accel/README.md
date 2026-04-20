# oximedia-accel

![Status: Stable](https://img.shields.io/badge/status-stable-green)

Hardware acceleration layer for OxiMedia using Vulkan compute shaders, with automatic CPU fallback for systems without GPU support.

Part of the [oximedia](https://github.com/cool-japan/oximedia) workspace — a comprehensive pure-Rust media processing framework.

Version: 0.1.4 — 2026-04-20 — 409 tests

## Features

- Automatic GPU device enumeration and selection via Vulkan
- Efficient GPU memory allocation and buffer transfer
- Compute kernels: image scaling, color conversion, motion estimation
- Automatic CPU fallback when GPU is unavailable
- Safe Vulkan API access via vulkano
- Task graph scheduling for concurrent GPU operations
- Memory arena and pool management
- Fence timeline for GPU synchronization
- Pipeline acceleration abstractions
- Profiling and performance statistics
- Prefetch and cache management

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oximedia-accel = "0.1.4"
```

```rust
use oximedia_accel::{AccelContext, HardwareAccel, ScaleFilter};
use oximedia_core::types::PixelFormat;

fn example() -> Result<(), Box<dyn std::error::Error>> {
    // Create acceleration context (automatically selects GPU or CPU)
    let accel = AccelContext::new()?;

    // Perform image scaling
    let input = vec![0u8; 1920 * 1080 * 3];
    let output = accel.scale_image(
        &input,
        1920, 1080,
        1280, 720,
        PixelFormat::Rgb24,
        ScaleFilter::Bilinear,
    )?;
    Ok(())
}
```

## API Overview

**Core types:**
- `AccelContext` — Main entry point; selects GPU or CPU backend automatically
- `HardwareAccel` (trait) — Unified interface for GPU and CPU implementations
- `ScaleFilter` — Scaling filter variants (nearest, bilinear, bicubic)
- `AccelError`, `AccelResult` — Error types

**Backends:**
- `VulkanAccel` — Vulkan compute backend
- `CpuFallback` — Pure-CPU fallback implementation

**Modules (37 source files, 401 public items):**
- `device`, `device_caps` — GPU device management and capability detection
- `buffer`, `pool`, `memory_arena`, `memory_bandwidth` — Memory management
- `kernels`, `shaders` — Compute kernels and SPIR-V shaders
- `task_graph`, `task_scheduler`, `dispatch` — Parallel task scheduling
- `pipeline_accel` — Pipeline-level acceleration
- `fence_timeline` — GPU synchronization primitives
- `ops` — High-level compute operations
- `cache`, `prefetch` — Caching and prefetch strategies
- `accel_profile`, `accel_stats` — Profiling and statistics
- `traits` — Core trait definitions
- `error` — Error types

## License

Apache-2.0 — Copyright 2024-2026 COOLJAPAN OU (Team Kitasan)
