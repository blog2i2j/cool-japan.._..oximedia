# oximedia-gpu TODO

## Current Status
- 60+ source files across core GPU ops, compute pipeline, memory management, kernels, and synchronization
- WGPU-based cross-platform GPU acceleration (Vulkan, Metal, DX12, WebGPU)
- Operations: RGB/YUV color conversion (BT.601/709/2020), bilinear/bicubic scaling, Gaussian blur, sharpen, edge detect, DCT/IDCT
- Kernel modules: color conversion, convolution, filter, resize, reduce, transform
- Memory: allocator, pool, managed buffers, buffer copy, upload queue
- Pipeline: compute pipeline manager, dispatch, shader compiler, pipeline cache
- Sync: fences, semaphores, barriers, events, fence pool, GPU timer
- Profiling: gpu_profiler, gpu_stats, occupancy, workgroup sizing
- Video: histogram, motion detection, video frame processor, tone mapping, denoising
- CPU fallback backend available
- Dependencies: wgpu, bytemuck, rayon, pollster, parking_lot

## Enhancements
- [x] N/A — GpuContext intentionally has no Default impl; callers must use GpuContext::new() to handle the fallible init explicitly (see lib.rs lines 612-621)
- [x] Add BT.2020 and BT.2100 color space support to `ops/colorspace.rs` for HDR content (BT.2020 in ColorSpace enum; BT.2100 HDR in oximedia-hdr)
- [x] Implement Lanczos scaling filter in `ops/scale.rs` alongside bilinear and bicubic
- [x] Enhance `ops/denoise.rs` with GPU-accelerated bilateral filter and NLM denoising (bilateral filter wired to wgpu compute shader `shaders/bilateral.wgsl`; NLM is CPU-only, GPU NLM is future work)
- [x] Add configurable workgroup size auto-tuning in `workgroup.rs` based on device limits
- [x] Improve `shader_cache.rs` with disk-persistent shader cache (DiskShaderCache with SHA-256 keying and platform cache directory)
- [x] Enhance `memory_pool.rs` with defragmentation and compaction for long-running sessions
- [x] Add proper error propagation in `ops/` functions instead of internal panics (all ops return Result, no panics)
- [x] Implement `compute_dispatch.rs` indirect dispatch for data-dependent workload sizes (`indirect_dispatch.rs` — IndirectDispatch + DrawIndirectArgsBuffer)
- [x] Add async compute queue support in `queue.rs` for overlapping compute and transfer (`async_compute.rs` — AsyncComputeQueue)

## New Features
- [x] Implement GPU-accelerated AV1/VP9 motion estimation using compute shaders (`motion_estimation.rs`)
- [x] Add GPU-based SSIM/PSNR quality metric computation in `compute_kernels.rs`
- [x] Implement GPU histogram equalization and tone curve application (`histogram_equalization.rs`)
- [x] Add GPU-accelerated optical flow computation for motion interpolation (`optical_flow.rs`)
- [x] Implement GPU-based chroma subsampling (4:2:0, 4:2:2 conversion)
- [x] Add multi-GPU load balancing with automatic frame distribution (`multi_gpu.rs`)
- [x] Implement GPU-based film grain synthesis matching `oximedia-denoise` grain model (`film_grain.rs`)
- [x] Add GPU-accelerated image compositing with alpha blending and blend modes
- [x] Implement GPU-based perspective transform and lens distortion correction (`perspective_transform.rs`)

## Performance
- [x] Profile and optimize `ops/filter.rs` Gaussian blur with separable two-pass implementation
- [x] Add shared memory (workgroup local) optimization to convolution kernels
- [x] Implement double-buffered GPU command submission to overlap CPU/GPU work (`double_buffer.rs`)
- [x] Optimize `texture.rs` TexturePool with LRU eviction for bounded memory usage
- [x] Add pipeline barrier optimization in `pipeline.rs` to minimize GPU stalls
- [x] Implement sub-allocation within large buffers in `buffer_pool.rs` to reduce bind overhead
- [x] Profile `compute_pass.rs` dispatch overhead and batch small dispatches together

## Testing
- [x] Add GPU vs CPU output comparison tests for all operations (verify correctness within tolerance)
- [x] Test `ops/colorspace.rs` with known color conversion test vectors (BT.601, BT.709)
- [x] Test `ops/scale.rs` with known downscale/upscale reference images
- [x] Add memory leak tests for `memory_pool.rs` and `buffer_pool.rs` (allocate/free cycles)
- [x] Test multi-GPU device selection and fallback behavior
- [x] Add performance regression benchmarks for core operations (color convert, scale, blur)
- [x] Test `shader_cache.rs` invalidation when shader source changes

## Documentation
- [ ] Document supported GPU backends and their feature differences
- [ ] Add performance benchmarks table comparing GPU vs CPU for each operation
- [ ] Document the shader compilation and caching pipeline
