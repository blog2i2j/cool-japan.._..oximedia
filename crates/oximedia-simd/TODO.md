# oximedia-simd TODO

## Current Status
- 35 source files with scalar fallbacks, conditional x86 (AVX2/AVX-512) and aarch64 (NEON) assembly backends
- 29 public modules: accumulator, alpha_premul, audio_ops, avx512, bitwise_ops, blend, blend_simd, color_convert_simd, color_space, convolution, dispatch, filter, fixed_point, gather_scatter, histogram, interleave, lookup_table, math_ops, matrix, min_max, pack_unpack, pixel_ops, prefix_sum, reduce, saturate, threshold, transpose, vector_math, yuv_ops
- Core APIs: forward_dct/inverse_dct, interpolate, sad, CPU feature detection with OnceLock caching
- Features: native-asm (cc build dep), runtime-dispatch, force-avx2, avx512, force-avx512, force-neon

## Enhancements
- [x] Add SATD (Sum of Absolute Transformed Differences) kernel alongside existing SAD for better motion estimation quality
- [x] Extend `DctSize` enum to include Dct64x64 for AV1 large block transforms
- [x] Add Hadamard transform support in `scalar` and platform-specific backends
- [x] Implement `InterpolationFilter::Lanczos` variant for higher-quality resampling
- [x] Add `BlockSize::Block8x8` and `Block4x4` to SAD operations for sub-block motion search (via sad_subblock.rs)
- [x] Extend `color_convert_simd` to support BT.2020 and BT.2100 HDR color matrices
- [x] Add runtime benchmark function that measures throughput for each detected SIMD tier

## New Features
- [x] Implement SSIM (Structural Similarity Index) kernel with SIMD acceleration
- [x] Add PSNR computation kernel operating on 16x16 and 32x32 blocks
- [x] Implement variance/standard deviation computation in `reduce` module for adaptive quantization support
- [x] Add `deblock_filter` module with SIMD-accelerated deblocking for codec post-processing
- [x] Implement `motion_search` module with diamond/hexagonal search patterns using SAD kernels
- [x] Add `resize` module with SIMD-accelerated bilinear and Lanczos image scaling
- [x] Implement `entropy_coding` SIMD helpers for bit packing/unpacking in codec bitstreams
- [x] Add Apple Silicon AMX (Apple Matrix Extension) detection in `CpuFeatures` for future acceleration paths

## Performance
- [x] Implement AVX-512 VNNI (Vector Neural Network Instructions) path for 8-bit SAD accumulation
      → `avx512::sad_8x8_vnni` and `sad_4x4_vnni` using `_mm512_dpbusd_epi32`; scalar fallback included
- [x] Add prefetch hints in `interpolate` for sequential block access patterns
      → `_mm_prefetch` with `_MM_HINT_T0` in bilinear and 8-tap inner loops; `#[cfg(target_arch = "x86_64")]` guarded
- [x] Implement 4-way parallel SAD search in AVX-512 path (process 4 candidate blocks simultaneously)
      → `avx512::sad_parallel_4way_avx512` using `_mm_sad_epu8` per candidate; scalar fallback included
- [x] Use `_mm256_maddubs_epi16` intrinsic in AVX2 path for faster u8 multiplication in interpolation
      → `x86::interpolate_8tap_avx2` with `dot8_maddubs` inner kernel; scalar fallback included
- [x] Add aligned allocation helper that guarantees 64-byte alignment for AVX-512 buffers
- [x] Profile and optimize `scalar::forward_dct_scalar` with butterfly decomposition for Dct32x32
      → `dct_butterfly.rs` implements the optimised butterfly decomposition

## Testing
- [x] Add correctness tests comparing SIMD outputs against scalar reference for all DCT sizes
      → `scalar_equivalence.rs`: forward 4x4/8x8/16x16/32x32, inverse 4x4/8x8/16x16/32x32 (32x32 added)
- [x] Add cross-platform CI tests ensuring scalar fallback produces identical results to SIMD paths
- [x] Test `CpuFeatures::detect` on actual x86_64 and aarch64 targets (not just compilation)
- [x] Add fuzzing targets for `interpolate` and `sad` with random buffer sizes near boundary conditions
- [x] Add criterion benchmarks for each kernel at each SIMD tier (scalar, SSE4.2, AVX2, AVX-512, NEON)
      → `benches/simd_benchmarks.rs`: SATD, SSIM, PSNR, deblock, motion_search, resize, entropy_coding, hadamard

## Documentation
- [x] Document SIMD tier selection logic and how `runtime-dispatch` feature interacts with `force-*` features
- [x] Add performance comparison table (scalar vs. AVX2 vs. AVX-512 vs. NEON) for core kernels
- [x] Document alignment requirements for each SIMD tier in module-level docs
