# oximedia-scaling TODO

## Current Status
- 30 modules covering bilinear, bicubic, Lanczos filtering, EWA resampling, content-aware scaling, adaptive scaling, aspect ratio preservation, batch scaling, chroma scaling, crop/pad, deinterlace, field scaling, half-pixel interpolation, perceptual sharpening, quality metrics, resolution ladders, ROI scaling, super resolution, thumbnails, tile-based processing
- Core types: ScalingParams, ScalingMode (Bilinear/Bicubic/Lanczos), AspectRatioMode, VideoScaler
- Dependencies: oximedia-core, rayon, serde, thiserror

## Enhancements
- [ ] Add `ScalingMode::NearestNeighbor` for pixel-art and retro content scaling without interpolation
- [ ] Extend `VideoScaler::calculate_dimensions` to handle non-square pixel aspect ratios (PAR correction)
- [ ] Add configurable Lanczos window size (2/3/4/5 taps) in `lanczos` module
- [ ] Implement `content_aware_scale` seam carving with forward energy for better seam selection
- [ ] Extend `resolution_ladder` with per-title encoding optimization using VIF/SSIM target thresholds
- [x] Add `chroma_scale` support for 4:2:0, 4:2:2, and 4:4:4 chroma subsampling with proper phase alignment
- [x] Implement `deinterlace` with motion-adaptive algorithm (bob for motion areas, weave for static)
- [x] Add `batch_scale` progress reporting callback and cancellation token support

## New Features
- [ ] Add `neural_upscale` module with lightweight inference for 2x/4x upscaling (patent-free architecture)
- [ ] Implement `hdr_scaling` module handling PQ/HLG tone mapping during resolution changes
- [ ] Add `film_grain_scale` module that removes grain before scaling and re-synthesizes at target resolution
- [ ] Implement `temporal_scaling` module for frame rate conversion (frame blending, motion-compensated interpolation)
- [ ] Add `watermark_safe_scale` that preserves watermark positions during scaling operations
- [ ] Implement `multi_pass_scale` for extreme scaling ratios (e.g., 8K->480p) with intermediate steps
- [ ] Add `scale_preview` for generating fast low-quality previews before committing to full-quality scale
- [ ] Implement `edge_directed_interpolation` module (NEDI-like) for improved diagonal edge rendering

## Performance
- [ ] Add SIMD intrinsics for `bicubic` and `lanczos` horizontal/vertical filter passes
- [ ] Implement rayon parallel row processing in `VideoScaler` for multi-core scaling
- [ ] Add tile-based processing in `tile` with cache-line-friendly memory access patterns
- [ ] Optimize `ewa_resample` by precomputing filter weight tables for common scale factors
- [ ] Add in-place scaling in `resampler` to avoid intermediate buffer allocations
- [ ] Implement ring-buffer row caching in vertical filter passes to minimize memory footprint

## Testing
- [ ] Add PSNR/SSIM quality regression tests comparing scaling output against reference implementations
- [ ] Test `aspect_preserve` with edge cases: 1:1, ultrawide (32:9), portrait (9:16), anamorphic (2.39:1)
- [ ] Add roundtrip tests: scale down then up, verify quality metric degradation is within bounds
- [ ] Test `batch_scale` with mixed input resolutions and aspect ratios in a single batch
- [ ] Add fuzz tests for `ScalingParams` with zero/negative/maximum dimensions

## Documentation
- [ ] Add quality comparison guide for Bilinear vs Bicubic vs Lanczos with visual examples
- [ ] Document the `resolution_ladder` algorithm and how `ContentDifficultyScore` affects rung selection
- [ ] Add guide for choosing between `content_aware_scale` and standard scaling for different content types
- [ ] Document the EWA resampling filter selection (Mitchell, Lanczos, Catmull-Rom) trade-offs
