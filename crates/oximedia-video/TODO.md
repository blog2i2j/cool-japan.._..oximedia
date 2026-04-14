# oximedia-video TODO

## Current Status
- 8 source files covering professional video processing operations
- Key features: block-based motion estimation/compensation, frame rate conversion with intermediate frame generation, video deinterlacing, scene change detection, 3:2 pulldown cadence detection, perceptual video fingerprinting, temporal noise reduction
- Dependencies: oximedia-core, thiserror, rayon, serde

## Enhancements
- [ ] Extend `motion_compensation` with sub-pixel motion estimation (half-pel and quarter-pel refinement)
- [ ] Add adaptive block size selection in motion estimation (variable block sizes from 4x4 to 64x64)
- [ ] Implement bidirectional motion estimation in `motion_compensation` for B-frame-style interpolation
- [ ] Add multiple deinterlace algorithms to `deinterlace` (bob, weave, Yadif-style adaptive, EEDI-style edge-directed)
- [x] Extend `scene_detection` with adaptive threshold based on content complexity histogram
- [ ] Improve `temporal_denoise` with motion-compensated temporal filtering (MCTF) using motion vectors
- [x] Add confidence scoring to `pulldown_detect` cadence detection with frame-level accuracy reporting
- [x] Extend `video_fingerprint` with rotation/scale invariance for robust content matching

## New Features
- [ ] Implement `super_resolution` module for AI-free upscaling (Lanczos, edge-directed interpolation, NEDI)
- [ ] Add `film_grain_synthesis` module for AV1 film grain parameter estimation and generation
- [ ] Implement `field_order_detect` for automatic top-field-first vs bottom-field-first detection
- [ ] Add `cadence_convert` module for frame rate conversion (24->30, 25->30, 50->60) with proper cadence
- [ ] Implement `video_stabilization` module using motion vectors from motion_compensation
- [ ] Add `shot_boundary_classifier` that categorizes cuts, dissolves, wipes, and fades
- [ ] Implement `duplicate_frame_detect` for detecting and removing duplicate/near-duplicate frames
- [ ] Add `quality_metrics` module (PSNR, SSIM, VMAF-like perceptual metric) for video quality assessment

## Performance
- [ ] Parallelize motion search in `motion_compensation` across blocks using rayon
- [ ] Implement diamond/hexagonal search patterns in motion estimation instead of full search
- [ ] Add SIMD-optimized SAD (Sum of Absolute Differences) and SSD computation for block matching
- [ ] Use integral images for fast variance computation in `scene_detection`
- [ ] Implement hierarchical (coarse-to-fine) motion estimation for large search ranges
- [ ] Add frame-level parallelism in `temporal_denoise` for processing independent pixel columns

## Testing
- [ ] Add round-trip tests for deinterlace (interlace synthetic content, deinterlace, measure PSNR)
- [ ] Test `frame_interpolation` with known linear motion to verify interpolated frame accuracy
- [ ] Add `scene_detection` tests with synthetic scene-change sequences (hard cuts, dissolves, flashes)
- [ ] Test `pulldown_detect` with synthetic 3:2 pulldown cadence patterns and verify detection accuracy
- [ ] Benchmark `motion_compensation` against known motion vectors for standard test sequences
- [ ] Test `video_fingerprint` collision rate with large synthetic frame datasets

## Documentation
- [ ] Document motion estimation algorithm selection guide (when to use which block size/search range)
- [ ] Add deinterlace algorithm comparison table with quality/speed tradeoffs
- [ ] Document video fingerprint format and matching threshold recommendations
