# oximedia-quality TODO

## Current Status
- 36 modules for video quality assessment and objective metrics
- Full-reference metrics: PSNR, SSIM, MS-SSIM, VMAF, VIF, FSIM
- No-reference metrics: NIQE, BRISQUE, Blockiness, Blur, Noise
- Advanced modules: aggregate_score, artifact_score, bitrate_quality, codec_quality, color_fidelity, compression_artifacts, histogram_quality, metrics, perceptual, perceptual_model, quality_gate, quality_preset, quality_report, reference_free, scene_quality, sharpness_score, spatial_quality, temporal_quality, vmaf_score
- Key types: QualityAssessor, Frame, MetricType, QualityScore, PoolingMethod, BatchAssessment
- Dependencies: oximedia-core, ndarray, rayon, serde

## Enhancements
- [ ] Replace `unreachable!()` in `QualityAssessor::assess_no_reference` with proper error return
- [x] Add per-region quality assessment in SSIM/PSNR (compute metrics for specific frame regions)
- [ ] Implement configurable VMAF model selection in `VmafCalculator` (phone model, 4K model)
- [ ] Extend `temporal_quality` with scene-aware temporal pooling (reset stats at scene cuts)
- [ ] Add chroma plane quality assessment in SSIM (currently luma-only in many implementations)
- [ ] Implement quality metric confidence intervals based on frame count
- [x] Extend `quality_gate` with multi-metric composite gates (pass only if SSIM > X AND VMAF > Y)
- [x] Add `quality_report` export to CSV format for spreadsheet analysis

## New Features
- [ ] Implement LPIPS (Learned Perceptual Image Patch Similarity) metric using pre-trained weights
- [ ] Add video quality assessment for HDR content (PQ/HLG-aware metrics)
- [ ] Implement real-time quality monitoring (running average with configurable window)
- [ ] Add A/B quality comparison tool — rank multiple encodes by perceptual quality
- [ ] Implement quality-bitrate curve generation (encode at multiple CRF values, plot VMAF vs bitrate)
- [ ] Add motion-compensated temporal quality analysis (account for frame motion in temporal metrics)
- [ ] Implement CIEDE2000 color difference metric for color fidelity assessment
- [ ] Add quality heatmap generation — spatial map of per-pixel quality for visualization

## Performance
- [ ] Implement SIMD-optimized SSIM computation using portable_simd or manual intrinsics
- [ ] Add GPU-accelerated PSNR/SSIM computation via compute shaders
- [ ] Parallelize `BatchAssessment` across frames using rayon with configurable thread count
- [ ] Optimize `VifCalculator` steerable pyramid decomposition with FFT-based filtering
- [ ] Cache Gaussian kernel weights in SSIM calculator to avoid recomputation per frame
- [ ] Use pre-allocated buffers in `Frame` operations to reduce allocation in tight loops

## Testing
- [ ] Add golden reference tests — compare metric outputs against known-correct values from published papers
- [ ] Test metric monotonicity — verify PSNR/SSIM increase as distortion decreases
- [ ] Add tests for 10-bit and 12-bit Frame data (HDR content)
- [ ] Test `PoolingMethod` with edge cases (single frame, all-identical scores, NaN values)
- [ ] Benchmark quality metrics against reference implementations (compare speed and accuracy)

## Documentation
- [ ] Add metric interpretation guide — what PSNR/SSIM/VMAF scores mean in practice
- [ ] Document quality assessment workflow from raw frames to aggregated report
- [ ] Add visual examples showing metric behavior on different distortion types (blur, blocking, noise)
