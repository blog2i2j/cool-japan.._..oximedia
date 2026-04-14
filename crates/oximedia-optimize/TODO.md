# oximedia-optimize TODO

## Current Status
- 43 source files/directories covering codec optimization and encoding tuning
- Core: Optimizer with RdoEngine, PsychoAnalyzer, MotionOptimizer, AqEngine
- RDO: rate-distortion optimization, lambda calculation, RDOQ (rate-distortion optimized quantization)
- Psychovisual: contrast sensitivity, visual masking models, perceptual optimization
- Motion: TZSearch, EPZS, UMH algorithms, bidirectional optimization, subpel refinement, MV prediction
- Encoding tools: partition selection, intra mode optimization, transform selection, loop filter tuning
- Presets: Fast/Medium/Slow/Placebo levels, content-type adaptation (Animation/Film/Screen/Generic)
- Advanced: lookahead analysis, GOP optimizer, two-pass encoding, CRF sweep, adaptive ladder, bitrate control

## Enhancements
- [ ] Implement vmaf_predict module -- currently a file exists but needs actual VMAF score prediction from features
- [ ] Wire roi_encode into the main Optimizer pipeline for region-of-interest based quality allocation
- [ ] Connect temporal_aq to the AqEngine for frame-level QP adaptation based on temporal complexity
- [ ] Improve scene_encode to use lookahead data for scene-cut-aware QP adjustment
- [ ] Add content-adaptive GOP structure selection in gop_optimizer (longer GOPs for static, shorter for action)
- [ ] Implement actual CABAC context optimization in entropy module (currently may be structural)
- [ ] Extend crf_sweep to output Pareto-optimal bitrate/quality curves for automated quality targeting
- [ ] Add grain synthesis detection in psycho module to preserve film grain without wasting bits

## New Features
- [x] Implement AV1 tile/frame parallel optimization -- select tile partitioning based on content complexity
- [x] Add per-frame bitrate allocation using Viterbi algorithm for optimal constant-quality distribution
- [ ] Implement machine-learning-based mode decision using oximedia-neural for fast RDO approximation
- [ ] Add encoding quality estimation without full encode (fast VMAF/SSIM prediction from frame features)
- [x] Implement denoising-aware optimization -- coordinate with pre-filter to avoid encoding noise
- [ ] Add HDR-aware psychovisual model using PQ/HLG transfer function aware masking thresholds
- [ ] Implement multi-pass encoding with rate redistribution between passes in two_pass module

## Performance
- [ ] Use rayon for parallel RDO evaluation across partition candidates (when parallel_rdo=true)
- [ ] Implement SIMD-accelerated SAD/SATD computation in motion search (currently scalar)
- [ ] Add block-level caching in RdoEngine -- cache cost estimates for repeated partition evaluations
- [ ] Implement early termination in partition search when cost clearly exceeds parent split
- [ ] Profile cache_optimizer and prefetch modules -- ensure they actually improve memory access patterns
- [ ] Use thread-local storage for per-thread scratch buffers in parallel motion search

## Testing
- [ ] Add RDO cost monotonicity test: higher lambda should prefer lower-rate modes
- [ ] Test motion search accuracy: known displacement input should find correct motion vector
- [ ] Verify AQ produces higher QP for flat regions and lower QP for detailed regions
- [ ] Test partition decision against brute-force: verify faster presets don't miss optimal split by >5% RD cost
- [ ] Add benchmark comparing Fast/Medium/Slow/Placebo encoding speed and quality metrics
- [ ] Test psychovisual masking: verify high-texture regions tolerate more quantization noise

## Documentation
- [ ] Document the optimization level trade-offs with encoding speed and quality impact measurements
- [ ] Add guide for tuning custom OptimizerConfig for specific content types
- [ ] Document the RDO pipeline with mathematical formulation of cost = distortion + lambda * rate
