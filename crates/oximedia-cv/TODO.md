# oximedia-cv TODO

## Current Status
- 100+ source files; comprehensive computer vision and image processing
- Image processing: resize, color conversion, filtering, edge detection, histogram
- Detection: face, object, corner, YOLO, lane detection
- Transforms: affine, perspective
- Enhancement: super-resolution (ESRGAN), denoising, sharpening
- ML: ONNX Runtime integration (feature-gated), preprocessing/postprocessing, tensor ops
- Tracking: optical flow, SORT v2, LK tracker, IoU tracker, CSRT, KCF, MOSSE
- Stabilization: motion estimation, smoothing, rolling shutter correction
- Scene detection: histogram, motion, edge, adaptive, classification
- Quality: PSNR, SSIM, report generation
- Interpolation: optical flow-based frame interpolation, occlusion handling, blend/warp
- Chroma key: green screen keying, spill suppression, compositing, auto-key
- Content-aware: seam carving, saliency, energy, hybrid scaling, protection maps
- Interlace: comb detection, field separation, telecine, pattern analysis, metrics
- Motion blur: PSF estimation, synthesis, removal, deconvolution
- Fingerprint: perceptual hash, chromaprint, temporal fingerprint, matching
- Additional: morphology, contour detection, Hough transform, superpixel, segmentation, depth estimation, pose estimation, color clustering, texture analysis, feature extraction/matching, histogram backprojection

## Enhancements
- [ ] Extend `detect/yolo.rs` with YOLOv8/v9 model support and dynamic input resolution
- [ ] Add multi-scale face detection in `detect/face.rs` with rotation-invariant detection
- [ ] Improve `tracking/csrt.rs` with adaptive spatial reliability maps for occlusion handling
- [ ] Extend `stabilize/motion.rs` with gyroscope data fusion for hybrid stabilization
- [ ] Add temporal consistency to `enhance/denoising.rs` for video denoising (frame-to-frame coherence)
- [x] Extend `scene/adaptive.rs` with gradual transition detection (dissolves, wipes) — implemented in `scene/transition_detect.rs` with `detect_dissolve` + `detect_wipe`
- [ ] Improve `chroma_key/auto_key.rs` with automatic background color detection
- [x] Add sub-pixel accuracy to `feature_match.rs` for high-precision registration — `SubPixelRefiner`, `SubPixelMatch`, `HomographyEstimator` (RANSAC+DLT), `subpixel_match_quality`, `filter_subpixel_matches`

## New Features
- [x] Implement semantic segmentation (person/background) for portrait mode effects — `segmentation/person_bg.rs` with `PersonBackgroundSegmenter` + `SegmentationMask`
- [ ] Add instance segmentation support in `segmentation.rs` with mask generation
- [x] Implement video matting (alpha matte extraction) as an alternative to chroma key — `BackgroundCapture`, `AlphaMatteExtractor`, `TemporalMattingSmoother`, `MattingQualityMetrics`, `ForegroundExtractor`, `compose`
- [ ] Add text detection and recognition (OCR) module for subtitle extraction from burned-in text
- [ ] Implement style transfer pipeline using ONNX models in `ml/`
- [ ] Add panorama stitching using feature matching and homography in `registration/`
- [ ] Implement action recognition using temporal feature analysis
- [x] Add lens distortion correction (barrel/pincushion) in `transform/` — `LensDistortionCorrector`, `LensDistortionSimulator`, `DistortionMap` (pre-computed remap), `FisheyeEquidistantCorrector`, `optimal_crop_rect` in `lens_distortion.rs`

## Performance
- [x] Replace `rustfft` with OxiFFT per COOLJAPAN policy for FFT-dependent algorithms — `fingerprint/chromaprint.rs` updated; `Cargo.toml` now uses `oxifft.workspace = true`
- [x] Add SIMD acceleration for `image/filter.rs` convolution kernels (3x3, 5x5, 7x7) — `convolve_3x3_simd` added with SSE 4.1 run-time dispatch
- [ ] Implement GPU-accelerated inference path in `ml/runtime.rs` via ONNX CUDA/ROCm backends
- [ ] Optimize `scale/seam_carving.rs` with dynamic programming forward energy on GPU
- [ ] Parallelize `scene/histogram.rs` frame comparison using rayon
- [ ] Add tiled processing in `enhance/super_resolution/` to reduce memory for large images
- [ ] Optimize `optical_flow_field.rs` with pyramid-based Lucas-Kanade for real-time performance
- [ ] Cache feature descriptors in `feature_extract.rs` to avoid recomputation across frames

## Testing
- [ ] Add accuracy benchmarks for `quality/psnr.rs` and `quality/ssim.rs` against reference implementations
- [ ] Test `tracking/` modules with standard MOT benchmark sequences
- [ ] Add visual regression tests for `chroma_key/` with known green screen footage
- [ ] Test `interlace/telecine.rs` detection accuracy on 3:2 pulldown content
- [ ] Add round-trip tests for `transform/` (apply affine -> inverse affine -> verify identity)
- [ ] Test `motion_blur/removal.rs` deconvolution produces measurable PSNR improvement
- [ ] Add performance regression tests for core operations (resize, filter, edge detect)

## Documentation
- [ ] Document ML model requirements (input shapes, normalization) for each detection module
- [ ] Add visual examples of each filter/transform operation
- [ ] Document tracking algorithm selection guide (SORT vs CSRT vs KCF trade-offs)
