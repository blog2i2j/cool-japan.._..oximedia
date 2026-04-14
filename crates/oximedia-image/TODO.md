# oximedia-image TODO

## Current Status
- 40+ modules for professional image sequence I/O and processing
- Formats: DPX (SMPTE 268M), OpenEXR, TIFF/BigTIFF, DNG (demosaic, parser, reader, writer)
- Image processing: convolution, edge_detect, morphology, advanced_morphology, blend_mode, color_adjust, color_balance, color_science, segmentation, inpaint/inpainting, stitch, texture_synthesis, mosaic
- Frequency domain: frequency_domain, image_pyramid, pyramid
- Noise: noise_estimation, noise_gen
- Color/tone: tone_curve, histogram_ops, depth_map, lens_correct
- Pipeline: pixel_pipeline, filter, filters, format_detect, sequence (SequencePattern), thumbnail_cache
- Metadata: exif_parser, metadata_xmp, icc_embed, raw, raw_decode
- Pixel types: U8, U10, U12, U16, U32, F16, F32; 9 color spaces; 12 compression methods
- Dependencies: oximedia-core, byteorder, half, tiff, flate2, oxiarc-deflate, weezl, rayon

## Enhancements
- [ ] Add multi-layer/multi-part support to `exr.rs` for compositing workflows (read/write individual layers)
- [ ] Extend `dpx.rs` with 10-bit packed pixel read/write (Method A and Method B packing)
- [ ] Implement full TIFF tag roundtrip in `tiff.rs` (preserve unknown tags during read-modify-write)
- [ ] Add adaptive thresholding modes (Otsu, triangle) to `segmentation.rs`
- [ ] Extend `convolution.rs` with separable kernel optimization (apply 1D horizontal then vertical)
- [x] Add bilateral filter implementation to `filters.rs` for edge-preserving denoising
- [x] Implement guided filter in `inpainting.rs` for structure-aware inpainting
- [x] Add sub-pixel alignment to `stitch.rs` for higher-accuracy panorama stitching

## New Features
- [ ] Add a `webp.rs` module for WebP image decode/encode (lossy and lossless)
- [ ] Add a `png.rs` module for PNG image decode/encode with APNG animation support
- [ ] Implement a `jpeg.rs` module for JPEG baseline decode/encode
- [ ] Add a `heif.rs` module for HEIF/AVIF still image container support
- [ ] Implement a `focus_stack.rs` module for focus stacking of image sequences
- [ ] Add a `hdr_bracket.rs` module for exposure bracketing and Debevec HDR merge
- [ ] Implement a `content_aware_resize.rs` module (seam carving) for intelligent resizing
- [ ] Add a `super_resolution.rs` module for image upscaling beyond simple interpolation
- [ ] Implement `color_lut.rs` for applying 3D LUTs directly to ImageData (integrate with oximedia-lut)

## Performance
- [ ] Add SIMD-accelerated pixel format conversion in `pixel_pipeline.rs` (U8<->F32 batch)
- [ ] Implement tiled parallel processing in `convolution.rs` for large images
- [ ] Add memory-mapped I/O path for DPX/EXR sequence reading to reduce allocation overhead
- [ ] Optimize `morphology.rs` dilation/erosion with sliding-window max/min algorithms
- [ ] Add GPU-accelerated path for `frequency_domain.rs` FFT operations
- [ ] Parallelize `stitch.rs` feature matching across image pairs with rayon

## Testing
- [ ] Add round-trip tests for DPX read/write at all supported bit depths (8, 10, 12, 16)
- [ ] Test EXR compression round-trips (None, Zip, Piz, B44) with tolerance checks
- [ ] Add DNG demosaic quality tests comparing against reference implementations
- [ ] Test `edge_detect.rs` Sobel/Canny operators against known edge maps
- [ ] Add `blend_mode.rs` tests verifying each mode against Photoshop/GIMP reference outputs
- [ ] Test `sequence.rs` pattern matching with edge cases (missing frames, non-contiguous ranges)

## Documentation
- [ ] Document supported DPX header fields and their mapping to ImageData metadata
- [ ] Add pixel format conversion matrix showing supported PixelType conversions
- [ ] Document the color science pipeline (color_science -> color_adjust -> color_balance)
