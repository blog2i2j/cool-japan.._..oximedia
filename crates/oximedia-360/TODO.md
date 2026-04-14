# oximedia-360 TODO

## Current Status
- 25+ modules covering all major 360° VR video processing tasks
- 544 tests passing
- Comprehensive projection, stereo, fisheye, EAC, tiled, SIMD, stabilization, spatial audio, and metadata support
- Dependencies: `thiserror`, `rayon`

## Enhancements
- [x] Add bicubic and Lanczos sampling to `projection` alongside existing `bilinear_sample_u8` (`sampling.rs`)
- [x] Support 16-bit and f32 pixel formats (`sampling.rs`: `sample_u8`, `sample_u16`, `sample_f32`)
- [x] Add configurable blend width parameter to `DualFisheyeStitcher` (`overlap_blend_width` field in builder)
- [x] Implement exposure compensation in `DualFisheyeStitcher` (`ExposureGain`, `auto_exposure` builder option)
- [x] Add multi-resolution blending (Laplacian pyramid) to `DualFisheyeStitcher` for seamless stitching
- [x] Support rectilinear sub-region extraction from equirectangular frames (`rectilinear.rs`)
- [x] Add rotation/orientation transforms to `SphericalCoord` — yaw/pitch/roll (`orientation.rs`)
- [x] Implement `sphere_to_equirect` round-trip accuracy validation utilities (`sphere_equirect_max_roundtrip_error_rad`, `compute_psnr`, `angular_distance_rad` in `projection.rs`)
- [x] Add RGBA support to stereo frame splitting/merging (`split_stereo_frame_rgba`, `merge_stereo_frames_rgba` in `stereo.rs`)

## New Features
- [x] Add `EAC` (Equi-Angular Cubemap) projection used by YouTube/Google for more uniform sampling (`eac.rs`)
- [x] Implement octahedral projection mapping for efficient VR video compression (`octahedral.rs`)
- [x] Add viewport rendering: extract a perspective view from equirectangular at given FOV/orientation (`viewport.rs`)
- [x] Implement mesh-based projection for custom lens models (lookup table warping) (`mesh_warp.rs`)
- [x] Add Apple Spatial Video metadata support (MV-HEVC stereo pairs) (`apple_spatial.rs`)
- [x] Implement V3D box parsing for VR180 format metadata (`v3d.rs`)
- [x] Add stabilization for 360 video using gyroscope/IMU metadata integration (`stabilization.rs`)
- [x] Implement FOV-dependent quality allocation (`FovQualityAllocator` with Linear/Cosine/Gaussian fall-off in `tile_selector.rs`)

## Performance
- [ ] Add SIMD-accelerated bilinear sampling using packed f32 operations
- [x] Implement tiled cubemap conversion for better cache locality (`tiled.rs`: `equirect_to_cube_tiled`)
- [x] Add parallel scanline processing with rayon (`tiled.rs`: `equirect_to_cube_parallel`, `resample_equirect_parallel`)
- [x] Pre-compute lookup tables for fisheye-to-equirect mapping (`fisheye_lut.rs`)
- [ ] Use `f32` fast-math approximations for `sin`/`cos`/`atan2` in hot projection loops

## Testing
- [x] Add round-trip tests: equirect -> cubemap -> equirect with PSNR threshold (`projection.rs`: `equirect_cube_equirect_psnr_above_threshold`)
- [x] Add property-based tests for `sphere_to_equirect` / `equirect_to_sphere` inverse relationship (`sphere_equirect_roundtrip_error_near_zero`)
- [ ] Test `DualFisheyeStitcher` with synthetic checkerboard fisheye images
- [x] Add edge-case tests for polar singularities in equirectangular projection (`north_pole_elevation_clamped_at_half_pi`, `south_pole_elevation_clamped_at_neg_half_pi`, `pole_roundtrip_does_not_panic`, `antimeridian_roundtrip_stable`)
- [ ] Benchmark cubemap face extraction at 4K and 8K resolutions

## Documentation
- [ ] Add visual diagrams showing coordinate system conventions (theta/phi orientation)
- [ ] Document supported stereo layouts (SBS, TB, frame-sequential) with pixel layout examples
- [ ] Add end-to-end example: load equirectangular image, extract cubemap faces, write output
