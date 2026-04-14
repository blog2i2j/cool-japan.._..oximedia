//! # oximedia-360
//!
//! 360° VR video processing for the OxiMedia Sovereign Media Framework.
//!
//! This crate provides equirectangular/cubemap projection conversions, stereoscopic
//! 3D frame splitting/merging, fisheye lens models with dual-fisheye stitching, and
//! Google Spatial Media v2 / ISOBMFF metadata serialisation — all in pure Rust with
//! zero C/Fortran dependencies.
//!
//! ## Modules
//!
//! - [`projection`]       — Equirectangular ↔ sphere ↔ cubemap coordinate transforms,
//!   bilinear sampling, and full equirect-to-cubemap image conversion.
//! - [`sampling`]         — High-quality resampling: bicubic (Mitchell-Netravali),
//!   Lanczos-2/3; supports u8, u16 and f32 pixel formats.
//! - [`stereo`]           — Side-by-side / top-bottom stereo frame splitting and
//!   merging, depth-based stereo synthesis, and parallax error metrics.
//! - [`fisheye`]          — Equidistant, equisolid, orthographic and stereographic
//!   fisheye projection models, plus dual-fisheye stitching with linear-alpha
//!   blending, exposure compensation and Laplacian-pyramid multi-resolution blending.
//! - [`fisheye_lut`]      — Pre-computed lookup tables for fisheye-to-equirect
//!   mapping; amortises trigonometric cost over many frames.
//! - [`eac`]              — Equi-Angular Cubemap (EAC) projection used by YouTube /
//!   Google for more uniform per-pixel solid-angle coverage.
//! - [`viewport`]         — Perspective viewport extraction from equirectangular
//!   panoramas at configurable FOV / yaw / pitch / roll.
//! - [`tiled`]            — Cache-friendly tiled cubemap conversion, parallel
//!   scanline processing (rayon), and SIMD-accelerated packed bilinear sampler.
//! - [`octahedral`]       — Octahedral projection for efficient VR video compression.
//! - [`orientation`]      — Yaw/pitch/roll orientation transforms on spherical coords.
//! - [`stabilization`]    — 360° video stabilisation via IMU/gyroscope integration.
//! - [`apple_spatial`]    — Apple Spatial Video metadata (MV-HEVC stereo pairs).
//! - [`v3d`]              — VR180 `v3d ` ISOBMFF box parsing / serialisation.
//! - [`spatial_metadata`] — Google Spatial Media v2 XMP serialisation and parsing,
//!   ISOBMFF `sv3d` and `st3d` box encoding.

pub mod apple_spatial;
pub mod eac;
pub mod fisheye;
pub mod fisheye_lut;
pub mod gaze_tracker;
pub mod headset_metadata;
pub mod mesh_warp;
pub mod octahedral;
pub mod orientation;
pub mod pixel_format;
pub mod point_cloud;
pub mod projection;
pub mod pyramid_blend;
pub mod rectilinear;
pub mod sampling;
pub mod seam_blending;
pub mod spatial_audio_360;
pub mod spatial_metadata;
pub mod stabilization;
pub mod stereo;
pub mod stitching_quality;
pub mod tile_selector;
pub mod tiled;
pub mod v3d;
pub mod viewport;
pub mod viewport_predictor;
pub mod vr180;

// ── Re-exports of key public types ──────────────────────────────────────────

pub use eac::{
    eac_tangent_to_uv, eac_to_equirect, eac_uv_to_sphere, eac_uv_to_tangent, equirect_to_eac,
    sphere_to_eac_face,
};
pub use fisheye::{
    detect_horizon, equirect_to_fisheye, fisheye_to_equirect, DualFisheyeStitcher,
    DualFisheyeStitcherBuilder, FisheyeModel, FisheyeParams,
};
pub use fisheye_lut::{FisheyeLut, FisheyeLutEntry};
pub use gaze_tracker::{
    angular_distance as gaze_angular_distance, AttentionHeatmap, GazeDwellEvent, GazeHistory,
    GazeRecord, HotspotAccumulator, HotspotCell,
};
pub use headset_metadata::{
    HeadsetMetadata, HeadsetOptimizedConfig, HeadsetType, ProjectionConfig,
};
pub use mesh_warp::{MeshBuilder, MeshVertex, RadialDistortionParams, WarpMesh};
pub use pixel_format::{sample_pixel, write_pixel, PixelFormat as VrPixelFormat};
pub use point_cloud::{direction_to_equirect_pixel, Point3d, PointCloud, PointCloudProjector};
pub use projection::{
    angular_distance_rad, bilinear_sample_u8, compute_psnr, cube_face_to_sphere, cube_to_equirect,
    equirect_to_cube, equirect_to_sphere, sphere_equirect_max_roundtrip_error_rad,
    sphere_to_cube_face, sphere_to_equirect, CubeFace, CubeFaceCoord, SphericalCoord, UvCoord,
};
pub use pyramid_blend::{
    blend_laplacian, reconstruct as reconstruct_pyramid, GaussianPyramid, LaplacianPyramid,
};
pub use rectilinear::{extract_rectilinear, RectilinearProjection, VirtualCamera};
pub use sampling::{
    sample_bicubic, sample_f32, sample_u16, sample_u8, FilterKernel, PixelComponent,
};
pub use seam_blending::{feather_face, feather_weights, SeamBlender, SeamQualityMetrics};
pub use spatial_audio_360::{
    encode_foa_sn3d, mix_ambisonics, AmbisonicsMetadata, AmbisonicsOrder, AudioSphereMap,
    BinauralHint, NormalisationConvention,
};
pub use spatial_metadata::{ProjectionType, SpatialMediaV2, StereoVideoBox, Sv3dBox};
pub use stereo::{
    merge_stereo_frames, merge_stereo_frames_rgba, split_stereo_frame, split_stereo_frame_rgba,
    stereo_from_depth, DepthMap, StereoCalibration, StereoLayout, StereoMetadata, StereoQuality,
};
pub use stitching_quality::{
    ColourMismatch, ParallaxDetector, ParallaxReport, SeamVisibilityAnalyser, StitchReport,
};
pub use tile_selector::{
    FovFalloff, FovQualityAllocator, FovTileQuality, QualityLadder, TileAssignment, TileGrid,
    TileIndex, TilePriority, TileSelector,
};
pub use tiled::{
    bilinear_sample_rgb_packed, equirect_to_cube_parallel, equirect_to_cube_tiled,
    resample_equirect_parallel,
};
pub use viewport::{render_viewport, render_viewport_with_coords, ViewportParams};
pub use viewport_predictor::{
    ConstantPositionPredictor, HeadPoseSample, LinearVelocityPredictor, PredictedOrientation,
    ViewportPredictor, ViewportRegion, WeightedHistoryPredictor,
};
pub use vr180::{Vr180Converter, Vr180PixelFormat};

// ─── Error type ─────────────────────────────────────────────────────────────

/// Errors produced by 360° VR video processing operations.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum VrError {
    /// Image or frame dimensions are invalid (zero, mismatched, etc.).
    #[error("invalid dimensions: {0}")]
    InvalidDimensions(String),

    /// A coordinate value is outside the valid range.
    #[error("invalid coordinate")]
    InvalidCoordinate,

    /// A projection conversion failed.
    #[error("projection error: {0}")]
    ProjectionError(String),

    /// Parsing of metadata or configuration failed.
    #[error("parse error: {0}")]
    ParseError(String),

    /// The supplied pixel buffer is too small for the declared dimensions.
    #[error("buffer too small: expected {expected} bytes, got {got}")]
    BufferTooSmall {
        /// Minimum required buffer size in bytes.
        expected: usize,
        /// Actual buffer size in bytes.
        got: usize,
    },

    /// A required cube-map face is missing.
    #[error("missing cube face: {0}")]
    MissingFace(String),

    /// The stereo layout is not supported for this operation.
    #[error("unsupported stereo layout: {0}")]
    UnsupportedLayout(String),
}
