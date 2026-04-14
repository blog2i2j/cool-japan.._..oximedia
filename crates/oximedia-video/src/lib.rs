//! Professional video processing operations for OxiMedia.
//!
//! Provides block-based motion estimation and compensation, frame rate
//! conversion with intermediate frame generation, video deinterlacing,
//! scene change detection, 3:2 pulldown cadence detection, perceptual
//! video fingerprinting, and temporal noise reduction.

#![warn(missing_docs, rust_2018_idioms, unreachable_pub, unsafe_code)]

pub mod adaptive_block_selection;
pub mod adaptive_complexity_detect;
pub mod adaptive_denoise;
pub mod av1_grain_params;
pub mod bidirectional_motion;
pub mod cadence_confidence;
pub mod cadence_convert;
pub mod color_space_convert;
pub mod complexity;
pub mod deinterlace;
pub mod duplicate_frame;
pub mod duplicate_frame_detect;
pub mod field_order_detect;
pub mod film_grain_synthesis;
pub mod frame_blending;
pub mod frame_interpolation;
pub mod frame_rate_convert;
pub mod grain;
pub mod hdr_meta;
pub mod hdr_tonemapping;
pub mod interlace_detector;
pub mod invariant_fingerprint;
pub mod lens_correction;
pub mod mctf;
pub mod motion_compensation;
pub mod motion_search;
pub mod noise_profile;
pub mod parallel_motion_search;
pub mod pulldown_detect;
pub mod quality_metrics;
pub mod quality_score;
pub mod scene_detection;
pub mod shot_boundary;
pub mod shot_boundary_classifier;
pub mod simd_ops;
pub mod slow_motion;
pub mod stabilization;
pub mod subpixel_motion_vector;
pub mod subpixel_refiner;
pub mod super_resolution;
pub mod superimpose;
pub mod temporal_denoise;
pub mod video_crop;
pub mod video_fingerprint;
pub mod vignette;
