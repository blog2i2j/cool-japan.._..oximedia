//! Format conversion utilities for pixel and audio formats.
//!
//! This module provides efficient and accurate format conversion functions for:
//!
//! - **Pixel format conversion**: YUV<->RGB, planar format conversions, grayscale
//! - **Audio sample format conversion**: Planar<->interleaved, sample type conversions
//!
//! # Color Space Matrices
//!
//! The module supports standard color space matrices:
//!
//! - **BT.601**: Standard definition (SD) television
//! - **BT.709**: High definition (HD) television
//!
//! # Examples
//!
//! ```
//! use oximedia_core::convert::pixel::{yuv420p_to_rgb24, ColorMatrix};
//!
//! let width = 640;
//! let height = 480;
//!
//! // YUV420p planes
//! let y_plane = vec![128u8; width * height];
//! let u_plane = vec![128u8; (width / 2) * (height / 2)];
//! let v_plane = vec![128u8; (width / 2) * (height / 2)];
//!
//! // Convert to RGB24
//! let rgb = yuv420p_to_rgb24(
//!     &y_plane,
//!     &u_plane,
//!     &v_plane,
//!     width,
//!     height,
//!     ColorMatrix::Bt709,
//! );
//! ```

pub mod audio;
pub mod pixel;
pub mod simd_pixel;
#[cfg(target_arch = "x86_64")]
pub(crate) mod simd_pixel_sse41;

pub use audio::{
    convert_sample_format, interleaved_to_planar, planar_to_interleaved, AudioConverter,
    SampleConverter,
};
pub use pixel::{
    gray8_to_rgb24, gray8_to_yuv420p, rgb24_to_gray8, rgb24_to_yuv420p, yuv420p_to_gray8,
    yuv420p_to_rgb24, yuv420p_to_yuv444p, yuv444p_to_yuv420p, ColorMatrix, PixelConverter,
};
