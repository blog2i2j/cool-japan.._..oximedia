//! Core types and traits for `OxiMedia`.
//!
//! `oximedia-core` provides the foundational types and traits used throughout
//! the `OxiMedia` multimedia framework. This includes:
//!
//! - **Types**: Rational numbers, timestamps, pixel/sample formats, codec IDs
//! - **Traits**: Decoder and demuxer interfaces
//! - **Error handling**: Unified error type with patent violation detection
//! - **Memory management**: Buffer pools for zero-copy operations
//! - **HDR support**: HDR metadata, transfer functions, color primaries, and conversions
//!
//! # Green List Only
//!
//! `OxiMedia` only supports patent-free codecs:
//!
//! | Video | Audio | Subtitle |
//! |-------|-------|----------|
//! | AV1   | Opus  | `WebVTT`   |
//! | VP9   | Vorbis| ASS/SSA  |
//! | VP8   | FLAC  | SRT      |
//! | Theora| PCM   |          |
//!
//! Attempting to use patent-encumbered codecs (H.264, H.265, AAC, etc.)
//! will result in a [`PatentViolation`](error::OxiError::PatentViolation) error.
//!
//! # Example
//!
//! ```
//! use oximedia_core::types::{Rational, Timestamp, PixelFormat, CodecId};
//! use oximedia_core::error::OxiResult;
//!
//! fn example() -> OxiResult<()> {
//!     // Create a timestamp at 1 second with millisecond precision
//!     let ts = Timestamp::new(1000, Rational::new(1, 1000));
//!     assert!((ts.to_seconds() - 1.0).abs() < f64::EPSILON);
//!
//!     // Check codec properties
//!     let codec = CodecId::Av1;
//!     assert!(codec.is_video());
//!
//!     // Check pixel format properties
//!     let format = PixelFormat::Yuv420p;
//!     assert!(format.is_planar());
//!     assert_eq!(format.plane_count(), 3);
//!
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]

pub mod alloc;
pub mod bitrate;
pub mod buffer_pool;
pub mod channel_layout;
pub mod codec_info;
pub mod codec_matrix;
pub mod codec_negotiation;
pub mod codec_params;
pub mod color_metadata;
pub mod convert;
pub mod downmix;
pub mod error;
pub mod error_context;
pub mod event_queue;
pub mod event_stream;
pub mod fourcc;
pub mod frame_info;
pub mod frame_pool;
pub mod frame_sharing;
pub mod hdr;
pub mod media_clock;
pub mod media_props;
pub mod media_segment;
pub mod media_time;
pub mod memory;
pub mod pixel_format;
pub mod pixel_format_color;
pub mod rational;
pub mod resource_handle;
pub mod ring_buffer;
pub mod sample_conv;
pub mod sample_format;
pub mod sync;
pub mod timestamp_arith;
pub mod traits;
pub mod type_registry;
pub mod types;
pub mod version;
pub mod work_queue;
pub mod work_steal;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

// Re-export commonly used items at crate root
pub use error::{OxiError, OxiResult};
pub use types::{CodecId, MediaType, PixelFormat, Rational, SampleFormat, Timestamp};

/// Initialises the OxiMedia WASM module.
///
/// Sets up panic hooks for better error messages in the browser console.
/// Called from `oximedia-wasm` init function.
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub fn wasm_init() {
    console_error_panic_hook::set_once();
}
