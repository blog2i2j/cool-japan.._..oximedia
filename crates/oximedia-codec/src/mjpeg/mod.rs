//! Motion JPEG (MJPEG) codec implementation.
//!
//! MJPEG encodes each video frame independently as a baseline JPEG image.
//! This is a patent-free, intra-only codec commonly used in:
//!
//! - Digital cameras and camcorders
//! - Video capture devices
//! - Webcams
//! - AVI and MOV containers
//!
//! # Architecture
//!
//! The MJPEG codec wraps `oximedia-image`'s pure-Rust JPEG baseline encoder
//! and decoder, adapting them to the `VideoEncoder`/`VideoDecoder` traits.
//!
//! Each frame is independently encoded (all frames are keyframes), which
//! provides:
//! - Random access to any frame
//! - Simple editing (cut/trim without re-encoding)
//! - Low encoding latency
//! - No inter-frame error propagation
//!
//! # Example
//!
//! ```ignore
//! use oximedia_codec::mjpeg::{MjpegEncoder, MjpegConfig};
//! use oximedia_codec::traits::VideoEncoder;
//!
//! let config = MjpegConfig::new(640, 480)?.with_quality(85);
//! let mut encoder = MjpegEncoder::new(config)?;
//!
//! encoder.send_frame(&video_frame)?;
//! while let Some(packet) = encoder.receive_packet()? {
//!     // Write MJPEG packet to container
//! }
//! ```

pub mod decoder;
pub mod encoder;
pub mod types;

pub use decoder::MjpegDecoder;
pub use encoder::MjpegEncoder;
pub use types::{MjpegConfig, MjpegError, MjpegFrameInfo, MjpegMarkerType};
