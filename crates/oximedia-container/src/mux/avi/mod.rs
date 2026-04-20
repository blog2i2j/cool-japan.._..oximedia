//! AVI muxer (MJPEG, H.264, RGB24) with OpenDML support.

pub mod writer;
pub use writer::{AudioConfig, AviError, AviMjpegWriter, VideoCodec, AVI_RIFF_SIZE_LIMIT};
