//! AVI demuxer (MJPEG, H.264, RGB24, PCM audio) with OpenDML support.

pub mod reader;
pub use reader::{AviAudioFormat, AviDemuxError, AviMjpegReader};
