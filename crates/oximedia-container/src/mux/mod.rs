//! Muxer implementations.
//!
//! This module provides muxers for writing compressed packets
//! to container formats.
//!
//! # Supported Formats
//!
//! - Matroska/`WebM` via [`MatroskaMuxer`]
//! - FLAC via [`FlacMuxer`]
//! - Ogg via [`OggMuxer`] (Opus, Vorbis, FLAC)
//! - WAV/RIFF via [`WavMuxer`]
//! - MPEG-TS via [`MpegTsMuxer`] (AV1/VP9/VP8/Opus/FLAC)
//!
//! # Patent Protection
//!
//! All muxers only support royalty-free codecs. Attempting to
//! mux patent-encumbered codecs will result in a
//! [`PatentViolation`](oximedia_core::OxiError::PatentViolation) error.
//!
//! # Example
//!
//! ```ignore
//! use oximedia_container::mux::{Muxer, MuxerConfig, MatroskaMuxer};
//! use oximedia_io::FileSource;
//!
//! // Create muxer configuration
//! let config = MuxerConfig::new()
//!     .with_title("My Video")
//!     .with_writing_app("OxiMedia");
//!
//! // Create muxer
//! let sink = FileSink::create("output.mkv").await?;
//! let mut muxer = MatroskaMuxer::new(sink, config);
//!
//! // Add streams
//! muxer.add_stream(video_stream_info)?;
//! muxer.add_stream(audio_stream_info)?;
//!
//! // Write header
//! muxer.write_header().await?;
//!
//! // Write packets
//! for packet in packets {
//!     muxer.write_packet(&packet).await?;
//! }
//!
//! // Finalize
//! muxer.write_trailer().await?;
//! ```

pub mod avi;
pub mod cmaf;
pub mod flac;
pub mod interleave;
pub mod matroska;
pub mod mp4;
pub mod mpegts;
pub mod ogg;
mod traits;
pub mod wav;
pub mod y4m;

pub use avi::{AviError, AviMjpegWriter};
pub use cmaf::{CmafBrand, CmafConfig, CmafMuxer, CmafSample, CmafSegment, CmafTrack, TrackType};
pub use flac::FlacMuxer;
pub use matroska::MatroskaMuxer;
pub use mp4::{
    AudioCodecInfo, BasicMp4Error, BasicMp4Muxer, FourCC, Mp4Config, Mp4FragmentMode, Mp4Mode,
    Mp4Muxer, Mp4Sample, Mp4TrackWriter, SimpleMp4Config, SimpleMp4Error, SimpleMp4Muxer,
    TrackCodec, VideoCodecInfo,
};
pub use mpegts::MpegTsMuxer;
pub use ogg::OggMuxer;
pub use traits::{Muxer, MuxerConfig, OutputFormat};
pub use wav::{WavFormat, WavFormatConfig, WavMuxer};
pub use y4m::{Y4mMuxer, Y4mMuxerBuilder};
