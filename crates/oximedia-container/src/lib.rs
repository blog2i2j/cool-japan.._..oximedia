//! `OxiMedia` Container Layer
//!
//! Container format handling with resilient parsing for:
//! - Matroska (.mkv) / `WebM` (.webm)
//! - Ogg (.ogg, .opus, .oga)
//! - FLAC (.flac)
//! - WAV (.wav)
//! - MP4 (.mp4) - AV1/VP9 only
//! - MPEG-TS (.ts, .m2ts) - AV1/VP9/VP8/Opus/FLAC only
//! - YUV4MPEG2 (.y4m) - Raw uncompressed video
//!
//! # Overview
//!
//! This crate provides demuxers and muxers for media container formats.
//! A demuxer reads a container file and extracts compressed packets,
//! while a muxer combines compressed packets into a container file.
//!
//! # Key Types
//!
//! - [`ContainerFormat`] - Enumeration of supported container formats
//! - [`probe_format`] - Detect container format from magic bytes
//! - [`Packet`] - Compressed media packet with timestamps
//! - [`StreamInfo`] - Information about a stream (codec, dimensions, etc.)
//! - [`Demuxer`] - Trait for container demuxers
//! - [`Muxer`] - Trait for container muxers
//!
//! # Demuxing Example
//!
//! ```ignore
//! use oximedia_container::{probe_format, demux::MatroskaDemuxer, Demuxer};
//! use oximedia_io::FileSource;
//!
//! // Detect format from file header
//! let mut source = FileSource::open("video.mkv").await?;
//! let mut buf = [0u8; 12];
//! source.read(&mut buf).await?;
//! let format = probe_format(&buf)?;
//! println!("Format: {:?}", format.format);
//!
//! // Demux the file
//! source.seek(std::io::SeekFrom::Start(0)).await?;
//! let mut demuxer = MatroskaDemuxer::new(source);
//! demuxer.probe().await?;
//!
//! for stream in demuxer.streams() {
//!     println!("Stream {}: {:?}", stream.index, stream.codec);
//! }
//!
//! while let Ok(packet) = demuxer.read_packet().await {
//!     println!("Packet: stream={}, size={}, keyframe={}",
//!              packet.stream_index, packet.size(), packet.is_keyframe());
//! }
//! ```
//!
//! # Muxing Example
//!
//! ```ignore
//! use oximedia_container::mux::{MatroskaMuxer, Muxer, MuxerConfig};
//!
//! let config = MuxerConfig::new()
//!     .with_title("My Video");
//!
//! let mut muxer = MatroskaMuxer::new(sink, config);
//! muxer.add_stream(video_info)?;
//! muxer.add_stream(audio_info)?;
//!
//! muxer.write_header().await?;
//!
//! for packet in packets {
//!     muxer.write_packet(&packet).await?;
//! }
//!
//! muxer.write_trailer().await?;
//! ```
//!
//! # Metadata Editing Example
//!
//! ```ignore
//! use oximedia_container::metadata::MetadataEditor;
//!
//! let mut editor = MetadataEditor::open("audio.flac").await?;
//!
//! // Read tags
//! if let Some(title) = editor.get_text("TITLE") {
//!     println!("Title: {}", title);
//! }
//!
//! // Modify tags
//! editor.set("TITLE", "New Title");
//! editor.set("ARTIST", "New Artist");
//!
//! // Save changes
//! editor.save().await?;
//! ```

pub mod attach;
pub mod bitrate_stats;
pub mod box_header;
pub mod caf;
pub mod chapters;
pub mod chunk_map;
pub mod container_probe;
pub(crate) mod container_probe_parsers;
pub mod cue;
pub mod data;
pub mod demux;
pub mod edit;
pub mod edit_list;
mod format;
pub mod fragment;
pub mod media_header;
pub mod metadata;
pub mod mkv_cluster;
pub mod multi_angle;
pub mod mux;
pub mod ogg_page;
mod packet;
pub mod preroll;
mod probe;
pub mod pts_dts;
pub mod pts_dts_batch;
pub mod sample_entry;
pub mod sample_table;
mod seek;
pub mod segment_index;
mod stream;
pub mod stream_index;
pub mod streaming;
pub mod subtitle_mux;
pub mod timecode;
pub mod track_header;
pub mod track_info;
pub mod tracks;

// Re-export main types at crate root
pub use container_probe::{DetailedContainerInfo, DetailedStreamInfo, MultiFormatProber};
pub use demux::Demuxer;
pub use format::ContainerFormat;
pub use mux::{Muxer, MuxerConfig};
pub use packet::{Packet, PacketFlags};
pub use probe::{probe_format, ProbeResult};
pub use seek::{
    MultiTrackSeeker, MultiTrackSeekerError, PtsSeekResult, SampleAccurateSeeker, SampleIndexEntry,
    SeekAccuracy, SeekFlags, SeekIndex, SeekIndexEntry, SeekPlan, SeekResult, SeekTarget,
    TrackIndex,
};
pub use stream::{CodecParams, Metadata, StreamInfo};
