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
//!
//! # Wave 4 API Additions
//!
//! ## MP4 Fragment Mode — `Mp4FragmentMode`
//!
//! [`mux::mp4::Mp4FragmentMode`] controls how the MP4 muxer arranges sample data:
//!
//! | Variant | Description |
//! |---------|-------------|
//! | `Progressive` | Classic MP4: single `moov` + `mdat`, optimal for download |
//! | `Fragmented { fragment_duration_ms }` | ISOBMFF fragments; each fragment is a self-contained `moof`+`mdat` pair |
//!
//! `Mp4Mode` is a backward-compatible type alias for `Mp4FragmentMode`.
//!
//! ```ignore
//! use oximedia_container::mux::mp4::{Mp4Muxer, Mp4Config, Mp4FragmentMode};
//!
//! // Progressive (default)
//! let config = Mp4Config::new().with_mode(Mp4FragmentMode::Progressive);
//!
//! // Fragmented — 4-second fragments for DASH/HLS delivery
//! let config = Mp4Config::new()
//!     .with_mode(Mp4FragmentMode::Fragmented { fragment_duration_ms: 4000 });
//! ```
//!
//! ## Sample-Accurate Seek Cursor — [`DecodeSkipCursor`]
//!
//! [`DecodeSkipCursor`] is returned by the `seek_sample_accurate()` methods on the
//! Matroska, MP4, and AVI demuxers. It locates the nearest keyframe at or before a
//! target PTS and records how many decoded samples must be discarded to reach the
//! precise presentation position.
//!
//! | Field | Type | Description |
//! |-------|------|-------------|
//! | `byte_offset` | `u64` | File offset where decoding should start |
//! | `sample_index` | `usize` | 0-based index of the keyframe sample |
//! | `skip_samples` | `u32` | Samples to decode-and-discard after seeking |
//! | `target_pts` | `i64` | Requested PTS in track timescale units |
//!
//! ```ignore
//! use oximedia_container::demux::MatroskaDemuxer;
//! use oximedia_io::FileSource;
//!
//! let mut demuxer = MatroskaDemuxer::new(FileSource::open("video.mkv").await?);
//! demuxer.probe().await?;
//!
//! // Seek to exactly 30 seconds (in track timescale units)
//! let cursor = demuxer.seek_sample_accurate(2_700_000).await?;
//! println!("Start decode at byte {}, skip {} samples",
//!          cursor.byte_offset, cursor.skip_samples);
//! ```
//!
//! ## CMAF Chunked Transfer — `CmafChunkMode` / `CmafChunkedConfig`
//!
//! [`streaming::mux::CmafChunkMode`] and [`streaming::mux::CmafChunkedConfig`] implement
//! chunked CMAF delivery as defined in ISO/IEC 23000-19, enabling sub-segment delivery
//! for LL-HLS and LL-DASH workflows.
//!
//! | Mode | Description |
//! |------|-------------|
//! | `Standard` | Whole-segment delivery (default, no chunking) |
//! | `Chunked` | Each chunk is one or more complete `moof`+`mdat` pairs |
//! | `LowLatencyChunked` | Each chunk is exactly one sample (minimum latency) |
//!
//! `CmafChunkedConfig` carries additional settings: `chunk_duration_ms`,
//! `max_samples_per_chunk`, `include_mfra`, `signal_low_latency` (writes `cmfl`
//! compatible brand in the `styp` box), and `part_target_duration_ms` for LL-HLS.
//!
//! ```ignore
//! use oximedia_container::streaming::mux::{CmafChunkedConfig, CmafChunkMode};
//!
//! let config = CmafChunkedConfig::new()
//!     .with_mode(CmafChunkMode::LowLatencyChunked)
//!     .with_low_latency(true);
//! ```
//!
//! ## Matroska v4 Block Addition Mapping — `BlockAdditionMapping`
//!
//! `demux::matroska::matroska_v4::BlockAdditionMapping` represents a Matroska v4
//! `BlockAdditionMapping` element (EBML ID 0x41CB), which carries auxiliary per-block
//! data channels such as HDR10+ metadata, Dolby Vision RPU data, or depth maps.
//!
//! | Field | Description |
//! |-------|-------------|
//! | `id_name` | Human-readable channel name (e.g., `"hdr10plus"`, `"dovi_rpu"`) |
//! | `id_type` | Numeric type per the Matroska Block Addition Mapping Registry |
//! | `id_extra_data` | Codec-specific configuration payload |
//!
//! Access via `StreamInfo::block_addition_mappings` after probing a Matroska track.

pub mod attach;
pub mod bitrate_stats;
pub mod box_header;
pub mod caf;
pub mod chapters;
pub mod chunk_map;
pub mod container_probe;
pub(crate) mod container_probe_parsers;
pub mod cue;
pub mod dash;
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
pub mod riff;
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
pub use dash::{
    emit_mpd, DashAdaptationSet, DashManifestConfig, DashRepresentation, DashSegmentTemplate,
    DashSegmentTimeline, DashSegmentTimelineEntry,
};
pub use demux::mpegts::scte35::{
    parse_splice_info_section, BreakDuration, Scte35Config, Scte35Parser, SpliceCommand,
    SpliceDescriptor, SpliceInfoSection, SpliceInsert, SpliceTime, SCTE35_DEFAULT_PID,
};
pub use demux::Demuxer;
pub use format::ContainerFormat;
pub use metadata::batch::{BatchMetadataUpdate, BatchResult};
pub use mux::mpegts::scte35::{
    emit_splice_insert, emit_splice_null, emit_time_signal, SpliceInsertConfig,
};
pub use mux::{Muxer, MuxerConfig};
pub use packet::{Packet, PacketFlags};
pub use probe::{probe_format, ProbeResult};
pub use seek::{
    DecodeSkipCursor, MultiTrackSeeker, MultiTrackSeekerError, PtsSeekResult, SampleAccurateSeeker,
    SampleIndexEntry, SeekAccuracy, SeekFlags, SeekIndex, SeekIndexEntry, SeekPlan, SeekResult,
    SeekTarget, TrackIndex,
};
pub use stream::{CodecParams, Metadata, StreamInfo};
