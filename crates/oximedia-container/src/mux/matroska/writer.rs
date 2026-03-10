//! Matroska muxer implementation.

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use async_trait::async_trait;
use oximedia_core::{CodecId, MediaType, OxiError, OxiResult, Rational};
use oximedia_io::MediaSource;
use std::io::SeekFrom;

use crate::demux::matroska::ebml::element_id;
use crate::mux::traits::{Muxer, MuxerConfig};
use crate::{ContainerFormat, Packet, StreamInfo};

use super::cluster::ClusterWriter;
use super::cues::CueWriter;

// ============================================================================
// Constants
// ============================================================================

/// Default timecode scale (1ms = 1,000,000 nanoseconds).
const DEFAULT_TIMECODE_SCALE: u64 = 1_000_000;

/// Maximum cluster duration in timecode units (5 seconds at default scale).
const DEFAULT_MAX_CLUSTER_DURATION: i64 = 5000;

/// Maximum cluster size in bytes (5 MB).
const DEFAULT_MAX_CLUSTER_SIZE: usize = 5 * 1024 * 1024;

/// EBML header for `WebM`.
const WEBM_DOC_TYPE: &[u8] = b"webm";

/// EBML header for Matroska.
const MATROSKA_DOC_TYPE: &[u8] = b"matroska";

// ============================================================================
// Matroska Muxer
// ============================================================================

/// Matroska/`WebM` muxer.
///
/// Creates Matroska or `WebM` container files from compressed media packets.
pub struct MatroskaMuxer<W> {
    /// The underlying writer.
    sink: W,

    /// Muxer configuration.
    config: MuxerConfig,

    /// Registered streams.
    streams: Vec<StreamInfo>,

    /// Timecode scale in nanoseconds.
    timecode_scale: u64,

    /// Whether header has been written.
    header_written: bool,

    /// Position of segment element (for size fixup).
    segment_position: u64,

    /// Position of segment info duration (for fixup).
    duration_position: Option<u64>,

    /// Current cluster writer.
    cluster_writer: Option<ClusterWriter>,

    /// Cue writer for seeking.
    cue_writer: CueWriter,

    /// Maximum duration in the stream (for final duration).
    max_timestamp: i64,

    /// Current write position.
    position: u64,

    /// Segment data start position.
    segment_data_start: u64,

    /// Output format (`WebM` or Matroska).
    output_format: ContainerFormat,

    /// Maximum cluster duration in timecode units.
    max_cluster_duration: i64,

    /// Maximum cluster size in bytes.
    max_cluster_size: usize,
}

impl<W> MatroskaMuxer<W> {
    /// Creates a new Matroska muxer.
    ///
    /// # Arguments
    ///
    /// * `sink` - The writer to output to
    /// * `config` - Muxer configuration
    #[must_use]
    pub fn new(sink: W, config: MuxerConfig) -> Self {
        let max_cluster_duration = config
            .output_format
            .max_cluster_duration_ms
            .map_or(DEFAULT_MAX_CLUSTER_DURATION, i64::from);

        let max_cluster_size = config
            .output_format
            .max_cluster_size
            .map_or(DEFAULT_MAX_CLUSTER_SIZE, |s| s as usize);

        Self {
            sink,
            config,
            streams: Vec::new(),
            timecode_scale: DEFAULT_TIMECODE_SCALE,
            header_written: false,
            segment_position: 0,
            duration_position: None,
            cluster_writer: None,
            cue_writer: CueWriter::new(),
            max_timestamp: 0,
            position: 0,
            segment_data_start: 0,
            output_format: ContainerFormat::Matroska,
            max_cluster_duration,
            max_cluster_size,
        }
    }

    /// Returns a reference to the underlying sink.
    #[must_use]
    pub const fn sink(&self) -> &W {
        &self.sink
    }

    /// Returns a mutable reference to the underlying sink.
    pub fn sink_mut(&mut self) -> &mut W {
        &mut self.sink
    }

    /// Consumes the muxer and returns the underlying sink.
    #[must_use]
    #[allow(dead_code)]
    pub fn into_sink(self) -> W {
        self.sink
    }

    /// Determines output format based on codecs used.
    fn determine_output_format(&mut self) {
        // Check if all codecs are WebM-compatible
        let all_webm_compatible = self.streams.iter().all(|s| {
            matches!(
                s.codec,
                CodecId::Vp8 | CodecId::Vp9 | CodecId::Av1 | CodecId::Opus | CodecId::Vorbis
            )
        });

        self.output_format = if all_webm_compatible {
            ContainerFormat::WebM
        } else {
            ContainerFormat::Matroska
        };
    }

    /// Returns the Matroska codec ID string for a codec.
    fn codec_id_string(codec: CodecId) -> &'static str {
        match codec {
            CodecId::Av1 => "V_AV1",
            CodecId::Vp9 => "V_VP9",
            CodecId::Vp8 => "V_VP8",
            CodecId::Theora => "V_THEORA",
            CodecId::Opus => "A_OPUS",
            CodecId::Vorbis => "A_VORBIS",
            CodecId::Flac => "A_FLAC",
            CodecId::Pcm => "A_PCM/INT/LIT",
            CodecId::WebVtt => "S_TEXT/WEBVTT",
            CodecId::Ass => "S_TEXT/ASS",
            CodecId::Ssa => "S_TEXT/SSA",
            CodecId::Srt => "S_TEXT/UTF8",
            _ => "V_UNCOMPRESSED", // Fallback for future codecs
        }
    }

    /// Converts a timestamp to timecode scale units.
    #[allow(clippy::cast_precision_loss)]
    fn to_timecode(&self, pts: i64, timebase: Rational) -> i64 {
        if timebase.den == 0 {
            return pts;
        }
        // Convert to nanoseconds then to timecode scale
        let ns = (pts as f64 * timebase.num as f64 / timebase.den as f64) * 1_000_000_000.0;
        (ns / self.timecode_scale as f64) as i64
    }
}

impl<W: MediaSource> MatroskaMuxer<W> {
    /// Writes bytes to the sink.
    async fn write_bytes(&mut self, data: &[u8]) -> OxiResult<()> {
        self.sink.write_all(data).await?;
        self.position += data.len() as u64;
        Ok(())
    }

    /// Writes an EBML element ID.
    async fn write_element_id(&mut self, id: u32) -> OxiResult<()> {
        let bytes = encode_element_id(id);
        self.write_bytes(&bytes).await
    }

    /// Writes an EBML element size.
    async fn write_element_size(&mut self, size: u64) -> OxiResult<()> {
        let bytes = encode_vint_size(size);
        self.write_bytes(&bytes).await
    }

    /// Writes an unknown/unbounded EBML element size.
    async fn write_unknown_size(&mut self) -> OxiResult<()> {
        // 8-byte unknown size marker
        self.write_bytes(&[0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF])
            .await
    }

    /// Writes a complete EBML unsigned integer element.
    async fn write_uint_element(&mut self, id: u32, value: u64) -> OxiResult<()> {
        let data = encode_uint(value);
        self.write_element_id(id).await?;
        self.write_element_size(data.len() as u64).await?;
        self.write_bytes(&data).await
    }

    /// Writes a complete EBML string element.
    #[allow(dead_code)]
    async fn write_string_element(&mut self, id: u32, value: &[u8]) -> OxiResult<()> {
        self.write_element_id(id).await?;
        self.write_element_size(value.len() as u64).await?;
        self.write_bytes(value).await
    }

    /// Writes a complete EBML float element (8 bytes).
    #[allow(dead_code)]
    async fn write_float_element(&mut self, id: u32, value: f64) -> OxiResult<()> {
        self.write_element_id(id).await?;
        self.write_element_size(8).await?;
        self.write_bytes(&value.to_be_bytes()).await
    }

    /// Writes a complete EBML binary element.
    #[allow(dead_code)]
    async fn write_binary_element(&mut self, id: u32, data: &[u8]) -> OxiResult<()> {
        self.write_element_id(id).await?;
        self.write_element_size(data.len() as u64).await?;
        self.write_bytes(data).await
    }

    /// Builds and returns the EBML header bytes.
    fn build_ebml_header(&self) -> Vec<u8> {
        let mut header = Vec::new();

        let doc_type = if self.output_format == ContainerFormat::WebM {
            WEBM_DOC_TYPE
        } else {
            MATROSKA_DOC_TYPE
        };

        // Build EBML header content
        let mut content = Vec::new();

        // EBMLVersion: 1
        content.extend(encode_element_id(element_id::EBML_VERSION));
        content.push(0x81); // Size: 1
        content.push(0x01);

        // EBMLReadVersion: 1
        content.extend(encode_element_id(element_id::EBML_READ_VERSION));
        content.push(0x81);
        content.push(0x01);

        // EBMLMaxIDLength: 4
        content.extend(encode_element_id(element_id::EBML_MAX_ID_LENGTH));
        content.push(0x81);
        content.push(0x04);

        // EBMLMaxSizeLength: 8
        content.extend(encode_element_id(element_id::EBML_MAX_SIZE_LENGTH));
        content.push(0x81);
        content.push(0x08);

        // DocType
        content.extend(encode_element_id(element_id::DOC_TYPE));
        content.extend(encode_vint_size(doc_type.len() as u64));
        content.extend_from_slice(doc_type);

        // DocTypeVersion: 4
        content.extend(encode_element_id(element_id::DOC_TYPE_VERSION));
        content.push(0x81);
        content.push(0x04);

        // DocTypeReadVersion: 2
        content.extend(encode_element_id(element_id::DOC_TYPE_READ_VERSION));
        content.push(0x81);
        content.push(0x02);

        // Write EBML element
        header.extend(encode_element_id(element_id::EBML));
        header.extend(encode_vint_size(content.len() as u64));
        header.extend(content);

        header
    }

    /// Writes the EBML header.
    async fn write_ebml_header(&mut self) -> OxiResult<()> {
        let header = self.build_ebml_header();
        self.write_bytes(&header).await
    }

    /// Writes the Segment element header (with unknown size for streaming).
    async fn write_segment_header(&mut self) -> OxiResult<()> {
        self.segment_position = self.position;
        self.write_element_id(element_id::SEGMENT).await?;
        self.write_unknown_size().await?;
        self.segment_data_start = self.position;
        Ok(())
    }

    /// Builds segment info content.
    fn build_segment_info(&self) -> Vec<u8> {
        let mut content = Vec::new();

        // TimecodeScale
        content.extend(encode_element_id(element_id::TIMECODE_SCALE));
        let ts_bytes = encode_uint(self.timecode_scale);
        content.extend(encode_vint_size(ts_bytes.len() as u64));
        content.extend(ts_bytes);

        // MuxingApp
        if let Some(ref app) = self.config.muxing_app {
            content.extend(encode_element_id(element_id::MUXING_APP));
            content.extend(encode_vint_size(app.len() as u64));
            content.extend_from_slice(app.as_bytes());
        }

        // WritingApp
        if let Some(ref app) = self.config.writing_app {
            content.extend(encode_element_id(element_id::WRITING_APP));
            content.extend(encode_vint_size(app.len() as u64));
            content.extend_from_slice(app.as_bytes());
        }

        // Title
        if let Some(ref title) = self.config.title {
            content.extend(encode_element_id(element_id::TITLE));
            content.extend(encode_vint_size(title.len() as u64));
            content.extend_from_slice(title.as_bytes());
        }

        // Duration placeholder (8 bytes float)
        content.extend(encode_element_id(element_id::DURATION));
        content.push(0x88); // Size: 8
        content.extend(&0.0_f64.to_be_bytes());

        content
    }

    /// Writes the Segment Info element.
    async fn write_segment_info(&mut self) -> OxiResult<()> {
        let content = self.build_segment_info();

        // Record duration position for later fixup
        // Duration is at the end of content, 8 bytes before the end
        self.duration_position = Some(
            self.position + 4 + vint_size(content.len() as u64) as u64 + content.len() as u64 - 8,
        );

        self.write_element_id(element_id::INFO).await?;
        self.write_element_size(content.len() as u64).await?;
        self.write_bytes(&content).await
    }

    /// Builds a single track entry.
    #[allow(clippy::too_many_lines)]
    fn build_track_entry(stream: &StreamInfo, track_num: u64) -> Vec<u8> {
        let mut content = Vec::new();

        // TrackNumber
        content.extend(encode_element_id(element_id::TRACK_NUMBER));
        let tn_bytes = encode_uint(track_num);
        content.extend(encode_vint_size(tn_bytes.len() as u64));
        content.extend(tn_bytes);

        // TrackUID
        content.extend(encode_element_id(element_id::TRACK_UID));
        let uid_bytes = encode_uint(track_num);
        content.extend(encode_vint_size(uid_bytes.len() as u64));
        content.extend(uid_bytes);

        // TrackType
        content.extend(encode_element_id(element_id::TRACK_TYPE));
        content.push(0x81);
        content.push(match stream.media_type {
            MediaType::Video => 1,
            MediaType::Audio => 2,
            MediaType::Subtitle => 17,
            _ => 0,
        });

        // CodecID
        let codec_id = Self::codec_id_string(stream.codec);
        content.extend(encode_element_id(element_id::CODEC_ID));
        content.extend(encode_vint_size(codec_id.len() as u64));
        content.extend_from_slice(codec_id.as_bytes());

        // CodecPrivate (if available)
        if let Some(ref extradata) = stream.codec_params.extradata {
            content.extend(encode_element_id(element_id::CODEC_PRIVATE));
            content.extend(encode_vint_size(extradata.len() as u64));
            content.extend_from_slice(extradata);
        }

        // Video-specific settings
        if stream.media_type == MediaType::Video {
            if let (Some(width), Some(height)) =
                (stream.codec_params.width, stream.codec_params.height)
            {
                let mut video_content = Vec::new();

                // PixelWidth
                video_content.extend(encode_element_id(element_id::PIXEL_WIDTH));
                let w_bytes = encode_uint(u64::from(width));
                video_content.extend(encode_vint_size(w_bytes.len() as u64));
                video_content.extend(w_bytes);

                // PixelHeight
                video_content.extend(encode_element_id(element_id::PIXEL_HEIGHT));
                let h_bytes = encode_uint(u64::from(height));
                video_content.extend(encode_vint_size(h_bytes.len() as u64));
                video_content.extend(h_bytes);

                // Write Video element
                content.extend(encode_element_id(element_id::VIDEO));
                content.extend(encode_vint_size(video_content.len() as u64));
                content.extend(video_content);
            }
        }

        // Audio-specific settings
        if stream.media_type == MediaType::Audio {
            let mut audio_content = Vec::new();

            // SamplingFrequency
            if let Some(sample_rate) = stream.codec_params.sample_rate {
                audio_content.extend(encode_element_id(element_id::SAMPLING_FREQUENCY));
                audio_content.push(0x88); // 8 bytes
                audio_content.extend(&f64::from(sample_rate).to_be_bytes());
            }

            // Channels
            if let Some(channels) = stream.codec_params.channels {
                audio_content.extend(encode_element_id(element_id::CHANNELS));
                audio_content.push(0x81);
                audio_content.push(channels);
            }

            if !audio_content.is_empty() {
                content.extend(encode_element_id(element_id::AUDIO));
                content.extend(encode_vint_size(audio_content.len() as u64));
                content.extend(audio_content);
            }
        }

        // Language
        if let Some(lang) = stream.metadata.get("language") {
            content.extend(encode_element_id(element_id::LANGUAGE));
            content.extend(encode_vint_size(lang.len() as u64));
            content.extend_from_slice(lang.as_bytes());
        }

        // Name
        if let Some(ref title) = stream.metadata.title {
            content.extend(encode_element_id(element_id::NAME));
            content.extend(encode_vint_size(title.len() as u64));
            content.extend_from_slice(title.as_bytes());
        }

        content
    }

    /// Writes the Tracks element.
    async fn write_tracks(&mut self) -> OxiResult<()> {
        let mut tracks_content = Vec::new();

        for (i, stream) in self.streams.iter().enumerate() {
            let track_entry = Self::build_track_entry(stream, (i + 1) as u64);

            // Write TrackEntry element
            tracks_content.extend(encode_element_id(element_id::TRACK_ENTRY));
            tracks_content.extend(encode_vint_size(track_entry.len() as u64));
            tracks_content.extend(track_entry);
        }

        self.write_element_id(element_id::TRACKS).await?;
        self.write_element_size(tracks_content.len() as u64).await?;
        self.write_bytes(&tracks_content).await
    }

    /// Starts a new cluster.
    async fn start_cluster(&mut self, timecode: i64) -> OxiResult<()> {
        // Finalize previous cluster if any
        if self.cluster_writer.is_some() {
            self.finalize_cluster();
        }

        let cluster_position = self.position - self.segment_data_start;

        self.write_element_id(element_id::CLUSTER).await?;
        self.write_unknown_size().await?;

        // Write cluster timestamp
        self.write_uint_element(element_id::TIMESTAMP, timecode as u64)
            .await?;

        self.cluster_writer = Some(ClusterWriter::new(
            timecode,
            cluster_position,
            self.max_cluster_duration,
            self.max_cluster_size,
        ));

        Ok(())
    }

    /// Finalizes the current cluster.
    fn finalize_cluster(&mut self) {
        // Cluster is already written, just clear the state
        self.cluster_writer = None;
    }

    /// Writes a `SimpleBlock`.
    async fn write_simple_block(
        &mut self,
        track_num: u64,
        timecode: i16,
        data: &[u8],
        keyframe: bool,
    ) -> OxiResult<()> {
        let mut block = Vec::new();

        // Track number as VINT
        block.extend(encode_vint(track_num));

        // Timecode (relative to cluster, signed 16-bit big-endian)
        block.extend(&timecode.to_be_bytes());

        // Flags
        let flags: u8 = if keyframe { 0x80 } else { 0x00 };
        block.push(flags);

        // Data
        block.extend_from_slice(data);

        self.write_element_id(element_id::SIMPLE_BLOCK).await?;
        self.write_element_size(block.len() as u64).await?;
        self.write_bytes(&block).await
    }

    /// Writes cues element.
    async fn write_cues(&mut self) -> OxiResult<()> {
        if !self.config.write_cues || self.cue_writer.cue_points.is_empty() {
            return Ok(());
        }

        let cues_content = self.cue_writer.build();
        if cues_content.is_empty() {
            return Ok(());
        }

        self.write_element_id(element_id::CUES).await?;
        self.write_element_size(cues_content.len() as u64).await?;
        self.write_bytes(&cues_content).await
    }

    /// Fixes up the duration in segment info.
    #[allow(clippy::cast_precision_loss)]
    async fn fixup_duration(&mut self) -> OxiResult<()> {
        if !self.config.write_duration {
            return Ok(());
        }

        if let Some(duration_pos) = self.duration_position {
            let duration = self.max_timestamp as f64;
            let duration_bytes = duration.to_be_bytes();

            // Save current position
            let current_pos = self.position;

            // Seek to duration position and write
            self.sink.seek(SeekFrom::Start(duration_pos)).await?;
            self.sink.write_all(&duration_bytes).await?;

            // Seek back to end
            self.sink.seek(SeekFrom::Start(current_pos)).await?;
        }

        Ok(())
    }
}

#[async_trait]
impl<W: MediaSource> Muxer for MatroskaMuxer<W> {
    fn add_stream(&mut self, info: StreamInfo) -> OxiResult<usize> {
        if self.header_written {
            return Err(OxiError::InvalidData(
                "Cannot add stream after header is written".into(),
            ));
        }

        let index = self.streams.len();
        self.streams.push(info);
        Ok(index)
    }

    async fn write_header(&mut self) -> OxiResult<()> {
        if self.header_written {
            return Err(OxiError::InvalidData("Header already written".into()));
        }

        if self.streams.is_empty() {
            return Err(OxiError::InvalidData("No streams configured".into()));
        }

        self.determine_output_format();

        self.write_ebml_header().await?;
        self.write_segment_header().await?;
        self.write_segment_info().await?;
        self.write_tracks().await?;

        self.header_written = true;
        Ok(())
    }

    async fn write_packet(&mut self, packet: &Packet) -> OxiResult<()> {
        if !self.header_written {
            return Err(OxiError::InvalidData("Header not written".into()));
        }

        if packet.stream_index >= self.streams.len() {
            return Err(OxiError::InvalidData(format!(
                "Invalid stream index: {}",
                packet.stream_index
            )));
        }

        let stream = &self.streams[packet.stream_index];
        let timecode = self.to_timecode(packet.pts(), stream.timebase);

        // Update max timestamp
        if timecode > self.max_timestamp {
            self.max_timestamp = timecode;
        }

        // Check if we need a new cluster
        let need_new_cluster = if let Some(ref cluster) = self.cluster_writer {
            cluster.should_start_new(timecode, packet.data.len())
        } else {
            true
        };

        if need_new_cluster {
            // Add cue point for keyframes at cluster boundaries
            if packet.is_keyframe() && self.config.write_cues {
                let cluster_position = if let Some(ref cluster) = self.cluster_writer {
                    cluster.position
                } else {
                    self.position - self.segment_data_start
                };
                self.cue_writer.add_cue_point(
                    timecode as u64,
                    (packet.stream_index + 1) as u64,
                    cluster_position,
                );
            }
            self.start_cluster(timecode).await?;
        }

        // Calculate relative timecode
        let cluster_timecode = self.cluster_writer.as_ref().map_or(0, |c| c.timecode);
        let relative_timecode = (timecode - cluster_timecode) as i16;

        // Write SimpleBlock
        self.write_simple_block(
            (packet.stream_index + 1) as u64,
            relative_timecode,
            &packet.data,
            packet.is_keyframe(),
        )
        .await?;

        // Update cluster state
        if let Some(ref mut cluster) = self.cluster_writer {
            cluster.add_block(timecode, packet.data.len());
        }

        Ok(())
    }

    async fn write_trailer(&mut self) -> OxiResult<()> {
        if !self.header_written {
            return Err(OxiError::InvalidData("Header not written".into()));
        }

        // Finalize last cluster
        self.finalize_cluster();

        // Write cues
        self.write_cues().await?;

        // Fix up duration
        self.fixup_duration().await?;

        Ok(())
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn config(&self) -> &MuxerConfig {
        &self.config
    }
}

// ============================================================================
// EBML Encoding Helpers
// ============================================================================

/// Encodes an element ID to bytes.
///
/// EBML element IDs already include their class marker in the value,
/// so we just output the bytes that make up the ID.
fn encode_element_id(id: u32) -> Vec<u8> {
    if id <= 0xFF {
        vec![id as u8]
    } else if id <= 0xFFFF {
        vec![(id >> 8) as u8, id as u8]
    } else if id <= 0xFF_FFFF {
        vec![(id >> 16) as u8, (id >> 8) as u8, id as u8]
    } else {
        vec![
            (id >> 24) as u8,
            (id >> 16) as u8,
            (id >> 8) as u8,
            id as u8,
        ]
    }
}

/// Encodes a VINT (variable integer).
fn encode_vint(value: u64) -> Vec<u8> {
    if value < 0x80 {
        vec![0x80 | value as u8]
    } else if value < 0x4000 {
        vec![0x40 | (value >> 8) as u8, value as u8]
    } else if value < 0x1F_FFFF {
        vec![0x20 | (value >> 16) as u8, (value >> 8) as u8, value as u8]
    } else if value < 0x0FFF_FFFF {
        vec![
            0x10 | (value >> 24) as u8,
            (value >> 16) as u8,
            (value >> 8) as u8,
            value as u8,
        ]
    } else {
        // Use 8-byte encoding for larger values
        vec![
            0x01,
            (value >> 48) as u8,
            (value >> 40) as u8,
            (value >> 32) as u8,
            (value >> 24) as u8,
            (value >> 16) as u8,
            (value >> 8) as u8,
            value as u8,
        ]
    }
}

/// Encodes a VINT size (element size).
fn encode_vint_size(size: u64) -> Vec<u8> {
    encode_vint(size)
}

/// Returns the size of a VINT encoding.
fn vint_size(value: u64) -> usize {
    if value < 0x7F {
        1
    } else if value < 0x3FFF {
        2
    } else if value < 0x1F_FFFF {
        3
    } else if value < 0x0FFF_FFFF {
        4
    } else {
        8
    }
}

/// Encodes an unsigned integer with minimal bytes.
fn encode_uint(value: u64) -> Vec<u8> {
    if value == 0 {
        vec![0]
    } else if value <= 0xFF {
        vec![value as u8]
    } else if value <= 0xFFFF {
        vec![(value >> 8) as u8, value as u8]
    } else if value <= 0xFF_FFFF {
        vec![(value >> 16) as u8, (value >> 8) as u8, value as u8]
    } else if value <= 0xFFFF_FFFF {
        vec![
            (value >> 24) as u8,
            (value >> 16) as u8,
            (value >> 8) as u8,
            value as u8,
        ]
    } else {
        vec![
            (value >> 56) as u8,
            (value >> 48) as u8,
            (value >> 40) as u8,
            (value >> 32) as u8,
            (value >> 24) as u8,
            (value >> 16) as u8,
            (value >> 8) as u8,
            value as u8,
        ]
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use oximedia_core::Timestamp;
    use oximedia_io::MemorySource;

    fn create_video_stream() -> StreamInfo {
        let mut stream = StreamInfo::new(0, CodecId::Vp9, Rational::new(1, 1000));
        stream.codec_params.width = Some(1920);
        stream.codec_params.height = Some(1080);
        stream
    }

    fn create_audio_stream() -> StreamInfo {
        let mut stream = StreamInfo::new(1, CodecId::Opus, Rational::new(1, 48000));
        stream.codec_params.sample_rate = Some(48000);
        stream.codec_params.channels = Some(2);
        stream
    }

    #[test]
    fn test_encode_element_id() {
        // 1-byte ID
        assert_eq!(encode_element_id(0xA3), vec![0xA3]);

        // 2-byte ID
        assert_eq!(encode_element_id(0x4286), vec![0x42, 0x86]);

        // 4-byte ID
        assert_eq!(encode_element_id(0x1A45_DFA3), vec![0x1A, 0x45, 0xDF, 0xA3]);
    }

    #[test]
    fn test_encode_vint() {
        assert_eq!(encode_vint(1), vec![0x81]);
        assert_eq!(encode_vint(127), vec![0xFF]);
        assert_eq!(encode_vint(128), vec![0x40, 0x80]);
    }

    #[test]
    fn test_encode_uint() {
        assert_eq!(encode_uint(0), vec![0]);
        assert_eq!(encode_uint(1), vec![1]);
        assert_eq!(encode_uint(255), vec![255]);
        assert_eq!(encode_uint(256), vec![1, 0]);
        assert_eq!(encode_uint(1_000_000), vec![0x0F, 0x42, 0x40]);
    }

    #[tokio::test]
    async fn test_muxer_new() {
        let sink = MemorySource::new_writable(4096);
        let config = MuxerConfig::new();
        let muxer = MatroskaMuxer::new(sink, config);

        assert!(!muxer.header_written);
        assert!(muxer.streams.is_empty());
    }

    #[tokio::test]
    async fn test_muxer_add_stream() {
        let sink = MemorySource::new_writable(4096);
        let config = MuxerConfig::new();
        let mut muxer = MatroskaMuxer::new(sink, config);

        let video = create_video_stream();
        let audio = create_audio_stream();

        let idx1 = muxer.add_stream(video).expect("operation should succeed");
        let idx2 = muxer.add_stream(audio).expect("operation should succeed");

        assert_eq!(idx1, 0);
        assert_eq!(idx2, 1);
        assert_eq!(muxer.streams.len(), 2);
    }

    #[tokio::test]
    async fn test_muxer_determine_format() {
        let sink = MemorySource::new_writable(4096);
        let config = MuxerConfig::new();
        let mut muxer = MatroskaMuxer::new(sink, config);

        // VP9 + Opus = WebM
        muxer
            .add_stream(create_video_stream())
            .expect("operation should succeed");
        muxer
            .add_stream(create_audio_stream())
            .expect("operation should succeed");
        muxer.determine_output_format();

        assert_eq!(muxer.output_format, ContainerFormat::WebM);

        // Add FLAC = Matroska
        let sink2 = MemorySource::new_writable(4096);
        let config2 = MuxerConfig::new();
        let mut muxer2 = MatroskaMuxer::new(sink2, config2);

        let flac_stream = StreamInfo::new(0, CodecId::Flac, Rational::new(1, 48000));
        muxer2
            .add_stream(flac_stream)
            .expect("operation should succeed");
        muxer2.determine_output_format();

        assert_eq!(muxer2.output_format, ContainerFormat::Matroska);
    }

    #[tokio::test]
    async fn test_muxer_codec_id_string() {
        assert_eq!(
            MatroskaMuxer::<MemorySource>::codec_id_string(CodecId::Av1),
            "V_AV1"
        );
        assert_eq!(
            MatroskaMuxer::<MemorySource>::codec_id_string(CodecId::Vp9),
            "V_VP9"
        );
        assert_eq!(
            MatroskaMuxer::<MemorySource>::codec_id_string(CodecId::Opus),
            "A_OPUS"
        );
        assert_eq!(
            MatroskaMuxer::<MemorySource>::codec_id_string(CodecId::Flac),
            "A_FLAC"
        );
    }

    #[tokio::test]
    async fn test_muxer_write_header() {
        let sink = MemorySource::new_writable(4096);
        let config = MuxerConfig::new()
            .with_title("Test Video")
            .with_muxing_app("TestApp");

        let mut muxer = MatroskaMuxer::new(sink, config);

        let video = create_video_stream();
        muxer.add_stream(video).expect("operation should succeed");

        let result = muxer.write_header().await;
        assert!(result.is_ok());
        assert!(muxer.header_written);
    }

    #[tokio::test]
    async fn test_muxer_write_header_no_streams() {
        let sink = MemorySource::new_writable(4096);
        let config = MuxerConfig::new();
        let mut muxer = MatroskaMuxer::new(sink, config);

        let result = muxer.write_header().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_muxer_write_packet() {
        let sink = MemorySource::new_writable(4096);
        let config = MuxerConfig::new();
        let mut muxer = MatroskaMuxer::new(sink, config);

        let video = create_video_stream();
        muxer.add_stream(video).expect("operation should succeed");
        muxer
            .write_header()
            .await
            .expect("operation should succeed");

        let packet = Packet::new(
            0,
            Bytes::from_static(&[1, 2, 3, 4]),
            Timestamp::new(0, Rational::new(1, 1000)),
            crate::PacketFlags::KEYFRAME,
        );

        let result = muxer.write_packet(&packet).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_muxer_write_trailer() {
        let sink = MemorySource::new_writable(4096);
        let config = MuxerConfig::new();
        let mut muxer = MatroskaMuxer::new(sink, config);

        let video = create_video_stream();
        muxer.add_stream(video).expect("operation should succeed");
        muxer
            .write_header()
            .await
            .expect("operation should succeed");

        let packet = Packet::new(
            0,
            Bytes::from_static(&[1, 2, 3, 4]),
            Timestamp::new(0, Rational::new(1, 1000)),
            crate::PacketFlags::KEYFRAME,
        );
        muxer
            .write_packet(&packet)
            .await
            .expect("operation should succeed");

        let result = muxer.write_trailer().await;
        assert!(result.is_ok());
    }
}
