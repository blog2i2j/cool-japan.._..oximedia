//! Matroska/`WebM` demuxer.
//!
//! This module provides a demuxer for Matroska (.mkv) and `WebM` (.webm)
//! container formats. Both use the EBML (Extensible Binary Meta Language)
//! format for structure.
//!
//! # `WebM` vs Matroska
//!
//! `WebM` is a subset of Matroska with restrictions:
//! - Video: VP8, VP9, or AV1 only
//! - Audio: Vorbis or Opus only
//! - No subtitles (except `WebVTT` in some implementations)
//!
//! # Example
//!
//! ```ignore
//! use oximedia_container::demux::MatroskaDemuxer;
//! use oximedia_io::FileSource;
//!
//! let source = FileSource::open("video.mkv").await?;
//! let mut demuxer = MatroskaDemuxer::new(source);
//!
//! let probe = demuxer.probe().await?;
//! println!("Detected: {:?}", probe.format);
//!
//! while let Ok(packet) = demuxer.read_packet().await {
//!     println!("Packet from stream {}: {} bytes",
//!              packet.stream_index, packet.size());
//! }
//! ```

pub mod ebml;
pub mod parser;
pub mod types;

use std::collections::HashMap;
use std::io::SeekFrom;

use async_trait::async_trait;
use bytes::Bytes;
use oximedia_core::{OxiError, OxiResult, Rational, Timestamp};
use oximedia_io::MediaSource;

use crate::demux::Demuxer;
use crate::{CodecParams, ContainerFormat, Metadata, Packet, PacketFlags, ProbeResult, StreamInfo};

use ebml::element_id;
use parser::MatroskaParser;
#[allow(clippy::wildcard_imports)]
use types::*;

// ============================================================================
// Constants
// ============================================================================

/// Default buffer size for reading.
const DEFAULT_BUFFER_SIZE: usize = 64 * 1024; // 64 KB

/// Maximum header size to buffer for initial parsing.
const MAX_HEADER_SIZE: usize = 16 * 1024 * 1024; // 16 MB

/// Nanoseconds per millisecond.
const NS_PER_MS: u64 = 1_000_000;

// ============================================================================
// Matroska Demuxer
// ============================================================================

/// Matroska/`WebM` demuxer.
///
/// Parses Matroska and `WebM` container formats, extracting stream
/// information and compressed packets.
pub struct MatroskaDemuxer<R> {
    /// The underlying reader.
    source: R,

    /// Read buffer.
    buffer: Vec<u8>,

    // ========================================================================
    // Parsed Metadata
    // ========================================================================
    /// EBML header.
    ebml_header: Option<EbmlHeader>,

    /// Segment information.
    segment_info: Option<SegmentInfo>,

    /// Parsed tracks.
    tracks: Vec<TrackEntry>,

    /// Cue points for seeking.
    cues: Vec<CuePoint>,

    /// Editions/chapters.
    editions: Vec<Edition>,

    /// Tags metadata.
    tags: Vec<Tag>,

    /// Seek head entries.
    seek_entries: Vec<SeekEntry>,

    // ========================================================================
    // Stream Mapping
    // ========================================================================
    /// Stream information for each track.
    streams: Vec<StreamInfo>,

    /// Map from Matroska track number to stream index.
    track_to_stream: HashMap<u64, usize>,

    // ========================================================================
    // Demuxing State
    // ========================================================================
    /// Position of segment data start (after EBML header).
    segment_start: u64,

    /// Current cluster state.
    current_cluster: Option<ClusterState>,

    /// Current position in the source.
    position: u64,

    /// Whether we've reached end of file.
    eof: bool,

    /// Whether headers have been parsed.
    header_parsed: bool,

    /// Detected format.
    format: Option<ContainerFormat>,
}

impl<R> MatroskaDemuxer<R> {
    /// Creates a new Matroska demuxer with the given source.
    ///
    /// After creation, call [`probe`](Demuxer::probe) to parse headers
    /// and detect streams.
    #[must_use]
    pub fn new(source: R) -> Self {
        Self {
            source,
            buffer: Vec::with_capacity(DEFAULT_BUFFER_SIZE),
            ebml_header: None,
            segment_info: None,
            tracks: Vec::new(),
            cues: Vec::new(),
            editions: Vec::new(),
            tags: Vec::new(),
            seek_entries: Vec::new(),
            streams: Vec::new(),
            track_to_stream: HashMap::new(),
            segment_start: 0,
            current_cluster: None,
            position: 0,
            eof: false,
            header_parsed: false,
            format: None,
        }
    }

    /// Returns a reference to the underlying source.
    #[must_use]
    pub const fn source(&self) -> &R {
        &self.source
    }

    /// Returns a mutable reference to the underlying source.
    pub fn source_mut(&mut self) -> &mut R {
        &mut self.source
    }

    /// Consumes the demuxer and returns the underlying source.
    #[must_use]
    #[allow(dead_code)]
    pub fn into_source(self) -> R {
        self.source
    }

    /// Returns the segment info if parsed.
    #[must_use]
    pub fn segment_info(&self) -> Option<&SegmentInfo> {
        self.segment_info.as_ref()
    }

    /// Returns the parsed tracks.
    #[must_use]
    pub fn tracks(&self) -> &[TrackEntry] {
        &self.tracks
    }

    /// Returns the cue points.
    #[must_use]
    pub fn cues(&self) -> &[CuePoint] {
        &self.cues
    }

    /// Returns the editions/chapters.
    #[must_use]
    pub fn editions(&self) -> &[Edition] {
        &self.editions
    }

    /// Returns the tags.
    #[must_use]
    pub fn tags(&self) -> &[Tag] {
        &self.tags
    }
}

impl<R: MediaSource> MatroskaDemuxer<R> {
    /// Reads data into the buffer, ensuring at least `min_size` bytes are available.
    ///
    /// # Errors
    ///
    /// Returns an error if reading fails.
    async fn ensure_buffer(&mut self, min_size: usize) -> OxiResult<()> {
        while self.buffer.len() < min_size && !self.eof {
            let mut temp = vec![0u8; DEFAULT_BUFFER_SIZE];
            let n = self.source.read(&mut temp).await?;
            if n == 0 {
                self.eof = true;
                break;
            }
            self.buffer.extend_from_slice(&temp[..n]);
            self.position += n as u64;
        }
        Ok(())
    }

    /// Seeks to a file position and clears the read buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if seeking fails.
    async fn seek_to_position(&mut self, position: u64) -> OxiResult<()> {
        self.source.seek(SeekFrom::Start(position)).await?;
        self.position = position;
        self.buffer.clear();
        self.eof = false;
        self.current_cluster = None;
        Ok(())
    }

    /// Consumes bytes from the buffer.
    fn consume_buffer(&mut self, n: usize) {
        self.buffer.drain(..n);
    }

    /// Reads a specific number of bytes from the source.
    ///
    /// # Errors
    ///
    /// Returns an error if reading fails or EOF is reached.
    #[allow(dead_code)]
    async fn read_bytes(&mut self, size: usize) -> OxiResult<Vec<u8>> {
        self.ensure_buffer(size).await?;
        if self.buffer.len() < size {
            return Err(OxiError::UnexpectedEof);
        }
        let data = self.buffer[..size].to_vec();
        self.consume_buffer(size);
        Ok(data)
    }

    /// Seeks to a position in the source.
    ///
    /// # Errors
    ///
    /// Returns an error if seeking fails.
    #[allow(dead_code)]
    async fn seek_to(&mut self, pos: u64) -> OxiResult<()> {
        self.source.seek(SeekFrom::Start(pos)).await?;
        self.position = pos;
        self.buffer.clear();
        self.eof = false;
        Ok(())
    }

    /// Parses the EBML header and segment elements.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing fails.
    async fn parse_headers(&mut self) -> OxiResult<()> {
        // Read enough data to parse headers
        self.ensure_buffer(MAX_HEADER_SIZE.min(1024 * 1024)).await?;

        // Parse EBML header
        let (header, consumed) = parser::parse_ebml_header(&self.buffer)?;
        self.ebml_header = Some(header);
        self.consume_buffer(consumed);

        // Determine format from DocType
        // Safety: `self.ebml_header` was just set to `Some(header)` on the line above.
        self.format = Some(
            match self
                .ebml_header
                .as_ref()
                .expect("ebml_header was set immediately before this line")
                .doc_type
            {
                DocType::WebM => ContainerFormat::WebM,
                DocType::Matroska => ContainerFormat::Matroska,
            },
        );

        // Parse segment element header
        self.ensure_buffer(12).await?;
        let mut parser = MatroskaParser::new(&self.buffer);
        let segment = parser.read_element()?;

        if segment.id != element_id::SEGMENT {
            return Err(OxiError::Parse {
                offset: self.position,
                message: format!("Expected Segment, got 0x{:X}", segment.id),
            });
        }

        self.segment_start = consumed as u64 + segment.header_size as u64;
        self.consume_buffer(segment.header_size);

        // Parse segment children
        self.parse_segment_children().await?;

        // Build stream info
        self.build_streams();

        self.header_parsed = true;
        Ok(())
    }

    /// Parses segment child elements.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing fails.
    async fn parse_segment_children(&mut self) -> OxiResult<()> {
        loop {
            self.ensure_buffer(12).await?;
            if self.buffer.is_empty() {
                break;
            }

            let mut parser = MatroskaParser::new(&self.buffer);
            let Ok(element) = parser.read_element() else {
                break;
            };

            // Stop at first cluster - we'll read packets from here
            if element.id == element_id::CLUSTER {
                break;
            }

            let header_size = element.header_size;

            // For unbounded elements, we need to scan for the next element
            if element.is_unbounded() {
                break;
            }

            #[allow(clippy::cast_possible_truncation)]
            let data_size = element.size as usize;

            // Read element data
            let total_size = header_size + data_size;
            self.ensure_buffer(total_size).await?;

            if self.buffer.len() < total_size {
                // Not enough data for this element, skip to clusters
                break;
            }

            let element_data = &self.buffer[header_size..total_size];

            match element.id {
                element_id::SEEK_HEAD => {
                    self.seek_entries = parser::parse_seek_head(element_data, element.size)?;
                }
                element_id::INFO => {
                    self.segment_info =
                        Some(parser::parse_segment_info(element_data, element.size)?);
                }
                element_id::TRACKS => {
                    self.tracks = parser::parse_tracks(element_data, element.size)?;
                }
                element_id::CUES => {
                    self.cues = parser::parse_cues(element_data, element.size)?;
                }
                element_id::CHAPTERS => {
                    self.editions = parser::parse_chapters(element_data, element.size)?;
                }
                element_id::TAGS => {
                    self.tags = parser::parse_tags(element_data, element.size)?;
                }
                // Skip attachments, void, CRC, and unknown elements
                _ => {}
            }

            self.consume_buffer(total_size);
        }

        Ok(())
    }

    /// Builds stream info from parsed tracks.
    #[allow(clippy::cast_possible_wrap)]
    fn build_streams(&mut self) {
        let timecode_scale = self
            .segment_info
            .as_ref()
            .map_or(NS_PER_MS, |info| info.timecode_scale);

        // Calculate timebase from timecode scale (nanoseconds)
        // timebase = timecode_scale / 1_000_000_000
        // Note: timecode_scale is always positive and fits in i64
        let timebase = Rational::new(timecode_scale as i64, 1_000_000_000);

        for (index, track) in self.tracks.iter().enumerate() {
            // Skip tracks without a supported codec
            let codec = match &track.oxi_codec {
                Some(c) => *c,
                None => continue, // Skip unsupported codecs
            };

            let mut stream = StreamInfo::new(index, codec, timebase);

            // Set duration if available
            if let Some(ref info) = self.segment_info {
                if let Some(duration) = info.duration {
                    // Duration is in timecode scale units
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        stream.duration = Some(duration as i64);
                    }
                }
            }

            // Set codec parameters
            match track.track_type {
                TrackType::Video => {
                    if let Some(ref video) = track.video {
                        stream.codec_params =
                            CodecParams::video(video.pixel_width, video.pixel_height);
                    }
                }
                TrackType::Audio => {
                    if let Some(ref audio) = track.audio {
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        {
                            stream.codec_params =
                                CodecParams::audio(audio.sampling_frequency as u32, audio.channels);
                        }
                    }
                }
                _ => {}
            }

            // Set extradata from codec private
            if let Some(ref private) = track.codec_private {
                stream.codec_params.extradata = Some(Bytes::copy_from_slice(private));
            }

            // Set metadata
            if let Some(ref name) = track.name {
                stream.metadata = Metadata::new().with_title(name);
            }
            if !track.language.is_empty() && track.language != "und" {
                stream.metadata = stream.metadata.with_entry("language", &track.language);
            }

            self.track_to_stream
                .insert(track.number, self.streams.len());
            self.streams.push(stream);
        }
    }

    /// Reads the next cluster or block.
    ///
    /// # Errors
    ///
    /// Returns an error if reading fails or EOF is reached.
    #[allow(clippy::cast_possible_truncation)]
    async fn read_next_block(&mut self) -> OxiResult<(Block, u64)> {
        loop {
            self.ensure_buffer(12).await?;

            // Only return EOF when buffer is empty (eof flag just means source is exhausted)
            if self.buffer.is_empty() {
                return Err(OxiError::Eof);
            }

            let mut parser = MatroskaParser::new(&self.buffer);
            let element = parser.read_element()?;

            match element.id {
                element_id::CLUSTER => {
                    // New cluster - parse timestamp
                    self.consume_buffer(element.header_size);
                    self.current_cluster = Some(ClusterState {
                        timecode: 0,
                        position: self.position,
                        size: if element.is_unbounded() {
                            None
                        } else {
                            Some(element.size)
                        },
                        data_position: 0,
                    });
                }
                element_id::TIMESTAMP => {
                    // Cluster timestamp
                    let data_size = element.size as usize;
                    let total_size = element.header_size + data_size;
                    self.ensure_buffer(total_size).await?;

                    let data = &self.buffer[element.header_size..total_size];
                    let (_, timecode) = ebml::read_uint(data).map_err(|e| OxiError::Parse {
                        offset: self.position,
                        message: format!("Failed to parse cluster timestamp: {e:?}"),
                    })?;

                    if let Some(ref mut cluster) = self.current_cluster {
                        cluster.timecode = timecode;
                    }
                    self.consume_buffer(total_size);
                }
                element_id::SIMPLE_BLOCK => {
                    // Simple block - parse and return
                    let data_size = element.size as usize;
                    let total_size = element.header_size + data_size;
                    self.ensure_buffer(total_size).await?;

                    let data = &self.buffer[element.header_size..total_size];
                    let block = parser::parse_simple_block(data)?;

                    let cluster_time = self.current_cluster.as_ref().map_or(0, |c| c.timecode);

                    self.consume_buffer(total_size);
                    return Ok((block, cluster_time));
                }
                element_id::BLOCK_GROUP => {
                    // Block group - parse and return
                    let data_size = element.size as usize;
                    let total_size = element.header_size + data_size;
                    self.ensure_buffer(total_size).await?;

                    let data = &self.buffer[element.header_size..total_size];
                    let block = parser::parse_block_group(data, element.size)?;

                    let cluster_time = self.current_cluster.as_ref().map_or(0, |c| c.timecode);

                    self.consume_buffer(total_size);
                    return Ok((block, cluster_time));
                }
                _ => {
                    // Skip unknown elements
                    if element.is_unbounded() {
                        // For unbounded elements at top level, try to find next known element
                        self.consume_buffer(element.header_size);
                    } else {
                        let total_size = element.header_size + element.size as usize;
                        self.ensure_buffer(total_size).await?;
                        self.consume_buffer(total_size.min(self.buffer.len()));
                    }
                }
            }
        }
    }

    /// Converts a block to a packet.
    #[allow(clippy::cast_possible_wrap)]
    fn block_to_packet(&self, block: &Block, cluster_time: u64) -> OxiResult<Packet> {
        // Find stream index
        let stream_index = self
            .track_to_stream
            .get(&block.header.track_number)
            .copied()
            .ok_or_else(|| OxiError::Parse {
                offset: 0,
                message: format!("Unknown track number: {}", block.header.track_number),
            })?;

        // Get stream for timebase
        let stream = &self.streams[stream_index];

        // Calculate timestamp
        // Block timecode is relative to cluster, in timecode scale units
        #[allow(clippy::cast_sign_loss)]
        let pts = cluster_time as i64 + i64::from(block.header.timecode);

        // Get frame data (use first frame if laced)
        let data = if block.frames.is_empty() {
            Bytes::new()
        } else {
            Bytes::copy_from_slice(&block.frames[0])
        };

        // Determine flags
        let mut flags = PacketFlags::empty();
        if block.is_keyframe() {
            flags |= PacketFlags::KEYFRAME;
        }
        if block.header.discardable {
            flags |= PacketFlags::DISCARD;
        }

        // Create timestamp
        let timestamp = Timestamp::new(pts, stream.timebase);

        Ok(Packet::new(stream_index, data, timestamp, flags))
    }

    /// Finds the best cue point for seeking to a target timestamp.
    ///
    /// Returns the cluster position for the cue point that is closest to
    /// (but not after) the target timestamp for the given track.
    ///
    /// # Arguments
    ///
    /// * `target_time` - Target timestamp in timecode scale units
    /// * `track_number` - Track number to seek on
    /// * `backward` - If true, find the cue point before target; otherwise after
    ///
    /// # Returns
    ///
    /// Returns `Some((cluster_position, cue_time))` if a suitable cue point is found,
    /// or `None` if no cue points are available for the track.
    #[allow(clippy::cast_possible_wrap)]
    fn find_cue_point(
        &self,
        target_time: u64,
        track_number: u64,
        backward: bool,
    ) -> Option<(u64, u64)> {
        if self.cues.is_empty() {
            return None;
        }

        let mut best_cue: Option<(u64, u64)> = None;

        for cue_point in &self.cues {
            // Find track position for this track
            let track_pos = cue_point
                .track_positions
                .iter()
                .find(|tp| tp.track == track_number)?;

            let cue_time = cue_point.time;
            let cluster_pos = self.segment_start + track_pos.cluster_position;

            if backward {
                // For backward seeks, we want the latest cue that's <= target
                if cue_time <= target_time {
                    match best_cue {
                        Some((_, best_time)) if cue_time > best_time => {
                            best_cue = Some((cluster_pos, cue_time));
                        }
                        None => {
                            best_cue = Some((cluster_pos, cue_time));
                        }
                        _ => {}
                    }
                }
            } else {
                // For forward seeks, we want the earliest cue that's >= target
                if cue_time >= target_time {
                    match best_cue {
                        Some((_, best_time)) if cue_time < best_time => {
                            best_cue = Some((cluster_pos, cue_time));
                        }
                        None => {
                            best_cue = Some((cluster_pos, cue_time));
                        }
                        _ => {}
                    }
                }
            }
        }

        best_cue
    }

    /// Performs a seek operation using cue points and cluster scanning.
    ///
    /// This implements a multi-stage seeking algorithm:
    /// 1. Use cue points to find the nearest cluster
    /// 2. Scan clusters to find the exact keyframe position
    /// 3. Position the demuxer for reading from that point
    ///
    /// # Errors
    ///
    /// Returns an error if seeking fails or the source is not seekable.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    async fn perform_seek(&mut self, target: crate::SeekTarget) -> OxiResult<()> {
        // Check if source is seekable
        if !self.source.is_seekable() {
            return Err(OxiError::unsupported("Source is not seekable"));
        }

        // Ensure headers are parsed
        if !self.header_parsed {
            return Err(OxiError::InvalidData(
                "Cannot seek before probing".to_string(),
            ));
        }

        // Determine which stream to use for seeking
        let stream_index = if let Some(idx) = target.stream_index {
            if idx >= self.streams.len() {
                return Err(OxiError::InvalidData(format!(
                    "Stream index {idx} out of range"
                )));
            }
            idx
        } else {
            // Use first video stream, or first stream if no video
            self.streams
                .iter()
                .position(StreamInfo::is_video)
                .unwrap_or(0)
        };

        // Get track number for this stream
        let track_number = self
            .tracks
            .get(stream_index)
            .map_or(1, |track| track.number);

        // Convert target position to timecode scale units
        let timecode_scale = self
            .segment_info
            .as_ref()
            .map_or(NS_PER_MS, |info| info.timecode_scale);

        // Convert from seconds to timecode units
        // target_time = (target_seconds * 1_000_000_000) / timecode_scale
        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let target_timecode = ((target.position * 1_000_000_000.0) / timecode_scale as f64) as u64;

        // Try to use cue points for efficient seeking
        if let Some((cluster_pos, _cue_time)) =
            self.find_cue_point(target_timecode, track_number, target.is_backward())
        {
            // Seek to the cluster position
            self.seek_to_position(cluster_pos).await?;

            // If we need frame-accurate seeking, scan forward to find exact position
            if target.is_frame_accurate() {
                return self
                    .scan_to_exact_position(target_timecode, stream_index)
                    .await;
            }

            return Ok(());
        }

        // No cue points available - fall back to binary search or linear scan
        // For now, just seek to the beginning as a safe fallback
        self.seek_to_position(self.segment_start).await?;

        Ok(())
    }

    /// Scans forward from current position to find exact timestamp.
    ///
    /// Used for frame-accurate seeking after positioning via cue points.
    ///
    /// # Errors
    ///
    /// Returns an error if scanning fails or the position is not found.
    async fn scan_to_exact_position(
        &mut self,
        target_timecode: u64,
        stream_index: usize,
    ) -> OxiResult<()> {
        // Read blocks until we find one with timestamp >= target
        loop {
            let result = self.read_next_block().await;

            match result {
                Ok((block, cluster_time)) => {
                    // Check if this block is from our target stream
                    if let Some(&block_stream_index) =
                        self.track_to_stream.get(&block.header.track_number)
                    {
                        if block_stream_index == stream_index {
                            // Calculate absolute timestamp
                            #[allow(clippy::cast_sign_loss)]
                            let block_time = cluster_time + block.header.timecode as u64;

                            // If we've reached or passed the target, we're done
                            if block_time >= target_timecode {
                                // We need to "put back" this block for the next read_packet call
                                // For now, we'll just return - the block will be re-read
                                return Ok(());
                            }
                        }
                    }
                }
                Err(OxiError::Eof) => {
                    // Reached end of file before finding target
                    return Ok(());
                }
                Err(e) => return Err(e),
            }
        }
    }
}

#[async_trait]
impl<R: MediaSource> Demuxer for MatroskaDemuxer<R> {
    /// Probes the format and parses container headers.
    ///
    /// # Errors
    ///
    /// Returns an error if the format cannot be detected or headers are invalid.
    async fn probe(&mut self) -> OxiResult<ProbeResult> {
        if !self.header_parsed {
            self.parse_headers().await?;
        }

        let format = self.format.unwrap_or(ContainerFormat::Matroska);
        let confidence = if self.ebml_header.is_some() && !self.tracks.is_empty() {
            0.99
        } else if self.ebml_header.is_some() {
            0.95
        } else {
            0.5
        };

        Ok(ProbeResult::new(format, confidence))
    }

    /// Reads the next packet from the container.
    ///
    /// # Errors
    ///
    /// - Returns `OxiError::Eof` when there are no more packets
    /// - Returns other errors for parse failures or I/O errors
    async fn read_packet(&mut self) -> OxiResult<Packet> {
        if !self.header_parsed {
            self.parse_headers().await?;
        }

        loop {
            let (block, cluster_time) = self.read_next_block().await?;

            // Skip blocks from unknown tracks
            if !self
                .track_to_stream
                .contains_key(&block.header.track_number)
            {
                continue;
            }

            // Skip invisible blocks
            if block.header.invisible {
                continue;
            }

            return self.block_to_packet(&block, cluster_time);
        }
    }

    /// Returns information about all streams in the container.
    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    /// Seeks to a target position in the container.
    async fn seek(&mut self, target: crate::SeekTarget) -> OxiResult<()> {
        self.perform_seek(target).await
    }

    /// Returns whether this demuxer supports seeking.
    fn is_seekable(&self) -> bool {
        self.source.is_seekable() && self.header_parsed
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_io::MemorySource;

    /// Creates a minimal valid WebM header for testing.
    fn create_webm_header() -> Vec<u8> {
        let mut data = Vec::new();

        // EBML header
        data.extend_from_slice(&[
            0x1A, 0x45, 0xDF, 0xA3, // EBML ID
            0x9F, // Size: 31 bytes
        ]);

        // EBMLVersion: 1
        data.extend_from_slice(&[0x42, 0x86, 0x81, 0x01]);

        // EBMLReadVersion: 1
        data.extend_from_slice(&[0x42, 0xF7, 0x81, 0x01]);

        // EBMLMaxIDLength: 4
        data.extend_from_slice(&[0x42, 0xF2, 0x81, 0x04]);

        // EBMLMaxSizeLength: 8
        data.extend_from_slice(&[0x42, 0xF3, 0x81, 0x08]);

        // DocType: "webm"
        data.extend_from_slice(&[0x42, 0x82, 0x84, b'w', b'e', b'b', b'm']);

        // DocTypeVersion: 4
        data.extend_from_slice(&[0x42, 0x87, 0x81, 0x04]);

        // DocTypeReadVersion: 2
        data.extend_from_slice(&[0x42, 0x85, 0x81, 0x02]);

        // Segment (unbounded size)
        data.extend_from_slice(&[
            0x18, 0x53, 0x80, 0x67, // Segment ID
            0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, // Unknown size
        ]);

        // Info element
        data.extend_from_slice(&[
            0x15, 0x49, 0xA9, 0x66, // Info ID
            0x8E, // Size: 14 bytes
        ]);

        // TimecodeScale: 1000000 (1ms)
        data.extend_from_slice(&[
            0x2A, 0xD7, 0xB1, // TimecodeScale ID
            0x83, // Size: 3 bytes
            0x0F, 0x42, 0x40, // Value: 1000000
        ]);

        // Duration: 5000.0
        data.extend_from_slice(&[
            0x44, 0x89, // Duration ID
            0x84, // Size: 4 bytes
            0x45, 0x9C, 0x40, 0x00, // Float: 5000.0
        ]);

        // Tracks element
        data.extend_from_slice(&[
            0x16, 0x54, 0xAE, 0x6B, // Tracks ID
            0x9E, // Size: 30 bytes
        ]);

        // TrackEntry
        data.extend_from_slice(&[
            0xAE, // TrackEntry ID
            0x9C, // Size: 28 bytes
        ]);

        // TrackNumber: 1
        data.extend_from_slice(&[0xD7, 0x81, 0x01]);

        // TrackUID: 12345
        data.extend_from_slice(&[0x73, 0xC5, 0x82, 0x30, 0x39]);

        // TrackType: 1 (video)
        data.extend_from_slice(&[0x83, 0x81, 0x01]);

        // CodecID: V_VP9
        data.extend_from_slice(&[0x86, 0x85, b'V', b'_', b'V', b'P', b'9']);

        // Video element
        data.extend_from_slice(&[
            0xE0, // Video ID
            0x88, // Size: 8 bytes
        ]);

        // PixelWidth: 1920
        data.extend_from_slice(&[0xB0, 0x82, 0x07, 0x80]);

        // PixelHeight: 1080
        data.extend_from_slice(&[0xBA, 0x82, 0x04, 0x38]);

        // Cluster
        data.extend_from_slice(&[
            0x1F, 0x43, 0xB6, 0x75, // Cluster ID
            0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, // Unknown size
        ]);

        // Cluster Timestamp: 0
        data.extend_from_slice(&[0xE7, 0x81, 0x00]);

        // SimpleBlock: track 1, timecode 0, keyframe
        let block_data = vec![0x01, 0x02, 0x03, 0x04]; // 4 bytes of data
        data.extend_from_slice(&[
            0xA3, // SimpleBlock ID
            0x88, // Size: 8 bytes
            0x81, // Track number: 1 (VINT)
            0x00, 0x00, // Timecode: 0
            0x80, // Flags: keyframe
        ]);
        data.extend_from_slice(&block_data);

        data
    }

    #[tokio::test]
    async fn test_matroska_demuxer_new() {
        let source = MemorySource::new(Bytes::new());
        let demuxer = MatroskaDemuxer::new(source);
        assert!(!demuxer.header_parsed);
        assert!(demuxer.streams().is_empty());
    }

    #[tokio::test]
    async fn test_matroska_demuxer_probe() {
        let data = create_webm_header();
        let source = MemorySource::new(Bytes::from(data));
        let mut demuxer = MatroskaDemuxer::new(source);

        let result = demuxer.probe().await.expect("probe should succeed");
        assert_eq!(result.format, ContainerFormat::WebM);
        assert!(result.confidence > 0.9);
        assert!(demuxer.header_parsed);
    }

    #[tokio::test]
    async fn test_matroska_demuxer_streams() {
        let data = create_webm_header();
        let source = MemorySource::new(Bytes::from(data));
        let mut demuxer = MatroskaDemuxer::new(source);

        demuxer.probe().await.expect("probe should succeed");

        let streams = demuxer.streams();
        assert!(!streams.is_empty());
        assert_eq!(streams[0].codec, oximedia_core::CodecId::Vp9);
    }

    #[tokio::test]
    async fn test_matroska_demuxer_read_packet() {
        let data = create_webm_header();
        let source = MemorySource::new(Bytes::from(data));
        let mut demuxer = MatroskaDemuxer::new(source);

        demuxer.probe().await.expect("probe should succeed");

        let packet = demuxer
            .read_packet()
            .await
            .expect("operation should succeed");
        assert_eq!(packet.stream_index, 0);
        assert!(packet.is_keyframe());
        assert_eq!(packet.size(), 4);
    }

    #[tokio::test]
    async fn test_matroska_demuxer_eof() {
        let data = create_webm_header();
        let source = MemorySource::new(Bytes::from(data));
        let mut demuxer = MatroskaDemuxer::new(source);

        demuxer.probe().await.expect("probe should succeed");

        // Read first packet
        let _ = demuxer
            .read_packet()
            .await
            .expect("operation should succeed");

        // Second read should return EOF
        let result = demuxer.read_packet().await;
        assert!(matches!(result, Err(OxiError::Eof)));
    }

    #[tokio::test]
    async fn test_segment_info() {
        let data = create_webm_header();
        let source = MemorySource::new(Bytes::from(data));
        let mut demuxer = MatroskaDemuxer::new(source);

        demuxer.probe().await.expect("probe should succeed");

        let info = demuxer.segment_info().expect("operation should succeed");
        assert_eq!(info.timecode_scale, 1_000_000);
        assert!(info.duration.is_some());
    }

    #[tokio::test]
    async fn test_tracks_info() {
        let data = create_webm_header();
        let source = MemorySource::new(Bytes::from(data));
        let mut demuxer = MatroskaDemuxer::new(source);

        demuxer.probe().await.expect("probe should succeed");

        let tracks = demuxer.tracks();
        assert!(!tracks.is_empty());
        assert_eq!(tracks[0].number, 1);
        assert_eq!(tracks[0].codec_id, "V_VP9");
        assert!(tracks[0].video.is_some());

        let video = tracks[0].video.as_ref().expect("operation should succeed");
        assert_eq!(video.pixel_width, 1920);
        assert_eq!(video.pixel_height, 1080);
    }
}
