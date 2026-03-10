//! Ogg container demuxer.
//!
//! Implements the Ogg bitstream format as specified in
//! [RFC 3533](https://www.rfc-editor.org/rfc/rfc3533).
//!
//! # Supported Codecs
//!
//! - **Opus**: High-quality, low-latency audio codec
//! - **Vorbis**: General-purpose audio codec
//! - **FLAC**: Lossless audio codec
//! - **Theora**: Video codec (limited support)
//!
//! # Example
//!
//! ```ignore
//! use oximedia_container::demux::OggDemuxer;
//! use oximedia_io::FileSource;
//!
//! let source = FileSource::open("audio.opus").await?;
//! let mut demuxer = OggDemuxer::new(source);
//!
//! let probe = demuxer.probe().await?;
//! println!("Format: {:?}", probe.format);
//!
//! for stream in demuxer.streams() {
//!     println!("Stream {}: {:?}", stream.index, stream.codec);
//! }
//!
//! while let Ok(packet) = demuxer.read_packet().await {
//!     println!("Packet: {} bytes", packet.size());
//! }
//! ```

mod page;
mod stream;

pub use page::{OggPage, PageFlags};
pub use stream::{identify_codec, LogicalStream};

use async_trait::async_trait;
use bytes::Bytes;
use oximedia_core::{OxiError, OxiResult, Rational, Timestamp};
use oximedia_io::MediaSource;
use std::collections::HashMap;

use crate::demux::Demuxer;
use crate::{CodecParams, ContainerFormat, Packet, PacketFlags, ProbeResult, StreamInfo};

/// Default buffer size for reading pages.
const READ_BUFFER_SIZE: usize = 8192;

/// Maximum number of streams to support.
const MAX_STREAMS: usize = 64;

/// Ogg demuxer.
///
/// Parses the Ogg bitstream format and extracts packets for each
/// logical stream. Supports multiplexed streams with different codecs.
///
/// # Thread Safety
///
/// The demuxer is `Send` but not `Sync`. It should be used from a single
/// task at a time.
pub struct OggDemuxer<R> {
    /// The underlying media source.
    source: R,

    /// Read buffer for page data.
    buffer: Vec<u8>,

    /// Current position in the buffer.
    buffer_pos: usize,

    /// Valid data length in the buffer.
    buffer_len: usize,

    /// Stream information for external access.
    streams: Vec<StreamInfo>,

    /// Logical stream state by serial number.
    logical_streams: HashMap<u32, LogicalStream>,

    /// Pending packets ready to be returned.
    pending_packets: Vec<Packet>,

    /// Current position in the source.
    position: u64,

    /// Whether we've reached end of file.
    eof: bool,

    /// Whether headers have been parsed.
    headers_parsed: bool,
}

impl<R> OggDemuxer<R> {
    /// Creates a new Ogg demuxer.
    ///
    /// After creation, call [`probe`](Demuxer::probe) to parse headers
    /// and detect streams.
    ///
    /// # Arguments
    ///
    /// * `source` - The media source to read from
    #[must_use]
    pub fn new(source: R) -> Self {
        Self {
            source,
            buffer: vec![0u8; READ_BUFFER_SIZE],
            buffer_pos: 0,
            buffer_len: 0,
            streams: Vec::new(),
            logical_streams: HashMap::new(),
            pending_packets: Vec::new(),
            position: 0,
            eof: false,
            headers_parsed: false,
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

    /// Returns the current stream count.
    #[must_use]
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }
}

impl<R: MediaSource> OggDemuxer<R> {
    /// Fills the buffer from the source.
    async fn fill_buffer(&mut self) -> OxiResult<usize> {
        // Move remaining data to the front
        if self.buffer_pos > 0 {
            let remaining = self.buffer_len - self.buffer_pos;
            if remaining > 0 {
                self.buffer.copy_within(self.buffer_pos..self.buffer_len, 0);
            }
            self.buffer_len = remaining;
            self.buffer_pos = 0;
        }

        // Read more data
        let bytes_read = self
            .source
            .read(&mut self.buffer[self.buffer_len..])
            .await?;
        if bytes_read == 0 {
            self.eof = true;
        }
        self.buffer_len += bytes_read;
        self.position += bytes_read as u64;

        Ok(bytes_read)
    }

    /// Ensures the buffer has at least `min_bytes` available.
    async fn ensure_buffer(&mut self, min_bytes: usize) -> OxiResult<bool> {
        while self.buffer_len - self.buffer_pos < min_bytes {
            if self.eof {
                return Ok(false);
            }
            let read = self.fill_buffer().await?;
            if read == 0 {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Reads the next Ogg page.
    async fn read_page(&mut self) -> OxiResult<Option<OggPage>> {
        // Ensure we have enough data for the page header
        if !self.ensure_buffer(page::MIN_HEADER_SIZE).await? {
            return Ok(None);
        }

        // Sync to page boundary if needed
        let sync_pos = self.sync_to_page().await?;
        if sync_pos.is_none() {
            return Ok(None);
        }

        // Read the segment count to determine page size
        let segment_count = self.buffer[self.buffer_pos + 26] as usize;
        let header_size = page::MIN_HEADER_SIZE + segment_count;

        // Ensure we have the full header
        if !self.ensure_buffer(header_size).await? {
            return Ok(None);
        }

        // Calculate data size from segment table
        let segment_table = &self.buffer[self.buffer_pos + 27..self.buffer_pos + header_size];
        let data_size: usize = segment_table.iter().map(|&s| usize::from(s)).sum();

        // Ensure we have the full page
        let page_size = header_size + data_size;
        if !self.ensure_buffer(page_size).await? {
            return Ok(None);
        }

        // Parse the page
        let page_data = &self.buffer[self.buffer_pos..self.buffer_pos + page_size];
        match OggPage::parse(page_data) {
            Ok((page, consumed)) => {
                self.buffer_pos += consumed;
                Ok(Some(page))
            }
            Err(e) => {
                // Skip this byte and try to resync
                self.buffer_pos += 1;
                Err(e)
            }
        }
    }

    /// Synchronizes to the next page boundary.
    async fn sync_to_page(&mut self) -> OxiResult<Option<usize>> {
        loop {
            // Look for OggS magic in current buffer
            let search_end = self.buffer_len.saturating_sub(3);
            for i in self.buffer_pos..search_end {
                if &self.buffer[i..i + 4] == page::OGG_MAGIC {
                    self.buffer_pos = i;
                    return Ok(Some(i));
                }
            }

            // Not found, need more data
            if self.eof {
                return Ok(None);
            }

            // Keep last 3 bytes (magic could span buffer boundary)
            let keep = 3.min(self.buffer_len - self.buffer_pos);
            self.buffer_pos = self.buffer_len - keep;

            if self.fill_buffer().await? == 0 {
                return Ok(None);
            }
        }
    }

    /// Processes a page and extracts packets.
    fn process_page(&mut self, page: &OggPage) -> OxiResult<()> {
        let serial = page.serial_number;

        // Handle BOS pages
        if page.is_bos() {
            return self.handle_bos_page(page);
        }

        // Check if stream exists
        if !self.logical_streams.contains_key(&serial) {
            return Ok(()); // Ignore unknown streams
        }

        // Extract packets from the page
        let packets = page.packets();
        let is_continuation = page.is_continuation();
        let granule_position = page.granule_position;
        let has_granule = page.has_granule();

        // Track if headers became complete during processing
        let mut headers_completed = false;

        // Collect packets to create (to avoid borrow issues)
        let mut packets_to_create = Vec::new();

        for (i, (data, complete)) in packets.into_iter().enumerate() {
            let Some(stream) = self.logical_streams.get_mut(&serial) else {
                break; // Stream was removed concurrently — should not happen
            };

            // Handle continuation from previous page
            let packet_data = if i == 0 && is_continuation {
                if stream.has_incomplete_packet() {
                    let mut buffer = stream.take_buffer();
                    buffer.extend_from_slice(&data);
                    buffer
                } else {
                    // Continuation but no buffered data - skip
                    continue;
                }
            } else {
                data
            };

            if !complete {
                // Store incomplete packet for next page
                stream.append_to_buffer(&packet_data);
                continue;
            }

            // Skip if still collecting headers
            if !stream.headers_complete() {
                stream.add_header(packet_data);
                if stream.headers_complete() {
                    headers_completed = true;
                }
                continue;
            }

            // Gather info for packet creation
            let stream_index = stream.stream_index;
            let sample_rate = stream.sample_rate();
            let last_granule = stream.last_granule;

            packets_to_create.push((stream_index, packet_data, sample_rate, last_granule));
        }

        // Update stream info if headers just completed
        if headers_completed {
            self.update_stream_info(serial);
        }

        // Create packets outside the mutable borrow scope
        for (stream_index, packet_data, sample_rate, last_granule) in packets_to_create {
            let timebase = Rational::new(1, i64::from(sample_rate.max(1)));
            let pts = if let Some(stream) = self.logical_streams.get(&serial) {
                if has_granule {
                    stream.granule_to_timebase(granule_position, timebase)
                } else {
                    stream.granule_to_timebase(last_granule, timebase)
                }
            } else {
                0i64
            };

            let packet = Packet::new(
                stream_index,
                Bytes::from(packet_data),
                Timestamp::new(pts, timebase),
                PacketFlags::KEYFRAME,
            );
            self.pending_packets.push(packet);
        }

        // Update last granule
        if has_granule {
            if let Some(stream) = self.logical_streams.get_mut(&serial) {
                stream.last_granule = granule_position;
            }
        }

        Ok(())
    }

    /// Handles a beginning-of-stream page.
    fn handle_bos_page(&mut self, page: &OggPage) -> OxiResult<()> {
        let serial = page.serial_number;

        // Skip if we already have this stream
        if self.logical_streams.contains_key(&serial) {
            return Ok(());
        }

        // Check stream limit
        if self.logical_streams.len() >= MAX_STREAMS {
            return Err(OxiError::unsupported("Too many streams in Ogg container"));
        }

        // Get the first packet (codec identification)
        let packets = page.packets();
        let first_packet = packets.first().map(|(data, _)| data.as_slice());

        let codec = first_packet
            .and_then(identify_codec)
            .ok_or_else(|| OxiError::unsupported("Unknown Ogg codec"))?;

        let stream_index = self.streams.len();
        let mut logical_stream = LogicalStream::new(serial, codec, stream_index);

        // Add the first header
        if let Some((data, _)) = packets.into_iter().next() {
            logical_stream.add_header(data);
        }

        // Create StreamInfo
        let timebase = Rational::new(1, i64::from(logical_stream.sample_rate().max(1)));
        let stream_info = StreamInfo::new(stream_index, codec, timebase);

        self.streams.push(stream_info);
        self.logical_streams.insert(serial, logical_stream);

        Ok(())
    }

    /// Updates stream info after headers are complete.
    fn update_stream_info(&mut self, serial: u32) {
        let Some(stream) = self.logical_streams.get(&serial) else {
            return;
        };

        if stream.stream_index >= self.streams.len() {
            return;
        }

        let info = &mut self.streams[stream.stream_index];

        // Update timebase with actual sample rate
        let sample_rate = stream.sample_rate();
        if sample_rate > 0 {
            info.timebase = Rational::new(1, i64::from(sample_rate));
        }

        // Update codec params
        info.codec_params = CodecParams::audio(sample_rate, 2); // Default to stereo

        // Extract channel count from headers if available
        if stream.codec == oximedia_core::CodecId::Opus {
            if let Some(header) = stream.headers.first() {
                if header.len() >= 10 {
                    info.codec_params.channels = Some(header[9]);
                }
            }
        }

        // Store codec headers as extradata
        if !stream.headers.is_empty() {
            let mut extradata = Vec::new();
            for header in &stream.headers {
                // Prepend length as 2-byte LE
                #[allow(clippy::cast_possible_truncation)]
                let len = header.len() as u16;
                extradata.extend_from_slice(&len.to_le_bytes());
                extradata.extend_from_slice(header);
            }
            info.codec_params.extradata = Some(Bytes::from(extradata));
        }
    }
}

#[async_trait]
impl<R: MediaSource> Demuxer for OggDemuxer<R> {
    async fn probe(&mut self) -> OxiResult<ProbeResult> {
        // Read pages until we have all BOS pages
        loop {
            if let Some(page) = self.read_page().await? {
                self.process_page(&page)?;

                // If this wasn't a BOS page, we're done with headers
                if !page.is_bos() {
                    break;
                }
            } else {
                // EOF during probe
                if self.streams.is_empty() {
                    return Err(OxiError::UnknownFormat);
                }
                break;
            }
        }

        // Continue reading until all headers are complete
        while !self.all_headers_complete() {
            if let Some(page) = self.read_page().await? {
                self.process_page(&page)?;
            } else {
                break;
            }
        }

        self.headers_parsed = true;

        Ok(ProbeResult::new(ContainerFormat::Ogg, 0.99))
    }

    async fn read_packet(&mut self) -> OxiResult<Packet> {
        // Return pending packets first
        if let Some(packet) = self.pending_packets.pop() {
            return Ok(packet);
        }

        // Read more pages
        loop {
            if let Some(page) = self.read_page().await? {
                self.process_page(&page)?;

                if let Some(packet) = self.pending_packets.pop() {
                    return Ok(packet);
                }
            } else {
                return Err(OxiError::Eof);
            }
        }
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    /// Seeks to a target position in the container.
    async fn seek(&mut self, target: crate::SeekTarget) -> OxiResult<()> {
        self.perform_seek(target).await
    }

    /// Returns whether this demuxer supports seeking.
    fn is_seekable(&self) -> bool {
        self.source.is_seekable() && self.headers_parsed
    }
}

impl<R: MediaSource> OggDemuxer<R> {
    /// Checks if all logical streams have complete headers.
    fn all_headers_complete(&self) -> bool {
        self.logical_streams
            .values()
            .all(LogicalStream::headers_complete)
    }

    /// Seeks to a file position.
    ///
    /// # Errors
    ///
    /// Returns an error if seeking fails.
    async fn seek_to_position(&mut self, position: u64) -> OxiResult<()> {
        use std::io::SeekFrom;
        self.source.seek(SeekFrom::Start(position)).await?;
        self.position = position;
        self.buffer_pos = 0;
        self.buffer_len = 0;
        self.eof = false;
        self.pending_packets.clear();
        Ok(())
    }

    /// Reads the granule position at a file position.
    ///
    /// Seeks to the position, reads a page, and returns the granule position.
    ///
    /// # Errors
    ///
    /// Returns an error if reading fails.
    #[allow(clippy::cast_possible_wrap)]
    async fn read_granule_at(&mut self, position: u64, serial: u32) -> OxiResult<Option<i64>> {
        let original_pos = self.source.position();

        self.seek_to_position(position).await?;

        // Read pages until we find one for the target serial
        for _ in 0..10 {
            // Limit search to avoid infinite loops
            if let Some(page) = self.read_page().await? {
                if page.serial_number == serial && page.has_granule() {
                    let granule = page.granule_position as i64;
                    // Restore position
                    self.seek_to_position(original_pos).await?;
                    return Ok(Some(granule));
                }
            } else {
                break;
            }
        }

        // Restore position
        self.seek_to_position(original_pos).await?;
        Ok(None)
    }

    /// Performs bisection search to find the target granule position.
    ///
    /// Uses binary search over file positions to find the page closest to
    /// the target granule position.
    ///
    /// # Arguments
    ///
    /// * `target_granule` - Target granule position
    /// * `serial` - Serial number of the logical stream
    /// * `file_size` - Total file size in bytes
    ///
    /// # Errors
    ///
    /// Returns an error if seeking or reading fails.
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    async fn bisect_seek(
        &mut self,
        target_granule: i64,
        serial: u32,
        file_size: u64,
    ) -> OxiResult<()> {
        let mut left = 0u64;
        let mut right = file_size;
        let mut best_pos = 0u64;

        // Binary search for the position
        while left < right {
            let mid = left + (right - left) / 2;

            // Align to page boundary (search backwards for OggS)
            let aligned_mid = self.align_to_page(mid).await?;

            // Read granule at this position
            match self.read_granule_at(aligned_mid, serial).await? {
                Some(granule) => match granule.cmp(&target_granule) {
                    std::cmp::Ordering::Less => {
                        // Target is after this position
                        best_pos = aligned_mid;
                        left = mid + 1;
                    }
                    std::cmp::Ordering::Greater => {
                        // Target is before this position
                        right = mid;
                    }
                    std::cmp::Ordering::Equal => {
                        // Exact match!
                        best_pos = aligned_mid;
                        break;
                    }
                },
                None => {
                    // Could not read granule, move search window
                    right = mid;
                }
            }

            // Prevent infinite loop
            if right.saturating_sub(left) < 4096 {
                break;
            }
        }

        // Seek to the best position found
        self.seek_to_position(best_pos).await?;
        Ok(())
    }

    /// Aligns a file position to the nearest preceding page boundary.
    ///
    /// Searches backwards from the given position to find an Ogg page header.
    ///
    /// # Errors
    ///
    /// Returns an error if seeking fails.
    async fn align_to_page(&mut self, position: u64) -> OxiResult<u64> {
        use std::io::SeekFrom;

        // Search backwards up to 64KB for a page header
        let search_start = position.saturating_sub(65536);

        self.source.seek(SeekFrom::Start(search_start)).await?;

        let mut search_buf = vec![0u8; (position - search_start).min(65536) as usize];
        let n = self.source.read(&mut search_buf).await?;

        // Search for "OggS" magic from the end
        for i in (0..n.saturating_sub(4)).rev() {
            if &search_buf[i..i + 4] == page::OGG_MAGIC {
                return Ok(search_start + i as u64);
            }
        }

        // If not found, return original position
        Ok(position)
    }

    /// Performs seek operation using bisection method.
    ///
    /// This implements granule position-based seeking for Ogg streams.
    ///
    /// # Errors
    ///
    /// Returns an error if seeking fails or the source is not seekable.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    async fn perform_seek(&mut self, target: crate::SeekTarget) -> OxiResult<()> {
        use oximedia_core::OxiError;

        // Check if source is seekable
        if !self.source.is_seekable() {
            return Err(OxiError::unsupported("Source is not seekable"));
        }

        // Ensure headers are parsed
        if !self.headers_parsed {
            return Err(OxiError::InvalidData(
                "Cannot seek before probing".to_string(),
            ));
        }

        if self.streams.is_empty() {
            return Err(OxiError::InvalidData("No streams available".to_string()));
        }

        // Determine which stream to use for seeking
        let stream_index = target.stream_index.unwrap_or(0);
        if stream_index >= self.streams.len() {
            return Err(OxiError::InvalidData(format!(
                "Stream index {stream_index} out of range"
            )));
        }

        // Find the logical stream by stream index
        let (serial, logical_stream) = self
            .logical_streams
            .iter()
            .find(|(_, ls)| ls.stream_index == stream_index)
            .ok_or_else(|| OxiError::InvalidData("Stream not found".to_string()))?;

        let serial = *serial;
        let sample_rate = logical_stream.sample_rate();

        // Convert target time to granule position
        // granule = time * sample_rate
        #[allow(clippy::cast_possible_truncation)]
        let target_granule = (target.position * f64::from(sample_rate)) as i64;

        // Get file size
        let file_size = self.source.len().unwrap_or(u64::MAX);

        // Perform bisection seek
        self.bisect_seek(target_granule, serial, file_size).await?;

        // Clear any pending packets
        self.pending_packets.clear();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_io::MemorySource;

    /// Creates a minimal valid Ogg page.
    fn create_ogg_page(serial: u32, flags: PageFlags, data: &[u8]) -> Vec<u8> {
        let mut page = Vec::new();
        page.extend_from_slice(page::OGG_MAGIC); // Magic
        page.push(0); // Version
        page.push(flags.bits()); // Flags
        page.extend_from_slice(&0u64.to_le_bytes()); // Granule position
        page.extend_from_slice(&serial.to_le_bytes()); // Serial
        page.extend_from_slice(&0u32.to_le_bytes()); // Sequence
        page.extend_from_slice(&0u32.to_le_bytes()); // CRC

        // Segment table
        let segments: Vec<u8> = if data.is_empty() {
            vec![0]
        } else {
            let mut segs = Vec::new();
            let mut remaining = data.len();
            while remaining > 255 {
                segs.push(255);
                remaining -= 255;
            }
            #[allow(clippy::cast_possible_truncation)]
            segs.push(remaining as u8);
            segs
        };

        #[allow(clippy::cast_possible_truncation)]
        page.push(segments.len() as u8);
        page.extend_from_slice(&segments);
        page.extend_from_slice(data);

        page
    }

    /// Creates an Opus identification header.
    fn create_opus_head() -> Vec<u8> {
        vec![
            b'O', b'p', b'u', b's', b'H', b'e', b'a', b'd', // Magic
            1,    // Version
            2,    // Channels
            0x00, 0x00, // Pre-skip
            0x80, 0xBB, 0x00, 0x00, // Sample rate (48000)
            0x00, 0x00, // Gain
            0,    // Mapping family
        ]
    }

    /// Creates an Opus comment header.
    fn create_opus_tags() -> Vec<u8> {
        let mut tags = vec![b'O', b'p', b'u', b's', b'T', b'a', b'g', b's'];
        // Vendor string length (0)
        tags.extend_from_slice(&0u32.to_le_bytes());
        // Comment count (0)
        tags.extend_from_slice(&0u32.to_le_bytes());
        tags
    }

    #[tokio::test]
    async fn test_ogg_demuxer_new() {
        let source = MemorySource::new(Bytes::new());
        let demuxer = OggDemuxer::new(source);
        assert!(!demuxer.headers_parsed);
        assert!(demuxer.streams.is_empty());
    }

    #[tokio::test]
    async fn test_ogg_demuxer_probe_empty() {
        let source = MemorySource::new(Bytes::new());
        let mut demuxer = OggDemuxer::new(source);

        let result = demuxer.probe().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_ogg_demuxer_probe_invalid() {
        let source = MemorySource::new(Bytes::from_static(b"not an ogg file"));
        let mut demuxer = OggDemuxer::new(source);

        let result = demuxer.probe().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_ogg_demuxer_probe_opus() {
        // Create a minimal Opus stream
        let mut data = Vec::new();

        // BOS page with OpusHead
        let opus_head = create_opus_head();
        data.extend(create_ogg_page(1, PageFlags::BOS, &opus_head));

        // Second header page with OpusTags
        let opus_tags = create_opus_tags();
        data.extend(create_ogg_page(1, PageFlags::empty(), &opus_tags));

        let source = MemorySource::new(Bytes::from(data));
        let mut demuxer = OggDemuxer::new(source);

        let result = demuxer.probe().await;
        assert!(result.is_ok());

        let probe = result.expect("operation should succeed");
        assert_eq!(probe.format, ContainerFormat::Ogg);

        // Check stream info
        assert_eq!(demuxer.streams.len(), 1);
        assert_eq!(demuxer.streams[0].codec, oximedia_core::CodecId::Opus);
    }

    #[tokio::test]
    async fn test_ogg_demuxer_read_packet() {
        // Create a stream with headers and one data packet
        let mut data = Vec::new();

        // BOS page with OpusHead
        let opus_head = create_opus_head();
        data.extend(create_ogg_page(1, PageFlags::BOS, &opus_head));

        // Header page with OpusTags
        let opus_tags = create_opus_tags();
        data.extend(create_ogg_page(1, PageFlags::empty(), &opus_tags));

        // Data packet
        let audio_data = vec![0u8; 100];
        data.extend(create_ogg_page(1, PageFlags::empty(), &audio_data));

        let source = MemorySource::new(Bytes::from(data));
        let mut demuxer = OggDemuxer::new(source);

        // Probe first
        demuxer.probe().await.expect("probe should succeed");

        // Read packet
        let result = demuxer.read_packet().await;
        assert!(result.is_ok());

        let packet = result.expect("operation should succeed");
        assert_eq!(packet.stream_index, 0);
        assert_eq!(packet.size(), 100);
    }

    #[tokio::test]
    async fn test_ogg_demuxer_eof() {
        let mut data = Vec::new();

        // Create minimal valid stream
        let opus_head = create_opus_head();
        data.extend(create_ogg_page(1, PageFlags::BOS, &opus_head));

        let opus_tags = create_opus_tags();
        data.extend(create_ogg_page(1, PageFlags::empty(), &opus_tags));

        let source = MemorySource::new(Bytes::from(data));
        let mut demuxer = OggDemuxer::new(source);

        demuxer.probe().await.expect("probe should succeed");

        // Should return EOF when no more packets
        let result = demuxer.read_packet().await;
        assert!(matches!(result, Err(OxiError::Eof)));
    }

    #[test]
    fn test_stream_count() {
        let source = MemorySource::new(Bytes::new());
        let demuxer = OggDemuxer::new(source);
        assert_eq!(demuxer.stream_count(), 0);
    }
}
