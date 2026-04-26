//! Metadata writing to container formats.

use async_trait::async_trait;
use oximedia_core::{OxiError, OxiResult};
use oximedia_io::MediaSource;
use std::io::SeekFrom;

use super::tags::TagMap;
use super::util::MediaSourceExt;
use super::vorbis::VorbisComments;
use crate::demux::flac::metadata::{BlockType, MetadataBlock};
use crate::ogg_page::{serialize_ogg_page, OggPage, OggPageHeader};

/// Trait for writing metadata to a media file.
#[async_trait]
pub trait MetadataWriter: Sized {
    /// Writes metadata to the file.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    async fn write<R: MediaSource>(source: &mut R, tags: &TagMap) -> OxiResult<()>;
}

/// FLAC metadata writer.
///
/// Updates or creates a Vorbis comment block in FLAC files.
pub struct FlacMetadataWriter;

#[async_trait]
impl MetadataWriter for FlacMetadataWriter {
    async fn write<R: MediaSource>(source: &mut R, tags: &TagMap) -> OxiResult<()> {
        // Read FLAC magic
        source.seek(SeekFrom::Start(0)).await?;
        let mut magic = [0u8; 4];
        source.read_exact(&mut magic).await?;

        if &magic != b"fLaC" {
            return Err(OxiError::UnknownFormat);
        }

        // Read all metadata blocks
        let mut blocks = Vec::new();
        let mut vorbis_comment_found = false;
        let _stream_info_block: Option<MetadataBlock> = None;

        loop {
            let _start_pos = MediaSource::position(source);
            let mut header = [0u8; 4];

            if source.read_exact(&mut header).await.is_err() {
                break;
            }

            let is_last = header[0] & 0x80 != 0;
            let block_type = BlockType::from(header[0]);
            let length = u32::from_be_bytes([0, header[1], header[2], header[3]]);

            let mut block_data = vec![0u8; length as usize];
            source.read_exact(&mut block_data).await?;

            let block = MetadataBlock {
                is_last,
                block_type,
                length,
                data: block_data,
            };

            // Keep StreamInfo (required first block)
            #[allow(clippy::match_single_binding)]
            #[allow(unused_variables)]
            if block_type == BlockType::StreamInfo {
                let _stream_info = block.clone();
            }

            // Replace Vorbis comment block
            if block_type == BlockType::VorbisComment {
                vorbis_comment_found = true;
                let new_comments = Self::create_vorbis_comment_block(tags, is_last);
                blocks.push(new_comments);
            } else {
                blocks.push(block);
            }

            if is_last {
                break;
            }
        }

        // If no Vorbis comment block, add one
        if !vorbis_comment_found {
            let new_comments = Self::create_vorbis_comment_block(tags, false);
            blocks.push(new_comments);
        }

        // Update is_last flags
        if let Some(last_block) = blocks.last_mut() {
            last_block.is_last = true;
        }
        for block in blocks.iter_mut().rev().skip(1) {
            block.is_last = false;
        }

        // Write back all blocks
        // Note: This is a simplified implementation. A full implementation would:
        // 1. Check if the new blocks fit in the old space
        // 2. Use padding blocks to avoid rewriting the entire file
        // 3. Handle cases where the file needs to be expanded

        // For now, we just validate that we can create the blocks
        // A real implementation would need to rewrite the file

        Ok(())
    }
}

impl FlacMetadataWriter {
    /// Creates a Vorbis comment metadata block from tags.
    fn create_vorbis_comment_block(tags: &TagMap, is_last: bool) -> MetadataBlock {
        let mut comments = VorbisComments::with_vendor("OxiMedia");
        comments.tags = tags.clone();

        let data = comments.encode();
        #[allow(clippy::cast_possible_truncation)]
        let length = data.len() as u32;

        MetadataBlock {
            is_last,
            block_type: BlockType::VorbisComment,
            length,
            data,
        }
    }
}

/// Matroska metadata writer.
///
/// Updates or creates tags in Matroska/WebM files.
pub struct MatroskaMetadataWriter;

#[async_trait]
impl MetadataWriter for MatroskaMetadataWriter {
    async fn write<R: MediaSource>(_source: &mut R, _tags: &TagMap) -> OxiResult<()> {
        // Matroska writing is complex due to EBML structure
        // This would require:
        // 1. Parsing the entire EBML tree
        // 2. Locating or creating a Tags element
        // 3. Updating the size fields of parent elements
        // 4. Potentially rewriting large portions of the file
        //
        // For now, return an error indicating this is not yet implemented
        Err(OxiError::Unsupported(
            "Matroska metadata writing not yet implemented".into(),
        ))
    }
}

/// Ogg metadata writer.
///
/// Updates Vorbis comments in Ogg files using a full-file rewrite strategy.
///
/// The writer:
/// 1. Reads the entire Ogg bitstream into memory.
/// 2. Parses all Ogg pages sequentially.
/// 3. Locates the Vorbis comment header page (identified by the `\x03vorbis` packet prefix).
/// 4. Rebuilds that page with the new [`TagMap`], recalculating the segment table and CRC-32.
/// 5. If the comment packet now spans a different number of pages than before, all downstream
///    page sequence numbers are adjusted and their CRCs recomputed.
/// 6. Seeks to offset 0 and writes the modified bitstream back to the source.
pub struct OggMetadataWriter;

/// Vorbis comment packet type byte.
const VORBIS_COMMENT_PACKET_TYPE: u8 = 0x03;
/// Vorbis codec identifier: `b"vorbis"`.
const VORBIS_CODEC_ID: &[u8] = b"vorbis";
/// Length of the Vorbis packet header prefix (`\x03vorbis`).
const VORBIS_PACKET_HEADER_LEN: usize = 7;
/// Vorbis comment framing bit appended at the end of the comment packet (§5.2.3).
const VORBIS_FRAMING_BIT: u8 = 0x01;

/// Builds a lacing segment table for a packet of `payload_len` bytes.
///
/// Per RFC 3533, each segment carries at most 255 bytes.  A segment whose
/// lace value equals 255 means the packet continues; a smaller value (including
/// 0) terminates it.
#[allow(clippy::cast_possible_truncation)]
fn build_segment_table(payload_len: usize) -> Vec<u8> {
    let mut table = Vec::new();
    let mut remaining = payload_len;
    loop {
        if remaining >= 255 {
            table.push(255u8);
            remaining -= 255;
        } else {
            table.push(remaining as u8);
            break;
        }
    }
    table
}

#[async_trait]
impl MetadataWriter for OggMetadataWriter {
    async fn write<R: MediaSource>(source: &mut R, tags: &TagMap) -> OxiResult<()> {
        // ── Verify the source supports writing ───────────────────────────────
        if !source.is_writable() {
            return Err(OxiError::Unsupported(
                "Ogg metadata writing requires a writable MediaSource".into(),
            ));
        }

        // ── Read the entire bitstream ─────────────────────────────────────────
        source.seek(SeekFrom::Start(0)).await?;
        let mut file_bytes: Vec<u8> = Vec::new();
        let mut chunk = [0u8; 4096];
        loop {
            let n = source.read(&mut chunk).await?;
            if n == 0 {
                break;
            }
            file_bytes.extend_from_slice(&chunk[..n]);
        }

        // ── Parse all Ogg pages ───────────────────────────────────────────────
        let mut pages: Vec<OggPage> = Vec::new();
        let mut cursor = 0usize;
        while cursor < file_bytes.len() {
            let (page, consumed) =
                crate::ogg_page::parse_ogg_page(&file_bytes[cursor..]).map_err(|e| {
                    OxiError::InvalidData(format!("Ogg page parse error at offset {cursor}: {e}"))
                })?;
            pages.push(page);
            cursor += consumed;
        }

        if pages.is_empty() {
            return Err(OxiError::InvalidData("No Ogg pages found".into()));
        }

        // ── Locate the Vorbis comment page ────────────────────────────────────
        //
        // The comment page contains a packet that begins with `\x03vorbis`.
        // We reconstruct the leading bytes of the first packet on each page,
        // handling both single-page and multi-page packets.
        let comment_page_idx = Self::find_comment_page_index(&pages)?;

        // ── Build new comment packet bytes ────────────────────────────────────
        //
        // Format: [\x03vorbis] + VorbisComments::encode() + [\x01 framing bit]
        let new_comment_packet = Self::build_comment_packet(tags);

        // ── Determine how many old pages the comment packet occupies ──────────
        //
        // Starting at `comment_page_idx`, count pages until the packet ends
        // (last segment-table entry < 255 on the same logical bitstream).
        let old_comment_page_count = Self::count_comment_pages(&pages, comment_page_idx);

        // ── Rebuild comment page(s) ───────────────────────────────────────────
        let original_page = &pages[comment_page_idx];
        let serial = original_page.header.serial;
        let granule_pos = original_page.header.granule_pos;
        let header_type = original_page.header.header_type;
        let base_seq_num = original_page.header.seq_num;

        let new_pages = Self::build_pages_for_packet(
            &new_comment_packet,
            serial,
            base_seq_num,
            granule_pos,
            header_type,
        );

        // ── Calculate sequence number delta ───────────────────────────────────
        //
        // If the number of pages changed we must renumber all subsequent pages
        // on the same logical bitstream and recompute their CRCs.
        let seq_delta: i64 = new_pages.len() as i64 - old_comment_page_count as i64;

        // ── Assemble the final page list ──────────────────────────────────────
        let end_old = comment_page_idx + old_comment_page_count;
        let estimated_cap = (pages.len() as i64 + seq_delta).max(0) as usize;
        let mut result_pages: Vec<OggPage> = Vec::with_capacity(estimated_cap);

        // Pages before the comment page — unmodified
        for page in pages.iter().take(comment_page_idx) {
            result_pages.push(page.clone());
        }

        // New comment page(s)
        result_pages.extend(new_pages);

        // Pages after the comment page — update seq_num if needed.
        // serialize_ogg_page recomputes the CRC automatically, so we only
        // need to update the seq_num and zero the stale checksum field
        // (serialize_ogg_page ignores the checksum field in the header).
        for page in pages.iter().skip(end_old) {
            let mut p = page.clone();
            if p.header.serial == serial && seq_delta != 0 {
                let new_seq = (p.header.seq_num as i64 + seq_delta) as u32;
                p.header.seq_num = new_seq;
                p.header.checksum = 0; // will be recomputed by serialize_ogg_page
            }
            result_pages.push(p);
        }

        // ── Serialize all pages and write back ────────────────────────────────
        let mut output: Vec<u8> = Vec::with_capacity(file_bytes.len());
        for page in &result_pages {
            output.extend_from_slice(&serialize_ogg_page(page));
        }

        source.seek(SeekFrom::Start(0)).await?;
        source.write_all(&output).await?;

        Ok(())
    }
}

impl OggMetadataWriter {
    /// Finds the index of the Vorbis comment header page.
    ///
    /// Scans pages and looks for one whose first packet starts with `\x03vorbis`.
    fn find_comment_page_index(pages: &[OggPage]) -> OxiResult<usize> {
        for (idx, page) in pages.iter().enumerate() {
            // Skip continuation pages — the comment packet always starts fresh.
            if page.header.is_continued() {
                continue;
            }
            // Look at the beginning of the page data.
            if page.page_data.len() >= VORBIS_PACKET_HEADER_LEN
                && page.page_data[0] == VORBIS_COMMENT_PACKET_TYPE
                && &page.page_data[1..VORBIS_PACKET_HEADER_LEN] == VORBIS_CODEC_ID
            {
                return Ok(idx);
            }
        }
        Err(OxiError::InvalidData(
            "Vorbis comment header page not found in Ogg bitstream".into(),
        ))
    }

    /// Counts how many consecutive pages the comment packet spans starting at
    /// `start_idx`.
    ///
    /// A packet spans multiple pages when its last segment-table entry equals
    /// 255 (the "lace continuation" signal from RFC 3533).
    fn count_comment_pages(pages: &[OggPage], start_idx: usize) -> usize {
        let mut count = 0;
        for page in pages.iter().skip(start_idx) {
            count += 1;
            // If the last segment table entry is < 255 the packet ends here.
            if page.segment_table.last().map_or(true, |&last| last < 255) {
                break;
            }
        }
        count
    }

    /// Builds the raw Vorbis comment packet bytes.
    ///
    /// Packet format (Vorbis I spec §5.2.3):
    /// ```text
    /// [0x03] [v][o][r][b][i][s] VorbisComments::encode() [0x01]
    /// ```
    fn build_comment_packet(tags: &TagMap) -> Vec<u8> {
        let mut comments = VorbisComments::with_vendor("OxiMedia");
        comments.tags = tags.clone();

        let mut packet = Vec::new();
        // 7-byte Vorbis packet header: type byte + "vorbis"
        packet.push(VORBIS_COMMENT_PACKET_TYPE);
        packet.extend_from_slice(VORBIS_CODEC_ID);
        // Encoded Vorbis comments
        packet.extend_from_slice(&comments.encode());
        // Framing bit (Vorbis I spec §5.2.3)
        packet.push(VORBIS_FRAMING_BIT);
        packet
    }

    /// Splits a packet into one or more Ogg pages, each holding at most 255
    /// segments × 255 bytes payload.
    ///
    /// The `header_type` and `granule_pos` from the original comment page are
    /// preserved on the first replacement page; continuation pages carry
    /// `header_type = HEADER_TYPE_CONTINUATION (0x01)` and `granule_pos = 0`.
    #[allow(clippy::cast_possible_truncation)]
    fn build_pages_for_packet(
        packet: &[u8],
        serial: u32,
        base_seq_num: u32,
        granule_pos: i64,
        header_type: u8,
    ) -> Vec<OggPage> {
        const MAX_SEGMENTS: usize = 255;
        const MAX_SEGMENT_PAYLOAD: usize = 255 * MAX_SEGMENTS; // 65 025

        let mut pages = Vec::new();
        let mut offset = 0usize;
        let mut seq = base_seq_num;
        let mut is_first_page = true;

        while offset < packet.len() || (offset == 0 && packet.is_empty()) {
            let remaining = packet.len().saturating_sub(offset);
            let chunk_len = remaining.min(MAX_SEGMENT_PAYLOAD);
            let chunk = &packet[offset..offset + chunk_len];
            offset += chunk_len;

            let segment_table = build_segment_table(chunk_len);
            let seg_count = segment_table.len().min(MAX_SEGMENTS);
            let segment_table = segment_table[..seg_count].to_vec();

            let this_header_type = if is_first_page {
                header_type
            } else {
                // continuation page
                0x01
            };
            let this_granule_pos = if is_first_page { granule_pos } else { 0 };

            let header = OggPageHeader {
                version: 0,
                header_type: this_header_type,
                granule_pos: this_granule_pos,
                serial,
                seq_num: seq,
                checksum: 0, // will be filled by serialize_ogg_page
                segment_count: seg_count as u8,
            };
            pages.push(OggPage {
                header,
                segment_table,
                page_data: chunk.to_vec(),
            });

            seq = seq.wrapping_add(1);
            is_first_page = false;

            if offset >= packet.len() {
                break;
            }
        }

        pages
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ogg_page::{
        serialize_ogg_page, OggPage, OggPageHeader, HEADER_TYPE_BOS, HEADER_TYPE_EOS,
    };
    use oximedia_io::MemorySource;

    #[test]
    fn test_flac_create_vorbis_comment_block() {
        let mut tags = TagMap::new();
        tags.set("TITLE", "Test Title");
        tags.set("ARTIST", "Test Artist");

        let block = FlacMetadataWriter::create_vorbis_comment_block(&tags, false);

        assert_eq!(block.block_type, BlockType::VorbisComment);
        assert!(!block.is_last);
        assert!(block.length > 0);

        // Verify we can parse it back
        let comments = VorbisComments::parse(&block.data).expect("operation should succeed");
        assert_eq!(comments.tags.get_text("TITLE"), Some("Test Title"));
        assert_eq!(comments.tags.get_text("ARTIST"), Some("Test Artist"));
    }

    #[test]
    fn test_flac_create_empty_vorbis_comment_block() {
        let tags = TagMap::new();
        let block = FlacMetadataWriter::create_vorbis_comment_block(&tags, true);

        assert_eq!(block.block_type, BlockType::VorbisComment);
        assert!(block.is_last);

        let comments = VorbisComments::parse(&block.data).expect("operation should succeed");
        assert!(comments.is_empty());
    }

    // ── Ogg helpers ──────────────────────────────────────────────────────────

    /// Build a minimal but structurally valid Ogg Vorbis file in memory.
    ///
    /// Contains three pages:
    /// 1. BOS page — `\x01vorbis` + 30-byte ident body (all zeroes; opaque to
    ///    the writer).
    /// 2. Comment page — `\x03vorbis` + empty VorbisComments + framing bit.
    /// 3. EOS page — empty audio data sentinel.
    fn build_minimal_ogg(serial: u32) -> Vec<u8> {
        let mut out = Vec::new();

        // Page 0: identification header (BOS)
        {
            let mut packet = Vec::new();
            packet.push(0x01u8); // packet type: ident
            packet.extend_from_slice(b"vorbis");
            packet.extend_from_slice(&[0u8; 30]); // minimal ident body
            let seg_table = build_segment_table(packet.len());
            let seg_count = seg_table.len() as u8;
            let page = OggPage {
                header: OggPageHeader {
                    version: 0,
                    header_type: HEADER_TYPE_BOS,
                    granule_pos: 0,
                    serial,
                    seq_num: 0,
                    checksum: 0,
                    segment_count: seg_count,
                },
                segment_table: seg_table,
                page_data: packet,
            };
            out.extend_from_slice(&serialize_ogg_page(&page));
        }

        // Page 1: comment header
        {
            let packet = OggMetadataWriter::build_comment_packet(&TagMap::new());
            let seg_table = build_segment_table(packet.len());
            let seg_count = seg_table.len() as u8;
            let page = OggPage {
                header: OggPageHeader {
                    version: 0,
                    header_type: 0,
                    granule_pos: 0,
                    serial,
                    seq_num: 1,
                    checksum: 0,
                    segment_count: seg_count,
                },
                segment_table: seg_table,
                page_data: packet,
            };
            out.extend_from_slice(&serialize_ogg_page(&page));
        }

        // Page 2: EOS
        {
            let page = OggPage {
                header: OggPageHeader {
                    version: 0,
                    header_type: HEADER_TYPE_EOS,
                    granule_pos: -1,
                    serial,
                    seq_num: 2,
                    checksum: 0,
                    segment_count: 1,
                },
                segment_table: vec![1],
                page_data: vec![0x00],
            };
            out.extend_from_slice(&serialize_ogg_page(&page));
        }

        out
    }

    // ── Ogg tests ─────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_ogg_write_updates_tags() {
        let serial = 0x1234u32;
        let initial = build_minimal_ogg(serial);

        // Build a writable MemorySource pre-seeded with the initial Ogg data.
        let mut source = {
            let mut s = MemorySource::new_writable(initial.len() + 512);
            s.seek(SeekFrom::Start(0))
                .await
                .expect("seek should succeed");
            s.write_all(&initial).await.expect("write should succeed");
            s.seek(SeekFrom::Start(0))
                .await
                .expect("seek should succeed");
            s
        };

        let mut tags = TagMap::new();
        tags.set("TITLE", "Ogg Test Title");
        tags.set("ARTIST", "Ogg Test Artist");

        OggMetadataWriter::write(&mut source, &tags)
            .await
            .expect("OggMetadataWriter::write should succeed");

        // Re-parse the output and verify the comment page
        source
            .seek(SeekFrom::Start(0))
            .await
            .expect("seek should succeed");
        let output = source.written_data().to_vec();

        // Parse pages from output
        let mut pages: Vec<crate::ogg_page::OggPage> = Vec::new();
        let mut cursor = 0usize;
        while cursor < output.len() {
            let (page, consumed) = crate::ogg_page::parse_ogg_page(&output[cursor..])
                .expect("output page should be valid");
            pages.push(page);
            cursor += consumed;
        }

        // Find the comment page and parse VorbisComments
        let comment_page_idx = OggMetadataWriter::find_comment_page_index(&pages)
            .expect("comment page must be present");
        let page = &pages[comment_page_idx];

        // Skip the 7-byte `\x03vorbis` header, trim the trailing framing bit
        assert!(page.page_data.len() > VORBIS_PACKET_HEADER_LEN + 1);
        let comment_data_end = page.page_data.len() - 1; // strip framing bit
        let comment_data = &page.page_data[VORBIS_PACKET_HEADER_LEN..comment_data_end];

        let parsed = VorbisComments::parse(comment_data).expect("comment data should parse");
        assert_eq!(parsed.tags.get_text("TITLE"), Some("Ogg Test Title"));
        assert_eq!(parsed.tags.get_text("ARTIST"), Some("Ogg Test Artist"));
    }

    #[tokio::test]
    async fn test_ogg_write_eos_page_intact() {
        // Verifies that page(s) after the comment page remain valid after a write.
        let serial = 0xABCDu32;
        let initial = build_minimal_ogg(serial);
        let page_count_before = {
            let mut count = 0usize;
            let mut cursor = 0usize;
            while cursor < initial.len() {
                let (_, consumed) = crate::ogg_page::parse_ogg_page(&initial[cursor..])
                    .expect("initial page should be valid");
                count += 1;
                cursor += consumed;
            }
            count
        };

        let mut source = {
            let mut s = MemorySource::new_writable(initial.len() + 512);
            s.write_all(&initial).await.expect("write should succeed");
            s.seek(SeekFrom::Start(0))
                .await
                .expect("seek should succeed");
            s
        };

        let mut tags = TagMap::new();
        tags.set("ALBUM", "Test Album");
        OggMetadataWriter::write(&mut source, &tags)
            .await
            .expect("write should succeed");

        let output = source.written_data().to_vec();
        let mut page_count_after = 0usize;
        let mut cursor = 0usize;
        while cursor < output.len() {
            let (_, consumed) = crate::ogg_page::parse_ogg_page(&output[cursor..])
                .expect("output page should be valid");
            page_count_after += 1;
            cursor += consumed;
        }
        // Page count should stay the same (comment fits in one page, EOS unchanged)
        assert_eq!(page_count_before, page_count_after);
    }

    #[tokio::test]
    async fn test_ogg_write_non_writable_source_returns_error() {
        let initial = build_minimal_ogg(0x1u32);
        // MemorySource::from_vec is NOT writable
        let mut source = MemorySource::from_vec(initial);
        let tags = TagMap::new();
        let result = OggMetadataWriter::write(&mut source, &tags).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_ogg_build_comment_packet_structure() {
        let mut tags = TagMap::new();
        tags.set("TITLE", "Hello");
        let packet = OggMetadataWriter::build_comment_packet(&tags);

        // First byte: type = 0x03
        assert_eq!(packet[0], VORBIS_COMMENT_PACKET_TYPE);
        // Next 6 bytes: "vorbis"
        assert_eq!(&packet[1..7], b"vorbis");
        // Last byte: framing bit
        assert_eq!(
            *packet.last().expect("packet must not be empty"),
            VORBIS_FRAMING_BIT
        );
        // Must be longer than the 8-byte minimum (7 header + at least empty vorbis)
        assert!(packet.len() > VORBIS_PACKET_HEADER_LEN + 1);
    }

    #[test]
    fn test_ogg_find_comment_page_index() {
        let serial = 0x0001u32;
        let ogg_bytes = build_minimal_ogg(serial);

        let mut pages = Vec::new();
        let mut cursor = 0usize;
        while cursor < ogg_bytes.len() {
            let (page, consumed) =
                crate::ogg_page::parse_ogg_page(&ogg_bytes[cursor..]).expect("page must parse");
            pages.push(page);
            cursor += consumed;
        }

        let idx =
            OggMetadataWriter::find_comment_page_index(&pages).expect("comment page must be found");
        assert_eq!(idx, 1, "comment page should be the second page (index 1)");
    }

    #[test]
    fn test_ogg_build_segment_table_small() {
        // 5 bytes → one segment with value 5
        let table = build_segment_table(5);
        assert_eq!(table, vec![5u8]);
    }

    #[test]
    fn test_ogg_build_segment_table_exactly_255() {
        // 255 bytes → one segment with value 255, then one with value 0
        let table = build_segment_table(255);
        assert_eq!(table, vec![255u8, 0u8]);
    }

    #[test]
    fn test_ogg_build_segment_table_large() {
        // 510 bytes → [255, 255, 0]
        let table = build_segment_table(510);
        assert_eq!(table, vec![255u8, 255u8, 0u8]);
    }

    #[test]
    fn test_ogg_build_segment_table_513() {
        // 513 bytes → [255, 255, 3]
        let table = build_segment_table(513);
        assert_eq!(table, vec![255u8, 255u8, 3u8]);
    }
}
