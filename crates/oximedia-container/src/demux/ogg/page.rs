//! Ogg page parsing.
//!
//! An Ogg page consists of a header followed by segment data.
//! This module implements parsing of the Ogg page structure as specified
//! in [RFC 3533](https://www.rfc-editor.org/rfc/rfc3533).

use bitflags::bitflags;
use nom::{
    bytes::complete::{tag, take},
    number::complete::{le_u32, le_u64, le_u8},
    IResult, Parser,
};
use oximedia_core::{OxiError, OxiResult};

/// Ogg page header magic bytes.
pub const OGG_MAGIC: &[u8; 4] = b"OggS";

/// Maximum page size (header + 255 segments * 255 bytes each).
#[allow(dead_code)]
pub const MAX_PAGE_SIZE: usize = 27 + 255 + 255 * 255;

/// Minimum page header size (magic + version + flags + granule + serial + seq + crc + segment count).
pub const MIN_HEADER_SIZE: usize = 27;

bitflags! {
    /// Ogg page flags as defined in RFC 3533.
    ///
    /// These flags indicate the page's role in the logical bitstream.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct PageFlags: u8 {
        /// Continuation of previous packet.
        ///
        /// Set when this page contains data that is a continuation of the
        /// packet begun on the previous page.
        const CONTINUATION = 0x01;

        /// Beginning of stream.
        ///
        /// Set on the first page of a logical bitstream. The first packet
        /// on this page should contain codec identification.
        const BOS = 0x02;

        /// End of stream.
        ///
        /// Set on the last page of a logical bitstream.
        const EOS = 0x04;
    }
}

impl Default for PageFlags {
    fn default() -> Self {
        Self::empty()
    }
}

/// Parsed Ogg page structure.
///
/// An Ogg page is the basic unit of the Ogg bitstream format. Each page
/// contains a header with metadata and a body containing segment data.
///
/// # Structure
///
/// ```text
/// +----------+---------+-------------------+----------------+
/// | Header   | Segment | Segment Data ...  | (next page)    |
/// | (27+ B)  | Table   |                   |                |
/// +----------+---------+-------------------+----------------+
/// ```
#[derive(Clone, Debug)]
pub struct OggPage {
    /// Stream structure version (always 0).
    pub version: u8,

    /// Page flags indicating page properties.
    pub flags: PageFlags,

    /// Absolute granule position.
    ///
    /// The interpretation of this field depends on the codec:
    /// - For audio: typically the sample count
    /// - For video: typically the frame count
    ///
    /// A value of -1 (all bits set) indicates no granule position.
    pub granule_position: u64,

    /// Stream serial number.
    ///
    /// Uniquely identifies the logical bitstream within a physical stream.
    pub serial_number: u32,

    /// Page sequence number.
    ///
    /// Monotonically increasing sequence number within a logical bitstream.
    pub page_sequence: u32,

    /// CRC32 checksum.
    ///
    /// Checksum of the entire page (with the checksum field set to zero).
    pub checksum: u32,

    /// Segment table.
    ///
    /// Each byte indicates the length of a segment. A segment length of 255
    /// indicates the packet continues in the next segment. A segment length
    /// less than 255 marks the end of a packet.
    pub segments: Vec<u8>,

    /// Page data (concatenated segments).
    pub data: Vec<u8>,
}

impl OggPage {
    /// Parses an Ogg page from bytes.
    ///
    /// Returns the parsed page and the number of bytes consumed.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is not a valid Ogg page.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let data = read_from_file();
    /// match OggPage::parse(&data) {
    ///     Ok((page, consumed)) => {
    ///         println!("Parsed page with {} segments", page.segments.len());
    ///     }
    ///     Err(e) => eprintln!("Parse error: {}", e),
    /// }
    /// ```
    pub fn parse(input: &[u8]) -> OxiResult<(Self, usize)> {
        match parse_page(input) {
            Ok((remaining, page)) => {
                let consumed = input.len() - remaining.len();
                Ok((page, consumed))
            }
            Err(_) => Err(OxiError::Parse {
                offset: 0,
                message: "Invalid Ogg page".into(),
            }),
        }
    }

    /// Checks if this is a beginning of stream page.
    ///
    /// BOS pages contain the codec identification header.
    #[must_use]
    pub fn is_bos(&self) -> bool {
        self.flags.contains(PageFlags::BOS)
    }

    /// Checks if this is an end of stream page.
    ///
    /// EOS pages mark the end of a logical bitstream.
    #[must_use]
    pub fn is_eos(&self) -> bool {
        self.flags.contains(PageFlags::EOS)
    }

    /// Checks if this is a continuation page.
    ///
    /// Continuation pages contain data that continues a packet from the
    /// previous page.
    #[must_use]
    pub fn is_continuation(&self) -> bool {
        self.flags.contains(PageFlags::CONTINUATION)
    }

    /// Checks if the granule position is valid.
    ///
    /// A granule position of all 1s (`u64::MAX`) indicates no valid position.
    #[must_use]
    pub fn has_granule(&self) -> bool {
        self.granule_position != u64::MAX
    }

    /// Extracts packets from page data using the segment table.
    ///
    /// Returns a vector of tuples containing:
    /// - The packet data
    /// - A boolean indicating if the packet is complete
    ///
    /// A packet is complete if it ends with a segment of size less than 255.
    /// Incomplete packets continue on the next page.
    #[must_use]
    pub fn packets(&self) -> Vec<(Vec<u8>, bool)> {
        let mut packets = Vec::new();
        let mut current_packet = Vec::new();
        let mut offset = 0;

        for &segment_size in &self.segments {
            let size = segment_size as usize;
            if offset + size <= self.data.len() {
                current_packet.extend_from_slice(&self.data[offset..offset + size]);
                offset += size;
            }

            // Segment size < 255 means end of packet
            if segment_size < 255 {
                let complete = true;
                packets.push((std::mem::take(&mut current_packet), complete));
            }
        }

        // If last segment was 255, packet continues on next page
        if !current_packet.is_empty() {
            packets.push((current_packet, false));
        }

        packets
    }

    /// Returns the total size of the page in bytes.
    #[must_use]
    pub fn total_size(&self) -> usize {
        MIN_HEADER_SIZE + self.segments.len() + self.data.len()
    }
}

/// Parses an Ogg page using nom combinators.
fn parse_page(input: &[u8]) -> IResult<&[u8], OggPage> {
    let (input, _) = tag(&OGG_MAGIC[..])(input)?;
    let (
        input,
        (version, flags, granule_position, serial_number, page_sequence, checksum, segment_count),
    ) = (le_u8, le_u8, le_u64, le_u32, le_u32, le_u32, le_u8).parse(input)?;

    let (input, segment_table) = take(segment_count as usize)(input)?;
    let segments: Vec<u8> = segment_table.to_vec();

    let data_size: usize = segments.iter().map(|&s| usize::from(s)).sum();
    let (input, data) = take(data_size)(input)?;

    Ok((
        input,
        OggPage {
            version,
            flags: PageFlags::from_bits_truncate(flags),
            granule_position,
            serial_number,
            page_sequence,
            checksum,
            segments,
            data: data.to_vec(),
        },
    ))
}

/// Calculates CRC32 for Ogg page verification.
///
/// Ogg uses a specific CRC32 polynomial (0x04C11DB7) with a different
/// table generation than the common CRC32 used in Ethernet/ZIP.
///
/// # Arguments
///
/// * `data` - The data to calculate CRC32 for
///
/// # Returns
///
/// The calculated CRC32 checksum.
#[must_use]
#[allow(dead_code)]
pub fn crc32_ogg(data: &[u8]) -> u32 {
    const CRC_TABLE: [u32; 256] = generate_crc_table();
    let mut crc = 0u32;
    for &byte in data {
        crc = (crc << 8) ^ CRC_TABLE[((crc >> 24) as u8 ^ byte) as usize];
    }
    crc
}

/// Generates the CRC32 lookup table for Ogg.
#[allow(dead_code, clippy::cast_possible_truncation)]
const fn generate_crc_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = (i as u32) << 24;
        let mut j = 0;
        while j < 8 {
            if crc & 0x8000_0000 != 0 {
                crc = (crc << 1) ^ 0x04C1_1DB7;
            } else {
                crc <<= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

/// Verifies the CRC32 checksum of a page.
///
/// The page data should include the header with the checksum field
/// set to its original value. This function will zero the checksum
/// bytes before calculating.
///
/// # Arguments
///
/// * `page_data` - The complete page data including header
///
/// # Returns
///
/// `true` if the checksum is valid, `false` otherwise.
#[must_use]
#[allow(dead_code)]
pub fn verify_page_crc(page_data: &[u8]) -> bool {
    if page_data.len() < MIN_HEADER_SIZE {
        return false;
    }

    // Extract the stored checksum
    let stored_crc =
        u32::from_le_bytes([page_data[22], page_data[23], page_data[24], page_data[25]]);

    // Calculate CRC with checksum field zeroed
    let mut data_copy = page_data.to_vec();
    data_copy[22] = 0;
    data_copy[23] = 0;
    data_copy[24] = 0;
    data_copy[25] = 0;

    let calculated_crc = crc32_ogg(&data_copy);
    stored_crc == calculated_crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_flags() {
        assert!(PageFlags::BOS.contains(PageFlags::BOS));
        assert!(!PageFlags::BOS.contains(PageFlags::EOS));

        let combined = PageFlags::BOS | PageFlags::EOS;
        assert!(combined.contains(PageFlags::BOS));
        assert!(combined.contains(PageFlags::EOS));
    }

    #[test]
    fn test_page_flags_default() {
        let flags = PageFlags::default();
        assert!(flags.is_empty());
    }

    #[test]
    fn test_crc32() {
        // Test with known value
        let data = b"OggS";
        let crc = crc32_ogg(data);
        assert_ne!(crc, 0);
    }

    #[test]
    fn test_crc32_empty() {
        let crc = crc32_ogg(&[]);
        assert_eq!(crc, 0);
    }

    #[test]
    fn test_page_parse_minimal() {
        // Construct a minimal valid Ogg page
        let mut page_data = Vec::new();
        page_data.extend_from_slice(OGG_MAGIC); // Magic
        page_data.push(0); // Version
        page_data.push(0x02); // Flags (BOS)
        page_data.extend_from_slice(&0u64.to_le_bytes()); // Granule position
        page_data.extend_from_slice(&1u32.to_le_bytes()); // Serial number
        page_data.extend_from_slice(&0u32.to_le_bytes()); // Page sequence
        page_data.extend_from_slice(&0u32.to_le_bytes()); // CRC (will be wrong)
        page_data.push(1); // Segment count
        page_data.push(5); // Segment table (5 bytes)
        page_data.extend_from_slice(b"hello"); // Data

        let result = OggPage::parse(&page_data);
        assert!(result.is_ok());

        let (page, consumed) = result.expect("operation should succeed");
        assert_eq!(consumed, page_data.len());
        assert!(page.is_bos());
        assert!(!page.is_eos());
        assert!(!page.is_continuation());
        assert_eq!(page.serial_number, 1);
        assert_eq!(page.data, b"hello");
    }

    #[test]
    fn test_page_parse_invalid() {
        let data = b"NotOgg";
        let result = OggPage::parse(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_packets_extraction() {
        // Create a page with multiple packets
        let mut page = OggPage {
            version: 0,
            flags: PageFlags::empty(),
            granule_position: 100,
            serial_number: 1,
            page_sequence: 0,
            checksum: 0,
            segments: vec![5, 10, 255, 20], // 3 packets, all complete
            data: vec![0; 5 + 10 + 255 + 20],
        };

        // Fill data with distinguishable values
        for i in 0..5 {
            page.data[i] = 1;
        }
        for i in 5..15 {
            page.data[i] = 2;
        }
        for i in 15..270 {
            page.data[i] = 3;
        }
        for i in 270..290 {
            page.data[i] = 4;
        }

        let packets = page.packets();
        assert_eq!(packets.len(), 3);

        // First packet: complete (5 bytes, segment < 255 ends packet)
        assert_eq!(packets[0].0.len(), 5);
        assert!(packets[0].1); // Complete

        // Second packet: complete (10 bytes, segment < 255 ends packet)
        assert_eq!(packets[1].0.len(), 10);
        assert!(packets[1].1); // Complete

        // Third packet: complete (255 + 20 = 275 bytes, segment 20 < 255 ends packet)
        assert_eq!(packets[2].0.len(), 255 + 20);
        assert!(packets[2].1); // Complete - packet ends with segment < 255
    }

    #[test]
    fn test_packets_incomplete() {
        // Create a page with an incomplete packet (ends with segment of 255)
        let page = OggPage {
            version: 0,
            flags: PageFlags::empty(),
            granule_position: u64::MAX, // No valid granule
            serial_number: 1,
            page_sequence: 0,
            checksum: 0,
            segments: vec![5, 255], // First complete, second incomplete
            data: vec![0; 5 + 255],
        };

        let packets = page.packets();
        assert_eq!(packets.len(), 2);

        // First packet: complete
        assert_eq!(packets[0].0.len(), 5);
        assert!(packets[0].1);

        // Second packet: incomplete (ends with segment 255, continues on next page)
        assert_eq!(packets[1].0.len(), 255);
        assert!(!packets[1].1);
    }

    #[test]
    fn test_has_granule() {
        let page = OggPage {
            version: 0,
            flags: PageFlags::empty(),
            granule_position: 100,
            serial_number: 1,
            page_sequence: 0,
            checksum: 0,
            segments: vec![],
            data: vec![],
        };
        assert!(page.has_granule());

        let page_no_granule = OggPage {
            version: 0,
            flags: PageFlags::empty(),
            granule_position: u64::MAX,
            serial_number: 1,
            page_sequence: 0,
            checksum: 0,
            segments: vec![],
            data: vec![],
        };
        assert!(!page_no_granule.has_granule());
    }

    #[test]
    fn test_total_size() {
        let page = OggPage {
            version: 0,
            flags: PageFlags::empty(),
            granule_position: 0,
            serial_number: 1,
            page_sequence: 0,
            checksum: 0,
            segments: vec![10, 20],
            data: vec![0; 30],
        };
        assert_eq!(page.total_size(), MIN_HEADER_SIZE + 2 + 30);
    }
}
