// Copyright 2024 The OxiMedia Project Developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Theora frame header parsing and generation.
//!
//! Implements the bitstream format for Theora frame headers following
//! the Theora specification and RFC 7845.

use crate::error::{CodecError, CodecResult};
use crate::theora::bitstream::{BitstreamReader, BitstreamWriter};
use crate::theora::tables::{ColorSpace, PixelAspectRatio, TheoraPixelFormat};

/// Theora identification header.
///
/// Contains stream-wide codec configuration.
#[derive(Debug, Clone)]
pub struct IdentificationHeader {
    /// Version major number (must be 3).
    pub version_major: u8,
    /// Version minor number.
    pub version_minor: u8,
    /// Version revision number.
    pub version_revision: u8,
    /// Encoded frame width (must be multiple of 16).
    pub frame_width: u32,
    /// Encoded frame height (must be multiple of 16).
    pub frame_height: u32,
    /// Displayed picture width.
    pub picture_width: u32,
    /// Displayed picture height.
    pub picture_height: u32,
    /// Picture offset X (from left).
    pub picture_offset_x: u32,
    /// Picture offset Y (from top).
    pub picture_offset_y: u32,
    /// Frame rate numerator.
    pub fps_numerator: u32,
    /// Frame rate denominator.
    pub fps_denominator: u32,
    /// Pixel aspect ratio.
    pub aspect_ratio: PixelAspectRatio,
    /// Color space.
    pub color_space: ColorSpace,
    /// Target bitrate (0 = unspecified).
    pub target_bitrate: u32,
    /// Quality hint (0-63).
    pub quality: u8,
    /// Keyframe granule shift.
    pub keyframe_granule_shift: u8,
    /// Pixel format.
    pub pixel_format: TheoraPixelFormat,
}

impl IdentificationHeader {
    /// Parse identification header from bitstream.
    pub fn parse(reader: &mut BitstreamReader) -> CodecResult<Self> {
        // Check packet type (0x80 for identification)
        let packet_type = reader.read_byte()?;
        if packet_type != 0x80 {
            return Err(CodecError::InvalidBitstream(format!(
                "Invalid identification header type: {packet_type:#x}"
            )));
        }

        // Check codec signature "theora"
        let mut signature = [0u8; 6];
        reader.read_bytes(&mut signature)?;
        if &signature != b"theora" {
            return Err(CodecError::InvalidBitstream(
                "Invalid Theora signature".to_string(),
            ));
        }

        // Version info
        let version_major = reader.read_byte()?;
        let version_minor = reader.read_byte()?;
        let version_revision = reader.read_byte()?;

        if version_major != 3 {
            return Err(CodecError::UnsupportedFeature(format!(
                "Unsupported Theora version: {version_major}"
            )));
        }

        // Frame dimensions
        let frame_width = (reader.read_u16()? as u32) << 4;
        let frame_height = (reader.read_u16()? as u32) << 4;

        // Picture region
        let picture_width = reader.read_bits(24)?;
        let picture_height = reader.read_bits(24)?;
        let picture_offset_x = reader.read_byte()? as u32;
        let picture_offset_y = reader.read_byte()? as u32;

        // Frame rate
        let fps_numerator = reader.read_u32()?;
        let fps_denominator = reader.read_u32()?;

        // Pixel aspect ratio
        let aspect_numerator = reader.read_bits(24)?;
        let aspect_denominator = reader.read_bits(24)?;
        let aspect_ratio = PixelAspectRatio::new(aspect_numerator, aspect_denominator);

        // Color space
        let color_space_val = reader.read_byte()?;
        let color_space = match color_space_val {
            0 => ColorSpace::Undefined,
            1 => ColorSpace::Rec470M,
            2 => ColorSpace::Rec470Bg,
            3 => ColorSpace::Rec709,
            _ => ColorSpace::Undefined,
        };

        // Target bitrate
        let target_bitrate = reader.read_bits(24)?;

        // Quality and keyframe info
        let quality = reader.read_bits(6)? as u8;
        let keyframe_granule_shift = reader.read_bits(5)? as u8;

        // Pixel format
        let pf_val = reader.read_bits(2)? as u8;
        let pixel_format = match pf_val {
            0 => TheoraPixelFormat::Yuv420,
            1 => TheoraPixelFormat::Yuv422,
            2 => TheoraPixelFormat::Yuv444,
            _ => {
                return Err(CodecError::InvalidBitstream(format!(
                    "Invalid pixel format: {pf_val}"
                )))
            }
        };

        // Skip reserved bits
        reader.read_bits(3)?;

        Ok(Self {
            version_major,
            version_minor,
            version_revision,
            frame_width,
            frame_height,
            picture_width,
            picture_height,
            picture_offset_x,
            picture_offset_y,
            fps_numerator,
            fps_denominator,
            aspect_ratio,
            color_space,
            target_bitrate,
            quality,
            keyframe_granule_shift,
            pixel_format,
        })
    }

    /// Write identification header to bitstream.
    pub fn write(&self, writer: &mut BitstreamWriter) -> CodecResult<()> {
        // Packet type
        writer.write_byte(0x80);

        // Codec signature
        for &byte in b"theora" {
            writer.write_byte(byte);
        }

        // Version info
        writer.write_byte(self.version_major);
        writer.write_byte(self.version_minor);
        writer.write_byte(self.version_revision);

        // Frame dimensions
        writer.write_u16((self.frame_width >> 4) as u16);
        writer.write_u16((self.frame_height >> 4) as u16);

        // Picture region
        writer.write_bits(self.picture_width, 24);
        writer.write_bits(self.picture_height, 24);
        writer.write_byte(self.picture_offset_x as u8);
        writer.write_byte(self.picture_offset_y as u8);

        // Frame rate
        writer.write_u32(self.fps_numerator);
        writer.write_u32(self.fps_denominator);

        // Pixel aspect ratio
        writer.write_bits(self.aspect_ratio.num, 24);
        writer.write_bits(self.aspect_ratio.den, 24);

        // Color space
        writer.write_byte(self.color_space as u8);

        // Target bitrate
        writer.write_bits(self.target_bitrate, 24);

        // Quality and keyframe info
        writer.write_bits(u32::from(self.quality), 6);
        writer.write_bits(u32::from(self.keyframe_granule_shift), 5);

        // Pixel format
        let pf_val = match self.pixel_format {
            TheoraPixelFormat::Yuv420 => 0,
            TheoraPixelFormat::Yuv422 => 1,
            TheoraPixelFormat::Yuv444 => 2,
        };
        writer.write_bits(pf_val, 2);

        // Reserved bits
        writer.write_bits(0, 3);

        Ok(())
    }
}

impl Default for IdentificationHeader {
    fn default() -> Self {
        Self {
            version_major: 3,
            version_minor: 2,
            version_revision: 1,
            frame_width: 1920,
            frame_height: 1080,
            picture_width: 1920,
            picture_height: 1080,
            picture_offset_x: 0,
            picture_offset_y: 0,
            fps_numerator: 30,
            fps_denominator: 1,
            aspect_ratio: PixelAspectRatio::default(),
            color_space: ColorSpace::Rec709,
            target_bitrate: 0,
            quality: 48,
            keyframe_granule_shift: 6,
            pixel_format: TheoraPixelFormat::Yuv420,
        }
    }
}

/// Theora comment header.
///
/// Contains metadata and user comments.
#[derive(Debug, Clone)]
pub struct CommentHeader {
    /// Vendor string.
    pub vendor: String,
    /// User comments.
    pub comments: Vec<(String, String)>,
}

impl CommentHeader {
    /// Create a new comment header.
    #[must_use]
    pub fn new(vendor: String) -> Self {
        Self {
            vendor,
            comments: Vec::new(),
        }
    }

    /// Add a comment.
    pub fn add_comment(&mut self, key: String, value: String) {
        self.comments.push((key, value));
    }

    /// Parse comment header from bitstream.
    pub fn parse(reader: &mut BitstreamReader) -> CodecResult<Self> {
        // Check packet type (0x81 for comments)
        let packet_type = reader.read_byte()?;
        if packet_type != 0x81 {
            return Err(CodecError::InvalidBitstream(format!(
                "Invalid comment header type: {packet_type:#x}"
            )));
        }

        // Check codec signature
        let mut signature = [0u8; 6];
        reader.read_bytes(&mut signature)?;
        if &signature != b"theora" {
            return Err(CodecError::InvalidBitstream(
                "Invalid Theora signature in comment header".to_string(),
            ));
        }

        // Read vendor string
        let vendor_length = reader.read_u32()? as usize;
        let mut vendor_bytes = vec![0u8; vendor_length];
        reader.read_bytes(&mut vendor_bytes)?;
        let vendor = String::from_utf8_lossy(&vendor_bytes).into_owned();

        // Read comments
        let num_comments = reader.read_u32()?;
        let mut comments = Vec::new();

        for _ in 0..num_comments {
            let comment_length = reader.read_u32()? as usize;
            let mut comment_bytes = vec![0u8; comment_length];
            reader.read_bytes(&mut comment_bytes)?;
            let comment = String::from_utf8_lossy(&comment_bytes).into_owned();

            // Split into key=value
            if let Some(pos) = comment.find('=') {
                let key = comment[..pos].to_string();
                let value = comment[pos + 1..].to_string();
                comments.push((key, value));
            }
        }

        Ok(Self { vendor, comments })
    }

    /// Write comment header to bitstream.
    pub fn write(&self, writer: &mut BitstreamWriter) -> CodecResult<()> {
        // Packet type
        writer.write_byte(0x81);

        // Codec signature
        for &byte in b"theora" {
            writer.write_byte(byte);
        }

        // Vendor string
        writer.write_u32(self.vendor.len() as u32);
        for &byte in self.vendor.as_bytes() {
            writer.write_byte(byte);
        }

        // Comments
        writer.write_u32(self.comments.len() as u32);
        for (key, value) in &self.comments {
            let comment = format!("{key}={value}");
            writer.write_u32(comment.len() as u32);
            for &byte in comment.as_bytes() {
                writer.write_byte(byte);
            }
        }

        Ok(())
    }
}

impl Default for CommentHeader {
    fn default() -> Self {
        Self::new("OxiMedia Theora Encoder".to_string())
    }
}

/// Theora setup header.
///
/// Contains codec tables and configuration.
#[derive(Debug, Clone)]
pub struct SetupHeader {
    /// Quantization parameters.
    pub quant_params: QuantizationParameters,
    /// Huffman tables.
    pub huffman_tables: HuffmanTables,
}

impl SetupHeader {
    /// Parse setup header from bitstream.
    pub fn parse(reader: &mut BitstreamReader) -> CodecResult<Self> {
        // Check packet type (0x82 for setup)
        let packet_type = reader.read_byte()?;
        if packet_type != 0x82 {
            return Err(CodecError::InvalidBitstream(format!(
                "Invalid setup header type: {packet_type:#x}"
            )));
        }

        // Check codec signature
        let mut signature = [0u8; 6];
        reader.read_bytes(&mut signature)?;
        if &signature != b"theora" {
            return Err(CodecError::InvalidBitstream(
                "Invalid Theora signature in setup header".to_string(),
            ));
        }

        // Parse quantization parameters
        let quant_params = QuantizationParameters::parse(reader)?;

        // Parse Huffman tables
        let huffman_tables = HuffmanTables::parse(reader)?;

        Ok(Self {
            quant_params,
            huffman_tables,
        })
    }

    /// Write setup header to bitstream.
    pub fn write(&self, writer: &mut BitstreamWriter) -> CodecResult<()> {
        // Packet type
        writer.write_byte(0x82);

        // Codec signature
        for &byte in b"theora" {
            writer.write_byte(byte);
        }

        // Write quantization parameters
        self.quant_params.write(writer)?;

        // Write Huffman tables
        self.huffman_tables.write(writer)?;

        Ok(())
    }
}

impl Default for SetupHeader {
    fn default() -> Self {
        Self {
            quant_params: QuantizationParameters::default(),
            huffman_tables: HuffmanTables::default(),
        }
    }
}

/// Quantization parameters for setup header.
#[derive(Debug, Clone)]
pub struct QuantizationParameters {
    /// Quantization ranges.
    pub ranges: Vec<QuantRange>,
    /// Base matrices.
    pub base_matrices: Vec<[u16; 64]>,
}

impl QuantizationParameters {
    /// Parse quantization parameters.
    fn parse(reader: &mut BitstreamReader) -> CodecResult<Self> {
        // Number of quant ranges
        let num_ranges = reader.read_bits(6)? as usize + 1;
        let mut ranges = Vec::new();

        for _ in 0..num_ranges {
            let range = QuantRange::parse(reader)?;
            ranges.push(range);
        }

        // Number of base matrices
        let num_matrices = reader.read_bits(6)? as usize + 1;
        let mut base_matrices = Vec::new();

        for _ in 0..num_matrices {
            let mut matrix = [0u16; 64];
            for val in &mut matrix {
                *val = reader.read_bits(8)? as u16;
            }
            base_matrices.push(matrix);
        }

        Ok(Self {
            ranges,
            base_matrices,
        })
    }

    /// Write quantization parameters.
    fn write(&self, writer: &mut BitstreamWriter) -> CodecResult<()> {
        // Number of quant ranges
        writer.write_bits((self.ranges.len() - 1) as u32, 6);
        for range in &self.ranges {
            range.write(writer)?;
        }

        // Number of base matrices
        writer.write_bits((self.base_matrices.len() - 1) as u32, 6);
        for matrix in &self.base_matrices {
            for &val in matrix.iter() {
                writer.write_bits(u32::from(val), 8);
            }
        }

        Ok(())
    }
}

impl Default for QuantizationParameters {
    fn default() -> Self {
        Self {
            ranges: vec![QuantRange::default()],
            base_matrices: vec![[16u16; 64]],
        }
    }
}

/// Quantization range.
#[derive(Debug, Clone, Copy)]
pub struct QuantRange {
    /// Minimum QI value.
    pub min_qi: u8,
    /// Maximum QI value.
    pub max_qi: u8,
}

impl QuantRange {
    /// Parse quantization range.
    fn parse(reader: &mut BitstreamReader) -> CodecResult<Self> {
        Ok(Self {
            min_qi: reader.read_bits(6)? as u8,
            max_qi: reader.read_bits(6)? as u8,
        })
    }

    /// Write quantization range.
    fn write(&self, writer: &mut BitstreamWriter) -> CodecResult<()> {
        writer.write_bits(u32::from(self.min_qi), 6);
        writer.write_bits(u32::from(self.max_qi), 6);
        Ok(())
    }
}

impl Default for QuantRange {
    fn default() -> Self {
        Self {
            min_qi: 0,
            max_qi: 63,
        }
    }
}

/// Huffman tables for setup header.
#[derive(Debug, Clone)]
pub struct HuffmanTables {
    /// Number of Huffman groups.
    pub num_groups: usize,
    /// Table entries for each group.
    pub tables: Vec<Vec<HuffmanEntry>>,
}

impl HuffmanTables {
    /// Parse Huffman tables.
    fn parse(reader: &mut BitstreamReader) -> CodecResult<Self> {
        let num_groups = reader.read_bits(5)? as usize + 1;
        let mut tables = Vec::new();

        for _ in 0..num_groups {
            let num_entries = reader.read_bits(5)? as usize + 1;
            let mut entries = Vec::new();

            for _ in 0..num_entries {
                let entry = HuffmanEntry::parse(reader)?;
                entries.push(entry);
            }

            tables.push(entries);
        }

        Ok(Self { num_groups, tables })
    }

    /// Write Huffman tables.
    fn write(&self, writer: &mut BitstreamWriter) -> CodecResult<()> {
        writer.write_bits((self.num_groups - 1) as u32, 5);

        for table in &self.tables {
            writer.write_bits((table.len() - 1) as u32, 5);
            for entry in table {
                entry.write(writer)?;
            }
        }

        Ok(())
    }
}

impl Default for HuffmanTables {
    fn default() -> Self {
        Self {
            num_groups: 1,
            tables: vec![vec![HuffmanEntry::default()]],
        }
    }
}

/// Huffman table entry.
#[derive(Debug, Clone, Copy)]
pub struct HuffmanEntry {
    /// Symbol value.
    pub symbol: u8,
    /// Code length in bits.
    pub length: u8,
}

impl HuffmanEntry {
    /// Parse Huffman entry.
    fn parse(reader: &mut BitstreamReader) -> CodecResult<Self> {
        Ok(Self {
            symbol: reader.read_byte()?,
            length: reader.read_bits(5)? as u8,
        })
    }

    /// Write Huffman entry.
    fn write(&self, writer: &mut BitstreamWriter) -> CodecResult<()> {
        writer.write_byte(self.symbol);
        writer.write_bits(u32::from(self.length), 5);
        Ok(())
    }
}

impl Default for HuffmanEntry {
    fn default() -> Self {
        Self {
            symbol: 0,
            length: 1,
        }
    }
}

/// Theora frame header.
#[derive(Debug, Clone)]
pub struct FrameHeader {
    /// Frame type (true = keyframe).
    pub is_keyframe: bool,
    /// Quality index (0-63).
    pub quality_index: u8,
}

impl FrameHeader {
    /// Parse frame header from bitstream.
    pub fn parse(reader: &mut BitstreamReader) -> CodecResult<Self> {
        let frame_type = reader.read_bit()?;
        let is_keyframe = !frame_type;

        let quality_index = reader.read_bits(6)? as u8;

        Ok(Self {
            is_keyframe,
            quality_index,
        })
    }

    /// Write frame header to bitstream.
    pub fn write(&self, writer: &mut BitstreamWriter) {
        writer.write_bit(!self.is_keyframe);
        writer.write_bits(u32::from(self.quality_index), 6);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identification_header_default() {
        let header = IdentificationHeader::default();
        assert_eq!(header.version_major, 3);
        assert_eq!(header.pixel_format, TheoraPixelFormat::Yuv420);
    }

    #[test]
    fn test_comment_header() {
        let mut header = CommentHeader::new("TestVendor".to_string());
        header.add_comment("TITLE".to_string(), "Test Video".to_string());

        assert_eq!(header.vendor, "TestVendor");
        assert_eq!(header.comments.len(), 1);
    }

    #[test]
    fn test_frame_header() {
        let mut writer = BitstreamWriter::new();
        let frame_header = FrameHeader {
            is_keyframe: true,
            quality_index: 30,
        };

        frame_header.write(&mut writer);
        writer.byte_align();

        let data = writer.into_vec();
        let mut reader = BitstreamReader::new(&data);
        let parsed = FrameHeader::parse(&mut reader).expect("should succeed");

        assert_eq!(parsed.is_keyframe, true);
        assert_eq!(parsed.quality_index, 30);
    }

    #[test]
    fn test_quant_range() {
        let range = QuantRange {
            min_qi: 10,
            max_qi: 50,
        };

        let mut writer = BitstreamWriter::new();
        range.write(&mut writer).expect("write should succeed");
        writer.byte_align();

        let data = writer.into_vec();
        let mut reader = BitstreamReader::new(&data);
        let parsed = QuantRange::parse(&mut reader).expect("should succeed");

        assert_eq!(parsed.min_qi, 10);
        assert_eq!(parsed.max_qi, 50);
    }
}
