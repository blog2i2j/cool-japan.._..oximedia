//! Tests for DNG module.

use std::collections::HashMap;

use crate::dng::constants::*;
use crate::dng::conversion::{dng_to_image_frame, image_frame_to_dng};
use crate::dng::demosaic::demosaic_bilinear;
use crate::dng::parser::{ByteOrder, TiffParser};
use crate::dng::processing::{apply_color_matrix, apply_white_balance};
use crate::dng::reader::DngReader;
use crate::dng::types::*;
use crate::dng::writer::DngWriter;
use crate::{ColorSpace, ImageData, ImageFrame, PixelType};

// Helper: build a minimal valid DNG in memory (little-endian)
fn build_minimal_dng(width: u32, height: u32, bps: u16, pattern: CfaPattern) -> Vec<u8> {
    let mut buf = Vec::new();

    // TIFF header
    buf.extend_from_slice(&[0x49, 0x49]); // little-endian
    buf.extend_from_slice(&42u16.to_le_bytes());
    // IFD offset placeholder (will fill after pixel data)
    let ifd_offset_pos = buf.len();
    buf.extend_from_slice(&0u32.to_le_bytes());

    // Pixel data: fill with a known pattern
    let data_offset = buf.len() as u32;
    let pixel_count = width as usize * height as usize;
    let data_size = pixel_count * 2; // 16-bit
    for i in 0..pixel_count {
        buf.extend_from_slice(&(i as u16).to_le_bytes());
    }

    // Align
    if buf.len() % 2 != 0 {
        buf.push(0);
    }

    let ifd_offset = buf.len() as u32;
    buf[ifd_offset_pos..ifd_offset_pos + 4].copy_from_slice(&ifd_offset.to_le_bytes());

    // Build IFD entries (sorted by tag)
    let mut tags: Vec<(u16, u16, u32, u32)> = vec![
        (TAG_IMAGE_WIDTH, 4, 1, width),
        (TAG_IMAGE_LENGTH, 4, 1, height),
        (TAG_BITS_PER_SAMPLE, 3, 1, u32::from(bps)),
        (TAG_COMPRESSION, 3, 1, 1),                    // uncompressed
        (TAG_PHOTOMETRIC_INTERPRETATION, 3, 1, 32803), // CFA
        (TAG_STRIP_OFFSETS, 4, 1, data_offset),
        (TAG_SAMPLES_PER_PIXEL, 3, 1, 1),
        (TAG_ROWS_PER_STRIP, 4, 1, height),
        (TAG_STRIP_BYTE_COUNTS, 4, 1, data_size as u32),
        (TAG_CFA_REPEAT_PATTERN_DIM, 3, 2, 2 | (2 << 16)),
        (
            TAG_CFA_PATTERN,
            1,
            4,
            u32::from_le_bytes(pattern.as_bytes()),
        ),
        // DNG Version [1, 4, 0, 0]
        (TAG_DNG_VERSION, 1, 4, u32::from_le_bytes([1, 4, 0, 0])),
    ];
    tags.sort_by_key(|t| t.0);

    let tag_count = tags.len() as u16;
    buf.extend_from_slice(&tag_count.to_le_bytes());

    for &(tag, dtype, count, value) in &tags {
        buf.extend_from_slice(&tag.to_le_bytes());
        buf.extend_from_slice(&dtype.to_le_bytes());
        buf.extend_from_slice(&count.to_le_bytes());
        buf.extend_from_slice(&value.to_le_bytes());
    }

    // Next IFD = 0
    buf.extend_from_slice(&0u32.to_le_bytes());

    buf
}

fn build_minimal_dng_be(width: u32, height: u32) -> Vec<u8> {
    let mut buf = Vec::new();

    // TIFF header (big-endian)
    buf.extend_from_slice(&[0x4D, 0x4D]); // "MM"
    buf.extend_from_slice(&42u16.to_be_bytes());
    let ifd_offset_pos = buf.len();
    buf.extend_from_slice(&0u32.to_be_bytes());

    // Pixel data
    let data_offset = buf.len() as u32;
    let pixel_count = width as usize * height as usize;
    let data_size = pixel_count * 2;
    for i in 0..pixel_count {
        buf.extend_from_slice(&(i as u16).to_be_bytes());
    }

    if buf.len() % 2 != 0 {
        buf.push(0);
    }

    let ifd_offset = buf.len() as u32;
    buf[ifd_offset_pos..ifd_offset_pos + 4].copy_from_slice(&ifd_offset.to_be_bytes());

    let mut tags: Vec<(u16, u16, u32, u32)> = vec![
        (TAG_IMAGE_WIDTH, 4, 1, width),
        (TAG_IMAGE_LENGTH, 4, 1, height),
        (TAG_BITS_PER_SAMPLE, 3, 1, 16),
        (TAG_COMPRESSION, 3, 1, 1),
        (TAG_PHOTOMETRIC_INTERPRETATION, 3, 1, 32803),
        (TAG_STRIP_OFFSETS, 4, 1, data_offset),
        (TAG_SAMPLES_PER_PIXEL, 3, 1, 1),
        (TAG_ROWS_PER_STRIP, 4, 1, height),
        (TAG_STRIP_BYTE_COUNTS, 4, 1, data_size as u32),
        (TAG_CFA_PATTERN, 1, 4, u32::from_be_bytes([0, 1, 1, 2])), // RGGB in BE
        (TAG_DNG_VERSION, 1, 4, u32::from_be_bytes([1, 4, 0, 0])),
    ];
    tags.sort_by_key(|t| t.0);

    let tag_count = tags.len() as u16;
    buf.extend_from_slice(&tag_count.to_be_bytes());

    for &(tag, dtype, count, value) in &tags {
        buf.extend_from_slice(&tag.to_be_bytes());
        buf.extend_from_slice(&dtype.to_be_bytes());
        buf.extend_from_slice(&count.to_be_bytes());
        buf.extend_from_slice(&value.to_be_bytes());
    }

    buf.extend_from_slice(&0u32.to_be_bytes());

    buf
}

#[test]
#[ignore]
fn test_tiff_header_parsing_le() {
    let data = build_minimal_dng(4, 4, 16, CfaPattern::Rggb);
    let result = TiffParser::parse(&data);
    assert!(result.is_ok());
    let (byte_order, ifds) = result.expect("parse failed");
    assert_eq!(byte_order, ByteOrder::LittleEndian);
    assert!(!ifds.is_empty());
}

#[test]
#[ignore]
fn test_tiff_header_parsing_be() {
    let data = build_minimal_dng_be(4, 4);
    let result = TiffParser::parse(&data);
    assert!(result.is_ok());
    let (byte_order, ifds) = result.expect("parse failed");
    assert_eq!(byte_order, ByteOrder::BigEndian);
    assert!(!ifds.is_empty());
}

#[test]
#[ignore]
fn test_ifd_parsing() {
    let data = build_minimal_dng(8, 6, 16, CfaPattern::Rggb);
    let (byte_order, ifds) = TiffParser::parse(&data).expect("parse failed");
    let parser = TiffParser { byte_order };
    let ifd = &ifds[0];

    let width = parser.get_tag_value_u32(ifd, TAG_IMAGE_WIDTH, &data);
    assert_eq!(width, Some(8));
    let height = parser.get_tag_value_u32(ifd, TAG_IMAGE_LENGTH, &data);
    assert_eq!(height, Some(6));
}

#[test]
#[ignore]
fn test_dng_detection_valid() {
    let data = build_minimal_dng(4, 4, 16, CfaPattern::Rggb);
    assert!(DngReader::is_dng(&data));
}

#[test]
#[ignore]
fn test_dng_detection_invalid_tiff() {
    // Valid TIFF but no DNG version tag
    let mut data = vec![0x49, 0x49]; // LE
    data.extend_from_slice(&42u16.to_le_bytes());
    data.extend_from_slice(&8u32.to_le_bytes()); // IFD at 8

    // Minimal IFD with just width
    let tag_count: u16 = 1;
    data.extend_from_slice(&tag_count.to_le_bytes());
    // ImageWidth tag
    data.extend_from_slice(&256u16.to_le_bytes()); // tag
    data.extend_from_slice(&4u16.to_le_bytes()); // LONG
    data.extend_from_slice(&1u32.to_le_bytes()); // count
    data.extend_from_slice(&10u32.to_le_bytes()); // value
                                                  // Next IFD
    data.extend_from_slice(&0u32.to_le_bytes());

    assert!(!DngReader::is_dng(&data));
}

#[test]
#[ignore]
fn test_dng_detection_garbage() {
    assert!(!DngReader::is_dng(&[0, 1, 2, 3]));
    assert!(!DngReader::is_dng(&[]));
    assert!(!DngReader::is_dng(&[0x49, 0x49, 0xFF, 0xFF]));
}

#[test]
#[ignore]
fn test_cfa_pattern_parsing() {
    assert_eq!(
        DngReader::parse_cfa_pattern(&[0, 1, 1, 2]).expect("parse"),
        CfaPattern::Rggb
    );
    assert_eq!(
        DngReader::parse_cfa_pattern(&[2, 1, 1, 0]).expect("parse"),
        CfaPattern::Bggr
    );
    assert_eq!(
        DngReader::parse_cfa_pattern(&[1, 0, 2, 1]).expect("parse"),
        CfaPattern::Grbg
    );
    assert_eq!(
        DngReader::parse_cfa_pattern(&[1, 2, 0, 1]).expect("parse"),
        CfaPattern::Gbrg
    );

    // Invalid pattern
    assert!(DngReader::parse_cfa_pattern(&[3, 3, 3, 3]).is_err());
    // Too short
    assert!(DngReader::parse_cfa_pattern(&[0, 1]).is_err());
}

#[test]
#[ignore]
fn test_bit_unpacking_8bit() {
    let data = vec![10, 20, 30, 40];
    let result = DngReader::unpack_bits(&data, 8, 4).expect("unpack");
    assert_eq!(result, vec![10, 20, 30, 40]);
}

#[test]
#[ignore]
fn test_bit_unpacking_16bit() {
    let data: Vec<u8> = vec![0x00, 0x04, 0xFF, 0x0F]; // 1024, 4095
    let result = DngReader::unpack_bits(&data, 16, 2).expect("unpack");
    assert_eq!(result, vec![1024, 4095]);
}

#[test]
#[ignore]
fn test_bit_unpacking_10bit() {
    // 10-bit: pack 4 values into 5 bytes
    // Values: 1023 (0x3FF), 512 (0x200), 0 (0x000), 1 (0x001)
    // Bit stream: 11_1111_1111 10_0000_0000 00_0000_0000 00_0000_0001
    // = 0xFF 0xE8 0x00 0x00 0x40 (40 bits = 5 bytes)
    let bits: u64 = (1023u64 << 30) | (512u64 << 20) | (0u64 << 10) | 1u64;
    let bytes = [
        ((bits >> 32) & 0xFF) as u8,
        ((bits >> 24) & 0xFF) as u8,
        ((bits >> 16) & 0xFF) as u8,
        ((bits >> 8) & 0xFF) as u8,
        (bits & 0xFF) as u8,
    ];

    let result = DngReader::unpack_bits(&bytes, 10, 4).expect("unpack");
    assert_eq!(result, vec![1023, 512, 0, 1]);
}

#[test]
#[ignore]
fn test_bit_unpacking_12bit() {
    // 12-bit: pack 2 values into 3 bytes
    // Values: 4095 (0xFFF), 2048 (0x800)
    // Bit stream: 1111_1111_1111 1000_0000_0000
    // = 0xFF 0xF8 0x00 (24 bits = 3 bytes)
    let bits: u32 = (4095u32 << 12) | 2048u32;
    let bytes = [
        ((bits >> 16) & 0xFF) as u8,
        ((bits >> 8) & 0xFF) as u8,
        (bits & 0xFF) as u8,
    ];

    let result = DngReader::unpack_bits(&bytes, 12, 2).expect("unpack");
    assert_eq!(result, vec![4095, 2048]);
}

#[test]
#[ignore]
fn test_bit_unpacking_14bit() {
    // 14-bit: pack 2 values into 28 bits (4 bytes with padding)
    // Values: 16383 (0x3FFF), 8192 (0x2000)
    // Bit stream (MSB first): 11_1111_1111_1111 10_0000_0000_0000 0000
    // 28 bits of real data + 4 bits padding = 32 bits = 4 bytes
    let bits: u64 = (16383u64 << 14) | 8192u64;
    // Shift left by 4 to align to MSB of 32 bits (32 - 28 = 4)
    let shifted = bits << 4;
    let bytes = [
        ((shifted >> 24) & 0xFF) as u8,
        ((shifted >> 16) & 0xFF) as u8,
        ((shifted >> 8) & 0xFF) as u8,
        (shifted & 0xFF) as u8,
    ];

    let result = DngReader::unpack_bits(&bytes, 14, 2).expect("unpack");
    assert_eq!(result, vec![16383, 8192]);
}

#[test]
#[ignore]
fn test_demosaic_bilinear_rggb() {
    // Create a 4x4 synthetic Bayer RGGB pattern
    // Pattern:
    //   R  G  R  G
    //   G  B  G  B
    //   R  G  R  G
    //   G  B  G  B
    //
    // Fill: R=1000, G=500, B=200
    let w = 4u32;
    let h = 4u32;
    let mut raw = vec![0u16; 16];

    for y in 0..4usize {
        for x in 0..4usize {
            let px = y % 2;
            let py = x % 2;
            raw[y * 4 + x] = match (px, py) {
                (0, 0) => 1000, // R
                (0, 1) => 500,  // G
                (1, 0) => 500,  // G
                (1, 1) => 200,  // B
                _ => 0,
            };
        }
    }

    let result = demosaic_bilinear(&raw, w, h, CfaPattern::Rggb).expect("demosaic");
    assert_eq!(result.len(), 16 * 3);

    // Check center pixel (1,1) which is a Blue pixel in RGGB
    // At (1,1): B=200 (known)
    // R should be interpolated from diagonals: (0,0)=1000, (0,2)=1000, (2,0)=1000, (2,2)=1000 -> 1000
    // G should be interpolated from 4-connected: (0,1)=500, (1,0)=500, (1,2)=500, (2,1)=500 -> 500
    let idx = (1 * 4 + 1) * 3;
    assert_eq!(result[idx + 2], 200, "Blue channel at (1,1)");
    assert_eq!(result[idx], 1000, "Red channel at (1,1) from diagonals");
    assert_eq!(result[idx + 1], 500, "Green channel at (1,1)");
}

#[test]
#[ignore]
fn test_demosaic_bilinear_bggr() {
    // BGGR: B G / G R
    let w = 4u32;
    let h = 4u32;
    let mut raw = vec![0u16; 16];

    for y in 0..4usize {
        for x in 0..4usize {
            let px = y % 2;
            let py = x % 2;
            raw[y * 4 + x] = match (px, py) {
                (0, 0) => 200,  // B
                (0, 1) => 500,  // G
                (1, 0) => 500,  // G
                (1, 1) => 1000, // R
                _ => 0,
            };
        }
    }

    let result = demosaic_bilinear(&raw, w, h, CfaPattern::Bggr).expect("demosaic");
    assert_eq!(result.len(), 16 * 3);

    // At (1,1): R=1000 (known)
    let idx = (1 * 4 + 1) * 3;
    assert_eq!(result[idx], 1000, "Red channel at (1,1)");
}

#[test]
#[ignore]
fn test_white_balance_application() {
    // Simple test: neutral = [0.5, 1.0, 0.5] means R and B get boosted
    let mut data = vec![100u16, 200, 100, 200, 400, 200];
    let wb = WhiteBalance {
        as_shot_neutral: [0.5, 1.0, 0.5],
    };

    apply_white_balance(&mut data, &wb, 65535);

    // Gains: R=2.0, G=1.0, B=2.0
    // Normalized by min (1.0): R=2.0, G=1.0, B=2.0
    assert_eq!(data[0], 200); // 100 * 2.0
    assert_eq!(data[1], 200); // 200 * 1.0
    assert_eq!(data[2], 200); // 100 * 2.0
}

#[test]
#[ignore]
fn test_color_matrix_identity() {
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut data = vec![0.5, 0.3, 0.1, 1.0, 0.0, 0.5];

    apply_color_matrix(&mut data, &identity);

    assert!((data[0] - 0.5).abs() < 1e-10);
    assert!((data[1] - 0.3).abs() < 1e-10);
    assert!((data[2] - 0.1).abs() < 1e-10);
    assert!((data[3] - 1.0).abs() < 1e-10);
    assert!((data[4] - 0.0).abs() < 1e-10);
    assert!((data[5] - 0.5).abs() < 1e-10);
}

#[test]
#[ignore]
fn test_color_matrix_transform() {
    // Simple swap: R->B, G->R, B->G
    let matrix = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]];
    let mut data = vec![1.0, 2.0, 3.0];

    apply_color_matrix(&mut data, &matrix);

    assert!((data[0] - 2.0).abs() < 1e-10); // was G
    assert!((data[1] - 3.0).abs() < 1e-10); // was B
    assert!((data[2] - 1.0).abs() < 1e-10); // was R
}

#[test]
#[ignore]
fn test_round_trip_write_read() {
    let width = 8u32;
    let height = 6u32;
    let pixel_count = width as usize * height as usize;

    // Create a test DNG image with known data
    let mut raw_data = Vec::with_capacity(pixel_count);
    for i in 0..pixel_count {
        raw_data.push((i * 100 % 65536) as u16);
    }

    let metadata = DngMetadata {
        dng_version: [1, 4, 0, 0],
        camera_model: "TestCam".to_string(),
        cfa_pattern: CfaPattern::Rggb,
        white_balance: WhiteBalance::default(),
        color_calibration: ColorCalibration::default(),
        black_level: vec![0.0],
        white_level: vec![65535],
        active_area: None,
        exif: HashMap::new(),
    };

    let image = DngImage {
        width,
        height,
        bit_depth: 16,
        channels: 1,
        raw_data: raw_data.clone(),
        metadata,
        is_demosaiced: false,
    };

    // Write
    let written = DngWriter::write(&image).expect("write failed");

    // Verify it is valid DNG
    assert!(DngReader::is_dng(&written), "Written data is not valid DNG");

    // Read back
    let read_back = DngReader::read(&written).expect("read failed");

    assert_eq!(read_back.width, width);
    assert_eq!(read_back.height, height);
    assert_eq!(read_back.channels, 1);
    assert_eq!(read_back.raw_data.len(), raw_data.len());

    // Verify pixel data matches (writer always stores as 16-bit)
    for i in 0..pixel_count {
        assert_eq!(
            read_back.raw_data[i], raw_data[i],
            "Pixel mismatch at index {i}"
        );
    }

    // Verify metadata
    assert_eq!(read_back.metadata.dng_version, [1, 4, 0, 0]);
    assert_eq!(read_back.metadata.cfa_pattern, CfaPattern::Rggb);
}

#[test]
#[ignore]
fn test_dng_to_image_frame_raw() {
    let image = DngImage {
        width: 4,
        height: 4,
        bit_depth: 16,
        channels: 1,
        raw_data: vec![100u16; 16],
        metadata: DngMetadata::default(),
        is_demosaiced: false,
    };

    let frame = dng_to_image_frame(&image, false).expect("conversion failed");
    assert_eq!(frame.width, 4);
    assert_eq!(frame.height, 4);
    assert_eq!(frame.components, 1);
    assert_eq!(frame.pixel_type, PixelType::U16);
    assert_eq!(frame.color_space, ColorSpace::Luma);
}

#[test]
#[ignore]
fn test_dng_to_image_frame_demosaiced() {
    let image = DngImage {
        width: 4,
        height: 4,
        bit_depth: 16,
        channels: 1,
        raw_data: vec![500u16; 16],
        metadata: DngMetadata::default(),
        is_demosaiced: false,
    };

    let frame = dng_to_image_frame(&image, true).expect("conversion failed");
    assert_eq!(frame.width, 4);
    assert_eq!(frame.height, 4);
    assert_eq!(frame.components, 3);
    assert_eq!(frame.pixel_type, PixelType::U16);
    assert_eq!(frame.color_space, ColorSpace::LinearRgb);
}

#[test]
#[ignore]
fn test_image_frame_to_dng() {
    let byte_data: Vec<u8> = (0..48u16).flat_map(|v| v.to_le_bytes()).collect();
    let frame = ImageFrame::new(
        0,
        4,
        4,
        PixelType::U16,
        3,
        ColorSpace::LinearRgb,
        ImageData::interleaved(byte_data),
    );

    let dng = image_frame_to_dng(&frame, None).expect("conversion failed");
    assert_eq!(dng.width, 4);
    assert_eq!(dng.height, 4);
    assert_eq!(dng.channels, 3);
    assert!(dng.is_demosaiced);
    assert_eq!(dng.raw_data.len(), 48);
}

#[test]
#[ignore]
fn test_dng_read_from_constructed_data() {
    let data = build_minimal_dng(8, 8, 16, CfaPattern::Bggr);
    let image = DngReader::read(&data).expect("read failed");

    assert_eq!(image.width, 8);
    assert_eq!(image.height, 8);
    assert_eq!(image.bit_depth, 16);
    assert_eq!(image.channels, 1);
    assert_eq!(image.metadata.cfa_pattern, CfaPattern::Bggr);
    assert!(!image.is_demosaiced);
    assert_eq!(image.raw_data.len(), 64);
}

#[test]
#[ignore]
fn test_metadata_only_read() {
    let data = build_minimal_dng(16, 12, 14, CfaPattern::Grbg);
    let metadata = DngReader::read_metadata(&data).expect("read metadata failed");

    assert_eq!(metadata.dng_version, [1, 4, 0, 0]);
    assert_eq!(metadata.cfa_pattern, CfaPattern::Grbg);
}

#[test]
#[ignore]
fn test_write_from_rgb() {
    let width = 4u32;
    let height = 4u32;
    let rgb_data: Vec<u16> = (0..48).collect();
    let metadata = DngMetadata::default();

    let written =
        DngWriter::write_from_rgb(&rgb_data, width, height, 16, &metadata).expect("write");
    assert!(DngReader::is_dng(&written));

    let read_back = DngReader::read(&written).expect("read");
    assert_eq!(read_back.width, 4);
    assert_eq!(read_back.height, 4);
    assert_eq!(read_back.channels, 3);
}

#[test]
#[ignore]
fn test_cfa_pattern_color_indices() {
    assert_eq!(CfaPattern::Rggb.color_indices(), [0, 1, 1, 2]);
    assert_eq!(CfaPattern::Bggr.color_indices(), [2, 1, 1, 0]);
    assert_eq!(CfaPattern::Grbg.color_indices(), [1, 0, 2, 1]);
    assert_eq!(CfaPattern::Gbrg.color_indices(), [1, 2, 0, 1]);
}

#[test]
#[ignore]
fn test_dng_compression_conversion() {
    assert_eq!(
        DngCompression::from_u16(1).expect("parse"),
        DngCompression::Uncompressed
    );
    assert_eq!(
        DngCompression::from_u16(7).expect("parse"),
        DngCompression::LosslessJpeg
    );
    assert_eq!(
        DngCompression::from_u16(8).expect("parse"),
        DngCompression::Deflate
    );
    assert_eq!(
        DngCompression::from_u16(34892).expect("parse"),
        DngCompression::LossyDng
    );
    assert!(DngCompression::from_u16(999).is_err());
}

#[test]
#[ignore]
fn test_demosaic_edge_handling() {
    // Test that demosaicing handles 2x2 (minimum) correctly
    let raw = vec![1000u16, 500, 500, 200];
    let result = demosaic_bilinear(&raw, 2, 2, CfaPattern::Rggb).expect("demosaic");
    assert_eq!(result.len(), 12); // 4 pixels * 3 channels
                                  // No panics or out-of-bounds is the main assertion
}

#[test]
#[ignore]
fn test_white_balance_neutral_identity() {
    // Neutral [1.0, 1.0, 1.0] should not change values
    let mut data = vec![100u16, 200, 300];
    let wb = WhiteBalance {
        as_shot_neutral: [1.0, 1.0, 1.0],
    };

    apply_white_balance(&mut data, &wb, 65535);

    assert_eq!(data, vec![100, 200, 300]);
}

#[test]
#[ignore]
fn test_default_metadata() {
    let meta = DngMetadata::default();
    assert_eq!(meta.dng_version, [1, 4, 0, 0]);
    assert_eq!(meta.cfa_pattern, CfaPattern::Rggb);
    assert_eq!(meta.white_balance.as_shot_neutral, [1.0, 1.0, 1.0]);
    assert_eq!(meta.color_calibration.illuminant_1, 21);
}
