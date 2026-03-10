//! JPEG-XL decoder implementation.
//!
//! Decodes JPEG-XL codestreams (bare and container format) into raw pixel data.
//! Currently supports lossless Modular mode for 8-bit and 16-bit images in
//! grayscale, RGB, and RGBA color spaces.

use super::bitreader::BitReader;
use super::modular::{ModularDecoder, ModularTransform};
use super::types::{JxlColorSpace, JxlHeader, JXL_CODESTREAM_SIGNATURE, JXL_CONTAINER_SIGNATURE};
use crate::error::{CodecError, CodecResult};

/// Decoded JPEG-XL image.
#[derive(Clone, Debug)]
pub struct DecodedImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Number of channels (1=gray, 3=RGB, 4=RGBA).
    pub channels: u8,
    /// Bits per sample (8 or 16).
    pub bit_depth: u8,
    /// Interleaved pixel data.
    /// For 8-bit: one byte per sample.
    /// For 16-bit: two bytes per sample (little-endian).
    pub data: Vec<u8>,
    /// Color space of the decoded image.
    pub color_space: JxlColorSpace,
}

impl DecodedImage {
    /// Total number of samples in the image.
    pub fn sample_count(&self) -> usize {
        self.width as usize * self.height as usize * self.channels as usize
    }

    /// Total size of pixel data in bytes.
    pub fn data_size(&self) -> usize {
        let bytes_per_sample = if self.bit_depth > 8 { 2 } else { 1 };
        self.sample_count() * bytes_per_sample
    }
}

/// JPEG-XL decoder.
///
/// Decodes JPEG-XL files (both bare codestream and ISOBMFF container format)
/// into raw pixel data.
pub struct JxlDecoder;

impl JxlDecoder {
    /// Create a new JPEG-XL decoder.
    pub fn new() -> Self {
        Self
    }

    /// Check if the data starts with a valid JXL signature.
    ///
    /// Returns `true` for both bare codestream (0xFF 0x0A) and
    /// container format signatures.
    pub fn is_jxl(data: &[u8]) -> bool {
        Self::is_codestream(data) || Self::is_container(data)
    }

    /// Check for bare codestream signature.
    pub fn is_codestream(data: &[u8]) -> bool {
        data.len() >= 2
            && data[0] == JXL_CODESTREAM_SIGNATURE[0]
            && data[1] == JXL_CODESTREAM_SIGNATURE[1]
    }

    /// Check for ISOBMFF container signature.
    pub fn is_container(data: &[u8]) -> bool {
        data.len() >= 12 && data[..12] == JXL_CONTAINER_SIGNATURE
    }

    /// Decode a JPEG-XL file from bytes.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - The data does not have a valid JXL signature
    /// - The header is malformed
    /// - The image data is corrupt
    /// - Unsupported features are encountered
    pub fn decode(&self, data: &[u8]) -> CodecResult<DecodedImage> {
        let codestream = self.extract_codestream(data)?;
        let mut reader = BitReader::new(&codestream);

        // Skip signature (2 bytes = 16 bits)
        let _ = reader.read_bits(16)?;

        // Parse size header
        let (width, height) = self.parse_size_header(&mut reader)?;

        // Parse image metadata
        let header = self.parse_image_metadata(&mut reader, width, height)?;
        header.validate()?;

        // Decode using modular mode
        let channels_data = self.decode_modular(&mut reader, &header)?;

        // Convert channel data to interleaved byte output
        let pixel_data = self.channels_to_interleaved(&channels_data, &header)?;

        Ok(DecodedImage {
            width: header.width,
            height: header.height,
            channels: header.num_channels,
            bit_depth: header.bits_per_sample,
            data: pixel_data,
            color_space: header.color_space,
        })
    }

    /// Read only the header from a JPEG-XL file without fully decoding.
    ///
    /// # Errors
    ///
    /// Returns error if the signature or header is invalid.
    pub fn read_header(&self, data: &[u8]) -> CodecResult<JxlHeader> {
        let codestream = self.extract_codestream(data)?;
        let mut reader = BitReader::new(&codestream);

        // Skip signature
        let _ = reader.read_bits(16)?;

        let (width, height) = self.parse_size_header(&mut reader)?;
        let header = self.parse_image_metadata(&mut reader, width, height)?;
        header.validate()?;
        Ok(header)
    }

    /// Extract the bare codestream from either format.
    ///
    /// If the data is a bare codestream, returns it as-is.
    /// If it is a container, extracts the jxlc box contents.
    fn extract_codestream<'a>(&self, data: &'a [u8]) -> CodecResult<&'a [u8]> {
        if Self::is_codestream(data) {
            return Ok(data);
        }
        if Self::is_container(data) {
            // Parse ISOBMFF boxes to find jxlc (codestream) box
            return self.find_jxlc_box(data);
        }
        Err(CodecError::InvalidBitstream(
            "Not a valid JPEG-XL file: invalid signature".into(),
        ))
    }

    /// Find the jxlc box in an ISOBMFF container.
    fn find_jxlc_box<'a>(&self, data: &'a [u8]) -> CodecResult<&'a [u8]> {
        let mut offset = 0;
        while offset + 8 <= data.len() {
            let box_size = u32::from_be_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;

            let box_type = &data[offset + 4..offset + 8];

            if box_size < 8 {
                break;
            }

            if box_type == b"jxlc" {
                let content_start = offset + 8;
                let content_end = offset + box_size;
                if content_end <= data.len() {
                    return Ok(&data[content_start..content_end]);
                }
                return Err(CodecError::InvalidBitstream(
                    "jxlc box extends past end of file".into(),
                ));
            }

            offset += box_size;
        }

        Err(CodecError::InvalidBitstream(
            "No jxlc (codestream) box found in container".into(),
        ))
    }

    /// Parse the JPEG-XL SizeHeader.
    ///
    /// The size header uses a compact variable-length encoding:
    /// - 1 bit: small flag
    /// - If small: 5 bits height_div8, 5 bits width_div8 (sizes * 8)
    /// - If not small: read height and width using U32 encoding
    fn parse_size_header(&self, reader: &mut BitReader) -> CodecResult<(u32, u32)> {
        let small = reader.read_bool()?;

        if small {
            let height_div8 = reader.read_bits(5)? + 1;
            let width_div8 = reader.read_bits(5)?;
            // Width uses ratio based on height if not specified
            let width_div8 = if width_div8 == 0 {
                height_div8
            } else {
                width_div8
            };
            Ok((width_div8 * 8, height_div8 * 8))
        } else {
            // Full U32 encoding for height and width
            let height = self.read_size_u32(reader)?;
            let width = self.read_size_u32(reader)?;
            Ok((width, height))
        }
    }

    /// Read a size value using JPEG-XL's SizeHeader U32 encoding.
    ///
    /// Distribution: d0=1, d1=1+read(9), d2=1+read(13), d3=1+read(18)
    fn read_size_u32(&self, reader: &mut BitReader) -> CodecResult<u32> {
        let selector = reader.read_bits(2)?;
        match selector {
            0 => Ok(1),
            1 => {
                let extra = reader.read_bits(9)?;
                Ok(1 + extra)
            }
            2 => {
                let extra = reader.read_bits(13)?;
                Ok(1 + extra)
            }
            3 => {
                let extra = reader.read_bits(18)?;
                Ok(1 + extra)
            }
            _ => Err(CodecError::InvalidBitstream("Invalid size selector".into())),
        }
    }

    /// Parse the ImageMetadata section.
    ///
    /// This is a simplified parser that reads the essential fields:
    /// - all_default flag
    /// - bit_depth
    /// - color space
    /// - alpha flag
    fn parse_image_metadata(
        &self,
        reader: &mut BitReader,
        width: u32,
        height: u32,
    ) -> CodecResult<JxlHeader> {
        // all_default flag: if true, use default 8-bit sRGB
        let all_default = reader.read_bool()?;

        if all_default {
            return Ok(JxlHeader {
                width,
                height,
                bits_per_sample: 8,
                num_channels: 3,
                is_float: false,
                has_alpha: false,
                color_space: JxlColorSpace::Srgb,
                orientation: 1,
            });
        }

        // Extra fields present
        let has_extra_fields = reader.read_bool()?;
        let orientation = if has_extra_fields {
            reader.read_bits(3)? as u8 + 1
        } else {
            1
        };

        // Bit depth
        let float_flag = reader.read_bool()?;
        let bits_per_sample = if float_flag {
            // Float samples: read exponent bits
            let _exp_bits = reader.read_bits(4)?;
            let mantissa_bits = reader.read_bits(4)? + 1;
            (mantissa_bits + 1) as u8 // approximate total bits
        } else {
            let depth_selector = reader.read_bits(2)?;
            match depth_selector {
                0 => 8,
                1 => 10,
                2 => 12,
                3 => {
                    let custom = reader.read_bits(6)?;
                    (custom + 1) as u8
                }
                _ => 8,
            }
        };

        // Color space
        let color_space_selector = reader.read_bits(2)?;
        let color_space = match color_space_selector {
            0 => JxlColorSpace::Srgb,
            1 => JxlColorSpace::LinearSrgb,
            2 => JxlColorSpace::Gray,
            3 => JxlColorSpace::Xyb,
            _ => JxlColorSpace::Srgb,
        };

        let num_color_channels = if color_space == JxlColorSpace::Gray {
            1u8
        } else {
            3u8
        };

        // Alpha
        let has_alpha = reader.read_bool()?;
        let num_channels = if has_alpha {
            num_color_channels + 1
        } else {
            num_color_channels
        };

        Ok(JxlHeader {
            width,
            height,
            bits_per_sample,
            num_channels,
            is_float: float_flag,
            has_alpha,
            color_space,
            orientation,
        })
    }

    /// Decode the image data using the Modular sub-codec.
    fn decode_modular(
        &self,
        reader: &mut BitReader,
        header: &JxlHeader,
    ) -> CodecResult<Vec<Vec<i32>>> {
        reader.align_to_byte();

        // Collect remaining data for the modular decoder
        let remaining_bits = reader.remaining_bits();
        if remaining_bits == 0 {
            return Err(CodecError::InvalidBitstream(
                "No image data after header".into(),
            ));
        }

        // Read all remaining bytes into a buffer for the modular decoder
        let remaining_bytes = (remaining_bits + 7) / 8;
        let mut data = Vec::with_capacity(remaining_bytes);
        for _ in 0..remaining_bytes {
            match reader.read_u8(8) {
                Ok(byte) => data.push(byte),
                Err(_) => break,
            }
        }

        let mut decoder = ModularDecoder::new();

        // Add RCT transform for RGB/RGBA images (3+ color channels)
        if header.color_channels() >= 3 {
            decoder.add_transform(ModularTransform::Rct {
                begin_channel: 0,
                rct_type: 0,
            });
        }

        decoder.decode_image(
            &data,
            header.width,
            header.height,
            header.num_channels as u32,
            header.bits_per_sample,
        )
    }

    /// Convert decoded channel data to interleaved byte output.
    fn channels_to_interleaved(
        &self,
        channels: &[Vec<i32>],
        header: &JxlHeader,
    ) -> CodecResult<Vec<u8>> {
        let pixel_count = header.width as usize * header.height as usize;
        let num_channels = header.num_channels as usize;
        let bytes_per_sample = header.bytes_per_sample();

        if channels.len() != num_channels {
            return Err(CodecError::Internal(format!(
                "Expected {} channels, got {}",
                num_channels,
                channels.len()
            )));
        }

        let total_bytes = pixel_count * num_channels * bytes_per_sample;
        let mut output = Vec::with_capacity(total_bytes);

        for i in 0..pixel_count {
            for ch in 0..num_channels {
                let value = channels[ch][i];

                match bytes_per_sample {
                    1 => {
                        // Clamp to [0, 255]
                        let clamped = value.clamp(0, 255) as u8;
                        output.push(clamped);
                    }
                    2 => {
                        // Clamp to [0, 65535], little-endian
                        let clamped = value.clamp(0, 65535) as u16;
                        output.push(clamped as u8);
                        output.push((clamped >> 8) as u8);
                    }
                    _ => {
                        // 32-bit: store as 4 bytes, little-endian
                        let bytes = (value as u32).to_le_bytes();
                        output.extend_from_slice(&bytes);
                    }
                }
            }
        }

        Ok(output)
    }
}

impl Default for JxlDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_is_codestream_signature() {
        assert!(JxlDecoder::is_codestream(&[0xFF, 0x0A, 0x00]));
        assert!(!JxlDecoder::is_codestream(&[0xFF, 0x0B, 0x00]));
        assert!(!JxlDecoder::is_codestream(&[0xFF]));
        assert!(!JxlDecoder::is_codestream(&[]));
    }

    #[test]
    #[ignore]
    fn test_is_container_signature() {
        let mut container = vec![0u8; 16];
        container[..12].copy_from_slice(&JXL_CONTAINER_SIGNATURE);
        assert!(JxlDecoder::is_container(&container));
        assert!(!JxlDecoder::is_container(&[0xFF, 0x0A]));
    }

    #[test]
    #[ignore]
    fn test_is_jxl() {
        assert!(JxlDecoder::is_jxl(&[0xFF, 0x0A]));
        let mut container = vec![0u8; 16];
        container[..12].copy_from_slice(&JXL_CONTAINER_SIGNATURE);
        assert!(JxlDecoder::is_jxl(&container));
        assert!(!JxlDecoder::is_jxl(&[0x00, 0x00]));
    }

    #[test]
    #[ignore]
    fn test_extract_codestream_bare() {
        let decoder = JxlDecoder::new();
        let data = [0xFF, 0x0A, 0x01, 0x02];
        let result = decoder.extract_codestream(&data).expect("ok");
        assert_eq!(result, &data);
    }

    #[test]
    #[ignore]
    fn test_extract_codestream_invalid() {
        let decoder = JxlDecoder::new();
        assert!(decoder.extract_codestream(&[0x00, 0x00]).is_err());
    }

    #[test]
    #[ignore]
    fn test_parse_size_header_small() {
        // small=1, height_div8=3 (24px), width_div8=0 (use height -> 24px)
        let decoder = JxlDecoder::new();
        let mut writer = super::super::bitreader::BitWriter::new();
        writer.write_bool(true); // small = true
        writer.write_bits(2, 5); // height_div8 - 1 = 2 -> height = 3*8 = 24
        writer.write_bits(0, 5); // width_div8 = 0 -> use height_div8
        let data = writer.finish();

        let mut reader = BitReader::new(&data);
        let (w, h) = decoder.parse_size_header(&mut reader).expect("ok");
        assert_eq!(h, 24);
        assert_eq!(w, 24);
    }

    #[test]
    #[ignore]
    fn test_read_header_invalid_data() {
        let decoder = JxlDecoder::new();
        assert!(decoder.read_header(&[0x00]).is_err());
    }

    #[test]
    #[ignore]
    fn test_decoded_image_metrics() {
        let img = DecodedImage {
            width: 10,
            height: 10,
            channels: 3,
            bit_depth: 8,
            data: vec![0u8; 300],
            color_space: JxlColorSpace::Srgb,
        };
        assert_eq!(img.sample_count(), 300);
        assert_eq!(img.data_size(), 300);
    }

    #[test]
    #[ignore]
    fn test_decoded_image_16bit() {
        let img = DecodedImage {
            width: 10,
            height: 10,
            channels: 3,
            bit_depth: 16,
            data: vec![0u8; 600],
            color_space: JxlColorSpace::Srgb,
        };
        assert_eq!(img.sample_count(), 300);
        assert_eq!(img.data_size(), 600);
    }
}
