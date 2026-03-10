//! JPEG-XL encoder implementation.
//!
//! Encodes raw pixel data into JPEG-XL codestreams. Currently supports
//! lossless Modular mode for 8-bit and 16-bit images in grayscale, RGB,
//! and RGBA color spaces.

use super::bitreader::BitWriter;
use super::modular::{ModularEncoder, ModularTransform};
use super::types::{JxlColorSpace, JxlConfig, JxlHeader, JXL_CODESTREAM_SIGNATURE};
use crate::error::{CodecError, CodecResult};

/// JPEG-XL encoder.
///
/// Encodes images to JPEG-XL format. Currently optimized for lossless encoding
/// using the Modular sub-codec with Reversible Color Transform and adaptive
/// prediction.
pub struct JxlEncoder {
    config: JxlConfig,
}

impl JxlEncoder {
    /// Create a new encoder with the given configuration.
    pub fn new(config: JxlConfig) -> Self {
        Self { config }
    }

    /// Create a lossless encoder with default effort.
    pub fn lossless() -> Self {
        Self {
            config: JxlConfig::new_lossless(),
        }
    }

    /// Create a lossless encoder with specified effort level.
    pub fn lossless_with_effort(effort: u8) -> Self {
        Self {
            config: JxlConfig::new_lossless().with_effort(effort),
        }
    }

    /// Encode an image to JPEG-XL format.
    ///
    /// # Arguments
    ///
    /// * `data` - Interleaved pixel data (e.g., RGBRGBRGB... for 8-bit RGB)
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `channels` - Number of channels (1=gray, 3=RGB, 4=RGBA)
    /// * `bit_depth` - Bits per sample (8 or 16)
    ///
    /// # Errors
    ///
    /// Returns error if parameters are invalid or encoding fails.
    pub fn encode(
        &self,
        data: &[u8],
        width: u32,
        height: u32,
        channels: u8,
        bit_depth: u8,
    ) -> CodecResult<Vec<u8>> {
        // Validate inputs
        let mut header = JxlHeader::srgb(width, height, channels)?;
        header.bits_per_sample = bit_depth;
        let expected_size = width as usize
            * height as usize
            * channels as usize
            * (if bit_depth > 8 { 2 } else { 1 });

        if data.len() < expected_size {
            return Err(CodecError::BufferTooSmall {
                needed: expected_size,
                have: data.len(),
            });
        }

        self.config.validate()?;

        // Convert interleaved input to separate channel buffers
        let channels_data = self.deinterleave(data, width, height, channels, bit_depth)?;

        // Encode using modular mode
        let modular_data = self.encode_modular(&channels_data, width, height, &header)?;

        // Build the final codestream
        let mut writer = BitWriter::with_capacity(modular_data.len() + 32);

        // Write signature
        self.write_signature(&mut writer);

        // Write size header
        self.write_size_header(&mut writer, width, height);

        // Write image metadata
        self.write_image_metadata(&mut writer, &header);

        // Align to byte boundary before modular data
        writer.align_to_byte();

        // Append modular-encoded data
        for &byte in &modular_data {
            writer.write_bits(byte as u32, 8);
        }

        Ok(writer.finish())
    }

    /// Write the JPEG-XL codestream signature (0xFF 0x0A).
    fn write_signature(&self, writer: &mut BitWriter) {
        writer.write_bits(JXL_CODESTREAM_SIGNATURE[0] as u32, 8);
        writer.write_bits(JXL_CODESTREAM_SIGNATURE[1] as u32, 8);
    }

    /// Write the SizeHeader.
    ///
    /// Uses the small encoding when possible (dimensions divisible by 8
    /// and <= 256), otherwise uses the full U32 encoding.
    fn write_size_header(&self, writer: &mut BitWriter, width: u32, height: u32) {
        let can_use_small = width > 0
            && height > 0
            && width % 8 == 0
            && height % 8 == 0
            && width / 8 <= 32
            && height / 8 <= 32;

        if can_use_small {
            writer.write_bool(true); // small = true
            let height_div8 = height / 8;
            let width_div8 = width / 8;
            writer.write_bits(height_div8 - 1, 5);
            if width_div8 == height_div8 {
                writer.write_bits(0, 5); // 0 means same as height
            } else {
                writer.write_bits(width_div8, 5);
            }
        } else {
            writer.write_bool(false); // small = false
            self.write_size_u32(writer, height);
            self.write_size_u32(writer, width);
        }
    }

    /// Write a size value using JPEG-XL's SizeHeader U32 encoding.
    fn write_size_u32(&self, writer: &mut BitWriter, value: u32) {
        if value == 1 {
            writer.write_bits(0, 2); // selector 0
        } else if value <= 512 {
            writer.write_bits(1, 2); // selector 1
            writer.write_bits(value - 1, 9);
        } else if value <= 8192 {
            writer.write_bits(2, 2); // selector 2
            writer.write_bits(value - 1, 13);
        } else {
            writer.write_bits(3, 2); // selector 3
            writer.write_bits(value - 1, 18);
        }
    }

    /// Write image metadata.
    fn write_image_metadata(&self, writer: &mut BitWriter, header: &JxlHeader) {
        // Check if we can use all_default (8-bit sRGB, no alpha, orientation 1)
        let is_default = header.bits_per_sample == 8
            && !header.is_float
            && header.color_space == JxlColorSpace::Srgb
            && !header.has_alpha
            && header.orientation == 1
            && header.num_channels == 3;

        if is_default {
            writer.write_bool(true); // all_default = true
            return;
        }

        writer.write_bool(false); // all_default = false

        // Extra fields (orientation)
        let has_extra = header.orientation != 1;
        writer.write_bool(has_extra);
        if has_extra {
            writer.write_bits((header.orientation - 1) as u32, 3);
        }

        // Bit depth
        writer.write_bool(header.is_float); // float flag
        if !header.is_float {
            match header.bits_per_sample {
                8 => writer.write_bits(0, 2),
                10 => writer.write_bits(1, 2),
                12 => writer.write_bits(2, 2),
                other => {
                    writer.write_bits(3, 2); // custom
                    writer.write_bits((other - 1) as u32, 6);
                }
            }
        }

        // Color space
        let cs_selector = match header.color_space {
            JxlColorSpace::Srgb => 0u32,
            JxlColorSpace::LinearSrgb => 1,
            JxlColorSpace::Gray => 2,
            JxlColorSpace::Xyb => 3,
        };
        writer.write_bits(cs_selector, 2);

        // Alpha
        writer.write_bool(header.has_alpha);
    }

    /// Encode image channels using the Modular sub-codec.
    fn encode_modular(
        &self,
        channels: &[Vec<i32>],
        width: u32,
        height: u32,
        header: &JxlHeader,
    ) -> CodecResult<Vec<u8>> {
        let mut encoder = ModularEncoder::new().with_effort(self.config.effort);

        // Add RCT for RGB/RGBA images (3+ color channels)
        if header.color_channels() >= 3 {
            encoder.add_transform(ModularTransform::Rct {
                begin_channel: 0,
                rct_type: 0,
            });
        }

        encoder.encode_image(channels, width, height, header.bits_per_sample)
    }

    /// Convert interleaved pixel data to separate channel buffers.
    fn deinterleave(
        &self,
        data: &[u8],
        width: u32,
        height: u32,
        channels: u8,
        bit_depth: u8,
    ) -> CodecResult<Vec<Vec<i32>>> {
        let pixel_count = width as usize * height as usize;
        let num_channels = channels as usize;
        let bytes_per_sample = if bit_depth > 8 { 2usize } else { 1usize };

        let mut channel_data: Vec<Vec<i32>> = (0..num_channels)
            .map(|_| Vec::with_capacity(pixel_count))
            .collect();

        for i in 0..pixel_count {
            for ch in 0..num_channels {
                let offset = (i * num_channels + ch) * bytes_per_sample;

                let value = if bytes_per_sample == 1 {
                    data[offset] as i32
                } else {
                    // 16-bit little-endian
                    let lo = data[offset] as i32;
                    let hi = data.get(offset + 1).copied().unwrap_or(0) as i32;
                    lo | (hi << 8)
                };

                channel_data[ch].push(value);
            }
        }

        Ok(channel_data)
    }
}

impl Default for JxlEncoder {
    fn default() -> Self {
        Self::lossless()
    }
}

#[cfg(test)]
mod tests {
    use super::super::decoder::JxlDecoder;
    use super::*;

    #[test]
    #[ignore]
    fn test_lossless_roundtrip_rgb8() {
        let width = 8u32;
        let height = 8u32;
        let channels = 3u8;
        let pixel_count = (width * height) as usize;

        // Generate test pattern
        let mut data = Vec::with_capacity(pixel_count * channels as usize);
        for i in 0..pixel_count {
            data.push(((i * 3) % 256) as u8); // R
            data.push(((i * 5 + 50) % 256) as u8); // G
            data.push(((i * 7 + 100) % 256) as u8); // B
        }

        let encoder = JxlEncoder::lossless();
        let encoded = encoder
            .encode(&data, width, height, channels, 8)
            .expect("encode ok");

        // Verify signature
        assert!(JxlDecoder::is_codestream(&encoded));

        let decoder = JxlDecoder::new();
        let decoded = decoder.decode(&encoded).expect("decode ok");

        assert_eq!(decoded.width, width);
        assert_eq!(decoded.height, height);
        assert_eq!(decoded.channels, channels);
        assert_eq!(decoded.bit_depth, 8);
        assert_eq!(decoded.data, data, "Lossless roundtrip failed for RGB8");
    }

    #[test]
    #[ignore]
    fn test_lossless_roundtrip_grayscale() {
        let width = 16u32;
        let height = 16u32;
        let channels = 1u8;
        let pixel_count = (width * height) as usize;

        let mut data = Vec::with_capacity(pixel_count);
        for i in 0..pixel_count {
            data.push((i % 256) as u8);
        }

        let encoder = JxlEncoder::lossless();
        let encoded = encoder
            .encode(&data, width, height, channels, 8)
            .expect("encode ok");

        let decoder = JxlDecoder::new();
        let decoded = decoder.decode(&encoded).expect("decode ok");

        assert_eq!(decoded.width, width);
        assert_eq!(decoded.height, height);
        assert_eq!(decoded.channels, 1);
        assert_eq!(
            decoded.data, data,
            "Lossless roundtrip failed for grayscale"
        );
    }

    #[test]
    #[ignore]
    fn test_lossless_roundtrip_16bit() {
        let width = 4u32;
        let height = 4u32;
        let channels = 3u8;
        let pixel_count = (width * height) as usize;

        // Generate 16-bit test data (little-endian)
        let mut data = Vec::with_capacity(pixel_count * channels as usize * 2);
        for i in 0..pixel_count {
            for ch in 0..channels as usize {
                let val = ((i * 1000 + ch * 3000) % 65536) as u16;
                data.push(val as u8); // low byte
                data.push((val >> 8) as u8); // high byte
            }
        }

        let encoder = JxlEncoder::lossless();
        let encoded = encoder
            .encode(&data, width, height, channels, 16)
            .expect("encode ok");

        let decoder = JxlDecoder::new();
        let decoded = decoder.decode(&encoded).expect("decode ok");

        assert_eq!(decoded.width, width);
        assert_eq!(decoded.height, height);
        assert_eq!(decoded.channels, channels);
        assert_eq!(decoded.bit_depth, 16);
        assert_eq!(decoded.data, data, "Lossless roundtrip failed for 16-bit");
    }

    #[test]
    #[ignore]
    fn test_lossless_roundtrip_rgba() {
        let width = 4u32;
        let height = 4u32;
        let channels = 4u8;
        let pixel_count = (width * height) as usize;

        let mut data = Vec::with_capacity(pixel_count * channels as usize);
        for i in 0..pixel_count {
            data.push(((i * 13) % 256) as u8); // R
            data.push(((i * 17 + 30) % 256) as u8); // G
            data.push(((i * 23 + 60) % 256) as u8); // B
            data.push(((i * 31 + 90) % 256) as u8); // A
        }

        let encoder = JxlEncoder::lossless();
        let encoded = encoder
            .encode(&data, width, height, channels, 8)
            .expect("encode ok");

        let decoder = JxlDecoder::new();
        let decoded = decoder.decode(&encoded).expect("decode ok");

        assert_eq!(decoded.width, width);
        assert_eq!(decoded.height, height);
        assert_eq!(decoded.channels, channels);
        assert_eq!(decoded.data, data, "Lossless roundtrip failed for RGBA");
    }

    #[test]
    #[ignore]
    fn test_lossless_roundtrip_flat_image() {
        // All-zero image (worst case for some compressors)
        let width = 32u32;
        let height = 32u32;
        let channels = 3u8;
        let data = vec![128u8; (width * height) as usize * channels as usize];

        let encoder = JxlEncoder::lossless();
        let encoded = encoder
            .encode(&data, width, height, channels, 8)
            .expect("encode ok");

        let decoder = JxlDecoder::new();
        let decoded = decoder.decode(&encoded).expect("decode ok");

        assert_eq!(
            decoded.data, data,
            "Lossless roundtrip failed for flat image"
        );
    }

    #[test]
    #[ignore]
    fn test_encode_invalid_buffer() {
        let encoder = JxlEncoder::lossless();
        let result = encoder.encode(&[0u8; 10], 100, 100, 3, 8);
        assert!(result.is_err());
    }

    #[test]
    #[ignore]
    fn test_encode_zero_dimensions() {
        let encoder = JxlEncoder::lossless();
        assert!(encoder.encode(&[], 0, 100, 3, 8).is_err());
        assert!(encoder.encode(&[], 100, 0, 3, 8).is_err());
    }

    #[test]
    #[ignore]
    fn test_signature_written() {
        let encoder = JxlEncoder::lossless();
        let data = vec![0u8; 64 * 64 * 3];
        let encoded = encoder.encode(&data, 64, 64, 3, 8).expect("ok");
        assert_eq!(encoded[0], 0xFF);
        assert_eq!(encoded[1], 0x0A);
    }

    #[test]
    #[ignore]
    fn test_size_header_small() {
        // 64x64 is divisible by 8 and fits in small encoding
        let encoder = JxlEncoder::new(JxlConfig::new_lossless());
        let mut writer = BitWriter::new();
        encoder.write_size_header(&mut writer, 64, 64);
        let data = writer.finish();

        // First bit should be 1 (small=true)
        assert_eq!(data[0] & 1, 1);
    }

    #[test]
    #[ignore]
    fn test_size_header_large() {
        // Non-power-of-8 dimensions require full encoding
        let encoder = JxlEncoder::new(JxlConfig::new_lossless());
        let mut writer = BitWriter::new();
        encoder.write_size_header(&mut writer, 1920, 1080);
        let data = writer.finish();

        // First bit should be 0 (small=false)
        assert_eq!(data[0] & 1, 0);
    }

    #[test]
    #[ignore]
    fn test_effort_levels() {
        let e1 = JxlEncoder::lossless_with_effort(1);
        let e9 = JxlEncoder::lossless_with_effort(9);
        assert_eq!(e1.config.effort, 1);
        assert_eq!(e9.config.effort, 9);
    }

    #[test]
    #[ignore]
    fn test_deinterleave_rgb() {
        let encoder = JxlEncoder::lossless();
        let data = [10u8, 20, 30, 40, 50, 60];
        let channels = encoder.deinterleave(&data, 2, 1, 3, 8).expect("ok");
        assert_eq!(channels.len(), 3);
        assert_eq!(channels[0], vec![10, 40]); // R
        assert_eq!(channels[1], vec![20, 50]); // G
        assert_eq!(channels[2], vec![30, 60]); // B
    }

    #[test]
    #[ignore]
    fn test_deinterleave_16bit() {
        let encoder = JxlEncoder::lossless();
        // Two 16-bit grayscale pixels: 0x0100 (256) and 0x0200 (512)
        let data = [0x00u8, 0x01, 0x00, 0x02];
        let channels = encoder.deinterleave(&data, 2, 1, 1, 16).expect("ok");
        assert_eq!(channels.len(), 1);
        assert_eq!(channels[0], vec![256, 512]);
    }
}
