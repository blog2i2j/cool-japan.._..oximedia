//! JPEG-XL encoder implementation.
//!
//! Encodes raw pixel data into JPEG-XL codestreams. Currently supports
//! lossless Modular mode for 8-bit and 16-bit images in grayscale, RGB,
//! and RGBA color spaces.

use super::bitreader::BitWriter;
use super::modular::{ModularEncoder, ModularTransform};
use super::types::{JxlAnimation, JxlColorSpace, JxlConfig, JxlHeader, JXL_CODESTREAM_SIGNATURE};
use crate::container::isobmff::make_box;
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
    pub(crate) fn write_signature(&self, writer: &mut BitWriter) {
        writer.write_bits(JXL_CODESTREAM_SIGNATURE[0] as u32, 8);
        writer.write_bits(JXL_CODESTREAM_SIGNATURE[1] as u32, 8);
    }

    /// Write the SizeHeader.
    ///
    /// Uses the small encoding when possible (dimensions divisible by 8
    /// and <= 256), otherwise uses the full U32 encoding.
    pub(crate) fn write_size_header(&self, writer: &mut BitWriter, width: u32, height: u32) {
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
    pub(crate) fn write_size_u32(&self, writer: &mut BitWriter, value: u32) {
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
    pub(crate) fn write_image_metadata(&self, writer: &mut BitWriter, header: &JxlHeader) {
        // Check if we can use all_default (8-bit sRGB, no alpha, orientation 1, no animation)
        let is_default = header.bits_per_sample == 8
            && !header.is_float
            && header.color_space == JxlColorSpace::Srgb
            && !header.has_alpha
            && header.orientation == 1
            && header.num_channels == 3
            && header.animation.is_none();

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

        // Animation header
        writer.write_bool(header.animation.is_some());
        if let Some(ref anim) = header.animation {
            Self::write_animation_header(writer, anim);
        }
    }

    /// Write the animation header fields.
    pub(crate) fn write_animation_header(writer: &mut BitWriter, anim: &JxlAnimation) {
        writer.write_bits(anim.tps_numerator, 32);
        writer.write_bits(anim.tps_denominator, 32);
        writer.write_bits(anim.num_loops, 32);
        writer.write_bool(anim.have_timecodes);
    }

    /// Write a per-frame header for animated codestreams.
    pub(crate) fn write_frame_header(writer: &mut BitWriter, duration_ticks: u32, is_last: bool) {
        writer.write_bits(duration_ticks, 32);
        writer.write_bool(is_last);
    }

    /// Encode image channels using the Modular sub-codec.
    pub(crate) fn encode_modular(
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
    pub(crate) fn deinterleave(
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

/// Pending frame data for the animated encoder.
struct PendingFrame {
    /// Modular-encoded data for this frame.
    modular_data: Vec<u8>,
    /// Duration in ticks for this frame.
    duration_ticks: u32,
}

/// Animated JPEG-XL encoder.
///
/// Builds a multi-frame JPEG-XL codestream by accumulating frames via
/// [`add_frame`](Self::add_frame) and then producing the final codestream
/// with [`finish`](Self::finish).
///
/// # Codestream Layout
///
/// ```text
/// [signature (2 bytes)] [size_header] [image_metadata with animation]
/// [frame_header_0] [align] [frame_data_0]
/// [frame_header_1] [align] [frame_data_1]
/// ...
/// [frame_header_N] [align] [frame_data_N]   (is_last = true)
/// ```
///
/// # Examples
///
/// ```ignore
/// use oximedia_codec::jpegxl::{AnimatedJxlEncoder, JxlAnimation};
///
/// let anim = JxlAnimation::millisecond();
/// let mut encoder = AnimatedJxlEncoder::new(anim, 8, 8, 3, 8)?;
/// encoder.add_frame(&frame0_data, 100)?;  // 100ms
/// encoder.add_frame(&frame1_data, 200)?;  // 200ms
/// let codestream = encoder.finish()?;
/// ```
pub struct AnimatedJxlEncoder {
    /// Animation header.
    animation: JxlAnimation,
    /// Image width (all frames must share).
    width: u32,
    /// Image height (all frames must share).
    height: u32,
    /// Number of channels (all frames must share).
    channels: u8,
    /// Bit depth (all frames must share).
    bit_depth: u8,
    /// Encoding effort.
    effort: u8,
    /// Accumulated frames.
    frames: Vec<PendingFrame>,
}

impl AnimatedJxlEncoder {
    /// Create a new animated encoder.
    ///
    /// # Arguments
    ///
    /// * `animation` - Animation timing configuration
    /// * `width` - Frame width in pixels (shared by all frames)
    /// * `height` - Frame height in pixels (shared by all frames)
    /// * `channels` - Number of channels (1=gray, 3=RGB, 4=RGBA)
    /// * `bit_depth` - Bits per sample (8 or 16)
    ///
    /// # Errors
    ///
    /// Returns error if dimensions, channels, or bit depth are invalid.
    pub fn new(
        animation: JxlAnimation,
        width: u32,
        height: u32,
        channels: u8,
        bit_depth: u8,
    ) -> CodecResult<Self> {
        // Validate via JxlHeader::srgb which checks channels and dimensions
        let mut header = JxlHeader::srgb(width, height, channels)?;
        header.bits_per_sample = bit_depth;
        header.validate()?;

        Ok(Self {
            animation,
            width,
            height,
            channels,
            bit_depth,
            effort: 7,
            frames: Vec::new(),
        })
    }

    /// Set encoding effort (1-9).
    pub fn with_effort(mut self, effort: u8) -> Self {
        self.effort = effort.clamp(1, 9);
        self
    }

    /// Add a frame to the animation.
    ///
    /// # Arguments
    ///
    /// * `data` - Interleaved pixel data for this frame (same layout as `JxlEncoder::encode`)
    /// * `duration_ticks` - Duration of this frame in ticks (relative to animation tick rate)
    ///
    /// # Errors
    ///
    /// Returns error if the data size is wrong or encoding fails.
    pub fn add_frame(&mut self, data: &[u8], duration_ticks: u32) -> CodecResult<()> {
        let bytes_per_sample: usize = if self.bit_depth > 8 { 2 } else { 1 };
        let expected_size =
            self.width as usize * self.height as usize * self.channels as usize * bytes_per_sample;

        if data.len() < expected_size {
            return Err(CodecError::BufferTooSmall {
                needed: expected_size,
                have: data.len(),
            });
        }

        // Use a temporary single-frame JxlEncoder for the modular encoding pipeline
        let single_encoder = JxlEncoder::lossless_with_effort(self.effort);
        let channels_data = single_encoder.deinterleave(
            data,
            self.width,
            self.height,
            self.channels,
            self.bit_depth,
        )?;

        let mut header = JxlHeader::srgb(self.width, self.height, self.channels)?;
        header.bits_per_sample = self.bit_depth;

        let modular_data =
            single_encoder.encode_modular(&channels_data, self.width, self.height, &header)?;

        self.frames.push(PendingFrame {
            modular_data,
            duration_ticks,
        });

        Ok(())
    }

    /// Number of frames added so far.
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Finalize and wrap the animated codestream in an ISOBMFF container.
    ///
    /// Produces a file beginning with a `ftyp` box (major brand `jxl `),
    /// followed by a `jxll` box and a single `jxlp` box containing the bare
    /// codestream with the `is_last` flag set.
    ///
    /// # Errors
    ///
    /// Returns error if no frames have been added or encoding fails.
    pub fn finish_isobmff(self) -> CodecResult<Vec<u8>> {
        let codestream = self.finish()?;

        // ftyp: major_brand="jxl ", minor_version=0, compatible_brands=["jxl ", "isom"]
        let mut ftyp_payload = Vec::with_capacity(16);
        ftyp_payload.extend_from_slice(b"jxl ");
        ftyp_payload.extend_from_slice(&0u32.to_be_bytes());
        ftyp_payload.extend_from_slice(b"jxl ");
        ftyp_payload.extend_from_slice(b"isom");
        let ftyp = make_box(*b"ftyp", &ftyp_payload);

        // jxll: level 5
        let jxll = make_box(*b"jxll", &[5u8]);

        // jxlp: index/flags with is_last bit set (bit 31), followed by bare codestream.
        let index_val: u32 = 0u32 | 0x8000_0000;
        let mut jxlp_payload = Vec::with_capacity(4 + codestream.len());
        jxlp_payload.extend_from_slice(&index_val.to_be_bytes());
        jxlp_payload.extend_from_slice(&codestream);
        let jxlp = make_box(*b"jxlp", &jxlp_payload);

        let mut out = Vec::with_capacity(ftyp.len() + jxll.len() + jxlp.len());
        out.extend_from_slice(&ftyp);
        out.extend_from_slice(&jxll);
        out.extend_from_slice(&jxlp);
        Ok(out)
    }

    /// Finalize the animated codestream.
    ///
    /// Produces the complete multi-frame JPEG-XL bare codestream. The last
    /// frame is automatically marked with `is_last = true`.
    ///
    /// # Errors
    ///
    /// Returns error if no frames have been added.
    pub fn finish(self) -> CodecResult<Vec<u8>> {
        if self.frames.is_empty() {
            return Err(CodecError::InvalidParameter(
                "Animated JPEG-XL requires at least one frame".into(),
            ));
        }

        let mut header = JxlHeader::srgb(self.width, self.height, self.channels)?;
        header.bits_per_sample = self.bit_depth;
        header.animation = Some(self.animation);

        // Estimate total capacity
        let total_modular: usize = self.frames.iter().map(|f| f.modular_data.len()).sum();
        let estimated_capacity = 64 + total_modular + self.frames.len() * 8;
        let mut writer = BitWriter::with_capacity(estimated_capacity);

        // Use a temporary encoder for header writing methods
        let helper = JxlEncoder::lossless();

        // Write signature
        helper.write_signature(&mut writer);

        // Write size header
        helper.write_size_header(&mut writer, self.width, self.height);

        // Write image metadata (includes animation header)
        helper.write_image_metadata(&mut writer, &header);

        // Write each frame
        let frame_count = self.frames.len();
        for (i, frame) in self.frames.into_iter().enumerate() {
            let is_last = i == frame_count - 1;

            // Write frame header
            JxlEncoder::write_frame_header(&mut writer, frame.duration_ticks, is_last);

            // Align to byte boundary before modular data
            writer.align_to_byte();

            // Write frame data length (so decoder knows where each frame ends)
            let data_len = frame.modular_data.len() as u32;
            writer.write_bits(data_len, 32);

            // Write modular-encoded frame data
            for &byte in &frame.modular_data {
                writer.write_bits(byte as u32, 8);
            }
        }

        Ok(writer.finish())
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

    // --- Animation encoder tests ---

    /// Generate a simple test frame with a deterministic pattern seeded by `seed`.
    fn make_test_frame(width: u32, height: u32, channels: u8, seed: u8) -> Vec<u8> {
        let pixel_count = (width * height) as usize;
        let mut data = Vec::with_capacity(pixel_count * channels as usize);
        for i in 0..pixel_count {
            for ch in 0..channels as usize {
                let val = ((i.wrapping_mul(3 + ch) + seed as usize * 37 + ch * 50) % 256) as u8;
                data.push(val);
            }
        }
        data
    }

    #[test]
    fn test_animated_encoder_three_frames_rgb() {
        let anim = JxlAnimation::millisecond();
        let width = 4u32;
        let height = 4u32;
        let channels = 3u8;

        let mut encoder =
            AnimatedJxlEncoder::new(anim, width, height, channels, 8).expect("create ok");

        let f0 = make_test_frame(width, height, channels, 0);
        let f1 = make_test_frame(width, height, channels, 1);
        let f2 = make_test_frame(width, height, channels, 2);

        encoder.add_frame(&f0, 100).expect("frame 0");
        encoder.add_frame(&f1, 200).expect("frame 1");
        encoder.add_frame(&f2, 150).expect("frame 2");

        assert_eq!(encoder.frame_count(), 3);

        let codestream = encoder.finish().expect("finish ok");

        // Verify signature
        assert!(JxlDecoder::is_codestream(&codestream));

        // Decode animated
        let decoder = JxlDecoder::new();
        let frames = decoder.decode_animated(&codestream).expect("decode ok");

        assert_eq!(frames.len(), 3);

        // Verify frame 0
        assert_eq!(frames[0].width, width);
        assert_eq!(frames[0].height, height);
        assert_eq!(frames[0].channels, channels);
        assert_eq!(frames[0].duration_ticks, 100);
        assert!(!frames[0].is_last);
        assert_eq!(frames[0].data, f0, "Frame 0 pixel data mismatch");

        // Verify frame 1
        assert_eq!(frames[1].duration_ticks, 200);
        assert!(!frames[1].is_last);
        assert_eq!(frames[1].data, f1, "Frame 1 pixel data mismatch");

        // Verify frame 2 (last)
        assert_eq!(frames[2].duration_ticks, 150);
        assert!(frames[2].is_last);
        assert_eq!(frames[2].data, f2, "Frame 2 pixel data mismatch");
    }

    #[test]
    fn test_animated_encoder_single_frame_with_animation_header() {
        let anim = JxlAnimation::millisecond().with_num_loops(1);
        let width = 2u32;
        let height = 2u32;
        let channels = 3u8;

        let mut encoder =
            AnimatedJxlEncoder::new(anim, width, height, channels, 8).expect("create ok");

        let frame_data = make_test_frame(width, height, channels, 42);
        encoder.add_frame(&frame_data, 500).expect("frame ok");

        let codestream = encoder.finish().expect("finish ok");

        let decoder = JxlDecoder::new();
        let frames = decoder.decode_animated(&codestream).expect("decode ok");

        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].duration_ticks, 500);
        assert!(frames[0].is_last);
        assert_eq!(frames[0].data, frame_data);
    }

    #[test]
    fn test_animated_encoder_zero_duration() {
        let anim = JxlAnimation::millisecond();
        let width = 2u32;
        let height = 2u32;
        let channels = 1u8;

        let mut encoder =
            AnimatedJxlEncoder::new(anim, width, height, channels, 8).expect("create ok");

        let f0 = vec![128u8; 4];
        encoder.add_frame(&f0, 0).expect("frame ok");

        let codestream = encoder.finish().expect("finish ok");

        let decoder = JxlDecoder::new();
        let frames = decoder.decode_animated(&codestream).expect("decode ok");

        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].duration_ticks, 0);
        assert!(frames[0].is_last);
    }

    #[test]
    fn test_animated_encoder_infinite_loop() {
        let anim = JxlAnimation::millisecond().with_num_loops(0);
        let width = 2u32;
        let height = 2u32;
        let channels = 3u8;

        let mut encoder =
            AnimatedJxlEncoder::new(anim, width, height, channels, 8).expect("create ok");
        encoder
            .add_frame(&make_test_frame(width, height, channels, 0), 100)
            .expect("ok");

        let codestream = encoder.finish().expect("finish ok");

        let decoder = JxlDecoder::new();
        let anim_header = decoder
            .read_animation_header(&codestream)
            .expect("header ok");
        let anim_header = anim_header.expect("should have animation");
        assert_eq!(anim_header.num_loops, 0);
    }

    #[test]
    fn test_animated_encoder_no_frames_error() {
        let anim = JxlAnimation::millisecond();
        let encoder = AnimatedJxlEncoder::new(anim, 4, 4, 3, 8).expect("create ok");
        assert!(encoder.finish().is_err());
    }

    #[test]
    fn test_animated_encoder_invalid_buffer_size() {
        let anim = JxlAnimation::millisecond();
        let mut encoder = AnimatedJxlEncoder::new(anim, 4, 4, 3, 8).expect("create ok");
        // Buffer too small (4*4*3 = 48 bytes needed, only 10 provided)
        assert!(encoder.add_frame(&[0u8; 10], 100).is_err());
    }

    #[test]
    fn test_animated_encoder_grayscale() {
        let anim = JxlAnimation::new(24, 1).expect("valid");
        let width = 4u32;
        let height = 4u32;
        let channels = 1u8;

        let mut encoder =
            AnimatedJxlEncoder::new(anim, width, height, channels, 8).expect("create ok");

        let f0 = make_test_frame(width, height, channels, 10);
        let f1 = make_test_frame(width, height, channels, 20);

        encoder.add_frame(&f0, 1).expect("frame 0");
        encoder.add_frame(&f1, 1).expect("frame 1");

        let codestream = encoder.finish().expect("finish ok");

        let decoder = JxlDecoder::new();
        let frames = decoder.decode_animated(&codestream).expect("decode ok");

        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].channels, 1);
        assert_eq!(frames[0].data, f0);
        assert_eq!(frames[1].data, f1);
        assert!(frames[1].is_last);
    }

    #[test]
    fn test_animated_encoder_rgba() {
        let anim = JxlAnimation::millisecond();
        let width = 4u32;
        let height = 4u32;
        let channels = 4u8;

        let mut encoder =
            AnimatedJxlEncoder::new(anim, width, height, channels, 8).expect("create ok");

        let f0 = make_test_frame(width, height, channels, 0);
        let f1 = make_test_frame(width, height, channels, 1);

        encoder.add_frame(&f0, 50).expect("frame 0");
        encoder.add_frame(&f1, 50).expect("frame 1");

        let codestream = encoder.finish().expect("finish ok");

        let decoder = JxlDecoder::new();
        let frames = decoder.decode_animated(&codestream).expect("decode ok");

        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].channels, 4);
        assert_eq!(frames[0].data, f0, "RGBA frame 0 mismatch");
        assert_eq!(frames[1].data, f1, "RGBA frame 1 mismatch");
    }

    #[test]
    fn test_animated_encoder_with_effort() {
        let anim = JxlAnimation::millisecond();
        let width = 4u32;
        let height = 4u32;
        let channels = 3u8;

        let mut encoder = AnimatedJxlEncoder::new(anim, width, height, channels, 8)
            .expect("create ok")
            .with_effort(3);

        let f0 = make_test_frame(width, height, channels, 0);
        encoder.add_frame(&f0, 100).expect("frame ok");

        let codestream = encoder.finish().expect("finish ok");

        let decoder = JxlDecoder::new();
        let frames = decoder.decode_animated(&codestream).expect("decode ok");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].data, f0);
    }

    #[test]
    fn test_animated_encoder_animation_header_roundtrip() {
        let anim = JxlAnimation::new(30, 1)
            .expect("valid")
            .with_num_loops(5)
            .with_timecodes(true);

        let width = 2u32;
        let height = 2u32;
        let channels = 3u8;

        let mut encoder =
            AnimatedJxlEncoder::new(anim.clone(), width, height, channels, 8).expect("create ok");
        encoder
            .add_frame(&make_test_frame(width, height, channels, 0), 1)
            .expect("ok");

        let codestream = encoder.finish().expect("finish ok");

        let decoder = JxlDecoder::new();
        let header_anim = decoder
            .read_animation_header(&codestream)
            .expect("read ok")
            .expect("should be animated");

        assert_eq!(header_anim.tps_numerator, 30);
        assert_eq!(header_anim.tps_denominator, 1);
        assert_eq!(header_anim.num_loops, 5);
        assert!(header_anim.have_timecodes);
    }

    #[test]
    fn test_animated_encoder_is_animated_check() {
        // Animated codestream
        let anim = JxlAnimation::millisecond();
        let mut anim_enc = AnimatedJxlEncoder::new(anim, 2, 2, 3, 8).expect("create ok");
        anim_enc
            .add_frame(&make_test_frame(2, 2, 3, 0), 100)
            .expect("ok");
        let animated_cs = anim_enc.finish().expect("ok");

        let decoder = JxlDecoder::new();
        assert!(decoder.is_animated(&animated_cs).expect("check ok"));

        // Non-animated codestream
        let encoder = JxlEncoder::lossless();
        let data = vec![0u8; 8 * 8 * 3];
        let still_cs = encoder.encode(&data, 8, 8, 3, 8).expect("ok");
        assert!(!decoder.is_animated(&still_cs).expect("check ok"));
    }

    #[test]
    fn test_still_image_backwards_compat() {
        // Ensure the existing single-frame encode/decode pipeline is unchanged
        let width = 4u32;
        let height = 4u32;
        let channels = 3u8;

        let data = make_test_frame(width, height, channels, 99);

        let encoder = JxlEncoder::lossless();
        let encoded = encoder
            .encode(&data, width, height, channels, 8)
            .expect("encode ok");

        // Decode with the standard decoder
        let decoder = JxlDecoder::new();
        let decoded = decoder.decode(&encoded).expect("decode ok");
        assert_eq!(decoded.data, data);

        // Also decode with decode_animated: should give 1 frame
        let frames = decoder
            .decode_animated(&encoded)
            .expect("animated decode ok");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].duration_ticks, 0);
        assert!(frames[0].is_last);
        assert_eq!(frames[0].data, data);
    }

    #[test]
    fn test_animated_encoder_many_frames() {
        let anim = JxlAnimation::millisecond();
        let width = 2u32;
        let height = 2u32;
        let channels = 3u8;

        let mut encoder =
            AnimatedJxlEncoder::new(anim, width, height, channels, 8).expect("create ok");

        let frame_count = 10;
        let mut test_frames = Vec::with_capacity(frame_count);
        for i in 0..frame_count {
            let f = make_test_frame(width, height, channels, i as u8);
            encoder.add_frame(&f, (i as u32 + 1) * 50).expect("ok");
            test_frames.push(f);
        }

        let codestream = encoder.finish().expect("finish ok");

        let decoder = JxlDecoder::new();
        let frames = decoder.decode_animated(&codestream).expect("decode ok");

        assert_eq!(frames.len(), frame_count);
        for (i, frame) in frames.iter().enumerate() {
            assert_eq!(frame.duration_ticks, (i as u32 + 1) * 50);
            assert_eq!(frame.data, test_frames[i], "Frame {i} data mismatch");
            assert_eq!(frame.is_last, i == frame_count - 1);
        }
    }
}
