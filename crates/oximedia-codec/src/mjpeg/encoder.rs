//! MJPEG encoder implementation.
//!
//! Wraps the `oximedia-image` JPEG baseline encoder to produce
//! Motion JPEG video — each frame is independently JPEG-encoded.

use super::types::{MjpegConfig, MjpegError};
use crate::error::{CodecError, CodecResult};
use crate::frame::VideoFrame;
use crate::traits::{BitrateMode, EncodedPacket, EncoderConfig, VideoEncoder};
use oximedia_core::{CodecId, PixelFormat, Rational};
use oximedia_image::jpeg::{JpegEncoder, JpegQuality};
use oximedia_image::{ColorSpace, ImageData, ImageFrame, PixelType};

/// MJPEG encoder.
///
/// Each video frame is independently encoded as a baseline JPEG image.
/// This is the standard Motion JPEG encoding scheme used in AVI, MOV,
/// and other container formats.
///
/// # Architecture
///
/// - Accepts `VideoFrame` (planar YUV or packed RGB)
/// - Converts to interleaved RGB if necessary
/// - Encodes as baseline JPEG via `oximedia-image`
/// - Adds AVI1 APP0 marker for MJPEG identification
/// - Outputs as `EncodedPacket` with keyframe=true (all MJPEG frames are intra)
#[derive(Debug)]
pub struct MjpegEncoder {
    /// Encoder configuration (unified codec config).
    config: EncoderConfig,
    /// JPEG quality (1-100).
    quality: u8,
    /// Frame counter for PTS generation.
    frame_count: u64,
    /// Pending output packets.
    output_queue: Vec<EncodedPacket>,
    /// Whether the encoder is in flush mode.
    flushing: bool,
    /// Internal JPEG encoder instance.
    jpeg_encoder: JpegEncoder,
}

impl MjpegEncoder {
    /// Create a new MJPEG encoder.
    ///
    /// # Arguments
    ///
    /// * `config` - MJPEG-specific configuration
    ///
    /// # Errors
    ///
    /// Returns `CodecError::InvalidParameter` if configuration is invalid.
    pub fn new(config: MjpegConfig) -> CodecResult<Self> {
        config.validate().map_err(CodecError::from)?;

        let encoder_config = EncoderConfig {
            codec: CodecId::Mjpeg,
            width: config.width,
            height: config.height,
            pixel_format: config.pixel_format,
            framerate: Rational::new(30, 1),
            bitrate: BitrateMode::Crf(f32::from(100 - config.quality)),
            preset: crate::traits::EncoderPreset::Medium,
            profile: Some("baseline".to_string()),
            keyint: 1, // Every frame is a keyframe in MJPEG
            threads: 0,
            timebase: Rational::new(1, 1000),
        };

        let jpeg_encoder = JpegEncoder::new(JpegQuality::new(config.quality));

        Ok(Self {
            config: encoder_config,
            quality: config.quality,
            frame_count: 0,
            output_queue: Vec::new(),
            flushing: false,
            jpeg_encoder,
        })
    }

    /// Create an MJPEG encoder with default settings for the given dimensions.
    ///
    /// # Errors
    ///
    /// Returns error if dimensions are invalid.
    pub fn with_dimensions(width: u32, height: u32) -> CodecResult<Self> {
        let config = MjpegConfig::new(width, height).map_err(CodecError::from)?;
        Self::new(config)
    }

    /// Get the current quality setting.
    #[must_use]
    pub fn quality(&self) -> u8 {
        self.quality
    }

    /// Get the number of frames encoded so far.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Encode a single video frame as JPEG.
    ///
    /// Converts the VideoFrame to an ImageFrame, encodes it as JPEG,
    /// and wraps the result with AVI1 marker.
    fn encode_frame(&mut self, frame: &VideoFrame) -> CodecResult<()> {
        // Convert VideoFrame → interleaved RGB bytes → ImageFrame
        let rgb_data = self.convert_to_rgb(frame)?;

        let image_frame = ImageFrame::new(
            self.frame_count as u32,
            frame.width,
            frame.height,
            PixelType::U8,
            3,
            ColorSpace::Srgb,
            ImageData::interleaved(rgb_data),
        );

        // Encode as JPEG
        let jpeg_data = self
            .jpeg_encoder
            .encode(&image_frame)
            .map_err(|e| CodecError::Internal(format!("MJPEG encode failed: {e}")))?;

        // Insert AVI1 marker into the JPEG stream
        let mjpeg_data = Self::inject_avi1_marker(&jpeg_data);

        #[allow(clippy::cast_possible_wrap)]
        let pts = self.frame_count as i64;

        self.output_queue.push(EncodedPacket {
            data: mjpeg_data,
            pts,
            dts: pts,
            keyframe: true, // All MJPEG frames are keyframes
            duration: Some(1),
        });

        self.frame_count += 1;
        Ok(())
    }

    /// Convert a `VideoFrame` to interleaved RGB bytes.
    ///
    /// Handles YUV420p, YUV422p, YUV444p, RGB24, and RGBA32 input formats.
    fn convert_to_rgb(&self, frame: &VideoFrame) -> CodecResult<Vec<u8>> {
        let w = frame.width as usize;
        let h = frame.height as usize;

        match frame.format {
            PixelFormat::Rgb24 => {
                // Already RGB — copy the first plane directly
                if frame.planes.is_empty() {
                    return Err(CodecError::InvalidParameter(
                        "RGB24 frame has no planes".to_string(),
                    ));
                }
                Ok(frame.planes[0].data.clone())
            }
            PixelFormat::Rgba32 => {
                // Strip alpha channel
                if frame.planes.is_empty() {
                    return Err(CodecError::InvalidParameter(
                        "RGBA32 frame has no planes".to_string(),
                    ));
                }
                let rgba = &frame.planes[0].data;
                let mut rgb = Vec::with_capacity(w * h * 3);
                for pixel in rgba.chunks_exact(4) {
                    rgb.push(pixel[0]);
                    rgb.push(pixel[1]);
                    rgb.push(pixel[2]);
                }
                Ok(rgb)
            }
            PixelFormat::Yuv420p => self.yuv420p_to_rgb(frame, w, h),
            PixelFormat::Yuv422p => self.yuv422p_to_rgb(frame, w, h),
            PixelFormat::Yuv444p => self.yuv444p_to_rgb(frame, w, h),
            other => Err(CodecError::InvalidParameter(format!(
                "MJPEG encoder does not support pixel format: {other:?}"
            ))),
        }
    }

    /// Convert YUV 4:2:0 planar to interleaved RGB.
    fn yuv420p_to_rgb(&self, frame: &VideoFrame, w: usize, h: usize) -> CodecResult<Vec<u8>> {
        if frame.planes.len() < 3 {
            return Err(CodecError::InvalidParameter(
                "YUV420p frame requires 3 planes".to_string(),
            ));
        }

        let y_plane = &frame.planes[0].data;
        let u_plane = &frame.planes[1].data;
        let v_plane = &frame.planes[2].data;

        let chroma_w = (w + 1) / 2;

        let mut rgb = Vec::with_capacity(w * h * 3);

        for row in 0..h {
            for col in 0..w {
                let y_idx = row * w + col;
                let c_row = row / 2;
                let c_col = col / 2;
                let c_idx = c_row * chroma_w + c_col;

                let y_val = y_plane.get(y_idx).copied().unwrap_or(0) as f32;
                let cb_val = u_plane.get(c_idx).copied().unwrap_or(128) as f32;
                let cr_val = v_plane.get(c_idx).copied().unwrap_or(128) as f32;

                let (r, g, b) = oximedia_image::jpeg::ycbcr_to_rgb(y_val, cb_val, cr_val);
                rgb.push(r);
                rgb.push(g);
                rgb.push(b);
            }
        }

        Ok(rgb)
    }

    /// Convert YUV 4:2:2 planar to interleaved RGB.
    fn yuv422p_to_rgb(&self, frame: &VideoFrame, w: usize, h: usize) -> CodecResult<Vec<u8>> {
        if frame.planes.len() < 3 {
            return Err(CodecError::InvalidParameter(
                "YUV422p frame requires 3 planes".to_string(),
            ));
        }

        let y_plane = &frame.planes[0].data;
        let u_plane = &frame.planes[1].data;
        let v_plane = &frame.planes[2].data;

        let chroma_w = (w + 1) / 2;

        let mut rgb = Vec::with_capacity(w * h * 3);

        for row in 0..h {
            for col in 0..w {
                let y_idx = row * w + col;
                let c_col = col / 2;
                let c_idx = row * chroma_w + c_col;

                let y_val = y_plane.get(y_idx).copied().unwrap_or(0) as f32;
                let cb_val = u_plane.get(c_idx).copied().unwrap_or(128) as f32;
                let cr_val = v_plane.get(c_idx).copied().unwrap_or(128) as f32;

                let (r, g, b) = oximedia_image::jpeg::ycbcr_to_rgb(y_val, cb_val, cr_val);
                rgb.push(r);
                rgb.push(g);
                rgb.push(b);
            }
        }

        Ok(rgb)
    }

    /// Convert YUV 4:4:4 planar to interleaved RGB.
    fn yuv444p_to_rgb(&self, frame: &VideoFrame, w: usize, h: usize) -> CodecResult<Vec<u8>> {
        if frame.planes.len() < 3 {
            return Err(CodecError::InvalidParameter(
                "YUV444p frame requires 3 planes".to_string(),
            ));
        }

        let y_plane = &frame.planes[0].data;
        let u_plane = &frame.planes[1].data;
        let v_plane = &frame.planes[2].data;

        let mut rgb = Vec::with_capacity(w * h * 3);

        for idx in 0..(w * h) {
            let y_val = y_plane.get(idx).copied().unwrap_or(0) as f32;
            let cb_val = u_plane.get(idx).copied().unwrap_or(128) as f32;
            let cr_val = v_plane.get(idx).copied().unwrap_or(128) as f32;

            let (r, g, b) = oximedia_image::jpeg::ycbcr_to_rgb(y_val, cb_val, cr_val);
            rgb.push(r);
            rgb.push(g);
            rgb.push(b);
        }

        Ok(rgb)
    }

    /// Inject an AVI1 APP0 marker into a JPEG byte stream.
    ///
    /// The AVI1 marker is placed after the JFIF APP0 segment.
    /// This identifies the JPEG frame as part of an MJPEG stream.
    ///
    /// Format: FF E0 <len> "AVI1" <polarity> <reserved×6>
    fn inject_avi1_marker(jpeg_data: &[u8]) -> Vec<u8> {
        // AVI1 APP0 segment content
        let avi1_payload: &[u8] = &[
            b'A', b'V', b'I', b'1', // Identifier
            0x00, // Polarity: 0 = even field
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Reserved
        ];

        // Find the end of the first APP0 (JFIF) marker to insert AVI1 after it
        let mut result = Vec::with_capacity(jpeg_data.len() + 2 + 2 + avi1_payload.len());

        // Look for APP0 marker (FF E0) after SOI (FF D8)
        let mut insert_pos = 2; // Default: after SOI
        if jpeg_data.len() >= 4 && jpeg_data[0] == 0xFF && jpeg_data[1] == 0xD8 {
            // Check for APP0 at position 2
            if jpeg_data.len() >= 6 && jpeg_data[2] == 0xFF && jpeg_data[3] == 0xE0 {
                // Read the APP0 segment length
                let app0_len = u16::from_be_bytes([jpeg_data[4], jpeg_data[5]]) as usize;
                insert_pos = 4 + app0_len; // After the full APP0 segment
            }
        }

        // Copy up to insert position
        let safe_pos = insert_pos.min(jpeg_data.len());
        result.extend_from_slice(&jpeg_data[..safe_pos]);

        // Write AVI1 APP0 marker
        result.push(0xFF);
        result.push(0xE0);
        let seg_len = (avi1_payload.len() + 2) as u16;
        result.extend_from_slice(&seg_len.to_be_bytes());
        result.extend_from_slice(avi1_payload);

        // Copy remaining JPEG data
        if safe_pos < jpeg_data.len() {
            result.extend_from_slice(&jpeg_data[safe_pos..]);
        }

        result
    }
}

impl VideoEncoder for MjpegEncoder {
    fn codec(&self) -> CodecId {
        CodecId::Mjpeg
    }

    fn send_frame(&mut self, frame: &VideoFrame) -> CodecResult<()> {
        if self.flushing {
            return Err(CodecError::InvalidParameter(
                "Cannot send frames after flush".to_string(),
            ));
        }
        self.encode_frame(frame)
    }

    fn receive_packet(&mut self) -> CodecResult<Option<EncodedPacket>> {
        if self.output_queue.is_empty() {
            Ok(None)
        } else {
            Ok(Some(self.output_queue.remove(0)))
        }
    }

    fn flush(&mut self) -> CodecResult<()> {
        self.flushing = true;
        // MJPEG has no buffered frames — each frame produces output immediately.
        // Nothing to flush.
        Ok(())
    }

    fn config(&self) -> &EncoderConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{FrameType, Plane};
    use oximedia_core::Timestamp;

    /// Create a synthetic RGB24 VideoFrame for testing.
    fn make_rgb_frame(width: u32, height: u32) -> VideoFrame {
        let size = (width * height * 3) as usize;
        let mut data = vec![0u8; size];
        // Create a simple gradient pattern
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                if idx + 2 < data.len() {
                    data[idx] = (x % 256) as u8; // R
                    data[idx + 1] = (y % 256) as u8; // G
                    data[idx + 2] = ((x + y) % 256) as u8; // B
                }
            }
        }
        let mut frame = VideoFrame::new(PixelFormat::Rgb24, width, height);
        frame.planes = vec![Plane::with_dimensions(
            data,
            (width * 3) as usize,
            width,
            height,
        )];
        frame.timestamp = Timestamp::new(0, Rational::new(1, 1000));
        frame.frame_type = FrameType::Key;
        frame
    }

    /// Create a synthetic YUV420p VideoFrame for testing.
    fn make_yuv420p_frame(width: u32, height: u32) -> VideoFrame {
        let y_size = (width * height) as usize;
        let chroma_w = ((width + 1) / 2) as usize;
        let chroma_h = ((height + 1) / 2) as usize;
        let c_size = chroma_w * chroma_h;

        let y_data: Vec<u8> = (0..y_size).map(|i| (i % 235 + 16) as u8).collect();
        let u_data: Vec<u8> = (0..c_size).map(|i| (i % 225 + 16) as u8).collect();
        let v_data: Vec<u8> = (0..c_size).map(|i| (i % 225 + 16) as u8).collect();

        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, width, height);
        frame.planes = vec![
            Plane::with_dimensions(y_data, width as usize, width, height),
            Plane::with_dimensions(u_data, chroma_w, (width + 1) / 2, (height + 1) / 2),
            Plane::with_dimensions(v_data, chroma_w, (width + 1) / 2, (height + 1) / 2),
        ];
        frame.timestamp = Timestamp::new(0, Rational::new(1, 1000));
        frame.frame_type = FrameType::Key;
        frame
    }

    #[test]
    fn test_encoder_decoder_roundtrip_psnr() {
        use crate::mjpeg::decoder::MjpegDecoder;
        use crate::traits::VideoDecoder;

        const W: u32 = 16;
        const H: u32 = 16;
        // Simple gradient pattern – all values within 0-255
        let mut rgb = vec![0u8; (W * H * 3) as usize];
        for row in 0..H as usize {
            for col in 0..W as usize {
                let idx = (row * W as usize + col) * 3;
                rgb[idx] = (row * 10 + col * 5).min(255) as u8;
                rgb[idx + 1] = (row * 5 + col * 10).min(255) as u8;
                rgb[idx + 2] = (row * 8 + col * 3).min(255) as u8;
            }
        }

        let mut frame = VideoFrame::new(PixelFormat::Rgb24, W, H);
        frame.planes = vec![Plane::with_dimensions(rgb.clone(), (W * 3) as usize, W, H)];
        frame.timestamp = Timestamp::new(0, Rational::new(1, 1000));
        frame.frame_type = FrameType::Key;

        let config = MjpegConfig::new(W, H)
            .expect("valid config")
            .with_quality(85)
            .with_pixel_format(PixelFormat::Rgb24);
        let mut enc = MjpegEncoder::new(config).expect("valid encoder");
        enc.send_frame(&frame).expect("encode");
        let pkt = enc.receive_packet().expect("receive").expect("packet");

        let mut dec = MjpegDecoder::new(W, H);
        dec.send_packet(&pkt.data, pkt.pts).expect("decode");
        let decoded = dec.receive_frame().expect("recv frame").expect("frame");

        let dec_data = &decoded.planes[0].data;
        assert_eq!(dec_data.len(), rgb.len(), "decoded data length mismatch");

        let mse: f64 = rgb
            .iter()
            .zip(dec_data.iter())
            .map(|(&a, &b)| {
                let d = a as f64 - b as f64;
                d * d
            })
            .sum::<f64>()
            / rgb.len() as f64;
        let psnr = if mse < 1e-10 {
            f64::INFINITY
        } else {
            20.0 * (255.0_f64).log10() - 10.0 * mse.log10()
        };
        assert!(
            psnr >= 28.0,
            "MJPEG codec roundtrip Q85 PSNR ≥ 28 dB, got {psnr:.2} dB"
        );
    }

    #[test]
    fn test_encoder_creation() {
        let config = MjpegConfig::new(320, 240).expect("valid config");
        let encoder = MjpegEncoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_encoder_with_dimensions() {
        let encoder = MjpegEncoder::with_dimensions(640, 480);
        assert!(encoder.is_ok());
        let enc = encoder.expect("valid encoder");
        assert_eq!(enc.quality(), 85);
        assert_eq!(enc.frame_count(), 0);
    }

    #[test]
    fn test_encoder_invalid_dimensions() {
        let result = MjpegEncoder::with_dimensions(0, 480);
        assert!(result.is_err());
    }

    #[test]
    fn test_encoder_codec_id() {
        let enc = MjpegEncoder::with_dimensions(16, 16).expect("valid encoder");
        assert_eq!(enc.codec(), CodecId::Mjpeg);
    }

    #[test]
    fn test_encode_rgb_frame() {
        let mut enc = MjpegEncoder::with_dimensions(16, 16).expect("valid encoder");
        let frame = make_rgb_frame(16, 16);

        let result = enc.send_frame(&frame);
        assert!(result.is_ok(), "send_frame failed: {result:?}");

        let packet = enc.receive_packet().expect("receive_packet failed");
        assert!(packet.is_some(), "Expected a packet");

        let pkt = packet.expect("packet should be Some");
        assert!(pkt.keyframe, "MJPEG frames are always keyframes");
        assert!(!pkt.data.is_empty(), "Encoded data should not be empty");
        assert_eq!(pkt.pts, 0);

        // Verify JPEG SOI marker
        assert_eq!(pkt.data[0], 0xFF);
        assert_eq!(pkt.data[1], 0xD8);

        // Verify JPEG EOI marker at end
        let len = pkt.data.len();
        assert_eq!(pkt.data[len - 2], 0xFF);
        assert_eq!(pkt.data[len - 1], 0xD9);
    }

    #[test]
    fn test_encode_yuv420p_frame() {
        let config = MjpegConfig::new(16, 16)
            .expect("valid config")
            .with_pixel_format(PixelFormat::Yuv420p);
        let mut enc = MjpegEncoder::new(config).expect("valid encoder");
        let frame = make_yuv420p_frame(16, 16);

        let result = enc.send_frame(&frame);
        assert!(result.is_ok(), "YUV420p encode failed: {result:?}");

        let packet = enc.receive_packet().expect("receive_packet failed");
        assert!(packet.is_some());
    }

    #[test]
    fn test_encode_multiple_frames() {
        let mut enc = MjpegEncoder::with_dimensions(16, 16).expect("valid encoder");

        for i in 0..5 {
            let mut frame = make_rgb_frame(16, 16);
            frame.timestamp = Timestamp::new(i, Rational::new(1, 1000));
            enc.send_frame(&frame).expect("send_frame failed");
        }

        assert_eq!(enc.frame_count(), 5);

        // Drain all packets
        for i in 0..5 {
            let pkt = enc
                .receive_packet()
                .expect("receive_packet failed")
                .expect("expected packet");
            assert_eq!(pkt.pts, i);
            assert!(pkt.keyframe);
        }

        // No more packets
        let pkt = enc.receive_packet().expect("receive_packet");
        assert!(pkt.is_none());
    }

    #[test]
    fn test_avi1_marker_injection() {
        // Create a minimal JPEG: SOI + APP0(JFIF) + EOI
        let mut jpeg = Vec::new();
        jpeg.extend_from_slice(&[0xFF, 0xD8]); // SOI
                                               // APP0 JFIF
        jpeg.extend_from_slice(&[0xFF, 0xE0]); // APP0 marker
        let jfif_data = b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00";
        let seg_len = (jfif_data.len() + 2) as u16;
        jpeg.extend_from_slice(&seg_len.to_be_bytes());
        jpeg.extend_from_slice(jfif_data);
        jpeg.extend_from_slice(&[0xFF, 0xD9]); // EOI

        let result = MjpegEncoder::inject_avi1_marker(&jpeg);

        // Should still start with SOI
        assert_eq!(result[0], 0xFF);
        assert_eq!(result[1], 0xD8);

        // Should contain AVI1 string somewhere after JFIF
        let avi1_pos = result.windows(4).position(|w| w == b"AVI1");
        assert!(avi1_pos.is_some(), "AVI1 marker not found in output");

        // Should end with EOI
        let len = result.len();
        assert_eq!(result[len - 2], 0xFF);
        assert_eq!(result[len - 1], 0xD9);
    }

    #[test]
    fn test_flush_prevents_more_frames() {
        let mut enc = MjpegEncoder::with_dimensions(16, 16).expect("valid encoder");
        enc.flush().expect("flush should succeed");

        let frame = make_rgb_frame(16, 16);
        let result = enc.send_frame(&frame);
        assert!(result.is_err(), "Should not accept frames after flush");
    }

    #[test]
    fn test_config_reflects_mjpeg() {
        let enc = MjpegEncoder::with_dimensions(640, 480).expect("valid encoder");
        let config = enc.config();
        assert_eq!(config.codec, CodecId::Mjpeg);
        assert_eq!(config.width, 640);
        assert_eq!(config.height, 480);
        assert_eq!(config.keyint, 1);
    }

    #[test]
    fn test_encode_quality_1() {
        let config = MjpegConfig::new(16, 16)
            .expect("valid config")
            .with_quality(1);
        let mut enc = MjpegEncoder::new(config).expect("valid encoder");
        let frame = make_rgb_frame(16, 16);
        enc.send_frame(&frame).expect("encode at quality 1");
        let pkt = enc
            .receive_packet()
            .expect("receive")
            .expect("packet expected");
        assert!(!pkt.data.is_empty());
    }

    #[test]
    fn test_encode_quality_100() {
        let config = MjpegConfig::new(16, 16)
            .expect("valid config")
            .with_quality(100);
        let mut enc = MjpegEncoder::new(config).expect("valid encoder");
        let frame = make_rgb_frame(16, 16);
        enc.send_frame(&frame).expect("encode at quality 100");
        let pkt = enc
            .receive_packet()
            .expect("receive")
            .expect("packet expected");
        assert!(!pkt.data.is_empty());
    }

    #[test]
    fn test_encode_non_multiple_of_8() {
        // JPEG MCU blocks are 8x8, test non-aligned dimensions
        let mut enc = MjpegEncoder::with_dimensions(13, 7).expect("valid encoder");
        let frame = make_rgb_frame(13, 7);
        enc.send_frame(&frame).expect("encode non-aligned frame");
        let pkt = enc
            .receive_packet()
            .expect("receive")
            .expect("packet expected");
        assert!(!pkt.data.is_empty());
    }

    #[test]
    fn test_higher_quality_produces_larger_output() {
        // Quality 100 should produce more data than quality 1
        let frame = make_rgb_frame(32, 32);

        let config_low = MjpegConfig::new(32, 32)
            .expect("valid config")
            .with_quality(1);
        let mut enc_low = MjpegEncoder::new(config_low).expect("valid encoder");
        enc_low.send_frame(&frame).expect("low quality encode");
        let pkt_low = enc_low.receive_packet().expect("receive").expect("packet");

        let config_high = MjpegConfig::new(32, 32)
            .expect("valid config")
            .with_quality(100);
        let mut enc_high = MjpegEncoder::new(config_high).expect("valid encoder");
        enc_high.send_frame(&frame).expect("high quality encode");
        let pkt_high = enc_high.receive_packet().expect("receive").expect("packet");

        assert!(
            pkt_high.data.len() >= pkt_low.data.len(),
            "Higher quality should produce larger or equal output: {} vs {}",
            pkt_high.data.len(),
            pkt_low.data.len()
        );
    }
}
