//! FFV1 encoder implementation.
//!
//! Encodes raw video frames into FFV1 lossless bitstreams as specified
//! in RFC 9043. Supports version 3 with range coder and CRC-32 error
//! detection.

use crate::error::{CodecError, CodecResult};
use crate::frame::VideoFrame;
use crate::traits::{EncodedPacket, EncoderConfig, VideoEncoder};
use oximedia_core::CodecId;

use super::crc32::crc32_mpeg2;
use super::prediction::predict_median;
use super::range_coder::SimpleRangeEncoder;
use super::types::{
    Ffv1ChromaType, Ffv1Colorspace, Ffv1Config, Ffv1Version, CONTEXT_COUNT, INITIAL_STATE,
};

/// FFV1 encoder.
///
/// Implements the `VideoEncoder` trait for encoding raw video frames
/// into FFV1 lossless bitstreams.
///
/// # Usage
///
/// ```ignore
/// use oximedia_codec::ffv1::Ffv1Encoder;
/// use oximedia_codec::traits::EncoderConfig;
///
/// let config = EncoderConfig::default();
/// let mut encoder = Ffv1Encoder::new(config)?;
/// encoder.send_frame(&frame)?;
/// if let Some(packet) = encoder.receive_packet()? {
///     // Write packet to container
/// }
/// ```
pub struct Ffv1Encoder {
    /// Base encoder configuration.
    config: EncoderConfig,
    /// FFV1-specific configuration.
    ffv1_config: Ffv1Config,
    /// Output packet queue.
    output_queue: Vec<EncodedPacket>,
    /// Whether the encoder is in flush mode.
    flushing: bool,
    /// Number of encoded frames.
    frame_count: u64,
    /// Per-plane context states for range coder.
    plane_states: Vec<Vec<u8>>,
}

impl Ffv1Encoder {
    /// Create a new FFV1 encoder with default FFV1 settings.
    pub fn new(config: EncoderConfig) -> CodecResult<Self> {
        let ffv1_config = Ffv1Config {
            version: Ffv1Version::V3,
            width: config.width,
            height: config.height,
            colorspace: Ffv1Colorspace::YCbCr,
            chroma_type: Ffv1ChromaType::Chroma420,
            bits_per_raw_sample: 8,
            num_h_slices: 1,
            num_v_slices: 1,
            ec: true,
            range_coder_mode: true,
            state_transition_delta: Vec::new(),
        };
        Self::with_ffv1_config(config, ffv1_config)
    }

    /// Create an FFV1 encoder with explicit FFV1 configuration.
    pub fn with_ffv1_config(config: EncoderConfig, ffv1: Ffv1Config) -> CodecResult<Self> {
        if config.width == 0 || config.height == 0 {
            return Err(CodecError::InvalidParameter(
                "frame dimensions must be nonzero".to_string(),
            ));
        }

        let mut ffv1_config = ffv1;
        ffv1_config.width = config.width;
        ffv1_config.height = config.height;
        ffv1_config.validate()?;

        let plane_count = ffv1_config.plane_count();
        let plane_states: Vec<Vec<u8>> = (0..plane_count)
            .map(|_| vec![INITIAL_STATE; CONTEXT_COUNT])
            .collect();

        Ok(Self {
            config,
            ffv1_config,
            output_queue: Vec::new(),
            flushing: false,
            frame_count: 0,
            plane_states,
        })
    }

    /// Reset all context states (done at keyframes).
    fn reset_states(&mut self) {
        for states in &mut self.plane_states {
            for s in states.iter_mut() {
                *s = INITIAL_STATE;
            }
        }
    }

    /// Generate the FFV1 extradata (configuration record) for the container.
    ///
    /// This must be stored in the container's codec private data so the
    /// decoder can initialize correctly.
    #[must_use]
    pub fn extradata(&self) -> Vec<u8> {
        let c = &self.ffv1_config;
        let mut data = Vec::with_capacity(16);
        data.push(c.version.as_u8());
        data.push(c.colorspace.as_u8());
        data.push(c.chroma_type.h_shift() as u8);
        data.push(c.chroma_type.v_shift() as u8);
        data.push(c.bits_per_raw_sample);
        data.push(if c.ec { 1 } else { 0 });
        data.push(c.num_h_slices as u8);
        data.push(c.num_v_slices as u8);
        data.extend_from_slice(&c.width.to_le_bytes());
        data.extend_from_slice(&c.height.to_le_bytes());
        data
    }

    /// Encode a single frame into a compressed bitstream.
    fn encode_frame(&mut self, frame: &VideoFrame) -> CodecResult<Vec<u8>> {
        // Extract all needed config values upfront to avoid borrow conflicts.
        let plane_count = self.ffv1_config.plane_count();
        let cfg_width = self.ffv1_config.width;
        let cfg_height = self.ffv1_config.height;
        let ec = self.ffv1_config.ec;
        let version = self.ffv1_config.version;
        let is_keyframe = self.frame_count % u64::from(self.config.keyint) == 0;

        // Collect plane dimensions
        let plane_dims: Vec<(u32, u32)> = (0..plane_count)
            .map(|i| self.ffv1_config.plane_dimensions(i))
            .collect();

        if is_keyframe {
            self.reset_states();
        }

        // Validate frame dimensions
        if frame.width != cfg_width || frame.height != cfg_height {
            return Err(CodecError::InvalidParameter(format!(
                "frame dimensions {}x{} do not match encoder config {}x{}",
                frame.width, frame.height, cfg_width, cfg_height
            )));
        }

        if frame.planes.len() < plane_count {
            return Err(CodecError::InvalidParameter(format!(
                "frame has {} planes, need at least {}",
                frame.planes.len(),
                plane_count
            )));
        }

        // Encode all planes into a single range-coded bitstream
        let mut encoder = SimpleRangeEncoder::new();

        for plane_idx in 0..plane_count {
            let (pw, ph) = plane_dims[plane_idx];
            let plane = &frame.planes[plane_idx];

            let states = &mut self.plane_states[plane_idx];
            let mut prev_line = vec![0i32; pw as usize];

            for y in 0..ph as usize {
                for x in 0..pw as usize {
                    // Get sample from plane data
                    let sample = if y < plane.height as usize && x < plane.width as usize {
                        i32::from(plane.data[y * plane.stride + x])
                    } else {
                        0
                    };

                    // Compute prediction
                    let left = if x > 0 {
                        // Use the actual sample we just encoded for the left neighbor
                        // (we need to track the reconstructed line)
                        i32::from(plane.data[y * plane.stride + x - 1])
                    } else {
                        0
                    };
                    let top = prev_line[x];
                    let top_left = if x > 0 { prev_line[x - 1] } else { 0 };

                    let pred = predict_median(left, top, top_left);
                    let residual = sample - pred;

                    encoder.put_symbol(states, residual);

                    // Update prev_line with actual sample for next row's prediction
                    if x == 0 && y > 0 {
                        // Fill prev_line from the previous row
                    }
                }

                // Update prev_line with the current row's actual samples
                for x in 0..pw as usize {
                    prev_line[x] = if y < plane.height as usize && x < plane.width as usize {
                        i32::from(plane.data[y * plane.stride + x])
                    } else {
                        0
                    };
                }
            }
        }

        let mut payload = encoder.finish();

        // Append CRC-32 for v3 with EC enabled
        if ec && version == Ffv1Version::V3 {
            let crc = crc32_mpeg2(&payload);
            payload.extend_from_slice(&crc.to_le_bytes());
        }

        Ok(payload)
    }
}

impl VideoEncoder for Ffv1Encoder {
    fn codec(&self) -> CodecId {
        CodecId::Ffv1
    }

    fn send_frame(&mut self, frame: &VideoFrame) -> CodecResult<()> {
        if self.flushing {
            return Err(CodecError::InvalidParameter(
                "encoder is flushing, cannot accept new frames".to_string(),
            ));
        }

        let pts = frame.timestamp.pts;
        let is_keyframe = self.frame_count % u64::from(self.config.keyint) == 0;

        let data = self.encode_frame(frame)?;

        let packet = EncodedPacket {
            data,
            pts,
            dts: pts,
            keyframe: is_keyframe,
            duration: None,
        };

        self.output_queue.push(packet);
        self.frame_count += 1;
        Ok(())
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
        Ok(())
    }

    fn config(&self) -> &EncoderConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Plane;
    use crate::traits::VideoDecoder;
    use oximedia_core::{PixelFormat, Rational, Timestamp};

    fn make_encoder_config(width: u32, height: u32) -> EncoderConfig {
        EncoderConfig {
            codec: CodecId::Ffv1,
            width,
            height,
            pixel_format: PixelFormat::Yuv420p,
            framerate: Rational::new(30, 1),
            bitrate: crate::traits::BitrateMode::Lossless,
            preset: crate::traits::EncoderPreset::Medium,
            profile: None,
            keyint: 1,
            threads: 1,
            timebase: Rational::new(1, 1000),
        }
    }

    fn make_test_frame(width: u32, height: u32) -> VideoFrame {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, width, height);
        frame.timestamp = Timestamp::new(0, Rational::new(1, 1000));

        // Y plane
        let y_size = (width * height) as usize;
        let mut y_data = vec![0u8; y_size];
        for y in 0..height as usize {
            for x in 0..width as usize {
                y_data[y * width as usize + x] = ((x + y) % 256) as u8;
            }
        }
        frame.planes.push(Plane::with_dimensions(
            y_data,
            width as usize,
            width,
            height,
        ));

        // U plane (half resolution for 4:2:0)
        let cw = (width + 1) / 2;
        let ch = (height + 1) / 2;
        let u_data = vec![128u8; (cw * ch) as usize];
        frame
            .planes
            .push(Plane::with_dimensions(u_data, cw as usize, cw, ch));

        // V plane
        let v_data = vec![128u8; (cw * ch) as usize];
        frame
            .planes
            .push(Plane::with_dimensions(v_data, cw as usize, cw, ch));

        frame
    }

    #[test]
    #[ignore]
    fn test_encoder_creation() {
        let config = make_encoder_config(320, 240);
        let enc = Ffv1Encoder::new(config).expect("valid config");
        assert_eq!(enc.codec(), CodecId::Ffv1);
    }

    #[test]
    #[ignore]
    fn test_encoder_invalid_dimensions() {
        let config = make_encoder_config(0, 240);
        assert!(Ffv1Encoder::new(config).is_err());
    }

    #[test]
    #[ignore]
    fn test_encoder_extradata() {
        let config = make_encoder_config(320, 240);
        let enc = Ffv1Encoder::new(config).expect("valid");
        let extra = enc.extradata();
        assert!(extra.len() >= 13);
        assert_eq!(extra[0], 3); // version V3
    }

    #[test]
    #[ignore]
    fn test_encode_single_frame() {
        let config = make_encoder_config(16, 16);
        let mut enc = Ffv1Encoder::new(config).expect("valid");
        let frame = make_test_frame(16, 16);

        enc.send_frame(&frame).expect("encode ok");
        let packet = enc.receive_packet().expect("ok");
        assert!(packet.is_some());
        let pkt = packet.expect("packet");
        assert!(pkt.keyframe);
        assert!(!pkt.data.is_empty());
    }

    #[test]
    #[ignore]
    fn test_encode_wrong_dimensions() {
        let config = make_encoder_config(16, 16);
        let mut enc = Ffv1Encoder::new(config).expect("valid");
        let frame = make_test_frame(32, 32); // wrong size
        assert!(enc.send_frame(&frame).is_err());
    }

    #[test]
    #[ignore]
    fn test_encoder_flush() {
        let config = make_encoder_config(16, 16);
        let mut enc = Ffv1Encoder::new(config).expect("valid");
        enc.flush().expect("flush ok");
        let frame = make_test_frame(16, 16);
        assert!(enc.send_frame(&frame).is_err());
    }

    #[test]
    #[ignore]
    fn test_lossless_roundtrip() {
        // Encode a frame, then decode it, verify pixel-perfect roundtrip
        let width = 16u32;
        let height = 16u32;

        let enc_config = make_encoder_config(width, height);
        let mut encoder = Ffv1Encoder::new(enc_config).expect("enc init");
        let frame = make_test_frame(width, height);

        encoder.send_frame(&frame).expect("encode");
        let packet = encoder.receive_packet().expect("ok").expect("has packet");

        // Now decode
        let extradata = encoder.extradata();
        let mut decoder =
            super::super::decoder::Ffv1Decoder::with_extradata(&extradata).expect("dec init");

        decoder
            .send_packet(&packet.data, packet.pts)
            .expect("decode");
        let decoded_frame = decoder.receive_frame().expect("ok").expect("has frame");

        // Verify lossless: all planes must match exactly
        assert_eq!(decoded_frame.planes.len(), frame.planes.len());
        for (pi, (orig_plane, dec_plane)) in frame
            .planes
            .iter()
            .zip(decoded_frame.planes.iter())
            .enumerate()
        {
            assert_eq!(
                orig_plane.width, dec_plane.width,
                "plane {pi} width mismatch"
            );
            assert_eq!(
                orig_plane.height, dec_plane.height,
                "plane {pi} height mismatch"
            );

            for y in 0..orig_plane.height as usize {
                for x in 0..orig_plane.width as usize {
                    let orig_sample = orig_plane.data[y * orig_plane.stride + x];
                    let dec_sample = dec_plane.data[y * dec_plane.stride + x];
                    assert_eq!(
                        orig_sample, dec_sample,
                        "plane {pi} sample mismatch at ({x}, {y}): orig={orig_sample}, decoded={dec_sample}"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn test_lossless_roundtrip_constant_frame() {
        let width = 8u32;
        let height = 8u32;
        let enc_config = make_encoder_config(width, height);
        let mut encoder = Ffv1Encoder::new(enc_config).expect("enc init");

        // Create a constant frame (all 100)
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, width, height);
        frame.timestamp = Timestamp::new(0, Rational::new(1, 1000));
        let y_data = vec![100u8; (width * height) as usize];
        frame.planes.push(Plane::with_dimensions(
            y_data,
            width as usize,
            width,
            height,
        ));
        let cw = (width + 1) / 2;
        let ch = (height + 1) / 2;
        frame.planes.push(Plane::with_dimensions(
            vec![128u8; (cw * ch) as usize],
            cw as usize,
            cw,
            ch,
        ));
        frame.planes.push(Plane::with_dimensions(
            vec![128u8; (cw * ch) as usize],
            cw as usize,
            cw,
            ch,
        ));

        encoder.send_frame(&frame).expect("encode");
        let packet = encoder.receive_packet().expect("ok").expect("packet");

        let extradata = encoder.extradata();
        let mut decoder =
            super::super::decoder::Ffv1Decoder::with_extradata(&extradata).expect("dec");

        decoder.send_packet(&packet.data, 0).expect("decode");
        let decoded = decoder.receive_frame().expect("ok").expect("frame");

        for (pi, (orig, dec)) in frame.planes.iter().zip(decoded.planes.iter()).enumerate() {
            for y in 0..orig.height as usize {
                for x in 0..orig.width as usize {
                    assert_eq!(
                        orig.data[y * orig.stride + x],
                        dec.data[y * dec.stride + x],
                        "mismatch at plane {pi} ({x}, {y})"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn test_lossless_roundtrip_random_pattern() {
        let width = 32u32;
        let height = 32u32;
        let enc_config = make_encoder_config(width, height);
        let mut encoder = Ffv1Encoder::new(enc_config).expect("enc init");

        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, width, height);
        frame.timestamp = Timestamp::new(1000, Rational::new(1, 1000));

        // Create a more complex pattern
        let y_size = (width * height) as usize;
        let mut y_data = vec![0u8; y_size];
        for i in 0..y_size {
            // Pseudo-random but deterministic pattern
            y_data[i] = ((i * 37 + 13) % 256) as u8;
        }
        frame.planes.push(Plane::with_dimensions(
            y_data,
            width as usize,
            width,
            height,
        ));
        let cw = (width + 1) / 2;
        let ch = (height + 1) / 2;
        let uv_size = (cw * ch) as usize;
        let mut u_data = vec![0u8; uv_size];
        let mut v_data = vec![0u8; uv_size];
        for i in 0..uv_size {
            u_data[i] = ((i * 53 + 7) % 256) as u8;
            v_data[i] = ((i * 71 + 23) % 256) as u8;
        }
        frame
            .planes
            .push(Plane::with_dimensions(u_data, cw as usize, cw, ch));
        frame
            .planes
            .push(Plane::with_dimensions(v_data, cw as usize, cw, ch));

        encoder.send_frame(&frame).expect("encode");
        let packet = encoder.receive_packet().expect("ok").expect("packet");

        let extradata = encoder.extradata();
        let mut decoder =
            super::super::decoder::Ffv1Decoder::with_extradata(&extradata).expect("dec");

        decoder.send_packet(&packet.data, 1000).expect("decode");
        let decoded = decoder.receive_frame().expect("ok").expect("frame");

        for (pi, (orig, dec)) in frame.planes.iter().zip(decoded.planes.iter()).enumerate() {
            for y in 0..orig.height as usize {
                for x in 0..orig.width as usize {
                    assert_eq!(
                        orig.data[y * orig.stride + x],
                        dec.data[y * dec.stride + x],
                        "mismatch at plane {pi} ({x}, {y})"
                    );
                }
            }
        }
    }
}
