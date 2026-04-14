//! VP8 decoder implementation.
//!
//! This module provides the main VP8 decoder that implements the
//! `VideoDecoder` trait from this crate.
//!
//! VP8 is a royalty-free video codec developed by Google as part of
//! the `WebM` project. This decoder is based on RFC 6386.

use crate::error::{CodecError, CodecResult};
use crate::frame::{FrameType, Plane, VideoFrame};
use crate::traits::{DecoderConfig, VideoDecoder};
use crate::vp8::frame_header::FrameHeader;
use oximedia_core::{CodecId, PixelFormat, Rational, Timestamp};

/// VP8 decoder.
///
/// A pure Rust implementation of the VP8 video decoder.
/// VP8 is simpler than VP9 and uses a different bitstream format
/// based on boolean arithmetic coding.
///
/// # Examples
///
/// ```
/// use oximedia_codec::vp8::Vp8Decoder;
/// use oximedia_codec::traits::{DecoderConfig, VideoDecoder};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let config = DecoderConfig::default();
/// let mut decoder = Vp8Decoder::new(config)?;
///
/// // Decoder is ready to receive packets
/// assert!(decoder.dimensions().is_none());
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct Vp8Decoder {
    /// Decoder configuration.
    #[allow(dead_code)]
    config: DecoderConfig,
    /// Current frame width.
    width: Option<u32>,
    /// Current frame height.
    height: Option<u32>,
    /// Pending frame output queue.
    output_queue: Vec<VideoFrame>,
    /// Last reference frame (for inter prediction).
    #[allow(dead_code)]
    last_frame: Option<VideoFrame>,
    /// Golden reference frame.
    #[allow(dead_code)]
    golden_frame: Option<VideoFrame>,
    /// Alternate reference frame.
    #[allow(dead_code)]
    altref_frame: Option<VideoFrame>,
    /// Whether the decoder is in flush mode.
    flushing: bool,
}

impl Vp8Decoder {
    /// Creates a new VP8 decoder with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Decoder configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_codec::vp8::Vp8Decoder;
    /// use oximedia_codec::traits::DecoderConfig;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let config = DecoderConfig::default();
    /// let decoder = Vp8Decoder::new(config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(config: DecoderConfig) -> CodecResult<Self> {
        Ok(Self {
            config,
            width: None,
            height: None,
            output_queue: Vec::new(),
            last_frame: None,
            golden_frame: None,
            altref_frame: None,
            flushing: false,
        })
    }

    /// Decodes a VP8 frame from the input data.
    fn decode_frame(&mut self, data: &[u8], pts: i64) -> CodecResult<()> {
        // Parse frame header
        let header = FrameHeader::parse(data)?;

        // Update dimensions for keyframes
        if header.is_keyframe() {
            self.width = Some(u32::from(header.width));
            self.height = Some(u32::from(header.height));
        }

        let (Some(width), Some(height)) = (self.width, self.height) else {
            return Err(CodecError::InvalidBitstream(
                "VP8: No keyframe received yet, cannot decode inter frame".to_string(),
            ));
        };

        // Create output frame
        // In a full implementation, we would decode the actual pixel data here.
        // For now, we create a frame with allocated but initialized planes.
        let frame = self.create_output_frame(&header, width, height, pts);

        // Update reference frames based on header flags
        if header.refresh_last {
            self.last_frame = Some(frame.clone());
        }
        if header.refresh_golden_frame {
            self.golden_frame = Some(frame.clone());
        }
        if header.refresh_alternate_frame {
            self.altref_frame = Some(frame.clone());
        }

        // Output frame if show_frame is set
        if header.show_frame {
            self.output_queue.push(frame);
        }

        Ok(())
    }

    /// Creates an output video frame with proper plane allocation.
    #[allow(clippy::unused_self)]
    fn create_output_frame(
        &self,
        header: &FrameHeader,
        width: u32,
        height: u32,
        pts: i64,
    ) -> VideoFrame {
        // VP8 always outputs YUV420p
        let format = PixelFormat::Yuv420p;

        // Calculate plane sizes for YUV420p
        let y_size = (width * height) as usize;
        let uv_width = width.div_ceil(2) as usize;
        let uv_height = height.div_ceil(2) as usize;
        let uv_size = uv_width * uv_height;

        // Allocate planes (in a real decoder, these would be filled with decoded data)
        let y_plane = Plane::new(vec![128u8; y_size], width as usize);
        let u_plane = Plane::new(vec![128u8; uv_size], uv_width);
        let v_plane = Plane::new(vec![128u8; uv_size], uv_width);

        // Create timestamp with 1ms timebase
        let timestamp = Timestamp::new(pts, Rational::new(1, 1000));

        let frame_type = if header.is_keyframe() {
            FrameType::Key
        } else {
            FrameType::Inter
        };

        VideoFrame {
            format,
            width,
            height,
            planes: vec![y_plane, u_plane, v_plane],
            timestamp,
            frame_type,
            color_info: crate::frame::ColorInfo::default(),
            corrupt: false,
        }
    }
}

impl VideoDecoder for Vp8Decoder {
    fn codec(&self) -> CodecId {
        CodecId::Vp8
    }

    fn send_packet(&mut self, data: &[u8], pts: i64) -> CodecResult<()> {
        if self.flushing {
            return Err(CodecError::InvalidParameter(
                "VP8: Cannot send packet while flushing".to_string(),
            ));
        }

        self.decode_frame(data, pts)
    }

    fn receive_frame(&mut self) -> CodecResult<Option<VideoFrame>> {
        if self.output_queue.is_empty() {
            if self.flushing {
                return Err(CodecError::Eof);
            }
            return Ok(None);
        }

        Ok(Some(self.output_queue.remove(0)))
    }

    fn flush(&mut self) -> CodecResult<()> {
        self.flushing = true;
        Ok(())
    }

    fn reset(&mut self) {
        self.output_queue.clear();
        self.last_frame = None;
        self.golden_frame = None;
        self.altref_frame = None;
        self.flushing = false;
        // Note: we keep width/height to allow continued decoding after seek
    }

    fn output_format(&self) -> Option<PixelFormat> {
        if self.width.is_some() && self.height.is_some() {
            Some(PixelFormat::Yuv420p)
        } else {
            None
        }
    }

    fn dimensions(&self) -> Option<(u32, u32)> {
        match (self.width, self.height) {
            (Some(w), Some(h)) => Some((w, h)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vp8_decoder_new() {
        let config = DecoderConfig::default();
        let decoder = Vp8Decoder::new(config).expect("should succeed");
        assert_eq!(decoder.codec(), CodecId::Vp8);
        assert!(decoder.output_format().is_none());
        assert!(decoder.dimensions().is_none());
    }

    #[test]
    fn test_decode_keyframe() {
        let config = DecoderConfig::default();
        let mut decoder = Vp8Decoder::new(config).expect("should succeed");

        // Valid VP8 keyframe header
        let keyframe = [
            0x10, // frame_type=0, version=0, show=1
            0x00, 0x00, // first_partition_size
            0x9D, 0x01, 0x2A, // sync code
            0x40, 0x01, // width=320
            0xF0, 0x00, // height=240
        ];

        decoder.send_packet(&keyframe, 0).expect("should succeed");
        assert_eq!(decoder.dimensions(), Some((320, 240)));
        assert_eq!(decoder.output_format(), Some(PixelFormat::Yuv420p));

        let frame = decoder.receive_frame().expect("should succeed");
        assert!(frame.is_some());

        let frame = frame.expect("should succeed");
        assert!(frame.is_keyframe());
        assert_eq!(frame.width, 320);
        assert_eq!(frame.height, 240);
        assert_eq!(frame.format, PixelFormat::Yuv420p);
    }

    #[test]
    fn test_inter_frame_without_keyframe() {
        let config = DecoderConfig::default();
        let mut decoder = Vp8Decoder::new(config).expect("should succeed");

        // Inter frame (without prior keyframe)
        let inter = [
            0x01, // frame_type=1 (inter)
            0x00, 0x00,
        ];

        // Should fail because no keyframe was received
        assert!(decoder.send_packet(&inter, 0).is_err());
    }

    #[test]
    fn test_flush() {
        let config = DecoderConfig::default();
        let mut decoder = Vp8Decoder::new(config).expect("should succeed");

        // Send a keyframe
        let keyframe = [0x10, 0x00, 0x00, 0x9D, 0x01, 0x2A, 0x40, 0x01, 0xF0, 0x00];
        decoder.send_packet(&keyframe, 0).expect("should succeed");
        let _ = decoder.receive_frame();

        // Flush
        decoder.flush().expect("should succeed");

        // Should return EOF when no more frames
        assert!(matches!(decoder.receive_frame(), Err(CodecError::Eof)));

        // Cannot send more packets while flushing
        assert!(decoder.send_packet(&keyframe, 0).is_err());
    }

    #[test]
    fn test_reset() {
        let config = DecoderConfig::default();
        let mut decoder = Vp8Decoder::new(config).expect("should succeed");

        // Send a keyframe
        let keyframe = [0x10, 0x00, 0x00, 0x9D, 0x01, 0x2A, 0x40, 0x01, 0xF0, 0x00];
        decoder.send_packet(&keyframe, 0).expect("should succeed");

        // Flush and reset
        decoder.flush().expect("should succeed");
        decoder.reset();

        // Should be able to send packets again
        decoder.send_packet(&keyframe, 0).expect("should succeed");
        let frame = decoder.receive_frame().expect("should succeed");
        assert!(frame.is_some());
    }

    #[test]
    fn test_no_frame_available() {
        let config = DecoderConfig::default();
        let mut decoder = Vp8Decoder::new(config).expect("should succeed");

        // No packets sent, no frames available
        let result = decoder.receive_frame().expect("should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn test_hidden_frame() {
        let config = DecoderConfig::default();
        let mut decoder = Vp8Decoder::new(config).expect("should succeed");

        // Keyframe with show_frame=0 (hidden)
        let hidden_keyframe = [
            0x00, // frame_type=0, version=0, show=0
            0x00, 0x00, 0x9D, 0x01, 0x2A, 0x40, 0x01, 0xF0, 0x00,
        ];

        decoder
            .send_packet(&hidden_keyframe, 0)
            .expect("should succeed");

        // Dimensions should be updated
        assert_eq!(decoder.dimensions(), Some((320, 240)));

        // But no frame should be output (hidden)
        let frame = decoder.receive_frame().expect("should succeed");
        assert!(frame.is_none());
    }
}
