//! GIF (Graphics Interchange Format) codec implementation.
//!
//! Supports GIF89a specification:
//! - Decoding and encoding
//! - Animation with multiple frames
//! - Transparency support
//! - Color quantization (median cut, octree)
//! - Dithering (Floyd-Steinberg, ordered)
//! - Loop count control
//!
//! # Examples
//!
//! ## Decoding
//!
//! ```ignore
//! use oximedia_codec::gif::GifDecoder;
//!
//! let data = std::fs::read("animation.gif")?;
//! let decoder = GifDecoder::new(&data)?;
//!
//! println!("Frames: {}", decoder.frame_count());
//! println!("Size: {}x{}", decoder.width(), decoder.height());
//!
//! for i in 0..decoder.frame_count() {
//!     let frame = decoder.decode_frame(i)?;
//!     // Process frame...
//! }
//! ```
//!
//! ## Encoding
//!
//! ```ignore
//! use oximedia_codec::gif::{GifEncoder, GifEncoderConfig, GifFrameConfig};
//!
//! let config = GifEncoderConfig {
//!     colors: 256,
//!     loop_count: 0, // Infinite loop
//!     ..Default::default()
//! };
//!
//! let mut encoder = GifEncoder::new(640, 480, config)?;
//!
//! let frame_config = GifFrameConfig {
//!     delay_time: 10, // 100ms
//!     ..Default::default()
//! };
//!
//! let data = encoder.encode(&frames, &[frame_config; frames.len()])?;
//! std::fs::write("output.gif", &data)?;
//! ```

mod decoder;
mod encoder;
mod lzw;
pub mod quality;

use crate::error::{CodecError, CodecResult};
use crate::frame::VideoFrame;

pub use decoder::{GifFrame, GraphicsControlExtension, ImageDescriptor, LogicalScreenDescriptor};
pub use encoder::{DitheringMethod, GifEncoderConfig, GifFrameConfig, QuantizationMethod};

use decoder::GifDecoderState;
use encoder::GifEncoderState;

/// GIF decoder for reading GIF89a files.
///
/// Supports:
/// - Single images and animations
/// - Transparency
/// - Interlaced images
/// - Global and local color tables
pub struct GifDecoder {
    state: GifDecoderState,
}

impl GifDecoder {
    /// Create a new GIF decoder from data.
    ///
    /// # Arguments
    ///
    /// * `data` - GIF file data
    ///
    /// # Errors
    ///
    /// Returns error if data is not a valid GIF file.
    pub fn new(data: &[u8]) -> CodecResult<Self> {
        let state = GifDecoderState::decode(data)?;
        Ok(Self { state })
    }

    /// Get canvas width.
    #[must_use]
    pub fn width(&self) -> u32 {
        u32::from(self.state.screen_descriptor.width)
    }

    /// Get canvas height.
    #[must_use]
    pub fn height(&self) -> u32 {
        u32::from(self.state.screen_descriptor.height)
    }

    /// Get number of frames.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.state.frames.len()
    }

    /// Get loop count (0 = infinite).
    #[must_use]
    pub fn loop_count(&self) -> u16 {
        self.state.loop_count
    }

    /// Get the logical screen descriptor.
    #[must_use]
    pub fn screen_descriptor(&self) -> &LogicalScreenDescriptor {
        &self.state.screen_descriptor
    }

    /// Get the global color table.
    #[must_use]
    pub fn global_color_table(&self) -> &[u8] {
        &self.state.global_color_table
    }

    /// Get frame information.
    ///
    /// # Arguments
    ///
    /// * `index` - Frame index
    ///
    /// # Errors
    ///
    /// Returns error if index is out of range.
    pub fn frame_info(&self, index: usize) -> CodecResult<&GifFrame> {
        self.state.frames.get(index).ok_or_else(|| {
            CodecError::InvalidParameter(format!("Frame index {} out of range", index))
        })
    }

    /// Decode a specific frame.
    ///
    /// # Arguments
    ///
    /// * `index` - Frame index to decode
    ///
    /// # Errors
    ///
    /// Returns error if decoding fails or index is out of range.
    pub fn decode_frame(&self, index: usize) -> CodecResult<VideoFrame> {
        self.state.frame_to_video_frame(index)
    }

    /// Decode all frames.
    ///
    /// # Errors
    ///
    /// Returns error if decoding fails.
    pub fn decode_all_frames(&self) -> CodecResult<Vec<VideoFrame>> {
        let mut frames = Vec::with_capacity(self.frame_count());
        for i in 0..self.frame_count() {
            frames.push(self.decode_frame(i)?);
        }
        Ok(frames)
    }

    /// Get frame delay time in milliseconds.
    ///
    /// # Arguments
    ///
    /// * `index` - Frame index
    ///
    /// # Errors
    ///
    /// Returns error if index is out of range.
    pub fn frame_delay_ms(&self, index: usize) -> CodecResult<u32> {
        let frame = self.frame_info(index)?;
        if let Some(ref control) = frame.control {
            // Delay time is in hundredths of a second
            Ok(u32::from(control.delay_time) * 10)
        } else {
            Ok(0)
        }
    }

    /// Check if frame has transparency.
    ///
    /// # Arguments
    ///
    /// * `index` - Frame index
    ///
    /// # Errors
    ///
    /// Returns error if index is out of range.
    pub fn frame_has_transparency(&self, index: usize) -> CodecResult<bool> {
        let frame = self.frame_info(index)?;
        Ok(frame.control.as_ref().map_or(false, |c| c.has_transparency))
    }

    /// Get frame disposal method.
    ///
    /// # Arguments
    ///
    /// * `index` - Frame index
    ///
    /// # Errors
    ///
    /// Returns error if index is out of range.
    pub fn frame_disposal_method(&self, index: usize) -> CodecResult<DisposalMethod> {
        let frame = self.frame_info(index)?;
        if let Some(ref control) = frame.control {
            Ok(DisposalMethod::from_value(control.disposal_method))
        } else {
            Ok(DisposalMethod::None)
        }
    }

    /// Check if this is an animated GIF.
    #[must_use]
    pub fn is_animated(&self) -> bool {
        self.frame_count() > 1
    }

    /// Get total animation duration in milliseconds.
    #[must_use]
    pub fn total_duration_ms(&self) -> u32 {
        let mut total = 0;
        for i in 0..self.frame_count() {
            if let Ok(delay) = self.frame_delay_ms(i) {
                total += delay;
            }
        }
        total
    }
}

/// GIF encoder for creating GIF89a files.
///
/// Supports:
/// - Single images and animations
/// - Color quantization with configurable palette size
/// - Dithering options
/// - Transparency
/// - Loop count control
pub struct GifEncoder {
    state: GifEncoderState,
}

impl GifEncoder {
    /// Create a new GIF encoder.
    ///
    /// # Arguments
    ///
    /// * `width` - Canvas width
    /// * `height` - Canvas height
    /// * `config` - Encoder configuration
    ///
    /// # Errors
    ///
    /// Returns error if parameters are invalid.
    pub fn new(width: u32, height: u32, config: GifEncoderConfig) -> CodecResult<Self> {
        let state = GifEncoderState::new(width, height, config)?;
        Ok(Self { state })
    }

    /// Encode frames to GIF data.
    ///
    /// # Arguments
    ///
    /// * `frames` - Frames to encode
    /// * `frame_configs` - Configuration for each frame
    ///
    /// # Errors
    ///
    /// Returns error if encoding fails.
    pub fn encode(
        &mut self,
        frames: &[VideoFrame],
        frame_configs: &[GifFrameConfig],
    ) -> CodecResult<Vec<u8>> {
        self.state.encode(frames, frame_configs)
    }

    /// Encode a single frame (non-animated GIF).
    ///
    /// # Arguments
    ///
    /// * `frame` - Frame to encode
    ///
    /// # Errors
    ///
    /// Returns error if encoding fails.
    pub fn encode_single(&mut self, frame: &VideoFrame) -> CodecResult<Vec<u8>> {
        let config = GifFrameConfig::default();
        self.encode(&[frame.clone()], &[config])
    }

    /// Encode multiple frames with the same delay time.
    ///
    /// # Arguments
    ///
    /// * `frames` - Frames to encode
    /// * `delay_ms` - Delay between frames in milliseconds
    ///
    /// # Errors
    ///
    /// Returns error if encoding fails.
    pub fn encode_animation(
        &mut self,
        frames: &[VideoFrame],
        delay_ms: u32,
    ) -> CodecResult<Vec<u8>> {
        let delay_time = (delay_ms / 10) as u16; // Convert to hundredths of a second
        let config = GifFrameConfig {
            delay_time,
            ..Default::default()
        };
        let configs = vec![config; frames.len()];
        self.encode(frames, &configs)
    }
}

/// Frame disposal method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisposalMethod {
    /// No disposal specified.
    None,
    /// Do not dispose (keep frame).
    Keep,
    /// Restore to background color.
    Background,
    /// Restore to previous frame.
    Previous,
}

impl DisposalMethod {
    /// Convert from GIF disposal method value.
    #[must_use]
    pub fn from_value(value: u8) -> Self {
        match value {
            0 => Self::None,
            1 => Self::Keep,
            2 => Self::Background,
            3 => Self::Previous,
            _ => Self::None,
        }
    }

    /// Convert to GIF disposal method value.
    #[must_use]
    pub fn to_value(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Keep => 1,
            Self::Background => 2,
            Self::Previous => 3,
        }
    }
}

/// Detect if data is a GIF file.
///
/// # Arguments
///
/// * `data` - Data to check
///
/// # Examples
///
/// ```ignore
/// let data = std::fs::read("image.gif")?;
/// if is_gif(&data) {
///     let decoder = GifDecoder::new(&data)?;
///     // ...
/// }
/// ```
#[must_use]
pub fn is_gif(data: &[u8]) -> bool {
    data.len() >= 6 && data.starts_with(b"GIF") && (&data[3..6] == b"89a" || &data[3..6] == b"87a")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{Plane, VideoFrame};
    use bytes::Bytes;
    use oximedia_core::PixelFormat;

    fn create_test_frame(width: u32, height: u32, color: [u8; 3]) -> VideoFrame {
        let size = (width * height * 4) as usize;
        let mut data = Vec::with_capacity(size);

        for _ in 0..(width * height) {
            data.extend_from_slice(&[color[0], color[1], color[2], 255]);
        }

        let plane = Plane {
            data,
            stride: (width * 4) as usize,
            width,
            height,
        };

        let mut frame = VideoFrame::new(PixelFormat::Rgba32, width, height);
        frame.planes = vec![plane];
        frame
    }

    #[test]
    fn test_is_gif() {
        assert!(is_gif(b"GIF89a\x00\x00"));
        assert!(is_gif(b"GIF87a\x00\x00"));
        assert!(!is_gif(b"PNG\x00\x00\x00"));
        assert!(!is_gif(b"GIF"));
    }

    #[test]
    fn test_disposal_method() {
        assert_eq!(DisposalMethod::from_value(0), DisposalMethod::None);
        assert_eq!(DisposalMethod::from_value(1), DisposalMethod::Keep);
        assert_eq!(DisposalMethod::from_value(2), DisposalMethod::Background);
        assert_eq!(DisposalMethod::from_value(3), DisposalMethod::Previous);

        assert_eq!(DisposalMethod::None.to_value(), 0);
        assert_eq!(DisposalMethod::Keep.to_value(), 1);
        assert_eq!(DisposalMethod::Background.to_value(), 2);
        assert_eq!(DisposalMethod::Previous.to_value(), 3);
    }

    #[test]
    fn test_encoder_single_frame() {
        let config = GifEncoderConfig::default();
        let mut encoder = GifEncoder::new(16, 16, config).expect("should succeed");

        let frame = create_test_frame(16, 16, [255, 0, 0]);
        let data = encoder.encode_single(&frame).expect("should succeed");

        assert!(is_gif(&data));
        assert!(data.len() > 100); // Should have header + data
    }

    #[test]
    fn test_encoder_animation() {
        let config = GifEncoderConfig {
            colors: 16,
            loop_count: 0,
            ..Default::default()
        };
        let mut encoder = GifEncoder::new(16, 16, config).expect("should succeed");

        let frames = vec![
            create_test_frame(16, 16, [255, 0, 0]),
            create_test_frame(16, 16, [0, 255, 0]),
            create_test_frame(16, 16, [0, 0, 255]),
        ];

        let data = encoder
            .encode_animation(&frames, 100)
            .expect("should succeed");

        assert!(is_gif(&data));

        // Decode and verify
        let decoder = GifDecoder::new(&data).expect("should succeed");
        assert_eq!(decoder.frame_count(), 3);
        assert_eq!(decoder.width(), 16);
        assert_eq!(decoder.height(), 16);
        assert!(decoder.is_animated());
    }

    #[test]
    fn test_roundtrip() {
        let config = GifEncoderConfig {
            colors: 256,
            ..Default::default()
        };
        let mut encoder = GifEncoder::new(8, 8, config).expect("should succeed");

        let original_frame = create_test_frame(8, 8, [128, 64, 192]);
        let data = encoder
            .encode_single(&original_frame)
            .expect("should succeed");

        let decoder = GifDecoder::new(&data).expect("should succeed");
        assert_eq!(decoder.frame_count(), 1);
        assert_eq!(decoder.width(), 8);
        assert_eq!(decoder.height(), 8);

        let decoded_frame = decoder.decode_frame(0).expect("should succeed");
        assert_eq!(decoded_frame.format, PixelFormat::Rgba32);
        assert_eq!(decoded_frame.width, 8);
        assert_eq!(decoded_frame.height, 8);
    }
}
