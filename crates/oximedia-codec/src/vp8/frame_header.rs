//! VP8 Frame header parsing.
//!
//! This module handles parsing of VP8 frame headers as specified in RFC 6386.
//! The frame header contains essential information about the frame including
//! its type, dimensions, and encoding parameters.

use crate::error::{CodecError, CodecResult};

/// VP8 frame types.
///
/// VP8 defines two types of frames:
/// - Keyframes (intra frames) that can be decoded independently
/// - Inter frames that reference previous frames
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FrameType {
    /// Keyframe (intra frame).
    ///
    /// A keyframe can be decoded without reference to any previous frames.
    /// It must be present at the start of a stream and after seeks.
    #[default]
    Key = 0,
    /// Inter frame (predicted frame).
    ///
    /// An inter frame uses motion compensation from reference frames.
    Inter = 1,
}

/// VP8 color space.
///
/// Specifies the color space used for the frame data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ColorSpace {
    /// YUV color space (default).
    #[default]
    Yuv = 0,
    /// Reserved for future use.
    Reserved = 1,
}

/// VP8 clamping type.
///
/// Specifies whether motion vector clamping is required.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ClampingType {
    /// Clamping is required for motion vectors.
    #[default]
    Required = 0,
    /// No clamping needed.
    None = 1,
}

/// VP8 Frame header.
///
/// Contains all the information parsed from a VP8 frame header,
/// including frame type, dimensions, and various encoding flags.
///
/// # Examples
///
/// ```
/// use oximedia_codec::vp8::{FrameHeader, FrameType};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Parse a VP8 keyframe header
/// let data = [
///     0x00,       // frame_type=0, version=0, show=0
///     0x00, 0x00, // first_partition_size
///     0x9D, 0x01, 0x2A, // sync code
///     0x40, 0x01, // width=320
///     0xF0, 0x00, // height=240
/// ];
///
/// let header = FrameHeader::parse(&data)?;
/// assert!(header.is_keyframe());
/// assert_eq!(header.width, 320);
/// assert_eq!(header.height, 240);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Default)]
#[allow(clippy::struct_excessive_bools)]
pub struct FrameHeader {
    /// Frame type (key or inter).
    pub frame_type: FrameType,
    /// Version number (0-3).
    pub version: u8,
    /// Whether this frame should be displayed.
    pub show_frame: bool,
    /// Size of the first data partition in bytes.
    pub first_partition_size: u32,
    /// Frame width in pixels.
    pub width: u16,
    /// Horizontal scale factor (0-3).
    pub horizontal_scale: u8,
    /// Frame height in pixels.
    pub height: u16,
    /// Vertical scale factor (0-3).
    pub vertical_scale: u8,
    /// Color space used.
    pub color_space: ColorSpace,
    /// Clamping type for motion vectors.
    pub clamping_type: ClampingType,
    /// Whether segmentation is enabled.
    pub segmentation_enabled: bool,
    /// Filter type (0 = normal, 1 = simple).
    pub filter_type: u8,
    /// Loop filter strength level (0-63).
    pub loop_filter_level: u8,
    /// Sharpness level (0-7).
    pub sharpness_level: u8,
    /// Whether mode/reference loop filter deltas are enabled.
    pub mode_ref_lf_delta_enabled: bool,
    /// Log2 of the number of DCT token partitions (0-3).
    pub log2_nbr_of_dct_partitions: u8,
    /// Base quantizer index (0-127).
    pub quant_index: u8,
    /// Whether to refresh the golden reference frame.
    pub refresh_golden_frame: bool,
    /// Whether to refresh the alternate reference frame.
    pub refresh_alternate_frame: bool,
    /// Copy buffer to golden frame (0-3).
    pub copy_buffer_to_golden: u8,
    /// Copy buffer to alternate frame (0-3).
    pub copy_buffer_to_alternate: u8,
    /// Sign bias for golden reference frame.
    pub sign_bias_golden: bool,
    /// Sign bias for alternate reference frame.
    pub sign_bias_alternate: bool,
    /// Whether to refresh entropy probabilities.
    pub refresh_entropy_probs: bool,
    /// Whether to refresh the last reference frame.
    pub refresh_last: bool,
}

impl FrameHeader {
    /// VP8 keyframe sync code.
    ///
    /// Every VP8 keyframe must contain this 3-byte sync code
    /// immediately following the frame tag.
    const SYNC_CODE: [u8; 3] = [0x9D, 0x01, 0x2A];

    /// Parses a VP8 frame header from the given data.
    ///
    /// # Arguments
    ///
    /// * `data` - The raw frame data starting at the frame header
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The data is too short
    /// - The sync code is invalid (for keyframes)
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_codec::vp8::FrameHeader;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let data = [
    ///     0x00, 0x00, 0x00,       // frame tag
    ///     0x9D, 0x01, 0x2A,       // sync code
    ///     0x40, 0x01, 0xF0, 0x00, // dimensions
    /// ];
    /// let header = FrameHeader::parse(&data)?;
    /// assert!(header.is_keyframe());
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::cast_possible_truncation)]
    pub fn parse(data: &[u8]) -> CodecResult<Self> {
        if data.len() < 3 {
            return Err(CodecError::InvalidBitstream(
                "VP8 frame data too short for header".to_string(),
            ));
        }

        let mut header = Self::default();

        // Parse 3-byte frame tag
        // Byte 0:
        //   Bit 0: frame_type (0 = key, 1 = inter)
        //   Bits 1-2: version
        //   Bit 3: show_frame
        //   Bits 4-7: first_partition_size (bits 0-3)
        // Bytes 1-2: first_partition_size (bits 4-18)

        let b0 = data[0];
        let b1 = data[1];
        let b2 = data[2];

        header.frame_type = if b0 & 0x01 == 0 {
            FrameType::Key
        } else {
            FrameType::Inter
        };

        header.version = (b0 >> 1) & 0x07;
        header.show_frame = (b0 >> 4) & 0x01 != 0;

        header.first_partition_size =
            (u32::from(b0 >> 5) & 0x07) | (u32::from(b1) << 3) | (u32::from(b2) << 11);

        let mut offset = 3;

        // Keyframe has additional header data
        if header.frame_type == FrameType::Key {
            if data.len() < offset + 7 {
                return Err(CodecError::InvalidBitstream(
                    "VP8 keyframe header too short for dimensions".to_string(),
                ));
            }

            // Check sync code
            if data[offset..offset + 3] != Self::SYNC_CODE {
                return Err(CodecError::InvalidBitstream(
                    "Invalid VP8 keyframe sync code".to_string(),
                ));
            }
            offset += 3;

            // Parse frame dimensions
            // 2 bytes: width | (horizontal_scale << 14)
            // 2 bytes: height | (vertical_scale << 14)
            let w0 = u16::from(data[offset]);
            let w1 = u16::from(data[offset + 1]);
            header.width = (w0 | (w1 << 8)) & 0x3FFF;
            header.horizontal_scale = (w1 >> 6) as u8;

            let h0 = u16::from(data[offset + 2]);
            let h1 = u16::from(data[offset + 3]);
            header.height = (h0 | (h1 << 8)) & 0x3FFF;
            header.vertical_scale = (h1 >> 6) as u8;

            // For keyframes, reference frame refresh flags are implied
            header.refresh_golden_frame = true;
            header.refresh_alternate_frame = true;
            header.refresh_last = true;
        }

        Ok(header)
    }

    /// Returns whether this is a keyframe.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_codec::vp8::{FrameHeader, FrameType};
    ///
    /// let mut header = FrameHeader::default();
    /// assert!(header.is_keyframe()); // Default is Key
    ///
    /// header.frame_type = FrameType::Inter;
    /// assert!(!header.is_keyframe());
    /// ```
    #[must_use]
    pub const fn is_keyframe(&self) -> bool {
        matches!(self.frame_type, FrameType::Key)
    }

    /// Returns the scaled width based on the horizontal scale factor.
    ///
    /// The scale factors are:
    /// - 0: 1x (no scaling)
    /// - 1: 5/4x
    /// - 2: 5/3x
    /// - 3: 2x
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_codec::vp8::FrameHeader;
    ///
    /// let mut header = FrameHeader::default();
    /// header.width = 320;
    /// header.horizontal_scale = 0;
    /// assert_eq!(header.scaled_width(), 320);
    ///
    /// header.horizontal_scale = 3; // 2x scale
    /// assert_eq!(header.scaled_width(), 640);
    /// ```
    #[must_use]
    #[allow(clippy::match_same_arms)]
    pub const fn scaled_width(&self) -> u32 {
        let (scale, div) = match self.horizontal_scale {
            0 => (1, 1), // 1x (no scaling)
            1 => (5, 4), // 5/4x
            2 => (5, 3), // 5/3x
            3 => (2, 1), // 2x
            _ => (1, 1), // Invalid, treat as 1x
        };
        (self.width as u32 * scale) / div
    }

    /// Returns the scaled height based on the vertical scale factor.
    ///
    /// The scale factors are the same as for width.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_codec::vp8::FrameHeader;
    ///
    /// let mut header = FrameHeader::default();
    /// header.height = 240;
    /// header.vertical_scale = 0;
    /// assert_eq!(header.scaled_height(), 240);
    ///
    /// header.vertical_scale = 3; // 2x scale
    /// assert_eq!(header.scaled_height(), 480);
    /// ```
    #[must_use]
    #[allow(clippy::match_same_arms)]
    pub const fn scaled_height(&self) -> u32 {
        let (scale, div) = match self.vertical_scale {
            0 => (1, 1), // 1x (no scaling)
            1 => (5, 4), // 5/4x
            2 => (5, 3), // 5/3x
            3 => (2, 1), // 2x
            _ => (1, 1), // Invalid, treat as 1x
        };
        (self.height as u32 * scale) / div
    }

    /// Returns the number of macroblocks in width.
    ///
    /// Macroblocks are 16x16 pixel blocks.
    #[must_use]
    #[allow(clippy::manual_div_ceil)]
    pub const fn mb_width(&self) -> u32 {
        // Use manual div_ceil because div_ceil is not const stable
        (self.width as u32 + 15) / 16
    }

    /// Returns the number of macroblocks in height.
    ///
    /// Macroblocks are 16x16 pixel blocks.
    #[must_use]
    #[allow(clippy::manual_div_ceil)]
    pub const fn mb_height(&self) -> u32 {
        // Use manual div_ceil because div_ceil is not const stable
        (self.height as u32 + 15) / 16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyframe_header() {
        // Keyframe with 320x240
        let data = [
            0x00, // frame_type=0, version=0, show=0
            0x00, 0x00, // first_partition_size
            0x9D, 0x01, 0x2A, // sync code
            0x40, 0x01, // width=320
            0xF0, 0x00, // height=240
        ];

        let header = FrameHeader::parse(&data).expect("should succeed");
        assert!(header.is_keyframe());
        assert_eq!(header.width, 320);
        assert_eq!(header.height, 240);
        assert!(header.refresh_golden_frame);
        assert!(header.refresh_alternate_frame);
        assert!(header.refresh_last);
    }

    #[test]
    fn test_invalid_sync() {
        let data = [
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Wrong sync code
            0x00, 0x00, 0x00, 0x00,
        ];

        assert!(FrameHeader::parse(&data).is_err());
    }

    #[test]
    fn test_inter_frame() {
        // Inter frame (no sync code or dimensions)
        let data = [
            0x01, // frame_type=1 (inter), version=0, show=0
            0x00, 0x00, // first_partition_size
        ];

        let header = FrameHeader::parse(&data).expect("should succeed");
        assert!(!header.is_keyframe());
        assert_eq!(header.frame_type, FrameType::Inter);
    }

    #[test]
    fn test_scaled_dimensions() {
        let mut header = FrameHeader::default();
        header.width = 320;
        header.height = 240;

        // No scale
        header.horizontal_scale = 0;
        header.vertical_scale = 0;
        assert_eq!(header.scaled_width(), 320);
        assert_eq!(header.scaled_height(), 240);

        // 2x scale
        header.horizontal_scale = 3;
        header.vertical_scale = 3;
        assert_eq!(header.scaled_width(), 640);
        assert_eq!(header.scaled_height(), 480);
    }

    #[test]
    fn test_macroblock_dimensions() {
        let mut header = FrameHeader::default();
        header.width = 320;
        header.height = 240;

        assert_eq!(header.mb_width(), 20);
        assert_eq!(header.mb_height(), 15);

        // Non-multiple of 16
        header.width = 321;
        header.height = 241;
        assert_eq!(header.mb_width(), 21);
        assert_eq!(header.mb_height(), 16);
    }

    #[test]
    fn test_too_short_data() {
        let data: [u8; 2] = [0x00, 0x00];
        assert!(FrameHeader::parse(&data).is_err());
    }

    #[test]
    fn test_keyframe_too_short() {
        // Keyframe but missing dimensions
        let data = [
            0x00, 0x00, 0x00, 0x9D, 0x01, 0x2A, // sync code but no dimensions
        ];
        assert!(FrameHeader::parse(&data).is_err());
    }

    #[test]
    fn test_frame_tag_parsing() {
        // show_frame=1, version=2
        let data = [
            0x14, // frame_type=0, version=2, show=1
            0x10, 0x20, // first_partition_size
            0x9D, 0x01, 0x2A, 0x00, 0x01, 0x00, 0x01,
        ];

        let header = FrameHeader::parse(&data).expect("should succeed");
        assert!(header.is_keyframe());
        assert!(header.show_frame);
        assert_eq!(header.version, 2);
    }
}
