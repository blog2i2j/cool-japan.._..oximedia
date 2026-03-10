//! GIF decoder implementation.
//!
//! Supports GIF89a format with:
//! - Interlaced and non-interlaced images
//! - Local and global color tables
//! - Transparency
//! - Animation with multiple frames
//! - Disposal methods

use super::lzw::LzwDecoder;
use crate::error::{CodecError, CodecResult};
use crate::frame::{Plane, VideoFrame};
use bytes::Bytes;
use oximedia_core::PixelFormat;
use std::io::{Cursor, Read};

/// GIF signature.
const GIF_SIGNATURE: &[u8] = b"GIF";

/// GIF89a version.
const GIF89A_VERSION: &[u8] = b"89a";

/// GIF87a version.
const GIF87A_VERSION: &[u8] = b"87a";

/// Extension introducer.
const EXTENSION_INTRODUCER: u8 = 0x21;

/// Image separator.
const IMAGE_SEPARATOR: u8 = 0x2C;

/// Trailer (end of GIF).
const TRAILER: u8 = 0x3B;

/// Block terminator.
#[allow(dead_code)]
const BLOCK_TERMINATOR: u8 = 0x00;

/// Graphics Control Extension label.
const GRAPHICS_CONTROL_LABEL: u8 = 0xF9;

/// Comment Extension label.
const COMMENT_LABEL: u8 = 0xFE;

/// Plain Text Extension label.
const PLAIN_TEXT_LABEL: u8 = 0x01;

/// Application Extension label.
const APPLICATION_LABEL: u8 = 0xFF;

/// Disposal method: No disposal specified.
#[allow(dead_code)]
const DISPOSAL_NONE: u8 = 0;

/// Disposal method: Do not dispose (keep frame).
#[allow(dead_code)]
const DISPOSAL_KEEP: u8 = 1;

/// Disposal method: Restore to background color.
#[allow(dead_code)]
const DISPOSAL_BACKGROUND: u8 = 2;

/// Disposal method: Restore to previous frame.
#[allow(dead_code)]
const DISPOSAL_PREVIOUS: u8 = 3;

/// Logical Screen Descriptor.
#[derive(Debug, Clone)]
pub struct LogicalScreenDescriptor {
    /// Canvas width in pixels.
    pub width: u16,
    /// Canvas height in pixels.
    pub height: u16,
    /// Global color table flag.
    pub has_global_color_table: bool,
    /// Color resolution (bits per color minus 1).
    pub color_resolution: u8,
    /// Sort flag.
    pub sort_flag: bool,
    /// Size of global color table (log2(size) - 1).
    pub global_color_table_size: u8,
    /// Background color index.
    pub background_color_index: u8,
    /// Pixel aspect ratio.
    pub pixel_aspect_ratio: u8,
}

/// Graphics Control Extension data.
#[derive(Debug, Clone, Default)]
pub struct GraphicsControlExtension {
    /// Disposal method.
    pub disposal_method: u8,
    /// User input flag.
    pub user_input_flag: bool,
    /// Transparency flag.
    pub has_transparency: bool,
    /// Delay time in hundredths of a second.
    pub delay_time: u16,
    /// Transparent color index.
    pub transparent_color_index: u8,
}

/// Image descriptor.
#[derive(Debug, Clone)]
pub struct ImageDescriptor {
    /// Left position on canvas.
    pub left: u16,
    /// Top position on canvas.
    pub top: u16,
    /// Image width.
    pub width: u16,
    /// Image height.
    pub height: u16,
    /// Local color table flag.
    pub has_local_color_table: bool,
    /// Interlace flag.
    pub interlaced: bool,
    /// Sort flag.
    pub sort_flag: bool,
    /// Size of local color table (log2(size) - 1).
    pub local_color_table_size: u8,
}

/// A single GIF frame.
#[derive(Debug, Clone)]
pub struct GifFrame {
    /// Image descriptor.
    pub descriptor: ImageDescriptor,
    /// Graphics control extension (if present).
    pub control: Option<GraphicsControlExtension>,
    /// Color table (local or global).
    pub color_table: Vec<u8>,
    /// Decompressed pixel indices.
    pub indices: Vec<u8>,
}

/// GIF decoder state.
pub struct GifDecoderState {
    /// Logical screen descriptor.
    pub screen_descriptor: LogicalScreenDescriptor,
    /// Global color table.
    pub global_color_table: Vec<u8>,
    /// Decoded frames.
    pub frames: Vec<GifFrame>,
    /// Loop count (0 = infinite).
    pub loop_count: u16,
    /// Background color (RGB).
    pub background_color: [u8; 3],
}

impl GifDecoderState {
    /// Decode a GIF file.
    ///
    /// # Errors
    ///
    /// Returns error if decoding fails or data is invalid.
    #[allow(clippy::too_many_lines)]
    pub fn decode(data: &[u8]) -> CodecResult<Self> {
        let mut cursor = Cursor::new(data);

        // Read and verify signature
        let mut signature = [0u8; 3];
        cursor
            .read_exact(&mut signature)
            .map_err(|_| CodecError::InvalidData("Failed to read GIF signature".into()))?;

        if &signature != GIF_SIGNATURE {
            return Err(CodecError::InvalidData("Invalid GIF signature".into()));
        }

        // Read and verify version
        let mut version = [0u8; 3];
        cursor
            .read_exact(&mut version)
            .map_err(|_| CodecError::InvalidData("Failed to read GIF version".into()))?;

        if &version != GIF89A_VERSION && &version != GIF87A_VERSION {
            return Err(CodecError::InvalidData(format!(
                "Unsupported GIF version: {:?}",
                version
            )));
        }

        // Read Logical Screen Descriptor
        let screen_descriptor = Self::read_screen_descriptor(&mut cursor)?;

        // Read Global Color Table if present
        let global_color_table = if screen_descriptor.has_global_color_table {
            let size = Self::color_table_size(screen_descriptor.global_color_table_size);
            Self::read_color_table(&mut cursor, size)?
        } else {
            Vec::new()
        };

        let mut state = Self {
            screen_descriptor: screen_descriptor.clone(),
            global_color_table: global_color_table.clone(),
            frames: Vec::new(),
            loop_count: 1,
            background_color: Self::get_background_color(&screen_descriptor, &global_color_table),
        };

        // Parse data stream
        let mut current_gce: Option<GraphicsControlExtension> = None;

        loop {
            let mut block_type = [0u8; 1];
            if cursor.read_exact(&mut block_type).is_err() {
                break;
            }

            match block_type[0] {
                IMAGE_SEPARATOR => {
                    let frame = Self::read_image(
                        &mut cursor,
                        &screen_descriptor,
                        &global_color_table,
                        current_gce.take(),
                    )?;
                    state.frames.push(frame);
                }
                EXTENSION_INTRODUCER => {
                    let ext = Self::read_extension(&mut cursor)?;
                    match ext {
                        Extension::GraphicsControl(gce) => {
                            current_gce = Some(gce);
                        }
                        Extension::Application(app) => {
                            if let Some(loop_count) = Self::parse_netscape_extension(&app) {
                                state.loop_count = loop_count;
                            }
                        }
                        Extension::Comment(_) | Extension::PlainText(_) => {
                            // Ignore comments and plain text
                        }
                    }
                }
                TRAILER => break,
                _ => {
                    return Err(CodecError::InvalidData(format!(
                        "Unknown block type: 0x{:02X}",
                        block_type[0]
                    )))
                }
            }
        }

        if state.frames.is_empty() {
            return Err(CodecError::InvalidData("No frames found in GIF".into()));
        }

        Ok(state)
    }

    /// Read Logical Screen Descriptor.
    fn read_screen_descriptor(cursor: &mut Cursor<&[u8]>) -> CodecResult<LogicalScreenDescriptor> {
        let mut buf = [0u8; 7];
        cursor
            .read_exact(&mut buf)
            .map_err(|_| CodecError::InvalidData("Failed to read screen descriptor".into()))?;

        let width = u16::from_le_bytes([buf[0], buf[1]]);
        let height = u16::from_le_bytes([buf[2], buf[3]]);
        let packed = buf[4];
        let background_color_index = buf[5];
        let pixel_aspect_ratio = buf[6];

        let has_global_color_table = (packed & 0x80) != 0;
        let color_resolution = ((packed & 0x70) >> 4) + 1;
        let sort_flag = (packed & 0x08) != 0;
        let global_color_table_size = packed & 0x07;

        Ok(LogicalScreenDescriptor {
            width,
            height,
            has_global_color_table,
            color_resolution,
            sort_flag,
            global_color_table_size,
            background_color_index,
            pixel_aspect_ratio,
        })
    }

    /// Read color table.
    fn read_color_table(cursor: &mut Cursor<&[u8]>, size: usize) -> CodecResult<Vec<u8>> {
        let mut table = vec![0u8; size * 3];
        cursor
            .read_exact(&mut table)
            .map_err(|_| CodecError::InvalidData("Failed to read color table".into()))?;
        Ok(table)
    }

    /// Calculate color table size from size field.
    fn color_table_size(size_field: u8) -> usize {
        1 << (size_field + 1)
    }

    /// Get background color from screen descriptor and global color table.
    fn get_background_color(
        descriptor: &LogicalScreenDescriptor,
        global_color_table: &[u8],
    ) -> [u8; 3] {
        if descriptor.has_global_color_table {
            let idx = descriptor.background_color_index as usize * 3;
            if idx + 2 < global_color_table.len() {
                return [
                    global_color_table[idx],
                    global_color_table[idx + 1],
                    global_color_table[idx + 2],
                ];
            }
        }
        [0, 0, 0]
    }

    /// Read extension block.
    fn read_extension(cursor: &mut Cursor<&[u8]>) -> CodecResult<Extension> {
        let mut label = [0u8; 1];
        cursor
            .read_exact(&mut label)
            .map_err(|_| CodecError::InvalidData("Failed to read extension label".into()))?;

        match label[0] {
            GRAPHICS_CONTROL_LABEL => {
                let gce = Self::read_graphics_control_extension(cursor)?;
                Ok(Extension::GraphicsControl(gce))
            }
            APPLICATION_LABEL => {
                let data = Self::read_data_sub_blocks(cursor)?;
                Ok(Extension::Application(data))
            }
            COMMENT_LABEL => {
                let data = Self::read_data_sub_blocks(cursor)?;
                Ok(Extension::Comment(data))
            }
            PLAIN_TEXT_LABEL => {
                let data = Self::read_data_sub_blocks(cursor)?;
                Ok(Extension::PlainText(data))
            }
            _ => {
                // Skip unknown extension
                Self::read_data_sub_blocks(cursor)?;
                Ok(Extension::Comment(Vec::new()))
            }
        }
    }

    /// Read Graphics Control Extension.
    fn read_graphics_control_extension(
        cursor: &mut Cursor<&[u8]>,
    ) -> CodecResult<GraphicsControlExtension> {
        let mut buf = [0u8; 6];
        cursor
            .read_exact(&mut buf)
            .map_err(|_| CodecError::InvalidData("Failed to read GCE".into()))?;

        let _block_size = buf[0];
        let packed = buf[1];
        let delay_time = u16::from_le_bytes([buf[2], buf[3]]);
        let transparent_color_index = buf[4];
        let _terminator = buf[5];

        let disposal_method = (packed & 0x1C) >> 2;
        let user_input_flag = (packed & 0x02) != 0;
        let has_transparency = (packed & 0x01) != 0;

        Ok(GraphicsControlExtension {
            disposal_method,
            user_input_flag,
            has_transparency,
            delay_time,
            transparent_color_index,
        })
    }

    /// Read data sub-blocks.
    fn read_data_sub_blocks(cursor: &mut Cursor<&[u8]>) -> CodecResult<Vec<u8>> {
        let mut data = Vec::new();
        loop {
            let mut block_size = [0u8; 1];
            cursor
                .read_exact(&mut block_size)
                .map_err(|_| CodecError::InvalidData("Failed to read block size".into()))?;

            if block_size[0] == 0 {
                break;
            }

            let size = block_size[0] as usize;
            let start_len = data.len();
            data.resize(start_len + size, 0);
            cursor
                .read_exact(&mut data[start_len..])
                .map_err(|_| CodecError::InvalidData("Failed to read data block".into()))?;
        }
        Ok(data)
    }

    /// Parse Netscape extension (NETSCAPE2.0) for loop count.
    fn parse_netscape_extension(data: &[u8]) -> Option<u16> {
        if data.len() >= 11 && &data[0..11] == b"NETSCAPE2.0" {
            if data.len() >= 16 && data[11] == 3 && data[12] == 1 {
                return Some(u16::from_le_bytes([data[13], data[14]]));
            }
        }
        None
    }

    /// Read image data.
    #[allow(clippy::too_many_lines)]
    fn read_image(
        cursor: &mut Cursor<&[u8]>,
        screen_descriptor: &LogicalScreenDescriptor,
        global_color_table: &[u8],
        control: Option<GraphicsControlExtension>,
    ) -> CodecResult<GifFrame> {
        // Read Image Descriptor
        let mut buf = [0u8; 9];
        cursor
            .read_exact(&mut buf)
            .map_err(|_| CodecError::InvalidData("Failed to read image descriptor".into()))?;

        let left = u16::from_le_bytes([buf[0], buf[1]]);
        let top = u16::from_le_bytes([buf[2], buf[3]]);
        let width = u16::from_le_bytes([buf[4], buf[5]]);
        let height = u16::from_le_bytes([buf[6], buf[7]]);
        let packed = buf[8];

        let has_local_color_table = (packed & 0x80) != 0;
        let interlaced = (packed & 0x40) != 0;
        let sort_flag = (packed & 0x20) != 0;
        let local_color_table_size = packed & 0x07;

        let descriptor = ImageDescriptor {
            left,
            top,
            width,
            height,
            has_local_color_table,
            interlaced,
            sort_flag,
            local_color_table_size,
        };

        // Read Local Color Table if present
        let color_table = if has_local_color_table {
            let size = Self::color_table_size(local_color_table_size);
            Self::read_color_table(cursor, size)?
        } else {
            global_color_table.to_vec()
        };

        // Read LZW minimum code size
        let mut lzw_min_code_size = [0u8; 1];
        cursor
            .read_exact(&mut lzw_min_code_size)
            .map_err(|_| CodecError::InvalidData("Failed to read LZW code size".into()))?;

        // Read image data sub-blocks
        let compressed_data = Self::read_data_sub_blocks(cursor)?;

        // Decompress image data
        let mut decoder = LzwDecoder::new(lzw_min_code_size[0])?;
        let expected_size = (width as usize) * (height as usize);
        let mut indices = decoder.decompress(&compressed_data, expected_size)?;

        // Deinterlace if needed
        if interlaced {
            indices = Self::deinterlace(&indices, width, height)?;
        }

        Ok(GifFrame {
            descriptor,
            control,
            color_table,
            indices,
        })
    }

    /// Deinterlace image data.
    fn deinterlace(indices: &[u8], width: u16, height: u16) -> CodecResult<Vec<u8>> {
        let width = width as usize;
        let height = height as usize;
        let mut deinterlaced = vec![0u8; width * height];

        // GIF interlacing uses 4 passes
        let passes = [
            (0, 8), // Pass 1: every 8th row, starting with row 0
            (4, 8), // Pass 2: every 8th row, starting with row 4
            (2, 4), // Pass 3: every 4th row, starting with row 2
            (1, 2), // Pass 4: every 2nd row, starting with row 1
        ];

        let mut src_idx = 0;
        for (start, step) in &passes {
            let mut y = *start;
            while y < height {
                if src_idx + width <= indices.len() {
                    let dst_idx = y * width;
                    deinterlaced[dst_idx..dst_idx + width]
                        .copy_from_slice(&indices[src_idx..src_idx + width]);
                    src_idx += width;
                }
                y += step;
            }
        }

        Ok(deinterlaced)
    }

    /// Convert a GIF frame to a VideoFrame.
    pub fn frame_to_video_frame(&self, frame_index: usize) -> CodecResult<VideoFrame> {
        if frame_index >= self.frames.len() {
            return Err(CodecError::InvalidParameter(format!(
                "Frame index {} out of range (total: {})",
                frame_index,
                self.frames.len()
            )));
        }

        let frame = &self.frames[frame_index];
        let width = self.screen_descriptor.width as u32;
        let height = self.screen_descriptor.height as u32;

        // Convert indexed colors to RGBA
        let mut rgba_data = vec![0u8; (width * height * 4) as usize];

        // Fill with background color initially
        for y in 0..height as usize {
            for x in 0..width as usize {
                let idx = (y * width as usize + x) * 4;
                rgba_data[idx] = self.background_color[0];
                rgba_data[idx + 1] = self.background_color[1];
                rgba_data[idx + 2] = self.background_color[2];
                rgba_data[idx + 3] = 255;
            }
        }

        // Draw the frame
        let frame_width = frame.descriptor.width as usize;
        let frame_height = frame.descriptor.height as usize;
        let left = frame.descriptor.left as usize;
        let top = frame.descriptor.top as usize;

        for y in 0..frame_height {
            for x in 0..frame_width {
                let canvas_x = left + x;
                let canvas_y = top + y;

                if canvas_x >= width as usize || canvas_y >= height as usize {
                    continue;
                }

                let src_idx = y * frame_width + x;
                if src_idx >= frame.indices.len() {
                    continue;
                }

                let color_index = frame.indices[src_idx] as usize;

                // Check for transparency
                if let Some(ref control) = frame.control {
                    if control.has_transparency
                        && color_index == control.transparent_color_index as usize
                    {
                        continue;
                    }
                }

                // Get color from palette
                let color_offset = color_index * 3;
                if color_offset + 2 < frame.color_table.len() {
                    let dst_idx = (canvas_y * width as usize + canvas_x) * 4;
                    rgba_data[dst_idx] = frame.color_table[color_offset];
                    rgba_data[dst_idx + 1] = frame.color_table[color_offset + 1];
                    rgba_data[dst_idx + 2] = frame.color_table[color_offset + 2];
                    rgba_data[dst_idx + 3] = 255;
                }
            }
        }

        let stride = (width * 4) as usize;
        let plane = Plane {
            data: rgba_data,
            stride,
            width,
            height,
        };

        let mut video_frame = VideoFrame::new(PixelFormat::Rgba32, width, height);
        video_frame.planes = vec![plane];

        Ok(video_frame)
    }
}

/// Extension types.
#[derive(Debug)]
enum Extension {
    /// Graphics Control Extension.
    GraphicsControl(GraphicsControlExtension),
    /// Application Extension.
    Application(Vec<u8>),
    /// Comment Extension.
    #[allow(dead_code)]
    Comment(Vec<u8>),
    /// Plain Text Extension.
    #[allow(dead_code)]
    PlainText(Vec<u8>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_table_size() {
        assert_eq!(GifDecoderState::color_table_size(0), 2);
        assert_eq!(GifDecoderState::color_table_size(1), 4);
        assert_eq!(GifDecoderState::color_table_size(7), 256);
    }

    #[test]
    fn test_deinterlace() {
        let width = 4;
        let height = 4;
        let indices: Vec<u8> = (0..16).collect();
        let result = GifDecoderState::deinterlace(&indices, width, height).expect("should succeed");
        assert_eq!(result.len(), 16);
    }
}
