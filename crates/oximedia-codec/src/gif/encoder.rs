//! GIF encoder implementation.
//!
//! Supports:
//! - Color quantization (median cut, octree algorithms)
//! - Dithering (Floyd-Steinberg, ordered)
//! - Animation encoding
//! - Transparency
//! - Disposal methods

use super::lzw::LzwEncoder;
use crate::error::{CodecError, CodecResult};
use crate::frame::VideoFrame;
use oximedia_core::PixelFormat;
use std::collections::HashMap;
use std::io::Write;

/// Maximum colors in GIF palette.
const MAX_COLORS: usize = 256;

/// GIF signature and version.
const GIF89A_HEADER: &[u8] = b"GIF89a";

/// Extension introducer.
const EXTENSION_INTRODUCER: u8 = 0x21;

/// Image separator.
const IMAGE_SEPARATOR: u8 = 0x2C;

/// Trailer.
const TRAILER: u8 = 0x3B;

/// Graphics Control Extension label.
const GRAPHICS_CONTROL_LABEL: u8 = 0xF9;

/// Application Extension label.
const APPLICATION_LABEL: u8 = 0xFF;

/// Disposal method: No disposal specified.
#[allow(dead_code)]
const DISPOSAL_NONE: u8 = 0;

/// Disposal method: Keep frame.
#[allow(dead_code)]
const DISPOSAL_KEEP: u8 = 1;

/// Disposal method: Restore to background.
#[allow(dead_code)]
const DISPOSAL_BACKGROUND: u8 = 2;

/// Disposal method: Restore to previous.
#[allow(dead_code)]
const DISPOSAL_PREVIOUS: u8 = 3;

/// Dithering method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DitheringMethod {
    /// No dithering.
    None,
    /// Floyd-Steinberg dithering.
    FloydSteinberg,
    /// Ordered (Bayer) dithering.
    Ordered,
}

/// Color quantization method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMethod {
    /// Median cut algorithm.
    MedianCut,
    /// Octree algorithm.
    Octree,
}

/// GIF encoder configuration.
#[derive(Debug, Clone)]
pub struct GifEncoderConfig {
    /// Number of colors in palette (2-256).
    pub colors: usize,
    /// Color quantization method.
    pub quantization: QuantizationMethod,
    /// Dithering method.
    pub dithering: DitheringMethod,
    /// Transparent color index (None = no transparency).
    pub transparent_index: Option<u8>,
    /// Loop count (0 = infinite, 1 = no loop).
    pub loop_count: u16,
}

impl Default for GifEncoderConfig {
    fn default() -> Self {
        Self {
            colors: 256,
            quantization: QuantizationMethod::MedianCut,
            dithering: DitheringMethod::None,
            transparent_index: None,
            loop_count: 0,
        }
    }
}

/// GIF frame configuration.
#[derive(Debug, Clone)]
pub struct GifFrameConfig {
    /// Delay time in hundredths of a second.
    pub delay_time: u16,
    /// Disposal method.
    pub disposal_method: u8,
    /// Left position on canvas.
    pub left: u16,
    /// Top position on canvas.
    pub top: u16,
}

impl Default for GifFrameConfig {
    fn default() -> Self {
        Self {
            delay_time: 10, // 100ms
            disposal_method: DISPOSAL_BACKGROUND,
            left: 0,
            top: 0,
        }
    }
}

/// GIF encoder state.
pub struct GifEncoderState {
    /// Encoder configuration.
    config: GifEncoderConfig,
    /// Canvas width.
    width: u32,
    /// Canvas height.
    height: u32,
    /// Output buffer.
    output: Vec<u8>,
    /// Global color palette.
    palette: Vec<u8>,
}

impl GifEncoderState {
    /// Create a new GIF encoder.
    pub fn new(width: u32, height: u32, config: GifEncoderConfig) -> CodecResult<Self> {
        if width == 0 || height == 0 || width > 65535 || height > 65535 {
            return Err(CodecError::InvalidParameter(format!(
                "Invalid dimensions: {}x{}",
                width, height
            )));
        }

        if !(2..=256).contains(&config.colors) {
            return Err(CodecError::InvalidParameter(format!(
                "Invalid color count: {}",
                config.colors
            )));
        }

        Ok(Self {
            config,
            width,
            height,
            output: Vec::new(),
            palette: Vec::new(),
        })
    }

    /// Encode frames to GIF.
    ///
    /// # Errors
    ///
    /// Returns error if encoding fails.
    pub fn encode(
        &mut self,
        frames: &[VideoFrame],
        frame_configs: &[GifFrameConfig],
    ) -> CodecResult<Vec<u8>> {
        if frames.is_empty() {
            return Err(CodecError::InvalidParameter("No frames to encode".into()));
        }

        if frames.len() != frame_configs.len() {
            return Err(CodecError::InvalidParameter(
                "Frame count mismatch with configs".into(),
            ));
        }

        self.output.clear();

        // Generate global palette from all frames
        self.palette = self.generate_global_palette(frames)?;

        // Write header
        self.write_header()?;

        // Write logical screen descriptor
        self.write_screen_descriptor()?;

        // Write global color table
        let palette = self.palette.clone();
        self.write_color_table(&palette)?;

        // Write Netscape extension for looping
        if frames.len() > 1 {
            self.write_netscape_extension()?;
        }

        // Write frames
        for (frame, frame_config) in frames.iter().zip(frame_configs) {
            self.write_frame(frame, frame_config)?;
        }

        // Write trailer
        self.output.write_all(&[TRAILER])?;

        Ok(self.output.clone())
    }

    /// Write GIF header.
    fn write_header(&mut self) -> CodecResult<()> {
        self.output.write_all(GIF89A_HEADER)?;
        Ok(())
    }

    /// Write Logical Screen Descriptor.
    fn write_screen_descriptor(&mut self) -> CodecResult<()> {
        let width = self.width as u16;
        let height = self.height as u16;

        self.output.write_all(&width.to_le_bytes())?;
        self.output.write_all(&height.to_le_bytes())?;

        // Packed field
        let global_color_table_flag = 1u8 << 7;
        let color_resolution = 7u8 << 4; // 8 bits per color
        let sort_flag = 0u8;
        let size = Self::color_table_size_field(self.palette.len() / 3);
        let packed = global_color_table_flag | color_resolution | sort_flag | size;

        self.output.write_all(&[packed])?;
        self.output.write_all(&[0])?; // Background color index
        self.output.write_all(&[0])?; // Pixel aspect ratio

        Ok(())
    }

    /// Write color table.
    fn write_color_table(&mut self, table: &[u8]) -> CodecResult<()> {
        let size = Self::next_power_of_two(table.len() / 3) * 3;
        self.output.write_all(table)?;

        // Pad to power of 2
        if table.len() < size {
            self.output
                .resize(self.output.len() + (size - table.len()), 0);
        }

        Ok(())
    }

    /// Write Netscape extension for animation looping.
    fn write_netscape_extension(&mut self) -> CodecResult<()> {
        self.output.write_all(&[EXTENSION_INTRODUCER])?;
        self.output.write_all(&[APPLICATION_LABEL])?;
        self.output.write_all(&[11])?; // Block size
        self.output.write_all(b"NETSCAPE2.0")?;
        self.output.write_all(&[3])?; // Sub-block size
        self.output.write_all(&[1])?; // Loop sub-block ID
        self.output
            .write_all(&self.config.loop_count.to_le_bytes())?;
        self.output.write_all(&[0])?; // Block terminator

        Ok(())
    }

    /// Write a single frame.
    fn write_frame(&mut self, frame: &VideoFrame, config: &GifFrameConfig) -> CodecResult<()> {
        // Write Graphics Control Extension
        self.write_graphics_control_extension(config)?;

        // Convert frame to indexed colors
        let rgba_data = self.frame_to_rgba(frame)?;
        let indices = self.quantize_frame(&rgba_data)?;

        // Write Image Descriptor
        self.write_image_descriptor(config)?;

        // Compress and write image data
        self.write_image_data(&indices)?;

        Ok(())
    }

    /// Write Graphics Control Extension.
    fn write_graphics_control_extension(&mut self, config: &GifFrameConfig) -> CodecResult<()> {
        self.output.write_all(&[EXTENSION_INTRODUCER])?;
        self.output.write_all(&[GRAPHICS_CONTROL_LABEL])?;
        self.output.write_all(&[4])?; // Block size

        // Packed field
        let disposal_method = (config.disposal_method & 0x07) << 2;
        let user_input_flag = 0u8;
        let transparency_flag = if self.config.transparent_index.is_some() {
            1u8
        } else {
            0u8
        };
        let packed = disposal_method | user_input_flag | transparency_flag;

        self.output.write_all(&[packed])?;
        self.output.write_all(&config.delay_time.to_le_bytes())?;
        self.output
            .write_all(&[self.config.transparent_index.unwrap_or(0)])?;
        self.output.write_all(&[0])?; // Block terminator

        Ok(())
    }

    /// Write Image Descriptor.
    fn write_image_descriptor(&mut self, config: &GifFrameConfig) -> CodecResult<()> {
        self.output.write_all(&[IMAGE_SEPARATOR])?;
        self.output.write_all(&config.left.to_le_bytes())?;
        self.output.write_all(&config.top.to_le_bytes())?;
        self.output.write_all(&(self.width as u16).to_le_bytes())?;
        self.output.write_all(&(self.height as u16).to_le_bytes())?;

        // Packed field (no local color table, no interlace)
        self.output.write_all(&[0])?;

        Ok(())
    }

    /// Write compressed image data.
    fn write_image_data(&mut self, indices: &[u8]) -> CodecResult<()> {
        // Calculate LZW minimum code size
        let color_bits = Self::bits_needed(self.palette.len() / 3);
        let min_code_size = color_bits.max(2);

        self.output.write_all(&[min_code_size])?;

        // Compress with LZW
        let mut encoder = LzwEncoder::new(min_code_size)?;
        let compressed = encoder.compress(indices)?;

        // Write data in sub-blocks
        let mut offset = 0;
        while offset < compressed.len() {
            let block_size = (compressed.len() - offset).min(255);
            self.output.write_all(&[block_size as u8])?;
            self.output
                .write_all(&compressed[offset..offset + block_size])?;
            offset += block_size;
        }

        // Block terminator
        self.output.write_all(&[0])?;

        Ok(())
    }

    /// Convert VideoFrame to RGBA data.
    fn frame_to_rgba(&self, frame: &VideoFrame) -> CodecResult<Vec<u8>> {
        if frame.width != self.width || frame.height != self.height {
            return Err(CodecError::InvalidParameter(format!(
                "Frame size {}x{} doesn't match canvas {}x{}",
                frame.width, frame.height, self.width, self.height
            )));
        }

        match frame.format {
            PixelFormat::Rgba32 => {
                if frame.planes.is_empty() {
                    return Err(CodecError::InvalidData("Frame has no planes".into()));
                }
                Ok(frame.planes[0].data.to_vec())
            }
            PixelFormat::Rgb24 => {
                if frame.planes.is_empty() {
                    return Err(CodecError::InvalidData("Frame has no planes".into()));
                }
                let rgb = &frame.planes[0].data;
                let mut rgba = Vec::with_capacity((self.width * self.height * 4) as usize);
                for chunk in rgb.chunks_exact(3) {
                    rgba.extend_from_slice(chunk);
                    rgba.push(255);
                }
                Ok(rgba)
            }
            _ => Err(CodecError::UnsupportedFeature(format!(
                "Pixel format {} not supported for GIF encoding",
                frame.format
            ))),
        }
    }

    /// Generate global palette from all frames using quantization.
    fn generate_global_palette(&self, frames: &[VideoFrame]) -> CodecResult<Vec<u8>> {
        // Collect all unique colors from all frames
        let mut all_colors = Vec::new();

        for frame in frames {
            let rgba = self.frame_to_rgba(frame)?;
            for chunk in rgba.chunks_exact(4) {
                let color = [chunk[0], chunk[1], chunk[2]];
                all_colors.push(color);
            }
        }

        // Apply quantization
        let palette = match self.config.quantization {
            QuantizationMethod::MedianCut => self.median_cut_quantize(&all_colors)?,
            QuantizationMethod::Octree => self.octree_quantize(&all_colors)?,
        };

        Ok(palette)
    }

    /// Median cut color quantization.
    fn median_cut_quantize(&self, colors: &[[u8; 3]]) -> CodecResult<Vec<u8>> {
        let target_colors = self.config.colors.min(MAX_COLORS);

        // Start with all colors in one bucket
        let mut buckets = vec![colors.to_vec()];

        // Split buckets until we have enough colors
        while buckets.len() < target_colors {
            // Find largest bucket
            let largest_idx = buckets
                .iter()
                .enumerate()
                .max_by_key(|(_, b)| b.len())
                .map(|(i, _)| i)
                .expect("buckets is non-empty inside the while loop");

            let bucket = buckets.remove(largest_idx);
            if bucket.is_empty() {
                break;
            }

            // Find channel with largest range
            let (mut min_r, mut max_r) = (255, 0);
            let (mut min_g, mut max_g) = (255, 0);
            let (mut min_b, mut max_b) = (255, 0);

            for color in &bucket {
                min_r = min_r.min(color[0]);
                max_r = max_r.max(color[0]);
                min_g = min_g.min(color[1]);
                max_g = max_g.max(color[1]);
                min_b = min_b.min(color[2]);
                max_b = max_b.max(color[2]);
            }

            let range_r = max_r - min_r;
            let range_g = max_g - min_g;
            let range_b = max_b - min_b;

            // Sort by channel with largest range
            let mut bucket = bucket;
            if range_r >= range_g && range_r >= range_b {
                bucket.sort_by_key(|c| c[0]);
            } else if range_g >= range_r && range_g >= range_b {
                bucket.sort_by_key(|c| c[1]);
            } else {
                bucket.sort_by_key(|c| c[2]);
            }

            // Split at median
            let mid = bucket.len() / 2;
            let (left, right) = bucket.split_at(mid);
            buckets.push(left.to_vec());
            buckets.push(right.to_vec());
        }

        // Average colors in each bucket to get palette
        let mut palette = Vec::with_capacity(target_colors * 3);
        for bucket in buckets {
            if bucket.is_empty() {
                continue;
            }

            let mut sum_r = 0u32;
            let mut sum_g = 0u32;
            let mut sum_b = 0u32;

            for color in &bucket {
                sum_r += u32::from(color[0]);
                sum_g += u32::from(color[1]);
                sum_b += u32::from(color[2]);
            }

            let count = bucket.len() as u32;
            palette.push((sum_r / count) as u8);
            palette.push((sum_g / count) as u8);
            palette.push((sum_b / count) as u8);
        }

        Ok(palette)
    }

    /// Octree color quantization.
    fn octree_quantize(&self, colors: &[[u8; 3]]) -> CodecResult<Vec<u8>> {
        let mut tree = OctreeQuantizer::new(self.config.colors);

        for &color in colors {
            tree.add_color(color);
        }

        let palette = tree.get_palette();
        Ok(palette)
    }

    /// Quantize frame to palette indices.
    fn quantize_frame(&self, rgba: &[u8]) -> CodecResult<Vec<u8>> {
        let mut indices = Vec::with_capacity((self.width * self.height) as usize);

        match self.config.dithering {
            DitheringMethod::None => {
                for chunk in rgba.chunks_exact(4) {
                    let color = [chunk[0], chunk[1], chunk[2]];
                    let index = self.find_closest_color(color);
                    indices.push(index);
                }
            }
            DitheringMethod::FloydSteinberg => {
                indices = self.floyd_steinberg_dither(rgba)?;
            }
            DitheringMethod::Ordered => {
                indices = self.ordered_dither(rgba)?;
            }
        }

        Ok(indices)
    }

    /// Find closest color in palette.
    fn find_closest_color(&self, color: [u8; 3]) -> u8 {
        let mut best_index = 0;
        let mut best_distance = u32::MAX;

        for i in 0..(self.palette.len() / 3) {
            let pal_r = self.palette[i * 3];
            let pal_g = self.palette[i * 3 + 1];
            let pal_b = self.palette[i * 3 + 2];

            let dr = i32::from(color[0]) - i32::from(pal_r);
            let dg = i32::from(color[1]) - i32::from(pal_g);
            let db = i32::from(color[2]) - i32::from(pal_b);

            let distance = (dr * dr + dg * dg + db * db) as u32;

            if distance < best_distance {
                best_distance = distance;
                best_index = i;
            }
        }

        best_index as u8
    }

    /// Floyd-Steinberg dithering.
    #[allow(clippy::cast_possible_wrap)]
    fn floyd_steinberg_dither(&self, rgba: &[u8]) -> CodecResult<Vec<u8>> {
        let width = self.width as usize;
        let height = self.height as usize;

        // Create error buffer
        let mut errors = vec![[0i16; 3]; width * height];
        let mut indices = Vec::with_capacity(width * height);

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 4;
                let pixel_idx = y * width + x;

                // Get original color with accumulated error
                let r = (i16::from(rgba[idx]) + errors[pixel_idx][0]).clamp(0, 255) as u8;
                let g = (i16::from(rgba[idx + 1]) + errors[pixel_idx][1]).clamp(0, 255) as u8;
                let b = (i16::from(rgba[idx + 2]) + errors[pixel_idx][2]).clamp(0, 255) as u8;

                // Find closest palette color
                let color = [r, g, b];
                let index = self.find_closest_color(color);
                indices.push(index);

                // Calculate error
                let pal_r = self.palette[index as usize * 3];
                let pal_g = self.palette[index as usize * 3 + 1];
                let pal_b = self.palette[index as usize * 3 + 2];

                let err_r = i16::from(r) - i16::from(pal_r);
                let err_g = i16::from(g) - i16::from(pal_g);
                let err_b = i16::from(b) - i16::from(pal_b);

                // Distribute error to neighbors
                if x + 1 < width {
                    let next_idx = pixel_idx + 1;
                    errors[next_idx][0] += err_r * 7 / 16;
                    errors[next_idx][1] += err_g * 7 / 16;
                    errors[next_idx][2] += err_b * 7 / 16;
                }

                if y + 1 < height {
                    if x > 0 {
                        let next_idx = pixel_idx + width - 1;
                        errors[next_idx][0] += err_r * 3 / 16;
                        errors[next_idx][1] += err_g * 3 / 16;
                        errors[next_idx][2] += err_b * 3 / 16;
                    }

                    let next_idx = pixel_idx + width;
                    errors[next_idx][0] += err_r * 5 / 16;
                    errors[next_idx][1] += err_g * 5 / 16;
                    errors[next_idx][2] += err_b * 5 / 16;

                    if x + 1 < width {
                        let next_idx = pixel_idx + width + 1;
                        errors[next_idx][0] += err_r / 16;
                        errors[next_idx][1] += err_g / 16;
                        errors[next_idx][2] += err_b / 16;
                    }
                }
            }
        }

        Ok(indices)
    }

    /// Ordered (Bayer) dithering.
    fn ordered_dither(&self, rgba: &[u8]) -> CodecResult<Vec<u8>> {
        // 4x4 Bayer matrix
        #[rustfmt::skip]
        const BAYER_MATRIX: [[i16; 4]; 4] = [
            [  0,  8,  2, 10 ],
            [ 12,  4, 14,  6 ],
            [  3, 11,  1,  9 ],
            [ 15,  7, 13,  5 ],
        ];

        let width = self.width as usize;
        let height = self.height as usize;
        let mut indices = Vec::with_capacity(width * height);

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 4;

                // Apply Bayer matrix threshold
                let threshold = BAYER_MATRIX[y % 4][x % 4] * 16 - 128;

                let r = (i16::from(rgba[idx]) + threshold).clamp(0, 255) as u8;
                let g = (i16::from(rgba[idx + 1]) + threshold).clamp(0, 255) as u8;
                let b = (i16::from(rgba[idx + 2]) + threshold).clamp(0, 255) as u8;

                let color = [r, g, b];
                let index = self.find_closest_color(color);
                indices.push(index);
            }
        }

        Ok(indices)
    }

    /// Calculate size field for color table.
    fn color_table_size_field(colors: usize) -> u8 {
        let size = Self::next_power_of_two(colors);
        let bits = Self::bits_needed(size);
        bits.saturating_sub(1)
    }

    /// Calculate next power of two.
    fn next_power_of_two(n: usize) -> usize {
        let mut power = 2;
        while power < n {
            power *= 2;
        }
        power
    }

    /// Calculate bits needed to represent n values.
    fn bits_needed(n: usize) -> u8 {
        if n <= 2 {
            1
        } else {
            (n as f64).log2().ceil() as u8
        }
    }
}

/// Octree node for color quantization.
struct OctreeNode {
    children: [Option<Box<OctreeNode>>; 8],
    color_sum: [u32; 3],
    pixel_count: u32,
    is_leaf: bool,
}

impl OctreeNode {
    fn new() -> Self {
        Self {
            children: Default::default(),
            color_sum: [0, 0, 0],
            pixel_count: 0,
            is_leaf: false,
        }
    }
}

/// Octree quantizer for color reduction.
struct OctreeQuantizer {
    root: OctreeNode,
    #[allow(dead_code)]
    max_colors: usize,
    leaf_count: usize,
}

impl OctreeQuantizer {
    fn new(max_colors: usize) -> Self {
        Self {
            root: OctreeNode::new(),
            max_colors,
            leaf_count: 0,
        }
    }

    fn add_color(&mut self, color: [u8; 3]) {
        Self::add_color_recursive(&mut self.root, color, 0, &mut self.leaf_count);
    }

    fn add_color_recursive(
        node: &mut OctreeNode,
        color: [u8; 3],
        depth: u8,
        leaf_count: &mut usize,
    ) {
        if depth >= 8 || node.is_leaf {
            node.color_sum[0] += u32::from(color[0]);
            node.color_sum[1] += u32::from(color[1]);
            node.color_sum[2] += u32::from(color[2]);
            node.pixel_count += 1;
            if !node.is_leaf {
                node.is_leaf = true;
                *leaf_count += 1;
            }
            return;
        }

        let index = Self::get_child_index(color, depth);

        if node.children[index].is_none() {
            node.children[index] = Some(Box::new(OctreeNode::new()));
        }

        if let Some(child) = &mut node.children[index] {
            Self::add_color_recursive(child, color, depth + 1, leaf_count);
        }
    }

    fn get_child_index(color: [u8; 3], depth: u8) -> usize {
        let shift = 7 - depth;
        let r_bit = ((color[0] >> shift) & 1) as usize;
        let g_bit = ((color[1] >> shift) & 1) as usize;
        let b_bit = ((color[2] >> shift) & 1) as usize;
        (r_bit << 2) | (g_bit << 1) | b_bit
    }

    fn get_palette(&self) -> Vec<u8> {
        let mut palette = Vec::new();
        self.collect_colors(&self.root, &mut palette);
        palette
    }

    fn collect_colors(&self, node: &OctreeNode, palette: &mut Vec<u8>) {
        if node.is_leaf && node.pixel_count > 0 {
            let r = (node.color_sum[0] / node.pixel_count) as u8;
            let g = (node.color_sum[1] / node.pixel_count) as u8;
            let b = (node.color_sum[2] / node.pixel_count) as u8;
            palette.extend_from_slice(&[r, g, b]);
            return;
        }

        for child in &node.children {
            if let Some(child) = child {
                self.collect_colors(child, palette);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_table_size_field() {
        assert_eq!(GifEncoderState::color_table_size_field(2), 0);
        assert_eq!(GifEncoderState::color_table_size_field(4), 1);
        assert_eq!(GifEncoderState::color_table_size_field(256), 7);
    }

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(GifEncoderState::next_power_of_two(1), 2);
        assert_eq!(GifEncoderState::next_power_of_two(3), 4);
        assert_eq!(GifEncoderState::next_power_of_two(100), 128);
    }

    #[test]
    fn test_bits_needed() {
        assert_eq!(GifEncoderState::bits_needed(2), 1);
        assert_eq!(GifEncoderState::bits_needed(4), 2);
        assert_eq!(GifEncoderState::bits_needed(256), 8);
    }
}
