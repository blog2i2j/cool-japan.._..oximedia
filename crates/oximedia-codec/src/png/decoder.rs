//! PNG decoder implementation.
//!
//! Implements a complete PNG 1.2 specification decoder with support for:
//! - All color types (Grayscale, RGB, Palette, GrayscaleAlpha, RGBA)
//! - All bit depths (1, 2, 4, 8, 16)
//! - Interlacing (Adam7)
//! - All filter types (0-4)
//! - Ancillary chunks (tRNS, gAMA, cHRM, etc.)
//! - CRC validation

use super::filter::{unfilter, FilterType};
use crate::error::{CodecError, CodecResult};
use bytes::Bytes;
use oxiarc_deflate::ZlibStreamDecoder;
use std::io::Read;

/// PNG signature bytes.
const PNG_SIGNATURE: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];

/// PNG color types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorType {
    /// Grayscale.
    Grayscale = 0,
    /// RGB truecolor.
    Rgb = 2,
    /// Indexed color (palette).
    Palette = 3,
    /// Grayscale with alpha.
    GrayscaleAlpha = 4,
    /// RGB with alpha.
    Rgba = 6,
}

impl ColorType {
    /// Create color type from byte.
    ///
    /// # Errors
    ///
    /// Returns error if color type is invalid.
    pub fn from_u8(value: u8) -> CodecResult<Self> {
        match value {
            0 => Ok(Self::Grayscale),
            2 => Ok(Self::Rgb),
            3 => Ok(Self::Palette),
            4 => Ok(Self::GrayscaleAlpha),
            6 => Ok(Self::Rgba),
            _ => Err(CodecError::InvalidData(format!(
                "Invalid color type: {value}"
            ))),
        }
    }

    /// Get number of samples per pixel.
    #[must_use]
    pub const fn samples_per_pixel(self) -> usize {
        match self {
            Self::Grayscale => 1,
            Self::Rgb => 3,
            Self::Palette => 1,
            Self::GrayscaleAlpha => 2,
            Self::Rgba => 4,
        }
    }

    /// Check if color type has alpha channel.
    #[must_use]
    pub const fn has_alpha(self) -> bool {
        matches!(self, Self::GrayscaleAlpha | Self::Rgba)
    }
}

/// PNG image header (IHDR chunk).
#[derive(Debug, Clone)]
pub struct ImageHeader {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Bit depth.
    pub bit_depth: u8,
    /// Color type.
    pub color_type: ColorType,
    /// Compression method (always 0 for PNG).
    pub compression: u8,
    /// Filter method (always 0 for PNG).
    pub filter_method: u8,
    /// Interlace method (0 = none, 1 = Adam7).
    pub interlace: u8,
}

impl ImageHeader {
    /// Parse IHDR chunk data.
    ///
    /// # Errors
    ///
    /// Returns error if IHDR data is invalid.
    pub fn parse(data: &[u8]) -> CodecResult<Self> {
        if data.len() < 13 {
            return Err(CodecError::InvalidData("IHDR too short".into()));
        }

        let width = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        let height = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
        let bit_depth = data[8];
        let color_type = ColorType::from_u8(data[9])?;
        let compression = data[10];
        let filter_method = data[11];
        let interlace = data[12];

        if width == 0 || height == 0 {
            return Err(CodecError::InvalidData("Invalid dimensions".into()));
        }

        if compression != 0 {
            return Err(CodecError::UnsupportedFeature(format!(
                "Unsupported compression method: {compression}"
            )));
        }

        if filter_method != 0 {
            return Err(CodecError::UnsupportedFeature(format!(
                "Unsupported filter method: {filter_method}"
            )));
        }

        if interlace > 1 {
            return Err(CodecError::InvalidData(format!(
                "Invalid interlace method: {interlace}"
            )));
        }

        // Validate bit depth for color type
        let valid_depths = match color_type {
            ColorType::Grayscale => &[1, 2, 4, 8, 16][..],
            ColorType::Rgb => &[8, 16][..],
            ColorType::Palette => &[1, 2, 4, 8][..],
            ColorType::GrayscaleAlpha => &[8, 16][..],
            ColorType::Rgba => &[8, 16][..],
        };

        if !valid_depths.contains(&bit_depth) {
            return Err(CodecError::InvalidData(format!(
                "Invalid bit depth {bit_depth} for color type {color_type:?}"
            )));
        }

        Ok(Self {
            width,
            height,
            bit_depth,
            color_type,
            compression,
            filter_method,
            interlace,
        })
    }

    /// Get bytes per pixel (rounded up).
    #[must_use]
    pub fn bytes_per_pixel(&self) -> usize {
        let bits_per_pixel = self.color_type.samples_per_pixel() * self.bit_depth as usize;
        (bits_per_pixel + 7) / 8
    }

    /// Get scanline length in bytes.
    #[must_use]
    pub fn scanline_length(&self) -> usize {
        let bits_per_scanline =
            self.width as usize * self.color_type.samples_per_pixel() * self.bit_depth as usize;
        (bits_per_scanline + 7) / 8
    }
}

/// PNG chunk.
#[derive(Debug)]
struct Chunk {
    /// Chunk type (4 bytes).
    chunk_type: [u8; 4],
    /// Chunk data.
    data: Vec<u8>,
}

impl Chunk {
    /// Read a chunk from data.
    fn read(data: &[u8], offset: &mut usize) -> CodecResult<Self> {
        if *offset + 12 > data.len() {
            return Err(CodecError::InvalidData("Incomplete chunk".into()));
        }

        let length = u32::from_be_bytes([
            data[*offset],
            data[*offset + 1],
            data[*offset + 2],
            data[*offset + 3],
        ]) as usize;
        *offset += 4;

        let chunk_type = [
            data[*offset],
            data[*offset + 1],
            data[*offset + 2],
            data[*offset + 3],
        ];
        *offset += 4;

        if *offset + length + 4 > data.len() {
            return Err(CodecError::InvalidData("Incomplete chunk data".into()));
        }

        let chunk_data = data[*offset..*offset + length].to_vec();
        *offset += length;

        let expected_crc = u32::from_be_bytes([
            data[*offset],
            data[*offset + 1],
            data[*offset + 2],
            data[*offset + 3],
        ]);
        *offset += 4;

        // Validate CRC
        let actual_crc = crc32(&chunk_type, &chunk_data);
        if actual_crc != expected_crc {
            return Err(CodecError::InvalidData(format!(
                "CRC mismatch for chunk {:?}",
                std::str::from_utf8(&chunk_type).unwrap_or("???")
            )));
        }

        Ok(Self {
            chunk_type,
            data: chunk_data,
        })
    }

    /// Get chunk type as string.
    fn type_str(&self) -> &str {
        std::str::from_utf8(&self.chunk_type).unwrap_or("????")
    }
}

/// Calculate CRC32 for PNG chunk.
fn crc32(chunk_type: &[u8; 4], data: &[u8]) -> u32 {
    let mut crc = !0u32;

    for &byte in chunk_type.iter().chain(data.iter()) {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                0xedb8_8320 ^ (crc >> 1)
            } else {
                crc >> 1
            };
        }
    }

    !crc
}

/// Adam7 interlacing pass information.
struct Adam7Pass {
    x_start: u32,
    y_start: u32,
    x_step: u32,
    y_step: u32,
}

const ADAM7_PASSES: [Adam7Pass; 7] = [
    Adam7Pass {
        x_start: 0,
        y_start: 0,
        x_step: 8,
        y_step: 8,
    },
    Adam7Pass {
        x_start: 4,
        y_start: 0,
        x_step: 8,
        y_step: 8,
    },
    Adam7Pass {
        x_start: 0,
        y_start: 4,
        x_step: 4,
        y_step: 8,
    },
    Adam7Pass {
        x_start: 2,
        y_start: 0,
        x_step: 4,
        y_step: 4,
    },
    Adam7Pass {
        x_start: 0,
        y_start: 2,
        x_step: 2,
        y_step: 4,
    },
    Adam7Pass {
        x_start: 1,
        y_start: 0,
        x_step: 2,
        y_step: 2,
    },
    Adam7Pass {
        x_start: 0,
        y_start: 1,
        x_step: 1,
        y_step: 2,
    },
];

/// PNG decoder.
pub struct PngDecoder {
    header: ImageHeader,
    palette: Option<Vec<u8>>,
    transparency: Option<Vec<u8>>,
    #[allow(dead_code)]
    gamma: Option<f64>,
    image_data: Vec<u8>,
}

impl PngDecoder {
    /// Create a new PNG decoder.
    ///
    /// # Errors
    ///
    /// Returns error if PNG data is invalid.
    pub fn new(data: &[u8]) -> CodecResult<Self> {
        if data.len() < 8 {
            return Err(CodecError::InvalidData("PNG data too short".into()));
        }

        // Validate PNG signature
        if &data[0..8] != PNG_SIGNATURE {
            return Err(CodecError::InvalidData("Invalid PNG signature".into()));
        }

        let mut offset = 8;
        let mut header: Option<ImageHeader> = None;
        let mut palette: Option<Vec<u8>> = None;
        let mut transparency: Option<Vec<u8>> = None;
        let mut gamma: Option<f64> = None;
        let mut idat_chunks = Vec::new();

        // Parse chunks
        while offset < data.len() {
            let chunk = Chunk::read(data, &mut offset)?;

            match chunk.type_str() {
                "IHDR" => {
                    if header.is_some() {
                        return Err(CodecError::InvalidData("Multiple IHDR chunks".into()));
                    }
                    header = Some(ImageHeader::parse(&chunk.data)?);
                }
                "PLTE" => {
                    if palette.is_some() {
                        return Err(CodecError::InvalidData("Multiple PLTE chunks".into()));
                    }
                    if chunk.data.len() % 3 != 0 || chunk.data.is_empty() {
                        return Err(CodecError::InvalidData("Invalid PLTE chunk".into()));
                    }
                    palette = Some(chunk.data);
                }
                "IDAT" => {
                    idat_chunks.push(chunk.data);
                }
                "IEND" => {
                    break;
                }
                "tRNS" => {
                    transparency = Some(chunk.data);
                }
                "gAMA" => {
                    if chunk.data.len() == 4 {
                        let gamma_int = u32::from_be_bytes([
                            chunk.data[0],
                            chunk.data[1],
                            chunk.data[2],
                            chunk.data[3],
                        ]);
                        gamma = Some(f64::from(gamma_int) / 100_000.0);
                    }
                }
                _ => {
                    // Skip unknown ancillary chunks
                }
            }
        }

        let header = header.ok_or_else(|| CodecError::InvalidData("Missing IHDR chunk".into()))?;

        if header.color_type == ColorType::Palette && palette.is_none() {
            return Err(CodecError::InvalidData(
                "Palette color type requires PLTE chunk".into(),
            ));
        }

        // Decompress IDAT data
        let compressed_data: Vec<u8> = idat_chunks.into_iter().flatten().collect();
        let mut decoder = ZlibStreamDecoder::new(&compressed_data[..]);
        let mut image_data = Vec::new();
        decoder
            .read_to_end(&mut image_data)
            .map_err(|e| CodecError::DecoderError(format!("DEFLATE decompression failed: {e}")))?;

        Ok(Self {
            header,
            palette,
            transparency,
            gamma,
            image_data,
        })
    }

    /// Get image width.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.header.width
    }

    /// Get image height.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.header.height
    }

    /// Get color type.
    #[must_use]
    pub const fn color_type(&self) -> ColorType {
        self.header.color_type
    }

    /// Get bit depth.
    #[must_use]
    pub const fn bit_depth(&self) -> u8 {
        self.header.bit_depth
    }

    /// Check if image is interlaced.
    #[must_use]
    pub const fn is_interlaced(&self) -> bool {
        self.header.interlace == 1
    }

    /// Decode the PNG image.
    ///
    /// # Errors
    ///
    /// Returns error if decoding fails.
    pub fn decode(&self) -> CodecResult<DecodedImage> {
        let raw_data = if self.is_interlaced() {
            self.decode_interlaced()?
        } else {
            self.decode_sequential()?
        };

        let rgba_data = self.convert_to_rgba(&raw_data)?;

        Ok(DecodedImage {
            width: self.header.width,
            height: self.header.height,
            data: Bytes::from(rgba_data),
        })
    }

    /// Decode sequential (non-interlaced) image.
    fn decode_sequential(&self) -> CodecResult<Vec<u8>> {
        let scanline_len = self.header.scanline_length();
        let bytes_per_pixel = self.header.bytes_per_pixel();
        let expected_len = (scanline_len + 1) * self.header.height as usize;

        if self.image_data.len() < expected_len {
            return Err(CodecError::InvalidData("Insufficient image data".into()));
        }

        let mut output = Vec::with_capacity(scanline_len * self.header.height as usize);
        let mut prev_scanline: Option<Vec<u8>> = None;

        for y in 0..self.header.height as usize {
            let offset = y * (scanline_len + 1);
            let filter_type = FilterType::from_u8(self.image_data[offset])?;
            let filtered = &self.image_data[offset + 1..offset + 1 + scanline_len];

            let scanline = unfilter(
                filter_type,
                filtered,
                prev_scanline.as_deref(),
                bytes_per_pixel,
            )?;

            output.extend_from_slice(&scanline);
            prev_scanline = Some(scanline);
        }

        Ok(output)
    }

    /// Decode interlaced (Adam7) image.
    #[allow(clippy::similar_names)]
    fn decode_interlaced(&self) -> CodecResult<Vec<u8>> {
        let bytes_per_pixel = self.header.bytes_per_pixel();
        let bits_per_pixel =
            self.header.color_type.samples_per_pixel() * self.header.bit_depth as usize;
        let bytes_per_sample = (self.header.bit_depth as usize + 7) / 8;

        let total_pixels =
            self.header.width as usize * self.header.height as usize * bytes_per_pixel;
        let mut output = vec![0u8; total_pixels];

        let mut data_offset = 0;

        for pass in &ADAM7_PASSES {
            let pass_width =
                (self.header.width.saturating_sub(pass.x_start) + pass.x_step - 1) / pass.x_step;
            let pass_height =
                (self.header.height.saturating_sub(pass.y_start) + pass.y_step - 1) / pass.y_step;

            if pass_width == 0 || pass_height == 0 {
                continue;
            }

            let scanline_bits = pass_width as usize * bits_per_pixel;
            let scanline_len = (scanline_bits + 7) / 8;

            let mut prev_scanline: Option<Vec<u8>> = None;

            for py in 0..pass_height {
                if data_offset >= self.image_data.len() {
                    return Err(CodecError::InvalidData(
                        "Insufficient data for interlaced image".into(),
                    ));
                }

                let filter_type = FilterType::from_u8(self.image_data[data_offset])?;
                data_offset += 1;

                if data_offset + scanline_len > self.image_data.len() {
                    return Err(CodecError::InvalidData(
                        "Insufficient data for scanline".into(),
                    ));
                }

                let filtered = &self.image_data[data_offset..data_offset + scanline_len];
                data_offset += scanline_len;

                let scanline = unfilter(
                    filter_type,
                    filtered,
                    prev_scanline.as_deref(),
                    bytes_per_pixel,
                )?;

                // Copy pixels to output
                let y = (pass.y_start + py * pass.y_step) as usize;
                for px in 0..pass_width as usize {
                    let x = pass.x_start as usize + px * pass.x_step as usize;
                    let src_offset = px * bytes_per_pixel;
                    let dst_offset = (y * self.header.width as usize + x) * bytes_per_pixel;

                    if src_offset + bytes_per_pixel <= scanline.len()
                        && dst_offset + bytes_per_pixel <= output.len()
                    {
                        output[dst_offset..dst_offset + bytes_per_pixel]
                            .copy_from_slice(&scanline[src_offset..src_offset + bytes_per_pixel]);
                    }
                }

                prev_scanline = Some(scanline);
            }
        }

        Ok(output)
    }

    /// Convert raw image data to RGBA format.
    #[allow(clippy::too_many_lines)]
    fn convert_to_rgba(&self, raw_data: &[u8]) -> CodecResult<Vec<u8>> {
        let pixel_count = (self.header.width * self.header.height) as usize;
        let mut rgba = vec![255u8; pixel_count * 4];

        match self.header.color_type {
            ColorType::Grayscale => {
                if self.header.bit_depth == 8 {
                    for i in 0..pixel_count {
                        let gray = raw_data[i];
                        rgba[i * 4] = gray;
                        rgba[i * 4 + 1] = gray;
                        rgba[i * 4 + 2] = gray;
                        rgba[i * 4 + 3] = 255;
                    }
                } else if self.header.bit_depth == 16 {
                    for i in 0..pixel_count {
                        let gray = raw_data[i * 2];
                        rgba[i * 4] = gray;
                        rgba[i * 4 + 1] = gray;
                        rgba[i * 4 + 2] = gray;
                        rgba[i * 4 + 3] = 255;
                    }
                } else {
                    // Expand low bit depths
                    self.expand_grayscale(raw_data, &mut rgba)?;
                }
            }
            ColorType::Rgb => {
                if self.header.bit_depth == 8 {
                    for i in 0..pixel_count {
                        rgba[i * 4] = raw_data[i * 3];
                        rgba[i * 4 + 1] = raw_data[i * 3 + 1];
                        rgba[i * 4 + 2] = raw_data[i * 3 + 2];
                        rgba[i * 4 + 3] = 255;
                    }
                } else if self.header.bit_depth == 16 {
                    for i in 0..pixel_count {
                        rgba[i * 4] = raw_data[i * 6];
                        rgba[i * 4 + 1] = raw_data[i * 6 + 2];
                        rgba[i * 4 + 2] = raw_data[i * 6 + 4];
                        rgba[i * 4 + 3] = 255;
                    }
                }
            }
            ColorType::Palette => {
                let palette = self
                    .palette
                    .as_ref()
                    .ok_or_else(|| CodecError::InvalidData("Missing palette".into()))?;
                self.expand_palette(raw_data, palette, &mut rgba)?;
            }
            ColorType::GrayscaleAlpha => {
                if self.header.bit_depth == 8 {
                    for i in 0..pixel_count {
                        let gray = raw_data[i * 2];
                        let alpha = raw_data[i * 2 + 1];
                        rgba[i * 4] = gray;
                        rgba[i * 4 + 1] = gray;
                        rgba[i * 4 + 2] = gray;
                        rgba[i * 4 + 3] = alpha;
                    }
                } else if self.header.bit_depth == 16 {
                    for i in 0..pixel_count {
                        let gray = raw_data[i * 4];
                        let alpha = raw_data[i * 4 + 2];
                        rgba[i * 4] = gray;
                        rgba[i * 4 + 1] = gray;
                        rgba[i * 4 + 2] = gray;
                        rgba[i * 4 + 3] = alpha;
                    }
                }
            }
            ColorType::Rgba => {
                if self.header.bit_depth == 8 {
                    rgba.copy_from_slice(raw_data);
                } else if self.header.bit_depth == 16 {
                    for i in 0..pixel_count {
                        rgba[i * 4] = raw_data[i * 8];
                        rgba[i * 4 + 1] = raw_data[i * 8 + 2];
                        rgba[i * 4 + 2] = raw_data[i * 8 + 4];
                        rgba[i * 4 + 3] = raw_data[i * 8 + 6];
                    }
                }
            }
        }

        // Apply transparency if present
        if let Some(trns) = &self.transparency {
            self.apply_transparency(&mut rgba, trns)?;
        }

        Ok(rgba)
    }

    /// Expand low bit-depth grayscale to 8-bit.
    fn expand_grayscale(&self, raw_data: &[u8], rgba: &mut [u8]) -> CodecResult<()> {
        let pixel_count = (self.header.width * self.header.height) as usize;
        let scale = 255 / ((1 << self.header.bit_depth) - 1);

        let mut bit_pos = 0;
        for i in 0..pixel_count {
            let byte_pos = bit_pos / 8;
            let shift = 8 - (bit_pos % 8) - self.header.bit_depth as usize;
            let mask = (1 << self.header.bit_depth) - 1;

            if byte_pos >= raw_data.len() {
                return Err(CodecError::InvalidData(
                    "Insufficient grayscale data".into(),
                ));
            }

            let value = (raw_data[byte_pos] >> shift) & mask;
            let gray = value * scale;

            rgba[i * 4] = gray;
            rgba[i * 4 + 1] = gray;
            rgba[i * 4 + 2] = gray;
            rgba[i * 4 + 3] = 255;

            bit_pos += self.header.bit_depth as usize;
        }

        Ok(())
    }

    /// Expand palette indices to RGBA.
    fn expand_palette(&self, raw_data: &[u8], palette: &[u8], rgba: &mut [u8]) -> CodecResult<()> {
        let pixel_count = (self.header.width * self.header.height) as usize;
        let mut bit_pos = 0;

        for i in 0..pixel_count {
            let byte_pos = bit_pos / 8;
            let shift = 8 - (bit_pos % 8) - self.header.bit_depth as usize;
            let mask = (1 << self.header.bit_depth) - 1;

            if byte_pos >= raw_data.len() {
                return Err(CodecError::InvalidData("Insufficient palette data".into()));
            }

            let index = ((raw_data[byte_pos] >> shift) & mask) as usize;
            let palette_offset = index * 3;

            if palette_offset + 3 > palette.len() {
                return Err(CodecError::InvalidData(format!(
                    "Invalid palette index: {index}"
                )));
            }

            rgba[i * 4] = palette[palette_offset];
            rgba[i * 4 + 1] = palette[palette_offset + 1];
            rgba[i * 4 + 2] = palette[palette_offset + 2];
            rgba[i * 4 + 3] = 255;

            bit_pos += self.header.bit_depth as usize;
        }

        Ok(())
    }

    /// Apply transparency data.
    fn apply_transparency(&self, rgba: &mut [u8], trns: &[u8]) -> CodecResult<()> {
        match self.header.color_type {
            ColorType::Grayscale => {
                if trns.len() < 2 {
                    return Ok(());
                }
                let transparent_gray = u16::from_be_bytes([trns[0], trns[1]]);
                let transparent_gray_8 = if self.header.bit_depth == 16 {
                    (transparent_gray >> 8) as u8
                } else {
                    transparent_gray as u8
                };

                for i in 0..rgba.len() / 4 {
                    if rgba[i * 4] == transparent_gray_8 {
                        rgba[i * 4 + 3] = 0;
                    }
                }
            }
            ColorType::Palette => {
                for i in 0..rgba.len() / 4 {
                    if i < trns.len() {
                        rgba[i * 4 + 3] = trns[i];
                    }
                }
            }
            ColorType::Rgb => {
                if trns.len() < 6 {
                    return Ok(());
                }
                let tr = u16::from_be_bytes([trns[0], trns[1]]);
                let tg = u16::from_be_bytes([trns[2], trns[3]]);
                let tb = u16::from_be_bytes([trns[4], trns[5]]);

                for i in 0..rgba.len() / 4 {
                    let r = u16::from(rgba[i * 4]);
                    let g = u16::from(rgba[i * 4 + 1]);
                    let b = u16::from(rgba[i * 4 + 2]);

                    if r == tr && g == tg && b == tb {
                        rgba[i * 4 + 3] = 0;
                    }
                }
            }
            _ => {}
        }

        Ok(())
    }
}

/// Decoded PNG image.
#[derive(Debug, Clone)]
pub struct DecodedImage {
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// RGBA pixel data.
    pub data: Bytes,
}

/// Chromaticity coordinates (cHRM chunk).
#[derive(Debug, Clone, Copy)]
pub struct Chromaticity {
    /// White point X.
    pub white_x: f64,
    /// White point Y.
    pub white_y: f64,
    /// Red X.
    pub red_x: f64,
    /// Red Y.
    pub red_y: f64,
    /// Green X.
    pub green_x: f64,
    /// Green Y.
    pub green_y: f64,
    /// Blue X.
    pub blue_x: f64,
    /// Blue Y.
    pub blue_y: f64,
}

/// Physical pixel dimensions (pHYs chunk).
#[derive(Debug, Clone, Copy)]
pub struct PhysicalDimensions {
    /// Pixels per unit, X axis.
    pub x: u32,
    /// Pixels per unit, Y axis.
    pub y: u32,
    /// Unit specifier (0 = unknown, 1 = meter).
    pub unit: u8,
}

/// Significant bits (sBIT chunk).
#[derive(Debug, Clone)]
pub struct SignificantBits {
    /// Significant bits for each channel.
    pub bits: Vec<u8>,
}

/// Text metadata (tEXt chunk).
#[derive(Debug, Clone)]
pub struct TextChunk {
    /// Keyword.
    pub keyword: String,
    /// Text value.
    pub text: String,
}

/// PNG metadata.
#[derive(Debug, Clone, Default)]
pub struct PngMetadata {
    /// Gamma value.
    pub gamma: Option<f64>,
    /// Chromaticity coordinates.
    pub chromaticity: Option<Chromaticity>,
    /// Physical dimensions.
    pub physical_dimensions: Option<PhysicalDimensions>,
    /// Significant bits.
    pub significant_bits: Option<SignificantBits>,
    /// Text chunks.
    pub text_chunks: Vec<TextChunk>,
    /// Background color.
    pub background_color: Option<(u16, u16, u16)>,
    /// Suggested palette.
    pub suggested_palette: Option<Vec<u8>>,
}

impl Chromaticity {
    /// Parse from cHRM chunk data.
    ///
    /// # Errors
    ///
    /// Returns error if chunk data is invalid.
    pub fn parse(data: &[u8]) -> CodecResult<Self> {
        if data.len() < 32 {
            return Err(CodecError::InvalidData("cHRM chunk too short".into()));
        }

        let white_x = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        let white_y = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
        let red_x = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);
        let red_y = u32::from_be_bytes([data[12], data[13], data[14], data[15]]);
        let green_x = u32::from_be_bytes([data[16], data[17], data[18], data[19]]);
        let green_y = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);
        let blue_x = u32::from_be_bytes([data[24], data[25], data[26], data[27]]);
        let blue_y = u32::from_be_bytes([data[28], data[29], data[30], data[31]]);

        Ok(Self {
            white_x: f64::from(white_x) / 100_000.0,
            white_y: f64::from(white_y) / 100_000.0,
            red_x: f64::from(red_x) / 100_000.0,
            red_y: f64::from(red_y) / 100_000.0,
            green_x: f64::from(green_x) / 100_000.0,
            green_y: f64::from(green_y) / 100_000.0,
            blue_x: f64::from(blue_x) / 100_000.0,
            blue_y: f64::from(blue_y) / 100_000.0,
        })
    }

    /// Get sRGB chromaticity.
    #[must_use]
    pub fn srgb() -> Self {
        Self {
            white_x: 0.3127,
            white_y: 0.329,
            red_x: 0.64,
            red_y: 0.33,
            green_x: 0.3,
            green_y: 0.6,
            blue_x: 0.15,
            blue_y: 0.06,
        }
    }
}

impl PhysicalDimensions {
    /// Parse from pHYs chunk data.
    ///
    /// # Errors
    ///
    /// Returns error if chunk data is invalid.
    pub fn parse(data: &[u8]) -> CodecResult<Self> {
        if data.len() < 9 {
            return Err(CodecError::InvalidData("pHYs chunk too short".into()));
        }

        let x = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        let y = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
        let unit = data[8];

        Ok(Self { x, y, unit })
    }

    /// Get DPI if unit is meter.
    #[must_use]
    pub fn dpi(&self) -> Option<(f64, f64)> {
        if self.unit == 1 {
            const METERS_PER_INCH: f64 = 0.0254;
            Some((
                f64::from(self.x) * METERS_PER_INCH,
                f64::from(self.y) * METERS_PER_INCH,
            ))
        } else {
            None
        }
    }
}

impl SignificantBits {
    /// Parse from sBIT chunk data.
    ///
    /// # Errors
    ///
    /// Returns error if chunk data is invalid.
    pub fn parse(data: &[u8]) -> CodecResult<Self> {
        if data.is_empty() {
            return Err(CodecError::InvalidData("sBIT chunk is empty".into()));
        }

        Ok(Self {
            bits: data.to_vec(),
        })
    }
}

impl TextChunk {
    /// Parse from tEXt chunk data.
    ///
    /// # Errors
    ///
    /// Returns error if chunk data is invalid.
    pub fn parse(data: &[u8]) -> CodecResult<Self> {
        // Find null separator
        let null_pos = data
            .iter()
            .position(|&b| b == 0)
            .ok_or_else(|| CodecError::InvalidData("tEXt chunk missing null separator".into()))?;

        let keyword = String::from_utf8_lossy(&data[..null_pos]).into_owned();
        let text = String::from_utf8_lossy(&data[null_pos + 1..]).into_owned();

        Ok(Self { keyword, text })
    }
}

/// Extended PNG decoder with metadata support.
pub struct PngDecoderExtended {
    /// Base decoder.
    pub decoder: PngDecoder,
    /// Metadata.
    pub metadata: PngMetadata,
}

impl PngDecoderExtended {
    /// Create a new extended PNG decoder.
    ///
    /// # Errors
    ///
    /// Returns error if PNG data is invalid.
    #[allow(clippy::too_many_lines)]
    pub fn new(data: &[u8]) -> CodecResult<Self> {
        if data.len() < 8 {
            return Err(CodecError::InvalidData("PNG data too short".into()));
        }

        if &data[0..8] != PNG_SIGNATURE {
            return Err(CodecError::InvalidData("Invalid PNG signature".into()));
        }

        let mut offset = 8;
        let mut header: Option<ImageHeader> = None;
        let mut palette: Option<Vec<u8>> = None;
        let mut transparency: Option<Vec<u8>> = None;
        let mut metadata = PngMetadata::default();
        let mut idat_chunks = Vec::new();

        while offset < data.len() {
            let chunk = Chunk::read(data, &mut offset)?;

            match chunk.type_str() {
                "IHDR" => {
                    if header.is_some() {
                        return Err(CodecError::InvalidData("Multiple IHDR chunks".into()));
                    }
                    header = Some(ImageHeader::parse(&chunk.data)?);
                }
                "PLTE" => {
                    if palette.is_some() {
                        return Err(CodecError::InvalidData("Multiple PLTE chunks".into()));
                    }
                    if chunk.data.len() % 3 != 0 || chunk.data.is_empty() {
                        return Err(CodecError::InvalidData("Invalid PLTE chunk".into()));
                    }
                    palette = Some(chunk.data);
                }
                "IDAT" => {
                    idat_chunks.push(chunk.data);
                }
                "IEND" => {
                    break;
                }
                "tRNS" => {
                    transparency = Some(chunk.data);
                }
                "gAMA" => {
                    if chunk.data.len() == 4 {
                        let gamma_int = u32::from_be_bytes([
                            chunk.data[0],
                            chunk.data[1],
                            chunk.data[2],
                            chunk.data[3],
                        ]);
                        metadata.gamma = Some(f64::from(gamma_int) / 100_000.0);
                    }
                }
                "cHRM" => {
                    metadata.chromaticity = Some(Chromaticity::parse(&chunk.data)?);
                }
                "pHYs" => {
                    metadata.physical_dimensions = Some(PhysicalDimensions::parse(&chunk.data)?);
                }
                "sBIT" => {
                    metadata.significant_bits = Some(SignificantBits::parse(&chunk.data)?);
                }
                "tEXt" => {
                    if let Ok(text_chunk) = TextChunk::parse(&chunk.data) {
                        metadata.text_chunks.push(text_chunk);
                    }
                }
                "bKGD" => {
                    if chunk.data.len() >= 6 {
                        let r = u16::from_be_bytes([chunk.data[0], chunk.data[1]]);
                        let g = u16::from_be_bytes([chunk.data[2], chunk.data[3]]);
                        let b = u16::from_be_bytes([chunk.data[4], chunk.data[5]]);
                        metadata.background_color = Some((r, g, b));
                    }
                }
                "sPLT" => {
                    metadata.suggested_palette = Some(chunk.data);
                }
                _ => {
                    // Skip unknown ancillary chunks
                }
            }
        }

        let header = header.ok_or_else(|| CodecError::InvalidData("Missing IHDR chunk".into()))?;

        if header.color_type == ColorType::Palette && palette.is_none() {
            return Err(CodecError::InvalidData(
                "Palette color type requires PLTE chunk".into(),
            ));
        }

        let compressed_data: Vec<u8> = idat_chunks.into_iter().flatten().collect();
        let mut zlib_decoder = ZlibStreamDecoder::new(&compressed_data[..]);
        let mut image_data = Vec::new();
        zlib_decoder
            .read_to_end(&mut image_data)
            .map_err(|e| CodecError::DecoderError(format!("DEFLATE decompression failed: {e}")))?;

        let decoder = PngDecoder {
            header,
            palette,
            transparency,
            gamma: metadata.gamma,
            image_data,
        };

        Ok(Self { decoder, metadata })
    }

    /// Decode the PNG image.
    ///
    /// # Errors
    ///
    /// Returns error if decoding fails.
    pub fn decode(&self) -> CodecResult<DecodedImage> {
        self.decoder.decode()
    }
}
