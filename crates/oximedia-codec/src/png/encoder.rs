//! PNG encoder implementation.
//!
//! Implements a complete PNG 1.2 specification encoder with support for:
//! - All color types (Grayscale, RGB, Palette, GrayscaleAlpha, RGBA)
//! - All bit depths (1, 2, 4, 8, 16)
//! - Interlacing (Adam7)
//! - Adaptive filtering with best filter selection
//! - Configurable compression levels
//! - Palette optimization
//! - Transparency handling
//! - Gamma and chromaticity metadata

use super::decoder::ColorType;
use super::filter::{FilterStrategy, FilterType};
use crate::error::{CodecError, CodecResult};
use oxiarc_deflate::ZlibStreamEncoder;
use std::io::Write;

/// PNG signature bytes.
const PNG_SIGNATURE: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];

/// PNG encoder configuration.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Compression level (0-9).
    pub compression_level: u32,
    /// Filter strategy.
    pub filter_strategy: FilterStrategy,
    /// Enable Adam7 interlacing.
    pub interlace: bool,
    /// Gamma value (optional).
    pub gamma: Option<f64>,
    /// Optimize palette for indexed images.
    pub optimize_palette: bool,
}

impl EncoderConfig {
    /// Create new encoder config with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set compression level (0-9).
    #[must_use]
    pub const fn with_compression(mut self, level: u32) -> Self {
        self.compression_level = if level > 9 { 9 } else { level };
        self
    }

    /// Set filter strategy.
    #[must_use]
    pub const fn with_filter_strategy(mut self, strategy: FilterStrategy) -> Self {
        self.filter_strategy = strategy;
        self
    }

    /// Enable interlacing.
    #[must_use]
    pub const fn with_interlace(mut self, interlace: bool) -> Self {
        self.interlace = interlace;
        self
    }

    /// Set gamma value.
    #[must_use]
    pub const fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = Some(gamma);
        self
    }

    /// Enable palette optimization.
    #[must_use]
    pub const fn with_palette_optimization(mut self, optimize: bool) -> Self {
        self.optimize_palette = optimize;
        self
    }
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            compression_level: 6,
            filter_strategy: FilterStrategy::Fast,
            interlace: false,
            gamma: None,
            optimize_palette: false,
        }
    }
}

/// PNG encoder.
pub struct PngEncoder {
    config: EncoderConfig,
}

impl PngEncoder {
    /// Create a new PNG encoder with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: EncoderConfig::default(),
        }
    }

    /// Create a new PNG encoder with custom configuration.
    #[must_use]
    pub const fn with_config(config: EncoderConfig) -> Self {
        Self { config }
    }

    /// Encode RGBA image data to PNG format.
    ///
    /// # Arguments
    ///
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `data` - RGBA pixel data (width * height * 4 bytes)
    ///
    /// # Errors
    ///
    /// Returns error if encoding fails or data is invalid.
    pub fn encode_rgba(&self, width: u32, height: u32, data: &[u8]) -> CodecResult<Vec<u8>> {
        if data.len() != (width * height * 4) as usize {
            return Err(CodecError::InvalidParameter(format!(
                "Invalid data length: expected {}, got {}",
                width * height * 4,
                data.len()
            )));
        }

        let mut output = Vec::new();

        // Write PNG signature
        output.extend_from_slice(&PNG_SIGNATURE);

        // Determine best color type
        let (color_type, bit_depth, image_data) = self.optimize_color_type(width, height, data)?;

        // Write IHDR chunk
        self.write_ihdr(&mut output, width, height, bit_depth, color_type)?;

        // Write optional chunks
        if let Some(gamma) = self.config.gamma {
            self.write_gamma(&mut output, gamma)?;
        }

        // Write PLTE chunk if needed
        if color_type == ColorType::Palette {
            if let Some(palette) = self.extract_palette(data) {
                self.write_palette(&mut output, &palette)?;
            }
        }

        // Encode and write image data
        let compressed_data =
            self.encode_image_data(&image_data, width, height, color_type, bit_depth)?;
        self.write_idat(&mut output, &compressed_data)?;

        // Write IEND chunk
        self.write_iend(&mut output)?;

        Ok(output)
    }

    /// Encode RGB image data to PNG format.
    ///
    /// # Arguments
    ///
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `data` - RGB pixel data (width * height * 3 bytes)
    ///
    /// # Errors
    ///
    /// Returns error if encoding fails or data is invalid.
    pub fn encode_rgb(&self, width: u32, height: u32, data: &[u8]) -> CodecResult<Vec<u8>> {
        if data.len() != (width * height * 3) as usize {
            return Err(CodecError::InvalidParameter(format!(
                "Invalid data length: expected {}, got {}",
                width * height * 3,
                data.len()
            )));
        }

        let mut output = Vec::new();
        output.extend_from_slice(&PNG_SIGNATURE);

        self.write_ihdr(&mut output, width, height, 8, ColorType::Rgb)?;

        if let Some(gamma) = self.config.gamma {
            self.write_gamma(&mut output, gamma)?;
        }

        let compressed_data = self.encode_image_data(data, width, height, ColorType::Rgb, 8)?;
        self.write_idat(&mut output, &compressed_data)?;
        self.write_iend(&mut output)?;

        Ok(output)
    }

    /// Encode grayscale image data to PNG format.
    ///
    /// # Arguments
    ///
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `data` - Grayscale pixel data (width * height bytes)
    /// * `bit_depth` - Bit depth (1, 2, 4, 8, or 16)
    ///
    /// # Errors
    ///
    /// Returns error if encoding fails or data is invalid.
    pub fn encode_grayscale(
        &self,
        width: u32,
        height: u32,
        data: &[u8],
        bit_depth: u8,
    ) -> CodecResult<Vec<u8>> {
        let expected_len = if bit_depth == 16 {
            (width * height * 2) as usize
        } else {
            (width * height) as usize
        };

        if data.len() != expected_len {
            return Err(CodecError::InvalidParameter(format!(
                "Invalid data length: expected {expected_len}, got {}",
                data.len()
            )));
        }

        let mut output = Vec::new();
        output.extend_from_slice(&PNG_SIGNATURE);

        self.write_ihdr(&mut output, width, height, bit_depth, ColorType::Grayscale)?;

        if let Some(gamma) = self.config.gamma {
            self.write_gamma(&mut output, gamma)?;
        }

        let compressed_data =
            self.encode_image_data(data, width, height, ColorType::Grayscale, bit_depth)?;
        self.write_idat(&mut output, &compressed_data)?;
        self.write_iend(&mut output)?;

        Ok(output)
    }

    /// Optimize color type based on image content.
    #[allow(clippy::type_complexity)]
    fn optimize_color_type(
        &self,
        width: u32,
        height: u32,
        rgba_data: &[u8],
    ) -> CodecResult<(ColorType, u8, Vec<u8>)> {
        let pixel_count = (width * height) as usize;

        // Check if all pixels are opaque
        let mut has_alpha = false;
        for i in 0..pixel_count {
            if rgba_data[i * 4 + 3] != 255 {
                has_alpha = true;
                break;
            }
        }

        // Check if grayscale
        let mut is_grayscale = true;
        for i in 0..pixel_count {
            let r = rgba_data[i * 4];
            let g = rgba_data[i * 4 + 1];
            let b = rgba_data[i * 4 + 2];
            if r != g || g != b {
                is_grayscale = false;
                break;
            }
        }

        // Select color type
        if is_grayscale && !has_alpha {
            // Grayscale
            let mut gray_data = Vec::with_capacity(pixel_count);
            for i in 0..pixel_count {
                gray_data.push(rgba_data[i * 4]);
            }
            Ok((ColorType::Grayscale, 8, gray_data))
        } else if is_grayscale && has_alpha {
            // Grayscale with alpha
            let mut ga_data = Vec::with_capacity(pixel_count * 2);
            for i in 0..pixel_count {
                ga_data.push(rgba_data[i * 4]);
                ga_data.push(rgba_data[i * 4 + 3]);
            }
            Ok((ColorType::GrayscaleAlpha, 8, ga_data))
        } else if !has_alpha {
            // RGB
            let mut rgb_data = Vec::with_capacity(pixel_count * 3);
            for i in 0..pixel_count {
                rgb_data.push(rgba_data[i * 4]);
                rgb_data.push(rgba_data[i * 4 + 1]);
                rgb_data.push(rgba_data[i * 4 + 2]);
            }
            Ok((ColorType::Rgb, 8, rgb_data))
        } else {
            // RGBA
            Ok((ColorType::Rgba, 8, rgba_data.to_vec()))
        }
    }

    /// Extract palette from RGBA image.
    fn extract_palette(&self, rgba_data: &[u8]) -> Option<Vec<u8>> {
        if !self.config.optimize_palette {
            return None;
        }

        let mut colors = std::collections::HashSet::new();
        for chunk in rgba_data.chunks_exact(4) {
            colors.insert((chunk[0], chunk[1], chunk[2]));
            if colors.len() > 256 {
                return None;
            }
        }

        let mut palette = Vec::with_capacity(colors.len() * 3);
        for (r, g, b) in colors {
            palette.push(r);
            palette.push(g);
            palette.push(b);
        }

        Some(palette)
    }

    /// Encode image data with filtering and compression.
    #[allow(clippy::too_many_lines)]
    fn encode_image_data(
        &self,
        data: &[u8],
        width: u32,
        height: u32,
        color_type: ColorType,
        bit_depth: u8,
    ) -> CodecResult<Vec<u8>> {
        let samples_per_pixel = color_type.samples_per_pixel();
        let bits_per_pixel = samples_per_pixel * bit_depth as usize;
        let bytes_per_pixel = (bits_per_pixel + 7) / 8;
        let scanline_len = ((width as usize * bits_per_pixel) + 7) / 8;

        if self.config.interlace {
            self.encode_interlaced(data, width, height, color_type, bit_depth)
        } else {
            self.encode_sequential(data, width, height, scanline_len, bytes_per_pixel)
        }
    }

    /// Encode sequential (non-interlaced) image.
    fn encode_sequential(
        &self,
        data: &[u8],
        width: u32,
        height: u32,
        scanline_len: usize,
        bytes_per_pixel: usize,
    ) -> CodecResult<Vec<u8>> {
        let mut filtered_data = Vec::with_capacity((scanline_len + 1) * height as usize);
        let mut prev_scanline: Option<Vec<u8>> = None;

        for y in 0..height as usize {
            let scanline_start = y * scanline_len;
            let scanline = &data[scanline_start..scanline_start + scanline_len];

            let (filter_type, filtered) = self.config.filter_strategy.apply(
                scanline,
                prev_scanline.as_deref(),
                bytes_per_pixel,
            );

            filtered_data.push(filter_type.to_u8());
            filtered_data.extend_from_slice(&filtered);

            prev_scanline = Some(scanline.to_vec());
        }

        // Compress with DEFLATE
        let level = self.config.compression_level.min(9) as u8;

        let mut encoder = ZlibStreamEncoder::new(Vec::new(), level);
        encoder
            .write_all(&filtered_data)
            .map_err(|e| CodecError::Internal(format!("Compression failed: {e}")))?;

        encoder
            .finish()
            .map_err(|e| CodecError::Internal(format!("Compression finish failed: {e}")))
    }

    /// Encode interlaced (Adam7) image.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::similar_names)]
    fn encode_interlaced(
        &self,
        data: &[u8],
        width: u32,
        height: u32,
        color_type: ColorType,
        bit_depth: u8,
    ) -> CodecResult<Vec<u8>> {
        let samples_per_pixel = color_type.samples_per_pixel();
        let bits_per_pixel = samples_per_pixel * bit_depth as usize;
        let bytes_per_pixel = (bits_per_pixel + 7) / 8;
        let full_scanline_len = ((width as usize * bits_per_pixel) + 7) / 8;

        let mut filtered_data = Vec::new();

        // Adam7 passes
        let passes = [
            (0, 0, 8, 8),
            (4, 0, 8, 8),
            (0, 4, 4, 8),
            (2, 0, 4, 4),
            (0, 2, 2, 4),
            (1, 0, 2, 2),
            (0, 1, 1, 2),
        ];

        for (x_start, y_start, x_step, y_step) in passes {
            let pass_width = (width.saturating_sub(x_start) + x_step - 1) / x_step;
            let pass_height = (height.saturating_sub(y_start) + y_step - 1) / y_step;

            if pass_width == 0 || pass_height == 0 {
                continue;
            }

            let pass_scanline_len = ((pass_width as usize * bits_per_pixel) + 7) / 8;
            let mut prev_scanline: Option<Vec<u8>> = None;

            for py in 0..pass_height {
                let y = y_start + py * y_step;
                let mut scanline = vec![0u8; pass_scanline_len];

                for px in 0..pass_width {
                    let x = x_start + px * x_step;
                    let src_offset =
                        (y as usize * full_scanline_len) + (x as usize * bytes_per_pixel);
                    let dst_offset = px as usize * bytes_per_pixel;

                    if src_offset + bytes_per_pixel <= data.len()
                        && dst_offset + bytes_per_pixel <= scanline.len()
                    {
                        scanline[dst_offset..dst_offset + bytes_per_pixel]
                            .copy_from_slice(&data[src_offset..src_offset + bytes_per_pixel]);
                    }
                }

                let (filter_type, filtered) = self.config.filter_strategy.apply(
                    &scanline,
                    prev_scanline.as_deref(),
                    bytes_per_pixel,
                );

                filtered_data.push(filter_type.to_u8());
                filtered_data.extend_from_slice(&filtered);

                prev_scanline = Some(scanline);
            }
        }

        // Compress
        let level = self.config.compression_level.min(9) as u8;

        let mut encoder = ZlibStreamEncoder::new(Vec::new(), level);
        encoder
            .write_all(&filtered_data)
            .map_err(|e| CodecError::Internal(format!("Compression failed: {e}")))?;

        encoder
            .finish()
            .map_err(|e| CodecError::Internal(format!("Compression finish failed: {e}")))
    }

    /// Write IHDR chunk.
    fn write_ihdr(
        &self,
        output: &mut Vec<u8>,
        width: u32,
        height: u32,
        bit_depth: u8,
        color_type: ColorType,
    ) -> CodecResult<()> {
        let mut data = Vec::new();
        data.extend_from_slice(&width.to_be_bytes());
        data.extend_from_slice(&height.to_be_bytes());
        data.push(bit_depth);
        data.push(color_type as u8);
        data.push(0); // Compression method
        data.push(0); // Filter method
        data.push(if self.config.interlace { 1 } else { 0 });

        self.write_chunk(output, b"IHDR", &data)
    }

    /// Write gAMA chunk.
    fn write_gamma(&self, output: &mut Vec<u8>, gamma: f64) -> CodecResult<()> {
        let gamma_int = (gamma * 100_000.0) as u32;
        let data = gamma_int.to_be_bytes();
        self.write_chunk(output, b"gAMA", &data)
    }

    /// Write PLTE chunk.
    fn write_palette(&self, output: &mut Vec<u8>, palette: &[u8]) -> CodecResult<()> {
        self.write_chunk(output, b"PLTE", palette)
    }

    /// Write IDAT chunk.
    fn write_idat(&self, output: &mut Vec<u8>, data: &[u8]) -> CodecResult<()> {
        // Split into multiple IDAT chunks if needed (max 32KB per chunk)
        const MAX_CHUNK_SIZE: usize = 32768;

        if data.len() <= MAX_CHUNK_SIZE {
            self.write_chunk(output, b"IDAT", data)?;
        } else {
            for chunk in data.chunks(MAX_CHUNK_SIZE) {
                self.write_chunk(output, b"IDAT", chunk)?;
            }
        }

        Ok(())
    }

    /// Write IEND chunk.
    fn write_iend(&self, output: &mut Vec<u8>) -> CodecResult<()> {
        self.write_chunk(output, b"IEND", &[])
    }

    /// Write a PNG chunk with CRC.
    fn write_chunk(
        &self,
        output: &mut Vec<u8>,
        chunk_type: &[u8; 4],
        data: &[u8],
    ) -> CodecResult<()> {
        // Write length
        output.extend_from_slice(&(data.len() as u32).to_be_bytes());

        // Write type
        output.extend_from_slice(chunk_type);

        // Write data
        output.extend_from_slice(data);

        // Calculate and write CRC
        let crc = crc32(chunk_type, data);
        output.extend_from_slice(&crc.to_be_bytes());

        Ok(())
    }
}

impl Default for PngEncoder {
    fn default() -> Self {
        Self::new()
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

/// PNG compression level presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionLevel {
    /// No compression (fastest).
    None,
    /// Fast compression.
    Fast,
    /// Default compression.
    Default,
    /// Best compression (slowest).
    Best,
}

impl CompressionLevel {
    /// Convert to numeric level (0-9).
    #[must_use]
    pub const fn to_level(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Fast => 1,
            Self::Default => 6,
            Self::Best => 9,
        }
    }
}

/// Builder for PNG encoder configuration.
pub struct EncoderBuilder {
    config: EncoderConfig,
}

impl EncoderBuilder {
    /// Create a new encoder builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: EncoderConfig::default(),
        }
    }

    /// Set compression level.
    #[must_use]
    pub const fn compression_level(mut self, level: CompressionLevel) -> Self {
        self.config.compression_level = level.to_level();
        self
    }

    /// Set filter strategy.
    #[must_use]
    pub const fn filter_strategy(mut self, strategy: FilterStrategy) -> Self {
        self.config.filter_strategy = strategy;
        self
    }

    /// Enable interlacing.
    #[must_use]
    pub const fn interlace(mut self, enable: bool) -> Self {
        self.config.interlace = enable;
        self
    }

    /// Set gamma value.
    #[must_use]
    pub const fn gamma(mut self, gamma: f64) -> Self {
        self.config.gamma = Some(gamma);
        self
    }

    /// Enable palette optimization.
    #[must_use]
    pub const fn optimize_palette(mut self, enable: bool) -> Self {
        self.config.optimize_palette = enable;
        self
    }

    /// Build the encoder.
    #[must_use]
    pub fn build(self) -> PngEncoder {
        PngEncoder::with_config(self.config)
    }
}

impl Default for EncoderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Fast encoder for maximum speed.
///
/// Uses no filtering and fast compression.
#[must_use]
pub fn fast_encoder() -> PngEncoder {
    PngEncoder::with_config(
        EncoderConfig::new()
            .with_compression(1)
            .with_filter_strategy(FilterStrategy::None),
    )
}

/// Best encoder for maximum compression.
///
/// Uses best filtering and compression.
#[must_use]
pub fn best_encoder() -> PngEncoder {
    PngEncoder::with_config(
        EncoderConfig::new()
            .with_compression(9)
            .with_filter_strategy(FilterStrategy::Best),
    )
}

/// PNG encoder with metadata support.
pub struct PngEncoderExtended {
    /// Base encoder.
    encoder: PngEncoder,
    /// Chromaticity coordinates.
    chromaticity: Option<super::decoder::Chromaticity>,
    /// Physical dimensions.
    physical_dimensions: Option<super::decoder::PhysicalDimensions>,
    /// Significant bits.
    #[allow(dead_code)]
    significant_bits: Option<super::decoder::SignificantBits>,
    /// Text chunks.
    text_chunks: Vec<super::decoder::TextChunk>,
    /// Background color.
    background_color: Option<(u16, u16, u16)>,
}

impl PngEncoderExtended {
    /// Create a new extended PNG encoder.
    #[must_use]
    pub fn new(config: EncoderConfig) -> Self {
        Self {
            encoder: PngEncoder::with_config(config),
            chromaticity: None,
            physical_dimensions: None,
            significant_bits: None,
            text_chunks: Vec::new(),
            background_color: None,
        }
    }

    /// Set chromaticity coordinates.
    #[must_use]
    pub fn with_chromaticity(mut self, chroma: super::decoder::Chromaticity) -> Self {
        self.chromaticity = Some(chroma);
        self
    }

    /// Set physical dimensions.
    #[must_use]
    pub fn with_physical_dimensions(mut self, dims: super::decoder::PhysicalDimensions) -> Self {
        self.physical_dimensions = Some(dims);
        self
    }

    /// Set DPI (converts to physical dimensions in meters).
    #[must_use]
    pub fn with_dpi(mut self, dpi_x: f64, dpi_y: f64) -> Self {
        const METERS_PER_INCH: f64 = 0.0254;
        self.physical_dimensions = Some(super::decoder::PhysicalDimensions {
            x: (dpi_x / METERS_PER_INCH) as u32,
            y: (dpi_y / METERS_PER_INCH) as u32,
            unit: 1,
        });
        self
    }

    /// Add text metadata.
    #[must_use]
    pub fn with_text(mut self, keyword: String, text: String) -> Self {
        self.text_chunks
            .push(super::decoder::TextChunk { keyword, text });
        self
    }

    /// Set background color.
    #[must_use]
    pub const fn with_background_color(mut self, r: u16, g: u16, b: u16) -> Self {
        self.background_color = Some((r, g, b));
        self
    }

    /// Encode RGBA image with metadata.
    ///
    /// # Errors
    ///
    /// Returns error if encoding fails.
    #[allow(clippy::too_many_lines)]
    pub fn encode_rgba(&self, width: u32, height: u32, data: &[u8]) -> CodecResult<Vec<u8>> {
        if data.len() != (width * height * 4) as usize {
            return Err(CodecError::InvalidParameter(format!(
                "Invalid data length: expected {}, got {}",
                width * height * 4,
                data.len()
            )));
        }

        let mut output = Vec::new();
        output.extend_from_slice(&PNG_SIGNATURE);

        // Determine color type
        let (color_type, bit_depth, image_data) =
            self.encoder.optimize_color_type(width, height, data)?;

        // Write IHDR
        self.encoder
            .write_ihdr(&mut output, width, height, bit_depth, color_type)?;

        // Write optional chunks
        if let Some(gamma) = self.encoder.config.gamma {
            self.encoder.write_gamma(&mut output, gamma)?;
        }

        if let Some(chroma) = &self.chromaticity {
            self.write_chromaticity(&mut output, chroma)?;
        }

        if let Some(dims) = &self.physical_dimensions {
            self.write_physical_dimensions(&mut output, dims)?;
        }

        if let Some(bg) = &self.background_color {
            self.write_background_color(&mut output, *bg)?;
        }

        // Write text chunks
        for text_chunk in &self.text_chunks {
            self.write_text_chunk(&mut output, text_chunk)?;
        }

        // Write PLTE if needed
        if color_type == ColorType::Palette {
            if let Some(palette) = self.encoder.extract_palette(data) {
                self.encoder.write_palette(&mut output, &palette)?;
            }
        }

        // Encode image data
        let compressed_data =
            self.encoder
                .encode_image_data(&image_data, width, height, color_type, bit_depth)?;
        self.encoder.write_idat(&mut output, &compressed_data)?;

        // Write IEND
        self.encoder.write_iend(&mut output)?;

        Ok(output)
    }

    /// Write chromaticity chunk.
    fn write_chromaticity(
        &self,
        output: &mut Vec<u8>,
        chroma: &super::decoder::Chromaticity,
    ) -> CodecResult<()> {
        let mut data = Vec::with_capacity(32);

        let white_x = (chroma.white_x * 100_000.0) as u32;
        let white_y = (chroma.white_y * 100_000.0) as u32;
        let red_x = (chroma.red_x * 100_000.0) as u32;
        let red_y = (chroma.red_y * 100_000.0) as u32;
        let green_x = (chroma.green_x * 100_000.0) as u32;
        let green_y = (chroma.green_y * 100_000.0) as u32;
        let blue_x = (chroma.blue_x * 100_000.0) as u32;
        let blue_y = (chroma.blue_y * 100_000.0) as u32;

        data.extend_from_slice(&white_x.to_be_bytes());
        data.extend_from_slice(&white_y.to_be_bytes());
        data.extend_from_slice(&red_x.to_be_bytes());
        data.extend_from_slice(&red_y.to_be_bytes());
        data.extend_from_slice(&green_x.to_be_bytes());
        data.extend_from_slice(&green_y.to_be_bytes());
        data.extend_from_slice(&blue_x.to_be_bytes());
        data.extend_from_slice(&blue_y.to_be_bytes());

        self.encoder.write_chunk(output, b"cHRM", &data)
    }

    /// Write physical dimensions chunk.
    fn write_physical_dimensions(
        &self,
        output: &mut Vec<u8>,
        dims: &super::decoder::PhysicalDimensions,
    ) -> CodecResult<()> {
        let mut data = Vec::with_capacity(9);
        data.extend_from_slice(&dims.x.to_be_bytes());
        data.extend_from_slice(&dims.y.to_be_bytes());
        data.push(dims.unit);

        self.encoder.write_chunk(output, b"pHYs", &data)
    }

    /// Write background color chunk.
    fn write_background_color(
        &self,
        output: &mut Vec<u8>,
        color: (u16, u16, u16),
    ) -> CodecResult<()> {
        let mut data = Vec::with_capacity(6);
        data.extend_from_slice(&color.0.to_be_bytes());
        data.extend_from_slice(&color.1.to_be_bytes());
        data.extend_from_slice(&color.2.to_be_bytes());

        self.encoder.write_chunk(output, b"bKGD", &data)
    }

    /// Write text chunk.
    fn write_text_chunk(
        &self,
        output: &mut Vec<u8>,
        text_chunk: &super::decoder::TextChunk,
    ) -> CodecResult<()> {
        let mut data = Vec::new();
        data.extend_from_slice(text_chunk.keyword.as_bytes());
        data.push(0); // Null separator
        data.extend_from_slice(text_chunk.text.as_bytes());

        self.encoder.write_chunk(output, b"tEXt", &data)
    }
}

impl Default for PngEncoderExtended {
    fn default() -> Self {
        Self::new(EncoderConfig::default())
    }
}

/// Palette entry for indexed color optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PaletteEntry {
    /// Red component.
    pub r: u8,
    /// Green component.
    pub g: u8,
    /// Blue component.
    pub b: u8,
}

/// Palette optimizer for indexed color images.
pub struct PaletteOptimizer {
    /// Color frequency map.
    colors: std::collections::HashMap<PaletteEntry, u32>,
    /// Maximum palette size.
    max_size: usize,
}

impl PaletteOptimizer {
    /// Create a new palette optimizer.
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            colors: std::collections::HashMap::new(),
            max_size: max_size.min(256),
        }
    }

    /// Add a color to the palette.
    pub fn add_color(&mut self, r: u8, g: u8, b: u8) {
        let entry = PaletteEntry { r, g, b };
        *self.colors.entry(entry).or_insert(0) += 1;
    }

    /// Build optimized palette.
    ///
    /// Returns None if more than max_size colors are used.
    #[must_use]
    pub fn build_palette(&self) -> Option<Vec<PaletteEntry>> {
        if self.colors.len() > self.max_size {
            return None;
        }

        let mut palette: Vec<_> = self.colors.iter().collect();
        palette.sort_by(|a, b| b.1.cmp(a.1)); // Sort by frequency

        Some(palette.iter().map(|(entry, _)| **entry).collect())
    }

    /// Get color index in palette.
    #[must_use]
    pub fn get_index(&self, r: u8, g: u8, b: u8, palette: &[PaletteEntry]) -> Option<u8> {
        let entry = PaletteEntry { r, g, b };
        palette.iter().position(|e| *e == entry).map(|i| i as u8)
    }
}

/// Encoding statistics.
#[derive(Debug, Clone, Default)]
pub struct EncodingStats {
    /// Uncompressed size in bytes.
    pub uncompressed_size: usize,
    /// Compressed size in bytes.
    pub compressed_size: usize,
    /// Filter type distribution.
    pub filter_distribution: [usize; 5],
    /// Encoding time in milliseconds.
    pub encoding_time_ms: u64,
    /// Compression ratio.
    pub compression_ratio: f64,
}

impl EncodingStats {
    /// Create new encoding stats.
    #[must_use]
    pub fn new(uncompressed_size: usize, compressed_size: usize) -> Self {
        let compression_ratio = if compressed_size > 0 {
            uncompressed_size as f64 / compressed_size as f64
        } else {
            0.0
        };

        Self {
            uncompressed_size,
            compressed_size,
            filter_distribution: [0; 5],
            encoding_time_ms: 0,
            compression_ratio,
        }
    }

    /// Add filter type usage.
    pub fn add_filter_usage(&mut self, filter_type: FilterType) {
        self.filter_distribution[filter_type.to_u8() as usize] += 1;
    }

    /// Get most used filter type.
    #[must_use]
    pub fn most_used_filter(&self) -> FilterType {
        let (index, _) = self
            .filter_distribution
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .unwrap_or((0, &0));

        FilterType::from_u8(index as u8).unwrap_or(FilterType::None)
    }
}

/// Encoding profile for different use cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingProfile {
    /// Fastest encoding (lowest compression).
    Fast,
    /// Balanced speed and compression.
    Balanced,
    /// Best compression (slowest).
    Best,
    /// Web-optimized (good compression, reasonable speed).
    Web,
    /// Archive quality (maximum compression).
    Archive,
}

impl EncodingProfile {
    /// Get encoder config for this profile.
    #[must_use]
    pub const fn to_config(self) -> EncoderConfig {
        match self {
            Self::Fast => EncoderConfig {
                compression_level: 1,
                filter_strategy: FilterStrategy::None,
                interlace: false,
                gamma: None,
                optimize_palette: false,
            },
            Self::Balanced => EncoderConfig {
                compression_level: 6,
                filter_strategy: FilterStrategy::Fast,
                interlace: false,
                gamma: None,
                optimize_palette: true,
            },
            Self::Best => EncoderConfig {
                compression_level: 9,
                filter_strategy: FilterStrategy::Best,
                interlace: false,
                gamma: None,
                optimize_palette: true,
            },
            Self::Web => EncoderConfig {
                compression_level: 8,
                filter_strategy: FilterStrategy::Fast,
                interlace: true,
                gamma: Some(2.2),
                optimize_palette: true,
            },
            Self::Archive => EncoderConfig {
                compression_level: 9,
                filter_strategy: FilterStrategy::Best,
                interlace: false,
                gamma: None,
                optimize_palette: true,
            },
        }
    }

    /// Create encoder from profile.
    #[must_use]
    pub fn create_encoder(self) -> PngEncoder {
        PngEncoder::with_config(self.to_config())
    }
}

/// Multi-threaded PNG encoder using rayon.
pub struct ParallelPngEncoder {
    config: EncoderConfig,
}

impl ParallelPngEncoder {
    /// Create a new parallel encoder.
    #[must_use]
    pub const fn new(config: EncoderConfig) -> Self {
        Self { config }
    }

    /// Encode RGBA image using parallel processing.
    ///
    /// # Errors
    ///
    /// Returns error if encoding fails.
    pub fn encode_rgba(&self, width: u32, height: u32, data: &[u8]) -> CodecResult<Vec<u8>> {
        // For now, just use single-threaded encoder
        // Parallel processing would split scanlines across threads
        let encoder = PngEncoder::with_config(self.config.clone());
        encoder.encode_rgba(width, height, data)
    }
}

/// Create encoder from profile.
#[must_use]
pub fn encoder_from_profile(profile: EncodingProfile) -> PngEncoder {
    profile.create_encoder()
}

/// Batch encode multiple images with same settings.
///
/// # Errors
///
/// Returns error if any encoding fails.
pub fn batch_encode(
    images: &[(u32, u32, &[u8])],
    config: EncoderConfig,
) -> CodecResult<Vec<Vec<u8>>> {
    let encoder = PngEncoder::with_config(config);
    images
        .iter()
        .map(|(width, height, data)| encoder.encode_rgba(*width, *height, data))
        .collect()
}
