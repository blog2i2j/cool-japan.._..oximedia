//! PNG (Portable Network Graphics) codec implementation.
//!
//! This module provides a complete PNG 1.2 specification implementation with:
//!
//! ## Features
//!
//! - **Decoding**: Full support for all PNG color types and bit depths
//! - **Encoding**: Optimized encoding with adaptive filtering
//! - **Color Types**: Grayscale, RGB, Palette, GrayscaleAlpha, RGBA
//! - **Bit Depths**: 1, 2, 4, 8, 16 bits per sample
//! - **Interlacing**: Adam7 interlacing support
//! - **Filtering**: All five PNG filter types (None, Sub, Up, Average, Paeth)
//! - **Compression**: DEFLATE compression via oxiarc-deflate (patent-free, pure Rust)
//! - **Validation**: CRC32 chunk validation
//! - **Metadata**: Gamma, transparency, and other ancillary chunks
//!
//! ## Examples
//!
//! ### Decoding
//!
//! ```ignore
//! use oximedia_codec::png::PngDecoder;
//!
//! let png_data = std::fs::read("image.png")?;
//! let decoder = PngDecoder::new(&png_data)?;
//! let image = decoder.decode()?;
//!
//! println!("Decoded {}x{} image", image.width, image.height);
//! ```
//!
//! ### Encoding
//!
//! ```ignore
//! use oximedia_codec::png::{PngEncoder, EncoderConfig, FilterStrategy};
//!
//! let config = EncoderConfig::new()
//!     .with_compression(9)
//!     .with_filter_strategy(FilterStrategy::Best);
//!
//! let encoder = PngEncoder::with_config(config);
//! let png_data = encoder.encode_rgba(width, height, &rgba_pixels)?;
//! std::fs::write("output.png", png_data)?;
//! ```
//!
//! ### Fast Encoding
//!
//! ```ignore
//! use oximedia_codec::png::fast_encoder;
//!
//! let encoder = fast_encoder();
//! let png_data = encoder.encode_rgb(width, height, &rgb_pixels)?;
//! ```
//!
//! ## Performance
//!
//! The encoder provides different strategies for balancing speed and compression:
//!
//! - **Fast**: No filtering, fast compression (~10x faster, larger files)
//! - **Default**: Fast filter selection, medium compression (good balance)
//! - **Best**: All filters tested, maximum compression (slowest, smallest files)
//!
//! ## Safety
//!
//! This implementation uses no unsafe code and is fully memory-safe.
//! All input data is validated, including CRC checks on all chunks.

pub mod apng;
mod decoder;
mod encoder;
mod filter;

// Re-export public API
pub use decoder::{
    Chromaticity, ColorType, DecodedImage, ImageHeader, PhysicalDimensions, PngDecoder,
    PngDecoderExtended, PngMetadata, SignificantBits, TextChunk,
};
pub use encoder::{
    batch_encode, best_encoder, encoder_from_profile, fast_encoder, CompressionLevel,
    EncoderBuilder, EncoderConfig, EncodingProfile, EncodingStats, PaletteEntry, PaletteOptimizer,
    ParallelPngEncoder, PngEncoder, PngEncoderExtended,
};
pub use filter::{FilterStrategy, FilterType};

use crate::error::{CodecError, CodecResult};
use bytes::Bytes;

/// Decode PNG data and return RGBA image.
///
/// This is a convenience function that creates a decoder and decodes the image.
///
/// # Arguments
///
/// * `data` - PNG file data
///
/// # Errors
///
/// Returns error if PNG data is invalid or decoding fails.
///
/// # Examples
///
/// ```ignore
/// let png_data = std::fs::read("image.png")?;
/// let image = oximedia_codec::png::decode(&png_data)?;
/// ```
pub fn decode(data: &[u8]) -> CodecResult<DecodedImage> {
    let decoder = PngDecoder::new(data)?;
    decoder.decode()
}

/// Encode RGBA image data to PNG format.
///
/// This is a convenience function that creates a default encoder and encodes the image.
///
/// # Arguments
///
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `rgba_data` - RGBA pixel data (width * height * 4 bytes)
///
/// # Errors
///
/// Returns error if encoding fails or data is invalid.
///
/// # Examples
///
/// ```ignore
/// let png_data = oximedia_codec::png::encode_rgba(width, height, &rgba_pixels)?;
/// std::fs::write("output.png", png_data)?;
/// ```
pub fn encode_rgba(width: u32, height: u32, rgba_data: &[u8]) -> CodecResult<Vec<u8>> {
    let encoder = PngEncoder::new();
    encoder.encode_rgba(width, height, rgba_data)
}

/// Encode RGB image data to PNG format.
///
/// This is a convenience function that creates a default encoder and encodes the image.
///
/// # Arguments
///
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `rgb_data` - RGB pixel data (width * height * 3 bytes)
///
/// # Errors
///
/// Returns error if encoding fails or data is invalid.
///
/// # Examples
///
/// ```ignore
/// let png_data = oximedia_codec::png::encode_rgb(width, height, &rgb_pixels)?;
/// std::fs::write("output.png", png_data)?;
/// ```
pub fn encode_rgb(width: u32, height: u32, rgb_data: &[u8]) -> CodecResult<Vec<u8>> {
    let encoder = PngEncoder::new();
    encoder.encode_rgb(width, height, rgb_data)
}

/// Encode grayscale image data to PNG format.
///
/// # Arguments
///
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `gray_data` - Grayscale pixel data
/// * `bit_depth` - Bit depth (1, 2, 4, 8, or 16)
///
/// # Errors
///
/// Returns error if encoding fails or data is invalid.
///
/// # Examples
///
/// ```ignore
/// let png_data = oximedia_codec::png::encode_grayscale(width, height, &gray_pixels, 8)?;
/// std::fs::write("output.png", png_data)?;
/// ```
pub fn encode_grayscale(
    width: u32,
    height: u32,
    gray_data: &[u8],
    bit_depth: u8,
) -> CodecResult<Vec<u8>> {
    let encoder = PngEncoder::new();
    encoder.encode_grayscale(width, height, gray_data, bit_depth)
}

/// PNG image information without decoding pixel data.
#[derive(Debug, Clone)]
pub struct PngInfo {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Color type.
    pub color_type: ColorType,
    /// Bit depth.
    pub bit_depth: u8,
    /// Whether image is interlaced.
    pub interlaced: bool,
}

/// Get PNG image information without decoding pixel data.
///
/// This is faster than full decoding when you only need metadata.
///
/// # Arguments
///
/// * `data` - PNG file data
///
/// # Errors
///
/// Returns error if PNG data is invalid.
///
/// # Examples
///
/// ```ignore
/// let png_data = std::fs::read("image.png")?;
/// let info = oximedia_codec::png::get_info(&png_data)?;
/// println!("Image size: {}x{}", info.width, info.height);
/// ```
pub fn get_info(data: &[u8]) -> CodecResult<PngInfo> {
    let decoder = PngDecoder::new(data)?;
    Ok(PngInfo {
        width: decoder.width(),
        height: decoder.height(),
        color_type: decoder.color_type(),
        bit_depth: decoder.bit_depth(),
        interlaced: decoder.is_interlaced(),
    })
}

/// Validate PNG file format.
///
/// Checks PNG signature and performs basic validation without full decode.
///
/// # Arguments
///
/// * `data` - PNG file data
///
/// # Errors
///
/// Returns error if PNG data is invalid.
///
/// # Examples
///
/// ```ignore
/// let png_data = std::fs::read("image.png")?;
/// if oximedia_codec::png::validate(&png_data).is_ok() {
///     println!("Valid PNG file");
/// }
/// ```
pub fn validate(data: &[u8]) -> CodecResult<()> {
    PngDecoder::new(data)?;
    Ok(())
}

/// Check if data appears to be PNG format.
///
/// Only checks the PNG signature without full validation.
///
/// # Examples
///
/// ```ignore
/// let data = std::fs::read("file.png")?;
/// if oximedia_codec::png::is_png(&data) {
///     println!("Appears to be PNG format");
/// }
/// ```
#[must_use]
pub fn is_png(data: &[u8]) -> bool {
    const PNG_SIGNATURE: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];
    data.len() >= 8 && &data[0..8] == PNG_SIGNATURE
}

/// Transcode PNG to different encoding settings.
///
/// Decodes and re-encodes PNG with different settings.
///
/// # Arguments
///
/// * `data` - Input PNG data
/// * `config` - Encoder configuration
///
/// # Errors
///
/// Returns error if decoding or encoding fails.
///
/// # Examples
///
/// ```ignore
/// let input = std::fs::read("input.png")?;
/// let config = EncoderConfig::new()
///     .with_compression(9)
///     .with_filter_strategy(FilterStrategy::Best);
/// let output = oximedia_codec::png::transcode(&input, config)?;
/// std::fs::write("output.png", output)?;
/// ```
pub fn transcode(data: &[u8], config: EncoderConfig) -> CodecResult<Vec<u8>> {
    let decoder = PngDecoder::new(data)?;
    let image = decoder.decode()?;
    let encoder = PngEncoder::with_config(config);
    encoder.encode_rgba(image.width, image.height, &image.data)
}

/// Optimize PNG file for smaller size.
///
/// Re-encodes PNG with maximum compression settings.
///
/// # Arguments
///
/// * `data` - Input PNG data
///
/// # Errors
///
/// Returns error if optimization fails.
///
/// # Examples
///
/// ```ignore
/// let input = std::fs::read("input.png")?;
/// let optimized = oximedia_codec::png::optimize(&input)?;
/// std::fs::write("output.png", optimized)?;
/// println!("Size reduced from {} to {} bytes", input.len(), optimized.len());
/// ```
pub fn optimize(data: &[u8]) -> CodecResult<Vec<u8>> {
    let config = EncoderConfig::new()
        .with_compression(9)
        .with_filter_strategy(FilterStrategy::Best)
        .with_palette_optimization(true);
    transcode(data, config)
}
