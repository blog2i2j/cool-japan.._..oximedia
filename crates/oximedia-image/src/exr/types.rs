//! Core types for `OpenEXR` format support.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]

use crate::error::{ImageError, ImageResult};
use std::collections::HashMap;

/// EXR magic number.
pub(crate) const EXR_MAGIC: u32 = 20000630;

/// EXR version (2.0).
pub(crate) const EXR_VERSION: u32 = 2;

/// Channel type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelType {
    /// Unsigned 32-bit integer.
    Uint = 0,
    /// 16-bit floating point (half).
    Half = 1,
    /// 32-bit floating point.
    Float = 2,
}

impl ChannelType {
    pub(crate) fn from_u32(value: u32) -> ImageResult<Self> {
        match value {
            0 => Ok(Self::Uint),
            1 => Ok(Self::Half),
            2 => Ok(Self::Float),
            _ => Err(ImageError::invalid_format("Invalid channel type")),
        }
    }

    pub(crate) const fn bytes_per_pixel(&self) -> usize {
        match self {
            Self::Uint | Self::Float => 4,
            Self::Half => 2,
        }
    }
}

/// Line order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineOrder {
    /// Increasing Y (top to bottom).
    IncreasingY = 0,
    /// Decreasing Y (bottom to top).
    DecreasingY = 1,
    /// Random Y access.
    RandomY = 2,
}

impl LineOrder {
    pub(crate) fn from_u8(value: u8) -> ImageResult<Self> {
        match value {
            0 => Ok(Self::IncreasingY),
            1 => Ok(Self::DecreasingY),
            2 => Ok(Self::RandomY),
            _ => Err(ImageError::invalid_format("Invalid line order")),
        }
    }
}

/// Compression type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExrCompression {
    /// No compression.
    None = 0,
    /// Run-length encoding.
    Rle = 1,
    /// Zlib compression (deflate).
    Zip = 2,
    /// Zlib compression per scanline.
    Zips = 3,
    /// PIZ wavelet compression.
    Piz = 4,
    /// PXR24 compression.
    Pxr24 = 5,
    /// B44 compression.
    B44 = 6,
    /// B44A compression.
    B44a = 7,
    /// DWAA compression.
    Dwaa = 8,
    /// DWAB compression.
    Dwab = 9,
}

impl ExrCompression {
    pub(crate) fn from_u8(value: u8) -> ImageResult<Self> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::Rle),
            2 => Ok(Self::Zip),
            3 => Ok(Self::Zips),
            4 => Ok(Self::Piz),
            5 => Ok(Self::Pxr24),
            6 => Ok(Self::B44),
            7 => Ok(Self::B44a),
            8 => Ok(Self::Dwaa),
            9 => Ok(Self::Dwab),
            _ => Err(ImageError::invalid_format("Invalid compression type")),
        }
    }
}

/// Channel description.
#[derive(Debug, Clone)]
pub struct Channel {
    /// Channel name (e.g., "R", "G", "B", "A", "Z").
    pub name: String,
    /// Channel type (half, float, uint).
    pub channel_type: ChannelType,
    /// X subsampling.
    pub x_sampling: u32,
    /// Y subsampling.
    pub y_sampling: u32,
}

/// EXR header.
#[derive(Debug, Clone)]
pub struct ExrHeader {
    /// Channels in the image.
    pub channels: Vec<Channel>,
    /// Compression method.
    pub compression: ExrCompression,
    /// Data window (actual pixel data bounds).
    pub data_window: (i32, i32, i32, i32), // (xMin, yMin, xMax, yMax)
    /// Display window (viewing bounds).
    pub display_window: (i32, i32, i32, i32),
    /// Line order.
    pub line_order: LineOrder,
    /// Pixel aspect ratio.
    pub pixel_aspect_ratio: f32,
    /// Screen window center.
    pub screen_window_center: (f32, f32),
    /// Screen window width.
    pub screen_window_width: f32,
    /// Tile width (from tiledesc attribute; 0 = scanline image).
    pub tile_width: u32,
    /// Tile height (from tiledesc attribute; 0 = scanline image).
    pub tile_height: u32,
    /// Custom attributes.
    pub attributes: HashMap<String, AttributeValue>,
}

/// Attribute value types.
#[derive(Debug, Clone)]
pub enum AttributeValue {
    /// Integer value.
    Int(i32),
    /// Float value.
    Float(f32),
    /// String value.
    String(String),
    /// Vector 2f.
    V2f(f32, f32),
    /// Vector 3f.
    V3f(f32, f32, f32),
    /// Box 2i.
    Box2i(i32, i32, i32, i32),
    /// Chromaticities.
    Chromaticities {
        /// Red X.
        red_x: f32,
        /// Red Y.
        red_y: f32,
        /// Green X.
        green_x: f32,
        /// Green Y.
        green_y: f32,
        /// Blue X.
        blue_x: f32,
        /// Blue Y.
        blue_y: f32,
        /// White X.
        white_x: f32,
        /// White Y.
        white_y: f32,
    },
}

impl Default for ExrHeader {
    fn default() -> Self {
        Self {
            channels: Vec::new(),
            compression: ExrCompression::None,
            data_window: (0, 0, 0, 0),
            display_window: (0, 0, 0, 0),
            line_order: LineOrder::IncreasingY,
            pixel_aspect_ratio: 1.0,
            screen_window_center: (0.0, 0.0),
            screen_window_width: 1.0,
            tile_width: 0,
            tile_height: 0,
            attributes: HashMap::new(),
        }
    }
}
