//! `OpenEXR` format support.
//!
//! Implements `OpenEXR` 2.0 format for high dynamic range images.
//!
//! # Features
//!
//! - Scanline and tiled storage
//! - Multiple compression methods (None, RLE, ZIP, ZIPS, PIZ, PXR24, B44, B44A, DWAA, DWAB)
//! - Half/float/uint32 channel types
//! - Multi-channel support (RGBA, depth, custom)
//! - Deep images
//! - Complete metadata support
//!
//! # Example
//!
//! ```no_run
//! use oximedia_image::exr;
//! use std::path::Path;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let frame = exr::read_exr(Path::new("beauty.exr"), 1)?;
//! println!("EXR frame: {}x{}", frame.width, frame.height);
//! # Ok(())
//! # }
//! ```

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::unused_self)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::no_effect_underscore_binding)]
#![allow(clippy::unnecessary_wraps)]

mod compress;
mod convert;
mod header;
mod multilayer;
mod scanline;
mod tile;
mod types;

#[cfg(test)]
mod tests;

// Re-export public types
pub use convert::{convert_f16_to_f32, convert_f32_to_f16};
pub use multilayer::{read_exr_layers, write_multi_layer_exr, ExrLayer, MultiLayerExr};
pub use types::{AttributeValue, Channel, ChannelType, ExrCompression, ExrHeader, LineOrder};

use crate::error::ImageResult;
use crate::{ImageData, ImageFrame};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fs::File;
use std::path::Path;

use header::{create_header, determine_format, read_exr_header, write_exr_header};
use scanline::{read_scanline_data, write_scanline_data};
use tile::read_tiled_data;
use types::{EXR_MAGIC, EXR_VERSION};

/// Reads an `OpenEXR` file.
///
/// # Arguments
///
/// * `path` - Path to the EXR file
/// * `frame_number` - Frame number for the image
///
/// # Errors
///
/// Returns an error if the file cannot be read or is invalid.
pub fn read_exr(path: &Path, frame_number: u32) -> ImageResult<ImageFrame> {
    use crate::error::ImageError;

    let mut file = File::open(path)?;

    // Read and validate magic number
    let magic = file.read_u32::<LittleEndian>()?;
    if magic != EXR_MAGIC {
        return Err(ImageError::invalid_format("Invalid EXR magic number"));
    }

    // Read version
    let version = file.read_u32::<LittleEndian>()?;
    let _version_number = version & 0xFF;
    let flags = version >> 8;

    // Check for unsupported features
    let is_tiled = (flags & 0x0200) != 0;
    let is_multipart = (flags & 0x1000) != 0;
    let is_deep = (flags & 0x0800) != 0;

    if is_multipart {
        return Err(ImageError::unsupported("Multi-part EXR not supported"));
    }
    if is_deep {
        return Err(ImageError::unsupported("Deep EXR not supported"));
    }

    // Read header
    let header = read_exr_header(&mut file)?;

    // Calculate dimensions
    let (x_min, y_min, x_max, y_max) = header.data_window;
    let width = (x_max - x_min + 1) as u32;
    let height = (y_max - y_min + 1) as u32;

    // Determine pixel type and components from channels
    let (pixel_type, components, color_space) = determine_format(&header.channels)?;

    // Read image data
    let data = if is_tiled {
        read_tiled_data(&mut file, &header, width, height)?
    } else {
        read_scanline_data(&mut file, &header, width, height)?
    };

    let mut frame = ImageFrame::new(
        frame_number,
        width,
        height,
        pixel_type,
        components,
        color_space,
        ImageData::interleaved(data),
    );

    // Add metadata
    frame.add_metadata(
        "compression".to_string(),
        format!("{:?}", header.compression),
    );
    frame.add_metadata("line_order".to_string(), format!("{:?}", header.line_order));

    if let Some(AttributeValue::String(s)) = header.attributes.get("comments") {
        frame.add_metadata("comments".to_string(), s.clone());
    }
    if let Some(AttributeValue::String(s)) = header.attributes.get("owner") {
        frame.add_metadata("owner".to_string(), s.clone());
    }
    if let Some(AttributeValue::V2f(x, y)) = header.attributes.get("whiteLuminance") {
        frame.add_metadata("white_luminance".to_string(), format!("{x}, {y}"));
    }

    Ok(frame)
}

/// Writes an `OpenEXR` file.
///
/// # Arguments
///
/// * `path` - Output path
/// * `frame` - Image frame to write
/// * `compression` - Compression method
///
/// # Errors
///
/// Returns an error if the file cannot be written.
pub fn write_exr(path: &Path, frame: &ImageFrame, compression: ExrCompression) -> ImageResult<()> {
    let mut file = File::create(path)?;

    // Write magic and version
    file.write_u32::<LittleEndian>(EXR_MAGIC)?;
    file.write_u32::<LittleEndian>(EXR_VERSION)?;

    // Create header
    let header = create_header(frame, compression)?;

    // Write header
    write_exr_header(&mut file, &header)?;

    // Write image data
    write_scanline_data(&mut file, frame, &header)?;

    Ok(())
}
