//! Multi-layer EXR support.
//!
//! Provides `ExrLayer` and `MultiLayerExr` types for compositing workflows
//! where different render passes (beauty, diffuse, specular, depth, etc.)
//! are stored in a single EXR file.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]

use super::header::{
    determine_format, read_exr_header, write_attribute, write_null_terminated_string,
    write_simple_attribute,
};
use super::scanline::read_scanline_data;
use super::types::{
    AttributeValue, Channel, ChannelType, ExrCompression, ExrHeader, EXR_MAGIC, EXR_VERSION,
};
use crate::error::{ImageError, ImageResult};
use crate::{ImageData, ImageFrame};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

/// An individual layer in a multi-layer EXR file.
///
/// Each layer has its own set of channels and data, suitable for compositing
/// workflows where different render passes (beauty, diffuse, specular, depth, etc.)
/// are stored in a single file.
#[derive(Debug, Clone)]
pub struct ExrLayer {
    /// Layer name (e.g., "beauty", "diffuse", "specular", "depth").
    pub name: String,
    /// Channels in this layer.
    pub channels: Vec<Channel>,
    /// Data window for this layer.
    pub data_window: (i32, i32, i32, i32),
    /// Width derived from data window.
    pub width: u32,
    /// Height derived from data window.
    pub height: u32,
    /// Raw pixel data (interleaved channel data).
    pub data: Vec<u8>,
    /// Custom attributes for this layer.
    pub attributes: HashMap<String, AttributeValue>,
}

impl ExrLayer {
    /// Creates a new EXR layer.
    #[must_use]
    pub fn new(name: &str, width: u32, height: u32, channels: Vec<Channel>, data: Vec<u8>) -> Self {
        Self {
            name: name.to_string(),
            channels,
            data_window: (
                0,
                0,
                (width.saturating_sub(1)) as i32,
                (height.saturating_sub(1)) as i32,
            ),
            width,
            height,
            data,
            attributes: HashMap::new(),
        }
    }

    /// Returns the number of bytes per pixel for this layer.
    #[must_use]
    pub fn bytes_per_pixel(&self) -> usize {
        self.channels
            .iter()
            .map(|c| c.channel_type.bytes_per_pixel())
            .sum()
    }

    /// Returns the total data size expected for this layer.
    #[must_use]
    pub fn expected_data_size(&self) -> usize {
        (self.width as usize) * (self.height as usize) * self.bytes_per_pixel()
    }

    /// Adds a custom attribute to this layer.
    pub fn add_attribute(&mut self, name: String, value: AttributeValue) {
        self.attributes.insert(name, value);
    }

    /// Extracts a single channel's data from the interleaved layer data.
    ///
    /// Returns the raw bytes for the specified channel, or None if the channel
    /// is not found.
    #[must_use]
    pub fn extract_channel(&self, channel_name: &str) -> Option<Vec<u8>> {
        let channel_idx = self.channels.iter().position(|c| c.name == channel_name)?;

        let bpp = self.bytes_per_pixel();
        let pixel_count = (self.width as usize) * (self.height as usize);

        // Calculate byte offset of this channel within a pixel
        let mut byte_offset = 0;
        for ch in &self.channels[..channel_idx] {
            byte_offset += ch.channel_type.bytes_per_pixel();
        }

        let ch_bytes = self.channels[channel_idx].channel_type.bytes_per_pixel();
        let mut output = Vec::with_capacity(pixel_count * ch_bytes);

        for px in 0..pixel_count {
            let start = px * bpp + byte_offset;
            let end = start + ch_bytes;
            if end <= self.data.len() {
                output.extend_from_slice(&self.data[start..end]);
            } else {
                output.extend(std::iter::repeat(0u8).take(ch_bytes));
            }
        }

        Some(output)
    }
}

/// Multi-layer EXR file representation.
///
/// Contains multiple layers, each with their own channels and data.
/// Useful for compositing workflows where render passes are stored together.
#[derive(Debug, Clone)]
pub struct MultiLayerExr {
    /// All layers in the file.
    pub layers: Vec<ExrLayer>,
    /// Global display window.
    pub display_window: (i32, i32, i32, i32),
    /// Compression method.
    pub compression: ExrCompression,
}

impl MultiLayerExr {
    /// Creates a new multi-layer EXR with the given display window.
    #[must_use]
    pub fn new(width: u32, height: u32, compression: ExrCompression) -> Self {
        Self {
            layers: Vec::new(),
            display_window: (
                0,
                0,
                (width.saturating_sub(1)) as i32,
                (height.saturating_sub(1)) as i32,
            ),
            compression,
        }
    }

    /// Adds a layer to the multi-layer EXR.
    pub fn add_layer(&mut self, layer: ExrLayer) {
        self.layers.push(layer);
    }

    /// Returns a reference to a layer by name.
    #[must_use]
    pub fn get_layer(&self, name: &str) -> Option<&ExrLayer> {
        self.layers.iter().find(|l| l.name == name)
    }

    /// Returns the number of layers.
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Returns a list of all layer names.
    #[must_use]
    pub fn layer_names(&self) -> Vec<&str> {
        self.layers.iter().map(|l| l.name.as_str()).collect()
    }

    /// Creates a multi-layer EXR from a single `ImageFrame` (single layer named "rgba").
    pub fn from_frame(frame: &ImageFrame, compression: ExrCompression) -> ImageResult<Self> {
        let channel_type = match frame.pixel_type {
            crate::PixelType::F16 => ChannelType::Half,
            crate::PixelType::F32 => ChannelType::Float,
            crate::PixelType::U32 => ChannelType::Uint,
            _ => return Err(ImageError::unsupported("Pixel type not supported for EXR")),
        };

        let channel_names: Vec<&str> = match frame.components {
            1 => vec!["Y"],
            3 => vec!["R", "G", "B"],
            4 => vec!["R", "G", "B", "A"],
            _ => {
                return Err(ImageError::unsupported(
                    "Component count not supported for EXR",
                ))
            }
        };

        let channels: Vec<Channel> = channel_names
            .iter()
            .map(|name| Channel {
                name: (*name).to_string(),
                channel_type,
                x_sampling: 1,
                y_sampling: 1,
            })
            .collect();

        let data = frame
            .data
            .as_slice()
            .ok_or_else(|| ImageError::unsupported("Planar data not supported for EXR"))?
            .to_vec();

        let layer = ExrLayer::new("rgba", frame.width, frame.height, channels, data);
        let mut exr = Self::new(frame.width, frame.height, compression);
        exr.add_layer(layer);
        Ok(exr)
    }

    /// Converts the first layer to an `ImageFrame`.
    pub fn to_frame(&self, frame_number: u32) -> ImageResult<ImageFrame> {
        let layer = self
            .layers
            .first()
            .ok_or_else(|| ImageError::invalid_format("No layers in multi-layer EXR"))?;

        let (pixel_type, components, color_space) = determine_format(&layer.channels)?;

        Ok(ImageFrame::new(
            frame_number,
            layer.width,
            layer.height,
            pixel_type,
            components,
            color_space,
            ImageData::interleaved(layer.data.clone()),
        ))
    }
}

/// Writes a multi-layer EXR to bytes.
///
/// Each layer is written as a separate "part" in a multi-part EXR file.
/// The format uses the OpenEXR 2.0 multi-part extension.
///
/// # Errors
///
/// Returns an error if the layers have invalid configurations.
pub fn write_multi_layer_exr(path: &Path, multi: &MultiLayerExr) -> ImageResult<()> {
    let mut file = File::create(path)?;

    // Write magic
    file.write_u32::<LittleEndian>(EXR_MAGIC)?;

    // Write version with multi-part flag
    let version = EXR_VERSION | (0x1000 << 8); // multi-part flag
    file.write_u32::<LittleEndian>(version)?;

    // Write headers for each part
    for layer in &multi.layers {
        // Write "name" attribute
        write_simple_attribute(&mut file, "name", "string", |f| {
            f.write_all(layer.name.as_bytes())?;
            Ok(())
        })?;

        // Write "type" attribute (scanlineimage)
        write_simple_attribute(&mut file, "type", "string", |f| {
            f.write_all(b"scanlineimage")?;
            Ok(())
        })?;

        // Write channels
        write_attribute(&mut file, "channels", "chlist", |f| {
            for channel in &layer.channels {
                write_null_terminated_string(f, &channel.name)?;
                f.write_u32::<LittleEndian>(channel.channel_type as u32)?;
                f.write_u8(0)?; // pLinear
                f.write_all(&[0u8; 3])?; // reserved
                f.write_u32::<LittleEndian>(channel.x_sampling)?;
                f.write_u32::<LittleEndian>(channel.y_sampling)?;
            }
            write_null_terminated_string(f, "")?;
            Ok(())
        })?;

        // Write compression
        write_simple_attribute(&mut file, "compression", "compression", |f| {
            f.write_u8(multi.compression as u8)?;
            Ok(())
        })?;

        // Write data window
        write_simple_attribute(&mut file, "dataWindow", "box2i", |f| {
            let dw = layer.data_window;
            f.write_i32::<LittleEndian>(dw.0)?;
            f.write_i32::<LittleEndian>(dw.1)?;
            f.write_i32::<LittleEndian>(dw.2)?;
            f.write_i32::<LittleEndian>(dw.3)?;
            Ok(())
        })?;

        // Write display window
        write_simple_attribute(&mut file, "displayWindow", "box2i", |f| {
            let dw = multi.display_window;
            f.write_i32::<LittleEndian>(dw.0)?;
            f.write_i32::<LittleEndian>(dw.1)?;
            f.write_i32::<LittleEndian>(dw.2)?;
            f.write_i32::<LittleEndian>(dw.3)?;
            Ok(())
        })?;

        // Write line order
        write_simple_attribute(&mut file, "lineOrder", "lineOrder", |f| {
            f.write_u8(0)?; // IncreasingY
            Ok(())
        })?;

        // Write pixel aspect ratio
        write_simple_attribute(&mut file, "pixelAspectRatio", "float", |f| {
            f.write_f32::<LittleEndian>(1.0)?;
            Ok(())
        })?;

        // Write screen window center
        write_simple_attribute(&mut file, "screenWindowCenter", "v2f", |f| {
            f.write_f32::<LittleEndian>(0.0)?;
            f.write_f32::<LittleEndian>(0.0)?;
            Ok(())
        })?;

        // Write screen window width
        write_simple_attribute(&mut file, "screenWindowWidth", "float", |f| {
            f.write_f32::<LittleEndian>(1.0)?;
            Ok(())
        })?;

        // End of this part's header
        file.write_u8(0)?;
    }

    // End of all headers (empty header)
    file.write_u8(0)?;

    // Write chunk offset tables and data for each part
    for layer in &multi.layers {
        let height = layer.height as usize;
        let bpp = layer.bytes_per_pixel();
        let scanline_bytes = (layer.width as usize) * bpp;

        // Write offset table placeholder
        let offset_table_pos = file.stream_position()?;
        for _ in 0..height {
            file.write_u64::<LittleEndian>(0)?;
        }

        let mut offsets = Vec::with_capacity(height);

        // Write scanlines
        for y in 0..height {
            let offset = file.stream_position()?;
            offsets.push(offset);

            // Scanline Y coordinate
            file.write_i32::<LittleEndian>(y as i32)?;

            // Data
            let start = y * scanline_bytes;
            let end = (start + scanline_bytes).min(layer.data.len());
            let scanline_data = if start < layer.data.len() {
                &layer.data[start..end]
            } else {
                &[]
            };

            // Write uncompressed size
            file.write_u32::<LittleEndian>(scanline_data.len() as u32)?;
            file.write_all(scanline_data)?;
        }

        // Update offset table
        let current_pos = file.stream_position()?;
        file.seek(SeekFrom::Start(offset_table_pos))?;
        for offset in &offsets {
            file.write_u64::<LittleEndian>(*offset)?;
        }
        file.seek(SeekFrom::Start(current_pos))?;
    }

    Ok(())
}

/// Reads a multi-layer EXR file and returns individual layers.
///
/// For standard single-part EXR files, returns a single layer.
/// For multi-part EXR files, returns all layers.
///
/// # Errors
///
/// Returns an error if the file is invalid or cannot be read.
pub fn read_exr_layers(path: &Path) -> ImageResult<MultiLayerExr> {
    let mut file = File::open(path)?;

    // Read and validate magic number
    let magic = file.read_u32::<LittleEndian>()?;
    if magic != EXR_MAGIC {
        return Err(ImageError::invalid_format("Invalid EXR magic number"));
    }

    // Read version
    let version = file.read_u32::<LittleEndian>()?;
    let flags = version >> 8;
    let is_multipart = (flags & 0x1000) != 0;

    if is_multipart {
        read_multipart_layers(&mut file)
    } else {
        read_singlepart_as_layer(&mut file)
    }
}

/// Reads a standard single-part EXR and wraps it as a single-layer `MultiLayerExr`.
fn read_singlepart_as_layer(file: &mut File) -> ImageResult<MultiLayerExr> {
    let header = read_exr_header(file)?;

    let (x_min, y_min, x_max, y_max) = header.data_window;
    let width = (x_max - x_min + 1) as u32;
    let height = (y_max - y_min + 1) as u32;

    let data = read_scanline_data(file, &header, width, height)?;

    let layer = ExrLayer {
        name: "default".to_string(),
        channels: header.channels.clone(),
        data_window: header.data_window,
        width,
        height,
        data,
        attributes: header.attributes.clone(),
    };

    let mut multi = MultiLayerExr::new(width, height, header.compression);
    multi.display_window = header.display_window;
    multi.add_layer(layer);

    Ok(multi)
}

/// Reads a multi-part EXR file with multiple headers and chunk tables.
fn read_multipart_layers(file: &mut File) -> ImageResult<MultiLayerExr> {
    // Read all part headers until we hit an empty header
    let mut part_headers: Vec<ExrHeader> = Vec::new();
    let mut part_names: Vec<String> = Vec::new();

    loop {
        // Try reading first attribute name
        let pos = file.stream_position()?;
        let first_byte = file.read_u8()?;
        if first_byte == 0 {
            // Empty header = end of headers
            break;
        }
        // Seek back and read full header
        file.seek(SeekFrom::Start(pos))?;

        let header = read_exr_header(file)?;

        let name = if let Some(AttributeValue::String(n)) = header.attributes.get("name") {
            n.clone()
        } else {
            format!("part_{}", part_headers.len())
        };

        part_names.push(name);
        part_headers.push(header);
    }

    if part_headers.is_empty() {
        return Err(ImageError::invalid_format(
            "No parts found in multi-part EXR",
        ));
    }

    // Use first header for display window
    let dw = part_headers[0].display_window;
    let width = (dw.2 - dw.0 + 1) as u32;
    let height = (dw.3 - dw.1 + 1) as u32;
    let compression = part_headers[0].compression;

    let mut multi = MultiLayerExr::new(width, height, compression);
    multi.display_window = dw;

    // Read chunk data for each part
    for (i, header) in part_headers.iter().enumerate() {
        let (x_min, y_min, x_max, y_max) = header.data_window;
        let pw = (x_max - x_min + 1) as u32;
        let ph = (y_max - y_min + 1) as u32;

        let data = read_scanline_data(file, header, pw, ph)?;

        let layer = ExrLayer {
            name: part_names[i].clone(),
            channels: header.channels.clone(),
            data_window: header.data_window,
            width: pw,
            height: ph,
            data,
            attributes: header.attributes.clone(),
        };

        multi.add_layer(layer);
    }

    Ok(multi)
}
