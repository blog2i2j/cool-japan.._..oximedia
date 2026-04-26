//! EXR header reading and writing.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]

use super::types::{AttributeValue, Channel, ChannelType, ExrCompression, ExrHeader, LineOrder};
use crate::error::{ImageError, ImageResult};
use crate::{ColorSpace, ImageFrame, PixelType};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};

pub(crate) fn read_exr_header(file: &mut File) -> ImageResult<ExrHeader> {
    let mut header = ExrHeader::default();

    loop {
        // Read attribute name
        let name = read_null_terminated_string(file)?;
        if name.is_empty() {
            break; // End of header
        }

        // Read attribute type (used as fallback for generic attrs)
        let attr_type = read_null_terminated_string(file)?;

        // Read attribute size
        let size = file.read_u32::<LittleEndian>()? as usize;

        // Dispatch on attribute NAME (the standard EXR attribute names are fixed).
        // The TYPE string is used for generic attributes whose name is user-defined.
        match name.as_str() {
            "channels" => {
                // Type = "chlist"
                header.channels = read_channels(file)?;
            }
            "compression" => {
                let comp = file.read_u8()?;
                header.compression = ExrCompression::from_u8(comp)?;
            }
            "dataWindow" => {
                let x_min = file.read_i32::<LittleEndian>()?;
                let y_min = file.read_i32::<LittleEndian>()?;
                let x_max = file.read_i32::<LittleEndian>()?;
                let y_max = file.read_i32::<LittleEndian>()?;
                header.data_window = (x_min, y_min, x_max, y_max);
            }
            "displayWindow" => {
                let x_min = file.read_i32::<LittleEndian>()?;
                let y_min = file.read_i32::<LittleEndian>()?;
                let x_max = file.read_i32::<LittleEndian>()?;
                let y_max = file.read_i32::<LittleEndian>()?;
                header.display_window = (x_min, y_min, x_max, y_max);
            }
            "lineOrder" => {
                let order = file.read_u8()?;
                header.line_order = LineOrder::from_u8(order)?;
            }
            "pixelAspectRatio" => {
                header.pixel_aspect_ratio = file.read_f32::<LittleEndian>()?;
            }
            "screenWindowCenter" => {
                let x = file.read_f32::<LittleEndian>()?;
                let y = file.read_f32::<LittleEndian>()?;
                header.screen_window_center = (x, y);
            }
            "screenWindowWidth" => {
                header.screen_window_width = file.read_f32::<LittleEndian>()?;
            }
            "tiledesc" => {
                // 9 bytes: u32 x_size, u32 y_size, u8 mode
                header.tile_width = file.read_u32::<LittleEndian>()?;
                header.tile_height = file.read_u32::<LittleEndian>()?;
                let _mode = file.read_u8()?;
            }
            _ => {
                // Generic attribute: dispatch on attr_type to decode into AttributeValue
                match attr_type.as_str() {
                    "int" => {
                        let value = file.read_i32::<LittleEndian>()?;
                        header.attributes.insert(name, AttributeValue::Int(value));
                    }
                    "float" => {
                        let value = file.read_f32::<LittleEndian>()?;
                        header.attributes.insert(name, AttributeValue::Float(value));
                    }
                    "string" => {
                        let mut sbuf = vec![0u8; size];
                        file.read_exact(&mut sbuf)?;
                        let s = String::from_utf8_lossy(&sbuf)
                            .trim_end_matches('\0')
                            .to_string();
                        header.attributes.insert(name, AttributeValue::String(s));
                    }
                    "v2f" => {
                        let x = file.read_f32::<LittleEndian>()?;
                        let y = file.read_f32::<LittleEndian>()?;
                        header.attributes.insert(name, AttributeValue::V2f(x, y));
                    }
                    "v3f" => {
                        let x = file.read_f32::<LittleEndian>()?;
                        let y = file.read_f32::<LittleEndian>()?;
                        let z = file.read_f32::<LittleEndian>()?;
                        header.attributes.insert(name, AttributeValue::V3f(x, y, z));
                    }
                    "box2i" => {
                        let x_min = file.read_i32::<LittleEndian>()?;
                        let y_min = file.read_i32::<LittleEndian>()?;
                        let x_max = file.read_i32::<LittleEndian>()?;
                        let y_max = file.read_i32::<LittleEndian>()?;
                        header
                            .attributes
                            .insert(name, AttributeValue::Box2i(x_min, y_min, x_max, y_max));
                    }
                    "chromaticities" => {
                        let red_x = file.read_f32::<LittleEndian>()?;
                        let red_y = file.read_f32::<LittleEndian>()?;
                        let green_x = file.read_f32::<LittleEndian>()?;
                        let green_y = file.read_f32::<LittleEndian>()?;
                        let blue_x = file.read_f32::<LittleEndian>()?;
                        let blue_y = file.read_f32::<LittleEndian>()?;
                        let white_x = file.read_f32::<LittleEndian>()?;
                        let white_y = file.read_f32::<LittleEndian>()?;
                        header.attributes.insert(
                            name,
                            AttributeValue::Chromaticities {
                                red_x,
                                red_y,
                                green_x,
                                green_y,
                                blue_x,
                                blue_y,
                                white_x,
                                white_y,
                            },
                        );
                    }
                    _ => {
                        // Skip unknown attribute
                        file.seek(SeekFrom::Current(size as i64))?;
                    }
                }
            }
        }
    }

    Ok(header)
}

fn read_channels(file: &mut File) -> ImageResult<Vec<Channel>> {
    let mut channels = Vec::new();

    loop {
        let name = read_null_terminated_string(file)?;
        if name.is_empty() {
            break;
        }

        let pixel_type = file.read_u32::<LittleEndian>()?;
        let channel_type = ChannelType::from_u32(pixel_type)?;

        // Skip pLinear (1 byte)
        file.read_u8()?;

        // Skip reserved (3 bytes)
        file.seek(SeekFrom::Current(3))?;

        let x_sampling = file.read_u32::<LittleEndian>()?;
        let y_sampling = file.read_u32::<LittleEndian>()?;

        channels.push(Channel {
            name,
            channel_type,
            x_sampling,
            y_sampling,
        });
    }

    Ok(channels)
}

pub(crate) fn read_null_terminated_string(file: &mut File) -> ImageResult<String> {
    let mut bytes = Vec::new();
    loop {
        let byte = file.read_u8()?;
        if byte == 0 {
            break;
        }
        bytes.push(byte);
    }
    Ok(String::from_utf8_lossy(&bytes).to_string())
}

pub(crate) fn determine_format(channels: &[Channel]) -> ImageResult<(PixelType, u8, ColorSpace)> {
    if channels.is_empty() {
        return Err(ImageError::invalid_format("No channels in EXR"));
    }

    // Determine pixel type from first channel
    let pixel_type = match channels[0].channel_type {
        ChannelType::Half => PixelType::F16,
        ChannelType::Float => PixelType::F32,
        ChannelType::Uint => PixelType::U32,
    };

    // Count RGBA channels
    let has_r = channels.iter().any(|c| c.name == "R");
    let has_g = channels.iter().any(|c| c.name == "G");
    let has_b = channels.iter().any(|c| c.name == "B");
    let has_a = channels.iter().any(|c| c.name == "A");
    let has_y = channels.iter().any(|c| c.name == "Y");

    let (components, color_space) = if has_r && has_g && has_b && has_a {
        (4, ColorSpace::LinearRgb)
    } else if has_r && has_g && has_b {
        (3, ColorSpace::LinearRgb)
    } else if has_y {
        (1, ColorSpace::Luma)
    } else {
        // Default to number of channels
        (channels.len() as u8, ColorSpace::LinearRgb)
    };

    Ok((pixel_type, components, color_space))
}

pub(crate) fn create_header(
    frame: &ImageFrame,
    compression: ExrCompression,
) -> ImageResult<ExrHeader> {
    let mut header = ExrHeader::default();

    // Create channels based on frame components
    let channel_type = match frame.pixel_type {
        PixelType::F16 => ChannelType::Half,
        PixelType::F32 => ChannelType::Float,
        PixelType::U32 => ChannelType::Uint,
        _ => return Err(ImageError::unsupported("Pixel type not supported for EXR")),
    };

    match frame.components {
        1 => {
            header.channels.push(Channel {
                name: "Y".to_string(),
                channel_type,
                x_sampling: 1,
                y_sampling: 1,
            });
        }
        3 => {
            for name in ["R", "G", "B"] {
                header.channels.push(Channel {
                    name: name.to_string(),
                    channel_type,
                    x_sampling: 1,
                    y_sampling: 1,
                });
            }
        }
        4 => {
            for name in ["R", "G", "B", "A"] {
                header.channels.push(Channel {
                    name: name.to_string(),
                    channel_type,
                    x_sampling: 1,
                    y_sampling: 1,
                });
            }
        }
        _ => {
            return Err(ImageError::unsupported(
                "Component count not supported for EXR",
            ))
        }
    }

    header.compression = compression;
    header.data_window = (0, 0, (frame.width - 1) as i32, (frame.height - 1) as i32);
    header.display_window = header.data_window;

    Ok(header)
}

pub(crate) fn write_exr_header(file: &mut File, header: &ExrHeader) -> ImageResult<()> {
    // Write channels attribute
    write_attribute(file, "channels", "chlist", |f| {
        for channel in &header.channels {
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
    write_simple_attribute(file, "compression", "compression", |f| {
        f.write_u8(header.compression as u8)?;
        Ok(())
    })?;

    // Write data window
    write_simple_attribute(file, "dataWindow", "box2i", |f| {
        f.write_i32::<LittleEndian>(header.data_window.0)?;
        f.write_i32::<LittleEndian>(header.data_window.1)?;
        f.write_i32::<LittleEndian>(header.data_window.2)?;
        f.write_i32::<LittleEndian>(header.data_window.3)?;
        Ok(())
    })?;

    // Write display window
    write_simple_attribute(file, "displayWindow", "box2i", |f| {
        f.write_i32::<LittleEndian>(header.display_window.0)?;
        f.write_i32::<LittleEndian>(header.display_window.1)?;
        f.write_i32::<LittleEndian>(header.display_window.2)?;
        f.write_i32::<LittleEndian>(header.display_window.3)?;
        Ok(())
    })?;

    // Write line order
    write_simple_attribute(file, "lineOrder", "lineOrder", |f| {
        f.write_u8(header.line_order as u8)?;
        Ok(())
    })?;

    // Write pixel aspect ratio
    write_simple_attribute(file, "pixelAspectRatio", "float", |f| {
        f.write_f32::<LittleEndian>(header.pixel_aspect_ratio)?;
        Ok(())
    })?;

    // Write screen window center
    write_simple_attribute(file, "screenWindowCenter", "v2f", |f| {
        f.write_f32::<LittleEndian>(header.screen_window_center.0)?;
        f.write_f32::<LittleEndian>(header.screen_window_center.1)?;
        Ok(())
    })?;

    // Write screen window width
    write_simple_attribute(file, "screenWindowWidth", "float", |f| {
        f.write_f32::<LittleEndian>(header.screen_window_width)?;
        Ok(())
    })?;

    // End of header
    file.write_u8(0)?;

    Ok(())
}

pub(crate) fn write_attribute<F>(
    file: &mut File,
    name: &str,
    attr_type: &str,
    write_data: F,
) -> ImageResult<()>
where
    F: FnOnce(&mut Vec<u8>) -> ImageResult<()>,
{
    write_null_terminated_string(file, name)?;
    write_null_terminated_string(file, attr_type)?;

    // Write data to temporary buffer to get size
    let mut data = Vec::new();
    write_data(&mut data)?;

    file.write_u32::<LittleEndian>(data.len() as u32)?;
    file.write_all(&data)?;

    Ok(())
}

pub(crate) fn write_simple_attribute<F>(
    file: &mut File,
    name: &str,
    attr_type: &str,
    write_data: F,
) -> ImageResult<()>
where
    F: FnOnce(&mut Vec<u8>) -> ImageResult<()>,
{
    write_null_terminated_string(file, name)?;
    write_null_terminated_string(file, attr_type)?;

    // Write data to temporary buffer to get size
    let mut data = Vec::new();
    write_data(&mut data)?;

    file.write_u32::<LittleEndian>(data.len() as u32)?;
    file.write_all(&data)?;

    Ok(())
}

pub(crate) fn write_null_terminated_string(
    file: &mut (impl Write + ?Sized),
    s: &str,
) -> ImageResult<()> {
    file.write_all(s.as_bytes())?;
    file.write_u8(0)?;
    Ok(())
}
