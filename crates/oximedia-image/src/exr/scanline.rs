//! Scanline image reading and writing for EXR format.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]

use super::compress::{compress_rle, compress_zip, decompress_rle, decompress_zip};
use super::types::{ExrCompression, ExrHeader};
use crate::error::{ImageError, ImageResult};
use crate::ImageFrame;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};

pub(crate) fn read_scanline_data(
    file: &mut File,
    header: &ExrHeader,
    width: u32,
    height: u32,
) -> ImageResult<Vec<u8>> {
    let pixel_count = (width * height) as usize;
    let bytes_per_pixel = header
        .channels
        .iter()
        .map(|c| c.channel_type.bytes_per_pixel())
        .sum::<usize>();

    let mut output = vec![0u8; pixel_count * bytes_per_pixel];

    // Read scanline offset table
    let scanline_count = height as usize;
    let mut offsets = Vec::with_capacity(scanline_count);
    for _ in 0..scanline_count {
        offsets.push(file.read_u64::<LittleEndian>()?);
    }

    // Read each scanline
    for (y, &offset) in offsets.iter().enumerate() {
        file.seek(SeekFrom::Start(offset))?;

        // Read scanline header
        let _y_coord = file.read_i32::<LittleEndian>()?;
        let pixel_data_size = file.read_u32::<LittleEndian>()? as usize;

        // Read compressed data
        let mut compressed = vec![0u8; pixel_data_size];
        file.read_exact(&mut compressed)?;

        // Decompress based on compression type
        let scanline_data = match header.compression {
            ExrCompression::None => compressed,
            ExrCompression::Rle => decompress_rle(&compressed)?,
            ExrCompression::Zip | ExrCompression::Zips => decompress_zip(&compressed)?,
            _ => {
                return Err(ImageError::unsupported(format!(
                    "Compression {:?} not yet implemented",
                    header.compression
                )))
            }
        };

        // Copy to output buffer
        let scanline_bytes = (width as usize) * bytes_per_pixel;
        let dest_offset = y * scanline_bytes;
        if dest_offset + scanline_bytes <= output.len() && scanline_bytes <= scanline_data.len() {
            output[dest_offset..dest_offset + scanline_bytes]
                .copy_from_slice(&scanline_data[..scanline_bytes]);
        }
    }

    Ok(output)
}

pub(crate) fn write_scanline_data(
    file: &mut File,
    frame: &ImageFrame,
    header: &ExrHeader,
) -> ImageResult<()> {
    let Some(data) = frame.data.as_slice() else {
        return Err(ImageError::unsupported("Planar data not supported for EXR"));
    };

    let height = frame.height as usize;
    let bytes_per_pixel = header
        .channels
        .iter()
        .map(|c| c.channel_type.bytes_per_pixel())
        .sum::<usize>();
    let scanline_bytes = (frame.width as usize) * bytes_per_pixel;

    // Write scanline offset table placeholder
    let offset_table_pos = file.stream_position()?;
    for _ in 0..height {
        file.write_u64::<LittleEndian>(0)?;
    }

    let mut offsets = Vec::new();

    // Write each scanline
    for y in 0..height {
        let scanline_offset = file.stream_position()?;
        offsets.push(scanline_offset);

        // Write scanline header
        file.write_i32::<LittleEndian>(y as i32)?;

        let scanline_start = y * scanline_bytes;
        let scanline_end = scanline_start + scanline_bytes;

        if scanline_end > data.len() {
            return Err(ImageError::invalid_format("Insufficient data for scanline"));
        }

        let scanline_data = &data[scanline_start..scanline_end];

        // Compress if needed
        let compressed = match header.compression {
            ExrCompression::None => scanline_data.to_vec(),
            ExrCompression::Rle => compress_rle(scanline_data)?,
            ExrCompression::Zip | ExrCompression::Zips => compress_zip(scanline_data)?,
            _ => {
                return Err(ImageError::unsupported(format!(
                    "Compression {:?} not yet implemented",
                    header.compression
                )))
            }
        };

        file.write_u32::<LittleEndian>(compressed.len() as u32)?;
        file.write_all(&compressed)?;
    }

    // Write scanline offset table
    file.seek(SeekFrom::Start(offset_table_pos))?;
    for offset in offsets {
        file.write_u64::<LittleEndian>(offset)?;
    }

    Ok(())
}
