//! Tiled image reading for EXR format.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]

use super::compress::{decompress_rle, decompress_zip};
use super::types::{ExrCompression, ExrHeader};
use crate::error::{ImageError, ImageResult};
use byteorder::{LittleEndian, ReadBytesExt};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

pub(crate) fn read_tiled_data(
    file: &mut File,
    header: &ExrHeader,
    width: u32,
    height: u32,
) -> ImageResult<Vec<u8>> {
    let w = width as usize;
    let h = height as usize;

    // Fall back to a default tile size if the tiledesc attribute was not parsed
    // (e.g. 64×64 is the standard OpenEXR default).
    let tile_width = if header.tile_width > 0 {
        header.tile_width as usize
    } else {
        64
    };
    let tile_height = if header.tile_height > 0 {
        header.tile_height as usize
    } else {
        64
    };

    let bytes_per_pixel: usize = header
        .channels
        .iter()
        .map(|c| c.channel_type.bytes_per_pixel())
        .sum();

    if bytes_per_pixel == 0 {
        return Err(ImageError::invalid_format(
            "EXR tiled: zero bytes per pixel",
        ));
    }

    let total_size = w * h * bytes_per_pixel;
    let mut output = vec![0u8; total_size];

    // Tile grid (ceiling division)
    let tiles_across = w.div_ceil(tile_width);
    let tiles_down = h.div_ceil(tile_height);
    let tile_count = tiles_across * tiles_down;

    // Read the tile offset table (i64 LE per entry).
    let mut tile_offsets: Vec<u64> = Vec::with_capacity(tile_count);
    for _ in 0..tile_count {
        let off = file.read_i64::<LittleEndian>()?;
        tile_offsets.push(off as u64);
    }

    for tile_row in 0..tiles_down {
        for tile_col in 0..tiles_across {
            let tile_idx = tile_row * tiles_across + tile_col;
            let offset = tile_offsets[tile_idx];

            file.seek(SeekFrom::Start(offset))?;

            // Per the EXR spec each tile chunk starts with:
            // tile_x (i32), tile_y (i32), level_x (i32), level_y (i32), data_size (i32)
            let _tile_x = file.read_i32::<LittleEndian>()?;
            let _tile_y = file.read_i32::<LittleEndian>()?;
            let _level_x = file.read_i32::<LittleEndian>()?;
            let _level_y = file.read_i32::<LittleEndian>()?;
            let pixel_data_size = file.read_i32::<LittleEndian>()? as usize;

            let mut compressed = vec![0u8; pixel_data_size];
            file.read_exact(&mut compressed)?;

            let tile_data = match header.compression {
                ExrCompression::None => compressed,
                ExrCompression::Rle => decompress_rle(&compressed)?,
                ExrCompression::Zip | ExrCompression::Zips => decompress_zip(&compressed)?,
                _ => {
                    return Err(ImageError::unsupported(format!(
                        "EXR tiled compression {:?} not yet implemented",
                        header.compression
                    )))
                }
            };

            // Position of this tile in the full image
            let x_start = tile_col * tile_width;
            let y_start = tile_row * tile_height;

            // Clamp to actual image edges
            let actual_tile_w = tile_width.min(w.saturating_sub(x_start));
            let actual_tile_h = tile_height.min(h.saturating_sub(y_start));

            // The tile pixel data is stored interleaved (matching the write convention
            // used by this library: [ch0px0 ch1px0 ... chNpx0][ch0px1 ...]).
            // We copy row by row from the tile buffer into the output image.
            let src_row_stride = tile_width * bytes_per_pixel;
            let dst_row_stride = w * bytes_per_pixel;
            let copy_len = actual_tile_w * bytes_per_pixel;

            for tile_y in 0..actual_tile_h {
                let src_offset = tile_y * src_row_stride;
                let dst_offset = (y_start + tile_y) * dst_row_stride + x_start * bytes_per_pixel;

                if src_offset + copy_len > tile_data.len() {
                    break;
                }
                if dst_offset + copy_len > output.len() {
                    break;
                }

                output[dst_offset..dst_offset + copy_len]
                    .copy_from_slice(&tile_data[src_offset..src_offset + copy_len]);
            }
        }
    }

    Ok(output)
}
