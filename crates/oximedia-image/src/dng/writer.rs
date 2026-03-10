//! DNG file writer.

use crate::error::{ImageError, ImageResult};

use super::constants::*;
use super::parser::tiff_type_size;
use super::types::{DngCompression, DngImage, DngMetadata};

// ==========================================
// DNG Writer
// ==========================================

/// Writer for DNG (Digital Negative) files.
pub struct DngWriter;

impl DngWriter {
    /// Write a DNG image to bytes.
    ///
    /// Writes an uncompressed DNG file with the specified metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if the image data is invalid.
    pub fn write(image: &DngImage) -> ImageResult<Vec<u8>> {
        let mut output = Vec::new();

        // TIFF header (little-endian)
        output.extend_from_slice(&[0x49, 0x49]); // "II" = little-endian
        output.extend_from_slice(&42u16.to_le_bytes()); // TIFF version

        // IFD offset placeholder (will be at end of pixel data)
        let ifd_offset_pos = output.len();
        output.extend_from_slice(&0u32.to_le_bytes());

        // Write pixel data immediately after header
        let data_offset = output.len() as u32;
        let pixel_bytes = Self::pack_pixel_data(image)?;
        let data_size = pixel_bytes.len() as u32;
        output.extend_from_slice(&pixel_bytes);

        // Align to word boundary
        if output.len() % 2 != 0 {
            output.push(0);
        }

        // Write IFD
        let ifd_offset = output.len() as u32;
        // Update IFD offset in header
        output[ifd_offset_pos..ifd_offset_pos + 4].copy_from_slice(&ifd_offset.to_le_bytes());

        Self::write_ifd(&mut output, image, data_offset, data_size)?;

        Ok(output)
    }

    /// Write a DNG from RGB data stored as a linear raw DNG.
    ///
    /// # Errors
    ///
    /// Returns an error if the dimensions or data are invalid.
    pub fn write_from_rgb(
        data: &[u16],
        width: u32,
        height: u32,
        bit_depth: u8,
        metadata: &DngMetadata,
    ) -> ImageResult<Vec<u8>> {
        let expected_len = width as usize * height as usize * 3;
        if data.len() < expected_len {
            return Err(ImageError::invalid_format(format!(
                "RGB data length {} is less than expected {} ({}x{}x3)",
                data.len(),
                expected_len,
                width,
                height
            )));
        }

        let image = DngImage {
            width,
            height,
            bit_depth,
            channels: 3,
            raw_data: data.to_vec(),
            metadata: metadata.clone(),
            is_demosaiced: true,
        };

        Self::write(&image)
    }

    fn pack_pixel_data(image: &DngImage) -> ImageResult<Vec<u8>> {
        // Always write as 16-bit uncompressed for simplicity and losslessness
        let mut bytes = Vec::with_capacity(image.raw_data.len() * 2);
        for &val in &image.raw_data {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        Ok(bytes)
    }

    fn write_ifd(
        output: &mut Vec<u8>,
        image: &DngImage,
        data_offset: u32,
        data_size: u32,
    ) -> ImageResult<()> {
        // Collect all tag entries we want to write
        let mut tags: Vec<(u16, u16, u32, u32)> = Vec::new();

        // ImageWidth
        tags.push((TAG_IMAGE_WIDTH, 4, 1, image.width)); // LONG
                                                         // ImageLength
        tags.push((TAG_IMAGE_LENGTH, 4, 1, image.height));
        // BitsPerSample (always 16 since we pack to 16-bit)
        tags.push((TAG_BITS_PER_SAMPLE, 3, 1, 16)); // SHORT
                                                    // Compression (uncompressed)
        tags.push((
            TAG_COMPRESSION,
            3,
            1,
            DngCompression::Uncompressed.to_u16() as u32,
        ));
        // PhotometricInterpretation
        let photometric: u32 = if image.channels == 1 {
            32803 // CFA
        } else {
            2 // RGB (for linear raw or demosaiced)
        };
        tags.push((TAG_PHOTOMETRIC_INTERPRETATION, 3, 1, photometric));
        // StripOffsets
        tags.push((TAG_STRIP_OFFSETS, 4, 1, data_offset));
        // SamplesPerPixel
        tags.push((TAG_SAMPLES_PER_PIXEL, 3, 1, u32::from(image.channels)));
        // RowsPerStrip
        tags.push((TAG_ROWS_PER_STRIP, 4, 1, image.height));
        // StripByteCounts
        tags.push((TAG_STRIP_BYTE_COUNTS, 4, 1, data_size));

        // Deferred data (strings, arrays stored after IFD)
        let mut deferred_data: Vec<u8> = Vec::new();

        // DNG Version tag (4 BYTE values, fits inline)
        let dng_ver = &image.metadata.dng_version;
        let dng_ver_u32 = u32::from_le_bytes([dng_ver[0], dng_ver[1], dng_ver[2], dng_ver[3]]);
        tags.push((TAG_DNG_VERSION, 1, 4, dng_ver_u32)); // BYTE

        // DNG Backward Version
        tags.push((
            TAG_DNG_BACKWARD_VERSION,
            1,
            4,
            u32::from_le_bytes([1, 1, 0, 0]),
        ));

        // CFA pattern (only for single-channel CFA data)
        if image.channels == 1 {
            // CFA Repeat Pattern Dim (2x2)
            let dim_val = 2u32 | (2u32 << 16); // two SHORTs packed
            tags.push((TAG_CFA_REPEAT_PATTERN_DIM, 3, 2, dim_val));

            // CFA Pattern (4 bytes, fits inline)
            let cfa_bytes = image.metadata.cfa_pattern.as_bytes();
            let cfa_u32 = u32::from_le_bytes(cfa_bytes);
            tags.push((TAG_CFA_PATTERN, 1, 4, cfa_u32));
        }

        // Camera model (deferred if > 4 bytes)
        if !image.metadata.camera_model.is_empty() {
            let model_bytes: Vec<u8> = image
                .metadata
                .camera_model
                .as_bytes()
                .iter()
                .copied()
                .chain(std::iter::once(0u8)) // null terminator
                .collect();
            let count = model_bytes.len() as u32;

            if count <= 4 {
                let mut inline = [0u8; 4];
                for (i, &b) in model_bytes.iter().enumerate().take(4) {
                    inline[i] = b;
                }
                tags.push((
                    TAG_UNIQUE_CAMERA_MODEL,
                    2,
                    count,
                    u32::from_le_bytes(inline),
                ));
            } else {
                // Will be written as deferred data, offset filled later
                tags.push((TAG_UNIQUE_CAMERA_MODEL, 2, count, 0)); // placeholder
                deferred_data.extend_from_slice(&model_bytes);
            }
        }

        // Software tag
        {
            let sw = b"OxiMedia DNG\0";
            let count = sw.len() as u32;
            tags.push((TAG_SOFTWARE, 2, count, 0)); // placeholder, deferred
            deferred_data.extend_from_slice(sw);
        }

        // Sort tags by tag number (TIFF requirement)
        tags.sort_by_key(|t| t.0);

        let tag_count = tags.len() as u16;

        // Calculate where deferred data starts
        let ifd_start = output.len();
        let ifd_entries_size = 2 + (tags.len() * 12) + 4; // count + entries + next_ifd_offset
        let deferred_start = (ifd_start + ifd_entries_size) as u32;

        // Fix up deferred offsets
        let mut deferred_offset = deferred_start;
        let mut deferred_items: Vec<(usize, u32)> = Vec::new(); // (tag_index, offset)

        for (i, tag) in tags.iter().enumerate() {
            let type_size = tiff_type_size(tag.1);
            let total_size = type_size * tag.2 as usize;
            if total_size > 4 {
                deferred_items.push((i, deferred_offset));
                deferred_offset += total_size as u32;
                // Align
                if deferred_offset % 2 != 0 {
                    deferred_offset += 1;
                }
            }
        }

        // Apply deferred offsets
        for (idx, offset) in &deferred_items {
            tags[*idx].3 = *offset;
        }

        // Write IFD entry count
        output.extend_from_slice(&tag_count.to_le_bytes());

        // Write tag entries
        for &(tag, dtype, count, value) in &tags {
            output.extend_from_slice(&tag.to_le_bytes());
            output.extend_from_slice(&dtype.to_le_bytes());
            output.extend_from_slice(&count.to_le_bytes());
            output.extend_from_slice(&value.to_le_bytes());
        }

        // Next IFD offset (0 = none)
        output.extend_from_slice(&0u32.to_le_bytes());

        // Write deferred data
        output.extend_from_slice(&deferred_data);

        Ok(())
    }
}
