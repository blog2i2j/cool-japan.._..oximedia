//! Conversion helpers between DNG images and ImageFrame.

use crate::error::{ImageError, ImageResult};
use crate::{ColorSpace, ImageData, ImageFrame, PixelType};

use super::demosaic::demosaic_bilinear;
use super::types::{DngImage, DngMetadata};

// ==========================================
// Conversion helpers
// ==========================================

/// Convert a DNG image to an `ImageFrame`.
///
/// If `demosaic` is true and the image is single-channel CFA data,
/// bilinear demosaicing will be applied to produce RGB output.
///
/// # Errors
///
/// Returns an error if demosaicing or conversion fails.
pub fn dng_to_image_frame(image: &DngImage, demosaic: bool) -> ImageResult<ImageFrame> {
    let pixel_type = match image.bit_depth {
        8 => PixelType::U8,
        10 => PixelType::U10,
        12 => PixelType::U12,
        14 | 16 => PixelType::U16,
        _ => {
            return Err(ImageError::unsupported(format!(
                "Bit depth {} not mappable to PixelType",
                image.bit_depth
            )))
        }
    };

    if demosaic && !image.is_demosaiced && image.channels == 1 {
        // Demosaic to RGB
        let rgb_data = demosaic_bilinear(
            &image.raw_data,
            image.width,
            image.height,
            image.metadata.cfa_pattern,
        )?;

        // Convert u16 to bytes (little-endian)
        let byte_data: Vec<u8> = rgb_data.iter().flat_map(|v| v.to_le_bytes()).collect();

        let mut frame = ImageFrame::new(
            0,
            image.width,
            image.height,
            PixelType::U16, // demosaiced is always u16 internally
            3,
            ColorSpace::LinearRgb,
            ImageData::interleaved(byte_data),
        );

        frame.add_metadata("format".to_string(), "dng".to_string());
        frame.add_metadata(
            "camera_model".to_string(),
            image.metadata.camera_model.clone(),
        );
        frame.add_metadata("bit_depth".to_string(), image.bit_depth.to_string());

        Ok(frame)
    } else {
        // Return raw data as-is
        let components = image.channels;
        let byte_data: Vec<u8> = if pixel_type == PixelType::U8 {
            image.raw_data.iter().map(|v| *v as u8).collect()
        } else {
            image
                .raw_data
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect()
        };

        let color_space = if components == 1 {
            ColorSpace::Luma
        } else {
            ColorSpace::LinearRgb
        };

        let mut frame = ImageFrame::new(
            0,
            image.width,
            image.height,
            pixel_type,
            components,
            color_space,
            ImageData::interleaved(byte_data),
        );

        frame.add_metadata("format".to_string(), "dng".to_string());
        frame.add_metadata(
            "camera_model".to_string(),
            image.metadata.camera_model.clone(),
        );

        Ok(frame)
    }
}

/// Convert an `ImageFrame` to a `DngImage`.
///
/// If metadata is not provided, default DNG metadata is used.
///
/// # Errors
///
/// Returns an error if the frame data cannot be converted.
pub fn image_frame_to_dng(
    frame: &ImageFrame,
    metadata: Option<&DngMetadata>,
) -> ImageResult<DngImage> {
    let data_bytes = frame
        .data
        .as_slice()
        .ok_or_else(|| ImageError::unsupported("Planar data not supported for DNG conversion"))?;

    let raw_data: Vec<u16> = match frame.pixel_type {
        PixelType::U8 => data_bytes.iter().map(|&b| u16::from(b)).collect(),
        PixelType::U16 | PixelType::U10 | PixelType::U12 => {
            let mut values = Vec::with_capacity(data_bytes.len() / 2);
            for chunk in data_bytes.chunks(2) {
                if chunk.len() == 2 {
                    values.push(u16::from_le_bytes([chunk[0], chunk[1]]));
                }
            }
            values
        }
        _ => {
            return Err(ImageError::unsupported(format!(
                "Pixel type {:?} not supported for DNG conversion",
                frame.pixel_type
            )));
        }
    };

    let bit_depth = frame.pixel_type.bit_depth();
    let channels = frame.components;
    let is_demosaiced = channels > 1;

    let mut meta = metadata.cloned().unwrap_or_default();
    // Transfer any relevant frame metadata
    if let Some(model) = frame.get_metadata("camera_model") {
        meta.camera_model = model.to_string();
    }

    Ok(DngImage {
        width: frame.width,
        height: frame.height,
        bit_depth,
        channels,
        raw_data,
        metadata: meta,
        is_demosaiced,
    })
}
