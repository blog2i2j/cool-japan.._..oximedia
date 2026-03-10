//! Bayer demosaicing algorithms for DNG processing.

use crate::error::{ImageError, ImageResult};

use super::types::CfaPattern;

// ==========================================
// Demosaicing (Bayer to RGB)
// ==========================================

/// Perform bilinear demosaicing on raw Bayer CFA data.
///
/// Takes raw single-channel sensor data and interpolates the missing color
/// values at each pixel position using bilinear averaging of neighboring pixels.
///
/// Returns RGB interleaved u16 data (3x the input pixel count).
///
/// # Errors
///
/// Returns an error if the data length does not match width * height.
pub fn demosaic_bilinear(
    raw: &[u16],
    width: u32,
    height: u32,
    pattern: CfaPattern,
) -> ImageResult<Vec<u16>> {
    let w = width as usize;
    let h = height as usize;
    let pixel_count = w * h;

    if raw.len() < pixel_count {
        return Err(ImageError::invalid_format(format!(
            "Raw data length {} is less than expected {} ({}x{})",
            raw.len(),
            pixel_count,
            width,
            height
        )));
    }

    let mut output = vec![0u16; pixel_count * 3];
    let indices = pattern.color_indices();

    // Classify each pixel position
    // indices: [top-left, top-right, bottom-left, bottom-right]
    // 0=R, 1=G, 2=B (G appears twice in different positions)

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let out_idx = idx * 3;

            // Determine which color this pixel is
            let pattern_x = x % 2;
            let pattern_y = y % 2;
            let color = indices[pattern_y * 2 + pattern_x];

            match color {
                0 => {
                    // Red pixel: R is known, interpolate G and B
                    output[out_idx] = raw[idx];
                    output[out_idx + 1] = interpolate_green_at_rb(raw, x, y, w, h);
                    output[out_idx + 2] = interpolate_diagonal(raw, x, y, w, h);
                }
                1 => {
                    // Green pixel: G is known, interpolate R and B
                    // Determine if this green is on a red row or blue row
                    let is_red_row = if pattern_y == 0 {
                        indices[pattern_y * 2] == 0 || indices[pattern_y * 2 + 1] == 0
                    } else {
                        indices[pattern_y * 2] == 0 || indices[pattern_y * 2 + 1] == 0
                    };

                    if is_red_row {
                        // Green on red row: R is horizontal neighbor, B is vertical
                        output[out_idx] = interpolate_horizontal(raw, x, y, w, h);
                        output[out_idx + 1] = raw[idx];
                        output[out_idx + 2] = interpolate_vertical(raw, x, y, w, h);
                    } else {
                        // Green on blue row: B is horizontal neighbor, R is vertical
                        output[out_idx] = interpolate_vertical(raw, x, y, w, h);
                        output[out_idx + 1] = raw[idx];
                        output[out_idx + 2] = interpolate_horizontal(raw, x, y, w, h);
                    }
                }
                2 => {
                    // Blue pixel: B is known, interpolate R and G
                    output[out_idx] = interpolate_diagonal(raw, x, y, w, h);
                    output[out_idx + 1] = interpolate_green_at_rb(raw, x, y, w, h);
                    output[out_idx + 2] = raw[idx];
                }
                _ => {
                    // Should not happen with valid CFA patterns
                    output[out_idx] = raw[idx];
                    output[out_idx + 1] = raw[idx];
                    output[out_idx + 2] = raw[idx];
                }
            }
        }
    }

    Ok(output)
}

/// Interpolate the green channel at a red or blue pixel position.
/// Uses the 4-connected neighbors (up, down, left, right).
fn interpolate_green_at_rb(raw: &[u16], x: usize, y: usize, w: usize, h: usize) -> u16 {
    let mut sum: u32 = 0;
    let mut count: u32 = 0;

    if x > 0 {
        sum += u32::from(raw[y * w + (x - 1)]);
        count += 1;
    }
    if x + 1 < w {
        sum += u32::from(raw[y * w + (x + 1)]);
        count += 1;
    }
    if y > 0 {
        sum += u32::from(raw[(y - 1) * w + x]);
        count += 1;
    }
    if y + 1 < h {
        sum += u32::from(raw[(y + 1) * w + x]);
        count += 1;
    }

    if count > 0 {
        (sum / count) as u16
    } else {
        0
    }
}

/// Interpolate using horizontal neighbors.
fn interpolate_horizontal(raw: &[u16], x: usize, y: usize, w: usize, _h: usize) -> u16 {
    let mut sum: u32 = 0;
    let mut count: u32 = 0;

    if x > 0 {
        sum += u32::from(raw[y * w + (x - 1)]);
        count += 1;
    }
    if x + 1 < w {
        sum += u32::from(raw[y * w + (x + 1)]);
        count += 1;
    }

    if count > 0 {
        (sum / count) as u16
    } else {
        0
    }
}

/// Interpolate using vertical neighbors.
fn interpolate_vertical(raw: &[u16], x: usize, y: usize, w: usize, h: usize) -> u16 {
    let mut sum: u32 = 0;
    let mut count: u32 = 0;

    if y > 0 {
        sum += u32::from(raw[(y - 1) * w + x]);
        count += 1;
    }
    if y + 1 < h {
        sum += u32::from(raw[(y + 1) * w + x]);
        count += 1;
    }

    if count > 0 {
        (sum / count) as u16
    } else {
        0
    }
}

/// Interpolate using diagonal neighbors.
fn interpolate_diagonal(raw: &[u16], x: usize, y: usize, w: usize, h: usize) -> u16 {
    let mut sum: u32 = 0;
    let mut count: u32 = 0;

    if x > 0 && y > 0 {
        sum += u32::from(raw[(y - 1) * w + (x - 1)]);
        count += 1;
    }
    if x + 1 < w && y > 0 {
        sum += u32::from(raw[(y - 1) * w + (x + 1)]);
        count += 1;
    }
    if x > 0 && y + 1 < h {
        sum += u32::from(raw[(y + 1) * w + (x - 1)]);
        count += 1;
    }
    if x + 1 < w && y + 1 < h {
        sum += u32::from(raw[(y + 1) * w + (x + 1)]);
        count += 1;
    }

    if count > 0 {
        (sum / count) as u16
    } else {
        0
    }
}
