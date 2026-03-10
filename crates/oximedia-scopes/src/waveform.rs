//! Waveform monitor for video signal analysis.
//!
//! Waveform monitors display the luminance (brightness) or RGB component values
//! across horizontal scanlines of the video. This is essential for:
//! - Checking exposure and dynamic range
//! - Ensuring legal broadcast levels (0-100 IRE)
//! - Identifying clipping and crushing
//! - Matching shots in color grading
//!
//! Supported modes:
//! - **Luma**: Y channel only (luminance)
//! - **RGB Parade**: R, G, B displayed side-by-side
//! - **RGB Overlay**: All RGB channels overlaid with color
//! - **YCbCr**: Y, Cb, Cr displayed side-by-side

use crate::render::{rgb_to_ycbcr, Canvas};
use crate::{ScopeConfig, ScopeData, ScopeType};
use oximedia_core::OxiResult;
use rayon::prelude::*;

/// Generates a luma (Y channel) waveform from RGB frame data.
///
/// The waveform shows the luminance distribution across the horizontal axis,
/// with intensity accumulated for each pixel position.
///
/// # Arguments
///
/// * `frame` - RGB24 frame data (width * height * 3 bytes)
/// * `width` - Frame width in pixels
/// * `height` - Frame height in pixels
/// * `config` - Scope configuration
///
/// # Errors
///
/// Returns an error if frame data is invalid or insufficient.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn generate_luma_waveform(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &ScopeConfig,
) -> OxiResult<ScopeData> {
    let expected_size = (width * height * 3) as usize;
    if frame.len() < expected_size {
        return Err(oximedia_core::OxiError::InvalidData(format!(
            "Frame data too small: expected {expected_size}, got {}",
            frame.len()
        )));
    }

    let scope_width = config.width;
    let scope_height = config.height;

    let mut canvas = Canvas::new(scope_width, scope_height);

    // Create accumulation buffer for intensity
    let mut accumulator = vec![0u32; (scope_width * scope_height) as usize];

    // Process frame in parallel by rows
    let row_accumulators: Vec<Vec<u32>> = (0..height)
        .into_par_iter()
        .map(|y| {
            let mut local_accum = vec![0u32; (scope_width * scope_height) as usize];

            for x in 0..width {
                let pixel_idx = ((y * width + x) * 3) as usize;
                let r = frame[pixel_idx];
                let g = frame[pixel_idx + 1];
                let b = frame[pixel_idx + 2];

                // Convert to luma (ITU-R BT.709)
                let (luma, _, _) = rgb_to_ycbcr(r, g, b);

                // Map to scope coordinates
                let scope_x = (x * scope_width) / width;
                let mapped = ((u32::from(luma) * scope_height) / 255).min(scope_height - 1);
                let scope_y = scope_height - 1 - mapped;

                let idx = (scope_y * scope_width + scope_x) as usize;
                if idx < local_accum.len() {
                    local_accum[idx] = local_accum[idx].saturating_add(1);
                }
            }

            local_accum
        })
        .collect();

    // Merge all row accumulators
    for row_accum in &row_accumulators {
        for (i, &val) in row_accum.iter().enumerate() {
            accumulator[i] = accumulator[i].saturating_add(val);
        }
    }

    // Find max value for normalization
    let max_val = accumulator.iter().copied().max().unwrap_or(1);

    // Draw accumulated waveform
    for y in 0..scope_height {
        for x in 0..scope_width {
            let idx = (y * scope_width + x) as usize;
            let count = accumulator[idx];

            if count > 0 {
                // Normalize to 0-255 range with gamma correction for better visibility
                let normalized = ((count as f32 / max_val as f32).sqrt() * 255.0) as u8;
                canvas.accumulate_pixel(x, y, normalized);
            }
        }
    }

    // Draw graticule
    if config.show_graticule {
        crate::render::draw_waveform_graticule(&mut canvas, config);
    }

    // Draw labels
    if config.show_labels {
        draw_waveform_labels(&mut canvas);
    }

    Ok(ScopeData {
        width: scope_width,
        height: scope_height,
        data: canvas.data,
        scope_type: ScopeType::WaveformLuma,
    })
}

/// Generates an RGB parade waveform (R|G|B side-by-side).
///
/// Each color channel is displayed in its own vertical section,
/// making it easy to compare and balance channels.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn generate_rgb_parade(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &ScopeConfig,
) -> OxiResult<ScopeData> {
    let expected_size = (width * height * 3) as usize;
    if frame.len() < expected_size {
        return Err(oximedia_core::OxiError::InvalidData(format!(
            "Frame data too small: expected {expected_size}, got {}",
            frame.len()
        )));
    }

    let scope_width = config.width;
    let scope_height = config.height;
    let section_width = scope_width / 3;

    let mut canvas = Canvas::new(scope_width, scope_height);

    // Three accumulators for R, G, B
    let mut accumulators = [
        vec![0u32; (section_width * scope_height) as usize],
        vec![0u32; (section_width * scope_height) as usize],
        vec![0u32; (section_width * scope_height) as usize],
    ];

    // Process frame
    for y in 0..height {
        for x in 0..width {
            let pixel_idx = ((y * width + x) * 3) as usize;
            let rgb = [frame[pixel_idx], frame[pixel_idx + 1], frame[pixel_idx + 2]];

            let scope_x = (x * section_width) / width;

            for (channel, &value) in rgb.iter().enumerate() {
                let mapped = ((u32::from(value) * scope_height) / 255).min(scope_height - 1);
                let scope_y = scope_height - 1 - mapped;
                let idx = (scope_y * section_width + scope_x) as usize;

                if idx < accumulators[channel].len() {
                    accumulators[channel][idx] = accumulators[channel][idx].saturating_add(1);
                }
            }
        }
    }

    // Find max for normalization across all channels
    let max_val = accumulators
        .iter()
        .flat_map(|acc| acc.iter().copied())
        .max()
        .unwrap_or(1);

    // Draw each channel with its color
    let colors = [
        [255, 0, 0, 255], // Red
        [0, 255, 0, 255], // Green
        [0, 0, 255, 255], // Blue
    ];

    for (channel, accumulator) in accumulators.iter().enumerate() {
        let offset_x = section_width * channel as u32;

        for y in 0..scope_height {
            for x in 0..section_width {
                let idx = (y * section_width + x) as usize;
                let count = accumulator[idx];

                if count > 0 {
                    let normalized = ((count as f32 / max_val as f32).sqrt() * 255.0) as u8;
                    let color = [
                        ((colors[channel][0] as u16 * u16::from(normalized)) / 255) as u8,
                        ((colors[channel][1] as u16 * u16::from(normalized)) / 255) as u8,
                        ((colors[channel][2] as u16 * u16::from(normalized)) / 255) as u8,
                        255,
                    ];
                    canvas.set_pixel(offset_x + x, y, color);
                }
            }
        }
    }

    // Draw graticule
    if config.show_graticule {
        crate::render::draw_parade_graticule(&mut canvas, config, 3);
    }

    // Draw labels
    if config.show_labels {
        draw_parade_labels(&mut canvas, &["R", "G", "B"]);
    }

    Ok(ScopeData {
        width: scope_width,
        height: scope_height,
        data: canvas.data,
        scope_type: ScopeType::WaveformRgbParade,
    })
}

/// Generates an RGB overlay waveform (all channels overlaid with color).
///
/// All RGB channels are displayed on the same scope, with each channel
/// shown in its respective color. White indicates areas where all channels align.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn generate_rgb_overlay(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &ScopeConfig,
) -> OxiResult<ScopeData> {
    let expected_size = (width * height * 3) as usize;
    if frame.len() < expected_size {
        return Err(oximedia_core::OxiError::InvalidData(format!(
            "Frame data too small: expected {expected_size}, got {}",
            frame.len()
        )));
    }

    let scope_width = config.width;
    let scope_height = config.height;

    let mut canvas = Canvas::new(scope_width, scope_height);

    // Three separate accumulators for R, G, B
    let mut accumulators = [
        vec![0u32; (scope_width * scope_height) as usize],
        vec![0u32; (scope_width * scope_height) as usize],
        vec![0u32; (scope_width * scope_height) as usize],
    ];

    // Process frame
    for y in 0..height {
        for x in 0..width {
            let pixel_idx = ((y * width + x) * 3) as usize;
            let rgb = [frame[pixel_idx], frame[pixel_idx + 1], frame[pixel_idx + 2]];

            let scope_x = (x * scope_width) / width;

            for (channel, &value) in rgb.iter().enumerate() {
                let mapped = ((u32::from(value) * scope_height) / 255).min(scope_height - 1);
                let scope_y = scope_height - 1 - mapped;
                let idx = (scope_y * scope_width + scope_x) as usize;

                if idx < accumulators[channel].len() {
                    accumulators[channel][idx] = accumulators[channel][idx].saturating_add(1);
                }
            }
        }
    }

    // Find max for normalization
    let max_val = accumulators
        .iter()
        .flat_map(|acc| acc.iter().copied())
        .max()
        .unwrap_or(1);

    // Composite all channels with additive blending
    for y in 0..scope_height {
        for x in 0..scope_width {
            let idx = (y * scope_width + x) as usize;

            let r_count = accumulators[0][idx];
            let g_count = accumulators[1][idx];
            let b_count = accumulators[2][idx];

            if r_count > 0 || g_count > 0 || b_count > 0 {
                let r = ((r_count as f32 / max_val as f32).sqrt() * 255.0) as u8;
                let g = ((g_count as f32 / max_val as f32).sqrt() * 255.0) as u8;
                let b = ((b_count as f32 / max_val as f32).sqrt() * 255.0) as u8;

                canvas.set_pixel(x, y, [r, g, b, 255]);
            }
        }
    }

    // Draw graticule
    if config.show_graticule {
        crate::render::draw_waveform_graticule(&mut canvas, config);
    }

    // Draw labels
    if config.show_labels {
        draw_waveform_labels(&mut canvas);
    }

    Ok(ScopeData {
        width: scope_width,
        height: scope_height,
        data: canvas.data,
        scope_type: ScopeType::WaveformRgbOverlay,
    })
}

/// Generates a YCbCr waveform (Y|Cb|Cr parade).
///
/// Displays the Y (luma), Cb (blue-difference), and Cr (red-difference)
/// components side-by-side for analyzing color information separately.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn generate_ycbcr_waveform(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &ScopeConfig,
) -> OxiResult<ScopeData> {
    let expected_size = (width * height * 3) as usize;
    if frame.len() < expected_size {
        return Err(oximedia_core::OxiError::InvalidData(format!(
            "Frame data too small: expected {expected_size}, got {}",
            frame.len()
        )));
    }

    let scope_width = config.width;
    let scope_height = config.height;
    let section_width = scope_width / 3;

    let mut canvas = Canvas::new(scope_width, scope_height);

    // Three accumulators for Y, Cb, Cr
    let mut accumulators = [
        vec![0u32; (section_width * scope_height) as usize],
        vec![0u32; (section_width * scope_height) as usize],
        vec![0u32; (section_width * scope_height) as usize],
    ];

    // Process frame
    for y in 0..height {
        for x in 0..width {
            let pixel_idx = ((y * width + x) * 3) as usize;
            let r = frame[pixel_idx];
            let g = frame[pixel_idx + 1];
            let b = frame[pixel_idx + 2];

            let (luma, cb, cr) = rgb_to_ycbcr(r, g, b);
            let ycbcr = [luma, cb, cr];

            let scope_x = (x * section_width) / width;

            for (channel, &value) in ycbcr.iter().enumerate() {
                let mapped = ((u32::from(value) * scope_height) / 255).min(scope_height - 1);
                let scope_y = scope_height - 1 - mapped;
                let idx = (scope_y * section_width + scope_x) as usize;

                if idx < accumulators[channel].len() {
                    accumulators[channel][idx] = accumulators[channel][idx].saturating_add(1);
                }
            }
        }
    }

    // Find max for normalization
    let max_val = accumulators
        .iter()
        .flat_map(|acc| acc.iter().copied())
        .max()
        .unwrap_or(1);

    // Draw each component in grayscale
    for (channel, accumulator) in accumulators.iter().enumerate() {
        let offset_x = section_width * channel as u32;

        for y in 0..scope_height {
            for x in 0..section_width {
                let idx = (y * section_width + x) as usize;
                let count = accumulator[idx];

                if count > 0 {
                    let normalized = ((count as f32 / max_val as f32).sqrt() * 255.0) as u8;
                    canvas.set_pixel(offset_x + x, y, [normalized, normalized, normalized, 255]);
                }
            }
        }
    }

    // Draw graticule
    if config.show_graticule {
        crate::render::draw_parade_graticule(&mut canvas, config, 3);
    }

    // Draw labels
    if config.show_labels {
        draw_parade_labels(&mut canvas, &["Y", "Cb", "Cr"]);
    }

    Ok(ScopeData {
        width: scope_width,
        height: scope_height,
        data: canvas.data,
        scope_type: ScopeType::WaveformYcbcr,
    })
}

/// Draws IRE labels on waveform display.
fn draw_waveform_labels(canvas: &mut Canvas) {
    let height = canvas.height;
    let color = crate::render::colors::WHITE;

    // Draw IRE level labels
    let labels = [(100, "100"), (75, "75"), (50, "50"), (0, "0")];

    for (ire, text) in &labels {
        let y = height - ((*ire as u32 * height) / 100);
        if y >= 8 && y < height {
            crate::render::draw_label(canvas, 2, y - 8, text, color);
        }
    }
}

/// Draws labels for parade display.
fn draw_parade_labels(canvas: &mut Canvas, labels: &[&str]) {
    let width = canvas.width;
    let section_width = width / labels.len() as u32;
    let color = crate::render::colors::WHITE;

    for (i, label) in labels.iter().enumerate() {
        let x = section_width * i as u32 + section_width / 2 - 3;
        crate::render::draw_label(canvas, x, 2, label, color);
    }
}

/// Detects out-of-gamut pixels in the frame.
///
/// Returns true if any pixels are outside legal broadcast range (16-235 for Y, 16-240 for `CbCr`).
#[must_use]
pub fn detect_out_of_gamut(frame: &[u8], width: u32, height: u32) -> bool {
    for y in 0..height {
        for x in 0..width {
            let pixel_idx = ((y * width + x) * 3) as usize;
            if pixel_idx + 2 >= frame.len() {
                return false;
            }

            let r = frame[pixel_idx];
            let g = frame[pixel_idx + 1];
            let b = frame[pixel_idx + 2];

            let (luma, cb, cr) = rgb_to_ycbcr(r, g, b);

            // Check legal broadcast levels (ITU-R BT.709)
            // Y: 16-235, CbCr: 16-240
            if !(16..=235).contains(&luma) || !(16..=240).contains(&cb) || !(16..=240).contains(&cr)
            {
                return true;
            }
        }
    }
    false
}

/// Calculates waveform statistics.
#[derive(Debug, Clone)]
pub struct WaveformStats {
    /// Average luminance (0-255).
    pub avg_luma: f32,

    /// Minimum luminance (0-255).
    pub min_luma: u8,

    /// Maximum luminance (0-255).
    pub max_luma: u8,

    /// Standard deviation of luminance.
    pub std_dev: f32,

    /// Percentage of pixels at or near black (< 16).
    pub black_clip_percent: f32,

    /// Percentage of pixels at or near white (> 235).
    pub white_clip_percent: f32,
}

/// Computes waveform statistics from frame data.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn compute_waveform_stats(frame: &[u8], width: u32, height: u32) -> WaveformStats {
    let mut sum = 0u64;
    let mut min = 255u8;
    let mut max = 0u8;
    let mut black_clip_count = 0u32;
    let mut white_clip_count = 0u32;
    let pixel_count = width * height;

    // First pass: compute mean, min, max, clipping
    for y in 0..height {
        for x in 0..width {
            let pixel_idx = ((y * width + x) * 3) as usize;
            if pixel_idx + 2 >= frame.len() {
                break;
            }

            let r = frame[pixel_idx];
            let g = frame[pixel_idx + 1];
            let b = frame[pixel_idx + 2];

            let (luma, _, _) = rgb_to_ycbcr(r, g, b);

            sum += u64::from(luma);
            min = min.min(luma);
            max = max.max(luma);

            if luma < 16 {
                black_clip_count += 1;
            }
            if luma > 235 {
                white_clip_count += 1;
            }
        }
    }

    let avg_luma = sum as f32 / pixel_count as f32;

    // Second pass: compute standard deviation
    let mut variance_sum = 0.0f32;
    for y in 0..height {
        for x in 0..width {
            let pixel_idx = ((y * width + x) * 3) as usize;
            if pixel_idx + 2 >= frame.len() {
                break;
            }

            let r = frame[pixel_idx];
            let g = frame[pixel_idx + 1];
            let b = frame[pixel_idx + 2];

            let (luma, _, _) = rgb_to_ycbcr(r, g, b);
            let diff = f32::from(luma) - avg_luma;
            variance_sum += diff * diff;
        }
    }

    let std_dev = (variance_sum / pixel_count as f32).sqrt();

    WaveformStats {
        avg_luma,
        min_luma: min,
        max_luma: max,
        std_dev,
        black_clip_percent: (black_clip_count as f32 / pixel_count as f32) * 100.0,
        white_clip_percent: (white_clip_count as f32 / pixel_count as f32) * 100.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frame(width: u32, height: u32) -> Vec<u8> {
        let mut frame = vec![0u8; (width * height * 3) as usize];

        // Create gradient from black to white
        for y in 0..height {
            let value = ((y * 255) / height) as u8;
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                frame[idx] = value;
                frame[idx + 1] = value;
                frame[idx + 2] = value;
            }
        }

        frame
    }

    fn small_scope_config() -> ScopeConfig {
        ScopeConfig {
            width: 64,
            height: 64,
            ..ScopeConfig::default()
        }
    }

    #[test]
    fn test_generate_luma_waveform() {
        let frame = create_test_frame(32, 32);
        let config = small_scope_config();

        let result = generate_luma_waveform(&frame, 32, 32, &config);
        assert!(result.is_ok());

        let scope = result.expect("should succeed in test");
        assert_eq!(scope.width, config.width);
        assert_eq!(scope.height, config.height);
        assert_eq!(scope.scope_type, ScopeType::WaveformLuma);
    }

    #[test]
    fn test_generate_rgb_parade() {
        let frame = create_test_frame(32, 32);
        let config = small_scope_config();

        let result = generate_rgb_parade(&frame, 32, 32, &config);
        assert!(result.is_ok());

        let scope = result.expect("should succeed in test");
        assert_eq!(scope.width, config.width);
        assert_eq!(scope.height, config.height);
    }

    #[test]
    fn test_generate_rgb_overlay() {
        let frame = create_test_frame(32, 32);
        let config = small_scope_config();

        let result = generate_rgb_overlay(&frame, 32, 32, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_ycbcr_waveform() {
        let frame = create_test_frame(32, 32);
        let config = small_scope_config();

        let result = generate_ycbcr_waveform(&frame, 32, 32, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_frame_size() {
        let frame = vec![0u8; 100]; // Too small
        let config = small_scope_config();

        let result = generate_luma_waveform(&frame, 32, 32, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_out_of_gamut() {
        // Create frame with legal values (mid-gray, safe)
        let legal_frame = vec![128u8; 10 * 10 * 3];
        // This should convert to legal Y/Cb/Cr values
        assert!(!detect_out_of_gamut(&legal_frame, 10, 10));

        // Create frame with illegal values (pure black RGB = 0,0,0)
        // Pure black converts to Y=0, Cb=128, Cr=128 which has Y < 16 (illegal)
        let illegal_frame = vec![0u8; 10 * 10 * 3];
        assert!(detect_out_of_gamut(&illegal_frame, 10, 10));
    }

    #[test]
    fn test_compute_waveform_stats() {
        let frame = create_test_frame(32, 32);
        let stats = compute_waveform_stats(&frame, 32, 32);

        assert!(stats.avg_luma > 0.0);
        assert!(stats.avg_luma < 255.0);
        assert!(stats.min_luma <= stats.max_luma);
        assert!(stats.std_dev >= 0.0);
    }
}
