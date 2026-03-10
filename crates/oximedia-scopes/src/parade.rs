//! Parade displays for multi-channel video analysis.
//!
//! Parade displays show multiple channels (RGB or YCbCr) side-by-side as
//! vertical intensity bars, making it easy to:
//! - Compare channel balance
//! - Identify color casts
//! - Match exposure across channels
//! - Analyze color grading consistency
//!
//! Unlike waveforms (which show horizontal distribution), parades show
//! vertical intensity distribution for each channel independently.

use crate::render::{rgb_to_ycbcr, Canvas};
use crate::{ScopeConfig, ScopeData, ScopeType};
use oximedia_core::OxiResult;

/// Generates an RGB parade display.
///
/// Shows red, green, and blue channels side-by-side as vertical bars.
/// Each bar's height represents the intensity of that channel.
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
#[allow(clippy::cast_precision_loss)]
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

    // Build vertical intensity distributions for each channel
    // For each column in the frame, accumulate pixel values at their intensity levels
    let mut distributions = [
        vec![vec![0u32; 256]; section_width as usize],
        vec![vec![0u32; 256]; section_width as usize],
        vec![vec![0u32; 256]; section_width as usize],
    ];

    for y in 0..height {
        for x in 0..width {
            let pixel_idx = ((y * width + x) * 3) as usize;
            let rgb = [frame[pixel_idx], frame[pixel_idx + 1], frame[pixel_idx + 2]];

            let scope_x = ((u64::from(x) * u64::from(section_width)) / u64::from(width)) as usize;

            for (channel, &value) in rgb.iter().enumerate() {
                if scope_x < distributions[channel].len() {
                    distributions[channel][scope_x][value as usize] += 1;
                }
            }
        }
    }

    // Find max for normalization
    let max_val = distributions
        .iter()
        .flat_map(|dist| dist.iter().flat_map(|col| col.iter()))
        .copied()
        .max()
        .unwrap_or(1);

    // Draw each channel
    let colors = [
        crate::render::colors::RED,
        crate::render::colors::GREEN,
        crate::render::colors::BLUE,
    ];

    for (channel, distribution) in distributions.iter().enumerate() {
        let offset_x = section_width * channel as u32;

        for (scope_x, column) in distribution.iter().enumerate() {
            for (intensity, &count) in column.iter().enumerate() {
                if count > 0 {
                    // Map intensity (0-255) to scope height
                    let mapped = ((intensity as u32 * scope_height) / 255).min(scope_height - 1);
                    let scope_y = scope_height - 1 - mapped;

                    // Calculate brightness based on count
                    let brightness = ((count as f32 / max_val as f32).sqrt() * 255.0) as u8;

                    let color = [
                        ((u16::from(colors[channel][0]) * u16::from(brightness)) / 255) as u8,
                        ((u16::from(colors[channel][1]) * u16::from(brightness)) / 255) as u8,
                        ((u16::from(colors[channel][2]) * u16::from(brightness)) / 255) as u8,
                        255,
                    ];

                    canvas.blend_pixel(offset_x + scope_x as u32, scope_y, color);
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
        scope_type: ScopeType::ParadeRgb,
    })
}

/// Generates a YCbCr parade display.
///
/// Shows Y (luma), Cb (blue chroma), and Cr (red chroma) channels side-by-side.
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
#[allow(clippy::cast_precision_loss)]
pub fn generate_ycbcr_parade(
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

    // Build vertical intensity distributions for Y, Cb, Cr
    let mut distributions = [
        vec![vec![0u32; 256]; section_width as usize],
        vec![vec![0u32; 256]; section_width as usize],
        vec![vec![0u32; 256]; section_width as usize],
    ];

    for y in 0..height {
        for x in 0..width {
            let pixel_idx = ((y * width + x) * 3) as usize;
            let r = frame[pixel_idx];
            let g = frame[pixel_idx + 1];
            let b = frame[pixel_idx + 2];

            let (luma, cb, cr) = rgb_to_ycbcr(r, g, b);
            let ycbcr = [luma, cb, cr];

            let scope_x = ((u64::from(x) * u64::from(section_width)) / u64::from(width)) as usize;

            for (channel, &value) in ycbcr.iter().enumerate() {
                if scope_x < distributions[channel].len() {
                    distributions[channel][scope_x][value as usize] += 1;
                }
            }
        }
    }

    // Find max for normalization
    let max_val = distributions
        .iter()
        .flat_map(|dist| dist.iter().flat_map(|col| col.iter()))
        .copied()
        .max()
        .unwrap_or(1);

    // Draw each component in grayscale
    for (channel, distribution) in distributions.iter().enumerate() {
        let offset_x = section_width * channel as u32;

        for (scope_x, column) in distribution.iter().enumerate() {
            for (intensity, &count) in column.iter().enumerate() {
                if count > 0 {
                    // Map intensity (0-255) to scope height
                    let mapped = ((intensity as u32 * scope_height) / 255).min(scope_height - 1);
                    let scope_y = scope_height - 1 - mapped;

                    // Calculate brightness based on count
                    let brightness = ((count as f32 / max_val as f32).sqrt() * 255.0) as u8;

                    let color = [brightness, brightness, brightness, 255];

                    canvas.blend_pixel(offset_x + scope_x as u32, scope_y, color);
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
        scope_type: ScopeType::ParadeYcbcr,
    })
}

/// Draws labels for parade sections.
fn draw_parade_labels(canvas: &mut Canvas, labels: &[&str]) {
    let width = canvas.width;
    let section_width = width / labels.len() as u32;
    let color = crate::render::colors::WHITE;

    for (i, label) in labels.iter().enumerate() {
        let x = section_width * i as u32 + section_width / 2 - 3;
        crate::render::draw_label(canvas, x, 2, label, color);
    }
}

/// Analyzes channel balance across the frame.
#[derive(Debug, Clone)]
pub struct ChannelBalance {
    /// Red channel average (0-255).
    pub red_avg: f32,

    /// Green channel average (0-255).
    pub green_avg: f32,

    /// Blue channel average (0-255).
    pub blue_avg: f32,

    /// Color cast indicator (-1 to +1 for each axis).
    pub color_cast: (f32, f32, f32),
}

/// Computes channel balance statistics.
///
/// Helps identify color casts and imbalances in the image.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn compute_channel_balance(frame: &[u8], width: u32, height: u32) -> ChannelBalance {
    let mut r_sum = 0u64;
    let mut g_sum = 0u64;
    let mut b_sum = 0u64;
    let pixel_count = width * height;

    for y in 0..height {
        for x in 0..width {
            let pixel_idx = ((y * width + x) * 3) as usize;
            if pixel_idx + 2 >= frame.len() {
                break;
            }

            r_sum += u64::from(frame[pixel_idx]);
            g_sum += u64::from(frame[pixel_idx + 1]);
            b_sum += u64::from(frame[pixel_idx + 2]);
        }
    }

    let red_avg = r_sum as f32 / pixel_count as f32;
    let green_avg = g_sum as f32 / pixel_count as f32;
    let blue_avg = b_sum as f32 / pixel_count as f32;

    // Calculate color cast (normalized to -1 to +1)
    let overall_avg = (red_avg + green_avg + blue_avg) / 3.0;

    let r_cast = (red_avg - overall_avg) / 128.0;
    let g_cast = (green_avg - overall_avg) / 128.0;
    let b_cast = (blue_avg - overall_avg) / 128.0;

    ChannelBalance {
        red_avg,
        green_avg,
        blue_avg,
        color_cast: (r_cast, g_cast, b_cast),
    }
}

/// Detects channel imbalance issues.
///
/// Returns true if there's a significant color cast (> 10% deviation).
#[must_use]
pub fn has_color_cast(balance: &ChannelBalance) -> bool {
    let threshold = 0.1; // 10% deviation

    balance.color_cast.0.abs() > threshold
        || balance.color_cast.1.abs() > threshold
        || balance.color_cast.2.abs() > threshold
}

/// Parade match line statistics.
///
/// Match lines help identify if channels are properly aligned at specific luminance levels.
#[derive(Debug, Clone)]
pub struct MatchLineStats {
    /// Average position of red channel peak (0-255).
    pub red_peak: u8,

    /// Average position of green channel peak (0-255).
    pub green_peak: u8,

    /// Average position of blue channel peak (0-255).
    pub blue_peak: u8,

    /// Whether channels are well-balanced (peaks within 10% of each other).
    pub is_balanced: bool,
}

/// Computes match line statistics for parade display.
#[must_use]
pub fn compute_match_line_stats(frame: &[u8], width: u32, height: u32) -> MatchLineStats {
    // Build histograms for each channel
    let mut histograms = [[0u32; 256]; 3];

    for y in 0..height {
        for x in 0..width {
            let pixel_idx = ((y * width + x) * 3) as usize;
            if pixel_idx + 2 >= frame.len() {
                break;
            }

            histograms[0][frame[pixel_idx] as usize] += 1;
            histograms[1][frame[pixel_idx + 1] as usize] += 1;
            histograms[2][frame[pixel_idx + 2] as usize] += 1;
        }
    }

    // Find peaks for each channel
    let find_peak = |histogram: &[u32; 256]| -> u8 {
        let max_pos = histogram
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map_or(128, |(pos, _)| pos);

        #[allow(clippy::cast_possible_truncation)]
        let result = max_pos as u8;
        result
    };

    let red_peak = find_peak(&histograms[0]);
    let green_peak = find_peak(&histograms[1]);
    let blue_peak = find_peak(&histograms[2]);

    // Check if balanced (peaks within 26 levels = ~10% of 255)
    let max_peak = red_peak.max(green_peak).max(blue_peak);
    let min_peak = red_peak.min(green_peak).min(blue_peak);
    let is_balanced = (max_peak - min_peak) <= 26;

    MatchLineStats {
        red_peak,
        green_peak,
        blue_peak,
        is_balanced,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frame(width: u32, height: u32) -> Vec<u8> {
        let mut frame = vec![0u8; (width * height * 3) as usize];

        // Create gradient
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                let value = ((x * 255) / width) as u8;

                frame[idx] = value;
                frame[idx + 1] = value;
                frame[idx + 2] = value;
            }
        }

        frame
    }

    fn create_color_cast_frame(width: u32, height: u32) -> Vec<u8> {
        let mut frame = vec![0u8; (width * height * 3) as usize];

        // Create frame with red color cast
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                frame[idx] = 200; // Red
                frame[idx + 1] = 100; // Green
                frame[idx + 2] = 100; // Blue
            }
        }

        frame
    }

    #[test]
    fn test_generate_rgb_parade() {
        let frame = create_test_frame(100, 100);
        let config = ScopeConfig::default();

        let result = generate_rgb_parade(&frame, 100, 100, &config);
        assert!(result.is_ok());

        let scope = result.expect("should succeed in test");
        assert_eq!(scope.width, config.width);
        assert_eq!(scope.height, config.height);
        assert_eq!(scope.scope_type, ScopeType::ParadeRgb);
    }

    #[test]
    fn test_generate_ycbcr_parade() {
        let frame = create_test_frame(100, 100);
        let config = ScopeConfig::default();

        let result = generate_ycbcr_parade(&frame, 100, 100, &config);
        assert!(result.is_ok());

        let scope = result.expect("should succeed in test");
        assert_eq!(scope.scope_type, ScopeType::ParadeYcbcr);
    }

    #[test]
    fn test_compute_channel_balance() {
        let frame = create_test_frame(50, 50);
        let balance = compute_channel_balance(&frame, 50, 50);

        // Neutral frame should have similar averages
        assert!((balance.red_avg - balance.green_avg).abs() < 5.0);
        assert!((balance.red_avg - balance.blue_avg).abs() < 5.0);
    }

    #[test]
    fn test_color_cast_detection() {
        let frame = create_color_cast_frame(50, 50);
        let balance = compute_channel_balance(&frame, 50, 50);

        // Should detect red color cast
        assert!(has_color_cast(&balance));
        assert!(balance.red_avg > balance.green_avg);
    }

    #[test]
    fn test_compute_match_line_stats() {
        let frame = create_test_frame(50, 50);
        let stats = compute_match_line_stats(&frame, 50, 50);

        // Neutral frame should have balanced peaks
        assert!(stats.is_balanced);
    }

    #[test]
    fn test_unbalanced_match_lines() {
        let mut frame = vec![0u8; 50 * 50 * 3];

        // Red channel peaked at 200, green at 100, blue at 50
        for i in (0..frame.len()).step_by(3) {
            frame[i] = 200;
            frame[i + 1] = 100;
            frame[i + 2] = 50;
        }

        let stats = compute_match_line_stats(&frame, 50, 50);

        // Should detect imbalance
        assert!(!stats.is_balanced);
        assert!(stats.red_peak > stats.green_peak);
        assert!(stats.green_peak > stats.blue_peak);
    }

    #[test]
    fn test_invalid_frame_size() {
        let frame = vec![0u8; 100]; // Too small
        let config = ScopeConfig::default();

        let result = generate_rgb_parade(&frame, 100, 100, &config);
        assert!(result.is_err());

        let result = generate_ycbcr_parade(&frame, 100, 100, &config);
        assert!(result.is_err());
    }
}
