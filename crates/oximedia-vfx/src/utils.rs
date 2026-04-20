//! Utility functions for VFX processing.

use crate::{Color, Frame, VfxResult};

/// Image processing utilities.
pub struct ImageUtils;

impl ImageUtils {
    /// Calculate luminance from RGB.
    #[must_use]
    pub fn luminance(r: u8, g: u8, b: u8) -> f32 {
        0.299 * f32::from(r) + 0.587 * f32::from(g) + 0.114 * f32::from(b)
    }

    /// Convert RGB to HSV.
    #[must_use]
    pub fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
        let r = f32::from(r) / 255.0;
        let g = f32::from(g) / 255.0;
        let b = f32::from(b) / 255.0;

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let delta = max - min;

        let h = if delta == 0.0 {
            0.0
        } else if max == r {
            60.0 * (((g - b) / delta) % 6.0)
        } else if max == g {
            60.0 * (((b - r) / delta) + 2.0)
        } else {
            60.0 * (((r - g) / delta) + 4.0)
        };

        let s = if max == 0.0 { 0.0 } else { delta / max };
        let v = max;

        (h, s, v)
    }

    /// Convert HSV to RGB.
    #[must_use]
    pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
        let c = v * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = v - c;

        let (r, g, b) = if h < 60.0 {
            (c, x, 0.0)
        } else if h < 120.0 {
            (x, c, 0.0)
        } else if h < 180.0 {
            (0.0, c, x)
        } else if h < 240.0 {
            (0.0, x, c)
        } else if h < 300.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };

        (
            ((r + m) * 255.0) as u8,
            ((g + m) * 255.0) as u8,
            ((b + m) * 255.0) as u8,
        )
    }

    /// Apply gamma correction.
    #[must_use]
    pub fn gamma_correct(value: u8, gamma: f32) -> u8 {
        let normalized = f32::from(value) / 255.0;
        let corrected = normalized.powf(1.0 / gamma);
        (corrected * 255.0) as u8
    }

    /// Clamp value to range.
    #[must_use]
    pub fn clamp_u8(value: i32) -> u8 {
        value.clamp(0, 255) as u8
    }

    /// Lerp between two values.
    #[must_use]
    pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t.clamp(0.0, 1.0)
    }

    /// Bilinear interpolation.
    #[must_use]
    pub fn bilinear(p00: f32, p10: f32, p01: f32, p11: f32, tx: f32, ty: f32) -> f32 {
        let a = Self::lerp(p00, p10, tx);
        let b = Self::lerp(p01, p11, tx);
        Self::lerp(a, b, ty)
    }

    /// Calculate image histogram.
    #[must_use]
    pub fn histogram(frame: &Frame, channel: usize) -> [u32; 256] {
        let mut hist = [0u32; 256];

        for y in 0..frame.height {
            for x in 0..frame.width {
                if let Some(pixel) = frame.get_pixel(x, y) {
                    if channel < 4 {
                        hist[pixel[channel] as usize] += 1;
                    }
                }
            }
        }

        hist
    }

    /// Auto-level frame based on histogram.
    pub fn auto_level(frame: &mut Frame) -> VfxResult<()> {
        // Calculate histograms for each channel
        let hist_r = Self::histogram(frame, 0);
        let hist_g = Self::histogram(frame, 1);
        let hist_b = Self::histogram(frame, 2);

        // Find min/max for each channel
        let (min_r, max_r) = Self::find_histogram_range(&hist_r, 0.01);
        let (min_g, max_g) = Self::find_histogram_range(&hist_g, 0.01);
        let (min_b, max_b) = Self::find_histogram_range(&hist_b, 0.01);

        // Apply level adjustment
        for y in 0..frame.height {
            for x in 0..frame.width {
                if let Some(pixel) = frame.get_pixel(x, y) {
                    let r = Self::level_adjust(pixel[0], min_r, max_r);
                    let g = Self::level_adjust(pixel[1], min_g, max_g);
                    let b = Self::level_adjust(pixel[2], min_b, max_b);
                    frame.set_pixel(x, y, [r, g, b, pixel[3]]);
                }
            }
        }

        Ok(())
    }

    fn find_histogram_range(hist: &[u32; 256], clip_percent: f32) -> (u8, u8) {
        let total: u32 = hist.iter().sum();
        let clip_count = (total as f32 * clip_percent) as u32;

        let mut acc = 0;
        let mut min = 0u8;
        for (i, &count) in hist.iter().enumerate() {
            acc += count;
            if acc >= clip_count {
                min = i as u8;
                break;
            }
        }

        acc = 0;
        let mut max = 255u8;
        for (i, &count) in hist.iter().enumerate().rev() {
            acc += count;
            if acc >= clip_count {
                max = i as u8;
                break;
            }
        }

        (min, max)
    }

    fn level_adjust(value: u8, min: u8, max: u8) -> u8 {
        if max <= min {
            return value;
        }

        let range = max - min;
        let adjusted = (f32::from(value.saturating_sub(min)) / f32::from(range)) * 255.0;
        adjusted.clamp(0.0, 255.0) as u8
    }

    /// Equalize histogram.
    pub fn equalize_histogram(frame: &mut Frame) -> VfxResult<()> {
        // Calculate cumulative distribution
        let hist = Self::histogram(frame, 0); // Use luminance
        let total = (frame.width * frame.height) as f32;

        let mut cdf = [0f32; 256];
        let mut acc = 0u32;
        for (i, &count) in hist.iter().enumerate() {
            acc += count;
            cdf[i] = acc as f32 / total;
        }

        // Apply equalization
        for y in 0..frame.height {
            for x in 0..frame.width {
                if let Some(pixel) = frame.get_pixel(x, y) {
                    let lum = Self::luminance(pixel[0], pixel[1], pixel[2]);
                    let idx = lum.clamp(0.0, 255.0) as usize;
                    let eq = (cdf[idx] * 255.0) as u8;

                    // Preserve color ratios
                    let scale = if lum > 0.0 { f32::from(eq) / lum } else { 1.0 };

                    let r = (f32::from(pixel[0]) * scale).clamp(0.0, 255.0) as u8;
                    let g = (f32::from(pixel[1]) * scale).clamp(0.0, 255.0) as u8;
                    let b = (f32::from(pixel[2]) * scale).clamp(0.0, 255.0) as u8;

                    frame.set_pixel(x, y, [r, g, b, pixel[3]]);
                }
            }
        }

        Ok(())
    }
}

/// Color manipulation utilities.
pub struct ColorUtils;

impl ColorUtils {
    /// Blend two colors with alpha.
    #[must_use]
    pub fn blend(bottom: Color, top: Color) -> Color {
        bottom.blend(top)
    }

    /// Multiply blend mode.
    #[must_use]
    pub fn multiply(bottom: Color, top: Color) -> Color {
        Color::new(
            ((f32::from(bottom.r) * f32::from(top.r)) / 255.0) as u8,
            ((f32::from(bottom.g) * f32::from(top.g)) / 255.0) as u8,
            ((f32::from(bottom.b) * f32::from(top.b)) / 255.0) as u8,
            bottom.a,
        )
    }

    /// Screen blend mode.
    #[must_use]
    pub fn screen(bottom: Color, top: Color) -> Color {
        Color::new(
            (255.0 - (255.0 - f32::from(bottom.r)) * (255.0 - f32::from(top.r)) / 255.0) as u8,
            (255.0 - (255.0 - f32::from(bottom.g)) * (255.0 - f32::from(top.g)) / 255.0) as u8,
            (255.0 - (255.0 - f32::from(bottom.b)) * (255.0 - f32::from(top.b)) / 255.0) as u8,
            bottom.a,
        )
    }

    /// Overlay blend mode.
    #[must_use]
    pub fn overlay(bottom: Color, top: Color) -> Color {
        fn overlay_channel(a: u8, b: u8) -> u8 {
            if a < 128 {
                ((2.0 * f32::from(a) * f32::from(b)) / 255.0) as u8
            } else {
                (255.0 - 2.0 * (255.0 - f32::from(a)) * (255.0 - f32::from(b)) / 255.0) as u8
            }
        }

        Color::new(
            overlay_channel(bottom.r, top.r),
            overlay_channel(bottom.g, top.g),
            overlay_channel(bottom.b, top.b),
            bottom.a,
        )
    }

    /// Adjust brightness.
    #[must_use]
    pub fn brightness(color: Color, amount: f32) -> Color {
        let amount = amount.clamp(-1.0, 1.0);
        let delta = (amount * 255.0) as i16;

        Color::new(
            (i16::from(color.r) + delta).clamp(0, 255) as u8,
            (i16::from(color.g) + delta).clamp(0, 255) as u8,
            (i16::from(color.b) + delta).clamp(0, 255) as u8,
            color.a,
        )
    }

    /// Adjust contrast.
    #[must_use]
    pub fn contrast(color: Color, amount: f32) -> Color {
        let factor = (amount + 1.0).max(0.0);

        fn adjust(value: u8, factor: f32) -> u8 {
            let v = f32::from(value) / 255.0;
            let adjusted = ((v - 0.5) * factor + 0.5) * 255.0;
            adjusted.clamp(0.0, 255.0) as u8
        }

        Color::new(
            adjust(color.r, factor),
            adjust(color.g, factor),
            adjust(color.b, factor),
            color.a,
        )
    }

    /// Adjust saturation.
    #[must_use]
    pub fn saturation(color: Color, amount: f32) -> Color {
        let (h, s, v) = ImageUtils::rgb_to_hsv(color.r, color.g, color.b);
        let new_s = (s * (amount + 1.0)).clamp(0.0, 1.0);
        let (r, g, b) = ImageUtils::hsv_to_rgb(h, new_s, v);
        Color::new(r, g, b, color.a)
    }

    /// Invert color.
    #[must_use]
    pub fn invert(color: Color) -> Color {
        Color::new(255 - color.r, 255 - color.g, 255 - color.b, color.a)
    }

    /// Grayscale conversion.
    #[must_use]
    pub fn grayscale(color: Color) -> Color {
        let gray = ImageUtils::luminance(color.r, color.g, color.b) as u8;
        Color::new(gray, gray, gray, color.a)
    }

    /// Sepia tone.
    #[must_use]
    pub fn sepia(color: Color) -> Color {
        let r = f32::from(color.r);
        let g = f32::from(color.g);
        let b = f32::from(color.b);

        let tr = (r * 0.393 + g * 0.769 + b * 0.189).min(255.0) as u8;
        let tg = (r * 0.349 + g * 0.686 + b * 0.168).min(255.0) as u8;
        let tb = (r * 0.272 + g * 0.534 + b * 0.131).min(255.0) as u8;

        Color::new(tr, tg, tb, color.a)
    }
}

/// Kernel utilities for convolution operations.
pub struct KernelUtils;

impl KernelUtils {
    /// Create Gaussian kernel.
    #[must_use]
    pub fn gaussian_kernel(size: usize, sigma: f32) -> Vec<Vec<f32>> {
        let mut kernel = vec![vec![0.0; size]; size];
        let center = size as f32 / 2.0;
        let mut sum = 0.0;

        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - center;
                let dy = y as f32 - center;
                let value = (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
                kernel[y][x] = value;
                sum += value;
            }
        }

        // Normalize
        for row in &mut kernel {
            for value in row {
                *value /= sum;
            }
        }

        kernel
    }

    /// Create box blur kernel.
    #[must_use]
    pub fn box_kernel(size: usize) -> Vec<Vec<f32>> {
        let value = 1.0 / (size * size) as f32;
        vec![vec![value; size]; size]
    }

    /// Create sharpen kernel.
    #[must_use]
    pub fn sharpen_kernel() -> Vec<Vec<f32>> {
        vec![
            vec![0.0, -1.0, 0.0],
            vec![-1.0, 5.0, -1.0],
            vec![0.0, -1.0, 0.0],
        ]
    }

    /// Create edge detection kernel (Sobel X).
    #[must_use]
    pub fn sobel_x_kernel() -> Vec<Vec<f32>> {
        vec![
            vec![-1.0, 0.0, 1.0],
            vec![-2.0, 0.0, 2.0],
            vec![-1.0, 0.0, 1.0],
        ]
    }

    /// Create edge detection kernel (Sobel Y).
    #[must_use]
    pub fn sobel_y_kernel() -> Vec<Vec<f32>> {
        vec![
            vec![-1.0, -2.0, -1.0],
            vec![0.0, 0.0, 0.0],
            vec![1.0, 2.0, 1.0],
        ]
    }

    /// Apply kernel to frame.
    ///
    /// # Errors
    ///
    /// Returns an error if a new frame cannot be allocated for the output.
    pub fn apply_kernel(frame: &Frame, kernel: &[Vec<f32>]) -> VfxResult<Frame> {
        let mut output = Frame::new(frame.width, frame.height)?;
        let k_size = kernel.len();
        let k_half = k_size as i32 / 2;

        for y in 0..frame.height {
            for x in 0..frame.width {
                let mut r = 0.0_f32;
                let mut g = 0.0_f32;
                let mut b = 0.0_f32;

                for ky in 0..k_size {
                    for kx in 0..k_size {
                        let px = (x as i32 + kx as i32 - k_half)
                            .max(0)
                            .min(frame.width as i32 - 1) as u32;
                        let py = (y as i32 + ky as i32 - k_half)
                            .max(0)
                            .min(frame.height as i32 - 1) as u32;

                        if let Some(pixel) = frame.get_pixel(px, py) {
                            let k = kernel[ky][kx];
                            r += f32::from(pixel[0]) * k;
                            g += f32::from(pixel[1]) * k;
                            b += f32::from(pixel[2]) * k;
                        }
                    }
                }

                output.set_pixel(
                    x,
                    y,
                    [
                        r.clamp(0.0, 255.0) as u8,
                        g.clamp(0.0, 255.0) as u8,
                        b.clamp(0.0, 255.0) as u8,
                        255,
                    ],
                );
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_luminance() {
        let lum = ImageUtils::luminance(255, 255, 255);
        assert!((lum - 255.0).abs() < 1.0);

        let lum = ImageUtils::luminance(0, 0, 0);
        assert!(lum < 1.0);
    }

    #[test]
    fn test_rgb_hsv_conversion() {
        let (h, s, v) = ImageUtils::rgb_to_hsv(255, 0, 0);
        assert!((h - 0.0).abs() < 1.0);
        assert!((s - 1.0).abs() < 0.01);
        assert!((v - 1.0).abs() < 0.01);

        let (r, g, b) = ImageUtils::hsv_to_rgb(h, s, v);
        assert_eq!(r, 255);
        assert!(g < 5);
        assert!(b < 5);
    }

    #[test]
    fn test_gamma_correction() {
        let corrected = ImageUtils::gamma_correct(128, 2.2);
        assert!(corrected > 128);
    }

    #[test]
    fn test_color_blend_modes() {
        let c1 = Color::rgb(128, 128, 128);
        let c2 = Color::rgb(255, 0, 0);

        let _multiply = ColorUtils::multiply(c1, c2);
        let _screen = ColorUtils::screen(c1, c2);
        let _overlay = ColorUtils::overlay(c1, c2);
    }

    #[test]
    fn test_color_adjustments() {
        let color = Color::rgb(128, 128, 128);

        let bright = ColorUtils::brightness(color, 0.5);
        assert!(bright.r > color.r);

        let _contrasted = ColorUtils::contrast(color, 0.5);
        let inverted = ColorUtils::invert(color);
        assert_eq!(inverted.r, 127);

        let gray = ColorUtils::grayscale(Color::rgb(255, 0, 0));
        assert_eq!(gray.r, gray.g);
        assert_eq!(gray.g, gray.b);
    }

    #[test]
    fn test_kernels() {
        let gaussian = KernelUtils::gaussian_kernel(3, 1.0);
        assert_eq!(gaussian.len(), 3);
        assert_eq!(gaussian[0].len(), 3);

        let sharpen = KernelUtils::sharpen_kernel();
        assert_eq!(sharpen.len(), 3);
    }

    #[test]
    fn test_histogram() {
        let frame = Frame::new(100, 100).expect("should succeed in test");
        let hist = ImageUtils::histogram(&frame, 0);
        assert_eq!(hist.len(), 256);
    }
}
