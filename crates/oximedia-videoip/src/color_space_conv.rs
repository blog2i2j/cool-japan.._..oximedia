#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]

//! Network-oriented color space conversion for video transport.
//!
//! This module provides lightweight color space conversions used during
//! video-over-IP encoding and decoding. It handles BT.601/BT.709/BT.2020
//! YCbCr <-> RGB conversions with proper clamping and rounding, optimized
//! for real-time streaming pipelines.

/// Color space standard for YCbCr conversions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorStandard {
    /// ITU-R BT.601 (SD video).
    Bt601,
    /// ITU-R BT.709 (HD video).
    Bt709,
    /// ITU-R BT.2020 (UHD/HDR video).
    Bt2020,
}

impl ColorStandard {
    /// Returns a human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Bt601 => "BT.601",
            Self::Bt709 => "BT.709",
            Self::Bt2020 => "BT.2020",
        }
    }

    /// Get the luma coefficients (Kr, Kg, Kb) for this standard.
    pub fn luma_coefficients(self) -> (f64, f64, f64) {
        match self {
            Self::Bt601 => (0.299, 0.587, 0.114),
            Self::Bt709 => (0.2126, 0.7152, 0.0722),
            Self::Bt2020 => (0.2627, 0.6780, 0.0593),
        }
    }
}

/// Range mode for YCbCr values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeMode {
    /// Limited/studio range (Y: 16-235, Cb/Cr: 16-240 for 8-bit).
    Limited,
    /// Full range (Y: 0-255, Cb/Cr: 0-255 for 8-bit).
    Full,
}

impl RangeMode {
    /// Returns a human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Limited => "limited (studio)",
            Self::Full => "full",
        }
    }
}

/// An RGB pixel (8-bit per channel).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RgbPixel {
    /// Red channel.
    pub r: u8,
    /// Green channel.
    pub g: u8,
    /// Blue channel.
    pub b: u8,
}

impl RgbPixel {
    /// Create a new RGB pixel.
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }
}

/// A YCbCr pixel (8-bit per component).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct YcbcrPixel {
    /// Luma (Y).
    pub y: u8,
    /// Blue-difference chroma (Cb/U).
    pub cb: u8,
    /// Red-difference chroma (Cr/V).
    pub cr: u8,
}

impl YcbcrPixel {
    /// Create a new YCbCr pixel.
    pub fn new(y: u8, cb: u8, cr: u8) -> Self {
        Self { y, cb, cr }
    }
}

/// Clamp a floating-point value to u8 range.
fn clamp_u8(v: f64) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

/// Convert an RGB pixel to YCbCr.
pub fn rgb_to_ycbcr(pixel: RgbPixel, standard: ColorStandard, range: RangeMode) -> YcbcrPixel {
    let (kr, _kg, kb) = standard.luma_coefficients();
    let r = f64::from(pixel.r);
    let g = f64::from(pixel.g);
    let b = f64::from(pixel.b);

    // Compute Y, Cb, Cr in full range [0,255]
    let y_full = kr * r + (1.0 - kr - kb) * g + kb * b;
    let cb_full = (b - y_full) / (2.0 * (1.0 - kb)) + 128.0;
    let cr_full = (r - y_full) / (2.0 * (1.0 - kr)) + 128.0;

    match range {
        RangeMode::Full => YcbcrPixel {
            y: clamp_u8(y_full),
            cb: clamp_u8(cb_full),
            cr: clamp_u8(cr_full),
        },
        RangeMode::Limited => {
            // Scale to limited range: Y [16,235], Cb/Cr [16,240]
            let y_lim = 16.0 + y_full * (219.0 / 255.0);
            let cb_lim = 16.0 + (cb_full - 128.0) * (224.0 / 255.0) + 112.0;
            let cr_lim = 16.0 + (cr_full - 128.0) * (224.0 / 255.0) + 112.0;
            YcbcrPixel {
                y: clamp_u8(y_lim),
                cb: clamp_u8(cb_lim),
                cr: clamp_u8(cr_lim),
            }
        }
    }
}

/// Convert a YCbCr pixel to RGB.
pub fn ycbcr_to_rgb(pixel: YcbcrPixel, standard: ColorStandard, range: RangeMode) -> RgbPixel {
    let (kr, _kg, kb) = standard.luma_coefficients();

    let (y_f, cb_f, cr_f) = match range {
        RangeMode::Full => (
            f64::from(pixel.y),
            f64::from(pixel.cb) - 128.0,
            f64::from(pixel.cr) - 128.0,
        ),
        RangeMode::Limited => {
            let y = (f64::from(pixel.y) - 16.0) * (255.0 / 219.0);
            let cb = (f64::from(pixel.cb) - 128.0) * (255.0 / 224.0);
            let cr = (f64::from(pixel.cr) - 128.0) * (255.0 / 224.0);
            (y, cb, cr)
        }
    };

    let r = y_f + 2.0 * (1.0 - kr) * cr_f;
    let g = y_f
        - 2.0 * kb * (1.0 - kb) / (1.0 - kr - kb) * cb_f
        - 2.0 * kr * (1.0 - kr) / (1.0 - kr - kb) * cr_f;
    let b = y_f + 2.0 * (1.0 - kb) * cb_f;

    RgbPixel {
        r: clamp_u8(r),
        g: clamp_u8(g),
        b: clamp_u8(b),
    }
}

/// Batch-convert RGB buffer to YCbCr buffer (packed RGB -> packed YCbCr).
///
/// Input buffer must have length divisible by 3.
pub fn batch_rgb_to_ycbcr(rgb: &[u8], standard: ColorStandard, range: RangeMode) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgb.len());
    let mut i = 0;
    while i + 2 < rgb.len() {
        let p = rgb_to_ycbcr(
            RgbPixel::new(rgb[i], rgb[i + 1], rgb[i + 2]),
            standard,
            range,
        );
        out.push(p.y);
        out.push(p.cb);
        out.push(p.cr);
        i += 3;
    }
    out
}

/// Batch-convert YCbCr buffer to RGB buffer (packed YCbCr -> packed RGB).
///
/// Input buffer must have length divisible by 3.
pub fn batch_ycbcr_to_rgb(ycbcr: &[u8], standard: ColorStandard, range: RangeMode) -> Vec<u8> {
    let mut out = Vec::with_capacity(ycbcr.len());
    let mut i = 0;
    while i + 2 < ycbcr.len() {
        let p = ycbcr_to_rgb(
            YcbcrPixel::new(ycbcr[i], ycbcr[i + 1], ycbcr[i + 2]),
            standard,
            range,
        );
        out.push(p.r);
        out.push(p.g);
        out.push(p.b);
        i += 3;
    }
    out
}

/// Detect the likely color standard from resolution dimensions.
pub fn detect_standard(width: u32, height: u32) -> ColorStandard {
    if width > 1920 || height > 1080 {
        ColorStandard::Bt2020
    } else if width > 720 || height > 576 {
        ColorStandard::Bt709
    } else {
        ColorStandard::Bt601
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_standard_name() {
        assert_eq!(ColorStandard::Bt601.name(), "BT.601");
        assert_eq!(ColorStandard::Bt709.name(), "BT.709");
        assert_eq!(ColorStandard::Bt2020.name(), "BT.2020");
    }

    #[test]
    fn test_range_mode_name() {
        assert_eq!(RangeMode::Limited.name(), "limited (studio)");
        assert_eq!(RangeMode::Full.name(), "full");
    }

    #[test]
    fn test_luma_coefficients() {
        let (kr, kg, kb) = ColorStandard::Bt709.luma_coefficients();
        assert!((kr + kg + kb - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_black_rgb_to_ycbcr() {
        let black = RgbPixel::new(0, 0, 0);
        let ycbcr = rgb_to_ycbcr(black, ColorStandard::Bt709, RangeMode::Full);
        assert_eq!(ycbcr.y, 0);
        assert_eq!(ycbcr.cb, 128);
        assert_eq!(ycbcr.cr, 128);
    }

    #[test]
    fn test_white_rgb_to_ycbcr() {
        let white = RgbPixel::new(255, 255, 255);
        let ycbcr = rgb_to_ycbcr(white, ColorStandard::Bt709, RangeMode::Full);
        assert_eq!(ycbcr.y, 255);
        assert_eq!(ycbcr.cb, 128);
        assert_eq!(ycbcr.cr, 128);
    }

    #[test]
    fn test_roundtrip_full_range() {
        let original = RgbPixel::new(100, 150, 200);
        let ycbcr = rgb_to_ycbcr(original, ColorStandard::Bt709, RangeMode::Full);
        let recovered = ycbcr_to_rgb(ycbcr, ColorStandard::Bt709, RangeMode::Full);
        // Allow +-2 due to rounding
        assert!((original.r as i16 - recovered.r as i16).unsigned_abs() <= 2);
        assert!((original.g as i16 - recovered.g as i16).unsigned_abs() <= 2);
        assert!((original.b as i16 - recovered.b as i16).unsigned_abs() <= 2);
    }

    #[test]
    fn test_roundtrip_limited_range() {
        let original = RgbPixel::new(80, 120, 160);
        let ycbcr = rgb_to_ycbcr(original, ColorStandard::Bt709, RangeMode::Limited);
        let recovered = ycbcr_to_rgb(ycbcr, ColorStandard::Bt709, RangeMode::Limited);
        assert!((original.r as i16 - recovered.r as i16).unsigned_abs() <= 3);
        assert!((original.g as i16 - recovered.g as i16).unsigned_abs() <= 3);
        assert!((original.b as i16 - recovered.b as i16).unsigned_abs() <= 3);
    }

    #[test]
    fn test_detect_standard_sd() {
        assert_eq!(detect_standard(720, 576), ColorStandard::Bt601);
    }

    #[test]
    fn test_detect_standard_hd() {
        assert_eq!(detect_standard(1920, 1080), ColorStandard::Bt709);
    }

    #[test]
    fn test_detect_standard_uhd() {
        assert_eq!(detect_standard(3840, 2160), ColorStandard::Bt2020);
    }

    #[test]
    fn test_batch_rgb_to_ycbcr() {
        let rgb = vec![0, 0, 0, 255, 255, 255];
        let ycbcr = batch_rgb_to_ycbcr(&rgb, ColorStandard::Bt709, RangeMode::Full);
        assert_eq!(ycbcr.len(), 6);
        assert_eq!(ycbcr[0], 0); // Y of black
        assert_eq!(ycbcr[3], 255); // Y of white
    }

    #[test]
    fn test_batch_ycbcr_to_rgb() {
        let ycbcr = vec![0, 128, 128, 255, 128, 128];
        let rgb = batch_ycbcr_to_rgb(&ycbcr, ColorStandard::Bt709, RangeMode::Full);
        assert_eq!(rgb.len(), 6);
        assert_eq!(rgb[0], 0); // R of black
        assert_eq!(rgb[3], 255); // R of white
    }

    #[test]
    fn test_bt601_red_pixel() {
        let red = RgbPixel::new(255, 0, 0);
        let ycbcr = rgb_to_ycbcr(red, ColorStandard::Bt601, RangeMode::Full);
        // Y should be ~76 for BT.601 pure red
        assert!(ycbcr.y > 70 && ycbcr.y < 85);
        // Cr should be high for red
        assert!(ycbcr.cr > 200);
    }

    #[test]
    fn test_clamp_u8_boundaries() {
        assert_eq!(clamp_u8(-10.0), 0);
        assert_eq!(clamp_u8(300.0), 255);
        assert_eq!(clamp_u8(128.0), 128);
    }
}
