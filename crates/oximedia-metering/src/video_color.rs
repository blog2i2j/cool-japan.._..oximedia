//! Video color metering.
//!
//! Implements color gamut and saturation analysis for video signals.

use crate::{MeteringError, MeteringResult};
use ndarray::Array2;

/// Color gamut standard.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColorGamut {
    /// Rec.709 (HDTV standard, sRGB).
    Rec709,
    /// Rec.2020 (UHDTV standard, wide color gamut).
    Rec2020,
    /// DCI-P3 (Digital Cinema Initiative).
    DciP3,
}

impl ColorGamut {
    /// Get the name of the gamut.
    pub fn name(&self) -> &str {
        match self {
            Self::Rec709 => "Rec.709",
            Self::Rec2020 => "Rec.2020",
            Self::DciP3 => "DCI-P3",
        }
    }
}

/// RGB color sample.
#[derive(Clone, Copy, Debug)]
pub struct RgbColor {
    /// Red component (0.0 to 1.0).
    pub r: f64,
    /// Green component (0.0 to 1.0).
    pub g: f64,
    /// Blue component (0.0 to 1.0).
    pub b: f64,
}

impl RgbColor {
    /// Create a new RGB color.
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }

    /// Convert to HSV (Hue, Saturation, Value).
    pub fn to_hsv(&self) -> HsvColor {
        let max = self.r.max(self.g).max(self.b);
        let min = self.r.min(self.g).min(self.b);
        let delta = max - min;

        let value = max;

        let saturation = if max > 0.0 { delta / max } else { 0.0 };

        let hue = if delta == 0.0 {
            0.0
        } else if max == self.r {
            60.0 * (((self.g - self.b) / delta) % 6.0)
        } else if max == self.g {
            60.0 * (((self.b - self.r) / delta) + 2.0)
        } else {
            60.0 * (((self.r - self.g) / delta) + 4.0)
        };

        let hue = if hue < 0.0 { hue + 360.0 } else { hue };

        HsvColor {
            h: hue,
            s: saturation,
            v: value,
        }
    }
}

/// HSV color representation.
#[derive(Clone, Copy, Debug)]
pub struct HsvColor {
    /// Hue (0.0 to 360.0 degrees).
    pub h: f64,
    /// Saturation (0.0 to 1.0).
    pub s: f64,
    /// Value (0.0 to 1.0).
    pub v: f64,
}

/// Color gamut meter.
pub struct GamutMeter {
    width: usize,
    height: usize,
    target_gamut: ColorGamut,
    in_gamut_count: usize,
    out_of_gamut_count: usize,
    max_saturation: f64,
    average_saturation: f64,
}

impl GamutMeter {
    /// Create a new gamut meter.
    ///
    /// # Arguments
    ///
    /// * `width` - Frame width
    /// * `height` - Frame height
    /// * `target_gamut` - Target color gamut
    pub fn new(width: usize, height: usize, target_gamut: ColorGamut) -> MeteringResult<Self> {
        if width == 0 || height == 0 {
            return Err(MeteringError::InvalidConfig(
                "Width and height must be positive".to_string(),
            ));
        }

        Ok(Self {
            width,
            height,
            target_gamut,
            in_gamut_count: 0,
            out_of_gamut_count: 0,
            max_saturation: 0.0,
            average_saturation: 0.0,
        })
    }

    /// Process RGB frame data.
    ///
    /// # Arguments
    ///
    /// * `r_channel` - Red channel (0.0 to 1.0)
    /// * `g_channel` - Green channel (0.0 to 1.0)
    /// * `b_channel` - Blue channel (0.0 to 1.0)
    pub fn process(
        &mut self,
        r_channel: &Array2<f64>,
        g_channel: &Array2<f64>,
        b_channel: &Array2<f64>,
    ) -> MeteringResult<()> {
        let (height, width) = r_channel.dim();

        if width != self.width || height != self.height {
            return Err(MeteringError::InvalidConfig(
                "Frame dimensions don't match".to_string(),
            ));
        }

        self.in_gamut_count = 0;
        self.out_of_gamut_count = 0;
        self.max_saturation = 0.0;
        let mut saturation_sum = 0.0;

        for y in 0..height {
            for x in 0..width {
                let rgb = RgbColor::new(r_channel[[y, x]], g_channel[[y, x]], b_channel[[y, x]]);

                // Check if in gamut
                if self.is_in_gamut(&rgb) {
                    self.in_gamut_count += 1;
                } else {
                    self.out_of_gamut_count += 1;
                }

                // Calculate saturation
                let hsv = rgb.to_hsv();
                if hsv.s > self.max_saturation {
                    self.max_saturation = hsv.s;
                }
                saturation_sum += hsv.s;
            }
        }

        let pixel_count = (width * height) as f64;
        self.average_saturation = saturation_sum / pixel_count;

        Ok(())
    }

    /// Check if a color is within the target gamut.
    fn is_in_gamut(&self, rgb: &RgbColor) -> bool {
        // Simplified gamut check: values should be in 0.0 to 1.0 range
        // A full implementation would use CIE xy chromaticity coordinates
        rgb.r >= 0.0 && rgb.r <= 1.0 && rgb.g >= 0.0 && rgb.g <= 1.0 && rgb.b >= 0.0 && rgb.b <= 1.0
    }

    /// Get the gamut coverage percentage.
    pub fn gamut_coverage_percentage(&self) -> f64 {
        let total = (self.in_gamut_count + self.out_of_gamut_count) as f64;
        if total > 0.0 {
            (self.in_gamut_count as f64 / total) * 100.0
        } else {
            0.0
        }
    }

    /// Get the number of out-of-gamut pixels.
    pub fn out_of_gamut_count(&self) -> usize {
        self.out_of_gamut_count
    }

    /// Get the maximum saturation in the frame.
    pub fn max_saturation(&self) -> f64 {
        self.max_saturation
    }

    /// Get the average saturation in the frame.
    pub fn average_saturation(&self) -> f64 {
        self.average_saturation
    }

    /// Get the target gamut.
    pub fn target_gamut(&self) -> ColorGamut {
        self.target_gamut
    }

    /// Reset the meter.
    pub fn reset(&mut self) {
        self.in_gamut_count = 0;
        self.out_of_gamut_count = 0;
        self.max_saturation = 0.0;
        self.average_saturation = 0.0;
    }
}

/// Saturation meter for color vibrancy analysis.
pub struct SaturationMeter {
    width: usize,
    height: usize,
    max_saturation: f64,
    average_saturation: f64,
    histogram: Vec<usize>,
    histogram_bins: usize,
}

impl SaturationMeter {
    /// Create a new saturation meter.
    ///
    /// # Arguments
    ///
    /// * `width` - Frame width
    /// * `height` - Frame height
    /// * `histogram_bins` - Number of histogram bins
    pub fn new(width: usize, height: usize, histogram_bins: usize) -> MeteringResult<Self> {
        if width == 0 || height == 0 {
            return Err(MeteringError::InvalidConfig(
                "Width and height must be positive".to_string(),
            ));
        }

        Ok(Self {
            width,
            height,
            max_saturation: 0.0,
            average_saturation: 0.0,
            histogram: vec![0; histogram_bins],
            histogram_bins,
        })
    }

    /// Process RGB frame data.
    pub fn process(
        &mut self,
        r_channel: &Array2<f64>,
        g_channel: &Array2<f64>,
        b_channel: &Array2<f64>,
    ) -> MeteringResult<()> {
        let (height, width) = r_channel.dim();

        if width != self.width || height != self.height {
            return Err(MeteringError::InvalidConfig(
                "Frame dimensions don't match".to_string(),
            ));
        }

        self.max_saturation = 0.0;
        self.histogram.fill(0);
        let mut saturation_sum = 0.0;

        for y in 0..height {
            for x in 0..width {
                let rgb = RgbColor::new(r_channel[[y, x]], g_channel[[y, x]], b_channel[[y, x]]);
                let hsv = rgb.to_hsv();

                if hsv.s > self.max_saturation {
                    self.max_saturation = hsv.s;
                }

                saturation_sum += hsv.s;

                // Update histogram
                let bin = (hsv.s * (self.histogram_bins - 1) as f64)
                    .clamp(0.0, (self.histogram_bins - 1) as f64)
                    as usize;
                self.histogram[bin] += 1;
            }
        }

        let pixel_count = (width * height) as f64;
        self.average_saturation = saturation_sum / pixel_count;

        Ok(())
    }

    /// Get the maximum saturation.
    pub fn max_saturation(&self) -> f64 {
        self.max_saturation
    }

    /// Get the average saturation.
    pub fn average_saturation(&self) -> f64 {
        self.average_saturation
    }

    /// Get the saturation histogram.
    pub fn histogram(&self) -> &[usize] {
        &self.histogram
    }

    /// Check if the frame is desaturated (low average saturation).
    pub fn is_desaturated(&self) -> bool {
        self.average_saturation < 0.1
    }

    /// Check if the frame is highly saturated.
    pub fn is_highly_saturated(&self) -> bool {
        self.average_saturation > 0.7
    }
}

/// Color temperature meter (simplified).
pub struct ColorTemperatureMeter {
    width: usize,
    height: usize,
    estimated_temperature_k: f64,
}

impl ColorTemperatureMeter {
    /// Create a new color temperature meter.
    pub fn new(width: usize, height: usize) -> MeteringResult<Self> {
        if width == 0 || height == 0 {
            return Err(MeteringError::InvalidConfig(
                "Width and height must be positive".to_string(),
            ));
        }

        Ok(Self {
            width,
            height,
            estimated_temperature_k: 6500.0, // D65 default
        })
    }

    /// Process RGB frame data.
    pub fn process(
        &mut self,
        r_channel: &Array2<f64>,
        g_channel: &Array2<f64>,
        b_channel: &Array2<f64>,
    ) -> MeteringResult<()> {
        let (height, width) = r_channel.dim();

        if width != self.width || height != self.height {
            return Err(MeteringError::InvalidConfig(
                "Frame dimensions don't match".to_string(),
            ));
        }

        // Calculate average RGB values
        let r_avg: f64 = r_channel.iter().sum::<f64>() / (width * height) as f64;
        let _g_avg: f64 = g_channel.iter().sum::<f64>() / (width * height) as f64;
        let b_avg: f64 = b_channel.iter().sum::<f64>() / (width * height) as f64;

        // Simplified color temperature estimation based on RGB ratios
        // This is a rough approximation
        // Higher R/B ratio = warmer (lower temperature K)
        // Lower R/B ratio = cooler (higher temperature K)
        let ratio = if b_avg > 0.0 { r_avg / b_avg } else { 1.0 };

        self.estimated_temperature_k = if ratio > 1.5 {
            // Warm (reddish) - high R/B ratio means low temperature
            2500.0 + (3.0 - ratio.min(3.0)) * 1500.0
        } else if ratio < 0.8 {
            // Cool (bluish) - low R/B ratio means high temperature
            7000.0 + (0.8 - ratio.max(0.3)) * 6000.0
        } else {
            // Neutral - map 0.8-1.5 to 6500K ± 500K
            6500.0 + (ratio - 1.15) * 1500.0
        };

        // Clamp to reasonable range
        self.estimated_temperature_k = self.estimated_temperature_k.clamp(2000.0, 10_000.0);

        Ok(())
    }

    /// Get the estimated color temperature in Kelvin.
    pub fn temperature_kelvin(&self) -> f64 {
        self.estimated_temperature_k
    }

    /// Check if the temperature is warm (< 5000K).
    pub fn is_warm(&self) -> bool {
        self.estimated_temperature_k < 5000.0
    }

    /// Check if the temperature is cool (> 7000K).
    pub fn is_cool(&self) -> bool {
        self.estimated_temperature_k > 7000.0
    }

    /// Check if the temperature is neutral (D65 range, 6000-7000K).
    pub fn is_neutral(&self) -> bool {
        self.estimated_temperature_k >= 6000.0 && self.estimated_temperature_k <= 7000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_to_hsv() {
        // Test pure red
        let red = RgbColor::new(1.0, 0.0, 0.0);
        let hsv = red.to_hsv();
        assert!((hsv.h - 0.0).abs() < 1.0);
        assert_eq!(hsv.s, 1.0);
        assert_eq!(hsv.v, 1.0);

        // Test gray (no saturation)
        let gray = RgbColor::new(0.5, 0.5, 0.5);
        let hsv = gray.to_hsv();
        assert_eq!(hsv.s, 0.0);
    }

    #[test]
    fn test_gamut_meter() {
        let mut meter =
            GamutMeter::new(100, 100, ColorGamut::Rec709).expect("test expectation failed");

        let r = Array2::from_elem((100, 100), 0.5);
        let g = Array2::from_elem((100, 100), 0.5);
        let b = Array2::from_elem((100, 100), 0.5);

        meter.process(&r, &g, &b).expect("process should succeed");

        assert_eq!(meter.gamut_coverage_percentage(), 100.0);
    }

    #[test]
    fn test_saturation_meter() {
        let mut meter = SaturationMeter::new(100, 100, 256).expect("test expectation failed");

        // Create highly saturated frame (pure red)
        let r = Array2::from_elem((100, 100), 1.0);
        let g = Array2::from_elem((100, 100), 0.0);
        let b = Array2::from_elem((100, 100), 0.0);

        meter.process(&r, &g, &b).expect("process should succeed");

        assert!(meter.is_highly_saturated());
        assert_eq!(meter.max_saturation(), 1.0);
    }

    #[test]
    fn test_desaturated_frame() {
        let mut meter = SaturationMeter::new(100, 100, 256).expect("test expectation failed");

        // Create grayscale frame
        let r = Array2::from_elem((100, 100), 0.5);
        let g = Array2::from_elem((100, 100), 0.5);
        let b = Array2::from_elem((100, 100), 0.5);

        meter.process(&r, &g, &b).expect("process should succeed");

        assert!(meter.is_desaturated());
    }

    #[test]
    fn test_color_temperature() {
        let mut meter = ColorTemperatureMeter::new(100, 100).expect("test expectation failed");

        // Warm frame (much more red than blue)
        let r = Array2::from_elem((100, 100), 0.9);
        let g = Array2::from_elem((100, 100), 0.5);
        let b = Array2::from_elem((100, 100), 0.2);

        meter.process(&r, &g, &b).expect("process should succeed");

        // The simplified algorithm should detect warmth
        // If this still fails, the temperature is logged for debugging
        let temp = meter.temperature_kelvin();
        assert!(
            temp < 6000.0,
            "Temperature {:.0}K should be warm (< 6000K)",
            temp
        );
    }
}
