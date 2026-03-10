//! Video luminance metering.
//!
//! Implements luminance measurement for video signals including:
//! - Peak luminance (nits)
//! - Average luminance
//! - Minimum luminance
//! - Luminance distribution

use crate::{MeteringError, MeteringResult};
use ndarray::Array2;

/// Luminance meter for video frames.
pub struct LuminanceMeter {
    width: usize,
    height: usize,
    peak_nits: f64,
    min_nits: f64,
    average_nits: f64,
    histogram: Vec<usize>,
    histogram_bins: usize,
    max_nits: f64,
}

impl LuminanceMeter {
    /// Create a new luminance meter.
    ///
    /// # Arguments
    ///
    /// * `width` - Frame width in pixels
    /// * `height` - Frame height in pixels
    /// * `max_nits` - Maximum expected luminance in nits (e.g., 1000 for HDR, 100 for SDR)
    /// * `histogram_bins` - Number of histogram bins
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn new(
        width: usize,
        height: usize,
        max_nits: f64,
        histogram_bins: usize,
    ) -> MeteringResult<Self> {
        if width == 0 || height == 0 {
            return Err(MeteringError::InvalidConfig(
                "Width and height must be positive".to_string(),
            ));
        }

        if max_nits <= 0.0 {
            return Err(MeteringError::InvalidConfig(
                "Max nits must be positive".to_string(),
            ));
        }

        Ok(Self {
            width,
            height,
            peak_nits: 0.0,
            min_nits: f64::INFINITY,
            average_nits: 0.0,
            histogram: vec![0; histogram_bins],
            histogram_bins,
            max_nits,
        })
    }

    /// Process a luminance frame.
    ///
    /// # Arguments
    ///
    /// * `luminance` - 2D array of luminance values in nits
    ///
    /// # Errors
    ///
    /// Returns error if frame dimensions don't match.
    pub fn process(&mut self, luminance: &Array2<f64>) -> MeteringResult<()> {
        let (height, width) = luminance.dim();

        if width != self.width || height != self.height {
            return Err(MeteringError::InvalidConfig(format!(
                "Frame dimensions {}x{} don't match expected {}x{}",
                width, height, self.width, self.height
            )));
        }

        // Reset metrics
        self.peak_nits = 0.0;
        self.min_nits = f64::INFINITY;
        let mut sum = 0.0;
        self.histogram.fill(0);

        // Analyze frame
        for &value in luminance {
            // Update peak and min
            if value > self.peak_nits {
                self.peak_nits = value;
            }
            if value < self.min_nits {
                self.min_nits = value;
            }

            // Update sum for average
            sum += value;

            // Update histogram
            let bin = ((value / self.max_nits) * (self.histogram_bins - 1) as f64)
                .clamp(0.0, (self.histogram_bins - 1) as f64) as usize;
            self.histogram[bin] += 1;
        }

        // Calculate average
        let pixel_count = (self.width * self.height) as f64;
        self.average_nits = sum / pixel_count;

        Ok(())
    }

    /// Get the peak luminance in nits.
    pub fn peak_nits(&self) -> f64 {
        self.peak_nits
    }

    /// Get the minimum luminance in nits.
    pub fn min_nits(&self) -> f64 {
        if self.min_nits.is_infinite() {
            0.0
        } else {
            self.min_nits
        }
    }

    /// Get the average luminance in nits.
    pub fn average_nits(&self) -> f64 {
        self.average_nits
    }

    /// Get the luminance histogram.
    pub fn histogram(&self) -> &[usize] {
        &self.histogram
    }

    /// Get the contrast ratio.
    ///
    /// Contrast ratio = Peak / Min
    pub fn contrast_ratio(&self) -> f64 {
        let min = if self.min_nits.is_infinite() || self.min_nits == 0.0 {
            0.001 // Prevent division by zero
        } else {
            self.min_nits
        };

        self.peak_nits / min
    }

    /// Get the dynamic range in stops.
    ///
    /// Dynamic range (stops) = log2(Peak / Min)
    pub fn dynamic_range_stops(&self) -> f64 {
        self.contrast_ratio().log2()
    }

    /// Check if frame is within SDR range (0-100 nits).
    pub fn is_sdr(&self) -> bool {
        self.peak_nits <= 100.0
    }

    /// Check if frame is HDR10 (up to 1000 nits).
    pub fn is_hdr10(&self) -> bool {
        self.peak_nits > 100.0 && self.peak_nits <= 1000.0
    }

    /// Check if frame is HDR10+ or Dolby Vision (up to 10000 nits).
    pub fn is_extreme_hdr(&self) -> bool {
        self.peak_nits > 1000.0
    }

    /// Reset the meter.
    pub fn reset(&mut self) {
        self.peak_nits = 0.0;
        self.min_nits = f64::INFINITY;
        self.average_nits = 0.0;
        self.histogram.fill(0);
    }
}

/// Black and white level meter for broadcast compliance.
pub struct BlackWhiteLevelMeter {
    width: usize,
    height: usize,
    black_level: f64,
    white_level: f64,
    below_black_count: usize,
    above_white_count: usize,
    black_threshold: f64,
    white_threshold: f64,
}

impl BlackWhiteLevelMeter {
    /// Create a new black/white level meter.
    ///
    /// # Arguments
    ///
    /// * `width` - Frame width
    /// * `height` - Frame height
    /// * `black_threshold` - Black level threshold (e.g., 0 for digital, 16/255 for video)
    /// * `white_threshold` - White level threshold (e.g., 1.0 for digital, 235/255 for video)
    pub fn new(
        width: usize,
        height: usize,
        black_threshold: f64,
        white_threshold: f64,
    ) -> MeteringResult<Self> {
        if width == 0 || height == 0 {
            return Err(MeteringError::InvalidConfig(
                "Width and height must be positive".to_string(),
            ));
        }

        Ok(Self {
            width,
            height,
            black_level: 0.0,
            white_level: 0.0,
            below_black_count: 0,
            above_white_count: 0,
            black_threshold,
            white_threshold,
        })
    }

    /// Process a video frame (normalized 0.0 to 1.0).
    pub fn process(&mut self, frame: &Array2<f64>) -> MeteringResult<()> {
        let (height, width) = frame.dim();

        if width != self.width || height != self.height {
            return Err(MeteringError::InvalidConfig(
                "Frame dimensions don't match".to_string(),
            ));
        }

        self.below_black_count = 0;
        self.above_white_count = 0;
        let mut min_val = f64::INFINITY;
        let mut max_val = 0.0;

        for &value in frame {
            if value < min_val {
                min_val = value;
            }
            if value > max_val {
                max_val = value;
            }

            if value < self.black_threshold {
                self.below_black_count += 1;
            }
            if value > self.white_threshold {
                self.above_white_count += 1;
            }
        }

        self.black_level = min_val;
        self.white_level = max_val;

        Ok(())
    }

    /// Get the black level (minimum value).
    pub fn black_level(&self) -> f64 {
        self.black_level
    }

    /// Get the white level (maximum value).
    pub fn white_level(&self) -> f64 {
        self.white_level
    }

    /// Get the number of pixels below black threshold.
    pub fn below_black_count(&self) -> usize {
        self.below_black_count
    }

    /// Get the number of pixels above white threshold.
    pub fn above_white_count(&self) -> usize {
        self.above_white_count
    }

    /// Check if frame is compliant (no pixels outside legal range).
    pub fn is_compliant(&self) -> bool {
        self.below_black_count == 0 && self.above_white_count == 0
    }

    /// Get the percentage of illegal pixels.
    pub fn illegal_pixel_percentage(&self) -> f64 {
        let total_pixels = (self.width * self.height) as f64;
        let illegal_pixels = (self.below_black_count + self.above_white_count) as f64;
        (illegal_pixels / total_pixels) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_luminance_meter() {
        let mut meter =
            LuminanceMeter::new(1920, 1080, 1000.0, 256).expect("test expectation failed");

        // Create test frame with known values
        let frame = Array2::from_shape_fn((1080, 1920), |(y, x)| {
            (x + y) as f64 / (1920 + 1080) as f64 * 100.0
        });

        meter.process(&frame).expect("process should succeed");

        assert!(meter.peak_nits() > 0.0);
        assert!(meter.average_nits() > 0.0);
        assert!(meter.min_nits() >= 0.0);
    }

    #[test]
    fn test_sdr_detection() {
        let mut meter =
            LuminanceMeter::new(100, 100, 1000.0, 256).expect("test expectation failed");

        // SDR frame (max 100 nits)
        let frame = Array2::from_elem((100, 100), 80.0);
        meter.process(&frame).expect("process should succeed");

        assert!(meter.is_sdr());
        assert!(!meter.is_hdr10());
    }

    #[test]
    fn test_hdr_detection() {
        let mut meter =
            LuminanceMeter::new(100, 100, 1000.0, 256).expect("test expectation failed");

        // HDR frame (500 nits)
        let frame = Array2::from_elem((100, 100), 500.0);
        meter.process(&frame).expect("process should succeed");

        assert!(!meter.is_sdr());
        assert!(meter.is_hdr10());
    }

    #[test]
    fn test_contrast_ratio() {
        let mut meter =
            LuminanceMeter::new(100, 100, 1000.0, 256).expect("test expectation failed");

        // Frame with known contrast
        let mut frame = Array2::zeros((100, 100));
        frame[[0, 0]] = 1.0; // Min
        frame[[99, 99]] = 100.0; // Max

        meter.process(&frame).expect("process should succeed");

        assert_eq!(meter.peak_nits(), 100.0);
        assert_eq!(meter.min_nits(), 0.0);
    }

    #[test]
    fn test_black_white_level_meter() {
        let mut meter =
            BlackWhiteLevelMeter::new(100, 100, 0.0, 1.0).expect("test expectation failed");

        // Compliant frame
        let frame = Array2::from_elem((100, 100), 0.5);
        meter.process(&frame).expect("process should succeed");

        assert!(meter.is_compliant());
        assert_eq!(meter.illegal_pixel_percentage(), 0.0);
    }

    #[test]
    fn test_illegal_pixels() {
        let mut meter =
            BlackWhiteLevelMeter::new(100, 100, 0.0, 1.0).expect("test expectation failed");

        // Frame with illegal pixels
        let mut frame = Array2::from_elem((100, 100), 0.5);
        frame[[0, 0]] = -0.1; // Below black
        frame[[99, 99]] = 1.1; // Above white

        meter.process(&frame).expect("process should succeed");

        assert!(!meter.is_compliant());
        assert!(meter.below_black_count() > 0);
        assert!(meter.above_white_count() > 0);
    }
}
