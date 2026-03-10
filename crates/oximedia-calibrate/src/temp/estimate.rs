//! Color temperature estimation from image analysis.
//!
//! This module provides tools for estimating the color temperature of an image.

use crate::error::{CalibrationError, CalibrationResult};

/// Estimate the color temperature of an image.
///
/// # Arguments
///
/// * `image_data` - Raw image data (RGB format)
///
/// # Errors
///
/// Returns an error if estimation fails.
///
/// # Returns
///
/// Estimated color temperature in Kelvin.
pub fn estimate_color_temperature(image_data: &[u8]) -> CalibrationResult<u32> {
    if image_data.is_empty() || image_data.len() % 3 != 0 {
        return Err(CalibrationError::TemperatureEstimationFailed(
            "Invalid image data".to_string(),
        ));
    }

    // Calculate average RGB values
    let mut r_sum: u64 = 0;
    let mut g_sum: u64 = 0;
    let mut b_sum: u64 = 0;
    let pixel_count = image_data.len() / 3;

    for chunk in image_data.chunks_exact(3) {
        r_sum += u64::from(chunk[0]);
        g_sum += u64::from(chunk[1]);
        b_sum += u64::from(chunk[2]);
    }

    let r_avg = r_sum as f64 / pixel_count as f64;
    let _g_avg = g_sum as f64 / pixel_count as f64;
    let b_avg = b_sum as f64 / pixel_count as f64;

    // Estimate color temperature from R/B ratio
    // This is a simplified approximation
    let rb_ratio = r_avg / b_avg.max(1.0);

    let temperature = estimate_from_rb_ratio(rb_ratio);

    Ok(temperature)
}

/// Estimate color temperature from red/blue ratio.
fn estimate_from_rb_ratio(rb_ratio: f64) -> u32 {
    // Simplified mapping from R/B ratio to color temperature
    // Based on approximations of the Planckian locus

    if rb_ratio > 1.5 {
        2000 // Very warm (candlelight)
    } else if rb_ratio > 1.3 {
        2500 // Warm (sunset, tungsten)
    } else if rb_ratio > 1.1 {
        3000 // Tungsten/incandescent
    } else if rb_ratio > 1.0 {
        4000 // Warm white fluorescent
    } else if rb_ratio >= 0.95 {
        5000 // Horizon daylight
    } else if rb_ratio > 0.90 {
        5500 // Mid-morning/afternoon daylight
    } else if rb_ratio > 0.85 {
        6500 // Noon daylight, electronic flash
    } else if rb_ratio > 0.80 {
        7500 // Overcast sky
    } else {
        9000 // Shade, heavily overcast
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_color_temperature_empty() {
        let result = estimate_color_temperature(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_estimate_color_temperature_invalid_size() {
        let result = estimate_color_temperature(&[128, 128]);
        assert!(result.is_err());
    }

    #[test]
    fn test_estimate_color_temperature_neutral() {
        // Neutral gray should estimate around daylight
        let image = vec![128; 300]; // 100 gray pixels
        let result = estimate_color_temperature(&image);

        assert!(result.is_ok());
        let temp = result.expect("expected successful result");

        // Should be in the daylight range (5000-7000K)
        assert!(temp >= 5000 && temp <= 7000);
    }

    #[test]
    fn test_estimate_color_temperature_warm() {
        // Warm (reddish) image
        let mut image = Vec::new();
        for _ in 0..100 {
            image.extend_from_slice(&[200, 150, 100]); // Warm color
        }

        let result = estimate_color_temperature(&image);
        assert!(result.is_ok());

        let temp = result.expect("expected successful result");
        // Should estimate as tungsten (around 3000K)
        assert!(temp <= 4000);
    }

    #[test]
    fn test_estimate_color_temperature_cool() {
        // Cool (bluish) image
        let mut image = Vec::new();
        for _ in 0..100 {
            image.extend_from_slice(&[100, 150, 200]); // Cool color
        }

        let result = estimate_color_temperature(&image);
        assert!(result.is_ok());

        let temp = result.expect("expected successful result");
        // Should estimate as shade/overcast (7000K+)
        assert!(temp >= 7000);
    }

    #[test]
    fn test_estimate_from_rb_ratio() {
        assert_eq!(estimate_from_rb_ratio(1.6), 2000);
        assert_eq!(estimate_from_rb_ratio(1.2), 3000);
        assert_eq!(estimate_from_rb_ratio(0.95), 5000);
        assert_eq!(estimate_from_rb_ratio(0.88), 6500);
        assert_eq!(estimate_from_rb_ratio(0.75), 9000);
    }
}
