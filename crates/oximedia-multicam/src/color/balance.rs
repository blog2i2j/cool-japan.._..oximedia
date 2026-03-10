//! White balance matching across camera angles.

use super::ColorStats;
use crate::{AngleId, Result};

/// White balance matcher
#[derive(Debug)]
pub struct WhiteBalanceMatching {
    /// Reference angle
    reference_angle: AngleId,
    /// White balance settings for each angle
    settings: Vec<WhiteBalanceSettings>,
}

/// White balance settings
#[derive(Debug, Clone, Copy)]
pub struct WhiteBalanceSettings {
    /// Angle identifier
    pub angle: AngleId,
    /// Color temperature (Kelvin)
    pub temperature: f32,
    /// Tint adjustment (-100 to +100)
    pub tint: f32,
    /// Red gain
    pub red_gain: f32,
    /// Green gain
    pub green_gain: f32,
    /// Blue gain
    pub blue_gain: f32,
}

impl WhiteBalanceSettings {
    /// Create default white balance settings
    #[must_use]
    pub fn new(angle: AngleId) -> Self {
        Self {
            angle,
            temperature: 6500.0,
            tint: 0.0,
            red_gain: 1.0,
            green_gain: 1.0,
            blue_gain: 1.0,
        }
    }

    /// Apply white balance to RGB values
    #[must_use]
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        [
            (rgb[0] * self.red_gain).clamp(0.0, 1.0),
            (rgb[1] * self.green_gain).clamp(0.0, 1.0),
            (rgb[2] * self.blue_gain).clamp(0.0, 1.0),
        ]
    }

    /// Calculate gains from temperature and tint
    pub fn calculate_gains(&mut self) {
        // Simplified color temperature to RGB conversion
        let temp_norm = (self.temperature - 2000.0) / 8000.0;

        // Red channel
        if temp_norm < 0.5 {
            self.red_gain = 0.6 + temp_norm;
        } else {
            self.red_gain = 1.1;
        }

        // Blue channel
        if temp_norm < 0.5 {
            self.blue_gain = 1.1;
        } else {
            self.blue_gain = 1.1 - (temp_norm - 0.5);
        }

        // Green channel (affected by tint)
        self.green_gain = 1.0 + (self.tint / 200.0);

        // Normalize to keep maximum gain at 1.0
        let max_gain = self.red_gain.max(self.green_gain).max(self.blue_gain);
        if max_gain > 1.0 {
            self.red_gain /= max_gain;
            self.green_gain /= max_gain;
            self.blue_gain /= max_gain;
        }
    }
}

impl WhiteBalanceMatching {
    /// Create a new white balance matcher
    #[must_use]
    pub fn new(angle_count: usize, reference_angle: AngleId) -> Self {
        Self {
            reference_angle,
            settings: (0..angle_count).map(WhiteBalanceSettings::new).collect(),
        }
    }

    /// Set reference angle
    pub fn set_reference_angle(&mut self, angle: AngleId) {
        self.reference_angle = angle;
    }

    /// Get reference angle
    #[must_use]
    pub fn reference_angle(&self) -> AngleId {
        self.reference_angle
    }

    /// Update white balance settings for an angle
    pub fn update_settings(&mut self, settings: WhiteBalanceSettings) {
        if settings.angle < self.settings.len() {
            self.settings[settings.angle] = settings;
        }
    }

    /// Auto white balance from color statistics
    pub fn auto_white_balance(&mut self, stats: &ColorStats) {
        if stats.angle >= self.settings.len() {
            return;
        }

        let mut settings = self.settings[stats.angle];
        settings.temperature = stats.temperature;
        settings.tint = stats.tint;
        settings.calculate_gains();

        self.settings[stats.angle] = settings;
    }

    /// Match white balance to reference
    ///
    /// # Errors
    ///
    /// Returns an error if matching fails
    pub fn match_to_reference(&mut self) -> Result<()> {
        if self.reference_angle >= self.settings.len() {
            return Err(crate::MultiCamError::AngleNotFound(self.reference_angle));
        }

        let reference = self.settings[self.reference_angle];

        for settings in &mut self.settings {
            if settings.angle != self.reference_angle {
                settings.temperature = reference.temperature;
                settings.tint = reference.tint;
                settings.calculate_gains();
            }
        }

        Ok(())
    }

    /// Get white balance settings for angle
    #[must_use]
    pub fn get_settings(&self, angle: AngleId) -> Option<&WhiteBalanceSettings> {
        self.settings.get(angle)
    }

    /// Apply white balance to RGB values
    #[must_use]
    pub fn apply_white_balance(&self, angle: AngleId, rgb: [f32; 3]) -> [f32; 3] {
        if let Some(settings) = self.get_settings(angle) {
            settings.apply(rgb)
        } else {
            rgb
        }
    }

    /// Detect gray card in image
    #[must_use]
    pub fn detect_gray_card(
        &self,
        image_data: &[u8],
        width: usize,
        height: usize,
    ) -> Option<[f32; 3]> {
        // Simplified gray card detection
        // In practice, this would use edge detection and color analysis
        if image_data.is_empty() || width == 0 || height == 0 {
            return None;
        }

        // Sample center region
        let center_x = width / 2;
        let center_y = height / 2;
        let sample_size = width.min(height) / 10;

        let mut sum_r = 0u64;
        let mut sum_g = 0u64;
        let mut sum_b = 0u64;
        let mut count = 0u64;

        for y in
            (center_y.saturating_sub(sample_size / 2))..(center_y + sample_size / 2).min(height)
        {
            for x in
                (center_x.saturating_sub(sample_size / 2))..(center_x + sample_size / 2).min(width)
            {
                let offset = (y * width + x) * 3;
                if offset + 2 < image_data.len() {
                    sum_r += u64::from(image_data[offset]);
                    sum_g += u64::from(image_data[offset + 1]);
                    sum_b += u64::from(image_data[offset + 2]);
                    count += 1;
                }
            }
        }

        if count > 0 {
            Some([
                (sum_r / count) as f32 / 255.0,
                (sum_g / count) as f32 / 255.0,
                (sum_b / count) as f32 / 255.0,
            ])
        } else {
            None
        }
    }

    /// Calculate white balance from gray reference
    #[must_use]
    pub fn calculate_from_gray(&self, gray_rgb: [f32; 3]) -> WhiteBalanceSettings {
        let mut settings = WhiteBalanceSettings::new(0);

        // Calculate gains to neutralize the gray patch
        let avg = (gray_rgb[0] + gray_rgb[1] + gray_rgb[2]) / 3.0;

        if avg > 0.0 {
            settings.red_gain = avg / gray_rgb[0].max(0.001);
            settings.green_gain = avg / gray_rgb[1].max(0.001);
            settings.blue_gain = avg / gray_rgb[2].max(0.001);

            // Normalize
            let max_gain = settings
                .red_gain
                .max(settings.green_gain)
                .max(settings.blue_gain);
            if max_gain > 1.0 {
                settings.red_gain /= max_gain;
                settings.green_gain /= max_gain;
                settings.blue_gain /= max_gain;
            }
        }

        settings
    }

    /// Calculate color temperature from RGB
    #[must_use]
    pub fn estimate_temperature(rgb: [f32; 3]) -> f32 {
        // Simplified color temperature estimation
        let ratio = if rgb[2] > 0.0 { rgb[0] / rgb[2] } else { 1.0 };

        // Map ratio to temperature (rough approximation)
        if ratio > 1.0 {
            // Warm (red > blue)
            2000.0 + (ratio - 1.0) * 5000.0
        } else {
            // Cool (blue > red)
            10000.0 - (1.0 - ratio) * 5000.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wb_settings_creation() {
        let settings = WhiteBalanceSettings::new(0);
        assert_eq!(settings.angle, 0);
        assert_eq!(settings.temperature, 6500.0);
    }

    #[test]
    fn test_apply_white_balance() {
        let mut settings = WhiteBalanceSettings::new(0);
        settings.red_gain = 1.2;
        settings.green_gain = 1.0;
        settings.blue_gain = 0.8;

        let rgb = [0.5, 0.5, 0.5];
        let result = settings.apply(rgb);

        assert!(result[0] > rgb[0]); // Red boosted
        assert!(result[2] < rgb[2]); // Blue reduced
    }

    #[test]
    fn test_calculate_gains() {
        let mut settings = WhiteBalanceSettings::new(0);
        settings.temperature = 3000.0; // Warm
        settings.calculate_gains();

        assert!(settings.red_gain < settings.blue_gain); // Warm light needs less red
    }

    #[test]
    fn test_wb_matcher_creation() {
        let matcher = WhiteBalanceMatching::new(3, 0);
        assert_eq!(matcher.reference_angle(), 0);
        assert_eq!(matcher.settings.len(), 3);
    }

    #[test]
    fn test_update_settings() {
        let mut matcher = WhiteBalanceMatching::new(3, 0);
        let mut settings = WhiteBalanceSettings::new(1);
        settings.temperature = 5000.0;

        matcher.update_settings(settings);
        assert_eq!(
            matcher
                .get_settings(1)
                .expect("multicam test operation should succeed")
                .temperature,
            5000.0
        );
    }

    #[test]
    fn test_match_to_reference() {
        let mut matcher = WhiteBalanceMatching::new(3, 0);
        let mut ref_settings = WhiteBalanceSettings::new(0);
        ref_settings.temperature = 5500.0;
        matcher.update_settings(ref_settings);

        assert!(matcher.match_to_reference().is_ok());

        // All angles should have same temperature as reference
        for i in 1..3 {
            assert_eq!(
                matcher
                    .get_settings(i)
                    .expect("multicam test operation should succeed")
                    .temperature,
                5500.0
            );
        }
    }

    #[test]
    fn test_detect_gray_card() {
        let matcher = WhiteBalanceMatching::new(1, 0);
        let image = vec![128u8; 1920 * 1080 * 3]; // Gray image

        let gray = matcher.detect_gray_card(&image, 1920, 1080);
        assert!(gray.is_some());

        let rgb = gray.expect("multicam test operation should succeed");
        assert!((rgb[0] - 0.5).abs() < 0.01);
        assert!((rgb[1] - 0.5).abs() < 0.01);
        assert!((rgb[2] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_calculate_from_gray() {
        let matcher = WhiteBalanceMatching::new(1, 0);
        let gray = [0.4, 0.5, 0.6]; // Bluish gray

        let settings = matcher.calculate_from_gray(gray);
        assert!(settings.blue_gain < settings.red_gain); // Need to reduce blue
    }

    #[test]
    fn test_estimate_temperature() {
        let warm_rgb = [0.8, 0.6, 0.4]; // Warm colors
        let cool_rgb = [0.4, 0.6, 0.8]; // Cool colors

        let warm_temp = WhiteBalanceMatching::estimate_temperature(warm_rgb);
        let cool_temp = WhiteBalanceMatching::estimate_temperature(cool_rgb);

        // Warm (red-biased) colors map to lower Kelvin values; cool (blue-biased) to higher
        assert!(cool_temp > warm_temp);
    }
}
