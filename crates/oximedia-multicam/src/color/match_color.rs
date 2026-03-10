//! Color matching across camera angles.

use super::{ColorMatrix, ColorStats};
use crate::{AngleId, Result};

/// Color matcher
#[derive(Debug)]
pub struct ColorMatcher {
    /// Reference angle for color matching
    reference_angle: AngleId,
    /// Color statistics for each angle
    stats: Vec<ColorStats>,
    /// Color correction matrices
    corrections: Vec<ColorMatrix>,
}

impl ColorMatcher {
    /// Create a new color matcher
    #[must_use]
    pub fn new(angle_count: usize, reference_angle: AngleId) -> Self {
        Self {
            reference_angle,
            stats: (0..angle_count).map(ColorStats::new).collect(),
            corrections: vec![ColorMatrix::identity(); angle_count],
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

    /// Update color statistics for an angle
    pub fn update_stats(&mut self, stats: ColorStats) {
        if stats.angle < self.stats.len() {
            self.stats[stats.angle] = stats;
        }
    }

    /// Calculate color correction for all angles
    ///
    /// # Errors
    ///
    /// Returns an error if calculation fails
    pub fn calculate_corrections(&mut self) -> Result<()> {
        if self.reference_angle >= self.stats.len() {
            return Err(crate::MultiCamError::AngleNotFound(self.reference_angle));
        }

        let reference = &self.stats[self.reference_angle];

        for (angle, stats) in self.stats.iter().enumerate() {
            if angle == self.reference_angle {
                self.corrections[angle] = ColorMatrix::identity();
                continue;
            }

            // Calculate color correction matrix
            let correction = self.calculate_correction_matrix(stats, reference);
            self.corrections[angle] = correction;
        }

        Ok(())
    }

    /// Calculate correction matrix from source to target statistics
    fn calculate_correction_matrix(&self, source: &ColorStats, target: &ColorStats) -> ColorMatrix {
        // Simple scaling correction
        let mut matrix = [[0.0; 3]; 3];

        for i in 0..3 {
            if source.mean_rgb[i] > 0.0 {
                matrix[i][i] = target.mean_rgb[i] / source.mean_rgb[i];
            } else {
                matrix[i][i] = 1.0;
            }
        }

        ColorMatrix { matrix }
    }

    /// Get correction matrix for angle
    #[must_use]
    pub fn get_correction(&self, angle: AngleId) -> Option<&ColorMatrix> {
        self.corrections.get(angle)
    }

    /// Apply color correction to RGB values
    #[must_use]
    pub fn apply_correction(&self, angle: AngleId, rgb: [f32; 3]) -> [f32; 3] {
        if let Some(matrix) = self.get_correction(angle) {
            let corrected = matrix.apply(rgb);
            [
                corrected[0].clamp(0.0, 1.0),
                corrected[1].clamp(0.0, 1.0),
                corrected[2].clamp(0.0, 1.0),
            ]
        } else {
            rgb
        }
    }

    /// Get color statistics for angle
    #[must_use]
    pub fn get_stats(&self, angle: AngleId) -> Option<&ColorStats> {
        self.stats.get(angle)
    }

    /// Calculate average color statistics across all angles
    #[must_use]
    pub fn average_stats(&self) -> ColorStats {
        let mut avg = ColorStats::new(0);
        // Reset to zero so we can accumulate without the default values
        avg.mean_rgb = [0.0, 0.0, 0.0];
        avg.std_rgb = [0.0, 0.0, 0.0];
        avg.temperature = 0.0;
        avg.tint = 0.0;

        let count = self.stats.len() as f32;

        for stats in &self.stats {
            for i in 0..3 {
                avg.mean_rgb[i] += stats.mean_rgb[i] / count;
                avg.std_rgb[i] += stats.std_rgb[i] / count;
            }
            avg.temperature += stats.temperature / count;
            avg.tint += stats.tint / count;
        }

        avg
    }

    /// Check color consistency across angles
    #[must_use]
    pub fn check_consistency(&self, threshold: f32) -> bool {
        if self.reference_angle >= self.stats.len() {
            return false;
        }

        let reference = &self.stats[self.reference_angle];

        for stats in &self.stats {
            if stats.angle != self.reference_angle {
                let distance = stats.distance_to(reference);
                if distance > threshold {
                    return false;
                }
            }
        }

        true
    }

    /// Get angles that need color correction
    #[must_use]
    pub fn angles_needing_correction(&self, threshold: f32) -> Vec<AngleId> {
        if self.reference_angle >= self.stats.len() {
            return Vec::new();
        }

        let reference = &self.stats[self.reference_angle];
        let mut angles = Vec::new();

        for stats in &self.stats {
            if stats.angle != self.reference_angle {
                let distance = stats.distance_to(reference);
                if distance > threshold {
                    angles.push(stats.angle);
                }
            }
        }

        angles
    }

    /// Reset all corrections
    pub fn reset_corrections(&mut self) {
        for correction in &mut self.corrections {
            *correction = ColorMatrix::identity();
        }
    }
}

/// Color transfer method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorTransferMethod {
    /// Mean matching
    Mean,
    /// Mean and standard deviation matching
    MeanStd,
    /// Histogram matching
    Histogram,
    /// 3D LUT-based transfer
    Lut3D,
}

/// Advanced color matcher with multiple methods
#[derive(Debug)]
pub struct AdvancedColorMatcher {
    /// Base matcher
    matcher: ColorMatcher,
    /// Transfer method
    method: ColorTransferMethod,
}

impl AdvancedColorMatcher {
    /// Create a new advanced color matcher
    #[must_use]
    pub fn new(angle_count: usize, reference_angle: AngleId, method: ColorTransferMethod) -> Self {
        Self {
            matcher: ColorMatcher::new(angle_count, reference_angle),
            method,
        }
    }

    /// Set transfer method
    pub fn set_method(&mut self, method: ColorTransferMethod) {
        self.method = method;
    }

    /// Get transfer method
    #[must_use]
    pub fn method(&self) -> ColorTransferMethod {
        self.method
    }

    /// Get base matcher
    #[must_use]
    pub fn matcher(&self) -> &ColorMatcher {
        &self.matcher
    }

    /// Get mutable base matcher
    pub fn matcher_mut(&mut self) -> &mut ColorMatcher {
        &mut self.matcher
    }

    /// Apply color transfer using selected method
    #[must_use]
    pub fn transfer_color(&self, angle: AngleId, rgb: [f32; 3]) -> [f32; 3] {
        match self.method {
            ColorTransferMethod::Mean | ColorTransferMethod::MeanStd => {
                self.matcher.apply_correction(angle, rgb)
            }
            ColorTransferMethod::Histogram => {
                // Placeholder for histogram matching
                self.matcher.apply_correction(angle, rgb)
            }
            ColorTransferMethod::Lut3D => {
                // Placeholder for 3D LUT
                self.matcher.apply_correction(angle, rgb)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_matcher_creation() {
        let matcher = ColorMatcher::new(3, 0);
        assert_eq!(matcher.reference_angle(), 0);
        assert_eq!(matcher.stats.len(), 3);
    }

    #[test]
    fn test_update_stats() {
        let mut matcher = ColorMatcher::new(3, 0);
        let mut stats = ColorStats::new(1);
        stats.mean_rgb = [0.8, 0.7, 0.6];

        matcher.update_stats(stats);
        assert_eq!(
            matcher
                .get_stats(1)
                .expect("multicam test operation should succeed")
                .mean_rgb,
            [0.8, 0.7, 0.6]
        );
    }

    #[test]
    fn test_calculate_corrections() {
        let mut matcher = ColorMatcher::new(2, 0);
        let result = matcher.calculate_corrections();
        assert!(result.is_ok());
    }

    #[test]
    fn test_apply_correction() {
        let matcher = ColorMatcher::new(2, 0);
        let rgb = [1.0, 0.5, 0.25];
        let corrected = matcher.apply_correction(0, rgb);
        assert_eq!(corrected, rgb); // Identity for reference angle
    }

    #[test]
    fn test_average_stats() {
        let mut matcher = ColorMatcher::new(2, 0);

        let mut stats1 = ColorStats::new(0);
        stats1.mean_rgb = [0.4, 0.4, 0.4];
        matcher.update_stats(stats1);

        let mut stats2 = ColorStats::new(1);
        stats2.mean_rgb = [0.6, 0.6, 0.6];
        matcher.update_stats(stats2);

        let avg = matcher.average_stats();
        assert!((avg.mean_rgb[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_check_consistency() {
        let matcher = ColorMatcher::new(3, 0);
        assert!(matcher.check_consistency(1.0)); // Should be consistent with large threshold
    }

    #[test]
    fn test_angles_needing_correction() {
        let mut matcher = ColorMatcher::new(3, 0);

        let mut stats = ColorStats::new(1);
        stats.mean_rgb = [1.0, 1.0, 1.0]; // Very different from default
        matcher.update_stats(stats);

        let angles = matcher.angles_needing_correction(0.1);
        assert!(!angles.is_empty());
    }

    #[test]
    fn test_advanced_matcher() {
        let matcher = AdvancedColorMatcher::new(3, 0, ColorTransferMethod::Mean);
        assert_eq!(matcher.method(), ColorTransferMethod::Mean);
    }

    #[test]
    fn test_transfer_color() {
        let matcher = AdvancedColorMatcher::new(2, 0, ColorTransferMethod::Mean);
        let rgb = [0.8, 0.6, 0.4];
        let transferred = matcher.transfer_color(0, rgb);
        assert!(transferred[0] <= 1.0);
        assert!(transferred[1] <= 1.0);
        assert!(transferred[2] <= 1.0);
    }
}
