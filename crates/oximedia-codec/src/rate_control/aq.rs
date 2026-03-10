//! Adaptive Quantization (AQ) module.
//!
//! Adaptive Quantization adjusts QP on a per-block basis to improve
//! perceptual quality. Different regions of a frame may benefit from
//! different quantization levels:
//!
//! - High detail areas: Lower QP for better quality
//! - Low detail areas: Higher QP to save bits
//! - Dark regions: Special handling to avoid banding
//! - Bright regions: Can tolerate more compression

#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::unused_self)]
#![allow(clippy::if_not_else)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::needless_pass_by_value)]
#![forbid(unsafe_code)]

/// Adaptive Quantization controller.
#[derive(Clone, Debug)]
pub struct AdaptiveQuantization {
    /// Frame width.
    width: u32,
    /// Frame height.
    height: u32,
    /// Block size for AQ analysis.
    block_size: u32,
    /// AQ mode.
    mode: AqMode,
    /// AQ strength (0.0-2.0).
    strength: f32,
    /// Enable dark region boost.
    dark_boost: bool,
    /// Dark threshold (0-255).
    dark_threshold: u8,
    /// Bright threshold (0-255).
    bright_threshold: u8,
    /// Enable psychovisual optimization.
    psy_enabled: bool,
    /// Psychovisual strength.
    psy_strength: f32,
}

/// AQ operation mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AqMode {
    /// No adaptive quantization.
    None,
    /// Variance-based AQ.
    #[default]
    Variance,
    /// Auto-variance AQ (adaptive strength).
    AutoVariance,
    /// Psychovisual AQ.
    Psychovisual,
    /// Combined variance and psychovisual.
    Combined,
}

impl AdaptiveQuantization {
    /// Create a new AQ controller.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            block_size: 16,
            mode: AqMode::Variance,
            strength: 1.0,
            dark_boost: true,
            dark_threshold: 40,
            bright_threshold: 220,
            psy_enabled: false,
            psy_strength: 1.0,
        }
    }

    /// Set AQ mode.
    pub fn set_mode(&mut self, mode: AqMode) {
        self.mode = mode;
        self.psy_enabled = matches!(mode, AqMode::Psychovisual | AqMode::Combined);
    }

    /// Set AQ strength.
    pub fn set_strength(&mut self, strength: f32) {
        self.strength = strength.clamp(0.0, 2.0);
    }

    /// Enable or disable dark region boost.
    pub fn set_dark_boost(&mut self, enable: bool) {
        self.dark_boost = enable;
    }

    /// Set dark and bright thresholds.
    pub fn set_thresholds(&mut self, dark: u8, bright: u8) {
        self.dark_threshold = dark;
        self.bright_threshold = bright;
    }

    /// Set psychovisual strength.
    pub fn set_psy_strength(&mut self, strength: f32) {
        self.psy_strength = strength.clamp(0.0, 2.0);
    }

    /// Calculate per-block QP offsets for a frame.
    #[must_use]
    pub fn calculate_offsets(&self, luma: &[u8], stride: usize) -> AqResult {
        if self.mode == AqMode::None {
            return AqResult::default();
        }

        let blocks_x = self.width / self.block_size;
        let blocks_y = self.height / self.block_size;
        let total_blocks = (blocks_x * blocks_y) as usize;

        if total_blocks == 0 {
            return AqResult::default();
        }

        let mut offsets = Vec::with_capacity(total_blocks);
        let mut variances = Vec::with_capacity(total_blocks);
        let mut energies = Vec::with_capacity(total_blocks);

        // First pass: calculate block statistics
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let stats = self.calculate_block_stats(luma, stride, bx, by);
                variances.push(stats.variance);
                energies.push(stats.energy);
            }
        }

        // Calculate reference values
        let avg_variance = self.calculate_average(&variances);
        let avg_energy = self.calculate_average(&energies);

        // Second pass: calculate QP offsets
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let stats = self.calculate_block_stats(luma, stride, bx, by);
                let offset = self.calculate_block_offset(&stats, avg_variance, avg_energy);
                offsets.push(offset);
            }
        }

        AqResult {
            offsets,
            blocks_x,
            blocks_y,
            avg_variance,
            avg_energy,
        }
    }

    /// Calculate statistics for a single block.
    fn calculate_block_stats(&self, luma: &[u8], stride: usize, bx: u32, by: u32) -> BlockStats {
        let start_x = (bx * self.block_size) as usize;
        let start_y = (by * self.block_size) as usize;
        let block_size = self.block_size as usize;

        let mut sum = 0u64;
        let mut sum_sq = 0u64;
        let mut min_val = 255u8;
        let mut max_val = 0u8;
        let mut count = 0u32;

        for y in 0..block_size {
            let row_start = (start_y + y) * stride + start_x;
            if row_start + block_size > luma.len() {
                continue;
            }

            for x in 0..block_size {
                let pixel = luma[row_start + x];
                sum += pixel as u64;
                sum_sq += (pixel as u64) * (pixel as u64);
                min_val = min_val.min(pixel);
                max_val = max_val.max(pixel);
                count += 1;
            }
        }

        if count == 0 {
            return BlockStats::default();
        }

        let mean = sum as f32 / count as f32;
        let mean_sq = sum_sq as f32 / count as f32;
        let variance = (mean_sq - mean * mean).max(0.0);

        // Calculate AC energy (sum of squared differences from mean)
        let energy = variance * count as f32;

        // Calculate edge strength (simplified)
        let edge_strength = (max_val - min_val) as f32;

        BlockStats {
            mean,
            variance,
            energy,
            edge_strength,
            min: min_val,
            max: max_val,
        }
    }

    /// Calculate QP offset for a block.
    fn calculate_block_offset(
        &self,
        stats: &BlockStats,
        avg_variance: f32,
        avg_energy: f32,
    ) -> f32 {
        let mut offset = match self.mode {
            AqMode::None => return 0.0,
            AqMode::Variance => self.variance_offset(stats.variance, avg_variance),
            AqMode::AutoVariance => {
                let auto_strength = self.calculate_auto_strength(stats.variance, avg_variance);
                self.variance_offset(stats.variance, avg_variance) * auto_strength
            }
            AqMode::Psychovisual => self.psychovisual_offset(stats, avg_energy),
            AqMode::Combined => {
                let var_offset = self.variance_offset(stats.variance, avg_variance);
                let psy_offset = self.psychovisual_offset(stats, avg_energy);
                var_offset * 0.5 + psy_offset * 0.5
            }
        };

        // Apply dark region boost
        if self.dark_boost && stats.mean < self.dark_threshold as f32 {
            let dark_factor = 1.0 - (stats.mean / self.dark_threshold as f32);
            offset -= self.strength * dark_factor * 2.0;
        }

        // Apply bright region handling
        if stats.mean > self.bright_threshold as f32 {
            let bright_factor = (stats.mean - self.bright_threshold as f32)
                / (255.0 - self.bright_threshold as f32);
            offset += self.strength * bright_factor * 1.0;
        }

        // Clamp to reasonable range
        offset.clamp(-6.0, 6.0)
    }

    /// Calculate variance-based QP offset.
    fn variance_offset(&self, variance: f32, avg_variance: f32) -> f32 {
        if avg_variance <= 0.0 {
            return 0.0;
        }

        // Log-based offset calculation
        // High variance (detail) -> negative offset (lower QP, better quality)
        // Low variance (flat) -> positive offset (higher QP, save bits)
        let ratio = variance / avg_variance;
        let log_ratio = ratio.ln();

        -log_ratio * self.strength * 2.0
    }

    /// Calculate auto-adjusted strength.
    fn calculate_auto_strength(&self, variance: f32, avg_variance: f32) -> f32 {
        // Reduce strength for very high or very low variance
        let ratio = variance / avg_variance.max(1.0);

        if !(0.1..=10.0).contains(&ratio) {
            0.5
        } else {
            1.0
        }
    }

    /// Calculate psychovisual QP offset.
    fn psychovisual_offset(&self, stats: &BlockStats, avg_energy: f32) -> f32 {
        if avg_energy <= 0.0 {
            return 0.0;
        }

        // Psychovisual model considers:
        // - Texture masking: high detail areas can hide artifacts
        // - Edge preservation: edges are perceptually important

        let energy_ratio = stats.energy / avg_energy;
        let energy_offset = -energy_ratio.ln() * self.psy_strength;

        // Edge importance factor
        let edge_factor = (stats.edge_strength / 128.0).min(1.0);
        let edge_offset = -edge_factor * self.psy_strength;

        (energy_offset + edge_offset) * 0.5
    }

    /// Calculate average of a slice.
    fn calculate_average(&self, values: &[f32]) -> f32 {
        if values.is_empty() {
            return 1.0;
        }
        values.iter().sum::<f32>() / values.len() as f32
    }

    /// Get AQ mode.
    #[must_use]
    pub fn mode(&self) -> AqMode {
        self.mode
    }

    /// Get AQ strength.
    #[must_use]
    pub fn strength(&self) -> f32 {
        self.strength
    }

    /// Check if AQ is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.mode != AqMode::None
    }
}

impl Default for AdaptiveQuantization {
    fn default() -> Self {
        Self::new(1920, 1080)
    }
}

/// Block statistics for AQ calculation.
#[derive(Clone, Copy, Debug, Default)]
struct BlockStats {
    /// Mean pixel value.
    mean: f32,
    /// Variance.
    variance: f32,
    /// AC energy.
    energy: f32,
    /// Edge strength.
    edge_strength: f32,
    /// Minimum pixel value (reserved for future use).
    #[allow(dead_code)]
    min: u8,
    /// Maximum pixel value (reserved for future use).
    #[allow(dead_code)]
    max: u8,
}

/// Result of AQ calculation.
#[derive(Clone, Debug, Default)]
pub struct AqResult {
    /// Per-block QP offsets.
    pub offsets: Vec<f32>,
    /// Number of blocks horizontally.
    pub blocks_x: u32,
    /// Number of blocks vertically.
    pub blocks_y: u32,
    /// Average variance.
    pub avg_variance: f32,
    /// Average energy.
    pub avg_energy: f32,
}

impl AqResult {
    /// Get offset for a specific block.
    #[must_use]
    pub fn get_offset(&self, bx: u32, by: u32) -> f32 {
        if bx >= self.blocks_x || by >= self.blocks_y {
            return 0.0;
        }
        let idx = (by * self.blocks_x + bx) as usize;
        self.offsets.get(idx).copied().unwrap_or(0.0)
    }

    /// Get offset at pixel coordinates.
    #[must_use]
    pub fn get_offset_at_pixel(&self, x: u32, y: u32, block_size: u32) -> f32 {
        let bx = x / block_size;
        let by = y / block_size;
        self.get_offset(bx, by)
    }

    /// Get total number of blocks.
    #[must_use]
    pub fn total_blocks(&self) -> usize {
        self.offsets.len()
    }

    /// Get average offset.
    #[must_use]
    pub fn average_offset(&self) -> f32 {
        if self.offsets.is_empty() {
            return 0.0;
        }
        self.offsets.iter().sum::<f32>() / self.offsets.len() as f32
    }

    /// Get offset range (min, max).
    #[must_use]
    pub fn offset_range(&self) -> (f32, f32) {
        if self.offsets.is_empty() {
            return (0.0, 0.0);
        }

        let min = self
            .offsets
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        let max = self
            .offsets
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        (min, max)
    }
}

/// AQ strength presets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AqStrength {
    /// No AQ.
    Off,
    /// Light AQ adjustment.
    Light,
    /// Medium AQ (default).
    Medium,
    /// Strong AQ adjustment.
    Strong,
    /// Maximum AQ adjustment.
    Maximum,
}

impl AqStrength {
    /// Get strength value.
    #[must_use]
    pub fn to_strength(self) -> f32 {
        match self {
            Self::Off => 0.0,
            Self::Light => 0.5,
            Self::Medium => 1.0,
            Self::Strong => 1.5,
            Self::Maximum => 2.0,
        }
    }

    /// Get AQ mode for this strength.
    #[must_use]
    pub fn to_mode(self) -> AqMode {
        if self == Self::Off {
            AqMode::None
        } else {
            AqMode::Variance
        }
    }
}

impl Default for AqStrength {
    fn default() -> Self {
        Self::Medium
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_uniform_frame(width: u32, height: u32, value: u8) -> Vec<u8> {
        vec![value; (width * height) as usize]
    }

    fn create_gradient_frame(width: u32, height: u32) -> Vec<u8> {
        let mut frame = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                frame.push(((x + y) % 256) as u8);
            }
        }
        frame
    }

    fn create_half_frame(width: u32, height: u32) -> Vec<u8> {
        let mut frame = Vec::with_capacity((width * height) as usize);
        for _y in 0..height {
            for x in 0..width {
                if x < width / 2 {
                    frame.push(50); // Dark left
                } else {
                    frame.push(200); // Bright right
                }
            }
        }
        frame
    }

    #[test]
    fn test_aq_creation() {
        let aq = AdaptiveQuantization::new(1920, 1080);
        assert!(aq.is_enabled());
        assert_eq!(aq.mode(), AqMode::Variance);
    }

    #[test]
    fn test_aq_disabled() {
        let mut aq = AdaptiveQuantization::new(64, 64);
        aq.set_mode(AqMode::None);

        let frame = create_gradient_frame(64, 64);
        let result = aq.calculate_offsets(&frame, 64);

        assert!(result.offsets.is_empty() || result.offsets.iter().all(|&o| o == 0.0));
    }

    #[test]
    fn test_uniform_frame_offsets() {
        let mut aq = AdaptiveQuantization::new(64, 64);
        aq.set_dark_boost(false);

        let frame = create_uniform_frame(64, 64, 128);
        let result = aq.calculate_offsets(&frame, 64);

        // Uniform frame should have near-zero offsets
        for offset in &result.offsets {
            assert!(
                offset.abs() < 1.0,
                "Offset {} too large for uniform frame",
                offset
            );
        }
    }

    #[test]
    fn test_gradient_frame_offsets() {
        let mut aq = AdaptiveQuantization::new(64, 64);
        aq.set_dark_boost(false);

        let frame = create_gradient_frame(64, 64);
        let result = aq.calculate_offsets(&frame, 64);

        // Should have some non-zero offsets
        assert!(!result.offsets.is_empty());
    }

    #[test]
    fn test_dark_boost() {
        let mut aq = AdaptiveQuantization::new(64, 64);
        aq.set_dark_boost(true);
        aq.set_thresholds(100, 200);

        let frame = create_half_frame(64, 64);
        let result = aq.calculate_offsets(&frame, 64);

        // Check that dark blocks get negative (quality boost) offsets
        let dark_offset = result.get_offset(0, 0); // Left side (dark)
        let bright_offset = result.get_offset(result.blocks_x - 1, 0); // Right side (bright)

        // Dark regions should have lower (more negative) offsets than bright
        assert!(dark_offset < bright_offset);
    }

    #[test]
    fn test_strength_setting() {
        let mut aq = AdaptiveQuantization::new(64, 64);

        aq.set_strength(0.5);
        assert!((aq.strength() - 0.5).abs() < f32::EPSILON);

        aq.set_strength(3.0); // Should clamp to 2.0
        assert!((aq.strength() - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_aq_result_methods() {
        let result = AqResult {
            offsets: vec![-1.0, 0.0, 1.0, 2.0],
            blocks_x: 2,
            blocks_y: 2,
            avg_variance: 100.0,
            avg_energy: 1000.0,
        };

        assert_eq!(result.total_blocks(), 4);
        assert!((result.average_offset() - 0.5).abs() < f32::EPSILON);

        let (min, max) = result.offset_range();
        assert!((min - (-1.0)).abs() < f32::EPSILON);
        assert!((max - 2.0).abs() < f32::EPSILON);

        assert!((result.get_offset(0, 0) - (-1.0)).abs() < f32::EPSILON);
        assert!((result.get_offset(1, 1) - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_aq_modes() {
        let mut aq = AdaptiveQuantization::new(64, 64);
        let frame = create_gradient_frame(64, 64);

        for mode in [
            AqMode::Variance,
            AqMode::AutoVariance,
            AqMode::Psychovisual,
            AqMode::Combined,
        ] {
            aq.set_mode(mode);
            let result = aq.calculate_offsets(&frame, 64);
            assert!(!result.offsets.is_empty());
        }
    }

    #[test]
    fn test_aq_strength_presets() {
        assert!((AqStrength::Off.to_strength() - 0.0).abs() < f32::EPSILON);
        assert!((AqStrength::Medium.to_strength() - 1.0).abs() < f32::EPSILON);
        assert!((AqStrength::Maximum.to_strength() - 2.0).abs() < f32::EPSILON);

        assert_eq!(AqStrength::Off.to_mode(), AqMode::None);
        assert_eq!(AqStrength::Medium.to_mode(), AqMode::Variance);
    }

    #[test]
    fn test_get_offset_at_pixel() {
        let result = AqResult {
            offsets: vec![1.0, 2.0, 3.0, 4.0],
            blocks_x: 2,
            blocks_y: 2,
            avg_variance: 100.0,
            avg_energy: 1000.0,
        };

        // Block size of 32: pixel (0,0) is block (0,0), pixel (33,0) is block (1,0)
        assert!((result.get_offset_at_pixel(0, 0, 32) - 1.0).abs() < f32::EPSILON);
        assert!((result.get_offset_at_pixel(33, 0, 32) - 2.0).abs() < f32::EPSILON);
        assert!((result.get_offset_at_pixel(0, 33, 32) - 3.0).abs() < f32::EPSILON);
        assert!((result.get_offset_at_pixel(33, 33, 32) - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_offset_bounds() {
        let mut aq = AdaptiveQuantization::new(64, 64);
        aq.set_strength(2.0);
        aq.set_dark_boost(true);

        // Create extreme frame
        let mut frame = vec![0u8; 64 * 64];
        for i in 0..(64 * 32) {
            frame[i] = 10; // Very dark top half
        }
        for i in (64 * 32)..(64 * 64) {
            frame[i] = 250; // Very bright bottom half
        }

        let result = aq.calculate_offsets(&frame, 64);

        // All offsets should be clamped to [-6, 6]
        for offset in &result.offsets {
            assert!(
                *offset >= -6.0 && *offset <= 6.0,
                "Offset {} out of bounds",
                offset
            );
        }
    }
}
