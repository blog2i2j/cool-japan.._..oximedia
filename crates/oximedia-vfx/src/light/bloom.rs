//! Bloom effect for HDR.

use crate::{EffectParams, Frame, VfxResult, VideoEffect};
use serde::{Deserialize, Serialize};

/// Bloom quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BloomQuality {
    /// Low quality (fast).
    Low,
    /// Medium quality.
    Medium,
    /// High quality (slow).
    High,
}

/// Bloom effect.
pub struct Bloom {
    quality: BloomQuality,
    threshold: f32,
    intensity: f32,
    radius: u32,
}

impl Bloom {
    /// Create a new bloom effect.
    #[must_use]
    pub const fn new(quality: BloomQuality) -> Self {
        Self {
            quality,
            threshold: 0.8,
            intensity: 1.0,
            radius: 20,
        }
    }

    /// Set bloom threshold (0.0 - 1.0).
    #[must_use]
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set bloom intensity.
    #[must_use]
    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity.max(0.0);
        self
    }

    /// Set bloom radius.
    #[must_use]
    pub fn with_radius(mut self, radius: u32) -> Self {
        self.radius = radius.max(1);
        self
    }
}

impl VideoEffect for Bloom {
    fn name(&self) -> &'static str {
        "Bloom"
    }

    fn description(&self) -> &'static str {
        "HDR bloom effect"
    }

    fn apply(
        &mut self,
        input: &Frame,
        output: &mut Frame,
        _params: &EffectParams,
    ) -> VfxResult<()> {
        // Copy input
        for y in 0..output.height {
            for x in 0..output.width {
                output.set_pixel(x, y, input.get_pixel(x, y).unwrap_or([0, 0, 0, 0]));
            }
        }

        // Extract bright areas and apply bloom
        let threshold_value = (self.threshold * 255.0) as u8;
        let radius = self.radius as i32;

        for y in 0..output.height {
            for x in 0..output.width {
                let pixel = input.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
                let max_channel = pixel[0].max(pixel[1]).max(pixel[2]);

                if max_channel > threshold_value {
                    let bloom_strength = (f32::from(max_channel - threshold_value)
                        / f32::from(255 - threshold_value))
                        * self.intensity;

                    for dy in -radius..=radius {
                        for dx in -radius..=radius {
                            let nx = (x as i32 + dx).max(0).min(output.width as i32 - 1) as u32;
                            let ny = (y as i32 + dy).max(0).min(output.height as i32 - 1) as u32;

                            let dist = ((dx * dx + dy * dy) as f32).sqrt();
                            let falloff = (1.0 - dist / radius as f32).max(0.0);
                            let bloom_amount = bloom_strength * falloff;

                            let Some(current) = output.get_pixel(nx, ny) else {
                                continue;
                            };
                            let bloomed = [
                                current[0]
                                    .saturating_add((f32::from(pixel[0]) * bloom_amount) as u8),
                                current[1]
                                    .saturating_add((f32::from(pixel[1]) * bloom_amount) as u8),
                                current[2]
                                    .saturating_add((f32::from(pixel[2]) * bloom_amount) as u8),
                                current[3],
                            ];
                            output.set_pixel(nx, ny, bloomed);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn supports_gpu(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom() {
        let mut bloom = Bloom::new(BloomQuality::Medium)
            .with_threshold(0.7)
            .with_intensity(1.5);

        let input = Frame::new(50, 50).expect("should succeed in test");
        let mut output = Frame::new(50, 50).expect("should succeed in test");
        let params = EffectParams::new();
        bloom
            .apply(&input, &mut output, &params)
            .expect("should succeed in test");
    }
}
