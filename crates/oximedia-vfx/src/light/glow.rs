//! Glow effect.

use crate::{EffectParams, Frame, VfxResult, VideoEffect};
use serde::{Deserialize, Serialize};

/// Glow mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GlowMode {
    /// Soft glow.
    Soft,
    /// Hard glow.
    Hard,
    /// Outer glow.
    Outer,
}

/// Glow effect.
pub struct Glow {
    mode: GlowMode,
    radius: u32,
    intensity: f32,
    threshold: u8,
}

impl Glow {
    /// Create a new glow effect.
    #[must_use]
    pub const fn new(mode: GlowMode) -> Self {
        Self {
            mode,
            radius: 10,
            intensity: 1.0,
            threshold: 128,
        }
    }

    /// Set glow radius.
    #[must_use]
    pub fn with_radius(mut self, radius: u32) -> Self {
        self.radius = radius.max(1);
        self
    }

    /// Set glow intensity.
    #[must_use]
    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity.max(0.0);
        self
    }

    /// Set brightness threshold.
    #[must_use]
    pub const fn with_threshold(mut self, threshold: u8) -> Self {
        self.threshold = threshold;
        self
    }
}

impl VideoEffect for Glow {
    fn name(&self) -> &'static str {
        "Glow"
    }

    fn description(&self) -> &'static str {
        "Glow effect with blur"
    }

    fn apply(
        &mut self,
        input: &Frame,
        output: &mut Frame,
        _params: &EffectParams,
    ) -> VfxResult<()> {
        // Copy input to output first
        for y in 0..output.height {
            for x in 0..output.width {
                let pixel = input.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
                output.set_pixel(x, y, pixel);
            }
        }

        // Apply glow
        let radius = self.radius as i32;
        for y in 0..output.height {
            for x in 0..output.width {
                let pixel = input.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
                let brightness =
                    (u32::from(pixel[0]) + u32::from(pixel[1]) + u32::from(pixel[2])) / 3;

                if brightness as u8 > self.threshold {
                    // Add glow around bright pixels
                    for dy in -radius..=radius {
                        for dx in -radius..=radius {
                            let nx = (x as i32 + dx).max(0).min(output.width as i32 - 1) as u32;
                            let ny = (y as i32 + dy).max(0).min(output.height as i32 - 1) as u32;

                            let dist = ((dx * dx + dy * dy) as f32).sqrt();
                            let falloff = (1.0 - dist / radius as f32).max(0.0);
                            let glow_amount = falloff * self.intensity;

                            let Some(current) = output.get_pixel(nx, ny) else {
                                continue;
                            };
                            let glowed = [
                                current[0]
                                    .saturating_add((f32::from(pixel[0]) * glow_amount) as u8),
                                current[1]
                                    .saturating_add((f32::from(pixel[1]) * glow_amount) as u8),
                                current[2]
                                    .saturating_add((f32::from(pixel[2]) * glow_amount) as u8),
                                current[3],
                            ];
                            output.set_pixel(nx, ny, glowed);
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
    fn test_glow() {
        let mut glow = Glow::new(GlowMode::Soft).with_radius(5).with_intensity(0.5);

        let input = Frame::new(50, 50).expect("should succeed in test");
        let mut output = Frame::new(50, 50).expect("should succeed in test");
        let params = EffectParams::new();
        glow.apply(&input, &mut output, &params)
            .expect("should succeed in test");
    }
}
