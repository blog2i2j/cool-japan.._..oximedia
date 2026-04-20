//! Snow particle system.

use super::{Particle, ParticleSystem};
use crate::{Color, EffectParams, Frame, VfxResult, VideoEffect};
use rand::RngExt;
use serde::{Deserialize, Serialize};

/// Snow style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SnowStyle {
    /// Light snowfall.
    Light,
    /// Heavy snowfall.
    Heavy,
    /// Blizzard.
    Blizzard,
}

/// Snow effect.
pub struct Snow {
    style: SnowStyle,
    system: ParticleSystem,
    wind: f32,
}

impl Snow {
    /// Create a new snow effect.
    #[must_use]
    pub fn new(style: SnowStyle) -> Self {
        let spawn_rate = match style {
            SnowStyle::Light => 50.0,
            SnowStyle::Heavy => 150.0,
            SnowStyle::Blizzard => 300.0,
        };

        let mut system = ParticleSystem::new(0);
        system.set_spawn_rate(spawn_rate);

        Self {
            style,
            system,
            wind: 0.0,
        }
    }

    /// Set wind strength (-1.0 to 1.0).
    #[must_use]
    pub fn with_wind(mut self, wind: f32) -> Self {
        self.wind = wind.clamp(-1.0, 1.0);
        self
    }

    fn spawn_snowflake(&mut self, width: u32, _height: u32) {
        let x = self.system.rng().random_range(0.0..width as f32);
        let y = -10.0;

        let mut particle = Particle::new(x, y);
        particle.vx = self.wind * 50.0;
        particle.vy = self.system.rng().random_range(20.0..80.0);
        particle.size = self.system.rng().random_range(1.0..4.0);
        particle.life = self.system.rng().random_range(5.0..15.0);
        particle.max_life = particle.life;

        self.system.spawn(particle);
    }
}

impl VideoEffect for Snow {
    fn name(&self) -> &'static str {
        "Snow"
    }

    fn description(&self) -> &'static str {
        "Realistic snow particle system"
    }

    fn apply(
        &mut self,
        input: &Frame,
        output: &mut Frame,
        _params: &EffectParams,
    ) -> VfxResult<()> {
        // Copy input to output
        for y in 0..output.height {
            for x in 0..output.width {
                output.set_pixel(x, y, input.get_pixel(x, y).unwrap_or([0, 0, 0, 0]));
            }
        }

        // Spawn new snowflakes
        let dt = 1.0 / 60.0; // Assume 60 FPS
        let spawn_count = (self.system.spawn_rate * dt) as u32;
        for _ in 0..spawn_count {
            self.spawn_snowflake(output.width, output.height);
        }

        // Update particles
        self.system.update(dt, 0.0); // No gravity, controlled fall speed

        // Render snowflakes
        for particle in self.system.particles() {
            if particle.x >= 0.0
                && particle.x < output.width as f32
                && particle.y >= 0.0
                && particle.y < output.height as f32
            {
                let opacity = particle.get_opacity();
                let color_value = (255.0 * opacity) as u8;

                let x = particle.x as u32;
                let y = particle.y as u32;

                // Draw snowflake
                for dy in 0..particle.size as u32 {
                    for dx in 0..particle.size as u32 {
                        let px = x.saturating_add(dx).min(output.width - 1);
                        let py = y.saturating_add(dy).min(output.height - 1);

                        let Some(current) = output.get_pixel(px, py) else {
                            continue;
                        };
                        let blended = Color::from_rgba(current).blend(Color::new(
                            color_value,
                            color_value,
                            color_value,
                            color_value,
                        ));
                        output.set_pixel(px, py, blended.to_rgba());
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snow() {
        let mut snow = Snow::new(SnowStyle::Light).with_wind(0.2);
        let input = Frame::new(100, 100).expect("should succeed in test");
        let mut output = Frame::new(100, 100).expect("should succeed in test");
        let params = EffectParams::new();
        snow.apply(&input, &mut output, &params)
            .expect("should succeed in test");
    }
}
