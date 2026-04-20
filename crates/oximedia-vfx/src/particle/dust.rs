//! Dust particle system.

use super::{Particle, ParticleSystem};
use crate::{Color, EffectParams, Frame, VfxResult, VideoEffect};
use rand::RngExt;
use serde::{Deserialize, Serialize};

/// Dust mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DustMode {
    /// Floating dust.
    Floating,
    /// Rising dust.
    Rising,
    /// Swirling dust.
    Swirling,
}

/// Dust effect.
pub struct Dust {
    mode: DustMode,
    system: ParticleSystem,
    density: f32,
}

impl Dust {
    /// Create a new dust effect.
    #[must_use]
    pub fn new(mode: DustMode) -> Self {
        let mut system = ParticleSystem::new(456);
        system.set_spawn_rate(30.0);

        Self {
            mode,
            system,
            density: 0.5,
        }
    }

    /// Set dust density (0.0 - 1.0).
    #[must_use]
    pub fn with_density(mut self, density: f32) -> Self {
        self.density = density.clamp(0.0, 1.0);
        self.system.set_spawn_rate(30.0 + density * 100.0);
        self
    }

    fn spawn_dust_particle(&mut self, width: u32, height: u32, time: f32) {
        let x = self.system.rng().random_range(0.0..width as f32);
        let y = self.system.rng().random_range(0.0..height as f32);

        let mut particle = Particle::new(x, y);

        match self.mode {
            DustMode::Floating => {
                particle.vx = self.system.rng().random_range(-10.0..10.0);
                particle.vy = self.system.rng().random_range(-5.0..5.0);
            }
            DustMode::Rising => {
                particle.vx = self.system.rng().random_range(-5.0..5.0);
                particle.vy = self.system.rng().random_range(-20.0..-5.0);
            }
            DustMode::Swirling => {
                let angle = time + self.system.rng().random_range(0.0..std::f32::consts::TAU);
                let speed = 20.0;
                particle.vx = angle.cos() * speed;
                particle.vy = angle.sin() * speed;
            }
        }

        particle.size = self.system.rng().random_range(1.0..3.0);
        particle.life = self.system.rng().random_range(3.0..10.0);
        particle.max_life = particle.life;

        self.system.spawn(particle);
    }
}

impl VideoEffect for Dust {
    fn name(&self) -> &'static str {
        "Dust"
    }

    fn description(&self) -> &'static str {
        "Atmospheric dust particle system"
    }

    fn apply(&mut self, input: &Frame, output: &mut Frame, params: &EffectParams) -> VfxResult<()> {
        // Copy input
        for y in 0..output.height {
            for x in 0..output.width {
                output.set_pixel(x, y, input.get_pixel(x, y).unwrap_or([0, 0, 0, 0]));
            }
        }

        // Spawn dust
        let dt = 1.0 / 60.0;
        let spawn_count = (self.system.spawn_rate * dt) as u32;
        for _ in 0..spawn_count {
            self.spawn_dust_particle(output.width, output.height, params.time as f32);
        }

        // Update particles
        self.system.update(dt, 0.0);

        // Render dust
        for particle in self.system.particles() {
            if particle.x >= 0.0
                && particle.x < output.width as f32
                && particle.y >= 0.0
                && particle.y < output.height as f32
            {
                let opacity = particle.get_opacity() * 0.3; // Dust is semi-transparent
                let dust_color = Color::new(
                    (200.0 * opacity) as u8,
                    (180.0 * opacity) as u8,
                    (150.0 * opacity) as u8,
                    (128.0 * opacity) as u8,
                );

                let x = particle.x as u32;
                let y = particle.y as u32;

                let Some(current) = output.get_pixel(x, y) else {
                    continue;
                };
                let blended = Color::from_rgba(current).blend(dust_color);
                output.set_pixel(x, y, blended.to_rgba());
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dust() {
        let mut dust = Dust::new(DustMode::Floating).with_density(0.7);
        let input = Frame::new(100, 100).expect("should succeed in test");
        let mut output = Frame::new(100, 100).expect("should succeed in test");
        let params = EffectParams::new();
        dust.apply(&input, &mut output, &params)
            .expect("should succeed in test");
    }
}
