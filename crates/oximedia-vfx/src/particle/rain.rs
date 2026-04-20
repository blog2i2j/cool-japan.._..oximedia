//! Rain particle system.

use super::{Particle, ParticleSystem};
use crate::{Color, EffectParams, Frame, VfxResult, VideoEffect};
use rand::RngExt;
use serde::{Deserialize, Serialize};

/// Rain intensity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RainIntensity {
    /// Light drizzle.
    Drizzle,
    /// Moderate rain.
    Moderate,
    /// Heavy rain.
    Heavy,
    /// Storm.
    Storm,
}

/// Rain effect.
pub struct Rain {
    intensity: RainIntensity,
    system: ParticleSystem,
    angle: f32,
}

impl Rain {
    /// Create a new rain effect.
    #[must_use]
    pub fn new(intensity: RainIntensity) -> Self {
        let spawn_rate = match intensity {
            RainIntensity::Drizzle => 100.0,
            RainIntensity::Moderate => 300.0,
            RainIntensity::Heavy => 600.0,
            RainIntensity::Storm => 1000.0,
        };

        let mut system = ParticleSystem::new(42);
        system.set_spawn_rate(spawn_rate);

        Self {
            intensity,
            system,
            angle: 0.0,
        }
    }

    /// Set rain angle in degrees.
    #[must_use]
    pub const fn with_angle(mut self, angle: f32) -> Self {
        self.angle = angle;
        self
    }

    fn spawn_raindrop(&mut self, width: u32, _height: u32) {
        let x = self.system.rng().random_range(0.0..width as f32);
        let y = -10.0;

        let angle_rad = self.angle.to_radians();
        let speed = self.system.rng().random_range(400.0..800.0);

        let mut particle = Particle::new(x, y);
        particle.vx = angle_rad.sin() * speed;
        particle.vy = angle_rad.cos() * speed;
        particle.size = self.system.rng().random_range(1.0..2.0);
        particle.life = 5.0;
        particle.max_life = 5.0;

        self.system.spawn(particle);
    }
}

impl VideoEffect for Rain {
    fn name(&self) -> &'static str {
        "Rain"
    }

    fn description(&self) -> &'static str {
        "Rain particle system"
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

        // Spawn raindrops
        let dt = 1.0 / 60.0;
        let spawn_count = (self.system.spawn_rate * dt) as u32;
        for _ in 0..spawn_count {
            self.spawn_raindrop(output.width, output.height);
        }

        // Update particles
        self.system.update(dt, 0.0);

        // Render raindrops as streaks
        for particle in self.system.particles() {
            let streak_length = 10;
            for i in 0..streak_length {
                let t = i as f32 / streak_length as f32;
                let px = particle.x - particle.vx * t * 0.01;
                let py = particle.y - particle.vy * t * 0.01;

                if px >= 0.0 && px < output.width as f32 && py >= 0.0 && py < output.height as f32 {
                    let opacity = particle.get_opacity() * (1.0 - t) * 0.5;
                    let color_value = (200.0 * opacity) as u8;

                    let Some(current) = output.get_pixel(px as u32, py as u32) else {
                        continue;
                    };
                    let blended = Color::from_rgba(current).blend(Color::new(
                        color_value,
                        color_value,
                        color_value,
                        color_value,
                    ));
                    output.set_pixel(px as u32, py as u32, blended.to_rgba());
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
    fn test_rain() {
        let mut rain = Rain::new(RainIntensity::Moderate).with_angle(15.0);
        let input = Frame::new(100, 100).expect("should succeed in test");
        let mut output = Frame::new(100, 100).expect("should succeed in test");
        let params = EffectParams::new();
        rain.apply(&input, &mut output, &params)
            .expect("should succeed in test");
    }
}
