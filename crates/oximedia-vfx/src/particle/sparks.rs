//! Spark particle system.

use super::{Particle, ParticleSystem};
use crate::{Color, EffectParams, Frame, VfxResult, VideoEffect};
use rand::RngExt;
use serde::{Deserialize, Serialize};

/// Spark type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SparkType {
    /// Fire sparks.
    Fire,
    /// Electric sparks.
    Electric,
    /// Magic sparks.
    Magic,
}

/// Sparks effect.
pub struct Sparks {
    spark_type: SparkType,
    system: ParticleSystem,
    source_x: f32,
    source_y: f32,
    color: Color,
}

impl Sparks {
    /// Create a new sparks effect.
    #[must_use]
    pub fn new(spark_type: SparkType) -> Self {
        let mut system = ParticleSystem::new(123);
        system.set_spawn_rate(50.0);

        let color = match spark_type {
            SparkType::Fire => Color::rgb(255, 128, 0),
            SparkType::Electric => Color::rgb(128, 200, 255),
            SparkType::Magic => Color::rgb(200, 100, 255),
        };

        Self {
            spark_type,
            system,
            source_x: 0.5,
            source_y: 0.5,
            color,
        }
    }

    /// Set spark source position (0.0 - 1.0).
    #[must_use]
    pub fn with_source(mut self, x: f32, y: f32) -> Self {
        self.source_x = x.clamp(0.0, 1.0);
        self.source_y = y.clamp(0.0, 1.0);
        self
    }

    /// Set spark color.
    #[must_use]
    pub const fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    fn spawn_spark(&mut self, width: u32, height: u32) {
        let x = self.source_x * width as f32;
        let y = self.source_y * height as f32;

        let angle = self.system.rng().random_range(0.0..std::f32::consts::TAU);
        let speed = self.system.rng().random_range(50.0..150.0);

        let mut particle = Particle::new(x, y);
        particle.vx = angle.cos() * speed;
        particle.vy = angle.sin() * speed - 100.0; // Upward bias
        particle.size = self.system.rng().random_range(1.0..3.0);
        particle.life = self.system.rng().random_range(0.5..2.0);
        particle.max_life = particle.life;

        self.system.spawn(particle);
    }
}

impl VideoEffect for Sparks {
    fn name(&self) -> &'static str {
        "Sparks"
    }

    fn description(&self) -> &'static str {
        "Spark particle system"
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

        // Spawn sparks
        let dt = 1.0 / 60.0;
        let spawn_count = (self.system.spawn_rate * dt) as u32;
        for _ in 0..spawn_count {
            self.spawn_spark(output.width, output.height);
        }

        // Update particles with gravity
        self.system.update(dt, 200.0);

        // Render sparks
        for particle in self.system.particles() {
            if particle.x >= 0.0
                && particle.x < output.width as f32
                && particle.y >= 0.0
                && particle.y < output.height as f32
            {
                let opacity = particle.get_opacity();
                let spark_color = Color::new(
                    (f32::from(self.color.r) * opacity) as u8,
                    (f32::from(self.color.g) * opacity) as u8,
                    (f32::from(self.color.b) * opacity) as u8,
                    (255.0 * opacity) as u8,
                );

                let x = particle.x as u32;
                let y = particle.y as u32;

                let Some(current) = output.get_pixel(x, y) else {
                    continue;
                };
                let blended = Color::from_rgba(current).blend(spark_color);
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
    fn test_sparks() {
        let mut sparks = Sparks::new(SparkType::Fire).with_source(0.5, 0.8);
        let input = Frame::new(100, 100).expect("should succeed in test");
        let mut output = Frame::new(100, 100).expect("should succeed in test");
        let params = EffectParams::new();
        sparks
            .apply(&input, &mut output, &params)
            .expect("should succeed in test");
    }
}
