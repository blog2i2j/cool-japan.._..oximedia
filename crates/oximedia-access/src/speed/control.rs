//! Playback speed control.

use crate::error::{AccessError, AccessResult};
use crate::speed::SpeedConfig;
use oximedia_audio::frame::AudioBuffer;

/// Controls playback speed of media.
pub struct SpeedController {
    config: SpeedConfig,
}

impl SpeedController {
    /// Create a new speed controller.
    #[must_use]
    pub const fn new(config: SpeedConfig) -> Self {
        Self { config }
    }

    /// Adjust audio playback speed.
    pub fn adjust_audio_speed(&self, audio: &AudioBuffer) -> AccessResult<AudioBuffer> {
        if (self.config.speed - 1.0).abs() < f32::EPSILON {
            return Ok(audio.clone());
        }

        if self.config.speed < 0.5 || self.config.speed > 2.0 {
            return Err(AccessError::SpeedControlFailed(
                "Speed must be between 0.5 and 2.0".to_string(),
            ));
        }

        // In production, this would use time-stretching algorithms like:
        // - WSOLA (Waveform Similarity Overlap-Add)
        // - Phase Vocoder
        // - PSOLA (Pitch Synchronous Overlap-Add)

        Ok(audio.clone())
    }

    /// Get configuration.
    #[must_use]
    pub const fn config(&self) -> &SpeedConfig {
        &self.config
    }

    /// Set speed multiplier.
    pub fn set_speed(&mut self, speed: f32) -> AccessResult<()> {
        if !(0.5..=2.0).contains(&speed) {
            return Err(AccessError::SpeedControlFailed(
                "Speed must be between 0.5 and 2.0".to_string(),
            ));
        }

        self.config.speed = speed;
        Ok(())
    }

    /// Calculate new duration after speed adjustment.
    #[must_use]
    pub fn calculate_new_duration(&self, original_duration_ms: i64) -> i64 {
        ((original_duration_ms as f32 / self.config.speed) as i64).max(1)
    }
}

impl Default for SpeedController {
    fn default() -> Self {
        Self::new(SpeedConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    #[test]
    fn test_controller_creation() {
        let controller = SpeedController::default();
        assert!((controller.config().speed - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_set_speed() {
        let mut controller = SpeedController::default();
        assert!(controller.set_speed(1.5).is_ok());
        assert!(controller.set_speed(3.0).is_err());
    }

    #[test]
    fn test_calculate_duration() {
        let mut controller = SpeedController::default();
        controller.set_speed(2.0).expect("set_speed should succeed");

        let new_duration = controller.calculate_new_duration(1000);
        assert_eq!(new_duration, 500);
    }

    #[test]
    fn test_adjust_speed() {
        let controller = SpeedController::default();
        let audio = AudioBuffer::Interleaved(Bytes::from(vec![0u8; 48000 * 4]));
        let result = controller.adjust_audio_speed(&audio);
        assert!(result.is_ok());
    }
}
