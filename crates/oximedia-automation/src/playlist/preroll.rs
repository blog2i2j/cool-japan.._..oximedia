//! Pre-roll management for seamless playout.

use std::time::Duration;
use tracing::info;

/// Pre-roll configuration.
#[derive(Debug, Clone)]
pub struct PrerollConfig {
    /// Number of frames to pre-roll
    pub frames: u64,
    /// Frame rate for timing calculation
    pub frame_rate: f64,
}

impl Default for PrerollConfig {
    fn default() -> Self {
        Self {
            frames: 150, // 5 seconds at 30fps
            frame_rate: 30.0,
        }
    }
}

/// Pre-roll manager for playlist items.
pub struct PrerollManager {
    config: PrerollConfig,
}

impl PrerollManager {
    /// Create a new pre-roll manager.
    pub fn new() -> Self {
        Self {
            config: PrerollConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: PrerollConfig) -> Self {
        Self { config }
    }

    /// Calculate pre-roll duration.
    pub fn preroll_duration(&self) -> Duration {
        let seconds = self.config.frames as f64 / self.config.frame_rate;
        Duration::from_secs_f64(seconds)
    }

    /// Get pre-roll frames.
    pub fn preroll_frames(&self) -> u64 {
        self.config.frames
    }

    /// Calculate when to start pre-roll for a scheduled time.
    pub fn calculate_preroll_start(
        &self,
        scheduled_time: std::time::SystemTime,
    ) -> std::time::SystemTime {
        scheduled_time - self.preroll_duration()
    }

    /// Check if it's time to start pre-roll.
    pub fn should_start_preroll(&self, scheduled_time: std::time::SystemTime) -> bool {
        let now = std::time::SystemTime::now();
        let preroll_start = self.calculate_preroll_start(scheduled_time);
        now >= preroll_start
    }

    /// Set frame rate.
    pub fn set_frame_rate(&mut self, frame_rate: f64) {
        info!("Setting pre-roll frame rate to: {}", frame_rate);
        self.config.frame_rate = frame_rate;
    }

    /// Set pre-roll frames.
    pub fn set_preroll_frames(&mut self, frames: u64) {
        info!("Setting pre-roll frames to: {}", frames);
        self.config.frames = frames;
    }
}

impl Default for PrerollManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preroll_manager_creation() {
        let manager = PrerollManager::new();
        assert_eq!(manager.preroll_frames(), 150);
    }

    #[test]
    fn test_preroll_duration() {
        let manager = PrerollManager::new();
        let duration = manager.preroll_duration();
        assert_eq!(duration.as_secs(), 5);
    }

    #[test]
    fn test_set_frame_rate() {
        let mut manager = PrerollManager::new();
        manager.set_frame_rate(60.0);

        let duration = manager.preroll_duration();
        assert_eq!(duration.as_millis(), 2500); // 150 frames at 60fps = 2.5 seconds
    }

    #[test]
    fn test_calculate_preroll_start() {
        let manager = PrerollManager::new();
        let scheduled = std::time::SystemTime::now() + Duration::from_secs(10);
        let preroll_start = manager.calculate_preroll_start(scheduled);

        let diff = scheduled
            .duration_since(preroll_start)
            .expect("duration_since should succeed");
        assert_eq!(diff.as_secs(), 5);
    }
}
