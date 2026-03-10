//! Replay buffer for instant replay.

use crate::{GamingError, GamingResult};
use std::time::Duration;

/// Replay buffer for storing recent frames.
pub struct ReplayBuffer {
    config: ReplayConfig,
    enabled: bool,
}

/// Replay buffer configuration.
#[derive(Debug, Clone)]
pub struct ReplayConfig {
    /// Buffer duration in seconds
    pub duration: u32,
    /// Video bitrate in kbps
    pub bitrate: u32,
    /// Audio enabled
    pub audio_enabled: bool,
}

impl ReplayBuffer {
    /// Create a new replay buffer.
    pub fn new(config: ReplayConfig) -> GamingResult<Self> {
        if config.duration < 5 || config.duration > 300 {
            return Err(GamingError::ReplayBufferError(
                "Duration must be between 5 and 300 seconds".to_string(),
            ));
        }

        Ok(Self {
            config,
            enabled: false,
        })
    }

    /// Enable replay buffer.
    pub fn enable(&mut self) -> GamingResult<()> {
        self.enabled = true;
        Ok(())
    }

    /// Disable replay buffer.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if replay buffer is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get buffer duration.
    #[must_use]
    pub fn duration(&self) -> Duration {
        Duration::from_secs(u64::from(self.config.duration))
    }
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            duration: 30,
            bitrate: 10000,
            audio_enabled: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replay_buffer_creation() {
        let config = ReplayConfig::default();
        let buffer = ReplayBuffer::new(config).expect("valid replay buffer");
        assert!(!buffer.is_enabled());
    }

    #[test]
    fn test_enable_disable() {
        let mut buffer = ReplayBuffer::new(ReplayConfig::default()).expect("valid replay buffer");
        buffer.enable().expect("enable should succeed");
        assert!(buffer.is_enabled());
        buffer.disable();
        assert!(!buffer.is_enabled());
    }

    #[test]
    fn test_invalid_duration() {
        let config = ReplayConfig {
            duration: 1,
            bitrate: 10000,
            audio_enabled: true,
        };
        let result = ReplayBuffer::new(config);
        assert!(result.is_err());
    }
}
