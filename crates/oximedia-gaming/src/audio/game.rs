//! Game audio capture.

use crate::GamingResult;

/// Game audio capture.
pub struct GameAudioCapture {
    device: Option<AudioDevice>,
}

/// Audio device information.
#[derive(Debug, Clone)]
pub struct AudioDevice {
    /// Device ID
    pub id: String,
    /// Device name
    pub name: String,
    /// Sample rate
    pub sample_rate: u32,
    /// Channels
    pub channels: u32,
}

impl GameAudioCapture {
    /// Create a new game audio capture.
    #[must_use]
    pub fn new() -> Self {
        Self { device: None }
    }

    /// List available audio devices.
    pub fn list_devices() -> GamingResult<Vec<AudioDevice>> {
        Ok(vec![AudioDevice {
            id: "default".to_string(),
            name: "Default Audio Device".to_string(),
            sample_rate: 48000,
            channels: 2,
        }])
    }

    /// Set audio device.
    pub fn set_device(&mut self, device: AudioDevice) {
        self.device = Some(device);
    }
}

impl Default for GameAudioCapture {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_game_audio_creation() {
        let _capture = GameAudioCapture::new();
    }

    #[test]
    fn test_list_devices() {
        let devices = GameAudioCapture::list_devices().expect("list devices should succeed");
        assert!(!devices.is_empty());
    }
}
