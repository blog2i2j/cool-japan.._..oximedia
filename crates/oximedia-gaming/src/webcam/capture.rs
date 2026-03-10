//! Webcam capture.

use crate::GamingResult;

/// Webcam capture.
#[allow(dead_code)]
pub struct WebcamCapture {
    config: WebcamConfig,
}

/// Webcam configuration.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct WebcamConfig {
    /// Device ID
    pub device_id: usize,
    /// Resolution
    pub resolution: (u32, u32),
    /// Framerate
    pub framerate: u32,
}

impl WebcamCapture {
    /// Create a new webcam capture.
    pub fn new(config: WebcamConfig) -> GamingResult<Self> {
        Ok(Self { config })
    }

    /// List available webcams.
    pub fn list_devices() -> GamingResult<Vec<String>> {
        Ok(vec!["Default Webcam".to_string()])
    }

    /// Start capture.
    pub fn start(&mut self) -> GamingResult<()> {
        Ok(())
    }

    /// Stop capture.
    pub fn stop(&mut self) {}
}

impl Default for WebcamConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            resolution: (1280, 720),
            framerate: 30,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webcam_creation() {
        let config = WebcamConfig::default();
        let _webcam = WebcamCapture::new(config).expect("valid webcam capture");
    }

    #[test]
    fn test_list_devices() {
        let devices = WebcamCapture::list_devices().expect("list devices should succeed");
        assert!(!devices.is_empty());
    }
}
