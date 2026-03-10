//! Screen capture implementation.
//!
//! Provides efficient screen capture for monitors, windows, and regions.

use crate::{GamingError, GamingResult};
use std::time::Duration;

/// Screen capture implementation.
pub struct ScreenCapture {
    config: CaptureConfig,
    state: CaptureState,
}

/// Capture configuration.
#[derive(Debug, Clone)]
pub struct CaptureConfig {
    /// Capture region
    pub region: CaptureRegion,
    /// Target framerate
    pub framerate: u32,
    /// Capture cursor
    pub capture_cursor: bool,
}

/// Capture region specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureRegion {
    /// Primary monitor (full screen)
    PrimaryMonitor,
    /// Specific monitor by index
    Monitor(usize),
    /// Specific window by handle/ID
    Window(u64),
    /// Custom region (x, y, width, height)
    Region {
        /// X coordinate of the region
        x: i32,
        /// Y coordinate of the region
        y: i32,
        /// Width of the region
        width: u32,
        /// Height of the region
        height: u32,
    },
}

/// Monitor information.
#[derive(Debug, Clone)]
pub struct MonitorInfo {
    /// Monitor index
    pub index: usize,
    /// Monitor name
    pub name: String,
    /// Resolution (width, height)
    pub resolution: (u32, u32),
    /// Position (x, y)
    pub position: (i32, i32),
    /// Refresh rate in Hz
    pub refresh_rate: u32,
    /// Is primary monitor
    pub is_primary: bool,
}

/// Capture state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CaptureState {
    Idle,
    Capturing,
    Paused,
}

/// Captured frame data.
#[derive(Debug, Clone)]
pub struct CapturedFrame {
    /// Frame data (RGBA or other format)
    pub data: Vec<u8>,
    /// Frame width
    pub width: u32,
    /// Frame height
    pub height: u32,
    /// Timestamp
    pub timestamp: Duration,
    /// Frame sequence number
    pub sequence: u64,
}

impl ScreenCapture {
    /// Create a new screen capture with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns error if screen capture initialization fails.
    pub fn new(config: CaptureConfig) -> GamingResult<Self> {
        // Validate configuration
        if config.framerate == 0 || config.framerate > 240 {
            return Err(GamingError::InvalidConfig(
                "Framerate must be between 1 and 240".to_string(),
            ));
        }

        Ok(Self {
            config,
            state: CaptureState::Idle,
        })
    }

    /// Start capturing frames.
    ///
    /// # Errors
    ///
    /// Returns error if capture fails to start.
    pub fn start(&mut self) -> GamingResult<()> {
        if self.state == CaptureState::Capturing {
            return Err(GamingError::InvalidConfig(
                "Capture already running".to_string(),
            ));
        }

        self.state = CaptureState::Capturing;
        Ok(())
    }

    /// Stop capturing frames.
    pub fn stop(&mut self) {
        self.state = CaptureState::Idle;
    }

    /// Pause capturing frames.
    ///
    /// # Errors
    ///
    /// Returns error if capture is not running.
    pub fn pause(&mut self) -> GamingResult<()> {
        if self.state != CaptureState::Capturing {
            return Err(GamingError::InvalidConfig(
                "Capture not running".to_string(),
            ));
        }

        self.state = CaptureState::Paused;
        Ok(())
    }

    /// Resume capturing frames.
    ///
    /// # Errors
    ///
    /// Returns error if capture is not paused.
    pub fn resume(&mut self) -> GamingResult<()> {
        if self.state != CaptureState::Paused {
            return Err(GamingError::InvalidConfig("Capture not paused".to_string()));
        }

        self.state = CaptureState::Capturing;
        Ok(())
    }

    /// Capture a single frame.
    ///
    /// # Errors
    ///
    /// Returns error if frame capture fails.
    pub fn capture_frame(&self) -> GamingResult<CapturedFrame> {
        if self.state != CaptureState::Capturing {
            return Err(GamingError::CaptureFailed(
                "Capture not running".to_string(),
            ));
        }

        // In a real implementation, this would capture the actual screen
        // For now, return a placeholder frame
        let (width, height) = match self.config.region {
            CaptureRegion::PrimaryMonitor | CaptureRegion::Monitor(_) => (1920, 1080),
            CaptureRegion::Window(_) => (1280, 720),
            CaptureRegion::Region { width, height, .. } => (width, height),
        };

        Ok(CapturedFrame {
            data: vec![0; (width * height * 4) as usize],
            width,
            height,
            timestamp: Duration::from_secs(0),
            sequence: 0,
        })
    }

    /// Get list of available monitors.
    ///
    /// # Errors
    ///
    /// Returns error if monitor enumeration fails.
    pub fn list_monitors() -> GamingResult<Vec<MonitorInfo>> {
        // In a real implementation, this would enumerate actual monitors
        Ok(vec![MonitorInfo {
            index: 0,
            name: "Primary Monitor".to_string(),
            resolution: (1920, 1080),
            position: (0, 0),
            refresh_rate: 60,
            is_primary: true,
        }])
    }

    /// Check if capture is active.
    #[must_use]
    pub fn is_capturing(&self) -> bool {
        self.state == CaptureState::Capturing
    }

    /// Get capture configuration.
    #[must_use]
    pub fn config(&self) -> &CaptureConfig {
        &self.config
    }
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            region: CaptureRegion::PrimaryMonitor,
            framerate: 60,
            capture_cursor: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_screen_capture_creation() {
        let config = CaptureConfig::default();
        let capture = ScreenCapture::new(config).expect("valid screen capture");
        assert!(!capture.is_capturing());
    }

    #[test]
    fn test_invalid_framerate() {
        let mut config = CaptureConfig::default();
        config.framerate = 0;
        let result = ScreenCapture::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_capture_lifecycle() {
        let config = CaptureConfig::default();
        let mut capture = ScreenCapture::new(config).expect("valid screen capture");

        capture.start().expect("start should succeed");
        assert!(capture.is_capturing());

        capture.pause().expect("pause should succeed");
        assert!(!capture.is_capturing());

        capture.resume().expect("resume should succeed");
        assert!(capture.is_capturing());

        capture.stop();
        assert!(!capture.is_capturing());
    }

    #[test]
    fn test_list_monitors() {
        let monitors = ScreenCapture::list_monitors().expect("list monitors should succeed");
        assert!(!monitors.is_empty());
    }

    #[test]
    fn test_capture_frame() {
        let config = CaptureConfig::default();
        let mut capture = ScreenCapture::new(config).expect("valid screen capture");

        capture.start().expect("start should succeed");
        let frame = capture
            .capture_frame()
            .expect("capture frame should succeed");

        assert!(frame.width > 0);
        assert!(frame.height > 0);
        assert!(!frame.data.is_empty());
    }

    #[test]
    fn test_capture_region() {
        let region = CaptureRegion::Region {
            x: 0,
            y: 0,
            width: 1280,
            height: 720,
        };

        let mut config = CaptureConfig::default();
        config.region = region;

        let mut capture = ScreenCapture::new(config).expect("valid screen capture");
        capture.start().expect("start should succeed");

        let frame = capture
            .capture_frame()
            .expect("capture frame should succeed");
        assert_eq!(frame.width, 1280);
        assert_eq!(frame.height, 720);
    }
}
