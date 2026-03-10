//! Game-specific capture optimization.
//!
//! Provides optimized capture profiles for different game genres.

use crate::GamingResult;

/// Game capture with genre-specific optimizations.
pub struct GameCapture {
    profile: GameProfile,
    window_handle: Option<u64>,
}

/// Game genre profiles with optimized settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameProfile {
    /// First-person shooter (ultra-low latency)
    Fps,
    /// Multiplayer online battle arena
    Moba,
    /// Fighting game (frame-perfect timing)
    Fighting,
    /// Racing game (high motion handling)
    Racing,
    /// Strategy game (large viewport)
    Strategy,
    /// Role-playing game
    Rpg,
    /// Platformer
    Platformer,
    /// Generic game
    Generic,
}

/// Game detection result.
#[derive(Debug, Clone)]
pub struct GameInfo {
    /// Game name
    pub name: String,
    /// Process ID
    pub pid: u32,
    /// Window handle
    pub window_handle: u64,
    /// Detected profile
    pub profile: GameProfile,
    /// Resolution
    pub resolution: (u32, u32),
    /// Supports hardware acceleration
    pub has_hardware_accel: bool,
}

impl GameCapture {
    /// Create a new game capture with the given profile.
    #[must_use]
    pub fn new(profile: GameProfile) -> Self {
        Self {
            profile,
            window_handle: None,
        }
    }

    /// Auto-detect running games and select appropriate profile.
    ///
    /// # Errors
    ///
    /// Returns error if game detection fails.
    pub fn auto_detect() -> GamingResult<Vec<GameInfo>> {
        // In a real implementation, this would scan running processes
        Ok(Vec::new())
    }

    /// Attach to a specific game window.
    ///
    /// # Errors
    ///
    /// Returns error if attachment fails.
    pub fn attach(&mut self, window_handle: u64) -> GamingResult<()> {
        self.window_handle = Some(window_handle);
        Ok(())
    }

    /// Detach from current game window.
    pub fn detach(&mut self) {
        self.window_handle = None;
    }

    /// Get recommended settings for the current profile.
    #[must_use]
    pub fn recommended_settings(&self) -> CaptureSettings {
        match self.profile {
            GameProfile::Fps => CaptureSettings {
                target_latency_ms: 30,
                max_framerate: 144,
                priority: CapturePriority::Latency,
                motion_prediction: true,
            },
            GameProfile::Moba => CaptureSettings {
                target_latency_ms: 50,
                max_framerate: 60,
                priority: CapturePriority::Balanced,
                motion_prediction: false,
            },
            GameProfile::Fighting => CaptureSettings {
                target_latency_ms: 16,
                max_framerate: 60,
                priority: CapturePriority::Latency,
                motion_prediction: false,
            },
            GameProfile::Racing => CaptureSettings {
                target_latency_ms: 40,
                max_framerate: 120,
                priority: CapturePriority::Quality,
                motion_prediction: true,
            },
            GameProfile::Strategy => CaptureSettings {
                target_latency_ms: 100,
                max_framerate: 60,
                priority: CapturePriority::Quality,
                motion_prediction: false,
            },
            GameProfile::Rpg => CaptureSettings {
                target_latency_ms: 80,
                max_framerate: 60,
                priority: CapturePriority::Quality,
                motion_prediction: false,
            },
            GameProfile::Platformer => CaptureSettings {
                target_latency_ms: 50,
                max_framerate: 60,
                priority: CapturePriority::Balanced,
                motion_prediction: true,
            },
            GameProfile::Generic => CaptureSettings {
                target_latency_ms: 60,
                max_framerate: 60,
                priority: CapturePriority::Balanced,
                motion_prediction: false,
            },
        }
    }

    /// Check if game window is active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.window_handle.is_some()
    }

    /// Get current profile.
    #[must_use]
    pub fn profile(&self) -> GameProfile {
        self.profile
    }
}

/// Capture settings optimized for game profile.
#[derive(Debug, Clone, Copy)]
pub struct CaptureSettings {
    /// Target latency in milliseconds
    pub target_latency_ms: u32,
    /// Maximum framerate
    pub max_framerate: u32,
    /// Capture priority
    pub priority: CapturePriority,
    /// Enable motion prediction
    pub motion_prediction: bool,
}

/// Capture priority mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapturePriority {
    /// Prioritize low latency
    Latency,
    /// Balance latency and quality
    Balanced,
    /// Prioritize high quality
    Quality,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_game_capture_creation() {
        let capture = GameCapture::new(GameProfile::Fps);
        assert_eq!(capture.profile(), GameProfile::Fps);
        assert!(!capture.is_active());
    }

    #[test]
    fn test_fps_profile_settings() {
        let capture = GameCapture::new(GameProfile::Fps);
        let settings = capture.recommended_settings();

        assert_eq!(settings.priority, CapturePriority::Latency);
        assert!(settings.target_latency_ms <= 30);
        assert!(settings.motion_prediction);
    }

    #[test]
    fn test_fighting_game_settings() {
        let capture = GameCapture::new(GameProfile::Fighting);
        let settings = capture.recommended_settings();

        // Fighting games need frame-perfect timing
        assert!(settings.target_latency_ms <= 16);
        assert_eq!(settings.priority, CapturePriority::Latency);
    }

    #[test]
    fn test_strategy_game_settings() {
        let capture = GameCapture::new(GameProfile::Strategy);
        let settings = capture.recommended_settings();

        // Strategy games can tolerate higher latency for better quality
        assert!(settings.target_latency_ms >= 60);
        assert_eq!(settings.priority, CapturePriority::Quality);
    }

    #[test]
    fn test_attach_detach() {
        let mut capture = GameCapture::new(GameProfile::Generic);

        capture.attach(12345).expect("attach should succeed");
        assert!(capture.is_active());

        capture.detach();
        assert!(!capture.is_active());
    }

    #[test]
    fn test_auto_detect() {
        let games = GameCapture::auto_detect().expect("auto detect should succeed");
        // Should return empty list if no games are running
        assert_eq!(games.len(), 0);
    }

    #[test]
    fn test_all_profiles_have_settings() {
        let profiles = [
            GameProfile::Fps,
            GameProfile::Moba,
            GameProfile::Fighting,
            GameProfile::Racing,
            GameProfile::Strategy,
            GameProfile::Rpg,
            GameProfile::Platformer,
            GameProfile::Generic,
        ];

        for profile in profiles {
            let capture = GameCapture::new(profile);
            let settings = capture.recommended_settings();
            assert!(settings.target_latency_ms > 0);
            assert!(settings.max_framerate > 0);
        }
    }
}
