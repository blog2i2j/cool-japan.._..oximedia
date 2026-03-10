#![allow(dead_code)]
//! Spectator and viewer mode management for game streaming.
//!
//! Provides functionality for managing spectator views during live game
//! streaming sessions, including camera presets, point-of-view switching,
//! delayed feeds for anti-cheat, and viewer interaction controls.

use std::collections::HashMap;

/// Unique identifier for a spectator camera view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CameraId(pub u32);

/// Spectator camera preset.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraPreset {
    /// Camera identifier.
    pub id: CameraId,
    /// Human-readable label (e.g., "Player 1 POV", "Overhead").
    pub label: String,
    /// Camera type.
    pub camera_type: CameraType,
    /// Priority for auto-switching (higher = preferred).
    pub priority: u32,
    /// Whether the camera is currently available.
    pub available: bool,
}

/// Type of spectator camera.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraType {
    /// First-person view of a player.
    PlayerPov,
    /// Third-person view following a player.
    ThirdPerson,
    /// Free-floating camera with full control.
    FreeCam,
    /// Fixed overhead map view.
    Overhead,
    /// Auto-director that switches based on action.
    AutoDirector,
    /// Replay camera for key moments.
    Replay,
}

/// Configuration for spectator delay (anti-cheat).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DelayConfig {
    /// Delay in seconds applied to the spectator feed.
    pub delay_seconds: f32,
    /// Whether delay is enabled.
    pub enabled: bool,
}

impl Default for DelayConfig {
    fn default() -> Self {
        Self {
            delay_seconds: 0.0,
            enabled: false,
        }
    }
}

impl DelayConfig {
    /// Create a new delay config with the given seconds.
    #[must_use]
    pub fn with_delay(seconds: f32) -> Self {
        Self {
            delay_seconds: seconds.max(0.0),
            enabled: seconds > 0.0,
        }
    }

    /// Effective delay in milliseconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn delay_ms(&self) -> u64 {
        if self.enabled {
            (self.delay_seconds * 1000.0) as u64
        } else {
            0
        }
    }
}

/// Viewer interaction permission levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ViewerPermission {
    /// View only, no interaction.
    ViewOnly,
    /// Can switch between available cameras.
    CameraSwitch,
    /// Can control free cam and switch cameras.
    FreeCamControl,
    /// Full control including replays and slow-mo.
    FullControl,
}

/// A connected spectator session.
#[derive(Debug, Clone)]
pub struct SpectatorSession {
    /// Session identifier.
    pub session_id: u64,
    /// Display name of the spectator.
    pub display_name: String,
    /// Current camera view.
    pub current_camera: CameraId,
    /// Permission level.
    pub permission: ViewerPermission,
    /// Whether the session is active.
    pub active: bool,
}

impl SpectatorSession {
    /// Create a new spectator session.
    #[must_use]
    pub fn new(session_id: u64, display_name: &str, permission: ViewerPermission) -> Self {
        Self {
            session_id,
            display_name: display_name.to_string(),
            current_camera: CameraId(0),
            permission,
            active: true,
        }
    }

    /// Check if the spectator can switch cameras.
    #[must_use]
    pub fn can_switch_camera(&self) -> bool {
        self.active && self.permission >= ViewerPermission::CameraSwitch
    }

    /// Check if the spectator can use free cam.
    #[must_use]
    pub fn can_free_cam(&self) -> bool {
        self.active && self.permission >= ViewerPermission::FreeCamControl
    }
}

/// Auto-director scoring for camera switching decisions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActionScore {
    /// Score for how interesting this camera view is (0.0 - 1.0).
    pub interest: f32,
    /// How long this camera has been active (seconds).
    pub active_duration: f32,
    /// Suggested minimum hold time before switching (seconds).
    pub min_hold: f32,
}

impl ActionScore {
    /// Create a new action score.
    #[must_use]
    pub fn new(interest: f32, active_duration: f32, min_hold: f32) -> Self {
        Self {
            interest: interest.clamp(0.0, 1.0),
            active_duration: active_duration.max(0.0),
            min_hold: min_hold.max(0.0),
        }
    }

    /// Whether the auto-director should consider switching away from this view.
    #[must_use]
    pub fn should_switch(&self, threshold: f32) -> bool {
        self.interest < threshold && self.active_duration >= self.min_hold
    }
}

/// Main spectator mode manager.
#[derive(Debug)]
pub struct SpectatorManager {
    /// Available camera presets.
    cameras: HashMap<CameraId, CameraPreset>,
    /// Active spectator sessions.
    sessions: Vec<SpectatorSession>,
    /// Delay configuration.
    delay: DelayConfig,
    /// Next camera ID to assign.
    next_camera_id: u32,
    /// Next session ID to assign.
    next_session_id: u64,
    /// Whether auto-director is enabled.
    auto_director_enabled: bool,
    /// Auto-director switch threshold.
    auto_switch_threshold: f32,
    /// Maximum number of spectators allowed.
    max_spectators: usize,
}

impl SpectatorManager {
    /// Create a new spectator manager.
    #[must_use]
    pub fn new(max_spectators: usize) -> Self {
        Self {
            cameras: HashMap::new(),
            sessions: Vec::new(),
            delay: DelayConfig::default(),
            next_camera_id: 1,
            next_session_id: 1,
            auto_director_enabled: false,
            auto_switch_threshold: 0.3,
            max_spectators,
        }
    }

    /// Add a camera preset and return its ID.
    pub fn add_camera(&mut self, label: &str, camera_type: CameraType, priority: u32) -> CameraId {
        let id = CameraId(self.next_camera_id);
        self.next_camera_id += 1;
        let preset = CameraPreset {
            id,
            label: label.to_string(),
            camera_type,
            priority,
            available: true,
        };
        self.cameras.insert(id, preset);
        id
    }

    /// Remove a camera preset by ID.
    pub fn remove_camera(&mut self, id: CameraId) -> Option<CameraPreset> {
        self.cameras.remove(&id)
    }

    /// Get a camera preset by ID.
    #[must_use]
    pub fn get_camera(&self, id: CameraId) -> Option<&CameraPreset> {
        self.cameras.get(&id)
    }

    /// Return the total number of camera presets.
    #[must_use]
    pub fn camera_count(&self) -> usize {
        self.cameras.len()
    }

    /// Admit a new spectator. Returns the session or `None` if full.
    pub fn admit_spectator(
        &mut self,
        name: &str,
        permission: ViewerPermission,
    ) -> Option<&SpectatorSession> {
        if self.active_session_count() >= self.max_spectators {
            return None;
        }
        let sid = self.next_session_id;
        self.next_session_id += 1;
        let session = SpectatorSession::new(sid, name, permission);
        self.sessions.push(session);
        self.sessions.last()
    }

    /// Disconnect a spectator by session ID.
    pub fn disconnect_spectator(&mut self, session_id: u64) -> bool {
        if let Some(session) = self
            .sessions
            .iter_mut()
            .find(|s| s.session_id == session_id)
        {
            session.active = false;
            true
        } else {
            false
        }
    }

    /// Count of active sessions.
    #[must_use]
    pub fn active_session_count(&self) -> usize {
        self.sessions.iter().filter(|s| s.active).count()
    }

    /// Set the feed delay.
    pub fn set_delay(&mut self, config: DelayConfig) {
        self.delay = config;
    }

    /// Get the current delay config.
    #[must_use]
    pub fn delay(&self) -> &DelayConfig {
        &self.delay
    }

    /// Enable or disable the auto-director.
    pub fn set_auto_director(&mut self, enabled: bool) {
        self.auto_director_enabled = enabled;
    }

    /// Whether the auto-director is currently enabled.
    #[must_use]
    pub fn auto_director_enabled(&self) -> bool {
        self.auto_director_enabled
    }

    /// Select the best camera based on action scores.
    #[must_use]
    pub fn select_best_camera(&self, scores: &HashMap<CameraId, ActionScore>) -> Option<CameraId> {
        self.cameras
            .values()
            .filter(|c| c.available)
            .filter_map(|c| scores.get(&c.id).map(|s| (c, s)))
            .max_by(|(a_cam, a_score), (b_cam, b_score)| {
                let a_val = a_score.interest + (a_cam.priority as f32 * 0.01);
                let b_val = b_score.interest + (b_cam.priority as f32 * 0.01);
                a_val
                    .partial_cmp(&b_val)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(c, _)| c.id)
    }

    /// Switch a spectator to a different camera, returning `true` if successful.
    pub fn switch_camera(&mut self, session_id: u64, camera_id: CameraId) -> bool {
        let camera_available = self.cameras.get(&camera_id).is_some_and(|c| c.available);

        if !camera_available {
            return false;
        }

        if let Some(session) = self
            .sessions
            .iter_mut()
            .find(|s| s.session_id == session_id && s.active)
        {
            if session.can_switch_camera() {
                session.current_camera = camera_id;
                return true;
            }
        }
        false
    }
}

impl Default for SpectatorManager {
    fn default() -> Self {
        Self::new(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delay_config_default() {
        let dc = DelayConfig::default();
        assert!(!dc.enabled);
        assert_eq!(dc.delay_ms(), 0);
    }

    #[test]
    fn test_delay_config_with_delay() {
        let dc = DelayConfig::with_delay(30.0);
        assert!(dc.enabled);
        assert_eq!(dc.delay_ms(), 30_000);
    }

    #[test]
    fn test_delay_config_negative() {
        let dc = DelayConfig::with_delay(-5.0);
        assert!(!dc.enabled);
        assert_eq!(dc.delay_ms(), 0);
    }

    #[test]
    fn test_spectator_session_permissions() {
        let s = SpectatorSession::new(1, "viewer1", ViewerPermission::ViewOnly);
        assert!(!s.can_switch_camera());
        assert!(!s.can_free_cam());

        let s2 = SpectatorSession::new(2, "viewer2", ViewerPermission::CameraSwitch);
        assert!(s2.can_switch_camera());
        assert!(!s2.can_free_cam());

        let s3 = SpectatorSession::new(3, "viewer3", ViewerPermission::FullControl);
        assert!(s3.can_switch_camera());
        assert!(s3.can_free_cam());
    }

    #[test]
    fn test_action_score_should_switch() {
        let score = ActionScore::new(0.2, 5.0, 3.0);
        assert!(score.should_switch(0.3));

        let score2 = ActionScore::new(0.5, 5.0, 3.0);
        assert!(!score2.should_switch(0.3));

        // Not held long enough
        let score3 = ActionScore::new(0.1, 1.0, 3.0);
        assert!(!score3.should_switch(0.3));
    }

    #[test]
    fn test_action_score_clamping() {
        let score = ActionScore::new(1.5, -2.0, -1.0);
        assert!((score.interest - 1.0).abs() < f32::EPSILON);
        assert!(score.active_duration.abs() < f32::EPSILON);
        assert!(score.min_hold.abs() < f32::EPSILON);
    }

    #[test]
    fn test_manager_add_remove_cameras() {
        let mut mgr = SpectatorManager::new(10);
        let c1 = mgr.add_camera("POV 1", CameraType::PlayerPov, 5);
        let c2 = mgr.add_camera("Overhead", CameraType::Overhead, 3);
        assert_eq!(mgr.camera_count(), 2);

        assert!(mgr.get_camera(c1).is_some());
        let removed = mgr.remove_camera(c2);
        assert!(removed.is_some());
        assert_eq!(removed.expect("should succeed").label, "Overhead");
        assert_eq!(mgr.camera_count(), 1);
    }

    #[test]
    fn test_manager_admit_spectator() {
        let mut mgr = SpectatorManager::new(2);
        let s1 = mgr.admit_spectator("Alice", ViewerPermission::CameraSwitch);
        assert!(s1.is_some());
        let s2 = mgr.admit_spectator("Bob", ViewerPermission::ViewOnly);
        assert!(s2.is_some());
        // Full
        let s3 = mgr.admit_spectator("Charlie", ViewerPermission::ViewOnly);
        assert!(s3.is_none());
        assert_eq!(mgr.active_session_count(), 2);
    }

    #[test]
    fn test_manager_disconnect_spectator() {
        let mut mgr = SpectatorManager::new(10);
        mgr.admit_spectator("Alice", ViewerPermission::ViewOnly);
        assert_eq!(mgr.active_session_count(), 1);
        assert!(mgr.disconnect_spectator(1));
        assert_eq!(mgr.active_session_count(), 0);
        assert!(!mgr.disconnect_spectator(999));
    }

    #[test]
    fn test_manager_switch_camera() {
        let mut mgr = SpectatorManager::new(10);
        let cam = mgr.add_camera("POV", CameraType::PlayerPov, 5);
        mgr.admit_spectator("Alice", ViewerPermission::CameraSwitch);

        assert!(mgr.switch_camera(1, cam));
        // ViewOnly cannot switch
        mgr.admit_spectator("Bob", ViewerPermission::ViewOnly);
        assert!(!mgr.switch_camera(2, cam));
    }

    #[test]
    fn test_manager_select_best_camera() {
        let mut mgr = SpectatorManager::new(10);
        let c1 = mgr.add_camera("Low", CameraType::PlayerPov, 1);
        let c2 = mgr.add_camera("High", CameraType::Overhead, 2);

        let mut scores = HashMap::new();
        scores.insert(c1, ActionScore::new(0.5, 1.0, 1.0));
        scores.insert(c2, ActionScore::new(0.9, 1.0, 1.0));

        let best = mgr.select_best_camera(&scores);
        assert_eq!(best, Some(c2));
    }

    #[test]
    fn test_manager_auto_director_toggle() {
        let mut mgr = SpectatorManager::new(10);
        assert!(!mgr.auto_director_enabled());
        mgr.set_auto_director(true);
        assert!(mgr.auto_director_enabled());
    }

    #[test]
    fn test_manager_set_delay() {
        let mut mgr = SpectatorManager::new(10);
        mgr.set_delay(DelayConfig::with_delay(15.0));
        assert!(mgr.delay().enabled);
        assert_eq!(mgr.delay().delay_ms(), 15_000);
    }
}
