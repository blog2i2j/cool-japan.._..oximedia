#![allow(dead_code)]
//! Tally light control system for multi-camera production.
//!
//! Manages on-air (program) and preview tally indicators for each camera,
//! supporting standard broadcast tally protocols and custom colour schemes.

use std::collections::HashMap;

/// Tally colour indicating the state of a camera.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TallyColor {
    /// Camera is off-air — tally light off.
    Off,
    /// Camera is on program (red).
    Red,
    /// Camera is on preview/next (green).
    Green,
    /// Camera is in ISO record mode (amber/yellow).
    Amber,
    /// Custom colour for vendor-specific extensions.
    Custom(u8),
}

impl TallyColor {
    /// Whether the tally is illuminated (anything other than Off).
    #[must_use]
    pub fn is_lit(self) -> bool {
        self != Self::Off
    }
}

/// Tally state for a single camera.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TallyState {
    /// Front tally (visible to talent / on-camera).
    pub front: TallyColor,
    /// Rear tally (visible to camera operator).
    pub rear: TallyColor,
}

impl TallyState {
    /// Create a new tally state with both lights off.
    #[must_use]
    pub fn off() -> Self {
        Self {
            front: TallyColor::Off,
            rear: TallyColor::Off,
        }
    }

    /// Create a program (on-air) tally state.
    #[must_use]
    pub fn program() -> Self {
        Self {
            front: TallyColor::Red,
            rear: TallyColor::Red,
        }
    }

    /// Create a preview tally state.
    #[must_use]
    pub fn preview() -> Self {
        Self {
            front: TallyColor::Green,
            rear: TallyColor::Green,
        }
    }

    /// Create an ISO record tally state.
    #[must_use]
    pub fn iso_record() -> Self {
        Self {
            front: TallyColor::Amber,
            rear: TallyColor::Amber,
        }
    }

    /// Whether either tally light is lit.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.front.is_lit() || self.rear.is_lit()
    }
}

impl Default for TallyState {
    fn default() -> Self {
        Self::off()
    }
}

/// Event emitted when a tally state changes.
#[derive(Debug, Clone)]
pub struct TallyChangeEvent {
    /// Camera angle whose tally changed.
    pub angle_id: usize,
    /// Previous tally state.
    pub previous: TallyState,
    /// New tally state.
    pub current: TallyState,
    /// Monotonic timestamp in microseconds.
    pub timestamp_us: u64,
}

/// Tally transport protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TallyProtocol {
    /// TSL UMD v3.1.
    TslUmd31,
    /// TSL UMD v5.0.
    TslUmd50,
    /// Ember+ (Lawo).
    EmberPlus,
    /// GPIO pin-based.
    Gpio,
    /// Internal only (no external transport).
    Internal,
}

/// Configuration for the tally system.
#[derive(Debug, Clone)]
pub struct TallyConfig {
    /// Transport protocol to use.
    pub protocol: TallyProtocol,
    /// Whether to follow the program bus automatically.
    pub auto_follow_program: bool,
    /// Whether to follow the preview bus automatically.
    pub auto_follow_preview: bool,
    /// Flash interval in milliseconds (0 = no flash).
    pub flash_interval_ms: u32,
}

impl Default for TallyConfig {
    fn default() -> Self {
        Self {
            protocol: TallyProtocol::Internal,
            auto_follow_program: true,
            auto_follow_preview: true,
            flash_interval_ms: 0,
        }
    }
}

/// Central tally controller that manages tally state for all registered cameras.
#[derive(Debug)]
pub struct TallyController {
    /// Configuration.
    config: TallyConfig,
    /// Per-angle tally state.
    states: HashMap<usize, TallyState>,
    /// History of tally changes (bounded).
    history: Vec<TallyChangeEvent>,
    /// Maximum history entries to retain.
    max_history: usize,
    /// Current monotonic timestamp counter.
    clock_us: u64,
}

impl TallyController {
    /// Create a new tally controller with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(TallyConfig::default())
    }

    /// Create a new tally controller with the given configuration.
    #[must_use]
    pub fn with_config(config: TallyConfig) -> Self {
        Self {
            config,
            states: HashMap::new(),
            history: Vec::new(),
            max_history: 1024,
            clock_us: 0,
        }
    }

    /// Register a camera angle.
    pub fn register_angle(&mut self, angle_id: usize) {
        self.states.entry(angle_id).or_insert_with(TallyState::off);
    }

    /// Set the tally state for a specific angle.
    pub fn set_state(&mut self, angle_id: usize, new_state: TallyState) {
        let previous = self
            .states
            .get(&angle_id)
            .copied()
            .unwrap_or_else(TallyState::off);
        self.states.insert(angle_id, new_state);
        if previous != new_state {
            self.record_change(angle_id, previous, new_state);
        }
    }

    /// Get the tally state for a specific angle.
    pub fn get_state(&self, angle_id: usize) -> TallyState {
        self.states
            .get(&angle_id)
            .copied()
            .unwrap_or_else(TallyState::off)
    }

    /// Set a camera to the program (on-air) tally.
    pub fn set_program(&mut self, angle_id: usize) {
        self.set_state(angle_id, TallyState::program());
    }

    /// Set a camera to the preview tally.
    pub fn set_preview(&mut self, angle_id: usize) {
        self.set_state(angle_id, TallyState::preview());
    }

    /// Turn off the tally for a camera.
    pub fn set_off(&mut self, angle_id: usize) {
        self.set_state(angle_id, TallyState::off());
    }

    /// Set program and preview tallies based on switcher bus state.
    /// All other registered cameras are turned off.
    pub fn update_from_buses(&mut self, program_angle: usize, preview_angle: usize) {
        let angle_ids: Vec<usize> = self.states.keys().copied().collect();
        for &aid in &angle_ids {
            if aid == program_angle {
                self.set_state(aid, TallyState::program());
            } else if aid == preview_angle {
                self.set_state(aid, TallyState::preview());
            } else {
                self.set_state(aid, TallyState::off());
            }
        }
    }

    /// Number of registered angles.
    #[must_use]
    pub fn angle_count(&self) -> usize {
        self.states.len()
    }

    /// Return all angles currently on program (red tally).
    #[must_use]
    pub fn program_angles(&self) -> Vec<usize> {
        self.states
            .iter()
            .filter(|(_, s)| s.front == TallyColor::Red)
            .map(|(&id, _)| id)
            .collect()
    }

    /// Return all angles currently on preview (green tally).
    #[must_use]
    pub fn preview_angles(&self) -> Vec<usize> {
        self.states
            .iter()
            .filter(|(_, s)| s.front == TallyColor::Green)
            .map(|(&id, _)| id)
            .collect()
    }

    /// Get the change history.
    #[must_use]
    pub fn history(&self) -> &[TallyChangeEvent] {
        &self.history
    }

    /// Get the active configuration.
    #[must_use]
    pub fn config(&self) -> &TallyConfig {
        &self.config
    }

    /// Advance the internal clock (for timestamping events).
    pub fn advance_clock(&mut self, delta_us: u64) {
        self.clock_us += delta_us;
    }

    /// Record a change event.
    fn record_change(&mut self, angle_id: usize, previous: TallyState, current: TallyState) {
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(TallyChangeEvent {
            angle_id,
            previous,
            current,
            timestamp_us: self.clock_us,
        });
    }
}

impl Default for TallyController {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tally_color_is_lit() {
        assert!(!TallyColor::Off.is_lit());
        assert!(TallyColor::Red.is_lit());
        assert!(TallyColor::Green.is_lit());
        assert!(TallyColor::Amber.is_lit());
        assert!(TallyColor::Custom(1).is_lit());
    }

    #[test]
    fn test_tally_state_off() {
        let s = TallyState::off();
        assert!(!s.is_active());
        assert_eq!(s.front, TallyColor::Off);
        assert_eq!(s.rear, TallyColor::Off);
    }

    #[test]
    fn test_tally_state_program() {
        let s = TallyState::program();
        assert!(s.is_active());
        assert_eq!(s.front, TallyColor::Red);
    }

    #[test]
    fn test_tally_state_preview() {
        let s = TallyState::preview();
        assert!(s.is_active());
        assert_eq!(s.front, TallyColor::Green);
    }

    #[test]
    fn test_tally_state_iso_record() {
        let s = TallyState::iso_record();
        assert_eq!(s.front, TallyColor::Amber);
        assert_eq!(s.rear, TallyColor::Amber);
    }

    #[test]
    fn test_controller_register_and_get() {
        let mut ctrl = TallyController::new();
        ctrl.register_angle(0);
        ctrl.register_angle(1);
        assert_eq!(ctrl.angle_count(), 2);
        assert_eq!(ctrl.get_state(0), TallyState::off());
    }

    #[test]
    fn test_controller_set_program() {
        let mut ctrl = TallyController::new();
        ctrl.register_angle(0);
        ctrl.set_program(0);
        assert_eq!(ctrl.get_state(0), TallyState::program());
        assert_eq!(ctrl.program_angles(), vec![0]);
    }

    #[test]
    fn test_controller_set_preview() {
        let mut ctrl = TallyController::new();
        ctrl.register_angle(0);
        ctrl.set_preview(0);
        assert_eq!(ctrl.get_state(0), TallyState::preview());
        assert_eq!(ctrl.preview_angles(), vec![0]);
    }

    #[test]
    fn test_controller_update_from_buses() {
        let mut ctrl = TallyController::new();
        ctrl.register_angle(0);
        ctrl.register_angle(1);
        ctrl.register_angle(2);
        ctrl.update_from_buses(1, 2);

        assert_eq!(ctrl.get_state(0), TallyState::off());
        assert_eq!(ctrl.get_state(1), TallyState::program());
        assert_eq!(ctrl.get_state(2), TallyState::preview());
    }

    #[test]
    fn test_controller_history() {
        let mut ctrl = TallyController::new();
        ctrl.register_angle(0);
        ctrl.set_program(0);
        ctrl.set_off(0);
        assert_eq!(ctrl.history().len(), 2);
    }

    #[test]
    fn test_controller_clock() {
        let mut ctrl = TallyController::new();
        ctrl.register_angle(0);
        ctrl.advance_clock(5000);
        ctrl.set_program(0);
        assert_eq!(
            ctrl.history()
                .last()
                .expect("multicam test operation should succeed")
                .timestamp_us,
            5000
        );
    }

    #[test]
    fn test_default_config() {
        let cfg = TallyConfig::default();
        assert_eq!(cfg.protocol, TallyProtocol::Internal);
        assert!(cfg.auto_follow_program);
        assert!(cfg.auto_follow_preview);
        assert_eq!(cfg.flash_interval_ms, 0);
    }

    #[test]
    fn test_controller_default_trait() {
        let ctrl = TallyController::default();
        assert_eq!(ctrl.angle_count(), 0);
    }
}
