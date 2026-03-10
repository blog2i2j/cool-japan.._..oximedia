#![allow(dead_code)]
//! Game controller input mapping and remapping.
//!
//! Provides flexible input mapping for game controllers, allowing users to
//! remap buttons, adjust analog stick dead zones, configure trigger sensitivity,
//! and create per-game profiles for streaming overlays.

use std::collections::HashMap;

/// A unique identifier for a controller button or axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ControllerInput {
    /// Face button A / Cross
    ButtonA,
    /// Face button B / Circle
    ButtonB,
    /// Face button X / Square
    ButtonX,
    /// Face button Y / Triangle
    ButtonY,
    /// Left bumper / L1
    LeftBumper,
    /// Right bumper / R1
    RightBumper,
    /// Left trigger / L2
    LeftTrigger,
    /// Right trigger / R2
    RightTrigger,
    /// Left stick click / L3
    LeftStickClick,
    /// Right stick click / R3
    RightStickClick,
    /// D-Pad up
    DpadUp,
    /// D-Pad down
    DpadDown,
    /// D-Pad left
    DpadLeft,
    /// D-Pad right
    DpadRight,
    /// Start / Options
    Start,
    /// Select / Share
    Select,
    /// Left stick X axis
    LeftStickX,
    /// Left stick Y axis
    LeftStickY,
    /// Right stick X axis
    RightStickX,
    /// Right stick Y axis
    RightStickY,
}

/// The action that a mapped input performs.
#[derive(Debug, Clone, PartialEq)]
pub enum MappedAction {
    /// Map to another controller input.
    Remap(ControllerInput),
    /// Map to a named game action (e.g., "jump", "shoot").
    GameAction(String),
    /// Disable this input (no-op).
    Disabled,
    /// Macro: sequence of actions with delays in milliseconds.
    Macro(Vec<(ControllerInput, u64)>),
}

/// Dead zone configuration for analog sticks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeadZoneConfig {
    /// Inner dead zone radius (0.0 - 1.0). Below this, output is zero.
    pub inner: f32,
    /// Outer dead zone radius (0.0 - 1.0). Above this, output is maximum.
    pub outer: f32,
}

impl Default for DeadZoneConfig {
    fn default() -> Self {
        Self {
            inner: 0.15,
            outer: 0.95,
        }
    }
}

impl DeadZoneConfig {
    /// Create a new dead zone configuration.
    ///
    /// # Errors
    ///
    /// Returns `None` if values are out of range or inner >= outer.
    #[must_use]
    pub fn new(inner: f32, outer: f32) -> Option<Self> {
        if inner < 0.0 || inner >= outer || outer > 1.0 {
            return None;
        }
        Some(Self { inner, outer })
    }

    /// Apply the dead zone to a raw axis value in the range -1.0..=1.0.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn apply(&self, raw: f32) -> f32 {
        let abs_val = raw.abs();
        if abs_val < self.inner {
            return 0.0;
        }
        if abs_val > self.outer {
            return raw.signum();
        }
        let range = self.outer - self.inner;
        if range <= 0.0 {
            return 0.0;
        }
        let normalized = (abs_val - self.inner) / range;
        normalized * raw.signum()
    }
}

/// Trigger sensitivity curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerCurve {
    /// Linear response.
    Linear,
    /// Aggressive (more sensitive at start).
    Aggressive,
    /// Relaxed (less sensitive at start, ramps up).
    Relaxed,
    /// Digital: full on above threshold, off below.
    Digital,
}

/// Configuration for a single trigger.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TriggerConfig {
    /// Sensitivity curve.
    pub curve: TriggerCurve,
    /// Activation threshold for digital mode (0.0 - 1.0).
    pub threshold: f32,
    /// Minimum output value (floor).
    pub min_output: f32,
    /// Maximum output value (ceiling).
    pub max_output: f32,
}

impl Default for TriggerConfig {
    fn default() -> Self {
        Self {
            curve: TriggerCurve::Linear,
            threshold: 0.5,
            min_output: 0.0,
            max_output: 1.0,
        }
    }
}

impl TriggerConfig {
    /// Apply the trigger curve to a raw value in range 0.0..=1.0.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn apply(&self, raw: f32) -> f32 {
        let clamped = raw.clamp(0.0, 1.0);
        let curved = match self.curve {
            TriggerCurve::Linear => clamped,
            TriggerCurve::Aggressive => clamped.sqrt(),
            TriggerCurve::Relaxed => clamped * clamped,
            TriggerCurve::Digital => {
                if clamped >= self.threshold {
                    1.0
                } else {
                    0.0
                }
            }
        };
        self.min_output + curved * (self.max_output - self.min_output)
    }
}

/// A complete controller mapping profile.
#[derive(Debug, Clone)]
pub struct ControllerProfile {
    /// Profile name.
    pub name: String,
    /// Button/axis remappings.
    pub mappings: HashMap<ControllerInput, MappedAction>,
    /// Left stick dead zone.
    pub left_stick_dead_zone: DeadZoneConfig,
    /// Right stick dead zone.
    pub right_stick_dead_zone: DeadZoneConfig,
    /// Left trigger configuration.
    pub left_trigger: TriggerConfig,
    /// Right trigger configuration.
    pub right_trigger: TriggerConfig,
    /// Whether the profile is active.
    pub active: bool,
}

impl ControllerProfile {
    /// Create a new default profile with the given name.
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            mappings: HashMap::new(),
            left_stick_dead_zone: DeadZoneConfig::default(),
            right_stick_dead_zone: DeadZoneConfig::default(),
            left_trigger: TriggerConfig::default(),
            right_trigger: TriggerConfig::default(),
            active: true,
        }
    }

    /// Add a mapping to this profile.
    pub fn add_mapping(&mut self, input: ControllerInput, action: MappedAction) {
        self.mappings.insert(input, action);
    }

    /// Remove a mapping from this profile.
    pub fn remove_mapping(&mut self, input: &ControllerInput) -> Option<MappedAction> {
        self.mappings.remove(input)
    }

    /// Look up the action for a given input.
    #[must_use]
    pub fn get_action(&self, input: &ControllerInput) -> Option<&MappedAction> {
        self.mappings.get(input)
    }

    /// Return the number of active mappings.
    #[must_use]
    pub fn mapping_count(&self) -> usize {
        self.mappings.len()
    }

    /// Check if a specific input has been remapped.
    #[must_use]
    pub fn is_remapped(&self, input: &ControllerInput) -> bool {
        self.mappings.contains_key(input)
    }
}

/// Manager that holds multiple controller profiles and switches between them.
#[derive(Debug)]
pub struct ControllerMappingManager {
    /// All available profiles.
    profiles: Vec<ControllerProfile>,
    /// Index of the currently active profile.
    active_index: Option<usize>,
}

impl ControllerMappingManager {
    /// Create a new empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            profiles: Vec::new(),
            active_index: None,
        }
    }

    /// Add a profile and return its index.
    pub fn add_profile(&mut self, profile: ControllerProfile) -> usize {
        let idx = self.profiles.len();
        if self.active_index.is_none() {
            self.active_index = Some(idx);
        }
        self.profiles.push(profile);
        idx
    }

    /// Switch to a profile by index. Returns `false` if index is out of range.
    pub fn set_active(&mut self, index: usize) -> bool {
        if index < self.profiles.len() {
            self.active_index = Some(index);
            true
        } else {
            false
        }
    }

    /// Get the currently active profile.
    #[must_use]
    pub fn active_profile(&self) -> Option<&ControllerProfile> {
        self.active_index.and_then(|i| self.profiles.get(i))
    }

    /// Get the number of profiles.
    #[must_use]
    pub fn profile_count(&self) -> usize {
        self.profiles.len()
    }

    /// Remove a profile by index. Returns the removed profile or `None`.
    pub fn remove_profile(&mut self, index: usize) -> Option<ControllerProfile> {
        if index >= self.profiles.len() {
            return None;
        }
        let removed = self.profiles.remove(index);
        // Adjust active index
        match self.active_index {
            Some(active) if active == index => {
                self.active_index = if self.profiles.is_empty() {
                    None
                } else {
                    Some(0)
                };
            }
            Some(active) if active > index => {
                self.active_index = Some(active - 1);
            }
            _ => {}
        }
        Some(removed)
    }

    /// Resolve the action for an input using the active profile.
    #[must_use]
    pub fn resolve_input(&self, input: &ControllerInput) -> Option<&MappedAction> {
        self.active_profile()?.get_action(input)
    }

    /// Apply left stick dead zone from active profile.
    #[must_use]
    pub fn apply_left_stick(&self, x: f32, y: f32) -> (f32, f32) {
        match self.active_profile() {
            Some(profile) => {
                let dz = &profile.left_stick_dead_zone;
                (dz.apply(x), dz.apply(y))
            }
            None => (x, y),
        }
    }

    /// Apply right stick dead zone from active profile.
    #[must_use]
    pub fn apply_right_stick(&self, x: f32, y: f32) -> (f32, f32) {
        match self.active_profile() {
            Some(profile) => {
                let dz = &profile.right_stick_dead_zone;
                (dz.apply(x), dz.apply(y))
            }
            None => (x, y),
        }
    }
}

impl Default for ControllerMappingManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dead_zone_default() {
        let dz = DeadZoneConfig::default();
        assert!((dz.inner - 0.15).abs() < f32::EPSILON);
        assert!((dz.outer - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dead_zone_apply_inside() {
        let dz = DeadZoneConfig::new(0.2, 0.9).expect("valid dead zone config");
        let result = dz.apply(0.1);
        assert!(
            (result).abs() < f32::EPSILON,
            "Inside dead zone should be 0"
        );
    }

    #[test]
    fn test_dead_zone_apply_outside() {
        let dz = DeadZoneConfig::new(0.2, 0.9).expect("valid dead zone config");
        let result = dz.apply(0.95);
        assert!(
            (result - 1.0).abs() < f32::EPSILON,
            "Beyond outer should clamp to 1.0"
        );
    }

    #[test]
    fn test_dead_zone_apply_negative() {
        let dz = DeadZoneConfig::new(0.2, 0.9).expect("valid dead zone config");
        let result = dz.apply(-0.95);
        assert!(
            (result + 1.0).abs() < f32::EPSILON,
            "Negative beyond outer should be -1.0"
        );
    }

    #[test]
    fn test_dead_zone_apply_mid_range() {
        let dz = DeadZoneConfig::new(0.2, 0.8).expect("valid dead zone config");
        let result = dz.apply(0.5);
        // (0.5 - 0.2) / (0.8 - 0.2) = 0.3 / 0.6 = 0.5
        assert!((result - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_dead_zone_invalid() {
        assert!(DeadZoneConfig::new(-0.1, 0.9).is_none());
        assert!(DeadZoneConfig::new(0.9, 0.2).is_none());
        assert!(DeadZoneConfig::new(0.5, 1.1).is_none());
        assert!(DeadZoneConfig::new(0.5, 0.5).is_none());
    }

    #[test]
    fn test_trigger_linear() {
        let tc = TriggerConfig::default();
        let result = tc.apply(0.5);
        assert!((result - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_trigger_digital() {
        let tc = TriggerConfig {
            curve: TriggerCurve::Digital,
            threshold: 0.3,
            min_output: 0.0,
            max_output: 1.0,
        };
        assert!((tc.apply(0.1)).abs() < f32::EPSILON, "Below threshold => 0");
        assert!(
            (tc.apply(0.5) - 1.0).abs() < f32::EPSILON,
            "Above threshold => 1"
        );
    }

    #[test]
    fn test_trigger_aggressive() {
        let tc = TriggerConfig {
            curve: TriggerCurve::Aggressive,
            ..TriggerConfig::default()
        };
        // sqrt(0.25) = 0.5
        assert!((tc.apply(0.25) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_trigger_relaxed() {
        let tc = TriggerConfig {
            curve: TriggerCurve::Relaxed,
            ..TriggerConfig::default()
        };
        // 0.5^2 = 0.25
        assert!((tc.apply(0.5) - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_profile_creation_and_mapping() {
        let mut profile = ControllerProfile::new("FPS");
        assert_eq!(profile.name, "FPS");
        assert_eq!(profile.mapping_count(), 0);

        profile.add_mapping(
            ControllerInput::ButtonA,
            MappedAction::GameAction("jump".to_string()),
        );
        assert_eq!(profile.mapping_count(), 1);
        assert!(profile.is_remapped(&ControllerInput::ButtonA));
        assert!(!profile.is_remapped(&ControllerInput::ButtonB));
    }

    #[test]
    fn test_profile_remove_mapping() {
        let mut profile = ControllerProfile::new("Test");
        profile.add_mapping(ControllerInput::ButtonX, MappedAction::Disabled);
        assert_eq!(profile.mapping_count(), 1);
        let removed = profile.remove_mapping(&ControllerInput::ButtonX);
        assert!(removed.is_some());
        assert_eq!(profile.mapping_count(), 0);
    }

    #[test]
    fn test_manager_add_and_switch() {
        let mut mgr = ControllerMappingManager::new();
        assert_eq!(mgr.profile_count(), 0);
        assert!(mgr.active_profile().is_none());

        let p1 = ControllerProfile::new("Default");
        let p2 = ControllerProfile::new("Racing");
        mgr.add_profile(p1);
        mgr.add_profile(p2);

        assert_eq!(mgr.profile_count(), 2);
        assert_eq!(
            mgr.active_profile()
                .expect("active profile should exist")
                .name,
            "Default"
        );

        assert!(mgr.set_active(1));
        assert_eq!(
            mgr.active_profile()
                .expect("active profile should exist")
                .name,
            "Racing"
        );

        assert!(!mgr.set_active(99));
    }

    #[test]
    fn test_manager_remove_profile() {
        let mut mgr = ControllerMappingManager::new();
        mgr.add_profile(ControllerProfile::new("A"));
        mgr.add_profile(ControllerProfile::new("B"));
        mgr.add_profile(ControllerProfile::new("C"));
        mgr.set_active(1);

        let removed = mgr.remove_profile(1);
        assert!(removed.is_some());
        assert_eq!(removed.expect("should succeed").name, "B");
        assert_eq!(mgr.profile_count(), 2);
        // Active should reset to 0 since the active was removed
        assert_eq!(
            mgr.active_profile()
                .expect("active profile should exist")
                .name,
            "A"
        );
    }

    #[test]
    fn test_manager_resolve_input() {
        let mut mgr = ControllerMappingManager::new();
        let mut p = ControllerProfile::new("Test");
        p.add_mapping(
            ControllerInput::ButtonA,
            MappedAction::Remap(ControllerInput::ButtonB),
        );
        mgr.add_profile(p);

        let action = mgr.resolve_input(&ControllerInput::ButtonA);
        assert!(action.is_some());
        assert!(mgr.resolve_input(&ControllerInput::DpadUp).is_none());
    }

    #[test]
    fn test_manager_apply_sticks() {
        let mut mgr = ControllerMappingManager::new();
        let p = ControllerProfile::new("Test");
        mgr.add_profile(p);

        // Inside default dead zone (0.15)
        let (lx, ly) = mgr.apply_left_stick(0.1, 0.05);
        assert!((lx).abs() < f32::EPSILON);
        assert!((ly).abs() < f32::EPSILON);

        let (rx, ry) = mgr.apply_right_stick(1.0, -1.0);
        assert!((rx - 1.0).abs() < f32::EPSILON);
        assert!((ry + 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_trigger_clamp() {
        let tc = TriggerConfig::default();
        // Values outside 0..1 should clamp
        assert!((tc.apply(-0.5)).abs() < f32::EPSILON);
        assert!((tc.apply(1.5) - 1.0).abs() < f32::EPSILON);
    }
}
