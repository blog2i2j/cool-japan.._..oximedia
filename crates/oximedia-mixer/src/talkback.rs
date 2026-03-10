#![allow(dead_code)]
//! Talkback and communication system for studio monitoring.
//!
//! Provides a professional talkback system for communication between
//! the control room and various studio zones (live rooms, booths, etc.).
//! Supports latching and momentary modes, multiple destination zones,
//! and automatic monitor dimming during talkback.

use serde::{Deserialize, Serialize};

/// Talkback activation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TalkbackMode {
    /// Momentary: active only while button is held.
    Momentary,
    /// Latching: toggle on/off with button press.
    Latching,
}

/// A destination zone that can receive talkback audio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TalkbackZone {
    /// Human-readable zone name.
    pub name: String,
    /// Whether this zone is enabled to receive talkback.
    pub enabled: bool,
    /// Gain applied to the talkback signal for this zone (0.0..=1.0).
    pub gain: f32,
}

impl TalkbackZone {
    /// Creates a new talkback zone with the given name.
    #[must_use]
    pub fn new(name: String) -> Self {
        Self {
            name,
            enabled: true,
            gain: 1.0,
        }
    }

    /// Returns the effective gain (0.0 if disabled).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn effective_gain(&self) -> f32 {
        if self.enabled {
            self.gain.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// Configuration for the talkback system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TalkbackConfig {
    /// Activation mode (momentary or latching).
    pub mode: TalkbackMode,
    /// Gain applied to the talkback microphone signal (0.0..=1.0).
    pub mic_gain: f32,
    /// Amount to dim monitor outputs during talkback (0.0 = full mute, 1.0 = no dim).
    pub dim_amount: f32,
    /// Whether monitor dimming is enabled during talkback.
    pub dim_enabled: bool,
    /// Maximum number of zones.
    pub max_zones: usize,
}

impl Default for TalkbackConfig {
    fn default() -> Self {
        Self {
            mode: TalkbackMode::Momentary,
            mic_gain: 0.8,
            dim_amount: 0.3,
            dim_enabled: true,
            max_zones: 16,
        }
    }
}

/// Professional talkback system.
///
/// Manages talkback activation, zone routing, and monitor dimming.
#[derive(Debug, Clone)]
pub struct TalkbackSystem {
    config: TalkbackConfig,
    zones: Vec<TalkbackZone>,
    active: bool,
    latched: bool,
}

impl TalkbackSystem {
    /// Creates a new talkback system with the given configuration.
    #[must_use]
    pub fn new(config: TalkbackConfig) -> Self {
        Self {
            config,
            zones: Vec::new(),
            active: false,
            latched: false,
        }
    }

    /// Returns the current configuration.
    #[must_use]
    pub fn config(&self) -> &TalkbackConfig {
        &self.config
    }

    /// Returns whether talkback is currently active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Returns the number of registered zones.
    #[must_use]
    pub fn zone_count(&self) -> usize {
        self.zones.len()
    }

    /// Adds a new talkback zone. Returns the zone index.
    ///
    /// Returns `None` if the maximum number of zones has been reached.
    pub fn add_zone(&mut self, name: String) -> Option<usize> {
        if self.zones.len() >= self.config.max_zones {
            return None;
        }
        let idx = self.zones.len();
        self.zones.push(TalkbackZone::new(name));
        Some(idx)
    }

    /// Returns a reference to a zone by index.
    #[must_use]
    pub fn zone(&self, index: usize) -> Option<&TalkbackZone> {
        self.zones.get(index)
    }

    /// Returns a mutable reference to a zone by index.
    pub fn zone_mut(&mut self, index: usize) -> Option<&mut TalkbackZone> {
        self.zones.get_mut(index)
    }

    /// Enables or disables a specific zone.
    ///
    /// Returns `false` if the zone index is out of bounds.
    pub fn set_zone_enabled(&mut self, index: usize, enabled: bool) -> bool {
        if let Some(zone) = self.zones.get_mut(index) {
            zone.enabled = enabled;
            true
        } else {
            false
        }
    }

    /// Sets the gain for a specific zone.
    ///
    /// Returns `false` if the zone index is out of bounds.
    pub fn set_zone_gain(&mut self, index: usize, gain: f32) -> bool {
        if let Some(zone) = self.zones.get_mut(index) {
            zone.gain = gain.clamp(0.0, 1.0);
            true
        } else {
            false
        }
    }

    /// Activates or deactivates talkback (button press).
    ///
    /// In momentary mode, `pressed=true` activates and `pressed=false` deactivates.
    /// In latching mode, each `pressed=true` toggles the state.
    pub fn button_event(&mut self, pressed: bool) {
        match self.config.mode {
            TalkbackMode::Momentary => {
                self.active = pressed;
            }
            TalkbackMode::Latching => {
                if pressed {
                    self.latched = !self.latched;
                    self.active = self.latched;
                }
            }
        }
    }

    /// Returns the monitor dim multiplier (1.0 = no dim).
    #[must_use]
    pub fn monitor_dim_factor(&self) -> f32 {
        if self.active && self.config.dim_enabled {
            self.config.dim_amount.clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    /// Computes the output gain for a given zone, accounting for
    /// mic gain, zone gain, and whether talkback is active.
    #[must_use]
    pub fn output_gain_for_zone(&self, index: usize) -> f32 {
        if !self.active {
            return 0.0;
        }
        match self.zones.get(index) {
            Some(zone) => zone.effective_gain() * self.config.mic_gain.clamp(0.0, 1.0),
            None => 0.0,
        }
    }

    /// Applies the talkback signal to a buffer for a given zone.
    ///
    /// Multiplies each sample by the output gain for the zone.
    pub fn process_zone(&self, index: usize, buffer: &mut [f32]) {
        let gain = self.output_gain_for_zone(index);
        for sample in buffer.iter_mut() {
            *sample *= gain;
        }
    }

    /// Returns the list of all zones.
    #[must_use]
    pub fn zones(&self) -> &[TalkbackZone] {
        &self.zones
    }

    /// Returns the number of enabled zones.
    #[must_use]
    pub fn enabled_zone_count(&self) -> usize {
        self.zones.iter().filter(|z| z.enabled).count()
    }

    /// Sets the talkback mode.
    pub fn set_mode(&mut self, mode: TalkbackMode) {
        self.config.mode = mode;
        // Reset state on mode change
        self.active = false;
        self.latched = false;
    }

    /// Sets the microphone gain.
    pub fn set_mic_gain(&mut self, gain: f32) {
        self.config.mic_gain = gain.clamp(0.0, 1.0);
    }

    /// Sets the dim amount (0.0 = full mute, 1.0 = no dim).
    pub fn set_dim_amount(&mut self, amount: f32) {
        self.config.dim_amount = amount.clamp(0.0, 1.0);
    }

    /// Enables or disables monitor dimming.
    pub fn set_dim_enabled(&mut self, enabled: bool) {
        self.config.dim_enabled = enabled;
    }
}

impl Default for TalkbackSystem {
    fn default() -> Self {
        Self::new(TalkbackConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = TalkbackConfig::default();
        assert_eq!(cfg.mode, TalkbackMode::Momentary);
        assert!((cfg.mic_gain - 0.8).abs() < f32::EPSILON);
        assert!(cfg.dim_enabled);
    }

    #[test]
    fn test_new_system() {
        let sys = TalkbackSystem::default();
        assert!(!sys.is_active());
        assert_eq!(sys.zone_count(), 0);
    }

    #[test]
    fn test_add_zone() {
        let mut sys = TalkbackSystem::default();
        let idx = sys.add_zone("Studio A".into());
        assert_eq!(idx, Some(0));
        assert_eq!(sys.zone_count(), 1);
        assert_eq!(sys.zone(0).expect("zone should succeed").name, "Studio A");
    }

    #[test]
    fn test_max_zones() {
        let config = TalkbackConfig {
            max_zones: 2,
            ..Default::default()
        };
        let mut sys = TalkbackSystem::new(config);
        assert!(sys.add_zone("A".into()).is_some());
        assert!(sys.add_zone("B".into()).is_some());
        assert!(sys.add_zone("C".into()).is_none());
    }

    #[test]
    fn test_momentary_mode() {
        let mut sys = TalkbackSystem::default();
        assert!(!sys.is_active());
        sys.button_event(true);
        assert!(sys.is_active());
        sys.button_event(false);
        assert!(!sys.is_active());
    }

    #[test]
    fn test_latching_mode() {
        let config = TalkbackConfig {
            mode: TalkbackMode::Latching,
            ..Default::default()
        };
        let mut sys = TalkbackSystem::new(config);
        assert!(!sys.is_active());
        sys.button_event(true); // toggle on
        assert!(sys.is_active());
        sys.button_event(false); // release — no change
        assert!(sys.is_active());
        sys.button_event(true); // toggle off
        assert!(!sys.is_active());
    }

    #[test]
    fn test_monitor_dim_factor_inactive() {
        let sys = TalkbackSystem::default();
        assert!((sys.monitor_dim_factor() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_monitor_dim_factor_active() {
        let mut sys = TalkbackSystem::default();
        sys.button_event(true);
        let expected = sys.config().dim_amount;
        assert!((sys.monitor_dim_factor() - expected).abs() < f32::EPSILON);
    }

    #[test]
    fn test_monitor_dim_disabled() {
        let config = TalkbackConfig {
            dim_enabled: false,
            ..Default::default()
        };
        let mut sys = TalkbackSystem::new(config);
        sys.button_event(true);
        assert!((sys.monitor_dim_factor() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_output_gain_inactive() {
        let mut sys = TalkbackSystem::default();
        sys.add_zone("Booth".into());
        assert!((sys.output_gain_for_zone(0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_output_gain_active() {
        let mut sys = TalkbackSystem::default();
        sys.add_zone("Booth".into());
        sys.button_event(true);
        let expected = 1.0 * 0.8; // zone gain * mic gain
        assert!((sys.output_gain_for_zone(0) - expected).abs() < f32::EPSILON);
    }

    #[test]
    fn test_process_zone() {
        let mut sys = TalkbackSystem::default();
        sys.add_zone("Live Room".into());
        sys.button_event(true);
        let mut buf = vec![1.0_f32; 4];
        sys.process_zone(0, &mut buf);
        let expected = 0.8; // mic_gain * zone_gain(1.0)
        for s in &buf {
            assert!((*s - expected).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_zone_enable_disable() {
        let mut sys = TalkbackSystem::default();
        sys.add_zone("Zone".into());
        sys.button_event(true);
        assert!(sys.output_gain_for_zone(0) > 0.0);
        sys.set_zone_enabled(0, false);
        assert!((sys.output_gain_for_zone(0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_set_zone_gain() {
        let mut sys = TalkbackSystem::default();
        sys.add_zone("Zone".into());
        sys.set_zone_gain(0, 0.5);
        sys.button_event(true);
        let expected = 0.5 * 0.8; // zone gain * mic gain
        assert!((sys.output_gain_for_zone(0) - expected).abs() < f32::EPSILON);
    }

    #[test]
    fn test_enabled_zone_count() {
        let mut sys = TalkbackSystem::default();
        sys.add_zone("A".into());
        sys.add_zone("B".into());
        sys.add_zone("C".into());
        assert_eq!(sys.enabled_zone_count(), 3);
        sys.set_zone_enabled(1, false);
        assert_eq!(sys.enabled_zone_count(), 2);
    }

    #[test]
    fn test_set_mode_resets_state() {
        let mut sys = TalkbackSystem::default();
        sys.button_event(true);
        assert!(sys.is_active());
        sys.set_mode(TalkbackMode::Latching);
        assert!(!sys.is_active());
    }

    #[test]
    fn test_zone_effective_gain_disabled() {
        let mut zone = TalkbackZone::new("Test".into());
        zone.enabled = false;
        assert!((zone.effective_gain()).abs() < f32::EPSILON);
    }

    #[test]
    fn test_zone_effective_gain_clamped() {
        let mut zone = TalkbackZone::new("Test".into());
        zone.gain = 2.0; // over max
        assert!((zone.effective_gain() - 1.0).abs() < f32::EPSILON);
    }
}
