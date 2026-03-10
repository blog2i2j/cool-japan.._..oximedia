#![allow(dead_code)]
//! Hardware control surface integration for audio post-production.
//!
//! Supports MIDI controllers, OSC, and Mackie Control protocol.

use crate::error::{AudioPostError, AudioPostResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// MIDI control change message
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MidiCC {
    /// Channel (0-15)
    pub channel: u8,
    /// Controller number (0-127)
    pub controller: u8,
    /// Value (0-127)
    pub value: u8,
}

impl MidiCC {
    /// Create a new MIDI CC message
    ///
    /// # Errors
    ///
    /// Returns an error if values are out of range
    pub fn new(channel: u8, controller: u8, value: u8) -> AudioPostResult<Self> {
        if channel > 15 {
            return Err(AudioPostError::Generic(format!(
                "MIDI channel must be 0-15, got {channel}"
            )));
        }
        if controller > 127 {
            return Err(AudioPostError::Generic(format!(
                "MIDI controller must be 0-127, got {controller}"
            )));
        }
        if value > 127 {
            return Err(AudioPostError::Generic(format!(
                "MIDI value must be 0-127, got {value}"
            )));
        }

        Ok(Self {
            channel,
            controller,
            value,
        })
    }

    /// Convert 7-bit value to normalized 0.0-1.0 range
    #[must_use]
    pub fn normalized_value(&self) -> f32 {
        f32::from(self.value) / 127.0
    }

    /// Convert 7-bit value to dB range (-inf to +12 dB for fader)
    #[must_use]
    pub fn to_db(&self) -> f32 {
        if self.value == 0 {
            -std::f32::INFINITY
        } else {
            let normalized = self.normalized_value();
            // Map 0-127 to -inf to +12 dB (typical fader range)
            -60.0 + (normalized * 72.0)
        }
    }
}

/// MIDI note message
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MidiNote {
    /// Channel (0-15)
    pub channel: u8,
    /// Note number (0-127)
    pub note: u8,
    /// Velocity (0-127)
    pub velocity: u8,
    /// Note on (true) or off (false)
    pub on: bool,
}

impl MidiNote {
    /// Create a new MIDI note message
    ///
    /// # Errors
    ///
    /// Returns an error if values are out of range
    pub fn new(channel: u8, note: u8, velocity: u8, on: bool) -> AudioPostResult<Self> {
        if channel > 15 {
            return Err(AudioPostError::Generic(format!(
                "MIDI channel must be 0-15, got {channel}"
            )));
        }
        if note > 127 {
            return Err(AudioPostError::Generic(format!(
                "MIDI note must be 0-127, got {note}"
            )));
        }
        if velocity > 127 {
            return Err(AudioPostError::Generic(format!(
                "MIDI velocity must be 0-127, got {velocity}"
            )));
        }

        Ok(Self {
            channel,
            note,
            velocity,
            on,
        })
    }
}

/// Control surface parameter
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ControlParameter {
    /// Fader position
    Fader(usize),
    /// Pan knob
    Pan(usize),
    /// Encoder/knob
    Encoder(usize),
    /// Button press
    Button(usize),
    /// Solo button
    Solo(usize),
    /// Mute button
    Mute(usize),
    /// Record enable button
    RecordEnable(usize),
    /// Select button
    Select(usize),
    /// Master fader
    MasterFader,
    /// Transport play
    TransportPlay,
    /// Transport stop
    TransportStop,
    /// Transport record
    TransportRecord,
    /// Transport rewind
    TransportRewind,
    /// Transport forward
    TransportForward,
}

/// MIDI control mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MidiMapping {
    /// Parameter mappings (MIDI CC -> Parameter)
    mappings: HashMap<(u8, u8), ControlParameter>, // (channel, controller) -> parameter
}

impl MidiMapping {
    /// Create a new MIDI mapping
    #[must_use]
    pub fn new() -> Self {
        Self {
            mappings: HashMap::new(),
        }
    }

    /// Map a MIDI CC to a parameter
    pub fn map_cc(&mut self, channel: u8, controller: u8, parameter: ControlParameter) {
        self.mappings.insert((channel, controller), parameter);
    }

    /// Get parameter for MIDI CC
    #[must_use]
    pub fn get_parameter(&self, channel: u8, controller: u8) -> Option<&ControlParameter> {
        self.mappings.get(&(channel, controller))
    }

    /// Remove mapping
    pub fn unmap(&mut self, channel: u8, controller: u8) -> Option<ControlParameter> {
        self.mappings.remove(&(channel, controller))
    }

    /// Create Mackie Control mapping
    #[must_use]
    pub fn mackie_control() -> Self {
        let mut mapping = Self::new();

        // Map 8 channel faders (controllers 0-7)
        for i in 0..8 {
            mapping.map_cc(0, i, ControlParameter::Fader(i as usize));
            mapping.map_cc(0, 16 + i, ControlParameter::Pan(i as usize));
        }

        // Master fader
        mapping.map_cc(0, 8, ControlParameter::MasterFader);

        mapping
    }

    /// Create generic DAW controller mapping
    #[must_use]
    pub fn generic_daw() -> Self {
        let mut mapping = Self::new();

        // Common DAW controller mappings
        for i in 0..8 {
            mapping.map_cc(0, i, ControlParameter::Fader(i as usize));
            mapping.map_cc(0, 16 + i, ControlParameter::Encoder(i as usize));
            mapping.map_cc(0, 32 + i, ControlParameter::Solo(i as usize));
            mapping.map_cc(0, 48 + i, ControlParameter::Mute(i as usize));
            mapping.map_cc(0, 64 + i, ControlParameter::RecordEnable(i as usize));
        }

        mapping
    }
}

impl Default for MidiMapping {
    fn default() -> Self {
        Self::new()
    }
}

/// OSC (Open Sound Control) message
#[derive(Debug, Clone, PartialEq)]
pub struct OscMessage {
    /// OSC address pattern
    pub address: String,
    /// OSC arguments
    pub args: Vec<OscArgument>,
}

impl OscMessage {
    /// Create a new OSC message
    #[must_use]
    pub fn new(address: &str) -> Self {
        Self {
            address: address.to_string(),
            args: Vec::new(),
        }
    }

    /// Add a float argument
    pub fn add_float(&mut self, value: f32) {
        self.args.push(OscArgument::Float(value));
    }

    /// Add an integer argument
    pub fn add_int(&mut self, value: i32) {
        self.args.push(OscArgument::Int(value));
    }

    /// Add a string argument
    pub fn add_string(&mut self, value: &str) {
        self.args.push(OscArgument::String(value.to_string()));
    }
}

/// OSC argument type
#[derive(Debug, Clone, PartialEq)]
pub enum OscArgument {
    /// Float32 value
    Float(f32),
    /// Int32 value
    Int(i32),
    /// String value
    String(String),
    /// Boolean value
    Bool(bool),
}

/// Control surface implementation
#[derive(Debug)]
pub struct ControlSurface {
    /// Surface name
    pub name: String,
    /// MIDI mapping
    pub midi_mapping: MidiMapping,
    /// Number of faders
    pub num_faders: usize,
    /// Number of encoders
    pub num_encoders: usize,
    /// Touch-sensitive faders
    pub touch_sensitive: bool,
    /// Motorized faders
    pub motorized: bool,
}

impl ControlSurface {
    /// Create a new control surface
    #[must_use]
    pub fn new(name: &str, num_faders: usize, num_encoders: usize) -> Self {
        Self {
            name: name.to_string(),
            midi_mapping: MidiMapping::new(),
            num_faders,
            num_encoders,
            touch_sensitive: false,
            motorized: false,
        }
    }

    /// Create a Mackie Control Universal surface
    #[must_use]
    pub fn mackie_control_universal() -> Self {
        Self {
            name: "Mackie Control Universal".to_string(),
            midi_mapping: MidiMapping::mackie_control(),
            num_faders: 9, // 8 channels + 1 master
            num_encoders: 8,
            touch_sensitive: true,
            motorized: true,
        }
    }

    /// Process MIDI CC message
    #[must_use]
    pub fn process_midi_cc(&self, cc: &MidiCC) -> Option<(ControlParameter, f32)> {
        self.midi_mapping
            .get_parameter(cc.channel, cc.controller)
            .map(|param| (*param, cc.normalized_value()))
    }
}

/// Mackie Control Universal protocol implementation
#[derive(Debug)]
pub struct MackieControl {
    /// Surface instance
    surface: ControlSurface,
    /// Fader positions (0.0-1.0)
    fader_positions: Vec<f32>,
    /// Encoder positions
    encoder_positions: Vec<i32>,
    /// Button states
    button_states: HashMap<usize, bool>,
}

impl MackieControl {
    /// Create a new Mackie Control instance
    #[must_use]
    pub fn new() -> Self {
        let surface = ControlSurface::mackie_control_universal();
        let num_faders = surface.num_faders;
        let num_encoders = surface.num_encoders;

        Self {
            surface,
            fader_positions: vec![0.75; num_faders], // Unity gain
            encoder_positions: vec![0; num_encoders],
            button_states: HashMap::new(),
        }
    }

    /// Get fader position
    #[must_use]
    pub fn get_fader(&self, index: usize) -> Option<f32> {
        self.fader_positions.get(index).copied()
    }

    /// Set fader position
    ///
    /// # Errors
    ///
    /// Returns an error if index is out of range
    pub fn set_fader(&mut self, index: usize, value: f32) -> AudioPostResult<()> {
        if index >= self.fader_positions.len() {
            return Err(AudioPostError::Generic(format!(
                "Fader index {index} out of range"
            )));
        }
        self.fader_positions[index] = value.clamp(0.0, 1.0);
        Ok(())
    }

    /// Get encoder position
    #[must_use]
    pub fn get_encoder(&self, index: usize) -> Option<i32> {
        self.encoder_positions.get(index).copied()
    }

    /// Increment encoder
    ///
    /// # Errors
    ///
    /// Returns an error if index is out of range
    pub fn increment_encoder(&mut self, index: usize, delta: i32) -> AudioPostResult<()> {
        if index >= self.encoder_positions.len() {
            return Err(AudioPostError::Generic(format!(
                "Encoder index {index} out of range"
            )));
        }
        self.encoder_positions[index] += delta;
        Ok(())
    }
}

impl Default for MackieControl {
    fn default() -> Self {
        Self::new()
    }
}

/// Hardware controller manager
#[derive(Debug)]
pub struct HardwareManager {
    /// Active control surfaces
    surfaces: Vec<ControlSurface>,
    /// OSC enabled
    pub osc_enabled: bool,
    /// OSC port
    pub osc_port: u16,
}

impl HardwareManager {
    /// Create a new hardware manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            surfaces: Vec::new(),
            osc_enabled: false,
            osc_port: 8000,
        }
    }

    /// Add a control surface
    pub fn add_surface(&mut self, surface: ControlSurface) {
        self.surfaces.push(surface);
    }

    /// Enable OSC
    pub fn enable_osc(&mut self, port: u16) {
        self.osc_enabled = true;
        self.osc_port = port;
    }

    /// Disable OSC
    pub fn disable_osc(&mut self) {
        self.osc_enabled = false;
    }

    /// Get number of active surfaces
    #[must_use]
    pub fn surface_count(&self) -> usize {
        self.surfaces.len()
    }
}

impl Default for HardwareManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_midi_cc_creation() {
        let cc = MidiCC::new(0, 1, 64).expect("failed to create");
        assert_eq!(cc.channel, 0);
        assert_eq!(cc.controller, 1);
        assert_eq!(cc.value, 64);
    }

    #[test]
    fn test_midi_cc_normalized() {
        let cc = MidiCC::new(0, 1, 127).expect("failed to create");
        assert_eq!(cc.normalized_value(), 1.0);
    }

    #[test]
    fn test_midi_cc_to_db() {
        let cc = MidiCC::new(0, 1, 0).expect("failed to create");
        assert!(cc.to_db().is_infinite() && cc.to_db().is_sign_negative());

        let cc = MidiCC::new(0, 1, 127).expect("failed to create");
        assert!((cc.to_db() - 12.0).abs() < 0.1);
    }

    #[test]
    fn test_invalid_midi_cc() {
        assert!(MidiCC::new(16, 0, 0).is_err());
        assert!(MidiCC::new(0, 128, 0).is_err());
        assert!(MidiCC::new(0, 0, 128).is_err());
    }

    #[test]
    fn test_midi_note_creation() {
        let note = MidiNote::new(0, 60, 100, true).expect("failed to create");
        assert_eq!(note.note, 60);
        assert_eq!(note.velocity, 100);
        assert!(note.on);
    }

    #[test]
    fn test_invalid_midi_note() {
        assert!(MidiNote::new(16, 60, 100, true).is_err());
        assert!(MidiNote::new(0, 128, 100, true).is_err());
        assert!(MidiNote::new(0, 60, 128, true).is_err());
    }

    #[test]
    fn test_midi_mapping() {
        let mut mapping = MidiMapping::new();
        mapping.map_cc(0, 1, ControlParameter::Fader(0));
        assert!(mapping.get_parameter(0, 1).is_some());
    }

    #[test]
    fn test_midi_mapping_unmap() {
        let mut mapping = MidiMapping::new();
        mapping.map_cc(0, 1, ControlParameter::Fader(0));
        let removed = mapping.unmap(0, 1);
        assert!(removed.is_some());
        assert!(mapping.get_parameter(0, 1).is_none());
    }

    #[test]
    fn test_mackie_control_mapping() {
        let mapping = MidiMapping::mackie_control();
        assert!(mapping.get_parameter(0, 0).is_some());
    }

    #[test]
    fn test_generic_daw_mapping() {
        let mapping = MidiMapping::generic_daw();
        assert!(mapping.get_parameter(0, 0).is_some());
        assert!(mapping.get_parameter(0, 32).is_some());
    }

    #[test]
    fn test_osc_message() {
        let mut msg = OscMessage::new("/volume/1");
        msg.add_float(0.75);
        assert_eq!(msg.address, "/volume/1");
        assert_eq!(msg.args.len(), 1);
    }

    #[test]
    fn test_osc_arguments() {
        let mut msg = OscMessage::new("/test");
        msg.add_float(1.0);
        msg.add_int(42);
        msg.add_string("test");
        assert_eq!(msg.args.len(), 3);
    }

    #[test]
    fn test_control_surface_creation() {
        let surface = ControlSurface::new("Test Surface", 8, 8);
        assert_eq!(surface.name, "Test Surface");
        assert_eq!(surface.num_faders, 8);
    }

    #[test]
    fn test_mackie_control_surface() {
        let surface = ControlSurface::mackie_control_universal();
        assert_eq!(surface.num_faders, 9);
        assert!(surface.motorized);
    }

    #[test]
    fn test_control_surface_process_midi() {
        let surface = ControlSurface::mackie_control_universal();
        let cc = MidiCC::new(0, 0, 64).expect("failed to create");
        let result = surface.process_midi_cc(&cc);
        assert!(result.is_some());
    }

    #[test]
    fn test_mackie_control_creation() {
        let mackie = MackieControl::new();
        assert_eq!(mackie.fader_positions.len(), 9);
    }

    #[test]
    fn test_mackie_control_fader() {
        let mut mackie = MackieControl::new();
        assert!(mackie.set_fader(0, 0.5).is_ok());
        assert_eq!(mackie.get_fader(0), Some(0.5));
    }

    #[test]
    fn test_mackie_control_encoder() {
        let mut mackie = MackieControl::new();
        assert!(mackie.increment_encoder(0, 1).is_ok());
        assert_eq!(mackie.get_encoder(0), Some(1));
    }

    #[test]
    fn test_hardware_manager() {
        let mut manager = HardwareManager::new();
        let surface = ControlSurface::new("Test", 8, 8);
        manager.add_surface(surface);
        assert_eq!(manager.surface_count(), 1);
    }

    #[test]
    fn test_hardware_manager_osc() {
        let mut manager = HardwareManager::new();
        manager.enable_osc(9000);
        assert!(manager.osc_enabled);
        assert_eq!(manager.osc_port, 9000);
    }

    #[test]
    fn test_hardware_manager_disable_osc() {
        let mut manager = HardwareManager::new();
        manager.enable_osc(9000);
        manager.disable_osc();
        assert!(!manager.osc_enabled);
    }
}
