//! Per-channel strip processing chain for broadcast routing consoles.
//!
//! A [`ChannelStrip`] models the signal path found on a single fader channel
//! of a professional routing console or broadcast mixing desk:
//!
//! 1. **Input gain** — trim / pad applied immediately after the input.
//! 2. **Gate** — noise gate with threshold, attack, release, and hold.
//! 3. **Compressor** — downward compressor with threshold, ratio, attack,
//!    release, and make-up gain.
//! 4. **High-pass filter** — first-order Butterworth HPF for rumble removal.
//! 5. **Low-pass filter** — first-order Butterworth LPF for brightness control.
//! 6. **Fader** — post-dynamics master level.
//! 7. **Mute / Solo / Safe** flags.
//! 8. **Aux sends** — up to 8 pre/post-fader sends to auxiliary buses.
//!
//! The [`StripBus`] owns a collection of named [`ChannelStrip`]s and provides
//! bulk operations (mute-all, solo-isolate, etc.).
//!
//! ## Example
//!
//! ```rust
//! use oximedia_routing::channel_strip::{ChannelStrip, StripBus};
//!
//! let mut strip = ChannelStrip::new("Mic 1");
//! strip.set_input_gain_db(-6.0);
//! strip.set_fader_db(-3.0);
//! strip.mute();
//!
//! assert!(strip.is_muted());
//! assert!((strip.input_gain_db() - -6.0).abs() < 1e-6);
//!
//! let mut bus = StripBus::new();
//! bus.add_strip(strip);
//! assert_eq!(bus.strip_count(), 1);
//! ```

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from channel strip operations.
#[derive(Debug, Error, PartialEq)]
pub enum StripError {
    /// A parameter value is outside the allowed range.
    #[error("parameter '{name}' value {value} out of range [{min}, {max}]")]
    OutOfRange {
        /// Parameter name.
        name: String,
        /// Supplied value.
        value: f32,
        /// Minimum allowed.
        min: f32,
        /// Maximum allowed.
        max: f32,
    },
    /// The referenced aux send index is out of range.
    #[error("aux send index {0} is out of range (max {1})")]
    AuxSendOutOfRange(usize, usize),
    /// A strip with the given name already exists in the bus.
    #[error("duplicate strip name: {0}")]
    DuplicateStrip(String),
    /// A strip with the given name does not exist in the bus.
    #[error("strip not found: {0}")]
    StripNotFound(String),
}

// ---------------------------------------------------------------------------
// Gate parameters
// ---------------------------------------------------------------------------

/// Noise gate parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateParams {
    /// Gate open/close state.
    pub enabled: bool,
    /// Threshold in dBFS. Signals below this level close the gate.
    pub threshold_db: f32,
    /// Attack time in milliseconds (5–500 ms).
    pub attack_ms: f32,
    /// Release time in milliseconds (10–4000 ms).
    pub release_ms: f32,
    /// Hold time in milliseconds (0–2000 ms).
    pub hold_ms: f32,
    /// Attenuation when gate is closed, in dB (typically 40–80 dB).
    pub range_db: f32,
}

impl Default for GateParams {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold_db: -60.0,
            attack_ms: 10.0,
            release_ms: 100.0,
            hold_ms: 50.0,
            range_db: 60.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Compressor parameters
// ---------------------------------------------------------------------------

/// Downward compressor parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressorParams {
    /// Enable/disable.
    pub enabled: bool,
    /// Threshold in dBFS. Signals above this level are compressed.
    pub threshold_db: f32,
    /// Compression ratio (1.0 = no compression, ∞ = limiting).
    pub ratio: f32,
    /// Attack time in milliseconds.
    pub attack_ms: f32,
    /// Release time in milliseconds.
    pub release_ms: f32,
    /// Make-up gain in dB.
    pub makeup_db: f32,
    /// Knee width in dB (0 = hard knee).
    pub knee_db: f32,
}

impl Default for CompressorParams {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold_db: -18.0,
            ratio: 4.0,
            attack_ms: 5.0,
            release_ms: 200.0,
            makeup_db: 0.0,
            knee_db: 6.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Filter parameters
// ---------------------------------------------------------------------------

/// First-order filter parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterParams {
    /// Whether the filter is active.
    pub enabled: bool,
    /// Cutoff frequency in Hz.
    pub cutoff_hz: f32,
}

impl FilterParams {
    /// HPF default (80 Hz).
    pub fn hpf_default() -> Self {
        Self {
            enabled: false,
            cutoff_hz: 80.0,
        }
    }

    /// LPF default (16 kHz).
    pub fn lpf_default() -> Self {
        Self {
            enabled: false,
            cutoff_hz: 16_000.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Aux send
// ---------------------------------------------------------------------------

/// A single auxiliary bus send from a channel strip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuxSend {
    /// Aux bus index (0-based).
    pub index: usize,
    /// Name of the aux bus.
    pub name: String,
    /// Send level in dB.
    pub level_db: f32,
    /// Whether this send is pre-fader (true) or post-fader (false).
    pub pre_fader: bool,
    /// Whether the send is muted.
    pub muted: bool,
}

impl AuxSend {
    /// Creates a new aux send with unity gain.
    pub fn new(index: usize, name: impl Into<String>) -> Self {
        Self {
            index,
            name: name.into(),
            level_db: 0.0,
            pre_fader: false,
            muted: false,
        }
    }
}

// ---------------------------------------------------------------------------
// ChannelStrip
// ---------------------------------------------------------------------------

/// Maximum number of aux sends per strip.
pub const MAX_AUX_SENDS: usize = 8;

/// Minimum fader/gain level in dB (−∞ approximation).
pub const FADER_MIN_DB: f32 = -120.0;
/// Maximum fader/gain level in dB.
pub const FADER_MAX_DB: f32 = 12.0;

/// A single channel strip in a routing/mixing console.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelStrip {
    /// Strip name.
    name: String,
    /// Input trim/pad in dB.
    input_gain_db: f32,
    /// Gate parameters.
    gate: GateParams,
    /// Compressor parameters.
    compressor: CompressorParams,
    /// High-pass filter.
    hpf: FilterParams,
    /// Low-pass filter.
    lpf: FilterParams,
    /// Post-dynamics master fader level in dB.
    fader_db: f32,
    /// Whether the strip is muted.
    muted: bool,
    /// Whether the strip is soloed.
    soloed: bool,
    /// Whether the strip is solo-safe (immune to other solos).
    solo_safe: bool,
    /// Aux sends.
    aux_sends: Vec<AuxSend>,
    /// Polarity invert.
    polarity_inverted: bool,
}

impl ChannelStrip {
    /// Creates a new channel strip with default settings.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            input_gain_db: 0.0,
            gate: GateParams::default(),
            compressor: CompressorParams::default(),
            hpf: FilterParams::hpf_default(),
            lpf: FilterParams::lpf_default(),
            fader_db: 0.0,
            muted: false,
            soloed: false,
            solo_safe: false,
            aux_sends: Vec::new(),
            polarity_inverted: false,
        }
    }

    /// Returns the strip name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the input gain in dB.
    pub fn input_gain_db(&self) -> f32 {
        self.input_gain_db
    }

    /// Sets the input gain in dB.
    pub fn set_input_gain_db(&mut self, gain_db: f32) -> Result<(), StripError> {
        if !(FADER_MIN_DB..=FADER_MAX_DB).contains(&gain_db) {
            return Err(StripError::OutOfRange {
                name: "input_gain_db".to_string(),
                value: gain_db,
                min: FADER_MIN_DB,
                max: FADER_MAX_DB,
            });
        }
        self.input_gain_db = gain_db;
        Ok(())
    }

    /// Returns the fader level in dB.
    pub fn fader_db(&self) -> f32 {
        self.fader_db
    }

    /// Sets the fader level in dB.
    pub fn set_fader_db(&mut self, fader_db: f32) -> Result<(), StripError> {
        if !(FADER_MIN_DB..=FADER_MAX_DB).contains(&fader_db) {
            return Err(StripError::OutOfRange {
                name: "fader_db".to_string(),
                value: fader_db,
                min: FADER_MIN_DB,
                max: FADER_MAX_DB,
            });
        }
        self.fader_db = fader_db;
        Ok(())
    }

    /// Returns the combined gain (input + fader) in linear scale.
    pub fn combined_gain_linear(&self) -> f32 {
        let total_db = self.input_gain_db + self.fader_db;
        db_to_linear(total_db)
    }

    /// Mutes the strip.
    pub fn mute(&mut self) {
        self.muted = true;
    }

    /// Unmutes the strip.
    pub fn unmute(&mut self) {
        self.muted = false;
    }

    /// Returns whether the strip is muted.
    pub fn is_muted(&self) -> bool {
        self.muted
    }

    /// Solos the strip.
    pub fn solo(&mut self) {
        self.soloed = true;
    }

    /// Un-solos the strip.
    pub fn unsolo(&mut self) {
        self.soloed = false;
    }

    /// Returns whether the strip is soloed.
    pub fn is_soloed(&self) -> bool {
        self.soloed
    }

    /// Sets solo-safe mode (strip remains audible when other channels are soloed).
    pub fn set_solo_safe(&mut self, safe: bool) {
        self.solo_safe = safe;
    }

    /// Returns whether the strip is solo-safe.
    pub fn is_solo_safe(&self) -> bool {
        self.solo_safe
    }

    /// Inverts the polarity.
    pub fn set_polarity_inverted(&mut self, inverted: bool) {
        self.polarity_inverted = inverted;
    }

    /// Returns whether the polarity is inverted.
    pub fn is_polarity_inverted(&self) -> bool {
        self.polarity_inverted
    }

    /// Returns a reference to the gate parameters.
    pub fn gate(&self) -> &GateParams {
        &self.gate
    }

    /// Returns a mutable reference to the gate parameters.
    pub fn gate_mut(&mut self) -> &mut GateParams {
        &mut self.gate
    }

    /// Returns a reference to the compressor parameters.
    pub fn compressor(&self) -> &CompressorParams {
        &self.compressor
    }

    /// Returns a mutable reference to the compressor parameters.
    pub fn compressor_mut(&mut self) -> &mut CompressorParams {
        &mut self.compressor
    }

    /// Returns a reference to the HPF parameters.
    pub fn hpf(&self) -> &FilterParams {
        &self.hpf
    }

    /// Returns a mutable reference to the HPF parameters.
    pub fn hpf_mut(&mut self) -> &mut FilterParams {
        &mut self.hpf
    }

    /// Returns a reference to the LPF parameters.
    pub fn lpf(&self) -> &FilterParams {
        &self.lpf
    }

    /// Returns a mutable reference to the LPF parameters.
    pub fn lpf_mut(&mut self) -> &mut FilterParams {
        &mut self.lpf
    }

    /// Adds an aux send. Returns `Err` if the index is already in use or exceeds the limit.
    pub fn add_aux_send(&mut self, send: AuxSend) -> Result<(), StripError> {
        if send.index >= MAX_AUX_SENDS {
            return Err(StripError::AuxSendOutOfRange(send.index, MAX_AUX_SENDS - 1));
        }
        if self.aux_sends.iter().any(|s| s.index == send.index) {
            // Replace existing send
            let pos = self
                .aux_sends
                .iter()
                .position(|s| s.index == send.index)
                .ok_or(StripError::AuxSendOutOfRange(send.index, MAX_AUX_SENDS - 1))?;
            self.aux_sends[pos] = send;
        } else {
            self.aux_sends.push(send);
        }
        Ok(())
    }

    /// Returns a slice of all aux sends.
    pub fn aux_sends(&self) -> &[AuxSend] {
        &self.aux_sends
    }

    /// Returns the aux send at the given index, if present.
    pub fn aux_send(&self, index: usize) -> Option<&AuxSend> {
        self.aux_sends.iter().find(|s| s.index == index)
    }

    /// Removes the aux send at the given index.
    pub fn remove_aux_send(&mut self, index: usize) {
        self.aux_sends.retain(|s| s.index != index);
    }

    /// Applies a gain `factor` (linear) to a slice of samples.
    ///
    /// If muted, the output is zeroed. Polarity inversion is applied if set.
    pub fn process_gain(&self, samples: &mut [f32]) {
        if self.muted {
            for s in samples.iter_mut() {
                *s = 0.0;
            }
            return;
        }
        let gain = self.combined_gain_linear();
        let factor = if self.polarity_inverted { -gain } else { gain };
        for s in samples.iter_mut() {
            *s *= factor;
        }
    }
}

// ---------------------------------------------------------------------------
// StripBus
// ---------------------------------------------------------------------------

/// A collection of named channel strips forming a routing/mixing bus.
pub struct StripBus {
    strips: Vec<ChannelStrip>,
}

impl StripBus {
    /// Creates an empty bus.
    pub fn new() -> Self {
        Self { strips: Vec::new() }
    }

    /// Adds a strip. Returns `Err` if a strip with the same name already exists.
    pub fn add_strip(&mut self, strip: ChannelStrip) -> Result<(), StripError> {
        if self.strips.iter().any(|s| s.name() == strip.name()) {
            return Err(StripError::DuplicateStrip(strip.name().to_string()));
        }
        self.strips.push(strip);
        Ok(())
    }

    /// Returns the number of strips.
    pub fn strip_count(&self) -> usize {
        self.strips.len()
    }

    /// Returns a reference to the strip with the given name.
    pub fn strip(&self, name: &str) -> Option<&ChannelStrip> {
        self.strips.iter().find(|s| s.name() == name)
    }

    /// Returns a mutable reference to the strip with the given name.
    pub fn strip_mut(&mut self, name: &str) -> Option<&mut ChannelStrip> {
        self.strips.iter_mut().find(|s| s.name() == name)
    }

    /// Removes and returns the strip with the given name.
    pub fn remove_strip(&mut self, name: &str) -> Option<ChannelStrip> {
        let pos = self.strips.iter().position(|s| s.name() == name)?;
        Some(self.strips.remove(pos))
    }

    /// Mutes all strips.
    pub fn mute_all(&mut self) {
        for strip in &mut self.strips {
            strip.mute();
        }
    }

    /// Unmutes all strips.
    pub fn unmute_all(&mut self) {
        for strip in &mut self.strips {
            strip.unmute();
        }
    }

    /// Returns the names of all soloed strips.
    pub fn soloed_strips(&self) -> Vec<&str> {
        self.strips
            .iter()
            .filter(|s| s.is_soloed())
            .map(|s| s.name())
            .collect()
    }

    /// Returns true if any strip is soloed (excluding solo-safe).
    pub fn any_soloed(&self) -> bool {
        self.strips.iter().any(|s| s.is_soloed())
    }

    /// Returns all strips as a slice.
    pub fn strips(&self) -> &[ChannelStrip] {
        &self.strips
    }
}

impl Default for StripBus {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Converts dB to linear gain.
fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_default_values() {
        let strip = ChannelStrip::new("Test Strip");
        assert_eq!(strip.name(), "Test Strip");
        assert!((strip.input_gain_db()).abs() < f32::EPSILON);
        assert!((strip.fader_db()).abs() < f32::EPSILON);
        assert!(!strip.is_muted());
        assert!(!strip.is_soloed());
        assert!(!strip.is_solo_safe());
        assert!(!strip.is_polarity_inverted());
    }

    #[test]
    fn test_input_gain_set_and_get() {
        let mut strip = ChannelStrip::new("Mic");
        strip.set_input_gain_db(-6.0).expect("valid gain");
        assert!((strip.input_gain_db() - -6.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_input_gain_out_of_range() {
        let mut strip = ChannelStrip::new("Mic");
        let result = strip.set_input_gain_db(50.0);
        assert!(matches!(result, Err(StripError::OutOfRange { .. })));
    }

    #[test]
    fn test_fader_set_and_get() {
        let mut strip = ChannelStrip::new("Fader Test");
        strip.set_fader_db(-3.0).expect("valid");
        assert!((strip.fader_db() - -3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_mute_and_unmute() {
        let mut strip = ChannelStrip::new("Mute Test");
        strip.mute();
        assert!(strip.is_muted());
        strip.unmute();
        assert!(!strip.is_muted());
    }

    #[test]
    fn test_muted_process_zeroes_output() {
        let mut strip = ChannelStrip::new("Muted");
        strip.mute();
        let mut samples = vec![1.0_f32, -1.0, 0.5, -0.5];
        strip.process_gain(&mut samples);
        for &s in &samples {
            assert!(s.abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_process_gain_applies_combined_gain() {
        let mut strip = ChannelStrip::new("Gain Apply");
        strip.set_input_gain_db(-6.0).expect("ok");
        strip.set_fader_db(-6.0).expect("ok");
        // Combined: -12 dB, linear ≈ 0.2512
        let expected_linear = db_to_linear(-12.0);
        let mut samples = vec![1.0_f32];
        strip.process_gain(&mut samples);
        assert!((samples[0] - expected_linear).abs() < 1e-5);
    }

    #[test]
    fn test_polarity_invert() {
        let mut strip = ChannelStrip::new("Polarity");
        strip.set_polarity_inverted(true);
        let mut samples = vec![1.0_f32];
        strip.process_gain(&mut samples);
        assert!(samples[0] < 0.0, "polarity should invert");
    }

    #[test]
    fn test_solo_and_solo_safe() {
        let mut strip = ChannelStrip::new("Solo Test");
        strip.solo();
        assert!(strip.is_soloed());
        strip.set_solo_safe(true);
        assert!(strip.is_solo_safe());
        strip.unsolo();
        assert!(!strip.is_soloed());
    }

    #[test]
    fn test_aux_send_add_and_get() {
        let mut strip = ChannelStrip::new("Aux Test");
        let send = AuxSend::new(0, "Reverb Return");
        strip.add_aux_send(send).expect("added");
        let aux = strip.aux_send(0).expect("exists");
        assert_eq!(aux.name, "Reverb Return");
    }

    #[test]
    fn test_aux_send_out_of_range() {
        let mut strip = ChannelStrip::new("OOR");
        let send = AuxSend::new(MAX_AUX_SENDS, "Bad");
        let result = strip.add_aux_send(send);
        assert!(matches!(result, Err(StripError::AuxSendOutOfRange(..))));
    }

    #[test]
    fn test_strip_bus_add_and_count() {
        let mut bus = StripBus::new();
        bus.add_strip(ChannelStrip::new("Ch 1")).expect("ok");
        bus.add_strip(ChannelStrip::new("Ch 2")).expect("ok");
        assert_eq!(bus.strip_count(), 2);
    }

    #[test]
    fn test_strip_bus_duplicate_error() {
        let mut bus = StripBus::new();
        bus.add_strip(ChannelStrip::new("Ch 1")).expect("ok");
        let result = bus.add_strip(ChannelStrip::new("Ch 1"));
        assert!(matches!(result, Err(StripError::DuplicateStrip(_))));
    }

    #[test]
    fn test_strip_bus_mute_all() {
        let mut bus = StripBus::new();
        bus.add_strip(ChannelStrip::new("A")).expect("ok");
        bus.add_strip(ChannelStrip::new("B")).expect("ok");
        bus.mute_all();
        for strip in bus.strips() {
            assert!(strip.is_muted());
        }
        bus.unmute_all();
        for strip in bus.strips() {
            assert!(!strip.is_muted());
        }
    }

    #[test]
    fn test_strip_bus_soloed_strips() {
        let mut bus = StripBus::new();
        let mut s1 = ChannelStrip::new("S1");
        s1.solo();
        bus.add_strip(s1).expect("ok");
        bus.add_strip(ChannelStrip::new("S2")).expect("ok");

        let soloed = bus.soloed_strips();
        assert_eq!(soloed.len(), 1);
        assert_eq!(soloed[0], "S1");
        assert!(bus.any_soloed());
    }

    #[test]
    fn test_strip_bus_remove() {
        let mut bus = StripBus::new();
        bus.add_strip(ChannelStrip::new("Remove Me")).expect("ok");
        assert_eq!(bus.strip_count(), 1);
        let removed = bus.remove_strip("Remove Me");
        assert!(removed.is_some());
        assert_eq!(bus.strip_count(), 0);
    }

    #[test]
    fn test_combined_gain_linear() {
        let mut strip = ChannelStrip::new("Linear");
        strip.set_input_gain_db(0.0).expect("ok");
        strip.set_fader_db(0.0).expect("ok");
        let lin = strip.combined_gain_linear();
        assert!((lin - 1.0).abs() < 1e-5, "unity gain = 1.0 linear");
    }

    #[test]
    fn test_gate_params_access() {
        let mut strip = ChannelStrip::new("Gate");
        strip.gate_mut().enabled = true;
        strip.gate_mut().threshold_db = -40.0;
        assert!(strip.gate().enabled);
        assert!((strip.gate().threshold_db - -40.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compressor_params_access() {
        let mut strip = ChannelStrip::new("Comp");
        strip.compressor_mut().enabled = true;
        strip.compressor_mut().ratio = 8.0;
        assert!(strip.compressor().enabled);
        assert!((strip.compressor().ratio - 8.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_filter_params_access() {
        let mut strip = ChannelStrip::new("Filter");
        strip.hpf_mut().enabled = true;
        strip.hpf_mut().cutoff_hz = 120.0;
        strip.lpf_mut().enabled = true;
        strip.lpf_mut().cutoff_hz = 12_000.0;

        assert!(strip.hpf().enabled);
        assert!((strip.hpf().cutoff_hz - 120.0).abs() < f32::EPSILON);
        assert!(strip.lpf().enabled);
        assert!((strip.lpf().cutoff_hz - 12_000.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_remove_aux_send() {
        let mut strip = ChannelStrip::new("Aux Remove");
        strip.add_aux_send(AuxSend::new(0, "Fx")).expect("ok");
        strip.add_aux_send(AuxSend::new(1, "Mon")).expect("ok");
        assert_eq!(strip.aux_sends().len(), 2);
        strip.remove_aux_send(0);
        assert_eq!(strip.aux_sends().len(), 1);
        assert!(strip.aux_send(0).is_none());
        assert!(strip.aux_send(1).is_some());
    }
}
