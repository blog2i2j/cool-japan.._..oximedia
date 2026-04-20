//! Stem mixing — stem definitions, level control, mute/solo, and bus routing.
//!
//! A *stem* is a named submix that groups related channels (e.g. "Music",
//! "Dialog", "Effects").  The `StemMixer` manages a set of stems, accumulates
//! audio buffers into each stem, applies level / mute / solo controls, and
//! routes the resulting stems to downstream buses.
//!
//! # Architecture
//!
//! ```text
//! Channels ──┐
//!            ├─► Stem (Music)   ──► Stem Bus ──► Master
//! Channels ──┤
//!            ├─► Stem (Dialog)  ──► Stem Bus ──► Master
//! Channels ──┘
//!            └─► Stem (Effects) ──► Stem Bus ──► Master
//! ```
//!
//! # Example
//!
//! ```rust
//! use oximedia_mixer::stem_mixer::{StemMixer, StemConfig, StemCategory};
//!
//! let mut mixer = StemMixer::new(48000);
//! let music_id = mixer.add_stem(StemConfig {
//!     name: "Music".into(),
//!     category: StemCategory::Music,
//!     ..Default::default()
//! }).expect("add_stem should succeed");
//! mixer.set_stem_level(music_id, 0.8).expect("set_stem_level with valid gain should succeed");
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{BusId, ChannelId, MixerError, MixerResult};

// ── Stem ID ─────────────────────────────────────────────────────────────────

/// Unique identifier for a stem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StemId(pub u32);

impl std::fmt::Display for StemId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Stem({})", self.0)
    }
}

// ── StemCategory ────────────────────────────────────────────────────────────

/// Semantic category of a stem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum StemCategory {
    /// Music / score.
    #[default]
    Music,
    /// Dialogue / voice-over.
    Dialog,
    /// Sound effects.
    Effects,
    /// Ambience / background.
    Ambience,
    /// Foley.
    Foley,
    /// Custom / user-defined category.
    Custom,
}

impl std::fmt::Display for StemCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StemCategory::Music => write!(f, "Music"),
            StemCategory::Dialog => write!(f, "Dialog"),
            StemCategory::Effects => write!(f, "Effects"),
            StemCategory::Ambience => write!(f, "Ambience"),
            StemCategory::Foley => write!(f, "Foley"),
            StemCategory::Custom => write!(f, "Custom"),
        }
    }
}

// ── StemConfig ───────────────────────────────────────────────────────────────

/// Configuration used when creating a stem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StemConfig {
    /// Human-readable name for this stem (e.g. "Music", "Dialog").
    pub name: String,
    /// Semantic category.
    pub category: StemCategory,
    /// Initial linear gain (1.0 = 0 dBFS).
    pub level: f32,
    /// Whether the stem starts muted.
    pub muted: bool,
    /// Optional bus to route this stem's output into.
    pub target_bus: Option<BusId>,
}

impl Default for StemConfig {
    fn default() -> Self {
        Self {
            name: String::from("Stem"),
            category: StemCategory::Custom,
            level: 1.0,
            muted: false,
            target_bus: None,
        }
    }
}

// ── StemSoloState ────────────────────────────────────────────────────────────

/// Tracks the solo state across all stems.
#[derive(Debug, Clone, Default)]
struct StemSoloState {
    /// IDs of stems that are currently soloed.
    soloed: std::collections::HashSet<StemId>,
}

impl StemSoloState {
    fn any_soloed(&self) -> bool {
        !self.soloed.is_empty()
    }

    fn is_soloed(&self, id: StemId) -> bool {
        self.soloed.contains(&id)
    }

    fn set_solo(&mut self, id: StemId, solo: bool) {
        if solo {
            self.soloed.insert(id);
        } else {
            self.soloed.remove(&id);
        }
    }
}

// ── Stem ─────────────────────────────────────────────────────────────────────

/// A single stem with its channels, level, mute, and solo states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stem {
    /// Unique identifier.
    pub id: StemId,
    /// Human-readable name.
    pub name: String,
    /// Semantic category.
    pub category: StemCategory,
    /// Linear output gain (0.0 – +∞, nominal 1.0).
    pub level: f32,
    /// Mute flag — if true the stem produces silence.
    pub muted: bool,
    /// Solo flag — if any stem is soloed, only soloed stems are heard.
    pub solo: bool,
    /// Channel IDs that feed into this stem.
    pub channels: Vec<ChannelId>,
    /// Optional downstream bus for routing.
    pub target_bus: Option<BusId>,
}

impl Stem {
    fn new(id: StemId, config: StemConfig) -> Self {
        Self {
            id,
            name: config.name,
            category: config.category,
            level: config.level.clamp(0.0, 4.0),
            muted: config.muted,
            solo: false,
            channels: Vec::new(),
            target_bus: config.target_bus,
        }
    }

    /// Returns true if this stem should be audible given `solo_state`.
    fn is_audible(&self, solo_state: &StemSoloState) -> bool {
        if self.muted {
            return false;
        }
        if solo_state.any_soloed() {
            return solo_state.is_soloed(self.id);
        }
        true
    }

    /// Effective linear gain for output, taking mute/solo into account.
    fn effective_gain(&self, solo_state: &StemSoloState) -> f32 {
        if self.is_audible(solo_state) {
            self.level
        } else {
            0.0
        }
    }
}

// ── StemMixerSnapshot ────────────────────────────────────────────────────────

/// Read-only metering snapshot for a single stem (updated after `process()`).
#[derive(Debug, Clone)]
pub struct StemSnapshot {
    /// Stem identifier.
    pub id: StemId,
    /// Stem name.
    pub name: String,
    /// Whether the stem was audible in the last process block.
    pub audible: bool,
    /// Peak level of the left channel in the last block (linear, 0.0–1.0+).
    pub peak_left: f32,
    /// Peak level of the right channel in the last block (linear, 0.0–1.0+).
    pub peak_right: f32,
}

// ── StemMixer ────────────────────────────────────────────────────────────────

/// Manages stems: creation, routing, level/mute/solo control, and processing.
pub struct StemMixer {
    /// Sample rate in Hz.
    sample_rate: u32,
    /// All stems indexed by ID.
    stems: HashMap<StemId, Stem>,
    /// Insertion-ordered IDs for deterministic iteration.
    order: Vec<StemId>,
    /// Next ID to assign.
    next_id: u32,
    /// Global solo state tracking which stems are soloed.
    solo_state: StemSoloState,
    /// Last metering snapshot per stem.
    snapshots: HashMap<StemId, StemSnapshot>,
}

impl StemMixer {
    /// Create a new `StemMixer` for the given sample rate.
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            stems: HashMap::new(),
            order: Vec::new(),
            next_id: 1,
            solo_state: StemSoloState::default(),
            snapshots: HashMap::new(),
        }
    }

    /// Returns the configured sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    // ── Stem management ──────────────────────────────────────────────────────

    /// Add a new stem and return its `StemId`.
    pub fn add_stem(&mut self, config: StemConfig) -> MixerResult<StemId> {
        let id = StemId(self.next_id);
        self.next_id += 1;
        self.stems.insert(id, Stem::new(id, config));
        self.order.push(id);
        Ok(id)
    }

    /// Remove a stem by ID.  Returns an error if the stem does not exist.
    pub fn remove_stem(&mut self, id: StemId) -> MixerResult<()> {
        if self.stems.remove(&id).is_none() {
            return Err(MixerError::InvalidParameter(format!(
                "Stem {id} not found"
            )));
        }
        self.order.retain(|&s| s != id);
        self.solo_state.set_solo(id, false);
        self.snapshots.remove(&id);
        Ok(())
    }

    /// Return a reference to a stem.
    pub fn stem(&self, id: StemId) -> MixerResult<&Stem> {
        self.stems
            .get(&id)
            .ok_or_else(|| MixerError::InvalidParameter(format!("Stem {id} not found")))
    }

    /// Return a mutable reference to a stem.
    pub fn stem_mut(&mut self, id: StemId) -> MixerResult<&mut Stem> {
        self.stems
            .get_mut(&id)
            .ok_or_else(|| MixerError::InvalidParameter(format!("Stem {id} not found")))
    }

    /// Iterate over all stems in insertion order.
    pub fn stems(&self) -> impl Iterator<Item = &Stem> {
        self.order.iter().filter_map(|id| self.stems.get(id))
    }

    // ── Channel assignment ───────────────────────────────────────────────────

    /// Assign a channel to a stem.  The same channel may only belong to one
    /// stem at a time; it is automatically removed from its previous stem.
    pub fn assign_channel(&mut self, channel: ChannelId, stem: StemId) -> MixerResult<()> {
        if !self.stems.contains_key(&stem) {
            return Err(MixerError::InvalidParameter(format!(
                "Stem {stem} not found"
            )));
        }
        // Remove from any existing stem.
        for s in self.stems.values_mut() {
            s.channels.retain(|&c| c != channel);
        }
        self.stems
            .get_mut(&stem)
            .ok_or_else(|| MixerError::InvalidParameter(format!("Stem {stem} not found")))?
            .channels
            .push(channel);
        Ok(())
    }

    /// Remove a channel from whatever stem currently contains it.
    pub fn unassign_channel(&mut self, channel: ChannelId) {
        for s in self.stems.values_mut() {
            s.channels.retain(|&c| c != channel);
        }
    }

    /// Return the stem ID that owns `channel`, if any.
    pub fn stem_for_channel(&self, channel: ChannelId) -> Option<StemId> {
        self.order
            .iter()
            .find(|&&id| {
                self.stems
                    .get(&id)
                    .map_or(false, |s| s.channels.contains(&channel))
            })
            .copied()
    }

    // ── Level / mute / solo ──────────────────────────────────────────────────

    /// Set linear gain for a stem (clamped to 0.0 – 4.0, i.e. ≤ +12 dBFS).
    pub fn set_stem_level(&mut self, id: StemId, level: f32) -> MixerResult<()> {
        if !level.is_finite() || level < 0.0 {
            return Err(MixerError::InvalidParameter(format!(
                "Level must be finite and non-negative, got {level}"
            )));
        }
        let stem = self.stem_mut(id)?;
        stem.level = level.min(4.0);
        Ok(())
    }

    /// Get the current linear gain for a stem.
    pub fn stem_level(&self, id: StemId) -> MixerResult<f32> {
        Ok(self.stem(id)?.level)
    }

    /// Mute or un-mute a stem.
    pub fn set_stem_mute(&mut self, id: StemId, muted: bool) -> MixerResult<()> {
        self.stem_mut(id)?.muted = muted;
        Ok(())
    }

    /// Solo or un-solo a stem (exclusive-solo semantics when multiple soloed).
    pub fn set_stem_solo(&mut self, id: StemId, solo: bool) -> MixerResult<()> {
        // Validate.
        if !self.stems.contains_key(&id) {
            return Err(MixerError::InvalidParameter(format!(
                "Stem {id} not found"
            )));
        }
        self.stems.get_mut(&id).map(|s| s.solo = solo);
        self.solo_state.set_solo(id, solo);
        Ok(())
    }

    /// Returns true if any stem is currently soloed.
    pub fn any_soloed(&self) -> bool {
        self.solo_state.any_soloed()
    }

    // ── Bus routing ──────────────────────────────────────────────────────────

    /// Route a stem's output to the given bus.
    pub fn route_stem_to_bus(&mut self, id: StemId, bus: BusId) -> MixerResult<()> {
        self.stem_mut(id)?.target_bus = Some(bus);
        Ok(())
    }

    /// Clear the bus routing for a stem (stem will not be routed anywhere).
    pub fn clear_stem_routing(&mut self, id: StemId) -> MixerResult<()> {
        self.stem_mut(id)?.target_bus = None;
        Ok(())
    }

    // ── Processing ───────────────────────────────────────────────────────────

    /// Process a stem mix.
    ///
    /// `channel_buffers` maps each `ChannelId` to its stereo output buffer
    /// `(left_samples, right_samples)` where each slice has `num_frames` samples.
    ///
    /// Returns a map from `StemId` to the accumulated stereo output buffer
    /// `(Vec<f32>, Vec<f32>)` for each stem, ready for downstream bus routing.
    pub fn process(
        &mut self,
        channel_buffers: &HashMap<ChannelId, (Vec<f32>, Vec<f32>)>,
        num_frames: usize,
    ) -> HashMap<StemId, (Vec<f32>, Vec<f32>)> {
        let mut outputs: HashMap<StemId, (Vec<f32>, Vec<f32>)> = HashMap::new();

        for &stem_id in &self.order {
            let Some(stem) = self.stems.get(&stem_id) else {
                continue;
            };

            let gain = stem.effective_gain(&self.solo_state);
            let mut left = vec![0.0_f32; num_frames];
            let mut right = vec![0.0_f32; num_frames];

            for &chan_id in &stem.channels {
                if let Some((ch_left, ch_right)) = channel_buffers.get(&chan_id) {
                    let l_len = ch_left.len().min(num_frames);
                    let r_len = ch_right.len().min(num_frames);
                    for i in 0..l_len {
                        left[i] += ch_left[i];
                    }
                    for i in 0..r_len {
                        right[i] += ch_right[i];
                    }
                }
            }

            // Apply stem gain.
            if gain != 1.0 {
                for s in left.iter_mut() {
                    *s *= gain;
                }
                for s in right.iter_mut() {
                    *s *= gain;
                }
            }

            // Update metering snapshot.
            let peak_l = left.iter().copied().fold(0.0_f32, |a, x| a.max(x.abs()));
            let peak_r = right.iter().copied().fold(0.0_f32, |a, x| a.max(x.abs()));
            self.snapshots.insert(
                stem_id,
                StemSnapshot {
                    id: stem_id,
                    name: stem.name.clone(),
                    audible: gain > 0.0,
                    peak_left: peak_l,
                    peak_right: peak_r,
                },
            );

            outputs.insert(stem_id, (left, right));
        }

        outputs
    }

    /// Return the most recent metering snapshot for a stem.
    pub fn snapshot(&self, id: StemId) -> Option<&StemSnapshot> {
        self.snapshots.get(&id)
    }

    /// Return all current metering snapshots.
    pub fn all_snapshots(&self) -> Vec<&StemSnapshot> {
        self.order
            .iter()
            .filter_map(|id| self.snapshots.get(id))
            .collect()
    }

    /// Return total number of stems.
    pub fn stem_count(&self) -> usize {
        self.stems.len()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::ChannelId;

    fn make_channel_id(n: u64) -> ChannelId {
        ChannelId(uuid::Uuid::from_u128(n as u128))
    }

    fn make_bus_id() -> BusId {
        BusId(uuid::Uuid::new_v4())
    }

    fn sine_buffer(frames: usize, amplitude: f32) -> Vec<f32> {
        (0..frames)
            .map(|i| amplitude * (i as f32 * std::f32::consts::TAU / 64.0).sin())
            .collect()
    }

    #[test]
    fn test_add_and_remove_stem() {
        let mut mixer = StemMixer::new(48000);
        let id = mixer
            .add_stem(StemConfig {
                name: "Music".into(),
                category: StemCategory::Music,
                ..Default::default()
            })
            .expect("add_stem should succeed");
        assert_eq!(mixer.stem_count(), 1);
        mixer.remove_stem(id).expect("remove_stem should succeed for existing id");
        assert_eq!(mixer.stem_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_stem_errors() {
        let mut mixer = StemMixer::new(48000);
        assert!(mixer.remove_stem(StemId(999)).is_err());
    }

    #[test]
    fn test_stem_level_clamps_and_stores() {
        let mut mixer = StemMixer::new(48000);
        let id = mixer
            .add_stem(StemConfig::default())
            .expect("add_stem should succeed");
        mixer.set_stem_level(id, 0.5).expect("set_stem_level with valid gain should succeed");
        assert!((mixer.stem_level(id).expect("stem_level should return value for existing id") - 0.5).abs() < 1e-6);
        // Clamped to 4.0
        mixer.set_stem_level(id, 100.0).expect("set_stem_level should clamp and succeed");
        assert!((mixer.stem_level(id).expect("stem_level should return value for existing id") - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_negative_level_errors() {
        let mut mixer = StemMixer::new(48000);
        let id = mixer.add_stem(StemConfig::default()).expect("add_stem should succeed");
        assert!(mixer.set_stem_level(id, -1.0).is_err());
    }

    #[test]
    fn test_mute_silences_output() {
        let mut mixer = StemMixer::new(48000);
        let id = mixer
            .add_stem(StemConfig {
                name: "FX".into(),
                category: StemCategory::Effects,
                ..Default::default()
            })
            .expect("add_stem should succeed");
        let chan = make_channel_id(1);
        mixer.assign_channel(chan, id).expect("assign_channel should succeed");
        mixer.set_stem_mute(id, true).expect("set_stem_mute should succeed");

        let buf = sine_buffer(64, 0.8);
        let mut ch_map = HashMap::new();
        ch_map.insert(chan, (buf.clone(), buf));
        let out = mixer.process(&ch_map, 64);
        let (l, _r) = out.get(&id).expect("stem output should be present in process result");
        let peak: f32 = l.iter().copied().fold(0.0, f32::max);
        assert!(
            peak < 1e-9,
            "Muted stem should be silent, got peak {peak}"
        );
    }

    #[test]
    fn test_solo_isolates_stem() {
        let mut mixer = StemMixer::new(48000);
        let id_music = mixer
            .add_stem(StemConfig {
                name: "Music".into(),
                category: StemCategory::Music,
                ..Default::default()
            })
            .expect("add_stem Music should succeed");
        let id_fx = mixer
            .add_stem(StemConfig {
                name: "FX".into(),
                category: StemCategory::Effects,
                ..Default::default()
            })
            .expect("add_stem FX should succeed");

        let chan_m = make_channel_id(10);
        let chan_f = make_channel_id(11);
        mixer.assign_channel(chan_m, id_music).expect("assign_channel to music stem should succeed");
        mixer.assign_channel(chan_f, id_fx).expect("assign_channel to fx stem should succeed");

        // Solo Music — FX should be silent.
        mixer.set_stem_solo(id_music, true).expect("set_stem_solo should succeed");

        let buf = sine_buffer(64, 1.0);
        let mut ch_map = HashMap::new();
        ch_map.insert(chan_m, (buf.clone(), buf.clone()));
        ch_map.insert(chan_f, (buf.clone(), buf));
        let out = mixer.process(&ch_map, 64);

        let (fx_l, _) = out.get(&id_fx).expect("fx stem output should be present in process result");
        let peak_fx: f32 = fx_l.iter().copied().fold(0.0, f32::max);
        assert!(
            peak_fx < 1e-9,
            "Non-soloed stem should be silent, got {peak_fx}"
        );

        let (mu_l, _) = out.get(&id_music).expect("music stem output should be present in process result");
        let peak_music: f32 = mu_l.iter().copied().fold(0.0_f32, |a, x| a.max(x.abs()));
        assert!(
            peak_music > 0.1,
            "Soloed stem should pass audio, got {peak_music}"
        );
    }

    #[test]
    fn test_channel_assignment_exclusive() {
        let mut mixer = StemMixer::new(48000);
        let id_a = mixer.add_stem(StemConfig::default()).expect("add_stem A should succeed");
        let id_b = mixer.add_stem(StemConfig::default()).expect("add_stem B should succeed");
        let chan = make_channel_id(5);

        mixer.assign_channel(chan, id_a).expect("assign_channel to stem A should succeed");
        assert_eq!(mixer.stem_for_channel(chan), Some(id_a));

        // Reassign to id_b — should leave id_a.
        mixer.assign_channel(chan, id_b).expect("reassign channel to stem B should succeed");
        assert_eq!(mixer.stem_for_channel(chan), Some(id_b));
        assert_eq!(mixer.stem(id_a).expect("stem A should still exist").channels.len(), 0);
    }

    #[test]
    fn test_bus_routing() {
        let mut mixer = StemMixer::new(48000);
        let id = mixer.add_stem(StemConfig::default()).expect("add_stem should succeed");
        let bus = make_bus_id();
        mixer.route_stem_to_bus(id, bus).expect("route_stem_to_bus should succeed");
        assert_eq!(mixer.stem(id).expect("stem should exist").target_bus, Some(bus));
        mixer.clear_stem_routing(id).expect("clear_stem_routing should succeed");
        assert_eq!(mixer.stem(id).expect("stem should still exist after clear").target_bus, None);
    }

    #[test]
    fn test_process_accumulates_channels() {
        let mut mixer = StemMixer::new(48000);
        let id = mixer
            .add_stem(StemConfig {
                name: "Dialog".into(),
                category: StemCategory::Dialog,
                level: 1.0,
                ..Default::default()
            })
            .expect("add_stem Dialog should succeed");
        let c1 = make_channel_id(20);
        let c2 = make_channel_id(21);
        mixer.assign_channel(c1, id).expect("assign channel c1 should succeed");
        mixer.assign_channel(c2, id).expect("assign channel c2 should succeed");

        let buf1 = vec![0.5_f32; 64];
        let buf2 = vec![0.3_f32; 64];
        let mut ch_map = HashMap::new();
        ch_map.insert(c1, (buf1.clone(), buf1));
        ch_map.insert(c2, (buf2.clone(), buf2));

        let out = mixer.process(&ch_map, 64);
        let (l, _) = out.get(&id).expect("dialog stem output should be present in process result");
        // 0.5 + 0.3 = 0.8
        assert!(
            (l[0] - 0.8).abs() < 1e-5,
            "Expected 0.8, got {}",
            l[0]
        );
    }

    #[test]
    fn test_snapshot_updated_after_process() {
        let mut mixer = StemMixer::new(48000);
        let id = mixer.add_stem(StemConfig::default()).expect("add_stem should succeed");
        let chan = make_channel_id(30);
        mixer.assign_channel(chan, id).expect("assign_channel should succeed");

        let buf = vec![0.7_f32; 64];
        let mut ch_map = HashMap::new();
        ch_map.insert(chan, (buf.clone(), buf));
        mixer.process(&ch_map, 64);

        let snap = mixer.snapshot(id).expect("snapshot should be present after process");
        assert!(snap.audible);
        assert!((snap.peak_left - 0.7).abs() < 1e-5);
    }
}
