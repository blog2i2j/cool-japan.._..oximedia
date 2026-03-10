//! Monitor mix management for the `OxiMedia` mixer.
//!
//! A *monitor mix* (also called a *cue mix* or *headphone mix*) is an
//! independent mix sent to a performer's headphones or a control-room
//! speaker while the main mix plays back.  This module supports multiple
//! simultaneous monitor mixes, each with per-channel send levels and a
//! global output trim.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

// ────────────────────────────────────────────────────────────────────────────
// MonitorMixId
// ────────────────────────────────────────────────────────────────────────────

/// Opaque identifier for a monitor mix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MonitorMixId(pub u32);

impl std::fmt::Display for MonitorMixId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MonitorMix({})", self.0)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// ChannelSend
// ────────────────────────────────────────────────────────────────────────────

/// Per-channel contribution to a monitor mix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelSend {
    /// Send level in the range `[0.0, 2.0]` (1.0 = unity).
    pub level: f32,
    /// Pan position for this send (`-1.0` = left, `0.0` = centre, `1.0` = right).
    pub pan: f32,
    /// Whether this send is muted.
    pub muted: bool,
}

impl Default for ChannelSend {
    fn default() -> Self {
        Self {
            level: 1.0,
            pan: 0.0,
            muted: false,
        }
    }
}

impl ChannelSend {
    /// Create a unity-gain, centre-panned, unmuted send.
    #[must_use]
    pub fn unity() -> Self {
        Self::default()
    }

    /// Effective send level: `level` if not muted, otherwise `0.0`.
    #[must_use]
    pub fn effective_level(&self) -> f32 {
        if self.muted {
            0.0
        } else {
            self.level
        }
    }

    /// Left channel gain from panning law (linear, constant-power not assumed).
    #[must_use]
    pub fn left_gain(&self) -> f32 {
        let t = (1.0 - self.pan) * 0.5; // 0..1 (1 = full left)
        self.effective_level() * t
    }

    /// Right channel gain from panning law.
    #[must_use]
    pub fn right_gain(&self) -> f32 {
        let t = (1.0 + self.pan) * 0.5; // 0..1 (1 = full right)
        self.effective_level() * t
    }
}

// ────────────────────────────────────────────────────────────────────────────
// MonitorMix
// ────────────────────────────────────────────────────────────────────────────

/// A single independent monitor mix.
#[derive(Debug, Clone)]
pub struct MonitorMix {
    /// Unique identifier.
    pub id: MonitorMixId,
    /// Display name (e.g., `"Drummer Headphones"`).
    pub name: String,
    /// Global output trim in dB (-∞ to +6 dB range; stored as linear).
    pub output_trim: f32,
    /// Whether this mix is globally muted.
    pub muted: bool,
    /// Per-channel sends, keyed by channel strip index.
    sends: HashMap<u32, ChannelSend>,
}

impl MonitorMix {
    /// Create a new monitor mix at unity output trim.
    #[must_use]
    pub fn new(id: MonitorMixId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            output_trim: 1.0,
            muted: false,
            sends: HashMap::new(),
        }
    }

    /// Set or update the send for `channel_id`.
    pub fn set_send(&mut self, channel_id: u32, send: ChannelSend) {
        self.sends.insert(channel_id, send);
    }

    /// Get the send for `channel_id`, or `None` if not configured.
    #[must_use]
    pub fn get_send(&self, channel_id: u32) -> Option<&ChannelSend> {
        self.sends.get(&channel_id)
    }

    /// Remove the send for `channel_id`.
    pub fn remove_send(&mut self, channel_id: u32) {
        self.sends.remove(&channel_id);
    }

    /// Set the output trim (linear scale, clamped to `[0.0, 2.0]`).
    pub fn set_output_trim(&mut self, trim: f32) {
        self.output_trim = trim.clamp(0.0, 2.0);
    }

    /// Effective output trim: `output_trim` if not globally muted, else `0.0`.
    #[must_use]
    pub fn effective_trim(&self) -> f32 {
        if self.muted {
            0.0
        } else {
            self.output_trim
        }
    }

    /// Mix `input_levels` (indexed by channel strip ID) to a stereo pair `(left, right)`.
    ///
    /// Channels not configured in this mix contribute nothing.
    #[must_use]
    pub fn mix_stereo(&self, input_levels: &HashMap<u32, f32>) -> (f32, f32) {
        let trim = self.effective_trim();
        let mut left = 0.0f32;
        let mut right = 0.0f32;

        for (&ch_id, &level) in input_levels {
            if let Some(send) = self.sends.get(&ch_id) {
                left += level * send.left_gain();
                right += level * send.right_gain();
            }
        }

        (left * trim, right * trim)
    }

    /// Number of configured sends in this mix.
    #[must_use]
    pub fn send_count(&self) -> usize {
        self.sends.len()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// MonitorMixManager
// ────────────────────────────────────────────────────────────────────────────

/// Manages a collection of independent monitor mixes.
#[derive(Debug, Default)]
pub struct MonitorMixManager {
    mixes: HashMap<MonitorMixId, MonitorMix>,
    next_id: u32,
}

impl MonitorMixManager {
    /// Create an empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new monitor mix and return its ID.
    pub fn create_mix(&mut self, name: impl Into<String>) -> MonitorMixId {
        let id = MonitorMixId(self.next_id);
        self.next_id += 1;
        self.mixes.insert(id, MonitorMix::new(id, name));
        id
    }

    /// Remove a monitor mix.
    pub fn remove_mix(&mut self, id: MonitorMixId) -> Option<MonitorMix> {
        self.mixes.remove(&id)
    }

    /// Get an immutable reference to a mix.
    #[must_use]
    pub fn get_mix(&self, id: MonitorMixId) -> Option<&MonitorMix> {
        self.mixes.get(&id)
    }

    /// Get a mutable reference to a mix.
    pub fn get_mix_mut(&mut self, id: MonitorMixId) -> Option<&mut MonitorMix> {
        self.mixes.get_mut(&id)
    }

    /// Number of monitor mixes.
    #[must_use]
    pub fn mix_count(&self) -> usize {
        self.mixes.len()
    }

    /// Mute all monitor mixes.
    pub fn mute_all(&mut self) {
        for mix in self.mixes.values_mut() {
            mix.muted = true;
        }
    }

    /// Un-mute all monitor mixes.
    pub fn unmute_all(&mut self) {
        for mix in self.mixes.values_mut() {
            mix.muted = false;
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_send_default() {
        let s = ChannelSend::default();
        assert!((s.level - 1.0).abs() < f32::EPSILON);
        assert!((s.pan - 0.0).abs() < f32::EPSILON);
        assert!(!s.muted);
    }

    #[test]
    fn test_channel_send_effective_level_muted() {
        let mut s = ChannelSend::unity();
        s.muted = true;
        assert!((s.effective_level()).abs() < f32::EPSILON);
    }

    #[test]
    fn test_channel_send_left_right_centre_pan() {
        let s = ChannelSend::unity();
        // At pan=0.0: left = 0.5, right = 0.5 (linear law).
        assert!((s.left_gain() - 0.5).abs() < 1e-6);
        assert!((s.right_gain() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_channel_send_full_left_pan() {
        let s = ChannelSend {
            level: 1.0,
            pan: -1.0,
            muted: false,
        };
        // pan=-1 → left=1.0, right=0.0
        assert!((s.left_gain() - 1.0).abs() < 1e-6);
        assert!((s.right_gain() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_channel_send_full_right_pan() {
        let s = ChannelSend {
            level: 1.0,
            pan: 1.0,
            muted: false,
        };
        assert!((s.left_gain() - 0.0).abs() < 1e-6);
        assert!((s.right_gain() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_monitor_mix_new_defaults() {
        let mix = MonitorMix::new(MonitorMixId(0), "Test");
        assert!((mix.output_trim - 1.0).abs() < f32::EPSILON);
        assert!(!mix.muted);
        assert_eq!(mix.send_count(), 0);
    }

    #[test]
    fn test_monitor_mix_set_and_get_send() {
        let mut mix = MonitorMix::new(MonitorMixId(0), "X");
        mix.set_send(3, ChannelSend::unity());
        assert!(mix.get_send(3).is_some());
        assert!(mix.get_send(4).is_none());
    }

    #[test]
    fn test_monitor_mix_remove_send() {
        let mut mix = MonitorMix::new(MonitorMixId(0), "X");
        mix.set_send(1, ChannelSend::unity());
        mix.remove_send(1);
        assert!(mix.get_send(1).is_none());
        assert_eq!(mix.send_count(), 0);
    }

    #[test]
    fn test_monitor_mix_set_output_trim_clamped() {
        let mut mix = MonitorMix::new(MonitorMixId(0), "X");
        mix.set_output_trim(5.0);
        assert!((mix.output_trim - 2.0).abs() < f32::EPSILON);
        mix.set_output_trim(-1.0);
        assert!((mix.output_trim - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_monitor_mix_effective_trim_when_muted() {
        let mut mix = MonitorMix::new(MonitorMixId(0), "X");
        mix.muted = true;
        assert!((mix.effective_trim() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_monitor_mix_stereo_mix() {
        let mut mix = MonitorMix::new(MonitorMixId(0), "Cue");
        // Channel 0: centre pan, level 1.0
        mix.set_send(
            0,
            ChannelSend {
                level: 1.0,
                pan: 0.0,
                muted: false,
            },
        );
        let mut inputs = HashMap::new();
        inputs.insert(0, 1.0f32);
        let (l, r) = mix.mix_stereo(&inputs);
        // left = right = 0.5 * trim(1.0)
        assert!((l - 0.5).abs() < 1e-6);
        assert!((r - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_manager_create_and_count() {
        let mut mgr = MonitorMixManager::new();
        mgr.create_mix("A");
        mgr.create_mix("B");
        assert_eq!(mgr.mix_count(), 2);
    }

    #[test]
    fn test_manager_remove_mix() {
        let mut mgr = MonitorMixManager::new();
        let id = mgr.create_mix("Temp");
        assert!(mgr.remove_mix(id).is_some());
        assert_eq!(mgr.mix_count(), 0);
    }

    #[test]
    fn test_manager_mute_and_unmute_all() {
        let mut mgr = MonitorMixManager::new();
        let id1 = mgr.create_mix("A");
        let id2 = mgr.create_mix("B");
        mgr.mute_all();
        assert!(mgr.get_mix(id1).expect("get_mix should succeed").muted);
        assert!(mgr.get_mix(id2).expect("get_mix should succeed").muted);
        mgr.unmute_all();
        assert!(!mgr.get_mix(id1).expect("get_mix should succeed").muted);
        assert!(!mgr.get_mix(id2).expect("get_mix should succeed").muted);
    }

    #[test]
    fn test_monitor_mix_id_display() {
        assert_eq!(MonitorMixId(7).to_string(), "MonitorMix(7)");
    }
}
