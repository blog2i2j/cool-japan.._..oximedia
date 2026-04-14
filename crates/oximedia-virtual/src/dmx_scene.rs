//! DMX lighting scene preset and cue-list management.
//!
//! This module provides:
//! - Per-channel DMX snapshots (512 channels / universe)
//! - Named scene presets with per-channel target values
//! - Crossfade engine: linear interpolation between current and target scene
//! - Cue lists with sequential playback and follow-timing
//! - Snapshot capture from a live fixture state
//!
//! DMX channel values are `u8` (0–255). Fractional fade values are kept
//! internally as `f64` and rounded on output.

use std::collections::HashMap;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of DMX channels in a single universe.
pub const DMX_CHANNELS: usize = 512;

// ---------------------------------------------------------------------------
// DmxSnapshot
// ---------------------------------------------------------------------------

/// A 512-channel DMX universe snapshot.
///
/// Channels are stored in a `Vec<u8>` of length [`DMX_CHANNELS`] (512).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DmxSnapshot {
    /// Channel values, indexed 0–511 (DMX addresses 1–512).
    pub channels: Vec<u8>,
    /// Optional universe identifier.
    pub universe: u16,
}

impl DmxSnapshot {
    /// Create an all-zero (blackout) snapshot for the given universe.
    #[must_use]
    pub fn blackout(universe: u16) -> Self {
        Self {
            channels: vec![0u8; DMX_CHANNELS],
            universe,
        }
    }

    /// Create a full-on (all 255) snapshot.
    #[must_use]
    pub fn full_on(universe: u16) -> Self {
        Self {
            channels: vec![255u8; DMX_CHANNELS],
            universe,
        }
    }

    /// Get a channel value (0-based index, saturates to 511).
    #[must_use]
    pub fn get(&self, ch: usize) -> u8 {
        self.channels.get(ch.min(DMX_CHANNELS - 1)).copied().unwrap_or(0)
    }

    /// Set a channel value (0-based index, silently clamps index).
    pub fn set(&mut self, ch: usize, value: u8) {
        let idx = ch.min(DMX_CHANNELS - 1);
        if let Some(slot) = self.channels.get_mut(idx) {
            *slot = value;
        }
    }

    /// Linear interpolation towards `target` at proportion `t ∈ [0, 1]`.
    ///
    /// Returns a new snapshot with interpolated channel values.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn lerp(&self, target: &Self, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0);
        let mut out = Self::blackout(self.universe);
        let len = self.channels.len().min(target.channels.len()).min(DMX_CHANNELS);
        for i in 0..len {
            let a = self.channels[i] as f64;
            let b = target.channels[i] as f64;
            out.channels[i] = (a + (b - a) * t).round() as u8;
        }
        out
    }

    /// Compute the maximum absolute channel difference to another snapshot.
    #[must_use]
    pub fn max_diff(&self, other: &Self) -> u8 {
        self.channels
            .iter()
            .zip(other.channels.iter())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0)
    }
}

impl Default for DmxSnapshot {
    fn default() -> Self {
        Self::blackout(0)
    }
}

// ---------------------------------------------------------------------------
// ScenePreset
// ---------------------------------------------------------------------------

/// A named DMX scene preset: a human-authored set of channel values.
///
/// Channels not explicitly set default to 0.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScenePreset {
    /// Unique preset name.
    pub name: String,
    /// Target universe.
    pub universe: u16,
    /// Per-channel overrides (channel index → value).
    pub channel_values: HashMap<usize, u8>,
    /// Default fade-in duration.
    pub default_fade_in: Duration,
    /// Default fade-out duration.
    pub default_fade_out: Duration,
}

impl ScenePreset {
    /// Create a new preset with no channel values.
    #[must_use]
    pub fn new(name: impl Into<String>, universe: u16) -> Self {
        Self {
            name: name.into(),
            universe,
            channel_values: HashMap::new(),
            default_fade_in: Duration::from_millis(500),
            default_fade_out: Duration::from_millis(500),
        }
    }

    /// Builder: set fade durations.
    #[must_use]
    pub fn with_fade(mut self, fade_in: Duration, fade_out: Duration) -> Self {
        self.default_fade_in = fade_in;
        self.default_fade_out = fade_out;
        self
    }

    /// Set a channel value.
    pub fn set_channel(&mut self, ch: usize, value: u8) {
        self.channel_values.insert(ch, value);
    }

    /// Get a channel value (defaults to 0 for unset channels).
    #[must_use]
    pub fn get_channel(&self, ch: usize) -> u8 {
        self.channel_values.get(&ch).copied().unwrap_or(0)
    }

    /// Convert to a full [`DmxSnapshot`].
    #[must_use]
    pub fn to_snapshot(&self) -> DmxSnapshot {
        let mut snap = DmxSnapshot::blackout(self.universe);
        for (&ch, &val) in &self.channel_values {
            if ch < DMX_CHANNELS {
                snap.channels[ch] = val;
            }
        }
        snap
    }

    /// Capture the preset from a live [`DmxSnapshot`] (copy all channels).
    pub fn capture_from_snapshot(&mut self, snap: &DmxSnapshot) {
        self.universe = snap.universe;
        for (ch, &val) in snap.channels.iter().enumerate() {
            if val > 0 {
                self.channel_values.insert(ch, val);
            } else {
                self.channel_values.remove(&ch);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Crossfader
// ---------------------------------------------------------------------------

/// State of an active crossfade.
#[derive(Debug, Clone)]
pub struct CrossfadeState {
    /// Snapshot at the start of the fade.
    pub from: DmxSnapshot,
    /// Target snapshot.
    pub to: DmxSnapshot,
    /// Total fade duration.
    pub duration: Duration,
    /// Elapsed time.
    pub elapsed: Duration,
    /// Whether the fade has completed.
    pub complete: bool,
}

impl CrossfadeState {
    /// Create a new crossfade.
    #[must_use]
    pub fn new(from: DmxSnapshot, to: DmxSnapshot, duration: Duration) -> Self {
        Self {
            from,
            to,
            duration,
            elapsed: Duration::ZERO,
            complete: false,
        }
    }

    /// Advance the fade by `dt` and return the current interpolated snapshot.
    pub fn advance(&mut self, dt: Duration) -> DmxSnapshot {
        if self.complete {
            return self.to.clone();
        }
        self.elapsed = (self.elapsed + dt).min(self.duration);
        let t = if self.duration.is_zero() {
            1.0
        } else {
            self.elapsed.as_secs_f64() / self.duration.as_secs_f64()
        };
        if t >= 1.0 {
            self.complete = true;
        }
        self.from.lerp(&self.to, t.clamp(0.0, 1.0))
    }

    /// Normalised progress `[0, 1]`.
    #[must_use]
    pub fn progress(&self) -> f64 {
        if self.duration.is_zero() {
            1.0
        } else {
            (self.elapsed.as_secs_f64() / self.duration.as_secs_f64()).clamp(0.0, 1.0)
        }
    }
}

// ---------------------------------------------------------------------------
// CueEntry
// ---------------------------------------------------------------------------

/// A single step in a cue list.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CueEntry {
    /// Cue number (for display).
    pub number: u32,
    /// Name of the [`ScenePreset`] to execute.
    pub preset_name: String,
    /// Fade-in override. Uses preset default if `None`.
    pub fade_in: Option<Duration>,
    /// Follow (auto-advance) delay after fade completes. `None` = manual.
    pub follow_delay: Option<Duration>,
}

impl CueEntry {
    /// Create a new cue entry referencing a named preset.
    #[must_use]
    pub fn new(number: u32, preset_name: impl Into<String>) -> Self {
        Self {
            number,
            preset_name: preset_name.into(),
            fade_in: None,
            follow_delay: None,
        }
    }

    /// Builder: set fade-in override.
    #[must_use]
    pub fn with_fade_in(mut self, d: Duration) -> Self {
        self.fade_in = Some(d);
        self
    }

    /// Builder: enable follow timing.
    #[must_use]
    pub fn with_follow(mut self, delay: Duration) -> Self {
        self.follow_delay = Some(delay);
        self
    }
}

// ---------------------------------------------------------------------------
// CueList
// ---------------------------------------------------------------------------

/// A sequential list of lighting cues with optional follow automation.
#[derive(Debug, Clone)]
pub struct CueList {
    /// Human-readable name.
    pub name: String,
    /// Ordered cue entries.
    entries: Vec<CueEntry>,
    /// Currently active cue index.
    active_index: Option<usize>,
    /// Active crossfade.
    active_fade: Option<CrossfadeState>,
    /// Time accumulated in the current follow delay.
    follow_elapsed: Duration,
    /// Current live snapshot (output).
    live: DmxSnapshot,
}

impl CueList {
    /// Create an empty cue list.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            entries: Vec::new(),
            active_index: None,
            active_fade: None,
            follow_elapsed: Duration::ZERO,
            live: DmxSnapshot::blackout(0),
        }
    }

    /// Append a cue.
    pub fn add_cue(&mut self, entry: CueEntry) {
        self.entries.push(entry);
    }

    /// Number of cues.
    #[must_use]
    pub fn cue_count(&self) -> usize {
        self.entries.len()
    }

    /// Current live cue index.
    #[must_use]
    pub fn active_index(&self) -> Option<usize> {
        self.active_index
    }

    /// Go to the next cue.  Returns `false` if already at the last cue.
    ///
    /// The `presets` map is consulted to look up the scene preset.
    pub fn go_next(
        &mut self,
        presets: &HashMap<String, ScenePreset>,
    ) -> bool {
        let next = match self.active_index {
            None => 0,
            Some(i) => i + 1,
        };
        if next >= self.entries.len() {
            return false;
        }
        self.go_to(next, presets);
        true
    }

    /// Go to a specific cue index, starting a crossfade.
    pub fn go_to(
        &mut self,
        index: usize,
        presets: &HashMap<String, ScenePreset>,
    ) {
        if index >= self.entries.len() {
            return;
        }
        let entry = &self.entries[index];
        let target_snap = presets
            .get(&entry.preset_name)
            .map(|p| p.to_snapshot())
            .unwrap_or_else(|| DmxSnapshot::blackout(self.live.universe));

        let fade = entry
            .fade_in
            .or_else(|| presets.get(&entry.preset_name).map(|p| p.default_fade_in))
            .unwrap_or(Duration::from_millis(500));

        self.active_fade = Some(CrossfadeState::new(self.live.clone(), target_snap, fade));
        self.active_index = Some(index);
        self.follow_elapsed = Duration::ZERO;
    }

    /// Advance the cue list by `dt`.
    ///
    /// Updates the live output snapshot and auto-advances on follow timing.
    /// Returns the current output snapshot.
    pub fn advance(
        &mut self,
        dt: Duration,
        presets: &HashMap<String, ScenePreset>,
    ) -> DmxSnapshot {
        // Advance active fade
        if let Some(fade) = &mut self.active_fade {
            self.live = fade.advance(dt);
            if fade.complete {
                self.active_fade = None;
            }
        }

        // Check follow timing (only when no fade is active)
        if self.active_fade.is_none() {
            if let Some(idx) = self.active_index {
                let entry = &self.entries[idx];
                if let Some(follow_delay) = entry.follow_delay {
                    self.follow_elapsed += dt;
                    if self.follow_elapsed >= follow_delay {
                        if self.go_next(presets) {
                            // Immediately process a zero-duration fade so the
                            // live output is updated in this same tick.
                            if let Some(fade) = &mut self.active_fade {
                                self.live = fade.advance(Duration::ZERO);
                                if fade.complete {
                                    self.active_fade = None;
                                }
                            }
                        }
                    }
                }
            }
        }

        self.live.clone()
    }

    /// Immediately snap to a preset without a fade.
    pub fn snap_to(
        &mut self,
        index: usize,
        presets: &HashMap<String, ScenePreset>,
    ) {
        if index >= self.entries.len() {
            return;
        }
        let entry = &self.entries[index];
        if let Some(preset) = presets.get(&entry.preset_name) {
            self.live = preset.to_snapshot();
        }
        self.active_index = Some(index);
        self.active_fade = None;
        self.follow_elapsed = Duration::ZERO;
    }

    /// Current live output snapshot.
    #[must_use]
    pub fn live_snapshot(&self) -> &DmxSnapshot {
        &self.live
    }
}

// ---------------------------------------------------------------------------
// SceneLibrary
// ---------------------------------------------------------------------------

/// Central registry of [`ScenePreset`]s.
#[derive(Debug, Default, Clone)]
pub struct SceneLibrary {
    presets: HashMap<String, ScenePreset>,
}

impl SceneLibrary {
    /// Create an empty library.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a preset (overwrites existing).
    pub fn store(&mut self, preset: ScenePreset) {
        self.presets.insert(preset.name.clone(), preset);
    }

    /// Get a preset by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&ScenePreset> {
        self.presets.get(name)
    }

    /// Remove a preset.
    pub fn remove(&mut self, name: &str) -> Option<ScenePreset> {
        self.presets.remove(name)
    }

    /// Number of presets.
    #[must_use]
    pub fn count(&self) -> usize {
        self.presets.len()
    }

    /// Snapshot of a named preset.
    #[must_use]
    pub fn snapshot(&self, name: &str) -> Option<DmxSnapshot> {
        self.presets.get(name).map(|p| p.to_snapshot())
    }

    /// Access the underlying presets map for passing to cue lists.
    #[must_use]
    pub fn presets(&self) -> &HashMap<String, ScenePreset> {
        &self.presets
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn make_preset(name: &str, ch0_val: u8) -> ScenePreset {
        let mut p = ScenePreset::new(name, 0);
        p.set_channel(0, ch0_val);
        p
    }

    #[test]
    fn test_snapshot_blackout_and_full() {
        let b = DmxSnapshot::blackout(0);
        assert_eq!(b.get(0), 0);
        let f = DmxSnapshot::full_on(0);
        assert_eq!(f.get(511), 255);
    }

    #[test]
    fn test_snapshot_lerp_midpoint() {
        let from = DmxSnapshot::blackout(0);
        let to = DmxSnapshot::full_on(0);
        let mid = from.lerp(&to, 0.5);
        // 0 + (255 - 0)*0.5 = 127.5 rounds to 128
        assert!(mid.get(0) >= 127 && mid.get(0) <= 128);
    }

    #[test]
    fn test_snapshot_max_diff() {
        let a = DmxSnapshot::blackout(0);
        let b = DmxSnapshot::full_on(0);
        assert_eq!(a.max_diff(&b), 255);
    }

    #[test]
    fn test_preset_to_snapshot() {
        let mut p = ScenePreset::new("wash", 0);
        p.set_channel(5, 200);
        let snap = p.to_snapshot();
        assert_eq!(snap.get(5), 200);
        assert_eq!(snap.get(0), 0);
    }

    #[test]
    fn test_preset_capture_from_snapshot() {
        let snap = DmxSnapshot::full_on(1);
        let mut p = ScenePreset::new("full", 0);
        p.capture_from_snapshot(&snap);
        assert_eq!(p.universe, 1);
        assert_eq!(p.get_channel(0), 255);
    }

    #[test]
    fn test_crossfader_advance_to_completion() {
        let from = DmxSnapshot::blackout(0);
        let to = DmxSnapshot::full_on(0);
        let mut fade = CrossfadeState::new(from, to, Duration::from_millis(100));
        let _ = fade.advance(Duration::from_millis(50));
        assert!(!fade.complete);
        let final_snap = fade.advance(Duration::from_millis(100));
        assert!(fade.complete);
        assert_eq!(final_snap.get(0), 255);
    }

    #[test]
    fn test_crossfader_zero_duration() {
        let from = DmxSnapshot::blackout(0);
        let to = DmxSnapshot::full_on(0);
        let mut fade = CrossfadeState::new(from, to, Duration::ZERO);
        let snap = fade.advance(Duration::ZERO);
        assert!(fade.complete);
        assert_eq!(snap.get(0), 255);
    }

    #[test]
    fn test_cue_list_go_next() {
        let mut lib = SceneLibrary::new();
        lib.store(make_preset("scene_a", 100));
        lib.store(make_preset("scene_b", 200));

        let mut cue_list = CueList::new("main");
        cue_list.add_cue(CueEntry::new(1, "scene_a").with_fade_in(Duration::ZERO));
        cue_list.add_cue(CueEntry::new(2, "scene_b").with_fade_in(Duration::ZERO));

        assert!(cue_list.go_next(lib.presets()));
        // advance to complete fade
        let _ = cue_list.advance(Duration::from_millis(100), lib.presets());
        assert_eq!(cue_list.active_index(), Some(0));

        assert!(cue_list.go_next(lib.presets()));
        let snap = cue_list.advance(Duration::from_millis(100), lib.presets());
        assert_eq!(cue_list.active_index(), Some(1));
        assert_eq!(snap.get(0), 200);
    }

    #[test]
    fn test_cue_list_snap_to() {
        let mut lib = SceneLibrary::new();
        lib.store(make_preset("scene_a", 77));

        let mut cue_list = CueList::new("main");
        cue_list.add_cue(CueEntry::new(1, "scene_a"));
        cue_list.snap_to(0, lib.presets());
        assert_eq!(cue_list.live_snapshot().get(0), 77);
    }

    #[test]
    fn test_cue_list_follow_timing() {
        let mut lib = SceneLibrary::new();
        lib.store(make_preset("a", 50));
        lib.store(make_preset("b", 150));

        let mut cue_list = CueList::new("follow");
        cue_list.add_cue(
            CueEntry::new(1, "a")
                .with_fade_in(Duration::ZERO)
                .with_follow(Duration::from_millis(50)),
        );
        cue_list.add_cue(CueEntry::new(2, "b").with_fade_in(Duration::ZERO));

        cue_list.go_to(0, lib.presets());
        // first advance completes the instant fade and starts follow timer
        let _ = cue_list.advance(Duration::from_millis(10), lib.presets());
        assert_eq!(cue_list.active_index(), Some(0));
        // advance past follow delay → auto-advance to cue 1
        let snap = cue_list.advance(Duration::from_millis(100), lib.presets());
        assert_eq!(cue_list.active_index(), Some(1));
        assert_eq!(snap.get(0), 150);
    }

    #[test]
    fn test_scene_library_store_and_retrieve() {
        let mut lib = SceneLibrary::new();
        lib.store(make_preset("night", 10));
        assert!(lib.get("night").is_some());
        assert_eq!(lib.count(), 1);
        let snap = lib.snapshot("night").expect("snap");
        assert_eq!(snap.get(0), 10);
        lib.remove("night");
        assert_eq!(lib.count(), 0);
    }
}
