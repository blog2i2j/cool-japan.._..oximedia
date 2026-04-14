//! Mix scene management — capture, recall, diff, and timed transitions.
//!
//! A **mix scene** (also called a *snapshot* or *memory*) captures the complete
//! parameter state of every channel and bus at a moment in time.  Scenes can
//! be recalled instantly or blended towards over a configurable number of
//! audio samples (a *timed transition*).
//!
//! # Usage
//!
//! ```rust
//! use oximedia_mixer::mix_scene::{MixScene, SceneLibrary, SceneTransition};
//!
//! let mut library = SceneLibrary::new();
//!
//! // Build a scene from arbitrary channel/bus state.
//! let mut scene = MixScene::new("Act 1");
//! scene.set_channel_fader(0, 0.8).unwrap();
//! scene.set_channel_pan(0, -0.3).unwrap();
//! let scene_id = library.store(scene);
//!
//! // Start a 2-second transition (at 48 000 Hz = 96 000 samples).
//! let current = MixScene::new("Current");
//! let transition = library.begin_transition(scene_id, &current, 96_000).unwrap();
//! assert!(!transition.is_complete());
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by mix-scene operations.
#[derive(Debug, thiserror::Error)]
pub enum SceneError {
    /// Scene identifier not found.
    #[error("Scene not found: {0}")]
    NotFound(SceneId),

    /// Transition target does not exist.
    #[error("Transition target scene not found: {0}")]
    TransitionTargetNotFound(SceneId),

    /// Parameter value out of range.
    #[error("Parameter '{name}' value {value} out of range [{min}, {max}]")]
    ParameterOutOfRange {
        /// Parameter name.
        name: String,
        /// Attempted value.
        value: f32,
        /// Minimum value.
        min: f32,
        /// Maximum value.
        max: f32,
    },
}

/// Result alias.
pub type SceneResult<T> = Result<T, SceneError>;

// ---------------------------------------------------------------------------
// Identifier
// ---------------------------------------------------------------------------

/// Opaque scene identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SceneId(pub u64);

impl std::fmt::Display for SceneId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Scene#{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Per-channel / per-bus state
// ---------------------------------------------------------------------------

/// State of one channel within a scene.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelSceneState {
    /// Fader level (0.0 = silence, 1.0 = unity, >1.0 = boost).
    pub fader: f32,
    /// Pan position (-1.0 = hard-left, 0.0 = centre, +1.0 = hard-right).
    pub pan: f32,
    /// Input gain in dB.
    pub input_gain_db: f32,
    /// Mute flag.
    pub muted: bool,
    /// Solo flag.
    pub solo: bool,
    /// Send levels keyed by bus index.
    pub sends: HashMap<u32, f32>,
}

impl Default for ChannelSceneState {
    fn default() -> Self {
        Self {
            fader: 1.0,
            pan: 0.0,
            input_gain_db: 0.0,
            muted: false,
            solo: false,
            sends: HashMap::new(),
        }
    }
}

impl ChannelSceneState {
    /// Linearly interpolate towards `other` by factor `t` (0.0 = self, 1.0 = other).
    ///
    /// Boolean flags snap to `other` when `t >= 0.5`.
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        // Interpolate send levels — union of both key-sets.
        let mut sends = HashMap::new();
        for (&bus, &val) in &self.sends {
            let other_val = other.sends.get(&bus).copied().unwrap_or(val);
            sends.insert(bus, lerp_f32(val, other_val, t));
        }
        for (&bus, &val) in &other.sends {
            sends.entry(bus).or_insert_with(|| lerp_f32(0.0, val, t));
        }
        Self {
            fader: lerp_f32(self.fader, other.fader, t),
            pan: lerp_f32(self.pan, other.pan, t),
            input_gain_db: lerp_f32(self.input_gain_db, other.input_gain_db, t),
            muted: if t >= 0.5 { other.muted } else { self.muted },
            solo: if t >= 0.5 { other.solo } else { self.solo },
            sends,
        }
    }
}

/// State of one bus within a scene.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BusSceneState {
    /// Bus fader level.
    pub fader: f32,
    /// Mute flag.
    pub muted: bool,
    /// Solo flag.
    pub solo: bool,
}

impl Default for BusSceneState {
    fn default() -> Self {
        Self {
            fader: 1.0,
            muted: false,
            solo: false,
        }
    }
}

impl BusSceneState {
    /// Linearly interpolate towards `other` by factor `t`.
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            fader: lerp_f32(self.fader, other.fader, t),
            muted: if t >= 0.5 { other.muted } else { self.muted },
            solo: if t >= 0.5 { other.solo } else { self.solo },
        }
    }
}

// ---------------------------------------------------------------------------
// SceneDiff
// ---------------------------------------------------------------------------

/// A per-channel description of what changed between two scenes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelDiff {
    /// Channel index.
    pub channel: u32,
    /// Fader delta (target − source).
    pub fader_delta: f32,
    /// Pan delta.
    pub pan_delta: f32,
    /// Input gain delta (dB).
    pub input_gain_db_delta: f32,
    /// Mute changed.
    pub mute_changed: bool,
    /// Solo changed.
    pub solo_changed: bool,
}

/// A per-bus description of what changed between two scenes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BusDiff {
    /// Bus index.
    pub bus: u32,
    /// Fader delta.
    pub fader_delta: f32,
    /// Mute changed.
    pub mute_changed: bool,
}

/// The complete diff between two scenes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneDiff {
    /// Per-channel differences.
    pub channels: Vec<ChannelDiff>,
    /// Per-bus differences.
    pub buses: Vec<BusDiff>,
}

impl SceneDiff {
    /// Returns `true` if the diff contains no meaningful changes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty() && self.buses.is_empty()
    }
}

// ---------------------------------------------------------------------------
// MixScene
// ---------------------------------------------------------------------------

/// A complete point-in-time snapshot of the mixer's parameter state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixScene {
    /// Unique identifier (assigned by [`SceneLibrary`]).
    pub id: SceneId,
    /// Human-readable name.
    pub name: String,
    /// Optional description / notes.
    pub notes: String,
    /// Creation timestamp in milliseconds since Unix epoch.
    pub created_at_ms: u64,
    /// Per-channel state keyed by channel index.
    pub channels: HashMap<u32, ChannelSceneState>,
    /// Per-bus state keyed by bus index.
    pub buses: HashMap<u32, BusSceneState>,
    /// Master bus fader level.
    pub master_fader: f32,
    /// Master mute.
    pub master_muted: bool,
}

impl MixScene {
    /// Create an empty scene with the given name.
    ///
    /// The `id` field is set to `SceneId(0)` and will be overwritten by
    /// [`SceneLibrary::store`].
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: SceneId(0),
            name: name.into(),
            notes: String::new(),
            created_at_ms: 0,
            channels: HashMap::new(),
            buses: HashMap::new(),
            master_fader: 1.0,
            master_muted: false,
        }
    }

    // ------------------------------------------------------------------
    // Setters
    // ------------------------------------------------------------------

    /// Set a channel's fader level.
    pub fn set_channel_fader(&mut self, channel: u32, fader: f32) -> SceneResult<()> {
        if !(0.0..=2.0).contains(&fader) {
            return Err(SceneError::ParameterOutOfRange {
                name: "channel_fader".to_string(),
                value: fader,
                min: 0.0,
                max: 2.0,
            });
        }
        self.channels.entry(channel).or_default().fader = fader;
        Ok(())
    }

    /// Set a channel's pan position.
    pub fn set_channel_pan(&mut self, channel: u32, pan: f32) -> SceneResult<()> {
        if !(-1.0..=1.0).contains(&pan) {
            return Err(SceneError::ParameterOutOfRange {
                name: "channel_pan".to_string(),
                value: pan,
                min: -1.0,
                max: 1.0,
            });
        }
        self.channels.entry(channel).or_default().pan = pan;
        Ok(())
    }

    /// Set channel mute state.
    pub fn set_channel_mute(&mut self, channel: u32, muted: bool) {
        self.channels.entry(channel).or_default().muted = muted;
    }

    /// Set channel input gain in dB.
    pub fn set_channel_input_gain_db(&mut self, channel: u32, db: f32) -> SceneResult<()> {
        if !(-96.0..=24.0).contains(&db) {
            return Err(SceneError::ParameterOutOfRange {
                name: "input_gain_db".to_string(),
                value: db,
                min: -96.0,
                max: 24.0,
            });
        }
        self.channels.entry(channel).or_default().input_gain_db = db;
        Ok(())
    }

    /// Set a send level for a channel to a particular bus.
    pub fn set_channel_send(&mut self, channel: u32, bus: u32, level: f32) -> SceneResult<()> {
        if !(0.0..=2.0).contains(&level) {
            return Err(SceneError::ParameterOutOfRange {
                name: "send_level".to_string(),
                value: level,
                min: 0.0,
                max: 2.0,
            });
        }
        self.channels
            .entry(channel)
            .or_default()
            .sends
            .insert(bus, level);
        Ok(())
    }

    /// Set bus fader level.
    pub fn set_bus_fader(&mut self, bus: u32, fader: f32) -> SceneResult<()> {
        if !(0.0..=2.0).contains(&fader) {
            return Err(SceneError::ParameterOutOfRange {
                name: "bus_fader".to_string(),
                value: fader,
                min: 0.0,
                max: 2.0,
            });
        }
        self.buses.entry(bus).or_default().fader = fader;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Diff
    // ------------------------------------------------------------------

    /// Compute the diff from `self` (source) to `other` (target).
    #[must_use]
    pub fn diff(&self, other: &Self) -> SceneDiff {
        const EPS: f32 = 1e-6;
        let mut channels = Vec::new();
        // Union of channel indices.
        let mut all_channels: Vec<u32> = self.channels.keys().chain(other.channels.keys()).copied().collect();
        all_channels.sort_unstable();
        all_channels.dedup();

        for ch in all_channels {
            let src = self.channels.get(&ch).cloned().unwrap_or_default();
            let dst = other.channels.get(&ch).cloned().unwrap_or_default();
            let fader_delta = dst.fader - src.fader;
            let pan_delta = dst.pan - src.pan;
            let gain_delta = dst.input_gain_db - src.input_gain_db;
            let mute_changed = dst.muted != src.muted;
            let solo_changed = dst.solo != src.solo;
            if fader_delta.abs() > EPS
                || pan_delta.abs() > EPS
                || gain_delta.abs() > EPS
                || mute_changed
                || solo_changed
            {
                channels.push(ChannelDiff {
                    channel: ch,
                    fader_delta,
                    pan_delta,
                    input_gain_db_delta: gain_delta,
                    mute_changed,
                    solo_changed,
                });
            }
        }

        let mut buses = Vec::new();
        let mut all_buses: Vec<u32> = self.buses.keys().chain(other.buses.keys()).copied().collect();
        all_buses.sort_unstable();
        all_buses.dedup();
        for b in all_buses {
            let src = self.buses.get(&b).cloned().unwrap_or_default();
            let dst = other.buses.get(&b).cloned().unwrap_or_default();
            let fader_delta = dst.fader - src.fader;
            let mute_changed = dst.muted != src.muted;
            if fader_delta.abs() > EPS || mute_changed {
                buses.push(BusDiff {
                    bus: b,
                    fader_delta,
                    mute_changed,
                });
            }
        }

        SceneDiff { channels, buses }
    }

    /// Linearly interpolate from `self` towards `other` by `t` (0.0..=1.0).
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);

        // Union channels
        let mut channels: HashMap<u32, ChannelSceneState> = HashMap::new();
        for (&ch, state) in &self.channels {
            let other_state = other.channels.get(&ch).cloned().unwrap_or_default();
            channels.insert(ch, state.lerp(&other_state, t));
        }
        for (&ch, state) in &other.channels {
            channels.entry(ch).or_insert_with(|| {
                let default = ChannelSceneState::default();
                default.lerp(state, t)
            });
        }

        // Union buses
        let mut buses: HashMap<u32, BusSceneState> = HashMap::new();
        for (&b, state) in &self.buses {
            let other_state = other.buses.get(&b).cloned().unwrap_or_default();
            buses.insert(b, state.lerp(&other_state, t));
        }
        for (&b, state) in &other.buses {
            buses.entry(b).or_insert_with(|| {
                let default = BusSceneState::default();
                default.lerp(state, t)
            });
        }

        Self {
            id: other.id,
            name: other.name.clone(),
            notes: other.notes.clone(),
            created_at_ms: other.created_at_ms,
            channels,
            buses,
            master_fader: lerp_f32(self.master_fader, other.master_fader, t),
            master_muted: if t >= 0.5 {
                other.master_muted
            } else {
                self.master_muted
            },
        }
    }
}

// ---------------------------------------------------------------------------
// SceneTransition
// ---------------------------------------------------------------------------

/// An in-progress timed transition from a source scene towards a target scene.
pub struct SceneTransition {
    source: MixScene,
    target: MixScene,
    total_samples: u64,
    elapsed_samples: u64,
}

impl SceneTransition {
    /// Create a new transition.
    #[must_use]
    pub fn new(source: MixScene, target: MixScene, total_samples: u64) -> Self {
        Self {
            source,
            target,
            total_samples: total_samples.max(1),
            elapsed_samples: 0,
        }
    }

    /// Advance the transition by `samples` samples.
    ///
    /// Returns the interpolated [`MixScene`] at the new position.
    pub fn advance(&mut self, samples: u64) -> MixScene {
        self.elapsed_samples =
            (self.elapsed_samples + samples).min(self.total_samples);
        let t = self.elapsed_samples as f32 / self.total_samples as f32;
        self.source.lerp(&self.target, t)
    }

    /// Returns `true` when the transition has fully reached the target.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.elapsed_samples >= self.total_samples
    }

    /// Progress fraction in the range 0.0..=1.0.
    #[must_use]
    pub fn progress(&self) -> f32 {
        self.elapsed_samples as f32 / self.total_samples as f32
    }

    /// Return the current interpolated scene without advancing.
    #[must_use]
    pub fn current_scene(&self) -> MixScene {
        let t = self.progress();
        self.source.lerp(&self.target, t)
    }

    /// Reference to the target scene.
    #[must_use]
    pub fn target(&self) -> &MixScene {
        &self.target
    }
}

// ---------------------------------------------------------------------------
// SceneLibrary
// ---------------------------------------------------------------------------

/// Ordered library of named mix scenes with undo history.
pub struct SceneLibrary {
    scenes: HashMap<SceneId, MixScene>,
    order: Vec<SceneId>,
    next_id: u64,
    undo_stack: Vec<Vec<(SceneId, MixScene)>>,
    redo_stack: Vec<Vec<(SceneId, MixScene)>>,
}

impl SceneLibrary {
    /// Create an empty library.
    #[must_use]
    pub fn new() -> Self {
        Self {
            scenes: HashMap::new(),
            order: Vec::new(),
            next_id: 1,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
        }
    }

    /// Store a scene in the library, assigning it a unique [`SceneId`].
    ///
    /// Returns the new identifier.
    pub fn store(&mut self, mut scene: MixScene) -> SceneId {
        let id = SceneId(self.next_id);
        self.next_id += 1;
        scene.id = id;
        self.push_undo();
        self.scenes.insert(id, scene);
        self.order.push(id);
        id
    }

    /// Get an immutable reference to a scene.
    pub fn get(&self, id: SceneId) -> SceneResult<&MixScene> {
        self.scenes.get(&id).ok_or(SceneError::NotFound(id))
    }

    /// Get a mutable reference to a scene.
    pub fn get_mut(&mut self, id: SceneId) -> SceneResult<&mut MixScene> {
        self.scenes.get_mut(&id).ok_or(SceneError::NotFound(id))
    }

    /// Delete a scene from the library.
    pub fn delete(&mut self, id: SceneId) -> SceneResult<MixScene> {
        let scene = self.scenes.remove(&id).ok_or(SceneError::NotFound(id))?;
        self.order.retain(|&s| s != id);
        Ok(scene)
    }

    /// Return scene identifiers in insertion order.
    #[must_use]
    pub fn ids(&self) -> &[SceneId] {
        &self.order
    }

    /// Total number of stored scenes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.scenes.len()
    }

    /// Returns `true` if the library contains no scenes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.scenes.is_empty()
    }

    /// Begin a timed transition towards the scene identified by `target_id`.
    ///
    /// `transition_samples` is the number of audio samples the transition spans.
    pub fn begin_transition(
        &self,
        target_id: SceneId,
        source_scene: &MixScene,
        transition_samples: u64,
    ) -> SceneResult<SceneTransition> {
        let target = self
            .scenes
            .get(&target_id)
            .ok_or(SceneError::TransitionTargetNotFound(target_id))?
            .clone();
        Ok(SceneTransition::new(
            source_scene.clone(),
            target,
            transition_samples,
        ))
    }

    /// Compute the diff between two stored scenes.
    pub fn diff(&self, from: SceneId, to: SceneId) -> SceneResult<SceneDiff> {
        let src = self.get(from)?;
        let dst = self.get(to)?;
        Ok(src.diff(dst))
    }

    // ------------------------------------------------------------------
    // Undo / redo
    // ------------------------------------------------------------------

    fn push_undo(&mut self) {
        let snapshot: Vec<(SceneId, MixScene)> =
            self.scenes.iter().map(|(&id, s)| (id, s.clone())).collect();
        self.undo_stack.push(snapshot);
        self.redo_stack.clear();
        // Limit undo depth.
        if self.undo_stack.len() > 64 {
            self.undo_stack.remove(0);
        }
    }

    /// Undo the last modifying operation.  Returns `false` if nothing to undo.
    pub fn undo(&mut self) -> bool {
        if let Some(snapshot) = self.undo_stack.pop() {
            // Save current state to redo stack.
            let current: Vec<(SceneId, MixScene)> =
                self.scenes.iter().map(|(&id, s)| (id, s.clone())).collect();
            self.redo_stack.push(current);
            self.scenes = snapshot.into_iter().collect();
            self.order = {
                let mut ids: Vec<SceneId> = self.scenes.keys().copied().collect();
                ids.sort();
                ids
            };
            true
        } else {
            false
        }
    }

    /// Redo a previously undone operation.  Returns `false` if nothing to redo.
    pub fn redo(&mut self) -> bool {
        if let Some(snapshot) = self.redo_stack.pop() {
            let current: Vec<(SceneId, MixScene)> =
                self.scenes.iter().map(|(&id, s)| (id, s.clone())).collect();
            self.undo_stack.push(current);
            self.scenes = snapshot.into_iter().collect();
            self.order = {
                let mut ids: Vec<SceneId> = self.scenes.keys().copied().collect();
                ids.sort();
                ids
            };
            true
        } else {
            false
        }
    }
}

impl Default for SceneLibrary {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_store_and_retrieve() {
        let mut lib = SceneLibrary::new();
        let mut scene = MixScene::new("Test Scene");
        scene.set_channel_fader(0, 0.7).unwrap();
        let id = lib.store(scene);
        let retrieved = lib.get(id).unwrap();
        assert!(approx_eq(
            retrieved.channels[&0].fader,
            0.7,
            1e-5
        ));
    }

    #[test]
    fn test_scene_diff_detects_fader_change() {
        let mut lib = SceneLibrary::new();
        let mut a = MixScene::new("A");
        a.set_channel_fader(0, 0.5).unwrap();
        let mut b = MixScene::new("B");
        b.set_channel_fader(0, 1.0).unwrap();
        let id_a = lib.store(a);
        let id_b = lib.store(b);
        let diff = lib.diff(id_a, id_b).unwrap();
        assert!(!diff.is_empty());
        assert!(approx_eq(diff.channels[0].fader_delta, 0.5, 1e-5));
    }

    #[test]
    fn test_lerp_midpoint() {
        let mut a = MixScene::new("A");
        a.set_channel_fader(0, 0.0).unwrap();
        let mut b = MixScene::new("B");
        b.set_channel_fader(0, 1.0).unwrap();
        let mid = a.lerp(&b, 0.5);
        assert!(approx_eq(mid.channels[&0].fader, 0.5, 1e-5));
    }

    #[test]
    fn test_transition_completes() {
        let lib = SceneLibrary::new();
        let source = MixScene::new("Source");
        let mut target = MixScene::new("Target");
        target.master_fader = 0.5;
        // Manually create a transition (no library required).
        let mut tr = SceneTransition::new(source, target, 100);
        assert!(!tr.is_complete());
        let _ = tr.advance(100);
        assert!(tr.is_complete());
        assert!(approx_eq(tr.progress(), 1.0, 1e-5));
    }

    #[test]
    fn test_transition_advance_interpolates() {
        let mut source = MixScene::new("Source");
        source.set_channel_fader(1, 0.0).unwrap();
        let mut target = MixScene::new("Target");
        target.set_channel_fader(1, 1.0).unwrap();
        let mut tr = SceneTransition::new(source, target, 1000);
        let mid = tr.advance(500);
        assert!(approx_eq(
            mid.channels.get(&1).map(|s| s.fader).unwrap_or(1.0),
            0.5,
            0.01
        ));
    }

    #[test]
    fn test_delete_scene() {
        let mut lib = SceneLibrary::new();
        let id = lib.store(MixScene::new("Delete Me"));
        assert_eq!(lib.len(), 1);
        lib.delete(id).unwrap();
        assert!(lib.is_empty());
        assert!(lib.get(id).is_err());
    }

    #[test]
    fn test_fader_out_of_range_rejected() {
        let mut scene = MixScene::new("Validation");
        assert!(scene.set_channel_fader(0, 3.0).is_err());
        assert!(scene.set_channel_fader(0, -1.0).is_err());
        assert!(scene.set_channel_fader(0, 1.0).is_ok());
    }

    #[test]
    fn test_pan_out_of_range_rejected() {
        let mut scene = MixScene::new("Pan");
        assert!(scene.set_channel_pan(0, 1.5).is_err());
        assert!(scene.set_channel_pan(0, -0.5).is_ok());
    }

    #[test]
    fn test_undo_redo() {
        let mut lib = SceneLibrary::new();
        let id = lib.store(MixScene::new("Scene 1"));
        lib.get_mut(id).unwrap().name = "Modified".to_string();
        // Store second scene (pushes undo).
        lib.store(MixScene::new("Scene 2"));
        assert_eq!(lib.len(), 2);
        // Undo should go back to 1 scene.
        assert!(lib.undo());
        assert_eq!(lib.len(), 1);
        // Redo should restore 2 scenes.
        assert!(lib.redo());
        assert_eq!(lib.len(), 2);
    }

    #[test]
    fn test_begin_transition_missing_target() {
        let lib = SceneLibrary::new();
        let dummy = MixScene::new("Dummy");
        assert!(lib.begin_transition(SceneId(999), &dummy, 100).is_err());
    }
}
