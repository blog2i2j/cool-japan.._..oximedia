#![allow(dead_code)]
//! Scene save/recall system for mixer states.
//!
//! Provides the ability to store and recall complete mixer configurations as
//! named scenes. Supports scene interpolation (crossfading between scenes),
//! scene libraries, and undo/redo for scene operations. This is essential
//! for live broadcast and theater mixing workflows.

use std::collections::HashMap;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

/// Maximum number of scenes in a library.
const MAX_SCENES: usize = 1000;

/// Unique identifier for a scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SceneId(pub u64);

/// A single channel's state snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelSnapshot {
    /// Channel name.
    pub name: String,
    /// Channel index.
    pub index: usize,
    /// Fader level (0.0..1.0).
    pub fader: f32,
    /// Pan position (-1.0..1.0).
    pub pan: f32,
    /// Mute state.
    pub mute: bool,
    /// Solo state.
    pub solo: bool,
    /// Input gain in dB.
    pub input_gain_db: f32,
    /// EQ enabled.
    pub eq_enabled: bool,
    /// Dynamics enabled.
    pub dynamics_enabled: bool,
}

impl Default for ChannelSnapshot {
    fn default() -> Self {
        Self {
            name: String::new(),
            index: 0,
            fader: 1.0,
            pan: 0.0,
            mute: false,
            solo: false,
            input_gain_db: 0.0,
            eq_enabled: true,
            dynamics_enabled: false,
        }
    }
}

/// What aspects of the mixer to include in a scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SceneScope {
    /// All channel parameters.
    All,
    /// Fader levels only.
    FadersOnly,
    /// Mute/solo states only.
    MuteSoloOnly,
    /// EQ settings only.
    EqOnly,
    /// Dynamics settings only.
    DynamicsOnly,
    /// Pan settings only.
    PanOnly,
}

/// A complete scene snapshot of the mixer state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    /// Unique scene identifier.
    pub id: SceneId,
    /// Scene name.
    pub name: String,
    /// Optional scene description/notes.
    pub description: String,
    /// Scene scope (which parameters are stored).
    pub scope: SceneScope,
    /// Channel snapshots.
    pub channels: Vec<ChannelSnapshot>,
    /// Creation timestamp (seconds since epoch).
    pub created_at: u64,
    /// Last modified timestamp.
    pub modified_at: u64,
    /// Scene tags for organization.
    pub tags: Vec<String>,
}

impl Scene {
    /// Creates a new scene with the given name.
    #[must_use]
    pub fn new(id: SceneId, name: &str) -> Self {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            id,
            name: name.to_string(),
            description: String::new(),
            scope: SceneScope::All,
            channels: Vec::new(),
            created_at: now,
            modified_at: now,
            tags: Vec::new(),
        }
    }

    /// Adds a channel snapshot.
    pub fn add_channel(&mut self, snapshot: ChannelSnapshot) {
        self.channels.push(snapshot);
    }

    /// Gets a channel snapshot by index.
    #[must_use]
    pub fn get_channel(&self, index: usize) -> Option<&ChannelSnapshot> {
        self.channels.iter().find(|c| c.index == index)
    }

    /// Returns the number of channel snapshots.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Adds a tag.
    pub fn add_tag(&mut self, tag: &str) {
        if !self.tags.contains(&tag.to_string()) {
            self.tags.push(tag.to_string());
        }
    }

    /// Checks if the scene has a specific tag.
    #[must_use]
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }
}

/// Interpolation result for crossfading between two scenes.
#[derive(Debug, Clone)]
pub struct InterpolatedSnapshot {
    /// Interpolated channel states.
    pub channels: Vec<ChannelSnapshot>,
    /// Interpolation progress (0.0 = scene A, 1.0 = scene B).
    pub progress: f32,
}

/// Interpolates between two channel snapshots.
#[must_use]
fn interpolate_channel(a: &ChannelSnapshot, b: &ChannelSnapshot, t: f32) -> ChannelSnapshot {
    ChannelSnapshot {
        name: if t < 0.5 {
            a.name.clone()
        } else {
            b.name.clone()
        },
        index: a.index,
        fader: a.fader + (b.fader - a.fader) * t,
        pan: a.pan + (b.pan - a.pan) * t,
        mute: if t < 0.5 { a.mute } else { b.mute },
        solo: if t < 0.5 { a.solo } else { b.solo },
        input_gain_db: a.input_gain_db + (b.input_gain_db - a.input_gain_db) * t,
        eq_enabled: if t < 0.5 { a.eq_enabled } else { b.eq_enabled },
        dynamics_enabled: if t < 0.5 {
            a.dynamics_enabled
        } else {
            b.dynamics_enabled
        },
    }
}

/// Interpolates between two scenes.
#[must_use]
pub fn interpolate_scenes(a: &Scene, b: &Scene, t: f32) -> InterpolatedSnapshot {
    let t_clamped = t.clamp(0.0, 1.0);
    let max_channels = a.channels.len().max(b.channels.len());
    let mut channels = Vec::with_capacity(max_channels);

    for i in 0..max_channels {
        let ch_a = a.channels.get(i);
        let ch_b = b.channels.get(i);
        match (ch_a, ch_b) {
            (Some(ca), Some(cb)) => channels.push(interpolate_channel(ca, cb, t_clamped)),
            (Some(ca), None) => channels.push(ca.clone()),
            (None, Some(cb)) => channels.push(cb.clone()),
            (None, None) => {}
        }
    }

    InterpolatedSnapshot {
        channels,
        progress: t_clamped,
    }
}

/// A library of scenes with recall and management operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneLibrary {
    /// Stored scenes.
    scenes: HashMap<SceneId, Scene>,
    /// Scene ordering for recall.
    order: Vec<SceneId>,
    /// Next scene ID counter.
    next_id: u64,
    /// Currently active scene ID.
    active_scene: Option<SceneId>,
    /// Undo stack of scene IDs (previous active scenes).
    undo_stack: Vec<SceneId>,
    /// Redo stack of scene IDs.
    redo_stack: Vec<SceneId>,
}

impl SceneLibrary {
    /// Creates a new empty scene library.
    #[must_use]
    pub fn new() -> Self {
        Self {
            scenes: HashMap::new(),
            order: Vec::new(),
            next_id: 1,
            active_scene: None,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
        }
    }

    /// Stores a new scene and returns its ID.
    ///
    /// Returns `None` if the library is full.
    pub fn store_scene(&mut self, name: &str, channels: Vec<ChannelSnapshot>) -> Option<SceneId> {
        if self.scenes.len() >= MAX_SCENES {
            return None;
        }
        let id = SceneId(self.next_id);
        self.next_id += 1;
        let mut scene = Scene::new(id, name);
        scene.channels = channels;
        self.scenes.insert(id, scene);
        self.order.push(id);
        Some(id)
    }

    /// Recalls a scene by ID.
    ///
    /// Returns the scene if found, and updates the active scene.
    pub fn recall_scene(&mut self, id: SceneId) -> Option<&Scene> {
        if self.scenes.contains_key(&id) {
            if let Some(prev) = self.active_scene {
                self.undo_stack.push(prev);
            }
            self.redo_stack.clear();
            self.active_scene = Some(id);
            self.scenes.get(&id)
        } else {
            None
        }
    }

    /// Undoes the last scene recall, returning the previous scene.
    pub fn undo(&mut self) -> Option<&Scene> {
        let prev_id = self.undo_stack.pop()?;
        if let Some(current) = self.active_scene {
            self.redo_stack.push(current);
        }
        self.active_scene = Some(prev_id);
        self.scenes.get(&prev_id)
    }

    /// Redoes the last undone scene recall.
    pub fn redo(&mut self) -> Option<&Scene> {
        let next_id = self.redo_stack.pop()?;
        if let Some(current) = self.active_scene {
            self.undo_stack.push(current);
        }
        self.active_scene = Some(next_id);
        self.scenes.get(&next_id)
    }

    /// Deletes a scene by ID.
    pub fn delete_scene(&mut self, id: SceneId) -> bool {
        if self.scenes.remove(&id).is_some() {
            self.order.retain(|sid| *sid != id);
            if self.active_scene == Some(id) {
                self.active_scene = None;
            }
            true
        } else {
            false
        }
    }

    /// Renames a scene.
    pub fn rename_scene(&mut self, id: SceneId, new_name: &str) -> bool {
        if let Some(scene) = self.scenes.get_mut(&id) {
            scene.name = new_name.to_string();
            true
        } else {
            false
        }
    }

    /// Gets a scene by ID.
    #[must_use]
    pub fn get_scene(&self, id: SceneId) -> Option<&Scene> {
        self.scenes.get(&id)
    }

    /// Gets the currently active scene.
    #[must_use]
    pub fn active_scene(&self) -> Option<&Scene> {
        self.active_scene.and_then(|id| self.scenes.get(&id))
    }

    /// Gets the number of stored scenes.
    #[must_use]
    pub fn scene_count(&self) -> usize {
        self.scenes.len()
    }

    /// Gets the scene order (list of scene IDs).
    #[must_use]
    pub fn scene_order(&self) -> &[SceneId] {
        &self.order
    }

    /// Finds scenes by tag.
    #[must_use]
    pub fn find_by_tag(&self, tag: &str) -> Vec<&Scene> {
        self.scenes.values().filter(|s| s.has_tag(tag)).collect()
    }

    /// Finds scenes by name substring.
    #[must_use]
    pub fn find_by_name(&self, query: &str) -> Vec<&Scene> {
        let lower = query.to_lowercase();
        self.scenes
            .values()
            .filter(|s| s.name.to_lowercase().contains(&lower))
            .collect()
    }

    /// Checks if the undo stack has entries.
    #[must_use]
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Checks if the redo stack has entries.
    #[must_use]
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }
}

impl Default for SceneLibrary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_channels(count: usize) -> Vec<ChannelSnapshot> {
        (0..count)
            .map(|i| ChannelSnapshot {
                name: format!("Ch {i}"),
                index: i,
                fader: 0.8,
                pan: 0.0,
                ..Default::default()
            })
            .collect()
    }

    #[test]
    fn test_scene_creation() {
        let scene = Scene::new(SceneId(1), "Test Scene");
        assert_eq!(scene.name, "Test Scene");
        assert_eq!(scene.channel_count(), 0);
    }

    #[test]
    fn test_scene_add_channel() {
        let mut scene = Scene::new(SceneId(1), "Test");
        scene.add_channel(ChannelSnapshot {
            name: "Ch0".to_string(),
            index: 0,
            ..Default::default()
        });
        assert_eq!(scene.channel_count(), 1);
        assert!(scene.get_channel(0).is_some());
        assert!(scene.get_channel(1).is_none());
    }

    #[test]
    fn test_scene_tags() {
        let mut scene = Scene::new(SceneId(1), "Test");
        scene.add_tag("live");
        scene.add_tag("broadcast");
        assert!(scene.has_tag("live"));
        assert!(scene.has_tag("broadcast"));
        assert!(!scene.has_tag("studio"));
        // Duplicate tag should not be added
        scene.add_tag("live");
        assert_eq!(scene.tags.len(), 2);
    }

    #[test]
    fn test_library_creation() {
        let lib = SceneLibrary::new();
        assert_eq!(lib.scene_count(), 0);
        assert!(lib.active_scene().is_none());
    }

    #[test]
    fn test_library_store_and_recall() {
        let mut lib = SceneLibrary::new();
        let channels = make_channels(4);
        let id = lib
            .store_scene("Scene 1", channels)
            .expect("id should be valid");
        assert_eq!(lib.scene_count(), 1);

        let scene = lib.recall_scene(id).expect("scene should be valid");
        assert_eq!(scene.name, "Scene 1");
        assert_eq!(scene.channel_count(), 4);
    }

    #[test]
    fn test_library_delete() {
        let mut lib = SceneLibrary::new();
        let id = lib
            .store_scene("Scene 1", make_channels(2))
            .expect("id should be valid");
        assert_eq!(lib.scene_count(), 1);
        assert!(lib.delete_scene(id));
        assert_eq!(lib.scene_count(), 0);
        assert!(!lib.delete_scene(id)); // Already deleted
    }

    #[test]
    fn test_library_rename() {
        let mut lib = SceneLibrary::new();
        let id = lib
            .store_scene("Original", make_channels(1))
            .expect("id should be valid");
        assert!(lib.rename_scene(id, "Renamed"));
        assert_eq!(
            lib.get_scene(id).expect("get_scene should succeed").name,
            "Renamed"
        );
    }

    #[test]
    fn test_library_undo_redo() {
        let mut lib = SceneLibrary::new();
        let id1 = lib
            .store_scene("Scene 1", make_channels(2))
            .expect("id1 should be valid");
        let id2 = lib
            .store_scene("Scene 2", make_channels(2))
            .expect("id2 should be valid");

        lib.recall_scene(id1);
        lib.recall_scene(id2);

        assert!(lib.can_undo());
        let prev = lib.undo().expect("prev should be valid");
        assert_eq!(prev.name, "Scene 1");

        assert!(lib.can_redo());
        let next = lib.redo().expect("next should be valid");
        assert_eq!(next.name, "Scene 2");
    }

    #[test]
    fn test_library_undo_empty() {
        let mut lib = SceneLibrary::new();
        assert!(!lib.can_undo());
        assert!(lib.undo().is_none());
    }

    #[test]
    fn test_library_find_by_name() {
        let mut lib = SceneLibrary::new();
        lib.store_scene("Live Show A", make_channels(1));
        lib.store_scene("Studio Mix", make_channels(1));
        lib.store_scene("Live Show B", make_channels(1));

        let results = lib.find_by_name("live");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_library_find_by_tag() {
        let mut lib = SceneLibrary::new();
        let id1 = lib
            .store_scene("Scene 1", make_channels(1))
            .expect("id1 should be valid");
        let id2 = lib
            .store_scene("Scene 2", make_channels(1))
            .expect("id2 should be valid");

        lib.scenes
            .get_mut(&id1)
            .expect("get_mut should succeed")
            .add_tag("broadcast");
        lib.scenes
            .get_mut(&id2)
            .expect("get_mut should succeed")
            .add_tag("studio");

        let results = lib.find_by_tag("broadcast");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_interpolate_scenes() {
        let mut scene_a = Scene::new(SceneId(1), "A");
        scene_a.add_channel(ChannelSnapshot {
            name: "Ch0".to_string(),
            index: 0,
            fader: 0.0,
            pan: -1.0,
            ..Default::default()
        });

        let mut scene_b = Scene::new(SceneId(2), "B");
        scene_b.add_channel(ChannelSnapshot {
            name: "Ch0".to_string(),
            index: 0,
            fader: 1.0,
            pan: 1.0,
            ..Default::default()
        });

        let result = interpolate_scenes(&scene_a, &scene_b, 0.5);
        assert_eq!(result.channels.len(), 1);
        assert!((result.channels[0].fader - 0.5).abs() < f32::EPSILON);
        assert!(result.channels[0].pan.abs() < f32::EPSILON);
    }

    #[test]
    fn test_interpolate_boundary() {
        let mut scene_a = Scene::new(SceneId(1), "A");
        scene_a.add_channel(ChannelSnapshot {
            name: "A-ch".to_string(),
            index: 0,
            fader: 0.2,
            ..Default::default()
        });

        let mut scene_b = Scene::new(SceneId(2), "B");
        scene_b.add_channel(ChannelSnapshot {
            name: "B-ch".to_string(),
            index: 0,
            fader: 0.8,
            ..Default::default()
        });

        // t=0 should be exactly scene A
        let at0 = interpolate_scenes(&scene_a, &scene_b, 0.0);
        assert!((at0.channels[0].fader - 0.2).abs() < f32::EPSILON);

        // t=1 should be exactly scene B
        let at1 = interpolate_scenes(&scene_a, &scene_b, 1.0);
        assert!((at1.channels[0].fader - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_scene_order() {
        let mut lib = SceneLibrary::new();
        let id1 = lib.store_scene("A", vec![]).expect("id1 should be valid");
        let id2 = lib.store_scene("B", vec![]).expect("id2 should be valid");
        let id3 = lib.store_scene("C", vec![]).expect("id3 should be valid");
        let order = lib.scene_order();
        assert_eq!(order, &[id1, id2, id3]);
    }

    #[test]
    fn test_channel_snapshot_default() {
        let snap = ChannelSnapshot::default();
        assert!((snap.fader - 1.0).abs() < f32::EPSILON);
        assert!(snap.pan.abs() < f32::EPSILON);
        assert!(!snap.mute);
        assert!(!snap.solo);
    }
}
