#![allow(dead_code)]
//! Switcher state snapshot save and recall system.
//!
//! Provides the ability to capture complete switcher state (bus assignments,
//! keyer settings, transition config, audio levels) into named snapshots
//! that can be recalled instantly during live production.

use std::collections::HashMap;

/// Unique identifier for a snapshot.
pub type SnapshotId = u32;

/// Categories of state included in a snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SnapshotScope {
    /// Program and preview bus assignments.
    Buses,
    /// Transition settings (type, duration, etc.).
    Transitions,
    /// Upstream and downstream keyer settings.
    Keyers,
    /// Audio mixer levels and routing.
    AudioMixer,
    /// Aux bus assignments.
    AuxBuses,
    /// Media player assignments.
    MediaPlayers,
    /// All state categories.
    All,
}

/// Represents a single bus assignment within a snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BusSnapshot {
    /// M/E row index.
    pub me_row: usize,
    /// Program source input ID.
    pub program_source: usize,
    /// Preview source input ID.
    pub preview_source: usize,
}

/// Represents a keyer state within a snapshot.
#[derive(Debug, Clone)]
pub struct KeyerSnapshot {
    /// Keyer index.
    pub keyer_id: usize,
    /// Whether the keyer is on-air.
    pub on_air: bool,
    /// Fill source input ID.
    pub fill_source: usize,
    /// Key source input ID.
    pub key_source: usize,
    /// Key clip level (0.0 to 1.0).
    pub clip: f32,
    /// Key gain level (0.0 to 1.0).
    pub gain: f32,
    /// Whether the keyer is pre-multiplied.
    pub pre_multiplied: bool,
}

impl KeyerSnapshot {
    /// Create a default keyer snapshot.
    pub fn default_for_id(keyer_id: usize) -> Self {
        Self {
            keyer_id,
            on_air: false,
            fill_source: 0,
            key_source: 0,
            clip: 0.5,
            gain: 1.0,
            pre_multiplied: false,
        }
    }
}

/// Represents a transition config within a snapshot.
#[derive(Debug, Clone)]
pub struct TransitionSnapshot {
    /// M/E row index.
    pub me_row: usize,
    /// Transition type name.
    pub transition_type: String,
    /// Duration in frames.
    pub duration_frames: u32,
}

/// Represents an aux bus assignment within a snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuxSnapshot {
    /// Aux bus index.
    pub aux_id: usize,
    /// Assigned source input ID.
    pub source: usize,
}

/// Represents an audio channel level within a snapshot.
#[derive(Debug, Clone)]
pub struct AudioLevelSnapshot {
    /// Channel index.
    pub channel: usize,
    /// Fader level in dB.
    pub level_db: f32,
    /// Whether the channel is muted.
    pub muted: bool,
}

/// A complete switcher state snapshot.
#[derive(Debug, Clone)]
pub struct SwitcherSnapshot {
    /// Snapshot ID.
    pub id: SnapshotId,
    /// Human-readable name.
    pub name: String,
    /// Optional description.
    pub description: String,
    /// Scopes included in this snapshot.
    pub scopes: Vec<SnapshotScope>,
    /// Bus assignments.
    pub buses: Vec<BusSnapshot>,
    /// Keyer states.
    pub keyers: Vec<KeyerSnapshot>,
    /// Transition configs.
    pub transitions: Vec<TransitionSnapshot>,
    /// Aux bus assignments.
    pub aux_buses: Vec<AuxSnapshot>,
    /// Audio levels.
    pub audio_levels: Vec<AudioLevelSnapshot>,
    /// Creation timestamp (seconds since epoch, or a logical counter).
    pub created_at: u64,
}

impl SwitcherSnapshot {
    /// Create a new empty snapshot.
    pub fn new(id: SnapshotId, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            description: String::new(),
            scopes: vec![SnapshotScope::All],
            buses: Vec::new(),
            keyers: Vec::new(),
            transitions: Vec::new(),
            aux_buses: Vec::new(),
            audio_levels: Vec::new(),
            created_at: 0,
        }
    }

    /// Set the description.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Set the scopes.
    pub fn with_scopes(mut self, scopes: Vec<SnapshotScope>) -> Self {
        self.scopes = scopes;
        self
    }

    /// Check if a scope is included.
    pub fn includes_scope(&self, scope: SnapshotScope) -> bool {
        self.scopes.contains(&SnapshotScope::All) || self.scopes.contains(&scope)
    }

    /// Get the number of bus assignments stored.
    pub fn bus_count(&self) -> usize {
        self.buses.len()
    }

    /// Get the number of keyer states stored.
    pub fn keyer_count(&self) -> usize {
        self.keyers.len()
    }
}

/// Snapshot recall mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecallMode {
    /// Instant recall (hard cut to new state).
    Instant,
    /// Gradual recall over a specified number of frames.
    Gradual {
        /// Number of frames for the transition.
        frames: u32,
    },
}

/// Snapshot bank manager.
///
/// Manages a bank of saved snapshots with save, recall, and
/// organizational operations.
#[derive(Debug)]
pub struct SnapshotBank {
    /// Stored snapshots by ID.
    snapshots: HashMap<SnapshotId, SwitcherSnapshot>,
    /// Next available snapshot ID.
    next_id: SnapshotId,
    /// Maximum number of snapshots allowed.
    max_snapshots: usize,
    /// ID of the last recalled snapshot.
    last_recalled: Option<SnapshotId>,
    /// Recall mode.
    recall_mode: RecallMode,
}

impl SnapshotBank {
    /// Create a new snapshot bank.
    pub fn new(max_snapshots: usize) -> Self {
        Self {
            snapshots: HashMap::new(),
            next_id: 1,
            max_snapshots,
            last_recalled: None,
            recall_mode: RecallMode::Instant,
        }
    }

    /// Save a snapshot into the bank.
    pub fn save(&mut self, mut snapshot: SwitcherSnapshot) -> Result<SnapshotId, String> {
        if self.snapshots.len() >= self.max_snapshots {
            return Err(format!("Snapshot bank full (max {})", self.max_snapshots));
        }
        let id = self.next_id;
        snapshot.id = id;
        self.snapshots.insert(id, snapshot);
        self.next_id += 1;
        Ok(id)
    }

    /// Recall a snapshot by ID.
    pub fn recall(&mut self, id: SnapshotId) -> Result<&SwitcherSnapshot, String> {
        if let Some(snapshot) = self.snapshots.get(&id) {
            self.last_recalled = Some(id);
            Ok(snapshot)
        } else {
            Err(format!("Snapshot {} not found", id))
        }
    }

    /// Delete a snapshot by ID.
    pub fn delete(&mut self, id: SnapshotId) -> Result<(), String> {
        if self.snapshots.remove(&id).is_some() {
            if self.last_recalled == Some(id) {
                self.last_recalled = None;
            }
            Ok(())
        } else {
            Err(format!("Snapshot {} not found", id))
        }
    }

    /// Get a snapshot by ID without recalling it.
    pub fn get(&self, id: SnapshotId) -> Option<&SwitcherSnapshot> {
        self.snapshots.get(&id)
    }

    /// Get all snapshot IDs and names.
    pub fn list(&self) -> Vec<(SnapshotId, &str)> {
        let mut result: Vec<_> = self
            .snapshots
            .iter()
            .map(|(&id, s)| (id, s.name.as_str()))
            .collect();
        result.sort_by_key(|(id, _)| *id);
        result
    }

    /// Get the number of stored snapshots.
    pub fn count(&self) -> usize {
        self.snapshots.len()
    }

    /// Get the maximum capacity.
    pub fn capacity(&self) -> usize {
        self.max_snapshots
    }

    /// Get the ID of the last recalled snapshot.
    pub fn last_recalled(&self) -> Option<SnapshotId> {
        self.last_recalled
    }

    /// Set the recall mode.
    pub fn set_recall_mode(&mut self, mode: RecallMode) {
        self.recall_mode = mode;
    }

    /// Get the recall mode.
    pub fn recall_mode(&self) -> RecallMode {
        self.recall_mode
    }

    /// Rename a snapshot.
    pub fn rename(&mut self, id: SnapshotId, new_name: &str) -> Result<(), String> {
        if let Some(snapshot) = self.snapshots.get_mut(&id) {
            snapshot.name = new_name.to_string();
            Ok(())
        } else {
            Err(format!("Snapshot {} not found", id))
        }
    }

    /// Clear all snapshots.
    pub fn clear(&mut self) {
        self.snapshots.clear();
        self.last_recalled = None;
    }

    /// Check if the bank is full.
    pub fn is_full(&self) -> bool {
        self.snapshots.len() >= self.max_snapshots
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_creation() {
        let snap = SwitcherSnapshot::new(1, "Show Open");
        assert_eq!(snap.id, 1);
        assert_eq!(snap.name, "Show Open");
        assert!(snap.description.is_empty());
    }

    #[test]
    fn test_snapshot_with_description() {
        let snap = SwitcherSnapshot::new(1, "News").with_description("6PM news open");
        assert_eq!(snap.description, "6PM news open");
    }

    #[test]
    fn test_snapshot_scopes() {
        let snap = SwitcherSnapshot::new(1, "Test")
            .with_scopes(vec![SnapshotScope::Buses, SnapshotScope::Keyers]);
        assert!(snap.includes_scope(SnapshotScope::Buses));
        assert!(snap.includes_scope(SnapshotScope::Keyers));
        assert!(!snap.includes_scope(SnapshotScope::AudioMixer));
    }

    #[test]
    fn test_snapshot_all_scope() {
        let snap = SwitcherSnapshot::new(1, "All");
        assert!(snap.includes_scope(SnapshotScope::Buses));
        assert!(snap.includes_scope(SnapshotScope::AudioMixer));
        assert!(snap.includes_scope(SnapshotScope::All));
    }

    #[test]
    fn test_keyer_snapshot_default() {
        let ks = KeyerSnapshot::default_for_id(0);
        assert_eq!(ks.keyer_id, 0);
        assert!(!ks.on_air);
        assert!((ks.gain - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_snapshot_bank_creation() {
        let bank = SnapshotBank::new(100);
        assert_eq!(bank.count(), 0);
        assert_eq!(bank.capacity(), 100);
        assert!(!bank.is_full());
    }

    #[test]
    fn test_snapshot_bank_save_recall() {
        let mut bank = SnapshotBank::new(10);
        let snap = SwitcherSnapshot::new(0, "Test 1");
        let id = bank.save(snap).expect("should succeed in test");
        assert_eq!(id, 1);
        assert_eq!(bank.count(), 1);

        let recalled = bank.recall(id).expect("should succeed in test");
        assert_eq!(recalled.name, "Test 1");
        assert_eq!(bank.last_recalled(), Some(id));
    }

    #[test]
    fn test_snapshot_bank_full() {
        let mut bank = SnapshotBank::new(2);
        bank.save(SwitcherSnapshot::new(0, "A"))
            .expect("should succeed in test");
        bank.save(SwitcherSnapshot::new(0, "B"))
            .expect("should succeed in test");
        assert!(bank.is_full());
        assert!(bank.save(SwitcherSnapshot::new(0, "C")).is_err());
    }

    #[test]
    fn test_snapshot_bank_delete() {
        let mut bank = SnapshotBank::new(10);
        let id = bank
            .save(SwitcherSnapshot::new(0, "X"))
            .expect("should succeed in test");
        assert_eq!(bank.count(), 1);
        bank.delete(id).expect("should succeed in test");
        assert_eq!(bank.count(), 0);
    }

    #[test]
    fn test_snapshot_bank_delete_not_found() {
        let mut bank = SnapshotBank::new(10);
        assert!(bank.delete(999).is_err());
    }

    #[test]
    fn test_snapshot_bank_recall_not_found() {
        let mut bank = SnapshotBank::new(10);
        assert!(bank.recall(999).is_err());
    }

    #[test]
    fn test_snapshot_bank_list() {
        let mut bank = SnapshotBank::new(10);
        bank.save(SwitcherSnapshot::new(0, "Alpha"))
            .expect("should succeed in test");
        bank.save(SwitcherSnapshot::new(0, "Beta"))
            .expect("should succeed in test");
        let list = bank.list();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].1, "Alpha");
        assert_eq!(list[1].1, "Beta");
    }

    #[test]
    fn test_snapshot_bank_rename() {
        let mut bank = SnapshotBank::new(10);
        let id = bank
            .save(SwitcherSnapshot::new(0, "Old"))
            .expect("should succeed in test");
        bank.rename(id, "New Name").expect("should succeed in test");
        let snap = bank.get(id).expect("should succeed in test");
        assert_eq!(snap.name, "New Name");
    }

    #[test]
    fn test_snapshot_bank_clear() {
        let mut bank = SnapshotBank::new(10);
        bank.save(SwitcherSnapshot::new(0, "A"))
            .expect("should succeed in test");
        bank.save(SwitcherSnapshot::new(0, "B"))
            .expect("should succeed in test");
        bank.clear();
        assert_eq!(bank.count(), 0);
        assert!(bank.last_recalled().is_none());
    }

    #[test]
    fn test_snapshot_bank_recall_mode() {
        let mut bank = SnapshotBank::new(10);
        assert_eq!(bank.recall_mode(), RecallMode::Instant);
        bank.set_recall_mode(RecallMode::Gradual { frames: 30 });
        assert_eq!(bank.recall_mode(), RecallMode::Gradual { frames: 30 });
    }

    #[test]
    fn test_snapshot_bus_count() {
        let mut snap = SwitcherSnapshot::new(1, "Test");
        snap.buses.push(BusSnapshot {
            me_row: 0,
            program_source: 1,
            preview_source: 2,
        });
        assert_eq!(snap.bus_count(), 1);
        assert_eq!(snap.keyer_count(), 0);
    }
}
