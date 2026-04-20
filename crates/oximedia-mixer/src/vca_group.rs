//! VCA (Voltage Controlled Amplifier) group manager for the mixer.
//!
//! A VCA group lets a single master fader control the gain of many channels
//! simultaneously while still allowing each channel's own fader to move
//! independently.  The master's contribution is **additive in dB**: if the
//! master is at +6 dB every linked channel is 6 dB louder than its own fader
//! setting, and if the master is muted all linked channels are silenced.
//!
//! # Model
//!
//! ```text
//! effective_gain_db = channel_fader_db + vca_master_trim_db
//! effective_linear  = linear(effective_gain_db)   (clamped to silence when muted)
//! ```
//!
//! Multiple VCA groups may be applied to a single channel (they stack
//! multiplicatively in linear gain space).  Each group carries its own
//! automation snapshot history so the complete control surface state can be
//! captured and recalled.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by VCA group operations.
#[derive(Debug, thiserror::Error)]
pub enum VcaGroupError {
    /// Attempted to use an identifier that does not exist.
    #[error("VCA group not found: id={0}")]
    NotFound(VcaGroupId),

    /// A channel is already a member of this group.
    #[error("Channel {channel} is already a member of group {group}")]
    AlreadyMember {
        /// The VCA group identifier.
        group: VcaGroupId,
        /// The channel identifier being added.
        channel: u32,
    },

    /// The trim value is outside the permitted range.
    #[error("Trim value {value} dB is outside the allowed range [{min}, {max}]")]
    TrimOutOfRange {
        /// Value that was attempted.
        value: f32,
        /// Minimum allowed value.
        min: f32,
        /// Maximum allowed value.
        max: f32,
    },
}

/// Result alias for VCA group operations.
pub type VcaGroupResult<T> = Result<T, VcaGroupError>;

// ---------------------------------------------------------------------------
// Identifier
// ---------------------------------------------------------------------------

/// Opaque identifier for a [`VcaGroup`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VcaGroupId(pub u32);

impl std::fmt::Display for VcaGroupId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VCA:{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Automation snapshot
// ---------------------------------------------------------------------------

/// A single point in the VCA group's automation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VcaAutomationPoint {
    /// Sample position (relative to session start).
    pub sample_position: u64,
    /// Trim value in dB at this point.
    pub trim_db: f32,
    /// Mute state at this point.
    pub muted: bool,
}

// ---------------------------------------------------------------------------
// VcaGroup
// ---------------------------------------------------------------------------

/// Minimum allowed trim in dB.
pub const VCA_TRIM_MIN_DB: f32 = -144.0;
/// Maximum allowed trim in dB (headroom beyond unity).
pub const VCA_TRIM_MAX_DB: f32 = 12.0;

/// A single VCA group: one master trim that applies to all member channels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VcaGroup {
    /// Unique identifier.
    pub id: VcaGroupId,
    /// Human-readable name.
    pub name: String,
    /// Colour hint for control surface display (HTML hex, e.g. `"#3A8FEB"`).
    pub color: String,
    /// Trim applied on top of each member channel's own fader (additive in dB).
    pub trim_db: f32,
    /// When `true` all member channels are silenced regardless of `trim_db`.
    pub muted: bool,
    /// Set of channel indices that belong to this group.
    members: Vec<u32>,
    /// Recorded automation points ordered by `sample_position`.
    automation: Vec<VcaAutomationPoint>,
}

impl VcaGroup {
    /// Create a new VCA group.
    #[must_use]
    pub fn new(id: VcaGroupId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            color: "#888888".to_string(),
            trim_db: 0.0,
            muted: false,
            members: Vec::new(),
            automation: Vec::new(),
        }
    }

    /// Add a channel to this group.  Returns `AlreadyMember` if the channel is
    /// already present.
    pub fn add_channel(&mut self, channel: u32) -> VcaGroupResult<()> {
        if self.members.contains(&channel) {
            return Err(VcaGroupError::AlreadyMember {
                group: self.id,
                channel,
            });
        }
        self.members.push(channel);
        Ok(())
    }

    /// Remove a channel from the group.  No-op if the channel is not a member.
    pub fn remove_channel(&mut self, channel: u32) {
        self.members.retain(|&c| c != channel);
    }

    /// Returns a slice of all member channel indices.
    #[must_use]
    pub fn members(&self) -> &[u32] {
        &self.members
    }

    /// Set the master trim in dB.  Clamped to [`VCA_TRIM_MIN_DB`] ..
    /// [`VCA_TRIM_MAX_DB`].
    pub fn set_trim_db(&mut self, db: f32) -> VcaGroupResult<()> {
        if db < VCA_TRIM_MIN_DB || db > VCA_TRIM_MAX_DB {
            return Err(VcaGroupError::TrimOutOfRange {
                value: db,
                min: VCA_TRIM_MIN_DB,
                max: VCA_TRIM_MAX_DB,
            });
        }
        self.trim_db = db;
        Ok(())
    }

    /// Compute the effective **linear** gain multiplier produced by this group
    /// for a member channel whose own fader is at `channel_gain_linear`.
    ///
    /// Returns `0.0` when the group is muted.
    #[must_use]
    pub fn effective_gain(&self, channel_gain_linear: f32) -> f32 {
        if self.muted {
            return 0.0;
        }
        channel_gain_linear * db_to_linear(self.trim_db)
    }

    /// Compute only the VCA linear multiplier (ignoring any per-channel gain).
    ///
    /// Returns `0.0` when muted.
    #[must_use]
    pub fn multiplier(&self) -> f32 {
        if self.muted {
            0.0
        } else {
            db_to_linear(self.trim_db)
        }
    }

    /// Record an automation point.
    ///
    /// Points are kept sorted by `sample_position`.  If an existing point at
    /// the same position exists it is replaced.
    pub fn record_automation(&mut self, sample_position: u64) {
        let point = VcaAutomationPoint {
            sample_position,
            trim_db: self.trim_db,
            muted: self.muted,
        };
        // Replace or insert in sorted order.
        match self
            .automation
            .binary_search_by_key(&sample_position, |p| p.sample_position)
        {
            Ok(idx) => self.automation[idx] = point,
            Err(idx) => self.automation.insert(idx, point),
        }
    }

    /// Play back automation at the given sample position.
    ///
    /// Linearly interpolates `trim_db` between neighbouring automation points.
    /// The mute state is taken from the nearest preceding point.
    pub fn play_automation(&mut self, sample_position: u64) {
        if self.automation.is_empty() {
            return;
        }
        match self
            .automation
            .binary_search_by_key(&sample_position, |p| p.sample_position)
        {
            Ok(idx) => {
                self.trim_db = self.automation[idx].trim_db;
                self.muted = self.automation[idx].muted;
            }
            Err(0) => {
                // Before first point — hold first value.
                self.trim_db = self.automation[0].trim_db;
                self.muted = self.automation[0].muted;
            }
            Err(idx) if idx >= self.automation.len() => {
                // After last point — hold last value.
                let last = &self.automation[self.automation.len() - 1];
                self.trim_db = last.trim_db;
                self.muted = last.muted;
            }
            Err(idx) => {
                // Between two points — interpolate trim, take mute from previous.
                let prev = &self.automation[idx - 1];
                let next = &self.automation[idx];
                let t = (sample_position - prev.sample_position) as f32
                    / (next.sample_position - prev.sample_position) as f32;
                self.trim_db = prev.trim_db + t * (next.trim_db - prev.trim_db);
                self.muted = prev.muted;
            }
        }
    }

    /// Clear all recorded automation.
    pub fn clear_automation(&mut self) {
        self.automation.clear();
    }

    /// Snapshot the current group state as a [`VcaGroupSnapshot`].
    #[must_use]
    pub fn snapshot(&self) -> VcaGroupSnapshot {
        VcaGroupSnapshot {
            id: self.id,
            name: self.name.clone(),
            trim_db: self.trim_db,
            muted: self.muted,
            members: self.members.clone(),
        }
    }

    /// Restore state from a [`VcaGroupSnapshot`] (does not affect automation).
    pub fn restore_snapshot(&mut self, snap: &VcaGroupSnapshot) {
        self.trim_db = snap.trim_db;
        self.muted = snap.muted;
        self.members = snap.members.clone();
    }
}

// ---------------------------------------------------------------------------
// Snapshot
// ---------------------------------------------------------------------------

/// Serializable point-in-time capture of a VCA group's state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VcaGroupSnapshot {
    /// Group identifier.
    pub id: VcaGroupId,
    /// Group name at snapshot time.
    pub name: String,
    /// Trim in dB at snapshot time.
    pub trim_db: f32,
    /// Mute state at snapshot time.
    pub muted: bool,
    /// Member channel indices at snapshot time.
    pub members: Vec<u32>,
}

// ---------------------------------------------------------------------------
// VcaGroupManager
// ---------------------------------------------------------------------------

/// Manages all VCA groups in a mixing session.
///
/// Provides group creation/deletion, membership queries, and batch gain
/// computation for the DSP pipeline.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct VcaGroupManager {
    groups: HashMap<VcaGroupId, VcaGroup>,
    next_id: u32,
}

impl VcaGroupManager {
    /// Create an empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new group and return its identifier.
    pub fn create_group(&mut self, name: impl Into<String>) -> VcaGroupId {
        let id = VcaGroupId(self.next_id);
        self.next_id += 1;
        self.groups.insert(id, VcaGroup::new(id, name));
        id
    }

    /// Remove a group.  Returns `NotFound` if it does not exist.
    pub fn remove_group(&mut self, id: VcaGroupId) -> VcaGroupResult<()> {
        self.groups
            .remove(&id)
            .map(|_| ())
            .ok_or(VcaGroupError::NotFound(id))
    }

    /// Get an immutable reference to a group.
    pub fn group(&self, id: VcaGroupId) -> VcaGroupResult<&VcaGroup> {
        self.groups.get(&id).ok_or(VcaGroupError::NotFound(id))
    }

    /// Get a mutable reference to a group.
    pub fn group_mut(&mut self, id: VcaGroupId) -> VcaGroupResult<&mut VcaGroup> {
        self.groups.get_mut(&id).ok_or(VcaGroupError::NotFound(id))
    }

    /// Return all group identifiers in creation order.
    #[must_use]
    pub fn group_ids(&self) -> Vec<VcaGroupId> {
        let mut ids: Vec<VcaGroupId> = self.groups.keys().copied().collect();
        ids.sort();
        ids
    }

    /// Compute the combined VCA multiplier for a channel that may belong to
    /// multiple groups.
    ///
    /// Each group's linear multiplier is **multiplied** together.  If the
    /// channel is not a member of any group the result is `1.0`.
    #[must_use]
    pub fn combined_multiplier(&self, channel: u32) -> f32 {
        self.groups
            .values()
            .filter(|g| g.members.contains(&channel))
            .map(|g| g.multiplier())
            .fold(1.0_f32, |acc, m| acc * m)
    }

    /// Advance automation playback for every group to `sample_position`.
    pub fn advance_automation(&mut self, sample_position: u64) {
        for group in self.groups.values_mut() {
            group.play_automation(sample_position);
        }
    }

    /// Snapshot all groups in the manager.
    #[must_use]
    pub fn snapshot_all(&self) -> Vec<VcaGroupSnapshot> {
        let mut snaps: Vec<VcaGroupSnapshot> = self.groups.values().map(|g| g.snapshot()).collect();
        snaps.sort_by_key(|s| s.id);
        snaps
    }

    /// Restore all groups from a collection of snapshots.
    ///
    /// Groups not present in the snapshot collection are left unchanged.
    pub fn restore_all(&mut self, snapshots: &[VcaGroupSnapshot]) {
        for snap in snapshots {
            if let Some(group) = self.groups.get_mut(&snap.id) {
                group.restore_snapshot(snap);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert dB to linear gain.
#[inline]
fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
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
    fn test_create_and_add_members() {
        let mut mgr = VcaGroupManager::new();
        let id = mgr.create_group("Drums");
        let group = mgr.group_mut(id).expect("group should exist after create_group");
        group.add_channel(0).expect("add_channel 0 should succeed");
        group.add_channel(1).expect("add_channel 1 should succeed");
        assert_eq!(group.members().len(), 2);
    }

    #[test]
    fn test_duplicate_member_rejected() {
        let mut mgr = VcaGroupManager::new();
        let id = mgr.create_group("Brass");
        let group = mgr.group_mut(id).expect("group should exist after create_group");
        group.add_channel(5).expect("first add_channel should succeed");
        assert!(group.add_channel(5).is_err());
    }

    #[test]
    fn test_trim_out_of_range_rejected() {
        let mut mgr = VcaGroupManager::new();
        let id = mgr.create_group("Strings");
        let group = mgr.group_mut(id).expect("group should exist after create_group");
        assert!(group.set_trim_db(100.0).is_err());
        assert!(group.set_trim_db(-200.0).is_err());
        assert!(group.set_trim_db(6.0).is_ok());
    }

    #[test]
    fn test_mute_returns_zero_gain() {
        let mut mgr = VcaGroupManager::new();
        let id = mgr.create_group("Keys");
        let group = mgr.group_mut(id).expect("group should exist after create_group");
        group.muted = true;
        assert_eq!(group.multiplier(), 0.0);
        assert_eq!(group.effective_gain(0.8), 0.0);
    }

    #[test]
    fn test_unity_trim_preserves_channel_gain() {
        let mut mgr = VcaGroupManager::new();
        let id = mgr.create_group("Guitars");
        let group = mgr.group(id).expect("group should exist after create_group");
        let result = group.effective_gain(0.5);
        assert!(approx_eq(result, 0.5, 1e-5));
    }

    #[test]
    fn test_combined_multiplier_two_groups() {
        let mut mgr = VcaGroupManager::new();
        let g1 = mgr.create_group("Group1");
        let g2 = mgr.create_group("Group2");
        mgr.group_mut(g1).expect("group1 should exist").add_channel(3).expect("add_channel to group1 should succeed");
        mgr.group_mut(g2).expect("group2 should exist").add_channel(3).expect("add_channel to group2 should succeed");
        // Both at unity → combined = 1.0
        assert!(approx_eq(mgr.combined_multiplier(3), 1.0, 1e-5));
        // g1 muted → combined = 0.0
        mgr.group_mut(g1).expect("group1 should still exist").muted = true;
        assert_eq!(mgr.combined_multiplier(3), 0.0);
    }

    #[test]
    fn test_automation_record_and_playback() {
        let mut mgr = VcaGroupManager::new();
        let id = mgr.create_group("Automation");
        {
            let g = mgr.group_mut(id).expect("automation group should exist");
            g.set_trim_db(-6.0).expect("set_trim_db -6 should succeed");
            g.record_automation(0);
            g.set_trim_db(0.0).expect("set_trim_db 0 should succeed");
            g.record_automation(1000);
        }
        // At midpoint trim should be ~-3 dB
        mgr.advance_automation(500);
        let trim = mgr.group(id).expect("automation group should still exist").trim_db;
        assert!(approx_eq(trim, -3.0, 0.1));
    }

    #[test]
    fn test_snapshot_and_restore() {
        let mut mgr = VcaGroupManager::new();
        let id = mgr.create_group("Snapshot");
        {
            let g = mgr.group_mut(id).expect("snapshot group should exist");
            g.add_channel(10).expect("add_channel 10 should succeed");
            g.set_trim_db(-3.0).expect("set_trim_db -3 should succeed");
        }
        let snaps = mgr.snapshot_all();
        // Mutate state
        mgr.group_mut(id).expect("snapshot group should still exist").set_trim_db(0.0).expect("set_trim_db 0 should succeed");
        // Restore
        mgr.restore_all(&snaps);
        assert!(approx_eq(mgr.group(id).expect("snapshot group should still exist after restore").trim_db, -3.0, 1e-5));
    }

    #[test]
    fn test_remove_channel() {
        let mut mgr = VcaGroupManager::new();
        let id = mgr.create_group("Remove");
        let g = mgr.group_mut(id).expect("remove group should exist");
        g.add_channel(7).expect("add_channel 7 should succeed");
        g.remove_channel(7);
        assert!(g.members().is_empty());
    }

    #[test]
    fn test_remove_group_not_found() {
        let mut mgr = VcaGroupManager::new();
        assert!(mgr.remove_group(VcaGroupId(999)).is_err());
    }
}
