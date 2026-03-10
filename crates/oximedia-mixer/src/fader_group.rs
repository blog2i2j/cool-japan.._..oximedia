#![allow(dead_code)]
//! Fader grouping and linking for mixer channels.
//!
//! Provides the ability to group faders so they move together,
//! supporting relative and absolute linking, group masters,
//! and fader-start behavior. Essential for managing large
//! mix sessions with many channels.

use serde::{Deserialize, Serialize};

/// Determines how linked faders follow the group master.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinkMode {
    /// Absolute: all faders move to the same value.
    Absolute,
    /// Relative: faders maintain their offset from the master.
    Relative,
    /// VCA-style: no signal summing, gain offset applied multiplicatively.
    Vca,
}

/// Identifier for a fader group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FaderGroupId(pub u32);

/// A single member within a fader group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaderMember {
    /// Channel index within the mixer.
    pub channel_index: u32,
    /// Offset from the group master value (used in relative mode).
    pub offset: f32,
    /// Whether this member is temporarily suspended from the group.
    pub suspended: bool,
}

impl FaderMember {
    /// Creates a new fader member with the given channel index.
    #[must_use]
    pub fn new(channel_index: u32) -> Self {
        Self {
            channel_index,
            offset: 0.0,
            suspended: false,
        }
    }

    /// Computes the effective gain for this member given the master value
    /// and the link mode.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn effective_gain(&self, master_value: f32, mode: LinkMode) -> f32 {
        if self.suspended {
            return master_value; // suspended members keep their current value
        }
        match mode {
            LinkMode::Absolute => master_value.clamp(0.0, 1.0),
            LinkMode::Relative => (master_value + self.offset).clamp(0.0, 1.0),
            LinkMode::Vca => (master_value * (1.0 + self.offset)).clamp(0.0, 1.0),
        }
    }
}

/// A fader group containing multiple channel faders.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaderGroup {
    /// Unique identifier.
    pub id: FaderGroupId,
    /// Human-readable name.
    pub name: String,
    /// Link mode for this group.
    pub mode: LinkMode,
    /// Master fader value (0.0..=1.0).
    pub master_value: f32,
    /// Whether the group is currently active.
    pub active: bool,
    /// Members of this group.
    members: Vec<FaderMember>,
}

impl FaderGroup {
    /// Creates a new fader group.
    #[must_use]
    pub fn new(id: FaderGroupId, name: String, mode: LinkMode) -> Self {
        Self {
            id,
            name,
            mode,
            master_value: 0.75,
            active: true,
            members: Vec::new(),
        }
    }

    /// Returns the number of members.
    #[must_use]
    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    /// Returns a slice of all members.
    #[must_use]
    pub fn members(&self) -> &[FaderMember] {
        &self.members
    }

    /// Adds a channel to this group.
    ///
    /// Returns `false` if the channel is already a member.
    pub fn add_member(&mut self, channel_index: u32) -> bool {
        if self.members.iter().any(|m| m.channel_index == channel_index) {
            return false;
        }
        self.members.push(FaderMember::new(channel_index));
        true
    }

    /// Removes a channel from this group.
    ///
    /// Returns `true` if the channel was found and removed.
    pub fn remove_member(&mut self, channel_index: u32) -> bool {
        let before = self.members.len();
        self.members.retain(|m| m.channel_index != channel_index);
        self.members.len() < before
    }

    /// Returns whether the given channel is a member.
    #[must_use]
    pub fn contains(&self, channel_index: u32) -> bool {
        self.members.iter().any(|m| m.channel_index == channel_index)
    }

    /// Sets the master fader value and returns the computed gain for each member.
    pub fn set_master(&mut self, value: f32) -> Vec<(u32, f32)> {
        self.master_value = value.clamp(0.0, 1.0);
        if !self.active {
            return Vec::new();
        }
        self.members
            .iter()
            .filter(|m| !m.suspended)
            .map(|m| (m.channel_index, m.effective_gain(self.master_value, self.mode)))
            .collect()
    }

    /// Sets the offset for a specific member (for relative/VCA modes).
    ///
    /// Returns `false` if the channel is not a member.
    pub fn set_member_offset(&mut self, channel_index: u32, offset: f32) -> bool {
        if let Some(member) = self.members.iter_mut().find(|m| m.channel_index == channel_index) {
            member.offset = offset;
            true
        } else {
            false
        }
    }

    /// Suspends a member from the group (it will stop following the master).
    ///
    /// Returns `false` if the channel is not a member.
    pub fn suspend_member(&mut self, channel_index: u32) -> bool {
        if let Some(member) = self.members.iter_mut().find(|m| m.channel_index == channel_index) {
            member.suspended = true;
            true
        } else {
            false
        }
    }

    /// Resumes a suspended member.
    ///
    /// Returns `false` if the channel is not a member.
    pub fn resume_member(&mut self, channel_index: u32) -> bool {
        if let Some(member) = self.members.iter_mut().find(|m| m.channel_index == channel_index) {
            member.suspended = false;
            true
        } else {
            false
        }
    }

    /// Returns the count of active (non-suspended) members.
    #[must_use]
    pub fn active_member_count(&self) -> usize {
        self.members.iter().filter(|m| !m.suspended).count()
    }

    /// Computes the effective gain for a specific member.
    #[must_use]
    pub fn gain_for_member(&self, channel_index: u32) -> Option<f32> {
        self.members
            .iter()
            .find(|m| m.channel_index == channel_index)
            .map(|m| m.effective_gain(self.master_value, self.mode))
    }
}

/// Manages multiple fader groups.
#[derive(Debug, Clone, Default)]
pub struct FaderGroupManager {
    groups: Vec<FaderGroup>,
    next_id: u32,
}

impl FaderGroupManager {
    /// Creates a new fader group manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            groups: Vec::new(),
            next_id: 1,
        }
    }

    /// Creates a new fader group and returns its identifier.
    pub fn create_group(&mut self, name: String, mode: LinkMode) -> FaderGroupId {
        let id = FaderGroupId(self.next_id);
        self.next_id += 1;
        self.groups.push(FaderGroup::new(id, name, mode));
        id
    }

    /// Returns the number of groups.
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Returns a reference to a group by its identifier.
    #[must_use]
    pub fn group(&self, id: FaderGroupId) -> Option<&FaderGroup> {
        self.groups.iter().find(|g| g.id == id)
    }

    /// Returns a mutable reference to a group by its identifier.
    pub fn group_mut(&mut self, id: FaderGroupId) -> Option<&mut FaderGroup> {
        self.groups.iter_mut().find(|g| g.id == id)
    }

    /// Removes a group by its identifier.
    ///
    /// Returns `true` if the group was found and removed.
    pub fn remove_group(&mut self, id: FaderGroupId) -> bool {
        let before = self.groups.len();
        self.groups.retain(|g| g.id != id);
        self.groups.len() < before
    }

    /// Finds all groups that a channel belongs to.
    #[must_use]
    pub fn groups_for_channel(&self, channel_index: u32) -> Vec<FaderGroupId> {
        self.groups
            .iter()
            .filter(|g| g.contains(channel_index))
            .map(|g| g.id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_group() {
        let group = FaderGroup::new(FaderGroupId(1), "Drums".into(), LinkMode::Relative);
        assert_eq!(group.member_count(), 0);
        assert!(group.active);
        assert_eq!(group.mode, LinkMode::Relative);
    }

    #[test]
    fn test_add_member() {
        let mut group = FaderGroup::new(FaderGroupId(1), "Drums".into(), LinkMode::Absolute);
        assert!(group.add_member(0));
        assert!(group.add_member(1));
        assert_eq!(group.member_count(), 2);
    }

    #[test]
    fn test_add_duplicate_member() {
        let mut group = FaderGroup::new(FaderGroupId(1), "G".into(), LinkMode::Absolute);
        assert!(group.add_member(5));
        assert!(!group.add_member(5));
        assert_eq!(group.member_count(), 1);
    }

    #[test]
    fn test_remove_member() {
        let mut group = FaderGroup::new(FaderGroupId(1), "G".into(), LinkMode::Absolute);
        group.add_member(0);
        group.add_member(1);
        assert!(group.remove_member(0));
        assert_eq!(group.member_count(), 1);
        assert!(!group.remove_member(99));
    }

    #[test]
    fn test_absolute_mode() {
        let mut group = FaderGroup::new(FaderGroupId(1), "G".into(), LinkMode::Absolute);
        group.add_member(0);
        group.add_member(1);
        let gains = group.set_master(0.6);
        assert_eq!(gains.len(), 2);
        for (_, g) in &gains {
            assert!((*g - 0.6).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_relative_mode() {
        let mut group = FaderGroup::new(FaderGroupId(1), "G".into(), LinkMode::Relative);
        group.add_member(0);
        group.add_member(1);
        group.set_member_offset(1, -0.1);
        let gains = group.set_master(0.8);
        // member 0: 0.8 + 0.0 = 0.8
        // member 1: 0.8 + (-0.1) = 0.7
        assert!((gains[0].1 - 0.8).abs() < f32::EPSILON);
        assert!((gains[1].1 - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_vca_mode() {
        let mut group = FaderGroup::new(FaderGroupId(1), "G".into(), LinkMode::Vca);
        group.add_member(0);
        let gains = group.set_master(0.5);
        // 0.5 * (1.0 + 0.0) = 0.5
        assert!((gains[0].1 - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_suspend_member() {
        let mut group = FaderGroup::new(FaderGroupId(1), "G".into(), LinkMode::Absolute);
        group.add_member(0);
        group.add_member(1);
        group.suspend_member(1);
        let gains = group.set_master(0.5);
        // Only member 0 should be in the list (member 1 is suspended)
        assert_eq!(gains.len(), 1);
        assert_eq!(gains[0].0, 0);
    }

    #[test]
    fn test_resume_member() {
        let mut group = FaderGroup::new(FaderGroupId(1), "G".into(), LinkMode::Absolute);
        group.add_member(0);
        group.suspend_member(0);
        assert_eq!(group.active_member_count(), 0);
        group.resume_member(0);
        assert_eq!(group.active_member_count(), 1);
    }

    #[test]
    fn test_gain_clamping() {
        let mut group = FaderGroup::new(FaderGroupId(1), "G".into(), LinkMode::Relative);
        group.add_member(0);
        group.set_member_offset(0, 0.5);
        let gains = group.set_master(0.9);
        // 0.9 + 0.5 = 1.4 => clamped to 1.0
        assert!((gains[0].1 - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_inactive_group() {
        let mut group = FaderGroup::new(FaderGroupId(1), "G".into(), LinkMode::Absolute);
        group.add_member(0);
        group.active = false;
        let gains = group.set_master(0.5);
        assert!(gains.is_empty());
    }

    #[test]
    fn test_manager_create_group() {
        let mut mgr = FaderGroupManager::new();
        let id = mgr.create_group("Drums".into(), LinkMode::Absolute);
        assert_eq!(mgr.group_count(), 1);
        assert!(mgr.group(id).is_some());
    }

    #[test]
    fn test_manager_remove_group() {
        let mut mgr = FaderGroupManager::new();
        let id = mgr.create_group("Drums".into(), LinkMode::Absolute);
        assert!(mgr.remove_group(id));
        assert_eq!(mgr.group_count(), 0);
        assert!(!mgr.remove_group(id));
    }

    #[test]
    fn test_manager_groups_for_channel() {
        let mut mgr = FaderGroupManager::new();
        let id1 = mgr.create_group("G1".into(), LinkMode::Absolute);
        let id2 = mgr.create_group("G2".into(), LinkMode::Relative);
        mgr.group_mut(id1).expect("group_mut should succeed").add_member(5);
        mgr.group_mut(id2).expect("group_mut should succeed").add_member(5);
        let groups = mgr.groups_for_channel(5);
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn test_contains() {
        let mut group = FaderGroup::new(FaderGroupId(1), "G".into(), LinkMode::Absolute);
        group.add_member(3);
        assert!(group.contains(3));
        assert!(!group.contains(4));
    }

    #[test]
    fn test_gain_for_member() {
        let mut group = FaderGroup::new(FaderGroupId(1), "G".into(), LinkMode::Absolute);
        group.add_member(0);
        group.master_value = 0.7;
        assert!((group.gain_for_member(0).expect("gain_for_member should succeed") - 0.7).abs() < f32::EPSILON);
        assert!(group.gain_for_member(99).is_none());
    }
}
