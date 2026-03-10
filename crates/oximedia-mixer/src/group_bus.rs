//! Group bus management for the `OxiMedia` mixer.
//!
//! A *group bus* (also called a *subgroup*) collects the output of multiple
//! channel strips and routes them to a single fader before reaching the master
//! bus.  This module models subgroup assignment, group-level fader and mute
//! controls, and mute-group logic.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

// ────────────────────────────────────────────────────────────────────────────
// GroupBusId
// ────────────────────────────────────────────────────────────────────────────

/// Opaque identifier for a group bus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GroupBusId(pub u32);

impl std::fmt::Display for GroupBusId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GroupBus({})", self.0)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// GroupBus
// ────────────────────────────────────────────────────────────────────────────

/// A single group (subgroup) bus.
#[derive(Debug, Clone)]
pub struct GroupBus {
    /// Unique bus identifier.
    pub id: GroupBusId,
    /// Human-readable label.
    pub name: String,
    /// Fader level in the range 0.0 (silence) … 2.0 (unity = 1.0, +6 dB = 2.0).
    pub fader: f32,
    /// Whether this bus is muted.
    pub muted: bool,
    /// Whether this bus is soloed.
    pub soloed: bool,
    /// Channel strip indices assigned to this group.
    pub members: HashSet<u32>,
    /// Optional colour tag (hex string, e.g. `"#FF6600"`).
    pub colour: Option<String>,
}

impl GroupBus {
    /// Create a new group bus at unity gain.
    #[must_use]
    pub fn new(id: GroupBusId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            fader: 1.0,
            muted: false,
            soloed: false,
            members: HashSet::new(),
            colour: None,
        }
    }

    /// Assign a channel strip to this group.
    pub fn add_member(&mut self, channel_id: u32) {
        self.members.insert(channel_id);
    }

    /// Remove a channel strip from this group.
    pub fn remove_member(&mut self, channel_id: u32) {
        self.members.remove(&channel_id);
    }

    /// Returns `true` if the channel is a member of this group.
    #[must_use]
    pub fn has_member(&self, channel_id: u32) -> bool {
        self.members.contains(&channel_id)
    }

    /// Set the fader level, clamped to `[0.0, 2.0]`.
    pub fn set_fader(&mut self, level: f32) {
        self.fader = level.clamp(0.0, 2.0);
    }

    /// Toggle mute state, returning the new state.
    pub fn toggle_mute(&mut self) -> bool {
        self.muted = !self.muted;
        self.muted
    }

    /// Effective output gain: `fader` if not muted, else `0.0`.
    #[must_use]
    pub fn effective_gain(&self) -> f32 {
        if self.muted {
            0.0
        } else {
            self.fader
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// MuteGroup
// ────────────────────────────────────────────────────────────────────────────

/// A *mute group* allows a single trigger to mute/unmute several buses at once.
#[derive(Debug, Clone, Default)]
pub struct MuteGroup {
    /// Buses that belong to this mute group (by `GroupBusId`).
    pub bus_ids: HashSet<GroupBusId>,
    /// Whether the group is currently engaged (all members muted).
    pub active: bool,
}

impl MuteGroup {
    /// Create a new, inactive mute group.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a bus to the mute group.
    pub fn add_bus(&mut self, id: GroupBusId) {
        self.bus_ids.insert(id);
    }

    /// Remove a bus from the mute group.
    pub fn remove_bus(&mut self, id: GroupBusId) {
        self.bus_ids.remove(&id);
    }

    /// Engage the mute group, muting all member buses.
    pub fn engage(&mut self, buses: &mut HashMap<GroupBusId, GroupBus>) {
        self.active = true;
        for id in &self.bus_ids {
            if let Some(b) = buses.get_mut(id) {
                b.muted = true;
            }
        }
    }

    /// Release the mute group, un-muting all member buses.
    pub fn release(&mut self, buses: &mut HashMap<GroupBusId, GroupBus>) {
        self.active = false;
        for id in &self.bus_ids {
            if let Some(b) = buses.get_mut(id) {
                b.muted = false;
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// GroupBusManager
// ────────────────────────────────────────────────────────────────────────────

/// Manages a collection of group buses and mute groups.
#[derive(Debug, Default)]
pub struct GroupBusManager {
    buses: HashMap<GroupBusId, GroupBus>,
    mute_groups: Vec<MuteGroup>,
    next_id: u32,
}

impl GroupBusManager {
    /// Create an empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new group bus and return its ID.
    pub fn create_bus(&mut self, name: impl Into<String>) -> GroupBusId {
        let id = GroupBusId(self.next_id);
        self.next_id += 1;
        self.buses.insert(id, GroupBus::new(id, name));
        id
    }

    /// Remove a bus by ID, returning it if it existed.
    pub fn remove_bus(&mut self, id: GroupBusId) -> Option<GroupBus> {
        self.buses.remove(&id)
    }

    /// Get an immutable reference to a bus.
    #[must_use]
    pub fn get_bus(&self, id: GroupBusId) -> Option<&GroupBus> {
        self.buses.get(&id)
    }

    /// Get a mutable reference to a bus.
    pub fn get_bus_mut(&mut self, id: GroupBusId) -> Option<&mut GroupBus> {
        self.buses.get_mut(&id)
    }

    /// Number of group buses.
    #[must_use]
    pub fn bus_count(&self) -> usize {
        self.buses.len()
    }

    /// Add a mute group and return its index.
    pub fn add_mute_group(&mut self, group: MuteGroup) -> usize {
        let idx = self.mute_groups.len();
        self.mute_groups.push(group);
        idx
    }

    /// Engage mute group `index`.
    pub fn engage_mute_group(&mut self, index: usize) {
        if index < self.mute_groups.len() {
            let ids: Vec<_> = self.mute_groups[index].bus_ids.iter().copied().collect();
            self.mute_groups[index].active = true;
            for id in ids {
                if let Some(b) = self.buses.get_mut(&id) {
                    b.muted = true;
                }
            }
        }
    }

    /// Release mute group `index`.
    pub fn release_mute_group(&mut self, index: usize) {
        if index < self.mute_groups.len() {
            let ids: Vec<_> = self.mute_groups[index].bus_ids.iter().copied().collect();
            self.mute_groups[index].active = false;
            for id in ids {
                if let Some(b) = self.buses.get_mut(&id) {
                    b.muted = false;
                }
            }
        }
    }

    /// Collect the effective output gain of all buses, keyed by ID.
    #[must_use]
    pub fn effective_gains(&self) -> HashMap<GroupBusId, f32> {
        self.buses
            .iter()
            .map(|(&id, bus)| (id, bus.effective_gain()))
            .collect()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_bus_default_state() {
        let bus = GroupBus::new(GroupBusId(1), "Drums");
        assert_eq!(bus.fader, 1.0);
        assert!(!bus.muted);
        assert!(!bus.soloed);
        assert!(bus.members.is_empty());
    }

    #[test]
    fn test_group_bus_add_remove_member() {
        let mut bus = GroupBus::new(GroupBusId(0), "Strings");
        bus.add_member(5);
        assert!(bus.has_member(5));
        bus.remove_member(5);
        assert!(!bus.has_member(5));
    }

    #[test]
    fn test_group_bus_set_fader_clamped() {
        let mut bus = GroupBus::new(GroupBusId(0), "Brass");
        bus.set_fader(3.0); // above max
        assert!((bus.fader - 2.0).abs() < f32::EPSILON);
        bus.set_fader(-1.0); // below min
        assert!((bus.fader - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_group_bus_toggle_mute() {
        let mut bus = GroupBus::new(GroupBusId(0), "Woodwind");
        assert!(!bus.muted);
        let state = bus.toggle_mute();
        assert!(state);
        assert!(bus.muted);
        let state2 = bus.toggle_mute();
        assert!(!state2);
    }

    #[test]
    fn test_effective_gain_muted() {
        let mut bus = GroupBus::new(GroupBusId(0), "VX");
        bus.set_fader(1.0);
        bus.muted = true;
        assert!((bus.effective_gain() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_effective_gain_unmuted() {
        let mut bus = GroupBus::new(GroupBusId(0), "VX");
        bus.set_fader(0.75);
        assert!((bus.effective_gain() - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn test_group_bus_display_id() {
        let id = GroupBusId(42);
        assert_eq!(id.to_string(), "GroupBus(42)");
    }

    #[test]
    fn test_manager_create_bus_increments_id() {
        let mut mgr = GroupBusManager::new();
        let id1 = mgr.create_bus("A");
        let id2 = mgr.create_bus("B");
        assert_ne!(id1, id2);
        assert_eq!(mgr.bus_count(), 2);
    }

    #[test]
    fn test_manager_remove_bus() {
        let mut mgr = GroupBusManager::new();
        let id = mgr.create_bus("Temp");
        assert!(mgr.remove_bus(id).is_some());
        assert_eq!(mgr.bus_count(), 0);
        assert!(mgr.remove_bus(id).is_none());
    }

    #[test]
    fn test_manager_get_bus_mut() {
        let mut mgr = GroupBusManager::new();
        let id = mgr.create_bus("Keys");
        mgr.get_bus_mut(id)
            .expect("get_bus_mut should succeed")
            .set_fader(0.5);
        assert!(
            (mgr.get_bus(id).expect("get_bus should succeed").fader - 0.5).abs() < f32::EPSILON
        );
    }

    #[test]
    fn test_manager_engage_release_mute_group() {
        let mut mgr = GroupBusManager::new();
        let id1 = mgr.create_bus("Bus1");
        let id2 = mgr.create_bus("Bus2");

        let mut mg = MuteGroup::new();
        mg.add_bus(id1);
        mg.add_bus(id2);
        let idx = mgr.add_mute_group(mg);

        mgr.engage_mute_group(idx);
        assert!(mgr.get_bus(id1).expect("get_bus should succeed").muted);
        assert!(mgr.get_bus(id2).expect("get_bus should succeed").muted);

        mgr.release_mute_group(idx);
        assert!(!mgr.get_bus(id1).expect("get_bus should succeed").muted);
        assert!(!mgr.get_bus(id2).expect("get_bus should succeed").muted);
    }

    #[test]
    fn test_effective_gains_map() {
        let mut mgr = GroupBusManager::new();
        let id = mgr.create_bus("Perc");
        mgr.get_bus_mut(id)
            .expect("get_bus_mut should succeed")
            .set_fader(0.8);
        let gains = mgr.effective_gains();
        assert!((gains[&id] - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_colour_tag_optional() {
        let mut bus = GroupBus::new(GroupBusId(0), "FX");
        assert!(bus.colour.is_none());
        bus.colour = Some("#FF0000".to_string());
        assert_eq!(bus.colour.as_deref(), Some("#FF0000"));
    }
}
