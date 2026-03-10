#![allow(dead_code)]
//! AAF track grouping and multi-track management.
//!
//! Provides facilities for organizing tracks into logical groups such as
//! stereo pairs, 5.1 surround bundles, multi-camera angles, and custom
//! groupings used in professional post-production workflows.
//!
//! Track groups help maintain channel assignments, synchronization
//! relationships, and editing behavior when tracks are linked.

use std::collections::HashMap;
use uuid::Uuid;

/// Type of track group describing the relationship between member tracks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GroupType {
    /// Stereo pair (left/right).
    StereoPair,
    /// 5.1 surround sound bundle (L, R, C, LFE, Ls, Rs).
    Surround51,
    /// 7.1 surround sound bundle.
    Surround71,
    /// Multi-camera angle group.
    MultiCam,
    /// Linked audio/video pair.
    AvLink,
    /// Custom user-defined group.
    Custom,
}

impl GroupType {
    /// Returns the expected number of channels for audio group types.
    #[must_use]
    pub const fn expected_channel_count(self) -> Option<usize> {
        match self {
            Self::StereoPair => Some(2),
            Self::Surround51 => Some(6),
            Self::Surround71 => Some(8),
            _ => None,
        }
    }

    /// Whether this group type enforces a fixed channel count.
    #[must_use]
    pub const fn has_fixed_size(self) -> bool {
        matches!(self, Self::StereoPair | Self::Surround51 | Self::Surround71)
    }

    /// Returns a human-readable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::StereoPair => "Stereo",
            Self::Surround51 => "5.1 Surround",
            Self::Surround71 => "7.1 Surround",
            Self::MultiCam => "Multi-Camera",
            Self::AvLink => "A/V Link",
            Self::Custom => "Custom",
        }
    }
}

/// Role of a track within a group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChannelRole {
    /// Left channel.
    Left,
    /// Right channel.
    Right,
    /// Center channel.
    Center,
    /// Low-frequency effects channel.
    Lfe,
    /// Left surround channel.
    LeftSurround,
    /// Right surround channel.
    RightSurround,
    /// Left back surround (7.1).
    LeftBack,
    /// Right back surround (7.1).
    RightBack,
    /// Mono channel.
    Mono,
    /// Primary video.
    PrimaryVideo,
    /// Camera angle N (multi-cam).
    CameraAngle(u32),
    /// Unassigned / generic role.
    Unassigned,
}

impl ChannelRole {
    /// Returns the standard 5.1 channel layout in order.
    #[must_use]
    pub fn surround_51_layout() -> Vec<Self> {
        vec![
            Self::Left,
            Self::Right,
            Self::Center,
            Self::Lfe,
            Self::LeftSurround,
            Self::RightSurround,
        ]
    }

    /// Returns the standard 7.1 channel layout in order.
    #[must_use]
    pub fn surround_71_layout() -> Vec<Self> {
        vec![
            Self::Left,
            Self::Right,
            Self::Center,
            Self::Lfe,
            Self::LeftSurround,
            Self::RightSurround,
            Self::LeftBack,
            Self::RightBack,
        ]
    }

    /// Whether this role is a surround channel.
    #[must_use]
    pub const fn is_surround(self) -> bool {
        matches!(
            self,
            Self::LeftSurround | Self::RightSurround | Self::LeftBack | Self::RightBack
        )
    }
}

/// A member track within a group, linking a track ID to its role.
#[derive(Debug, Clone)]
pub struct GroupMember {
    /// Unique track identifier.
    pub track_id: u32,
    /// Mob ID that owns this track.
    pub mob_id: Uuid,
    /// Role of this track within the group.
    pub role: ChannelRole,
    /// Optional display name override.
    pub display_name: Option<String>,
    /// Whether this member is solo'd.
    pub solo: bool,
    /// Whether this member is muted.
    pub muted: bool,
}

impl GroupMember {
    /// Create a new group member.
    #[must_use]
    pub fn new(track_id: u32, mob_id: Uuid, role: ChannelRole) -> Self {
        Self {
            track_id,
            mob_id,
            role,
            display_name: None,
            solo: false,
            muted: false,
        }
    }

    /// Set the display name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = Some(name.into());
        self
    }
}

/// A logical group of related tracks.
#[derive(Debug, Clone)]
pub struct TrackGroup {
    /// Unique identifier for this group.
    pub group_id: Uuid,
    /// Human-readable name for the group.
    pub name: String,
    /// Type of group.
    pub group_type: GroupType,
    /// Member tracks.
    members: Vec<GroupMember>,
    /// Whether edits to one member apply to all.
    pub gang_editing: bool,
    /// Color tag for UI display (ARGB hex).
    pub color_tag: Option<u32>,
}

impl TrackGroup {
    /// Create a new track group.
    #[must_use]
    pub fn new(name: impl Into<String>, group_type: GroupType) -> Self {
        Self {
            group_id: Uuid::new_v4(),
            name: name.into(),
            group_type,
            members: Vec::new(),
            gang_editing: true,
            color_tag: None,
        }
    }

    /// Create a stereo pair group with left/right members.
    #[must_use]
    pub fn stereo_pair(
        name: impl Into<String>,
        left_track: u32,
        right_track: u32,
        mob_id: Uuid,
    ) -> Self {
        let mut group = Self::new(name, GroupType::StereoPair);
        group
            .members
            .push(GroupMember::new(left_track, mob_id, ChannelRole::Left));
        group
            .members
            .push(GroupMember::new(right_track, mob_id, ChannelRole::Right));
        group
    }

    /// Add a member to this group.
    pub fn add_member(&mut self, member: GroupMember) {
        self.members.push(member);
    }

    /// Remove a member by track ID.
    pub fn remove_member(&mut self, track_id: u32) -> bool {
        let before = self.members.len();
        self.members.retain(|m| m.track_id != track_id);
        self.members.len() < before
    }

    /// Get all members.
    #[must_use]
    pub fn members(&self) -> &[GroupMember] {
        &self.members
    }

    /// Get number of members.
    #[must_use]
    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    /// Find a member by track ID.
    #[must_use]
    pub fn find_member(&self, track_id: u32) -> Option<&GroupMember> {
        self.members.iter().find(|m| m.track_id == track_id)
    }

    /// Check if the group has the correct number of members for its type.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        match self.group_type.expected_channel_count() {
            Some(expected) => self.members.len() == expected,
            None => !self.members.is_empty(),
        }
    }

    /// Validate the group structure.
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if self.members.is_empty() {
            issues.push("Group has no members".to_string());
        }

        if let Some(expected) = self.group_type.expected_channel_count() {
            if self.members.len() != expected {
                issues.push(format!(
                    "Expected {} members for {}, got {}",
                    expected,
                    self.group_type.label(),
                    self.members.len()
                ));
            }
        }

        // Check for duplicate track IDs
        let mut seen = std::collections::HashSet::new();
        for member in &self.members {
            if !seen.insert(member.track_id) {
                issues.push(format!("Duplicate track ID: {}", member.track_id));
            }
        }

        issues
    }

    /// Mute all members.
    pub fn mute_all(&mut self) {
        for member in &mut self.members {
            member.muted = true;
        }
    }

    /// Unmute all members.
    pub fn unmute_all(&mut self) {
        for member in &mut self.members {
            member.muted = false;
        }
    }

    /// Solo a specific member and unsolo all others.
    pub fn solo_member(&mut self, track_id: u32) {
        for member in &mut self.members {
            member.solo = member.track_id == track_id;
        }
    }

    /// Get all muted member track IDs.
    #[must_use]
    pub fn muted_tracks(&self) -> Vec<u32> {
        self.members
            .iter()
            .filter(|m| m.muted)
            .map(|m| m.track_id)
            .collect()
    }
}

/// Registry managing multiple track groups.
#[derive(Debug, Clone)]
pub struct TrackGroupRegistry {
    /// All registered groups indexed by group ID.
    groups: HashMap<Uuid, TrackGroup>,
    /// Index mapping track IDs to group IDs for fast lookup.
    track_index: HashMap<u32, Vec<Uuid>>,
}

impl TrackGroupRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            groups: HashMap::new(),
            track_index: HashMap::new(),
        }
    }

    /// Register a new track group.
    pub fn register(&mut self, group: TrackGroup) {
        let gid = group.group_id;
        for member in group.members() {
            self.track_index
                .entry(member.track_id)
                .or_default()
                .push(gid);
        }
        self.groups.insert(gid, group);
    }

    /// Remove a group by ID.
    pub fn remove(&mut self, group_id: &Uuid) -> Option<TrackGroup> {
        if let Some(group) = self.groups.remove(group_id) {
            // Clean up track index
            for member in group.members() {
                if let Some(ids) = self.track_index.get_mut(&member.track_id) {
                    ids.retain(|id| id != group_id);
                }
            }
            Some(group)
        } else {
            None
        }
    }

    /// Get a group by ID.
    #[must_use]
    pub fn get(&self, group_id: &Uuid) -> Option<&TrackGroup> {
        self.groups.get(group_id)
    }

    /// Find all groups containing a given track ID.
    #[must_use]
    pub fn groups_for_track(&self, track_id: u32) -> Vec<&TrackGroup> {
        self.track_index
            .get(&track_id)
            .map(|ids| ids.iter().filter_map(|id| self.groups.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get the total number of groups.
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Get all groups.
    #[must_use]
    pub fn all_groups(&self) -> Vec<&TrackGroup> {
        self.groups.values().collect()
    }
}

impl Default for TrackGroupRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_type_channel_count() {
        assert_eq!(GroupType::StereoPair.expected_channel_count(), Some(2));
        assert_eq!(GroupType::Surround51.expected_channel_count(), Some(6));
        assert_eq!(GroupType::Surround71.expected_channel_count(), Some(8));
        assert_eq!(GroupType::MultiCam.expected_channel_count(), None);
        assert_eq!(GroupType::Custom.expected_channel_count(), None);
    }

    #[test]
    fn test_group_type_has_fixed_size() {
        assert!(GroupType::StereoPair.has_fixed_size());
        assert!(GroupType::Surround51.has_fixed_size());
        assert!(!GroupType::MultiCam.has_fixed_size());
        assert!(!GroupType::Custom.has_fixed_size());
    }

    #[test]
    fn test_group_type_label() {
        assert_eq!(GroupType::StereoPair.label(), "Stereo");
        assert_eq!(GroupType::Surround51.label(), "5.1 Surround");
        assert_eq!(GroupType::Surround71.label(), "7.1 Surround");
        assert_eq!(GroupType::MultiCam.label(), "Multi-Camera");
    }

    #[test]
    fn test_channel_role_surround_layouts() {
        assert_eq!(ChannelRole::surround_51_layout().len(), 6);
        assert_eq!(ChannelRole::surround_71_layout().len(), 8);
    }

    #[test]
    fn test_channel_role_is_surround() {
        assert!(ChannelRole::LeftSurround.is_surround());
        assert!(ChannelRole::RightSurround.is_surround());
        assert!(ChannelRole::LeftBack.is_surround());
        assert!(!ChannelRole::Left.is_surround());
        assert!(!ChannelRole::Center.is_surround());
    }

    #[test]
    fn test_group_member_creation() {
        let mob_id = Uuid::new_v4();
        let member = GroupMember::new(1, mob_id, ChannelRole::Left).with_name("Left Speaker");
        assert_eq!(member.track_id, 1);
        assert_eq!(member.role, ChannelRole::Left);
        assert_eq!(member.display_name.as_deref(), Some("Left Speaker"));
        assert!(!member.solo);
        assert!(!member.muted);
    }

    #[test]
    fn test_stereo_pair_creation() {
        let mob_id = Uuid::new_v4();
        let group = TrackGroup::stereo_pair("Dialogue", 1, 2, mob_id);
        assert_eq!(group.group_type, GroupType::StereoPair);
        assert_eq!(group.member_count(), 2);
        assert!(group.is_complete());
        assert!(group.validate().is_empty());
    }

    #[test]
    fn test_track_group_add_remove() {
        let mob_id = Uuid::new_v4();
        let mut group = TrackGroup::new("Custom Group", GroupType::Custom);
        group.add_member(GroupMember::new(1, mob_id, ChannelRole::Mono));
        group.add_member(GroupMember::new(2, mob_id, ChannelRole::Mono));
        assert_eq!(group.member_count(), 2);

        assert!(group.remove_member(1));
        assert_eq!(group.member_count(), 1);
        assert!(!group.remove_member(99));
    }

    #[test]
    fn test_track_group_find_member() {
        let mob_id = Uuid::new_v4();
        let group = TrackGroup::stereo_pair("Stereo", 1, 2, mob_id);
        let left = group.find_member(1).expect("left should be valid");
        assert_eq!(left.role, ChannelRole::Left);
        assert!(group.find_member(99).is_none());
    }

    #[test]
    fn test_track_group_validate_incomplete() {
        let mob_id = Uuid::new_v4();
        let mut group = TrackGroup::new("Bad 5.1", GroupType::Surround51);
        group.add_member(GroupMember::new(1, mob_id, ChannelRole::Left));
        let issues = group.validate();
        assert!(!issues.is_empty());
        assert!(!group.is_complete());
    }

    #[test]
    fn test_track_group_mute_unmute() {
        let mob_id = Uuid::new_v4();
        let mut group = TrackGroup::stereo_pair("Stereo", 1, 2, mob_id);
        group.mute_all();
        assert_eq!(group.muted_tracks().len(), 2);
        group.unmute_all();
        assert_eq!(group.muted_tracks().len(), 0);
    }

    #[test]
    fn test_track_group_solo() {
        let mob_id = Uuid::new_v4();
        let mut group = TrackGroup::stereo_pair("Stereo", 1, 2, mob_id);
        group.solo_member(1);
        let members = group.members();
        assert!(members[0].solo);
        assert!(!members[1].solo);
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut registry = TrackGroupRegistry::new();
        let mob_id = Uuid::new_v4();
        let group = TrackGroup::stereo_pair("Stereo", 1, 2, mob_id);
        let gid = group.group_id;
        registry.register(group);

        assert_eq!(registry.group_count(), 1);
        assert!(registry.get(&gid).is_some());
    }

    #[test]
    fn test_registry_groups_for_track() {
        let mut registry = TrackGroupRegistry::new();
        let mob_id = Uuid::new_v4();
        let group = TrackGroup::stereo_pair("Stereo", 1, 2, mob_id);
        registry.register(group);

        let groups = registry.groups_for_track(1);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].group_type, GroupType::StereoPair);

        let empty = registry.groups_for_track(99);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_registry_remove() {
        let mut registry = TrackGroupRegistry::new();
        let mob_id = Uuid::new_v4();
        let group = TrackGroup::stereo_pair("Stereo", 1, 2, mob_id);
        let gid = group.group_id;
        registry.register(group);
        assert_eq!(registry.group_count(), 1);

        let removed = registry.remove(&gid);
        assert!(removed.is_some());
        assert_eq!(registry.group_count(), 0);
    }

    #[test]
    fn test_validate_duplicate_tracks() {
        let mob_id = Uuid::new_v4();
        let mut group = TrackGroup::new("Bad Group", GroupType::Custom);
        group.add_member(GroupMember::new(1, mob_id, ChannelRole::Left));
        group.add_member(GroupMember::new(1, mob_id, ChannelRole::Right));
        let issues = group.validate();
        assert!(issues.iter().any(|i| i.contains("Duplicate")));
    }
}
