#![allow(dead_code)]
//! Automation system state snapshot capture and restore.
//!
//! Provides the ability to capture full snapshots of broadcast automation state,
//! enabling checkpoint/restore, state comparison, and rollback operations.

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Unique identifier for a state snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SnapshotId(u64);

impl SnapshotId {
    /// Create a new snapshot ID from a raw value.
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Return the raw numeric value.
    pub fn value(self) -> u64 {
        self.0
    }
}

/// Represents the status of a single channel at snapshot time.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelSnapshot {
    /// Channel identifier.
    pub channel_id: String,
    /// Whether the channel is currently on-air.
    pub on_air: bool,
    /// Currently playing item ID, if any.
    pub current_item: Option<String>,
    /// Elapsed time in the current item (milliseconds).
    pub elapsed_ms: u64,
    /// Number of queued items remaining.
    pub queued_items: usize,
    /// Channel-level metadata key-value pairs.
    pub metadata: HashMap<String, String>,
}

impl ChannelSnapshot {
    /// Create a default off-air channel snapshot.
    pub fn off_air(channel_id: &str) -> Self {
        Self {
            channel_id: channel_id.to_string(),
            on_air: false,
            current_item: None,
            elapsed_ms: 0,
            queued_items: 0,
            metadata: HashMap::new(),
        }
    }

    /// Create an on-air channel snapshot.
    pub fn on_air(channel_id: &str, current_item: &str, elapsed_ms: u64) -> Self {
        Self {
            channel_id: channel_id.to_string(),
            on_air: true,
            current_item: Some(current_item.to_string()),
            elapsed_ms,
            queued_items: 0,
            metadata: HashMap::new(),
        }
    }

    /// Return the remaining queue depth.
    pub fn queue_depth(&self) -> usize {
        self.queued_items
    }
}

/// Device status captured in a snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceStatus {
    /// Device is online and operational.
    Online,
    /// Device is offline.
    Offline,
    /// Device has an error condition.
    Error(String),
    /// Device status is unknown.
    Unknown,
}

/// Snapshot of a single device's state.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceSnapshot {
    /// Device identifier.
    pub device_id: String,
    /// Device status.
    pub status: DeviceStatus,
    /// Device type description.
    pub device_type: String,
    /// Additional properties.
    pub properties: HashMap<String, String>,
}

impl DeviceSnapshot {
    /// Create a new device snapshot.
    pub fn new(device_id: &str, device_type: &str, status: DeviceStatus) -> Self {
        Self {
            device_id: device_id.to_string(),
            status,
            device_type: device_type.to_string(),
            properties: HashMap::new(),
        }
    }

    /// Check if the device is online.
    pub fn is_online(&self) -> bool {
        self.status == DeviceStatus::Online
    }
}

/// A full system state snapshot.
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    /// Unique snapshot identifier.
    pub id: SnapshotId,
    /// Timestamp when the snapshot was captured (millis since epoch).
    pub timestamp_ms: u64,
    /// Human-readable label.
    pub label: String,
    /// Channel states.
    pub channels: Vec<ChannelSnapshot>,
    /// Device states.
    pub devices: Vec<DeviceSnapshot>,
    /// System-level key-value properties.
    pub system_properties: HashMap<String, String>,
}

impl StateSnapshot {
    /// Create a new empty snapshot with the given ID and label.
    pub fn new(id: SnapshotId, label: &str) -> Self {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;
        Self {
            id,
            timestamp_ms,
            label: label.to_string(),
            channels: Vec::new(),
            devices: Vec::new(),
            system_properties: HashMap::new(),
        }
    }

    /// Add a channel snapshot.
    pub fn add_channel(&mut self, channel: ChannelSnapshot) {
        self.channels.push(channel);
    }

    /// Add a device snapshot.
    pub fn add_device(&mut self, device: DeviceSnapshot) {
        self.devices.push(device);
    }

    /// Return the number of channels captured.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Return the number of devices captured.
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Find a channel snapshot by ID.
    pub fn find_channel(&self, channel_id: &str) -> Option<&ChannelSnapshot> {
        self.channels.iter().find(|c| c.channel_id == channel_id)
    }

    /// Find a device snapshot by ID.
    pub fn find_device(&self, device_id: &str) -> Option<&DeviceSnapshot> {
        self.devices.iter().find(|d| d.device_id == device_id)
    }

    /// Count how many channels are on-air.
    pub fn on_air_count(&self) -> usize {
        self.channels.iter().filter(|c| c.on_air).count()
    }

    /// Count how many devices are online.
    pub fn online_device_count(&self) -> usize {
        self.devices.iter().filter(|d| d.is_online()).count()
    }
}

/// Result of comparing two snapshots.
#[derive(Debug, Clone)]
pub struct SnapshotDiff {
    /// Channels that were added (present in new but not old).
    pub added_channels: Vec<String>,
    /// Channels that were removed (present in old but not new).
    pub removed_channels: Vec<String>,
    /// Channels whose on-air status changed.
    pub on_air_changes: Vec<(String, bool, bool)>,
    /// Devices whose status changed.
    pub device_status_changes: Vec<(String, DeviceStatus, DeviceStatus)>,
}

impl SnapshotDiff {
    /// Return true if there are no differences.
    pub fn is_empty(&self) -> bool {
        self.added_channels.is_empty()
            && self.removed_channels.is_empty()
            && self.on_air_changes.is_empty()
            && self.device_status_changes.is_empty()
    }

    /// Total number of changes detected.
    pub fn total_changes(&self) -> usize {
        self.added_channels.len()
            + self.removed_channels.len()
            + self.on_air_changes.len()
            + self.device_status_changes.len()
    }
}

/// Compare two snapshots and produce a diff.
pub fn diff_snapshots(old: &StateSnapshot, new: &StateSnapshot) -> SnapshotDiff {
    let old_ids: HashMap<&str, &ChannelSnapshot> = old
        .channels
        .iter()
        .map(|c| (c.channel_id.as_str(), c))
        .collect();
    let new_ids: HashMap<&str, &ChannelSnapshot> = new
        .channels
        .iter()
        .map(|c| (c.channel_id.as_str(), c))
        .collect();

    let added_channels: Vec<String> = new_ids
        .keys()
        .filter(|k| !old_ids.contains_key(*k))
        .map(|k| (*k).to_string())
        .collect();
    let removed_channels: Vec<String> = old_ids
        .keys()
        .filter(|k| !new_ids.contains_key(*k))
        .map(|k| (*k).to_string())
        .collect();

    let mut on_air_changes = Vec::new();
    for (id, old_ch) in &old_ids {
        if let Some(new_ch) = new_ids.get(id) {
            if old_ch.on_air != new_ch.on_air {
                on_air_changes.push(((*id).to_string(), old_ch.on_air, new_ch.on_air));
            }
        }
    }

    let old_devs: HashMap<&str, &DeviceSnapshot> = old
        .devices
        .iter()
        .map(|d| (d.device_id.as_str(), d))
        .collect();
    let new_devs: HashMap<&str, &DeviceSnapshot> = new
        .devices
        .iter()
        .map(|d| (d.device_id.as_str(), d))
        .collect();

    let mut device_status_changes = Vec::new();
    for (id, old_dev) in &old_devs {
        if let Some(new_dev) = new_devs.get(id) {
            if old_dev.status != new_dev.status {
                device_status_changes.push((
                    (*id).to_string(),
                    old_dev.status.clone(),
                    new_dev.status.clone(),
                ));
            }
        }
    }

    SnapshotDiff {
        added_channels,
        removed_channels,
        on_air_changes,
        device_status_changes,
    }
}

/// Manages a history of state snapshots.
pub struct SnapshotStore {
    /// Maximum number of snapshots to keep.
    max_snapshots: usize,
    /// Stored snapshots (newest last).
    snapshots: Vec<StateSnapshot>,
    /// Next snapshot ID counter.
    next_id: u64,
}

impl SnapshotStore {
    /// Create a new snapshot store with the given capacity.
    pub fn new(max_snapshots: usize) -> Self {
        Self {
            max_snapshots: max_snapshots.max(1),
            snapshots: Vec::new(),
            next_id: 1,
        }
    }

    /// Capture a new snapshot with the given label, returning its ID.
    pub fn capture(&mut self, label: &str) -> SnapshotId {
        let id = SnapshotId::new(self.next_id);
        self.next_id += 1;
        let snapshot = StateSnapshot::new(id, label);
        self.snapshots.push(snapshot);
        if self.snapshots.len() > self.max_snapshots {
            self.snapshots.remove(0);
        }
        id
    }

    /// Return the number of stored snapshots.
    pub fn count(&self) -> usize {
        self.snapshots.len()
    }

    /// Get a snapshot by ID.
    pub fn get(&self, id: SnapshotId) -> Option<&StateSnapshot> {
        self.snapshots.iter().find(|s| s.id == id)
    }

    /// Get a mutable snapshot by ID.
    pub fn get_mut(&mut self, id: SnapshotId) -> Option<&mut StateSnapshot> {
        self.snapshots.iter_mut().find(|s| s.id == id)
    }

    /// Get the most recent snapshot.
    pub fn latest(&self) -> Option<&StateSnapshot> {
        self.snapshots.last()
    }

    /// Remove a snapshot by ID. Returns true if found and removed.
    pub fn remove(&mut self, id: SnapshotId) -> bool {
        let len_before = self.snapshots.len();
        self.snapshots.retain(|s| s.id != id);
        self.snapshots.len() < len_before
    }

    /// Clear all snapshots.
    pub fn clear(&mut self) {
        self.snapshots.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_id_roundtrip() {
        let id = SnapshotId::new(42);
        assert_eq!(id.value(), 42);
    }

    #[test]
    fn test_channel_snapshot_off_air() {
        let ch = ChannelSnapshot::off_air("ch1");
        assert_eq!(ch.channel_id, "ch1");
        assert!(!ch.on_air);
        assert!(ch.current_item.is_none());
        assert_eq!(ch.queue_depth(), 0);
    }

    #[test]
    fn test_channel_snapshot_on_air() {
        let ch = ChannelSnapshot::on_air("ch2", "item_001", 5000);
        assert!(ch.on_air);
        assert_eq!(ch.current_item.as_deref(), Some("item_001"));
        assert_eq!(ch.elapsed_ms, 5000);
    }

    #[test]
    fn test_device_snapshot_online() {
        let dev = DeviceSnapshot::new("vtr1", "VTR", DeviceStatus::Online);
        assert!(dev.is_online());
    }

    #[test]
    fn test_device_snapshot_offline() {
        let dev = DeviceSnapshot::new("vtr2", "VTR", DeviceStatus::Offline);
        assert!(!dev.is_online());
    }

    #[test]
    fn test_state_snapshot_add_channels_and_devices() {
        let mut snap = StateSnapshot::new(SnapshotId::new(1), "test");
        snap.add_channel(ChannelSnapshot::off_air("ch1"));
        snap.add_channel(ChannelSnapshot::on_air("ch2", "item", 100));
        snap.add_device(DeviceSnapshot::new("dev1", "VTR", DeviceStatus::Online));
        assert_eq!(snap.channel_count(), 2);
        assert_eq!(snap.device_count(), 1);
        assert_eq!(snap.on_air_count(), 1);
        assert_eq!(snap.online_device_count(), 1);
    }

    #[test]
    fn test_state_snapshot_find_channel() {
        let mut snap = StateSnapshot::new(SnapshotId::new(1), "find");
        snap.add_channel(ChannelSnapshot::off_air("alpha"));
        assert!(snap.find_channel("alpha").is_some());
        assert!(snap.find_channel("beta").is_none());
    }

    #[test]
    fn test_state_snapshot_find_device() {
        let mut snap = StateSnapshot::new(SnapshotId::new(1), "find_dev");
        snap.add_device(DeviceSnapshot::new(
            "router1",
            "Router",
            DeviceStatus::Online,
        ));
        assert!(snap.find_device("router1").is_some());
        assert!(snap.find_device("router2").is_none());
    }

    #[test]
    fn test_diff_snapshots_no_change() {
        let old = StateSnapshot::new(SnapshotId::new(1), "old");
        let new = StateSnapshot::new(SnapshotId::new(2), "new");
        let diff = diff_snapshots(&old, &new);
        assert!(diff.is_empty());
        assert_eq!(diff.total_changes(), 0);
    }

    #[test]
    fn test_diff_snapshots_added_channel() {
        let old = StateSnapshot::new(SnapshotId::new(1), "old");
        let mut new = StateSnapshot::new(SnapshotId::new(2), "new");
        new.add_channel(ChannelSnapshot::off_air("ch1"));
        let diff = diff_snapshots(&old, &new);
        assert_eq!(diff.added_channels.len(), 1);
        assert_eq!(diff.total_changes(), 1);
    }

    #[test]
    fn test_diff_snapshots_on_air_change() {
        let mut old = StateSnapshot::new(SnapshotId::new(1), "old");
        old.add_channel(ChannelSnapshot::off_air("ch1"));
        let mut new = StateSnapshot::new(SnapshotId::new(2), "new");
        new.add_channel(ChannelSnapshot::on_air("ch1", "item", 0));
        let diff = diff_snapshots(&old, &new);
        assert_eq!(diff.on_air_changes.len(), 1);
        let (ref ch, was_on, is_on) = diff.on_air_changes[0];
        assert_eq!(ch, "ch1");
        assert!(!was_on);
        assert!(is_on);
    }

    #[test]
    fn test_snapshot_store_capture_and_get() {
        let mut store = SnapshotStore::new(10);
        let id = store.capture("first");
        assert_eq!(store.count(), 1);
        assert!(store.get(id).is_some());
        assert_eq!(store.get(id).expect("get should succeed").label, "first");
    }

    #[test]
    fn test_snapshot_store_eviction() {
        let mut store = SnapshotStore::new(3);
        let id1 = store.capture("a");
        let _id2 = store.capture("b");
        let _id3 = store.capture("c");
        let _id4 = store.capture("d");
        assert_eq!(store.count(), 3);
        // oldest (id1) should have been evicted
        assert!(store.get(id1).is_none());
    }

    #[test]
    fn test_snapshot_store_remove() {
        let mut store = SnapshotStore::new(10);
        let id = store.capture("to_remove");
        assert!(store.remove(id));
        assert_eq!(store.count(), 0);
        assert!(!store.remove(id));
    }

    #[test]
    fn test_snapshot_store_latest() {
        let mut store = SnapshotStore::new(10);
        assert!(store.latest().is_none());
        store.capture("first");
        store.capture("second");
        assert_eq!(
            store.latest().expect("latest should succeed").label,
            "second"
        );
    }

    #[test]
    fn test_snapshot_store_clear() {
        let mut store = SnapshotStore::new(10);
        store.capture("a");
        store.capture("b");
        store.clear();
        assert_eq!(store.count(), 0);
    }
}
