//! Version history tracking for archived media assets.
//!
//! Provides a complete change-log for every asset in the archive: who changed
//! what, when, and why. Supports diffing between versions, rollback, and
//! retention-aware pruning.

#![allow(dead_code)]

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// The kind of change recorded for a version entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeKind {
    /// Asset was initially ingested.
    Ingest,
    /// Metadata was updated.
    MetadataUpdate,
    /// The media essence (file content) was replaced.
    EssenceReplace,
    /// Asset was re-encoded or transcoded.
    Transcode,
    /// Checksums were recomputed / fixity re-verified.
    FixityRefresh,
    /// Asset was migrated to a new format.
    FormatMigration,
    /// Asset was restored from backup.
    Restore,
    /// A custom / user-defined change.
    Custom,
}

impl fmt::Display for ChangeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Ingest => "ingest",
            Self::MetadataUpdate => "metadata_update",
            Self::EssenceReplace => "essence_replace",
            Self::Transcode => "transcode",
            Self::FixityRefresh => "fixity_refresh",
            Self::FormatMigration => "format_migration",
            Self::Restore => "restore",
            Self::Custom => "custom",
        };
        write!(f, "{label}")
    }
}

/// A single version entry in an asset's history.
#[derive(Debug, Clone)]
pub struct VersionEntry {
    /// Monotonically increasing version number (1-based).
    pub version: u64,
    /// Unix timestamp (seconds) when the version was created.
    pub timestamp: u64,
    /// The kind of change that produced this version.
    pub change_kind: ChangeKind,
    /// Human-readable description of the change.
    pub description: String,
    /// Identity of the actor who made the change.
    pub actor: String,
    /// SHA-256 hex digest of the asset at this version (if available).
    pub checksum: Option<String>,
    /// Size of the asset in bytes at this version (if available).
    pub size_bytes: Option<u64>,
    /// Optional key/value metadata snapshot.
    pub metadata: HashMap<String, String>,
}

impl VersionEntry {
    /// Create a new version entry with required fields.
    pub fn new(version: u64, timestamp: u64, change_kind: ChangeKind, actor: &str) -> Self {
        Self {
            version,
            timestamp,
            change_kind,
            description: String::new(),
            actor: actor.to_string(),
            checksum: None,
            size_bytes: None,
            metadata: HashMap::new(),
        }
    }

    /// Set the description.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Set the checksum.
    pub fn with_checksum(mut self, cs: &str) -> Self {
        self.checksum = Some(cs.to_string());
        self
    }

    /// Set the size in bytes.
    pub fn with_size(mut self, bytes: u64) -> Self {
        self.size_bytes = Some(bytes);
        self
    }

    /// Add a metadata key/value pair.
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// Result of comparing two version entries.
#[derive(Debug, Clone)]
pub struct VersionDiff {
    /// Version numbers being compared (older, newer).
    pub versions: (u64, u64),
    /// Whether the checksum changed.
    pub checksum_changed: bool,
    /// Whether the file size changed.
    pub size_changed: bool,
    /// Metadata keys that were added in the newer version.
    pub added_metadata: Vec<String>,
    /// Metadata keys that were removed in the newer version.
    pub removed_metadata: Vec<String>,
    /// Metadata keys whose values changed.
    pub changed_metadata: Vec<String>,
}

/// Complete version history for a single asset.
#[derive(Debug, Clone)]
pub struct VersionHistory {
    /// Unique asset identifier.
    pub asset_id: String,
    /// Ordered list of version entries (oldest first).
    entries: Vec<VersionEntry>,
}

impl VersionHistory {
    /// Create a new, empty version history for the given asset.
    pub fn new(asset_id: &str) -> Self {
        Self {
            asset_id: asset_id.to_string(),
            entries: Vec::new(),
        }
    }

    /// Append a version entry. Returns the version number assigned.
    pub fn push(&mut self, mut entry: VersionEntry) -> u64 {
        let next_ver = self.entries.len() as u64 + 1;
        entry.version = next_ver;
        self.entries.push(entry);
        next_ver
    }

    /// Number of versions recorded.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no versions have been recorded.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get a specific version (1-based).
    pub fn get(&self, version: u64) -> Option<&VersionEntry> {
        if version == 0 || version as usize > self.entries.len() {
            return None;
        }
        Some(&self.entries[version as usize - 1])
    }

    /// Get the latest version entry.
    pub fn latest(&self) -> Option<&VersionEntry> {
        self.entries.last()
    }

    /// Iterate over all entries.
    pub fn iter(&self) -> impl Iterator<Item = &VersionEntry> {
        self.entries.iter()
    }

    /// Filter entries by change kind.
    pub fn filter_by_kind(&self, kind: ChangeKind) -> Vec<&VersionEntry> {
        self.entries
            .iter()
            .filter(|e| e.change_kind == kind)
            .collect()
    }

    /// Filter entries by actor.
    pub fn filter_by_actor(&self, actor: &str) -> Vec<&VersionEntry> {
        self.entries.iter().filter(|e| e.actor == actor).collect()
    }

    /// Filter entries within a timestamp range (inclusive).
    pub fn filter_by_time_range(&self, start: u64, end: u64) -> Vec<&VersionEntry> {
        self.entries
            .iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .collect()
    }

    /// Compute a diff between two versions.
    pub fn diff(&self, older: u64, newer: u64) -> Option<VersionDiff> {
        let old_entry = self.get(older)?;
        let new_entry = self.get(newer)?;

        let checksum_changed = old_entry.checksum != new_entry.checksum;
        let size_changed = old_entry.size_bytes != new_entry.size_bytes;

        let old_keys: std::collections::HashSet<_> = old_entry.metadata.keys().collect();
        let new_keys: std::collections::HashSet<_> = new_entry.metadata.keys().collect();

        let added_metadata: Vec<String> = new_keys
            .difference(&old_keys)
            .map(|k| (*k).clone())
            .collect();
        let removed_metadata: Vec<String> = old_keys
            .difference(&new_keys)
            .map(|k| (*k).clone())
            .collect();
        let changed_metadata: Vec<String> = old_keys
            .intersection(&new_keys)
            .filter(|k| old_entry.metadata.get(**k) != new_entry.metadata.get(**k))
            .map(|k| (*k).clone())
            .collect();

        Some(VersionDiff {
            versions: (older, newer),
            checksum_changed,
            size_changed,
            added_metadata,
            removed_metadata,
            changed_metadata,
        })
    }

    /// Prune old entries keeping only the latest `keep` versions.
    /// Returns the number of entries removed.
    pub fn prune(&mut self, keep: usize) -> usize {
        if self.entries.len() <= keep {
            return 0;
        }
        let remove_count = self.entries.len() - keep;
        self.entries.drain(..remove_count);
        // Re-number remaining entries
        for (i, entry) in self.entries.iter_mut().enumerate() {
            entry.version = i as u64 + 1;
        }
        remove_count
    }

    /// Serialize the history to a JSON string.
    pub fn to_json(&self) -> String {
        let mut parts = Vec::new();
        for e in &self.entries {
            let cs = e.checksum.as_deref().unwrap_or("");
            let sz = e.size_bytes.map_or("null".to_string(), |s| s.to_string());
            parts.push(format!(
                r#"{{"version":{},"timestamp":{},"kind":"{}","actor":"{}","description":"{}","checksum":"{}","size":{}}}"#,
                e.version, e.timestamp, e.change_kind, e.actor, e.description, cs, sz
            ));
        }
        format!(
            r#"{{"asset_id":"{}","versions":[{}]}}"#,
            self.asset_id,
            parts.join(",")
        )
    }

    /// Total number of distinct actors who made changes.
    pub fn unique_actors(&self) -> usize {
        let actors: std::collections::HashSet<_> = self.entries.iter().map(|e| &e.actor).collect();
        actors.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_history() -> VersionHistory {
        let mut h = VersionHistory::new("asset-001");
        h.push(
            VersionEntry::new(0, 1000, ChangeKind::Ingest, "alice")
                .with_description("Initial ingest")
                .with_checksum("aaa")
                .with_size(100),
        );
        h.push(
            VersionEntry::new(0, 2000, ChangeKind::MetadataUpdate, "bob")
                .with_description("Updated title")
                .with_checksum("aaa")
                .with_size(100)
                .with_metadata("title", "New Title"),
        );
        h.push(
            VersionEntry::new(0, 3000, ChangeKind::EssenceReplace, "alice")
                .with_description("Re-encoded to ProRes")
                .with_checksum("bbb")
                .with_size(200),
        );
        h
    }

    #[test]
    fn test_push_and_len() {
        let h = sample_history();
        assert_eq!(h.len(), 3);
        assert!(!h.is_empty());
    }

    #[test]
    fn test_version_numbering() {
        let h = sample_history();
        assert_eq!(h.get(1).expect("get should succeed").version, 1);
        assert_eq!(h.get(2).expect("get should succeed").version, 2);
        assert_eq!(h.get(3).expect("get should succeed").version, 3);
    }

    #[test]
    fn test_get_invalid_version() {
        let h = sample_history();
        assert!(h.get(0).is_none());
        assert!(h.get(99).is_none());
    }

    #[test]
    fn test_latest() {
        let h = sample_history();
        let latest = h.latest().expect("latest should be valid");
        assert_eq!(latest.version, 3);
        assert_eq!(latest.change_kind, ChangeKind::EssenceReplace);
    }

    #[test]
    fn test_empty_history() {
        let h = VersionHistory::new("empty");
        assert!(h.is_empty());
        assert!(h.latest().is_none());
    }

    #[test]
    fn test_filter_by_kind() {
        let h = sample_history();
        let ingests = h.filter_by_kind(ChangeKind::Ingest);
        assert_eq!(ingests.len(), 1);
        assert_eq!(ingests[0].actor, "alice");
    }

    #[test]
    fn test_filter_by_actor() {
        let h = sample_history();
        let alice = h.filter_by_actor("alice");
        assert_eq!(alice.len(), 2);
    }

    #[test]
    fn test_filter_by_time_range() {
        let h = sample_history();
        let range = h.filter_by_time_range(1500, 2500);
        assert_eq!(range.len(), 1);
        assert_eq!(range[0].version, 2);
    }

    #[test]
    fn test_diff_checksum_changed() {
        let h = sample_history();
        let d = h.diff(1, 3).expect("d should be valid");
        assert!(d.checksum_changed);
        assert!(d.size_changed);
    }

    #[test]
    fn test_diff_metadata_added() {
        let h = sample_history();
        let d = h.diff(1, 2).expect("d should be valid");
        assert!(d.added_metadata.contains(&"title".to_string()));
        assert!(!d.checksum_changed);
    }

    #[test]
    fn test_diff_invalid_version() {
        let h = sample_history();
        assert!(h.diff(1, 99).is_none());
    }

    #[test]
    fn test_prune() {
        let mut h = sample_history();
        let removed = h.prune(2);
        assert_eq!(removed, 1);
        assert_eq!(h.len(), 2);
        assert_eq!(
            h.get(1).expect("get should succeed").change_kind,
            ChangeKind::MetadataUpdate
        );
    }

    #[test]
    fn test_prune_no_op() {
        let mut h = sample_history();
        let removed = h.prune(10);
        assert_eq!(removed, 0);
        assert_eq!(h.len(), 3);
    }

    #[test]
    fn test_unique_actors() {
        let h = sample_history();
        assert_eq!(h.unique_actors(), 2);
    }

    #[test]
    fn test_to_json_contains_asset_id() {
        let h = sample_history();
        let json = h.to_json();
        assert!(json.contains("asset-001"));
        assert!(json.contains("\"version\":1"));
    }

    #[test]
    fn test_change_kind_display() {
        assert_eq!(ChangeKind::Ingest.to_string(), "ingest");
        assert_eq!(ChangeKind::Transcode.to_string(), "transcode");
        assert_eq!(ChangeKind::FormatMigration.to_string(), "format_migration");
    }

    #[test]
    fn test_iter() {
        let h = sample_history();
        let versions: Vec<u64> = h.iter().map(|e| e.version).collect();
        assert_eq!(versions, vec![1, 2, 3]);
    }
}
