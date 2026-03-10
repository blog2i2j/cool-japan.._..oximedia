#![allow(dead_code)]
//! Object version management and history tracking.
//!
//! Provides version numbering, tagging, diff summaries, and
//! rollback support for versioned storage objects.

use std::collections::HashMap;

/// Unique version identifier.
pub type VersionId = u64;

/// Version status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VersionStatus {
    /// Active / current version.
    Active,
    /// Superseded by a newer version.
    Superseded,
    /// Soft-deleted (recoverable).
    Deleted,
    /// Archived / cold storage.
    Archived,
}

/// Metadata about a single object version.
#[derive(Debug, Clone)]
pub struct VersionInfo {
    /// Version number (monotonically increasing per object).
    pub version_id: VersionId,
    /// Size of this version in bytes.
    pub size_bytes: u64,
    /// Content hash (hex-encoded).
    pub content_hash: String,
    /// Unix timestamp of version creation.
    pub created_at: u64,
    /// Author or system that created the version.
    pub author: String,
    /// Human-readable change description.
    pub message: String,
    /// Current status.
    pub status: VersionStatus,
    /// User-assigned tags.
    pub tags: Vec<String>,
}

/// Summary of differences between two versions.
#[derive(Debug, Clone, Default)]
pub struct VersionDiff {
    /// Size difference in bytes (positive = newer is larger).
    pub size_delta: i64,
    /// Whether the content hash changed.
    pub content_changed: bool,
    /// Tags added in the newer version.
    pub tags_added: Vec<String>,
    /// Tags removed in the newer version.
    pub tags_removed: Vec<String>,
}

/// Policy for automatic version retention.
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Maximum number of versions to keep per object.
    pub max_versions: Option<u32>,
    /// Maximum age of versions in seconds.
    pub max_age_secs: Option<u64>,
    /// Whether to keep at least one version regardless of policy.
    pub keep_latest: bool,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            max_versions: Some(100),
            max_age_secs: None,
            keep_latest: true,
        }
    }
}

/// Version history for a single object key.
#[derive(Debug)]
pub struct ObjectVersionHistory {
    /// Object key.
    key: String,
    /// Ordered list of versions (oldest first).
    versions: Vec<VersionInfo>,
    /// Next version ID.
    next_id: VersionId,
    /// Retention policy.
    policy: RetentionPolicy,
}

impl ObjectVersionHistory {
    /// Create a new version history for the given key.
    pub fn new(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            versions: Vec::new(),
            next_id: 1,
            policy: RetentionPolicy::default(),
        }
    }

    /// Create with a custom retention policy.
    pub fn with_policy(key: impl Into<String>, policy: RetentionPolicy) -> Self {
        Self {
            key: key.into(),
            versions: Vec::new(),
            next_id: 1,
            policy,
        }
    }

    /// Object key.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Number of versions stored.
    pub fn version_count(&self) -> usize {
        self.versions.len()
    }

    /// Add a new version and return its ID. Previous active version is superseded.
    pub fn add_version(
        &mut self,
        size_bytes: u64,
        content_hash: impl Into<String>,
        author: impl Into<String>,
        message: impl Into<String>,
    ) -> VersionId {
        // Mark previous active version as superseded
        for v in &mut self.versions {
            if v.status == VersionStatus::Active {
                v.status = VersionStatus::Superseded;
            }
        }
        let id = self.next_id;
        self.next_id += 1;
        self.versions.push(VersionInfo {
            version_id: id,
            size_bytes,
            content_hash: content_hash.into(),
            created_at: id * 1000, // simple monotonic timestamp for testing
            author: author.into(),
            message: message.into(),
            status: VersionStatus::Active,
            tags: Vec::new(),
        });
        self.apply_retention();
        id
    }

    /// Get the currently active version.
    pub fn current(&self) -> Option<&VersionInfo> {
        self.versions
            .iter()
            .rev()
            .find(|v| v.status == VersionStatus::Active)
    }

    /// Get a specific version by ID.
    pub fn get(&self, version_id: VersionId) -> Option<&VersionInfo> {
        self.versions.iter().find(|v| v.version_id == version_id)
    }

    /// List all versions (oldest first).
    pub fn list(&self) -> &[VersionInfo] {
        &self.versions
    }

    /// Soft-delete a version.
    pub fn delete_version(&mut self, version_id: VersionId) -> Result<(), VersionError> {
        let v = self
            .versions
            .iter_mut()
            .find(|v| v.version_id == version_id)
            .ok_or(VersionError::NotFound(version_id))?;
        v.status = VersionStatus::Deleted;
        Ok(())
    }

    /// Rollback to a specific version (makes it active, supersedes current).
    pub fn rollback_to(&mut self, version_id: VersionId) -> Result<(), VersionError> {
        // Ensure target version exists and is not deleted
        let target = self
            .versions
            .iter()
            .find(|v| v.version_id == version_id)
            .ok_or(VersionError::NotFound(version_id))?;
        if target.status == VersionStatus::Deleted {
            return Err(VersionError::VersionDeleted(version_id));
        }
        // Supersede current
        for v in &mut self.versions {
            if v.status == VersionStatus::Active {
                v.status = VersionStatus::Superseded;
            }
        }
        // Activate target
        if let Some(v) = self.versions.iter_mut().find(|v| v.version_id == version_id) {
            v.status = VersionStatus::Active;
        }
        Ok(())
    }

    /// Tag a version.
    pub fn tag_version(
        &mut self,
        version_id: VersionId,
        tag: impl Into<String>,
    ) -> Result<(), VersionError> {
        let v = self
            .versions
            .iter_mut()
            .find(|v| v.version_id == version_id)
            .ok_or(VersionError::NotFound(version_id))?;
        let tag = tag.into();
        if !v.tags.contains(&tag) {
            v.tags.push(tag);
        }
        Ok(())
    }

    /// Compute diff between two versions.
    pub fn diff(
        &self,
        old_id: VersionId,
        new_id: VersionId,
    ) -> Result<VersionDiff, VersionError> {
        let old = self.get(old_id).ok_or(VersionError::NotFound(old_id))?;
        let new = self.get(new_id).ok_or(VersionError::NotFound(new_id))?;
        let size_delta = new.size_bytes as i64 - old.size_bytes as i64;
        let content_changed = old.content_hash != new.content_hash;
        let tags_added: Vec<String> = new
            .tags
            .iter()
            .filter(|t| !old.tags.contains(t))
            .cloned()
            .collect();
        let tags_removed: Vec<String> = old
            .tags
            .iter()
            .filter(|t| !new.tags.contains(t))
            .cloned()
            .collect();
        Ok(VersionDiff {
            size_delta,
            content_changed,
            tags_added,
            tags_removed,
        })
    }

    /// Apply retention policy, removing old superseded/deleted versions.
    fn apply_retention(&mut self) {
        if let Some(max) = self.policy.max_versions {
            let max = max as usize;
            while self.versions.len() > max {
                // Remove oldest non-active version
                if let Some(idx) = self
                    .versions
                    .iter()
                    .position(|v| v.status != VersionStatus::Active)
                {
                    self.versions.remove(idx);
                } else {
                    break;
                }
            }
        }
    }

    /// Total storage consumed by all versions.
    pub fn total_storage_bytes(&self) -> u64 {
        self.versions.iter().map(|v| v.size_bytes).sum()
    }
}

/// Version store managing multiple objects.
#[derive(Debug, Default)]
pub struct VersionStore {
    /// Per-object version histories.
    objects: HashMap<String, ObjectVersionHistory>,
}

impl VersionStore {
    /// Create an empty version store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create a version history for a key.
    pub fn get_or_create(&mut self, key: impl Into<String>) -> &mut ObjectVersionHistory {
        let key = key.into();
        self.objects
            .entry(key.clone())
            .or_insert_with(|| ObjectVersionHistory::new(key))
    }

    /// Get a version history by key.
    pub fn get(&self, key: &str) -> Option<&ObjectVersionHistory> {
        self.objects.get(key)
    }

    /// Number of tracked objects.
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// Total versions across all objects.
    pub fn total_versions(&self) -> usize {
        self.objects.values().map(|h| h.version_count()).sum()
    }
}

/// Errors from versioning operations.
#[derive(Debug, Clone, PartialEq)]
pub enum VersionError {
    /// Version not found.
    NotFound(VersionId),
    /// Version is deleted and cannot be activated.
    VersionDeleted(VersionId),
    /// Object key not found.
    ObjectNotFound(String),
}

impl std::fmt::Display for VersionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "version {id} not found"),
            Self::VersionDeleted(id) => write!(f, "version {id} is deleted"),
            Self::ObjectNotFound(key) => write!(f, "object '{key}' not found"),
        }
    }
}

impl std::error::Error for VersionError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_history() {
        let h = ObjectVersionHistory::new("media/video.mp4");
        assert_eq!(h.key(), "media/video.mp4");
        assert_eq!(h.version_count(), 0);
        assert!(h.current().is_none());
    }

    #[test]
    fn test_add_version() {
        let mut h = ObjectVersionHistory::new("file.mp4");
        let v1 = h.add_version(1000, "abc123", "user1", "initial upload");
        assert_eq!(v1, 1);
        assert_eq!(h.version_count(), 1);
        let cur = h.current().expect("current version should exist");
        assert_eq!(cur.status, VersionStatus::Active);
        assert_eq!(cur.size_bytes, 1000);
    }

    #[test]
    fn test_supersede_on_new_version() {
        let mut h = ObjectVersionHistory::new("file.mp4");
        let v1 = h.add_version(1000, "aaa", "u", "v1");
        let _v2 = h.add_version(2000, "bbb", "u", "v2");
        let old = h.get(v1).expect("get should succeed");
        assert_eq!(old.status, VersionStatus::Superseded);
        let cur = h.current().expect("current version should exist");
        assert_eq!(cur.size_bytes, 2000);
    }

    #[test]
    fn test_rollback() {
        let mut h = ObjectVersionHistory::new("file.mp4");
        let v1 = h.add_version(1000, "aaa", "u", "v1");
        let _v2 = h.add_version(2000, "bbb", "u", "v2");
        h.rollback_to(v1).expect("rollback should succeed");
        let cur = h.current().expect("current version should exist");
        assert_eq!(cur.version_id, v1);
    }

    #[test]
    fn test_rollback_deleted_fails() {
        let mut h = ObjectVersionHistory::new("file.mp4");
        let v1 = h.add_version(1000, "aaa", "u", "v1");
        let _v2 = h.add_version(2000, "bbb", "u", "v2");
        h.delete_version(v1).expect("delete version should succeed");
        let err = h.rollback_to(v1).unwrap_err();
        assert!(matches!(err, VersionError::VersionDeleted(_)));
    }

    #[test]
    fn test_delete_version() {
        let mut h = ObjectVersionHistory::new("file.mp4");
        let v1 = h.add_version(1000, "aaa", "u", "v1");
        h.delete_version(v1).expect("delete version should succeed");
        assert_eq!(h.get(v1).expect("get should succeed").status, VersionStatus::Deleted);
    }

    #[test]
    fn test_version_not_found() {
        let h = ObjectVersionHistory::new("file.mp4");
        assert!(h.get(999).is_none());
    }

    #[test]
    fn test_tag_version() {
        let mut h = ObjectVersionHistory::new("file.mp4");
        let v1 = h.add_version(1000, "aaa", "u", "v1");
        h.tag_version(v1, "release").expect("tag should succeed");
        h.tag_version(v1, "release").expect("tag should succeed"); // duplicate is ok
        assert_eq!(h.get(v1).expect("get should succeed").tags, vec!["release"]);
    }

    #[test]
    fn test_diff() {
        let mut h = ObjectVersionHistory::new("file.mp4");
        let v1 = h.add_version(1000, "aaa", "u", "v1");
        let v2 = h.add_version(1500, "bbb", "u", "v2");
        h.tag_version(v2, "approved").expect("tag should succeed");
        let d = h.diff(v1, v2).expect("diff should succeed");
        assert_eq!(d.size_delta, 500);
        assert!(d.content_changed);
        assert_eq!(d.tags_added, vec!["approved"]);
    }

    #[test]
    fn test_total_storage() {
        let mut h = ObjectVersionHistory::new("file.mp4");
        h.add_version(1000, "a", "u", "v1");
        h.add_version(2000, "b", "u", "v2");
        assert_eq!(h.total_storage_bytes(), 3000);
    }

    #[test]
    fn test_version_store() {
        let mut store = VersionStore::new();
        let hist = store.get_or_create("file.mp4");
        hist.add_version(100, "x", "u", "init");
        assert_eq!(store.object_count(), 1);
        assert_eq!(store.total_versions(), 1);
    }

    #[test]
    fn test_retention_policy() {
        let policy = RetentionPolicy {
            max_versions: Some(3),
            max_age_secs: None,
            keep_latest: true,
        };
        let mut h = ObjectVersionHistory::with_policy("file.mp4", policy);
        h.add_version(100, "a", "u", "v1");
        h.add_version(200, "b", "u", "v2");
        h.add_version(300, "c", "u", "v3");
        h.add_version(400, "d", "u", "v4");
        // Should have trimmed to 3
        assert!(h.version_count() <= 3);
    }

    #[test]
    fn test_version_error_display() {
        let e = VersionError::NotFound(7);
        assert!(e.to_string().contains("7"));
        let e2 = VersionError::ObjectNotFound("key".into());
        assert!(e2.to_string().contains("key"));
    }
}
