//! Asset version comparison for the MAM system.
//!
//! Computes a structured diff between two [`AssetVersion`] snapshots,
//! returning a list of [`FieldChange`] entries that describe which fields
//! changed, were added, or were removed.

#![allow(dead_code)]

use crate::version_control::AssetVersion;

// ── FieldChange ───────────────────────────────────────────────────────────────

/// Describes a single changed field when comparing two [`AssetVersion`] records.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldChange {
    /// Name of the field that changed.
    pub field: String,
    /// String representation of the value in `v1` (`None` if absent in `v1`).
    pub old_value: Option<String>,
    /// String representation of the value in `v2` (`None` if absent in `v2`).
    pub new_value: Option<String>,
}

impl FieldChange {
    /// Create a new `FieldChange`.
    #[must_use]
    pub fn new(field: &str, old_value: Option<String>, new_value: Option<String>) -> Self {
        Self {
            field: field.to_owned(),
            old_value,
            new_value,
        }
    }

    /// Returns `true` if this represents a value that was added (absent in v1).
    #[must_use]
    pub fn is_addition(&self) -> bool {
        self.old_value.is_none() && self.new_value.is_some()
    }

    /// Returns `true` if this represents a value that was removed (absent in v2).
    #[must_use]
    pub fn is_removal(&self) -> bool {
        self.old_value.is_some() && self.new_value.is_none()
    }

    /// Returns `true` if both values are present but differ.
    #[must_use]
    pub fn is_modification(&self) -> bool {
        self.old_value.is_some() && self.new_value.is_some() && self.old_value != self.new_value
    }
}

// ── AssetVersionDiff ──────────────────────────────────────────────────────────

/// Utilities for computing the diff between two [`AssetVersion`] snapshots.
///
/// # Example
///
/// ```rust
/// use oximedia_mam::version_compare::AssetVersionDiff;
/// use oximedia_mam::version_control::{AssetVersion, VersionAction};
///
/// let v1 = AssetVersion {
///     version_id: 1, asset_id: 10, version_num: 1,
///     action: VersionAction::Created, timestamp_epoch: 1_000,
///     size_bytes: 1024, path: "/store/v1.mp4".into(),
/// };
/// let v2 = AssetVersion {
///     version_id: 2, asset_id: 10, version_num: 2,
///     action: VersionAction::Transcoded, timestamp_epoch: 2_000,
///     size_bytes: 2048, path: "/store/v2.mp4".into(),
/// };
///
/// let changes = AssetVersionDiff::compute(&v1, &v2);
/// assert!(!changes.is_empty());
/// ```
pub struct AssetVersionDiff;

impl AssetVersionDiff {
    /// Compute the diff between `v1` and `v2`.
    ///
    /// Compares all tracked fields of [`AssetVersion`] and returns one
    /// [`FieldChange`] entry for each field whose value differs.
    /// Fields that are identical in both versions are omitted.
    #[must_use]
    pub fn compute(v1: &AssetVersion, v2: &AssetVersion) -> Vec<FieldChange> {
        let mut changes = Vec::new();

        // version_id
        Self::compare_u64(&mut changes, "version_id", v1.version_id, v2.version_id);
        // asset_id — normally equal; included for completeness
        Self::compare_u64(&mut changes, "asset_id", v1.asset_id, v2.asset_id);
        // version_num
        Self::compare_u32(&mut changes, "version_num", v1.version_num, v2.version_num);
        // action
        let a1 = format!("{:?}", v1.action);
        let a2 = format!("{:?}", v2.action);
        if a1 != a2 {
            changes.push(FieldChange::new("action", Some(a1), Some(a2)));
        }
        // timestamp_epoch
        Self::compare_u64(
            &mut changes,
            "timestamp_epoch",
            v1.timestamp_epoch,
            v2.timestamp_epoch,
        );
        // size_bytes
        Self::compare_u64(&mut changes, "size_bytes", v1.size_bytes, v2.size_bytes);
        // path
        if v1.path != v2.path {
            changes.push(FieldChange::new(
                "path",
                Some(v1.path.clone()),
                Some(v2.path.clone()),
            ));
        }

        changes
    }

    /// Returns the names of all fields that changed.
    #[must_use]
    pub fn changed_fields(v1: &AssetVersion, v2: &AssetVersion) -> Vec<String> {
        Self::compute(v1, v2)
            .into_iter()
            .map(|c| c.field)
            .collect()
    }

    /// Returns `true` if `v1` and `v2` are identical in all tracked fields.
    #[must_use]
    pub fn are_equal(v1: &AssetVersion, v2: &AssetVersion) -> bool {
        Self::compute(v1, v2).is_empty()
    }

    // ── helpers ───────────────────────────────────────────────────────────────
    fn compare_u64(out: &mut Vec<FieldChange>, field: &str, old: u64, new: u64) {
        if old != new {
            out.push(FieldChange::new(
                field,
                Some(old.to_string()),
                Some(new.to_string()),
            ));
        }
    }

    fn compare_u32(out: &mut Vec<FieldChange>, field: &str, old: u32, new: u32) {
        if old != new {
            out.push(FieldChange::new(
                field,
                Some(old.to_string()),
                Some(new.to_string()),
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::version_control::VersionAction;

    fn make_version(
        version_id: u64,
        version_num: u32,
        action: VersionAction,
        ts: u64,
        size: u64,
        path: &str,
    ) -> AssetVersion {
        AssetVersion {
            version_id,
            asset_id: 10,
            version_num,
            action,
            timestamp_epoch: ts,
            size_bytes: size,
            path: path.to_owned(),
        }
    }

    #[test]
    fn test_identical_versions_no_changes() {
        let v = make_version(1, 1, VersionAction::Created, 1000, 1024, "/v1.mp4");
        let changes = AssetVersionDiff::compute(&v, &v);
        assert!(changes.is_empty());
    }

    #[test]
    fn test_are_equal_true_for_identical() {
        let v = make_version(1, 1, VersionAction::Created, 1000, 1024, "/v1.mp4");
        assert!(AssetVersionDiff::are_equal(&v, &v));
    }

    #[test]
    fn test_different_version_num_detected() {
        let v1 = make_version(1, 1, VersionAction::Created, 1000, 1024, "/v1.mp4");
        let v2 = make_version(2, 2, VersionAction::Edited, 2000, 2048, "/v2.mp4");
        let changes = AssetVersionDiff::compute(&v1, &v2);
        let fields: Vec<&str> = changes.iter().map(|c| c.field.as_str()).collect();
        assert!(fields.contains(&"version_num"), "version_num should be in changes");
        assert!(fields.contains(&"size_bytes"), "size_bytes should be in changes");
        assert!(fields.contains(&"path"), "path should be in changes");
    }

    #[test]
    fn test_action_change_detected() {
        let v1 = make_version(1, 1, VersionAction::Created, 1000, 1024, "/v.mp4");
        let v2 = make_version(1, 1, VersionAction::Transcoded, 1000, 1024, "/v.mp4");
        let changes = AssetVersionDiff::compute(&v1, &v2);
        let fields: Vec<&str> = changes.iter().map(|c| c.field.as_str()).collect();
        assert!(fields.contains(&"action"), "action change must be detected");
    }

    #[test]
    fn test_path_change_detected() {
        let v1 = make_version(1, 1, VersionAction::Created, 1000, 100, "/old.mp4");
        let v2 = make_version(1, 1, VersionAction::Created, 1000, 100, "/new.mp4");
        let changes = AssetVersionDiff::compute(&v1, &v2);
        let path_change = changes.iter().find(|c| c.field == "path").expect("path change expected");
        assert!(path_change.is_modification());
        assert_eq!(path_change.old_value.as_deref(), Some("/old.mp4"));
        assert_eq!(path_change.new_value.as_deref(), Some("/new.mp4"));
    }

    #[test]
    fn test_changed_fields_returns_field_names() {
        let v1 = make_version(1, 1, VersionAction::Created, 1000, 100, "/a.mp4");
        let v2 = make_version(2, 2, VersionAction::Edited, 2000, 200, "/b.mp4");
        let fields = AssetVersionDiff::changed_fields(&v1, &v2);
        assert!(!fields.is_empty());
        assert!(fields.contains(&"path".to_owned()));
    }

    #[test]
    fn test_field_change_is_modification() {
        let fc = FieldChange::new("size", Some("100".into()), Some("200".into()));
        assert!(fc.is_modification());
        assert!(!fc.is_addition());
        assert!(!fc.is_removal());
    }

    #[test]
    fn test_field_change_is_addition() {
        let fc = FieldChange::new("field", None, Some("value".into()));
        assert!(fc.is_addition());
        assert!(!fc.is_modification());
    }

    #[test]
    fn test_field_change_is_removal() {
        let fc = FieldChange::new("field", Some("value".into()), None);
        assert!(fc.is_removal());
        assert!(!fc.is_modification());
    }

    #[test]
    fn test_size_only_change() {
        let v1 = make_version(1, 1, VersionAction::Created, 1000, 512, "/same.mp4");
        let v2 = make_version(1, 1, VersionAction::Created, 1000, 1024, "/same.mp4");
        let changes = AssetVersionDiff::compute(&v1, &v2);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].field, "size_bytes");
    }
}
