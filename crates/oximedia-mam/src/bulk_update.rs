//! Bulk metadata update operations for the MAM system.
//!
//! Provides [`MetadataStore`], a simple in-memory key-value store per asset,
//! and [`BulkMetadataUpdate`], which accumulates a set of field-value pairs
//! and applies them atomically to a slice of asset IDs.

#![allow(dead_code)]

use std::collections::HashMap;

// ── MetadataStore ─────────────────────────────────────────────────────────────

/// Simple in-memory metadata store mapping asset IDs to their string key-value fields.
///
/// In production this would be backed by a database; here it is an in-memory
/// `HashMap` suitable for testing and prototyping.
#[derive(Debug, Default)]
pub struct MetadataStore {
    /// Per-asset field map: asset_id → (field_name → value).
    data: HashMap<u64, HashMap<String, String>>,
}

impl MetadataStore {
    /// Create an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a single field on an asset.
    pub fn set(&mut self, asset_id: u64, key: &str, value: &str) {
        self.data
            .entry(asset_id)
            .or_default()
            .insert(key.to_owned(), value.to_owned());
    }

    /// Get a field value for an asset, if present.
    #[must_use]
    pub fn get(&self, asset_id: u64, key: &str) -> Option<&str> {
        self.data
            .get(&asset_id)
            .and_then(|m| m.get(key))
            .map(String::as_str)
    }

    /// Return all fields for an asset.
    #[must_use]
    pub fn fields(&self, asset_id: u64) -> Option<&HashMap<String, String>> {
        self.data.get(&asset_id)
    }

    /// Return `true` if the store has any data for `asset_id`.
    #[must_use]
    pub fn contains_asset(&self, asset_id: u64) -> bool {
        self.data.contains_key(&asset_id)
    }

    /// Return the number of assets tracked by this store.
    #[must_use]
    pub fn asset_count(&self) -> usize {
        self.data.len()
    }
}

// ── BulkMetadataUpdate ────────────────────────────────────────────────────────

/// Accumulates field-value pairs and applies them in bulk to a set of assets.
///
/// # Example
///
/// ```rust
/// use oximedia_mam::bulk_update::{BulkMetadataUpdate, MetadataStore};
///
/// let mut store = MetadataStore::new();
/// let mut update = BulkMetadataUpdate::new();
/// update.set("status", "approved");
/// update.set("reviewed_by", "alice");
///
/// let updated = update.apply_to(&[1, 2, 3], &mut store);
/// assert_eq!(updated, 3);
/// assert_eq!(store.get(1, "status"), Some("approved"));
/// ```
#[derive(Debug, Default)]
pub struct BulkMetadataUpdate {
    /// Ordered list of (key, value) pairs to set.
    fields: Vec<(String, String)>,
}

impl BulkMetadataUpdate {
    /// Create an empty update.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or overwrite a field in this bulk update.
    ///
    /// If `key` has already been added, the previous value is replaced.
    pub fn set(&mut self, key: &str, value: &str) {
        // Replace existing entry for the same key, or append
        if let Some(entry) = self.fields.iter_mut().find(|(k, _)| k == key) {
            entry.1 = value.to_owned();
        } else {
            self.fields.push((key.to_owned(), value.to_owned()));
        }
    }

    /// Return the number of field-value pairs in this update.
    #[must_use]
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Apply this update to all `asset_ids` in `store`.
    ///
    /// Returns the number of assets that were updated.
    /// Assets that do not yet exist in the store are created automatically.
    pub fn apply_to(&self, asset_ids: &[u64], store: &mut MetadataStore) -> usize {
        if self.fields.is_empty() {
            return 0;
        }
        for &id in asset_ids {
            for (key, value) in &self.fields {
                store.set(id, key, value);
            }
        }
        asset_ids.len()
    }

    /// Clear all accumulated field-value pairs.
    pub fn clear(&mut self) {
        self.fields.clear();
    }

    /// Return an iterator over the (key, value) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.fields.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── MetadataStore ────────────────────────────────────────────────────────
    #[test]
    fn test_store_empty() {
        let store = MetadataStore::new();
        assert_eq!(store.asset_count(), 0);
        assert!(!store.contains_asset(1));
    }

    #[test]
    fn test_store_set_and_get() {
        let mut store = MetadataStore::new();
        store.set(1, "title", "My Film");
        assert_eq!(store.get(1, "title"), Some("My Film"));
    }

    #[test]
    fn test_store_get_missing_key() {
        let mut store = MetadataStore::new();
        store.set(1, "title", "X");
        assert!(store.get(1, "director").is_none());
    }

    #[test]
    fn test_store_get_missing_asset() {
        let store = MetadataStore::new();
        assert!(store.get(99, "title").is_none());
    }

    #[test]
    fn test_store_overwrite() {
        let mut store = MetadataStore::new();
        store.set(1, "status", "draft");
        store.set(1, "status", "approved");
        assert_eq!(store.get(1, "status"), Some("approved"));
    }

    #[test]
    fn test_store_contains_asset() {
        let mut store = MetadataStore::new();
        store.set(5, "key", "val");
        assert!(store.contains_asset(5));
        assert!(!store.contains_asset(6));
    }

    #[test]
    fn test_store_fields() {
        let mut store = MetadataStore::new();
        store.set(1, "a", "1");
        store.set(1, "b", "2");
        let fields = store.fields(1).expect("should exist");
        assert_eq!(fields.len(), 2);
    }

    // ── BulkMetadataUpdate ───────────────────────────────────────────────────
    #[test]
    fn test_bulk_new_empty() {
        let u = BulkMetadataUpdate::new();
        assert_eq!(u.field_count(), 0);
    }

    #[test]
    fn test_bulk_set() {
        let mut u = BulkMetadataUpdate::new();
        u.set("status", "approved");
        assert_eq!(u.field_count(), 1);
    }

    #[test]
    fn test_bulk_set_replaces_duplicate_key() {
        let mut u = BulkMetadataUpdate::new();
        u.set("status", "draft");
        u.set("status", "approved");
        assert_eq!(u.field_count(), 1);
        let kv: Vec<_> = u.iter().collect();
        assert_eq!(kv[0], ("status", "approved"));
    }

    #[test]
    fn test_bulk_apply_to_returns_count() {
        let mut store = MetadataStore::new();
        let mut u = BulkMetadataUpdate::new();
        u.set("status", "approved");
        let updated = u.apply_to(&[1, 2, 3], &mut store);
        assert_eq!(updated, 3);
    }

    #[test]
    fn test_bulk_apply_to_sets_fields() {
        let mut store = MetadataStore::new();
        let mut u = BulkMetadataUpdate::new();
        u.set("status", "approved");
        u.set("reviewed_by", "alice");
        u.apply_to(&[10, 20], &mut store);
        assert_eq!(store.get(10, "status"), Some("approved"));
        assert_eq!(store.get(20, "reviewed_by"), Some("alice"));
    }

    #[test]
    fn test_bulk_apply_empty_update_returns_zero() {
        let mut store = MetadataStore::new();
        let u = BulkMetadataUpdate::new();
        assert_eq!(u.apply_to(&[1, 2, 3], &mut store), 0);
    }

    #[test]
    fn test_bulk_apply_to_empty_ids() {
        let mut store = MetadataStore::new();
        let mut u = BulkMetadataUpdate::new();
        u.set("key", "val");
        assert_eq!(u.apply_to(&[], &mut store), 0);
    }

    #[test]
    fn test_bulk_clear() {
        let mut u = BulkMetadataUpdate::new();
        u.set("a", "b");
        u.clear();
        assert_eq!(u.field_count(), 0);
    }

    #[test]
    fn test_bulk_iter() {
        let mut u = BulkMetadataUpdate::new();
        u.set("x", "1");
        u.set("y", "2");
        let pairs: Vec<_> = u.iter().collect();
        assert_eq!(pairs.len(), 2);
    }
}
