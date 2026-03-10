//! IMF Composition Map utilities.
//!
//! Provides a mapping layer that correlates virtual track files referenced
//! in a CPL with physical MXF assets listed in the AssetMap and PKL.
//! This is used to resolve file paths before decode/transcode operations.

use std::collections::HashMap;
use uuid::Uuid;

/// Represents a resolved asset file entry
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedAsset {
    /// Asset UUID as it appears in the CPL
    pub asset_id: Uuid,
    /// Absolute file path on disk
    pub file_path: String,
    /// MIME type (e.g. "application/mxf")
    pub mime_type: String,
    /// File size in bytes (0 if unknown)
    pub size_bytes: u64,
}

impl ResolvedAsset {
    /// Create a new resolved asset entry
    #[allow(dead_code)]
    pub fn new(
        asset_id: Uuid,
        file_path: impl Into<String>,
        mime_type: impl Into<String>,
        size_bytes: u64,
    ) -> Self {
        Self {
            asset_id,
            file_path: file_path.into(),
            mime_type: mime_type.into(),
            size_bytes,
        }
    }

    /// True if the asset is an MXF essence file.
    #[allow(dead_code)]
    pub fn is_mxf(&self) -> bool {
        self.mime_type == "application/mxf"
    }
}

/// Maps CPL asset UUIDs to resolved physical file information.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct CompositionMap {
    entries: HashMap<Uuid, ResolvedAsset>,
}

impl CompositionMap {
    /// Create an empty map.
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace an entry for `asset_id`.
    #[allow(dead_code)]
    pub fn insert(&mut self, asset: ResolvedAsset) {
        self.entries.insert(asset.asset_id, asset);
    }

    /// Look up an asset by UUID.
    #[allow(dead_code)]
    pub fn get(&self, asset_id: &Uuid) -> Option<&ResolvedAsset> {
        self.entries.get(asset_id)
    }

    /// Remove an asset entry, returning it if present.
    #[allow(dead_code)]
    pub fn remove(&mut self, asset_id: &Uuid) -> Option<ResolvedAsset> {
        self.entries.remove(asset_id)
    }

    /// Number of entries in the map.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if the map contains no entries.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return all resolved MXF assets.
    #[allow(dead_code)]
    pub fn mxf_assets(&self) -> Vec<&ResolvedAsset> {
        self.entries.values().filter(|a| a.is_mxf()).collect()
    }

    /// Resolve a list of asset UUIDs to file paths.
    ///
    /// Returns `Err` with the first UUID that could not be resolved.
    #[allow(dead_code)]
    pub fn resolve_paths(&self, ids: &[Uuid]) -> Result<Vec<String>, Uuid> {
        ids.iter()
            .map(|id| self.entries.get(id).map(|a| a.file_path.clone()).ok_or(*id))
            .collect()
    }

    /// Merge another map into this one; existing entries are overwritten.
    #[allow(dead_code)]
    pub fn merge(&mut self, other: CompositionMap) {
        for (id, asset) in other.entries {
            self.entries.insert(id, asset);
        }
    }

    /// Total size in bytes of all mapped assets.
    #[allow(dead_code)]
    pub fn total_size_bytes(&self) -> u64 {
        self.entries.values().map(|a| a.size_bytes).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_asset(path: &str, mime: &str, size: u64) -> ResolvedAsset {
        ResolvedAsset::new(Uuid::new_v4(), path, mime, size)
    }

    #[test]
    fn test_resolved_asset_new() {
        let id = Uuid::new_v4();
        let a = ResolvedAsset::new(id, "/tmp/video.mxf", "application/mxf", 1024);
        assert_eq!(a.asset_id, id);
        assert_eq!(a.size_bytes, 1024);
    }

    #[test]
    fn test_is_mxf_true() {
        let a = make_asset("/tmp/x.mxf", "application/mxf", 0);
        assert!(a.is_mxf());
    }

    #[test]
    fn test_is_mxf_false() {
        let a = make_asset("/tmp/x.xml", "text/xml", 0);
        assert!(!a.is_mxf());
    }

    #[test]
    fn test_map_empty_on_creation() {
        let map = CompositionMap::new();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn test_insert_and_get() {
        let mut map = CompositionMap::new();
        let a = make_asset("/tmp/a.mxf", "application/mxf", 500);
        let id = a.asset_id;
        map.insert(a);
        assert_eq!(map.len(), 1);
        assert_eq!(
            map.get(&id).expect("get should succeed").file_path,
            "/tmp/a.mxf"
        );
    }

    #[test]
    fn test_get_missing() {
        let map = CompositionMap::new();
        assert!(map.get(&Uuid::new_v4()).is_none());
    }

    #[test]
    fn test_remove() {
        let mut map = CompositionMap::new();
        let a = make_asset("/x.mxf", "application/mxf", 0);
        let id = a.asset_id;
        map.insert(a);
        let removed = map.remove(&id);
        assert!(removed.is_some());
        assert!(map.is_empty());
    }

    #[test]
    fn test_mxf_assets_filter() {
        let mut map = CompositionMap::new();
        map.insert(make_asset("/a.mxf", "application/mxf", 100));
        map.insert(make_asset("/b.xml", "text/xml", 200));
        assert_eq!(map.mxf_assets().len(), 1);
    }

    #[test]
    fn test_resolve_paths_success() {
        let mut map = CompositionMap::new();
        let a = make_asset("/a.mxf", "application/mxf", 0);
        let id = a.asset_id;
        map.insert(a);
        let paths = map.resolve_paths(&[id]).expect("paths should be valid");
        assert_eq!(paths[0], "/a.mxf");
    }

    #[test]
    fn test_resolve_paths_missing_returns_err() {
        let map = CompositionMap::new();
        let missing = Uuid::new_v4();
        let result = map.resolve_paths(&[missing]);
        assert_eq!(result.unwrap_err(), missing);
    }

    #[test]
    fn test_merge_combines_entries() {
        let mut map1 = CompositionMap::new();
        map1.insert(make_asset("/a.mxf", "application/mxf", 100));

        let mut map2 = CompositionMap::new();
        map2.insert(make_asset("/b.mxf", "application/mxf", 200));

        map1.merge(map2);
        assert_eq!(map1.len(), 2);
    }

    #[test]
    fn test_total_size_bytes() {
        let mut map = CompositionMap::new();
        map.insert(make_asset("/a.mxf", "application/mxf", 300));
        map.insert(make_asset("/b.mxf", "application/mxf", 700));
        assert_eq!(map.total_size_bytes(), 1000);
    }

    #[test]
    fn test_insert_overwrites_existing() {
        let mut map = CompositionMap::new();
        let id = Uuid::new_v4();
        map.insert(ResolvedAsset::new(id, "/old.mxf", "application/mxf", 1));
        map.insert(ResolvedAsset::new(id, "/new.mxf", "application/mxf", 2));
        assert_eq!(map.len(), 1);
        assert_eq!(
            map.get(&id).expect("get should succeed").file_path,
            "/new.mxf"
        );
    }
}
