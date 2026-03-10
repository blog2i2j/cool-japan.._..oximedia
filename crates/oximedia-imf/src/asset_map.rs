//! IMF Asset Map (ASSETMAP.xml) - high-level public API
//!
//! This module provides a high-level, self-contained representation of an IMF
//! Asset Map.  It is intentionally independent of the lower-level private
//! `assetmap` module so that consumers can work with a clean, ergonomic API.

#![allow(dead_code)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]

/// Category of an asset stored in an IMF package.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetCategory {
    /// Packing List (PKL)
    Pkl,
    /// Composition Playlist (CPL)
    Cpl,
    /// MXF essence file (video / audio / data)
    Mxf,
    /// Timed-text / subtitle file
    TimedText,
    /// Arbitrary sidecar file
    Sidecar,
}

impl AssetCategory {
    /// Returns `true` if this category represents an essence essence asset
    /// (i.e. not a metadata XML document).
    pub fn is_essence(&self) -> bool {
        matches!(self, Self::Mxf | Self::TimedText)
    }
}

/// A reference to a single asset within the Asset Map.
#[derive(Debug, Clone)]
pub struct AssetRef {
    /// Asset UUID (as a plain string, e.g. `"urn:uuid:…"` or bare).
    pub uuid: String,
    /// Relative path to the file within the package volume.
    pub path: String,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Optional hash (SHA-1 or MD-5 hex-encoded).
    pub hash: Option<String>,
}

impl AssetRef {
    /// Create a new `AssetRef`.
    pub fn new(uuid: impl Into<String>, path: impl Into<String>, size_bytes: u64) -> Self {
        Self {
            uuid: uuid.into(),
            path: path.into(),
            size_bytes,
            hash: None,
        }
    }

    /// Returns `true` when a hash is present.
    pub fn has_hash(&self) -> bool {
        self.hash.is_some()
    }

    /// Returns just the file-name portion of `path`, falling back to the full
    /// path string when no separator is present.
    pub fn filename(&self) -> String {
        self.path
            .rsplit('/')
            .next()
            .map(|s| s.to_string())
            .unwrap_or_else(|| self.path.clone())
    }
}

/// High-level representation of an IMF Asset Map (ASSETMAP.xml).
#[derive(Debug, Clone)]
pub struct ImfAssetMap {
    /// All assets registered in this map.
    pub assets: Vec<AssetRef>,
    /// Number of physical volumes that make up the package.
    pub volume_count: u32,
    /// Creator string (application / organisation).
    pub creator: String,
}

impl ImfAssetMap {
    /// Create an empty `ImfAssetMap`.
    pub fn new(creator: impl Into<String>) -> Self {
        Self {
            assets: Vec::new(),
            volume_count: 1,
            creator: creator.into(),
        }
    }

    /// Append an asset to the map.
    pub fn add(&mut self, asset: AssetRef) {
        self.assets.push(asset);
    }

    /// Find an asset by its UUID string (exact match).
    pub fn find_by_uuid(&self, uuid: &str) -> Option<&AssetRef> {
        self.assets.iter().find(|a| a.uuid == uuid)
    }

    /// Find an asset by its path (exact match).
    pub fn find_by_path(&self, path: &str) -> Option<&AssetRef> {
        self.assets.iter().find(|a| a.path == path)
    }

    /// Return all assets that belong to a given `AssetCategory`.
    ///
    /// Matching is performed by file-extension heuristic:
    /// - `.mxf` → `Mxf`
    /// - `.xml` with "pkl" in name → `Pkl`
    /// - `.xml` with "cpl" in name → `Cpl`
    /// - `.ttml` / `.srt` / `.vtt` / `.xml` (timed-text) → `TimedText`
    /// - everything else → `Sidecar`
    pub fn essence_assets(&self) -> Vec<&AssetRef> {
        self.assets
            .iter()
            .filter(|a| {
                let lower = a.path.to_lowercase();
                lower.ends_with(".mxf")
                    || lower.ends_with(".ttml")
                    || lower.ends_with(".srt")
                    || lower.ends_with(".vtt")
            })
            .collect()
    }

    /// Return the total size (in bytes) of all registered assets.
    pub fn total_size_bytes(&self) -> u64 {
        self.assets.iter().map(|a| a.size_bytes).sum()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_asset(uuid: &str, path: &str, size: u64) -> AssetRef {
        AssetRef::new(uuid, path, size)
    }

    // --- AssetCategory ---

    #[test]
    fn test_asset_category_is_essence_mxf() {
        assert!(AssetCategory::Mxf.is_essence());
    }

    #[test]
    fn test_asset_category_is_essence_timed_text() {
        assert!(AssetCategory::TimedText.is_essence());
    }

    #[test]
    fn test_asset_category_is_essence_pkl_false() {
        assert!(!AssetCategory::Pkl.is_essence());
    }

    #[test]
    fn test_asset_category_is_essence_cpl_false() {
        assert!(!AssetCategory::Cpl.is_essence());
    }

    #[test]
    fn test_asset_category_is_essence_sidecar_false() {
        assert!(!AssetCategory::Sidecar.is_essence());
    }

    // --- AssetRef ---

    #[test]
    fn test_asset_ref_has_hash_false() {
        let a = make_asset("uuid-1", "video.mxf", 1024);
        assert!(!a.has_hash());
    }

    #[test]
    fn test_asset_ref_has_hash_true() {
        let mut a = make_asset("uuid-1", "video.mxf", 1024);
        a.hash = Some("deadbeef".to_string());
        assert!(a.has_hash());
    }

    #[test]
    fn test_asset_ref_filename_simple() {
        let a = make_asset("uuid-1", "video.mxf", 1024);
        assert_eq!(a.filename(), "video.mxf");
    }

    #[test]
    fn test_asset_ref_filename_with_path() {
        let a = make_asset("uuid-2", "assets/audio/en.mxf", 512);
        assert_eq!(a.filename(), "en.mxf");
    }

    // --- ImfAssetMap ---

    #[test]
    fn test_asset_map_add_and_count() {
        let mut map = ImfAssetMap::new("OxiMedia Test");
        map.add(make_asset("u1", "a.mxf", 100));
        map.add(make_asset("u2", "b.mxf", 200));
        assert_eq!(map.assets.len(), 2);
    }

    #[test]
    fn test_asset_map_find_by_uuid() {
        let mut map = ImfAssetMap::new("OxiMedia");
        map.add(make_asset("uuid-abc", "video.mxf", 999));
        let found = map.find_by_uuid("uuid-abc");
        assert!(found.is_some());
        assert_eq!(found.expect("test expectation failed").path, "video.mxf");
    }

    #[test]
    fn test_asset_map_find_by_uuid_missing() {
        let map = ImfAssetMap::new("OxiMedia");
        assert!(map.find_by_uuid("does-not-exist").is_none());
    }

    #[test]
    fn test_asset_map_find_by_path() {
        let mut map = ImfAssetMap::new("OxiMedia");
        map.add(make_asset("u3", "subtitle.ttml", 50));
        assert!(map.find_by_path("subtitle.ttml").is_some());
    }

    #[test]
    fn test_asset_map_find_by_path_missing() {
        let map = ImfAssetMap::new("OxiMedia");
        assert!(map.find_by_path("nope.xml").is_none());
    }

    #[test]
    fn test_asset_map_essence_assets() {
        let mut map = ImfAssetMap::new("OxiMedia");
        map.add(make_asset("u1", "video.mxf", 8000));
        map.add(make_asset("u2", "subtitle.ttml", 200));
        map.add(make_asset("u3", "PKL.xml", 100));
        let essence = map.essence_assets();
        assert_eq!(essence.len(), 2);
    }

    #[test]
    fn test_asset_map_total_size_bytes() {
        let mut map = ImfAssetMap::new("OxiMedia");
        map.add(make_asset("u1", "a.mxf", 1000));
        map.add(make_asset("u2", "b.mxf", 500));
        assert_eq!(map.total_size_bytes(), 1500);
    }

    #[test]
    fn test_asset_map_total_size_bytes_empty() {
        let map = ImfAssetMap::new("OxiMedia");
        assert_eq!(map.total_size_bytes(), 0);
    }

    #[test]
    fn test_asset_map_creator() {
        let map = ImfAssetMap::new("TestCreator");
        assert_eq!(map.creator, "TestCreator");
    }

    #[test]
    fn test_asset_map_volume_count_default() {
        let map = ImfAssetMap::new("OxiMedia");
        assert_eq!(map.volume_count, 1);
    }
}
