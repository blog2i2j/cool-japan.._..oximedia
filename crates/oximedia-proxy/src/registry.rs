//! Proxy registry: maps original source files to their proxy counterparts.
//!
//! The `ProxyRegistry` is an in-memory (and serializable) mapping from original
//! media paths to one or more proxy entries, each described by a `ProxySpec`.
//! It supports multi-resolution proxies per source and can be persisted to JSON.

use crate::spec::{ProxyCodec, ProxySpec};
use crate::{ProxyError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Represents a single proxy file registered for an original.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyEntry {
    /// Path to the proxy file.
    pub proxy_path: PathBuf,
    /// The spec used to create this proxy.
    pub spec: ProxySpec,
    /// Creation timestamp (Unix seconds, approximate).
    pub created_at: u64,
    /// File size in bytes (0 if unknown).
    pub file_size: u64,
    /// Whether this proxy has been verified to exist.
    pub verified: bool,
}

impl ProxyEntry {
    /// Create a new proxy entry.
    #[must_use]
    pub fn new(proxy_path: PathBuf, spec: ProxySpec) -> Self {
        Self {
            proxy_path,
            spec,
            created_at: 0,
            file_size: 0,
            verified: false,
        }
    }

    /// Check if the proxy file exists on disk.
    #[must_use]
    pub fn exists(&self) -> bool {
        self.proxy_path.exists()
    }

    /// Get the codec used for this proxy.
    #[must_use]
    pub fn codec(&self) -> &ProxyCodec {
        &self.spec.codec
    }
}

/// Entry in the registry for a single original file.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RegistryRecord {
    /// Path to the original file.
    pub original_path: PathBuf,
    /// Proxies available for this original.
    pub proxies: Vec<ProxyEntry>,
    /// User-defined tags.
    pub tags: Vec<String>,
}

impl RegistryRecord {
    /// Create a new registry record for an original path.
    #[must_use]
    pub fn new(original_path: PathBuf) -> Self {
        Self {
            original_path,
            proxies: Vec::new(),
            tags: Vec::new(),
        }
    }

    /// Add a proxy entry.
    pub fn add_proxy(&mut self, entry: ProxyEntry) {
        self.proxies.push(entry);
    }

    /// Find a proxy by spec name.
    #[must_use]
    pub fn find_proxy_by_spec(&self, spec_name: &str) -> Option<&ProxyEntry> {
        self.proxies.iter().find(|e| e.spec.name == spec_name)
    }

    /// Find the best matching proxy for a given maximum video bitrate.
    #[must_use]
    pub fn best_proxy_for_bitrate(&self, max_bitrate: u64) -> Option<&ProxyEntry> {
        self.proxies
            .iter()
            .filter(|e| e.spec.video_bitrate <= max_bitrate)
            .max_by_key(|e| e.spec.video_bitrate)
    }

    /// Remove all proxies that no longer exist on disk.
    pub fn purge_missing(&mut self) -> usize {
        let before = self.proxies.len();
        self.proxies.retain(|e| e.exists());
        before - self.proxies.len()
    }
}

/// Maps original media paths to their proxy files.
///
/// The registry is keyed by the canonical string representation of the original file path.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProxyRegistry {
    records: HashMap<String, RegistryRecord>,
    /// Registry version for future compatibility.
    version: u32,
}

impl ProxyRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
            version: 1,
        }
    }

    /// Register a proxy for an original file.
    ///
    /// If the original is already registered, the proxy is added to its record.
    pub fn register(&mut self, original: &Path, proxy_path: &Path, spec: ProxySpec) {
        let key = original.to_string_lossy().into_owned();
        let entry = ProxyEntry::new(proxy_path.to_path_buf(), spec);
        self.records
            .entry(key)
            .or_insert_with(|| RegistryRecord::new(original.to_path_buf()))
            .add_proxy(entry);
    }

    /// Look up the record for an original file.
    #[must_use]
    pub fn get(&self, original: &Path) -> Option<&RegistryRecord> {
        let key = original.to_string_lossy();
        self.records.get(key.as_ref())
    }

    /// Look up the record for an original file (mutable).
    pub fn get_mut(&mut self, original: &Path) -> Option<&mut RegistryRecord> {
        let key = original.to_string_lossy().into_owned();
        self.records.get_mut(&key)
    }

    /// Remove an original and all its proxies from the registry.
    ///
    /// Returns the removed record if it existed.
    pub fn remove(&mut self, original: &Path) -> Option<RegistryRecord> {
        let key = original.to_string_lossy().into_owned();
        self.records.remove(&key)
    }

    /// Find all proxies with a given spec name across all originals.
    #[must_use]
    pub fn find_by_spec(&self, spec_name: &str) -> Vec<(&RegistryRecord, &ProxyEntry)> {
        self.records
            .values()
            .flat_map(|r| {
                r.proxies
                    .iter()
                    .filter(|e| e.spec.name == spec_name)
                    .map(move |e| (r, e))
            })
            .collect()
    }

    /// Total number of original files registered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Total number of proxy entries across all originals.
    #[must_use]
    pub fn proxy_count(&self) -> usize {
        self.records.values().map(|r| r.proxies.len()).sum()
    }

    /// Purge all proxy entries that don't exist on disk.
    ///
    /// Returns total number of entries removed.
    pub fn purge_missing(&mut self) -> usize {
        self.records
            .values_mut()
            .map(RegistryRecord::purge_missing)
            .sum()
    }

    /// Remove originals that have no proxies.
    ///
    /// Returns number of originals removed.
    pub fn remove_empty_records(&mut self) -> usize {
        let before = self.records.len();
        self.records.retain(|_, r| !r.proxies.is_empty());
        before - self.records.len()
    }

    /// Serialize the registry to JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| ProxyError::MetadataError(e.to_string()))
    }

    /// Deserialize the registry from JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if deserialization fails.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| ProxyError::MetadataError(e.to_string()))
    }

    /// Save registry to a JSON file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = self.to_json()?;
        std::fs::write(path, json).map_err(ProxyError::IoError)
    }

    /// Load registry from a JSON file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(ProxyError::IoError)?;
        Self::from_json(&content)
    }

    /// Iterate over all records.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &RegistryRecord)> {
        self.records.iter().map(|(k, v)| (k.as_str(), v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{ProxyResolutionMode, ProxySpec};

    fn tmp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("oximedia-proxy-registry-{name}"))
    }

    fn make_spec(name: &str) -> ProxySpec {
        ProxySpec::new(
            name,
            ProxyResolutionMode::ScaleFactor(0.25),
            ProxyCodec::H264,
            2_000_000,
        )
    }

    #[test]
    fn test_proxy_entry_new() {
        let p = tmp_path("proxy.mp4");
        let entry = ProxyEntry::new(p.clone(), make_spec("Test"));
        assert_eq!(entry.proxy_path, p);
        assert_eq!(entry.spec.name, "Test");
        assert!(!entry.verified);
    }

    #[test]
    fn test_proxy_entry_codec() {
        let entry = ProxyEntry::new(tmp_path("p.mp4"), make_spec("Q"));
        assert_eq!(entry.codec(), &ProxyCodec::H264);
    }

    #[test]
    fn test_proxy_entry_not_exists() {
        let entry = ProxyEntry::new(PathBuf::from("/nonexistent/proxy.mp4"), make_spec("Q"));
        assert!(!entry.exists());
    }

    #[test]
    fn test_registry_record_new() {
        let rec = RegistryRecord::new(PathBuf::from("/src/clip.mov"));
        assert_eq!(rec.original_path, PathBuf::from("/src/clip.mov"));
        assert!(rec.proxies.is_empty());
    }

    #[test]
    fn test_registry_record_add_proxy() {
        let mut rec = RegistryRecord::new(PathBuf::from("/src/clip.mov"));
        rec.add_proxy(ProxyEntry::new(
            PathBuf::from("/proxy/clip.mp4"),
            make_spec("Q"),
        ));
        assert_eq!(rec.proxies.len(), 1);
    }

    #[test]
    fn test_registry_record_find_by_spec() {
        let mut rec = RegistryRecord::new(PathBuf::from("/src/clip.mov"));
        rec.add_proxy(ProxyEntry::new(
            PathBuf::from("/proxy/clip.mp4"),
            make_spec("Quarter"),
        ));
        rec.add_proxy(ProxyEntry::new(
            PathBuf::from("/proxy/clip_h.mp4"),
            make_spec("Half"),
        ));
        assert!(rec.find_proxy_by_spec("Quarter").is_some());
        assert!(rec.find_proxy_by_spec("Half").is_some());
        assert!(rec.find_proxy_by_spec("Missing").is_none());
    }

    #[test]
    fn test_registry_record_best_proxy_for_bitrate() {
        let mut rec = RegistryRecord::new(PathBuf::from("/src/clip.mov"));
        rec.add_proxy(ProxyEntry::new(PathBuf::from("/p1.mp4"), make_spec("Low")));
        let mut high_spec = make_spec("High");
        high_spec.video_bitrate = 10_000_000;
        rec.add_proxy(ProxyEntry::new(PathBuf::from("/p2.mp4"), high_spec));

        let best = rec
            .best_proxy_for_bitrate(3_000_000)
            .expect("should succeed in test");
        assert_eq!(best.spec.name, "Low");

        let best_all = rec
            .best_proxy_for_bitrate(100_000_000)
            .expect("should succeed in test");
        assert_eq!(best_all.spec.name, "High");
    }

    #[test]
    fn test_proxy_registry_new() {
        let reg = ProxyRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
        assert_eq!(reg.proxy_count(), 0);
    }

    #[test]
    fn test_proxy_registry_register_and_get() {
        let mut reg = ProxyRegistry::new();
        let original = Path::new("/media/clip001.mov");
        let proxy = Path::new("/proxy/clip001.mp4");
        reg.register(original, proxy, make_spec("Quarter"));
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.proxy_count(), 1);

        let rec = reg.get(original).expect("should succeed in test");
        assert_eq!(rec.proxies.len(), 1);
        assert_eq!(
            rec.proxies[0].proxy_path,
            PathBuf::from("/proxy/clip001.mp4")
        );
    }

    #[test]
    fn test_proxy_registry_multiple_proxies_per_original() {
        let mut reg = ProxyRegistry::new();
        let original = Path::new("/media/clip001.mov");
        reg.register(original, Path::new("/proxy/q.mp4"), make_spec("Quarter"));
        reg.register(original, Path::new("/proxy/h.mp4"), make_spec("Half"));
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.proxy_count(), 2);
    }

    #[test]
    fn test_proxy_registry_remove() {
        let mut reg = ProxyRegistry::new();
        let original = Path::new("/media/clip001.mov");
        reg.register(original, Path::new("/p.mp4"), make_spec("Q"));
        let removed = reg.remove(original);
        assert!(removed.is_some());
        assert!(reg.is_empty());
    }

    #[test]
    fn test_proxy_registry_find_by_spec() {
        let mut reg = ProxyRegistry::new();
        reg.register(
            Path::new("/a.mov"),
            Path::new("/pa.mp4"),
            make_spec("Quarter"),
        );
        reg.register(
            Path::new("/b.mov"),
            Path::new("/pb.mp4"),
            make_spec("Quarter"),
        );
        reg.register(Path::new("/c.mov"), Path::new("/pc.mp4"), make_spec("Half"));
        let matches = reg.find_by_spec("Quarter");
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_proxy_registry_json_roundtrip() {
        let mut reg = ProxyRegistry::new();
        reg.register(
            Path::new("/orig.mov"),
            Path::new("/proxy.mp4"),
            make_spec("Q"),
        );
        let json = reg.to_json().expect("should succeed in test");
        assert!(!json.is_empty());
        let loaded = ProxyRegistry::from_json(&json).expect("should succeed in test");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.proxy_count(), 1);
    }

    #[test]
    fn test_proxy_registry_remove_empty_records() {
        let mut reg = ProxyRegistry::new();
        // Add a record with no proxies (manually)
        reg.records.insert(
            "/empty.mov".to_string(),
            RegistryRecord::new(PathBuf::from("/empty.mov")),
        );
        reg.register(
            Path::new("/full.mov"),
            Path::new("/proxy.mp4"),
            make_spec("Q"),
        );
        assert_eq!(reg.len(), 2);
        let removed = reg.remove_empty_records();
        assert_eq!(removed, 1);
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn test_proxy_registry_iter() {
        let mut reg = ProxyRegistry::new();
        reg.register(Path::new("/a.mov"), Path::new("/p.mp4"), make_spec("Q"));
        let count = reg.iter().count();
        assert_eq!(count, 1);
    }
}
