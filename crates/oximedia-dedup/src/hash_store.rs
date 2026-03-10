//! Persistent hash store for deduplication lookups.
//!
//! Provides an in-memory hash store backed by sorted vectors for fast
//! lookup, insertion, and range queries. Designed to hold millions of
//! hash entries with minimal overhead.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// HashEntry
// ---------------------------------------------------------------------------

/// A single hash entry in the store.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HashEntry {
    /// The 256-bit hash digest as a hex string.
    pub digest: String,
    /// The file path that produced this hash.
    pub file_path: PathBuf,
    /// File size in bytes.
    pub file_size: u64,
    /// Unix timestamp when the entry was added.
    pub added_at: u64,
}

impl HashEntry {
    /// Create a new hash entry.
    pub fn new(digest: &str, file_path: PathBuf, file_size: u64, added_at: u64) -> Self {
        Self {
            digest: digest.to_string(),
            file_path,
            file_size,
            added_at,
        }
    }
}

// ---------------------------------------------------------------------------
// HashStore
// ---------------------------------------------------------------------------

/// An in-memory hash store that maps digests to file entries.
///
/// Supports fast lookup by digest and retrieval of duplicate groups.
pub struct HashStore {
    /// Map from digest string to list of entries with that digest.
    entries: HashMap<String, Vec<HashEntry>>,
    /// Total number of entries.
    count: usize,
}

impl HashStore {
    /// Create an empty hash store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            count: 0,
        }
    }

    /// Create a hash store with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(cap),
            count: 0,
        }
    }

    /// Insert an entry into the store.
    pub fn insert(&mut self, entry: HashEntry) {
        self.entries
            .entry(entry.digest.clone())
            .or_default()
            .push(entry);
        self.count += 1;
    }

    /// Check if a digest exists in the store.
    #[must_use]
    pub fn contains(&self, digest: &str) -> bool {
        self.entries.contains_key(digest)
    }

    /// Look up entries by digest.
    #[must_use]
    pub fn get(&self, digest: &str) -> Option<&[HashEntry]> {
        self.entries.get(digest).map(Vec::as_slice)
    }

    /// Return the total number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Return `true` if the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Return the number of unique digests.
    #[must_use]
    pub fn unique_count(&self) -> usize {
        self.entries.len()
    }

    /// Return all digest groups that have more than one entry (duplicates).
    #[must_use]
    pub fn duplicate_groups(&self) -> Vec<(&str, &[HashEntry])> {
        self.entries
            .iter()
            .filter(|(_, v)| v.len() > 1)
            .map(|(k, v)| (k.as_str(), v.as_slice()))
            .collect()
    }

    /// Return the number of duplicate entries (entries beyond the first in each group).
    #[must_use]
    pub fn duplicate_count(&self) -> usize {
        self.entries
            .values()
            .filter(|v| v.len() > 1)
            .map(|v| v.len() - 1)
            .sum()
    }

    /// Return the total file size of all duplicate entries.
    #[must_use]
    pub fn duplicate_bytes(&self) -> u64 {
        self.entries
            .values()
            .filter(|v| v.len() > 1)
            .flat_map(|v| v.iter().skip(1))
            .map(|e| e.file_size)
            .sum()
    }

    /// Remove all entries for a given digest. Returns the removed entries.
    pub fn remove(&mut self, digest: &str) -> Vec<HashEntry> {
        if let Some(removed) = self.entries.remove(digest) {
            self.count -= removed.len();
            removed
        } else {
            Vec::new()
        }
    }

    /// Clear the entire store.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.count = 0;
    }

    /// Return an iterator over all unique digests.
    pub fn digests(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    /// Return deduplication ratio as a fraction.
    ///
    /// `1.0` means no duplicates, lower means more duplication.
    #[must_use]
    pub fn dedup_ratio(&self) -> f64 {
        if self.count == 0 {
            return 1.0;
        }
        self.unique_count() as f64 / self.count as f64
    }
}

impl Default for HashStore {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(digest: &str, path: &str, size: u64) -> HashEntry {
        HashEntry::new(digest, PathBuf::from(path), size, 1000)
    }

    #[test]
    fn test_new_store_empty() {
        let store = HashStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert_eq!(store.unique_count(), 0);
    }

    #[test]
    fn test_insert_and_lookup() {
        let mut store = HashStore::new();
        store.insert(make_entry("aaa", "/a.mp4", 100));
        assert!(store.contains("aaa"));
        assert!(!store.contains("bbb"));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_get_entries() {
        let mut store = HashStore::new();
        store.insert(make_entry("hash1", "/file1.mp4", 200));
        let entries = store.get("hash1").expect("operation should succeed");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].file_size, 200);
    }

    #[test]
    fn test_duplicate_groups() {
        let mut store = HashStore::new();
        store.insert(make_entry("dup", "/a.mp4", 100));
        store.insert(make_entry("dup", "/b.mp4", 100));
        store.insert(make_entry("unique", "/c.mp4", 50));

        let groups = store.duplicate_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].0, "dup");
        assert_eq!(groups[0].1.len(), 2);
    }

    #[test]
    fn test_duplicate_count() {
        let mut store = HashStore::new();
        store.insert(make_entry("d", "/a.mp4", 10));
        store.insert(make_entry("d", "/b.mp4", 10));
        store.insert(make_entry("d", "/c.mp4", 10));
        store.insert(make_entry("u", "/d.mp4", 10));

        assert_eq!(store.duplicate_count(), 2); // 3-1 = 2 duplicates
    }

    #[test]
    fn test_duplicate_bytes() {
        let mut store = HashStore::new();
        store.insert(make_entry("h", "/a.mp4", 1000));
        store.insert(make_entry("h", "/b.mp4", 1000));
        store.insert(make_entry("h", "/c.mp4", 1500));

        // skip first, sum rest = 1000 + 1500 = 2500
        assert_eq!(store.duplicate_bytes(), 2500);
    }

    #[test]
    fn test_remove() {
        let mut store = HashStore::new();
        store.insert(make_entry("rm", "/a.mp4", 10));
        store.insert(make_entry("rm", "/b.mp4", 20));
        store.insert(make_entry("keep", "/c.mp4", 30));

        let removed = store.remove("rm");
        assert_eq!(removed.len(), 2);
        assert!(!store.contains("rm"));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut store = HashStore::new();
        let removed = store.remove("nope");
        assert!(removed.is_empty());
    }

    #[test]
    fn test_clear() {
        let mut store = HashStore::new();
        store.insert(make_entry("a", "/x.mp4", 1));
        store.insert(make_entry("b", "/y.mp4", 2));
        store.clear();
        assert!(store.is_empty());
        assert_eq!(store.unique_count(), 0);
    }

    #[test]
    fn test_dedup_ratio_no_dupes() {
        let mut store = HashStore::new();
        store.insert(make_entry("a", "/1.mp4", 10));
        store.insert(make_entry("b", "/2.mp4", 20));
        assert!((store.dedup_ratio() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_dedup_ratio_all_dupes() {
        let mut store = HashStore::new();
        store.insert(make_entry("same", "/1.mp4", 10));
        store.insert(make_entry("same", "/2.mp4", 10));
        store.insert(make_entry("same", "/3.mp4", 10));
        assert!((store.dedup_ratio() - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_dedup_ratio_empty() {
        let store = HashStore::new();
        assert!((store.dedup_ratio() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_with_capacity() {
        let store = HashStore::with_capacity(1000);
        assert!(store.is_empty());
    }
}
