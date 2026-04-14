//! Partial archive restore: extract individual entries by path.
//!
//! An [`Archive`] is a simple in-memory store of named byte blobs.  Real
//! implementations would load data from disk / tape, but this module provides
//! the logic for path-based lookup so it can be used as a building block.
//!
//! # Example
//! ```rust
//! use oximedia_archive::partial_restore::{Archive, PartialRestorer};
//!
//! let mut archive = Archive::new();
//! archive.insert("media/clip.mkv", b"mkv data here".to_vec());
//!
//! let data = PartialRestorer::extract_by_path(&archive, "media/clip.mkv");
//! assert!(data.is_some());
//! assert_eq!(data.unwrap(), b"mkv data here");
//!
//! let missing = PartialRestorer::extract_by_path(&archive, "nonexistent.mkv");
//! assert!(missing.is_none());
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

/// A simple key-value archive mapping path strings to byte payloads.
///
/// For production use this would be backed by a file system, object store, or
/// tape; here it is fully in-memory so that `PartialRestorer` can be tested
/// without I/O.
#[derive(Debug, Default, Clone)]
pub struct Archive {
    entries: HashMap<String, Vec<u8>>,
}

impl Archive {
    /// Create an empty archive.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert an entry.
    ///
    /// If an entry with `path` already exists it is replaced.
    pub fn insert(&mut self, path: &str, data: Vec<u8>) {
        self.entries.insert(path.to_string(), data);
    }

    /// Remove an entry and return its data, or `None` if not found.
    pub fn remove(&mut self, path: &str) -> Option<Vec<u8>> {
        self.entries.remove(path)
    }

    /// Return the number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if the archive contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over `(path, data)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &[u8])> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v.as_slice()))
    }

    /// Return all stored paths.
    #[must_use]
    pub fn paths(&self) -> Vec<&str> {
        self.entries.keys().map(String::as_str).collect()
    }
}

/// Extracts individual entries from an [`Archive`] by their path.
pub struct PartialRestorer;

impl PartialRestorer {
    /// Extract a single entry by its exact path.
    ///
    /// # Arguments
    /// * `archive` – the archive to extract from.
    /// * `path`    – the exact path to look up (case-sensitive).
    ///
    /// # Returns
    /// `Some(Vec<u8>)` with a clone of the stored data, or `None` if the path
    /// does not exist in the archive.
    #[must_use]
    pub fn extract_by_path(archive: &Archive, path: &str) -> Option<Vec<u8>> {
        archive.entries.get(path).cloned()
    }

    /// Extract multiple entries in one call.
    ///
    /// Returns a `Vec` of `(path, Option<Vec<u8>>)` pairs preserving the
    /// input order.  Paths not found in the archive have `None` data.
    #[must_use]
    pub fn extract_batch(archive: &Archive, paths: &[&str]) -> Vec<(String, Option<Vec<u8>>)> {
        paths
            .iter()
            .map(|&p| (p.to_string(), Self::extract_by_path(archive, p)))
            .collect()
    }

    /// Extract all entries whose paths start with `prefix`.
    ///
    /// Useful for restoring a sub-directory of the archive at once.
    #[must_use]
    pub fn extract_by_prefix(archive: &Archive, prefix: &str) -> Vec<(String, Vec<u8>)> {
        archive
            .entries
            .iter()
            .filter(|(k, _)| k.starts_with(prefix))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_archive() -> Archive {
        let mut a = Archive::new();
        a.insert("video/clip1.mkv", b"clip1 data".to_vec());
        a.insert("video/clip2.mkv", b"clip2 data".to_vec());
        a.insert("audio/track.flac", b"audio data".to_vec());
        a
    }

    #[test]
    fn test_extract_by_path_found() {
        let archive = make_archive();
        let data = PartialRestorer::extract_by_path(&archive, "video/clip1.mkv");
        assert_eq!(data, Some(b"clip1 data".to_vec()));
    }

    #[test]
    fn test_extract_by_path_not_found() {
        let archive = make_archive();
        let data = PartialRestorer::extract_by_path(&archive, "nonexistent.mkv");
        assert!(data.is_none());
    }

    #[test]
    fn test_extract_by_path_case_sensitive() {
        let archive = make_archive();
        // Path lookup is case-sensitive.
        assert!(PartialRestorer::extract_by_path(&archive, "Video/clip1.mkv").is_none());
        assert!(PartialRestorer::extract_by_path(&archive, "video/clip1.mkv").is_some());
    }

    #[test]
    fn test_extract_batch() {
        let archive = make_archive();
        let batch = PartialRestorer::extract_batch(
            &archive,
            &["video/clip1.mkv", "missing.wav", "audio/track.flac"],
        );
        assert_eq!(batch.len(), 3);
        assert!(batch[0].1.is_some());
        assert!(batch[1].1.is_none());
        assert!(batch[2].1.is_some());
    }

    #[test]
    fn test_extract_by_prefix() {
        let archive = make_archive();
        let mut results = PartialRestorer::extract_by_prefix(&archive, "video/");
        results.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "video/clip1.mkv");
        assert_eq!(results[1].0, "video/clip2.mkv");
    }

    #[test]
    fn test_archive_insert_replaces() {
        let mut a = Archive::new();
        a.insert("file.mp4", b"v1".to_vec());
        a.insert("file.mp4", b"v2".to_vec());
        assert_eq!(a.len(), 1);
        assert_eq!(
            PartialRestorer::extract_by_path(&a, "file.mp4"),
            Some(b"v2".to_vec())
        );
    }

    #[test]
    fn test_archive_remove() {
        let mut a = Archive::new();
        a.insert("x.mkv", b"data".to_vec());
        let removed = a.remove("x.mkv");
        assert_eq!(removed, Some(b"data".to_vec()));
        assert!(a.is_empty());
    }

    #[test]
    fn test_archive_empty() {
        let a = Archive::new();
        assert!(a.is_empty());
        assert_eq!(a.len(), 0);
    }
}
