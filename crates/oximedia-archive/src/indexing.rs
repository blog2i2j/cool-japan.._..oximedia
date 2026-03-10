//! Archive indexing and full-text search.
//!
//! Provides an in-memory index of archive entries with tag-based and
//! substring-based query support.

#![allow(dead_code)]

// ── Data structures ──────────────────────────────────────────────────────────

/// A single entry in the archive index.
#[derive(Debug, Clone)]
pub struct ArchiveEntry {
    /// Unique numeric identifier.
    pub id: u64,
    /// Filesystem path of the asset.
    pub path: String,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Hex checksum (any algorithm).
    pub checksum: String,
    /// MIME-style media type, e.g. `"video/mp4"`.
    pub media_type: String,
    /// Free-form tags.
    pub tags: Vec<String>,
    /// Unix timestamp of when the entry was indexed.
    pub indexed_at: u64,
}

impl ArchiveEntry {
    /// Creates a new entry with an empty checksum, no tags, and timestamp 0.
    #[must_use]
    pub fn new(id: u64, path: &str, size: u64, media_type: &str) -> Self {
        Self {
            id,
            path: path.to_string(),
            size_bytes: size,
            checksum: String::new(),
            media_type: media_type.to_string(),
            tags: Vec::new(),
            indexed_at: 0,
        }
    }

    /// Adds a tag to the entry (deduplicates automatically).
    pub fn add_tag(&mut self, tag: &str) {
        let tag = tag.to_string();
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    /// Returns `true` if the entry matches a case-insensitive substring query
    /// against the path, media type, or any tag.
    #[must_use]
    pub fn matches_query(&self, query: &str) -> bool {
        if query.is_empty() {
            return true;
        }
        let q = query.to_lowercase();
        self.path.to_lowercase().contains(&q)
            || self.media_type.to_lowercase().contains(&q)
            || self.tags.iter().any(|t| t.to_lowercase().contains(&q))
    }
}

// ── ArchiveIndex ─────────────────────────────────────────────────────────────

/// In-memory archive index.
#[derive(Debug, Default)]
pub struct ArchiveIndex {
    entries: Vec<ArchiveEntry>,
    next_id: u64,
}

impl ArchiveIndex {
    /// Creates an empty index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an asset to the index, returning its assigned ID.
    pub fn add(&mut self, path: &str, size: u64, media_type: &str) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.entries
            .push(ArchiveEntry::new(id, path, size, media_type));
        id
    }

    /// Searches entries by substring query (path, media type, tags).
    #[must_use]
    pub fn search(&self, query: &str) -> Vec<&ArchiveEntry> {
        self.entries
            .iter()
            .filter(|e| e.matches_query(query))
            .collect()
    }

    /// Returns all entries of a given media type (exact match, case-insensitive).
    #[must_use]
    pub fn by_type(&self, media_type: &str) -> Vec<&ArchiveEntry> {
        let mt = media_type.to_lowercase();
        self.entries
            .iter()
            .filter(|e| e.media_type.to_lowercase() == mt)
            .collect()
    }

    /// Returns all entries that have the given tag (case-insensitive).
    #[must_use]
    pub fn by_tag(&self, tag: &str) -> Vec<&ArchiveEntry> {
        let t = tag.to_lowercase();
        self.entries
            .iter()
            .filter(|e| e.tags.iter().any(|et| et.to_lowercase() == t))
            .collect()
    }

    /// Removes an entry by ID.  Returns `true` if an entry was removed.
    pub fn remove(&mut self, id: u64) -> bool {
        let before = self.entries.len();
        self.entries.retain(|e| e.id != id);
        self.entries.len() < before
    }

    /// Returns the sum of all entry sizes.
    #[must_use]
    pub fn total_size(&self) -> u64 {
        self.entries.iter().map(|e| e.size_bytes).sum()
    }

    /// Returns the number of indexed entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns a reference to the entry with the given ID, if present.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&ArchiveEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Returns a mutable reference to the entry with the given ID, if present.
    #[must_use]
    pub fn get_mut(&mut self, id: u64) -> Option<&mut ArchiveEntry> {
        self.entries.iter_mut().find(|e| e.id == id)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_index() -> ArchiveIndex {
        let mut idx = ArchiveIndex::new();
        let id0 = idx.add("/archive/film.mov", 1_000_000, "video/quicktime");
        let id1 = idx.add("/archive/audio.wav", 500_000, "audio/wav");
        let id2 = idx.add("/archive/photo.tiff", 200_000, "image/tiff");
        idx.get_mut(id0)
            .expect("get_mut should succeed")
            .add_tag("rushes");
        idx.get_mut(id1)
            .expect("get_mut should succeed")
            .add_tag("score");
        idx.get_mut(id2)
            .expect("get_mut should succeed")
            .add_tag("rushes");
        idx.get_mut(id2)
            .expect("get_mut should succeed")
            .add_tag("still");
        let _ = (id0, id1, id2);
        idx
    }

    #[test]
    fn test_index_add_assigns_sequential_ids() {
        let mut idx = ArchiveIndex::new();
        let a = idx.add("/a", 1, "video/mp4");
        let b = idx.add("/b", 1, "video/mp4");
        assert_eq!(a, 0);
        assert_eq!(b, 1);
    }

    #[test]
    fn test_index_len() {
        let idx = build_index();
        assert_eq!(idx.len(), 3);
    }

    #[test]
    fn test_index_total_size() {
        let idx = build_index();
        assert_eq!(idx.total_size(), 1_700_000);
    }

    #[test]
    fn test_search_by_path_substring() {
        let idx = build_index();
        let results = idx.search("film");
        assert_eq!(results.len(), 1);
        assert!(results[0].path.contains("film"));
    }

    #[test]
    fn test_search_case_insensitive() {
        let idx = build_index();
        let results = idx.search("AUDIO");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_search_empty_query_returns_all() {
        let idx = build_index();
        assert_eq!(idx.search("").len(), 3);
    }

    #[test]
    fn test_by_type_exact() {
        let idx = build_index();
        let results = idx.by_type("audio/wav");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_by_type_case_insensitive() {
        let idx = build_index();
        let results = idx.by_type("IMAGE/TIFF");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_by_tag() {
        let idx = build_index();
        let results = idx.by_tag("rushes");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_remove_entry() {
        let mut idx = build_index();
        let id_to_remove = idx.add("/tmp/extra", 42, "application/octet-stream");
        assert!(idx.remove(id_to_remove));
        assert!(idx.get(id_to_remove).is_none());
    }

    #[test]
    fn test_remove_nonexistent_returns_false() {
        let mut idx = build_index();
        assert!(!idx.remove(9999));
    }

    #[test]
    fn test_add_tag_deduplication() {
        let mut entry = ArchiveEntry::new(0, "/test", 1, "video/mp4");
        entry.add_tag("raw");
        entry.add_tag("raw");
        assert_eq!(entry.tags.len(), 1);
    }

    #[test]
    fn test_matches_query_by_tag() {
        let mut entry = ArchiveEntry::new(0, "/x", 1, "video/mp4");
        entry.add_tag("documentary");
        assert!(entry.matches_query("documentary"));
        assert!(!entry.matches_query("fiction"));
    }
}
