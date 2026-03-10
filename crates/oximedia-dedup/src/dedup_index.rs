//! Persistent-style deduplication index.
//!
//! Tracks content hashes together with occurrence counts and space-saving
//! metrics.  The index operates entirely in memory; persistence to disk can
//! be added by serialising `DedupIndex` with serde.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ---------------------------------------------------------------------------
// DedupEntry
// ---------------------------------------------------------------------------

/// A single entry in the deduplication index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DedupEntry {
    /// Unique identifier assigned by the index.
    pub id: u64,
    /// Content hash (arbitrary bytes, e.g. BLAKE3 digest).
    pub content_hash: Vec<u8>,
    /// Size of the deduplicated content in bytes.
    pub size_bytes: u64,
    /// Unix epoch timestamp when this hash was first seen.
    pub first_seen_epoch: u64,
    /// Number of times this hash has been seen (≥ 1 after creation).
    pub occurrence_count: u32,
}

impl DedupEntry {
    /// Returns `true` if this hash has been seen more than once.
    #[must_use]
    pub fn is_duplicate(&self) -> bool {
        self.occurrence_count > 1
    }

    /// Space saved by deduplication.
    ///
    /// If the content appeared N times, only one copy is stored.
    /// Savings = `(N - 1) * size_bytes`.
    #[must_use]
    pub fn space_savings(&self) -> u64 {
        if self.occurrence_count <= 1 {
            return 0;
        }
        (self.occurrence_count as u64 - 1).saturating_mul(self.size_bytes)
    }

    /// Return the last-seen epoch, which equals `first_seen_epoch` until we
    /// track updates (here we simply alias for interface completeness).
    #[must_use]
    pub fn first_seen(&self) -> u64 {
        self.first_seen_epoch
    }

    /// Hex representation of the content hash.
    #[must_use]
    pub fn hash_hex(&self) -> String {
        self.content_hash
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// DedupIndex
// ---------------------------------------------------------------------------

/// In-memory deduplication index.
///
/// Each unique content hash is stored once; subsequent insertions increment
/// the occurrence counter.
pub struct DedupIndex {
    /// All known entries, ordered by insertion.
    pub entries: Vec<DedupEntry>,
    /// Next ID to assign.
    pub next_id: u64,
}

impl DedupIndex {
    /// Create a new, empty index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_id: 1,
        }
    }

    /// Add a content hash to the index, or increment its counter if already present.
    ///
    /// Returns the `id` of the (existing or newly created) entry.
    pub fn add_or_increment(&mut self, hash: Vec<u8>, size_bytes: u64, epoch: u64) -> u64 {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.content_hash == hash) {
            entry.occurrence_count += 1;
            return entry.id;
        }

        let id = self.next_id;
        self.next_id += 1;
        self.entries.push(DedupEntry {
            id,
            content_hash: hash,
            size_bytes,
            first_seen_epoch: epoch,
            occurrence_count: 1,
        });
        id
    }

    /// Find an entry by its content hash.
    #[must_use]
    pub fn find_by_hash(&self, hash: &[u8]) -> Option<&DedupEntry> {
        self.entries.iter().find(|e| e.content_hash == hash)
    }

    /// Find an entry by its assigned ID.
    #[must_use]
    pub fn find_by_id(&self, id: u64) -> Option<&DedupEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Return all entries that are duplicates (occurrence_count > 1).
    #[must_use]
    pub fn find_duplicates(&self) -> Vec<&DedupEntry> {
        self.entries.iter().filter(|e| e.is_duplicate()).collect()
    }

    /// Total space saved across all duplicate entries.
    #[must_use]
    pub fn total_space_savings(&self) -> u64 {
        self.entries.iter().map(|e| e.space_savings()).sum()
    }

    /// Number of unique content hashes in the index.
    #[must_use]
    pub fn unique_count(&self) -> usize {
        self.entries.len()
    }

    /// Total number of content insertions (sum of all occurrence counts).
    #[must_use]
    pub fn total_insertions(&self) -> u64 {
        self.entries.iter().map(|e| e.occurrence_count as u64).sum()
    }

    /// Remove an entry by hash.  Returns `true` if an entry was removed.
    pub fn remove_by_hash(&mut self, hash: &[u8]) -> bool {
        if let Some(pos) = self.entries.iter().position(|e| e.content_hash == hash) {
            self.entries.swap_remove(pos);
            true
        } else {
            false
        }
    }

    /// Clear the entire index.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.next_id = 1;
    }

    /// Return entries sorted by occurrence count (descending).
    #[must_use]
    pub fn most_common(&self) -> Vec<&DedupEntry> {
        let mut sorted: Vec<&DedupEntry> = self.entries.iter().collect();
        sorted.sort_by(|a, b| b.occurrence_count.cmp(&a.occurrence_count));
        sorted
    }

    /// Report: percentage of insertions that were duplicates.
    ///
    /// Returns `0.0` if no insertions have been made.
    #[must_use]
    pub fn duplicate_rate(&self) -> f64 {
        let total = self.total_insertions();
        if total == 0 {
            return 0.0;
        }
        let unique = self.unique_count() as u64;
        let dupes = total.saturating_sub(unique);
        dupes as f64 / total as f64
    }
}

impl Default for DedupIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(s: &str) -> Vec<u8> {
        s.as_bytes().to_vec()
    }

    // ---- DedupEntry tests ----

    #[test]
    fn test_entry_is_duplicate_false_when_once() {
        let e = DedupEntry {
            id: 1,
            content_hash: hash("abc"),
            size_bytes: 1024,
            first_seen_epoch: 0,
            occurrence_count: 1,
        };
        assert!(!e.is_duplicate());
    }

    #[test]
    fn test_entry_is_duplicate_true_when_multiple() {
        let e = DedupEntry {
            id: 1,
            content_hash: hash("abc"),
            size_bytes: 1024,
            first_seen_epoch: 0,
            occurrence_count: 3,
        };
        assert!(e.is_duplicate());
    }

    #[test]
    fn test_entry_space_savings_zero_when_once() {
        let e = DedupEntry {
            id: 1,
            content_hash: hash("abc"),
            size_bytes: 500,
            first_seen_epoch: 0,
            occurrence_count: 1,
        };
        assert_eq!(e.space_savings(), 0);
    }

    #[test]
    fn test_entry_space_savings_correct() {
        let e = DedupEntry {
            id: 1,
            content_hash: hash("abc"),
            size_bytes: 1000,
            first_seen_epoch: 0,
            occurrence_count: 4,
        };
        // (4-1) * 1000 = 3000
        assert_eq!(e.space_savings(), 3000);
    }

    #[test]
    fn test_entry_hash_hex() {
        let e = DedupEntry {
            id: 1,
            content_hash: vec![0xDE, 0xAD, 0xBE, 0xEF],
            size_bytes: 0,
            first_seen_epoch: 0,
            occurrence_count: 1,
        };
        assert_eq!(e.hash_hex(), "deadbeef");
    }

    // ---- DedupIndex tests ----

    #[test]
    fn test_index_new_empty() {
        let idx = DedupIndex::new();
        assert_eq!(idx.unique_count(), 0);
        assert_eq!(idx.total_insertions(), 0);
        assert_eq!(idx.total_space_savings(), 0);
    }

    #[test]
    fn test_add_or_increment_new_entry() {
        let mut idx = DedupIndex::new();
        let id = idx.add_or_increment(hash("file_a"), 100, 1000);
        assert_eq!(idx.unique_count(), 1);
        assert_eq!(id, 1);
    }

    #[test]
    fn test_add_or_increment_existing_entry() {
        let mut idx = DedupIndex::new();
        let id1 = idx.add_or_increment(hash("file_a"), 100, 1000);
        let id2 = idx.add_or_increment(hash("file_a"), 100, 2000);
        assert_eq!(id1, id2, "Same hash should return same ID");
        assert_eq!(idx.unique_count(), 1, "Should still be one unique entry");
        let entry = idx
            .find_by_hash(&hash("file_a"))
            .expect("operation should succeed");
        assert_eq!(entry.occurrence_count, 2);
    }

    #[test]
    fn test_find_by_hash_found() {
        let mut idx = DedupIndex::new();
        idx.add_or_increment(hash("alpha"), 200, 0);
        let entry = idx.find_by_hash(&hash("alpha"));
        assert!(entry.is_some());
        assert_eq!(entry.expect("operation should succeed").size_bytes, 200);
    }

    #[test]
    fn test_find_by_hash_not_found() {
        let idx = DedupIndex::new();
        assert!(idx.find_by_hash(&hash("missing")).is_none());
    }

    #[test]
    fn test_find_by_id() {
        let mut idx = DedupIndex::new();
        let id = idx.add_or_increment(hash("entry_1"), 512, 100);
        let entry = idx.find_by_id(id);
        assert!(entry.is_some());
        assert_eq!(
            entry.expect("operation should succeed").content_hash,
            hash("entry_1")
        );
    }

    #[test]
    fn test_find_duplicates() {
        let mut idx = DedupIndex::new();
        idx.add_or_increment(hash("unique"), 10, 0);
        idx.add_or_increment(hash("dup"), 20, 0);
        idx.add_or_increment(hash("dup"), 20, 1);
        idx.add_or_increment(hash("dup"), 20, 2);

        let dups = idx.find_duplicates();
        assert_eq!(dups.len(), 1);
        assert_eq!(dups[0].content_hash, hash("dup"));
        assert_eq!(dups[0].occurrence_count, 3);
    }

    #[test]
    fn test_total_space_savings() {
        let mut idx = DedupIndex::new();
        // 1 occurrence → 0 savings
        idx.add_or_increment(hash("a"), 100, 0);
        // 3 occurrences → (3-1)*200 = 400 savings
        idx.add_or_increment(hash("b"), 200, 0);
        idx.add_or_increment(hash("b"), 200, 1);
        idx.add_or_increment(hash("b"), 200, 2);

        assert_eq!(idx.total_space_savings(), 400);
    }

    #[test]
    fn test_remove_by_hash() {
        let mut idx = DedupIndex::new();
        idx.add_or_increment(hash("to_remove"), 50, 0);
        assert_eq!(idx.unique_count(), 1);
        let removed = idx.remove_by_hash(&hash("to_remove"));
        assert!(removed);
        assert_eq!(idx.unique_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_returns_false() {
        let mut idx = DedupIndex::new();
        assert!(!idx.remove_by_hash(&hash("ghost")));
    }

    #[test]
    fn test_clear() {
        let mut idx = DedupIndex::new();
        idx.add_or_increment(hash("x"), 10, 0);
        idx.add_or_increment(hash("y"), 20, 0);
        idx.clear();
        assert_eq!(idx.unique_count(), 0);
        assert_eq!(idx.next_id, 1);
    }

    #[test]
    fn test_duplicate_rate() {
        let mut idx = DedupIndex::new();
        // 1 unique, seen 3 times → 2 duplicate insertions out of 3 total ≈ 0.667
        idx.add_or_increment(hash("h"), 100, 0);
        idx.add_or_increment(hash("h"), 100, 1);
        idx.add_or_increment(hash("h"), 100, 2);

        let rate = idx.duplicate_rate();
        let expected = 2.0 / 3.0;
        assert!((rate - expected).abs() < 1e-9);
    }

    #[test]
    fn test_most_common_ordering() {
        let mut idx = DedupIndex::new();
        idx.add_or_increment(hash("rare"), 10, 0);
        idx.add_or_increment(hash("common"), 100, 0);
        idx.add_or_increment(hash("common"), 100, 1);
        idx.add_or_increment(hash("common"), 100, 2);
        idx.add_or_increment(hash("mid"), 50, 0);
        idx.add_or_increment(hash("mid"), 50, 1);

        let mc = idx.most_common();
        assert_eq!(mc[0].content_hash, hash("common"));
        assert_eq!(mc[1].content_hash, hash("mid"));
        assert_eq!(mc[2].content_hash, hash("rare"));
    }

    #[test]
    fn test_total_insertions() {
        let mut idx = DedupIndex::new();
        idx.add_or_increment(hash("a"), 1, 0);
        idx.add_or_increment(hash("a"), 1, 1);
        idx.add_or_increment(hash("b"), 1, 0);
        // a: 2, b: 1 → total = 3
        assert_eq!(idx.total_insertions(), 3);
    }
}
