//! Content-addressable deduplication storage for OxiMedia.
//!
//! Provides hash-based object addressing, chunk-level deduplication,
//! and reference counting to safely delete unreferenced chunks.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

/// Hash algorithm used to content-address chunks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HashAlgorithm {
    /// SHA-256 (32 bytes)
    Sha256,
    /// BLAKE3 (32 bytes, faster)
    Blake3,
}

impl HashAlgorithm {
    /// Returns the expected digest length in bytes
    pub fn digest_len(&self) -> usize {
        match self {
            HashAlgorithm::Sha256 => 32,
            HashAlgorithm::Blake3 => 32,
        }
    }

    /// Returns the algorithm name
    pub fn name(&self) -> &'static str {
        match self {
            HashAlgorithm::Sha256 => "sha256",
            HashAlgorithm::Blake3 => "blake3",
        }
    }
}

/// A content digest (fixed-length hash)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContentHash {
    algorithm: HashAlgorithm,
    digest: Vec<u8>,
}

impl ContentHash {
    /// Creates a content hash from raw bytes
    pub fn new(algorithm: HashAlgorithm, digest: Vec<u8>) -> Self {
        Self { algorithm, digest }
    }

    /// Returns the hex string representation
    pub fn hex(&self) -> String {
        self.digest.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Returns the algorithm
    pub fn algorithm(&self) -> HashAlgorithm {
        self.algorithm
    }

    /// Returns a reference to the raw digest bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.digest
    }

    /// Creates a deterministic test hash from a u64 seed
    pub fn from_seed(seed: u64) -> Self {
        let mut digest = vec![0u8; 32];
        let bytes = seed.to_le_bytes();
        for (i, b) in bytes.iter().enumerate() {
            digest[i] = *b;
        }
        Self::new(HashAlgorithm::Sha256, digest)
    }
}

impl std::fmt::Display for ContentHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.algorithm.name(), self.hex())
    }
}

/// A stored chunk of data
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Content hash (used as the storage key)
    pub hash: ContentHash,
    /// Actual chunk data
    pub data: Vec<u8>,
    /// Number of logical objects referencing this chunk
    pub ref_count: u64,
}

impl Chunk {
    /// Creates a new chunk with ref_count = 1
    pub fn new(hash: ContentHash, data: Vec<u8>) -> Self {
        Self {
            hash,
            data,
            ref_count: 1,
        }
    }

    /// Returns the chunk size in bytes
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the chunk has no references and can be deleted
    pub fn is_orphaned(&self) -> bool {
        self.ref_count == 0
    }
}

/// A logical object composed of an ordered list of chunk hashes
#[derive(Debug, Clone)]
pub struct DedupObject {
    /// Object key (logical name)
    pub key: String,
    /// Ordered chunk hashes composing this object
    pub chunks: Vec<ContentHash>,
    /// Total logical size (sum of chunk sizes)
    pub logical_size: u64,
    /// Content type
    pub content_type: Option<String>,
}

impl DedupObject {
    /// Creates a new object
    pub fn new(key: impl Into<String>, chunks: Vec<ContentHash>, logical_size: u64) -> Self {
        Self {
            key: key.into(),
            chunks,
            logical_size,
            content_type: None,
        }
    }

    /// Sets the content type
    pub fn with_content_type(mut self, ct: impl Into<String>) -> Self {
        self.content_type = Some(ct.into());
        self
    }

    /// Returns the number of chunks
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }
}

/// Storage statistics
#[derive(Debug, Clone, Default)]
pub struct DedupStats {
    /// Total number of stored chunks
    pub total_chunks: u64,
    /// Total number of logical objects
    pub total_objects: u64,
    /// Total physical bytes stored (sum of unique chunk sizes)
    pub physical_bytes: u64,
    /// Total logical bytes (sum of all object sizes, including duplicates)
    pub logical_bytes: u64,
    /// Number of deduplicated (saved) bytes
    pub dedup_saved_bytes: u64,
}

impl DedupStats {
    /// Returns the deduplication ratio (logical / physical)
    pub fn dedup_ratio(&self) -> f64 {
        if self.physical_bytes == 0 {
            1.0
        } else {
            self.logical_bytes as f64 / self.physical_bytes as f64
        }
    }

    /// Returns the space savings percentage
    pub fn savings_percent(&self) -> f64 {
        if self.logical_bytes == 0 {
            0.0
        } else {
            (self.dedup_saved_bytes as f64 / self.logical_bytes as f64) * 100.0
        }
    }
}

/// Content-addressable deduplication store
pub struct DedupStore {
    /// chunk hash → Chunk
    chunks: HashMap<ContentHash, Chunk>,
    /// object key → DedupObject
    objects: HashMap<String, DedupObject>,
}

impl DedupStore {
    /// Creates a new empty dedup store
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
            objects: HashMap::new(),
        }
    }

    /// Stores a chunk, incrementing ref_count if it already exists.
    /// Returns true if the chunk was newly stored (dedup miss).
    pub fn store_chunk(&mut self, hash: ContentHash, data: Vec<u8>) -> bool {
        if let Some(existing) = self.chunks.get_mut(&hash) {
            existing.ref_count += 1;
            false // dedup hit
        } else {
            self.chunks.insert(hash.clone(), Chunk::new(hash, data));
            true // new chunk
        }
    }

    /// Retrieves a chunk by its hash
    pub fn get_chunk(&self, hash: &ContentHash) -> Option<&Chunk> {
        self.chunks.get(hash)
    }

    /// Decrements the ref_count of a chunk. Returns true if the chunk was removed.
    pub fn release_chunk(&mut self, hash: &ContentHash) -> bool {
        if let Some(chunk) = self.chunks.get_mut(hash) {
            if chunk.ref_count > 0 {
                chunk.ref_count -= 1;
            }
            if chunk.is_orphaned() {
                self.chunks.remove(hash);
                return true;
            }
        }
        false
    }

    /// Stores a logical object and ensures all referenced chunks have their
    /// ref_counts incremented (chunks must already be stored).
    ///
    /// If an object with the same key already exists, it is replaced and old
    /// chunk references are decremented.
    pub fn put_object(&mut self, object: DedupObject) {
        // If replacing, release old chunks
        if let Some(old) = self.objects.remove(&object.key) {
            for h in &old.chunks {
                self.release_chunk(h);
            }
        }
        // Increment ref counts for new object's chunks
        for h in &object.chunks {
            if let Some(chunk) = self.chunks.get_mut(h) {
                chunk.ref_count += 1;
            }
        }
        self.objects.insert(object.key.clone(), object);
    }

    /// Retrieves a logical object by key
    pub fn get_object(&self, key: &str) -> Option<&DedupObject> {
        self.objects.get(key)
    }

    /// Deletes a logical object, releasing all its chunk references
    pub fn delete_object(&mut self, key: &str) -> bool {
        if let Some(obj) = self.objects.remove(key) {
            for h in &obj.chunks {
                self.release_chunk(h);
            }
            true
        } else {
            false
        }
    }

    /// Collects orphaned chunks (ref_count == 0).
    /// Returns the number of chunks removed.
    pub fn gc(&mut self) -> usize {
        let orphaned: Vec<ContentHash> = self
            .chunks
            .iter()
            .filter(|(_, c)| c.is_orphaned())
            .map(|(h, _)| h.clone())
            .collect();
        let count = orphaned.len();
        for h in orphaned {
            self.chunks.remove(&h);
        }
        count
    }

    /// Returns storage statistics
    pub fn stats(&self) -> DedupStats {
        let physical_bytes: u64 = self.chunks.values().map(|c| c.size() as u64).sum();
        let logical_bytes: u64 = self.objects.values().map(|o| o.logical_size).sum();
        let dedup_saved_bytes = logical_bytes.saturating_sub(physical_bytes);
        DedupStats {
            total_chunks: self.chunks.len() as u64,
            total_objects: self.objects.len() as u64,
            physical_bytes,
            logical_bytes,
            dedup_saved_bytes,
        }
    }

    /// Returns the number of unique stored chunks
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Returns the number of stored objects
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// Returns true if the given chunk hash is present
    pub fn has_chunk(&self, hash: &ContentHash) -> bool {
        self.chunks.contains_key(hash)
    }
}

impl Default for DedupStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(seed: u64) -> ContentHash {
        ContentHash::from_seed(seed)
    }

    fn chunk(seed: u64, size: usize) -> (ContentHash, Vec<u8>) {
        (hash(seed), vec![seed as u8; size])
    }

    #[test]
    fn test_hash_algorithm_digest_len() {
        assert_eq!(HashAlgorithm::Sha256.digest_len(), 32);
        assert_eq!(HashAlgorithm::Blake3.digest_len(), 32);
    }

    #[test]
    fn test_hash_algorithm_name() {
        assert_eq!(HashAlgorithm::Sha256.name(), "sha256");
        assert_eq!(HashAlgorithm::Blake3.name(), "blake3");
    }

    #[test]
    fn test_content_hash_hex() {
        let h = ContentHash::new(HashAlgorithm::Sha256, vec![0xDE, 0xAD]);
        assert_eq!(&h.hex()[..4], "dead");
    }

    #[test]
    fn test_content_hash_display() {
        let h = ContentHash::from_seed(42);
        let s = h.to_string();
        assert!(s.starts_with("sha256:"));
    }

    #[test]
    fn test_content_hash_equality() {
        assert_eq!(hash(1), hash(1));
        assert_ne!(hash(1), hash(2));
    }

    #[test]
    fn test_chunk_new_ref_count() {
        let (h, d) = chunk(1, 100);
        let c = Chunk::new(h, d);
        assert_eq!(c.ref_count, 1);
        assert_eq!(c.size(), 100);
        assert!(!c.is_orphaned());
    }

    #[test]
    fn test_chunk_orphaned() {
        let (h, d) = chunk(2, 10);
        let mut c = Chunk::new(h, d);
        c.ref_count = 0;
        assert!(c.is_orphaned());
    }

    #[test]
    fn test_dedup_store_chunk_dedup() {
        let mut store = DedupStore::new();
        let (h, d) = chunk(1, 64);
        let is_new = store.store_chunk(h.clone(), d.clone());
        assert!(is_new);
        let is_new2 = store.store_chunk(h.clone(), d.clone());
        assert!(!is_new2);
        // ref count should be 2 now
        assert_eq!(
            store.get_chunk(&h).expect("chunk should exist").ref_count,
            2
        );
    }

    #[test]
    fn test_dedup_store_release_chunk() {
        let mut store = DedupStore::new();
        let (h, d) = chunk(3, 32);
        store.store_chunk(h.clone(), d);
        let removed = store.release_chunk(&h);
        // ref_count was 1, now 0 → should be removed
        assert!(removed);
        assert!(!store.has_chunk(&h));
    }

    #[test]
    fn test_dedup_store_get_object() {
        let mut store = DedupStore::new();
        let (h, d) = chunk(10, 512);
        store.store_chunk(h.clone(), d);
        let obj = DedupObject::new("video.mp4", vec![h.clone()], 512);
        store.put_object(obj);
        let fetched = store.get_object("video.mp4").expect("object should exist");
        assert_eq!(fetched.chunk_count(), 1);
        assert_eq!(fetched.logical_size, 512);
    }

    #[test]
    fn test_dedup_store_delete_object_releases_chunks() {
        let mut store = DedupStore::new();
        let (h, d) = chunk(20, 100);
        store.store_chunk(h.clone(), d);
        let obj = DedupObject::new("img.png", vec![h.clone()], 100);
        store.put_object(obj);
        // put_object increments ref; store_chunk starts at 1 → total 2
        assert_eq!(
            store.get_chunk(&h).expect("chunk should exist").ref_count,
            2
        );
        store.delete_object("img.png");
        // After delete, ref_count drops by 1 → 1 (store_chunk added 1)
        assert!(store.has_chunk(&h));
    }

    #[test]
    fn test_dedup_store_gc() {
        let mut store = DedupStore::new();
        let (h, d) = chunk(99, 50);
        store.store_chunk(h.clone(), d);
        // Force ref_count to 0 directly via release
        store.release_chunk(&h);
        // Chunk should already be gone after release
        assert!(!store.has_chunk(&h));
    }

    #[test]
    fn test_dedup_store_stats_ratio() {
        let mut store = DedupStore::new();
        // Store one 100-byte chunk referenced by two objects
        let (h, d) = chunk(7, 100);
        store.store_chunk(h.clone(), d);
        let o1 = DedupObject::new("a.mp4", vec![h.clone()], 100);
        let o2 = DedupObject::new("b.mp4", vec![h.clone()], 100);
        store.put_object(o1);
        store.put_object(o2);
        let stats = store.stats();
        // Physical: 100 bytes, logical: 200 bytes
        assert!(stats.dedup_ratio() >= 1.9);
        assert!(stats.savings_percent() > 0.0);
    }

    #[test]
    fn test_dedup_stats_dedup_ratio_zero_physical() {
        let stats = DedupStats::default();
        assert_eq!(stats.dedup_ratio(), 1.0);
    }

    #[test]
    fn test_dedup_object_content_type() {
        let obj = DedupObject::new("clip.webm", vec![], 0).with_content_type("video/webm");
        assert_eq!(obj.content_type.as_deref(), Some("video/webm"));
    }
}
