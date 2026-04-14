//! Additional storage utility APIs implementing TODO items.
//!
//! This module provides simpler, task-focused API facades:
//! - `WalSerializer` — binary serialize/deserialize WAL entries
//! - `SimpleDedupStore` — SHA-256 content-addressed put/get_by_hash
//! - `AccessLogAnalyzer` — `top_keys(n)` from access frequency
//! - `StorageMetricsSnapshot` — lightweight metrics snapshot
//! - `StorageNamespace` — prefix-scoped in-memory object store
//! - `VersionedObject` / `SimpleVersionStore` — key-versioned blobs
//! - `SimpleRetentionManager` — age-based expiry purge
//! - `CompressedStore` — identity-stub compress/decompress wrapper
//! - `IntegrityChecker` — SHA-256 batch verify with `IntegrityViolation`

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use sha2::{Digest, Sha256};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// WalSerializer — binary format
// ---------------------------------------------------------------------------
// Wire format per entry:
//   [op_byte: u8]
//   [key_len: u16 BE]
//   [key: key_len bytes]
//   [data_len: u32 BE]
//   [data: data_len bytes]
//
// op_byte values:  0 = Put,  1 = Delete

/// WAL operation discriminant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimpleWalOp {
    /// Store data at key.
    Put { key: String, data: Vec<u8> },
    /// Remove key.
    Delete { key: String },
}

/// Binary serializer / deserializer for WAL entries.
pub struct WalSerializer;

impl WalSerializer {
    const OP_PUT: u8 = 0;
    const OP_DELETE: u8 = 1;

    /// Serializes a single WAL operation to bytes.
    ///
    /// Format: `[op_byte][key_len u16 BE][key][data_len u32 BE][data]`.
    /// For `Delete`, data_len is 0 and no data bytes follow.
    #[must_use]
    pub fn serialize(op: &SimpleWalOp) -> Vec<u8> {
        match op {
            SimpleWalOp::Put { key, data } => {
                let key_bytes = key.as_bytes();
                let mut buf = Vec::with_capacity(1 + 2 + key_bytes.len() + 4 + data.len());
                buf.push(Self::OP_PUT);
                buf.extend_from_slice(&(key_bytes.len() as u16).to_be_bytes());
                buf.extend_from_slice(key_bytes);
                buf.extend_from_slice(&(data.len() as u32).to_be_bytes());
                buf.extend_from_slice(data);
                buf
            }
            SimpleWalOp::Delete { key } => {
                let key_bytes = key.as_bytes();
                let mut buf = Vec::with_capacity(1 + 2 + key_bytes.len() + 4);
                buf.push(Self::OP_DELETE);
                buf.extend_from_slice(&(key_bytes.len() as u16).to_be_bytes());
                buf.extend_from_slice(key_bytes);
                buf.extend_from_slice(&0u32.to_be_bytes()); // data_len = 0
                buf
            }
        }
    }

    /// Deserializes a sequence of WAL operations from `bytes`.
    ///
    /// Stops at the first parse error.  Returns all successfully parsed ops.
    #[must_use]
    pub fn deserialize(bytes: &[u8]) -> Vec<SimpleWalOp> {
        let mut ops = Vec::new();
        let mut pos = 0usize;

        while pos < bytes.len() {
            // op_byte
            if pos + 1 > bytes.len() {
                break;
            }
            let op_byte = bytes[pos];
            pos += 1;

            // key_len (u16 BE)
            if pos + 2 > bytes.len() {
                break;
            }
            let key_len = u16::from_be_bytes([bytes[pos], bytes[pos + 1]]) as usize;
            pos += 2;

            // key
            if pos + key_len > bytes.len() {
                break;
            }
            let key = match std::str::from_utf8(&bytes[pos..pos + key_len]) {
                Ok(s) => s.to_string(),
                Err(_) => break,
            };
            pos += key_len;

            // data_len (u32 BE)
            if pos + 4 > bytes.len() {
                break;
            }
            let data_len =
                u32::from_be_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
                    as usize;
            pos += 4;

            // data
            if pos + data_len > bytes.len() {
                break;
            }
            let data = bytes[pos..pos + data_len].to_vec();
            pos += data_len;

            let op = match op_byte {
                Self::OP_PUT => SimpleWalOp::Put { key, data },
                Self::OP_DELETE => SimpleWalOp::Delete { key },
                _ => break,
            };
            ops.push(op);
        }
        ops
    }
}

// ---------------------------------------------------------------------------
// SimpleDedupStore — SHA-256 content-addressed store
// ---------------------------------------------------------------------------

/// Content-addressed store that maps SHA-256 digests to byte blobs and
/// tracks named keys → digest mapping for deduplication.
#[derive(Debug, Default)]
pub struct SimpleDedupStore {
    /// hash → data
    blobs: HashMap<[u8; 32], Vec<u8>>,
    /// named key → hash
    index: HashMap<String, [u8; 32]>,
}

impl SimpleDedupStore {
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Stores `value` under `key`.
    ///
    /// Returns the SHA-256 hash of `value`.
    pub fn put(&mut self, key: impl Into<String>, value: Vec<u8>) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(&value);
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        self.blobs.insert(hash, value);
        self.index.insert(key.into(), hash);
        hash
    }

    /// Retrieves data by its SHA-256 hash.
    #[must_use]
    pub fn get_by_hash(&self, hash: &[u8; 32]) -> Option<&[u8]> {
        self.blobs.get(hash).map(|v| v.as_slice())
    }

    /// Retrieves data by its named key.
    #[must_use]
    pub fn get_by_key(&self, key: &str) -> Option<&[u8]> {
        let hash = self.index.get(key)?;
        self.blobs.get(hash).map(|v| v.as_slice())
    }

    /// Returns the hash for the given key, if known.
    #[must_use]
    pub fn hash_of(&self, key: &str) -> Option<[u8; 32]> {
        self.index.get(key).copied()
    }

    /// Number of unique blobs stored.
    #[must_use]
    pub fn blob_count(&self) -> usize {
        self.blobs.len()
    }

    /// Number of named keys tracked.
    #[must_use]
    pub fn key_count(&self) -> usize {
        self.index.len()
    }
}

// ---------------------------------------------------------------------------
// AccessLogAnalyzer — top_keys from access frequency
// ---------------------------------------------------------------------------

/// Lightweight access frequency counter.
#[derive(Debug, Default)]
pub struct AccessLogAnalyzer {
    /// key → access count
    counts: HashMap<String, u64>,
}

impl AccessLogAnalyzer {
    /// Creates an empty analyzer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one access to `key`.
    pub fn record(&mut self, key: impl Into<String>) {
        *self.counts.entry(key.into()).or_insert(0) += 1;
    }

    /// Records `count` accesses to `key` at once.
    pub fn record_n(&mut self, key: impl Into<String>, count: u64) {
        *self.counts.entry(key.into()).or_insert(0) += count;
    }

    /// Returns the top `n` most-accessed keys, sorted by descending count.
    ///
    /// Ties are broken by lexicographic key order.
    #[must_use]
    pub fn top_keys(&self, n: usize) -> Vec<(String, u64)> {
        let mut pairs: Vec<(String, u64)> = self.counts.clone().into_iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        pairs.truncate(n);
        pairs
    }

    /// Total accesses recorded.
    #[must_use]
    pub fn total_accesses(&self) -> u64 {
        self.counts.values().sum()
    }

    /// Number of distinct keys.
    #[must_use]
    pub fn distinct_keys(&self) -> usize {
        self.counts.len()
    }
}

// ---------------------------------------------------------------------------
// StorageMetricsSnapshot
// ---------------------------------------------------------------------------

/// A point-in-time snapshot of key storage metrics.
#[derive(Debug, Clone, Default)]
pub struct StorageMetricsSnapshot {
    /// Total number of objects in the store.
    pub object_count: u64,
    /// Total bytes consumed by all objects.
    pub total_bytes: u64,
    /// Number of successful read operations.
    pub reads: u64,
    /// Number of successful write operations.
    pub writes: u64,
    /// Number of delete operations.
    pub deletes: u64,
    /// Number of errors encountered.
    pub errors: u64,
}

impl StorageMetricsSnapshot {
    /// Creates a snapshot from raw counter values.
    #[must_use]
    pub fn new(
        object_count: u64,
        total_bytes: u64,
        reads: u64,
        writes: u64,
        deletes: u64,
        errors: u64,
    ) -> Self {
        Self {
            object_count,
            total_bytes,
            reads,
            writes,
            deletes,
            errors,
        }
    }

    /// Error rate as a fraction of total operations.
    #[must_use]
    pub fn error_rate(&self) -> f64 {
        let total = self.reads + self.writes + self.deletes;
        if total == 0 {
            0.0
        } else {
            self.errors as f64 / total as f64
        }
    }
}

// ---------------------------------------------------------------------------
// StorageNamespace — prefix-scoped in-memory object store
// ---------------------------------------------------------------------------

/// A prefix-scoped in-memory object store.
///
/// All keys are automatically namespaced with the configured prefix.
#[derive(Debug)]
pub struct StorageNamespace {
    /// The namespace prefix (e.g. `"media/"`).
    prefix: String,
    /// Internal object storage: full-path key → data.
    data: HashMap<String, Vec<u8>>,
}

impl StorageNamespace {
    /// Creates a new namespace with the given prefix.
    #[must_use]
    pub fn new(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
            data: HashMap::new(),
        }
    }

    /// Returns the namespace prefix.
    #[must_use]
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    fn full_key(&self, key: &str) -> String {
        format!("{}{}", self.prefix, key)
    }

    /// Stores `value` at `key` (relative to the namespace prefix).
    pub fn put(&mut self, key: &str, value: Vec<u8>) {
        self.data.insert(self.full_key(key), value);
    }

    /// Retrieves the value stored at `key`, if present.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&[u8]> {
        self.data.get(&self.full_key(key)).map(|v| v.as_slice())
    }

    /// Deletes the object at `key`. Returns `true` if the key existed.
    pub fn delete(&mut self, key: &str) -> bool {
        self.data.remove(&self.full_key(key)).is_some()
    }

    /// Lists all relative keys in this namespace.
    #[must_use]
    pub fn list_keys(&self) -> Vec<String> {
        let prefix_len = self.prefix.len();
        let mut keys: Vec<String> = self
            .data
            .keys()
            .filter_map(|k| {
                if k.starts_with(&self.prefix) {
                    Some(k[prefix_len..].to_string())
                } else {
                    None
                }
            })
            .collect();
        keys.sort();
        keys
    }

    /// Number of objects in this namespace.
    #[must_use]
    pub fn object_count(&self) -> usize {
        self.data.len()
    }
}

// ---------------------------------------------------------------------------
// VersionedObject / SimpleVersionStore
// ---------------------------------------------------------------------------

/// A single versioned snapshot of an object.
#[derive(Debug, Clone)]
pub struct VersionedObject {
    /// Object key.
    pub key: String,
    /// Version number (1-based, monotonically increasing per key).
    pub version: u32,
    /// Object data.
    pub data: Vec<u8>,
    /// Unix timestamp at which this version was created.
    pub created_at: u64,
}

/// Simple key-versioned blob store.
///
/// Each `put` adds a new immutable version.  Old versions are retained and
/// can be retrieved by version number.
#[derive(Debug, Default)]
pub struct SimpleVersionStore {
    /// key → ordered list of versions (oldest first)
    store: HashMap<String, Vec<VersionedObject>>,
}

impl SimpleVersionStore {
    /// Creates an empty version store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Stores a new version of `key` with the given `data` and `created_at` timestamp.
    ///
    /// Returns the version number assigned to this snapshot.
    pub fn put(&mut self, key: impl Into<String>, data: Vec<u8>, created_at: u64) -> u32 {
        let key = key.into();
        let versions = self.store.entry(key.clone()).or_default();
        let version = versions.len() as u32 + 1;
        versions.push(VersionedObject {
            key,
            version,
            data,
            created_at,
        });
        version
    }

    /// Retrieves a specific version of `key`.  Returns `None` if not found.
    #[must_use]
    pub fn get_version(&self, key: &str, version: u32) -> Option<&VersionedObject> {
        self.store.get(key)?.iter().find(|v| v.version == version)
    }

    /// Returns the latest version of `key`, or `None` if the key is unknown.
    #[must_use]
    pub fn latest(&self, key: &str) -> Option<&VersionedObject> {
        self.store.get(key)?.last()
    }

    /// Returns all versions of `key`, oldest first.
    #[must_use]
    pub fn history(&self, key: &str) -> &[VersionedObject] {
        self.store.get(key).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Number of distinct keys tracked.
    #[must_use]
    pub fn key_count(&self) -> usize {
        self.store.len()
    }
}

// ---------------------------------------------------------------------------
// SimpleRetentionManager — age-based expiry
// ---------------------------------------------------------------------------

/// Object metadata used by the retention manager.
#[derive(Debug, Clone)]
pub struct RetainedObject {
    /// Object key.
    pub key: String,
    /// Unix timestamp when the object was created.
    pub created_at: u64,
}

impl RetainedObject {
    /// Creates a new retained object descriptor.
    #[must_use]
    pub fn new(key: impl Into<String>, created_at: u64) -> Self {
        Self {
            key: key.into(),
            created_at,
        }
    }
}

/// Retention manager that tracks object ages and identifies expired objects.
#[derive(Debug)]
pub struct SimpleRetentionManager {
    max_age_secs: u64,
    objects: Vec<RetainedObject>,
}

impl SimpleRetentionManager {
    /// Creates a new retention manager.
    ///
    /// Objects older than `max_age_secs` are considered expired.
    #[must_use]
    pub fn new(max_age_secs: u64) -> Self {
        Self {
            max_age_secs,
            objects: Vec::new(),
        }
    }

    /// Registers an object to be tracked by this manager.
    pub fn track(&mut self, obj: RetainedObject) {
        self.objects.push(obj);
    }

    /// Removes and returns all objects whose age (now_ts - created_at) exceeds
    /// `max_age_secs`.
    ///
    /// `now_ts` is a Unix epoch timestamp in seconds.
    pub fn purge_expired(&mut self, now_ts: u64) -> Vec<RetainedObject> {
        let threshold = now_ts.saturating_sub(self.max_age_secs);
        let mut expired = Vec::new();
        self.objects.retain(|obj| {
            if obj.created_at <= threshold {
                expired.push(obj.clone());
                false
            } else {
                true
            }
        });
        expired
    }

    /// Returns all currently tracked (non-expired) objects.
    #[must_use]
    pub fn tracked(&self) -> &[RetainedObject] {
        &self.objects
    }

    /// Maximum age in seconds.
    #[must_use]
    pub fn max_age_secs(&self) -> u64 {
        self.max_age_secs
    }
}

// ---------------------------------------------------------------------------
// CompressedStore — identity-stub compress/decompress
// ---------------------------------------------------------------------------

/// A simple content store with stub compression.
///
/// Per the COOLJAPAN Pure Rust policy, actual compression must use OxiARC
/// (`oxiarc-*`).  This stub stores data unmodified while presenting the same
/// compressed/decompressed interface.
#[derive(Debug, Default)]
pub struct CompressedStore {
    /// Internally stored (would-be-compressed) data.
    inner: HashMap<String, Vec<u8>>,
}

impl CompressedStore {
    /// Creates a new empty compressed store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Stub LZ4-style compress: returns data unchanged.
    ///
    /// Replace with `oxiarc_lz4::compress(data)` when the OxiARC LZ4
    /// feature is available.
    #[must_use]
    pub fn compress_lz4_stub(data: &[u8]) -> Vec<u8> {
        data.to_vec()
    }

    /// Stub LZ4-style decompress: returns data unchanged.
    ///
    /// Replace with `oxiarc_lz4::decompress(data)` when the OxiARC LZ4
    /// feature is available.
    #[must_use]
    pub fn decompress_lz4_stub(data: &[u8]) -> Vec<u8> {
        data.to_vec()
    }

    /// Stores `data` at `key`, applying the stub compressor.
    pub fn put(&mut self, key: impl Into<String>, data: Vec<u8>) {
        let compressed = Self::compress_lz4_stub(&data);
        self.inner.insert(key.into(), compressed);
    }

    /// Retrieves and decompresses data at `key`.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.inner.get(key).map(|d| Self::decompress_lz4_stub(d))
    }

    /// Number of objects stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

// ---------------------------------------------------------------------------
// IntegrityChecker — SHA-256 batch verify
// ---------------------------------------------------------------------------

/// A violation found during integrity checking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegrityViolation {
    /// Object key that failed verification.
    pub key: String,
    /// Expected SHA-256 hash (32 bytes).
    pub expected: [u8; 32],
    /// Actual SHA-256 hash computed from the stored data.
    pub actual: [u8; 32],
}

impl IntegrityViolation {
    /// Creates a new violation record.
    #[must_use]
    pub fn new(key: impl Into<String>, expected: [u8; 32], actual: [u8; 32]) -> Self {
        Self {
            key: key.into(),
            expected,
            actual,
        }
    }
}

/// Batch SHA-256 integrity checker.
pub struct IntegrityChecker;

impl IntegrityChecker {
    /// Verifies every key in `expected_checksums` against `store`.
    ///
    /// - If a key exists in the store, computes its SHA-256 and compares.
    /// - If a key is absent from the store, records a violation with
    ///   `actual = [0u8; 32]`.
    ///
    /// Returns a `Vec<IntegrityViolation>` for every mismatch found.
    #[must_use]
    pub fn verify(
        store: &HashMap<String, Vec<u8>>,
        expected_checksums: &HashMap<String, [u8; 32]>,
    ) -> Vec<IntegrityViolation> {
        let mut violations = Vec::new();
        for (key, expected) in expected_checksums {
            let actual = match store.get(key.as_str()) {
                Some(data) => {
                    let mut hasher = Sha256::new();
                    hasher.update(data);
                    let result = hasher.finalize();
                    let mut hash = [0u8; 32];
                    hash.copy_from_slice(&result);
                    hash
                }
                None => [0u8; 32],
            };
            if &actual != expected {
                violations.push(IntegrityViolation::new(key.clone(), *expected, actual));
            }
        }
        violations
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- WalSerializer ---

    #[test]
    fn test_wal_serialize_put_roundtrip() {
        let op = SimpleWalOp::Put {
            key: "media/video.mp4".to_string(),
            data: b"hello world".to_vec(),
        };
        let bytes = WalSerializer::serialize(&op);
        let ops = WalSerializer::deserialize(&bytes);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0], op);
    }

    #[test]
    fn test_wal_serialize_delete_roundtrip() {
        let op = SimpleWalOp::Delete {
            key: "media/old.mp4".to_string(),
        };
        let bytes = WalSerializer::serialize(&op);
        let ops = WalSerializer::deserialize(&bytes);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0], op);
    }

    #[test]
    fn test_wal_serialize_multiple_ops() {
        let ops: Vec<SimpleWalOp> = vec![
            SimpleWalOp::Put {
                key: "a".to_string(),
                data: vec![1, 2, 3],
            },
            SimpleWalOp::Delete {
                key: "b".to_string(),
            },
            SimpleWalOp::Put {
                key: "c".to_string(),
                data: vec![],
            },
        ];
        let mut buf = Vec::new();
        for op in &ops {
            buf.extend_from_slice(&WalSerializer::serialize(op));
        }
        let decoded = WalSerializer::deserialize(&buf);
        assert_eq!(decoded, ops);
    }

    #[test]
    fn test_wal_deserialize_empty_data() {
        let ops = WalSerializer::deserialize(&[]);
        assert!(ops.is_empty());
    }

    #[test]
    fn test_wal_put_wire_format() {
        // Verify exact byte layout
        let op = SimpleWalOp::Put {
            key: "k".to_string(),
            data: vec![0xAB],
        };
        let bytes = WalSerializer::serialize(&op);
        // [op=0][key_len=0,1][b'k'][data_len=0,0,0,1][0xAB]
        assert_eq!(bytes, vec![0, 0, 1, b'k', 0, 0, 0, 1, 0xAB]);
    }

    // --- SimpleDedupStore ---

    #[test]
    fn test_dedup_put_returns_hash() {
        let mut store = SimpleDedupStore::new();
        let hash = store.put("file.mp4", b"data".to_vec());
        assert_ne!(hash, [0u8; 32]);
    }

    #[test]
    fn test_dedup_get_by_hash() {
        let mut store = SimpleDedupStore::new();
        let hash = store.put("file.mp4", b"hello".to_vec());
        let retrieved = store.get_by_hash(&hash);
        assert_eq!(retrieved, Some(b"hello".as_slice()));
    }

    #[test]
    fn test_dedup_get_by_key() {
        let mut store = SimpleDedupStore::new();
        store.put("video.mp4", b"content".to_vec());
        assert_eq!(store.get_by_key("video.mp4"), Some(b"content".as_slice()));
    }

    #[test]
    fn test_dedup_deduplication() {
        let mut store = SimpleDedupStore::new();
        let h1 = store.put("a.txt", b"same".to_vec());
        let h2 = store.put("b.txt", b"same".to_vec());
        assert_eq!(h1, h2, "identical content should have the same hash");
        assert_eq!(store.blob_count(), 1, "only one blob should be stored");
        assert_eq!(store.key_count(), 2);
    }

    #[test]
    fn test_dedup_hash_of() {
        let mut store = SimpleDedupStore::new();
        let hash = store.put("k", b"v".to_vec());
        assert_eq!(store.hash_of("k"), Some(hash));
        assert_eq!(store.hash_of("missing"), None);
    }

    // --- AccessLogAnalyzer ---

    #[test]
    fn test_access_log_top_keys() {
        let mut log = AccessLogAnalyzer::new();
        log.record("a");
        log.record("b");
        log.record("b");
        log.record("c");
        log.record("c");
        log.record("c");
        let top = log.top_keys(2);
        assert_eq!(top[0].0, "c");
        assert_eq!(top[0].1, 3);
        assert_eq!(top[1].0, "b");
        assert_eq!(top[1].1, 2);
    }

    #[test]
    fn test_access_log_top_keys_fewer_than_n() {
        let mut log = AccessLogAnalyzer::new();
        log.record("x");
        let top = log.top_keys(10);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_access_log_total_accesses() {
        let mut log = AccessLogAnalyzer::new();
        log.record_n("a", 5);
        log.record_n("b", 3);
        assert_eq!(log.total_accesses(), 8);
    }

    #[test]
    fn test_access_log_empty() {
        let log = AccessLogAnalyzer::new();
        assert_eq!(log.top_keys(5).len(), 0);
        assert_eq!(log.total_accesses(), 0);
    }

    // --- StorageMetricsSnapshot ---

    #[test]
    fn test_metrics_snapshot_error_rate() {
        let snap = StorageMetricsSnapshot::new(100, 1_000_000, 50, 30, 10, 5);
        let rate = snap.error_rate();
        // total = 90, errors = 5 → 5/90
        assert!((rate - 5.0 / 90.0).abs() < 1e-9);
    }

    #[test]
    fn test_metrics_snapshot_no_ops() {
        let snap = StorageMetricsSnapshot::new(0, 0, 0, 0, 0, 0);
        assert_eq!(snap.error_rate(), 0.0);
    }

    // --- StorageNamespace ---

    #[test]
    fn test_namespace_prefix() {
        let ns = StorageNamespace::new("media/");
        assert_eq!(ns.prefix(), "media/");
    }

    #[test]
    fn test_namespace_put_get() {
        let mut ns = StorageNamespace::new("ns/");
        ns.put("video.mp4", b"bytes".to_vec());
        assert_eq!(ns.get("video.mp4"), Some(b"bytes".as_slice()));
    }

    #[test]
    fn test_namespace_delete() {
        let mut ns = StorageNamespace::new("ns/");
        ns.put("f", vec![1, 2, 3]);
        assert!(ns.delete("f"));
        assert!(ns.get("f").is_none());
        assert!(!ns.delete("f")); // second delete → false
    }

    #[test]
    fn test_namespace_list_keys() {
        let mut ns = StorageNamespace::new("prefix/");
        ns.put("b.mp4", vec![]);
        ns.put("a.mp4", vec![]);
        let keys = ns.list_keys();
        assert_eq!(keys, vec!["a.mp4", "b.mp4"]);
    }

    // --- SimpleVersionStore ---

    #[test]
    fn test_version_store_put_and_get() {
        let mut store = SimpleVersionStore::new();
        let v1 = store.put("video.mp4", b"v1".to_vec(), 1000);
        let v2 = store.put("video.mp4", b"v2".to_vec(), 2000);
        assert_eq!(v1, 1);
        assert_eq!(v2, 2);
        let obj = store.get_version("video.mp4", 1).expect("version 1");
        assert_eq!(obj.data, b"v1");
    }

    #[test]
    fn test_version_store_latest() {
        let mut store = SimpleVersionStore::new();
        store.put("file", b"a".to_vec(), 100);
        store.put("file", b"b".to_vec(), 200);
        let latest = store.latest("file").expect("latest");
        assert_eq!(latest.data, b"b");
        assert_eq!(latest.version, 2);
    }

    #[test]
    fn test_version_store_history() {
        let mut store = SimpleVersionStore::new();
        store.put("f", vec![1], 10);
        store.put("f", vec![2], 20);
        store.put("f", vec![3], 30);
        assert_eq!(store.history("f").len(), 3);
    }

    #[test]
    fn test_version_store_missing_key() {
        let store = SimpleVersionStore::new();
        assert!(store.latest("missing").is_none());
        assert_eq!(store.history("missing").len(), 0);
    }

    // --- SimpleRetentionManager ---

    #[test]
    fn test_retention_manager_purge_expired() {
        let mut mgr = SimpleRetentionManager::new(3600); // 1 hour
        mgr.track(RetainedObject::new("old.mp4", 1000));
        mgr.track(RetainedObject::new("new.mp4", 5000));
        let purged = mgr.purge_expired(5000); // now = 5000, threshold = 1400 → old has ts 1000
        assert_eq!(purged.len(), 1);
        assert_eq!(purged[0].key, "old.mp4");
        assert_eq!(mgr.tracked().len(), 1);
    }

    #[test]
    fn test_retention_manager_no_expired() {
        let mut mgr = SimpleRetentionManager::new(86400);
        mgr.track(RetainedObject::new("recent.mp4", 9000));
        let purged = mgr.purge_expired(10000); // age = 1000 < 86400
        assert!(purged.is_empty());
    }

    #[test]
    fn test_retention_manager_max_age() {
        let mgr = SimpleRetentionManager::new(7200);
        assert_eq!(mgr.max_age_secs(), 7200);
    }

    // --- CompressedStore ---

    #[test]
    fn test_compressed_store_put_get_roundtrip() {
        let mut store = CompressedStore::new();
        store.put("key", b"hello world".to_vec());
        let got = store.get("key").expect("should retrieve");
        assert_eq!(got, b"hello world");
    }

    #[test]
    fn test_compressed_store_len() {
        let mut store = CompressedStore::new();
        assert!(store.is_empty());
        store.put("a", vec![1, 2, 3]);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_compress_decompress_stub_identity() {
        let data = b"some binary data \x00\xFF\xAB";
        let compressed = CompressedStore::compress_lz4_stub(data);
        let decompressed = CompressedStore::decompress_lz4_stub(&compressed);
        assert_eq!(decompressed, data);
    }

    // --- IntegrityChecker ---

    #[test]
    fn test_integrity_checker_all_valid() {
        let mut store: HashMap<String, Vec<u8>> = HashMap::new();
        store.insert("file.mp4".to_string(), b"content".to_vec());

        let mut expected: HashMap<String, [u8; 32]> = HashMap::new();
        let mut hasher = Sha256::new();
        hasher.update(b"content");
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&hasher.finalize());
        expected.insert("file.mp4".to_string(), hash);

        let violations = IntegrityChecker::verify(&store, &expected);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_integrity_checker_detects_corruption() {
        let mut store: HashMap<String, Vec<u8>> = HashMap::new();
        store.insert("file.mp4".to_string(), b"corrupted".to_vec());

        let mut expected: HashMap<String, [u8; 32]> = HashMap::new();
        let mut hasher = Sha256::new();
        hasher.update(b"original");
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&hasher.finalize());
        expected.insert("file.mp4".to_string(), hash);

        let violations = IntegrityChecker::verify(&store, &expected);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].key, "file.mp4");
    }

    #[test]
    fn test_integrity_checker_missing_key() {
        let store: HashMap<String, Vec<u8>> = HashMap::new();
        let mut expected: HashMap<String, [u8; 32]> = HashMap::new();
        expected.insert("missing.mp4".to_string(), [0xAB; 32]);
        let violations = IntegrityChecker::verify(&store, &expected);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].key, "missing.mp4");
        assert_eq!(violations[0].actual, [0u8; 32]);
    }

    #[test]
    fn test_integrity_checker_empty() {
        let store: HashMap<String, Vec<u8>> = HashMap::new();
        let expected: HashMap<String, [u8; 32]> = HashMap::new();
        let violations = IntegrityChecker::verify(&store, &expected);
        assert!(violations.is_empty());
    }
}
