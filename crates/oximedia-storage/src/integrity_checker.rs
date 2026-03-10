#![allow(dead_code)]
//! Data integrity verification for stored objects.
//!
//! Provides checksum computation, verification passes, and corruption
//! detection for objects managed by the storage layer.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ChecksumAlgo
// ---------------------------------------------------------------------------

/// Supported checksum algorithms for integrity checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChecksumAlgo {
    /// CRC-32 (fast, 32-bit).
    Crc32,
    /// Simple Adler-32 checksum.
    Adler32,
    /// 64-bit FNV-1a hash.
    Fnv1a64,
}

impl std::fmt::Display for ChecksumAlgo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Crc32 => write!(f, "crc32"),
            Self::Adler32 => write!(f, "adler32"),
            Self::Fnv1a64 => write!(f, "fnv1a64"),
        }
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust checksum implementations
// ---------------------------------------------------------------------------

/// Compute CRC-32 (ISO 3309).
fn compute_crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// Compute Adler-32.
fn compute_adler32(data: &[u8]) -> u32 {
    const MOD_ADLER: u32 = 65521;
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + u32::from(byte)) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;
    }
    (b << 16) | a
}

/// Compute FNV-1a 64-bit hash.
fn compute_fnv1a64(data: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Compute a hex-encoded checksum for any supported algorithm.
pub fn compute_checksum(algo: ChecksumAlgo, data: &[u8]) -> String {
    match algo {
        ChecksumAlgo::Crc32 => format!("{:08x}", compute_crc32(data)),
        ChecksumAlgo::Adler32 => format!("{:08x}", compute_adler32(data)),
        ChecksumAlgo::Fnv1a64 => format!("{:016x}", compute_fnv1a64(data)),
    }
}

// ---------------------------------------------------------------------------
// IntegrityRecord
// ---------------------------------------------------------------------------

/// Stored integrity record for a single object.
#[derive(Debug, Clone)]
pub struct IntegrityRecord {
    /// Object key / path.
    pub key: String,
    /// Algorithm used.
    pub algorithm: ChecksumAlgo,
    /// Expected checksum (hex).
    pub expected: String,
    /// Object size in bytes at the time the checksum was computed.
    pub size_bytes: u64,
    /// Epoch timestamp when the record was created.
    pub created_epoch: u64,
}

impl IntegrityRecord {
    /// Create a new integrity record.
    pub fn new(
        key: impl Into<String>,
        algorithm: ChecksumAlgo,
        expected: impl Into<String>,
        size_bytes: u64,
        created_epoch: u64,
    ) -> Self {
        Self {
            key: key.into(),
            algorithm,
            expected: expected.into(),
            size_bytes,
            created_epoch,
        }
    }

    /// Verify the record against actual data.
    pub fn verify(&self, data: &[u8]) -> VerifyResult {
        let actual = compute_checksum(self.algorithm, data);
        #[allow(clippy::cast_precision_loss)]
        let size_match = self.size_bytes == data.len() as u64;
        if actual == self.expected && size_match {
            VerifyResult::Ok
        } else if !size_match {
            VerifyResult::SizeMismatch {
                expected: self.size_bytes,
                actual: data.len() as u64,
            }
        } else {
            VerifyResult::ChecksumMismatch {
                expected: self.expected.clone(),
                actual,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// VerifyResult
// ---------------------------------------------------------------------------

/// Result of an integrity verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyResult {
    /// Data is intact.
    Ok,
    /// Checksum does not match.
    ChecksumMismatch {
        /// Expected checksum.
        expected: String,
        /// Actual checksum.
        actual: String,
    },
    /// Size does not match.
    SizeMismatch {
        /// Expected size.
        expected: u64,
        /// Actual size.
        actual: u64,
    },
}

impl VerifyResult {
    /// Whether the verification passed.
    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Ok)
    }
}

// ---------------------------------------------------------------------------
// IntegrityStore
// ---------------------------------------------------------------------------

/// In-memory store of integrity records, keyed by object key.
#[derive(Debug, Default)]
pub struct IntegrityStore {
    /// Records indexed by key.
    records: HashMap<String, IntegrityRecord>,
}

impl IntegrityStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or update an integrity record.
    pub fn insert(&mut self, record: IntegrityRecord) {
        self.records.insert(record.key.clone(), record);
    }

    /// Get the record for a key.
    pub fn get(&self, key: &str) -> Option<&IntegrityRecord> {
        self.records.get(key)
    }

    /// Remove the record for a key.
    pub fn remove(&mut self, key: &str) -> Option<IntegrityRecord> {
        self.records.remove(key)
    }

    /// Number of records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Verify data for a given key.
    pub fn verify(&self, key: &str, data: &[u8]) -> Option<VerifyResult> {
        self.records.get(key).map(|rec| rec.verify(data))
    }

    /// Run integrity checks on all records given a data provider closure.
    pub fn verify_all<F>(&self, data_provider: F) -> Vec<(String, VerifyResult)>
    where
        F: Fn(&str) -> Option<Vec<u8>>,
    {
        let mut results = Vec::new();
        for (key, record) in &self.records {
            if let Some(data) = data_provider(key) {
                results.push((key.clone(), record.verify(&data)));
            }
        }
        results
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checksum_algo_display() {
        assert_eq!(ChecksumAlgo::Crc32.to_string(), "crc32");
        assert_eq!(ChecksumAlgo::Adler32.to_string(), "adler32");
        assert_eq!(ChecksumAlgo::Fnv1a64.to_string(), "fnv1a64");
    }

    #[test]
    fn test_crc32_deterministic() {
        let a = compute_checksum(ChecksumAlgo::Crc32, b"hello");
        let b = compute_checksum(ChecksumAlgo::Crc32, b"hello");
        assert_eq!(a, b);
    }

    #[test]
    fn test_crc32_differs() {
        let a = compute_checksum(ChecksumAlgo::Crc32, b"hello");
        let b = compute_checksum(ChecksumAlgo::Crc32, b"world");
        assert_ne!(a, b);
    }

    #[test]
    fn test_adler32_deterministic() {
        let a = compute_checksum(ChecksumAlgo::Adler32, b"test");
        let b = compute_checksum(ChecksumAlgo::Adler32, b"test");
        assert_eq!(a, b);
    }

    #[test]
    fn test_adler32_known_empty() {
        // Adler-32 of empty input = 0x00000001
        let h = compute_checksum(ChecksumAlgo::Adler32, b"");
        assert_eq!(h, "00000001");
    }

    #[test]
    fn test_fnv1a64_length() {
        let h = compute_checksum(ChecksumAlgo::Fnv1a64, b"data");
        assert_eq!(h.len(), 16);
    }

    #[test]
    fn test_integrity_record_verify_ok() {
        let data = b"hello world";
        let expected = compute_checksum(ChecksumAlgo::Crc32, data);
        let rec = IntegrityRecord::new(
            "obj/1",
            ChecksumAlgo::Crc32,
            &expected,
            data.len() as u64,
            1000,
        );
        assert_eq!(rec.verify(data), VerifyResult::Ok);
    }

    #[test]
    fn test_integrity_record_verify_checksum_mismatch() {
        let data = b"hello world";
        let rec = IntegrityRecord::new(
            "obj/1",
            ChecksumAlgo::Crc32,
            "00000000",
            data.len() as u64,
            1000,
        );
        match rec.verify(data) {
            VerifyResult::ChecksumMismatch { expected, actual } => {
                assert_eq!(expected, "00000000");
                assert_ne!(actual, "00000000");
            }
            other => panic!("expected ChecksumMismatch, got {:?}", other),
        }
    }

    #[test]
    fn test_integrity_record_verify_size_mismatch() {
        let data = b"hello";
        let expected = compute_checksum(ChecksumAlgo::Crc32, data);
        let rec = IntegrityRecord::new("obj/1", ChecksumAlgo::Crc32, &expected, 999, 1000);
        match rec.verify(data) {
            VerifyResult::SizeMismatch {
                expected: e,
                actual: a,
            } => {
                assert_eq!(e, 999);
                assert_eq!(a, 5);
            }
            other => panic!("expected SizeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn test_integrity_store_insert_get() {
        let mut store = IntegrityStore::new();
        assert!(store.is_empty());
        let rec = IntegrityRecord::new("k", ChecksumAlgo::Crc32, "aabb", 10, 0);
        store.insert(rec);
        assert_eq!(store.len(), 1);
        assert!(store.get("k").is_some());
        assert!(store.get("missing").is_none());
    }

    #[test]
    fn test_integrity_store_remove() {
        let mut store = IntegrityStore::new();
        store.insert(IntegrityRecord::new("k", ChecksumAlgo::Crc32, "aa", 1, 0));
        assert!(store.remove("k").is_some());
        assert!(store.is_empty());
    }

    #[test]
    fn test_integrity_store_verify() {
        let mut store = IntegrityStore::new();
        let data = b"payload";
        let cs = compute_checksum(ChecksumAlgo::Adler32, data);
        store.insert(IntegrityRecord::new(
            "obj",
            ChecksumAlgo::Adler32,
            &cs,
            data.len() as u64,
            0,
        ));
        let result = store.verify("obj", data).expect("verify should succeed");
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_result_is_ok() {
        assert!(VerifyResult::Ok.is_ok());
        assert!(!VerifyResult::ChecksumMismatch {
            expected: String::new(),
            actual: String::new(),
        }
        .is_ok());
    }
}
