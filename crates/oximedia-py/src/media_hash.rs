#![allow(dead_code)]
//! Media content hashing and fingerprinting for Python bindings.
//!
//! Provides hash algorithms, perceptual fingerprinting stubs, and
//! composite hash generation for identifying media assets across
//! the Python/Rust boundary.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// HashAlgorithm
// ---------------------------------------------------------------------------

/// Supported hash algorithm identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HashAlgorithm {
    /// CRC-32 (fast, 32-bit).
    Crc32,
    /// Simple 64-bit FNV-1a hash.
    Fnv1a64,
    /// Simple 128-bit pair constructed from two FNV-style rounds.
    Simple128,
}

impl fmt::Display for HashAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Crc32 => write!(f, "crc32"),
            Self::Fnv1a64 => write!(f, "fnv1a64"),
            Self::Simple128 => write!(f, "simple128"),
        }
    }
}

// ---------------------------------------------------------------------------
// HashValue
// ---------------------------------------------------------------------------

/// A computed hash value stored as a hex string.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HashValue {
    /// The algorithm used to compute this hash.
    pub algorithm: HashAlgorithm,
    /// Hex-encoded digest.
    pub hex_digest: String,
}

impl HashValue {
    /// Create a new hash value.
    pub fn new(algorithm: HashAlgorithm, hex_digest: impl Into<String>) -> Self {
        Self {
            algorithm,
            hex_digest: hex_digest.into(),
        }
    }
}

impl fmt::Display for HashValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.algorithm, self.hex_digest)
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust hash helpers (no external deps)
// ---------------------------------------------------------------------------

/// Compute CRC-32 (ISO 3309 / ITU-T V.42) of `data`.
fn crc32(data: &[u8]) -> u32 {
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

/// Compute FNV-1a 64-bit hash of `data`.
fn fnv1a_64(data: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Compute a simple 128-bit hash as (fnv1a(data), fnv1a(data reversed)).
fn simple_128(data: &[u8]) -> (u64, u64) {
    let h1 = fnv1a_64(data);
    let reversed: Vec<u8> = data.iter().rev().copied().collect();
    let h2 = fnv1a_64(&reversed);
    (h1, h2)
}

// ---------------------------------------------------------------------------
// hash_bytes
// ---------------------------------------------------------------------------

/// Hash a byte slice with the requested algorithm.
pub fn hash_bytes(algorithm: HashAlgorithm, data: &[u8]) -> HashValue {
    let hex_digest = match algorithm {
        HashAlgorithm::Crc32 => format!("{:08x}", crc32(data)),
        HashAlgorithm::Fnv1a64 => format!("{:016x}", fnv1a_64(data)),
        HashAlgorithm::Simple128 => {
            let (h1, h2) = simple_128(data);
            format!("{:016x}{:016x}", h1, h2)
        }
    };
    HashValue::new(algorithm, hex_digest)
}

// ---------------------------------------------------------------------------
// CompositeHash
// ---------------------------------------------------------------------------

/// A collection of hashes computed by multiple algorithms for one blob.
#[derive(Debug, Clone)]
pub struct CompositeHash {
    /// Map from algorithm to hash value.
    pub hashes: HashMap<HashAlgorithm, HashValue>,
}

impl CompositeHash {
    /// Compute hashes using all given algorithms.
    pub fn compute(data: &[u8], algorithms: &[HashAlgorithm]) -> Self {
        let mut hashes = HashMap::new();
        for &algo in algorithms {
            hashes.insert(algo, hash_bytes(algo, data));
        }
        Self { hashes }
    }

    /// Get the hash for a specific algorithm, if computed.
    pub fn get(&self, algorithm: HashAlgorithm) -> Option<&HashValue> {
        self.hashes.get(&algorithm)
    }

    /// Number of hashes in the composite.
    pub fn len(&self) -> usize {
        self.hashes.len()
    }

    /// Whether the composite has no hashes.
    pub fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }
}

// ---------------------------------------------------------------------------
// PerceptualFingerprint
// ---------------------------------------------------------------------------

/// A perceptual fingerprint represented as a fixed-size bit vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerceptualFingerprint {
    /// Raw fingerprint bits stored as bytes.
    pub bits: Vec<u8>,
    /// Logical bit length.
    pub bit_length: usize,
}

impl PerceptualFingerprint {
    /// Create a fingerprint from raw bytes and a logical bit length.
    pub fn new(bits: Vec<u8>, bit_length: usize) -> Self {
        Self { bits, bit_length }
    }

    /// Compute the Hamming distance between two fingerprints of equal length.
    pub fn hamming_distance(&self, other: &Self) -> Option<u32> {
        if self.bit_length != other.bit_length || self.bits.len() != other.bits.len() {
            return None;
        }
        let dist: u32 = self
            .bits
            .iter()
            .zip(other.bits.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum();
        Some(dist)
    }

    /// Similarity in [0.0, 1.0] based on Hamming distance.
    #[allow(clippy::cast_precision_loss)]
    pub fn similarity(&self, other: &Self) -> Option<f64> {
        self.hamming_distance(other).map(|d| {
            if self.bit_length == 0 {
                1.0
            } else {
                1.0 - (d as f64 / self.bit_length as f64)
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_algorithm_display() {
        assert_eq!(HashAlgorithm::Crc32.to_string(), "crc32");
        assert_eq!(HashAlgorithm::Fnv1a64.to_string(), "fnv1a64");
        assert_eq!(HashAlgorithm::Simple128.to_string(), "simple128");
    }

    #[test]
    fn test_hash_value_display() {
        let hv = HashValue::new(HashAlgorithm::Crc32, "deadbeef");
        assert_eq!(hv.to_string(), "crc32:deadbeef");
    }

    #[test]
    fn test_crc32_empty() {
        let h = hash_bytes(HashAlgorithm::Crc32, b"");
        assert_eq!(h.hex_digest.len(), 8);
    }

    #[test]
    fn test_crc32_deterministic() {
        let h1 = hash_bytes(HashAlgorithm::Crc32, b"hello world");
        let h2 = hash_bytes(HashAlgorithm::Crc32, b"hello world");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_crc32_differs() {
        let h1 = hash_bytes(HashAlgorithm::Crc32, b"hello");
        let h2 = hash_bytes(HashAlgorithm::Crc32, b"world");
        assert_ne!(h1.hex_digest, h2.hex_digest);
    }

    #[test]
    fn test_fnv1a64_deterministic() {
        let h1 = hash_bytes(HashAlgorithm::Fnv1a64, b"test data");
        let h2 = hash_bytes(HashAlgorithm::Fnv1a64, b"test data");
        assert_eq!(h1, h2);
        assert_eq!(h1.hex_digest.len(), 16);
    }

    #[test]
    fn test_simple128_length() {
        let h = hash_bytes(HashAlgorithm::Simple128, b"payload");
        assert_eq!(h.hex_digest.len(), 32);
    }

    #[test]
    fn test_composite_hash() {
        let ch = CompositeHash::compute(b"data", &[HashAlgorithm::Crc32, HashAlgorithm::Fnv1a64]);
        assert_eq!(ch.len(), 2);
        assert!(ch.get(HashAlgorithm::Crc32).is_some());
        assert!(ch.get(HashAlgorithm::Fnv1a64).is_some());
        assert!(ch.get(HashAlgorithm::Simple128).is_none());
    }

    #[test]
    fn test_composite_hash_empty() {
        let ch = CompositeHash::compute(b"data", &[]);
        assert!(ch.is_empty());
    }

    #[test]
    fn test_fingerprint_hamming_same() {
        let fp = PerceptualFingerprint::new(vec![0xFF, 0x00], 16);
        assert_eq!(fp.hamming_distance(&fp), Some(0));
    }

    #[test]
    fn test_fingerprint_hamming_different() {
        let a = PerceptualFingerprint::new(vec![0xFF], 8);
        let b = PerceptualFingerprint::new(vec![0x00], 8);
        assert_eq!(a.hamming_distance(&b), Some(8));
    }

    #[test]
    fn test_fingerprint_hamming_mismatch() {
        let a = PerceptualFingerprint::new(vec![0xFF], 8);
        let b = PerceptualFingerprint::new(vec![0xFF, 0x00], 16);
        assert!(a.hamming_distance(&b).is_none());
    }

    #[test]
    fn test_fingerprint_similarity() {
        let a = PerceptualFingerprint::new(vec![0xFF], 8);
        let b = PerceptualFingerprint::new(vec![0xFF], 8);
        let sim = a.similarity(&b).expect("sim should be valid");
        assert!((sim - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fingerprint_similarity_zero() {
        let a = PerceptualFingerprint::new(vec![0xFF], 8);
        let b = PerceptualFingerprint::new(vec![0x00], 8);
        let sim = a.similarity(&b).expect("sim should be valid");
        assert!(sim.abs() < f64::EPSILON);
    }
}
