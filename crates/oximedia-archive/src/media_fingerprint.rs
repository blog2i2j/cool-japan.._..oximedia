#![allow(dead_code)]
//! Perceptual fingerprinting for archive deduplication and identification.
//!
//! Generates compact perceptual fingerprints from media content that can be used
//! for duplicate detection, content identification, and similarity matching
//! across large archive collections.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

/// Algorithm used for fingerprint generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FingerprintAlgorithm {
    /// Average hash - simple, fast, less robust to transformations.
    AverageHash,
    /// Difference hash - captures gradient information.
    DifferenceHash,
    /// Perceptual hash using DCT-based frequency analysis.
    PerceptualHash,
    /// Block mean value hash for robust matching.
    BlockMeanHash,
    /// Audio chromaprint-style fingerprinting.
    AudioChromaprint,
}

impl fmt::Display for FingerprintAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AverageHash => write!(f, "AverageHash"),
            Self::DifferenceHash => write!(f, "DifferenceHash"),
            Self::PerceptualHash => write!(f, "PerceptualHash"),
            Self::BlockMeanHash => write!(f, "BlockMeanHash"),
            Self::AudioChromaprint => write!(f, "AudioChromaprint"),
        }
    }
}

/// A media fingerprint consisting of a fixed-size hash.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Fingerprint {
    /// The algorithm used to generate this fingerprint.
    pub algorithm: FingerprintAlgorithm,
    /// The raw hash bits stored as bytes.
    pub hash_bytes: Vec<u8>,
    /// Bit length of the hash.
    pub bit_length: usize,
}

impl Fingerprint {
    /// Creates a new fingerprint from raw bytes.
    pub fn new(algorithm: FingerprintAlgorithm, hash_bytes: Vec<u8>, bit_length: usize) -> Self {
        Self {
            algorithm,
            hash_bytes,
            bit_length,
        }
    }

    /// Creates a fingerprint from a hex string.
    pub fn from_hex(algorithm: FingerprintAlgorithm, hex_str: &str) -> Option<Self> {
        let bytes: Result<Vec<u8>, _> = (0..hex_str.len())
            .step_by(2)
            .map(|i| {
                let end = (i + 2).min(hex_str.len());
                u8::from_str_radix(&hex_str[i..end], 16)
            })
            .collect();
        bytes.ok().map(|b| {
            let bit_length = b.len() * 8;
            Self::new(algorithm, b, bit_length)
        })
    }

    /// Returns the fingerprint as a hex string.
    pub fn to_hex(&self) -> String {
        self.hash_bytes
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect()
    }

    /// Computes the Hamming distance between two fingerprints.
    ///
    /// Returns `None` if the fingerprints have different bit lengths.
    pub fn hamming_distance(&self, other: &Fingerprint) -> Option<u32> {
        if self.bit_length != other.bit_length || self.hash_bytes.len() != other.hash_bytes.len() {
            return None;
        }
        let distance: u32 = self
            .hash_bytes
            .iter()
            .zip(other.hash_bytes.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum();
        Some(distance)
    }

    /// Computes the similarity score (0.0 to 1.0) between two fingerprints.
    #[allow(clippy::cast_precision_loss)]
    pub fn similarity(&self, other: &Fingerprint) -> Option<f64> {
        self.hamming_distance(other)
            .map(|d| 1.0 - (d as f64 / self.bit_length as f64))
    }

    /// Returns true if two fingerprints are considered a match at the given threshold.
    pub fn matches(&self, other: &Fingerprint, max_distance: u32) -> bool {
        self.hamming_distance(other)
            .map(|d| d <= max_distance)
            .unwrap_or(false)
    }
}

impl fmt::Display for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.algorithm, self.to_hex())
    }
}

/// A fingerprint record associated with a specific media asset.
#[derive(Debug, Clone)]
pub struct FingerprintRecord {
    /// Path to the media asset.
    pub asset_path: PathBuf,
    /// Generated fingerprints (may have multiple algorithms).
    pub fingerprints: Vec<Fingerprint>,
    /// Optional content identifier.
    pub content_id: Option<String>,
    /// File size in bytes for quick pre-filtering.
    pub file_size: u64,
}

impl FingerprintRecord {
    /// Creates a new fingerprint record.
    pub fn new(asset_path: PathBuf, file_size: u64) -> Self {
        Self {
            asset_path,
            fingerprints: Vec::new(),
            content_id: None,
            file_size,
        }
    }

    /// Adds a fingerprint to this record.
    pub fn add_fingerprint(&mut self, fp: Fingerprint) {
        self.fingerprints.push(fp);
    }

    /// Sets the content identifier.
    pub fn with_content_id(mut self, id: &str) -> Self {
        self.content_id = Some(id.to_string());
        self
    }

    /// Returns fingerprints for a given algorithm.
    pub fn get_fingerprint(&self, algorithm: FingerprintAlgorithm) -> Option<&Fingerprint> {
        self.fingerprints
            .iter()
            .find(|fp| fp.algorithm == algorithm)
    }
}

/// Result of a duplicate search.
#[derive(Debug, Clone)]
pub struct DuplicateMatch {
    /// Path to the matching asset.
    pub match_path: PathBuf,
    /// Algorithm used for matching.
    pub algorithm: FingerprintAlgorithm,
    /// Hamming distance between the fingerprints.
    pub distance: u32,
    /// Similarity score (0.0 to 1.0).
    pub similarity: f64,
}

impl fmt::Display for DuplicateMatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (distance={}, similarity={:.3}, algo={})",
            self.match_path.display(),
            self.distance,
            self.similarity,
            self.algorithm
        )
    }
}

/// Fingerprint index for fast duplicate detection across an archive.
#[derive(Debug)]
pub struct FingerprintIndex {
    /// All records indexed by path.
    records: HashMap<String, FingerprintRecord>,
    /// Default matching threshold (max Hamming distance).
    threshold: u32,
    /// Default algorithm for searches.
    default_algorithm: FingerprintAlgorithm,
}

impl FingerprintIndex {
    /// Creates a new fingerprint index.
    pub fn new(default_algorithm: FingerprintAlgorithm, threshold: u32) -> Self {
        Self {
            records: HashMap::new(),
            threshold,
            default_algorithm,
        }
    }

    /// Inserts a fingerprint record into the index.
    pub fn insert(&mut self, record: FingerprintRecord) {
        let key = record.asset_path.to_string_lossy().to_string();
        self.records.insert(key, record);
    }

    /// Returns the number of indexed records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns true if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Returns the configured threshold.
    pub fn threshold(&self) -> u32 {
        self.threshold
    }

    /// Sets a new threshold.
    pub fn set_threshold(&mut self, threshold: u32) {
        self.threshold = threshold;
    }

    /// Searches for duplicates of the given fingerprint.
    #[allow(clippy::cast_precision_loss)]
    pub fn find_duplicates(&self, query: &Fingerprint) -> Vec<DuplicateMatch> {
        let mut matches = Vec::new();

        for (_, record) in &self.records {
            if let Some(fp) = record.get_fingerprint(query.algorithm) {
                if let Some(distance) = query.hamming_distance(fp) {
                    if distance <= self.threshold {
                        let similarity = 1.0 - (distance as f64 / query.bit_length as f64);
                        matches.push(DuplicateMatch {
                            match_path: record.asset_path.clone(),
                            algorithm: query.algorithm,
                            distance,
                            similarity,
                        });
                    }
                }
            }
        }

        matches.sort_by(|a, b| a.distance.cmp(&b.distance));
        matches
    }

    /// Finds all duplicate groups in the index.
    #[allow(clippy::cast_precision_loss)]
    pub fn find_all_duplicate_groups(&self) -> Vec<Vec<String>> {
        let mut groups: Vec<Vec<String>> = Vec::new();
        let mut visited: HashMap<String, bool> = HashMap::new();

        let keys: Vec<String> = self.records.keys().cloned().collect();

        for key in &keys {
            if visited.contains_key(key) {
                continue;
            }
            let record = &self.records[key];
            if let Some(fp) = record.get_fingerprint(self.default_algorithm) {
                let mut group = vec![key.clone()];
                visited.insert(key.clone(), true);

                for other_key in &keys {
                    if visited.contains_key(other_key) {
                        continue;
                    }
                    let other_record = &self.records[other_key];
                    if let Some(other_fp) = other_record.get_fingerprint(self.default_algorithm) {
                        if fp.matches(other_fp, self.threshold) {
                            group.push(other_key.clone());
                            visited.insert(other_key.clone(), true);
                        }
                    }
                }

                if group.len() > 1 {
                    groups.push(group);
                }
            }
        }

        groups
    }

    /// Returns a record by path.
    pub fn get(&self, path: &str) -> Option<&FingerprintRecord> {
        self.records.get(path)
    }

    /// Removes a record by path.
    pub fn remove(&mut self, path: &str) -> Option<FingerprintRecord> {
        self.records.remove(path)
    }
}

/// Simulates generating an average hash fingerprint from pixel data.
///
/// In production, this would accept actual pixel data; here we use a
/// deterministic simulation based on the provided bytes.
pub fn generate_average_hash(data: &[u8], hash_size: usize) -> Fingerprint {
    let total_bits = hash_size * hash_size;
    let byte_count = (total_bits + 7) / 8;
    let mut hash_bytes = vec![0u8; byte_count];

    if data.is_empty() {
        return Fingerprint::new(FingerprintAlgorithm::AverageHash, hash_bytes, total_bits);
    }

    // Compute average of all bytes
    #[allow(clippy::cast_precision_loss)]
    let avg: f64 = data.iter().map(|&b| b as f64).sum::<f64>() / data.len() as f64;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let avg_byte = avg as u8;

    // Each bit is set if corresponding data sample exceeds average
    for i in 0..total_bits {
        let sample_idx = i % data.len();
        if data[sample_idx] > avg_byte {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            hash_bytes[byte_idx] |= 1 << bit_idx;
        }
    }

    Fingerprint::new(FingerprintAlgorithm::AverageHash, hash_bytes, total_bits)
}

/// Simulates generating a difference hash fingerprint from pixel data.
pub fn generate_difference_hash(data: &[u8], hash_size: usize) -> Fingerprint {
    let total_bits = hash_size * hash_size;
    let byte_count = (total_bits + 7) / 8;
    let mut hash_bytes = vec![0u8; byte_count];

    if data.len() < 2 {
        return Fingerprint::new(FingerprintAlgorithm::DifferenceHash, hash_bytes, total_bits);
    }

    // Each bit is set if the next sample is greater than the current
    for i in 0..total_bits {
        let idx_a = i % data.len();
        let idx_b = (i + 1) % data.len();
        if data[idx_b] > data[idx_a] {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            hash_bytes[byte_idx] |= 1 << bit_idx;
        }
    }

    Fingerprint::new(FingerprintAlgorithm::DifferenceHash, hash_bytes, total_bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_algorithm_display() {
        assert_eq!(FingerprintAlgorithm::AverageHash.to_string(), "AverageHash");
        assert_eq!(
            FingerprintAlgorithm::DifferenceHash.to_string(),
            "DifferenceHash"
        );
        assert_eq!(
            FingerprintAlgorithm::PerceptualHash.to_string(),
            "PerceptualHash"
        );
        assert_eq!(
            FingerprintAlgorithm::BlockMeanHash.to_string(),
            "BlockMeanHash"
        );
        assert_eq!(
            FingerprintAlgorithm::AudioChromaprint.to_string(),
            "AudioChromaprint"
        );
    }

    #[test]
    fn test_fingerprint_from_hex() {
        let fp = Fingerprint::from_hex(FingerprintAlgorithm::AverageHash, "ff00ab")
            .expect("fp should be valid");
        assert_eq!(fp.hash_bytes, vec![0xff, 0x00, 0xab]);
        assert_eq!(fp.bit_length, 24);
    }

    #[test]
    fn test_fingerprint_to_hex() {
        let fp = Fingerprint::new(
            FingerprintAlgorithm::AverageHash,
            vec![0xde, 0xad, 0xbe, 0xef],
            32,
        );
        assert_eq!(fp.to_hex(), "deadbeef");
    }

    #[test]
    fn test_hamming_distance_identical() {
        let fp1 = Fingerprint::new(FingerprintAlgorithm::AverageHash, vec![0xff, 0x00], 16);
        let fp2 = Fingerprint::new(FingerprintAlgorithm::AverageHash, vec![0xff, 0x00], 16);
        assert_eq!(fp1.hamming_distance(&fp2), Some(0));
    }

    #[test]
    fn test_hamming_distance_different() {
        let fp1 = Fingerprint::new(FingerprintAlgorithm::AverageHash, vec![0xff], 8);
        let fp2 = Fingerprint::new(FingerprintAlgorithm::AverageHash, vec![0x00], 8);
        assert_eq!(fp1.hamming_distance(&fp2), Some(8));
    }

    #[test]
    fn test_hamming_distance_incompatible() {
        let fp1 = Fingerprint::new(FingerprintAlgorithm::AverageHash, vec![0xff], 8);
        let fp2 = Fingerprint::new(FingerprintAlgorithm::AverageHash, vec![0xff, 0x00], 16);
        assert_eq!(fp1.hamming_distance(&fp2), None);
    }

    #[test]
    fn test_similarity_identical() {
        let fp1 = Fingerprint::new(FingerprintAlgorithm::AverageHash, vec![0xab, 0xcd], 16);
        let fp2 = fp1.clone();
        let sim = fp1.similarity(&fp2).expect("sim should be valid");
        assert!((sim - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_similarity_opposite() {
        let fp1 = Fingerprint::new(FingerprintAlgorithm::AverageHash, vec![0xff], 8);
        let fp2 = Fingerprint::new(FingerprintAlgorithm::AverageHash, vec![0x00], 8);
        let sim = fp1.similarity(&fp2).expect("sim should be valid");
        assert!(sim.abs() < f64::EPSILON);
    }

    #[test]
    fn test_matches() {
        let fp1 = Fingerprint::new(FingerprintAlgorithm::AverageHash, vec![0xff], 8);
        let fp2 = Fingerprint::new(FingerprintAlgorithm::AverageHash, vec![0xfe], 8);
        assert!(fp1.matches(&fp2, 1));
        assert!(!fp1.matches(&fp2, 0));
    }

    #[test]
    fn test_fingerprint_display() {
        let fp = Fingerprint::new(FingerprintAlgorithm::AverageHash, vec![0xab, 0xcd], 16);
        let display = format!("{}", fp);
        assert!(display.contains("AverageHash"));
        assert!(display.contains("abcd"));
    }

    #[test]
    fn test_fingerprint_record() {
        let mut record = FingerprintRecord::new(PathBuf::from("/test.mxf"), 1024);
        let fp = Fingerprint::new(FingerprintAlgorithm::AverageHash, vec![0xaa], 8);
        record.add_fingerprint(fp);
        assert!(record
            .get_fingerprint(FingerprintAlgorithm::AverageHash)
            .is_some());
        assert!(record
            .get_fingerprint(FingerprintAlgorithm::DifferenceHash)
            .is_none());
    }

    #[test]
    fn test_fingerprint_index_insert_and_search() {
        let mut index = FingerprintIndex::new(FingerprintAlgorithm::AverageHash, 2);
        let mut record = FingerprintRecord::new(PathBuf::from("/a.mxf"), 100);
        record.add_fingerprint(Fingerprint::new(
            FingerprintAlgorithm::AverageHash,
            vec![0xff],
            8,
        ));
        index.insert(record);

        let query = Fingerprint::new(FingerprintAlgorithm::AverageHash, vec![0xfe], 8);
        let results = index.find_duplicates(&query);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].distance, 1);
    }

    #[test]
    fn test_fingerprint_index_no_match() {
        let mut index = FingerprintIndex::new(FingerprintAlgorithm::AverageHash, 1);
        let mut record = FingerprintRecord::new(PathBuf::from("/a.mxf"), 100);
        record.add_fingerprint(Fingerprint::new(
            FingerprintAlgorithm::AverageHash,
            vec![0xff],
            8,
        ));
        index.insert(record);

        let query = Fingerprint::new(FingerprintAlgorithm::AverageHash, vec![0x00], 8);
        let results = index.find_duplicates(&query);
        assert!(results.is_empty());
    }

    #[test]
    fn test_generate_average_hash() {
        let data = vec![100, 200, 50, 150, 80, 220, 30, 180];
        let fp = generate_average_hash(&data, 2);
        assert_eq!(fp.algorithm, FingerprintAlgorithm::AverageHash);
        assert_eq!(fp.bit_length, 4);
    }

    #[test]
    fn test_generate_difference_hash() {
        let data = vec![10, 20, 15, 25, 5, 30];
        let fp = generate_difference_hash(&data, 2);
        assert_eq!(fp.algorithm, FingerprintAlgorithm::DifferenceHash);
        assert_eq!(fp.bit_length, 4);
    }

    #[test]
    fn test_duplicate_match_display() {
        let dm = DuplicateMatch {
            match_path: PathBuf::from("/dup.mxf"),
            algorithm: FingerprintAlgorithm::AverageHash,
            distance: 3,
            similarity: 0.875,
        };
        let display = format!("{}", dm);
        assert!(display.contains("dup.mxf"));
        assert!(display.contains("distance=3"));
    }
}
