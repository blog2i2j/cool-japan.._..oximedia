//! Cryptographic and content-based hashing for deduplication.
//!
//! This module provides:
//! - BLAKE3 cryptographic hashing for exact duplicate detection
//! - Content-based chunking with rolling hash (Rabin fingerprinting)
//! - Chunk-level deduplication for efficient storage
//! - Hash indexing and comparison

use crate::{DedupError, DedupResult};
use blake3::Hasher;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Size of read buffer for hashing
const BUFFER_SIZE: usize = 65536; // 64 KB

/// Default chunk size for content-based chunking
const DEFAULT_CHUNK_SIZE: usize = 4096; // 4 KB

/// Minimum chunk size
const MIN_CHUNK_SIZE: usize = 1024; // 1 KB

/// Maximum chunk size
const MAX_CHUNK_SIZE: usize = 1048576; // 1 MB

/// Rabin polynomial for rolling hash
const RABIN_POLYNOMIAL: u64 = 0x3DA3_358B_4DC1_73E9;

/// File hash result containing BLAKE3 hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileHash {
    hash: [u8; 32],
}

impl FileHash {
    /// Create from byte array.
    #[must_use]
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self { hash: bytes }
    }

    /// Get hash as bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.hash
    }

    /// Convert to hex string.
    #[must_use]
    pub fn to_hex(&self) -> String {
        hex::encode(self.hash)
    }

    /// Create from hex string.
    ///
    /// # Errors
    ///
    /// Returns an error if the hex string is invalid.
    pub fn from_hex(s: &str) -> DedupResult<Self> {
        let bytes =
            hex::decode(s).map_err(|e| DedupError::Hash(format!("Invalid hex string: {e}")))?;
        if bytes.len() != 32 {
            return Err(DedupError::Hash(format!(
                "Invalid hash length: expected 32, got {}",
                bytes.len()
            )));
        }
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&bytes);
        Ok(Self::from_bytes(hash))
    }

    /// Compute Hamming distance between two hashes.
    #[must_use]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        self.hash
            .iter()
            .zip(other.hash.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }

    /// Compute similarity (0.0-1.0) based on Hamming distance.
    #[must_use]
    pub fn similarity(&self, other: &Self) -> f64 {
        let distance = self.hamming_distance(other);
        let max_distance = 256; // 32 bytes * 8 bits
        1.0 - (f64::from(distance) / f64::from(max_distance))
    }
}

/// Compute BLAKE3 hash of a file.
///
/// # Errors
///
/// Returns an error if the file cannot be read.
pub fn compute_file_hash(path: impl AsRef<Path>) -> DedupResult<FileHash> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(BUFFER_SIZE, file);
    let mut hasher = Hasher::new();
    let mut buffer = vec![0u8; BUFFER_SIZE];

    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }

    let hash = hasher.finalize();
    Ok(FileHash::from_bytes(*hash.as_bytes()))
}

/// Compute BLAKE3 hash of data.
#[must_use]
pub fn compute_data_hash(data: &[u8]) -> FileHash {
    let hash = blake3::hash(data);
    FileHash::from_bytes(*hash.as_bytes())
}

/// Content chunk with hash.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Offset in file
    pub offset: u64,

    /// Size of chunk
    pub size: usize,

    /// BLAKE3 hash of chunk
    pub hash: FileHash,
}

/// Rolling hash for content-based chunking.
pub struct RollingHash {
    window_size: usize,
    polynomial: u64,
    mod_mask: u64,
    hash: u64,
    window: Vec<u8>,
    window_pos: usize,
    pow_table: Vec<u64>,
}

impl RollingHash {
    /// Create a new rolling hash with specified window size.
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        let polynomial = RABIN_POLYNOMIAL;
        let mod_mask = (1u64 << 63) - 1; // Use lower 63 bits

        // Precompute powers of polynomial
        let mut pow_table = vec![1u64; window_size];
        for i in 1..window_size {
            pow_table[i] = pow_table[i - 1].wrapping_mul(polynomial) & mod_mask;
        }

        Self {
            window_size,
            polynomial,
            mod_mask,
            hash: 0,
            window: vec![0u8; window_size],
            window_pos: 0,
            pow_table,
        }
    }

    /// Roll the hash with a new byte.
    pub fn roll(&mut self, byte: u8) -> u64 {
        // Remove oldest byte's contribution
        let old_byte = self.window[self.window_pos];
        let old_contribution =
            u64::from(old_byte).wrapping_mul(self.pow_table[self.window_size - 1]);
        self.hash = self.hash.wrapping_sub(old_contribution) & self.mod_mask;

        // Shift hash and add new byte
        self.hash = self.hash.wrapping_mul(self.polynomial) & self.mod_mask;
        self.hash = self.hash.wrapping_add(u64::from(byte)) & self.mod_mask;

        // Update window
        self.window[self.window_pos] = byte;
        self.window_pos = (self.window_pos + 1) % self.window_size;

        self.hash
    }

    /// Get current hash value.
    #[must_use]
    pub fn hash(&self) -> u64 {
        self.hash
    }

    /// Reset the rolling hash.
    pub fn reset(&mut self) {
        self.hash = 0;
        self.window.fill(0);
        self.window_pos = 0;
    }
}

/// Chunker for content-based chunking.
pub struct Chunker {
    chunk_size: usize,
    min_size: usize,
    max_size: usize,
    rolling_hash: RollingHash,
}

impl Chunker {
    /// Create a new chunker with specified parameters.
    #[must_use]
    pub fn new(chunk_size: usize) -> Self {
        let chunk_size = chunk_size.clamp(MIN_CHUNK_SIZE, MAX_CHUNK_SIZE);
        let min_size = chunk_size / 2;
        let max_size = chunk_size * 2;
        let rolling_hash = RollingHash::new(64); // 64-byte window

        Self {
            chunk_size,
            min_size,
            max_size,
            rolling_hash,
        }
    }

    /// Check if this is a chunk boundary.
    fn is_boundary(&self, hash: u64) -> bool {
        // Use lower bits as boundary marker
        let mask = (1u64 << 12) - 1; // 12-bit mask for 4KB average chunks
        (hash & mask) == 0
    }

    /// Chunk data into content-defined chunks.
    #[must_use]
    pub fn chunk(&mut self, data: &[u8]) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let mut chunk_start = 0;
        let mut offset = 0u64;

        self.rolling_hash.reset();

        for (pos, &byte) in data.iter().enumerate() {
            let hash = self.rolling_hash.roll(byte);
            let chunk_len = pos - chunk_start;

            // Check for chunk boundary
            if chunk_len >= self.min_size && (self.is_boundary(hash) || chunk_len >= self.max_size)
            {
                let chunk_data = &data[chunk_start..pos + 1];
                let chunk_hash = compute_data_hash(chunk_data);

                chunks.push(Chunk {
                    offset,
                    size: chunk_data.len(),
                    hash: chunk_hash,
                });

                chunk_start = pos + 1;
                offset += chunk_data.len() as u64;
                self.rolling_hash.reset();
            }
        }

        // Add remaining data as final chunk
        if chunk_start < data.len() {
            let chunk_data = &data[chunk_start..];
            let chunk_hash = compute_data_hash(chunk_data);

            chunks.push(Chunk {
                offset,
                size: chunk_data.len(),
                hash: chunk_hash,
            });
        }

        chunks
    }

    /// Chunk a file into content-defined chunks.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read.
    pub fn chunk_file(&mut self, path: impl AsRef<Path>) -> DedupResult<Vec<Chunk>> {
        let file = File::open(path)?;
        let mut reader = BufReader::with_capacity(BUFFER_SIZE, file);
        let mut chunks = Vec::new();
        let mut chunk_buffer = Vec::new();
        let mut offset = 0u64;
        let mut buffer = vec![0u8; BUFFER_SIZE];

        self.rolling_hash.reset();

        loop {
            let n = reader.read(&mut buffer)?;
            if n == 0 {
                break;
            }

            for &byte in &buffer[..n] {
                chunk_buffer.push(byte);
                let hash = self.rolling_hash.roll(byte);
                let chunk_len = chunk_buffer.len();

                // Check for chunk boundary
                if chunk_len >= self.min_size
                    && (self.is_boundary(hash) || chunk_len >= self.max_size)
                {
                    let chunk_hash = compute_data_hash(&chunk_buffer);

                    chunks.push(Chunk {
                        offset,
                        size: chunk_buffer.len(),
                        hash: chunk_hash,
                    });

                    offset += chunk_buffer.len() as u64;
                    chunk_buffer.clear();
                    self.rolling_hash.reset();
                }
            }
        }

        // Add remaining data as final chunk
        if !chunk_buffer.is_empty() {
            let chunk_hash = compute_data_hash(&chunk_buffer);

            chunks.push(Chunk {
                offset,
                size: chunk_buffer.len(),
                hash: chunk_hash,
            });
        }

        Ok(chunks)
    }
}

/// Chunk index for deduplication.
pub struct ChunkIndex {
    chunks: Vec<(FileHash, Vec<String>)>,
}

impl ChunkIndex {
    /// Create a new chunk index.
    #[must_use]
    pub fn new() -> Self {
        Self { chunks: Vec::new() }
    }

    /// Add chunks from a file.
    pub fn add_file(&mut self, file_path: &str, chunks: &[Chunk]) {
        for chunk in chunks {
            if let Some((_, files)) = self.chunks.iter_mut().find(|(h, _)| h == &chunk.hash) {
                if !files.contains(&file_path.to_string()) {
                    files.push(file_path.to_string());
                }
            } else {
                self.chunks
                    .push((chunk.hash.clone(), vec![file_path.to_string()]));
            }
        }
    }

    /// Find duplicate chunks.
    #[must_use]
    pub fn find_duplicates(&self) -> Vec<(FileHash, Vec<String>)> {
        self.chunks
            .iter()
            .filter(|(_, files)| files.len() > 1)
            .cloned()
            .collect()
    }

    /// Get total number of chunks.
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Get number of duplicate chunks.
    #[must_use]
    pub fn duplicate_count(&self) -> usize {
        self.chunks
            .iter()
            .filter(|(_, files)| files.len() > 1)
            .count()
    }

    /// Calculate deduplication ratio.
    #[must_use]
    pub fn dedup_ratio(&self) -> f64 {
        if self.chunks.is_empty() {
            return 0.0;
        }
        let total: usize = self.chunks.iter().map(|(_, files)| files.len()).sum();
        let unique = self.chunks.len();
        if total == 0 {
            return 0.0;
        }
        1.0 - (unique as f64 / total as f64)
    }
}

impl Default for ChunkIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute similarity between two files based on shared chunks.
///
/// # Errors
///
/// Returns an error if files cannot be read or chunked.
pub fn compute_chunk_similarity(
    path1: impl AsRef<Path>,
    path2: impl AsRef<Path>,
    chunk_size: usize,
) -> DedupResult<f64> {
    let mut chunker = Chunker::new(chunk_size);

    let chunks1 = chunker.chunk_file(path1)?;
    let chunks2 = chunker.chunk_file(path2)?;

    if chunks1.is_empty() || chunks2.is_empty() {
        return Ok(0.0);
    }

    // Count shared chunks
    let hashes1: Vec<_> = chunks1.iter().map(|c| &c.hash).collect();
    let hashes2: Vec<_> = chunks2.iter().map(|c| &c.hash).collect();

    let shared = hashes1.iter().filter(|h| hashes2.contains(h)).count();

    let total = hashes1.len().max(hashes2.len());
    Ok(shared as f64 / total as f64)
}

/// Helper function to decode hex strings
mod hex {
    use super::DedupError;

    pub fn encode(bytes: [u8; 32]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect::<String>()
    }

    pub fn decode(s: &str) -> Result<Vec<u8>, DedupError> {
        if s.len() % 2 != 0 {
            return Err(DedupError::Hash("Odd hex string length".to_string()));
        }

        (0..s.len())
            .step_by(2)
            .map(|i| {
                u8::from_str_radix(&s[i..i + 2], 16)
                    .map_err(|e| DedupError::Hash(format!("Invalid hex digit: {e}")))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_data_hash() {
        let data = b"Hello, World!";
        let hash = compute_data_hash(data);
        assert_eq!(hash.as_bytes().len(), 32);
    }

    #[test]
    fn test_hash_hex() {
        let data = b"Test";
        let hash = compute_data_hash(data);
        let hex = hash.to_hex();
        assert_eq!(hex.len(), 64); // 32 bytes * 2 hex chars

        let decoded = FileHash::from_hex(&hex).expect("operation should succeed");
        assert_eq!(hash, decoded);
    }

    #[test]
    fn test_hash_similarity() {
        let hash1 = compute_data_hash(b"Hello");
        let hash2 = compute_data_hash(b"Hello");
        let hash3 = compute_data_hash(b"World");

        assert_eq!(hash1.similarity(&hash2), 1.0);
        assert!(hash1.similarity(&hash3) < 1.0);
    }

    #[test]
    fn test_rolling_hash() {
        let mut rh = RollingHash::new(8);
        let data = b"Hello, World!";

        for &byte in data {
            rh.roll(byte);
        }

        assert!(rh.hash() != 0);

        rh.reset();
        assert_eq!(rh.hash(), 0);
    }

    #[test]
    fn test_chunker() {
        let mut chunker = Chunker::new(DEFAULT_CHUNK_SIZE);
        let data = vec![0u8; 100_000]; // 100 KB of zeros

        let chunks = chunker.chunk(&data);
        assert!(!chunks.is_empty());

        // Verify chunk properties
        let total_size: usize = chunks.iter().map(|c| c.size).sum();
        assert_eq!(total_size, data.len());

        // Verify offsets
        let mut expected_offset = 0u64;
        for chunk in &chunks {
            assert_eq!(chunk.offset, expected_offset);
            expected_offset += chunk.size as u64;
        }
    }

    #[test]
    fn test_chunk_index() {
        let mut index = ChunkIndex::new();

        let chunks1 = vec![
            Chunk {
                offset: 0,
                size: 100,
                hash: compute_data_hash(b"chunk1"),
            },
            Chunk {
                offset: 100,
                size: 100,
                hash: compute_data_hash(b"chunk2"),
            },
        ];

        let chunks2 = vec![
            Chunk {
                offset: 0,
                size: 100,
                hash: compute_data_hash(b"chunk1"), // Duplicate
            },
            Chunk {
                offset: 100,
                size: 100,
                hash: compute_data_hash(b"chunk3"),
            },
        ];

        index.add_file("file1.txt", &chunks1);
        index.add_file("file2.txt", &chunks2);

        assert_eq!(index.chunk_count(), 3); // chunk1, chunk2, chunk3
        assert_eq!(index.duplicate_count(), 1); // chunk1 is duplicated

        let duplicates = index.find_duplicates();
        assert_eq!(duplicates.len(), 1);
        assert_eq!(duplicates[0].1.len(), 2); // chunk1 in both files
    }

    #[test]
    fn test_dedup_ratio() {
        let mut index = ChunkIndex::new();
        assert_eq!(index.dedup_ratio(), 0.0);

        // Add unique chunks
        let chunks = vec![
            Chunk {
                offset: 0,
                size: 100,
                hash: compute_data_hash(b"a"),
            },
            Chunk {
                offset: 100,
                size: 100,
                hash: compute_data_hash(b"b"),
            },
        ];
        index.add_file("file1", &chunks);

        // No deduplication yet
        assert_eq!(index.dedup_ratio(), 0.0);

        // Add duplicate chunks
        index.add_file("file2", &chunks);

        // 50% deduplication (2 unique, 4 total references)
        assert!((index.dedup_ratio() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_hamming_distance() {
        let hash1 = FileHash::from_bytes([0xFF; 32]);
        let hash2 = FileHash::from_bytes([0x00; 32]);
        assert_eq!(hash1.hamming_distance(&hash2), 256);

        let hash3 = FileHash::from_bytes([0xFF; 32]);
        assert_eq!(hash1.hamming_distance(&hash3), 0);
    }
}
