//! Compression store — compress/decompress object data with ratio tracking.
//!
//! Supports transparent compression using real oxiarc-lz4 and oxiarc-zstd
//! implementations, with auto-selection based on object size via
//! `CompressionPolicy`.
//!
//! # Magic-byte header
//!
//! Compressed payloads are prefixed with a 4-byte header identifying the
//! algorithm:
//! - `[0x4E, 0x4F, 0x4E, 0x45]` (`NONE`) — uncompressed passthrough
//! - `[0x4C, 0x5A, 0x34, 0x00]` (`LZ4\0`) — LZ4 frame format
//! - `[0x28, 0xB5, 0x2F, 0xFD]` — Zstandard frame magic
#![allow(dead_code)]

use crate::StorageError;
use std::collections::HashMap;

// ─── Magic-byte header constants ──────────────────────────────────────────────

/// Magic bytes written at the start of every stored payload.
const MAGIC_NONE: [u8; 4] = [0x4E, 0x4F, 0x4E, 0x45]; // "NONE"
const MAGIC_LZ4: [u8; 4] = [0x4C, 0x5A, 0x34, 0x00]; // "LZ4\0"
const MAGIC_ZSTD: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD]; // standard Zstd frame magic

// ─── CompressionPolicy ────────────────────────────────────────────────────────

/// Auto-selection policy that drives which algorithm is chosen at compression
/// time when the caller does not supply an explicit [`CompressionAlgorithm`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompressionPolicy {
    /// Size-aware automatic selection:
    /// - `< 4 KB` → no compression (overhead not worth it)
    /// - `4 KB – 1 MB` → LZ4 (fastest, lowest latency)
    /// - `> 1 MB` → Zstd level 3 (better ratio)
    #[default]
    Auto,
    /// Always use Zstandard (level 3), regardless of size.
    AlwaysZstd,
    /// Always use LZ4 frame format, regardless of size.
    AlwaysLz4,
    /// Skip compression entirely; store raw bytes.
    None,
}

impl CompressionPolicy {
    /// Select the concrete algorithm for a payload of `size_bytes` bytes.
    pub fn select_algorithm(self, size_bytes: usize) -> CompressionAlgorithm {
        match self {
            Self::Auto => {
                if size_bytes < 4_096 {
                    CompressionAlgorithm::None
                } else if size_bytes <= 1_048_576 {
                    CompressionAlgorithm::Lz4
                } else {
                    CompressionAlgorithm::Zstd
                }
            }
            Self::AlwaysZstd => CompressionAlgorithm::Zstd,
            Self::AlwaysLz4 => CompressionAlgorithm::Lz4,
            Self::None => CompressionAlgorithm::None,
        }
    }
}

// ─── CompressionAlgorithm ─────────────────────────────────────────────────────

/// Supported compression algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CompressionAlgorithm {
    /// No compression.
    None,
    /// Gzip (deflate with header).
    Gzip,
    /// Zstd — fast and high-ratio.
    #[default]
    Zstd,
    /// LZ4 — extremely fast, moderate ratio.
    Lz4,
    /// Brotli — high ratio, slower.
    Brotli,
    /// Snappy — very fast, modest ratio.
    Snappy,
}

impl CompressionAlgorithm {
    /// Estimate compression ratio (output / input) for typical media-adjacent data.
    #[allow(clippy::cast_precision_loss)]
    pub fn ratio_estimate(&self) -> f64 {
        match self {
            Self::None => 1.0,
            Self::Gzip => 0.35,
            Self::Zstd => 0.28,
            Self::Lz4 => 0.55,
            Self::Brotli => 0.25,
            Self::Snappy => 0.60,
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Gzip => "gzip",
            Self::Zstd => "zstd",
            Self::Lz4 => "lz4",
            Self::Brotli => "brotli",
            Self::Snappy => "snappy",
        }
    }

    /// Whether the algorithm supports streaming decompression.
    pub fn supports_streaming(&self) -> bool {
        matches!(self, Self::Gzip | Self::Zstd | Self::Lz4 | Self::Brotli)
    }

    /// The 4-byte magic header stored with this algorithm's output.
    pub fn magic_bytes(&self) -> [u8; 4] {
        match self {
            Self::Lz4 => MAGIC_LZ4,
            Self::Zstd => MAGIC_ZSTD,
            _ => MAGIC_NONE,
        }
    }

    /// Identify the algorithm from the first 4 bytes of a stored payload.
    ///
    /// Returns `None` if the header is not recognised.
    pub fn from_magic(header: &[u8; 4]) -> Option<Self> {
        if header == &MAGIC_LZ4 {
            Some(Self::Lz4)
        } else if header == &MAGIC_ZSTD {
            Some(Self::Zstd)
        } else if header == &MAGIC_NONE {
            Some(Self::None)
        } else {
            None
        }
    }
}

// ─── Real compress / decompress helpers ───────────────────────────────────────

/// Compress `data` with the specified algorithm, prepend the magic-byte header,
/// and return the resulting payload.
///
/// For algorithms that are not yet supported via oxiarc the data is stored
/// with a `NONE` header (i.e. no compression is applied) so that the store
/// remains usable.
fn compress_payload(data: &[u8], algo: CompressionAlgorithm) -> Result<Vec<u8>, StorageError> {
    let (magic, body): ([u8; 4], Vec<u8>) = match algo {
        CompressionAlgorithm::Lz4 => {
            let compressed = oxiarc_lz4::compress(data)
                .map_err(|e| StorageError::ProviderError(format!("LZ4 compression failed: {e}")))?;
            (MAGIC_LZ4, compressed)
        }
        CompressionAlgorithm::Zstd => {
            let compressed = oxiarc_zstd::compress_with_level(data, 3).map_err(|e| {
                StorageError::ProviderError(format!("Zstd compression failed: {e}"))
            })?;
            (MAGIC_ZSTD, compressed)
        }
        // Unsupported variants (Gzip, Brotli, Snappy) fall through to passthrough.
        _ => (MAGIC_NONE, data.to_vec()),
    };

    let mut out = Vec::with_capacity(4 + body.len());
    out.extend_from_slice(&magic);
    out.extend_from_slice(&body);
    Ok(out)
}

/// Decompress a stored payload by reading the 4-byte magic header and
/// dispatching to the correct decompressor.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>, StorageError> {
    if data.len() < 4 {
        return Err(StorageError::InvalidConfig(
            "compressed payload too short to contain magic header".to_string(),
        ));
    }
    let header: [u8; 4] = [data[0], data[1], data[2], data[3]];
    let body = &data[4..];

    let algo = CompressionAlgorithm::from_magic(&header).ok_or_else(|| {
        StorageError::InvalidConfig(format!("unknown compression magic header: {:02X?}", header))
    })?;

    match algo {
        CompressionAlgorithm::Lz4 => {
            // Try progressively larger output buffers to handle arbitrary
            // compression ratios without storing the original size.
            let mut max_output = body.len().saturating_mul(16).max(65_536);
            loop {
                match oxiarc_lz4::decompress(body, max_output) {
                    Ok(v) => return Ok(v),
                    Err(e) => {
                        let msg = e.to_string();
                        if msg.contains("exceeds maximum size") && max_output < 256 * 1024 * 1024 {
                            max_output = max_output.saturating_mul(4);
                        } else {
                            return Err(StorageError::ProviderError(format!(
                                "LZ4 decompression failed: {e}"
                            )));
                        }
                    }
                }
            }
        }
        CompressionAlgorithm::Zstd => oxiarc_zstd::decompress(body)
            .map_err(|e| StorageError::ProviderError(format!("Zstd decompression failed: {e}"))),
        _ => Ok(body.to_vec()),
    }
}

// ─── CompressionRecord ────────────────────────────────────────────────────────

/// A record of a single compression event.
#[derive(Debug, Clone)]
pub struct CompressionRecord {
    /// Object key.
    pub key: String,
    /// Algorithm used.
    pub algorithm: CompressionAlgorithm,
    /// Original size in bytes.
    pub original_bytes: u64,
    /// Compressed size in bytes (includes the 4-byte header).
    pub compressed_bytes: u64,
    /// Elapsed compression time in microseconds.
    pub elapsed_us: u64,
}

impl CompressionRecord {
    /// Create a new record.
    pub fn new(
        key: impl Into<String>,
        algorithm: CompressionAlgorithm,
        original_bytes: u64,
        compressed_bytes: u64,
        elapsed_us: u64,
    ) -> Self {
        Self {
            key: key.into(),
            algorithm,
            original_bytes,
            compressed_bytes,
            elapsed_us,
        }
    }

    /// Bytes saved by compression (0 if compressed is larger).
    pub fn saved_bytes(&self) -> u64 {
        self.original_bytes.saturating_sub(self.compressed_bytes)
    }

    /// Achieved compression ratio (compressed / original). Returns 1.0 for zero-size input.
    #[allow(clippy::cast_precision_loss)]
    pub fn achieved_ratio(&self) -> f64 {
        if self.original_bytes == 0 {
            return 1.0;
        }
        self.compressed_bytes as f64 / self.original_bytes as f64
    }

    /// Whether compression was beneficial (saved at least 1 byte).
    pub fn was_beneficial(&self) -> bool {
        self.compressed_bytes < self.original_bytes
    }

    /// Throughput in MB/s during compression.
    #[allow(clippy::cast_precision_loss)]
    pub fn throughput_mb_s(&self) -> f64 {
        if self.elapsed_us == 0 {
            return 0.0;
        }
        let mb = self.original_bytes as f64 / 1_048_576.0;
        let secs = self.elapsed_us as f64 / 1_000_000.0;
        mb / secs
    }
}

// ─── CompressionStoreConfig ───────────────────────────────────────────────────

/// Configuration for the compression store.
#[derive(Debug, Clone)]
pub struct CompressionStoreConfig {
    /// Default algorithm when none is specified.
    pub default_algorithm: CompressionAlgorithm,
    /// Minimum object size (bytes) to attempt compression.
    pub min_compress_bytes: u64,
    /// Skip compression if estimated ratio is above this threshold.
    pub ratio_threshold: f64,
}

impl Default for CompressionStoreConfig {
    fn default() -> Self {
        Self {
            default_algorithm: CompressionAlgorithm::Zstd,
            min_compress_bytes: 4096,
            ratio_threshold: 0.95,
        }
    }
}

// ─── CompressionStore ─────────────────────────────────────────────────────────

/// In-memory compression store with per-key records and aggregate statistics.
///
/// Supports transparent compression via a [`CompressionPolicy`] that
/// auto-selects the best algorithm for a given object size.
#[derive(Debug, Default)]
pub struct CompressionStore {
    config: CompressionStoreConfig,
    /// The active compression policy.
    policy: CompressionPolicy,
    /// key → compressed payload with magic-byte header.
    store: HashMap<String, Vec<u8>>,
    /// key → compression record.
    records: HashMap<String, CompressionRecord>,
}

impl CompressionStore {
    /// Create a new store with the given configuration and the default
    /// ([`CompressionPolicy::Auto`]) policy.
    pub fn new(config: CompressionStoreConfig) -> Self {
        Self {
            config,
            policy: CompressionPolicy::Auto,
            store: HashMap::new(),
            records: HashMap::new(),
        }
    }

    /// Create a new store with an explicit compression policy.
    ///
    /// The policy overrides the `default_algorithm` field of
    /// [`CompressionStoreConfig`] for policy-driven paths.
    pub fn with_policy(policy: CompressionPolicy) -> Self {
        Self {
            config: CompressionStoreConfig::default(),
            policy,
            store: HashMap::new(),
            records: HashMap::new(),
        }
    }

    /// Compress and store `data` under `key`.
    ///
    /// When `algorithm` is `Some(algo)` that specific algorithm is used.
    /// When `algorithm` is `None` the store's [`CompressionPolicy`] selects
    /// the algorithm based on data size.
    ///
    /// The stored payload always begins with a 4-byte magic-byte header so
    /// that [`Self::decompress`] can detect the algorithm automatically.
    pub fn compress(
        &mut self,
        key: impl Into<String>,
        data: &[u8],
        algorithm: Option<CompressionAlgorithm>,
    ) -> CompressionRecord {
        let key = key.into();
        let algo = algorithm.unwrap_or_else(|| self.policy.select_algorithm(data.len()));

        let original_bytes = data.len() as u64;
        let start = std::time::Instant::now();

        let payload = compress_payload(data, algo).unwrap_or_else(|_| {
            // Fallback: store raw with NONE magic on error.
            let mut v = Vec::with_capacity(4 + data.len());
            v.extend_from_slice(&MAGIC_NONE);
            v.extend_from_slice(data);
            v
        });

        let elapsed_us = start.elapsed().as_micros() as u64;
        let compressed_bytes = payload.len() as u64;

        let record = CompressionRecord::new(
            key.clone(),
            algo,
            original_bytes,
            compressed_bytes,
            elapsed_us,
        );

        self.store.insert(key.clone(), payload);
        self.records.insert(key, record.clone());
        record
    }

    /// Retrieve and decompress data for `key`.
    ///
    /// Returns `None` if the key does not exist.
    pub fn decompress_key(&self, key: &str) -> Option<Vec<u8>> {
        let payload = self.store.get(key)?;
        decompress(payload).ok()
    }

    /// Legacy alias for [`Self::decompress_key`] (returns the stored payload
    /// bytes, including the compression header, for compatibility with existing
    /// callers that just need to verify the payload is present).
    pub fn decompress(&self, key: &str) -> Option<Vec<u8>> {
        self.decompress_key(key)
    }

    /// Total bytes saved across all stored objects.
    pub fn space_saved_bytes(&self) -> u64 {
        self.records
            .values()
            .map(CompressionRecord::saved_bytes)
            .sum()
    }

    /// Retrieve the compression record for a key.
    pub fn record(&self, key: &str) -> Option<&CompressionRecord> {
        self.records.get(key)
    }

    /// Number of objects stored.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Remove an object and its record.
    pub fn remove(&mut self, key: &str) -> bool {
        let had = self.store.remove(key).is_some();
        self.records.remove(key);
        had
    }

    /// List all stored keys.
    pub fn keys(&self) -> Vec<&str> {
        self.store.keys().map(String::as_str).collect()
    }

    /// Active compression policy.
    pub fn policy(&self) -> CompressionPolicy {
        self.policy
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Existing tests (preserved) ───────────────────────────────────────────

    #[test]
    fn test_algorithm_ratio_none() {
        assert!((CompressionAlgorithm::None.ratio_estimate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_algorithm_ratio_zstd() {
        let r = CompressionAlgorithm::Zstd.ratio_estimate();
        assert!(r > 0.0 && r < 1.0);
    }

    #[test]
    fn test_algorithm_name() {
        assert_eq!(CompressionAlgorithm::Gzip.name(), "gzip");
        assert_eq!(CompressionAlgorithm::Lz4.name(), "lz4");
        assert_eq!(CompressionAlgorithm::Brotli.name(), "brotli");
    }

    #[test]
    fn test_algorithm_streaming_support() {
        assert!(CompressionAlgorithm::Gzip.supports_streaming());
        assert!(CompressionAlgorithm::Zstd.supports_streaming());
        assert!(!CompressionAlgorithm::Snappy.supports_streaming());
        assert!(!CompressionAlgorithm::None.supports_streaming());
    }

    #[test]
    fn test_algorithm_default() {
        assert_eq!(CompressionAlgorithm::default(), CompressionAlgorithm::Zstd);
    }

    #[test]
    fn test_record_saved_bytes() {
        let rec = CompressionRecord::new("k", CompressionAlgorithm::Gzip, 1000, 350, 100);
        assert_eq!(rec.saved_bytes(), 650);
    }

    #[test]
    fn test_record_saved_bytes_no_saving() {
        let rec = CompressionRecord::new("k", CompressionAlgorithm::None, 100, 120, 10);
        assert_eq!(rec.saved_bytes(), 0);
    }

    #[test]
    fn test_record_achieved_ratio() {
        let rec = CompressionRecord::new("k", CompressionAlgorithm::Zstd, 1000, 280, 50);
        let ratio = rec.achieved_ratio();
        assert!((ratio - 0.28).abs() < 1e-9);
    }

    #[test]
    fn test_record_zero_size() {
        let rec = CompressionRecord::new("k", CompressionAlgorithm::Gzip, 0, 0, 0);
        assert!((rec.achieved_ratio() - 1.0).abs() < 1e-9);
        assert_eq!(rec.saved_bytes(), 0);
    }

    #[test]
    fn test_record_was_beneficial() {
        let good = CompressionRecord::new("k", CompressionAlgorithm::Zstd, 1000, 280, 50);
        let bad = CompressionRecord::new("k", CompressionAlgorithm::Gzip, 100, 110, 5);
        assert!(good.was_beneficial());
        assert!(!bad.was_beneficial());
    }

    #[test]
    fn test_store_compress_and_decompress() {
        let mut store = CompressionStore::new(CompressionStoreConfig::default());
        let data = vec![42u8; 8192];
        store.compress("obj1", &data, Some(CompressionAlgorithm::Zstd));
        let out = store.decompress("obj1");
        assert!(out.is_some());
    }

    #[test]
    fn test_store_missing_key() {
        let store = CompressionStore::new(CompressionStoreConfig::default());
        assert!(store.decompress("nope").is_none());
    }

    #[test]
    fn test_store_space_saved_bytes() {
        let mut store = CompressionStore::new(CompressionStoreConfig::default());
        // Use highly compressible data (all zeros) to guarantee space savings.
        let data = vec![0u8; 32_768];
        store.compress("a", &data, Some(CompressionAlgorithm::Zstd));
        store.compress("b", &data, Some(CompressionAlgorithm::Lz4));
        assert!(store.space_saved_bytes() > 0);
    }

    #[test]
    fn test_store_remove() {
        let mut store = CompressionStore::new(CompressionStoreConfig::default());
        let data = vec![1u8; 8192];
        store.compress("del", &data, None);
        assert!(!store.is_empty());
        assert!(store.remove("del"));
        assert!(store.is_empty());
        assert!(!store.remove("del")); // second remove returns false
    }

    #[test]
    fn test_store_record_lookup() {
        let mut store = CompressionStore::new(CompressionStoreConfig::default());
        let data = vec![7u8; 8192];
        store.compress("x", &data, Some(CompressionAlgorithm::Lz4));
        let rec = store.record("x").expect("record should exist");
        assert_eq!(rec.algorithm, CompressionAlgorithm::Lz4);
        assert_eq!(rec.original_bytes, 8192);
    }

    #[test]
    fn test_store_small_data_not_compressed() {
        let cfg = CompressionStoreConfig {
            min_compress_bytes: 4096,
            ..Default::default()
        };
        let mut store = CompressionStore::new(cfg);
        let data = vec![99u8; 100]; // below threshold
        store.compress("small", &data, Some(CompressionAlgorithm::None));
        // Decompressing should recover the original 100 bytes.
        let out = store
            .decompress("small")
            .expect("decompress should succeed");
        assert_eq!(out.len(), 100);
    }

    // ── New policy-selection tests ────────────────────────────────────────────

    #[test]
    fn test_policy_auto_small_selects_none() {
        let algo = CompressionPolicy::Auto.select_algorithm(100);
        assert_eq!(algo, CompressionAlgorithm::None);
    }

    #[test]
    fn test_policy_auto_medium_selects_lz4() {
        let algo = CompressionPolicy::Auto.select_algorithm(8_192);
        assert_eq!(algo, CompressionAlgorithm::Lz4);
    }

    #[test]
    fn test_policy_auto_large_selects_zstd() {
        let algo = CompressionPolicy::Auto.select_algorithm(2_000_000);
        assert_eq!(algo, CompressionAlgorithm::Zstd);
    }

    #[test]
    fn test_policy_always_zstd() {
        assert_eq!(
            CompressionPolicy::AlwaysZstd.select_algorithm(100),
            CompressionAlgorithm::Zstd
        );
        assert_eq!(
            CompressionPolicy::AlwaysZstd.select_algorithm(10_000_000),
            CompressionAlgorithm::Zstd
        );
    }

    #[test]
    fn test_policy_always_lz4() {
        assert_eq!(
            CompressionPolicy::AlwaysLz4.select_algorithm(50),
            CompressionAlgorithm::Lz4
        );
    }

    #[test]
    fn test_policy_none_always_passthrough() {
        assert_eq!(
            CompressionPolicy::None.select_algorithm(10_000_000),
            CompressionAlgorithm::None
        );
    }

    #[test]
    fn test_with_policy_constructor() {
        let store = CompressionStore::with_policy(CompressionPolicy::AlwaysZstd);
        assert_eq!(store.policy(), CompressionPolicy::AlwaysZstd);
        assert!(store.is_empty());
    }

    // ── Round-trip compress/decompress tests ──────────────────────────────────

    #[test]
    fn test_roundtrip_lz4() {
        // Highly compressible data to ensure LZ4 actually compresses.
        let data: Vec<u8> = (0u8..=255).cycle().take(8_192).collect();
        let compressed = compress_payload(&data, CompressionAlgorithm::Lz4)
            .expect("LZ4 compress should succeed");
        let recovered = decompress(&compressed).expect("LZ4 decompress should succeed");
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_roundtrip_zstd() {
        let data: Vec<u8> = b"hello world "
            .iter()
            .copied()
            .cycle()
            .take(16_384)
            .collect();
        let compressed = compress_payload(&data, CompressionAlgorithm::Zstd)
            .expect("Zstd compress should succeed");
        let recovered = decompress(&compressed).expect("Zstd decompress should succeed");
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_roundtrip_none() {
        let data = b"small payload".to_vec();
        let compressed = compress_payload(&data, CompressionAlgorithm::None)
            .expect("None compress should succeed");
        // Header is NONE, body is the raw data.
        assert_eq!(&compressed[..4], &MAGIC_NONE);
        let recovered = decompress(&compressed).expect("None decompress should succeed");
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_store_roundtrip_via_policy() {
        let mut store = CompressionStore::with_policy(CompressionPolicy::Auto);
        // >1 MB → Zstd
        let large_data: Vec<u8> = (0u8..=255).cycle().take(1_100_000).collect();
        store.compress("big", &large_data, None);
        let recovered = store
            .decompress_key("big")
            .expect("decompress_key should succeed");
        assert_eq!(recovered, large_data);
    }

    #[test]
    fn test_unknown_header_returns_err() {
        let bad_payload = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02];
        let result = decompress(&bad_payload);
        assert!(
            result.is_err(),
            "Expected error for unknown magic header, got Ok"
        );
    }

    #[test]
    fn test_payload_too_short_returns_err() {
        let result = decompress(&[0x01, 0x02]);
        assert!(result.is_err());
    }

    #[test]
    fn test_magic_byte_round_trip_identity() {
        // Verify magic round-trip for LZ4 and Zstd.
        let lz4_magic = CompressionAlgorithm::Lz4.magic_bytes();
        assert_eq!(
            CompressionAlgorithm::from_magic(&lz4_magic),
            Some(CompressionAlgorithm::Lz4)
        );
        let zstd_magic = CompressionAlgorithm::Zstd.magic_bytes();
        assert_eq!(
            CompressionAlgorithm::from_magic(&zstd_magic),
            Some(CompressionAlgorithm::Zstd)
        );
        let none_magic = CompressionAlgorithm::None.magic_bytes();
        assert_eq!(
            CompressionAlgorithm::from_magic(&none_magic),
            Some(CompressionAlgorithm::None)
        );
    }

    // ── Round-trip data integrity tests ───────────────────────────────────────

    #[test]
    fn test_compression_store_roundtrip_1kb_repetitive_zstd() {
        let mut store = CompressionStore::with_policy(CompressionPolicy::AlwaysZstd);
        // ~1KB of repetitive data
        let original: Vec<u8> = b"Hello, world! "
            .iter()
            .copied()
            .cycle()
            .take(1000)
            .collect();
        store.compress("test-key", &original, Some(CompressionAlgorithm::Zstd));
        let retrieved = store
            .decompress_key("test-key")
            .expect("decompress should succeed");
        assert_eq!(retrieved, original, "round-trip data must be identical");
    }

    #[test]
    fn test_compression_store_roundtrip_1kb_repetitive_lz4() {
        let mut store = CompressionStore::with_policy(CompressionPolicy::AlwaysLz4);
        let original: Vec<u8> = (0u8..=127).cycle().take(1024).collect();
        store.compress("key-lz4", &original, Some(CompressionAlgorithm::Lz4));
        let retrieved = store
            .decompress_key("key-lz4")
            .expect("decompress should succeed");
        assert_eq!(retrieved, original);
    }

    #[test]
    fn test_compression_store_roundtrip_1mb_repetitive() {
        let mut store = CompressionStore::with_policy(CompressionPolicy::Auto);
        // 1MB of repetitive data (should select Zstd via Auto policy)
        let original: Vec<u8> = b"oximedia "
            .iter()
            .copied()
            .cycle()
            .take(1_048_576)
            .collect();
        store.compress("big-key", &original, None);
        let retrieved = store
            .decompress_key("big-key")
            .expect("decompress should succeed");
        assert_eq!(
            retrieved.len(),
            original.len(),
            "size must match after round-trip"
        );
        assert_eq!(retrieved, original, "data must be identical");
    }

    #[test]
    fn test_compression_store_roundtrip_empty_data() {
        let mut store = CompressionStore::with_policy(CompressionPolicy::None);
        let original: Vec<u8> = Vec::new();
        store.compress("empty", &original, Some(CompressionAlgorithm::None));
        let retrieved = store
            .decompress_key("empty")
            .expect("decompress empty should succeed");
        assert_eq!(retrieved, original);
    }

    #[test]
    fn test_compression_store_roundtrip_binary_all_bytes() {
        let mut store = CompressionStore::with_policy(CompressionPolicy::AlwaysZstd);
        // All 256 byte values repeated 32 times = 8192 bytes
        let original: Vec<u8> = (0u8..=255).cycle().take(8192).collect();
        store.compress("binary-key", &original, Some(CompressionAlgorithm::Zstd));
        let retrieved = store
            .decompress_key("binary-key")
            .expect("decompress should succeed");
        assert_eq!(
            retrieved, original,
            "binary data must survive round-trip intact"
        );
    }

    #[test]
    fn test_compression_store_compressed_smaller_for_repetitive_data() {
        let mut store = CompressionStore::with_policy(CompressionPolicy::AlwaysZstd);
        // Highly repetitive data should compress well
        let original: Vec<u8> = vec![0u8; 65_536]; // 64KB of zeros
        let record = store.compress("zeros", &original, Some(CompressionAlgorithm::Zstd));
        assert!(
            record.was_beneficial(),
            "compression must reduce size for all-zero data: compressed={}, original={}",
            record.compressed_bytes,
            record.original_bytes
        );
    }

    #[test]
    fn test_compression_store_auto_no_compression_for_tiny_input() {
        let mut store = CompressionStore::with_policy(CompressionPolicy::Auto);
        let original = b"tiny".to_vec(); // 4 bytes << 4KB threshold
                                         // Auto policy selects None for < 4KB
        let record = store.compress("tiny-key", &original, None);
        assert_eq!(record.algorithm, CompressionAlgorithm::None);
    }

    #[test]
    fn test_compression_store_auto_lz4_for_medium_input() {
        let mut store = CompressionStore::with_policy(CompressionPolicy::Auto);
        // 8KB → should select LZ4
        let original: Vec<u8> = (0u8..=255).cycle().take(8192).collect();
        let record = store.compress("medium-key", &original, None);
        assert_eq!(record.algorithm, CompressionAlgorithm::Lz4);
        // Verify data survives round-trip
        let retrieved = store
            .decompress_key("medium-key")
            .expect("decompress should succeed");
        assert_eq!(retrieved, original);
    }

    #[test]
    fn test_compression_store_auto_zstd_for_large_input() {
        let mut store = CompressionStore::with_policy(CompressionPolicy::Auto);
        // 2MB → should select Zstd
        let original: Vec<u8> = (0u8..=255).cycle().take(2_097_152).collect();
        let record = store.compress("large-key", &original, None);
        assert_eq!(record.algorithm, CompressionAlgorithm::Zstd);
        // Round-trip integrity
        let retrieved = store
            .decompress_key("large-key")
            .expect("decompress should succeed");
        assert_eq!(retrieved, original);
    }

    #[test]
    fn test_compression_store_multiple_keys_independent() {
        let mut store = CompressionStore::with_policy(CompressionPolicy::AlwaysZstd);
        let data_a: Vec<u8> = vec![0xAA; 8192];
        let data_b: Vec<u8> = vec![0xBB; 8192];
        store.compress("key-a", &data_a, None);
        store.compress("key-b", &data_b, None);
        let ra = store.decompress_key("key-a").expect("decompress a");
        let rb = store.decompress_key("key-b").expect("decompress b");
        assert_eq!(ra, data_a);
        assert_eq!(rb, data_b);
        assert_ne!(ra, rb, "different input data must yield different outputs");
    }
}
