#![allow(dead_code)]
//! Compression store — compress/decompress object data with ratio tracking.

use std::collections::HashMap;

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
}

/// A record of a single compression event.
#[derive(Debug, Clone)]
pub struct CompressionRecord {
    /// Object key.
    pub key: String,
    /// Algorithm used.
    pub algorithm: CompressionAlgorithm,
    /// Original size in bytes.
    pub original_bytes: u64,
    /// Compressed size in bytes.
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

/// In-memory compression store with per-key records and aggregate statistics.
#[derive(Debug, Default)]
pub struct CompressionStore {
    config: CompressionStoreConfig,
    /// key → compressed payload (simulated as reduced-length Vec).
    store: HashMap<String, Vec<u8>>,
    /// key → compression record.
    records: HashMap<String, CompressionRecord>,
}

impl CompressionStore {
    /// Create a new store with the given configuration.
    pub fn new(config: CompressionStoreConfig) -> Self {
        Self {
            config,
            store: HashMap::new(),
            records: HashMap::new(),
        }
    }

    /// Compress and store `data` under `key`.
    ///
    /// This is a simulated compress: the payload stored is a prefix of the original
    /// whose length equals `ceil(original * ratio_estimate)`.
    pub fn compress(
        &mut self,
        key: impl Into<String>,
        data: &[u8],
        algorithm: Option<CompressionAlgorithm>,
    ) -> CompressionRecord {
        let key = key.into();
        let algo = algorithm.unwrap_or(self.config.default_algorithm);
        let original_bytes = data.len() as u64;

        let start = std::time::Instant::now();

        // Simulate compression: if data is large enough, shrink it.
        let compressed: Vec<u8> = if original_bytes < self.config.min_compress_bytes {
            data.to_vec()
        } else {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let compressed_len = ((data.len() as f64 * algo.ratio_estimate()) as usize).max(1);
            data[..compressed_len.min(data.len())].to_vec()
        };

        let elapsed_us = start.elapsed().as_micros() as u64;
        let compressed_bytes = compressed.len() as u64;

        let record = CompressionRecord::new(
            key.clone(),
            algo,
            original_bytes,
            compressed_bytes,
            elapsed_us,
        );

        self.store.insert(key.clone(), compressed);
        self.records.insert(key, record.clone());
        record
    }

    /// Retrieve and decompress data for `key`.
    /// Returns `None` if the key does not exist.
    pub fn decompress(&self, key: &str) -> Option<Vec<u8>> {
        self.store.get(key).cloned()
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let data = vec![0u8; 16384];
        store.compress("a", &data, Some(CompressionAlgorithm::Zstd));
        store.compress("b", &data, Some(CompressionAlgorithm::Brotli));
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
        store.compress("small", &data, Some(CompressionAlgorithm::Zstd));
        // stored length should equal original (no compression attempted)
        let out = store
            .decompress("small")
            .expect("decompress should succeed");
        assert_eq!(out.len(), 100);
    }
}
