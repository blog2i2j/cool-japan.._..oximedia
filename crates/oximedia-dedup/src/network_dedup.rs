//! Network-aware deduplication for distributed media libraries.
//!
//! This module provides mechanisms to deduplicate media across multiple nodes in
//! a distributed system.  Rather than requiring every node to download every file,
//! nodes exchange compact **fingerprint manifests** and only transfer content when
//! necessary.
//!
//! # Design
//!
//! Each node maintains a local [`NodeManifest`] containing fingerprint summaries
//! (Blake3 hex digest, perceptual hash bits, duration, file size) for its local
//! media files.  Manifests are serialisable as JSON so they can be transmitted over
//! HTTP or any byte channel without coupling to a particular transport.
//!
//! The [`NetworkDedupEngine`] accepts manifests from multiple remote nodes and
//! computes cross-node duplicate groups by:
//!
//! 1. **Exact match** – identical Blake3 digests → definite duplicate.
//! 2. **Perceptual match** – Hamming distance on 64-bit pHash ≤ configured
//!    threshold → near-duplicate candidate.
//! 3. **Duration guard** – files with very different durations (> `duration_tolerance_s`)
//!    are excluded from perceptual matching to reduce false positives.
//!
//! # Example
//!
//! ```rust
//! use oximedia_dedup::network_dedup::{
//!     NetworkDedupEngine, NetworkDedupConfig, NodeManifest, FileRecord,
//! };
//!
//! let mut engine = NetworkDedupEngine::new(NetworkDedupConfig::default());
//!
//! let mut manifest_a = NodeManifest::new("node-a".to_string());
//! manifest_a.add_file(FileRecord::new(
//!     "node-a:/videos/movie.mp4".to_string(),
//!     "abcdef01".repeat(8),
//!     Some(0xDEAD_BEEF_1234_5678),
//!     Some(7200.0),
//!     Some(4_000_000_000),
//! ));
//!
//! let mut manifest_b = NodeManifest::new("node-b".to_string());
//! manifest_b.add_file(FileRecord::new(
//!     "node-b:/archive/movie_copy.mp4".to_string(),
//!     "abcdef01".repeat(8),
//!     Some(0xDEAD_BEEF_1234_5678),
//!     Some(7200.0),
//!     Some(4_000_000_000),
//! ));
//!
//! engine.add_manifest(manifest_a);
//! engine.add_manifest(manifest_b);
//!
//! let groups = engine.find_cross_node_duplicates();
//! assert!(!groups.is_empty());
//! ```

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// FileRecord
// ---------------------------------------------------------------------------

/// A single file entry within a node's manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRecord {
    /// Logical URI for this file (e.g. `"node-a:/path/to/file.mp4"`).
    pub uri: String,
    /// Lower-case hexadecimal BLAKE3 digest (64 hex characters).
    pub blake3_hex: String,
    /// Optional 64-bit perceptual hash.
    pub phash: Option<u64>,
    /// Optional duration in seconds.
    pub duration_s: Option<f64>,
    /// Optional file size in bytes.
    pub file_size: Option<u64>,
}

impl FileRecord {
    /// Create a new `FileRecord`.
    #[must_use]
    pub fn new(
        uri: String,
        blake3_hex: String,
        phash: Option<u64>,
        duration_s: Option<f64>,
        file_size: Option<u64>,
    ) -> Self {
        Self {
            uri,
            blake3_hex,
            phash,
            duration_s,
            file_size,
        }
    }

    /// Return `true` if this record has a valid-looking Blake3 hex digest.
    ///
    /// BLAKE3 produces 32 bytes → 64 hex characters.
    #[must_use]
    pub fn has_valid_digest(&self) -> bool {
        self.blake3_hex.len() == 64
            && self
                .blake3_hex
                .chars()
                .all(|c| c.is_ascii_hexdigit())
    }

    /// Compute the Hamming distance to another record's perceptual hash.
    ///
    /// Returns `None` if either record lacks a perceptual hash.
    #[must_use]
    pub fn phash_distance(&self, other: &Self) -> Option<u32> {
        match (self.phash, other.phash) {
            (Some(a), Some(b)) => Some((a ^ b).count_ones()),
            _ => None,
        }
    }

    /// Return the logical node name extracted from the URI prefix
    /// (`"node-a:/foo"` → `"node-a"`).
    #[must_use]
    pub fn node_name(&self) -> Option<&str> {
        self.uri.split(':').next()
    }
}

// ---------------------------------------------------------------------------
// NodeManifest
// ---------------------------------------------------------------------------

/// Fingerprint manifest for a single node.
///
/// A manifest holds all [`FileRecord`]s known to a particular node and can be
/// serialised/deserialised as JSON for transport.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeManifest {
    /// Human-readable node identifier.
    pub node_id: String,
    /// The file records.
    pub records: Vec<FileRecord>,
    /// Creation timestamp (Unix seconds) — informational only.
    pub created_at: u64,
}

impl NodeManifest {
    /// Create an empty manifest for `node_id`.
    #[must_use]
    pub fn new(node_id: String) -> Self {
        Self {
            node_id,
            records: Vec::new(),
            created_at: 0,
        }
    }

    /// Add a file record to the manifest.
    pub fn add_file(&mut self, record: FileRecord) {
        self.records.push(record);
    }

    /// Return the number of records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Return `true` if the manifest has no records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Serialise the manifest to a JSON string.
    ///
    /// # Errors
    ///
    /// Returns a `serde_json::Error` if serialisation fails (which should never
    /// happen for this type).
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialise a manifest from a JSON string.
    ///
    /// # Errors
    ///
    /// Returns a `serde_json::Error` if the JSON is malformed or the schema
    /// doesn't match.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the [`NetworkDedupEngine`].
#[derive(Debug, Clone)]
pub struct NetworkDedupConfig {
    /// Maximum Hamming distance for perceptual hash matching.
    pub phash_max_distance: u32,
    /// Maximum difference in duration (seconds) for two files to be considered
    /// near-duplicate candidates during perceptual matching.
    pub duration_tolerance_s: f64,
    /// Minimum file size (bytes) to include a record in perceptual matching.
    /// Very small files are excluded to avoid spurious perceptual matches.
    pub min_file_size: u64,
}

impl Default for NetworkDedupConfig {
    fn default() -> Self {
        Self {
            phash_max_distance: 10,
            duration_tolerance_s: 5.0,
            min_file_size: 65_536, // 64 KiB
        }
    }
}

// ---------------------------------------------------------------------------
// DuplicateGroup
// ---------------------------------------------------------------------------

/// A group of cross-node duplicate files.
#[derive(Debug, Clone)]
pub struct CrossNodeGroup {
    /// The URIs of all files in this duplicate group.
    pub uris: Vec<String>,
    /// How the duplicates were detected.
    pub method: DuplicateMethod,
    /// Perceptual hash distance (0 for exact duplicates, None if method != Perceptual).
    pub phash_distance: Option<u32>,
}

/// Detection method for cross-node duplicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DuplicateMethod {
    /// Identical BLAKE3 cryptographic digest.
    ExactHash,
    /// Perceptual hash Hamming distance within configured threshold.
    PerceptualHash,
}

// ---------------------------------------------------------------------------
// NetworkDedupEngine
// ---------------------------------------------------------------------------

/// Engine for detecting duplicates across distributed media nodes.
///
/// Add manifests from each remote node then call [`find_cross_node_duplicates`]
/// to get a list of [`CrossNodeGroup`]s.
#[derive(Debug)]
pub struct NetworkDedupEngine {
    config: NetworkDedupConfig,
    manifests: Vec<NodeManifest>,
}

impl NetworkDedupEngine {
    /// Create a new engine with the given configuration.
    #[must_use]
    pub fn new(config: NetworkDedupConfig) -> Self {
        Self {
            config,
            manifests: Vec::new(),
        }
    }

    /// Add a [`NodeManifest`] to the engine.
    pub fn add_manifest(&mut self, manifest: NodeManifest) {
        self.manifests.push(manifest);
    }

    /// Return the number of manifests registered.
    #[must_use]
    pub fn manifest_count(&self) -> usize {
        self.manifests.len()
    }

    /// Return the total number of file records across all manifests.
    #[must_use]
    pub fn total_records(&self) -> usize {
        self.manifests.iter().map(|m| m.records.len()).sum()
    }

    /// Find duplicate groups across all registered node manifests.
    ///
    /// The algorithm runs two passes:
    ///
    /// 1. **Exact pass** – groups records by Blake3 hex digest.  Only groups
    ///    that span at least two *different* nodes are returned.
    /// 2. **Perceptual pass** – for records with pHash and not already grouped
    ///    as exact duplicates, applies Hamming-distance comparison with the
    ///    configured threshold and duration guard.
    #[must_use]
    pub fn find_cross_node_duplicates(&self) -> Vec<CrossNodeGroup> {
        let mut groups = Vec::new();

        // Flatten all records with their node id attached.
        let all: Vec<(&str, &FileRecord)> = self
            .manifests
            .iter()
            .flat_map(|m| m.records.iter().map(move |r| (m.node_id.as_str(), r)))
            .collect();

        // --- Pass 1: exact hash ---
        let mut by_digest: HashMap<&str, Vec<(&str, &FileRecord)>> = HashMap::new();
        for &(node, rec) in &all {
            by_digest
                .entry(rec.blake3_hex.as_str())
                .or_default()
                .push((node, rec));
        }

        let mut exact_uris: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        for (_digest, records) in &by_digest {
            if records.len() < 2 {
                continue;
            }
            // Check that at least two *different* nodes are represented.
            let nodes: std::collections::HashSet<&str> =
                records.iter().map(|(n, _)| *n).collect();
            if nodes.len() < 2 {
                continue;
            }
            let uris: Vec<String> = records.iter().map(|(_, r)| r.uri.clone()).collect();
            for u in &uris {
                exact_uris.insert(u.clone());
            }
            groups.push(CrossNodeGroup {
                uris,
                method: DuplicateMethod::ExactHash,
                phash_distance: Some(0),
            });
        }

        // --- Pass 2: perceptual hash ---
        let phash_candidates: Vec<(&str, &FileRecord)> = all
            .iter()
            .filter(|(_, r)| {
                r.phash.is_some()
                    && !exact_uris.contains(&r.uri)
                    && r.file_size
                        .map(|s| s >= self.config.min_file_size)
                        .unwrap_or(true)
            })
            .copied()
            .collect();

        let n = phash_candidates.len();
        let mut grouped = vec![false; n];

        for i in 0..n {
            if grouped[i] {
                continue;
            }
            let (node_i, rec_i) = phash_candidates[i];
            let mut grp_uris = vec![rec_i.uri.clone()];
            let mut min_dist = u32::MAX;

            for j in (i + 1)..n {
                if grouped[j] {
                    continue;
                }
                let (node_j, rec_j) = phash_candidates[j];
                // Only match across different nodes.
                if node_i == node_j {
                    continue;
                }
                // Duration guard.
                if let (Some(d1), Some(d2)) =
                    (rec_i.duration_s, rec_j.duration_s)
                {
                    if (d1 - d2).abs() > self.config.duration_tolerance_s {
                        continue;
                    }
                }
                if let Some(dist) = rec_i.phash_distance(rec_j) {
                    if dist <= self.config.phash_max_distance {
                        grp_uris.push(rec_j.uri.clone());
                        grouped[j] = true;
                        if dist < min_dist {
                            min_dist = dist;
                        }
                    }
                }
            }

            if grp_uris.len() >= 2 {
                grouped[i] = true;
                groups.push(CrossNodeGroup {
                    uris: grp_uris,
                    method: DuplicateMethod::PerceptualHash,
                    phash_distance: if min_dist == u32::MAX {
                        None
                    } else {
                        Some(min_dist)
                    },
                });
            }
        }

        groups
    }

    /// Return a summary of how many duplicates span how many nodes.
    #[must_use]
    pub fn cross_node_summary(&self) -> CrossNodeSummary {
        let groups = self.find_cross_node_duplicates();
        let exact_groups = groups
            .iter()
            .filter(|g| g.method == DuplicateMethod::ExactHash)
            .count();
        let perceptual_groups = groups
            .iter()
            .filter(|g| g.method == DuplicateMethod::PerceptualHash)
            .count();
        let total_duplicate_files: usize = groups.iter().map(|g| g.uris.len()).sum();
        CrossNodeSummary {
            total_groups: groups.len(),
            exact_groups,
            perceptual_groups,
            total_duplicate_files,
        }
    }
}

/// Summary of cross-node deduplication results.
#[derive(Debug, Clone)]
pub struct CrossNodeSummary {
    /// Total number of duplicate groups found.
    pub total_groups: usize,
    /// Groups detected via exact hash.
    pub exact_groups: usize,
    /// Groups detected via perceptual hash.
    pub perceptual_groups: usize,
    /// Total number of file URIs across all duplicate groups.
    pub total_duplicate_files: usize,
}

impl Default for NetworkDedupEngine {
    fn default() -> Self {
        Self::new(NetworkDedupConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(uri: &str, digest: &str, phash: Option<u64>, dur: Option<f64>) -> FileRecord {
        FileRecord::new(
            uri.to_string(),
            digest.to_string(),
            phash,
            dur,
            Some(1_000_000),
        )
    }

    /// Build an engine with two nodes that share one exact duplicate.
    fn two_node_exact() -> NetworkDedupEngine {
        let mut engine = NetworkDedupEngine::new(NetworkDedupConfig::default());

        let digest = "a".repeat(64);
        let mut ma = NodeManifest::new("node-a".to_string());
        ma.add_file(make_record("node-a:/movie.mp4", &digest, None, Some(3600.0)));

        let mut mb = NodeManifest::new("node-b".to_string());
        mb.add_file(make_record(
            "node-b:/backup/movie.mp4",
            &digest,
            None,
            Some(3600.0),
        ));

        engine.add_manifest(ma);
        engine.add_manifest(mb);
        engine
    }

    #[test]
    fn test_exact_cross_node_duplicate() {
        let engine = two_node_exact();
        let groups = engine.find_cross_node_duplicates();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].method, DuplicateMethod::ExactHash);
        assert_eq!(groups[0].uris.len(), 2);
    }

    #[test]
    fn test_no_duplicate_same_node() {
        // Same digest but same node — should NOT appear in cross-node groups.
        let mut engine = NetworkDedupEngine::new(NetworkDedupConfig::default());
        let digest = "b".repeat(64);
        let mut ma = NodeManifest::new("node-a".to_string());
        ma.add_file(make_record("node-a:/v1.mp4", &digest, None, None));
        ma.add_file(make_record("node-a:/v2.mp4", &digest, None, None));
        engine.add_manifest(ma);

        let groups = engine.find_cross_node_duplicates();
        assert!(groups.is_empty(), "same-node duplicates must be excluded");
    }

    #[test]
    fn test_perceptual_cross_node_match() {
        let mut engine = NetworkDedupEngine::new(NetworkDedupConfig::default());
        // pHash distance of 2 — well within default threshold of 10.
        let base: u64 = 0xFF00_FF00_FF00_FF00;
        let close: u64 = base ^ 0b11; // 2 bits different

        let mut ma = NodeManifest::new("node-a".to_string());
        ma.add_file(make_record(
            "node-a:/clip.mp4",
            &"c".repeat(64),
            Some(base),
            Some(60.0),
        ));
        let mut mb = NodeManifest::new("node-b".to_string());
        mb.add_file(make_record(
            "node-b:/clip_re.mp4",
            &"d".repeat(64),
            Some(close),
            Some(60.0),
        ));
        engine.add_manifest(ma);
        engine.add_manifest(mb);

        let groups = engine.find_cross_node_duplicates();
        let perceptual: Vec<_> = groups
            .iter()
            .filter(|g| g.method == DuplicateMethod::PerceptualHash)
            .collect();
        assert_eq!(perceptual.len(), 1);
        assert_eq!(perceptual[0].phash_distance, Some(2));
    }

    #[test]
    fn test_duration_guard_excludes_mismatch() {
        let mut engine = NetworkDedupEngine::new(NetworkDedupConfig::default());
        let base: u64 = 0xAAAA_AAAA_AAAA_AAAA;

        let mut ma = NodeManifest::new("node-a".to_string());
        ma.add_file(make_record(
            "node-a:/short.mp4",
            &"e".repeat(64),
            Some(base),
            Some(30.0),
        ));
        let mut mb = NodeManifest::new("node-b".to_string());
        mb.add_file(make_record(
            "node-b:/long.mp4",
            &"f".repeat(64),
            Some(base ^ 1), // distance 1, but durations differ by 60 s
            Some(90.0),
        ));
        engine.add_manifest(ma);
        engine.add_manifest(mb);

        let groups = engine.find_cross_node_duplicates();
        // Duration differs by 60 s > tolerance 5 s → no perceptual match.
        let perceptual: Vec<_> = groups
            .iter()
            .filter(|g| g.method == DuplicateMethod::PerceptualHash)
            .collect();
        assert!(perceptual.is_empty(), "duration guard should exclude this pair");
    }

    #[test]
    fn test_manifest_serialise_roundtrip() {
        let mut m = NodeManifest::new("node-z".to_string());
        m.add_file(FileRecord::new(
            "node-z:/test.mp4".to_string(),
            "0".repeat(64),
            Some(12345),
            Some(99.9),
            Some(1024),
        ));
        let json = m.to_json().expect("serialise should succeed");
        let m2 = NodeManifest::from_json(&json).expect("deserialise should succeed");
        assert_eq!(m2.node_id, "node-z");
        assert_eq!(m2.records.len(), 1);
        assert_eq!(m2.records[0].phash, Some(12345));
    }

    #[test]
    fn test_file_record_valid_digest() {
        let good = FileRecord::new(
            "n:/f.mp4".to_string(),
            "a1b2c3".repeat(10) + "a1b2c3",  // 66 chars — invalid
            None,
            None,
            None,
        );
        // 66 hex chars is not 64 → invalid.
        assert!(!good.has_valid_digest());

        let valid = FileRecord::new(
            "n:/f.mp4".to_string(),
            "0".repeat(64),
            None,
            None,
            None,
        );
        assert!(valid.has_valid_digest());
    }

    #[test]
    fn test_phash_distance_calculation() {
        let a = FileRecord::new("n:/a.mp4".to_string(), "0".repeat(64), Some(0xFF), None, None);
        let b = FileRecord::new("n:/b.mp4".to_string(), "0".repeat(64), Some(0xFE), None, None);
        // 0xFF ^ 0xFE = 0x01 → 1 bit
        assert_eq!(a.phash_distance(&b), Some(1));

        let no_hash = FileRecord::new("n:/c.mp4".to_string(), "0".repeat(64), None, None, None);
        assert_eq!(a.phash_distance(&no_hash), None);
    }

    #[test]
    fn test_cross_node_summary() {
        let engine = two_node_exact();
        let summary = engine.cross_node_summary();
        assert_eq!(summary.total_groups, 1);
        assert_eq!(summary.exact_groups, 1);
        assert_eq!(summary.perceptual_groups, 0);
        assert_eq!(summary.total_duplicate_files, 2);
    }

    #[test]
    fn test_empty_engine() {
        let engine = NetworkDedupEngine::new(NetworkDedupConfig::default());
        assert_eq!(engine.manifest_count(), 0);
        assert_eq!(engine.total_records(), 0);
        let groups = engine.find_cross_node_duplicates();
        assert!(groups.is_empty());
    }

    #[test]
    fn test_node_name_extraction() {
        let rec = FileRecord::new(
            "node-alpha:/path/to/file.mp4".to_string(),
            "0".repeat(64),
            None,
            None,
            None,
        );
        assert_eq!(rec.node_name(), Some("node-alpha"));
    }

    #[test]
    fn test_three_node_perceptual_cluster() {
        // Three nodes all have pHashes within distance 10 of each other.
        // The greedy algorithm may group different pairs; we only require that
        // at least two of them end up in a perceptual group.
        let base: u64 = 0x0F0F_0F0F_0F0F_0F0F;
        let mut engine = NetworkDedupEngine::new(NetworkDedupConfig::default());

        for (node, delta) in [("n1", 0u64), ("n2", 0b1), ("n3", 0b11)] {
            let mut m = NodeManifest::new(node.to_string());
            m.add_file(make_record(
                &format!("{node}:/clip.mp4"),
                // Use distinct digests so exact-hash pass doesn't fire.
                &format!("{:0>64}", node),
                Some(base ^ delta),
                Some(120.0),
            ));
            engine.add_manifest(m);
        }

        let groups = engine.find_cross_node_duplicates();
        // At least one perceptual group should exist — n1/n2 and n1/n3 are
        // all within distance 3 which is well below the default threshold of 10.
        // The greedy pass guarantees at least the first pair is found.
        let perceptual_total: usize = groups
            .iter()
            .filter(|g| g.method == DuplicateMethod::PerceptualHash)
            .map(|g| g.uris.len())
            .sum();
        assert!(
            perceptual_total >= 2,
            "expected at least 2 files in perceptual groups, got {perceptual_total}"
        );
    }
}
