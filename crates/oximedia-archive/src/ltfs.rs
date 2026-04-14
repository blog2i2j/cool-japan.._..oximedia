//! LTFS (Linear Tape File System) metadata support.
//!
//! LTFS is an industry-standard format (SNIA LTFSv2) that makes LTO-5 and later
//! tape cartridges appear as a conventional file system to the operating system.
//! The tape is divided into two partitions:
//!
//! - **Index partition (partition 0)**: holds the LTFS label and the primary index.
//! - **Data partition (partition 1)**: holds file extents and optional mid-tape indexes.
//!
//! This module provides pure-Rust data structures and serialization helpers for:
//! - The LTFS **volume label** (written at BOT of both partitions)
//! - The LTFS **index** (an XML document describing the volume's directory tree)
//! - **Directory** and **file** nodes in the LTFS namespace
//! - **Extent** records mapping logical file byte ranges to tape block addresses
//! - **Schema validation** — checking that an index is structurally well-formed
//!
//! The serialization uses serde/serde_json for the data structures themselves;
//! the LTFS XML is produced/consumed by lightweight hand-written code so that
//! the crate has no xml-parser dependency.
//!
//! Reference: SNIA LTFS Format Specification version 2.5

use crate::{ArchiveError, ArchiveResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// LTFS version constants
// ---------------------------------------------------------------------------

/// LTFS format version 2.5 (current LTO-9/10 generation).
pub const LTFS_FORMAT_VERSION: &str = "2.5.0";

/// Minimum LTFS format version this implementation can read.
pub const LTFS_MIN_SUPPORTED_VERSION: &str = "2.0.0";

/// LTFS XML namespace.
pub const LTFS_XML_NAMESPACE: &str = "http://www.ibm.com/ltfs";

// ---------------------------------------------------------------------------
// LTFS Volume Label
// ---------------------------------------------------------------------------

/// The LTFS volume label written at BOT (Block 0) of both tape partitions.
///
/// The label is a fixed-size header that identifies the cartridge as LTFS and
/// records the block size and partition layout chosen during format time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LtfsVolumeLabel {
    /// LTFS format version string (e.g. `"2.5.0"`).
    pub format_version: String,

    /// Universally unique identifier for this volume (UUID v4, no hyphens).
    pub volume_uuid: String,

    /// Block size in bytes used when the tape was formatted.
    /// Common values: 65536 (64 KiB), 524288 (512 KiB), 1048576 (1 MiB).
    pub block_size: u64,

    /// The compression flag written during format.
    pub compression: bool,

    /// Timestamp of format operation.
    pub format_time: DateTime<Utc>,

    /// Human-readable cartridge label (barcode or user-assigned name).
    pub cartridge_label: String,

    /// Tape media type descriptor (e.g. `"LTO-9"`).
    pub media_type: String,
}

impl LtfsVolumeLabel {
    /// Create a new volume label with the given UUID and cartridge label.
    #[must_use]
    pub fn new(
        volume_uuid: impl Into<String>,
        cartridge_label: impl Into<String>,
        media_type: impl Into<String>,
        block_size: u64,
        compression: bool,
    ) -> Self {
        Self {
            format_version: LTFS_FORMAT_VERSION.to_string(),
            volume_uuid: volume_uuid.into(),
            cartridge_label: cartridge_label.into(),
            media_type: media_type.into(),
            block_size,
            compression,
            format_time: Utc::now(),
        }
    }

    /// Check whether the format version is supported by this implementation.
    #[must_use]
    pub fn is_version_supported(&self) -> bool {
        // Simple prefix check: major.minor must be >= 2.0
        self.format_version.starts_with("2.")
    }

    /// Serialize the label as a minimal LTFS-style XML string.
    ///
    /// In a real LTFS implementation this would be written as the first
    /// block(s) of both tape partitions.
    #[must_use]
    pub fn to_xml(&self) -> String {
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<ltfslabel version="{ver}" xmlns="{ns}">
  <creator>OxiMedia LTFS</creator>
  <formattime>{fmt}</formattime>
  <volumeuuid>{uuid}</volumeuuid>
  <blocksize>{bs}</blocksize>
  <compression>{comp}</compression>
  <cartridelabel>{cl}</cartridelabel>
  <mediatype>{mt}</mediatype>
</ltfslabel>"#,
            ver = self.format_version,
            ns = LTFS_XML_NAMESPACE,
            fmt = self.format_time.to_rfc3339(),
            uuid = self.volume_uuid,
            bs = self.block_size,
            comp = self.compression,
            cl = self.cartridge_label,
            mt = self.media_type,
        )
    }
}

impl fmt::Display for LtfsVolumeLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LtfsLabel[uuid={} label={} media={} block={}B]",
            self.volume_uuid, self.cartridge_label, self.media_type, self.block_size
        )
    }
}

// ---------------------------------------------------------------------------
// Extent — a contiguous range of tape blocks for one file's data
// ---------------------------------------------------------------------------

/// A single extent record mapping a logical file byte range to a tape location.
///
/// LTFS files may be stored in multiple extents (e.g. because the write was
/// interrupted and resumed later, or because the file was appended to).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LtfsExtent {
    /// Partition number (0 = index partition, 1 = data partition).
    pub partition: u8,

    /// Start block address on the tape (0-based, within the partition).
    pub start_block: u64,

    /// Offset within `start_block` in bytes (for sub-block granularity).
    pub byte_offset: u64,

    /// Number of bytes in this extent.
    pub byte_count: u64,

    /// Logical file offset at which this extent's data begins.
    pub file_offset: u64,
}

impl LtfsExtent {
    /// Create a new tape extent.
    #[must_use]
    pub fn new(
        partition: u8,
        start_block: u64,
        byte_offset: u64,
        byte_count: u64,
        file_offset: u64,
    ) -> Self {
        Self {
            partition,
            start_block,
            byte_offset,
            byte_count,
            file_offset,
        }
    }

    /// End block (exclusive) — the first tape block after this extent.
    ///
    /// This is a rough estimate assuming `block_size`-byte blocks with no
    /// partial blocks.
    #[must_use]
    pub fn end_block_estimate(&self, block_size: u64) -> u64 {
        if block_size == 0 {
            return self.start_block;
        }
        let total_bytes = self.byte_offset + self.byte_count;
        let blocks = total_bytes.div_ceil(block_size);
        self.start_block + blocks
    }
}

// ---------------------------------------------------------------------------
// File node
// ---------------------------------------------------------------------------

/// Metadata for a single file in the LTFS namespace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LtfsFileNode {
    /// File name (not the full path).
    pub name: String,

    /// Logical size of the file in bytes.
    pub length: u64,

    /// File creation time.
    pub creation_time: DateTime<Utc>,

    /// Last modification time.
    pub modification_time: DateTime<Utc>,

    /// Last access time.
    pub access_time: DateTime<Utc>,

    /// POSIX permission bits (octal, e.g. `0o644`).
    pub permissions: u32,

    /// Ordered list of extents that make up the file's data.
    pub extents: Vec<LtfsExtent>,

    /// User-defined extended attributes (key → UTF-8 value).
    pub extended_attributes: HashMap<String, String>,

    /// Checksum of the file data (hex string; algorithm determined by caller).
    pub checksum: Option<String>,
}

impl LtfsFileNode {
    /// Create a new file node with default timestamps set to `now`.
    #[must_use]
    pub fn new(name: impl Into<String>, length: u64) -> Self {
        let now = Utc::now();
        Self {
            name: name.into(),
            length,
            creation_time: now,
            modification_time: now,
            access_time: now,
            permissions: 0o644,
            extents: Vec::new(),
            extended_attributes: HashMap::new(),
            checksum: None,
        }
    }

    /// Add an extent to the file.
    pub fn add_extent(&mut self, extent: LtfsExtent) {
        self.extents.push(extent);
    }

    /// Set the file checksum.
    pub fn set_checksum(&mut self, checksum: impl Into<String>) {
        self.checksum = Some(checksum.into());
    }

    /// Total number of bytes covered by all extents.
    #[must_use]
    pub fn extent_total_bytes(&self) -> u64 {
        self.extents
            .iter()
            .map(|e| e.byte_count)
            .fold(0u64, |acc, n| acc.saturating_add(n))
    }

    /// Return `true` if the sum of extent byte-counts equals the declared file length.
    #[must_use]
    pub fn extents_consistent(&self) -> bool {
        self.extent_total_bytes() == self.length
    }
}

// ---------------------------------------------------------------------------
// Directory node
// ---------------------------------------------------------------------------

/// A directory node in the LTFS namespace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LtfsDirNode {
    /// Directory name (not the full path).
    pub name: String,

    /// Creation time.
    pub creation_time: DateTime<Utc>,

    /// Modification time.
    pub modification_time: DateTime<Utc>,

    /// Child files in this directory.
    pub files: Vec<LtfsFileNode>,

    /// Child subdirectories.
    pub subdirs: Vec<LtfsDirNode>,
}

impl LtfsDirNode {
    /// Create a new, empty directory node.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            name: name.into(),
            creation_time: now,
            modification_time: now,
            files: Vec::new(),
            subdirs: Vec::new(),
        }
    }

    /// Add a file to this directory.
    pub fn add_file(&mut self, file: LtfsFileNode) {
        self.files.push(file);
    }

    /// Add a subdirectory.
    pub fn add_subdir(&mut self, dir: LtfsDirNode) {
        self.subdirs.push(dir);
    }

    /// Recursively count the total number of files in this directory tree.
    #[must_use]
    pub fn file_count(&self) -> usize {
        let own = self.files.len();
        let children: usize = self.subdirs.iter().map(|d| d.file_count()).sum();
        own + children
    }

    /// Recursively sum the logical sizes of all files in this directory tree.
    #[must_use]
    pub fn total_logical_bytes(&self) -> u64 {
        let own: u64 = self
            .files
            .iter()
            .map(|f| f.length)
            .fold(0u64, |acc, n| acc.saturating_add(n));
        let children: u64 = self
            .subdirs
            .iter()
            .map(|d| d.total_logical_bytes())
            .fold(0u64, |acc, n| acc.saturating_add(n));
        own.saturating_add(children)
    }
}

// ---------------------------------------------------------------------------
// LTFS Index
// ---------------------------------------------------------------------------

/// The LTFS index XML document (in parsed form).
///
/// The index is written at BOT of the index partition after formatting and
/// appended to the data partition as a mid-tape index whenever the file system
/// is unmounted cleanly.  It contains the complete directory tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LtfsIndex {
    /// LTFS format version.
    pub format_version: String,

    /// UUID of the volume this index belongs to (must match label).
    pub volume_uuid: String,

    /// Monotonically increasing generation number.  Incremented on each
    /// successful unmount.
    pub generation: u64,

    /// Timestamp when this index was written.
    pub created_at: DateTime<Utc>,

    /// The root directory of the LTFS namespace.
    pub root: LtfsDirNode,

    /// Block address of the previous index on this partition (for rollback).
    pub previous_generation_offset: Option<u64>,
}

impl LtfsIndex {
    /// Create a new index for the given volume UUID.
    #[must_use]
    pub fn new(volume_uuid: impl Into<String>) -> Self {
        Self {
            format_version: LTFS_FORMAT_VERSION.to_string(),
            volume_uuid: volume_uuid.into(),
            generation: 1,
            created_at: Utc::now(),
            root: LtfsDirNode::new(""),
            previous_generation_offset: None,
        }
    }

    /// Increment the generation counter (call before each unmount write).
    pub fn bump_generation(&mut self) {
        self.generation = self.generation.saturating_add(1);
        self.created_at = Utc::now();
    }

    /// Total number of files in the index.
    #[must_use]
    pub fn total_file_count(&self) -> usize {
        self.root.file_count()
    }

    /// Total logical bytes of all files in the index.
    #[must_use]
    pub fn total_logical_bytes(&self) -> u64 {
        self.root.total_logical_bytes()
    }

    /// Validate that the index is structurally consistent.
    ///
    /// Currently checks:
    /// - Format version is recognised (`"2.x.y"`).
    /// - Volume UUID is non-empty.
    /// - Generation is non-zero.
    /// - Every file's extents are consistent (sum == declared length).
    pub fn validate(&self) -> ArchiveResult<()> {
        if !self.format_version.starts_with("2.") {
            return Err(ArchiveError::Validation(format!(
                "unsupported LTFS format version: {}",
                self.format_version
            )));
        }
        if self.volume_uuid.is_empty() {
            return Err(ArchiveError::Validation(
                "LTFS index has empty volume UUID".to_string(),
            ));
        }
        if self.generation == 0 {
            return Err(ArchiveError::Validation(
                "LTFS index generation must be >= 1".to_string(),
            ));
        }
        Self::validate_dir(&self.root)?;
        Ok(())
    }

    fn validate_dir(dir: &LtfsDirNode) -> ArchiveResult<()> {
        for file in &dir.files {
            if !file.extents.is_empty() && !file.extents_consistent() {
                return Err(ArchiveError::Validation(format!(
                    "LTFS file '{}' extent bytes ({}) != declared length ({})",
                    file.name,
                    file.extent_total_bytes(),
                    file.length
                )));
            }
        }
        for sub in &dir.subdirs {
            Self::validate_dir(sub)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LtfsIndexBuilder — fluent builder for constructing LTFS indexes
// ---------------------------------------------------------------------------

/// Builder for constructing an `LtfsIndex` incrementally.
///
/// ```rust
/// use oximedia_archive::ltfs::{LtfsIndexBuilder, LtfsExtent};
///
/// let index = LtfsIndexBuilder::new("vol-uuid-1234")
///     .add_file("root", "video.mkv", 1_000_000, vec![
///         LtfsExtent::new(1, 100, 0, 1_000_000, 0),
///     ])
///     .build();
///
/// assert_eq!(index.total_file_count(), 1);
/// assert_eq!(index.total_logical_bytes(), 1_000_000);
/// ```
pub struct LtfsIndexBuilder {
    index: LtfsIndex,
}

impl LtfsIndexBuilder {
    /// Create a new builder with the given volume UUID.
    #[must_use]
    pub fn new(volume_uuid: impl Into<String>) -> Self {
        Self {
            index: LtfsIndex::new(volume_uuid),
        }
    }

    /// Set an explicit generation number.
    #[must_use]
    pub fn generation(mut self, gen: u64) -> Self {
        self.index.generation = gen;
        self
    }

    /// Add a file directly to the root directory.
    #[must_use]
    pub fn add_file(
        mut self,
        _dir: &str,
        name: &str,
        length: u64,
        extents: Vec<LtfsExtent>,
    ) -> Self {
        let mut file = LtfsFileNode::new(name, length);
        for e in extents {
            file.add_extent(e);
        }
        self.index.root.add_file(file);
        self
    }

    /// Add a subdirectory to the root.
    #[must_use]
    pub fn add_subdir(mut self, dir: LtfsDirNode) -> Self {
        self.index.root.add_subdir(dir);
        self
    }

    /// Set the previous generation offset.
    #[must_use]
    pub fn previous_generation_offset(mut self, offset: u64) -> Self {
        self.index.previous_generation_offset = Some(offset);
        self
    }

    /// Consume the builder and return the finished index.
    #[must_use]
    pub fn build(self) -> LtfsIndex {
        self.index
    }
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

/// Serialize an `LtfsIndex` to a compact JSON string.
pub fn serialize_index(index: &LtfsIndex) -> ArchiveResult<String> {
    serde_json::to_string(index).map_err(|e| ArchiveError::Validation(e.to_string()))
}

/// Deserialize an `LtfsIndex` from a JSON string.
pub fn deserialize_index(json: &str) -> ArchiveResult<LtfsIndex> {
    serde_json::from_str(json).map_err(|e| ArchiveError::Validation(e.to_string()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_label() -> LtfsVolumeLabel {
        LtfsVolumeLabel::new(
            "550e8400-e29b-41d4-a716-446655440000",
            "LTO9-TAPE-001",
            "LTO-9",
            524_288,
            true,
        )
    }

    #[test]
    fn test_label_version_supported() {
        let label = make_label();
        assert!(label.is_version_supported());
    }

    #[test]
    fn test_label_unsupported_version() {
        let mut label = make_label();
        label.format_version = "1.0.0".to_string();
        assert!(!label.is_version_supported());
    }

    #[test]
    fn test_label_to_xml_contains_uuid() {
        let label = make_label();
        let xml = label.to_xml();
        assert!(xml.contains("550e8400-e29b-41d4-a716-446655440000"));
        assert!(xml.contains("ltfslabel"));
    }

    #[test]
    fn test_label_display() {
        let label = make_label();
        let s = label.to_string();
        assert!(s.contains("LTO9-TAPE-001"));
    }

    #[test]
    fn test_extent_end_block_estimate() {
        // 1 MiB extent starting at block 10, 512KiB block size → 2 blocks → end = 12
        let extent = LtfsExtent::new(1, 10, 0, 1_048_576, 0);
        assert_eq!(extent.end_block_estimate(524_288), 12);
    }

    #[test]
    fn test_extent_zero_block_size_no_panic() {
        let extent = LtfsExtent::new(1, 5, 0, 100, 0);
        assert_eq!(extent.end_block_estimate(0), 5); // returns start_block
    }

    #[test]
    fn test_file_node_extents_consistent() {
        let mut file = LtfsFileNode::new("test.mkv", 1000);
        file.add_extent(LtfsExtent::new(1, 0, 0, 600, 0));
        file.add_extent(LtfsExtent::new(1, 1, 0, 400, 600));
        assert!(file.extents_consistent());
    }

    #[test]
    fn test_file_node_extents_inconsistent() {
        let mut file = LtfsFileNode::new("test.mkv", 1000);
        file.add_extent(LtfsExtent::new(1, 0, 0, 500, 0)); // only 500, not 1000
        assert!(!file.extents_consistent());
    }

    #[test]
    fn test_dir_node_file_count_recursive() {
        let mut root = LtfsDirNode::new("root");
        root.add_file(LtfsFileNode::new("a.mkv", 100));
        let mut sub = LtfsDirNode::new("sub");
        sub.add_file(LtfsFileNode::new("b.mkv", 200));
        sub.add_file(LtfsFileNode::new("c.mkv", 300));
        root.add_subdir(sub);
        assert_eq!(root.file_count(), 3);
    }

    #[test]
    fn test_dir_node_total_logical_bytes() {
        let mut root = LtfsDirNode::new("root");
        root.add_file(LtfsFileNode::new("a.mkv", 1_000_000));
        let mut sub = LtfsDirNode::new("sub");
        sub.add_file(LtfsFileNode::new("b.mkv", 2_000_000));
        root.add_subdir(sub);
        assert_eq!(root.total_logical_bytes(), 3_000_000);
    }

    #[test]
    fn test_index_validate_ok() {
        let mut index = LtfsIndex::new("vol-uuid-0001");
        let mut file = LtfsFileNode::new("video.mkv", 500);
        file.add_extent(LtfsExtent::new(1, 0, 0, 500, 0));
        index.root.add_file(file);
        assert!(index.validate().is_ok());
    }

    #[test]
    fn test_index_validate_bad_version() {
        let mut index = LtfsIndex::new("vol-uuid-0002");
        index.format_version = "1.0.0".to_string();
        assert!(index.validate().is_err());
    }

    #[test]
    fn test_index_validate_empty_uuid() {
        let index = LtfsIndex::new("");
        assert!(index.validate().is_err());
    }

    #[test]
    fn test_index_validate_zero_generation() {
        let mut index = LtfsIndex::new("vol-uuid-0003");
        index.generation = 0;
        assert!(index.validate().is_err());
    }

    #[test]
    fn test_index_validate_inconsistent_extents() {
        let mut index = LtfsIndex::new("vol-uuid-0004");
        let mut file = LtfsFileNode::new("bad.mkv", 1000);
        file.add_extent(LtfsExtent::new(1, 0, 0, 500, 0)); // only 500 bytes
        index.root.add_file(file);
        assert!(index.validate().is_err());
    }

    #[test]
    fn test_builder_creates_correct_index() {
        let index = LtfsIndexBuilder::new("builder-vol-uuid")
            .generation(5)
            .add_file(
                "root",
                "archive.tar",
                2_000,
                vec![LtfsExtent::new(1, 0, 0, 2_000, 0)],
            )
            .build();

        assert_eq!(index.generation, 5);
        assert_eq!(index.total_file_count(), 1);
        assert_eq!(index.total_logical_bytes(), 2_000);
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let mut index = LtfsIndex::new("roundtrip-vol-uuid");
        index
            .root
            .add_file(LtfsFileNode::new("film.mkv", 8_000_000_000));
        let json = serialize_index(&index).expect("serialize should succeed");
        let restored = deserialize_index(&json).expect("deserialize should succeed");
        assert_eq!(restored.volume_uuid, index.volume_uuid);
        assert_eq!(restored.total_file_count(), 1);
    }

    #[test]
    fn test_bump_generation_increments() {
        let mut index = LtfsIndex::new("gen-test-uuid");
        assert_eq!(index.generation, 1);
        index.bump_generation();
        assert_eq!(index.generation, 2);
        index.bump_generation();
        assert_eq!(index.generation, 3);
    }

    #[test]
    fn test_builder_with_subdir() {
        let mut subdir = LtfsDirNode::new("dailies");
        subdir.add_file(LtfsFileNode::new("day01.mkv", 500));
        subdir.add_file(LtfsFileNode::new("day02.mkv", 600));

        let index = LtfsIndexBuilder::new("subdir-vol-uuid")
            .add_subdir(subdir)
            .build();

        assert_eq!(index.total_file_count(), 2);
    }

    #[test]
    fn test_label_xml_contains_media_type() {
        let label = make_label();
        let xml = label.to_xml();
        assert!(xml.contains("LTO-9"));
    }
}
