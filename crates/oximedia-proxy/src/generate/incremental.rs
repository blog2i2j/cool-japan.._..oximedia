//! Incremental proxy generation with segment change detection.
//!
//! This module provides content-hash-based comparison of media segments,
//! allowing the proxy pipeline to skip re-encoding unchanged segments
//! and only re-process segments that have been modified since the last proxy run.

use crate::{ProxyError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Size of each logical segment in bytes used for hash windowing.
const DEFAULT_SEGMENT_SIZE: usize = 1024 * 512; // 512 KiB

/// A content hash for a single segment of a media file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentHash {
    /// Zero-based index of this segment within the source file.
    pub index: u64,
    /// Byte offset where this segment starts.
    pub offset: u64,
    /// Byte length of this segment.
    pub length: u64,
    /// Pure-Rust CRC-32 hash of the segment bytes (hex string).
    pub hash: String,
}

impl SegmentHash {
    /// Create a new segment hash record.
    pub fn new(index: u64, offset: u64, length: u64, hash: &str) -> Self {
        Self {
            index,
            offset,
            length: length,
            hash: hash.to_string(),
        }
    }
}

/// Persisted manifest of all segment hashes for one source file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentManifest {
    /// Absolute path of the source media file.
    pub source_path: String,
    /// Total byte length of the source file at manifest creation time.
    pub total_size: u64,
    /// Segment size used when building this manifest.
    pub segment_size: usize,
    /// Per-segment hashes, keyed by segment index.
    pub segments: HashMap<u64, SegmentHash>,
}

impl SegmentManifest {
    /// Create an empty manifest for `source_path`.
    pub fn new(source_path: impl Into<String>, total_size: u64, segment_size: usize) -> Self {
        Self {
            source_path: source_path.into(),
            total_size,
            segment_size,
            segments: HashMap::new(),
        }
    }

    /// Insert or replace a segment hash.
    pub fn insert(&mut self, seg: SegmentHash) {
        self.segments.insert(seg.index, seg);
    }

    /// Number of segments in the manifest.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Serialize the manifest to JSON bytes.
    pub fn to_json(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| ProxyError::GenerationError(e.to_string()))
    }

    /// Deserialize a manifest from JSON bytes.
    pub fn from_json(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data).map_err(|e| ProxyError::GenerationError(e.to_string()))
    }
}

/// Result of comparing an old manifest against freshly computed hashes.
#[derive(Debug, Clone)]
pub struct ChangeSet {
    /// Segments that are new (no hash existed in the previous manifest).
    pub added: Vec<u64>,
    /// Segments whose hash has changed compared to the previous manifest.
    pub modified: Vec<u64>,
    /// Segment indices present in the old manifest but absent in the new one
    /// (source file shrank or segment count decreased).
    pub removed: Vec<u64>,
    /// Segments that are identical in both manifests.
    pub unchanged: Vec<u64>,
}

impl ChangeSet {
    /// `true` if no segments changed, were added, or were removed.
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.modified.is_empty() && self.removed.is_empty()
    }

    /// Total number of segments that require re-encoding.
    pub fn dirty_count(&self) -> usize {
        self.added.len() + self.modified.len()
    }

    /// Total number of segments (dirty + clean).
    pub fn total_count(&self) -> usize {
        self.added.len() + self.modified.len() + self.removed.len() + self.unchanged.len()
    }
}

/// Pure-Rust CRC-32 (IEEE 802.3 polynomial).
fn crc32_ieee(data: &[u8]) -> u32 {
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

/// Engine for building segment manifests and detecting changes.
pub struct IncrementalEngine {
    segment_size: usize,
}

impl IncrementalEngine {
    /// Create a new engine with the default segment size (512 KiB).
    pub fn new() -> Self {
        Self {
            segment_size: DEFAULT_SEGMENT_SIZE,
        }
    }

    /// Create an engine with a custom segment size in bytes.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` when `segment_size` is zero.
    pub fn with_segment_size(segment_size: usize) -> Result<Self> {
        if segment_size == 0 {
            return Err(ProxyError::InvalidInput(
                "Segment size must be greater than zero".to_string(),
            ));
        }
        Ok(Self { segment_size })
    }

    /// Build a `SegmentManifest` from raw `data` for the given `source_path`.
    #[allow(clippy::cast_possible_truncation)]
    pub fn build_manifest(&self, source_path: impl Into<String>, data: &[u8]) -> SegmentManifest {
        let total_size = data.len() as u64;
        let mut manifest = SegmentManifest::new(source_path, total_size, self.segment_size);

        for (index, chunk) in data.chunks(self.segment_size).enumerate() {
            let index = index as u64;
            let offset = index * self.segment_size as u64;
            let length = chunk.len() as u64;
            let hash = format!("{:08x}", crc32_ieee(chunk));
            manifest.insert(SegmentHash::new(index, offset, length, &hash));
        }

        manifest
    }

    /// Build a manifest from a file on disk.
    ///
    /// # Errors
    ///
    /// Returns `IoError` when the file cannot be read.
    pub fn build_manifest_from_file(&self, path: &Path) -> Result<SegmentManifest> {
        let data = std::fs::read(path)?;
        let source = path.display().to_string();
        Ok(self.build_manifest(source, &data))
    }

    /// Compare `old` manifest against freshly-computed `new_data` for the same source.
    ///
    /// Segments are compared by their CRC-32 hash; any change (even a single byte) is
    /// detected as a modification.
    #[allow(clippy::cast_possible_truncation)]
    pub fn diff(&self, old: &SegmentManifest, new_data: &[u8]) -> ChangeSet {
        let new_manifest = self.build_manifest(old.source_path.clone(), new_data);

        let mut added = Vec::new();
        let mut modified = Vec::new();
        let mut unchanged = Vec::new();
        let mut removed = Vec::new();

        // Check new segments against old
        for (idx, new_seg) in &new_manifest.segments {
            match old.segments.get(idx) {
                None => added.push(*idx),
                Some(old_seg) if old_seg.hash != new_seg.hash => modified.push(*idx),
                _ => unchanged.push(*idx),
            }
        }

        // Any segment that existed before but is now absent has been removed
        for idx in old.segments.keys() {
            if !new_manifest.segments.contains_key(idx) {
                removed.push(*idx);
            }
        }

        added.sort_unstable();
        modified.sort_unstable();
        unchanged.sort_unstable();
        removed.sort_unstable();

        ChangeSet {
            added,
            modified,
            removed,
            unchanged,
        }
    }

    /// Decide which input→output segment pairs need re-encoding given a change set
    /// and an output directory.
    ///
    /// Returns a list of `(segment_index, output_path)` tuples for dirty segments.
    pub fn dirty_segments(
        &self,
        change_set: &ChangeSet,
        output_dir: impl AsRef<Path>,
        base_name: &str,
    ) -> Vec<(u64, PathBuf)> {
        let dir = output_dir.as_ref();
        let mut pairs = Vec::new();
        for &idx in change_set.added.iter().chain(change_set.modified.iter()) {
            let filename = format!("{base_name}_seg{idx:06}.tmp");
            pairs.push((idx, dir.join(filename)));
        }
        pairs.sort_unstable_by_key(|(i, _)| *i);
        pairs
    }
}

impl Default for IncrementalEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data(size: usize, fill: u8) -> Vec<u8> {
        vec![fill; size]
    }

    #[test]
    fn test_engine_default_segment_size() {
        let engine = IncrementalEngine::new();
        assert_eq!(engine.segment_size, DEFAULT_SEGMENT_SIZE);
    }

    #[test]
    fn test_engine_custom_segment_size() {
        let engine = IncrementalEngine::with_segment_size(1024).expect("should succeed in test");
        assert_eq!(engine.segment_size, 1024);
    }

    #[test]
    fn test_engine_zero_segment_size_err() {
        let result = IncrementalEngine::with_segment_size(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_manifest_empty_data() {
        let engine = IncrementalEngine::with_segment_size(64).expect("should succeed in test");
        let manifest = engine.build_manifest("/src/empty.mov", b"");
        assert_eq!(manifest.total_size, 0);
        assert_eq!(manifest.segment_count(), 0);
    }

    #[test]
    fn test_build_manifest_single_segment() {
        let engine = IncrementalEngine::with_segment_size(256).expect("should succeed in test");
        let data = make_data(100, 0xAB);
        let manifest = engine.build_manifest("/src/clip.mov", &data);
        assert_eq!(manifest.total_size, 100);
        assert_eq!(manifest.segment_count(), 1);
        assert_eq!(manifest.segments[&0].offset, 0);
        assert_eq!(manifest.segments[&0].length, 100);
    }

    #[test]
    fn test_build_manifest_multiple_segments() {
        let engine = IncrementalEngine::with_segment_size(64).expect("should succeed in test");
        let data = make_data(200, 0x00);
        let manifest = engine.build_manifest("/src/clip.mov", &data);
        // 200 / 64 = 3 full + 1 partial = 4 segments
        assert_eq!(manifest.segment_count(), 4);
    }

    #[test]
    fn test_manifest_json_roundtrip() {
        let engine = IncrementalEngine::with_segment_size(64).expect("should succeed in test");
        let data = make_data(128, 0xFF);
        let manifest = engine.build_manifest("/src/file.mov", &data);
        let json = manifest.to_json().expect("should succeed in test");
        let restored = SegmentManifest::from_json(&json).expect("should succeed in test");
        assert_eq!(restored.source_path, "/src/file.mov");
        assert_eq!(restored.segment_count(), manifest.segment_count());
        for (idx, seg) in &manifest.segments {
            assert_eq!(restored.segments[idx].hash, seg.hash);
        }
    }

    #[test]
    fn test_diff_no_changes() {
        let engine = IncrementalEngine::with_segment_size(64).expect("should succeed in test");
        let data = make_data(200, 0xAA);
        let old_manifest = engine.build_manifest("/src/clip.mov", &data);
        let cs = engine.diff(&old_manifest, &data);
        assert!(cs.is_empty());
        assert_eq!(cs.unchanged.len(), 4);
    }

    #[test]
    fn test_diff_single_segment_modified() {
        let engine = IncrementalEngine::with_segment_size(64).expect("should succeed in test");
        let old_data = make_data(128, 0xAA);
        let old_manifest = engine.build_manifest("/src/clip.mov", &old_data);

        let mut new_data = old_data.clone();
        // Modify first byte of second segment
        new_data[64] = 0xFF;

        let cs = engine.diff(&old_manifest, &new_data);
        assert_eq!(cs.modified.len(), 1);
        assert_eq!(cs.modified[0], 1); // second segment index = 1
        assert_eq!(cs.unchanged.len(), 1); // first segment unchanged
        assert!(!cs.is_empty());
        assert_eq!(cs.dirty_count(), 1);
    }

    #[test]
    fn test_diff_data_grew() {
        let engine = IncrementalEngine::with_segment_size(64).expect("should succeed in test");
        let old_data = make_data(64, 0xAA);
        let old_manifest = engine.build_manifest("/src/clip.mov", &old_data);

        let mut new_data = old_data.clone();
        new_data.extend_from_slice(&make_data(64, 0xBB));

        let cs = engine.diff(&old_manifest, &new_data);
        assert_eq!(cs.added.len(), 1); // one new segment
        assert_eq!(cs.unchanged.len(), 1); // first segment still identical
    }

    #[test]
    fn test_diff_data_shrank() {
        let engine = IncrementalEngine::with_segment_size(64).expect("should succeed in test");
        let old_data = make_data(128, 0xAA);
        let old_manifest = engine.build_manifest("/src/clip.mov", &old_data);

        let new_data = make_data(64, 0xAA);
        let cs = engine.diff(&old_manifest, &new_data);
        assert_eq!(cs.removed.len(), 1);
        assert_eq!(cs.unchanged.len(), 1);
    }

    #[test]
    fn test_dirty_segments_paths() {
        let engine = IncrementalEngine::with_segment_size(64).expect("should succeed in test");
        let old_data = make_data(128, 0xAA);
        let old_manifest = engine.build_manifest("/src/clip.mov", &old_data);

        let mut new_data = old_data.clone();
        new_data[64] = 0xFF; // change segment 1

        let cs = engine.diff(&old_manifest, &new_data);
        let pairs = engine.dirty_segments(&cs, "/proxy/segments", "clip");
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, 1);
        assert!(pairs[0].1.to_str().is_some_and(|s| s.contains("seg000001")));
    }

    #[test]
    fn test_change_set_total_count() {
        let cs = ChangeSet {
            added: vec![5],
            modified: vec![2, 3],
            removed: vec![10],
            unchanged: vec![0, 1, 4],
        };
        assert_eq!(cs.total_count(), 7);
        assert_eq!(cs.dirty_count(), 3);
    }

    #[test]
    fn test_crc32_deterministic() {
        let a = crc32_ieee(b"hello world");
        let b = crc32_ieee(b"hello world");
        assert_eq!(a, b);
    }

    #[test]
    fn test_crc32_sensitivity() {
        let a = crc32_ieee(b"hello");
        let b = crc32_ieee(b"Hello");
        assert_ne!(a, b);
    }
}

// ============================================================================
// On-Demand Frame Generation
// ============================================================================

/// Status of a single proxy frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrameStatus {
    /// Frame has not been requested yet.
    Pending,
    /// Frame generation is currently in progress.
    Generating,
    /// Frame has been successfully generated.
    Ready,
    /// Frame generation failed.
    Failed,
}

/// Metadata for a single proxy frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameRecord {
    /// Frame index (0-based).
    pub index: u64,
    /// Presentation timestamp in milliseconds.
    pub pts_ms: u64,
    /// Path where the generated frame is (or will be) stored.
    pub output_path: String,
    /// Current generation status.
    pub status: FrameStatus,
    /// Size of the generated frame file in bytes, if ready.
    pub size_bytes: Option<u64>,
}

impl FrameRecord {
    /// Create a new frame record in `Pending` state.
    pub fn new(index: u64, pts_ms: u64, output_path: impl Into<String>) -> Self {
        Self {
            index,
            pts_ms,
            output_path: output_path.into(),
            status: FrameStatus::Pending,
            size_bytes: None,
        }
    }

    /// Mark the frame as generating.
    pub fn set_generating(&mut self) {
        self.status = FrameStatus::Generating;
    }

    /// Mark the frame as ready with its file size.
    pub fn set_ready(&mut self, size_bytes: u64) {
        self.status = FrameStatus::Ready;
        self.size_bytes = Some(size_bytes);
    }

    /// Mark the frame as failed.
    pub fn set_failed(&mut self) {
        self.status = FrameStatus::Failed;
        self.size_bytes = None;
    }

    /// Whether the frame is available for use.
    pub fn is_ready(&self) -> bool {
        self.status == FrameStatus::Ready
    }
}

/// On-demand proxy frame generator.
///
/// Instead of transcoding an entire source file upfront, `OnDemandProxyGenerator`
/// tracks individual frame records and generates them only when explicitly
/// requested via [`OnDemandProxyGenerator::request_frame`].
///
/// This is ideal for non-linear editing workflows where only a subset of frames
/// from a long source clip may ever be displayed.
#[derive(Debug)]
pub struct OnDemandProxyGenerator {
    /// Source path this generator is associated with.
    source_path: String,
    /// All known frames, keyed by frame index.
    frames: HashMap<u64, FrameRecord>,
    /// Directory where generated frame files are placed.
    output_dir: PathBuf,
    /// Base name used when constructing frame file names.
    base_name: String,
    /// Total number of frames in the source (may be 0 if unknown).
    total_frames: u64,
}

impl OnDemandProxyGenerator {
    /// Create a new on-demand generator.
    ///
    /// # Arguments
    /// * `source_path` – Absolute path of the source media file.
    /// * `output_dir` – Directory where frame proxy files will be stored.
    /// * `base_name` – Stem used when naming frame files.
    /// * `total_frames` – Total number of frames in the source (0 = unknown).
    pub fn new(
        source_path: impl Into<String>,
        output_dir: impl Into<PathBuf>,
        base_name: impl Into<String>,
        total_frames: u64,
    ) -> Self {
        Self {
            source_path: source_path.into(),
            frames: HashMap::new(),
            output_dir: output_dir.into(),
            base_name: base_name.into(),
            total_frames,
        }
    }

    /// Register a frame at `index` with its presentation timestamp.
    ///
    /// If the frame was already registered its record is returned unchanged.
    /// Newly registered frames start in `Pending` state.
    pub fn register_frame(&mut self, index: u64, pts_ms: u64) -> &FrameRecord {
        self.frames.entry(index).or_insert_with(|| {
            let fname = format!("{}_frame{:08}.proxy", self.base_name, index);
            let path = self.output_dir.join(fname).display().to_string();
            FrameRecord::new(index, pts_ms, path)
        })
    }

    /// Request generation of a specific frame by index.
    ///
    /// If the frame is already `Ready` or `Generating` this is a no-op.
    /// Otherwise the frame transitions to `Generating` and a simulated
    /// generation step is performed (in real usage, callers would drive the
    /// actual transcoder and call [`OnDemandProxyGenerator::mark_ready`]).
    ///
    /// Returns `true` if generation was newly triggered, `false` if the frame
    /// was already in a terminal or active state.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if `index` has not been registered.
    pub fn request_frame(&mut self, index: u64) -> Result<bool> {
        let frame = self.frames.get_mut(&index).ok_or_else(|| {
            ProxyError::InvalidInput(format!(
                "Frame {index} not registered in on-demand generator for '{}'",
                self.source_path
            ))
        })?;
        match frame.status {
            FrameStatus::Ready | FrameStatus::Generating => Ok(false),
            FrameStatus::Pending | FrameStatus::Failed => {
                frame.set_generating();
                Ok(true)
            }
        }
    }

    /// Mark a frame as successfully generated.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if `index` has not been registered.
    pub fn mark_ready(&mut self, index: u64, size_bytes: u64) -> Result<()> {
        let frame = self.frames.get_mut(&index).ok_or_else(|| {
            ProxyError::InvalidInput(format!("Frame {index} not registered"))
        })?;
        frame.set_ready(size_bytes);
        Ok(())
    }

    /// Mark a frame as failed.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if `index` has not been registered.
    pub fn mark_failed(&mut self, index: u64) -> Result<()> {
        let frame = self.frames.get_mut(&index).ok_or_else(|| {
            ProxyError::InvalidInput(format!("Frame {index} not registered"))
        })?;
        frame.set_failed();
        Ok(())
    }

    /// Return a reference to a frame record.
    pub fn get_frame(&self, index: u64) -> Option<&FrameRecord> {
        self.frames.get(&index)
    }

    /// Return the number of registered frames.
    pub fn registered_count(&self) -> usize {
        self.frames.len()
    }

    /// Return the number of frames in `Ready` state.
    pub fn ready_count(&self) -> usize {
        self.frames.values().filter(|f| f.is_ready()).count()
    }

    /// Return the number of frames in `Pending` state.
    pub fn pending_count(&self) -> usize {
        self.frames
            .values()
            .filter(|f| f.status == FrameStatus::Pending)
            .count()
    }

    /// Fraction of registered frames that are ready, in `[0.0, 1.0]`.
    pub fn completion_ratio(&self) -> f64 {
        if self.frames.is_empty() {
            return 0.0;
        }
        self.ready_count() as f64 / self.frames.len() as f64
    }

    /// Sorted list of frame indices that still need generation.
    pub fn pending_indices(&self) -> Vec<u64> {
        let mut v: Vec<u64> = self
            .frames
            .values()
            .filter(|f| matches!(f.status, FrameStatus::Pending | FrameStatus::Failed))
            .map(|f| f.index)
            .collect();
        v.sort_unstable();
        v
    }

    /// Source path associated with this generator.
    pub fn source_path(&self) -> &str {
        &self.source_path
    }

    /// Declared total frame count (0 = unknown).
    pub fn total_frames(&self) -> u64 {
        self.total_frames
    }
}

#[cfg(test)]
mod on_demand_tests {
    use super::*;

    fn make_gen(total: u64) -> OnDemandProxyGenerator {
        let out = std::env::temp_dir().join("oximedia-proxy-inc-proxy");
        OnDemandProxyGenerator::new("/src/clip.mov", out, "clip", total)
    }

    #[test]
    fn test_register_and_request_frame() {
        let mut gen = make_gen(100);
        gen.register_frame(0, 0);
        gen.register_frame(5, 200);

        assert_eq!(gen.registered_count(), 2);
        assert_eq!(gen.pending_count(), 2);

        let triggered = gen.request_frame(0).expect("request_frame should succeed");
        assert!(triggered);
        assert_eq!(
            gen.get_frame(0).expect("frame 0 exists").status,
            FrameStatus::Generating
        );
    }

    #[test]
    fn test_request_already_generating_is_noop() {
        let mut gen = make_gen(10);
        gen.register_frame(1, 40);
        gen.request_frame(1).expect("first request should succeed");
        let again = gen.request_frame(1).expect("second request should succeed");
        assert!(!again, "already generating — should return false");
    }

    #[test]
    fn test_mark_ready() {
        let mut gen = make_gen(10);
        gen.register_frame(2, 80);
        gen.request_frame(2).expect("request should succeed");
        gen.mark_ready(2, 4096).expect("mark_ready should succeed");

        let frame = gen.get_frame(2).expect("frame 2 should exist");
        assert!(frame.is_ready());
        assert_eq!(frame.size_bytes, Some(4096));
        assert_eq!(gen.ready_count(), 1);
    }

    #[test]
    fn test_mark_failed() {
        let mut gen = make_gen(10);
        gen.register_frame(3, 120);
        gen.request_frame(3).expect("request should succeed");
        gen.mark_failed(3).expect("mark_failed should succeed");

        let frame = gen.get_frame(3).expect("frame 3 should exist");
        assert_eq!(frame.status, FrameStatus::Failed);
    }

    #[test]
    fn test_request_unregistered_frame_errors() {
        let mut gen = make_gen(10);
        let result = gen.request_frame(99);
        assert!(result.is_err());
    }

    #[test]
    fn test_completion_ratio() {
        let mut gen = make_gen(10);
        gen.register_frame(0, 0);
        gen.register_frame(1, 40);
        gen.register_frame(2, 80);

        gen.request_frame(0).expect("ok");
        gen.mark_ready(0, 100).expect("ok");

        let ratio = gen.completion_ratio();
        assert!((ratio - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_pending_indices_sorted() {
        let mut gen = make_gen(20);
        gen.register_frame(10, 400);
        gen.register_frame(2, 80);
        gen.register_frame(7, 280);

        let pending = gen.pending_indices();
        assert_eq!(pending, vec![2, 7, 10]);
    }

    #[test]
    fn test_completion_ratio_empty() {
        let gen = make_gen(0);
        assert!((gen.completion_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ready_request_is_noop() {
        let mut gen = make_gen(5);
        gen.register_frame(0, 0);
        gen.request_frame(0).expect("ok");
        gen.mark_ready(0, 512).expect("ok");
        let triggered = gen.request_frame(0).expect("third request should succeed");
        assert!(!triggered, "already ready — should return false");
    }

    #[test]
    fn test_failed_frame_can_be_rerequested() {
        let mut gen = make_gen(5);
        gen.register_frame(0, 0);
        gen.request_frame(0).expect("ok");
        gen.mark_failed(0).expect("ok");
        // After failure, requesting again should transition back to Generating
        let triggered = gen.request_frame(0).expect("retry should succeed");
        assert!(triggered);
        assert_eq!(
            gen.get_frame(0).expect("frame exists").status,
            FrameStatus::Generating
        );
    }
}
