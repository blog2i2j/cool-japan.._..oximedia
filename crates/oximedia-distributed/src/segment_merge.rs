#![allow(dead_code)]
//! Distributed merge/concatenation of encoded segments after parallel encoding.
//!
//! After a video is split into segments and encoded in parallel across multiple
//! workers, the segments must be reassembled into the final output file. This
//! module provides:
//!
//! - [`SegmentManifest`] to track all segments belonging to a job.
//! - [`MergeStrategy`] for different concatenation approaches.
//! - [`SegmentMerger`] that validates, orders, and merges segments.
//! - Gap/overlap detection and configurable error handling.

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

use uuid::Uuid;

use crate::{DistributedError, Result};

// ---------------------------------------------------------------------------
// SegmentInfo
// ---------------------------------------------------------------------------

/// Metadata about a single encoded segment.
#[derive(Debug, Clone)]
pub struct SegmentInfo {
    /// Unique segment ID.
    pub id: Uuid,
    /// Job this segment belongs to.
    pub job_id: Uuid,
    /// Zero-based segment index within the job.
    pub index: u32,
    /// Total number of segments expected for this job.
    pub total_segments: u32,
    /// Start time of this segment in the source timeline (microseconds).
    pub start_time_us: i64,
    /// End time of this segment in the source timeline (microseconds).
    pub end_time_us: i64,
    /// Size of the encoded segment data in bytes.
    pub byte_size: u64,
    /// Worker that produced this segment.
    pub worker_id: Option<String>,
    /// Storage location (e.g., an object store key).
    pub storage_path: String,
    /// Whether the segment has been validated (checksum OK, decodable, etc.).
    pub validated: bool,
}

impl SegmentInfo {
    /// Duration of this segment in microseconds.
    pub fn duration_us(&self) -> i64 {
        self.end_time_us - self.start_time_us
    }
}

// ---------------------------------------------------------------------------
// MergeStrategy
// ---------------------------------------------------------------------------

/// Strategy for how segments are merged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Simple byte-level concatenation (suitable for transport streams).
    ByteConcat,
    /// Mux-level concatenation that rewrites container headers/indices.
    ContainerRemux,
    /// Use segment map (sidx) to build a fragmented output.
    FragmentedMp4,
}

impl Default for MergeStrategy {
    fn default() -> Self {
        Self::ContainerRemux
    }
}

impl fmt::Display for MergeStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ByteConcat => write!(f, "ByteConcat"),
            Self::ContainerRemux => write!(f, "ContainerRemux"),
            Self::FragmentedMp4 => write!(f, "FragmentedMp4"),
        }
    }
}

// ---------------------------------------------------------------------------
// GapPolicy
// ---------------------------------------------------------------------------

/// How to handle gaps or overlaps between adjacent segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GapPolicy {
    /// Reject the merge if any gap or overlap is detected.
    Strict,
    /// Allow gaps up to the specified tolerance (microseconds).
    AllowGaps { tolerance_us: i64 },
    /// Fill detected gaps with silence/black frames.
    FillGaps,
    /// Ignore gaps and overlaps entirely.
    Ignore,
}

impl Default for GapPolicy {
    fn default() -> Self {
        Self::AllowGaps {
            tolerance_us: 1_000, // 1ms
        }
    }
}

// ---------------------------------------------------------------------------
// MergeConfig
// ---------------------------------------------------------------------------

/// Configuration for the segment merger.
#[derive(Debug, Clone)]
pub struct MergeConfig {
    /// How to concatenate segments.
    pub strategy: MergeStrategy,
    /// How to handle gaps/overlaps.
    pub gap_policy: GapPolicy,
    /// Whether all segments must be validated before merge.
    pub require_validation: bool,
    /// Output storage path.
    pub output_path: String,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            strategy: MergeStrategy::default(),
            gap_policy: GapPolicy::default(),
            require_validation: true,
            output_path: String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// SegmentManifest
// ---------------------------------------------------------------------------

/// Tracks all segments for a given job and their readiness for merging.
#[derive(Debug, Clone)]
pub struct SegmentManifest {
    /// Job ID this manifest belongs to.
    pub job_id: Uuid,
    /// Expected total number of segments.
    pub expected_count: u32,
    /// Registered segments keyed by index.
    segments: HashMap<u32, SegmentInfo>,
}

impl SegmentManifest {
    /// Create a new manifest for the given job.
    pub fn new(job_id: Uuid, expected_count: u32) -> Self {
        Self {
            job_id,
            expected_count,
            segments: HashMap::new(),
        }
    }

    /// Register a completed segment.
    pub fn register_segment(&mut self, segment: SegmentInfo) -> Result<()> {
        if segment.job_id != self.job_id {
            return Err(DistributedError::Job(format!(
                "Segment job_id {} does not match manifest job_id {}",
                segment.job_id, self.job_id
            )));
        }
        if segment.index >= self.expected_count {
            return Err(DistributedError::Segmentation(format!(
                "Segment index {} exceeds expected count {}",
                segment.index, self.expected_count
            )));
        }
        if self.segments.contains_key(&segment.index) {
            return Err(DistributedError::Segmentation(format!(
                "Segment index {} already registered",
                segment.index
            )));
        }
        self.segments.insert(segment.index, segment);
        Ok(())
    }

    /// Check whether all expected segments have been registered.
    pub fn is_complete(&self) -> bool {
        self.segments.len() as u32 == self.expected_count
    }

    /// Return the number of registered segments.
    pub fn registered_count(&self) -> u32 {
        self.segments.len() as u32
    }

    /// Return missing segment indices.
    pub fn missing_indices(&self) -> Vec<u32> {
        (0..self.expected_count)
            .filter(|i| !self.segments.contains_key(i))
            .collect()
    }

    /// Get segments ordered by index.
    pub fn ordered_segments(&self) -> Vec<&SegmentInfo> {
        let mut indices: Vec<u32> = self.segments.keys().copied().collect();
        indices.sort();
        indices
            .iter()
            .filter_map(|i| self.segments.get(i))
            .collect()
    }

    /// Total byte size of all registered segments.
    pub fn total_byte_size(&self) -> u64 {
        self.segments.values().map(|s| s.byte_size).sum()
    }
}

// ---------------------------------------------------------------------------
// GapInfo
// ---------------------------------------------------------------------------

/// Describes a gap or overlap between two adjacent segments.
#[derive(Debug, Clone)]
pub struct GapInfo {
    /// Index of the first segment.
    pub before_index: u32,
    /// Index of the second segment.
    pub after_index: u32,
    /// Gap duration in microseconds (negative = overlap).
    pub gap_us: i64,
}

impl fmt::Display for GapInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.gap_us >= 0 {
            write!(
                f,
                "Gap of {}us between segments {} and {}",
                self.gap_us, self.before_index, self.after_index
            )
        } else {
            write!(
                f,
                "Overlap of {}us between segments {} and {}",
                -self.gap_us, self.before_index, self.after_index
            )
        }
    }
}

// ---------------------------------------------------------------------------
// MergeResult
// ---------------------------------------------------------------------------

/// Result of a successful merge operation.
#[derive(Debug, Clone)]
pub struct MergeResult {
    /// Job ID.
    pub job_id: Uuid,
    /// Output path.
    pub output_path: String,
    /// Total output size in bytes.
    pub total_bytes: u64,
    /// Number of segments merged.
    pub segment_count: u32,
    /// Total duration in microseconds.
    pub total_duration_us: i64,
    /// Detected gaps/overlaps (informational, even if policy allowed them).
    pub gaps: Vec<GapInfo>,
    /// Time taken for the merge operation.
    pub merge_duration: std::time::Duration,
    /// Strategy used.
    pub strategy: MergeStrategy,
}

// ---------------------------------------------------------------------------
// SegmentMerger
// ---------------------------------------------------------------------------

/// Validates and merges encoded segments into a final output.
pub struct SegmentMerger {
    config: MergeConfig,
}

impl SegmentMerger {
    /// Create a new merger with the given configuration.
    pub fn new(config: MergeConfig) -> Self {
        Self { config }
    }

    /// Validate a manifest and detect any gaps/overlaps.
    pub fn validate(&self, manifest: &SegmentManifest) -> Result<Vec<GapInfo>> {
        // Check completeness.
        if !manifest.is_complete() {
            let missing = manifest.missing_indices();
            return Err(DistributedError::Segmentation(format!(
                "Manifest incomplete; missing segments: {:?}",
                missing
            )));
        }

        // Check validation status.
        if self.config.require_validation {
            for seg in manifest.segments.values() {
                if !seg.validated {
                    return Err(DistributedError::Segmentation(format!(
                        "Segment {} (index {}) has not been validated",
                        seg.id, seg.index
                    )));
                }
            }
        }

        // Detect gaps/overlaps.
        let ordered = manifest.ordered_segments();
        let mut gaps = Vec::new();

        for pair in ordered.windows(2) {
            let prev = pair[0];
            let next = pair[1];
            let gap_us = next.start_time_us - prev.end_time_us;
            if gap_us != 0 {
                gaps.push(GapInfo {
                    before_index: prev.index,
                    after_index: next.index,
                    gap_us,
                });
            }
        }

        // Enforce gap policy.
        match self.config.gap_policy {
            GapPolicy::Strict => {
                if let Some(g) = gaps.first() {
                    return Err(DistributedError::Segmentation(format!(
                        "Strict gap policy violated: {g}"
                    )));
                }
            }
            GapPolicy::AllowGaps { tolerance_us } => {
                for g in &gaps {
                    if g.gap_us.unsigned_abs() > tolerance_us as u64 {
                        return Err(DistributedError::Segmentation(format!(
                            "Gap exceeds tolerance ({tolerance_us}us): {g}"
                        )));
                    }
                }
            }
            GapPolicy::FillGaps | GapPolicy::Ignore => { /* allow */ }
        }

        Ok(gaps)
    }

    /// Merge all segments in the manifest into a final output.
    ///
    /// This performs validation, then constructs a [`MergeResult`] describing
    /// the merged output. In a real system the actual I/O would happen here;
    /// this implementation computes the metadata.
    pub fn merge(&self, manifest: &SegmentManifest) -> Result<MergeResult> {
        let start = Instant::now();
        let gaps = self.validate(manifest)?;

        let ordered = manifest.ordered_segments();
        let total_bytes: u64 = ordered.iter().map(|s| s.byte_size).sum();
        let total_duration_us = ordered.last().map(|s| s.end_time_us).unwrap_or(0)
            - ordered.first().map(|s| s.start_time_us).unwrap_or(0);

        Ok(MergeResult {
            job_id: manifest.job_id,
            output_path: self.config.output_path.clone(),
            total_bytes,
            segment_count: ordered.len() as u32,
            total_duration_us,
            gaps,
            merge_duration: start.elapsed(),
            strategy: self.config.strategy,
        })
    }

    /// Get the merge configuration.
    pub fn config(&self) -> &MergeConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Helper: create test segment
// ---------------------------------------------------------------------------

fn make_test_segment(
    job_id: Uuid,
    index: u32,
    total: u32,
    start_us: i64,
    end_us: i64,
) -> SegmentInfo {
    SegmentInfo {
        id: Uuid::new_v4(),
        job_id,
        index,
        total_segments: total,
        start_time_us: start_us,
        end_time_us: end_us,
        byte_size: 1024 * (index as u64 + 1),
        worker_id: Some(format!("worker-{index}")),
        storage_path: format!("/tmp/seg_{index}.ts"),
        validated: true,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_manifest(count: u32, gap_us: i64) -> SegmentManifest {
        let job_id = Uuid::new_v4();
        let mut manifest = SegmentManifest::new(job_id, count);
        let segment_len_us: i64 = 5_000_000; // 5 seconds

        for i in 0..count {
            let start = i as i64 * segment_len_us + i as i64 * gap_us;
            let end = start + segment_len_us;
            let seg = make_test_segment(job_id, i, count, start, end);
            manifest.register_segment(seg).expect("register ok");
        }
        manifest
    }

    #[test]
    fn test_manifest_creation_and_completeness() {
        let job_id = Uuid::new_v4();
        let manifest = SegmentManifest::new(job_id, 4);
        assert!(!manifest.is_complete());
        assert_eq!(manifest.missing_indices(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_manifest_register_and_complete() {
        let manifest = setup_manifest(3, 0);
        assert!(manifest.is_complete());
        assert_eq!(manifest.registered_count(), 3);
        assert!(manifest.missing_indices().is_empty());
    }

    #[test]
    fn test_manifest_rejects_wrong_job_id() {
        let job_id = Uuid::new_v4();
        let mut manifest = SegmentManifest::new(job_id, 2);
        let mut seg = make_test_segment(job_id, 0, 2, 0, 5_000_000);
        seg.job_id = Uuid::new_v4(); // wrong job
        let result = manifest.register_segment(seg);
        assert!(result.is_err());
    }

    #[test]
    fn test_manifest_rejects_duplicate_index() {
        let job_id = Uuid::new_v4();
        let mut manifest = SegmentManifest::new(job_id, 2);
        let seg1 = make_test_segment(job_id, 0, 2, 0, 5_000_000);
        let seg2 = make_test_segment(job_id, 0, 2, 0, 5_000_000);
        manifest.register_segment(seg1).expect("register ok");
        let result = manifest.register_segment(seg2);
        assert!(result.is_err());
    }

    #[test]
    fn test_manifest_rejects_out_of_range_index() {
        let job_id = Uuid::new_v4();
        let mut manifest = SegmentManifest::new(job_id, 2);
        let seg = make_test_segment(job_id, 5, 2, 0, 5_000_000);
        let result = manifest.register_segment(seg);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_no_gaps() {
        let manifest = setup_manifest(4, 0);
        let merger = SegmentMerger::new(MergeConfig {
            output_path: "/tmp/output.mp4".to_string(),
            ..MergeConfig::default()
        });
        let result = merger.merge(&manifest).expect("merge ok");
        assert_eq!(result.segment_count, 4);
        assert!(result.gaps.is_empty());
        assert!(result.total_bytes > 0);
        assert!(result.total_duration_us > 0);
    }

    #[test]
    fn test_merge_small_gap_within_tolerance() {
        // Default tolerance is 1000us; set gap to 500us
        let manifest = setup_manifest(3, 500);
        let merger = SegmentMerger::new(MergeConfig {
            output_path: "/tmp/output.mp4".to_string(),
            gap_policy: GapPolicy::AllowGaps { tolerance_us: 1000 },
            ..MergeConfig::default()
        });
        let result = merger.merge(&manifest).expect("merge ok");
        assert_eq!(result.gaps.len(), 2); // gaps between seg 0-1 and 1-2
    }

    #[test]
    fn test_merge_strict_rejects_gap() {
        let manifest = setup_manifest(2, 100);
        let merger = SegmentMerger::new(MergeConfig {
            gap_policy: GapPolicy::Strict,
            output_path: "/tmp/output.mp4".to_string(),
            ..MergeConfig::default()
        });
        let result = merger.merge(&manifest);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_incomplete_manifest_fails() {
        let job_id = Uuid::new_v4();
        let mut manifest = SegmentManifest::new(job_id, 3);
        let seg = make_test_segment(job_id, 0, 3, 0, 5_000_000);
        manifest.register_segment(seg).expect("register ok");

        let merger = SegmentMerger::new(MergeConfig::default());
        let result = merger.merge(&manifest);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_rejects_unvalidated_segment() {
        let job_id = Uuid::new_v4();
        let mut manifest = SegmentManifest::new(job_id, 1);
        let mut seg = make_test_segment(job_id, 0, 1, 0, 5_000_000);
        seg.validated = false;
        manifest.register_segment(seg).expect("register ok");

        let merger = SegmentMerger::new(MergeConfig {
            require_validation: true,
            output_path: "/tmp/out.mp4".to_string(),
            ..MergeConfig::default()
        });
        let result = merger.merge(&manifest);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_ignores_gaps_with_ignore_policy() {
        let manifest = setup_manifest(2, 999_999);
        let merger = SegmentMerger::new(MergeConfig {
            gap_policy: GapPolicy::Ignore,
            output_path: "/tmp/output.mp4".to_string(),
            ..MergeConfig::default()
        });
        let result = merger.merge(&manifest).expect("merge ok");
        assert_eq!(result.gaps.len(), 1);
    }

    #[test]
    fn test_total_byte_size() {
        let manifest = setup_manifest(3, 0);
        // byte_size = 1024*(index+1): 1024 + 2048 + 3072 = 6144
        assert_eq!(manifest.total_byte_size(), 6144);
    }

    #[test]
    fn test_ordered_segments_returns_sorted() {
        let job_id = Uuid::new_v4();
        let mut manifest = SegmentManifest::new(job_id, 3);
        // Insert in reverse order
        for i in (0..3).rev() {
            let seg = make_test_segment(job_id, i, 3, i as i64 * 1000, (i as i64 + 1) * 1000);
            manifest.register_segment(seg).expect("register ok");
        }
        let ordered = manifest.ordered_segments();
        for (i, seg) in ordered.iter().enumerate() {
            assert_eq!(seg.index, i as u32);
        }
    }
}
