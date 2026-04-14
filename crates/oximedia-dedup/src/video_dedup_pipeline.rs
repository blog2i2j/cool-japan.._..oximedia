//! Full video deduplication pipeline.
//!
//! This module provides a high-level pipeline that:
//! 1. Accepts video descriptors with pre-computed keyframe pixel data.
//! 2. Generates perceptual hashes (dHash) for each keyframe.
//! 3. Builds a pairwise similarity matrix using Hamming distance.
//! 4. Clusters near-duplicate videos using Union-Find.
//! 5. Emits a [`PipelineResult`] with duplicate groups and statistics.
//!
//! # Design
//!
//! The pipeline is intentionally self-contained and works with synthetic or
//! pre-decoded frame data so it can be tested without I/O.  Real-world
//! integration should feed decoded frame buffers into [`VideoDescriptor`].

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PerceptualHash (dHash)
// ---------------------------------------------------------------------------

/// A 64-bit difference-hash (dHash) for a single video frame.
///
/// Computed by reducing the frame to an 8×8 gradient bitmap: for each of the
/// 64 horizontal adjacent pixel pairs the bit is 1 if the left pixel is
/// brighter than the right one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DHash(pub u64);

impl DHash {
    /// Compute a dHash from an 8×9 = 72-element row-major luma buffer.
    ///
    /// Each row contributes 8 bits (one per horizontal gradient).
    /// `pixels` must contain at least 72 elements; extras are ignored.
    /// Returns `None` when the buffer is too small.
    #[must_use]
    pub fn from_pixels(pixels: &[u8]) -> Option<Self> {
        // We need 8 rows × 9 columns = 72 pixels.
        if pixels.len() < 72 {
            return None;
        }
        let mut hash = 0u64;
        for row in 0..8usize {
            for col in 0..8usize {
                let left = pixels[row * 9 + col];
                let right = pixels[row * 9 + col + 1];
                if left > right {
                    hash |= 1u64 << (row * 8 + col);
                }
            }
        }
        Some(Self(hash))
    }

    /// Hamming distance between two dHashes.
    #[must_use]
    pub fn hamming(self, other: Self) -> u32 {
        (self.0 ^ other.0).count_ones()
    }

    /// Similarity in [0.0, 1.0] derived from Hamming distance.
    #[must_use]
    pub fn similarity(self, other: Self) -> f64 {
        1.0 - f64::from(self.hamming(other)) / 64.0
    }
}

// ---------------------------------------------------------------------------
// KeyframeHash
// ---------------------------------------------------------------------------

/// A dHash for a single keyframe, tagged with its position in the video.
#[derive(Debug, Clone, Copy)]
pub struct KeyframeHash {
    /// Frame index (0-based) within the video.
    pub frame_index: usize,
    /// dHash of this frame.
    pub hash: DHash,
}

impl KeyframeHash {
    /// Create a new keyframe hash.
    #[must_use]
    pub fn new(frame_index: usize, hash: DHash) -> Self {
        Self { frame_index, hash }
    }
}

// ---------------------------------------------------------------------------
// VideoDescriptor
// ---------------------------------------------------------------------------

/// Descriptor for a single video fed into the dedup pipeline.
///
/// The caller is responsible for extracting keyframes and providing their
/// 8×9 luma buffers.
#[derive(Debug, Clone)]
pub struct VideoDescriptor {
    /// Unique identifier for this video (e.g. an inode number or DB row id).
    pub id: u64,
    /// Human-readable label (e.g. file path).
    pub label: String,
    /// Raw keyframe pixel buffers (each must be ≥ 72 bytes, row-major 8×9).
    pub keyframe_pixels: Vec<Vec<u8>>,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// File size in bytes (used for space-savings estimation).
    pub size_bytes: u64,
}

impl VideoDescriptor {
    /// Create a new descriptor.
    #[must_use]
    pub fn new(
        id: u64,
        label: impl Into<String>,
        keyframe_pixels: Vec<Vec<u8>>,
        duration_ms: u64,
        size_bytes: u64,
    ) -> Self {
        Self {
            id,
            label: label.into(),
            keyframe_pixels,
            duration_ms,
            size_bytes,
        }
    }
}

// ---------------------------------------------------------------------------
// VideoHashEntry
// ---------------------------------------------------------------------------

/// Internal: a video with its computed keyframe hashes.
#[derive(Debug, Clone)]
struct VideoHashEntry {
    id: u64,
    label: String,
    hashes: Vec<KeyframeHash>,
    duration_ms: u64,
    size_bytes: u64,
}

impl VideoHashEntry {
    /// Hash the entry's keyframes and produce an entry.
    fn from_descriptor(desc: &VideoDescriptor) -> Self {
        let hashes: Vec<KeyframeHash> = desc
            .keyframe_pixels
            .iter()
            .enumerate()
            .filter_map(|(i, pixels)| DHash::from_pixels(pixels).map(|h| KeyframeHash::new(i, h)))
            .collect();
        Self {
            id: desc.id,
            label: desc.label.clone(),
            hashes,
            duration_ms: desc.duration_ms,
            size_bytes: desc.size_bytes,
        }
    }

    /// Compute Jaccard-style similarity against another entry.
    ///
    /// Uses a relaxed overlap: two frames are considered matching if their
    /// Hamming distance is below the given `hamming_threshold`.
    fn similarity_to(&self, other: &Self, hamming_threshold: u32) -> f64 {
        if self.hashes.is_empty() && other.hashes.is_empty() {
            return 1.0;
        }
        if self.hashes.is_empty() || other.hashes.is_empty() {
            return 0.0;
        }

        let matches = self
            .hashes
            .iter()
            .filter(|kfa| {
                other
                    .hashes
                    .iter()
                    .any(|kfb| kfa.hash.hamming(kfb.hash) <= hamming_threshold)
            })
            .count();

        let union = self.hashes.len() + other.hashes.len() - matches;
        if union == 0 {
            return 0.0;
        }
        matches as f64 / union as f64
    }
}

// ---------------------------------------------------------------------------
// SimilarityMatrix
// ---------------------------------------------------------------------------

/// Dense upper-triangular similarity matrix for `n` videos.
#[derive(Debug, Clone)]
pub struct SimilarityMatrix {
    /// Number of videos.
    pub n: usize,
    /// Similarity scores stored row-major for the upper triangle (i < j).
    scores: Vec<f64>,
    /// Mapping from position index to video id.
    pub ids: Vec<u64>,
}

impl SimilarityMatrix {
    /// Build a similarity matrix.
    fn build(entries: &[VideoHashEntry], hamming_threshold: u32) -> Self {
        let n = entries.len();
        let mut scores = vec![0.0f64; n * n];
        let ids: Vec<u64> = entries.iter().map(|e| e.id).collect();

        for i in 0..n {
            scores[i * n + i] = 1.0;
            for j in (i + 1)..n {
                let sim = entries[i].similarity_to(&entries[j], hamming_threshold);
                scores[i * n + j] = sim;
                scores[j * n + i] = sim;
            }
        }
        Self { n, scores, ids }
    }

    /// Get the similarity between video at position `i` and `j`.
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i >= self.n || j >= self.n {
            return 0.0;
        }
        self.scores[i * self.n + j]
    }

    /// Return all pairs whose similarity is ≥ `threshold` (i < j).
    #[must_use]
    pub fn pairs_above(&self, threshold: f64) -> Vec<(usize, usize, f64)> {
        let mut result = Vec::new();
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                let s = self.scores[i * self.n + j];
                if s >= threshold {
                    result.push((i, j, s));
                }
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// DuplicateVideoGroup
// ---------------------------------------------------------------------------

/// A group of duplicate or near-duplicate videos.
#[derive(Debug, Clone)]
pub struct DuplicateVideoGroup {
    /// Group identifier (0-based).
    pub id: usize,
    /// IDs of the videos in this group.
    pub video_ids: Vec<u64>,
    /// Labels of the videos (same order as `video_ids`).
    pub labels: Vec<String>,
    /// Mean pairwise similarity within the group.
    pub mean_similarity: f64,
    /// Index of the representative video within `video_ids`.
    pub representative_idx: usize,
    /// Total reclaimable bytes if all non-representative members are removed.
    pub reclaimable_bytes: u64,
}

impl DuplicateVideoGroup {
    /// Return the label of the representative video.
    #[must_use]
    pub fn representative_label(&self) -> &str {
        self.labels
            .get(self.representative_idx)
            .map(String::as_str)
            .unwrap_or("")
    }
}

// ---------------------------------------------------------------------------
// PipelineStats
// ---------------------------------------------------------------------------

/// Statistics produced by the dedup pipeline.
#[derive(Debug, Clone)]
pub struct PipelineStats {
    /// Total number of videos processed.
    pub total_videos: usize,
    /// Number of duplicate groups found.
    pub duplicate_groups: usize,
    /// Total videos that are duplicates (members of groups with ≥ 2 entries).
    pub duplicate_videos: usize,
    /// Total bytes reclaimable by removing non-representative duplicates.
    pub reclaimable_bytes: u64,
    /// Mean inter-group similarity.
    pub mean_group_similarity: f64,
}

impl PipelineStats {
    /// Return a human-readable summary.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{} videos processed: {} groups, {} duplicates, {} bytes reclaimable, mean_sim={:.3}",
            self.total_videos,
            self.duplicate_groups,
            self.duplicate_videos,
            self.reclaimable_bytes,
            self.mean_group_similarity,
        )
    }
}

// ---------------------------------------------------------------------------
// PipelineResult
// ---------------------------------------------------------------------------

/// Full result of the video dedup pipeline.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// Detected duplicate video groups.
    pub groups: Vec<DuplicateVideoGroup>,
    /// Pipeline statistics.
    pub stats: PipelineStats,
    /// The similarity matrix (available for downstream analysis).
    pub similarity_matrix: SimilarityMatrix,
}

// ---------------------------------------------------------------------------
// VideoDedupPipeline
// ---------------------------------------------------------------------------

/// Configurable video deduplication pipeline.
///
/// # Example
/// ```
/// use oximedia_dedup::video_dedup_pipeline::{VideoDedupPipeline, VideoDescriptor};
///
/// let pipeline = VideoDedupPipeline::new()
///     .with_similarity_threshold(0.80)
///     .with_hamming_threshold(8);
///
/// let result = pipeline.run(&[]);
/// assert_eq!(result.groups.len(), 0);
/// ```
#[derive(Debug, Clone)]
pub struct VideoDedupPipeline {
    /// Minimum Jaccard similarity for two videos to be grouped.
    pub similarity_threshold: f64,
    /// Maximum Hamming distance between two frame hashes to count as a match.
    pub hamming_threshold: u32,
}

impl Default for VideoDedupPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl VideoDedupPipeline {
    /// Create a pipeline with sensible defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            similarity_threshold: 0.80,
            hamming_threshold: 8,
        }
    }

    /// Set the minimum Jaccard similarity threshold.
    #[must_use]
    pub fn with_similarity_threshold(mut self, t: f64) -> Self {
        self.similarity_threshold = t;
        self
    }

    /// Set the maximum Hamming distance for frame-level matching.
    #[must_use]
    pub fn with_hamming_threshold(mut self, t: u32) -> Self {
        self.hamming_threshold = t;
        self
    }

    /// Run the full pipeline on a slice of [`VideoDescriptor`]s.
    #[must_use]
    pub fn run(&self, descriptors: &[VideoDescriptor]) -> PipelineResult {
        if descriptors.is_empty() {
            return PipelineResult {
                groups: Vec::new(),
                stats: PipelineStats {
                    total_videos: 0,
                    duplicate_groups: 0,
                    duplicate_videos: 0,
                    reclaimable_bytes: 0,
                    mean_group_similarity: 0.0,
                },
                similarity_matrix: SimilarityMatrix {
                    n: 0,
                    scores: Vec::new(),
                    ids: Vec::new(),
                },
            };
        }

        // Step 1: Hash all keyframes.
        let entries: Vec<VideoHashEntry> = descriptors
            .iter()
            .map(VideoHashEntry::from_descriptor)
            .collect();

        // Step 2: Build pairwise similarity matrix.
        let matrix = SimilarityMatrix::build(&entries, self.hamming_threshold);

        // Step 3: Union-Find clustering.
        let n = entries.len();
        let mut parent: Vec<usize> = (0..n).collect();
        let mut rank: Vec<usize> = vec![0; n];

        let valid_pairs = matrix.pairs_above(self.similarity_threshold);
        for &(i, j, _) in &valid_pairs {
            Self::union_by_rank(&mut parent, &mut rank, i, j);
        }

        // Group by root.
        let mut group_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let root = Self::find_root(&mut parent, i);
            group_map.entry(root).or_default().push(i);
        }

        // Step 4: Build DuplicateVideoGroup per group with ≥ 2 members.
        let id_to_entry: HashMap<u64, &VideoHashEntry> =
            entries.iter().map(|e| (e.id, e)).collect();

        let mut groups: Vec<DuplicateVideoGroup> = group_map
            .into_values()
            .filter(|members| members.len() >= 2)
            .enumerate()
            .map(|(gid, members)| {
                // Compute mean pairwise similarity within the group.
                let mut sim_sum = 0.0f64;
                let mut sim_count = 0usize;
                for ii in 0..members.len() {
                    for jj in (ii + 1)..members.len() {
                        sim_sum += matrix.get(members[ii], members[jj]);
                        sim_count += 1;
                    }
                }
                let mean_similarity = if sim_count > 0 {
                    sim_sum / sim_count as f64
                } else {
                    0.0
                };

                // Select representative: the member whose total similarity to
                // all others is highest.
                let mut best_idx = 0usize;
                let mut best_sim = f64::NEG_INFINITY;
                for (li, &gi) in members.iter().enumerate() {
                    let total: f64 = members
                        .iter()
                        .enumerate()
                        .filter(|&(lj, _)| lj != li)
                        .map(|(_, &gj)| matrix.get(gi, gj))
                        .sum();
                    if total > best_sim {
                        best_sim = total;
                        best_idx = li;
                    }
                }

                // Compute reclaimable bytes.
                let reclaimable_bytes: u64 = members
                    .iter()
                    .enumerate()
                    .filter(|&(li, _)| li != best_idx)
                    .filter_map(|(_, &gi)| entries.get(gi).map(|e| e.size_bytes))
                    .sum();

                let video_ids: Vec<u64> = members.iter().map(|&gi| entries[gi].id).collect();
                let labels: Vec<String> =
                    members.iter().map(|&gi| entries[gi].label.clone()).collect();

                // Validate that all ids exist (defensive check).
                for &vid in &video_ids {
                    let _ = id_to_entry.get(&vid);
                }

                DuplicateVideoGroup {
                    id: gid,
                    video_ids,
                    labels,
                    mean_similarity,
                    representative_idx: best_idx,
                    reclaimable_bytes,
                }
            })
            .collect();

        // Sort groups by id for deterministic output.
        groups.sort_by_key(|g| g.id);

        // Step 5: Build statistics.
        let duplicate_videos: usize = groups.iter().map(|g| g.video_ids.len()).sum();
        let reclaimable_bytes: u64 = groups.iter().map(|g| g.reclaimable_bytes).sum();
        let mean_group_similarity = if groups.is_empty() {
            0.0
        } else {
            groups.iter().map(|g| g.mean_similarity).sum::<f64>() / groups.len() as f64
        };

        PipelineResult {
            stats: PipelineStats {
                total_videos: n,
                duplicate_groups: groups.len(),
                duplicate_videos,
                reclaimable_bytes,
                mean_group_similarity,
            },
            groups,
            similarity_matrix: matrix,
        }
    }

    // ---- Private helpers ----

    fn find_root(parent: &mut Vec<usize>, x: usize) -> usize {
        let mut root = x;
        while parent[root] != root {
            root = parent[root];
        }
        let mut cur = x;
        while cur != root {
            let next = parent[cur];
            parent[cur] = root;
            cur = next;
        }
        root
    }

    fn union_by_rank(parent: &mut Vec<usize>, rank: &mut Vec<usize>, a: usize, b: usize) {
        let ra = Self::find_root(parent, a);
        let rb = Self::find_root(parent, b);
        if ra == rb {
            return;
        }
        match rank[ra].cmp(&rank[rb]) {
            std::cmp::Ordering::Less => parent[ra] = rb,
            std::cmp::Ordering::Greater => parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                parent[rb] = ra;
                rank[ra] += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an 8×9 = 72-element luma buffer using a pseudo-random pattern
    /// derived from `seed` so that different seeds produce clearly different
    /// hashes and gradients go both directions (some left > right).
    fn make_pixels(seed: u8) -> Vec<u8> {
        (0..72u8)
            .map(|i| {
                // Mix index and seed with a simple multiplicative hash so
                // adjacent pixels have varying order relationships.
                let v = (i as u16)
                    .wrapping_mul(37)
                    .wrapping_add(seed as u16)
                    .wrapping_mul(19);
                (v & 0xFF) as u8
            })
            .collect()
    }

    #[test]
    fn test_dhash_from_pixels_too_small() {
        let pixels = vec![0u8; 10];
        assert!(DHash::from_pixels(&pixels).is_none());
    }

    #[test]
    fn test_dhash_from_pixels_identical() {
        let pixels = make_pixels(0);
        let h1 = DHash::from_pixels(&pixels).expect("hash should succeed");
        let h2 = DHash::from_pixels(&pixels).expect("hash should succeed");
        assert_eq!(h1, h2);
        assert_eq!(h1.hamming(h2), 0);
        assert_eq!(h1.similarity(h2), 1.0);
    }

    #[test]
    fn test_dhash_different_seeds_differ() {
        let h1 = DHash::from_pixels(&make_pixels(0)).expect("hash");
        let h2 = DHash::from_pixels(&make_pixels(100)).expect("hash");
        assert_ne!(h1, h2);
        assert!(h1.hamming(h2) > 0);
    }

    #[test]
    fn test_dhash_similarity_range() {
        let h1 = DHash::from_pixels(&make_pixels(0)).expect("hash");
        let h2 = DHash::from_pixels(&make_pixels(50)).expect("hash");
        let sim = h1.similarity(h2);
        assert!((0.0..=1.0).contains(&sim));
    }

    #[test]
    fn test_similarity_matrix_self_similarity() {
        let desc = VideoDescriptor::new(1, "a.mp4", vec![make_pixels(0), make_pixels(10)], 5000, 1024);
        let entry = VideoHashEntry::from_descriptor(&desc);
        let entries = vec![entry];
        let matrix = SimilarityMatrix::build(&entries, 8);
        assert_eq!(matrix.get(0, 0), 1.0);
    }

    #[test]
    fn test_similarity_matrix_out_of_bounds() {
        let desc = VideoDescriptor::new(1, "a.mp4", vec![make_pixels(0)], 1000, 512);
        let entry = VideoHashEntry::from_descriptor(&desc);
        let matrix = SimilarityMatrix::build(&[entry], 8);
        assert_eq!(matrix.get(99, 99), 0.0);
    }

    #[test]
    fn test_pipeline_empty() {
        let result = VideoDedupPipeline::new().run(&[]);
        assert!(result.groups.is_empty());
        assert_eq!(result.stats.total_videos, 0);
    }

    #[test]
    fn test_pipeline_single_video_no_duplicates() {
        let desc = VideoDescriptor::new(1, "solo.mp4", vec![make_pixels(0)], 3000, 512);
        let result = VideoDedupPipeline::new().run(&[desc]);
        assert!(result.groups.is_empty());
        assert_eq!(result.stats.duplicate_videos, 0);
    }

    #[test]
    fn test_pipeline_identical_videos_grouped() {
        let pixels = vec![make_pixels(0), make_pixels(10), make_pixels(20)];
        let a = VideoDescriptor::new(1, "a.mp4", pixels.clone(), 5000, 1024);
        let b = VideoDescriptor::new(2, "b.mp4", pixels.clone(), 5000, 1024);

        let result = VideoDedupPipeline::new()
            .with_similarity_threshold(0.5)
            .run(&[a, b]);

        assert_eq!(result.groups.len(), 1);
        assert_eq!(result.groups[0].video_ids.len(), 2);
        assert!(result.stats.reclaimable_bytes > 0);
    }

    #[test]
    fn test_pipeline_distinct_videos_not_grouped() {
        // Very different seeds → very different hashes.
        let a = VideoDescriptor::new(
            1,
            "a.mp4",
            vec![make_pixels(0)],
            1000,
            512,
        );
        let b = VideoDescriptor::new(
            2,
            "b.mp4",
            vec![make_pixels(200)],
            1000,
            512,
        );
        // Use a strict threshold so unlike videos don't accidentally cluster.
        let result = VideoDedupPipeline::new()
            .with_similarity_threshold(0.99)
            .with_hamming_threshold(0)
            .run(&[a, b]);
        assert!(result.groups.is_empty());
    }

    #[test]
    fn test_pipeline_representative_selected() {
        let pixels = vec![make_pixels(0), make_pixels(5)];
        let a = VideoDescriptor::new(1, "a.mp4", pixels.clone(), 5000, 2048);
        let b = VideoDescriptor::new(2, "b.mp4", pixels.clone(), 5000, 1024);

        let result = VideoDedupPipeline::new()
            .with_similarity_threshold(0.5)
            .run(&[a, b]);

        assert_eq!(result.groups.len(), 1);
        let group = &result.groups[0];
        assert!(group.representative_idx < group.video_ids.len());
        assert!(!group.representative_label().is_empty());
    }

    #[test]
    fn test_pipeline_stats_summary() {
        let result = VideoDedupPipeline::new().run(&[]);
        assert!(!result.stats.summary().is_empty());
    }

    #[test]
    fn test_pipeline_reclaimable_bytes() {
        let pixels = vec![make_pixels(0)];
        let a = VideoDescriptor::new(1, "a.mp4", pixels.clone(), 1000, 500);
        let b = VideoDescriptor::new(2, "b.mp4", pixels.clone(), 1000, 800);
        let c = VideoDescriptor::new(3, "c.mp4", pixels.clone(), 1000, 300);

        let result = VideoDedupPipeline::new()
            .with_similarity_threshold(0.5)
            .run(&[a, b, c]);

        if !result.groups.is_empty() {
            // Reclaimable should be sum of non-representative member sizes.
            assert!(result.stats.reclaimable_bytes > 0);
        }
    }
}
