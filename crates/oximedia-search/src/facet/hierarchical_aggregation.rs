//! Three-level hierarchical facet aggregation for media search.
//!
//! This module extends the two-level [`crate::facet::aggregation::HierarchicalFacets`]
//! with a true **three-level** hierarchy that models the real-world taxonomy of
//! media codecs:
//!
//! ```text
//! format   (video / audio / image / other)
//!   └── codec  (mp4, webm, flac, …)
//!         └── profile  (high, main, baseline, …)
//! ```
//!
//! Profiles are inferred from the `file_path` of each result item using a set
//! of well-known codec profile keywords (e.g. `_high`, `_main`, `_baseline`
//! for H.264/AVC, `_main10` for HEVC, `_simple` for MPEG-4, etc.).  When no
//! profile keyword is detected the item is placed in the synthetic `"default"`
//! sub-bucket so that every codec always has at least one profile child.
//!
//! # Example
//!
//! ```
//! use oximedia_search::SearchResultItem;
//! use oximedia_search::facet::hierarchical_aggregation::{
//!     aggregate_three_level, ThreeLevelFacets,
//! };
//! use uuid::Uuid;
//!
//! let item = SearchResultItem {
//!     asset_id: Uuid::new_v4(),
//!     score: 1.0,
//!     title: None,
//!     description: None,
//!     file_path: "clip_high.mp4".to_string(),
//!     mime_type: Some("video/mp4".to_string()),
//!     duration_ms: None,
//!     created_at: 0,
//!     modified_at: None,
//!     file_size: None,
//!     matched_fields: vec![],
//!     thumbnail_url: None,
//! };
//! let facets = aggregate_three_level(&[item]);
//! assert_eq!(facets.format_tree.len(), 1);
//! let fmt = &facets.format_tree[0];
//! assert_eq!(fmt.label, "video");
//! assert_eq!(fmt.children.len(), 1);
//! let codec = &fmt.children[0];
//! assert_eq!(codec.label, "mp4");
//! assert_eq!(codec.children.len(), 1);
//! let profile = &codec.children[0];
//! assert_eq!(profile.label, "high");
//! ```

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Data types
// ─────────────────────────────────────────────────────────────────────────────

/// A single node in the three-level facet tree.
///
/// The tree has exactly three levels:
/// - **Level 0** (root children): format, e.g. `"video"`, `"audio"`, `"image"`
/// - **Level 1** (codec children): codec/container, e.g. `"mp4"`, `"webm"`
/// - **Level 2** (profile children): codec profile, e.g. `"high"`, `"main"`
///
/// `count` is the number of result items that passed through this node
/// (bottom-up propagation): a format node's count is the sum of all codec
/// counts under it, and a codec node's count is the sum of all profile counts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FacetNode {
    /// Human-readable label for this facet value.
    pub label: String,
    /// Number of documents associated with this node (inclusive of all
    /// descendants).
    pub count: usize,
    /// Children at the next level of the hierarchy.
    pub children: Vec<FacetNode>,
}

impl FacetNode {
    /// Create a leaf node (no children).
    #[must_use]
    pub fn leaf(label: impl Into<String>, count: usize) -> Self {
        Self {
            label: label.into(),
            count,
            children: Vec::new(),
        }
    }

    /// Create an internal node with the given children.
    ///
    /// `count` should equal the sum of all child counts.
    #[must_use]
    pub fn internal(label: impl Into<String>, count: usize, children: Vec<Self>) -> Self {
        Self {
            label: label.into(),
            count,
            children,
        }
    }

    /// Return `true` if this node is a leaf (has no children).
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Depth of the sub-tree rooted at this node (1 for a leaf).
    #[must_use]
    pub fn depth(&self) -> usize {
        if self.children.is_empty() {
            1
        } else {
            1 + self.children.iter().map(Self::depth).max().unwrap_or(0)
        }
    }

    /// Find a direct child by label (case-sensitive).
    #[must_use]
    pub fn child(&self, label: &str) -> Option<&Self> {
        self.children.iter().find(|c| c.label == label)
    }

    /// Total count accumulated from all leaf descendants.
    #[must_use]
    pub fn leaf_count(&self) -> usize {
        if self.children.is_empty() {
            self.count
        } else {
            self.children.iter().map(Self::leaf_count).sum()
        }
    }
}

/// Result of three-level hierarchical facet aggregation.
///
/// `format_tree` contains one [`FacetNode`] per format (video/audio/image/other),
/// each with codec children, each codec having profile children.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThreeLevelFacets {
    /// Three-level format → codec → profile hierarchy.
    pub format_tree: Vec<FacetNode>,
    /// Total number of items aggregated.
    pub total_items: usize,
}

impl ThreeLevelFacets {
    /// Return the top-level format node for `format`, if present.
    #[must_use]
    pub fn format(&self, format: &str) -> Option<&FacetNode> {
        self.format_tree.iter().find(|f| f.label == format)
    }

    /// Navigate to a codec node within a format.
    #[must_use]
    pub fn codec(&self, format: &str, codec: &str) -> Option<&FacetNode> {
        self.format(format)?.child(codec)
    }

    /// Navigate to a profile node within a format and codec.
    #[must_use]
    pub fn profile(&self, format: &str, codec: &str, profile: &str) -> Option<&FacetNode> {
        self.codec(format, codec)?.child(profile)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Profile detection
// ─────────────────────────────────────────────────────────────────────────────

/// Known codec profile keywords, ordered from most specific to least specific
/// so that the first match wins.
///
/// Matching is done case-insensitively against the file path.
const PROFILE_KEYWORDS: &[(&str, &str)] = &[
    // H.264 / AVC
    ("_high10", "high10"),
    ("_high422", "high422"),
    ("_high444", "high444"),
    ("-high10", "high10"),
    ("-high422", "high422"),
    ("-high444", "high444"),
    ("_high", "high"),
    ("-high", "high"),
    (".high", "high"),
    ("_main10", "main10"),
    ("-main10", "main10"),
    ("_main", "main"),
    ("-main", "main"),
    (".main", "main"),
    ("_baseline", "baseline"),
    ("-baseline", "baseline"),
    (".baseline", "baseline"),
    // HEVC / H.265
    ("_main10", "main10"),
    ("-main10", "main10"),
    ("_main", "main"),
    ("-main", "main"),
    ("_rext", "rext"),
    ("-rext", "rext"),
    // AV1
    ("_seq0", "seq0"),
    ("_seq1", "seq1"),
    ("_seq2", "seq2"),
    ("_profile0", "profile0"),
    ("_profile1", "profile1"),
    ("_profile2", "profile2"),
    // VP9
    ("_profile0", "profile0"),
    ("_profile1", "profile1"),
    ("_profile2", "profile2"),
    ("_profile3", "profile3"),
    // Generic quality
    ("_lossless", "lossless"),
    ("-lossless", "lossless"),
    ("_lossy", "lossy"),
    ("-lossy", "lossy"),
    // MPEG-4 / MPEG-2
    ("_simple", "simple"),
    ("-simple", "simple"),
    ("_advanced", "advanced"),
    ("-advanced", "advanced"),
    ("_422p", "422p"),
    ("-422p", "422p"),
    ("_444p", "444p"),
    ("-444p", "444p"),
];

/// Detect a codec profile from a file path.
///
/// Returns the first matching profile keyword, or `"default"` if none match.
#[must_use]
pub fn detect_profile(file_path: &str) -> &'static str {
    let lower = file_path.to_ascii_lowercase();
    for (keyword, profile) in PROFILE_KEYWORDS {
        if lower.contains(keyword) {
            return profile;
        }
    }
    "default"
}

// ─────────────────────────────────────────────────────────────────────────────
// Format / codec helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Derive the coarse media format from a MIME type string.
fn format_from_mime(mime: &str) -> &'static str {
    if mime.starts_with("video/") {
        "video"
    } else if mime.starts_with("audio/") {
        "audio"
    } else if mime.starts_with("image/") {
        "image"
    } else {
        "other"
    }
}

/// Extract the codec/container label from a MIME type string.
///
/// For `"video/mp4"` this returns `"mp4"`.
fn codec_from_mime(mime: &str) -> &str {
    mime.split_once('/').map(|(_, sub)| sub).unwrap_or("unknown")
}

// ─────────────────────────────────────────────────────────────────────────────
// Aggregation
// ─────────────────────────────────────────────────────────────────────────────

/// Build [`ThreeLevelFacets`] from a slice of search result items.
///
/// Each item contributes to exactly one path in the three-level hierarchy:
/// `format > codec > profile`.
///
/// Profiles are detected from the item's `file_path` using
/// [`detect_profile`].  When no profile keyword is found the item is placed
/// under the `"default"` profile bucket.
///
/// The returned tree is sorted at every level: nodes with higher counts come
/// first; ties are broken alphabetically by label.
#[must_use]
pub fn aggregate_three_level(items: &[crate::SearchResultItem]) -> ThreeLevelFacets {
    // Three-level accumulator: format -> codec -> profile -> count
    let mut acc: HashMap<&str, HashMap<String, HashMap<String, usize>>> = HashMap::new();

    for item in items {
        let mime = item.mime_type.as_deref().unwrap_or("application/octet-stream");
        let format = format_from_mime(mime);
        let codec = codec_from_mime(mime).to_string();
        let profile = detect_profile(&item.file_path).to_string();

        *acc.entry(format)
            .or_default()
            .entry(codec)
            .or_default()
            .entry(profile)
            .or_insert(0) += 1;
    }

    // Convert accumulator into sorted FacetNode trees.
    let mut format_tree: Vec<FacetNode> = acc
        .into_iter()
        .map(|(format, codecs)| {
            let mut codec_nodes: Vec<FacetNode> = codecs
                .into_iter()
                .map(|(codec, profiles)| {
                    let codec_count: usize = profiles.values().sum();
                    let mut profile_nodes: Vec<FacetNode> = profiles
                        .into_iter()
                        .map(|(profile, count)| FacetNode::leaf(profile, count))
                        .collect();
                    profile_nodes.sort_by(|a, b| {
                        b.count.cmp(&a.count).then_with(|| a.label.cmp(&b.label))
                    });
                    FacetNode::internal(codec, codec_count, profile_nodes)
                })
                .collect();
            codec_nodes.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.label.cmp(&b.label)));
            let format_count: usize = codec_nodes.iter().map(|n| n.count).sum();
            FacetNode::internal(format, format_count, codec_nodes)
        })
        .collect();

    format_tree.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.label.cmp(&b.label)));

    ThreeLevelFacets {
        format_tree,
        total_items: items.len(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Merging
// ─────────────────────────────────────────────────────────────────────────────

/// Merge two [`ThreeLevelFacets`] into one, combining counts at each level.
///
/// This is useful when results arrive from multiple shards and need to be
/// combined before presenting to the caller.
#[must_use]
pub fn merge_three_level(a: ThreeLevelFacets, b: ThreeLevelFacets) -> ThreeLevelFacets {
    // Build a combined accumulator.
    let mut acc: HashMap<String, HashMap<String, HashMap<String, usize>>> = HashMap::new();

    let collect_tree = |tree: Vec<FacetNode>, acc: &mut HashMap<String, HashMap<String, HashMap<String, usize>>>| {
        for fmt_node in tree {
            for codec_node in fmt_node.children {
                for profile_node in codec_node.children {
                    *acc.entry(fmt_node.label.clone())
                        .or_default()
                        .entry(codec_node.label.clone())
                        .or_default()
                        .entry(profile_node.label.clone())
                        .or_insert(0) += profile_node.count;
                }
            }
        }
    };

    collect_tree(a.format_tree, &mut acc);
    collect_tree(b.format_tree, &mut acc);

    let total = a.total_items + b.total_items;

    let mut format_tree: Vec<FacetNode> = acc
        .into_iter()
        .map(|(format, codecs)| {
            let mut codec_nodes: Vec<FacetNode> = codecs
                .into_iter()
                .map(|(codec, profiles)| {
                    let codec_count: usize = profiles.values().sum();
                    let mut profile_nodes: Vec<FacetNode> = profiles
                        .into_iter()
                        .map(|(profile, count)| FacetNode::leaf(profile, count))
                        .collect();
                    profile_nodes.sort_by(|a, b| {
                        b.count.cmp(&a.count).then_with(|| a.label.cmp(&b.label))
                    });
                    FacetNode::internal(codec, codec_count, profile_nodes)
                })
                .collect();
            codec_nodes.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.label.cmp(&b.label)));
            let format_count: usize = codec_nodes.iter().map(|n| n.count).sum();
            FacetNode::internal(format, format_count, codec_nodes)
        })
        .collect();

    format_tree.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.label.cmp(&b.label)));

    ThreeLevelFacets {
        format_tree,
        total_items: total,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SearchResultItem;
    use uuid::Uuid;

    fn make_item(mime: Option<&str>, file_path: &str) -> SearchResultItem {
        SearchResultItem {
            asset_id: Uuid::new_v4(),
            score: 1.0,
            title: None,
            description: None,
            file_path: file_path.to_string(),
            mime_type: mime.map(str::to_string),
            duration_ms: None,
            created_at: 0,
            modified_at: None,
            file_size: None,
            matched_fields: Vec::new(),
            thumbnail_url: None,
        }
    }

    // ── detect_profile ────────────────────────────────────────────────────

    #[test]
    fn test_detect_profile_high() {
        assert_eq!(detect_profile("clip_high.mp4"), "high");
        assert_eq!(detect_profile("video-high.mkv"), "high");
    }

    #[test]
    fn test_detect_profile_main() {
        assert_eq!(detect_profile("output_main.mp4"), "main");
        assert_eq!(detect_profile("file-main.ts"), "main");
    }

    #[test]
    fn test_detect_profile_baseline() {
        assert_eq!(detect_profile("stream_baseline.mp4"), "baseline");
        assert_eq!(detect_profile("mobile-baseline.mp4"), "baseline");
    }

    #[test]
    fn test_detect_profile_default_fallback() {
        assert_eq!(detect_profile("video.mp4"), "default");
        assert_eq!(detect_profile("untitled.mkv"), "default");
        assert_eq!(detect_profile(""), "default");
    }

    #[test]
    fn test_detect_profile_main10() {
        assert_eq!(detect_profile("hdr_main10.mkv"), "main10");
        assert_eq!(detect_profile("uhd-main10.mp4"), "main10");
    }

    #[test]
    fn test_detect_profile_lossless() {
        assert_eq!(detect_profile("master_lossless.flac"), "lossless");
        assert_eq!(detect_profile("archive-lossless.wav"), "lossless");
    }

    // ── aggregate_three_level ─────────────────────────────────────────────

    #[test]
    fn test_empty_items() {
        let facets = aggregate_three_level(&[]);
        assert!(facets.format_tree.is_empty());
        assert_eq!(facets.total_items, 0);
    }

    #[test]
    fn test_single_item_builds_three_levels() {
        let item = make_item(Some("video/mp4"), "clip_high.mp4");
        let facets = aggregate_three_level(&[item]);

        assert_eq!(facets.format_tree.len(), 1);
        let fmt = &facets.format_tree[0];
        assert_eq!(fmt.label, "video");
        assert_eq!(fmt.count, 1);

        assert_eq!(fmt.children.len(), 1);
        let codec = &fmt.children[0];
        assert_eq!(codec.label, "mp4");
        assert_eq!(codec.count, 1);

        assert_eq!(codec.children.len(), 1);
        let profile = &codec.children[0];
        assert_eq!(profile.label, "high");
        assert_eq!(profile.count, 1);
        assert!(profile.is_leaf());
    }

    #[test]
    fn test_multiple_profiles_under_same_codec() {
        let items = vec![
            make_item(Some("video/mp4"), "a_high.mp4"),
            make_item(Some("video/mp4"), "b_main.mp4"),
            make_item(Some("video/mp4"), "c_high.mp4"),
            make_item(Some("video/mp4"), "d.mp4"), // default
        ];
        let facets = aggregate_three_level(&items);
        assert_eq!(facets.total_items, 4);

        let fmt = facets.format("video").expect("video format should exist");
        assert_eq!(fmt.count, 4);

        let codec = fmt.child("mp4").expect("mp4 codec should exist");
        assert_eq!(codec.count, 4);

        let high = codec.child("high").expect("high profile should exist");
        assert_eq!(high.count, 2);

        let main = codec.child("main").expect("main profile should exist");
        assert_eq!(main.count, 1);

        let default = codec.child("default").expect("default profile should exist");
        assert_eq!(default.count, 1);
    }

    #[test]
    fn test_multiple_formats() {
        let items = vec![
            make_item(Some("video/mp4"), "clip.mp4"),
            make_item(Some("audio/flac"), "track.flac"),
            make_item(Some("image/png"), "photo.png"),
        ];
        let facets = aggregate_three_level(&items);
        assert_eq!(facets.format_tree.len(), 3);

        assert!(facets.format("video").is_some());
        assert!(facets.format("audio").is_some());
        assert!(facets.format("image").is_some());
    }

    #[test]
    fn test_format_counts_propagate_correctly() {
        let items = vec![
            make_item(Some("video/mp4"), "a_high.mp4"),
            make_item(Some("video/mp4"), "b_main.mp4"),
            make_item(Some("video/webm"), "c.webm"),
        ];
        let facets = aggregate_three_level(&items);

        let fmt = facets.format("video").expect("video should exist");
        assert_eq!(fmt.count, 3); // propagated from children

        let mp4 = fmt.child("mp4").expect("mp4 should exist");
        assert_eq!(mp4.count, 2);

        let webm = fmt.child("webm").expect("webm should exist");
        assert_eq!(webm.count, 1);
    }

    #[test]
    fn test_tree_sorted_by_count_desc() {
        let items = vec![
            make_item(Some("video/mp4"), "a.mp4"),
            make_item(Some("video/mp4"), "b.mp4"),
            make_item(Some("video/mp4"), "c.mp4"),
            make_item(Some("audio/flac"), "d.flac"),
        ];
        let facets = aggregate_three_level(&items);

        // Video (3) should come before audio (1).
        assert_eq!(facets.format_tree[0].label, "video");
        assert_eq!(facets.format_tree[1].label, "audio");
    }

    #[test]
    fn test_navigate_via_profile_accessor() {
        let items = vec![
            make_item(Some("video/mp4"), "stream_high.mp4"),
            make_item(Some("video/mp4"), "stream_baseline.mp4"),
        ];
        let facets = aggregate_three_level(&items);

        let profile = facets
            .profile("video", "mp4", "high")
            .expect("video/mp4/high should exist");
        assert_eq!(profile.count, 1);

        let baseline = facets
            .profile("video", "mp4", "baseline")
            .expect("video/mp4/baseline should exist");
        assert_eq!(baseline.count, 1);
    }

    #[test]
    fn test_facet_node_depth() {
        let profile = FacetNode::leaf("high", 3);
        assert_eq!(profile.depth(), 1);

        let codec = FacetNode::internal("mp4", 3, vec![profile]);
        assert_eq!(codec.depth(), 2);

        let fmt = FacetNode::internal("video", 3, vec![codec]);
        assert_eq!(fmt.depth(), 3);
    }

    #[test]
    fn test_leaf_count_matches_items() {
        let items: Vec<SearchResultItem> = (0..10)
            .map(|i| make_item(Some("video/mp4"), if i % 2 == 0 { "v_high.mp4" } else { "v_main.mp4" }))
            .collect();
        let facets = aggregate_three_level(&items);
        let fmt = facets.format("video").expect("video should exist");
        // leaf_count sums profile children counts (all leaves)
        assert_eq!(fmt.leaf_count(), 10);
    }

    #[test]
    fn test_merge_two_facets() {
        let items_a = vec![
            make_item(Some("video/mp4"), "a_high.mp4"),
            make_item(Some("video/mp4"), "b_high.mp4"),
        ];
        let items_b = vec![
            make_item(Some("video/mp4"), "c_main.mp4"),
            make_item(Some("audio/flac"), "d.flac"),
        ];
        let fa = aggregate_three_level(&items_a);
        let fb = aggregate_three_level(&items_b);
        let merged = merge_three_level(fa, fb);

        assert_eq!(merged.total_items, 4);

        let fmt = merged.format("video").expect("video should exist");
        assert_eq!(fmt.count, 3);

        let high = merged.profile("video", "mp4", "high").expect("high should exist");
        assert_eq!(high.count, 2);

        let main = merged.profile("video", "mp4", "main").expect("main should exist");
        assert_eq!(main.count, 1);

        assert!(merged.format("audio").is_some());
    }

    #[test]
    fn test_no_mime_type_falls_to_other() {
        let item = make_item(None, "file.bin");
        let facets = aggregate_three_level(&[item]);

        // No mime type => "application/octet-stream" => "other" format
        let other = facets.format("other").expect("other format should exist");
        assert_eq!(other.count, 1);
    }

    #[test]
    fn test_serialization_round_trip() {
        let items = vec![
            make_item(Some("video/mp4"), "clip_high.mp4"),
            make_item(Some("audio/flac"), "track_lossless.flac"),
        ];
        let facets = aggregate_three_level(&items);

        let json = serde_json::to_string(&facets).expect("should serialize");
        let restored: ThreeLevelFacets =
            serde_json::from_str(&json).expect("should deserialize");

        assert_eq!(restored.total_items, facets.total_items);
        assert_eq!(restored.format_tree.len(), facets.format_tree.len());
    }
}
