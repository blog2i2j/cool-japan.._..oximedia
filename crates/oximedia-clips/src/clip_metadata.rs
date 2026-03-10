//! Extended clip metadata attributes for `OxiMedia`.
//!
//! Provides a typed attribute system for clip metadata, supporting both
//! technical properties (codec, resolution) and editorial properties
//! (description, tags). Multiple `ClipMetadata` sets can be merged or diffed.

#![allow(dead_code)]

use std::collections::HashMap;

/// A typed attribute that can be attached to a clip.
#[derive(Debug, Clone, PartialEq)]
pub enum ClipAttribute {
    /// Codec name (e.g. "h264", "prores").
    Codec(String),
    /// Frame rate as a rational number (numerator, denominator).
    FrameRate(u32, u32),
    /// Pixel resolution (width, height).
    Resolution(u32, u32),
    /// Bit depth (e.g. 8, 10, 12).
    BitDepth(u8),
    /// Free-form textual description.
    Description(String),
    /// A single tag/keyword.
    Tag(String),
    /// Numeric rating (0–100).
    Rating(u8),
    /// Whether the clip has been flagged for review.
    ReviewFlag(bool),
}

impl ClipAttribute {
    /// Returns `true` if this attribute describes a technical property
    /// (codec, frame rate, resolution, bit depth).
    #[must_use]
    pub fn is_technical(&self) -> bool {
        matches!(
            self,
            Self::Codec(_) | Self::FrameRate(_, _) | Self::Resolution(_, _) | Self::BitDepth(_)
        )
    }

    /// Returns a string key identifying the attribute kind.
    #[must_use]
    pub fn kind_key(&self) -> &'static str {
        match self {
            Self::Codec(_) => "codec",
            Self::FrameRate(_, _) => "frame_rate",
            Self::Resolution(_, _) => "resolution",
            Self::BitDepth(_) => "bit_depth",
            Self::Description(_) => "description",
            Self::Tag(_) => "tag",
            Self::Rating(_) => "rating",
            Self::ReviewFlag(_) => "review_flag",
        }
    }
}

/// Metadata collection for a single clip.
#[derive(Debug, Clone, Default)]
pub struct ClipMetadata {
    /// Clip identifier.
    pub clip_id: u64,
    /// Named attributes.
    attributes: HashMap<String, ClipAttribute>,
}

impl ClipMetadata {
    /// Create a new, empty metadata set for `clip_id`.
    #[must_use]
    pub fn new(clip_id: u64) -> Self {
        Self {
            clip_id,
            attributes: HashMap::new(),
        }
    }

    /// Insert or replace an attribute, using its `kind_key()` as the map key.
    pub fn set_attribute(&mut self, attr: ClipAttribute) {
        let key = attr.kind_key().to_string();
        self.attributes.insert(key, attr);
    }

    /// Returns `true` if an attribute with this kind key is present.
    #[must_use]
    pub fn has_attribute(&self, kind: &str) -> bool {
        self.attributes.contains_key(kind)
    }

    /// Retrieve an attribute by kind key.
    #[must_use]
    pub fn get_attribute(&self, kind: &str) -> Option<&ClipAttribute> {
        self.attributes.get(kind)
    }

    /// Returns the number of attributes.
    #[must_use]
    pub fn attribute_count(&self) -> usize {
        self.attributes.len()
    }

    /// Returns an iterator over all (key, attribute) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ClipAttribute)> {
        self.attributes.iter().map(|(k, v)| (k.as_str(), v))
    }
}

/// A named collection of `ClipMetadata` entries.
#[derive(Debug, Clone, Default)]
pub struct ClipMetadataSet {
    entries: HashMap<u64, ClipMetadata>,
}

impl ClipMetadataSet {
    /// Create a new empty set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace the metadata for a clip.
    pub fn insert(&mut self, meta: ClipMetadata) {
        self.entries.insert(meta.clip_id, meta);
    }

    /// Retrieve metadata for a clip.
    #[must_use]
    pub fn get(&self, clip_id: u64) -> Option<&ClipMetadata> {
        self.entries.get(&clip_id)
    }

    /// Merge `other` into `self`. Attributes in `other` overwrite those in `self`
    /// for the same clip and kind key.
    pub fn merge(&mut self, other: &Self) {
        for (clip_id, other_meta) in &other.entries {
            let entry = self
                .entries
                .entry(*clip_id)
                .or_insert_with(|| ClipMetadata::new(*clip_id));
            for (_, attr) in other_meta.iter() {
                entry.set_attribute(attr.clone());
            }
        }
    }

    /// Returns a list of (clip_id, kind_key) pairs where `self` and `other` differ.
    /// Includes attributes present in one but not the other.
    #[must_use]
    pub fn diff(&self, other: &Self) -> Vec<(u64, String)> {
        let mut diffs = Vec::new();

        // Check all keys in self
        for (clip_id, meta) in &self.entries {
            let other_meta = other.entries.get(clip_id);
            for (key, attr) in meta.iter() {
                let matches = other_meta
                    .and_then(|m| m.get_attribute(key))
                    .map_or(false, |a| a == attr);
                if !matches {
                    diffs.push((*clip_id, key.to_string()));
                }
            }
        }

        // Check keys in other not in self
        for (clip_id, meta) in &other.entries {
            let self_meta = self.entries.get(clip_id);
            for (key, _) in meta.iter() {
                let in_self = self_meta.map_or(false, |m| m.has_attribute(key));
                if !in_self {
                    let entry = (*clip_id, key.to_string());
                    if !diffs.contains(&entry) {
                        diffs.push(entry);
                    }
                }
            }
        }

        diffs
    }

    /// Total number of clips in this set.
    #[must_use]
    pub fn clip_count(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_is_technical() {
        assert!(ClipAttribute::Codec("h264".into()).is_technical());
    }

    #[test]
    fn test_frame_rate_is_technical() {
        assert!(ClipAttribute::FrameRate(25, 1).is_technical());
    }

    #[test]
    fn test_description_not_technical() {
        assert!(!ClipAttribute::Description("nice shot".into()).is_technical());
    }

    #[test]
    fn test_tag_not_technical() {
        assert!(!ClipAttribute::Tag("hero".into()).is_technical());
    }

    #[test]
    fn test_rating_not_technical() {
        assert!(!ClipAttribute::Rating(80).is_technical());
    }

    #[test]
    fn test_clip_metadata_has_attribute_after_set() {
        let mut m = ClipMetadata::new(1);
        m.set_attribute(ClipAttribute::Codec("prores".into()));
        assert!(m.has_attribute("codec"));
    }

    #[test]
    fn test_clip_metadata_missing_attribute() {
        let m = ClipMetadata::new(1);
        assert!(!m.has_attribute("codec"));
    }

    #[test]
    fn test_clip_metadata_attribute_count() {
        let mut m = ClipMetadata::new(1);
        m.set_attribute(ClipAttribute::Codec("h264".into()));
        m.set_attribute(ClipAttribute::BitDepth(8));
        assert_eq!(m.attribute_count(), 2);
    }

    #[test]
    fn test_clip_metadata_overwrite() {
        let mut m = ClipMetadata::new(1);
        m.set_attribute(ClipAttribute::Codec("h264".into()));
        m.set_attribute(ClipAttribute::Codec("hevc".into()));
        assert_eq!(m.attribute_count(), 1);
        assert_eq!(
            m.get_attribute("codec"),
            Some(&ClipAttribute::Codec("hevc".into()))
        );
    }

    #[test]
    fn test_set_merge_overwrites() {
        let mut base = ClipMetadataSet::new();
        let mut m1 = ClipMetadata::new(10);
        m1.set_attribute(ClipAttribute::Codec("h264".into()));
        base.insert(m1);

        let mut overlay = ClipMetadataSet::new();
        let mut m2 = ClipMetadata::new(10);
        m2.set_attribute(ClipAttribute::Codec("hevc".into()));
        overlay.insert(m2);

        base.merge(&overlay);
        let merged = base.get(10).expect("get should succeed");
        assert_eq!(
            merged.get_attribute("codec"),
            Some(&ClipAttribute::Codec("hevc".into()))
        );
    }

    #[test]
    fn test_set_merge_adds_new_clip() {
        let mut base = ClipMetadataSet::new();
        let mut overlay = ClipMetadataSet::new();
        let mut m = ClipMetadata::new(99);
        m.set_attribute(ClipAttribute::BitDepth(10));
        overlay.insert(m);
        base.merge(&overlay);
        assert!(base.get(99).is_some());
    }

    #[test]
    fn test_diff_same_sets_empty() {
        let mut s1 = ClipMetadataSet::new();
        let mut m = ClipMetadata::new(1);
        m.set_attribute(ClipAttribute::Rating(90));
        s1.insert(m);
        let s2 = s1.clone();
        assert!(s1.diff(&s2).is_empty());
    }

    #[test]
    fn test_diff_detects_difference() {
        let mut s1 = ClipMetadataSet::new();
        let mut m1 = ClipMetadata::new(1);
        m1.set_attribute(ClipAttribute::Rating(90));
        s1.insert(m1);

        let mut s2 = ClipMetadataSet::new();
        let mut m2 = ClipMetadata::new(1);
        m2.set_attribute(ClipAttribute::Rating(50));
        s2.insert(m2);

        let diffs = s1.diff(&s2);
        assert!(!diffs.is_empty());
    }

    #[test]
    fn test_clip_count() {
        let mut s = ClipMetadataSet::new();
        s.insert(ClipMetadata::new(1));
        s.insert(ClipMetadata::new(2));
        assert_eq!(s.clip_count(), 2);
    }
}
