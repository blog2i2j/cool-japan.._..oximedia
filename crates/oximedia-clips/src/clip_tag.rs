#![allow(dead_code)]
//! Structured tagging system for clips.
//!
//! This module provides a hierarchical, typed tag system for organizing and
//! categorizing clips. Unlike simple keywords, tags have a namespace, category,
//! optional value, and color. Supports tag inheritance, bulk operations,
//! frequency analysis, and auto-suggest based on existing tags.

use std::collections::{HashMap, HashSet};

/// Unique identifier for a tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TagId(pub u64);

/// Color label for visual tag display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TagColor {
    /// Red color.
    Red,
    /// Orange color.
    Orange,
    /// Yellow color.
    Yellow,
    /// Green color.
    Green,
    /// Blue color.
    Blue,
    /// Purple color.
    Purple,
    /// Gray (no specific color).
    Gray,
}

impl Default for TagColor {
    fn default() -> Self {
        Self::Gray
    }
}

/// A single structured tag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tag {
    /// Tag identifier.
    pub id: TagId,
    /// Namespace (e.g., "project", "client", "scene").
    pub namespace: String,
    /// Category within the namespace (e.g., "location", "character").
    pub category: String,
    /// Tag value/name.
    pub value: String,
    /// Display color.
    pub color: TagColor,
    /// Optional parent tag ID for hierarchical tags.
    pub parent: Option<TagId>,
}

impl Tag {
    /// Creates a new tag.
    #[must_use]
    pub fn new(
        id: TagId,
        namespace: impl Into<String>,
        category: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        Self {
            id,
            namespace: namespace.into(),
            category: category.into(),
            value: value.into(),
            color: TagColor::default(),
            parent: None,
        }
    }

    /// Returns the fully qualified tag path (namespace:category:value).
    #[must_use]
    pub fn full_path(&self) -> String {
        format!("{}:{}:{}", self.namespace, self.category, self.value)
    }

    /// Returns true if this tag matches the given namespace.
    #[must_use]
    pub fn is_namespace(&self, ns: &str) -> bool {
        self.namespace == ns
    }

    /// Returns true if this tag matches the given category.
    #[must_use]
    pub fn is_category(&self, cat: &str) -> bool {
        self.category == cat
    }
}

/// Association between a clip and a tag.
#[derive(Debug, Clone)]
pub struct ClipTagBinding {
    /// Clip identifier.
    pub clip_id: u64,
    /// Tag identifier.
    pub tag_id: TagId,
    /// Confidence score (0.0 to 1.0) for auto-generated tags.
    pub confidence: f64,
    /// Whether this tag was manually applied.
    pub manual: bool,
}

impl ClipTagBinding {
    /// Creates a new manual tag binding.
    #[must_use]
    pub fn manual(clip_id: u64, tag_id: TagId) -> Self {
        Self {
            clip_id,
            tag_id,
            confidence: 1.0,
            manual: true,
        }
    }

    /// Creates a new auto-generated tag binding.
    #[must_use]
    pub fn auto_generated(clip_id: u64, tag_id: TagId, confidence: f64) -> Self {
        Self {
            clip_id,
            tag_id,
            confidence: confidence.clamp(0.0, 1.0),
            manual: false,
        }
    }
}

/// Tag frequency counter for analysis.
#[derive(Debug, Clone)]
pub struct TagFrequency {
    /// Tag identifier.
    pub tag_id: TagId,
    /// Number of clips using this tag.
    pub count: usize,
    /// Percentage of total clips using this tag.
    pub percentage: f64,
}

/// Tag registry and clip-tag relationship manager.
#[derive(Debug)]
pub struct TagRegistry {
    /// All registered tags, keyed by ID.
    tags: HashMap<TagId, Tag>,
    /// Clip-to-tags mapping.
    clip_tags: HashMap<u64, HashSet<TagId>>,
    /// Tag-to-clips mapping (reverse index).
    tag_clips: HashMap<TagId, HashSet<u64>>,
    /// Next available tag ID.
    next_id: u64,
}

impl Default for TagRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl TagRegistry {
    /// Creates a new empty tag registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tags: HashMap::new(),
            clip_tags: HashMap::new(),
            tag_clips: HashMap::new(),
            next_id: 1,
        }
    }

    /// Registers a new tag and returns its ID.
    pub fn register_tag(
        &mut self,
        namespace: impl Into<String>,
        category: impl Into<String>,
        value: impl Into<String>,
    ) -> TagId {
        let id = TagId(self.next_id);
        self.next_id += 1;
        let tag = Tag::new(id, namespace, category, value);
        self.tags.insert(id, tag);
        id
    }

    /// Registers a tag with a specific color.
    pub fn register_colored_tag(
        &mut self,
        namespace: impl Into<String>,
        category: impl Into<String>,
        value: impl Into<String>,
        color: TagColor,
    ) -> TagId {
        let id = self.register_tag(namespace, category, value);
        if let Some(tag) = self.tags.get_mut(&id) {
            tag.color = color;
        }
        id
    }

    /// Returns a reference to a tag by ID.
    #[must_use]
    pub fn get_tag(&self, id: TagId) -> Option<&Tag> {
        self.tags.get(&id)
    }

    /// Returns the total number of registered tags.
    #[must_use]
    pub fn tag_count(&self) -> usize {
        self.tags.len()
    }

    /// Assigns a tag to a clip.
    pub fn assign_tag(&mut self, clip_id: u64, tag_id: TagId) -> bool {
        if !self.tags.contains_key(&tag_id) {
            return false;
        }
        self.clip_tags.entry(clip_id).or_default().insert(tag_id);
        self.tag_clips.entry(tag_id).or_default().insert(clip_id);
        true
    }

    /// Removes a tag from a clip.
    pub fn remove_tag(&mut self, clip_id: u64, tag_id: TagId) -> bool {
        let removed_from_clip = self
            .clip_tags
            .get_mut(&clip_id)
            .map_or(false, |tags| tags.remove(&tag_id));
        let removed_from_tag = self
            .tag_clips
            .get_mut(&tag_id)
            .map_or(false, |clips| clips.remove(&clip_id));
        removed_from_clip && removed_from_tag
    }

    /// Returns all tag IDs assigned to a clip.
    #[must_use]
    pub fn tags_for_clip(&self, clip_id: u64) -> Vec<TagId> {
        self.clip_tags
            .get(&clip_id)
            .map(|tags| {
                let mut v: Vec<TagId> = tags.iter().copied().collect();
                v.sort();
                v
            })
            .unwrap_or_default()
    }

    /// Returns all clip IDs that have a given tag.
    #[must_use]
    pub fn clips_with_tag(&self, tag_id: TagId) -> Vec<u64> {
        self.tag_clips
            .get(&tag_id)
            .map(|clips| {
                let mut v: Vec<u64> = clips.iter().copied().collect();
                v.sort();
                v
            })
            .unwrap_or_default()
    }

    /// Returns clips that have ALL of the given tags.
    #[must_use]
    pub fn clips_with_all_tags(&self, tag_ids: &[TagId]) -> Vec<u64> {
        if tag_ids.is_empty() {
            return Vec::new();
        }

        let mut result: Option<HashSet<u64>> = None;
        for tag_id in tag_ids {
            let clips = self.tag_clips.get(tag_id).cloned().unwrap_or_default();
            result = Some(match result {
                Some(prev) => prev.intersection(&clips).copied().collect(),
                None => clips,
            });
        }

        let mut v: Vec<u64> = result.unwrap_or_default().into_iter().collect();
        v.sort();
        v
    }

    /// Returns all tags in a given namespace.
    #[must_use]
    pub fn tags_in_namespace(&self, namespace: &str) -> Vec<&Tag> {
        self.tags
            .values()
            .filter(|t| t.namespace == namespace)
            .collect()
    }

    /// Returns tag frequency analysis.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn tag_frequencies(&self) -> Vec<TagFrequency> {
        let total_clips = self.clip_tags.len();
        let mut freqs: Vec<TagFrequency> = self
            .tag_clips
            .iter()
            .map(|(tag_id, clips)| {
                let count = clips.len();
                let percentage = if total_clips == 0 {
                    0.0
                } else {
                    (count as f64 / total_clips as f64) * 100.0
                };
                TagFrequency {
                    tag_id: *tag_id,
                    count,
                    percentage,
                }
            })
            .collect();
        freqs.sort_by(|a, b| b.count.cmp(&a.count));
        freqs
    }

    /// Suggests tags based on tags already assigned to a clip.
    /// Returns tags that frequently co-occur with the clip's existing tags.
    #[must_use]
    pub fn suggest_tags(&self, clip_id: u64, max_suggestions: usize) -> Vec<TagId> {
        let existing_tags = self.tags_for_clip(clip_id);
        if existing_tags.is_empty() {
            return Vec::new();
        }

        let existing_set: HashSet<TagId> = existing_tags.iter().copied().collect();
        let mut co_occurrence: HashMap<TagId, usize> = HashMap::new();

        // For each existing tag, look at other clips with that tag
        for tag_id in &existing_tags {
            if let Some(sibling_clips) = self.tag_clips.get(tag_id) {
                for sibling_clip_id in sibling_clips {
                    if *sibling_clip_id == clip_id {
                        continue;
                    }
                    if let Some(sibling_tags) = self.clip_tags.get(sibling_clip_id) {
                        for st in sibling_tags {
                            if !existing_set.contains(st) {
                                *co_occurrence.entry(*st).or_insert(0) += 1;
                            }
                        }
                    }
                }
            }
        }

        let mut suggestions: Vec<(TagId, usize)> = co_occurrence.into_iter().collect();
        suggestions.sort_by(|a, b| b.1.cmp(&a.1));
        suggestions
            .into_iter()
            .take(max_suggestions)
            .map(|(id, _)| id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tag_new() {
        let tag = Tag::new(TagId(1), "project", "location", "studio-a");
        assert_eq!(tag.id, TagId(1));
        assert_eq!(tag.namespace, "project");
        assert_eq!(tag.category, "location");
        assert_eq!(tag.value, "studio-a");
        assert_eq!(tag.color, TagColor::Gray);
    }

    #[test]
    fn test_tag_full_path() {
        let tag = Tag::new(TagId(1), "project", "location", "studio-a");
        assert_eq!(tag.full_path(), "project:location:studio-a");
    }

    #[test]
    fn test_tag_is_namespace() {
        let tag = Tag::new(TagId(1), "project", "location", "studio-a");
        assert!(tag.is_namespace("project"));
        assert!(!tag.is_namespace("client"));
    }

    #[test]
    fn test_tag_is_category() {
        let tag = Tag::new(TagId(1), "project", "location", "studio-a");
        assert!(tag.is_category("location"));
        assert!(!tag.is_category("character"));
    }

    #[test]
    fn test_clip_tag_binding_manual() {
        let binding = ClipTagBinding::manual(1, TagId(10));
        assert_eq!(binding.clip_id, 1);
        assert_eq!(binding.tag_id, TagId(10));
        assert!(binding.manual);
        assert!((binding.confidence - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clip_tag_binding_auto() {
        let binding = ClipTagBinding::auto_generated(1, TagId(10), 0.85);
        assert!(!binding.manual);
        assert!((binding.confidence - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clip_tag_binding_auto_clamp() {
        let binding = ClipTagBinding::auto_generated(1, TagId(10), 1.5);
        assert!((binding.confidence - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_registry_register_tag() {
        let mut reg = TagRegistry::new();
        let id = reg.register_tag("project", "location", "studio-a");
        assert_eq!(reg.tag_count(), 1);
        let tag = reg.get_tag(id).expect("get_tag should succeed");
        assert_eq!(tag.value, "studio-a");
    }

    #[test]
    fn test_registry_register_colored_tag() {
        let mut reg = TagRegistry::new();
        let id = reg.register_colored_tag("project", "status", "approved", TagColor::Green);
        let tag = reg.get_tag(id).expect("get_tag should succeed");
        assert_eq!(tag.color, TagColor::Green);
    }

    #[test]
    fn test_registry_assign_and_query() {
        let mut reg = TagRegistry::new();
        let t1 = reg.register_tag("project", "location", "studio-a");
        let t2 = reg.register_tag("project", "location", "studio-b");

        assert!(reg.assign_tag(1, t1));
        assert!(reg.assign_tag(1, t2));
        assert!(reg.assign_tag(2, t1));

        let tags = reg.tags_for_clip(1);
        assert_eq!(tags.len(), 2);
        let clips = reg.clips_with_tag(t1);
        assert_eq!(clips.len(), 2);
    }

    #[test]
    fn test_registry_remove_tag() {
        let mut reg = TagRegistry::new();
        let t1 = reg.register_tag("project", "location", "studio-a");
        reg.assign_tag(1, t1);
        assert!(reg.remove_tag(1, t1));
        assert!(reg.tags_for_clip(1).is_empty());
        assert!(reg.clips_with_tag(t1).is_empty());
    }

    #[test]
    fn test_registry_assign_invalid_tag() {
        let mut reg = TagRegistry::new();
        assert!(!reg.assign_tag(1, TagId(999)));
    }

    #[test]
    fn test_registry_clips_with_all_tags() {
        let mut reg = TagRegistry::new();
        let t1 = reg.register_tag("p", "c", "v1");
        let t2 = reg.register_tag("p", "c", "v2");
        reg.assign_tag(1, t1);
        reg.assign_tag(1, t2);
        reg.assign_tag(2, t1);

        let both = reg.clips_with_all_tags(&[t1, t2]);
        assert_eq!(both, vec![1]);
    }

    #[test]
    fn test_registry_tags_in_namespace() {
        let mut reg = TagRegistry::new();
        reg.register_tag("project", "location", "studio-a");
        reg.register_tag("project", "character", "alice");
        reg.register_tag("client", "name", "acme");
        let project_tags = reg.tags_in_namespace("project");
        assert_eq!(project_tags.len(), 2);
    }

    #[test]
    fn test_registry_tag_frequencies() {
        let mut reg = TagRegistry::new();
        let t1 = reg.register_tag("p", "c", "popular");
        let t2 = reg.register_tag("p", "c", "rare");
        reg.assign_tag(1, t1);
        reg.assign_tag(2, t1);
        reg.assign_tag(3, t1);
        reg.assign_tag(1, t2);

        let freqs = reg.tag_frequencies();
        assert_eq!(freqs[0].tag_id, t1);
        assert_eq!(freqs[0].count, 3);
    }

    #[test]
    fn test_registry_suggest_tags() {
        let mut reg = TagRegistry::new();
        let t_interview = reg.register_tag("p", "type", "interview");
        let t_outdoor = reg.register_tag("p", "loc", "outdoor");
        let t_sunny = reg.register_tag("p", "weather", "sunny");

        // Clip 1: interview + outdoor + sunny
        reg.assign_tag(1, t_interview);
        reg.assign_tag(1, t_outdoor);
        reg.assign_tag(1, t_sunny);
        // Clip 2: interview + outdoor
        reg.assign_tag(2, t_interview);
        reg.assign_tag(2, t_outdoor);

        // Clip 3 has interview only; suggest based on co-occurrence
        reg.assign_tag(3, t_interview);
        let suggestions = reg.suggest_tags(3, 5);
        assert!(!suggestions.is_empty());
        // outdoor should be most suggested (co-occurs with interview in 2 clips)
        assert!(suggestions.contains(&t_outdoor));
    }
}
