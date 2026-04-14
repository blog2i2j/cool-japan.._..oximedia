//! Batch tag operations: copy, merge, and strip tags across [`TagMap`]s.
//!
//! This module provides standalone free functions for common metadata
//! transformation tasks that operate on one or more [`TagMap`] instances:
//!
//! - [`copy_all_tags`] — copy every tag from one map to another.
//! - [`merge_tags`] — produce a new merged map according to a [`TagPreference`].
//! - [`strip_tags`] — remove a specified set of tags from a map.
//!
//! # Example
//!
//! ```ignore
//! use oximedia_container::metadata::{TagMap, TagValue};
//! use oximedia_container::metadata::batch_ops::{copy_all_tags, merge_tags, strip_tags, TagPreference};
//!
//! let mut src = TagMap::new();
//! src.set("TITLE", "Source Title");
//! src.set("ARTIST", "Source Artist");
//!
//! let mut dst = TagMap::new();
//! copy_all_tags(&src, &mut dst);
//! assert_eq!(dst.get_text("TITLE"), Some("Source Title"));
//!
//! let mut base = TagMap::new();
//! base.set("TITLE", "Base");
//! base.set("ALBUM", "Only in Base");
//! let mut overlay = TagMap::new();
//! overlay.set("TITLE", "Overlay");
//! overlay.set("ARTIST", "Only in Overlay");
//!
//! let merged = merge_tags(&base, &overlay, TagPreference::PreferB);
//! assert_eq!(merged.get_text("TITLE"), Some("Overlay"));
//! assert_eq!(merged.get_text("ALBUM"), Some("Only in Base"));
//! assert_eq!(merged.get_text("ARTIST"), Some("Only in Overlay"));
//!
//! let mut map = TagMap::new();
//! map.set("TITLE", "Keep");
//! map.set("COMMENT", "Strip");
//! map.set("ENCODER", "Strip too");
//! strip_tags(&mut map, &["COMMENT", "ENCODER"]);
//! assert!(map.get_text("COMMENT").is_none());
//! assert_eq!(map.get_text("TITLE"), Some("Keep"));
//! ```

#![forbid(unsafe_code)]

use super::tags::{TagMap, TagValue};

// ─── TagPreference ─────────────────────────────────────────────────────────────

/// Policy that controls which value wins when [`merge_tags`] encounters a key
/// present in both tag maps.
///
/// | Variant | Effect |
/// |---------|--------|
/// | [`PreferA`] | Values from `a` always overwrite those from `b`. |
/// | [`PreferB`] | Values from `b` always overwrite those from `a`. |
/// | [`Merge`]   | `a` fills all keys; `b` fills only keys absent in `a`. |
///
/// [`PreferA`]: TagPreference::PreferA
/// [`PreferB`]: TagPreference::PreferB
/// [`Merge`]:   TagPreference::Merge
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagPreference {
    /// Values from the first map (`a`) win on conflict.
    PreferA,
    /// Values from the second map (`b`) win on conflict.
    PreferB,
    /// `a` wins for non-empty fields; `b` fills any gaps left by `a`.
    ///
    /// Specifically: iterate all keys from both maps and take `a`'s value
    /// whenever `a` has a non-empty entry for that key; otherwise take `b`.
    Merge,
}

// ─── copy_all_tags ─────────────────────────────────────────────────────────────

/// Copies **all** tags from `src` into `dst`, replacing any existing values.
///
/// After this call `dst` will contain every key-value pair that was in `src`.
/// Keys already in `dst` but absent from `src` are left unchanged.
///
/// # Example
///
/// ```ignore
/// let mut src = TagMap::new();
/// src.set("TITLE", "My Title");
/// src.set("ARTIST", "Some Artist");
///
/// let mut dst = TagMap::new();
/// dst.set("ALBUM", "Existing Album");
///
/// copy_all_tags(&src, &mut dst);
/// // dst now has TITLE, ARTIST (from src) and ALBUM (unchanged)
/// assert_eq!(dst.get_text("TITLE"), Some("My Title"));
/// assert_eq!(dst.get_text("ALBUM"), Some("Existing Album"));
/// ```
pub fn copy_all_tags(src: &TagMap, dst: &mut TagMap) {
    for (key, value) in src.iter() {
        dst.set(key, value.clone());
    }
}

// ─── merge_tags ───────────────────────────────────────────────────────────────

/// Merges two tag maps into a new [`TagMap`] according to `prefer`.
///
/// Both maps are iterated; the resulting map will contain every key that
/// appears in either `a` or `b`.  When the same key appears in both maps,
/// `prefer` determines which value is kept.
///
/// See [`TagPreference`] for a detailed description of each policy.
#[must_use]
pub fn merge_tags(a: &TagMap, b: &TagMap, prefer: TagPreference) -> TagMap {
    let mut result = TagMap::new();

    match prefer {
        TagPreference::PreferA => {
            // Start with all of b, then overwrite with all of a.
            for (key, value) in b.iter() {
                result.set(key, value.clone());
            }
            for (key, value) in a.iter() {
                result.set(key, value.clone());
            }
        }
        TagPreference::PreferB => {
            // Start with all of a, then overwrite with all of b.
            for (key, value) in a.iter() {
                result.set(key, value.clone());
            }
            for (key, value) in b.iter() {
                result.set(key, value.clone());
            }
        }
        TagPreference::Merge => {
            // a wins for non-empty fields; b fills missing keys.
            // First pass: copy all of a.
            for (key, value) in a.iter() {
                let is_non_empty = match value {
                    TagValue::Text(s) => !s.is_empty(),
                    TagValue::Binary(b) => !b.is_empty(),
                };
                if is_non_empty {
                    result.set(key, value.clone());
                }
            }
            // Second pass: fill gaps from b.
            for (key, value) in b.iter() {
                if result.get(key).is_none() {
                    result.set(key, value.clone());
                }
            }
        }
    }

    result
}

// ─── strip_tags ───────────────────────────────────────────────────────────────

/// Removes all tags listed in `tags_to_remove` from `meta`.
///
/// Tag keys are compared case-insensitively.  Keys that do not exist in `meta`
/// are silently ignored.
///
/// # Example
///
/// ```ignore
/// let mut meta = TagMap::new();
/// meta.set("TITLE", "Keep this");
/// meta.set("COMMENT", "Remove this");
/// meta.set("ENCODER", "Remove this too");
///
/// strip_tags(&mut meta, &["COMMENT", "ENCODER"]);
///
/// assert_eq!(meta.get_text("TITLE"), Some("Keep this"));
/// assert!(meta.get_text("COMMENT").is_none());
/// assert!(meta.get_text("ENCODER").is_none());
/// ```
pub fn strip_tags(meta: &mut TagMap, tags_to_remove: &[&str]) {
    for &key in tags_to_remove {
        meta.remove(key);
    }
}

// ─── copy_selected_tags ───────────────────────────────────────────────────────

/// Copies only the tags listed in `keys` from `src` to `dst`.
///
/// Keys in `keys` that do not exist in `src` are silently ignored.
/// Existing values in `dst` for those keys are replaced.
///
/// # Example
///
/// ```ignore
/// let mut src = TagMap::new();
/// src.set("TITLE", "My Title");
/// src.set("ARTIST", "My Artist");
/// src.set("COMMENT", "Internal note");
///
/// let mut dst = TagMap::new();
/// copy_selected_tags(&src, &mut dst, &["TITLE", "ARTIST"]);
/// assert_eq!(dst.get_text("TITLE"), Some("My Title"));
/// assert_eq!(dst.get_text("ARTIST"), Some("My Artist"));
/// assert!(dst.get_text("COMMENT").is_none()); // not copied
/// ```
pub fn copy_selected_tags(src: &TagMap, dst: &mut TagMap, keys: &[&str]) {
    for &key in keys {
        if let Some(value) = src.get(key) {
            dst.set(key, value.clone());
        }
    }
}

// ─── rename_tag ───────────────────────────────────────────────────────────��───

/// Renames a tag key in `meta` from `from` to `to`, preserving its value.
///
/// If `from` is absent the function is a no-op and returns `false`.
/// If `to` already exists its previous value is overwritten.
///
/// Returns `true` if the rename was performed.
pub fn rename_tag(meta: &mut TagMap, from: &str, to: &str) -> bool {
    if let Some(value) = meta.get(from).cloned() {
        meta.remove(from);
        meta.set(to, value);
        true
    } else {
        false
    }
}

// ─── TagCopySession ───────────────────────────────────────────────────────────

/// Represents a *filter* applied to a tag-copy operation.
///
/// When building a [`TagCopySession`] you can restrict which tags are
/// transferred by attaching one or more `TagFilter`s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TagFilter {
    /// Only copy tags whose keys are listed here (case-insensitive).
    AllowList(Vec<String>),
    /// Skip tags whose keys are listed here (case-insensitive).
    BlockList(Vec<String>),
    /// Copy all tags (no filtering). This is the default.
    PassThrough,
}

impl TagFilter {
    /// Returns `true` if `key` passes this filter.
    #[must_use]
    pub fn allows(&self, key: &str) -> bool {
        let key_upper = key.to_ascii_uppercase();
        match self {
            Self::AllowList(list) => list
                .iter()
                .any(|k| k.to_ascii_uppercase() == key_upper),
            Self::BlockList(list) => !list
                .iter()
                .any(|k| k.to_ascii_uppercase() == key_upper),
            Self::PassThrough => true,
        }
    }
}

impl Default for TagFilter {
    fn default() -> Self {
        Self::PassThrough
    }
}

/// Session that copies tags from one [`TagMap`] to one or more destinations.
///
/// A `TagCopySession` accumulates a single source and applies one [`TagFilter`]
/// when transferring to each destination.  Call [`execute`] to perform the
/// transfer; all previously recorded destinations are updated in a single pass
/// over the source map.
///
/// # Example
///
/// ```ignore
/// use oximedia_container::metadata::batch_ops::{TagCopySession, TagFilter};
/// use oximedia_container::metadata::TagMap;
///
/// let mut src = TagMap::new();
/// src.set("TITLE", "Track 1");
/// src.set("ARTIST", "Someone");
/// src.set("ENCODER", "libopus 1.3");
///
/// let mut dst1 = TagMap::new();
/// let mut dst2 = TagMap::new();
///
/// let count = TagCopySession::new(&src)
///     .with_filter(TagFilter::BlockList(vec!["ENCODER".to_string()]))
///     .add_destination(&mut dst1)
///     .add_destination(&mut dst2)
///     .execute();
///
/// assert_eq!(count, 4); // 2 tags × 2 destinations
/// assert!(dst1.get_text("ENCODER").is_none());
/// assert_eq!(dst2.get_text("ARTIST"), Some("Someone"));
/// ```
///
/// [`execute`]: TagCopySession::execute
pub struct TagCopySession<'a> {
    source: &'a TagMap,
    filter: TagFilter,
    destinations: Vec<&'a mut TagMap>,
}

impl<'a> TagCopySession<'a> {
    /// Creates a new session with `source` as the tag origin.
    #[must_use]
    pub fn new(source: &'a TagMap) -> Self {
        Self {
            source,
            filter: TagFilter::PassThrough,
            destinations: Vec::new(),
        }
    }

    /// Sets the tag filter for this session.
    #[must_use]
    pub fn with_filter(mut self, filter: TagFilter) -> Self {
        self.filter = filter;
        self
    }

    /// Adds a destination [`TagMap`] that will receive the copied tags.
    #[must_use]
    pub fn add_destination(mut self, dst: &'a mut TagMap) -> Self {
        self.destinations.push(dst);
        self
    }

    /// Executes the copy, returning the total number of tag-value pairs written
    /// across all destinations.
    pub fn execute(self) -> usize {
        let mut count = 0usize;
        // Collect filtered entries once.
        let entries: Vec<(&str, &TagValue)> = self
            .source
            .iter()
            .filter(|(k, _)| self.filter.allows(k))
            .collect();

        for dst in self.destinations {
            for (key, value) in &entries {
                dst.set(*key, (*value).clone());
                count += 1;
            }
        }
        count
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_map(pairs: &[(&str, &str)]) -> TagMap {
        let mut m = TagMap::new();
        for &(k, v) in pairs {
            m.set(k, v);
        }
        m
    }

    // ── copy_all_tags ────────────────────────────────────────────────────

    #[test]
    fn test_copy_all_tags_basic() {
        let src = make_map(&[("TITLE", "Title"), ("ARTIST", "Artist")]);
        let mut dst = TagMap::new();
        copy_all_tags(&src, &mut dst);
        assert_eq!(dst.get_text("TITLE"), Some("Title"));
        assert_eq!(dst.get_text("ARTIST"), Some("Artist"));
    }

    #[test]
    fn test_copy_all_tags_overwrites_existing() {
        let src = make_map(&[("TITLE", "New Title")]);
        let mut dst = make_map(&[("TITLE", "Old Title"), ("ALBUM", "Existing")]);
        copy_all_tags(&src, &mut dst);
        assert_eq!(dst.get_text("TITLE"), Some("New Title"));
        // Keys absent from src are preserved
        assert_eq!(dst.get_text("ALBUM"), Some("Existing"));
    }

    #[test]
    fn test_copy_all_tags_from_empty() {
        let src = TagMap::new();
        let mut dst = make_map(&[("TITLE", "Keep")]);
        copy_all_tags(&src, &mut dst);
        assert_eq!(dst.get_text("TITLE"), Some("Keep"));
    }

    #[test]
    fn test_copy_all_tags_into_empty() {
        let src = make_map(&[("TITLE", "T"), ("ARTIST", "A"), ("ALBUM", "B")]);
        let mut dst = TagMap::new();
        copy_all_tags(&src, &mut dst);
        assert_eq!(dst.get_text("TITLE"), Some("T"));
        assert_eq!(dst.get_text("ARTIST"), Some("A"));
        assert_eq!(dst.get_text("ALBUM"), Some("B"));
    }

    // ── merge_tags ───────────────────────────────────────────────────────

    #[test]
    fn test_merge_prefer_a_on_conflict() {
        let a = make_map(&[("TITLE", "A-Title"), ("ARTIST", "A-Artist")]);
        let b = make_map(&[("TITLE", "B-Title"), ("ALBUM", "B-Album")]);
        let merged = merge_tags(&a, &b, TagPreference::PreferA);
        assert_eq!(merged.get_text("TITLE"), Some("A-Title")); // a wins
        assert_eq!(merged.get_text("ARTIST"), Some("A-Artist")); // only in a
        assert_eq!(merged.get_text("ALBUM"), Some("B-Album")); // only in b
    }

    #[test]
    fn test_merge_prefer_b_on_conflict() {
        let a = make_map(&[("TITLE", "A-Title"), ("ARTIST", "A-Artist")]);
        let b = make_map(&[("TITLE", "B-Title"), ("ALBUM", "B-Album")]);
        let merged = merge_tags(&a, &b, TagPreference::PreferB);
        assert_eq!(merged.get_text("TITLE"), Some("B-Title")); // b wins
        assert_eq!(merged.get_text("ARTIST"), Some("A-Artist")); // only in a
        assert_eq!(merged.get_text("ALBUM"), Some("B-Album")); // only in b
    }

    #[test]
    fn test_merge_merge_policy() {
        let a = make_map(&[("TITLE", "A-Title"), ("ALBUM", "")]);
        let b = make_map(&[("TITLE", "B-Title"), ("ALBUM", "B-Album"), ("ARTIST", "B-Artist")]);
        let merged = merge_tags(&a, &b, TagPreference::Merge);
        // a has non-empty TITLE → a wins
        assert_eq!(merged.get_text("TITLE"), Some("A-Title"));
        // a has empty ALBUM → b fills it
        assert_eq!(merged.get_text("ALBUM"), Some("B-Album"));
        // ARTIST absent in a → b fills it
        assert_eq!(merged.get_text("ARTIST"), Some("B-Artist"));
    }

    #[test]
    fn test_merge_both_empty() {
        let a = TagMap::new();
        let b = TagMap::new();
        let merged = merge_tags(&a, &b, TagPreference::Merge);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_merge_only_a_has_values() {
        let a = make_map(&[("TITLE", "A"), ("ARTIST", "B")]);
        let b = TagMap::new();
        let merged = merge_tags(&a, &b, TagPreference::PreferB);
        assert_eq!(merged.get_text("TITLE"), Some("A"));
        assert_eq!(merged.get_text("ARTIST"), Some("B"));
    }

    // ── strip_tags ───────────────────────────────────────────────────────

    #[test]
    fn test_strip_tags_basic() {
        let mut meta = make_map(&[("TITLE", "Keep"), ("COMMENT", "Remove"), ("ENCODER", "Remove")]);
        strip_tags(&mut meta, &["COMMENT", "ENCODER"]);
        assert_eq!(meta.get_text("TITLE"), Some("Keep"));
        assert!(meta.get_text("COMMENT").is_none());
        assert!(meta.get_text("ENCODER").is_none());
    }

    #[test]
    fn test_strip_tags_case_insensitive() {
        let mut meta = make_map(&[("COMMENT", "x"), ("TITLE", "y")]);
        // Lower-case key should still match
        strip_tags(&mut meta, &["comment"]);
        assert!(meta.get_text("COMMENT").is_none());
        assert_eq!(meta.get_text("TITLE"), Some("y"));
    }

    #[test]
    fn test_strip_tags_absent_key_is_noop() {
        let mut meta = make_map(&[("TITLE", "T")]);
        strip_tags(&mut meta, &["NONEXISTENT", "ALSO_GONE"]);
        assert_eq!(meta.get_text("TITLE"), Some("T"));
    }

    #[test]
    fn test_strip_tags_empty_list() {
        let mut meta = make_map(&[("TITLE", "T"), ("ARTIST", "A")]);
        strip_tags(&mut meta, &[]);
        assert_eq!(meta.get_text("TITLE"), Some("T"));
        assert_eq!(meta.get_text("ARTIST"), Some("A"));
    }

    #[test]
    fn test_strip_all_tags() {
        let mut meta = make_map(&[("A", "1"), ("B", "2"), ("C", "3")]);
        strip_tags(&mut meta, &["A", "B", "C"]);
        assert!(meta.is_empty());
    }

    // ── copy_selected_tags ────────────────────────────────────────���──────

    #[test]
    fn test_copy_selected_tags_subset() {
        let src = make_map(&[("TITLE", "T"), ("ARTIST", "A"), ("COMMENT", "C")]);
        let mut dst = TagMap::new();
        copy_selected_tags(&src, &mut dst, &["TITLE", "ARTIST"]);
        assert_eq!(dst.get_text("TITLE"), Some("T"));
        assert_eq!(dst.get_text("ARTIST"), Some("A"));
        assert!(dst.get_text("COMMENT").is_none());
    }

    #[test]
    fn test_copy_selected_tags_absent_key_skipped() {
        let src = make_map(&[("TITLE", "T")]);
        let mut dst = TagMap::new();
        copy_selected_tags(&src, &mut dst, &["TITLE", "NONEXISTENT"]);
        assert_eq!(dst.get_text("TITLE"), Some("T"));
    }

    #[test]
    fn test_copy_selected_tags_overwrites() {
        let src = make_map(&[("TITLE", "New")]);
        let mut dst = make_map(&[("TITLE", "Old")]);
        copy_selected_tags(&src, &mut dst, &["TITLE"]);
        assert_eq!(dst.get_text("TITLE"), Some("New"));
    }

    // ── rename_tag ────────────────────────────────────��──────────────────

    #[test]
    fn test_rename_tag_basic() {
        let mut meta = make_map(&[("OLDKEY", "value"), ("OTHER", "x")]);
        let did_rename = rename_tag(&mut meta, "OLDKEY", "NEWKEY");
        assert!(did_rename);
        assert_eq!(meta.get_text("NEWKEY"), Some("value"));
        assert!(meta.get_text("OLDKEY").is_none());
        assert_eq!(meta.get_text("OTHER"), Some("x")); // unaffected
    }

    #[test]
    fn test_rename_tag_absent_is_noop() {
        let mut meta = make_map(&[("TITLE", "T")]);
        let did_rename = rename_tag(&mut meta, "ABSENT", "NEW");
        assert!(!did_rename);
        assert_eq!(meta.get_text("TITLE"), Some("T"));
        assert!(meta.get_text("NEW").is_none());
    }

    // ── TagFilter ────────────────────────────────────────────────────────

    #[test]
    fn test_tag_filter_pass_through() {
        let f = TagFilter::PassThrough;
        assert!(f.allows("TITLE"));
        assert!(f.allows("ANYTHING"));
    }

    #[test]
    fn test_tag_filter_allow_list() {
        let f = TagFilter::AllowList(vec!["TITLE".into(), "ARTIST".into()]);
        assert!(f.allows("TITLE"));
        assert!(f.allows("title")); // case-insensitive
        assert!(!f.allows("COMMENT"));
    }

    #[test]
    fn test_tag_filter_block_list() {
        let f = TagFilter::BlockList(vec!["ENCODER".into()]);
        assert!(f.allows("TITLE"));
        assert!(!f.allows("encoder")); // case-insensitive block
    }

    // ── TagCopySession ───────────────────────────────────────────────────

    #[test]
    fn test_tag_copy_session_single_destination() {
        let src = make_map(&[("TITLE", "T"), ("ARTIST", "A"), ("ENCODER", "enc")]);
        let mut dst = TagMap::new();
        let count = TagCopySession::new(&src)
            .with_filter(TagFilter::BlockList(vec!["ENCODER".into()]))
            .add_destination(&mut dst)
            .execute();
        assert_eq!(count, 2);
        assert_eq!(dst.get_text("TITLE"), Some("T"));
        assert_eq!(dst.get_text("ARTIST"), Some("A"));
        assert!(dst.get_text("ENCODER").is_none());
    }

    #[test]
    fn test_tag_copy_session_multiple_destinations() {
        let src = make_map(&[("TITLE", "T"), ("ARTIST", "A")]);
        let mut dst1 = TagMap::new();
        let mut dst2 = TagMap::new();
        let count = TagCopySession::new(&src)
            .add_destination(&mut dst1)
            .add_destination(&mut dst2)
            .execute();
        // 2 tags × 2 destinations = 4
        assert_eq!(count, 4);
        assert_eq!(dst1.get_text("TITLE"), Some("T"));
        assert_eq!(dst2.get_text("ARTIST"), Some("A"));
    }

    #[test]
    fn test_tag_copy_session_no_destinations() {
        let src = make_map(&[("TITLE", "T")]);
        let count = TagCopySession::new(&src).execute();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_tag_copy_session_allow_list_filter() {
        let src = make_map(&[("TITLE", "T"), ("ARTIST", "A"), ("COMMENT", "C")]);
        let mut dst = TagMap::new();
        let count = TagCopySession::new(&src)
            .with_filter(TagFilter::AllowList(vec!["TITLE".into()]))
            .add_destination(&mut dst)
            .execute();
        assert_eq!(count, 1);
        assert_eq!(dst.get_text("TITLE"), Some("T"));
        assert!(dst.get_text("ARTIST").is_none());
    }

    #[test]
    fn test_tag_copy_session_empty_source() {
        let src = TagMap::new();
        let mut dst = TagMap::new();
        dst.set("TITLE", "Existing");
        let count = TagCopySession::new(&src)
            .add_destination(&mut dst)
            .execute();
        assert_eq!(count, 0);
        // dst untouched
        assert_eq!(dst.get_text("TITLE"), Some("Existing"));
    }
}
