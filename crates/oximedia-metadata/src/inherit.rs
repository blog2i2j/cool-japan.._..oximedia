//! Metadata inheritance — merge parent and child metadata maps.
//!
//! [`MetadataInheritance`] merges two flat metadata maps (parent and child).
//! Child values take precedence over parent values; fields only in the parent
//! are inherited.

use std::collections::HashMap;

/// A flat metadata map: field name → string value.
pub type MetadataMap = HashMap<String, String>;

/// Utility for merging parent and child metadata maps.
///
/// The merge strategy is:
/// - All parent fields are included.
/// - Any child field **overrides** the corresponding parent field.
/// - Fields present only in the child are included as-is.
pub struct MetadataInheritance;

impl MetadataInheritance {
    /// Merge `parent` and `child` metadata maps.
    ///
    /// Child values take precedence over parent values for any common key.
    /// The returned map contains all keys from both maps.
    pub fn merge(parent: &MetadataMap, child: &MetadataMap) -> MetadataMap {
        let mut result = parent.clone();
        for (key, value) in child {
            result.insert(key.clone(), value.clone());
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: &[(&str, &str)]) -> MetadataMap {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn test_child_overrides_parent() {
        let parent = map(&[("title", "Parent Title"), ("artist", "Parent Artist")]);
        let child = map(&[("title", "Child Title")]);
        let result = MetadataInheritance::merge(&parent, &child);
        assert_eq!(result["title"], "Child Title");
        assert_eq!(result["artist"], "Parent Artist"); // inherited
    }

    #[test]
    fn test_parent_only_fields_inherited() {
        let parent = map(&[("album", "Some Album"), ("year", "2024")]);
        let child = map(&[]);
        let result = MetadataInheritance::merge(&parent, &child);
        assert_eq!(result["album"], "Some Album");
        assert_eq!(result["year"], "2024");
    }

    #[test]
    fn test_child_only_fields_included() {
        let parent = map(&[]);
        let child = map(&[("track", "01")]);
        let result = MetadataInheritance::merge(&parent, &child);
        assert_eq!(result["track"], "01");
    }

    #[test]
    fn test_both_empty() {
        let result = MetadataInheritance::merge(&map(&[]), &map(&[]));
        assert!(result.is_empty());
    }

    #[test]
    fn test_merge_does_not_mutate_inputs() {
        let parent = map(&[("a", "1")]);
        let child = map(&[("a", "2"), ("b", "3")]);
        let result = MetadataInheritance::merge(&parent, &child);
        assert_eq!(parent["a"], "1"); // unchanged
        assert_eq!(result["a"], "2");
        assert_eq!(result["b"], "3");
    }
}
