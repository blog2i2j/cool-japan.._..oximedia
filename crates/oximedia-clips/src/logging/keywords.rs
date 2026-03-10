//! Keyword management for clips.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// A keyword with hierarchical support (e.g., "People/John Doe").
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Keyword {
    /// Full keyword path (e.g., "People/John Doe").
    pub path: String,

    /// Individual components.
    pub components: Vec<String>,
}

impl Keyword {
    /// Creates a new keyword from a path.
    #[must_use]
    pub fn new(path: impl Into<String>) -> Self {
        let path = path.into();
        let components: Vec<String> = path.split('/').map(String::from).collect();

        Self { path, components }
    }

    /// Returns the parent keyword, if any.
    #[must_use]
    pub fn parent(&self) -> Option<Self> {
        if self.components.len() <= 1 {
            return None;
        }

        let parent_components = &self.components[..self.components.len() - 1];
        let parent_path = parent_components.join("/");

        Some(Self {
            path: parent_path,
            components: parent_components.to_vec(),
        })
    }

    /// Returns the leaf name (last component).
    #[must_use]
    pub fn leaf(&self) -> Option<&str> {
        self.components.last().map(String::as_str)
    }

    /// Returns the depth of this keyword (number of components).
    #[must_use]
    pub fn depth(&self) -> usize {
        self.components.len()
    }

    /// Checks if this keyword is a descendant of another.
    #[must_use]
    pub fn is_descendant_of(&self, other: &Self) -> bool {
        if self.components.len() <= other.components.len() {
            return false;
        }

        self.components
            .iter()
            .zip(&other.components)
            .all(|(a, b)| a == b)
    }
}

impl std::fmt::Display for Keyword {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.path)
    }
}

/// A collection of keywords organized hierarchically.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KeywordCollection {
    keywords: HashSet<Keyword>,
}

impl KeywordCollection {
    /// Creates a new empty keyword collection.
    #[must_use]
    pub fn new() -> Self {
        Self {
            keywords: HashSet::new(),
        }
    }

    /// Adds a keyword to the collection.
    pub fn add(&mut self, keyword: Keyword) {
        self.keywords.insert(keyword);
    }

    /// Removes a keyword from the collection.
    pub fn remove(&mut self, keyword: &Keyword) -> bool {
        self.keywords.remove(keyword)
    }

    /// Checks if the collection contains a keyword.
    #[must_use]
    pub fn contains(&self, keyword: &Keyword) -> bool {
        self.keywords.contains(keyword)
    }

    /// Returns all keywords.
    #[must_use]
    pub fn all(&self) -> Vec<&Keyword> {
        self.keywords.iter().collect()
    }

    /// Returns all root keywords (no parent).
    #[must_use]
    pub fn roots(&self) -> Vec<&Keyword> {
        self.keywords
            .iter()
            .filter(|k| k.parent().is_none())
            .collect()
    }

    /// Returns children of a given keyword.
    #[must_use]
    pub fn children(&self, parent: &Keyword) -> Vec<&Keyword> {
        self.keywords
            .iter()
            .filter(|k| k.parent().as_ref().is_some_and(|p| p.path == parent.path))
            .collect()
    }

    /// Returns the number of keywords.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keywords.len()
    }

    /// Checks if the collection is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keywords.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyword_creation() {
        let keyword = Keyword::new("People/John Doe");
        assert_eq!(keyword.path, "People/John Doe");
        assert_eq!(keyword.components, vec!["People", "John Doe"]);
        assert_eq!(keyword.leaf(), Some("John Doe"));
        assert_eq!(keyword.depth(), 2);
    }

    #[test]
    fn test_keyword_parent() {
        let keyword = Keyword::new("People/Actors/John Doe");
        let parent = keyword.parent().expect("parent should succeed");
        assert_eq!(parent.path, "People/Actors");

        let root = Keyword::new("People");
        assert!(root.parent().is_none());
    }

    #[test]
    fn test_keyword_descendant() {
        let parent = Keyword::new("People");
        let child = Keyword::new("People/John Doe");
        let grandchild = Keyword::new("People/John Doe/Interview");

        assert!(child.is_descendant_of(&parent));
        assert!(grandchild.is_descendant_of(&parent));
        assert!(grandchild.is_descendant_of(&child));
        assert!(!parent.is_descendant_of(&child));
    }

    #[test]
    fn test_keyword_collection() {
        let mut collection = KeywordCollection::new();
        collection.add(Keyword::new("People"));
        collection.add(Keyword::new("People/John Doe"));
        collection.add(Keyword::new("Locations"));

        assert_eq!(collection.len(), 3);
        assert_eq!(collection.roots().len(), 2);

        let people = Keyword::new("People");
        let children = collection.children(&people);
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].path, "People/John Doe");
    }
}
