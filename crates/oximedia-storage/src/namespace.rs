#![allow(dead_code)]
//! Namespace management for hierarchical object storage organization.

use std::collections::HashMap;

/// A storage namespace — a logical prefix/container grouping.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Namespace {
    /// Namespace identifier (e.g. "media", "media/video").
    name: String,
}

impl Namespace {
    /// Create a new namespace.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    /// Validate namespace name: non-empty, only alphanumeric/hyphen/underscore/slash.
    pub fn is_valid(&self) -> bool {
        if self.name.is_empty() {
            return false;
        }
        self.name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == '/')
    }

    /// Namespace name string.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Parent namespace, if any.
    pub fn parent(&self) -> Option<Namespace> {
        let trimmed = self.name.trim_end_matches('/');
        trimmed
            .rfind('/')
            .map(|pos| Namespace::new(&trimmed[..pos]))
    }

    /// Depth of nesting (number of `/` separators + 1).
    pub fn depth(&self) -> usize {
        self.name.trim_end_matches('/').matches('/').count() + 1
    }
}

impl std::fmt::Display for Namespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// A key within a namespace.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NamespacedKey {
    namespace: Namespace,
    key: String,
}

impl NamespacedKey {
    /// Create a new namespaced key.
    pub fn new(namespace: Namespace, key: impl Into<String>) -> Self {
        Self {
            namespace,
            key: key.into(),
        }
    }

    /// Full storage path: `<namespace>/<key>`.
    pub fn path(&self) -> String {
        format!("{}/{}", self.namespace.name(), self.key)
    }

    /// The raw key component (without namespace prefix).
    pub fn key(&self) -> &str {
        &self.key
    }

    /// The namespace this key belongs to.
    pub fn namespace(&self) -> &Namespace {
        &self.namespace
    }
}

impl std::fmt::Display for NamespacedKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.path())
    }
}

/// Metadata stored alongside a namespace.
#[derive(Debug, Clone, Default)]
pub struct NamespaceMeta {
    /// Human-readable description.
    pub description: String,
    /// Whether the namespace is read-only.
    pub read_only: bool,
    /// Maximum object count allowed (0 = unlimited).
    pub max_objects: usize,
    /// Current object count.
    pub object_count: usize,
}

/// Manages a collection of namespaces.
#[derive(Debug, Default)]
pub struct NamespaceManager {
    namespaces: HashMap<String, NamespaceMeta>,
}

impl NamespaceManager {
    /// Create a new manager with no namespaces.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a namespace. Returns `false` if it already exists or is invalid.
    pub fn create(&mut self, ns: &Namespace, meta: NamespaceMeta) -> bool {
        if !ns.is_valid() || self.namespaces.contains_key(ns.name()) {
            return false;
        }
        self.namespaces.insert(ns.name().to_owned(), meta);
        true
    }

    /// Check whether a namespace exists.
    pub fn exists(&self, ns: &Namespace) -> bool {
        self.namespaces.contains_key(ns.name())
    }

    /// Delete a namespace. Returns `false` if it did not exist.
    pub fn delete(&mut self, ns: &Namespace) -> bool {
        self.namespaces.remove(ns.name()).is_some()
    }

    /// List all namespace names.
    pub fn list_namespaces(&self) -> Vec<String> {
        let mut names: Vec<String> = self.namespaces.keys().cloned().collect();
        names.sort();
        names
    }

    /// Retrieve metadata for a namespace.
    pub fn meta(&self, ns: &Namespace) -> Option<&NamespaceMeta> {
        self.namespaces.get(ns.name())
    }

    /// Retrieve mutable metadata.
    pub fn meta_mut(&mut self, ns: &Namespace) -> Option<&mut NamespaceMeta> {
        self.namespaces.get_mut(ns.name())
    }

    /// Increment object count in a namespace.
    pub fn increment_count(&mut self, ns: &Namespace) -> bool {
        if let Some(meta) = self.namespaces.get_mut(ns.name()) {
            meta.object_count += 1;
            true
        } else {
            false
        }
    }

    /// Total number of managed namespaces.
    pub fn count(&self) -> usize {
        self.namespaces.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_namespace_is_valid() {
        assert!(Namespace::new("media").is_valid());
        assert!(Namespace::new("media/video").is_valid());
        assert!(Namespace::new("my-ns_01").is_valid());
        assert!(!Namespace::new("").is_valid());
        assert!(!Namespace::new("bad name!").is_valid());
    }

    #[test]
    fn test_namespace_name() {
        let ns = Namespace::new("media/audio");
        assert_eq!(ns.name(), "media/audio");
    }

    #[test]
    fn test_namespace_parent_some() {
        let ns = Namespace::new("media/video/raw");
        let parent = ns.parent().expect("parent should exist");
        assert_eq!(parent.name(), "media/video");
    }

    #[test]
    fn test_namespace_parent_none() {
        let ns = Namespace::new("toplevel");
        assert!(ns.parent().is_none());
    }

    #[test]
    fn test_namespace_depth() {
        assert_eq!(Namespace::new("a").depth(), 1);
        assert_eq!(Namespace::new("a/b").depth(), 2);
        assert_eq!(Namespace::new("a/b/c").depth(), 3);
    }

    #[test]
    fn test_namespaced_key_path() {
        let ns = Namespace::new("media/video");
        let nk = NamespacedKey::new(ns, "clip001.mp4");
        assert_eq!(nk.path(), "media/video/clip001.mp4");
    }

    #[test]
    fn test_namespaced_key_components() {
        let ns = Namespace::new("archive");
        let nk = NamespacedKey::new(ns.clone(), "tape001.tar");
        assert_eq!(nk.key(), "tape001.tar");
        assert_eq!(nk.namespace(), &ns);
    }

    #[test]
    fn test_manager_create_and_exists() {
        let mut mgr = NamespaceManager::new();
        let ns = Namespace::new("docs");
        assert!(mgr.create(&ns, NamespaceMeta::default()));
        assert!(mgr.exists(&ns));
    }

    #[test]
    fn test_manager_duplicate_create() {
        let mut mgr = NamespaceManager::new();
        let ns = Namespace::new("dupe");
        assert!(mgr.create(&ns, NamespaceMeta::default()));
        assert!(!mgr.create(&ns, NamespaceMeta::default())); // second attempt fails
    }

    #[test]
    fn test_manager_invalid_namespace_rejected() {
        let mut mgr = NamespaceManager::new();
        let bad = Namespace::new("bad ns!");
        assert!(!mgr.create(&bad, NamespaceMeta::default()));
    }

    #[test]
    fn test_manager_delete() {
        let mut mgr = NamespaceManager::new();
        let ns = Namespace::new("temp");
        mgr.create(&ns, NamespaceMeta::default());
        assert!(mgr.delete(&ns));
        assert!(!mgr.exists(&ns));
        assert!(!mgr.delete(&ns)); // deleting again returns false
    }

    #[test]
    fn test_manager_list_namespaces_sorted() {
        let mut mgr = NamespaceManager::new();
        mgr.create(&Namespace::new("zzz"), NamespaceMeta::default());
        mgr.create(&Namespace::new("aaa"), NamespaceMeta::default());
        mgr.create(&Namespace::new("mmm"), NamespaceMeta::default());
        let list = mgr.list_namespaces();
        assert_eq!(list, vec!["aaa", "mmm", "zzz"]);
    }

    #[test]
    fn test_manager_increment_count() {
        let mut mgr = NamespaceManager::new();
        let ns = Namespace::new("cnt");
        mgr.create(&ns, NamespaceMeta::default());
        mgr.increment_count(&ns);
        mgr.increment_count(&ns);
        let meta = mgr.meta(&ns).expect("meta should succeed");
        assert_eq!(meta.object_count, 2);
    }

    #[test]
    fn test_namespace_display() {
        let ns = Namespace::new("hello/world");
        assert_eq!(ns.to_string(), "hello/world");
    }

    #[test]
    fn test_namespaced_key_display() {
        let ns = Namespace::new("ns");
        let nk = NamespacedKey::new(ns, "file.bin");
        assert_eq!(nk.to_string(), "ns/file.bin");
    }
}
