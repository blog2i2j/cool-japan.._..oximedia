//! Clip version management.
//!
//! Provides a tree-based version history system for clips, allowing branching
//! edit histories similar to a version-control system.

#![allow(dead_code)]

/// A single version node in the version tree.
#[derive(Debug, Clone)]
pub struct ClipVersion {
    /// Unique version identifier (assigned by `VersionTree`).
    pub id: u64,
    /// The clip this version belongs to.
    pub clip_id: String,
    /// Sequential version number within this clip's history.
    pub version_number: u32,
    /// Human-readable description of the changes in this version.
    pub description: String,
    /// Unix timestamp (ms) when this version was created.
    pub created_ms: u64,
    /// ID of the parent version, or `None` for a root version.
    pub parent_version: Option<u64>,
}

impl ClipVersion {
    /// Return `true` if this version has no parent (i.e., it is a root).
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.parent_version.is_none()
    }

    /// Return how many milliseconds old this version is given a current time.
    #[must_use]
    pub fn age_ms(&self, now_ms: u64) -> u64 {
        now_ms.saturating_sub(self.created_ms)
    }
}

/// A tree of [`ClipVersion`] nodes supporting branching histories.
#[derive(Debug, Clone, Default)]
pub struct VersionTree {
    /// All version nodes stored in this tree.
    pub versions: Vec<ClipVersion>,
    /// Next ID to assign (auto-incremented).
    pub next_id: u64,
}

impl VersionTree {
    /// Create an empty version tree.
    #[must_use]
    pub fn new() -> Self {
        Self {
            versions: Vec::new(),
            next_id: 1,
        }
    }

    /// Create a root version for the given clip and return its ID.
    pub fn create_root(&mut self, clip_id: &str, desc: &str, now_ms: u64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.versions.push(ClipVersion {
            id,
            clip_id: clip_id.to_owned(),
            version_number: 1,
            description: desc.to_owned(),
            created_ms: now_ms,
            parent_version: None,
        });
        id
    }

    /// Create a branch version whose parent is `parent_id`.
    ///
    /// Returns `None` if `parent_id` does not exist in this tree.
    pub fn create_branch(&mut self, parent_id: u64, desc: &str, now_ms: u64) -> Option<u64> {
        let parent = self.versions.iter().find(|v| v.id == parent_id)?;
        let clip_id = parent.clip_id.clone();
        let version_number = parent.version_number + 1;

        let id = self.next_id;
        self.next_id += 1;
        self.versions.push(ClipVersion {
            id,
            clip_id,
            version_number,
            description: desc.to_owned(),
            created_ms: now_ms,
            parent_version: Some(parent_id),
        });
        Some(id)
    }

    /// Return all ancestors of `version_id`, from immediate parent to root.
    ///
    /// Returns an empty `Vec` if `version_id` does not exist or has no parent.
    #[must_use]
    pub fn ancestors(&self, version_id: u64) -> Vec<&ClipVersion> {
        let mut result = Vec::new();
        let mut current_id = version_id;

        loop {
            let version = match self.versions.iter().find(|v| v.id == current_id) {
                Some(v) => v,
                None => break,
            };
            match version.parent_version {
                None => break,
                Some(pid) => {
                    if let Some(parent) = self.versions.iter().find(|v| v.id == pid) {
                        result.push(parent);
                        current_id = pid;
                    } else {
                        break;
                    }
                }
            }
        }

        result
    }

    /// Return all descendants of `version_id` (direct and indirect children).
    ///
    /// Returns an empty `Vec` if `version_id` does not exist or has no children.
    #[must_use]
    pub fn descendants(&self, version_id: u64) -> Vec<&ClipVersion> {
        let mut result = Vec::new();
        let mut frontier = vec![version_id];

        while let Some(current) = frontier.pop() {
            for v in &self.versions {
                if v.parent_version == Some(current) {
                    result.push(v);
                    frontier.push(v.id);
                }
            }
        }

        result
    }

    /// Return all "tip" versions — those with no children.
    #[must_use]
    pub fn tip_versions(&self) -> Vec<&ClipVersion> {
        self.versions
            .iter()
            .filter(|v| {
                !self
                    .versions
                    .iter()
                    .any(|other| other.parent_version == Some(v.id))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tree() -> (VersionTree, u64) {
        let mut tree = VersionTree::new();
        let root = tree.create_root("clip_1", "Initial cut", 1_000);
        (tree, root)
    }

    // --- ClipVersion ---

    #[test]
    fn test_is_root_true() {
        let (tree, root) = make_tree();
        let v = tree
            .versions
            .iter()
            .find(|v| v.id == root)
            .expect("iter should succeed");
        assert!(v.is_root());
    }

    #[test]
    fn test_is_root_false_for_branch() {
        let (mut tree, root) = make_tree();
        let branch = tree
            .create_branch(root, "colour grade", 2_000)
            .expect("create_branch should succeed");
        let v = tree
            .versions
            .iter()
            .find(|v| v.id == branch)
            .expect("iter should succeed");
        assert!(!v.is_root());
    }

    #[test]
    fn test_age_ms_normal() {
        let (tree, root) = make_tree();
        let v = tree
            .versions
            .iter()
            .find(|v| v.id == root)
            .expect("iter should succeed");
        assert_eq!(v.age_ms(5_000), 4_000);
    }

    #[test]
    fn test_age_ms_future_creation() {
        let (tree, root) = make_tree();
        let v = tree
            .versions
            .iter()
            .find(|v| v.id == root)
            .expect("iter should succeed");
        // now_ms < created_ms → saturating sub returns 0
        assert_eq!(v.age_ms(0), 0);
    }

    // --- VersionTree ---

    #[test]
    fn test_create_root_returns_id() {
        let mut tree = VersionTree::new();
        let id = tree.create_root("clip_x", "root", 0);
        assert!(id >= 1);
    }

    #[test]
    fn test_create_branch_valid_parent() {
        let (mut tree, root) = make_tree();
        let branch = tree.create_branch(root, "branch", 2_000);
        assert!(branch.is_some());
    }

    #[test]
    fn test_create_branch_invalid_parent() {
        let mut tree = VersionTree::new();
        let branch = tree.create_branch(999, "orphan", 0);
        assert!(branch.is_none());
    }

    #[test]
    fn test_ancestors_single_parent() {
        let (mut tree, root) = make_tree();
        let b1 = tree
            .create_branch(root, "b1", 2_000)
            .expect("create_branch should succeed");
        let ancestors = tree.ancestors(b1);
        assert_eq!(ancestors.len(), 1);
        assert_eq!(ancestors[0].id, root);
    }

    #[test]
    fn test_ancestors_chain() {
        let (mut tree, root) = make_tree();
        let b1 = tree
            .create_branch(root, "b1", 2_000)
            .expect("create_branch should succeed");
        let b2 = tree
            .create_branch(b1, "b2", 3_000)
            .expect("create_branch should succeed");
        let ancestors = tree.ancestors(b2);
        // Should return b1 then root
        assert_eq!(ancestors.len(), 2);
        assert_eq!(ancestors[0].id, b1);
        assert_eq!(ancestors[1].id, root);
    }

    #[test]
    fn test_ancestors_root_has_none() {
        let (tree, root) = make_tree();
        let ancestors = tree.ancestors(root);
        assert!(ancestors.is_empty());
    }

    #[test]
    fn test_descendants_from_root() {
        let (mut tree, root) = make_tree();
        let b1 = tree
            .create_branch(root, "b1", 2_000)
            .expect("create_branch should succeed");
        let b2 = tree
            .create_branch(root, "b2", 2_500)
            .expect("create_branch should succeed");
        let b1b = tree
            .create_branch(b1, "b1b", 3_000)
            .expect("create_branch should succeed");
        let desc = tree.descendants(root);
        let ids: Vec<u64> = desc.iter().map(|v| v.id).collect();
        assert!(ids.contains(&b1));
        assert!(ids.contains(&b2));
        assert!(ids.contains(&b1b));
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn test_descendants_leaf_is_empty() {
        let (tree, root) = make_tree();
        let desc = tree.descendants(root);
        // root has no children yet
        assert!(desc.is_empty());
    }

    #[test]
    fn test_tip_versions_only_root() {
        let (tree, root) = make_tree();
        let tips = tree.tip_versions();
        assert_eq!(tips.len(), 1);
        assert_eq!(tips[0].id, root);
    }

    #[test]
    fn test_tip_versions_after_branching() {
        let (mut tree, root) = make_tree();
        let b1 = tree
            .create_branch(root, "b1", 2_000)
            .expect("create_branch should succeed");
        let b2 = tree
            .create_branch(root, "b2", 2_500)
            .expect("create_branch should succeed");
        let tips = tree.tip_versions();
        // root now has children; b1 and b2 are tips
        let tip_ids: Vec<u64> = tips.iter().map(|v| v.id).collect();
        assert!(!tip_ids.contains(&root));
        assert!(tip_ids.contains(&b1));
        assert!(tip_ids.contains(&b2));
    }

    #[test]
    fn test_version_numbers_increment() {
        let (mut tree, root) = make_tree();
        let root_v = tree
            .versions
            .iter()
            .find(|v| v.id == root)
            .expect("iter should succeed");
        assert_eq!(root_v.version_number, 1);
        let branch = tree
            .create_branch(root, "v2", 0)
            .expect("create_branch should succeed");
        let branch_v = tree
            .versions
            .iter()
            .find(|v| v.id == branch)
            .expect("iter should succeed");
        assert_eq!(branch_v.version_number, 2);
    }
}
