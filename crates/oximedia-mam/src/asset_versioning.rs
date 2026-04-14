//! Asset version tree with fork, merge, changelog, and semantic version tracking.
//!
//! Each asset can have a lineage of versions forming a directed acyclic graph
//! (DAG).  A version may be forked from any existing node, and two diverged
//! forks can be merged back into a new node that records both parents.
//!
//! # Key concepts
//!
//! * `AssetVersion` – a single immutable snapshot of an asset at a point in time.
//! * `SemanticVersion` – `major.minor.patch` following SemVer conventions.
//! * `VersionChangeType` – classifies the nature of the change (breaking, feature, fix, …).
//! * `VersionChangelogEntry` – a human-readable note describing what changed.
//! * `VersionTree` – an in-memory graph of all versions for a single asset.
//! * `VersionRegistry` – maps asset IDs to their `VersionTree`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// SemanticVersion
// ---------------------------------------------------------------------------

/// A `major.minor.patch` semantic version.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SemanticVersion {
    /// Breaking-change counter.
    pub major: u32,
    /// Backward-compatible feature counter.
    pub minor: u32,
    /// Backward-compatible bug-fix counter.
    pub patch: u32,
    /// Optional pre-release label (e.g. `"rc.1"`).
    pub pre_release: Option<String>,
}

impl SemanticVersion {
    /// Create a new semantic version.
    #[must_use]
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
            pre_release: None,
        }
    }

    /// Initial version `1.0.0`.
    #[must_use]
    pub const fn initial() -> Self {
        Self::new(1, 0, 0)
    }

    /// Return a new version with the patch number incremented.
    #[must_use]
    pub fn bump_patch(&self) -> Self {
        Self::new(self.major, self.minor, self.patch + 1)
    }

    /// Return a new version with the minor number incremented and patch reset.
    #[must_use]
    pub fn bump_minor(&self) -> Self {
        Self::new(self.major, self.minor + 1, 0)
    }

    /// Return a new version with the major number incremented and minor/patch reset.
    #[must_use]
    pub fn bump_major(&self) -> Self {
        Self::new(self.major + 1, 0, 0)
    }

    /// Set a pre-release label.
    #[must_use]
    pub fn with_pre_release(mut self, label: impl Into<String>) -> Self {
        self.pre_release = Some(label.into());
        self
    }
}

impl std::fmt::Display for SemanticVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if let Some(pre) = &self.pre_release {
            write!(f, "-{pre}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// VersionChangeType
// ---------------------------------------------------------------------------

/// The nature of the change introduced by a new version.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VersionChangeType {
    /// A breaking change in content or metadata contract.
    Breaking,
    /// A new feature or content addition that is backward-compatible.
    Feature,
    /// A small fix or correction.
    Fix,
    /// Metadata-only update (no content change).
    MetadataOnly,
    /// Content replacement (e.g. new encode of the same scene).
    ContentReplace,
    /// A merge of two diverged forks.
    Merge,
    /// A fork point (start of a new branch from an existing version).
    Fork,
    /// Other or unspecified change.
    Other(String),
}

impl VersionChangeType {
    /// Whether this change type is considered breaking.
    #[must_use]
    pub fn is_breaking(&self) -> bool {
        matches!(self, Self::Breaking)
    }
}

// ---------------------------------------------------------------------------
// VersionChangelogEntry
// ---------------------------------------------------------------------------

/// A single changelog entry describing what changed between two versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionChangelogEntry {
    /// Unique id of this entry.
    pub id: Uuid,
    /// Version this entry is attached to.
    pub version_id: Uuid,
    /// Author who created this entry.
    pub author_id: Uuid,
    /// Classification of the change.
    pub change_type: VersionChangeType,
    /// Human-readable description of the change.
    pub summary: String,
    /// Optional detailed body (Markdown).
    pub body: Option<String>,
    /// When the entry was recorded.
    pub recorded_at: DateTime<Utc>,
}

impl VersionChangelogEntry {
    /// Create a new changelog entry.
    #[must_use]
    pub fn new(
        version_id: Uuid,
        author_id: Uuid,
        change_type: VersionChangeType,
        summary: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            version_id,
            author_id,
            change_type,
            summary: summary.into(),
            body: None,
            recorded_at: Utc::now(),
        }
    }

    /// Attach a detailed body to the entry.
    #[must_use]
    pub fn with_body(mut self, body: impl Into<String>) -> Self {
        self.body = Some(body.into());
        self
    }
}

// ---------------------------------------------------------------------------
// AssetVersion
// ---------------------------------------------------------------------------

/// The status of a version in its lifecycle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VersionStatus {
    /// Currently active / in use.
    Active,
    /// Superseded by a later version but still accessible.
    Superseded,
    /// Explicitly deprecated — consumers should migrate.
    Deprecated,
    /// Archived to cold storage.
    Archived,
    /// Draft — not yet published.
    Draft,
}

/// Immutable snapshot of an asset at a specific point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetVersion {
    /// Unique id of this version node.
    pub id: Uuid,
    /// The asset this version belongs to.
    pub asset_id: Uuid,
    /// Semantic version label.
    pub semver: SemanticVersion,
    /// Display label (e.g. `"v2.1.0 – final grade"`).
    pub label: String,
    /// Parent version(s).  One parent = linear progression; two = merge node.
    pub parent_ids: Vec<Uuid>,
    /// Lifecycle status.
    pub status: VersionStatus,
    /// Storage path / URI for this version's content.
    pub storage_uri: Option<String>,
    /// Content hash for integrity verification (SHA-256 hex).
    pub content_hash: Option<String>,
    /// Arbitrary key-value metadata snapshot for this version.
    pub metadata: HashMap<String, String>,
    /// ID of the user who created this version.
    pub created_by: Uuid,
    /// When this version was created.
    pub created_at: DateTime<Utc>,
    /// Optional short description of why this version was created.
    pub description: Option<String>,
    /// Whether this is tagged as a release (pinned) version.
    pub is_release: bool,
}

impl AssetVersion {
    /// Create a new version node with no parents (root of the tree).
    #[must_use]
    pub fn new_root(
        asset_id: Uuid,
        semver: SemanticVersion,
        label: impl Into<String>,
        created_by: Uuid,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            asset_id,
            semver,
            label: label.into(),
            parent_ids: vec![],
            status: VersionStatus::Active,
            storage_uri: None,
            content_hash: None,
            metadata: HashMap::new(),
            created_by,
            created_at: Utc::now(),
            description: None,
            is_release: false,
        }
    }

    /// Create a new version derived from a single parent (linear progression).
    #[must_use]
    pub fn new_child(
        parent: &AssetVersion,
        semver: SemanticVersion,
        label: impl Into<String>,
        created_by: Uuid,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            asset_id: parent.asset_id,
            semver,
            label: label.into(),
            parent_ids: vec![parent.id],
            status: VersionStatus::Active,
            storage_uri: None,
            content_hash: None,
            metadata: HashMap::new(),
            created_by,
            created_at: Utc::now(),
            description: None,
            is_release: false,
        }
    }

    /// Create a merge version from two parents.
    #[must_use]
    pub fn new_merge(
        asset_id: Uuid,
        parent_a: Uuid,
        parent_b: Uuid,
        semver: SemanticVersion,
        label: impl Into<String>,
        created_by: Uuid,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            asset_id,
            semver,
            label: label.into(),
            parent_ids: vec![parent_a, parent_b],
            status: VersionStatus::Active,
            storage_uri: None,
            content_hash: None,
            metadata: HashMap::new(),
            created_by,
            created_at: Utc::now(),
            description: None,
            is_release: false,
        }
    }

    /// Mark this version as a pinned release.
    pub fn mark_as_release(&mut self) {
        self.is_release = true;
    }

    /// Deprecate this version.
    pub fn deprecate(&mut self) {
        self.status = VersionStatus::Deprecated;
    }
}

// ---------------------------------------------------------------------------
// VersionTree
// ---------------------------------------------------------------------------

/// Error type for version tree operations.
#[derive(Debug, thiserror::Error)]
pub enum VersionTreeError {
    /// A version with this ID already exists in the tree.
    #[error("Version already exists: {0}")]
    AlreadyExists(Uuid),
    /// No version found for the given ID.
    #[error("Version not found: {0}")]
    NotFound(Uuid),
    /// The asset IDs of the merge parents do not match.
    #[error("Merge parent asset mismatch: expected {expected}, got {got}")]
    MergeAssetMismatch { expected: Uuid, got: Uuid },
    /// Cannot fork from a deprecated or archived version.
    #[error("Cannot fork from version with status {0:?}")]
    InvalidForkSource(VersionStatus),
}

/// A DAG of `AssetVersion` nodes for a single asset.
pub struct VersionTree {
    /// Asset this tree belongs to.
    pub asset_id: Uuid,
    /// All version nodes keyed by their ID.
    versions: HashMap<Uuid, AssetVersion>,
    /// Changelog entries keyed by version ID.
    changelog: HashMap<Uuid, Vec<VersionChangelogEntry>>,
    /// ID of the latest active ("head") version.
    head_id: Option<Uuid>,
}

impl VersionTree {
    /// Create a new empty version tree for an asset.
    #[must_use]
    pub fn new(asset_id: Uuid) -> Self {
        Self {
            asset_id,
            versions: HashMap::new(),
            changelog: HashMap::new(),
            head_id: None,
        }
    }

    /// Insert a version root (no parents).
    ///
    /// # Errors
    ///
    /// Returns [`VersionTreeError::AlreadyExists`] if a version with the same ID is already present.
    pub fn add_root(&mut self, version: AssetVersion) -> Result<(), VersionTreeError> {
        if self.versions.contains_key(&version.id) {
            return Err(VersionTreeError::AlreadyExists(version.id));
        }
        let vid = version.id;
        self.versions.insert(vid, version);
        self.head_id = Some(vid);
        Ok(())
    }

    /// Fork an existing version into a new child version.
    ///
    /// The forked version is given the next patch version by default.
    ///
    /// # Errors
    ///
    /// Returns an error if the parent version is not found or has an invalid status
    /// for forking, or if a version with the generated ID already exists.
    pub fn fork(
        &mut self,
        parent_id: Uuid,
        label: impl Into<String>,
        created_by: Uuid,
    ) -> Result<Uuid, VersionTreeError> {
        let parent = self
            .versions
            .get(&parent_id)
            .ok_or(VersionTreeError::NotFound(parent_id))?;

        if matches!(
            parent.status,
            VersionStatus::Archived | VersionStatus::Deprecated
        ) {
            return Err(VersionTreeError::InvalidForkSource(parent.status.clone()));
        }

        let new_semver = parent.semver.bump_minor();
        let new_version = AssetVersion::new_child(parent, new_semver, label, created_by);
        let new_id = new_version.id;
        self.versions.insert(new_id, new_version);
        // Update head to the new fork tip
        self.head_id = Some(new_id);
        Ok(new_id)
    }

    /// Merge two diverged versions into a new merge node.
    ///
    /// # Errors
    ///
    /// Returns an error if either parent is not found or the asset IDs do not match.
    pub fn merge(
        &mut self,
        parent_a_id: Uuid,
        parent_b_id: Uuid,
        semver: SemanticVersion,
        label: impl Into<String>,
        created_by: Uuid,
    ) -> Result<Uuid, VersionTreeError> {
        let (asset_id_a, asset_id_b) = {
            let a = self
                .versions
                .get(&parent_a_id)
                .ok_or(VersionTreeError::NotFound(parent_a_id))?;
            let b = self
                .versions
                .get(&parent_b_id)
                .ok_or(VersionTreeError::NotFound(parent_b_id))?;
            (a.asset_id, b.asset_id)
        };
        if asset_id_a != asset_id_b {
            return Err(VersionTreeError::MergeAssetMismatch {
                expected: asset_id_a,
                got: asset_id_b,
            });
        }

        let merge_version =
            AssetVersion::new_merge(asset_id_a, parent_a_id, parent_b_id, semver, label, created_by);
        let merge_id = merge_version.id;
        self.versions.insert(merge_id, merge_version);
        self.head_id = Some(merge_id);
        Ok(merge_id)
    }

    /// Append a changelog entry to a specific version.
    ///
    /// # Errors
    ///
    /// Returns [`VersionTreeError::NotFound`] if the version does not exist.
    pub fn add_changelog_entry(
        &mut self,
        entry: VersionChangelogEntry,
    ) -> Result<(), VersionTreeError> {
        if !self.versions.contains_key(&entry.version_id) {
            return Err(VersionTreeError::NotFound(entry.version_id));
        }
        self.changelog
            .entry(entry.version_id)
            .or_default()
            .push(entry);
        Ok(())
    }

    /// Retrieve a version by ID.
    #[must_use]
    pub fn get(&self, id: &Uuid) -> Option<&AssetVersion> {
        self.versions.get(id)
    }

    /// Retrieve a mutable reference to a version.
    pub fn get_mut(&mut self, id: &Uuid) -> Option<&mut AssetVersion> {
        self.versions.get_mut(id)
    }

    /// All versions in insertion order (non-deterministic iteration).
    #[must_use]
    pub fn all_versions(&self) -> Vec<&AssetVersion> {
        self.versions.values().collect()
    }

    /// Changelog entries for a specific version.
    #[must_use]
    pub fn changelog_for(&self, version_id: &Uuid) -> &[VersionChangelogEntry] {
        self.changelog
            .get(version_id)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// The current head version (most recently added).
    #[must_use]
    pub fn head(&self) -> Option<&AssetVersion> {
        self.head_id.and_then(|id| self.versions.get(&id))
    }

    /// Collect the linear ancestry path from `version_id` up to the root.
    ///
    /// For merge nodes the first parent is followed (left-biased traversal).
    /// Returns the path from root to `version_id`.
    #[must_use]
    pub fn ancestry_path(&self, version_id: Uuid) -> Vec<Uuid> {
        let mut path = Vec::new();
        let mut current = version_id;
        loop {
            path.push(current);
            match self.versions.get(&current).and_then(|v| v.parent_ids.first()) {
                Some(&parent) => current = parent,
                None => break,
            }
        }
        path.reverse();
        path
    }

    /// Count the total number of version nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.versions.len()
    }

    /// Whether the tree has no versions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.versions.is_empty()
    }

    /// Return all release-tagged versions sorted by semver.
    #[must_use]
    pub fn releases(&self) -> Vec<&AssetVersion> {
        let mut releases: Vec<&AssetVersion> =
            self.versions.values().filter(|v| v.is_release).collect();
        releases.sort_by(|a, b| a.semver.cmp(&b.semver));
        releases
    }
}

// ---------------------------------------------------------------------------
// VersionRegistry
// ---------------------------------------------------------------------------

/// Registry mapping asset IDs to their `VersionTree`.
#[derive(Default)]
pub struct VersionRegistry {
    trees: HashMap<Uuid, VersionTree>,
}

impl VersionRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create a `VersionTree` for an asset.
    pub fn tree_for(&mut self, asset_id: Uuid) -> &mut VersionTree {
        self.trees
            .entry(asset_id)
            .or_insert_with(|| VersionTree::new(asset_id))
    }

    /// Look up an existing tree (immutable).
    #[must_use]
    pub fn get_tree(&self, asset_id: &Uuid) -> Option<&VersionTree> {
        self.trees.get(asset_id)
    }

    /// Number of assets tracked in the registry.
    #[must_use]
    pub fn asset_count(&self) -> usize {
        self.trees.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_user() -> Uuid {
        Uuid::new_v4()
    }

    fn make_asset() -> Uuid {
        Uuid::new_v4()
    }

    #[test]
    fn semver_display_and_bump() {
        let v = SemanticVersion::new(1, 2, 3);
        assert_eq!(v.to_string(), "1.2.3");

        let patched = v.bump_patch();
        assert_eq!(patched.to_string(), "1.2.4");

        let minored = v.bump_minor();
        assert_eq!(minored.to_string(), "1.3.0");

        let majored = v.bump_major();
        assert_eq!(majored.to_string(), "2.0.0");
    }

    #[test]
    fn semver_pre_release() {
        let v = SemanticVersion::initial().with_pre_release("rc.1");
        assert_eq!(v.to_string(), "1.0.0-rc.1");
        assert!(v.pre_release.is_some());
    }

    #[test]
    fn version_tree_root_and_head() {
        let asset_id = make_asset();
        let user_id = make_user();
        let mut tree = VersionTree::new(asset_id);

        let root = AssetVersion::new_root(asset_id, SemanticVersion::initial(), "v1.0.0", user_id);
        let root_id = root.id;
        tree.add_root(root).expect("add_root should succeed");

        assert_eq!(tree.len(), 1);
        assert_eq!(tree.head().map(|v| v.id), Some(root_id));
    }

    #[test]
    fn version_tree_fork_creates_child() {
        let asset_id = make_asset();
        let user_id = make_user();
        let mut tree = VersionTree::new(asset_id);

        let root = AssetVersion::new_root(asset_id, SemanticVersion::initial(), "v1.0.0", user_id);
        let root_id = root.id;
        tree.add_root(root).expect("add_root should succeed");

        let fork_id = tree
            .fork(root_id, "v1.1.0 – colour grade", user_id)
            .expect("fork should succeed");

        assert_eq!(tree.len(), 2);
        let fork = tree.get(&fork_id).expect("fork version should exist");
        assert_eq!(fork.parent_ids, vec![root_id]);
        assert_eq!(fork.semver.to_string(), "1.1.0");
    }

    #[test]
    fn version_tree_merge_two_forks() {
        let asset_id = make_asset();
        let user_id = make_user();
        let mut tree = VersionTree::new(asset_id);

        let root = AssetVersion::new_root(asset_id, SemanticVersion::initial(), "root", user_id);
        let root_id = root.id;
        tree.add_root(root).unwrap();

        let fork_a = tree.fork(root_id, "fork-a", user_id).unwrap();
        let fork_b = tree.fork(root_id, "fork-b", user_id).unwrap();

        let merge_semver = SemanticVersion::new(2, 0, 0);
        let merge_id = tree
            .merge(fork_a, fork_b, merge_semver.clone(), "merged", user_id)
            .expect("merge should succeed");

        let merged = tree.get(&merge_id).expect("merge version should exist");
        assert_eq!(merged.parent_ids.len(), 2);
        assert_eq!(merged.semver, merge_semver);
    }

    #[test]
    fn ancestry_path_follows_first_parent() {
        let asset_id = make_asset();
        let user_id = make_user();
        let mut tree = VersionTree::new(asset_id);

        let root = AssetVersion::new_root(asset_id, SemanticVersion::initial(), "root", user_id);
        let root_id = root.id;
        tree.add_root(root).unwrap();
        let child_id = tree.fork(root_id, "child", user_id).unwrap();
        let grandchild_id = tree.fork(child_id, "grandchild", user_id).unwrap();

        let path = tree.ancestry_path(grandchild_id);
        assert_eq!(path, vec![root_id, child_id, grandchild_id]);
    }

    #[test]
    fn changelog_entry_lifecycle() {
        let asset_id = make_asset();
        let user_id = make_user();
        let mut tree = VersionTree::new(asset_id);

        let root = AssetVersion::new_root(asset_id, SemanticVersion::initial(), "root", user_id);
        let root_id = root.id;
        tree.add_root(root).unwrap();

        let entry = VersionChangelogEntry::new(
            root_id,
            user_id,
            VersionChangeType::Fix,
            "Fixed colour cast in the shadows",
        );
        tree.add_changelog_entry(entry).expect("add_changelog_entry should succeed");

        let entries = tree.changelog_for(&root_id);
        assert_eq!(entries.len(), 1);
        assert!(matches!(entries[0].change_type, VersionChangeType::Fix));
    }

    #[test]
    fn release_tagging_and_listing() {
        let asset_id = make_asset();
        let user_id = make_user();
        let mut tree = VersionTree::new(asset_id);

        let mut root =
            AssetVersion::new_root(asset_id, SemanticVersion::initial(), "v1.0.0", user_id);
        root.mark_as_release();
        let root_id = root.id;
        tree.add_root(root).unwrap();

        let child_id = tree.fork(root_id, "v1.1.0 – WIP", user_id).unwrap();
        // child is NOT a release

        let releases = tree.releases();
        assert_eq!(releases.len(), 1);
        assert_eq!(releases[0].id, root_id);

        // Now mark child as release
        tree.get_mut(&child_id).unwrap().mark_as_release();
        assert_eq!(tree.releases().len(), 2);
    }

    #[test]
    fn registry_creates_trees_on_demand() {
        let mut registry = VersionRegistry::new();
        let asset_a = make_asset();
        let asset_b = make_asset();

        // Access creates a new tree
        let _ = registry.tree_for(asset_a);
        let _ = registry.tree_for(asset_b);

        assert_eq!(registry.asset_count(), 2);
        assert!(registry.get_tree(&asset_a).is_some());
    }

    #[test]
    fn fork_from_deprecated_fails() {
        let asset_id = make_asset();
        let user_id = make_user();
        let mut tree = VersionTree::new(asset_id);

        let mut root =
            AssetVersion::new_root(asset_id, SemanticVersion::initial(), "v1.0.0", user_id);
        root.deprecate();
        let root_id = root.id;
        tree.add_root(root).unwrap();

        let result = tree.fork(root_id, "should-fail", user_id);
        assert!(result.is_err());
    }
}
