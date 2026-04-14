//! Project snapshot and version management with branching, merging, and
//! fast-forward detection.
//!
//! Provides a git-inspired snapshot repository where every commit records its
//! parent, branches are named sequences of commits, and merges use BFS to
//! locate the common ancestor.

use crate::operation_log::Operation;
use crate::three_way_merge::MergeResult;
use crate::CollabError;
use std::collections::{HashMap, HashSet, VecDeque};

// ─────────────────────────────────────────────────────────────────────────────
// Domain types
// ─────────────────────────────────────────────────────────────────────────────

/// An immutable point-in-time snapshot of a project.
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// Unique identifier for this snapshot.
    pub id: u64,
    /// Wall-clock creation time in milliseconds.
    pub timestamp_ms: i64,
    /// User who created this snapshot.
    pub author_id: u32,
    /// Human-readable commit message.
    pub description: String,
    /// Hash of the project state at this point.
    pub state_hash: u64,
    /// Parent snapshot id (None for the root commit).
    pub parent_id: Option<u64>,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
}

/// A lightweight delta snapshot that stores the ops needed to reproduce the
/// full state from a base snapshot.
#[derive(Debug, Clone)]
pub struct DeltaSnapshot {
    /// The base snapshot this delta is relative to.
    pub base_id: u64,
    /// Operations that, applied to `base_id`, reproduce the current state.
    pub ops: Vec<Operation>,
    /// Unique identifier for this delta snapshot.
    pub id: u64,
}

/// A named branch tracking a linear sequence of commits.
#[derive(Debug, Clone)]
pub struct SnapshotBranch {
    /// Branch name (e.g. "main", "feature/audio-fx").
    pub name: String,
    /// The snapshot id at the tip of this branch.
    pub head_id: u64,
    /// Ordered list of snapshot ids from oldest to newest.
    pub commits: Vec<u64>,
}

impl SnapshotBranch {
    fn new(name: impl Into<String>, head_id: u64) -> Self {
        Self {
            name: name.into(),
            head_id,
            commits: vec![head_id],
        }
    }
}

/// The full snapshot repository.
#[derive(Debug)]
pub struct SnapshotRepository {
    /// All snapshots indexed by id.
    pub snapshots: HashMap<u64, Snapshot>,
    /// All branches indexed by name.
    pub branches: HashMap<String, SnapshotBranch>,
    /// The currently active branch.
    pub current_branch: String,
    /// Internal id counter.
    next_id: u64,
}

impl SnapshotRepository {
    /// Create a repository with an initial "main" branch containing a single
    /// root commit (state_hash = 0, author_id = 0).
    pub fn new() -> Self {
        let root = Snapshot {
            id: 1,
            timestamp_ms: 0,
            author_id: 0,
            description: "Initial commit".to_string(),
            state_hash: 0,
            parent_id: None,
            metadata: HashMap::new(),
        };
        let mut snapshots = HashMap::new();
        snapshots.insert(1, root);

        let branch = SnapshotBranch::new("main", 1);
        let mut branches = HashMap::new();
        branches.insert("main".to_string(), branch);

        Self {
            snapshots,
            branches,
            current_branch: "main".to_string(),
            next_id: 2,
        }
    }

    fn next_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

impl Default for SnapshotRepository {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Create a new snapshot on the current branch and return its id.
///
/// The new snapshot's `parent_id` is set to the current branch head.
pub fn commit(
    repo: &mut SnapshotRepository,
    description: String,
    author_id: u32,
    state_hash: u64,
) -> u64 {
    let id = repo.next_id();
    let parent_id = repo.branches.get(&repo.current_branch).map(|b| b.head_id);

    let snap = Snapshot {
        id,
        timestamp_ms: 0, // callers may override metadata["timestamp_ms"]
        author_id,
        description,
        state_hash,
        parent_id,
        metadata: HashMap::new(),
    };

    repo.snapshots.insert(id, snap);

    if let Some(branch) = repo.branches.get_mut(&repo.current_branch) {
        branch.head_id = id;
        branch.commits.push(id);
    }

    id
}

/// Create a new branch starting at the head of `from_branch`.
///
/// Returns `CollabError::InvalidOperation` if `from_branch` does not exist
/// or `branch_name` already exists.
pub fn branch_from(
    repo: &mut SnapshotRepository,
    branch_name: &str,
    from_branch: &str,
) -> Result<(), CollabError> {
    if repo.branches.contains_key(branch_name) {
        return Err(CollabError::InvalidOperation(format!(
            "branch '{}' already exists",
            branch_name
        )));
    }

    let head_id = repo
        .branches
        .get(from_branch)
        .ok_or_else(|| {
            CollabError::InvalidOperation(format!("branch '{}' not found", from_branch))
        })?
        .head_id;

    // Copy the parent's commit list up to the fork point, then set the new
    // branch head to the same commit.
    let parent_commits = repo
        .branches
        .get(from_branch)
        .map(|b| b.commits.clone())
        .unwrap_or_default();

    let new_branch = SnapshotBranch {
        name: branch_name.to_string(),
        head_id,
        commits: parent_commits,
    };

    repo.branches.insert(branch_name.to_string(), new_branch);
    Ok(())
}

/// Merge `source` branch into `target` branch.
///
/// - **Fast-forward**: if the target head is an ancestor of the source head,
///   the target head is simply advanced (no new commit created). Returns
///   `MergeResult::Clean(new_head_id)`.
/// - **True merge (non-fast-forward)**: a new merge commit is created on the
///   target branch with `parent_id = target_head`. The state_hash of the merge
///   commit is computed as `source_head_hash ^ target_head_hash` (a simple
///   XOR-based placeholder; production systems would re-render the merged
///   state). Returns `MergeResult::Clean(merge_commit_id)`.
/// - If either branch does not exist: returns
///   `MergeResult::Conflict(...)` wrapping a `CollabError`-style description
///   (we model it as a `MergeConflict<u64>` with path = error message).
///
/// This function never returns `MergeResult::Conflict` for differing content
/// (that is handled at the `ProjectMerger` level). It only returns
/// `MergeResult::Conflict` on structural errors.
pub fn merge_branch(repo: &mut SnapshotRepository, source: &str, target: &str) -> MergeResult<u64> {
    use crate::three_way_merge::MergeConflict;

    let source_head = match repo.branches.get(source) {
        Some(b) => b.head_id,
        None => {
            return MergeResult::Conflict(MergeConflict {
                path: format!("branch '{}'", source),
                base: 0,
                ours: 0,
                theirs: 0,
            });
        }
    };

    let target_head = match repo.branches.get(target) {
        Some(b) => b.head_id,
        None => {
            return MergeResult::Conflict(MergeConflict {
                path: format!("branch '{}'", target),
                base: 0,
                ours: 0,
                theirs: 0,
            });
        }
    };

    // ── Fast-forward check ───────────────────────────────────────────────────
    // If target_head is an ancestor of source_head, we can fast-forward.
    if is_ancestor(repo, target_head, source_head) {
        // Copy the source commits that are new to the target.
        let source_commits = repo
            .branches
            .get(source)
            .map(|b| b.commits.clone())
            .unwrap_or_default();

        if let Some(target_branch) = repo.branches.get_mut(target) {
            // Append only the commits that follow target_head.
            let split = source_commits
                .iter()
                .position(|&id| id == target_head)
                .map(|pos| pos + 1)
                .unwrap_or(0);
            for &id in &source_commits[split..] {
                if !target_branch.commits.contains(&id) {
                    target_branch.commits.push(id);
                }
            }
            target_branch.head_id = source_head;
        }

        return MergeResult::Clean(source_head);
    }

    // ── True merge commit ────────────────────────────────────────────────────
    let source_hash = repo
        .snapshots
        .get(&source_head)
        .map(|s| s.state_hash)
        .unwrap_or(0);
    let target_hash = repo
        .snapshots
        .get(&target_head)
        .map(|s| s.state_hash)
        .unwrap_or(0);

    let merge_hash = source_hash ^ target_hash;
    let merge_id = repo.next_id();

    let merge_snap = Snapshot {
        id: merge_id,
        timestamp_ms: 0,
        author_id: 0,
        description: format!("Merge '{}' into '{}'", source, target),
        state_hash: merge_hash,
        parent_id: Some(target_head),
        metadata: {
            let mut m = HashMap::new();
            m.insert("merge_source_head".to_string(), source_head.to_string());
            m
        },
    };

    repo.snapshots.insert(merge_id, merge_snap);

    if let Some(branch) = repo.branches.get_mut(target) {
        branch.head_id = merge_id;
        branch.commits.push(merge_id);
    }

    MergeResult::Clean(merge_id)
}

/// Return the commits on `branch` in chronological order (oldest first).
pub fn history<'r>(repo: &'r SnapshotRepository, branch: &str) -> Vec<&'r Snapshot> {
    let commits = match repo.branches.get(branch) {
        Some(b) => &b.commits,
        None => return Vec::new(),
    };

    commits
        .iter()
        .filter_map(|id| repo.snapshots.get(id))
        .collect()
}

/// Find the most recent common ancestor of `id1` and `id2` using BFS from
/// both sides meeting in the middle.
///
/// Returns `None` if no common ancestor exists (disconnected graph).
pub fn find_common_ancestor(repo: &SnapshotRepository, id1: u64, id2: u64) -> Option<u64> {
    // Collect the full ancestor set for id1.
    let ancestors1 = collect_ancestors(repo, id1);

    // BFS from id2; return the first node that appears in ancestors1.
    let mut queue: VecDeque<u64> = VecDeque::new();
    let mut visited: HashSet<u64> = HashSet::new();
    queue.push_back(id2);
    visited.insert(id2);

    while let Some(cur) = queue.pop_front() {
        if ancestors1.contains(&cur) {
            return Some(cur);
        }
        if let Some(snap) = repo.snapshots.get(&cur) {
            if let Some(parent) = snap.parent_id {
                if visited.insert(parent) {
                    queue.push_back(parent);
                }
            }
        }
    }

    None
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Return a set of all ancestors of `id` (including `id` itself).
fn collect_ancestors(repo: &SnapshotRepository, id: u64) -> HashSet<u64> {
    let mut visited = HashSet::new();
    let mut queue: VecDeque<u64> = VecDeque::new();
    queue.push_back(id);
    visited.insert(id);

    while let Some(cur) = queue.pop_front() {
        if let Some(snap) = repo.snapshots.get(&cur) {
            if let Some(parent) = snap.parent_id {
                if visited.insert(parent) {
                    queue.push_back(parent);
                }
            }
        }
    }

    visited
}

/// Return `true` if `candidate` is an ancestor of `descendant` (or equal).
fn is_ancestor(repo: &SnapshotRepository, candidate: u64, descendant: u64) -> bool {
    collect_ancestors(repo, descendant).contains(&candidate)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Snapshot basics ──────────────────────────────────────────────────────

    #[test]
    fn test_repo_initialises_with_root() {
        let repo = SnapshotRepository::new();
        assert!(repo.snapshots.contains_key(&1));
        assert!(repo.branches.contains_key("main"));
        assert_eq!(repo.current_branch, "main");
    }

    #[test]
    fn test_commit_creates_snapshot() {
        let mut repo = SnapshotRepository::new();
        let id = commit(&mut repo, "first".into(), 1, 0xAB);
        assert!(repo.snapshots.contains_key(&id));
        assert_eq!(repo.snapshots[&id].state_hash, 0xAB);
        assert_eq!(repo.branches["main"].head_id, id);
    }

    #[test]
    fn test_commit_sets_parent() {
        let mut repo = SnapshotRepository::new();
        let id1 = commit(&mut repo, "c1".into(), 1, 1);
        let id2 = commit(&mut repo, "c2".into(), 1, 2);
        assert_eq!(repo.snapshots[&id2].parent_id, Some(id1));
    }

    #[test]
    fn test_history_chronological() {
        let mut repo = SnapshotRepository::new();
        let id1 = commit(&mut repo, "c1".into(), 1, 1);
        let id2 = commit(&mut repo, "c2".into(), 1, 2);
        let h = history(&repo, "main");
        // root (1), id1, id2
        assert_eq!(h.len(), 3);
        assert_eq!(h[1].id, id1);
        assert_eq!(h[2].id, id2);
    }

    #[test]
    fn test_history_empty_branch() {
        let repo = SnapshotRepository::new();
        let h = history(&repo, "nonexistent");
        assert!(h.is_empty());
    }

    // ── branch_from ──────────────────────────────────────────────────────────

    #[test]
    fn test_branch_from_creates_branch() {
        let mut repo = SnapshotRepository::new();
        commit(&mut repo, "c1".into(), 1, 1);
        branch_from(&mut repo, "feature", "main").expect("branch_from should succeed");
        assert!(repo.branches.contains_key("feature"));
        assert_eq!(
            repo.branches["feature"].head_id,
            repo.branches["main"].head_id
        );
    }

    #[test]
    fn test_branch_from_duplicate_fails() {
        let mut repo = SnapshotRepository::new();
        branch_from(&mut repo, "dev", "main").expect("first branch_from should succeed");
        let result = branch_from(&mut repo, "dev", "main");
        assert!(result.is_err());
    }

    #[test]
    fn test_branch_from_nonexistent_source_fails() {
        let mut repo = SnapshotRepository::new();
        let result = branch_from(&mut repo, "new", "ghost");
        assert!(result.is_err());
    }

    // ── merge_branch: fast-forward ───────────────────────────────────────────

    #[test]
    fn test_merge_fast_forward() {
        let mut repo = SnapshotRepository::new();
        // main: root → c1
        let c1 = commit(&mut repo, "c1".into(), 1, 1);

        // Branch "dev" off main at c1, then add c2 on dev.
        branch_from(&mut repo, "dev", "main").expect("branch_from should succeed");
        repo.current_branch = "dev".to_string();
        let c2 = commit(&mut repo, "c2".into(), 1, 2);

        // Merge dev into main (fast-forward: main < dev).
        repo.current_branch = "main".to_string();
        let result = merge_branch(&mut repo, "dev", "main");
        assert!(result.is_clean(), "should be a clean fast-forward");
        assert_eq!(result.clean_value().copied(), Some(c2));
        assert_eq!(repo.branches["main"].head_id, c2);
        let _ = c1; // used to set up parent chain
    }

    #[test]
    fn test_merge_nonexistent_source_returns_conflict_variant() {
        let mut repo = SnapshotRepository::new();
        let result = merge_branch(&mut repo, "ghost", "main");
        assert!(result.is_conflict());
    }

    #[test]
    fn test_merge_nonexistent_target_returns_conflict_variant() {
        let mut repo = SnapshotRepository::new();
        let result = merge_branch(&mut repo, "main", "ghost");
        assert!(result.is_conflict());
    }

    // ── merge_branch: true merge ─────────────────────────────────────────────

    #[test]
    fn test_merge_true_merge_creates_commit() {
        let mut repo = SnapshotRepository::new();

        // main: root → m1
        let _m1 = commit(&mut repo, "m1".into(), 1, 0x10);

        // Branch feature from root (before m1), add f1 on feature.
        branch_from(&mut repo, "feature", "main").expect("should work");

        // Add another commit on main so histories diverge.
        let _m2 = commit(&mut repo, "m2".into(), 1, 0x20);

        // Add a commit on feature.
        repo.current_branch = "feature".to_string();
        let _f1 = commit(&mut repo, "f1".into(), 2, 0x30);

        // Now merge feature into main (diverged → true merge).
        repo.current_branch = "main".to_string();
        let result = merge_branch(&mut repo, "feature", "main");
        assert!(result.is_clean());
        let merge_id = result.clean_value().copied().expect("merge id");
        assert_eq!(repo.branches["main"].head_id, merge_id);
        let merge_snap = &repo.snapshots[&merge_id];
        assert!(merge_snap.description.contains("feature"));
    }

    // ── find_common_ancestor ─────────────────────────────────────────────────

    #[test]
    fn test_common_ancestor_linear() {
        let mut repo = SnapshotRepository::new();
        let c1 = commit(&mut repo, "c1".into(), 1, 1);
        let c2 = commit(&mut repo, "c2".into(), 1, 2);
        let ancestor = find_common_ancestor(&repo, c1, c2);
        assert_eq!(ancestor, Some(c1));
    }

    #[test]
    fn test_common_ancestor_same_node() {
        let mut repo = SnapshotRepository::new();
        let c1 = commit(&mut repo, "c1".into(), 1, 1);
        let ancestor = find_common_ancestor(&repo, c1, c1);
        assert_eq!(ancestor, Some(c1));
    }

    #[test]
    fn test_common_ancestor_diverged_branches() {
        let mut repo = SnapshotRepository::new();
        // root → c1 (shared)
        let c1 = commit(&mut repo, "c1".into(), 1, 1);

        // Branch A: c1 → a1
        branch_from(&mut repo, "branchA", "main").expect("ok");
        repo.current_branch = "branchA".to_string();
        let a1 = commit(&mut repo, "a1".into(), 1, 10);

        // Branch B: c1 → b1
        repo.current_branch = "main".to_string();
        branch_from(&mut repo, "branchB", "main").expect("ok");
        repo.current_branch = "branchB".to_string();
        let b1 = commit(&mut repo, "b1".into(), 2, 20);

        let ancestor = find_common_ancestor(&repo, a1, b1);
        // c1 is the fork point (both branches were created from main at c1).
        assert_eq!(ancestor, Some(c1));
    }

    #[test]
    fn test_common_ancestor_root() {
        let repo = SnapshotRepository::new();
        // root snapshot (id=1) has no parent; its own common ancestor with
        // itself is itself.
        let ancestor = find_common_ancestor(&repo, 1, 1);
        assert_eq!(ancestor, Some(1));
    }

    // ── DeltaSnapshot ────────────────────────────────────────────────────────

    #[test]
    fn test_delta_snapshot_stores_ops() {
        use crate::operation_log::{OpType, Operation};
        let op = Operation::new(
            1,
            1,
            0,
            "path",
            OpType::Insert {
                index: 0,
                value: 1.0,
            },
        );
        let ds = DeltaSnapshot {
            base_id: 1,
            ops: vec![op],
            id: 42,
        };
        assert_eq!(ds.ops.len(), 1);
        assert_eq!(ds.base_id, 1);
        assert_eq!(ds.id, 42);
    }

    // ── history order ────────────────────────────────────────────────────────

    #[test]
    fn test_history_includes_root() {
        let repo = SnapshotRepository::new();
        let h = history(&repo, "main");
        assert!(!h.is_empty());
        assert_eq!(h[0].id, 1);
    }

    #[test]
    fn test_multiple_commits_history() {
        let mut repo = SnapshotRepository::new();
        for i in 1u64..=5 {
            commit(&mut repo, format!("c{}", i), 1, i);
        }
        let h = history(&repo, "main");
        assert_eq!(h.len(), 6); // root + 5
        for (idx, snap) in h.iter().enumerate().skip(1) {
            assert_eq!(snap.description, format!("c{}", idx));
        }
    }

    // ── Branch creation and fast-forward/non-fast-forward tests ─────────────

    /// Creating a branch from a non-main branch preserves the fork-point head.
    #[test]
    fn test_branch_creation_from_feature_branch() {
        let mut repo = SnapshotRepository::new();
        let c1 = commit(&mut repo, "c1".into(), 1, 0xAA);

        // Create "feature" from "main".
        branch_from(&mut repo, "feature", "main").expect("feature branch");

        // Add a commit on feature.
        repo.current_branch = "feature".to_string();
        let f1 = commit(&mut repo, "f1".into(), 2, 0xBB);

        // Create "hotfix" from "feature".
        branch_from(&mut repo, "hotfix", "feature").expect("hotfix branch");

        assert_eq!(
            repo.branches["hotfix"].head_id, f1,
            "hotfix should start at feature head (f1)"
        );

        // hotfix commit history should include root, c1, and f1.
        let h = history(&repo, "hotfix");
        let ids: Vec<u64> = h.iter().map(|s| s.id).collect();
        assert!(ids.contains(&c1), "hotfix history should contain c1");
        assert!(ids.contains(&f1), "hotfix history should contain f1");
    }

    /// Fast-forward merge: target is a strict ancestor of source → head advances.
    #[test]
    fn test_fast_forward_detection_advances_head_without_new_commit() {
        let mut repo = SnapshotRepository::new();
        let before_count = repo.snapshots.len();

        // Branch "dev" from main.
        branch_from(&mut repo, "dev", "main").expect("dev branch");
        repo.current_branch = "dev".to_string();
        let d1 = commit(&mut repo, "d1".into(), 1, 0x11);

        // Fast-forward main to dev (main ≡ root < d1).
        repo.current_branch = "main".to_string();
        let result = merge_branch(&mut repo, "dev", "main");

        assert!(result.is_clean(), "must be a clean fast-forward");
        assert_eq!(
            result.clean_value().copied(),
            Some(d1),
            "fast-forward result must equal dev head"
        );
        assert_eq!(
            repo.branches["main"].head_id, d1,
            "main head should advance to d1"
        );
        // No new snapshot should have been created (snapshot count unchanged
        // by the merge — d1 was already in the repo).
        assert_eq!(
            repo.snapshots.len(),
            before_count + 1, // only d1 was added by commit()
            "fast-forward must not create an extra merge commit"
        );
    }

    /// Non-fast-forward (diverged branches): a new merge commit is created.
    #[test]
    fn test_non_fast_forward_detection_creates_merge_commit() {
        let mut repo = SnapshotRepository::new();

        // main: root → m1
        let _m1 = commit(&mut repo, "m1".into(), 1, 0x10);

        // Branch "side" from main.
        branch_from(&mut repo, "side", "main").expect("side branch");

        // Add m2 on main so histories diverge.
        let _m2 = commit(&mut repo, "m2".into(), 1, 0x20);
        let main_head_before = repo.branches["main"].head_id;

        // Add s1 on side.
        repo.current_branch = "side".to_string();
        let _s1 = commit(&mut repo, "s1".into(), 2, 0x30);

        // Merge side into main — should produce a real merge commit.
        repo.current_branch = "main".to_string();
        let snap_count_before = repo.snapshots.len();
        let result = merge_branch(&mut repo, "side", "main");

        assert!(result.is_clean(), "merge must succeed cleanly");
        let merge_id = result.clean_value().copied().expect("merge commit id");

        // A new snapshot should have been created.
        assert_eq!(
            repo.snapshots.len(),
            snap_count_before + 1,
            "non-fast-forward must create a merge commit snapshot"
        );

        // The merge commit's parent must be the old main head.
        let merge_snap = &repo.snapshots[&merge_id];
        assert_eq!(
            merge_snap.parent_id,
            Some(main_head_before),
            "merge commit parent must be the old main head"
        );

        // The merge commit description must reference "side".
        assert!(
            merge_snap.description.contains("side"),
            "merge description should mention source branch 'side'"
        );
    }

    /// Diverged branches: verify common ancestor is correct.
    #[test]
    fn test_diverged_branches_common_ancestor_is_fork_point() {
        let mut repo = SnapshotRepository::new();
        let fork = commit(&mut repo, "fork".into(), 1, 0xFF);

        branch_from(&mut repo, "left", "main").expect("left");
        branch_from(&mut repo, "right", "main").expect("right");

        repo.current_branch = "left".to_string();
        let l1 = commit(&mut repo, "l1".into(), 1, 0x01);

        repo.current_branch = "right".to_string();
        let r1 = commit(&mut repo, "r1".into(), 2, 0x02);

        let ancestor = find_common_ancestor(&repo, l1, r1);
        assert_eq!(
            ancestor,
            Some(fork),
            "common ancestor of l1 and r1 must be the fork point"
        );
    }

    /// After fast-forward, the merged branch shares all commits of the source.
    #[test]
    fn test_fast_forward_target_commits_include_source_commits() {
        let mut repo = SnapshotRepository::new();
        branch_from(&mut repo, "feat", "main").expect("feat branch");
        repo.current_branch = "feat".to_string();
        let f1 = commit(&mut repo, "f1".into(), 1, 1);
        let f2 = commit(&mut repo, "f2".into(), 1, 2);

        repo.current_branch = "main".to_string();
        let result = merge_branch(&mut repo, "feat", "main");
        assert!(result.is_clean());

        let main_commits: Vec<u64> = repo.branches["main"].commits.iter().copied().collect();
        assert!(main_commits.contains(&f1), "main must include f1 after ff");
        assert!(main_commits.contains(&f2), "main must include f2 after ff");
    }
}
