//! Conflict resolution strategies for collaborative video editing.
//!
//! Provides pluggable resolution strategies — last-write-wins, semantic merge,
//! and user-preference ordering — along with a `ConflictResolver` that applies
//! them to a queue of conflicting operations and records the resolution outcome.
//!
//! # Design overview
//!
//! * [`ResolutionStrategy`] — selectable algorithm for picking a winner.
//! * [`ConflictingOp`] — a pair of operations that touch the same resource.
//! * [`ResolutionOutcome`] — the result after applying a strategy.
//! * [`ConflictResolver`] — stateful resolver that tracks resolved conflicts.
//! * [`UserPreferenceMap`] — per-user priority weights used by the
//!   `UserPreference` strategy.

#![allow(dead_code)]

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Strategy
// ─────────────────────────────────────────────────────────────────────────────

/// Selectable conflict resolution algorithm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolutionStrategy {
    /// The operation with the higher wall-clock timestamp wins.
    LastWriteWins,
    /// Attempt a semantic three-way merge; fall back to `LastWriteWins` when
    /// the values cannot be merged automatically.
    SemanticMerge,
    /// Use a per-user priority table to decide which author's edit takes
    /// precedence.  Users with higher priority values win ties.
    UserPreference,
    /// The operation with the lower Lamport clock value wins (first-write
    /// wins).
    FirstWriteWins,
    /// Prefer the operation authored by the session owner.
    OwnerPriority,
}

impl std::fmt::Display for ResolutionStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::LastWriteWins => "last_write_wins",
            Self::SemanticMerge => "semantic_merge",
            Self::UserPreference => "user_preference",
            Self::FirstWriteWins => "first_write_wins",
            Self::OwnerPriority => "owner_priority",
        };
        write!(f, "{s}")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Operation type
// ─────────────────────────────────────────────────────────────────────────────

/// The kind of value an operation carries.
#[derive(Debug, Clone, PartialEq)]
pub enum OpValue {
    /// A scalar floating-point value (e.g. volume, opacity).
    Scalar(f64),
    /// A free-form text value (e.g. clip title, colour hex).
    Text(String),
    /// A raw byte blob (e.g. serialised keyframe data).
    Bytes(Vec<u8>),
}

/// A single edit operation submitted by a collaborator.
#[derive(Debug, Clone)]
pub struct EditOp {
    /// Unique identifier for this operation.
    pub id: u64,
    /// Author's user identifier.
    pub author_id: String,
    /// Whether the author owns the session.
    pub is_owner: bool,
    /// Lamport clock value at submission time.
    pub lamport: u64,
    /// Wall-clock timestamp in milliseconds (Unix epoch).
    pub wall_ms: u64,
    /// Identifier of the resource being edited (e.g. "track:3:gain").
    pub resource_id: String,
    /// The new value to apply.
    pub value: OpValue,
}

impl EditOp {
    /// Create a new edit operation.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: u64,
        author_id: impl Into<String>,
        is_owner: bool,
        lamport: u64,
        wall_ms: u64,
        resource_id: impl Into<String>,
        value: OpValue,
    ) -> Self {
        Self {
            id,
            author_id: author_id.into(),
            is_owner,
            lamport,
            wall_ms,
            resource_id: resource_id.into(),
            value,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Conflicting pair
// ─────────────────────────────────────────────────────────────────────────────

/// A pair of operations that conflict because they modify the same resource
/// concurrently.
#[derive(Debug, Clone)]
pub struct ConflictingOp {
    /// The "local" operation (first to arrive at this node).
    pub local: EditOp,
    /// The "remote" operation (arrived later or from a remote peer).
    pub remote: EditOp,
}

impl ConflictingOp {
    /// Create a new conflicting-op pair.
    pub fn new(local: EditOp, remote: EditOp) -> Self {
        Self { local, remote }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Outcome
// ─────────────────────────────────────────────────────────────────────────────

/// The side that was chosen as the winner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Winner {
    Local,
    Remote,
    /// Both sides were merged into a new value (only possible with
    /// `SemanticMerge`).
    Merged,
}

impl std::fmt::Display for Winner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Local => "local",
            Self::Remote => "remote",
            Self::Merged => "merged",
        };
        write!(f, "{s}")
    }
}

/// The outcome of resolving a single conflict.
#[derive(Debug, Clone)]
pub struct ResolutionOutcome {
    /// Identifier of the conflicting operation pair (derived from the local op id).
    pub conflict_id: u64,
    /// Resource that was in conflict.
    pub resource_id: String,
    /// Which side won (or was merged).
    pub winner: Winner,
    /// The winning (or merged) value to apply.
    pub resolved_value: OpValue,
    /// Strategy that was used.
    pub strategy_used: ResolutionStrategy,
    /// Human-readable description of the resolution decision.
    pub reason: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// User preference map
// ─────────────────────────────────────────────────────────────────────────────

/// Per-user priority weights for the `UserPreference` strategy.
///
/// Higher weights indicate higher priority.  Users not listed default to
/// weight `0`.
#[derive(Debug, Clone, Default)]
pub struct UserPreferenceMap {
    weights: HashMap<String, i64>,
}

impl UserPreferenceMap {
    /// Create an empty preference map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a user's priority weight.
    pub fn set(&mut self, user_id: impl Into<String>, weight: i64) {
        self.weights.insert(user_id.into(), weight);
    }

    /// Get a user's priority weight (default `0`).
    pub fn get(&self, user_id: &str) -> i64 {
        self.weights.get(user_id).copied().unwrap_or(0)
    }

    /// Remove a user's entry.
    pub fn remove(&mut self, user_id: &str) {
        self.weights.remove(user_id);
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Resolver
// ─────────────────────────────────────────────────────────────────────────────

/// Error type for resolver operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolverError {
    /// The two operations do not share the same resource.
    ResourceMismatch { local: String, remote: String },
}

impl std::fmt::Display for ResolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ResourceMismatch { local, remote } => {
                write!(
                    f,
                    "resource mismatch: local={local:?} remote={remote:?}"
                )
            }
        }
    }
}

impl std::error::Error for ResolverError {}

/// Stateful conflict resolver.
///
/// Maintains a log of resolved conflicts for audit purposes and applies the
/// configured strategy to each new conflict.
#[derive(Debug)]
pub struct ConflictResolver {
    /// Active resolution strategy.
    strategy: ResolutionStrategy,
    /// Per-user weights (consulted by [`ResolutionStrategy::UserPreference`]).
    user_prefs: UserPreferenceMap,
    /// Cumulative log of resolution outcomes.
    resolution_log: Vec<ResolutionOutcome>,
}

impl ConflictResolver {
    /// Create a new resolver with the given strategy.
    pub fn new(strategy: ResolutionStrategy) -> Self {
        Self {
            strategy,
            user_prefs: UserPreferenceMap::new(),
            resolution_log: Vec::new(),
        }
    }

    /// Create a resolver using `UserPreference` strategy with a pre-built map.
    pub fn with_user_preferences(user_prefs: UserPreferenceMap) -> Self {
        Self {
            strategy: ResolutionStrategy::UserPreference,
            user_prefs,
            resolution_log: Vec::new(),
        }
    }

    /// Replace the current strategy.
    pub fn set_strategy(&mut self, strategy: ResolutionStrategy) {
        self.strategy = strategy;
    }

    /// Return the current strategy.
    pub fn strategy(&self) -> &ResolutionStrategy {
        &self.strategy
    }

    /// Return a reference to the user-preference map.
    pub fn user_prefs(&self) -> &UserPreferenceMap {
        &self.user_prefs
    }

    /// Return a mutable reference to the user-preference map.
    pub fn user_prefs_mut(&mut self) -> &mut UserPreferenceMap {
        &mut self.user_prefs
    }

    /// Resolve a single conflict and record the outcome.
    ///
    /// Returns an error if the two operations do not target the same resource.
    pub fn resolve(&mut self, conflict: ConflictingOp) -> Result<ResolutionOutcome, ResolverError> {
        if conflict.local.resource_id != conflict.remote.resource_id {
            return Err(ResolverError::ResourceMismatch {
                local: conflict.local.resource_id.clone(),
                remote: conflict.remote.resource_id.clone(),
            });
        }

        let outcome = match &self.strategy {
            ResolutionStrategy::LastWriteWins => {
                self.resolve_last_write_wins(&conflict)
            }
            ResolutionStrategy::FirstWriteWins => {
                self.resolve_first_write_wins(&conflict)
            }
            ResolutionStrategy::SemanticMerge => {
                self.resolve_semantic_merge(&conflict)
            }
            ResolutionStrategy::UserPreference => {
                self.resolve_user_preference(&conflict)
            }
            ResolutionStrategy::OwnerPriority => {
                self.resolve_owner_priority(&conflict)
            }
        };

        self.resolution_log.push(outcome.clone());
        Ok(outcome)
    }

    /// Resolve a batch of conflicts and return all outcomes.
    pub fn resolve_batch(
        &mut self,
        conflicts: Vec<ConflictingOp>,
    ) -> Vec<Result<ResolutionOutcome, ResolverError>> {
        conflicts.into_iter().map(|c| self.resolve(c)).collect()
    }

    /// Return a view of the resolution log.
    pub fn log(&self) -> &[ResolutionOutcome] {
        &self.resolution_log
    }

    /// Number of resolved conflicts.
    pub fn resolved_count(&self) -> usize {
        self.resolution_log.len()
    }

    /// Clear the resolution log.
    pub fn clear_log(&mut self) {
        self.resolution_log.clear();
    }

    // ── private helpers ────────────────────────────────────────────────────

    fn resolve_last_write_wins(&self, c: &ConflictingOp) -> ResolutionOutcome {
        let (winner, value, reason) = if c.remote.wall_ms >= c.local.wall_ms {
            (
                Winner::Remote,
                c.remote.value.clone(),
                format!(
                    "remote wall_ms ({}) >= local wall_ms ({})",
                    c.remote.wall_ms, c.local.wall_ms
                ),
            )
        } else {
            (
                Winner::Local,
                c.local.value.clone(),
                format!(
                    "local wall_ms ({}) > remote wall_ms ({})",
                    c.local.wall_ms, c.remote.wall_ms
                ),
            )
        };
        ResolutionOutcome {
            conflict_id: c.local.id,
            resource_id: c.local.resource_id.clone(),
            winner,
            resolved_value: value,
            strategy_used: ResolutionStrategy::LastWriteWins,
            reason,
        }
    }

    fn resolve_first_write_wins(&self, c: &ConflictingOp) -> ResolutionOutcome {
        let (winner, value, reason) = if c.local.lamport <= c.remote.lamport {
            (
                Winner::Local,
                c.local.value.clone(),
                format!(
                    "local lamport ({}) <= remote lamport ({})",
                    c.local.lamport, c.remote.lamport
                ),
            )
        } else {
            (
                Winner::Remote,
                c.remote.value.clone(),
                format!(
                    "remote lamport ({}) < local lamport ({})",
                    c.remote.lamport, c.local.lamport
                ),
            )
        };
        ResolutionOutcome {
            conflict_id: c.local.id,
            resource_id: c.local.resource_id.clone(),
            winner,
            resolved_value: value,
            strategy_used: ResolutionStrategy::FirstWriteWins,
            reason,
        }
    }

    fn resolve_semantic_merge(&self, c: &ConflictingOp) -> ResolutionOutcome {
        // Attempt element-wise merge for Scalar values by averaging.
        // For all other types fall through to last-write-wins.
        match (&c.local.value, &c.remote.value) {
            (OpValue::Scalar(a), OpValue::Scalar(b)) => {
                let merged = (a + b) / 2.0;
                ResolutionOutcome {
                    conflict_id: c.local.id,
                    resource_id: c.local.resource_id.clone(),
                    winner: Winner::Merged,
                    resolved_value: OpValue::Scalar(merged),
                    strategy_used: ResolutionStrategy::SemanticMerge,
                    reason: format!("merged scalars {a} and {b} → {merged}"),
                }
            }
            _ => {
                // Non-scalar: fall back to last-write-wins.
                let mut outcome = self.resolve_last_write_wins(c);
                outcome.strategy_used = ResolutionStrategy::SemanticMerge;
                outcome.reason = format!(
                    "semantic merge fallback (non-scalar): {}",
                    outcome.reason
                );
                outcome
            }
        }
    }

    fn resolve_user_preference(&self, c: &ConflictingOp) -> ResolutionOutcome {
        let local_prio = self.user_prefs.get(&c.local.author_id);
        let remote_prio = self.user_prefs.get(&c.remote.author_id);

        let (winner, value, reason) = if local_prio >= remote_prio {
            (
                Winner::Local,
                c.local.value.clone(),
                format!(
                    "local user {} priority ({local_prio}) >= remote user {} priority ({remote_prio})",
                    c.local.author_id, c.remote.author_id
                ),
            )
        } else {
            (
                Winner::Remote,
                c.remote.value.clone(),
                format!(
                    "remote user {} priority ({remote_prio}) > local user {} priority ({local_prio})",
                    c.remote.author_id, c.local.author_id
                ),
            )
        };
        ResolutionOutcome {
            conflict_id: c.local.id,
            resource_id: c.local.resource_id.clone(),
            winner,
            resolved_value: value,
            strategy_used: ResolutionStrategy::UserPreference,
            reason,
        }
    }

    fn resolve_owner_priority(&self, c: &ConflictingOp) -> ResolutionOutcome {
        let (winner, value, reason) = match (c.local.is_owner, c.remote.is_owner) {
            (true, false) => (
                Winner::Local,
                c.local.value.clone(),
                "local author is session owner".to_string(),
            ),
            (false, true) => (
                Winner::Remote,
                c.remote.value.clone(),
                "remote author is session owner".to_string(),
            ),
            _ => {
                // Neither or both are owners — fall back to LWW.
                let mut outcome = self.resolve_last_write_wins(c);
                outcome.strategy_used = ResolutionStrategy::OwnerPriority;
                outcome.reason = format!(
                    "owner_priority fallback (no clear owner): {}",
                    outcome.reason
                );
                return outcome;
            }
        };
        ResolutionOutcome {
            conflict_id: c.local.id,
            resource_id: c.local.resource_id.clone(),
            winner,
            resolved_value: value,
            strategy_used: ResolutionStrategy::OwnerPriority,
            reason,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper builders.

    fn scalar_op(id: u64, author: &str, is_owner: bool, lamport: u64, wall_ms: u64, val: f64) -> EditOp {
        EditOp::new(
            id,
            author,
            is_owner,
            lamport,
            wall_ms,
            "track:1:gain",
            OpValue::Scalar(val),
        )
    }

    fn text_op(id: u64, author: &str, wall_ms: u64, text: &str) -> EditOp {
        EditOp::new(
            id,
            author,
            false,
            0,
            wall_ms,
            "clip:1:title",
            OpValue::Text(text.to_string()),
        )
    }

    fn conflict(local: EditOp, remote: EditOp) -> ConflictingOp {
        ConflictingOp::new(local, remote)
    }

    // ── LastWriteWins ──────────────────────────────────────────────────────

    #[test]
    fn test_lww_remote_wins_on_higher_wall_ms() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::LastWriteWins);
        let c = conflict(
            scalar_op(1, "alice", false, 1, 1000, 0.8),
            scalar_op(2, "bob", false, 2, 2000, 0.5),
        );
        let outcome = resolver.resolve(c).expect("resolution should succeed");
        assert_eq!(outcome.winner, Winner::Remote);
        assert_eq!(outcome.resolved_value, OpValue::Scalar(0.5));
    }

    #[test]
    fn test_lww_local_wins_on_higher_wall_ms() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::LastWriteWins);
        let c = conflict(
            scalar_op(1, "alice", false, 2, 3000, 0.8),
            scalar_op(2, "bob", false, 1, 1000, 0.5),
        );
        let outcome = resolver.resolve(c).expect("resolution should succeed");
        assert_eq!(outcome.winner, Winner::Local);
        assert_eq!(outcome.resolved_value, OpValue::Scalar(0.8));
    }

    // ── FirstWriteWins ─────────────────────────────────────────────────────

    #[test]
    fn test_fww_local_wins_on_lower_lamport() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::FirstWriteWins);
        let c = conflict(
            scalar_op(1, "alice", false, 5, 9000, 0.3),
            scalar_op(2, "bob", false, 10, 1000, 0.9),
        );
        let outcome = resolver.resolve(c).expect("resolution should succeed");
        assert_eq!(outcome.winner, Winner::Local);
    }

    #[test]
    fn test_fww_remote_wins_on_lower_lamport() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::FirstWriteWins);
        let c = conflict(
            scalar_op(1, "alice", false, 10, 1000, 0.3),
            scalar_op(2, "bob", false, 3, 9000, 0.9),
        );
        let outcome = resolver.resolve(c).expect("resolution should succeed");
        assert_eq!(outcome.winner, Winner::Remote);
    }

    // ── SemanticMerge ──────────────────────────────────────────────────────

    #[test]
    fn test_semantic_merge_scalars_averaged() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::SemanticMerge);
        let c = conflict(
            scalar_op(1, "alice", false, 1, 1000, 0.4),
            scalar_op(2, "bob", false, 2, 2000, 0.8),
        );
        let outcome = resolver.resolve(c).expect("resolution should succeed");
        assert_eq!(outcome.winner, Winner::Merged);
        match outcome.resolved_value {
            OpValue::Scalar(v) => assert!((v - 0.6).abs() < 1e-10, "expected ~0.6, got {v}"),
            other => panic!("expected Scalar, got {other:?}"),
        }
    }

    #[test]
    fn test_semantic_merge_text_falls_back_to_lww() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::SemanticMerge);
        let c = conflict(
            text_op(1, "alice", 1000, "old title"),
            text_op(2, "bob", 5000, "new title"),
        );
        let outcome = resolver.resolve(c).expect("resolution should succeed");
        // LWW fallback: higher wall_ms wins → remote
        assert_eq!(outcome.winner, Winner::Remote);
        assert_eq!(
            outcome.resolved_value,
            OpValue::Text("new title".to_string())
        );
        assert_eq!(outcome.strategy_used, ResolutionStrategy::SemanticMerge);
    }

    // ── UserPreference ─────────────────────────────────────────────────────

    #[test]
    fn test_user_preference_higher_prio_wins() {
        let mut prefs = UserPreferenceMap::new();
        prefs.set("director", 100);
        prefs.set("assistant", 10);
        let mut resolver = ConflictResolver::with_user_preferences(prefs);

        let mut local = scalar_op(1, "assistant", false, 1, 1000, 0.3);
        local.resource_id = "track:1:gain".to_string();
        let mut remote = scalar_op(2, "director", false, 2, 500, 0.9);
        remote.resource_id = "track:1:gain".to_string();

        let outcome = resolver
            .resolve(ConflictingOp::new(local, remote))
            .expect("resolution should succeed");
        assert_eq!(outcome.winner, Winner::Remote); // director wins
    }

    // ── OwnerPriority ──────────────────────────────────────────────────────

    #[test]
    fn test_owner_priority_owner_wins() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::OwnerPriority);
        let c = conflict(
            scalar_op(1, "alice", true, 5, 1000, 1.0),  // alice is owner
            scalar_op(2, "bob", false, 3, 9000, 0.0),
        );
        let outcome = resolver.resolve(c).expect("resolution should succeed");
        assert_eq!(outcome.winner, Winner::Local);
    }

    #[test]
    fn test_owner_priority_remote_owner_wins() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::OwnerPriority);
        let c = conflict(
            scalar_op(1, "alice", false, 5, 1000, 1.0),
            scalar_op(2, "bob", true, 3, 9000, 0.0), // bob is owner
        );
        let outcome = resolver.resolve(c).expect("resolution should succeed");
        assert_eq!(outcome.winner, Winner::Remote);
    }

    // ── Error handling ─────────────────────────────────────────────────────

    #[test]
    fn test_resource_mismatch_returns_error() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::LastWriteWins);
        let local = EditOp::new(1, "a", false, 1, 1000, "res:1", OpValue::Scalar(0.5));
        let remote = EditOp::new(2, "b", false, 2, 2000, "res:2", OpValue::Scalar(0.9));
        let result = resolver.resolve(ConflictingOp::new(local, remote));
        assert!(result.is_err());
        let err = result.expect_err("should be an error");
        assert!(matches!(err, ResolverError::ResourceMismatch { .. }));
    }

    // ── Log ────────────────────────────────────────────────────────────────

    #[test]
    fn test_resolution_log_grows() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::LastWriteWins);
        for i in 0u64..5 {
            let local = scalar_op(i * 2, "a", false, i, i * 1000, 0.1);
            let remote = scalar_op(i * 2 + 1, "b", false, i + 1, i * 1000 + 500, 0.9);
            resolver
                .resolve(ConflictingOp::new(local, remote))
                .expect("resolution should succeed");
        }
        assert_eq!(resolver.resolved_count(), 5);
    }

    #[test]
    fn test_resolution_log_cleared() {
        let mut resolver = ConflictResolver::new(ResolutionStrategy::LastWriteWins);
        let c = conflict(
            scalar_op(1, "a", false, 1, 1000, 0.1),
            scalar_op(2, "b", false, 2, 2000, 0.9),
        );
        resolver.resolve(c).expect("resolution should succeed");
        assert_eq!(resolver.resolved_count(), 1);
        resolver.clear_log();
        assert_eq!(resolver.resolved_count(), 0);
    }

    // ── UserPreferenceMap ──────────────────────────────────────────────────

    #[test]
    fn test_user_preference_map_default_zero() {
        let map = UserPreferenceMap::new();
        assert_eq!(map.get("unknown_user"), 0);
    }

    #[test]
    fn test_user_preference_map_set_and_remove() {
        let mut map = UserPreferenceMap::new();
        map.set("alice", 50);
        assert_eq!(map.get("alice"), 50);
        map.remove("alice");
        assert_eq!(map.get("alice"), 0);
    }

    // ── Strategy display ───────────────────────────────────────────────────

    #[test]
    fn test_strategy_display() {
        assert_eq!(
            ResolutionStrategy::LastWriteWins.to_string(),
            "last_write_wins"
        );
        assert_eq!(
            ResolutionStrategy::SemanticMerge.to_string(),
            "semantic_merge"
        );
    }
}
