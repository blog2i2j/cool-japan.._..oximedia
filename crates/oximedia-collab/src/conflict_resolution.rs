//! Conflict resolution policies for collaborative video editing.
//!
//! Extends the lower-level [`crate::conflict_resolve`] module with a
//! higher-level policy layer:
//!
//! * [`ConflictPolicy`] — selectable automatic resolution strategies.
//! * [`ConflictDetector`] — identifies simultaneous / overlapping edits.
//! * [`ConflictReport`] — structured summary with resolution recommendation.
//! * [`PolicyEngine`] — applies the active policy and produces reports.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ─────────────────────────────────────────────────────────────────────────────
// Re-use key types from conflict_resolve
// ─────────────────────────────────────────────────────────────────────────────

/// A logical-clock value (Lamport timestamp).
pub type LogicalClock = u64;

/// A unique identifier for an edit operation.
pub type EditId = Uuid;

/// A region of the timeline identified by track and time range (ms).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Region {
    /// Track identifier.
    pub track_id: String,
    /// Start time in milliseconds.
    pub start_ms: i64,
    /// End time in milliseconds.
    pub end_ms: i64,
}

impl Region {
    /// Create a new region.
    pub fn new(track_id: impl Into<String>, start_ms: i64, end_ms: i64) -> Self {
        Self {
            track_id: track_id.into(),
            start_ms,
            end_ms,
        }
    }

    /// Whether this region overlaps `other` on the same track.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.track_id == other.track_id
            && self.start_ms < other.end_ms
            && other.start_ms < self.end_ms
    }

    /// Duration of this region in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> i64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// Lightweight description of an edit for conflict analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditDescriptor {
    /// Unique operation ID.
    pub id: EditId,
    /// Submitting user.
    pub author: Uuid,
    /// Lamport clock at submission time.
    pub clock: LogicalClock,
    /// Wall-clock timestamp in milliseconds (Unix epoch).
    pub wall_ms: u64,
    /// Affected timeline region.
    pub region: Region,
    /// Free-form operation label (e.g. "Trim", "Gain", "Cut").
    pub label: String,
}

impl EditDescriptor {
    /// Create a new edit descriptor.
    pub fn new(
        author: Uuid,
        clock: LogicalClock,
        wall_ms: u64,
        region: Region,
        label: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            author,
            clock,
            wall_ms,
            region,
            label: label.into(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ConflictPolicy
// ─────────────────────────────────────────────────────────────────────────────

/// Policy that governs how simultaneous edits to the same region are resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictPolicy {
    /// The edit with the highest Lamport clock wins; ties broken by wall-clock.
    LastWriteWins,
    /// The edit with the lowest Lamport clock wins; ties broken by wall-clock.
    FirstWriteWins,
    /// Neither edit is auto-applied; a human reviewer must choose.
    ManualReview,
    /// Attempt a content-level merge; fall back to [`ManualReview`](Self::ManualReview)
    /// if the merge produces ambiguity.
    MergeStrategy,
    /// The edit originating from a user with higher priority (e.g. Owner)
    /// always wins.  Falls back to [`LastWriteWins`](Self::LastWriteWins) if
    /// both users have equal priority.
    RolePriority,
}

impl ConflictPolicy {
    /// Human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::LastWriteWins => "LastWriteWins",
            Self::FirstWriteWins => "FirstWriteWins",
            Self::ManualReview => "ManualReview",
            Self::MergeStrategy => "MergeStrategy",
            Self::RolePriority => "RolePriority",
        }
    }

    /// Whether this policy ever requires human intervention.
    #[must_use]
    pub fn may_require_manual_review(self) -> bool {
        matches!(self, Self::ManualReview | Self::MergeStrategy)
    }

    /// Whether this policy is fully deterministic (no ambiguity possible).
    #[must_use]
    pub fn is_deterministic(self) -> bool {
        matches!(self, Self::LastWriteWins | Self::FirstWriteWins)
    }
}

impl std::fmt::Display for ConflictPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Resolution outcome
// ─────────────────────────────────────────────────────────────────────────────

/// The outcome of applying a [`ConflictPolicy`] to a detected conflict.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResolutionOutcome {
    /// The existing (already-applied) edit wins; the incoming edit is dropped.
    ExistingWins,
    /// The incoming edit wins; the existing edit is superseded.
    IncomingWins,
    /// Both edits are applied without modification (KeepBoth scenario).
    Merged,
    /// No automatic resolution; a reviewer must decide.
    PendingReview,
}

impl std::fmt::Display for ResolutionOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExistingWins => write!(f, "ExistingWins"),
            Self::IncomingWins => write!(f, "IncomingWins"),
            Self::Merged => write!(f, "Merged"),
            Self::PendingReview => write!(f, "PendingReview"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ConflictReport
// ─────────────────────────────────────────────────────────────────────────────

/// Severity of a detected conflict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ConflictSeverity {
    /// Edits overlap but affect independent attributes; merge is usually safe.
    Low,
    /// Edits compete for the same attribute; one must be dropped or reconciled.
    Medium,
    /// Edits are structurally incompatible (e.g. delete vs modify).
    High,
}

impl std::fmt::Display for ConflictSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "low"),
            Self::Medium => write!(f, "medium"),
            Self::High => write!(f, "high"),
        }
    }
}

/// A structured conflict report produced by the [`PolicyEngine`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictReport {
    /// Unique report ID.
    pub id: Uuid,
    /// The edit that was already applied to shared state.
    pub existing: EditDescriptor,
    /// The edit that arrived and conflicted with `existing`.
    pub incoming: EditDescriptor,
    /// Policy that was in effect when the conflict was detected.
    pub policy: ConflictPolicy,
    /// Computed resolution outcome.
    pub outcome: ResolutionOutcome,
    /// Assessed severity.
    pub severity: ConflictSeverity,
    /// Human-readable recommendation explaining the chosen outcome.
    pub recommendation: String,
    /// Wall-clock timestamp when the report was generated (ms since Unix epoch).
    pub generated_at_ms: u64,
}

impl ConflictReport {
    /// Whether this conflict still needs manual attention.
    #[must_use]
    pub fn needs_review(&self) -> bool {
        self.outcome == ResolutionOutcome::PendingReview
    }

    /// One-line summary suitable for a notification or audit log.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "[{severity}] {policy}: {existing_label} (clock {ec}) vs {incoming_label} (clock {ic}) on {track}[{s}–{e}ms] → {outcome}",
            severity = self.severity,
            policy = self.policy,
            existing_label = self.existing.label,
            ec = self.existing.clock,
            incoming_label = self.incoming.label,
            ic = self.incoming.clock,
            track = self.existing.region.track_id,
            s = self.existing.region.start_ms,
            e = self.existing.region.end_ms,
            outcome = self.outcome,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ConflictDetector
// ─────────────────────────────────────────────────────────────────────────────

/// Detects conflicts between an incoming edit and the set of already-applied
/// edits for each track.
#[derive(Debug, Default)]
pub struct ConflictDetector {
    /// Applied edits per track.
    applied: HashMap<String, Vec<EditDescriptor>>,
}

impl ConflictDetector {
    /// Create a new detector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an edit as applied (no conflict checking).
    pub fn register(&mut self, edit: EditDescriptor) {
        self.applied
            .entry(edit.region.track_id.clone())
            .or_default()
            .push(edit);
    }

    /// Find all previously-applied edits that overlap with `incoming`.
    #[must_use]
    pub fn detect(&self, incoming: &EditDescriptor) -> Vec<&EditDescriptor> {
        let track_edits = match self.applied.get(&incoming.region.track_id) {
            Some(edits) => edits,
            None => return vec![],
        };
        track_edits
            .iter()
            .filter(|existing| existing.region.overlaps(&incoming.region))
            .collect()
    }

    /// Number of applied edits across all tracks.
    #[must_use]
    pub fn applied_count(&self) -> usize {
        self.applied.values().map(Vec::len).sum()
    }

    /// Tracks that have at least one applied edit.
    #[must_use]
    pub fn active_tracks(&self) -> Vec<&str> {
        self.applied.keys().map(String::as_str).collect()
    }

    /// Remove an applied edit by ID.  Returns `true` if found.
    pub fn remove(&mut self, id: &EditId) -> bool {
        for edits in self.applied.values_mut() {
            if let Some(pos) = edits.iter().position(|e| &e.id == id) {
                edits.remove(pos);
                return true;
            }
        }
        false
    }

    /// Clear all applied edits for a given track.
    pub fn clear_track(&mut self, track_id: &str) {
        self.applied.remove(track_id);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers for policy application
// ─────────────────────────────────────────────────────────────────────────────

/// Per-user role priority for [`ConflictPolicy::RolePriority`].
/// Higher value = higher priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum UserPriority {
    Viewer = 0,
    Editor = 1,
    Owner = 2,
}

/// Assess conflict severity from two edit descriptors.
fn assess_severity(existing: &EditDescriptor, incoming: &EditDescriptor) -> ConflictSeverity {
    // Identical author (same user editing twice quickly) — treat as low.
    if existing.author == incoming.author {
        return ConflictSeverity::Low;
    }
    // Regions that are fully overlapping and same label = high structural conflict.
    if existing.region == incoming.region && existing.label == incoming.label {
        return ConflictSeverity::High;
    }
    ConflictSeverity::Medium
}

/// Choose the resolution outcome for two edits under a given policy.
fn choose_outcome(
    existing: &EditDescriptor,
    incoming: &EditDescriptor,
    policy: ConflictPolicy,
    existing_priority: UserPriority,
    incoming_priority: UserPriority,
) -> (ResolutionOutcome, String) {
    match policy {
        ConflictPolicy::LastWriteWins => {
            let (winner, loser, outcome) = if incoming.clock > existing.clock {
                ("incoming", "existing", ResolutionOutcome::IncomingWins)
            } else if existing.clock > incoming.clock {
                ("existing", "incoming", ResolutionOutcome::ExistingWins)
            } else {
                // Tie-break by wall clock
                if incoming.wall_ms >= existing.wall_ms {
                    ("incoming", "existing", ResolutionOutcome::IncomingWins)
                } else {
                    ("existing", "incoming", ResolutionOutcome::ExistingWins)
                }
            };
            (
                outcome,
                format!(
                    "LastWriteWins: {} edit (clock {}) supersedes {} edit (clock {}).",
                    winner,
                    if winner == "incoming" { incoming.clock } else { existing.clock },
                    loser,
                    if loser == "existing" { existing.clock } else { incoming.clock },
                ),
            )
        }

        ConflictPolicy::FirstWriteWins => {
            let (winner, loser, outcome) = if existing.clock <= incoming.clock {
                ("existing", "incoming", ResolutionOutcome::ExistingWins)
            } else {
                ("incoming", "existing", ResolutionOutcome::IncomingWins)
            };
            (
                outcome,
                format!(
                    "FirstWriteWins: {} edit (clock {}) takes precedence over {} edit (clock {}).",
                    winner,
                    if winner == "existing" { existing.clock } else { incoming.clock },
                    loser,
                    if loser == "incoming" { incoming.clock } else { existing.clock },
                ),
            )
        }

        ConflictPolicy::ManualReview => (
            ResolutionOutcome::PendingReview,
            "ManualReview: a reviewer must choose between the two conflicting edits.".to_string(),
        ),

        ConflictPolicy::MergeStrategy => {
            // Attempt a merge: if regions differ only in boundary the merge is
            // likely safe; otherwise escalate to manual review.
            let start_overlap = existing.region.start_ms.max(incoming.region.start_ms);
            let end_overlap = existing.region.end_ms.min(incoming.region.end_ms);
            let overlap_ms = (end_overlap - start_overlap).max(0);
            let existing_dur = existing.region.duration_ms().max(1);
            let overlap_ratio = overlap_ms as f64 / existing_dur as f64;

            if overlap_ratio < 0.5 {
                // Low overlap – edits are mostly independent, merge is safe.
                (
                    ResolutionOutcome::Merged,
                    format!(
                        "MergeStrategy: low overlap ({:.0}%) — edits merged automatically.",
                        overlap_ratio * 100.0
                    ),
                )
            } else {
                // High overlap – too risky to auto-merge.
                (
                    ResolutionOutcome::PendingReview,
                    format!(
                        "MergeStrategy: high overlap ({:.0}%) — escalated to manual review.",
                        overlap_ratio * 100.0
                    ),
                )
            }
        }

        ConflictPolicy::RolePriority => {
            if existing_priority > incoming_priority {
                (
                    ResolutionOutcome::ExistingWins,
                    format!(
                        "RolePriority: existing edit (priority {:?}) wins over incoming (priority {:?}).",
                        existing_priority, incoming_priority
                    ),
                )
            } else if incoming_priority > existing_priority {
                (
                    ResolutionOutcome::IncomingWins,
                    format!(
                        "RolePriority: incoming edit (priority {:?}) overrides existing (priority {:?}).",
                        incoming_priority, existing_priority
                    ),
                )
            } else {
                // Equal priority — fall back to LastWriteWins
                let (outcome, rec) = choose_outcome(
                    existing,
                    incoming,
                    ConflictPolicy::LastWriteWins,
                    existing_priority,
                    incoming_priority,
                );
                (outcome, format!("RolePriority tie-break via LastWriteWins — {}", rec))
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PolicyEngine
// ─────────────────────────────────────────────────────────────────────────────

/// Applies the active [`ConflictPolicy`] to incoming edits and produces
/// [`ConflictReport`]s.
///
/// The engine maintains its own [`ConflictDetector`] and can also be told
/// about user priorities (for [`ConflictPolicy::RolePriority`]).
#[derive(Debug)]
pub struct PolicyEngine {
    /// Active policy.
    pub policy: ConflictPolicy,
    /// Underlying detector.
    detector: ConflictDetector,
    /// User priority table (defaults to `Editor` if absent).
    priorities: HashMap<Uuid, UserPriority>,
    /// All generated reports in chronological order.
    pub reports: Vec<ConflictReport>,
    /// Monotonic report counter for debugging.
    report_count: u64,
}

impl PolicyEngine {
    /// Create a new policy engine with the given policy.
    pub fn new(policy: ConflictPolicy) -> Self {
        Self {
            policy,
            detector: ConflictDetector::new(),
            priorities: HashMap::new(),
            reports: Vec::new(),
            report_count: 0,
        }
    }

    /// Register a user's priority (used by [`ConflictPolicy::RolePriority`]).
    pub fn set_priority(&mut self, user_id: Uuid, priority: UserPriority) {
        self.priorities.insert(user_id, priority);
    }

    /// Look up a user's priority, defaulting to `Editor`.
    fn priority_of(&self, user_id: &Uuid) -> UserPriority {
        self.priorities
            .get(user_id)
            .copied()
            .unwrap_or(UserPriority::Editor)
    }

    /// Submit an incoming edit.
    ///
    /// If no conflicts are detected the edit is registered as applied and
    /// `None` is returned.  If one or more conflicts are found, a
    /// [`ConflictReport`] is generated for the first conflict and returned.
    /// The edit is applied to the detector only when the outcome is
    /// `IncomingWins` or `Merged`.
    pub fn submit(
        &mut self,
        incoming: EditDescriptor,
        now_ms: u64,
    ) -> Option<ConflictReport> {
        let conflicts = self.detector.detect(&incoming);

        if conflicts.is_empty() {
            self.detector.register(incoming);
            return None;
        }

        // Use the first conflicting edit to generate a report.
        let existing = conflicts[0].clone();
        let severity = assess_severity(&existing, &incoming);
        let existing_prio = self.priority_of(&existing.author);
        let incoming_prio = self.priority_of(&incoming.author);

        let (outcome, recommendation) = choose_outcome(
            &existing,
            &incoming,
            self.policy,
            existing_prio,
            incoming_prio,
        );

        // Apply the incoming edit if the policy says it wins or merges.
        match &outcome {
            ResolutionOutcome::IncomingWins | ResolutionOutcome::Merged => {
                // Under IncomingWins, remove the superseded existing edit.
                if outcome == ResolutionOutcome::IncomingWins {
                    self.detector.remove(&existing.id);
                }
                self.detector.register(incoming.clone());
            }
            ResolutionOutcome::ExistingWins | ResolutionOutcome::PendingReview => {
                // Either existing is kept as-is or we wait for a reviewer.
            }
        }

        self.report_count += 1;
        let report = ConflictReport {
            id: Uuid::new_v4(),
            existing,
            incoming,
            policy: self.policy,
            outcome,
            severity,
            recommendation,
            generated_at_ms: now_ms,
        };
        self.reports.push(report.clone());
        Some(report)
    }

    /// Change the active policy.
    pub fn set_policy(&mut self, policy: ConflictPolicy) {
        self.policy = policy;
    }

    /// All reports that still need manual review.
    #[must_use]
    pub fn pending_reviews(&self) -> Vec<&ConflictReport> {
        self.reports
            .iter()
            .filter(|r| r.needs_review())
            .collect()
    }

    /// Total number of conflicts detected so far.
    #[must_use]
    pub fn total_conflicts(&self) -> u64 {
        self.report_count
    }

    /// Number of applied edits across all tracks.
    #[must_use]
    pub fn applied_count(&self) -> usize {
        self.detector.applied_count()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn region(track: &str, start: i64, end: i64) -> Region {
        Region::new(track, start, end)
    }

    fn edit(author: Uuid, clock: u64, wall_ms: u64, r: Region, label: &str) -> EditDescriptor {
        EditDescriptor::new(author, clock, wall_ms, r, label)
    }

    fn user() -> Uuid {
        Uuid::new_v4()
    }

    // ── Region ───────────────────────────────────────────────────────────────

    #[test]
    fn test_region_overlap_detected() {
        let a = region("t1", 0, 1000);
        let b = region("t1", 500, 1500);
        assert!(a.overlaps(&b));
    }

    #[test]
    fn test_region_no_overlap_adjacent() {
        let a = region("t1", 0, 1000);
        let b = region("t1", 1000, 2000);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_region_different_track_no_overlap() {
        let a = region("t1", 0, 1000);
        let b = region("t2", 0, 1000);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_region_duration() {
        let r = region("t1", 200, 800);
        assert_eq!(r.duration_ms(), 600);
    }

    // ── ConflictPolicy metadata ───────────────────────────────────────────────

    #[test]
    fn test_policy_names() {
        assert_eq!(ConflictPolicy::LastWriteWins.name(), "LastWriteWins");
        assert_eq!(ConflictPolicy::ManualReview.name(), "ManualReview");
        assert_eq!(ConflictPolicy::MergeStrategy.name(), "MergeStrategy");
    }

    #[test]
    fn test_policy_determinism_flags() {
        assert!(ConflictPolicy::LastWriteWins.is_deterministic());
        assert!(ConflictPolicy::FirstWriteWins.is_deterministic());
        assert!(!ConflictPolicy::ManualReview.is_deterministic());
        assert!(!ConflictPolicy::MergeStrategy.is_deterministic());
    }

    #[test]
    fn test_policy_manual_review_flag() {
        assert!(ConflictPolicy::ManualReview.may_require_manual_review());
        assert!(ConflictPolicy::MergeStrategy.may_require_manual_review());
        assert!(!ConflictPolicy::LastWriteWins.may_require_manual_review());
    }

    // ── ConflictDetector ─────────────────────────────────────────────────────

    #[test]
    fn test_detector_no_conflict_different_tracks() {
        let mut det = ConflictDetector::new();
        let u = user();
        det.register(edit(u, 1, 1000, region("t1", 0, 500), "Trim"));
        let incoming = edit(u, 2, 2000, region("t2", 0, 500), "Gain");
        assert!(det.detect(&incoming).is_empty());
    }

    #[test]
    fn test_detector_conflict_same_track() {
        let mut det = ConflictDetector::new();
        let u = user();
        det.register(edit(u, 1, 1000, region("t1", 0, 1000), "Trim"));
        let incoming = edit(u, 2, 2000, region("t1", 500, 1500), "Gain");
        assert_eq!(det.detect(&incoming).len(), 1);
    }

    #[test]
    fn test_detector_remove_edit() {
        let mut det = ConflictDetector::new();
        let u = user();
        let e = edit(u, 1, 1000, region("t1", 0, 1000), "Cut");
        let id = e.id;
        det.register(e);
        assert_eq!(det.applied_count(), 1);
        assert!(det.remove(&id));
        assert_eq!(det.applied_count(), 0);
    }

    // ── PolicyEngine — LastWriteWins ──────────────────────────────────────────

    #[test]
    fn test_lww_incoming_wins_higher_clock() {
        let mut eng = PolicyEngine::new(ConflictPolicy::LastWriteWins);
        let u1 = user();
        let u2 = user();

        let e1 = edit(u1, 1, 1000, region("t1", 0, 1000), "Trim");
        eng.submit(e1, 1000);

        let e2 = edit(u2, 5, 5000, region("t1", 0, 1000), "Trim");
        let report = eng.submit(e2, 5000).expect("conflict expected");
        assert_eq!(report.outcome, ResolutionOutcome::IncomingWins);
    }

    #[test]
    fn test_lww_existing_wins_higher_clock() {
        let mut eng = PolicyEngine::new(ConflictPolicy::LastWriteWins);
        let u1 = user();
        let u2 = user();

        let e1 = edit(u1, 10, 1000, region("t1", 0, 1000), "Trim");
        eng.submit(e1, 1000);

        let e2 = edit(u2, 2, 2000, region("t1", 0, 1000), "Trim");
        let report = eng.submit(e2, 2000).expect("conflict expected");
        assert_eq!(report.outcome, ResolutionOutcome::ExistingWins);
    }

    // ── PolicyEngine — FirstWriteWins ─────────────────────────────────────────

    #[test]
    fn test_fww_existing_wins() {
        let mut eng = PolicyEngine::new(ConflictPolicy::FirstWriteWins);
        let u1 = user();
        let u2 = user();

        eng.submit(edit(u1, 1, 1000, region("t1", 0, 1000), "Insert"), 1000);
        let report = eng
            .submit(edit(u2, 5, 5000, region("t1", 200, 800), "Delete"), 5000)
            .expect("conflict expected");
        assert_eq!(report.outcome, ResolutionOutcome::ExistingWins);
    }

    // ── PolicyEngine — ManualReview ───────────────────────────────────────────

    #[test]
    fn test_manual_review_produces_pending() {
        let mut eng = PolicyEngine::new(ConflictPolicy::ManualReview);
        let u1 = user();
        let u2 = user();

        eng.submit(edit(u1, 1, 1000, region("t1", 0, 1000), "Gain"), 1000);
        let report = eng
            .submit(edit(u2, 2, 2000, region("t1", 0, 1000), "Gain"), 2000)
            .expect("conflict expected");
        assert_eq!(report.outcome, ResolutionOutcome::PendingReview);
        assert!(report.needs_review());
        assert_eq!(eng.pending_reviews().len(), 1);
    }

    // ── PolicyEngine — MergeStrategy ──────────────────────────────────────────

    #[test]
    fn test_merge_strategy_low_overlap_auto_merges() {
        let mut eng = PolicyEngine::new(ConflictPolicy::MergeStrategy);
        let u1 = user();
        let u2 = user();

        // existing: 0–1000 ms, incoming: 900–1500 ms → overlap = 100 / 1000 = 10 %
        eng.submit(edit(u1, 1, 1000, region("t1", 0, 1000), "Trim"), 1000);
        let report = eng
            .submit(edit(u2, 2, 2000, region("t1", 900, 1500), "Gain"), 2000)
            .expect("conflict expected");
        assert_eq!(report.outcome, ResolutionOutcome::Merged);
    }

    #[test]
    fn test_merge_strategy_high_overlap_escalates() {
        let mut eng = PolicyEngine::new(ConflictPolicy::MergeStrategy);
        let u1 = user();
        let u2 = user();

        // existing: 0–1000, incoming: 0–900 → overlap = 900/1000 = 90% → escalate
        eng.submit(edit(u1, 1, 1000, region("t1", 0, 1000), "Cut"), 1000);
        let report = eng
            .submit(edit(u2, 2, 2000, region("t1", 0, 900), "Cut"), 2000)
            .expect("conflict expected");
        assert_eq!(report.outcome, ResolutionOutcome::PendingReview);
    }

    // ── PolicyEngine — RolePriority ───────────────────────────────────────────

    #[test]
    fn test_role_priority_owner_beats_editor() {
        let mut eng = PolicyEngine::new(ConflictPolicy::RolePriority);
        let owner = user();
        let editor = user();
        eng.set_priority(owner, UserPriority::Owner);
        eng.set_priority(editor, UserPriority::Editor);

        // editor submits first (lower clock), owner submits second (higher clock)
        eng.submit(edit(editor, 1, 1000, region("t1", 0, 1000), "Trim"), 1000);
        let report = eng
            .submit(edit(owner, 5, 5000, region("t1", 0, 1000), "Trim"), 5000)
            .expect("conflict expected");
        assert_eq!(report.outcome, ResolutionOutcome::IncomingWins);
    }

    #[test]
    fn test_role_priority_equal_falls_back_to_lww() {
        let mut eng = PolicyEngine::new(ConflictPolicy::RolePriority);
        let u1 = user();
        let u2 = user();
        eng.set_priority(u1, UserPriority::Editor);
        eng.set_priority(u2, UserPriority::Editor);

        eng.submit(edit(u1, 1, 1000, region("t1", 0, 1000), "Gain"), 1000);
        let report = eng
            .submit(edit(u2, 5, 5000, region("t1", 0, 1000), "Gain"), 5000)
            .expect("conflict expected");
        // Equal priority → LWW → incoming (clock 5) wins
        assert_eq!(report.outcome, ResolutionOutcome::IncomingWins);
    }

    // ── ConflictReport formatting ─────────────────────────────────────────────

    #[test]
    fn test_report_summary_contains_key_fields() {
        let mut eng = PolicyEngine::new(ConflictPolicy::LastWriteWins);
        let u1 = user();
        let u2 = user();

        eng.submit(edit(u1, 1, 1000, region("audio", 0, 2000), "Gain"), 1000);
        let report = eng
            .submit(edit(u2, 5, 5000, region("audio", 0, 2000), "Gain"), 5000)
            .expect("conflict expected");

        let summary = report.summary();
        assert!(summary.contains("LastWriteWins"), "should contain policy name");
        assert!(summary.contains("audio"), "should contain track id");
        assert!(summary.contains("IncomingWins"), "should contain outcome");
    }

    // ── Severity assessment ───────────────────────────────────────────────────

    #[test]
    fn test_severity_same_author_is_low() {
        let u = user();
        let e = edit(u, 1, 1000, region("t1", 0, 1000), "Trim");
        let i = edit(u, 2, 2000, region("t1", 0, 1000), "Trim");
        assert_eq!(assess_severity(&e, &i), ConflictSeverity::Low);
    }

    #[test]
    fn test_severity_different_author_same_region_label_is_high() {
        let u1 = user();
        let u2 = user();
        let r = region("t1", 0, 1000);
        let e = edit(u1, 1, 1000, r.clone(), "Cut");
        let i = edit(u2, 2, 2000, r, "Cut");
        assert_eq!(assess_severity(&e, &i), ConflictSeverity::High);
    }
}
